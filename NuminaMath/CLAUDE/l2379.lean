import Mathlib

namespace NUMINAMATH_CALUDE_max_non_managers_l2379_237983

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l2379_237983


namespace NUMINAMATH_CALUDE_floor_abs_sum_l2379_237940

theorem floor_abs_sum (x : ℝ) (h : x = -5.7) : 
  ⌊|x|⌋ + |⌊x⌋| = 11 := by
sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l2379_237940


namespace NUMINAMATH_CALUDE_UA_intersect_B_equals_two_three_l2379_237966

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3, 4}

def A : Set Int := {x ∈ U | x * (x^2 - 1) = 0}

def B : Set Int := {x ∈ U | x ≥ 0 ∧ x^2 ≤ 9}

theorem UA_intersect_B_equals_two_three : (U \ A) ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_UA_intersect_B_equals_two_three_l2379_237966


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l2379_237963

theorem sqrt_expression_equals_two : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l2379_237963


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2379_237928

/-- Represents the number of students in each grade -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (population : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPopulation := population.freshmen + population.sophomores + population.juniors
  let ratio := totalSample / totalPopulation
  { freshmen := population.freshmen * ratio,
    sophomores := population.sophomores * ratio,
    juniors := population.juniors * ratio }

theorem correct_stratified_sample :
  let population := GradePopulation.mk 560 540 520
  let sample := stratifiedSample population 81
  sample = SampleSize.mk 28 27 26 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2379_237928


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2379_237976

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (x - a)^3 / ((a - b) * (a - c)) + (x - b)^3 / ((b - a) * (b - c)) + (x - c)^3 / ((c - a) * (c - b)) =
  a + b + c - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2379_237976


namespace NUMINAMATH_CALUDE_d_t_eventually_two_exists_n_d_t_two_from_m_l2379_237934

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The t-th iteration of d applied to n -/
def d_t (t n : ℕ) : ℕ :=
  match t with
  | 0 => n
  | t + 1 => d (d_t t n)

/-- For any n > 1, the sequence d_t(n) eventually becomes 2 -/
theorem d_t_eventually_two (n : ℕ) (h : n > 1) :
  ∃ k, ∀ t, t ≥ k → d_t t n = 2 := by sorry

/-- For any m, there exists an n such that d_t(n) becomes 2 from the m-th term onwards -/
theorem exists_n_d_t_two_from_m (m : ℕ) :
  ∃ n, ∀ t, t ≥ m → d_t t n = 2 := by sorry

end NUMINAMATH_CALUDE_d_t_eventually_two_exists_n_d_t_two_from_m_l2379_237934


namespace NUMINAMATH_CALUDE_extended_triangle_pc_length_l2379_237915

/-- Triangle ABC with sides AB, BC, CA, extended to point P -/
structure ExtendedTriangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  PC : ℝ

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : ExtendedTriangle) : Prop :=
  t.PC / (t.PC + t.BC) = t.CA / t.AB

theorem extended_triangle_pc_length 
  (t : ExtendedTriangle) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t) : 
  t.PC = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_pc_length_l2379_237915


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2379_237935

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2379_237935


namespace NUMINAMATH_CALUDE_flu_infection_equation_l2379_237999

/-- 
Given:
- One person initially has the flu
- Each person infects x people on average in each round
- There are two rounds of infection
- After two rounds, 144 people have the flu

Prove that (1 + x)^2 = 144 correctly represents the total number of infected people.
-/
theorem flu_infection_equation (x : ℝ) : (1 + x)^2 = 144 :=
sorry

end NUMINAMATH_CALUDE_flu_infection_equation_l2379_237999


namespace NUMINAMATH_CALUDE_simplify_sqrt_x6_plus_x3_l2379_237927

theorem simplify_sqrt_x6_plus_x3 (x : ℝ) : 
  Real.sqrt (x^6 + x^3) = |x| * Real.sqrt |x| * Real.sqrt (x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x6_plus_x3_l2379_237927


namespace NUMINAMATH_CALUDE_square_diagonal_triangle_l2379_237992

theorem square_diagonal_triangle (s : ℝ) (h : s = 10) :
  let diagonal := s * Real.sqrt 2
  diagonal = 10 * Real.sqrt 2 ∧ s = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_triangle_l2379_237992


namespace NUMINAMATH_CALUDE_largest_three_digit_special_divisibility_l2379_237921

theorem largest_three_digit_special_divisibility : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n % 11 = 0) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) → 
    (∀ d : ℕ, d ∈ m.digits 10 → d ≠ 0 → m % d = 0) →
    (m % 11 = 0) → m ≤ n) ∧
  n = 924 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_divisibility_l2379_237921


namespace NUMINAMATH_CALUDE_jack_lifetime_l2379_237980

theorem jack_lifetime :
  ∀ (L : ℝ),
  (L = (1/6)*L + (1/12)*L + (1/7)*L + 5 + (1/2)*L + 4) →
  L = 84 := by
sorry

end NUMINAMATH_CALUDE_jack_lifetime_l2379_237980


namespace NUMINAMATH_CALUDE_tangent_curves_l2379_237930

theorem tangent_curves (m : ℝ) : 
  (∃ x y : ℝ, y = x^3 + 2 ∧ y^2 - m*x = 1 ∧ 
   ∀ x' : ℝ, x' ≠ x → (x'^3 + 2)^2 - m*x' ≠ 1) ↔ 
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_curves_l2379_237930


namespace NUMINAMATH_CALUDE_georgia_carnation_problem_l2379_237941

/-- The number of teachers Georgia sent a dozen carnations to -/
def num_teachers : ℕ := 4

/-- The cost of a single carnation in cents -/
def single_carnation_cost : ℕ := 50

/-- The cost of a dozen carnations in cents -/
def dozen_carnation_cost : ℕ := 400

/-- The number of Georgia's friends -/
def num_friends : ℕ := 14

/-- The total amount Georgia spent in cents -/
def total_spent : ℕ := 2500

theorem georgia_carnation_problem :
  num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost ≤ total_spent ∧
  (num_teachers + 1) * dozen_carnation_cost + num_friends * single_carnation_cost > total_spent :=
by sorry

end NUMINAMATH_CALUDE_georgia_carnation_problem_l2379_237941


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2379_237962

/-- The function f(x) = x³ - 2x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 2)

/-- The slope of the tangent line at P -/
def m : ℝ := f' P.1

/-- Theorem: The equation of the tangent line to y = f(x) at P(1, 2) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - P.1) + P.2 ↔ x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2379_237962


namespace NUMINAMATH_CALUDE_felix_trees_chopped_l2379_237933

/-- Calculates the minimum number of trees chopped given the total spent on sharpening,
    cost per sharpening, and trees chopped before resharpening is needed. -/
def min_trees_chopped (total_spent : ℕ) (cost_per_sharpening : ℕ) (trees_per_sharpening : ℕ) : ℕ :=
  (total_spent / cost_per_sharpening) * trees_per_sharpening

/-- Proves that Felix has chopped down at least 150 trees given the problem conditions. -/
theorem felix_trees_chopped :
  let total_spent : ℕ := 48
  let cost_per_sharpening : ℕ := 8
  let trees_per_sharpening : ℕ := 25
  min_trees_chopped total_spent cost_per_sharpening trees_per_sharpening = 150 := by
  sorry

#eval min_trees_chopped 48 8 25  -- Should output 150

end NUMINAMATH_CALUDE_felix_trees_chopped_l2379_237933


namespace NUMINAMATH_CALUDE_clothing_problem_l2379_237978

/-- Calculates the remaining clothing pieces after donations and discarding --/
def remaining_clothing (initial : ℕ) (donated1 : ℕ) (donated2_multiplier : ℕ) (discarded : ℕ) : ℕ :=
  initial - (donated1 + donated1 * donated2_multiplier) - discarded

/-- Theorem stating that given the specific values in the problem, 
    the remaining clothing pieces is 65 --/
theorem clothing_problem : 
  remaining_clothing 100 5 3 15 = 65 := by
  sorry

end NUMINAMATH_CALUDE_clothing_problem_l2379_237978


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l2379_237967

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y : ℝ, y > 0 ∧ y < 1/6 → (y - 3) / (12 * y^2 - 50 * y + 12) ≠ 0) ∧ 
  ((1/6 : ℝ) - 3) / (12 * (1/6)^2 - 50 * (1/6) + 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l2379_237967


namespace NUMINAMATH_CALUDE_floor_area_less_than_10_l2379_237985

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that each wall requires more paint than the floor -/
def more_paint_on_walls (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧
  r.width * r.height > r.length * r.width

/-- The floor area of the room -/
def floor_area (r : Room) : ℝ :=
  r.length * r.width

/-- Theorem stating that for a room with height 3 meters and more paint required for walls than floor,
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_10 (r : Room) 
  (h1 : r.height = 3)
  (h2 : more_paint_on_walls r) : 
  floor_area r < 10 := by
  sorry


end NUMINAMATH_CALUDE_floor_area_less_than_10_l2379_237985


namespace NUMINAMATH_CALUDE_tangency_quad_area_theorem_l2379_237939

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The area of the trapezoid -/
  trapezoidArea : ℝ
  /-- The area of the quadrilateral formed by tangency points -/
  tangencyQuadArea : ℝ
  /-- Assumption that the trapezoid is circumscribed around the circle -/
  isCircumscribed : Prop
  /-- Assumption that the trapezoid is isosceles -/
  isIsosceles : Prop

/-- Theorem stating the relationship between the areas -/
theorem tangency_quad_area_theorem (t : CircumscribedTrapezoid)
  (h1 : t.radius = 1)
  (h2 : t.trapezoidArea = 5)
  : t.tangencyQuadArea = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_tangency_quad_area_theorem_l2379_237939


namespace NUMINAMATH_CALUDE_relationship_xyz_l2379_237926

noncomputable def x : ℝ := Real.sqrt 2
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 0.7 / Real.log 5

theorem relationship_xyz : z < y ∧ y < x := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l2379_237926


namespace NUMINAMATH_CALUDE_movie_theater_open_hours_l2379_237975

/-- A movie theater with multiple screens showing movies throughout the day. -/
structure MovieTheater where
  screens : ℕ
  total_movies : ℕ
  movie_duration : ℕ

/-- Calculate the number of hours a movie theater is open. -/
def theater_open_hours (theater : MovieTheater) : ℕ :=
  (theater.total_movies * theater.movie_duration) / theater.screens

/-- Theorem: A movie theater with 6 screens showing 24 movies, each lasting 2 hours, is open for 8 hours. -/
theorem movie_theater_open_hours :
  let theater := MovieTheater.mk 6 24 2
  theater_open_hours theater = 8 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_open_hours_l2379_237975


namespace NUMINAMATH_CALUDE_function_equation_solution_l2379_237937

theorem function_equation_solution (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = 2^x + (1/2)^x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2379_237937


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l2379_237945

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define a circle in Cartesian coordinates
def is_circle (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_equation_is_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    is_circle x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l2379_237945


namespace NUMINAMATH_CALUDE_johns_score_less_than_winning_score_l2379_237918

/-- In a blackjack game, given the scores of three players and the winning score,
    prove that the score of the player who didn't win is less than the winning score. -/
theorem johns_score_less_than_winning_score 
  (theodore_score : ℕ) 
  (zoey_score : ℕ) 
  (john_score : ℕ) 
  (winning_score : ℕ) 
  (h1 : theodore_score = 13)
  (h2 : zoey_score = 19)
  (h3 : winning_score = 19)
  (h4 : zoey_score = winning_score)
  (h5 : john_score ≠ zoey_score) : 
  john_score < winning_score :=
sorry

end NUMINAMATH_CALUDE_johns_score_less_than_winning_score_l2379_237918


namespace NUMINAMATH_CALUDE_expression_equality_l2379_237948

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Prove that 27^(1/3) + lg 4 + 2 * lg 5 - e^(ln 3) = 2 -/
theorem expression_equality : 27^(1/3) + lg 4 + 2 * lg 5 - Real.exp (Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2379_237948


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2379_237965

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x

-- State the theorem
theorem quadratic_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f m x = 3) →
  m = -4 ∨ m = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2379_237965


namespace NUMINAMATH_CALUDE_ten_men_absent_l2379_237947

/-- Represents the work scenario with men and days -/
structure WorkScenario where
  totalMen : ℕ
  originalDays : ℕ
  actualDays : ℕ

/-- Calculates the number of absent men given a work scenario -/
def absentMen (w : WorkScenario) : ℕ :=
  w.totalMen - (w.totalMen * w.originalDays) / w.actualDays

/-- The theorem stating that 10 men became absent in the given scenario -/
theorem ten_men_absent : absentMen ⟨60, 50, 60⟩ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_men_absent_l2379_237947


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_l2379_237913

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^9 + b^9 = 76 -/
theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^9 + b^9 = 76 := by
  sorry

#check sum_of_ninth_powers

end NUMINAMATH_CALUDE_sum_of_ninth_powers_l2379_237913


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2379_237969

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l2379_237969


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2379_237956

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_z : z = (x + y) / 2) :
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 3 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 3 ∧ z = (x + y) / 2 ∧
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2379_237956


namespace NUMINAMATH_CALUDE_morks_tax_rate_l2379_237924

theorem morks_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) : 
  mork_tax_rate > 0 →
  mork_income > 0 →
  let mindy_income := 3 * mork_income
  let mindy_tax_rate := 0.3
  let total_income := mork_income + mindy_income
  let total_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_tax_rate := total_tax / total_income
  combined_tax_rate = 0.325 →
  mork_tax_rate = 0.4 := by
sorry

end NUMINAMATH_CALUDE_morks_tax_rate_l2379_237924


namespace NUMINAMATH_CALUDE_barbara_age_when_mike_24_l2379_237911

/-- Given that Mike is 16 years old and Barbara is half his age, 
    prove that Barbara will be 16 years old when Mike is 24. -/
theorem barbara_age_when_mike_24 (mike_current_age barbara_current_age mike_future_age : ℕ) : 
  mike_current_age = 16 →
  barbara_current_age = mike_current_age / 2 →
  mike_future_age = 24 →
  barbara_current_age + (mike_future_age - mike_current_age) = 16 :=
by sorry

end NUMINAMATH_CALUDE_barbara_age_when_mike_24_l2379_237911


namespace NUMINAMATH_CALUDE_distance_EC_l2379_237904

/-- Given five points A, B, C, D, E on a line, with known distances between consecutive points,
    prove that the distance between E and C is 150. -/
theorem distance_EC (A B C D E : ℝ) 
  (h_AB : |A - B| = 30)
  (h_BC : |B - C| = 80)
  (h_CD : |C - D| = 236)
  (h_DE : |D - E| = 86)
  (h_EA : |E - A| = 40)
  (h_line : ∃ (t : ℝ → ℝ), t A < t B ∧ t B < t C ∧ t C < t D ∧ t D < t E) :
  |E - C| = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_EC_l2379_237904


namespace NUMINAMATH_CALUDE_diane_allison_age_ratio_l2379_237981

/-- Proves that the ratio of Diane's age to Allison's age when Diane turns 30 is 2:1 -/
theorem diane_allison_age_ratio :
  -- Diane's current age
  ∀ (diane_current_age : ℕ),
  -- Sum of Alex's and Allison's current ages
  ∀ (alex_allison_sum : ℕ),
  -- Diane's age when she turns 30
  ∀ (diane_future_age : ℕ),
  -- Alex's age when Diane turns 30
  ∀ (alex_future_age : ℕ),
  -- Allison's age when Diane turns 30
  ∀ (allison_future_age : ℕ),
  -- Conditions
  diane_current_age = 16 →
  alex_allison_sum = 47 →
  diane_future_age = 30 →
  alex_future_age = 2 * diane_future_age →
  -- Conclusion
  (diane_future_age : ℚ) / (allison_future_age : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diane_allison_age_ratio_l2379_237981


namespace NUMINAMATH_CALUDE_carwash_problem_l2379_237909

/-- Represents the carwash problem with modified constraints to ensure consistency --/
theorem carwash_problem 
  (car_price SUV_price truck_price motorcycle_price bus_price : ℕ)
  (total_raised : ℕ)
  (num_SUVs num_trucks num_motorcycles : ℕ)
  (max_vehicles : ℕ)
  (h1 : car_price = 7)
  (h2 : SUV_price = 12)
  (h3 : truck_price = 10)
  (h4 : motorcycle_price = 15)
  (h5 : bus_price = 18)
  (h6 : total_raised = 500)
  (h7 : num_SUVs = 3)
  (h8 : num_trucks = 8)
  (h9 : num_motorcycles = 5)
  (h10 : max_vehicles = 20)  -- Modified to make the problem consistent
  : ∃ (num_cars num_buses : ℕ), 
    (num_cars + num_buses + num_SUVs + num_trucks + num_motorcycles ≤ max_vehicles) ∧ 
    (num_cars % 2 = 0) ∧ 
    (num_buses % 2 = 1) ∧
    (car_price * num_cars + bus_price * num_buses + 
     SUV_price * num_SUVs + truck_price * num_trucks + 
     motorcycle_price * num_motorcycles = total_raised) := by
  sorry


end NUMINAMATH_CALUDE_carwash_problem_l2379_237909


namespace NUMINAMATH_CALUDE_occupation_assignment_l2379_237972

-- Define the people and professions
inductive Person : Type
  | A | B | C

inductive Profession : Type
  | Teacher | Journalist | Doctor

-- Define the age relation
def OlderThan (p1 p2 : Person) : Prop := sorry

-- Define the profession assignment
def Occupation (p : Person) (prof : Profession) : Prop := sorry

theorem occupation_assignment :
  -- C is older than the doctor
  (∀ p, Occupation p Profession.Doctor → OlderThan Person.C p) →
  -- A's age is different from the journalist
  (∀ p, Occupation p Profession.Journalist → p ≠ Person.A) →
  -- The journalist is younger than B
  (∀ p, Occupation p Profession.Journalist → OlderThan Person.B p) →
  -- Each person has exactly one profession
  (∀ p, ∃! prof, Occupation p prof) →
  -- Each profession is assigned to exactly one person
  (∀ prof, ∃! p, Occupation p prof) →
  -- The only valid assignment is:
  Occupation Person.A Profession.Doctor ∧
  Occupation Person.B Profession.Teacher ∧
  Occupation Person.C Profession.Journalist :=
by sorry

end NUMINAMATH_CALUDE_occupation_assignment_l2379_237972


namespace NUMINAMATH_CALUDE_unicorn_flower_bloom_l2379_237944

theorem unicorn_flower_bloom :
  let num_unicorns : ℕ := 12
  let journey_length : ℕ := 15000  -- in meters
  let step_length : ℕ := 3  -- in meters
  let flowers_per_step : ℕ := 7
  
  (journey_length / step_length) * num_unicorns * flowers_per_step = 420000 :=
by
  sorry

end NUMINAMATH_CALUDE_unicorn_flower_bloom_l2379_237944


namespace NUMINAMATH_CALUDE_simplify_expression_l2379_237905

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  10 * x^3 * y^2 / (15 * x^2 * y^3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2379_237905


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_5_l2379_237998

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (3/5) * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_equals_5 (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_5_l2379_237998


namespace NUMINAMATH_CALUDE_spending_limit_ratio_l2379_237952

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sally_cards_conditions (cards : SallysCards) : Prop :=
  cards.gold.balance = (1/3) * cards.gold.limit ∧
  cards.platinum.balance = (1/4) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (5/12) * cards.platinum.limit

/-- The theorem stating the ratio of spending limits -/
theorem spending_limit_ratio (cards : SallysCards) 
  (h : sally_cards_conditions cards) : 
  cards.platinum.limit = (1/2) * cards.gold.limit := by
  sorry

#check spending_limit_ratio

end NUMINAMATH_CALUDE_spending_limit_ratio_l2379_237952


namespace NUMINAMATH_CALUDE_package_not_qualified_l2379_237916

/-- The standard net weight of a biscuit in grams -/
def standard_weight : ℝ := 350

/-- The acceptable deviation from the standard weight in grams -/
def acceptable_deviation : ℝ := 5

/-- The weight of the package in question in grams -/
def package_weight : ℝ := 358

/-- A package is qualified if its weight is within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  (standard_weight - acceptable_deviation ≤ weight) ∧
  (weight ≤ standard_weight + acceptable_deviation)

/-- Theorem stating that the package with weight 358 grams is not qualified -/
theorem package_not_qualified : ¬(is_qualified package_weight) := by
  sorry

end NUMINAMATH_CALUDE_package_not_qualified_l2379_237916


namespace NUMINAMATH_CALUDE_product_evaluation_l2379_237961

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2379_237961


namespace NUMINAMATH_CALUDE_number_comparisons_l2379_237912

theorem number_comparisons :
  (31^11 < 17^14) ∧
  (33^75 > 63^60) ∧
  (82^33 > 26^44) ∧
  (29^31 > 80^23) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l2379_237912


namespace NUMINAMATH_CALUDE_pentagon_area_l2379_237964

/-- Given a grid with distance m between adjacent points, prove that for a quadrilateral ABCD with area 23, the area of pentagon EFGHI is 28. -/
theorem pentagon_area (m : ℝ) (area_ABCD : ℝ) : 
  m > 0 → area_ABCD = 23 → ∃ (area_EFGHI : ℝ), area_EFGHI = 28 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l2379_237964


namespace NUMINAMATH_CALUDE_marbles_lost_l2379_237914

theorem marbles_lost (doug : ℕ) (ed_initial ed_final : ℕ) 
  (h1 : ed_initial = doug + 29)
  (h2 : ed_final = doug + 12) :
  ed_initial - ed_final = 17 := by
sorry

end NUMINAMATH_CALUDE_marbles_lost_l2379_237914


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2379_237982

theorem smallest_n_congruence (n : ℕ) : ∃ (m : ℕ), m > 0 ∧ (∀ k : ℕ, 0 < k → k < m → (7^k : ℤ) % 5 ≠ (k^7 : ℤ) % 5) ∧ (7^m : ℤ) % 5 = (m^7 : ℤ) % 5 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2379_237982


namespace NUMINAMATH_CALUDE_waysToChooseIsCorrect_l2379_237920

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people -/
def waysToChoose : ℕ :=
  let totalPeople : ℕ := 10
  let committeeSize : ℕ := 3
  totalPeople * Nat.choose (totalPeople - 1) committeeSize

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people is 840 -/
theorem waysToChooseIsCorrect : waysToChoose = 840 := by
  sorry

end NUMINAMATH_CALUDE_waysToChooseIsCorrect_l2379_237920


namespace NUMINAMATH_CALUDE_binomial_12_10_l2379_237943

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_10_l2379_237943


namespace NUMINAMATH_CALUDE_prob_not_yellow_is_seven_tenths_l2379_237989

/-- Represents the contents of a bag of jelly beans -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of selecting a non-yellow jelly bean -/
def probNotYellow (bag : JellyBeanBag) : ℚ :=
  let total := bag.red + bag.green + bag.yellow + bag.blue
  let notYellow := bag.red + bag.green + bag.blue
  notYellow / total

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_not_yellow_is_seven_tenths :
  probNotYellow { red := 4, green := 7, yellow := 9, blue := 10 } = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_yellow_is_seven_tenths_l2379_237989


namespace NUMINAMATH_CALUDE_not_all_linear_functions_increasing_l2379_237974

/-- A linear function from ℝ to ℝ -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

/-- A function is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: Not all linear functions with non-zero slope are increasing on ℝ -/
theorem not_all_linear_functions_increasing :
  ¬(∀ k b : ℝ, k ≠ 0 → IsIncreasing (LinearFunction k b)) := by sorry

end NUMINAMATH_CALUDE_not_all_linear_functions_increasing_l2379_237974


namespace NUMINAMATH_CALUDE_expression_evaluation_l2379_237991

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2379_237991


namespace NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_9_sum_of_digits_l2379_237958

def is_divisible_by_all_less_than_9 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 9 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_9_sum_of_digits :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_9 N ∧ sum_of_digits N = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_9_sum_of_digits_l2379_237958


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2379_237953

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (2 - |x|) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2379_237953


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2379_237929

/-- Triangle ABC with given properties --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  angle_ABC : Real

/-- Angle bisector AD in triangle ABC --/
structure AngleBisector (T : Triangle) where
  AD : ℝ
  is_bisector : Bool

/-- Area of triangle ADC --/
def area_ADC (T : Triangle) (AB : AngleBisector T) : ℝ := sorry

/-- Main theorem --/
theorem triangle_area_theorem (T : Triangle) (AB : AngleBisector T) :
  T.angle_ABC = 90 ∧ T.AB = 90 ∧ T.BC = 56 ∧ T.AC = 2 * T.BC - 6 ∧ AB.is_bisector = true →
  abs (area_ADC T AB - 1363) < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2379_237929


namespace NUMINAMATH_CALUDE_log2_derivative_l2379_237995

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) := Real.log x

-- Define the logarithm with base 2
noncomputable def log2 (x : ℝ) := ln x / ln 2

-- State the theorem
theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * ln 2) :=
sorry

end NUMINAMATH_CALUDE_log2_derivative_l2379_237995


namespace NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l2379_237977

def total_men : ℕ := 500
def married_men : ℕ := 350
def men_with_tv : ℕ := 375
def men_with_radio : ℕ := 450
def men_with_car : ℕ := 325
def men_with_refrigerator : ℕ := 275
def men_with_ac : ℕ := 300

theorem max_men_with_all_items_and_married (men_with_all_items_and_married : ℕ) :
  men_with_all_items_and_married ≤ men_with_refrigerator :=
by sorry

end NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l2379_237977


namespace NUMINAMATH_CALUDE_isosceles_hyperbola_l2379_237946

/-- 
Given that C ≠ 0 and A and B do not vanish simultaneously,
the equation A x(x^2 - y^2) - (A^2 - B^2) x y = C represents
an isosceles hyperbola with asymptotes A x + B y = 0 and B x - A y = 0
-/
theorem isosceles_hyperbola (A B C : ℝ) (h1 : C ≠ 0) (h2 : ¬(A = 0 ∧ B = 0)) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, A * (x t) * ((x t)^2 - (y t)^2) - (A^2 - B^2) * (x t) * (y t) = C) ∧ 
    (∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ 
      A * (x t1) + B * (y t1) = 0 ∧ 
      B * (x t2) - A * (y t2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_hyperbola_l2379_237946


namespace NUMINAMATH_CALUDE_jake_weight_is_152_l2379_237919

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 212

theorem jake_weight_is_152 :
  (jake_weight - 32 = 2 * sister_weight) →
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 152 := by sorry

end NUMINAMATH_CALUDE_jake_weight_is_152_l2379_237919


namespace NUMINAMATH_CALUDE_son_times_younger_l2379_237993

theorem son_times_younger (father_age son_age : ℕ) (h1 : father_age = 36) (h2 : son_age = 9) (h3 : father_age - son_age = 27) :
  father_age / son_age = 4 := by
sorry

end NUMINAMATH_CALUDE_son_times_younger_l2379_237993


namespace NUMINAMATH_CALUDE_medium_supermarkets_sample_l2379_237932

/-- Represents the number of supermarkets to be sampled in a stratified sampling method. -/
def stratified_sample (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) : ℕ :=
  let total := total_large + total_medium + total_small
  (sample_size * total_medium) / total

/-- Theorem stating that the number of medium-sized supermarkets to be sampled is 20. -/
theorem medium_supermarkets_sample :
  stratified_sample 200 400 1400 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_sample_l2379_237932


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2379_237955

/-- The probability of getting exactly k positive answers out of n questions
    when each question has a probability p of getting a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers when asking 6 questions
    to a Magic 8 Ball, where each question has a 1/2 chance of getting a positive answer. -/
theorem magic_8_ball_probability : binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2379_237955


namespace NUMINAMATH_CALUDE_ellipse_properties_l2379_237987

-- Define the ellipse and its properties
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_def : c = Real.sqrt (a^2 - b^2)
  h_e_def : e = c / a

-- Define points and line
def F₁ (E : Ellipse) : ℝ × ℝ := (-E.c, 0)
def F₂ (E : Ellipse) : ℝ × ℝ := (E.c, 0)

-- Define the properties we want to prove
def perimeter_ABF₂ (E : Ellipse) (A B : ℝ × ℝ) : ℝ := sorry

def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_properties (E : Ellipse) (A B : ℝ × ℝ) (h_A_on_C h_B_on_C : (A.1^2 / E.a^2) + (A.2^2 / E.b^2) = 1) 
  (h_l : ∃ (t : ℝ), A = F₁ E + t • (B - F₁ E)) :
  (perimeter_ABF₂ E A B = 4 * E.a) ∧ 
  (dot_product (A - F₁ E) (A - F₂ E) = 5 * E.c^2 → E.e ≥ Real.sqrt 7 / 7) ∧
  (dot_product (A - F₁ E) (A - F₂ E) = 6 * E.c^2 → E.e ≤ Real.sqrt 7 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2379_237987


namespace NUMINAMATH_CALUDE_min_value_theorem_l2379_237917

theorem min_value_theorem (a b c : ℝ) :
  (∀ x y : ℝ, 3*x + 4*y - 5 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ 3*x + 4*y + 5) →
  2 ≤ a + b - c :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2379_237917


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l2379_237900

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ d : ℝ, d = 6 ∧ ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l2379_237900


namespace NUMINAMATH_CALUDE_power_minus_product_equals_one_l2379_237986

theorem power_minus_product_equals_one : 3^2 - (4 * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_minus_product_equals_one_l2379_237986


namespace NUMINAMATH_CALUDE_average_pop_percentage_l2379_237959

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (popped : ℕ) (total : ℕ) : ℚ :=
  (popped : ℚ) / (total : ℚ) * 100

/-- Theorem: The average percentage of popped kernels across three bags is 82% -/
theorem average_pop_percentage :
  let bag1 := popPercentage 60 75
  let bag2 := popPercentage 42 50
  let bag3 := popPercentage 82 100
  (bag1 + bag2 + bag3) / 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_pop_percentage_l2379_237959


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l2379_237907

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 5}
def solution_set2 : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := |1 - (2*x - 1)/3| ≤ 2
def inequality2 (x : ℝ) : Prop := (2 - x)*(x + 3) < 2 - x

-- Theorem statements
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = solution_set2 := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l2379_237907


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l2379_237950

/-- A circle is defined by its center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- A point is defined by its coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point (x, y) is on a circle if and only if (x-a)^2 + (y-b)^2 = r^2 -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.a)^2 + (p.y - c.b)^2 = c.r^2

/-- Theorem: A circle passes through the origin if and only if a^2 + b^2 = r^2 -/
theorem circle_passes_through_origin (c : Circle) :
  isOnCircle origin c ↔ c.a^2 + c.b^2 = c.r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l2379_237950


namespace NUMINAMATH_CALUDE_same_function_fifth_root_power_l2379_237984

theorem same_function_fifth_root_power (x : ℝ) : x = (x^5)^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_same_function_fifth_root_power_l2379_237984


namespace NUMINAMATH_CALUDE_expansion_properties_l2379_237936

def binomial_sum (n : ℕ) : ℕ := 2^n

def constant_term (n : ℕ) : ℕ := Nat.choose n (n / 2)

theorem expansion_properties :
  ∃ (n : ℕ), 
    binomial_sum n = 64 ∧ 
    constant_term n = 15 := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l2379_237936


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_l2379_237957

theorem min_max_abs_quadratic (p q : ℝ) :
  (∃ (M : ℝ), M ≥ 1/2 ∧ ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |x^2 + p*x + q| ≤ M) → M ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_l2379_237957


namespace NUMINAMATH_CALUDE_floor_x_floor_x_eq_42_l2379_237949

theorem floor_x_floor_x_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_eq_42_l2379_237949


namespace NUMINAMATH_CALUDE_alchemist_safe_combinations_l2379_237923

/-- The number of different herbs available to the alchemist. -/
def num_herbs : ℕ := 4

/-- The number of different gems available to the alchemist. -/
def num_gems : ℕ := 6

/-- The number of unstable combinations of herbs and gems. -/
def num_unstable : ℕ := 3

/-- The total number of possible combinations of herbs and gems. -/
def total_combinations : ℕ := num_herbs * num_gems

/-- The number of safe combinations available for the alchemist's elixir. -/
def safe_combinations : ℕ := total_combinations - num_unstable

theorem alchemist_safe_combinations :
  safe_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_alchemist_safe_combinations_l2379_237923


namespace NUMINAMATH_CALUDE_art_dealer_etchings_sold_l2379_237902

theorem art_dealer_etchings_sold (total_earnings : ℕ) (price_low : ℕ) (price_high : ℕ) (num_low : ℕ) :
  total_earnings = 630 →
  price_low = 35 →
  price_high = 45 →
  num_low = 9 →
  ∃ (num_high : ℕ), num_low * price_low + num_high * price_high = total_earnings ∧ num_low + num_high = 16 :=
by sorry

end NUMINAMATH_CALUDE_art_dealer_etchings_sold_l2379_237902


namespace NUMINAMATH_CALUDE_shopping_trip_expenditure_l2379_237938

theorem shopping_trip_expenditure (total : ℝ) (other_percent : ℝ)
  (h1 : total > 0)
  (h2 : 0 ≤ other_percent ∧ other_percent ≤ 100)
  (h3 : 50 + 10 + other_percent = 100)
  (h4 : 0.04 * 50 + 0.08 * other_percent = 5.2) :
  other_percent = 40 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_expenditure_l2379_237938


namespace NUMINAMATH_CALUDE_find_c_minus_d_l2379_237922

-- Define the functions
def f (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g (x : ℝ) : ℝ := -4 * x + 6
def h (c d : ℝ) (x : ℝ) : ℝ := f c d (g x)

-- State the theorem
theorem find_c_minus_d (c d : ℝ) :
  (∀ x, h c d x = x - 8) →
  (∀ x, h c d (x + 8) = x) →
  c - d = 25/4 := by
sorry

end NUMINAMATH_CALUDE_find_c_minus_d_l2379_237922


namespace NUMINAMATH_CALUDE_reciprocal_counterexample_l2379_237931

theorem reciprocal_counterexample : ∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_counterexample_l2379_237931


namespace NUMINAMATH_CALUDE_unique_valid_multiplication_l2379_237925

def is_valid_multiplication (a b : Nat) : Prop :=
  a % 10 = 5 ∧
  b % 10 = 5 ∧
  (a * (b / 10 % 10)) % 100 = 25 ∧
  (a / 10 % 10) % 2 = 0 ∧
  b / 10 % 10 < 3 ∧
  1000 ≤ a * b ∧ a * b < 10000

theorem unique_valid_multiplication :
  ∀ a b : Nat, is_valid_multiplication a b → (a = 365 ∧ b = 25) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_multiplication_l2379_237925


namespace NUMINAMATH_CALUDE_tom_family_plates_l2379_237903

/-- Calculates the total number of plates used by a family during a stay -/
def total_plates_used (family_size : ℕ) (days : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) : ℕ :=
  family_size * days * meals_per_day * plates_per_meal

/-- Theorem: The total number of plates used by Tom's family during their 4-day stay is 144 -/
theorem tom_family_plates : 
  total_plates_used 6 4 3 2 = 144 := by
  sorry


end NUMINAMATH_CALUDE_tom_family_plates_l2379_237903


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2379_237971

theorem complex_expression_simplification (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (a + b) / ((Real.sqrt a - Real.sqrt b) ^ 2) *
  ((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b ^ 2) /
   (0.5 * Real.sqrt (0.25 * ((a / b) + (b / a)) ^ 2 - 1)) +
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b ^ 2 * Real.sqrt a) /
   (1.5 * Real.sqrt b - 2 * Real.sqrt a)) =
  -2 * b * (a + 3 * Real.sqrt (a * b)) := by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2379_237971


namespace NUMINAMATH_CALUDE_parking_lot_useable_percentage_l2379_237997

/-- Proves that the percentage of a parking lot useable for parking is 80%, given specific conditions. -/
theorem parking_lot_useable_percentage :
  ∀ (length width : ℝ) (area_per_car : ℝ) (num_cars : ℕ),
    length = 400 →
    width = 500 →
    area_per_car = 10 →
    num_cars = 16000 →
    (((num_cars : ℝ) * area_per_car) / (length * width)) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_useable_percentage_l2379_237997


namespace NUMINAMATH_CALUDE_sandwich_availability_l2379_237990

theorem sandwich_availability (total : ℕ) (sold_out : ℕ) (available : ℕ) 
  (h1 : total = 50) 
  (h2 : sold_out = 33) 
  (h3 : available = total - sold_out) : 
  available = 17 := by
sorry

end NUMINAMATH_CALUDE_sandwich_availability_l2379_237990


namespace NUMINAMATH_CALUDE_sin_3alpha_inequality_l2379_237979

theorem sin_3alpha_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 6) :
  2 * Real.sin α < Real.sin (3 * α) ∧ Real.sin (3 * α) < 3 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_sin_3alpha_inequality_l2379_237979


namespace NUMINAMATH_CALUDE_coins_equal_dollar_l2379_237960

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The theorem stating that the sum of the coins equals 100% of a dollar -/
theorem coins_equal_dollar :
  (nickel_value + 2 * dime_value + quarter_value + half_dollar_value) / cents_per_dollar * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_equal_dollar_l2379_237960


namespace NUMINAMATH_CALUDE_gcd_statements_l2379_237994

theorem gcd_statements : 
  (Nat.gcd 16 12 = 4) ∧ 
  (Nat.gcd 78 36 = 6) ∧ 
  (Nat.gcd 105 315 = 105) ∧
  (Nat.gcd 85 357 ≠ 34) := by
sorry

end NUMINAMATH_CALUDE_gcd_statements_l2379_237994


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2379_237970

theorem geometric_sequence_sixth_term 
  (a : ℝ) -- first term
  (a₇ : ℝ) -- 7th term
  (h₁ : a = 1024)
  (h₂ : a₇ = 16)
  : ∃ r : ℝ, r > 0 ∧ a * r^6 = a₇ ∧ a * r^5 = 32 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2379_237970


namespace NUMINAMATH_CALUDE_carnation_percentage_l2379_237988

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  carnations : ℝ
  violets : ℝ
  tulips : ℝ
  roses : ℝ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 3 ∧
  shop.roses = shop.tulips

/-- Theorem stating the percentage of carnations in the flower shop -/
theorem carnation_percentage (shop : FlowerShop) 
  (h : validFlowerShop shop) (h_pos : shop.carnations > 0) : 
  shop.carnations / (shop.carnations + shop.violets + shop.tulips + shop.roses) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l2379_237988


namespace NUMINAMATH_CALUDE_degrees_to_radians_l2379_237901

theorem degrees_to_radians (degrees : ℝ) (radians : ℝ) : 
  degrees = 12 → radians = degrees * (π / 180) → radians = π / 15 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_l2379_237901


namespace NUMINAMATH_CALUDE_kaleb_books_l2379_237968

theorem kaleb_books (initial_books sold_books new_books : ℕ) : 
  initial_books = 34 → sold_books = 17 → new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_books_l2379_237968


namespace NUMINAMATH_CALUDE_two_integers_problem_l2379_237954

theorem two_integers_problem (x y : ℕ+) :
  (x / Nat.gcd x y + y / Nat.gcd x y : ℚ) = 18 →
  Nat.lcm x y = 975 →
  (x = 75 ∧ y = 195) ∨ (x = 195 ∧ y = 75) := by
  sorry

end NUMINAMATH_CALUDE_two_integers_problem_l2379_237954


namespace NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l2379_237973

/-- Given positive real numbers a and b satisfying a + 4b = 4,
    prove the maximum value of ab and the minimum value of 1/a + 4/b -/
theorem ab_max_and_reciprocal_sum_min (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) : 
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4*y = 4 ∧ a*b ≤ x*y) ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + 4*y = 4 → a*b ≤ x*y) ∧
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 4*y = 4 ∧ 1/x + 4/y ≥ 25/4) ∧ 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + 4*y = 4 → 1/x + 4/y ≥ 25/4) := by
  sorry

end NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l2379_237973


namespace NUMINAMATH_CALUDE_prom_attendance_l2379_237951

theorem prom_attendance (total_students : ℕ) (couples : ℕ) (solo_students : ℕ) : 
  total_students = 123 → couples = 60 → solo_students = total_students - 2 * couples →
  solo_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_prom_attendance_l2379_237951


namespace NUMINAMATH_CALUDE_cheerleader_group_composition_cheerleader_group_composition_result_l2379_237908

theorem cheerleader_group_composition 
  (total_females : Nat) 
  (males_chose_malt : Nat) 
  (females_chose_malt : Nat) : Nat :=
  let total_malt := males_chose_malt + females_chose_malt
  let total_coke := total_malt / 2
  let total_cheerleaders := total_malt + total_coke
  let total_males := total_cheerleaders - total_females
  
  have h1 : total_females = 16 := by sorry
  have h2 : males_chose_malt = 6 := by sorry
  have h3 : females_chose_malt = 8 := by sorry
  
  total_males

theorem cheerleader_group_composition_result : 
  cheerleader_group_composition 16 6 8 = 5 := by sorry

end NUMINAMATH_CALUDE_cheerleader_group_composition_cheerleader_group_composition_result_l2379_237908


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2379_237910

theorem two_numbers_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 30) (h3 : x * y = 200) :
  x - y = 10 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2379_237910


namespace NUMINAMATH_CALUDE_all_points_same_number_l2379_237942

-- Define a type for points in the plane
structure Point := (x : ℝ) (y : ℝ)

-- Define a function that assigns a real number to each point
def assign : Point → ℝ := sorry

-- Define a predicate for the inscribed circle property
def inscribedCircleProperty (assign : Point → ℝ) : Prop :=
  ∀ A B C : Point,
  ∃ I : Point,
  assign I = (assign A + assign B + assign C) / 3

-- Theorem statement
theorem all_points_same_number
  (h : inscribedCircleProperty assign) :
  ∀ P Q : Point, assign P = assign Q :=
sorry

end NUMINAMATH_CALUDE_all_points_same_number_l2379_237942


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2379_237996

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of tangency between a line and a circle -/
def is_tangent (circle line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (ρ₀ θ₀ : ℝ), circle ρ₀ θ₀ ∧ line ρ₀ θ₀ ∧
    ∀ (ρ θ : ℝ), circle ρ θ ∧ line ρ θ → (ρ = ρ₀ ∧ θ = θ₀)

theorem line_tangent_to_circle :
  is_tangent circle_equation line_equation :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2379_237996


namespace NUMINAMATH_CALUDE_solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l2379_237906

-- System of linear equations 1
theorem solve_system_1 (x y : ℝ) : 
  x = 7 * y ∧ 2 * x + y = 30 → x = 14 ∧ y = 2 := by sorry

-- System of linear equations 2
theorem solve_system_2 (x y : ℝ) : 
  x / 2 + y / 3 = 7 ∧ x / 3 - y / 4 = -1 → x = 6 ∧ y = 12 := by sorry

-- Linear inequality
theorem solve_inequality (x : ℝ) :
  4 + 3 * (x - 1) > -5 ↔ x > -2 := by sorry

-- System of linear inequalities
theorem solve_inequality_system (x : ℝ) :
  (1 / 2 * (x - 2) + 3 > 7 ∧ -1 / 3 * (x + 3) - 4 > -10) ↔ (x > 10 ∧ x < 15) := by sorry

end NUMINAMATH_CALUDE_solve_system_1_solve_system_2_solve_inequality_solve_inequality_system_l2379_237906
