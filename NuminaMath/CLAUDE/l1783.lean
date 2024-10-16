import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_area_l1783_178387

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1783_178387


namespace NUMINAMATH_CALUDE_track_meet_boys_count_l1783_178386

theorem track_meet_boys_count :
  ∀ (total girls boys : ℕ),
  total = 55 →
  total = girls + boys →
  (3 * girls : ℚ) / 5 + (2 * girls : ℚ) / 5 = girls →
  (2 * girls : ℚ) / 5 = 10 →
  boys = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_track_meet_boys_count_l1783_178386


namespace NUMINAMATH_CALUDE_square_area_increase_l1783_178332

/-- The increase in area of a square when its side length is increased by 2 -/
theorem square_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l1783_178332


namespace NUMINAMATH_CALUDE_triangle_n_range_l1783_178352

-- Define a triangle with the given properties
structure Triangle where
  n : ℝ
  angle1 : ℝ := 180 - n
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  angle_difference : max angle1 (max angle2 angle3) - min angle1 (min angle2 angle3) = 24

-- Theorem statement
theorem triangle_n_range (t : Triangle) : 104 ≤ t.n ∧ t.n ≤ 136 := by
  sorry

end NUMINAMATH_CALUDE_triangle_n_range_l1783_178352


namespace NUMINAMATH_CALUDE_complex_subtraction_l1783_178370

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1783_178370


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1783_178371

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p + q = 65 ∧ p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1783_178371


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1783_178353

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    focal length 2√6, and an asymptote l such that the distance from (1,0) to l is √6/3,
    prove that the equation of the hyperbola is x²/2 - y²/4 = 1. -/
theorem hyperbola_equation (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_focal : Real.sqrt (a^2 + b^2) = Real.sqrt 6)
  (h_asymptote : ∃ (k : ℝ), k * a = b ∧ k * b = a)
  (h_distance : (b / Real.sqrt (a^2 + b^2)) = Real.sqrt 6 / 3) :
  a^2 = 2 ∧ b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1783_178353


namespace NUMINAMATH_CALUDE_curve_equation_l1783_178373

/-- Given vectors and their relationships, prove the equation of the resulting curve. -/
theorem curve_equation (x y : ℝ) : 
  let m₁ : ℝ × ℝ := (0, x)
  let n₁ : ℝ × ℝ := (1, 1)
  let m₂ : ℝ × ℝ := (x, 0)
  let n₂ : ℝ × ℝ := (y^2, 1)
  let m : ℝ × ℝ := m₁ + Real.sqrt 2 • n₂
  let n : ℝ × ℝ := m₂ - Real.sqrt 2 • n₁
  (m.1 * n.2 = m.2 * n.1) →  -- m is parallel to n
  x^2 / 2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_l1783_178373


namespace NUMINAMATH_CALUDE_interest_calculation_l1783_178394

/-- Given a principal amount and number of years, proves that if simple interest
    at 5% per annum is 50 and compound interest at the same rate is 51.25,
    then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 50 →
  P * ((1 + 5/100)^n - 1) = 51.25 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l1783_178394


namespace NUMINAMATH_CALUDE_inequality_solution_l1783_178357

theorem inequality_solution (x : ℝ) : 
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ 4 < x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1783_178357


namespace NUMINAMATH_CALUDE_infinite_intersecting_lines_l1783_178303

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry  -- Definition of skew lines

/-- A set of three pairwise skew lines -/
structure SkewLineTriple where
  a : Line3D
  b : Line3D
  c : Line3D
  skew_ab : are_skew a b
  skew_bc : are_skew b c
  skew_ca : are_skew c a

/-- The set of lines intersecting all three lines in a SkewLineTriple -/
def intersecting_lines (triple : SkewLineTriple) : Set Line3D :=
  sorry  -- Definition of the set of intersecting lines

/-- Theorem stating that there are infinitely many intersecting lines -/
theorem infinite_intersecting_lines (triple : SkewLineTriple) :
  Set.Infinite (intersecting_lines triple) :=
sorry

end NUMINAMATH_CALUDE_infinite_intersecting_lines_l1783_178303


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1783_178326

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, a x y ≠ 0 ∧ b x y ≠ 0 → (a x y = b x y)

/-- The first monomial 5x^4y -/
def mono1 (x y : ℕ) : ℚ := 5 * x^4 * y

/-- The second monomial 5x^ny^m -/
def mono2 (n m x y : ℕ) : ℚ := 5 * x^n * y^m

theorem like_terms_exponent_sum :
  are_like_terms mono1 (mono2 n m) → n + m = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1783_178326


namespace NUMINAMATH_CALUDE_slope_intercept_form_l1783_178311

theorem slope_intercept_form (x y : ℝ) :
  3 * x - 2 * y = 4 ↔ y = (3/2) * x - 2 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_form_l1783_178311


namespace NUMINAMATH_CALUDE_f_is_even_f_monotonicity_on_0_1_l1783_178314

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 / (x^2 - 1)

-- Theorem for the even property of f
theorem f_is_even (a : ℝ) (ha : a ≠ 0) :
  ∀ x, x ≠ 1 ∧ x ≠ -1 → f a (-x) = f a x :=
sorry

-- Theorem for the monotonicity of f on (0, 1)
theorem f_monotonicity_on_0_1 (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 →
    (a > 0 → f a x₁ > f a x₂) ∧
    (a < 0 → f a x₁ < f a x₂) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_f_monotonicity_on_0_1_l1783_178314


namespace NUMINAMATH_CALUDE_prob_three_l_is_one_fifty_fifth_l1783_178356

def total_cards : ℕ := 12
def l_cards : ℕ := 4

def prob_three_l : ℚ :=
  (l_cards / total_cards) *
  ((l_cards - 1) / (total_cards - 1)) *
  ((l_cards - 2) / (total_cards - 2))

theorem prob_three_l_is_one_fifty_fifth :
  prob_three_l = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_l_is_one_fifty_fifth_l1783_178356


namespace NUMINAMATH_CALUDE_train_platform_length_l1783_178376

/-- Given a train passing a pole in t seconds and a platform in 6t seconds at constant velocity,
    prove that the length of the platform is 5 times the length of the train. -/
theorem train_platform_length
  (t : ℝ)
  (train_length : ℝ)
  (platform_length : ℝ)
  (velocity : ℝ)
  (h1 : velocity = train_length / t)
  (h2 : velocity = (train_length + platform_length) / (6 * t))
  : platform_length = 5 * train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_l1783_178376


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l1783_178310

/-- The number of daps equivalent to 1 dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to 1 dip -/
def dops_per_dip : ℚ := 3 / 10

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 60

theorem daps_equivalent_to_dips : 
  (daps_per_dop * dops_per_dip * target_dips : ℚ) = 45/2 := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l1783_178310


namespace NUMINAMATH_CALUDE_janes_number_l1783_178368

theorem janes_number : ∃ x : ℚ, 5 * (3 * x + 16) = 250 ∧ x = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_janes_number_l1783_178368


namespace NUMINAMATH_CALUDE_unique_valid_number_l1783_178395

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- It begins with digit 1
  n / 100000 = 1 ∧
  -- It ends with digit 7
  n % 10 = 7 ∧
  -- If the last digit is decreased by 1 and moved to the first place,
  -- the resulting number is five times the original number
  (6 * 100000 + n / 10) = 5 * n

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1783_178395


namespace NUMINAMATH_CALUDE_abc_inequality_l1783_178333

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 ∧ 1/a + 1/b + 1/c ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1783_178333


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1783_178364

/-- The volume of a sphere inscribed in a right circular cylinder -/
theorem inscribed_sphere_volume (h : ℝ) (d : ℝ) (h_pos : h > 0) (d_pos : d > 0) :
  let r : ℝ := d / 2
  let cylinder_volume : ℝ := π * r^2 * h
  let sphere_volume : ℝ := (4/3) * π * r^3
  h = 12 ∧ d = 10 → sphere_volume = (500/3) * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1783_178364


namespace NUMINAMATH_CALUDE_four_distinct_solutions_l1783_178380

theorem four_distinct_solutions (p q : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (x₁^2 + p * |x₁| = q * x₁ - 1) ∧
    (x₂^2 + p * |x₂| = q * x₂ - 1) ∧
    (x₃^2 + p * |x₃| = q * x₃ - 1) ∧
    (x₄^2 + p * |x₄| = q * x₄ - 1)) ↔
  (p + |q| + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_four_distinct_solutions_l1783_178380


namespace NUMINAMATH_CALUDE_journey_time_calculation_l1783_178398

/-- Proves that if walking twice the distance of running takes 30 minutes,
    then walking one-third and running two-thirds of the same distance takes 24 minutes,
    given that running speed is twice the walking speed. -/
theorem journey_time_calculation (v : ℝ) (S : ℝ) (h1 : v > 0) (h2 : S > 0) :
  (2 * S / v + S / (2 * v) = 30) →
  (S / v + 2 * S / (2 * v) = 24) :=
by sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l1783_178398


namespace NUMINAMATH_CALUDE_fraction_simplification_l1783_178340

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 5 + 2 * Real.sqrt 45) = (25 * Real.sqrt 3 - 45 * Real.sqrt 5) / 330 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1783_178340


namespace NUMINAMATH_CALUDE_simplify_fraction_l1783_178375

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1783_178375


namespace NUMINAMATH_CALUDE_special_triangle_sides_l1783_178377

/-- An isosceles triangle with perimeter 60 and centroid on the inscribed circle. -/
structure SpecialTriangle where
  -- Two equal sides
  a : ℝ
  -- Third side
  b : ℝ
  -- Perimeter is 60
  perimeter_eq : 2 * a + b = 60
  -- a > 0 and b > 0
  a_pos : a > 0
  b_pos : b > 0
  -- Centroid on inscribed circle condition
  centroid_on_inscribed : 3 * (a * b) = 60 * (a + b - (2 * a + b) / 2)

/-- The sides of a special triangle are 25, 25, and 10. -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 25 ∧ t.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l1783_178377


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1783_178345

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1783_178345


namespace NUMINAMATH_CALUDE_weight_of_b_l1783_178396

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l1783_178396


namespace NUMINAMATH_CALUDE_existence_of_sum_equality_l1783_178363

theorem existence_of_sum_equality (A : Set ℕ) 
  (h : ∀ n : ℕ, ∃ m ∈ A, n ≤ m ∧ m < n + 100) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c + d :=
sorry

end NUMINAMATH_CALUDE_existence_of_sum_equality_l1783_178363


namespace NUMINAMATH_CALUDE_quadratic_odd_coefficients_irrational_roots_l1783_178374

theorem quadratic_odd_coefficients_irrational_roots (a b c : ℤ) :
  (Odd a ∧ Odd b ∧ Odd c) →
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_odd_coefficients_irrational_roots_l1783_178374


namespace NUMINAMATH_CALUDE_train_distance_l1783_178372

/-- The distance between two towns given train speeds and meeting time -/
theorem train_distance (faster_speed slower_speed meeting_time : ℝ) : 
  faster_speed = 48 ∧ 
  faster_speed = slower_speed + 6 ∧ 
  meeting_time = 5 →
  (faster_speed + slower_speed) * meeting_time = 450 := by
sorry

end NUMINAMATH_CALUDE_train_distance_l1783_178372


namespace NUMINAMATH_CALUDE_system_solution_l1783_178317

theorem system_solution : ∃ (a b c d : ℝ), 
  (a + c = -4 ∧ 
   a * c + b + d = 6 ∧ 
   a * d + b * c = -5 ∧ 
   b * d = 2) ∧
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1783_178317


namespace NUMINAMATH_CALUDE_fish_pond_problem_l1783_178348

theorem fish_pond_problem (initial_fish : ℕ) : 
  (∃ (initial_tadpoles : ℕ),
    initial_tadpoles = 3 * initial_fish ∧
    initial_tadpoles / 2 = (initial_fish - 7) + 32) →
  initial_fish = 50 := by
sorry

end NUMINAMATH_CALUDE_fish_pond_problem_l1783_178348


namespace NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l1783_178312

/-- Given the total number of votes and the margin of loss, 
    calculate the percentage of votes received by the losing candidate. -/
theorem losing_candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h1 : total_votes = 7800)
  (h2 : loss_margin = 2340) :
  (total_votes - loss_margin) * 100 / total_votes = 70 := by
  sorry

end NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l1783_178312


namespace NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l1783_178322

-- Define the characteristics of a parallelepiped
structure Parallelepiped :=
  (has_parallel_faces : Bool)

-- Define planar figures
inductive PlanarFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the analogy relation
def is_analogous (p : Parallelepiped) (f : PlanarFigure) : Prop :=
  match f with
  | PlanarFigure.Parallelogram => p.has_parallel_faces
  | _ => False

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (p : Parallelepiped) (f : PlanarFigure),
    p.has_parallel_faces →
    is_analogous p f →
    f = PlanarFigure.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l1783_178322


namespace NUMINAMATH_CALUDE_rmb_notes_problem_l1783_178324

theorem rmb_notes_problem (x y z : ℕ) : 
  x + y + z = 33 →
  x + 5 * y + 10 * z = 187 →
  y = x - 5 →
  (x = 12 ∧ y = 7 ∧ z = 14) :=
by sorry

end NUMINAMATH_CALUDE_rmb_notes_problem_l1783_178324


namespace NUMINAMATH_CALUDE_martha_cards_remaining_l1783_178350

theorem martha_cards_remaining (initial_cards : ℝ) (cards_to_emily : ℝ) (cards_to_olivia : ℝ) :
  initial_cards = 76.5 →
  cards_to_emily = 3.1 →
  cards_to_olivia = 5.2 →
  initial_cards - (cards_to_emily + cards_to_olivia) = 68.2 := by
sorry

end NUMINAMATH_CALUDE_martha_cards_remaining_l1783_178350


namespace NUMINAMATH_CALUDE_total_profit_is_2034_l1783_178361

/-- Represents a group of piglets with their selling and feeding information -/
structure PigletGroup where
  count : Nat
  sellingPrice : Nat
  sellingTime : Nat
  initialFeedCost : Nat
  initialFeedTime : Nat
  laterFeedCost : Nat
  laterFeedTime : Nat

/-- Calculates the profit for a single piglet group -/
def groupProfit (group : PigletGroup) : Int :=
  group.count * group.sellingPrice - 
  group.count * (group.initialFeedCost * group.initialFeedTime + 
                 group.laterFeedCost * group.laterFeedTime)

/-- The farmer's piglet groups -/
def pigletGroups : List PigletGroup := [
  ⟨3, 375, 11, 13, 8, 15, 3⟩,
  ⟨4, 425, 14, 14, 5, 16, 9⟩,
  ⟨2, 475, 18, 15, 10, 18, 8⟩,
  ⟨1, 550, 20, 20, 20, 20, 0⟩
]

/-- Theorem stating the total profit is $2034 -/
theorem total_profit_is_2034 : 
  (pigletGroups.map groupProfit).sum = 2034 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_2034_l1783_178361


namespace NUMINAMATH_CALUDE_class_strength_solution_l1783_178359

/-- Represents the problem of finding the original class strength --/
def find_original_class_strength (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : Prop :=
  ∃ (original_strength : ℕ),
    (original_strength : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((original_strength : ℝ) + new_students) * (original_avg - avg_decrease)

/-- The theorem stating the solution to the class strength problem --/
theorem class_strength_solution : 
  find_original_class_strength 40 18 32 4 → 
  ∃ (original_strength : ℕ), original_strength = 18 :=
by
  sorry

#check class_strength_solution

end NUMINAMATH_CALUDE_class_strength_solution_l1783_178359


namespace NUMINAMATH_CALUDE_particle_prob_origin_prob_form_l1783_178335

/-- A particle starts at (4,4) and moves randomly until it hits a coordinate axis. 
    At each step, it moves to one of (a-1, b), (a, b-1), or (a-1, b-1) with equal probability. -/
def particle_movement (a b : ℕ) : Fin 3 → ℕ × ℕ
| 0 => (a - 1, b)
| 1 => (a, b - 1)
| 2 => (a - 1, b - 1)

/-- The probability of the particle reaching (0,0) when starting from (4,4) -/
def prob_reach_origin : ℚ :=
  63 / 3^8

/-- Theorem stating that the probability of reaching (0,0) is 63/3^8 -/
theorem particle_prob_origin : 
  prob_reach_origin = 63 / 3^8 := by sorry

/-- The probability can be expressed as m/3^n where m is not divisible by 3 -/
theorem prob_form (m n : ℕ) (h : m % 3 ≠ 0) : 
  prob_reach_origin = m / 3^n := by sorry

end NUMINAMATH_CALUDE_particle_prob_origin_prob_form_l1783_178335


namespace NUMINAMATH_CALUDE_principal_is_15000_l1783_178334

/-- Represents the loan details and calculations -/
structure Loan where
  principal : ℝ
  interestRates : Fin 3 → ℝ
  totalInterest : ℝ

/-- Calculates the total interest paid over 3 years -/
def totalInterestPaid (loan : Loan) : ℝ :=
  (loan.interestRates 0 + loan.interestRates 1 + loan.interestRates 2) * loan.principal

/-- Theorem stating that given the conditions, the principal amount is 15000 -/
theorem principal_is_15000 (loan : Loan)
  (h1 : loan.interestRates 0 = 0.10)
  (h2 : loan.interestRates 1 = 0.12)
  (h3 : loan.interestRates 2 = 0.14)
  (h4 : loan.totalInterest = 5400)
  (h5 : totalInterestPaid loan = loan.totalInterest) :
  loan.principal = 15000 := by
  sorry

#check principal_is_15000

end NUMINAMATH_CALUDE_principal_is_15000_l1783_178334


namespace NUMINAMATH_CALUDE_beaver_group_size_l1783_178399

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is 20 beavers -/
theorem beaver_group_size :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end NUMINAMATH_CALUDE_beaver_group_size_l1783_178399


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l1783_178320

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passes_through (line : RegressionLine) (point : ℝ × ℝ) : Prop :=
  let (x, y) := point
  y = line.slope * x + line.intercept

theorem regression_lines_intersection
  (t1 t2 : RegressionLine)
  (s t : ℝ)
  (h1 : passes_through t1 (s, t))
  (h2 : passes_through t2 (s, t)) :
  ∃ (x y : ℝ), passes_through t1 (x, y) ∧ passes_through t2 (x, y) ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l1783_178320


namespace NUMINAMATH_CALUDE_bike_savings_time_l1783_178346

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 600

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 60 + 40 + 20

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks it takes to save enough money for the bike -/
def weeks_to_save : ℕ := 24

/-- Theorem stating that it takes 24 weeks to save enough money for the bike -/
theorem bike_savings_time :
  birthday_money + weekly_earnings * weeks_to_save = bike_cost := by
  sorry

end NUMINAMATH_CALUDE_bike_savings_time_l1783_178346


namespace NUMINAMATH_CALUDE_snack_cost_per_person_l1783_178358

/-- Calculates the cost per person for a group of friends buying snacks -/
theorem snack_cost_per_person 
  (num_friends : ℕ) 
  (num_fish_cakes : ℕ) 
  (fish_cake_price : ℕ) 
  (num_tteokbokki : ℕ) 
  (tteokbokki_price : ℕ) 
  (h1 : num_friends = 4) 
  (h2 : num_fish_cakes = 5) 
  (h3 : fish_cake_price = 200) 
  (h4 : num_tteokbokki = 7) 
  (h5 : tteokbokki_price = 800) : 
  (num_fish_cakes * fish_cake_price + num_tteokbokki * tteokbokki_price) / num_friends = 1650 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_per_person_l1783_178358


namespace NUMINAMATH_CALUDE_min_students_for_duplicate_vote_l1783_178336

theorem min_students_for_duplicate_vote (n : ℕ) (h : n = 10) :
  let combinations := n.choose 2
  ∃ k : ℕ, k > combinations ∧
    ∀ m : ℕ, m < k → ∃ f : Fin m → Fin n × Fin n,
      Function.Injective f ∧
      ∀ i : Fin m, (f i).1 < (f i).2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_for_duplicate_vote_l1783_178336


namespace NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_34_l1783_178391

def game_condition (M : ℕ) : Prop :=
  M ≤ 1999 ∧
  3 * M < 2000 ∧
  3 * M + 80 < 2000 ∧
  3 * (3 * M + 80) < 2000 ∧
  3 * (3 * M + 80) + 80 < 2000 ∧
  3 * (3 * (3 * M + 80) + 80) ≥ 2000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 34 → ¬(game_condition n) ∧ game_condition 34 :=
by sorry

theorem sum_of_digits_34 : (3 : ℕ) + 4 = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_34_l1783_178391


namespace NUMINAMATH_CALUDE_fraction_equality_l1783_178388

theorem fraction_equality (x : ℝ) : 
  (4 + 2*x) / (7 + 3*x) = (2 + 3*x) / (4 + 5*x) ↔ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1783_178388


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l1783_178323

theorem trig_expression_equals_one : 
  (Real.cos (36 * π / 180) * Real.sin (24 * π / 180) + 
   Real.sin (144 * π / 180) * Real.sin (84 * π / 180)) / 
  (Real.cos (44 * π / 180) * Real.sin (16 * π / 180) + 
   Real.sin (136 * π / 180) * Real.sin (76 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l1783_178323


namespace NUMINAMATH_CALUDE_triangle_properties_l1783_178302

theorem triangle_properties (A B C : ℝ) (a b c R : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  R > 0 →
  a = Real.sqrt 3 →
  A = π/3 →
  2 * R = a / Real.sin A →
  2 * R = b / Real.sin B →
  2 * R = c / Real.sin C →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  R = 1 ∧ ∀ (b' c' : ℝ), b' * c' ≤ 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1783_178302


namespace NUMINAMATH_CALUDE_train_meeting_correct_l1783_178329

/-- Represents the properties of two trains meeting between two cities -/
structure TrainMeeting where
  normal_meet_time : ℝ  -- in hours
  early_a_distance : ℝ  -- in km
  early_b_distance : ℝ  -- in km
  early_time : ℝ        -- in hours

/-- The solution to the train meeting problem -/
def train_meeting_solution (tm : TrainMeeting) : ℝ × ℝ × ℝ :=
  let distance := 660
  let speed_a := 115
  let speed_b := 85
  (distance, speed_a, speed_b)

/-- Theorem stating that the given solution satisfies the train meeting conditions -/
theorem train_meeting_correct (tm : TrainMeeting) 
  (h1 : tm.normal_meet_time = 3 + 18/60)
  (h2 : tm.early_a_distance = 14)
  (h3 : tm.early_b_distance = 9)
  (h4 : tm.early_time = 3) :
  let (distance, speed_a, speed_b) := train_meeting_solution tm
  (speed_a + speed_b) * tm.normal_meet_time = distance ∧
  speed_a * (tm.normal_meet_time + 24/60) = distance - tm.early_a_distance + speed_b * tm.early_time ∧
  speed_b * (tm.normal_meet_time + 36/60) = distance - tm.early_b_distance + speed_a * tm.early_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_meeting_correct_l1783_178329


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1783_178309

theorem nested_fraction_evaluation :
  2 + 2 / (2 + 2 / (2 + 3)) = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1783_178309


namespace NUMINAMATH_CALUDE_movie_production_profit_l1783_178347

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let num_main_actors : ℕ := 2
  let num_supporting_actors : ℕ := 3
  let num_extra : ℕ := 1
  let total_people : ℕ := 50
  let actor_fees := num_main_actors * main_actor_fee + 
                    num_supporting_actors * supporting_actor_fee + 
                    num_extra * extra_fee
  let food_cost := num_main_actors * main_actor_food + 
                   (num_supporting_actors + num_extra) * supporting_actor_food + 
                   (total_people - num_main_actors - num_supporting_actors - num_extra) * crew_food
  let equipment_cost := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_cost + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_production_profit :
  movie_production 500 100 50 10 5 3 850 10000 := by
  sorry

end NUMINAMATH_CALUDE_movie_production_profit_l1783_178347


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_DFEJ_l1783_178330

/-- Given a right isosceles triangle ABC with side lengths AB = AC = 10 and BC = 10√2,
    and points D, E, F as midpoints of AB, BC, AC respectively,
    and J as the midpoint of DE,
    the area of quadrilateral DFEJ is 6.25. -/
theorem area_of_quadrilateral_DFEJ (A B C D E F J : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  A = (0, 0) →
  B = (0, 10) →
  C = (10, 0) →
  d A B = 10 →
  d A C = 10 →
  d B C = 10 * Real.sqrt 2 →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  J = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  abs ((D.1 * F.2 + F.1 * E.2 + E.1 * J.2 + J.1 * D.2) -
       (D.2 * F.1 + F.2 * E.1 + E.2 * J.1 + J.2 * D.1)) / 2 = 6.25 :=
by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_DFEJ_l1783_178330


namespace NUMINAMATH_CALUDE_polynomial_value_for_quadratic_roots_l1783_178300

theorem polynomial_value_for_quadratic_roots : 
  ∀ x : ℝ, x^2 - 4*x + 1 = 0 → 
  (x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3) ∨
  (x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_for_quadratic_roots_l1783_178300


namespace NUMINAMATH_CALUDE_hospital_staff_ratio_l1783_178393

theorem hospital_staff_ratio (total : ℕ) (nurses : ℕ) (doctors : ℕ) :
  total = 250 →
  nurses = 150 →
  doctors = total - nurses →
  (doctors : ℚ) / nurses = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_hospital_staff_ratio_l1783_178393


namespace NUMINAMATH_CALUDE_amount_lent_to_C_l1783_178325

/-- Amount lent to B in rupees -/
def amount_B : ℝ := 5000

/-- Duration of loan to B in years -/
def duration_B : ℝ := 2

/-- Duration of loan to C in years -/
def duration_C : ℝ := 4

/-- Annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- Total interest received from both B and C in rupees -/
def total_interest : ℝ := 1760

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem amount_lent_to_C : ∃ (amount_C : ℝ),
  amount_C = 3000 ∧
  simple_interest amount_B interest_rate duration_B +
  simple_interest amount_C interest_rate duration_C = total_interest :=
sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_l1783_178325


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1783_178307

/-- Given that -l, a, b, c, and -9 form a geometric sequence, prove that b = -3 and ac = 9 -/
theorem geometric_sequence_problem (l a b c : ℝ) 
  (h1 : ∃ (r : ℝ), a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : 
  b = -3 ∧ a * c = 9 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1783_178307


namespace NUMINAMATH_CALUDE_negative_two_squared_times_negative_two_squared_l1783_178316

theorem negative_two_squared_times_negative_two_squared : -2^2 * (-2)^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_times_negative_two_squared_l1783_178316


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l1783_178384

/-- The probability that two groups of tourists can contact each other -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

/-- Theorem: Given two groups of tourists with 5 and 8 members respectively,
    and the probability p that a tourist from the first group has the phone number
    of a tourist from the second group, the probability that the two groups
    will be able to contact each other is 1 - (1-p)^40. -/
theorem tourist_contact_probability (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ 40 := by
  sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l1783_178384


namespace NUMINAMATH_CALUDE_simplify_expression_l1783_178343

theorem simplify_expression :
  (Real.sqrt 2 + 1) ^ (1 - Real.sqrt 3) / (Real.sqrt 2 - 1) ^ (1 + Real.sqrt 3) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1783_178343


namespace NUMINAMATH_CALUDE_angle_I_measure_l1783_178315

-- Define the pentagon and its angles
structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.F > 0 ∧ p.G > 0 ∧ p.H > 0 ∧ p.I > 0 ∧ p.J > 0 ∧
  p.F + p.G + p.H + p.I + p.J = 540

-- Define the conditions given in the problem
def satisfies_conditions (p : Pentagon) : Prop :=
  p.F = p.G ∧ p.G = p.H ∧
  p.I = p.J ∧
  p.I = p.F + 30

-- Theorem statement
theorem angle_I_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : satisfies_conditions p) : 
  p.I = 126 := by
  sorry

end NUMINAMATH_CALUDE_angle_I_measure_l1783_178315


namespace NUMINAMATH_CALUDE_club_sports_intersection_l1783_178366

/-- Given a club with 310 members, where 138 play tennis, 255 play baseball,
    and 11 play no sports, prove that 94 people play both tennis and baseball. -/
theorem club_sports_intersection (total : ℕ) (tennis : ℕ) (baseball : ℕ) (no_sport : ℕ)
    (h_total : total = 310)
    (h_tennis : tennis = 138)
    (h_baseball : baseball = 255)
    (h_no_sport : no_sport = 11) :
    tennis + baseball - (total - no_sport) = 94 := by
  sorry

end NUMINAMATH_CALUDE_club_sports_intersection_l1783_178366


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1783_178362

-- Define the function f(x) = -x³
def f (x : ℝ) : ℝ := -x^3

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1783_178362


namespace NUMINAMATH_CALUDE_f_composition_value_l1783_178327

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x + 2 * Real.cos (2 * x) else -Real.exp (2 * x)

theorem f_composition_value : f (f (Real.pi / 2)) = -1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1783_178327


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l1783_178397

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

-- Define the theorem
theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ (a : ℕ → ℝ),
    is_positive_geometric_sequence a →
    a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
    a 5 + a 6 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l1783_178397


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l1783_178369

/-- A decreasing function on (0, +∞) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l1783_178369


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l1783_178337

/-- The length of a rectangular garden with perimeter 900 m and breadth 190 m is 260 m. -/
theorem rectangular_garden_length : 
  ∀ (length breadth : ℝ),
  breadth = 190 →
  2 * (length + breadth) = 900 →
  length = 260 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l1783_178337


namespace NUMINAMATH_CALUDE_pressure_valve_problem_l1783_178318

/-- Represents the constant ratio between pressure change and temperature -/
def k : ℚ := (5 * 4 - 6) / (10 + 20)

/-- The pressure-temperature relationship function -/
def pressure_temp_relation (x t : ℚ) : Prop :=
  (5 * x - 6) / (t + 20) = k

theorem pressure_valve_problem :
  pressure_temp_relation 4 10 →
  pressure_temp_relation (34/5) 40 :=
by sorry

end NUMINAMATH_CALUDE_pressure_valve_problem_l1783_178318


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1783_178308

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (6, 2)
  let b : ℝ × ℝ := (-2, k)
  parallel a b → k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1783_178308


namespace NUMINAMATH_CALUDE_farah_order_match_sticks_l1783_178367

/-- The number of boxes Farah ordered -/
def num_boxes : ℕ := 4

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks Farah ordered -/
def total_match_sticks : ℕ := num_boxes * matchboxes_per_box * sticks_per_matchbox

theorem farah_order_match_sticks :
  total_match_sticks = 24000 := by
  sorry

end NUMINAMATH_CALUDE_farah_order_match_sticks_l1783_178367


namespace NUMINAMATH_CALUDE_banana_price_reduction_l1783_178306

/-- Proves that a price reduction resulting in 64 more bananas for Rs. 40.00001 
    and a new price of Rs. 3 per dozen represents a 40% reduction from the original price. -/
theorem banana_price_reduction (original_price : ℚ) : 
  (40.00001 / 3 - 40.00001 / original_price = 64 / 12) →
  (3 / original_price = 0.6) := by
  sorry

#eval (1 - 3/5) * 100 -- Should evaluate to 40

end NUMINAMATH_CALUDE_banana_price_reduction_l1783_178306


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1783_178351

theorem sufficient_not_necessary_condition (x : ℝ) :
  {x | 1 / x > 1} ⊂ {x | Real.exp (x - 1) < 1} ∧ {x | 1 / x > 1} ≠ {x | Real.exp (x - 1) < 1} :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1783_178351


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l1783_178319

-- Define a function to count the number of divisors of a positive integer
def countDivisors (n : ℕ+) : ℕ := sorry

-- Define a function to check if a number has exactly 12 divisors
def hasTwelveDivisors (n : ℕ+) : Prop :=
  countDivisors n = 12

-- Theorem statement
theorem smallest_with_twelve_divisors :
  ∀ n : ℕ+, hasTwelveDivisors n → n ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l1783_178319


namespace NUMINAMATH_CALUDE_solve_equation_l1783_178389

theorem solve_equation : ∃ x : ℝ, 7 * (x - 1) = 21 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1783_178389


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1783_178305

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := y^2/4 + x^2/2 = 1

-- Define the line
def Line (x y m k : ℝ) : Prop := y = k*x + m

-- Define the intersection condition
def Intersects (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ 
  Line x₁ y₁ m (y₁ / x₁) ∧ Line x₂ y₂ m (y₂ / x₂) ∧
  x₁ ≠ x₂ ∧ y₁ ≠ y₂

-- Define the vector condition
def VectorCondition (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ),
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧
  Line x₁ y₁ m (y₁ / x₁) ∧ Line x₂ y₂ m (y₂ / x₂) ∧
  x₁ + 2*x₂ = 0 ∧ y₁ + 2*y₂ = 3*m

theorem ellipse_intersection_theorem (m : ℝ) : 
  Intersects m ∧ VectorCondition m → 
  (2/3 < m ∧ m < 2) ∨ (-2 < m ∧ m < -2/3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1783_178305


namespace NUMINAMATH_CALUDE_maria_cookies_l1783_178341

theorem maria_cookies (cookies_per_bag : ℕ) (chocolate_chip : ℕ) (baggies : ℕ) 
  (h1 : cookies_per_bag = 8)
  (h2 : chocolate_chip = 5)
  (h3 : baggies = 3) :
  cookies_per_bag * baggies - chocolate_chip = 19 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_l1783_178341


namespace NUMINAMATH_CALUDE_chocolate_division_l1783_178354

/-- Represents the number of chocolate pieces Maria has after a given number of days -/
def chocolatePieces (days : ℕ) : ℕ :=
  9 + 8 * days

theorem chocolate_division :
  (chocolatePieces 3 = 25) ∧
  (∀ n : ℕ, chocolatePieces n ≠ 2014) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l1783_178354


namespace NUMINAMATH_CALUDE_negative_one_in_M_l1783_178328

def M : Set ℝ := {x | x^2 - 1 = 0}

theorem negative_one_in_M : (-1 : ℝ) ∈ M := by sorry

end NUMINAMATH_CALUDE_negative_one_in_M_l1783_178328


namespace NUMINAMATH_CALUDE_euro_equation_solution_l1783_178321

def euro (x y : ℝ) : ℝ := 2 * x * y

theorem euro_equation_solution :
  ∀ x : ℝ, euro 6 (euro 4 x) = 480 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_euro_equation_solution_l1783_178321


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l1783_178331

/-- Represents a round-robin tournament --/
structure Tournament where
  n : ℕ  -- number of teams
  games : Fin n → Fin n → Bool
  -- games i j is true if team i wins against team j
  irreflexive : ∀ i, games i i = false
  asymmetric : ∀ i j, games i j = !games j i

/-- The number of wins for a team in a tournament --/
def wins (t : Tournament) (i : Fin t.n) : ℕ :=
  (Finset.univ.filter (λ j => t.games i j)).card

/-- The maximum number of wins in a tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.univ.sup (wins t)

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  (Finset.univ.filter (λ i => wins t i = max_wins t)).card

theorem max_teams_tied_for_most_wins :
  ∃ t : Tournament, t.n = 8 ∧ num_teams_with_max_wins t = 7 ∧
  ∀ t' : Tournament, t'.n = 8 → num_teams_with_max_wins t' ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l1783_178331


namespace NUMINAMATH_CALUDE_odd_nines_composite_l1783_178382

theorem odd_nines_composite (k : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^(2*k) - 9 = a * b :=
sorry

end NUMINAMATH_CALUDE_odd_nines_composite_l1783_178382


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1783_178385

def initial_amount : ℕ := 85

def book_prices : List ℕ := [4, 6, 3, 7, 5, 8, 2, 6, 3, 5, 7, 4, 5, 6, 3]

theorem money_left_after_purchase : 
  initial_amount - (book_prices.sum) = 11 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1783_178385


namespace NUMINAMATH_CALUDE_snowball_distance_l1783_178342

/-- The sum of an arithmetic sequence with first term 6, common difference 5, and 25 terms -/
def arithmetic_sum (first_term : ℕ) (common_diff : ℕ) (num_terms : ℕ) : ℕ :=
  (num_terms * (2 * first_term + (num_terms - 1) * common_diff)) / 2

/-- Theorem stating that the sum of the specific arithmetic sequence is 1650 -/
theorem snowball_distance : arithmetic_sum 6 5 25 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_snowball_distance_l1783_178342


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1783_178313

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 12) % 8 = 0 ∧ (n - 12) % 12 = 0 ∧ (n - 12) % 22 = 0 ∧ (n - 12) % 24 = 0

theorem smallest_number_divisible_by_all : 
  (is_divisible_by_all 252) ∧ 
  (∀ m : ℕ, m < 252 → ¬(is_divisible_by_all m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1783_178313


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l1783_178378

theorem inverse_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  (3*i - 2*i⁻¹)⁻¹ = -i/5 := by sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l1783_178378


namespace NUMINAMATH_CALUDE_paris_time_correct_l1783_178390

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24
  mValid : minutes < 60

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a datetime with date and time -/
structure DateTime where
  date : Date
  time : Time

def time_difference : ℤ := -7

def beijing_time : DateTime := {
  date := { year := 2023, month := 10, day := 26 },
  time := { hours := 5, minutes := 0, hValid := by sorry, mValid := by sorry }
}

/-- Calculates the Paris time given the Beijing time and time difference -/
def calculate_paris_time (beijing : DateTime) (diff : ℤ) : DateTime :=
  sorry

theorem paris_time_correct :
  let paris_time := calculate_paris_time beijing_time time_difference
  paris_time.date.day = 25 ∧
  paris_time.date.month = 10 ∧
  paris_time.time.hours = 22 ∧
  paris_time.time.minutes = 0 :=
by sorry

end NUMINAMATH_CALUDE_paris_time_correct_l1783_178390


namespace NUMINAMATH_CALUDE_shirt_price_change_l1783_178338

theorem shirt_price_change (original_price : ℝ) (decrease_percent : ℝ) : 
  original_price > 0 →
  decrease_percent ≥ 0 →
  (1.15 * original_price) * (1 - decrease_percent / 100) = 97.75 →
  decrease_percent = 0 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_change_l1783_178338


namespace NUMINAMATH_CALUDE_odd_sum_representation_l1783_178383

theorem odd_sum_representation (a b : ℤ) (h : Odd (a + b)) :
  ∀ n : ℤ, ∃ x y : ℤ, n = x^2 - y^2 + a*x + b*y :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_representation_l1783_178383


namespace NUMINAMATH_CALUDE_maxwells_walking_speed_l1783_178381

/-- Proves that Maxwell's walking speed is 24 km/h given the problem conditions -/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (brads_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 72) 
  (h2 : brads_speed = 12) 
  (h3 : maxwell_distance = 24) : 
  maxwell_distance / (maxwell_distance / brads_speed) = 24 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_walking_speed_l1783_178381


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1783_178304

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The theorem to be proved -/
theorem parabola_shift_theorem :
  let original := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal original 3
  let final := shift_vertical shifted_left (-1)
  final = Parabola.mk (-2) 12 (-19) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1783_178304


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_sum_products_l1783_178360

theorem cube_sum_geq_triple_sum_products
  (a b c : ℝ)
  (ha : a ≥ 0)
  (hb : b ≥ 0)
  (hc : c ≥ 0)
  (h_sum_squares : a^2 + b^2 + c^2 ≥ 3) :
  (a + b + c)^3 ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_sum_products_l1783_178360


namespace NUMINAMATH_CALUDE_solve_for_m_l1783_178301

theorem solve_for_m (x : ℝ) (m : ℝ) : 
  (-3 * x = -5 * x + 4) → 
  (m^x - 9 = 0) → 
  (m = 3 ∨ m = -3) := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l1783_178301


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1783_178344

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the problem
theorem triangle_angle_proof (t : Triangle) (m n : ℝ × ℝ) :
  m = (t.a + t.c, -t.b) →
  n = (t.a - t.c, t.b) →
  m.1 * n.1 + m.2 * n.2 = t.b * t.c →
  0 < t.A →
  t.A < π →
  t.A = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1783_178344


namespace NUMINAMATH_CALUDE_father_total_spending_l1783_178365

def heaven_spending : ℕ := 2 * 5 + 4 * 5
def brother_eraser_spending : ℕ := 10 * 4
def brother_highlighter_spending : ℕ := 30

theorem father_total_spending :
  heaven_spending + brother_eraser_spending + brother_highlighter_spending = 100 := by
  sorry

end NUMINAMATH_CALUDE_father_total_spending_l1783_178365


namespace NUMINAMATH_CALUDE_average_problem_l1783_178349

theorem average_problem (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1783_178349


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1783_178355

theorem a_plus_b_value (a b : ℝ) : 
  ((a + 2)^2 = 1 ∧ 3^3 = b - 3) → (a + b = 29 ∨ a + b = 27) := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1783_178355


namespace NUMINAMATH_CALUDE_ac_in_open_interval_sum_of_endpoints_l1783_178392

/-- Represents a triangle ABC with an angle bisector from A to D on BC -/
structure AngleBisectorTriangle where
  -- The length of side AB
  ab : ℝ
  -- The length of CD (part of BC)
  cd : ℝ
  -- The length of AC
  ac : ℝ
  -- Assumption that AB = 15
  ab_eq : ab = 15
  -- Assumption that CD = 5
  cd_eq : cd = 5
  -- Assumption that AC is positive
  ac_pos : ac > 0
  -- Assumption that ABC forms a valid triangle
  triangle_inequality : ac + cd + (75 / ac) > ab ∧ ab + cd + (75 / ac) > ac ∧ ab + ac > cd + (75 / ac)
  -- Assumption that AD is the angle bisector
  angle_bisector : ab / ac = (75 / ac) / cd

/-- The main theorem stating that AC must be in the open interval (5, 25) -/
theorem ac_in_open_interval (t : AngleBisectorTriangle) : 5 < t.ac ∧ t.ac < 25 := by
  sorry

/-- The sum of the endpoints of the interval is 30 -/
theorem sum_of_endpoints : 5 + 25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ac_in_open_interval_sum_of_endpoints_l1783_178392


namespace NUMINAMATH_CALUDE_balloon_count_l1783_178339

theorem balloon_count (initial_balloons : Real) (friend_balloons : Real) 
  (h1 : initial_balloons = 7.0) 
  (h2 : friend_balloons = 5.0) : 
  initial_balloons + friend_balloons = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l1783_178339


namespace NUMINAMATH_CALUDE_quadratic_roots_real_l1783_178379

theorem quadratic_roots_real (a b c : ℝ) : 
  let discriminant := 4 * (b^2 + c^2)
  discriminant ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_l1783_178379
