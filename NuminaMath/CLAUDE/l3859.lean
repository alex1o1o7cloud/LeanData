import Mathlib

namespace NUMINAMATH_CALUDE_t_value_proof_l3859_385913

theorem t_value_proof :
  let t := 1 / (1 - Real.rpow 2 (1/4))
  t = -(1 + Real.rpow 2 (1/4)) * (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_t_value_proof_l3859_385913


namespace NUMINAMATH_CALUDE_max_apartments_five_by_five_l3859_385923

/-- Represents a building with a given number of floors and windows per floor. -/
structure Building where
  floors : ℕ
  windowsPerFloor : ℕ

/-- Calculates the maximum number of apartments in a building. -/
def maxApartments (b : Building) : ℕ :=
  b.floors * b.windowsPerFloor

/-- Theorem stating that for a 5-story building with 5 windows per floor,
    the maximum number of apartments is 25. -/
theorem max_apartments_five_by_five :
  ∀ (b : Building),
    b.floors = 5 →
    b.windowsPerFloor = 5 →
    maxApartments b = 25 := by
  sorry

#check max_apartments_five_by_five

end NUMINAMATH_CALUDE_max_apartments_five_by_five_l3859_385923


namespace NUMINAMATH_CALUDE_unique_albums_count_l3859_385936

/-- Represents a music collection -/
structure MusicCollection where
  total : ℕ
  shared : ℕ
  unique : ℕ

/-- Theorem about the number of unique albums in two collections -/
theorem unique_albums_count
  (andrew : MusicCollection)
  (john : MusicCollection)
  (h1 : andrew.total = 23)
  (h2 : andrew.shared = 11)
  (h3 : john.shared = 11)
  (h4 : john.unique = 8)
  : andrew.unique + john.unique = 20 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l3859_385936


namespace NUMINAMATH_CALUDE_four_birds_joined_l3859_385948

/-- The number of birds that joined the fence -/
def birds_joined (initial_birds final_birds : ℕ) : ℕ :=
  final_birds - initial_birds

/-- Proof that 4 birds joined the fence -/
theorem four_birds_joined :
  let initial_birds : ℕ := 2
  let final_birds : ℕ := 6
  birds_joined initial_birds final_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_birds_joined_l3859_385948


namespace NUMINAMATH_CALUDE_function_properties_l3859_385970

noncomputable def f (m n x : ℝ) : ℝ := m * x + n / x

theorem function_properties (m n : ℝ) :
  (∃ a, f m n 1 = a ∧ 3 + a - 8 = 0) →
  (m = 1 ∧ n = 4) ∧
  (∀ x, x < -2 → (deriv (f m n)) x > 0) ∧
  (∀ x, -2 < x → x < 0 → (deriv (f m n)) x < 0) ∧
  (∀ x, 0 < x → x < 2 → (deriv (f m n)) x < 0) ∧
  (∀ x, 2 < x → (deriv (f m n)) x > 0) ∧
  (∀ x, x ≠ 0 → (deriv (f m n)) x < 1) ∧
  (∀ α, (0 ≤ α ∧ α < π/4) ∨ (π/2 < α ∧ α < π) ↔ 
    ∃ x, x ≠ 0 ∧ Real.tan α = (deriv (f m n)) x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3859_385970


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3859_385996

theorem angle_sum_theorem (x₁ x₂ : Real) (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi) 
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi) 
  (eq₁ : Real.sin x₁ ^ 3 - Real.cos x₁ ^ 3 = (1 / Real.cos x₁) - (1 / Real.sin x₁))
  (eq₂ : Real.sin x₂ ^ 3 - Real.cos x₂ ^ 3 = (1 / Real.cos x₂) - (1 / Real.sin x₂)) :
  x₁ + x₂ = 3 * Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3859_385996


namespace NUMINAMATH_CALUDE_value_of_a_l3859_385998

theorem value_of_a (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (hab : a * b = 2)
  (hbc : b * c = 3)
  (hcd : c * d = 4)
  (hde : d * e = 15)
  (hea : e * a = 10) :
  a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3859_385998


namespace NUMINAMATH_CALUDE_triangle_properties_l3859_385975

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties and maximum area of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 4 * Real.cos t.C + Real.cos (2 * t.C) = 4 * Real.cos t.C * (Real.cos (t.C / 2))^2)
  (h2 : |t.b * Real.cos t.A - (1/2) * t.a * Real.cos t.B| = 2) : 
  t.C = π/3 ∧ 
  (∃ (S : ℝ), S ≤ 2 * Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3859_385975


namespace NUMINAMATH_CALUDE_determinant_max_value_l3859_385946

theorem determinant_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b - 1 ≥ x * y - 1) →
  a * b - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_determinant_max_value_l3859_385946


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_l3859_385986

/-- Parabola represented by parametric equations x = 4t² and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Line with slope 1 passing through a point -/
structure Line where
  slope : ℝ := 1
  point : ℝ × ℝ

/-- Represents the intersection points of the line and the parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The focus of a parabola with equation y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (i : Intersection) :
  l.point = focus → 
  (∃ t₁ t₂ : ℝ, 
    i.A = (4 * t₁^2, 4 * t₁) ∧ 
    i.B = (4 * t₂^2, 4 * t₂) ∧ 
    i.A.2 = l.slope * i.A.1 + (l.point.2 - l.slope * l.point.1) ∧
    i.B.2 = l.slope * i.B.1 + (l.point.2 - l.slope * l.point.1)) →
  Real.sqrt ((i.A.1 - i.B.1)^2 + (i.A.2 - i.B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_l3859_385986


namespace NUMINAMATH_CALUDE_existence_of_six_snakes_l3859_385967

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A snake is a polyline with 5 segments connecting 6 points -/
structure Snake where
  points : Fin 6 → Point
  is_valid : Bool

/-- Check if two snakes are different -/
def are_different_snakes (s1 s2 : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the angle condition -/
def satisfies_angle_condition (s : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the half-plane condition -/
def satisfies_half_plane_condition (s : Snake) : Bool :=
  sorry

/-- The main theorem stating that a configuration of 6 points exists
    that can form 6 different valid snakes -/
theorem existence_of_six_snakes :
  ∃ (points : Fin 6 → Point),
    ∃ (snakes : Fin 6 → Snake),
      (∀ i : Fin 6, (snakes i).points = points) ∧
      (∀ i : Fin 6, (snakes i).is_valid) ∧
      (∀ i j : Fin 6, i ≠ j → are_different_snakes (snakes i) (snakes j)) ∧
      (∀ i : Fin 6, satisfies_angle_condition (snakes i)) ∧
      (∀ i : Fin 6, satisfies_half_plane_condition (snakes i)) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_six_snakes_l3859_385967


namespace NUMINAMATH_CALUDE_root_sum_squares_l3859_385966

theorem root_sum_squares (a b c d : ℝ) : 
  (a^4 - 12*a^3 + 47*a^2 - 60*a + 24 = 0) →
  (b^4 - 12*b^3 + 47*b^2 - 60*b + 24 = 0) →
  (c^4 - 12*c^3 + 47*c^2 - 60*c + 24 = 0) →
  (d^4 - 12*d^3 + 47*d^2 - 60*d + 24 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 147 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3859_385966


namespace NUMINAMATH_CALUDE_total_donation_l3859_385910

def charity_donation (cassandra james stephanie alex : ℕ) : Prop :=
  cassandra = 5000 ∧
  james = cassandra - 276 ∧
  stephanie = 2 * james ∧
  alex = (3 * (cassandra + stephanie)) / 4 ∧
  cassandra + james + stephanie + alex = 31008

theorem total_donation :
  ∃ (cassandra james stephanie alex : ℕ),
    charity_donation cassandra james stephanie alex :=
by
  sorry

end NUMINAMATH_CALUDE_total_donation_l3859_385910


namespace NUMINAMATH_CALUDE_brothers_selection_probability_l3859_385925

theorem brothers_selection_probability
  (prob_X_initial : ℚ) (prob_Y_initial : ℚ)
  (prob_X_interview : ℚ) (prob_X_test : ℚ)
  (prob_Y_interview : ℚ) (prob_Y_test : ℚ)
  (h1 : prob_X_initial = 1 / 7)
  (h2 : prob_Y_initial = 2 / 5)
  (h3 : prob_X_interview = 3 / 4)
  (h4 : prob_X_test = 4 / 9)
  (h5 : prob_Y_interview = 5 / 8)
  (h6 : prob_Y_test = 7 / 10) :
  prob_X_initial * prob_X_interview * prob_X_test *
  prob_Y_initial * prob_Y_interview * prob_Y_test = 7 / 840 := by
  sorry

end NUMINAMATH_CALUDE_brothers_selection_probability_l3859_385925


namespace NUMINAMATH_CALUDE_variety_show_theorem_l3859_385920

/-- Represents the number of acts in the variety show -/
def num_acts : ℕ := 7

/-- Represents the number of acts with adjacency restrictions -/
def num_restricted_acts : ℕ := 3

/-- Represents the number of acts without adjacency restrictions -/
def num_unrestricted_acts : ℕ := num_acts - num_restricted_acts

/-- Represents the number of spaces available for restricted acts -/
def num_spaces : ℕ := num_unrestricted_acts + 1

/-- The number of ways to arrange the variety show program -/
def variety_show_arrangements : ℕ :=
  (num_spaces.choose num_restricted_acts) * 
  (Nat.factorial num_restricted_acts) * 
  (Nat.factorial num_unrestricted_acts)

theorem variety_show_theorem : 
  variety_show_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_theorem_l3859_385920


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l3859_385953

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv f x = 1 + Real.log x :=
sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l3859_385953


namespace NUMINAMATH_CALUDE_freshman_sample_size_l3859_385972

/-- Calculates the number of students to be sampled from a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation sampleSize stratumSize : ℕ) : ℕ :=
  (stratumSize * sampleSize) / totalPopulation

/-- The number of students to be sampled from the freshman year in a stratified sampling -/
theorem freshman_sample_size :
  let totalPopulation : ℕ := 4500
  let sampleSize : ℕ := 150
  let freshmanSize : ℕ := 1200
  stratifiedSampleSize totalPopulation sampleSize freshmanSize = 40 := by
sorry

#eval stratifiedSampleSize 4500 150 1200

end NUMINAMATH_CALUDE_freshman_sample_size_l3859_385972


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3859_385973

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3859_385973


namespace NUMINAMATH_CALUDE_investment_problem_l3859_385939

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem : 
  let principal : ℝ := 4000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 4840.000000000001 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l3859_385939


namespace NUMINAMATH_CALUDE_davantes_boy_friends_l3859_385900

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Davante's total number of friends
def total_friends : ℕ := 2 * days_in_week

-- Define the number of Davante's friends who are girls
def girl_friends : ℕ := 3

-- Theorem statement
theorem davantes_boy_friends :
  total_friends - girl_friends = 11 :=
sorry

end NUMINAMATH_CALUDE_davantes_boy_friends_l3859_385900


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3859_385937

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 6 93 = 45 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3859_385937


namespace NUMINAMATH_CALUDE_min_dot_product_planar_vectors_l3859_385984

/-- Given planar vectors a and b satisfying |2a - b| ≤ 3, 
    the minimum value of a · b is -9/8 -/
theorem min_dot_product_planar_vectors 
  (a b : ℝ × ℝ) 
  (h : ‖(2 : ℝ) • a - b‖ ≤ 3) : 
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (x : ℝ), x = a.1 * b.1 + a.2 * b.2 → m ≤ x :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_planar_vectors_l3859_385984


namespace NUMINAMATH_CALUDE_expression_value_l3859_385915

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  (x^5 + 3*y^3) / 9 = 141 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3859_385915


namespace NUMINAMATH_CALUDE_product_of_sums_equals_x_l3859_385916

theorem product_of_sums_equals_x : ∃ X : ℕ,
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_x_l3859_385916


namespace NUMINAMATH_CALUDE_band_and_chorus_not_orchestra_l3859_385942

/-- Represents the number of students in different musical groups at Liberty High School -/
structure MusicGroups where
  total : ℕ
  band : ℕ
  chorus : ℕ
  orchestra : ℕ
  inAnyGroup : ℕ

/-- Calculates the number of students in both band and chorus but not in orchestra -/
def studentsInBandAndChorusNotOrchestra (g : MusicGroups) : ℕ :=
  g.band + g.chorus + g.orchestra - g.inAnyGroup - g.orchestra

/-- Theorem stating the number of students in both band and chorus but not in orchestra -/
theorem band_and_chorus_not_orchestra (g : MusicGroups) 
    (h1 : g.total = 250)
    (h2 : g.band = 80)
    (h3 : g.chorus = 110)
    (h4 : g.orchestra = 60)
    (h5 : g.inAnyGroup = 190) :
    studentsInBandAndChorusNotOrchestra g = 30 := by
  sorry

end NUMINAMATH_CALUDE_band_and_chorus_not_orchestra_l3859_385942


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3859_385995

theorem trigonometric_simplification (θ : ℝ) : 
  (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) + 
  (1 - Real.cos θ + Real.sin θ) / (1 + Real.cos θ + Real.sin θ) = 
  2 * (Real.sin θ)⁻¹ := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3859_385995


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3859_385907

theorem negation_of_universal_proposition :
  ¬(∀ n : ℤ, n % 5 = 0 → Odd n) ↔ ∃ n : ℤ, n % 5 = 0 ∧ ¬(Odd n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3859_385907


namespace NUMINAMATH_CALUDE_darcy_shirts_count_darcy_shirts_proof_l3859_385941

theorem darcy_shirts_count : ℕ :=
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11

  -- Define a function to calculate the total number of clothing items
  let total_clothing (shirts : ℕ) : ℕ := shirts + total_shorts

  -- Define a function to calculate the number of folded items
  let folded_items : ℕ := folded_shirts + folded_shorts

  -- The number of shirts that satisfies the conditions
  20

theorem darcy_shirts_proof (shirts : ℕ) : 
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11
  let total_clothing := shirts + total_shorts
  let folded_items := folded_shirts + folded_shorts

  shirts = 20 ↔ 
    total_clothing - folded_items = remaining_to_fold :=
by sorry

end NUMINAMATH_CALUDE_darcy_shirts_count_darcy_shirts_proof_l3859_385941


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3859_385988

/-- Given two vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  b = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3859_385988


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_three_l3859_385997

theorem greatest_two_digit_multiple_of_three : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0 → n ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_three_l3859_385997


namespace NUMINAMATH_CALUDE_charles_housesitting_hours_l3859_385934

/-- Proves that Charles housesat for 10 hours given the conditions of the problem -/
theorem charles_housesitting_hours : 
  let housesitting_rate : ℝ := 15
  let dog_walking_rate : ℝ := 22
  let num_dogs : ℕ := 3
  let total_earnings : ℝ := 216
  ∃ (h : ℝ), h * housesitting_rate + (num_dogs : ℝ) * dog_walking_rate = total_earnings ∧ h = 10 := by
  sorry

end NUMINAMATH_CALUDE_charles_housesitting_hours_l3859_385934


namespace NUMINAMATH_CALUDE_enumeration_pattern_correct_l3859_385969

/-- Represents the number in a square of the enumerated grid -/
def square_number (m n : ℕ) : ℕ := Nat.choose (m + n - 1) 2 + n

/-- The enumeration pattern for the squared paper -/
def enumeration_pattern : ℕ → ℕ → ℕ := square_number

theorem enumeration_pattern_correct :
  ∀ (m n : ℕ), enumeration_pattern m n = square_number m n :=
by sorry

end NUMINAMATH_CALUDE_enumeration_pattern_correct_l3859_385969


namespace NUMINAMATH_CALUDE_inequality_proof_l3859_385901

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3859_385901


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3859_385943

theorem point_in_third_quadrant (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) :
  m < 0 ∧ n < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3859_385943


namespace NUMINAMATH_CALUDE_crab_meat_cost_per_pound_l3859_385950

/-- The cost of crab meat per pound given Johnny's crab dish production and expenses -/
theorem crab_meat_cost_per_pound 
  (dishes_per_day : ℕ) 
  (meat_per_dish : ℚ) 
  (weekly_expense : ℕ) 
  (closed_days : ℕ) : 
  dishes_per_day = 40 → 
  meat_per_dish = 3/2 → 
  weekly_expense = 1920 → 
  closed_days = 3 → 
  (weekly_expense : ℚ) / ((7 - closed_days) * dishes_per_day * meat_per_dish) = 8 := by
  sorry

end NUMINAMATH_CALUDE_crab_meat_cost_per_pound_l3859_385950


namespace NUMINAMATH_CALUDE_arithmetic_reciprocal_sequence_l3859_385947

theorem arithmetic_reciprocal_sequence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (harith : ∃ d ≠ 0, b = a + d ∧ c = a + 2*d) :
  ¬(∃ r ≠ 0, (1/b - 1/a) = r ∧ (1/c - 1/b) = r) ∧
  ¬(∃ q ≠ 1, (1/b) / (1/a) = q ∧ (1/c) / (1/b) = q) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_reciprocal_sequence_l3859_385947


namespace NUMINAMATH_CALUDE_max_original_points_l3859_385931

/-- Represents a rectangular matrix of points on a grid -/
structure RectMatrix where
  rows : ℕ
  cols : ℕ

/-- The maximum grid size -/
def maxGridSize : ℕ := 19

/-- The number of additional points -/
def additionalPoints : ℕ := 45

/-- Checks if a rectangular matrix fits within the maximum grid size -/
def fitsInGrid (rect : RectMatrix) : Prop :=
  rect.rows ≤ maxGridSize ∧ rect.cols ≤ maxGridSize

/-- Checks if a rectangular matrix can be expanded by adding the additional points -/
def canBeExpanded (small rect : RectMatrix) : Prop :=
  (rect.rows - small.rows) * (rect.cols - small.cols) = additionalPoints

/-- The theorem stating the maximum number of points in the original matrix -/
theorem max_original_points : 
  ∃ (small large : RectMatrix), 
    fitsInGrid small ∧ 
    fitsInGrid large ∧
    canBeExpanded small large ∧
    (small.rows = large.rows ∨ small.cols = large.cols) ∧
    small.rows * small.cols = 285 ∧
    ∀ (other : RectMatrix), 
      fitsInGrid other → 
      (∃ (expanded : RectMatrix), 
        fitsInGrid expanded ∧ 
        canBeExpanded other expanded ∧ 
        (other.rows = expanded.rows ∨ other.cols = expanded.cols)) →
      other.rows * other.cols ≤ 285 :=
sorry

end NUMINAMATH_CALUDE_max_original_points_l3859_385931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l3859_385926

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l3859_385926


namespace NUMINAMATH_CALUDE_novosibirsk_divisible_by_three_l3859_385968

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Nat

/-- Checks if a mapping is valid for the word "NOVOSIBIRSK" -/
def isValidMapping (m : LetterToDigitMap) : Prop :=
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'V' ∧ m 'N' ≠ m 'S' ∧ m 'N' ≠ m 'I' ∧ m 'N' ≠ m 'B' ∧ m 'N' ≠ m 'R' ∧ m 'N' ≠ m 'K' ∧
  m 'O' ≠ m 'V' ∧ m 'O' ≠ m 'S' ∧ m 'O' ≠ m 'I' ∧ m 'O' ≠ m 'B' ∧ m 'O' ≠ m 'R' ∧ m 'O' ≠ m 'K' ∧
  m 'V' ≠ m 'S' ∧ m 'V' ≠ m 'I' ∧ m 'V' ≠ m 'B' ∧ m 'V' ≠ m 'R' ∧ m 'V' ≠ m 'K' ∧
  m 'S' ≠ m 'I' ∧ m 'S' ≠ m 'B' ∧ m 'S' ≠ m 'R' ∧ m 'S' ≠ m 'K' ∧
  m 'I' ≠ m 'B' ∧ m 'I' ≠ m 'R' ∧ m 'I' ≠ m 'K' ∧
  m 'B' ≠ m 'R' ∧ m 'B' ≠ m 'K' ∧
  m 'R' ≠ m 'K'

/-- Calculates the sum of digits for "NOVOSIBIRSK" using the given mapping -/
def sumOfDigits (m : LetterToDigitMap) : Nat :=
  m 'N' + m 'O' + m 'V' + m 'O' + m 'S' + m 'I' + m 'B' + m 'I' + m 'R' + m 'S' + m 'K'

/-- Theorem: There exists a valid mapping for "NOVOSIBIRSK" that results in a number divisible by 3 -/
theorem novosibirsk_divisible_by_three : ∃ (m : LetterToDigitMap), isValidMapping m ∧ sumOfDigits m % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_novosibirsk_divisible_by_three_l3859_385968


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l3859_385993

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2012 + Real.sqrt 2013 →
  Q = -Real.sqrt 2012 - Real.sqrt 2013 →
  R = Real.sqrt 2012 - Real.sqrt 2013 →
  S = Real.sqrt 2013 - Real.sqrt 2012 →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l3859_385993


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l3859_385908

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l3859_385908


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3859_385959

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l3859_385959


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3859_385917

def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3859_385917


namespace NUMINAMATH_CALUDE_exponential_inequality_l3859_385962

theorem exponential_inequality (a x y : ℝ) (ha : 0 < a ∧ a < 1) (hxy : x > y) : a^x < a^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3859_385962


namespace NUMINAMATH_CALUDE_vegan_menu_fraction_l3859_385909

theorem vegan_menu_fraction (vegan_dishes : ℕ) (total_dishes : ℕ) (soy_dishes : ℕ) :
  vegan_dishes = 6 →
  vegan_dishes = total_dishes / 3 →
  soy_dishes = 4 →
  (vegan_dishes - soy_dishes : ℚ) / total_dishes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_vegan_menu_fraction_l3859_385909


namespace NUMINAMATH_CALUDE_pet_store_kittens_l3859_385921

/-- The total number of kittens after receiving more -/
def total_kittens (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: If a pet store initially has 6 kittens and receives 3 more, 
    the total number of kittens will be 9 -/
theorem pet_store_kittens : total_kittens 6 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l3859_385921


namespace NUMINAMATH_CALUDE_special_matrix_sum_l3859_385940

/-- Represents a 3x3 matrix with the given structure -/
structure SpecialMatrix where
  v : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  sum_equality : ℝ
  sum_row_1 : v + 50 + w = sum_equality
  sum_row_2 : 196 + x + y = sum_equality
  sum_row_3 : 269 + z + 123 = sum_equality
  sum_col_1 : v + 196 + 269 = sum_equality
  sum_col_2 : 50 + x + z = sum_equality
  sum_col_3 : w + y + 123 = sum_equality
  sum_diag_1 : v + x + 123 = sum_equality
  sum_diag_2 : w + x + 269 = sum_equality

/-- Theorem: In the SpecialMatrix, y + z = 196 -/
theorem special_matrix_sum (m : SpecialMatrix) : m.y + m.z = 196 := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_sum_l3859_385940


namespace NUMINAMATH_CALUDE_jump_rope_challenge_l3859_385957

structure Jumper where
  initialRate : ℝ
  breakPatterns : List (ℝ × ℝ)
  speedChanges : List ℝ

def calculateSkips (j : Jumper) (totalTime : ℝ) : ℝ :=
  sorry

theorem jump_rope_challenge :
  let leah : Jumper := {
    initialRate := 5,
    breakPatterns := [(120, 20), (120, 25), (120, 30)],
    speedChanges := [0.5, 0.5, 0.5]
  }
  let matt : Jumper := {
    initialRate := 3,
    breakPatterns := [(180, 15), (180, 15)],
    speedChanges := [-0.25, -0.25]
  }
  let linda : Jumper := {
    initialRate := 4,
    breakPatterns := [(240, 10), (240, 15)],
    speedChanges := [-0.1, 0.2]
  }
  let totalTime : ℝ := 600
  (calculateSkips leah totalTime = 3540) ∧
  (calculateSkips matt totalTime = 1635) ∧
  (calculateSkips linda totalTime = 2412) ∧
  (calculateSkips leah totalTime + calculateSkips matt totalTime + calculateSkips linda totalTime = 7587) :=
by
  sorry

end NUMINAMATH_CALUDE_jump_rope_challenge_l3859_385957


namespace NUMINAMATH_CALUDE_difference_of_squares_l3859_385954

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3859_385954


namespace NUMINAMATH_CALUDE_infinitely_many_y_greater_than_sqrt_n_l3859_385981

theorem infinitely_many_y_greater_than_sqrt_n
  (x y : ℕ → ℕ+)
  (h : ∀ n : ℕ, n ≥ 1 → (y (n + 1) : ℚ) / (x (n + 1) : ℚ) > (y n : ℚ) / (x n : ℚ)) :
  Set.Infinite {n : ℕ | (y n : ℝ) > Real.sqrt n} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_y_greater_than_sqrt_n_l3859_385981


namespace NUMINAMATH_CALUDE_vector_magnitude_l3859_385927

/-- The magnitude of a 2D vector (1, 2) is √5 -/
theorem vector_magnitude : ∀ (a : ℝ × ℝ), a = (1, 2) → ‖a‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3859_385927


namespace NUMINAMATH_CALUDE_diana_charge_amount_l3859_385922

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem diana_charge_amount :
  ∃ (P : ℝ),
    (P > 0) ∧
    (P < 80.25) ∧
    (P + simple_interest P 0.07 1 = 80.25) ∧
    (abs (P - 75) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_diana_charge_amount_l3859_385922


namespace NUMINAMATH_CALUDE_biology_score_calculation_l3859_385965

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 62
def average_score : ℕ := 74
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 85 := by
sorry

end NUMINAMATH_CALUDE_biology_score_calculation_l3859_385965


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l3859_385930

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Proves that the cost to insulate a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l3859_385930


namespace NUMINAMATH_CALUDE_speed_increase_for_time_reduction_car_speed_increase_l3859_385951

/-- Calculates the required speed increase for a car to reduce its travel time --/
theorem speed_increase_for_time_reduction 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (time_reduction : ℝ) : ℝ :=
  let initial_time := distance / initial_speed
  let final_time := initial_time - time_reduction
  let final_speed := distance / final_time
  final_speed - initial_speed

/-- Proves that a car traveling at 60 km/h needs to increase its speed by 60 km/h
    to travel 1 km in half a minute less time --/
theorem car_speed_increase : 
  speed_increase_for_time_reduction 60 1 (1/120) = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_for_time_reduction_car_speed_increase_l3859_385951


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3859_385933

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l3859_385933


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_fixed_point_satisfies_equation_l3859_385976

/-- The fixed point on the graph of y = 9x^2 + kx - 5k -/
theorem fixed_point_quadratic (k : ℝ) : 
  9 * (-3)^2 + k * (-3) - 5 * k = 81 := by
  sorry

/-- The fixed point (-3, 81) satisfies the equation for all k -/
theorem fixed_point_satisfies_equation (k : ℝ) :
  ∃ (x y : ℝ), x = -3 ∧ y = 81 ∧ y = 9 * x^2 + k * x - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_fixed_point_satisfies_equation_l3859_385976


namespace NUMINAMATH_CALUDE_triangular_to_square_ratio_l3859_385963

/-- A polyhedron with only triangular and square faces -/
structure Polyhedron :=
  (triangular_faces : ℕ)
  (square_faces : ℕ)

/-- Property that no two faces of the same type share an edge -/
def no_same_type_edge_sharing (p : Polyhedron) : Prop :=
  ∀ (edge : ℕ), (∃! square_face : ℕ, square_face ≤ p.square_faces) ∧
                (∃! triangular_face : ℕ, triangular_face ≤ p.triangular_faces)

theorem triangular_to_square_ratio (p : Polyhedron) 
  (h : no_same_type_edge_sharing p) (h_pos : p.square_faces > 0) : 
  (p.triangular_faces : ℚ) / p.square_faces = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangular_to_square_ratio_l3859_385963


namespace NUMINAMATH_CALUDE_floor_equality_l3859_385992

theorem floor_equality (n : ℤ) (h : n > 2) :
  ⌊(n * (n + 1) : ℚ) / (4 * n - 2)⌋ = ⌊(n + 1 : ℚ) / 4⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l3859_385992


namespace NUMINAMATH_CALUDE_anushas_share_multiple_l3859_385964

/-- Proves that the multiple of Anusha's share is 12 given the problem conditions -/
theorem anushas_share_multiple (anusha babu esha : ℕ) (m : ℕ) : 
  anusha = 84 →
  m * anusha = 8 * babu →
  8 * babu = 6 * esha →
  anusha + babu + esha = 378 →
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_anushas_share_multiple_l3859_385964


namespace NUMINAMATH_CALUDE_angle_4_value_l3859_385994

theorem angle_4_value (angle_1 angle_2 angle_3 angle_4 angle_A angle_B : ℝ) :
  angle_1 + angle_2 = 180 →
  angle_3 = angle_4 →
  angle_3 = (1 / 2) * angle_4 →
  angle_A = 80 →
  angle_B = 50 →
  angle_4 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_4_value_l3859_385994


namespace NUMINAMATH_CALUDE_rickshaw_charge_calculation_l3859_385914

/-- Rickshaw charge calculation -/
theorem rickshaw_charge_calculation 
  (initial_charge : ℝ) 
  (additional_charge : ℝ) 
  (total_distance : ℝ) 
  (total_charge : ℝ) :
  initial_charge = 13.5 →
  additional_charge = 2.5 →
  total_distance = 13 →
  total_charge = 103.5 →
  initial_charge + additional_charge * (total_distance - 1) = total_charge :=
by sorry

end NUMINAMATH_CALUDE_rickshaw_charge_calculation_l3859_385914


namespace NUMINAMATH_CALUDE_k_range_for_positive_f_l3859_385979

/-- Given a function f(x) = 32x - (k + 1)3^x + 2 that is always positive for real x,
    prove that k is in the range (-∞, 2^(-1)). -/
theorem k_range_for_positive_f (k : ℝ) :
  (∀ x : ℝ, 32 * x - (k + 1) * 3^x + 2 > 0) →
  k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_positive_f_l3859_385979


namespace NUMINAMATH_CALUDE_water_tank_problem_l3859_385991

/-- A water tank problem during the rainy season -/
theorem water_tank_problem (tank_capacity : ℝ) (initial_fill_fraction : ℝ) 
  (day1_collection : ℝ) (day2_extra : ℝ) :
  tank_capacity = 100 →
  initial_fill_fraction = 2/5 →
  day1_collection = 15 →
  day2_extra = 5 →
  let initial_water := initial_fill_fraction * tank_capacity
  let day1_total := initial_water + day1_collection
  let day2_collection := day1_collection + day2_extra
  let day2_total := day1_total + day2_collection
  let day3_collection := tank_capacity - day2_total
  day3_collection = 25 := by
sorry


end NUMINAMATH_CALUDE_water_tank_problem_l3859_385991


namespace NUMINAMATH_CALUDE_unique_invariant_quadratic_l3859_385932

/-- A quadratic equation that remains unchanged when its roots are used as coefficients. -/
def InvariantQuadratic (p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧
  ∃ (x y : ℝ),
    x^2 + p*x + q = 0 ∧
    y^2 + p*y + q = 0 ∧
    x ≠ y ∧
    x^2 + y*x + (x*y) = 0 ∧
    p = -(x + y) ∧
    q = x * y

theorem unique_invariant_quadratic :
  ∀ (p q : ℤ), InvariantQuadratic p q → p = 1 ∧ q = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_invariant_quadratic_l3859_385932


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3859_385906

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 7*x + 1) * (x^2 + 3*x + 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3859_385906


namespace NUMINAMATH_CALUDE_middle_circle_radius_l3859_385949

/-- Configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- Radius of the smallest circle -/
  r_min : ℝ
  /-- Radius of the largest circle -/
  r_max : ℝ
  /-- Radius of the middle circle -/
  r_mid : ℝ

/-- The theorem stating the relationship between the radii of the circles -/
theorem middle_circle_radius (c : CircleConfiguration) 
  (h_min : c.r_min = 12)
  (h_max : c.r_max = 24) :
  c.r_mid = 12 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_middle_circle_radius_l3859_385949


namespace NUMINAMATH_CALUDE_building_height_l3859_385902

/-- Given a building and a pole casting shadows, calculates the height of the building. -/
theorem building_height (building_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  building_shadow = 63 →
  pole_height = 14 →
  pole_shadow = 18 →
  (building_shadow / pole_shadow) * pole_height = 49 := by
  sorry

#check building_height

end NUMINAMATH_CALUDE_building_height_l3859_385902


namespace NUMINAMATH_CALUDE_servant_service_duration_l3859_385955

def yearly_payment : ℕ := 800
def uniform_price : ℕ := 300
def actual_payment : ℕ := 600

def months_served : ℕ := 7

theorem servant_service_duration :
  yearly_payment = 800 ∧
  uniform_price = 300 ∧
  actual_payment = 600 →
  months_served = 7 :=
by sorry

end NUMINAMATH_CALUDE_servant_service_duration_l3859_385955


namespace NUMINAMATH_CALUDE_pencil_length_after_sharpening_l3859_385956

/-- Calculates the final length of a pencil after sharpening on four consecutive days. -/
def final_pencil_length (initial_length : ℕ) (day1 day2 day3 day4 : ℕ) : ℕ :=
  initial_length - (day1 + day2 + day3 + day4)

/-- Theorem stating that given specific initial length and sharpening amounts, the final pencil length is 36 inches. -/
theorem pencil_length_after_sharpening :
  final_pencil_length 50 2 3 4 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_after_sharpening_l3859_385956


namespace NUMINAMATH_CALUDE_library_books_count_l3859_385928

theorem library_books_count : ∃ (n : ℕ), 
  500 < n ∧ n < 650 ∧ 
  ∃ (r : ℕ), n = 12 * r + 7 ∧
  ∃ (l : ℕ), n = 25 * l - 5 ∧
  n = 595 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l3859_385928


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l3859_385912

/-- Sequence c_n defined recursively -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c n - 4 * c (n + 1) + 2008

/-- Sequence a_n defined in terms of c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : 
  ∃ k : ℤ, a n = k^2 := by sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l3859_385912


namespace NUMINAMATH_CALUDE_digit_equality_l3859_385974

theorem digit_equality (a b c d e f : ℕ) 
  (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) 
  (h_d : d < 10) (h_e : e < 10) (h_f : f < 10) :
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) -
  (100000 * f + 10000 * d + 1000 * e + 100 * b + 10 * c + a) ∣ 271 →
  b = d ∧ c = e := by
sorry

end NUMINAMATH_CALUDE_digit_equality_l3859_385974


namespace NUMINAMATH_CALUDE_egg_difference_l3859_385989

/-- The number of eggs needed for one chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for one cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- The number of chocolate cakes -/
def num_chocolate_cakes : ℕ := 5

/-- The number of cheesecakes -/
def num_cheesecakes : ℕ := 9

/-- Theorem: The difference in eggs needed for 9 cheesecakes and 5 chocolate cakes is 57 -/
theorem egg_difference : 
  num_cheesecakes * cheesecake_eggs - num_chocolate_cakes * chocolate_cake_eggs = 57 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_l3859_385989


namespace NUMINAMATH_CALUDE_cows_bought_man_bought_43_cows_l3859_385961

/-- The number of cows a man bought in a year, given various events --/
theorem cows_bought (initial : ℕ) (died sold increased gift current : ℕ) : ℕ :=
  let after_last_year := initial - died - sold
  let after_increase := after_last_year + increased
  let after_gift := after_increase + gift
  current - after_gift

/-- The specific problem instance --/
theorem man_bought_43_cows : cows_bought 39 25 6 24 8 83 = 43 := by
  sorry

end NUMINAMATH_CALUDE_cows_bought_man_bought_43_cows_l3859_385961


namespace NUMINAMATH_CALUDE_total_worksheets_l3859_385929

theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  problems_per_worksheet = 4 →
  graded_worksheets = 8 →
  problems_left = 32 →
  graded_worksheets + (problems_left / problems_per_worksheet) = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_worksheets_l3859_385929


namespace NUMINAMATH_CALUDE_line_at_distance_iff_tangent_to_cylinder_l3859_385905

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  a : Point3D
  b : Point3D

/-- A cylinder in 3D space defined by its axis (a line) and radius -/
structure Cylinder where
  axis : Line3D
  radius : ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ := sorry

/-- Check if a line is tangent to a cylinder -/
def is_tangent_to_cylinder (l : Line3D) (c : Cylinder) : Prop := sorry

/-- Check if a line passes through a point -/
def line_passes_through_point (l : Line3D) (p : Point3D) : Prop := sorry

/-- Main theorem: A line passing through M is at distance d from AB iff it's tangent to the cylinder -/
theorem line_at_distance_iff_tangent_to_cylinder 
  (M : Point3D) (AB : Line3D) (d : ℝ) (l : Line3D) : 
  (line_passes_through_point l M ∧ distance_point_to_line M AB = d) ↔ 
  is_tangent_to_cylinder l (Cylinder.mk AB d) :=
sorry

end NUMINAMATH_CALUDE_line_at_distance_iff_tangent_to_cylinder_l3859_385905


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l3859_385944

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point C
def condition (C : ℝ × ℝ) : Prop :=
  let (x, y) := C
  (x - 3) * (x + 1) + y * y = 5

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x - y + 3 = 0

-- State the theorem
theorem trajectory_and_intersection :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ C, condition C ↔ (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2) ∧
    (center = (1, 0) ∧ radius = 3) ∧
    (∃ M N : ℝ × ℝ,
      M ≠ N ∧
      condition M ∧ condition N ∧
      line_l M ∧ line_l N ∧
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l3859_385944


namespace NUMINAMATH_CALUDE_no_rain_no_snow_probability_l3859_385980

theorem no_rain_no_snow_probability
  (rain_prob : ℚ)
  (snow_prob : ℚ)
  (rain_prob_def : rain_prob = 4 / 10)
  (snow_prob_def : snow_prob = 1 / 5)
  (events_independent : True) :
  (1 - rain_prob) * (1 - snow_prob) = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_no_snow_probability_l3859_385980


namespace NUMINAMATH_CALUDE_complex_point_location_l3859_385938

theorem complex_point_location (z : ℂ) : 
  (2 + Complex.I) * z = Complex.abs (1 - 2 * Complex.I) →
  Real.sign (z.re) > 0 ∧ Real.sign (z.im) < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l3859_385938


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3859_385958

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = (m^2 - m : ℝ) + (m^2 - 3*m + 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3859_385958


namespace NUMINAMATH_CALUDE_special_circle_properties_l3859_385903

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- The circle passes through these two points
  pointA : ℝ × ℝ := (1, 4)
  pointB : ℝ × ℝ := (3, 2)
  -- The center lies on this line
  centerLine : ℝ → ℝ := fun x => 3 - x

/-- The equation of the circle -/
def circleEquation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4

/-- A point on the circle -/
def pointOnCircle (c : SpecialCircle) (p : ℝ × ℝ) : Prop :=
  circleEquation c p.1 p.2

theorem special_circle_properties (c : SpecialCircle) :
  -- The circle equation is correct
  (∀ x y, circleEquation c x y ↔ pointOnCircle c (x, y)) ∧
  -- The maximum value of x+y for points on the circle
  (∃ max : ℝ, max = 3 + 2 * Real.sqrt 2 ∧
    ∀ p, pointOnCircle c p → p.1 + p.2 ≤ max) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_properties_l3859_385903


namespace NUMINAMATH_CALUDE_largest_common_term_up_to_150_l3859_385990

theorem largest_common_term_up_to_150 :
  ∀ k ∈ Finset.range 151,
    (∃ n : ℕ, k = 2 + 8 * n) ∧
    (∃ m : ℕ, k = 3 + 9 * m) →
    k ≤ 138 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_up_to_150_l3859_385990


namespace NUMINAMATH_CALUDE_system_solutions_l3859_385918

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₃

/-- The solutions to the system of equations -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ y : ℝ,
  system x₁ x₂ x₃ x₄ x₅ y →
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
   x₂ = y * x₁ ∧ x₃ = y * x₂ ∧ x₄ = y * x₃) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3859_385918


namespace NUMINAMATH_CALUDE_max_pages_proof_l3859_385977

/-- The cost in cents to copy 5 pages -/
def cost_per_5_pages : ℚ := 8

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1 / 10

/-- The available money in dollars -/
def available_money : ℚ := 30

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := 1687

theorem max_pages_proof :
  let discounted_money : ℚ := available_money * 100 * (1 - discount_rate)
  let pages_per_cent : ℚ := 5 / cost_per_5_pages
  ⌊discounted_money * pages_per_cent⌋ = max_pages := by
  sorry

end NUMINAMATH_CALUDE_max_pages_proof_l3859_385977


namespace NUMINAMATH_CALUDE_max_color_transitions_l3859_385919

/-- Represents a strategy for painting fence sections -/
def PaintingStrategy := Nat → Bool

/-- The number of fence sections -/
def numSections : Nat := 100

/-- Counts the number of color transitions in a given painting strategy -/
def countTransitions (strategy : PaintingStrategy) : Nat :=
  (List.range (numSections - 1)).filter (fun i => strategy i ≠ strategy (i + 1)) |>.length

/-- Theorem stating that the maximum number of guaranteed color transitions is 49 -/
theorem max_color_transitions :
  ∃ (strategy : PaintingStrategy),
    ∀ (otherStrategy : PaintingStrategy),
      countTransitions (fun i => if i % 2 = 0 then strategy (i / 2) else otherStrategy (i / 2)) ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_color_transitions_l3859_385919


namespace NUMINAMATH_CALUDE_intersection_distance_l3859_385952

theorem intersection_distance : ∃ (C D : ℝ × ℝ),
  (C.2 = 2 ∧ C.2 = 3 * C.1^2 + 2 * C.1 - 5) ∧
  (D.2 = 2 ∧ D.2 = 3 * D.1^2 + 2 * D.1 - 5) ∧
  C ≠ D ∧
  |C.1 - D.1| = 2 * Real.sqrt 22 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3859_385952


namespace NUMINAMATH_CALUDE_parabola_transformation_l3859_385978

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := λ x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := λ x => p.f x + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { f := λ x => 3 * x^2 }

/-- The final parabola after transformations -/
def final_parabola : Parabola :=
  { f := λ x => 3 * (x + 1)^2 - 2 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal original_parabola 1) (-2)).f = final_parabola.f :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3859_385978


namespace NUMINAMATH_CALUDE_prob_green_face_specific_cube_l3859_385985

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def prob_green_face (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 5 green faces and 1 yellow face is 5/6 -/
theorem prob_green_face_specific_cube :
  let cube : ColoredCube := ⟨6, 5, 1⟩
  prob_green_face cube = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_specific_cube_l3859_385985


namespace NUMINAMATH_CALUDE_sum_product_uniqueness_l3859_385924

theorem sum_product_uniqueness (S P : ℝ) (x y : ℝ) 
  (h_sum : x + y = S) (h_product : x * y = P) :
  (x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
  (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_uniqueness_l3859_385924


namespace NUMINAMATH_CALUDE_coin_equation_max_quarters_l3859_385911

/-- Represents the number of quarters (and nickels) -/
def q : ℕ := sorry

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 485

/-- The equation representing the total value of coins -/
theorem coin_equation : 25 * q + 5 * q + 20 * q = total_value := sorry

/-- The maximum number of quarters is 9 -/
theorem max_quarters : q ≤ 9 ∧ ∃ (n : ℕ), n = q ∧ 25 * n + 5 * n + 20 * n = total_value := by sorry

end NUMINAMATH_CALUDE_coin_equation_max_quarters_l3859_385911


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l3859_385982

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
    let rec aux (m : ℕ) : String :=
      if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
    aux n

theorem sum_of_binary_numbers :
  let a := binary_to_nat "1100"
  let b := binary_to_nat "101"
  let c := binary_to_nat "11"
  let d := binary_to_nat "11011"
  let e := binary_to_nat "100"
  nat_to_binary (a + b + c + d + e) = "1000101" := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l3859_385982


namespace NUMINAMATH_CALUDE_minimum_value_of_polynomial_l3859_385945

theorem minimum_value_of_polynomial (a b : ℝ) : 
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 ≥ 1947 ∧ 
  ∃ (a b : ℝ), 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 = 1947 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_polynomial_l3859_385945


namespace NUMINAMATH_CALUDE_modulo_31_problem_l3859_385987

theorem modulo_31_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 ≡ n [ZMOD 31] ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_modulo_31_problem_l3859_385987


namespace NUMINAMATH_CALUDE_school_boys_count_l3859_385999

/-- Proves that in a school with 48 total students and a boy-to-girl ratio of 7:1, the number of boys is 42 -/
theorem school_boys_count (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 48)
  (h2 : boys + girls = total_students)
  (h3 : boys = 7 * girls) : 
  boys = 42 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3859_385999


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3859_385983

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 3 * (s * Real.sqrt 2) → 4 * S / (4 * s) = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3859_385983


namespace NUMINAMATH_CALUDE_square_plot_side_length_l3859_385960

/-- Given a square plot with a lamp at one corner, prove that the side length is 21 m. -/
theorem square_plot_side_length (light_reach : ℝ) (lit_area : ℝ) : 
  light_reach = 21 → lit_area = 346.36 → light_reach = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_square_plot_side_length_l3859_385960


namespace NUMINAMATH_CALUDE_problem_solution_l3859_385935

theorem problem_solution (p q : ℝ) (h1 : p > 1) (h2 : q > 1)
  (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3859_385935


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_sixth_l3859_385971

theorem tan_theta_plus_pi_sixth (θ : Real) 
  (h1 : Real.sqrt 2 * Real.sin (θ - Real.pi/4) * Real.cos (Real.pi + θ) = Real.cos (2*θ))
  (h2 : Real.sin θ ≠ 0) : 
  Real.tan (θ + Real.pi/6) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_sixth_l3859_385971


namespace NUMINAMATH_CALUDE_other_sales_percentage_l3859_385904

/-- The Paper Boutique's sales percentages -/
structure SalesPercentages where
  pens : ℝ
  pencils : ℝ
  notebooks : ℝ
  total : ℝ
  pens_percent : pens = 25
  pencils_percent : pencils = 30
  notebooks_percent : notebooks = 20
  total_sum : total = 100

/-- Theorem: The percentage of sales that are neither pens, pencils, nor notebooks is 25% -/
theorem other_sales_percentage (s : SalesPercentages) : 
  s.total - (s.pens + s.pencils + s.notebooks) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l3859_385904
