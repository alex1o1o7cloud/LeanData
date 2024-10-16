import Mathlib

namespace NUMINAMATH_CALUDE_phi_bound_l3974_397487

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f x + 1

def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

def phi (f : ℝ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  iterate f n x - x

theorem phi_bound (f : ℝ → ℝ) (n : ℕ) :
  is_non_decreasing f →
  satisfies_functional_equation f →
  ∀ x y, |phi f n x - phi f n y| < 1 := by
  sorry

end NUMINAMATH_CALUDE_phi_bound_l3974_397487


namespace NUMINAMATH_CALUDE_f_neg_two_eq_five_l3974_397468

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem f_neg_two_eq_five
  (h1 : is_even (λ x => f x + x))
  (h2 : f 2 = 1) :
  f (-2) = 5 :=
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_five_l3974_397468


namespace NUMINAMATH_CALUDE_triangle_theorem_l3974_397416

noncomputable def triangle_problem (a b c A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  let m_x := m.1
  let m_y := m.2
  let n_x := n.1
  let n_y := n.2
  (m_x = Real.sin A ∧ m_y = 1) ∧ 
  (n_x = Real.cos A ∧ n_y = Real.sqrt 3) ∧
  (m_x / n_x = m_y / n_y) ∧  -- parallel vectors condition
  (a = 2) ∧ 
  (b = 2 * Real.sqrt 2) ∧
  (A = Real.pi / 6) ∧
  ((1 / 2 * a * b * Real.sin C = 1 + Real.sqrt 3) ∨ 
   (1 / 2 * a * b * Real.sin C = Real.sqrt 3 - 1))

theorem triangle_theorem (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  triangle_problem a b c A B C m n := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3974_397416


namespace NUMINAMATH_CALUDE_triangle_area_l3974_397414

theorem triangle_area (a b c : ℝ) (h1 : a = 17) (h2 : b = 144) (h3 : c = 145) :
  (1/2) * a * b = 1224 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3974_397414


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l3974_397437

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_recurrence : ∀ n, a (n + 1) = 2 * a n) : 
  ∀ n, a (n + 1) > a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l3974_397437


namespace NUMINAMATH_CALUDE_smallest_modulus_z_l3974_397484

theorem smallest_modulus_z (z : ℂ) (h : 3 * Complex.abs (z - 8) + 2 * Complex.abs (z - Complex.I * 7) = 26) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ 3 * Complex.abs (w - 8) + 2 * Complex.abs (w - Complex.I * 7) = 26 ∧ Complex.abs w = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_modulus_z_l3974_397484


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l3974_397463

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_to_nat (d : ℕ) : ℕ :=
  526000 + d * 100 + 84

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_3 (digit_to_nat d) ∧
  ∀ (d' : ℕ), d' < d → ¬is_divisible_by_3 (digit_to_nat d') :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l3974_397463


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l3974_397428

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l3974_397428


namespace NUMINAMATH_CALUDE_jack_evening_emails_l3974_397467

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The difference between morning and evening emails -/
def morning_evening_difference : ℕ := 2

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_emails - morning_evening_difference

theorem jack_evening_emails :
  evening_emails = 7 :=
sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l3974_397467


namespace NUMINAMATH_CALUDE_triangle_angle_b_sixty_degrees_l3974_397403

theorem triangle_angle_b_sixty_degrees 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (condition : c / (a + b) + a / (b + c) = 1) : 
  angle_b = π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_b_sixty_degrees_l3974_397403


namespace NUMINAMATH_CALUDE_unique_divisible_by_20_l3974_397499

def is_divisible_by_20 (n : ℕ) : Prop := ∃ k : ℕ, n = 20 * k

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 480 + x

theorem unique_divisible_by_20 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_20 (four_digit_number x) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_20_l3974_397499


namespace NUMINAMATH_CALUDE_ms_delmont_class_size_l3974_397483

/-- Proves the number of students in Ms. Delmont's class given the cupcake distribution -/
theorem ms_delmont_class_size 
  (total_cupcakes : ℕ)
  (adults_received : ℕ)
  (mrs_donnelly_class : ℕ)
  (leftover_cupcakes : ℕ)
  (h1 : total_cupcakes = 40)
  (h2 : adults_received = 4)
  (h3 : mrs_donnelly_class = 16)
  (h4 : leftover_cupcakes = 2) :
  total_cupcakes - adults_received - mrs_donnelly_class - leftover_cupcakes = 18 := by
  sorry

#check ms_delmont_class_size

end NUMINAMATH_CALUDE_ms_delmont_class_size_l3974_397483


namespace NUMINAMATH_CALUDE_midpoint_vector_sum_l3974_397453

-- Define the triangle ABC and its midpoints
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
axiom D_midpoint : D = (A + B) / 2
axiom E_midpoint : E = (B + C) / 2
axiom F_midpoint : F = (C + A) / 2

-- Define vector operations
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- State the theorem
theorem midpoint_vector_sum :
  vec D B - vec D E + vec C F = vec C D :=
sorry

end NUMINAMATH_CALUDE_midpoint_vector_sum_l3974_397453


namespace NUMINAMATH_CALUDE_power_function_through_fixed_point_l3974_397477

-- Define the fixed point
def P : ℝ × ℝ := (4, 2)

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_through_fixed_point :
  f P.1 = P.2 ∧ ∀ x > 0, f x = Real.sqrt x := by sorry

end NUMINAMATH_CALUDE_power_function_through_fixed_point_l3974_397477


namespace NUMINAMATH_CALUDE_gardener_mowing_time_l3974_397476

theorem gardener_mowing_time (B : ℝ) (together : ℝ) (h1 : B = 5) (h2 : together = 1.875) :
  ∃ A : ℝ, A = 3 ∧ 1 / A + 1 / B = 1 / together :=
by sorry

end NUMINAMATH_CALUDE_gardener_mowing_time_l3974_397476


namespace NUMINAMATH_CALUDE_seat_arrangement_count_l3974_397433

/-- The number of ways to select and arrange 3 people from a group of 7 --/
def seatArrangements : ℕ := 70

/-- The number of people in the class --/
def totalPeople : ℕ := 7

/-- The number of people to be rearranged --/
def peopleToRearrange : ℕ := 3

/-- The number of ways to arrange 3 people in a circle (considering rotations as identical) --/
def circularArrangements : ℕ := 2

theorem seat_arrangement_count :
  seatArrangements = circularArrangements * (Nat.choose totalPeople peopleToRearrange) := by
  sorry

end NUMINAMATH_CALUDE_seat_arrangement_count_l3974_397433


namespace NUMINAMATH_CALUDE_tamara_height_is_62_l3974_397493

/-- Calculates Tamara's height given Kim's height and the age difference effect -/
def tamaraHeight (kimHeight : ℝ) (ageDifference : ℕ) : ℝ :=
  2 * kimHeight - 4

/-- Calculates Gavin's height given Kim's height -/
def gavinHeight (kimHeight : ℝ) : ℝ :=
  2 * kimHeight + 6

/-- The combined height of all three people -/
def combinedHeight : ℝ := 200

/-- The age difference between Tamara and Kim -/
def ageDifference : ℕ := 5

/-- The change in height ratio per year of age difference -/
def ratioChangePerYear : ℝ := 0.2

theorem tamara_height_is_62 :
  ∃ (kimHeight : ℝ),
    tamaraHeight kimHeight ageDifference +
    kimHeight +
    gavinHeight kimHeight = combinedHeight ∧
    tamaraHeight kimHeight ageDifference = 62 := by
  sorry

end NUMINAMATH_CALUDE_tamara_height_is_62_l3974_397493


namespace NUMINAMATH_CALUDE_distinct_d_values_l3974_397418

theorem distinct_d_values (a b c : ℂ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃! (s : Finset ℂ), s.card = 6 ∧ 
  (∀ d : ℂ, d ∈ s ↔ 
    (∀ z : ℂ, (z - a) * (z - b) * (z - c) = (z - d^2 * a) * (z - d^2 * b) * (z - d^2 * c))) :=
by sorry

end NUMINAMATH_CALUDE_distinct_d_values_l3974_397418


namespace NUMINAMATH_CALUDE_characterize_M_l3974_397457

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | (m-1)*x - 1 = 0}

def M : Set ℝ := {m | A ∩ B m = B m}

theorem characterize_M : M = {3/2, 4/3, 1} := by sorry

end NUMINAMATH_CALUDE_characterize_M_l3974_397457


namespace NUMINAMATH_CALUDE_marble_probability_l3974_397439

theorem marble_probability (total : ℕ) (blue : ℕ) (red_white_prob : ℚ) : 
  total = 30 → blue = 5 → red_white_prob = 5/6 → (total - blue : ℚ) / total = red_white_prob :=
by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3974_397439


namespace NUMINAMATH_CALUDE_no_common_solution_l3974_397429

theorem no_common_solution :
  ¬∃ x : ℚ, (6 * (x - 2/3) - (x + 7) = 11) ∧ ((2*x - 1)/3 = (2*x + 1)/6 - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3974_397429


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3974_397412

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((Complex.I / (1 + 2 * Complex.I)) * Complex.I) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3974_397412


namespace NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3974_397442

-- Define a sphere with center and radius
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the eight spheres
def octantSpheres : List Sphere :=
  [⟨(2, 2, 2), 2⟩, ⟨(-2, 2, 2), 2⟩, ⟨(2, -2, 2), 2⟩, ⟨(2, 2, -2), 2⟩,
   ⟨(-2, -2, 2), 2⟩, ⟨(-2, 2, -2), 2⟩, ⟨(2, -2, -2), 2⟩, ⟨(-2, -2, -2), 2⟩]

-- Function to check if a sphere is tangent to coordinate planes
def isTangentToCoordinatePlanes (s : Sphere) : Prop :=
  let (x, y, z) := s.center
  (|x| = s.radius ∨ |y| = s.radius ∨ |z| = s.radius)

-- Function to check if a sphere contains another sphere
def containsSphere (outer : Sphere) (inner : Sphere) : Prop :=
  let (x₁, y₁, z₁) := outer.center
  let (x₂, y₂, z₂) := inner.center
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)^(1/2) + inner.radius ≤ outer.radius

-- Theorem statement
theorem smallest_enclosing_sphere_radius :
  ∃ (r : ℝ), r = 2 + 2 * Real.sqrt 3 ∧
  (∀ s ∈ octantSpheres, isTangentToCoordinatePlanes s) ∧
  (∀ r' : ℝ, r' < r →
    ∃ s ∈ octantSpheres, ¬containsSphere ⟨(0, 0, 0), r'⟩ s) ∧
  (∀ s ∈ octantSpheres, containsSphere ⟨(0, 0, 0), r⟩ s) := by
  sorry

end NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3974_397442


namespace NUMINAMATH_CALUDE_product_of_algebraic_expressions_l3974_397469

theorem product_of_algebraic_expressions (a b : ℝ) :
  (-8 * a * b) * ((3 / 4) * a^2 * b) = -6 * a^3 * b^2 := by sorry

end NUMINAMATH_CALUDE_product_of_algebraic_expressions_l3974_397469


namespace NUMINAMATH_CALUDE_book_cost_problem_l3974_397427

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) :
  total_cost = 360 ∧ loss_percent = 0.15 ∧ gain_percent = 0.19 →
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent) = cost_gain * (1 + gain_percent) ∧
    cost_loss = 210 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3974_397427


namespace NUMINAMATH_CALUDE_banana_permutations_l3974_397472

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem banana_permutations :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  multinomial_coefficient total_letters [b_count, a_count, n_count] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l3974_397472


namespace NUMINAMATH_CALUDE_pigeon_percentage_among_non_sparrows_l3974_397482

def bird_distribution (pigeon sparrow crow dove : ℝ) : Prop :=
  pigeon + sparrow + crow + dove = 100 ∧
  pigeon = 40 ∧
  sparrow = 20 ∧
  crow = 15 ∧
  dove = 25

theorem pigeon_percentage_among_non_sparrows 
  (pigeon sparrow crow dove : ℝ) 
  (h : bird_distribution pigeon sparrow crow dove) : 
  (pigeon / (pigeon + crow + dove)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_percentage_among_non_sparrows_l3974_397482


namespace NUMINAMATH_CALUDE_pencils_per_row_l3974_397475

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 12 → num_rows = 3 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l3974_397475


namespace NUMINAMATH_CALUDE_coronavirus_state_after_three_days_l3974_397488

/-- Represents the state of Coronavirus cases on a given day -/
structure CoronavirusState where
  positiveCase : ℕ
  hospitalizedCase : ℕ
  deaths : ℕ

/-- Calculates the next day's Coronavirus state based on the current state and rates -/
def nextDayState (state : CoronavirusState) (newCaseRate : ℝ) (recoveryRate : ℝ) 
                 (hospitalizationRate : ℝ) (hospitalizationIncreaseRate : ℝ)
                 (deathRate : ℝ) (deathIncreaseRate : ℝ) : CoronavirusState :=
  sorry

/-- Theorem stating the Coronavirus state after 3 days given initial conditions -/
theorem coronavirus_state_after_three_days 
  (initialState : CoronavirusState)
  (newCaseRate : ℝ)
  (recoveryRate : ℝ)
  (hospitalizationRate : ℝ)
  (hospitalizationIncreaseRate : ℝ)
  (deathRate : ℝ)
  (deathIncreaseRate : ℝ)
  (h1 : initialState.positiveCase = 2000)
  (h2 : newCaseRate = 0.15)
  (h3 : recoveryRate = 0.05)
  (h4 : hospitalizationRate = 0.03)
  (h5 : hospitalizationIncreaseRate = 0.10)
  (h6 : deathRate = 0.02)
  (h7 : deathIncreaseRate = 0.05) :
  let day1 := nextDayState initialState newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day2 := nextDayState day1 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day3 := nextDayState day2 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  day3.positiveCase = 2420 ∧ day3.hospitalizedCase = 92 ∧ day3.deaths = 57 :=
sorry

end NUMINAMATH_CALUDE_coronavirus_state_after_three_days_l3974_397488


namespace NUMINAMATH_CALUDE_lucy_fish_count_l3974_397411

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l3974_397411


namespace NUMINAMATH_CALUDE_share_multiple_is_four_l3974_397460

/-- Represents the shares of three people in a division problem. -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Proves that the multiple of a's share is 4 under given conditions. -/
theorem share_multiple_is_four 
  (total : ℝ) 
  (shares : Shares) 
  (h_total : total = 880)
  (h_c_share : shares.c = 160)
  (h_sum : shares.a + shares.b + shares.c = total)
  (h_equal : ∃ x : ℝ, x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c) :
  ∃ x : ℝ, x = 4 ∧ x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c := by
  sorry

end NUMINAMATH_CALUDE_share_multiple_is_four_l3974_397460


namespace NUMINAMATH_CALUDE_exist_three_fractions_product_one_l3974_397413

/-- The sequence of fractions from 1/2017 to 2017/1 -/
def fraction_sequence : Fin 2017 → Rat := λ i => (i + 1) / (2018 - (i + 1))

/-- Theorem: There exist three fractions in the sequence whose product is 1 -/
theorem exist_three_fractions_product_one :
  ∃ (i j k : Fin 2017), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    fraction_sequence i * fraction_sequence j * fraction_sequence k = 1 := by
  sorry

end NUMINAMATH_CALUDE_exist_three_fractions_product_one_l3974_397413


namespace NUMINAMATH_CALUDE_sum_of_greater_than_l3974_397479

theorem sum_of_greater_than (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_greater_than_l3974_397479


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3974_397425

/-- Given a line y = mx + 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    prove that the possible slopes m satisfy m^2 ≥ 1/55. -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 3) → m^2 ≥ 1/55 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l3974_397425


namespace NUMINAMATH_CALUDE_salazar_oranges_l3974_397498

theorem salazar_oranges (initial_oranges : ℕ) (sold_fraction : ℚ) 
  (rotten_oranges : ℕ) (remaining_oranges : ℕ) :
  initial_oranges = 7 * 12 →
  sold_fraction = 3 / 7 →
  rotten_oranges = 4 →
  remaining_oranges = 32 →
  ∃ (f : ℚ), 
    0 ≤ f ∧ f ≤ 1 ∧
    (1 - f) * initial_oranges - sold_fraction * ((1 - f) * initial_oranges) - rotten_oranges = remaining_oranges ∧
    f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_salazar_oranges_l3974_397498


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3974_397426

/-- The function f(x) = |2x+1| + |2x-3| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

/-- Theorem for the range of a -/
theorem range_of_a (a : ℝ) : (∀ x, f x > |1 - 3*a|) → -1 < a ∧ a < 5/3 := by sorry

/-- Theorem for the range of m -/
theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 4*Real.sqrt 2*t + f m = 0) → 
  -3/2 ≤ m ∧ m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3974_397426


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3974_397424

/-- Given two vectors in ℝ², prove that if k * a + b is perpendicular to a - 3 * b, then k = 19 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) :
  k = 19 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3974_397424


namespace NUMINAMATH_CALUDE_solution_l3974_397434

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + p.2)^2 = p.1^2 + p.2^2}

-- Define the x-axis and y-axis
def X_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def Y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem stating that S is equivalent to the union of X_axis and Y_axis
theorem solution : S = X_axis ∪ Y_axis := by
  sorry

end NUMINAMATH_CALUDE_solution_l3974_397434


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3974_397496

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 800)
  (h2 : selling_price = 1080) :
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3974_397496


namespace NUMINAMATH_CALUDE_triangle_transformation_l3974_397407

theorem triangle_transformation (n : ℕ) (remaining_fraction : ℚ) :
  n = 3 ∧ 
  remaining_fraction = (8 / 9 : ℚ)^n → 
  remaining_fraction = 512 / 729 := by
sorry

end NUMINAMATH_CALUDE_triangle_transformation_l3974_397407


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_divisible_by_four_l3974_397474

theorem consecutive_odd_sum_divisible_by_four (n : ℤ) : 
  4 ∣ ((2*n + 1) + (2*n + 3)) := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_divisible_by_four_l3974_397474


namespace NUMINAMATH_CALUDE_chord_equation_l3974_397455

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

end NUMINAMATH_CALUDE_chord_equation_l3974_397455


namespace NUMINAMATH_CALUDE_smallest_x_squared_l3974_397447

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  h : AB = 120 ∧ CD = 25

/-- A circle is tangent to AD if its center is on AB and touches AD -/
def is_tangent_circle (t : Trapezoid) (center : ℝ) : Prop :=
  0 ≤ center ∧ center ≤ t.AB ∧ ∃ (point : ℝ), 0 ≤ point ∧ point ≤ t.x

/-- The theorem stating the smallest possible value of x^2 -/
theorem smallest_x_squared (t : Trapezoid) : 
  (∃ center, is_tangent_circle t center) → 
  (∀ y, (∃ center, is_tangent_circle { AB := t.AB, CD := t.CD, x := y, h := t.h } center) → 
    t.x^2 ≤ y^2) → 
  t.x^2 = 3443.75 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_squared_l3974_397447


namespace NUMINAMATH_CALUDE_zero_exponent_l3974_397450

theorem zero_exponent (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l3974_397450


namespace NUMINAMATH_CALUDE_hiking_team_participants_l3974_397473

theorem hiking_team_participants (min_gloves : ℕ) (gloves_per_participant : ℕ) : 
  min_gloves = 86 → gloves_per_participant = 2 → min_gloves / gloves_per_participant = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_team_participants_l3974_397473


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l3974_397456

/-- Given a regular tetrahedron with face area S and volume V, 
    the radius of its inscribed sphere is 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) : ℝ :=
  3 * V / (4 * S)

/-- The calculated radius is indeed the radius of the inscribed sphere -/
theorem inscribed_sphere_radius_regular_tetrahedron_is_correct 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) :
  inscribed_sphere_radius_regular_tetrahedron S V S_pos V_pos = 
    3 * V / (4 * S) := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l3974_397456


namespace NUMINAMATH_CALUDE_no_root_greater_than_four_l3974_397415

-- Define the three equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x - 2)^2 = (2*x - 3)^2
def equation3 (x : ℝ) : Prop := (x^2 - 16 : ℝ) = 2*x - 4

-- Theorem stating that no root is greater than 4 for all equations
theorem no_root_greater_than_four :
  (∀ x > 4, ¬ equation1 x) ∧
  (∀ x > 4, ¬ equation2 x) ∧
  (∀ x > 4, ¬ equation3 x) :=
by sorry

end NUMINAMATH_CALUDE_no_root_greater_than_four_l3974_397415


namespace NUMINAMATH_CALUDE_zilla_savings_calculation_l3974_397485

def monthly_savings (total_earnings : ℝ) (rent_amount : ℝ) : ℝ :=
  let after_tax := total_earnings * 0.9
  let rent_percent := rent_amount / after_tax
  let groceries := after_tax * 0.3
  let entertainment := after_tax * 0.2
  let transportation := after_tax * 0.12
  let total_expenses := rent_amount + groceries + entertainment + transportation
  let remaining := after_tax - total_expenses
  remaining * 0.15

theorem zilla_savings_calculation (total_earnings : ℝ) (h1 : total_earnings > 0) 
  (h2 : monthly_savings total_earnings 133 = 77.52) : 
  ∃ (e : ℝ), e = total_earnings ∧ monthly_savings e 133 = 77.52 :=
by
  sorry

#eval monthly_savings 1900 133

end NUMINAMATH_CALUDE_zilla_savings_calculation_l3974_397485


namespace NUMINAMATH_CALUDE_number_ratio_l3974_397497

theorem number_ratio (f s t : ℝ) : 
  s = 4 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  f ≤ s ∧ f ≤ t →
  t / f = 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l3974_397497


namespace NUMINAMATH_CALUDE_viewing_angle_midpoint_l3974_397405

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the viewing angle function
noncomputable def viewingAngle (c : Circle) (p : Point) : ℝ := sorry

-- Define the line AB
def lineAB (A B : Point) : Set Point := sorry

-- Theorem statement
theorem viewing_angle_midpoint (O : Circle) (A B : Point) :
  let α := viewingAngle O A
  let β := viewingAngle O B
  let γ := (α + β) / 2
  ∃ (C₁ C₂ : Point), C₁ ∈ lineAB A B ∧ C₂ ∈ lineAB A B ∧
    viewingAngle O C₁ = γ ∧ viewingAngle O C₂ = γ ∧
    (α = β → (C₁ = A ∧ C₂ = B) ∨ (C₁ = B ∧ C₂ = A)) :=
by sorry


end NUMINAMATH_CALUDE_viewing_angle_midpoint_l3974_397405


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3974_397451

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 110 kg increases the average weight by 5 kg, then the weight
    of the replaced person is 60 kg. -/
theorem replaced_person_weight
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h_initial_count : initial_count = 10)
  (h_new_person_weight : new_person_weight = 110)
  (h_average_increase : average_increase = 5)
  : ∃ (initial_average : ℝ) (replaced_weight : ℝ),
    initial_count * (initial_average + average_increase) =
    initial_count * initial_average + new_person_weight - replaced_weight ∧
    replaced_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3974_397451


namespace NUMINAMATH_CALUDE_jane_started_with_87_crayons_l3974_397462

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended up with -/
def remaining_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_started_with_87_crayons :
  initial_crayons = eaten_crayons + remaining_crayons :=
by sorry

end NUMINAMATH_CALUDE_jane_started_with_87_crayons_l3974_397462


namespace NUMINAMATH_CALUDE_magic_deck_problem_l3974_397409

/-- Given a magician selling magic card decks, this theorem proves
    the number of decks left unsold at the end of the day. -/
theorem magic_deck_problem (initial_decks : ℕ) (price_per_deck : ℕ) (total_earnings : ℕ) :
  initial_decks = 16 →
  price_per_deck = 7 →
  total_earnings = 56 →
  initial_decks - (total_earnings / price_per_deck) = 8 := by
  sorry

end NUMINAMATH_CALUDE_magic_deck_problem_l3974_397409


namespace NUMINAMATH_CALUDE_largest_divisible_n_ten_is_divisible_largest_n_is_ten_l3974_397470

theorem largest_divisible_n : ∀ n : ℕ, n > 10 → ¬(n + 15 ∣ n^3 + 250) := by
  sorry

theorem ten_is_divisible : (10 + 15 ∣ 10^3 + 250) := by
  sorry

theorem largest_n_is_ten : 
  ∀ n : ℕ, n > 0 → (n + 15 ∣ n^3 + 250) → n ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_ten_is_divisible_largest_n_is_ten_l3974_397470


namespace NUMINAMATH_CALUDE_bicycle_costs_l3974_397490

theorem bicycle_costs (B H L : ℝ) 
  (total_cost : B + H + L = 480)
  (bicycle_helmet_ratio : B = 5 * H)
  (lock_helmet_ratio : L = 0.5 * H)
  (lock_total_ratio : L = 0.1 * 480) : 
  B = 360 ∧ H = 72 ∧ L = 48 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_costs_l3974_397490


namespace NUMINAMATH_CALUDE_pear_vendor_theorem_l3974_397401

/-- Represents the actions of a pear vendor over two days --/
def pear_vendor_problem (initial_pears : ℝ) : Prop :=
  let day1_sold := 0.8 * initial_pears
  let day1_remaining := initial_pears - day1_sold
  let day1_thrown := 0.5 * day1_remaining
  let day2_start := day1_remaining - day1_thrown
  let day2_sold := 0.8 * day2_start
  let day2_thrown := day2_start - day2_sold
  let total_thrown := day1_thrown + day2_thrown
  (total_thrown / initial_pears) * 100 = 12

/-- Theorem stating that the pear vendor throws away 12% of the initial pears --/
theorem pear_vendor_theorem :
  ∀ initial_pears : ℝ, initial_pears > 0 → pear_vendor_problem initial_pears :=
by
  sorry

end NUMINAMATH_CALUDE_pear_vendor_theorem_l3974_397401


namespace NUMINAMATH_CALUDE_place_balls_in_boxes_theorem_l3974_397432

/-- The number of ways to place 4 distinct balls into 4 distinct boxes such that exactly two boxes remain empty -/
def place_balls_in_boxes : ℕ :=
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 4
  let n_empty_boxes : ℕ := 2
  -- The actual calculation is not implemented here
  84

/-- Theorem stating that the number of ways to place 4 distinct balls into 4 distinct boxes
    such that exactly two boxes remain empty is 84 -/
theorem place_balls_in_boxes_theorem :
  place_balls_in_boxes = 84 := by
  sorry

end NUMINAMATH_CALUDE_place_balls_in_boxes_theorem_l3974_397432


namespace NUMINAMATH_CALUDE_total_books_count_l3974_397404

/-- Given that Sandy has 10 books, Benny has 24 books, and Tim has 33 books,
    prove that they have 67 books in total. -/
theorem total_books_count (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) : 
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l3974_397404


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3974_397458

/-- A geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_fifth_term
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 7 * seq.a 9 = 25) :
  seq.a 5 = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3974_397458


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_32767_l3974_397492

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_32767 :
  sum_of_digits (greatest_prime_divisor 32767) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_32767_l3974_397492


namespace NUMINAMATH_CALUDE_registration_methods_count_l3974_397419

/-- Represents the number of courses -/
def num_courses : ℕ := 3

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- 
Calculates the number of ways to distribute n distinct objects into k distinct boxes,
where each box must contain at least one object.
-/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of registration methods is 150 -/
theorem registration_methods_count : distribution_count num_students num_courses = 150 :=
  sorry

end NUMINAMATH_CALUDE_registration_methods_count_l3974_397419


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3974_397436

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The first line has equation y = 3x + 5 -/
def line1 : ℝ → ℝ := λ x => 3 * x + 5

/-- The second line has equation y = (6k)x + 1 -/
def line2 (k : ℝ) : ℝ → ℝ := λ x => 6 * k * x + 1

theorem parallel_lines_k_value :
  ∀ k : ℝ, parallel (line1 0 - line1 1) (line2 k 0 - line2 k 1) → k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3974_397436


namespace NUMINAMATH_CALUDE_bryans_books_l3974_397438

theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 9) 
  (h2 : books_per_shelf = 56) : 
  num_shelves * books_per_shelf = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l3974_397438


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3974_397430

theorem sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  (∃ x : ℝ, x > -1 ∧ (x + 1) * (x - 3) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3974_397430


namespace NUMINAMATH_CALUDE_k_value_l3974_397446

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3974_397446


namespace NUMINAMATH_CALUDE_smallest_y_value_l3974_397461

theorem smallest_y_value (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 18)) → y ≥ -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l3974_397461


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l3974_397471

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- A round-robin tournament with n players -/
structure RoundRobinTournament (n : ℕ) where
  players : Fin n → Type
  plays_once : ∀ (i j : Fin n), i ≠ j → Type

theorem ten_player_tournament_matches :
  ∀ (t : RoundRobinTournament 10),
  num_matches 10 = 45 := by
  sorry

#eval num_matches 10

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l3974_397471


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3974_397400

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3974_397400


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3974_397489

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (m - 1) * x₁^2 + 2 * x₁ + 1 = 0 ∧ 
    (m - 1) * x₂^2 + 2 * x₂ + 1 = 0) ↔ 
  (m ≤ 2 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3974_397489


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3974_397478

theorem absolute_value_sum (a : ℝ) (h : 3 < a ∧ a < 4) : |a - 3| + |a - 4| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3974_397478


namespace NUMINAMATH_CALUDE_solve_flowers_problem_l3974_397423

def flowers_problem (lilies sunflowers daisies total_flowers : ℕ) : Prop :=
  let other_flowers := lilies + sunflowers + daisies
  let roses := total_flowers - other_flowers
  (lilies = 40) ∧ (sunflowers = 40) ∧ (daisies = 40) ∧ (total_flowers = 160) →
  roses = 40

theorem solve_flowers_problem :
  ∀ (lilies sunflowers daisies total_flowers : ℕ),
  flowers_problem lilies sunflowers daisies total_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flowers_problem_l3974_397423


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3974_397454

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3974_397454


namespace NUMINAMATH_CALUDE_remainder_base12_2543_div_9_l3974_397491

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2543
def base12_2543 : List Nat := [2, 5, 4, 3]

-- Theorem statement
theorem remainder_base12_2543_div_9 :
  (base12ToDecimal base12_2543) % 9 = 8 := by
  sorry


end NUMINAMATH_CALUDE_remainder_base12_2543_div_9_l3974_397491


namespace NUMINAMATH_CALUDE_balance_rearrangements_l3974_397464

def word : String := "BALANCE"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  Nat.factorial vowels.length / (Nat.factorial 2)  -- 2 is the count of repeated 'A's

def consonant_arrangements : ℕ :=
  Nat.factorial consonants.length

theorem balance_rearrangements :
  vowel_arrangements * consonant_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_balance_rearrangements_l3974_397464


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3974_397445

/-- Proves that for a parabola y^2 = 2px with p > 0, if a point P(2,m) on the parabola
    is at a distance of 4 from its focus, then p = 4. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) (h1 : p > 0) :
  m^2 = 2*p*2 →  -- Point P(2,m) is on the parabola y^2 = 2px
  (2 - p/2)^2 + m^2 = 4^2 →  -- Distance from P to focus F(p/2, 0) is 4
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3974_397445


namespace NUMINAMATH_CALUDE_machine_production_rate_l3974_397448

/-- Given an industrial machine that made 8 shirts in 4 minutes today,
    prove that it can make 2 shirts per minute. -/
theorem machine_production_rate (shirts_today : ℕ) (minutes_today : ℕ) 
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  shirts_today / minutes_today = 2 := by
sorry

end NUMINAMATH_CALUDE_machine_production_rate_l3974_397448


namespace NUMINAMATH_CALUDE_largest_prime_for_integer_sqrt_l3974_397486

theorem largest_prime_for_integer_sqrt : ∃ (p : ℕ), 
  Prime p ∧ 
  (∃ (q : ℕ), q^2 = 17*p + 625) ∧
  (∀ (p' : ℕ), Prime p' → (∃ (q' : ℕ), q'^2 = 17*p' + 625) → p' ≤ p) ∧
  p = 67 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_for_integer_sqrt_l3974_397486


namespace NUMINAMATH_CALUDE_group_size_after_new_member_l3974_397444

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 + 32) / (n + 1) = 15 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_group_size_after_new_member_l3974_397444


namespace NUMINAMATH_CALUDE_expand_polynomial_l3974_397421

theorem expand_polynomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3974_397421


namespace NUMINAMATH_CALUDE_tim_laundry_cycle_l3974_397435

/-- Ronald's laundry cycle in days -/
def ronald_cycle : ℕ := 6

/-- Number of days until they both do laundry on the same day again -/
def next_common_day : ℕ := 18

/-- Tim's laundry cycle in days -/
def tim_cycle : ℕ := 3

theorem tim_laundry_cycle :
  (ronald_cycle ∣ next_common_day) ∧
  (tim_cycle ∣ next_common_day) ∧
  (tim_cycle < ronald_cycle) ∧
  (∀ x : ℕ, x < tim_cycle → ¬(x ∣ next_common_day ∧ x ∣ ronald_cycle)) :=
sorry

end NUMINAMATH_CALUDE_tim_laundry_cycle_l3974_397435


namespace NUMINAMATH_CALUDE_akiras_weight_l3974_397495

/-- Given the weights of pairs of people, determine Akira's weight -/
theorem akiras_weight (akira jamie rabia : ℕ) 
  (h1 : akira + jamie = 101)
  (h2 : akira + rabia = 91)
  (h3 : rabia + jamie = 88) :
  akira = 52 := by
  sorry

end NUMINAMATH_CALUDE_akiras_weight_l3974_397495


namespace NUMINAMATH_CALUDE_square_perimeter_l3974_397410

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 64)
  (h3 : square_area = 2 * rectangle_length * rectangle_width) : 
  4 * Real.sqrt square_area = 256 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3974_397410


namespace NUMINAMATH_CALUDE_function_count_l3974_397417

theorem function_count (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x * y) + f x + f y - f x * f y ≥ 2) ↔ 
  (∀ x : ℝ, f x = 1 ∨ f x = 2) :=
sorry

end NUMINAMATH_CALUDE_function_count_l3974_397417


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3974_397494

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, (42 * z.val + 8) % 24 = 4 → y ≤ z) ∧
    (42 * y.val + 8) % 24 = 4 ∧
    y.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3974_397494


namespace NUMINAMATH_CALUDE_eggs_per_box_l3974_397441

/-- Given a chicken coop with hens that lay eggs daily, prove the number of eggs per box -/
theorem eggs_per_box 
  (num_hens : ℕ) 
  (eggs_per_hen_per_day : ℕ) 
  (days_per_week : ℕ) 
  (boxes_per_week : ℕ) 
  (h1 : num_hens = 270)
  (h2 : eggs_per_hen_per_day = 1)
  (h3 : days_per_week = 7)
  (h4 : boxes_per_week = 315) :
  (num_hens * eggs_per_hen_per_day * days_per_week) / boxes_per_week = 6 :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3974_397441


namespace NUMINAMATH_CALUDE_bertsDogWeight_l3974_397459

/-- Calculates the adult weight of a golden retriever given its growth pattern -/
def goldenRetrieverAdultWeight (initialWeight : ℕ) (finalIncrease : ℕ) : ℕ :=
  ((initialWeight * 2 * 2 * 2) + finalIncrease)

/-- Theorem stating that the adult weight of Bert's golden retriever is 78 pounds -/
theorem bertsDogWeight :
  goldenRetrieverAdultWeight 6 30 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bertsDogWeight_l3974_397459


namespace NUMINAMATH_CALUDE_line_AB_equation_l3974_397465

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define that P is the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ is_midpoint A B →
  ∀ x y : ℝ, (y = x - 1) ↔ (y - A.2 = A.2 - A.1 * (x - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_line_AB_equation_l3974_397465


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l3974_397481

theorem smallest_four_digit_solution : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (5 * x ≡ 25 [ZMOD 20]) ∧
  (3 * x + 4 ≡ 10 [ZMOD 7]) ∧
  (-x + 3 ≡ 2 * x [ZMOD 15]) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬((5 * y ≡ 25 [ZMOD 20]) ∧
      (3 * y + 4 ≡ 10 [ZMOD 7]) ∧
      (-y + 3 ≡ 2 * y [ZMOD 15]))) ∧
  x = 1021 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l3974_397481


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l3974_397431

/-- The number of sides in the regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees)
    is equal to 17 + (360 / 17) -/
theorem regular_17gon_symmetry_sum :
  (L n : ℚ) + R n = 17 + 360 / 17 := by sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l3974_397431


namespace NUMINAMATH_CALUDE_total_stars_is_116_l3974_397449

/-- The number of people in the Young Pioneers group -/
def n : ℕ := sorry

/-- The total number of lucky stars planned to be made -/
def total_stars : ℕ := sorry

/-- Condition 1: If each person makes 10 stars, they will be 6 stars short of completing the plan -/
axiom condition1 : 10 * n + 6 = total_stars

/-- Condition 2: If 4 of them each make 8 stars and the rest each make 12 stars, they will just complete the plan -/
axiom condition2 : 4 * 8 + (n - 4) * 12 = total_stars

/-- Theorem: The total number of lucky stars planned to be made is 116 -/
theorem total_stars_is_116 : total_stars = 116 := by sorry

end NUMINAMATH_CALUDE_total_stars_is_116_l3974_397449


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3974_397440

theorem simplify_trig_expression : 
  (1 - Real.cos (30 * π / 180)) * (1 + Real.cos (30 * π / 180)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3974_397440


namespace NUMINAMATH_CALUDE_relative_speed_calculation_l3974_397408

/-- Convert meters per second to kilometers per hour -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

/-- Convert centimeters per minute to kilometers per hour -/
def cmpm_to_kmh (speed_cmpm : ℝ) : ℝ :=
  speed_cmpm * 0.0006

/-- Calculate the relative speed of two objects moving in opposite directions -/
def relative_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem relative_speed_calculation (speed1_ms : ℝ) (speed2_cmpm : ℝ) 
  (h1 : speed1_ms = 12.5)
  (h2 : speed2_cmpm = 1800) :
  relative_speed (ms_to_kmh speed1_ms) (cmpm_to_kmh speed2_cmpm) = 46.08 := by
  sorry

#check relative_speed_calculation

end NUMINAMATH_CALUDE_relative_speed_calculation_l3974_397408


namespace NUMINAMATH_CALUDE_p_and_q_true_l3974_397406

theorem p_and_q_true (h1 : p ∨ q) (h2 : p ∧ q) : p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l3974_397406


namespace NUMINAMATH_CALUDE_number_divided_by_002_l3974_397443

theorem number_divided_by_002 :
  ∃ x : ℝ, x / 0.02 = 201.79999999999998 ∧ x = 4.0359999999999996 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_002_l3974_397443


namespace NUMINAMATH_CALUDE_researcher_can_reach_oasis_l3974_397402

/-- Represents a traveler in the desert -/
structure Traveler where
  food : ℕ
  position : ℕ

/-- Represents the state of the journey -/
structure JourneyState where
  researcher : Traveler
  porters : List Traveler
  day : ℕ

def oasisDistance : ℕ := 380
def dailyTravel : ℕ := 60
def maxFood : ℕ := 4

def canReachOasis (initialState : JourneyState) : Prop :=
  ∃ (finalState : JourneyState),
    finalState.researcher.position = oasisDistance ∧
    finalState.day ≤ initialState.researcher.food * maxFood ∧
    ∀ porter ∈ finalState.porters, porter.position = 0

theorem researcher_can_reach_oasis :
  ∃ (initialState : JourneyState),
    initialState.researcher.food = maxFood ∧
    initialState.researcher.position = 0 ∧
    initialState.porters.length = 2 ∧
    (∀ porter ∈ initialState.porters, porter.food = maxFood ∧ porter.position = 0) ∧
    initialState.day = 0 ∧
    canReachOasis initialState :=
  sorry

end NUMINAMATH_CALUDE_researcher_can_reach_oasis_l3974_397402


namespace NUMINAMATH_CALUDE_related_transitive_l3974_397420

/-- A function is great if it satisfies the given condition for all nonnegative integers m and n. -/
def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

/-- Two sequences are related (∼) if there exists a great function satisfying the given conditions. -/
def Related (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

/-- The main theorem to be proved. -/
theorem related_transitive (A B C D : ℕ → ℤ) 
  (hAB : Related A B) (hBC : Related B C) (hCD : Related C D) : Related D A := by
  sorry

end NUMINAMATH_CALUDE_related_transitive_l3974_397420


namespace NUMINAMATH_CALUDE_sample_frequency_calculation_l3974_397452

theorem sample_frequency_calculation (total_volume : ℕ) (num_groups : ℕ) 
  (freq_3 freq_4 freq_5 freq_6 : ℕ) (ratio_group_1 : ℚ) :
  total_volume = 80 →
  num_groups = 6 →
  freq_3 = 10 →
  freq_4 = 12 →
  freq_5 = 14 →
  freq_6 = 20 →
  ratio_group_1 = 1/5 →
  ∃ (freq_1 freq_2 : ℕ),
    freq_1 = 16 ∧
    freq_2 = 8 ∧
    freq_1 + freq_2 + freq_3 + freq_4 + freq_5 + freq_6 = total_volume ∧
    freq_1 = (ratio_group_1 * total_volume).num := by
  sorry

end NUMINAMATH_CALUDE_sample_frequency_calculation_l3974_397452


namespace NUMINAMATH_CALUDE_find_b_l3974_397466

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, p x = 2 * x - 7)
  (h2 : ∀ x, q x = 3 * x - b)
  (h3 : p (q 4) = 7) : 
  b = 5 := by sorry

end NUMINAMATH_CALUDE_find_b_l3974_397466


namespace NUMINAMATH_CALUDE_root_of_log_equation_l3974_397480

theorem root_of_log_equation :
  ∃! x : ℝ, x > 1 ∧ Real.log x = x - 5 ∧ 5 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_root_of_log_equation_l3974_397480


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3974_397422

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 8 * x + c = 0) →
  a + 2 * c = 14 →
  a < c →
  (a = (7 - Real.sqrt 17) / 2 ∧ c = (7 + Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3974_397422
