import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3196_319674

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (10, 5)

-- Define the property that A and C are diagonally opposite
def diagonally_opposite (A C : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the property of a parallelogram
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1 = D.1 - C.1) ∧ (B.2 - A.2 = D.2 - C.2)

-- Theorem statement
theorem parallelogram_vertex_sum :
  ∀ D : ℝ × ℝ,
  is_parallelogram A B C D →
  diagonally_opposite A C →
  D.1 + D.2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3196_319674


namespace NUMINAMATH_CALUDE_xy_equals_five_l3196_319655

theorem xy_equals_five (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_five_l3196_319655


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l3196_319641

/-- Given an inverse proportion function f(x) = -6/x, prove that y₁ < y₂ 
    where (2, y₁) and (-1, y₂) lie on the graph of f. -/
theorem inverse_proportion_inequality (y₁ y₂ : ℝ) : 
  y₁ = -6/2 → y₂ = -6/(-1) → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l3196_319641


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3196_319678

/-- Given a bag of black and white marbles, if removing one black marble
    results in 1/8 of the remaining marbles being black, and removing three
    white marbles results in 1/6 of the remaining marbles being black,
    then the initial number of marbles in the bag is 9. -/
theorem marble_bag_problem (x y : ℕ) : 
  x > 0 → y > 0 →
  (x - 1 : ℚ) / (x + y - 1 : ℚ) = 1 / 8 →
  x / (x + y - 3 : ℚ) = 1 / 6 →
  x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3196_319678


namespace NUMINAMATH_CALUDE_circle_equation_tangent_line_equation_l3196_319698

-- Define the circle
def circle_center : ℝ × ℝ := (-1, 2)
def line_m (x y : ℝ) : ℝ := x + 2*y + 7

-- Define the point Q
def point_Q : ℝ × ℝ := (1, 6)

-- Theorem for the circle equation
theorem circle_equation : 
  ∃ (r : ℝ), ∀ (x y : ℝ), 
  (x + 1)^2 + (y - 2)^2 = r^2 ∧ 
  (∃ (x₀ y₀ : ℝ), line_m x₀ y₀ = 0 ∧ ((x₀ + 1)^2 + (y₀ - 2)^2 = r^2)) :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 2)^2 = 20) →
  (point_Q.1 + 1)^2 + (point_Q.2 - 2)^2 = 20 →
  (y - point_Q.2 = -(x - point_Q.1) / 2) ↔ (x + 2*y - 13 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_line_equation_l3196_319698


namespace NUMINAMATH_CALUDE_two_places_distribution_three_places_distribution_ambulance_distribution_l3196_319658

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of places --/
def num_places : ℕ := 3

/-- The number of ambulances --/
def num_ambulances : ℕ := 20

/-- The number of ways to distribute 4 volunteers to 2 places with 2 volunteers in each place --/
theorem two_places_distribution (n : ℕ) (h : n = num_volunteers) : 
  Nat.choose n 2 = 6 := by sorry

/-- The number of ways to distribute 4 volunteers to 3 places with at least one volunteer in each place --/
theorem three_places_distribution (n m : ℕ) (h1 : n = num_volunteers) (h2 : m = num_places) : 
  6 * Nat.factorial (m - 1) = 36 := by sorry

/-- The number of ways to distribute 20 identical ambulances to 3 places with at least one ambulance in each place --/
theorem ambulance_distribution (a m : ℕ) (h1 : a = num_ambulances) (h2 : m = num_places) : 
  Nat.choose (a - 1) (m - 1) = 171 := by sorry

end NUMINAMATH_CALUDE_two_places_distribution_three_places_distribution_ambulance_distribution_l3196_319658


namespace NUMINAMATH_CALUDE_no_seven_digit_number_divisible_by_another_l3196_319659

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 : ℕ),
    d1 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d2 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d3 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d4 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d5 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d6 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d7 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧
    d6 ≠ d7 ∧
    n = d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7

theorem no_seven_digit_number_divisible_by_another :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a % b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_seven_digit_number_divisible_by_another_l3196_319659


namespace NUMINAMATH_CALUDE_light_glow_start_time_l3196_319661

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Calculates the difference between two times in seconds -/
def timeDiffInSeconds (t1 t2 : Time) : Nat :=
  (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) -
  (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) 
  (start_time : Time) : 
  glow_interval = 17 →
  glow_count = 292 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  start_time = { hours := 1, minutes := 58, seconds := 3 } →
  timeDiffInSeconds end_time start_time = glow_interval * glow_count :=
by sorry

end NUMINAMATH_CALUDE_light_glow_start_time_l3196_319661


namespace NUMINAMATH_CALUDE_multiples_of_12_between_15_and_205_l3196_319648

theorem multiples_of_12_between_15_and_205 : 
  (Finset.filter (fun n => 12 ∣ n) (Finset.Ioo 15 205)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_15_and_205_l3196_319648


namespace NUMINAMATH_CALUDE_crayon_difference_is_1040_l3196_319653

/-- The number of crayons Willy and Lucy have combined, minus the number of crayons Max has -/
def crayon_difference (willy_crayons lucy_crayons max_crayons : ℕ) : ℕ :=
  (willy_crayons + lucy_crayons) - max_crayons

/-- Theorem stating that the difference in crayons is 1040 -/
theorem crayon_difference_is_1040 :
  crayon_difference 1400 290 650 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_is_1040_l3196_319653


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3196_319663

/-- Represents the fishing schedule in a coastal village over three days -/
structure FishingSchedule where
  everyday : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterdayCount : Nat
  todayCount : Nat

/-- Calculates the number of people fishing tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  schedule.everyday +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.everyday))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule)
  (h1 : schedule.everyday = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { everyday := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3196_319663


namespace NUMINAMATH_CALUDE_square_vertex_coordinates_l3196_319620

def is_valid_vertex (x y : ℕ) : Prop :=
  (Nat.gcd x y = 2) ∧ 
  (2 * (x^2 + y^2) = 10 * Nat.lcm x y)

theorem square_vertex_coordinates : 
  ∀ x y : ℕ, is_valid_vertex x y ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
sorry

end NUMINAMATH_CALUDE_square_vertex_coordinates_l3196_319620


namespace NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l3196_319621

theorem range_of_s_squared_minus_c_squared 
  (x y z : ℝ) 
  (r : ℝ) 
  (hr : r = Real.sqrt (x^2 + y^2 + z^2)) 
  (s : ℝ) 
  (hs : s = y / r) 
  (c : ℝ) 
  (hc : c = x / r) : 
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l3196_319621


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l3196_319615

/-- The number of streetlights -/
def total_streetlights : ℕ := 400

/-- Alla's position when the snapshot is taken -/
def alla_snapshot : ℕ := 55

/-- Boris's position when the snapshot is taken -/
def boris_snapshot : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

theorem alla_boris_meeting :
  ∀ (v_a v_b : ℝ), v_a > 0 → v_b > 0 →
  (alla_snapshot - 1 : ℝ) / v_a = (total_streetlights - boris_snapshot : ℝ) / v_b →
  (meeting_point - 1 : ℝ) / v_a = (total_streetlights - meeting_point : ℝ) / v_b :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l3196_319615


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3196_319654

theorem consecutive_integers_sum (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3196_319654


namespace NUMINAMATH_CALUDE_h_max_at_3_l3196_319656

/-- Linear function f(x) = -2x + 8 -/
def f (x : ℝ) : ℝ := -2 * x + 8

/-- Linear function g(x) = 2x - 4 -/
def g (x : ℝ) : ℝ := 2 * x - 4

/-- Product function h(x) = f(x) * g(x) -/
def h (x : ℝ) : ℝ := f x * g x

/-- Theorem stating that h(x) reaches its maximum at x = 3 -/
theorem h_max_at_3 : ∀ x : ℝ, h x ≤ h 3 := by sorry

end NUMINAMATH_CALUDE_h_max_at_3_l3196_319656


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l3196_319636

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let n : ℕ := 6    -- Number of terms
  let S : ℚ := a * (1 - r^n) / (1 - r)  -- Formula for sum of geometric series
  S = 364/243
  := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l3196_319636


namespace NUMINAMATH_CALUDE_cos_minus_sin_identity_l3196_319640

theorem cos_minus_sin_identity (θ : Real) (a b : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_sin : Real.sin (2 * θ) = a)
  (h_cos : Real.cos (2 * θ) = b) :
  Real.cos θ - Real.sin θ = Real.sqrt (1 - a) :=
sorry

end NUMINAMATH_CALUDE_cos_minus_sin_identity_l3196_319640


namespace NUMINAMATH_CALUDE_coordinates_sum_of_point_b_l3196_319669

/-- Given two points A and B, where A is at the origin and B is on the line y=5,
    and the slope of segment AB is 3/4, prove that the sum of the x- and y-coordinates of B is 35/3 -/
theorem coordinates_sum_of_point_b (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_point_b_l3196_319669


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3196_319690

/-- An arithmetic sequence {a_n} with specified properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 7 - 2 * a 4 = -1)
  (h_third : a 3 = 0) :
  ∃ d : ℚ, (∀ n, a n = a 1 + (n - 1) * d) ∧ d = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3196_319690


namespace NUMINAMATH_CALUDE_iron_nickel_percentage_l3196_319631

/-- Represents the exchange of quarters for nickels, including special iron nickels --/
def nickel_exchange (num_quarters : ℕ) (total_value : ℚ) (iron_nickel_value : ℚ) : Prop :=
  ∃ (num_iron_nickels : ℕ),
    let num_nickels : ℕ := num_quarters * 5
    let regular_nickel_value : ℚ := 1/20
    (num_iron_nickels : ℚ) * iron_nickel_value + 
    ((num_nickels - num_iron_nickels) : ℚ) * regular_nickel_value = total_value ∧
    (num_iron_nickels : ℚ) / (num_nickels : ℚ) = 1/5

theorem iron_nickel_percentage 
  (h : nickel_exchange 20 64 3) : 
  ∃ (num_iron_nickels : ℕ), 
    (num_iron_nickels : ℚ) / 100 = 1/5 :=
by
  sorry

#check iron_nickel_percentage

end NUMINAMATH_CALUDE_iron_nickel_percentage_l3196_319631


namespace NUMINAMATH_CALUDE_max_tan_alpha_l3196_319666

theorem max_tan_alpha (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan (α + β) = 9 * Real.tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ 
    ∀ (γ : Real), (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ Real.tan (γ + δ) = 9 * Real.tan δ)) 
      → Real.tan γ ≤ max_tan_α :=
sorry

end NUMINAMATH_CALUDE_max_tan_alpha_l3196_319666


namespace NUMINAMATH_CALUDE_equation_real_roots_range_l3196_319689

theorem equation_real_roots_range (a : ℝ) : 
  (∀ x : ℝ, (2 + 3*a) / (5 - a) > 0) ↔ a ∈ Set.Ioo (-2/3 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_equation_real_roots_range_l3196_319689


namespace NUMINAMATH_CALUDE_unique_prime_square_l3196_319687

theorem unique_prime_square (p : ℕ) : 
  Prime p ∧ ∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2 ↔ p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_square_l3196_319687


namespace NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l3196_319603

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 :=
by sorry

theorem min_value_product_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 8 ∧
    (2 * a + b) * (a + 3 * c) * (b * c + 2) < 128 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_product_achieved_l3196_319603


namespace NUMINAMATH_CALUDE_playlist_composition_l3196_319667

theorem playlist_composition (initial_hip_hop_ratio : Real) 
  (country_percentage : Real) (hip_hop_percentage : Real) : 
  initial_hip_hop_ratio = 0.65 →
  country_percentage = 0.4 →
  hip_hop_percentage = (1 - country_percentage) * initial_hip_hop_ratio →
  hip_hop_percentage = 0.39 := by
  sorry

end NUMINAMATH_CALUDE_playlist_composition_l3196_319667


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3196_319638

theorem opposite_of_negative_fraction : 
  (-(-(1 : ℚ) / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3196_319638


namespace NUMINAMATH_CALUDE_puppy_cost_puppy_cost_proof_l3196_319664

/-- The cost of a puppy in a pet shop, given the following conditions:
  * There are 2 puppies and 4 kittens in the pet shop.
  * A kitten costs $15.
  * The total stock is worth $100. -/
theorem puppy_cost : ℕ :=
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 4
  let kitten_cost : ℕ := 15
  let total_stock_value : ℕ := 100
  20

/-- Proof that the cost of a puppy is $20. -/
theorem puppy_cost_proof : puppy_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_puppy_cost_proof_l3196_319664


namespace NUMINAMATH_CALUDE_circle_radius_l3196_319668

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 40) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3196_319668


namespace NUMINAMATH_CALUDE_ratio_problem_l3196_319616

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (x + 2 * y) = 4 / 5) : 
  x / y = 18 / 11 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3196_319616


namespace NUMINAMATH_CALUDE_max_value_of_f_on_S_l3196_319675

/-- The set S of real numbers x where x^4 - 13x^2 + 36 ≤ 0 -/
def S : Set ℝ := {x : ℝ | x^4 - 13*x^2 + 36 ≤ 0}

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- Theorem stating that the maximum value of f(x) on S is 18 -/
theorem max_value_of_f_on_S : ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), x ∈ S → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_S_l3196_319675


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3196_319600

/-- For all real numbers a and b, a²b - 25b = b(a + 5)(a - 5) -/
theorem polynomial_factorization (a b : ℝ) : a^2 * b - 25 * b = b * (a + 5) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3196_319600


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l3196_319608

theorem rational_inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l3196_319608


namespace NUMINAMATH_CALUDE_samples_per_box_l3196_319630

theorem samples_per_box (boxes_opened : ℕ) (samples_leftover : ℕ) (customers : ℕ) : 
  boxes_opened = 12 → samples_leftover = 5 → customers = 235 → 
  ∃ (samples_per_box : ℕ), samples_per_box * boxes_opened - samples_leftover = customers ∧ samples_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_samples_per_box_l3196_319630


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3196_319622

/-- An arithmetic progression is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The first three terms of the arithmetic progression -/
def first_term (x : ℝ) : ℝ := 2 * x - 1
def second_term (x : ℝ) : ℝ := 3 * x + 4
def third_term (x : ℝ) : ℝ := 5 * x + 6

/-- The theorem stating that x = 3 for the given arithmetic progression -/
theorem arithmetic_progression_x_value :
  ∃ x : ℝ, is_arithmetic_progression (fun n => match n with
    | 0 => first_term x
    | 1 => second_term x
    | _ => third_term x) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3196_319622


namespace NUMINAMATH_CALUDE_days_to_finish_book_l3196_319617

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem days_to_finish_book :
  ⌈(total_pages : ℝ) / pages_per_day⌉ = 13 := by sorry

end NUMINAMATH_CALUDE_days_to_finish_book_l3196_319617


namespace NUMINAMATH_CALUDE_workers_count_l3196_319691

-- Define the work function
def work (workers : ℕ) (hours : ℕ) : ℕ := workers * hours

-- Define the problem parameters
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def second_hours : ℕ := 6
def second_depth : ℕ := 55
def extra_workers : ℕ := 65

theorem workers_count :
  ∃ (W : ℕ), 
    (work W initial_hours) * second_depth = 
    (work (W + extra_workers) second_hours) * initial_depth ∧
    W = 45 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l3196_319691


namespace NUMINAMATH_CALUDE_diana_wins_prob_l3196_319643

/-- Represents the number of sides on Diana's die -/
def diana_sides : ℕ := 8

/-- Represents the number of sides on Apollo's die -/
def apollo_sides : ℕ := 6

/-- Calculates the probability of Diana rolling higher than Apollo -/
def prob_diana_higher : ℚ :=
  (diana_sides * (diana_sides - 1) - apollo_sides * (apollo_sides - 1)) / (2 * diana_sides * apollo_sides)

/-- Theorem stating that the probability of Diana rolling higher than Apollo is 9/16 -/
theorem diana_wins_prob : prob_diana_higher = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_diana_wins_prob_l3196_319643


namespace NUMINAMATH_CALUDE_external_diagonals_invalid_l3196_319665

theorem external_diagonals_invalid (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 8 →
  ¬(a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_invalid_l3196_319665


namespace NUMINAMATH_CALUDE_equation_solution_l3196_319681

theorem equation_solution (t : ℝ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * Real.sin t ^ 2 - Real.sin (2 * t) + 3 * Real.cos t ^ 2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3196_319681


namespace NUMINAMATH_CALUDE_room_area_calculation_l3196_319651

/-- Given a rectangular carpet covering 30% of a room's floor area,
    if the carpet measures 4 feet by 9 feet,
    then the total floor area is 120 square feet. -/
theorem room_area_calculation (carpet_length carpet_width carpet_coverage total_area : ℝ) : 
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_coverage = 0.30 →
  carpet_coverage * total_area = carpet_length * carpet_width →
  total_area = 120 := by
sorry

end NUMINAMATH_CALUDE_room_area_calculation_l3196_319651


namespace NUMINAMATH_CALUDE_line_vector_at_5_l3196_319610

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_5 :
  (∀ t : ℝ, ∃ x y z : ℝ, line_vector t = (x, y, z)) →
  line_vector (-1) = (2, 6, 16) →
  line_vector 1 = (1, 3, 8) →
  line_vector 4 = (-2, -6, -16) →
  line_vector 5 = (-4, -12, -8) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_at_5_l3196_319610


namespace NUMINAMATH_CALUDE_parabola_directrix_l3196_319685

/-- A parabola is defined by its equation relating x and y coordinates. -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry. -/
structure Directrix where
  equation : ℝ → ℝ → Prop

/-- For a parabola with equation x² = (1/4)y, its directrix has equation y = -1/16. -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  (∀ x y, p.equation x y ↔ x^2 = (1/4) * y) →
  (∀ x y, d.equation x y ↔ y = -1/16) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3196_319685


namespace NUMINAMATH_CALUDE_boxed_flowers_cost_l3196_319642

theorem boxed_flowers_cost (first_batch_total : ℕ) (second_batch_total : ℕ) 
  (second_batch_multiplier : ℕ) (price_difference : ℕ) :
  first_batch_total = 2000 →
  second_batch_total = 4200 →
  second_batch_multiplier = 3 →
  price_difference = 6 →
  ∃ (x : ℕ), 
    x * first_batch_total = second_batch_multiplier * second_batch_total * (x - price_difference) ∧
    x = 20 :=
by sorry

end NUMINAMATH_CALUDE_boxed_flowers_cost_l3196_319642


namespace NUMINAMATH_CALUDE_natasha_money_l3196_319657

/-- Represents the amount of money each person has -/
structure Money where
  cosima : ℕ
  carla : ℕ
  natasha : ℕ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.carla = 2 * m.cosima ∧
  m.natasha = 3 * m.carla ∧
  (7 * (m.cosima + m.carla + m.natasha) - 5 * (m.cosima + m.carla + m.natasha)) = 180

/-- The theorem to prove -/
theorem natasha_money (m : Money) : 
  problem_conditions m → m.natasha = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_natasha_money_l3196_319657


namespace NUMINAMATH_CALUDE_circle_equation_l3196_319606

/-- The equation of a circle passing through points A(1, -1) and B(-1, 1) with its center on the line x + y - 2 = 0 -/
theorem circle_equation :
  ∃ (h k : ℝ),
    (h + k - 2 = 0) ∧
    ((1 - h)^2 + (-1 - k)^2 = (h - 1)^2 + (k - 1)^2) ∧
    ((1 - h)^2 + (-1 - k)^2 = 4) ∧
    (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ↔ (x - 1)^2 + (y - 1)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3196_319606


namespace NUMINAMATH_CALUDE_todd_repayment_l3196_319695

/-- Calculates the amount Todd repaid his brother --/
def amount_repaid (loan : ℝ) (ingredients_cost : ℝ) (snow_cones_sold : ℕ) (price_per_snow_cone : ℝ) (remaining_money : ℝ) : ℝ :=
  (snow_cones_sold : ℝ) * price_per_snow_cone - ingredients_cost + loan - remaining_money

/-- Proves that Todd repaid his brother $110 --/
theorem todd_repayment : 
  amount_repaid 100 75 200 0.75 65 = 110 := by
  sorry

#eval amount_repaid 100 75 200 0.75 65

end NUMINAMATH_CALUDE_todd_repayment_l3196_319695


namespace NUMINAMATH_CALUDE_sally_cards_bought_l3196_319632

def cards_bought (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received)

theorem sally_cards_bought :
  cards_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_cards_bought_l3196_319632


namespace NUMINAMATH_CALUDE_remainder_theorem_l3196_319607

-- Define the polynomial x^3 + x^2 + x + 1
def f (x : ℂ) : ℂ := x^3 + x^2 + x + 1

-- Define the polynomial x^60 + x^45 + x^30 + x^15 + 1
def g (x : ℂ) : ℂ := x^60 + x^45 + x^30 + x^15 + 1

theorem remainder_theorem :
  ∃ (q : ℂ → ℂ), ∀ x, g x = f x * q x + 5 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3196_319607


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l3196_319650

theorem arithmetic_mean_greater_than_harmonic_mean 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) (hne : a ≠ b) : 
  (a + b) / 2 > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_harmonic_mean_l3196_319650


namespace NUMINAMATH_CALUDE_endpoint_coordinate_product_l3196_319629

/-- Given a line segment with midpoint (3, -5) and one endpoint at (7, -1),
    the product of the coordinates of the other endpoint is 9. -/
theorem endpoint_coordinate_product : 
  ∀ (x y : ℝ), 
  (3 = (x + 7) / 2) →  -- midpoint x-coordinate
  (-5 = (y + (-1)) / 2) →  -- midpoint y-coordinate
  x * y = 9 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_product_l3196_319629


namespace NUMINAMATH_CALUDE_lucy_father_age_twice_l3196_319684

theorem lucy_father_age_twice (lucy_birth_year father_birth_year : ℕ) 
  (h1 : lucy_birth_year = 2000) 
  (h2 : father_birth_year = 1960) : 
  ∃ (year : ℕ), year = 2040 ∧ 
  (year - father_birth_year = 2 * (year - lucy_birth_year)) :=
sorry

end NUMINAMATH_CALUDE_lucy_father_age_twice_l3196_319684


namespace NUMINAMATH_CALUDE_quadratic_function_value_at_three_l3196_319624

/-- A quadratic function f(x) = ax^2 + bx + c with roots at x=1 and x=5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value_at_three
  (a b c : ℝ)
  (root_one : QuadraticFunction a b c 1 = 0)
  (root_five : QuadraticFunction a b c 5 = 0)
  (min_value : ∀ x, QuadraticFunction a b c x ≥ 36)
  (attains_min : ∃ x, QuadraticFunction a b c x = 36) :
  QuadraticFunction a b c 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_at_three_l3196_319624


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3196_319627

theorem consecutive_odd_numbers_sum (k : ℤ) : 
  (2*k - 1) + (2*k + 1) + (2*k + 3) = (2*k - 1) + 128 → 2*k - 1 = 61 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3196_319627


namespace NUMINAMATH_CALUDE_cylinder_radius_approximation_l3196_319697

noncomputable def cylinder_radius (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  let cylinder_volume := 2 * rectangle_area
  let cylinder_height := square_side
  Real.sqrt (cylinder_volume / (Real.pi * cylinder_height))

theorem cylinder_radius_approximation :
  ∀ (ε : ℝ), ε > 0 →
  abs (cylinder_radius 2500 10 - 1.59514) < ε :=
sorry

end NUMINAMATH_CALUDE_cylinder_radius_approximation_l3196_319697


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3196_319604

/-- Calculates the total cost of car repair given the following parameters:
  * rate1: hourly rate of the first mechanic
  * hours1: hours worked per day by the first mechanic
  * days1: number of days worked by the first mechanic
  * rate2: hourly rate of the second mechanic
  * hours2: hours worked per day by the second mechanic
  * days2: number of days worked by the second mechanic
  * parts_cost: cost of parts used in the repair
-/
def total_repair_cost (rate1 hours1 days1 rate2 hours2 days2 parts_cost : ℕ) : ℕ :=
  rate1 * hours1 * days1 + rate2 * hours2 * days2 + parts_cost

/-- Theorem stating that the total repair cost for the given scenario is $14,420 -/
theorem repair_cost_calculation :
  total_repair_cost 60 8 14 75 6 10 3200 = 14420 := by
  sorry

#eval total_repair_cost 60 8 14 75 6 10 3200

end NUMINAMATH_CALUDE_repair_cost_calculation_l3196_319604


namespace NUMINAMATH_CALUDE_spinner_probability_theorem_l3196_319694

/-- Represents the probability of landing on each part of a circular spinner -/
structure SpinnerProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ
  D : ℚ

/-- Theorem: If a circular spinner has probabilities 1/4 for A, 1/3 for B, and 1/6 for D,
    then the probability for C is 1/4 -/
theorem spinner_probability_theorem (sp : SpinnerProbabilities) 
  (hA : sp.A = 1/4)
  (hB : sp.B = 1/3)
  (hD : sp.D = 1/6)
  (hSum : sp.A + sp.B + sp.C + sp.D = 1) :
  sp.C = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_theorem_l3196_319694


namespace NUMINAMATH_CALUDE_product_simplification_l3196_319662

theorem product_simplification :
  (240 : ℚ) / 18 * (9 : ℚ) / 160 * (10 : ℚ) / 3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3196_319662


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3196_319619

theorem decimal_to_fraction (n d : ℕ) (h : n = 16) :
  (n : ℚ) / d = 32 / 100 → d = 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3196_319619


namespace NUMINAMATH_CALUDE_sqrt_81_equals_9_l3196_319671

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_9_l3196_319671


namespace NUMINAMATH_CALUDE_coeff_x20_Q_greater_than_P_l3196_319645

-- Define the two expressions
def P (x : ℝ) : ℝ := (1 - x^2 + x^3)^1000
def Q (x : ℝ) : ℝ := (1 + x^2 - x^3)^1000

-- Define a function to get the coefficient of x^20 in a polynomial
noncomputable def coeff_x20 (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem coeff_x20_Q_greater_than_P :
  coeff_x20 Q > coeff_x20 P := by sorry

end NUMINAMATH_CALUDE_coeff_x20_Q_greater_than_P_l3196_319645


namespace NUMINAMATH_CALUDE_weight_of_four_moles_of_compound_l3196_319683

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℝ) (num_moles : ℝ) : ℝ :=
  molecular_weight * num_moles

/-- Theorem: The weight of 4 moles of a compound with molecular weight 312 g/mol is 1248 grams -/
theorem weight_of_four_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 312) : weight_of_moles molecular_weight 4 = 1248 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_four_moles_of_compound_l3196_319683


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3196_319682

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 4*y + 3 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3196_319682


namespace NUMINAMATH_CALUDE_complex_power_difference_zero_l3196_319602

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero :
  (1 + i)^24 - (1 - i)^24 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_zero_l3196_319602


namespace NUMINAMATH_CALUDE_compound_composition_l3196_319699

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := 1

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℚ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℚ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℚ := 16.00

/-- Total molecular weight of the compound in g/mol -/
def total_weight : ℚ := 122

/-- Number of Phosphorus atoms in the compound -/
def P_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 4

theorem compound_composition :
  n * Al_weight + P_count * P_weight + O_count * O_weight = total_weight :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l3196_319699


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3196_319677

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3196_319677


namespace NUMINAMATH_CALUDE_card_arrangement_exists_l3196_319646

/-- Represents a card with two sides, each containing a natural number -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Represents the set of n cards -/
def CardSet (n : Nat) := {cards : Finset Card // cards.card = n}

/-- Predicate to check if a set of cards satisfies the problem conditions -/
def ValidCardSet (n : Nat) (cards : CardSet n) : Prop :=
  (∀ i : Nat, i ∈ Finset.range n → (cards.val.filter (λ c => c.side1 = i + 1 ∨ c.side2 = i + 1)).card = 2) ∧
  (∀ c : Card, c ∈ cards.val → c.side1 ≤ n ∧ c.side2 ≤ n)

/-- Represents an arrangement of cards on the table -/
def Arrangement (n : Nat) := Fin n → Bool

/-- Predicate to check if an arrangement is valid (shows numbers 1 to n exactly once) -/
def ValidArrangement (n : Nat) (cards : CardSet n) (arr : Arrangement n) : Prop :=
  ∀ i : Fin n, ∃! c : Card, c ∈ cards.val ∧
    ((arr i = true ∧ c.side1 = i + 1) ∨ (arr i = false ∧ c.side2 = i + 1))

theorem card_arrangement_exists (n : Nat) (cards : CardSet n) 
  (h : ValidCardSet n cards) : ∃ arr : Arrangement n, ValidArrangement n cards arr := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_exists_l3196_319646


namespace NUMINAMATH_CALUDE_monika_movies_l3196_319647

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def bean_bags : ℕ := 20
def bean_cost : ℝ := 1.25
def total_spent : ℝ := 347

theorem monika_movies :
  (total_spent - (mall_expense + bean_bags * bean_cost)) / movie_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_monika_movies_l3196_319647


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3196_319660

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 - 2*m*(1 : ℝ) + 1 = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3196_319660


namespace NUMINAMATH_CALUDE_monomial_like_terms_sum_l3196_319635

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c₁ c₂ : ℚ, a x y = c₁ ∧ b x y = c₂

theorem monomial_like_terms_sum (m n : ℕ) :
  like_terms (fun x y => 5 * x^m * y) (fun x y => -3 * x^2 * y^n) →
  m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_monomial_like_terms_sum_l3196_319635


namespace NUMINAMATH_CALUDE_book_cost_problem_l3196_319693

/-- Proves that given two books with a total cost of 480, where one is sold at a 15% loss 
and the other at a 19% gain, and both are sold at the same price, 
the cost of the book sold at a loss is 280. -/
theorem book_cost_problem (c1 c2 : ℝ) : 
  c1 + c2 = 480 →
  c1 * 0.85 = c2 * 1.19 →
  c1 = 280 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3196_319693


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_conditional_l3196_319605

-- Define p and q as propositions
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def is_sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem sufficient_condition_implies_conditional 
  (h : is_sufficient_condition p q) : (p → q) = True :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_conditional_l3196_319605


namespace NUMINAMATH_CALUDE_mary_took_three_crayons_l3196_319625

/-- Given an initial number of crayons and the number left after some are taken,
    calculate the number of crayons taken. -/
def crayons_taken (initial : ℕ) (left : ℕ) : ℕ := initial - left

theorem mary_took_three_crayons :
  let initial_crayons : ℕ := 7
  let crayons_left : ℕ := 4
  crayons_taken initial_crayons crayons_left = 3 := by
sorry

end NUMINAMATH_CALUDE_mary_took_three_crayons_l3196_319625


namespace NUMINAMATH_CALUDE_stock_market_value_l3196_319612

/-- Given a stock with a 10% dividend rate and an 8% yield, its market value is $125. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) : 
  dividend_rate = 0.1 → yield = 0.08 → (dividend_rate * face_value) / yield = 125 := by
  sorry

end NUMINAMATH_CALUDE_stock_market_value_l3196_319612


namespace NUMINAMATH_CALUDE_competition_outcomes_l3196_319639

/-- The number of participants in the competition -/
def n : ℕ := 6

/-- The number of places to be filled (1st, 2nd, 3rd) -/
def k : ℕ := 3

/-- The number of different ways to arrange k distinct items from a set of n distinct items -/
def arrangement_count (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem competition_outcomes :
  arrangement_count n k = 120 :=
sorry

end NUMINAMATH_CALUDE_competition_outcomes_l3196_319639


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l3196_319673

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : Nat
  seated_people : Nat

/-- Checks if a seating arrangement is valid (no isolated seats). -/
def is_valid_seating (table : CircularTable) : Prop :=
  ∀ n : Nat, n < table.total_chairs → 
    ∃ m : Nat, m < table.seated_people ∧ 
      (n = m ∨ n = (m + 1) % table.total_chairs ∨ n = (m - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem to be proved. -/
theorem smallest_valid_seating :
  ∀ table : CircularTable, 
    table.total_chairs = 60 →
    (is_valid_seating table ↔ table.seated_people ≥ 15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l3196_319673


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3196_319601

/-- The cost of a candy bar given the conditions -/
theorem candy_bar_cost : 
  ∀ (candy_cost chocolate_cost : ℝ),
  candy_cost + chocolate_cost = 3 →
  candy_cost = chocolate_cost + 3 →
  candy_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3196_319601


namespace NUMINAMATH_CALUDE_max_y_value_l3196_319623

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  y ≤ 24 ∧ ∃ (x₀ : ℤ), x₀ * 24 + 6 * x₀ + 5 * 24 = -6 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3196_319623


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3196_319649

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x : ℝ) => (3/2) * x
  let intersection_projection_is_focus := 
    ∃ (x y : ℝ), hyperbola x y ∧ y = line x ∧ 
    (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ x = c)
  2

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3196_319649


namespace NUMINAMATH_CALUDE_candy_duration_l3196_319680

theorem candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) :
  neighbors_candy = 66 →
  sister_candy = 15 →
  daily_consumption = 9 →
  (neighbors_candy + sister_candy) / daily_consumption = 9 :=
by sorry

end NUMINAMATH_CALUDE_candy_duration_l3196_319680


namespace NUMINAMATH_CALUDE_alan_cd_cost_l3196_319628

/-- The total cost of CDs Alan buys -/
def total_cost (price_avn : ℝ) (num_dark : ℕ) (num_90s : ℕ) : ℝ :=
  let price_dark := 2 * price_avn
  let cost_dark := num_dark * price_dark
  let cost_avn := price_avn
  let cost_others := cost_dark + cost_avn
  let cost_90s := 0.4 * cost_others
  cost_dark + cost_avn + cost_90s

/-- Theorem stating the total cost of Alan's CD purchase -/
theorem alan_cd_cost :
  total_cost 12 2 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_alan_cd_cost_l3196_319628


namespace NUMINAMATH_CALUDE_find_p_l3196_319614

theorem find_p (a b p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : a^2 - 5*p*a + 2*p^3 = 0)
  (h2 : b^2 - 5*p*b + 2*p^3 = 0)
  (h3 : ∃! x, x^2 - a*x + b = 0) :
  p = 3 := by sorry

end NUMINAMATH_CALUDE_find_p_l3196_319614


namespace NUMINAMATH_CALUDE_square_plus_one_ge_double_abs_l3196_319637

theorem square_plus_one_ge_double_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_double_abs_l3196_319637


namespace NUMINAMATH_CALUDE_some_magical_creatures_are_mystical_beings_l3196_319613

-- Define our sets
variable (U : Type) -- Universe set
variable (D : Set U) -- Set of dragons
variable (M : Set U) -- Set of magical creatures
variable (B : Set U) -- Set of mystical beings

-- Define our premises
variable (h1 : D ⊆ M) -- All dragons are magical creatures
variable (h2 : ∃ x, x ∈ B ∩ D) -- Some mystical beings are dragons

-- Theorem to prove
theorem some_magical_creatures_are_mystical_beings : 
  ∃ x, x ∈ M ∩ B := by sorry

end NUMINAMATH_CALUDE_some_magical_creatures_are_mystical_beings_l3196_319613


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l3196_319696

/-- Proves that the rate of a stream is 5 km/hr given the boat's speed in still water,
    distance traveled downstream, and time taken. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 16 →
  distance = 84 →
  time = 4 →
  ∃ stream_rate : ℝ, 
    stream_rate = 5 ∧
    distance = (boat_speed + stream_rate) * time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_rate_calculation_l3196_319696


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3196_319618

/-- A quadratic function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a * x + b

/-- The composition of f with itself -/
def f_comp (a b x : ℝ) : ℝ := f a b (f a b x)

/-- Theorem: If f(f(x)) = 0 has four distinct real solutions and
    the sum of two of these solutions is -1, then b ≤ -1/4 -/
theorem quadratic_inequality (a b : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f_comp a b w = 0 ∧ f_comp a b x = 0 ∧ f_comp a b y = 0 ∧ f_comp a b z = 0) →
  (∃ p q : ℝ, f_comp a b p = 0 ∧ f_comp a b q = 0 ∧ p + q = -1) →
  b ≤ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3196_319618


namespace NUMINAMATH_CALUDE_greatest_n_with_222_digits_l3196_319679

def a (n : ℕ) : ℚ := (2 * 10^(n+1) - 20 - 18*n) / 81

def number_of_digits (q : ℚ) : ℕ := sorry

theorem greatest_n_with_222_digits : 
  ∃ (n : ℕ), (∀ m : ℕ, number_of_digits (a m) = 222 → m ≤ n) ∧ 
  number_of_digits (a n) = 222 ∧ n = 222 := by sorry

end NUMINAMATH_CALUDE_greatest_n_with_222_digits_l3196_319679


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3196_319611

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l3196_319611


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3196_319672

theorem simplify_sqrt_expression :
  Real.sqrt (37 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3196_319672


namespace NUMINAMATH_CALUDE_root_product_equality_l3196_319692

-- Define the quadratic equations
def quadratic1 (x p c : ℝ) : ℝ := x^2 + p*x + c
def quadratic2 (x q c : ℝ) : ℝ := x^2 + q*x + c

-- Define the theorem
theorem root_product_equality (p q c : ℝ) (α β γ δ : ℝ) :
  quadratic1 α p c = 0 →
  quadratic1 β p c = 0 →
  quadratic2 γ q c = 0 →
  quadratic2 δ q c = 0 →
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = (p^2 - q^2) * c + c^2 - p*c - q*c :=
by sorry

end NUMINAMATH_CALUDE_root_product_equality_l3196_319692


namespace NUMINAMATH_CALUDE_rectangle_composition_l3196_319626

theorem rectangle_composition (total_width total_height : ℕ) 
  (h_width : total_width = 3322) (h_height : total_height = 2020) : ∃ (r s : ℕ),
  2 * r + s = total_height ∧ 2 * r + 3 * s = total_width ∧ s = 651 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_composition_l3196_319626


namespace NUMINAMATH_CALUDE_problem_solution_l3196_319652

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem statement
theorem problem_solution : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3196_319652


namespace NUMINAMATH_CALUDE_min_value_sum_l3196_319633

theorem min_value_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (a * b)) :
  ∀ x y, x > 0 → y > 0 → 2 = Real.sqrt (x * y) → a + 4 * b ≤ x + 4 * y :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3196_319633


namespace NUMINAMATH_CALUDE_mango_purchase_amount_l3196_319609

-- Define the variables
def grapes_kg : ℕ := 3
def grapes_rate : ℕ := 70
def mango_rate : ℕ := 55
def total_paid : ℕ := 705

-- Define the theorem
theorem mango_purchase_amount :
  ∃ (m : ℕ), 
    grapes_kg * grapes_rate + m * mango_rate = total_paid ∧ 
    m = 9 :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_amount_l3196_319609


namespace NUMINAMATH_CALUDE_parabola_translation_l3196_319686

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 5 0 0
  let translated := translate original 2 3
  (5 * x^2) + 3 = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3196_319686


namespace NUMINAMATH_CALUDE_bianca_albums_l3196_319644

theorem bianca_albums (total_pics : ℕ) (main_album_pics : ℕ) (pics_per_album : ℕ) : 
  total_pics = 33 → main_album_pics = 27 → pics_per_album = 2 → 
  (total_pics - main_album_pics) / pics_per_album = 3 := by
  sorry

end NUMINAMATH_CALUDE_bianca_albums_l3196_319644


namespace NUMINAMATH_CALUDE_circles_intersect_and_common_chord_l3196_319670

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the intersection of the circles
def intersect : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y

-- Define the common chord equation
def commonChord (x y : ℝ) : Prop := 3*x - 2*y = 0

theorem circles_intersect_and_common_chord :
  intersect ∧ (∀ x y : ℝ, circle1 x y ∧ circle2 x y → commonChord x y) :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_and_common_chord_l3196_319670


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3196_319634

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (3, -1)

/-- Predicate to check if a point is on the parabola -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Predicate to check if a point is on the x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0

/-- Definition of a square inscribed in the parabola region -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (vertex_on_parabola : on_parabola (center.1 - side_length/2, center.2 - side_length/2))
  (bottom_left_on_x_axis : on_x_axis (center.1 - side_length/2, center.2 + side_length/2))
  (bottom_right_on_x_axis : on_x_axis (center.1 + side_length/2, center.2 + side_length/2))
  (top_right_on_parabola : on_parabola (center.1 + side_length/2, center.2 - side_length/2))

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side_length^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3196_319634


namespace NUMINAMATH_CALUDE_correct_regression_equation_l3196_319688

/-- Represents the selling price of a product in yuan per piece -/
def SellingPrice : Type := ℝ

/-- Represents the sales volume of a product in pieces -/
def SalesVolume : Type := ℝ

/-- Represents a regression equation for sales volume based on selling price -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Indicates that two variables are negatively correlated -/
def NegativelyCorrelated (x : Type) (y : Type) : Prop := sorry

/-- Checks if a regression equation is valid for negatively correlated variables -/
def IsValidRegression (eq : RegressionEquation) (x : Type) (y : Type) : Prop := 
  NegativelyCorrelated x y → eq.slope < 0

/-- The correct regression equation for the given problem -/
def CorrectEquation : RegressionEquation := { slope := -2, intercept := 100 }

/-- Theorem stating that the CorrectEquation is valid for the given problem -/
theorem correct_regression_equation : 
  IsValidRegression CorrectEquation SellingPrice SalesVolume := sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l3196_319688


namespace NUMINAMATH_CALUDE_egg_count_l3196_319676

theorem egg_count (initial_eggs used_eggs chickens eggs_per_chicken : ℕ) :
  initial_eggs ≥ used_eggs →
  (initial_eggs - used_eggs) + chickens * eggs_per_chicken =
  initial_eggs - used_eggs + chickens * eggs_per_chicken :=
by sorry

end NUMINAMATH_CALUDE_egg_count_l3196_319676
