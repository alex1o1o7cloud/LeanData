import Mathlib

namespace NUMINAMATH_CALUDE_angle_B_measure_l89_8959

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the measure of an angle in a quadrilateral
def angle_measure (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- Theorem statement
theorem angle_B_measure (q : Quadrilateral) :
  angle_measure q 0 + angle_measure q 2 = 100 →
  angle_measure q 1 = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l89_8959


namespace NUMINAMATH_CALUDE_card_division_impossibility_l89_8900

theorem card_division_impossibility :
  ∀ k : ℕ,
  ¬ (2016 - 3 * k = 3 * (3 * k + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_card_division_impossibility_l89_8900


namespace NUMINAMATH_CALUDE_farm_section_area_l89_8973

/-- Given a farm with a total area of 300 acres divided into 5 equal sections,
    prove that the area of each section is 60 acres. -/
theorem farm_section_area (total_area : ℝ) (num_sections : ℕ) (section_area : ℝ) :
  total_area = 300 ∧ num_sections = 5 ∧ section_area * num_sections = total_area →
  section_area = 60 := by
  sorry

end NUMINAMATH_CALUDE_farm_section_area_l89_8973


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l89_8923

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l89_8923


namespace NUMINAMATH_CALUDE_cubic_factorization_l89_8988

theorem cubic_factorization (m : ℝ) : m^3 - 16*m = m*(m+4)*(m-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l89_8988


namespace NUMINAMATH_CALUDE_unit_circle_dot_product_l89_8948

theorem unit_circle_dot_product 
  (x₁ y₁ x₂ y₂ θ : ℝ) 
  (h₁ : x₁^2 + y₁^2 = 1) 
  (h₂ : x₂^2 + y₂^2 = 1)
  (h₃ : π/2 < θ ∧ θ < π)
  (h₄ : Real.sin (θ + π/4) = 3/5) : 
  x₁ * x₂ + y₁ * y₂ = -Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_dot_product_l89_8948


namespace NUMINAMATH_CALUDE_polynomial_rearrangement_l89_8956

theorem polynomial_rearrangement (x : ℝ) : 
  x^4 + 2*x^3 - 3*x^2 - 4*x + 1 = (x+1)^4 - 2*(x+1)^3 - 3*(x+1)^2 + 4*(x+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_rearrangement_l89_8956


namespace NUMINAMATH_CALUDE_difference_of_squares_numbers_l89_8924

def is_difference_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > b ∧ n = a * a - b * b

theorem difference_of_squares_numbers :
  is_difference_of_squares 2020 ∧
  is_difference_of_squares 2022 ∧
  is_difference_of_squares 2023 ∧
  is_difference_of_squares 2024 ∧
  ¬is_difference_of_squares 2021 :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_numbers_l89_8924


namespace NUMINAMATH_CALUDE_gmat_score_difference_l89_8907

theorem gmat_score_difference (x y : ℝ) (h1 : x > y) (h2 : x / y = 4) :
  x - y = 3 * y := by
sorry

end NUMINAMATH_CALUDE_gmat_score_difference_l89_8907


namespace NUMINAMATH_CALUDE_valid_arrangements_l89_8998

/-- The number of ways to arrange plates on a circular table. -/
def circularArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + green + orange + yellow))

/-- The number of arrangements with green plates adjacent. -/
def greenAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + orange + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   Nat.factorial orange * Nat.factorial yellow * 
   (blue + red + 1 + orange + yellow))

/-- The number of arrangements with orange plates adjacent. -/
def orangeAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + green + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * 
   Nat.factorial 1 * Nat.factorial yellow * 
   (blue + red + green + 1 + yellow))

/-- The number of arrangements with both green and orange plates adjacent. -/
def bothAdjacentArrangements (blue red green orange yellow : ℕ) : ℕ :=
  (Nat.factorial (blue + red + 1 + 1 + yellow)) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * 
   (blue + red + 1 + 1 + yellow))

/-- The main theorem stating the number of valid arrangements. -/
theorem valid_arrangements (blue red green orange yellow : ℕ) 
  (h_blue : blue = 6) (h_red : red = 3) (h_green : green = 3) 
  (h_orange : orange = 2) (h_yellow : yellow = 1) :
  circularArrangements blue red green orange yellow - 
  (greenAdjacentArrangements blue red green orange yellow + 
   orangeAdjacentArrangements blue red green orange yellow - 
   bothAdjacentArrangements blue red green orange yellow) =
  circularArrangements 6 3 3 2 1 - 
  (greenAdjacentArrangements 6 3 3 2 1 + 
   orangeAdjacentArrangements 6 3 3 2 1 - 
   bothAdjacentArrangements 6 3 3 2 1) := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l89_8998


namespace NUMINAMATH_CALUDE_odd_function_product_negative_l89_8953

theorem odd_function_product_negative (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_product_negative_l89_8953


namespace NUMINAMATH_CALUDE_mean_age_is_eleven_l89_8952

/-- Represents the ages of children in the Euler family -/
def euler_ages : List ℕ := [10, 12, 8]

/-- Represents the ages of children in the Gauss family -/
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]

/-- Calculates the mean age of all children from both families -/
def mean_age : ℚ := (euler_ages.sum + gauss_ages.sum) / (euler_ages.length + gauss_ages.length)

theorem mean_age_is_eleven : mean_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_age_is_eleven_l89_8952


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l89_8939

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, -2; 0, 5]) : 
  (B^2)⁻¹ = !![9, -16; 0, 25] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l89_8939


namespace NUMINAMATH_CALUDE_spinner_probability_l89_8913

def spinner1 : Finset ℕ := {2, 4, 5, 7, 9}
def spinner2 : Finset ℕ := {3, 4, 6, 8, 10, 12}

def isEven (n : ℕ) : Bool := n % 2 = 0

def productIsEven (x : ℕ) (y : ℕ) : Bool := isEven (x * y)

def favorableOutcomes : ℕ := (spinner1.card * spinner2.card) - 
  (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card

theorem spinner_probability : 
  (favorableOutcomes : ℚ) / (spinner1.card * spinner2.card) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l89_8913


namespace NUMINAMATH_CALUDE_fraction_equality_l89_8975

theorem fraction_equality (p q : ℚ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l89_8975


namespace NUMINAMATH_CALUDE_fare_comparison_l89_8946

/-- Fare calculation for city A -/
def fareA (x : ℝ) : ℝ := 10 + 2 * (x - 3)

/-- Fare calculation for city B -/
def fareB (x : ℝ) : ℝ := 8 + 2.5 * (x - 3)

/-- Theorem stating the condition for city A's fare to be higher than city B's -/
theorem fare_comparison (x : ℝ) :
  x > 3 → (fareA x > fareB x ↔ 3 < x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_fare_comparison_l89_8946


namespace NUMINAMATH_CALUDE_reciprocal_power_l89_8906

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_l89_8906


namespace NUMINAMATH_CALUDE_two_reciprocal_sets_l89_8967

-- Define a reciprocal set
def ReciprocalSet (A : Set ℝ) : Prop :=
  A.Nonempty ∧ (0 ∉ A) ∧ ∀ x ∈ A, (1 / x) ∈ A

-- Define the three sets
def Set1 (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

def Set2 : Set ℝ := {x : ℝ | x^2 - 4*x + 1 < 0}

def Set3 : Set ℝ := {y : ℝ | ∃ x : ℝ, 
  (0 ≤ x ∧ x < 1 ∧ y = 2*x + 2/5) ∨ 
  (1 ≤ x ∧ x ≤ 2 ∧ y = x + 1/x)}

-- Theorem to prove
theorem two_reciprocal_sets : 
  ∃ (a : ℝ), (ReciprocalSet (Set2) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set1 a)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set2) ∧ ¬ReciprocalSet (Set3)) ∨
             (ReciprocalSet (Set1 a) ∧ ReciprocalSet (Set3) ∧ ¬ReciprocalSet (Set2)) :=
sorry

end NUMINAMATH_CALUDE_two_reciprocal_sets_l89_8967


namespace NUMINAMATH_CALUDE_linear_function_properties_l89_8994

/-- A linear function f(x) = k(x + 2) where k ≠ 0 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 2)

/-- g is f shifted 2 units upwards -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-2) = 0) ∧
  (g k 1 = -2 → k = -4/3) ∧
  (0 > f k 0 ∧ f k 0 > -2 → -1 < k ∧ k < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l89_8994


namespace NUMINAMATH_CALUDE_proportional_equation_inequality_l89_8933

theorem proportional_equation_inequality (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  b / d = c / a → ¬(a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_proportional_equation_inequality_l89_8933


namespace NUMINAMATH_CALUDE_chef_cakes_problem_l89_8980

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cakes_problem_l89_8980


namespace NUMINAMATH_CALUDE_largest_divisor_when_square_divisible_by_50_l89_8931

theorem largest_divisor_when_square_divisible_by_50 (n : ℕ) (h1 : n > 0) (h2 : 50 ∣ n^2) :
  ∃ (d : ℕ), d ∣ n ∧ d = 10 ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_when_square_divisible_by_50_l89_8931


namespace NUMINAMATH_CALUDE_tarts_distribution_l89_8968

/-- Represents the number of tarts eaten by a child in a 10-minute interval -/
structure EatingRate :=
  (tarts : ℕ)

/-- Represents the total eating time in minutes -/
def total_time : ℕ := 90

/-- Represents the total number of tarts eaten -/
def total_tarts : ℕ := 35

/-- Zhenya's eating rate -/
def zhenya_rate : EatingRate := ⟨5⟩

/-- Sasha's eating rate -/
def sasha_rate : EatingRate := ⟨3⟩

/-- Calculates the number of tarts eaten by a child given their eating rate and number of 10-minute intervals -/
def tarts_eaten (rate : EatingRate) (intervals : ℕ) : ℕ := rate.tarts * intervals

/-- The main theorem to prove -/
theorem tarts_distribution :
  ∃ (zhenya_intervals sasha_intervals : ℕ),
    zhenya_intervals + sasha_intervals = total_time / 10 ∧
    tarts_eaten zhenya_rate zhenya_intervals + tarts_eaten sasha_rate sasha_intervals = total_tarts ∧
    tarts_eaten zhenya_rate zhenya_intervals = 20 ∧
    tarts_eaten sasha_rate sasha_intervals = 15 :=
sorry

end NUMINAMATH_CALUDE_tarts_distribution_l89_8968


namespace NUMINAMATH_CALUDE_susan_walk_distance_l89_8904

theorem susan_walk_distance (total_distance : ℝ) (difference : ℝ) :
  total_distance = 15 ∧ difference = 3 →
  ∃ susan_distance erin_distance : ℝ,
    susan_distance + erin_distance = total_distance ∧
    erin_distance = susan_distance - difference ∧
    susan_distance = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l89_8904


namespace NUMINAMATH_CALUDE_factorization_of_4m_squared_minus_64_l89_8930

theorem factorization_of_4m_squared_minus_64 (m : ℝ) : 4 * m^2 - 64 = 4 * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4m_squared_minus_64_l89_8930


namespace NUMINAMATH_CALUDE_equation_solution_l89_8992

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 3 ∧ x + 25 / (x - 3) = -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l89_8992


namespace NUMINAMATH_CALUDE_product_remainder_l89_8958

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem product_remainder (a₁ : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) :
  a₁ = 2 → d = 10 → n = 21 → m = 7 →
  (arithmetic_sequence a₁ d n).prod % m = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l89_8958


namespace NUMINAMATH_CALUDE_pascal_triangle_48th_number_l89_8921

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of elements in the row of Pascal's triangle -/
def row_size : ℕ := 51

/-- The index of the number we're looking for in the row -/
def target_index : ℕ := 48

/-- The theorem stating that the 48th number in the row with 51 numbers 
    of Pascal's triangle is 19600 -/
theorem pascal_triangle_48th_number : 
  binomial (row_size - 1) (target_index - 1) = 19600 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_48th_number_l89_8921


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l89_8957

theorem x_eq_3_sufficient_not_necessary_for_x_sq_eq_9 :
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  (∀ x : ℝ, x = 3 → x^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l89_8957


namespace NUMINAMATH_CALUDE_emily_fishing_catch_l89_8901

/-- The total weight of fish Emily caught -/
def total_fish_weight (trout_count catfish_count bluegill_count sunfish_count bass_count : ℕ)
  (trout_weight catfish_weight bluegill_weight sunfish_weight bass_weight : ℝ) : ℝ :=
  trout_count * trout_weight +
  catfish_count * catfish_weight +
  bluegill_count * bluegill_weight +
  sunfish_count * sunfish_weight +
  bass_count * bass_weight

/-- Emily's fishing catch theorem -/
theorem emily_fishing_catch :
  total_fish_weight 4 3 5 6 2 2.3 1.5 2.5 1.75 3.8 = 44.3 := by
  sorry

end NUMINAMATH_CALUDE_emily_fishing_catch_l89_8901


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l89_8966

/-- Represent a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The problem statement --/
theorem least_sum_of_bases : 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ 
  baseRepresentation 58 c = baseRepresentation 85 d ∧
  (∀ (c' d' : ℕ), c' > 0 → d' > 0 → 
    baseRepresentation 58 c' = baseRepresentation 85 d' → 
    c + d ≤ c' + d') ∧
  c + d = 15 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l89_8966


namespace NUMINAMATH_CALUDE_equation_solutions_l89_8944

theorem equation_solutions :
  (∃ x : ℝ, x + 2*x = 12.6 ∧ x = 4.2) ∧
  (∃ x : ℝ, (1/4)*x + 1/2 = 3/5 ∧ x = 2/5) ∧
  (∃ x : ℝ, x + 1.3*x = 46 ∧ x = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l89_8944


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l89_8989

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem stating the ratio of rectangle sides given the square configuration -/
theorem rectangle_side_ratio
  (config : RectangleSquareConfig)
  (h1 : config.inner_square_side + 2 * config.rectangle_short_side = 2 * config.inner_square_side)
  (h2 : config.rectangle_long_side + config.inner_square_side = 2 * config.inner_square_side)
  (h3 : (2 * config.inner_square_side) ^ 2 = 4 * config.inner_square_side ^ 2) :
  config.rectangle_long_side / config.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l89_8989


namespace NUMINAMATH_CALUDE_square_sum_identity_l89_8991

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l89_8991


namespace NUMINAMATH_CALUDE_second_train_speed_l89_8936

/-- Proves that the speed of the second train is 120 kmph given the conditions of the problem -/
theorem second_train_speed 
  (first_train_departure : ℕ) -- Departure time of the first train in hours after midnight
  (second_train_departure : ℕ) -- Departure time of the second train in hours after midnight
  (first_train_speed : ℝ) -- Speed of the first train in kmph
  (meeting_distance : ℝ) -- Distance where the trains meet in km
  (h1 : first_train_departure = 9) -- First train leaves at 9 a.m.
  (h2 : second_train_departure = 15) -- Second train leaves at 3 p.m.
  (h3 : first_train_speed = 30) -- First train speed is 30 kmph
  (h4 : meeting_distance = 720) -- Trains meet 720 km away from Delhi
  : ∃ (second_train_speed : ℝ), second_train_speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l89_8936


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l89_8950

/-- A complex number is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- Given complex number z -/
def z : ℂ := 1 - 2 * Complex.I

/-- Theorem: The complex number z = 1 - 2i is in the fourth quadrant -/
theorem z_in_fourth_quadrant : in_fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l89_8950


namespace NUMINAMATH_CALUDE_second_worker_loading_time_l89_8942

/-- The time it takes for the second worker to load one truck alone, given that:
    1. The first worker can load one truck in 5 hours
    2. Both workers together can load one truck in approximately 2.2222222222222223 hours
-/
theorem second_worker_loading_time :
  let first_worker_time : ℝ := 5
  let combined_time : ℝ := 2.2222222222222223
  let second_worker_time : ℝ := (first_worker_time * combined_time) / (first_worker_time - combined_time)
  ‖second_worker_time - 1.4285714285714286‖ < 0.0001 := by
sorry


end NUMINAMATH_CALUDE_second_worker_loading_time_l89_8942


namespace NUMINAMATH_CALUDE_a_parallel_b_iff_m_eq_one_l89_8981

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- The vectors a and b -/
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

/-- Theorem stating that a is parallel to b if and only if m = 1 -/
theorem a_parallel_b_iff_m_eq_one :
  ∀ m : ℝ, parallel a (b m) ↔ m = 1 := by sorry

end NUMINAMATH_CALUDE_a_parallel_b_iff_m_eq_one_l89_8981


namespace NUMINAMATH_CALUDE_heathers_weight_l89_8985

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) :
  emily_weight = 9 →
  weight_difference = 78 →
  emily_weight + weight_difference = 87 :=
by sorry

end NUMINAMATH_CALUDE_heathers_weight_l89_8985


namespace NUMINAMATH_CALUDE_sarahs_age_l89_8983

theorem sarahs_age (ana billy mark sarah : ℕ) : 
  ana + 3 = 15 →
  billy = ana / 2 →
  mark = billy + 4 →
  sarah = 3 * mark - 4 →
  sarah = 26 := by
sorry

end NUMINAMATH_CALUDE_sarahs_age_l89_8983


namespace NUMINAMATH_CALUDE_ski_price_after_discounts_l89_8954

def original_price : ℝ := 200
def discount1 : ℝ := 0.40
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.10

theorem ski_price_after_discounts :
  let price1 := original_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let final_price := price2 * (1 - discount3)
  final_price = 86.40 := by sorry

end NUMINAMATH_CALUDE_ski_price_after_discounts_l89_8954


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l89_8902

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l89_8902


namespace NUMINAMATH_CALUDE_square_difference_l89_8970

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l89_8970


namespace NUMINAMATH_CALUDE_nancy_future_games_l89_8908

/-- The number of games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_future_games : games_next_month 9 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_future_games_l89_8908


namespace NUMINAMATH_CALUDE_sequence_double_plus_one_greater_l89_8914

/-- Definition of the property $\{a_n\} > M$ -/
def sequence_greater_than (a : ℕ → ℝ) (M : ℝ) : Prop :=
  ∀ n : ℕ, a n ≥ M ∨ a (n + 1) ≥ M

/-- Main theorem -/
theorem sequence_double_plus_one_greater (a : ℕ → ℝ) (M : ℝ) :
  sequence_greater_than a M → sequence_greater_than (fun n ↦ 2 * a n + 1) (2 * M + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_double_plus_one_greater_l89_8914


namespace NUMINAMATH_CALUDE_polygon_sides_l89_8972

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l89_8972


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l89_8938

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10 * a * b) :
  |((a + b) / (a - b))| = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l89_8938


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l89_8912

/-- Given a bus that travels at 45 km/hr including stoppages and stops for 10 minutes per hour,
    prove that its speed excluding stoppages is 54 km/hr. -/
theorem bus_speed_excluding_stoppages :
  let speed_with_stoppages : ℝ := 45
  let stop_time_per_hour : ℝ := 10 / 60
  let travel_time_per_hour : ℝ := 1 - stop_time_per_hour
  let speed_without_stoppages : ℝ := speed_with_stoppages / travel_time_per_hour
  speed_without_stoppages = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l89_8912


namespace NUMINAMATH_CALUDE_friendship_fraction_l89_8909

theorem friendship_fraction :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 →
    (1 : ℚ) / 3 * y = (2 : ℚ) / 5 * x →
    ((1 : ℚ) / 3 * y + (2 : ℚ) / 5 * x) / (x + y : ℚ) = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_friendship_fraction_l89_8909


namespace NUMINAMATH_CALUDE_system_solution_l89_8987

theorem system_solution : ∃ (x y : ℝ), (x + y = 4 ∧ x - 2*y = 1) ∧ (x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l89_8987


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l89_8965

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l89_8965


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l89_8962

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 3
  let c : ℝ := -1
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 3*x₁ - 1 = 0 ∧ x₂^2 + 3*x₂ - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l89_8962


namespace NUMINAMATH_CALUDE_problem_solving_distribution_l89_8951

theorem problem_solving_distribution (x y z : ℕ) : 
  x + y + z = 100 →  -- Total problems
  x + 2*y + 3*z = 180 →  -- Sum of problems solved by each person
  x - z = 20  -- Difference between difficult and easy problems
:= by sorry

end NUMINAMATH_CALUDE_problem_solving_distribution_l89_8951


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_12_l89_8964

theorem no_linear_term_implies_m_equals_12 (m : ℝ) : 
  (∃ a b c : ℝ, (mx + 8) * (2 - 3*x) = a*x^2 + b*x + c ∧ b = 0) → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_12_l89_8964


namespace NUMINAMATH_CALUDE_joseph_drives_one_mile_more_than_kyle_l89_8945

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Joseph's driving speed in mph -/
def joseph_speed : ℝ := 50

/-- Joseph's driving time in hours -/
def joseph_time : ℝ := 2.5

/-- Kyle's driving speed in mph -/
def kyle_speed : ℝ := 62

/-- Kyle's driving time in hours -/
def kyle_time : ℝ := 2

theorem joseph_drives_one_mile_more_than_kyle :
  distance joseph_speed joseph_time - distance kyle_speed kyle_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_joseph_drives_one_mile_more_than_kyle_l89_8945


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l89_8961

theorem cubic_equation_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2020*a^2 + 1010 = 0 →
  b^3 - 2020*b^2 + 1010 = 0 →
  c^3 - 2020*c^2 + 1010 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l89_8961


namespace NUMINAMATH_CALUDE_class_average_weight_l89_8915

/-- Given two sections A and B in a class, calculate the average weight of the whole class. -/
theorem class_average_weight 
  (students_A : ℕ) 
  (students_B : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) : 
  students_A = 50 → 
  students_B = 40 → 
  avg_weight_A = 50 → 
  avg_weight_B = 70 → 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 
    (50 * 50 + 70 * 40) / (50 + 40) := by
  sorry

#eval (50 * 50 + 70 * 40) / (50 + 40)  -- This will evaluate to approximately 58.89

end NUMINAMATH_CALUDE_class_average_weight_l89_8915


namespace NUMINAMATH_CALUDE_point_b_value_l89_8903

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moving right on a number line -/
def moveRight (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_b_value (a b : Point) (h1 : a.value = -3) (h2 : b = moveRight a 4) :
  b.value = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l89_8903


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l89_8993

-- Problem 1
theorem problem_1 : 123^2 - 124 * 122 = 1 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (-2*a^2*b)^3 / (-a*b) * (1/2*a^2*b)^3 = a^11*b^5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l89_8993


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l89_8918

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 5 →
  let total_age := team_size * (captain_age + 7)
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + (captain_age + wicket_keeper_age_diff))
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l89_8918


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l89_8949

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

def center1 : ℝ × ℝ := (0, 3)
def center2 : ℝ × ℝ := (4, 0)

def radius1 : ℝ := 3
def radius2 : ℝ := 2

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l89_8949


namespace NUMINAMATH_CALUDE_candle_illumination_theorem_l89_8929

/-- Represents a wall in a room -/
structure Wall where
  -- Add necessary properties for a wall

/-- Represents a candle in a room -/
structure Candle where
  -- Add necessary properties for a candle

/-- Represents a room with walls and a candle -/
structure Room where
  walls : List Wall
  candle : Candle

/-- Predicate to check if a wall is completely illuminated by a candle -/
def is_completely_illuminated (w : Wall) (c : Candle) : Prop :=
  sorry

/-- Theorem stating that for a room with n walls (where n is 10 or 6),
    there exists a configuration where a single candle can be placed
    such that no wall is completely illuminated -/
theorem candle_illumination_theorem (n : Nat) (h : n = 10 ∨ n = 6) :
  ∃ (r : Room), r.walls.length = n ∧ ∀ w ∈ r.walls, ¬is_completely_illuminated w r.candle :=
sorry

end NUMINAMATH_CALUDE_candle_illumination_theorem_l89_8929


namespace NUMINAMATH_CALUDE_tank_b_height_is_six_l89_8977

/-- The height of a cylindrical tank B, given another tank A and their properties --/
def heightOfTankB (heightA circumferenceA circumferenceB : ℝ) : ℝ :=
  let radiusA := circumferenceA / (2 * Real.pi)
  let radiusB := circumferenceB / (2 * Real.pi)
  let capacityA := Real.pi * radiusA^2 * heightA
  6

/-- Theorem stating that the height of tank B is 6 meters under given conditions --/
theorem tank_b_height_is_six (heightA circumferenceA circumferenceB : ℝ)
  (h_heightA : heightA = 10)
  (h_circumferenceA : circumferenceA = 6)
  (h_circumferenceB : circumferenceB = 10)
  (h_capacity_ratio : Real.pi * (circumferenceA / (2 * Real.pi))^2 * heightA = 
                      0.6 * (Real.pi * (circumferenceB / (2 * Real.pi))^2 * heightOfTankB heightA circumferenceA circumferenceB)) :
  heightOfTankB heightA circumferenceA circumferenceB = 6 := by
  sorry

#check tank_b_height_is_six

end NUMINAMATH_CALUDE_tank_b_height_is_six_l89_8977


namespace NUMINAMATH_CALUDE_inverse_proportion_relation_l89_8941

/-- Given that the points (2, y₁) and (3, y₂) lie on the graph of the inverse proportion function y = 6/x,
    prove that y₁ > y₂. -/
theorem inverse_proportion_relation (y₁ y₂ : ℝ) :
  (2 : ℝ) * y₁ = 6 ∧ (3 : ℝ) * y₂ = 6 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relation_l89_8941


namespace NUMINAMATH_CALUDE_solve_equation_l89_8960

theorem solve_equation : ∃ m : ℤ, 2^4 - 3 = 3^3 + m ∧ m = -14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l89_8960


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l89_8926

/-- A figure on a grid --/
structure GridFigure where
  -- Add necessary properties of the figure
  area : ℕ

/-- A triangle on a grid --/
structure GridTriangle where
  -- Add necessary properties of a triangle
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut a figure into triangles --/
def cut_into_triangles (figure : GridFigure) : List GridTriangle :=
  sorry

/-- Function to check if triangles can form a square --/
def can_form_square (triangles : List GridTriangle) : Bool :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) :
  ∃ (triangles : List GridTriangle),
    (triangles.length = 5) ∧
    (cut_into_triangles figure = triangles) ∧
    (can_form_square triangles = true) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l89_8926


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l89_8974

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ
  apricot : ℕ
  mixed_fruit : ℕ

/-- Defines the relationships between jelly sales -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 4 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.apricot = s.grape / 2 ∧
  s.mixed_fruit = 3 * s.raspberry ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 8

theorem strawberry_jelly_sales (s : JellySales) (h : valid_jelly_sales s) :
  s.strawberry = 12 :=
sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l89_8974


namespace NUMINAMATH_CALUDE_business_value_l89_8982

theorem business_value (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  man_share = 2/3 →
  sold_fraction = 3/4 →
  sale_price = 6500 →
  (sale_price / sold_fraction / man_share : ℚ) = 39000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_l89_8982


namespace NUMINAMATH_CALUDE_equation_solution_l89_8925

theorem equation_solution (x y : ℝ) : 3 * x - 4 * y = 5 → x = (1/3) * (5 + 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l89_8925


namespace NUMINAMATH_CALUDE_underlined_numbers_are_correct_l89_8922

def sequence_term (n : ℕ) : ℕ := 3 * n - 2

def has_same_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 ∧ d₂ ∈ n.digits 10 → d₁ = d₂

def underlined_numbers : Set ℕ :=
  {n | ∃ k, sequence_term k = n ∧ 10 < n ∧ n < 100000 ∧ has_same_digits n}

theorem underlined_numbers_are_correct : underlined_numbers = 
  {22, 55, 88, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 
   11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999} := by
  sorry

end NUMINAMATH_CALUDE_underlined_numbers_are_correct_l89_8922


namespace NUMINAMATH_CALUDE_new_person_weight_l89_8978

theorem new_person_weight (W : ℝ) :
  let initial_avg := W / 20
  let intermediate_avg := (W - 95) / 19
  let final_avg := initial_avg + 4.2
  let new_person_weight := (final_avg * 20) - (W - 95)
  new_person_weight = 179 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l89_8978


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_is_six_thirteenths_l89_8963

/-- A regular nonagon is a 9-sided regular polygon -/
structure RegularNonagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def diagonal_intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem diagonal_intersection_probability_is_six_thirteenths (n : RegularNonagon) :
  diagonal_intersection_probability n = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_is_six_thirteenths_l89_8963


namespace NUMINAMATH_CALUDE_equation_equivalence_l89_8986

theorem equation_equivalence (x y : ℝ) :
  (x - 60) / 3 = (4 - 3 * x) / 6 + y ↔ x = (124 + 6 * y) / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l89_8986


namespace NUMINAMATH_CALUDE_expand_expression_l89_8995

theorem expand_expression (x y : ℝ) : 24 * (3 * x + 4 * y - 2) = 72 * x + 96 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l89_8995


namespace NUMINAMATH_CALUDE_max_value_theorem_l89_8905

-- Define the line l
def line_l (y : ℝ) : Prop := y = 8

-- Define the circle C
def circle_C (x y : ℝ) : Prop := ∃ φ, x = 2 * Real.cos φ ∧ y = 2 + 2 * Real.sin φ

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := ∃ α, θ = α ∧ 0 < α ∧ α < Real.pi / 2

-- Define the ray ON
def ray_ON (θ : ℝ) : Prop := ∃ α, θ = α + Real.pi / 2 ∧ 0 < α ∧ α < Real.pi / 2

-- Define the theorem
theorem max_value_theorem :
  ∃ (OP OM OQ ON : ℝ),
    (∀ y, line_l y → ∃ x, circle_C x y) →
    (∀ θ, ray_OM θ → ∃ x y, circle_C x y) →
    (∀ θ, ray_ON θ → ∃ x y, circle_C x y) →
    (∀ α, 0 < α → α < Real.pi / 2 →
      ∃ (OP OM OQ ON : ℝ),
        (OP / OM) * (OQ / ON) ≤ 1 / 16) ∧
    (∃ α, 0 < α ∧ α < Real.pi / 2 ∧
      (OP / OM) * (OQ / ON) = 1 / 16) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l89_8905


namespace NUMINAMATH_CALUDE_binomial_11_1_l89_8935

theorem binomial_11_1 : (11 : ℕ).choose 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_1_l89_8935


namespace NUMINAMATH_CALUDE_meeting_point_difference_l89_8916

/-- The distance between points R and S in miles -/
def total_distance : ℕ := 80

/-- The constant speed of the man starting from R in miles per hour -/
def speed_R : ℕ := 5

/-- The initial speed of the man starting from S in miles per hour -/
def initial_speed_S : ℕ := 4

/-- The hourly increase in speed for the man starting from S in miles per hour -/
def speed_increase_S : ℕ := 1

/-- The number of hours it takes for the men to meet -/
def meeting_time : ℕ := 8

/-- The distance traveled by the man starting from R -/
def distance_R : ℕ := speed_R * meeting_time

/-- The distance traveled by the man starting from S -/
def distance_S : ℕ := initial_speed_S * meeting_time + (meeting_time - 1) * meeting_time / 2

/-- The difference in distances traveled by the two men -/
def x : ℤ := distance_S - distance_R

theorem meeting_point_difference : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_difference_l89_8916


namespace NUMINAMATH_CALUDE_tan_half_angle_special_case_l89_8940

theorem tan_half_angle_special_case (α : Real) 
  (h1 : 5 * Real.sin (2 * α) = 6 * Real.cos α) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan (α / 2) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_angle_special_case_l89_8940


namespace NUMINAMATH_CALUDE_prob_two_even_balls_l89_8971

/-- The probability of drawing two even-numbered balls without replacement from 16 balls numbered 1 to 16 is 7/30. -/
theorem prob_two_even_balls (n : ℕ) (h : n = 16) :
  (Nat.card {i : Fin n | i.val % 2 = 0} : ℚ) / n *
  ((Nat.card {i : Fin n | i.val % 2 = 0} - 1) : ℚ) / (n - 1) = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_balls_l89_8971


namespace NUMINAMATH_CALUDE_expand_expression_l89_8917

theorem expand_expression (x y : ℝ) : 5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l89_8917


namespace NUMINAMATH_CALUDE_probability_non_littermates_correct_l89_8934

/-- Represents the number of dogs with a specific number of littermates -/
structure DogGroup where
  count : Nat
  littermates : Nat

/-- Represents the total number of dogs and their groupings by littermates -/
structure BreedingKennel where
  totalDogs : Nat
  groups : List DogGroup

/-- Calculates the probability of selecting two non-littermate dogs from a breeding kennel -/
def probabilityNonLittermates (kennel : BreedingKennel) : Rat :=
  sorry

theorem probability_non_littermates_correct (kennel : BreedingKennel) :
  kennel.totalDogs = 20 ∧
  kennel.groups = [
    ⟨8, 1⟩,
    ⟨6, 2⟩,
    ⟨4, 3⟩,
    ⟨2, 4⟩
  ] →
  probabilityNonLittermates kennel = 82 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_non_littermates_correct_l89_8934


namespace NUMINAMATH_CALUDE_marble_distribution_l89_8927

theorem marble_distribution (x : ℕ) 
  (liam : ℕ) (mia : ℕ) (noah : ℕ) (olivia : ℕ) : 
  liam = x ∧ 
  mia = 3 * x ∧ 
  noah = 12 * x ∧ 
  olivia = 24 * x ∧ 
  liam + mia + noah + olivia = 160 → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l89_8927


namespace NUMINAMATH_CALUDE_power_equation_solution_l89_8920

theorem power_equation_solution (x : ℝ) : (2^4 * 3^6 : ℝ) = 9 * 6^x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l89_8920


namespace NUMINAMATH_CALUDE_monday_profit_ratio_l89_8976

def total_profit : ℚ := 1200
def wednesday_profit : ℚ := 500
def tuesday_profit : ℚ := (1 / 4) * total_profit

def monday_profit : ℚ := total_profit - tuesday_profit - wednesday_profit

theorem monday_profit_ratio : 
  monday_profit / total_profit = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_monday_profit_ratio_l89_8976


namespace NUMINAMATH_CALUDE_smallest_cube_on_unit_cube_surface_l89_8911

-- Define a cube type
structure Cube where
  edgeLength : ℝ

-- Define the unit cube K1
def K1 : Cube := ⟨1⟩

-- Define the property that a cube's vertices lie on the surface of K1
def verticesOnSurfaceOfK1 (c : Cube) : Prop := sorry

-- Theorem statement
theorem smallest_cube_on_unit_cube_surface :
  ∃ (minCube : Cube), 
    verticesOnSurfaceOfK1 minCube ∧ 
    minCube.edgeLength = 1 / Real.sqrt 2 ∧
    ∀ (c : Cube), verticesOnSurfaceOfK1 c → c.edgeLength ≥ minCube.edgeLength :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_on_unit_cube_surface_l89_8911


namespace NUMINAMATH_CALUDE_area_KLMN_value_l89_8999

/-- Triangle ABC with points K, L, N, and M -/
structure TriangleABC where
  -- Define the sides of the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the positions of points K, L, and N
  AK : ℝ
  AL : ℝ
  BN : ℝ
  -- Ensure the triangle satisfies the given conditions
  h_AB : AB = 14
  h_BC : BC = 13
  h_AC : AC = 15
  h_AK : AK = 15/14
  h_AL : AL = 1
  h_BN : BN = 9

/-- The area of quadrilateral KLMN in the given triangle -/
def areaKLMN (t : TriangleABC) : ℝ := sorry

/-- Theorem stating that the area of KLMN is 36503/1183 -/
theorem area_KLMN_value (t : TriangleABC) : areaKLMN t = 36503/1183 := by sorry

end NUMINAMATH_CALUDE_area_KLMN_value_l89_8999


namespace NUMINAMATH_CALUDE_diamond_operation_result_l89_8969

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem diamond_operation_result :
  diamond (diamond Element.three Element.four) (diamond Element.two Element.one) = Element.two := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_result_l89_8969


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l89_8928

theorem rectangle_width_length_ratio :
  ∀ w : ℝ,
  w > 0 →
  2 * w + 2 * 10 = 30 →
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l89_8928


namespace NUMINAMATH_CALUDE_range_of_a_l89_8955

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  p a ∧ q a → a = 1 ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l89_8955


namespace NUMINAMATH_CALUDE_min_area_line_equation_l89_8932

/-- The equation of the line passing through (3, 1) that minimizes the area of the triangle formed by its x and y intercepts and the origin --/
theorem min_area_line_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), x / a + y / b = 1 → (3 / a + 1 / b = 1)) ∧
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∀ (x y : ℝ), x / a' + y / b' = 1 → (3 / a' + 1 / b' = 1)) →
    a * b ≤ a' * b') ∧
  a = 6 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_line_equation_l89_8932


namespace NUMINAMATH_CALUDE_smallest_relatively_prime_to_180_l89_8997

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_relatively_prime_to_180 :
  ∃ (x : ℕ), x > 1 ∧ is_relatively_prime x 180 ∧
  ∀ (y : ℕ), y > 1 ∧ y < x → ¬(is_relatively_prime y 180) :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_smallest_relatively_prime_to_180_l89_8997


namespace NUMINAMATH_CALUDE_spools_per_beret_l89_8943

theorem spools_per_beret (total_spools : ℕ) (num_berets : ℕ) 
  (h1 : total_spools = 33) 
  (h2 : num_berets = 11) 
  (h3 : num_berets > 0) : 
  total_spools / num_berets = 3 := by
  sorry

end NUMINAMATH_CALUDE_spools_per_beret_l89_8943


namespace NUMINAMATH_CALUDE_double_fibonacci_sum_convergence_l89_8937

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def G (n : ℕ) : ℕ := 2 * fibonacci n

theorem double_fibonacci_sum_convergence :
  (∑' n, (G n : ℝ) / 5^n) = 10/19 := by sorry

end NUMINAMATH_CALUDE_double_fibonacci_sum_convergence_l89_8937


namespace NUMINAMATH_CALUDE_regular_pay_is_2_40_l89_8979

/-- Calculates the regular pay per hour given the following conditions:
  - Regular week: 5 working days, 8 hours per day
  - Overtime pay: Rs. 3.20 per hour
  - Total earnings in 4 weeks: Rs. 432
  - Total hours worked in 4 weeks: 175 hours
-/
def regularPayPerHour (
  workingDaysPerWeek : ℕ)
  (workingHoursPerDay : ℕ)
  (overtimePay : ℚ)
  (totalEarnings : ℚ)
  (totalHoursWorked : ℕ) : ℚ :=
  let regularHoursPerWeek := workingDaysPerWeek * workingHoursPerDay
  let totalRegularHours := 4 * regularHoursPerWeek
  let overtimeHours := totalHoursWorked - totalRegularHours
  let overtimeEarnings := overtimeHours * overtimePay
  let regularEarnings := totalEarnings - overtimeEarnings
  regularEarnings / totalRegularHours

/-- Proves that the regular pay per hour is Rs. 2.40 given the specified conditions. -/
theorem regular_pay_is_2_40 :
  regularPayPerHour 5 8 (32/10) 432 175 = 24/10 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_is_2_40_l89_8979


namespace NUMINAMATH_CALUDE_cleaner_used_is_80_l89_8984

/-- Represents the flow rate of cleaner through a pipe at different time intervals -/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (after25min : ℝ)

/-- Calculates the total amount of cleaner used over a 30-minute period -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  flow.initial * 15 + flow.after15min * 10 + flow.after25min * 5

/-- The flow rates given in the problem -/
def problemFlow : FlowRate :=
  { initial := 2
  , after15min := 3
  , after25min := 4 }

/-- Theorem stating that the total cleaner used is 80 ounces -/
theorem cleaner_used_is_80 : totalCleanerUsed problemFlow = 80 := by
  sorry

end NUMINAMATH_CALUDE_cleaner_used_is_80_l89_8984


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l89_8996

theorem missing_fraction_sum (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 4 : ℚ) + (-9 / 20 : ℚ) + (-5 / 6 : ℚ) + x = 0.8333333333333334 
  → x = 0.8333333333333334 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l89_8996


namespace NUMINAMATH_CALUDE_fraction_of_72_l89_8990

theorem fraction_of_72 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_72_l89_8990


namespace NUMINAMATH_CALUDE_periodic_function_value_l89_8910

/-- Given a function f(x) = a*sin(π*x + θ) + b*cos(π*x + θ) + 3,
    where a, b, θ are non-zero real numbers, and f(2016) = -1,
    prove that f(2017) = 7. -/
theorem periodic_function_value (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hθ : θ ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + θ) + b * Real.cos (π * x + θ) + 3
  f 2016 = -1 → f 2017 = 7 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l89_8910


namespace NUMINAMATH_CALUDE_firetruck_reachable_area_l89_8947

/-- Represents the speed of the firetruck in different terrains --/
structure FiretruckSpeed where
  road : ℝ
  field : ℝ

/-- Calculates the area reachable by a firetruck given its speed and time --/
def reachable_area (speed : FiretruckSpeed) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area reachable by the firetruck in 15 minutes --/
theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 18
  let time := 15 / 60  -- 15 minutes in hours
  reachable_area speed time = 1194.75 := by
  sorry

end NUMINAMATH_CALUDE_firetruck_reachable_area_l89_8947


namespace NUMINAMATH_CALUDE_gcf_of_abc_l89_8919

def a : ℕ := 90
def b : ℕ := 126
def c : ℕ := 180

-- The condition that c is the product of a and b divided by some integer
axiom exists_divisor : ∃ k : ℕ, k ≠ 0 ∧ c * k = a * b

-- Define the greatest common factor function
def gcf (x y z : ℕ) : ℕ := Nat.gcd x (Nat.gcd y z)

theorem gcf_of_abc : gcf a b c = 18 := by sorry

end NUMINAMATH_CALUDE_gcf_of_abc_l89_8919
