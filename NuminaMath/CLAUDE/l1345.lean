import Mathlib

namespace NUMINAMATH_CALUDE_total_odd_initial_plums_l1345_134589

def is_odd_initial (name : String) : Bool :=
  let initial := name.front
  let position := initial.toUpper.toNat - 'A'.toNat + 1
  position % 2 ≠ 0

def plums_picked (name : String) (amount : ℕ) : ℕ :=
  if is_odd_initial name then amount else 0

theorem total_odd_initial_plums :
  let melanie_plums := plums_picked "Melanie" 4
  let dan_plums := plums_picked "Dan" 9
  let sally_plums := plums_picked "Sally" 3
  let ben_plums := plums_picked "Ben" (2 * (4 + 9))
  let peter_plums := plums_picked "Peter" (((3 * 3) / 4) - ((3 * 1) / 4))
  melanie_plums + dan_plums + sally_plums + ben_plums + peter_plums = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_odd_initial_plums_l1345_134589


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1345_134512

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 - 4 * x + 5 = (Real.sqrt 2 * x - Real.sqrt 2 + Complex.I * Real.sqrt 3) * 
                        (Real.sqrt 2 * x - Real.sqrt 2 - Complex.I * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1345_134512


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l1345_134505

/-- Given two drums X and Y with oil, prove that the ratio of Y's capacity to X's capacity is 2 -/
theorem drum_capacity_ratio (C_X C_Y : ℝ) : 
  C_X > 0 → C_Y > 0 → 
  (1/2 : ℝ) * C_X + (1/5 : ℝ) * C_Y = (0.45 : ℝ) * C_Y → 
  C_Y / C_X = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l1345_134505


namespace NUMINAMATH_CALUDE_lava_lamp_probability_is_one_seventh_l1345_134595

/-- The probability of a specific arrangement of lava lamps -/
def lava_lamp_probability : ℚ :=
  let total_lamps : ℕ := 4 + 4  -- 4 red + 4 blue lamps
  let lamps_on : ℕ := 4  -- 4 lamps are turned on
  let remaining_lamps : ℕ := total_lamps - 2  -- excluding leftmost and rightmost
  let remaining_on : ℕ := lamps_on - 1  -- excluding the rightmost lamp which is on
  let favorable_arrangements : ℕ := Nat.choose remaining_lamps (total_lamps / 2 - 1)  -- arranging remaining red lamps
  let favorable_on_choices : ℕ := Nat.choose (total_lamps - 1) (lamps_on - 1)  -- choosing remaining on lamps
  let total_arrangements : ℕ := Nat.choose total_lamps (total_lamps / 2)  -- total ways to arrange red and blue lamps
  let total_on_choices : ℕ := Nat.choose total_lamps lamps_on  -- total ways to choose on lamps
  (favorable_arrangements * favorable_on_choices : ℚ) / (total_arrangements * total_on_choices)

/-- The probability of the specific lava lamp arrangement is 1/7 -/
theorem lava_lamp_probability_is_one_seventh : lava_lamp_probability = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_is_one_seventh_l1345_134595


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_linear_l1345_134549

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    for all rational numbers x < y < z < t in arithmetic progression,
    f(y) + f(z) = f(x) + f(t) -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f y + f z = f x + f t

/-- The main theorem stating that any function satisfying the arithmetic progression property
    is a linear function -/
theorem arithmetic_progression_implies_linear
  (f : ℚ → ℚ) (h : ArithmeticProgressionProperty f) :
  ∃ (C : ℚ), ∀ (x : ℚ), f x = C * x := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_linear_l1345_134549


namespace NUMINAMATH_CALUDE_sin_negative_600_degrees_l1345_134544

theorem sin_negative_600_degrees : Real.sin (- 600 * π / 180) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_600_degrees_l1345_134544


namespace NUMINAMATH_CALUDE_dihedral_angle_range_in_regular_prism_l1345_134521

theorem dihedral_angle_range_in_regular_prism (n : ℕ) (h : n > 2) :
  ∃ θ : ℝ, ((n - 2 : ℝ) / n) * π < θ ∧ θ < π :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_in_regular_prism_l1345_134521


namespace NUMINAMATH_CALUDE_function_value_at_a_plus_one_l1345_134506

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_value_at_a_plus_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_a_plus_one_l1345_134506


namespace NUMINAMATH_CALUDE_boat_distance_proof_l1345_134532

/-- The distance covered by a boat traveling downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 8) :
  distance_downstream boat_speed stream_speed time = 168 := by
  sorry

#eval distance_downstream 16 5 8

end NUMINAMATH_CALUDE_boat_distance_proof_l1345_134532


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l1345_134514

/-- The hyperbola equation x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- The line equation y = k(x-1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The number of intersection points between the line and the hyperbola -/
def intersection_count (k : ℝ) : ℕ := sorry

theorem hyperbola_line_intersection (k : ℝ) :
  (intersection_count k = 2 ↔ k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∪ 
                            Set.Ioo (-1) 1 ∪ 
                            Set.Ioo 1 (2 * Real.sqrt 3 / 3)) ∧
  (intersection_count k = 1 ↔ k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -2 * Real.sqrt 3 / 3) ∧
  (intersection_count k = 0 ↔ k ∈ Set.Iic (-2 * Real.sqrt 3 / 3) ∪ 
                            Set.Ici (2 * Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l1345_134514


namespace NUMINAMATH_CALUDE_push_up_sets_l1345_134590

/-- Represents the number of push-ups done by each person -/
structure PushUps where
  zachary : ℕ
  david : ℕ
  emily : ℕ

/-- Calculates the number of complete sets of push-ups done together -/
def completeSets (p : PushUps) : ℕ :=
  1

theorem push_up_sets (p : PushUps) 
  (h1 : p.zachary = 47)
  (h2 : p.david = p.zachary + 15)
  (h3 : p.emily = 2 * p.david) :
  completeSets p = 1 := by
  sorry

#check push_up_sets

end NUMINAMATH_CALUDE_push_up_sets_l1345_134590


namespace NUMINAMATH_CALUDE_valid_numbers_count_and_max_l1345_134526

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ),
    is_prime x ∧ is_prime y ∧
    n = 4000 + 100 * a + 10 * b + 5 ∧
    n = 5 * x * 11 * y

theorem valid_numbers_count_and_max :
  (∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    (∀ n, is_valid_number n → n ∈ s) ∧
    s.card = 3) ∧
  (∃ m : ℕ, is_valid_number m ∧ ∀ n, is_valid_number n → n ≤ m) ∧
  (∃ m : ℕ, is_valid_number m ∧ m = 4785) :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_and_max_l1345_134526


namespace NUMINAMATH_CALUDE_no_three_digit_integers_with_five_units_divisible_by_ten_l1345_134567

theorem no_three_digit_integers_with_five_units_divisible_by_ten :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit positive integer
    n % 10 = 5 ∧          -- 5 in the units place
    n % 10 = 0            -- divisible by 10
  := by sorry

end NUMINAMATH_CALUDE_no_three_digit_integers_with_five_units_divisible_by_ten_l1345_134567


namespace NUMINAMATH_CALUDE_tan_theta_value_l1345_134553

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.cos (θ - π/3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1345_134553


namespace NUMINAMATH_CALUDE_digital_earth_capabilities_l1345_134546

structure DigitalEarth where
  simulate_environment : Bool
  monitor_crops : Bool
  predict_submersion : Bool
  simulate_past : Bool
  predict_future : Bool

theorem digital_earth_capabilities (de : DigitalEarth) :
  de.simulate_environment ∧
  de.monitor_crops ∧
  de.predict_submersion ∧
  de.simulate_past →
  ¬ de.predict_future :=
by sorry

end NUMINAMATH_CALUDE_digital_earth_capabilities_l1345_134546


namespace NUMINAMATH_CALUDE_students_liking_computing_l1345_134573

theorem students_liking_computing (total : ℕ) (both : ℕ) 
  (h1 : total = 33)
  (h2 : both = 3)
  (h3 : ∀ (pe_only computing_only : ℕ), 
    pe_only + computing_only + both = total → 
    pe_only = computing_only / 2) :
  ∃ (pe_only computing_only : ℕ),
    pe_only + computing_only + both = total ∧
    pe_only = computing_only / 2 ∧
    computing_only + both = 23 := by
sorry

end NUMINAMATH_CALUDE_students_liking_computing_l1345_134573


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l1345_134519

theorem consecutive_numbers_product (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 48) : 
  n * (n + 2) = 255 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l1345_134519


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1345_134564

/-- The constant term in the expansion of (3+x)(x+1/x)^6 -/
def constant_term : ℕ := 60

/-- Theorem: The constant term in the expansion of (3+x)(x+1/x)^6 is 60 -/
theorem constant_term_expansion :
  constant_term = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1345_134564


namespace NUMINAMATH_CALUDE_triangle_angle_c_ninety_degrees_l1345_134559

/-- Given a triangle ABC with sides a, b, c and angles A, B, C respectively,
    if b^2 + c^2 - bc = a^2 and a/b = √3, then angle C = 90°. -/
theorem triangle_angle_c_ninety_degrees 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + c^2 - b*c = a^2)
  (h_ratio : a/b = Real.sqrt 3) : 
  C = Real.pi/2 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_c_ninety_degrees_l1345_134559


namespace NUMINAMATH_CALUDE_bakery_batches_l1345_134599

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after each batch -/
def baguettes_sold : List ℕ := [37, 52, 49]

/-- The number of baguettes left unsold -/
def baguettes_left : ℕ := 6

/-- The number of batches of baguettes the bakery makes a day -/
def num_batches : ℕ := 3

theorem bakery_batches :
  (baguettes_per_batch * num_batches) = (baguettes_sold.sum + baguettes_left) :=
by sorry

end NUMINAMATH_CALUDE_bakery_batches_l1345_134599


namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l1345_134523

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a n ≠ -1) ∧
  (∀ n, a (n + 2) = (a n + 2006) / (a (n + 1) + 1))

/-- The theorem stating the properties of the recurrence sequence -/
theorem recurrence_sequence_property (a : ℕ → ℤ) (h : RecurrenceSequence a) :
  ∃ x y : ℤ, x * y = 2006 ∧ (∀ n, a n = x ∨ a n = y) ∧ (∀ n, a n = a (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l1345_134523


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1345_134536

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_strictly_increasing :
  (∀ x > 0, x^2 * (deriv f x) + 2*x * f x = exp x / x) →
  f 2 = exp 2 / 8 →
  StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1345_134536


namespace NUMINAMATH_CALUDE_function_derivative_at_zero_l1345_134513

theorem function_derivative_at_zero 
  (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_at_zero_l1345_134513


namespace NUMINAMATH_CALUDE_variance_scaled_and_shifted_l1345_134548

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_scaled_and_shifted
  (h : variance x = 1) :
  variance (fun i => 2 * x i + 1) = 4 := by sorry

end NUMINAMATH_CALUDE_variance_scaled_and_shifted_l1345_134548


namespace NUMINAMATH_CALUDE_james_car_transaction_l1345_134594

/-- The amount James is out of pocket after selling his old car and buying a new one -/
def out_of_pocket (old_car_value : ℝ) (old_car_sale_percentage : ℝ) 
                  (new_car_sticker : ℝ) (new_car_buy_percentage : ℝ) : ℝ :=
  new_car_sticker * new_car_buy_percentage - old_car_value * old_car_sale_percentage

/-- Theorem stating that James is out of pocket by $11,000 -/
theorem james_car_transaction : 
  out_of_pocket 20000 0.8 30000 0.9 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_james_car_transaction_l1345_134594


namespace NUMINAMATH_CALUDE_f_pi_sixth_l1345_134554

/-- The function f(x) = sin x + a cos x, where a < 0 and max f(x) = 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

/-- Theorem stating that f(π/6) = -1 under given conditions -/
theorem f_pi_sixth (a : ℝ) (h1 : a < 0) (h2 : ∀ x, f a x ≤ 2) (h3 : ∃ x, f a x = 2) :
  f a (Real.pi / 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_sixth_l1345_134554


namespace NUMINAMATH_CALUDE_part_I_part_II_l1345_134502

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Define set N with parameter a
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part I
theorem part_I : 
  M ∩ (U \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1345_134502


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1345_134527

/-- Calculates the man's speed against the current with wind resistance -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (wind_resistance_factor : ℝ) (current_increase_factor : ℝ) : ℝ :=
  let speed_still_water := speed_with_current - current_speed
  let effective_speed_still_water := speed_still_water * (1 - wind_resistance_factor)
  let new_current_speed := current_speed * (1 + current_increase_factor)
  effective_speed_still_water - new_current_speed

/-- Theorem stating the man's speed against the current -/
theorem mans_speed_against_current :
  speed_against_current 22 5 0.15 0.1 = 8.95 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1345_134527


namespace NUMINAMATH_CALUDE_ab_value_l1345_134578

/-- Represents the alphabet with corresponding number values. -/
def alphabet : List (Char × Nat) :=
  ('a', 1) :: ('b', 2) :: ('c', 3) :: ('d', 4) :: ('e', 5) :: ('f', 6) :: ('g', 7) ::
  ('h', 8) :: ('i', 9) :: ('j', 10) :: ('k', 11) :: ('l', 12) :: ('m', 13) ::
  ('n', 14) :: ('o', 15) :: ('p', 16) :: ('q', 17) :: ('r', 18) :: ('s', 19) ::
  ('t', 20) :: ('u', 21) :: ('v', 22) :: ('w', 23) :: ('x', 24) :: ('y', 25) ::
  ('z', 26) :: []

/-- Get the number value of a character based on its position in the alphabet. -/
def letterValue (c : Char) : Nat :=
  match alphabet.find? (fun p => p.1 == c) with
  | some (_, v) => v
  | none => 0

/-- Calculate the number value of a word. -/
def wordValue (word : String) : Nat :=
  let letterSum := word.toList.map letterValue |>.sum
  letterSum * word.length

/-- Theorem: The number value of the word "ab" is 6. -/
theorem ab_value : wordValue "ab" = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1345_134578


namespace NUMINAMATH_CALUDE_lcm_of_36_and_154_l1345_134511

theorem lcm_of_36_and_154 :
  let a := 36
  let b := 154
  let hcf := 14
  hcf = Nat.gcd a b →
  Nat.lcm a b = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_154_l1345_134511


namespace NUMINAMATH_CALUDE_negation_equivalence_l1345_134522

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1345_134522


namespace NUMINAMATH_CALUDE_inequality_not_always_satisfied_l1345_134503

theorem inequality_not_always_satisfied :
  ∃ (p q r : ℝ), p < 1 ∧ q < 2 ∧ r < 3 ∧ p^2 + 2*q*r ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_satisfied_l1345_134503


namespace NUMINAMATH_CALUDE_customers_who_tipped_l1345_134579

theorem customers_who_tipped (initial_customers additional_customers no_tip_customers : ℕ) :
  initial_customers + additional_customers - no_tip_customers =
  (initial_customers + additional_customers) - no_tip_customers :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l1345_134579


namespace NUMINAMATH_CALUDE_square_area_subtraction_l1345_134531

theorem square_area_subtraction (s : ℝ) (x : ℝ) : 
  s = 4 → s^2 + s - x = 4 → x = 16 := by sorry

end NUMINAMATH_CALUDE_square_area_subtraction_l1345_134531


namespace NUMINAMATH_CALUDE_total_layers_is_112_l1345_134501

/-- Represents an artist working on a multi-layered painting project -/
structure Artist where
  hours_per_week : ℕ
  hours_per_layer : ℕ

/-- Calculates the number of layers an artist can complete in a given number of weeks -/
def layers_completed (artist : Artist) (weeks : ℕ) : ℕ :=
  (artist.hours_per_week * weeks) / artist.hours_per_layer

/-- The duration of the project in weeks -/
def project_duration : ℕ := 4

/-- The team of artists working on the project -/
def artist_team : List Artist := [
  { hours_per_week := 30, hours_per_layer := 3 },
  { hours_per_week := 40, hours_per_layer := 5 },
  { hours_per_week := 20, hours_per_layer := 2 }
]

/-- Theorem: The total number of layers completed by all artists in the project is 112 -/
theorem total_layers_is_112 : 
  (artist_team.map (λ a => layers_completed a project_duration)).sum = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_layers_is_112_l1345_134501


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1345_134598

def f (x : ℝ) : ℝ := x^2 + 3*x + 2

theorem max_value_of_f_in_interval :
  ∃ (M : ℝ), M = 42 ∧ 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧
  (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l1345_134598


namespace NUMINAMATH_CALUDE_largest_number_l1345_134593

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 997/1000) 
  (hb : b = 9799/10000) 
  (hc : c = 999/1000) 
  (hd : d = 9979/10000) 
  (he : e = 979/1000) : 
  c = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1345_134593


namespace NUMINAMATH_CALUDE_trouser_original_price_l1345_134541

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 50 ∧ discount_percentage = 0.5 ∧ sale_price = (1 - discount_percentage) * original_price →
  original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l1345_134541


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1345_134557

/-- For a rhombus with area K and one diagonal three times the length of the other,
    the side length s is equal to √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  let s := Real.sqrt ((5 * K) / 3)
  let area := (1/2) * d * (3*d)
  area = K ∧ 
  s^2 = (d/2)^2 + (3*d/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1345_134557


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1345_134518

/-- Given a point A with coordinates (-2, 3), this theorem proves that its symmetric point B
    with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetric_point_wrt_x_axis :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-2, -3)
  (A.1 = B.1) ∧ (A.2 = -B.2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1345_134518


namespace NUMINAMATH_CALUDE_deck_size_proof_l1345_134520

theorem deck_size_proof (spades : ℕ) (prob_not_spade : ℚ) (total : ℕ) : 
  spades = 13 → 
  prob_not_spade = 3/4 → 
  (total - spades : ℚ) / total = prob_not_spade → 
  total = 52 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l1345_134520


namespace NUMINAMATH_CALUDE_complex_power_sum_l1345_134547

theorem complex_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 5)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 10) :
  α₁^6 + α₂^6 + α₃^6 = 44 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1345_134547


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l1345_134596

theorem greatest_integer_with_gcf_nine : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 45 = 9 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 45 = 9 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l1345_134596


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1345_134586

theorem necessary_not_sufficient_condition (a b c : ℝ) (h : b ≠ 0) (h' : c ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 = y ∧ (a / x = x / c) ∧ (a / y ≠ y / c)) ∧
  ((a / b = b / c) → b^2 = a * c) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1345_134586


namespace NUMINAMATH_CALUDE_root_value_theorem_l1345_134504

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 3 = 0 → 4 * m^2 - 6 * m + 2017 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1345_134504


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1345_134571

theorem complex_magnitude_equality : ∃ t : ℝ, t > 0 ∧ Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 :=
by
  use 2 * Real.sqrt 29
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1345_134571


namespace NUMINAMATH_CALUDE_grid_routes_3x2_l1345_134597

theorem grid_routes_3x2 :
  let total_moves : ℕ := 3 + 2
  let right_moves : ℕ := 3
  let down_moves : ℕ := 2
  let num_routes : ℕ := Nat.choose total_moves down_moves
  num_routes = 10 := by sorry

end NUMINAMATH_CALUDE_grid_routes_3x2_l1345_134597


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1345_134570

theorem trig_expression_equality : 
  (Real.sin (π/4) + Real.cos (π/6)) / (3 - 2 * Real.cos (π/3)) - 
  Real.sin (π/3) * (1 - Real.sin (π/6)) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1345_134570


namespace NUMINAMATH_CALUDE_plane_to_center_distance_l1345_134584

/-- Represents a point on the surface of a sphere -/
structure SpherePoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points on the sphere -/
def distance (p q : SpherePoint) : ℝ := sorry

/-- The radius of the sphere -/
def sphereRadius : ℝ := 13

/-- Theorem: The distance from the plane passing through A, B, C to the sphere center -/
theorem plane_to_center_distance 
  (A B C : SpherePoint) 
  (h1 : distance A B = 6)
  (h2 : distance B C = 8)
  (h3 : distance C A = 10) :
  ∃ (d : ℝ), d = 12 ∧ d^2 + sphereRadius^2 = (distance A B)^2 + (distance B C)^2 + (distance C A)^2 := by
  sorry

end NUMINAMATH_CALUDE_plane_to_center_distance_l1345_134584


namespace NUMINAMATH_CALUDE_same_last_four_digits_l1345_134535

theorem same_last_four_digits (N : ℕ) : 
  N > 0 ∧ 
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
  (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d →
  N / 1000 = 937 :=
by sorry

end NUMINAMATH_CALUDE_same_last_four_digits_l1345_134535


namespace NUMINAMATH_CALUDE_unique_arith_seq_pair_a_eq_one_third_l1345_134539

/-- Two arithmetic sequences satisfying given conditions -/
structure ArithSeqPair where
  a : ℝ
  q : ℝ
  h_a_pos : a > 0
  h_b1_a1 : (a + 1) - a = 1
  h_b2_a2 : (a + q + 2) - (a + q) = 2
  h_b3_a3 : (a + 2*q + 3) - (a + 2*q) = 3
  h_unique : ∃! q, (a * q^2 - 4 * a * q + 3 * a - 1 = 0) ∧ q ≠ 0

/-- If two arithmetic sequences satisfy the given conditions and one is unique, then a = 1/3 -/
theorem unique_arith_seq_pair_a_eq_one_third (p : ArithSeqPair) : p.a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_arith_seq_pair_a_eq_one_third_l1345_134539


namespace NUMINAMATH_CALUDE_keith_card_spend_l1345_134562

/-- The amount Keith spent on cards -/
def total_spent (digimon_packs : ℕ) (digimon_price : ℚ) (baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + baseball_price

/-- Proof that Keith spent $23.86 on cards -/
theorem keith_card_spend :
  total_spent 4 (4.45 : ℚ) (6.06 : ℚ) = (23.86 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spend_l1345_134562


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_inequality_l1345_134533

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_inequality_l1345_134533


namespace NUMINAMATH_CALUDE_complement_intersection_subset_range_l1345_134583

-- Define the sets A and B
def A : Set ℝ := {x | 2 < x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Theorem 1: Prove the complement intersection when a = 2
theorem complement_intersection :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem 2: Prove the range of a for which B is a subset of A
theorem subset_range :
  {a : ℝ | B a ⊆ A} = {a | 3 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_subset_range_l1345_134583


namespace NUMINAMATH_CALUDE_tank_capacity_l1345_134530

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_water : ℚ) :
  initial_fraction = 1/4 →
  final_fraction = 3/4 →
  added_water = 200 →
  (final_fraction - initial_fraction) * (added_water / (final_fraction - initial_fraction)) = 400 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1345_134530


namespace NUMINAMATH_CALUDE_sum_inequality_l1345_134585

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  (1/a^2 + 1/b^2 + 1/c^2) ≥ ((4*b*c/(a^2 + 1) + 4*a*c/(b^2 + 1) + 4*a*b/(c^2 + 1)))^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1345_134585


namespace NUMINAMATH_CALUDE_prob_heads_tails_heads_l1345_134558

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The probability of getting heads, then tails, then heads when flipping a fair coin three times -/
theorem prob_heads_tails_heads (p : ℝ) (h_fair : fair_coin p) : 
  prob_sequence p 3 = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_heads_tails_heads_l1345_134558


namespace NUMINAMATH_CALUDE_test_question_count_l1345_134524

/-- Given a test with two-point and four-point questions, prove the total number of questions. -/
theorem test_question_count (two_point_count four_point_count : ℕ) 
  (h1 : two_point_count = 30)
  (h2 : four_point_count = 10) :
  two_point_count + four_point_count = 40 := by
  sorry

#check test_question_count

end NUMINAMATH_CALUDE_test_question_count_l1345_134524


namespace NUMINAMATH_CALUDE_remaining_money_after_tickets_l1345_134592

/-- Calculates the remaining money after buying tickets -/
def remaining_money (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  olivia_money + nigel_money - num_tickets * ticket_price

/-- Proves that Olivia and Nigel have $83 left after buying tickets -/
theorem remaining_money_after_tickets : remaining_money 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_tickets_l1345_134592


namespace NUMINAMATH_CALUDE_broker_commission_rate_change_l1345_134556

/-- Proves that the new commission rate is 5% given the conditions of the problem -/
theorem broker_commission_rate_change
  (original_rate : ℝ)
  (business_slump : ℝ)
  (new_rate : ℝ)
  (h1 : original_rate = 0.04)
  (h2 : business_slump = 0.20000000000000007)
  (h3 : original_rate * (1 - business_slump) = new_rate) :
  new_rate = 0.05 := by
  sorry

#eval (0.04 / 0.7999999999999999 : Float)

end NUMINAMATH_CALUDE_broker_commission_rate_change_l1345_134556


namespace NUMINAMATH_CALUDE_cube_circumscribed_sphere_radius_l1345_134576

/-- The radius of the circumscribed sphere of a cube with edge length 1 is √3/2 -/
theorem cube_circumscribed_sphere_radius :
  let cube_edge_length : ℝ := 1
  let circumscribed_sphere_radius : ℝ := (Real.sqrt 3) / 2
  cube_edge_length = 1 →
  circumscribed_sphere_radius = (Real.sqrt 3) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_cube_circumscribed_sphere_radius_l1345_134576


namespace NUMINAMATH_CALUDE_radiator_fluid_calculation_l1345_134515

theorem radiator_fluid_calculation (initial_antifreeze_percentage : Real)
                                   (drain_amount : Real)
                                   (replacement_antifreeze_percentage : Real)
                                   (final_antifreeze_percentage : Real) :
  initial_antifreeze_percentage = 0.10 →
  drain_amount = 2.2857 →
  replacement_antifreeze_percentage = 0.80 →
  final_antifreeze_percentage = 0.50 →
  ∃ x : Real, x = 4 ∧
    initial_antifreeze_percentage * x - 
    initial_antifreeze_percentage * drain_amount + 
    replacement_antifreeze_percentage * drain_amount = 
    final_antifreeze_percentage * x :=
by
  sorry

end NUMINAMATH_CALUDE_radiator_fluid_calculation_l1345_134515


namespace NUMINAMATH_CALUDE_pirate_overtakes_at_six_hours_l1345_134507

/-- Represents the chase scenario between a pirate ship and a merchant vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_pirate_speed : ℝ
  initial_merchant_speed : ℝ
  speed_change_time : ℝ
  final_pirate_speed : ℝ
  final_merchant_speed : ℝ

/-- Calculates the time when the pirate ship overtakes the merchant vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that the pirate ship overtakes the merchant vessel after 6 hours -/
theorem pirate_overtakes_at_six_hours (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 15)
  (h2 : scenario.initial_pirate_speed = 14)
  (h3 : scenario.initial_merchant_speed = 10)
  (h4 : scenario.speed_change_time = 3)
  (h5 : scenario.final_pirate_speed = 12)
  (h6 : scenario.final_merchant_speed = 11) :
  overtake_time scenario = 6 :=
  sorry

end NUMINAMATH_CALUDE_pirate_overtakes_at_six_hours_l1345_134507


namespace NUMINAMATH_CALUDE_triangle_polynomial_equivalence_l1345_134580

/-- The triangle polynomial function -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- Triangle sides property -/
def is_triangle (x y z : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating the equivalence between the positivity of f and the triangle sides property -/
theorem triangle_polynomial_equivalence (x y z : ℝ) :
  is_triangle (|x|) (|y|) (|z|) ↔ 0 < f x y z :=
sorry

end NUMINAMATH_CALUDE_triangle_polynomial_equivalence_l1345_134580


namespace NUMINAMATH_CALUDE_no_prime_for_expression_l1345_134555

theorem no_prime_for_expression (m n : ℕ) : 
  ¬(Nat.Prime (n^2 + 2018*m*n + 2019*m + n - 2019*m^2)) := by
sorry

end NUMINAMATH_CALUDE_no_prime_for_expression_l1345_134555


namespace NUMINAMATH_CALUDE_product_comparison_l1345_134568

theorem product_comparison (a : Fin 10 → ℝ) 
  (h_pos : ∀ i, a i > 0) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ l ≠ m → 
      a i * a j * a k > a l * a m)) ∨
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ l m n o, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ 
      n ≠ i ∧ n ≠ j ∧ n ≠ k ∧ o ≠ i ∧ o ≠ j ∧ o ≠ k ∧ 
      l ≠ m ∧ l ≠ n ∧ l ≠ o ∧ m ≠ n ∧ m ≠ o ∧ n ≠ o → 
      a i * a j * a k > a l * a m * a n * a o)) := by
sorry

end NUMINAMATH_CALUDE_product_comparison_l1345_134568


namespace NUMINAMATH_CALUDE_product_mod_25_l1345_134534

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (123 * 156 * 198) % 25 = m ∧ m = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l1345_134534


namespace NUMINAMATH_CALUDE_inradius_eq_centroid_height_l1345_134563

/-- A non-equilateral triangle with sides a, b, and c, where a + b = 2c -/
structure NonEquilateralTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_equilateral : a ≠ b ∨ b ≠ c ∨ a ≠ c
  side_relation : a + b = 2 * c

/-- The inradius of a triangle -/
def inradius (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- The vertical distance from the base c to the centroid -/
def centroid_height (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating that the inradius is equal to the vertical distance from the base to the centroid -/
theorem inradius_eq_centroid_height (t : NonEquilateralTriangle) :
  inradius t = centroid_height t :=
sorry

end NUMINAMATH_CALUDE_inradius_eq_centroid_height_l1345_134563


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1345_134577

def complex_i : ℂ := Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : (1 + complex_i) * z = 1 - 2 * complex_i^3) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1345_134577


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1345_134525

theorem geometric_series_sum : 
  let a : ℚ := 1/3
  let r : ℚ := -1/3
  let n : ℕ := 5
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 61/243 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1345_134525


namespace NUMINAMATH_CALUDE_cos_135_degrees_l1345_134509

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l1345_134509


namespace NUMINAMATH_CALUDE_evaluate_expressions_l1345_134550

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def ln (x : ℝ) := Real.log x

-- Define the main theorem
theorem evaluate_expressions :
  (2 * Real.sqrt 3 * (12 ^ (1/6)) * (3 ^ (3/2)) = 6) ∧
  ((1/2) * lg 25 + lg 2 + ln (Real.sqrt (Real.exp 1)) - 
   (Real.log 27 / Real.log 2) * (Real.log 2 / Real.log 3) - 
   7 ^ (Real.log 3 / Real.log 7) = -9/2) := by
sorry

end NUMINAMATH_CALUDE_evaluate_expressions_l1345_134550


namespace NUMINAMATH_CALUDE_eight_equidistant_points_l1345_134537

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- A point in 3D space -/
structure Point3D where
  -- We don't need to define the specifics of a point for this problem

/-- Distance between a point and a plane -/
def distance (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry -- Actual implementation not needed for this statement

/-- The set of points at a given distance from a plane -/
def pointsAtDistance (plane : Plane3D) (d : ℝ) : Set Point3D :=
  {p : Point3D | distance p plane = d}

/-- The theorem stating that there are exactly 8 points at given distances from three planes -/
theorem eight_equidistant_points (plane1 plane2 plane3 : Plane3D) (m n p : ℝ) :
  ∃! (points : Finset Point3D),
    points.card = 8 ∧
    ∀ point ∈ points,
      distance point plane1 = m ∧
      distance point plane2 = n ∧
      distance point plane3 = p :=
  sorry


end NUMINAMATH_CALUDE_eight_equidistant_points_l1345_134537


namespace NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1345_134552

/-- The distance from the school to the Martyrs' Cemetery in kilometers -/
def distance : ℝ := 216

/-- The scheduled time for the journey in minutes -/
def scheduledTime : ℝ := 180

/-- The time saved in minutes when increasing speed by one-fifth after 1 hour -/
def timeSaved1 : ℝ := 20

/-- The time saved in minutes when increasing speed by one-third after 72km -/
def timeSaved2 : ℝ := 30

/-- The distance traveled at original speed before increasing by one-third -/
def initialDistance : ℝ := 72

theorem martyrs_cemetery_distance :
  (distance = 216) ∧
  (scheduledTime * (1 - 5/6) = timeSaved1) ∧
  (scheduledTime * (1 - 3/4) > timeSaved2) ∧
  (initialDistance / (1 - 2/3) = distance) :=
sorry

end NUMINAMATH_CALUDE_martyrs_cemetery_distance_l1345_134552


namespace NUMINAMATH_CALUDE_bface_hex_to_decimal_l1345_134572

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, should not be reached for valid hex digits

/-- Converts a hexadecimal string to its decimal value -/
def hexadecimal_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

theorem bface_hex_to_decimal :
  hexadecimal_to_decimal "BFACE" = 785102 := by
  sorry

end NUMINAMATH_CALUDE_bface_hex_to_decimal_l1345_134572


namespace NUMINAMATH_CALUDE_log_2_base_10_l1345_134591

theorem log_2_base_10 (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) (h3 : 2^9 = 512) (h4 : 2^12 = 4096) :
  Real.log 2 / Real.log 10 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_l1345_134591


namespace NUMINAMATH_CALUDE_inequality_proof_l1345_134508

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2)
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1345_134508


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l1345_134540

/-- The number of members in the Book club -/
def total_members : ℕ := 24

/-- The number of boys in the Book club -/
def num_boys : ℕ := 12

/-- The number of girls in the Book club -/
def num_girls : ℕ := 12

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least one boy and one girl -/
def probability_mixed_committee : ℚ := 171 / 177

theorem mixed_committee_probability :
  (1 : ℚ) - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / 
  (Nat.choose total_members committee_size : ℚ) = probability_mixed_committee := by
  sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l1345_134540


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1345_134575

/-- Three lines intersect at the same point if and only if m = -9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x + 2*y + 5 = 0) ↔ m = -9 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1345_134575


namespace NUMINAMATH_CALUDE_remainder_sum_l1345_134566

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1345_134566


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1345_134581

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - 2 * Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1345_134581


namespace NUMINAMATH_CALUDE_rangers_apprentice_reading_l1345_134569

theorem rangers_apprentice_reading (total_books : Nat) (pages_per_book : Nat) 
  (books_read_first_month : Nat) (pages_left_to_finish : Nat) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  pages_left_to_finish = 1000 →
  (((total_books - books_read_first_month) * pages_per_book - pages_left_to_finish) / pages_per_book) / 
  (total_books - books_read_first_month) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rangers_apprentice_reading_l1345_134569


namespace NUMINAMATH_CALUDE_prob_four_successes_in_five_trials_l1345_134543

/-- The probability of exactly 4 successes in 5 independent Bernoulli trials with p = 1/3 -/
theorem prob_four_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/3
  let k : ℕ := 4
  Nat.choose n k * p^k * (1-p)^(n-k) = 10/243 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_successes_in_five_trials_l1345_134543


namespace NUMINAMATH_CALUDE_fabric_equation_correct_l1345_134529

/-- Represents the fabric purchase scenario --/
structure FabricPurchase where
  total_meters : ℝ
  total_cost : ℝ
  blue_cost_per_meter : ℝ
  black_cost_per_meter : ℝ

/-- The equation correctly represents the fabric purchase scenario --/
theorem fabric_equation_correct (fp : FabricPurchase)
  (h1 : fp.total_meters = 138)
  (h2 : fp.total_cost = 540)
  (h3 : fp.blue_cost_per_meter = 3)
  (h4 : fp.black_cost_per_meter = 5) :
  ∃ x : ℝ, fp.blue_cost_per_meter * x + fp.black_cost_per_meter * (fp.total_meters - x) = fp.total_cost :=
by sorry

end NUMINAMATH_CALUDE_fabric_equation_correct_l1345_134529


namespace NUMINAMATH_CALUDE_power_product_squared_l1345_134538

theorem power_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l1345_134538


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l1345_134516

-- Define a function f: ℝ → ℝ with an inverse
def f : ℝ → ℝ := sorry

-- Assume f is bijective (has an inverse)
axiom f_bijective : Function.Bijective f

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Condition: f(x) + f(-x) = 4 for all x
axiom f_condition (x : ℝ) : f x + f (-x) = 4

-- Theorem to prove
theorem inverse_sum_theorem (x : ℝ) : f_inv (x - 3) + f_inv (7 - x) = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l1345_134516


namespace NUMINAMATH_CALUDE_evaluate_expression_l1345_134587

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) :
  2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1345_134587


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1345_134561

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 887 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1345_134561


namespace NUMINAMATH_CALUDE_original_salary_l1345_134542

def salary_change (x : ℝ) : ℝ := (1 + 0.1) * (1 - 0.05) * x

theorem original_salary : 
  ∃ (x : ℝ), salary_change x = 2090 ∧ x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_original_salary_l1345_134542


namespace NUMINAMATH_CALUDE_intersection_with_y_axis_l1345_134517

/-- The intersection point of y = -4x + 2 with the y-axis is (0, 2) -/
theorem intersection_with_y_axis :
  let f (x : ℝ) := -4 * x + 2
  (0, f 0) = (0, 2) := by sorry

end NUMINAMATH_CALUDE_intersection_with_y_axis_l1345_134517


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1345_134551

/-- A parabola with parameter p > 0 and two points A and B on it, intersected by a line through its focus F -/
structure ParabolaWithIntersection where
  p : ℝ
  hp : p > 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A.2^2 = 2 * p * A.1
  hB : B.2^2 = 2 * p * B.1
  hAF : A.1 = 3 - p/2
  hBF : B.1 = 2 - p/2

/-- The theorem stating that under the given conditions, p = 12/5 -/
theorem parabola_intersection_theorem (pwi : ParabolaWithIntersection) : pwi.p = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1345_134551


namespace NUMINAMATH_CALUDE_n_gon_determination_l1345_134588

/-- The number of elements required to determine an n-gon uniquely -/
def elementsRequired (n : ℕ) : ℕ := 2 * n - 3

/-- The minimum number of sides required among the elements -/
def minSidesRequired (n : ℕ) : ℕ := n - 2

/-- Predicate to check if a number is at least 3 -/
def isAtLeastThree (n : ℕ) : Prop := n ≥ 3

/-- Theorem stating the number of elements and minimum sides required to determine an n-gon -/
theorem n_gon_determination (n : ℕ) (h : isAtLeastThree n) :
  elementsRequired n = 2 * n - 3 ∧ minSidesRequired n = n - 2 :=
by sorry

end NUMINAMATH_CALUDE_n_gon_determination_l1345_134588


namespace NUMINAMATH_CALUDE_kyle_weekly_papers_l1345_134510

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_only : ℕ) : ℕ :=
  6 * weekday_houses + (weekday_houses - sunday_skip + sunday_only)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_weekly_papers :
  weekly_papers 100 10 30 = 720 := by
  sorry

#eval weekly_papers 100 10 30

end NUMINAMATH_CALUDE_kyle_weekly_papers_l1345_134510


namespace NUMINAMATH_CALUDE_exactly_25_sixes_probability_l1345_134560

/-- A cube made of 27 dice -/
structure CubeOfDice :=
  (size : Nat)
  (h_size : size = 27)

/-- The probability of a specific outcome on the surface of the cube -/
def surface_probability (c : CubeOfDice) : ℚ :=
  31 / (2^13 * 3^18)

/-- Theorem: The probability of exactly 25 sixes on the surface of a cube made of 27 dice -/
theorem exactly_25_sixes_probability (c : CubeOfDice) : 
  surface_probability c = 31 / (2^13 * 3^18) := by
  sorry

end NUMINAMATH_CALUDE_exactly_25_sixes_probability_l1345_134560


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1345_134574

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≤ -1 ∨ x ≥ 1) → x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1345_134574


namespace NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1345_134582

/-- A point in the Euclidean plane -/
structure Point :=
  (x y : ℝ)

/-- A triangle in the Euclidean plane -/
structure Triangle :=
  (A B C : Point)

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The touch point of the incircle on a side of a triangle -/
def touchPoint (t : Triangle) : Point :=
  sorry

/-- Theorem: Given the centroid, incenter, and touch point of the incircle on a side,
    a unique triangle can be constructed -/
theorem triangle_construction_uniqueness 
  (M I Q_a : Point) : 
  ∃! t : Triangle, 
    centroid t = M ∧ 
    incenter t = I ∧ 
    touchPoint t = Q_a :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1345_134582


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_logic_l1345_134500

theorem quadratic_inequality_and_logic :
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, x^2 + x + 1 < 0) ∧
  ((∀ x : ℝ, x^2 + x + 1 ≥ 0) → 
   ((∃ x : ℝ, x^2 + x + 1 < 0) ∨ (∀ x : ℝ, x^2 + x + 1 ≥ 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_logic_l1345_134500


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1345_134565

theorem infinite_geometric_series_first_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 40) 
  (h3 : S = a / (1 - r)) : 
  a = 30 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1345_134565


namespace NUMINAMATH_CALUDE_max_x_elements_l1345_134528

/-- Represents the number of elements of each type -/
structure Elements where
  fire : ℕ
  stone : ℕ
  metal : ℕ

/-- Represents the alchemical reactions -/
def reaction1 (e : Elements) : Elements :=
  { fire := e.fire - 1, stone := e.stone - 1, metal := e.metal + 1 }

def reaction2 (e : Elements) : Elements :=
  { fire := e.fire, stone := e.stone + 2, metal := e.metal - 1 }

/-- Creates an element X -/
def createX (e : Elements) : Elements :=
  { fire := e.fire - 2, stone := e.stone - 3, metal := e.metal - 1 }

/-- The initial state of elements -/
def initialElements : Elements :=
  { fire := 50, stone := 50, metal := 0 }

/-- Checks if the number of elements is non-negative -/
def isValid (e : Elements) : Prop :=
  e.fire ≥ 0 ∧ e.stone ≥ 0 ∧ e.metal ≥ 0

/-- Theorem: The maximum number of X elements that can be created is 14 -/
theorem max_x_elements : 
  ∃ (n : ℕ) (e : Elements), 
    n = 14 ∧ 
    isValid e ∧ 
    ∀ m : ℕ, m > n → 
      ¬∃ (f : Elements), isValid f ∧ 
        (∃ (seq : List (Elements → Elements)), 
          f = (seq.foldl (λ acc g => g acc) initialElements) ∧
          (createX^[m]) f = f) :=
sorry

end NUMINAMATH_CALUDE_max_x_elements_l1345_134528


namespace NUMINAMATH_CALUDE_t_equality_and_inequality_l1345_134545

/-- The function t(n, s) represents the maximum number of edges in a graph with n vertices
    that does not contain s independent vertices. -/
noncomputable def t (n s : ℕ) : ℕ := sorry

/-- Theorem stating the equality and inequality for t(n, s) -/
theorem t_equality_and_inequality (n s : ℕ) : 
  t (n - s) s + (n - s) * (s - 1) + (s.choose 2) = t n s ∧ 
  t n s ≤ ⌊((s - 1 : ℚ) / (2 * s : ℚ)) * (n^2 : ℚ)⌋ := by sorry

end NUMINAMATH_CALUDE_t_equality_and_inequality_l1345_134545
