import Mathlib

namespace NUMINAMATH_CALUDE_box_volume_condition_unique_x_existence_l2933_293327

theorem box_volume_condition (x : ℕ) : Bool := 
  (x > 5) ∧ ((x + 5) * (x - 5) * (x^2 + 25) < 700)

theorem unique_x_existence : 
  ∃! x : ℕ, box_volume_condition x := by
  sorry

end NUMINAMATH_CALUDE_box_volume_condition_unique_x_existence_l2933_293327


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2933_293384

theorem smallest_n_for_candy_purchase : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m).gcd 10 = 10 ∧ (15 * m).gcd 16 = 16 ∧ (15 * m).gcd 18 = 18 → n ≤ m) ∧
  (15 * n).gcd 10 = 10 ∧ (15 * n).gcd 16 = 16 ∧ (15 * n).gcd 18 = 18 ∧
  n = 48 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2933_293384


namespace NUMINAMATH_CALUDE_angle_P_measure_l2933_293356

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  P : ℝ  -- Angle P in degrees
  Q : ℝ  -- Angle Q in degrees
  R : ℝ  -- Angle R in degrees
  S : ℝ  -- Angle S in degrees
  angle_relation : P = 3*Q ∧ P = 4*R ∧ P = 6*S
  sum_360 : P + Q + R + S = 360

/-- The measure of angle P in a SpecialQuadrilateral is 206 degrees -/
theorem angle_P_measure (quad : SpecialQuadrilateral) : 
  ⌊quad.P⌋ = 206 := by sorry

end NUMINAMATH_CALUDE_angle_P_measure_l2933_293356


namespace NUMINAMATH_CALUDE_regular_polygon_with_20_diagonals_has_8_sides_l2933_293310

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 20 diagonals has 8 sides -/
theorem regular_polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_20_diagonals_has_8_sides_l2933_293310


namespace NUMINAMATH_CALUDE_not_prime_n_l2933_293380

theorem not_prime_n (p a b c n : ℕ) : 
  Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ a + (n-1) * b →
  p^2 ∣ b + (n-1) * c →
  p^2 ∣ c + (n-1) * a →
  ¬ Prime n :=
by sorry


end NUMINAMATH_CALUDE_not_prime_n_l2933_293380


namespace NUMINAMATH_CALUDE_x_value_proof_l2933_293386

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2933_293386


namespace NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l2933_293347

/-- Represents the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of pourings required to reach exactly one-fifth of the original water -/
def requiredPourings : ℕ := 8

theorem water_remaining_after_required_pourings :
  waterRemaining requiredPourings = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_water_remaining_after_required_pourings_l2933_293347


namespace NUMINAMATH_CALUDE_monotonicity_intervals_max_value_on_interval_min_value_on_interval_l2933_293362

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval of interest
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for intervals of monotonicity
theorem monotonicity_intervals (x : ℝ) :
  (∀ y z, y < x → x < z → y < -1 → z < -1 → f y < f z) ∧
  (∀ y z, y < x → x < z → -1 < y → z < 1 → f y > f z) ∧
  (∀ y z, y < x → x < z → 1 < y → f y < f z) :=
sorry

-- Statement for maximum value on the interval
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 2 :=
sorry

-- Statement for minimum value on the interval
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -18 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_max_value_on_interval_min_value_on_interval_l2933_293362


namespace NUMINAMATH_CALUDE_f_intersects_twice_l2933_293387

/-- An even function that is monotonically increasing for positive x and satisfies f(1) * f(2) < 0 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an even function -/
axiom f_even : ∀ x, f (-x) = f x

/-- f is monotonically increasing for positive x -/
axiom f_increasing : ∀ x y, 0 < x → x < y → f x < f y

/-- f(1) * f(2) < 0 -/
axiom f_sign_change : f 1 * f 2 < 0

/-- The number of intersection points between f and the x-axis -/
def num_intersections : ℕ :=
  sorry

/-- Theorem: The number of intersection points between f and the x-axis is 2 -/
theorem f_intersects_twice : num_intersections = 2 :=
  sorry

end NUMINAMATH_CALUDE_f_intersects_twice_l2933_293387


namespace NUMINAMATH_CALUDE_event3_mutually_exclusive_l2933_293364

-- Define the set of numbers
def NumberSet : Set Nat := {n : Nat | 1 ≤ n ∧ n ≤ 9}

-- Define the property of being even
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define the property of being odd
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Define the events
def Event1 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ ((IsEven a ∧ IsOdd b) ∨ (IsOdd a ∧ IsEven b))

def Event2 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsOdd a ∧ IsOdd b)

def Event3 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∧ IsEven b)

def Event4 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∨ IsEven b)

-- Theorem statement
theorem event3_mutually_exclusive :
  ∀ a b : Nat,
    (Event3 a b → ¬Event1 a b) ∧
    (Event3 a b → ¬Event2 a b) ∧
    (Event3 a b → ¬Event4 a b) :=
sorry

end NUMINAMATH_CALUDE_event3_mutually_exclusive_l2933_293364


namespace NUMINAMATH_CALUDE_find_divisor_l2933_293382

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 217) (h2 : quotient = 54) (h3 : remainder = 1) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2933_293382


namespace NUMINAMATH_CALUDE_angle_system_solutions_l2933_293345

theorem angle_system_solutions :
  ∀ x y : ℝ,
  0 ≤ x ∧ x < 2 * Real.pi ∧ 0 ≤ y ∧ y < 2 * Real.pi →
  Real.sin x + Real.cos y = 0 ∧ Real.cos x * Real.sin y = -1/2 →
  (x = Real.pi/4 ∧ y = 5*Real.pi/4) ∨
  (x = 3*Real.pi/4 ∧ y = 3*Real.pi/4) ∨
  (x = 5*Real.pi/4 ∧ y = Real.pi/4) ∨
  (x = 7*Real.pi/4 ∧ y = 7*Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_angle_system_solutions_l2933_293345


namespace NUMINAMATH_CALUDE_car_sale_profit_l2933_293322

def original_price : ℕ := 50000
def loss_percentage : ℚ := 10 / 100
def gain_percentage : ℚ := 20 / 100

def friend_selling_price : ℕ := 54000

theorem car_sale_profit (original_price : ℕ) (loss_percentage gain_percentage : ℚ) 
  (friend_selling_price : ℕ) : 
  let man_selling_price : ℚ := (1 - loss_percentage) * original_price
  let friend_buying_price : ℚ := man_selling_price
  (1 + gain_percentage) * friend_buying_price = friend_selling_price := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_l2933_293322


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_property_l2933_293358

-- Define the property for the sequence
def isPrimeSequenceWithProperty (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ n, (p (n + 1) : ℤ) - 2 * (p n : ℤ) = 1 ∨ (p (n + 1) : ℤ) - 2 * (p n : ℤ) = -1)

-- State the theorem
theorem no_infinite_prime_sequence_with_property :
  ¬ ∃ p : ℕ → ℕ, isPrimeSequenceWithProperty p :=
sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_property_l2933_293358


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2933_293315

-- Problem 1
theorem problem_1 : -3 - (-10) + (-9) - 10 = -12 := by sorry

-- Problem 2
theorem problem_2 : (1/4 : ℚ) + (-1/8) + (-7/8) - (3/4) = -3/2 := by sorry

-- Problem 3
theorem problem_3 : -25 * (-18) + (-25) * 12 + 25 * (-10) = -100 := by sorry

-- Problem 4
theorem problem_4 : -48 * (-1/6 + 3/4 - 1/24) = -26 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2933_293315


namespace NUMINAMATH_CALUDE_smallest_number_problem_l2933_293366

theorem smallest_number_problem (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 100)
  (h4 : c = 2 * a)
  (h5 : c - b = 10) : 
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l2933_293366


namespace NUMINAMATH_CALUDE_overlap_area_is_half_unit_l2933_293316

-- Define the grid and triangles
def Grid := Fin 3 × Fin 3

def Triangle1 : Set Grid := {(0, 2), (2, 0), (0, 0)}
def Triangle2 : Set Grid := {(2, 2), (0, 0), (1, 0)}

-- Define the area of overlap
def overlap_area (t1 t2 : Set Grid) : ℝ :=
  sorry

-- Theorem statement
theorem overlap_area_is_half_unit :
  overlap_area Triangle1 Triangle2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_half_unit_l2933_293316


namespace NUMINAMATH_CALUDE_cubic_factorization_l2933_293349

theorem cubic_factorization (a : ℝ) : a^3 - 3*a = a*(a + Real.sqrt 3)*(a - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2933_293349


namespace NUMINAMATH_CALUDE_sum_of_powers_l2933_293333

theorem sum_of_powers (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2933_293333


namespace NUMINAMATH_CALUDE_simplified_fraction_l2933_293399

theorem simplified_fraction (a : ℤ) (ha : a > 0) :
  let expr := (a + 1) / a - a / (a + 1)
  let simplified := (2 * a + 1) / (a * (a + 1))
  expr = simplified ∧ (a = 2023 → 2 * a + 1 = 4047) := by
  sorry

#eval 2 * 2023 + 1

end NUMINAMATH_CALUDE_simplified_fraction_l2933_293399


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l2933_293360

theorem orphanage_donation_percentage (total_income : ℝ) 
  (children_percentage : ℝ) (num_children : ℕ) (wife_percentage : ℝ) 
  (remaining_amount : ℝ) :
  total_income = 1200000 →
  children_percentage = 0.2 →
  num_children = 3 →
  wife_percentage = 0.3 →
  remaining_amount = 60000 →
  let distributed_percentage := children_percentage * num_children + wife_percentage
  let distributed_amount := distributed_percentage * total_income
  let amount_before_donation := total_income - distributed_amount
  let donation_amount := amount_before_donation - remaining_amount
  donation_amount / amount_before_donation = 0.5 := by sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l2933_293360


namespace NUMINAMATH_CALUDE_complex_function_property_l2933_293336

/-- A function g on complex numbers defined by g(z) = (c+di)z, where c and d are real numbers. -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating that if g(z) = (c+di)z where c and d are real numbers, 
    and for all complex z, |g(z) - z| = |g(z)|, and |c+di| = 7, then d^2 = 195/4. -/
theorem complex_function_property (c d : ℝ) : 
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) → 
  Complex.abs (c + d * Complex.I) = 7 → 
  d^2 = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l2933_293336


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2933_293375

/-- Given a function g : ℝ → ℝ satisfying the functional equation
    2g(x) - 3g(1/x) = x^2 for all x ≠ 0, prove that g(2) = 8.25 -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 3 * g (1/x) = x^2) : 
  g 2 = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2933_293375


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l2933_293377

def is_prime (n : ℕ) : Prop := sorry

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_sum_of_four_primes : 
  ∃ (a b c d : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (10 < d) ∧ (d < 100) ∧
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
    (digits a ++ digits b ++ digits c ++ digits d).Nodup ∧
    (digits a ++ digits b ++ digits c ++ digits d).length = 9 ∧
    (∀ i, i ∈ digits a ++ digits b ++ digits c ++ digits d → 1 ≤ i ∧ i ≤ 9) ∧
    a + b + c + d = 53 ∧
    (∀ w x y z : ℕ, 
      is_prime w ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧
      (10 < z) ∧ (z < 100) ∧
      (w < 10) ∧ (x < 10) ∧ (y < 10) ∧
      (digits w ++ digits x ++ digits y ++ digits z).Nodup ∧
      (digits w ++ digits x ++ digits y ++ digits z).length = 9 ∧
      (∀ i, i ∈ digits w ++ digits x ++ digits y ++ digits z → 1 ≤ i ∧ i ≤ 9) →
      w + x + y + z ≥ 53) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_primes_l2933_293377


namespace NUMINAMATH_CALUDE_ceiling_minus_fractional_part_l2933_293326

theorem ceiling_minus_fractional_part (x : ℝ) : ⌈x⌉ - (x - ⌊x⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_fractional_part_l2933_293326


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l2933_293324

/-- The duration of the red light in seconds -/
def red_duration : ℕ := 30

/-- The duration of the yellow light in seconds -/
def yellow_duration : ℕ := 5

/-- The duration of the green light in seconds -/
def green_duration : ℕ := 40

/-- The total duration of one traffic light cycle -/
def total_duration : ℕ := red_duration + yellow_duration + green_duration

/-- The probability of seeing a red light -/
def red_light_probability : ℚ := red_duration / total_duration

theorem red_light_probability_is_two_fifths :
  red_light_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l2933_293324


namespace NUMINAMATH_CALUDE_student_height_probability_l2933_293303

theorem student_height_probability (p_less_160 p_between_160_175 : ℝ) :
  p_less_160 = 0.2 →
  p_between_160_175 = 0.5 →
  1 - p_less_160 - p_between_160_175 = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_student_height_probability_l2933_293303


namespace NUMINAMATH_CALUDE_power_series_expansion_of_exp_l2933_293340

open Real

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the n-th term of the power series
noncomputable def power_series_term (a : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (log a)^n / (Nat.factorial n : ℝ) * x^n

-- Theorem statement
theorem power_series_expansion_of_exp (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, f a x = ∑' n, power_series_term a n x :=
sorry

end NUMINAMATH_CALUDE_power_series_expansion_of_exp_l2933_293340


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2933_293332

theorem sum_of_three_numbers (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b * c = (a + b) * (a + c)) : a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2933_293332


namespace NUMINAMATH_CALUDE_min_value_problem_l2933_293379

theorem min_value_problem (x : ℝ) (h : x ≥ 3/2) :
  (∀ y, y ≥ 3/2 → (2*x^2 - 2*x + 1)/(x - 1) ≤ (2*y^2 - 2*y + 1)/(y - 1)) →
  (2*x^2 - 2*x + 1)/(x - 1) = 2*Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2933_293379


namespace NUMINAMATH_CALUDE_exist_six_lines_equal_angles_l2933_293378

/-- A line in 3D space represented by a point and a direction vector -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A set of 6 lines in 3D space -/
def SixLines : Type := Fin 6 → Line3D

/-- Predicate to check if all pairs of lines are non-parallel -/
def all_non_parallel (lines : SixLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- Predicate to check if all pairwise angles are equal -/
def all_angles_equal (lines : SixLines) : Prop :=
  ∀ i j k l, i ≠ j → k ≠ l → angle (lines i) (lines j) = angle (lines k) (lines l)

/-- Theorem stating the existence of 6 lines satisfying the conditions -/
theorem exist_six_lines_equal_angles : 
  ∃ (lines : SixLines), all_non_parallel lines ∧ all_angles_equal lines :=
sorry

end NUMINAMATH_CALUDE_exist_six_lines_equal_angles_l2933_293378


namespace NUMINAMATH_CALUDE_monday_walking_speed_l2933_293374

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  distance_per_day : ℝ
  total_time : ℝ

/-- Theorem stating that Jonathan's Monday walking speed is 2 miles per hour -/
theorem monday_walking_speed (routine : ExerciseRoutine) 
  (h1 : routine.wednesday_speed = 3)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.distance_per_day = 6)
  (h4 : routine.total_time = 6)
  (h5 : routine.distance_per_day / routine.monday_speed + 
        routine.distance_per_day / routine.wednesday_speed + 
        routine.distance_per_day / routine.friday_speed = routine.total_time) :
  routine.monday_speed = 2 := by
  sorry

#check monday_walking_speed

end NUMINAMATH_CALUDE_monday_walking_speed_l2933_293374


namespace NUMINAMATH_CALUDE_two_numbers_product_sum_l2933_293394

theorem two_numbers_product_sum (n : Nat) : n = 45 →
  ∃ x y : Nat, x ∈ Finset.range (n + 1) ∧ 
             y ∈ Finset.range (n + 1) ∧ 
             x < y ∧
             (Finset.sum (Finset.range (n + 1)) id - x - y = x * y) ∧
             y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_sum_l2933_293394


namespace NUMINAMATH_CALUDE_smallest_value_l2933_293395

theorem smallest_value (y : ℝ) (h : y = 8) :
  let a := 5 / (y - 1)
  let b := 5 / (y + 1)
  let c := 5 / y
  let d := (5 + y) / 10
  let e := y - 5
  b < a ∧ b < c ∧ b < d ∧ b < e :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l2933_293395


namespace NUMINAMATH_CALUDE_mileage_scientific_notation_equality_l2933_293353

-- Define the original mileage
def original_mileage : ℝ := 42000

-- Define the scientific notation representation
def scientific_notation : ℝ := 4.2 * (10^4)

-- Theorem to prove the equality
theorem mileage_scientific_notation_equality :
  original_mileage = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_mileage_scientific_notation_equality_l2933_293353


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2933_293367

def A : Set ℤ := {x | x^2 ≤ 16}
def B : Set ℤ := {x | -1 ≤ x ∧ x < 4}

theorem complement_intersection_theorem : 
  (A \ (A ∩ B)) = {-4, -3, -2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2933_293367


namespace NUMINAMATH_CALUDE_terry_more_stickers_than_steven_l2933_293348

/-- Given the number of stickers each person has, prove that Terry has 20 more stickers than Steven -/
theorem terry_more_stickers_than_steven 
  (ryan_stickers : ℕ) 
  (steven_stickers : ℕ) 
  (terry_stickers : ℕ) 
  (total_stickers : ℕ) 
  (h1 : ryan_stickers = 30)
  (h2 : steven_stickers = 3 * ryan_stickers)
  (h3 : terry_stickers > steven_stickers)
  (h4 : ryan_stickers + steven_stickers + terry_stickers = total_stickers)
  (h5 : total_stickers = 230) :
  terry_stickers - steven_stickers = 20 := by
sorry

end NUMINAMATH_CALUDE_terry_more_stickers_than_steven_l2933_293348


namespace NUMINAMATH_CALUDE_box_length_is_twelve_l2933_293355

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

/-- Theorem: If 40 building blocks with given dimensions can fit into a box with given height and width,
    then the length of the box is 12 inches -/
theorem box_length_is_twelve
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.height = 8)
  (h2 : box.width = 10)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box ≥ 40 * volume block) :
  box.length = 12 :=
sorry

end NUMINAMATH_CALUDE_box_length_is_twelve_l2933_293355


namespace NUMINAMATH_CALUDE_prime_power_sum_condition_l2933_293317

theorem prime_power_sum_condition (n : ℕ) :
  Nat.Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_condition_l2933_293317


namespace NUMINAMATH_CALUDE_h_greater_than_two_l2933_293300

theorem h_greater_than_two (x : ℝ) (hx : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end NUMINAMATH_CALUDE_h_greater_than_two_l2933_293300


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2933_293390

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2933_293390


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2933_293309

/-- Given that α is inversely proportional to β, prove that when α = 5 and β = 20, 
    then α = 10 when β = 10 -/
theorem inverse_proportion_problem (α β : ℝ) (k : ℝ) 
    (h1 : α * β = k)  -- α is inversely proportional to β
    (h2 : 5 * 20 = k) -- α = 5 when β = 20
    : 10 * 10 = k :=  -- α = 10 when β = 10
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2933_293309


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2933_293388

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2/3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2933_293388


namespace NUMINAMATH_CALUDE_triangle_formation_check_l2933_293371

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation_check :
  ¬(can_form_triangle 3 3 6) ∧
  ¬(can_form_triangle 2 3 6) ∧
  can_form_triangle 5 8 12 ∧
  ¬(can_form_triangle 4 7 11) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_check_l2933_293371


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2933_293338

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 2 - Real.sqrt 26 ∨ x = 2 + Real.sqrt 26}

/-- The y-coordinate of the intersection points of two parabolas -/
def intersection_y : ℝ := 48

/-- The first parabola function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 18

/-- The second parabola function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 4

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = y ∧ g x = y) ↔ (x ∈ intersection_x ∧ y = intersection_y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2933_293338


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_three_eighths_l2933_293319

theorem sum_of_fractions_equals_three_eighths :
  let sum := (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) +
             (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ))
  sum = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_three_eighths_l2933_293319


namespace NUMINAMATH_CALUDE_total_paths_XZ_l2933_293350

-- Define the number of paths between points
def paths_XY : ℕ := 2
def paths_YZ : ℕ := 2
def direct_paths_XZ : ℕ := 2

-- Theorem statement
theorem total_paths_XZ : paths_XY * paths_YZ + direct_paths_XZ = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_XZ_l2933_293350


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l2933_293302

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) : 
  total_students = 500 → cat_owners = 75 → 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l2933_293302


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l2933_293339

/-- Calculates the total selling amount for cloth given the length, cost price, and loss per metre -/
def total_selling_amount (cloth_length : ℕ) (cost_price_per_metre : ℕ) (loss_per_metre : ℕ) : ℕ :=
  let selling_price_per_metre := cost_price_per_metre - loss_per_metre
  cloth_length * selling_price_per_metre

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 95 Rs per metre 
    and a loss of 5 Rs per metre is 18000 Rs -/
theorem shopkeeper_cloth_sale : 
  total_selling_amount 200 95 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l2933_293339


namespace NUMINAMATH_CALUDE_lily_break_time_l2933_293352

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  words_per_minute : ℕ
  minutes_before_break : ℕ
  total_minutes : ℕ
  total_words : ℕ

/-- Calculates the break time in minutes for a given typing scenario -/
def calculate_break_time (scenario : TypingScenario) : ℕ :=
  sorry

/-- Theorem stating that Lily's break time is 2 minutes -/
theorem lily_break_time :
  let lily_scenario : TypingScenario := {
    words_per_minute := 15,
    minutes_before_break := 10,
    total_minutes := 19,
    total_words := 255
  }
  calculate_break_time lily_scenario = 2 :=
by sorry

end NUMINAMATH_CALUDE_lily_break_time_l2933_293352


namespace NUMINAMATH_CALUDE_basketball_handshakes_l2933_293391

/-- The number of handshakes in a basketball game with specific conditions -/
theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let opposing_team_handshakes := team_size * team_size
  let same_team_handshakes := num_teams * (team_size * (team_size - 1) / 2)
  let player_referee_handshakes := (num_teams * team_size) * num_referees
  opposing_team_handshakes + same_team_handshakes + player_referee_handshakes = 102 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l2933_293391


namespace NUMINAMATH_CALUDE_stamp_collection_duration_l2933_293318

/-- Proves the collection duration for two stamp collectors given their collection rates and total stamps --/
theorem stamp_collection_duration (total_stamps : ℕ) (rate1 rate2 : ℕ) (extra_weeks : ℕ) : 
  total_stamps = 300 →
  rate1 = 5 →
  rate2 = 3 →
  extra_weeks = 20 →
  ∃ (weeks1 weeks2 : ℕ), 
    weeks1 = 30 ∧
    weeks2 = 50 ∧
    weeks2 = weeks1 + extra_weeks ∧
    total_stamps = rate1 * weeks1 + rate2 * weeks2 :=
by sorry


end NUMINAMATH_CALUDE_stamp_collection_duration_l2933_293318


namespace NUMINAMATH_CALUDE_factorial_20_divisibility_l2933_293321

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_dividing (base k : ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / base) 0

theorem factorial_20_divisibility : 
  (highest_power_dividing 12 6 20 = 6) ∧ 
  (highest_power_dividing 10 4 20 = 4) := by
  sorry

end NUMINAMATH_CALUDE_factorial_20_divisibility_l2933_293321


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_6_18_24_l2933_293397

def gcd3 (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem gcd_lcm_sum_6_18_24 : 
  gcd3 6 18 24 + lcm3 6 18 24 = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_6_18_24_l2933_293397


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l2933_293307

/-- The number of times Terrell lifts the weights initially -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The weight of each dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells used in each lift -/
def num_dumbbells : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def required_lifts : ℚ := 12.5

theorem terrell_lifting_equivalence :
  (num_dumbbells * initial_weight * initial_lifts : ℚ) = 
  (num_dumbbells * new_weight * required_lifts) :=
by sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l2933_293307


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2933_293369

theorem largest_constant_inequality (x y : ℝ) :
  (∃ (C : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C * (x + y)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ D * (x + y)) → D ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2933_293369


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2933_293314

/-- The number of ways to select k items from n distinct items, where order matters. -/
def permutations (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

/-- The number of ways to select 2 books from 5 different books and give them to 2 students. -/
def book_distribution_ways : ℕ := permutations 5 2

theorem book_distribution_theorem : book_distribution_ways = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l2933_293314


namespace NUMINAMATH_CALUDE_correct_subtraction_l2933_293365

theorem correct_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2933_293365


namespace NUMINAMATH_CALUDE_camping_site_problem_l2933_293398

theorem camping_site_problem (total : ℕ) (two_weeks_ago : ℕ) (difference : ℕ) :
  total = 150 →
  two_weeks_ago = 40 →
  difference = 10 →
  ∃ (three_weeks_ago last_week : ℕ),
    three_weeks_ago + two_weeks_ago + last_week = total ∧
    two_weeks_ago = three_weeks_ago + difference ∧
    last_week = 80 :=
by
  sorry

#check camping_site_problem

end NUMINAMATH_CALUDE_camping_site_problem_l2933_293398


namespace NUMINAMATH_CALUDE_mystery_number_problem_l2933_293320

theorem mystery_number_problem (x : ℝ) : (x + 12) / 8 = 9 → 35 - x / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_problem_l2933_293320


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_two_l2933_293334

theorem sum_of_roots_greater_than_two (x₁ x₂ : ℝ) 
  (h₁ : 5 * x₁^3 - 6 = 0) 
  (h₂ : 6 * x₂^3 - 5 = 0) : 
  x₁ + x₂ > 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_two_l2933_293334


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2933_293351

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n) →  -- geometric sequence
  (a 1)^2 - 10*(a 1) + 16 = 0 →  -- a_1 is a root
  (a 19)^2 - 10*(a 19) + 16 = 0 →  -- a_19 is a root
  a 8 * a 10 * a 12 = 64 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2933_293351


namespace NUMINAMATH_CALUDE_double_percentage_increase_l2933_293342

theorem double_percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1 + 44 / 100 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_double_percentage_increase_l2933_293342


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l2933_293330

theorem bugs_eating_flowers :
  let bug_amounts : List ℝ := [2.5, 3, 1.5, 2, 4, 0.5, 3]
  bug_amounts.sum = 16.5 := by
sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l2933_293330


namespace NUMINAMATH_CALUDE_sum_pascal_row_21st_triangular_l2933_293325

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of entries in the n-th row of Pascal's triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

theorem sum_pascal_row_21st_triangular : 
  pascal_row_sum (triangular_number 21 - 1) = 2^230 := by sorry

end NUMINAMATH_CALUDE_sum_pascal_row_21st_triangular_l2933_293325


namespace NUMINAMATH_CALUDE_calculation_proof_l2933_293396

theorem calculation_proof : (Real.sqrt 2 - Real.sqrt 3) * (Real.sqrt 2 + Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2933_293396


namespace NUMINAMATH_CALUDE_quadruplet_babies_l2933_293372

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ c : ℕ, 5 * c = number_of_triplet_sets)
  (h_twins : number_of_twin_sets = 2 * number_of_triplet_sets)
  (h_quintuplets : number_of_quintuplet_sets = number_of_quadruplet_sets / 2)
  (h_sum : 2 * number_of_twin_sets + 3 * number_of_triplet_sets + 
           4 * number_of_quadruplet_sets + 5 * number_of_quintuplet_sets = total_babies) :
  4 * number_of_quadruplet_sets = 145 :=
by sorry

-- Define variables
variable (number_of_twin_sets number_of_triplet_sets number_of_quadruplet_sets number_of_quintuplet_sets : ℕ)

end NUMINAMATH_CALUDE_quadruplet_babies_l2933_293372


namespace NUMINAMATH_CALUDE_equation_solution_l2933_293376

theorem equation_solution (m : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ 4 * (x - 1) - m * x + 6 = 8) → 
  m^2 + 2*m - 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2933_293376


namespace NUMINAMATH_CALUDE_eulers_formula_l2933_293329

theorem eulers_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2933_293329


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l2933_293312

theorem radical_product_equals_27 : Real.sqrt (Real.sqrt (Real.sqrt 27 * 27) * 81) * Real.sqrt 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l2933_293312


namespace NUMINAMATH_CALUDE_inequality_proof_l2933_293383

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2933_293383


namespace NUMINAMATH_CALUDE_initial_erasers_count_l2933_293313

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 118

/-- The number of erasers Jason placed in the drawer -/
def erasers_added : ℕ := 131

/-- The total number of erasers after Jason added some -/
def total_erasers : ℕ := 270

/-- The initial number of erasers in the drawer -/
def initial_erasers : ℕ := total_erasers - erasers_added

theorem initial_erasers_count : initial_erasers = 139 := by
  sorry

end NUMINAMATH_CALUDE_initial_erasers_count_l2933_293313


namespace NUMINAMATH_CALUDE_cyclic_iff_perpendicular_l2933_293344

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def is_convex (q : Quadrilateral) : Prop := sorry

def are_perpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

def is_intersection (p : Point) (p1 p2 p3 p4 : Point) : Prop := sorry

def is_midpoint (m : Point) (p1 p2 : Point) : Prop := sorry

def is_cyclic (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem cyclic_iff_perpendicular (q : Quadrilateral) (P M : Point) :
  is_convex q →
  are_perpendicular q.A q.C q.B q.D →
  is_intersection P q.A q.C q.B q.D →
  is_midpoint M q.A q.B →
  (is_cyclic q ↔ are_perpendicular P M q.D q.C) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_iff_perpendicular_l2933_293344


namespace NUMINAMATH_CALUDE_octal_376_equals_decimal_254_l2933_293357

def octal_to_decimal (octal : ℕ) : ℕ :=
  (octal / 100) * 8^2 + ((octal / 10) % 10) * 8^1 + (octal % 10) * 8^0

theorem octal_376_equals_decimal_254 : octal_to_decimal 376 = 254 := by
  sorry

end NUMINAMATH_CALUDE_octal_376_equals_decimal_254_l2933_293357


namespace NUMINAMATH_CALUDE_ratio_of_distances_l2933_293306

/-- Given five consecutive points on a line, prove the ratio of two specific distances -/
theorem ratio_of_distances (E F G H I : ℝ) (hEF : |E - F| = 3) (hFG : |F - G| = 6) 
  (hGH : |G - H| = 4) (hHI : |H - I| = 2) (hOrder : E < F ∧ F < G ∧ G < H ∧ H < I) : 
  |E - G| / |H - I| = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_distances_l2933_293306


namespace NUMINAMATH_CALUDE_stating_sum_of_digits_special_product_l2933_293359

/-- 
Represents the product of numbers of the form (10^k - 1) where k is a power of 2 up to 2^n.
-/
def specialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (10^(2^i) - 1)) 9

/-- 
Represents the sum of digits of a natural number in decimal notation.
-/
def sumOfDigits (m : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the sum of digits of the special product is equal to 9 · 2^n.
-/
theorem sum_of_digits_special_product (n : ℕ) : 
  sumOfDigits (specialProduct n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_stating_sum_of_digits_special_product_l2933_293359


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2933_293323

/-- Represents a license plate with 4 digits and 2 letters -/
structure LicensePlate where
  digits : Fin 10 → Fin 10
  letters : Fin 2 → Fin 26

/-- Checks if a sequence of 4 digits is a palindrome -/
def isPalindrome4 (s : Fin 4 → Fin 10) : Prop :=
  s 0 = s 3 ∧ s 1 = s 2

/-- Checks if a sequence of 2 letters is a palindrome -/
def isPalindrome2 (s : Fin 2 → Fin 26) : Prop :=
  s 0 = s 1

/-- The probability of a license plate containing at least one palindrome sequence -/
def palindromeProbability : ℚ :=
  5 / 104

/-- The main theorem stating the probability of a license plate containing at least one palindrome sequence -/
theorem license_plate_palindrome_probability :
  palindromeProbability = 5 / 104 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2933_293323


namespace NUMINAMATH_CALUDE_y_plus_z_negative_l2933_293393

theorem y_plus_z_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_negative_l2933_293393


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2933_293305

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (2*x - a < 1 ∧ x - 2*b > 3) ↔ (-1 < x ∧ x < 1)) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2933_293305


namespace NUMINAMATH_CALUDE_complex_equality_sum_l2933_293304

theorem complex_equality_sum (a b : ℕ) (h : a > 0 ∧ b > 0) :
  Complex.abs ((a : ℂ) + Complex.I) * Complex.abs (2 + Complex.I) =
  Complex.abs ((b : ℂ) - Complex.I) / Complex.abs (2 - Complex.I) →
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l2933_293304


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2933_293381

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2933_293381


namespace NUMINAMATH_CALUDE_num_dogs_in_pool_l2933_293385

-- Define the total number of legs/paws in the pool
def total_legs : ℕ := 24

-- Define the number of humans in the pool
def num_humans : ℕ := 2

-- Define the number of legs per human
def legs_per_human : ℕ := 2

-- Define the number of legs per dog
def legs_per_dog : ℕ := 4

-- Theorem to prove
theorem num_dogs_in_pool : 
  (total_legs - num_humans * legs_per_human) / legs_per_dog = 5 := by
  sorry


end NUMINAMATH_CALUDE_num_dogs_in_pool_l2933_293385


namespace NUMINAMATH_CALUDE_quadratic_standard_form_l2933_293343

theorem quadratic_standard_form :
  ∀ x : ℝ, (x + 3) * (2 * x - 1) = -4 ↔ 2 * x^2 + 5 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_standard_form_l2933_293343


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2933_293389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- State the theorem
theorem range_of_a_for_decreasing_f :
  (∀ a : ℝ, (∀ x : ℝ, (∀ y : ℝ, x < y → f a x > f a y)) ↔ a ∈ Set.Iic (-3)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2933_293389


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2933_293392

/-- The number of children who got on the bus at a stop -/
def children_at_stop (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem bus_stop_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 18) 
  (h2 : final = 25) :
  children_at_stop initial final = 7 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2933_293392


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_empty_l2933_293368

def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x^2 - 2*x < 3}

theorem intersection_M_complement_N_empty :
  M ∩ (Set.univ \ N) = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_empty_l2933_293368


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l2933_293346

/-- The area of triangle ABC with vertices A(1,3), B(5,1), and C(4,4) is 1/6 of the area of a 6 × 5 rectangle. -/
theorem triangle_area_fraction (A B C : ℝ × ℝ) (h_A : A = (1, 3)) (h_B : B = (5, 1)) (h_C : C = (4, 4)) :
  let triangle_area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  let rectangle_area := 6 * 5
  triangle_area / rectangle_area = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l2933_293346


namespace NUMINAMATH_CALUDE_music_tool_cost_l2933_293311

/-- The cost of Joan's music tool purchase -/
theorem music_tool_cost (trumpet_cost song_book_cost total_spent : ℚ)
  (h1 : trumpet_cost = 149.16)
  (h2 : song_book_cost = 4.14)
  (h3 : total_spent = 163.28) :
  total_spent - (trumpet_cost + song_book_cost) = 9.98 := by
  sorry

end NUMINAMATH_CALUDE_music_tool_cost_l2933_293311


namespace NUMINAMATH_CALUDE_cubic_expression_equal_sixty_times_ten_power_l2933_293331

theorem cubic_expression_equal_sixty_times_ten_power : 
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_equal_sixty_times_ten_power_l2933_293331


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2933_293308

def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2933_293308


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2933_293373

theorem sum_of_four_consecutive_integers_divisible_by_two :
  ∀ n : ℤ, ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2933_293373


namespace NUMINAMATH_CALUDE_sector_area_l2933_293301

/-- Given a circular sector with central angle 3 radians and perimeter 5, its area is 3/2. -/
theorem sector_area (θ : Real) (p : Real) (S : Real) : 
  θ = 3 → p = 5 → S = (θ * (p - θ)) / (2 * (2 + θ)) → S = 3/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2933_293301


namespace NUMINAMATH_CALUDE_f_monotone_increasing_max_a_value_l2933_293341

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1) / (Real.exp x)

theorem f_monotone_increasing :
  ∀ x y, 1 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

theorem max_a_value :
  (∀ x, x > 0 → Real.exp x * f x ≥ a + Real.log x) → a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_max_a_value_l2933_293341


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l2933_293354

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if two circles touch externally -/
def touch_externally (c1 c2 : Circle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_theorem (C1 C2 : Circle) (P Q R : Point) :
  C1.radius = 12 →
  touch_externally C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle R C2 →
  collinear P Q R →
  distance P Q = 7 →
  distance P R = 17 →
  C2.radius = 10 := by sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l2933_293354


namespace NUMINAMATH_CALUDE_delicious_delhi_bill_l2933_293370

/-- Calculates the total bill for a meal at Delicious Delhi restaurant --/
def calculate_bill (
  samosa_price : ℚ)
  (pakora_price : ℚ)
  (lassi_price : ℚ)
  (biryani_price : ℚ)
  (naan_price : ℚ)
  (samosa_quantity : ℕ)
  (pakora_quantity : ℕ)
  (lassi_quantity : ℕ)
  (biryani_quantity : ℕ)
  (naan_quantity : ℕ)
  (biryani_discount_rate : ℚ)
  (service_fee_rate : ℚ)
  (tip_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  sorry

theorem delicious_delhi_bill :
  calculate_bill 2 3 2 (11/2) (3/2) 3 4 1 2 1 (1/10) (3/100) (1/5) (2/25) = 4125/100 :=
sorry

end NUMINAMATH_CALUDE_delicious_delhi_bill_l2933_293370


namespace NUMINAMATH_CALUDE_equation_solutions_l2933_293361

theorem equation_solutions :
  (∃ x : ℝ, 7 * x + 2 * (3 * x - 3) = 20 ∧ x = 2) ∧
  (∃ x : ℝ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2933_293361


namespace NUMINAMATH_CALUDE_max_reflections_is_nine_l2933_293335

/-- Represents the angle between two mirrors in degrees -/
def mirror_angle : ℝ := 10

/-- Represents the increase in angle of incidence after each reflection in degrees -/
def angle_increase : ℝ := 10

/-- Represents the maximum angle at which reflection is possible in degrees -/
def max_reflection_angle : ℝ := 90

/-- Calculates the angle of incidence after a given number of reflections -/
def angle_after_reflections (n : ℕ) : ℝ := n * angle_increase

/-- Determines if reflection is possible after a given number of reflections -/
def is_reflection_possible (n : ℕ) : Prop :=
  angle_after_reflections n ≤ max_reflection_angle

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 9

/-- Theorem stating that the maximum number of reflections is 9 -/
theorem max_reflections_is_nine :
  (∀ n : ℕ, is_reflection_possible n → n ≤ max_reflections) ∧
  is_reflection_possible max_reflections ∧
  ¬is_reflection_possible (max_reflections + 1) :=
sorry

end NUMINAMATH_CALUDE_max_reflections_is_nine_l2933_293335


namespace NUMINAMATH_CALUDE_construction_company_stone_order_l2933_293328

/-- The weight of stone ordered by a construction company -/
theorem construction_company_stone_order
  (concrete : ℝ) (bricks : ℝ) (total : ℝ)
  (h1 : concrete = 0.16666666666666666)
  (h2 : bricks = 0.16666666666666666)
  (h3 : total = 0.8333333333333334) :
  total - (concrete + bricks) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_construction_company_stone_order_l2933_293328


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l2933_293363

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l2933_293363


namespace NUMINAMATH_CALUDE_jack_hand_in_amount_l2933_293337

/-- Represents the number of bills of each denomination in the till -/
structure TillContents where
  hundreds : Nat
  fifties : Nat
  twenties : Nat
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of the bills in the till -/
def totalValue (t : TillContents) : Nat :=
  100 * t.hundreds + 50 * t.fifties + 20 * t.twenties + 10 * t.tens + 5 * t.fives + t.ones

/-- Calculates the amount to be handed in to the main office -/
def amountToHandIn (t : TillContents) (amountToLeave : Nat) : Nat :=
  totalValue t - amountToLeave

/-- Theorem stating that given Jack's till contents and the amount to leave,
    the amount to hand in is $142 -/
theorem jack_hand_in_amount :
  let jacksTill : TillContents := {
    hundreds := 2,
    fifties := 1,
    twenties := 5,
    tens := 3,
    fives := 7,
    ones := 27
  }
  let amountToLeave := 300
  amountToHandIn jacksTill amountToLeave = 142 := by
  sorry


end NUMINAMATH_CALUDE_jack_hand_in_amount_l2933_293337
