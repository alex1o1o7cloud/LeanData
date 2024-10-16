import Mathlib

namespace NUMINAMATH_CALUDE_exists_n_f_div_g_eq_2012_l3817_381722

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_f_div_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_f_div_g_eq_2012_l3817_381722


namespace NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l3817_381783

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 3 = 0

-- Define the given parallel line
def parallel_line (x y : ℝ) : Prop := x + 2*y + 11 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -2)

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x + 2*y + 1 = 0

-- Theorem statement
theorem line_through_circle_center_parallel_to_given_line :
  ∀ (x y : ℝ),
    (target_line x y ↔ 
      (∃ (t : ℝ), x = circle_center.1 + t ∧ y = circle_center.2 - t/2) ∧
      (∀ (x₁ y₁ x₂ y₂ : ℝ), target_line x₁ y₁ ∧ target_line x₂ y₂ → 
        y₂ - y₁ = -(1/2) * (x₂ - x₁))) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_parallel_to_given_line_l3817_381783


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3817_381704

/-- An arithmetic sequence with a_1 = 3 and a_5 = 7 has its 9th term equal to 11 -/
theorem arithmetic_sequence_ninth_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                                -- first term condition
    a 5 = 7 →                                -- fifth term condition
    a 9 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3817_381704


namespace NUMINAMATH_CALUDE_complex_magnitude_l3817_381799

theorem complex_magnitude (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + i) / 2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3817_381799


namespace NUMINAMATH_CALUDE_charity_fundraising_l3817_381759

theorem charity_fundraising (people : ℕ) (total_amount : ℕ) (amount_per_person : ℕ) :
  people = 8 →
  total_amount = 3000 →
  amount_per_person * people = total_amount →
  amount_per_person = 375 := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l3817_381759


namespace NUMINAMATH_CALUDE_additional_spend_for_free_delivery_l3817_381786

/-- The minimum amount required for free delivery -/
def min_for_free_delivery : ℚ := 35

/-- The price of chicken per pound -/
def chicken_price_per_pound : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The price of lettuce -/
def lettuce_price : ℚ := 3

/-- The price of cherry tomatoes -/
def cherry_tomatoes_price : ℚ := 5/2

/-- The price of a sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The price of a broccoli head -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The price of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price_per_pound * chicken_amount +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_count +
  broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The theorem stating how much more Alice needs to spend for free delivery -/
theorem additional_spend_for_free_delivery :
  min_for_free_delivery - cart_total = 11 := by sorry

end NUMINAMATH_CALUDE_additional_spend_for_free_delivery_l3817_381786


namespace NUMINAMATH_CALUDE_divisibility_implies_divisibility_l3817_381749

theorem divisibility_implies_divisibility (a b m n : ℕ) 
  (ha : a > 1) (hcoprime : Nat.Coprime a b) :
  (((a^m + 1) ∣ (a^n + 1)) → (m ∣ n)) ∧
  (((a^m + b^m) ∣ (a^n + b^n)) → (m ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_divisibility_l3817_381749


namespace NUMINAMATH_CALUDE_badge_exchange_l3817_381762

theorem badge_exchange (x : ℕ) : 
  -- Vasya initially had 5 more badges than Tolya
  let vasya_initial := x + 5
  -- Vasya exchanged 24% of his badges for 20% of Tolya's badges
  let vasya_final := vasya_initial - (24 * vasya_initial) / 100 + (20 * x) / 100
  let tolya_final := x - (20 * x) / 100 + (24 * vasya_initial) / 100
  -- After the exchange, Vasya had one badge less than Tolya
  vasya_final + 1 = tolya_final →
  -- Prove that Tolya initially had 45 badges and Vasya initially had 50 badges
  x = 45 ∧ vasya_initial = 50 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_l3817_381762


namespace NUMINAMATH_CALUDE_cost_difference_l3817_381766

/-- Represents the pricing policy of the store -/
def pencil_cost (quantity : ℕ) : ℚ :=
  if quantity < 40 then 4 else (7/2)

/-- Calculate the total cost for a given quantity of pencils -/
def total_cost (quantity : ℕ) : ℚ :=
  (pencil_cost quantity) * quantity

/-- The number of pencils Joy bought -/
def joy_pencils : ℕ := 30

/-- The number of pencils Colleen bought -/
def colleen_pencils : ℕ := 50

/-- Theorem stating the difference in cost between Colleen's and Joy's purchases -/
theorem cost_difference : 
  total_cost colleen_pencils - total_cost joy_pencils = 55 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l3817_381766


namespace NUMINAMATH_CALUDE_alexa_first_day_pages_l3817_381771

/-- The number of pages Alexa read on the first day of reading a Nancy Drew mystery. -/
def pages_read_first_day (total_pages second_day_pages pages_left : ℕ) : ℕ :=
  total_pages - second_day_pages - pages_left

/-- Theorem stating that Alexa read 18 pages on the first day. -/
theorem alexa_first_day_pages :
  pages_read_first_day 95 58 19 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alexa_first_day_pages_l3817_381771


namespace NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l3817_381728

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ ∃ y > 0, y + 81 / y = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l3817_381728


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3817_381795

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 19

/-- The difference between the grasshopper's jump and the frog's jump in inches -/
def grasshopper_frog_diff : ℕ := 4

/-- The difference between the frog's jump and the mouse's jump in inches -/
def frog_mouse_diff : ℕ := 44

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := grasshopper_jump - grasshopper_frog_diff

theorem frog_jump_distance : frog_jump = 15 := by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l3817_381795


namespace NUMINAMATH_CALUDE_unique_conjugate_pair_l3817_381765

/-- A quadratic trinomial function -/
def QuadraticTrinomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Conjugate numbers for a function -/
def Conjugate (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y ∧ f y = x

theorem unique_conjugate_pair (a b c : ℝ) (x y : ℝ) :
  x ≠ y →
  let f := QuadraticTrinomial a b c
  Conjugate f x y →
  ∀ u v : ℝ, Conjugate f u v → (u = x ∧ v = y) ∨ (u = y ∧ v = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_conjugate_pair_l3817_381765


namespace NUMINAMATH_CALUDE_product_w_z_l3817_381754

/-- A parallelogram with side lengths defined in terms of w and z -/
structure Parallelogram (w z : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 50)
  (fg_eq : fg = 4 * z^2)
  (gh_eq : gh = 3 * w + 6)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of w and z in the given parallelogram is 88√2/3 -/
theorem product_w_z (w z : ℝ) (p : Parallelogram w z) : w * z = 88 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_w_z_l3817_381754


namespace NUMINAMATH_CALUDE_solution_existence_l3817_381717

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations has a solution -/
def has_solution (K : ℤ) : Prop :=
  ∃ (x y : ℝ), (2 * (floor x) + y = 3/2) ∧ ((floor x - x)^2 - 2 * (floor y) = K)

/-- The theorem stating the conditions for the existence of a solution -/
theorem solution_existence (K : ℤ) :
  has_solution K ↔ ∃ (M : ℤ), K = 4*M - 2 ∧ has_solution (4*M - 2) :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l3817_381717


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l3817_381742

/-- The number of kids from Lawrence county who stay home during summer break -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 590796 kids from Lawrence county stay home during summer break -/
theorem lawrence_county_kids_at_home : 
  kids_stay_home 1201565 610769 = 590796 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l3817_381742


namespace NUMINAMATH_CALUDE_billboard_count_l3817_381752

theorem billboard_count (B : ℕ) : 
  (B + 20 + 23) / 3 = 20 → B = 17 := by
  sorry

end NUMINAMATH_CALUDE_billboard_count_l3817_381752


namespace NUMINAMATH_CALUDE_A_when_half_in_A_B_values_l3817_381746

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

theorem A_when_half_in_A (a : ℝ) (h : (1/2 : ℝ) ∈ A a) : 
  A a = {-(1/4), 1/2} := by sorry

def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

theorem B_values : B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_when_half_in_A_B_values_l3817_381746


namespace NUMINAMATH_CALUDE_total_cost_ratio_l3817_381740

-- Define the cost of shorts
variable (x : ℝ)

-- Define the costs of other items based on the given conditions
def cost_tshirt : ℝ := x
def cost_boots : ℝ := 4 * x
def cost_shinguards : ℝ := 2 * x

-- State the theorem
theorem total_cost_ratio : 
  (x + cost_tshirt x + cost_boots x + cost_shinguards x) / x = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_ratio_l3817_381740


namespace NUMINAMATH_CALUDE_travel_time_ratio_l3817_381716

/-- Proves that the ratio of the time taken to travel a fixed distance at a given speed
    to the time taken to travel the same distance in a given time is equal to a specific ratio. -/
theorem travel_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 360 ∧ original_time = 6 ∧ new_speed = 40 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l3817_381716


namespace NUMINAMATH_CALUDE_system_solution_l3817_381745

theorem system_solution : 
  ∃! (x y : ℚ), (2010 * x - 2011 * y = 2009) ∧ (2009 * x - 2008 * y = 2010) ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3817_381745


namespace NUMINAMATH_CALUDE_sue_driving_days_l3817_381727

theorem sue_driving_days (total_cost : ℚ) (sister_days : ℚ) (sue_payment : ℚ) :
  total_cost = 2100 →
  sister_days = 4 →
  sue_payment = 900 →
  ∃ (sue_days : ℚ), sue_days + sister_days = 7 ∧ sue_days / (7 - sue_days) = sue_payment / (total_cost - sue_payment) ∧ sue_days = 3 :=
by sorry

end NUMINAMATH_CALUDE_sue_driving_days_l3817_381727


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3817_381748

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a + 1 ≥ 0 ∧ 3 - 2*x > 0))) → 
  -1 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3817_381748


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3817_381744

/-- Represents the total number of students in the school -/
def total_students : ℕ := 600 + 500 + 400

/-- Represents the number of students in the first grade -/
def first_grade_students : ℕ := 600

/-- Represents the number of first-grade students in the sample -/
def first_grade_sample : ℕ := 30

/-- Theorem stating that the total sample size is 75 given the conditions -/
theorem stratified_sample_size :
  ∃ (n : ℕ),
    n * first_grade_students = total_students * first_grade_sample ∧
    n = 75 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3817_381744


namespace NUMINAMATH_CALUDE_one_third_1206_is_100_5_percent_of_400_l3817_381750

theorem one_third_1206_is_100_5_percent_of_400 :
  (1206 / 3) / 400 = 1.005 := by
  sorry

end NUMINAMATH_CALUDE_one_third_1206_is_100_5_percent_of_400_l3817_381750


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l3817_381763

def num_houses : ℕ := 5

def num_correct_deliveries : ℕ := 3

def probability_correct_deliveries : ℚ :=
  (num_houses.choose num_correct_deliveries : ℚ) / (num_houses.factorial : ℚ)

theorem correct_delivery_probability :
  probability_correct_deliveries = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l3817_381763


namespace NUMINAMATH_CALUDE_sum_in_range_l3817_381780

theorem sum_in_range : 
  let sum := (5/4 : ℚ) + (13/3 : ℚ) + (73/12 : ℚ)
  11.5 < sum ∧ sum < 12 := by sorry

end NUMINAMATH_CALUDE_sum_in_range_l3817_381780


namespace NUMINAMATH_CALUDE_self_checkout_increase_is_20_percent_l3817_381720

/-- The percentage increase in complaints when the self-checkout is broken -/
def self_checkout_increase (normal_complaints : ℕ) (short_staffed_increase : ℚ) (total_complaints : ℕ) : ℚ :=
  let short_staffed_complaints := normal_complaints * (1 + short_staffed_increase)
  let daily_complaints_both := total_complaints / 3
  (daily_complaints_both - short_staffed_complaints) / short_staffed_complaints * 100

/-- Theorem stating that the percentage increase when self-checkout is broken is 20% -/
theorem self_checkout_increase_is_20_percent :
  self_checkout_increase 120 (1/3) 576 = 20 := by
  sorry

end NUMINAMATH_CALUDE_self_checkout_increase_is_20_percent_l3817_381720


namespace NUMINAMATH_CALUDE_sequence_properties_l3817_381787

/-- An arithmetic sequence a_n with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 4 = 14 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- A sequence b_n with given conditions -/
def b_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 1 ∧ b 4 = 6

/-- The difference sequence a_n - b_n is geometric -/
def difference_is_geometric (a b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, (a (n + 1) - b (n + 1)) / (a n - b n) = r

/-- The existence of a maximum term in b_n -/
def b_has_max (b : ℕ → ℝ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → b n ≤ b k

/-- The main theorem -/
theorem sequence_properties
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : b_sequence b)
  (hd : difference_is_geometric a b)
  (hm : b_has_max b) :
  (∀ n : ℕ, n > 0 → a n = 4 * n - 2) ∧
  (∀ n : ℕ, n > 0 → b n = 4 * n - 2 - 2^(n-1)) ∧
  (∃ k : ℕ, (k = 3 ∨ k = 4) ∧ ∀ n : ℕ, n > 0 → b n ≤ b k) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3817_381787


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficient_l3817_381735

theorem polynomial_factor_coefficient (a b : ℤ) : 
  (∃ (c d : ℤ), ∀ (x : ℝ), 
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4)) →
  a = 112 ∧ b = -152 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficient_l3817_381735


namespace NUMINAMATH_CALUDE_apple_cost_is_75_cents_l3817_381753

/-- The cost of an apple given the amount paid and change received -/
def appleCost (amountPaid change : ℚ) : ℚ :=
  amountPaid - change

/-- Proof that the apple costs $0.75 given the conditions -/
theorem apple_cost_is_75_cents (amountPaid change : ℚ) 
  (h1 : amountPaid = 5)
  (h2 : change = 4.25) : 
  appleCost amountPaid change = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_75_cents_l3817_381753


namespace NUMINAMATH_CALUDE_simplify_power_sum_l3817_381705

theorem simplify_power_sum : (-2)^2003 + 2^2004 + (-2)^2005 - 2^2006 = 5 * 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_sum_l3817_381705


namespace NUMINAMATH_CALUDE_sam_final_marbles_l3817_381719

/-- Represents the number of marbles each person has -/
structure Marbles where
  steve : ℕ
  sam : ℕ
  sally : ℕ

/-- Represents the initial distribution of marbles -/
def initial_marbles (steve_marbles : ℕ) : Marbles :=
  { steve := steve_marbles,
    sam := 2 * steve_marbles,
    sally := 2 * steve_marbles - 5 }

/-- Represents the distribution of marbles after the exchange -/
def final_marbles (m : Marbles) : Marbles :=
  { steve := m.steve + 3,
    sam := m.sam - 6,
    sally := m.sally + 3 }

/-- Theorem stating that Sam ends up with 8 marbles -/
theorem sam_final_marbles :
  ∀ (initial : Marbles),
    initial.sam = 2 * initial.steve →
    initial.sally = initial.sam - 5 →
    (final_marbles initial).steve = 10 →
    (final_marbles initial).sam = 8 :=
by sorry

end NUMINAMATH_CALUDE_sam_final_marbles_l3817_381719


namespace NUMINAMATH_CALUDE_sum_remainders_divisible_by_500_l3817_381776

/-- The set of all possible remainders when 3^n (n is a nonnegative integer) is divided by 500 -/
def R : Finset ℕ :=
  sorry

/-- The sum of all elements in R -/
def S : ℕ := sorry

/-- Theorem: The sum of all distinct remainders when 3^n (n is a nonnegative integer) 
    is divided by 500 is divisible by 500 -/
theorem sum_remainders_divisible_by_500 : 500 ∣ S := by
  sorry

end NUMINAMATH_CALUDE_sum_remainders_divisible_by_500_l3817_381776


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l3817_381714

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_12th_term
  (a : ℕ → ℝ)
  (is_geometric : GeometricSequence a)
  (fifth_term : a 5 = 5)
  (eighth_term : a 8 = 40) :
  a 12 = 640 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l3817_381714


namespace NUMINAMATH_CALUDE_divided_square_longer_side_l3817_381756

/-- Represents a square divided into a trapezoid and hexagon -/
structure DividedSquare where
  side_length : ℝ
  trapezoid_area : ℝ
  hexagon_area : ℝ
  longer_parallel_side : ℝ

/-- The properties of our specific divided square -/
def my_square : DividedSquare where
  side_length := 2
  trapezoid_area := 2
  hexagon_area := 2
  longer_parallel_side := 2  -- This is what we want to prove

theorem divided_square_longer_side (s : DividedSquare) 
  (h1 : s.side_length = 2)
  (h2 : s.trapezoid_area = s.hexagon_area)
  (h3 : s.trapezoid_area + s.hexagon_area = s.side_length ^ 2)
  (h4 : s.trapezoid_area = (s.longer_parallel_side + s.side_length) * (s.side_length / 2) / 2) :
  s.longer_parallel_side = 2 := by
  sorry

#check divided_square_longer_side my_square

end NUMINAMATH_CALUDE_divided_square_longer_side_l3817_381756


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3817_381724

theorem smallest_k_with_remainder_one : ∃ k : ℕ, 
  k > 1 ∧ 
  k % 10 = 1 ∧ 
  k % 15 = 1 ∧ 
  k % 9 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 10 = 1 ∧ m % 15 = 1 ∧ m % 9 = 1 → k ≤ m) ∧
  k = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l3817_381724


namespace NUMINAMATH_CALUDE_equation_solution_l3817_381777

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x + 2) = x + 3 := by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3817_381777


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3817_381770

theorem arithmetic_sequence_sixth_term 
  (a : ℕ → ℚ)  -- a is a sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_first : a 1 = 3/8)  -- first term is 3/8
  (h_eleventh : a 11 = 5/6)  -- eleventh term is 5/6
  : a 6 = 29/48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3817_381770


namespace NUMINAMATH_CALUDE_oliver_workout_ratio_l3817_381774

/-- Oliver's workout schedule problem -/
theorem oliver_workout_ratio :
  let monday : ℕ := 4  -- Monday's workout hours
  let tuesday : ℕ := monday - 2  -- Tuesday's workout hours
  let thursday : ℕ := 2 * tuesday  -- Thursday's workout hours
  let total : ℕ := 18  -- Total workout hours over four days
  let wednesday : ℕ := total - (monday + tuesday + thursday)  -- Wednesday's workout hours
  (wednesday : ℚ) / monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_workout_ratio_l3817_381774


namespace NUMINAMATH_CALUDE_tangent_line_parabola_l3817_381706

/-- The equation of the tangent line to the parabola y = x^2 that is parallel to the line y = 2x is 2x - y - 1 = 0 -/
theorem tangent_line_parabola (x y : ℝ) : 
  (y = x^2) →  -- parabola equation
  (∃ m : ℝ, m = 2) →  -- parallel to y = 2x
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ 
    ∀ x₀ y₀ : ℝ, y₀ = x₀^2 → (y₀ - (x₀^2) = m * (x - x₀))) →  -- tangent line equation
  (2 * x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l3817_381706


namespace NUMINAMATH_CALUDE_inequality_proof_l3817_381782

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b*c) / (a*(b+c)) + (b^2 + c*a) / (b*(c+a)) + (c^2 + a*b) / (c*(a+b)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3817_381782


namespace NUMINAMATH_CALUDE_berry_cobbler_cartons_l3817_381781

/-- The number of cartons of berries Maria needs for her cobbler -/
theorem berry_cobbler_cartons (strawberries blueberries additional : ℕ) 
  (h1 : strawberries = 4)
  (h2 : blueberries = 8)
  (h3 : additional = 9) :
  strawberries + blueberries + additional = 21 := by
  sorry

end NUMINAMATH_CALUDE_berry_cobbler_cartons_l3817_381781


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l3817_381713

theorem max_value_of_x_plus_y : ∃ (max : ℤ),
  (max = 13) ∧
  (∀ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 → x + y ≤ max) ∧
  (∃ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 ∧ x + y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l3817_381713


namespace NUMINAMATH_CALUDE_multiples_of_2002_count_l3817_381793

theorem multiples_of_2002_count : ∃ (count : ℕ), 
  count = (Finset.filter 
    (λ (pair : ℕ × ℕ) => 
      let (i, j) := pair
      2002 ∣ (10^j - 10^i) ∧ i < j)
    (Finset.product (Finset.range 200) (Finset.range 200))).card ∧
  count = 6468 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_2002_count_l3817_381793


namespace NUMINAMATH_CALUDE_only_happiness_symmetrical_l3817_381731

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
| happiness : ChineseCharacter  -- 喜
| longevity : ChineseCharacter  -- 寿
| blessing : ChineseCharacter   -- 福
| prosperity : ChineseCharacter -- 禄

-- Define symmetry for Chinese characters
def isSymmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.happiness => true
  | _ => false

-- Theorem statement
theorem only_happiness_symmetrical :
  ∀ c : ChineseCharacter, isSymmetrical c ↔ c = ChineseCharacter.happiness :=
by sorry

end NUMINAMATH_CALUDE_only_happiness_symmetrical_l3817_381731


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l3817_381723

theorem circumcircle_radius_of_triangle (a b c : ℚ) :
  a = 15/2 ∧ b = 10 ∧ c = 25/2 →
  a^2 + b^2 = c^2 →
  (c/2 : ℚ) = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l3817_381723


namespace NUMINAMATH_CALUDE_triangle_area_from_lines_l3817_381725

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines (f g h : ℝ → ℝ) :
  (f = fun x ↦ x + 2) →
  (g = fun x ↦ -3*x + 9) →
  (h = fun _ ↦ 2) →
  let p₁ := (0, 2)
  let p₂ := (7/3, 2)
  let p₃ := (7/4, 15/4)
  let base := p₂.1 - p₁.1
  let height := p₃.2 - 2
  1/2 * base * height = 49/24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_lines_l3817_381725


namespace NUMINAMATH_CALUDE_simplify_fraction_l3817_381761

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3817_381761


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_plus_7_l3817_381764

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_plus_7 :
  units_digit (factorial_sum 25 + 7) = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_plus_7_l3817_381764


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_area_l3817_381708

/-- Given a rectangle with dimensions 32 cm * 10 cm, if the area of a square is five times
    the area of this rectangle, then the perimeter of the square is 160 cm. -/
theorem square_perimeter_from_rectangle_area : 
  let rectangle_length : ℝ := 32
  let rectangle_width : ℝ := 10
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_area_l3817_381708


namespace NUMINAMATH_CALUDE_ninth_row_sum_l3817_381734

/-- Yang Hui's Triangle (Pascal's Triangle) -/
def yangHuiTriangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Sum of elements in a row of Yang Hui's Triangle -/
def rowSum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map (yangHuiTriangle n) |>.sum

/-- Theorem: The sum of all numbers in the 9th row of Yang Hui's Triangle is 2^8 -/
theorem ninth_row_sum : rowSum 8 = 2^8 := by
  sorry

end NUMINAMATH_CALUDE_ninth_row_sum_l3817_381734


namespace NUMINAMATH_CALUDE_inscribed_square_in_acute_triangle_l3817_381741

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngledTriangle (A B C : Point) : Prop := sorry

/-- A square is inscribed in a triangle if all its vertices lie on the sides of the triangle -/
def IsInscribedSquare (K L M N : Point) (A B C : Point) : Prop := sorry

/-- Two points lie on the same side of a triangle -/
def LieOnSameSide (P Q : Point) (A B C : Point) : Prop := sorry

theorem inscribed_square_in_acute_triangle 
  (A B C : Point) (h : IsAcuteAngledTriangle A B C) :
  ∃ (K L M N : Point), 
    IsInscribedSquare K L M N A B C ∧ 
    LieOnSameSide L M A B C ∧
    ((LieOnSameSide K N A B C ∧ ¬LieOnSameSide K N B C A) ∨
     (LieOnSameSide K N B C A ∧ ¬LieOnSameSide K N A B C)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_in_acute_triangle_l3817_381741


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3817_381769

/-- Given an initial angle of 45 degrees that is rotated 510 degrees clockwise,
    the resulting new acute angle measures 75 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 45 → 
  rotation = 510 → 
  (((rotation % 360) - initial_angle) % 180) = 75 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3817_381769


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3817_381772

/-- Proves that the cost of an adult ticket is 8 dollars given the specified conditions -/
theorem adult_ticket_cost
  (total_attendees : ℕ)
  (num_children : ℕ)
  (child_ticket_cost : ℕ)
  (total_revenue : ℕ)
  (h1 : total_attendees = 22)
  (h2 : num_children = 18)
  (h3 : child_ticket_cost = 1)
  (h4 : total_revenue = 50) :
  (total_revenue - num_children * child_ticket_cost) / (total_attendees - num_children) = 8 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3817_381772


namespace NUMINAMATH_CALUDE_product_of_roots_l3817_381785

theorem product_of_roots (x : ℝ) : 
  let equation := (16 : ℝ) * x^2 + 60 * x - 200
  let product_of_roots := -200 / 16
  equation = 0 → product_of_roots = -(25 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3817_381785


namespace NUMINAMATH_CALUDE_range_of_a_l3817_381726

/-- The set of real numbers x satisfying the condition p -/
def set_p (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The set of real numbers x satisfying the condition q -/
def set_q : Set ℝ := {x | x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0}

/-- The theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (set_p a)ᶜ ⊂ (set_q)ᶜ → 
  (set_p a)ᶜ ≠ (set_q)ᶜ → 
  (-4 ≤ a ∧ a < 0) ∨ (a ≤ -4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3817_381726


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3817_381789

theorem fraction_equivalence (a1 a2 b1 b2 : ℝ) :
  (∀ x : ℝ, x + a2 ≠ 0 → (x + a1) / (x + a2) = b1 / b2) ↔ (b2 = b1 ∧ b1 * a2 = a1 * b2) :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3817_381789


namespace NUMINAMATH_CALUDE_equation_solution_l3817_381736

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 34 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3817_381736


namespace NUMINAMATH_CALUDE_bus_ticket_cost_l3817_381743

theorem bus_ticket_cost 
  (total_tickets : ℕ)
  (senior_ticket_cost : ℕ)
  (total_sales : ℕ)
  (senior_tickets_sold : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_ticket_cost = 10)
  (h3 : total_sales = 855)
  (h4 : senior_tickets_sold = 24) :
  (total_sales - senior_tickets_sold * senior_ticket_cost) / (total_tickets - senior_tickets_sold) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bus_ticket_cost_l3817_381743


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_M_l3817_381739

-- Define the circle C
def circle_C (k : ℝ) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = k ∧ k > 0

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (yA yB xE xF : ℝ),
    circle_C k 0 yA ∧ circle_C k 0 yB ∧ yA > yB ∧
    circle_C k xE 0 ∧ circle_C k xF 0 ∧ xE > xF

-- Define the midpoint M of AE
def midpoint_M (x y yA xE : ℝ) : Prop :=
  x = (0 + xE) / 2 ∧ y = (yA + 0) / 2

-- Theorem statement
theorem trajectory_of_midpoint_M
  (k : ℝ) (x y : ℝ) :
  circle_C k x y →
  intersection_points k →
  (∃ (yA xE : ℝ), midpoint_M x y yA xE) →
  x > 1 →
  y > 2 + Real.sqrt 3 →
  (y - 2)^2 - (x - 1)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_M_l3817_381739


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3817_381702

theorem polynomial_divisibility : ∃ q : Polynomial ℂ, 
  X^66 + X^55 + X^44 + X^33 + X^22 + X^11 + 1 = 
  q * (X^6 + X^5 + X^4 + X^3 + X^2 + X + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3817_381702


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3817_381778

theorem sqrt_calculation : Real.sqrt (1/2) * Real.sqrt 8 - (Real.sqrt 3)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3817_381778


namespace NUMINAMATH_CALUDE_y_exceeds_x_by_25_percent_l3817_381712

theorem y_exceeds_x_by_25_percent (x y : ℝ) (h : x = 0.8 * y) : 
  (y - x) / x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_y_exceeds_x_by_25_percent_l3817_381712


namespace NUMINAMATH_CALUDE_tan_beta_value_l3817_381710

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3817_381710


namespace NUMINAMATH_CALUDE_grain_mass_calculation_l3817_381711

theorem grain_mass_calculation (given_mass : ℝ) (given_fraction : ℝ) (total_mass : ℝ) : 
  given_mass = 0.5 → given_fraction = 0.2 → given_mass = given_fraction * total_mass → total_mass = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_grain_mass_calculation_l3817_381711


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3817_381779

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : f a b c 0 = 0)
  (h2 : f a b c 4 = 0)
  (h3 : f a b c 3 = 9) :
  -b / (2 * a) = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3817_381779


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3817_381751

theorem absolute_value_inequality (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3817_381751


namespace NUMINAMATH_CALUDE_system_solution_l3817_381709

theorem system_solution (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), 
    (a^(7*x) * a^(15*y) = (a^19)^(1/2)) ∧ 
    ((a^(25*y))^(1/3) / (a^(13*x))^(1/2) = a^(1/12)) ∧
    x = 1/2 ∧ y = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3817_381709


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3817_381733

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3817_381733


namespace NUMINAMATH_CALUDE_find_other_number_l3817_381775

theorem find_other_number (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 2310)
  (h_gcd : Nat.gcd A B = 30)
  (h_A : A = 770) : 
  B = 90 := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l3817_381775


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3817_381768

theorem tan_alpha_value (α : Real) (h : Real.tan α = -1/2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3817_381768


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3817_381730

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The diameter of the circle is along the base of the triangle -/
  diameter_along_base : Bool

/-- Theorem: The radius of the inscribed circle in the given isosceles triangle is 120/13 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 24) 
  (h3 : triangle.diameter_along_base = true) : 
  triangle.radius = 120 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3817_381730


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_intersecting_circles_l3817_381758

/-- Given two circles with radii R and r that touch a common line and intersect each other,
    the radius ρ of the circumcircle of the triangle formed by their two points of tangency
    and one point of intersection is equal to √(R * r). -/
theorem circumcircle_radius_of_intersecting_circles (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (ρ : ℝ), ρ > 0 ∧ ρ * ρ = R * r := by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_intersecting_circles_l3817_381758


namespace NUMINAMATH_CALUDE_xy_value_from_absolute_sum_l3817_381755

theorem xy_value_from_absolute_sum (x y : ℝ) :
  |x - 5| + |y + 3| = 0 → x * y = -15 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_from_absolute_sum_l3817_381755


namespace NUMINAMATH_CALUDE_algebraic_identities_l3817_381790

theorem algebraic_identities :
  (∃ (x : ℝ), x^2 = 3 ∧ x > 0) ∧ 
  (∃ (y : ℝ), y^2 = 2 ∧ y > 0) →
  (3 * Real.sqrt 3 - (Real.sqrt 12 + Real.sqrt (1/3)) = 2 * Real.sqrt 3 / 3) ∧
  ((1 + Real.sqrt 2) * (2 - Real.sqrt 2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3817_381790


namespace NUMINAMATH_CALUDE_value_added_after_doubling_l3817_381788

theorem value_added_after_doubling (x : ℝ) (v : ℝ) : 
  x = 4 → 2 * x + v = x / 2 + 20 → v = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_added_after_doubling_l3817_381788


namespace NUMINAMATH_CALUDE_zero_in_P_and_two_not_in_P_l3817_381718

-- Define the set P
def P : Set Int := sorry

-- Define the properties of P
axiom P_contains_positive : ∃ x : Int, x > 0 ∧ x ∈ P
axiom P_contains_negative : ∃ x : Int, x < 0 ∧ x ∈ P
axiom P_contains_odd : ∃ x : Int, x % 2 ≠ 0 ∧ x ∈ P
axiom P_contains_even : ∃ x : Int, x % 2 = 0 ∧ x ∈ P
axiom P_not_contains_neg_one : -1 ∉ P
axiom P_closed_under_addition : ∀ x y : Int, x ∈ P → y ∈ P → (x + y) ∈ P

-- Theorem to prove
theorem zero_in_P_and_two_not_in_P : 0 ∈ P ∧ 2 ∉ P := by
  sorry

end NUMINAMATH_CALUDE_zero_in_P_and_two_not_in_P_l3817_381718


namespace NUMINAMATH_CALUDE_f_of_3x_plus_2_l3817_381794

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_3x_plus_2 (x : ℝ) : f (3 * x + 2) = 9 * x^2 + 12 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3x_plus_2_l3817_381794


namespace NUMINAMATH_CALUDE_not_perfect_square_l3817_381784

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n^2)) : ¬ ∃ m : ℕ, (n : ℤ)^2 + d = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3817_381784


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l3817_381760

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def total_subjects : ℕ := 5
def average_marks : ℕ := 78

theorem physics_marks_calculation :
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks := average_marks * total_subjects
  total_marks - known_marks = 82 := by
  sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l3817_381760


namespace NUMINAMATH_CALUDE_expression_evaluation_l3817_381703

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = 1/3) :
  a / (a - b) * (1 / b - 1 / a) + (a - 1) / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3817_381703


namespace NUMINAMATH_CALUDE_malcom_brandon_card_difference_l3817_381707

theorem malcom_brandon_card_difference :
  ∀ (brandon_cards malcom_cards_initial malcom_cards_after : ℕ),
    brandon_cards = 20 →
    malcom_cards_initial > brandon_cards →
    malcom_cards_after = 14 →
    malcom_cards_after * 2 = malcom_cards_initial →
    malcom_cards_initial - brandon_cards = 8 :=
by sorry

end NUMINAMATH_CALUDE_malcom_brandon_card_difference_l3817_381707


namespace NUMINAMATH_CALUDE_probability_not_snowing_l3817_381796

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2/7) : 
  1 - p_snow = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l3817_381796


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3817_381747

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 3 = 0) → (x₂^2 - 2*x₂ - 3 = 0) → (x₁ + x₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3817_381747


namespace NUMINAMATH_CALUDE_profit_calculation_l3817_381767

theorem profit_calculation :
  let selling_price : ℝ := 84
  let profit_percentage : ℝ := 0.4
  let loss_percentage : ℝ := 0.2
  let cost_price_profit_item : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss_item : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit_item + cost_price_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l3817_381767


namespace NUMINAMATH_CALUDE_fox_coins_proof_l3817_381701

def cross_bridge (initial_coins : ℕ) : ℕ := 
  3 * initial_coins - 50

def cross_bridge_n_times (initial_coins : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_coins
  | m + 1 => cross_bridge (cross_bridge_n_times initial_coins m)

theorem fox_coins_proof :
  cross_bridge_n_times 25 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fox_coins_proof_l3817_381701


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l3817_381700

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l3817_381700


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l3817_381738

/-- Given two similar squares with an area ratio of 1:9 and the smaller square's side length of 5 cm,
    prove that the larger square's side length is 15 cm. -/
theorem similar_squares_side_length (small_side : ℝ) (large_side : ℝ) : 
  small_side = 5 →  -- The side length of the smaller square is 5 cm
  (large_side / small_side)^2 = 9 →  -- The ratio of their areas is 1:9
  large_side = 15 :=  -- The side length of the larger square is 15 cm
by
  sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l3817_381738


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l3817_381773

/-- The line kx+y+k+1=0 intersects the ellipse x^2/25 + y^2/16 = 1 for all real values of k -/
theorem line_intersects_ellipse (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + k + 1 = 0) ∧ (x^2 / 25 + y^2 / 16 = 1) := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l3817_381773


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3817_381729

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3817_381729


namespace NUMINAMATH_CALUDE_negative_squared_times_squared_l3817_381732

theorem negative_squared_times_squared (a : ℝ) : -a^2 * a^2 = -a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_squared_times_squared_l3817_381732


namespace NUMINAMATH_CALUDE_weight_of_second_person_l3817_381797

/-- Proves that the weight of the second person who joined the group is 78 kg -/
theorem weight_of_second_person
  (initial_average : ℝ)
  (final_average : ℝ)
  (initial_members : ℕ)
  (weight_first_person : ℝ)
  (h_initial_average : initial_average = 48)
  (h_final_average : final_average = 51)
  (h_initial_members : initial_members = 23)
  (h_weight_first_person : weight_first_person = 93)
  : ∃ (weight_second_person : ℝ),
    weight_second_person = 78 ∧
    (initial_members : ℝ) * final_average =
      initial_members * initial_average + weight_first_person + weight_second_person :=
by sorry

end NUMINAMATH_CALUDE_weight_of_second_person_l3817_381797


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l3817_381715

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a2 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1/5 → a 3 = 5 → (a 2 = 1 ∨ a 2 = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l3817_381715


namespace NUMINAMATH_CALUDE_y_intercept_range_l3817_381757

-- Define the points A and B
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)

-- Define the line l: x + y - c = 0
def line_l (c : ℝ) (x y : ℝ) : Prop := x + y - c = 0

-- Define what it means for a point to be on the line
def point_on_line (p : ℝ × ℝ) (c : ℝ) : Prop :=
  line_l c p.1 p.2

-- Define what it means for a line to intersect a segment
def intersects_segment (c : ℝ) : Prop :=
  ∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    point_on_line ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) c

-- State the theorem
theorem y_intercept_range :
  ∀ c : ℝ, intersects_segment c → c ∈ Set.Icc (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_y_intercept_range_l3817_381757


namespace NUMINAMATH_CALUDE_china_coal_production_2003_l3817_381721

/-- Represents a large number in different formats -/
structure LargeNumber where
  value : Nat
  word_representation : String
  billion_representation : String

/-- Converts a natural number to its word representation -/
def nat_to_words (n : Nat) : String :=
  sorry

/-- Converts a natural number to its billion representation -/
def nat_to_billions (n : Nat) : String :=
  sorry

/-- Theorem stating the correct representations of China's coal production in 2003 -/
theorem china_coal_production_2003 :
  let production : LargeNumber := {
    value := 15500000000,
    word_representation := nat_to_words 15500000000,
    billion_representation := nat_to_billions 15500000000
  }
  production.word_representation = "one hundred and fifty-five billion" ∧
  production.billion_representation = "155 billion" :=
sorry

end NUMINAMATH_CALUDE_china_coal_production_2003_l3817_381721


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achieved_l3817_381791

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + 2*x + 2) + Real.sqrt (x^2 - 2*x + 2) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achieved :
  ∃ x : ℝ, Real.sqrt (x^2 + 2*x + 2) + Real.sqrt (x^2 - 2*x + 2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_min_value_achieved_l3817_381791


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3817_381792

/-- Given a total distance of 200 miles driven over 6 hours, 
    prove that the average speed is 100/3 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 200 →
  time = 6 →
  speed = distance / time →
  speed = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3817_381792


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3817_381798

theorem age_ratio_problem (a b : ℕ) : 
  a + b = 60 →                 -- Sum of present ages is 60
  ∃ k : ℕ, a = k * b →         -- A's age is some multiple of B's age
  a + b + 6 = 66 →             -- Sum of ages 3 years hence is 66
  a = 60 ∧ b = 5 :=            -- Implies A's age is 60 and B's age is 5
by sorry

-- The ratio can be derived from a = 60 and b = 5

end NUMINAMATH_CALUDE_age_ratio_problem_l3817_381798


namespace NUMINAMATH_CALUDE_difference_of_sum_and_product_l3817_381737

theorem difference_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_product_l3817_381737
