import Mathlib

namespace NUMINAMATH_CALUDE_max_profit_is_33000_l2826_282619

/-- Profit function for the first store -/
def L₁ (x : ℝ) : ℝ := -5 * x^2 + 900 * x - 16000

/-- Profit function for the second store -/
def L₂ (x : ℝ) : ℝ := 300 * x - 2000

/-- Total number of vehicles sold -/
def total_vehicles : ℝ := 110

/-- Total profit function -/
def S (x : ℝ) : ℝ := L₁ x + L₂ (total_vehicles - x)

theorem max_profit_is_33000 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_vehicles ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ total_vehicles → S y ≤ S x) ∧
  S x = 33000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_33000_l2826_282619


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l2826_282611

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l2826_282611


namespace NUMINAMATH_CALUDE_smallest_n_years_for_90_percent_depreciation_l2826_282627

-- Define the depreciation rate
def depreciation_rate : ℝ := 0.9

-- Define the target depreciation
def target_depreciation : ℝ := 0.1

-- Define the approximation of log 3
def log3_approx : ℝ := 0.477

-- Define the function to check if n years of depreciation meets the target
def meets_target (n : ℕ) : Prop := depreciation_rate ^ n ≤ target_depreciation

-- Statement to prove
theorem smallest_n_years_for_90_percent_depreciation :
  ∃ n : ℕ, meets_target n ∧ ∀ m : ℕ, m < n → ¬meets_target m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_years_for_90_percent_depreciation_l2826_282627


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_nine_l2826_282672

theorem no_solution_iff_k_equals_nine :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 7 → (x - 3) / (x - 1) ≠ (x - k) / (x - 7)) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_nine_l2826_282672


namespace NUMINAMATH_CALUDE_stating_binary_arithmetic_equality_l2826_282644

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 101₂ -/
def b101 : List Bool := [true, false, true]

/-- Represents the binary number 1001₂ -/
def b1001 : List Bool := [true, false, false, true]

/-- Represents the binary number 11₂ -/
def b11 : List Bool := [true, true]

/-- Represents the binary number 10101₂ (the expected result) -/
def b10101 : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the binary arithmetic operation 
1101₂ + 111₂ - 101₂ + 1001₂ - 11₂ equals 10101₂
-/
theorem binary_arithmetic_equality : 
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b101 + 
  binary_to_nat b1001 - binary_to_nat b11 = binary_to_nat b10101 := by
  sorry

end NUMINAMATH_CALUDE_stating_binary_arithmetic_equality_l2826_282644


namespace NUMINAMATH_CALUDE_third_meeting_at_45km_l2826_282624

/-- Two people moving with constant speeds on a 100 km path between points A and B -/
structure TwoMovers :=
  (speed_ratio : ℚ)
  (first_meet : ℚ)
  (second_meet : ℚ)

/-- The third meeting point of two movers given their speed ratio and first two meeting points -/
def third_meeting_point (m : TwoMovers) : ℚ :=
  100 - (3 / 8) * 200

/-- Theorem stating that under given conditions, the third meeting point is 45 km from A -/
theorem third_meeting_at_45km (m : TwoMovers) 
  (h1 : m.first_meet = 20)
  (h2 : m.second_meet = 80)
  (h3 : m.speed_ratio = 3 / 5) :
  third_meeting_point m = 45 := by
  sorry

#eval third_meeting_point { speed_ratio := 3 / 5, first_meet := 20, second_meet := 80 }

end NUMINAMATH_CALUDE_third_meeting_at_45km_l2826_282624


namespace NUMINAMATH_CALUDE_worker_arrival_time_l2826_282606

/-- Proves that a worker walking at 4/5 of her normal speed arrives 10 minutes later -/
theorem worker_arrival_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_time = 40)
  (h2 : normal_speed > 0) :
  let reduced_speed := (4/5 : ℝ) * normal_speed
  let new_time := normal_time * (normal_speed / reduced_speed)
  new_time - normal_time = 10 := by
sorry


end NUMINAMATH_CALUDE_worker_arrival_time_l2826_282606


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l2826_282632

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 200)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l2826_282632


namespace NUMINAMATH_CALUDE_monthly_revenue_is_4000_l2826_282605

/-- A store's financial data -/
structure StoreFinancials where
  initial_investment : ℕ
  monthly_expenses : ℕ
  payback_period : ℕ

/-- Calculate the monthly revenue required to break even -/
def calculate_monthly_revenue (store : StoreFinancials) : ℕ :=
  (store.initial_investment + store.monthly_expenses * store.payback_period) / store.payback_period

/-- Theorem: Given the store's financial data, the monthly revenue is $4000 -/
theorem monthly_revenue_is_4000 (store : StoreFinancials) 
    (h1 : store.initial_investment = 25000)
    (h2 : store.monthly_expenses = 1500)
    (h3 : store.payback_period = 10) :
  calculate_monthly_revenue store = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_revenue_is_4000_l2826_282605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2826_282687

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term :
  (∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2826_282687


namespace NUMINAMATH_CALUDE_safe_locks_and_keys_l2826_282660

/-- Represents the number of committee members -/
def n : ℕ := 11

/-- Represents the size of the smallest group that can open the safe -/
def k : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose n (k - 1)

/-- Calculates the total number of keys needed -/
def num_keys : ℕ := num_locks * k

/-- Theorem stating the minimum number of locks and keys needed -/
theorem safe_locks_and_keys : num_locks = 462 ∧ num_keys = 2772 := by
  sorry

#eval num_locks -- Should output 462
#eval num_keys  -- Should output 2772

end NUMINAMATH_CALUDE_safe_locks_and_keys_l2826_282660


namespace NUMINAMATH_CALUDE_unique_solution_geometric_series_l2826_282689

theorem unique_solution_geometric_series :
  ∃! x : ℝ, |x| < 1 ∧ x = (1 : ℝ) / (1 + x) ∧ x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_geometric_series_l2826_282689


namespace NUMINAMATH_CALUDE_banana_price_theorem_l2826_282697

/-- The cost of a banana in pence -/
def banana_cost : ℚ := 1.25

/-- The number of pence in a shilling -/
def pence_per_shilling : ℕ := 12

/-- The number of shillings in a pound -/
def shillings_per_pound : ℕ := 20

/-- The number of bananas in a dozen dozen -/
def dozen_dozen : ℕ := 12 * 12

theorem banana_price_theorem :
  let pence_per_pound : ℕ := pence_per_shilling * shillings_per_pound
  let bananas_per_fiver : ℚ := (5 * pence_per_pound : ℚ) / banana_cost
  let sixpences_for_16_dozen_dozen : ℚ := (16 * dozen_dozen * banana_cost) / 6
  sixpences_for_16_dozen_dozen = bananas_per_fiver / 2 :=
by sorry


end NUMINAMATH_CALUDE_banana_price_theorem_l2826_282697


namespace NUMINAMATH_CALUDE_rectangle_distance_l2826_282698

theorem rectangle_distance (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 6 →
  large_area = 12 →
  let small_width := small_perimeter / 6
  let small_length := 2 * small_width
  let large_width := 3 * small_width
  let large_length := 2 * small_length
  large_width * large_length = large_area →
  let horizontal_distance := large_length
  let vertical_distance := large_width - small_width
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_distance_l2826_282698


namespace NUMINAMATH_CALUDE_truncated_prism_cross_section_area_l2826_282674

/-- Theorem: Square root of cross-section area in a truncated prism -/
theorem truncated_prism_cross_section_area 
  (S' S Q : ℝ) (n m : ℕ) 
  (h1 : 0 < S') (h2 : 0 < S) (h3 : 0 < Q) (h4 : 0 < n) (h5 : 0 < m) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) := by
  sorry

end NUMINAMATH_CALUDE_truncated_prism_cross_section_area_l2826_282674


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2826_282623

/-- The equation (x + 3) / (kx - 2) = x + 1 has exactly one solution if and only if k = -7 + 2√10 or k = -7 - 2√10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x + 1 ∧ k * x - 2 ≠ 0) ↔ 
  (k = -7 + 2 * Real.sqrt 10 ∨ k = -7 - 2 * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2826_282623


namespace NUMINAMATH_CALUDE_isosceles_triangles_bound_l2826_282659

/-- The largest number of isosceles triangles whose vertices belong to some set of n points in the plane without three colinear points -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of positive real constants a and b bounding f(n) -/
theorem isosceles_triangles_bound :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ n : ℕ, n ≥ 3 → (a * n^2 : ℝ) < f n ∧ (f n : ℝ) < b * n^2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_bound_l2826_282659


namespace NUMINAMATH_CALUDE_min_abs_z_l2826_282638

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) : 
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 56 / 17 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l2826_282638


namespace NUMINAMATH_CALUDE_four_number_sequence_l2826_282676

theorem four_number_sequence : ∃ (a₁ a₂ a₃ a₄ : ℝ),
  (a₂^2 = a₁ * a₃) ∧
  (2 * a₃ = a₂ + a₄) ∧
  (a₁ + a₄ = 21) ∧
  (a₂ + a₃ = 18) ∧
  ((a₁ = 3 ∧ a₂ = 6 ∧ a₃ = 12 ∧ a₄ = 18) ∨
   (a₁ = 18.75 ∧ a₂ = 11.25 ∧ a₃ = 6.75 ∧ a₄ = 2.25)) :=
by
  sorry


end NUMINAMATH_CALUDE_four_number_sequence_l2826_282676


namespace NUMINAMATH_CALUDE_triangle_problem_l2826_282673

open Real

theorem triangle_problem (A B C a b c : Real) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (cos C) / (cos B) = (2 * a - c) / b →
  tan (A + π/4) = 7 →
  B = π/3 ∧ cos C = (3 * sqrt 3 - 4) / 10 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2826_282673


namespace NUMINAMATH_CALUDE_second_meeting_time_correct_l2826_282655

/-- Represents a vehicle on a race track -/
structure Vehicle where
  name : String
  lap_time : ℕ  -- lap time in seconds

/-- Calculates the time until two vehicles meet at the starting point for the second time -/
def timeToSecondMeeting (v1 v2 : Vehicle) : ℚ :=
  (Nat.lcm v1.lap_time v2.lap_time : ℚ) / 60

/-- The main theorem to prove -/
theorem second_meeting_time_correct 
  (magic : Vehicle) 
  (bull : Vehicle) 
  (h1 : magic.lap_time = 150)
  (h2 : bull.lap_time = 3600 / 40) :
  timeToSecondMeeting magic bull = 7.5 := by
  sorry

#eval timeToSecondMeeting 
  { name := "The Racing Magic", lap_time := 150 } 
  { name := "The Charging Bull", lap_time := 3600 / 40 }

end NUMINAMATH_CALUDE_second_meeting_time_correct_l2826_282655


namespace NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l2826_282603

/-- Represents the composition of trail mix -/
structure TrailMix where
  peanuts : ℝ
  chocolate_chips : ℝ
  raisins : ℝ

/-- The total weight of the trail mix -/
def total_weight (mix : TrailMix) : ℝ :=
  mix.peanuts + mix.chocolate_chips + mix.raisins

theorem brandy_trail_mix_peanuts :
  ∀ (mix : TrailMix),
    mix.chocolate_chips = 0.17 →
    mix.raisins = 0.08 →
    total_weight mix = 0.42 →
    mix.peanuts = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l2826_282603


namespace NUMINAMATH_CALUDE_smallest_divisible_number_after_2013_l2826_282666

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, i > 0 ∧ i < 10 → n % i = 0

theorem smallest_divisible_number_after_2013 :
  ∃ (n : ℕ),
    n ≥ 2013000 ∧
    is_divisible_by_all_less_than_10 n ∧
    (∀ m : ℕ, 2013000 ≤ m ∧ m < n → ¬is_divisible_by_all_less_than_10 m) ∧
    n = 2013480 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_after_2013_l2826_282666


namespace NUMINAMATH_CALUDE_shelter_dogs_l2826_282609

theorem shelter_dogs (total_animals cats : ℕ) 
  (h1 : total_animals = 1212)
  (h2 : cats = 645) : 
  total_animals - cats = 567 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_l2826_282609


namespace NUMINAMATH_CALUDE_inequality_proof_l2826_282631

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2826_282631


namespace NUMINAMATH_CALUDE_car_efficiency_improvement_l2826_282669

/-- Represents the additional miles a car can travel after improving fuel efficiency -/
def additional_miles (initial_efficiency : ℝ) (tank_capacity : ℝ) (efficiency_improvement : ℝ) : ℝ :=
  tank_capacity * (initial_efficiency * (1 + efficiency_improvement) - initial_efficiency)

/-- Theorem stating the additional miles a car can travel after modification -/
theorem car_efficiency_improvement :
  additional_miles 33 16 0.25 = 132 := by
  sorry

end NUMINAMATH_CALUDE_car_efficiency_improvement_l2826_282669


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2826_282679

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 13 (m + 1) = Nat.choose 13 (2 * m - 3)) ↔ (m = 4 ∨ m = 5) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2826_282679


namespace NUMINAMATH_CALUDE_bulb_over_4000_hours_probability_l2826_282639

-- Define the probabilities
def prob_x : ℝ := 0.60  -- Probability of a bulb coming from factory X
def prob_y : ℝ := 1 - prob_x  -- Probability of a bulb coming from factory Y
def prob_x_over_4000 : ℝ := 0.59  -- Probability of factory X's bulb lasting over 4000 hours
def prob_y_over_4000 : ℝ := 0.65  -- Probability of factory Y's bulb lasting over 4000 hours

-- Define the theorem
theorem bulb_over_4000_hours_probability :
  prob_x * prob_x_over_4000 + prob_y * prob_y_over_4000 = 0.614 :=
by sorry

end NUMINAMATH_CALUDE_bulb_over_4000_hours_probability_l2826_282639


namespace NUMINAMATH_CALUDE_inequality_solution_l2826_282678

theorem inequality_solution (x : ℝ) :
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ (5 ≤ x ∧ x ≤ 7) ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2826_282678


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2826_282617

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 30th term of the arithmetic sequence with first term 3 and common difference 6 is 177 -/
theorem thirtieth_term_of_sequence : arithmetic_sequence 3 6 30 = 177 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2826_282617


namespace NUMINAMATH_CALUDE_radical_equality_l2826_282630

theorem radical_equality (a b c : ℤ) :
  Real.sqrt (a + b / c) = a * Real.sqrt (b / c) ↔ c = b * (a^2 - 1) / a :=
by sorry

end NUMINAMATH_CALUDE_radical_equality_l2826_282630


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2826_282661

theorem shirt_price_proof (total_cost : ℝ) (shirt_price : ℝ) (coat_price : ℝ) 
  (h1 : total_cost = 600)
  (h2 : shirt_price + coat_price = total_cost)
  (h3 : shirt_price = (1/3) * coat_price) :
  shirt_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2826_282661


namespace NUMINAMATH_CALUDE_add_fractions_simplest_form_l2826_282681

theorem add_fractions_simplest_form :
  (7 : ℚ) / 8 + (3 : ℚ) / 5 = (59 : ℚ) / 40 ∧ 
  ∀ n d : ℤ, (d ≠ 0 ∧ (59 : ℚ) / 40 = (n : ℚ) / d) → n.gcd d = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_simplest_form_l2826_282681


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2826_282618

theorem rectangle_area_error_percentage (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let actual_area := L * W
  let measured_area := 1.10 * L * 0.95 * W
  let error_percentage := (measured_area - actual_area) / actual_area * 100
  error_percentage = 4.5 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2826_282618


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2826_282629

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2826_282629


namespace NUMINAMATH_CALUDE_owls_on_fence_l2826_282620

theorem owls_on_fence (initial_owls final_owls joined_owls : ℕ) : 
  final_owls = initial_owls + joined_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l2826_282620


namespace NUMINAMATH_CALUDE_intersection_circle_equation_l2826_282656

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2 + t * Real.cos a ∧ p.2 = 1 + t * Real.sin a}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2/2 = 1}

-- Define the intersection points M and N
def intersection_points : Set (ℝ × ℝ) :=
  C₁ (Real.pi/4) ∩ C₂

-- State the theorem
theorem intersection_circle_equation :
  ∀ M N : ℝ × ℝ,
  M ∈ intersection_points → N ∈ intersection_points → M ≠ N →
  ∀ P : ℝ × ℝ,
  P ∈ {P : ℝ × ℝ | (P.1 - 1/3)^2 + (P.2 + 2/3)^2 = 8/9} ↔
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t • M + (1 - t) • N) :=
by sorry


end NUMINAMATH_CALUDE_intersection_circle_equation_l2826_282656


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_23_l2826_282642

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_23 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 23 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_23_l2826_282642


namespace NUMINAMATH_CALUDE_chairs_difference_l2826_282647

theorem chairs_difference (initial : ℕ) (remaining : ℕ) : 
  initial = 15 → remaining = 3 → initial - remaining = 12 := by
  sorry

end NUMINAMATH_CALUDE_chairs_difference_l2826_282647


namespace NUMINAMATH_CALUDE_fraction_identity_condition_l2826_282685

theorem fraction_identity_condition (a b c d : ℝ) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) →
  a / b = c / d :=
by sorry

end NUMINAMATH_CALUDE_fraction_identity_condition_l2826_282685


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2826_282612

theorem quadratic_two_zeros (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + b*x₁ - 3 = 0) ∧ 
    (x₂^2 + b*x₂ - 3 = 0) ∧ 
    (∀ x : ℝ, x^2 + b*x - 3 = 0 → (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2826_282612


namespace NUMINAMATH_CALUDE_cube_root_of_square_l2826_282648

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l2826_282648


namespace NUMINAMATH_CALUDE_k_range_for_unique_integer_solution_l2826_282696

/-- Given a real number k, this function represents the system of inequalities -/
def inequality_system (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (5+2*k)*x + 5*k < 0

/-- This theorem states that if -2 is the only integer solution to the inequality system,
    then k is in the range [-3, 2) -/
theorem k_range_for_unique_integer_solution :
  (∀ x : ℤ, inequality_system (x : ℝ) k ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_for_unique_integer_solution_l2826_282696


namespace NUMINAMATH_CALUDE_problem_solution_l2826_282614

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1|

theorem problem_solution (m : ℝ) (a b c : ℝ) 
  (h1 : m > 0)
  (h2 : Set.Icc (-3 : ℝ) 3 = {x : ℝ | f m (x + 1) ≥ 0})
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : 1/a + 1/(2*b) + 1/(3*c) = m) :
  m = 3 ∧ a + 2*b + 3*c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2826_282614


namespace NUMINAMATH_CALUDE_grapes_filling_days_l2826_282613

/-- The number of days required to fill a certain number of drums of grapes -/
def days_to_fill_grapes (pickers : ℕ) (drums_per_day : ℕ) (total_drums : ℕ) : ℕ :=
  total_drums / drums_per_day

/-- Theorem stating that it takes 77 days to fill 17017 drums of grapes -/
theorem grapes_filling_days :
  days_to_fill_grapes 235 221 17017 = 77 := by
  sorry

end NUMINAMATH_CALUDE_grapes_filling_days_l2826_282613


namespace NUMINAMATH_CALUDE_alternating_arrangement_count_l2826_282650

/-- The number of ways to arrange n elements from a set of m elements. -/
def A (m n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 4 boys and 4 girls in a row,
    such that no two girls are adjacent and no two boys are adjacent. -/
def alternating_arrangement : ℕ := sorry

/-- Theorem stating that the number of alternating arrangements
    of 4 boys and 4 girls is equal to 2A₄⁴A₄⁴. -/
theorem alternating_arrangement_count :
  alternating_arrangement = 2 * A 4 4 * A 4 4 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangement_count_l2826_282650


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l2826_282690

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Theorem 1
theorem P_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem 2
theorem P_parallel_to_y_axis (a : ℝ) :
  P a = (4, 8) ↔ (P a).1 = 4 :=
sorry

-- Theorem 3
theorem P_second_quadrant_equidistant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2022 = 2021 :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_P_second_quadrant_equidistant_l2826_282690


namespace NUMINAMATH_CALUDE_triangle_angle_with_sine_half_l2826_282653

theorem triangle_angle_with_sine_half (α : Real) :
  0 < α ∧ α < π ∧ Real.sin α = 1/2 → α = π/6 ∨ α = 5*π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_with_sine_half_l2826_282653


namespace NUMINAMATH_CALUDE_stating_number_of_regions_correct_l2826_282626

/-- 
Given n lines in a plane where no two lines are parallel and no three lines are concurrent,
this function returns the number of regions the plane is divided into.
-/
def number_of_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- 
Theorem stating that n lines in a plane, with no two lines parallel and no three lines concurrent,
divide the plane into (n(n+1)/2) + 1 regions.
-/
theorem number_of_regions_correct (n : ℕ) : 
  number_of_regions n = n * (n + 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_stating_number_of_regions_correct_l2826_282626


namespace NUMINAMATH_CALUDE_count_five_digit_integers_l2826_282636

/-- The set of digits to be used -/
def digits : Multiset ℕ := {3, 3, 6, 6, 6, 7, 8, 8}

/-- The number of digits required for each integer -/
def required_digits : ℕ := 5

/-- The function to count valid integers -/
def count_valid_integers (d : Multiset ℕ) (r : ℕ) : ℕ :=
  (d.card.factorial) / ((d.count 3).factorial * (d.count 6).factorial * (d.count 8).factorial)

/-- The main theorem -/
theorem count_five_digit_integers : 
  count_valid_integers digits required_digits = 1680 :=
sorry

end NUMINAMATH_CALUDE_count_five_digit_integers_l2826_282636


namespace NUMINAMATH_CALUDE_integer_triple_product_sum_l2826_282684

theorem integer_triple_product_sum (a b c : ℤ) : 
  (a * b * c = 4 * (a + b + c) ∧ c = 2 * (a + b)) ↔ 
  ((a = 1 ∧ b = 6 ∧ c = 14) ∨ 
   (a = -1 ∧ b = -6 ∧ c = -14) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 10) ∨ 
   (a = -2 ∧ b = -3 ∧ c = -10) ∨ 
   (b = -a ∧ c = 0)) := by
sorry

end NUMINAMATH_CALUDE_integer_triple_product_sum_l2826_282684


namespace NUMINAMATH_CALUDE_abc_product_value_l2826_282645

theorem abc_product_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 3) :
  a * b * c = 10 + 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_value_l2826_282645


namespace NUMINAMATH_CALUDE_wax_requirement_l2826_282671

theorem wax_requirement (current_wax : ℕ) (additional_wax : ℕ) : 
  current_wax = 11 → additional_wax = 481 → current_wax + additional_wax = 492 := by
  sorry

end NUMINAMATH_CALUDE_wax_requirement_l2826_282671


namespace NUMINAMATH_CALUDE_integer_expression_l2826_282621

theorem integer_expression (n : ℕ) : ∃ (k : ℤ), 
  (3^(2*n) : ℚ) / 112 - (4^(2*n) : ℚ) / 63 + (5^(2*n) : ℚ) / 144 = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l2826_282621


namespace NUMINAMATH_CALUDE_bo_flashcard_knowledge_percentage_l2826_282640

theorem bo_flashcard_knowledge_percentage :
  let total_flashcards : ℕ := 800
  let days_to_learn : ℕ := 40
  let words_per_day : ℕ := 16
  let total_words_to_learn : ℕ := days_to_learn * words_per_day
  let words_already_known : ℕ := total_flashcards - total_words_to_learn
  let percentage_known : ℚ := (words_already_known : ℚ) / (total_flashcards : ℚ) * 100
  percentage_known = 20 := by
sorry

end NUMINAMATH_CALUDE_bo_flashcard_knowledge_percentage_l2826_282640


namespace NUMINAMATH_CALUDE_carlo_friday_practice_time_l2826_282600

/-- Represents Carlo's practice times for each day of the week -/
structure PracticeTimes where
  M : ℕ  -- Monday
  T : ℕ  -- Tuesday
  W : ℕ  -- Wednesday
  Th : ℕ -- Thursday
  F : ℕ  -- Friday

/-- Conditions for Carlo's practice schedule -/
def valid_practice_schedule (pt : PracticeTimes) : Prop :=
  pt.M = 2 * pt.T ∧
  pt.T = pt.W - 10 ∧
  pt.W = pt.Th + 5 ∧
  pt.Th = 50 ∧
  pt.M + pt.T + pt.W + pt.Th + pt.F = 300

/-- Theorem stating that given the conditions, Carlo should practice 60 minutes on Friday -/
theorem carlo_friday_practice_time (pt : PracticeTimes) 
  (h : valid_practice_schedule pt) : pt.F = 60 := by
  sorry

end NUMINAMATH_CALUDE_carlo_friday_practice_time_l2826_282600


namespace NUMINAMATH_CALUDE_angle_between_vectors_is_pi_over_3_l2826_282664

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors_is_pi_over_3 (a b : ℝ × ℝ) 
  (h1 : a • (a + b) = 5)
  (h2 : ‖a‖ = 2)
  (h3 : ‖b‖ = 1) : 
  angle_between_vectors a b = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_is_pi_over_3_l2826_282664


namespace NUMINAMATH_CALUDE_expression_evaluation_l2826_282646

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -8
  (Real.sqrt (9 * x * y) - 2 * Real.sqrt (x^3 * y) + Real.sqrt (x * y^3)) = 20 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2826_282646


namespace NUMINAMATH_CALUDE_average_and_subtraction_l2826_282637

theorem average_and_subtraction (y : ℝ) : 
  (15 + 25 + y) / 3 = 22 → y - 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_and_subtraction_l2826_282637


namespace NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2826_282694

/-- A pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (is_square_base : base_side > 0)
  (lateral_faces_isosceles_right : True)

/-- A cube placed inside a pyramid -/
structure CubeInPyramid :=
  (pyramid : Pyramid)
  (bottom_on_base : True)
  (top_touches_midpoints : True)

/-- The volume of a cube -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Theorem: The volume of the cube in the given pyramid configuration is 1 -/
theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : CubeInPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : c.pyramid = p) : 
  ∃ (edge_length : ℝ), cube_volume edge_length = 1 :=
sorry

end NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2826_282694


namespace NUMINAMATH_CALUDE_M_factorization_l2826_282608

/-- The polynomial M(x, y, z) = x^3 + y^3 + z^3 - 3xyz -/
def M (x y z : ℝ) : ℝ := x^3 + y^3 + z^3 - 3*x*y*z

/-- The factorization of M(x, y, z) -/
theorem M_factorization (x y z : ℝ) :
  M x y z = (x + y + z) * (x^2 + y^2 + z^2 - x*y - y*z - z*x) := by
  sorry

end NUMINAMATH_CALUDE_M_factorization_l2826_282608


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2826_282668

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2826_282668


namespace NUMINAMATH_CALUDE_common_chord_intersection_l2826_282657

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point where two circles intersect -/
def IntersectionPoint (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

/-- The common chord of two intersecting circles -/
def CommonChord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  IntersectionPoint c1 c2

/-- Theorem: For any three circles in a plane that intersect pairwise, 
    the common chords of these pairs of circles intersect at a single point -/
theorem common_chord_intersection (c1 c2 c3 : Circle) 
  (h12 : (CommonChord c1 c2).Nonempty)
  (h23 : (CommonChord c2 c3).Nonempty)
  (h31 : (CommonChord c3 c1).Nonempty) :
  ∃ p, p ∈ CommonChord c1 c2 ∧ p ∈ CommonChord c2 c3 ∧ p ∈ CommonChord c3 c1 :=
sorry

end NUMINAMATH_CALUDE_common_chord_intersection_l2826_282657


namespace NUMINAMATH_CALUDE_evaluate_expression_l2826_282604

theorem evaluate_expression (x y : ℚ) (hx : x = 3) (hy : y = -3) :
  (4 + y * x * (y + x) - 4^2) / (y - 4 + y^2) = -6 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2826_282604


namespace NUMINAMATH_CALUDE_toms_fruit_bowl_l2826_282680

/-- The number of fruits remaining in Tom's fruit bowl after eating some fruits -/
def remaining_fruits (initial_oranges initial_lemons eaten : ℕ) : ℕ :=
  initial_oranges + initial_lemons - eaten

/-- Theorem: Given Tom's fruit bowl with 3 oranges and 6 lemons, after eating 3 fruits, 6 fruits remain -/
theorem toms_fruit_bowl : remaining_fruits 3 6 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_bowl_l2826_282680


namespace NUMINAMATH_CALUDE_bowling_team_new_average_l2826_282665

def bowling_team_average (original_players : ℕ) (original_average : ℚ) (new_player1_weight : ℚ) (new_player2_weight : ℚ) : ℚ :=
  let original_total_weight := original_players * original_average
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

theorem bowling_team_new_average :
  bowling_team_average 7 121 110 60 = 113 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_new_average_l2826_282665


namespace NUMINAMATH_CALUDE_cat_relocation_l2826_282641

theorem cat_relocation (initial_cats : ℕ) (first_removal : ℕ) : 
  initial_cats = 1800 →
  first_removal = 600 →
  initial_cats - first_removal - (initial_cats - first_removal) / 2 = 600 := by
sorry

end NUMINAMATH_CALUDE_cat_relocation_l2826_282641


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2826_282688

theorem quadratic_transformation (a b c : ℝ) :
  (∃ m q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = 5 * (x - 3)^2 + 7) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q ∧ p = 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2826_282688


namespace NUMINAMATH_CALUDE_stating_football_league_equation_l2826_282654

/-- 
The number of matches in a football league where each pair of classes plays a match,
given the number of class teams.
-/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- 
Theorem stating that for a football league with x class teams, where each pair plays a match,
and there are 15 matches in total, the equation relating x to the number of matches is correct.
-/
theorem football_league_equation (x : ℕ) : 
  (number_of_matches x = 15) ↔ (x * (x - 1) / 2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_stating_football_league_equation_l2826_282654


namespace NUMINAMATH_CALUDE_two_solutions_l2826_282670

-- Define the matrix evaluation rule
def matrixEval (a b c d : ℝ) : ℝ := a * b - c * d + c

-- Define the equation
def equation (x : ℝ) : Prop := matrixEval (3 * x) x 2 (2 * x) = 2

-- Theorem statement
theorem two_solutions :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂) ∧
  (∀ x : ℝ, equation x → x = 0 ∨ x = 4/3) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l2826_282670


namespace NUMINAMATH_CALUDE_high_school_relationships_l2826_282634

/-- The number of people in the group -/
def n : ℕ := 12

/-- The number of categories for each pair -/
def categories : ℕ := 3

/-- The number of pairs in a group of n people -/
def pairs (n : ℕ) : ℕ := n.choose 2

/-- The total number of pair categorizations -/
def totalCategorizations (n : ℕ) (categories : ℕ) : ℕ :=
  pairs n * categories

theorem high_school_relationships :
  totalCategorizations n categories = 198 := by sorry

end NUMINAMATH_CALUDE_high_school_relationships_l2826_282634


namespace NUMINAMATH_CALUDE_total_insect_legs_l2826_282615

/-- Given the number of insects and legs per insect, calculate the total number of insect legs -/
theorem total_insect_legs (num_insects : ℕ) (legs_per_insect : ℕ) :
  num_insects * legs_per_insect = num_insects * legs_per_insect := by
  sorry

/-- Ezekiel's report on insects in the laboratory -/
def ezekiels_report : ℕ := 9

/-- Number of legs each insect has -/
def legs_per_insect : ℕ := 6

/-- Calculate the total number of insect legs in the laboratory -/
def total_legs : ℕ := ezekiels_report * legs_per_insect

#eval total_legs  -- This will output 54

end NUMINAMATH_CALUDE_total_insect_legs_l2826_282615


namespace NUMINAMATH_CALUDE_function_value_ordering_l2826_282695

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom even : ∀ x, f (-x) = f x
axiom periodic : ∀ x, f (x + 1) = f (x - 1)
axiom monotonic : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

-- State the theorem
theorem function_value_ordering : f (-3/2) < f (4/3) ∧ f (4/3) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_ordering_l2826_282695


namespace NUMINAMATH_CALUDE_train_crossing_time_l2826_282601

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 480 ∧ train_speed_kmh = 216 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2826_282601


namespace NUMINAMATH_CALUDE_max_ratio_system_l2826_282675

theorem max_ratio_system (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∀ ε > 0, ∃ x' y' z' u' : ℕ+,
    x' + y' = z' + u' ∧
    2 * x' * y' = z' * u' ∧
    x' ≥ y' ∧
    (x' : ℝ) / y' > 3 + 2 * Real.sqrt 2 - ε :=
sorry

end NUMINAMATH_CALUDE_max_ratio_system_l2826_282675


namespace NUMINAMATH_CALUDE_bacteria_growth_l2826_282658

def b (t : ℝ) : ℝ := 105 + 104 * t - 1000 * t^2

theorem bacteria_growth (t : ℝ) :
  (deriv b 5 = 0) ∧
  (deriv b 10 = -10000) ∧
  (∀ t ∈ Set.Ioo 0 5, deriv b t > 0) ∧
  (∀ t ∈ Set.Ioi 5, deriv b t < 0) := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2826_282658


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2826_282628

theorem rationalize_denominator :
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2826_282628


namespace NUMINAMATH_CALUDE_smallest_c_value_l2826_282633

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  c ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2826_282633


namespace NUMINAMATH_CALUDE_car_journey_average_speed_l2826_282691

-- Define the parameters of the journey
def distance_part1 : ℝ := 18  -- km
def time_part1 : ℝ := 24      -- minutes
def time_part2 : ℝ := 35      -- minutes
def speed_part2 : ℝ := 72     -- km/h

-- Define the theorem
theorem car_journey_average_speed :
  let total_distance := distance_part1 + speed_part2 * (time_part2 / 60)
  let total_time := time_part1 + time_part2
  let average_speed := total_distance / (total_time / 60)
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.005 ∧ |average_speed - 61.02| < ε :=
by sorry

end NUMINAMATH_CALUDE_car_journey_average_speed_l2826_282691


namespace NUMINAMATH_CALUDE_common_divisors_84_90_l2826_282686

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (max 84 90 + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_84_90_l2826_282686


namespace NUMINAMATH_CALUDE_complex_expressions_equality_l2826_282635

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equality to be proved
theorem complex_expressions_equality :
  ((-1/2 : ℂ) + (Real.sqrt 3/2)*i) * (2 - i) * (3 + i) = 
    (-3/2 : ℂ) + (5*Real.sqrt 3/2) + ((7*Real.sqrt 3 + 1)/2)*i ∧
  ((Real.sqrt 2 + Real.sqrt 2*i)^2 * (4 + 5*i)) / ((5 - 4*i) * (1 - i)) = 
    (62/41 : ℂ) + (80/41)*i :=
by sorry

-- Axiom for i^2 = -1
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_complex_expressions_equality_l2826_282635


namespace NUMINAMATH_CALUDE_construction_team_equation_l2826_282663

/-- Represents the equation for a construction team's road-laying project -/
theorem construction_team_equation (x : ℝ) (h : x > 0) :
  let total_length : ℝ := 480
  let efficiency_increase : ℝ := 0.5
  let days_ahead : ℝ := 4
  (total_length / x) - (total_length / ((1 + efficiency_increase) * x)) = days_ahead :=
by sorry

end NUMINAMATH_CALUDE_construction_team_equation_l2826_282663


namespace NUMINAMATH_CALUDE_five_solutions_for_f_f_x_eq_8_l2826_282643

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 1 else x + 4

theorem five_solutions_for_f_f_x_eq_8 :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 8 :=
sorry

end NUMINAMATH_CALUDE_five_solutions_for_f_f_x_eq_8_l2826_282643


namespace NUMINAMATH_CALUDE_ellipse_a_range_l2826_282602

/-- An ellipse with equation x²/(a-5) + y²/2 = 1 and foci on the x-axis -/
structure Ellipse (a : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (a - 5) + y^2 / 2 = 1
  foci_on_x : True  -- This is a placeholder for the condition that foci are on the x-axis

/-- The range of a for an ellipse with the given equation and foci on the x-axis -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l2826_282602


namespace NUMINAMATH_CALUDE_puppies_per_cage_l2826_282692

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 18) 
  (h2 : sold_puppies = 3) 
  (h3 : num_cages = 3) 
  : (initial_puppies - sold_puppies) / num_cages = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l2826_282692


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l2826_282682

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (5 - x) = x * Real.sqrt (5 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ f a ∧ f b ∧ ∀ (x : ℝ), f x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l2826_282682


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2826_282651

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - 2*x)^(1/3 : ℝ) = -2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2826_282651


namespace NUMINAMATH_CALUDE_compare_fractions_l2826_282667

theorem compare_fractions : 2/3 < Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l2826_282667


namespace NUMINAMATH_CALUDE_abs_neg_2023_l2826_282677

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l2826_282677


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2826_282652

theorem sum_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℝ := -18
  let b : ℝ := 54
  let c : ℝ := -72
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2826_282652


namespace NUMINAMATH_CALUDE_least_value_theorem_l2826_282607

theorem least_value_theorem (x y z : ℕ+) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) :
  (∃ (k : ℕ), x + k + z = 26 ∧ ∀ (m : ℕ), x + m + z = 26 → m ≥ k) →
  (∃ (k : ℕ), k = 6) := by
  sorry

end NUMINAMATH_CALUDE_least_value_theorem_l2826_282607


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l2826_282616

theorem probability_at_least_one_head (p : ℝ) : 
  p = 1 - (1/2)^4 → p = 15/16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l2826_282616


namespace NUMINAMATH_CALUDE_folded_paper_length_l2826_282683

/-- Given a rectangle with sides of lengths 1 and √2, where one vertex is folded to touch the opposite side, the length d of the folded edge is √2 - 1. -/
theorem folded_paper_length (a b d : ℝ) : 
  a = 1 → b = Real.sqrt 2 → 
  d = Real.sqrt ((b - d)^2 + a^2) → 
  d = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_length_l2826_282683


namespace NUMINAMATH_CALUDE_quadratic_sum_l2826_282610

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℚ) : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℚ) :
  (∃ f : ℚ → ℚ, f = quadratic a b c ∧
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (f 3 = -2) ∧
    (∀ x, f (6 - x) = f x) ∧
    (f 0 = 5)) →
  a + b + c = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2826_282610


namespace NUMINAMATH_CALUDE_two_card_probability_l2826_282699

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (size : cards.card = 52)

/-- The probability of selecting two cards from a standard 52-card deck
    that are neither of the same value nor the same suit is 12/17. -/
theorem two_card_probability (d : Deck) : 
  let first_card := d.cards.card
  let second_card := d.cards.card - 1
  let favorable_outcomes := 3 * 12
  (favorable_outcomes : ℚ) / second_card = 12 / 17 := by sorry

end NUMINAMATH_CALUDE_two_card_probability_l2826_282699


namespace NUMINAMATH_CALUDE_range_of_a_l2826_282662

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∧ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → False ∧
  (∀ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0 → False) ∨ 
  (∀ x : ℝ, x^2 + a*x + 1 > 0) → 
  a ≤ -2 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2826_282662


namespace NUMINAMATH_CALUDE_intersection_M_N_l2826_282622

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x = 0}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2826_282622


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2826_282649

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0) → 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0 → x = -1 ∧ y = 2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2826_282649


namespace NUMINAMATH_CALUDE_find_incorrect_value_l2826_282693

/-- Represents the problem of finding the incorrect value in a mean calculation --/
theorem find_incorrect_value (n : ℕ) (initial_mean correct_mean correct_value : ℚ) :
  n = 30 ∧ 
  initial_mean = 180 ∧ 
  correct_mean = 180 + 2/3 ∧ 
  correct_value = 155 →
  ∃ incorrect_value : ℚ,
    incorrect_value = 175 ∧
    n * initial_mean = (n - 1) * correct_mean + incorrect_value ∧
    n * correct_mean = (n - 1) * correct_mean + correct_value :=
by sorry

end NUMINAMATH_CALUDE_find_incorrect_value_l2826_282693


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2826_282625

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 14*y + 73 = -y^2 + 6*x

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 10 + Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2826_282625
