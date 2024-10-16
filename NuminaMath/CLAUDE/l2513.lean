import Mathlib

namespace NUMINAMATH_CALUDE_bryans_pushups_l2513_251349

/-- 
Given Bryan's push-up routine:
- He does 3 sets of push-ups
- In the third set, he does 5 fewer push-ups than in the other sets
- He did 40 push-ups in total

This theorem proves that the number of push-ups in each of the first two sets is 15.
-/
theorem bryans_pushups (x : ℕ) : 
  x + x + (x - 5) = 40 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bryans_pushups_l2513_251349


namespace NUMINAMATH_CALUDE_distribute_equals_choose_l2513_251393

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribute_equals_choose :
  distribute 10 7 = choose 9 6 := by sorry

end NUMINAMATH_CALUDE_distribute_equals_choose_l2513_251393


namespace NUMINAMATH_CALUDE_pentagon_sum_l2513_251373

/-- Given integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u, v)
    B is the reflection of A across y = -x
    C is the reflection of B across the y-axis
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 500, then u + v = 21 -/
theorem pentagon_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v)
  (harea : 6 * u * v - 2 * v^2 = 500) : u + v = 21 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l2513_251373


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l2513_251361

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 800
  let pitcher2_capacity : ℚ := 700
  let pitcher1_juice_fraction : ℚ := 1/4
  let pitcher2_juice_fraction : ℚ := 3/7
  let total_juice := pitcher1_capacity * pitcher1_juice_fraction + pitcher2_capacity * pitcher2_juice_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 1/3 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l2513_251361


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l2513_251345

theorem orphanage_donation_percentage
  (total_income : ℝ)
  (children_percentage : ℝ)
  (wife_percentage : ℝ)
  (final_amount : ℝ)
  (h1 : total_income = 400000)
  (h2 : children_percentage = 0.6)
  (h3 : wife_percentage = 0.25)
  (h4 : final_amount = 40000) :
  let remaining := total_income * (1 - children_percentage - wife_percentage)
  let donation := remaining - final_amount
  donation / remaining = 1/3 := by sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l2513_251345


namespace NUMINAMATH_CALUDE_am_gm_inequality_l2513_251314

theorem am_gm_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l2513_251314


namespace NUMINAMATH_CALUDE_expression_equality_l2513_251389

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 / y) :
  (x - 2 / x) * (y + 2 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2513_251389


namespace NUMINAMATH_CALUDE_chris_savings_l2513_251394

/-- Chris's savings problem -/
theorem chris_savings (total : ℕ) (grandmother : ℕ) (parents : ℕ) (aunt_uncle : ℕ) 
  (h1 : total = 279)
  (h2 : grandmother = 25)
  (h3 : parents = 75)
  (h4 : aunt_uncle = 20) :
  total - (grandmother + parents + aunt_uncle) = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_savings_l2513_251394


namespace NUMINAMATH_CALUDE_math_competition_max_a_l2513_251392

theorem math_competition_max_a (total : ℕ) (only_a only_b only_c both_ab both_ac both_bc all_three : ℕ) :
  total = 39 →
  only_a + only_b + only_c + both_ab + both_ac + both_bc + all_three = total →
  only_a = (both_ab + both_ac + all_three) + 5 →
  only_b + both_bc = 2 * (only_c + both_bc) →
  only_a = only_b + only_c →
  ∃ (max_a : ℕ), max_a = only_a + both_ab + both_ac + all_three ∧
    max_a ≤ 23 ∧
    ∀ (a : ℕ), a = only_a + both_ab + both_ac + all_three → a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_math_competition_max_a_l2513_251392


namespace NUMINAMATH_CALUDE_integer_sum_of_powers_l2513_251377

theorem integer_sum_of_powers (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_powers_l2513_251377


namespace NUMINAMATH_CALUDE_spoonfuls_per_bowl_l2513_251303

/-- Proves that the number of spoonfuls in each bowl is 25 -/
theorem spoonfuls_per_bowl
  (clusters_per_spoonful : ℕ)
  (clusters_per_box : ℕ)
  (bowls_per_box : ℕ)
  (h1 : clusters_per_spoonful = 4)
  (h2 : clusters_per_box = 500)
  (h3 : bowls_per_box = 5) :
  clusters_per_box / (bowls_per_box * clusters_per_spoonful) = 25 := by
  sorry

end NUMINAMATH_CALUDE_spoonfuls_per_bowl_l2513_251303


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2513_251317

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃! (x y : ℤ), (x < 1 ∧ x > m - 1) ∧ (y < 1 ∧ y > m - 1) ∧ x ≠ y

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2513_251317


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2513_251379

theorem unknown_blanket_rate (blanket_count_1 blanket_count_2 blanket_count_3 : ℕ)
  (price_1 price_2 average_price : ℚ) (unknown_rate : ℚ) :
  blanket_count_1 = 4 →
  blanket_count_2 = 5 →
  blanket_count_3 = 2 →
  price_1 = 100 →
  price_2 = 150 →
  average_price = 150 →
  (blanket_count_1 * price_1 + blanket_count_2 * price_2 + blanket_count_3 * unknown_rate) / 
    (blanket_count_1 + blanket_count_2 + blanket_count_3) = average_price →
  unknown_rate = 250 := by
sorry


end NUMINAMATH_CALUDE_unknown_blanket_rate_l2513_251379


namespace NUMINAMATH_CALUDE_milk_powder_sampling_l2513_251351

/-- Represents a system sampling method. -/
structure SystemSampling where
  totalItems : ℕ
  sampledItems : ℕ
  firstSampledNumber : ℕ

/-- Calculates the number of the nth sampled item in a system sampling method. -/
def nthSampledNumber (s : SystemSampling) (n : ℕ) : ℕ :=
  s.firstSampledNumber + (n - 1) * (s.totalItems / s.sampledItems)

/-- Theorem stating that for the given system sampling parameters, 
    the 41st sampled item will be numbered 607. -/
theorem milk_powder_sampling :
  let s : SystemSampling := {
    totalItems := 3000,
    sampledItems := 200,
    firstSampledNumber := 7
  }
  nthSampledNumber s 41 = 607 := by
  sorry

end NUMINAMATH_CALUDE_milk_powder_sampling_l2513_251351


namespace NUMINAMATH_CALUDE_johnny_fish_count_l2513_251356

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 40 →
  sony_multiplier = 4 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 8 := by
sorry

end NUMINAMATH_CALUDE_johnny_fish_count_l2513_251356


namespace NUMINAMATH_CALUDE_trapezoid_diagonals_l2513_251372

/-- A trapezoid with bases a and c, legs b and d, and diagonals e and f. -/
structure Trapezoid (a c b d e f : ℝ) : Prop where
  positive_a : 0 < a
  positive_c : 0 < c
  positive_b : 0 < b
  positive_d : 0 < d
  positive_e : 0 < e
  positive_f : 0 < f
  a_greater_c : a > c

/-- The diagonals of a trapezoid can be expressed in terms of its sides. -/
theorem trapezoid_diagonals (a c b d e f : ℝ) (trap : Trapezoid a c b d e f) :
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_diagonals_l2513_251372


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2513_251347

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 233 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2513_251347


namespace NUMINAMATH_CALUDE_grocery_bag_capacity_l2513_251367

theorem grocery_bag_capacity (bag_capacity : ℕ) (green_beans : ℕ) (milk : ℕ) (carrot_multiplier : ℕ) :
  bag_capacity = 20 →
  green_beans = 4 →
  milk = 6 →
  carrot_multiplier = 2 →
  bag_capacity - (green_beans + milk + carrot_multiplier * green_beans) = 2 := by
  sorry

end NUMINAMATH_CALUDE_grocery_bag_capacity_l2513_251367


namespace NUMINAMATH_CALUDE_hotel_bill_friends_count_prove_hotel_bill_friends_count_l2513_251300

theorem hotel_bill_friends_count : ℕ → Prop :=
  fun total_friends =>
    let standard_pay := 100
    let extra_pay := 100
    let actual_extra_pay := 220
    let standard_payers := 5
    let total_bill := standard_payers * standard_pay + extra_pay
    let share_per_friend := total_bill / total_friends
    total_friends = standard_payers + 1 ∧
    share_per_friend * total_friends = total_bill ∧
    share_per_friend + extra_pay = actual_extra_pay

theorem prove_hotel_bill_friends_count : 
  ∃ (n : ℕ), hotel_bill_friends_count n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bill_friends_count_prove_hotel_bill_friends_count_l2513_251300


namespace NUMINAMATH_CALUDE_min_modulus_m_l2513_251381

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum modulus of m is √(2 + 2√5). -/
theorem min_modulus_m (m : ℂ) : 
  (∃ x : ℝ, x^2 + m*x + 1 + 2*Complex.I = 0) → 
  Complex.abs m ≥ Real.sqrt (2 + 2 * Real.sqrt 5) ∧ 
  ∃ m₀ : ℂ, (∃ x : ℝ, x^2 + m₀*x + 1 + 2*Complex.I = 0) ∧ 
            Complex.abs m₀ = Real.sqrt (2 + 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_min_modulus_m_l2513_251381


namespace NUMINAMATH_CALUDE_polygon_side_containment_l2513_251385

/-- A polygon is a set of points in the plane. -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A line in the plane. -/
def Line : Type := Set (ℝ × ℝ)

/-- The number of sides of a polygon. -/
def numSides (p : Polygon) : ℕ := sorry

/-- A line contains a side of a polygon. -/
def containsSide (l : Line) (p : Polygon) : Prop := sorry

/-- A line contains exactly one side of a polygon. -/
def containsExactlyOneSide (l : Line) (p : Polygon) : Prop := sorry

/-- Main theorem about 13-sided polygons and polygons with more than 13 sides. -/
theorem polygon_side_containment :
  (∀ p : Polygon, numSides p = 13 → ∃ l : Line, containsExactlyOneSide l p) ∧
  (∀ n : ℕ, n > 13 → ∃ p : Polygon, numSides p = n ∧ 
    ∀ l : Line, containsSide l p → ¬containsExactlyOneSide l p) :=
sorry

end NUMINAMATH_CALUDE_polygon_side_containment_l2513_251385


namespace NUMINAMATH_CALUDE_crosswalk_wait_probability_l2513_251368

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℝ := 40

/-- Represents the minimum waiting time in seconds -/
def min_wait_time : ℝ := 15

/-- Theorem: The probability of waiting at least 15 seconds for a green light
    when encountering a red light that lasts 40 seconds is 5/8 -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time) / red_light_duration = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_wait_probability_l2513_251368


namespace NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2513_251391

theorem prime_square_difference_divisibility (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - b^2) % (p - a^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2513_251391


namespace NUMINAMATH_CALUDE_video_game_lives_l2513_251350

/-- 
Given a video game scenario where:
- x is the initial number of lives
- y is the number of power-ups collected
- Each power-up gives 5 extra lives
- The player lost 13 lives
- After these events, the player ended up with 70 lives

Prove that the initial number of lives (x) is equal to 83 minus 5 times 
the number of power-ups collected (y).
-/
theorem video_game_lives (x y : ℤ) : 
  (x - 13 + 5 * y = 70) → (x = 83 - 5 * y) := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l2513_251350


namespace NUMINAMATH_CALUDE_installation_solution_l2513_251341

/-- Represents the number of installations of each type -/
structure Installations where
  type1 : ℕ
  type2 : ℕ
  type3 : ℕ

/-- Checks if the given installation numbers satisfy all conditions -/
def satisfiesConditions (i : Installations) : Prop :=
  i.type1 + i.type2 + i.type3 ≥ 100 ∧
  i.type2 = 4 * i.type1 ∧
  ∃ k : ℕ, i.type3 = k * i.type1 ∧
  5 * i.type3 = i.type2 + 22

theorem installation_solution :
  ∃ i : Installations, satisfiesConditions i ∧ i.type1 = 22 ∧ i.type2 = 88 ∧ i.type3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_installation_solution_l2513_251341


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2513_251312

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (1 + Complex.I) / (1 + a * Complex.I)) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2513_251312


namespace NUMINAMATH_CALUDE_least_n_divisibility_l2513_251355

theorem least_n_divisibility (n : ℕ) : n = 5 ↔ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*n → 
    (∃ m : ℕ, m ≥ 1 ∧ m ≤ 2*n ∧ (n^2 - n + m) % m = 0) ∧ 
    (∃ l : ℕ, l ≥ 1 ∧ l ≤ 2*n ∧ (n^2 - n + l) % l ≠ 0)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∀ k : ℕ, 1 ≤ k ∧ k ≤ 2*m → 
      (∃ p : ℕ, p ≥ 1 ∧ p ≤ 2*m ∧ (m^2 - m + p) % p = 0) ∧ 
      (∃ q : ℕ, q ≥ 1 ∧ q ≤ 2*m ∧ (m^2 - m + q) % q ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisibility_l2513_251355


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2513_251380

/-- Given a point P(-2,3) in a plane rectangular coordinate system,
    its coordinates with respect to the origin are (2,-3). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-2, 3)
  let origin_symmetric (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  origin_symmetric P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2513_251380


namespace NUMINAMATH_CALUDE_triangle_properties_l2513_251331

/-- Given a triangle ABC with the following properties:
  - a = 2√2
  - sin C = √2 * sin A
  - cos C = √2/4
  Prove that:
  - c = 4
  - The area of the triangle is 2√7
-/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 * Real.sqrt 2 →
  Real.sin C = Real.sqrt 2 * Real.sin A →
  Real.cos C = Real.sqrt 2 / 4 →
  c = 4 ∧
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2513_251331


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2513_251340

theorem gcd_of_three_numbers : Nat.gcd 13847 (Nat.gcd 21353 34691) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2513_251340


namespace NUMINAMATH_CALUDE_min_value_of_f_l2513_251310

/-- The function f(x,y) represents the given expression -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y + 20

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -5) ∧ (∃ x y : ℝ, f x y = -5) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2513_251310


namespace NUMINAMATH_CALUDE_ellipse_properties_l2513_251328

/-- The ellipse C: x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l: y = kx, where k ≠ 0 -/
def line_l (k x y : ℝ) : Prop := y = k * x ∧ k ≠ 0

/-- M and N are intersection points of line l and ellipse C -/
def intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  ellipse_C M.1 M.2 ∧ ellipse_C N.1 N.2 ∧
  line_l k M.1 M.2 ∧ line_l k N.1 N.2

/-- F₁ and F₂ are the foci of the ellipse C -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-1, 0) ∧ F₂ = (1, 0)

/-- B is the top vertex of the ellipse C -/
def top_vertex (B : ℝ × ℝ) : Prop :=
  B = (0, Real.sqrt 3)

/-- The perimeter of quadrilateral MF₁NF₂ is 8 -/
def perimeter_is_8 (M N F₁ F₂ : ℝ × ℝ) : Prop :=
  dist M F₁ + dist F₁ N + dist N F₂ + dist F₂ M = 8

/-- The product of the slopes of lines BM and BN is -3/4 -/
def slope_product (M N B : ℝ × ℝ) : Prop :=
  ((M.2 - B.2) / (M.1 - B.1)) * ((N.2 - B.2) / (N.1 - B.1)) = -3/4

theorem ellipse_properties (k : ℝ) (M N F₁ F₂ B : ℝ × ℝ) :
  intersection_points M N k →
  foci F₁ F₂ →
  top_vertex B →
  perimeter_is_8 M N F₁ F₂ ∧ slope_product M N B :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2513_251328


namespace NUMINAMATH_CALUDE_shirts_produced_theorem_l2513_251384

/-- An industrial machine that produces shirts. -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  minutes_worked_today : ℕ

/-- Calculates the total number of shirts produced by the machine today. -/
def shirts_produced_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_per_minute * machine.minutes_worked_today

/-- Theorem stating that a machine producing 6 shirts per minute working for 12 minutes produces 72 shirts. -/
theorem shirts_produced_theorem (machine : ShirtMachine)
    (h1 : machine.shirts_per_minute = 6)
    (h2 : machine.minutes_worked_today = 12) :
    shirts_produced_today machine = 72 := by
  sorry

end NUMINAMATH_CALUDE_shirts_produced_theorem_l2513_251384


namespace NUMINAMATH_CALUDE_smallest_three_digit_equal_sum_l2513_251365

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Proposition: 999 is the smallest three-digit number n such that
    Σ(n) = Σ(2n) = Σ(3n) = ... = Σ(n^2), where Σ(n) denotes the sum of the digits of n -/
theorem smallest_three_digit_equal_sum : 
  ∀ n : ℕ, 100 ≤ n → n < 999 → 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ sumOfDigits n ≠ sumOfDigits (k * n)) ∨
  sumOfDigits n ≠ sumOfDigits (n * n) :=
by sorry

#check smallest_three_digit_equal_sum

end NUMINAMATH_CALUDE_smallest_three_digit_equal_sum_l2513_251365


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2513_251375

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2) →
  p 2 (3^19 + 11^13) ∧ 
  ∀ q : ℕ, p q (3^19 + 11^13) → q ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2513_251375


namespace NUMINAMATH_CALUDE_s_iff_m_range_p_or_q_and_not_q_implies_m_range_l2513_251334

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), x^2 / (4 - m) + y^2 / m = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def s (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 = 0

-- Theorem 1
theorem s_iff_m_range (m : ℝ) : s m ↔ m < 0 ∨ m ≥ 2 := by sorry

-- Theorem 2
theorem p_or_q_and_not_q_implies_m_range (m : ℝ) : (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_s_iff_m_range_p_or_q_and_not_q_implies_m_range_l2513_251334


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2513_251322

theorem compound_interest_rate (P : ℝ) (t : ℕ) (CI : ℝ) (r : ℝ) : 
  P = 4500 →
  t = 2 →
  CI = 945.0000000000009 →
  (P + CI) = P * (1 + r) ^ t →
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2513_251322


namespace NUMINAMATH_CALUDE_area_between_circles_l2513_251321

/-- Given two concentric circles where the outer radius is twice the inner radius
    and the width between circles is 3, prove the area between circles is 27π -/
theorem area_between_circles (r : ℝ) (h1 : r > 0) (h2 : 2 * r - r = 3) :
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l2513_251321


namespace NUMINAMATH_CALUDE_range_of_m_l2513_251360

-- Define the sets A and B
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 10}
def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, x ∈ A → x ∈ B m) →
  (∃ x : ℝ, x ∈ B m ∧ x ∉ A) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2513_251360


namespace NUMINAMATH_CALUDE_bus_seats_columns_l2513_251359

/-- The number of rows in each bus -/
def rows : ℕ := 10

/-- The number of buses -/
def buses : ℕ := 6

/-- The total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- The number of columns of seats in each bus -/
def columns : ℕ := 4

theorem bus_seats_columns :
  columns * rows * buses = total_students :=
sorry

end NUMINAMATH_CALUDE_bus_seats_columns_l2513_251359


namespace NUMINAMATH_CALUDE_same_height_time_l2513_251302

/-- Represents the height of a ball as a function of time -/
def ball_height (a : ℝ) (h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time :
  ∀ (a : ℝ) (h : ℝ),
  a ≠ 0 →
  ∃ (t : ℝ),
  t = 2.2 ∧
  ball_height a h t = ball_height a h (t - 2) :=
sorry

end NUMINAMATH_CALUDE_same_height_time_l2513_251302


namespace NUMINAMATH_CALUDE_ratio_is_sixteen_thirteenths_l2513_251353

/-- An arithmetic sequence with a non-zero common difference where a₉, a₃, and a₁ form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : a 3 ^ 2 = a 1 * a 9

/-- The ratio of (a₂ + a₄ + a₁₀) to (a₁ + a₃ + a₉) is 16/13 -/
theorem ratio_is_sixteen_thirteenths (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_sixteen_thirteenths_l2513_251353


namespace NUMINAMATH_CALUDE_cannot_fit_rectangles_l2513_251313

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- The large rectangle -/
def largeRectangle : Rectangle := { width := 13, height := 7 }

/-- The small rectangle -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- The number of small rectangles -/
def numSmallRectangles : ℕ := 15

/-- Theorem stating that it's not possible to fit 15 small rectangles into the large rectangle -/
theorem cannot_fit_rectangles : 
  area largeRectangle > numSmallRectangles * area smallRectangle :=
sorry

end NUMINAMATH_CALUDE_cannot_fit_rectangles_l2513_251313


namespace NUMINAMATH_CALUDE_sqrt_sum_given_diff_l2513_251342

theorem sqrt_sum_given_diff (x : ℝ) :
  Real.sqrt (100 - x^2) - Real.sqrt (36 - x^2) = 5 →
  Real.sqrt (100 - x^2) + Real.sqrt (36 - x^2) = 12.8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_given_diff_l2513_251342


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2513_251301

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 69)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2513_251301


namespace NUMINAMATH_CALUDE_crackers_duration_l2513_251397

theorem crackers_duration (crackers_per_sandwich : ℕ) (sandwiches_per_night : ℕ)
  (sleeves_per_box : ℕ) (crackers_per_sleeve : ℕ) (num_boxes : ℕ)
  (h1 : crackers_per_sandwich = 2)
  (h2 : sandwiches_per_night = 5)
  (h3 : sleeves_per_box = 4)
  (h4 : crackers_per_sleeve = 28)
  (h5 : num_boxes = 5) :
  (num_boxes * sleeves_per_box * crackers_per_sleeve) / (crackers_per_sandwich * sandwiches_per_night) = 56 := by
  sorry

end NUMINAMATH_CALUDE_crackers_duration_l2513_251397


namespace NUMINAMATH_CALUDE_sum_even_divisors_180_l2513_251362

/-- Sum of positive even divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive even divisors of 180 is 468 -/
theorem sum_even_divisors_180 : sumEvenDivisors 180 = 468 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_180_l2513_251362


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2513_251369

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - a|

-- Theorem for part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ∧ ∃ y : ℝ, f 1 y = 2 :=
sorry

-- Theorem for part II
theorem range_of_a_when_two_not_in_solution_set :
  ∀ a : ℝ, (f a 2 > 5) ↔ (a < -5/2 ∨ a > 5/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2513_251369


namespace NUMINAMATH_CALUDE_largest_intersection_is_eight_l2513_251318

/-- A polynomial of degree 6 -/
def P (a b c : ℝ) (x : ℝ) : ℝ :=
  x^6 - 14*x^5 + 45*x^4 - 30*x^3 + a*x^2 + b*x + c

/-- A linear function -/
def L (d e : ℝ) (x : ℝ) : ℝ :=
  d*x + e

/-- The difference between P and L -/
def Q (a b c d e : ℝ) (x : ℝ) : ℝ :=
  P a b c x - L d e x

theorem largest_intersection_is_eight (a b c d e : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧
    ∀ x : ℝ, Q a b c d e x = 0 ↔ (x = p ∨ x = q ∨ x = r) ∧
    ∀ x : ℝ, x ≠ p ∧ x ≠ q ∧ x ≠ r → Q a b c d e x > 0) →
  r = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_is_eight_l2513_251318


namespace NUMINAMATH_CALUDE_range_of_x_plus_cos_y_l2513_251383

theorem range_of_x_plus_cos_y (x y : ℝ) (h : 2 * x + Real.cos (2 * y) = 1) :
  ∃ (z : ℝ), z = x + Real.cos y ∧ -1 ≤ z ∧ z ≤ 5/4 ∧
  (∃ (x' y' : ℝ), 2 * x' + Real.cos (2 * y') = 1 ∧ x' + Real.cos y' = -1) ∧
  (∃ (x'' y'' : ℝ), 2 * x'' + Real.cos (2 * y'') = 1 ∧ x'' + Real.cos y'' = 5/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_cos_y_l2513_251383


namespace NUMINAMATH_CALUDE_spending_difference_l2513_251325

/-- Represents the price of masks and the quantities purchased by Jiajia and Qiqi. -/
structure MaskPurchase where
  a : ℝ  -- Price of N95 mask in yuan
  b : ℝ  -- Price of regular medical mask in yuan
  jiajia_n95 : ℕ := 5  -- Number of N95 masks Jiajia bought
  jiajia_regular : ℕ := 2  -- Number of regular masks Jiajia bought
  qiqi_n95 : ℕ := 2  -- Number of N95 masks Qiqi bought
  qiqi_regular : ℕ := 5  -- Number of regular masks Qiqi bought

/-- The price difference between N95 and regular masks is 3 yuan. -/
def price_difference (m : MaskPurchase) : Prop :=
  m.a = m.b + 3

/-- The difference in spending between Jiajia and Qiqi is 9 yuan. -/
theorem spending_difference (m : MaskPurchase) 
  (h : price_difference m) : 
  (m.jiajia_n95 : ℝ) * m.a + (m.jiajia_regular : ℝ) * m.b - 
  ((m.qiqi_n95 : ℝ) * m.a + (m.qiqi_regular : ℝ) * m.b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l2513_251325


namespace NUMINAMATH_CALUDE_value_of_b_l2513_251304

theorem value_of_b (a c b : ℝ) : 
  a = 105 → 
  c = 70 → 
  a^4 = 21 * 25 * 15 * b * c^3 → 
  b = 0.045 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2513_251304


namespace NUMINAMATH_CALUDE_probability_total_more_than_seven_is_five_twelfths_l2513_251346

/-- The number of possible outcomes when throwing a pair of dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (total > 7) when throwing a pair of dice -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probability_total_more_than_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_more_than_seven_is_five_twelfths :
  probability_total_more_than_seven = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_total_more_than_seven_is_five_twelfths_l2513_251346


namespace NUMINAMATH_CALUDE_min_cups_to_fill_cylinder_l2513_251364

def cylinder_capacity : ℚ := 980
def cup_capacity : ℚ := 80

theorem min_cups_to_fill_cylinder :
  ⌈cylinder_capacity / cup_capacity⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_cups_to_fill_cylinder_l2513_251364


namespace NUMINAMATH_CALUDE_count_satisfying_pairs_l2513_251306

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 25) ∧ ((a - 3)^2 + b^2 < 20) ∧ (a^2 + (b - 3)^2 < 20)

theorem count_satisfying_pairs :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_inequalities p.1 p.2) ∧
    s.card = 7 :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_pairs_l2513_251306


namespace NUMINAMATH_CALUDE_number_difference_l2513_251348

theorem number_difference (L S : ℕ) (hL : L = 1600) (hDiv : L = S * 16 + 15) : L - S = 1501 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2513_251348


namespace NUMINAMATH_CALUDE_add_5_23_base6_l2513_251395

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 -/
def addBase6 (a b : ℕ) : ℕ := base10To6 (base6To10 a + base6To10 b)

theorem add_5_23_base6 : addBase6 5 23 = 32 := by sorry

end NUMINAMATH_CALUDE_add_5_23_base6_l2513_251395


namespace NUMINAMATH_CALUDE_function_value_proof_l2513_251366

/-- Given a function f(x, z) = 2x^2 + y - z where f(2, 3) = 100, prove that f(5, 7) = 138 -/
theorem function_value_proof (y : ℝ) : 
  let f : ℝ → ℝ → ℝ := λ x z ↦ 2 * x^2 + y - z
  (f 2 3 = 100) → (f 5 7 = 138) := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l2513_251366


namespace NUMINAMATH_CALUDE_bridget_initial_skittles_l2513_251315

/-- Proves that Bridget initially has 4 Skittles given the problem conditions. -/
theorem bridget_initial_skittles : 
  ∀ (bridget_initial henry_skittles bridget_final : ℕ),
  henry_skittles = 4 →
  bridget_final = bridget_initial + henry_skittles →
  bridget_final = 8 →
  bridget_initial = 4 := by sorry

end NUMINAMATH_CALUDE_bridget_initial_skittles_l2513_251315


namespace NUMINAMATH_CALUDE_massager_vibration_increase_l2513_251305

/-- Given a massager with a lowest setting of 1600 vibrations per second
    and a highest setting that produces 768,000 vibrations in 5 minutes,
    prove that the percentage increase from lowest to highest setting is 60% -/
theorem massager_vibration_increase (lowest : ℝ) (highest_total : ℝ) (duration : ℝ) :
  lowest = 1600 →
  highest_total = 768000 →
  duration = 5 * 60 →
  (highest_total / duration - lowest) / lowest * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_massager_vibration_increase_l2513_251305


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2513_251343

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2513_251343


namespace NUMINAMATH_CALUDE_lcm_and_gcd_of_36_and_48_l2513_251344

theorem lcm_and_gcd_of_36_and_48 :
  (Nat.lcm 36 48 = 144) ∧ (Nat.gcd 36 48 = 12) := by
  sorry

end NUMINAMATH_CALUDE_lcm_and_gcd_of_36_and_48_l2513_251344


namespace NUMINAMATH_CALUDE_evaluate_expression_l2513_251371

theorem evaluate_expression : (2 * 4 * 6) * (1/2 + 1/4 + 1/6) = 44 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2513_251371


namespace NUMINAMATH_CALUDE_units_digit_base_8_l2513_251329

theorem units_digit_base_8 (n₁ n₂ : ℕ) (h₁ : n₁ = 198) (h₂ : n₂ = 53) :
  (((n₁ - 3) * (n₂ + 7)) % 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_base_8_l2513_251329


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l2513_251337

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-axis intersection point of a line -/
def yAxisIntersection (l : Line) : ℝ × ℝ := sorry

/-- The specific line passing through (5, 25) and (-5, 5) -/
def specificLine : Line := { x₁ := 5, y₁ := 25, x₂ := -5, y₂ := 5 }

theorem line_intersects_y_axis :
  yAxisIntersection specificLine = (0, 15) := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l2513_251337


namespace NUMINAMATH_CALUDE_square_sequence_problem_l2513_251307

/-- The number of squares in the nth figure of the sequence -/
def g (n : ℕ) : ℕ :=
  2 * n^2 + 4 * n + 3

theorem square_sequence_problem :
  g 0 = 3 ∧ g 1 = 9 ∧ g 2 = 19 ∧ g 3 = 33 → g 100 = 20403 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_problem_l2513_251307


namespace NUMINAMATH_CALUDE_binomial_difference_squares_l2513_251374

theorem binomial_difference_squares (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_difference_squares_l2513_251374


namespace NUMINAMATH_CALUDE_twelfth_number_is_seven_l2513_251376

/-- A circular arrangement of 20 numbers -/
def CircularArrangement := Fin 20 → ℕ

/-- The property that the sum of any six consecutive numbers is 24 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 20, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 24

theorem twelfth_number_is_seven
  (arr : CircularArrangement)
  (h_sum : SumProperty arr)
  (h_first : arr 0 = 1) :
  arr 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_number_is_seven_l2513_251376


namespace NUMINAMATH_CALUDE_train_passing_length_l2513_251354

/-- The length of a train passing another train in opposite direction -/
theorem train_passing_length (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 50) (h2 : v2 = 62) (h3 : t = 9) :
  let relative_speed := (v1 + v2) * (1000 / 3600)
  let train_length := relative_speed * t
  ∃ ε > 0, |train_length - 280| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_length_l2513_251354


namespace NUMINAMATH_CALUDE_raft_capacity_raft_problem_l2513_251309

/-- Calculates the number of people that can fit on a raft given certain conditions -/
theorem raft_capacity (max_capacity : ℕ) (reduction_full : ℕ) (life_jackets_needed : ℕ) : ℕ :=
  let capacity_with_jackets := max_capacity - reduction_full
  let jacket_to_person_ratio := capacity_with_jackets / reduction_full
  let reduction := life_jackets_needed / jacket_to_person_ratio
  max_capacity - reduction

/-- Proves that 17 people can fit on the raft under given conditions -/
theorem raft_problem : raft_capacity 21 7 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_raft_capacity_raft_problem_l2513_251309


namespace NUMINAMATH_CALUDE_x_gt_one_necessary_not_sufficient_for_x_gt_two_l2513_251316

theorem x_gt_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_necessary_not_sufficient_for_x_gt_two_l2513_251316


namespace NUMINAMATH_CALUDE_angle_c_value_l2513_251336

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem angle_c_value (t : Triangle) 
  (h : t.a^2 + t.b^2 = t.c^2 + Real.sqrt 3 * t.a * t.b) : 
  t.C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_value_l2513_251336


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2513_251326

/-- Given vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 2*b, then k = 2 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, -1))
  (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 2 * b.1, a.2 - 2 * b.2) = 0) :
  k = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2513_251326


namespace NUMINAMATH_CALUDE_sequence_formula_l2513_251387

def S₁ (n : ℕ) : ℕ := n^2

def S₂ (n : ℕ) : ℕ := n^2 + n + 1

def a₁ (n : ℕ) : ℕ := 2*n - 1

def a₂ (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2*n

theorem sequence_formula (n : ℕ) (h : n ≥ 1) :
  (∀ k, S₁ k - S₁ (k-1) = a₁ k) ∧
  (∀ k, S₂ k - S₂ (k-1) = a₂ k) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2513_251387


namespace NUMINAMATH_CALUDE_derangement_of_five_l2513_251378

/-- Calculates the number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (k + 1) * (derangement (k + 1) + derangement k)

/-- The number of derangements for 5 elements is 44 -/
theorem derangement_of_five : derangement 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_derangement_of_five_l2513_251378


namespace NUMINAMATH_CALUDE_josh_initial_money_l2513_251335

-- Define the given conditions
def spent_on_drink : ℝ := 1.75
def spent_additional : ℝ := 1.25
def money_left : ℝ := 6.00

-- Define the theorem
theorem josh_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = spent_on_drink + spent_additional + money_left ∧
    initial_money = 9.00 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l2513_251335


namespace NUMINAMATH_CALUDE_factors_of_polynomial_l2513_251386

theorem factors_of_polynomial (x : ℝ) : 
  (x^4 - 4*x^2 + 4 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2)) ∧ 
  (x^4 - 4*x^2 + 4 ≠ (x - 1) * (x^3 + x^2 + x + 1)) ∧
  (x^4 - 4*x^2 + 4 ≠ (x^2 + 2) * (x^2 - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_polynomial_l2513_251386


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l2513_251398

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := (x + 3)^2 + 3

-- Theorem statement
theorem intersection_point_y_coordinate :
  shifted_function 0 = 12 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l2513_251398


namespace NUMINAMATH_CALUDE_total_football_games_l2513_251308

theorem total_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l2513_251308


namespace NUMINAMATH_CALUDE_vasya_petya_number_ambiguity_l2513_251332

theorem vasya_petya_number_ambiguity (a : ℝ) (ha : a ≠ 0) :
  ∃ b : ℝ, b ≠ a ∧ a^4 + a^2 = b^4 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_vasya_petya_number_ambiguity_l2513_251332


namespace NUMINAMATH_CALUDE_price_change_l2513_251327

/-- Calculates the final price of an item after three price changes -/
theorem price_change (initial_price : ℝ) : 
  initial_price = 320 → 
  (initial_price * 1.15 * 0.9 * 1.25) = 414 := by
  sorry


end NUMINAMATH_CALUDE_price_change_l2513_251327


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l2513_251324

theorem max_product_constrained_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  x * y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l2513_251324


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2513_251363

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2513_251363


namespace NUMINAMATH_CALUDE_intersection_sum_l2513_251333

/-- The quadratic function h(x) = -x^2 - 4x + 1 -/
def h (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- The function j(x) = -h(x) -/
def j (x : ℝ) : ℝ := -h x

/-- The function k(x) = h(-x) -/
def k (x : ℝ) : ℝ := h (-x)

/-- The number of intersection points between y = h(x) and y = j(x) -/
def c : ℕ := 2

/-- The number of intersection points between y = h(x) and y = k(x) -/
def d : ℕ := 1

/-- Theorem: Given the functions h, j, k, and the intersection counts c and d, 10c + d = 21 -/
theorem intersection_sum : 10 * c + d = 21 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2513_251333


namespace NUMINAMATH_CALUDE_hollow_circles_count_l2513_251352

/-- Represents the pattern of circles, where each number is the position of a hollow circle in the repeating sequence -/
def hollow_circle_positions : List Nat := [2, 5, 9]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 9

/-- The total number of circles in the sequence -/
def total_circles : Nat := 2001

/-- Calculates the number of hollow circles in a sequence of given length -/
def count_hollow_circles (n : Nat) : Nat :=
  (n / sequence_length) * hollow_circle_positions.length + 
  (hollow_circle_positions.filter (· ≤ n % sequence_length)).length

theorem hollow_circles_count :
  count_hollow_circles total_circles = 667 := by
  sorry

end NUMINAMATH_CALUDE_hollow_circles_count_l2513_251352


namespace NUMINAMATH_CALUDE_longest_tennis_match_duration_l2513_251396

theorem longest_tennis_match_duration (hours : ℕ) (minutes : ℕ) : 
  hours = 11 ∧ minutes = 5 → hours * 60 + minutes = 665 := by sorry

end NUMINAMATH_CALUDE_longest_tennis_match_duration_l2513_251396


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l2513_251382

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (h1 : R = 15) (h2 : OM = 13) (h3 : AB = 18) :
  ∃ (AM MB : ℝ), AM + MB = AB ∧ AM = 14 ∧ MB = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l2513_251382


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2513_251390

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3*a) ↔ (a ≥ 4 ∨ a ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2513_251390


namespace NUMINAMATH_CALUDE_otimes_twelve_nine_l2513_251399

/-- The custom binary operation ⊗ -/
def otimes (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

/-- Theorem stating that 12 ⊗ 9 = 124/9 -/
theorem otimes_twelve_nine : otimes 12 9 = 124 / 9 := by
  sorry

end NUMINAMATH_CALUDE_otimes_twelve_nine_l2513_251399


namespace NUMINAMATH_CALUDE_product_with_9999_l2513_251319

theorem product_with_9999 : ∃ x : ℕ, x * 9999 = 4691100843 ∧ x = 469143 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l2513_251319


namespace NUMINAMATH_CALUDE_range_of_a_a_upper_bound_range_of_a_characterization_l2513_251370

def A : Set ℝ := {x | x - 1 > 0}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty → a > 1 := by sorry

theorem a_upper_bound : ∀ a : ℝ, a > 1 → (A ∩ B a).Nonempty := by sorry

theorem range_of_a_characterization :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_a_upper_bound_range_of_a_characterization_l2513_251370


namespace NUMINAMATH_CALUDE_hanson_employees_count_l2513_251358

theorem hanson_employees_count :
  ∃ (E : ℕ) (M B B' : ℤ), 
    M = E * B + 2 ∧ 
    3 * M = E * B' + 1 → 
    E = 5 := by
  sorry

end NUMINAMATH_CALUDE_hanson_employees_count_l2513_251358


namespace NUMINAMATH_CALUDE_average_carnations_l2513_251357

def bouquet1 : ℕ := 9
def bouquet2 : ℕ := 14
def bouquet3 : ℕ := 13
def total_bouquets : ℕ := 3

theorem average_carnations :
  (bouquet1 + bouquet2 + bouquet3) / total_bouquets = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_carnations_l2513_251357


namespace NUMINAMATH_CALUDE_sara_quarters_l2513_251323

def initial_quarters : ℕ := 21
def dad_gave : ℕ := 49
def spent : ℕ := 15
def mom_gave_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters (x : ℕ) : 
  initial_quarters + dad_gave - spent + (mom_gave_dollars * quarters_per_dollar) + x = 63 + x := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2513_251323


namespace NUMINAMATH_CALUDE_fruit_group_sizes_l2513_251339

def bananas : ℕ := 527
def oranges : ℕ := 386
def apples : ℕ := 319

def banana_groups : ℕ := 11
def orange_groups : ℕ := 103
def apple_groups : ℕ := 17

def banana_group_size : ℕ := bananas / banana_groups
def orange_group_size : ℕ := oranges / orange_groups
def apple_group_size : ℕ := apples / apple_groups

theorem fruit_group_sizes :
  banana_group_size = 47 ∧ orange_group_size = 3 ∧ apple_group_size = 18 :=
sorry

end NUMINAMATH_CALUDE_fruit_group_sizes_l2513_251339


namespace NUMINAMATH_CALUDE_triangle_properties_l2513_251388

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the incircle of a triangle -/
def incircle_radius (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Theorem: In triangle ABC, if a = 8 and the incircle radius is √3, then A = π/3 and the area is 11√3 -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 8) 
  (h2 : incircle_radius t = Real.sqrt 3) : 
  t.A = π/3 ∧ triangle_area t = 11 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2513_251388


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2513_251330

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 0 ∧ x + 1 ≥ 0}
  S = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2513_251330


namespace NUMINAMATH_CALUDE_no_valid_solution_l2513_251320

theorem no_valid_solution : ¬∃ (x y z : ℕ+), 
  (x * y * z = 4 * (x + y + z)) ∧ 
  (x * y = z + x) ∧ 
  ∃ (k : ℕ+), x * y * z = k * k :=
by sorry

end NUMINAMATH_CALUDE_no_valid_solution_l2513_251320


namespace NUMINAMATH_CALUDE_max_score_calculation_l2513_251338

/-- Given the percentages scored by three students and their average score,
    calculate the maximum possible score in the exam. -/
theorem max_score_calculation (p1 p2 p3 avg : ℝ) 
    (h1 : p1 = 64)
    (h2 : p2 = 36)
    (h3 : p3 = 44)
    (h4 : avg = 432)
    (h5 : (p1 + p2 + p3) / 300 * max_score = avg) :
  max_score = 900 := by
  sorry

#check max_score_calculation

end NUMINAMATH_CALUDE_max_score_calculation_l2513_251338


namespace NUMINAMATH_CALUDE_number_of_possible_a_values_l2513_251311

theorem number_of_possible_a_values : ∃ (S : Finset ℕ),
  (∀ a ∈ S, ∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) ∧
  (∀ a : ℕ, (∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) → a ∈ S) ∧
  Finset.card S = 513 :=
sorry

end NUMINAMATH_CALUDE_number_of_possible_a_values_l2513_251311
