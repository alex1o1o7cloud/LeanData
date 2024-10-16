import Mathlib

namespace NUMINAMATH_CALUDE_power_difference_theorem_l2073_207311

def solution_set : Set (ℕ × ℕ) := {(0, 1), (2, 1), (2, 2), (1, 2)}

theorem power_difference_theorem :
  {(m, n) : ℕ × ℕ | (3:ℤ)^m - (2:ℤ)^n ∈ ({-1, 5, 7} : Set ℤ)} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_power_difference_theorem_l2073_207311


namespace NUMINAMATH_CALUDE_min_episodes_watched_l2073_207398

/-- Represents the number of episodes aired on each day of the week -/
def weekly_schedule : List Nat := [0, 1, 1, 1, 1, 2, 2]

/-- The total number of episodes in the TV series -/
def total_episodes : Nat := 60

/-- The duration of Xiaogao's trip in days -/
def trip_duration : Nat := 17

/-- Calculates the maximum number of episodes that can be aired during the trip -/
def max_episodes_during_trip (schedule : List Nat) (duration : Nat) : Nat :=
  sorry

/-- Theorem: The minimum number of episodes Xiaogao can watch is 39 -/
theorem min_episodes_watched : 
  total_episodes - max_episodes_during_trip weekly_schedule trip_duration = 39 := by
  sorry

end NUMINAMATH_CALUDE_min_episodes_watched_l2073_207398


namespace NUMINAMATH_CALUDE_investment_difference_l2073_207305

theorem investment_difference (x y : ℝ) : 
  x = 1000 →
  x + y = 1000 →
  0.02 * x + 0.04 * (x + y) = 92 →
  y = 800 := by
sorry

end NUMINAMATH_CALUDE_investment_difference_l2073_207305


namespace NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2073_207322

/-- Given an ellipse and a line passing through its midpoint, prove the line's equation -/
theorem line_through_ellipse_midpoint (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := (1, 1)
  let ellipse (p : ℝ × ℝ) := p.1^2 / 4 + p.2^2 / 3 = 1
  ellipse A ∧ ellipse B ∧  -- A and B are on the ellipse
  (∃ (k m : ℝ), ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M)) ∧  -- A, B, and M are collinear
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ (k m : ℝ), k = 3 ∧ m = -7 ∧ ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M) :=
by sorry


end NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2073_207322


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_same_digit_palindromes_l2073_207392

def is_three_digit_same_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ n = 100 * d + 10 * d + d

theorem greatest_common_factor_of_three_digit_same_digit_palindromes :
  ∃ (gcf : ℕ), gcf = 111 ∧
  (∀ n : ℕ, is_three_digit_same_digit_palindrome n → gcf ∣ n) ∧
  (∀ m : ℕ, (∀ n : ℕ, is_three_digit_same_digit_palindrome n → m ∣ n) → m ≤ gcf) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_same_digit_palindromes_l2073_207392


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l2073_207390

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 40 →
  editors > 38 →
  (total = writers + editors - x + 2 * x) →
  (∀ y : ℕ, y > x → ¬(total = writers + editors - y + 2 * y)) →
  x = 21 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l2073_207390


namespace NUMINAMATH_CALUDE_pascal_triangle_29th_row_28th_number_l2073_207393

theorem pascal_triangle_29th_row_28th_number : Nat.choose 29 27 = 406 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_29th_row_28th_number_l2073_207393


namespace NUMINAMATH_CALUDE_distance_calculation_l2073_207344

/-- The distance between A and B's homes from the city -/
def distance_difference : ℝ := 3

/-- The ratio of A's walking speed to B's walking speed -/
def walking_speed_ratio : ℝ := 1.5

/-- The ratio of B's truck speed to A's car speed -/
def vehicle_speed_ratio : ℝ := 1.5

/-- The ratio of A's car speed to A's walking speed -/
def car_to_walk_ratio : ℝ := 2

/-- B's distance from the city -/
def b_distance : ℝ := 13.5

/-- A's distance from the city -/
def a_distance : ℝ := 16.5

/-- The theorem stating that given the conditions, A lives 16.5 km from the city and B lives 13.5 km from the city -/
theorem distance_calculation :
  (a_distance - b_distance = distance_difference) ∧
  (a_distance = 16.5) ∧
  (b_distance = 13.5) := by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l2073_207344


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2073_207347

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2073_207347


namespace NUMINAMATH_CALUDE_min_absolute_difference_l2073_207302

/-- The minimum absolute difference between n and m, given f(m) = g(n) -/
theorem min_absolute_difference (f g : ℝ → ℝ) (m n : ℝ) : 
  (f = fun x ↦ Real.exp x + 2 * x) →
  (g = fun x ↦ 4 * x) →
  (f m = g n) →
  ∃ (min_diff : ℝ), 
    (∀ (m' n' : ℝ), f m' = g n' → |n' - m'| ≥ min_diff) ∧ 
    (min_diff = 1/2 - 1/2 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_min_absolute_difference_l2073_207302


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l2073_207318

theorem system_of_inequalities_solution (x : ℝ) :
  (2 * x + 1 > x ∧ x < -3 * x + 8) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l2073_207318


namespace NUMINAMATH_CALUDE_decimal_division_equivalence_l2073_207388

theorem decimal_division_equivalence : 
  ∀ (a b : ℚ), a = 11.7 ∧ b = 2.6 → 
    (a / b = 117 / 26) ∧ (a / b = 4.5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_equivalence_l2073_207388


namespace NUMINAMATH_CALUDE_pq_relation_l2073_207324

theorem pq_relation (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 ∨ q = 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pq_relation_l2073_207324


namespace NUMINAMATH_CALUDE_z_equation_solution_l2073_207315

theorem z_equation_solution :
  let z : ℝ := Real.sqrt ((Real.sqrt 29) / 2 + 7 / 2)
  ∃! (d e f : ℕ+),
    z^100 = 2*z^98 + 14*z^96 + 11*z^94 - z^50 + (d : ℝ)*z^46 + (e : ℝ)*z^44 + (f : ℝ)*z^40 ∧
    d + e + f = 205 := by
  sorry

end NUMINAMATH_CALUDE_z_equation_solution_l2073_207315


namespace NUMINAMATH_CALUDE_valid_numbers_l2073_207364

def is_valid_number (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (Nat.divisors n).card = n / 2

theorem valid_numbers : {n : ℕ | is_valid_number n} = {8, 12} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2073_207364


namespace NUMINAMATH_CALUDE_area_relation_l2073_207353

-- Define the triangles
structure Triangle :=
  (O A B : ℝ × ℝ)

-- Define properties of isosceles right triangles
def IsIsoscelesRight (t : Triangle) : Prop :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xO)^2 + (yB - yO)^2 ∧
  (xB - xA)^2 + (yB - yA)^2 = 2 * ((xA - xO)^2 + (yA - yO)^2)

-- Define the area of a triangle
def Area (t : Triangle) : ℝ :=
  let (xO, yO) := t.O
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  0.5 * abs ((xA - xO) * (yB - yO) - (xB - xO) * (yA - yO))

-- Theorem statement
theorem area_relation (OAB OBC OCD : Triangle) :
  IsIsoscelesRight OAB ∧ IsIsoscelesRight OBC ∧ IsIsoscelesRight OCD →
  Area OCD = 12 →
  Area OAB = 3 :=
by sorry

end NUMINAMATH_CALUDE_area_relation_l2073_207353


namespace NUMINAMATH_CALUDE_john_david_pushup_difference_l2073_207350

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did more than Zachary -/
def david_extra_pushups : ℕ := 39

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 58

/-- The number of push-ups done by John -/
def john_pushups : ℕ := david_pushups

theorem john_david_pushup_difference :
  david_pushups - john_pushups = 0 :=
sorry

end NUMINAMATH_CALUDE_john_david_pushup_difference_l2073_207350


namespace NUMINAMATH_CALUDE_rebecca_hours_l2073_207307

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_hours :
  ∀ x : ℕ,
  (x + (2*x - 10) + (2*x - 18) = 157) →
  (2*x - 18 = 56) :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_hours_l2073_207307


namespace NUMINAMATH_CALUDE_transaction_result_l2073_207330

theorem transaction_result : 
  ∀ (house_cost store_cost : ℕ),
  (house_cost * 3 / 4 = 15000) →
  (store_cost * 5 / 4 = 10000) →
  (house_cost + store_cost) - (15000 + 10000) = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_transaction_result_l2073_207330


namespace NUMINAMATH_CALUDE_divisor_calculation_l2073_207384

theorem divisor_calculation (dividend quotient remainder : ℚ) :
  dividend = 13/3 →
  quotient = -61 →
  remainder = -19 →
  ∃ divisor : ℚ, dividend = divisor * quotient + remainder ∧ divisor = -70/183 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l2073_207384


namespace NUMINAMATH_CALUDE_karlson_candy_theorem_l2073_207319

/-- The maximum number of candies Karlson can eat given n initial units -/
def max_candies (n : ℕ) : ℕ := Nat.choose n 2

/-- The theorem stating that for 31 initial units, the maximum number of candies is 465 -/
theorem karlson_candy_theorem :
  max_candies 31 = 465 := by
  sorry

end NUMINAMATH_CALUDE_karlson_candy_theorem_l2073_207319


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2073_207320

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2073_207320


namespace NUMINAMATH_CALUDE_division_remainder_l2073_207396

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 149 →
  divisor = 16 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2073_207396


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2073_207394

theorem quadratic_roots_relation (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 7 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 7 = 0) → 
  (x₁ + x₂ = 5/3) ∧ (x₁ * x₂ = -7/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2073_207394


namespace NUMINAMATH_CALUDE_triangle_properties_l2073_207370

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) : 
  (t.a * Real.cos t.C + (t.c - 3 * t.b) * Real.cos t.A = 0) → 
  (Real.cos t.A = 1 / 3) ∧
  (Real.sqrt 2 = 1 / 2 * t.b * t.c * Real.sin t.A) →
  (t.b - t.c = 2) →
  (t.a = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2073_207370


namespace NUMINAMATH_CALUDE_factors_of_36_l2073_207326

/-- The number of distinct positive factors of 36 is 9. -/
theorem factors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_36_l2073_207326


namespace NUMINAMATH_CALUDE_internet_price_difference_l2073_207300

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- in Mbps
  price : ℕ  -- in dollars

/-- The problem setup -/
def internetProblem : Prop :=
  ∃ (current twentyMbps thirtyMbps : InternetService),
    -- Current service
    current.speed = 10 ∧ current.price = 20 ∧
    -- 30 Mbps service
    thirtyMbps.speed = 30 ∧ thirtyMbps.price = 2 * current.price ∧
    -- 20 Mbps service
    twentyMbps.speed = 20 ∧ twentyMbps.price > current.price ∧
    -- Yearly savings
    (thirtyMbps.price - twentyMbps.price) * 12 = 120 ∧
    -- The statement to prove
    twentyMbps.price = current.price + 10

theorem internet_price_difference :
  internetProblem :=
sorry

end NUMINAMATH_CALUDE_internet_price_difference_l2073_207300


namespace NUMINAMATH_CALUDE_exists_special_sequence_l2073_207373

/-- An integer sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) (m : ℕ) : Prop :=
  (a 0 = 1) ∧ 
  (a 1 = 337) ∧ 
  (∀ n : ℕ, n ≥ 1 → (a (n+1) * a (n-1) - a n ^ 2) + 3 * (a (n+1) + a (n-1) - 2 * a n) / 4 = m) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n + 1) * (2 * a n + 1) / 6 = k ^ 2)

/-- Theorem stating the existence of a natural number m and a sequence satisfying the conditions -/
theorem exists_special_sequence : ∃ (m : ℕ) (a : ℕ → ℤ), SpecialSequence a m := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l2073_207373


namespace NUMINAMATH_CALUDE_greatest_k_value_l2073_207314

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 5 = 0 ∧ 
    x₂^2 + k*x₂ + 5 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 61) →
  k ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l2073_207314


namespace NUMINAMATH_CALUDE_blue_balls_count_l2073_207379

theorem blue_balls_count (red_balls : ℕ) (blue_balls : ℕ) 
  (h1 : red_balls = 25)
  (h2 : red_balls = 2 * blue_balls + 3) :
  blue_balls = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2073_207379


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l2073_207361

theorem sin_alpha_minus_pi_third (α : Real) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 2 * Real.tan α * Real.sin α = 3) : 
  Real.sin (α - π/3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l2073_207361


namespace NUMINAMATH_CALUDE_sauce_correction_l2073_207382

theorem sauce_correction (x : ℝ) : 
  (0.4 * x - 1 + 2.5 = 0.6 * x - 1.5) → x = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sauce_correction_l2073_207382


namespace NUMINAMATH_CALUDE_no_solution_l2073_207366

theorem no_solution : ¬ ∃ (n : ℕ), (823435^15 % n = 0) ∧ (n^5 - n^n = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2073_207366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2073_207328

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 15 - a₅, then a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2073_207328


namespace NUMINAMATH_CALUDE_sachin_lending_rate_l2073_207308

/-- Calculates simple interest --/
def simpleInterest (principal time rate : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sachin_lending_rate :
  let borrowed_amount : ℚ := 5000
  let borrowed_time : ℚ := 2
  let borrowed_rate : ℚ := 4
  let sachin_gain_per_year : ℚ := 112.5
  let borrowed_interest := simpleInterest borrowed_amount borrowed_time borrowed_rate
  let total_gain := sachin_gain_per_year * borrowed_time
  let total_interest_from_rahul := borrowed_interest + total_gain
  let rahul_rate := (total_interest_from_rahul * 100) / (borrowed_amount * borrowed_time)
  rahul_rate = 6.25 := by sorry

end NUMINAMATH_CALUDE_sachin_lending_rate_l2073_207308


namespace NUMINAMATH_CALUDE_min_sum_inverse_ratio_l2073_207345

theorem min_sum_inverse_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / (3 * Real.rpow 2 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inverse_ratio_l2073_207345


namespace NUMINAMATH_CALUDE_bicycle_spokes_l2073_207389

/-- Represents a bicycle with front and back wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Calculates the total number of spokes on a bicycle -/
def total_spokes (b : Bicycle) : ℕ :=
  b.front_spokes + b.back_spokes

/-- Theorem: A bicycle with 20 front spokes and twice as many back spokes has 60 spokes in total -/
theorem bicycle_spokes :
  ∀ b : Bicycle, b.front_spokes = 20 ∧ b.back_spokes = 2 * b.front_spokes →
  total_spokes b = 60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spokes_l2073_207389


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2073_207316

theorem polynomial_division_theorem :
  let dividend : Polynomial ℚ := X^4 * 6 + X^3 * 9 - X^2 * 5 + X * 2 - 8
  let divisor : Polynomial ℚ := X * 3 + 4
  let quotient : Polynomial ℚ := X^3 * 2 - X^2 * 1 + X * 1 - 2
  let remainder : Polynomial ℚ := -8/3
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2073_207316


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2073_207303

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  ¬(∀ x : ℝ, x > 1 → x > 3) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2073_207303


namespace NUMINAMATH_CALUDE_julia_number_l2073_207312

theorem julia_number (j m : ℂ) : 
  j * m = 48 - 24*I → 
  m = 7 + 4*I → 
  j = 432/65 - 360/65*I := by sorry

end NUMINAMATH_CALUDE_julia_number_l2073_207312


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2073_207333

theorem board_cut_theorem (total_length shorter_length longer_length : ℝ) :
  total_length = 20 ∧
  total_length = shorter_length + longer_length ∧
  2 * shorter_length = longer_length + 4 →
  shorter_length = 8 := by
sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2073_207333


namespace NUMINAMATH_CALUDE_residue_negative_1237_mod_37_l2073_207310

theorem residue_negative_1237_mod_37 : ∃ k : ℤ, -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by sorry

end NUMINAMATH_CALUDE_residue_negative_1237_mod_37_l2073_207310


namespace NUMINAMATH_CALUDE_teresas_pencils_l2073_207363

/-- Teresa's pencil distribution problem -/
theorem teresas_pencils (colored_pencils black_pencils : ℕ) 
  (num_siblings pencils_per_sibling : ℕ) : 
  colored_pencils = 14 →
  black_pencils = 35 →
  num_siblings = 3 →
  pencils_per_sibling = 13 →
  colored_pencils + black_pencils - num_siblings * pencils_per_sibling = 10 :=
by sorry

end NUMINAMATH_CALUDE_teresas_pencils_l2073_207363


namespace NUMINAMATH_CALUDE_perpendicular_lines_solution_l2073_207337

theorem perpendicular_lines_solution (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + a^2 - 1 = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_solution_l2073_207337


namespace NUMINAMATH_CALUDE_albatrocity_to_finchester_distance_l2073_207301

/-- The distance from Albatrocity to Finchester in miles -/
def distance : ℝ := 75

/-- The speed of the pigeon in still air in miles per hour -/
def pigeon_speed : ℝ := 40

/-- The wind speed from Albatrocity to Finchester in miles per hour -/
def wind_speed : ℝ := 10

/-- The time for a round trip without wind in hours -/
def no_wind_time : ℝ := 3.75

/-- The time for a round trip with wind in hours -/
def wind_time : ℝ := 4

theorem albatrocity_to_finchester_distance :
  (2 * distance / pigeon_speed = no_wind_time) ∧
  (distance / (pigeon_speed + wind_speed) + distance / (pigeon_speed - wind_speed) = wind_time) →
  distance = 75 := by sorry

end NUMINAMATH_CALUDE_albatrocity_to_finchester_distance_l2073_207301


namespace NUMINAMATH_CALUDE_cookies_left_l2073_207387

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l2073_207387


namespace NUMINAMATH_CALUDE_evans_needed_amount_l2073_207336

/-- The amount Evan still needs to buy the watch -/
def amount_needed (david_found : ℕ) (evan_initial : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - (evan_initial + david_found)

/-- Theorem stating the amount Evan still needs -/
theorem evans_needed_amount : amount_needed 12 1 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_evans_needed_amount_l2073_207336


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l2073_207355

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l2073_207355


namespace NUMINAMATH_CALUDE_average_increase_is_three_l2073_207385

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new inning -/
def averageIncrease (b : Batsman) (newRuns : ℕ) : ℚ :=
  let newAverage := (b.totalRuns + newRuns) / (b.innings + 1)
  newAverage - b.average

/-- The theorem to be proved -/
theorem average_increase_is_three :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.totalRuns + 84) / 17 = 36 →
    averageIncrease b 84 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_three_l2073_207385


namespace NUMINAMATH_CALUDE_graphing_to_scientific_ratio_l2073_207365

/-- Represents the cost of calculators and the transaction details -/
structure CalculatorPurchase where
  basic_cost : ℝ
  scientific_cost : ℝ
  graphing_cost : ℝ
  total_spent : ℝ

/-- The conditions of the calculator purchase problem -/
def calculator_problem : CalculatorPurchase :=
  { basic_cost := 8
  , scientific_cost := 16
  , graphing_cost := 72 - 8 - 16
  , total_spent := 100 - 28 }

/-- Theorem stating that the ratio of graphing to scientific calculator cost is 3:1 -/
theorem graphing_to_scientific_ratio :
  calculator_problem.graphing_cost / calculator_problem.scientific_cost = 3 := by
  sorry


end NUMINAMATH_CALUDE_graphing_to_scientific_ratio_l2073_207365


namespace NUMINAMATH_CALUDE_ellipse_distance_sum_constant_l2073_207304

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope m passing through point P -/
structure Line where
  m : ℝ
  P : Point

theorem ellipse_distance_sum_constant
  (C : Ellipse)
  (h_ecc : C.a^2 - C.b^2 = (C.a / 2)^2) -- eccentricity is 1/2
  (h_chord : 2 * C.b^2 / C.a = 3) -- chord length condition
  (P : Point)
  (h_P_on_axis : P.y = 0 ∧ P.x^2 ≤ C.a^2) -- P is on the major axis
  (l : Line)
  (h_l_slope : l.m = C.b / C.a) -- line l has slope b/a
  (h_l_through_P : l.P = P) -- line l passes through P
  (A B : Point)
  (h_A_on_C : A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1) -- A is on ellipse C
  (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1) -- B is on ellipse C
  (h_A_on_l : A.y = l.m * (A.x - P.x)) -- A is on line l
  (h_B_on_l : B.y = l.m * (B.x - P.x)) -- B is on line l
  : (A.x - P.x)^2 + (A.y - P.y)^2 + (B.x - P.x)^2 + (B.y - P.y)^2 = C.a^2 + C.b^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_distance_sum_constant_l2073_207304


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2073_207376

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ ^ 6 / a ^ 2) + (Real.cos θ ^ 6 / b ^ 2) = 1 / (a + b)) :
  (Real.sin θ ^ 12 / a ^ 5) + (Real.cos θ ^ 12 / b ^ 5) = 1 / (a + b) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2073_207376


namespace NUMINAMATH_CALUDE_yeast_growth_proof_l2073_207335

def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (time / interval)

theorem yeast_growth_proof (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) :
  initial_population = 50 →
  growth_factor = 3 →
  interval = 5 →
  time = 20 →
  yeast_population initial_population growth_factor interval time = 4050 :=
by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_proof_l2073_207335


namespace NUMINAMATH_CALUDE_sqrt_of_point_zero_one_l2073_207331

theorem sqrt_of_point_zero_one : Real.sqrt 0.01 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_point_zero_one_l2073_207331


namespace NUMINAMATH_CALUDE_peach_problem_l2073_207341

theorem peach_problem (steven jake jill hanna lucy : ℕ) : 
  steven = 19 →
  jake = steven - 12 →
  jake = 3 * jill →
  hanna = jake + 3 →
  lucy = hanna + 5 →
  lucy + jill = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_problem_l2073_207341


namespace NUMINAMATH_CALUDE_population_doubling_time_l2073_207357

/-- The annual birth rate per 1000 people -/
def birth_rate : ℝ := 39.4

/-- The annual death rate per 1000 people -/
def death_rate : ℝ := 19.4

/-- The number of years for the population to double -/
def doubling_time : ℝ := 35

/-- Theorem stating that given the birth and death rates, the population will double in 35 years -/
theorem population_doubling_time :
  let net_growth_rate := birth_rate - death_rate
  let percentage_growth_rate := net_growth_rate / 10  -- Converted to percentage
  70 / percentage_growth_rate = doubling_time := by sorry

end NUMINAMATH_CALUDE_population_doubling_time_l2073_207357


namespace NUMINAMATH_CALUDE_essay_completion_time_l2073_207372

-- Define the essay parameters
def essay_length : ℕ := 1200
def initial_speed : ℕ := 400
def initial_duration : ℕ := 2
def subsequent_speed : ℕ := 200

-- Theorem statement
theorem essay_completion_time :
  let initial_words := initial_speed * initial_duration
  let remaining_words := essay_length - initial_words
  let subsequent_duration := remaining_words / subsequent_speed
  initial_duration + subsequent_duration = 4 := by
  sorry

end NUMINAMATH_CALUDE_essay_completion_time_l2073_207372


namespace NUMINAMATH_CALUDE_perfect_squares_among_options_l2073_207339

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_squares_among_options :
  (is_perfect_square (3^3 * 4^5 * 7^7) = false) ∧
  (is_perfect_square (3^4 * 4^4 * 7^6) = true) ∧
  (is_perfect_square (3^6 * 4^3 * 7^8) = true) ∧
  (is_perfect_square (3^5 * 4^6 * 7^5) = false) ∧
  (is_perfect_square (3^4 * 4^6 * 7^7) = false) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_among_options_l2073_207339


namespace NUMINAMATH_CALUDE_book_cost_l2073_207368

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one := cost_of_three / 3
  8 * cost_of_one = 120 := by sorry

end NUMINAMATH_CALUDE_book_cost_l2073_207368


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l2073_207367

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Theorem statement
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  (∀ i j : Fin 4, q.sides i = q.sides j) → is_rhombus q :=
by
  sorry


end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l2073_207367


namespace NUMINAMATH_CALUDE_factorial_simplification_l2073_207354

theorem factorial_simplification : (15 : ℕ).factorial / ((12 : ℕ).factorial + 3 * (11 : ℕ).factorial) = 4680 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2073_207354


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2073_207399

theorem integer_solutions_of_equation : 
  ∀ x y : ℤ, x^2 - x*y - 6*y^2 + 2*x + 19*y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2073_207399


namespace NUMINAMATH_CALUDE_train_bridge_problem_l2073_207381

/-- Given a train crossing a bridge, this theorem proves the length and speed of the train. -/
theorem train_bridge_problem (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) 
  (h1 : bridge_length = 1000)
  (h2 : total_time = 60)
  (h3 : on_bridge_time = 40) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = 200 ∧ 
    train_speed = 20 ∧
    (bridge_length + train_length) / total_time = (bridge_length - train_length) / on_bridge_time :=
by
  sorry

#check train_bridge_problem

end NUMINAMATH_CALUDE_train_bridge_problem_l2073_207381


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2073_207309

theorem quadratic_equation_solution :
  let a : ℝ := 5
  let b : ℝ := -2 * Real.sqrt 15
  let c : ℝ := -2
  let x₁ : ℝ := -1 + Real.sqrt 15 / 5
  let x₂ : ℝ := 1 + Real.sqrt 15 / 5
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2073_207309


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_sum_l2073_207378

/-- The sum of the coefficients of the last three terms in the binomial expansion -/
def sum_of_last_three_coefficients (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- The theorem stating that if the sum of the coefficients of the last three terms 
    in the expansion of (√x + 2/√x)^n is 79, then n = 12 -/
theorem binomial_expansion_coefficient_sum (n : ℕ) : 
  sum_of_last_three_coefficients n = 79 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_sum_l2073_207378


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l2073_207349

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 5 / 8 ∧ 
  (∀ (r s : ℕ+), (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < 5 / 8 → s ≥ q) →
  q - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l2073_207349


namespace NUMINAMATH_CALUDE_sqrt_12_plus_sqrt_27_l2073_207362

theorem sqrt_12_plus_sqrt_27 : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_sqrt_27_l2073_207362


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l2073_207321

theorem t_shirts_per_package (total_t_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_t_shirts = 39)
  (h2 : num_packages = 3)
  (h3 : total_t_shirts % num_packages = 0) :
  total_t_shirts / num_packages = 13 := by
  sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l2073_207321


namespace NUMINAMATH_CALUDE_marbles_distribution_l2073_207332

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 20 →
  num_boys = 2 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2073_207332


namespace NUMINAMATH_CALUDE_tangent_alpha_equals_four_l2073_207375

theorem tangent_alpha_equals_four (α : Real) 
  (h : 3 * Real.tan α - Real.sin α + 4 * Real.cos α = 12) : 
  Real.tan α = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_alpha_equals_four_l2073_207375


namespace NUMINAMATH_CALUDE_cell_phone_price_l2073_207306

/-- The price of a cell phone given the total cost and monthly payments --/
theorem cell_phone_price (total_cost : ℕ) (monthly_payment : ℕ) (num_months : ℕ) 
  (h1 : total_cost = 30)
  (h2 : monthly_payment = 7)
  (h3 : num_months = 4) :
  total_cost - (monthly_payment * num_months) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_price_l2073_207306


namespace NUMINAMATH_CALUDE_bread_baking_time_l2073_207329

theorem bread_baking_time (rise_time bake_time : ℕ) (num_balls : ℕ) : 
  rise_time = 3 → 
  bake_time = 2 → 
  num_balls = 4 → 
  (rise_time * num_balls) + (bake_time * num_balls) = 20 :=
by sorry

end NUMINAMATH_CALUDE_bread_baking_time_l2073_207329


namespace NUMINAMATH_CALUDE_team_selection_count_l2073_207374

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 10 boys and 12 girls. -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total_boys (team_size / 2) * Nat.choose total_girls (team_size / 2)

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_count :
  select_team 10 12 8 = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2073_207374


namespace NUMINAMATH_CALUDE_sqrt_identity_l2073_207369

theorem sqrt_identity (θ : Real) (h : θ = 40 * π / 180) :
  Real.sqrt (16 - 12 * Real.sin θ) = 4 + Real.sqrt 3 * (1 / Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l2073_207369


namespace NUMINAMATH_CALUDE_gcd_12740_220_minus_10_l2073_207346

theorem gcd_12740_220_minus_10 : Nat.gcd 12740 220 - 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12740_220_minus_10_l2073_207346


namespace NUMINAMATH_CALUDE_log_sum_adjacent_terms_l2073_207325

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem log_sum_adjacent_terms 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a5 : a 5 = 10) : 
  Real.log (a 4) + Real.log (a 6) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_adjacent_terms_l2073_207325


namespace NUMINAMATH_CALUDE_root_of_unity_sum_iff_cube_root_l2073_207327

theorem root_of_unity_sum_iff_cube_root (x y : ℂ) : 
  (Complex.abs x = 1 ∧ Complex.abs y = 1 ∧ x ≠ y) → 
  (Complex.abs (x + y) = 1 ↔ (y / x) ^ 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_root_of_unity_sum_iff_cube_root_l2073_207327


namespace NUMINAMATH_CALUDE_replacement_sugar_percentage_l2073_207340

/-- Represents a sugar solution with a given weight and sugar percentage -/
structure SugarSolution where
  weight : ℝ
  sugarPercentage : ℝ

/-- Calculates the amount of sugar in a solution -/
def sugarAmount (solution : SugarSolution) : ℝ :=
  solution.weight * solution.sugarPercentage

theorem replacement_sugar_percentage
  (original : SugarSolution)
  (replacement : SugarSolution)
  (final : SugarSolution)
  (h1 : original.sugarPercentage = 0.10)
  (h2 : final.sugarPercentage = 0.14)
  (h3 : final.weight = original.weight)
  (h4 : replacement.weight = original.weight / 4)
  (h5 : sugarAmount final = sugarAmount original - sugarAmount original / 4 + sugarAmount replacement) :
  replacement.sugarPercentage = 0.26 := by
sorry

end NUMINAMATH_CALUDE_replacement_sugar_percentage_l2073_207340


namespace NUMINAMATH_CALUDE_cats_remaining_l2073_207359

def siamese_cats : ℕ := 38
def house_cats : ℕ := 25
def cats_sold : ℕ := 45

theorem cats_remaining : siamese_cats + house_cats - cats_sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l2073_207359


namespace NUMINAMATH_CALUDE_task_completion_correct_l2073_207352

/-- Represents the number of days it takes for a person to complete the task alone -/
structure PersonWorkRate where
  days : ℝ
  days_positive : days > 0

/-- Represents the scenario of two people working on a task -/
structure WorkScenario where
  person_a : PersonWorkRate
  person_b : PersonWorkRate
  days_a_alone : ℝ
  days_together : ℝ
  days_a_alone_nonnegative : days_a_alone ≥ 0
  days_together_nonnegative : days_together ≥ 0

/-- The equation representing the completion of the task -/
def task_completion_equation (scenario : WorkScenario) : Prop :=
  (scenario.days_together + scenario.days_a_alone) / scenario.person_a.days +
  scenario.days_together / scenario.person_b.days = 1

/-- The theorem stating that the given equation correctly represents the completion of the task -/
theorem task_completion_correct (scenario : WorkScenario)
  (h1 : scenario.person_a.days = 3)
  (h2 : scenario.person_b.days = 5)
  (h3 : scenario.days_a_alone = 1) :
  task_completion_equation scenario :=
sorry

end NUMINAMATH_CALUDE_task_completion_correct_l2073_207352


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2012_l2073_207395

/-- Given an angle α = 2012°, this theorem proves that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_2012 (α : Real) (h : α = 2012) :
  ∃ (θ : Real), 0 < θ ∧ θ ≤ 360 ∧ θ = α % 360 ∧ θ = 212 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2012_l2073_207395


namespace NUMINAMATH_CALUDE_factorization_equality_minimum_value_minimum_achieved_l2073_207380

-- Problem 1
theorem factorization_equality (m n : ℝ) : 
  m^2 - 4*m*n + 3*n^2 = (m - 3*n) * (m - n) := by sorry

-- Problem 2
theorem minimum_value (m : ℝ) : 
  m^2 - 3*m + 2015 ≥ 2012 + 3/4 := by sorry

-- The minimum is achievable
theorem minimum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ m : ℝ, m^2 - 3*m + 2015 < 2012 + 3/4 + ε := by sorry

end NUMINAMATH_CALUDE_factorization_equality_minimum_value_minimum_achieved_l2073_207380


namespace NUMINAMATH_CALUDE_target_probabilities_l2073_207313

/-- Probability of hitting a target -/
structure TargetProbability where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Assumptions about the probabilities -/
axiom prob_bounds (p : TargetProbability) :
  0 ≤ p.A ∧ p.A ≤ 1 ∧
  0 ≤ p.B ∧ p.B ≤ 1 ∧
  0 ≤ p.C ∧ p.C ≤ 1

/-- Given probabilities -/
def given_probs : TargetProbability :=
  { A := 0.7, B := 0.6, C := 0.5 }

/-- Probability of at least one person hitting the target -/
def prob_at_least_one (p : TargetProbability) : ℝ :=
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C)

/-- Probability of exactly two people hitting the target -/
def prob_exactly_two (p : TargetProbability) : ℝ :=
  p.A * p.B * (1 - p.C) + p.A * (1 - p.B) * p.C + (1 - p.A) * p.B * p.C

/-- Probability of hitting exactly k times in n trials -/
def prob_k_of_n (p q : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * q^(n - k)

theorem target_probabilities (p : TargetProbability) 
  (h : p = given_probs) : 
  prob_at_least_one p = 0.94 ∧ 
  prob_exactly_two p = 0.44 ∧ 
  prob_k_of_n p.A (1 - p.A) 3 2 = 0.441 := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l2073_207313


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2073_207334

-- Define the sets A, B, and C
def A (x : ℝ) : Set ℝ := {2, -1, x^2 - x + 1}
def B (x y : ℝ) : Set ℝ := {2*y, -4, x + 4}
def C : Set ℝ := {-1}

-- State the theorem
theorem union_of_A_and_B (x y : ℝ) :
  (A x ∩ B x y = C) →
  (A x ∪ B x y = {2, -1, x^2 - x + 1, 2*y, -4, x + 4}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2073_207334


namespace NUMINAMATH_CALUDE_mikeys_leaves_theorem_l2073_207343

/-- Given an initial number of leaves and the remaining number of leaves,
    calculate the number of leaves that blew away. -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that for Mikey's specific case, 
    the number of leaves blown away is 244. -/
theorem mikeys_leaves_theorem :
  leaves_blown_away 356 112 = 244 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_theorem_l2073_207343


namespace NUMINAMATH_CALUDE_population_change_l2073_207323

theorem population_change (P : ℝ) : 
  P > 0 →
  (P * 1.25 * 0.75 = 18750) →
  P = 20000 := by
sorry

end NUMINAMATH_CALUDE_population_change_l2073_207323


namespace NUMINAMATH_CALUDE_jacob_younger_than_michael_l2073_207342

/-- Represents the age difference between Michael and Jacob -/
def age_difference (jacob_age michael_age : ℕ) : ℕ := michael_age - jacob_age

/-- Proves that Jacob is 14 years younger than Michael given the problem conditions -/
theorem jacob_younger_than_michael :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age < michael_age) →                        -- Jacob is younger than Michael
    (michael_age + 9 = 2 * (jacob_age + 9)) →          -- 9 years from now, Michael will be twice as old as Jacob
    (jacob_age + 4 = 9) →                              -- Jacob will be 9 years old in 4 years
    age_difference jacob_age michael_age = 14 :=        -- The age difference is 14 years
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_jacob_younger_than_michael_l2073_207342


namespace NUMINAMATH_CALUDE_root_in_interval_l2073_207371

def f (x : ℝ) := x^3 + x - 8

theorem root_in_interval :
  f 1 < 0 →
  f 1.5 < 0 →
  f 1.75 < 0 →
  f 2 > 0 →
  ∃ x, x ∈ Set.Ioo 1.75 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2073_207371


namespace NUMINAMATH_CALUDE_two_digit_integers_count_l2073_207348

def digits : List ℕ := [2, 3, 4, 7]
def tens_digits : List ℕ := [2, 3]
def units_digits : List ℕ := [4, 7]

theorem two_digit_integers_count : 
  (List.length tens_digits) * (List.length units_digits) = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_count_l2073_207348


namespace NUMINAMATH_CALUDE_find_k_l2073_207358

theorem find_k (k : ℚ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2073_207358


namespace NUMINAMATH_CALUDE_remainder_97_103_times_7_mod_17_l2073_207317

theorem remainder_97_103_times_7_mod_17 : (97^103 * 7) % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_103_times_7_mod_17_l2073_207317


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l2073_207386

theorem triangle_side_inequality (a b c : ℝ) (h_area : 1 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) (h_order : a ≤ b ∧ b ≤ c) : b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l2073_207386


namespace NUMINAMATH_CALUDE_inequality_proof_l2073_207397

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2073_207397


namespace NUMINAMATH_CALUDE_triangle_max_area_l2073_207351

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area is √3 when (a+b)(sin A - sin B) = (c-b)sin C and a = 2 -/
theorem triangle_max_area (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) →
  (a = 2) →
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2073_207351


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l2073_207338

def sumFunction (n : ℕ) : ℚ :=
  (n^4 - 1) / (n^4 + 1)

def sumRange : List ℕ := [2, 3, 4, 5]

theorem sum_equals_fraction :
  (sumRange.map sumFunction).sum = 21182880 / 349744361 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l2073_207338


namespace NUMINAMATH_CALUDE_unit_digit_4137_754_l2073_207360

theorem unit_digit_4137_754 : (4137^754) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_4137_754_l2073_207360


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l2073_207383

/-- Given a circle of radius r that touches two identical circles and a smaller circle,
    all externally tangent to each other, the radius of the smallest circle is r/6. -/
theorem smallest_circle_radius (r : ℝ) (hr : r > 0) : ∃ (r_small : ℝ), r_small = r / 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l2073_207383


namespace NUMINAMATH_CALUDE_modular_inverse_32_mod_33_l2073_207377

theorem modular_inverse_32_mod_33 : ∃ x : ℕ, x ≤ 32 ∧ (32 * x) % 33 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_32_mod_33_l2073_207377


namespace NUMINAMATH_CALUDE_exponential_inequality_l2073_207391

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2073_207391


namespace NUMINAMATH_CALUDE_theresa_chocolate_bars_double_kayla_l2073_207356

/-- Represents the number of items Kayla bought -/
structure KaylasItems where
  chocolateBars : ℕ
  sodaCans : ℕ
  total : ℕ
  total_eq : chocolateBars + sodaCans = total

/-- Represents the number of items Theresa bought -/
structure TheresasItems where
  chocolateBars : ℕ
  sodaCans : ℕ

/-- The given conditions of the problem -/
class ProblemConditions where
  kayla : KaylasItems
  theresa : TheresasItems
  kayla_total_15 : kayla.total = 15
  theresa_double_kayla : theresa.chocolateBars = 2 * kayla.chocolateBars ∧
                         theresa.sodaCans = 2 * kayla.sodaCans

theorem theresa_chocolate_bars_double_kayla
  [conditions : ProblemConditions] :
  conditions.theresa.chocolateBars = 2 * conditions.kayla.chocolateBars :=
by sorry

end NUMINAMATH_CALUDE_theresa_chocolate_bars_double_kayla_l2073_207356
