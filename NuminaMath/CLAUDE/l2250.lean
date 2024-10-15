import Mathlib

namespace NUMINAMATH_CALUDE_second_term_value_l2250_225001

theorem second_term_value (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A / B = 3 / 4) (h4 : (A + 10) / (B + 10) = 4 / 5) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_term_value_l2250_225001


namespace NUMINAMATH_CALUDE_total_pencils_count_l2250_225085

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def total_people : ℕ := 8

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_box * total_people

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l2250_225085


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2250_225054

/-- Represents the price reduction and resulting sales and profit changes for a toy product. -/
structure ToyPricing where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit after price reduction. -/
def dailyProfitAfterReduction (t : ToyPricing) : ℕ :=
  (t.initialProfit - t.priceReduction) * (t.initialSales + t.salesIncrease * t.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in the target daily profit. -/
theorem price_reduction_achieves_target_profit (t : ToyPricing) 
  (h1 : t.initialSales = 20)
  (h2 : t.initialProfit = 40)
  (h3 : t.salesIncrease = 2)
  (h4 : t.targetProfit = 1200)
  (h5 : t.priceReduction = 20) :
  dailyProfitAfterReduction t = t.targetProfit :=
by
  sorry

#eval dailyProfitAfterReduction { 
  initialSales := 20, 
  initialProfit := 40, 
  salesIncrease := 2, 
  priceReduction := 20, 
  targetProfit := 1200 
}

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2250_225054


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_25_l2250_225026

theorem least_product_of_primes_above_25 (p q : ℕ) : 
  p.Prime → q.Prime → p > 25 → q > 25 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 899 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 25 → s > 25 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_25_l2250_225026


namespace NUMINAMATH_CALUDE_combined_speed_difference_l2250_225059

-- Define the speed functions for each train
def zA (s : ℝ) : ℝ := s^2 + 2*s
def zB (s : ℝ) : ℝ := 2*s^2 + 3*s + 1
def zC (s : ℝ) : ℝ := s^3 - 4*s

-- Define the time constraints for each train
def trainA_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 7
def trainB_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 5
def trainC_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 4

-- Theorem statement
theorem combined_speed_difference :
  trainA_time_constraint 7 ∧
  trainA_time_constraint 2 ∧
  trainB_time_constraint 5 ∧
  trainB_time_constraint 2 ∧
  trainC_time_constraint 4 ∧
  trainC_time_constraint 2 →
  (zA 7 - zA 2) + (zB 5 - zB 2) + (zC 4 - zC 2) = 154 := by
  sorry

end NUMINAMATH_CALUDE_combined_speed_difference_l2250_225059


namespace NUMINAMATH_CALUDE_root_product_l2250_225091

theorem root_product (d e : ℤ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 → x^7 - d*x^3 - e = 0) → 
  d * e = 70 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l2250_225091


namespace NUMINAMATH_CALUDE_expression_values_l2250_225084

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l2250_225084


namespace NUMINAMATH_CALUDE_sunday_max_available_l2250_225033

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the team members
inductive Member
  | Alice
  | Bob
  | Cara
  | Dave
  | Ella

-- Define a function to represent the availability of each member on each day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cara, Day.Monday => false
  | Member.Cara, Day.Tuesday => false
  | Member.Cara, Day.Thursday => false
  | Member.Cara, Day.Saturday => false
  | Member.Cara, Day.Sunday => false
  | Member.Dave, Day.Wednesday => false
  | Member.Dave, Day.Saturday => false
  | Member.Ella, Day.Monday => false
  | Member.Ella, Day.Friday => false
  | Member.Ella, Day.Saturday => false
  | _, _ => true

-- Define a function to count the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cara, Member.Dave, Member.Ella]).length

-- Theorem: Sunday has the maximum number of available team members
theorem sunday_max_available :
  ∀ d : Day, countAvailable Day.Sunday ≥ countAvailable d := by
  sorry


end NUMINAMATH_CALUDE_sunday_max_available_l2250_225033


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l2250_225077

theorem sum_of_even_integers (a b c d : ℤ) 
  (h1 : Even a) (h2 : Even b) (h3 : Even c) (h4 : Even d)
  (eq1 : a - b + c = 8)
  (eq2 : b - c + d = 10)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 6) :
  a + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l2250_225077


namespace NUMINAMATH_CALUDE_min_floor_sum_l2250_225057

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l2250_225057


namespace NUMINAMATH_CALUDE_product_of_primes_even_l2250_225096

theorem product_of_primes_even (P Q : ℕ+) : 
  Prime P.val → Prime Q.val → Prime (P.val - Q.val) → Prime (P.val + Q.val) → 
  Even (P.val * Q.val * (P.val - Q.val) * (P.val + Q.val)) := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_even_l2250_225096


namespace NUMINAMATH_CALUDE_line_l_properties_l2250_225047

/-- A line passing through (-2, 1) with y-intercept twice the x-intercept -/
def line_l (x y : ℝ) : Prop := 2*x + y + 3 = 0

theorem line_l_properties :
  (∃ x y : ℝ, line_l x y ∧ x = -2 ∧ y = 1) ∧
  (∃ a : ℝ, a ≠ 0 → line_l a 0 ∧ line_l 0 (2*a)) :=
sorry

end NUMINAMATH_CALUDE_line_l_properties_l2250_225047


namespace NUMINAMATH_CALUDE_area_swept_by_small_square_l2250_225050

/-- The area swept by a small square sliding along three sides of a larger square -/
theorem area_swept_by_small_square (large_side small_side : ℝ) :
  large_side > 0 ∧ small_side > 0 ∧ large_side > small_side →
  let swept_area := large_side^2 - (large_side - 2*small_side)^2
  swept_area = 36 ∧ large_side = 10 ∧ small_side = 1 := by
  sorry

#check area_swept_by_small_square

end NUMINAMATH_CALUDE_area_swept_by_small_square_l2250_225050


namespace NUMINAMATH_CALUDE_systematic_sampling_seat_number_l2250_225025

/-- Systematic sampling function that returns the seat numbers in the sample -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

theorem systematic_sampling_seat_number
  (totalStudents : ℕ) (sampleSize : ℕ) (knownSeats : List ℕ) :
  totalStudents = 52 →
  sampleSize = 4 →
  knownSeats = [3, 29, 42] →
  let sample := systematicSample totalStudents sampleSize
  (∀ s ∈ knownSeats, s ∈ sample) →
  ∃ s ∈ sample, s = 16 ∧ s ∉ knownSeats :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_seat_number_l2250_225025


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2250_225067

theorem sufficient_not_necessary_condition :
  (∀ x > 0, x + (1/18) / (2*x) ≥ 1/3) ∧
  (∃ a ≠ 1/18, ∀ x > 0, x + a / (2*x) ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2250_225067


namespace NUMINAMATH_CALUDE_celyna_candy_purchase_l2250_225076

/-- Prove that given the conditions of Celyna's candy purchase, the amount of candy B is 500 grams -/
theorem celyna_candy_purchase (candy_a_weight : ℝ) (candy_a_cost : ℝ) (candy_b_cost : ℝ) (average_price : ℝ) :
  candy_a_weight = 300 →
  candy_a_cost = 5 →
  candy_b_cost = 7 →
  average_price = 1.5 →
  ∃ x : ℝ, x = 500 ∧ 
    (candy_a_cost + candy_b_cost) / ((candy_a_weight + x) / 100) = average_price :=
by sorry

end NUMINAMATH_CALUDE_celyna_candy_purchase_l2250_225076


namespace NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l2250_225094

/-- The time Nadine spends cleaning her dog -/
def dog_cleaning_time (hosing_time shampoo_time shampoo_count : ℕ) : ℕ :=
  hosing_time + shampoo_time * shampoo_count

/-- Theorem stating the total time Nadine spends cleaning her dog -/
theorem nadine_dog_cleaning_time :
  dog_cleaning_time 10 15 3 = 55 := by
  sorry

end NUMINAMATH_CALUDE_nadine_dog_cleaning_time_l2250_225094


namespace NUMINAMATH_CALUDE_johns_local_taxes_l2250_225043

/-- Proves that given John's hourly wage and local tax rate, the amount of local taxes paid in cents per hour is 60 cents. -/
theorem johns_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_local_taxes_l2250_225043


namespace NUMINAMATH_CALUDE_circle_area_doubling_l2250_225037

theorem circle_area_doubling (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 2 * π * r^2) → (r = n * (Real.sqrt 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_doubling_l2250_225037


namespace NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_thirty_sixth_l2250_225013

/-- The probability of rolling a specific number on a fair 6-sided die -/
def prob_single : ℚ := 1 / 6

/-- The set of numbers on a 6-sided die -/
def die_numbers : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of numbers on a 6-sided die that are divisible by 4 -/
def divisible_by_four : Set ℕ := {n ∈ die_numbers | n % 4 = 0}

/-- The probability that both dice show numbers divisible by 4 -/
def prob_both_divisible_by_four : ℚ := prob_single * prob_single

theorem prob_both_divisible_by_four_is_one_thirty_sixth :
  prob_both_divisible_by_four = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_thirty_sixth_l2250_225013


namespace NUMINAMATH_CALUDE_rectangle_thirteen_squares_l2250_225074

/-- A rectangle can be divided into 13 equal squares if and only if its side ratio is 13:1 or 1:13 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ (a = s ∧ b = 13 * s ∨ a = 13 * s ∧ b = s)) ↔
  (a * b = 13 * (a.min b) * (a.min b)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_thirteen_squares_l2250_225074


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2250_225080

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔ 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2250_225080


namespace NUMINAMATH_CALUDE_smallest_constant_l2250_225093

-- Define the properties of the function f
def FunctionProperties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 0 = 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Theorem statement
theorem smallest_constant (f : ℝ → ℝ) (h : FunctionProperties f) :
  (∃ c > 0, ∀ x ∈ Set.Icc 0 1, f x ≤ c * x) ∧
  (∀ c < 2, ∃ x ∈ Set.Icc 0 1, f x > c * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_l2250_225093


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l2250_225062

theorem chord_length_in_circle (r d c : ℝ) (hr : r = 3) (hd : d = 2) :
  r^2 = d^2 + (c/2)^2 → c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l2250_225062


namespace NUMINAMATH_CALUDE_intersection_equals_T_l2250_225014

-- Define set S
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}

-- Define set T
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Theorem statement
theorem intersection_equals_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l2250_225014


namespace NUMINAMATH_CALUDE_ages_sum_l2250_225086

theorem ages_sum (diane_future_age diane_current_age : ℕ) 
  (h1 : diane_future_age = 30)
  (h2 : diane_current_age = 16) : ∃ (alex_age allison_age : ℕ), 
  (diane_future_age = alex_age / 2) ∧ 
  (diane_future_age = allison_age * 2) ∧
  (alex_age + allison_age = 47) :=
by sorry

end NUMINAMATH_CALUDE_ages_sum_l2250_225086


namespace NUMINAMATH_CALUDE_parabola_point_and_line_intersection_l2250_225024

/-- Parabola C defined by y^2 = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point P on the parabola C -/
def P : ℝ × ℝ := (1, 2)

/-- Point Q symmetrical to P across the x-axis -/
def Q : ℝ × ℝ := (P.1, -P.2)

/-- Origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of triangle POQ -/
def area_POQ : ℝ := 2

/-- Slopes of lines PA and PB -/
def k₁ : ℝ := sorry
def k₂ : ℝ := sorry

/-- Fixed point that AB passes through -/
def fixed_point : ℝ × ℝ := (0, -2)

theorem parabola_point_and_line_intersection :
  (P ∈ C) ∧
  (P.2 > 0) ∧
  (area_POQ = 2) ∧
  (k₁ * k₂ = 4) →
  (P = (1, 2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ C → B ∈ C →
    (A.2 - P.2) / (A.1 - P.1) = k₁ →
    (B.2 - P.2) / (B.1 - P.1) = k₂ →
    ∃ (m b : ℝ), (A.2 = m * A.1 + b) ∧ (B.2 = m * B.1 + b) ∧
    (fixed_point.2 = m * fixed_point.1 + b)) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_and_line_intersection_l2250_225024


namespace NUMINAMATH_CALUDE_factorization_equality_l2250_225016

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2250_225016


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2250_225022

/-- The value of m for a hyperbola with given equation and asymptote form -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (y^2 / 16 - x^2 / 9 = 1) →
  (∃ (m : ℝ), ∀ (x y : ℝ), y = m * x ∨ y = -m * x) →
  (∃ (m : ℝ), m = 4/3 ∧ (∀ (x y : ℝ), y = m * x ∨ y = -m * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2250_225022


namespace NUMINAMATH_CALUDE_divisibility_relation_l2250_225088

theorem divisibility_relation (p a b n : ℕ) : 
  p ≥ 3 → 
  Nat.Prime p → 
  Nat.Coprime a b → 
  p ∣ (a^(2^n) + b^(2^n)) → 
  2^(n+1) ∣ (p-1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2250_225088


namespace NUMINAMATH_CALUDE_extreme_values_and_inequality_l2250_225095

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + 1) / Real.exp x

theorem extreme_values_and_inequality (m : ℝ) (h₁ : m ≥ 0) :
  (m > 0 → (∃ (min_x max_x : ℝ), min_x = 1 - m ∧ max_x = 1 ∧
    ∀ x, f m x ≥ f m min_x ∧ f m x ≤ f m max_x)) ∧
  (m ∈ Set.Ioo 1 2 → ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 1 m → x₂ ∈ Set.Icc 1 m →
    f m x₁ > -x₂ + 1 + 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_inequality_l2250_225095


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2250_225015

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- If a complex number z satisfies (1-i)z = 2i, then z is in the second quadrant -/
theorem z_in_second_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  in_second_quadrant z := by
  sorry


end NUMINAMATH_CALUDE_z_in_second_quadrant_l2250_225015


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2250_225032

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2250_225032


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2250_225031

theorem tan_alpha_value (h : Real.tan (π - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2250_225031


namespace NUMINAMATH_CALUDE_pharmacy_service_l2250_225070

/-- The number of customers served by three workers in a day -/
def customers_served (regular_hours work_rate reduced_hours : ℕ) : ℕ :=
  work_rate * (2 * regular_hours + reduced_hours)

/-- Theorem: Given the specific conditions, the total number of customers served is 154 -/
theorem pharmacy_service : customers_served 8 7 6 = 154 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_service_l2250_225070


namespace NUMINAMATH_CALUDE_positive_expression_l2250_225078

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2250_225078


namespace NUMINAMATH_CALUDE_hour_hand_angle_after_one_hour_l2250_225035

/-- Represents the angle turned by the hour hand of a watch. -/
def angle_turned (hours : ℝ) : ℝ :=
  -30 * hours

/-- The theorem states that the angle turned by the hour hand after 1 hour is -30°. -/
theorem hour_hand_angle_after_one_hour :
  angle_turned 1 = -30 := by sorry

end NUMINAMATH_CALUDE_hour_hand_angle_after_one_hour_l2250_225035


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2250_225040

theorem cookie_jar_problem (initial_cookies : ℕ) (x : ℕ) 
  (h1 : initial_cookies = 7)
  (h2 : initial_cookies - 1 = (initial_cookies + x) / 2) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2250_225040


namespace NUMINAMATH_CALUDE_contest_questions_l2250_225099

theorem contest_questions (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (a b c : ℕ), 10 < a ∧ a ≤ b ∧ b ≤ c ∧ c < 13) 
  (h3 : 4 * n = 10 + 13 + a + b + c) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_contest_questions_l2250_225099


namespace NUMINAMATH_CALUDE_tan_10pi_minus_theta_l2250_225063

theorem tan_10pi_minus_theta (θ : Real) (h1 : π < θ) (h2 : θ < 2*π) 
  (h3 : Real.cos (θ - 9*π) = -3/5) : Real.tan (10*π - θ) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_10pi_minus_theta_l2250_225063


namespace NUMINAMATH_CALUDE_abs_m_minus_n_eq_two_sqrt_three_l2250_225048

theorem abs_m_minus_n_eq_two_sqrt_three (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_eq_two_sqrt_three_l2250_225048


namespace NUMINAMATH_CALUDE_sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l2250_225049

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem sum_squares_two_odds_not_perfect_square (a b : ℤ) (ha : is_odd a) (hb : is_odd b) :
  ¬∃ n : ℤ, a^2 + b^2 = n^2 := by sorry

theorem sum_squares_three_odds_not_perfect_square (a b c : ℤ) (ha : is_odd a) (hb : is_odd b) (hc : is_odd c) :
  ¬∃ m : ℤ, a^2 + b^2 + c^2 = m^2 := by sorry

end NUMINAMATH_CALUDE_sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l2250_225049


namespace NUMINAMATH_CALUDE_sphere_and_cube_l2250_225009

/-- Given a sphere with surface area 256π cm² circumscribed around a cube,
    prove its volume and the cube's side length. -/
theorem sphere_and_cube (S : Real) (r : Real) (V : Real) (s : Real) : 
  S = 256 * Real.pi → -- Surface area of the sphere
  S = 4 * Real.pi * r^2 → -- Surface area formula
  V = (4/3) * Real.pi * r^3 → -- Volume formula for sphere
  2 * r = s * Real.sqrt 3 → -- Relation between sphere diameter and cube diagonal
  V = (2048/3) * Real.pi ∧ s = (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_and_cube_l2250_225009


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_twelve_l2250_225038

theorem ab_plus_cd_equals_twelve 
  (a b c d : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_twelve_l2250_225038


namespace NUMINAMATH_CALUDE_expression_simplification_l2250_225039

theorem expression_simplification : 
  ((3 + 4 + 5 + 6)^2 / 4) + ((3 * 6 + 9)^2 / 3) = 324 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2250_225039


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l2250_225045

/-- A hexagon is a polygon with 6 sides. -/
def Hexagon : Type := Unit

/-- The number of sides in a hexagon. -/
def num_sides (h : Hexagon) : ℕ := 6

/-- The number of diagonals in a polygon. -/
def num_diagonals (h : Hexagon) : ℕ := sorry

/-- Theorem: The number of diagonals in a hexagon is 9. -/
theorem hexagon_diagonals (h : Hexagon) : num_diagonals h = 9 := by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l2250_225045


namespace NUMINAMATH_CALUDE_inequality_range_l2250_225012

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4*a) ↔ -1 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2250_225012


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2250_225003

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_one_var f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2250_225003


namespace NUMINAMATH_CALUDE_number_puzzle_l2250_225087

theorem number_puzzle : ∃ N : ℚ, N = 90 ∧ 3 + (1/2) * (1/3) * (1/5) * N = (1/15) * N := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2250_225087


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2250_225090

-- Equation 1
theorem equation_one_solution (x : ℚ) :
  x / (x - 1) = 3 / (2 * x - 2) - 2 ↔ x = 7 / 6 :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ (x : ℚ), (5 * x + 2) / (x^2 + x) = 3 / (x + 1) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2250_225090


namespace NUMINAMATH_CALUDE_messages_per_member_per_day_l2250_225044

theorem messages_per_member_per_day :
  let initial_members : ℕ := 150
  let removed_members : ℕ := 20
  let remaining_members : ℕ := initial_members - removed_members
  let total_weekly_messages : ℕ := 45500
  let messages_per_day : ℕ := total_weekly_messages / 7
  let messages_per_member_per_day : ℕ := messages_per_day / remaining_members
  messages_per_member_per_day = 50 :=
by sorry

end NUMINAMATH_CALUDE_messages_per_member_per_day_l2250_225044


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2250_225004

theorem inequality_equivalence (y : ℝ) : 
  (7 / 30 + |y - 19 / 60| < 17 / 30) ↔ (-1 / 60 < y ∧ y < 13 / 20) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2250_225004


namespace NUMINAMATH_CALUDE_grading_multiple_proof_l2250_225036

/-- Given a grading method that subtracts a multiple of incorrect responses
    from correct responses, prove that the multiple is 2 for a specific case. -/
theorem grading_multiple_proof (total_questions : ℕ) (correct_responses : ℕ) (score : ℕ) :
  total_questions = 100 →
  correct_responses = 87 →
  score = 61 →
  ∃ (m : ℚ), score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_grading_multiple_proof_l2250_225036


namespace NUMINAMATH_CALUDE_percentage_problem_l2250_225097

theorem percentage_problem (P : ℝ) : P = 20 → 0.25 * 1280 = (P / 100) * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2250_225097


namespace NUMINAMATH_CALUDE_fraction_expression_value_l2250_225083

theorem fraction_expression_value (m n p : ℝ) (h : m + n - p = 0) :
  m * (1 / n - 1 / p) + n * (1 / m - 1 / p) - p * (1 / m + 1 / n) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_value_l2250_225083


namespace NUMINAMATH_CALUDE_travel_time_ratio_l2250_225029

theorem travel_time_ratio : 
  let distance : ℝ := 600
  let initial_time : ℝ := 5
  let new_speed : ℝ := 80
  let new_time : ℝ := distance / new_speed
  new_time / initial_time = 1.5 := by sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l2250_225029


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l2250_225010

theorem sunflower_seed_contest (player1 player2 player3 total : ℕ) : 
  player1 = 78 → 
  player2 = 53 → 
  player3 = player2 + 30 → 
  total = player1 + player2 + player3 → 
  total = 214 := by
sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l2250_225010


namespace NUMINAMATH_CALUDE_amanda_walk_distance_l2250_225066

/-- Amanda's walk to Kimberly's house -/
theorem amanda_walk_distance :
  let initial_speed : ℝ := 2
  let time_before_break : ℝ := 1.5
  let break_duration : ℝ := 0.5
  let speed_after_break : ℝ := 3
  let total_time : ℝ := 3.5
  let distance_before_break := initial_speed * time_before_break
  let time_after_break := total_time - break_duration - time_before_break
  let distance_after_break := speed_after_break * time_after_break
  let total_distance := distance_before_break + distance_after_break
  total_distance = 7.5 := by sorry

end NUMINAMATH_CALUDE_amanda_walk_distance_l2250_225066


namespace NUMINAMATH_CALUDE_rectangle_with_border_area_l2250_225061

/-- Calculates the combined area of a rectangle and its border -/
def combinedArea (length width borderWidth : Real) : Real :=
  (length + 2 * borderWidth) * (width + 2 * borderWidth)

theorem rectangle_with_border_area :
  let length : Real := 0.6
  let width : Real := 0.35
  let borderWidth : Real := 0.05
  combinedArea length width borderWidth = 0.315 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_with_border_area_l2250_225061


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2250_225079

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ 
  (x : ℤ) ≡ 4 [ZMOD 7] ∧
  ∀ y : ℕ+, ((y : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ (y : ℤ) ≡ 4 [ZMOD 7]) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2250_225079


namespace NUMINAMATH_CALUDE_fraction_equality_l2250_225027

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + 2*y) / (x - 4*y) = -3) : 
  (2*x + 8*y) / (4*x - 2*y) = 38/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2250_225027


namespace NUMINAMATH_CALUDE_unique_integer_proof_l2250_225030

theorem unique_integer_proof : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (24.7 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25) :=
by
  use 612
  sorry

end NUMINAMATH_CALUDE_unique_integer_proof_l2250_225030


namespace NUMINAMATH_CALUDE_grade_assignment_count_l2250_225069

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l2250_225069


namespace NUMINAMATH_CALUDE_f_properties_l2250_225052

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 * Real.log x

-- State the theorem
theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- Minimum value of f(x) is -1/(2e)
  (∃ x : ℝ, x > 0 ∧ f a x = -(1/(2*Real.exp 1)) ∧ ∀ y : ℝ, y > 0 → f a y ≥ -(1/(2*Real.exp 1))) →
  -- f(x) is decreasing on (0, e^(-1/2))
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < Real.exp (-1/2) → f a x > f a y) ∧
  -- f(x) is increasing on (e^(-1/2), +∞)
  (∀ x y : ℝ, Real.exp (-1/2) < x ∧ x < y → f a x < f a y) ∧
  -- For all x > 0, f(x) > x^2/e^x - 3/4
  (∀ x : ℝ, x > 0 → f a x > x^2 / Real.exp x - 3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2250_225052


namespace NUMINAMATH_CALUDE_scientific_notation_of_wetland_area_l2250_225053

/-- Proves that 29.47 thousand is equal to 2.947 × 10^4 in scientific notation -/
theorem scientific_notation_of_wetland_area :
  (29.47 * 1000 : ℝ) = 2.947 * (10 ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_wetland_area_l2250_225053


namespace NUMINAMATH_CALUDE_range_of_f_l2250_225051

def f (x : ℝ) : ℝ := |x| + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2250_225051


namespace NUMINAMATH_CALUDE_reciprocal_sum_relation_l2250_225060

theorem reciprocal_sum_relation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_relation_l2250_225060


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_product_mod_100_l2250_225046

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * Nat.factorial (i + 1)) 1

theorem trailing_zeros_factorial_product_mod_100 :
  trailingZeros (factorialProduct 50) % 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_factorial_product_mod_100_l2250_225046


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2250_225020

theorem solution_implies_m_value (m : ℝ) : 
  (2 * 2 + m - 1 = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2250_225020


namespace NUMINAMATH_CALUDE_female_officers_count_l2250_225005

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 17 / 100 →
  female_ratio = 1 / 2 →
  ∃ (total_female : ℕ), 
    (female_on_duty_ratio * total_female = female_ratio * total_on_duty) ∧
    total_female = 500 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l2250_225005


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2250_225098

/-- Properties of a regular polygon with 24-degree exterior angles -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  (180 * (n - 2) = 2340 ∧ (n * (n - 3)) / 2 = 90) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2250_225098


namespace NUMINAMATH_CALUDE_high_school_students_l2250_225075

theorem high_school_students (total_students : ℕ) 
  (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 40)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 450)
  (h5 : total_students = (music_students - both_students) + (art_students - both_students) + both_students + neither_students) :
  total_students = 500 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l2250_225075


namespace NUMINAMATH_CALUDE_length_MN_is_eleven_thirds_l2250_225021

/-- Triangle ABC with given side lengths and points M and N -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Point M on AB such that CM is the angle bisector of ∠ACB
  M : ℝ
  -- Point N on AB such that CN is the altitude to AB
  N : ℝ
  -- Conditions
  h_AB : AB = 50
  h_BC : BC = 20
  h_AC : AC = 40
  h_M_angle_bisector : M = AB / 3
  h_N_altitude : N = BC * (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

/-- The length of MN in the given triangle -/
def length_MN (t : TriangleABC) : ℝ := t.M - t.N

/-- Theorem stating that the length of MN is 11/3 -/
theorem length_MN_is_eleven_thirds (t : TriangleABC) :
  length_MN t = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_length_MN_is_eleven_thirds_l2250_225021


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2250_225006

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2022) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  506 ≤ N ∧ ∃ (p' q' r' s' t' : ℕ+), p' + q' + r' + s' + t' = 2022 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) = 506 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2250_225006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2250_225092

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_solution :
  ∀ (a : ℕ → ℝ),
    is_arithmetic_sequence a →
    a 0 = 2^2 →
    a 2 = 5^2 →
    ∃ x : ℝ, x > 0 ∧ a 1 = x^2 ∧ x = Real.sqrt 14.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2250_225092


namespace NUMINAMATH_CALUDE_multiplication_table_even_fraction_l2250_225089

/-- The size of the multiplication table (16 in this case) -/
def table_size : ℕ := 16

/-- A number is even if it's divisible by 2 -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- The count of even numbers in the range [0, table_size - 1] -/
def even_count : ℕ := (table_size + 1) / 2

/-- The count of odd numbers in the range [0, table_size - 1] -/
def odd_count : ℕ := table_size - even_count

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of entries where both factors are odd -/
def odd_entries : ℕ := odd_count * odd_count

/-- The number of entries where at least one factor is even -/
def even_entries : ℕ := total_entries - odd_entries

/-- The fraction of even entries in the multiplication table -/
def even_fraction : ℚ := even_entries / total_entries

theorem multiplication_table_even_fraction :
  even_fraction = 3/4 := by sorry

end NUMINAMATH_CALUDE_multiplication_table_even_fraction_l2250_225089


namespace NUMINAMATH_CALUDE_nickel_piles_count_l2250_225082

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value Rocco has in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of nickels is 9 -/
theorem nickel_piles_count : 
  ∃ (nickel_piles : ℕ), 
    nickel_piles = 9 ∧
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = 
    total_value :=
by sorry

end NUMINAMATH_CALUDE_nickel_piles_count_l2250_225082


namespace NUMINAMATH_CALUDE_staff_meeting_attendance_l2250_225068

theorem staff_meeting_attendance (total_doughnuts served_doughnuts left_doughnuts doughnuts_per_staff : ℕ) :
  served_doughnuts = 50 →
  doughnuts_per_staff = 2 →
  left_doughnuts = 12 →
  total_doughnuts = served_doughnuts - left_doughnuts →
  (total_doughnuts / doughnuts_per_staff : ℕ) = 19 :=
by sorry

end NUMINAMATH_CALUDE_staff_meeting_attendance_l2250_225068


namespace NUMINAMATH_CALUDE_student_calculation_l2250_225058

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 124 → 
  (2 * chosen_number) - 138 = 110 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_l2250_225058


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l2250_225056

/-- Proves that the new supervisor's salary is $960 given the conditions of the problem -/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_avg_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_avg_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_avg : initial_avg_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_avg : new_avg_salary = 440)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 960 ∧
    (num_workers : ℚ) * initial_avg_salary + old_supervisor_salary = (num_total : ℚ) * initial_avg_salary ∧
    (num_workers : ℚ) * initial_avg_salary + new_supervisor_salary = (num_total : ℚ) * new_avg_salary :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l2250_225056


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2250_225064

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l2250_225064


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2250_225019

/-- The volume of the tetrahedron formed by alternating vertices of a cube -/
theorem blue_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := cube_volume / 3
  blue_tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2250_225019


namespace NUMINAMATH_CALUDE_max_min_value_of_expression_l2250_225041

theorem max_min_value_of_expression (a b : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ Real.sqrt 3) 
  (hb : 1 ≤ b ∧ b ≤ Real.sqrt 3) :
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = 1) ∧
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = Real.sqrt 3) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) → 
                y ∈ Set.Icc 1 (Real.sqrt 3) → 
                1 ≤ (x^2 + y^2 - 1) / (x * y) ∧ 
                (x^2 + y^2 - 1) / (x * y) ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_of_expression_l2250_225041


namespace NUMINAMATH_CALUDE_sqrt_x4_eq_x2_l2250_225008

theorem sqrt_x4_eq_x2 : ∀ x : ℝ, Real.sqrt (x^4) = x^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x4_eq_x2_l2250_225008


namespace NUMINAMATH_CALUDE_athlete_exercise_time_l2250_225073

/-- Prove that given an athlete who burns 10 calories per minute while running,
    4 calories per minute while walking, burns 450 calories in total,
    and spends 35 minutes running, the total exercise time is 60 minutes. -/
theorem athlete_exercise_time
  (calories_per_minute_running : ℕ)
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (time_running : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : time_running = 35) :
  time_running + (total_calories_burned - calories_per_minute_running * time_running) / calories_per_minute_walking = 60 :=
by sorry

end NUMINAMATH_CALUDE_athlete_exercise_time_l2250_225073


namespace NUMINAMATH_CALUDE_no_valid_triples_l2250_225071

theorem no_valid_triples : 
  ¬∃ (a b c : ℤ) (x : ℚ), 
    a < 0 ∧ 
    b^2 - 4*a*c = 5 ∧ 
    a * x^2 + b * x + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_triples_l2250_225071


namespace NUMINAMATH_CALUDE_bookstore_profit_percentage_l2250_225034

/-- Given three textbooks with their cost and selling prices, prove that the total profit percentage
    based on the combined selling prices is approximately 20.94%. -/
theorem bookstore_profit_percentage
  (cost1 : ℝ) (sell1 : ℝ) (cost2 : ℝ) (sell2 : ℝ) (cost3 : ℝ) (sell3 : ℝ)
  (h1 : cost1 = 44)
  (h2 : sell1 = 55)
  (h3 : cost2 = 58)
  (h4 : sell2 = 72)
  (h5 : cost3 = 83)
  (h6 : sell3 = 107) :
  let total_profit := (sell1 - cost1) + (sell2 - cost2) + (sell3 - cost3)
  let total_selling_price := sell1 + sell2 + sell3
  let profit_percentage := (total_profit / total_selling_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |profit_percentage - 20.94| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_bookstore_profit_percentage_l2250_225034


namespace NUMINAMATH_CALUDE_polyhedron_volume_l2250_225011

/-- The volume of a polyhedron composed of a regular quadrilateral prism and two regular quadrilateral pyramids -/
theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h_prism : prism_volume = Real.sqrt 2 - 1)
  (h_pyramid : pyramid_volume = 1 / 6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l2250_225011


namespace NUMINAMATH_CALUDE_money_problem_l2250_225028

theorem money_problem (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l2250_225028


namespace NUMINAMATH_CALUDE_function_negative_on_interval_l2250_225042

theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  -Real.sqrt 2 / 2 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_function_negative_on_interval_l2250_225042


namespace NUMINAMATH_CALUDE_pirate_coin_distribution_l2250_225002

/-- The number of rounds in the coin distribution process -/
def y : ℕ := sorry

/-- The total number of coins Pete has after distribution -/
def peteCoins : ℕ := y * (y + 1) / 2

/-- The total number of coins Paul has after distribution -/
def paulCoins : ℕ := y

/-- The ratio of Pete's coins to Paul's coins -/
def coinRatio : ℕ := 5

theorem pirate_coin_distribution :
  peteCoins = coinRatio * paulCoins ∧ peteCoins + paulCoins = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_distribution_l2250_225002


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l2250_225072

-- Define a square with a given area
def Square (area : ℝ) : Type :=
  { side : ℝ // side * side = area }

-- Define the perimeter of a square
def perimeter (s : Square 625) : ℝ :=
  4 * s.val

-- Theorem statement
theorem square_perimeter_from_area :
  ∀ s : Square 625, perimeter s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l2250_225072


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l2250_225017

def is_1962_digit (n : ℕ) : Prop := 10^1961 ≤ n ∧ n < 10^1962

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9 
  (n : ℕ) 
  (h1 : is_1962_digit n) 
  (h2 : n % 9 = 0) : 
  let a := sum_of_digits n
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l2250_225017


namespace NUMINAMATH_CALUDE_transformed_line_equation_l2250_225018

/-- Given a line and a scaling transformation, prove the equation of the transformed line -/
theorem transformed_line_equation (x y x' y' : ℝ) :
  (x - 2 * y = 2) →  -- Original line equation
  (x' = x) →         -- Scaling transformation for x
  (y' = 2 * y) →     -- Scaling transformation for y
  (x' - y' - 2 = 0)  -- Resulting line equation
:= by sorry

end NUMINAMATH_CALUDE_transformed_line_equation_l2250_225018


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2250_225065

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2250_225065


namespace NUMINAMATH_CALUDE_max_value_theorem_l2250_225000

theorem max_value_theorem (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b c : ℝ), 3 * a + 4 * b + 2 * c = 12 →
    a^2 * b + a^2 * c + b * c^2 ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2250_225000


namespace NUMINAMATH_CALUDE_solution_set_not_three_elements_l2250_225023

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a^(|x - b|)

-- Theorem statement
theorem solution_set_not_three_elements
  (a b m n p : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0) :
  ¬ ∃ (x y z : ℝ),
    (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧
    (∀ w, m * (f a b w)^2 + n * (f a b w) + p = 0 ↔ w = x ∨ w = y ∨ w = z) :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_not_three_elements_l2250_225023


namespace NUMINAMATH_CALUDE_range_of_x_minus_cosy_l2250_225055

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cosy_l2250_225055


namespace NUMINAMATH_CALUDE_sum_equality_seven_eight_l2250_225007

theorem sum_equality_seven_eight (S : Finset ℤ) (h : S.card = 15) :
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 7 ∧ s = T.sum id} =
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 8 ∧ s = T.sum id} :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_seven_eight_l2250_225007


namespace NUMINAMATH_CALUDE_bank_transfer_result_l2250_225081

def initial_balance : ℕ := 27004
def transfer_amount : ℕ := 69

theorem bank_transfer_result :
  initial_balance - transfer_amount = 26935 :=
by sorry

end NUMINAMATH_CALUDE_bank_transfer_result_l2250_225081
