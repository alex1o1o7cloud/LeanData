import Mathlib

namespace NUMINAMATH_CALUDE_max_full_pikes_l1385_138549

/-- The maximum number of full pikes given initial conditions -/
theorem max_full_pikes (initial_pikes : ℕ) (full_requirement : ℕ) 
  (h1 : initial_pikes = 30)
  (h2 : full_requirement = 3) : 
  ∃ (max_full : ℕ), max_full = 9 ∧ 
  (∀ (n : ℕ), n ≤ initial_pikes - 1 → n * full_requirement ≤ initial_pikes - 1 ↔ n ≤ max_full) :=
sorry

end NUMINAMATH_CALUDE_max_full_pikes_l1385_138549


namespace NUMINAMATH_CALUDE_num_arrangements_eq_360_l1385_138550

/-- The number of volunteers --/
def num_volunteers : ℕ := 6

/-- The number of people to be selected --/
def num_selected : ℕ := 4

/-- The number of distinct tasks --/
def num_tasks : ℕ := 4

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_eq_360 : 
  (num_volunteers.factorial) / ((num_volunteers - num_selected).factorial) = 360 :=
sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_360_l1385_138550


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1385_138510

def complexI : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + complexI) = 2 - 3 * complexI) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1385_138510


namespace NUMINAMATH_CALUDE_quadratic_symmetry_and_value_l1385_138520

/-- A quadratic function with symmetry around x = 5.5 and p(0) = -4 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry_and_value (a b c : ℝ) :
  (∀ x, p a b c (5.5 - x) = p a b c (5.5 + x)) →  -- Symmetry around x = 5.5
  p a b c 0 = -4 →                                -- p(0) = -4
  p a b c 11 = -4 :=                              -- Conclusion: p(11) = -4
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_and_value_l1385_138520


namespace NUMINAMATH_CALUDE_second_stop_count_l1385_138512

/-- The number of students who got on the bus during the first stop -/
def first_stop_students : ℕ := 39

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- The number of students who got on the bus during the second stop -/
def second_stop_students : ℕ := total_students - first_stop_students

theorem second_stop_count : second_stop_students = 29 := by
  sorry

end NUMINAMATH_CALUDE_second_stop_count_l1385_138512


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1385_138539

/-- The curve function f(x) = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_slope :
  ∃ a : ℝ,
    (f a = (f' a) * a) ∧  -- Point (a, f(a)) lies on the tangent line
    (f' a = 13) -- The slope of the tangent line is 13
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1385_138539


namespace NUMINAMATH_CALUDE_complete_square_problems_l1385_138546

theorem complete_square_problems :
  (∀ a b : ℝ, a + b = 5 ∧ a * b = 2 → a^2 + b^2 = 21) ∧
  (∀ a b : ℝ, a + b = 10 ∧ a^2 + b^2 = 50^2 → a * b = -1200) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_problems_l1385_138546


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1385_138502

theorem contradiction_assumption (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1385_138502


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1385_138562

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 →
    a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + a*b^5 ≤ max) ∧
  (x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 22884) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧
    x₀^5*y₀ + x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 + x₀*y₀^5 = 22884) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1385_138562


namespace NUMINAMATH_CALUDE_negative_two_classification_l1385_138584

theorem negative_two_classification :
  (∃ (n : ℤ), n = -2) →  -- -2 is an integer
  (∃ (q : ℚ), q = -2 ∧ q < 0)  -- -2 is a negative rational number
:= by sorry

end NUMINAMATH_CALUDE_negative_two_classification_l1385_138584


namespace NUMINAMATH_CALUDE_candy_bar_multiple_l1385_138528

def fred_candy_bars : ℕ := 12
def uncle_bob_extra_candy_bars : ℕ := 6
def jacqueline_percentage : ℚ := 40 / 100
def jacqueline_percentage_amount : ℕ := 120

theorem candy_bar_multiple :
  let uncle_bob_candy_bars := fred_candy_bars + uncle_bob_extra_candy_bars
  let total_fred_uncle_bob := fred_candy_bars + uncle_bob_candy_bars
  let jacqueline_candy_bars := jacqueline_percentage_amount / jacqueline_percentage
  jacqueline_candy_bars / total_fred_uncle_bob = 10 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_multiple_l1385_138528


namespace NUMINAMATH_CALUDE_total_three_digit_numbers_l1385_138551

/-- Represents a card with two numbers -/
structure Card where
  side1 : Nat
  side2 : Nat
  different : side1 ≠ side2

/-- The set of cards given in the problem -/
def problemCards : Finset Card := sorry

/-- The number of ways to arrange 3 cards -/
def cardArrangements : Nat := sorry

/-- The number of ways to choose sides for 3 cards -/
def sideChoices : Nat := sorry

/-- Theorem stating the total number of different three-digit numbers -/
theorem total_three_digit_numbers : 
  cardArrangements * sideChoices = 48 := by sorry

end NUMINAMATH_CALUDE_total_three_digit_numbers_l1385_138551


namespace NUMINAMATH_CALUDE_candy_mixture_price_l1385_138545

/-- Calculates the selling price per pound of a candy mixture -/
theorem candy_mixture_price (total_weight : ℝ) (cheap_weight : ℝ) (cheap_price : ℝ) (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : cheap_weight = 64)
  (h3 : cheap_price = 2)
  (h4 : expensive_price = 3)
  : (cheap_weight * cheap_price + (total_weight - cheap_weight) * expensive_price) / total_weight = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l1385_138545


namespace NUMINAMATH_CALUDE_exact_fare_payment_l1385_138548

/-- The bus fare in kopecks -/
def busFare : ℕ := 5

/-- The smallest coin denomination in kopecks -/
def smallestCoin : ℕ := 10

/-- The number of passengers is always a multiple of 4 -/
def numPassengers (k : ℕ) : ℕ := 4 * k

/-- The minimum number of coins required for exact fare payment -/
def minCoins (k : ℕ) : ℕ := 5 * k

theorem exact_fare_payment (k : ℕ) (h : k > 0) :
  ∀ (n : ℕ), n < minCoins k → ¬∃ (coins : List ℕ),
    (∀ c ∈ coins, c ≥ smallestCoin) ∧
    coins.length = n ∧
    coins.sum = busFare * numPassengers k :=
  sorry

#check exact_fare_payment

end NUMINAMATH_CALUDE_exact_fare_payment_l1385_138548


namespace NUMINAMATH_CALUDE_max_cards_saved_is_34_l1385_138578

/-- The set of digits that remain valid when flipped upside down -/
def valid_digits : Finset Nat := {1, 6, 8, 9}

/-- The set of digits that can be used in the tens place -/
def tens_digits : Finset Nat := {0, 1, 6, 8, 9}

/-- The total number of three-digit numbers -/
def total_numbers : Nat := 900

/-- The number of valid reversible three-digit numbers -/
def reversible_numbers : Nat := valid_digits.card * tens_digits.card * valid_digits.card

/-- The number of palindromic reversible numbers -/
def palindromic_numbers : Nat := valid_digits.card * 3

/-- The maximum number of cards that can be saved -/
def max_cards_saved : Nat := (reversible_numbers - palindromic_numbers) / 2

theorem max_cards_saved_is_34 : max_cards_saved = 34 := by
  sorry

#eval max_cards_saved

end NUMINAMATH_CALUDE_max_cards_saved_is_34_l1385_138578


namespace NUMINAMATH_CALUDE_starting_number_proof_l1385_138558

theorem starting_number_proof (n : ℕ) (h1 : n > 0) (h2 : n ≤ 79) (h3 : n % 11 = 0)
  (h4 : ∀ k, k ∈ Finset.range 6 → (n - k * 11) % 11 = 0)
  (h5 : ∀ m, m < n - 5 * 11 → ¬(∃ l, l ∈ Finset.range 6 ∧ m = n - l * 11)) :
  n - 5 * 11 = 22 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l1385_138558


namespace NUMINAMATH_CALUDE_candidate_probability_l1385_138523

/-- Represents the probability space of job candidates -/
structure CandidateSpace where
  /-- Probability of having intermediate or advanced Excel skills -/
  excel_skills : ℝ
  /-- Probability of having intermediate Excel skills -/
  intermediate_excel : ℝ
  /-- Probability of having advanced Excel skills -/
  advanced_excel : ℝ
  /-- Probability of being willing to work night shifts among those with Excel skills -/
  night_shift_willing : ℝ
  /-- Probability of not being willing to work weekends among those willing to work night shifts -/
  weekend_unwilling : ℝ
  /-- Ensure probabilities are valid -/
  excel_skills_valid : excel_skills = intermediate_excel + advanced_excel
  excel_skills_prob : excel_skills = 0.45
  intermediate_excel_prob : intermediate_excel = 0.25
  advanced_excel_prob : advanced_excel = 0.20
  night_shift_willing_prob : night_shift_willing = 0.32
  weekend_unwilling_prob : weekend_unwilling = 0.60

/-- The main theorem to prove -/
theorem candidate_probability (cs : CandidateSpace) :
  cs.excel_skills * cs.night_shift_willing * cs.weekend_unwilling = 0.0864 := by
  sorry

end NUMINAMATH_CALUDE_candidate_probability_l1385_138523


namespace NUMINAMATH_CALUDE_f_min_value_when_a_is_one_f_inequality_solution_range_l1385_138547

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem 1: Minimum value of f when a = 1
theorem f_min_value_when_a_is_one :
  ∃ (min : ℝ), min = 3/2 ∧ ∀ (x : ℝ), f x 1 ≥ min :=
sorry

-- Theorem 2: Range of a for which f(x) < 5/x + a has a solution in [1, 2]
theorem f_inequality_solution_range :
  ∀ (a : ℝ), a > 0 →
    (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f x a < 5/x + a) ↔ (11/2 < a ∧ a < 9/2) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_when_a_is_one_f_inequality_solution_range_l1385_138547


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1385_138516

theorem triangle_third_side_length 
  (a b : ℝ) 
  (γ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hγ : γ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧ c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1385_138516


namespace NUMINAMATH_CALUDE_ratio_six_three_percent_l1385_138508

/-- Expresses a ratio as a percentage -/
def ratioToPercent (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ) * 100

/-- The ratio 6:3 expressed as a percent is 200% -/
theorem ratio_six_three_percent : ratioToPercent 6 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_six_three_percent_l1385_138508


namespace NUMINAMATH_CALUDE_max_difference_correct_l1385_138529

/-- Represents a convex N-gon divided into triangles by non-intersecting diagonals --/
structure ConvexNgon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  is_valid : triangles = N - 2 ∧ diagonals = N - 3

/-- Represents a coloring of the triangles in the N-gon --/
structure Coloring (N : ℕ) where
  ngon : ConvexNgon N
  white_triangles : ℕ
  black_triangles : ℕ
  is_valid : white_triangles + black_triangles = ngon.triangles
  adjacent_different : True  -- Represents the condition that adjacent triangles have different colors

/-- The maximum difference between white and black triangles for a given N --/
def max_difference (N : ℕ) : ℕ :=
  if N % 3 = 1 then
    N / 3 - 1
  else
    N / 3

/-- The theorem stating the maximum difference between white and black triangles --/
theorem max_difference_correct (N : ℕ) (c : Coloring N) :
  (c.white_triangles : ℤ) - (c.black_triangles : ℤ) ≤ max_difference N := by
  sorry

end NUMINAMATH_CALUDE_max_difference_correct_l1385_138529


namespace NUMINAMATH_CALUDE_ternary_121_eq_decimal_16_l1385_138585

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_eq_decimal_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_eq_decimal_16_l1385_138585


namespace NUMINAMATH_CALUDE_difference_of_squares_l1385_138544

theorem difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1385_138544


namespace NUMINAMATH_CALUDE_profit_calculation_l1385_138599

/-- The profit calculation for a product with given purchase price, markup percentage, and discount. -/
theorem profit_calculation (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  purchase_price = 200 →
  markup_percent = 1.25 →
  discount_percent = 0.9 →
  purchase_price * markup_percent * discount_percent - purchase_price = 25 := by
  sorry

#check profit_calculation

end NUMINAMATH_CALUDE_profit_calculation_l1385_138599


namespace NUMINAMATH_CALUDE_gcd_of_all_P_is_one_l1385_138581

-- Define P as a function of n, where n represents the first of the three consecutive even integers
def P (n : ℕ) : ℕ := 2 * n * (2 * n + 2) * (2 * n + 4) + 2

-- Theorem stating that the greatest common divisor of all P(n) is 1
theorem gcd_of_all_P_is_one : ∃ (d : ℕ), d > 0 ∧ (∀ (n : ℕ), n > 0 → d ∣ P n) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_all_P_is_one_l1385_138581


namespace NUMINAMATH_CALUDE_tony_drives_five_days_a_week_l1385_138534

/-- Represents the problem of determining Tony's work commute frequency --/
def TonysDriving (car_efficiency : ℝ) (round_trip : ℝ) (tank_capacity : ℝ) (gas_price : ℝ) (total_spent : ℝ) (weeks : ℕ) : Prop :=
  let gallons_per_day := round_trip / car_efficiency
  let total_gallons := total_spent / gas_price
  let gallons_per_week := total_gallons / weeks
  gallons_per_week / gallons_per_day = 5

/-- Theorem stating that given the problem conditions, Tony drives to work 5 days a week --/
theorem tony_drives_five_days_a_week :
  TonysDriving 25 50 10 2 80 4 := by
  sorry

end NUMINAMATH_CALUDE_tony_drives_five_days_a_week_l1385_138534


namespace NUMINAMATH_CALUDE_investment_solution_l1385_138579

def investment_problem (x y r1 r2 total_investment desired_interest : ℝ) : Prop :=
  x + y = total_investment ∧
  r1 * x + r2 * y = desired_interest

theorem investment_solution :
  investment_problem 6000 4000 0.09 0.11 10000 980 := by
  sorry

end NUMINAMATH_CALUDE_investment_solution_l1385_138579


namespace NUMINAMATH_CALUDE_line_circle_distance_range_l1385_138507

/-- The range of k for which a line y = k(x+2) has at least three points on the circle x^2 + y^2 = 4 at distance 1 from it -/
theorem line_circle_distance_range :
  ∀ k : ℝ,
  (∃ (A B C : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|k * (A.1 + 2) - A.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (B.1 + 2) - B.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (C.1 + 2) - C.2| / Real.sqrt (k^2 + 1) = 1))
  ↔
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_distance_range_l1385_138507


namespace NUMINAMATH_CALUDE_garden_length_l1385_138537

theorem garden_length (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 2 * width →
  2 * length + 2 * width = 240 →
  length = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_length_l1385_138537


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1385_138576

theorem inequality_equivalence (x : ℝ) : (x + 1) / 2 ≥ x / 3 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1385_138576


namespace NUMINAMATH_CALUDE_rectangle_square_probability_l1385_138515

theorem rectangle_square_probability (rectangle_A_area square_B_perimeter : ℝ) :
  rectangle_A_area = 30 →
  square_B_perimeter = 16 →
  let square_B_side := square_B_perimeter / 4
  let square_B_area := square_B_side ^ 2
  let area_difference := rectangle_A_area - square_B_area
  (area_difference / rectangle_A_area) = 7 / 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_probability_l1385_138515


namespace NUMINAMATH_CALUDE_boatman_downstream_distance_l1385_138595

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stationary : ℝ
  upstream : ℝ
  current : ℝ
  downstream : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance traveled by the boatman along the current -/
theorem boatman_downstream_distance 
  (speed : BoatSpeed)
  (h1 : distance speed.upstream 3 = 3) -- 3 km against current in 3 hours
  (h2 : distance speed.stationary 2 = 3) -- 3 km in stationary water in 2 hours
  (h3 : speed.current = speed.stationary - speed.upstream)
  (h4 : speed.downstream = speed.stationary + speed.current) :
  distance speed.downstream 0.5 = 1 := by
  sorry

#check boatman_downstream_distance

end NUMINAMATH_CALUDE_boatman_downstream_distance_l1385_138595


namespace NUMINAMATH_CALUDE_pool_filling_rates_l1385_138596

theorem pool_filling_rates (r₁ r₂ r₃ : ℝ) 
  (h1 : r₁ + r₂ = 1 / 70)
  (h2 : r₁ + r₃ = 1 / 84)
  (h3 : r₂ + r₃ = 1 / 140) :
  r₁ = 1 / 105 ∧ r₂ = 1 / 210 ∧ r₃ = 1 / 420 ∧ r₁ + r₂ + r₃ = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rates_l1385_138596


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1385_138586

/-- Represents a rectangular field with a width that is one-third of its length -/
structure RectangularField where
  length : ℝ
  width : ℝ
  width_is_third_of_length : width = length / 3
  perimeter_is_72 : 2 * (length + width) = 72

/-- The area of a rectangular field with the given conditions is 243 square meters -/
theorem rectangular_field_area (field : RectangularField) : field.length * field.width = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1385_138586


namespace NUMINAMATH_CALUDE_first_agency_daily_charge_is_correct_l1385_138591

/-- The daily charge of the first car rental agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first car rental agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The daily charge of the second car rental agency -/
def second_agency_daily_charge : ℝ := 18.25

/-- The per-mile charge of the second car rental agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the costs become equal -/
def miles_equal_cost : ℝ := 25

theorem first_agency_daily_charge_is_correct :
  first_agency_daily_charge + first_agency_mile_charge * miles_equal_cost =
  second_agency_daily_charge + second_agency_mile_charge * miles_equal_cost :=
by sorry

end NUMINAMATH_CALUDE_first_agency_daily_charge_is_correct_l1385_138591


namespace NUMINAMATH_CALUDE_factorization_sum_l1385_138513

def P (y : ℤ) : ℤ := y^6 - y^3 - 2*y - 2

def is_irreducible_factor (q : ℤ → ℤ) : Prop :=
  (∀ y, q y ∣ P y) ∧ 
  (∀ f g : ℤ → ℤ, (∀ y, q y = f y * g y) → (∀ y, f y = 1 ∨ g y = 1))

theorem factorization_sum (q₁ q₂ q₃ q₄ : ℤ → ℤ) :
  is_irreducible_factor q₁ ∧
  is_irreducible_factor q₂ ∧
  is_irreducible_factor q₃ ∧
  is_irreducible_factor q₄ ∧
  (∀ y, P y = q₁ y * q₂ y * q₃ y * q₄ y) →
  q₁ 3 + q₂ 3 + q₃ 3 + q₄ 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1385_138513


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_distance_l1385_138569

def Ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

def LeftFocus (x y : ℝ) : Prop := x = -Real.sqrt 7 ∧ y = 0
def RightFocus (x y : ℝ) : Prop := x = Real.sqrt 7 ∧ y = 0

def RightTriangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∨
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0 ∨
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

theorem ellipse_right_triangle_distance (x y xf₁ yf₁ xf₂ yf₂ : ℝ) :
  Ellipse x y →
  LeftFocus xf₁ yf₁ →
  RightFocus xf₂ yf₂ →
  RightTriangle x y xf₁ yf₁ xf₂ yf₂ →
  |y| = 9/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_distance_l1385_138569


namespace NUMINAMATH_CALUDE_sum_of_six_l1385_138554

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Properties of the specific arithmetic sequence -/
def special_sequence (seq : ArithmeticSequence) : Prop :=
  seq.a 1 = 2 ∧ seq.S 4 = 20

theorem sum_of_six (seq : ArithmeticSequence) (h : special_sequence seq) : 
  seq.S 6 = 42 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_six_l1385_138554


namespace NUMINAMATH_CALUDE_average_study_time_difference_l1385_138598

def daily_differences : List Int := [10, -10, 20, 30, -20]

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / daily_differences.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l1385_138598


namespace NUMINAMATH_CALUDE_union_and_intersection_when_m_2_intersection_empty_iff_l1385_138597

def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 3 * m + 3}

theorem union_and_intersection_when_m_2 :
  (A ∪ B 2 = {x | -3 < x ∧ x < 9}) ∧
  (A ∩ (Set.univ \ B 2) = {x | -3 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 5 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_m_2_intersection_empty_iff_l1385_138597


namespace NUMINAMATH_CALUDE_simplify_expression_l1385_138567

theorem simplify_expression (a b : ℝ) : 2*a*(2*a^2 + a*b) - a^2*b = 4*a^3 + a^2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1385_138567


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_l1385_138522

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_l1385_138522


namespace NUMINAMATH_CALUDE_andrew_game_preparation_time_l1385_138582

/-- Represents the time in minutes required to prepare each type of game -/
structure GamePreparationTime where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of games of each type to be prepared -/
structure GameCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total preparation time for all games -/
def totalPreparationTime (prep : GamePreparationTime) (counts : GameCounts) : ℕ :=
  prep.typeA * counts.typeA + prep.typeB * counts.typeB + prep.typeC * counts.typeC

/-- Theorem: Given the specific game preparation times and counts, the total preparation time is 350 minutes -/
theorem andrew_game_preparation_time :
  let prep : GamePreparationTime := { typeA := 15, typeB := 25, typeC := 30 }
  let counts : GameCounts := { typeA := 5, typeB := 5, typeC := 5 }
  totalPreparationTime prep counts = 350 := by
  sorry

end NUMINAMATH_CALUDE_andrew_game_preparation_time_l1385_138582


namespace NUMINAMATH_CALUDE_abs_plus_square_zero_implies_product_l1385_138543

theorem abs_plus_square_zero_implies_product (a b : ℝ) : 
  |a - 1| + (b + 2)^2 = 0 → a * b^a = -2 := by
sorry

end NUMINAMATH_CALUDE_abs_plus_square_zero_implies_product_l1385_138543


namespace NUMINAMATH_CALUDE_intersection_point_unique_intersection_point_correct_l1385_138555

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 7) / 3 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 1) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + 7 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (10, 4, -3)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 :=
by
  sorry

theorem intersection_point_correct :
  line intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane intersection_point.1 intersection_point.2.1 intersection_point.2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_intersection_point_correct_l1385_138555


namespace NUMINAMATH_CALUDE_cubic_factorization_l1385_138564

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1385_138564


namespace NUMINAMATH_CALUDE_macaroon_weight_l1385_138557

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l1385_138557


namespace NUMINAMATH_CALUDE_ball_color_probability_l1385_138526

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1 / 2
  let k : ℕ := 4
  Nat.choose n k * p^n = 35 / 128 := by sorry

end NUMINAMATH_CALUDE_ball_color_probability_l1385_138526


namespace NUMINAMATH_CALUDE_triangle_properties_l1385_138587

theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →  -- Sine rule
  (Real.sin A + Real.sin B)^2 = (2 * Real.sin B + Real.sin C) * Real.sin C →  -- Given equation
  Real.sin A > Real.sqrt 3 / 3 →  -- Given inequality
  (c - a = a * Real.cos C) ∧ (c > a) ∧ (C > π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1385_138587


namespace NUMINAMATH_CALUDE_einstein_fundraising_l1385_138592

/-- Einstein's fundraising problem -/
theorem einstein_fundraising 
  (goal : ℕ)
  (pizza_price potato_price soda_price : ℚ)
  (pizza_sold potato_sold soda_sold : ℕ) :
  goal = 500 ∧ 
  pizza_price = 12 ∧ 
  potato_price = 3/10 ∧ 
  soda_price = 2 ∧
  pizza_sold = 15 ∧ 
  potato_sold = 40 ∧ 
  soda_sold = 25 →
  (goal : ℚ) - (pizza_price * pizza_sold + potato_price * potato_sold + soda_price * soda_sold) = 258 :=
by sorry


end NUMINAMATH_CALUDE_einstein_fundraising_l1385_138592


namespace NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l1385_138573

/-- The area of overlap between a rectangle and a circle with shared center -/
theorem rectangle_circle_overlap_area 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_radius : ℝ) 
  (h1 : rectangle_width = 8) 
  (h2 : rectangle_height = 2 * Real.sqrt 2) 
  (h3 : circle_radius = 2) : 
  ∃ (overlap_area : ℝ), overlap_area = 2 * Real.pi + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_overlap_area_l1385_138573


namespace NUMINAMATH_CALUDE_min_workers_team_a_l1385_138540

theorem min_workers_team_a (a b : ℕ) : 
  (∃ c : ℕ, c > 0 ∧ 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_team_a_l1385_138540


namespace NUMINAMATH_CALUDE_ribbon_leftover_l1385_138514

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) : 
  total_ribbon = 18 → num_gifts = 6 → ribbon_per_gift = 2 →
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l1385_138514


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l1385_138563

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 5)^2 + log10 2 * log10 5 + log10 20 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l1385_138563


namespace NUMINAMATH_CALUDE_angle_inequality_l1385_138541

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
sorry

end NUMINAMATH_CALUDE_angle_inequality_l1385_138541


namespace NUMINAMATH_CALUDE_gumballs_remaining_is_sixty_l1385_138566

/-- The number of gumballs remaining in the bowl after Pedro takes out 40% -/
def remaining_gumballs (alicia_gumballs : ℕ) : ℕ :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_out := (40 * total_gumballs) / 100
  total_gumballs - taken_out

/-- Theorem stating that given Alicia has 20 gumballs, the number of gumballs
    remaining in the bowl after Pedro takes out 40% is 60 -/
theorem gumballs_remaining_is_sixty :
  remaining_gumballs 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_remaining_is_sixty_l1385_138566


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1385_138538

open Real

theorem trigonometric_identity (x : ℝ) : 
  sin x * cos x + sin x^3 * cos x + sin x^5 * (1 / cos x) = tan x := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1385_138538


namespace NUMINAMATH_CALUDE_parallel_perpendicular_relation_l1385_138536

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_relation 
  (m n : Line) (α β : Plane) :
  (parallel m α ∧ parallel n β ∧ perpendicular α β) → 
  (line_perpendicular m n ∨ line_parallel m n) = False := by
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_relation_l1385_138536


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l1385_138553

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The doubling period of bacteria in seconds -/
def doubling_period : ℕ := 30

/-- The duration of the experiment in minutes -/
def experiment_duration : ℕ := 4

/-- The number of bacteria after the experiment -/
def final_bacteria_count : ℕ := 65536

/-- The number of doubling periods in the experiment -/
def doubling_periods : ℕ := (experiment_duration * seconds_per_minute) / doubling_period

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := final_bacteria_count / (2 ^ doubling_periods)

theorem initial_bacteria_count : initial_bacteria = 256 := by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l1385_138553


namespace NUMINAMATH_CALUDE_arithmetic_sequence_300th_term_l1385_138518

/-- 
Given an arithmetic sequence where:
- The first term is 6
- The common difference is 4
Prove that the 300th term is equal to 1202
-/
theorem arithmetic_sequence_300th_term : 
  let a : ℕ → ℕ := λ n => 6 + (n - 1) * 4
  a 300 = 1202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_300th_term_l1385_138518


namespace NUMINAMATH_CALUDE_expression_equals_four_l1385_138542

theorem expression_equals_four :
  let a := 7 + Real.sqrt 48
  let b := 7 - Real.sqrt 48
  (a^2023 + b^2023)^2 - (a^2023 - b^2023)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l1385_138542


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1385_138533

/-- An arithmetic sequence with first term 2 and the sum of the 3rd and 5th terms equal to 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms is 10
  a 2 - a 1 = 1 :=                     -- common difference is 1
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1385_138533


namespace NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l1385_138524

/-- Represents a letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- Calculates the maximum number of unsuccessful attempts for a given lock -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem stating that a lock with 5 rings and 10 letters per ring has 99,999 unsuccessful attempts -/
theorem letter_lock_unsuccessful_attempts :
  let lock : LetterLock := { num_rings := 5, letters_per_ring := 10 }
  max_unsuccessful_attempts lock = 99999 := by
  sorry

#eval max_unsuccessful_attempts { num_rings := 5, letters_per_ring := 10 }

end NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l1385_138524


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1385_138560

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (S : ℕ → ℝ)  -- S is the sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- Given condition
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum of arithmetic sequence
  : ∃ d : ℝ, (∀ n, a (n+1) - a n = d) ∧ d = -2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1385_138560


namespace NUMINAMATH_CALUDE_sugar_price_inconsistency_l1385_138575

/-- Represents the price and consumption change of sugar -/
structure SugarPriceChange where
  p₀ : ℝ  -- Initial price
  p₁ : ℝ  -- New price
  r : ℝ   -- Reduction in consumption (as a decimal)

/-- Checks if the given sugar price change is consistent -/
def is_consistent (s : SugarPriceChange) : Prop :=
  s.r = (s.p₁ - s.p₀) / s.p₁

/-- Theorem stating that the given conditions are inconsistent -/
theorem sugar_price_inconsistency :
  ¬ is_consistent ⟨3, 5, 0.4⟩ := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_inconsistency_l1385_138575


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1385_138517

theorem not_prime_sum_product (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ (Nat.Prime (a * b + c * d).natAbs) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1385_138517


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1385_138535

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1385_138535


namespace NUMINAMATH_CALUDE_bucket_full_weight_bucket_full_weight_proof_l1385_138561

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when completely full. -/
theorem bucket_full_weight (c d : ℝ) : ℝ :=
  let three_fourths_weight := c
  let one_third_weight := d
  let full_weight := (8/5 : ℝ) * c - (3/5 : ℝ) * d
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_full_weight_proof (c d : ℝ) :
  bucket_full_weight c d = (8/5 : ℝ) * c - (3/5 : ℝ) * d := by
  sorry

end NUMINAMATH_CALUDE_bucket_full_weight_bucket_full_weight_proof_l1385_138561


namespace NUMINAMATH_CALUDE_last_remaining_is_c_implies_start_is_f_l1385_138532

/-- Represents the children in the circle -/
inductive Child : Type
| a | b | c | d | e | f

/-- The number of children in the circle -/
def numChildren : Nat := 6

/-- The number of words in the song -/
def songWords : Nat := 9

/-- Function to determine the last remaining child given a starting position -/
def lastRemaining (start : Child) : Child :=
  sorry

/-- Theorem stating that if c is the last remaining child, the starting position must be f -/
theorem last_remaining_is_c_implies_start_is_f :
  lastRemaining Child.f = Child.c :=
sorry

end NUMINAMATH_CALUDE_last_remaining_is_c_implies_start_is_f_l1385_138532


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1385_138574

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (29*x)/(53*y)) :
  Real.sqrt x / Real.sqrt y = 91/42 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1385_138574


namespace NUMINAMATH_CALUDE_quadratic_product_property_l1385_138577

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℤ :=
  p.b^2 - 4 * p.a * p.c

/-- Predicate for a quadratic polynomial having distinct roots -/
def has_distinct_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p ≠ 0

/-- The product of the roots of a quadratic polynomial -/
def root_product (p : QuadraticPolynomial) : ℚ :=
  (p.c : ℚ) / (p.a : ℚ)

/-- The product of the coefficients of a quadratic polynomial -/
def coeff_product (p : QuadraticPolynomial) : ℤ :=
  p.a * p.b * p.c

theorem quadratic_product_property (p : QuadraticPolynomial) 
  (h_distinct : has_distinct_roots p)
  (h_product : (coeff_product p : ℚ) = root_product p) :
  ∃ (n : ℤ), n < 0 ∧ coeff_product p = n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_product_property_l1385_138577


namespace NUMINAMATH_CALUDE_bus_seats_l1385_138506

theorem bus_seats (west_lake : Nat) (east_lake : Nat)
  (h1 : west_lake = 138)
  (h2 : east_lake = 115)
  (h3 : ∀ x : Nat, x > 1 ∧ x ∣ west_lake ∧ x ∣ east_lake → x ≤ 23) :
  23 > 1 ∧ 23 ∣ west_lake ∧ 23 ∣ east_lake :=
by sorry

#check bus_seats

end NUMINAMATH_CALUDE_bus_seats_l1385_138506


namespace NUMINAMATH_CALUDE_missing_number_proof_l1385_138571

theorem missing_number_proof (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) 
  (h3 : a^3 = 21 * x * 15 * b) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1385_138571


namespace NUMINAMATH_CALUDE_reciprocal_and_fraction_operations_l1385_138531

theorem reciprocal_and_fraction_operations :
  (∀ a b c : ℚ, (a + b) / c = -2 → c / (a + b) = -1/2) ∧
  (5/12 - 1/9 + 2/3) / (1/36) = 35 ∧
  (-1/36) / (5/12 - 1/9 + 2/3) = -1/35 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_fraction_operations_l1385_138531


namespace NUMINAMATH_CALUDE_ice_cream_scoops_prove_ice_cream_scoops_l1385_138552

-- Define the given conditions
def aaron_savings : ℚ := 40
def carson_savings : ℚ := 40
def total_savings : ℚ := aaron_savings + carson_savings
def dinner_bill_ratio : ℚ := 3 / 4
def scoop_cost : ℚ := 3 / 2
def change_per_person : ℚ := 1

-- Define the theorem
theorem ice_cream_scoops : ℚ :=
  let dinner_bill := dinner_bill_ratio * total_savings
  let remaining_after_dinner := total_savings - dinner_bill
  let ice_cream_spending := remaining_after_dinner - 2 * change_per_person
  let total_scoops := ice_cream_spending / scoop_cost
  total_scoops / 2

-- The theorem to prove
theorem prove_ice_cream_scoops : ice_cream_scoops = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_prove_ice_cream_scoops_l1385_138552


namespace NUMINAMATH_CALUDE_left_handed_or_throwers_count_l1385_138504

/-- Represents a football team with specific player distributions -/
structure FootballTeam where
  total_players : Nat
  throwers : Nat
  left_handed : Nat
  right_handed : Nat

/-- Calculates the number of players who are either left-handed or throwers -/
def left_handed_or_throwers (team : FootballTeam) : Nat :=
  team.left_handed + team.throwers

/-- Theorem stating the number of players who are either left-handed or throwers in the given scenario -/
theorem left_handed_or_throwers_count (team : FootballTeam) :
  team.total_players = 70 →
  team.throwers = 34 →
  team.left_handed = (team.total_players - team.throwers) / 3 →
  team.right_handed = team.total_players - team.throwers - team.left_handed + team.throwers →
  left_handed_or_throwers team = 46 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_or_throwers_count_l1385_138504


namespace NUMINAMATH_CALUDE_total_leaves_l1385_138565

theorem total_leaves (initial_leaves additional_leaves : ℝ) :
  initial_leaves + additional_leaves = initial_leaves + additional_leaves :=
by sorry

end NUMINAMATH_CALUDE_total_leaves_l1385_138565


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1385_138527

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

def specific_arrangement : List Char := ['X', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ :=
  1 / (total_tiles.factorial / (x_tiles.factorial * o_tiles.factorial))

theorem probability_of_specific_arrangement :
  probability_of_arrangement = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1385_138527


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_plus_one_l1385_138509

theorem cubic_greater_than_quadratic_plus_one (x : ℝ) (h : x > 1) : 2 * x^3 > x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_plus_one_l1385_138509


namespace NUMINAMATH_CALUDE_cheryl_different_colors_probability_l1385_138588

/-- Represents the number of marbles of each color in the box -/
def initial_marbles : Nat := 2

/-- Represents the total number of colors -/
def total_colors : Nat := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : Nat := initial_marbles * total_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : Nat := 2

/-- Calculates the probability of Cheryl not getting two marbles of the same color -/
theorem cheryl_different_colors_probability :
  let total_outcomes := (total_marbles.choose marbles_drawn) * 
                        ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                        ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  let favorable_outcomes := (total_colors.choose marbles_drawn) * 
                            ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                            ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

#eval initial_marbles -- 2
#eval total_colors -- 4
#eval total_marbles -- 8
#eval marbles_drawn -- 2

end NUMINAMATH_CALUDE_cheryl_different_colors_probability_l1385_138588


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1385_138525

-- Define the siblings
inductive Sibling : Type
| Emily : Sibling
| Sam : Sibling
| Nora : Sibling
| Oliver : Sibling
| Jack : Sibling

-- Define the pizza consumption for each sibling
def pizza_consumption (s : Sibling) : Rat :=
  match s with
  | Sibling.Emily => 1/6
  | Sibling.Sam => 1/4
  | Sibling.Nora => 1/3
  | Sibling.Oliver => 1/8
  | Sibling.Jack => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define a function to compare pizza consumption
def consumes_more (s1 s2 : Sibling) : Prop :=
  pizza_consumption s1 > pizza_consumption s2

-- State the theorem
theorem pizza_consumption_order :
  consumes_more Sibling.Nora Sibling.Sam ∧
  consumes_more Sibling.Sam Sibling.Emily ∧
  consumes_more Sibling.Emily Sibling.Jack ∧
  consumes_more Sibling.Jack Sibling.Oliver :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1385_138525


namespace NUMINAMATH_CALUDE_marys_candy_count_l1385_138568

/-- Given that Megan has 5 pieces of candy, and Mary has 3 times as much candy as Megan
    plus an additional 10 pieces, prove that Mary's total candy is 25 pieces. -/
theorem marys_candy_count (megan_candy : ℕ) (mary_initial_multiplier : ℕ) (mary_additional_candy : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_initial_multiplier = 3)
  (h3 : mary_additional_candy = 10) :
  megan_candy * mary_initial_multiplier + mary_additional_candy = 25 := by
  sorry

end NUMINAMATH_CALUDE_marys_candy_count_l1385_138568


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1385_138505

theorem perfect_square_trinomial (a b : ℝ) : 9*a^2 - 24*a*b + 16*b^2 = (3*a + 4*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1385_138505


namespace NUMINAMATH_CALUDE_interest_calculation_l1385_138594

theorem interest_calculation (initial_investment second_investment second_interest : ℝ) 
  (h1 : initial_investment = 5000)
  (h2 : second_investment = 20000)
  (h3 : second_interest = 1000)
  (h4 : second_interest = second_investment * (second_interest / second_investment))
  (h5 : initial_investment > 0)
  (h6 : second_investment > 0) :
  initial_investment * (second_interest / second_investment) = 250 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l1385_138594


namespace NUMINAMATH_CALUDE_lisa_candy_weeks_l1385_138511

/-- The number of weeks it takes Lisa to eat all her candies -/
def weeks_to_eat_candies (total_candies : ℕ) (candies_mon_wed : ℕ) (candies_other_days : ℕ) : ℕ :=
  total_candies / (2 * candies_mon_wed + 5 * candies_other_days)

/-- Theorem stating that it takes 4 weeks for Lisa to eat all her candies -/
theorem lisa_candy_weeks : weeks_to_eat_candies 36 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candy_weeks_l1385_138511


namespace NUMINAMATH_CALUDE_percentage_relationship_l1385_138583

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4117647058823529)) :
  y = x * (1 + 0.7) := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1385_138583


namespace NUMINAMATH_CALUDE_cannot_determine_f_triple_prime_l1385_138593

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem cannot_determine_f_triple_prime (a b c : ℝ) :
  (∃ x, f a b c x = a * x^4 + b * x^2 + c) →
  ((12 * a + 2 * b) = 2) →
  ¬ (∃! y, (24 * a * (-1) = y)) :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_f_triple_prime_l1385_138593


namespace NUMINAMATH_CALUDE_vanya_masha_speed_ratio_l1385_138580

/-- Represents the scenario of Vanya and Masha's journey to school -/
structure SchoolJourney where
  d : ℝ  -- Total distance from home to school
  vanya_speed : ℝ  -- Vanya's speed
  masha_speed : ℝ  -- Masha's speed

/-- The theorem stating the relationship between Vanya and Masha's speeds -/
theorem vanya_masha_speed_ratio (journey : SchoolJourney) :
  journey.d > 0 →  -- Ensure the distance is positive
  (2/3 * journey.d) / journey.vanya_speed = (1/6 * journey.d) / journey.masha_speed →  -- Condition from overtaking point
  (1/2 * journey.d) / journey.masha_speed = journey.d / journey.vanya_speed →  -- Condition when Vanya reaches school
  journey.vanya_speed / journey.masha_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_vanya_masha_speed_ratio_l1385_138580


namespace NUMINAMATH_CALUDE_system_solution_l1385_138530

theorem system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 147/28 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1385_138530


namespace NUMINAMATH_CALUDE_equation_solution_l1385_138500

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ 
  x = -9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1385_138500


namespace NUMINAMATH_CALUDE_measure_15_minutes_with_7_and_11_l1385_138521

/-- Represents an hourglass that measures a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time_elapsed : ℕ
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Checks if it's possible to measure the target time with given hourglasses. -/
def can_measure_time (target : ℕ) (h1 h2 : Hourglass) : Prop :=
  ∃ (state : MeasurementState), state.time_elapsed = target ∧
    state.hourglass1 = h1 ∧ state.hourglass2 = h2

/-- Theorem stating that 15 minutes can be measured using 7-minute and 11-minute hourglasses. -/
theorem measure_15_minutes_with_7_and_11 :
  can_measure_time 15 (Hourglass.mk 7) (Hourglass.mk 11) := by
  sorry


end NUMINAMATH_CALUDE_measure_15_minutes_with_7_and_11_l1385_138521


namespace NUMINAMATH_CALUDE_teachers_not_picking_square_l1385_138572

theorem teachers_not_picking_square (total_teachers : ℕ) (square_teachers : ℕ) 
  (h1 : total_teachers = 20) 
  (h2 : square_teachers = 7) : 
  total_teachers - square_teachers = 13 := by
  sorry

end NUMINAMATH_CALUDE_teachers_not_picking_square_l1385_138572


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l1385_138519

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
  | 0 => 2  -- Grade 10
  | 1 => 2  -- Grade 11
  | 2 => 1  -- Grade 12

/-- Total sample size -/
def sample_size : ℕ := 45

/-- Calculates the number of students sampled from a specific grade -/
def students_sampled (grade : Fin 3) : ℕ :=
  (sample_size * grade_ratio grade) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

/-- Theorem stating that the number of grade 10 students in the sample is 18 -/
theorem grade_10_sample_size :
  students_sampled 0 = 18 := by sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l1385_138519


namespace NUMINAMATH_CALUDE_bicycling_time_l1385_138501

-- Define the distance in kilometers
def distance : ℝ := 96

-- Define the speed in kilometers per hour
def speed : ℝ := 6

-- Theorem: The time taken is 16 hours
theorem bicycling_time : distance / speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_bicycling_time_l1385_138501


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1385_138503

/-- Given two individuals with different tax rates and income levels, 
    calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (income_ratio : ℚ) : 
  mork_rate = 45/100 → 
  mindy_rate = 25/100 → 
  income_ratio = 4 → 
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1385_138503


namespace NUMINAMATH_CALUDE_academy_league_games_l1385_138570

/-- The number of teams in the Academy League -/
def num_teams : ℕ := 8

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- Calculates the total number of games in a season for the Academy League -/
def total_games (n : ℕ) (nc : ℕ) : ℕ :=
  (n * (n - 1)) + (n * nc)

/-- Theorem stating that the total number of games in the Academy League season is 104 -/
theorem academy_league_games :
  total_games num_teams non_conference_games = 104 := by
  sorry

end NUMINAMATH_CALUDE_academy_league_games_l1385_138570


namespace NUMINAMATH_CALUDE_parabola_with_directrix_y_2_l1385_138559

/-- Represents a parabola in 2D space -/
structure Parabola where
  /-- The equation of the parabola in the form x² = ky, where k is a non-zero real number -/
  equation : ℝ → ℝ → Prop

/-- Represents the directrix of a parabola -/
structure Directrix where
  /-- The y-coordinate of the horizontal directrix -/
  y : ℝ

/-- 
Given a parabola with a horizontal directrix y = 2, 
prove that its standard equation is x² = -8y 
-/
theorem parabola_with_directrix_y_2 (p : Parabola) (d : Directrix) :
  d.y = 2 → p.equation = fun x y ↦ x^2 = -8*y := by
  sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_y_2_l1385_138559


namespace NUMINAMATH_CALUDE_second_half_speed_l1385_138590

theorem second_half_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 400) 
  (h2 : total_time = 30) 
  (h3 : first_half_speed = 20) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 10 :=
sorry

end NUMINAMATH_CALUDE_second_half_speed_l1385_138590


namespace NUMINAMATH_CALUDE_equation_solution_l1385_138589

theorem equation_solution : 
  ∃ y : ℝ, 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) ∧ y = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1385_138589


namespace NUMINAMATH_CALUDE_remaining_for_coffee_l1385_138556

def initial_amount : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_price : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6

def total_spent : ℝ :=
  celery_cost +
  (cereal_original_price * (1 - cereal_discount)) +
  bread_cost +
  (milk_original_price * (1 - milk_discount)) +
  (potato_cost * potato_quantity)

theorem remaining_for_coffee :
  initial_amount - total_spent = 26 :=
sorry

end NUMINAMATH_CALUDE_remaining_for_coffee_l1385_138556
