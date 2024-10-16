import Mathlib

namespace NUMINAMATH_CALUDE_cubic_quadratic_relation_l1440_144045

theorem cubic_quadratic_relation (A B C D : ℝ) (u v w : ℝ) (p q : ℝ) :
  (A * u^3 + B * u^2 + C * u + D = 0) →
  (A * v^3 + B * v^2 + C * v + D = 0) →
  (A * w^3 + B * w^2 + C * w + D = 0) →
  (u^2 + p * u^2 + q = 0) →
  (v^2 + p * v^2 + q = 0) →
  (p = (B^2 - 2*C) / A^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_relation_l1440_144045


namespace NUMINAMATH_CALUDE_solve_q_l1440_144004

theorem solve_q (p q : ℝ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p*q = 6) : 
  q = (9 + Real.sqrt 57) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_q_l1440_144004


namespace NUMINAMATH_CALUDE_factory_comparison_l1440_144010

def factoryA : List ℝ := [3, 5, 6, 7, 7, 8, 8, 8, 9, 10]
def factoryB : List ℝ := [4, 6, 6, 7, 8, 8, 8, 8, 8, 8]

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def range (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem factory_comparison :
  let x₁ := mode factoryA
  let y₁ := mode factoryB
  let x₂ := mean factoryA
  let y₂ := mean factoryB
  let x₃ := range factoryA
  let y₃ := range factoryB
  let x₄ := variance factoryA
  let y₄ := variance factoryB
  (x₁ = y₁) ∧ (x₂ = y₂) ∧ (x₃ > y₃) ∧ (x₄ > y₄) := by sorry

end NUMINAMATH_CALUDE_factory_comparison_l1440_144010


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1440_144008

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∃ x : ℝ, f' a b x = 0 ∧ x = 2) →
  (f' a b 1 = -3) →
  (a = -1 ∧ b = 0) ∧
  (∃ x_max x_min : ℝ, f (-1) 0 c x_max - f (-1) 0 c x_min = 4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1440_144008


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1440_144061

theorem circle_radius_proof (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1440_144061


namespace NUMINAMATH_CALUDE_multiply_98_by_98_l1440_144062

theorem multiply_98_by_98 : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_by_98_l1440_144062


namespace NUMINAMATH_CALUDE_one_acrobat_l1440_144052

/-- Represents the count of animals at the zoo -/
structure ZooCount where
  acrobats : ℕ
  elephants : ℕ
  monkeys : ℕ

/-- Checks if the given ZooCount satisfies the conditions of the problem -/
def isValidCount (count : ZooCount) : Prop :=
  2 * count.acrobats + 4 * count.elephants + 2 * count.monkeys = 134 ∧
  count.acrobats + count.elephants + count.monkeys = 45

/-- Theorem stating that there is exactly one acrobat in the valid zoo count -/
theorem one_acrobat :
  ∃! (count : ZooCount), isValidCount count ∧ count.acrobats = 1 := by
  sorry

#check one_acrobat

end NUMINAMATH_CALUDE_one_acrobat_l1440_144052


namespace NUMINAMATH_CALUDE_product_remainder_mod_25_l1440_144054

theorem product_remainder_mod_25 : (1523 * 1857 * 1919 * 2012) % 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_25_l1440_144054


namespace NUMINAMATH_CALUDE_ellipse_slope_l1440_144050

theorem ellipse_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e : ℝ := 1/3
  let c : ℝ := a * e
  let k : ℝ := (b^2/a) / (c - (-a))
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    (x = -a ∧ y = 0) ∨ (x = c ∧ y = b^2/a)) →
  k = 2/3 := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_l1440_144050


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l1440_144040

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h_total : total_players = 70)
  (h_throwers : throwers = 49)
  (h_throwers_right_handed : throwers ≤ total_players)
  (h_non_throwers_division : (total_players - throwers) % 3 = 0)
  : throwers + ((total_players - throwers) * 2 / 3) = 63 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l1440_144040


namespace NUMINAMATH_CALUDE_positive_integer_solutions_for_equation_l1440_144020

theorem positive_integer_solutions_for_equation :
  ∀ m n : ℕ+,
  m^2 = n^2 + m + n + 2018 ↔ (m = 1010 ∧ n = 1008) ∨ (m = 506 ∧ n = 503) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_for_equation_l1440_144020


namespace NUMINAMATH_CALUDE_banana_arrangements_l1440_144063

def word : String := "BANANA"

/-- The number of unique arrangements of letters in the word -/
def num_arrangements (w : String) : ℕ := sorry

theorem banana_arrangements :
  num_arrangements word = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1440_144063


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1440_144028

theorem polynomial_coefficient_equality 
  (a b c d : ℚ) :
  (∀ x : ℚ, (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) = 
    18 * x^6 - 2 * x^5 + 16 * x^4 - 28/3 * x^3 + 8/3 * x^2 - 4 * x + 2) →
  b = -1/3 ∧ c = 14/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1440_144028


namespace NUMINAMATH_CALUDE_yvonne_success_probability_l1440_144029

theorem yvonne_success_probability 
  (p_xavier : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_yvonne_not_zelda : ℝ) 
  (h1 : p_xavier = 1/3) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_yvonne_not_zelda = 0.0625) : 
  ∃ p_yvonne : ℝ, p_yvonne = 0.5 ∧ 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_yvonne_not_zelda :=
by sorry

end NUMINAMATH_CALUDE_yvonne_success_probability_l1440_144029


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1440_144024

def f (a c x : ℝ) : ℝ := x^2 - a*x + c

theorem quadratic_function_properties (a c : ℝ) :
  (∀ x, f a c x > 1 ↔ x < -1 ∨ x > 3) →
  (∀ x m, m^2 - 4*m < f a c (2^x)) →
  (∀ x₁ x₂, x₁ ∈ [-1, 5] → x₂ ∈ [-1, 5] → |f a c x₁ - f a c x₂| ≤ 10) →
  (a = 2 ∧ c = -2) ∧
  (∀ m, m > 1 ∧ m < 3) ∧
  (a ≥ 10 - 2*Real.sqrt 10 ∧ a ≤ -2 + 2*Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1440_144024


namespace NUMINAMATH_CALUDE_cake_muffin_mix_probability_l1440_144059

theorem cake_muffin_mix_probability :
  ∀ (total buyers cake_buyers muffin_buyers both_buyers : ℕ),
    total = 100 →
    cake_buyers = 50 →
    muffin_buyers = 40 →
    both_buyers = 18 →
    (total - (cake_buyers + muffin_buyers - both_buyers)) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_probability_l1440_144059


namespace NUMINAMATH_CALUDE_number_less_than_hundred_million_l1440_144009

theorem number_less_than_hundred_million :
  ∃ x : ℕ,
    x < 100000000 ∧
    x + 1000000 = 100000000 ∧
    x = 99000000 ∧
    x / 1000000 = 99 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_hundred_million_l1440_144009


namespace NUMINAMATH_CALUDE_joey_study_time_l1440_144088

/-- Calculates the total study time for Joey's SAT exam preparation --/
theorem joey_study_time (weekday_hours : ℕ) (weekday_nights : ℕ) (weekend_hours : ℕ) (weekend_days : ℕ) (weeks : ℕ) : 
  weekday_hours = 2 →
  weekday_nights = 5 →
  weekend_hours = 3 →
  weekend_days = 2 →
  weeks = 6 →
  (weekday_hours * weekday_nights + weekend_hours * weekend_days) * weeks = 96 := by
  sorry

#check joey_study_time

end NUMINAMATH_CALUDE_joey_study_time_l1440_144088


namespace NUMINAMATH_CALUDE_divide_n_plus_one_l1440_144048

theorem divide_n_plus_one (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divide_n_plus_one_l1440_144048


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1440_144032

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1440_144032


namespace NUMINAMATH_CALUDE_exists_four_digit_number_divisible_by_101_when_reversed_l1440_144089

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number has distinct non-zero digits -/
def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ ((n / 100) % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ (n / 10) % 10) ∧ (n % 10 ≠ (n / 100) % 10) ∧ (n % 10 ≠ n / 1000) ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ ((n / 10) % 10 ≠ n / 1000) ∧
  ((n / 100) % 10 ≠ n / 1000)

theorem exists_four_digit_number_divisible_by_101_when_reversed :
  ∃ n : ℕ, has_distinct_nonzero_digits n ∧ (n + reverse n) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_four_digit_number_divisible_by_101_when_reversed_l1440_144089


namespace NUMINAMATH_CALUDE_hard_candy_coloring_is_30_l1440_144037

/-- The amount of food colouring used for each lollipop in milliliters -/
def lollipop_coloring : ℕ := 8

/-- The number of lollipops made in a day -/
def lollipops_made : ℕ := 150

/-- The number of hard candies made in a day -/
def hard_candies_made : ℕ := 20

/-- The total amount of food colouring used in a day in milliliters -/
def total_coloring : ℕ := 1800

/-- The amount of food colouring needed for each hard candy in milliliters -/
def hard_candy_coloring : ℕ := (total_coloring - lollipop_coloring * lollipops_made) / hard_candies_made

theorem hard_candy_coloring_is_30 : hard_candy_coloring = 30 := by
  sorry

end NUMINAMATH_CALUDE_hard_candy_coloring_is_30_l1440_144037


namespace NUMINAMATH_CALUDE_unoccupied_business_seats_count_l1440_144022

/-- Represents the seating configuration and occupancy of an airplane. -/
structure AirplaneSeating where
  firstClassSeats : ℕ
  businessClassSeats : ℕ
  economyClassSeats : ℕ
  firstClassOccupied : ℕ
  economyClassOccupied : ℕ
  businessAndFirstOccupied : ℕ

/-- Calculates the number of unoccupied seats in business class. -/
def unoccupiedBusinessSeats (a : AirplaneSeating) : ℕ :=
  a.businessClassSeats - (a.businessAndFirstOccupied - a.firstClassOccupied)

/-- Theorem stating the number of unoccupied seats in business class. -/
theorem unoccupied_business_seats_count
  (a : AirplaneSeating)
  (h1 : a.firstClassSeats = 10)
  (h2 : a.businessClassSeats = 30)
  (h3 : a.economyClassSeats = 50)
  (h4 : a.economyClassOccupied = a.economyClassSeats / 2)
  (h5 : a.businessAndFirstOccupied = a.economyClassOccupied)
  (h6 : a.firstClassOccupied = 3) :
  unoccupiedBusinessSeats a = 8 := by
  sorry

#eval unoccupiedBusinessSeats {
  firstClassSeats := 10,
  businessClassSeats := 30,
  economyClassSeats := 50,
  firstClassOccupied := 3,
  economyClassOccupied := 25,
  businessAndFirstOccupied := 25
}

end NUMINAMATH_CALUDE_unoccupied_business_seats_count_l1440_144022


namespace NUMINAMATH_CALUDE_tan_a_pi_third_equals_sqrt_three_l1440_144000

-- Define the function for logarithm with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem tan_a_pi_third_equals_sqrt_three 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log_base a 16 = 2) : 
  Real.tan (a * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_pi_third_equals_sqrt_three_l1440_144000


namespace NUMINAMATH_CALUDE_det_A_eq_zero_iff_x_eq_52_19_l1440_144006

def A (x : ℚ) : Matrix (Fin 3) (Fin 3) ℚ :=
  !![3, 1, -1;
     4, x, 2;
     1, 3, 6]

theorem det_A_eq_zero_iff_x_eq_52_19 :
  ∀ x : ℚ, Matrix.det (A x) = 0 ↔ x = 52 / 19 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_zero_iff_x_eq_52_19_l1440_144006


namespace NUMINAMATH_CALUDE_remaining_payment_example_l1440_144064

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem stating that the remaining payment is $1350 given a 10% deposit of $150 -/
theorem remaining_payment_example : remaining_payment (10 / 100) 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_example_l1440_144064


namespace NUMINAMATH_CALUDE_floral_shop_sale_l1440_144025

/-- Represents the number of bouquets sold on each day of a three-day sale. -/
structure SaleData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Theorem stating the conditions of the sale and the result to be proven. -/
theorem floral_shop_sale (sale : SaleData) : 
  sale.tuesday = 3 * sale.monday ∧ 
  sale.wednesday = sale.tuesday / 3 ∧
  sale.monday + sale.tuesday + sale.wednesday = 60 →
  sale.monday = 12 := by
  sorry

end NUMINAMATH_CALUDE_floral_shop_sale_l1440_144025


namespace NUMINAMATH_CALUDE_probability_yellow_second_is_67_135_l1440_144077

/-- Represents the contents of a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Bag X contents -/
def bagX : Bag := { white := 4, black := 5 }

/-- Bag Y contents -/
def bagY : Bag := { yellow := 7, blue := 3 }

/-- Bag Z contents -/
def bagZ : Bag := { yellow := 3, blue := 6 }

/-- Probability of drawing a yellow marble as the second marble -/
def probabilityYellowSecond : ℚ :=
  (bagX.white * bagY.yellow) / (bagX.total * bagY.total) +
  (bagX.black * bagZ.yellow) / (bagX.total * bagZ.total)

theorem probability_yellow_second_is_67_135 : probabilityYellowSecond = 67 / 135 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_second_is_67_135_l1440_144077


namespace NUMINAMATH_CALUDE_f_above_g_implies_m_less_than_5_l1440_144047

/-- The function f(x) = |x - 2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x + 3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem: If f(x) is always above g(x) for all real x, then m < 5 -/
theorem f_above_g_implies_m_less_than_5 (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry


end NUMINAMATH_CALUDE_f_above_g_implies_m_less_than_5_l1440_144047


namespace NUMINAMATH_CALUDE_min_tan_angle_ocular_rays_l1440_144017

def G : Set (ℕ × ℕ) := {p | p.1 ≤ 20 ∧ p.2 ≤ 20 ∧ p.1 > 0 ∧ p.2 > 0}

def isOcularRay (m : ℚ) : Prop := ∃ p ∈ G, m = p.2 / p.1

def tanAngleBetweenRays (m1 m2 : ℚ) : ℚ := |m1 - m2| / (1 + m1 * m2)

def A : Set ℚ := {a | ∃ m1 m2, isOcularRay m1 ∧ isOcularRay m2 ∧ m1 ≠ m2 ∧ a = tanAngleBetweenRays m1 m2}

theorem min_tan_angle_ocular_rays :
  ∃ a ∈ A, a = (1 : ℚ) / 722 ∧ ∀ b ∈ A, (1 : ℚ) / 722 ≤ b :=
sorry

end NUMINAMATH_CALUDE_min_tan_angle_ocular_rays_l1440_144017


namespace NUMINAMATH_CALUDE_original_number_proof_l1440_144035

theorem original_number_proof (r : ℝ) : 
  r * (1 + 0.125) - r * (1 - 0.25) = 30 → r = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1440_144035


namespace NUMINAMATH_CALUDE_coin_problem_l1440_144083

theorem coin_problem (total_coins : ℕ) (total_value : ℕ) :
  total_coins = 30 →
  total_value = 86 →
  ∃ (five_jiao : ℕ) (one_jiao : ℕ),
    five_jiao + one_jiao = total_coins ∧
    5 * five_jiao + one_jiao = total_value ∧
    five_jiao = 14 ∧
    one_jiao = 16 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l1440_144083


namespace NUMINAMATH_CALUDE_fraction_simplification_l1440_144021

/-- Given a, b, c, x, y, z are real numbers, prove that the given complex fraction 
    is equal to the simplified form. -/
theorem fraction_simplification (a b c x y z : ℝ) :
  (c * z * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + 
   b * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * z + b * z) = 
  a^3 * x^3 + c^3 * z^3 + (3 * c * z * a^3 * y^3 + 3 * b * z * c^3 * x^3) / (c * z + b * z) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1440_144021


namespace NUMINAMATH_CALUDE_find_s_value_l1440_144069

/-- Given a function g(x) = 3x^4 + 2x^3 - x^2 - 4x + s, 
    prove that s = -4 when g(-1) = 0 -/
theorem find_s_value (s : ℝ) : 
  (let g := λ x : ℝ => 3*x^4 + 2*x^3 - x^2 - 4*x + s
   g (-1) = 0) → s = -4 := by
  sorry

end NUMINAMATH_CALUDE_find_s_value_l1440_144069


namespace NUMINAMATH_CALUDE_correct_answer_l1440_144012

/-- Represents the current ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  (ages.john - 3 = 2 * (ages.mary - 3)) ∧
  (ages.john - 5 = 3 * (ages.mary - 5))

/-- The function to calculate the number of years until the ratio of ages is 3:2 -/
def yearsUntilRatio (ages : Ages) : ℕ :=
  let x : ℕ := 1  -- We claim this is the answer
  x

/-- The theorem stating that the answer is correct -/
theorem correct_answer (ages : Ages) (h : satisfiesConditions ages) : 
  (ages.john + yearsUntilRatio ages) * 2 = (ages.mary + yearsUntilRatio ages) * 3 := by
  sorry

#check correct_answer

end NUMINAMATH_CALUDE_correct_answer_l1440_144012


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1440_144026

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Hyperbola structure -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / 3 - y^2 = 1

/-- The theorem statement -/
theorem parabola_focus_distance (parab : Parabola) (hyper : Hyperbola) :
  (parab.equation 2 0 → hyper.equation 2 0) →  -- The foci coincide at (2, 0)
  (∀ b : ℝ, parab.equation 2 b →               -- For any point (2, b) on the parabola
    (2 - parab.p / 2)^2 + b^2 = 4^2) :=        -- The distance to the focus is 4
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1440_144026


namespace NUMINAMATH_CALUDE_meeting_arrangements_count_l1440_144093

/-- Represents the number of schools in the community -/
def num_schools : ℕ := 4

/-- Represents the number of members in each school -/
def members_per_school : ℕ := 6

/-- Represents the number of representatives each school sends -/
def reps_per_school : ℕ := 2

/-- The number of ways to arrange the leadership meeting -/
def meeting_arrangements : ℕ := num_schools * (members_per_school.choose reps_per_school) * (members_per_school.choose reps_per_school)^(num_schools - 1)

/-- Theorem stating that the number of meeting arrangements is 202500 -/
theorem meeting_arrangements_count : meeting_arrangements = 202500 := by
  sorry

end NUMINAMATH_CALUDE_meeting_arrangements_count_l1440_144093


namespace NUMINAMATH_CALUDE_robins_full_pages_l1440_144015

/-- The number of full pages in a photo album -/
def full_pages (total_photos : ℕ) (photos_per_page : ℕ) : ℕ :=
  total_photos / photos_per_page

/-- Theorem: Robin's photo album has 181 full pages -/
theorem robins_full_pages :
  full_pages 2176 12 = 181 := by
  sorry

end NUMINAMATH_CALUDE_robins_full_pages_l1440_144015


namespace NUMINAMATH_CALUDE_b_joined_after_five_months_l1440_144095

/-- Represents the number of months after A started the business that B joined as a partner. -/
def months_before_b_joined : ℕ := 5

/-- Represents A's initial investment in rupees. -/
def a_investment : ℕ := 3500

/-- Represents B's investment in rupees. -/
def b_investment : ℕ := 9000

/-- Represents the total number of months in a year. -/
def months_in_year : ℕ := 12

/-- Theorem stating that B joined the business 5 months after A started, given the conditions. -/
theorem b_joined_after_five_months :
  let a_capital := a_investment * months_in_year
  let b_capital := b_investment * (months_in_year - months_before_b_joined)
  (a_capital : ℚ) / b_capital = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_b_joined_after_five_months_l1440_144095


namespace NUMINAMATH_CALUDE_dormitory_places_l1440_144078

theorem dormitory_places : ∃ (x y : ℕ),
  (2 * x + 3 * y > 30) ∧
  (2 * x + 3 * y < 70) ∧
  (4 * (2 * x + 3 * y) = 5 * (3 * x + 2 * y)) ∧
  (2 * x + 3 * y = 50) :=
by sorry

end NUMINAMATH_CALUDE_dormitory_places_l1440_144078


namespace NUMINAMATH_CALUDE_intersection_sequence_correct_l1440_144016

def A : Set ℕ := {n | ∃ m : ℕ+, n = m * (m + 1)}
def B : Set ℕ := {n | ∃ m : ℕ+, n = 3 * m - 1}

def intersection_sequence (k : ℕ+) : ℕ := 9 * k^2 - 9 * k + 2

theorem intersection_sequence_correct :
  ∀ k : ℕ+, (intersection_sequence k) ∈ A ∩ B ∧
  (∀ n ∈ A ∩ B, n < intersection_sequence k → 
    ∃ j : ℕ+, j < k ∧ n = intersection_sequence j) :=
sorry

end NUMINAMATH_CALUDE_intersection_sequence_correct_l1440_144016


namespace NUMINAMATH_CALUDE_middle_number_in_ratio_l1440_144038

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ b / c = 2 / 5 ∧ a^2 + b^2 + c^2 = 1862 → b = 14 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_in_ratio_l1440_144038


namespace NUMINAMATH_CALUDE_class_book_count_l1440_144039

/-- Calculates the total number of books a class has from the library --/
def totalBooks (initial borrowed₁ returned borrowed₂ : ℕ) : ℕ :=
  initial + borrowed₁ - returned + borrowed₂

/-- Theorem: The class currently has 80 books from the library --/
theorem class_book_count :
  totalBooks 54 23 12 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_book_count_l1440_144039


namespace NUMINAMATH_CALUDE_joannes_weekly_earnings_is_812_48_l1440_144084

/-- Calculates Joanne's weekly earnings after deductions, bonuses, and allowances -/
def joannes_weekly_earnings : ℝ :=
  let main_job_hours : ℝ := 8 * 5
  let main_job_rate : ℝ := 16
  let main_job_base_pay : ℝ := main_job_hours * main_job_rate
  let main_job_bonus_rate : ℝ := 0.1
  let main_job_bonus : ℝ := main_job_base_pay * main_job_bonus_rate
  let main_job_total : ℝ := main_job_base_pay + main_job_bonus
  let main_job_deduction_rate : ℝ := 0.05
  let main_job_deduction : ℝ := main_job_total * main_job_deduction_rate
  let main_job_net : ℝ := main_job_total - main_job_deduction

  let part_time_regular_hours : ℝ := 2 * 4
  let part_time_friday_hours : ℝ := 3
  let part_time_rate : ℝ := 13.5
  let part_time_friday_bonus : ℝ := 2
  let part_time_regular_pay : ℝ := part_time_regular_hours * part_time_rate
  let part_time_friday_pay : ℝ := part_time_friday_hours * (part_time_rate + part_time_friday_bonus)
  let part_time_total : ℝ := part_time_regular_pay + part_time_friday_pay
  let part_time_deduction_rate : ℝ := 0.07
  let part_time_deduction : ℝ := part_time_total * part_time_deduction_rate
  let part_time_net : ℝ := part_time_total - part_time_deduction

  main_job_net + part_time_net

/-- Theorem: Joanne's weekly earnings after deductions, bonuses, and allowances is $812.48 -/
theorem joannes_weekly_earnings_is_812_48 : joannes_weekly_earnings = 812.48 := by
  sorry

end NUMINAMATH_CALUDE_joannes_weekly_earnings_is_812_48_l1440_144084


namespace NUMINAMATH_CALUDE_total_days_proof_l1440_144073

def total_days (x y z t : ℕ) : ℕ := (x + y + z) * t

theorem total_days_proof (x y z t : ℕ) : 
  total_days x y z t = (x + y + z) * t := by
  sorry

end NUMINAMATH_CALUDE_total_days_proof_l1440_144073


namespace NUMINAMATH_CALUDE_shanghai_masters_matches_l1440_144097

/-- Represents the Shanghai Masters tennis tournament structure -/
structure ShangHaiMasters where
  totalPlayers : Nat
  groupCount : Nat
  playersPerGroup : Nat
  advancingPerGroup : Nat

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the Shanghai Masters tournament -/
def totalMatches (tournament : ShangHaiMasters) : Nat :=
  let groupMatches := tournament.groupCount * roundRobinMatches tournament.playersPerGroup
  let knockoutMatches := tournament.groupCount * tournament.advancingPerGroup / 2
  let finalMatches := 2
  groupMatches + knockoutMatches + finalMatches

/-- Theorem stating that the total number of matches in the Shanghai Masters is 16 -/
theorem shanghai_masters_matches :
  ∃ (tournament : ShangHaiMasters),
    tournament.totalPlayers = 8 ∧
    tournament.groupCount = 2 ∧
    tournament.playersPerGroup = 4 ∧
    tournament.advancingPerGroup = 2 ∧
    totalMatches tournament = 16 := by
  sorry


end NUMINAMATH_CALUDE_shanghai_masters_matches_l1440_144097


namespace NUMINAMATH_CALUDE_stamps_problem_l1440_144074

/-- The number of stamps Kylie and Nelly have together -/
def total_stamps (kylie_stamps : ℕ) (nelly_extra_stamps : ℕ) : ℕ :=
  kylie_stamps + (kylie_stamps + nelly_extra_stamps)

/-- Theorem: Given Kylie has 34 stamps and Nelly has 44 more stamps than Kylie,
    the total number of stamps they have together is 112. -/
theorem stamps_problem : total_stamps 34 44 = 112 := by
  sorry

end NUMINAMATH_CALUDE_stamps_problem_l1440_144074


namespace NUMINAMATH_CALUDE_tom_trout_count_l1440_144056

/-- Given that Melanie catches 8 trout and Tom catches 2 times as many trout as Melanie,
    prove that Tom catches 16 trout. -/
theorem tom_trout_count (melanie_trout : ℕ) (tom_multiplier : ℕ) 
    (h1 : melanie_trout = 8)
    (h2 : tom_multiplier = 2) : 
  tom_multiplier * melanie_trout = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_trout_count_l1440_144056


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1440_144031

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1440_144031


namespace NUMINAMATH_CALUDE_external_tangent_lines_of_circles_l1440_144005

-- Define the circles
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 9
def circle_B (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the external common tangent lines
def external_tangent_lines (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x - 3) ∨ y = -(Real.sqrt 3 / 3) * (x - 3)

-- Theorem statement
theorem external_tangent_lines_of_circles :
  ∀ x y : ℝ, (circle_A x y ∨ circle_B x y) → external_tangent_lines x y :=
by
  sorry

end NUMINAMATH_CALUDE_external_tangent_lines_of_circles_l1440_144005


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_power_fifteen_l1440_144070

/-- The sum of the tens digit and the ones digit of 7^15 is 7 -/
theorem sum_of_digits_seven_power_fifteen : ∃ (a b : ℕ), 
  7^15 % 100 = 10 * a + b ∧ a + b = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_power_fifteen_l1440_144070


namespace NUMINAMATH_CALUDE_uniqueness_not_algorithm_characteristic_l1440_144072

/-- Represents characteristics of an algorithm -/
inductive AlgorithmCharacteristic
  | Abstraction
  | Precision
  | Finiteness
  | Uniqueness

/-- Predicate to check if a given characteristic is a valid algorithm characteristic -/
def isValidAlgorithmCharacteristic (c : AlgorithmCharacteristic) : Prop :=
  match c with
  | AlgorithmCharacteristic.Abstraction => True
  | AlgorithmCharacteristic.Precision => True
  | AlgorithmCharacteristic.Finiteness => True
  | AlgorithmCharacteristic.Uniqueness => False

theorem uniqueness_not_algorithm_characteristic :
  ¬(isValidAlgorithmCharacteristic AlgorithmCharacteristic.Uniqueness) :=
by sorry

end NUMINAMATH_CALUDE_uniqueness_not_algorithm_characteristic_l1440_144072


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1440_144066

-- Define the parabola and line
def parabola (b x : ℝ) : ℝ := b * x^2 + 4
def line (x : ℝ) : ℝ := 2 * x + 2

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop :=
  ∃! x, parabola b x = line x

-- Theorem statement
theorem parabola_tangent_to_line :
  ∀ b : ℝ, is_tangent b → b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1440_144066


namespace NUMINAMATH_CALUDE_equation_solution_l1440_144042

theorem equation_solution (x y : ℝ) : y^2 = 4*y - Real.sqrt (x - 3) - 4 → x + 2*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1440_144042


namespace NUMINAMATH_CALUDE_final_class_size_l1440_144003

theorem final_class_size (initial_size second_year_join final_year_leave : ℕ) :
  initial_size = 150 →
  second_year_join = 30 →
  final_year_leave = 15 →
  initial_size + second_year_join - final_year_leave = 165 := by
  sorry

end NUMINAMATH_CALUDE_final_class_size_l1440_144003


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l1440_144076

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l1440_144076


namespace NUMINAMATH_CALUDE_binomial_19_10_l1440_144068

theorem binomial_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end NUMINAMATH_CALUDE_binomial_19_10_l1440_144068


namespace NUMINAMATH_CALUDE_ratio_equality_l1440_144014

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1440_144014


namespace NUMINAMATH_CALUDE_stratified_sampling_management_l1440_144081

theorem stratified_sampling_management (total_employees : ℕ) (management_personnel : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : management_personnel = 32)
  (h3 : sample_size = 20) :
  (management_personnel : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_management_l1440_144081


namespace NUMINAMATH_CALUDE_bca_equals_341_l1440_144018

def repeating_decimal_bc (b c : ℕ) : ℚ :=
  (10 * b + c : ℚ) / 99

def repeating_decimal_bcabc (b c a : ℕ) : ℚ :=
  (10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 99999

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem bca_equals_341 (b c a : ℕ) 
  (hb : is_digit b) (hc : is_digit c) (ha : is_digit a)
  (h_eq : repeating_decimal_bc b c + repeating_decimal_bcabc b c a = 41 / 111) :
  100 * b + 10 * c + a = 341 := by
sorry

end NUMINAMATH_CALUDE_bca_equals_341_l1440_144018


namespace NUMINAMATH_CALUDE_complex_modulus_one_l1440_144096

theorem complex_modulus_one (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I * 2) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l1440_144096


namespace NUMINAMATH_CALUDE_guaranteed_scores_theorem_l1440_144099

/-- Represents a player in the card game -/
inductive Player : Type
| First : Player
| Second : Player

/-- The card game with given conditions -/
structure CardGame where
  first_player_cards : Finset Nat
  second_player_cards : Finset Nat
  total_turns : Nat

/-- Define the game with the given conditions -/
def game : CardGame :=
  { first_player_cards := Finset.range 1000 |>.image (fun n => 2 * n + 2),
    second_player_cards := Finset.range 1001 |>.image (fun n => 2 * n + 1),
    total_turns := 1000 }

/-- The score a player can guarantee for themselves -/
def guaranteed_score (player : Player) (g : CardGame) : Nat :=
  match player with
  | Player.First => g.total_turns - 1
  | Player.Second => 1

/-- Theorem stating the guaranteed scores for both players -/
theorem guaranteed_scores_theorem (g : CardGame) :
  (guaranteed_score Player.First g = 999) ∧
  (guaranteed_score Player.Second g = 1) :=
sorry

end NUMINAMATH_CALUDE_guaranteed_scores_theorem_l1440_144099


namespace NUMINAMATH_CALUDE_no_interior_points_with_sum_20_l1440_144060

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A circle with center at the origin and radius 2 -/
def insideCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 < 4

theorem no_interior_points_with_sum_20 :
  ¬ ∃ (p : Point), insideCircle p ∧
    ∃ (a b : Point), 
      a.x^2 + a.y^2 = 4 ∧ 
      b.x^2 + b.y^2 = 4 ∧ 
      a.x = -b.x ∧ 
      a.y = -b.y ∧
      distSquared p a + distSquared p b = 20 :=
by sorry

end NUMINAMATH_CALUDE_no_interior_points_with_sum_20_l1440_144060


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l1440_144094

def history_class (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ) : ℕ :=
  total_boys - (students_with_notebooks - girls_with_notebooks)

theorem boys_without_notebooks :
  history_class 16 20 11 = 7 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l1440_144094


namespace NUMINAMATH_CALUDE_solution_problem_l1440_144007

theorem solution_problem (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_problem_l1440_144007


namespace NUMINAMATH_CALUDE_prob_condition_one_before_two_l1440_144019

/-- Represents the state of ball draws as a sorted list of integers -/
def DrawState := List Nat

/-- The probability of reaching a certain draw state -/
def StateProbability := DrawState → ℚ

/-- Checks if some ball has been drawn at least three times -/
def conditionOne (state : DrawState) : Prop :=
  state.head! ≥ 3

/-- Checks if every ball has been drawn at least once -/
def conditionTwo (state : DrawState) : Prop :=
  state.length = 3 ∧ state.all (· > 0)

/-- The probability of condition one occurring before condition two -/
def probConditionOneBeforeTwo (probMap : StateProbability) : ℚ :=
  probMap [3, 0, 0] + probMap [3, 1, 0] + probMap [3, 2, 0]

theorem prob_condition_one_before_two :
  ∃ (probMap : StateProbability),
    (∀ state, conditionOne state → conditionTwo state → probMap state = 0) →
    (probMap [0, 0, 0] = 1) →
    (∀ state, probMap state ≥ 0) →
    (∀ state, probMap state ≤ 1) →
    probConditionOneBeforeTwo probMap = 13 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_condition_one_before_two_l1440_144019


namespace NUMINAMATH_CALUDE_cube_root_of_19683_l1440_144036

theorem cube_root_of_19683 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 19683) : x = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_19683_l1440_144036


namespace NUMINAMATH_CALUDE_geometric_progression_b_equals_four_l1440_144098

-- Define a geometric progression
def is_geometric_progression (seq : Fin 5 → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 4, seq (i + 1) = seq i * q

-- State the theorem
theorem geometric_progression_b_equals_four
  (seq : Fin 5 → ℝ)
  (h_gp : is_geometric_progression seq)
  (h_first : seq 0 = 1)
  (h_last : seq 4 = 16) :
  seq 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_b_equals_four_l1440_144098


namespace NUMINAMATH_CALUDE_ellipse_equation_l1440_144044

/-- Given an ellipse with focal distance 8 and the sum of distances from any point 
    on the ellipse to the two foci being 10, prove that its standard equation is 
    either x²/25 + y²/9 = 1 or y²/25 + x²/9 = 1 -/
theorem ellipse_equation (focal_distance : ℝ) (sum_distances : ℝ) 
  (h1 : focal_distance = 8) (h2 : sum_distances = 10) :
  (∃ x y : ℝ, x^2/25 + y^2/9 = 1) ∨ (∃ x y : ℝ, y^2/25 + x^2/9 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1440_144044


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l1440_144057

def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

theorem quadratic_function_satisfies_conditions :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l1440_144057


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l1440_144027

theorem necessary_and_sufficient_condition (p q : Prop) 
  (h1 : p → q) (h2 : q → p) : 
  (p ↔ q) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l1440_144027


namespace NUMINAMATH_CALUDE_triangle_area_l1440_144091

/-- Given a triangle with perimeter 48 and inradius 2.5, its area is 60 -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : P = 48) 
    (h2 : r = 2.5) 
    (h3 : A = r * P / 2) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1440_144091


namespace NUMINAMATH_CALUDE_initial_number_solution_l1440_144092

theorem initial_number_solution : 
  ∃ x : ℤ, x - 12 * 3 * 2 = 1234490 ∧ x = 1234562 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_solution_l1440_144092


namespace NUMINAMATH_CALUDE_sequence_common_difference_l1440_144079

theorem sequence_common_difference (k x a : ℝ) : 
  (20 + k = x) ∧ (50 + k = a * x) ∧ (100 + k = a^2 * x) → a = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_common_difference_l1440_144079


namespace NUMINAMATH_CALUDE_proportion_problem_l1440_144043

theorem proportion_problem (x y : ℝ) : 
  x / 5 = 5 / 6 → x = 0.9 → y / x = 5 / 6 → y = 0.75 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l1440_144043


namespace NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l1440_144055

/-- Given a cone with radius 2 inches and height 6 inches, and a sphere with the same physical volume
    but half the density of the cone's material, the radius of the sphere is ∛12 inches. -/
theorem sphere_radius_from_cone_volume (cone_radius : ℝ) (cone_height : ℝ) (sphere_radius : ℝ) :
  cone_radius = 2 →
  cone_height = 6 →
  (1 / 3) * Real.pi * cone_radius^2 * cone_height = (4 / 3) * Real.pi * sphere_radius^3 / 2 →
  sphere_radius = (12 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l1440_144055


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_500_l1440_144058

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_500 :
  (25 : ℝ) / 100 * 500 = 125 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_500_l1440_144058


namespace NUMINAMATH_CALUDE_sequence_product_l1440_144002

theorem sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n - 1) = 2 * a n) (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1440_144002


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1440_144071

theorem quadratic_roots_property : ∃ (x y : ℝ), 
  (x + y = 10) ∧ 
  (|x - y| = 6) ∧ 
  (∀ z : ℝ, z^2 - 10*z + 16 = 0 ↔ (z = x ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1440_144071


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1440_144075

theorem min_value_quadratic : 
  ∃ (min : ℝ), min = -39 ∧ ∀ (x : ℝ), x^2 + 14*x + 10 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1440_144075


namespace NUMINAMATH_CALUDE_black_pens_per_student_l1440_144001

/-- Represents the problem of calculating the number of black pens each student received. -/
theorem black_pens_per_student (num_students : ℕ) (red_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) (remaining_pens_per_student : ℕ) :
  num_students = 3 →
  red_pens_per_student = 62 →
  pens_taken_first_month = 37 →
  pens_taken_second_month = 41 →
  remaining_pens_per_student = 79 →
  (num_students * (red_pens_per_student + 43) - pens_taken_first_month - pens_taken_second_month) / num_students = remaining_pens_per_student :=
by sorry

#check black_pens_per_student

end NUMINAMATH_CALUDE_black_pens_per_student_l1440_144001


namespace NUMINAMATH_CALUDE_soccer_club_girls_l1440_144041

theorem soccer_club_girls (total : ℕ) (attendees : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 30 →
  attendees = 18 →
  attendees = boys + girls / 3 →
  total = boys + girls →
  girls = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l1440_144041


namespace NUMINAMATH_CALUDE_binomial_probability_l1440_144067

/-- A binomially distributed random variable with given mean and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 3
  var_eq : n * p * (1 - p) = 10 / 9

/-- The probability mass function for a binomial distribution -/
def binomialPMF (rv : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose rv.n k) * (rv.p ^ k) * ((1 - rv.p) ^ (rv.n - k))

theorem binomial_probability (rv : BinomialRV) : 
  binomialPMF rv 4 = 10 / 243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l1440_144067


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l1440_144023

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y - 9 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y + 1 = 0
  ∀ (P₁ : ℝ × ℝ) (P₂ : ℝ × ℝ),
  l₁ P₁.1 P₁.2 → l₂ P₂.1 P₂.2 →
  ∃ (P₁' : ℝ × ℝ) (P₂' : ℝ × ℝ),
  l₁ P₁'.1 P₁'.2 ∧ l₂ P₂'.1 P₂'.2 ∧
  Real.sqrt 10 = ‖(P₁'.1 - P₂'.1, P₁'.2 - P₂'.2)‖ ∧
  ∀ (Q₁ : ℝ × ℝ) (Q₂ : ℝ × ℝ),
  l₁ Q₁.1 Q₁.2 → l₂ Q₂.1 Q₂.2 →
  Real.sqrt 10 ≤ ‖(Q₁.1 - Q₂.1, Q₁.2 - Q₂.2)‖ :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parallel_lines_l1440_144023


namespace NUMINAMATH_CALUDE_yeast_growth_proof_l1440_144065

/-- Calculates the yeast population after a given time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval_duration : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (total_time / interval_duration)

/-- Proves that the yeast population grows to 1350 after 18 minutes -/
theorem yeast_growth_proof :
  yeast_population 50 3 5 18 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_proof_l1440_144065


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l1440_144046

theorem unique_triplet_solution (a b p : ℕ+) (h_prime : Nat.Prime p) :
  (a + b : ℕ+) ^ (p : ℕ) = p ^ (a : ℕ) + p ^ (b : ℕ) ↔ a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l1440_144046


namespace NUMINAMATH_CALUDE_find_p_l1440_144085

def U : Set ℕ := {1, 2, 3, 4}

def M (p : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + p = 0}

theorem find_p : ∃ p : ℝ, (U \ M p) = {2, 3} → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l1440_144085


namespace NUMINAMATH_CALUDE_polynomial_product_theorem_l1440_144086

theorem polynomial_product_theorem (p q : ℚ) : 
  (∀ x, (x^2 + p*x - 1/3) * (x^2 - 3*x + q) = x^4 + (q - 3*p - 1/3)*x^2 - q/3) → 
  (p = 3 ∧ q = -1/3 ∧ (-2*p^2*q)^2 + 3*p*q = 33) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_theorem_l1440_144086


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1440_144087

theorem complex_modulus_product : 
  Complex.abs ((10 - 5 * Complex.I) * (7 + 24 * Complex.I)) = 125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1440_144087


namespace NUMINAMATH_CALUDE_max_y_value_l1440_144033

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l1440_144033


namespace NUMINAMATH_CALUDE_bamboo_tube_rice_problem_l1440_144053

theorem bamboo_tube_rice_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, n < 8 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 + a 1 + a 2 = 39/10 →                         -- bottom three joints
  a 5 + a 6 + a 7 + a 8 = 3 →                       -- top four joints
  a 4 = 1 :=                                        -- middle joint
by sorry

end NUMINAMATH_CALUDE_bamboo_tube_rice_problem_l1440_144053


namespace NUMINAMATH_CALUDE_ellipse_rolling_conditions_l1440_144080

/-- 
An ellipse with semi-axes a and b rolls without slipping on the curve y = c sin(x/a) 
and completes one revolution in one period of the sine curve. 
This theorem states the conditions that a, b, and c must satisfy.
-/
theorem ellipse_rolling_conditions 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c ≠ 0) 
  (h_ellipse : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) 
  (h_curve : ∀ (x : ℝ), ∃ (y : ℝ), y = c * Real.sin (x / a)) 
  (h_roll : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ y = c * Real.sin (x / a)) 
  (h_period : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), c * Real.sin (x / a) = c * Real.sin ((x + T) / a)) :
  b ≥ a ∧ c^2 = b^2 - a^2 ∧ c * b^2 < a^3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_rolling_conditions_l1440_144080


namespace NUMINAMATH_CALUDE_inequality_proof_l1440_144030

theorem inequality_proof (a b : ℝ) : (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1440_144030


namespace NUMINAMATH_CALUDE_pentagon_count_l1440_144090

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons that can be formed
    by selecting 5 points from 15 distinct points on the circumference of a circle
    is equal to 3003 -/
theorem pentagon_count :
  binomial num_points pentagon_vertices = 3003 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_count_l1440_144090


namespace NUMINAMATH_CALUDE_opposite_hands_theorem_l1440_144049

/-- The time in minutes past 10:00 when the minute hand will be exactly opposite
    the place where the hour hand was four minutes ago, eight minutes from now. -/
def opposite_hands_time : ℝ :=
  let t : ℝ := 29.09090909090909  -- Approximate value of 29 1/11
  t

/-- Theorem stating that the calculated time satisfies the given conditions -/
theorem opposite_hands_theorem :
  let t := opposite_hands_time
  -- Time is between 10:00 and 11:00
  0 < t ∧ t < 60 ∧
  -- Minute hand position 8 minutes from now
  let minute_pos := 6 * (t + 8)
  -- Hour hand position 4 minutes ago
  let hour_pos := 30 + 0.5 * (t - 4)
  -- Hands are opposite (180 degrees apart)
  |minute_pos - hour_pos| = 180 := by
  sorry

#eval opposite_hands_time

end NUMINAMATH_CALUDE_opposite_hands_theorem_l1440_144049


namespace NUMINAMATH_CALUDE_age_multiple_l1440_144051

def rons_current_age : ℕ := 43
def maurices_current_age : ℕ := 7
def years_passed : ℕ := 5

theorem age_multiple : 
  (rons_current_age + years_passed) / (maurices_current_age + years_passed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_multiple_l1440_144051


namespace NUMINAMATH_CALUDE_equation_not_linear_l1440_144082

/-- A linear equation in two variables contains exactly two variables and the highest degree of terms involving these variables is 1. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

/-- The equation xy = 3 -/
def equation (x y : ℝ) : ℝ := x * y - 3

theorem equation_not_linear : ¬ is_linear_equation_in_two_variables equation := by
  sorry

end NUMINAMATH_CALUDE_equation_not_linear_l1440_144082


namespace NUMINAMATH_CALUDE_tripod_height_after_damage_l1440_144013

/-- Represents the height of a tripod after one leg is shortened -/
def tripod_height (leg_length : ℝ) (initial_height : ℝ) (shortened_length : ℝ) : ℝ :=
  -- Define the function to calculate the new height
  sorry

theorem tripod_height_after_damage :
  let leg_length : ℝ := 6
  let initial_height : ℝ := 5
  let shortened_length : ℝ := 1
  tripod_height leg_length initial_height shortened_length = 5 := by
  sorry

#check tripod_height_after_damage

end NUMINAMATH_CALUDE_tripod_height_after_damage_l1440_144013


namespace NUMINAMATH_CALUDE_cloth_price_calculation_l1440_144011

/-- The original cost price of one metre of cloth before discount -/
def original_price : ℝ := 95

/-- The number of metres of cloth sold -/
def metres_sold : ℝ := 200

/-- The selling price after discount for all metres sold -/
def selling_price : ℝ := 18000

/-- The loss per metre -/
def loss_per_metre : ℝ := 5

/-- The discount percentage -/
def discount_percent : ℝ := 10

theorem cloth_price_calculation :
  (metres_sold * original_price * (1 - discount_percent / 100) = selling_price) ∧
  (original_price - loss_per_metre = selling_price / metres_sold) := by
  sorry

end NUMINAMATH_CALUDE_cloth_price_calculation_l1440_144011


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_four_l1440_144034

theorem subset_implies_a_geq_four (a : ℝ) :
  let A : Set ℝ := {x | 1 < x ∧ x < 2}
  let B : Set ℝ := {x | x^2 - a*x + 3 ≤ 0}
  A ⊆ B → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_four_l1440_144034
