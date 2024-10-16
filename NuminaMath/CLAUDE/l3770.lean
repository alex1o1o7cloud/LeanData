import Mathlib

namespace NUMINAMATH_CALUDE_find_other_number_l3770_377090

theorem find_other_number (A B : ℕ+) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3770_377090


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3770_377082

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 7) 
  (h_S3 : a 0 + a 1 + a 2 = 21) : 
  (a 1 / a 0 = 1) ∨ (a 1 / a 0 = -1/2) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3770_377082


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_l3770_377033

theorem smallest_four_digit_divisible_by_four : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 → n ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_l3770_377033


namespace NUMINAMATH_CALUDE_max_value_m_l3770_377083

/-- Represents a number in base 8 as XYZ₈ -/
def base8_repr (X Y Z : ℕ) : ℕ := 64 * X + 8 * Y + Z

/-- Represents a number in base 12 as ZYX₁₂ -/
def base12_repr (X Y Z : ℕ) : ℕ := 144 * Z + 12 * Y + X

/-- Theorem stating the maximum value of m given the conditions -/
theorem max_value_m (m : ℕ) (X Y Z : ℕ) 
  (h1 : m > 0)
  (h2 : m = base8_repr X Y Z)
  (h3 : m = base12_repr X Y Z)
  (h4 : X < 8 ∧ Y < 8 ∧ Z < 8)  -- X, Y, Z are single digits in base 8
  (h5 : Z < 12 ∧ Y < 12 ∧ X < 12)  -- Z, Y, X are single digits in base 12
  : m ≤ 475 :=
sorry

end NUMINAMATH_CALUDE_max_value_m_l3770_377083


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3401_l3770_377051

theorem largest_prime_factor_of_3401 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3401 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3401 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3401_l3770_377051


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3770_377001

theorem quadratic_roots_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * x₂ = 2 →
  (x₁^2 + 2) * (x₂^2 + 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3770_377001


namespace NUMINAMATH_CALUDE_existence_of_s_l3770_377095

theorem existence_of_s (a : ℕ → ℕ) (k r : ℕ) (h1 : ∀ n m : ℕ, n ≤ m → a n ≤ a m) 
  (h2 : k > 0) (h3 : r > 0) (h4 : r = a r * (k + 1)) :
  ∃ s : ℕ, s > 0 ∧ s = a s * k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_s_l3770_377095


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l3770_377022

theorem divisible_by_nine_sequence (N : ℕ) : 
  (∃ (seq : List ℕ), 
    seq.length = 1110 ∧ 
    (∀ n ∈ seq, n % 9 = 0) ∧
    (∀ n ∈ seq, N ≤ n ∧ n ≤ 10000) ∧
    (∀ m, N ≤ m ∧ m ≤ 10000 ∧ m % 9 = 0 → m ∈ seq)) →
  N = 27 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l3770_377022


namespace NUMINAMATH_CALUDE_correct_calculation_l3770_377099

theorem correct_calculation (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3770_377099


namespace NUMINAMATH_CALUDE_complex_equation_result_l3770_377076

theorem complex_equation_result (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  m + i = (1 + 2 * i) * n * i → n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l3770_377076


namespace NUMINAMATH_CALUDE_age_difference_proof_l3770_377050

/-- Proves that the age difference between a man and his son is 34 years. -/
theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 32 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 34 := by
  sorry

#check age_difference_proof

end NUMINAMATH_CALUDE_age_difference_proof_l3770_377050


namespace NUMINAMATH_CALUDE_truck_wash_price_l3770_377025

/-- Proves that the price of a truck wash is $6 given the conditions of Laura's carwash --/
theorem truck_wash_price (car_price : ℕ) (suv_price : ℕ) (total_raised : ℕ) 
  (num_cars num_suvs num_trucks : ℕ) :
  car_price = 5 →
  suv_price = 7 →
  num_cars = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  total_raised = 100 →
  ∃ (truck_price : ℕ), 
    truck_price = 6 ∧ 
    car_price * num_cars + suv_price * num_suvs + truck_price * num_trucks = total_raised :=
by sorry

end NUMINAMATH_CALUDE_truck_wash_price_l3770_377025


namespace NUMINAMATH_CALUDE_village_population_l3770_377026

theorem village_population (population_95_percent : ℝ) (h : population_95_percent = 57200) :
  ∃ total_population : ℕ, 
    (↑total_population : ℝ) ≥ population_95_percent / 0.95 ∧ 
    (↑total_population : ℝ) < population_95_percent / 0.95 + 1 ∧
    total_population = 60211 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3770_377026


namespace NUMINAMATH_CALUDE_florist_roses_l3770_377004

theorem florist_roses (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l3770_377004


namespace NUMINAMATH_CALUDE_seokjin_position_l3770_377000

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on the stairs,
    prove that Seokjin is 3 steps higher than Jungkook. -/
theorem seokjin_position (jungkook yoojeong seokjin : ℕ) 
  (h1 : jungkook = 19)
  (h2 : yoojeong = jungkook + 8)
  (h3 : seokjin = yoojeong - 5) :
  seokjin - jungkook = 3 := by
sorry

end NUMINAMATH_CALUDE_seokjin_position_l3770_377000


namespace NUMINAMATH_CALUDE_nested_fraction_equals_five_thirds_l3770_377080

theorem nested_fraction_equals_five_thirds :
  1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_five_thirds_l3770_377080


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3770_377005

theorem jake_weight_loss (jake_current : ℕ) (combined_weight : ℕ) (weight_loss : ℕ) : 
  jake_current = 198 →
  combined_weight = 293 →
  jake_current - weight_loss = 2 * (combined_weight - jake_current) →
  weight_loss = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l3770_377005


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3770_377045

/-- A hyperbola with asymptotes y = ±2x passing through (1, 0) -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

theorem hyperbola_properties :
  ∀ (x y : ℝ),
    -- The equation represents a hyperbola
    (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
    -- With asymptotes y = ±2x
    (∃ (k : ℝ), k ≠ 0 ∧ (y = 2*x ∨ y = -2*x) → (x^2 - y^2/4 = k)) →
    -- Passing through the point (1, 0)
    hyperbola 1 0 →
    -- Then the hyperbola has the equation x² - y²/4 = 1
    hyperbola x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3770_377045


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3770_377097

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3770_377097


namespace NUMINAMATH_CALUDE_gumball_distribution_l3770_377020

/-- Theorem: Gumball Distribution
Given:
- Joanna initially had 40 gumballs
- Jacques initially had 60 gumballs
- They each purchased 4 times their initial amount
- They put all gumballs together and shared equally

Prove: Each person gets 250 gumballs -/
theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat)
  (h1 : joanna_initial = 40)
  (h2 : jacques_initial = 60)
  (purchase_multiplier : Nat)
  (h3 : purchase_multiplier = 4) :
  let joanna_total := joanna_initial + joanna_initial * purchase_multiplier
  let jacques_total := jacques_initial + jacques_initial * purchase_multiplier
  let total_gumballs := joanna_total + jacques_total
  (total_gumballs / 2 : Nat) = 250 := by
  sorry

end NUMINAMATH_CALUDE_gumball_distribution_l3770_377020


namespace NUMINAMATH_CALUDE_triangle_problem_l3770_377067

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * b * c * Real.sin A = 45 * Real.sqrt 3 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (2 * R * Real.sin A = a) →
  Real.cos A = -1 / 2 ∧ R = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3770_377067


namespace NUMINAMATH_CALUDE_geometry_class_ratio_l3770_377006

theorem geometry_class_ratio (total_students : ℕ) (boys_under_6ft : ℕ) :
  total_students = 38 →
  (2 : ℚ) / 3 * total_students = 25 →
  boys_under_6ft = 19 →
  (boys_under_6ft : ℚ) / 25 = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_geometry_class_ratio_l3770_377006


namespace NUMINAMATH_CALUDE_line_through_1_0_perpendicular_to_polar_axis_l3770_377087

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define a line in polar coordinates
structure PolarLine where
  equation : PolarPoint → Prop

-- Define the polar axis
def polarAxis : PolarLine :=
  { equation := fun p => p.θ = 0 }

-- Define perpendicularity in polar coordinates
def perpendicular (l1 l2 : PolarLine) : Prop :=
  sorry

-- Define the point (1, 0) in polar coordinates
def point_1_0 : PolarPoint :=
  { ρ := 1, θ := 0 }

-- The theorem to be proved
theorem line_through_1_0_perpendicular_to_polar_axis :
  ∃ (l : PolarLine),
    l.equation = fun p => p.ρ * Real.cos p.θ = 1 ∧
    l.equation point_1_0 ∧
    perpendicular l polarAxis :=
  sorry

end NUMINAMATH_CALUDE_line_through_1_0_perpendicular_to_polar_axis_l3770_377087


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l3770_377034

theorem coefficient_x_cubed_expansion : 
  let f : Polynomial ℚ := (X - 1) * (2 * X + 1)^5
  (f.coeff 3) = -40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l3770_377034


namespace NUMINAMATH_CALUDE_janes_minnows_l3770_377066

theorem janes_minnows (prize_minnows : ℕ) (total_players : ℕ) (win_percentage : ℚ) (leftover_minnows : ℕ) 
  (h1 : prize_minnows = 3)
  (h2 : total_players = 800)
  (h3 : win_percentage = 15 / 100)
  (h4 : leftover_minnows = 240) :
  prize_minnows * (win_percentage * total_players).floor + leftover_minnows = 600 := by
  sorry

end NUMINAMATH_CALUDE_janes_minnows_l3770_377066


namespace NUMINAMATH_CALUDE_barbara_candies_left_l3770_377008

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : Float) (used : Float) : Float :=
  initial - used

/-- Theorem: If Barbara initially has 18.0 candies and uses 9.0 candies,
    then the number of candies she has left is 9.0. -/
theorem barbara_candies_left :
  candies_left 18.0 9.0 = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_left_l3770_377008


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l3770_377036

theorem sum_and_ratio_implies_difference (x y : ℝ) 
  (h1 : x + y = 500) 
  (h2 : x / y = 0.75) : 
  y - x = 71.42 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l3770_377036


namespace NUMINAMATH_CALUDE_not_q_is_false_l3770_377044

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_false_l3770_377044


namespace NUMINAMATH_CALUDE_frame_width_proof_l3770_377063

theorem frame_width_proof (photo_width : ℝ) (photo_height : ℝ) (frame_width : ℝ) :
  photo_width = 12 →
  photo_height = 18 →
  (photo_width + 2 * frame_width) * (photo_height + 2 * frame_width) - photo_width * photo_height = photo_width * photo_height →
  frame_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_proof_l3770_377063


namespace NUMINAMATH_CALUDE_select_two_from_seven_l3770_377018

theorem select_two_from_seven : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_seven_l3770_377018


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3770_377038

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3770_377038


namespace NUMINAMATH_CALUDE_complex_to_polar_l3770_377052

open Complex

theorem complex_to_polar (θ : ℝ) : 
  (1 + Real.sin θ + I * Real.cos θ) / (1 + Real.sin θ - I * Real.cos θ) = 
  Complex.exp (I * (π / 2 - θ)) :=
sorry

end NUMINAMATH_CALUDE_complex_to_polar_l3770_377052


namespace NUMINAMATH_CALUDE_log_equation_solution_set_l3770_377046

theorem log_equation_solution_set :
  let S : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}
  ∀ x : ℝ, (Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)) ↔ x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_set_l3770_377046


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l3770_377027

open Real

theorem min_shift_for_symmetry (x : ℝ) :
  let f (x : ℝ) := cos (2 * x) + Real.sqrt 3 * sin (2 * x)
  ∃ m : ℝ, m > 0 ∧ 
    (∀ x, f (x + m) = f (-x + m)) ∧
    (∀ m' : ℝ, m' > 0 ∧ (∀ x, f (x + m') = f (-x + m')) → m ≤ m') ∧
    m = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l3770_377027


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3770_377094

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3770_377094


namespace NUMINAMATH_CALUDE_direct_proportion_function_m_l3770_377043

theorem direct_proportion_function_m (m : ℝ) : 
  (m^2 - 3 = 1 ∧ m + 2 ≠ 0) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_function_m_l3770_377043


namespace NUMINAMATH_CALUDE_initial_sheep_count_l3770_377048

theorem initial_sheep_count (horses : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) :
  horses = 100 →
  chickens = 9 →
  goats = 37 →
  male_animals = 53 →
  ∃ (sheep : ℕ), 
    (((horses + sheep + chickens) / 2 : ℚ) + goats : ℚ) = (2 * male_animals : ℚ) ∧
    sheep = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_sheep_count_l3770_377048


namespace NUMINAMATH_CALUDE_max_profit_pork_zongzi_l3770_377054

/-- Represents the wholesale and retail prices of zongzi -/
structure ZongziPrices where
  porkWholesale : ℝ
  redBeanWholesale : ℝ
  porkRetail : ℝ

/-- Represents the daily sales and profit of pork zongzi -/
structure PorkZongziSales where
  price : ℝ
  quantity : ℝ
  profit : ℝ

/-- The conditions given in the problem -/
def zongziConditions (z : ZongziPrices) : Prop :=
  z.porkWholesale = z.redBeanWholesale + 10 ∧
  z.porkWholesale + 2 * z.redBeanWholesale = 100

/-- The relationship between price and quantity sold for pork zongzi -/
def porkZongziDemand (basePrice baseQuantity : ℝ) (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.quantity = baseQuantity - 2 * (s.price - basePrice)

/-- The profit function for pork zongzi -/
def porkZongziProfit (z : ZongziPrices) (s : PorkZongziSales) : Prop :=
  s.profit = (s.price - z.porkWholesale) * s.quantity

/-- The main theorem stating the maximum profit -/
theorem max_profit_pork_zongzi (z : ZongziPrices) (s : PorkZongziSales) :
  zongziConditions z →
  porkZongziDemand 50 100 z s →
  porkZongziProfit z s →
  ∃ maxProfit : ℝ, maxProfit = 1800 ∧ ∀ s', porkZongziProfit z s' → s'.profit ≤ maxProfit :=
sorry

end NUMINAMATH_CALUDE_max_profit_pork_zongzi_l3770_377054


namespace NUMINAMATH_CALUDE_product_sum_relation_l3770_377049

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3770_377049


namespace NUMINAMATH_CALUDE_triple_debt_days_l3770_377037

def loan_amount : ℝ := 20
def daily_interest_rate : ℝ := 0.10

def days_to_triple_debt : ℕ := 20

theorem triple_debt_days :
  ∀ n : ℕ, (n : ℝ) ≥ (days_to_triple_debt : ℝ) ↔
  loan_amount * (1 + n * daily_interest_rate) ≥ 3 * loan_amount :=
by sorry

end NUMINAMATH_CALUDE_triple_debt_days_l3770_377037


namespace NUMINAMATH_CALUDE_max_min_product_l3770_377077

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + c * a = 27 → 
  ∀ m : ℝ, m = min (a * b) (min (b * c) (c * a)) → 
  m ≤ 6.75 ∧ ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 9 ∧ 
    a' * b' + b' * c' + c' * a' = 27 ∧ 
    min (a' * b') (min (b' * c') (c' * a')) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_max_min_product_l3770_377077


namespace NUMINAMATH_CALUDE_cora_cookie_expenditure_l3770_377032

def cookies_per_day : ℕ := 3
def cookie_cost : ℕ := 18
def days_in_april : ℕ := 30

theorem cora_cookie_expenditure :
  cookies_per_day * cookie_cost * days_in_april = 1620 := by
  sorry

end NUMINAMATH_CALUDE_cora_cookie_expenditure_l3770_377032


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l3770_377013

theorem largest_integer_for_negative_quadratic : 
  ∃ (n : ℤ), n = 7 ∧ n^2 - 11*n + 24 < 0 ∧ ∀ (m : ℤ), m > n → m^2 - 11*m + 24 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l3770_377013


namespace NUMINAMATH_CALUDE_sum_of_a_values_l3770_377030

theorem sum_of_a_values (a b : ℝ) (h1 : a + 1/b = 8) (h2 : b + 1/a = 3) : 
  ∃ (a₁ a₂ : ℝ), (a₁ + 1/b = 8 ∧ b + 1/a₁ = 3) ∧ 
                 (a₂ + 1/b = 8 ∧ b + 1/a₂ = 3) ∧ 
                 a₁ ≠ a₂ ∧ 
                 a₁ + a₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l3770_377030


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l3770_377078

theorem divisibility_by_eight (a : ℤ) (h : Even a) :
  (∃ k : ℤ, a * (a^2 + 20) = 8 * k) ∧
  (∃ l : ℤ, a * (a^2 - 20) = 8 * l) ∧
  (∃ m : ℤ, a * (a^2 - 4) = 8 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l3770_377078


namespace NUMINAMATH_CALUDE_planted_fraction_is_404_841_l3770_377081

/-- Represents a rectangular field with a planted right triangular area and an unplanted square --/
structure PlantedField where
  length : ℝ
  width : ℝ
  square_side : ℝ
  hypotenuse_distance : ℝ

/-- The fraction of the field that is planted --/
def planted_fraction (field : PlantedField) : ℚ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration --/
theorem planted_fraction_is_404_841 :
  let field : PlantedField := {
    length := 5,
    width := 6,
    square_side := 33 / 41,
    hypotenuse_distance := 3
  }
  planted_fraction field = 404 / 841 := by sorry

end NUMINAMATH_CALUDE_planted_fraction_is_404_841_l3770_377081


namespace NUMINAMATH_CALUDE_abc_fraction_value_l3770_377091

theorem abc_fraction_value (a b c : ℕ+) 
  (h : a^2*b + b^2*c + a*c^2 + a + b + c = 2*(a*b + b*c + a*c)) :
  (c : ℚ)^2017 / ((a : ℚ)^2016 + (b : ℚ)^2018) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l3770_377091


namespace NUMINAMATH_CALUDE_number_relations_with_180_l3770_377017

theorem number_relations_with_180 :
  (∃ n : ℤ, n = 180 + 15 ∧ n = 195) ∧
  (∃ m : ℤ, m = 180 - 15 ∧ m = 165) := by
  sorry

end NUMINAMATH_CALUDE_number_relations_with_180_l3770_377017


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l3770_377085

-- Define the fractions of pizza eaten by each friend
def samuel_fraction : ℚ := 1/6
def teresa_fraction : ℚ := 2/5
def uma_fraction : ℚ := 1/4

-- Define the amount of pizza eaten by Victor
def victor_fraction : ℚ := 1 - (samuel_fraction + teresa_fraction + uma_fraction)

-- Define a function to compare two fractions
def eats_more (a b : ℚ) : Prop := a > b

-- Theorem stating the order of pizza consumption
theorem pizza_consumption_order :
  eats_more teresa_fraction uma_fraction ∧
  eats_more uma_fraction victor_fraction ∧
  eats_more victor_fraction samuel_fraction :=
sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l3770_377085


namespace NUMINAMATH_CALUDE_tan_triple_angle_l3770_377062

theorem tan_triple_angle (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l3770_377062


namespace NUMINAMATH_CALUDE_certain_number_proof_l3770_377096

theorem certain_number_proof : ∃ x : ℝ, 0.80 * x = (4/5 * 20) + 16 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3770_377096


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l3770_377035

-- Define the cost of mangoes and oranges
def mango_cost : ℝ := 0.60
def orange_cost : ℝ := 0.40

-- Define the weight of mangoes and oranges Kelly buys
def mango_weight : ℝ := 5
def orange_weight : ℝ := 5

-- Define the discount percentage
def discount_rate : ℝ := 0.10

-- Define the function to calculate the total cost after discount
def total_cost_after_discount (m_cost o_cost m_weight o_weight disc_rate : ℝ) : ℝ :=
  let total_cost := (m_cost * 2 * m_weight) + (o_cost * 4 * o_weight)
  total_cost * (1 - disc_rate)

-- Theorem statement
theorem fruit_purchase_cost :
  total_cost_after_discount mango_cost orange_cost mango_weight orange_weight discount_rate = 12.60 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l3770_377035


namespace NUMINAMATH_CALUDE_andrew_kept_stickers_l3770_377070

def total_stickers : ℕ := 2000

def daniel_stickers : ℕ := (total_stickers * 5) / 100

def fred_stickers : ℕ := daniel_stickers + 120

def emily_stickers : ℕ := ((daniel_stickers + fred_stickers) * 50) / 100

def gina_stickers : ℕ := 80

def hannah_stickers : ℕ := ((emily_stickers + gina_stickers) * 20) / 100

def total_given_away : ℕ := daniel_stickers + fred_stickers + emily_stickers + gina_stickers + hannah_stickers

theorem andrew_kept_stickers : total_stickers - total_given_away = 1392 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_stickers_l3770_377070


namespace NUMINAMATH_CALUDE_power_sum_calculation_l3770_377029

theorem power_sum_calculation : (-1: ℤ)^53 + (2 : ℚ)^(2^4 + 5^2 - 4^3) = -1 + 1 / 8388608 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_calculation_l3770_377029


namespace NUMINAMATH_CALUDE_magic_stick_height_difference_l3770_377064

-- Define the edge length of the large cube in meters
def large_cube_edge : ℝ := 1

-- Define the edge length of the small cubes in centimeters
def small_cube_edge : ℝ := 1

-- Define the height of Mount Everest in meters
def everest_height : ℝ := 8844

-- Conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem magic_stick_height_difference :
  let large_cube_volume : ℝ := large_cube_edge ^ 3
  let small_cube_volume : ℝ := (small_cube_edge * cm_to_m) ^ 3
  let num_small_cubes : ℝ := large_cube_volume / small_cube_volume
  let magic_stick_height : ℝ := num_small_cubes * small_cube_edge * cm_to_m
  magic_stick_height - everest_height = 1156 := by
  sorry

end NUMINAMATH_CALUDE_magic_stick_height_difference_l3770_377064


namespace NUMINAMATH_CALUDE_total_seashells_l3770_377047

theorem total_seashells (sally tom jessica alex : ℝ) 
  (h1 : sally = 9.5)
  (h2 : tom = 7.2)
  (h3 : jessica = 5.3)
  (h4 : alex = 12.8) :
  sally + tom + jessica + alex = 34.8 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l3770_377047


namespace NUMINAMATH_CALUDE_fiona_earnings_l3770_377068

-- Define the time worked each day in hours
def monday_hours : ℝ := 1.5
def tuesday_hours : ℝ := 1.25
def wednesday_hours : ℝ := 3.1667
def thursday_hours : ℝ := 0.75

-- Define the hourly rate
def hourly_rate : ℝ := 4

-- Define the total hours worked
def total_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours

-- Define the weekly earnings
def weekly_earnings : ℝ := total_hours * hourly_rate

-- Theorem statement
theorem fiona_earnings : 
  ∃ ε > 0, |weekly_earnings - 26.67| < ε :=
sorry

end NUMINAMATH_CALUDE_fiona_earnings_l3770_377068


namespace NUMINAMATH_CALUDE_gregs_ppo_reward_l3770_377088

theorem gregs_ppo_reward (
  ppo_percentage : Real)
  (coinrun_max_reward : Real)
  (procgen_max_reward : Real)
  (h1 : ppo_percentage = 0.9)
  (h2 : coinrun_max_reward = procgen_max_reward / 2)
  (h3 : procgen_max_reward = 240)
  : ppo_percentage * coinrun_max_reward = 108 := by
  sorry

end NUMINAMATH_CALUDE_gregs_ppo_reward_l3770_377088


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3770_377084

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 + Real.log x else 3 * x + 4

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3770_377084


namespace NUMINAMATH_CALUDE_ajay_dal_transaction_gain_l3770_377010

/-- Represents the transaction of buying and selling dal -/
structure DalTransaction where
  quantity1 : ℝ
  price1 : ℝ
  quantity2 : ℝ
  price2 : ℝ
  selling_price : ℝ

/-- Calculate the total gain from a dal transaction -/
def calculate_gain (t : DalTransaction) : ℝ :=
  let total_quantity := t.quantity1 + t.quantity2
  let total_cost := t.quantity1 * t.price1 + t.quantity2 * t.price2
  let total_revenue := total_quantity * t.selling_price
  total_revenue - total_cost

/-- Theorem stating that Ajay's total gain in the dal transaction is 27.50 rs -/
theorem ajay_dal_transaction_gain :
  let t : DalTransaction := {
    quantity1 := 15,
    price1 := 14.50,
    quantity2 := 10,
    price2 := 13,
    selling_price := 15
  }
  calculate_gain t = 27.50 := by
  sorry

end NUMINAMATH_CALUDE_ajay_dal_transaction_gain_l3770_377010


namespace NUMINAMATH_CALUDE_function_inequality_l3770_377024

/-- Given a function f: ℝ → ℝ satisfying certain conditions, 
    prove that the set of x where f(x) > 1/e^x is (ln 3, +∞) -/
theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x > -f x) 
  (h2 : f (Real.log 3) = 1/3) :
  {x : ℝ | f x > Real.exp (-x)} = Set.Ioi (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3770_377024


namespace NUMINAMATH_CALUDE_wills_game_cost_l3770_377065

/-- Proves that the cost of Will's new game is $47 --/
theorem wills_game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 83 →
  num_toys = 9 →
  toy_price = 4 →
  initial_money = game_cost + (num_toys * toy_price) →
  game_cost = 47 := by
sorry

end NUMINAMATH_CALUDE_wills_game_cost_l3770_377065


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l3770_377089

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) 
  (vegan_with_gluten : ℕ) 
  (low_sugar_gluten_free_vegan : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  vegan_with_gluten = 4 →
  low_sugar_gluten_free_vegan = 1 →
  low_sugar_gluten_free_vegan = total_dishes / 24 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l3770_377089


namespace NUMINAMATH_CALUDE_lucy_fish_total_l3770_377069

theorem lucy_fish_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 212 → additional = 68 → total = initial + additional → total = 280 := by
sorry

end NUMINAMATH_CALUDE_lucy_fish_total_l3770_377069


namespace NUMINAMATH_CALUDE_inequality_solution_l3770_377028

theorem inequality_solution : 
  let S : Set ℚ := {-3, -1/2, 1/3, 2}
  ∀ x ∈ S, 2*(x-1)+3 < 0 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3770_377028


namespace NUMINAMATH_CALUDE_log_difference_equals_one_l3770_377014

theorem log_difference_equals_one (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (α + π / 4) = 3) : 
  Real.log (8 * Real.sin α + 6 * Real.cos α) - Real.log (4 * Real.sin α - Real.cos α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_one_l3770_377014


namespace NUMINAMATH_CALUDE_yellow_ball_count_l3770_377007

theorem yellow_ball_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob_red : ℚ) : 
  red = 9 →
  yellow + red = total →
  prob_red = 1/3 →
  prob_red = red / total →
  yellow = 18 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l3770_377007


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3770_377075

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 4
  (b^2 - 4*a*c) = 33 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3770_377075


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3770_377053

theorem max_sum_of_squares (x y z : ℕ+) 
  (h1 : x.val * y.val * z.val = (14 - x.val) * (14 - y.val) * (14 - z.val))
  (h2 : x.val + y.val + z.val < 28) :
  x.val^2 + y.val^2 + z.val^2 ≤ 219 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3770_377053


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3770_377098

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 10*x + 16 = 0
def equation2 (x : ℝ) : Prop := 2*x*(x-1) = x-1

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 2 ∨ x = 8) :=
by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3770_377098


namespace NUMINAMATH_CALUDE_function_transformation_l3770_377060

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = 19 * x^2 + 55 * x - 44) → 
  (∀ x, f x = 19 * x^2 + 93 * x + 30) :=
by sorry

end NUMINAMATH_CALUDE_function_transformation_l3770_377060


namespace NUMINAMATH_CALUDE_golu_travel_distance_l3770_377074

theorem golu_travel_distance (x : ℝ) :
  x > 0 ∧ x^2 + 6^2 = 10^2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_golu_travel_distance_l3770_377074


namespace NUMINAMATH_CALUDE_notebook_count_l3770_377073

theorem notebook_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : n^2 + 20 = (n + 1)^2 - 9) : 
  n^2 + 20 = 216 :=
by sorry

end NUMINAMATH_CALUDE_notebook_count_l3770_377073


namespace NUMINAMATH_CALUDE_solve_for_r_l3770_377021

theorem solve_for_r : 
  let r := Real.sqrt (8^2 + 15^2) / Real.sqrt 25
  r = 17 / 5 := by sorry

end NUMINAMATH_CALUDE_solve_for_r_l3770_377021


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3770_377059

theorem simplify_trig_expression (x : ℝ) :
  (Real.sqrt 2 / 4) * Real.sin (π / 4 - x) + (Real.sqrt 6 / 4) * Real.cos (π / 4 - x) =
  (Real.sqrt 2 / 2) * Real.sin (7 * π / 12 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3770_377059


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3770_377086

-- Define the plane and lines
variable (α : Plane)
variable (m n : Line)

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3770_377086


namespace NUMINAMATH_CALUDE_least_grood_number_l3770_377042

theorem least_grood_number (n : ℕ) : n ≥ 10 ↔ (n * (n + 1) : ℚ) / 4 > n^2 := by sorry

end NUMINAMATH_CALUDE_least_grood_number_l3770_377042


namespace NUMINAMATH_CALUDE_discount_difference_l3770_377092

def original_price : ℝ := 12000

def single_discount_rate : ℝ := 0.45
def successive_discount_rate1 : ℝ := 0.35
def successive_discount_rate2 : ℝ := 0.10

def price_after_single_discount : ℝ := original_price * (1 - single_discount_rate)
def price_after_successive_discounts : ℝ := original_price * (1 - successive_discount_rate1) * (1 - successive_discount_rate2)

theorem discount_difference :
  price_after_successive_discounts - price_after_single_discount = 420 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3770_377092


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l3770_377055

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l3770_377055


namespace NUMINAMATH_CALUDE_committee_probability_l3770_377023

/-- The probability of selecting a 5-person committee with at least one boy and one girl
    from a group of 25 members (10 boys and 15 girls) is equal to 475/506. -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total_members = 25 →
  boys = 10 →
  girls = 15 →
  committee_size = 5 →
  (Nat.choose total_members committee_size - Nat.choose boys committee_size - Nat.choose girls committee_size) /
  Nat.choose total_members committee_size = 475 / 506 := by
  sorry

#eval Nat.choose 25 5
#eval Nat.choose 10 5
#eval Nat.choose 15 5

end NUMINAMATH_CALUDE_committee_probability_l3770_377023


namespace NUMINAMATH_CALUDE_hayden_ride_payment_l3770_377071

def hayden_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (gas_gallons : ℝ) (gas_price : ℝ) 
  (num_reviews : ℕ) (review_bonus : ℝ) (num_rides : ℕ) (total_owed : ℝ) : Prop :=
  let base_earnings := hourly_wage * hours_worked + gas_gallons * gas_price + num_reviews * review_bonus
  let ride_earnings := total_owed - base_earnings
  ride_earnings / num_rides = 5

theorem hayden_ride_payment : 
  hayden_earnings 15 8 17 3 2 20 3 226 := by sorry

end NUMINAMATH_CALUDE_hayden_ride_payment_l3770_377071


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3770_377058

theorem coefficient_x4_in_expansion : 
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 3^(8 - k) * if k = 4 then 1 else 0) = 5670 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3770_377058


namespace NUMINAMATH_CALUDE_exam_papers_count_l3770_377031

theorem exam_papers_count (average : ℝ) (new_average : ℝ) (geography_increase : ℝ) (history_increase : ℝ) :
  average = 63 →
  new_average = 65 →
  geography_increase = 20 →
  history_increase = 2 →
  ∃ n : ℕ, n * average + geography_increase + history_increase = n * new_average ∧ n = 11 :=
by sorry

end NUMINAMATH_CALUDE_exam_papers_count_l3770_377031


namespace NUMINAMATH_CALUDE_least_m_for_x_bound_l3770_377072

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem least_m_for_x_bound : 
  ∃ m : ℕ, m = 89 ∧ x m ≤ 5 + 1 / (2^10) ∧ ∀ k < m, x k > 5 + 1 / (2^10) :=
sorry

end NUMINAMATH_CALUDE_least_m_for_x_bound_l3770_377072


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_subgraph_l3770_377011

/-- A simple graph with 10 vertices and n edges, where edges are colored in two colors -/
structure ColoredGraph (n : ℕ) :=
  (edges : Fin n → Fin 10 × Fin 10)
  (color : Fin n → Bool)
  (simple : ∀ i : Fin n, (edges i).1 ≠ (edges i).2)

/-- A monochromatic triangle in a colored graph -/
def has_monochromatic_triangle (G : ColoredGraph n) : Prop :=
  ∃ (i j k : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges j ≠ G.edges k ∧
    G.color i = G.color j ∧ G.color j = G.color k

/-- A monochromatic quadrilateral in a colored graph -/
def has_monochromatic_quadrilateral (G : ColoredGraph n) : Prop :=
  ∃ (i j k l : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges i ≠ G.edges l ∧ 
    G.edges j ≠ G.edges k ∧ G.edges j ≠ G.edges l ∧ G.edges k ≠ G.edges l ∧
    G.color i = G.color j ∧ G.color j = G.color k ∧ G.color k = G.color l

/-- The main theorem -/
theorem smallest_n_for_monochromatic_subgraph : 
  (∀ G : ColoredGraph 31, has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G) ∧
  (∃ G : ColoredGraph 30, ¬(has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_subgraph_l3770_377011


namespace NUMINAMATH_CALUDE_janet_action_figures_l3770_377009

/-- The number of action figures Janet initially owns -/
def initial_figures : ℕ := 10

/-- The number of new action figures Janet buys -/
def new_figures : ℕ := 4

/-- The total number of action figures Janet has at the end -/
def total_figures : ℕ := 24

/-- The number of action figures Janet sold -/
def sold_figures : ℕ := 6

theorem janet_action_figures :
  ∃ (x : ℕ),
    x = sold_figures ∧
    initial_figures - x + new_figures +
    2 * (initial_figures - x + new_figures) = total_figures :=
by sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3770_377009


namespace NUMINAMATH_CALUDE_train_distance_problem_l3770_377039

/-- The distance between two points A and B, given two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (hv1 : v1 = 50) (hv2 : v2 = 60) (hd : d = 100) :
  let x := (v1 * d) / (v2 - v1)
  x + (x + d) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3770_377039


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_24_l3770_377002

-- Define q as the largest prime with 2021 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2021 digits
axiom q_digits : 10^2020 ≤ q ∧ q < 10^2021

-- Theorem to prove
theorem smallest_k_divisible_by_24 :
  ∃ k : ℕ, k > 0 ∧ 24 ∣ (q^2 - k) ∧ ∀ m : ℕ, m > 0 → 24 ∣ (q^2 - m) → k ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_24_l3770_377002


namespace NUMINAMATH_CALUDE_nancy_work_hours_nancy_specific_case_l3770_377040

/-- Given Nancy's earnings and work hours, calculate the number of hours needed to earn a target amount -/
theorem nancy_work_hours (earnings : ℝ) (work_hours : ℝ) (target_amount : ℝ) :
  earnings > 0 ∧ work_hours > 0 ∧ target_amount > 0 →
  let hourly_rate := earnings / work_hours
  (target_amount / hourly_rate) = (target_amount * work_hours) / earnings :=
by sorry

/-- Nancy's specific work scenario -/
theorem nancy_specific_case :
  let earnings := 28
  let work_hours := 4
  let target_amount := 70
  let hours_needed := (target_amount * work_hours) / earnings
  hours_needed = 10 :=
by sorry

end NUMINAMATH_CALUDE_nancy_work_hours_nancy_specific_case_l3770_377040


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l3770_377061

def a (n : ℕ+) : ℝ := 3 * 2^(n.val - 1)

def S (n : ℕ+) : ℝ := 3 * (2^n.val - 1)

theorem geometric_sequence_sum_inequality {k : ℝ} :
  (∀ n : ℕ+, a (n + 1) + a n = 9 * 2^(n.val - 1)) →
  (∀ n : ℕ+, S n > k * a n - 2) →
  k < 5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l3770_377061


namespace NUMINAMATH_CALUDE_min_value_theorem_l3770_377012

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h : x * y - 2 * x - y + 1 = 0) : 
  ∀ z w : ℝ, z > 1 → w > 1 → z * w - 2 * z - w + 1 = 0 → 
  (3 / 2) * x^2 + y^2 ≤ (3 / 2) * z^2 + w^2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3770_377012


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l3770_377079

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  area : ℝ := s^2

/-- Represents a rectangle with width w and height h -/
structure Rectangle (w h : ℝ) where
  area : ℝ := w * h

/-- The theorem statement -/
theorem square_rectangle_area_ratio 
  (s : ℝ) 
  (w h : ℝ) 
  (square : Square s) 
  (rect : Rectangle w h) 
  (h1 : rect.area = 0.25 * square.area) 
  (h2 : w = 8 * h) : 
  square.area / rect.area = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l3770_377079


namespace NUMINAMATH_CALUDE_parabola_vertex_l3770_377003

/-- The vertex of a parabola defined by y = x^2 + 2x - 3 is (-1, -4) -/
theorem parabola_vertex : 
  let f (x : ℝ) := x^2 + 2*x - 3
  ∃! (a b : ℝ), (∀ x, f x = (x - a)^2 + b) ∧ (a = -1 ∧ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3770_377003


namespace NUMINAMATH_CALUDE_parametric_line_position_vector_l3770_377056

/-- A line in a plane parameterized by t -/
structure ParametricLine where
  a : ℝ × ℝ  -- Point on the line
  d : ℝ × ℝ  -- Direction vector

/-- The position vector on a parametric line at a given t -/
def position_vector (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

theorem parametric_line_position_vector :
  ∀ (line : ParametricLine),
    position_vector line 5 = (4, -1) →
    position_vector line (-1) = (-2, 13) →
    position_vector line 8 = (7, -8/3) := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_position_vector_l3770_377056


namespace NUMINAMATH_CALUDE_at_op_four_six_l3770_377041

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 2 * a^2 - 2 * b^2

-- Theorem statement
theorem at_op_four_six : at_op 4 6 = -40 := by sorry

end NUMINAMATH_CALUDE_at_op_four_six_l3770_377041


namespace NUMINAMATH_CALUDE_other_sales_is_fifteen_percent_l3770_377057

/-- The percentage of sales not attributed to books, magazines, or stationery -/
def other_sales_percentage (books magazines stationery : ℝ) : ℝ :=
  100 - (books + magazines + stationery)

/-- Theorem stating that the percentage of other sales is 15% -/
theorem other_sales_is_fifteen_percent :
  other_sales_percentage 45 30 10 = 15 := by
  sorry

#eval other_sales_percentage 45 30 10

end NUMINAMATH_CALUDE_other_sales_is_fifteen_percent_l3770_377057


namespace NUMINAMATH_CALUDE_fifth_term_is_13_l3770_377015

/-- A sequence where the difference between consecutive terms increases by 1 each time -/
def increasing_diff_seq (a₁ : ℕ) (d₁ : ℕ) : ℕ → ℕ
| 0 => a₁
| n + 1 => increasing_diff_seq a₁ d₁ n + d₁ + n

theorem fifth_term_is_13 (a₁ d₁ : ℕ) :
  a₁ = 3 ∧ d₁ = 1 →
  increasing_diff_seq a₁ d₁ 1 = 4 ∧
  increasing_diff_seq a₁ d₁ 2 = 6 ∧
  increasing_diff_seq a₁ d₁ 3 = 9 →
  increasing_diff_seq a₁ d₁ 4 = 13 := by
  sorry

#eval increasing_diff_seq 3 1 4  -- Should output 13

end NUMINAMATH_CALUDE_fifth_term_is_13_l3770_377015


namespace NUMINAMATH_CALUDE_second_outlet_pipe_rate_l3770_377019

-- Define the volume of the tank in cubic inches
def tank_volume : ℝ := 30 * 1728

-- Define the inlet pipe rate in cubic inches per minute
def inlet_rate : ℝ := 3

-- Define the first outlet pipe rate in cubic inches per minute
def outlet_rate_1 : ℝ := 6

-- Define the time to empty the tank in minutes
def emptying_time : ℝ := 3456

-- Define the unknown rate of the second outlet pipe
def outlet_rate_2 : ℝ := 12

-- Theorem statement
theorem second_outlet_pipe_rate : 
  tank_volume / (outlet_rate_1 + outlet_rate_2 - inlet_rate) = emptying_time :=
sorry

end NUMINAMATH_CALUDE_second_outlet_pipe_rate_l3770_377019


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3770_377016

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2*x + 1 = 0 ∧ m * y^2 + 2*y + 1 = 0) ↔ 
  (m ≤ 1 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3770_377016


namespace NUMINAMATH_CALUDE_max_pencils_purchased_l3770_377093

/-- Given a pencil price of 25 cents and $50 available, 
    prove that the maximum number of pencils that can be purchased is 200. -/
theorem max_pencils_purchased (pencil_price : ℕ) (available_money : ℕ) :
  pencil_price = 25 →
  available_money = 5000 →
  (∀ n : ℕ, n * pencil_price ≤ available_money → n ≤ 200) ∧
  200 * pencil_price ≤ available_money :=
by
  sorry

#check max_pencils_purchased

end NUMINAMATH_CALUDE_max_pencils_purchased_l3770_377093
