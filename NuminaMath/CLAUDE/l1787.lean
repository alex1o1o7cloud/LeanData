import Mathlib

namespace NUMINAMATH_CALUDE_factors_of_2310_l1787_178710

theorem factors_of_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2310_l1787_178710


namespace NUMINAMATH_CALUDE_greatest_n_value_l1787_178706

theorem greatest_n_value (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 ∧ ∃ (m : ℤ), m = 11 ∧ 93 * m^3 ≤ 145800 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l1787_178706


namespace NUMINAMATH_CALUDE_square_units_digits_correct_l1787_178745

/-- The set of all possible units digits of squares of whole numbers -/
def square_units_digits : Set Nat :=
  {0, 1, 4, 5, 6, 9}

/-- Function to get the units digit of a number -/
def units_digit (n : Nat) : Nat :=
  n % 10

/-- Theorem stating that the set of all possible units digits of squares of whole numbers
    is exactly {0, 1, 4, 5, 6, 9} -/
theorem square_units_digits_correct :
  ∀ n : Nat, ∃ m : Nat, units_digit (m * m) ∈ square_units_digits ∧
  ∀ k : Nat, units_digit (k * k) ∈ square_units_digits := by
  sorry

#check square_units_digits_correct

end NUMINAMATH_CALUDE_square_units_digits_correct_l1787_178745


namespace NUMINAMATH_CALUDE_water_distribution_l1787_178799

structure Bottle where
  volume : ℝ
  h_volume : volume > 0 ∧ volume < 1

def total_volume (bottles : List Bottle) : ℝ :=
  bottles.foldl (fun acc b => acc + b.volume) 0

theorem water_distribution (n : ℕ) (h_n : n ≥ 1) (bottles : List Bottle) 
  (h_total : total_volume bottles = n / 2) :
  ∃ (distribution : List ℝ), 
    distribution.length = n ∧ 
    (∀ v ∈ distribution, v ≤ 1) ∧
    (total_volume bottles = distribution.foldl (· + ·) 0) := by
  sorry

end NUMINAMATH_CALUDE_water_distribution_l1787_178799


namespace NUMINAMATH_CALUDE_competition_probabilities_l1787_178716

def score_prob (p1 p2 p3 : ℝ) : ℝ × ℝ :=
  let prob_300 := p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
  let prob_400 := p1 * p2 * p3
  (prob_300, prob_300 + prob_400)

theorem competition_probabilities :
  let (prob_300, prob_at_least_300) := score_prob 0.8 0.7 0.6
  prob_300 = 0.228 ∧ prob_at_least_300 = 0.564 := by
  sorry

end NUMINAMATH_CALUDE_competition_probabilities_l1787_178716


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1787_178784

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation (x y : ℝ) :
  (∃ x₀ y₀, ellipse x₀ y₀) →  -- Ellipse exists
  (∃ x₁ y₁, asymptote x₁ y₁) →  -- Asymptote exists
  (∀ x₂ y₂, hyperbola_C x₂ y₂ ↔ 
    (∃ c : ℝ, c > 0 ∧  -- Foci distance
    (x₂ - c)^2 + y₂^2 = (x₂ + c)^2 + y₂^2 ∧  -- Same foci as ellipse
    (∃ t : ℝ, x₂ = t ∧ y₂ = Real.sqrt 3 * t)))  -- Asymptote condition
  :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1787_178784


namespace NUMINAMATH_CALUDE_parabola_focus_l1787_178776

/-- The parabola is defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola with equation y = ax^2 has coordinates (0, 1/(4a)) -/
def is_focus (a x y : ℝ) : Prop := x = 0 ∧ y = 1 / (4 * a)

/-- Prove that the focus of the parabola y = (1/4)x^2 has coordinates (0, 1) -/
theorem parabola_focus :
  is_focus (1/4) 0 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1787_178776


namespace NUMINAMATH_CALUDE_exists_71_cubes_l1787_178785

/-- Represents the number of cubes after a series of divisions -/
def num_cubes : ℕ → ℕ
| 0 => 1
| (n + 1) => num_cubes n + 7

/-- Theorem stating that it's possible to obtain 71 cubes through the division process -/
theorem exists_71_cubes : ∃ n : ℕ, num_cubes n = 71 := by
  sorry

end NUMINAMATH_CALUDE_exists_71_cubes_l1787_178785


namespace NUMINAMATH_CALUDE_tylers_age_l1787_178734

theorem tylers_age :
  ∀ (tyler_age brother_age : ℕ),
  tyler_age = brother_age - 3 →
  tyler_age + brother_age = 11 →
  tyler_age = 4 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l1787_178734


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l1787_178726

def g (p q r s : ℤ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 + r*x + s

theorem polynomial_constant_term 
  (p q r s : ℤ) 
  (h1 : p + q + r + s = 168)
  (h2 : ∀ x : ℝ, g p q r s x = 0 → (∃ n : ℤ, x = -n ∧ n > 0))
  (h3 : ∀ x : ℝ, (g p q r s x = 0) → (g p q r s (-x) = 0)) :
  s = 144 := by sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l1787_178726


namespace NUMINAMATH_CALUDE_square_area_error_l1787_178738

theorem square_area_error (side_error : Real) (area_error : Real) : 
  side_error = 0.19 → area_error = 0.4161 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1787_178738


namespace NUMINAMATH_CALUDE_shooting_sequences_l1787_178713

-- Define the number of targets in each column
def targets_A : ℕ := 3
def targets_B : ℕ := 2
def targets_C : ℕ := 3

-- Define the total number of targets
def total_targets : ℕ := targets_A + targets_B + targets_C

-- Theorem statement
theorem shooting_sequences :
  (total_targets.factorial) / (targets_A.factorial * targets_B.factorial * targets_C.factorial) = 560 := by
  sorry

end NUMINAMATH_CALUDE_shooting_sequences_l1787_178713


namespace NUMINAMATH_CALUDE_grain_production_theorem_l1787_178732

theorem grain_production_theorem (planned_wheat planned_corn actual_wheat actual_corn : ℝ) :
  planned_wheat + planned_corn = 18 →
  actual_wheat + actual_corn = 20 →
  actual_wheat = planned_wheat * 1.12 →
  actual_corn = planned_corn * 1.10 →
  actual_wheat = 11.2 ∧ actual_corn = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_grain_production_theorem_l1787_178732


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1787_178712

theorem necessary_but_not_sufficient_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x^2 - 3*a*x + 2*a^2 ≤ 0 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ x^2 - 3*a*x + 2*a^2 > 0) →
  a > 1 := by
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1787_178712


namespace NUMINAMATH_CALUDE_office_staff_composition_l1787_178794

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 480

/-- Represents the average salary of all employees in Rs/month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in Rs/month. -/
def avg_salary_officers : ℕ := 440

/-- Represents the average salary of non-officers in Rs/month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions of the problem. -/
theorem office_staff_composition :
  num_officers = 15 ∧
  num_non_officers = 480 ∧
  avg_salary_all * (num_officers + num_non_officers) = 
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers :=
by sorry


end NUMINAMATH_CALUDE_office_staff_composition_l1787_178794


namespace NUMINAMATH_CALUDE_night_shift_arrangements_count_l1787_178788

/-- The number of days in the shift schedule -/
def num_days : ℕ := 6

/-- The number of people available for shifts -/
def num_people : ℕ := 4

/-- The number of scenarios for arranging consecutive shifts -/
def num_scenarios : ℕ := 6

/-- Calculates the number of different night shift arrangements -/
def night_shift_arrangements : ℕ := 
  num_scenarios * (num_people.factorial / (num_people - 2).factorial) * 
  ((num_people - 2).factorial / (num_people - 4).factorial)

/-- Theorem stating the number of different night shift arrangements -/
theorem night_shift_arrangements_count : night_shift_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_night_shift_arrangements_count_l1787_178788


namespace NUMINAMATH_CALUDE_initial_bananas_count_l1787_178764

-- Define the number of bananas left on the tree
def bananas_left : ℕ := 430

-- Define the number of bananas eaten by each person
def raj_eaten : ℕ := 120
def asha_eaten : ℕ := 100
def vijay_eaten : ℕ := 80

-- Define the ratios of remaining to eaten bananas for each person
def raj_ratio : ℕ := 2
def asha_ratio : ℕ := 3
def vijay_ratio : ℕ := 4

-- Define the function to calculate the total number of bananas
def total_bananas : ℕ :=
  bananas_left +
  (raj_ratio * raj_eaten + raj_eaten) +
  (asha_ratio * asha_eaten + asha_eaten) +
  (vijay_ratio * vijay_eaten + vijay_eaten)

-- Theorem statement
theorem initial_bananas_count :
  total_bananas = 1290 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l1787_178764


namespace NUMINAMATH_CALUDE_map_scale_l1787_178711

/-- If 15 cm on a map represents 90 km, then 20 cm represents 120 km -/
theorem map_scale (map_length : ℝ) (actual_distance : ℝ) : 
  (15 * map_length = 90 * actual_distance) → 
  (20 * map_length = 120 * actual_distance) := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l1787_178711


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1787_178735

theorem necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1787_178735


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l1787_178762

theorem sum_of_possible_x_values (x z : ℝ) (h1 : |x - z| = 100) (h2 : |z - 12| = 60) : 
  ∃ (x1 x2 x3 x4 : ℝ), 
    (|x1 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x2 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x3 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x4 - z| = 100 ∧ |z - 12| = 60) ∧
    x1 + x2 + x3 + x4 = 48 ∧
    (∀ y : ℝ, (|y - z| = 100 ∧ |z - 12| = 60) → (y = x1 ∨ y = x2 ∨ y = x3 ∨ y = x4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l1787_178762


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l1787_178722

theorem magnitude_of_complex_number (z : ℂ) : z = Complex.I * (3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l1787_178722


namespace NUMINAMATH_CALUDE_f_composition_nine_equals_one_eighth_l1787_178757

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x else 2^x

theorem f_composition_nine_equals_one_eighth :
  f (f 9) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_nine_equals_one_eighth_l1787_178757


namespace NUMINAMATH_CALUDE_french_fries_cooking_time_l1787_178786

/-- Calculates the remaining cooking time in seconds given the recommended time in minutes and the actual cooking time in seconds. -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem stating that for a recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds. -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_french_fries_cooking_time_l1787_178786


namespace NUMINAMATH_CALUDE_min_sum_squares_l1787_178790

theorem min_sum_squares (a b t : ℝ) (h : a + b = t) :
  ∃ (min : ℝ), min = t^2 / 2 ∧ ∀ (x y : ℝ), x + y = t → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1787_178790


namespace NUMINAMATH_CALUDE_polygon_internal_diagonals_l1787_178720

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields and constraints
  sides : ℕ
  sides_eq : sides = n
  sides_ge_3 : n ≥ 3

/-- A diagonal of a polygon -/
structure Diagonal (p : Polygon n) where
  -- Add necessary fields and constraints

/-- Predicate to check if a diagonal is completely inside the polygon -/
def is_inside (d : Diagonal p) : Prop :=
  -- Define the condition for a diagonal to be inside the polygon
  sorry

/-- The number of complete internal diagonals in a polygon -/
def num_internal_diagonals (p : Polygon n) : ℕ :=
  -- Define the number of internal diagonals
  sorry

/-- Theorem: Any polygon with more than 3 sides has at least one internal diagonal,
    and the minimum number of internal diagonals is n-3 -/
theorem polygon_internal_diagonals (n : ℕ) (h : n > 3) (p : Polygon n) :
  (∃ d : Diagonal p, is_inside d) ∧ num_internal_diagonals p = n - 3 :=
sorry

end NUMINAMATH_CALUDE_polygon_internal_diagonals_l1787_178720


namespace NUMINAMATH_CALUDE_least_positive_congruence_l1787_178775

theorem least_positive_congruence :
  ∃! x : ℕ, x > 0 ∧ x + 5600 ≡ 325 [ZMOD 15] ∧ ∀ y : ℕ, y > 0 → y + 5600 ≡ 325 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_congruence_l1787_178775


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1787_178701

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1787_178701


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1787_178791

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (n = 3) ∧ 
  (∀ m : ℕ, m < n → ¬(
    (∃ p : ℤ, m^2 = (p+2)^5 - p^5) ∧ 
    (∃ k : ℕ, 3*m + 100 = k^2) ∧ 
    Odd m
  )) ∧
  (∃ p : ℤ, n^2 = (p+2)^5 - p^5) ∧ 
  (∃ k : ℕ, 3*n + 100 = k^2) ∧ 
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1787_178791


namespace NUMINAMATH_CALUDE_shop_width_l1787_178773

/-- Proves that the width of a rectangular shop is 8 feet given the specified conditions. -/
theorem shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 2400)
  (h2 : length = 10)
  (h3 : annual_rent_per_sqft = 360) :
  monthly_rent * 12 / (annual_rent_per_sqft * length) = 8 :=
by sorry

end NUMINAMATH_CALUDE_shop_width_l1787_178773


namespace NUMINAMATH_CALUDE_janice_started_sentences_l1787_178778

/-- Represents Janice's typing session --/
structure TypingSession where
  initial_speed : ℕ
  first_duration : ℕ
  second_speed : ℕ
  second_duration : ℕ
  third_speed : ℕ
  third_duration : ℕ
  erased_sentences : ℕ
  final_speed : ℕ
  final_duration : ℕ
  total_sentences : ℕ

/-- Calculates the number of sentences Janice started with --/
def sentences_started_with (session : TypingSession) : ℕ :=
  session.total_sentences -
  (session.initial_speed * session.first_duration +
   session.second_speed * session.second_duration +
   session.third_speed * session.third_duration -
   session.erased_sentences +
   session.final_speed * session.final_duration)

/-- Theorem stating that Janice started with 246 sentences --/
theorem janice_started_sentences (session : TypingSession)
  (h1 : session.initial_speed = 6)
  (h2 : session.first_duration = 10)
  (h3 : session.second_speed = 7)
  (h4 : session.second_duration = 10)
  (h5 : session.third_speed = 7)
  (h6 : session.third_duration = 15)
  (h7 : session.erased_sentences = 35)
  (h8 : session.final_speed = 5)
  (h9 : session.final_duration = 18)
  (h10 : session.total_sentences = 536) :
  sentences_started_with session = 246 := by
  sorry

end NUMINAMATH_CALUDE_janice_started_sentences_l1787_178778


namespace NUMINAMATH_CALUDE_dragon_boat_purchase_equations_l1787_178744

/-- Represents the purchase of items during the Dragon Boat Festival -/
structure DragonBoatPurchase where
  lotus_pouches : ℕ
  color_ropes : ℕ
  total_items : ℕ
  total_cost : ℕ
  lotus_price : ℕ
  rope_price : ℕ

/-- Theorem stating that the given system of equations correctly represents the purchase -/
theorem dragon_boat_purchase_equations (p : DragonBoatPurchase)
  (h1 : p.total_items = 20)
  (h2 : p.total_cost = 72)
  (h3 : p.lotus_price = 4)
  (h4 : p.rope_price = 3) :
  p.lotus_pouches + p.color_ropes = p.total_items ∧
  p.lotus_price * p.lotus_pouches + p.rope_price * p.color_ropes = p.total_cost :=
by sorry

end NUMINAMATH_CALUDE_dragon_boat_purchase_equations_l1787_178744


namespace NUMINAMATH_CALUDE_parallel_vectors_a_value_l1787_178709

def m (a : ℝ) : Fin 2 → ℝ := ![a, -2]
def n (a : ℝ) : Fin 2 → ℝ := ![1, 2-a]

theorem parallel_vectors_a_value :
  ∀ a : ℝ, (∃ k : ℝ, k ≠ 0 ∧ m a = k • n a) → (a = 1 + Real.sqrt 3 ∨ a = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_a_value_l1787_178709


namespace NUMINAMATH_CALUDE_profit_calculation_correct_l1787_178759

/-- Represents the profit distribution in a partnership business --/
structure ProfitDistribution where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  a_period : ℚ
  b_period : ℚ
  c_period : ℚ
  c_profit : ℚ

/-- Calculates the profit shares and total profit for a given profit distribution --/
def calculate_profits (pd : ProfitDistribution) : 
  (ℚ × ℚ × ℚ × ℚ) :=
  sorry

/-- Theorem stating the correctness of profit calculation --/
theorem profit_calculation_correct (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 2 * pd.b_investment)
  (h2 : pd.b_investment = 3 * pd.c_investment)
  (h3 : pd.a_period = 2 * pd.b_period)
  (h4 : pd.b_period = 3 * pd.c_period)
  (h5 : pd.c_profit = 3000) :
  calculate_profits pd = (108000, 27000, 3000, 138000) :=
  sorry

end NUMINAMATH_CALUDE_profit_calculation_correct_l1787_178759


namespace NUMINAMATH_CALUDE_number_properties_l1787_178740

def number : ℤ := 2023

theorem number_properties :
  (- number = -2023) ∧
  ((1 : ℚ) / number = 1 / 2023) ∧
  (|number| = 2023) := by
  sorry

end NUMINAMATH_CALUDE_number_properties_l1787_178740


namespace NUMINAMATH_CALUDE_partner_b_investment_l1787_178705

/-- Calculates the investment of partner B in a partnership business. -/
theorem partner_b_investment
  (a_investment : ℕ)
  (c_investment : ℕ)
  (total_profit : ℕ)
  (a_profit_share : ℕ)
  (h1 : a_investment = 6300)
  (h2 : c_investment = 10500)
  (h3 : total_profit = 12600)
  (h4 : a_profit_share = 3780) :
  ∃ b_investment : ℕ,
    b_investment = 13700 ∧
    (a_investment : ℚ) / (a_investment + b_investment + c_investment : ℚ) =
    (a_profit_share : ℚ) / (total_profit : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_partner_b_investment_l1787_178705


namespace NUMINAMATH_CALUDE_ten_zeros_in_expansion_l1787_178752

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The main theorem: The number of trailing zeros in (10^11 - 2)^2 is 10 -/
theorem ten_zeros_in_expansion : trailingZeros ((10^11 - 2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_zeros_in_expansion_l1787_178752


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1787_178796

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1787_178796


namespace NUMINAMATH_CALUDE_boat_price_theorem_l1787_178774

theorem boat_price_theorem (total_price : ℚ) : 
  (total_price * (6/10 : ℚ) + -- Pankrác's payment
   (total_price - total_price * (6/10 : ℚ)) * (4/10 : ℚ) + -- Servác's payment
   30 = total_price) → -- Bonifác's payment
  total_price = 125 := by
sorry

end NUMINAMATH_CALUDE_boat_price_theorem_l1787_178774


namespace NUMINAMATH_CALUDE_three_suit_probability_value_l1787_178771

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each suit in a standard deck -/
def suit_size : ℕ := 13

/-- The probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def three_suit_probability : ℚ :=
  (suit_size : ℚ) / deck_size *
  (suit_size : ℚ) / (deck_size - 1) *
  (suit_size : ℚ) / (deck_size - 2)

theorem three_suit_probability_value :
  three_suit_probability = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_suit_probability_value_l1787_178771


namespace NUMINAMATH_CALUDE_percentage_calculation_l1787_178715

theorem percentage_calculation (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1787_178715


namespace NUMINAMATH_CALUDE_twelve_point_sphere_l1787_178704

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a tetrahedron is equifacial -/
def isEquifacial (t : Tetrahedron) : Prop := sorry

/-- Calculates the base of an altitude for a face of the tetrahedron -/
def altitudeBase (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

/-- Calculates the midpoint of an altitude of the tetrahedron -/
def altitudeMidpoint (t : Tetrahedron) (vertex : Fin 4) : Point3D := sorry

/-- Calculates the intersection point of the altitudes of a face -/
def faceAltitudeIntersection (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

/-- Checks if a point lies on a sphere -/
def pointOnSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Main theorem: For an equifacial tetrahedron, there exists a sphere containing
    the bases of altitudes, midpoints of altitudes, and face altitude intersections -/
theorem twelve_point_sphere (t : Tetrahedron) (h : isEquifacial t) : 
  ∃ s : Sphere, 
    (∀ face : Fin 4, pointOnSphere (altitudeBase t face) s) ∧ 
    (∀ vertex : Fin 4, pointOnSphere (altitudeMidpoint t vertex) s) ∧
    (∀ face : Fin 4, pointOnSphere (faceAltitudeIntersection t face) s) := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_sphere_l1787_178704


namespace NUMINAMATH_CALUDE_factor_expression_l1787_178723

theorem factor_expression (x y : ℝ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1787_178723


namespace NUMINAMATH_CALUDE_pieces_per_box_l1787_178770

/-- Proves that the number of pieces per box is 6 given the initial conditions --/
theorem pieces_per_box (initial_boxes : Real) (boxes_given_away : Real) (remaining_pieces : ℕ) :
  initial_boxes = 14.0 →
  boxes_given_away = 7.0 →
  remaining_pieces = 42 →
  (remaining_pieces : Real) / (initial_boxes - boxes_given_away) = 6 := by
  sorry

#check pieces_per_box

end NUMINAMATH_CALUDE_pieces_per_box_l1787_178770


namespace NUMINAMATH_CALUDE_expression_evaluation_l1787_178741

theorem expression_evaluation : 
  let x : ℤ := -2
  (-x^2 + 5 + 4*x) + (5*x - 4 + 2*x^2) = -13 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1787_178741


namespace NUMINAMATH_CALUDE_triangle_area_l1787_178739

def a : Fin 2 → ℝ := ![4, -3]
def b : Fin 2 → ℝ := ![6, 1]

theorem triangle_area : 
  (1/2 : ℝ) * |a 0 * b 1 - a 1 * b 0| = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1787_178739


namespace NUMINAMATH_CALUDE_sandy_fish_purchase_l1787_178793

/-- Given that Sandy initially had 26 fish and now has 32 fish, 
    prove that she bought 6 fish. -/
theorem sandy_fish_purchase :
  ∀ (initial_fish current_fish purchased_fish : ℕ),
  initial_fish = 26 →
  current_fish = 32 →
  purchased_fish = current_fish - initial_fish →
  purchased_fish = 6 := by
sorry

end NUMINAMATH_CALUDE_sandy_fish_purchase_l1787_178793


namespace NUMINAMATH_CALUDE_range_of_a_l1787_178760

-- Define the complex number z
def z (x a : ℝ) : ℂ := x + (x - a) * Complex.I

-- Define the condition that |z| > |z+i| for all x in (1,2)
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x ∧ x < 2 → Complex.abs (z x a) > Complex.abs (z x a + Complex.I)

-- Theorem statement
theorem range_of_a (a : ℝ) : condition a → a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1787_178760


namespace NUMINAMATH_CALUDE_distance_to_focus_l1787_178743

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the point P
structure Point (x y : ℝ) where
  on_parabola : parabola x y
  distance_to_y_axis : x = 4

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem distance_to_focus (P : Point x y) : 
  Real.sqrt ((x - 2)^2 + y^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1787_178743


namespace NUMINAMATH_CALUDE_mike_ride_distance_l1787_178748

/-- Represents the taxi fare structure -/
structure TaxiFare where
  base_fare : ℚ
  per_mile_rate : ℚ
  toll : ℚ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare_structure : TaxiFare) (distance : ℚ) : ℚ :=
  fare_structure.base_fare + fare_structure.toll + fare_structure.per_mile_rate * distance

theorem mike_ride_distance (mike_fare annie_fare : TaxiFare) 
  (h1 : mike_fare.base_fare = 2.5)
  (h2 : mike_fare.per_mile_rate = 0.25)
  (h3 : mike_fare.toll = 0)
  (h4 : annie_fare.base_fare = 2.5)
  (h5 : annie_fare.per_mile_rate = 0.25)
  (h6 : annie_fare.toll = 5)
  (h7 : calculate_fare annie_fare 26 = calculate_fare mike_fare (46 : ℚ)) :
  ∃ x : ℚ, calculate_fare mike_fare x = calculate_fare annie_fare 26 ∧ x = 46 := by
  sorry

#eval (46 : ℚ)

end NUMINAMATH_CALUDE_mike_ride_distance_l1787_178748


namespace NUMINAMATH_CALUDE_bedbug_growth_proof_l1787_178787

def bedbug_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

theorem bedbug_growth_proof :
  bedbug_population 30 3 4 = 2430 := by
  sorry

end NUMINAMATH_CALUDE_bedbug_growth_proof_l1787_178787


namespace NUMINAMATH_CALUDE_min_value_of_f_l1787_178750

-- Define the function
def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1787_178750


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1787_178727

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), (n ≥ 1000 ∧ n < 10000) ∧
             (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
             2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
             n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1787_178727


namespace NUMINAMATH_CALUDE_system_solution_l1787_178707

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 4 ∧ 3 * x - 2 * y = 13 := by
  use 3, -2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1787_178707


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1787_178754

/-- The length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) :
  train_length = 100 →
  crossing_time = 34.997200223982084 →
  train_speed_kmph = 36 →
  let train_speed_ms := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 249.97200223982084 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1787_178754


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1787_178718

theorem integer_fraction_characterization (a b : ℕ) :
  (∃ k : ℤ, (a^3 + 1 : ℤ) = k * (2*a*b^2 + 1)) ↔
  (∃ n : ℕ, a = 2*n^2 + 1 ∧ b = n) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1787_178718


namespace NUMINAMATH_CALUDE_rhombohedron_volume_l1787_178708

/-- The volume of a rhombohedron formed by extruding a rhombus -/
theorem rhombohedron_volume
  (d1 : ℝ) (d2 : ℝ) (h : ℝ)
  (hd1 : d1 = 25)
  (hd2 : d2 = 50)
  (hh : h = 20) :
  (d1 * d2 / 2) * h = 12500 := by
  sorry

end NUMINAMATH_CALUDE_rhombohedron_volume_l1787_178708


namespace NUMINAMATH_CALUDE_abc_inequality_l1787_178742

theorem abc_inequality : ∀ (a b c : ℕ),
  a = 20^22 → b = 21^21 → c = 22^20 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1787_178742


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l1787_178767

theorem roots_sum_of_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r^12 + s^12 = 940802 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l1787_178767


namespace NUMINAMATH_CALUDE_sophie_and_hannah_fruits_l1787_178703

/-- The number of fruits eaten by Sophie and Hannah in 30 days -/
def total_fruits (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) : ℕ :=
  30 * (sophie_oranges_per_day + hannah_grapes_per_day)

/-- Theorem stating that Sophie and Hannah eat 1800 fruits in 30 days -/
theorem sophie_and_hannah_fruits :
  total_fruits 20 40 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sophie_and_hannah_fruits_l1787_178703


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_ratio_l1787_178758

structure RegularHexagon where
  vertices : Finset (Fin 6 → ℝ × ℝ)
  is_regular : sorry
  is_divided : sorry

def center (h : RegularHexagon) : ℝ × ℝ := sorry

def small_triangle (h : RegularHexagon) (i j : Fin 6) (g : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def large_triangle (h : RegularHexagon) (i j k : Fin 6) : Set (ℝ × ℝ) := sorry

def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_triangle_area_ratio (h : RegularHexagon) :
  let g := center h
  let small_tri := small_triangle h 0 1 g
  let large_tri := large_triangle h 0 3 5
  area small_tri / area large_tri = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_area_ratio_l1787_178758


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_l1787_178733

theorem sum_of_squares_positive (a b c : ℝ) (sum_zero : a + b + c = 0) (prod_neg : a * b * c < 0) :
  a^2 + b^2 > 0 ∧ b^2 + c^2 > 0 ∧ c^2 + a^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_l1787_178733


namespace NUMINAMATH_CALUDE_range_of_trig_function_l1787_178700

theorem range_of_trig_function :
  ∀ x : ℝ, -4 * Real.sqrt 3 / 9 ≤ 2 * Real.sin x ^ 2 * Real.cos x ∧
           2 * Real.sin x ^ 2 * Real.cos x ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l1787_178700


namespace NUMINAMATH_CALUDE_bret_in_seat_three_l1787_178779

-- Define the type for seats
inductive Seat
| one
| two
| three
| four

-- Define the type for people
inductive Person
| Abby
| Bret
| Carl
| Dana

-- Define the seating arrangement as a function from Seat to Person
def SeatingArrangement := Seat → Person

-- Define what it means for two people to be adjacent
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∃ (seat1 seat2 : Seat), 
    (s seat1 = p1 ∧ s seat2 = p2) ∧ 
    (seat1 = Seat.one ∧ seat2 = Seat.two ∨
     seat1 = Seat.two ∧ seat2 = Seat.three ∨
     seat1 = Seat.three ∧ seat2 = Seat.four ∨
     seat2 = Seat.one ∧ seat1 = Seat.two ∨
     seat2 = Seat.two ∧ seat1 = Seat.three ∨
     seat2 = Seat.three ∧ seat1 = Seat.four)

-- Define what it means for one person to be between two others
def between (s : SeatingArrangement) (p1 p2 p3 : Person) : Prop :=
  (s Seat.one = p1 ∧ s Seat.two = p2 ∧ s Seat.three = p3) ∨
  (s Seat.two = p1 ∧ s Seat.three = p2 ∧ s Seat.four = p3) ∨
  (s Seat.four = p1 ∧ s Seat.three = p2 ∧ s Seat.two = p3) ∨
  (s Seat.three = p1 ∧ s Seat.two = p2 ∧ s Seat.one = p3)

theorem bret_in_seat_three :
  ∀ (s : SeatingArrangement),
    (s Seat.two = Person.Abby) →
    (¬ adjacent s Person.Bret Person.Dana) →
    (¬ between s Person.Carl Person.Abby Person.Dana) →
    (s Seat.three = Person.Bret) :=
by sorry

end NUMINAMATH_CALUDE_bret_in_seat_three_l1787_178779


namespace NUMINAMATH_CALUDE_pyramid_volume_is_2000_div_3_l1787_178746

/-- A triangle in 2D space --/
structure Triangle where
  A : (ℝ × ℝ)
  B : (ℝ × ℝ)
  C : (ℝ × ℝ)

/-- The triangle described in the problem --/
def problemTriangle : Triangle :=
  { A := (0, 0),
    B := (30, 0),
    C := (15, 20) }

/-- Function to calculate the volume of the pyramid formed by folding the triangle --/
def pyramidVolume (t : Triangle) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the volume of the pyramid is 2000/3 --/
theorem pyramid_volume_is_2000_div_3 :
  pyramidVolume problemTriangle = 2000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_2000_div_3_l1787_178746


namespace NUMINAMATH_CALUDE_jeans_cost_proof_l1787_178749

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def quarters_left : ℕ := 97

def jeans_cost : ℚ := initial_amount - pizza_cost - soda_cost - (quarters_left : ℚ) * (1 / 4)

theorem jeans_cost_proof : jeans_cost = 11.50 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_proof_l1787_178749


namespace NUMINAMATH_CALUDE_not_all_products_are_effe_l1787_178731

/-- Represents a two-digit number --/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Represents a number of the form effe where e and f are single digits --/
def EffeNumber := { n : ℕ // ∃ e f : ℕ, e < 10 ∧ f < 10 ∧ n = 1001 * e + 110 * f }

/-- States that not all products of two-digit numbers result in an effe number --/
theorem not_all_products_are_effe : 
  ¬ (∀ a b : TwoDigitNumber, ∃ n : EffeNumber, a.val * b.val = n.val) :=
sorry

end NUMINAMATH_CALUDE_not_all_products_are_effe_l1787_178731


namespace NUMINAMATH_CALUDE_season_games_first_part_l1787_178765

theorem season_games_first_part 
  (total_games : ℕ) 
  (win_rate_first : ℚ) 
  (win_rate_second : ℚ) 
  (win_rate_overall : ℚ) :
  total_games = 125 →
  win_rate_first = 3/4 →
  win_rate_second = 1/2 →
  win_rate_overall = 7/10 →
  ∃ (first_part : ℕ),
    first_part = 100 ∧
    win_rate_first * first_part + win_rate_second * (total_games - first_part) = 
      win_rate_overall * total_games :=
by sorry

end NUMINAMATH_CALUDE_season_games_first_part_l1787_178765


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1787_178763

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) * (Real.sqrt 5 / Real.sqrt 12) = Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1787_178763


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l1787_178730

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 6's when rolling three standard dice -/
def expected_sixes : ℚ :=
  0 * (prob_not_six ^ 3) +
  1 * (3 * prob_six * prob_not_six ^ 2) +
  2 * (3 * prob_six ^ 2 * prob_not_six) +
  3 * (prob_six ^ 3)

theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_sixes_is_half_l1787_178730


namespace NUMINAMATH_CALUDE_grandfather_age_l1787_178761

/-- Given Yuna's age and the age differences between family members, calculate her grandfather's age -/
theorem grandfather_age (yuna_age : ℕ) (father_diff : ℕ) (grandfather_diff : ℕ) : 
  yuna_age = 8 → father_diff = 20 → grandfather_diff = 25 → 
  yuna_age + father_diff + grandfather_diff = 53 := by
  sorry

#check grandfather_age

end NUMINAMATH_CALUDE_grandfather_age_l1787_178761


namespace NUMINAMATH_CALUDE_P_greater_than_Q_l1787_178789

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > 2*x := by sorry

end NUMINAMATH_CALUDE_P_greater_than_Q_l1787_178789


namespace NUMINAMATH_CALUDE_remainder_theorem_l1787_178766

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^101 + 3^51 + 1) * q + 303 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1787_178766


namespace NUMINAMATH_CALUDE_distance_between_lines_l1787_178777

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  r : ℝ
  /-- Distance between adjacent parallel lines -/
  d : ℝ
  /-- Length of the first chord -/
  chord1 : ℝ
  /-- Length of the second chord -/
  chord2 : ℝ
  /-- Length of the third chord -/
  chord3 : ℝ
  /-- The first and third chords are equal -/
  chord1_eq_chord3 : chord1 = chord3
  /-- The first chord has length 42 -/
  chord1_eq_42 : chord1 = 42
  /-- The second chord has length 36 -/
  chord2_eq_36 : chord2 = 36

/-- The distance between adjacent parallel lines is 7.65 -/
theorem distance_between_lines (c : CircleWithParallelLines) : c.d = 7.65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l1787_178777


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l1787_178755

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem vegetable_ghee_mixture_weight
  (weight_a : ℝ) (weight_b : ℝ) (ratio_a : ℝ) (ratio_b : ℝ) (total_volume : ℝ) :
  weight_a = 900 →
  weight_b = 700 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a +
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000 = 3.280 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l1787_178755


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l1787_178728

theorem solution_of_linear_equation (x y m : ℝ) : 
  x = 1 → y = m → 3 * x - 4 * y = 7 → m = -1 := by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l1787_178728


namespace NUMINAMATH_CALUDE_constant_term_value_l1787_178747

theorem constant_term_value (y : ℝ) (c : ℝ) : 
  y = 2 → 5 * y^2 - 8 * y + c = 59 → c = 55 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l1787_178747


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l1787_178783

/-- Given a cube of side length 9 that is painted and cut into smaller cubes,
    if there are 12 smaller cubes with paint on exactly 2 sides,
    then the side length of the smaller cubes is 4.5. -/
theorem smaller_cube_side_length 
  (large_cube_side : ℝ) 
  (small_cubes_two_sides : ℕ) 
  (small_cube_side : ℝ) : 
  large_cube_side = 9 → 
  small_cubes_two_sides = 12 → 
  small_cubes_two_sides = 12 * (large_cube_side / small_cube_side - 1) → 
  small_cube_side = 4.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l1787_178783


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1787_178780

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1787_178780


namespace NUMINAMATH_CALUDE_set_equality_implies_x_zero_l1787_178717

theorem set_equality_implies_x_zero (x : ℝ) : 
  ({1, x^2} : Set ℝ) = ({1, x} : Set ℝ) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_x_zero_l1787_178717


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1787_178725

-- Define the polynomial, divisor, and quotient
def P (z : ℝ) : ℝ := 2*z^4 - 3*z^3 + 5*z^2 - 7*z + 6
def D (z : ℝ) : ℝ := 2*z - 3
def Q (z : ℝ) : ℝ := z^3 + z^2 - 4*z + 5

-- State the theorem
theorem polynomial_division_remainder :
  ∃ (R : ℝ), ∀ (z : ℝ), P z = D z * Q z + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1787_178725


namespace NUMINAMATH_CALUDE_max_area_is_10000_l1787_178795

/-- Represents a rectangular playground --/
structure Playground where
  length : ℝ
  width : ℝ

/-- The perimeter of the playground is 400 feet --/
def perimeterConstraint (p : Playground) : Prop :=
  2 * p.length + 2 * p.width = 400

/-- The length of the playground is at least 100 feet --/
def lengthConstraint (p : Playground) : Prop :=
  p.length ≥ 100

/-- The width of the playground is at least 60 feet --/
def widthConstraint (p : Playground) : Prop :=
  p.width ≥ 60

/-- The area of the playground --/
def area (p : Playground) : ℝ :=
  p.length * p.width

/-- Theorem stating that the maximum area of the playground is 10000 square feet --/
theorem max_area_is_10000 :
  ∃ (p : Playground),
    perimeterConstraint p ∧
    lengthConstraint p ∧
    widthConstraint p ∧
    (∀ (q : Playground),
      perimeterConstraint q →
      lengthConstraint q →
      widthConstraint q →
      area q ≤ area p) ∧
    area p = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_is_10000_l1787_178795


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l1787_178772

-- Part 1
theorem inequality_solution (x : ℝ) :
  x - (3 * x - 1) ≤ 2 * x + 3 ↔ x ≥ -1/2 := by sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * (x - 1) < 4 * x - 2 ∧ (1 + 4 * x) / 3 > x - 1) ↔ x > -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l1787_178772


namespace NUMINAMATH_CALUDE_sweet_salty_difference_l1787_178702

/-- Represents the number of cookies of each type --/
structure CookieCount where
  sweet : ℕ
  salty : ℕ
  chocolate : ℕ

/-- The initial number of cookies Paco had --/
def initialCookies : CookieCount :=
  { sweet := 39, salty := 18, chocolate := 12 }

/-- The number of cookies Paco ate --/
def eatenCookies : CookieCount :=
  { sweet := 27, salty := 6, chocolate := 8 }

/-- Theorem stating the difference between sweet and salty cookies eaten --/
theorem sweet_salty_difference :
  eatenCookies.sweet - eatenCookies.salty = 21 := by
  sorry


end NUMINAMATH_CALUDE_sweet_salty_difference_l1787_178702


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l1787_178729

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through given x -/
def RegressionLine.point_at (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect_at_average_point 
  (l₁ l₂ : RegressionLine) (s t : ℝ) : 
  (∀ (x : ℝ), l₁.point_at s = (s, t) ∧ l₂.point_at s = (s, t)) → 
  l₁.point_at s = l₂.point_at s := by
  sorry

#check regression_lines_intersect_at_average_point

end NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l1787_178729


namespace NUMINAMATH_CALUDE_remainder_of_n_l1787_178751

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 4) (h2 : n^3 % 7 = 6) : n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1787_178751


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1787_178769

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ q : ℤ, 4^m - 4^n = 3^(k+1) * q) ↔ (∃ p : ℤ, m - n = 3^k * p) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1787_178769


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1787_178753

/-- Represents the age ratio problem between a father and his son Ronit -/
theorem father_son_age_ratio :
  ∀ (ronit_age : ℕ) (father_age : ℕ),
  father_age = 4 * ronit_age →
  father_age + 8 = (5/2) * (ronit_age + 8) →
  (father_age + 16) = 2 * (ronit_age + 16) :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1787_178753


namespace NUMINAMATH_CALUDE_age_difference_l1787_178768

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := 16

-- Theorem to prove
theorem age_difference : greg_age - marcia_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1787_178768


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1787_178724

/-- If three consecutive integers have a product of 504, their sum is 24. -/
theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 504) → (a + b + c = 24) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1787_178724


namespace NUMINAMATH_CALUDE_total_equipment_cost_l1787_178737

/-- The number of players on the team -/
def num_players : ℕ := 25

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The cost of cleats in dollars -/
def cleats_cost : ℚ := 40

/-- The cost of a water bottle in dollars -/
def water_bottle_cost : ℚ := 12

/-- The total cost of equipment for all players on the team -/
theorem total_equipment_cost : 
  num_players * (jersey_cost + shorts_cost + socks_cost + cleats_cost + water_bottle_cost) = 2475 := by
  sorry

end NUMINAMATH_CALUDE_total_equipment_cost_l1787_178737


namespace NUMINAMATH_CALUDE_number_multiplied_by_15_l1787_178782

theorem number_multiplied_by_15 :
  ∃ x : ℝ, x * 15 = 150 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_number_multiplied_by_15_l1787_178782


namespace NUMINAMATH_CALUDE_ursula_hourly_wage_l1787_178721

/-- Calculates the hourly wage given annual salary and working hours --/
def hourly_wage (annual_salary : ℕ) (hours_per_day : ℕ) (days_per_month : ℕ) : ℚ :=
  (annual_salary : ℚ) / (12 * hours_per_day * days_per_month)

/-- Proves that Ursula's hourly wage is $8.50 given her work conditions --/
theorem ursula_hourly_wage :
  hourly_wage 16320 8 20 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ursula_hourly_wage_l1787_178721


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1787_178719

theorem sum_of_quadratic_roots (a b c d e : ℝ) (h : ∀ x, a * x^2 + b * x + c = d * x + e) :
  let x₁ := (10 + 4 * Real.sqrt 5) / 2
  let x₂ := (10 - 4 * Real.sqrt 5) / 2
  x₁ + x₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1787_178719


namespace NUMINAMATH_CALUDE_inequality_solution_l1787_178781

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  log a (x^2 - x - 2) > log a (x - 2/a) + 1

-- Theorem statement
theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → ∀ x, inequality a x ↔ x > 1 + a) ∧
  (0 < a ∧ a < 1 → ¬∃ x, inequality a x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1787_178781


namespace NUMINAMATH_CALUDE_zilla_savings_l1787_178756

/-- Given Zilla's monthly earnings and spending habits, calculate her savings --/
theorem zilla_savings (E : ℝ) (h1 : E * 0.07 = 133) (h2 : E > 0) : E - (E * 0.07 + E * 0.5) = 817 := by
  sorry

end NUMINAMATH_CALUDE_zilla_savings_l1787_178756


namespace NUMINAMATH_CALUDE_constant_value_l1787_178736

-- Define the function [[x]]
def bracket (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c

-- State the theorem
theorem constant_value :
  ∃ c : ℝ, (∀ x : ℝ, bracket x c = x^2 + 2*x + c) ∧ bracket 2 c = 12 → c = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l1787_178736


namespace NUMINAMATH_CALUDE_calculate_books_arlo_book_count_l1787_178797

/-- Given a ratio of books to pens and a total number of items, calculate the number of books. -/
theorem calculate_books (book_ratio : ℕ) (pen_ratio : ℕ) (total_items : ℕ) : ℕ :=
  let total_ratio := book_ratio + pen_ratio
  let items_per_part := total_items / total_ratio
  book_ratio * items_per_part

/-- Prove that given a ratio of books to pens of 7:3 and a total of 400 stationery items, the number of books is 280. -/
theorem arlo_book_count : calculate_books 7 3 400 = 280 := by
  sorry

end NUMINAMATH_CALUDE_calculate_books_arlo_book_count_l1787_178797


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l1787_178792

/-- Given that -32a^(2m)b and b^(3-n)a^4 are like terms, prove that m^n = n^m -/
theorem like_terms_exponent_relation (a b m n : ℕ) : 
  (2 * m = 4 ∧ 3 - n = 1) → m^n = n^m := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l1787_178792


namespace NUMINAMATH_CALUDE_students_failed_l1787_178798

def Q : ℕ := 14

theorem students_failed (x : ℕ) (h1 : x < 4 * Q) 
  (h2 : x % 3 = 0) (h3 : x % 7 = 0) (h4 : x % 2 = 0) 
  (h5 : x = 42) : x - (x / 3 + x / 7 + x / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_students_failed_l1787_178798


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1787_178714

theorem power_of_two_equality (y : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^y → y = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1787_178714
