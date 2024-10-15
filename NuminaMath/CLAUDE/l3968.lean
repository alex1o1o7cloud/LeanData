import Mathlib

namespace NUMINAMATH_CALUDE_multiples_of_15_between_25_and_205_l3968_396846

theorem multiples_of_15_between_25_and_205 : 
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 205 \ Finset.range 26)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_25_and_205_l3968_396846


namespace NUMINAMATH_CALUDE_sum_min_max_x_l3968_396822

theorem sum_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 8) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 8 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 4 :=
sorry

end NUMINAMATH_CALUDE_sum_min_max_x_l3968_396822


namespace NUMINAMATH_CALUDE_xy_value_when_sum_of_abs_is_zero_l3968_396847

theorem xy_value_when_sum_of_abs_is_zero (x y : ℝ) :
  |x - 1| + |y + 2| = 0 → x * y = -2 := by
sorry

end NUMINAMATH_CALUDE_xy_value_when_sum_of_abs_is_zero_l3968_396847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3968_396898

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (h1 : seq.S 4 = 3 * seq.S 2)
  (h2 : seq.a 7 = 15) :
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3968_396898


namespace NUMINAMATH_CALUDE_least_number_divisibility_l3968_396855

theorem least_number_divisibility (x : ℕ) : x = 171011 ↔ 
  (∀ y : ℕ, y < x → ¬(41 ∣ (1076 + y) ∧ 59 ∣ (1076 + y) ∧ 67 ∣ (1076 + y))) ∧
  (41 ∣ (1076 + x) ∧ 59 ∣ (1076 + x) ∧ 67 ∣ (1076 + x)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l3968_396855


namespace NUMINAMATH_CALUDE_max_value_of_f_l3968_396890

-- Define the function f
def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) := by
sorry


end NUMINAMATH_CALUDE_max_value_of_f_l3968_396890


namespace NUMINAMATH_CALUDE_georginas_parrot_days_l3968_396820

/-- The number of days Georgina has had her parrot -/
def days_with_parrot (total_phrases current_phrases_per_week initial_phrases days_per_week : ℕ) : ℕ :=
  ((total_phrases - initial_phrases) / current_phrases_per_week) * days_per_week

/-- Proof that Georgina has had her parrot for 49 days -/
theorem georginas_parrot_days : 
  days_with_parrot 17 2 3 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_georginas_parrot_days_l3968_396820


namespace NUMINAMATH_CALUDE_positive_real_pair_with_integer_product_and_floor_sum_l3968_396878

theorem positive_real_pair_with_integer_product_and_floor_sum (x y : ℝ) : 
  x > 0 → y > 0 → (∃ n : ℤ, x * y = n) → x + y = ⌊x^2 - y^2⌋ → 
  ∃ d : ℕ, d ≥ 2 ∧ x = d ∧ y = d - 1 := by
sorry

end NUMINAMATH_CALUDE_positive_real_pair_with_integer_product_and_floor_sum_l3968_396878


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l3968_396832

/-- A digit in base d is a natural number less than d. -/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- The value of a two-digit number AB in base d. -/
def TwoDigitValue (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d (d : ℕ) (A B : Digit d) 
  (h_d : d > 7)
  (h_sum : TwoDigitValue d A B + TwoDigitValue d A A = 1 * d * d + 7 * d + 2) :
  A.val - B.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l3968_396832


namespace NUMINAMATH_CALUDE_cricket_equipment_cost_l3968_396833

theorem cricket_equipment_cost (bat_cost : ℕ) (ball_cost : ℕ) : 
  (7 * bat_cost + 6 * ball_cost = 3800) →
  (3 * bat_cost + 5 * ball_cost = 1750) →
  (bat_cost = 500) →
  ball_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_cricket_equipment_cost_l3968_396833


namespace NUMINAMATH_CALUDE_consecutive_integer_product_l3968_396837

theorem consecutive_integer_product (n : ℤ) : 
  (6 ∣ n * (n + 1)) ∨ (n * (n + 1) % 18 = 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_l3968_396837


namespace NUMINAMATH_CALUDE_difference_of_squares_l3968_396880

theorem difference_of_squares : 601^2 - 597^2 = 4792 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3968_396880


namespace NUMINAMATH_CALUDE_hoseok_candy_count_l3968_396824

/-- The number of candies Hoseok has of type A -/
def candies_A : ℕ := 2

/-- The number of candies Hoseok has of type B -/
def candies_B : ℕ := 5

/-- The total number of candies Hoseok has -/
def total_candies : ℕ := candies_A + candies_B

theorem hoseok_candy_count : total_candies = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_candy_count_l3968_396824


namespace NUMINAMATH_CALUDE_polynomial_identity_l3968_396875

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3968_396875


namespace NUMINAMATH_CALUDE_nested_expression_equals_one_l3968_396800

def nested_expression : ℤ :=
  (3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1)

theorem nested_expression_equals_one : nested_expression = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_one_l3968_396800


namespace NUMINAMATH_CALUDE_line_points_determine_k_l3968_396842

/-- A line contains the points (6,10), (-2,k), and (-10,6). -/
def line_contains_points (k : ℝ) : Prop :=
  ∃ (m b : ℝ), 
    (10 = m * 6 + b) ∧
    (k = m * (-2) + b) ∧
    (6 = m * (-10) + b)

/-- If a line contains the points (6,10), (-2,k), and (-10,6), then k = 8. -/
theorem line_points_determine_k :
  ∀ k : ℝ, line_contains_points k → k = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_line_points_determine_k_l3968_396842


namespace NUMINAMATH_CALUDE_spinner_probability_l3968_396806

theorem spinner_probability : ∀ (p_C p_D p_E : ℚ),
  (p_C = p_D) →
  (p_D = p_E) →
  (1/5 : ℚ) + (1/5 : ℚ) + p_C + p_D + p_E = 1 →
  p_C = (1/5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3968_396806


namespace NUMINAMATH_CALUDE_y_share_per_x_rupee_l3968_396888

/-- Given a sum divided among x, y, and z, prove that y gets 9/20 rupees for each rupee x gets. -/
theorem y_share_per_x_rupee (x y z : ℝ) (total : ℝ) (y_share : ℝ) (y_per_x : ℝ) : 
  total = 234 →
  y_share = 54 →
  x + y + z = total →
  y = y_per_x * x →
  z = 0.5 * x →
  y_per_x = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_y_share_per_x_rupee_l3968_396888


namespace NUMINAMATH_CALUDE_isabel_ds_games_l3968_396834

theorem isabel_ds_games (initial_games : ℕ) (remaining_games : ℕ) (given_games : ℕ) : 
  initial_games = 90 → remaining_games = 3 → given_games = initial_games - remaining_games → given_games = 87 := by
  sorry

end NUMINAMATH_CALUDE_isabel_ds_games_l3968_396834


namespace NUMINAMATH_CALUDE_ratio_a5_a7_l3968_396861

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  decreasing : ∀ n, a (n + 1) < a n
  geometric : ∀ n k, a (n + k) = a n * (a 2 / a 1) ^ k
  prop1 : a 2 * a 8 = 6
  prop2 : a 4 + a 6 = 5

/-- The main theorem about the ratio of a_5 to a_7 -/
theorem ratio_a5_a7 (seq : SpecialGeometricSequence) : seq.a 5 / seq.a 7 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a5_a7_l3968_396861


namespace NUMINAMATH_CALUDE_f_properties_l3968_396879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1 / (a^x)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = f a x) ∧
  (∀ x y : ℝ, 0 ≤ x → x < y → f a x < f a y) ∧
  (∀ x y : ℝ, x < y → y ≤ 0 → f a x > f a y) ∧
  (Set.Ioo (-2 : ℝ) 0 = {x : ℝ | f a (x - 1) > f a (2*x + 1)}) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3968_396879


namespace NUMINAMATH_CALUDE_pet_store_birds_l3968_396887

theorem pet_store_birds (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) (dogs : ℕ) :
  total_animals = 180 →
  talking_birds = 64 →
  non_talking_birds = 13 →
  dogs = 40 →
  talking_birds = 4 * ((total_animals - (talking_birds + non_talking_birds + dogs)) / 4) →
  talking_birds + non_talking_birds = 124 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3968_396887


namespace NUMINAMATH_CALUDE_complement_intersection_l3968_396848

def U : Set ℕ := {x | 0 < x ∧ x < 7}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 4}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_l3968_396848


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l3968_396895

theorem quadratic_always_positive_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l3968_396895


namespace NUMINAMATH_CALUDE_bob_earnings_l3968_396892

def regular_rate : ℕ := 5
def overtime_rate : ℕ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

def calculate_earnings (hours_worked : ℕ) : ℕ :=
  regular_rate * regular_hours + 
  overtime_rate * (hours_worked - regular_hours)

theorem bob_earnings : 
  calculate_earnings first_week_hours + calculate_earnings second_week_hours = 472 := by
  sorry

end NUMINAMATH_CALUDE_bob_earnings_l3968_396892


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3968_396827

theorem smallest_n_congruence : ∃ n : ℕ+, (∀ m : ℕ+, 19 * m ≡ 1453 [MOD 8] → n ≤ m) ∧ 19 * n ≡ 1453 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3968_396827


namespace NUMINAMATH_CALUDE_complex_magnitude_3_minus_10i_l3968_396831

theorem complex_magnitude_3_minus_10i :
  Complex.abs (3 - 10 * Complex.I) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_3_minus_10i_l3968_396831


namespace NUMINAMATH_CALUDE_max_value_of_a_l3968_396899

theorem max_value_of_a (x y : ℝ) (hx : x > 1/3) (hy : y > 1) :
  (∀ a : ℝ, (9 * x^2) / (a^2 * (y - 1)) + (y^2) / (a^2 * (3 * x - 1)) ≥ 1) →
  (∃ a_max : ℝ, a_max = 2 * Real.sqrt 2 ∧
    ∀ a : ℝ, (9 * x^2) / (a^2 * (y - 1)) + (y^2) / (a^2 * (3 * x - 1)) ≥ 1 → a ≤ a_max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3968_396899


namespace NUMINAMATH_CALUDE_inverse_of_periodic_function_l3968_396852

def PeriodicFunction (f : ℝ → ℝ) :=
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x

def SmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) :=
  PeriodicFunction f ∧ T > 0 ∧ ∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S

def InverseInInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo a b, g (f x) = x ∧ f (g x) = x

theorem inverse_of_periodic_function
  (f : ℝ → ℝ) (T : ℝ)
  (h_periodic : SmallestPositivePeriod f T)
  (h_inverse : InverseInInterval f 0 T) :
  ∃ g : ℝ → ℝ, ∀ x ∈ Set.Ioo T (2 * T),
    g (f x) = x ∧ f (g x) = x ∧ g x = (Classical.choose h_inverse) (x - T) + T :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_periodic_function_l3968_396852


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l3968_396826

/-- Given sets A, B, and C defined as follows:
    A = {x | 2 ≤ x < 7}
    B = {x | 3 < x ≤ 10}
    C = {x | a-5 < x < a}
    Prove that if C is a non-empty subset of A ∪ B, then 7 ≤ a ≤ 10. -/
theorem set_inclusion_implies_a_range (a : ℝ) :
  let A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
  let B : Set ℝ := {x | 3 < x ∧ x ≤ 10}
  let C : Set ℝ := {x | a - 5 < x ∧ x < a}
  C.Nonempty → C ⊆ A ∪ B → 7 ≤ a ∧ a ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l3968_396826


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3968_396808

theorem necessary_not_sufficient (a b : ℝ) : 
  ((a > b) → (a > b - 1)) ∧ ¬((a > b - 1) → (a > b)) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3968_396808


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3968_396835

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 72 = 1 ∧ 
    (13 * b) % 72 = 1 ∧ 
    ((3 * a + 9 * b) % 72) % 72 = 6 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3968_396835


namespace NUMINAMATH_CALUDE_cricket_players_l3968_396843

theorem cricket_players (B C Both Total : ℕ) : 
  B = 7 → 
  Both = 3 → 
  Total = 9 → 
  Total = B + C - Both → 
  C = 5 :=
by sorry

end NUMINAMATH_CALUDE_cricket_players_l3968_396843


namespace NUMINAMATH_CALUDE_triangle_angles_theorem_l3968_396891

theorem triangle_angles_theorem (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  Real.tan A * Real.tan C = 2 + Real.sqrt 3 -- Given condition
  →
  ((A = π / 4 ∧ B = π / 3 ∧ C = 5 * π / 12) ∨
   (A = 5 * π / 12 ∧ B = π / 3 ∧ C = π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_theorem_l3968_396891


namespace NUMINAMATH_CALUDE_right_triangle_division_l3968_396897

/-- In a right triangle divided by lines parallel to the legs through a point on the hypotenuse,
    if the areas of the two smaller triangles are m and n times the area of the square respectively,
    then n = 1/(4m). -/
theorem right_triangle_division (m n : ℝ) : m > 0 → n > 0 → n = 1 / (4 * m) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_division_l3968_396897


namespace NUMINAMATH_CALUDE_total_spent_is_36_98_l3968_396836

/-- Calculates the total amount spent on video games --/
def total_spent (football_price : ℝ) (football_discount : ℝ) 
                (strategy_price : ℝ) (strategy_tax : ℝ)
                (batman_price_euro : ℝ) (exchange_rate : ℝ) : ℝ :=
  let football_discounted := football_price * (1 - football_discount)
  let strategy_with_tax := strategy_price * (1 + strategy_tax)
  let batman_price_usd := batman_price_euro * exchange_rate
  football_discounted + strategy_with_tax + batman_price_usd

/-- Theorem stating the total amount spent on video games --/
theorem total_spent_is_36_98 :
  total_spent 16 0.1 9.46 0.05 11 1.15 = 36.98 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_36_98_l3968_396836


namespace NUMINAMATH_CALUDE_work_completion_time_l3968_396894

/-- The number of days it takes to complete the remaining work after additional persons join -/
def remaining_days (initial_persons : ℕ) (total_days : ℕ) (days_worked : ℕ) (additional_persons : ℕ) : ℚ :=
  let initial_work_rate := 1 / (initial_persons * total_days : ℚ)
  let work_done := initial_persons * days_worked * initial_work_rate
  let remaining_work := 1 - work_done
  let new_work_rate := (initial_persons + additional_persons : ℚ) * initial_work_rate
  remaining_work / new_work_rate

theorem work_completion_time :
  remaining_days 12 18 6 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3968_396894


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3968_396860

/-- The atomic weight of Copper (Cu) in g/mol -/
def atomic_weight_Cu : ℝ := 63.546

/-- The atomic weight of Carbon (C) in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- The atomic weight of Oxygen (O) in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Cu atoms in the compound -/
def num_Cu : ℕ := 1

/-- The number of C atoms in the compound -/
def num_C : ℕ := 1

/-- The number of O atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  (num_Cu : ℝ) * atomic_weight_Cu +
  (num_C : ℝ) * atomic_weight_C +
  (num_O : ℝ) * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 123.554 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3968_396860


namespace NUMINAMATH_CALUDE_clock_cost_price_l3968_396856

theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_difference : ℝ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (sold_at_10_percent : ℝ) * cost_price * 1.1 + 
    (sold_at_20_percent : ℝ) * cost_price * 1.2 - 
    (total_clocks : ℝ) * cost_price * 1.15 = uniform_profit_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l3968_396856


namespace NUMINAMATH_CALUDE_f_properties_l3968_396801

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, f x b c = -f (-x) b c → c = 0) ∧
  (∃! x, f x 0 c = 0) ∧
  (∀ x, f (-x) b c + f x b c = 2 * c) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3968_396801


namespace NUMINAMATH_CALUDE_family_park_cost_l3968_396844

/-- Calculates the total cost for a family to visit a park and one attraction -/
def total_cost (num_children num_adults entrance_fee child_attraction_fee adult_attraction_fee : ℕ) : ℕ :=
  (num_children + num_adults) * entrance_fee + 
  num_children * child_attraction_fee + 
  num_adults * adult_attraction_fee

/-- Proves that the total cost for the given family is $55 -/
theorem family_park_cost : 
  total_cost 4 3 5 2 4 = 55 := by
  sorry

end NUMINAMATH_CALUDE_family_park_cost_l3968_396844


namespace NUMINAMATH_CALUDE_lcm_of_135_and_195_l3968_396872

theorem lcm_of_135_and_195 : Nat.lcm 135 195 = 1755 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_135_and_195_l3968_396872


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l3968_396868

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l3968_396868


namespace NUMINAMATH_CALUDE_q_is_false_l3968_396866

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l3968_396866


namespace NUMINAMATH_CALUDE_coin_flip_difference_l3968_396823

/-- Given 211 total coin flips with 65 heads, the difference between the number of tails and heads is 81. -/
theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) 
    (h1 : total_flips = 211)
    (h2 : heads = 65) : 
  total_flips - heads - heads = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_difference_l3968_396823


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3968_396825

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3968_396825


namespace NUMINAMATH_CALUDE_profit_percentage_l3968_396889

theorem profit_percentage (cost selling : ℝ) (h : cost > 0) :
  60 * cost = 40 * selling →
  (selling - cost) / cost * 100 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3968_396889


namespace NUMINAMATH_CALUDE_family_age_relations_l3968_396838

structure Family where
  rachel_age : ℕ
  grandfather_age : ℕ
  mother_age : ℕ
  father_age : ℕ
  aunt_age : ℕ

def family_ages : Family where
  rachel_age := 12
  grandfather_age := 7 * 12
  mother_age := (7 * 12) / 2
  father_age := (7 * 12) / 2 + 5
  aunt_age := 7 * 12 - 8

theorem family_age_relations (f : Family) :
  f.rachel_age = 12 ∧
  f.grandfather_age = 7 * f.rachel_age ∧
  f.mother_age = f.grandfather_age / 2 ∧
  f.father_age = f.mother_age + 5 ∧
  f.aunt_age = f.grandfather_age - 8 →
  f = family_ages :=
by sorry

end NUMINAMATH_CALUDE_family_age_relations_l3968_396838


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3968_396870

theorem complex_modulus_problem (z : ℂ) (h : z - 2*Complex.I = z*Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3968_396870


namespace NUMINAMATH_CALUDE_fraction_equality_l3968_396883

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3968_396883


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3968_396814

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 60) (h2 : b = 80) 
  (h3 : c^2 = a^2 + b^2) : c = 100 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3968_396814


namespace NUMINAMATH_CALUDE_modified_star_angle_sum_l3968_396859

/-- A modified n-pointed star --/
structure ModifiedStar where
  n : ℕ
  is_valid : n ≥ 6

/-- The sum of interior angles of the modified star --/
def interior_angle_sum (star : ModifiedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles of a modified n-pointed star is 180(n-2) degrees --/
theorem modified_star_angle_sum (star : ModifiedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_modified_star_angle_sum_l3968_396859


namespace NUMINAMATH_CALUDE_expression_factorization_l3968_396865

/-- 
Given a, b, and c, prove that the expression 
a^4 (b^2 - c^2) + b^4 (c^2 - a^2) + c^4 (a^2 - b^2) 
can be factorized into the form (a - b)(b - c)(c - a) q(a, b, c),
where q(a, b, c) = a^3 b^2 + a^2 b^3 + b^3 c^2 + b^2 c^3 + c^3 a^2 + c^2 a^3
-/
theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * (a^3 * b^2 + a^2 * b^3 + b^3 * c^2 + b^2 * c^3 + c^3 * a^2 + c^2 * a^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3968_396865


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3968_396869

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def increasing_f := ∀ x y, x < y → f x < f y
def f_zero_is_neg_one := f 0 = -1
def f_three_is_one := f 3 = 1

-- Define the solution set
def solution_set (f : ℝ → ℝ) := {x : ℝ | |f x| < 1}

-- State the theorem
theorem solution_set_is_open_interval
  (h_increasing : increasing_f f)
  (h_zero : f_zero_is_neg_one f)
  (h_three : f_three_is_one f) :
  solution_set f = Set.Ioo 0 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3968_396869


namespace NUMINAMATH_CALUDE_square_difference_plus_constant_problem_solution_l3968_396884

theorem square_difference_plus_constant (a b c : ℤ) :
  a ^ 2 - b ^ 2 + c = (a + b) * (a - b) + c := by sorry

theorem problem_solution :
  632 ^ 2 - 568 ^ 2 + 100 = 76900 := by sorry

end NUMINAMATH_CALUDE_square_difference_plus_constant_problem_solution_l3968_396884


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l3968_396877

def is_eventually_periodic (a : ℕ → ℚ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ ∀ n ≥ m, a (n + k) = a n

theorem sequence_eventually_periodic
  (a : ℕ → ℚ)
  (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
  (h2 : ∀ n : ℕ, |a n| ≤ 2)
  : is_eventually_periodic a :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l3968_396877


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3968_396819

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  sorry


end NUMINAMATH_CALUDE_complex_fraction_simplification_l3968_396819


namespace NUMINAMATH_CALUDE_marks_lawyer_hourly_rate_l3968_396873

/-- Calculates the lawyer's hourly rate for Mark's speeding ticket case -/
theorem marks_lawyer_hourly_rate 
  (base_fine : ℕ) 
  (speed_fine_rate : ℕ) 
  (marks_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hours : ℕ) 
  (total_owed : ℕ) 
  (h1 : base_fine = 50)
  (h2 : speed_fine_rate = 2)
  (h3 : marks_speed = 75)
  (h4 : speed_limit = 30)
  (h5 : court_costs = 300)
  (h6 : lawyer_hours = 3)
  (h7 : total_owed = 820) :
  (total_owed - (2 * (base_fine + speed_fine_rate * (marks_speed - speed_limit)) + court_costs)) / lawyer_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_marks_lawyer_hourly_rate_l3968_396873


namespace NUMINAMATH_CALUDE_sum_of_sqrt_sequence_l3968_396864

theorem sum_of_sqrt_sequence :
  Real.sqrt 6 + Real.sqrt (6 + 8) + Real.sqrt (6 + 8 + 10) + 
  Real.sqrt (6 + 8 + 10 + 12) + Real.sqrt (6 + 8 + 10 + 12 + 14) = 
  Real.sqrt 6 + Real.sqrt 14 + Real.sqrt 24 + 6 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_sequence_l3968_396864


namespace NUMINAMATH_CALUDE_value_of_k_l3968_396821

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-collinear vectors e₁ and e₂
variable (e₁ e₂ : V)
variable (h_non_collinear : ∀ (a b : ℝ), a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- Define the given vector relationships
variable (h_AB : B - A = 2 • e₁ + k • e₂)
variable (h_CB : B - C = e₁ + 3 • e₂)
variable (h_CD : D - C = 2 • e₁ - e₂)

-- Define collinearity of points A, B, and D
variable (h_collinear : ∃ (t : ℝ), B - A = t • (D - B))

-- Theorem statement
theorem value_of_k : k = -8 := by sorry

end NUMINAMATH_CALUDE_value_of_k_l3968_396821


namespace NUMINAMATH_CALUDE_math_team_combinations_l3968_396854

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of girls in the math club --/
def num_girls : ℕ := 4

/-- The number of boys in the math club --/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team --/
def girls_in_team : ℕ := 2

/-- The number of boys to be chosen for the team --/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l3968_396854


namespace NUMINAMATH_CALUDE_embroidery_project_time_l3968_396816

/-- Represents the embroidery project details -/
structure EmbroideryProject where
  flower_stitches : ℕ
  flower_speed : ℕ
  unicorn_stitches : ℕ
  unicorn_speed : ℕ
  godzilla_stitches : ℕ
  godzilla_speed : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzilla : ℕ
  break_duration : ℕ
  work_duration : ℕ

/-- Calculates the total time needed for the embroidery project -/
def total_time (project : EmbroideryProject) : ℕ :=
  let total_stitches := project.flower_stitches * project.num_flowers +
                        project.unicorn_stitches * project.num_unicorns +
                        project.godzilla_stitches * project.num_godzilla
  let total_work_time := (total_stitches / project.flower_speed * project.num_flowers +
                          total_stitches / project.unicorn_speed * project.num_unicorns +
                          total_stitches / project.godzilla_speed * project.num_godzilla)
  let num_breaks := total_work_time / project.work_duration
  let total_break_time := num_breaks * project.break_duration
  total_work_time + total_break_time

/-- The main theorem stating the total time for the given embroidery project -/
theorem embroidery_project_time :
  let project : EmbroideryProject := {
    flower_stitches := 60,
    flower_speed := 4,
    unicorn_stitches := 180,
    unicorn_speed := 5,
    godzilla_stitches := 800,
    godzilla_speed := 3,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzilla := 1,
    break_duration := 5,
    work_duration := 30
  }
  total_time project = 1310 := by
  sorry


end NUMINAMATH_CALUDE_embroidery_project_time_l3968_396816


namespace NUMINAMATH_CALUDE_expected_teachers_with_masters_l3968_396811

def total_teachers : ℕ := 320
def masters_degree_ratio : ℚ := 1 / 4

theorem expected_teachers_with_masters :
  (total_teachers : ℚ) * masters_degree_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_expected_teachers_with_masters_l3968_396811


namespace NUMINAMATH_CALUDE_bruce_age_bruce_current_age_l3968_396886

theorem bruce_age : ℕ → Prop :=
  fun b =>
    let son_age : ℕ := 8
    let future_years : ℕ := 6
    (b + future_years = 3 * (son_age + future_years)) →
    b = 36

-- Proof
theorem bruce_current_age : ∃ b : ℕ, bruce_age b :=
  sorry

end NUMINAMATH_CALUDE_bruce_age_bruce_current_age_l3968_396886


namespace NUMINAMATH_CALUDE_xyz_product_l3968_396817

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3968_396817


namespace NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l3968_396862

/-- A line in the xy-plane defined by the equation kx - y + 1 = k, where k is a real number. -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 + 1 = k}

/-- The fixed point (1, 1) -/
def fixed_point : ℝ × ℝ := (1, 1)

/-- Theorem stating that all lines pass through the fixed point (1, 1) -/
theorem all_lines_pass_through_fixed_point :
  ∀ k : ℝ, fixed_point ∈ line k := by
  sorry


end NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l3968_396862


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l3968_396804

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) 
  (h3 : 1 ≤ a - 2*b) (h4 : a - 2*b ≤ 3) : 
  ∃ (x : ℝ), x = a + 3*b ∧ -11/3 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l3968_396804


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3968_396876

/-- Proves that the charge for each additional 1/5 mile is $0.40 given the initial and total charges -/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_charge : ℚ) (ride_length : ℚ) 
  (h1 : initial_charge = 2.1)
  (h2 : total_charge = 17.7)
  (h3 : ride_length = 8) :
  let additional_increments := (ride_length * 5) - 1
  (total_charge - initial_charge) / additional_increments = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3968_396876


namespace NUMINAMATH_CALUDE_digit_sum_l3968_396881

/-- Given two digits x and y, if 3x * y4 = 156, then x + y = 13 -/
theorem digit_sum (x y : Nat) : 
  x ≤ 9 → y ≤ 9 → (30 + x) * (10 * y + 4) = 156 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l3968_396881


namespace NUMINAMATH_CALUDE_smallest_among_four_l3968_396841

theorem smallest_among_four (a b c d : ℝ) :
  a = |-2| ∧ b = -1 ∧ c = 0 ∧ d = -1/2 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_four_l3968_396841


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3968_396850

/-- Geometric sequence with first term 3 and second sum 9 -/
def geometric_sequence (n : ℕ) : ℝ :=
  3 * 2^(n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℝ :=
  3 * (2^n - 1)

theorem geometric_sequence_properties :
  (geometric_sequence 1 = 3) ∧
  (geometric_sum 2 = 9) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sum n = 3 * (2^n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3968_396850


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3968_396882

def M : ℕ := 35 * 36 * 65 * 280

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors M) * 62 = sum_of_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3968_396882


namespace NUMINAMATH_CALUDE_product_sum_fractions_l3968_396829

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l3968_396829


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l3968_396809

/-- The combined cost of a garden table and a bench -/
def combined_cost (bench_cost : ℝ) (table_cost : ℝ) : ℝ :=
  bench_cost + table_cost

theorem garden_furniture_cost :
  ∀ (bench_cost : ℝ) (table_cost : ℝ),
  bench_cost = 250.0 →
  table_cost = 2 * bench_cost →
  combined_cost bench_cost table_cost = 750.0 := by
sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l3968_396809


namespace NUMINAMATH_CALUDE_blue_corduroy_glasses_count_l3968_396813

theorem blue_corduroy_glasses_count (total_students : ℕ) 
  (blue_shirt_percent : ℚ) (corduroy_percent : ℚ) (glasses_percent : ℚ) :
  total_students = 1500 →
  blue_shirt_percent = 35 / 100 →
  corduroy_percent = 20 / 100 →
  glasses_percent = 15 / 100 →
  ⌊total_students * blue_shirt_percent * corduroy_percent * glasses_percent⌋ = 15 := by
sorry

end NUMINAMATH_CALUDE_blue_corduroy_glasses_count_l3968_396813


namespace NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l3968_396885

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the theorem
theorem function_domain_range_implies_b_equals_two :
  ∀ b : ℝ,
  (∀ x ∈ Set.Icc 2 (2*b), f x ∈ Set.Icc 2 (2*b)) ∧
  (∀ y ∈ Set.Icc 2 (2*b), ∃ x ∈ Set.Icc 2 (2*b), f x = y) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l3968_396885


namespace NUMINAMATH_CALUDE_intersection_segment_length_l3968_396812

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l3968_396812


namespace NUMINAMATH_CALUDE_find_x_l3968_396810

-- Define the binary operation
def binary_op (n : ℤ) (x : ℤ) : ℤ := n - (n * x)

-- State the theorem
theorem find_x : ∃ x : ℤ, 
  (∀ n : ℕ, n > 2 → binary_op n x ≥ 10) ∧ 
  (binary_op 2 x < 10) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3968_396810


namespace NUMINAMATH_CALUDE_new_person_weight_l3968_396858

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  n = 15 ∧ weight_increase = 3.8 ∧ replaced_weight = 75 →
  n * weight_increase + replaced_weight = 132 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3968_396858


namespace NUMINAMATH_CALUDE_f_diff_max_min_eq_one_l3968_396805

/-- The function f(x) = x^2 - 2bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x - 1

/-- The closed interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

/-- The statement that the difference between the maximum and minimum values of f(x) on [0, 1] is 1 -/
def diffMaxMin (b : ℝ) : Prop :=
  ∃ (max min : ℝ), (∀ x ∈ I, f b x ≤ max) ∧
                   (∀ x ∈ I, min ≤ f b x) ∧
                   (max - min = 1)

/-- The main theorem -/
theorem f_diff_max_min_eq_one :
  ∀ b : ℝ, diffMaxMin b ↔ (b = 0 ∨ b = 1) :=
sorry

end NUMINAMATH_CALUDE_f_diff_max_min_eq_one_l3968_396805


namespace NUMINAMATH_CALUDE_seans_total_spend_is_21_l3968_396851

/-- The total amount Sean spent on his Sunday purchases -/
def seans_total_spend : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let focaccia : ℝ := 4.00
  let latte : ℝ := 2.50
  let num_lattes : ℕ := 2

  almond_croissant + salami_cheese_croissant + plain_croissant + focaccia + (num_lattes : ℝ) * latte

/-- Theorem stating that Sean's total spend is $21.00 -/
theorem seans_total_spend_is_21 : seans_total_spend = 21 := by
  sorry

end NUMINAMATH_CALUDE_seans_total_spend_is_21_l3968_396851


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3968_396803

theorem rectangle_dimension_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.15 * L
  let new_area := 1.035 * (L * B)
  let new_breadth := new_area / new_length
  (new_breadth / B) = 0.9 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3968_396803


namespace NUMINAMATH_CALUDE_always_bal_answer_l3968_396863

/-- Represents a guest in the castle -/
structure Guest where
  is_reliable : Bool

/-- Represents the possible questions that can be asked -/
inductive Question
  | q1  -- "Правильно ли ответить «бaл» на вопрос, надежны ли вы?"
  | q2  -- "Надежны ли вы в том и только в том случае, если «бaл» означает «да»?"

/-- The answer "бaл" -/
def bal : String := "бaл"

/-- Function representing a guest's response to a question -/
def guest_response (g : Guest) (q : Question) : String :=
  match q with
  | Question.q1 => bal
  | Question.q2 => bal

/-- Theorem stating that any guest will always answer "бaл" to either question -/
theorem always_bal_answer (g : Guest) (q : Question) :
  guest_response g q = bal := by sorry

end NUMINAMATH_CALUDE_always_bal_answer_l3968_396863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3968_396830

/-- An arithmetic sequence {a_n} with a_2 = 2 and S_11 = 66 -/
def a (n : ℕ) : ℚ :=
  sorry

/-- The sum of the first n terms of the sequence a -/
def S (n : ℕ) : ℚ :=
  sorry

/-- The sequence b_n defined as 1 / (a_n * a_n+1) -/
def b (n : ℕ) : ℚ :=
  1 / (a n * a (n + 1))

/-- The sum of the first n terms of sequence b -/
def b_sum (n : ℕ) : ℚ :=
  sorry

theorem arithmetic_sequence_property :
  a 2 = 2 ∧ S 11 = 66 ∧ ∀ n : ℕ, n > 0 → b_sum n < 1 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3968_396830


namespace NUMINAMATH_CALUDE_negative_sum_l3968_396840

theorem negative_sum : (-2) + (-5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3968_396840


namespace NUMINAMATH_CALUDE_expression_evaluation_l3968_396874

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3968_396874


namespace NUMINAMATH_CALUDE_vector_identity_l3968_396896

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- For any four points A, B, C, and D in a real inner product space,
    the vector DA + CD - CB is equal to BA. -/
theorem vector_identity (A B C D : V) :
  (D - A) + (C - D) - (C - B) = B - A :=
sorry

end NUMINAMATH_CALUDE_vector_identity_l3968_396896


namespace NUMINAMATH_CALUDE_triangle_segment_ratio_l3968_396845

-- Define the triangle and points
variable (A B C E D F : ℝ × ℝ)

-- Define the conditions
variable (h_E_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • B)
variable (h_D_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ D = (1 - s) • B + s • C)
variable (h_AE_EB : ∃ k : ℝ, dist A E = k * dist E B ∧ k = 1/3)
variable (h_CD_DB : ∃ m : ℝ, dist C D = m * dist D B ∧ m = 1/2)
variable (h_F_intersect : ∃ u v : ℝ, 0 < u ∧ u < 1 ∧ 0 < v ∧ v < 1 ∧
  F = (1 - u) • A + u • D ∧ F = (1 - v) • C + v • E)

-- Define the theorem
theorem triangle_segment_ratio :
  dist E F / dist F C + dist A F / dist F D = 3/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_segment_ratio_l3968_396845


namespace NUMINAMATH_CALUDE_largest_number_problem_l3968_396815

theorem largest_number_problem (a b c d e : ℕ) 
  (sum1 : a + b + c + d = 350)
  (sum2 : a + b + c + e = 370)
  (sum3 : a + b + d + e = 390)
  (sum4 : a + c + d + e = 410)
  (sum5 : b + c + d + e = 430) :
  max a (max b (max c (max d e))) = 138 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3968_396815


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l3968_396853

theorem sqrt_nested_expression : Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4)) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l3968_396853


namespace NUMINAMATH_CALUDE_club_truncator_probability_l3968_396802

/-- Represents the outcome of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- The total number of matches played by Club Truncator -/
def total_matches : ℕ := 5

/-- The probability of each match result -/
def match_probability : ℚ := 1 / 3

/-- Calculates the probability of having more wins than losses in the season -/
noncomputable def prob_more_wins_than_losses : ℚ := sorry

/-- The main theorem stating the probability of more wins than losses -/
theorem club_truncator_probability : prob_more_wins_than_losses = 32 / 81 := by sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l3968_396802


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3968_396839

theorem trigonometric_identity : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180) ^ 2 - Real.sin (35 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3968_396839


namespace NUMINAMATH_CALUDE_unique_paths_equal_binomial_coefficient_l3968_396867

/-- The number of rows in the grid -/
def n : ℕ := 6

/-- The number of columns in the grid -/
def m : ℕ := 6

/-- The total number of steps required to reach the destination -/
def total_steps : ℕ := n + m

/-- The number of ways to choose n right moves out of total_steps moves -/
def num_paths : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of unique paths from A to B is equal to C(12,6) -/
theorem unique_paths_equal_binomial_coefficient : 
  num_paths = 924 := by sorry

end NUMINAMATH_CALUDE_unique_paths_equal_binomial_coefficient_l3968_396867


namespace NUMINAMATH_CALUDE_problem_solution_l3968_396818

theorem problem_solution (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 3) : 
  1 / (a * b + c - 1) + 1 / (b * c + a - 1) + 1 / (c * a + b - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3968_396818


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3968_396871

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - x^2 + x - 2

-- Define the roots
variable (p q r : ℝ)

-- State the theorem
theorem sum_of_cubes_of_roots :
  (f p = 0) → (f q = 0) → (f r = 0) → 
  p ≠ q → q ≠ r → r ≠ p →
  p^3 + q^3 + r^3 = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3968_396871


namespace NUMINAMATH_CALUDE_ram_distances_l3968_396807

/-- Represents a mountain on the map -/
structure Mountain where
  name : String
  scale : ℝ  -- km per inch

/-- Represents a location on the map -/
structure Location where
  name : String
  distanceA : ℝ  -- distance from mountain A in inches
  distanceB : ℝ  -- distance from mountain B in inches

def map_distance : ℝ := 312  -- inches
def actual_distance : ℝ := 136  -- km

def mountainA : Mountain := { name := "A", scale := 1 }
def mountainB : Mountain := { name := "B", scale := 2 }

def ram_location : Location := { name := "Ram", distanceA := 25, distanceB := 40 }

/-- Calculates the actual distance from a location to a mountain -/
def actual_distance_to_mountain (loc : Location) (m : Mountain) : ℝ :=
  if m.name = "A" then loc.distanceA * m.scale else loc.distanceB * m.scale

theorem ram_distances :
  actual_distance_to_mountain ram_location mountainA = 25 ∧
  actual_distance_to_mountain ram_location mountainB = 80 := by
  sorry

end NUMINAMATH_CALUDE_ram_distances_l3968_396807


namespace NUMINAMATH_CALUDE_inequality_proof_l3968_396849

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2) :
  (abs (a + b + c - a * b * c) ≤ 2) ∧
  (abs (a^3 + b^3 + c^3 - 3 * a * b * c) ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3968_396849


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l3968_396893

theorem probability_of_y_selection 
  (prob_x : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_x = 1/5)
  (h2 : prob_both = 0.05714285714285714) :
  prob_both / prob_x = 0.2857142857142857 := by
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l3968_396893


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3968_396857

theorem smallest_prime_dividing_sum : ∃ p : ℕ, 
  Nat.Prime p ∧ p > 5 ∧ p ∣ (2^14 + 3^15) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (2^14 + 3^15) → q ≥ p := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3968_396857


namespace NUMINAMATH_CALUDE_power_inequality_l3968_396828

theorem power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^(a/5) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3968_396828
