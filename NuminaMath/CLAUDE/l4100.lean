import Mathlib

namespace polynomial_constant_term_l4100_410059

def g (p q r s : ℤ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 + r*x + s

theorem polynomial_constant_term 
  (p q r s : ℤ) 
  (h1 : p + q + r + s = 168)
  (h2 : ∀ x : ℝ, g p q r s x = 0 → (∃ n : ℤ, x = -n ∧ n > 0))
  (h3 : ∀ x : ℝ, (g p q r s x = 0) → (g p q r s (-x) = 0)) :
  s = 144 := by sorry

end polynomial_constant_term_l4100_410059


namespace partnership_profit_l4100_410004

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  /-- A's investment ratio relative to B --/
  a_ratio : ℚ
  /-- B's investment ratio relative to C --/
  b_ratio : ℚ
  /-- B's share of the profit --/
  b_share : ℕ

/-- Calculates the total profit given the partnership details --/
def calculate_total_profit (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that given the specified partnership details, the total profit is 7700 --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_ratio = 3)
  (h2 : p.b_ratio = 2/3)
  (h3 : p.b_share = 1400) :
  calculate_total_profit p = 7700 :=
sorry

end partnership_profit_l4100_410004


namespace range_of_trig_function_l4100_410066

theorem range_of_trig_function :
  ∀ x : ℝ, -4 * Real.sqrt 3 / 9 ≤ 2 * Real.sin x ^ 2 * Real.cos x ∧
           2 * Real.sin x ^ 2 * Real.cos x ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end range_of_trig_function_l4100_410066


namespace fish_lives_12_years_l4100_410082

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by sorry

end fish_lives_12_years_l4100_410082


namespace hyperbola_equation_l4100_410084

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

end hyperbola_equation_l4100_410084


namespace magnitude_of_complex_number_l4100_410039

theorem magnitude_of_complex_number (z : ℂ) : z = Complex.I * (3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end magnitude_of_complex_number_l4100_410039


namespace inequality_proof_l4100_410010

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_increasing : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 5 → f x < f y

-- State the theorem
theorem inequality_proof : f 4 > f (-Real.pi) ∧ f (-Real.pi) > f 3 :=
sorry

end inequality_proof_l4100_410010


namespace number_multiplied_by_15_l4100_410072

theorem number_multiplied_by_15 :
  ∃ x : ℝ, x * 15 = 150 ∧ x = 10 := by sorry

end number_multiplied_by_15_l4100_410072


namespace grain_production_theorem_l4100_410097

theorem grain_production_theorem (planned_wheat planned_corn actual_wheat actual_corn : ℝ) :
  planned_wheat + planned_corn = 18 →
  actual_wheat + actual_corn = 20 →
  actual_wheat = planned_wheat * 1.12 →
  actual_corn = planned_corn * 1.10 →
  actual_wheat = 11.2 ∧ actual_corn = 8.8 := by
  sorry

end grain_production_theorem_l4100_410097


namespace not_all_products_are_effe_l4100_410096

/-- Represents a two-digit number --/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Represents a number of the form effe where e and f are single digits --/
def EffeNumber := { n : ℕ // ∃ e f : ℕ, e < 10 ∧ f < 10 ∧ n = 1001 * e + 110 * f }

/-- States that not all products of two-digit numbers result in an effe number --/
theorem not_all_products_are_effe : 
  ¬ (∀ a b : TwoDigitNumber, ∃ n : EffeNumber, a.val * b.val = n.val) :=
sorry

end not_all_products_are_effe_l4100_410096


namespace sum_x_and_5_nonpositive_l4100_410034

theorem sum_x_and_5_nonpositive (x : ℝ) : (x + 5 ≤ 0) ↔ (∀ y : ℝ, y ≤ 0 → x + 5 ≤ y) := by
  sorry

end sum_x_and_5_nonpositive_l4100_410034


namespace quadratic_root_implies_u_value_l4100_410025

theorem quadratic_root_implies_u_value (u : ℝ) : 
  (3 * (((-15 - Real.sqrt 205) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 205) / 6) + u = 0) → 
  u = 5/3 := by
sorry

end quadratic_root_implies_u_value_l4100_410025


namespace square_units_digits_correct_l4100_410046

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

end square_units_digits_correct_l4100_410046


namespace expression_evaluation_l4100_410042

theorem expression_evaluation : 
  let x : ℤ := -2
  (-x^2 + 5 + 4*x) + (5*x - 4 + 2*x^2) = -13 :=
by sorry

end expression_evaluation_l4100_410042


namespace intersection_implies_a_value_l4100_410033

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by sorry

end intersection_implies_a_value_l4100_410033


namespace vegetable_ghee_mixture_weight_l4100_410089

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

end vegetable_ghee_mixture_weight_l4100_410089


namespace partner_b_investment_l4100_410092

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

end partner_b_investment_l4100_410092


namespace janice_started_sentences_l4100_410071

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

end janice_started_sentences_l4100_410071


namespace pyramid_volume_is_2000_div_3_l4100_410047

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

end pyramid_volume_is_2000_div_3_l4100_410047


namespace regression_lines_intersect_at_average_point_l4100_410087

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

end regression_lines_intersect_at_average_point_l4100_410087


namespace smallest_value_theorem_l4100_410013

theorem smallest_value_theorem (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  ∃ (min_val : ℝ), min_val = (1 : ℝ) / 2 ∧ ∀ (x : ℝ), 8 * x^3 + 6 * x^2 + 7 * x + 5 = 4 → 3 * x + 2 ≥ min_val :=
by sorry

end smallest_value_theorem_l4100_410013


namespace average_of_four_numbers_l4100_410083

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := by
sorry

end average_of_four_numbers_l4100_410083


namespace equal_water_amounts_l4100_410031

theorem equal_water_amounts (hot_fill_time cold_fill_time : ℝ) 
  (h_hot : hot_fill_time = 23)
  (h_cold : cold_fill_time = 19) :
  let delay := 2
  let hot_rate := 1 / hot_fill_time
  let cold_rate := 1 / cold_fill_time
  let total_time := hot_fill_time / 2 + delay
  hot_rate * total_time = cold_rate * (total_time - delay) :=
by sorry

end equal_water_amounts_l4100_410031


namespace vanaspati_percentage_in_original_mixture_l4100_410027

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_percentage : ℝ

/-- Calculates the percentage of vanaspati in a ghee mixture -/
def vanaspati_percentage (mixture : GheeMixture) : ℝ :=
  100 - mixture.pure_percentage

theorem vanaspati_percentage_in_original_mixture 
  (original : GheeMixture)
  (h_original_total : original.total = 10)
  (h_original_pure : original.pure_percentage = 60)
  (h_after_addition : 
    let new_total := original.total + 10
    let new_pure := original.total * (original.pure_percentage / 100) + 10
    (100 - (new_pure / new_total * 100)) = 20) :
  vanaspati_percentage original = 40 := by
  sorry

#eval vanaspati_percentage { total := 10, pure_percentage := 60 }

end vanaspati_percentage_in_original_mixture_l4100_410027


namespace smallest_four_digit_divisible_by_first_five_primes_l4100_410060

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), (n ≥ 1000 ∧ n < 10000) ∧
             (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
             2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
             n = 2310 :=
by sorry

end smallest_four_digit_divisible_by_first_five_primes_l4100_410060


namespace greatest_n_value_l4100_410093

theorem greatest_n_value (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 ∧ ∃ (m : ℤ), m = 11 ∧ 93 * m^3 ≤ 145800 := by
  sorry

end greatest_n_value_l4100_410093


namespace gcd_1426_1643_l4100_410006

theorem gcd_1426_1643 : Nat.gcd 1426 1643 = 31 := by
  sorry

end gcd_1426_1643_l4100_410006


namespace exists_71_cubes_l4100_410085

/-- Represents the number of cubes after a series of divisions -/
def num_cubes : ℕ → ℕ
| 0 => 1
| (n + 1) => num_cubes n + 7

/-- Theorem stating that it's possible to obtain 71 cubes through the division process -/
theorem exists_71_cubes : ∃ n : ℕ, num_cubes n = 71 := by
  sorry

end exists_71_cubes_l4100_410085


namespace french_fries_cooking_time_l4100_410058

/-- Calculates the remaining cooking time in seconds given the recommended time in minutes and the actual cooking time in seconds. -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem stating that for a recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds. -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end french_fries_cooking_time_l4100_410058


namespace sophie_and_hannah_fruits_l4100_410062

/-- The number of fruits eaten by Sophie and Hannah in 30 days -/
def total_fruits (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) : ℕ :=
  30 * (sophie_oranges_per_day + hannah_grapes_per_day)

/-- Theorem stating that Sophie and Hannah eat 1800 fruits in 30 days -/
theorem sophie_and_hannah_fruits :
  total_fruits 20 40 = 1800 := by
  sorry

end sophie_and_hannah_fruits_l4100_410062


namespace six_digit_divisibility_l4100_410007

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : c ≥ 0 ∧ c ≤ 9) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) % 1001 = 0 := by
  sorry

end six_digit_divisibility_l4100_410007


namespace season_games_first_part_l4100_410069

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

end season_games_first_part_l4100_410069


namespace competition_probabilities_l4100_410076

def score_prob (p1 p2 p3 : ℝ) : ℝ × ℝ :=
  let prob_300 := p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
  let prob_400 := p1 * p2 * p3
  (prob_300, prob_300 + prob_400)

theorem competition_probabilities :
  let (prob_300, prob_at_least_300) := score_prob 0.8 0.7 0.6
  prob_300 = 0.228 ∧ prob_at_least_300 = 0.564 := by
  sorry

end competition_probabilities_l4100_410076


namespace circle_diameter_theorem_l4100_410094

theorem circle_diameter_theorem (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 16 * Real.pi → A = Real.pi * r^2 → d = 2 * r → 3 * d = 24 := by
  sorry

end circle_diameter_theorem_l4100_410094


namespace sugar_ratio_l4100_410026

theorem sugar_ratio (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ = 24) (h₂ : a₄ = 3)
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :
  a₂ / a₁ = 1 / 2 := by
sorry

end sugar_ratio_l4100_410026


namespace land_conversion_rates_l4100_410028

/-- Represents the daily conversion rates and conditions for land conversion --/
structure LandConversion where
  total_area : ℝ
  rate_ratio : ℝ
  time_difference : ℝ
  team_b_rate : ℝ

/-- Theorem stating the correct daily conversion rates given the conditions --/
theorem land_conversion_rates (lc : LandConversion)
  (h1 : lc.total_area = 1500)
  (h2 : lc.rate_ratio = 1.2)
  (h3 : lc.time_difference = 5)
  (h4 : lc.total_area / lc.team_b_rate - lc.time_difference = lc.total_area / (lc.rate_ratio * lc.team_b_rate)) :
  lc.team_b_rate = 50 ∧ lc.rate_ratio * lc.team_b_rate = 60 := by
  sorry

end land_conversion_rates_l4100_410028


namespace projection_property_l4100_410021

/-- A projection that takes (3, -3) to (75/26, -15/26) -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property :
  projection (3, -3) = (75/26, -15/26) →
  projection ((5, 7) + (-3, -4)) = (35/26, -7/26) :=
by
  sorry

end projection_property_l4100_410021


namespace root_in_interval_l4100_410032

-- Define the function f(x) = x^3 + x - 1
def f (x : ℝ) : ℝ := x^3 + x - 1

-- State the theorem
theorem root_in_interval :
  f 0.6 < 0 → f 0.7 > 0 → ∃ x ∈ Set.Ioo 0.6 0.7, f x = 0 := by
  sorry

end root_in_interval_l4100_410032


namespace car_speed_problem_l4100_410000

theorem car_speed_problem (x : ℝ) :
  x > 0 →
  (x + 60) / 2 = 75 →
  x = 90 :=
by
  sorry

end car_speed_problem_l4100_410000


namespace necessary_but_not_sufficient_condition_l4100_410080

theorem necessary_but_not_sufficient_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x^2 - 3*a*x + 2*a^2 ≤ 0 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ x^2 - 3*a*x + 2*a^2 > 0) →
  a > 1 := by
sorry

end necessary_but_not_sufficient_condition_l4100_410080


namespace divisibility_equivalence_l4100_410045

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ q : ℤ, 4^m - 4^n = 3^(k+1) * q) ↔ (∃ p : ℤ, m - n = 3^k * p) := by
  sorry

end divisibility_equivalence_l4100_410045


namespace lattice_points_on_segment_l4100_410016

/-- The number of lattice points on a line segment -/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment -/
theorem lattice_points_on_segment : latticePointCount 5 23 60 353 = 56 := by
  sorry

end lattice_points_on_segment_l4100_410016


namespace bridge_length_calculation_l4100_410088

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

end bridge_length_calculation_l4100_410088


namespace factor_expression_l4100_410040

theorem factor_expression (x y : ℝ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end factor_expression_l4100_410040


namespace least_positive_congruence_l4100_410054

theorem least_positive_congruence :
  ∃! x : ℕ, x > 0 ∧ x + 5600 ≡ 325 [ZMOD 15] ∧ ∀ y : ℕ, y > 0 → y + 5600 ≡ 325 [ZMOD 15] → x ≤ y :=
by sorry

end least_positive_congruence_l4100_410054


namespace largest_two_digit_prime_factor_l4100_410070

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ is_two_digit p ∧ (p ∣ binomial_coefficient 300 150) ∧
  ∀ (q : ℕ), Nat.Prime q → is_two_digit q → (q ∣ binomial_coefficient 300 150) → q ≤ p :=
by
  use 89
  sorry

#check largest_two_digit_prime_factor

end largest_two_digit_prime_factor_l4100_410070


namespace ratio_squares_equality_l4100_410023

theorem ratio_squares_equality : (1625^2 - 1612^2) / (1631^2 - 1606^2) = 13 / 25 := by
  sorry

end ratio_squares_equality_l4100_410023


namespace shooting_sequences_l4100_410065

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

end shooting_sequences_l4100_410065


namespace parabola_focus_l4100_410055

/-- The parabola is defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola with equation y = ax^2 has coordinates (0, 1/(4a)) -/
def is_focus (a x y : ℝ) : Prop := x = 0 ∧ y = 1 / (4 * a)

/-- Prove that the focus of the parabola y = (1/4)x^2 has coordinates (0, 1) -/
theorem parabola_focus :
  is_focus (1/4) 0 1 :=
sorry

end parabola_focus_l4100_410055


namespace fraction_sum_equality_l4100_410012

theorem fraction_sum_equality : (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end fraction_sum_equality_l4100_410012


namespace fifteenth_prime_l4100_410008

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 5 = 11) → (nth_prime 15 = 47) :=
sorry

end fifteenth_prime_l4100_410008


namespace f_composition_nine_equals_one_eighth_l4100_410077

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x else 2^x

theorem f_composition_nine_equals_one_eighth :
  f (f 9) = 1/8 := by
  sorry

end f_composition_nine_equals_one_eighth_l4100_410077


namespace intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l4100_410018

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3*a}

-- Theorem for part (1)
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (A ∩ B a).Nonempty ↔ (4/3 ≤ a ∧ a < 4) :=
sorry

-- Theorem for part (2)
theorem union_equals_interval_iff_a_equals_two (a : ℝ) :
  A ∪ B a = {x : ℝ | 2 < x ∧ x < 6} ↔ a = 2 :=
sorry

end intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l4100_410018


namespace perfect_squares_between_50_and_250_l4100_410002

theorem perfect_squares_between_50_and_250 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 250) (Finset.range 16)).card = 8 := by
  sorry

end perfect_squares_between_50_and_250_l4100_410002


namespace fence_painting_rate_l4100_410022

theorem fence_painting_rate (num_fences : ℕ) (fence_length : ℕ) (total_earnings : ℚ) :
  num_fences = 50 →
  fence_length = 500 →
  total_earnings = 5000 →
  total_earnings / (num_fences * fence_length : ℚ) = 0.20 := by
  sorry

end fence_painting_rate_l4100_410022


namespace remainder_of_n_l4100_410053

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 4) (h2 : n^3 % 7 = 6) : n % 7 = 5 := by
  sorry

end remainder_of_n_l4100_410053


namespace income_savings_percentage_l4100_410078

theorem income_savings_percentage (I S : ℝ) 
  (h1 : S > 0) 
  (h2 : I > S) 
  (h3 : (I - S) + (1.35 * I - 2 * S) = 2 * (I - S)) : 
  S / I = 0.35 := by
sorry

end income_savings_percentage_l4100_410078


namespace largest_angle_in_789_ratio_triangle_l4100_410020

/-- Given a triangle with interior angles in a 7:8:9 ratio, 
    the largest interior angle measures 67.5 degrees. -/
theorem largest_angle_in_789_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 180 →
    b = (8/7) * a →
    c = (9/7) * a →
    max a (max b c) = 67.5 :=
by sorry

end largest_angle_in_789_ratio_triangle_l4100_410020


namespace power_of_two_equality_l4100_410074

theorem power_of_two_equality (y : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^y → y = 37 := by
  sorry

end power_of_two_equality_l4100_410074


namespace base_six_addition_l4100_410001

/-- Given a base-6 addition 4AB₆ + 41₆ = 53A₆, prove that A + B = 9 in base 10 -/
theorem base_six_addition (A B : ℕ) : 
  (4 * 6^2 + A * 6 + B) + (4 * 6 + 1) = 5 * 6^2 + 3 * 6 + A → A + B = 9 :=
by sorry

end base_six_addition_l4100_410001


namespace percentage_calculation_l4100_410075

theorem percentage_calculation (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end percentage_calculation_l4100_410075


namespace boat_price_theorem_l4100_410091

theorem boat_price_theorem (total_price : ℚ) : 
  (total_price * (6/10 : ℚ) + -- Pankrác's payment
   (total_price - total_price * (6/10 : ℚ)) * (4/10 : ℚ) + -- Servác's payment
   30 = total_price) → -- Bonifác's payment
  total_price = 125 := by
sorry

end boat_price_theorem_l4100_410091


namespace max_tan_A_l4100_410017

theorem max_tan_A (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : 3 * Real.sin A = Real.cos (A + B) * Real.sin B) :
  ∃ (max_tan_A : Real), ∀ (A' B' : Real),
    0 < A' → A' < π/2 → 0 < B' → B' < π/2 →
    3 * Real.sin A' = Real.cos (A' + B') * Real.sin B' →
    Real.tan A' ≤ max_tan_A ∧
    max_tan_A = Real.sqrt 3 / 12 := by
  sorry

end max_tan_A_l4100_410017


namespace zilla_savings_l4100_410049

/-- Given Zilla's monthly earnings and spending habits, calculate her savings --/
theorem zilla_savings (E : ℝ) (h1 : E * 0.07 = 133) (h2 : E > 0) : E - (E * 0.07 + E * 0.5) = 817 := by
  sorry

end zilla_savings_l4100_410049


namespace distance_between_lines_l4100_410056

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

end distance_between_lines_l4100_410056


namespace three_suit_probability_value_l4100_410050

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

end three_suit_probability_value_l4100_410050


namespace green_blue_difference_after_two_borders_l4100_410030

/-- Calculates the number of tiles in a border of a hexagonal figure -/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of green tiles to a hexagonal figure -/
def add_border (fig : HexFigure) (border_size : ℕ) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles border_size }

theorem green_blue_difference_after_two_borders :
  let initial_figure : HexFigure := { blue_tiles := 14, green_tiles := 8 }
  let first_border := add_border initial_figure 3
  let second_border := add_border first_border 5
  second_border.green_tiles - second_border.blue_tiles = 42 := by
  sorry

end green_blue_difference_after_two_borders_l4100_410030


namespace map_scale_l4100_410079

/-- If 15 cm on a map represents 90 km, then 20 cm represents 120 km -/
theorem map_scale (map_length : ℝ) (actual_distance : ℝ) : 
  (15 * map_length = 90 * actual_distance) → 
  (20 * map_length = 120 * actual_distance) := by
  sorry

end map_scale_l4100_410079


namespace polynomial_division_remainder_l4100_410061

-- Define the polynomial, divisor, and quotient
def P (z : ℝ) : ℝ := 2*z^4 - 3*z^3 + 5*z^2 - 7*z + 6
def D (z : ℝ) : ℝ := 2*z - 3
def Q (z : ℝ) : ℝ := z^3 + z^2 - 4*z + 5

-- State the theorem
theorem polynomial_division_remainder :
  ∃ (R : ℝ), ∀ (z : ℝ), P z = D z * Q z + R :=
by sorry

end polynomial_division_remainder_l4100_410061


namespace square_area_error_l4100_410064

theorem square_area_error (side_error : Real) (area_error : Real) : 
  side_error = 0.19 → area_error = 0.4161 := by
  sorry

end square_area_error_l4100_410064


namespace min_value_of_f_l4100_410052

-- Define the function
def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -1) := by
  sorry

end min_value_of_f_l4100_410052


namespace range_of_a_l4100_410081

-- Define the complex number z
def z (x a : ℝ) : ℂ := x + (x - a) * Complex.I

-- Define the condition that |z| > |z+i| for all x in (1,2)
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x ∧ x < 2 → Complex.abs (z x a) > Complex.abs (z x a + Complex.I)

-- Theorem statement
theorem range_of_a (a : ℝ) : condition a → a ≤ 1/2 :=
by sorry

end range_of_a_l4100_410081


namespace grandfather_age_l4100_410090

/-- Given Yuna's age and the age differences between family members, calculate her grandfather's age -/
theorem grandfather_age (yuna_age : ℕ) (father_diff : ℕ) (grandfather_diff : ℕ) : 
  yuna_age = 8 → father_diff = 20 → grandfather_diff = 25 → 
  yuna_age + father_diff + grandfather_diff = 53 := by
  sorry

#check grandfather_age

end grandfather_age_l4100_410090


namespace f_seven_half_value_l4100_410003

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_half_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_unit : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f 7.5 = -0.5 := by
  sorry

end f_seven_half_value_l4100_410003


namespace expected_sixes_is_half_l4100_410067

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

end expected_sixes_is_half_l4100_410067


namespace carries_cucumber_harvest_l4100_410014

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected cucumber harvest given garden dimensions and planting parameters -/
def expected_harvest (garden : GardenDimensions) (plants_per_sqft : ℝ) (cucumbers_per_plant : ℝ) : ℝ :=
  garden.length * garden.width * plants_per_sqft * cucumbers_per_plant

/-- Theorem stating that Carrie's garden will yield 9000 cucumbers -/
theorem carries_cucumber_harvest :
  let garden := GardenDimensions.mk 10 12
  let plants_per_sqft := 5
  let cucumbers_per_plant := 15
  expected_harvest garden plants_per_sqft cucumbers_per_plant = 9000 := by
  sorry


end carries_cucumber_harvest_l4100_410014


namespace initial_bananas_count_l4100_410068

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

end initial_bananas_count_l4100_410068


namespace valid_selection_count_l4100_410005

/-- Represents a dad in the TV show -/
structure Dad :=
  (id : Nat)

/-- Represents a kid in the TV show -/
structure Kid :=
  (id : Nat)
  (isGirl : Bool)
  (dad : Dad)

/-- Represents the selection of one dad and three kids -/
structure Selection :=
  (selectedDad : Dad)
  (selectedKids : Finset Kid)

/-- The set of all dads -/
def allDads : Finset Dad := sorry

/-- The set of all kids -/
def allKids : Finset Kid := sorry

/-- Kimi is a boy -/
def kimi : Kid := sorry

/-- Stone is a boy -/
def stone : Kid := sorry

/-- Predicate to check if a selection is valid -/
def isValidSelection (s : Selection) : Prop :=
  s.selectedKids.card = 3 ∧
  (∃ k ∈ s.selectedKids, k.isGirl) ∧
  (kimi ∈ s.selectedKids ↔ kimi.dad = s.selectedDad) ∧
  (stone ∈ s.selectedKids ↔ stone.dad ≠ s.selectedDad)

/-- The set of all possible valid selections -/
def allValidSelections : Finset Selection :=
  sorry

theorem valid_selection_count :
  allValidSelections.card = 12 :=
sorry

end valid_selection_count_l4100_410005


namespace simplify_and_rationalize_l4100_410044

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) * (Real.sqrt 5 / Real.sqrt 12) = Real.sqrt 5 / 4 := by
  sorry

end simplify_and_rationalize_l4100_410044


namespace number_properties_l4100_410041

def number : ℤ := 2023

theorem number_properties :
  (- number = -2023) ∧
  ((1 : ℚ) / number = 1 / 2023) ∧
  (|number| = 2023) := by
  sorry

end number_properties_l4100_410041


namespace product_relation_l4100_410009

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z :=
by sorry

end product_relation_l4100_410009


namespace quadrilateral_area_l4100_410036

/-- Quadrilateral PQRS with given side lengths -/
structure Quadrilateral :=
  (PS : ℝ)
  (SR : ℝ)
  (PQ : ℝ)
  (RQ : ℝ)

/-- The area of the quadrilateral PQRS is 36 -/
theorem quadrilateral_area (q : Quadrilateral) 
  (h1 : q.PS = 3)
  (h2 : q.SR = 4)
  (h3 : q.PQ = 13)
  (h4 : q.RQ = 12) : 
  ∃ (area : ℝ), area = 36 := by
  sorry

#check quadrilateral_area

end quadrilateral_area_l4100_410036


namespace arccos_one_half_equals_pi_third_l4100_410035

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l4100_410035


namespace solution_of_linear_equation_l4100_410086

theorem solution_of_linear_equation (x y m : ℝ) : 
  x = 1 → y = m → 3 * x - 4 * y = 7 → m = -1 := by sorry

end solution_of_linear_equation_l4100_410086


namespace sum_of_possible_x_values_l4100_410043

theorem sum_of_possible_x_values (x z : ℝ) (h1 : |x - z| = 100) (h2 : |z - 12| = 60) : 
  ∃ (x1 x2 x3 x4 : ℝ), 
    (|x1 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x2 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x3 - z| = 100 ∧ |z - 12| = 60) ∧
    (|x4 - z| = 100 ∧ |z - 12| = 60) ∧
    x1 + x2 + x3 + x4 = 48 ∧
    (∀ y : ℝ, (|y - z| = 100 ∧ |z - 12| = 60) → (y = x1 ∨ y = x2 ∨ y = x3 ∨ y = x4)) :=
by sorry

end sum_of_possible_x_values_l4100_410043


namespace P_greater_than_Q_l4100_410057

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > 2*x := by sorry

end P_greater_than_Q_l4100_410057


namespace remainder_theorem_l4100_410037

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^101 + 3^51 + 1) * q + 303 :=
sorry

end remainder_theorem_l4100_410037


namespace sum_of_squares_positive_l4100_410098

theorem sum_of_squares_positive (a b c : ℝ) (sum_zero : a + b + c = 0) (prod_neg : a * b * c < 0) :
  a^2 + b^2 > 0 ∧ b^2 + c^2 > 0 ∧ c^2 + a^2 > 0 := by
  sorry

end sum_of_squares_positive_l4100_410098


namespace smaller_cube_side_length_l4100_410073

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

end smaller_cube_side_length_l4100_410073


namespace constant_term_value_l4100_410048

theorem constant_term_value (y : ℝ) (c : ℝ) : 
  y = 2 → 5 * y^2 - 8 * y + c = 59 → c = 55 := by
  sorry

end constant_term_value_l4100_410048


namespace inequality_solution_system_of_inequalities_solution_l4100_410051

-- Part 1
theorem inequality_solution (x : ℝ) :
  x - (3 * x - 1) ≤ 2 * x + 3 ↔ x ≥ -1/2 := by sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * (x - 1) < 4 * x - 2 ∧ (1 + 4 * x) / 3 > x - 1) ↔ x > -1 := by sorry

end inequality_solution_system_of_inequalities_solution_l4100_410051


namespace shop_width_l4100_410099

/-- Proves that the width of a rectangular shop is 8 feet given the specified conditions. -/
theorem shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 2400)
  (h2 : length = 10)
  (h3 : annual_rent_per_sqft = 360) :
  monthly_rent * 12 / (annual_rent_per_sqft * length) = 8 :=
by sorry

end shop_width_l4100_410099


namespace course_selection_count_l4100_410015

/-- The number of different course selection schemes for 3 students choosing from 3 elective courses -/
def course_selection_schemes : ℕ := 18

/-- The number of elective courses -/
def num_courses : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 3

/-- Proposition that each student chooses only one course -/
axiom one_course_per_student : True

/-- Proposition that exactly one course has no students -/
axiom one_empty_course : True

/-- Theorem stating that the number of different course selection schemes is 18 -/
theorem course_selection_count : course_selection_schemes = 18 := by
  sorry

end course_selection_count_l4100_410015


namespace twelve_point_sphere_l4100_410063

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

end twelve_point_sphere_l4100_410063


namespace fly_journey_l4100_410019

theorem fly_journey (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 90) :
  let d := 2 * r
  let b := Real.sqrt (d^2 - s^2)
  d + s + b = 314 :=
by sorry

end fly_journey_l4100_410019


namespace average_gas_mileage_calculation_l4100_410011

theorem average_gas_mileage_calculation (distance_to_university : ℝ) (sedan_efficiency : ℝ)
  (weekend_trip_distance : ℝ) (truck_efficiency : ℝ)
  (h1 : distance_to_university = 150)
  (h2 : sedan_efficiency = 25)
  (h3 : weekend_trip_distance = 200)
  (h4 : truck_efficiency = 15) :
  let total_distance := distance_to_university + weekend_trip_distance
  let sedan_gas_used := distance_to_university / sedan_efficiency
  let truck_gas_used := weekend_trip_distance / truck_efficiency
  let total_gas_used := sedan_gas_used + truck_gas_used
  total_distance / total_gas_used = 1050 / 58 := by sorry

end average_gas_mileage_calculation_l4100_410011


namespace malcolm_route_ratio_l4100_410029

/-- Malcolm's route to school problem -/
theorem malcolm_route_ratio : 
  ∀ (r : ℝ), 
  (6 + 6*r + (1/3)*(6 + 6*r) + 18 = 42) → 
  r = 17/4 := by
sorry

end malcolm_route_ratio_l4100_410029


namespace ursula_hourly_wage_l4100_410038

/-- Calculates the hourly wage given annual salary and working hours --/
def hourly_wage (annual_salary : ℕ) (hours_per_day : ℕ) (days_per_month : ℕ) : ℚ :=
  (annual_salary : ℚ) / (12 * hours_per_day * days_per_month)

/-- Proves that Ursula's hourly wage is $8.50 given her work conditions --/
theorem ursula_hourly_wage :
  hourly_wage 16320 8 20 = 17/2 := by
  sorry

end ursula_hourly_wage_l4100_410038


namespace pieces_per_box_l4100_410095

/-- Proves that the number of pieces per box is 6 given the initial conditions --/
theorem pieces_per_box (initial_boxes : Real) (boxes_given_away : Real) (remaining_pieces : ℕ) :
  initial_boxes = 14.0 →
  boxes_given_away = 7.0 →
  remaining_pieces = 42 →
  (remaining_pieces : Real) / (initial_boxes - boxes_given_away) = 6 := by
  sorry

#check pieces_per_box

end pieces_per_box_l4100_410095


namespace zeros_after_one_in_10000_to_50_l4100_410024

theorem zeros_after_one_in_10000_to_50 : 
  (∃ n : ℕ, 10000^50 = 10^n ∧ n = 200) :=
by sorry

end zeros_after_one_in_10000_to_50_l4100_410024
