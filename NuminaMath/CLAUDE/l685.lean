import Mathlib

namespace sin_seven_pi_sixths_l685_68563

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end sin_seven_pi_sixths_l685_68563


namespace increasing_cubic_function_a_range_l685_68544

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

-- State the theorem
theorem increasing_cubic_function_a_range :
  (∀ x y : ℝ, x < y → f a x < f a y) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by sorry

end increasing_cubic_function_a_range_l685_68544


namespace g_of_3_equals_135_l685_68522

/-- Given that g(x) = 3x^4 - 5x^3 + 2x^2 + x + 6, prove that g(3) = 135 -/
theorem g_of_3_equals_135 : 
  let g : ℝ → ℝ := λ x ↦ 3*x^4 - 5*x^3 + 2*x^2 + x + 6
  g 3 = 135 := by sorry

end g_of_3_equals_135_l685_68522


namespace unique_solution_power_equation_l685_68533

theorem unique_solution_power_equation (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x : ℝ, a^x + b^x = c^x :=
by
  sorry

end unique_solution_power_equation_l685_68533


namespace probability_at_least_one_heart_in_top_three_l685_68584

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of hearts in a standard deck
def numHearts : ℕ := 13

-- Define the number of cards we're considering (top three)
def topCards : ℕ := 3

-- Theorem statement
theorem probability_at_least_one_heart_in_top_three :
  let prob : ℚ := 1 - (totalCards - numHearts).descFactorial topCards / totalCards.descFactorial topCards
  prob = 325 / 425 := by sorry

end probability_at_least_one_heart_in_top_three_l685_68584


namespace elena_pen_purchase_l685_68597

theorem elena_pen_purchase (cost_x : ℝ) (cost_y : ℝ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_x = 4 →
  cost_y = 2.8 →
  total_pens = 12 →
  total_cost = 40 →
  ∃ (x y : ℕ), x + y = total_pens ∧ x * cost_x + y * cost_y = total_cost ∧ x = 5 :=
by sorry

end elena_pen_purchase_l685_68597


namespace obtuse_angle_line_range_l685_68510

/-- The slope of a line forming an obtuse angle with the x-axis is negative -/
def obtuse_angle_slope (a : ℝ) : Prop := a^2 + 2*a < 0

/-- The range of a for a line (a^2 + 2a)x - y + 1 = 0 forming an obtuse angle -/
theorem obtuse_angle_line_range (a : ℝ) : 
  obtuse_angle_slope a ↔ -2 < a ∧ a < 0 := by sorry

end obtuse_angle_line_range_l685_68510


namespace unique_n_reaching_three_l685_68562

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 3 else n / 2

def iterateG (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem unique_n_reaching_three :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧ ∃ k : ℕ, iterateG n k = 3 := by
  sorry

end unique_n_reaching_three_l685_68562


namespace orchid_rose_difference_l685_68578

/-- Given the initial and final counts of roses and orchids in a vase,
    prove that there are 10 more orchids than roses after adding new flowers. -/
theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) : 
  initial_roses = 9 →
  initial_orchids = 6 →
  final_roses = 3 →
  final_orchids = 13 →
  final_orchids - final_roses = 10 := by
  sorry

end orchid_rose_difference_l685_68578


namespace kims_weekly_production_l685_68570

/-- Represents Kim's daily sweater production for a week --/
structure WeeklyKnitting where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of sweaters knit in a week --/
def totalSweaters (week : WeeklyKnitting) : ℕ :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Theorem stating that Kim's total sweater production for the given week is 34 --/
theorem kims_weekly_production :
  ∃ (week : WeeklyKnitting),
    week.monday = 8 ∧
    week.tuesday = week.monday + 2 ∧
    week.wednesday = week.tuesday - 4 ∧
    week.thursday = week.tuesday - 4 ∧
    week.friday = week.monday / 2 ∧
    totalSweaters week = 34 := by
  sorry


end kims_weekly_production_l685_68570


namespace sufficient_condition_for_inequality_l685_68512

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (a ≥ 5) → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧ 
  ∃ b : ℝ, b < 5 ∧ (∀ x ∈ Set.Icc 1 2, x^2 - b ≤ 0) :=
by sorry

end sufficient_condition_for_inequality_l685_68512


namespace fourth_drawn_is_92_l685_68558

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (groupNumber : ℕ) : ℕ :=
  firstDrawn + (groupNumber - 1) * (populationSize / sampleSize)

/-- Theorem: The fourth drawn number in the given systematic sampling scenario is 92 -/
theorem fourth_drawn_is_92 :
  systematicSample 600 20 2 4 = 92 := by
  sorry

#eval systematicSample 600 20 2 4

end fourth_drawn_is_92_l685_68558


namespace sin_sum_simplification_l685_68504

theorem sin_sum_simplification :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end sin_sum_simplification_l685_68504


namespace average_age_combined_l685_68565

/-- The average age of a combined group of fifth-graders and parents -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) 
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) :
  num_fifth_graders = 40 →
  num_parents = 50 →
  avg_age_fifth_graders = 10 →
  avg_age_parents = 35 →
  (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / 
  (num_fifth_graders + num_parents : ℚ) = 215 / 9 := by
sorry

end average_age_combined_l685_68565


namespace quadrilateral_formation_count_l685_68572

theorem quadrilateral_formation_count :
  let rod_lengths : Finset ℕ := Finset.range 25
  let chosen_rods : Finset ℕ := {4, 9, 12}
  let remaining_rods := rod_lengths \ chosen_rods
  (remaining_rods.filter (fun d => 
    d + 4 + 9 > 12 ∧ d + 4 + 12 > 9 ∧ d + 9 + 12 > 4 ∧ 4 + 9 + 12 > d
  )).card = 22 := by
  sorry

end quadrilateral_formation_count_l685_68572


namespace neither_sufficient_nor_necessary_l685_68532

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem neither_sufficient_nor_necessary (a b : V) : 
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧ 
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) := by
  sorry

end neither_sufficient_nor_necessary_l685_68532


namespace nicole_collected_400_cards_l685_68592

/-- The number of Pokemon cards Nicole collected -/
def nicole_cards : ℕ := 400

/-- The number of Pokemon cards Cindy collected -/
def cindy_cards : ℕ := 2 * nicole_cards

/-- The number of Pokemon cards Rex collected -/
def rex_cards : ℕ := (nicole_cards + cindy_cards) / 2

/-- The number of people Rex divided his cards among (himself and 3 siblings) -/
def num_people : ℕ := 4

/-- The number of cards Rex has left after dividing -/
def rex_leftover : ℕ := 150

theorem nicole_collected_400_cards :
  nicole_cards = 400 ∧
  cindy_cards = 2 * nicole_cards ∧
  rex_cards = (nicole_cards + cindy_cards) / 2 ∧
  rex_cards = num_people * rex_leftover :=
sorry

end nicole_collected_400_cards_l685_68592


namespace valid_fractions_characterization_l685_68539

def is_valid_fraction (n d : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ d ≥ 10 ∧ d < 100 ∧
  (∃ (a b c : Nat), (a > 0 ∧ b > 0 ∧ c > 0) ∧
    ((n = 10 * a + b ∧ d = 10 * a + c ∧ n * c = d * b) ∨
     (n = 10 * a + b ∧ d = 10 * c + b ∧ n * c = d * a) ∨
     (n = 10 * a + c ∧ d = 10 * b + c ∧ n * b = d * a)))

theorem valid_fractions_characterization :
  {p : Nat × Nat | is_valid_fraction p.1 p.2} =
  {(26, 65), (16, 64), (19, 95), (49, 98)} := by
  sorry

end valid_fractions_characterization_l685_68539


namespace quadratic_equation_roots_l685_68595

theorem quadratic_equation_roots (x : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - (3/2) = 0) ∧ (2 * x₂^2 - 3 * x₂ - (3/2) = 0) :=
by
  sorry

end quadratic_equation_roots_l685_68595


namespace train_crossing_time_l685_68507

def train_length : ℝ := 450
def train_speed_kmh : ℝ := 54

theorem train_crossing_time : 
  ∀ (platform_length : ℝ) (train_speed_ms : ℝ),
    platform_length = train_length →
    train_speed_ms = train_speed_kmh * (1000 / 3600) →
    (2 * train_length) / train_speed_ms = 60 := by
  sorry

end train_crossing_time_l685_68507


namespace new_person_weight_l685_68553

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 :=
by sorry

end new_person_weight_l685_68553


namespace least_integer_greater_than_sqrt_500_l685_68546

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end least_integer_greater_than_sqrt_500_l685_68546


namespace shelter_cat_count_l685_68518

/-- Calculates the total number of cats and kittens in an animal shelter --/
theorem shelter_cat_count (total_adults : ℕ) (female_ratio : ℚ) (litter_ratio : ℚ) (avg_kittens : ℕ) : 
  total_adults = 100 →
  female_ratio = 1/2 →
  litter_ratio = 1/2 →
  avg_kittens = 4 →
  total_adults + (total_adults * female_ratio * litter_ratio * avg_kittens) = 200 := by
sorry

end shelter_cat_count_l685_68518


namespace transformed_sin_equals_cos_l685_68525

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem transformed_sin_equals_cos :
  ∀ x : ℝ, f x = g (2 * (x + π / 4)) :=
by
  sorry

end transformed_sin_equals_cos_l685_68525


namespace roberts_birthday_l685_68530

/-- The number of years until Robert turns 30 -/
def years_until_30 (patrick_age : ℕ) (robert_age : ℕ) : ℕ :=
  30 - robert_age

/-- Robert's current age is twice Patrick's age -/
def robert_age (patrick_age : ℕ) : ℕ :=
  2 * patrick_age

theorem roberts_birthday (patrick_age : ℕ) (h1 : patrick_age = 14) :
  years_until_30 patrick_age (robert_age patrick_age) = 2 := by
  sorry

end roberts_birthday_l685_68530


namespace triangle_base_length_l685_68599

/-- Given a triangle with area 16 m² and height 8 m, prove its base length is 4 m -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 16 → height = 8 → area = (base * height) / 2 → base = 4 := by
sorry

end triangle_base_length_l685_68599


namespace inequality_solution_and_function_property_l685_68585

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} ∧
    ∀ x : ℝ, x ∈ S ↔ f (2*x) + f (x+4) ≥ 6) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → f (a*b) > f (a-b+1)) :=
by sorry

end inequality_solution_and_function_property_l685_68585


namespace cubic_factorization_l685_68564

theorem cubic_factorization (x : ℝ) : 
  x^3 + x^2 - 2*x - 2 = (x + 1) * (x - Real.sqrt 2) * (x + Real.sqrt 2) := by
  sorry

end cubic_factorization_l685_68564


namespace possible_sum_BC_ge_90_l685_68521

/-- Represents an acute triangle with angles A, B, and C --/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  sum_180 : A + B + C = 180
  ordered : A > B ∧ B > C

/-- 
Theorem: In an acute triangle with angles A > B > C, 
it's possible for the sum of B and C to be greater than or equal to 90°
--/
theorem possible_sum_BC_ge_90 (t : AcuteTriangle) : 
  ∃ (x y z : Real), x > y ∧ y > z ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x + y + z = 180 ∧ y + z ≥ 90 := by
  sorry

end possible_sum_BC_ge_90_l685_68521


namespace shekars_math_marks_l685_68551

def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ math_marks : ℕ,
    math_marks = average_marks * total_subjects - (science_marks + social_studies_marks + english_marks + biology_marks) :=
by
  sorry

end shekars_math_marks_l685_68551


namespace overall_profit_percentage_l685_68568

def book_a_cost : ℚ := 50
def book_b_cost : ℚ := 75
def book_c_cost : ℚ := 100
def book_a_sell : ℚ := 60
def book_b_sell : ℚ := 90
def book_c_sell : ℚ := 120

def total_investment_cost : ℚ := book_a_cost + book_b_cost + book_c_cost
def total_revenue : ℚ := book_a_sell + book_b_sell + book_c_sell
def total_profit : ℚ := total_revenue - total_investment_cost
def profit_percentage : ℚ := (total_profit / total_investment_cost) * 100

theorem overall_profit_percentage :
  profit_percentage = 20 := by sorry

end overall_profit_percentage_l685_68568


namespace solve_equation_y_l685_68567

theorem solve_equation_y (y : ℝ) (hy : y ≠ 0) :
  (7 * y)^4 = (14 * y)^3 ↔ y = 8 / 7 := by
sorry

end solve_equation_y_l685_68567


namespace negative_y_ceil_floor_product_l685_68538

theorem negative_y_ceil_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 :=
by sorry

end negative_y_ceil_floor_product_l685_68538


namespace angle_sum_inequality_l685_68542

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real)
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 :=
by sorry

end angle_sum_inequality_l685_68542


namespace negation_equivalence_l685_68573

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (x < 1 ∨ x^2 ≥ 4)) ↔ (∀ x : ℝ, (x ≥ 1 ∧ x^2 < 4)) :=
by sorry

end negation_equivalence_l685_68573


namespace inverse_proportion_points_order_l685_68531

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) :
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 →
  -4 / x₁ = -1 →
  -4 / x₂ = 3 →
  -4 / x₃ = 5 →
  x₂ < x₃ ∧ x₃ < x₁ :=
by sorry

end inverse_proportion_points_order_l685_68531


namespace sum_of_digits_of_square_ones_l685_68547

def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_of_square_ones (n : ℕ) : 
  sum_of_digits ((ones n)^2) = n^2 :=
sorry

end sum_of_digits_of_square_ones_l685_68547


namespace dividend_calculation_l685_68501

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 50)
  (h2 : quotient = 70)
  (h3 : remainder = 20) :
  divisor * quotient + remainder = 3520 := by
sorry

end dividend_calculation_l685_68501


namespace four_digit_number_theorem_l685_68571

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_two_digits (n : ℕ) : ℕ :=
  n / 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def is_permutation (a b : ℕ) : Prop :=
  a / 10 = b % 10 ∧ a % 10 = b / 10

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

theorem four_digit_number_theorem :
  ∃! n : ℕ, 
    is_valid_four_digit_number n ∧
    is_permutation (first_two_digits n) (last_two_digits n) ∧
    first_two_digits n - last_two_digits n = sum_of_digits (first_two_digits n) ∧
    n = 5445 :=
by
  sorry

end four_digit_number_theorem_l685_68571


namespace prime_square_mod_24_l685_68548

theorem prime_square_mod_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 24 = 1 :=
sorry

end prime_square_mod_24_l685_68548


namespace polynomial_characterization_l685_68583

-- Define the polynomial type
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equality condition for the polynomial
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SumProductZero a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def QuarticQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  PolynomialCondition P → QuarticQuadraticForm P :=
by
  sorry

end polynomial_characterization_l685_68583


namespace quadratic_equation_solution_l685_68576

theorem quadratic_equation_solution : ∃ x1 x2 : ℝ, 
  x1 = 95 ∧ 
  x2 = -105 ∧ 
  x1^2 + 10*x1 - 9975 = 0 ∧ 
  x2^2 + 10*x2 - 9975 = 0 := by
sorry

end quadratic_equation_solution_l685_68576


namespace equation_implies_fraction_value_l685_68513

theorem equation_implies_fraction_value (a x y : ℝ) :
  x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (Real.log (x - a) - Real.log (a - y)) →
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
  sorry

end equation_implies_fraction_value_l685_68513


namespace paint_project_total_l685_68566

/-- The total amount of paint needed for a project, given the amount left from a previous project and the amount that needs to be bought. -/
def total_paint (left_over : ℕ) (to_buy : ℕ) : ℕ :=
  left_over + to_buy

/-- Theorem stating that the total amount of paint needed is 333 liters. -/
theorem paint_project_total :
  total_paint 157 176 = 333 := by
  sorry

end paint_project_total_l685_68566


namespace debby_photos_l685_68588

/-- Calculates the number of photographs Debby kept after her vacation -/
theorem debby_photos (N : ℝ) : 
  let zoo_percent : ℝ := 0.60
  let museum_percent : ℝ := 0.25
  let gallery_percent : ℝ := 0.15
  let zoo_keep : ℝ := 0.70
  let museum_keep : ℝ := 0.50
  let gallery_keep : ℝ := 1

  let zoo_photos : ℝ := zoo_percent * N
  let museum_photos : ℝ := museum_percent * N
  let gallery_photos : ℝ := gallery_percent * N

  let kept_zoo : ℝ := zoo_keep * zoo_photos
  let kept_museum : ℝ := museum_keep * museum_photos
  let kept_gallery : ℝ := gallery_keep * gallery_photos

  let total_kept : ℝ := kept_zoo + kept_museum + kept_gallery

  total_kept = 0.695 * N :=
by sorry

end debby_photos_l685_68588


namespace parallel_vectors_k_value_l685_68519

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k_value (k : ℝ) :
  (∀ i : Fin 2, ∃ t : ℝ, a i + k * b i = t * c i) →
  k = 1/2 := by sorry

end parallel_vectors_k_value_l685_68519


namespace shaded_area_ratio_l685_68514

/-- Given two squares ABCD and CEFG with the same side length sharing a common vertex C,
    the ratio of the shaded area to the area of square ABCD is 2 - √2 -/
theorem shaded_area_ratio (l : ℝ) (h : l > 0) : 
  let diagonal := l * Real.sqrt 2
  let small_side := diagonal - l
  let shaded_area := l^2 - 2 * (1/2 * small_side * l)
  shaded_area / l^2 = 2 - Real.sqrt 2 := by
  sorry

end shaded_area_ratio_l685_68514


namespace base9_85_equals_77_l685_68589

-- Define a function to convert a base-9 number to base-10
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

-- Theorem statement
theorem base9_85_equals_77 :
  base9ToBase10 [5, 8] = 77 := by
  sorry

end base9_85_equals_77_l685_68589


namespace inequality_range_l685_68524

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
sorry

end inequality_range_l685_68524


namespace decimal_to_binary_87_l685_68534

theorem decimal_to_binary_87 : 
  (87 : ℕ).digits 2 = [1, 1, 1, 0, 1, 0, 1] :=
by sorry

end decimal_to_binary_87_l685_68534


namespace roots_of_polynomial_l685_68596

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x + 1) * (x - 4) = p x) :=
sorry

end roots_of_polynomial_l685_68596


namespace max_workers_l685_68506

/-- Represents the number of workers on the small field -/
def n : ℕ := sorry

/-- The total number of workers in the crew -/
def total_workers : ℕ := 2 * n + 4

/-- The area of the small field -/
def small_area : ℝ := sorry

/-- The area of the large field -/
def large_area : ℝ := 2 * small_area

/-- The time taken to complete work on the small field -/
def small_field_time : ℝ := sorry

/-- The time taken to complete work on the large field -/
def large_field_time : ℝ := sorry

/-- The condition that the small field is still being worked on when the large field is finished -/
axiom work_condition : small_field_time > large_field_time

/-- The theorem stating the maximum number of workers in the crew -/
theorem max_workers : total_workers ≤ 10 := by sorry

end max_workers_l685_68506


namespace allocation_schemes_eq_36_l685_68559

/-- Represents the number of teachers --/
def num_teachers : ℕ := 4

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the condition that each school must receive at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- Calculates the number of ways to allocate teachers to schools --/
def allocation_schemes (n_teachers : ℕ) (n_schools : ℕ) (min_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 36 --/
theorem allocation_schemes_eq_36 : 
  allocation_schemes num_teachers num_schools min_teachers_per_school = 36 := by
  sorry

end allocation_schemes_eq_36_l685_68559


namespace water_evaporation_proof_l685_68550

-- Define the initial composition of solution y
def solution_y_composition : ℝ := 0.3

-- Define the initial amount of solution y
def initial_amount : ℝ := 6

-- Define the amount of solution y added after evaporation
def amount_added : ℝ := 2

-- Define the amount remaining after evaporation
def amount_remaining : ℝ := 4

-- Define the new composition of the solution
def new_composition : ℝ := 0.4

-- Define the amount of water evaporated
def water_evaporated : ℝ := 2

-- Theorem statement
theorem water_evaporation_proof :
  let initial_liquid_x := solution_y_composition * initial_amount
  let added_liquid_x := solution_y_composition * amount_added
  let total_liquid_x := initial_liquid_x + added_liquid_x
  let new_total_amount := total_liquid_x / new_composition
  new_total_amount = amount_remaining + amount_added →
  water_evaporated = amount_added :=
by sorry

end water_evaporation_proof_l685_68550


namespace garden_area_l685_68552

theorem garden_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 2 * (length + width) →
  length = 3 * width →
  perimeter = 84 →
  length * width = 330.75 := by
  sorry

end garden_area_l685_68552


namespace a_2_value_a_n_formula_l685_68515

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom relation (n : ℕ) (hn : n > 0) : 
  2 * S n / n = sequence_a (n + 1) - (1/3) * n^2 - n - 2/3

theorem a_2_value : sequence_a 2 = 4 := by sorry

theorem a_n_formula (n : ℕ) (hn : n > 0) : sequence_a n = n^2 := by sorry

end a_2_value_a_n_formula_l685_68515


namespace triangle_side_length_l685_68554

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the relationship between side lengths and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) (h1 : t.a = 2) (h2 : t.B = 135 * π / 180)
    (h3 : (1/2) * t.a * t.c * Real.sin t.B = 4) : t.b = 2 * Real.sqrt 13 := by
  sorry

end triangle_side_length_l685_68554


namespace solve_equation_l685_68582

-- Define the * operation
def star (a b : ℚ) : ℚ := 2 * a + 3 * b

-- Theorem statement
theorem solve_equation (x : ℚ) :
  star 5 (star 7 x) = -4 → x = -56/9 := by
  sorry

end solve_equation_l685_68582


namespace adult_attraction_cost_is_four_l685_68500

/-- Represents the cost structure and family composition for a park visit -/
structure ParkVisit where
  entrance_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Calculates the cost of an adult attraction ticket given the park visit details -/
def adult_attraction_cost (visit : ParkVisit) : ℕ :=
  let total_people := visit.num_children + visit.num_parents + visit.num_grandparents
  let entrance_cost := total_people * visit.entrance_fee
  let children_attraction_cost := visit.num_children * visit.child_attraction_fee
  let adult_attraction_total := visit.total_cost - entrance_cost - children_attraction_cost
  let num_adults := visit.num_parents + visit.num_grandparents
  adult_attraction_total / num_adults

theorem adult_attraction_cost_is_four : 
  adult_attraction_cost ⟨5, 2, 4, 2, 1, 55⟩ = 4 := by
  sorry

end adult_attraction_cost_is_four_l685_68500


namespace length_of_EF_l685_68502

/-- A rectangle intersecting a circle -/
structure RectangleIntersectingCircle where
  /-- Length of AB -/
  AB : ℝ
  /-- Length of BC -/
  BC : ℝ
  /-- Length of DE -/
  DE : ℝ
  /-- Length of EF -/
  EF : ℝ

/-- Theorem stating the length of EF in the given configuration -/
theorem length_of_EF (r : RectangleIntersectingCircle) 
  (h1 : r.AB = 4)
  (h2 : r.BC = 5)
  (h3 : r.DE = 3) :
  r.EF = 7 := by
  sorry

#check length_of_EF

end length_of_EF_l685_68502


namespace apple_capacity_l685_68517

def bookbag_capacity : ℕ := 20
def other_fruit_weight : ℕ := 3

theorem apple_capacity : bookbag_capacity - other_fruit_weight = 17 := by
  sorry

end apple_capacity_l685_68517


namespace sine_cosine_sum_equals_one_l685_68593

theorem sine_cosine_sum_equals_one : 
  Real.sin (π / 2 + π / 3) + Real.cos (π / 2 - π / 6) = 1 := by sorry

end sine_cosine_sum_equals_one_l685_68593


namespace simultaneous_work_time_l685_68587

/-- The time taken for two workers to fill a truck when working simultaneously -/
theorem simultaneous_work_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 8) :
  1 / (rate1 + rate2) = 24 / 7 := by sorry

end simultaneous_work_time_l685_68587


namespace average_of_p_and_q_l685_68591

theorem average_of_p_and_q (p q : ℝ) (h : (5 / 4) * (p + q) = 15) : (p + q) / 2 = 6 := by
  sorry

end average_of_p_and_q_l685_68591


namespace binary_expression_equals_expected_result_l685_68549

/-- Converts a list of binary digits to a natural number. -/
def binary_to_nat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 2 * acc + d) 0

/-- Calculates the result of the given binary expression. -/
def binary_expression_result : Nat :=
  let a := binary_to_nat [1, 0, 1, 1, 0]
  let b := binary_to_nat [1, 0, 1, 0]
  let c := binary_to_nat [1, 1, 1, 0, 0]
  let d := binary_to_nat [1, 1, 1, 0]
  a + b - c + d

/-- The expected result in binary. -/
def expected_result : Nat :=
  binary_to_nat [0, 1, 1, 1, 0]

theorem binary_expression_equals_expected_result :
  binary_expression_result = expected_result := by
  sorry

end binary_expression_equals_expected_result_l685_68549


namespace estimate_red_balls_l685_68598

/-- Represents the result of drawing a ball -/
inductive BallColor
| Red
| White

/-- Represents a bag of balls -/
structure BallBag where
  totalBalls : Nat
  redBalls : Nat
  whiteBalls : Nat
  totalBalls_eq : totalBalls = redBalls + whiteBalls

/-- Represents the result of multiple draws -/
structure DrawResult where
  totalDraws : Nat
  redDraws : Nat
  whiteDraws : Nat
  totalDraws_eq : totalDraws = redDraws + whiteDraws

/-- Theorem stating the estimated number of red balls -/
theorem estimate_red_balls 
  (bag : BallBag) 
  (draws : DrawResult) 
  (h1 : bag.totalBalls = 8) 
  (h2 : draws.totalDraws = 100) 
  (h3 : draws.redDraws = 75) :
  (bag.totalBalls : ℚ) * (draws.redDraws : ℚ) / (draws.totalDraws : ℚ) = 6 := by
  sorry

end estimate_red_balls_l685_68598


namespace factory_works_ten_hours_per_day_l685_68560

/-- Represents a chocolate factory with its production parameters -/
structure ChocolateFactory where
  production_rate : ℕ  -- candies per hour
  order_size : ℕ       -- total candies to produce
  days_to_complete : ℕ -- number of days to complete the order

/-- Calculates the number of hours the factory works each day -/
def hours_per_day (factory : ChocolateFactory) : ℚ :=
  (factory.order_size / factory.production_rate : ℚ) / factory.days_to_complete

/-- Theorem stating that for the given parameters, the factory works 10 hours per day -/
theorem factory_works_ten_hours_per_day :
  let factory := ChocolateFactory.mk 50 4000 8
  hours_per_day factory = 10 := by
  sorry

end factory_works_ten_hours_per_day_l685_68560


namespace counterexample_exists_l685_68505

theorem counterexample_exists (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b : ℝ, c * b^2 ≥ a * b^2 := by
  sorry

end counterexample_exists_l685_68505


namespace plane_distance_ratio_l685_68537

theorem plane_distance_ratio (total_distance bus_distance : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : bus_distance = 720)
  : (total_distance - (2/3 * bus_distance + bus_distance)) / total_distance = 1/3 := by
  sorry

end plane_distance_ratio_l685_68537


namespace veg_eaters_count_l685_68535

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  bothVegNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegNonVeg

/-- Theorem: The number of people who eat veg in the family is 21 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 13)
  (h2 : fd.bothVegNonVeg = 8) : 
  totalVegEaters fd = 21 := by
  sorry

end veg_eaters_count_l685_68535


namespace no_intersection_points_l685_68529

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ is 0 -/
theorem no_intersection_points : ∀ θ : ℝ, 
  ¬∃ r : ℝ, (r = 3 * Real.cos θ ∧ r = 6 * Real.sin θ) :=
by sorry

end no_intersection_points_l685_68529


namespace sum_243_81_base3_l685_68545

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ := sorry

/-- Adds two numbers represented in base 3 -/
def addBase3 (a b : List ℕ) : List ℕ := sorry

/-- Checks if a list of digits is a valid base 3 representation -/
def isValidBase3 (l : List ℕ) : Prop := sorry

theorem sum_243_81_base3 :
  let a := toBase3 243
  let b := toBase3 81
  let sum := addBase3 a b
  isValidBase3 a ∧ isValidBase3 b ∧ isValidBase3 sum ∧ sum = [0, 0, 0, 0, 1, 1] := by sorry

end sum_243_81_base3_l685_68545


namespace integral_proof_l685_68574

open Real

noncomputable def f (x : ℝ) : ℝ := 
  -20/27 * ((1 + x^(3/4))^(1/5) / x^(3/20))^9

theorem integral_proof (x : ℝ) (h : x > 0) : 
  deriv f x = (((1 + x^(3/4))^4)^(1/5)) / (x^2 * x^(7/20)) :=
by sorry

end integral_proof_l685_68574


namespace sqrt_four_fourth_powers_sum_l685_68520

theorem sqrt_four_fourth_powers_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end sqrt_four_fourth_powers_sum_l685_68520


namespace book_price_is_two_l685_68586

/-- The price of a book in rubles -/
def book_price : ℝ := 2

/-- The amount paid for the book in rubles -/
def amount_paid : ℝ := 1

/-- The remaining amount to be paid for the book -/
def remaining_amount : ℝ := book_price - amount_paid

theorem book_price_is_two :
  book_price = 2 ∧
  amount_paid = 1 ∧
  remaining_amount = book_price - amount_paid ∧
  remaining_amount = amount_paid + (book_price - (book_price - amount_paid)) :=
by sorry

end book_price_is_two_l685_68586


namespace binomial_expansion_sum_zero_l685_68557

theorem binomial_expansion_sum_zero (n : ℕ) (b : ℕ) (h1 : n ≥ 2) (h2 : b > 0) :
  let a := 3 * b
  (n.choose 1 * (a - 2 * b) ^ (n - 1) + n.choose 2 * (a - 2 * b) ^ (n - 2) = 0) ↔ n = 3 :=
by sorry

end binomial_expansion_sum_zero_l685_68557


namespace sum_of_roots_l685_68526

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 + 6*x^2 + 16*x = -15) 
  (hy : y^3 + 6*y^2 + 16*y = -17) : 
  x + y = -4 := by
sorry

end sum_of_roots_l685_68526


namespace polynomial_constant_term_l685_68508

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The polynomial g(x) = x^4 + px^3 + qx^2 + rx + s -/
def g (poly : Polynomial4) (x : ℤ) : ℤ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- A polynomial has all negative integer roots -/
def has_all_negative_integer_roots (poly : Polynomial4) : Prop :=
  ∃ (a b c d : ℤ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    ∀ (x : ℤ), g poly x = (x + a) * (x + b) * (x + c) * (x + d)

theorem polynomial_constant_term (poly : Polynomial4) :
  has_all_negative_integer_roots poly →
  poly.p + poly.q + poly.r + poly.s = 8091 →
  poly.s = 8064 := by
  sorry

end polynomial_constant_term_l685_68508


namespace complex_equation_solution_l685_68575

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + Complex.I → z = 3/4 + Complex.I := by
  sorry

end complex_equation_solution_l685_68575


namespace triangle_area_qin_jiushao_l685_68541

theorem triangle_area_qin_jiushao (a b c : ℝ) (h₁ : a = Real.sqrt 2) (h₂ : b = Real.sqrt 3) (h₃ : c = 2) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = Real.sqrt 23 / 4 := by
sorry

end triangle_area_qin_jiushao_l685_68541


namespace similar_triangles_ab_length_l685_68594

/-- Two triangles are similar -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_ab_length :
  ∀ (P Q R X Y Z A B C : ℝ × ℝ),
  let pqr : Set (Fin 3 → ℝ × ℝ) := {![P, Q, R]}
  let xyz : Set (Fin 3 → ℝ × ℝ) := {![X, Y, Z]}
  let abc : Set (Fin 3 → ℝ × ℝ) := {![A, B, C]}
  similar_triangles pqr xyz →
  similar_triangles xyz abc →
  dist P Q = 8 →
  dist Q R = 16 →
  dist B C = 24 →
  dist Y Z = 12 →
  dist A B = 12 :=
by sorry

end similar_triangles_ab_length_l685_68594


namespace min_sum_dimensions_l685_68579

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ (a b c : ℕ+), a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 42 :=
sorry

end min_sum_dimensions_l685_68579


namespace price_increase_percentage_l685_68540

theorem price_increase_percentage (initial_price : ℝ) : 
  initial_price > 0 →
  let new_egg_price := initial_price * 1.1
  let new_apple_price := initial_price * 1.02
  let initial_total := initial_price * 2
  let new_total := new_egg_price + new_apple_price
  (new_total - initial_total) / initial_total = 0.04 :=
by
  sorry

#check price_increase_percentage

end price_increase_percentage_l685_68540


namespace vacation_cost_from_dog_walking_vacation_cost_proof_l685_68523

/-- Calculates the total cost of a vacation based on dog walking earnings --/
theorem vacation_cost_from_dog_walking 
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : ℚ
  :=
  let total_earnings := start_charge * num_dogs + per_block_charge * total_blocks
  total_earnings

theorem vacation_cost_proof
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : vacation_cost_from_dog_walking start_charge per_block_charge num_dogs total_blocks family_members h1 h2 h3 h4 h5 = 200 := by
  sorry

end vacation_cost_from_dog_walking_vacation_cost_proof_l685_68523


namespace mean_goals_is_6_l685_68543

/-- The number of players who scored 5 goals -/
def players_5 : ℕ := 4

/-- The number of players who scored 6 goals -/
def players_6 : ℕ := 3

/-- The number of players who scored 7 goals -/
def players_7 : ℕ := 2

/-- The number of players who scored 8 goals -/
def players_8 : ℕ := 1

/-- The total number of goals scored -/
def total_goals : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 8 * players_8

/-- The total number of players -/
def total_players : ℕ := players_5 + players_6 + players_7 + players_8

/-- The mean number of goals scored -/
def mean_goals : ℚ := total_goals / total_players

theorem mean_goals_is_6 : mean_goals = 6 := by sorry

end mean_goals_is_6_l685_68543


namespace rectangle_diagonal_l685_68511

/-- The length of the diagonal of a rectangle with length 40 and width 40√2 is 40√3 -/
theorem rectangle_diagonal : 
  ∀ (l w d : ℝ), 
  l = 40 → 
  w = 40 * Real.sqrt 2 → 
  d = Real.sqrt (l^2 + w^2) → 
  d = 40 * Real.sqrt 3 := by
sorry

end rectangle_diagonal_l685_68511


namespace revenue_change_l685_68581

theorem revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.19) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.0685 := by
sorry

end revenue_change_l685_68581


namespace decimal_to_fraction_sum_l685_68580

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.3421 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.3421 → a ≤ c ∧ b ≤ d →
  a + b = 13421 :=
by sorry

end decimal_to_fraction_sum_l685_68580


namespace fraction_equality_l685_68569

theorem fraction_equality : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5000 = 750.0000000000001 := by
  sorry

end fraction_equality_l685_68569


namespace range_of_a_l685_68556

/-- The condition for two distinct real roots -/
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0

/-- The condition for a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  ¬(has_two_distinct_real_roots a ∨ is_hyperbola a) → a ≥ 3 := by
  sorry

end range_of_a_l685_68556


namespace three_digit_number_theorem_l685_68509

theorem three_digit_number_theorem (x y z : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ x ≠ 0 →
  let n := 100 * x + 10 * y + z
  let sum_digits := x + y + z
  n / sum_digits = 13 ∧ n % sum_digits = 15 →
  n = 106 ∨ n = 145 ∨ n = 184 := by
sorry

end three_digit_number_theorem_l685_68509


namespace min_value_theorem_min_value_equality_l685_68516

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2/5)^(1/5) :=
by sorry

theorem min_value_equality :
  let x := (2/5)^(1/5)
  3 * x + 5 + 2 / x^5 = 10 + 3 * (2/5)^(1/5) :=
by sorry

end min_value_theorem_min_value_equality_l685_68516


namespace paving_cost_l685_68536

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end paving_cost_l685_68536


namespace equidistant_point_x_coordinate_l685_68561

theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ (y : ℝ), (y^2 + 6*y + 9 = y^2 + 25) → y = x) ∧
    x = 8/3 := by
  sorry

end equidistant_point_x_coordinate_l685_68561


namespace prob_live_to_25_given_20_l685_68528

/-- The probability of an animal living to 25 years given it has lived to 20 years -/
theorem prob_live_to_25_given_20 (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) 
  (h3 : 0 ≤ p_20 ∧ p_20 ≤ 1) 
  (h4 : 0 ≤ p_25 ∧ p_25 ≤ 1) 
  (h5 : p_25 ≤ p_20) : 
  p_25 / p_20 = 0.5 := by sorry

end prob_live_to_25_given_20_l685_68528


namespace divisible_by_four_sum_consecutive_odds_l685_68555

theorem divisible_by_four_sum_consecutive_odds (a : ℤ) : ∃ (x y : ℤ), 
  4 * a = x + y ∧ Odd x ∧ Odd y ∧ y = x + 2 := by
  sorry

end divisible_by_four_sum_consecutive_odds_l685_68555


namespace triangle_sine_relations_l685_68577

theorem triangle_sine_relations (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = 7 * a * Real.sin B →
  Real.sin A = 1/7 ∧ 
  (B = π/3 → Real.sin C = 13/14) := by
  sorry

end triangle_sine_relations_l685_68577


namespace geometric_progression_seventh_term_l685_68590

theorem geometric_progression_seventh_term 
  (b₁ q : ℚ) 
  (sum_first_three : b₁ + b₁*q + b₁*q^2 = 91)
  (arithmetic_progression : 2*(b₁*q + 27) = (b₁ + 25) + (b₁*q^2 + 1)) :
  b₁*q^6 = (35 * 46656) / 117649 ∨ b₁*q^6 = (63 * 4096) / 117649 := by
sorry

end geometric_progression_seventh_term_l685_68590


namespace max_value_of_sum_products_l685_68527

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 120 → 
  a * b + b * c + c * d ≤ 3600 :=
by sorry

end max_value_of_sum_products_l685_68527


namespace desk_chair_relationship_l685_68503

def chair_heights : List ℝ := [37.0, 40.0, 42.0, 45.0]
def desk_heights : List ℝ := [70.0, 74.8, 78.0, 82.8]

def linear_function (x : ℝ) : ℝ := 1.6 * x + 10.8

theorem desk_chair_relationship :
  ∀ (i : Fin 4),
    linear_function (chair_heights.get i) = desk_heights.get i :=
by sorry

end desk_chair_relationship_l685_68503
