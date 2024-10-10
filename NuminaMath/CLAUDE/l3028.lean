import Mathlib

namespace square_area_ratio_l3028_302897

theorem square_area_ratio (d : ℝ) (h : d > 0) :
  let small_square_side := d / Real.sqrt 2
  let big_square_side := d
  let small_square_area := small_square_side ^ 2
  let big_square_area := big_square_side ^ 2
  big_square_area / small_square_area = 2 := by
sorry

end square_area_ratio_l3028_302897


namespace town_population_is_300_l3028_302889

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

/-- The total number of people attending the meeting -/
def total_attending : ℕ := females_attending + males_attending

/-- The total population of the town -/
def town_population : ℕ := 2 * total_attending

theorem town_population_is_300 : town_population = 300 := by
  sorry

end town_population_is_300_l3028_302889


namespace meat_division_l3028_302823

theorem meat_division (pot1_weight pot2_weight total_meat : ℕ) 
  (h1 : pot1_weight = 645)
  (h2 : pot2_weight = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1_weight + meat1 = pot2_weight + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

#check meat_division

end meat_division_l3028_302823


namespace pretzel_problem_l3028_302841

theorem pretzel_problem (john_pretzels alan_pretzels marcus_pretzels initial_pretzels : ℕ) :
  john_pretzels = 28 →
  alan_pretzels = john_pretzels - 9 →
  marcus_pretzels = john_pretzels + 12 →
  marcus_pretzels = 40 →
  initial_pretzels = john_pretzels + alan_pretzels + marcus_pretzels →
  initial_pretzels = 87 := by
  sorry

end pretzel_problem_l3028_302841


namespace double_average_l3028_302882

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let new_avg := 2 * original_avg
  new_avg = 160 := by sorry

end double_average_l3028_302882


namespace sequence_contains_30_l3028_302883

theorem sequence_contains_30 : ∃ n : ℕ+, n * (n + 1) = 30 := by
  sorry

end sequence_contains_30_l3028_302883


namespace expected_distinct_values_formula_l3028_302866

/-- The number of elements in our set -/
def n : ℕ := 2013

/-- The probability of choosing any specific value -/
def p : ℚ := 1 / n

/-- The probability of not choosing a specific value -/
def q : ℚ := 1 - p

/-- The expected number of distinct values in a set of n elements,
    each chosen independently and randomly from {1, ..., n} -/
def expected_distinct_values : ℚ := n * (1 - q^n)

/-- Theorem stating that the expected number of distinct values
    is equal to the formula derived in the solution -/
theorem expected_distinct_values_formula :
  expected_distinct_values = n * (1 - (n - 1 : ℚ)^n / n^n) :=
sorry

end expected_distinct_values_formula_l3028_302866


namespace oranges_per_box_l3028_302881

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) : 
  total_oranges = 24 → num_boxes = 3 → oranges_per_box * num_boxes = total_oranges → oranges_per_box = 8 := by
  sorry

end oranges_per_box_l3028_302881


namespace weight_loss_challenge_l3028_302892

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0) 
  (h2 : initial_weight > 0) : 
  (0.85 * initial_weight + clothes_weight_percentage * 0.85 * initial_weight) / initial_weight = 0.867 → 
  clothes_weight_percentage = 0.02 := by
sorry

end weight_loss_challenge_l3028_302892


namespace smallest_positive_angle_l3028_302853

/-- Given a point P on the unit circle with coordinates (sin(2π/3), cos(2π/3)) 
    on the terminal side of angle α, prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (P : ℝ × ℝ) (α : ℝ) : 
  P.1 = Real.sin (2 * Real.pi / 3) →
  P.2 = Real.cos (2 * Real.pi / 3) →
  P ∈ {Q : ℝ × ℝ | ∃ t : ℝ, Q.1 = Real.cos t ∧ Q.2 = Real.sin t ∧ t ≥ 0 ∧ t < 2 * Real.pi} →
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : ℝ, (∃ k : ℤ, β = α + 2 * Real.pi * k) → β ≥ 11 * Real.pi / 6) :=
by sorry

end smallest_positive_angle_l3028_302853


namespace height_after_growth_spurt_height_approximation_l3028_302815

/-- Calculates the height of a person after a year of growth with specific conditions. -/
theorem height_after_growth_spurt (initial_height : ℝ) 
  (initial_growth_rate : ℝ) (initial_growth_months : ℕ) 
  (growth_increase_rate : ℝ) (total_months : ℕ) : ℝ :=
  let inches_to_meters := 0.0254
  let height_after_initial_growth := initial_height + initial_growth_rate * initial_growth_months
  let remaining_months := total_months - initial_growth_months
  let first_variable_growth := initial_growth_rate * (1 + growth_increase_rate)
  let variable_growth_sum := first_variable_growth * 
    (1 - (1 + growth_increase_rate) ^ remaining_months) / growth_increase_rate
  (height_after_initial_growth + variable_growth_sum) * inches_to_meters

/-- The height after growth spurt is approximately 2.59 meters. -/
theorem height_approximation : 
  ∃ ε > 0, |height_after_growth_spurt 66 2 3 0.1 12 - 2.59| < ε :=
sorry

end height_after_growth_spurt_height_approximation_l3028_302815


namespace power_calculation_l3028_302822

theorem power_calculation : 4^2011 * (-0.25)^2010 - 1 = 3 := by sorry

end power_calculation_l3028_302822


namespace remainder_theorem_l3028_302821

theorem remainder_theorem (n : ℤ) (h : n % 11 = 5) : (4 * n - 9) % 11 = 0 := by
  sorry

end remainder_theorem_l3028_302821


namespace correct_aprons_tomorrow_l3028_302827

def aprons_to_sew_tomorrow (total : ℕ) (already_sewn : ℕ) (today_multiplier : ℕ) : ℕ :=
  let today_sewn := already_sewn * today_multiplier
  let total_sewn := already_sewn + today_sewn
  let remaining := total - total_sewn
  remaining / 2

theorem correct_aprons_tomorrow :
  aprons_to_sew_tomorrow 150 13 3 = 49 := by
  sorry

end correct_aprons_tomorrow_l3028_302827


namespace quadratic_inequality_solution_l3028_302877

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 36*x + 320 ≤ 16} = Set.Icc 16 19 := by sorry

end quadratic_inequality_solution_l3028_302877


namespace cone_volume_l3028_302884

/-- The volume of a cone with base radius 1 and slant height 2√7 is √3π. -/
theorem cone_volume (r h s : ℝ) : 
  r = 1 → s = 2 * Real.sqrt 7 → h^2 + r^2 = s^2 → (1/3) * π * r^2 * h = Real.sqrt 3 * π :=
by sorry

end cone_volume_l3028_302884


namespace flagpole_shadow_length_l3028_302894

theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 28)
  (h3 : building_shadow = 70)
  : ∃ (flagpole_shadow : ℝ), 
    flagpole_height / flagpole_shadow = building_height / building_shadow ∧ 
    flagpole_shadow = 45 := by
  sorry

end flagpole_shadow_length_l3028_302894


namespace unique_prime_in_special_form_l3028_302865

def special_form (n : ℕ) : ℚ :=
  (1 / 11) * ((10^(2*n) - 1) / 9)

theorem unique_prime_in_special_form :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, (special_form n : ℚ) = p ∧ p = 101 :=
by
  sorry

end unique_prime_in_special_form_l3028_302865


namespace greatest_power_of_eleven_l3028_302887

theorem greatest_power_of_eleven (n : ℕ+) : 
  (Finset.card (Nat.divisors n) = 72) →
  (Finset.card (Nat.divisors (11 * n)) = 96) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) ∧ k = 2 :=
by sorry

end greatest_power_of_eleven_l3028_302887


namespace arithmetic_sequence_property_l3028_302825

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_condition a → 2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l3028_302825


namespace single_elimination_matches_l3028_302875

/-- The number of matches required in a single-elimination tournament -/
def matches_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n participants,
    the number of matches required to determine the winner is n - 1 -/
theorem single_elimination_matches (n : ℕ) (h : n > 0) : 
  matches_required n = n - 1 := by
  sorry

end single_elimination_matches_l3028_302875


namespace construction_delay_l3028_302803

/-- Represents the construction project -/
structure ConstructionProject where
  total_days : ℕ
  initial_workers : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculates the delay in days if additional workers were not added -/
def calculate_delay (project : ConstructionProject) : ℕ :=
  let total_work := project.total_days * project.initial_workers
  let work_done_before_addition := project.days_before_addition * project.initial_workers
  let remaining_work := total_work - work_done_before_addition
  let days_with_additional_workers := project.total_days - project.days_before_addition
  let work_done_after_addition := days_with_additional_workers * (project.initial_workers + project.additional_workers)
  (remaining_work + project.initial_workers - 1) / project.initial_workers - days_with_additional_workers

theorem construction_delay (project : ConstructionProject) 
  (h1 : project.total_days = 100)
  (h2 : project.initial_workers = 100)
  (h3 : project.additional_workers = 100)
  (h4 : project.days_before_addition = 20) :
  calculate_delay project = 80 := by
  sorry

#eval calculate_delay { total_days := 100, initial_workers := 100, additional_workers := 100, days_before_addition := 20 }

end construction_delay_l3028_302803


namespace solution_set_implies_m_range_l3028_302899

open Real

theorem solution_set_implies_m_range (m : ℝ) :
  (∀ x : ℝ, |x - 3| - 2 - (-|x + 1| + 4) ≥ m + 1) →
  m ≤ -3 :=
by
  sorry

end solution_set_implies_m_range_l3028_302899


namespace middle_number_proof_l3028_302852

theorem middle_number_proof (A B C : ℝ) (hC : C = 56) (hDiff : C - A = 32) (hRatio : B / C = 5 / 7) : B = 40 := by
  sorry

end middle_number_proof_l3028_302852


namespace modular_congruence_solution_l3028_302857

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end modular_congruence_solution_l3028_302857


namespace f_extrema_l3028_302879

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem f_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  (∀ x ∈ Set.Icc a b, f x ≤ f (Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc a b, f (Real.pi / 2) ≤ f x) := by
  sorry

#check f_extrema

end f_extrema_l3028_302879


namespace cube_volume_from_surface_area_l3028_302805

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end cube_volume_from_surface_area_l3028_302805


namespace secret_spread_days_l3028_302818

/-- The number of people who know the secret after n days -/
def people_knowing_secret (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The proposition that it takes 7 days for at least 2186 people to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, n = 7 ∧ 
  people_knowing_secret (n - 1) < 2186 ∧ people_knowing_secret n ≥ 2186 :=
sorry

end secret_spread_days_l3028_302818


namespace sum_of_coefficients_zero_l3028_302855

theorem sum_of_coefficients_zero (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
sorry

end sum_of_coefficients_zero_l3028_302855


namespace number_of_cars_in_race_l3028_302802

/-- The number of cars in a race where:
  1. Each car starts with 3 people.
  2. After the halfway point, each car has 4 people.
  3. At the end of the race, there are 80 people in total. -/
theorem number_of_cars_in_race : ℕ :=
  let initial_people_per_car : ℕ := 3
  let final_people_per_car : ℕ := 4
  let total_people_at_end : ℕ := 80
  20

#check number_of_cars_in_race

end number_of_cars_in_race_l3028_302802


namespace complement_of_P_is_singleton_two_l3028_302880

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x ∈ U | x^2 < 2}

theorem complement_of_P_is_singleton_two :
  (U \ P) = {2} := by sorry

end complement_of_P_is_singleton_two_l3028_302880


namespace complement_of_A_union_B_l3028_302846

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} := by sorry

end complement_of_A_union_B_l3028_302846


namespace pure_imaginary_fraction_l3028_302837

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a + Complex.I) / (1 + 2 * Complex.I)) → a = -2 := by
  sorry

end pure_imaginary_fraction_l3028_302837


namespace geometric_sequence_sum_l3028_302873

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n) 
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q) 
  (h_sum_1_2 : a 1 + a 2 = 3/4)
  (h_sum_3_to_6 : a 3 + a 4 + a 5 + a 6 = 15) :
  a 7 + a 8 + a 9 = 112 := by
  sorry

end geometric_sequence_sum_l3028_302873


namespace value_of_m_area_of_triangle_max_y_intercept_l3028_302860

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 - x - m^2 + 6*m - 7

-- Theorem 1: If the graph passes through A(-1, 2), then m = 5
theorem value_of_m (m : ℝ) : f m (-1) = 2 → m = 5 := by sorry

-- Theorem 2: If m = 5, the area of triangle ABC is 5/3
theorem area_of_triangle : 
  let m := 5
  let x1 := (- 2/3 : ℝ)  -- x-coordinate of point C
  let x2 := (1 : ℝ)      -- x-coordinate of point B
  (1/2 : ℝ) * |x2 - x1| * 2 = 5/3 := by sorry

-- Theorem 3: The maximum y-coordinate of the y-intercept is 2
theorem max_y_intercept : 
  ∃ (m : ℝ), ∀ (m' : ℝ), f m' 0 ≤ f m 0 ∧ f m 0 = 2 := by sorry

end value_of_m_area_of_triangle_max_y_intercept_l3028_302860


namespace farmer_plant_beds_l3028_302893

theorem farmer_plant_beds (bean_seedlings : ℕ) (bean_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkin_per_row : ℕ) 
  (radishes : ℕ) (radish_per_row : ℕ) 
  (rows_per_bed : ℕ) : 
  bean_seedlings = 64 → 
  bean_per_row = 8 → 
  pumpkin_seeds = 84 → 
  pumpkin_per_row = 7 → 
  radishes = 48 → 
  radish_per_row = 6 → 
  rows_per_bed = 2 → 
  (bean_seedlings / bean_per_row + 
   pumpkin_seeds / pumpkin_per_row + 
   radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

#check farmer_plant_beds

end farmer_plant_beds_l3028_302893


namespace triangle_one_two_two_l3028_302831

/-- Triangle inequality theorem for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_one_two_two :
  can_form_triangle 1 2 2 :=
sorry

end triangle_one_two_two_l3028_302831


namespace expression_equality_l3028_302867

theorem expression_equality (a b c d : ℕ) : 
  a = 12 → b = 13 → c = 16 → d = 11 → 3 * a^2 - 3 * b + 2 * c * d^2 = 4265 := by
  sorry

end expression_equality_l3028_302867


namespace circle_circumference_l3028_302890

theorem circle_circumference (r : ℝ) (h : r > 0) : 
  (2 * r^2 = π * r^2) → (2 * π * r = 4 * r) :=
by sorry

end circle_circumference_l3028_302890


namespace ace_king_probability_l3028_302804

/-- The probability of drawing an Ace first and a King second from a modified deck -/
theorem ace_king_probability (total_cards : ℕ) (num_aces : ℕ) (num_kings : ℕ) 
  (h1 : total_cards = 54) 
  (h2 : num_aces = 5) 
  (h3 : num_kings = 4) : 
  (num_aces : ℚ) / total_cards * num_kings / (total_cards - 1) = 10 / 1426 := by
  sorry

end ace_king_probability_l3028_302804


namespace sand_lost_during_journey_l3028_302832

theorem sand_lost_during_journey (initial_sand final_sand : ℝ) 
  (h1 : initial_sand = 4.1)
  (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
sorry

end sand_lost_during_journey_l3028_302832


namespace football_cards_per_box_l3028_302830

theorem football_cards_per_box (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ) (total_cards : ℕ) :
  basketball_boxes = 9 →
  basketball_cards_per_box = 15 →
  total_cards = 255 →
  let football_boxes := basketball_boxes - 3
  let basketball_cards := basketball_boxes * basketball_cards_per_box
  let football_cards := total_cards - basketball_cards
  football_cards / football_boxes = 20 := by
sorry

end football_cards_per_box_l3028_302830


namespace units_digit_17_25_l3028_302876

theorem units_digit_17_25 : 17^25 % 10 = 7 := by
  sorry

end units_digit_17_25_l3028_302876


namespace bowling_ball_weight_l3028_302868

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (5 * bowling_ball_weight = 3 * canoe_weight) →
    (3 * canoe_weight = 105) →
    bowling_ball_weight = 21 :=
by
  sorry

end bowling_ball_weight_l3028_302868


namespace quadratic_equation_magnitude_unique_l3028_302806

theorem quadratic_equation_magnitude_unique :
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
sorry

end quadratic_equation_magnitude_unique_l3028_302806


namespace system_solutions_correct_l3028_302856

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℚ, x - y = 2 ∧ 2*x + y = 7 ∧ x = 3 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, x - 2*y = 3 ∧ (1/2)*x + (3/4)*y = 13/4 ∧ x = 5 ∧ y = 1) :=
by sorry

end system_solutions_correct_l3028_302856


namespace closest_integer_to_cube_root_216_l3028_302809

theorem closest_integer_to_cube_root_216 : 
  ∀ n : ℤ, |n - (216 : ℝ)^(1/3)| ≥ |6 - (216 : ℝ)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_216_l3028_302809


namespace constant_rate_walking_l3028_302810

/-- Given a constant walking rate where 600 metres are covered in 4 minutes,
    prove that the distance covered in 6 minutes is 900 metres. -/
theorem constant_rate_walking (rate : ℝ) (h1 : rate > 0) (h2 : rate * 4 = 600) :
  rate * 6 = 900 := by
  sorry

end constant_rate_walking_l3028_302810


namespace quadratic_roots_property_l3028_302862

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
sorry

end quadratic_roots_property_l3028_302862


namespace four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l3028_302824

/-- A function that checks if it's possible to form a triangle with given number of lolly sticks -/
def can_form_triangle (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a + b + c = n ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that it's impossible to form a triangle with 4 lolly sticks -/
theorem four_sticks_impossible : ¬ can_form_triangle 4 :=
sorry

/-- Theorem stating that it's possible to form triangles with 3, 5, 6, and 7 lolly sticks -/
theorem other_sticks_possible :
  can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7 :=
sorry

/-- Main theorem combining the above results -/
theorem lolly_stick_triangle :
  ¬ can_form_triangle 4 ∧
  (can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7) :=
sorry

end four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l3028_302824


namespace expression_value_l3028_302861

theorem expression_value (m n : ℝ) (h : m + 2*n = 1) : 3*m^2 + 6*m*n + 6*n = 3 := by
  sorry

end expression_value_l3028_302861


namespace square_area_ratio_l3028_302851

theorem square_area_ratio (side_c side_d : ℝ) (hc : side_c = 24) (hd : side_d = 54) :
  (side_c^2) / (side_d^2) = 16 / 81 := by
  sorry

end square_area_ratio_l3028_302851


namespace inscribed_cylinder_properties_l3028_302835

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  /-- The cylinder's diameter equals its height -/
  height_eq_diameter : cylinder_height = 2 * cylinder_radius
  /-- The axes of the cylinder and cone coincide -/
  axes_coincide : True

/-- The space left in the cone above the cylinder -/
def space_above_cylinder (c : InscribedCylinder) : ℝ :=
  c.cone_altitude - c.cylinder_height

theorem inscribed_cylinder_properties (c : InscribedCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 20) : 
  c.cylinder_radius = 40 / 9 ∧ space_above_cylinder c = 100 / 9 := by
  sorry


end inscribed_cylinder_properties_l3028_302835


namespace four_by_seven_same_color_corners_l3028_302848

/-- A coloring of a rectangular board. -/
def Coloring (m n : ℕ) := Fin m → Fin n → Bool

/-- A rectangle on the board, represented by its corners. -/
structure Rectangle (m n : ℕ) where
  top_left : Fin m × Fin n
  bottom_right : Fin m × Fin n
  h_valid : top_left.1 < bottom_right.1
  w_valid : top_left.2 < bottom_right.2

/-- Check if all corners of a rectangle have the same color. -/
def sameColorCorners (c : Coloring m n) (r : Rectangle m n) : Prop :=
  let (i₁, j₁) := r.top_left
  let (i₂, j₂) := r.bottom_right
  c i₁ j₁ = c i₁ j₂ ∧ c i₁ j₁ = c i₂ j₁ ∧ c i₁ j₁ = c i₂ j₂

/-- The main theorem: any 4x7 coloring has a rectangle with same-colored corners. -/
theorem four_by_seven_same_color_corners :
  ∀ (c : Coloring 4 7), ∃ (r : Rectangle 4 7), sameColorCorners c r :=
sorry


end four_by_seven_same_color_corners_l3028_302848


namespace lemon_profit_problem_l3028_302864

theorem lemon_profit_problem (buy_lemons : ℕ) (buy_cost : ℚ) (sell_lemons : ℕ) (sell_price : ℚ) (target_profit : ℚ) : 
  buy_lemons = 4 →
  buy_cost = 15 →
  sell_lemons = 6 →
  sell_price = 25 →
  target_profit = 120 →
  ∃ (n : ℕ), n = 286 ∧ 
    n * (sell_price / sell_lemons - buy_cost / buy_lemons) ≥ target_profit ∧
    (n - 1) * (sell_price / sell_lemons - buy_cost / buy_lemons) < target_profit :=
by sorry

end lemon_profit_problem_l3028_302864


namespace z_in_second_quadrant_l3028_302839

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i³) = i
def equation (z : ℂ) : Prop := z * (1 + i^3) = i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end z_in_second_quadrant_l3028_302839


namespace complex_expression_evaluation_l3028_302820

theorem complex_expression_evaluation :
  let a : ℝ := 3.67
  let b : ℝ := 4.83
  let c : ℝ := 2.57
  let d : ℝ := -0.12
  let x : ℝ := 7.25
  let y : ℝ := -0.55
  
  let expression : ℝ := (3*a * (4*b - 2*y)^2) / (5*c * d^3 * 0.5*x) - (2*x * y^3) / (a * b^2 * c)
  
  ∃ ε > 0, |expression - (-57.179729)| < ε ∧ ε < 0.000001 :=
by
  sorry

end complex_expression_evaluation_l3028_302820


namespace max_distance_P_to_D_l3028_302871

/-- A square with side length 1 in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 1)

/-- A point P in the same plane as the square -/
def P : ℝ × ℝ := sorry

/-- Distance between two points in 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum distance between P and D -/
theorem max_distance_P_to_D (square : Square) 
  (h1 : distance P square.A = u)
  (h2 : distance P square.B = v)
  (h3 : distance P square.C = w)
  (h4 : u^2 + w^2 = v^2) : 
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), 
      distance P' square.A = u → 
      distance P' square.B = v → 
      distance P' square.C = w → 
      u^2 + w^2 = v^2 → 
      distance P' square.D ≤ max_dist :=
sorry

end max_distance_P_to_D_l3028_302871


namespace nancy_metal_bead_sets_l3028_302898

/-- The number of metal bead sets Nancy bought -/
def metal_bead_sets : ℕ := 2

/-- The cost of one set of crystal beads in dollars -/
def crystal_bead_cost : ℕ := 9

/-- The cost of one set of metal beads in dollars -/
def metal_bead_cost : ℕ := 10

/-- The total amount Nancy spent in dollars -/
def total_spent : ℕ := 29

/-- Proof that Nancy bought 2 sets of metal beads -/
theorem nancy_metal_bead_sets :
  crystal_bead_cost + metal_bead_cost * metal_bead_sets = total_spent :=
by sorry

end nancy_metal_bead_sets_l3028_302898


namespace inverse_mod_89_l3028_302812

theorem inverse_mod_89 (h : (5⁻¹ : ZMod 89) = 39) : (25⁻¹ : ZMod 89) = 8 := by
  sorry

end inverse_mod_89_l3028_302812


namespace total_books_l3028_302845

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
  sorry

end total_books_l3028_302845


namespace lucky_years_2023_to_2027_l3028_302829

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2023_to_2027 : 
  ¬(isLuckyYear 2023) ∧
  (isLuckyYear 2024) ∧
  (isLuckyYear 2025) ∧
  (isLuckyYear 2026) ∧
  (isLuckyYear 2027) := by
  sorry

end lucky_years_2023_to_2027_l3028_302829


namespace initial_water_amount_l3028_302819

/-- 
Given a bucket with an initial amount of water, prove that this amount is 3 gallons
when adding 6.8 gallons results in a total of 9.8 gallons.
-/
theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end initial_water_amount_l3028_302819


namespace magical_red_knights_fraction_l3028_302811

theorem magical_red_knights_fraction 
  (total_knights : ℕ) 
  (total_knights_pos : total_knights > 0)
  (red_knights : ℕ) 
  (blue_knights : ℕ) 
  (magical_knights : ℕ) 
  (red_knights_fraction : red_knights = (3 * total_knights) / 8)
  (blue_knights_fraction : blue_knights = total_knights - red_knights)
  (magical_knights_fraction : magical_knights = total_knights / 4)
  (magical_ratio : ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    red_knights * p * 3 = blue_knights * p * q ∧ 
    red_knights * p + blue_knights * p = magical_knights * q) :
  ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    7 * p = 3 * q ∧ 
    red_knights * p = magical_knights * q := by
  sorry

end magical_red_knights_fraction_l3028_302811


namespace coTerminalAngle_equiv_neg525_l3028_302836

/-- The angle that shares the same terminal side as -525° -/
def coTerminalAngle (k : ℤ) : ℝ := 195 + k * 360

/-- Proves that coTerminalAngle shares the same terminal side as -525° -/
theorem coTerminalAngle_equiv_neg525 :
  ∀ k : ℤ, ∃ n : ℤ, coTerminalAngle k = -525 + n * 360 := by sorry

end coTerminalAngle_equiv_neg525_l3028_302836


namespace cos_pi_sixth_minus_alpha_l3028_302885

theorem cos_pi_sixth_minus_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = -Real.sqrt 3 / 3) : 
  Real.cos (π / 6 - α) = -1 / 3 := by
  sorry

end cos_pi_sixth_minus_alpha_l3028_302885


namespace ellipse_ratio_l3028_302816

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-latus rectum c,
    if a² + b² - 3c² = 0, then (a + c) / (a - c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry

end ellipse_ratio_l3028_302816


namespace golf_club_average_rounds_l3028_302850

/-- Represents the data for golfers and rounds played -/
structure GolfData where
  rounds : List Nat
  golfers : List Nat

/-- Calculates the average rounds played and rounds to the nearest whole number -/
def averageRoundsRounded (data : GolfData) : Nat :=
  let totalRounds := (List.zip data.rounds data.golfers).map (fun (r, g) => r * g) |>.sum
  let totalGolfers := data.golfers.sum
  Int.toNat ((totalRounds * 2 + totalGolfers) / (2 * totalGolfers))

/-- Theorem stating that for the given golf data, the rounded average is 3 -/
theorem golf_club_average_rounds : 
  averageRoundsRounded { rounds := [1, 2, 3, 4, 5], golfers := [6, 3, 2, 4, 4] } = 3 := by
  sorry

end golf_club_average_rounds_l3028_302850


namespace limit_ratio_recursive_sequences_l3028_302843

/-- Two sequences satisfying given recursive relations -/
def RecursiveSequences (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ b 1 = 7 ∧
  ∀ n, a (n + 1) = b n - 2 * a n ∧ b (n + 1) = 3 * b n - 4 * a n

/-- The limit of the ratio of two sequences -/
def LimitRatio (a b : ℕ → ℝ) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n / b n - l| < ε

/-- The main theorem stating the limit of the ratio of the sequences -/
theorem limit_ratio_recursive_sequences (a b : ℕ → ℝ) (h : RecursiveSequences a b) :
  LimitRatio a b (1/4) := by
  sorry

end limit_ratio_recursive_sequences_l3028_302843


namespace lunchroom_students_l3028_302800

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end lunchroom_students_l3028_302800


namespace fraction_equation_solution_l3028_302859

theorem fraction_equation_solution (x : ℝ) : (x - 3) / (x + 3) = 2 → x = -9 := by
  sorry

end fraction_equation_solution_l3028_302859


namespace no_five_digit_sum_20_div_9_l3028_302813

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_five_digit_sum_20_div_9 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → ¬(n % 9 = 0) :=
by sorry

end no_five_digit_sum_20_div_9_l3028_302813


namespace square_difference_l3028_302833

theorem square_difference (a b : ℝ) :
  ∃ A : ℝ, (5*a + 3*b)^2 = (5*a - 3*b)^2 + A ∧ A = 60*a*b := by
  sorry

end square_difference_l3028_302833


namespace fuel_tank_capacity_l3028_302888

theorem fuel_tank_capacity 
  (fuel_a_ethanol_percentage : ℝ)
  (fuel_b_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : fuel_a_ethanol_percentage = 0.12)
  (h2 : fuel_b_ethanol_percentage = 0.16)
  (h3 : total_ethanol = 28)
  (h4 : fuel_a_volume = 99.99999999999999)
  : ∃ (capacity : ℝ), 
    fuel_a_ethanol_percentage * fuel_a_volume + 
    fuel_b_ethanol_percentage * (capacity - fuel_a_volume) = total_ethanol ∧
    capacity = 200 :=
by sorry

end fuel_tank_capacity_l3028_302888


namespace sum_of_roots_quadratic_l3028_302886

theorem sum_of_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -8
  let c : ℝ := -7
  let sum_of_roots := -b / a
  sum_of_roots = 8 := by
sorry

end sum_of_roots_quadratic_l3028_302886


namespace xyz_equals_five_l3028_302874

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq_a : a = (b^2 + c^2) / (x - 3))
  (eq_b : b = (a^2 + c^2) / (y - 3))
  (eq_c : c = (a^2 + b^2) / (z - 3))
  (sum_prod : x*y + y*z + z*x = 11)
  (sum : x + y + z = 5) :
  x * y * z = 5 := by
sorry

end xyz_equals_five_l3028_302874


namespace complex_moduli_product_l3028_302895

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_moduli_product_l3028_302895


namespace maintenance_check_time_l3028_302891

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time * 1.2 = 30) → initial_time = 25 := by
  sorry

end maintenance_check_time_l3028_302891


namespace max_marks_calculation_l3028_302834

theorem max_marks_calculation (passing_percentage : ℝ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_percentage = 0.40 →
  scored_marks = 212 →
  short_marks = 44 →
  ∃ max_marks : ℕ, max_marks = 640 ∧ 
    (scored_marks + short_marks : ℝ) / max_marks = passing_percentage :=
by sorry

end max_marks_calculation_l3028_302834


namespace probability_second_red_given_first_red_is_five_ninths_l3028_302872

/-- Represents the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls. -/
def probability_second_red_given_first_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  if total_balls = red_balls + white_balls ∧ red_balls > 0 then
    (red_balls - 1) / (total_balls - 1)
  else
    0

/-- Theorem stating that the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls, is 5/9. -/
theorem probability_second_red_given_first_red_is_five_ninths :
  probability_second_red_given_first_red 10 6 4 = 5 / 9 := by
  sorry

end probability_second_red_given_first_red_is_five_ninths_l3028_302872


namespace decimal_sum_to_fraction_l3028_302840

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end decimal_sum_to_fraction_l3028_302840


namespace perpendicular_bisector_of_intersecting_circles_l3028_302842

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∀ (A B : ℝ × ℝ),
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) :=
sorry

end perpendicular_bisector_of_intersecting_circles_l3028_302842


namespace max_visible_cubes_is_400_l3028_302814

/-- The dimension of the cube --/
def n : ℕ := 12

/-- The number of unit cubes on one face of the cube --/
def face_count : ℕ := n^2

/-- The number of unit cubes along one edge of the cube --/
def edge_count : ℕ := n

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of edges shared between two visible faces --/
def shared_edges : ℕ := 3

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_count - shared_edges * (edge_count - 1) + 1

/-- Theorem stating that the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_is_400 : max_visible_cubes = 400 := by
  sorry

end max_visible_cubes_is_400_l3028_302814


namespace cos_double_angle_with_tan_l3028_302849

theorem cos_double_angle_with_tan (α : Real) (h : Real.tan α = 1/2) : 
  Real.cos (2 * α) = 3/5 := by
  sorry

end cos_double_angle_with_tan_l3028_302849


namespace oabc_shape_oabc_not_rhombus_l3028_302838

/-- Given distinct points A, B, and C on a coordinate plane with origin O,
    prove that OABC can form either a parallelogram or a straight line, but not a rhombus. -/
theorem oabc_shape (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  (∃ (k : ℝ), k ≠ 0 ∧ k ≠ 1 ∧ x₂ = k * x₁ ∧ y₂ = k * y₁) ∨ 
  (x₁ + x₂ = 2*x₁ - x₂ ∧ y₁ + y₂ = 2*y₁ - y₂) :=
by sorry

/-- The figure OABC cannot form a rhombus. -/
theorem oabc_not_rhombus (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  ¬(x₁^2 + y₁^2 = x₂^2 + y₂^2 ∧ 
    x₁^2 + y₁^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2 ∧ 
    x₂^2 + y₂^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2) :=
by sorry

end oabc_shape_oabc_not_rhombus_l3028_302838


namespace f_lower_bound_l3028_302878

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x

-- State the theorem
theorem f_lower_bound (a : ℝ) (h_a : a > 0) :
  ∀ x > 0, f a x ≥ a * (2 - Real.log a) := by
  sorry

end f_lower_bound_l3028_302878


namespace jess_walks_to_gallery_l3028_302847

/-- The number of blocks Jess walks to work -/
def total_blocks : ℕ := 25

/-- The number of blocks Jess walks to the store -/
def blocks_to_store : ℕ := 11

/-- The number of blocks Jess walks from the gallery to work -/
def blocks_gallery_to_work : ℕ := 8

/-- The number of blocks Jess walks to the gallery -/
def blocks_to_gallery : ℕ := total_blocks - blocks_to_store - blocks_gallery_to_work

theorem jess_walks_to_gallery : blocks_to_gallery = 6 := by
  sorry

end jess_walks_to_gallery_l3028_302847


namespace quadratic_equation_result_l3028_302807

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x - 1 = 0) : 2*x^2 - 2*x + 2021 = 2023 := by
  sorry

end quadratic_equation_result_l3028_302807


namespace fraction_equality_l3028_302808

theorem fraction_equality (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^4 / n^5) * (n^4 / m^3) = m / n := by
  sorry

end fraction_equality_l3028_302808


namespace train_platform_crossing_time_l3028_302854

theorem train_platform_crossing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 800) :
  let train_speed := train_length / tree_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 200 :=
by sorry

end train_platform_crossing_time_l3028_302854


namespace chord_passes_through_fixed_point_l3028_302801

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Parabola C with equation x^2 = 4y -/
def parabolaC (p : Point) : Prop :=
  p.x^2 = 4 * p.y

/-- Dot product of two vectors represented by points -/
def dotProduct (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- Condition that the dot product of OA and OB is -4 -/
def dotProductCondition (a b : Point) : Prop :=
  dotProduct a b = -4

/-- Line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem stating that if a chord AB of parabola C satisfies the dot product condition,
    then the line AB always passes through the point (0, 2) -/
theorem chord_passes_through_fixed_point 
  (a b : Point) (l : Line) 
  (h1 : parabolaC a) 
  (h2 : parabolaC b) 
  (h3 : dotProductCondition a b) 
  (h4 : linePassesThrough l a) 
  (h5 : linePassesThrough l b) : 
  linePassesThrough l (Point.mk 0 2) :=
sorry

end chord_passes_through_fixed_point_l3028_302801


namespace max_regions_intersected_by_line_l3028_302826

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The number of regions that the planes of a tetrahedron divide space into -/
def num_regions_tetrahedron : ℕ := 15

/-- The maximum number of regions a line can intersect -/
def max_intersected_regions (t : Tetrahedron) (l : Line) : ℕ := sorry

/-- Theorem stating the maximum number of regions a line can intersect -/
theorem max_regions_intersected_by_line (t : Tetrahedron) :
  ∃ l : Line, max_intersected_regions t l = 5 ∧
  ∀ l' : Line, max_intersected_regions t l' ≤ 5 :=
sorry

end max_regions_intersected_by_line_l3028_302826


namespace probability_prime_sum_two_dice_l3028_302858

/-- A fair die with sides numbered from 1 to 6 -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoRolls : Type := Die × Die

/-- Function to check if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The sum of two dice rolls -/
def rollSum (roll : TwoRolls) : ℕ := sorry

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset TwoRolls := sorry

/-- The set of outcomes where the sum is prime -/
def primeOutcomes : Finset TwoRolls := sorry

/-- Theorem: The probability of rolling a prime sum with two fair dice is 5/12 -/
theorem probability_prime_sum_two_dice : 
  (Finset.card primeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 5 / 12 := by sorry

end probability_prime_sum_two_dice_l3028_302858


namespace system_one_solution_set_system_two_solution_set_l3028_302817

-- System 1
theorem system_one_solution_set :
  {x : ℝ | 3*x > x + 6 ∧ (1/2)*x < -x + 5} = {x : ℝ | 3 < x ∧ x < 10/3} := by sorry

-- System 2
theorem system_two_solution_set :
  {x : ℝ | 2*x - 1 < 5 - 2*(x-1) ∧ (3+5*x)/3 > 1} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end system_one_solution_set_system_two_solution_set_l3028_302817


namespace problem_1_l3028_302869

theorem problem_1 : (5 / 17) * (-4) - (5 / 17) * 15 + (-5 / 17) * (-2) = -5 := by
  sorry

end problem_1_l3028_302869


namespace smallest_bob_number_l3028_302828

/-- Alice's number -/
def alice_number : ℕ := 45

/-- Bob's number is a natural number -/
def bob_number : ℕ := sorry

/-- Every prime factor of Alice's number is also a prime factor of Bob's number -/
axiom bob_has_alice_prime_factors :
  ∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ bob_number

/-- Bob's number is the smallest possible given the conditions -/
axiom bob_number_is_smallest :
  ∀ n : ℕ, (∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ n) → bob_number ≤ n

theorem smallest_bob_number : bob_number = 15 := by sorry

end smallest_bob_number_l3028_302828


namespace hyperbola_eccentricity_l3028_302863

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (b / a) = π / 6) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l3028_302863


namespace min_y_value_l3028_302896

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 56*y) : 
  y ≥ 28 - 2 * Real.sqrt 212 := by
sorry

end min_y_value_l3028_302896


namespace correct_alarm_time_l3028_302844

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60 := by sorry

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time := by sorry

theorem correct_alarm_time :
  let alarmSetTime : Time := ⟨7, 0, by sorry⟩
  let museumArrivalTime : Time := ⟨8, 50, by sorry⟩
  let museumVisitDuration : ℕ := 90 -- in minutes
  let returnHomeTime : Time := ⟨11, 50, by sorry⟩
  
  let totalTripTime := timeDifference alarmSetTime returnHomeTime
  let walkingTime := totalTripTime - museumVisitDuration
  let oneWayWalkingTime := walkingTime / 2
  
  let museumDepartureTime := addMinutes museumArrivalTime museumVisitDuration
  let actualReturnTime := addMinutes museumDepartureTime oneWayWalkingTime
  
  let correctTime := addMinutes actualReturnTime 30

  correctTime = ⟨12, 0, by sorry⟩ := by sorry

end correct_alarm_time_l3028_302844


namespace negative_to_even_power_l3028_302870

theorem negative_to_even_power (a : ℝ) : (-a)^4 = a^4 := by
  sorry

end negative_to_even_power_l3028_302870
