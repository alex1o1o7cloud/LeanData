import Mathlib

namespace mariels_dogs_count_l3288_328800

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ := 5

/-- The number of dogs the other walker has -/
def other_walkers_dogs : ℕ := 3

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each human has -/
def human_legs : ℕ := 2

/-- The total number of legs tangled in leashes -/
def total_legs : ℕ := 36

/-- The number of dog walkers -/
def num_walkers : ℕ := 2

theorem mariels_dogs_count :
  mariels_dogs * dog_legs + 
  other_walkers_dogs * dog_legs + 
  num_walkers * human_legs = total_legs := by sorry

end mariels_dogs_count_l3288_328800


namespace lcm_of_9_and_14_l3288_328824

theorem lcm_of_9_and_14 : Nat.lcm 9 14 = 126 := by
  sorry

end lcm_of_9_and_14_l3288_328824


namespace tv_sets_b_is_30_l3288_328821

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop a -/
def tv_sets_a : ℕ := 20

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: The number of TV sets in shop b is 30 -/
theorem tv_sets_b_is_30 : 
  num_shops * average_tv_sets - (tv_sets_a + tv_sets_c + tv_sets_d + tv_sets_e) = 30 := by
  sorry

end tv_sets_b_is_30_l3288_328821


namespace cube_root_of_square_l3288_328875

theorem cube_root_of_square (x : ℝ) : x > 0 → (x^2)^(1/3) = x^(2/3) := by sorry

end cube_root_of_square_l3288_328875


namespace polynomial_decomposition_l3288_328863

theorem polynomial_decomposition (x : ℝ) : 
  1 + x^5 + x^10 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) := by
  sorry

end polynomial_decomposition_l3288_328863


namespace quadratic_crosses_origin_l3288_328898

/-- Given a quadratic function g(x) = ax^2 + bx where a ≠ 0 and b ≠ 0,
    the graph crosses the x-axis at the origin. -/
theorem quadratic_crosses_origin (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (g 0 = 0) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε \ {0}, g x ≠ 0) :=
by sorry

end quadratic_crosses_origin_l3288_328898


namespace line_segment_endpoint_l3288_328823

theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (x - 1)^2 + (8 - 3)^2 = 15^2 → 
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end line_segment_endpoint_l3288_328823


namespace honey_jar_problem_l3288_328801

/-- The proportion of honey remaining after each extraction -/
def remaining_proportion : ℚ := 75 / 100

/-- The number of times the extraction process is repeated -/
def num_extractions : ℕ := 6

/-- The amount of honey remaining after all extractions (in grams) -/
def final_honey : ℚ := 420

/-- Calculates the initial amount of honey given the final amount and extraction process -/
def initial_honey : ℚ := final_honey / remaining_proportion ^ num_extractions

theorem honey_jar_problem :
  initial_honey * remaining_proportion ^ num_extractions = final_honey :=
sorry

end honey_jar_problem_l3288_328801


namespace function_value_at_2_l3288_328830

/-- Given a function f(x) = ax^5 - bx + |x| - 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem function_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x + |x| - 1)
    (h2 : f (-2) = 2) : 
  f 2 = 0 := by
  sorry

end function_value_at_2_l3288_328830


namespace galaxy_composition_l3288_328869

/-- Represents the counts of celestial bodies in a galaxy -/
structure GalaxyComposition where
  planets : ℕ
  solarSystems : ℕ
  stars : ℕ
  moonSystems : ℕ

/-- Calculates the composition of a galaxy based on given ratios and planet count -/
def calculateGalaxyComposition (planetCount : ℕ) : GalaxyComposition :=
  let solarSystems := planetCount * 8
  let stars := solarSystems * 4
  let moonSystems := planetCount * 3 / 5
  { planets := planetCount
  , solarSystems := solarSystems
  , stars := stars
  , moonSystems := moonSystems }

/-- Theorem stating the composition of the galaxy given the conditions -/
theorem galaxy_composition :
  let composition := calculateGalaxyComposition 20
  composition.planets = 20 ∧
  composition.solarSystems = 160 ∧
  composition.stars = 640 ∧
  composition.moonSystems = 12 :=
by sorry

end galaxy_composition_l3288_328869


namespace sales_executive_target_earning_l3288_328884

/-- Calculates the target monthly earning for a sales executive --/
def target_monthly_earning (fixed_salary : ℝ) (commission_rate : ℝ) (required_sales : ℝ) : ℝ :=
  fixed_salary + commission_rate * required_sales

/-- Proves that the target monthly earning is $5000 given the specified conditions --/
theorem sales_executive_target_earning :
  target_monthly_earning 1000 0.05 80000 = 5000 := by
sorry

end sales_executive_target_earning_l3288_328884


namespace point_in_fourth_quadrant_l3288_328899

def complex_number : ℂ := 2 - Complex.I

theorem point_in_fourth_quadrant (z : ℂ) (h : z = complex_number) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end point_in_fourth_quadrant_l3288_328899


namespace quadratic_inequality_solution_set_l3288_328861

theorem quadratic_inequality_solution_set (x : ℝ) : 
  -x^2 + 2*x + 3 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_solution_set_l3288_328861


namespace area_increase_rect_to_circle_l3288_328841

/-- Increase in area when changing a rectangular field to a circular field -/
theorem area_increase_rect_to_circle (length width : ℝ) (h1 : length = 60) (h2 : width = 20) :
  let rect_area := length * width
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * radius^2
  ∃ ε > 0, abs (circle_area - rect_area - 837.94) < ε :=
by sorry

end area_increase_rect_to_circle_l3288_328841


namespace count_valid_numbers_l3288_328843

def is_valid_number (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (100000 * a + 19880 + b) % 12 = 0

theorem count_valid_numbers :
  ∃ (S : Finset (Nat × Nat)),
    (∀ (p : Nat × Nat), p ∈ S ↔ is_valid_number p.1 p.2) ∧
    S.card = 9 :=
sorry

end count_valid_numbers_l3288_328843


namespace baseball_ratio_l3288_328846

theorem baseball_ratio (games_played : ℕ) (games_won : ℕ) 
  (h1 : games_played = 10) (h2 : games_won = 5) :
  (games_played : ℚ) / (games_played - games_won) = 2 := by
  sorry

end baseball_ratio_l3288_328846


namespace larger_number_problem_l3288_328883

theorem larger_number_problem (x y : ℤ) : 
  x + y = 56 → y = x + 12 → y = 34 := by
  sorry

end larger_number_problem_l3288_328883


namespace mean_of_specific_numbers_l3288_328850

theorem mean_of_specific_numbers :
  let numbers : List ℝ := [12, 14, 16, 18]
  (numbers.sum / numbers.length : ℝ) = 15 := by
  sorry

end mean_of_specific_numbers_l3288_328850


namespace one_thirds_in_nine_thirds_l3288_328858

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end one_thirds_in_nine_thirds_l3288_328858


namespace first_day_distance_l3288_328855

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  (total_distance * (1 - ratio) / (1 - ratio^days)) = 192 :=
sorry

end first_day_distance_l3288_328855


namespace arithmetic_expression_evaluation_l3288_328886

theorem arithmetic_expression_evaluation :
  37 + (87 / 29) + (15 * 19) - 100 - (450 / 15) + 13 = 208 := by
  sorry

end arithmetic_expression_evaluation_l3288_328886


namespace unique_solution_l3288_328857

/-- Represents a three-digit number in the form ABA --/
def ABA (A B : ℕ) : ℕ := 100 * A + 10 * B + A

/-- Represents a four-digit number in the form CCDC --/
def CCDC (C D : ℕ) : ℕ := 1000 * C + 100 * C + 10 * D + C

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    (ABA A B)^2 = CCDC C D ∧
    CCDC C D < 100000 ∧
    A = 2 ∧ B = 1 ∧ C = 4 ∧ D = 9 :=
by sorry

end unique_solution_l3288_328857


namespace golden_ratio_pentagon_l3288_328807

theorem golden_ratio_pentagon (θ : Real) : 
  θ = 108 * Real.pi / 180 →  -- Interior angle of a regular pentagon
  2 * Real.sin (18 * Real.pi / 180) = (Real.sqrt 5 - 1) / 2 →
  Real.sin θ / Real.sin (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end golden_ratio_pentagon_l3288_328807


namespace fifteen_degrees_to_radians_l3288_328889

/-- Conversion of 15 degrees to radians -/
theorem fifteen_degrees_to_radians : 
  (15 : ℝ) * π / 180 = π / 12 := by sorry

end fifteen_degrees_to_radians_l3288_328889


namespace mean_quiz_score_l3288_328829

def quiz_scores : List ℝ := [88, 90, 94, 86, 85, 91]

theorem mean_quiz_score : 
  (quiz_scores.sum / quiz_scores.length : ℝ) = 89 := by sorry

end mean_quiz_score_l3288_328829


namespace mark_bill_calculation_l3288_328866

def original_bill : ℝ := 500
def first_late_charge_rate : ℝ := 0.02
def second_late_charge_rate : ℝ := 0.03

def final_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem mark_bill_calculation : final_amount = 525.30 := by
  sorry

end mark_bill_calculation_l3288_328866


namespace unit_circle_sector_angle_l3288_328842

/-- In a unit circle, a sector with area 1 has a central angle of 2 radians -/
theorem unit_circle_sector_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 1 → area = 1 → angle = 2 * area / r → angle = 2 := by sorry

end unit_circle_sector_angle_l3288_328842


namespace john_initial_diamonds_l3288_328810

/-- Represents the number of diamonds each pirate has -/
structure DiamondCount where
  bill : ℕ
  sam : ℕ
  john : ℕ

/-- Represents the average mass of diamonds for each pirate -/
structure AverageMass where
  bill : ℝ
  sam : ℝ
  john : ℝ

/-- The initial distribution of diamonds -/
def initial_distribution : DiamondCount :=
  { bill := 12, sam := 12, john := 9 }

/-- The distribution after the theft events -/
def final_distribution : DiamondCount :=
  { bill := initial_distribution.bill,
    sam := initial_distribution.sam,
    john := initial_distribution.john }

/-- The change in average mass for each pirate -/
def mass_change : AverageMass :=
  { bill := -1, sam := -2, john := 4 }

theorem john_initial_diamonds :
  initial_distribution.john = 9 →
  (initial_distribution.bill * mass_change.bill +
   initial_distribution.sam * mass_change.sam +
   initial_distribution.john * mass_change.john = 0) :=
by
  sorry


end john_initial_diamonds_l3288_328810


namespace square_area_ratio_l3288_328895

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 2 * b) : b^2 = 4 * a^2 := by
  sorry

end square_area_ratio_l3288_328895


namespace linear_function_above_x_axis_l3288_328840

/-- A linear function y = ax + a + 2 is above the x-axis for -2 ≤ x ≤ 1 if and only if
    -1 < a < 2 and a ≠ 0 -/
theorem linear_function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → a * x + a + 2 > 0) ↔ (-1 < a ∧ a < 2 ∧ a ≠ 0) := by
  sorry

end linear_function_above_x_axis_l3288_328840


namespace sequence_decreasing_l3288_328817

/-- Given real numbers a and b such that b > a > 1, define the sequence x_n as follows:
    x_n = 2^n * (b^(1/2^n) - a^(1/2^n))
    This theorem states that the sequence is decreasing. -/
theorem sequence_decreasing (a b : ℝ) (h1 : b > a) (h2 : a > 1) :
  ∀ n : ℕ, (2^n * (b^(1/(2^n)) - a^(1/(2^n)))) > (2^(n+1) * (b^(1/(2^(n+1))) - a^(1/(2^(n+1))))) :=
by sorry

end sequence_decreasing_l3288_328817


namespace slopes_equal_necessary_not_sufficient_for_parallel_l3288_328827

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Theorem statement
theorem slopes_equal_necessary_not_sufficient_for_parallel :
  -- Given two lines
  ∀ (l1 l2 : Line),
  -- l1 has intercept 1
  l1.intercept = 1 →
  -- Necessary condition
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  -- Not sufficient condition
  ∃ l2 : Line, l1.slope = l2.slope ∧ ¬(parallel l1 l2) :=
by
  sorry

end slopes_equal_necessary_not_sufficient_for_parallel_l3288_328827


namespace super_mindmaster_codes_l3288_328865

theorem super_mindmaster_codes (colors : ℕ) (slots : ℕ) : 
  colors = 9 → slots = 5 → colors ^ slots = 59049 := by
  sorry

end super_mindmaster_codes_l3288_328865


namespace min_fold_length_l3288_328839

/-- Given a rectangle ABCD with AB = 6 and AD = 12, when corner B is folded to edge AD
    creating a fold line MN, this function represents the length of MN (l)
    as a function of t, where t = sin θ and θ = �angle MNB -/
def fold_length (t : ℝ) : ℝ := 6 * t

/-- The theorem states that the minimum value of the fold length is 0 -/
theorem min_fold_length :
  ∃ (t : ℝ), t ≥ 0 ∧ t ≤ 1 ∧ ∀ (s : ℝ), s ≥ 0 → s ≤ 1 → fold_length t ≤ fold_length s :=
by sorry

end min_fold_length_l3288_328839


namespace probability_ABABABBB_proof_l3288_328836

/-- The probability of arranging 5 A tiles and 3 B tiles in the specific order ABABABBB -/
def probability_ABABABBB : ℚ :=
  1 / 56

/-- The total number of ways to arrange 5 A tiles and 3 B tiles in a row -/
def total_arrangements : ℕ :=
  Nat.choose 8 5

theorem probability_ABABABBB_proof :
  probability_ABABABBB = (1 : ℚ) / total_arrangements := by
  sorry

#eval probability_ABABABBB
#eval total_arrangements

end probability_ABABABBB_proof_l3288_328836


namespace total_boxes_eq_sum_l3288_328874

/-- The total number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := sorry

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The additional number of boxes Kaylee needs to sell -/
def additional_boxes : ℕ := 12

/-- Theorem stating that the total number of boxes is equal to the sum of all sold boxes and additional boxes -/
theorem total_boxes_eq_sum :
  total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes + additional_boxes := by
  sorry

end total_boxes_eq_sum_l3288_328874


namespace age_difference_constant_l3288_328845

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Represents the current year --/
def CurrentYear : Type := Unit

/-- Represents a future year --/
structure FutureYear where
  yearsFromNow : ℕ

/-- The age difference between two people --/
def ageDifference (p1 p2 : Person) : ℕ :=
  if p1.age ≥ p2.age then p1.age - p2.age else p2.age - p1.age

/-- The age of a person after a number of years --/
def ageAfterYears (p : Person) (y : ℕ) : ℕ :=
  p.age + y

theorem age_difference_constant
  (a : ℕ)
  (n : ℕ)
  (xiaoShen : Person)
  (xiaoWang : Person)
  (h1 : xiaoShen.age = a)
  (h2 : xiaoWang.age = a - 8)
  : ageDifference
      { age := ageAfterYears xiaoShen (n + 3) }
      { age := ageAfterYears xiaoWang (n + 3) } = 8 := by
  sorry


end age_difference_constant_l3288_328845


namespace quadratic_roots_l3288_328837

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 0 ∧ x₂ = 4/5) ∧ 
  (∀ x : ℝ, 5 * x^2 = 4 * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_roots_l3288_328837


namespace solve_for_a_l3288_328813

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → a = 1 := by
  sorry

end solve_for_a_l3288_328813


namespace units_digit_of_m_squared_plus_three_to_m_l3288_328897

/-- The units digit of m^2 + 3^m is 5, where m = 2023^2 + 3^2023 -/
theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ) : 
  m = 2023^2 + 3^2023 → (m^2 + 3^m) % 10 = 5 := by
  sorry

end units_digit_of_m_squared_plus_three_to_m_l3288_328897


namespace problem_statement_l3288_328881

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.log x - a * x

theorem problem_statement :
  (∃ (a : ℝ), ∀ (x y : ℝ), 1 < x ∧ x < y → f a y ≤ f a x) ∧
  (∃ (a : ℝ), ∀ (x₁ x₂ : ℝ), Real.exp 1 ≤ x₁ ∧ x₁ ≤ Real.exp 2 ∧
                              Real.exp 1 ≤ x₂ ∧ x₂ ≤ Real.exp 2 →
                              f a x₁ ≤ (deriv (f a)) x₂ + a) :=
by
  sorry

end problem_statement_l3288_328881


namespace negation_of_universal_proposition_l3288_328891

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end negation_of_universal_proposition_l3288_328891


namespace polynomial_root_problem_l3288_328819

/-- Given two polynomials g and f, where g has three distinct roots that are also roots of f,
    prove that f(1) = -1333 -/
theorem polynomial_root_problem (a b c : ℝ) : 
  let g := fun x : ℝ => x^3 + a*x^2 + x + 8
  let f := fun x : ℝ => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ x : ℝ, g x = 0 → f x = 0) →
  f 1 = -1333 := by
sorry

end polynomial_root_problem_l3288_328819


namespace quadratic_inequality_range_l3288_328882

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) → -4 ≤ a ∧ a ≤ 0 :=
by sorry

end quadratic_inequality_range_l3288_328882


namespace trigonometric_identity_l3288_328818

theorem trigonometric_identity :
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) +
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) +
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end trigonometric_identity_l3288_328818


namespace exactly_two_pass_probability_l3288_328871

def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 3/4

theorem exactly_two_pass_probability : 
  let prob_AB := prob_A * prob_B * (1 - prob_C)
  let prob_AC := prob_A * (1 - prob_B) * prob_C
  let prob_BC := (1 - prob_A) * prob_B * prob_C
  prob_AB + prob_AC + prob_BC = 33/80 :=
by sorry

end exactly_two_pass_probability_l3288_328871


namespace billy_reads_three_books_l3288_328870

theorem billy_reads_three_books 
  (free_time_per_day : ℕ) 
  (weekend_days : ℕ) 
  (video_game_percentage : ℚ) 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (h1 : free_time_per_day = 8)
  (h2 : weekend_days = 2)
  (h3 : video_game_percentage = 3/4)
  (h4 : pages_per_hour = 60)
  (h5 : pages_per_book = 80) :
  (free_time_per_day * weekend_days * (1 - video_game_percentage) * pages_per_hour) / pages_per_book = 3 := by
  sorry

end billy_reads_three_books_l3288_328870


namespace complex_square_simplification_l3288_328868

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end complex_square_simplification_l3288_328868


namespace combination_sum_identity_l3288_328814

theorem combination_sum_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 := by
  sorry

end combination_sum_identity_l3288_328814


namespace triangle_negative_five_sixths_one_half_l3288_328877

/-- The triangle operation on rational numbers -/
def triangle (a b : ℚ) : ℚ := b - a

theorem triangle_negative_five_sixths_one_half :
  triangle (-5/6) (1/2) = 4/3 := by
  sorry

end triangle_negative_five_sixths_one_half_l3288_328877


namespace second_square_area_is_676_l3288_328835

/-- An isosceles right triangle with inscribed squares -/
structure TriangleWithSquares where
  /-- Side length of the first inscribed square -/
  a : ℝ
  /-- Area of the first inscribed square is 169 -/
  h_area : a^2 = 169

/-- The area of the second inscribed square -/
def second_square_area (t : TriangleWithSquares) : ℝ :=
  (2 * t.a)^2

theorem second_square_area_is_676 (t : TriangleWithSquares) :
  second_square_area t = 676 := by
  sorry

end second_square_area_is_676_l3288_328835


namespace quadratic_function_theorem_l3288_328802

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 0) 
  (h3 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x := by
  sorry

end quadratic_function_theorem_l3288_328802


namespace power_sum_division_l3288_328887

theorem power_sum_division (x y : ℕ) (hx : x = 3) (hy : y = 4) : (x^5 + 3*y^3) / 9 = 48 := by
  sorry

end power_sum_division_l3288_328887


namespace complex_magnitude_fourth_power_l3288_328803

theorem complex_magnitude_fourth_power : Complex.abs ((2 + 3*Complex.I)^4) = 169 := by
  sorry

end complex_magnitude_fourth_power_l3288_328803


namespace f_derivative_at_zero_l3288_328831

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem f_derivative_at_zero : 
  deriv f 0 = 720 := by
  sorry

end f_derivative_at_zero_l3288_328831


namespace line_through_circle_center_l3288_328876

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0) → 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0 → x = -1 ∧ y = 2) → 
  a = 1 := by
sorry

end line_through_circle_center_l3288_328876


namespace prob_at_least_one_karnataka_is_five_sixths_l3288_328834

/-- The probability of selecting at least one student from Karnataka -/
def prob_at_least_one_karnataka : ℚ :=
  let total_students : ℕ := 10
  let maharashtra_students : ℕ := 4
  let goa_students : ℕ := 3
  let karnataka_students : ℕ := 3
  let students_to_select : ℕ := 4
  1 - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
      (Nat.choose total_students students_to_select : ℚ)

/-- Theorem stating that the probability of selecting at least one student from Karnataka is 5/6 -/
theorem prob_at_least_one_karnataka_is_five_sixths :
  prob_at_least_one_karnataka = 5 / 6 := by
  sorry

end prob_at_least_one_karnataka_is_five_sixths_l3288_328834


namespace unique_solution_square_equation_l3288_328880

theorem unique_solution_square_equation :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end unique_solution_square_equation_l3288_328880


namespace min_n_for_divisibility_by_20_l3288_328894

theorem min_n_for_divisibility_by_20 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℕ), S.card = n →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧
    ¬∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  n = 9 :=
by sorry

end min_n_for_divisibility_by_20_l3288_328894


namespace problem_solution_l3288_328878

theorem problem_solution : 
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (5 * Real.sqrt 65) / 13 := by
  sorry

end problem_solution_l3288_328878


namespace ice_cream_volume_l3288_328849

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let sphere_volume := (4/3) * π * r^3
  cone_volume + sphere_volume = 72 * π :=
by
  sorry

end ice_cream_volume_l3288_328849


namespace integer_solution_exists_l3288_328890

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end integer_solution_exists_l3288_328890


namespace youngsville_population_change_l3288_328811

def initial_population : ℕ := 684
def growth_rate : ℚ := 25 / 100
def decline_rate : ℚ := 40 / 100

theorem youngsville_population_change :
  let increased_population := initial_population + (initial_population * growth_rate).floor
  let final_population := increased_population - (increased_population * decline_rate).floor
  final_population = 513 := by sorry

end youngsville_population_change_l3288_328811


namespace hat_problem_l3288_328873

/-- The number of customers -/
def n : ℕ := 5

/-- The probability that no customer gets their own hat when n customers randomly take hats -/
def prob_no_own_hat (n : ℕ) : ℚ :=
  sorry

theorem hat_problem : prob_no_own_hat n = 11/30 := by
  sorry

end hat_problem_l3288_328873


namespace initial_average_marks_l3288_328879

theorem initial_average_marks 
  (n : ℕ) 
  (incorrect_mark correct_mark : ℕ) 
  (correct_average : ℚ) : 
  n = 10 → 
  incorrect_mark = 90 → 
  correct_mark = 10 → 
  correct_average = 92 → 
  (n * correct_average + (incorrect_mark - correct_mark)) / n = 100 :=
by sorry

end initial_average_marks_l3288_328879


namespace min_value_condition_l3288_328844

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

theorem min_value_condition (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f b x ≥ 1) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f b x = 1) ↔ 
  b = Real.sqrt 2 ∨ b = -3/2 :=
sorry

end min_value_condition_l3288_328844


namespace max_silver_tokens_l3288_328864

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver
  | BlueToSilver
  | BothToSilver

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver => 
      { red := tc.red - 4, blue := tc.blue + 1, silver := tc.silver + 2 }
  | ExchangeRule.BlueToSilver => 
      { red := tc.red + 1, blue := tc.blue - 5, silver := tc.silver + 2 }
  | ExchangeRule.BothToSilver => 
      { red := tc.red - 3, blue := tc.blue - 3, silver := tc.silver + 3 }

/-- Checks if an exchange is possible --/
def canExchange (tc : TokenCount) (rule : ExchangeRule) : Prop :=
  match rule with
  | ExchangeRule.RedToSilver => tc.red ≥ 4
  | ExchangeRule.BlueToSilver => tc.blue ≥ 5
  | ExchangeRule.BothToSilver => tc.red ≥ 3 ∧ tc.blue ≥ 3

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (final : TokenCount), 
    (∃ (exchanges : List ExchangeRule), 
      final = exchanges.foldl applyExchange { red := 100, blue := 100, silver := 0 } ∧
      ∀ rule, ¬(canExchange final rule)) ∧
    final.silver = 85 :=
  sorry


end max_silver_tokens_l3288_328864


namespace root_sum_reciprocal_l3288_328833

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
  sorry

end root_sum_reciprocal_l3288_328833


namespace parabola_intersections_l3288_328832

-- Define the parabola
def W (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line x = 4
def L (x : ℝ) : Prop := x = 4

-- Define points A and B
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (4, -4)

-- Define point P
structure Point (x₀ y₀ : ℝ) : Prop :=
  (on_parabola : W x₀ y₀)
  (x_constraint : x₀ < 4)
  (y_constraint : y₀ ≥ 0)

-- Define the area of triangle PAB
def area_PAB (x₀ y₀ : ℝ) : ℝ := 4 * (4 - x₀)

-- Define the perpendicularity condition
def perp_condition (x₀ y₀ : ℝ) : Prop :=
  (4 - y₀^2/4)^2 = (4 - y₀) * (4 + y₀)

-- Define the area of triangle PMN
def area_PMN (y₀ : ℝ) : ℝ := y₀^2

theorem parabola_intersections 
  (x₀ y₀ : ℝ) (p : Point x₀ y₀) :
  (area_PAB x₀ y₀ = 4 → x₀ = 3 ∧ y₀ = 2 * Real.sqrt 3) ∧
  (perp_condition x₀ y₀ → Real.sqrt ((4 - x₀)^2 + (4 - y₀)^2) = 4 * Real.sqrt 2) ∧
  (area_PMN y₀ = area_PAB x₀ y₀ → area_PMN y₀ = 8) :=
sorry

end parabola_intersections_l3288_328832


namespace seed_mixture_problem_l3288_328867

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The problem statement -/
theorem seed_mixture_problem (X Y : SeedMixture) (mixture_weight : ℝ) :
  X.ryegrass = 40 →
  Y.ryegrass = 25 →
  Y.fescue = 75 →
  X.ryegrass + X.bluegrass + X.fescue = 100 →
  Y.ryegrass + Y.bluegrass + Y.fescue = 100 →
  mixture_weight * 30 / 100 = X.ryegrass * (mixture_weight * 100 / 3 / 100) + Y.ryegrass * (mixture_weight * 200 / 3 / 100) →
  X.bluegrass = 60 := by
  sorry


end seed_mixture_problem_l3288_328867


namespace inequality_proof_l3288_328851

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end inequality_proof_l3288_328851


namespace expression_value_l3288_328847

theorem expression_value (x y : ℝ) (h1 : x = 3 * y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end expression_value_l3288_328847


namespace single_tile_replacement_impossible_l3288_328893

/-- Represents the two types of tiles used for paving -/
inductive TileType
  | Rectangle4x1
  | Square2x2

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ)
  (height : ℕ)
  (tiling : List (TileType))

/-- Checks if a tiling is valid for the given floor -/
def is_valid_tiling (floor : Floor) : Prop :=
  sorry

/-- Represents the operation of replacing a single tile -/
def replace_single_tile (floor : Floor) (old_type new_type : TileType) : Floor :=
  sorry

/-- The main theorem stating that replacing a single tile
    with a different type always results in an invalid tiling -/
theorem single_tile_replacement_impossible (floor : Floor) :
  ∀ (old_type new_type : TileType),
    old_type ≠ new_type →
    is_valid_tiling floor →
    ¬(is_valid_tiling (replace_single_tile floor old_type new_type)) :=
  sorry

end single_tile_replacement_impossible_l3288_328893


namespace parabola_shift_theorem_l3288_328862

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

/-- The shifted parabola function -/
def shifted_parabola (n : ℝ) (x : ℝ) : ℝ := original_parabola (x - n)

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_shift_theorem (n : ℝ) (y1 y2 : ℝ) 
  (h1 : n > 0)
  (h2 : shifted_parabola n 2 = y1)
  (h3 : shifted_parabola n 4 = y2)
  (h4 : y1 > y2) :
  n = 6 := by sorry

end parabola_shift_theorem_l3288_328862


namespace cube_plus_self_equality_l3288_328804

theorem cube_plus_self_equality (m n : ℤ) : m^3 = n^3 + n → m = 0 ∧ n = 0 := by
  sorry

end cube_plus_self_equality_l3288_328804


namespace walnut_trees_planted_l3288_328854

theorem walnut_trees_planted (trees_before planting : ℕ) (trees_after : ℕ) : 
  trees_before = 22 → trees_after = 55 → planting = trees_after - trees_before :=
by
  sorry

#check walnut_trees_planted 22 33 55

end walnut_trees_planted_l3288_328854


namespace hypotenuse_segment_ratio_l3288_328885

/-- Represents a right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- Length of the shorter leg -/
  short_leg : ℝ
  /-- Length of the longer leg -/
  long_leg : ℝ
  /-- The longer leg is 3 times the shorter leg -/
  leg_ratio : long_leg = 3 * short_leg
  /-- Length of the segment of the hypotenuse adjacent to the shorter leg -/
  hyp_short : ℝ
  /-- Length of the segment of the hypotenuse adjacent to the longer leg -/
  hyp_long : ℝ
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : short_leg ^ 2 + long_leg ^ 2 = (hyp_short + hyp_long) ^ 2

/-- The main theorem: the ratio of hypotenuse segments is 9:1 -/
theorem hypotenuse_segment_ratio (t : RightTriangleWithAltitude) : 
  t.hyp_long / t.hyp_short = 9 := by
  sorry

end hypotenuse_segment_ratio_l3288_328885


namespace lindas_bills_l3288_328828

/-- Represents the number of bills of each denomination -/
structure BillCount where
  fives : ℕ
  tens : ℕ

/-- Calculates the total value of bills -/
def totalValue (bc : BillCount) : ℕ :=
  5 * bc.fives + 10 * bc.tens

/-- Calculates the total number of bills -/
def totalBills (bc : BillCount) : ℕ :=
  bc.fives + bc.tens

theorem lindas_bills :
  ∃ (bc : BillCount), totalValue bc = 80 ∧ totalBills bc = 12 ∧ bc.fives = 8 := by
  sorry

end lindas_bills_l3288_328828


namespace area_of_similar_rectangle_l3288_328822

-- Define the properties of rectangle R1
def R1_side : ℝ := 3
def R1_area : ℝ := 24

-- Define the diagonal of rectangle R2
def R2_diagonal : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle :
  let R1_other_side := R1_area / R1_side
  let ratio := R1_other_side / R1_side
  let R2_side := (R2_diagonal^2 / (1 + ratio^2))^(1/2)
  R2_side * (ratio * R2_side) = 28800 / 219 :=
by sorry

end area_of_similar_rectangle_l3288_328822


namespace zu_chongzhi_complex_theory_incorrect_l3288_328809

-- Define a structure for a scientist-field pairing
structure ScientistFieldPair where
  scientist : String
  field : String

-- Define the list of pairings
def pairings : List ScientistFieldPair := [
  { scientist := "Descartes", field := "Analytic Geometry" },
  { scientist := "Pascal", field := "Probability Theory" },
  { scientist := "Cantor", field := "Set Theory" },
  { scientist := "Zu Chongzhi", field := "Complex Number Theory" }
]

-- Define a function to check if a pairing is correct based on historical contributions
def isCorrectPairing (pair : ScientistFieldPair) : Bool :=
  match pair with
  | { scientist := "Descartes", field := "Analytic Geometry" } => true
  | { scientist := "Pascal", field := "Probability Theory" } => true
  | { scientist := "Cantor", field := "Set Theory" } => true
  | { scientist := "Zu Chongzhi", field := "Complex Number Theory" } => false
  | _ => false

-- Theorem: The pairing of Zu Chongzhi with Complex Number Theory is incorrect
theorem zu_chongzhi_complex_theory_incorrect :
  ∃ pair ∈ pairings, pair.scientist = "Zu Chongzhi" ∧ pair.field = "Complex Number Theory" ∧ ¬(isCorrectPairing pair) :=
by
  sorry

end zu_chongzhi_complex_theory_incorrect_l3288_328809


namespace range_of_a_l3288_328848

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) →  -- p is true
  (∃ x : ℝ, x > 0 ∧ 2^x - a ≤ 0) →  -- q is false
  a ∈ Set.Ioo 1 2 :=  -- a is in the open interval (1, 2)
by sorry

end range_of_a_l3288_328848


namespace product_equality_l3288_328896

theorem product_equality (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^12 + 21 * x^8 * y^3 + 21 * x^4 * y^6 + 49 * y^9) =
  81 * x^16 - 2401 * y^12 := by
  sorry

end product_equality_l3288_328896


namespace nellie_gift_wrap_sales_l3288_328853

/-- Given that Nellie needs to sell 45 rolls of gift wrap in total and has already sold some, 
    prove that she needs to sell 28 more rolls. -/
theorem nellie_gift_wrap_sales (total_needed : ℕ) (sold_to_grandmother : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) 
    (h1 : total_needed = 45)
    (h2 : sold_to_grandmother = 1)
    (h3 : sold_to_uncle = 10)
    (h4 : sold_to_neighbor = 6) :
    total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end nellie_gift_wrap_sales_l3288_328853


namespace equation_solution_l3288_328806

theorem equation_solution :
  ∀ x : ℚ, x ≠ 3 → ((x + 5) / (x - 3) = 4 ↔ x = 17 / 3) :=
by sorry

end equation_solution_l3288_328806


namespace at_least_one_greater_than_one_l3288_328815

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by
  sorry

end at_least_one_greater_than_one_l3288_328815


namespace special_function_property_l3288_328820

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to be proved -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end special_function_property_l3288_328820


namespace fraction_lower_bound_l3288_328872

theorem fraction_lower_bound (p q r s : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 := by
  sorry

end fraction_lower_bound_l3288_328872


namespace sugar_solution_sweetness_l3288_328888

theorem sugar_solution_sweetness (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) (hab : a > b) :
  (b + t) / (a + t) > b / a :=
by sorry

end sugar_solution_sweetness_l3288_328888


namespace expected_bullets_is_1_89_l3288_328816

/-- The expected number of remaining bullets in a shooting scenario -/
def expected_remaining_bullets (total_bullets : ℕ) (hit_probability : ℝ) : ℝ :=
  let miss_probability := 1 - hit_probability
  let p_zero := miss_probability * miss_probability
  let p_one := miss_probability * hit_probability
  let p_two := hit_probability
  1 * p_one + 2 * p_two

/-- The theorem stating that the expected number of remaining bullets is 1.89 -/
theorem expected_bullets_is_1_89 :
  expected_remaining_bullets 3 0.9 = 1.89 := by sorry

end expected_bullets_is_1_89_l3288_328816


namespace compound_animals_l3288_328860

theorem compound_animals (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = dogs - dogs / 5 →
  frogs = 2 * dogs →
  cats + dogs + frogs = 304 →
  frogs = 160 := by sorry

end compound_animals_l3288_328860


namespace log_and_power_equality_l3288_328826

theorem log_and_power_equality : 
  (Real.log 32 - Real.log 4) / Real.log 2 + (27 : ℝ) ^ (2/3) = 12 := by sorry

end log_and_power_equality_l3288_328826


namespace imaginary_part_of_complex_fraction_l3288_328808

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (2 - i)) = 3 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l3288_328808


namespace arithmetic_sequence_40th_term_l3288_328805

/-- Given an arithmetic sequence where the first term is 3 and the twentieth term is 63,
    prove that the fortieth term is 126. -/
theorem arithmetic_sequence_40th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                 -- first term is 3
    a 19 = 63 →                               -- twentieth term is 63
    a 39 = 126 := by                          -- fortieth term is 126
sorry


end arithmetic_sequence_40th_term_l3288_328805


namespace toms_total_cost_is_48_l3288_328892

/-- Represents the fruit prices and quantities --/
structure FruitPurchase where
  lemon_price : ℝ
  papaya_price : ℝ
  mango_price : ℝ
  orange_price : ℝ
  apple_price : ℝ
  pineapple_price : ℝ
  lemon_qty : ℕ
  papaya_qty : ℕ
  mango_qty : ℕ
  orange_qty : ℕ
  apple_qty : ℕ
  pineapple_qty : ℕ

/-- Calculates the total cost after all discounts --/
def totalCostAfterDiscounts (purchase : FruitPurchase) (customer_number : ℕ) : ℝ :=
  sorry

/-- Theorem stating that Tom's total cost after all discounts is $48 --/
theorem toms_total_cost_is_48 :
  let purchase : FruitPurchase := {
    lemon_price := 2,
    papaya_price := 1,
    mango_price := 4,
    orange_price := 3,
    apple_price := 1.5,
    pineapple_price := 5,
    lemon_qty := 8,
    papaya_qty := 6,
    mango_qty := 5,
    orange_qty := 3,
    apple_qty := 8,
    pineapple_qty := 2
  }
  totalCostAfterDiscounts purchase 7 = 48 := by sorry

end toms_total_cost_is_48_l3288_328892


namespace greatest_fourth_term_of_arithmetic_sequence_l3288_328838

theorem greatest_fourth_term_of_arithmetic_sequence 
  (a : ℕ) 
  (d : ℕ) 
  (sum_eq_65 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65) 
  (a_positive : a > 0) :
  ∀ (b : ℕ) (e : ℕ), 
    b > 0 → 
    b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 → 
    b + 3*e ≤ a + 3*d :=
by sorry

end greatest_fourth_term_of_arithmetic_sequence_l3288_328838


namespace area_S_eq_four_sqrt_three_thirds_l3288_328825

/-- A rhombus with side length 4 and one angle of 150 degrees -/
structure Rhombus150 where
  side_length : ℝ
  angle_F : ℝ
  side_length_eq : side_length = 4
  angle_F_eq : angle_F = 150 * π / 180

/-- The region S inside the rhombus closer to vertex F than to other vertices -/
def region_S (r : Rhombus150) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region S -/
noncomputable def area_S (r : Rhombus150) : ℝ :=
  sorry

/-- Theorem stating that the area of region S is 4√3/3 -/
theorem area_S_eq_four_sqrt_three_thirds (r : Rhombus150) :
  area_S r = 4 * Real.sqrt 3 / 3 :=
sorry

end area_S_eq_four_sqrt_three_thirds_l3288_328825


namespace cube_sum_eq_product_squares_l3288_328852

theorem cube_sum_eq_product_squares (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end cube_sum_eq_product_squares_l3288_328852


namespace complex_equation_solution_l3288_328856

theorem complex_equation_solution :
  ∃ (z : ℂ), (4 : ℂ) - 3 * Complex.I * z = (2 : ℂ) + 5 * Complex.I * z ∧ z = -(1/4) * Complex.I :=
by
  sorry

end complex_equation_solution_l3288_328856


namespace root_implies_sum_l3288_328812

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_implies_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by sorry

end root_implies_sum_l3288_328812


namespace inequality_solution_set_l3288_328859

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2*x > -m*x)) → m = -1 := by
  sorry

end inequality_solution_set_l3288_328859
