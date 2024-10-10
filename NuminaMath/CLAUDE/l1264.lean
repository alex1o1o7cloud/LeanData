import Mathlib

namespace smallest_multiple_of_45_and_60_not_25_l1264_126404

theorem smallest_multiple_of_45_and_60_not_25 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 60 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 60 ∣ m ∧ ¬(25 ∣ m) → n ≤ m :=
by sorry

end smallest_multiple_of_45_and_60_not_25_l1264_126404


namespace total_money_proof_l1264_126408

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
  sorry

end total_money_proof_l1264_126408


namespace square_border_pieces_l1264_126486

/-- The number of pieces on one side of the square arrangement -/
def side_length : ℕ := 12

/-- The total number of pieces in the border of a square arrangement -/
def border_pieces (n : ℕ) : ℕ := 2 * n + 2 * (n - 2)

/-- Theorem stating that in a 12x12 square arrangement, there are 44 pieces in the border -/
theorem square_border_pieces :
  border_pieces side_length = 44 := by
  sorry

#eval border_pieces side_length

end square_border_pieces_l1264_126486


namespace max_plus_min_equals_16_l1264_126493

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_plus_min_equals_16 :
  ∃ (M m : ℝ),
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ M) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = M) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, m ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = m) ∧
    M + m = 16 :=
by sorry

end max_plus_min_equals_16_l1264_126493


namespace quotient_problem_l1264_126435

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end quotient_problem_l1264_126435


namespace problem_statement_l1264_126422

theorem problem_statement (a b : ℝ) (h : 2 * a^2 - 3 * b + 5 = 0) :
  9 * b - 6 * a^2 + 3 = 18 := by
  sorry

end problem_statement_l1264_126422


namespace angle_properties_l1264_126445

def angle_set (α : Real) : Set Real :=
  {x | ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3}

theorem angle_properties (α : Real) 
  (h : ∃ x y : Real, x = 1 ∧ y = Real.sqrt 3 ∧ x * Real.cos α = x ∧ y * Real.sin α = y) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (angle_set α = {α}) := by
  sorry

end angle_properties_l1264_126445


namespace kitten_weight_l1264_126468

theorem kitten_weight (k d1 d2 : ℝ) 
  (total_weight : k + d1 + d2 = 30)
  (larger_dog_relation : k + d2 = 3 * d1)
  (smaller_dog_relation : k + d1 = d2 + 10) :
  k = 25 / 2 := by
sorry

end kitten_weight_l1264_126468


namespace fixed_point_of_f_l1264_126437

/-- The function f(x) defined as a^(x-2) + 3 for some base a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

/-- Theorem stating that (2, 4) is a fixed point of f(x) for any base a > 0 -/
theorem fixed_point_of_f (a : ℝ) (h : a > 0) : f a 2 = 4 := by sorry

end fixed_point_of_f_l1264_126437


namespace weight_loss_calculation_l1264_126497

/-- Proves that a measured weight loss of 9.22% with 2% added clothing weight
    corresponds to an actual weight loss of approximately 5.55% -/
theorem weight_loss_calculation (measured_loss : Real) (clothing_weight : Real) :
  measured_loss = 9.22 ∧ clothing_weight = 2 →
  ∃ actual_loss : Real,
    (100 - actual_loss) * (1 + clothing_weight / 100) = 100 - measured_loss ∧
    abs (actual_loss - 5.55) < 0.01 := by
  sorry

end weight_loss_calculation_l1264_126497


namespace min_months_for_committee_repetition_l1264_126452

theorem min_months_for_committee_repetition 
  (total_members : Nat) 
  (women : Nat) 
  (men : Nat) 
  (committee_size : Nat) 
  (h1 : total_members = 13)
  (h2 : women = 6)
  (h3 : men = 7)
  (h4 : committee_size = 5)
  (h5 : women + men = total_members) :
  let total_committees := Nat.choose total_members committee_size
  let women_only_committees := Nat.choose women committee_size
  let men_only_committees := Nat.choose men committee_size
  let valid_committees := total_committees - women_only_committees - men_only_committees
  valid_committees + 1 = 1261 := by
  sorry

end min_months_for_committee_repetition_l1264_126452


namespace sum_of_coefficients_l1264_126476

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
  sorry

end sum_of_coefficients_l1264_126476


namespace veg_eaters_count_l1264_126499

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (diet : FamilyDiet) : ℕ :=
  diet.onlyVeg + diet.bothVegAndNonVeg

/-- Theorem stating that for a given family diet, the total number of vegetarian eaters
    is equal to the sum of those who eat only vegetarian and those who eat both -/
theorem veg_eaters_count (diet : FamilyDiet) :
  totalVegEaters diet = diet.onlyVeg + diet.bothVegAndNonVeg := by
  sorry

/-- Example family with the given dietary information -/
def exampleFamily : FamilyDiet where
  onlyVeg := 19
  onlyNonVeg := 9
  bothVegAndNonVeg := 12

#eval totalVegEaters exampleFamily

end veg_eaters_count_l1264_126499


namespace existence_of_alpha_for_tan_l1264_126427

open Real

theorem existence_of_alpha_for_tan : ∃ α : ℝ, 
  (∃ α₀ : ℝ, tan (π / 2 - α₀) = 1) ∧ 
  (¬∀ α₁ : ℝ, tan (π / 2 - α₁) = 1) := by
  sorry

end existence_of_alpha_for_tan_l1264_126427


namespace sci_fi_readers_l1264_126403

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 400 → literary = 230 → both = 80 → 
  total = sci_fi + literary - both →
  sci_fi = 250 := by
sorry

end sci_fi_readers_l1264_126403


namespace min_value_when_a_is_one_range_of_a_for_nonnegative_f_l1264_126413

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem min_value_when_a_is_one :
  ∃ (x : ℝ), ∀ (y : ℝ), f 1 x ≤ f 1 y ∧ f 1 x = 0 :=
sorry

theorem range_of_a_for_nonnegative_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 0) ↔ a ∈ Set.Ici 1 :=
sorry

end min_value_when_a_is_one_range_of_a_for_nonnegative_f_l1264_126413


namespace sixth_term_value_l1264_126485

def sequence_property (s : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → s (n + 1) = (1 / 4) * (s n + s (n + 2))

theorem sixth_term_value (s : ℕ → ℚ) :
  sequence_property s →
  s 1 = 3 →
  s 5 = 48 →
  s 6 = 2001 / 14 := by
sorry

end sixth_term_value_l1264_126485


namespace tiling_comparison_l1264_126498

/-- Number of ways to tile a grid with rectangles -/
def tiling_count (grid_size : ℕ × ℕ) (tile_size : ℕ × ℕ) : ℕ := sorry

/-- Theorem: For any n > 1, the number of ways to tile a 3n × 3n grid with 1 × 3 rectangles
    is greater than the number of ways to tile a 2n × 2n grid with 1 × 2 rectangles -/
theorem tiling_comparison (n : ℕ) (h : n > 1) :
  tiling_count (3*n, 3*n) (1, 3) > tiling_count (2*n, 2*n) (1, 2) := by
  sorry

end tiling_comparison_l1264_126498


namespace students_present_l1264_126419

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 50 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 43 := by
  sorry

end students_present_l1264_126419


namespace meals_left_to_distribute_l1264_126400

theorem meals_left_to_distribute (initial_meals additional_meals distributed_meals : ℕ) :
  initial_meals + additional_meals - distributed_meals =
  (initial_meals + additional_meals) - distributed_meals :=
by sorry

end meals_left_to_distribute_l1264_126400


namespace polynomial_evaluation_l1264_126439

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 7 = 7 := by
sorry

end polynomial_evaluation_l1264_126439


namespace cos_120_degrees_l1264_126415

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end cos_120_degrees_l1264_126415


namespace base_conversion_1729_to_base7_l1264_126434

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- States that 1729 in base 10 is equal to 5020 in base 7 --/
theorem base_conversion_1729_to_base7 :
  1729 = base7ToBase10 5 0 2 0 := by
  sorry

end base_conversion_1729_to_base7_l1264_126434


namespace cyclist_speed_proof_l1264_126412

/-- The speed of the first cyclist in meters per second -/
def v : ℝ := 7

/-- The speed of the second cyclist in meters per second -/
def second_cyclist_speed : ℝ := 8

/-- The circumference of the circular track in meters -/
def track_circumference : ℝ := 300

/-- The time taken for the cyclists to meet at the starting point in seconds -/
def meeting_time : ℝ := 20

theorem cyclist_speed_proof :
  v * meeting_time + second_cyclist_speed * meeting_time = track_circumference :=
by sorry

end cyclist_speed_proof_l1264_126412


namespace number_of_late_classmates_l1264_126436

/-- The number of late classmates given Charlize's lateness, classmates' additional lateness, and total late time -/
def late_classmates (charlize_lateness : ℕ) (classmate_additional_lateness : ℕ) (total_late_time : ℕ) : ℕ :=
  (total_late_time - charlize_lateness) / (charlize_lateness + classmate_additional_lateness)

/-- Theorem stating that the number of late classmates is 4 given the specific conditions -/
theorem number_of_late_classmates :
  late_classmates 20 10 140 = 4 := by
  sorry

end number_of_late_classmates_l1264_126436


namespace minimum_bailing_rate_l1264_126421

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 6)
  (h4 : boat_capacity = 60) :
  ∃ (min_bailing_rate : ℝ),
    min_bailing_rate = 4.5 ∧
    ∀ (bailing_rate : ℝ),
      bailing_rate ≥ min_bailing_rate →
      (water_intake_rate - bailing_rate) * (distance_to_shore / rowing_speed * 60) ≤ boat_capacity :=
by sorry

end minimum_bailing_rate_l1264_126421


namespace largest_perimeter_is_164_l1264_126480

/-- Represents a rectangle with integer side lengths -/
structure IntRectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of an IntRectangle -/
def perimeter (r : IntRectangle) : ℕ :=
  2 * (r.length + r.width)

/-- Calculates the area of an IntRectangle -/
def area (r : IntRectangle) : ℕ :=
  r.length * r.width

/-- Checks if a rectangle satisfies the given condition -/
def satisfiesCondition (r : IntRectangle) : Prop :=
  4 * perimeter r = area r - 1

/-- Theorem stating that the largest possible perimeter of a rectangle satisfying the condition is 164 -/
theorem largest_perimeter_is_164 :
  ∀ r : IntRectangle, satisfiesCondition r → perimeter r ≤ 164 :=
by sorry

end largest_perimeter_is_164_l1264_126480


namespace pomelo_price_at_6kg_l1264_126446

-- Define the relationship between weight and price
def price_function (x : ℝ) : ℝ := 1.4 * x

-- Theorem statement
theorem pomelo_price_at_6kg : price_function 6 = 8.4 := by
  sorry

end pomelo_price_at_6kg_l1264_126446


namespace count_symmetric_scanning_codes_l1264_126402

/-- A symmetric scanning code is a 5x5 grid that remains unchanged when rotated by multiples of 90° or reflected across diagonal or midpoint lines. -/
def SymmetricScanningCode : Type := Unit

/-- The number of distinct symmetry groups in a 5x5 symmetric scanning code -/
def numSymmetryGroups : ℕ := 5

/-- The number of color choices for each symmetry group -/
def numColorChoices : ℕ := 2

/-- The total number of color combinations for all symmetry groups -/
def totalColorCombinations : ℕ := numColorChoices ^ numSymmetryGroups

/-- The number of invalid color combinations (all white or all black) -/
def invalidColorCombinations : ℕ := 2

theorem count_symmetric_scanning_codes :
  (totalColorCombinations - invalidColorCombinations : ℕ) = 30 := by
  sorry

end count_symmetric_scanning_codes_l1264_126402


namespace bicycle_count_l1264_126426

theorem bicycle_count (tricycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  tricycles = 7 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧ bicycles = 16 :=
by sorry

end bicycle_count_l1264_126426


namespace ordering_proof_l1264_126401

theorem ordering_proof (a b c : ℝ) : 
  a = (1/2)^(1/3) → b = (1/3)^(1/2) → c = Real.log (3/Real.pi) → c < b ∧ b < a := by
  sorry

end ordering_proof_l1264_126401


namespace lcm_hcf_problem_l1264_126481

theorem lcm_hcf_problem (A B : ℕ+) (h1 : B = 671) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 61) : A = 210 := by
  sorry

end lcm_hcf_problem_l1264_126481


namespace neg_a_fourth_times_neg_a_squared_l1264_126479

theorem neg_a_fourth_times_neg_a_squared (a : ℝ) : -a^4 * (-a)^2 = -a^6 := by
  sorry

end neg_a_fourth_times_neg_a_squared_l1264_126479


namespace polynomial_factorization_l1264_126414

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end polynomial_factorization_l1264_126414


namespace log_equality_l1264_126455

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5)) :=
by sorry

end log_equality_l1264_126455


namespace functional_equation_solution_l1264_126460

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → f (x + y) = g (1/x + 1/y) * (x*y)^2008) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c * x^2008 ∧ g x = c * x^2008 := by
sorry

end functional_equation_solution_l1264_126460


namespace farm_work_earnings_l1264_126464

/-- Calculates the total money collected given hourly rate, hours worked, and tips. -/
def total_money_collected (hourly_rate : ℕ) (hours_worked : ℕ) (tips : ℕ) : ℕ :=
  hourly_rate * hours_worked + tips

/-- Proves that given the specified conditions, the total money collected is $240. -/
theorem farm_work_earnings : total_money_collected 10 19 50 = 240 := by
  sorry

end farm_work_earnings_l1264_126464


namespace complement_of_intersection_l1264_126473

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set N
def N : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_intersection (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2, 4} ∧ N = {3, 4, 5}) :
  (U \ (M ∩ N)) = {1, 2, 3, 5} := by
  sorry

end complement_of_intersection_l1264_126473


namespace sum_equals_369_l1264_126411

theorem sum_equals_369 : 333 + 33 + 3 = 369 := by
  sorry

end sum_equals_369_l1264_126411


namespace base_number_problem_l1264_126466

theorem base_number_problem (x y a : ℝ) (h1 : x * y = 1) 
  (h2 : a ^ ((x + y)^2) / a ^ ((x - y)^2) = 625) : a = 5 := by
  sorry

end base_number_problem_l1264_126466


namespace total_population_of_three_cities_l1264_126494

/-- Given the populations of three cities with specific relationships, 
    prove that their total population is 56000. -/
theorem total_population_of_three_cities 
  (pop_lake_view pop_seattle pop_boise : ℕ) : 
  pop_lake_view = 24000 →
  pop_lake_view = pop_seattle + 4000 →
  pop_boise = (3 * pop_seattle) / 5 →
  pop_lake_view + pop_seattle + pop_boise = 56000 := by
  sorry

end total_population_of_three_cities_l1264_126494


namespace thinnest_gold_foil_scientific_notation_l1264_126438

theorem thinnest_gold_foil_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000092 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end thinnest_gold_foil_scientific_notation_l1264_126438


namespace lomonosov_digit_mapping_l1264_126487

theorem lomonosov_digit_mapping :
  ∃ (L O M N S V H C B : ℕ),
    (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧
    (V < 10) ∧ (H < 10) ∧ (C < 10) ∧ (B < 10) ∧
    (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧ (L ≠ H) ∧ (L ≠ C) ∧ (L ≠ B) ∧
    (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧ (O ≠ H) ∧ (O ≠ C) ∧ (O ≠ B) ∧
    (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧ (M ≠ H) ∧ (M ≠ C) ∧ (M ≠ B) ∧
    (N ≠ S) ∧ (N ≠ V) ∧ (N ≠ H) ∧ (N ≠ C) ∧ (N ≠ B) ∧
    (S ≠ V) ∧ (S ≠ H) ∧ (S ≠ C) ∧ (S ≠ B) ∧
    (V ≠ H) ∧ (V ≠ C) ∧ (V ≠ B) ∧
    (H ≠ C) ∧ (H ≠ B) ∧
    (C ≠ B) ∧
    (L + O / M + O + H + O / C = O * 10 + B) ∧
    (O < M) ∧ (O < C) := by
  sorry

end lomonosov_digit_mapping_l1264_126487


namespace triangle_side_length_l1264_126467

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2 / 3 →
  b = 3 :=
by sorry

end triangle_side_length_l1264_126467


namespace power_function_decreasing_interval_l1264_126429

/-- A power function passing through (2, 4) is monotonically decreasing on (-∞, 0) -/
theorem power_function_decreasing_interval 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^α) 
  (h2 : f 2 = 4) :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → f y < f x :=
by sorry

end power_function_decreasing_interval_l1264_126429


namespace birth_rate_calculation_l1264_126443

/-- The number of people born every two seconds in a city -/
def birth_rate : ℕ := sorry

/-- The death rate in the city (people per two seconds) -/
def death_rate : ℕ := 1

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 259200

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation : 
  birth_rate = (daily_net_increase / intervals_per_day) + death_rate := by
  sorry

end birth_rate_calculation_l1264_126443


namespace f_at_three_equals_five_l1264_126477

/-- A quadratic function f(x) = ax^2 + bx + 2 satisfying f(1) = 4 and f(2) = 5 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

/-- Theorem: Given f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 5, prove that f(3) = 5 -/
theorem f_at_three_equals_five (a b : ℝ) (h1 : f a b 1 = 4) (h2 : f a b 2 = 5) :
  f a b 3 = 5 := by
  sorry

end f_at_three_equals_five_l1264_126477


namespace closest_to_fraction_l1264_126447

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_fraction (x : ℝ) (h : x ∈ options) :
  |403 / 0.21 - 2000| ≤ |403 / 0.21 - x| :=
by sorry

end closest_to_fraction_l1264_126447


namespace female_students_count_l1264_126418

-- Define the total number of students
def total_students : ℕ := 8

-- Define the number of combinations
def total_combinations : ℕ := 30

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem female_students_count :
  ∃ f : ℕ, (f = 2 ∨ f = 3) ∧
  (∃ m : ℕ, m + f = total_students ∧
  combinations m 2 * combinations f 1 = total_combinations) :=
sorry

end female_students_count_l1264_126418


namespace hua_method_is_golden_ratio_l1264_126442

/-- Represents the possible methods used in optimal selection -/
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- The optimal selection method popularized by Hua Luogeng -/
def huaMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

/-- Theorem stating that Hua Luogeng's optimal selection method uses the Golden ratio -/
theorem hua_method_is_golden_ratio :
  huaMethod = OptimalSelectionMethod.GoldenRatio :=
by sorry

end hua_method_is_golden_ratio_l1264_126442


namespace determine_d_value_l1264_126449

theorem determine_d_value : ∃ d : ℝ, 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 3) → d = 15 := by
  sorry

end determine_d_value_l1264_126449


namespace candy_distribution_l1264_126489

theorem candy_distribution (left right : ℕ) : 
  left + right = 27 →
  right - left = (left + left) + 3 →
  left = 6 ∧ right = 21 := by
sorry

end candy_distribution_l1264_126489


namespace profit_is_085_l1264_126444

/-- Calculates the total profit for Niko's sock reselling business --/
def calculate_profit : ℝ :=
  let initial_cost : ℝ := 9 * 2
  let discount_rate : ℝ := 0.1
  let discount : ℝ := initial_cost * discount_rate
  let cost_after_discount : ℝ := initial_cost - discount
  let shipping_storage : ℝ := 5
  let total_cost : ℝ := cost_after_discount + shipping_storage
  let resell_price_4_pairs : ℝ := 4 * (2 + 2 * 0.25)
  let resell_price_5_pairs : ℝ := 5 * (2 + 0.2)
  let total_resell_price : ℝ := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax_rate : ℝ := 0.05
  let sales_tax : ℝ := total_resell_price * sales_tax_rate
  let total_revenue : ℝ := total_resell_price + sales_tax
  total_revenue - total_cost

theorem profit_is_085 : calculate_profit = 0.85 := by
  sorry

end profit_is_085_l1264_126444


namespace negation_of_absolute_value_less_than_zero_l1264_126430

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) := by
  sorry

end negation_of_absolute_value_less_than_zero_l1264_126430


namespace subset_implies_x_equals_one_l1264_126410

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 2/x}

theorem subset_implies_x_equals_one (x : ℝ) (h : B x ⊆ A) : x = 1 := by
  sorry

end subset_implies_x_equals_one_l1264_126410


namespace triangle_count_is_102_l1264_126451

/-- Represents a rectangle divided into a 6x2 grid with diagonal lines -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  grid_width : ℕ
  grid_height : ℕ
  has_diagonals : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the specific GridRectangle is 102 -/
theorem triangle_count_is_102 :
  ∃ (rect : GridRectangle),
    rect.width = 6 ∧
    rect.height = 2 ∧
    rect.grid_width = 6 ∧
    rect.grid_height = 2 ∧
    rect.has_diagonals = true ∧
    count_triangles rect = 102 :=
  sorry

end triangle_count_is_102_l1264_126451


namespace consecutive_integers_puzzle_l1264_126471

theorem consecutive_integers_puzzle (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end consecutive_integers_puzzle_l1264_126471


namespace g_expression_and_minimum_l1264_126457

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 3)

noncomputable def N (a : ℝ) : ℝ := 1 - 1/a

noncomputable def g (a : ℝ) : ℝ := M a - N a

theorem g_expression_and_minimum (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ((1/3 ≤ a ∧ a ≤ 1/2 → g a = a - 2 + 1/a) ∧
   (1/2 < a ∧ a ≤ 1 → g a = 9*a - 6 + 1/a)) ∧
  (∀ b, 1/3 ≤ b ∧ b ≤ 1 → g b ≥ 1/2) ∧
  g (1/2) = 1/2 := by sorry

end g_expression_and_minimum_l1264_126457


namespace quadratic_inequality_problem_l1264_126488

theorem quadratic_inequality_problem (a c m : ℝ) :
  (∀ x, ax^2 + x + c > 0 ↔ 1 < x ∧ x < 3) →
  let A := {x | (-1/4)*x^2 + 2*x - 3 > 0}
  let B := {x | x + m > 0}
  A ⊆ B →
  (a = -1/4 ∧ c = -3/4) ∧ m ≥ -2 :=
by sorry

end quadratic_inequality_problem_l1264_126488


namespace profit_percentage_is_fifty_percent_l1264_126484

def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 30000

def total_cost : ℕ := purchase_price + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_fifty_percent :
  (profit : ℚ) / (total_cost : ℚ) * 100 = 50 := by sorry

end profit_percentage_is_fifty_percent_l1264_126484


namespace quadratic_equation_standard_form_l1264_126433

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (2*x - 1)^2 = (x + 1)*(3*x + 4) ↔ x^2 - 11*x - 3 = 0 := by
  sorry

end quadratic_equation_standard_form_l1264_126433


namespace problem_solution_l1264_126462

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_solution :
  (N ⊆ M) ∧
  (∀ a b : ℝ, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) := by
  sorry

end problem_solution_l1264_126462


namespace cos_210_degrees_l1264_126490

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l1264_126490


namespace line_equation_l1264_126458

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

-- Define points A, B, M, and N
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- Define the conditions
def conditions : Prop :=
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  point_A.1 > 0 ∧ point_A.2 > 0 ∧
  point_B.1 > 0 ∧ point_B.2 > 0 ∧
  point_M.2 = 0 ∧
  point_N.1 = 0 ∧
  (point_M.1 - point_A.1)^2 + (point_M.2 - point_A.2)^2 =
    (point_N.1 - point_B.1)^2 + (point_N.2 - point_B.2)^2 ∧
  (point_M.1 - point_N.1)^2 + (point_M.2 - point_N.2)^2 = 12

-- Theorem statement
theorem line_equation (h : conditions) : 
  ∀ x y, line_l x y ↔ (x, y) ∈ {p | ∃ t, p = (1-t) • point_M + t • point_N} :=
sorry

end line_equation_l1264_126458


namespace inequality_theorem_l1264_126482

theorem inequality_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z + Real.sqrt (x^2 + y^2 + z^2))) / 
  ((x^2 + y^2 + z^2) * (y*z + z*x + x*y)) ≤ (3 + Real.sqrt 3) / 9 := by
  sorry

end inequality_theorem_l1264_126482


namespace subtraction_and_simplification_l1264_126453

theorem subtraction_and_simplification :
  (12 : ℚ) / 25 - (3 : ℚ) / 75 = (11 : ℚ) / 25 := by sorry

end subtraction_and_simplification_l1264_126453


namespace subcommittee_formation_count_l1264_126431

def total_republicans : ℕ := 12
def total_democrats : ℕ := 10
def subcommittee_republicans : ℕ := 5
def subcommittee_democrats : ℕ := 4

theorem subcommittee_formation_count :
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 166320 := by
  sorry

end subcommittee_formation_count_l1264_126431


namespace cloth_cost_price_per_meter_l1264_126405

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cloth_cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 66)
  (h2 : total_selling_price = 660)
  (h3 : profit_per_meter = 5) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 5 :=
by sorry

end cloth_cost_price_per_meter_l1264_126405


namespace quadratic_equation_exponent_l1264_126424

/-- Given that 2x^m + (2-m)x - 5 = 0 is a quadratic equation in terms of x, prove that m = 2 -/
theorem quadratic_equation_exponent (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ 2*x^m + (2-m)*x - 5 = a*x^2 + b*x + c) → m = 2 :=
by sorry

end quadratic_equation_exponent_l1264_126424


namespace ps_length_is_sqrt_461_l1264_126461

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  angleQ_is_right : Bool
  angleR_is_right : Bool

/-- The length of PS in the right quadrilateral PQRS -/
def length_PS (quad : RightQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral PQRS with given side lengths and right angles, PS = √461 -/
theorem ps_length_is_sqrt_461 (quad : RightQuadrilateral) 
  (h1 : quad.PQ = 6)
  (h2 : quad.QR = 10)
  (h3 : quad.RS = 25)
  (h4 : quad.angleQ_is_right = true)
  (h5 : quad.angleR_is_right = true) :
  length_PS quad = Real.sqrt 461 :=
by sorry

end ps_length_is_sqrt_461_l1264_126461


namespace no_integer_solution_l1264_126474

theorem no_integer_solution : ¬∃ y : ℤ, (8 : ℝ)^3 + 4^3 + 2^10 = 2^y := by
  sorry

end no_integer_solution_l1264_126474


namespace time_at_15mph_is_3_hours_l1264_126478

/-- Represents the running scenario with three different speeds -/
structure RunningScenario where
  time_at_15mph : ℝ
  time_at_10mph : ℝ
  time_at_8mph : ℝ

/-- The total time of the run is 14 hours -/
def total_time (run : RunningScenario) : ℝ :=
  run.time_at_15mph + run.time_at_10mph + run.time_at_8mph

/-- The total distance covered is 164 miles -/
def total_distance (run : RunningScenario) : ℝ :=
  15 * run.time_at_15mph + 10 * run.time_at_10mph + 8 * run.time_at_8mph

/-- Theorem stating that the time spent running at 15 mph was 3 hours -/
theorem time_at_15mph_is_3_hours :
  ∃ (run : RunningScenario),
    total_time run = 14 ∧
    total_distance run = 164 ∧
    run.time_at_15mph = 3 ∧
    run.time_at_10mph ≥ 0 ∧
    run.time_at_8mph ≥ 0 :=
  sorry

end time_at_15mph_is_3_hours_l1264_126478


namespace sticker_distribution_count_l1264_126495

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 30 := by sorry

end sticker_distribution_count_l1264_126495


namespace course_selection_schemes_l1264_126450

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def min_courses : ℕ := 2
def max_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 +
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_schemes : total_selections = 64 := by
  sorry

end course_selection_schemes_l1264_126450


namespace laundry_time_proof_l1264_126425

/-- Calculates the total time for laundry given the number of loads, time per load for washing, and time for drying. -/
def totalLaundryTime (numLoads : ℕ) (washTimePerLoad : ℕ) (dryTime : ℕ) : ℕ :=
  numLoads * washTimePerLoad + dryTime

/-- Proves that given the specified conditions, the total laundry time is 165 minutes. -/
theorem laundry_time_proof :
  totalLaundryTime 2 45 75 = 165 := by
  sorry

end laundry_time_proof_l1264_126425


namespace arithmetic_perfect_power_sequence_exists_l1264_126409

/-- An arithmetic sequence of perfect powers -/
def ArithmeticPerfectPowerSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ i j : ℕ, i < n ∧ j < n →
    (∃ (base exponent : ℕ), exponent > 1 ∧ a i = base ^ exponent) ∧
    (a j - a i = d * (j - i))

theorem arithmetic_perfect_power_sequence_exists :
  ∃ (a : ℕ → ℕ), ArithmeticPerfectPowerSequence a 2003 :=
sorry

end arithmetic_perfect_power_sequence_exists_l1264_126409


namespace polar_to_cartesian_line_l1264_126416

/-- Given a line l with polar equation θ = 2π/3, its Cartesian coordinate equation is √3x + y = 0 -/
theorem polar_to_cartesian_line (l : Set (ℝ × ℝ)) :
  (∀ (r θ : ℝ), (r, θ) ∈ l ↔ θ = 2 * Real.pi / 3) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ Real.sqrt 3 * x + y = 0) :=
sorry

end polar_to_cartesian_line_l1264_126416


namespace twentieth_fisherman_catch_l1264_126441

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fish_per_nineteen : ℕ) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : fish_per_nineteen = 400) : 
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 :=
by
  sorry

#check twentieth_fisherman_catch

end twentieth_fisherman_catch_l1264_126441


namespace total_spent_is_20_l1264_126496

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 4

/-- The price of a keychain in dollars -/
def keychain_price : ℕ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℕ := 3

/-- The number of bracelets Paula buys -/
def paula_bracelets : ℕ := 2

/-- The number of keychains Paula buys -/
def paula_keychains : ℕ := 1

/-- The number of coloring books Olive buys -/
def olive_coloring_books : ℕ := 1

/-- The number of bracelets Olive buys -/
def olive_bracelets : ℕ := 1

/-- The total amount spent by Paula and Olive -/
def total_spent : ℕ :=
  paula_bracelets * bracelet_price +
  paula_keychains * keychain_price +
  olive_coloring_books * coloring_book_price +
  olive_bracelets * bracelet_price

theorem total_spent_is_20 : total_spent = 20 := by
  sorry

end total_spent_is_20_l1264_126496


namespace intersection_M_N_l1264_126491

-- Define set M
def M : Set ℤ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℤ := {x | ∃ n : ℕ, x = 2 * n}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l1264_126491


namespace eddie_dump_rate_l1264_126465

/-- Given that Sam dumps tea for 6 hours at 60 crates per hour,
    and Eddie takes 4 hours to dump the same amount,
    prove that Eddie's rate is 90 crates per hour. -/
theorem eddie_dump_rate 
  (sam_hours : ℕ) 
  (sam_rate : ℕ) 
  (eddie_hours : ℕ) 
  (h1 : sam_hours = 6)
  (h2 : sam_rate = 60)
  (h3 : eddie_hours = 4)
  (h4 : sam_hours * sam_rate = eddie_hours * eddie_rate) :
  eddie_rate = 90 :=
by
  sorry

#check eddie_dump_rate

end eddie_dump_rate_l1264_126465


namespace color_congruent_triangle_l1264_126475

/-- A type representing the 1992 colors used to color the plane -/
def Color := Fin 1992

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is on the edge of a triangle (excluding vertices) -/
def on_edge (p : Point) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem color_congruent_triangle 
  (coloring : Coloring) 
  (color_exists : ∀ c : Color, ∃ p : Point, coloring p = c) 
  (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    ∀ (e1 e2 : Fin 3), ∃ (p1 p2 : Point) (c : Color), 
      on_edge p1 T' ∧ on_edge p2 T' ∧ 
      coloring p1 = c ∧ coloring p2 = c := by sorry

end color_congruent_triangle_l1264_126475


namespace family_gallery_photos_l1264_126428

/-- Proves that the initial number of photos in the family gallery was 400 --/
theorem family_gallery_photos : 
  ∀ (P : ℕ), 
  (P + (P / 2) + (P / 2 + 120) = 920) → 
  P = 400 := by
sorry

end family_gallery_photos_l1264_126428


namespace f_max_at_neg_two_l1264_126407

/-- The function f(x) = -x^2 - 4x + 16 -/
def f (x : ℝ) : ℝ := -x^2 - 4*x + 16

/-- The statement that f(x) attains its maximum value when x = -2 -/
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
sorry

end f_max_at_neg_two_l1264_126407


namespace inequality_proof_l1264_126469

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end inequality_proof_l1264_126469


namespace non_shaded_perimeter_l1264_126420

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer1 outer2 shaded : Rectangle) 
  (h1 : outer1.width = 12 ∧ outer1.height = 9)
  (h2 : outer2.width = 5 ∧ outer2.height = 3)
  (h3 : shaded.width = 6 ∧ shaded.height = 3)
  (h4 : area outer1 + area outer2 = 117)
  (h5 : area shaded = 108) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 12 := by
  sorry

end non_shaded_perimeter_l1264_126420


namespace infinite_geometric_series_first_term_l1264_126432

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
  sorry

end infinite_geometric_series_first_term_l1264_126432


namespace vertical_shift_l1264_126417

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a constant k for the vertical shift
variable (k : ℝ)

-- Define a point (x, y) on the graph of y = f(x)
variable (x y : ℝ)

-- Theorem: If (x, y) is on the graph of y = f(x), then (x, y + k) is on the graph of y = f(x) + k
theorem vertical_shift (h : y = f x) : (y + k) = (f x + k) := by sorry

end vertical_shift_l1264_126417


namespace divisibility_by_ten_l1264_126459

theorem divisibility_by_ten (a : ℤ) : 
  (10 ∣ (a^10 + 1)) ↔ (a % 10 = 3 ∨ a % 10 = 7 ∨ a % 10 = -3 ∨ a % 10 = -7) := by
  sorry

end divisibility_by_ten_l1264_126459


namespace arrangement_theorem_l1264_126454

/-- The number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 8 people in a row with 3 specific people not adjacent -/
def arrangements_with_non_adjacent (total : ℕ) (non_adjacent : ℕ) : ℕ :=
  permutations (total - non_adjacent + 1) non_adjacent * permutations (total - non_adjacent) (total - non_adjacent)

theorem arrangement_theorem :
  arrangements_with_non_adjacent 8 3 = permutations 6 3 * permutations 5 5 := by
  sorry

end arrangement_theorem_l1264_126454


namespace unique_positive_solution_arctan_equation_l1264_126456

theorem unique_positive_solution_arctan_equation :
  ∃! y : ℝ, y > 0 ∧ Real.arctan (1 / y) + Real.arctan (1 / y^2) = π / 4 :=
by sorry

end unique_positive_solution_arctan_equation_l1264_126456


namespace min_value_expression_l1264_126472

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ Real.sqrt 15 := by
  sorry

end min_value_expression_l1264_126472


namespace solve_linear_equation_l1264_126406

theorem solve_linear_equation (x : ℝ) : 5 * x - 3 = 17 ↔ x = 4 := by sorry

end solve_linear_equation_l1264_126406


namespace tank_dimension_l1264_126470

/-- Proves that the third dimension of a rectangular tank is 2 feet given specific conditions -/
theorem tank_dimension (x : ℝ) : 
  (4 : ℝ) > 0 ∧ 
  (5 : ℝ) > 0 ∧ 
  x > 0 ∧
  (20 : ℝ) > 0 ∧
  1520 = 20 * (2 * (4 * 5) + 2 * (4 * x) + 2 * (5 * x)) →
  x = 2 := by
  sorry

#check tank_dimension

end tank_dimension_l1264_126470


namespace arithmetic_geometric_progression_l1264_126440

theorem arithmetic_geometric_progression (a₁ a₂ a₃ a₄ d : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0)
  (h_d_nonzero : d ≠ 0)
  (h_arithmetic : a₂ = a₁ + d ∧ a₃ = a₁ + 2*d ∧ a₄ = a₁ + 3*d)
  (h_geometric : a₃^2 = a₁ * a₄) :
  d / a₁ = -1/4 := by
sorry

end arithmetic_geometric_progression_l1264_126440


namespace g_13_equals_201_l1264_126492

def g (n : ℕ) : ℕ := n^2 + n + 19

theorem g_13_equals_201 : g 13 = 201 := by sorry

end g_13_equals_201_l1264_126492


namespace equation_solution_l1264_126463

theorem equation_solution :
  ∃ x : ℝ, (3 / 4 + 1 / x = 7 / 8) ∧ (x = 8) := by
  sorry

end equation_solution_l1264_126463


namespace reinforcement_size_l1264_126483

/-- Calculates the size of a reinforcement given initial garrison size, provision duration, and new provision duration after reinforcement arrival. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (new_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  (remaining_provisions / new_duration) - initial_garrison

/-- Proves that the reinforcement size is 1900 given the problem conditions. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 54 15 20 = 1900 := by
  sorry

end reinforcement_size_l1264_126483


namespace irrational_shift_exists_rational_shift_not_exists_l1264_126423

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem irrational_shift_exists :
  ∃ (α : ℝ), ∀ (i : Fin n), ¬(∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

theorem rational_shift_not_exists :
  ¬(∃ (α : ℝ), ∀ (i : Fin n), ∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

end irrational_shift_exists_rational_shift_not_exists_l1264_126423


namespace minimum_bailing_rate_for_steve_and_leroy_l1264_126448

/-- Represents the fishing scenario with Steve and LeRoy --/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  boat_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach shore without sinking --/
def minimum_bailing_rate (scenario : FishingScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum bailing rate for the given scenario --/
theorem minimum_bailing_rate_for_steve_and_leroy :
  let scenario : FishingScenario := {
    distance_to_shore := 2,
    water_intake_rate := 12,
    boat_capacity := 40,
    rowing_speed := 3
  }
  minimum_bailing_rate scenario = 11 := by sorry

end minimum_bailing_rate_for_steve_and_leroy_l1264_126448
