import Mathlib

namespace panthers_second_half_score_l3566_356659

theorem panthers_second_half_score 
  (total_first_half : ℕ)
  (cougars_lead_first_half : ℕ)
  (total_game : ℕ)
  (cougars_lead_total : ℕ)
  (h1 : total_first_half = 38)
  (h2 : cougars_lead_first_half = 16)
  (h3 : total_game = 58)
  (h4 : cougars_lead_total = 22) :
  ∃ (cougars_first cougars_second panthers_first panthers_second : ℕ),
    cougars_first + panthers_first = total_first_half ∧
    cougars_first = panthers_first + cougars_lead_first_half ∧
    cougars_first + cougars_second + panthers_first + panthers_second = total_game ∧
    (cougars_first + cougars_second) - (panthers_first + panthers_second) = cougars_lead_total ∧
    panthers_second = 7 :=
by sorry

end panthers_second_half_score_l3566_356659


namespace sum_of_absolute_coefficients_l3566_356627

/-- Given that (2x-1)^5 + (x+2)^4 = a + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove that |a| + |a₂| + |a₄| = 30 -/
theorem sum_of_absolute_coefficients (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (2*x - 1)^5 + (x + 2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  |a| + |a₂| + |a₄| = 30 := by
  sorry

end sum_of_absolute_coefficients_l3566_356627


namespace cubic_km_to_m_strip_l3566_356605

/-- The length of a strip formed by cutting a cubic kilometer into cubic meters and laying them out in a single line -/
def strip_length : ℝ := 1000000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem cubic_km_to_m_strip : 
  strip_length = (km_to_m ^ 3) / km_to_m := by sorry

end cubic_km_to_m_strip_l3566_356605


namespace negative_three_cubed_equality_l3566_356672

theorem negative_three_cubed_equality : (-3)^3 = -3^3 := by sorry

end negative_three_cubed_equality_l3566_356672


namespace function_composition_equality_l3566_356603

/-- Given two functions f and g, where f(x) = Ax³ - B and g(x) = Bx², 
    with B ≠ 0 and f(g(2)) = 0, prove that A = 1 / (64B²) -/
theorem function_composition_equality (A B : ℝ) 
  (hB : B ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x^3 - B)
  (hg : ∀ x, g x = B * x^2)
  (h_comp : f (g 2) = 0) :
  A = 1 / (64 * B^2) := by
sorry

end function_composition_equality_l3566_356603


namespace f_value_at_2_l3566_356667

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) (h2 : f (-2) = 10) :
  f 2 = -26 := by sorry

end f_value_at_2_l3566_356667


namespace shekars_english_score_l3566_356656

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def biology_score : ℕ := 95
def average_score : ℕ := 77
def total_subjects : ℕ := 5

theorem shekars_english_score :
  let known_scores_sum := math_score + science_score + social_studies_score + biology_score
  let total_score := average_score * total_subjects
  total_score - known_scores_sum = 67 := by
  sorry

end shekars_english_score_l3566_356656


namespace white_balls_count_l3566_356676

theorem white_balls_count (a b c : ℕ) : 
  a + b + c = 20 → -- Total number of balls
  (a : ℚ) / (20 + b) = a / 20 - 1 / 25 → -- Probability change when doubling blue balls
  b / (20 - a) = b / 20 + 1 / 16 → -- Probability change when removing white balls
  a = 4 := by
  sorry

end white_balls_count_l3566_356676


namespace percentage_decrease_in_hours_l3566_356620

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears_without_assistant : ℝ
  hours_without_assistant : ℝ
  bears_with_assistant : ℝ
  hours_with_assistant : ℝ

/-- The conditions of Jane's toy bear production --/
def production_conditions (p : BearProduction) : Prop :=
  p.bears_with_assistant = 1.8 * p.bears_without_assistant ∧
  (p.bears_with_assistant / p.hours_with_assistant) = 2 * (p.bears_without_assistant / p.hours_without_assistant)

/-- The theorem stating the percentage decrease in hours worked --/
theorem percentage_decrease_in_hours (p : BearProduction) 
  (h : production_conditions p) : 
  (p.hours_without_assistant - p.hours_with_assistant) / p.hours_without_assistant * 100 = 10 := by
  sorry


end percentage_decrease_in_hours_l3566_356620


namespace galyas_number_puzzle_l3566_356677

theorem galyas_number_puzzle (N : ℕ) : (∀ k : ℝ, ((k * N + N) / N - N = k - 2021)) ↔ N = 2022 := by sorry

end galyas_number_puzzle_l3566_356677


namespace isosceles_right_triangle_inscribed_circle_theorem_l3566_356644

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangleWithCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The height from the right angle to the hypotenuse -/
  h : ℝ
  /-- The height is twice the radius -/
  height_radius_relation : h = 2 * r
  /-- The radius is √2/4 -/
  radius_value : r = Real.sqrt 2 / 4

/-- The theorem to be proved -/
theorem isosceles_right_triangle_inscribed_circle_theorem 
  (triangle : IsoscelesRightTriangleWithCircle) : 
  triangle.h - triangle.r = Real.sqrt 2 / 4 := by
  sorry


end isosceles_right_triangle_inscribed_circle_theorem_l3566_356644


namespace expression_value_l3566_356666

theorem expression_value : 
  let a : ℚ := 1/3
  let b : ℚ := 3
  (2 * a⁻¹ + a⁻¹ / b) / a = 21 := by sorry

end expression_value_l3566_356666


namespace trigonometric_equation_solution_l3566_356606

theorem trigonometric_equation_solution (x : ℝ) 
  (h_eq : 8.459 * (Real.cos (x^2))^2 * (Real.tan (x^2) + 2 * Real.tan x) + 
          (Real.tan x)^3 * (1 - (Real.sin (x^2))^2) * (2 - Real.tan x * Real.tan (x^2)) = 0)
  (h_cos : Real.cos x ≠ 0)
  (h_x_sq : ∀ n : ℤ, x^2 ≠ Real.pi/2 + Real.pi * n)
  (h_x_1 : ∀ m : ℤ, x ≠ Real.pi/4 + Real.pi * m/2)
  (h_x_2 : ∀ l : ℤ, x ≠ Real.pi/2 + Real.pi * l) :
  ∃ k : ℕ, x = -1 + Real.sqrt (Real.pi * k + 1) ∨ x = -1 - Real.sqrt (Real.pi * k + 1) :=
sorry

end trigonometric_equation_solution_l3566_356606


namespace no_real_roots_for_nonzero_k_l3566_356699

theorem no_real_roots_for_nonzero_k (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x^2 + k*x + 2*k^2 ≠ 0 := by
sorry

end no_real_roots_for_nonzero_k_l3566_356699


namespace candy_spending_l3566_356663

/-- The fraction of a dollar spent on candy given initial quarters and remaining cents -/
def fraction_spent (initial_quarters : ℕ) (remaining_cents : ℕ) : ℚ :=
  (initial_quarters * 25 - remaining_cents) / 100

/-- Theorem stating that given 14 quarters initially and 300 cents remaining,
    the fraction of a dollar spent on candy is 1/2 -/
theorem candy_spending :
  fraction_spent 14 300 = 1/2 := by
  sorry

end candy_spending_l3566_356663


namespace coefficient_x4_in_binomial_expansion_l3566_356635

theorem coefficient_x4_in_binomial_expansion :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end coefficient_x4_in_binomial_expansion_l3566_356635


namespace second_journey_half_time_l3566_356658

/-- Represents a journey with distance and speed -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Theorem stating that under given conditions, the time of the second journey is half of the first -/
theorem second_journey_half_time (j1 j2 : Journey) 
  (h1 : j1.distance = 80)
  (h2 : j2.distance = 160)
  (h3 : j2.speed = 4 * j1.speed) :
  (j2.distance / j2.speed) = (1/2) * (j1.distance / j1.speed) := by
  sorry

#check second_journey_half_time

end second_journey_half_time_l3566_356658


namespace consecutive_integer_product_divisibility_l3566_356600

theorem consecutive_integer_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 11 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 22 * m) ∧
  (∃ m : ℤ, n = 33 * m) ∧
  (∃ m : ℤ, n = 66 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 44 * m) :=
by sorry

end consecutive_integer_product_divisibility_l3566_356600


namespace fraction_sum_equals_one_l3566_356624

theorem fraction_sum_equals_one : 3/5 - 1/10 + 1/2 = 1 := by
  sorry

end fraction_sum_equals_one_l3566_356624


namespace rotten_eggs_probability_l3566_356607

/-- The probability of selecting 2 rotten eggs from a pack of 36 eggs containing 3 rotten eggs -/
theorem rotten_eggs_probability (total_eggs : ℕ) (rotten_eggs : ℕ) (selected_eggs : ℕ) : 
  total_eggs = 36 → rotten_eggs = 3 → selected_eggs = 2 →
  (Nat.choose rotten_eggs selected_eggs : ℚ) / (Nat.choose total_eggs selected_eggs) = 1 / 420 :=
by sorry

end rotten_eggs_probability_l3566_356607


namespace oliver_vowel_learning_days_l3566_356647

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of days Oliver needs to finish learning all vowels -/
def days_to_learn_vowels : ℕ := num_vowels * days_per_alphabet

/-- Theorem: Oliver needs 25 days to finish learning all vowels -/
theorem oliver_vowel_learning_days : days_to_learn_vowels = 25 := by
  sorry

end oliver_vowel_learning_days_l3566_356647


namespace class_size_l3566_356661

theorem class_size (n : ℕ) 
  (h1 : 30 * 160 + (n - 30) * 156 = n * 159) : n = 40 := by
  sorry

#check class_size

end class_size_l3566_356661


namespace simplify_expression_l3566_356696

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l3566_356696


namespace starting_number_sequence_l3566_356602

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79 ∧ 
   (∃ (a b c d : ℕ), n < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    n % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0) ∧
   (∀ m : ℕ, m < n → ¬(∃ (a b c d : ℕ), m < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    m % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0))) →
  n = 33 := by
sorry

end starting_number_sequence_l3566_356602


namespace teena_speed_calculation_l3566_356653

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Loe's speed in miles per hour -/
def loe_speed : ℝ := 40

/-- Initial distance Teena is behind Loe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time after which Teena is ahead of Loe in hours -/
def time_elapsed : ℝ := 1.5

/-- Distance Teena is ahead of Loe after time_elapsed in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    initial_distance_behind + final_distance_ahead + (loe_speed * time_elapsed) := by
  sorry

end teena_speed_calculation_l3566_356653


namespace arithmetic_square_root_of_ten_l3566_356611

theorem arithmetic_square_root_of_ten : Real.sqrt 10 = Real.sqrt 10 := by
  sorry

end arithmetic_square_root_of_ten_l3566_356611


namespace polar_midpoint_specific_case_l3566_356643

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/3) and (10, 2π/3) in polar coordinates is (5√3, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (2*π/3)
  r = 5 * Real.sqrt 3 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end polar_midpoint_specific_case_l3566_356643


namespace triangle_third_side_l3566_356610

theorem triangle_third_side (a b c : ℝ) : 
  a = 1 → b = 5 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c = 5 :=
by sorry

end triangle_third_side_l3566_356610


namespace leadership_structure_count_is_correct_l3566_356623

def tribe_size : ℕ := 15
def num_kings : ℕ := 1
def num_knights : ℕ := 2
def squires_per_knight : ℕ := 3

def leadership_structure_count : ℕ :=
  tribe_size * (tribe_size - 1).choose num_knights *
  (tribe_size - num_kings - num_knights).choose squires_per_knight *
  (tribe_size - num_kings - num_knights - squires_per_knight).choose squires_per_knight

theorem leadership_structure_count_is_correct :
  leadership_structure_count = 27392400 := by sorry

end leadership_structure_count_is_correct_l3566_356623


namespace rectangle_division_perimeter_l3566_356670

theorem rectangle_division_perimeter (a b x y : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < x) ∧ (x < a) ∧ (0 < y) ∧ (y < b) →
  (∃ (k₁ k₂ k₃ : ℤ),
    2 * (x + y) = k₁ ∧
    2 * (x + b - y) = k₂ ∧
    2 * (a - x + y) = k₃) →
  ∃ (k₄ : ℤ), 2 * (a - x + b - y) = k₄ :=
by sorry

end rectangle_division_perimeter_l3566_356670


namespace simultaneous_integers_l3566_356640

theorem simultaneous_integers (x : ℤ) :
  (∃ y z u : ℤ, (x - 3) = 7 * y ∧ (x - 2) = 5 * z ∧ (x - 4) = 3 * u) ↔
  (∃ t : ℤ, x = 105 * t + 52) :=
by sorry

end simultaneous_integers_l3566_356640


namespace cube_surface_area_from_volume_l3566_356613

theorem cube_surface_area_from_volume : 
  ∀ (v : ℝ) (s : ℝ), 
  v = 729 →  -- Given volume
  v = s^3 →  -- Volume formula
  6 * s^2 = 486 -- Surface area formula and result
  := by sorry

end cube_surface_area_from_volume_l3566_356613


namespace square_d_perimeter_l3566_356650

def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

def square_area (side_length : ℝ) : ℝ := side_length ^ 2

theorem square_d_perimeter (perimeter_c : ℝ) (h1 : perimeter_c = 32) :
  let side_c := perimeter_c / 4
  let area_c := square_area side_c
  let area_d := area_c / 3
  let side_d := Real.sqrt area_d
  square_perimeter side_d = (32 * Real.sqrt 3) / 3 := by
sorry

end square_d_perimeter_l3566_356650


namespace angle_measure_proof_l3566_356622

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x - 2) = 90) → x = 23 := by
  sorry

end angle_measure_proof_l3566_356622


namespace zoo_guides_theorem_l3566_356669

/-- The total number of children addressed by zoo guides --/
def total_children (total_guides : ℕ) 
                   (english_guides : ℕ) 
                   (french_guides : ℕ) 
                   (english_children : ℕ) 
                   (french_children : ℕ) 
                   (spanish_children : ℕ) : ℕ :=
  let spanish_guides := total_guides - english_guides - french_guides
  english_guides * english_children + 
  french_guides * french_children + 
  spanish_guides * spanish_children

/-- Theorem stating the total number of children addressed by zoo guides --/
theorem zoo_guides_theorem : 
  total_children 22 10 6 19 25 30 = 520 := by
  sorry

end zoo_guides_theorem_l3566_356669


namespace projectiles_meeting_time_l3566_356638

/-- Theorem: Time for two projectiles to meet --/
theorem projectiles_meeting_time
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : distance = 2520)
  (h2 : speed1 = 432)
  (h3 : speed2 = 576) :
  (distance / (speed1 + speed2)) * 60 = 150 :=
by sorry

end projectiles_meeting_time_l3566_356638


namespace convergence_of_iterative_process_l3566_356682

theorem convergence_of_iterative_process (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, 2^(-k : ℤ) * (a - b) < (1 : ℝ) / 2002 := by
  sorry

end convergence_of_iterative_process_l3566_356682


namespace four_pockets_sixteen_coins_l3566_356681

/-- The total number of coins in multiple pockets -/
def total_coins (num_pockets : ℕ) (coins_per_pocket : ℕ) : ℕ :=
  num_pockets * coins_per_pocket

/-- Theorem: Given 4 pockets with 16 coins each, the total number of coins is 64 -/
theorem four_pockets_sixteen_coins : total_coins 4 16 = 64 := by
  sorry

end four_pockets_sixteen_coins_l3566_356681


namespace solve_wardrobe_problem_l3566_356616

def wardrobe_problem (socks shoes tshirts new_socks : ℕ) : Prop :=
  ∃ pants : ℕ,
    let current_items := 2 * socks + 2 * shoes + tshirts + pants
    current_items + 2 * new_socks = 2 * current_items ∧
    pants = 5

theorem solve_wardrobe_problem :
  wardrobe_problem 20 5 10 35 :=
by
  sorry

end solve_wardrobe_problem_l3566_356616


namespace apple_distribution_l3566_356619

theorem apple_distribution (x y : ℕ) : 
  y = 5 * x + 12 ∧ 0 < 8 * x - y ∧ 8 * x - y < 8 → 
  (x = 5 ∧ y = 37) ∨ (x = 6 ∧ y = 42) := by
  sorry

end apple_distribution_l3566_356619


namespace sarah_investment_l3566_356668

/-- Proves that given a total investment of $250,000 and the investment in real estate
    being 6 times the investment in mutual funds, the amount invested in real estate
    is $214,285.71. -/
theorem sarah_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
    (h1 : total = 250000)
    (h2 : real_estate = 6 * mutual_funds)
    (h3 : total = real_estate + mutual_funds) :
  real_estate = 214285.71 := by
  sorry

end sarah_investment_l3566_356668


namespace square_root_range_l3566_356690

theorem square_root_range (x : ℝ) : ∃ y : ℝ, y = Real.sqrt (x - 5) ↔ x ≥ 5 := by sorry

end square_root_range_l3566_356690


namespace hotel_rooms_booked_l3566_356665

theorem hotel_rooms_booked (single_room_cost double_room_cost total_revenue double_rooms : ℕ)
  (h1 : single_room_cost = 35)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196)
  : ∃ single_rooms : ℕ, single_rooms + double_rooms = 260 ∧ 
    single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue := by
  sorry

end hotel_rooms_booked_l3566_356665


namespace second_group_size_l3566_356680

/-- Represents the number of man-days required to complete the work -/
def totalManDays : ℕ := 18 * 20

/-- Proves that 12 men can complete the work in 30 days, given that 18 men can complete it in 20 days -/
theorem second_group_size (days : ℕ) (h : days = 30) : 
  (totalManDays / days : ℕ) = 12 := by
  sorry

#check second_group_size

end second_group_size_l3566_356680


namespace unique_solution_system_l3566_356655

theorem unique_solution_system : 
  ∃! (x y : ℕ+), (x : ℝ)^(y : ℝ) + 3 = (y : ℝ)^(x : ℝ) + 1 ∧ 
                 2 * (x : ℝ)^(y : ℝ) + 4 = (y : ℝ)^(x : ℝ) + 9 ∧
                 x = 3 ∧ y = 1 := by
  sorry

end unique_solution_system_l3566_356655


namespace polynomial_coefficient_sum_l3566_356645

theorem polynomial_coefficient_sum : 
  ∀ (A B C D E : ℝ), 
  (∀ x : ℝ, (2*x + 3)*(4*x^3 - 2*x^2 + x - 7) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = -20 := by
sorry

end polynomial_coefficient_sum_l3566_356645


namespace pencil_difference_l3566_356695

theorem pencil_difference (price : ℚ) (liam_count mia_count : ℕ) : 
  price > 0.01 →
  price * liam_count = 2.10 →
  price * mia_count = 2.82 →
  mia_count - liam_count = 12 := by
sorry

end pencil_difference_l3566_356695


namespace function_local_max_condition_l3566_356626

/-- Given a real constant a, prove that for a function f(x) = (x-a)²(x+b)e^x 
    where b is real and x=a is a local maximum point of f(x), 
    then b must be less than -a. -/
theorem function_local_max_condition (a : ℝ) :
  ∀ b : ℝ, (∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = (x - a)^2 * (x + b) * Real.exp x) ∧
    (IsLocalMax f a)) →
  b < -a :=
by sorry

end function_local_max_condition_l3566_356626


namespace largest_perfect_square_factor_1800_l3566_356642

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_1800 :
  largest_perfect_square_factor 1800 = 900 := by
  sorry

end largest_perfect_square_factor_1800_l3566_356642


namespace quad_pair_f_one_l3566_356641

/-- Two quadratic polynomials satisfying specific conditions -/
structure QuadraticPair :=
  (f g : ℝ → ℝ)
  (quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c)
  (h1 : f 2 = 2 ∧ f 3 = 2)
  (h2 : g 2 = 2 ∧ g 3 = 2)
  (h3 : g 1 = 3)
  (h4 : f 4 = 7)
  (h5 : g 4 = 4)

/-- The main theorem stating that f(1) = 7 for the given conditions -/
theorem quad_pair_f_one (qp : QuadraticPair) : qp.f 1 = 7 := by
  sorry

end quad_pair_f_one_l3566_356641


namespace paving_cost_l3566_356651

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end paving_cost_l3566_356651


namespace rudolph_travel_distance_l3566_356662

/-- Represents the number of stop signs Rudolph encountered -/
def total_stop_signs : ℕ := 17 - 3

/-- Represents the number of stop signs per mile -/
def stop_signs_per_mile : ℕ := 2

/-- Calculates the number of miles Rudolph traveled -/
def miles_traveled : ℚ := total_stop_signs / stop_signs_per_mile

theorem rudolph_travel_distance :
  miles_traveled = 7 := by sorry

end rudolph_travel_distance_l3566_356662


namespace circular_seating_arrangement_l3566_356630

theorem circular_seating_arrangement (n : ℕ) (h1 : n ≤ 6) (h2 : Nat.factorial (n - 1) = 144) : n = 6 := by
  sorry

end circular_seating_arrangement_l3566_356630


namespace xyz_value_l3566_356693

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 26 / 3 := by
sorry

end xyz_value_l3566_356693


namespace shopping_remainder_l3566_356632

theorem shopping_remainder (initial_amount : ℝ) (grocery_fraction : ℝ) (household_fraction : ℝ) (personal_care_fraction : ℝ) 
  (h1 : initial_amount = 450)
  (h2 : grocery_fraction = 3/5)
  (h3 : household_fraction = 1/6)
  (h4 : personal_care_fraction = 1/10) : 
  initial_amount - (grocery_fraction * initial_amount + household_fraction * initial_amount + personal_care_fraction * initial_amount) = 60 := by
  sorry

end shopping_remainder_l3566_356632


namespace hyperbola_equation_l3566_356615

-- Define the hyperbola
def Hyperbola (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- State the theorem
theorem hyperbola_equation :
  ∀ (a b c : ℝ),
    -- Conditions
    (2 * a = 8) →  -- Distance between vertices
    (c / a = 5 / 4) →  -- Eccentricity
    (c^2 = a^2 + b^2) →  -- Relation between a, b, and c
    -- Conclusion
    Hyperbola a b c = Hyperbola 4 3 5 := by
  sorry

end hyperbola_equation_l3566_356615


namespace triangle_altitude_and_median_l3566_356652

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : (ℝ × ℝ) := (4, 0)
  B : (ℝ × ℝ) := (6, 7)
  C : (ℝ × ℝ) := (0, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromB (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -12 }

def Triangle.medianFromB (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (t.altitudeFromB = { a := 3, b := 2, c := -12 }) ∧
  (t.medianFromB = { a := 5, b := 1, c := -20 }) := by
  sorry

end triangle_altitude_and_median_l3566_356652


namespace product_of_half_and_two_thirds_l3566_356675

theorem product_of_half_and_two_thirds (x y : ℚ) : 
  x = 1/2 → y = 2/3 → x * y = 1/3 := by
  sorry

end product_of_half_and_two_thirds_l3566_356675


namespace problem_1_problem_2_problem_3_problem_4_l3566_356692

-- Problem 1
theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

-- Problem 2
theorem problem_2 : (-3/4)^2 * (-8 + 1/3) = -69/16 := by sorry

-- Problem 3
theorem problem_3 : 16 / (-1/2) * 3/8 - |(-45)| / 9 = -17 := by sorry

-- Problem 4
theorem problem_4 : -1^2024 - (2 - 0.75) * 2/7 * (4 - (-5)^2) = 13/2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l3566_356692


namespace white_triangle_coincidence_l3566_356689

/-- Represents the number of triangles of each color in each half of the diagram -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the diagram is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- Calculates the number of coinciding white triangle pairs given the initial counts and other coinciding pairs -/
def coinciding_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  counts.white - (counts.red - 2 * pairs.red_red - pairs.red_blue) - (counts.blue - 2 * pairs.blue_blue - pairs.red_blue)

/-- Theorem stating that under the given conditions, 6 pairs of white triangles exactly coincide -/
theorem white_triangle_coincidence (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 5 ∧ counts.blue = 4 ∧ counts.white = 7 ∧ 
  pairs.red_red = 3 ∧ pairs.blue_blue = 2 ∧ pairs.red_blue = 1 →
  coinciding_white_pairs counts pairs = 6 := by
  sorry

end white_triangle_coincidence_l3566_356689


namespace simplify_square_roots_l3566_356612

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 1457 / 500 := by
  sorry

end simplify_square_roots_l3566_356612


namespace prob_sum_leq_8_is_13_18_l3566_356664

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum ≤ 8) when rolling two dice -/
def favorable_outcomes : ℕ := 26

/-- The probability of the sum being less than or equal to 8 when two dice are tossed -/
def prob_sum_leq_8 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_leq_8_is_13_18 : prob_sum_leq_8 = 13 / 18 := by
  sorry

end prob_sum_leq_8_is_13_18_l3566_356664


namespace probability_of_event_D_is_one_l3566_356601

theorem probability_of_event_D_is_one :
  ∀ x : ℝ,
  (∃ (P_N P_D_given_N P_D : ℝ),
    P_N = 3/8 ∧
    P_D_given_N = x^2 ∧
    P_D = 5/8 + (3/8) * x^2 ∧
    0 ≤ P_N ∧ P_N ≤ 1 ∧
    0 ≤ P_D_given_N ∧ P_D_given_N ≤ 1 ∧
    0 ≤ P_D ∧ P_D ≤ 1) →
  P_D = 1 :=
sorry

end probability_of_event_D_is_one_l3566_356601


namespace max_teams_in_tournament_l3566_356687

/-- Represents a chess tournament with teams of 3 players each --/
structure ChessTournament where
  numTeams : ℕ
  maxGames : ℕ := 250

/-- Calculate the total number of games in the tournament --/
def totalGames (t : ChessTournament) : ℕ :=
  (9 * t.numTeams * (t.numTeams - 1)) / 2

/-- Theorem stating the maximum number of teams in the tournament --/
theorem max_teams_in_tournament (t : ChessTournament) :
  (∀ n : ℕ, n ≤ t.numTeams → totalGames { numTeams := n, maxGames := t.maxGames } ≤ t.maxGames) →
  t.numTeams ≤ 7 :=
sorry

end max_teams_in_tournament_l3566_356687


namespace f_divisibility_l3566_356686

/-- Sequence a defined recursively -/
def a (r s : ℕ) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product of first n terms of sequence a -/
def f (r s : ℕ) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem -/
theorem f_divisibility (r s n k : ℕ) (hr : r > 0) (hs : s > 0) (hk : k > 0) (hnk : n > k) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end f_divisibility_l3566_356686


namespace blocks_added_l3566_356684

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) :
  final_blocks - initial_blocks = 30 := by
sorry

end blocks_added_l3566_356684


namespace residue_products_l3566_356671

theorem residue_products (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∀ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') →
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∃ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') ∧
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) = (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) :=
by sorry

end residue_products_l3566_356671


namespace prob_second_red_given_first_red_is_half_l3566_356678

/-- Represents the probability of drawing a red ball as the second draw, given that the first draw was red, from a box containing red and white balls. -/
def probability_second_red_given_first_red (total_red : ℕ) (total_white : ℕ) : ℚ :=
  if total_red > 0 then
    (total_red - 1 : ℚ) / (total_red + total_white - 1 : ℚ)
  else
    0

/-- Theorem stating that in a box with 4 red balls and 3 white balls, 
    if two balls are drawn without replacement and the first ball is red, 
    the probability that the second ball is also red is 1/2. -/
theorem prob_second_red_given_first_red_is_half :
  probability_second_red_given_first_red 4 3 = 1/2 := by
  sorry

end prob_second_red_given_first_red_is_half_l3566_356678


namespace line_transformation_l3566_356673

open Matrix

-- Define the rotation matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

-- Define the scaling matrix N
def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

-- Define the combined transformation matrix NM
def NM : Matrix (Fin 2) (Fin 2) ℝ := N * M

theorem line_transformation (x y : ℝ) :
  (NM.mulVec ![x, y] = ![x, x]) ↔ (3 * x + 2 * y = 0) := by sorry

end line_transformation_l3566_356673


namespace range_of_x_l3566_356636

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end range_of_x_l3566_356636


namespace oregon_migration_l3566_356646

/-- The number of people moving to Oregon -/
def people_moving : ℕ := 3500

/-- The number of days over which people are moving -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ := people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem oregon_migration :
  round_to_nearest average_per_hour = 29 := by
  sorry

end oregon_migration_l3566_356646


namespace combination_equality_l3566_356688

theorem combination_equality (a : ℕ) : 
  (Nat.choose 17 (2*a - 1) + Nat.choose 17 (2*a) = Nat.choose 18 12) → 
  (a = 3 ∨ a = 6) := by
  sorry

end combination_equality_l3566_356688


namespace field_dimension_l3566_356679

/-- The value of m for a rectangular field with given dimensions and area -/
theorem field_dimension (m : ℝ) : (3*m + 11) * (m - 3) = 80 → m = 6 := by
  sorry

end field_dimension_l3566_356679


namespace jenny_recycling_l3566_356637

theorem jenny_recycling (total_weight : ℕ) (can_weight : ℕ) (num_cans : ℕ)
  (bottle_price : ℕ) (can_price : ℕ) (total_earnings : ℕ) :
  total_weight = 100 →
  can_weight = 2 →
  num_cans = 20 →
  bottle_price = 10 →
  can_price = 3 →
  total_earnings = 160 →
  ∃ (bottle_weight : ℕ), 
    bottle_weight = 6 ∧
    bottle_weight * ((total_weight - (can_weight * num_cans)) / bottle_weight) = 
      total_weight - (can_weight * num_cans) ∧
    bottle_price * ((total_weight - (can_weight * num_cans)) / bottle_weight) + 
      can_price * num_cans = total_earnings :=
by sorry

end jenny_recycling_l3566_356637


namespace first_pumpkin_weight_l3566_356697

/-- The weight of the first pumpkin given the total weight of two pumpkins and the weight of the second pumpkin -/
theorem first_pumpkin_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 12.7)
  (h2 : second_weight = 8.7) : 
  total_weight - second_weight = 4 := by
  sorry

end first_pumpkin_weight_l3566_356697


namespace geometric_sequence_sixth_term_l3566_356698

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_roots : a 3 * a 7 = 256)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end geometric_sequence_sixth_term_l3566_356698


namespace arithmetic_mean_of_fractions_l3566_356614

theorem arithmetic_mean_of_fractions : 
  let a := (7 : ℚ) / 10
  let b := (4 : ℚ) / 5
  let c := (3 : ℚ) / 4
  c = (a + b) / 2 := by sorry

end arithmetic_mean_of_fractions_l3566_356614


namespace total_amount_proof_l3566_356631

/-- Proves that the total amount is $93,750 given the spending conditions -/
theorem total_amount_proof (raw_materials : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : cash_percentage = 0.20)
  (h4 : ∃ total : ℝ, total = raw_materials + machinery + cash_percentage * total) :
  ∃ total : ℝ, total = 93750 := by
  sorry

end total_amount_proof_l3566_356631


namespace third_vertex_x_coord_l3566_356604

/-- An equilateral triangle with two vertices at (5, 0) and (5, 8) -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ := (5, 0)
  v2 : ℝ × ℝ := (5, 8)
  v3 : ℝ × ℝ
  equilateral : sorry
  v3_in_first_quadrant : v3.1 > 0 ∧ v3.2 > 0

/-- The x-coordinate of the third vertex is 5 + 4√3 -/
theorem third_vertex_x_coord (t : EquilateralTriangle) : t.v3.1 = 5 + 4 * Real.sqrt 3 := by
  sorry

end third_vertex_x_coord_l3566_356604


namespace karls_total_income_l3566_356628

/-- Represents the prices of items in Karl's store -/
structure Prices where
  tshirt : ℚ
  pants : ℚ
  skirt : ℚ
  refurbished_tshirt : ℚ

/-- Represents the quantities of items sold -/
structure QuantitiesSold where
  tshirt : ℕ
  pants : ℕ
  skirt : ℕ
  refurbished_tshirt : ℕ

/-- Calculates the total income given prices and quantities sold -/
def totalIncome (prices : Prices) (quantities : QuantitiesSold) : ℚ :=
  prices.tshirt * quantities.tshirt +
  prices.pants * quantities.pants +
  prices.skirt * quantities.skirt +
  prices.refurbished_tshirt * quantities.refurbished_tshirt

/-- Theorem stating that Karl's total income is $53 -/
theorem karls_total_income :
  let prices : Prices := {
    tshirt := 5,
    pants := 4,
    skirt := 6,
    refurbished_tshirt := 5/2
  }
  let quantities : QuantitiesSold := {
    tshirt := 2,
    pants := 1,
    skirt := 4,
    refurbished_tshirt := 6
  }
  totalIncome prices quantities = 53 := by
  sorry


end karls_total_income_l3566_356628


namespace jack_bike_percentage_l3566_356639

def original_paycheck : ℝ := 125
def tax_rate : ℝ := 0.20
def savings_amount : ℝ := 20

theorem jack_bike_percentage :
  let after_tax := original_paycheck * (1 - tax_rate)
  let remaining := after_tax - savings_amount
  let bike_percentage := (remaining / after_tax) * 100
  bike_percentage = 80 := by sorry

end jack_bike_percentage_l3566_356639


namespace alice_probability_after_three_turns_l3566_356685

/-- Represents the probability of Alice having the ball after three turns in the baseball game. -/
def aliceProbabilityAfterThreeTurns : ℚ :=
  let aliceKeepProb : ℚ := 1/2
  let aliceTossProb : ℚ := 1/2
  let bobTossProb : ℚ := 3/5
  let bobKeepProb : ℚ := 2/5
  
  -- Alice passes to Bob, Bob passes to Alice, Alice keeps
  let seq1 : ℚ := aliceTossProb * bobTossProb * aliceKeepProb
  -- Alice passes to Bob, Bob passes to Alice, Alice passes to Bob
  let seq2 : ℚ := aliceTossProb * bobTossProb * aliceTossProb
  -- Alice keeps, Alice keeps, Alice keeps
  let seq3 : ℚ := aliceKeepProb * aliceKeepProb * aliceKeepProb
  -- Alice keeps, Alice passes to Bob, Bob passes to Alice
  let seq4 : ℚ := aliceKeepProb * aliceTossProb * bobTossProb
  
  seq1 + seq2 + seq3 + seq4

/-- Theorem stating that the probability of Alice having the ball after three turns is 23/40. -/
theorem alice_probability_after_three_turns :
  aliceProbabilityAfterThreeTurns = 23/40 := by
  sorry

end alice_probability_after_three_turns_l3566_356685


namespace stratified_sampling_total_l3566_356609

theorem stratified_sampling_total (senior junior freshman sampled_freshman : ℕ) 
  (h1 : senior = 1000)
  (h2 : junior = 1200)
  (h3 : freshman = 1500)
  (h4 : sampled_freshman = 75) :
  (senior + junior + freshman) * sampled_freshman / freshman = 185 := by
  sorry

end stratified_sampling_total_l3566_356609


namespace library_books_before_grant_l3566_356649

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant : books_before = 5935 := by
  sorry

end library_books_before_grant_l3566_356649


namespace adams_age_l3566_356621

theorem adams_age (adam_age eve_age : ℕ) : 
  adam_age = eve_age - 5 →
  eve_age + 1 = 3 * (adam_age - 4) →
  adam_age = 9 := by
sorry

end adams_age_l3566_356621


namespace cylinder_surface_area_ratio_l3566_356618

theorem cylinder_surface_area_ratio (a : ℝ) (h : a > 0) :
  let r := a / (2 * Real.pi)
  let side_area := a^2
  let base_area := Real.pi * r^2
  let total_area := 2 * base_area + side_area
  (total_area / side_area) = (1 + 2 * Real.pi) / (2 * Real.pi) := by
sorry

end cylinder_surface_area_ratio_l3566_356618


namespace loop_requirement_correct_l3566_356657

/-- Represents a mathematical operation that may or may not require a loop statement --/
inductive MathOperation
  | GeometricSum
  | CompareNumbers
  | PiecewiseFunction
  | LargestNaturalNumber

/-- Determines if a given mathematical operation requires a loop statement --/
def requires_loop (op : MathOperation) : Prop :=
  match op with
  | MathOperation.GeometricSum => true
  | MathOperation.CompareNumbers => false
  | MathOperation.PiecewiseFunction => false
  | MathOperation.LargestNaturalNumber => true

theorem loop_requirement_correct :
  (requires_loop MathOperation.GeometricSum) ∧
  (¬requires_loop MathOperation.CompareNumbers) ∧
  (¬requires_loop MathOperation.PiecewiseFunction) ∧
  (requires_loop MathOperation.LargestNaturalNumber) :=
by sorry

#check loop_requirement_correct

end loop_requirement_correct_l3566_356657


namespace video_game_players_l3566_356633

/-- The number of friends who quit the game -/
def quit_players : ℕ := 5

/-- The number of lives each remaining player had -/
def lives_per_player : ℕ := 5

/-- The total number of lives after some players quit -/
def total_lives : ℕ := 15

/-- The initial number of friends playing the video game online -/
def initial_players : ℕ := 8

theorem video_game_players :
  initial_players = quit_players + total_lives / lives_per_player := by
  sorry

end video_game_players_l3566_356633


namespace exponent_equation_solution_l3566_356629

theorem exponent_equation_solution :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = 9^(y - 1) ↔ y = 0 :=
by
  sorry

end exponent_equation_solution_l3566_356629


namespace sufficient_but_not_necessary_l3566_356660

def A (θ : Real) : Set Real := {1, Real.sin θ}
def B : Set Real := {1/2, 2}

theorem sufficient_but_not_necessary :
  (∀ θ : Real, θ = 5 * Real.pi / 6 → A θ ∩ B = {1/2}) ∧
  (∃ θ : Real, θ ≠ 5 * Real.pi / 6 ∧ A θ ∩ B = {1/2}) :=
sorry

end sufficient_but_not_necessary_l3566_356660


namespace odd_function_properties_l3566_356654

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = f(x-1)
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (x - 1)

-- Define periodicity with period 2
def periodic_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Define symmetry about (k, 0) for all integer k
def symmetric_about_int (f : ℝ → ℝ) : Prop :=
  ∀ (k : ℤ) (x : ℝ), f (2 * k - x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : odd_function f) (h_prop : property_f f) :
  periodic_2 f ∧ symmetric_about_int f := by
  sorry

end odd_function_properties_l3566_356654


namespace not_p_sufficient_not_necessary_for_q_l3566_356694

-- Define the statements p and q
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, ¬(p a) → q a) ∧ ¬(∀ a : ℝ, q a → ¬(p a)) := by
  sorry

end not_p_sufficient_not_necessary_for_q_l3566_356694


namespace fraction_sum_equals_four_l3566_356648

theorem fraction_sum_equals_four : 
  (2 : ℚ) / 15 + 4 / 15 + 6 / 15 + 8 / 15 + 10 / 15 + 30 / 15 = 4 := by
  sorry

end fraction_sum_equals_four_l3566_356648


namespace frequency_count_theorem_l3566_356634

theorem frequency_count_theorem (sample_size : ℕ) (relative_frequency : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : relative_frequency = 0.2) :
  (sample_size : ℝ) * relative_frequency = 20 := by
  sorry

end frequency_count_theorem_l3566_356634


namespace range_of_a_l3566_356625

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, (¬(¬(p a) ∨ ¬(q a))) → range_a a :=
sorry

end range_of_a_l3566_356625


namespace circle_condition_l3566_356617

theorem circle_condition (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - k*x + 2*y + k^2 - 2 = 0 ↔ ∃ r > 0, ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) ↔
  -2 < k ∧ k < 2 :=
by sorry

end circle_condition_l3566_356617


namespace rectangle_area_l3566_356608

/-- Theorem: Area of a rectangle with one side 15 and diagonal 17 is 120 -/
theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 := by
  sorry

end rectangle_area_l3566_356608


namespace circle_area_increase_l3566_356674

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by sorry

end circle_area_increase_l3566_356674


namespace average_of_abc_is_three_l3566_356691

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2012 * C - 4024 * A = 8048)
  (eq2 : 2012 * B + 6036 * A = 10010) :
  (A + B + C) / 3 = 3 := by
sorry

end average_of_abc_is_three_l3566_356691


namespace average_increase_percentage_l3566_356683

def S : Finset Int := {6, 7, 10, 12, 15}
def N : Int := 34

theorem average_increase_percentage (S : Finset Int) (N : Int) :
  S = {6, 7, 10, 12, 15} →
  N = 34 →
  let original_sum := S.sum id
  let original_count := S.card
  let original_avg := original_sum / original_count
  let new_sum := original_sum + N
  let new_count := original_count + 1
  let new_avg := new_sum / new_count
  let increase := new_avg - original_avg
  let percentage_increase := (increase / original_avg) * 100
  percentage_increase = 40 := by
sorry

end average_increase_percentage_l3566_356683
