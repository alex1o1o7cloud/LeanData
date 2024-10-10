import Mathlib

namespace set_operations_and_subset_l3120_312004

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (∃ a : ℝ, C a ⊆ B) →
  (Set.compl (A ∩ B) = {x : ℝ | x < 3 ∨ 6 ≤ x}) ∧
  (Set.compl B ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (Set.Icc 2 8 = {a : ℝ | C a ⊆ B}) := by
  sorry

end set_operations_and_subset_l3120_312004


namespace cosine_vertical_shift_l3120_312064

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (hmax : d + a = 7) 
  (hmin : d - a = 1) : 
  d = 4 := by
sorry

end cosine_vertical_shift_l3120_312064


namespace unique_solution_for_equation_l3120_312017

theorem unique_solution_for_equation : 
  ∃! (n m : ℕ), (n - 1) * 2^(n - 1) + 5 = m^2 + 4*m ∧ n = 6 ∧ m = 11 := by
  sorry

end unique_solution_for_equation_l3120_312017


namespace track_length_proof_l3120_312002

/-- The length of the circular track -/
def track_length : ℝ := 520

/-- The distance Brenda runs to the first meeting point -/
def brenda_first_distance : ℝ := 80

/-- The distance Sue runs past the first meeting point to the second meeting point -/
def sue_second_distance : ℝ := 180

/-- Theorem stating the track length given the conditions -/
theorem track_length_proof :
  ∀ (x : ℝ),
  x > 0 →
  (x / 2 - brenda_first_distance) / brenda_first_distance = 
  (x / 2 - (sue_second_distance + brenda_first_distance)) / (x / 2 + sue_second_distance) →
  x = track_length := by
sorry

end track_length_proof_l3120_312002


namespace sin_50_plus_sqrt3_tan_10_l3120_312072

theorem sin_50_plus_sqrt3_tan_10 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_l3120_312072


namespace C_power_50_l3120_312079

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end C_power_50_l3120_312079


namespace largest_n_binomial_equality_l3120_312081

theorem largest_n_binomial_equality : ∃ (n : ℕ), (
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧
  (∀ m : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n)
) ∧ n = 6 := by
  sorry

end largest_n_binomial_equality_l3120_312081


namespace circle_area_ratio_concentric_circles_area_ratio_l3120_312056

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > r₁ → 
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = (r₂^2 / r₁^2) - 1 :=
by sorry

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 :=
by sorry

end circle_area_ratio_concentric_circles_area_ratio_l3120_312056


namespace serena_mother_triple_age_l3120_312033

/-- The number of years it will take for the mother to be three times as old as the daughter. -/
def years_until_triple_age (daughter_age : ℕ) (mother_age : ℕ) : ℕ :=
  (mother_age - 3 * daughter_age) / 2

/-- Theorem stating that it will take 6 years for Serena's mother to be three times as old as Serena. -/
theorem serena_mother_triple_age :
  years_until_triple_age 9 39 = 6 := by
  sorry

end serena_mother_triple_age_l3120_312033


namespace test_mean_score_l3120_312065

theorem test_mean_score (mean : ℝ) (std_dev : ℝ) (lowest_score : ℝ) : 
  std_dev = 10 →
  lowest_score = mean - 2 * std_dev →
  lowest_score = 20 →
  mean = 40 := by
sorry

end test_mean_score_l3120_312065


namespace projection_matrix_values_l3120_312007

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 3/7; c, 4/7]
  is_projection_matrix P ↔ a = 1 ∧ c = 3/7 := by
sorry

end projection_matrix_values_l3120_312007


namespace fraction_to_decimal_l3120_312051

theorem fraction_to_decimal : (45 : ℚ) / (2^2 * 5^3) = (9 : ℚ) / 100 := by
  sorry

end fraction_to_decimal_l3120_312051


namespace sqrt_difference_equality_l3120_312010

theorem sqrt_difference_equality : Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end sqrt_difference_equality_l3120_312010


namespace solution_set_characterization_l3120_312028

open Real

theorem solution_set_characterization 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_initial : f 0 = 2)
  (h_bound : ∀ x, f x + f' x > 1) :
  ∀ x, (exp x * f x > exp x + 1) ↔ x > 0 := by
sorry

end solution_set_characterization_l3120_312028


namespace sine_function_period_l3120_312025

/-- Given a sinusoidal function with angular frequency ω > 0 and smallest positive period 2π/3, prove that ω = 3. -/
theorem sine_function_period (ω : ℝ) : ω > 0 → (∀ x, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + 2 * π / 3) + π / 6)) → ω = 3 := by
  sorry

end sine_function_period_l3120_312025


namespace square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l3120_312003

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3*n^2 ∧ b = 2*m*n :=
sorry

-- Part 2
theorem square_root_three_specific_case (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l3120_312003


namespace acute_triangle_side_range_l3120_312040

theorem acute_triangle_side_range (x : ℝ) : 
  x > 0 → 
  (∀ α β γ : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2 ∧ 
   α + β + γ = π ∧
   x^2 = 2^2 + 3^2 - 2*2*3*Real.cos γ ∧
   2^2 = 3^2 + x^2 - 2*3*x*Real.cos α ∧
   3^2 = 2^2 + x^2 - 2*2*x*Real.cos β) →
  Real.sqrt 5 < x ∧ x < Real.sqrt 13 :=
by sorry

end acute_triangle_side_range_l3120_312040


namespace fraction_sum_l3120_312014

theorem fraction_sum (a b : ℕ) (h1 : a.Coprime b) (h2 : a > 0) (h3 : b > 0)
  (h4 : (5 : ℚ) / 6 * (a^2 : ℚ) / (b^2 : ℚ) = 2 * (a : ℚ) / (b : ℚ)) :
  a + b = 17 := by
sorry

end fraction_sum_l3120_312014


namespace eliza_ironed_17_clothes_l3120_312019

/-- Calculates the total number of clothes Eliza ironed given the time spent and ironing rates. -/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Proves that Eliza ironed 17 pieces of clothes given the conditions. -/
theorem eliza_ironed_17_clothes :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

#eval total_clothes_ironed 15 20 2 3

end eliza_ironed_17_clothes_l3120_312019


namespace sum_not_prime_l3120_312090

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : 
  ¬ Nat.Prime (a.val + b.val + c.val + d.val) := by
  sorry

end sum_not_prime_l3120_312090


namespace james_candy_bar_sales_l3120_312094

/-- Proves that James sells 5 boxes of candy bars given the conditions of the fundraiser -/
theorem james_candy_bar_sales :
  let boxes_to_bars : ℕ → ℕ := λ x => 10 * x
  let selling_price : ℚ := 3/2
  let buying_price : ℚ := 1
  let profit_per_bar : ℚ := selling_price - buying_price
  let total_profit : ℚ := 25
  ∃ (num_boxes : ℕ), 
    (boxes_to_bars num_boxes : ℚ) * profit_per_bar = total_profit ∧ 
    num_boxes = 5 :=
by
  sorry

end james_candy_bar_sales_l3120_312094


namespace quadratic_condition_for_x_greater_than_two_l3120_312096

theorem quadratic_condition_for_x_greater_than_two :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2) :=
by sorry

end quadratic_condition_for_x_greater_than_two_l3120_312096


namespace ellipse_equation_equivalence_l3120_312098

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt (x^2 + (y + 2)^2) = 10) ↔
  (y^2 / 25 + x^2 / 21 = 1) :=
sorry

end ellipse_equation_equivalence_l3120_312098


namespace elevator_weight_problem_l3120_312044

theorem elevator_weight_problem (adults_avg_weight : ℝ) (elevator_max_weight : ℝ) (next_person_max_weight : ℝ) :
  adults_avg_weight = 140 →
  elevator_max_weight = 600 →
  next_person_max_weight = 52 →
  (elevator_max_weight - 3 * adults_avg_weight - next_person_max_weight) = 128 :=
by sorry

end elevator_weight_problem_l3120_312044


namespace tim_soda_cans_l3120_312050

/-- The number of soda cans Tim has at the end of the scenario -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  remaining + remaining / 2

/-- Theorem stating that Tim ends up with 24 cans of soda -/
theorem tim_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end tim_soda_cans_l3120_312050


namespace car_speed_second_hour_l3120_312057

/-- Proves that given a car's speed of 95 km/h in the first hour and an average speed of 77.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 95)
  (h2 : average_speed = 77.5) : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 60 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry


end car_speed_second_hour_l3120_312057


namespace determinant_transformation_l3120_312005

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
  sorry

end determinant_transformation_l3120_312005


namespace partial_fraction_decomposition_l3120_312041

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -7) :
  (4 * x - 6) / (x^2 - 2*x - 63) = (15 / 8) / (x - 9) + (17 / 8) / (x + 7) := by
  sorry

end partial_fraction_decomposition_l3120_312041


namespace p_is_true_l3120_312069

theorem p_is_true (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
by sorry

end p_is_true_l3120_312069


namespace fraction_males_first_class_l3120_312036

/-- Given a flight with passengers, prove the fraction of males in first class -/
theorem fraction_males_first_class
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_percentage : ℚ)
  (females_in_coach : ℕ)
  (h_total : total_passengers = 120)
  (h_female : female_percentage = 30 / 100)
  (h_first_class : first_class_percentage = 10 / 100)
  (h_females_coach : females_in_coach = 28) :
  (↑(total_passengers * first_class_percentage.num - (total_passengers * female_percentage.num - females_in_coach)) /
   ↑(total_passengers * first_class_percentage.num) : ℚ) = 1 / 3 :=
by sorry

end fraction_males_first_class_l3120_312036


namespace gcd_of_B_is_five_l3120_312058

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x-2) + (x-1) + x + (x+1) + (x+2)}

theorem gcd_of_B_is_five :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
sorry

end gcd_of_B_is_five_l3120_312058


namespace picture_hanging_l3120_312053

theorem picture_hanging (board_width : ℕ) (picture_width : ℕ) (num_pictures : ℕ) :
  board_width = 320 ∧ picture_width = 30 ∧ num_pictures = 6 →
  (board_width - num_pictures * picture_width) / (num_pictures + 1) = 20 :=
by sorry

end picture_hanging_l3120_312053


namespace function_inequality_implies_a_range_l3120_312052

open Real

theorem function_inequality_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ →
    |x₁ + a * log x₁ - (x₂ + a * log x₂)| < |1 / x₁ - 1 / x₂|) →
  a < 8 / 3 := by
  sorry

end function_inequality_implies_a_range_l3120_312052


namespace gcd_14658_11241_l3120_312006

theorem gcd_14658_11241 : Nat.gcd 14658 11241 = 3 := by
  sorry

end gcd_14658_11241_l3120_312006


namespace sqrt_10_power_identity_l3120_312021

theorem sqrt_10_power_identity : (Real.sqrt 10 + 3)^2023 * (Real.sqrt 10 - 3)^2022 = Real.sqrt 10 + 3 := by
  sorry

end sqrt_10_power_identity_l3120_312021


namespace train_length_train_length_specific_l3120_312035

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) : Real :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 114.99 kmph passing a man walking at 5 kmph in the opposite direction in 6 seconds has a length of approximately 199.98 meters. -/
theorem train_length_specific : 
  ∃ (ε : Real), ε > 0 ∧ abs (train_length 114.99 5 6 - 199.98) < ε :=
by
  sorry

end train_length_train_length_specific_l3120_312035


namespace product_of_roots_l3120_312068

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x * y = -34 :=
by
  sorry

end product_of_roots_l3120_312068


namespace handbag_price_adjustment_l3120_312084

/-- Calculates the final price of a handbag after a price increase followed by a discount -/
theorem handbag_price_adjustment (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * 1.2) * 0.8 = 48 := by sorry

end handbag_price_adjustment_l3120_312084


namespace expression_result_l3120_312043

theorem expression_result : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end expression_result_l3120_312043


namespace inverse_proportion_l3120_312078

/-- Given that x is inversely proportional to y, prove that when x = 5 for y = -4, 
    then x = 2 for y = -10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x is inversely proportional to y
    (h2 : 5 * (-4) = k)  -- x = 5 when y = -4
    : x = 2 ∧ y = -10 → x * y = k := by
  sorry

end inverse_proportion_l3120_312078


namespace investment_result_approx_17607_l3120_312055

/-- Calculates the final amount of an investment after tax and compound interest --/
def investment_after_tax (initial_investment : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  let compound_factor := 1 + interest_rate * (1 - tax_rate)
  initial_investment * compound_factor ^ years

/-- Theorem stating that the investment result is approximately $17,607 --/
theorem investment_result_approx_17607 :
  ∃ ε > 0, |investment_after_tax 15000 0.05 0.10 4 - 17607| < ε :=
sorry

end investment_result_approx_17607_l3120_312055


namespace sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l3120_312077

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds : 
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧ 
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l3120_312077


namespace bacterial_eradication_l3120_312060

/-- Represents the state of the bacterial culture at a given minute -/
structure BacterialState where
  minute : ℕ
  infected : ℕ
  nonInfected : ℕ

/-- Models the evolution of the bacterial culture over time -/
def evolve (n : ℕ) : ℕ → BacterialState
  | 0 => ⟨0, 1, n - 1⟩
  | t + 1 => 
    let prev := evolve n t
    ⟨t + 1, 2 * prev.infected, (prev.nonInfected * 2) - (2 * prev.infected)⟩

/-- Theorem stating that the bacterial culture will be eradicated in n minutes -/
theorem bacterial_eradication (n : ℕ) (h : n > 0) : 
  (evolve n (n - 1)).nonInfected = 0 ∧ (evolve n n).infected = 0 := by
  sorry


end bacterial_eradication_l3120_312060


namespace interior_angles_sum_l3120_312048

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 3) - 2) = 3780) := by
  sorry

end interior_angles_sum_l3120_312048


namespace smallest_value_of_expression_l3120_312086

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : a = 2*b - c) : 
  ∃ (a₀ b₀ c₀ : ℤ), ∀ (a' b' c' : ℤ), 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| ≤ |Complex.abs (a' + b'*ω + c'*ω^3)| ∧ 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| = 0 :=
sorry

end smallest_value_of_expression_l3120_312086


namespace three_squares_divisible_to_not_divisible_l3120_312075

theorem three_squares_divisible_to_not_divisible (N : ℕ) :
  (∃ (n : ℕ) (a b c : ℤ), N = 9^n * (a^2 + b^2 + c^2) ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ ¬(3 ∣ k) ∧ ¬(3 ∣ m) ∧ ¬(3 ∣ n)) :=
by sorry

end three_squares_divisible_to_not_divisible_l3120_312075


namespace xias_initial_sticker_count_l3120_312034

/-- Theorem: Xia's initial sticker count
Given that Xia shared 100 stickers with her friends, had 5 sheets of stickers left,
and each sheet contains 10 stickers, prove that she initially had 150 stickers. -/
theorem xias_initial_sticker_count
  (shared_stickers : ℕ)
  (remaining_sheets : ℕ)
  (stickers_per_sheet : ℕ)
  (h1 : shared_stickers = 100)
  (h2 : remaining_sheets = 5)
  (h3 : stickers_per_sheet = 10) :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 :=
by sorry

end xias_initial_sticker_count_l3120_312034


namespace votes_for_candidate_D_l3120_312093

def total_votes : ℕ := 1000000
def invalid_percentage : ℚ := 25 / 100
def candidate_A_percentage : ℚ := 45 / 100
def candidate_B_percentage : ℚ := 30 / 100
def candidate_C_percentage : ℚ := 20 / 100
def candidate_D_percentage : ℚ := 5 / 100

theorem votes_for_candidate_D :
  (total_votes : ℚ) * (1 - invalid_percentage) * candidate_D_percentage = 37500 := by
  sorry

end votes_for_candidate_D_l3120_312093


namespace quadratic_b_value_l3120_312009

/-- Given a quadratic function y = ax² + bx + c, prove that b = 3 when
    (2, y₁) and (-2, y₂) are points on the graph and y₁ - y₂ = 12 -/
theorem quadratic_b_value (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 12 →
  b = 3 := by
sorry

end quadratic_b_value_l3120_312009


namespace complex_equation_implies_fourth_quadrant_l3120_312070

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_equation_implies_fourth_quadrant (z : ℂ) :
  (z + 3 * Complex.I) * (3 + Complex.I) = 7 - Complex.I →
  is_in_fourth_quadrant z := by
sorry

end complex_equation_implies_fourth_quadrant_l3120_312070


namespace equation_represents_two_lines_l3120_312091

theorem equation_represents_two_lines :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    x^2 - 50*y^2 - 16*x + 64 = 0 ↔ (x = a*y + b ∨ x = -a*y + b) :=
by sorry

end equation_represents_two_lines_l3120_312091


namespace parallel_vectors_x_value_l3120_312076

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (x, 1)
  parallel a b → x = -1 := by
sorry

end parallel_vectors_x_value_l3120_312076


namespace cube_sum_problem_l3120_312083

theorem cube_sum_problem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by sorry

end cube_sum_problem_l3120_312083


namespace exponent_multiplication_l3120_312031

theorem exponent_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end exponent_multiplication_l3120_312031


namespace range_of_a_l3120_312001

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = a^2}

theorem range_of_a (a : ℝ) :
  (set_A ∩ set_B a = ∅) ↔ (-2*Real.sqrt 2 - 2 < a ∧ a < 2*Real.sqrt 2 + 2) :=
sorry

end range_of_a_l3120_312001


namespace cookie_batch_size_l3120_312023

theorem cookie_batch_size (batch_count : ℕ) (oatmeal_count : ℕ) (total_count : ℕ) : 
  batch_count = 2 → oatmeal_count = 4 → total_count = 10 → 
  ∃ (cookies_per_batch : ℕ), cookies_per_batch = 3 ∧ batch_count * cookies_per_batch + oatmeal_count = total_count :=
by
  sorry

end cookie_batch_size_l3120_312023


namespace vector_triangle_l3120_312088

/-- Given vectors a and b, if 4a, 3b - 2a, and c form a triangle, then c = (4, -6) -/
theorem vector_triangle (a b c : ℝ × ℝ) : 
  a = (1, -3) → 
  b = (-2, 4) → 
  4 • a + (3 • b - 2 • a) + c = (0, 0) → 
  c = (4, -6) := by
sorry

end vector_triangle_l3120_312088


namespace signup_theorem_l3120_312000

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of competitions --/
def num_competitions : ℕ := 3

/-- The total number of ways to sign up --/
def total_ways : ℕ := num_competitions ^ num_students

/-- The number of ways to sign up if each event has participants --/
def ways_with_all_events : ℕ := 
  (Nat.choose num_students (num_students - num_competitions)) * (Nat.factorial num_competitions)

theorem signup_theorem : 
  total_ways = 81 ∧ ways_with_all_events = 36 := by
  sorry

end signup_theorem_l3120_312000


namespace h_is_even_l3120_312013

-- Define k as an even function
def k_even (k : ℝ → ℝ) : Prop :=
  ∀ x, k (-x) = k x

-- Define h using k
def h (k : ℝ → ℝ) (x : ℝ) : ℝ :=
  |k (x^5)|

-- Theorem statement
theorem h_is_even (k : ℝ → ℝ) (h_even : k_even k) :
  ∀ x, h k (-x) = h k x :=
by sorry

end h_is_even_l3120_312013


namespace interest_rate_calculation_l3120_312045

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 3600 → 
  t = 2 → 
  P * ((1 + r)^t - 1) - P * r * t = diff → 
  diff = 36 → 
  r = 0.1 := by
sorry

end interest_rate_calculation_l3120_312045


namespace canoe_oar_probability_l3120_312067

theorem canoe_oar_probability (p_row : ℝ) (h_p_row : p_row = 0.84) : 
  ∃ (p_right : ℝ), 
    (p_right = 1 - Real.sqrt (1 - p_row)) ∧ 
    (p_right = 0.6) := by
  sorry

end canoe_oar_probability_l3120_312067


namespace quadratic_roots_imply_c_value_l3120_312016

theorem quadratic_roots_imply_c_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 8 ∨ x = (-8 - Real.sqrt 16) / 8) →
  c = 3 := by
sorry

end quadratic_roots_imply_c_value_l3120_312016


namespace sneeze_interval_l3120_312071

/-- Given a sneezing fit lasting 2 minutes with 40 sneezes in total, 
    prove that the time between each sneeze is 3 seconds. -/
theorem sneeze_interval (duration_minutes : ℕ) (total_sneezes : ℕ) 
  (h1 : duration_minutes = 2) 
  (h2 : total_sneezes = 40) : 
  (duration_minutes * 60) / total_sneezes = 3 := by
  sorry

end sneeze_interval_l3120_312071


namespace arithmetic_sequence_sum_ratio_l3120_312027

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if a_2 : a_3 = 5 : 2, then S_3 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) = a n + (a 2 - a 1)) →
  (a 2 : ℝ) / (a 3) = 5 / 2 →
  (S 3 : ℝ) / (S 5) = 3 / 2 := by
sorry

end arithmetic_sequence_sum_ratio_l3120_312027


namespace lucas_class_size_l3120_312097

theorem lucas_class_size (n : ℕ) (best_rank : ℕ) (worst_rank : ℕ)
  (h1 : best_rank = 30)
  (h2 : worst_rank = 45)
  (h3 : n = best_rank + worst_rank - 1) :
  n = 74 := by
sorry

end lucas_class_size_l3120_312097


namespace spider_leg_count_l3120_312066

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size : ℕ := spider_legs / 2 + 10

/-- The total number of spider legs in the group -/
def total_legs : ℕ := group_size * spider_legs

theorem spider_leg_count : total_legs = 112 := by
  sorry

end spider_leg_count_l3120_312066


namespace condition_relationship_l3120_312092

theorem condition_relationship :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) :=
by sorry

end condition_relationship_l3120_312092


namespace stratified_sampling_proof_l3120_312037

theorem stratified_sampling_proof (total_population : ℕ) (female_students : ℕ) 
  (sampled_female : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 1200)
  (h2 : female_students = 500)
  (h3 : sampled_female = 40)
  (h4 : (sample_size : ℚ) / total_population = (sampled_female : ℚ) / female_students) :
  sample_size = 96 := by
sorry

end stratified_sampling_proof_l3120_312037


namespace total_ballpoint_pens_l3120_312008

theorem total_ballpoint_pens (red_pens blue_pens : ℕ) 
  (h1 : red_pens = 37) 
  (h2 : blue_pens = 17) : 
  red_pens + blue_pens = 54 := by
  sorry

end total_ballpoint_pens_l3120_312008


namespace valentine_cards_theorem_l3120_312042

theorem valentine_cards_theorem (x y : ℕ) : 
  x * y = x + y + 30 → x * y = 64 := by
  sorry

end valentine_cards_theorem_l3120_312042


namespace remaining_note_denomination_l3120_312099

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : total_notes = 108)
  (h3 : fifty_notes = 97) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
sorry


end remaining_note_denomination_l3120_312099


namespace summer_locations_l3120_312087

/-- Represents a location with temperature data --/
structure Location where
  temperatures : Finset ℕ
  median : ℕ
  mean : ℕ
  mode : Option ℕ
  variance : Option ℚ

/-- Checks if a location meets the summer criterion --/
def meetsSummerCriterion (loc : Location) : Prop :=
  loc.temperatures.card = 5 ∧ ∀ t ∈ loc.temperatures.toSet, t ≥ 22

/-- Location A --/
def locationA : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 24,
  mean := 0,  -- Not given
  mode := some 22,
  variance := none
}

/-- Location B --/
def locationB : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 27,
  mean := 24,
  mode := none,
  variance := none
}

/-- Location C --/
def locationC : Location := {
  temperatures := {32},  -- We only know one temperature
  median := 0,  -- Not given
  mean := 26,
  mode := none,
  variance := some (108/10)
}

theorem summer_locations :
  meetsSummerCriterion locationA ∧
  meetsSummerCriterion locationC ∧
  ¬ (meetsSummerCriterion locationB) :=
sorry

end summer_locations_l3120_312087


namespace michaels_brothers_ages_l3120_312089

theorem michaels_brothers_ages (michael_age : ℕ) (older_brother_age : ℕ) (younger_brother_age : ℕ) :
  older_brother_age = 2 * (michael_age - 1) + 1 →
  younger_brother_age = older_brother_age / 3 →
  michael_age + older_brother_age + younger_brother_age = 28 →
  younger_brother_age = 5 := by
  sorry

end michaels_brothers_ages_l3120_312089


namespace quadratic_roots_product_l3120_312022

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end quadratic_roots_product_l3120_312022


namespace travel_time_calculation_l3120_312024

/-- Given a person traveling at a constant speed for a certain distance,
    calculate the time taken for the journey. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed = 75) (h2 : distance = 300) :
  distance / speed = 4 := by
  sorry

end travel_time_calculation_l3120_312024


namespace unique_base_solution_l3120_312063

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

-- Define the equation in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [7, 3, 6, 4] h + to_decimal [8, 4, 2, 1] h = to_decimal [1, 7, 2, 8, 5] h

-- Theorem statement
theorem unique_base_solution :
  ∃! h : Nat, h > 1 ∧ equation_holds h :=
sorry

end unique_base_solution_l3120_312063


namespace job_completion_time_l3120_312095

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 10

/-- The time the person works before Annie takes over -/
def person_work_time : ℝ := 3

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_remaining_time : ℝ := 8

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 15

theorem job_completion_time :
  (person_work_time / person_total_time) + (annie_remaining_time / annie_time) = 1 :=
sorry

end job_completion_time_l3120_312095


namespace bob_total_earnings_l3120_312046

-- Define constants
def regular_rate : ℚ := 5
def overtime_rate : ℚ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define function to calculate weekly earnings
def weekly_earnings (hours_worked : ℕ) : ℚ :=
  let regular := min hours_worked regular_hours
  let overtime := max (hours_worked - regular_hours) 0
  regular * regular_rate + overtime * overtime_rate

-- Theorem statement
theorem bob_total_earnings :
  weekly_earnings first_week_hours + weekly_earnings second_week_hours = 472 :=
by sorry

end bob_total_earnings_l3120_312046


namespace largest_circle_equation_l3120_312047

/-- The standard equation of the circle with the largest area, centered at (2, -3) and tangent to the line 2mx-y-2m-1=0 (m ∈ ℝ) -/
theorem largest_circle_equation (m : ℝ) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5 ∧ 
  ∀ (x' y' r : ℝ), 
    ((x' - 2)^2 + (y' + 3)^2 = r^2) → 
    (2*m*x' - y' - 2*m - 1 = 0) → 
    r^2 ≤ 5 := by
  sorry

#check largest_circle_equation

end largest_circle_equation_l3120_312047


namespace reaping_capacity_theorem_l3120_312032

/-- Represents the reaping capacity of a group of men -/
structure ReapingCapacity where
  men : ℕ
  hectares : ℝ
  days : ℕ

/-- Given the reaping capacity of one group, calculate the reaping capacity of another group -/
def calculate_reaping_capacity (base : ReapingCapacity) (target : ReapingCapacity) : Prop :=
  (target.men : ℝ) / base.men * (base.hectares / base.days) * target.days = target.hectares

/-- Theorem stating the relationship between the reaping capacities of two groups -/
theorem reaping_capacity_theorem (base target : ReapingCapacity) :
  base.men = 10 ∧ base.hectares = 80 ∧ base.days = 24 ∧
  target.men = 36 ∧ target.hectares = 360 ∧ target.days = 30 →
  calculate_reaping_capacity base target := by
  sorry

end reaping_capacity_theorem_l3120_312032


namespace f_properties_l3120_312059

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2))^2 + Real.sqrt 3 * Real.sin x

theorem f_properties :
  (∃ (M : ℝ), ∀ x, f x ≤ M ∧ (∃ x, f x = M) ∧ M = 3) ∧
  (∀ k : ℤ, f (Real.pi / 3 + 2 * k * Real.pi) = 3) ∧
  (∀ α : ℝ, Real.tan (α / 2) = 1 / 2 → f α = (8 + 4 * Real.sqrt 3) / 5) :=
by sorry

end f_properties_l3120_312059


namespace solution_of_system_l3120_312085

theorem solution_of_system (x y : ℚ) :
  (x + 5) / (x - 4) = (x - 7) / (x + 3) ∧ x + y = 20 →
  x = 13 / 19 ∧ y = 367 / 19 := by
sorry


end solution_of_system_l3120_312085


namespace inequality_proof_l3120_312073

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  (a + b + c) / 4 ≥ (Real.sqrt (a * b - 1)) / (b + c) + 
                    (Real.sqrt (b * c - 1)) / (c + a) + 
                    (Real.sqrt (c * a - 1)) / (a + b) :=
by sorry

end inequality_proof_l3120_312073


namespace line_equation_general_form_l3120_312049

/-- A line passing through a point with a given direction vector -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralForm where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line passing through (5,4) with direction vector (1,2),
    its general form equation is 2x - y - 6 = 0 -/
theorem line_equation_general_form (l : Line) 
    (h1 : l.point = (5, 4)) 
    (h2 : l.direction = (1, 2)) : 
    ∃ (gf : GeneralForm), gf.a = 2 ∧ gf.b = -1 ∧ gf.c = -6 :=
sorry

end line_equation_general_form_l3120_312049


namespace function_property_l3120_312074

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  is_even_function f :=
sorry

end function_property_l3120_312074


namespace impossible_triangle_angles_l3120_312080

-- Define a triangle
structure Triangle where
  -- We don't need to specify the actual properties of a triangle here

-- Define the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := 180

-- Theorem: It is impossible for the sum of the interior angles of a triangle to be 360°
theorem impossible_triangle_angles (t : Triangle) : sum_of_interior_angles t ≠ 360 := by
  sorry

end impossible_triangle_angles_l3120_312080


namespace area_between_concentric_circles_l3120_312062

/-- The area of the region between two concentric circles -/
theorem area_between_concentric_circles
  (r : ℝ) -- radius of the inner circle
  (h : r > 0) -- assumption that r is positive
  (width : ℝ) -- width of the region between circles
  (h_width : width = 3 * r - r) -- definition of width
  (h_width_value : width = 4) -- given width value
  : (π * (3 * r)^2 - π * r^2) = 8 * π * r^2 := by
  sorry

end area_between_concentric_circles_l3120_312062


namespace gcd_g_y_l3120_312039

def g (y : ℤ) : ℤ := (3*y + 5)*(8*y + 1)*(11*y + 3)*(y + 15)

theorem gcd_g_y (y : ℤ) (h : ∃ k : ℤ, y = 4060 * k) : 
  Int.gcd (g y) y = 5 := by sorry

end gcd_g_y_l3120_312039


namespace sum_of_squares_theorem_l3120_312030

theorem sum_of_squares_theorem (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1/3 := by sorry

end sum_of_squares_theorem_l3120_312030


namespace late_students_total_time_l3120_312054

theorem late_students_total_time (charlize_late : ℕ) 
  (h1 : charlize_late = 20)
  (ana_late : ℕ) 
  (h2 : ana_late = charlize_late + 5)
  (ben_late : ℕ) 
  (h3 : ben_late = charlize_late - 15)
  (clara_late : ℕ) 
  (h4 : clara_late = 2 * charlize_late)
  (daniel_late : ℕ) 
  (h5 : daniel_late = clara_late - 10) :
  charlize_late + ana_late + ben_late + clara_late + daniel_late = 120 := by
  sorry

end late_students_total_time_l3120_312054


namespace inequality_proof_l3120_312012

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
sorry

end inequality_proof_l3120_312012


namespace inequality_proof_l3120_312061

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ 
  (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by sorry

end inequality_proof_l3120_312061


namespace cone_height_from_lateral_surface_l3120_312020

/-- 
Given a cone whose lateral surface is a semicircle with radius a,
prove that the height of the cone is (√3/2)a
-/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let slant_height := a
  let base_circumference := π * a
  let base_radius := a / 2
  let height := Real.sqrt ((3 * a^2) / 4)
  height = (Real.sqrt 3 / 2) * a :=
by sorry

end cone_height_from_lateral_surface_l3120_312020


namespace train_speed_calculation_l3120_312026

/-- Proves that given the conditions of two trains passing each other, the speed of the first train is 72 kmph -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 time_to_cross : ℝ) 
  (h1 : length_train1 = 380)
  (h2 : length_train2 = 540)
  (h3 : speed_train2 = 36)
  (h4 : time_to_cross = 91.9926405887529)
  : (length_train1 + length_train2) / time_to_cross * 3.6 + speed_train2 = 72 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l3120_312026


namespace lizzy_money_after_loan_l3120_312011

def calculate_final_amount (initial_amount : ℝ) (loan_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

theorem lizzy_money_after_loan (initial_amount loan_amount interest_rate : ℝ) 
  (h1 : initial_amount = 30)
  (h2 : loan_amount = 15)
  (h3 : interest_rate = 0.2) :
  calculate_final_amount initial_amount loan_amount interest_rate = 33 := by
  sorry

end lizzy_money_after_loan_l3120_312011


namespace sphere_radius_from_segment_l3120_312015

/-- A spherical segment is a portion of a sphere cut off by a plane. -/
structure SphericalSegment where
  base_diameter : ℝ
  height : ℝ

/-- The radius of a sphere given a spherical segment. -/
def sphere_radius (segment : SphericalSegment) : ℝ :=
  sorry

theorem sphere_radius_from_segment (segment : SphericalSegment) 
  (h1 : segment.base_diameter = 24)
  (h2 : segment.height = 8) :
  sphere_radius segment = 13 := by
  sorry

end sphere_radius_from_segment_l3120_312015


namespace total_upload_hours_l3120_312018

def upload_hours (days : ℕ) (videos_per_day : ℕ) (hours_per_video : ℚ) : ℚ :=
  (days : ℚ) * (videos_per_day : ℚ) * hours_per_video

def june_upload_hours : ℚ :=
  upload_hours 10 5 2 +  -- June 1st to June 10th
  upload_hours 10 10 1 + -- June 11th to June 20th
  upload_hours 5 7 3 +   -- June 21st to June 25th
  upload_hours 5 15 (1/2) -- June 26th to June 30th

theorem total_upload_hours : june_upload_hours = 342.5 := by
  sorry

end total_upload_hours_l3120_312018


namespace brick_weight_l3120_312038

theorem brick_weight : ∃ x : ℝ, x = 2 + x / 3 → x = 3 := by
  sorry

end brick_weight_l3120_312038


namespace computer_table_cost_price_l3120_312029

/-- Proves that the cost price of a computer table is 6672 when the selling price is 8340 and the markup is 25% -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8340)
  (h2 : markup_percentage = 25) : 
  selling_price / (1 + markup_percentage / 100) = 6672 := by
  sorry

end computer_table_cost_price_l3120_312029


namespace termite_ridden_not_collapsing_l3120_312082

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = (5 * total_homes) / 8)
  (h2 : collapsing = (11 * termite_ridden) / 16) :
  (termite_ridden - collapsing) = (25 * total_homes) / 128 :=
by sorry

end termite_ridden_not_collapsing_l3120_312082
