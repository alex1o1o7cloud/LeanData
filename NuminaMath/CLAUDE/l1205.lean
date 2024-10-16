import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1205_120500

theorem sqrt_product_equality : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1205_120500


namespace NUMINAMATH_CALUDE_rajan_profit_share_l1205_120536

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  rajan_investment : ℕ) (rajan_duration : ℕ)
  (rakesh_investment : ℕ) (rakesh_duration : ℕ)
  (mukesh_investment : ℕ) (mukesh_duration : ℕ)
  (total_profit : ℕ) : ℕ :=
  let rajan_ratio := rajan_investment * rajan_duration
  let rakesh_ratio := rakesh_investment * rakesh_duration
  let mukesh_ratio := mukesh_investment * mukesh_duration
  let total_ratio := rajan_ratio + rakesh_ratio + mukesh_ratio
  (rajan_ratio * total_profit) / total_ratio

/-- Theorem stating that Rajan's share of the profit is 2400 --/
theorem rajan_profit_share :
  calculate_profit_share 20000 12 25000 4 15000 8 4600 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_rajan_profit_share_l1205_120536


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1205_120523

/-- A complex number z is in the first quadrant if its real part is positive and its imaginary part is positive. -/
def in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- Given a real number a, construct the complex number z = (3i - a) / i -/
def z (a : ℝ) : ℂ :=
  Complex.I * 3 - a

/-- The condition a > -1 is sufficient but not necessary for z(a) to be in the first quadrant -/
theorem sufficient_not_necessary (a : ℝ) :
  (∃ a₁ : ℝ, a₁ > -1 ∧ in_first_quadrant (z a₁)) ∧
  (∃ a₂ : ℝ, in_first_quadrant (z a₂) ∧ ¬(a₂ > -1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1205_120523


namespace NUMINAMATH_CALUDE_square_root_problem_l1205_120590

theorem square_root_problem (a : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : Real.sqrt n = a + 3) (h3 : Real.sqrt n = 2*a - 15) : n = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1205_120590


namespace NUMINAMATH_CALUDE_chess_board_pawn_placement_l1205_120545

theorem chess_board_pawn_placement :
  let board_size : ℕ := 5
  let num_pawns : ℕ := 5
  let ways_to_place_in_rows : ℕ := Nat.factorial board_size
  let ways_to_arrange_pawns : ℕ := Nat.factorial num_pawns
  ways_to_place_in_rows * ways_to_arrange_pawns = 14400 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_board_pawn_placement_l1205_120545


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l1205_120594

/-- The number of pages of math homework Rachel has to complete -/
def math_homework : ℕ := 8

/-- The number of pages of biology homework Rachel has to complete -/
def biology_homework : ℕ := 3

/-- The total number of pages of math and biology homework Rachel has to complete -/
def total_homework : ℕ := math_homework + biology_homework

theorem rachel_homework_pages :
  total_homework = 11 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l1205_120594


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l1205_120514

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 1/10

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 13/8

/-- The total cost of all flower pots -/
def total_cost : ℚ := 33/4

theorem flower_pots_total_cost :
  let prices := List.range num_pots |>.map (fun i => largest_pot_price - i * price_diff)
  prices.sum = total_cost := by sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l1205_120514


namespace NUMINAMATH_CALUDE_pet_store_bird_dog_ratio_l1205_120511

/-- Given a pet store with dogs, cats, birds, and fish, prove the ratio of birds to dogs. -/
theorem pet_store_bird_dog_ratio 
  (dogs : ℕ) 
  (cats : ℕ) 
  (birds : ℕ) 
  (fish : ℕ) 
  (h1 : dogs = 6) 
  (h2 : cats = dogs / 2) 
  (h3 : fish = 3 * dogs) 
  (h4 : dogs + cats + birds + fish = 39) : 
  birds / dogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_dog_ratio_l1205_120511


namespace NUMINAMATH_CALUDE_second_number_value_l1205_120522

theorem second_number_value (x : ℝ) : 3 + x * (8 - 3) = 24.16 → x = 4.232 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1205_120522


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1205_120534

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ x : ℝ, f x = 0 ∧ x = 5 - 2 * Real.sqrt 2 ∧ ∀ y : ℝ, f y = 0 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1205_120534


namespace NUMINAMATH_CALUDE_area_under_curve_l1205_120586

-- Define the curve
def f (x : ℝ) := x^2

-- Define the boundaries
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_under_curve :
  (∫ x in lower_bound..upper_bound, f x) = (1 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l1205_120586


namespace NUMINAMATH_CALUDE_congruence_problem_l1205_120597

theorem congruence_problem (x : ℤ) : (5 * x + 8) % 19 = 3 → (5 * x + 9) % 19 = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1205_120597


namespace NUMINAMATH_CALUDE_jacoby_trip_cost_l1205_120535

def trip_cost (hourly_rate job_hours cookie_price cookies_sold 
               lottery_ticket_cost lottery_winnings sister_gift sister_count
               additional_needed : ℕ) : ℕ :=
  let job_earnings := hourly_rate * job_hours
  let cookie_earnings := cookie_price * cookies_sold
  let sister_gifts := sister_gift * sister_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + sister_gifts
  let total_after_ticket := total_earned - lottery_ticket_cost
  total_after_ticket + additional_needed

theorem jacoby_trip_cost : 
  trip_cost 20 10 4 24 10 500 500 2 3214 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_jacoby_trip_cost_l1205_120535


namespace NUMINAMATH_CALUDE_missing_number_proof_l1205_120572

theorem missing_number_proof (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1205_120572


namespace NUMINAMATH_CALUDE_range_of_a_l1205_120539

def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

def C (a : ℝ) : Set ℝ := {t | ∃ x ∈ A a, t = x^2}

theorem range_of_a (a : ℝ) (h1 : a ≥ -2) (h2 : C a ⊆ B a) : 
  a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1205_120539


namespace NUMINAMATH_CALUDE_high_school_students_l1205_120554

theorem high_school_students (total_students : ℕ) 
  (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 40)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 450)
  (h5 : total_students = (music_students - both_students) + (art_students - both_students) + both_students + neither_students) :
  total_students = 500 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l1205_120554


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1205_120584

theorem lcm_hcf_problem (a b : ℕ+) (h1 : a = 8) (h2 : Nat.lcm a b = 24) (h3 : Nat.gcd a b = 4) : b = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1205_120584


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1205_120564

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote y = 4/3 * x is 5/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 4 / 3) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1205_120564


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1205_120519

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2,2) and (10,6)
    (x - 2)^2 + (y - 2)^2 = (x - 10)^2 + (y - 6)^2 ∧
    -- The line connecting (2,2) and (10,6) is perpendicular to y = mx + b
    (6 - 2) = -1 / m * (10 - 2)) →
  m + b = 14 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l1205_120519


namespace NUMINAMATH_CALUDE_vertical_pairwise_sets_l1205_120520

/-- Definition of a vertical pairwise set -/
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- Set M₁: y = 1/x² -/
def M₁ : Set (ℝ × ℝ) :=
  {p | p.2 = 1 / (p.1 ^ 2) ∧ p.1 ≠ 0}

/-- Set M₂: y = sin x + 1 -/
def M₂ : Set (ℝ × ℝ) :=
  {p | p.2 = Real.sin p.1 + 1}

/-- Set M₄: y = 2ˣ - 2 -/
def M₄ : Set (ℝ × ℝ) :=
  {p | p.2 = 2 ^ p.1 - 2}

/-- Theorem: M₁, M₂, and M₄ are vertical pairwise sets -/
theorem vertical_pairwise_sets :
  is_vertical_pairwise_set M₁ ∧
  is_vertical_pairwise_set M₂ ∧
  is_vertical_pairwise_set M₄ := by
  sorry

end NUMINAMATH_CALUDE_vertical_pairwise_sets_l1205_120520


namespace NUMINAMATH_CALUDE_emerie_nickels_l1205_120516

/-- The number of coin types -/
def num_coin_types : ℕ := 3

/-- The number of coins Zain has -/
def zain_coins : ℕ := 48

/-- The number of quarters Emerie has -/
def emerie_quarters : ℕ := 6

/-- The number of dimes Emerie has -/
def emerie_dimes : ℕ := 7

/-- The number of extra coins Zain has for each type -/
def extra_coins_per_type : ℕ := 10

theorem emerie_nickels : 
  (zain_coins - num_coin_types * extra_coins_per_type) - (emerie_quarters + emerie_dimes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_emerie_nickels_l1205_120516


namespace NUMINAMATH_CALUDE_car_speed_calculation_l1205_120538

/-- Proves that a car's speed is 52 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (gallon_to_liter : ℝ) (km_to_mile : ℝ) :
  fuel_efficiency = 32 →
  fuel_consumed = 3.9 →
  time = 5.7 →
  gallon_to_liter = 3.8 →
  km_to_mile = 1.6 →
  (fuel_consumed * gallon_to_liter * fuel_efficiency) / (time * km_to_mile) = 52 := by
sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l1205_120538


namespace NUMINAMATH_CALUDE_equal_sum_sequence_2017_sum_l1205_120581

/-- An equal sum sequence with a given first term and common sum. -/
def EqualSumSequence (a : ℕ → ℕ) (first_term : ℕ) (common_sum : ℕ) : Prop :=
  a 1 = first_term ∧ ∀ n : ℕ, a n + a (n + 1) = common_sum

/-- The sum of the first n terms of a sequence. -/
def SequenceSum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2017_sum :
    ∀ a : ℕ → ℕ, EqualSumSequence a 2 5 → SequenceSum a 2017 = 5042 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_2017_sum_l1205_120581


namespace NUMINAMATH_CALUDE_fathers_age_ratio_l1205_120585

theorem fathers_age_ratio (R : ℕ) : 
  let F := 4 * R
  let father_age_after_8 := F + 8
  let ronit_age_after_8 := R + 8
  let father_age_after_16 := F + 16
  let ronit_age_after_16 := R + 16
  (∃ M : ℕ, father_age_after_8 = M * ronit_age_after_8) ∧ 
  (father_age_after_16 = 2 * ronit_age_after_16) →
  (father_age_after_8 : ℚ) / ronit_age_after_8 = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_ratio_l1205_120585


namespace NUMINAMATH_CALUDE_cubic_equation_geometric_progression_l1205_120546

theorem cubic_equation_geometric_progression (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   x^3 + 16*x^2 + a*x + 64 = 0 ∧
   y^3 + 16*y^2 + a*y + 64 = 0 ∧
   z^3 + 16*z^2 + a*z + 64 = 0 ∧
   ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 64 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_geometric_progression_l1205_120546


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_120_perfect_square_l1205_120547

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem smallest_multiplier_for_120_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (120 * n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (120 * m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_120_perfect_square_l1205_120547


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l1205_120578

theorem sum_of_quotient_dividend_divisor (n d : ℕ) (h1 : n = 45) (h2 : d = 3) :
  n / d + n + d = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l1205_120578


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l1205_120598

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 78 →
  last_six_avg = 75 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 :=
by sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l1205_120598


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1205_120504

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the number of black balls
def black_balls : ℕ := 6

-- Theorem statement
theorem probability_of_red_ball :
  (red_balls : ℚ) / total_balls = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1205_120504


namespace NUMINAMATH_CALUDE_square_difference_to_fourth_power_l1205_120502

theorem square_difference_to_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_to_fourth_power_l1205_120502


namespace NUMINAMATH_CALUDE_lcm_18_24_l1205_120571

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1205_120571


namespace NUMINAMATH_CALUDE_proportion_proof_l1205_120576

theorem proportion_proof (a b c d : ℝ) : 
  a = 3 →
  a / b = 0.6 →
  a * d = 12 →
  a / b = c / d →
  (a, b, c, d) = (3, 5, 2.4, 4) := by
  sorry

end NUMINAMATH_CALUDE_proportion_proof_l1205_120576


namespace NUMINAMATH_CALUDE_charles_initial_bananas_l1205_120558

/-- The initial number of bananas Willie has -/
def willie_initial : ℕ := 48

/-- The final number of bananas Willie has -/
def willie_final : ℕ := 13

/-- The number of bananas Charles loses -/
def charles_loss : ℕ := 35

/-- The initial number of bananas Charles has -/
def charles_initial : ℕ := charles_loss

theorem charles_initial_bananas :
  charles_initial = 35 :=
sorry

end NUMINAMATH_CALUDE_charles_initial_bananas_l1205_120558


namespace NUMINAMATH_CALUDE_cc_eq_c_of_ab_eq_c_ad_eq_cd_of_ab_eq_c_l1205_120550

class SpecialBinaryOp (S : Type) where
  mul : S → S → S
  mul_property : ∀ (a b c d : S), mul (mul a b) (mul c d) = mul a d

namespace SpecialBinaryOp

variable {S : Type} [SpecialBinaryOp S]

theorem cc_eq_c_of_ab_eq_c (a b c : S) (h : mul a b = c) : mul c c = c := by
  sorry

theorem ad_eq_cd_of_ab_eq_c (a b c d : S) (h : mul a b = c) : mul a d = mul c d := by
  sorry

end SpecialBinaryOp

end NUMINAMATH_CALUDE_cc_eq_c_of_ab_eq_c_ad_eq_cd_of_ab_eq_c_l1205_120550


namespace NUMINAMATH_CALUDE_greatest_of_five_consecutive_integers_sum_cube_l1205_120562

theorem greatest_of_five_consecutive_integers_sum_cube (n : ℤ) (m : ℤ) : 
  (5 * n + 10 = m^3) → 
  (∀ k : ℤ, k > n → 5 * k + 10 ≠ m^3) → 
  202 = n + 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_five_consecutive_integers_sum_cube_l1205_120562


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1205_120551

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_condition (k : ℝ) :
  is_perfect_square_trinomial 1 k 9 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1205_120551


namespace NUMINAMATH_CALUDE_equation_solutions_l1205_120559

def is_solution (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (1 : ℚ) / a + 1 / b + 1 / c = 1

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(3, 3, 3), (2, 3, 6), (2, 4, 4)} ∪ {(1, t, -t) | t : ℤ}

theorem equation_solutions :
  ∀ (a b c : ℤ), is_solution a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1205_120559


namespace NUMINAMATH_CALUDE_total_concert_attendance_l1205_120543

def first_concert_attendance : ℕ := 65899
def second_concert_difference : ℕ := 119

theorem total_concert_attendance : 
  let second_concert_attendance := first_concert_attendance + second_concert_difference
  let third_concert_attendance := 2 * second_concert_attendance
  first_concert_attendance + second_concert_attendance + third_concert_attendance = 263953 := by
sorry

end NUMINAMATH_CALUDE_total_concert_attendance_l1205_120543


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1205_120563

theorem bicycle_cost_price (profit_A_to_B : ℝ) (loss_B_to_C : ℝ) (profit_C_to_D : ℝ) (loss_D_to_E : ℝ) (price_E : ℝ)
  (h1 : profit_A_to_B = 0.20)
  (h2 : loss_B_to_C = 0.15)
  (h3 : profit_C_to_D = 0.30)
  (h4 : loss_D_to_E = 0.10)
  (h5 : price_E = 285) :
  price_E / ((1 + profit_A_to_B) * (1 - loss_B_to_C) * (1 + profit_C_to_D) * (1 - loss_D_to_E)) =
  285 / (1.20 * 0.85 * 1.30 * 0.90) := by
sorry

#eval 285 / (1.20 * 0.85 * 1.30 * 0.90)

end NUMINAMATH_CALUDE_bicycle_cost_price_l1205_120563


namespace NUMINAMATH_CALUDE_edge_length_of_total_72_l1205_120552

/-- Represents a rectangular prism with equal edge lengths -/
structure EqualEdgePrism where
  edge_length : ℝ
  total_length : ℝ
  total_length_eq : total_length = 12 * edge_length

/-- Theorem: If the sum of all edge lengths in an equal edge prism is 72 cm, 
    then the length of one edge is 6 cm -/
theorem edge_length_of_total_72 (prism : EqualEdgePrism) 
  (h : prism.total_length = 72) : prism.edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_edge_length_of_total_72_l1205_120552


namespace NUMINAMATH_CALUDE_money_problem_l1205_120548

theorem money_problem (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1205_120548


namespace NUMINAMATH_CALUDE_expected_coffee_days_expected_tea_days_expected_more_coffee_days_l1205_120507

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
| Prime
| Composite
| RollAgain

/-- Represents a fair eight-sided die with the given rules -/
def fairDie : Fin 8 → DieOutcome
| 1 => DieOutcome.RollAgain
| 2 => DieOutcome.Prime
| 3 => DieOutcome.Prime
| 4 => DieOutcome.Composite
| 5 => DieOutcome.Prime
| 6 => DieOutcome.Composite
| 7 => DieOutcome.Prime
| 8 => DieOutcome.Composite

/-- The probability of getting a prime number -/
def primeProbability : ℚ := 4 / 7

/-- The probability of getting a composite number -/
def compositeProbability : ℚ := 3 / 7

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

theorem expected_coffee_days (p : ℚ) (d : ℕ) (h : p = primeProbability) : 
  ⌊p * d⌋ = 209 :=
sorry

theorem expected_tea_days (p : ℚ) (d : ℕ) (h : p = compositeProbability) : 
  ⌊p * d⌋ = 156 :=
sorry

theorem expected_more_coffee_days : 
  ⌊primeProbability * daysInYear⌋ - ⌊compositeProbability * daysInYear⌋ = 53 :=
sorry

end NUMINAMATH_CALUDE_expected_coffee_days_expected_tea_days_expected_more_coffee_days_l1205_120507


namespace NUMINAMATH_CALUDE_prob_more_heads_than_tails_fair_coin_l1205_120595

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability of getting more heads than tails in 3 flips of a fair coin -/
def prob_more_heads_than_tails (p : ℝ) : ℝ :=
  prob_k_heads 3 2 p + prob_k_heads 3 3 p

theorem prob_more_heads_than_tails_fair_coin :
  ∀ p : ℝ, fair_coin p → prob_more_heads_than_tails p = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_prob_more_heads_than_tails_fair_coin_l1205_120595


namespace NUMINAMATH_CALUDE_student_score_l1205_120540

-- Define the number of questions
def num_questions : ℕ := 5

-- Define the points per question
def points_per_question : ℕ := 20

-- Define the number of correct answers
def num_correct_answers : ℕ := 4

-- Theorem statement
theorem student_score (total_score : ℕ) :
  total_score = num_correct_answers * points_per_question :=
by sorry

end NUMINAMATH_CALUDE_student_score_l1205_120540


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1205_120570

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + k = 0 → y = x) → 
  k = 4 := by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1205_120570


namespace NUMINAMATH_CALUDE_perfect_square_identification_l1205_120579

theorem perfect_square_identification (a b : ℝ) : 
  (∃ x : ℝ, a^2 - 4*a + 4 = x^2) ∧ 
  (¬∃ x : ℝ, 1 + 4*a^2 = x^2) ∧ 
  (¬∃ x : ℝ, 4*b^2 + 4*b - 1 = x^2) ∧ 
  (¬∃ x : ℝ, a^2 + a*b + b^2 = x^2) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l1205_120579


namespace NUMINAMATH_CALUDE_rosies_pies_l1205_120521

/-- Given that Rosie can make 3 pies from 12 apples, this theorem proves
    how many pies she can make from 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_three_pies = 12)
  (h2 : total_apples = 36) :
  (total_apples / apples_per_three_pies) * 3 = 9 :=
sorry

end NUMINAMATH_CALUDE_rosies_pies_l1205_120521


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1205_120530

theorem x_range_for_quadratic_inequality :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0) →
  ∃ a b : ℝ, a = (-1 + Real.sqrt 7) / 2 ∧ b = (1 + Real.sqrt 3) / 2 ∧
    ∀ x : ℝ, (a < x ∧ x < b) ↔ (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2 * x - m + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1205_120530


namespace NUMINAMATH_CALUDE_book_code_is_mirror_l1205_120533

/-- Represents the coding system --/
structure CodeSystem where
  book : String
  mirror : String
  board : String
  writing_item : String

/-- The given coding rules --/
def given_code : CodeSystem :=
  { book := "certain_item",
    mirror := "board",
    board := "board",
    writing_item := "2" }

/-- Theorem: The code for 'book' is 'mirror' --/
theorem book_code_is_mirror (code : CodeSystem) (h1 : code.book = "certain_item") 
  (h2 : code.mirror = "board") : code.book = "mirror" :=
by sorry

end NUMINAMATH_CALUDE_book_code_is_mirror_l1205_120533


namespace NUMINAMATH_CALUDE_joe_initial_money_l1205_120588

/-- The amount of money Joe spends on video games each month -/
def monthly_spend : ℕ := 50

/-- The amount of money Joe earns from selling games each month -/
def monthly_earn : ℕ := 30

/-- The number of months Joe can continue buying and selling games -/
def months : ℕ := 12

/-- The initial amount of money Joe has -/
def initial_money : ℕ := (monthly_spend - monthly_earn) * months

theorem joe_initial_money :
  initial_money = 240 :=
sorry

end NUMINAMATH_CALUDE_joe_initial_money_l1205_120588


namespace NUMINAMATH_CALUDE_problem_statement_l1205_120555

theorem problem_statement : ((16^15 / 16^14)^3 * 8^3) / 2^9 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1205_120555


namespace NUMINAMATH_CALUDE_base_k_conversion_l1205_120573

theorem base_k_conversion (k : ℕ) : 
  (0 < k ∧ k < 10) → (k^2 + 7*k + 5 = 125) → k = 8 :=
by sorry

end NUMINAMATH_CALUDE_base_k_conversion_l1205_120573


namespace NUMINAMATH_CALUDE_only_5_6_10_forms_triangle_l1205_120542

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: Among the given sets, only (5, 6, 10) can form a triangle -/
theorem only_5_6_10_forms_triangle :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 6 11 ∧
  can_form_triangle 5 6 10 ∧
  ¬ can_form_triangle 4 4 8 :=
sorry

end NUMINAMATH_CALUDE_only_5_6_10_forms_triangle_l1205_120542


namespace NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_is_196_l1205_120527

theorem rectangle_area_difference : ℕ → Prop :=
  fun diff =>
    ∀ l w : ℕ,
      (l > 0 ∧ w > 0) →  -- Ensure positive side lengths
      (2 * l + 2 * w = 60) →  -- Perimeter condition
      ∃ l_max w_max l_min w_min : ℕ,
        (l_max > 0 ∧ w_max > 0 ∧ l_min > 0 ∧ w_min > 0) →
        (2 * l_max + 2 * w_max = 60) →
        (2 * l_min + 2 * w_min = 60) →
        (∀ l' w' : ℕ, (l' > 0 ∧ w' > 0) → (2 * l' + 2 * w' = 60) → 
          l' * w' ≤ l_max * w_max ∧ l' * w' ≥ l_min * w_min) →
        diff = l_max * w_max - l_min * w_min

theorem rectangle_area_difference_is_196 : rectangle_area_difference 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_is_196_l1205_120527


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l1205_120589

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l1205_120589


namespace NUMINAMATH_CALUDE_equation_solution_l1205_120596

theorem equation_solution :
  ∃ x : ℚ, x - 1 ≠ 0 ∧ 1 - 1 / (x - 1) = 2 * x / (1 - x) ∧ x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1205_120596


namespace NUMINAMATH_CALUDE_special_circle_equation_midpoint_trajectory_l1205_120510

/-- A circle passing through two points with its center on a line -/
structure SpecialCircle where
  -- The circle passes through these two points
  A : ℝ × ℝ := (1, 0)
  B : ℝ × ℝ := (-1, -2)
  -- The center C lies on this line
  center_line : ℝ × ℝ → Prop := fun (x, y) ↦ x - y + 1 = 0

/-- The endpoint B of line segment AB -/
def endpointB : ℝ × ℝ := (4, 3)

theorem special_circle_equation (c : SpecialCircle) :
  ∃ (center : ℝ × ℝ),
    c.center_line center ∧
    ∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (c.A.1 - center.1)^2 + (c.A.2 - center.2)^2 ∧
       (x - center.1)^2 + (y - center.2)^2 = (c.B.1 - center.1)^2 + (c.B.2 - center.2)^2) :=
sorry

theorem midpoint_trajectory (c : SpecialCircle) :
  ∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
    ∃ (a : ℝ × ℝ), 
      (a.1 + 1)^2 + a.2^2 = 4 ∧
      x = (a.1 + endpointB.1) / 2 ∧
      y = (a.2 + endpointB.2) / 2 :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_midpoint_trajectory_l1205_120510


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1205_120560

theorem min_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  (1 / (a + 2*b)) + (4 / (2*a + b)) ≥ 3 :=
by sorry

theorem min_value_achieved (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ a₀ + b₀ = 1 ∧ (1 / (a₀ + 2*b₀)) + (4 / (2*a₀ + b₀)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1205_120560


namespace NUMINAMATH_CALUDE_expression_simplification_l1205_120565

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1205_120565


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l1205_120501

theorem quadratic_function_comparison : ∀ (y₁ y₂ : ℝ),
  y₁ = -(1:ℝ)^2 + 2 →
  y₂ = -(3:ℝ)^2 + 2 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_comparison_l1205_120501


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1205_120537

theorem simplify_complex_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (224 - 22 * Real.sqrt 105) / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1205_120537


namespace NUMINAMATH_CALUDE_B_power_48_l1205_120512

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_48 : 
  B ^ 48 = !![0, 0, 0; 0, 1, 0; 0, 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_48_l1205_120512


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1205_120529

theorem cube_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = 1) 
  (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1205_120529


namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l1205_120569

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square :
  ∀ n : ℕ, is_perfect_square (sum_of_factorials n) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l1205_120569


namespace NUMINAMATH_CALUDE_fraction_addition_l1205_120549

theorem fraction_addition (d : ℝ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1205_120549


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1205_120531

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 2) = 27 ^ y ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1205_120531


namespace NUMINAMATH_CALUDE_defective_smartphones_l1205_120508

theorem defective_smartphones (total : ℕ) (prob : ℝ) (defective : ℕ) : 
  total = 220 → 
  prob = 0.14470734744707348 →
  (defective : ℝ) / total * ((defective : ℝ) - 1) / (total - 1) = prob →
  defective = 84 :=
by sorry

end NUMINAMATH_CALUDE_defective_smartphones_l1205_120508


namespace NUMINAMATH_CALUDE_driving_distance_differences_l1205_120513

/-- Represents the driving scenario with Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_extra_time : ℝ := 2
  han_extra_speed : ℝ := 10
  jan_extra_time : ℝ := 3
  jan_extra_speed : ℝ := 15
  han_extra_distance : ℝ := 100

/-- Calculate the distance driven by Ian -/
def ian_distance (scenario : DrivingScenario) : ℝ :=
  scenario.ian_time * scenario.ian_speed

/-- Calculate the distance driven by Han -/
def han_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.han_extra_time) * (scenario.ian_speed + scenario.han_extra_speed)

/-- Calculate the distance driven by Jan -/
def jan_distance (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_time + scenario.jan_extra_time) * (scenario.ian_speed + scenario.jan_extra_speed)

/-- The main theorem stating the differences in distances driven -/
theorem driving_distance_differences (scenario : DrivingScenario) :
  jan_distance scenario - ian_distance scenario = 150 ∧
  jan_distance scenario - han_distance scenario = 150 := by
  sorry


end NUMINAMATH_CALUDE_driving_distance_differences_l1205_120513


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1205_120515

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 3x^2 - 2x - 6 = 0 -/
def are_roots (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x - 6 = 0 ∧ 3 * y^2 - 2 * y - 6 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  are_roots (a 1) (a 10) →
  a 4 * a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1205_120515


namespace NUMINAMATH_CALUDE_multiply_by_seven_l1205_120593

theorem multiply_by_seven (x : ℝ) : 7 * x = 50.68 → x = 7.24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l1205_120593


namespace NUMINAMATH_CALUDE_no_real_solutions_l1205_120524

theorem no_real_solutions : ¬∃ x : ℝ, |x| - 4 = (3 * |x|) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1205_120524


namespace NUMINAMATH_CALUDE_cutting_theorem_l1205_120574

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat

/-- State of the cutting process -/
structure CuttingState where
  strings : List PearlString

/-- Cutting rules -/
def cut_strings (k : Nat) (state : CuttingState) : CuttingState := sorry

/-- Predicate to check if a state has a white pearl of length 1 -/
def has_single_white_pearl (state : CuttingState) : Prop := sorry

/-- Predicate to check if a state has a black pearl string of length > 1 -/
def has_multiple_black_pearls (state : CuttingState) : Prop := sorry

/-- The cutting process -/
def cutting_process (k : Nat) (b w : Nat) : CuttingState := sorry

/-- Main theorem -/
theorem cutting_theorem (k : Nat) (b w : Nat) 
  (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  let final_state := cutting_process k b w
  has_single_white_pearl final_state → has_multiple_black_pearls final_state :=
by
  sorry


end NUMINAMATH_CALUDE_cutting_theorem_l1205_120574


namespace NUMINAMATH_CALUDE_sphere_radius_tangent_to_truncated_cone_l1205_120566

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_tangent_to_truncated_cone 
  (r_bottom r_top h : ℝ) 
  (h_positive : 0 < h) 
  (r_bottom_positive : 0 < r_bottom) 
  (r_top_positive : 0 < r_top) 
  (r_bottom_gt_r_top : r_top < r_bottom) 
  (h_truncated_cone : r_bottom = 24 ∧ r_top = 6 ∧ h = 20) :
  let r := (17 * Real.sqrt 2) / 2
  r = (Real.sqrt ((h^2 + (r_bottom - r_top)^2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_tangent_to_truncated_cone_l1205_120566


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1205_120532

/-- The number of ways to choose a starting lineup for a basketball team -/
def starting_lineup_count (total_members : ℕ) (center_capable : ℕ) : ℕ :=
  center_capable * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for a specific basketball team -/
theorem basketball_lineup_count :
  starting_lineup_count 12 4 = 31680 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1205_120532


namespace NUMINAMATH_CALUDE_simple_interest_months_l1205_120503

/-- Simple interest calculation -/
theorem simple_interest_months (P R SI : ℚ) (h1 : P = 10000) (h2 : R = 4/100) (h3 : SI = 400) :
  SI = P * R * (12/12) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_months_l1205_120503


namespace NUMINAMATH_CALUDE_pie_division_l1205_120518

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 8/9 ∧ num_people = 4 → 
  total_pie / num_people = 2/9 := by sorry

end NUMINAMATH_CALUDE_pie_division_l1205_120518


namespace NUMINAMATH_CALUDE_mork_mindy_tax_rate_l1205_120561

/-- The combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (mindy_income_ratio : ℝ) 
  (h1 : mork_rate = 0.4) 
  (h2 : mindy_rate = 0.3) 
  (h3 : mindy_income_ratio = 2) : 
  (mork_rate + mindy_rate * mindy_income_ratio) / (1 + mindy_income_ratio) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_mork_mindy_tax_rate_l1205_120561


namespace NUMINAMATH_CALUDE_rachels_to_christines_ratio_l1205_120583

def strawberries_per_pie : ℕ := 3
def christines_strawberries : ℕ := 10
def total_pies : ℕ := 10

theorem rachels_to_christines_ratio :
  let total_strawberries := strawberries_per_pie * total_pies
  let rachels_strawberries := total_strawberries - christines_strawberries
  (rachels_strawberries : ℚ) / christines_strawberries = 2 := by sorry

end NUMINAMATH_CALUDE_rachels_to_christines_ratio_l1205_120583


namespace NUMINAMATH_CALUDE_inequality_proof_l1205_120567

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1205_120567


namespace NUMINAMATH_CALUDE_least_integer_with_2035_divisors_l1205_120577

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the least positive integer with exactly 2035 distinct positive divisors -/
def n : ℕ := sorry

/-- m and k are integers such that n = m * 6^k and 6 is not a divisor of m -/
def m : ℕ := sorry
def k : ℕ := sorry

theorem least_integer_with_2035_divisors :
  num_divisors n = 2035 ∧
  n = m * 6^k ∧
  ¬(6 ∣ m) ∧
  ∀ i : ℕ, i < n → num_divisors i < 2035 →
  m + k = 26 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_2035_divisors_l1205_120577


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1205_120509

/-- Given a quadratic equation x^2 - mx - 6 = 0 with one root as 3,
    prove that the other root is -2 and m = 1. -/
theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 6 = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - m*y - 6 = 0 ∧ y = -2) ∧ m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1205_120509


namespace NUMINAMATH_CALUDE_focal_length_of_hyperbola_l1205_120541

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem focal_length_of_hyperbola (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x y, hyperbola m x y ↔ asymptote m x y) : 
  ∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 := by
  sorry

end NUMINAMATH_CALUDE_focal_length_of_hyperbola_l1205_120541


namespace NUMINAMATH_CALUDE_initial_crayon_packs_l1205_120525

theorem initial_crayon_packs : ℕ := by
  -- Define the cost of one pack of crayons
  let cost_per_pack : ℚ := 5/2

  -- Define the number of additional packs Michael buys
  let additional_packs : ℕ := 2

  -- Define the total value after purchase
  let total_value : ℚ := 15

  -- Define the initial number of packs (to be proven)
  let initial_packs : ℕ := 4

  -- Prove that the initial number of packs is 4
  have h : (cost_per_pack * (initial_packs + additional_packs : ℚ)) = total_value := by sorry

  -- Return the result
  exact initial_packs

end NUMINAMATH_CALUDE_initial_crayon_packs_l1205_120525


namespace NUMINAMATH_CALUDE_evaluate_g_l1205_120517

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 4 - 2 * g (-2) = 47 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l1205_120517


namespace NUMINAMATH_CALUDE_min_total_diff_three_students_l1205_120599

/-- Represents a student's ability characteristics as a list of 12 binary values -/
def Student := List Bool

/-- Calculates the number of different ability characteristics between two students -/
def diffCount (a b : Student) : Nat :=
  List.sum (List.map (fun (x, y) => if x = y then 0 else 1) (List.zip a b))

/-- Checks if two students have a significant comprehensive ability difference -/
def significantDiff (a b : Student) : Prop :=
  diffCount a b ≥ 7

/-- Calculates the total number of different ability characteristics among three students -/
def totalDiff (a b c : Student) : Nat :=
  diffCount a b + diffCount b c + diffCount c a

/-- Theorem: The minimum total number of different ability characteristics among three students
    with significant differences between each pair is 22 -/
theorem min_total_diff_three_students (a b c : Student) :
  (List.length a = 12 ∧ List.length b = 12 ∧ List.length c = 12) →
  (significantDiff a b ∧ significantDiff b c ∧ significantDiff c a) →
  totalDiff a b c ≥ 22 ∧ ∃ (x y z : Student), totalDiff x y z = 22 :=
sorry

end NUMINAMATH_CALUDE_min_total_diff_three_students_l1205_120599


namespace NUMINAMATH_CALUDE_kathryn_remaining_money_l1205_120505

def kathryn_finances (salary : ℕ) (rent : ℕ) (food_travel : ℕ) (shared_rent : ℕ) : Prop :=
  salary = 5000 ∧
  rent = 1200 ∧
  food_travel = 2 * rent ∧
  shared_rent = rent / 2 ∧
  salary - (shared_rent + food_travel) = 2000

theorem kathryn_remaining_money :
  ∃ (salary rent food_travel shared_rent : ℕ),
    kathryn_finances salary rent food_travel shared_rent :=
by
  sorry

end NUMINAMATH_CALUDE_kathryn_remaining_money_l1205_120505


namespace NUMINAMATH_CALUDE_rectangle_thirteen_squares_l1205_120553

/-- A rectangle can be divided into 13 equal squares if and only if its side ratio is 13:1 or 1:13 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ (a = s ∧ b = 13 * s ∨ a = 13 * s ∧ b = s)) ↔
  (a * b = 13 * (a.min b) * (a.min b)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_thirteen_squares_l1205_120553


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1205_120580

theorem inequality_system_integer_solutions (x : ℤ) :
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1205_120580


namespace NUMINAMATH_CALUDE_division_problem_l1205_120526

theorem division_problem (dividend quotient remainder : ℕ) (h : dividend = 162 ∧ quotient = 9 ∧ remainder = 9) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1205_120526


namespace NUMINAMATH_CALUDE_power_mod_23_l1205_120575

theorem power_mod_23 : 17^2001 % 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l1205_120575


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1205_120582

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.9166666666666666)) :
  x / y = 12 :=
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1205_120582


namespace NUMINAMATH_CALUDE_negation_equivalence_l1205_120592

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1205_120592


namespace NUMINAMATH_CALUDE_log_inequalities_l1205_120544

theorem log_inequalities (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x < x - 1) ∧ (Real.log x > (x - 1) / x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequalities_l1205_120544


namespace NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1205_120506

-- Define the types of polygons
inductive Polygon
  | RegularPentagon
  | IrregularHexagon
  | RegularHexagon
  | IrregularPentagon
  | EquilateralTriangle

-- Function to get the number of lines of symmetry for each polygon
def linesOfSymmetry (p : Polygon) : ℕ :=
  match p with
  | Polygon.RegularPentagon => 5
  | Polygon.IrregularHexagon => 0
  | Polygon.RegularHexagon => 6
  | Polygon.IrregularPentagon => 0
  | Polygon.EquilateralTriangle => 3

-- Theorem stating that the regular hexagon has the most lines of symmetry
theorem regular_hexagon_most_symmetry :
  ∀ p : Polygon, linesOfSymmetry Polygon.RegularHexagon ≥ linesOfSymmetry p :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_most_symmetry_l1205_120506


namespace NUMINAMATH_CALUDE_certain_number_proof_l1205_120591

theorem certain_number_proof (x : ℝ) : 
  (x / 10) - (x / 2000) = 796 → x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1205_120591


namespace NUMINAMATH_CALUDE_eight_elevenths_rounded_l1205_120528

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(⌊q * 10^n + 1/2⌋)) / 10^n

-- State the theorem
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 2 = 73/100 := by
  sorry

end NUMINAMATH_CALUDE_eight_elevenths_rounded_l1205_120528


namespace NUMINAMATH_CALUDE_well_depth_calculation_l1205_120587

/-- The depth of the well in feet -/
def well_depth : ℝ := 1255.64

/-- The time it takes for the stone to hit the bottom and the sound to reach the top -/
def total_time : ℝ := 10

/-- The gravitational constant for the stone's fall -/
def gravity_constant : ℝ := 16

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1100

/-- The function describing the stone's fall distance after t seconds -/
def stone_fall (t : ℝ) : ℝ := gravity_constant * t^2

/-- Theorem stating that the calculated well depth is correct given the conditions -/
theorem well_depth_calculation :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧ 
    t_fall + (well_depth / sound_velocity) = total_time ∧ 
    stone_fall t_fall = well_depth := by
  sorry

end NUMINAMATH_CALUDE_well_depth_calculation_l1205_120587


namespace NUMINAMATH_CALUDE_expected_rolls_for_2010_l1205_120556

/-- Represents the probability of getting a certain sum with a fair six-sided die -/
def probability (n : ℕ) : ℚ :=
  sorry

/-- Represents the expected number of rolls to reach a sum of n with a fair six-sided die -/
def expected_rolls (n : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the expected number of rolls to reach a sum of 2010 -/
theorem expected_rolls_for_2010 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000000 ∧ 
  abs (expected_rolls 2010 - 574523809/1000000) < ε :=
sorry

end NUMINAMATH_CALUDE_expected_rolls_for_2010_l1205_120556


namespace NUMINAMATH_CALUDE_sum_of_roots_l1205_120568

theorem sum_of_roots (a b c d k : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  k > 0 →
  a + b = k →
  c + d = k^2 →
  c^2 - 4*a*c - 5*b = 0 →
  d^2 - 4*a*d - 5*b = 0 →
  a^2 - 4*c*a - 5*d = 0 →
  b^2 - 4*c*b - 5*d = 0 →
  a + b + c + d = k + k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1205_120568


namespace NUMINAMATH_CALUDE_scarf_wool_calculation_l1205_120557

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := sorry

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem scarf_wool_calculation :
  scarf_wool * aaron_scarves + 
  sweater_wool * (aaron_sweaters + enid_sweaters) = 
  total_wool ∧ scarf_wool = 3 := by sorry

end NUMINAMATH_CALUDE_scarf_wool_calculation_l1205_120557
