import Mathlib

namespace directrix_of_parabola_l1366_136641

-- Define the parabola and the line conditions
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus_line (x y : ℝ) := 2 * x + 3 * y - 8 = 0

-- Theorem stating that the directrix of the parabola is x = -4
theorem directrix_of_parabola (p : ℝ) (hx : ∃ x, ∃ y, focus_line x y) (hp : parabola p) :
  ∃ k : ℝ, k = 4 → ∀ x y : ℝ, (-x) = -4 :=
by
  sorry

end directrix_of_parabola_l1366_136641


namespace china_GDP_in_2016_l1366_136698

noncomputable def GDP_2016 (a r : ℝ) : ℝ := a * (1 + r / 100)^5

theorem china_GDP_in_2016 (a r : ℝ) :
  GDP_2016 a r = a * (1 + r / 100)^5 :=
by
  -- proof
  sorry

end china_GDP_in_2016_l1366_136698


namespace find_k_l1366_136650

theorem find_k (k : ℝ) (h : 0.5 * |-2 * k| * |k| = 1) : k = 1 ∨ k = -1 :=
sorry

end find_k_l1366_136650


namespace inequality_proof_l1366_136670

theorem inequality_proof (a b c d : ℕ) (h₀: a + c ≤ 1982) (h₁: (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)) (h₂: (a:ℚ)/b + (c:ℚ)/d < 1) :
  1 - (a:ℚ)/b - (c:ℚ)/d > 1 / (1983 ^ 3) :=
sorry

end inequality_proof_l1366_136670


namespace perimeter_of_resulting_figure_l1366_136680

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l1366_136680


namespace grid_satisfies_conditions_l1366_136638

-- Define the range of consecutive integers
def consecutive_integers : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Define a function to check mutual co-primality condition for two given numbers
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the 3x3 grid arrangement as a matrix
@[reducible]
def grid : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![8, 9, 10],
    ![5, 7, 11],
    ![6, 13, 12]]

-- Define a predicate to check if the grid cells are mutually coprime for adjacent elements
def mutually_coprime_adjacent (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
    ∀ (i j k l : Fin 3), 
    (|i - k| + |j - l| = 1 ∨ |i - k| = |j - l| ∧ |i - k| = 1) → coprime (m i j) (m k l)

-- The main theorem statement
theorem grid_satisfies_conditions : 
  (∀ (i j : Fin 3), grid i j ∈ consecutive_integers) ∧ 
  mutually_coprime_adjacent grid :=
by
  sorry

end grid_satisfies_conditions_l1366_136638


namespace calculate_distance_to_friend_l1366_136648

noncomputable def distance_to_friend (d t : ℝ) : Prop :=
  (d = 45 * (t + 1)) ∧ (d = 45 + 65 * (t - 0.75))

theorem calculate_distance_to_friend : ∃ d t: ℝ, distance_to_friend d t ∧ d = 155 :=
by
  exists 155
  exists 2.4375
  sorry

end calculate_distance_to_friend_l1366_136648


namespace solve_inequality_l1366_136639

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 < 12 - x ↔ 
  (Real.sqrt 5) ≠ 0 ∧
  (1 - 3 * (Real.sqrt 5)) / 2 < x ∧ 
  x < (1 + 3 * (Real.sqrt 5)) / 2 :=
sorry

end solve_inequality_l1366_136639


namespace pentagonal_number_formula_l1366_136664

def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n + 1)) / 2

theorem pentagonal_number_formula (n : ℕ) :
  pentagonal_number n = (n * (3 * n + 1)) / 2 :=
by
  sorry

end pentagonal_number_formula_l1366_136664


namespace count_positive_integers_l1366_136666

theorem count_positive_integers (n : ℕ) (x : ℝ) (h1 : n ≤ 1500) :
  (∃ x : ℝ, n = ⌊x⌋ + ⌊3*x⌋ + ⌊5*x⌋) ↔ n = 668 :=
by
  sorry

end count_positive_integers_l1366_136666


namespace gcd_equation_solution_l1366_136692

theorem gcd_equation_solution (x y : ℕ) (h : Nat.gcd x y + x * y / Nat.gcd x y = x + y) : y ∣ x ∨ x ∣ y :=
 by
 sorry

end gcd_equation_solution_l1366_136692


namespace calculate_three_Z_five_l1366_136607

def Z (a b : ℤ) : ℤ := b + 15 * a - a^3

theorem calculate_three_Z_five : Z 3 5 = 23 :=
by
  -- The proof goes here
  sorry

end calculate_three_Z_five_l1366_136607


namespace positive_integers_sequence_l1366_136659

theorem positive_integers_sequence (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a ∣ (b + c + d)) (h5 : b ∣ (a + c + d)) 
  (h6 : c ∣ (a + b + d)) (h7 : d ∣ (a + b + c)) : 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 9) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 8 ∧ d = 12) ∨ 
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 10) ∨ 
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 21) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 10 ∧ d = 15) :=
sorry

end positive_integers_sequence_l1366_136659


namespace find_m_l1366_136647

def is_ellipse (x y m : ℝ) : Prop :=
  (x^2 / (m + 1) + y^2 / m = 1)

def has_eccentricity (e : ℝ) (m : ℝ) : Prop :=
  e = Real.sqrt (1 - m / (m + 1))

theorem find_m (m : ℝ) (h_m : m > 0) (h_ellipse : ∀ x y, is_ellipse x y m) (h_eccentricity : has_eccentricity (1 / 2) m) : m = 3 :=
by
  sorry

end find_m_l1366_136647


namespace esther_biking_speed_l1366_136610

theorem esther_biking_speed (d x : ℝ)
  (h_bike_speed : x > 0)
  (h_average_speed : 5 = 2 * d / (d / x + d / 3)) :
  x = 15 :=
by
  sorry

end esther_biking_speed_l1366_136610


namespace base13_addition_l1366_136652

/--
Given two numbers in base 13: 528₁₃ and 274₁₃, prove that their sum is 7AC₁₃.
-/
theorem base13_addition :
  let u1 := 8
  let t1 := 2
  let h1 := 5
  let u2 := 4
  let t2 := 7
  let h2 := 2
  -- Add the units digits: 8 + 4 = 12; 12 is C in base 13
  let s1 := 12 -- 'C' in base 13
  let carry1 := 1
  -- Add the tens digits along with the carry: 2 + 7 + 1 = 10; 10 is A in base 13
  let s2 := 10 -- 'A' in base 13
  -- Add the hundreds digits: 5 + 2 = 7
  let s3 := 7 -- 7 in base 13
  s1 = 12 ∧ s2 = 10 ∧ s3 = 7 :=
by
  sorry

end base13_addition_l1366_136652


namespace sequence_monotonic_b_gt_neg3_l1366_136665

theorem sequence_monotonic_b_gt_neg3 (b : ℝ) :
  (∀ n : ℕ, n > 0 → (n+1)^2 + b*(n+1) > n^2 + b*n) ↔ b > -3 :=
by sorry

end sequence_monotonic_b_gt_neg3_l1366_136665


namespace valid_configuration_exists_l1366_136601

noncomputable def unique_digits (digits: List ℕ) := (digits.length = List.length (List.eraseDup digits)) ∧ ∀ (d : ℕ), d ∈ digits ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem valid_configuration_exists :
  ∃ a b c d e f g h i j : ℕ,
  unique_digits [a, b, c, d, e, f, g, h, i, j] ∧
  a * (100 * b + 10 * c + d) * (100 * e + 10 * f + g) = 1000 * h + 100 * i + 10 * 9 + 71 := 
by
  sorry

end valid_configuration_exists_l1366_136601


namespace binom_1300_2_eq_844350_l1366_136602

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l1366_136602


namespace min_distance_PQ_l1366_136629

theorem min_distance_PQ :
  let P_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = 4
  let Q_circle (ρ θ : ℝ) := ρ^2 = 4 * ρ * Real.cos θ - 3
  ∃ (P Q : ℝ × ℝ), 
    (∃ ρP θP, P = (ρP * Real.cos θP, ρP * Real.sin θP) ∧ P_line ρP θP) ∧
    (∃ ρQ θQ, Q = (ρQ * Real.cos θQ, ρQ * Real.sin θQ) ∧ Q_circle ρQ θQ) ∧
    ∀ R S : ℝ × ℝ, 
      (∃ ρR θR, R = (ρR * Real.cos θR, ρR * Real.sin θR) ∧ P_line ρR θR) →
      (∃ ρS θS, S = (ρS * Real.cos θS, ρS * Real.sin θS) ∧ Q_circle ρS θS) →
      dist P Q ≤ dist R S :=
  sorry

end min_distance_PQ_l1366_136629


namespace early_time_l1366_136645

noncomputable def speed1 : ℝ := 5 -- km/hr
noncomputable def timeLate : ℝ := 5 / 60 -- convert minutes to hours
noncomputable def speed2 : ℝ := 10 -- km/hr
noncomputable def distance : ℝ := 2.5 -- km

theorem early_time (speed1 speed2 distance : ℝ) (timeLate : ℝ) :
  (distance / speed1 - timeLate) * 60 - (distance / speed2) * 60 = 10 :=
by
  sorry

end early_time_l1366_136645


namespace spherical_to_rectangular_coords_l1366_136603

theorem spherical_to_rectangular_coords
  (ρ θ φ : ℝ)
  (hρ : ρ = 6)
  (hθ : θ = 7 * Real.pi / 4)
  (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = -3 * Real.sqrt 6 ∧ y = -3 * Real.sqrt 6 ∧ z = 3 :=
by
  sorry

end spherical_to_rectangular_coords_l1366_136603


namespace greatest_q_minus_r_l1366_136613

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1001 = 17 * q + r ∧ q - r = 43 :=
by
  sorry

end greatest_q_minus_r_l1366_136613


namespace calculate_expression_l1366_136686

theorem calculate_expression : (0.25)^(-0.5) + (1/27)^(-1/3) - 625^(0.25) = 0 := 
by 
  sorry

end calculate_expression_l1366_136686


namespace dinner_guest_arrangement_l1366_136679

noncomputable def number_of_ways (n k : ℕ) : ℕ :=
  if n < k then 0 else Nat.factorial n / Nat.factorial (n - k)

theorem dinner_guest_arrangement :
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements = 5040 :=
by
  -- Definitions and preliminary calculations
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements

  -- This is where the proof would go, but we insert sorry to skip it for now
  sorry

end dinner_guest_arrangement_l1366_136679


namespace equal_cubes_l1366_136605

theorem equal_cubes (r s : ℤ) (hr : 0 ≤ r) (hs : 0 ≤ s)
  (h : |r^3 - s^3| = |6 * r^2 - 6 * s^2|) : r = s :=
by
  sorry

end equal_cubes_l1366_136605


namespace trigonometric_inequality_C_trigonometric_inequality_D_l1366_136678

theorem trigonometric_inequality_C (x : Real) : Real.cos (3*Real.pi/5) > Real.cos (-4*Real.pi/5) :=
by
  sorry

theorem trigonometric_inequality_D (y : Real) : Real.sin (Real.pi/10) < Real.cos (Real.pi/10) :=
by
  sorry

end trigonometric_inequality_C_trigonometric_inequality_D_l1366_136678


namespace nuts_in_tree_l1366_136621

theorem nuts_in_tree (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : squirrels = nuts + 2) : nuts = 2 :=
by
  sorry

end nuts_in_tree_l1366_136621


namespace n_power_2020_plus_4_composite_l1366_136675

theorem n_power_2020_plus_4_composite {n : ℕ} (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b := 
by
  sorry

end n_power_2020_plus_4_composite_l1366_136675


namespace complement_intersection_l1366_136624

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) (hM : M = {1, 3, 5, 7}) (hN : N = {2, 5, 8}) :
  (U \ M) ∩ N = {2, 8} :=
by
  sorry

end complement_intersection_l1366_136624


namespace sqrt_18_mul_sqrt_32_eq_24_l1366_136615
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l1366_136615


namespace two_sin_cos_15_eq_half_l1366_136633

open Real

theorem two_sin_cos_15_eq_half : 2 * sin (π / 12) * cos (π / 12) = 1 / 2 :=
by
  sorry

end two_sin_cos_15_eq_half_l1366_136633


namespace complex_mul_l1366_136687

theorem complex_mul (i : ℂ) (hi : i * i = -1) : (1 - i) * (3 + i) = 4 - 2 * i :=
by
  sorry

end complex_mul_l1366_136687


namespace line_through_points_a_minus_b_l1366_136699

theorem line_through_points_a_minus_b :
  ∃ a b : ℝ, 
  (∀ x, (x = 3 → 7 = a * 3 + b) ∧ (x = 6 → 19 = a * 6 + b)) → 
  a - b = 9 :=
by
  sorry

end line_through_points_a_minus_b_l1366_136699


namespace percent_asian_population_in_West_l1366_136623

-- Define the populations in different regions
def population_NE := 2
def population_MW := 3
def population_South := 4
def population_West := 10

-- Define the total population
def total_population := population_NE + population_MW + population_South + population_West

-- Calculate the percentage of the population in the West
def percentage_in_West := (population_West * 100) / total_population

-- The proof statement
theorem percent_asian_population_in_West : percentage_in_West = 53 := by
  sorry -- proof to be completed

end percent_asian_population_in_West_l1366_136623


namespace no_such_k_l1366_136620

theorem no_such_k (u : ℕ → ℝ) (v : ℕ → ℝ)
  (h1 : u 0 = 6) (h2 : v 0 = 4)
  (h3 : ∀ n, u (n + 1) = (3 / 5) * u n - (4 / 5) * v n)
  (h4 : ∀ n, v (n + 1) = (4 / 5) * u n + (3 / 5) * v n) :
  ¬ ∃ k, u k = 7 ∧ v k = 2 :=
by
  sorry

end no_such_k_l1366_136620


namespace total_toothpicks_for_grid_l1366_136657

-- Defining the conditions
def grid_height := 30
def grid_width := 15

-- Define the function that calculates the total number of toothpicks
def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  horizontal_toothpicks + vertical_toothpicks

-- The theorem stating the problem and its answer
theorem total_toothpicks_for_grid : total_toothpicks grid_height grid_width = 945 :=
by {
  -- Here we would write the proof steps. Using sorry for now.
  sorry
}

end total_toothpicks_for_grid_l1366_136657


namespace solve_for_a_l1366_136619

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x >= 0 then 4 ^ x else 2 ^ (a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 := 
by {
  sorry
}

end solve_for_a_l1366_136619


namespace vector_addition_correct_l1366_136668

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A]

def vector_addition (da cd cb ba : A) : Prop :=
  da + cd - cb = ba

theorem vector_addition_correct (da cd cb ba : A) :
  vector_addition da cd cb ba :=
  sorry

end vector_addition_correct_l1366_136668


namespace find_number_l1366_136616

theorem find_number :
  (∃ x : ℝ, x * (3 + Real.sqrt 5) = 1) ∧ (x = (3 - Real.sqrt 5) / 4) :=
sorry

end find_number_l1366_136616


namespace f_of_3_l1366_136673

def f (x : ℚ) : ℚ := (x + 3) / (x - 6)

theorem f_of_3 : f 3 = -2 := by
  sorry

end f_of_3_l1366_136673


namespace problem_1_problem_2_l1366_136672

theorem problem_1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : abc ≤ 3 * Real.sqrt 3 := 
sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : 
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) > (a + b + c) / 3 := 
sorry

end problem_1_problem_2_l1366_136672


namespace half_abs_diff_squares_l1366_136608

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end half_abs_diff_squares_l1366_136608


namespace minimum_value_of_fraction_l1366_136655

theorem minimum_value_of_fraction (x : ℝ) (h : x > 0) : 
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 - 1 ∧ ∀ y, y = (x^2 + x + 3) / (x + 1) -> y ≥ m :=
sorry

end minimum_value_of_fraction_l1366_136655


namespace quadratic_roots_range_l1366_136642

theorem quadratic_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - (2 * m + 1) * x + m^2 = 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 - (2 * m + 1) * y + m^2 = 0)) ↔ m > -1 / 4 :=
by sorry

end quadratic_roots_range_l1366_136642


namespace brendan_remaining_money_l1366_136684

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l1366_136684


namespace certain_number_l1366_136625

theorem certain_number (N : ℝ) (k : ℝ) 
  (h1 : (1 / 2) ^ 22 * N ^ k = 1 / 18 ^ 22) 
  (h2 : k = 11) 
  : N = 81 := 
by
  sorry

end certain_number_l1366_136625


namespace functional_equation_zero_solution_l1366_136683

theorem functional_equation_zero_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_zero_solution_l1366_136683


namespace sale_price_is_correct_l1366_136656

def original_price : ℝ := 100
def percentage_decrease : ℝ := 0.30
def sale_price : ℝ := original_price * (1 - percentage_decrease)

theorem sale_price_is_correct : sale_price = 70 := by
  sorry

end sale_price_is_correct_l1366_136656


namespace second_hand_bisect_angle_l1366_136626

theorem second_hand_bisect_angle :
  ∃ x : ℚ, (6 * x - 360 * (x - 1) = 360 * (x - 1) - 0.5 * x) ∧ (x = 1440 / 1427) :=
by
  sorry

end second_hand_bisect_angle_l1366_136626


namespace ticket_cost_at_30_years_l1366_136653

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l1366_136653


namespace num_true_statements_is_two_l1366_136694

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem num_true_statements_is_two :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end num_true_statements_is_two_l1366_136694


namespace probability_two_points_square_l1366_136622

def gcd (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c  

theorem probability_two_points_square {a b c : ℕ} (hx : gcd a b c = 1)
  (h : (26 - Real.pi) / 32 = (a - b * Real.pi) / c) : a + b + c = 59 :=
by
  sorry

end probability_two_points_square_l1366_136622


namespace average_class_score_l1366_136696

theorem average_class_score : 
  ∀ (n total score_per_100 score_per_0 avg_rest : ℕ), 
  n = 20 → 
  total = 800 → 
  score_per_100 = 2 → 
  score_per_0 = 3 → 
  avg_rest = 40 → 
  ((score_per_100 * 100 + score_per_0 * 0 + (n - (score_per_100 + score_per_0)) * avg_rest) / n = 40)
:= by
  intros n total score_per_100 score_per_0 avg_rest h_n h_total h_100 h_0 h_rest
  sorry

end average_class_score_l1366_136696


namespace lcm_of_numbers_l1366_136674

/-- Define the numbers involved -/
def a := 456
def b := 783
def c := 935
def d := 1024
def e := 1297

/-- Prove the LCM of these numbers is 2308474368000 -/
theorem lcm_of_numbers :
  Int.lcm (Int.lcm (Int.lcm (Int.lcm a b) c) d) e = 2308474368000 :=
by
  sorry

end lcm_of_numbers_l1366_136674


namespace power_rule_example_l1366_136654

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l1366_136654


namespace pythagorean_theorem_l1366_136637

theorem pythagorean_theorem (a b c : ℕ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l1366_136637


namespace compute_exp_l1366_136649

theorem compute_exp : 3 * 3^4 + 9^30 / 9^28 = 324 := 
by sorry

end compute_exp_l1366_136649


namespace medians_concurrent_l1366_136636

/--
For any triangle ABC, there exists a point G, known as the centroid, such that
the sum of the vectors from G to each of the vertices A, B, and C is the zero vector.
-/
theorem medians_concurrent 
  (A B C : ℝ×ℝ) : 
  ∃ G : ℝ×ℝ, (G -ᵥ A) + (G -ᵥ B) + (G -ᵥ C) = (0, 0) := 
by 
  -- proof will go here
  sorry 

end medians_concurrent_l1366_136636


namespace symmetric_point_l1366_136630

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1: P = (2, 1)) (h2 : x - y + 1 = 0) :
  (b - 1) = -(a - 2) ∧ (a + 2) / 2 - (b + 1) / 2 + 1 = 0 → (a, b) = (0, 3) := 
sorry

end symmetric_point_l1366_136630


namespace total_production_by_june_l1366_136682

def initial_production : ℕ := 10

def common_ratio : ℕ := 3

def production_june : ℕ :=
  let a := initial_production
  let r := common_ratio
  a * ((r^6 - 1) / (r - 1))

theorem total_production_by_june : production_june = 3640 :=
by sorry

end total_production_by_june_l1366_136682


namespace isosceles_triangle_perimeter_l1366_136604

-- Define the lengths of the sides of the isosceles triangle
def side1 : ℕ := 12
def side2 : ℕ := 12
def base : ℕ := 17

-- Define the perimeter as the sum of all three sides
def perimeter : ℕ := side1 + side2 + base

-- State the theorem that needs to be proved
theorem isosceles_triangle_perimeter : perimeter = 41 := by
  -- Insert the proof here
  sorry

end isosceles_triangle_perimeter_l1366_136604


namespace shaded_region_area_l1366_136627

theorem shaded_region_area (r : ℝ) (π : ℝ) (h1 : r = 5) : 
  4 * ((1/2 * π * r * r) - (1/2 * r * r)) = 50 * π - 50 :=
by 
  sorry

end shaded_region_area_l1366_136627


namespace count_solutions_g_composition_eq_l1366_136693

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the main theorem
theorem count_solutions_g_composition_eq :
  ∃ (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
by
  sorry

end count_solutions_g_composition_eq_l1366_136693


namespace even_product_implies_even_factor_l1366_136685

theorem even_product_implies_even_factor (a b : ℕ) (h : Even (a * b)) : Even a ∨ Even b :=
by
  sorry

end even_product_implies_even_factor_l1366_136685


namespace largest_y_value_l1366_136688

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l1366_136688


namespace find_number_l1366_136660

theorem find_number (S Q R N : ℕ) (hS : S = 555 + 445) (hQ : Q = 2 * (555 - 445)) (hR : R = 50) (h_eq : N = S * Q + R) :
  N = 220050 :=
by
  rw [hS, hQ, hR] at h_eq
  norm_num at h_eq
  exact h_eq

end find_number_l1366_136660


namespace find_teachers_and_students_l1366_136635

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l1366_136635


namespace total_photos_l1366_136614

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l1366_136614


namespace value_of_m_l1366_136697

theorem value_of_m (z1 z2 m : ℝ) (h1 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z1 = 0)
  (h2 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z2 = 0)
  (h3 : |z1 - z2| = 3) : m = 4 ∨ m = 17 / 2 := sorry

end value_of_m_l1366_136697


namespace circle_complete_the_square_l1366_136632

/-- Given the equation x^2 - 6x + y^2 - 10y + 18 = 0, show that it can be transformed to  
    (x - 3)^2 + (y - 5)^2 = 4^2 -/
theorem circle_complete_the_square :
  ∀ x y : ℝ, x^2 - 6 * x + y^2 - 10 * y + 18 = 0 ↔ (x - 3)^2 + (y - 5)^2 = 4^2 :=
by
  sorry

end circle_complete_the_square_l1366_136632


namespace simplify_trig_expression_tan_alpha_value_l1366_136677

-- Proof Problem (1)
theorem simplify_trig_expression :
  (∃ θ : ℝ, θ = (20:ℝ) ∧ 
    (∃ α : ℝ, α = (160:ℝ) ∧ 
      (∃ β : ℝ, β = 1 - 2 * (Real.sin θ) * (Real.cos θ) ∧ 
        (∃ γ : ℝ, γ = 1 - (Real.sin θ)^2 ∧ 
          (Real.sqrt β) / ((Real.sin α) - (Real.sqrt γ)) = -1)))) :=
sorry

-- Proof Problem (2)
theorem tan_alpha_value (α : ℝ) (h : Real.tan α = 1 / 3) :
  1 / (4 * (Real.cos α)^2 - 6 * (Real.sin α) * (Real.cos α)) = 5 / 9 :=
sorry

end simplify_trig_expression_tan_alpha_value_l1366_136677


namespace rectangle_new_area_l1366_136606

theorem rectangle_new_area (l w : ℝ) (h_area : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 497 :=
by
  sorry

end rectangle_new_area_l1366_136606


namespace find_k_l1366_136646

-- Define the vectors a, b, and c
def vecA : ℝ × ℝ := (2, -1)
def vecB : ℝ × ℝ := (1, 1)
def vecC : ℝ × ℝ := (-5, 1)

-- Define the condition for two vectors being parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- Define the target statement to be proven
theorem find_k : ∃ k : ℝ, parallel (vecA.1 + k * vecB.1, vecA.2 + k * vecB.2) vecC ∧ k = 1/2 := 
sorry

end find_k_l1366_136646


namespace geometric_sequence_sum_l1366_136651

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 7) 
  (h2 : a 2 + a 3 + a 4 = 14) 
  (geom_seq : ∃ q, ∀ n, a (n + 1) = q * a n ∧ q = 2) :
  a 4 + a 5 + a 6 = 56 := 
by
  sorry

end geometric_sequence_sum_l1366_136651


namespace find_t_l1366_136643

def utility (hours_math hours_reading hours_painting : ℕ) : ℕ :=
  hours_math^2 + hours_reading * hours_painting

def utility_wednesday (t : ℕ) : ℕ :=
  utility 4 t (12 - t)

def utility_thursday (t : ℕ) : ℕ :=
  utility 3 (t + 1) (11 - t)

theorem find_t (t : ℕ) (h : utility_wednesday t = utility_thursday t) : t = 2 :=
by
  sorry

end find_t_l1366_136643


namespace expected_waiting_time_l1366_136676

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l1366_136676


namespace calculate_f3_times_l1366_136662

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4 * n + 2

theorem calculate_f3_times : f (f (f 3)) = 170 := by
  sorry

end calculate_f3_times_l1366_136662


namespace find_value_l1366_136690

theorem find_value (a b c : ℝ) (h1 : a + b = 8) (h2 : a * b = c^2 + 16) : a + 2 * b + 3 * c = 12 := by
  sorry

end find_value_l1366_136690


namespace intersection_of_sets_l1366_136669

theorem intersection_of_sets {A B : Set Nat} (hA : A = {1, 3, 9}) (hB : B = {1, 5, 9}) :
  A ∩ B = {1, 9} :=
sorry

end intersection_of_sets_l1366_136669


namespace remainder_31_31_plus_31_mod_32_l1366_136681

theorem remainder_31_31_plus_31_mod_32 : (31 ^ 31 + 31) % 32 = 30 := 
by sorry

end remainder_31_31_plus_31_mod_32_l1366_136681


namespace no_natural_number_solution_for_divisibility_by_2020_l1366_136663

theorem no_natural_number_solution_for_divisibility_by_2020 :
  ¬ ∃ k : ℕ, (k^3 - 3 * k^2 + 2 * k + 2) % 2020 = 0 :=
sorry

end no_natural_number_solution_for_divisibility_by_2020_l1366_136663


namespace proposition_4_correct_l1366_136609

section

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Definitions of perpendicular and parallel relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (x y : Line) : Prop := sorry

theorem proposition_4_correct (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end

end proposition_4_correct_l1366_136609


namespace time_spent_on_type_a_problems_l1366_136628

-- Define the conditions
def total_questions := 200
def examination_duration_hours := 3
def type_a_problems := 100
def type_b_problems := total_questions - type_a_problems
def type_a_time_coeff := 2

-- Convert examination duration to minutes
def examination_duration_minutes := examination_duration_hours * 60

-- Variables for time per problem
variable (x : ℝ)

-- The total time spent
def total_time_spent : ℝ := type_a_problems * (type_a_time_coeff * x) + type_b_problems * x

-- Statement we need to prove
theorem time_spent_on_type_a_problems :
  total_time_spent x = examination_duration_minutes → type_a_problems * (type_a_time_coeff * x) = 120 :=
by
  sorry

end time_spent_on_type_a_problems_l1366_136628


namespace quadratic_condition_l1366_136617

theorem quadratic_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 3 = 0) → a ≠ 1 :=
by
  sorry

end quadratic_condition_l1366_136617


namespace circle_center_distance_l1366_136691

theorem circle_center_distance (R : ℝ) : 
  ∃ (d : ℝ), 
  (∀ (θ : ℝ), θ = 30 → 
  ∀ (r : ℝ), r = 2.5 →
  ∀ (center_on_other_side : ℝ), center_on_other_side = R + R →
  d = 5) :=
by 
  use 5
  intros θ θ_eq r r_eq center_on_other_side center_eq
  sorry

end circle_center_distance_l1366_136691


namespace square_perimeter_l1366_136611

theorem square_perimeter (x : ℝ) (h : x * x + x * x = (2 * Real.sqrt 2) * (2 * Real.sqrt 2)) :
    4 * x = 8 :=
by
  sorry

end square_perimeter_l1366_136611


namespace white_balls_in_bag_l1366_136661

theorem white_balls_in_bag : 
  ∀ x : ℕ, (3 + 2 + x ≠ 0) → (2 : ℚ) / (3 + 2 + x) = 1 / 4 → x = 3 :=
by
  intro x
  intro h1
  intro h2
  sorry

end white_balls_in_bag_l1366_136661


namespace inverse_mod_53_l1366_136612

theorem inverse_mod_53 (h : 17 * 13 % 53 = 1) : 36 * 40 % 53 = 1 :=
by
  -- Given condition: 17 * 13 % 53 = 1
  -- Derived condition: (-17) * -13 % 53 = 1 which is equivalent to 17 * 13 % 53 = 1
  -- So we need to find: 36 * x % 53 = 1 where x = -13 % 53 => x = 40
  sorry

end inverse_mod_53_l1366_136612


namespace problem_statement_l1366_136640

noncomputable def count_valid_numbers : Nat :=
  let digits := [1, 2, 3, 4, 5]
  let repeated_digit_choices := 5
  let positions_for_repeated_digits := Nat.choose 5 2
  let cases_for_tens_and_hundreds :=
    2 * 3 + 2 + 1
  let two_remaining_digits_permutations := 2
  repeated_digit_choices * positions_for_repeated_digits * cases_for_tens_and_hundreds * two_remaining_digits_permutations

theorem problem_statement : count_valid_numbers = 800 := by
  sorry

end problem_statement_l1366_136640


namespace stock_yield_calculation_l1366_136695

theorem stock_yield_calculation (par_value market_value annual_dividend : ℝ)
  (h1 : par_value = 100)
  (h2 : market_value = 80)
  (h3 : annual_dividend = 0.04 * par_value) :
  (annual_dividend / market_value) * 100 = 5 :=
by
  sorry

end stock_yield_calculation_l1366_136695


namespace division_result_l1366_136689

open Polynomial

noncomputable def dividend := (X ^ 6 - 5 * X ^ 4 + 3 * X ^ 3 - 7 * X ^ 2 + 2 * X - 8 : Polynomial ℤ)
noncomputable def divisor := (X - 3 : Polynomial ℤ)
noncomputable def expected_quotient := (X ^ 5 + 3 * X ^ 4 + 4 * X ^ 3 + 15 * X ^ 2 + 38 * X + 116 : Polynomial ℤ)
noncomputable def expected_remainder := (340 : ℤ)

theorem division_result : (dividend /ₘ divisor) = expected_quotient ∧ (dividend %ₘ divisor) = C expected_remainder := by
  sorry

end division_result_l1366_136689


namespace walking_time_estimate_l1366_136658

-- Define constants for distance, speed, and time conversion factor
def distance : ℝ := 1000
def speed : ℝ := 4000
def time_conversion : ℝ := 60

-- Define the expected time to walk from home to school in minutes
def expected_time : ℝ := 15

-- Prove the time calculation
theorem walking_time_estimate : (distance / speed) * time_conversion = expected_time :=
by
  sorry

end walking_time_estimate_l1366_136658


namespace solution_1_solution_2_l1366_136634

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |x - 3|

theorem solution_1 (x : ℝ) : (f x (-1) >= 2) ↔ (x >= 2) :=
by
  sorry

theorem solution_2 (a : ℝ) : 
  (∃ x : ℝ, f x a <= -(a / 2)) ↔ (a <= 2 ∨ a >= 6) :=
by
  sorry

end solution_1_solution_2_l1366_136634


namespace remainder_when_x_minus_y_div_18_l1366_136671

variable (k m : ℤ)
variable (x y : ℤ)
variable (h1 : x = 72 * k + 65)
variable (h2 : y = 54 * m + 22)

theorem remainder_when_x_minus_y_div_18 :
  (x - y) % 18 = 7 := by
sorry

end remainder_when_x_minus_y_div_18_l1366_136671


namespace equation_holds_l1366_136667

-- Positive integers less than 10
def is_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

theorem equation_holds (a b c : ℕ) (ha : is_lt_10 a) (hb : is_lt_10 b) (hc : is_lt_10 c) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 :=
by
  sorry

end equation_holds_l1366_136667


namespace sufficient_and_necessary_condition_l1366_136600

variable {a : ℕ → ℝ}
variable {a1 a2 : ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  (∀ n, a n = a1 * q ^ n)

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1))

theorem sufficient_and_necessary_condition
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h_geom : geometric_sequence a a1 q)
  (h_a1_pos : a1 > 0)
  (h_a1_lt_a2 : a1 < a1 * q) :
  increasing a ↔ a1 < a1 * q := 
sorry

end sufficient_and_necessary_condition_l1366_136600


namespace ten_men_ten_boys_work_time_l1366_136631

theorem ten_men_ten_boys_work_time :
  (∀ (total_work : ℝ) (man_work_rate boy_work_rate : ℝ),
    15 * 10 * man_work_rate = total_work ∧
    20 * 15 * boy_work_rate = total_work →
    (10 * man_work_rate + 10 * boy_work_rate) * 10 = total_work) :=
by
  sorry

end ten_men_ten_boys_work_time_l1366_136631


namespace find_a9_l1366_136644

variable (S : ℕ → ℚ) (a : ℕ → ℚ) (n : ℕ) (d : ℚ)

-- Conditions
axiom sum_first_six : S 6 = 3
axiom sum_first_eleven : S 11 = 18
axiom Sn_definition : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n)
axiom arithmetic_sequence : ∀ n, a (n + 1) = a 1 + n * d

-- Problem statement
theorem find_a9 : a 9 = 3 := sorry

end find_a9_l1366_136644


namespace sequence_an_general_formula_and_sum_bound_l1366_136618

theorem sequence_an_general_formula_and_sum_bound (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (1 / 4) * (a n + 1) ^ 2)
  (h2 : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h3 : ∀ n, T n = (1 / 2) * (1 - (1 / (2 * n + 1))))
  (h4 : ∀ n, 0 < a n) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n < 1 / 2) := 
by
  sorry

end sequence_an_general_formula_and_sum_bound_l1366_136618
