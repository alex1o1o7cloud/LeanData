import Mathlib

namespace functional_equation_solution_l631_63133

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) :=
sorry

end functional_equation_solution_l631_63133


namespace find_x_of_equation_l631_63184

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l631_63184


namespace probability_at_least_one_girl_l631_63162

theorem probability_at_least_one_girl (boys girls : ℕ) (total : ℕ) (choose_two : ℕ) : 
  boys = 3 → girls = 2 → total = boys + girls → choose_two = 2 → 
  1 - (Nat.choose boys choose_two) / (Nat.choose total choose_two) = 7 / 10 :=
by
  sorry

end probability_at_least_one_girl_l631_63162


namespace rooms_per_floor_l631_63127

-- Definitions for each of the conditions
def numberOfFloors : ℕ := 4
def hoursPerRoom : ℕ := 6
def hourlyRate : ℕ := 15
def totalEarnings : ℕ := 3600

-- Statement of the problem
theorem rooms_per_floor : 
  (totalEarnings / hourlyRate) / hoursPerRoom / numberOfFloors = 10 := 
  sorry

end rooms_per_floor_l631_63127


namespace li_to_zhang_l631_63112

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end li_to_zhang_l631_63112


namespace antifreeze_solution_l631_63126

theorem antifreeze_solution (x : ℝ) 
  (h1 : 26 * x + 13 * 0.54 = 39 * 0.58) : 
  x = 0.6 := 
by 
  sorry

end antifreeze_solution_l631_63126


namespace average_math_chemistry_l631_63159

variables (M P C : ℕ)

axiom h1 : M + P = 60
axiom h2 : C = P + 20

theorem average_math_chemistry : (M + C) / 2 = 40 :=
by
  sorry

end average_math_chemistry_l631_63159


namespace count_isosceles_triangles_perimeter_25_l631_63178

theorem count_isosceles_triangles_perimeter_25 : 
  ∃ n : ℕ, (
    n = 6 ∧ 
    (∀ x b : ℕ, 
      2 * x + b = 25 → 
      b < 2 * x → 
      b > 0 →
      ∃ m : ℕ, 
        m = (x - 7) / 5
    ) 
  ) := sorry

end count_isosceles_triangles_perimeter_25_l631_63178


namespace paving_path_DE_time_l631_63194

-- Define the conditions
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance for Worker 1
variable (d2 : ℝ) -- Total distance for Worker 2
variable (AD DE EF FC : ℝ) -- Distances in the path of Worker 2

-- Define the statement
theorem paving_path_DE_time :
  (AD + DE + EF + FC) = d2 ∧
  x = 9 * v ∧
  d2 = 10.8 * v ∧
  d2 = AD + DE + EF + FC ∧
  (∀ t, t = (DE / (1.2 * v)) * 60) ∧
  t = 45 :=
by
  sorry

end paving_path_DE_time_l631_63194


namespace find_x_l631_63186

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - q.1, p.2 + q.2)

theorem find_x (x y : ℤ) :
  star (3, 3) (0, 0) = star (x, y) (3, 2) → x = 6 :=
by
  intro h
  sorry

end find_x_l631_63186


namespace odd_integers_equality_l631_63168

-- Definitions
def is_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

def divides (d n : ℤ) := ∃ k : ℤ, n = d * k

-- Main statement
theorem odd_integers_equality (a b : ℤ) (ha_pos : 0 < a) (hb_pos : 0 < b)
 (ha_odd : is_odd a) (hb_odd : is_odd b)
 (h_div : divides (2 * a * b + 1) (a^2 + b^2 + 1))
 : a = b :=
by 
  sorry

end odd_integers_equality_l631_63168


namespace tan_alpha_eq_neg_five_twelfths_l631_63119

-- Define the angle α and the given conditions
variables (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π)

-- The goal is to prove that tan α = -5 / 12
theorem tan_alpha_eq_neg_five_twelfths (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π) :
  Real.tan α = -5 / 12 :=
sorry

end tan_alpha_eq_neg_five_twelfths_l631_63119


namespace compute_abs_a_plus_b_plus_c_l631_63131

variable (a b c : ℝ)

theorem compute_abs_a_plus_b_plus_c (h1 : a^2 - b * c = 14)
                                   (h2 : b^2 - c * a = 14)
                                   (h3 : c^2 - a * b = -3) :
                                   |a + b + c| = 5 :=
sorry

end compute_abs_a_plus_b_plus_c_l631_63131


namespace julie_net_monthly_income_is_l631_63169

section JulieIncome

def starting_pay : ℝ := 5.00
def additional_experience_pay_per_year : ℝ := 0.50
def years_of_experience : ℕ := 3
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 6
def bi_weekly_bonus : ℝ := 50.00
def tax_rate : ℝ := 0.12
def insurance_premium_per_month : ℝ := 40.00
def missed_days : ℕ := 1

-- Calculate Julie's net monthly income
def net_monthly_income : ℝ :=
    let hourly_wage := starting_pay + additional_experience_pay_per_year * years_of_experience
    let daily_earnings := hourly_wage * work_hours_per_day
    let weekly_earnings := daily_earnings * (work_days_per_week - missed_days)
    let bi_weekly_earnings := weekly_earnings * 2
    let gross_monthly_income := bi_weekly_earnings * 2 + bi_weekly_bonus * 2
    let tax_deduction := gross_monthly_income * tax_rate
    let total_deductions := tax_deduction + insurance_premium_per_month
    gross_monthly_income - total_deductions

theorem julie_net_monthly_income_is : net_monthly_income = 963.20 :=
    sorry

end JulieIncome

end julie_net_monthly_income_is_l631_63169


namespace geometric_sequence_sum_l631_63141

/-- Given a geometric sequence with common ratio r = 2, and the sum of the first four terms
    equals 1, the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a r : ℝ) (h : r = 2) (h_sum_four : a * (1 + r + r^2 + r^3) = 1) :
  a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) = 17 :=
by
  sorry

end geometric_sequence_sum_l631_63141


namespace find_a_b_l631_63192

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → a = 3 ∧ b = 2 :=
by
  intro h
  sorry

end find_a_b_l631_63192


namespace angle_value_l631_63142

theorem angle_value (x y : ℝ) (h_parallel : True)
  (h_alt_int_ang : x = y)
  (h_triangle_sum : 2 * x + x + 60 = 180) : 
  y = 40 := 
by
  sorry

end angle_value_l631_63142


namespace smallest_z_is_14_l631_63132

-- Define the consecutive even integers and the equation.
def w (k : ℕ) := 2 * k
def x (k : ℕ) := 2 * k + 2
def y (k : ℕ) := 2 * k + 4
def z (k : ℕ) := 2 * k + 6

theorem smallest_z_is_14 : ∃ k : ℕ, z k = 14 ∧ w k ^ 3 + x k ^ 3 + y k ^ 3 = z k ^ 3 :=
by sorry

end smallest_z_is_14_l631_63132


namespace map_distance_l631_63167

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l631_63167


namespace games_bought_from_friend_is_21_l631_63172

-- Definitions from the conditions
def games_bought_at_garage_sale : ℕ := 8
def non_working_games : ℕ := 23
def good_games : ℕ := 6

-- The total number of games John has is the sum of good and non-working games
def total_games : ℕ := good_games + non_working_games

-- The number of games John bought from his friend
def games_from_friend : ℕ := total_games - games_bought_at_garage_sale

-- Statement to prove
theorem games_bought_from_friend_is_21 : games_from_friend = 21 := by
  sorry

end games_bought_from_friend_is_21_l631_63172


namespace work_completion_time_for_A_l631_63158

theorem work_completion_time_for_A 
  (B_work_rate : ℝ)
  (combined_work_rate : ℝ)
  (x : ℝ) 
  (B_work_rate_def : B_work_rate = 1 / 6)
  (combined_work_rate_def : combined_work_rate = 3 / 10) :
  (1 / x) + B_work_rate = combined_work_rate →
  x = 7.5 := 
by
  sorry

end work_completion_time_for_A_l631_63158


namespace Eliza_first_more_than_300_paperclips_on_Thursday_l631_63170

theorem Eliza_first_more_than_300_paperclips_on_Thursday :
  ∃ k : ℕ, 5 * 3^k > 300 ∧ k = 4 := 
by
  sorry

end Eliza_first_more_than_300_paperclips_on_Thursday_l631_63170


namespace valid_triangle_count_l631_63118

def point := (ℤ × ℤ)

def isValidPoint (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4

def isCollinear (p1 p2 p3 : point) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def isValidTriangle (p1 p2 p3 : point) : Prop :=
  isValidPoint p1 ∧ isValidPoint p2 ∧ isValidPoint p3 ∧ ¬isCollinear p1 p2 p3

def numberOfValidTriangles : ℕ :=
  sorry -- This will contain the combinatorial calculations from the solution.

theorem valid_triangle_count : numberOfValidTriangles = 520 :=
  sorry -- Proof will show combinatorial result from counting non-collinear combinations.

end valid_triangle_count_l631_63118


namespace expected_number_of_draws_l631_63188

-- Given conditions
def redBalls : ℕ := 2
def blackBalls : ℕ := 5
def totalBalls : ℕ := redBalls + blackBalls

-- Definition of expected number of draws
noncomputable def expected_draws : ℚ :=
  (2 * (1/21) + 3 * (2/21) + 4 * (3/21) + 5 * (4/21) + 
   6 * (5/21) + 7 * (6/21))

-- The theorem statement to prove
theorem expected_number_of_draws :
  expected_draws = 16 / 3 := by
  sorry

end expected_number_of_draws_l631_63188


namespace hyperbola_center_coordinates_l631_63117

theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), 
  (∀ x y : ℝ, 
    ((4 * y - 6) ^ 2 / 36 - (5 * x - 3) ^ 2 / 49 = -1) ↔
    ((x - h) ^ 2 / ((7 / 5) ^ 2) - (y - k) ^ 2 / ((3 / 2) ^ 2) = 1)) ∧
  h = 3 / 5 ∧ k = 3 / 2 :=
by sorry

end hyperbola_center_coordinates_l631_63117


namespace fish_in_pond_l631_63179

noncomputable def number_of_fish (marked_first: ℕ) (marked_second: ℕ) (catch_first: ℕ) (catch_second: ℕ) : ℕ :=
  (marked_first * catch_second) / marked_second

theorem fish_in_pond (h1 : marked_first = 30) (h2 : marked_second = 2) (h3 : catch_first = 30) (h4 : catch_second = 40) :
  number_of_fish marked_first marked_second catch_first catch_second = 600 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fish_in_pond_l631_63179


namespace range_of_b_l631_63146

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (f x b) b

theorem range_of_b (b : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f x b = y) → (∀ z : ℝ, ∃ x : ℝ, g x b = z) → b ≥ 4 ∨ b ≤ -2 :=
sorry

end range_of_b_l631_63146


namespace condition_not_right_triangle_l631_63130

theorem condition_not_right_triangle 
  (AB BC AC : ℕ) (angleA angleB angleC : ℕ)
  (h_A : AB = 3 ∧ BC = 4 ∧ AC = 5)
  (h_B : AB / BC = 3 / 4 ∧ BC / AC = 4 / 5 ∧ AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB)
  (h_C : angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 ∧ angleA + angleB + angleC = 180)
  (h_D : angleA = 40 ∧ angleB = 50 ∧ angleA + angleB + angleC = 180) :
  angleA = 45 ∧ angleB = 60 ∧ angleC = 75 ∧ (¬ (angleA = 90 ∨ angleB = 90 ∨ angleC = 90)) :=
sorry

end condition_not_right_triangle_l631_63130


namespace quadratic_roots_l631_63147

theorem quadratic_roots (a b k : ℝ) (h₁ : a + b = -2) (h₂ : a * b = k / 3)
    (h₃ : |a - b| = 1/2 * (a^2 + b^2)) : k = 0 ∨ k = 6 :=
sorry

end quadratic_roots_l631_63147


namespace lcm_14_21_35_l631_63185

-- Define the numbers
def a : ℕ := 14
def b : ℕ := 21
def c : ℕ := 35

-- Define the prime factorizations
def prime_factors_14 : List (ℕ × ℕ) := [(2, 1), (7, 1)]
def prime_factors_21 : List (ℕ × ℕ) := [(3, 1), (7, 1)]
def prime_factors_35 : List (ℕ × ℕ) := [(5, 1), (7, 1)]

-- Prove the least common multiple
theorem lcm_14_21_35 : Nat.lcm (Nat.lcm a b) c = 210 := by
  sorry

end lcm_14_21_35_l631_63185


namespace find_y_squared_l631_63114

theorem find_y_squared (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : 2 * x - y = 20) : y ^ 2 = 4 := 
sorry

end find_y_squared_l631_63114


namespace part1_part2_l631_63125

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m
def h (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 :=
by
  sorry

theorem part2 (m : ℝ) : (∃ x : ℝ, f x < g x m) ↔ m > 4 :=
by
  sorry

end part1_part2_l631_63125


namespace sequence_a31_value_l631_63136

theorem sequence_a31_value 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₀ : a 1 = 0) 
  (h₁ : ∀ n, a (n + 1) = a n + b n) 
  (h₂ : b 15 + b 16 = 15)
  (h₃ : ∀ m n : ℕ, (b n - b m) = (n - m) * (b 2 - b 1)) :
  a 31 = 225 :=
by
  sorry

end sequence_a31_value_l631_63136


namespace red_balls_number_l631_63104

namespace BallDrawing

variable (x : ℕ) -- define x as the number of red balls

noncomputable def total_balls : ℕ := x + 4
noncomputable def yellow_ball_probability : ℚ := 4 / total_balls x

theorem red_balls_number : yellow_ball_probability x = 0.2 → x = 16 :=
by
  unfold yellow_ball_probability
  sorry

end BallDrawing

end red_balls_number_l631_63104


namespace cubes_product_fraction_l631_63115

theorem cubes_product_fraction :
  (4^3 * 6^3 * 8^3 * 9^3 : ℚ) / (10^3 * 12^3 * 14^3 * 15^3) = 576 / 546875 := 
sorry

end cubes_product_fraction_l631_63115


namespace measure_ADC_l631_63140

-- Definitions
def angle_measures (x y ADC : ℝ) : Prop :=
  2 * x + 60 + 2 * y = 180 ∧ x + y = 60 ∧ x + y + ADC = 180

-- Goal
theorem measure_ADC (x y ADC : ℝ) (h : angle_measures x y ADC) : ADC = 120 :=
by {
  -- Solution could go here, skipped for brevity
  sorry
}

end measure_ADC_l631_63140


namespace not_periodic_fraction_l631_63191

theorem not_periodic_fraction :
  ¬ ∃ (n k : ℕ), ∀ m ≥ n + k, ∃ l, 10^m + l = 10^(m+n) + l ∧ ((0.1234567891011121314 : ℝ) = (0.1234567891011121314 + l / (10^(m+n)))) :=
sorry

end not_periodic_fraction_l631_63191


namespace train_speed_in_kmh_l631_63122

theorem train_speed_in_kmh 
  (train_length : ℕ) 
  (crossing_time : ℕ) 
  (conversion_factor : ℕ) 
  (hl : train_length = 120) 
  (ht : crossing_time = 6) 
  (hc : conversion_factor = 36) :
  train_length / crossing_time * conversion_factor / 10 = 72 := by
  sorry

end train_speed_in_kmh_l631_63122


namespace trivia_game_answer_l631_63106

theorem trivia_game_answer (correct_first_half : Nat)
    (points_per_question : Nat) (final_score : Nat) : 
    correct_first_half = 8 → 
    points_per_question = 8 →
    final_score = 80 →
    (final_score - correct_first_half * points_per_question) / points_per_question = 2 :=
by
    intros h1 h2 h3
    sorry

end trivia_game_answer_l631_63106


namespace hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l631_63175

-- Definitions for the given problem
def one_hundred_million : ℕ := 100000000
def ten_million : ℕ := 10000000
def one_million : ℕ := 1000000
def ten_thousand : ℕ := 10000

-- Proving the statements
theorem hundred_million_is_ten_times_ten_million :
  one_hundred_million = 10 * ten_million :=
by
  sorry

theorem one_million_is_hundred_times_ten_thousand :
  one_million = 100 * ten_thousand :=
by
  sorry

end hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l631_63175


namespace interval_monotonic_increase_max_min_values_range_of_m_l631_63113

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

-- The interval of monotonic increase for f(x)
theorem interval_monotonic_increase :
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} = 
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} := 
by sorry

-- Maximum and minimum values of f(x) when x ∈ [π/4, π/2]
theorem max_min_values (x : ℝ) (h : x ∈ Set.Icc (π / 4) (π / 2)) :
  (f x ≤ 0 ∧ (f x = 0 ↔ x = π / 3)) ∧ (f x ≥ -1/2 ∧ (f x = -1/2 ↔ x = π / 2)) :=
by sorry

-- Range of m for the inequality |f(x) - m| < 1 when x ∈ [π/4, π/2]
theorem range_of_m (m : ℝ) (h : ∀ x ∈ Set.Icc (π / 4) (π / 2), |f x - m| < 1) :
  m ∈ Set.Ioo (-1) (1/2) :=
by sorry

end interval_monotonic_increase_max_min_values_range_of_m_l631_63113


namespace four_integers_sum_6_7_8_9_l631_63160

theorem four_integers_sum_6_7_8_9 (a b c d : ℕ)
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  (a = 1) ∧ (b = 2) ∧ (c = 3) ∧ (d = 4) := 
by 
  sorry

end four_integers_sum_6_7_8_9_l631_63160


namespace slope_of_line_AF_parabola_l631_63180

theorem slope_of_line_AF_parabola (A : ℝ × ℝ)
  (hA_on_parabola : A.snd ^ 2 = 4 * A.fst)
  (h_dist_focus : Real.sqrt ((A.fst - 1) ^ 2 + A.snd ^ 2) = 4) :
  (A.snd / (A.fst - 1) = Real.sqrt 3 ∨ A.snd / (A.fst - 1) = -Real.sqrt 3) :=
sorry

end slope_of_line_AF_parabola_l631_63180


namespace quadratic_two_roots_l631_63152

theorem quadratic_two_roots (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, (x = x₁ ∨ x = x₂) ↔ (x^2 + b*x - 3 = 0)) :=
by
  -- Indicate that a proof is required here
  sorry

end quadratic_two_roots_l631_63152


namespace total_area_of_farm_l631_63163

-- Define the number of sections and area of each section
def number_of_sections : ℕ := 5
def area_of_each_section : ℕ := 60

-- State the problem as proving the total area of the farm
theorem total_area_of_farm : number_of_sections * area_of_each_section = 300 :=
by sorry

end total_area_of_farm_l631_63163


namespace yellow_surface_area_fraction_minimal_l631_63123

theorem yellow_surface_area_fraction_minimal 
  (total_cubes : ℕ)
  (edge_length : ℕ)
  (yellow_cubes : ℕ)
  (blue_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (yellow_fraction : ℚ) :
  total_cubes = 64 ∧
  edge_length = 4 ∧
  yellow_cubes = 16 ∧
  blue_cubes = 48 ∧
  total_surface_area = 6 * edge_length * edge_length ∧
  yellow_surface_area = 15 →
  yellow_fraction = (yellow_surface_area : ℚ) / total_surface_area :=
sorry

end yellow_surface_area_fraction_minimal_l631_63123


namespace number_of_2_face_painted_cubes_l631_63144

-- Condition definitions based on the problem statement
def painted_faces (n : ℕ) (type : String) : ℕ :=
  if type = "corner" then 8
  else if type = "edge" then 12
  else if type = "face" then 24
  else if type = "inner" then 9
  else 0

-- The mathematical proof statement
theorem number_of_2_face_painted_cubes : painted_faces 27 "edge" = 12 :=
by
  sorry

end number_of_2_face_painted_cubes_l631_63144


namespace minimum_value_of_expression_l631_63120

theorem minimum_value_of_expression {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ m : ℝ, m = 0.75 ∧ ∀ z : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y = 1 ∧ z = 2 * x + 3 * y ^ 2) → z ≥ m :=
sorry

end minimum_value_of_expression_l631_63120


namespace jessica_has_100_dollars_l631_63137

-- Define the variables for Rodney, Ian, and Jessica
variables (R I J : ℝ)

-- Given conditions
axiom rodney_more_than_ian : R = I + 35
axiom ian_half_of_jessica : I = J / 2
axiom jessica_more_than_rodney : J = R + 15

-- The statement to prove
theorem jessica_has_100_dollars : J = 100 :=
by
  -- Proof will be completed here
  sorry

end jessica_has_100_dollars_l631_63137


namespace water_drain_rate_l631_63193

theorem water_drain_rate
  (total_volume : ℕ)
  (total_time : ℕ)
  (H1 : total_volume = 300)
  (H2 : total_time = 25) :
  total_volume / total_time = 12 := 
by
  sorry

end water_drain_rate_l631_63193


namespace regular_polygon_sides_l631_63166

-- Define the number of sides
def n : ℕ := sorry

-- The interior angle condition
def interior_angle_condition (n : ℕ) : Prop := 
  let interior_angle := (180 * (n - 2)) / n
  interior_angle = 160

-- Prove the statement
theorem regular_polygon_sides (h : interior_angle_condition n) : n = 18 := 
  sorry

end regular_polygon_sides_l631_63166


namespace new_parabola_through_point_l631_63161

def original_parabola (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

theorem new_parabola_through_point : 
  (∃ b : ℝ, ∀ x : ℝ, (x ^ 2 + 2 * x - 1 + b) = (x ^ 2 + 2 * x + 3)) :=
by
  sorry

end new_parabola_through_point_l631_63161


namespace wall_length_l631_63102

theorem wall_length (mirror_side length width : ℝ) (h1 : mirror_side = 21) (h2 : width = 28) 
  (h3 : 2 * mirror_side^2 = width * length) : length = 31.5 := by
  sorry

end wall_length_l631_63102


namespace ned_trays_per_trip_l631_63107

def trays_from_table1 : ℕ := 27
def trays_from_table2 : ℕ := 5
def total_trips : ℕ := 4
def total_trays : ℕ := trays_from_table1 + trays_from_table2
def trays_per_trip : ℕ := total_trays / total_trips

theorem ned_trays_per_trip :
  trays_per_trip = 8 :=
by
  -- proof is skipped
  sorry

end ned_trays_per_trip_l631_63107


namespace relationship_between_m_and_n_l631_63151

theorem relationship_between_m_and_n
  (a : ℝ) (b : ℝ) (ha : a > 2) (hb : b ≠ 0)
  (m : ℝ := a + 1 / (a - 2))
  (n : ℝ := 2^(2 - b^2)) :
  m > n :=
sorry

end relationship_between_m_and_n_l631_63151


namespace christine_stickers_l631_63181

theorem christine_stickers (stickers_has stickers_needs : ℕ) (h_has : stickers_has = 11) (h_needs : stickers_needs = 19) : 
  stickers_has + stickers_needs = 30 :=
by 
  sorry

end christine_stickers_l631_63181


namespace number_of_children_l631_63143

theorem number_of_children 
  (A C : ℕ) 
  (h1 : A + C = 201) 
  (h2 : 8 * A + 4 * C = 964) : 
  C = 161 := 
sorry

end number_of_children_l631_63143


namespace clara_cookies_l631_63187

theorem clara_cookies (x : ℕ) :
  50 * 12 + x * 20 + 70 * 16 = 3320 → x = 80 :=
by
  sorry

end clara_cookies_l631_63187


namespace hyperbola_condition_l631_63149

theorem hyperbola_condition (k : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (4 + k) + y^2 / (1 - k) = 1)) ↔ (k < -4 ∨ k > 1) :=
by 
  sorry

end hyperbola_condition_l631_63149


namespace total_watermelons_l631_63150

/-- Proof statement: Jason grew 37 watermelons and Sandy grew 11 watermelons. 
    Prove that they grew a total of 48 watermelons. -/
theorem total_watermelons (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ) 
                         (h1 : jason_watermelons = 37) (h2 : sandy_watermelons = 11) :
  total_watermelons = 48 :=
by
  sorry

end total_watermelons_l631_63150


namespace even_iff_b_eq_zero_l631_63110

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Given that f' is an even function, prove that b = 0.
theorem even_iff_b_eq_zero (h : ∀ x : ℝ, f' x = f' (-x)) : b = 0 :=
  sorry

end even_iff_b_eq_zero_l631_63110


namespace mopping_time_is_30_l631_63173

def vacuuming_time := 45
def dusting_time := 60
def brushing_time_per_cat := 5
def number_of_cats := 3
def total_free_time := 180
def free_time_left := 30

def total_cleaning_time := total_free_time - free_time_left
def brushing_time := brushing_time_per_cat * number_of_cats
def time_other_tasks := vacuuming_time + dusting_time + brushing_time

theorem mopping_time_is_30 : total_cleaning_time - time_other_tasks = 30 := by
  -- Calculation proof would go here
  sorry

end mopping_time_is_30_l631_63173


namespace at_least_30_cents_probability_l631_63116

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l631_63116


namespace quarterly_to_annual_rate_l631_63195

theorem quarterly_to_annual_rate (annual_rate : ℝ) (quarterly_rate : ℝ) (n : ℕ) (effective_annual_rate : ℝ) : 
  annual_rate = 4.5 →
  quarterly_rate = annual_rate / 4 →
  n = 4 →
  effective_annual_rate = (1 + quarterly_rate / 100)^n →
  effective_annual_rate * 100 = 4.56 :=
by
  intros h1 h2 h3 h4
  sorry

end quarterly_to_annual_rate_l631_63195


namespace cricket_team_right_handed_count_l631_63155

theorem cricket_team_right_handed_count 
  (total throwers non_throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h_total : total = 70)
  (h_throwers : throwers = 37)
  (h_non_throwers : non_throwers = total - throwers)
  (h_left_handed_non_throwers : left_handed_non_throwers = non_throwers / 3)
  (h_right_handed_non_throwers : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h_all_throwers_right_handed : ∀ (t : ℕ), t = throwers → t = right_handed_non_throwers + (total - throwers) - (non_throwers / 3)) :
  right_handed_non_throwers + throwers = 59 := 
by 
  sorry

end cricket_team_right_handed_count_l631_63155


namespace determine_a_range_l631_63189

noncomputable def single_element_intersection (a : ℝ) : Prop :=
  let A := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a * x + 1)}
  let B := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, |x|)}
  (∃ p : ℝ × ℝ, p ∈ A ∧ p ∈ B) ∧ 
  ∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ A ∧ p₁ ∈ B → p₂ ∈ A ∧ p₂ ∈ B → p₁ = p₂

theorem determine_a_range : 
  ∀ a : ℝ, single_element_intersection a ↔ a ∈ Set.Iic (-1) ∨ a ∈ Set.Ici 1 :=
sorry

end determine_a_range_l631_63189


namespace heather_blocks_remaining_l631_63139

-- Definitions of the initial amount of blocks and the amount shared
def initial_blocks : ℕ := 86
def shared_blocks : ℕ := 41

-- The statement to be proven
theorem heather_blocks_remaining : (initial_blocks - shared_blocks = 45) :=
by sorry

end heather_blocks_remaining_l631_63139


namespace cost_of_eraser_l631_63101

theorem cost_of_eraser 
  (s n c : ℕ)
  (h1 : s > 18)
  (h2 : n > 2)
  (h3 : c > n)
  (h4 : s * c * n = 3978) : 
  c = 17 :=
sorry

end cost_of_eraser_l631_63101


namespace smallest_positive_period_l631_63156

-- Define a predicate for a function to have a period
def is_periodic {α : Type*} [AddGroup α] (f : α → ℝ) (T : α) : Prop :=
  ∀ x, f (x) = f (x - T)

-- The actual problem statement
theorem smallest_positive_period {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, f (3 * x) = f (3 * x - 3 / 2)) : 
  is_periodic f (1 / 2) ∧ 
  ¬ (∃ T : ℝ, 0 < T ∧ T < 1 / 2 ∧ is_periodic f T) :=
by
  sorry

end smallest_positive_period_l631_63156


namespace greatest_product_sum_2000_l631_63171

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l631_63171


namespace valid_pairs_l631_63196

-- Define the target function and condition
def satisfies_condition (k l : ℤ) : Prop :=
  (7 * k - 5) * (4 * l - 3) = (5 * k - 3) * (6 * l - 1)

-- The theorem stating the exact pairs that satisfy the condition
theorem valid_pairs :
  ∀ (k l : ℤ), satisfies_condition k l ↔
    (k = 0 ∧ l = 6) ∨
    (k = 1 ∧ l = -1) ∨
    (k = 6 ∧ l = -6) ∨
    (k = 13 ∧ l = -7) ∨
    (k = -2 ∧ l = -22) ∨
    (k = -3 ∧ l = -15) ∨
    (k = -8 ∧ l = -10) ∨
    (k = -15 ∧ l = -9) :=
by
  sorry

end valid_pairs_l631_63196


namespace calories_per_slice_l631_63103

theorem calories_per_slice (n k t c : ℕ) (h1 : n = 8) (h2 : k = n / 2) (h3 : k * c = t) (h4 : t = 1200) : c = 300 :=
by sorry

end calories_per_slice_l631_63103


namespace a_is_perfect_square_l631_63199

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l631_63199


namespace largest_of_consecutive_even_integers_l631_63174

theorem largest_of_consecutive_even_integers (x : ℤ) (h : 25 * (x + 24) = 10000) : x + 48 = 424 :=
sorry

end largest_of_consecutive_even_integers_l631_63174


namespace monotonically_increasing_iff_l631_63198

noncomputable def f (x : ℝ) (a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) := 
sorry

end monotonically_increasing_iff_l631_63198


namespace magnitude_of_vector_l631_63121

open Complex

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - I) : 
  ‖(2 / z + z^2)‖ = Real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_l631_63121


namespace Georgie_prank_l631_63190

theorem Georgie_prank (w : ℕ) (condition1 : w = 8) : 
  ∃ (ways : ℕ), ways = 336 := 
by
  sorry

end Georgie_prank_l631_63190


namespace factorize_3m2_minus_12_l631_63153

theorem factorize_3m2_minus_12 (m : ℤ) : 
  3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := 
sorry

end factorize_3m2_minus_12_l631_63153


namespace eel_cost_l631_63157

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 :=
by
  sorry

end eel_cost_l631_63157


namespace factorization_1_factorization_2_factorization_3_factorization_4_l631_63197

-- Problem 1
theorem factorization_1 (a b : ℝ) : 
  4 * a^2 + 12 * a * b + 9 * b^2 = (2 * a + 3 * b)^2 :=
by sorry

-- Problem 2
theorem factorization_2 (a b : ℝ) : 
  16 * a^2 * (a - b) + 4 * b^2 * (b - a) = 4 * (a - b) * (2 * a - b) * (2 * a + b) :=
by sorry

-- Problem 3
theorem factorization_3 (m n : ℝ) : 
  25 * (m + n)^2 - 9 * (m - n)^2 = 4 * (4 * m + n) * (m + 4 * n) :=
by sorry

-- Problem 4
theorem factorization_4 (a b : ℝ) : 
  4 * a^2 - b^2 - 4 * a + 1 = (2 * a - 1 + b) * (2 * a - 1 - b) :=
by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_l631_63197


namespace total_people_correct_l631_63109

-- Define the daily changes as given conditions
def daily_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

-- Define the total number of people given 'a' and daily changes
def total_people (a : ℝ) : ℝ :=
  7 * a + daily_changes.sum

-- Lean statement for proving the total number of people
theorem total_people_correct (a : ℝ) : 
  total_people a = 7 * a + 13.2 :=
by
  -- This statement needs a proof, so we leave a placeholder 'sorry'
  sorry

end total_people_correct_l631_63109


namespace greatest_three_digit_multiple_of_17_l631_63182

theorem greatest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), n ≤ 999 ∧ n ≥ 100 ∧ (∃ k : ℕ, n = 17 * k) ∧ 
  (∀ m : ℕ, m ≤ 999 → m ≥ 100 → (∃ k : ℕ, m = 17 * k) → m ≤ n) ∧ n = 986 := 
sorry

end greatest_three_digit_multiple_of_17_l631_63182


namespace probability_of_A_not_losing_l631_63124

/-- The probability of player A winning is 0.3,
    and the probability of a draw between player A and player B is 0.4.
    Hence, the probability of player A not losing is 0.7. -/
theorem probability_of_A_not_losing (pA_win p_draw : ℝ) (hA_win : pA_win = 0.3) (h_draw : p_draw = 0.4) : 
  (pA_win + p_draw = 0.7) :=
by
  sorry

end probability_of_A_not_losing_l631_63124


namespace total_distance_both_l631_63165

-- Define conditions
def speed_onur : ℝ := 35  -- km/h
def speed_hanil : ℝ := 45  -- km/h
def daily_hours_onur : ℝ := 7
def additional_distance_hanil : ℝ := 40
def days_in_week : ℕ := 7

-- Define the daily biking distance for Onur and Hanil
def distance_onur_daily : ℝ := speed_onur * daily_hours_onur
def distance_hanil_daily : ℝ := distance_onur_daily + additional_distance_hanil

-- Define the number of days Onur and Hanil bike in a week
def working_days_onur : ℕ := 5
def working_days_hanil : ℕ := 6

-- Define the total distance covered by Onur and Hanil in a week
def total_distance_onur_week : ℝ := distance_onur_daily * working_days_onur
def total_distance_hanil_week : ℝ := distance_hanil_daily * working_days_hanil

-- Proof statement
theorem total_distance_both : total_distance_onur_week + total_distance_hanil_week = 2935 := by
  sorry

end total_distance_both_l631_63165


namespace shorter_trisector_length_eq_l631_63148

theorem shorter_trisector_length_eq :
  ∀ (DE EF DF FG : ℝ), DE = 6 → EF = 8 → DF = Real.sqrt (DE^2 + EF^2) → 
  FG = 2 * (24 / (3 + 4 * Real.sqrt 3)) → 
  FG = (192 * Real.sqrt 3 - 144) / 39 :=
by
  intros
  sorry

end shorter_trisector_length_eq_l631_63148


namespace cost_of_each_shirt_is_8_l631_63108

-- Define the conditions
variables (S : ℝ)
def shirts_cost := 4 * S
def pants_cost := 2 * 18
def jackets_cost := 2 * 60
def total_cost := shirts_cost S + pants_cost + jackets_cost
def carrie_pays := 94

-- The goal is to prove that S equals 8 given the conditions above
theorem cost_of_each_shirt_is_8
  (h1 : carrie_pays = total_cost S / 2) : S = 8 :=
sorry

end cost_of_each_shirt_is_8_l631_63108


namespace elliot_book_pages_l631_63145

theorem elliot_book_pages : 
  ∀ (initial_pages read_per_day days_in_week remaining_pages total_pages: ℕ), 
    initial_pages = 149 → 
    read_per_day = 20 → 
    days_in_week = 7 → 
    remaining_pages = 92 → 
    total_pages = initial_pages + (read_per_day * days_in_week) + remaining_pages → 
    total_pages = 381 :=
by
  intros initial_pages read_per_day days_in_week remaining_pages total_pages
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  assumption

end elliot_book_pages_l631_63145


namespace ascending_order_l631_63177

theorem ascending_order (a b c d : ℝ) (h1 : a = -6) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = Real.pi) :
  a < b ∧ b < c ∧ c < d :=
by
  sorry

end ascending_order_l631_63177


namespace bill_difference_is_zero_l631_63135

theorem bill_difference_is_zero
    (a b : ℝ)
    (h1 : 0.25 * a = 5)
    (h2 : 0.15 * b = 3) :
    a - b = 0 := 
by 
  sorry

end bill_difference_is_zero_l631_63135


namespace statement_true_when_b_le_a_div_5_l631_63183

theorem statement_true_when_b_le_a_div_5
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₀ : ∀ x : ℝ, f x = 5 * x + 3)
  (h₁ : ∀ x : ℝ, |f x + 7| < a ↔ |x + 2| < b)
  (h₂ : 0 < a)
  (h₃ : 0 < b) :
  b ≤ a / 5 :=
by
  sorry

end statement_true_when_b_le_a_div_5_l631_63183


namespace cos_difference_simplify_l631_63100

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end cos_difference_simplify_l631_63100


namespace puppies_per_cage_l631_63128

theorem puppies_per_cage (initial_puppies : ℕ) (sold_puppies : ℕ) (remaining_puppies : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 9)
  (h5 : puppies_per_cage = remaining_puppies / cages) : 
  puppies_per_cage = 9 := 
by
  -- The proof should go here
  sorry

end puppies_per_cage_l631_63128


namespace find_p_l631_63176

variable (m n p : ℝ)

theorem find_p (h1 : m = n / 7 - 2 / 5)
               (h2 : m + p = (n + 21) / 7 - 2 / 5) : p = 3 := by
  sorry

end find_p_l631_63176


namespace wall_width_l631_63164

theorem wall_width (area height : ℕ) (h1 : area = 16) (h2 : height = 4) : area / height = 4 :=
by
  sorry

end wall_width_l631_63164


namespace angles_geometric_sequence_count_l631_63129

def is_geometric_sequence (a b c : ℝ) : Prop :=
  (a = b * c) ∨ (b = a * c) ∨ (c = a * b)

theorem angles_geometric_sequence_count : 
  ∃! (angles : Finset ℝ), 
    (∀ θ ∈ angles, 0 < θ ∧ θ < 2 * Real.pi ∧ ¬∃ k : ℤ, θ = k * (Real.pi / 2)) ∧
    ∀ θ ∈ angles,
      is_geometric_sequence (Real.sin θ ^ 2) (Real.cos θ) (Real.tan θ) ∧
    angles.card = 2 := 
sorry

end angles_geometric_sequence_count_l631_63129


namespace mass_percentage_O_is_26_2_l631_63111

noncomputable def mass_percentage_O_in_Benzoic_acid : ℝ :=
  let molar_mass_C := 12.01
  let molar_mass_H := 1.01
  let molar_mass_O := 16.00
  let molar_mass_Benzoic_acid := (7 * molar_mass_C) + (6 * molar_mass_H) + (2 * molar_mass_O)
  let mass_O_in_Benzoic_acid := 2 * molar_mass_O
  (mass_O_in_Benzoic_acid / molar_mass_Benzoic_acid) * 100

theorem mass_percentage_O_is_26_2 :
  mass_percentage_O_in_Benzoic_acid = 26.2 := by
  sorry

end mass_percentage_O_is_26_2_l631_63111


namespace evaluate_f_at_3_l631_63154

theorem evaluate_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x + 3) : f 3 = 7 :=
by
  -- proof goes here
  sorry

end evaluate_f_at_3_l631_63154


namespace ratio_lateral_surface_area_to_surface_area_l631_63138

theorem ratio_lateral_surface_area_to_surface_area (r : ℝ) (h : ℝ) (V_sphere V_cone A_cone A_sphere : ℝ)
    (h_eq : h = r)
    (V_sphere_eq : V_sphere = (4 / 3) * Real.pi * r^3)
    (V_cone_eq : V_cone = (1 / 3) * Real.pi * (2 * r)^2 * h)
    (V_eq : V_sphere = V_cone)
    (A_cone_eq : A_cone = 2 * Real.sqrt 5 * Real.pi * r^2)
    (A_sphere_eq : A_sphere = 4 * Real.pi * r^2) :
    A_cone / A_sphere = Real.sqrt 5 / 2 := by
  sorry

end ratio_lateral_surface_area_to_surface_area_l631_63138


namespace find_complex_number_l631_63105

open Complex

theorem find_complex_number (a b : ℝ) (z : ℂ) 
  (h₁ : (∀ b: ℝ, (b^2 + 4 * b + 4 = 0) ∧ (b + a = 0))) :
  z = 2 - 2 * Complex.I :=
  sorry

end find_complex_number_l631_63105


namespace parallelepiped_analogy_l631_63134

-- Define the possible plane figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the concept of a parallelepiped
structure Parallelepiped : Type

-- The theorem asserting the parallelogram is the correct analogy
theorem parallelepiped_analogy : 
  ∀ (fig : PlaneFigure), 
    (fig = PlaneFigure.Parallelogram) ↔ 
    (fig = PlaneFigure.Parallelogram) :=
by sorry

end parallelepiped_analogy_l631_63134
