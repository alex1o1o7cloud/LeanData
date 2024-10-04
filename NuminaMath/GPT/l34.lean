import Mathlib

namespace contrapositive_l34_34849

theorem contrapositive (a b : ℝ) :
  (a > b → a^2 > b^2) → (a^2 ≤ b^2 → a ≤ b) :=
by
  intro h
  sorry

end contrapositive_l34_34849


namespace log_lt_x_l34_34523

theorem log_lt_x (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x := 
sorry

end log_lt_x_l34_34523


namespace shoe_size_percentage_difference_l34_34177

theorem shoe_size_percentage_difference :
  ∀ (size8_len size15_len size17_len : ℝ)
  (h1 : size8_len = size15_len - (7 * (1 / 5)))
  (h2 : size17_len = size15_len + (2 * (1 / 5)))
  (h3 : size15_len = 10.4),
  ((size17_len - size8_len) / size8_len) * 100 = 20 := by
  intros size8_len size15_len size17_len h1 h2 h3
  sorry

end shoe_size_percentage_difference_l34_34177


namespace y_work_time_l34_34896

noncomputable def total_work := 1 

noncomputable def work_rate_x := 1 / 40
noncomputable def work_x_in_8_days := 8 * work_rate_x
noncomputable def remaining_work := total_work - work_x_in_8_days

noncomputable def work_rate_y := remaining_work / 36

theorem y_work_time :
  (1 / work_rate_y) = 45 :=
by
  sorry

end y_work_time_l34_34896


namespace smallest_fraction_divides_exactly_l34_34744

theorem smallest_fraction_divides_exactly (a b c p q r m n : ℕ)
    (h1: a = 6) (h2: b = 5) (h3: c = 10) (h4: p = 7) (h5: q = 14) (h6: r = 21)
    (h1_frac: 6/7 = a/p) (h2_frac: 5/14 = b/q) (h3_frac: 10/21 = c/r)
    (h_lcm: m = Nat.lcm p (Nat.lcm q r)) (h_gcd: n = Nat.gcd a (Nat.gcd b c)) :
  (n/m) = 1/42 :=
by 
  sorry

end smallest_fraction_divides_exactly_l34_34744


namespace four_digit_number_sum_of_digits_2023_l34_34374

theorem four_digit_number_sum_of_digits_2023 (a b c d : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  1000 * a + 100 * b + 10 * c + d = a + b + c + d + 2023 → 
  (1000 * a + 100 * b + 10 * c + d = 1997 ∨ 1000 * a + 100 * b + 10 * c + d = 2015) :=
by
  sorry

end four_digit_number_sum_of_digits_2023_l34_34374


namespace problem_statement_l34_34504

noncomputable def S (k : ℕ) : ℚ := sorry

theorem problem_statement (k : ℕ) (a_k : ℚ) :
  S (k - 1) < 10 → S k > 10 → a_k = 6 / 7 :=
sorry

end problem_statement_l34_34504


namespace rectangle_sides_l34_34627

theorem rectangle_sides (n : ℕ) (hpos : n > 0)
  (h1 : (∃ (a : ℕ), (a^2 * n = n)))
  (h2 : (∃ (b : ℕ), (b^2 * (n + 98) = n))) :
  (∃ (l w : ℕ), l * w = n ∧ 
  ((n = 126 ∧ (l = 3 ∧ w = 42 ∨ l = 6 ∧ w = 21)) ∨
  (n = 1152 ∧ l = 24 ∧ w = 48))) :=
sorry

end rectangle_sides_l34_34627


namespace rainfall_on_wednesday_l34_34330

theorem rainfall_on_wednesday 
  (rain_on_monday : ℝ)
  (rain_on_tuesday : ℝ)
  (total_rain : ℝ) 
  (hmonday : rain_on_monday = 0.16666666666666666) 
  (htuesday : rain_on_tuesday = 0.4166666666666667) 
  (htotal : total_rain = 0.6666666666666666) :
  total_rain - (rain_on_monday + rain_on_tuesday) = 0.0833333333333333 :=
by
  -- Proof would go here
  sorry

end rainfall_on_wednesday_l34_34330


namespace no_pairs_satisfy_equation_l34_34817

theorem no_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ¬ (2 / a + 2 / b = 1 / (a + b)) :=
by
  intros a b ha hb h
  -- the proof would go here
  sorry

end no_pairs_satisfy_equation_l34_34817


namespace coupon_probability_l34_34581

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l34_34581


namespace paint_usage_correct_l34_34040

-- Define the parameters representing paint usage and number of paintings
def largeCanvasPaint : Nat := 3
def smallCanvasPaint : Nat := 2
def largePaintings : Nat := 3
def smallPaintings : Nat := 4

-- Define the total paint used
def totalPaintUsed : Nat := largeCanvasPaint * largePaintings + smallCanvasPaint * smallPaintings

-- Prove that total paint used is 17 ounces
theorem paint_usage_correct : totalPaintUsed = 17 :=
  by
    sorry

end paint_usage_correct_l34_34040


namespace total_spider_legs_l34_34036

theorem total_spider_legs (num_legs_single_spider group_spider_count: ℕ) 
      (h1: num_legs_single_spider = 8) 
      (h2: group_spider_count = (num_legs_single_spider / 2) + 10) :
      group_spider_count * num_legs_single_spider = 112 := 
by
  sorry

end total_spider_legs_l34_34036


namespace value_of_expression_l34_34597

theorem value_of_expression (b : ℚ) (h : b = 1/3) : (3 * b⁻¹ + (b⁻¹ / 3)) / b = 30 :=
by
  rw [h]
  sorry

end value_of_expression_l34_34597


namespace tyler_cd_purchase_l34_34591

theorem tyler_cd_purchase :
  ∀ (initial_cds : ℕ) (given_away_fraction : ℝ) (final_cds : ℕ) (bought_cds : ℕ),
    initial_cds = 21 →
    given_away_fraction = 1 / 3 →
    final_cds = 22 →
    bought_cds = 8 →
    final_cds = initial_cds - initial_cds * given_away_fraction + bought_cds :=
by
  intros
  sorry

end tyler_cd_purchase_l34_34591


namespace polynomial_evaluation_l34_34499

theorem polynomial_evaluation (a : ℝ) (h : a^2 + 3 * a = 2) : 2 * a^2 + 6 * a - 10 = -6 := by
  sorry

end polynomial_evaluation_l34_34499


namespace expected_value_of_win_is_51_l34_34314

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l34_34314


namespace range_of_a_l34_34657

open Real

theorem range_of_a (a : ℝ) :
  ((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∨ (a^2 - 2 * a - 3 < 0)) ∧
  ¬((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∧ (a^2 - 2 * a - 3 < 0)) ↔
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l34_34657


namespace initial_average_mark_l34_34569

-- Define the initial conditions
def num_students : ℕ := 9
def excluded_students_avg : ℕ := 44
def remaining_students_avg : ℕ := 80

-- Define the variables for total marks we calculated in the solution
def total_marks_initial := num_students * (num_students * excluded_students_avg / 5 + remaining_students_avg / (num_students - 5) * (num_students - 5))

-- The theorem we need to prove:
theorem initial_average_mark :
  (num_students * (excluded_students_avg * 5 + remaining_students_avg * (num_students - 5))) / num_students = 60 := 
  by
  -- step-by-step solution proof could go here, but we use sorry as placeholder
  sorry

end initial_average_mark_l34_34569


namespace find_a100_l34_34798

theorem find_a100 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, a n = (2 * (S n)^2) / (2 * (S n) - 1))
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 100 = -2 / 39203 := 
sorry

-- Explanation of the statement:
-- 'theorem find_a100': We define a theorem to find a_100.
-- 'a : ℕ → ℝ': a is a sequence of real numbers.
-- 'S : ℕ → ℝ': S is a sequence representing the sum of the first n terms.
-- 'h1' to 'h3': Given conditions from the problem statement.
-- 'a 100 = -2 / 39203' : The statement to prove.

end find_a100_l34_34798


namespace composite_number_iff_ge_2_l34_34653

theorem composite_number_iff_ge_2 (n : ℕ) : 
  ¬(Prime (3^(2*n+1) - 2^(2*n+1) - 6^n)) ↔ n ≥ 2 := by
  sorry

end composite_number_iff_ge_2_l34_34653


namespace tan_315_eq_neg_one_l34_34934

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34934


namespace children_count_125_l34_34352

def numberOfChildren (a : ℕ) : Prop :=
  a % 8 = 5 ∧ a % 10 = 7 ∧ 100 ≤ a ∧ a ≤ 150

theorem children_count_125 : ∃ a : ℕ, numberOfChildren a ∧ a = 125 := by
  use 125
  unfold numberOfChildren
  apply And.intro
  apply And.intro
  · norm_num
  · norm_num
  · split
  repeat {norm_num}
  sorry

end children_count_125_l34_34352


namespace ab_value_l34_34822

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a ^ 2 + b ^ 2 = 35) : a * b = 13 :=
by
  sorry

end ab_value_l34_34822


namespace solve_for_a_and_b_l34_34228

theorem solve_for_a_and_b (a b : ℤ) :
  (∀ x : ℤ, (x + a) * (x - 2) = x^2 + b * x - 6) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_for_a_and_b_l34_34228


namespace find_constants_for_B_l34_34108
open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2, 4], ![2, 0, 2], ![4, 2, 0]]

def I3 : Matrix (Fin 3) (Fin 3) ℝ := 1

def zeros : Matrix (Fin 3) (Fin 3) ℝ := 0

theorem find_constants_for_B : 
  ∃ (s t u : ℝ), s = 0 ∧ t = -36 ∧ u = -48 ∧ (B^3 + s • B^2 + t • B + u • I3 = zeros) :=
sorry

end find_constants_for_B_l34_34108


namespace tan_315_eq_neg1_l34_34915

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34915


namespace limit_of_hours_for_overtime_l34_34303

theorem limit_of_hours_for_overtime
  (R : Real) (O : Real) (total_compensation : Real) (total_hours_worked : Real) (L : Real)
  (hR : R = 14)
  (hO : O = 1.75 * R)
  (hTotalCompensation : total_compensation = 998)
  (hTotalHoursWorked : total_hours_worked = 57.88)
  (hEquation : (R * L) + ((total_hours_worked - L) * O) = total_compensation) :
  L = 40 := 
  sorry

end limit_of_hours_for_overtime_l34_34303


namespace side_length_of_square_l34_34425

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34425


namespace kate_bought_wands_l34_34391

theorem kate_bought_wands (price_per_wand : ℕ)
                           (additional_cost : ℕ)
                           (total_money_collected : ℕ)
                           (number_of_wands_sold : ℕ)
                           (total_wands_bought : ℕ) :
  price_per_wand = 60 → additional_cost = 5 → total_money_collected = 130 → 
  number_of_wands_sold = total_money_collected / (price_per_wand + additional_cost) →
  total_wands_bought = number_of_wands_sold + 1 →
  total_wands_bought = 3 := by
  sorry

end kate_bought_wands_l34_34391


namespace estimate_red_balls_l34_34244

-- Definitions based on conditions
def total_balls : ℕ := 20
def total_draws : ℕ := 100
def red_draws : ℕ := 30

-- The theorem statement
theorem estimate_red_balls (h1 : total_balls = 20) (h2 : total_draws = 100) (h3 : red_draws = 30) :
  (total_balls * (red_draws / total_draws) : ℤ) = 6 := 
by
  sorry

end estimate_red_balls_l34_34244


namespace hannah_quarters_l34_34515

theorem hannah_quarters :
  ∃ n : ℕ, 40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3 ∧ 
  (n = 171 ∨ n = 339) :=
by
  sorry

end hannah_quarters_l34_34515


namespace discount_is_28_l34_34037

-- Definitions
def price_notebook : ℕ := 15
def price_planner : ℕ := 10
def num_notebooks : ℕ := 4
def num_planners : ℕ := 8
def total_cost_with_discount : ℕ := 112

-- The original cost without discount
def original_cost : ℕ := num_notebooks * price_notebook + num_planners * price_planner

-- The discount amount
def discount_amount : ℕ := original_cost - total_cost_with_discount

-- Proof statement
theorem discount_is_28 : discount_amount = 28 := by
  sorry

end discount_is_28_l34_34037


namespace shape_described_by_theta_eq_c_is_plane_l34_34211

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end shape_described_by_theta_eq_c_is_plane_l34_34211


namespace side_length_of_square_l34_34431

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l34_34431


namespace new_students_joined_l34_34568

theorem new_students_joined (orig_avg_age new_avg_age : ℕ) (decrease_in_avg_age : ℕ) (orig_strength : ℕ) (new_students_avg_age : ℕ) :
  orig_avg_age = 40 ∧ new_avg_age = 36 ∧ decrease_in_avg_age = 4 ∧ orig_strength = 18 ∧ new_students_avg_age = 32 →
  ∃ x : ℕ, ((orig_strength * orig_avg_age) + (x * new_students_avg_age) = new_avg_age * (orig_strength + x)) ∧ x = 18 :=
by
  sorry

end new_students_joined_l34_34568


namespace minimize_transportation_cost_l34_34142

open Real

theorem minimize_transportation_cost :
  ∃ (v : ℝ) (h : v ∈ Ioo 0 100), 
    (∀ w ∈ Ioo 0 100, (50000 / v + 5 * v) ≤ (50000 / w + 5 * w)) ∧ 
    (50000 / v + 5 * v) = 1000 :=
by
  sorry -- Proof goes here

end minimize_transportation_cost_l34_34142


namespace trajectory_ellipse_l34_34077

/--
Given two fixed points A(-2,0) and B(2,0) in the Cartesian coordinate system, 
if a moving point P satisfies |PA| + |PB| = 6, 
then prove that the equation of the trajectory for point P is (x^2) / 9 + (y^2) / 5 = 1.
-/
theorem trajectory_ellipse (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hPA_PB : dist P A + dist P B = 6) :
  (P.1 ^ 2) / 9 + (P.2 ^ 2) / 5 = 1 :=
sorry

end trajectory_ellipse_l34_34077


namespace downstream_speed_l34_34768

noncomputable def speed_downstream (Vu Vs : ℝ) : ℝ :=
  2 * Vs - Vu

theorem downstream_speed (Vu Vs : ℝ) (hVu : Vu = 30) (hVs : Vs = 45) :
  speed_downstream Vu Vs = 60 := by
  rw [hVu, hVs]
  dsimp [speed_downstream]
  linarith

end downstream_speed_l34_34768


namespace trig_problem_1_trig_problem_2_l34_34754

-- Problem (1)
theorem trig_problem_1 (α : ℝ) (h1 : Real.tan (π + α) = -4 / 3) (h2 : 3 * Real.sin α / 4 = -Real.cos α)
  : Real.sin α = -4 / 5 ∧ Real.cos α = 3 / 5 := by
  sorry

-- Problem (2)
theorem trig_problem_2 : Real.sin (25 * π / 6) + Real.cos (26 * π / 3) + Real.tan (-25 * π / 4) = -1 := by
  sorry

end trig_problem_1_trig_problem_2_l34_34754


namespace baseball_card_distribution_l34_34408

theorem baseball_card_distribution (total_cards : ℕ) (capacity_4 : ℕ) (capacity_6 : ℕ) (capacity_8 : ℕ) :
  total_cards = 137 →
  capacity_4 = 4 →
  capacity_6 = 6 →
  capacity_8 = 8 →
  (total_cards % capacity_4) % capacity_6 = 1 :=
by
  intros
  sorry

end baseball_card_distribution_l34_34408


namespace range_of_a_l34_34632

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a <= x ∧ x < y ∧ y <= b → f y <= f x

theorem range_of_a (f : ℝ → ℝ) :
  odd_function f →
  decreasing_on_interval f (-1) 1 →
  (∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (2 * a - 1) < 0) →
  (∀ a : ℝ, 0 < a ∧ a < 1) :=
sorry

end range_of_a_l34_34632


namespace find_all_functions_l34_34686

theorem find_all_functions (f : ℕ → ℕ) : 
  (∀ a b : ℕ, 0 < a → 0 < b → f (a^2 + b^2) = f a * f b) →
  (∀ a : ℕ, 0 < a → f (a^2) = f a ^ 2) →
  (∀ n : ℕ, 0 < n → f n = 1) :=
by
  intros h1 h2 a ha
  sorry

end find_all_functions_l34_34686


namespace equal_points_in_tournament_l34_34672

-- Definitions specific to the conditions
def round_robin_tournament (teams : ℕ) := ∀ i j : ℕ, i ≠ j → game_result

structure game_result :=
  (team1 : ℕ)
  (team2 : ℕ)
  (result : Result)

inductive Result
| win (team : ℕ) : Result
| draw : Result
| loss (team : ℕ) : Result

-- The main theorem
theorem equal_points_in_tournament :
  ∀ (teams : ℕ),
  teams = 28 →
  (∃ G : round_robin_tournament 28,
    ∃ N : ℕ,
      N > (3 / 4) * (teams * (teams - 1) / 2) →
      (∃ (points : ℕ → ℕ) (i j : ℕ), i ≠ j ∧ points i = points j)) := 
sorry

end equal_points_in_tournament_l34_34672


namespace baker_usual_pastries_l34_34619

variable (P : ℕ)

theorem baker_usual_pastries
  (h1 : 2 * 14 + 4 * 25 - (2 * P + 4 * 10) = 48) : P = 20 :=
by
  sorry

end baker_usual_pastries_l34_34619


namespace expected_value_8_sided_die_l34_34308

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l34_34308


namespace tan_315_degrees_l34_34997

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l34_34997


namespace tom_total_payment_l34_34003

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l34_34003


namespace sixth_power_sum_l34_34659

/-- Given:
     (1) a + b = 1
     (2) a^2 + b^2 = 3
     (3) a^3 + b^3 = 4
     (4) a^4 + b^4 = 7
     (5) a^5 + b^5 = 11
    Prove:
     a^6 + b^6 = 18 -/
theorem sixth_power_sum (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sixth_power_sum_l34_34659


namespace sequence_term_position_l34_34025

theorem sequence_term_position :
  ∃ n : ℕ, ∀ k : ℕ, (k = 7 + 6 * (n - 1)) → k = 2005 → n = 334 :=
by
  sorry

end sequence_term_position_l34_34025


namespace geometric_sequence_arithmetic_condition_l34_34507

noncomputable def geometric_sequence_ratio (q : ℝ) : Prop :=
  q > 0

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  2 * a₃ = a₁ + 2 * a₂

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * q ^ n

theorem geometric_sequence_arithmetic_condition
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (q : ℝ)
  (hq : geometric_sequence_ratio q)
  (h_arith : arithmetic_sequence (a 0) (geometric_sequence a q 1) (geometric_sequence a q 2)) :
  (geometric_sequence a q 9 + geometric_sequence a q 10) / 
  (geometric_sequence a q 7 + geometric_sequence a q 8) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_arithmetic_condition_l34_34507


namespace tan_315_eq_neg_one_l34_34935

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34935


namespace customers_in_other_countries_l34_34302

-- Definitions for conditions
def total_customers : ℕ := 7422
def us_customers : ℕ := 723

-- Statement to prove
theorem customers_in_other_countries : total_customers - us_customers = 6699 :=
by
  sorry

end customers_in_other_countries_l34_34302


namespace negation_of_one_odd_l34_34418

-- Given a, b, c are natural numbers
def exactly_one_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 1)

def not_exactly_one_odd (a b c : ℕ) : Prop :=
  ¬ exactly_one_odd a b c

def at_least_two_odd (a b c : ℕ) : Prop :=
  (a % 2 = 1 ∧ b % 2 = 1) ∨
  (a % 2 = 1 ∧ c % 2 = 1) ∨
  (b % 2 = 1 ∧ c % 2 = 1)

def all_even (a b c : ℕ) : Prop :=
  (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)

theorem negation_of_one_odd (a b c : ℕ) : ¬ exactly_one_odd a b c ↔ all_even a b c ∨ at_least_two_odd a b c := by
  sorry

end negation_of_one_odd_l34_34418


namespace proof_problem_l34_34542

-- Conditions: p and q are solutions to the quadratic equation 3x^2 - 5x - 8 = 0
def is_solution (p q : ℝ) : Prop := (3 * p^2 - 5 * p - 8 = 0) ∧ (3 * q^2 - 5 * q - 8 = 0)

-- Question: Compute the value of (3 * p^2 - 3 * q^2) / (p - q) given the conditions
theorem proof_problem (p q : ℝ) (h : is_solution p q) :
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := sorry

end proof_problem_l34_34542


namespace evaluate_expression_l34_34130

theorem evaluate_expression : 
  let a := 2
  let b := 1 / 2
  2 * (a^2 - 2 * a * b) - 3 * (a^2 - a * b - 4 * b^2) = -2 :=
by
  let a := 2
  let b := 1 / 2
  sorry

end evaluate_expression_l34_34130


namespace tan_315_eq_neg1_l34_34920

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34920


namespace arc_length_of_path_l34_34337

def correspondence_rule (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) : ℝ × ℝ :=
  (Real.sqrt m, Real.sqrt n)

def point_A : ℝ × ℝ := (2, 6)
def point_B : ℝ × ℝ := (6, 2)

def line_AB (x y : ℝ) : Prop := x + y = 8

theorem arc_length_of_path :
  let A' := correspondence_rule 2 6 (by norm_num) (by norm_num)
  let B' := correspondence_rule 6 2 (by norm_num) (by norm_num) in
  ∃ θ : ℝ,
  cos θ = (sqrt 2 / sqrt 8) ∧ sin θ = (sqrt 6 / sqrt 8) ∧
  cos (-θ) = (sqrt 6 / sqrt 8) ∧ sin (-θ) = (sqrt 2 / sqrt 8) ∧
  (π / 3 ≤ θ ∧ θ ≤ (2 * π) / 3) ∧
  (2 * sqrt 8 * abs θ / 2) = arc_length_A'B' :=
sorry

end arc_length_of_path_l34_34337


namespace min_distance_ants_l34_34165

open Real

theorem min_distance_ants (points : Fin 1390 → ℝ × ℝ) :
  (∀ i j : Fin 1390, i ≠ j → dist (points i) (points j) > 0.02) → 
  (∀ i : Fin 1390, |(points i).snd| < 0.01) → 
  ∃ i j : Fin 1390, i ≠ j ∧ dist (points i) (points j) > 10 :=
by
  sorry

end min_distance_ants_l34_34165


namespace sugar_per_chocolate_bar_l34_34548

-- Definitions from conditions
def total_sugar : ℕ := 177
def lollipop_sugar : ℕ := 37
def chocolate_bar_count : ℕ := 14

-- Proof problem statement
theorem sugar_per_chocolate_bar : 
  (total_sugar - lollipop_sugar) / chocolate_bar_count = 10 := 
by 
  sorry

end sugar_per_chocolate_bar_l34_34548


namespace box_width_l34_34127

theorem box_width (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) : 
  rate = 4 ∧ time = 21 ∧ length = 7 ∧ depth = 2 ∧ volume = rate * time ∧ volume = length * width * depth → width = 6 :=
by
  sorry

end box_width_l34_34127


namespace paulina_convertibles_l34_34118

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end paulina_convertibles_l34_34118


namespace ratio_lcm_gcf_l34_34156

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 2^2 * 3^2 * 7) (h₂ : b = 2 * 3^2 * 5 * 7) :
  (Nat.lcm a b) / (Nat.gcd a b) = 10 := by
  sorry

end ratio_lcm_gcf_l34_34156


namespace reflection_of_C_over_y_eq_x_l34_34148

def point_reflection_over_yx := ∀ (A B C : (ℝ × ℝ)), 
  A = (6, 2) → 
  B = (2, 5) → 
  C = (2, 2) → 
  (reflect_y_eq_x C) = (2, 2)
where reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_C_over_y_eq_x :
  point_reflection_over_yx :=
by 
  sorry

end reflection_of_C_over_y_eq_x_l34_34148


namespace problem_solution_l34_34506

theorem problem_solution
  (a b c : ℕ)
  (h_pos_a : 0 < a ∧ a ≤ 10)
  (h_pos_b : 0 < b ∧ b ≤ 10)
  (h_pos_c : 0 < c ∧ c ≤ 10)
  (h1 : abc % 11 = 2)
  (h2 : 7 * c % 11 = 3)
  (h3 : 8 * b % 11 = 4 + b % 11) : 
  (a + b + c) % 11 = 0 := 
by
  sorry

end problem_solution_l34_34506


namespace probability_A_level_l34_34029

theorem probability_A_level (p_B : ℝ) (p_C : ℝ) (h_B : p_B = 0.03) (h_C : p_C = 0.01) : 
  (1 - (p_B + p_C)) = 0.96 :=
by
  -- Proof is omitted
  sorry

end probability_A_level_l34_34029


namespace imaginary_part_of_z_l34_34479

-- Define the problem conditions and what to prove
theorem imaginary_part_of_z (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_z_l34_34479


namespace marathon_finishers_l34_34170

-- Define the conditions
def totalParticipants : ℕ := 1250
def peopleGaveUp (F : ℕ) : ℕ := F + 124

-- Define the final statement to be proved
theorem marathon_finishers (F : ℕ) (h1 : totalParticipants = F + peopleGaveUp F) : F = 563 :=
by sorry

end marathon_finishers_l34_34170


namespace tan_315_eq_neg1_l34_34918

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34918


namespace polynomial_g_correct_l34_34540

noncomputable def polynomial_g : Polynomial ℚ := 
  Polynomial.C (-41 / 2) + Polynomial.X * 41 / 2 + Polynomial.X ^ 2

theorem polynomial_g_correct
  (f g : Polynomial ℚ)
  (h1 : f ≠ 0)
  (h2 : g ≠ 0)
  (hx : ∀ x, f.eval (g.eval x) = (Polynomial.eval x f) * (Polynomial.eval x g))
  (h3 : Polynomial.eval 3 g = 50) :
  g = polynomial_g :=
sorry

end polynomial_g_correct_l34_34540


namespace solve_real_equation_l34_34791

theorem solve_real_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x ^ 3 + 3 * x ^ 2 - x) / (x ^ 2 + 4 * x + 3) + x = -7 ↔ x = -5 / 2 ∨ x = -4 := 
by
  sorry

end solve_real_equation_l34_34791


namespace cricket_innings_l34_34570

theorem cricket_innings (n : ℕ) (h1 : (32 * n + 137) / (n + 1) = 37) : n = 20 :=
sorry

end cricket_innings_l34_34570


namespace sqrt_sq_eq_abs_l34_34818

theorem sqrt_sq_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| :=
sorry

end sqrt_sq_eq_abs_l34_34818


namespace inequality_a6_b6_l34_34217

theorem inequality_a6_b6 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end inequality_a6_b6_l34_34217


namespace compute_expression_l34_34637

theorem compute_expression : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end compute_expression_l34_34637


namespace range_of_a_l34_34661

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l34_34661


namespace min_value_a_plus_2b_l34_34524

theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 20) : a + 2 * b = 4 * Real.sqrt 10 :=
by
  sorry

end min_value_a_plus_2b_l34_34524


namespace total_handshakes_tournament_l34_34827

/-- 
In a women's doubles tennis tournament, four teams of two women competed. After the tournament, 
each woman shook hands only once with each of the other players, except with her own partner.
Prove that the total number of unique handshakes is 24.
-/
theorem total_handshakes_tournament : 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  total_handshakes = 24 :=
by 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  have : total_handshakes = 24 := sorry
  exact this

end total_handshakes_tournament_l34_34827


namespace hours_per_day_l34_34173

variable (M : ℕ)

noncomputable def H : ℕ := 9
noncomputable def D1 : ℕ := 24
noncomputable def Men2 : ℕ := 12
noncomputable def D2 : ℕ := 16

theorem hours_per_day (H_new : ℝ) : 
  (M * H * D1 : ℝ) = (Men2 * H_new * D2) → 
  H_new = (M * 9 : ℝ) / 8 := 
  sorry

end hours_per_day_l34_34173


namespace triangle_area_gt_half_l34_34587

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l34_34587


namespace complement_union_sets_l34_34076

open Set

theorem complement_union_sets :
  ∀ (U A B : Set ℕ), (U = {1, 2, 3, 4}) → (A = {2, 3}) → (B = {3, 4}) → (U \ (A ∪ B) = {1}) :=
by
  intros U A B hU hA hB
  rw [hU, hA, hB]
  simp 
  sorry

end complement_union_sets_l34_34076


namespace ratio_of_a_over_5_to_b_over_4_l34_34610

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a/5) / (b/4) = 1 :=
sorry

end ratio_of_a_over_5_to_b_over_4_l34_34610


namespace range_of_a_l34_34400

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x^2
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem range_of_a (a : ℝ) (h : ∀ s t : ℝ, (1/2 ≤ s ∧ s ≤ 2) → (1/2 ≤ t ∧ t ≤ 2) → f a s ≥ g t) : a ≥ 1 :=
sorry

end range_of_a_l34_34400


namespace volume_increase_factor_l34_34015

   variable (π : ℝ) (r h : ℝ)

   def original_volume : ℝ := π * r^2 * h

   def new_height : ℝ := 3 * h

   def new_radius : ℝ := 2.5 * r

   def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

   theorem volume_increase_factor :
     new_volume π r h = 18.75 * original_volume π r h := 
   by
     sorry
   
end volume_increase_factor_l34_34015


namespace set_intersection_complement_l34_34226

def U := {x : ℝ | x > -3}
def A := {x : ℝ | x < -2 ∨ x > 3}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

theorem set_intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x < -2 ∨ x > 4} :=
by sorry

end set_intersection_complement_l34_34226


namespace jane_sleep_hours_for_second_exam_l34_34106

theorem jane_sleep_hours_for_second_exam :
  ∀ (score1 score2 hours1 hours2 : ℝ),
  score1 * hours1 = 675 →
  (score1 + score2) / 2 = 85 →
  score2 * hours2 = 675 →
  hours2 = 135 / 19 :=
by
  intros score1 score2 hours1 hours2 h1 h2 h3
  sorry

end jane_sleep_hours_for_second_exam_l34_34106


namespace number_of_ordered_pairs_xy_2007_l34_34736

theorem number_of_ordered_pairs_xy_2007 : 
  ∃ n, n = 6 ∧ (∀ x y : ℕ, x * y = 2007 → x > 0 ∧ y > 0) :=
sorry

end number_of_ordered_pairs_xy_2007_l34_34736


namespace expected_value_of_win_is_51_l34_34313

noncomputable def expected_value_of_win : ℝ :=
  (∑ n in (finset.range 8).map (λ x, x + 1), (1/8) * 2 * (n : ℝ)^2)

theorem expected_value_of_win_is_51 : expected_value_of_win = 51 := 
by 
  sorry

end expected_value_of_win_is_51_l34_34313


namespace tangent_315_deg_l34_34977

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34977


namespace coordinates_of_point_l34_34245

theorem coordinates_of_point (x y : ℝ) (hx : x < 0) (hy : y > 0) (dx : |x| = 3) (dy : |y| = 2) :
  (x, y) = (-3, 2) := 
sorry

end coordinates_of_point_l34_34245


namespace proof_problem_l34_34846

def is_solution (x : ℝ) : Prop :=
  4 * Real.cos x * Real.cos (2 * x) * Real.cos (3 * x) = Real.cos (6 * x)

noncomputable def solution (l n : ℤ) : ℝ :=
  max (Real.pi / 3 * (3 * l + 1)) (Real.pi / 4 * (2 * n + 1))

theorem proof_problem (x : ℝ) (l n : ℤ) : is_solution x → x = solution l n :=
sorry

end proof_problem_l34_34846


namespace initial_average_age_of_students_l34_34567

theorem initial_average_age_of_students 
(A : ℕ) 
(h1 : 23 * A + 46 = (A + 1) * 24) : 
  A = 22 :=
by
  sorry

end initial_average_age_of_students_l34_34567


namespace power_function_through_point_l34_34069

-- Define the condition that the power function passes through the point (2, 8)
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x^α) (h₂ : f 2 = 8) :
  α = 3 ∧ ∀ x, f x = x^3 :=
by
  -- Proof will be provided here
  sorry

end power_function_through_point_l34_34069


namespace blood_expiration_date_blood_expiration_final_date_l34_34905

theorem blood_expiration_date :
  let seconds_per_day := 86400
  let jan_days := 31
  let ten_fact := 10.factorial
  let total_days := ten_fact / seconds_per_day
  (jan_days + total_days) = 42 :=
by
  let seconds_per_day := 86400
  let jan_days := 31
  let ten_fact := 10.factorial
  let total_days := ten_fact / seconds_per_day
  exact Nat.div_add_div 10.factorial 86400 sorry -- where the modular operation ensures the division approximation
  let end_day := jan_days + total_days
  exact end_day = 42

-- Sanity check: Complete the theorem to see the overall goal
theorem blood_expiration_final_date :
  let days_in_jan := 31
  let days_after_jan := 42 - days_in_jan
  days_after_jan = 11 :=
by
  let days_in_jan := 31
  let days_after_jan := 42 - days_in_jan
  exact days_after_jan = 11

end blood_expiration_date_blood_expiration_final_date_l34_34905


namespace tan_315_eq_neg_one_l34_34990

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34990


namespace max_coins_Martha_can_take_l34_34038

/-- 
  Suppose a total of 2010 coins are distributed in 5 boxes with quantities 
  initially forming consecutive natural numbers. Martha can perform a 
  transformation where she takes one coin from a box with at least 4 coins and 
  distributes one coin to each of the other boxes. Prove that the maximum number 
  of coins that Martha can take away is 2004.
-/
theorem max_coins_Martha_can_take : 
  ∃ (a : ℕ), 2010 = a + (a+1) + (a+2) + (a+3) + (a+4) ∧ 
  ∀ (f : ℕ → ℕ) (h : (∃ b ≥ 4, f b = 400 + b)), 
  (∃ n : ℕ, f n = 4) → (∃ n : ℕ, f n = 3) → 
  (∃ n : ℕ, f n = 2) → (∃ n : ℕ, f n = 1) → 
  (∃ m : ℕ, f m = 2004) := 
by
  sorry

end max_coins_Martha_can_take_l34_34038


namespace find_m_l34_34365

theorem find_m (x : ℝ) (m : ℝ) (h1 : x > 2) (h2 : x - 3 * m + 1 > 0) : m = 1 :=
sorry

end find_m_l34_34365


namespace m_power_of_prime_no_m_a_k_l34_34169

-- Part (i)
theorem m_power_of_prime (m : ℕ) (p : ℕ) (k : ℕ) (h1 : m ≥ 1) (h2 : Prime p) (h3 : m * (m + 1) = p^k) : m = 1 :=
by sorry

-- Part (ii)
theorem no_m_a_k (m a k : ℕ) (h1 : m ≥ 1) (h2 : a ≥ 1) (h3 : k ≥ 2) (h4 : m * (m + 1) = a^k) : False :=
by sorry

end m_power_of_prime_no_m_a_k_l34_34169


namespace coins_division_remainder_l34_34898

theorem coins_division_remainder
  (n : ℕ)
  (h1 : n % 6 = 4)
  (h2 : n % 5 = 3)
  (h3 : n = 28) :
  n % 7 = 0 :=
by
  sorry

end coins_division_remainder_l34_34898


namespace tan_315_eq_neg1_l34_34944

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34944


namespace smallest_n_satisfies_l34_34449

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 1.5 else 1 / (n^2 - 1)

def sequence_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, sequence (k + 1)

theorem smallest_n_satisfies (n : ℕ) (h : n = 100) : 
  | sequence_sum n - 2.25 | < 0.01 :=
by
  sorry

end smallest_n_satisfies_l34_34449


namespace infinite_pairs_natural_numbers_l34_34269

theorem infinite_pairs_natural_numbers :
  ∃ (infinite_pairs : ℕ × ℕ → Prop), (∀ a b : ℕ, infinite_pairs (a, b) ↔ (b ∣ (a^2 + 1) ∧ a ∣ (b^2 + 1))) ∧
    ∀ n : ℕ, ∃ (a b : ℕ), infinite_pairs (a, b) :=
sorry

end infinite_pairs_natural_numbers_l34_34269


namespace number_of_children_l34_34354

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l34_34354


namespace midpoint_parallel_l34_34398

-- Define the circumcircle, midpoint of arcs, and parallelism relation in Lean
noncomputable def midpoint (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def are_parallel (P Q R S : Point) : Prop := sorry

theorem midpoint_parallel (A B C P Q : Point) 
    (circ : Circle)
    (hcirc : circ = circumcircle A B C)
    (M : Point)
    (M_def : M = midpoint A B C)
    (N : Point)
    (N_def : N = midpoint B C A) :
  are_parallel M N P Q :=
sorry

end midpoint_parallel_l34_34398


namespace side_length_of_square_l34_34420

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l34_34420


namespace cans_ounces_per_day_l34_34055

-- Definitions of the conditions
def daily_soda_cans : ℕ := 5
def daily_water_ounces : ℕ := 64
def weekly_fluid_ounces : ℕ := 868

-- Theorem statement proving the number of ounces per can of soda
theorem cans_ounces_per_day (h_soda_daily : daily_soda_cans * 7 = 35)
    (h_weekly_soda : weekly_fluid_ounces - daily_water_ounces * 7 = 420) 
    (h_total_weekly : 35 = ((daily_soda_cans * 7))):
  420 / 35 = 12 := by
  sorry

end cans_ounces_per_day_l34_34055


namespace find_number_l34_34456

-- Define the condition given in the problem
def condition (x : ℕ) : Prop :=
  x / 5 + 6 = 65

-- Prove that the solution satisfies the condition
theorem find_number : ∃ x : ℕ, condition x ∧ x = 295 :=
by
  -- Skip the actual proof steps
  sorry

end find_number_l34_34456


namespace number_of_days_l34_34229

noncomputable def days_to_lay_bricks (b c f : ℕ) : ℕ :=
(b * b) / f

theorem number_of_days (b c f : ℕ) (h_nonzero_f : f ≠ 0) (h_bc_pos : b > 0 ∧ c > 0) :
  days_to_lay_bricks b c f = (b * b) / f :=
by 
  sorry

end number_of_days_l34_34229


namespace side_length_of_square_l34_34421

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l34_34421


namespace probability_blue_or_orange_is_five_thirteen_l34_34757

-- Define the number of jelly beans of each color.
def red_jelly_beans : ℕ := 7
def green_jelly_beans : ℕ := 8
def yellow_jelly_beans : ℕ := 9
def blue_jelly_beans : ℕ := 10
def orange_jelly_beans : ℕ := 5

-- Define the total number of jelly beans.
def total_jelly_beans : ℕ := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans + orange_jelly_beans

-- Define the number of blue or orange jelly beans.
def blue_or_orange_jelly_beans : ℕ := blue_jelly_beans + orange_jelly_beans

-- Define the expected probability
def expected_probability : ℚ := 5 / 13

-- The theorem we aim to prove
theorem probability_blue_or_orange_is_five_thirteen :
  (blue_or_orange_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = expected_probability :=
by
  sorry

end probability_blue_or_orange_is_five_thirteen_l34_34757


namespace rectangle_side_ratio_l34_34316

theorem rectangle_side_ratio (s x y : ℝ) 
  (h1 : 8 * (x * y) = (9 - 1) * s^2) 
  (h2 : s + 4 * y = 3 * s) 
  (h3 : 2 * x + y = 3 * s) : 
  x / y = 2.5 :=
by
  sorry

end rectangle_side_ratio_l34_34316


namespace coordinates_of_P_tangent_line_equation_l34_34503

-- Define point P and center of the circle
def point_P : ℝ × ℝ := (-2, 1)
def center_C : ℝ × ℝ := (-1, 0)

-- Define the circle equation (x + 1)^2 + y^2 = 2
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the tangent line at point P
def tangent_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Prove the coordinates of point P are (-2, 1) given the conditions
theorem coordinates_of_P (n : ℝ) (h1 : n > 0) (h2 : circle_equation (-2) n) :
  point_P = (-2, 1) :=
by
  -- Proof steps would go here
  sorry

-- Prove the equation of the tangent line to the circle C passing through point P is x - y + 3 = 0
theorem tangent_line_equation :
  tangent_line (-2) 1 :=
by
  -- Proof steps would go here
  sorry

end coordinates_of_P_tangent_line_equation_l34_34503


namespace ball_cost_l34_34214

theorem ball_cost (C x y : ℝ)
  (H1 :  x = 1/3 * (C/2 + y + 5) )
  (H2 :  y = 1/4 * (C/2 + x + 5) )
  (H3 :  C/2 + x + y + 5 = C ) : C = 20 := 
by
  sorry

end ball_cost_l34_34214


namespace mini_bottles_needed_to_fill_jumbo_l34_34178

def mini_bottle_capacity : ℕ := 45
def jumbo_bottle_capacity : ℕ := 600

-- The problem statement expressed as a Lean theorem.
theorem mini_bottles_needed_to_fill_jumbo :
  (jumbo_bottle_capacity + mini_bottle_capacity - 1) / mini_bottle_capacity = 14 :=
by
  sorry

end mini_bottles_needed_to_fill_jumbo_l34_34178


namespace solve_for_x_l34_34300

theorem solve_for_x (x : ℝ) (h : 0.4 * x = (1 / 3) * x + 110) : x = 1650 :=
by sorry

end solve_for_x_l34_34300


namespace profit_percent_is_approx_6_point_35_l34_34605

noncomputable def selling_price : ℝ := 2552.36
noncomputable def cost_price : ℝ := 2400
noncomputable def profit_amount : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_percent_is_approx_6_point_35 : abs (profit_percent - 6.35) < 0.01 := sorry

end profit_percent_is_approx_6_point_35_l34_34605


namespace probability_three_draws_exceed_eight_l34_34474

-- Definitions of the conditions in the problem
def chip_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

def sum_exceeds_eight (l: List ℕ) : Bool :=
  (l.sum > 8)

def three_draws_required (draws: List ℕ) : Prop :=
  draws.length = 3 ∧ sum_exceeds_eight draws ∧ sum_exceeds_eight draws.take 2 = false

-- Formal statement of the problem
theorem probability_three_draws_exceed_eight :
  (∑ x in { l : List ℕ | l.length = 3 ∧ sum_exceeds_eight l ∧ sum_exceeds_eight l.take 2 = false }, 1) / 
  (∑ x in { l : List ℕ | l.length = 3 }, 1) = 3 / 5 := sorry

end probability_three_draws_exceed_eight_l34_34474


namespace systematic_sampling_l34_34825

theorem systematic_sampling (N n : ℕ) (hN : N = 1650) (hn : n = 35) :
  let E := 5 
  let segments := 35 
  let individuals_per_segment := 47 
  1650 % 35 = E ∧ 
  (1650 - E) / 35 = individuals_per_segment :=
by 
  sorry

end systematic_sampling_l34_34825


namespace opposite_neg_two_l34_34711

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l34_34711


namespace spherical_coordinates_neg_y_l34_34769

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arctan2 (Real.sqrt (x^2 + y^2)) z
  let θ := Real.arctan2 y x
  (ρ, θ, φ)

theorem spherical_coordinates_neg_y
  (ρ θ φ : ℝ)
  (hρ : ρ = 3)
  (hθ : θ = 5 * Real.pi / 6)
  (hφ : φ = Real.pi / 4) :
  rectangular_to_spherical
    (ρ * Real.sin φ * Real.cos θ)
    (-ρ * Real.sin φ * Real.sin θ)
    (ρ * Real.cos φ) = (3, Real.pi / 6, Real.pi / 4) :=
by
  have h : spherical_to_rectangular 3 (5 * Real.pi / 6) (Real.pi / 4) =
           (3 * Real.sin (Real.pi / 4) * Real.cos (5 * Real.pi / 6), 
            3 * Real.sin (Real.pi / 4) * Real.sin (5 * Real.pi / 6), 
            3 * Real.cos (Real.pi / 4)) := by sorry
  have h_neg_y : spherical_to_rectangular 3 (Real.pi / 6) (Real.pi / 4) =
           (3 * Real.sin (Real.pi / 4) * Real.cos (Real.pi / 6), 
            3 * Real.sin (Real.pi / 4) * Real.sin (Real.pi / 6), 
            3 * Real.cos (Real.pi / 4)) := by sorry
  sorry


end spherical_coordinates_neg_y_l34_34769


namespace possible_values_of_b_l34_34708

theorem possible_values_of_b (b : ℝ) (h : ∃ x y : ℝ, y = 2 * x + b ∧ y > 0 ∧ x = 0) : b > 0 :=
sorry

end possible_values_of_b_l34_34708


namespace side_length_of_square_l34_34426

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34426


namespace least_number_to_make_divisible_by_3_l34_34743

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_make_divisible_by_3 : ∃ k : ℕ, (∃ n : ℕ, 
  sum_of_digits 625573 ≡ 28 [MOD 3] ∧ 
  (625573 + k) % 3 = 0 ∧ 
  k = 2) :=
by
  sorry

end least_number_to_make_divisible_by_3_l34_34743


namespace opposite_of_neg_two_l34_34729

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l34_34729


namespace repeating_decimal_fraction_product_l34_34013

theorem repeating_decimal_fraction_product :
  let x := (0.0012 : ℝ) in
  let num_denom_product := 13332 in
  (real.to_rat x).num * (real.to_rat x).denom = num_denom_product :=
by
  -- Mathematic proof steps go here
  sorry

end repeating_decimal_fraction_product_l34_34013


namespace chuck_vs_dave_ride_time_l34_34048

theorem chuck_vs_dave_ride_time (D E : ℕ) (h1 : D = 10) (h2 : E = 65) (h3 : E = 13 * C / 10) :
  (C / D = 5) :=
by
  sorry

end chuck_vs_dave_ride_time_l34_34048


namespace inverse_proportional_l34_34565

/-- Given that α is inversely proportional to β and α = -3 when β = -6,
    prove that α = 9/4 when β = 8. --/
theorem inverse_proportional (α β : ℚ) 
  (h1 : α * β = 18)
  (h2 : β = 8) : 
  α = 9 / 4 :=
by
  sorry

end inverse_proportional_l34_34565


namespace counties_percentage_l34_34484

theorem counties_percentage (a b c : ℝ) (ha : a = 0.2) (hb : b = 0.35) (hc : c = 0.25) :
  a + b + c = 0.8 :=
by
  rw [ha, hb, hc]
  sorry

end counties_percentage_l34_34484


namespace tan_315_eq_neg1_l34_34989

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34989


namespace inequality_bounds_l34_34343

theorem inequality_bounds (x y : ℝ) : |y - 3 * x| < 2 * x ↔ x > 0 ∧ x < y ∧ y < 5 * x := by
  sorry

end inequality_bounds_l34_34343


namespace obtuse_triangle_side_range_l34_34222

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end obtuse_triangle_side_range_l34_34222


namespace alice_met_tweedledee_l34_34698

noncomputable def brother_statement (day : ℕ) : Prop :=
  sorry -- Define the exact logical structure of the statement "I am lying today, and my name is Tweedledum" here

theorem alice_met_tweedledee (day : ℕ) : brother_statement day → (∃ (b : String), b = "Tweedledee") :=
by
  sorry -- provide the proof here

end alice_met_tweedledee_l34_34698


namespace sector_area_l34_34070

theorem sector_area (C : ℝ) (θ : ℝ) (r : ℝ) (S : ℝ)
  (hC : C = (8 * Real.pi / 9) + 4)
  (hθ : θ = (80 * Real.pi / 180))
  (hne : θ * r / 2 + r = C) :
  S = (1 / 2) * θ * r^2 → S = 8 * Real.pi / 9 :=
by
  sorry

end sector_area_l34_34070


namespace work_alone_days_l34_34171

theorem work_alone_days (A B C : ℝ) 
    (hB : B = 7) 
    (hC : C = 28/3) 
    (combined_work_time : ℝ)
    (combined_work_rate : ℝ)
    (habc_rate : (1/A + 1/B + 3/28 = combined_work_rate))
    (h_combined_time : combined_work_time = 2):
  A = 4 := 
  sorry

end work_alone_days_l34_34171


namespace paula_candies_l34_34553

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l34_34553


namespace simplify_expression_l34_34045

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l34_34045


namespace cubes_with_two_or_three_blue_faces_l34_34032

theorem cubes_with_two_or_three_blue_faces 
  (four_inch_cube : ℝ)
  (painted_blue_faces : ℝ)
  (one_inch_cubes : ℝ) :
  (four_inch_cube = 4) →
  (painted_blue_faces = 6) →
  (one_inch_cubes = 64) →
  (num_cubes_with_two_or_three_blue_faces = 32) :=
sorry

end cubes_with_two_or_three_blue_faces_l34_34032


namespace maximize_revenue_l34_34903

noncomputable def revenue (p : ℝ) : ℝ := 100 * p - 4 * p^2

theorem maximize_revenue : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 20 ∧ (∀ q : ℝ, 0 ≤ q ∧ q ≤ 20 → revenue q ≤ revenue p) ∧ p = 12.5 := by
  sorry

end maximize_revenue_l34_34903


namespace find_minimum_x_and_values_l34_34519

theorem find_minimum_x_and_values (x y z w : ℝ) (h1 : y = x - 2003)
  (h2 : z = 2 * y - 2003)
  (h3 : w = 3 * z - 2003)
  (h4 : 0 ≤ x)
  (h5 : 0 ≤ y)
  (h6 : 0 ≤ z)
  (h7 : 0 ≤ w) :
  x ≥ 10015 / 3 ∧ 
  (x = 10015 / 3 → y = 4006 / 3 ∧ z = 2003 / 3 ∧ w = 0) := by
  sorry

end find_minimum_x_and_values_l34_34519


namespace opposite_of_neg_two_l34_34717

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l34_34717


namespace remaining_work_hours_l34_34518

theorem remaining_work_hours (initial_hours_per_week initial_weeks total_earnings first_weeks first_week_hours : ℝ) 
  (hourly_wage remaining_weeks remaining_earnings total_hours_required : ℝ) : 
  15 = initial_hours_per_week →
  15 = initial_weeks →
  4500 = total_earnings →
  3 = first_weeks →
  5 = first_week_hours →
  hourly_wage = total_earnings / (initial_hours_per_week * initial_weeks) →
  remaining_earnings = total_earnings - (first_week_hours * hourly_wage * first_weeks) →
  remaining_weeks = initial_weeks - first_weeks →
  total_hours_required = remaining_earnings / (hourly_wage * remaining_weeks) →
  total_hours_required = 17.5 :=
by
  intros
  sorry

end remaining_work_hours_l34_34518


namespace factorize_expression_l34_34057

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l34_34057


namespace total_cupcakes_l34_34291

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (total_cupcakes : ℕ) 
  (h1 : children = 8) (h2 : cupcakes_per_child = 12) : total_cupcakes = 96 := 
by
  sorry

end total_cupcakes_l34_34291


namespace stamps_count_l34_34120

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l34_34120


namespace max_value_expr_l34_34539

theorem max_value_expr (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_sum : a + b + c = 2) :
  ∃ M, (∀ a b c, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 2 → (frac_expr a b c) ≤ M) ∧ (exists a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 2 ∧ (frac_expr a b c) = M) :=
by
  have frac_expr := (λ a b c, (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c))
  sorry

end max_value_expr_l34_34539


namespace opposite_of_two_thirds_l34_34575

theorem opposite_of_two_thirds : - (2/3) = -2/3 :=
by
  sorry

end opposite_of_two_thirds_l34_34575


namespace coupon_probability_l34_34585

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l34_34585


namespace prob_white_ball_second_l34_34099

structure Bag :=
  (black_balls : ℕ)
  (white_balls : ℕ)

def total_balls (bag : Bag) := bag.black_balls + bag.white_balls

def prob_white_second_after_black_first (bag : Bag) : ℚ :=
  if bag.black_balls > 0 ∧ bag.white_balls > 0 ∧ total_balls bag > 1 then
    (bag.white_balls : ℚ) / (total_balls bag - 1)
  else 0

theorem prob_white_ball_second 
  (bag : Bag)
  (h_black : bag.black_balls = 4)
  (h_white : bag.white_balls = 3)
  (h_total : total_balls bag = 7) :
  prob_white_second_after_black_first bag = 1 / 2 :=
by
  sorry

end prob_white_ball_second_l34_34099


namespace expected_value_of_win_l34_34311

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l34_34311


namespace mark_team_free_throws_l34_34694

theorem mark_team_free_throws (F : ℕ) : 
  let mark_2_pointers := 25
  let mark_3_pointers := 8
  let opp_2_pointers := 2 * mark_2_pointers
  let opp_3_pointers := 1 / 2 * mark_3_pointers
  let total_points := 201
  2 * mark_2_pointers + 3 * mark_3_pointers + F + 2 * mark_2_pointers + 3 / 2 * mark_3_pointers + F / 2 = total_points →
  F = 10 := by
  sorry

end mark_team_free_throws_l34_34694


namespace find_primes_pqr_eq_5_sum_l34_34647

theorem find_primes_pqr_eq_5_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  p * q * r = 5 * (p + q + r) → (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨
                                         (p = 5 ∧ q = 2 ∧ r = 7) ∨ (p = 5 ∧ q = 7 ∧ r = 2) ∨
                                         (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

end find_primes_pqr_eq_5_sum_l34_34647


namespace tangent_315_deg_l34_34981

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34981


namespace find_x_squared_plus_y_squared_l34_34232

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l34_34232


namespace tan_315_proof_l34_34955

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34955


namespace tan_315_eq_neg_one_l34_34937

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34937


namespace tan_315_degree_l34_34969

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34969


namespace solve_fraction_problem_l34_34345

noncomputable def x_value (a b c d : ℤ) : ℝ :=
  (a + b * Real.sqrt c) / d

theorem solve_fraction_problem (a b c d : ℤ) (h1 : x_value a b c d = (5 + 5 * Real.sqrt 5) / 4)
  (h2 : (4 * x_value a b c d) / 5 - 2 = 5 / x_value a b c d) :
  (a * c * d) / b = 20 := by
  sorry

end solve_fraction_problem_l34_34345


namespace smallest_n_mod5_l34_34199

theorem smallest_n_mod5 :
  ∃ n : ℕ, n > 0 ∧ 6^n % 5 = n^6 % 5 ∧ ∀ m : ℕ, m > 0 ∧ 6^m % 5 = m^6 % 5 → n ≤ m :=
by
  sorry

end smallest_n_mod5_l34_34199


namespace problem_correct_l34_34781

noncomputable def problem := 
  1 - (1 / 2)⁻¹ * Real.sin (60 * Real.pi / 180) + abs (2^0 - Real.sqrt 3) = 0

theorem problem_correct : problem := by
  sorry

end problem_correct_l34_34781


namespace proof_AC_time_l34_34603

noncomputable def A : ℝ := 1/10
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 1/30

def rate_A_B (A B : ℝ) := A + B = 1/6
def rate_B_C (B C : ℝ) := B + C = 1/10
def rate_A_B_C (A B C : ℝ) := A + B + C = 1/5

theorem proof_AC_time {A B C : ℝ} (h1 : rate_A_B A B) (h2 : rate_B_C B C) (h3 : rate_A_B_C A B C) : 
  (1 : ℝ) / (A + C) = 7.5 :=
sorry

end proof_AC_time_l34_34603


namespace tan_315_eq_neg1_l34_34986

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34986


namespace fractional_part_shaded_eq_four_fifths_l34_34899

-- defining the conditions
def large_square_divided : Prop :=
  ∀ (n : ℕ), let area := 1 in area / 16^(n+1)

def shaded_squares_sequence (n : ℕ) : ℚ :=
  12 / 16^(n + 1)

-- proving the fractional part theorem
theorem fractional_part_shaded_eq_four_fifths : 
  (∑' n, shaded_squares_sequence n) = 4 / 5 := 
sorry

end fractional_part_shaded_eq_four_fifths_l34_34899


namespace min_cubes_required_l34_34033

theorem min_cubes_required (length width height volume_cube : ℝ) 
  (h_length : length = 14.5) 
  (h_width : width = 17.8) 
  (h_height : height = 7.2) 
  (h_volume_cube : volume_cube = 3) : 
  ⌈(length * width * height) / volume_cube⌉ = 624 := sorry

end min_cubes_required_l34_34033


namespace probability_G_is_one_fourth_l34_34902

-- Definitions and conditions
variables (p_E p_F p_G p_H : ℚ)
axiom probability_E : p_E = 1/3
axiom probability_F : p_F = 1/6
axiom prob_G_eq_H : p_G = p_H
axiom total_prob_sum : p_E + p_F + p_G + p_G = 1

-- Theorem statement
theorem probability_G_is_one_fourth : p_G = 1/4 :=
by 
  -- Lean proof omitted, only the statement required
  sorry

end probability_G_is_one_fourth_l34_34902


namespace rachel_picked_apples_l34_34126

-- Defining the conditions
def original_apples : ℕ := 11
def grown_apples : ℕ := 2
def apples_left : ℕ := 6

-- Defining the equation
def equation (x : ℕ) : Prop :=
  original_apples - x + grown_apples = apples_left

-- Stating the theorem
theorem rachel_picked_apples : ∃ x : ℕ, equation x ∧ x = 7 :=
by 
  -- proof skipped 
  sorry

end rachel_picked_apples_l34_34126


namespace coefficient_x3y7_in_expansion_l34_34742

theorem coefficient_x3y7_in_expansion :
  let expression := (2/3 : ℚ) * x - (1/3 : ℚ) * y,
      expansion := (expression ^ 10 : ℚ),
      term := finset.sum finset.univ (λ (k : ℕ), (nat.choose 10 k) * ((2/3 * x)^k) * ((-1/3 * y)^(10 - k))),
      target_term := (x^3 * y^7) : ℚ
  in term.coeff (x^3 * y^7) = (x^3 * y^7) * (-960/59049 : ℚ) :=
sorry

end coefficient_x3y7_in_expansion_l34_34742


namespace solve_system_of_equations_l34_34705

theorem solve_system_of_equations (x y z : ℝ) : 
  (y * z = 3 * y + 2 * z - 8) ∧
  (z * x = 4 * z + 3 * x - 8) ∧
  (x * y = 2 * x + y - 1) ↔ 
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5 / 2 ∧ z = -1)) :=
by
  sorry

end solve_system_of_equations_l34_34705


namespace option_C_correct_l34_34541

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for parallel and perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def line_parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement based on problem c) translation
theorem option_C_correct (H1 : line_parallel m n) (H2 : perpendicular m α) : perpendicular n α :=
sorry

end option_C_correct_l34_34541


namespace find_f_19_l34_34525

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the given function

-- Define the conditions
axiom even_function : ∀ x : ℝ, f x = f (-x) 
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x

-- The statement we need to prove
theorem find_f_19 : f 19 = 0 := 
by
  sorry -- placeholder for the proof

end find_f_19_l34_34525


namespace probability_B_wins_at_least_one_match_l34_34529

theorem probability_B_wins_at_least_one_match :
  let P_A := 0.5
  let P_B := 0.3
  let P_T := 0.2
  let P_B_not_winning := 1 - P_B
  let P_B_wins_at_least_one := (P_B * P_B) + (P_B * P_B_not_winning) + (P_B_not_winning * P_B)
  P_B_wins_at_least_one = 0.51 :=
by
  let P_A := 0.5
  let P_B := 0.3
  let P_T := 0.2
  let P_B_not_winning := 1 - P_B
  let P_B_wins_at_least_one := (P_B * P_B) + (P_B * P_B_not_winning) + (P_B_not_winning * P_B)
  show P_B_wins_at_least_one = 0.51
  sorry

end probability_B_wins_at_least_one_match_l34_34529


namespace opposite_of_neg_two_l34_34716

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l34_34716


namespace fraction_addition_simplified_form_l34_34346

theorem fraction_addition_simplified_form :
  (7 / 8) + (3 / 5) = 59 / 40 := 
by sorry

end fraction_addition_simplified_form_l34_34346


namespace expression_equals_a5_l34_34883

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end expression_equals_a5_l34_34883


namespace base_n_multiple_of_5_l34_34652

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 4 + n + 3 * n^2 + 5 * n^3 + n^4 + 4 * n^5

-- The main theorem to be proven
theorem base_n_multiple_of_5 (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 100) : 
  f n % 5 ≠ 0 :=
by sorry

end base_n_multiple_of_5_l34_34652


namespace inequality_proof_l34_34062

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b + b^2 / c + c^2 / a) + (a + b + c) ≥ (6 * (a^2 + b^2 + c^2) / (a + b + c)) :=
by
  sorry

end inequality_proof_l34_34062


namespace wall_width_8_l34_34612

theorem wall_width_8 (w h l : ℝ) (V : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h) 
  (vol_eq : w * h * l = 129024) : 
  w = 8 := 
by 
  sorry

end wall_width_8_l34_34612


namespace value_of_2a_minus_1_l34_34221

theorem value_of_2a_minus_1 (a : ℝ) (h : ∀ x : ℝ, (x = 2 → (3 / 2) * x - 2 * a = 0)) : 2 * a - 1 = 2 :=
sorry

end value_of_2a_minus_1_l34_34221


namespace tan_315_degree_l34_34972

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34972


namespace tan_315_eq_neg1_l34_34983

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34983


namespace sin_gt_sub_cubed_l34_34796

theorem sin_gt_sub_cubed (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  Real.sin x > x - x^3 / 6 := 
by 
  sorry

end sin_gt_sub_cubed_l34_34796


namespace find_integer_for_prime_l34_34058

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem find_integer_for_prime (n : ℤ) :
  is_prime (4 * n^4 + 1) ↔ n = 1 :=
by
  sorry

end find_integer_for_prime_l34_34058


namespace total_blocks_l34_34194

def initial_blocks := 2
def multiplier := 3
def father_blocks := multiplier * initial_blocks

theorem total_blocks :
  initial_blocks + father_blocks = 8 :=
by 
  -- skipping the proof with sorry
  sorry

end total_blocks_l34_34194


namespace not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l34_34764

noncomputable def f (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem not_bounded_on_neg_infty_zero (a x : ℝ) (h : a = 1) : 
  ¬ ∃ M > 0, ∀ x < 0, |f a x| ≤ M :=
by sorry

-- Problem (2)
theorem range_of_a_bounded_on_zero_infty (a : ℝ) : 
  (∀ x ≥ 0, |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l34_34764


namespace simplify_expression_value_at_3_value_at_4_l34_34414

-- Define the original expression
def original_expr (x : ℕ) : ℚ := (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2 * x + 1))

-- Property 1: Simplify the expression
theorem simplify_expression (x : ℕ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  original_expr x = (x - 1) / (x + 2) :=
sorry

-- Property 2: Evaluate the expression at x = 3
theorem value_at_3 : original_expr 3 = 2 / 5 :=
sorry

-- Property 3: Evaluate the expression at x = 4
theorem value_at_4 : original_expr 4 = 1 / 2 :=
sorry

end simplify_expression_value_at_3_value_at_4_l34_34414


namespace number_of_children_l34_34353

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l34_34353


namespace min_f_value_l34_34650

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (∫ θ in 0..x, (1 / cos θ)) + (∫ θ in x..(π / 2), (1 / sin θ))

theorem min_f_value : ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ is_minimum f x ∧ f x = ln (3 + 2 * sqrt 2) :=
sorry

end min_f_value_l34_34650


namespace dan_helmet_crater_difference_l34_34785

theorem dan_helmet_crater_difference :
  ∀ (r d : ℕ), 
  (r = 75) ∧ (d = 35) ∧ (r = 15 + (d + (r - 15 - d))) ->
  ((d - (r - 15 - d)) = 10) :=
by
  intros r d h
  have hr : r = 75 := h.1
  have hd : d = 35 := h.2.1
  have h_combined : r = 15 + (d + (r - 15 - d)) := h.2.2
  sorry

end dan_helmet_crater_difference_l34_34785


namespace largest_possible_product_is_3886_l34_34409

theorem largest_possible_product_is_3886 :
  ∃ a b c d : ℕ, 5 ≤ a ∧ a ≤ 8 ∧
               5 ≤ b ∧ b ≤ 8 ∧
               5 ≤ c ∧ c ≤ 8 ∧
               5 ≤ d ∧ d ≤ 8 ∧
               a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
               b ≠ c ∧ b ≠ d ∧
               c ≠ d ∧
               (max ((10 * a + b) * (10 * c + d))
                    ((10 * c + b) * (10 * a + d))) = 3886 :=
sorry

end largest_possible_product_is_3886_l34_34409


namespace tan_315_eq_neg1_l34_34943

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34943


namespace system_solution_l34_34704

theorem system_solution (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧ 
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4) ∧ (y = -1) :=
by
  sorry

end system_solution_l34_34704


namespace tan_315_eq_neg1_l34_34966

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34966


namespace sin_2x_from_tan_pi_minus_x_l34_34498

theorem sin_2x_from_tan_pi_minus_x (x : ℝ) (h : Real.tan (Real.pi - x) = 3) : Real.sin (2 * x) = -3 / 5 := by
  sorry

end sin_2x_from_tan_pi_minus_x_l34_34498


namespace find_A_plus_C_l34_34103

-- This will bring in the entirety of the necessary library and supports the digit verification and operations.

-- Definitions of digits and constraints
variables {A B C D : ℕ}

-- Given conditions in the problem
def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

def multiplication_condition_1 (A B C D : ℕ) : Prop :=
  C * D = A

def multiplication_condition_2 (A B C D : ℕ) : Prop :=
  10 * B * D + C * D = 11 * C

-- The final problem statement
theorem find_A_plus_C (A B C D : ℕ) (h1 : distinct_digits A B C D) 
  (h2 : multiplication_condition_1 A B C D) 
  (h3 : multiplication_condition_2 A B C D) : 
  A + C = 10 :=
sorry

end find_A_plus_C_l34_34103


namespace solve_symbols_values_l34_34469

def square_value : Nat := 423 / 47

def boxminus_and_boxtimes_relation (boxminus boxtimes : Nat) : Prop :=
  1448 = 282 * boxminus + 9 * boxtimes

def boxtimes_value : Nat := 38 / 9

def boxplus_value : Nat := 846 / 423

theorem solve_symbols_values :
  ∃ (square boxplus boxtimes boxminus : Nat),
    square = 9 ∧
    boxplus = 2 ∧
    boxtimes = 8 ∧
    boxminus = 5 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + 9 * boxtimes ∧
    9 * boxtimes = 38 ∧
    423 * boxplus / 3 = 282 := by
  sorry

end solve_symbols_values_l34_34469


namespace tan_315_eq_neg1_l34_34924

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34924


namespace cone_csa_l34_34297

theorem cone_csa (r l : ℝ) (h_r : r = 8) (h_l : l = 18) : 
  (Real.pi * r * l) = 144 * Real.pi :=
by 
  rw [h_r, h_l]
  norm_num
  sorry

end cone_csa_l34_34297


namespace annette_weights_more_l34_34910

variable (A C S B : ℝ)

theorem annette_weights_more :
  A + C = 95 ∧
  C + S = 87 ∧
  A + S = 97 ∧
  C + B = 100 ∧
  A + C + B = 155 →
  A - S = 8 := by
  sorry

end annette_weights_more_l34_34910


namespace find_a2018_l34_34104

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 0.5 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

-- The statement to prove
theorem find_a2018 (a : ℕ → ℝ) (h : seq a) : a 2018 = -1 := by
  sorry

end find_a2018_l34_34104


namespace opposite_of_neg_two_l34_34730

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l34_34730


namespace parabola_line_dot_product_l34_34072

theorem parabola_line_dot_product (k x1 x2 y1 y2 : ℝ) 
  (h_line: ∀ x, y = k * x + 2)
  (h_parabola: ∀ x, y = (1 / 4) * x ^ 2) 
  (h_A: y1 = k * x1 + 2 ∧ y1 = (1 / 4) * x1 ^ 2)
  (h_B: y2 = k * x2 + 2 ∧ y2 = (1 / 4) * x2 ^ 2) :
  x1 * x2 + y1 * y2 = -4 := 
sorry

end parabola_line_dot_product_l34_34072


namespace altered_solution_contains_60_liters_of_detergent_l34_34613

-- Definitions corresponding to the conditions
def initial_ratio_bleach_to_detergent_to_water : ℚ := 2 / 40 / 100
def initial_ratio_bleach_to_detergent : ℚ := 1 / 20
def initial_ratio_detergent_to_water : ℚ := 1 / 5

def altered_ratio_bleach_to_detergent : ℚ := 3 / 20
def altered_ratio_detergent_to_water : ℚ := 1 / 5

def water_in_altered_solution : ℚ := 300

-- We need to find the amount of detergent in the altered solution
def amount_of_detergent_in_altered_solution : ℚ := 20

-- The proportion and the final amount calculation
theorem altered_solution_contains_60_liters_of_detergent :
  (300 / 100) * (20) = 60 :=
by
  sorry

end altered_solution_contains_60_liters_of_detergent_l34_34613


namespace find_px_l34_34626

theorem find_px (p : ℕ → ℚ) (h1 : p 1 = 1) (h2 : p 2 = 1 / 4) (h3 : p 3 = 1 / 9) 
  (h4 : p 4 = 1 / 16) (h5 : p 5 = 1 / 25) : p 6 = 1 / 18 :=
sorry

end find_px_l34_34626


namespace total_walking_time_l34_34892

open Nat

def walking_time (distance speed : ℕ) : ℕ :=
distance / speed

def number_of_rests (distance : ℕ) : ℕ :=
(distance / 10) - 1

def resting_time_in_minutes (rests : ℕ) : ℕ :=
rests * 5

def resting_time_in_hours (rest_time : ℕ) : ℚ :=
rest_time / 60

def total_time (walking_time resting_time : ℚ) : ℚ :=
walking_time + resting_time

theorem total_walking_time (distance speed : ℕ) (rest_per_10 : ℕ) (rest_time : ℕ) :
  speed = 10 →
  rest_per_10 = 10 →
  rest_time = 5 →
  distance = 50 →
  total_time (walking_time distance speed) (resting_time_in_hours (resting_time_in_minutes (number_of_rests distance))) = 5 + 1 / 3 :=
sorry

end total_walking_time_l34_34892


namespace original_price_of_car_l34_34493

theorem original_price_of_car (P : ℝ) 
  (h₁ : 0.561 * P + 200 = 7500) : 
  P = 13012.48 := 
sorry

end original_price_of_car_l34_34493


namespace total_distance_travelled_l34_34079

-- Definition of the given conditions
def distance_to_market := 30  -- miles
def time_to_home := 0.5  -- hours
def speed_to_home := 20  -- miles per hour

-- The statement we want to prove: Total distance traveled is 40 miles.
theorem total_distance_travelled : 
  distance_to_market + speed_to_home * time_to_home = 40 := 
by 
  sorry

end total_distance_travelled_l34_34079


namespace cost_of_green_shirts_l34_34139

noncomputable def total_cost_kindergarten : ℝ := 101 * 5.8
noncomputable def total_cost_first_grade : ℝ := 113 * 5
noncomputable def total_cost_second_grade : ℝ := 107 * 5.6
noncomputable def total_cost_all_but_third : ℝ := total_cost_kindergarten + total_cost_first_grade + total_cost_second_grade
noncomputable def total_third_grade : ℝ := 2317 - total_cost_all_but_third
noncomputable def cost_per_third_grade_shirt : ℝ := total_third_grade / 108

theorem cost_of_green_shirts : cost_per_third_grade_shirt = 5.25 := sorry

end cost_of_green_shirts_l34_34139


namespace relationship_among_abc_l34_34755

noncomputable def a := Real.sqrt 5 + 2
noncomputable def b := 2 - Real.sqrt 5
noncomputable def c := Real.sqrt 5 - 2

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end relationship_among_abc_l34_34755


namespace min_positive_period_abs_tan_2x_l34_34441

theorem min_positive_period_abs_tan_2x : 
  ∃ T > 0, (∀ x : ℝ, |Real.tan (2 * x) + T| = |Real.tan (2 * x)|)
  ∧ (∀ T' > 0, (∀ x : ℝ, |Real.tan (2 * (x + T'))| = |Real.tan (2 * x) → T' ≥ T)) :=
sorry

end min_positive_period_abs_tan_2x_l34_34441


namespace card_probability_correct_l34_34665

noncomputable def calc_card_prob (total_cards : ℕ) (draws : ℕ) : ℚ :=
  (1 / (rat.of_nat total_cards)) ^ draws

def card_prob_scenario : Prop :=
  let total_cards := 52
  let draws := 5
  let suit_cards := 13
  let prob_first_card := 1
  let prob_other_suits := (suit_cards * 3 / total_cards)   -- cards from other suits for the second draw
  let prob_third_fourth_fifth :=
    (suit_cards * 2 / total_cards) *  -- drawing remaining suit cards
    (suit_cards / total_cards) *       -- as constrained
    (suit_cards / total_cards)         -- last needed conditions
  
  prob_first_card * prob_other_suits * prob_third_fourth_fifth = 15 / 512

theorem card_probability_correct : card_prob_scenario :=
by {
  sorry
}

end card_probability_correct_l34_34665


namespace andrei_stamps_l34_34122

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l34_34122


namespace managers_meeting_l34_34317

/-- A meeting has to be conducted with 4 managers out of 7 managers.
Two specific managers (A and B) will not attend the meeting together.
Prove that the number of ways to select the managers for the meeting is 25. -/
theorem managers_meeting : 
  let total_managers := 7
  let choose_managers := 4
  let specific_managers_not_together := 2
  (∑ k in {0, 1}, Nat.choose (total_managers - specific_managers_not_together) (choose_managers - k) * Nat.choose specific_managers_not_together k) = 25 :=
by
  sorry

end managers_meeting_l34_34317


namespace num_triangles_square_even_num_triangles_rect_even_l34_34161

-- Problem (a): Proving that the number of triangles is even 
theorem num_triangles_square_even (a : ℕ) (n : ℕ) (h : a * a = n * (3 * 4 / 2)) : 
  n % 2 = 0 :=
sorry

-- Problem (b): Proving that the number of triangles is even
theorem num_triangles_rect_even (L W k : ℕ) (hL : L = k * 2) (hW : W = k * 1) (h : L * W = k * 1 * 2 / 2) :
  k % 2 = 0 :=
sorry

end num_triangles_square_even_num_triangles_rect_even_l34_34161


namespace range_of_f_lt_zero_l34_34415

noncomputable
def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero 
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_neg2_zero : f (-2) = 0) :
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end range_of_f_lt_zero_l34_34415


namespace range_of_m_l34_34668

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end range_of_m_l34_34668


namespace distance_p_ran_l34_34264

variable (d t v : ℝ)
-- d: head start distance in meters
-- t: time in minutes
-- v: speed of q in meters per minute

theorem distance_p_ran (h1 : d = 0.3 * v * t) : 1.3 * v * t = 1.3 * v * t :=
by
  sorry

end distance_p_ran_l34_34264


namespace ab_cd_divisible_eq_one_l34_34574

theorem ab_cd_divisible_eq_one (a b c d : ℕ) (h1 : ∃ e : ℕ, e = ab - cd ∧ (e ∣ a) ∧ (e ∣ b) ∧ (e ∣ c) ∧ (e ∣ d)) : ab - cd = 1 :=
sorry

end ab_cd_divisible_eq_one_l34_34574


namespace opposite_neg_two_l34_34709

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l34_34709


namespace area_ratio_of_squares_l34_34535

theorem area_ratio_of_squares (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let E := (s / 2,  -s / 2)
  let F := (3 * s / 2, s / 2)
  let G := (s / 2, 3 * s / 2)
  let H := (-s / 2, s / 2)
  let area_ABCD := s * s
  let side_EFGH := Real.sqrt ((3 * s / 2 - s / 2) ^ 2 + (s / 2 + s / 2) ^ 2)
  let area_EFGH := side_EFGH ^ 2
  in area_EFGH / area_ABCD = 5 / 2 :=
by
  intros
  sorry
  
end area_ratio_of_squares_l34_34535


namespace constant_term_is_24_l34_34377

noncomputable def constant_term_of_binomial_expansion 
  (a : ℝ) (hx : π * a^2 = 4 * π) : ℝ :=
  if ha : a = 2 then 24 else 0

theorem constant_term_is_24
  (a : ℝ) (hx : π * a^2 = 4 * π) :
  constant_term_of_binomial_expansion a hx = 24 :=
by
  sorry

end constant_term_is_24_l34_34377


namespace travel_speed_l34_34179

theorem travel_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 195) (h_time : time = 3) : 
  distance / time = 65 :=
by 
  rw [h_distance, h_time]
  norm_num

end travel_speed_l34_34179


namespace inequality_am_gm_cauchy_schwarz_equality_iff_l34_34563

theorem inequality_am_gm_cauchy_schwarz 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_iff (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) 
  ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_am_gm_cauchy_schwarz_equality_iff_l34_34563


namespace tangent_315_deg_l34_34979

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34979


namespace total_simple_interest_l34_34630

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest : simple_interest 2500 10 4 = 1000 := 
by
  sorry

end total_simple_interest_l34_34630


namespace alligator_population_at_end_of_year_l34_34868

-- Define the conditions
def initial_population : ℕ := 4
def doubling_period_months : ℕ := 6
def total_months : ℕ := 12

-- Define the proof goal
theorem alligator_population_at_end_of_year (initial_population doubling_period_months total_months : ℕ)
  (h_init : initial_population = 4)
  (h_double : doubling_period_months = 6)
  (h_total : total_months = 12) :
  initial_population * (2 ^ (total_months / doubling_period_months)) = 16 := 
by
  sorry

end alligator_population_at_end_of_year_l34_34868


namespace number_of_initials_is_10000_l34_34808

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l34_34808


namespace smallest_b_for_factorization_l34_34060

theorem smallest_b_for_factorization : ∃ (b : ℕ), (∀ p q : ℤ, (x^2 + (b * x) + 2352) = (x + p) * (x + q) → p + q = b ∧ p * q = 2352) ∧ b = 112 := 
sorry

end smallest_b_for_factorization_l34_34060


namespace ratio_y_to_x_l34_34196

variable (x y z : ℝ)

-- Conditions
def condition1 (x y z : ℝ) := 0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)
def condition2 (y z : ℝ) := ∃ k : ℝ, z = k * y
def condition3 (y z : ℝ) := z = 7 * y
def condition4 (x y : ℝ) := y = 5 * x / 7

theorem ratio_y_to_x (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 y z) (h3 : condition3 y z) (h4 : condition4 x y) : y / x = 5 / 7 :=
by
  sorry

end ratio_y_to_x_l34_34196


namespace mass_percentage_H_calculation_l34_34797

noncomputable def molar_mass_CaH2 : ℝ := 42.09
noncomputable def molar_mass_H2O : ℝ := 18.015
noncomputable def molar_mass_H2SO4 : ℝ := 98.079

noncomputable def moles_CaH2 : ℕ := 3
noncomputable def moles_H2O : ℕ := 4
noncomputable def moles_H2SO4 : ℕ := 2

noncomputable def mass_H_CaH2 : ℝ := 3 * 2 * 1.008
noncomputable def mass_H_H2O : ℝ := 4 * 2 * 1.008
noncomputable def mass_H_H2SO4 : ℝ := 2 * 2 * 1.008

noncomputable def total_mass_H : ℝ :=
  mass_H_CaH2 + mass_H_H2O + mass_H_H2SO4

noncomputable def total_mass_mixture : ℝ :=
  (moles_CaH2 * molar_mass_CaH2) + (moles_H2O * molar_mass_H2O) + (moles_H2SO4 * molar_mass_H2SO4)

noncomputable def mass_percentage_H : ℝ :=
  (total_mass_H / total_mass_mixture) * 100

theorem mass_percentage_H_calculation :
  abs (mass_percentage_H - 4.599) < 0.001 :=
by
  sorry

end mass_percentage_H_calculation_l34_34797


namespace alligator_doubling_l34_34867

theorem alligator_doubling (initial_alligators : ℕ) (doubling_period_months : ℕ) : 
  initial_alligators = 4 → doubling_period_months = 6 → 
  let final_alligators := initial_alligators * 2^2 in 
  final_alligators = 16 :=
by
  sorry

end alligator_doubling_l34_34867


namespace incorrect_statement_l34_34092

theorem incorrect_statement (p q : Prop) (hp : ¬ p) (hq : q) : ¬ (¬ q) :=
by
  sorry

end incorrect_statement_l34_34092


namespace solve_real_numbers_l34_34342

theorem solve_real_numbers (x y : ℝ) :
  (x = 3 * x^2 * y - y^3) ∧ (y = x^3 - 3 * x * y^2) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 + Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = (Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 + Real.sqrt 2)) / 2)) :=
by
  sorry

end solve_real_numbers_l34_34342


namespace tan_315_eq_neg_one_l34_34996

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34996


namespace arithmetic_sequence_common_diff_l34_34279

theorem arithmetic_sequence_common_diff (d : ℝ) (a : ℕ → ℝ) 
  (h_first_term : a 0 = 24) 
  (h_arithmetic_sequence : ∀ n, a (n + 1) = a n + d)
  (h_ninth_term_nonneg : 24 + 8 * d ≥ 0) 
  (h_tenth_term_neg : 24 + 9 * d < 0) : 
  -3 ≤ d ∧ d < -8/3 :=
by 
  sorry

end arithmetic_sequence_common_diff_l34_34279


namespace opposite_of_neg_two_l34_34715

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l34_34715


namespace butterfat_milk_mixture_l34_34609

theorem butterfat_milk_mixture :
  ∃ (x : ℝ), 0.10 * x + 0.45 * 8 = 0.20 * (x + 8) ∧ x = 20 := by
  sorry

end butterfat_milk_mixture_l34_34609


namespace opposite_of_neg_two_l34_34724

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l34_34724


namespace option_C_equals_a5_l34_34887

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end option_C_equals_a5_l34_34887


namespace sum_of_common_ratios_of_sequences_l34_34687

def arithmetico_geometric_sequence (a b c : ℕ → ℝ) (r : ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n + d ∧ b (n + 1) = r * b n + d

theorem sum_of_common_ratios_of_sequences {m n : ℝ}
    {a1 a2 a3 b1 b2 b3 : ℝ}
    (p q : ℝ)
    (h_a1 : a1 = m)
    (h_a2 : a2 = m * p + 5)
    (h_a3 : a3 = m * p^2 + 5 * p + 5)
    (h_b1 : b1 = n)
    (h_b2 : b2 = n * q + 5)
    (h_b3 : b3 = n * q^2 + 5 * q + 5)
    (h_cond : a3 - b3 = 3 * (a2 - b2)) :
    p + q = 4 :=
by
  sorry

end sum_of_common_ratios_of_sequences_l34_34687


namespace smallest_rational_number_is_neg_one_l34_34187

theorem smallest_rational_number_is_neg_one : 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  (min (min a b) (min c d)) = d := 
by 
  let a := -6 / 7
  let b := 2
  let c := 0
  let d := -1
  have h1 : a < 0 := by norm_num
  have h2 : d < 0 := by norm_num
  have h3 : c = 0 := by norm_num
  have h4 : b > 0 := by norm_num
  have h5 : abs a = 6 / 7 := by norm_cast
  have h6 : abs d = 1 := by norm_num
  have h7 : abs a < abs d := by norm_num
  have h8 : d < a := by linarith
  exact min_eq_right h8

end smallest_rational_number_is_neg_one_l34_34187


namespace necessary_and_sufficient_condition_l34_34859

theorem necessary_and_sufficient_condition (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ 0 > b :=
by
  sorry

end necessary_and_sufficient_condition_l34_34859


namespace lattice_points_count_l34_34083

theorem lattice_points_count : ∃ n : ℕ, n = 8 ∧ (∃ x y : ℤ, x^2 - y^2 = 51) :=
by
  sorry

end lattice_points_count_l34_34083


namespace solution_is_permutations_of_2_neg2_4_l34_34197

-- Definitions of the conditions
def cond1 (x y z : ℤ) : Prop := x * y + y * z + z * x = -4
def cond2 (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 24
def cond3 (x y z : ℤ) : Prop := x^3 + y^3 + z^3 + 3 * x * y * z = 16

-- The set of all integer solutions as permutations of (2, -2, 4)
def is_solution (x y z : ℤ) : Prop :=
  (x = 2 ∧ y = -2 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = -2) ∨
  (x = -2 ∧ y = 2 ∧ z = 4) ∨ (x = -2 ∧ y = 4 ∧ z = 2) ∨
  (x = 4 ∧ y = 2 ∧ z = -2) ∨ (x = 4 ∧ y = -2 ∧ z = 2)

-- Lean statement for the proof problem
theorem solution_is_permutations_of_2_neg2_4 (x y z : ℤ) :
  cond1 x y z → cond2 x y z → cond3 x y z → is_solution x y z :=
by
  -- sorry, the proof goes here
  sorry

end solution_is_permutations_of_2_neg2_4_l34_34197


namespace math_problem_l34_34878

theorem math_problem :
  3 ^ (2 + 4 + 6) - (3 ^ 2 + 3 ^ 4 + 3 ^ 6) + (3 ^ 2 * 3 ^ 4 * 3 ^ 6) = 1062242 :=
by
  sorry

end math_problem_l34_34878


namespace find_widgets_l34_34301

theorem find_widgets (a b c d e f : ℕ) : 
  (3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255) →
  (3 ^ a * 11 ^ b * 5 ^ c * 7 ^ d * 13 ^ e * 17 ^ f = 351125648000) →
  c = 3 :=
by
  sorry

end find_widgets_l34_34301


namespace exists_infinite_arith_prog_exceeding_M_l34_34560

def sum_of_digits(n : ℕ) : ℕ :=
n.digits 10 |> List.sum

theorem exists_infinite_arith_prog_exceeding_M (M : ℝ) :
  ∃ (a d : ℕ), ¬ (10 ∣ d) ∧ (∀ n : ℕ, a + n * d > 0) ∧ (∀ n : ℕ, sum_of_digits (a + n * d) > M) := by
sorry

end exists_infinite_arith_prog_exceeding_M_l34_34560


namespace unique_positive_integer_solution_l34_34011

theorem unique_positive_integer_solution (n p : ℕ) (x y : ℕ) :
  (x + p * y = n ∧ x + y = p^2 ∧ x > 0 ∧ y > 0) ↔ 
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ, n ≠ p^k ∧ ∃! t : ℕ × ℕ, (t.1 + p * t.2 = n ∧ t.1 + t.2 = p^2 ∧ t.1 > 0 ∧ t.2 > 0)) :=
by
  sorry

end unique_positive_integer_solution_l34_34011


namespace sports_lottery_systematic_sampling_l34_34252

-- Definition of the sports lottery condition
def is_first_prize_ticket (n : ℕ) : Prop := n % 1000 = 345

-- Statement of the proof problem
theorem sports_lottery_systematic_sampling :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → is_first_prize_ticket n) →
  ∃ interval, (∀ segment_start : ℕ,  segment_start < 1000 → is_first_prize_ticket (segment_start + interval * 999))
  := by sorry

end sports_lottery_systematic_sampling_l34_34252


namespace max_fraction_sum_l34_34538

theorem max_fraction_sum (a b c : ℝ) 
  (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0)
  (h_sum: a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_fraction_sum_l34_34538


namespace schedule_arrangements_l34_34760

open Finset

noncomputable def num_valid_arrangements : ℕ :=
  card {l : List ℕ // 
    l ~ [0, 1, 2, 3, 4, 5] ∧ -- l is a permutation of [0, 1, 2, 3, 4, 5]
    l.head ≠ 5 ∧ -- Physical Education (5) is not in the first period
    l.nth 3 ≠ some 1  -- Mathematics (1) is not in the fourth period
  }

theorem schedule_arrangements : num_valid_arrangements = 480 := 
by
  sorry

end schedule_arrangements_l34_34760


namespace gcd_of_three_numbers_l34_34208

-- Definition of the numbers we are interested in
def a : ℕ := 9118
def b : ℕ := 12173
def c : ℕ := 33182

-- Statement of the problem to prove GCD
theorem gcd_of_three_numbers : Int.gcd (Int.gcd a b) c = 47 := 
sorry  -- Proof skipped

end gcd_of_three_numbers_l34_34208


namespace students_taking_all_three_classes_l34_34866

variables (total_students Y B P N : ℕ)
variables (X₁ X₂ X₃ X₄ : ℕ)  -- variables representing students taking exactly two classes or all three

theorem students_taking_all_three_classes:
  total_students = 20 →
  Y = 10 →  -- Number of students taking yoga
  B = 13 →  -- Number of students taking bridge
  P = 9 →   -- Number of students taking painting
  N = 9 →   -- Number of students taking at least two classes
  X₂ + X₃ + X₄ = 9 →  -- This equation represents the total number of students taking at least two classes, where \( X₄ \) represents students taking all three (c).
  4 + X₃ + X₄ - (9 - X₃) + 1 + (9 - X₄ - X₂) + X₂ = 11 →
  X₄ = 3 :=                     -- Proving that the number of students taking all three classes is 3.
sorry

end students_taking_all_three_classes_l34_34866


namespace shaded_area_l34_34180

-- Define the points as per the problem
structure Point where
  x : ℝ
  y : ℝ

@[simp]
def A : Point := ⟨0, 0⟩
@[simp]
def B : Point := ⟨0, 7⟩
@[simp]
def C : Point := ⟨7, 7⟩
@[simp]
def D : Point := ⟨7, 0⟩
@[simp]
def E : Point := ⟨7, 0⟩
@[simp]
def F : Point := ⟨14, 0⟩
@[simp]
def G : Point := ⟨10.5, 7⟩

-- Define function for area of a triangle given three points
def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((P.x - R.x) * (Q.y - P.y) - (P.x - Q.x) * (R.y - P.y))

-- The theorem stating the area of the shaded region
theorem shaded_area : triangle_area D G H - triangle_area D E H = 24.5 := by
  sorry

end shaded_area_l34_34180


namespace walkway_area_correct_l34_34848

/-- Define the dimensions of a single flower bed. --/
def flower_bed_length : ℝ := 8
def flower_bed_width : ℝ := 3

/-- Define the number of flower beds in rows and columns. --/
def rows : ℕ := 4
def cols : ℕ := 3

/-- Define the width of the walkways surrounding the flower beds. --/
def walkway_width : ℝ := 2

/-- Calculate the total dimensions of the garden including walkways. --/
def total_garden_width : ℝ := (cols * flower_bed_length) + ((cols + 1) * walkway_width)
def total_garden_height : ℝ := (rows * flower_bed_width) + ((rows + 1) * walkway_width)

/-- Calculate the total area of the garden including walkways. --/
def total_garden_area : ℝ := total_garden_width * total_garden_height

/-- Calculate the total area of the flower beds. --/
def flower_bed_area : ℝ := flower_bed_length * flower_bed_width
def total_flower_beds_area : ℝ := rows * cols * flower_bed_area

/-- Calculate the total area of the walkways. --/
def walkway_area := total_garden_area - total_flower_beds_area

theorem walkway_area_correct : walkway_area = 416 := 
by
  -- Proof omitted
  sorry

end walkway_area_correct_l34_34848


namespace complex_modulus_eq_one_l34_34655

open Complex

theorem complex_modulus_eq_one (a b : ℝ) (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 2 - Complex.I) :
  abs (a - b * Complex.I) = 1 := by
  sorry

end complex_modulus_eq_one_l34_34655


namespace calc_sqrt_expr_l34_34042

theorem calc_sqrt_expr :
  (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end calc_sqrt_expr_l34_34042


namespace not_a_function_l34_34463

theorem not_a_function (angle_sine : ℝ → ℝ) 
                       (side_length_area : ℝ → ℝ) 
                       (sides_sum_int_angles : ℕ → ℝ)
                       (person_age_height : ℕ → Set ℝ) :
  (∃ y₁ y₂, y₁ ∈ person_age_height 20 ∧ y₂ ∈ person_age_height 20 ∧ y₁ ≠ y₂) :=
by {
  sorry
}

end not_a_function_l34_34463


namespace problem_probability_l34_34050

open ProbabilityTheory

/-- Two boxes labeled as A and B, each containing four balls with labels 1, 2, 3, and 4.
    One ball is drawn from each box, and each ball has an equal chance of being drawn.
    (1) Prove that the probability that the two drawn balls have consecutive numbers is 3/8.
    (2) Prove that the probability that the sum of the numbers on the two drawn balls is divisible by 3 is 5/16. -/
theorem problem_probability :
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)] in
  let consecutive := [(1,2), (2,1), (2,3), (3,2), (3,4), (4,3)] in
  let divisible_by_3 := [(1,2), (2,1), (2,4), (3,3), (4,2)] in
  (prob.of_finset (outcomes.to_finset) consecutive) = 3 / 8 ∧
  (prob.of_finset (outcomes.to_finset) divisible_by_3) = 5 / 16 := by
    sorry

end problem_probability_l34_34050


namespace abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l34_34615

theorem abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0 :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ (¬ ∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l34_34615


namespace tan_315_eq_neg1_l34_34954

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34954


namespace initial_money_in_wallet_l34_34455

theorem initial_money_in_wallet (x : ℝ) 
  (h1 : x = 78 + 16) : 
  x = 94 :=
by
  sorry

end initial_money_in_wallet_l34_34455


namespace junior_score_proof_l34_34380

noncomputable def class_total_score (total_students : ℕ) (average_class_score : ℕ) : ℕ :=
total_students * average_class_score

noncomputable def number_of_juniors (total_students : ℕ) (percent_juniors : ℕ) : ℕ :=
percent_juniors * total_students / 100

noncomputable def number_of_seniors (total_students juniors : ℕ) : ℕ :=
total_students - juniors

noncomputable def total_senior_score (seniors average_senior_score : ℕ) : ℕ :=
seniors * average_senior_score

noncomputable def total_junior_score (total_score senior_score : ℕ) : ℕ :=
total_score - senior_score

noncomputable def junior_score (junior_total_score juniors : ℕ) : ℕ :=
junior_total_score / juniors

theorem junior_score_proof :
  ∀ (total_students: ℕ) (percent_juniors average_class_score average_senior_score : ℕ),
  total_students = 20 →
  percent_juniors = 15 →
  average_class_score = 85 →
  average_senior_score = 84 →
  (junior_score (total_junior_score (class_total_score total_students average_class_score)
                                    (total_senior_score (number_of_seniors total_students (number_of_juniors total_students percent_juniors))
                                                        average_senior_score))
                (number_of_juniors total_students percent_juniors)) = 91 :=
by
  intros
  sorry

end junior_score_proof_l34_34380


namespace unique_y_for_diamond_l34_34338

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond :
  ∃! y : ℝ, diamond 4 y = 21 :=
by
  sorry

end unique_y_for_diamond_l34_34338


namespace min_cost_speed_l34_34599

noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

theorem min_cost_speed 
  (v : ℝ) 
  (u : ℝ) 
  (other_costs : ℝ) 
  (h1 : u = (1/200) * v^3) 
  (h2 : u = 40) 
  (h3 : v = 20) 
  (h4 : other_costs = 270) 
  (b : ℝ) 
  : ∃ v_min, v_min = 30 ∧ 
    ∀ (v : ℝ), (0 < v ∧ v ≤ b) → 
    ((fuel_cost v / v + other_costs / v) ≥ (fuel_cost v_min / v_min + other_costs / v_min)) := 
sorry

end min_cost_speed_l34_34599


namespace problem_1_problem_2_l34_34113

-- Define the sets A, B, C
def SetA (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def SetB : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def SetC : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

-- Problem 1
theorem problem_1 (a : ℝ) : SetA a = SetB → a = 5 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (SetA a ∩ SetB).Nonempty ∧ (SetA a ∩ SetC = ∅) → a = -2 := by
  sorry

end problem_1_problem_2_l34_34113


namespace not_prime_1001_base_l34_34497

theorem not_prime_1001_base (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (n^3 + 1) :=
sorry

end not_prime_1001_base_l34_34497


namespace book_cost_price_l34_34620

theorem book_cost_price
  (C : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : SP = 1.25 * C)
  (h2 : 0.95 * P = SP)
  (h3 : SP = 62.5) : 
  C = 50 := 
by
  sorry

end book_cost_price_l34_34620


namespace sum_of_reciprocals_is_two_thirds_l34_34872

theorem sum_of_reciprocals_is_two_thirds (x y : ℕ) (hx_diff : x - y = 4) (hy_val : y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  (1 / (↑x : ℚ) + 1 / (↑y : ℚ) = 2 / 3) :=
by
  sorry

end sum_of_reciprocals_is_two_thirds_l34_34872


namespace ratio_of_b_to_sum_a_c_l34_34864

theorem ratio_of_b_to_sum_a_c (a b c : ℕ) (h1 : a + b + c = 60) (h2 : a = 1/3 * (b + c)) (h3 : c = 35) : b = 1/5 * (a + c) :=
by
  sorry

end ratio_of_b_to_sum_a_c_l34_34864


namespace subset_A_B_l34_34075

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2} -- Definition of set A
def B (a : ℝ) := {x : ℝ | x > a} -- Definition of set B

theorem subset_A_B (a : ℝ) : a < 1 → A ⊆ B a :=
by
  sorry

end subset_A_B_l34_34075


namespace parabola_equation_l34_34660

variables (a b c p : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : p > 0)
variables (h_eccentricity : c / a = 2)
variables (h_b : b = Real.sqrt (3) * a)
variables (h_c : c = Real.sqrt (a^2 + b^2))
variables (d : ℝ) (h_distance : d = 2) (h_d_formula : d = (a * p) / (2 * c))

theorem parabola_equation (h : (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ (c / a = 2) ∧ (b = (Real.sqrt 3) * a) ∧ (c = Real.sqrt (a^2 + b^2)) ∧ (d = 2) ∧ (d = (a * p) / (2 * c))) : x^2 = 16 * y :=
by {
  -- Lean does not require an actual proof here, so we use sorry.
  sorry
}

end parabola_equation_l34_34660


namespace worth_of_presents_is_33536_36_l34_34681

noncomputable def total_worth_of_presents : ℝ :=
  let ring := 4000
  let car := 2000
  let bracelet := 2 * ring
  let gown := bracelet / 2
  let jewelry := 1.2 * ring
  let painting := 3000 * 1.2
  let honeymoon := 180000 / 110
  let watch := 5500
  ring + car + bracelet + gown + jewelry + painting + honeymoon + watch

theorem worth_of_presents_is_33536_36 : total_worth_of_presents = 33536.36 := by
  sorry

end worth_of_presents_is_33536_36_l34_34681


namespace probability_of_missing_coupons_l34_34579

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l34_34579


namespace side_length_of_square_l34_34429

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l34_34429


namespace circle_equation_proof_l34_34478

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define a predicate for the circle being tangent to the y-axis
def tangent_y_axis (center : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r = abs center.1

-- Define the equation of the circle given center and radius
def circle_eqn (center : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2

-- State the theorem
theorem circle_equation_proof :
  tangent_y_axis circle_center →
  ∃ r, r = 1 ∧ circle_eqn circle_center r :=
sorry

end circle_equation_proof_l34_34478


namespace tan_315_eq_neg1_l34_34968

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34968


namespace inequality_holds_for_all_x_l34_34366

variable (a x : ℝ)

theorem inequality_holds_for_all_x (h : a ∈ Set.Ioc (-2 : ℝ) 4): ∀ x : ℝ, (x^2 - a*x + 9 > 0) :=
sorry

end inequality_holds_for_all_x_l34_34366


namespace factorize_expression_l34_34203

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l34_34203


namespace least_faces_combined_l34_34741

noncomputable def least_number_of_faces (c d : ℕ) : ℕ :=
c + d

theorem least_faces_combined (c d : ℕ) (h_cge8 : c ≥ 8) (h_dge8 : d ≥ 8)
  (h_sum9_prob : 8 / (c * d) = 1 / 2 * 16 / (c * d))
  (h_sum15_prob : ∃ m : ℕ, m / (c * d) = 1 / 15) :
  least_number_of_faces c d = 28 := sorry

end least_faces_combined_l34_34741


namespace negation_of_existence_l34_34860

theorem negation_of_existence (h: ¬ ∃ x : ℝ, x^2 + 1 < 0) : ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l34_34860


namespace total_boys_in_groups_l34_34578

-- Definitions of number of groups
def total_groups : ℕ := 35
def groups_with_1_boy : ℕ := 10
def groups_with_at_least_2_boys : ℕ := 19
def groups_with_3_boys_twice_groups_with_3_girls (groups_with_3_boys groups_with_3_girls : ℕ) : Prop :=
  groups_with_3_boys = 2 * groups_with_3_girls

theorem total_boys_in_groups :
  ∃ (groups_with_3_girls groups_with_3_boys groups_with_1_girl_2_boys : ℕ),
    groups_with_1_boy + groups_with_at_least_2_boys + groups_with_3_girls = total_groups
    ∧ groups_with_3_boys_twice_groups_with_3_girls groups_with_3_boys groups_with_3_girls
    ∧ groups_with_1_girl_2_boys + groups_with_3_boys = groups_with_at_least_2_boys
    ∧ (groups_with_1_boy * 1 + groups_with_1_girl_2_boys * 2 + groups_with_3_boys * 3) = 60 :=
sorry

end total_boys_in_groups_l34_34578


namespace simplify_expression_l34_34047

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l34_34047


namespace wire_cut_ratio_l34_34183

-- Define lengths a and b
variable (a b : ℝ)

-- Define perimeter equal condition
axiom perimeter_eq : 4 * (a / 4) = 6 * (b / 6)

-- The statement to prove
theorem wire_cut_ratio (h : 4 * (a / 4) = 6 * (b / 6)) : a / b = 1 :=
by
  sorry

end wire_cut_ratio_l34_34183


namespace tan_315_proof_l34_34960

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34960


namespace difference_brothers_l34_34486

def aaron_brothers : ℕ := 4
def bennett_brothers : ℕ := 6

theorem difference_brothers : 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end difference_brothers_l34_34486


namespace exists_numbering_for_nonagon_no_numbering_for_decagon_l34_34470

-- Definitions for the problem setup
variable (n : ℕ) 
variable (A : Fin n → Point)
variable (O : Point)

-- Definition for the numbering function
variable (f : Fin (2 * n) → ℕ)

-- First statement for n = 9
theorem exists_numbering_for_nonagon :
  ∃ (f : Fin 18 → ℕ), (∀ i : Fin 9, f (i : Fin 9) + f (i + 9) + f ((i + 1) % 9) = 15) :=
sorry

-- Second statement for n = 10
theorem no_numbering_for_decagon :
  ¬ ∃ (f : Fin 20 → ℕ), (∀ i : Fin 10, f (i : Fin 10) + f (i + 10) + f ((i + 1) % 10) = 16) :=
sorry

end exists_numbering_for_nonagon_no_numbering_for_decagon_l34_34470


namespace pentagon_right_angles_l34_34839

theorem pentagon_right_angles (angles : Finset ℕ) :
  angles = {0, 1, 2, 3} ↔ ∀ (k : ℕ), k ∈ angles ↔ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 540 ∧ (a = 90 ∨ b = 90 ∨ c = 90 ∨ d = 90 ∨ e = 90) 
  ∧ Finset.card (Finset.filter (λ x => x = 90) {a, b, c, d, e}) = k := 
sorry

end pentagon_right_angles_l34_34839


namespace find_pairs_l34_34790

open BigOperators

theorem find_pairs (a b : ℕ) (h1 : a < b) (h2 : ∑ k in Finset.range (b-1) - Finset.range (a+1), k = 1998) :
  (a, b) = (1997, 1999) ∨ 
  (a, b) = (664, 668) ∨ 
  (a, b) = (497, 502) ∨ 
  (a, b) = (217, 227) ∨ 
  (a, b) = (160, 173) ∨ 
  (a, b) = (60, 88) ∨ 
  (a, b) = (37, 74) ∨ 
  (a, b) = (35, 73) :=
sorry

end find_pairs_l34_34790


namespace real_solutions_eq_l34_34792

def satisfies_equations (x y : ℝ) : Prop :=
  (4 * x + 5 * y = 13) ∧ (2 * x - 3 * y = 1)

theorem real_solutions_eq {x y : ℝ} : satisfies_equations x y ↔ (x = 2 ∧ y = 1) :=
by sorry

end real_solutions_eq_l34_34792


namespace books_bought_l34_34635

theorem books_bought (cost_crayons cost_calculators total_money cost_per_bag bags_bought cost_per_book remaining_money books_bought : ℕ) 
  (h1: cost_crayons = 5 * 5)
  (h2: cost_calculators = 3 * 5)
  (h3: total_money = 200)
  (h4: cost_per_bag = 10)
  (h5: bags_bought = 11)
  (h6: remaining_money = total_money - (cost_crayons + cost_calculators) - (bags_bought * cost_per_bag)) :
  books_bought = remaining_money / cost_per_book → books_bought = 10 :=
by
  sorry

end books_bought_l34_34635


namespace number_of_correct_conclusions_l34_34487

theorem number_of_correct_conclusions : 
    (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
    (∀ x : ℝ, (x ≠ 0 → x - Real.sin x ≠ 0)) ∧
    (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
    (¬ (∀ x : ℝ, x - Real.log x > 0))
    → 3 = 3 :=
by
  sorry

end number_of_correct_conclusions_l34_34487


namespace tan_315_eq_neg1_l34_34931

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34931


namespace find_divisor_l34_34382

theorem find_divisor (Q R D V : ℤ) (hQ : Q = 65) (hR : R = 5) (hV : V = 1565) (hEquation : V = D * Q + R) : D = 24 :=
by
  sorry

end find_divisor_l34_34382


namespace triangle_inequality_l34_34017

-- Define the lengths of the existing sticks
def a := 4
def b := 7

-- Define the list of potential third sticks
def potential_sticks := [3, 6, 11, 12]

-- Define the triangle inequality conditions
def valid_length (c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Prove that the valid length satisfying these conditions is 6
theorem triangle_inequality : ∃ c ∈ potential_sticks, valid_length c ∧ c = 6 :=
by
  sorry

end triangle_inequality_l34_34017


namespace order_of_a_b_c_l34_34656

noncomputable def a : ℝ := (Real.log 5) / 5
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 4) / 4

theorem order_of_a_b_c : a < c ∧ c < b := by
  sorry

end order_of_a_b_c_l34_34656


namespace candies_per_friend_l34_34559

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l34_34559


namespace minimum_value_of_expression_l34_34234

theorem minimum_value_of_expression {x : ℝ} (hx : x > 0) : (2 / x + x / 2) ≥ 2 :=
by sorry

end minimum_value_of_expression_l34_34234


namespace side_length_of_square_l34_34437

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l34_34437


namespace find_point_D_l34_34246

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end find_point_D_l34_34246


namespace deposit_correct_l34_34411

-- Define the conditions
def monthly_income : ℝ := 10000
def deposit_percentage : ℝ := 0.25

-- Define the deposit calculation based on the conditions
def deposit_amount (income : ℝ) (percentage : ℝ) : ℝ :=
  percentage * income

-- Theorem: Prove that the deposit amount is Rs. 2500
theorem deposit_correct :
    deposit_amount monthly_income deposit_percentage = 2500 :=
  sorry

end deposit_correct_l34_34411


namespace least_number_divisible_l34_34154

theorem least_number_divisible (n : ℕ) :
  (∃ n, (n + 3) % 24 = 0 ∧ (n + 3) % 32 = 0 ∧ (n + 3) % 36 = 0 ∧ (n + 3) % 54 = 0) →
  n = 861 :=
by
  sorry

end least_number_divisible_l34_34154


namespace chocolate_chip_cookie_count_l34_34405

-- Let cookies_per_bag be the number of cookies in each bag
def cookies_per_bag : ℕ := 5

-- Let oatmeal_cookies be the number of oatmeal cookies
def oatmeal_cookies : ℕ := 2

-- Let num_baggies be the number of baggies
def num_baggies : ℕ := 7

-- Define the total number of cookies as num_baggies * cookies_per_bag
def total_cookies : ℕ := num_baggies * cookies_per_bag

-- Define the number of chocolate chip cookies as total_cookies - oatmeal_cookies
def chocolate_chip_cookies : ℕ := total_cookies - oatmeal_cookies

-- Prove that the number of chocolate chip cookies is 33
theorem chocolate_chip_cookie_count : chocolate_chip_cookies = 33 := by
  sorry

end chocolate_chip_cookie_count_l34_34405


namespace weight_of_5_moles_H₂CO₃_l34_34155

-- Definitions based on the given conditions
def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_H₂CO₃_H : ℕ := 2
def num_H₂CO₃_C : ℕ := 1
def num_H₂CO₃_O : ℕ := 3

def molecular_weight (num_H num_C num_O : ℕ) 
                     (weight_H weight_C weight_O : ℝ) : ℝ :=
  num_H * weight_H + num_C * weight_C + num_O * weight_O

-- Main proof statement
theorem weight_of_5_moles_H₂CO₃ :
  5 * molecular_weight num_H₂CO₃_H num_H₂CO₃_C num_H₂CO₃_O 
                       atomic_weight_H atomic_weight_C atomic_weight_O 
  = 310.12 := by
  sorry

end weight_of_5_moles_H₂CO₃_l34_34155


namespace alex_initial_silk_l34_34184

theorem alex_initial_silk (m_per_dress : ℕ) (m_per_friend : ℕ) (num_friends : ℕ) (num_dresses : ℕ) (initial_silk : ℕ) :
  m_per_dress = 5 ∧ m_per_friend = 20 ∧ num_friends = 5 ∧ num_dresses = 100 ∧ 
  (initial_silk - (num_friends * m_per_friend)) / m_per_dress * m_per_dress = num_dresses * m_per_dress → 
  initial_silk = 600 :=
by
  intros
  sorry

end alex_initial_silk_l34_34184


namespace rational_comparison_correct_l34_34907

-- Definitions based on conditions 
def positive_gt_zero (a : ℚ) : Prop := 0 < a
def negative_lt_zero (a : ℚ) : Prop := a < 0
def positive_gt_negative (a b : ℚ) : Prop := positive_gt_zero a ∧ negative_lt_zero b ∧ a > b
def negative_comparison (a b : ℚ) : Prop := negative_lt_zero a ∧ negative_lt_zero b ∧ abs a > abs b ∧ a < b

-- Theorem to prove
theorem rational_comparison_correct :
  (0 < - (1 / 2)) = false ∧
  ((4 / 5) < - (6 / 7)) = false ∧
  ((9 / 8) > (8 / 9)) = true ∧
  (-4 > -3) = false :=
by
  -- Mark the proof as unfinished.
  sorry

end rational_comparison_correct_l34_34907


namespace fraction_meaningful_l34_34144

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_l34_34144


namespace D_double_prime_coordinates_l34_34117

-- The coordinates of points A, B, C, D as given in the problem
def A : (ℝ × ℝ) := (3, 6)
def B : (ℝ × ℝ) := (5, 10)
def C : (ℝ × ℝ) := (7, 6)
def D : (ℝ × ℝ) := (5, 2)

-- Reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def D' : ℝ × ℝ := reflect_x D

-- Translate the point (x, y) by (dx, dy)
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

-- Reflect across the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combined translation and reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := translate p 0 (-2)
  let p_reflected := reflect_y_eq_x p_translated
  translate p_reflected 0 2

def D'' : ℝ × ℝ := reflect_y_eq_x_plus_2 D'

theorem D_double_prime_coordinates : D'' = (-4, 7) := by
  sorry

end D_double_prime_coordinates_l34_34117


namespace hcf_of_abc_l34_34375

-- Given conditions
variables (a b c : ℕ)
def lcm_abc := Nat.lcm (Nat.lcm a b) c
def product_abc := a * b * c

-- Statement to prove
theorem hcf_of_abc (H1 : lcm_abc a b c = 1200) (H2 : product_abc a b c = 108000) : 
  Nat.gcd (Nat.gcd a b) c = 90 :=
by
  sorry

end hcf_of_abc_l34_34375


namespace inversely_proportional_x_y_l34_34566

theorem inversely_proportional_x_y {x y k : ℝ}
    (h_inv_proportional : x * y = k)
    (h_k : k = 75)
    (h_y : y = 45) :
    x = 5 / 3 :=
by
  sorry

end inversely_proportional_x_y_l34_34566


namespace opposite_neg_two_l34_34710

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l34_34710


namespace cos_b_eq_one_div_sqrt_two_l34_34361

variable {a b c : ℝ} -- Side lengths
variable {A B C : ℝ} -- Angles in radians

-- Conditions of the problem
variables (h1 : c = 2 * a) 
          (h2 : b^2 = a * c) 
          (h3 : a^2 + b^2 = c^2 - 2 * a * b * Real.cos C)
          (h4 : A + B + C = Real.pi)

theorem cos_b_eq_one_div_sqrt_two
    (h1 : c = 2 * a)
    (h2 : b = a * Real.sqrt 2)
    (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
    (h4 : A + B + C = Real.pi )
    : Real.cos B = 1 / Real.sqrt 2 := 
sorry

end cos_b_eq_one_div_sqrt_two_l34_34361


namespace parallel_MN_PQ_l34_34397

open_locale big_operators

variables (A B C M N P Q : Type) [geometry_type A B C M N P Q]

-- Conditions
def M_is_midpoint_arc_AB (M A B C : Type) : Prop :=
  midpoint_arc M A B (circle (M A B C)) ∧ ¬contain_point (M A B C) C

def N_is_midpoint_arc_BC (N B C A : Type) : Prop :=
  midpoint_arc N B C (circle (N B C A)) ∧ ¬contain_point (N B C A) A

-- Statement to prove
theorem parallel_MN_PQ :
  M_is_midpoint_arc_AB M A B C →
  N_is_midpoint_arc_BC N B C A →
  MN_parallel_PQ M N P Q :=
sorry

end parallel_MN_PQ_l34_34397


namespace ratio_of_sister_to_Aaron_l34_34039

noncomputable def Aaron_age := 15
variable (H S : ℕ)
axiom Henry_age_relation : H = 4 * S
axiom combined_age : H + S + Aaron_age = 240

theorem ratio_of_sister_to_Aaron : (S : ℚ) / Aaron_age = 3 := 
by
  -- Proof omitted
  sorry

end ratio_of_sister_to_Aaron_l34_34039


namespace number_of_always_true_inequalities_l34_34505

theorem number_of_always_true_inequalities (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  (a + c > b + d) ∧
  (¬(a - c > b - d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 - 3 > -2 - (-2))) ∧
  (¬(a * c > b * d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 * 3 > -2 * (-2))) ∧
  (¬(a / c > b / d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 / 3 > (-2) / (-2))) :=
by
  sorry

end number_of_always_true_inequalities_l34_34505


namespace product_bc_l34_34453

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l34_34453


namespace tan_315_degree_l34_34975

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34975


namespace range_of_x_l34_34218

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) : 
  12 ≤ x := 
sorry

end range_of_x_l34_34218


namespace simplify_expression_l34_34046

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l34_34046


namespace small_n_for_sum_l34_34448

def a : ℕ → ℝ
| 1     := 1.5
| (n+2) := 1 / (n + 2)^2 - 1

noncomputable def S (n : ℕ) : ℝ :=
1.5 + (∑ k in finset.range (n - 1 + 1), 1 / (k + 1)^2 - 1)

theorem small_n_for_sum :
  ∃ n : ℕ, |S n - 2.25| < 0.01 ∧ n = 100 :=
begin
  sorry
end

end small_n_for_sum_l34_34448


namespace number_of_initials_sets_l34_34814

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l34_34814


namespace erika_flips_coin_probability_l34_34054

theorem erika_flips_coin_probability:
  let coin := {12, 24}
  let die := {1, 2, 3, 4, 5, 6}
  let p_coin_12 := (1 : ℚ) / 2
  let p_die_4 := (1 : ℚ) / 6
  let p_coin_24 := (1 : ℚ) / 2
  let p_die_2 := (1 : ℚ) / 6
  let p := p_coin_12 * p_die_4 + p_coin_24 * p_die_2
  in p = 1 / 6 := 
sorry

end erika_flips_coin_probability_l34_34054


namespace impossible_all_matches_outside_own_country_l34_34131

theorem impossible_all_matches_outside_own_country (n : ℕ) (h_teams : n = 16) : 
  ¬ ∀ (T : Fin n → Fin n → Prop), (∀ i j, i ≠ j → T i j) ∧ 
  (∀ i, ∀ j, i ≠ j → T i j → T j i) ∧ 
  (∀ i, T i i = false) → 
  ∀ i, ∃ j, T i j ∧ i ≠ j :=
by
  intro H
  sorry

end impossible_all_matches_outside_own_country_l34_34131


namespace smallest_n_for_2n_3n_5n_conditions_l34_34596

theorem smallest_n_for_2n_3n_5n_conditions : 
  ∃ n : ℕ, 
    (∀ k : ℕ, 2 * n ≠ k^2) ∧          -- 2n is a perfect square
    (∀ k : ℕ, 3 * n ≠ k^3) ∧          -- 3n is a perfect cube
    (∀ k : ℕ, 5 * n ≠ k^5) ∧          -- 5n is a perfect fifth power
    n = 11250 :=
sorry

end smallest_n_for_2n_3n_5n_conditions_l34_34596


namespace simplify_expression_l34_34413

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end simplify_expression_l34_34413


namespace max_n_factorable_l34_34649

theorem max_n_factorable :
  ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 24 → 3 * B + A = n) ∧ (n = 73) :=
sorry

end max_n_factorable_l34_34649


namespace intersection_points_of_parabolas_l34_34840

open Real

theorem intersection_points_of_parabolas (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ y1 y2 : ℝ, y1 = c ∧ y2 = (-2 * b^2 / (9 * a)) + c ∧ 
    ((y1 = a * (0)^2 + b * (0) + c) ∧ (y2 = a * (-b / (3 * a))^2 + b * (-b / (3 * a)) + c))) :=
by
  sorry

end intersection_points_of_parabolas_l34_34840


namespace tangent_315_deg_l34_34978

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34978


namespace houses_in_block_l34_34625

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) (h1 : junk_mail_per_house = 2) (h2 : total_junk_mail = 14) :
  total_junk_mail / junk_mail_per_house = 7 := by
  sorry

end houses_in_block_l34_34625


namespace range_of_a_l34_34219

variable (x a : ℝ)

-- Definitions of conditions as hypotheses
def condition_p (x : ℝ) := |x + 1| ≤ 2
def condition_q (x a : ℝ) := x ≤ a
def sufficient_not_necessary (p q : Prop) := p → q ∧ ¬(q → p)

-- The theorem statement
theorem range_of_a : sufficient_not_necessary (condition_p x) (condition_q x a) → 1 ≤ a ∧ ∀ b, b < 1 → sufficient_not_necessary (condition_p x) (condition_q x b) → false :=
by
  intro h
  sorry

end range_of_a_l34_34219


namespace shaded_area_ratio_l34_34766

theorem shaded_area_ratio
  (large_square_area : ℕ := 25)
  (grid_dimension : ℕ := 5)
  (shaded_square_area : ℕ := 2)
  (num_squares : ℕ := 25)
  (ratio : ℚ := 2 / 25) :
  (shaded_square_area : ℚ) / large_square_area = ratio := 
by
  sorry

end shaded_area_ratio_l34_34766


namespace div_expression_calc_l34_34332

theorem div_expression_calc :
  (3752 / (39 * 2) + 5030 / (39 * 10) = 61) :=
by
  sorry -- Proof of the theorem

end div_expression_calc_l34_34332


namespace all_propositions_imply_l34_34201

variables (p q r : Prop)

theorem all_propositions_imply (hpqr : p ∧ q ∧ r)
                               (hnpqr : ¬p ∧ q ∧ ¬r)
                               (hpnqr : p ∧ ¬q ∧ r)
                               (hnpnqr : ¬p ∧ ¬q ∧ ¬r) :
  (p → q) ∨ r :=
by { sorry }

end all_propositions_imply_l34_34201


namespace intersection_complement_l34_34803

open Set

noncomputable def U : Set ℝ := {-1, 0, 1, 4}
def A : Set ℝ := {-1, 1}
def B : Set ℝ := {1, 4}
def C_U_B : Set ℝ := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} :=
by
  sorry

end intersection_complement_l34_34803


namespace sum_of_roots_eq_l34_34158

theorem sum_of_roots_eq (k : ℝ) : ∃ x1 x2 : ℝ, (2 * x1 ^ 2 - 3 * x1 + k = 7) ∧ (2 * x2 ^ 2 - 3 * x2 + k = 7) ∧ (x1 + x2 = 3 / 2) :=
by sorry

end sum_of_roots_eq_l34_34158


namespace first_position_remainder_one_l34_34858

theorem first_position_remainder_one (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2023)
(h2 : ∀ b c d : ℕ, b = a ∧ c = a + 2 ∧ d = a + 4 → 
  b % 3 ≠ c % 3 ∧ c % 3 ≠ d % 3 ∧ d % 3 ≠ b % 3):
  a % 3 = 1 :=
sorry

end first_position_remainder_one_l34_34858


namespace vanya_exam_scores_l34_34299

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores_l34_34299


namespace cost_of_five_dozens_l34_34489

-- Define cost per dozen given the total cost for two dozen
noncomputable def cost_per_dozen : ℝ := 15.60 / 2

-- Define the number of dozen apples we want to calculate the cost for
def number_of_dozens := 5

-- Define the total cost for the given number of dozens
noncomputable def total_cost (n : ℕ) : ℝ := n * cost_per_dozen

-- State the theorem
theorem cost_of_five_dozens : total_cost number_of_dozens = 39 :=
by
  unfold total_cost cost_per_dozen
  sorry

end cost_of_five_dozens_l34_34489


namespace sujis_age_l34_34735

theorem sujis_age (x : ℕ) (Abi Suji : ℕ)
  (h1 : Abi = 5 * x)
  (h2 : Suji = 4 * x)
  (h3 : (Abi + 3) / (Suji + 3) = 11 / 9) : 
  Suji = 24 := 
by 
  sorry

end sujis_age_l34_34735


namespace recycling_drive_l34_34028

theorem recycling_drive (S : ℕ) 
  (h1 : ∀ (n : ℕ), n = 280 * S) -- Each section collected 280 kilos in two weeks
  (h2 : ∀ (t : ℕ), t = 2000 - 320) -- After the third week, they needed 320 kilos more to reach their target of 2000 kilos
  : S = 3 :=
by
  sorry

end recycling_drive_l34_34028


namespace josh_initial_wallet_l34_34390

noncomputable def initial_wallet_amount (investment final_wallet: ℕ) (stock_increase_percentage: ℕ): ℕ :=
  let investment_value_after_rise := investment + (investment * stock_increase_percentage / 100)
  final_wallet - investment_value_after_rise

theorem josh_initial_wallet : initial_wallet_amount 2000 2900 30 = 300 :=
by
  sorry

end josh_initial_wallet_l34_34390


namespace triangle_area_gt_half_l34_34588

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l34_34588


namespace discount_price_l34_34465

theorem discount_price (P : ℝ) (h : P > 0) (discount : ℝ) (h_discount : discount = 0.80) : 
  (P - P * discount) = P * 0.20 :=
by
  sorry

end discount_price_l34_34465


namespace factorize_expression_l34_34205

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l34_34205


namespace number_of_initials_sets_l34_34815

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l34_34815


namespace simplify_expression_l34_34129

theorem simplify_expression (x : ℝ) : (2 * x)^5 + (3 * x) * x^4 + 2 * x^3 = 35 * x^5 + 2 * x^3 :=
by
  sorry

end simplify_expression_l34_34129


namespace smallest_y_of_arithmetic_sequence_l34_34262

theorem smallest_y_of_arithmetic_sequence
  (x y z d : ℝ)
  (h_arith_series_x : x = y - d)
  (h_arith_series_z : z = y + d)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_product : x * y * z = 216) : y = 6 :=
sorry

end smallest_y_of_arithmetic_sequence_l34_34262


namespace simplify_complex_expr_correct_l34_34412

noncomputable def simplify_complex_expr (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (4 - 2 * i) - 2 * i * (3 - 2 * i) + (1 + i) * (2 + i)

theorem simplify_complex_expr_correct (i : ℂ) (h : i^2 = -1) : 
  simplify_complex_expr i h = 9 - 9 * i :=
by
  sorry

end simplify_complex_expr_correct_l34_34412


namespace solution_set_of_abs_fraction_eq_fraction_l34_34145

-- Problem Statement
theorem solution_set_of_abs_fraction_eq_fraction :
  { x : ℝ | |x / (x - 1)| = x / (x - 1) } = { x : ℝ | x ≤ 0 ∨ x > 1 } :=
by
  sorry

end solution_set_of_abs_fraction_eq_fraction_l34_34145


namespace number_of_unlocked_cells_l34_34318

-- Establish the conditions from the problem description.
def total_cells : ℕ := 2004

-- Helper function to determine if a number is a perfect square.
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- Counting the number of perfect squares in the range from 1 to total_cells.
def perfect_squares_up_to (n : ℕ) : ℕ :=
  (Nat.sqrt n)

-- The theorem that needs to be proved.
theorem number_of_unlocked_cells : perfect_squares_up_to total_cells = 44 :=
by
  sorry

end number_of_unlocked_cells_l34_34318


namespace no_solution_m_l34_34096

noncomputable def fractional_eq (x m : ℝ) : Prop :=
  2 / (x - 2) + m * x / (x^2 - 4) = 3 / (x + 2)

theorem no_solution_m (m : ℝ) : 
  (¬ ∃ x, fractional_eq x m) ↔ (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end no_solution_m_l34_34096


namespace work_rate_c_l34_34750

variables (rate_a rate_b rate_c : ℚ)

-- Given conditions
axiom h1 : rate_a + rate_b = 1 / 15
axiom h2 : rate_a + rate_b + rate_c = 1 / 6

theorem work_rate_c : rate_c = 1 / 10 :=
by sorry

end work_rate_c_l34_34750


namespace spirangle_length_l34_34495

theorem spirangle_length :
  let a := 2
  let d := 2
  let l := 200
  let n := (l - a) / d + 1
  let Sn := n * (a + l) / 2
  let final_segment := Sn + 201
  final_segment = 10301 :=
by
  let a := 2
  let d := 2
  let l := 200
  let n := (l - a) / d + 1
  let Sn := n * (a + l) / 2
  let final_segment := Sn + 201
  exact eq.refl final_segment

end spirangle_length_l34_34495


namespace initial_amount_l34_34163

theorem initial_amount (X : ℝ) (h1 : 0.70 * X = 2800) : X = 4000 :=
by
  sorry

end initial_amount_l34_34163


namespace cistern_fill_time_l34_34305

theorem cistern_fill_time
  (T : ℝ)
  (H1 : 0 < T)
  (rate_first_tap : ℝ := 1 / T)
  (rate_second_tap : ℝ := 1 / 6)
  (net_rate : ℝ := 1 / 12)
  (H2 : rate_first_tap - rate_second_tap = net_rate) :
  T = 4 :=
sorry

end cistern_fill_time_l34_34305


namespace calorie_limit_l34_34255

variable (breakfastCalories lunchCalories dinnerCalories extraCalories : ℕ)
variable (plannedCalories : ℕ)

-- Given conditions
axiom breakfast_calories : breakfastCalories = 400
axiom lunch_calories : lunchCalories = 900
axiom dinner_calories : dinnerCalories = 1100
axiom extra_calories : extraCalories = 600

-- To Prove
theorem calorie_limit (h : plannedCalories = (breakfastCalories + lunchCalories + dinnerCalories - extraCalories)) :
  plannedCalories = 1800 := by sorry

end calorie_limit_l34_34255


namespace tan_315_eq_neg1_l34_34932

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34932


namespace vlad_taller_than_sister_l34_34292

def height_vlad_meters : ℝ := 1.905
def height_sister_cm : ℝ := 86.36

theorem vlad_taller_than_sister :
  (height_vlad_meters * 100 - height_sister_cm = 104.14) :=
by 
  sorry

end vlad_taller_than_sister_l34_34292


namespace factorize_expression_l34_34204

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l34_34204


namespace equalSumSeqDefinition_l34_34190

def isEqualSumSeq (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n - 1) + s n = s (n + 1)

theorem equalSumSeqDefinition (s : ℕ → ℝ) :
  isEqualSumSeq s ↔ 
  ∀ n : ℕ, n > 0 → s n = s (n - 1) + s (n + 1) :=
by
  sorry

end equalSumSeqDefinition_l34_34190


namespace side_length_of_square_l34_34434

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34434


namespace evaluate_f_at_3_l34_34663

def f (x : ℤ) : ℤ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem evaluate_f_at_3 : f 3 = 181 := by
  sorry

end evaluate_f_at_3_l34_34663


namespace opposite_neg_two_l34_34713

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l34_34713


namespace graph_not_in_first_quadrant_l34_34235

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (h_not_in_first_quadrant : ∀ x : ℝ, a^x + b - 1 ≤ 0) : 
  0 < a ∧ a < 1 ∧ b ≤ 0 :=
sorry

end graph_not_in_first_quadrant_l34_34235


namespace remainder_when_3m_divided_by_5_l34_34466

theorem remainder_when_3m_divided_by_5 (m : ℤ) (hm : m % 5 = 2) : (3 * m) % 5 = 1 := 
sorry

end remainder_when_3m_divided_by_5_l34_34466


namespace Kenneth_money_left_l34_34685

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l34_34685


namespace geometric_sequence_a12_l34_34676

noncomputable def a_n (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r ^ (n - 1)

theorem geometric_sequence_a12 (a1 r : ℝ) 
  (h1 : a_n a1 r 7 * a_n a1 r 9 = 4)
  (h2 : a_n a1 r 4 = 1) :
  a_n a1 r 12 = 16 := sorry

end geometric_sequence_a12_l34_34676


namespace players_count_l34_34186

def total_socks : ℕ := 22
def socks_per_player : ℕ := 2

theorem players_count : total_socks / socks_per_player = 11 :=
by
  sorry

end players_count_l34_34186


namespace hamburgers_leftover_l34_34319

-- Define the number of hamburgers made and served
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove the number of leftover hamburgers
theorem hamburgers_leftover : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_leftover_l34_34319


namespace proof_problem_l34_34273

axiom is_line (m : Type) : Prop
axiom is_plane (α : Type) : Prop
axiom is_subset_of_plane (m : Type) (β : Type) : Prop
axiom is_perpendicular (a : Type) (b : Type) : Prop
axiom is_parallel (a : Type) (b : Type) : Prop

theorem proof_problem
  (m n : Type) 
  (α β : Type)
  (h1 : is_line m)
  (h2 : is_line n)
  (h3 : is_plane α)
  (h4 : is_plane β)
  (h_prop2 : is_parallel α β → is_subset_of_plane m α → is_parallel m β)
  (h_prop3 : is_perpendicular n α → is_perpendicular n β → is_perpendicular m α → is_perpendicular m β)
  : (is_subset_of_plane m β → is_perpendicular α β → ¬ (is_perpendicular m α)) ∧ 
    (is_parallel m α → is_parallel m β → ¬ (is_parallel α β)) :=
sorry

end proof_problem_l34_34273


namespace tangent_315_deg_l34_34980

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34980


namespace equal_cake_distribution_l34_34894

theorem equal_cake_distribution (total_cakes : ℕ) (total_friends : ℕ) (h_cakes : total_cakes = 150) (h_friends : total_friends = 50) :
  total_cakes / total_friends = 3 := by
  sorry

end equal_cake_distribution_l34_34894


namespace tan_315_proof_l34_34956

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34956


namespace second_pirate_gets_diamond_l34_34753

theorem second_pirate_gets_diamond (coins_bag1 coins_bag2 : ℕ) :
  (coins_bag1 ≤ 1 ∧ coins_bag2 ≤ 1) ∨ (coins_bag1 > 1 ∨ coins_bag2 > 1) →
  (∃ n k : ℕ, n % 2 = 0 → (coins_bag1 + n) = (coins_bag2 + k)) :=
sorry

end second_pirate_gets_diamond_l34_34753


namespace side_length_of_square_l34_34422

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l34_34422


namespace opposite_of_neg_three_l34_34734

theorem opposite_of_neg_three : -(-3) = 3 :=
by 
  sorry

end opposite_of_neg_three_l34_34734


namespace not_exists_cube_in_sequence_l34_34532

-- Lean statement of the proof problem
theorem not_exists_cube_in_sequence : ∀ n : ℕ, ¬ ∃ k : ℤ, 2 ^ (2 ^ n) + 1 = k ^ 3 := 
by 
    intro n
    intro ⟨k, h⟩
    sorry

end not_exists_cube_in_sequence_l34_34532


namespace savings_value_l34_34275

def total_cost_individual (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let cost (n : ℕ) : ℝ := 
    let paid_windows := n - (n / 6) -- one free window per five
    cost_per_window * paid_windows
  let discount (amount : ℝ) : ℝ :=
    if s > 10 then 0.95 * amount else amount
  discount (cost g) + discount (cost s)

def total_cost_joint (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let n := g + s
  let paid_windows := n - (n / 6) -- one free window per five
  let joint_cost := cost_per_window * paid_windows
  if n > 10 then 0.95 * joint_cost else joint_cost

def savings (g : ℕ) (s : ℕ) : ℝ :=
  total_cost_individual g s - total_cost_joint g s

theorem savings_value (g s : ℕ) (hg : g = 9) (hs : s = 13) : savings g s = 162 := 
by 
  simp [savings, total_cost_individual, total_cost_joint, hg, hs]
  -- Detailed calculation is omitted, since it's not required according to the instructions.
  sorry

end savings_value_l34_34275


namespace problem_I_problem_II_problem_III_l34_34777

variables {pA pB : ℝ}

-- Given conditions
def probability_A : ℝ := 0.7
def probability_B : ℝ := 0.6

-- Questions reformulated as proof goals
theorem problem_I : 
  sorry := 
 sorry

theorem problem_II : 
  -- Find: Probability that at least one of A or B succeeds on the first attempt
  sorry := 
 sorry

theorem problem_III : 
  -- Find: Probability that A succeeds exactly one more time than B in two attempts each
  sorry := 
 sorry

end problem_I_problem_II_problem_III_l34_34777


namespace power_mod_eq_nine_l34_34706

theorem power_mod_eq_nine :
  ∃ n : ℕ, 13^6 ≡ n [MOD 11] ∧ 0 ≤ n ∧ n < 11 ∧ n = 9 :=
by
  sorry

end power_mod_eq_nine_l34_34706


namespace projection_of_b_onto_a_l34_34364

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (4, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := (dot_product u v) / (magnitude u)

theorem projection_of_b_onto_a : projection b a = 2 * sqrt 5 / 5 := by
  sorry

end projection_of_b_onto_a_l34_34364


namespace am_gm_inequality_l34_34111

theorem am_gm_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_inequality_l34_34111


namespace max_pieces_with_3_cuts_l34_34461

theorem max_pieces_with_3_cuts (cake : Type) : 
  (∀ (cuts : ℕ), cuts = 3 → (∃ (max_pieces : ℕ), max_pieces = 8)) := by
  sorry

end max_pieces_with_3_cuts_l34_34461


namespace anita_apples_l34_34191

theorem anita_apples (num_students : ℕ) (apples_per_student : ℕ) (total_apples : ℕ) 
  (h1 : num_students = 60) 
  (h2 : apples_per_student = 6) 
  (h3 : total_apples = num_students * apples_per_student) : 
  total_apples = 360 := 
by
  sorry

end anita_apples_l34_34191


namespace mean_value_theorem_for_integrals_l34_34404

variable {a b : ℝ} (f : ℝ → ℝ)

theorem mean_value_theorem_for_integrals (h_cont : ContinuousOn f (Set.Icc a b)) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) :=
sorry

end mean_value_theorem_for_integrals_l34_34404


namespace tan_315_eq_neg1_l34_34988

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34988


namespace expression_simplification_l34_34780

theorem expression_simplification (x y : ℝ) : x^2 + (y - x) * (y + x) = y^2 :=
by
  sorry

end expression_simplification_l34_34780


namespace sum_gcd_lcm_60_429_l34_34745

theorem sum_gcd_lcm_60_429 : 
  let a := 60
  let b := 429
  gcd a b + lcm a b = 8583 :=
by
  -- Definitions of a and b
  let a := 60
  let b := 429
  
  -- The GCD and LCM calculations would go here
  
  -- Proof body (skipped with 'sorry')
  sorry

end sum_gcd_lcm_60_429_l34_34745


namespace cody_money_l34_34472

theorem cody_money (a b c d : ℕ) (h₁ : a = 45) (h₂ : b = 9) (h₃ : c = 19) (h₄ : d = a + b - c) : d = 35 :=
by
  rw [h₁, h₂, h₃] at h₄
  simp at h₄
  exact h₄

end cody_money_l34_34472


namespace opposite_of_neg_two_is_two_l34_34723

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l34_34723


namespace opposite_of_neg_two_is_two_l34_34719

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l34_34719


namespace cupcakes_sold_l34_34212

theorem cupcakes_sold (initial_made sold additional final : ℕ) (h1 : initial_made = 42) (h2 : additional = 39) (h3 : final = 59) :
  (initial_made - sold + additional = final) -> sold = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  sorry

end cupcakes_sold_l34_34212


namespace point_M_trajectory_quadrilateral_ABCD_range_l34_34500

theorem point_M_trajectory :
  (∀ (x y : ℝ), (0.5 * (1 - x)^2 + 0.5 * y^2 = 0.5 * (4 - x)^2 + 0.5 * y^2) ↔ (x^2 + y^2 = 4)) :=
by
  sorry

theorem quadrilateral_ABCD_range :
  (∀ (d1 d2 : ℝ), (d1^2 + d2^2 = 2) →
    (∃ t : ℝ, (0 ≤ t ∧ t ≤ 2) ∧ 
      (4*sqrt((4 - t)*(2 + t)) ∈ Set.Icc (4*sqrt(2)) 6))) :=
by
  sorry

end point_M_trajectory_quadrilateral_ABCD_range_l34_34500


namespace twice_as_many_juniors_as_seniors_l34_34329

theorem twice_as_many_juniors_as_seniors (j s : ℕ) (h : (1/3 : ℝ) * j = (2/3 : ℝ) * s) : j = 2 * s :=
by
  --proof steps here
  sorry

end twice_as_many_juniors_as_seniors_l34_34329


namespace food_remaining_l34_34546

-- Definitions for conditions
def first_week_donations : ℕ := 40
def second_week_donations := 2 * first_week_donations
def total_donations := first_week_donations + second_week_donations
def percentage_given_out : ℝ := 0.70
def amount_given_out := percentage_given_out * total_donations

-- Proof goal
theorem food_remaining (h1 : first_week_donations = 40)
                      (h2 : second_week_donations = 2 * first_week_donations)
                      (h3 : percentage_given_out = 0.70) :
                      total_donations - amount_given_out = 36 := by
  sorry

end food_remaining_l34_34546


namespace smallest_rel_prime_l34_34462

theorem smallest_rel_prime (n : ℕ) (h : n > 1) (rel_prime : ∀ p ∈ [2, 3, 5, 7], ¬ p ∣ n) : n = 11 :=
by sorry

end smallest_rel_prime_l34_34462


namespace tan_315_eq_neg1_l34_34948

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34948


namespace tan_315_eq_neg_one_l34_34993

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34993


namespace side_length_of_square_l34_34436

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l34_34436


namespace unique_solution_k_values_l34_34707

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, k * x ^ 2 - 3 * x + 2 = 0) ↔ (k = 0 ∨ k = 9 / 8) :=
by
  sorry

end unique_solution_k_values_l34_34707


namespace coeff_m6n6_in_m_plus_n_pow_12_l34_34459

theorem coeff_m6n6_in_m_plus_n_pow_12 : 
  (∃ c : ℕ, (m + n)^12 = c * m^6 * n^6 + ...) → c = 924 := by
sorry

end coeff_m6n6_in_m_plus_n_pow_12_l34_34459


namespace fred_gave_balloons_to_sandy_l34_34355

-- Define the number of balloons Fred originally had
def original_balloons : ℕ := 709

-- Define the number of balloons Fred has now
def current_balloons : ℕ := 488

-- Define the number of balloons Fred gave to Sandy
def balloons_given := original_balloons - current_balloons

-- Theorem: The number of balloons given to Sandy is 221
theorem fred_gave_balloons_to_sandy : balloons_given = 221 :=
by
  sorry

end fred_gave_balloons_to_sandy_l34_34355


namespace group_B_fluctuates_less_l34_34509

-- Conditions
def mean_A : ℝ := 80
def mean_B : ℝ := 90
def variance_A : ℝ := 10
def variance_B : ℝ := 5

-- Goal
theorem group_B_fluctuates_less :
  variance_B < variance_A :=
  by
    sorry

end group_B_fluctuates_less_l34_34509


namespace function_increasing_intervals_l34_34572

theorem function_increasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y - x) < δ → f y > f x) ∨ 
  (∀ x : ℝ, ∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, abs (y - x) < δ ∧ f y < f x) :=
sorry

end function_increasing_intervals_l34_34572


namespace a_n_is_perfect_square_l34_34074

theorem a_n_is_perfect_square :
  ∀ (a b : ℕ → ℤ), a 0 = 1 → b 0 = 0 →
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∀ n, ∃ k : ℤ, a n = k * k :=
by
  sorry

end a_n_is_perfect_square_l34_34074


namespace salary_january_l34_34417

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 :=
by 
  sorry

end salary_january_l34_34417


namespace max_mondays_in_51_days_l34_34593

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l34_34593


namespace train_length_l34_34296

theorem train_length 
  (V : ℝ → ℝ) (L : ℝ) 
  (length_of_train : ∀ (t : ℝ), t = 8 → V t = L / 8) 
  (pass_platform : ∀ (d t : ℝ), d = L + 273 → t = 20 → V t = d / t) 
  : L = 182 := 
by
  sorry

end train_length_l34_34296


namespace fliers_left_l34_34164

theorem fliers_left (total : ℕ) (morning_fraction afternoon_fraction : ℚ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1/5)
  (h3 : afternoon_fraction = 1/4) :
  let morning_sent := total * morning_fraction
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  remaining_after_afternoon = 600 :=
by
  sorry

end fliers_left_l34_34164


namespace probability_alex_paired_with_jordan_l34_34381

theorem probability_alex_paired_with_jordan :
  ∀ (students : Finset ℕ) (Alex Jordan : ℕ),
    (students.card = 40) →
    (Alex ∉ students) →
    (Jordan ∉ students) →
    (Alex ≠ Jordan) →
    (∀ s ∈ students, s ≠ Alex ∧ s ≠ Jordan) →
    (∃ pairs : Finset (Finset ℕ), ∀ pair ∈ pairs, pair.card = 2 ∧ (∀ p ∈ pair, p ∈ students ∪ {Alex, Jordan})) →
    (∃! pair ∈ pairs, Alex ∈ pair ∧ Jordan ∈ pair) →
    (∀ pair ∈ pairs, Alex ∈ pair → Jordan ∈ pair) →
    (students.card.choose 1 = 1 / 39) :=
begin
  sorry
end

end probability_alex_paired_with_jordan_l34_34381


namespace range_of_a_l34_34508

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l34_34508


namespace derivative_y_wrt_x_l34_34752

noncomputable theory
open Real

-- Given conditions
def f_x (t : ℝ) : ℝ := arcsin (sin t)
def f_y (t : ℝ) : ℝ := arccos (cos t)

-- The problem statement: Prove that the derivative of y with respect to x is 1
theorem derivative_y_wrt_x (t : ℝ) (ht1 : -π/2 ≤ t ∧ t ≤ π/2) (ht2: 0 ≤ t ∧ t ≤ π) :
  ∂ (λ t, arccos (cos t)) / ∂ (λ t, arcsin (sin t)) = 1 :=
sorry

end derivative_y_wrt_x_l34_34752


namespace quadratic_range_l34_34446

theorem quadratic_range (x y : ℝ) 
    (h1 : y = (x - 1)^2 + 1)
    (h2 : 2 ≤ y ∧ y < 5) : 
    (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) :=
by
  sorry

end quadratic_range_l34_34446


namespace range_of_m_l34_34800

theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, m * x0^2 + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) → -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l34_34800


namespace distance_sum_l34_34784

theorem distance_sum (a : ℝ) (x y : ℝ) 
  (AB CD : ℝ) (A B C D P Q M N : ℝ)
  (h_AB : AB = 4) (h_CD : CD = 8) 
  (h_M_AB : M = (A + B) / 2) (h_N_CD : N = (C + D) / 2)
  (h_P_AB : P ∈ [A, B]) (h_Q_CD : Q ∈ [C, D])
  (h_x : x = dist P M) (h_y : y = dist Q N)
  (h_y_eq_2x : y = 2 * x) (h_x_eq_a : x = a) :
  x + y = 3 * a := 
by
  sorry

end distance_sum_l34_34784


namespace edward_rides_l34_34787

theorem edward_rides (total_tickets tickets_spent tickets_per_ride rides : ℕ)
    (h1 : total_tickets = 79)
    (h2 : tickets_spent = 23)
    (h3 : tickets_per_ride = 7)
    (h4 : rides = (total_tickets - tickets_spent) / tickets_per_ride) :
    rides = 8 := by sorry

end edward_rides_l34_34787


namespace find_pairs_l34_34646

def is_integer_solution (p q : ℝ) : Prop :=
  ∃ (a b : ℝ), (a * b = q ∧ a + b = p ∧ a * b ∈ ℤ ∧ a + b ∈ ℤ)

theorem find_pairs (p q : ℝ) (h : p + q = 1998) :
  ((p = 1998 ∧ q = 0) ∨ (p = -2002 ∧ q = 4000)) ∧ is_integer_solution p q :=
by sorry

end find_pairs_l34_34646


namespace tan_315_eq_neg_one_l34_34991

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34991


namespace bike_distance_from_rest_l34_34027

variable (u : ℝ) (a : ℝ) (t : ℝ)

theorem bike_distance_from_rest (h1 : u = 0) (h2 : a = 0.5) (h3 : t = 8) : 
  (1 / 2 * a * t^2 = 16) :=
by
  sorry

end bike_distance_from_rest_l34_34027


namespace number_of_real_values_of_p_l34_34644

theorem number_of_real_values_of_p :
  ∃ p_values : Finset ℝ, (∀ p ∈ p_values, ∀ x, x^2 - 2 * p * x + 3 * p = 0 → (x = p)) ∧ Finset.card p_values = 2 :=
by
  sorry

end number_of_real_values_of_p_l34_34644


namespace minimum_value_am_bn_l34_34064

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l34_34064


namespace paula_candies_distribution_l34_34556

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l34_34556


namespace find_m_l34_34667

theorem find_m (m : ℝ) :
  (∀ x y : ℝ, (3 * x + (m + 1) * y - (m - 7) = 0) → 
              (m * x + 2 * y + 3 * m = 0)) →
  (m + 1 ≠ 0) →
  m = -3 :=
by
  sorry

end find_m_l34_34667


namespace max_k_possible_l34_34906

-- Given the sequence formed by writing all three-digit numbers from 100 to 999 consecutively
def digits_sequence : List Nat := List.join (List.map (fun n => [n / 100, (n / 10) % 10, n % 10]) (List.range' 100 (999 - 100 + 1)))

-- Function to get a k-digit number from the sequence
def get_k_digit_number (seq : List Nat) (start k : Nat) : List Nat := seq.drop start |>.take k

-- Statement to prove the maximum k
theorem max_k_possible : ∃ k : Nat, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 5 = get_k_digit_number digits_sequence start2 5) ∧ (¬ ∃ k' > 5, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 k' = get_k_digit_number digits_sequence start2 k')) :=
sorry

end max_k_possible_l34_34906


namespace abs_case_inequality_solution_l34_34577

theorem abs_case_inequality_solution (x : ℝ) :
  (|x + 1| + |x - 4| ≥ 7) ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 5) :=
by
  sorry

end abs_case_inequality_solution_l34_34577


namespace max_mondays_in_first_51_days_l34_34595

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l34_34595


namespace water_loss_per_jump_l34_34107

def pool_capacity : ℕ := 2000 -- in liters
def jump_limit : ℕ := 1000
def clean_threshold : ℝ := 0.80

theorem water_loss_per_jump :
  (pool_capacity * (1 - clean_threshold)) * 1000 / jump_limit = 400 :=
by
  -- We prove that the water lost per jump in mL is 400
  sorry

end water_loss_per_jump_l34_34107


namespace peters_brother_read_percentage_l34_34841

-- Definitions based on given conditions
def total_books : ℕ := 20
def peter_read_percentage : ℕ := 40
def difference_between_peter_and_brother : ℕ := 6

-- Statement to prove
theorem peters_brother_read_percentage :
  peter_read_percentage / 100 * total_books - difference_between_peter_and_brother = 2 → 
  2 / total_books * 100 = 10 := by
  sorry

end peters_brother_read_percentage_l34_34841


namespace copper_percentage_alloy_l34_34909

theorem copper_percentage_alloy (x : ℝ) :
  (x / 100 * 45 + 0.21 * (108 - 45) = 0.1975 * 108) → x = 18 :=
by 
  sorry

end copper_percentage_alloy_l34_34909


namespace plant_height_increase_l34_34442

theorem plant_height_increase (total_increase : ℕ) (century_in_years : ℕ) (decade_in_years : ℕ) (years_in_2_centuries : ℕ) (num_decades : ℕ) : 
  total_increase = 1800 →
  century_in_years = 100 →
  decade_in_years = 10 →
  years_in_2_centuries = 2 * century_in_years →
  num_decades = years_in_2_centuries / decade_in_years →
  total_increase / num_decades = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end plant_height_increase_l34_34442


namespace tan_315_eq_neg1_l34_34967

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34967


namespace four_letter_initial_sets_l34_34812

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l34_34812


namespace present_value_l34_34020

theorem present_value (BD TD PV : ℝ) (hBD : BD = 42) (hTD : TD = 36)
  (h : BD = TD + (TD^2 / PV)) : PV = 216 :=
sorry

end present_value_l34_34020


namespace quadratic_one_solution_m_l34_34801

theorem quadratic_one_solution_m (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + m = 0) → 
  (∀ (x y : ℝ), 3 * x^2 - 7 * x + m = 0 → 3 * y^2 - 7 * y + m = 0 → x = y) → 
  m = 49 / 12 :=
by
  sorry

end quadratic_one_solution_m_l34_34801


namespace tan_315_eq_neg1_l34_34985

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34985


namespace mn_parallel_pq_l34_34396

open EuclideanGeometry

-- Let M be the midpoint of the arc AB of the circumcircle of triangle ABC
-- that does not contain point C, and N be the midpoint of the arc BC that 
-- does not contain point A. Prove that MN is parallel to PQ.

theorem mn_parallel_pq
  (A B C M N P Q : Point)
  (circumcircle : Circle)
  (triangleABC : Triangle A B C)
  (M_is_middle_arc_AB : is_arc_midpoint circumcircle A B C M)
  (N_is_middle_arc_BC : is_arc_midpoint circumcircle B C A N) :
   parallel MN PQ := sorry

end mn_parallel_pq_l34_34396


namespace find_y_z_l34_34160

theorem find_y_z (x y z : ℚ) (h1 : (x + y) / (z - x) = 9 / 2) (h2 : (y + z) / (y - x) = 5) (h3 : x = 43 / 4) :
  y = 12 / 17 + 17 ∧ z = 5 / 68 + 17 := 
by sorry

end find_y_z_l34_34160


namespace composite_divisor_bound_l34_34700

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end composite_divisor_bound_l34_34700


namespace greatest_three_digit_number_l34_34153

theorem greatest_three_digit_number
  (n : ℕ) (h_3digit : 100 ≤ n ∧ n < 1000) (h_mod7 : n % 7 = 2) (h_mod4 : n % 4 = 1) :
  n = 989 :=
sorry

end greatest_three_digit_number_l34_34153


namespace first_position_remainder_one_l34_34857

theorem first_position_remainder_one (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2023)
(h2 : ∀ b c d : ℕ, b = a ∧ c = a + 2 ∧ d = a + 4 → 
  b % 3 ≠ c % 3 ∧ c % 3 ≠ d % 3 ∧ d % 3 ≠ b % 3):
  a % 3 = 1 :=
sorry

end first_position_remainder_one_l34_34857


namespace time_worked_together_l34_34464

noncomputable def combined_rate (P_rate Q_rate : ℝ) : ℝ :=
  P_rate + Q_rate

theorem time_worked_together (P_rate Q_rate : ℝ) (t additional_time job_completed : ℝ) :
  P_rate = 1 / 4 ∧ Q_rate = 1 / 15 ∧ additional_time = 1 / 5 ∧ job_completed = (additional_time * P_rate) →
  (t * combined_rate P_rate Q_rate + job_completed = 1) → 
  t = 3 :=
sorry

end time_worked_together_l34_34464


namespace exist_five_natural_numbers_sum_and_product_equal_ten_l34_34491

theorem exist_five_natural_numbers_sum_and_product_equal_ten : 
  ∃ (n_1 n_2 n_3 n_4 n_5 : ℕ), 
  n_1 + n_2 + n_3 + n_4 + n_5 = 10 ∧ 
  n_1 * n_2 * n_3 * n_4 * n_5 = 10 := 
sorry

end exist_five_natural_numbers_sum_and_product_equal_ten_l34_34491


namespace tan_315_proof_l34_34961

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34961


namespace ellipse_equation_l34_34368

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (M : ℝ × ℝ)
  (h1 : 2 * a = 4)
  (h2 : 2 * b = 2 * a / 2)
  (h3 : M = (2, 1))
  (line_eq : ∀ k : ℝ, (y = 1 + k * (x - 2))) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 4 + y^2 = 1)) ∧
  (∃ k : ℝ, (k = -1/2) ∧ (∀ x y : ℝ, (y - 1 = k * (x - 2)) → (x + 2*y - 4 = 0))) :=
by
  sorry

end ellipse_equation_l34_34368


namespace worker_idle_days_l34_34041

theorem worker_idle_days (W I : ℕ) 
  (h1 : 20 * W - 3 * I = 280)
  (h2 : W + I = 60) : 
  I = 40 :=
sorry

end worker_idle_days_l34_34041


namespace value_of_a_l34_34258

theorem value_of_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a < 13) (h2 : 13 ∣ 12^20 + a) : a = 12 :=
by sorry

end value_of_a_l34_34258


namespace toy_train_produces_5_consecutive_same_tune_l34_34904

noncomputable def probability_same_tune (plays : ℕ) (p : ℚ) (tunes : ℕ) : ℚ :=
  p ^ plays

theorem toy_train_produces_5_consecutive_same_tune :
  probability_same_tune 5 (1/3) 3 = 1/243 :=
by
  sorry

end toy_train_produces_5_consecutive_same_tune_l34_34904


namespace Sherry_catches_train_within_5_minutes_l34_34270

-- Defining the probabilities given in the conditions
def P_A : ℝ := 0.75  -- Probability of train arriving
def P_N : ℝ := 0.75  -- Probability of Sherry not noticing the train

-- Event that no train arrives combined with event that train arrives but not noticed
def P_not_catch_in_a_minute : ℝ := 1 - P_A + P_A * P_N

-- Generalizing to 5 minutes
def P_not_catch_in_5_minutes : ℝ := P_not_catch_in_a_minute ^ 5

-- Probability Sherry catches the train within 5 minutes
def P_C : ℝ := 1 - P_not_catch_in_5_minutes

theorem Sherry_catches_train_within_5_minutes : P_C = 1 - (13 / 16) ^ 5 := by
  sorry

end Sherry_catches_train_within_5_minutes_l34_34270


namespace value_of_B_l34_34000

theorem value_of_B (B : ℝ) : 3 * B ^ 2 + 3 * B + 2 = 29 ↔ (B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2) :=
by sorry

end value_of_B_l34_34000


namespace minimum_value_condition_l34_34795

theorem minimum_value_condition (x a : ℝ) (h1 : x > a) (h2 : ∀ y, y > a → x + 4 / (y - a) > 9) : a = 6 :=
sorry

end minimum_value_condition_l34_34795


namespace factorize_expression_l34_34202

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l34_34202


namespace tangent_315_deg_l34_34982

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34982


namespace tan_315_eq_neg1_l34_34923

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34923


namespace smallest_n_for_partition_condition_l34_34690

theorem smallest_n_for_partition_condition :
  ∃ n : ℕ, n = 4 ∧ ∀ T, (T = {i : ℕ | 2 ≤ i ∧ i ≤ n}) →
  (∀ A B, (T = A ∪ B ∧ A ∩ B = ∅) →
   (∃ a b c, (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B) ∧ (a + b = c))) := sorry

end smallest_n_for_partition_condition_l34_34690


namespace set_A_is_2_3_l34_34890

noncomputable def A : Set ℤ := { x : ℤ | 3 / (x - 1) > 1 }

theorem set_A_is_2_3 : A = {2, 3} :=
by
  sorry

end set_A_is_2_3_l34_34890


namespace tan_315_eq_neg1_l34_34963

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34963


namespace max_red_socks_l34_34623

theorem max_red_socks (r b g t : ℕ) (h1 : t ≤ 2500) (h2 : r + b + g = t) 
  (h3 : (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 / 3) * t * (t - 1)) : 
  r ≤ 1625 :=
by 
  sorry

end max_red_socks_l34_34623


namespace tan_315_eq_neg1_l34_34914

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34914


namespace paula_candies_distribution_l34_34555

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l34_34555


namespace number_of_initials_sets_l34_34816

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l34_34816


namespace expression_equal_a_five_l34_34884

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end expression_equal_a_five_l34_34884


namespace intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l34_34370

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a ^ 2 + 1)) < 0 }

-- Theorem for question (1): Intersection of A and B when a = 2
theorem intersection_of_A_and_B_when_a_is_2 :
  setA 2 ∩ setB 2 = { x | 4 < x ∧ x < 5 } :=
sorry

-- Theorem for question (2): Range of a such that B ⊆ A
theorem range_of_a_such_that_B_subset_A :
  { a : ℝ | setB a ⊆ setA a } = { x | 1 < x ∧ x ≤ 3 } ∪ { -1 } :=
sorry

end intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l34_34370


namespace area_outside_small_squares_l34_34767

theorem area_outside_small_squares (a b : ℕ) (ha : a = 10) (hb : b = 4) (n : ℕ) (hn: n = 2) :
  a^2 - n * b^2 = 68 :=
by
  rw [ha, hb, hn]
  sorry

end area_outside_small_squares_l34_34767


namespace time_saved_is_six_minutes_l34_34776

-- Conditions
def distance_monday : ℝ := 3
def distance_wednesday : ℝ := 4
def distance_friday : ℝ := 5

def speed_monday : ℝ := 6
def speed_wednesday : ℝ := 4
def speed_friday : ℝ := 5

def speed_constant : ℝ := 5

-- Question (proof statement)
theorem time_saved_is_six_minutes : 
  (distance_monday / speed_monday + distance_wednesday / speed_wednesday + distance_friday / speed_friday) - (distance_monday + distance_wednesday + distance_friday) / speed_constant = 0.1 :=
by
  sorry

end time_saved_is_six_minutes_l34_34776


namespace side_length_of_square_l34_34430

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l34_34430


namespace coupon_probability_l34_34583

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l34_34583


namespace g_of_12_l34_34512

def g (n : ℕ) : ℕ := n^2 - n + 23

theorem g_of_12 : g 12 = 155 :=
by
  sorry

end g_of_12_l34_34512


namespace cost_price_of_book_l34_34751

theorem cost_price_of_book
  (SP : Real)
  (profit_percentage : Real)
  (h1 : SP = 300)
  (h2 : profit_percentage = 0.20) :
  ∃ CP : Real, CP = 250 :=
by
  -- Proof of the statement
  sorry

end cost_price_of_book_l34_34751


namespace rank_matrix_sum_l34_34257

theorem rank_matrix_sum (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h : ∀ i j, A i j = ↑i + ↑j) : Matrix.rank A = 2 := by
  sorry

end rank_matrix_sum_l34_34257


namespace number_of_sheep_total_number_of_animals_l34_34116

theorem number_of_sheep (ratio_sh_horse : 5 / 7 * horses = sheep) 
    (horse_food_per_day : horses * 230 = 12880) :
    sheep = 40 :=
by
  -- These are all the given conditions
  sorry

theorem total_number_of_animals (sheep : ℕ) (horses : ℕ)
    (H1 : sheep = 40) (H2 : horses = 56) :
    sheep + horses = 96 :=
by
  -- Given conditions for the total number of animals on the farm
  sorry

end number_of_sheep_total_number_of_animals_l34_34116


namespace lowest_price_per_component_l34_34306

def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def components_per_month : ℕ := 150

theorem lowest_price_per_component (price_per_component : ℝ) :
  let total_cost_per_component := production_cost_per_component + shipping_cost_per_component
  let total_production_and_shipping_cost := total_cost_per_component * components_per_month
  let total_cost := total_production_and_shipping_cost + fixed_monthly_costs
  price_per_component = total_cost / components_per_month → price_per_component = 196 :=
by
  sorry

end lowest_price_per_component_l34_34306


namespace polynomial_at_x_neg_four_l34_34873

noncomputable def f (x : ℝ) : ℝ :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

theorem polynomial_at_x_neg_four : 
  f (-4) = 220 := by
  sorry

end polynomial_at_x_neg_four_l34_34873


namespace num_four_letter_initials_l34_34805

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l34_34805


namespace time_difference_is_16_point_5_l34_34639

noncomputable def time_difference : ℝ :=
  let danny_to_steve : ℝ := 33
  let steve_to_danny := 2 * danny_to_steve -- Steve takes twice the time as Danny
  let emma_to_houses : ℝ := 40
  let danny_halfway := danny_to_steve / 2 -- Halfway point for Danny
  let steve_halfway := steve_to_danny / 2 -- Halfway point for Steve
  let emma_halfway := emma_to_houses / 2 -- Halfway point for Emma
  -- Additional times to the halfway point
  let steve_additional := steve_halfway - danny_halfway
  let emma_additional := emma_halfway - danny_halfway
  -- The final result is the maximum of these times
  max steve_additional emma_additional

theorem time_difference_is_16_point_5 : time_difference = 16.5 :=
  by
  sorry

end time_difference_is_16_point_5_l34_34639


namespace value_of_a_minus_c_l34_34094

theorem value_of_a_minus_c
  (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) :
  a - c = -200 := sorry

end value_of_a_minus_c_l34_34094


namespace puppies_brought_in_correct_l34_34483

-- Define the initial number of puppies in the shelter
def initial_puppies: Nat := 2

-- Define the number of puppies adopted per day
def puppies_adopted_per_day: Nat := 4

-- Define the number of days over which the puppies are adopted
def adoption_days: Nat := 9

-- Define the total number of puppies adopted after the given days
def total_puppies_adopted: Nat := puppies_adopted_per_day * adoption_days

-- Define the number of puppies brought in
def puppies_brought_in: Nat := total_puppies_adopted - initial_puppies

-- Prove that the number of puppies brought in is 34
theorem puppies_brought_in_correct: puppies_brought_in = 34 := by
  -- proof omitted, filled with sorry to skip the proof
  sorry

end puppies_brought_in_correct_l34_34483


namespace eric_days_waited_l34_34492

def num_chickens := 4
def eggs_per_chicken_per_day := 3
def total_eggs := 36

def eggs_per_day := num_chickens * eggs_per_chicken_per_day
def num_days := total_eggs / eggs_per_day

theorem eric_days_waited : num_days = 3 :=
by
  sorry

end eric_days_waited_l34_34492


namespace eccentricity_of_ellipse_l34_34851

noncomputable def calculate_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_ellipse : 
  (calculate_eccentricity 5 4) = 3 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l34_34851


namespace evaluate_polynomial_l34_34788

theorem evaluate_polynomial : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_l34_34788


namespace cost_of_pen_is_30_l34_34482

noncomputable def mean_expenditure_per_day : ℕ := 500
noncomputable def days_in_week : ℕ := 7
noncomputable def total_expenditure : ℕ := mean_expenditure_per_day * days_in_week

noncomputable def mon_expenditure : ℕ := 450
noncomputable def tue_expenditure : ℕ := 600
noncomputable def wed_expenditure : ℕ := 400
noncomputable def thurs_expenditure : ℕ := 500
noncomputable def sat_expenditure : ℕ := 550
noncomputable def sun_expenditure : ℕ := 300

noncomputable def fri_notebook_cost : ℕ := 50
noncomputable def fri_earphone_cost : ℕ := 620

noncomputable def total_non_fri_expenditure : ℕ := 
  mon_expenditure + tue_expenditure + wed_expenditure + 
  thurs_expenditure + sat_expenditure + sun_expenditure

noncomputable def fri_expenditure : ℕ := 
  total_expenditure - total_non_fri_expenditure

noncomputable def fri_pen_cost : ℕ := 
  fri_expenditure - (fri_earphone_cost + fri_notebook_cost)

theorem cost_of_pen_is_30 : fri_pen_cost = 30 :=
  sorry

end cost_of_pen_is_30_l34_34482


namespace find_coefficients_sum_l34_34227

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end find_coefficients_sum_l34_34227


namespace merchant_discount_l34_34901

-- Definitions used in Lean 4 statement coming directly from conditions
def initial_cost_price : Real := 100
def marked_up_percentage : Real := 0.80
def profit_percentage : Real := 0.35

-- To prove the percentage discount offered
theorem merchant_discount (cp mp sp discount percentage_discount : Real) 
  (H1 : cp = initial_cost_price)
  (H2 : mp = cp + (marked_up_percentage * cp))
  (H3 : sp = cp + (profit_percentage * cp))
  (H4 : discount = mp - sp)
  (H5 : percentage_discount = (discount / mp) * 100) :
  percentage_discount = 25 := 
sorry

end merchant_discount_l34_34901


namespace divisibility_of_product_l34_34294

def three_consecutive_integers (a1 a2 a3 : ℤ) : Prop :=
  a1 = a2 - 1 ∧ a3 = a2 + 1

theorem divisibility_of_product (a1 a2 a3 : ℤ) (h : three_consecutive_integers a1 a2 a3) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by
  cases h with
  | intro ha1 ha3 =>
    sorry

end divisibility_of_product_l34_34294


namespace min_value_of_f_l34_34059

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x - 3 / x

theorem min_value_of_f : ∃ x < 0, ∀ y : ℝ, y = f x → y ≥ 1 + 2 * Real.sqrt 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end min_value_of_f_l34_34059


namespace no_three_distinct_rational_roots_l34_34112

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ ∃ (u v w : ℚ), 
    u + v + w = -(2 * a + 1) ∧ 
    u * v + v * w + w * u = (2 * a^2 + 2 * a - 3) ∧ 
    u * v * w = b := sorry

end no_three_distinct_rational_roots_l34_34112


namespace value_of_y_square_plus_inverse_square_l34_34520

variable {y : ℝ}
variable (h : 35 = y^4 + 1 / y^4)

theorem value_of_y_square_plus_inverse_square (h : 35 = y^4 + 1 / y^4) : y^2 + 1 / y^2 = Real.sqrt 37 := 
sorry

end value_of_y_square_plus_inverse_square_l34_34520


namespace emma_list_count_l34_34645

theorem emma_list_count : 
  let m1 := 900
  let m2 := 27000
  let d := 30
  (m1 / d <= m2 / d) → (m2 / d - m1 / d + 1 = 871) :=
by
  intros m1 m2 d h
  have h1 : m1 / d ≤ m2 / d := h
  have h2 : m2 / d - m1 / d + 1 = 871 := by sorry
  exact h2

end emma_list_count_l34_34645


namespace solve_x_l34_34496

theorem solve_x (x : ℚ) : (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := 
by
  sorry

end solve_x_l34_34496


namespace solutions_eq1_solutions_eq2_l34_34136

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l34_34136


namespace find_m_n_find_a_l34_34073

def quadratic_roots (x : ℝ) (m n : ℝ) : Prop := 
  x^2 + m * x - 3 = 0

theorem find_m_n {m n : ℝ} : 
  quadratic_roots (-1) m n ∧ quadratic_roots n m n → 
  m = -2 ∧ n = 3 := 
sorry

def f (x m : ℝ) : ℝ := 
  x^2 + m * x - 3

theorem find_a {a m : ℝ} (h : m = -2) : 
  f 3 m = f (2 * a - 3) m → 
  a = 1 ∨ a = 3 := 
sorry

end find_m_n_find_a_l34_34073


namespace identify_quadratic_l34_34600

def is_quadratic (eq : String) : Prop :=
  eq = "x^2 - 2x + 1 = 0"

theorem identify_quadratic :
  is_quadratic "x^2 - 2x + 1 = 0" :=
by
  sorry

end identify_quadratic_l34_34600


namespace min_value_expression_l34_34067

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l34_34067


namespace tan_315_eq_neg1_l34_34917

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34917


namespace find_a_l34_34834

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 + 2

theorem find_a (a : ℝ) : deriv (deriv (f a)) (-1) = 3 → a = 1 :=
by
  intro h
  sorry

end find_a_l34_34834


namespace sports_popularity_order_l34_34192

theorem sports_popularity_order :
  let soccer := (13 : ℚ) / 40
  let baseball := (9 : ℚ) / 30
  let basketball := (7 : ℚ) / 20
  let volleyball := (3 : ℚ) / 10
  basketball > soccer ∧ soccer > baseball ∧ baseball = volleyball :=
by
  sorry

end sports_popularity_order_l34_34192


namespace prism_volume_eq_400_l34_34141

noncomputable def prism_volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_eq_400 
  (a b c : ℝ)
  (h1 : a * b = 40)
  (h2 : a * c = 50)
  (h3 : b * c = 80) :
  prism_volume a b c = 400 :=
by
  sorry

end prism_volume_eq_400_l34_34141


namespace max_mondays_in_first_51_days_l34_34594

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l34_34594


namespace find_f1_find_f8_inequality_l34_34110

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f x
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_multiplicative : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y
axiom f_of_2 : f 2 = 4

-- Statements to prove
theorem find_f1 : f 1 = 1 := sorry
theorem find_f8 : f 8 = 64 := sorry
theorem inequality : ∀ x : ℝ, 3 < x → x ≤ 7 / 2 → 16 * f (1 / (x - 3)) ≥ f (2 * x + 1) := sorry

end find_f1_find_f8_inequality_l34_34110


namespace A_B_symmetric_x_axis_l34_34531

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Theorem stating the symmetry relationship between points A and B with respect to the x-axis
theorem A_B_symmetric_x_axis (xA yA xB yB : ℝ) (hA : A = (xA, yA)) (hB : B = (xB, yB)) :
  xA = xB ∧ yA = -yB := by
  sorry

end A_B_symmetric_x_axis_l34_34531


namespace pages_with_money_l34_34850

def cost_per_page : ℝ := 3.5
def total_money : ℝ := 15 * 100

theorem pages_with_money : ⌊total_money / cost_per_page⌋ = 428 :=
by sorry

end pages_with_money_l34_34850


namespace solve_system_l34_34837

theorem solve_system (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : 2 * b - 3 * a = 4) : b = 2 :=
by {
  -- Given the conditions, we need to show that b = 2
  sorry
}

end solve_system_l34_34837


namespace triangle_inequalities_l34_34321

theorem triangle_inequalities (a b c h_a h_b h_c : ℝ) (ha_eq : h_a = b * Real.sin (arc_c)) (hb_eq : h_b = a * Real.sin (arc_c)) (hc_eq : h_c = a * Real.sin (arc_b)) (h : a > b) (h2 : b > c) :
  (a + h_a > b + h_b) ∧ (b + h_b > c + h_c) :=
by
  sorry

end triangle_inequalities_l34_34321


namespace systematic_sampling_third_group_number_l34_34172

theorem systematic_sampling_third_group_number :
  ∀ (total_members groups sample_number group_5_number group_gap : ℕ),
  total_members = 200 →
  groups = 40 →
  sample_number = total_members / groups →
  group_5_number = 22 →
  group_gap = 5 →
  (group_this_number : ℕ) = group_5_number - (5 - 3) * group_gap →
  group_this_number = 12 :=
by
  intros total_members groups sample_number group_5_number group_gap Htotal Hgroups Hsample Hgroup5 Hgap Hthis_group
  sorry

end systematic_sampling_third_group_number_l34_34172


namespace evaluate_six_applications_problem_solution_l34_34543

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

theorem evaluate_six_applications (θ : ℚ) : 
  r (r (r (r (r (r θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

theorem problem_solution : r (r (r (r (r (r 30))))) = 158 / 253 :=
by
  have h : r (r (r (r (r (r 30))))) = (8 + 5 * 30) / (13 + 8 * 30) := by
    exact evaluate_six_applications 30
  rw [h]
  norm_num

end evaluate_six_applications_problem_solution_l34_34543


namespace find_integer_value_of_a_l34_34852

-- Define the conditions for the equation and roots
def equation_has_two_distinct_negative_integer_roots (a : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ (a^2 - 1) * x1^2 - 2 * (5 * a + 1) * x1 + 24 = 0 ∧ (a^2 - 1) * x2^2 - 2 * (5 * a + 1) * x2 + 24 = 0 ∧
  x1 = 6 / (a - 1) ∧ x2 = 4 / (a + 1)

-- Prove that the only integer value of a that satisfies these conditions is -2
theorem find_integer_value_of_a : 
  ∃ (a : ℤ), equation_has_two_distinct_negative_integer_roots a ∧ a = -2 := 
sorry

end find_integer_value_of_a_l34_34852


namespace fraction_simplification_l34_34782

theorem fraction_simplification : 
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1 / 3 :=
by
  sorry

end fraction_simplification_l34_34782


namespace number_of_laborers_l34_34031

theorem number_of_laborers (x : ℕ) :
  (18 * (x - 10)) = x → x = 11 :=
by
  assume h : 18 * (x - 10) = x
  have : 18 * (x - 10) = x, from h
  sorry

end number_of_laborers_l34_34031


namespace bob_salary_is_14400_l34_34693

variables (mario_salary_current : ℝ) (mario_salary_last_year : ℝ) (bob_salary_last_year : ℝ) (bob_salary_current : ℝ)

-- Given Conditions
axiom mario_salary_increase : mario_salary_current = 4000
axiom mario_salary_equation : 1.40 * mario_salary_last_year = mario_salary_current
axiom bob_salary_last_year_equation : bob_salary_last_year = 3 * mario_salary_current
axiom bob_salary_increase : bob_salary_current = bob_salary_last_year + 0.20 * bob_salary_last_year

-- Theorem to prove
theorem bob_salary_is_14400 
    (mario_salary_last_year_eq : mario_salary_last_year = 4000 / 1.40)
    (bob_salary_last_year_eq : bob_salary_last_year = 3 * 4000)
    (bob_salary_current_eq : bob_salary_current = 12000 + 0.20 * 12000) :
    bob_salary_current = 14400 := 
by
  sorry

end bob_salary_is_14400_l34_34693


namespace green_eyed_brunettes_percentage_l34_34325

noncomputable def green_eyed_brunettes_proportion (a b c d : ℕ) 
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7) 
  (h3 : c / (c + d) = 0.1) : Prop :=
  d / (a + b + c + d) = 0.54

-- The main theorem to be proved
theorem green_eyed_brunettes_percentage (a b c d : ℕ)
  (h1 : a / (a + b) = 0.65)
  (h2 : b / (b + c) = 0.7)
  (h3 : c / (c + d) = 0.1) : 
  green_eyed_brunettes_proportion a b c d h1 h2 h3 := 
sorry

end green_eyed_brunettes_percentage_l34_34325


namespace avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l34_34053

-- Average fuel consumption per kilometer
noncomputable def avgFuelConsumption (initial_fuel: ℝ) (final_fuel: ℝ) (distance: ℝ) : ℝ :=
  (initial_fuel - final_fuel) / distance

-- Relationship between remaining fuel Q and distance x
noncomputable def remainingFuel (initial_fuel: ℝ) (consumption_rate: ℝ) (distance: ℝ) : ℝ :=
  initial_fuel - consumption_rate * distance

-- Check if the car can return home without refueling
noncomputable def canReturnHome (initial_fuel: ℝ) (consumption_rate: ℝ) (round_trip_distance: ℝ) (alarm_fuel_level: ℝ) : Bool :=
  initial_fuel - consumption_rate * round_trip_distance ≥ alarm_fuel_level

-- Theorem statements to prove
theorem avg_fuel_consumption_correct :
  avgFuelConsumption 45 27 180 = 0.1 :=
sorry

theorem remaining_fuel_correct :
  ∀ x, remainingFuel 45 0.1 x = 45 - 0.1 * x :=
sorry

theorem cannot_return_home_without_refueling :
  ¬canReturnHome 45 0.1 (220 * 2) 3 :=
sorry

end avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l34_34053


namespace bad_carrots_eq_13_l34_34614

-- Define the number of carrots picked by Haley
def haley_picked : ℕ := 39

-- Define the number of carrots picked by her mom
def mom_picked : ℕ := 38

-- Define the number of good carrots
def good_carrots : ℕ := 64

-- Define the total number of carrots picked
def total_carrots : ℕ := haley_picked + mom_picked

-- State the theorem to prove the number of bad carrots
theorem bad_carrots_eq_13 : total_carrots - good_carrots = 13 := by
  sorry

end bad_carrots_eq_13_l34_34614


namespace reduced_fraction_numerator_l34_34762

theorem reduced_fraction_numerator :
  let numerator := 4128 
  let denominator := 4386 
  let gcd := Nat.gcd numerator denominator
  let reduced_numerator := numerator / gcd 
  let reduced_denominator := denominator / gcd 
  (reduced_numerator : ℚ) / (reduced_denominator : ℚ) = 16 / 17 → reduced_numerator = 16 :=
by
  intros
  sorry

end reduced_fraction_numerator_l34_34762


namespace opposite_of_neg_two_l34_34718

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l34_34718


namespace bella_steps_l34_34911

-- Define the conditions and the necessary variables
variable (b : ℝ) (distance : ℝ) (steps_per_foot : ℝ)

-- Given constants
def bella_speed := b
def ella_speed := 4 * b
def combined_speed := bella_speed + ella_speed
def total_distance := 15840
def feet_per_step := 3

-- Define the main theorem to prove the number of steps Bella takes
theorem bella_steps : (total_distance / combined_speed) * bella_speed / feet_per_step = 1056 := by
  sorry

end bella_steps_l34_34911


namespace food_bank_remaining_l34_34545

theorem food_bank_remaining :
  ∀ (f1 f2 : ℕ) (p : ℚ),
  f1 = 40 →
  f2 = 2 * f1 →
  p = 0.7 →
  (f1 + f2) - (p * (f1 + f2)).toNat = 36 :=
by
  intros f1 f2 p h1 h2 h3
  sorry

end food_bank_remaining_l34_34545


namespace range_of_f_l34_34392

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x) ^ 3 + (Real.arcsin x) ^ 3

theorem range_of_f : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
           ∃ y : ℝ, y = f x ∧ (y ≥ (Real.pi ^ 3) / 32) ∧ (y ≤ (7 * (Real.pi ^ 3)) / 8) :=
sorry

end range_of_f_l34_34392


namespace incenter_divides_segment_l34_34140

variables (A B C I M : Type) (R r : ℝ)

-- Definitions based on conditions
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_circumcircle (C : Type) : Prop := sorry
def angle_bisector_intersects_at (A B C M : Type) : Prop := sorry
def divides_segment (I M : Type) (a b : ℝ) : Prop := sorry

-- Proof problem statement
theorem incenter_divides_segment (h1 : is_circumcircle C)
                                   (h2 : is_incenter I A B C)
                                   (h3 : angle_bisector_intersects_at A B C M)
                                   (h4 : divides_segment I M a b) :
  a * b = 2 * R * r :=
sorry

end incenter_divides_segment_l34_34140


namespace five_x_plus_four_is_25_over_7_l34_34091

theorem five_x_plus_four_is_25_over_7 (x : ℚ) (h : 5 * x - 8 = 12 * x + 15) : 5 * (x + 4) = 25 / 7 := by
  sorry

end five_x_plus_four_is_25_over_7_l34_34091


namespace als_initial_portion_l34_34322

theorem als_initial_portion (a b c : ℝ)
  (h1 : a + b + c = 1200)
  (h2 : a - 150 + 3 * b + 3 * c = 1800) :
  a = 825 :=
sorry

end als_initial_portion_l34_34322


namespace quiz_answer_key_combinations_l34_34530

noncomputable def num_ways_answer_key : ℕ :=
  let true_false_combinations := 2^4
  let valid_true_false_combinations := true_false_combinations - 2
  let multi_choice_combinations := 4 * 4
  valid_true_false_combinations * multi_choice_combinations

theorem quiz_answer_key_combinations : num_ways_answer_key = 224 := 
by
  sorry

end quiz_answer_key_combinations_l34_34530


namespace perimeter_proof_l34_34100

noncomputable def perimeter (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
  else if x > (Real.sqrt 3) / 3 ∧ x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
  else if x > (2 * Real.sqrt 3) / 3 ∧ x ≤ Real.sqrt 3 then 3 * Real.sqrt 6 * (Real.sqrt 3 - x)
  else 0

theorem perimeter_proof (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.sqrt 3) :
  perimeter x = 
    if x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) :=
by 
  sorry

end perimeter_proof_l34_34100


namespace arithmetic_seq_problem_l34_34247

variable (a : ℕ → ℕ)
variable h_seq : ArithmeticSequence a
variable h_cond : a 5 + a 13 = 40

theorem arithmetic_seq_problem : a 8 + a 9 + a 10 = 60 :=
by
  sorry

end arithmetic_seq_problem_l34_34247


namespace tan_315_eq_neg1_l34_34953

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34953


namespace product_of_bc_l34_34452

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l34_34452


namespace sum_of_gcd_values_l34_34327

open Nat

theorem sum_of_gcd_values (n : ℕ) (h : n > 0) :
  (Finset.sum (Finset.map ⟨λ n, gcd (5 * n + 6) (2 * n + 3), λ _ _, Finset.mem_univ _⟩ (Finset.range (n + 1))) = 4) :=
by
  sorry

end sum_of_gcd_values_l34_34327


namespace slower_speed_for_on_time_arrival_l34_34691

variable (distance : ℝ) (actual_speed : ℝ) (time_early : ℝ)

theorem slower_speed_for_on_time_arrival 
(h1 : distance = 20)
(h2 : actual_speed = 40)
(h3 : time_early = 1 / 15) :
  actual_speed - (600 / 17) = 4.71 :=
by 
  sorry

end slower_speed_for_on_time_arrival_l34_34691


namespace sum_F_G_H_l34_34372

theorem sum_F_G_H : 
  ∀ (F G H : ℕ), 
    (F < 10 ∧ G < 10 ∧ H < 10) ∧ 
    ∃ k : ℤ, 
      (F - 8 + 6 - 1 + G - 2 - H - 11 * k = 0) → 
        F + G + H = 23 :=
by sorry

end sum_F_G_H_l34_34372


namespace AFG_is_isosceles_l34_34109

-- Definitions of the geometric entities as per conditions

-- Axiomatically define the points and their relationships as per isosceles trapezoid
variables {A B C D E F G : Point}

-- Additional necessary conditions and axioms as per the problem statement
axiom is_isosceles_trapezoid (ABCD : IsoscelesTrapezoid A B C D) : 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧ ABCD.AB.parallel ABCD.CD

axiom inscribed_circle (ω : Circle) (BCD : Triangle B C D) (meets_at : MeetCircleAt ω BCD E) :
    ω.inscribed_in BCD ∧ E ∈ CD ∧ Circle.is_tangent_at ω CD E

axiom point_F_on_bisector (F : Point) (D A C : Point) (bisector : AngleBisector (∠ D A C) F) :
    is_internal_angle_bisector D A C F

axiom EF_perp_CD (E F CD : Line) : (Line.perpendicular_to E F CD)

axiom circumscribed_circle (circum_ACF : Circle) (ACF : Triangle A C F) (meets_at : MeetCircleAt circum_ACF ACF C G) :
    circum_ACF.circumscribed_in ACF ∧ C ∈ CD ∧ G ∈ CD

-- Theorem to prove triangle AFG is isosceles.
theorem AFG_is_isosceles (ABCD : IsoscelesTrapezoid A B C D) (ω : Circle) (BCD : Triangle B C D) 
    (triangle_inscribed : MeetCircleAt ω BCD E) (F : Point) (bisector : AngleBisector (∠ D A C) F) 
    (EF_CD_perpendicular : Line.perpendicular_to E F (Line.mk CD)) 
    (circum_ACF : Circle) (meet_ACF_at_CG : MeetCircleAt circum_ACF (Triangle.mk A C F) C G) :
    Triangle.is_isosceles (Triangle.mk A F G) :=
sorry

end AFG_is_isosceles_l34_34109


namespace opposite_of_neg_two_l34_34728

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l34_34728


namespace quadrilateral_area_inequality_l34_34604

theorem quadrilateral_area_inequality
  (a b c d S : ℝ)
  (hS : 0 ≤ S)
  (h : S = (a + b) / 4 * (c + d) / 4)
  : S ≤ (a + b) / 4 * (c + d) / 4 := by
  sorry

end quadrilateral_area_inequality_l34_34604


namespace num_prime_factors_30_fact_l34_34331

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def is_prime (n : ℕ) : Bool :=
  if h : n ≤ 1 then false else
    let divisors := List.range (n - 2) |>.map (· + 2)
    !divisors.any (· ∣ n)

def primes_upto (n : ℕ) : List ℕ :=
  List.range (n - 1) |>.map (· + 1) |>.filter is_prime

def count_primes_factorial_upto (n : ℕ) : ℕ :=
  (primes_upto n).length

theorem num_prime_factors_30_fact : count_primes_factorial_upto 30 = 10 := sorry

end num_prime_factors_30_fact_l34_34331


namespace expression_equal_a_five_l34_34886

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end expression_equal_a_five_l34_34886


namespace missing_jar_size_l34_34534

theorem missing_jar_size (x : ℕ) (h₁ : 3 * 16 + 3 * x + 3 * 40 = 252) 
                          (h₂ : 3 + 3 + 3 = 9) : x = 28 := 
by 
  sorry

end missing_jar_size_l34_34534


namespace percentage_difference_l34_34019

theorem percentage_difference : (70 / 100 : ℝ) * 100 - (60 / 100 : ℝ) * 80 = 22 := by
  sorry

end percentage_difference_l34_34019


namespace solve_quadratic_l34_34844

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 5 * x ^ 2 + 9 * x - 18 = 0) : x = 6 / 5 :=
by
  sorry

end solve_quadratic_l34_34844


namespace side_length_of_square_l34_34423

theorem side_length_of_square (d : ℝ) (h₁ : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  use 2
  split
  · rfl
  · rw [h₁]
    sorry

end side_length_of_square_l34_34423


namespace candies_per_friend_l34_34558

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l34_34558


namespace tan_315_degrees_l34_34998

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l34_34998


namespace adjacent_probability_l34_34756

theorem adjacent_probability (n : ℕ) (hn : n ≥ 2) : 
  let total_perms := nat.factorial n in
  let fav_perms := 2 * (n - 1) * nat.factorial (n - 2) in
  fav_perms / total_perms = 2 / n :=
sorry

end adjacent_probability_l34_34756


namespace option_C_equals_a5_l34_34888

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end option_C_equals_a5_l34_34888


namespace cyclist_wait_time_l34_34295

theorem cyclist_wait_time
  (hiker_speed : ℝ)
  (hiker_speed_pos : hiker_speed = 4)
  (cyclist_speed : ℝ)
  (cyclist_speed_pos : cyclist_speed = 24)
  (waiting_time_minutes : ℝ)
  (waiting_time_minutes_pos : waiting_time_minutes = 5) :
  (waiting_time_minutes / 60) * cyclist_speed = 2 →
  (2 / hiker_speed) * 60 = 30 :=
by
  intros
  sorry

end cyclist_wait_time_l34_34295


namespace WR_eq_35_l34_34675

theorem WR_eq_35 (PQ ZY SX : ℝ) (hPQ : PQ = 30) (hZY : ZY = 15) (hSX : SX = 10) :
    let WS := ZY - SX
    let SR := PQ
    let WR := WS + SR
    WR = 35 := by
  sorry

end WR_eq_35_l34_34675


namespace tan_315_eq_neg1_l34_34933

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34933


namespace simplify_expression_l34_34044

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l34_34044


namespace vector_BC_l34_34098

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_BC (BA CA BC : ℝ × ℝ) (BA_def : BA = (1, 2)) (CA_def : CA = (4, 5)) (BC_def : BC = vector_sub BA CA) : BC = (-3, -3) :=
by
  subst BA_def
  subst CA_def
  subst BC_def
  sorry

end vector_BC_l34_34098


namespace find_S_l34_34090

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S) ^ 2) (h2 : S > 0) :
  S = 333332 := 
sorry

end find_S_l34_34090


namespace bars_sold_this_week_l34_34388

-- Definitions based on conditions
def total_bars : Nat := 18
def bars_sold_last_week : Nat := 5
def bars_needed_to_sell : Nat := 6

-- Statement of the proof problem
theorem bars_sold_this_week : (total_bars - (bars_needed_to_sell + bars_sold_last_week)) = 2 := by
  -- proof goes here
  sorry

end bars_sold_this_week_l34_34388


namespace round_robin_10_person_tournament_l34_34034

noncomputable def num_matches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem round_robin_10_person_tournament :
  num_matches 10 = 45 :=
by
  sorry

end round_robin_10_person_tournament_l34_34034


namespace opposite_of_neg_two_l34_34731

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l34_34731


namespace moles_of_C2H6_l34_34209

-- Define the reactive coefficients
def ratio_C := 2
def ratio_H2 := 3
def ratio_C2H6 := 1

-- Given conditions
def moles_C := 6
def moles_H2 := 9

-- Function to calculate moles of C2H6 formed
def moles_C2H6_formed (m_C : ℕ) (m_H2 : ℕ) : ℕ :=
  min (m_C * ratio_C2H6 / ratio_C) (m_H2 * ratio_C2H6 / ratio_H2)

-- Theorem statement: the number of moles of C2H6 formed is 3
theorem moles_of_C2H6 : moles_C2H6_formed moles_C moles_H2 = 3 :=
by {
  -- Sorry is used since we are not providing the proof here
  sorry
}

end moles_of_C2H6_l34_34209


namespace tan_315_eq_neg1_l34_34916

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34916


namespace tan_315_eq_neg1_l34_34984

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34984


namespace fred_money_last_week_l34_34829

theorem fred_money_last_week (F_current F_earned F_last_week : ℕ) 
  (h_current : F_current = 86)
  (h_earned : F_earned = 63)
  (h_last_week : F_last_week = 23) :
  F_current - F_earned = F_last_week := 
by
  sorry

end fred_money_last_week_l34_34829


namespace rectangle_area_eq_l34_34775

theorem rectangle_area_eq (a b c d x y z w : ℝ)
  (h1 : a = x + y) (h2 : b = y + z) (h3 : c = z + w) (h4 : d = w + x) :
  a + c = b + d :=
by
  sorry

end rectangle_area_eq_l34_34775


namespace simplify_expression_l34_34703

def a : ℚ := (3 / 4) * 60
def b : ℚ := (8 / 5) * 60
def c : ℚ := 63

theorem simplify_expression : a - b + c = 12 := by
  sorry

end simplify_expression_l34_34703


namespace suji_age_problem_l34_34268

theorem suji_age_problem (x : ℕ) 
  (h1 : 5 * x + 6 = 13 * (4 * x + 6) / 11)
  (h2 : 11 * (4 * x + 6) = 9 * (3 * x + 6)) :
  4 * x = 16 :=
by
  sorry

end suji_age_problem_l34_34268


namespace expected_value_of_win_l34_34312

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l34_34312


namespace determinant_value_l34_34043

variable (a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 : ℝ)

def matrix_det : ℝ :=
  Matrix.det ![
    ![a1, b1, c1, d1],
    ![a1, b2, c2, d2],
    ![a1, b2, c3, d3],
    ![a1, b2, c3, d4]
  ]

theorem determinant_value : 
  matrix_det a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 = 
  a1 * (b2 - b1) * (c3 - c2) * (d4 - d3) :=
by
  sorry

end determinant_value_l34_34043


namespace number_is_209_given_base_value_is_100_l34_34476

theorem number_is_209_given_base_value_is_100 (n : ℝ) (base_value : ℝ) (H : base_value = 100) (percentage : ℝ) (H1 : percentage = 2.09) : n = 209 :=
by
  sorry

end number_is_209_given_base_value_is_100_l34_34476


namespace no_valid_solution_l34_34836

theorem no_valid_solution (x y z : ℤ) (h1 : x = 11 * y + 4) 
  (h2 : 2 * x = 24 * y + 3) (h3 : x + z = 34 * y + 5) : 
  ¬ ∃ (y : ℤ), 13 * y - x + 7 * z = 0 :=
by
  sorry

end no_valid_solution_l34_34836


namespace coeff_m6n6_in_mn_12_l34_34460

open BigOperators

theorem coeff_m6n6_in_mn_12 (m n : ℕ) : 
  (∑ k in finset.range (13), (nat.choose 12 k) * m^k * n^(12 - k)) = 
  (nat.choose 12 6) * m^6 * n^6 :=
by sorry

end coeff_m6n6_in_mn_12_l34_34460


namespace ab_leq_one_l34_34521

theorem ab_leq_one (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

end ab_leq_one_l34_34521


namespace garden_table_ratio_l34_34315

theorem garden_table_ratio (x y : ℝ) (h₁ : x + y = 750) (h₂ : y = 250) : x / y = 2 :=
by
  -- Proof omitted
  sorry

end garden_table_ratio_l34_34315


namespace tan_double_angle_identity_l34_34359

theorem tan_double_angle_identity (theta : ℝ) (h1 : 0 < theta ∧ theta < Real.pi / 2)
  (h2 : Real.sin theta - Real.cos theta = Real.sqrt 5 / 5) :
  Real.tan (2 * theta) = -(4 / 3) := 
by
  sorry

end tan_double_angle_identity_l34_34359


namespace solve_equation_2021_l34_34133

theorem solve_equation_2021 (x : ℝ) (hx : 0 ≤ x) : 
  2021 * x = 2022 * (x ^ (2021 : ℕ)) ^ (1 / (2021 : ℕ)) - 1 → x = 1 := 
by
  sorry

end solve_equation_2021_l34_34133


namespace frog_hops_ratio_l34_34457

theorem frog_hops_ratio :
  ∀ (F1 F2 F3 : ℕ),
    F1 = 4 * F2 →
    F1 + F2 + F3 = 99 →
    F2 = 18 →
    (F2 : ℚ) / (F3 : ℚ) = 2 :=
by
  intros F1 F2 F3 h1 h2 h3
  -- algebraic manipulations and proof to be filled here
  sorry

end frog_hops_ratio_l34_34457


namespace evaluate_expression_l34_34789

theorem evaluate_expression (a : ℝ) : (a^7 + a^7 + a^7 - a^7) = a^8 :=
by
  sorry

end evaluate_expression_l34_34789


namespace gain_percent_is_correct_l34_34018

theorem gain_percent_is_correct :
  let CP : ℝ := 450
  let SP : ℝ := 520
  let gain : ℝ := SP - CP
  let gain_percent : ℝ := (gain / CP) * 100
  gain_percent = 15.56 :=
by
  sorry

end gain_percent_is_correct_l34_34018


namespace bill_tossed_21_objects_l34_34634

-- Definitions based on the conditions from step a)
def ted_sticks := 10
def ted_rocks := 10
def bill_sticks := ted_sticks + 6
def bill_rocks := ted_rocks / 2

-- The condition of total objects tossed by Bill
def bill_total_objects := bill_sticks + bill_rocks

-- The theorem we want to prove
theorem bill_tossed_21_objects :
  bill_total_objects = 21 :=
  by
  sorry

end bill_tossed_21_objects_l34_34634


namespace banana_price_l34_34549

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price_l34_34549


namespace max_value_abs_diff_PQ_PR_l34_34502

-- Definitions for the points on the given curves
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Statement of the problem as a theorem
theorem max_value_abs_diff_PQ_PR (P Q R : ℝ × ℝ)
(hyp_P : hyperbola P.1 P.2)
(hyp_Q : circle1 Q.1 Q.2)
(hyp_R : circle2 R.1 R.2) :
  max (abs (dist P Q - dist P R)) = 10 :=
sorry

end max_value_abs_diff_PQ_PR_l34_34502


namespace coupon_probability_l34_34584

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l34_34584


namespace monomial_sum_l34_34378

theorem monomial_sum (m n : ℤ) (h1 : n = 2) (h2 : m + 2 = 1) : m + n = 1 := by
  sorry

end monomial_sum_l34_34378


namespace arithmetic_sequence_sum_and_mean_l34_34636

theorem arithmetic_sequence_sum_and_mean :
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  S = 121 ∧ (S / n) = 11 :=
by
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  have h1 : S = 121 := sorry
  have h2 : (S / n) = 11 := by
    rw [h1]
    exact sorry
  exact ⟨h1, h2⟩

end arithmetic_sequence_sum_and_mean_l34_34636


namespace linear_function_not_passing_second_quadrant_l34_34012

noncomputable def probability_not_passing_second_quadrant : ℚ :=
  let ks := {-2, -1, 1, 2, 3}
  let bs := {-2, -1, 1, 2, 3}
  let total_pairs := 5 * 4 -- Total pairs (5 choices for k, 4 remaining choices for b)
  let favorable_pairs := 3 * 2 -- Positive k (3 choices) and Negative b (2 choices)
  favorable_pairs / total_pairs

theorem linear_function_not_passing_second_quadrant :
  probability_not_passing_second_quadrant = 3 / 10 :=
sorry

end linear_function_not_passing_second_quadrant_l34_34012


namespace ball_arrangement_problem_l34_34326

-- Defining the problem statement and conditions
theorem ball_arrangement_problem : 
  (∃ (A : ℕ), 
    (∀ (b : Fin 6 → ℕ), 
      (b 0 = 1 ∨ b 1 = 1) ∧ (b 0 = 2 ∨ b 1 = 2) ∧ -- 1 adjacent to 2
      b 4 ≠ 5 ∧ b 4 ≠ 6 ∧                 -- 5 not adjacent to 6 condition
      b 5 ≠ 5 ∧ b 5 ≠ 6     -- Add all other necessary conditions for arrangement
    ) →
    A = 144)
:= sorry

end ball_arrangement_problem_l34_34326


namespace prob_red_or_blue_l34_34869

-- Total marbles and given probabilities
def total_marbles : ℕ := 120
def prob_white : ℚ := 1 / 4
def prob_green : ℚ := 1 / 3

-- Problem statement
theorem prob_red_or_blue : (1 - (prob_white + prob_green)) = 5 / 12 :=
by
  sorry

end prob_red_or_blue_l34_34869


namespace tan_315_eq_neg1_l34_34951

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34951


namespace prime_transformation_l34_34458

theorem prime_transformation (p : ℕ) (prime_p : Nat.Prime p) (h : p = 3) : ∃ q : ℕ, q = 13 * p + 2 ∧ Nat.Prime q :=
by
  use 41
  sorry

end prime_transformation_l34_34458


namespace system_of_equations_solution_l34_34347

theorem system_of_equations_solution (x y z : ℝ) :
  (4 * x^2 / (1 + 4 * x^2) = y ∧
   4 * y^2 / (1 + 4 * y^2) = z ∧
   4 * z^2 / (1 + 4 * z^2) = x) →
  ((x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) :=
by
  sorry

end system_of_equations_solution_l34_34347


namespace tan_315_eq_neg1_l34_34921

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34921


namespace six_digit_number_theorem_l34_34629

-- Define the problem conditions
def six_digit_number_condition (N : ℕ) (x : ℕ) : Prop :=
  N = 200000 + x ∧ N < 1000000 ∧ (10 * x + 2 = 3 * N)

-- Define the value of x
def value_of_x : ℕ := 85714

-- Main theorem to prove
theorem six_digit_number_theorem (N : ℕ) (x : ℕ) (h1 : x = value_of_x) :
  six_digit_number_condition N x → N = 285714 :=
by
  intros h
  sorry

end six_digit_number_theorem_l34_34629


namespace interest_rate_difference_correct_l34_34631

noncomputable def interest_rate_difference (P r R T : ℝ) :=
  let I := P * r * T
  let I' := P * R * T
  (I' - I) = 140

theorem interest_rate_difference_correct:
  ∀ (P r R T : ℝ),
  P = 1000 ∧ T = 7 ∧ interest_rate_difference P r R T →
  (R - r) = 0.02 :=
by
  intros P r R T h
  sorry

end interest_rate_difference_correct_l34_34631


namespace measure_limsup_measure_liminf_l34_34830

variables {Ω : Type*} {μ : MeasureTheory.Measure Ω} {𝓕 : Set (Set Ω)}
          {A : Set Ω} {A_n : ℕ → Set Ω}

-- Part (a)
theorem measure_limsup (hμ : MeasureTheory.Measure Ω)
                       (h_in_mf : ∀ n, A_n n ∈ 𝓕)
                       (h_mon : ∀ n, A_n n ⊆ A_n (n + 1))
                       (h_union : A = ⋃ n, A_n n) :
  Tendsto (λ n, μ.measureOf (A_n n)) at_top (𝓝 (μ.measureOf A)) :=
sorry

-- Part (b)
theorem measure_liminf (hμ : MeasureTheory.Measure Ω)
                       (h_in_mf : ∀ n, A_n n ∈ 𝓕)
                       (h_mon_dec : ∀ n, A_n (n + 1) ⊆ A_n n)
                       (h_meas_finite : ∃ m, μ.measureOf (A_n m) < ⊤)
                       (h_inter : A = ⋂ n, A_n n) :
  Tendsto (λ n, μ.measureOf (A_n n)) at_top (𝓝 (μ.measureOf A)) :=
sorry

end measure_limsup_measure_liminf_l34_34830


namespace cat_food_more_than_dog_food_l34_34772

-- Define the number of packages and cans per package for cat food
def cat_food_packages : ℕ := 9
def cat_food_cans_per_package : ℕ := 10

-- Define the number of packages and cans per package for dog food
def dog_food_packages : ℕ := 7
def dog_food_cans_per_package : ℕ := 5

-- Total number of cans of cat food
def total_cat_food_cans : ℕ := cat_food_packages * cat_food_cans_per_package

-- Total number of cans of dog food
def total_dog_food_cans : ℕ := dog_food_packages * dog_food_cans_per_package

-- Prove the difference between the total cans of cat food and total cans of dog food
theorem cat_food_more_than_dog_food : total_cat_food_cans - total_dog_food_cans = 55 := by
  -- Provide the calculation results directly
  have h_cat : total_cat_food_cans = 90 := by rfl
  have h_dog : total_dog_food_cans = 35 := by rfl
  calc
    total_cat_food_cans - total_dog_food_cans = 90 - 35 := by rw [h_cat, h_dog]
    _ = 55 := rfl

end cat_food_more_than_dog_food_l34_34772


namespace smallest_possible_value_l34_34819

theorem smallest_possible_value (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  ∃ (v : ℝ), v = 
  (⌊ (2 * x + y) / z ⌋ + ⌊ (2 * y + z) / x ⌋ + ⌊ (2 * z + x) / y ⌋ + ⌊ (x + y + z) / (x + y) ⌋) ∧ 
  v = 4 :=
sorry

end smallest_possible_value_l34_34819


namespace tan_315_eq_neg1_l34_34926

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34926


namespace xy_equals_one_l34_34061

-- Define the mathematical theorem
theorem xy_equals_one (x y : ℝ) (h : x + y = 1 / x + 1 / y) (h₂ : x + y ≠ 0) : x * y = 1 := 
by
  sorry

end xy_equals_one_l34_34061


namespace eqn_solution_set_l34_34737

theorem eqn_solution_set :
  {x : ℝ | x ^ 2 - 1 = 0} = {-1, 1} := 
sorry

end eqn_solution_set_l34_34737


namespace rational_roots_of_quadratic_l34_34213

theorem rational_roots_of_quadratic (k : ℤ) (h : k > 0) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
by
  sorry

end rational_roots_of_quadratic_l34_34213


namespace tan_315_eq_neg_one_l34_34992

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34992


namespace price_reduction_correct_l34_34771

theorem price_reduction_correct :
  ∃ x : ℝ, (0.3 - x) * (500 + 4000 * x) = 180 ∧ x = 0.1 :=
by
  sorry

end price_reduction_correct_l34_34771


namespace fraction_division_l34_34086

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l34_34086


namespace infinite_nested_radical_solution_l34_34674

theorem infinite_nested_radical_solution (x : ℝ) (h : x = Real.sqrt (4 + 3 * x)) : x = 4 := 
by 
  sorry

end infinite_nested_radical_solution_l34_34674


namespace loss_percentage_is_17_l34_34419

noncomputable def loss_percentage (CP SP : ℝ) := ((CP - SP) / CP) * 100

theorem loss_percentage_is_17 :
  let CP : ℝ := 1500
  let SP : ℝ := 1245
  loss_percentage CP SP = 17 :=
by
  sorry

end loss_percentage_is_17_l34_34419


namespace expr_div_24_l34_34842

theorem expr_div_24 (a : ℤ) : 24 ∣ ((a^2 + 3*a + 1)^2 - 1) := 
by 
  sorry

end expr_div_24_l34_34842


namespace total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l34_34480

-- Initial conditions
def cost_4_oranges : Nat := 12
def cost_7_oranges : Nat := 28
def total_oranges : Nat := 28

-- Calculate the total cost for 28 oranges
theorem total_cost_28_oranges
  (x y : Nat) 
  (h1 : 4 * x + 7 * y = total_oranges) 
  (h2 : total_oranges = 28) 
  (h3 : x = 7) 
  (h4 : y = 0) : 
  7 * cost_4_oranges = 84 := 
by sorry

-- Calculate the average cost per orange
theorem avg_cost_per_orange 
  (total_cost : Nat) 
  (h1 : total_cost = 84)
  (h2 : total_oranges = 28) : 
  total_cost / total_oranges = 3 := 
by sorry

-- Calculate the cost for 6 oranges
theorem cost_6_oranges 
  (avg_cost : Nat)
  (h1 : avg_cost = 3)
  (n : Nat) 
  (h2 : n = 6) : 
  n * avg_cost = 18 := 
by sorry

end total_cost_28_oranges_avg_cost_per_orange_cost_6_oranges_l34_34480


namespace find_clique_of_size_6_l34_34673

-- Defining the conditions of the graph G
variable (G : SimpleGraph (Fin 12))

-- Condition: For any subset of 9 vertices, there exists a subset of 5 vertices that form a complete subgraph K_5.
def condition (s : Finset (Fin 12)) : Prop :=
  s.card = 9 → ∃ t : Finset (Fin 12), t ⊆ s ∧ t.card = 5 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v)

-- The theorem to prove given the conditions
theorem find_clique_of_size_6 (h : ∀ s : Finset (Fin 12), condition G s) : 
  ∃ t : Finset (Fin 12), t.card = 6 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v) :=
sorry

end find_clique_of_size_6_l34_34673


namespace mn_parallel_pq_l34_34394

-- Definitions based on the given conditions
variables {α : Type*} [euclidean_geometry α]
variables {A B C M N P Q O : α} -- Points of triangle and midpoints on the circumcircle

-- Midpoints of arcs without certain vertices
def is_midpoint_arc (O : α) (A B M : α) : Prop := ∃ (circ : circle α), circ.center = O ∧ circ.contains A ∧ circ.contains B ∧ M = midpoint (arc_of_circumcircle circ A B)

-- Define the problem statement
theorem mn_parallel_pq
  (hM : is_midpoint_arc O A B M) -- M is the midpoint of arc AB (arc not containing C)
  (hN : is_midpoint_arc O B C N) -- N is the midpoint of arc BC (arc not containing A)
  (hperp1 : X ⊥ Y) -- Other conditions (like perpendicularity) might be stated similarly
  : MN ∥ PQ := sorry

end mn_parallel_pq_l34_34394


namespace tan_315_eq_neg1_l34_34941

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34941


namespace polynomial_form_l34_34206

theorem polynomial_form (P : Polynomial ℝ) (hP : P ≠ 0)
    (h : ∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x)) :
    ∃ k : ℕ, k > 0 ∧ P = (X^2 + 1) ^ k :=
by sorry

end polynomial_form_l34_34206


namespace shape_of_theta_eq_c_l34_34210

-- Definitions based on given conditions
def azimuthal_angle (rho theta phi : ℝ) : ℝ := theta

-- The main theorem we want to prove
theorem shape_of_theta_eq_c (c : ℝ) : 
  (∀ ρ φ, azimuthal_angle ρ c φ = c) → 
  (∃ a b, is_plane (λ (ρ θ φ : ℝ), θ = c) a b) :=
sorry

end shape_of_theta_eq_c_l34_34210


namespace derivative_at_zero_l34_34399

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + x)

-- Statement of the problem: The derivative of f at 0 is 1
theorem derivative_at_zero : deriv f 0 = 1 := 
  sorry

end derivative_at_zero_l34_34399


namespace cyclic_sum_inequality_l34_34393

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (h_product : a * b * c = 1) :
  (a^6 / ((a - b) * (a - c)) + b^6 / ((b - c) * (b - a)) + c^6 / ((c - a) * (c - b)) > 15) := 
by sorry

end cyclic_sum_inequality_l34_34393


namespace range_of_a_l34_34097

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| < 4) ↔ (-5 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l34_34097


namespace incorrect_transformation_when_c_zero_l34_34016

theorem incorrect_transformation_when_c_zero {a b c : ℝ} (h : a * c = b * c) (hc : c = 0) : a ≠ b :=
by
  sorry

end incorrect_transformation_when_c_zero_l34_34016


namespace solve_equation_l34_34023

theorem solve_equation : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end solve_equation_l34_34023


namespace tan_315_eq_neg1_l34_34929

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34929


namespace tan_315_eq_neg1_l34_34945

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34945


namespace total_spots_l34_34633

variable (P : ℕ)
variable (Bill_spots : ℕ := 2 * P - 1)

-- Given conditions
variable (h1 : Bill_spots = 39)

-- Theorem we need to prove
theorem total_spots (P : ℕ) (Bill_spots : ℕ := 2 * P - 1) (h1 : Bill_spots = 39) : 
  Bill_spots + P = 59 := 
by
  sorry

end total_spots_l34_34633


namespace number_of_programs_correct_l34_34195

-- Conditions definition
def solo_segments := 5
def chorus_segments := 3

noncomputable def number_of_programs : ℕ :=
  let solo_permutations := Nat.factorial solo_segments
  let available_spaces := solo_segments + 1
  let chorus_placements := Nat.choose (available_spaces - 1) chorus_segments
  solo_permutations * chorus_placements

theorem number_of_programs_correct : number_of_programs = 7200 :=
  by
    -- The proof is omitted
    sorry

end number_of_programs_correct_l34_34195


namespace tan_315_eq_neg1_l34_34947

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34947


namespace tan_315_degree_l34_34970

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34970


namespace percentage_of_men_in_company_l34_34477

theorem percentage_of_men_in_company 
  (M W : ℝ) 
  (h1 : 0.60 * M + 0.35 * W = 50) 
  (h2 : M + W = 100) : 
  M = 60 :=
by
  sorry

end percentage_of_men_in_company_l34_34477


namespace a_n_strictly_monotonic_increasing_l34_34641

noncomputable def a_n (n : ℕ) : ℝ := 
  2 * ((1 + 1 / (n : ℝ)) ^ (2 * n + 1)) / (((1 + 1 / (n : ℝ)) ^ n) + ((1 + 1 / (n : ℝ)) ^ (n + 1)))

theorem a_n_strictly_monotonic_increasing : ∀ n : ℕ, a_n (n + 1) > a_n n :=
sorry

end a_n_strictly_monotonic_increasing_l34_34641


namespace expression_meaningful_l34_34447

theorem expression_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by
  sorry

end expression_meaningful_l34_34447


namespace train_length_l34_34182

variable (L_train : ℝ)
variable (speed_kmhr : ℝ := 45)
variable (time_seconds : ℝ := 30)
variable (bridge_length_m : ℝ := 275)
variable (train_speed_ms : ℝ := speed_kmhr * (1000 / 3600))
variable (total_distance : ℝ := train_speed_ms * time_seconds)

theorem train_length
  (h_total : total_distance = L_train + bridge_length_m) :
  L_train = 100 :=
by 
  sorry

end train_length_l34_34182


namespace quadratic_real_roots_condition_l34_34360

theorem quadratic_real_roots_condition (a b c : ℝ) (q : b^2 - 4 * a * c ≥ 0) (h : a ≠ 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ a ≠ 0) ↔ ((∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) ∨ (∃ x : ℝ, a * x ^ 2 + b * x + c = 0)) :=
by
  sorry

end quadratic_real_roots_condition_l34_34360


namespace problem1_problem2_problem3_problem4_l34_34912

-- (1) Prove (1 + sqrt 3) * (2 - sqrt 3) = -1 + sqrt 3
theorem problem1 : (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 :=
by sorry

-- (2) Prove (sqrt 36 * sqrt 12) / sqrt 3 = 12
theorem problem2 : (Real.sqrt 36 * Real.sqrt 12) / Real.sqrt 3 = 12 :=
by sorry

-- (3) Prove sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4
theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = (5 * Real.sqrt 2) / 4 :=
by sorry

-- (4) Prove (3 * sqrt 18 + (1 / 5) * sqrt 50 - 4 * sqrt (1 / 2)) / sqrt 32 = 2
theorem problem4 : (3 * Real.sqrt 18 + (1 / 5) * Real.sqrt 50 - 4 * Real.sqrt (1 / 2)) / Real.sqrt 32 = 2 :=
by sorry

end problem1_problem2_problem3_problem4_l34_34912


namespace investment_value_l34_34488

-- Define the compound interest calculation
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Given values
def P : ℝ := 8000
def r : ℝ := 0.05
def n : ℕ := 7

-- The theorem statement in Lean 4
theorem investment_value :
  round (compound_interest P r n) = 11257 :=
by
  sorry

end investment_value_l34_34488


namespace min_value_a1_l34_34847

noncomputable def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = r * seq n

theorem min_value_a1 (a1 a2 : ℕ) (seq : ℕ → ℕ)
  (h1 : is_geometric_sequence seq)
  (h2 : ∀ n : ℕ, seq n > 0)
  (h3 : seq 20 + seq 21 = 20^21) :
  ∃ a b : ℕ, a1 = 2^a * 5^b ∧ a + b = 24 :=
sorry

end min_value_a1_l34_34847


namespace tan_315_eq_neg1_l34_34919

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34919


namespace bud_age_is_eight_l34_34779

def uncle_age : ℕ := 24

def bud_age (uncle_age : ℕ) : ℕ := uncle_age / 3

theorem bud_age_is_eight : bud_age uncle_age = 8 :=
by
  sorry

end bud_age_is_eight_l34_34779


namespace probability_of_missing_coupons_l34_34580

noncomputable def calc_probability : ℚ :=
  (nat.choose 11 3) / (nat.choose 17 9)

theorem probability_of_missing_coupons :
  calc_probability = (3 / 442 : ℚ) :=
by
  sorry

end probability_of_missing_coupons_l34_34580


namespace ratio_of_volumes_l34_34333

-- Definitions based on given conditions
def V1 : ℝ := sorry -- Volume of the first vessel
def V2 : ℝ := sorry -- Volume of the second vessel

-- Given condition
def condition : Prop := (3 / 4) * V1 = (5 / 8) * V2

-- The theorem to prove the ratio V1 / V2 is 5 / 6
theorem ratio_of_volumes (h : condition) : V1 / V2 = 5 / 6 :=
sorry

end ratio_of_volumes_l34_34333


namespace water_consumed_l34_34320

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end water_consumed_l34_34320


namespace two_students_cover_all_questions_l34_34471

-- Define the main properties
variables (students : Finset ℕ) (questions : Finset ℕ)
variable (solves : ℕ → ℕ → Prop)

-- Assume the given conditions
axiom total_students : students.card = 8
axiom total_questions : questions.card = 8
axiom each_question_solved_by_min_5_students : ∀ q, q ∈ questions → 
(∃ student_set : Finset ℕ, student_set.card ≥ 5 ∧ ∀ s ∈ student_set, solves s q)

-- The theorem to be proven
theorem two_students_cover_all_questions :
  ∃ s1 s2 : ℕ, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ q ∈ questions, solves s1 q ∨ solves s2 q :=
sorry -- proof to be written

end two_students_cover_all_questions_l34_34471


namespace rows_seating_exactly_10_people_exists_l34_34240

theorem rows_seating_exactly_10_people_exists :
  ∃ y x : ℕ, 73 = 10 * y + 9 * x ∧ (73 - 10 * y) % 9 = 0 := 
sorry

end rows_seating_exactly_10_people_exists_l34_34240


namespace f_value_l34_34263

def B := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) : ℝ := sorry

axiom f_property : ∀ x ∈ B, f x + f (2 - (1 / x)) = Real.log (abs (x ^ 2))

theorem f_value : f 2023 = Real.log 2023 :=
by
  sorry

end f_value_l34_34263


namespace opposite_of_neg_two_l34_34727

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l34_34727


namespace mitzi_amount_brought_l34_34266

-- Define the amounts spent on different items
def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23

-- Define the amount of money left
def amount_left : ℕ := 9

-- Define the total amount spent
def total_spent : ℕ :=
  ticket_cost + food_cost + tshirt_cost

-- Define the total amount brought to the amusement park
def amount_brought : ℕ :=
  total_spent + amount_left

-- Prove that the amount of money Mitzi brought to the amusement park is 75
theorem mitzi_amount_brought : amount_brought = 75 := by
  sorry

end mitzi_amount_brought_l34_34266


namespace children_count_125_l34_34351

def numberOfChildren (a : ℕ) : Prop :=
  a % 8 = 5 ∧ a % 10 = 7 ∧ 100 ≤ a ∧ a ≤ 150

theorem children_count_125 : ∃ a : ℕ, numberOfChildren a ∧ a = 125 := by
  use 125
  unfold numberOfChildren
  apply And.intro
  apply And.intro
  · norm_num
  · norm_num
  · split
  repeat {norm_num}
  sorry

end children_count_125_l34_34351


namespace find_k_l34_34215

-- Defining the vectors and the condition for parallelism
def vector_a := (2, 1)
def vector_b (k : ℝ) := (k, 3)

def vector_parallel_condition (k : ℝ) : Prop :=
  let a2b := (2 + 2 * k, 7)
  let a2nb := (4 - k, -1)
  (2 + 2 * k) * (-1) = 7 * (4 - k)

theorem find_k (k : ℝ) (h : vector_parallel_condition k) : k = 6 :=
by
  sorry

end find_k_l34_34215


namespace candy_cost_l34_34407

-- Definitions and assumptions from problem conditions
def cents_per_page := 1
def pages_per_book := 150
def books_read := 12
def leftover_cents := 300  -- $3 in cents

-- Total pages read
def total_pages_read := pages_per_book * books_read

-- Total earnings in cents
def total_cents_earned := total_pages_read * cents_per_page

-- Cost of the candy in cents
def candy_cost_cents := total_cents_earned - leftover_cents

-- Theorem statement
theorem candy_cost : candy_cost_cents = 1500 := 
  by 
    -- proof goes here
    sorry

end candy_cost_l34_34407


namespace isosceles_triangle_angles_l34_34385

theorem isosceles_triangle_angles (a b : ℝ) (h₁ : a = 80 ∨ b = 80) (h₂ : a + b + c = 180) (h_iso : a = b ∨ a = c ∨ b = c) :
  (a = 80 ∧ b = 20 ∧ c = 80)
  ∨ (a = 80 ∧ b = 80 ∧ c = 20)
  ∨ (a = 50 ∧ b = 50 ∧ c = 80) :=
by sorry

end isosceles_triangle_angles_l34_34385


namespace sum_of_three_numbers_is_98_l34_34288

variable (A B C : ℕ) (h_ratio1 : A = 2 * (B / 3)) (h_ratio2 : B = 30) (h_ratio3 : B = 5 * (C / 8))

theorem sum_of_three_numbers_is_98 : A + B + C = 98 := by
  sorry

end sum_of_three_numbers_is_98_l34_34288


namespace probability_of_selecting_GEARS_letter_l34_34200

def bag : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'S']
def target_word : List Char := ['G', 'E', 'A', 'R', 'S']

theorem probability_of_selecting_GEARS_letter :
  (6 : ℚ) / 8 = 3 / 4 :=
by
  sorry

end probability_of_selecting_GEARS_letter_l34_34200


namespace min_triangles_bound_four_color_edge_coloring_exists_l34_34658

noncomputable theory

def P : Set Point := {P_1, P_2, ..., P_1994}

structure PointsGroups :=
  (groups : List (List Point))
  (group_sizes : ∀ g ∈ groups, 3 ≤ g.length)
  (partition_sum : groups.foldr (λ g acc, g.length + acc) 0 = 1994)
  (partition_unique : (∀ p1 p2 ∈ P, (∃ g, g ∈ groups ∧ p1 ∈ g ∧ p2 ∈ g) ⊕ (∀ g, g ∈ groups → p1 ∈ g → p2 ∉ g)))

def graph (pg : PointsGroups) : Graph Point :=
{ V := P,
  E := { (p1, p2) | ∃ g ∈ pg.groups, p1 ∈ g ∧ p2 ∈ g ∧ p1 ≠ p2 } }

def min_triangles (G : Graph Point) : ℕ :=
∑ g in G.V.groups, if g.length ≥ 3 then (g.length.choose 3) else 0

theorem min_triangles_bound (P : Set Point) (pg : PointsGroups) :
  ∃ G : Graph Point, min_triangles (graph pg) = 168544 := sorry

theorem four_color_edge_coloring_exists (P : Set Point) (pg : PointsGroups) (G : Graph Point)
  (hG : min_triangles (graph pg) = 168544) :
  ∃ (col : G.E → Fin 4), ∀ (p1 p2 p3 : Point), 
    (G.adj p1 p2 ∧ G.adj p2 p3 ∧ G.adj p1 p3) → (col (p1, p2) ≠ col (p2, p3) ∨ col (p2, p3) ≠ col (p1, p3) ∨ col (p1, p2) ≠ col (p1, p3)) := sorry

end min_triangles_bound_four_color_edge_coloring_exists_l34_34658


namespace available_milk_for_me_l34_34089

def initial_milk_litres : ℝ := 1
def myeongseok_milk_litres : ℝ := 0.1
def mingu_milk_litres : ℝ := myeongseok_milk_litres + 0.2
def minjae_milk_litres : ℝ := 0.3

theorem available_milk_for_me :
  initial_milk_litres - (myeongseok_milk_litres + mingu_milk_litres + minjae_milk_litres) = 0.3 :=
by sorry

end available_milk_for_me_l34_34089


namespace brownies_maximum_l34_34078

theorem brownies_maximum (m n : ℕ) (h1 : (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)) :
  m * n ≤ 144 :=
sorry

end brownies_maximum_l34_34078


namespace expected_ties_approx_l34_34589

noncomputable def expected_number_of_ties : ℚ :=
  ∑ k in Finset.range 5 + 1, Nat.choose (2 * k) k / (2^(2 * k))

theorem expected_ties_approx :
  (expected_number_of_ties : ℚ) ≈ 1.707 :=
by
  sorry

end expected_ties_approx_l34_34589


namespace peculiar_looking_less_than_500_l34_34340

def is_composite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def peculiar_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 7 = 0 ∨ n % 11 = 0)

theorem peculiar_looking_less_than_500 :
  ∃ n, n = 33 ∧ ∀ k, k < 500 → peculiar_looking k → k = n :=
sorry

end peculiar_looking_less_than_500_l34_34340


namespace problem_l34_34402

noncomputable def F (x : ℝ) : ℝ :=
  (1 + x^2 - x^3) / (2 * x * (1 - x))

theorem problem (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  F x + F ((x - 1) / x) = 1 + x :=
by
  sorry

end problem_l34_34402


namespace percent_of_x_is_y_l34_34895

-- Given the condition
def condition (x y : ℝ) : Prop :=
  0.70 * (x - y) = 0.30 * (x + y)

-- Prove y / x = 0.40
theorem percent_of_x_is_y (x y : ℝ) (h : condition x y) : y / x = 0.40 :=
by
  sorry

end percent_of_x_is_y_l34_34895


namespace larger_number_is_299_l34_34467

theorem larger_number_is_299 {a b : ℕ} (hcf : Nat.gcd a b = 23) (lcm_factors : ∃ k1 k2 : ℕ, Nat.lcm a b = 23 * k1 * k2 ∧ k1 = 12 ∧ k2 = 13) :
  max a b = 299 :=
by
  sorry

end larger_number_is_299_l34_34467


namespace minimum_period_f_l34_34282

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x / 2 + Real.pi / 4)

theorem minimum_period_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end minimum_period_f_l34_34282


namespace g_diff_l34_34230

def g (n : ℤ) : ℤ := (1 / 4 : ℤ) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_diff (r : ℤ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_diff_l34_34230


namespace jessica_can_mail_letter_l34_34389

-- Define the constants
def paper_weight := 1/5 -- each piece of paper weighs 1/5 ounce
def envelope_weight := 2/5 -- envelope weighs 2/5 ounce
def num_papers := 8

-- Calculate the total weight
def total_weight := num_papers * paper_weight + envelope_weight

-- Define stamping rates
def international_rate := 2 -- $2 per ounce internationally

-- Calculate the required postage
def required_postage := total_weight * international_rate

-- Define the available stamp values
inductive Stamp
| one_dollar : Stamp
| fifty_cents : Stamp

-- Function to calculate the total value of a given stamp combination
def stamp_value : List Stamp → ℝ
| [] => 0
| (Stamp.one_dollar :: rest) => 1 + stamp_value rest
| (Stamp.fifty_cents :: rest) => 0.5 + stamp_value rest

-- State the theorem to be proved
theorem jessica_can_mail_letter :
  ∃ stamps : List Stamp, stamp_value stamps = required_postage := by
sorry

end jessica_can_mail_letter_l34_34389


namespace value_of_r_l34_34688

theorem value_of_r (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1
  let r := 4^s - s
  r = 16377 := by
  let s := 2^3 - 1
  let r := 4^s - s
  sorry

end value_of_r_l34_34688


namespace students_did_not_eat_2_l34_34267

-- Define the given conditions
def total_students : ℕ := 20
def total_crackers_eaten : ℕ := 180
def crackers_per_pack : ℕ := 10

-- Calculate the number of packs eaten
def packs_eaten : ℕ := total_crackers_eaten / crackers_per_pack

-- Calculate the number of students who did not eat their animal crackers
def students_who_did_not_eat : ℕ := total_students - packs_eaten

-- Prove that the number of students who did not eat their animal crackers is 2
theorem students_did_not_eat_2 :
  students_who_did_not_eat = 2 :=
  by
    sorry

end students_did_not_eat_2_l34_34267


namespace opposite_of_two_l34_34445

theorem opposite_of_two : ∃ x : ℤ, 2 + x = 0 ∧ x = -2 :=
by
  exists  -2
  split
  . simp
  . refl

end opposite_of_two_l34_34445


namespace opposite_of_neg_two_l34_34733

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l34_34733


namespace condition_is_necessary_but_not_sufficient_l34_34166

noncomputable def sequence_satisfies_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 7 = 2 * a 5

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem condition_is_necessary_but_not_sufficient (a : ℕ → ℤ) :
  (sequence_satisfies_condition a ∧ (¬ arithmetic_sequence a)) ∨
  (arithmetic_sequence a → sequence_satisfies_condition a) :=
sorry

end condition_is_necessary_but_not_sufficient_l34_34166


namespace opposite_of_neg_two_l34_34732

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l34_34732


namespace rectangle_area_inscribed_circle_l34_34174

theorem rectangle_area_inscribed_circle {r w l : ℕ} (h1 : r = 7) (h2 : w = 2 * r) (h3 : l = 3 * w) : l * w = 588 :=
by 
  -- The proof details are omitted as per instructions.
  sorry

end rectangle_area_inscribed_circle_l34_34174


namespace find_natural_numbers_l34_34642

theorem find_natural_numbers (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by {
  sorry
}

end find_natural_numbers_l34_34642


namespace side_length_of_square_l34_34427

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34427


namespace find_x2_plus_y2_l34_34105

open Real

theorem find_x2_plus_y2 (x y : ℝ) 
  (h1 : (x + y) ^ 4 + (x - y) ^ 4 = 4112)
  (h2 : x ^ 2 - y ^ 2 = 16) :
  x ^ 2 + y ^ 2 = 34 := 
sorry

end find_x2_plus_y2_l34_34105


namespace bobby_consumption_l34_34778

theorem bobby_consumption :
  let initial_candy := 28
  let additional_candy_portion := 3/4 * 42
  let chocolate_portion := 1/2 * 63
  initial_candy + additional_candy_portion + chocolate_portion = 91 := 
by {
  let initial_candy : ℝ := 28
  let additional_candy_portion : ℝ := 3/4 * 42
  let chocolate_portion : ℝ := 1/2 * 63
  sorry
}

end bobby_consumption_l34_34778


namespace Roja_speed_is_8_l34_34562

def Pooja_speed : ℝ := 3
def time_in_hours : ℝ := 4
def distance_between_them : ℝ := 44

theorem Roja_speed_is_8 :
  ∃ R : ℝ, R + Pooja_speed = (distance_between_them / time_in_hours) ∧ R = 8 :=
by
  sorry

end Roja_speed_is_8_l34_34562


namespace triangle_area_l34_34248

theorem triangle_area (area_WXYZ : ℝ) (side_small_squares : ℝ) 
  (AB_eq_AC : (AB = AC)) (A_on_center : (A = O)) :
  area_WXYZ = 64 ∧ side_small_squares = 2 →
  ∃ (area_triangle_ABC : ℝ), area_triangle_ABC = 8 :=
by
  intros h
  sorry

end triangle_area_l34_34248


namespace count_increasing_8digit_no_repeat_more_than_twice_l34_34536

theorem count_increasing_8digit_no_repeat_more_than_twice : 
  let M := (Nat.choose 16 8) - 9 * (Nat.choose 13 5) + (Nat.choose 9 2) * (Nat.choose 10 2) in
  M = 1907 :=
by
  let M := (Nat.choose 16 8) - 9 * (Nat.choose 13 5) + (Nat.choose 9 2) * (Nat.choose 10 2)
  have hM : M = 1907 := sorry
  exact hM

end count_increasing_8digit_no_repeat_more_than_twice_l34_34536


namespace num_four_letter_initials_l34_34807

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l34_34807


namespace tan_315_proof_l34_34957

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34957


namespace number_of_ping_pong_balls_l34_34900

def sales_tax_rate : ℝ := 0.16

def total_cost_with_tax (B x : ℝ) : ℝ := B * x * (1 + sales_tax_rate)

def total_cost_without_tax (B x : ℝ) : ℝ := (B + 3) * x

theorem number_of_ping_pong_balls
  (B x : ℝ) (h₁ : total_cost_with_tax B x = total_cost_without_tax B x) :
  B = 18.75 := 
sorry

end number_of_ping_pong_balls_l34_34900


namespace tan_315_proof_l34_34958

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34958


namespace Anne_mom_toothpaste_usage_l34_34450

theorem Anne_mom_toothpaste_usage
  (total_toothpaste : ℕ)
  (dad_usage_per_brush : ℕ)
  (sibling_usage_per_brush : ℕ)
  (num_brushes_per_day : ℕ)
  (total_days : ℕ)
  (total_toothpaste_used : ℕ)
  (M : ℕ)
  (family_use_model : total_toothpaste = total_toothpaste_used + 3 * num_brushes_per_day * M)
  (total_toothpaste_used_def : total_toothpaste_used = 5 * (dad_usage_per_brush * num_brushes_per_day + 2 * sibling_usage_per_brush * num_brushes_per_day))
  (given_values : total_toothpaste = 105 ∧ dad_usage_per_brush = 3 ∧ sibling_usage_per_brush = 1 ∧ num_brushes_per_day = 3 ∧ total_days = 5)
  : M = 2 := by
  sorry

end Anne_mom_toothpaste_usage_l34_34450


namespace num_four_letter_initials_l34_34806

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l34_34806


namespace new_person_weight_l34_34101

theorem new_person_weight
  (initial_avg_weight : ℝ := 57)
  (num_people : ℕ := 8)
  (weight_to_replace : ℝ := 55)
  (weight_increase_first : ℝ := 1.5)
  (weight_increase_second : ℝ := 2)
  (weight_increase_third : ℝ := 2.5)
  (weight_increase_fourth : ℝ := 3)
  (weight_increase_fifth : ℝ := 3.5)
  (weight_increase_sixth : ℝ := 4)
  (weight_increase_seventh : ℝ := 4.5) :
  ∃ x : ℝ, x = 67 :=
by
  sorry

end new_person_weight_l34_34101


namespace positive_difference_between_numbers_l34_34738

theorem positive_difference_between_numbers:
  ∃ x y : ℤ, x + y = 40 ∧ 3 * y - 4 * x = 7 ∧ |y - x| = 6 := by
  sorry

end positive_difference_between_numbers_l34_34738


namespace diamond_fifteen_two_l34_34283

def diamond (a b : ℤ) : ℤ := a + (a / (b + 1))

theorem diamond_fifteen_two : diamond 15 2 = 20 := 
by 
    sorry

end diamond_fifteen_two_l34_34283


namespace opposite_of_neg_two_l34_34726

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l34_34726


namespace tan_315_eq_neg1_l34_34952

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34952


namespace total_distance_traveled_l34_34115

theorem total_distance_traveled :
  let time1 := 3  -- hours
  let speed1 := 70  -- km/h
  let time2 := 4  -- hours
  let speed2 := 80  -- km/h
  let time3 := 3  -- hours
  let speed3 := 65  -- km/h
  let time4 := 2  -- hours
  let speed4 := 90  -- km/h
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  distance1 + distance2 + distance3 + distance4 = 905 :=
by
  sorry

end total_distance_traveled_l34_34115


namespace a_plus_b_l34_34068

open Real

-- given condition
def series_sum : ℝ := ∑' i : ℕ, (sin^2 (10 / 3^i * π / 180) / cos (30 / 3^i * π / 180))

-- main statement
theorem a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (1 : ℝ) / (a + sqrt b) = series_sum) : a + b = 15 :=
sorry

end a_plus_b_l34_34068


namespace max_mondays_in_51_days_l34_34592

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l34_34592


namespace tan_315_eq_neg1_l34_34942

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34942


namespace tom_total_payment_l34_34005

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l34_34005


namespace tan_315_eq_neg_one_l34_34936

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34936


namespace prime_iff_satisfies_condition_l34_34561

def satisfies_condition (n : ℕ) : Prop :=
  if n = 2 then True
  else if 2 < n then ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬ (k ∣ n)
  else False

theorem prime_iff_satisfies_condition (n : ℕ) : Prime n ↔ satisfies_condition n := by
  sorry

end prime_iff_satisfies_condition_l34_34561


namespace avg_diff_l34_34095

theorem avg_diff (a x c : ℝ) (h1 : (a + x) / 2 = 40) (h2 : (x + c) / 2 = 60) :
  c - a = 40 :=
by
  sorry

end avg_diff_l34_34095


namespace solution_set_of_inequality_l34_34526

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 6

theorem solution_set_of_inequality (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ (-1/3 : ℝ) < m ∧ m < 3 :=
by 
  sorry

end solution_set_of_inequality_l34_34526


namespace first_applicant_earnings_l34_34485

def first_applicant_salary : ℕ := 42000
def first_applicant_training_cost_per_month : ℕ := 1200
def first_applicant_training_months : ℕ := 3
def second_applicant_salary : ℕ := 45000
def second_applicant_bonus_percentage : ℕ := 1
def company_earnings_from_second_applicant : ℕ := 92000
def earnings_difference : ℕ := 850

theorem first_applicant_earnings 
  (salary1 : first_applicant_salary = 42000)
  (train_cost_per_month : first_applicant_training_cost_per_month = 1200)
  (train_months : first_applicant_training_months = 3)
  (salary2 : second_applicant_salary = 45000)
  (bonus_percentage : second_applicant_bonus_percentage = 1)
  (earnings2 : company_earnings_from_second_applicant = 92000)
  (earning_diff : earnings_difference = 850) :
  (company_earnings_from_second_applicant - (second_applicant_salary + (second_applicant_salary * second_applicant_bonus_percentage / 100)) - earnings_difference) = 45700 := 
by 
  sorry

end first_applicant_earnings_l34_34485


namespace sum_last_two_digits_9_pow_23_plus_11_pow_23_l34_34014

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end sum_last_two_digits_9_pow_23_plus_11_pow_23_l34_34014


namespace canoe_kayak_ratio_l34_34875

-- Define the number of canoes and kayaks
variables (c k : ℕ)

-- Define the conditions
def rental_cost_eq : Prop := 15 * c + 18 * k = 405
def canoe_more_kayak_eq : Prop := c = k + 5

-- Statement to prove
theorem canoe_kayak_ratio (h1 : rental_cost_eq c k) (h2 : canoe_more_kayak_eq c k) : c / k = 3 / 2 :=
by sorry

end canoe_kayak_ratio_l34_34875


namespace tan_315_eq_neg_one_l34_34938

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34938


namespace cos_double_angle_l34_34662

theorem cos_double_angle (α : ℝ) (h : ‖(Real.cos α, Real.sqrt 2 / 2)‖ = Real.sqrt 3 / 2) : Real.cos (2 * α) = -1 / 2 :=
sorry

end cos_double_angle_l34_34662


namespace find_initial_girls_l34_34654

variable (b g : ℕ)

theorem find_initial_girls 
  (h1 : 3 * (g - 18) = b)
  (h2 : 4 * (b - 36) = g - 18) :
  g = 31 := 
by
  sorry

end find_initial_girls_l34_34654


namespace expression_equals_a5_l34_34882

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end expression_equals_a5_l34_34882


namespace sin_tan_relation_l34_34357

theorem sin_tan_relation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -(2 / 5) := 
sorry

end sin_tan_relation_l34_34357


namespace expected_value_xi_probability_event_A_l34_34510

-- Define basic conditions
def probability_of_success : ℝ := 1 / 3
def number_of_experiments : ℕ := 4
def success_or_failure (results : Fin number_of_experiments → Bool) : ℕ :=
  (results.to_list.filter id).length

-- Define the random variable ξ
def xi (results : Fin number_of_experiments → Bool) : ℕ :=
  abs ((success_or_failure results) - (number_of_experiments - (success_or_failure results)))

-- Event A: "The solution set for the inequality ξ x^2 - ξ x + 1 > 0 is the set of all real numbers ℝ"
def satisfies_event_A (ξ : ℕ) : Prop :=
  match ξ with
  | 0 => true
  | 2 => true
  | 4 => false
  | _ => false

-- Expected value of random variable ξ
theorem expected_value_xi (results : Fin number_of_experiments → Bool) : 
  ∑ (xi_value : ℕ) in {0, 2, 4}, xi_value * P(xi = xi_value) = 148 / 81 := sorry

-- Probability of event A
theorem probability_event_A : 
  P(satisfies_event_A ξ) = 64 / 81 := sorry

end expected_value_xi_probability_event_A_l34_34510


namespace find_x_set_l34_34879

theorem find_x_set (a : ℝ) (h : 0 < a ∧ a < 1) : 
  {x : ℝ | a ^ (x + 3) > a ^ (2 * x)} = {x : ℝ | x > 3} :=
sorry

end find_x_set_l34_34879


namespace solve_quadratic_equation_l34_34845

theorem solve_quadratic_equation:
  (∀ x : ℝ, (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 →
    x = ( -17 + Real.sqrt 569) / 4 ∨ x = ( -17 - Real.sqrt 569) / 4) :=
by
  sorry

end solve_quadratic_equation_l34_34845


namespace percentage_of_passengers_in_first_class_l34_34697

theorem percentage_of_passengers_in_first_class (total_passengers : ℕ) (percentage_female : ℝ) (females_coach : ℕ) 
  (males_perc_first_class : ℝ) (Perc_first_class : ℝ) : 
  total_passengers = 120 → percentage_female = 0.45 → females_coach = 46 → males_perc_first_class = (1/3) → 
  Perc_first_class = 10 := by
  sorry

end percentage_of_passengers_in_first_class_l34_34697


namespace tetrahedron_probability_l34_34765

theorem tetrahedron_probability :
  let faces := {0, 1, 2, 3}
  let event_A := {p : ℕ × ℕ | p.1 ∈ faces ∧ p.2 ∈ faces ∧ p.1^2 + p.2^2 ≤ 4}
  ∑ b in event_A, 1 = 6 → (6: ℝ) / (16: ℝ) = (3: ℝ) / (8: ℝ) := 
by
  sorry

end tetrahedron_probability_l34_34765


namespace num_triangles_with_longest_side_6_l34_34376

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem num_triangles_with_longest_side_6 : 
  ∃ (count : ℕ), (∃ (sides : finset (ℕ × ℕ × ℕ)), 
  ∀ (a b c ∈ sides), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ max a (max b c) = 6 ∧ (is_triangle a b c)) ∧ 
  count = 4 := 
sorry

end num_triangles_with_longest_side_6_l34_34376


namespace tan_315_eq_neg1_l34_34927

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34927


namespace max_value_of_function_l34_34281

noncomputable def function_y (x : ℝ) : ℝ := x + Real.sin x

theorem max_value_of_function : 
  ∀ (a b : ℝ), a = 0 → b = Real.pi → 
  (∀ x : ℝ, x ∈ Set.Icc a b → x + Real.sin x ≤ Real.pi) :=
by
  intros a b ha hb x hx
  sorry

end max_value_of_function_l34_34281


namespace sum_cubes_identity_l34_34821

theorem sum_cubes_identity (x y z : ℝ) (h1 : x + y + z = 10) (h2 : xy + yz + zx = 20) :
    x^3 + y^3 + z^3 - 3 * x * y * z = 400 := by
  sorry

end sum_cubes_identity_l34_34821


namespace one_fourth_div_one_eighth_l34_34085

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l34_34085


namespace domain_of_function_l34_34440

theorem domain_of_function (x : ℝ) :
  (x^2 - 5*x + 6 ≥ 0) → (x ≠ 2) → (x < 2 ∨ x ≥ 3) :=
by
  intros h1 h2
  sorry

end domain_of_function_l34_34440


namespace height_of_stack_of_pots_l34_34770

-- Definitions corresponding to problem conditions
def pot_thickness : ℕ := 1

def top_pot_diameter : ℕ := 16

def bottom_pot_diameter : ℕ := 4

def diameter_decrement : ℕ := 2

-- Number of pots calculation
def num_pots : ℕ := (top_pot_diameter - bottom_pot_diameter) / diameter_decrement + 1

-- The total vertical distance from the bottom of the lowest pot to the top of the highest pot
def total_vertical_distance : ℕ := 
  let inner_heights := num_pots * (top_pot_diameter - pot_thickness + bottom_pot_diameter - pot_thickness) / 2
  let total_thickness := num_pots * pot_thickness
  inner_heights + total_thickness

theorem height_of_stack_of_pots : total_vertical_distance = 65 := 
sorry

end height_of_stack_of_pots_l34_34770


namespace range_of_real_number_a_l34_34237

theorem range_of_real_number_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 1 = 0 → x = a) ↔ (a = 0 ∨ a ≥ 9/4) :=
sorry

end range_of_real_number_a_l34_34237


namespace expression_simplification_l34_34271

noncomputable def given_expression : ℝ :=
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1)))

noncomputable def expected_expression : ℝ :=
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3)

theorem expression_simplification :
  given_expression = expected_expression :=
by
  sorry

end expression_simplification_l34_34271


namespace find_m_l34_34876

theorem find_m (m : ℤ) (h₀ : 0 ≤ m) (h₁ : m < 31) (h₂ : 79453 % 31 = m) : m = 0 :=
by
  sorry

end find_m_l34_34876


namespace enclosed_area_l34_34198

theorem enclosed_area {x y : ℝ} (h : x^2 + y^2 = 2 * |x| + 2 * |y|) : ∃ (A : ℝ), A = 8 :=
sorry

end enclosed_area_l34_34198


namespace correct_answer_l34_34746

theorem correct_answer (a b c : ℝ) : a - (b + c) = a - b - c :=
by sorry

end correct_answer_l34_34746


namespace class_average_l34_34162

theorem class_average (n : ℕ) (h₁ : n = 100) (h₂ : 25 ≤ n) 
  (h₃ : 50 ≤ n) (h₄ : 25 * 80 + 50 * 65 + (n - 75) * 90 = 7500) :
  (25 * 80 + 50 * 65 + (n - 75) * 90) / n = 75 := 
by
  sorry

end class_average_l34_34162


namespace divides_y_l34_34544

theorem divides_y
  (x y : ℤ)
  (h1 : 2 * x + 1 ∣ 8 * y) : 
  2 * x + 1 ∣ y :=
sorry

end divides_y_l34_34544


namespace triangle_integer_solutions_l34_34286

theorem triangle_integer_solutions (x : ℕ) (h1 : 13 < x) (h2 : x < 43) : 
  ∃ (n : ℕ), n = 29 :=
by 
  sorry

end triangle_integer_solutions_l34_34286


namespace candies_per_friend_l34_34557

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : additional_candies = 4) (h3 : num_friends = 6) : 
  (initial_candies + additional_candies) / num_friends = 4 := 
by
  sorry

end candies_per_friend_l34_34557


namespace sports_parade_children_l34_34349

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l34_34349


namespace smallest_five_digit_int_equiv_5_mod_9_l34_34877

theorem smallest_five_digit_int_equiv_5_mod_9 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, (10000 ≤ m ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m :=
by
  use 10000
  sorry

end smallest_five_digit_int_equiv_5_mod_9_l34_34877


namespace number_of_girls_l34_34102

variable (b g d : ℕ)

-- Conditions
axiom boys_count : b = 1145
axiom difference : d = 510
axiom boys_equals_girls_plus_difference : b = g + d

-- Theorem to prove
theorem number_of_girls : g = 635 := by
  sorry

end number_of_girls_l34_34102


namespace perpendicular_lines_l34_34220

theorem perpendicular_lines (a : ℝ) :
  (∃ l₁ l₂ : ℝ, 2 * l₁ + l₂ + 1 = 0 ∧ l₁ + a * l₂ + 3 = 0 ∧ 2 * l₁ + 1 * l₂ + 1 * a = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l34_34220


namespace range_of_a_l34_34167

-- Define the assumptions and target proof
theorem range_of_a {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0)
  : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0 → a < 3 :=
by
  intro a h
  sorry

end range_of_a_l34_34167


namespace lines_parallel_l34_34802

/--
Given two lines represented by the equations \(2x + my - 2m + 4 = 0\) and \(mx + 2y - m + 2 = 0\), 
prove that the value of \(m\) that makes these two lines parallel is \(m = -2\).
-/
theorem lines_parallel (m : ℝ) : 
    (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0) ∧ (∀ x y : ℝ, m * x + 2 * y - m + 2 = 0) 
    → m = -2 :=
by
  sorry

end lines_parallel_l34_34802


namespace toaster_total_cost_l34_34256

theorem toaster_total_cost :
  let MSRP := 30
  let insurance_rate := 0.20
  let premium_upgrade := 7
  let recycling_fee := 5
  let tax_rate := 0.50

  -- Calculate costs
  let insurance_cost := insurance_rate * MSRP
  let total_insurance_cost := insurance_cost + premium_upgrade
  let cost_before_tax := MSRP + total_insurance_cost + recycling_fee
  let state_tax := tax_rate * cost_before_tax
  let total_cost := cost_before_tax + state_tax

  -- Total cost Jon must pay
  total_cost = 72 :=
by
  sorry

end toaster_total_cost_l34_34256


namespace fraction_division_l34_34087

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l34_34087


namespace factorize_expression_l34_34056

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l34_34056


namespace compound_interest_double_l34_34608

theorem compound_interest_double (t : ℕ) (r : ℝ) (n : ℕ) (P : ℝ) :
  r = 0.15 → n = 1 → (2 : ℝ) < (1 + r)^t → t ≥ 5 :=
by
  intros hr hn h
  sorry

end compound_interest_double_l34_34608


namespace tan_315_degree_l34_34973

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34973


namespace side_length_of_square_l34_34428

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l34_34428


namespace min_sum_ab_l34_34063

theorem min_sum_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = a + b + 3) : 
  a + b ≥ 6 := 
sorry

end min_sum_ab_l34_34063


namespace solve_problems_l34_34128

variable (initial_problems : ℕ) 
variable (additional_problems : ℕ)

theorem solve_problems
  (h1 : initial_problems = 12) 
  (h2 : additional_problems = 7) : 
  initial_problems + additional_problems = 19 := 
by 
  sorry

end solve_problems_l34_34128


namespace ad_parallel_mp_l34_34384

open EuclideanGeometry Real

theorem ad_parallel_mp
  (ABC : Triangle)
  (hABC_acute : ABC.isAcute)
  (hAB_lt_AC : ABC.side3 < ABC.side2)
  (I : Point := ABC.incenter)
  (D : Point := ABC.incirclePointBC)
  (E : Point := lineThrough AD intersectsCircleAt ABC.circumcircle at anotherPoint)
  (M : Point := midPoint of ABC.side1)
  (N : Point := midPoint of arc(ABC.angle))
  (P : Point := lineThrough (N, E) intersectsCircleAtBIC ABC.circumcircle anotherPoint) :
  Parallel AD MP := sorry

end ad_parallel_mp_l34_34384


namespace arithmetic_seq_a3_value_l34_34386

-- Given the arithmetic sequence {a_n}, where
-- a_1 + a_2 + a_3 + a_4 + a_5 = 20
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_seq_a3_value {a : ℕ → ℝ}
    (h_seq : arithmetic_seq a)
    (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
by
  sorry

end arithmetic_seq_a3_value_l34_34386


namespace new_boarders_l34_34284

theorem new_boarders (init_boarders : ℕ) (init_day_students : ℕ) (ratio_b : ℕ) (ratio_d : ℕ) (ratio_new_b : ℕ) (ratio_new_d : ℕ) (x : ℕ) :
    init_boarders = 240 →
    ratio_b = 8 →
    ratio_d = 17 →
    ratio_new_b = 3 →
    ratio_new_d = 7 →
    init_day_students = (init_boarders * ratio_d) / ratio_b →
    (ratio_new_b * init_day_students) = ratio_new_d * (init_boarders + x) →
    x = 21 :=
by sorry

end new_boarders_l34_34284


namespace profits_to_revenues_ratio_l34_34528

theorem profits_to_revenues_ratio (R P: ℝ) 
    (rev_2009: R_2009 = 0.8 * R) 
    (profit_2009_rev_2009: P_2009 = 0.2 * R_2009)
    (profit_2009: P_2009 = 1.6 * P):
    (P / R) * 100 = 10 :=
by
  sorry

end profits_to_revenues_ratio_l34_34528


namespace sticker_price_l34_34410

theorem sticker_price (y : ℝ) (h1 : ∀ (p : ℝ), p = 0.8 * y - 60 → p ≤ y)
  (h2 : ∀ (q : ℝ), q = 0.7 * y → q ≤ y)
  (h3 : (0.8 * y - 60) + 20 = 0.7 * y) :
  y = 400 :=
by
  sorry

end sticker_price_l34_34410


namespace solutions_eq1_solutions_eq2_l34_34137

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l34_34137


namespace kenneth_money_left_l34_34682

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l34_34682


namespace opposite_of_neg_two_l34_34725

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l34_34725


namespace power_mod_equiv_l34_34403

-- Define the main theorem
theorem power_mod_equiv {a n k : ℕ} (h₁ : a ≥ 2) (h₂ : n ≥ 1) :
  (a^k ≡ 1 [MOD (a^n - 1)]) ↔ (k % n = 0) :=
by sorry

end power_mod_equiv_l34_34403


namespace no_n_satisfies_mod_5_l34_34643

theorem no_n_satisfies_mod_5 (n : ℤ) : (n^3 + 2*n - 1) % 5 ≠ 0 :=
by
  sorry

end no_n_satisfies_mod_5_l34_34643


namespace hundred_div_point_two_five_eq_four_hundred_l34_34344

theorem hundred_div_point_two_five_eq_four_hundred : 100 / 0.25 = 400 := by
  sorry

end hundred_div_point_two_five_eq_four_hundred_l34_34344


namespace tan_315_degree_l34_34974

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34974


namespace sports_parade_children_l34_34350

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l34_34350


namespace maggie_earnings_correct_l34_34114

def subscriptions_sold_to_parents : ℕ := 4
def subscriptions_sold_to_grandfather : ℕ := 1
def subscriptions_sold_to_next_door_neighbor : ℕ := 2
def subscriptions_sold_to_another_neighbor : ℕ := 2 * subscriptions_sold_to_next_door_neighbor
def price_per_subscription : ℕ := 5
def family_bonus_per_subscription : ℕ := 2
def neighbor_bonus_per_subscription : ℕ := 1
def base_bonus_threshold : ℕ := 10
def base_bonus : ℕ := 10
def extra_bonus_per_subscription : ℝ := 0.5

-- Define total subscriptions sold
def total_subscriptions_sold : ℕ := 
  subscriptions_sold_to_parents + subscriptions_sold_to_grandfather + 
  subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor

-- Define earnings from subscriptions
def earnings_from_subscriptions : ℕ := total_subscriptions_sold * price_per_subscription

-- Define bonuses
def family_bonus : ℕ :=
  (subscriptions_sold_to_parents + subscriptions_sold_to_grandfather) * family_bonus_per_subscription

def neighbor_bonus : ℕ := 
  (subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor) * neighbor_bonus_per_subscription

def total_bonus : ℕ := family_bonus + neighbor_bonus

-- Define additional boss bonus
def additional_boss_bonus : ℝ := 
  if total_subscriptions_sold > base_bonus_threshold then 
    base_bonus + extra_bonus_per_subscription * (total_subscriptions_sold - base_bonus_threshold) 
  else 0

-- Define total earnings
def total_earnings : ℝ :=
  earnings_from_subscriptions + total_bonus + additional_boss_bonus

-- Theorem statement
theorem maggie_earnings_correct : total_earnings = 81.5 :=
by
  unfold total_earnings
  unfold earnings_from_subscriptions
  unfold total_bonus
  unfold family_bonus
  unfold neighbor_bonus
  unfold additional_boss_bonus
  unfold total_subscriptions_sold
  simp
  norm_cast
  sorry

end maggie_earnings_correct_l34_34114


namespace platform_length_proof_l34_34891

-- Given conditions
def train_length : ℝ := 300
def time_to_cross_platform : ℝ := 27
def time_to_cross_pole : ℝ := 18

-- The length of the platform L to be proved
def length_of_platform (L : ℝ) : Prop := 
  (train_length / time_to_cross_pole) = (train_length + L) / time_to_cross_platform

theorem platform_length_proof : length_of_platform 150 :=
by
  sorry

end platform_length_proof_l34_34891


namespace combinatorial_proof_l34_34356

noncomputable def combinatorial_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) : ℕ :=
  let summation_term (i : ℕ) := Nat.choose k i * Nat.choose n (m - i)
  List.sum (List.map summation_term (List.range (k + 1)))

theorem combinatorial_proof (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) :
  combinatorial_identity n m k h1 h2 h3 = Nat.choose (n + k) m :=
sorry

end combinatorial_proof_l34_34356


namespace alice_favorite_number_l34_34185

/-- Definition of what Alice loves -/
def is_fav_number (n : ℕ) : Prop :=
  (n ≥ 70 ∧ n ≤ 150) ∧
  (n % 13 = 0) ∧
  (n % 3 ≠ 0) ∧
  (Nat.digits 10 n).sum.Prime

/-- Verifies that the number 104 is Alice's favorite number -/
theorem alice_favorite_number : is_fav_number 104 :=
  by
    sorry

end alice_favorite_number_l34_34185


namespace sum_of_points_probabilities_l34_34159

-- Define probabilities for the sums of 2, 3, and 4
def P_A : ℚ := 1 / 36
def P_B : ℚ := 2 / 36
def P_C : ℚ := 3 / 36

-- Theorem statement
theorem sum_of_points_probabilities :
  (P_A < P_B) ∧ (P_B < P_C) :=
  sorry

end sum_of_points_probabilities_l34_34159


namespace equality_condition_l34_34786

theorem equality_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) → a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end equality_condition_l34_34786


namespace product_of_roots_cubicEq_l34_34334

noncomputable def cubicEq : Polynomial ℝ := Polynomial.Cubic 1 (-12) 48 28

theorem product_of_roots_cubicEq : cubicEq.roots.prod = -28 := 
sorry

end product_of_roots_cubicEq_l34_34334


namespace find_y_l34_34820

theorem find_y (x y : ℕ) (h1 : x^2 = y + 3) (h2 : x = 6) : y = 33 := 
by
  sorry

end find_y_l34_34820


namespace complement_fraction_irreducible_l34_34606

theorem complement_fraction_irreducible (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.gcd (b - a) b = 1 :=
sorry

end complement_fraction_irreducible_l34_34606


namespace max_min_diff_c_l34_34689

theorem max_min_diff_c (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  ∃ c_max c_min, 
  (∀ c', (a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18) → c_min ≤ c' ∧ c' ≤ c_max) 
  ∧ (c_max - c_min = 6) :=
  sorry

end max_min_diff_c_l34_34689


namespace total_books_l34_34290

open Finset

-- Define a set of students as a finite type
inductive Student : Type
| student1 : Student
| student2 : Student
| student3 : Student
| student4 : Student
| student5 : Student
| student6 : Student

-- Define a set of books as an indeterminate set
inductive Book : Type
| book : Nat → Book

-- Define a relation representing each student owning a set of books
def owns (s : Student) (b : Book) : Prop :=
  match s with
  | Student.student1 => b = Book.book 1 ∨ b = Book.book 2 ∨ b = Book.book 3 ∨ b = Book.book 4 ∨ b = Book.book 5
  | Student.student2 => b = Book.book 1 ∨ b = Book.book 6 ∨ b = Book.book 7 ∨ b = Book.book 8 ∨ b = Book.book 9
  | Student.student3 => b = Book.book 2 ∨ b = Book.book 6 ∨ b = Book.book 10 ∨ b = Book.book 11 ∨ b = Book.book 12
  | Student.student4 => b = Book.book 3 ∨ b = Book.book 7 ∨ b = Book.book 10 ∨ b = Book.book 13 ∨ b = Book.book 14
  | Student.student5 => b = Book.book 4 ∨ b = Book.book 8 ∨ b = Book.book 11 ∨ b = Book.book 13 ∨ b = Book.book 15
  | Student.student6 => b = Book.book 5 ∨ b = Book.book 9 ∨ b = Book.book 12 ∨ b = Book.book 14 ∨ b = Book.book 15

-- Define the lean proof problem statement
theorem total_books {s1 s2 s3 s4 s5 s6 : Student}: 
  (∀ (s1 s2 : Student), s1 ≠ s2 → ∃! b : Book, owns s1 b ∧ owns s2 b) →
  (∀ b : Book, ∃! s1 s2 : Student, s1 ≠ s2 ∧ owns s1 b ∧ owns s2 b) →
  ∃ (n : Nat), n = 15 :=
by
  sorry

end total_books_l34_34290


namespace tan_315_degree_l34_34971

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l34_34971


namespace sum_of_midpoint_coordinates_l34_34278

theorem sum_of_midpoint_coordinates 
  (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : (x1, y1, z1) = (2, 3, 4)) 
  (h2 : (x2, y2, z2) = (8, 15, 12)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 + (z1 + z2) / 2 = 22 := 
by
  sorry

end sum_of_midpoint_coordinates_l34_34278


namespace four_letter_initial_sets_l34_34813

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l34_34813


namespace tan_315_eq_neg1_l34_34949

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34949


namespace least_possible_value_l34_34010

noncomputable def least_value_expression (x : ℝ) : ℝ :=
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024

theorem least_possible_value : ∃ x : ℝ, least_value_expression x = 2023 :=
  sorry

end least_possible_value_l34_34010


namespace smallest_w_l34_34664

theorem smallest_w (w : ℕ) (h1 : 1916 = 2^2 * 479) (h2 : w > 0) : w = 74145392000 ↔ 
  (∀ p e, (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11) → (∃ k, (1916 * w = p^e * k ∧ e ≥ if p = 2 then 6 else 3))) :=
sorry

end smallest_w_l34_34664


namespace find_sum_l34_34831

variable {x y : ℝ}

theorem find_sum (h1 : x ≠ y)
    (h2 : matrix.det ![![2, 5, 10], ![4, x, y], ![4, y, x]] = 0) :
    x + y = 30 := 
sorry

end find_sum_l34_34831


namespace least_number_of_cookies_l34_34406

theorem least_number_of_cookies (c : ℕ) :
  (c % 6 = 5) ∧ (c % 8 = 7) ∧ (c % 9 = 6) → c = 23 :=
by
  sorry

end least_number_of_cookies_l34_34406


namespace probability_of_winning_exactly_once_l34_34008

-- Define the probability of player A winning a match
def prob_win_A (p : ℝ) : Prop := (1 - p) ^ 3 = 1 - 63 / 64

-- Define the binomial probability for exactly one win in three matches
def binomial_prob (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2

theorem probability_of_winning_exactly_once (p : ℝ) (h : prob_win_A p) : binomial_prob p = 9 / 64 :=
sorry

end probability_of_winning_exactly_once_l34_34008


namespace expected_sample_size_l34_34759

noncomputable def highSchoolTotalStudents (f s j : ℕ) : ℕ :=
  f + s + j

noncomputable def expectedSampleSize (total : ℕ) (p : ℝ) : ℝ :=
  total * p

theorem expected_sample_size :
  let f := 400
  let s := 320
  let j := 280
  let p := 0.2
  let total := highSchoolTotalStudents f s j
  expectedSampleSize total p = 200 :=
by
  sorry

end expected_sample_size_l34_34759


namespace ratio_brownies_to_cookies_l34_34193

-- Conditions and definitions
def total_items : ℕ := 104
def cookies_sold : ℕ := 48
def brownies_sold : ℕ := total_items - cookies_sold

-- Problem statement
theorem ratio_brownies_to_cookies : (brownies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 7 ∧ (cookies_sold : ℕ) / (Nat.gcd brownies_sold cookies_sold) = 6 :=
by
  sorry

end ratio_brownies_to_cookies_l34_34193


namespace collinear_points_x_value_l34_34236

theorem collinear_points_x_value
  (x : ℝ)
  (h : ∃ m : ℝ, m = (1 - (-4)) / (-1 - 2) ∧ m = (-9 - (-4)) / (x - 2)) :
  x = 5 :=
by
  sorry

end collinear_points_x_value_l34_34236


namespace side_length_of_square_l34_34432

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34432


namespace triangle_angle_C_30_degrees_l34_34678

theorem triangle_angle_C_30_degrees 
  (A B C : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) 
  (h3 : A + B + C = 180) 
  : C = 30 :=
  sorry

end triangle_angle_C_30_degrees_l34_34678


namespace value_of_sqrt_x_plus_one_over_sqrt_x_l34_34259

noncomputable def find_value (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) : ℝ :=
  sqrt(x) + 1/sqrt(x)

theorem value_of_sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx_pos : 0 < x) (hx : x + 1/x = 50) :
  find_value x hx_pos hx = 2 * sqrt(13) :=
sorry

end value_of_sqrt_x_plus_one_over_sqrt_x_l34_34259


namespace cos_double_angle_of_parallel_vectors_l34_34804

variables {α : Type*}

/-- Given vectors a and b specified by the problem, if they are parallel, then cos 2α = 7/9. -/
theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α)) 
  (hb : b = (Real.cos α, 1)) 
  (parallel : a.1 * b.2 = a.2 * b.1) : 
  Real.cos (2 * α) = 7/9 := 
by 
  sorry

end cos_double_angle_of_parallel_vectors_l34_34804


namespace quadratic_completion_l34_34739

theorem quadratic_completion (x : ℝ) :
  (x^2 + 6 * x - 2) = ((x + 3)^2 - 11) := sorry

end quadratic_completion_l34_34739


namespace sports_club_problem_l34_34243

theorem sports_club_problem (total_members : ℕ) (members_playing_badminton : ℕ) 
  (members_playing_tennis : ℕ) (members_not_playing_either : ℕ) 
  (h_total_members : total_members = 100) (h_badminton : members_playing_badminton = 60) 
  (h_tennis : members_playing_tennis = 70) (h_neither : members_not_playing_either = 10) : 
  (members_playing_badminton + members_playing_tennis - 
   (total_members - members_not_playing_either) = 40) :=
by {
  sorry
}

end sports_club_problem_l34_34243


namespace sum_of_c_and_d_l34_34666

theorem sum_of_c_and_d (c d : ℝ) 
  (h1 : ∀ x, x ≠ 2 ∧ x ≠ -1 → x^2 + c * x + d ≠ 0)
  (h_asymp_2 : 2^2 + c * 2 + d = 0)
  (h_asymp_neg1 : (-1)^2 + c * (-1) + d = 0) :
  c + d = -3 :=
by 
  -- Proof placeholder
  sorry

end sum_of_c_and_d_l34_34666


namespace composite_divisor_le_sqrt_l34_34701

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem composite_divisor_le_sqrt (n : ℕ) (h : is_composite n) :
  ∃ d, 1 < d ∧ d ≤ nat.sqrt n ∧ n % d = 0 :=
sorry

end composite_divisor_le_sqrt_l34_34701


namespace solve_abs_inequality_l34_34863

theorem solve_abs_inequality (x : ℝ) : 
    (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ( -4 ≤ x ∧ x ≤ -1 ∨ 3 ≤ x ∧ x ≤ 6) := 
by
    sorry

end solve_abs_inequality_l34_34863


namespace tom_total_payment_l34_34004

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l34_34004


namespace opposite_neg_two_l34_34712

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l34_34712


namespace solution_to_problem_l34_34598

def problem_statement : Prop :=
  (2.017 * 2016 - 10.16 * 201.7 = 2017)

theorem solution_to_problem : problem_statement :=
by
  sorry

end solution_to_problem_l34_34598


namespace roots_of_transformed_quadratic_l34_34513

theorem roots_of_transformed_quadratic (a b p q s1 s2 : ℝ)
    (h_quad_eq : s1 ^ 2 + a * s1 + b = 0 ∧ s2 ^ 2 + a * s2 + b = 0)
    (h_sum_roots : s1 + s2 = -a)
    (h_prod_roots : s1 * s2 = b) :
        p = -(a ^ 4 - 4 * a ^ 2 * b + 2 * b ^ 2) ∧ 
        q = b ^ 4 :=
by
  sorry

end roots_of_transformed_quadratic_l34_34513


namespace parallel_line_through_point_l34_34854

theorem parallel_line_through_point (C : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = 2) ∧ (∃ l : ℝ, ∀ x y : ℝ, 3 * x + y + l = 0) → 
  (3 * 1 + 2 + C = 0) → C = -5 :=
by
  sorry

end parallel_line_through_point_l34_34854


namespace mike_took_23_green_marbles_l34_34638

-- Definition of the conditions
def original_green_marbles : ℕ := 32
def remaining_green_marbles : ℕ := 9

-- Definition of the statement we want to prove
theorem mike_took_23_green_marbles : original_green_marbles - remaining_green_marbles = 23 := by
  sorry

end mike_took_23_green_marbles_l34_34638


namespace number_of_grids_l34_34335

/-
Mathematically Equivalent Proof Problem:
Given a 4x4 grid with unique numbers from 1 to 16,
each row and column are in increasing order,
1 is at position (0,0) and 16 is at (3,3),
prove the number of such grids equals 14400.
-/

noncomputable def count_4x4_grids : ℕ := 14400

theorem number_of_grids :
  ∃ count : ℕ,
  (count = count_4x4_grids) ∧ 
  (∃ grid : Array (Array ℕ),
  (grid.size = 4 ∧ 
   ∀ i < grid.size, (grid[i].size = 4) ∧
   (∀ j < grid[i].size, (1 ≤ grid[i][j] ∧ grid[i][j] ≤ 16)) ∧
   (∀ i < 4, ∀ j < 4, i > 0 → grid[i][0] > grid[i-1][0]) ∧
   (∀ i < 4, ∀ j < 4, j > 0 → grid[i][j] > grid[i][j-1]) ∧
   (grid[0][0] = 1) ∧
   (grid[3][3] = 16))) := 
begin
  use 14400, 
  split,
  {
    -- Definition of count_4x4_grids is 14400
    refl,
  },
  {
    -- Skipping detailed construction and verification of the grid
    sorry,
  }
end

end number_of_grids_l34_34335


namespace who_had_second_value_card_in_first_game_l34_34147

variable (A B C : ℕ)
variable (x y z : ℕ)
variable (points_A points_B points_C : ℕ)

-- Provided conditions
variable (h1 : x < y ∧ y < z)
variable (h2 : points_A = 20)
variable (h3 : points_B = 10)
variable (h4 : points_C = 9)
variable (number_of_games : ℕ)
variable (h5 : number_of_games = 3)
variable (h6 : A + B + C = 39)  -- This corresponds to points_A + points_B + points_C = 39.
variable (h7 : ∃ x y z, x + y + z = 13 ∧ x < y ∧ y < z)
variable (h8 : B = z)

-- Question/Proof to establish
theorem who_had_second_value_card_in_first_game :
  ∃ p : ℕ, p = C :=
sorry

end who_had_second_value_card_in_first_game_l34_34147


namespace S_contains_finite_but_not_infinite_arith_progressions_l34_34702

noncomputable def S : Set ℤ := {n | ∃ k : ℕ, n = Int.floor (k * Real.pi)}

theorem S_contains_finite_but_not_infinite_arith_progressions :
  (∀ (k : ℕ), ∃ (a d : ℤ), ∀ (i : ℕ) (h : i < k), (a + i * d) ∈ S) ∧
  ¬(∃ (a d : ℤ), ∀ (n : ℕ), (a + n * d) ∈ S) :=
by
  sorry

end S_contains_finite_but_not_infinite_arith_progressions_l34_34702


namespace expression_equal_a_five_l34_34885

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end expression_equal_a_five_l34_34885


namespace find_e_value_l34_34893

theorem find_e_value : (14 ^ 2) * (5 ^ 3) * 568 = 13916000 := by
  sorry

end find_e_value_l34_34893


namespace difference_of_squares_l34_34125

theorem difference_of_squares (n : ℕ) : (n+1)^2 - n^2 = 2*n + 1 :=
by
  sorry

end difference_of_squares_l34_34125


namespace average_reduction_10_percent_l34_34758

noncomputable def average_percentage_reduction
  (initial_price : ℝ) (final_price : ℝ) (reductions : ℕ) : ℝ :=
let x := (final_price / initial_price) ^ (1 / reductions : ℝ) in
1 - x

theorem average_reduction_10_percent :
  average_percentage_reduction 50 40.5 2 = 0.1 :=
by
  unfold average_percentage_reduction
  have h : (40.5 / 50 : ℝ) ^ (1 / 2 : ℝ) = 0.9 :=
    by sorry -- calculation of the square root
  rw h
  norm_num
  sorry

end average_reduction_10_percent_l34_34758


namespace student_percentage_in_math_l34_34181

theorem student_percentage_in_math (M H T : ℝ) (H_his : H = 84) (H_third : T = 69) (H_avg : (M + H + T) / 3 = 75) : M = 72 :=
by
  sorry

end student_percentage_in_math_l34_34181


namespace smallest_prime_linear_pair_l34_34007

def is_prime (n : ℕ) : Prop := ¬(∃ k > 1, k < n ∧ k ∣ n)

theorem smallest_prime_linear_pair :
  ∃ a b : ℕ, is_prime a ∧ is_prime b ∧ a + b = 180 ∧ a > b ∧ b = 7 := 
by
  sorry

end smallest_prime_linear_pair_l34_34007


namespace tan_315_proof_l34_34959

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l34_34959


namespace midpoint_parallel_l34_34395

open ComplexCongruence

theorem midpoint_parallel (A B C M N P Q O I : Point)
    (circumcircle : Circle)
    (h_circumcircle : ∀ X, X ∈ circumcircle ↔ X = A ∨ X = B ∨ X = C)
    (hM : ∀ arc, arc ≠ C ∧ arc.midpoint = M → M ∈ circumcircle)
    (hN : ∀ arc, arc ≠ A ∧ arc.midpoint = N → N ∈ circumcircle)
    (hO : O = circumcenter A B C)
    (hP : P ∈ Line I Z)
    (hQ : Q ∈ Line I Z)
    (hPQ_perp : PQ ⊥ BI)
    (hMN_perp : MN ⊥ BI) :
  MN ∥ PQ := 
sorry

end midpoint_parallel_l34_34395


namespace Kenneth_money_left_l34_34684

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l34_34684


namespace senate_arrangement_l34_34026

def countArrangements : ℕ :=
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- The calculation for arrangements considering fixed elements, and permutations adjusted for rotation
  12 * (Nat.factorial 10 / 2)

theorem senate_arrangement :
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- Total ways to arrange the members around the table under the given conditions
  countArrangements = 21772800 :=
by
  sorry

end senate_arrangement_l34_34026


namespace complex_multiplication_l34_34401

-- Definition of the imaginary unit
def is_imaginary_unit (i : ℂ) : Prop := i * i = -1

theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) : (1 + i) * (1 - i) = 2 :=
by
  -- Given that i is the imaginary unit satisfying i^2 = -1
  -- We need to show that (1 + i) * (1 - i) = 2
  sorry

end complex_multiplication_l34_34401


namespace math_problem_l34_34277

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def a_n (n : ℕ) : ℕ := 3 * n - 5

theorem math_problem (C5_4 : ℕ) (C6_4 : ℕ) (C7_4 : ℕ) :
  C5_4 = binomial 5 4 →
  C6_4 = binomial 6 4 →
  C7_4 = binomial 7 4 →
  C5_4 + C6_4 + C7_4 = 55 →
  ∃ n : ℕ, a_n n = 55 ∧ n = 20 :=
by
  sorry

end math_problem_l34_34277


namespace remainder_when_four_times_n_minus_9_divided_by_11_l34_34501

theorem remainder_when_four_times_n_minus_9_divided_by_11 
  (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end remainder_when_four_times_n_minus_9_divided_by_11_l34_34501


namespace PJ_approx_10_81_l34_34677

noncomputable def PJ_length (P Q R J : Type) (PQ PR QR : ℝ) : ℝ :=
  if PQ = 30 ∧ PR = 29 ∧ QR = 27 then 10.81 else 0

theorem PJ_approx_10_81 (P Q R J : Type) (PQ PR QR : ℝ):
  PQ = 30 ∧ PR = 29 ∧ QR = 27 → PJ_length P Q R J PQ PR QR = 10.81 :=
by sorry

end PJ_approx_10_81_l34_34677


namespace calculate_X_l34_34826

theorem calculate_X
  (top_seg1 : ℕ) (top_seg2 : ℕ) (X : ℕ)
  (vert_seg : ℕ)
  (bottom_seg1 : ℕ) (bottom_seg2 : ℕ) (bottom_seg3 : ℕ)
  (h1 : top_seg1 = 3) (h2 : top_seg2 = 2)
  (h3 : vert_seg = 4)
  (h4 : bottom_seg1 = 4) (h5 : bottom_seg2 = 2) (h6 : bottom_seg3 = 5)
  (h_eq : 5 + X = 11) :
  X = 6 :=
by
  -- Proof is omitted as per instructions.
  sorry

end calculate_X_l34_34826


namespace max_necklaces_with_beads_l34_34475

noncomputable def necklace_problem : Prop :=
  ∃ (necklaces : ℕ),
    let green_beads := 200
    let white_beads := 100
    let orange_beads := 50
    let beads_per_pattern_green := 3
    let beads_per_pattern_white := 1
    let beads_per_pattern_orange := 1
    necklaces = orange_beads ∧
    green_beads / beads_per_pattern_green >= necklaces ∧
    white_beads / beads_per_pattern_white >= necklaces ∧
    orange_beads / beads_per_pattern_orange >= necklaces

theorem max_necklaces_with_beads : necklace_problem :=
  sorry

end max_necklaces_with_beads_l34_34475


namespace company_fund_initial_amount_l34_34280

theorem company_fund_initial_amount (n : ℕ) 
  (h : 45 * n + 95 = 50 * n - 5) : 50 * n - 5 = 995 := by
  sorry

end company_fund_initial_amount_l34_34280


namespace taxi_fare_miles_l34_34865

theorem taxi_fare_miles (total_spent : ℝ) (tip : ℝ) (base_fare : ℝ) (additional_fare_rate : ℝ) (base_mile : ℝ) (additional_mile_unit : ℝ) (x : ℝ) :
  (total_spent = 15) →
  (tip = 3) →
  (base_fare = 3) →
  (additional_fare_rate = 0.25) →
  (base_mile = 0.5) →
  (additional_mile_unit = 0.1) →
  (x = base_mile + (total_spent - tip - base_fare) / (additional_fare_rate / additional_mile_unit)) →
  x = 4.1 :=
by
  intros
  sorry

end taxi_fare_miles_l34_34865


namespace taehyung_collected_most_points_l34_34748

def largest_collector : Prop :=
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  taehyung_points > yoongi_points ∧ 
  taehyung_points > jungkook_points ∧ 
  taehyung_points > yuna_points ∧ 
  taehyung_points > yoojung_points

theorem taehyung_collected_most_points : largest_collector :=
by
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  sorry

end taehyung_collected_most_points_l34_34748


namespace side_length_of_square_l34_34438

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l34_34438


namespace downstream_distance_80_l34_34146

-- Conditions
variables (Speed_boat Speed_stream Distance_upstream : ℝ)

-- Assign given values
def speed_boat := 36 -- kmph
def speed_stream := 12 -- kmph
def distance_upstream := 40 -- km

-- Effective speeds
def speed_downstream := speed_boat + speed_stream -- kmph
def speed_upstream := speed_boat - speed_stream -- kmph

-- Downstream distance
noncomputable def distance_downstream : ℝ := 80 -- km

-- Theorem
theorem downstream_distance_80 :
  speed_boat = 36 → speed_stream = 12 → distance_upstream = 40 →
  (distance_upstream / speed_upstream = distance_downstream / speed_downstream) :=
by
  sorry

end downstream_distance_80_l34_34146


namespace smallest_x_for_perfect_cube_l34_34021

theorem smallest_x_for_perfect_cube :
  ∃ (x : ℕ) (h : x > 0), x = 36 ∧ (∃ (k : ℕ), 1152 * x = k ^ 3) := by
  sorry

end smallest_x_for_perfect_cube_l34_34021


namespace opposite_of_neg_two_l34_34714

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l34_34714


namespace geom_seq_inverse_sum_l34_34249

theorem geom_seq_inverse_sum 
  (a_2 a_3 a_4 a_5 : ℚ) 
  (h1 : a_2 * a_5 = -3 / 4) 
  (h2 : a_2 + a_3 + a_4 + a_5 = 5 / 4) :
  1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = -4 / 3 :=
sorry

end geom_seq_inverse_sum_l34_34249


namespace fraction_in_range_l34_34747

theorem fraction_in_range : 
  (2:ℝ) / 5 < (4:ℝ) / 7 ∧ (4:ℝ) / 7 < 3 / 4 := by
  sorry

end fraction_in_range_l34_34747


namespace subset_M_N_l34_34835

-- Definitions of M and N as per the problem statement
def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | 1 / x < 2}

-- Lean statement for the proof problem: M ⊆ N
theorem subset_M_N : M ⊆ N := by
  -- Proof will be provided here
  sorry

end subset_M_N_l34_34835


namespace intersection_M_N_eq_set_l34_34514

universe u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {y | ∃ x, x ∈ M ∧ y = 2 * x + 1}

-- Prove the intersection M ∩ N = {-1, 1}
theorem intersection_M_N_eq_set : M ∩ N = {-1, 1} :=
by
  simp [Set.ext_iff, M, N]
  sorry

end intersection_M_N_eq_set_l34_34514


namespace average_shifted_samples_l34_34223

variables (x1 x2 x3 x4 : ℝ)

theorem average_shifted_samples (h : (x1 + x2 + x3 + x4) / 4 = 2) :
  ((x1 + 3) + (x2 + 3) + (x3 + 3) + (x4 + 3)) / 4 = 5 :=
by
  sorry

end average_shifted_samples_l34_34223


namespace problem_1_problem_2_l34_34537

-- defining S_n as set of binary vectors of length n
def S (n : ℕ) := { A : Fin n -> ℕ // ∀ i, A i = 0 ∨ A i = 1 }

-- definition of distance function d
def d {n : ℕ} (U V : S n) : ℕ :=
  Finset.card { i : Fin n | U.val i ≠ V.val i }

-- Problem 1: Lean statement
theorem problem_1 : 
  let U : S 6 := ⟨λ _, 1, λ i, or.inr rfl⟩,
  m = finset.card { V : S 6 | d U V = 2 } :=
  sorry

-- Problem 2: Lean statement
theorem problem_2 (U : S n) : 
  let V_set := { V : S n }, 
  (finset.sum V_set (λ V, d U V)) = n * 2^(n-1) :=
  sorry

end problem_1_problem_2_l34_34537


namespace negation_equiv_l34_34093

theorem negation_equiv (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l34_34093


namespace exponent_of_4_l34_34293

theorem exponent_of_4 (x : ℕ) (h₁ : (1 / 4 : ℚ) ^ 2 = 1 / 16) (h₂ : 16384 * (1 / 16 : ℚ) = 1024) :
  4 ^ x = 1024 → x = 5 :=
by
  sorry

end exponent_of_4_l34_34293


namespace fruit_seller_price_l34_34763

theorem fruit_seller_price (C : ℝ) (h1 : 1.05 * C = 14.823529411764707) : 
  0.85 * C = 12 := 
sorry

end fruit_seller_price_l34_34763


namespace zero_of_my_function_l34_34289

-- Define the function y = e^(2x) - 1
noncomputable def my_function (x : ℝ) : ℝ :=
  Real.exp (2 * x) - 1

-- Statement that the zero of the function is at x = 0
theorem zero_of_my_function : my_function 0 = 0 :=
by sorry

end zero_of_my_function_l34_34289


namespace minimum_value_am_bn_l34_34065

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l34_34065


namespace min_stamps_l34_34680

theorem min_stamps : ∃ (x y : ℕ), 5 * x + 7 * y = 35 ∧ x + y = 5 :=
by
  have : ∀ (x y : ℕ), 5 * x + 7 * y = 35 → x + y = 5 → True := sorry
  sorry

end min_stamps_l34_34680


namespace equal_intercepts_no_second_quadrant_l34_34853

/- Given line equation (a + 1)x + y + 2 - a = 0 and a \in ℝ. -/
def line_eq (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/- If the line l has equal intercepts on both coordinate axes, 
   then a = 0 or a = 2. -/
theorem equal_intercepts (a : ℝ) :
  (∃ x y : ℝ, line_eq a x 0 ∧ line_eq a 0 y ∧ x = y) →
  a = 0 ∨ a = 2 :=
sorry

/- If the line l does not pass through the second quadrant,
   then a ≤ -1. -/
theorem no_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → ¬ line_eq a x y) →
  a ≤ -1 :=
sorry

end equal_intercepts_no_second_quadrant_l34_34853


namespace one_thirds_in_fraction_l34_34088

theorem one_thirds_in_fraction : (9 / 5) / (1 / 3) = 27 / 5 := by
  sorry

end one_thirds_in_fraction_l34_34088


namespace marilyn_ends_up_with_55_caps_l34_34547

def marilyn_initial_caps := 165
def caps_shared_with_nancy := 78
def caps_received_from_charlie := 23

def remaining_caps (initial caps_shared caps_received: ℕ) :=
  initial - caps_shared + caps_received

def caps_given_away (total_caps: ℕ) :=
  total_caps / 2

def final_caps (initial caps_shared caps_received: ℕ) :=
  remaining_caps initial caps_shared caps_received - caps_given_away (remaining_caps initial caps_shared caps_received)

theorem marilyn_ends_up_with_55_caps :
  final_caps marilyn_initial_caps caps_shared_with_nancy caps_received_from_charlie = 55 :=
by
  sorry

end marilyn_ends_up_with_55_caps_l34_34547


namespace ice_cream_vendor_l34_34189

theorem ice_cream_vendor (M : ℕ) (h3 : 50 - (3 / 5) * 50 = 20) (h4 : (2 / 3) * M = 2 * M / 3) 
  (h5 : (50 - 30) + M - (2 * M / 3) = 38) :
  M = 12 :=
by
  sorry

end ice_cream_vendor_l34_34189


namespace percentage_increase_l34_34590

variable (A B C : ℝ)
variable (h1 : A = 0.71 * C)
variable (h2 : A = 0.05 * B)

theorem percentage_increase (A B C : ℝ) (h1 : A = 0.71 * C) (h2 : A = 0.05 * B) : (B - C) / C = 13.2 :=
by
  sorry

end percentage_increase_l34_34590


namespace tan_315_eq_neg1_l34_34928

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34928


namespace tan_315_eq_neg1_l34_34922

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34922


namespace fraction_sum_5625_l34_34443

theorem fraction_sum_5625 : 
  ∃ (a b : ℕ), 0.5625 = (9 : ℚ) / 16 ∧ (a + b = 25) := 
by 
  sorry

end fraction_sum_5625_l34_34443


namespace place_value_ratio_l34_34251

def number : ℝ := 90347.6208
def place_value_0 : ℝ := 10000 -- tens of thousands
def place_value_6 : ℝ := 0.1 -- tenths

theorem place_value_ratio : 
  place_value_0 / place_value_6 = 100000 := by 
    sorry

end place_value_ratio_l34_34251


namespace don_eats_80_pizzas_l34_34336

variable (D Daria : ℝ)

-- Condition 1: Daria consumes 2.5 times the amount of pizza that Don does.
def condition1 : Prop := Daria = 2.5 * D

-- Condition 2: Together, they eat 280 pizzas.
def condition2 : Prop := D + Daria = 280

-- Conclusion: The number of pizzas Don eats is 80.
theorem don_eats_80_pizzas (h1 : condition1 D Daria) (h2 : condition2 D Daria) : D = 80 :=
by
  sorry

end don_eats_80_pizzas_l34_34336


namespace sum_of_fourth_powers_l34_34379

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 :=
by sorry

end sum_of_fourth_powers_l34_34379


namespace pink_tulips_l34_34143

theorem pink_tulips (total_tulips : ℕ)
    (blue_ratio : ℚ) (red_ratio : ℚ)
    (h_total : total_tulips = 56)
    (h_blue_ratio : blue_ratio = 3/8)
    (h_red_ratio : red_ratio = 3/7) :
    ∃ pink_tulips : ℕ, pink_tulips = total_tulips - ((blue_ratio * total_tulips) + (red_ratio * total_tulips)) ∧ pink_tulips = 11 := by
  sorry

end pink_tulips_l34_34143


namespace first_number_remainder_one_l34_34856

theorem first_number_remainder_one (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 2023) :
  (∀ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 → (a % 3 ≠ b % 3 ∧ a % 3 ≠ c % 3 ∧ b % 3 ≠ c % 3))
  → (n % 3 = 1) :=
sorry

end first_number_remainder_one_l34_34856


namespace inequality1_inequality2_l34_34124

variable (a b c d : ℝ)

theorem inequality1 : 
  (a + c)^2 * (b + d)^2 ≥ 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) :=
  sorry

theorem inequality2 : 
  (a + c)^2 * (b + d)^2 ≥ 4 * b * c * (c * d + d * a + a * b) :=
  sorry

end inequality1_inequality2_l34_34124


namespace lucy_total_journey_l34_34838

-- Define the length of Lucy's journey
def lucy_journey (x : ℝ) : Prop :=
  (1 / 4) * x + 25 + (1 / 6) * x = x

-- State the theorem
theorem lucy_total_journey : ∃ x : ℝ, lucy_journey x ∧ x = 300 / 7 := by
  sorry

end lucy_total_journey_l34_34838


namespace sufficient_but_not_necessary_l34_34022

theorem sufficient_but_not_necessary (x : ℝ) : (x < -2 → x ≤ 0) → ¬(x ≤ 0 → x < -2) :=
by
  sorry

end sufficient_but_not_necessary_l34_34022


namespace gcd_12012_21021_l34_34648

-- Definitions
def factors_12012 : List ℕ := [2, 2, 3, 7, 11, 13] -- Factors of 12,012
def factors_21021 : List ℕ := [3, 7, 7, 11, 13] -- Factors of 21,021

def common_factors := [3, 7, 11, 13] -- Common factors between 12,012 and 21,021

def gcd (ls : List ℕ) : ℕ :=
ls.foldr Nat.gcd 0 -- Function to calculate gcd of list of numbers

-- Main statement
theorem gcd_12012_21021 : gcd common_factors = 1001 := by
  -- Proof is not required, so we use sorry to skip the proof.
  sorry

end gcd_12012_21021_l34_34648


namespace time_between_ticks_at_6_oclock_l34_34328

theorem time_between_ticks_at_6_oclock (ticks6 ticks12 intervals6 intervals12 total_time12: ℕ) (time_per_tick : ℕ) :
  ticks6 = 6 →
  ticks12 = 12 →
  total_time12 = 66 →
  intervals12 = ticks12 - 1 →
  time_per_tick = total_time12 / intervals12 →
  intervals6 = ticks6 - 1 →
  (time_per_tick * intervals6) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end time_between_ticks_at_6_oclock_l34_34328


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l34_34134

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l34_34134


namespace total_collisions_100_balls_l34_34616

def num_of_collisions (n: ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_collisions_100_balls :
  num_of_collisions 100 = 4950 :=
by
  sorry

end total_collisions_100_balls_l34_34616


namespace coupon_probability_l34_34582

theorem coupon_probability :
  (Nat.choose 6 6 * Nat.choose 11 3 : ℚ) / Nat.choose 17 9 = 3 / 442 :=
by
  sorry

end coupon_probability_l34_34582


namespace tangent_315_deg_l34_34976

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l34_34976


namespace tan_315_eq_neg1_l34_34962

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34962


namespace transformed_expression_value_l34_34611

-- Defining the new operations according to the problem's conditions
def new_minus (a b : ℕ) : ℕ := a + b
def new_plus (a b : ℕ) : ℕ := a * b
def new_times (a b : ℕ) : ℕ := a / b
def new_div (a b : ℕ) : ℕ := a - b

-- Problem statement
theorem transformed_expression_value : new_minus 6 (new_plus 9 (new_times 8 (new_div 3 25))) = 5 :=
sorry

end transformed_expression_value_l34_34611


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l34_34135

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l34_34135


namespace side_length_of_square_l34_34435

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34435


namespace tan_315_degrees_l34_34999

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l34_34999


namespace least_possible_multiple_l34_34239

theorem least_possible_multiple (x y z k : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 1 ≤ k)
  (h1 : 3 * x = k * z) (h2 : 4 * y = k * z) (h3 : x - y + z = 19) : 3 * x = 12 :=
by
  sorry

end least_possible_multiple_l34_34239


namespace digits_product_l34_34774

-- Define the conditions
variables (A B : ℕ)

-- Define the main problem statement using the conditions and expected answer
theorem digits_product (h1 : A + B = 12) (h2 : (10 * A + B) % 3 = 0) : A * B = 35 := 
by
  sorry

end digits_product_l34_34774


namespace contrapositive_equivalence_l34_34601
-- Importing the necessary libraries

-- Declaring the variables P and Q as propositions
variables (P Q : Prop)

-- The statement that we need to prove
theorem contrapositive_equivalence :
  (P → ¬ Q) ↔ (Q → ¬ P) :=
sorry

end contrapositive_equivalence_l34_34601


namespace cos_of_angle_through_point_l34_34511

-- Define the point P and the angle α
def P : ℝ × ℝ := (4, 3)
def α : ℝ := sorry  -- α is an angle such that its terminal side passes through P

-- Define the squared distance from the origin to the point P
noncomputable def distance_squared : ℝ := P.1^2 + P.2^2

-- Define cos α
noncomputable def cosα : ℝ := P.1 / (Real.sqrt distance_squared)

-- State the theorem
theorem cos_of_angle_through_point : cosα = 4 / 5 := 
by sorry

end cos_of_angle_through_point_l34_34511


namespace option_C_equals_a5_l34_34889

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end option_C_equals_a5_l34_34889


namespace remainder_of_polynomial_division_l34_34157

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ((x + 2) ^ 2023) % (x^2 + x + 1) = 1 :=
by
  sorry

end remainder_of_polynomial_division_l34_34157


namespace volleyball_team_l34_34550

theorem volleyball_team :
  let total_combinations := (Nat.choose 15 6)
  let without_triplets := (Nat.choose 12 6)
  total_combinations - without_triplets = 4081 :=
by
  -- Definitions based on the problem conditions
  let team_size := 15
  let starters := 6
  let triplets := 3
  let total_combinations := Nat.choose team_size starters
  let without_triplets := Nat.choose (team_size - triplets) starters
  -- Identify the proof goal
  have h : total_combinations - without_triplets = 4081 := sorry
  exact h

end volleyball_team_l34_34550


namespace red_bowl_values_possible_l34_34149

theorem red_bowl_values_possible (r b y : ℕ) 
(h1 : r + b + y = 27)
(h2 : 15 * r + 3 * b + 18 * y = 378) : 
  r = 11 ∨ r = 16 ∨ r = 21 := 
  sorry

end red_bowl_values_possible_l34_34149


namespace cards_problem_l34_34006

-- Define the conditions and goal
theorem cards_problem 
    (L R : ℕ) 
    (h1 : L + 6 = 3 * (R - 6))
    (h2 : R + 2 = 2 * (L - 2)) : 
    L = 66 := 
by 
  -- proof goes here
  sorry

end cards_problem_l34_34006


namespace alberto_biked_more_than_bjorn_l34_34242

-- Define the distances traveled by Bjorn and Alberto after 5 hours.
def b_distance : ℝ := 75
def a_distance : ℝ := 100

-- Statement to prove the distance difference after 5 hours.
theorem alberto_biked_more_than_bjorn : a_distance - b_distance = 25 := 
by
  -- Proof is skipped, focusing only on the statement.
  sorry

end alberto_biked_more_than_bjorn_l34_34242


namespace one_fourth_div_one_eighth_l34_34084

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l34_34084


namespace coupon_probability_l34_34586

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l34_34586


namespace part1_part2_l34_34833

def proposition_p (m : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 2 * x - 4 ≥ m^2 - 5 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - 2 * x + m - 1 ≤ 0

theorem part1 (m : ℝ) : proposition_p m → 1 ≤ m ∧ m ≤ 4 := 
sorry

theorem part2 (m : ℝ) : (proposition_p m ∨ proposition_q m) → m ≤ 4 := 
sorry

end part1_part2_l34_34833


namespace discount_on_soap_l34_34624

theorem discount_on_soap :
  (let chlorine_price := 10
   let chlorine_discount := 0.20 * chlorine_price
   let discounted_chlorine_price := chlorine_price - chlorine_discount

   let soap_price := 16

   let total_savings := 26

   let chlorine_savings := 3 * chlorine_price - 3 * discounted_chlorine_price
   let soap_savings := total_savings - chlorine_savings

   let discount_per_soap := soap_savings / 5
   let discount_percentage_per_soap := (discount_per_soap / soap_price) * 100
   discount_percentage_per_soap = 25) := sorry

end discount_on_soap_l34_34624


namespace find_f_13_l34_34024

variable (f : ℤ → ℤ)

def is_odd_function (f : ℤ → ℤ) := ∀ x : ℤ, f (-x) = -f (x)
def has_period_4 (f : ℤ → ℤ) := ∀ x : ℤ, f (x + 4) = f (x)

theorem find_f_13 (h1 : is_odd_function f) (h2 : has_period_4 f) (h3 : f (-1) = 2) : f 13 = -2 :=
by
  sorry

end find_f_13_l34_34024


namespace expected_value_8_sided_die_l34_34307

-- Define the roll outcomes and their associated probabilities
def roll_outcome (n : ℕ) : ℕ := 2 * n^2

-- Define the expected value calculation
def expected_value (sides : ℕ) : ℚ := ∑ i in range (1, sides+1), (1 / sides) * roll_outcome i

-- Prove the expected value calculation for an 8-sided fair die
theorem expected_value_8_sided_die : expected_value 8 = 51 := by
  sorry

end expected_value_8_sided_die_l34_34307


namespace house_prices_and_yields_l34_34272

theorem house_prices_and_yields :
  ∃ x y : ℝ, 
    (425 = (y / 100) * x) ∧ 
    (459 = ((y - 0.5) / 100) * (6/5) * x) ∧ 
    (x = 8500) ∧ 
    (y = 5) ∧ 
    ((6/5) * x = 10200) ∧ 
    (y - 0.5 = 4.5) :=
by
  sorry

end house_prices_and_yields_l34_34272


namespace time_to_print_800_flyers_l34_34761

theorem time_to_print_800_flyers (x : ℝ) (h1 : 0 < x) :
  (1 / 6) + (1 / x) = 1 / 1.5 ↔ ∀ y : ℝ, 800 / 6 + 800 / x = 800 / 1.5 :=
by sorry

end time_to_print_800_flyers_l34_34761


namespace positive_X_solution_l34_34051

def boxtimes (X Y : ℤ) : ℤ := X^2 - 2 * X + Y^2

theorem positive_X_solution (X : ℤ) (h : boxtimes X 7 = 164) : X = 13 :=
by
  sorry

end positive_X_solution_l34_34051


namespace kenneth_money_left_l34_34683

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l34_34683


namespace min_value_expression_l34_34066

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l34_34066


namespace opposite_of_neg_two_is_two_l34_34720

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l34_34720


namespace tan_315_eq_neg_one_l34_34995

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34995


namespace adam_cat_food_packages_l34_34773

theorem adam_cat_food_packages (c : ℕ) 
  (dog_food_packages : ℕ := 7) 
  (cans_per_cat_package : ℕ := 10) 
  (cans_per_dog_package : ℕ := 5) 
  (extra_cat_food_cans : ℕ := 55) 
  (total_dog_cans : ℕ := dog_food_packages * cans_per_dog_package) 
  (total_cat_cans : ℕ := c * cans_per_cat_package)
  (h : total_cat_cans = total_dog_cans + extra_cat_food_cans) : 
  c = 9 :=
by
  sorry

end adam_cat_food_packages_l34_34773


namespace no_isomorphic_components_after_edge_removal_l34_34241

variables {V : Type*} [Fintype V] [DecidableEq V]

/-- A connected graph with specific vertex degrees. -/
structure special_graph (G : SimpleGraph V) : Prop :=
(connected : G.Connected)
(degree_three_vertices : Fintype.card {v : V // G.degree v = 3} = 4)
(degree_four_vertices  : ∀ v : V, G.degree v = 4 ∨ G.degree v = 3)

theorem no_isomorphic_components_after_edge_removal (G : SimpleGraph V) 
  (hG : special_graph G) :
  ∀ (e : G.EdgeSet),
    ¬ (G.deleteEdge e).Cuts e.toApex.connected
    ∧ (G.deleteEdge e).Cuts e.toApex.symmetric :=
sorry

end no_isomorphic_components_after_edge_removal_l34_34241


namespace product_bc_l34_34454

theorem product_bc {b c : ℤ} (h1 : ∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) :
    b * c = 110 :=
sorry

end product_bc_l34_34454


namespace tan_315_eq_neg_one_l34_34940

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34940


namespace Flynn_tv_minutes_weekday_l34_34651

theorem Flynn_tv_minutes_weekday :
  ∀ (tv_hours_per_weekend : ℕ)
    (tv_hours_per_year : ℕ)
    (weeks_per_year : ℕ) 
    (weekdays_per_week : ℕ),
  tv_hours_per_weekend = 2 →
  tv_hours_per_year = 234 →
  weeks_per_year = 52 →
  weekdays_per_week = 5 →
  (tv_hours_per_year - (tv_hours_per_weekend * weeks_per_year)) / (weekdays_per_week * weeks_per_year) * 60
  = 30 :=
by
  intros tv_hours_per_weekend tv_hours_per_year weeks_per_year weekdays_per_week
        h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end Flynn_tv_minutes_weekday_l34_34651


namespace solve_for_y_l34_34132

theorem solve_for_y (y : ℝ) (h : y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) : y = -4 :=
by {
  sorry
}

end solve_for_y_l34_34132


namespace gcd_of_2535_5929_11629_l34_34348

theorem gcd_of_2535_5929_11629 : Nat.gcd (Nat.gcd 2535 5929) 11629 = 1 := by
  sorry

end gcd_of_2535_5929_11629_l34_34348


namespace find_m_n_l34_34207

theorem find_m_n : ∃ (m n : ℕ), m > n ∧ m^3 - n^3 = 999 ∧ ((m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9)) :=
by
  sorry

end find_m_n_l34_34207


namespace pauline_convertibles_l34_34119

noncomputable def convertibles_count (total_cars : ℕ) (percent_regular : ℝ) (percent_trucks : ℝ) : ℕ :=
  total_cars - (percent_regular * total_cars).to_nat - (percent_trucks * total_cars).to_nat

theorem pauline_convertibles :
  let total_cars := 125
  let percent_regular := 0.64
  let percent_trucks := 0.08
  convertibles_count total_cars percent_regular percent_trucks = 35 :=
by
  sorry

end pauline_convertibles_l34_34119


namespace coefficient_of_x2_in_expansion_l34_34571

theorem coefficient_of_x2_in_expansion :
  (x - (2 : ℤ)/x) ^ 4 = 8 * x^2 := sorry

end coefficient_of_x2_in_expansion_l34_34571


namespace timeTakenByBobIs30_l34_34387

-- Define the conditions
def timeTakenByAlice : ℕ := 40
def fractionOfTimeBobTakes : ℚ := 3 / 4

-- Define the statement to be proven
theorem timeTakenByBobIs30 : (fractionOfTimeBobTakes * timeTakenByAlice : ℚ) = 30 := 
by
  sorry

end timeTakenByBobIs30_l34_34387


namespace sqrt_x_plus_inv_sqrt_x_l34_34260

variable (x : ℝ) (hx : 0 < x) (h : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + 1 / x = 50) : 
  sqrt x + 1 / sqrt x = 2 * sqrt 13 := 
sorry

end sqrt_x_plus_inv_sqrt_x_l34_34260


namespace stamps_count_l34_34121

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l34_34121


namespace paula_candies_l34_34551

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l34_34551


namespace xy_fraction_l34_34373

theorem xy_fraction (x y : ℚ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) :
  x * y = -1 / 5 := 
by sorry

end xy_fraction_l34_34373


namespace percentage_of_green_eyed_brunettes_l34_34324

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l34_34324


namespace tangent_line_at_A_tangent_line_through_B_l34_34071

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_A : 
  ∃ m b, m = -1 ∧ b = 4 ∧ (∀ x, 1 ≤ x → (x + b = 4)) :=
sorry

theorem tangent_line_through_B :
  ∃ m b, m = 4 ∧ b = -8 ∧ (∀ x, 1 ≤ x → (4*x + b = 8)) :=
sorry

end tangent_line_at_A_tangent_line_through_B_l34_34071


namespace distance_between_closest_points_l34_34150

noncomputable def distance_closest_points :=
  let center1 : ℝ × ℝ := (5, 3)
  let center2 : ℝ × ℝ := (20, 7)
  let radius1 := center1.2  -- radius of first circle is y-coordinate of its center
  let radius2 := center2.2  -- radius of second circle is y-coordinate of its center
  let distance_centers := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - radius1 - radius2

theorem distance_between_closest_points :
  distance_closest_points = Real.sqrt 241 - 10 :=
sorry

end distance_between_closest_points_l34_34150


namespace intersection_points_l34_34009

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x - 5
def parabola2 (x : ℝ) : ℝ := x ^ 2 - 2 * x + 3

theorem intersection_points :
  { p : ℝ × ℝ | p.snd = parabola1 p.fst ∧ p.snd = parabola2 p.fst } =
  { (1, -14), (4, -5) } :=
by
  sorry

end intersection_points_l34_34009


namespace andrei_stamps_l34_34123

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l34_34123


namespace distances_sum_in_triangle_l34_34253

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (a b c P AO BO CO : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = a + b + c

def point_inside_triangle (O : Type) : Prop := 
  ∃ (A B C : Type), True -- Placeholder for the actual geometric condition

def distances_to_vertices (O : Type) (AO BO CO : ℝ) : Prop := 
  AO >= 0 ∧ BO >= 0 ∧ CO >= 0

theorem distances_sum_in_triangle
  (h1 : triangle_sides a b c)
  (h2 : triangle_perimeter a b c P)
  (h3 : point_inside_triangle O)
  (h4 : distances_to_vertices O AO BO CO) :
  P / 2 < AO + BO + CO ∧ AO + BO + CO < P :=
sorry

end distances_sum_in_triangle_l34_34253


namespace tank_fill_time_l34_34254

-- Define the conditions
def start_time : ℕ := 1 -- 1 pm
def first_hour_rainfall : ℕ := 2 -- 2 inches rainfall in the first hour from 1 pm to 2 pm
def next_four_hours_rate : ℕ := 1 -- 1 inch/hour rainfall rate from 2 pm to 6 pm
def following_rate : ℕ := 3 -- 3 inches/hour rainfall rate from 6 pm onwards
def tank_height : ℕ := 18 -- 18 inches tall fish tank

-- Define what needs to be proved
theorem tank_fill_time : 
  ∃ t : ℕ, t = 22 ∧ (tank_height ≤ (first_hour_rainfall + 4 * next_four_hours_rate + (t - 6)) + (t - 6 - 4) * following_rate) := 
by 
  sorry

end tank_fill_time_l34_34254


namespace new_cost_percentage_l34_34607

def cost (t b : ℝ) := t * b^5

theorem new_cost_percentage (t b : ℝ) : 
  let C := cost t b
  let W := cost (3 * t) (2 * b)
  W = 96 * C :=
by
  sorry

end new_cost_percentage_l34_34607


namespace opposite_of_neg_two_is_two_l34_34721

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l34_34721


namespace side_length_of_square_l34_34424

theorem side_length_of_square (d : ℝ) (s : ℝ) (h1 : d = 2 * Real.sqrt 2) (h2 : d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34424


namespace area_of_storm_eye_l34_34783

theorem area_of_storm_eye : 
  let large_quarter_circle_area := (1 / 4) * π * 5^2
  let small_circle_area := π * 2^2
  let storm_eye_area := large_quarter_circle_area - small_circle_area
  storm_eye_area = (9 * π) / 4 :=
by
  sorry

end area_of_storm_eye_l34_34783


namespace find_a_and_b_l34_34363

theorem find_a_and_b (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a * b - 9) : a = 3 ∧ b = 3 :=
by
  sorry

end find_a_and_b_l34_34363


namespace correct_statement_B_l34_34871

-- Definitions as per the conditions
noncomputable def total_students : ℕ := 6700
noncomputable def selected_students : ℕ := 300

-- Definitions as per the question
def is_population (n : ℕ) : Prop := n = 6700
def is_sample (m n : ℕ) : Prop := m = 300 ∧ n = 6700
def is_individual (m n : ℕ) : Prop := m < n
def is_census (m n : ℕ) : Prop := m = n

-- The statement that needs to be proved
theorem correct_statement_B : 
  is_sample selected_students total_students :=
by
  -- Proof steps would go here
  sorry

end correct_statement_B_l34_34871


namespace rhombus_area_l34_34468

-- Definition of a rhombus with given conditions
structure Rhombus where
  side : ℝ
  d1 : ℝ
  d2 : ℝ

noncomputable def Rhombus.area (r : Rhombus) : ℝ :=
  (r.d1 * r.d2) / 2

noncomputable example : Rhombus :=
{ side := 20,
  d1 := 16,
  d2 := 8 * Real.sqrt 21 }

theorem rhombus_area : 
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  Rhombus.area r = 64 * Real.sqrt 21 :=
by
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  sorry

end rhombus_area_l34_34468


namespace side_length_of_square_l34_34433

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l34_34433


namespace cream_butterfat_percentage_l34_34481

theorem cream_butterfat_percentage (x : ℝ) (h1 : 1 * (x / 100) + 3 * (5.5 / 100) = 4 * (6.5 / 100)) : 
  x = 9.5 :=
by
  sorry

end cream_butterfat_percentage_l34_34481


namespace necessary_but_not_sufficient_l34_34828

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (a > b - 1) ∧ ¬(a > b - 1 → a > b) :=
sorry

end necessary_but_not_sufficient_l34_34828


namespace first_number_remainder_one_l34_34855

theorem first_number_remainder_one (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 2023) :
  (∀ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 → (a % 3 ≠ b % 3 ∧ a % 3 ≠ c % 3 ∧ b % 3 ≠ c % 3))
  → (n % 3 = 1) :=
sorry

end first_number_remainder_one_l34_34855


namespace find_a_l34_34794

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Theorem to be proved
theorem find_a (a : ℝ) 
  (h1 : A a ∪ B a = {1, 3, a}) : a = 0 ∨ a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l34_34794


namespace opposite_of_neg_two_is_two_l34_34722

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l34_34722


namespace ratio_eq_l34_34367

variable (a b c d : ℚ)

theorem ratio_eq :
  (a / b = 5 / 2) →
  (c / d = 7 / 3) →
  (d / b = 5 / 4) →
  (a / c = 6 / 7) :=
by
  intros h1 h2 h3
  sorry

end ratio_eq_l34_34367


namespace total_cost_football_games_l34_34843

-- Define the initial conditions
def games_this_year := 14
def games_last_year := 29
def price_this_year := 45
def price_lowest := 40
def price_highest := 65
def one_third_games_last_year := games_last_year / 3
def one_fourth_games_last_year := games_last_year / 4

-- Define the assertions derived from the conditions
def games_lowest_price := 9  -- rounded down from games_last_year / 3
def games_highest_price := 7  -- rounded down from games_last_year / 4
def remaining_games := games_last_year - (games_lowest_price + games_highest_price)

-- Define the costs calculation
def cost_this_year := games_this_year * price_this_year
def cost_lowest_price_games := games_lowest_price * price_lowest
def cost_highest_price_games := games_highest_price * price_highest
def total_cost := cost_this_year + cost_lowest_price_games + cost_highest_price_games

-- The theorem statement
theorem total_cost_football_games (h1 : games_lowest_price = 9) (h2 : games_highest_price = 7) 
  (h3 : cost_this_year = 630) (h4 : cost_lowest_price_games = 360) (h5 : cost_highest_price_games = 455) :
  total_cost = 1445 :=
by
  -- Since this is just the statement, we can simply put 'sorry' here.
  sorry

end total_cost_football_games_l34_34843


namespace tan_315_eq_neg1_l34_34946

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l34_34946


namespace distribute_stickers_l34_34516

-- Definitions based on conditions
def stickers : ℕ := 10
def sheets : ℕ := 5

-- Theorem stating the equivalence of distributing the stickers onto sheets
theorem distribute_stickers :
  (Nat.choose (stickers + sheets - 1) (sheets - 1)) = 1001 :=
by 
  -- Here is where the proof would go, but we skip it with sorry for the purpose of this task
  sorry

end distribute_stickers_l34_34516


namespace expression_equals_a5_l34_34881

theorem expression_equals_a5 (a : ℝ) : a^4 * a = a^5 := 
by sorry

end expression_equals_a5_l34_34881


namespace systematic_sampling_example_l34_34621

theorem systematic_sampling_example (rows seats : ℕ) (all_seats_filled : Prop) (chosen_seat : ℕ):
  rows = 50 ∧ seats = 60 ∧ all_seats_filled ∧ chosen_seat = 18 → sampling_method = "systematic_sampling" :=
by
  sorry

end systematic_sampling_example_l34_34621


namespace value_of_m_l34_34224

-- Problem Statement
theorem value_of_m (m : ℝ) : (∃ x : ℝ, (m-2)*x^(|m|-1) + 16 = 0 ∧ |m| - 1 = 1) → m = -2 :=
by
  sorry

end value_of_m_l34_34224


namespace star_7_3_eq_neg_5_l34_34339

def star_operation (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem star_7_3_eq_neg_5 : star_operation 7 3 = -5 :=
by
  -- proof goes here
  sorry

end star_7_3_eq_neg_5_l34_34339


namespace product_of_bc_l34_34451

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l34_34451


namespace mod_37_5_l34_34862

theorem mod_37_5 : 37 % 5 = 2 := 
by
  sorry

end mod_37_5_l34_34862


namespace parabola_passes_through_A_C_l34_34799

theorem parabola_passes_through_A_C : ∃ (a b : ℝ), (2 = a * 1^2 + b * 1 + 1) ∧ (1 = a * 2^2 + b * 2 + 1) :=
by {
  sorry
}

end parabola_passes_through_A_C_l34_34799


namespace original_gross_profit_percentage_l34_34304

theorem original_gross_profit_percentage 
  (C : ℝ) -- Cost of the product
  (h1 : 1.15 * C = 92) -- New selling price equation implying 15% gross profit increase
  (h2 : 88 - C = 8) -- Original gross profit in dollar terms
  : ((88 - C) / C) * 100 = 10 := 
sorry

end original_gross_profit_percentage_l34_34304


namespace tan_315_eq_neg1_l34_34950

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l34_34950


namespace validate_expression_l34_34699

-- Define the expression components
def a := 100
def b := 6
def c := 7
def d := 52
def e := 8
def f := 9

-- Define the expression using the given numbers and operations
def expression := (a - b) * c - d + e + f

-- The theorem statement asserting that the expression evaluates to 623
theorem validate_expression : expression = 623 := 
by
  -- Proof would go here
  sorry

end validate_expression_l34_34699


namespace quadratic_decreasing_condition_l34_34669

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m)^2 - 1

-- Conditions and the proof problem wrapped as a theorem statement
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ 3 → quadratically_decreasing x m) → m ≥ 3 :=
sorry

-- Helper function defining the decreasing condition
def quadratically_decreasing (x m : ℝ) : Prop :=
∀ y : ℝ, y < x → quadratic_function y m > quadratic_function x m

end quadratic_decreasing_condition_l34_34669


namespace tommy_initial_balloons_l34_34740

theorem tommy_initial_balloons :
  ∃ x : ℝ, x + 78.5 = 132.25 ∧ x = 53.75 := by
  sorry

end tommy_initial_balloons_l34_34740


namespace vincent_spent_224_l34_34151

-- Defining the given conditions as constants
def num_books_animal : ℕ := 10
def num_books_outer_space : ℕ := 1
def num_books_trains : ℕ := 3
def cost_per_book : ℕ := 16

-- Summarizing the total number of books
def total_books : ℕ := num_books_animal + num_books_outer_space + num_books_trains
-- Calculating the total cost
def total_cost : ℕ := total_books * cost_per_book

-- Lean statement to prove that Vincent spent $224
theorem vincent_spent_224 : total_cost = 224 := by
  sorry

end vincent_spent_224_l34_34151


namespace smallest_rat_num_l34_34188

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end smallest_rat_num_l34_34188


namespace tan_315_eq_neg1_l34_34913

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l34_34913


namespace find_cost_price_l34_34861

variable (C : ℝ)

theorem find_cost_price (h : 56 - C = C - 42) : C = 49 :=
by
  sorry

end find_cost_price_l34_34861


namespace equal_probability_among_children_l34_34152

theorem equal_probability_among_children
    (n : ℕ := 100)
    (p : ℝ := 0.232818)
    (k : ℕ := 18)
    (h_pos : 0 < p)
    (h_lt : p < 1)
    (num_outcomes : ℕ := 2^k) :
  ∃ (dist : Fin n → Fin num_outcomes),
    ∀ i : Fin num_outcomes, ∃ j : Fin n, dist j = i ∧ p ^ k * (1 - p) ^ (num_outcomes - k) = 1 / n :=
by
  sorry

end equal_probability_among_children_l34_34152


namespace sector_radius_of_circle_l34_34416

theorem sector_radius_of_circle :
  ∃ r : ℝ, let θ := 42 * Real.pi / 180 in
  let area := 82.5 in
  θ / (2 * Real.pi) * Real.pi * r^2 = area ∧ r = 15 :=
begin
  sorry, -- The proof is omitted.
end

end sector_radius_of_circle_l34_34416


namespace tan_315_eq_neg1_l34_34964

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34964


namespace vibrations_proof_l34_34696

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end vibrations_proof_l34_34696


namespace certain_number_N_l34_34494

theorem certain_number_N (G N : ℕ) (hG : G = 127)
  (h₁ : ∃ k : ℕ, N = G * k + 10)
  (h₂ : ∃ m : ℕ, 2045 = G * m + 13) :
  N = 2042 :=
sorry

end certain_number_N_l34_34494


namespace total_journey_time_eq_5_l34_34749

-- Define constants for speed and times
def speed1 : ℕ := 40
def speed2 : ℕ := 60
def total_distance : ℕ := 240
def time1 : ℕ := 3

-- Noncomputable definition to avoid computation issues
noncomputable def journey_time : ℕ :=
  let distance1 := speed1 * time1
  let distance2 := total_distance - distance1
  let time2 := distance2 / speed2
  time1 + time2

-- Theorem to state the total journey time
theorem total_journey_time_eq_5 : journey_time = 5 := by
  sorry

end total_journey_time_eq_5_l34_34749


namespace most_likely_outcome_is_draw_l34_34618

variable (P_A_wins : ℝ) (P_A_not_loses : ℝ)

def P_draw (P_A_wins P_A_not_loses : ℝ) : ℝ := 
  P_A_not_loses - P_A_wins

def P_B_wins (P_A_not_loses P_A_wins : ℝ) : ℝ :=
  1 - P_A_not_loses

theorem most_likely_outcome_is_draw 
  (h₁: P_A_wins = 0.3) 
  (h₂: P_A_not_loses = 0.7)
  (h₃: 0 ≤ P_A_wins) 
  (h₄: P_A_wins ≤ 1) 
  (h₅: 0 ≤ P_A_not_loses) 
  (h₆: P_A_not_loses ≤ 1) : 
  max (P_A_wins) (max (P_B_wins P_A_not_loses P_A_wins) (P_draw P_A_wins P_A_not_loses)) = P_draw P_A_wins P_A_not_loses :=
by
  sorry

end most_likely_outcome_is_draw_l34_34618


namespace highway_extension_l34_34176

def initial_length : ℕ := 200
def final_length : ℕ := 650
def first_day_construction : ℕ := 50
def second_day_construction : ℕ := 3 * first_day_construction
def total_construction : ℕ := first_day_construction + second_day_construction
def total_extension_needed : ℕ := final_length - initial_length
def miles_still_needed : ℕ := total_extension_needed - total_construction

theorem highway_extension : miles_still_needed = 250 := by
  sorry

end highway_extension_l34_34176


namespace number_of_distinct_sequences_l34_34082

theorem number_of_distinct_sequences :
  let letters := ["B", "A", "N", "A", "N", "A"]
  let possible_sequences := { s : List String // s.head? = some "B" ∧ s.getLast? = some "N" ∧ s.dedup.length = s.length ∧ ∀ c ∈ s, c ∈ letters }
  true := possible_sequences.card = 3 :=
sorry

end number_of_distinct_sequences_l34_34082


namespace tom_total_payment_l34_34002

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l34_34002


namespace cubic_yard_to_cubic_meter_and_liters_l34_34081

theorem cubic_yard_to_cubic_meter_and_liters :
  (1 : ℝ) * (0.9144 : ℝ)^3 = 0.764554 ∧ 0.764554 * 1000 = 764.554 :=
by
  sorry

end cubic_yard_to_cubic_meter_and_liters_l34_34081


namespace library_growth_rate_l34_34001

theorem library_growth_rate (C_2022 C_2024: ℝ) (h₁ : C_2022 = 100000) (h₂ : C_2024 = 144000) :
  ∃ x : ℝ, (1 + x) ^ 2 = C_2024 / C_2022 ∧ x = 0.2 := 
by {
  sorry
}

end library_growth_rate_l34_34001


namespace concatenated_number_divisible_by_37_l34_34138

theorem concatenated_number_divisible_by_37
  (a b : ℕ) (ha : 100 ≤ a ∧ a ≤ 999) (hb : 100 ≤ b ∧ b ≤ 999)
  (h₁ : a % 37 ≠ 0) (h₂ : b % 37 ≠ 0) (h₃ : (a + b) % 37 = 0) :
  (1000 * a + b) % 37 = 0 :=
sorry

end concatenated_number_divisible_by_37_l34_34138


namespace number_of_hydrogen_atoms_l34_34030

/-- 
A compound has a certain number of Hydrogen, 1 Chromium, and 4 Oxygen atoms. 
The molecular weight of the compound is 118. How many Hydrogen atoms are in the compound?
-/
theorem number_of_hydrogen_atoms
  (H Cr O : ℕ)
  (mw_H : ℕ := 1)
  (mw_Cr : ℕ := 52)
  (mw_O : ℕ := 16)
  (H_weight : ℕ := H * mw_H)
  (Cr_weight : ℕ := 1 * mw_Cr)
  (O_weight : ℕ := 4 * mw_O)
  (total_weight : ℕ := 118)
  (weight_without_H : ℕ := Cr_weight + O_weight) 
  (H_weight_calculated : ℕ := total_weight - weight_without_H) :
  H = 2 :=
  by
    sorry

end number_of_hydrogen_atoms_l34_34030


namespace max_y_diff_eq_0_l34_34573

-- Definitions for the given conditions
def eq1 (x : ℝ) : ℝ := 4 - 2 * x + x^2
def eq2 (x : ℝ) : ℝ := 2 + 2 * x + x^2

-- Statement of the proof problem
theorem max_y_diff_eq_0 : 
  (∀ x y, eq1 x = y ∧ eq2 x = y → y = (13 / 4)) →
  ∀ (x1 x2 : ℝ), (∃ y1 y2, eq1 x1 = y1 ∧ eq2 x1 = y1 ∧ eq1 x2 = y2 ∧ eq2 x2 = y2) → 
  (x1 = x2) → (y1 = y2) →
  0 = 0 := 
by
  sorry

end max_y_diff_eq_0_l34_34573


namespace tan_315_eq_neg1_l34_34925

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l34_34925


namespace new_average_increased_by_40_percent_l34_34276

theorem new_average_increased_by_40_percent 
  (n : ℕ) (initial_avg : ℝ) (initial_marks : ℝ) (new_marks : ℝ) (new_avg : ℝ)
  (h1 : n = 37)
  (h2 : initial_avg = 73)
  (h3 : initial_marks = (initial_avg * n))
  (h4 : new_marks = (initial_marks * 1.40))
  (h5 : new_avg = (new_marks / n)) :
  new_avg = 102.2 :=
sorry

end new_average_increased_by_40_percent_l34_34276


namespace arithmetic_sequence_solution_l34_34225

variable (a d : ℤ)
variable (n : ℕ)

/-- Given the following conditions:
1. The sum of the first three terms of an arithmetic sequence is -3.
2. The product of the first three terms is 8,
This theorem proves that:
1. The general term formula of the sequence is 3 * n - 7.
2. The sum of the first n terms is (3 / 2) * n ^ 2 - (11 / 2) * n.
-/
theorem arithmetic_sequence_solution
  (h1 : (a - d) + a + (a + d) = -3)
  (h2 : (a - d) * a * (a + d) = 8) :
  (∃ a d : ℤ, (∀ n : ℕ, (n ≥ 1) → (3 * n - 7 = a + (n - 1) * d) ∧ (∃ S : ℕ → ℤ, S n = (3 / 2) * n ^ 2 - (11 / 2) * n))) :=
by
  sorry

end arithmetic_sequence_solution_l34_34225


namespace tan_315_eq_neg_one_l34_34939

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l34_34939


namespace prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l34_34897

theorem prime_of_form_4k_plus_1_as_sum_of_two_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 4 * k + 1) :
  ∃ a b : ℤ, p = a^2 + b^2 :=
sorry

theorem prime_of_form_8k_plus_3_as_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 8 * k + 3) :
  ∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
sorry

end prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l34_34897


namespace percentage_increase_l34_34880

theorem percentage_increase (P Q : ℝ)
  (price_decreased : ∀ P', P' = 0.80 * P)
  (revenue_increased : ∀ R R', R = P * Q ∧ R' = 1.28000000000000025 * R)
  : ∃ Q', Q' = 1.6000000000000003125 * Q :=
by
  sorry

end percentage_increase_l34_34880


namespace number_of_initials_is_10000_l34_34810

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l34_34810


namespace shaded_region_area_l34_34908

def isosceles_triangle (AB AC BC : ℝ) (BAC : ℝ) : Prop :=
  AB = AC ∧ BAC = 120 ∧ BC = 32

def circle_with_diameter (diameter : ℝ) (radius : ℝ) : Prop :=
  radius = diameter / 2

theorem shaded_region_area :
  ∀ (AB AC BC : ℝ) (BAC : ℝ) (O : Type) (a b c : ℕ),
    isosceles_triangle AB AC BC BAC →
    circle_with_diameter BC 8 →
    (a = 43) ∧ (b = 128) ∧ (c = 3) →
    a + b + c = 174 :=
by
  sorry

end shaded_region_area_l34_34908


namespace marsupial_protein_l34_34035

theorem marsupial_protein (absorbed : ℝ) (percent_absorbed : ℝ) (consumed : ℝ) :
  absorbed = 16 ∧ percent_absorbed = 0.4 → consumed = 40 :=
by
  sorry

end marsupial_protein_l34_34035


namespace functional_eq_solution_l34_34341

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_solution f) :
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 :=
sorry

end functional_eq_solution_l34_34341


namespace cost_per_amulet_is_30_l34_34052

variable (days_sold : ℕ := 2)
variable (amulets_per_day : ℕ := 25)
variable (price_per_amulet : ℕ := 40)
variable (faire_percentage : ℕ := 10)
variable (profit : ℕ := 300)

def total_amulets_sold := days_sold * amulets_per_day
def total_revenue := total_amulets_sold * price_per_amulet
def faire_cut := total_revenue * faire_percentage / 100
def revenue_after_faire := total_revenue - faire_cut
def total_cost := revenue_after_faire - profit
def cost_per_amulet := total_cost / total_amulets_sold

theorem cost_per_amulet_is_30 : cost_per_amulet = 30 := by
  sorry

end cost_per_amulet_is_30_l34_34052


namespace num_12_digit_with_consecutive_ones_l34_34517

theorem num_12_digit_with_consecutive_ones :
  let total := 3^12
  let F12 := 985
  total - F12 = 530456 :=
by
  let total := 3^12
  let F12 := 985
  have h : total - F12 = 530456
  sorry
  exact h

end num_12_digit_with_consecutive_ones_l34_34517


namespace tan_315_eq_neg_one_l34_34994

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l34_34994


namespace four_letter_initial_sets_l34_34811

theorem four_letter_initial_sets : 
  (∃ (A B C D : Fin 10), true) → (10 * 10 * 10 * 10 = 10000) :=
by
  intro h,
  sorry

end four_letter_initial_sets_l34_34811


namespace smallest_abs_sum_of_products_l34_34832

noncomputable def g (x : ℝ) : ℝ := x^4 + 16 * x^3 + 69 * x^2 + 112 * x + 64

theorem smallest_abs_sum_of_products :
  (∀ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 → 
   |w1 * w2 + w3 * w4| ≥ 8) ∧ 
  (∃ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 ∧ 
   |w1 * w2 + w3 * w4| = 8) :=
sorry

end smallest_abs_sum_of_products_l34_34832


namespace island_population_percentage_l34_34323

theorem island_population_percentage :
  -- Defining conditions
  (∀ a b : ℕ, (a + b ≠ 0) → (a.toRat / (a + b).toRat = 65 / 100) →
   ∀ b c : ℕ, (b + c ≠ 0) → (b.toRat / (b + c).toRat = 70 / 100) →
   ∀ c d : ℕ, (c + d ≠ 0) → (c.toRat / (c + d).toRat = 10 / 100) →
  
  -- Correct answer based on conditions
  ∃ a b c d : ℕ, 
    let total := a + b + c + d in 
    total ≠ 0 ∧ 
    (d.toRat / total.toRat = 54 / 100)) := 
sorry

end island_population_percentage_l34_34323


namespace first_generation_tail_length_l34_34533

theorem first_generation_tail_length
  (length_first_gen : ℝ)
  (H : (1.25:ℝ) * (1.25:ℝ) * length_first_gen = 25) :
  length_first_gen = 16 := by
  sorry

end first_generation_tail_length_l34_34533


namespace tan_315_eq_neg1_l34_34987

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l34_34987


namespace find_x_squared_plus_y_squared_l34_34231

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l34_34231


namespace line_direction_vector_correct_l34_34622

theorem line_direction_vector_correct :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 1) ∧ 
  ∃ (v w : ℝ), (v = A ∧ w = B) :=
by
  sorry

end line_direction_vector_correct_l34_34622


namespace probability_of_license_expected_value_attempts_l34_34628

-- Lean statement for the probability of obtaining a driver's license
theorem probability_of_license (p1 p2 p3 : ℝ) (hp1 : p1 = 0.9) (hp2 : p2 = 0.7) (hp3 : p3 = 0.6) :
  p1 * p2 * p3 = 0.378 :=
by {
  rw [hp1, hp2, hp3], -- Substitute given probabilities
  norm_num, -- Evaluate the multiplication
}

-- Lean statement for the expected value of the number of attempts
theorem expected_value_attempts (p1 p2 p3 : ℝ) (hp1 : p1 = 0.1) (hp2 : p2 = 0.27) (hp3 : p3 = 0.63) :
  1 * p1 + 2 * p2 + 3 * p3 = 2.53 :=
by {
  rw [hp1, hp2, hp3], -- Substitute given probabilities
  norm_num, -- Evaluate the expression
}

end probability_of_license_expected_value_attempts_l34_34628


namespace fewer_seats_on_right_than_left_l34_34527

theorem fewer_seats_on_right_than_left : 
  ∀ (left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats : ℕ),
    left_seats = 15 →
    back_seat_capacity = 9 →
    people_per_seat = 3 →
    bus_capacity = 90 →
    right_seats = (bus_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat →
    fewer_seats = left_seats - right_seats →
    fewer_seats = 3 :=
by
  intros left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats
  sorry

end fewer_seats_on_right_than_left_l34_34527


namespace number_of_initials_is_10000_l34_34809

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l34_34809


namespace tan_315_eq_neg1_l34_34930

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l34_34930


namespace average_salary_of_all_workers_l34_34383

-- Definitions of conditions
def T : ℕ := 7
def total_workers : ℕ := 56
def W : ℕ := total_workers - T
def A_T : ℕ := 12000
def A_W : ℕ := 6000

-- Definition of total salary and average salary
def total_salary : ℕ := (T * A_T) + (W * A_W)

theorem average_salary_of_all_workers : total_salary / total_workers = 6750 := 
  by sorry

end average_salary_of_all_workers_l34_34383


namespace number_is_165_l34_34602

def is_between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def is_odd (n : ℕ) : Prop := n % 2 = 1
def contains_digit_5 (n : ℕ) : Prop := ∃ k : ℕ, 10^k * 5 ≤ n ∧ n < 10^(k+1) * 5
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem number_is_165 : 
  (is_between 165 144 169) ∧ 
  (is_odd 165) ∧ 
  (contains_digit_5 165) ∧ 
  (is_divisible_by_3 165) :=
by 
  sorry 

end number_is_165_l34_34602


namespace perpendicular_lines_solve_for_a_l34_34823

theorem perpendicular_lines_solve_for_a :
  ∀ (a : ℝ), 
  ((3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0) → 
  (a = 0 ∨ a = 1) :=
by
  intro a h
  sorry

end perpendicular_lines_solve_for_a_l34_34823


namespace paula_candies_l34_34552

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l34_34552


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l34_34261

variable (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50)

theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_52 : (Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52) :=
by
  sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_52_l34_34261


namespace angles_with_same_terminal_side_l34_34285

theorem angles_with_same_terminal_side (k : ℤ) : 
  (∃ (α : ℝ), α = -437 + k * 360) ↔ (∃ (β : ℝ), β = 283 + k * 360) := 
by
  sorry

end angles_with_same_terminal_side_l34_34285


namespace find_x_l34_34824

def binop (a b : ℤ) : ℤ := a * b + a + b + 2

theorem find_x :
  ∃ x : ℤ, binop x 3 = 1 ∧ x = -1 :=
by
  sorry

end find_x_l34_34824


namespace calculate_expression_l34_34490

variable {a : ℝ}

theorem calculate_expression (h₁ : a ≠ 0) (h₂ : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := 
sorry

end calculate_expression_l34_34490


namespace side_length_of_square_l34_34439

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l34_34439


namespace percentage_increase_l34_34238

theorem percentage_increase (A B x y : ℝ) (h1 : A / B = (5 * y^2) / (6 * x)) (h2 : 2 * x + 3 * y = 42) :  
  (B - A) / A * 100 = ((126 - 9 * y - 5 * y^2) / (5 * y^2)) * 100 :=
by
  sorry

end percentage_increase_l34_34238


namespace sum_first_six_terms_l34_34250

variable (a1 q : ℤ)
variable (n : ℕ)

noncomputable def geometric_sum (a1 q : ℤ) (n : ℕ) : ℤ :=
  a1 * (1 - q^n) / (1 - q)

theorem sum_first_six_terms :
  geometric_sum (-1) 2 6 = 63 :=
sorry

end sum_first_six_terms_l34_34250


namespace proof_x_square_ab_a_square_l34_34233

variable {x b a : ℝ}

/-- Given that x < b < a < 0 where x, b, and a are real numbers, we need to prove x^2 > ab > a^2. -/
theorem proof_x_square_ab_a_square (hx : x < b) (hb : b < a) (ha : a < 0) :
  x^2 > ab ∧ ab > a^2 := 
by
  sorry

end proof_x_square_ab_a_square_l34_34233


namespace total_distance_travelled_l34_34080

theorem total_distance_travelled (distance_to_market : ℕ) (travel_time_minutes : ℕ) (speed_mph : ℕ) 
  (h1 : distance_to_market = 30) 
  (h2 : travel_time_minutes = 30) 
  (h3 : speed_mph = 20) : 
  (distance_to_market + ((travel_time_minutes / 60) * speed_mph) = 40) :=
by
  sorry

end total_distance_travelled_l34_34080


namespace simplify_and_evaluate_division_l34_34564

theorem simplify_and_evaluate_division (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a ^ 3 / (a ^ 2 - 4 * a + 4)) = 1 / 3 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_division_l34_34564


namespace exponential_ordering_l34_34216

noncomputable def a := (0.4:ℝ)^(0.3:ℝ)
noncomputable def b := (0.3:ℝ)^(0.4:ℝ)
noncomputable def c := (0.3:ℝ)^(-0.2:ℝ)

theorem exponential_ordering : b < a ∧ a < c := by
  sorry

end exponential_ordering_l34_34216


namespace salt_cups_l34_34695

theorem salt_cups (S : ℕ) (h1 : 8 = S + 1) : S = 7 := by
  -- Problem conditions
  -- 1. The recipe calls for 8 cups of sugar.
  -- 2. Mary needs to add 1 more cup of sugar than cups of salt.
  -- This corresponds to h1.

  -- Prove S = 7
  sorry

end salt_cups_l34_34695


namespace necessary_but_not_sufficient_l34_34473

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 2 → (x^2 - x - 2 >= 0) ∨ (x >= -1 ∧ x < 2)) ∧ ((-1 < x ∧ x < 2) → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_l34_34473


namespace complex_number_condition_l34_34670

theorem complex_number_condition (b : ℝ) :
  (2 + b) / 5 = (2 * b - 1) / 5 → b = 3 :=
by
  sorry

end complex_number_condition_l34_34670


namespace sum_of_coeffs_eq_one_l34_34371

theorem sum_of_coeffs_eq_one (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) (x : ℝ) :
  (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1 :=
  sorry

end sum_of_coeffs_eq_one_l34_34371


namespace tan_315_eq_neg1_l34_34965

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l34_34965


namespace arithmetic_geometric_sequence_a1_l34_34362

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ)
  (h1 : a 1 + a 6 = 11)
  (h2 : a 3 * a 4 = 32 / 9) :
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 :=
sorry

end arithmetic_geometric_sequence_a1_l34_34362


namespace cube_painted_probability_l34_34175

theorem cube_painted_probability :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 76
  let total_ways := Nat.choose total_cubes 2
  let favorable_ways := cubes_with_3_faces * cubes_with_no_faces
  let probability := (favorable_ways : ℚ) / total_ways
  probability = (2 : ℚ) / 205 :=
by
  sorry

end cube_painted_probability_l34_34175


namespace sum_xyz_eq_two_l34_34358

-- Define the variables x, y, and z to be real numbers
variables (x y z : ℝ)

-- Given condition
def condition : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0

-- The theorem to prove
theorem sum_xyz_eq_two (h : condition x y z) : x + y + z = 2 :=
sorry

end sum_xyz_eq_two_l34_34358


namespace opposites_of_each_other_l34_34522

theorem opposites_of_each_other (a b : ℚ) (h : a + b = 0) : a = -b :=
  sorry

end opposites_of_each_other_l34_34522


namespace opposite_of_2_is_minus_2_l34_34444

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end opposite_of_2_is_minus_2_l34_34444


namespace evaluate_expression_l34_34049

theorem evaluate_expression : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end evaluate_expression_l34_34049


namespace probability_order_correct_l34_34793

inductive Phenomenon
| Certain
| VeryLikely
| Possible
| Impossible
| NotVeryLikely

open Phenomenon

def probability_order : Phenomenon → ℕ
| Certain       => 5
| VeryLikely    => 4
| Possible      => 3
| NotVeryLikely => 2
| Impossible    => 1

theorem probability_order_correct :
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] =
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] :=
by
  -- skips the proof
  sorry

end probability_order_correct_l34_34793


namespace hulk_jump_kilometer_l34_34274

theorem hulk_jump_kilometer (n : ℕ) (h : ∀ n : ℕ, n ≥ 1 → (2^(n-1) : ℕ) ≤ 1000 → n-1 < 10) : n = 11 :=
by
  sorry

end hulk_jump_kilometer_l34_34274


namespace find_b_from_quadratic_l34_34265

theorem find_b_from_quadratic (b n : ℤ)
  (h1 : b > 0)
  (h2 : (x : ℤ) → (x + n)^2 - 6 = x^2 + b * x + 19) :
  b = 10 :=
sorry

end find_b_from_quadratic_l34_34265


namespace part1_sales_volume_part2_price_reduction_l34_34692

noncomputable def daily_sales_volume (x : ℝ) : ℝ :=
  100 + 200 * x

noncomputable def profit_eq (x : ℝ) : Prop :=
  (4 - 2 - x) * (100 + 200 * x) = 300

theorem part1_sales_volume (x : ℝ) : daily_sales_volume x = 100 + 200 * x :=
sorry

theorem part2_price_reduction (hx : profit_eq (1 / 2)) : 1 / 2 = 1 / 2 :=
sorry

end part1_sales_volume_part2_price_reduction_l34_34692


namespace inequality_no_real_solutions_l34_34679

theorem inequality_no_real_solutions (a b : ℝ) 
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : 
  |b| ≤ 1 :=
sorry

end inequality_no_real_solutions_l34_34679


namespace solution_set_of_inequality_l34_34287

open Set Real

theorem solution_set_of_inequality :
  {x : ℝ | sqrt (x + 3) > 3 - x} = {x : ℝ | 1 < x} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end solution_set_of_inequality_l34_34287


namespace expected_value_equals_51_l34_34309

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l34_34309


namespace victoria_gym_sessions_l34_34874

-- Define the initial conditions
def starts_on_monday := true
def sessions_per_two_week_cycle := 6
def total_sessions := 30

-- Define the sought day of the week when all gym sessions are completed
def final_day := "Thursday"

-- The theorem stating the problem
theorem victoria_gym_sessions : 
  starts_on_monday →
  sessions_per_two_week_cycle = 6 →
  total_sessions = 30 →
  final_day = "Thursday" := 
by
  intros
  exact sorry

end victoria_gym_sessions_l34_34874


namespace value_of_a_minus_b_l34_34576

theorem value_of_a_minus_b (a b : ℝ)
  (h1 : ∃ (x : ℝ), x = 3 ∧ (ax / (x - 1)) = 1)
  (h2 : ∀ (x : ℝ), (ax / (x - 1)) < 1 ↔ (x < b ∨ x > 3)) :
  a - b = -1 / 3 :=
by
  sorry

end value_of_a_minus_b_l34_34576


namespace groups_of_three_in_class_of_fifteen_l34_34671

theorem groups_of_three_in_class_of_fifteen : 
  ∀ (n k : ℕ), n = 15 → k = 3 → nat.choose n k = 455 :=
by
  intros n k hn hk
  rw [hn, hk, nat.choose] -- Using the combination formula for computations
  sorry

end groups_of_three_in_class_of_fifteen_l34_34671


namespace expected_value_equals_51_l34_34310

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l34_34310


namespace paula_candies_distribution_l34_34554

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end paula_candies_distribution_l34_34554


namespace problem1_problem2_l34_34168

-- Definitions and Lean statement for Problem 1
noncomputable def curve1 (x : ℝ) : ℝ := x / (2 * x - 1)
def point1 : ℝ × ℝ := (1, 1)
noncomputable def tangent_line1 (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem1 : tangent_line1 (point1.fst) (curve1 (point1.fst)) :=
sorry -- proof goes here

-- Definitions and Lean statement for Problem 2
def parabola (x : ℝ) : ℝ := x^2
def point2 : ℝ × ℝ := (2, 3)
noncomputable def tangent_line2a (x y : ℝ) : Prop := 2 * x - y - 1 = 0
noncomputable def tangent_line2b (x y : ℝ) : Prop := 6 * x - y - 9 = 0

theorem problem2 : (tangent_line2a point2.fst point2.snd ∨ tangent_line2b point2.fst point2.snd) :=
sorry -- proof goes here

end problem1_problem2_l34_34168


namespace length_of_faster_train_l34_34298

/-- 
Let the faster train have a speed of 144 km per hour, the slower train a speed of 
72 km per hour, and the time taken for the faster train to cross a man in the 
slower train be 19 seconds. Then the length of the faster train is 380 meters.
-/
theorem length_of_faster_train 
  (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_to_cross : ℝ)
  (h_speed_faster_train : speed_faster_train = 144) 
  (h_speed_slower_train : speed_slower_train = 72) 
  (h_time_to_cross : time_to_cross = 19) :
  (speed_faster_train - speed_slower_train) * (5 / 18) * time_to_cross = 380 :=
by
  sorry

end length_of_faster_train_l34_34298


namespace dad_gave_nickels_l34_34870

-- Definitions
def original_nickels : ℕ := 9
def total_nickels_after : ℕ := 12

-- Theorem to be proven
theorem dad_gave_nickels {original_nickels total_nickels_after : ℕ} : 
    total_nickels_after - original_nickels = 3 := 
by
  /- Sorry proof omitted -/
  sorry

end dad_gave_nickels_l34_34870


namespace david_reading_time_l34_34640

theorem david_reading_time (total_time : ℕ) (math_time : ℕ) (spelling_time : ℕ) 
  (reading_time : ℕ) (h1 : total_time = 60) (h2 : math_time = 15) 
  (h3 : spelling_time = 18) (h4 : reading_time = total_time - (math_time + spelling_time)) : 
  reading_time = 27 := by
  sorry

end david_reading_time_l34_34640


namespace part_I_solution_set_part_II_range_of_a_l34_34369

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 6

-- (I) Prove the solution set of f(x) < 0 when a = 5
theorem part_I_solution_set : 
  (∀ x : ℝ, f x 5 < 0 ↔ (-3 < x ∧ x < -2)) := by
  sorry

-- (II) Prove the range of a such that f(x) > 0 for all x ∈ ℝ 
theorem part_II_range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f x a > 0) ↔ (-2*Real.sqrt 6 < a ∧ a < 2*Real.sqrt 6)) := by
  sorry

end part_I_solution_set_part_II_range_of_a_l34_34369


namespace number_of_children_per_seat_l34_34617

variable (children : ℕ) (seats : ℕ)

theorem number_of_children_per_seat (h1 : children = 58) (h2 : seats = 29) :
  children / seats = 2 := by
  sorry

end number_of_children_per_seat_l34_34617
