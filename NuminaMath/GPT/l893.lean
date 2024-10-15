import Mathlib

namespace NUMINAMATH_GPT_carson_gold_stars_l893_89394

theorem carson_gold_stars (gold_stars_yesterday gold_stars_today : ℕ) (h1 : gold_stars_yesterday = 6) (h2 : gold_stars_today = 9) : 
  gold_stars_yesterday + gold_stars_today = 15 := 
by
  sorry

end NUMINAMATH_GPT_carson_gold_stars_l893_89394


namespace NUMINAMATH_GPT_exist_indices_eq_l893_89304

theorem exist_indices_eq (p q n : ℕ) (x : ℕ → ℤ) 
    (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_n : 0 < n) 
    (h_pq_n : p + q < n) 
    (h_x0 : x 0 = 0) 
    (h_xn : x n = 0) 
    (h_step : ∀ i, 1 ≤ i ∧ i ≤ n → (x i - x (i - 1) = p ∨ x i - x (i - 1) = -q)) :
    ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end NUMINAMATH_GPT_exist_indices_eq_l893_89304


namespace NUMINAMATH_GPT_correct_factorization_l893_89371

variable (x y : ℝ)

theorem correct_factorization :
  x^2 - 2 * x * y + x = x * (x - 2 * y + 1) :=
by sorry

end NUMINAMATH_GPT_correct_factorization_l893_89371


namespace NUMINAMATH_GPT_compute_expression_l893_89331

theorem compute_expression : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l893_89331


namespace NUMINAMATH_GPT_hens_count_l893_89318

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 144) 
  (h3 : H ≥ 10) (h4 : C ≥ 5) : H = 24 :=
by
  sorry

end NUMINAMATH_GPT_hens_count_l893_89318


namespace NUMINAMATH_GPT_number_of_customers_who_tipped_is_3_l893_89369

-- Definitions of conditions
def charge_per_lawn : ℤ := 33
def lawns_mowed : ℤ := 16
def total_earnings : ℤ := 558
def tip_per_customer : ℤ := 10

-- Calculate intermediate values
def earnings_from_mowing : ℤ := lawns_mowed * charge_per_lawn
def earnings_from_tips : ℤ := total_earnings - earnings_from_mowing
def number_of_tips : ℤ := earnings_from_tips / tip_per_customer

-- Theorem stating our proof
theorem number_of_customers_who_tipped_is_3 : number_of_tips = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_customers_who_tipped_is_3_l893_89369


namespace NUMINAMATH_GPT_find_constant_a_l893_89340

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (Real.exp x - 1)

theorem find_constant_a (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_constant_a_l893_89340


namespace NUMINAMATH_GPT_problem_solution_l893_89325

theorem problem_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := 
  sorry

end NUMINAMATH_GPT_problem_solution_l893_89325


namespace NUMINAMATH_GPT_field_trip_savings_l893_89376

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end NUMINAMATH_GPT_field_trip_savings_l893_89376


namespace NUMINAMATH_GPT_pave_hall_with_stones_l893_89313

def hall_length_m : ℕ := 36
def hall_breadth_m : ℕ := 15
def stone_length_dm : ℕ := 4
def stone_breadth_dm : ℕ := 5

def to_decimeters (m : ℕ) : ℕ := m * 10

def hall_length_dm : ℕ := to_decimeters hall_length_m
def hall_breadth_dm : ℕ := to_decimeters hall_breadth_m

def hall_area_dm2 : ℕ := hall_length_dm * hall_breadth_dm
def stone_area_dm2 : ℕ := stone_length_dm * stone_breadth_dm

def number_of_stones_required : ℕ := hall_area_dm2 / stone_area_dm2

theorem pave_hall_with_stones :
  number_of_stones_required = 2700 :=
sorry

end NUMINAMATH_GPT_pave_hall_with_stones_l893_89313


namespace NUMINAMATH_GPT_temperature_43_l893_89384

theorem temperature_43 (T W Th F : ℝ)
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : T = 37) : F = 43 :=
by
  sorry

end NUMINAMATH_GPT_temperature_43_l893_89384


namespace NUMINAMATH_GPT_first_place_beats_joe_by_two_points_l893_89389

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end NUMINAMATH_GPT_first_place_beats_joe_by_two_points_l893_89389


namespace NUMINAMATH_GPT_solve_system_eqns_l893_89343

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eqns_l893_89343


namespace NUMINAMATH_GPT_sqrt_sum_gt_l893_89326

theorem sqrt_sum_gt (a b : ℝ) (ha : a = 2) (hb : b = 3) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_gt_l893_89326


namespace NUMINAMATH_GPT_bert_puzzle_days_l893_89324

noncomputable def words_per_pencil : ℕ := 1050
noncomputable def words_per_puzzle : ℕ := 75

theorem bert_puzzle_days : words_per_pencil / words_per_puzzle = 14 := by
  sorry

end NUMINAMATH_GPT_bert_puzzle_days_l893_89324


namespace NUMINAMATH_GPT_laura_owes_correct_amount_l893_89365

def principal : ℝ := 35
def annual_rate : ℝ := 0.07
def time_years : ℝ := 1
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T
def total_amount_owed (P : ℝ) (I : ℝ) : ℝ := P + I

theorem laura_owes_correct_amount :
  total_amount_owed principal (interest principal annual_rate time_years) = 37.45 :=
sorry

end NUMINAMATH_GPT_laura_owes_correct_amount_l893_89365


namespace NUMINAMATH_GPT_sqrt_continued_fraction_l893_89333

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_sqrt_continued_fraction_l893_89333


namespace NUMINAMATH_GPT_second_movie_time_difference_l893_89367

def first_movie_length := 90 -- 1 hour and 30 minutes in minutes
def popcorn_time := 10 -- Time spent making popcorn in minutes
def fries_time := 2 * popcorn_time -- Time spent making fries in minutes
def total_time := 4 * 60 -- Total time for cooking and watching movies in minutes

theorem second_movie_time_difference :
  (total_time - (popcorn_time + fries_time + first_movie_length)) - first_movie_length = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_movie_time_difference_l893_89367


namespace NUMINAMATH_GPT_m_eq_n_is_necessary_but_not_sufficient_l893_89303

noncomputable def circle_condition (m n : ℝ) : Prop :=
  m = n ∧ m > 0

theorem m_eq_n_is_necessary_but_not_sufficient 
  (m n : ℝ) :
  (circle_condition m n → mx^2 + ny^2 = 3 → False) ∧
  (mx^2 + ny^2 = 3 → circle_condition m n) :=
by 
  sorry

end NUMINAMATH_GPT_m_eq_n_is_necessary_but_not_sufficient_l893_89303


namespace NUMINAMATH_GPT_billy_ate_72_cherries_l893_89337

-- Definitions based on conditions:
def initial_cherries : Nat := 74
def remaining_cherries : Nat := 2

-- Problem: How many cherries did Billy eat?
def cherries_eaten := initial_cherries - remaining_cherries

theorem billy_ate_72_cherries : cherries_eaten = 72 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_billy_ate_72_cherries_l893_89337


namespace NUMINAMATH_GPT_simplify_and_rationalize_l893_89374

noncomputable def simplify_expr : ℝ :=
  1 / (1 - (1 / (Real.sqrt 5 - 2)))

theorem simplify_and_rationalize :
  simplify_expr = (1 - Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l893_89374


namespace NUMINAMATH_GPT_fgh_deriv_at_0_l893_89335

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- Function Values at x = 0
axiom f_zero : f 0 = 1
axiom g_zero : g 0 = 2
axiom h_zero : h 0 = 3

-- Derivatives of the pairwise products at x = 0
axiom d_gh_zero : (deriv (λ x => g x * h x)) 0 = 4
axiom d_hf_zero : (deriv (λ x => h x * f x)) 0 = 5
axiom d_fg_zero : (deriv (λ x => f x * g x)) 0 = 6

-- We need to prove that the derivative of the product of f, g, h at x = 0 is 16
theorem fgh_deriv_at_0 : (deriv (λ x => f x * g x * h x)) 0 = 16 := by
  sorry

end NUMINAMATH_GPT_fgh_deriv_at_0_l893_89335


namespace NUMINAMATH_GPT_elvins_first_month_bill_l893_89368

variable (F C : ℕ)

def total_bill_first_month := F + C
def total_bill_second_month := F + 2 * C

theorem elvins_first_month_bill :
  total_bill_first_month F C = 46 ∧
  total_bill_second_month F C = 76 ∧
  total_bill_second_month F C - total_bill_first_month F C = 30 →
  total_bill_first_month F C = 46 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_elvins_first_month_bill_l893_89368


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_l893_89393

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_l893_89393


namespace NUMINAMATH_GPT_ab_cd_eq_one_l893_89310

theorem ab_cd_eq_one (a b c d : ℕ) (p : ℕ) 
  (h_div_a : a % p = 0)
  (h_div_b : b % p = 0)
  (h_div_c : c % p = 0)
  (h_div_d : d % p = 0)
  (h_div_ab_cd : (a * b - c * d) % p = 0) : 
  (a * b - c * d) = 1 :=
sorry

end NUMINAMATH_GPT_ab_cd_eq_one_l893_89310


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l893_89354

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 2 * x + 1 > 0) ↔ (a > 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l893_89354


namespace NUMINAMATH_GPT_jack_jill_next_in_step_l893_89348

theorem jack_jill_next_in_step (stride_jack : ℕ) (stride_jill : ℕ) : 
  stride_jack = 64 → stride_jill = 56 → Nat.lcm stride_jack stride_jill = 448 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_jack_jill_next_in_step_l893_89348


namespace NUMINAMATH_GPT_total_time_in_range_l893_89311

-- Definitions for the problem conditions
def section1 := 240 -- km
def section2 := 300 -- km
def section3 := 400 -- km

def speed1 := 40 -- km/h
def speed2 := 75 -- km/h
def speed3 := 80 -- km/h

-- The time it takes to cover a section at a certain speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Total time to cover all sections with different speed assignments
def total_time (s1 s2 s3 v1 v2 v3 : ℕ) : ℕ :=
  time s1 v1 + time s2 v2 + time s3 v3

-- Prove that the total time is within the range [15, 17]
theorem total_time_in_range :
  (total_time section1 section2 section3 speed3 speed2 speed1 = 15) ∧
  (total_time section1 section2 section3 speed1 speed2 speed3 = 17) →
  ∃ (T : ℕ), 15 ≤ T ∧ T ≤ 17 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_time_in_range_l893_89311


namespace NUMINAMATH_GPT_possible_days_l893_89327

namespace AnyaVanyaProblem

-- Conditions
def AnyaLiesOn (d : String) : Prop := d = "Tuesday" ∨ d = "Wednesday" ∨ d = "Thursday"
def AnyaTellsTruthOn (d : String) : Prop := ¬AnyaLiesOn d

def VanyaLiesOn (d : String) : Prop := d = "Thursday" ∨ d = "Friday" ∨ d = "Saturday"
def VanyaTellsTruthOn (d : String) : Prop := ¬VanyaLiesOn d

-- Statements
def AnyaStatement (d : String) : Prop := d = "Friday"
def VanyaStatement (d : String) : Prop := d = "Tuesday"

-- Proof problem
theorem possible_days (d : String) : 
  (AnyaTellsTruthOn d ↔ AnyaStatement d) ∧ (VanyaTellsTruthOn d ↔ VanyaStatement d)
  → d = "Tuesday" ∨ d = "Thursday" ∨ d = "Friday" := 
sorry

end AnyaVanyaProblem

end NUMINAMATH_GPT_possible_days_l893_89327


namespace NUMINAMATH_GPT_red_balls_count_is_correct_l893_89396

-- Define conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def non_red_purple_prob : ℝ := 0.9

-- Define the number of red balls
def number_of_red_balls (red_balls : ℕ) : Prop :=
  total_balls - (white_balls + green_balls + yellow_balls + purple_balls) = red_balls
  
-- The proof statement
theorem red_balls_count_is_correct : number_of_red_balls 7 := by
  sorry

end NUMINAMATH_GPT_red_balls_count_is_correct_l893_89396


namespace NUMINAMATH_GPT_cost_of_book_sold_at_loss_l893_89364

theorem cost_of_book_sold_at_loss
  (C1 C2 : ℝ)
  (total_cost : C1 + C2 = 360)
  (selling_price1 : 0.85 * C1 = 1.19 * C2) :
  C1 = 210 :=
sorry

end NUMINAMATH_GPT_cost_of_book_sold_at_loss_l893_89364


namespace NUMINAMATH_GPT_passengers_off_in_texas_l893_89320

variable (x : ℕ) -- number of passengers who got off in Texas
variable (initial_passengers : ℕ := 124)
variable (texas_boarding : ℕ := 24)
variable (nc_off : ℕ := 47)
variable (nc_boarding : ℕ := 14)
variable (virginia_passengers : ℕ := 67)

theorem passengers_off_in_texas {x : ℕ} :
  (initial_passengers - x + texas_boarding - nc_off + nc_boarding) = virginia_passengers → 
  x = 48 :=
by
  sorry

end NUMINAMATH_GPT_passengers_off_in_texas_l893_89320


namespace NUMINAMATH_GPT_frank_can_buy_seven_candies_l893_89345

def tickets_won_whackamole := 33
def tickets_won_skeeball := 9
def cost_per_candy := 6

theorem frank_can_buy_seven_candies : (tickets_won_whackamole + tickets_won_skeeball) / cost_per_candy = 7 :=
by
  sorry

end NUMINAMATH_GPT_frank_can_buy_seven_candies_l893_89345


namespace NUMINAMATH_GPT_toucan_count_correct_l893_89350

def initial_toucans : ℕ := 2
def toucans_joined : ℕ := 1
def total_toucans : ℕ := initial_toucans + toucans_joined

theorem toucan_count_correct : total_toucans = 3 := by
  sorry

end NUMINAMATH_GPT_toucan_count_correct_l893_89350


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_point_l893_89351

theorem line_intersects_x_axis_at_point : 
  let x1 := 3
  let y1 := 7
  let x2 := -1
  let y2 := 3
  let m := (y2 - y1) / (x2 - x1) -- slope formula
  let b := y1 - m * x1        -- y-intercept formula
  let x_intersect := -b / m  -- x-coordinate where the line intersects x-axis
  (x_intersect, 0) = (-4, 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_point_l893_89351


namespace NUMINAMATH_GPT_domain_of_function_l893_89359

theorem domain_of_function :
  ∀ x : ℝ, 3 * x - 2 > 0 ∧ 2 * x - 1 > 0 ↔ x > (2 / 3) := by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_function_l893_89359


namespace NUMINAMATH_GPT_sin_neg_600_eq_sqrt_3_div_2_l893_89353

theorem sin_neg_600_eq_sqrt_3_div_2 :
  Real.sin (-(600 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sin_neg_600_eq_sqrt_3_div_2_l893_89353


namespace NUMINAMATH_GPT_arithmetic_sequence_probability_l893_89328

def favorable_sequences : List (List ℕ) :=
  [[1, 2, 3], [1, 3, 5], [2, 3, 4], [2, 4, 6], [3, 4, 5], [4, 5, 6], 
   [3, 2, 1], [5, 3, 1], [4, 3, 2], [6, 4, 2], [5, 4, 3], [6, 5, 4], 
   [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := favorable_sequences.length

theorem arithmetic_sequence_probability : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_probability_l893_89328


namespace NUMINAMATH_GPT_geo_seq_product_l893_89387

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m / a 1

theorem geo_seq_product
  {a : ℕ → ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_seq : geometric_sequence a)
  (h_roots : ∃ x y, (x*x - 10 * x + 16 = 0) ∧ (y*y - 10 * y + 16 = 0) ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end NUMINAMATH_GPT_geo_seq_product_l893_89387


namespace NUMINAMATH_GPT_sin_neg_pi_l893_89302

theorem sin_neg_pi : Real.sin (-Real.pi) = 0 := by
  sorry

end NUMINAMATH_GPT_sin_neg_pi_l893_89302


namespace NUMINAMATH_GPT_hiring_manager_acceptance_l893_89388

theorem hiring_manager_acceptance 
    (average_age : ℤ) (std_dev : ℤ) (num_ages : ℤ)
    (applicant_ages_are_int : ∀ (x : ℤ), x ≥ (average_age - std_dev) ∧ x ≤ (average_age + std_dev)) :
    (∃ k : ℤ, (average_age + k * std_dev) - (average_age - k * std_dev) + 1 = num_ages) → k = 1 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_hiring_manager_acceptance_l893_89388


namespace NUMINAMATH_GPT_original_average_l893_89395

theorem original_average (A : ℝ)
  (h : 2 * A = 160) : A = 80 :=
by sorry

end NUMINAMATH_GPT_original_average_l893_89395


namespace NUMINAMATH_GPT_problem_l893_89332

def polynomial (x : ℝ) : ℝ := 9 * x ^ 3 - 27 * x + 54

theorem problem (a b c : ℝ) 
  (h_roots : polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0) :
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_problem_l893_89332


namespace NUMINAMATH_GPT_length_of_second_train_l893_89330

/-- 
  Given:
  * Speed of train 1 is 60 km/hr.
  * Speed of train 2 is 40 km/hr.
  * Length of train 1 is 500 meters.
  * Time to cross each other is 44.99640028797697 seconds.

  Then the length of train 2 is 750 meters.
-/
theorem length_of_second_train (v1 v2 t : ℝ) (d1 L : ℝ) : 
  v1 = 60 ∧
  v2 = 40 ∧
  t = 44.99640028797697 ∧
  d1 = 500 ∧
  L = ((v1 + v2) * (1000 / 3600) * t - d1) →
  L = 750 :=
by sorry

end NUMINAMATH_GPT_length_of_second_train_l893_89330


namespace NUMINAMATH_GPT_international_call_cost_per_minute_l893_89349

theorem international_call_cost_per_minute 
  (local_call_minutes : Nat)
  (international_call_minutes : Nat)
  (local_rate : Nat)
  (total_cost_cents : Nat) 
  (spent_dollars : Nat) 
  (spent_cents : Nat)
  (local_call_cost : Nat)
  (international_call_total_cost : Nat) : 
  local_call_minutes = 45 → 
  international_call_minutes = 31 → 
  local_rate = 5 → 
  total_cost_cents = spent_dollars * 100 → 
  spent_dollars = 10 → 
  local_call_cost = local_call_minutes * local_rate → 
  spent_cents = spent_dollars * 100 → 
  total_cost_cents = spent_cents →  
  international_call_total_cost = total_cost_cents - local_call_cost → 
  international_call_total_cost / international_call_minutes = 25 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_international_call_cost_per_minute_l893_89349


namespace NUMINAMATH_GPT_smallest_points_2016_l893_89334

theorem smallest_points_2016 (n : ℕ) :
  n = 28225 →
  ∀ (points : Fin n → (ℤ × ℤ)),
  ∃ i j : Fin n, i ≠ j ∧
    let dist_sq := (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 
    ∃ k : ℤ, dist_sq = 2016 * k :=
by
  intro h points
  sorry

end NUMINAMATH_GPT_smallest_points_2016_l893_89334


namespace NUMINAMATH_GPT_max_n_factorable_l893_89380

theorem max_n_factorable :
  ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 24 → 3 * B + A = n) ∧ (n = 73) :=
sorry

end NUMINAMATH_GPT_max_n_factorable_l893_89380


namespace NUMINAMATH_GPT_trigonometric_identity_l893_89361

theorem trigonometric_identity (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) : 
  Real.cos x / (Real.sin x - 1) = 1/2 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l893_89361


namespace NUMINAMATH_GPT_ball_count_in_box_eq_57_l893_89308

theorem ball_count_in_box_eq_57 (N : ℕ) (h : N - 44 = 70 - N) : N = 57 :=
sorry

end NUMINAMATH_GPT_ball_count_in_box_eq_57_l893_89308


namespace NUMINAMATH_GPT_wall_building_l893_89314

-- Definitions based on conditions
def total_work (m d : ℕ) : ℕ := m * d

-- Prove that if 30 men including 10 twice as efficient men work for 3 days, they can build the wall
theorem wall_building (m₁ m₂ d₁ d₂ : ℕ) (h₁ : total_work m₁ d₁ = total_work m₂ d₂) (m₁_eq : m₁ = 20) (d₁_eq : d₁ = 6) 
(h₂ : m₂ = 40) : d₂ = 3 :=
  sorry

end NUMINAMATH_GPT_wall_building_l893_89314


namespace NUMINAMATH_GPT_base_length_l893_89305

-- Definition: Isosceles triangle
structure IsoscelesTriangle :=
  (perimeter : ℝ)
  (side : ℝ)

-- Conditions: Perimeter and one side of the isosceles triangle
def given_triangle : IsoscelesTriangle := {
  perimeter := 26,
  side := 11
}

-- The problem to solve: length of the base given the perimeter and one side
theorem base_length : 
  (given_triangle.perimeter = 26 ∧ given_triangle.side = 11) →
  (∃ b : ℝ, b = 11 ∨ b = 7.5) :=
by 
  sorry

end NUMINAMATH_GPT_base_length_l893_89305


namespace NUMINAMATH_GPT_bob_more_than_ken_l893_89336

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := by
  -- proof steps to be filled in
  sorry

end NUMINAMATH_GPT_bob_more_than_ken_l893_89336


namespace NUMINAMATH_GPT_reasoning_is_inductive_l893_89322

-- Define conditions
def conducts_electricity (metal : String) : Prop :=
  metal = "copper" ∨ metal = "iron" ∨ metal = "aluminum" ∨ metal = "gold" ∨ metal = "silver"

-- Define the inductive reasoning type
def is_inductive_reasoning : Prop := 
  ∀ metals, conducts_electricity metals → (∀ m : String, conducts_electricity m → conducts_electricity m)

-- The theorem to prove
theorem reasoning_is_inductive : is_inductive_reasoning :=
by
  sorry

end NUMINAMATH_GPT_reasoning_is_inductive_l893_89322


namespace NUMINAMATH_GPT_ladder_distance_l893_89385

theorem ladder_distance (x : ℝ) (h1 : (13:ℝ) = Real.sqrt (x ^ 2 + 12 ^ 2)) : 
  x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_ladder_distance_l893_89385


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l893_89370

theorem rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) (d1 : ℝ)
  (h_area : area = 432) 
  (h_d2 : d2 = 24) :
  d1 = 36 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l893_89370


namespace NUMINAMATH_GPT_arc_length_of_pentagon_side_l893_89301

theorem arc_length_of_pentagon_side 
  (r : ℝ) (h : r = 4) :
  (2 * r * Real.pi * (72 / 360)) = (8 * Real.pi / 5) :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_pentagon_side_l893_89301


namespace NUMINAMATH_GPT_shaded_fraction_is_four_fifteenths_l893_89316

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ)
  let r := (1/16 : ℚ)
  a / (1 - r)

theorem shaded_fraction_is_four_fifteenths :
  shaded_fraction = (4 / 15 : ℚ) := sorry

end NUMINAMATH_GPT_shaded_fraction_is_four_fifteenths_l893_89316


namespace NUMINAMATH_GPT_triangle_construction_l893_89381

-- Define the problem statement in Lean
theorem triangle_construction (a b c : ℝ) :
  correct_sequence = [3, 1, 4, 2] :=
sorry

end NUMINAMATH_GPT_triangle_construction_l893_89381


namespace NUMINAMATH_GPT_hundredth_number_is_201_l893_89347

-- Mathematical definition of the sequence
def counting_sequence (n : ℕ) : ℕ :=
  3 + (n - 1) * 2

-- Statement to prove
theorem hundredth_number_is_201 : counting_sequence 100 = 201 :=
by
  sorry

end NUMINAMATH_GPT_hundredth_number_is_201_l893_89347


namespace NUMINAMATH_GPT_original_class_strength_l893_89386

theorem original_class_strength (T N : ℕ) (h1 : T = 40 * N) (h2 : T + 12 * 32 = 36 * (N + 12)) : N = 12 :=
by
  sorry

end NUMINAMATH_GPT_original_class_strength_l893_89386


namespace NUMINAMATH_GPT_sum_squares_of_roots_l893_89309

def a := 8
def b := 12
def c := -14

theorem sum_squares_of_roots : (b^2 - 2 * a * c)/(a^2) = 23/4 := by
  sorry

end NUMINAMATH_GPT_sum_squares_of_roots_l893_89309


namespace NUMINAMATH_GPT_find_xy_yz_xz_l893_89373

noncomputable def xy_yz_xz (x y z : ℝ) : ℝ := x * y + y * z + x * z

theorem find_xy_yz_xz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 48) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + x * z + x^2 = 64) :
  xy_yz_xz x y z = 32 :=
sorry

end NUMINAMATH_GPT_find_xy_yz_xz_l893_89373


namespace NUMINAMATH_GPT_value_of_a_minus_b_l893_89356

theorem value_of_a_minus_b (a b : ℝ) 
  (h₁ : (a-4)*(a+4) = 28*a - 112) 
  (h₂ : (b-4)*(b+4) = 28*b - 112) 
  (h₃ : a ≠ b)
  (h₄ : a > b) :
  a - b = 20 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l893_89356


namespace NUMINAMATH_GPT_complex_number_C_l893_89329

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end NUMINAMATH_GPT_complex_number_C_l893_89329


namespace NUMINAMATH_GPT_sequence_is_geometric_l893_89382

theorem sequence_is_geometric (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 3) 
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  ∀ n, a n = 2 ^ (n - 1) + 1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_is_geometric_l893_89382


namespace NUMINAMATH_GPT_range_of_k_l893_89360

def f : ℝ → ℝ := sorry

axiom cond1 (a b : ℝ) : f (a + b) = f a + f b + 2 * a * b
axiom cond2 (k : ℝ) : ∀ x : ℝ, f (x + k) = f (k - x)
axiom cond3 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y

theorem range_of_k (k : ℝ) : k ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_k_l893_89360


namespace NUMINAMATH_GPT_solution_set_x_l893_89312

theorem solution_set_x (x : ℝ) (h₁ : 33 * 32 ≤ x)
  (h₂ : ⌊x⌋ + ⌈x⌉ = 5) : 2 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_x_l893_89312


namespace NUMINAMATH_GPT_abs_negative_five_l893_89375

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_negative_five_l893_89375


namespace NUMINAMATH_GPT_tank_capacity_l893_89391

theorem tank_capacity (x : ℝ) (h₁ : 0.40 * x = 60) : x = 150 :=
by
  -- a suitable proof would go here
  -- since we are only interested in the statement, we place sorry in place of the proof
  sorry

end NUMINAMATH_GPT_tank_capacity_l893_89391


namespace NUMINAMATH_GPT_additional_treetags_l893_89355

noncomputable def initial_numerals : Finset ℕ := {1, 2, 3, 4}
noncomputable def initial_letters : Finset Char := {'A', 'E', 'I'}
noncomputable def initial_symbols : Finset Char := {'!', '@', '#', '$'}
noncomputable def added_numeral : Finset ℕ := {5}
noncomputable def added_symbols : Finset Char := {'&'}

theorem additional_treetags : 
  let initial_treetags := initial_numerals.card * initial_letters.card * initial_symbols.card
  let new_numerals := initial_numerals ∪ added_numeral
  let new_symbols := initial_symbols ∪ added_symbols
  let new_treetags := new_numerals.card * initial_letters.card * new_symbols.card
  new_treetags - initial_treetags = 27 := 
by 
  sorry

end NUMINAMATH_GPT_additional_treetags_l893_89355


namespace NUMINAMATH_GPT_P_equals_neg12_l893_89363

def P (a b : ℝ) : ℝ :=
  (2 * a + 3 * b)^2 - (2 * a + b) * (2 * a - b) - 2 * b * (3 * a + 5 * b)

lemma simplified_P (a b : ℝ) : P a b = 6 * a * b :=
  by sorry

theorem P_equals_neg12 (a b : ℝ) (h : b = -2 / a) : P a b = -12 :=
  by sorry

end NUMINAMATH_GPT_P_equals_neg12_l893_89363


namespace NUMINAMATH_GPT_admission_methods_correct_l893_89346

-- Define the number of famous schools.
def famous_schools : ℕ := 8

-- Define the number of students.
def students : ℕ := 3

-- Define the total number of different admission methods:
def admission_methods (schools : ℕ) (students : ℕ) : ℕ :=
  Nat.choose schools 2 * 3

-- The theorem stating the desired result.
theorem admission_methods_correct :
  admission_methods famous_schools students = 84 :=
by
  sorry

end NUMINAMATH_GPT_admission_methods_correct_l893_89346


namespace NUMINAMATH_GPT_prove_x_value_l893_89372

-- Definitions of the conditions
variable (x y z w : ℕ)
variable (h1 : x = y + 8)
variable (h2 : y = z + 15)
variable (h3 : z = w + 25)
variable (h4 : w = 90)

-- The goal is to prove x = 138 given the conditions
theorem prove_x_value : x = 138 := by
  sorry

end NUMINAMATH_GPT_prove_x_value_l893_89372


namespace NUMINAMATH_GPT_flour_baking_soda_ratio_l893_89319

theorem flour_baking_soda_ratio 
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = 2000)
  (h2 : 5 * flour = 6 * sugar)
  (h3 : 8 * (baking_soda + 60) = flour) :
  flour / baking_soda = 10 := by
  sorry

end NUMINAMATH_GPT_flour_baking_soda_ratio_l893_89319


namespace NUMINAMATH_GPT_extreme_point_at_one_l893_89398

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end NUMINAMATH_GPT_extreme_point_at_one_l893_89398


namespace NUMINAMATH_GPT_line_slope_is_neg_half_l893_89300

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem line_slope_is_neg_half : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) := by
  sorry

end NUMINAMATH_GPT_line_slope_is_neg_half_l893_89300


namespace NUMINAMATH_GPT_prime_squared_difference_divisible_by_24_l893_89341

theorem prime_squared_difference_divisible_by_24 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) :
  24 ∣ (p^2 - q^2) :=
sorry

end NUMINAMATH_GPT_prime_squared_difference_divisible_by_24_l893_89341


namespace NUMINAMATH_GPT_find_p_l893_89306

-- Conditions: Consider the quadratic equation 2x^2 + px + q = 0 where p and q are integers.
-- Roots of the equation differ by 2.
-- q = 4

theorem find_p (p : ℤ) (q : ℤ) (h1 : q = 4) (h2 : ∃ x₁ x₂ : ℝ, 2 * x₁^2 + p * x₁ + q = 0 ∧ 2 * x₂^2 + p * x₂ + q = 0 ∧ |x₁ - x₂| = 2) :
  p = 7 ∨ p = -7 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l893_89306


namespace NUMINAMATH_GPT_only_zero_sol_l893_89397

theorem only_zero_sol (x y z t : ℤ) : x^2 + y^2 + z^2 + t^2 = 2 * x * y * z * t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end NUMINAMATH_GPT_only_zero_sol_l893_89397


namespace NUMINAMATH_GPT_plane_eq_of_point_and_parallel_l893_89390

theorem plane_eq_of_point_and_parallel (A B C D : ℤ) 
  (h1 : A = 3) (h2 : B = -2) (h3 : C = 4) 
  (point : ℝ × ℝ × ℝ) (hpoint : point = (2, -3, 5))
  (h4 : 3 * (2 : ℝ) - 2 * (-3 : ℝ) + 4 * (5 : ℝ) + (D : ℝ) = 0)
  (hD : D = -32)
  (hGCD : Int.gcd (Int.natAbs 3) (Int.gcd (Int.natAbs (-2)) (Int.gcd (Int.natAbs 4) (Int.natAbs (-32)))) = 1) : 
  3 * (x : ℝ) - 2 * (y : ℝ) + 4 * (z : ℝ) - 32 = 0 :=
sorry

end NUMINAMATH_GPT_plane_eq_of_point_and_parallel_l893_89390


namespace NUMINAMATH_GPT_max_popsicles_l893_89315

def popsicles : ℕ := 1
def box_3 : ℕ := 3
def box_5 : ℕ := 5
def box_10 : ℕ := 10
def cost_popsicle : ℕ := 1
def cost_box_3 : ℕ := 2
def cost_box_5 : ℕ := 3
def cost_box_10 : ℕ := 4
def budget : ℕ := 10

theorem max_popsicles : 
  ∀ (popsicle_count : ℕ) (b3_count : ℕ) (b5_count : ℕ) (b10_count : ℕ),
    popsicle_count * cost_popsicle + b3_count * cost_box_3 + b5_count * cost_box_5 + b10_count * cost_box_10 ≤ budget →
    popsicle_count * popsicles + b3_count * box_3 + b5_count * box_5 + b10_count * box_10 ≤ 23 →
    ∃ p b3 b5 b10, popsicle_count = p ∧ b3_count = b3 ∧ b5_count = b5 ∧ b10_count = b10 ∧
    (p * cost_popsicle + b3 * cost_box_3 + b5 * cost_box_5 + b10 * cost_box_10 ≤ budget) ∧
    (p * popsicles + b3 * box_3 + b5 * box_5 + b10 * box_10 = 23) :=
by sorry

end NUMINAMATH_GPT_max_popsicles_l893_89315


namespace NUMINAMATH_GPT_sum_repeating_decimals_l893_89379

theorem sum_repeating_decimals : (0.14 + 0.27) = (41 / 99) := by
  sorry

end NUMINAMATH_GPT_sum_repeating_decimals_l893_89379


namespace NUMINAMATH_GPT_seed_mixture_x_percentage_l893_89339

theorem seed_mixture_x_percentage (x y : ℝ) (h : 0.40 * x + 0.25 * y = 0.30 * (x + y)) : 
  (x / (x + y)) * 100 = 33.33 := sorry

end NUMINAMATH_GPT_seed_mixture_x_percentage_l893_89339


namespace NUMINAMATH_GPT_S_eq_Z_l893_89357

noncomputable def set_satisfies_conditions (S : Set ℤ) (a : Fin n → ℤ) :=
  (∀ i : Fin n, a i ∈ S) ∧
  (∀ i j : Fin n, (a i - a j) ∈ S) ∧
  (∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) ∧
  (Nat.gcd (List.foldr Nat.gcd 0 (Fin.val <$> List.finRange n)) = 1)

theorem S_eq_Z (S : Set ℤ) (a : Fin n → ℤ) (h_cond : set_satisfies_conditions S a) : S = Set.univ :=
  sorry

end NUMINAMATH_GPT_S_eq_Z_l893_89357


namespace NUMINAMATH_GPT_find_n_l893_89366

theorem find_n
  (n : ℕ)
  (h1 : 2287 % n = r)
  (h2 : 2028 % n = r)
  (h3 : 1806 % n = r)
  (h_r_non_zero : r ≠ 0) : 
  n = 37 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l893_89366


namespace NUMINAMATH_GPT_total_hours_correct_l893_89383

/-- Definitions for the times each person has left to finish their homework. -/
noncomputable def Jacob_time : ℕ := 18
noncomputable def Greg_time : ℕ := Jacob_time - 6
noncomputable def Patrick_time : ℕ := 2 * Greg_time - 4

/-- Proving the total time left for Patrick, Greg, and Jacob to finish their homework. -/

theorem total_hours_correct : Jacob_time + Greg_time + Patrick_time = 50 := by
  sorry

end NUMINAMATH_GPT_total_hours_correct_l893_89383


namespace NUMINAMATH_GPT_compute_remainder_l893_89377

/-- T is the sum of all three-digit positive integers 
  where the digits are distinct, the hundreds digit is at least 2,
  and the digit 1 is not used in any place. -/
def T : ℕ := 
  let hundreds_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 56 * 100
  let tens_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49 * 10
  let units_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49
  hundreds_sum + tens_sum + units_sum

/-- Theorem: Compute the remainder when T is divided by 1000. -/
theorem compute_remainder : T % 1000 = 116 := by
  sorry

end NUMINAMATH_GPT_compute_remainder_l893_89377


namespace NUMINAMATH_GPT_base_difference_is_correct_l893_89307

-- Definitions of given conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 324 => 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Lean statement to prove the equivalence
theorem base_difference_is_correct : base9_to_base10 324 - base6_to_base10 231 = 174 :=
by
  sorry

end NUMINAMATH_GPT_base_difference_is_correct_l893_89307


namespace NUMINAMATH_GPT_intersection_M_N_l893_89321

def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {x | x^2 - 4 * x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l893_89321


namespace NUMINAMATH_GPT_geometric_sequence_a5_l893_89342

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (hq : q = 2) (h_a2a6 : a 2 * a 6 = 16) :
  a 5 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l893_89342


namespace NUMINAMATH_GPT_operation_value_l893_89358

-- Define the operations as per the conditions.
def star (m n : ℤ) : ℤ := n^2 - m
def hash (m k : ℤ) : ℚ := (k + 2 * m) / 3

-- State the theorem we want to prove.
theorem operation_value : hash (star 3 3) (star 2 5) = 35 / 3 :=
  by
  sorry

end NUMINAMATH_GPT_operation_value_l893_89358


namespace NUMINAMATH_GPT_nineteen_times_eight_pow_n_plus_seventeen_is_composite_l893_89323

theorem nineteen_times_eight_pow_n_plus_seventeen_is_composite 
  (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
sorry

end NUMINAMATH_GPT_nineteen_times_eight_pow_n_plus_seventeen_is_composite_l893_89323


namespace NUMINAMATH_GPT_tan_alpha_sub_60_l893_89378

theorem tan_alpha_sub_60 
  (alpha : ℝ) 
  (h : Real.tan alpha = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (alpha - 60 * Real.pi / 180) = (Real.sqrt 3) / 7 :=
by sorry

end NUMINAMATH_GPT_tan_alpha_sub_60_l893_89378


namespace NUMINAMATH_GPT_probability_after_5_rounds_l893_89317

def initial_coins : ℕ := 5
def rounds : ℕ := 5
def final_probability : ℚ := 1 / 2430000

structure Player :=
  (name : String)
  (initial_coins : ℕ)
  (final_coins : ℕ)

def Abby : Player := ⟨"Abby", 5, 5⟩
def Bernardo : Player := ⟨"Bernardo", 4, 3⟩
def Carl : Player := ⟨"Carl", 3, 3⟩
def Debra : Player := ⟨"Debra", 4, 5⟩

def check_final_state (players : List Player) : Prop :=
  ∀ (p : Player), p ∈ players →
  (p.name = "Abby" ∧ p.final_coins = 5 ∨
   p.name = "Bernardo" ∧ p.final_coins = 3 ∨
   p.name = "Carl" ∧ p.final_coins = 3 ∨
   p.name = "Debra" ∧ p.final_coins = 5)

theorem probability_after_5_rounds :
  ∃ prob : ℚ, prob = final_probability ∧ check_final_state [Abby, Bernardo, Carl, Debra] :=
sorry

end NUMINAMATH_GPT_probability_after_5_rounds_l893_89317


namespace NUMINAMATH_GPT_additional_money_needed_l893_89362

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end NUMINAMATH_GPT_additional_money_needed_l893_89362


namespace NUMINAMATH_GPT_john_average_score_change_l893_89392

/-- Given John's scores on his biology exams, calculate the change in his average score after the fourth exam. -/
theorem john_average_score_change :
  let first_three_scores := [84, 88, 95]
  let fourth_score := 92
  let first_average := (84 + 88 + 95) / 3
  let new_average := (84 + 88 + 95 + 92) / 4
  new_average - first_average = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_john_average_score_change_l893_89392


namespace NUMINAMATH_GPT_sum_of_segments_l893_89352

noncomputable def segment_sum (AB_len CB_len FG_len : ℕ) : ℝ :=
  199 * (Real.sqrt (AB_len * AB_len + CB_len * CB_len) +
         Real.sqrt (AB_len * AB_len + FG_len * FG_len))

theorem sum_of_segments : segment_sum 5 6 8 = 199 * (Real.sqrt 61 + Real.sqrt 89) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_segments_l893_89352


namespace NUMINAMATH_GPT_desired_average_l893_89344

theorem desired_average (P1 P2 P3 : ℝ) (A : ℝ) 
  (hP1 : P1 = 74) 
  (hP2 : P2 = 84) 
  (hP3 : P3 = 67) 
  (hA : A = (P1 + P2 + P3) / 3) : 
  A = 75 :=
  sorry

end NUMINAMATH_GPT_desired_average_l893_89344


namespace NUMINAMATH_GPT_pencils_per_student_l893_89338

-- Define the number of pens
def numberOfPens : ℕ := 1001

-- Define the number of pencils
def numberOfPencils : ℕ := 910

-- Define the maximum number of students
def maxNumberOfStudents : ℕ := 91

-- Using the given conditions, prove that each student gets 10 pencils
theorem pencils_per_student :
  (numberOfPencils / maxNumberOfStudents) = 10 :=
by sorry

end NUMINAMATH_GPT_pencils_per_student_l893_89338


namespace NUMINAMATH_GPT_clock_angle_9_30_l893_89399

theorem clock_angle_9_30 : 
  let hour_hand_pos := 9.5 
  let minute_hand_pos := 6 
  let degrees_per_division := 30 
  let divisions_apart := hour_hand_pos - minute_hand_pos
  let angle := divisions_apart * degrees_per_division
  angle = 105 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_9_30_l893_89399
