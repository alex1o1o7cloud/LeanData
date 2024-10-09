import Mathlib

namespace acid_solution_replaced_l2396_239690

theorem acid_solution_replaced (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.35) → P = 0.20 :=
by
  intro h
  sorry

end acid_solution_replaced_l2396_239690


namespace AM_GM_problem_l2396_239607

theorem AM_GM_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := 
sorry

end AM_GM_problem_l2396_239607


namespace school_pays_570_l2396_239691

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l2396_239691


namespace smallest_int_a_for_inequality_l2396_239615

theorem smallest_int_a_for_inequality (a : ℤ) : 
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) → 
  a = 1 := 
sorry

end smallest_int_a_for_inequality_l2396_239615


namespace total_shells_correct_l2396_239608

def morning_shells : ℕ := 292
def afternoon_shells : ℕ := 324

theorem total_shells_correct : morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_correct_l2396_239608


namespace cricket_run_rate_l2396_239663

theorem cricket_run_rate (x : ℝ) (hx : 3.2 * x + 6.25 * 40 = 282) : x = 10 :=
by sorry

end cricket_run_rate_l2396_239663


namespace calculate_average_age_l2396_239656

variables (k : ℕ) (female_to_male_ratio : ℚ) (avg_young_female : ℚ) (avg_old_female : ℚ) (avg_young_male : ℚ) (avg_old_male : ℚ)

theorem calculate_average_age 
  (h_ratio : female_to_male_ratio = 7/8)
  (h_avg_yf : avg_young_female = 26)
  (h_avg_of : avg_old_female = 42)
  (h_avg_ym : avg_young_male = 28)
  (h_avg_om : avg_old_male = 46) : 
  (534/15 : ℚ) = 36 :=
by sorry

end calculate_average_age_l2396_239656


namespace inverse_of_B_squared_l2396_239631

noncomputable def B_inv : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end inverse_of_B_squared_l2396_239631


namespace number_of_graphic_novels_l2396_239611

theorem number_of_graphic_novels (total_books novels_percent comics_percent : ℝ) 
  (h_total : total_books = 120) 
  (h_novels_percent : novels_percent = 0.65) 
  (h_comics_percent : comics_percent = 0.20) :
  total_books - (novels_percent * total_books + comics_percent * total_books) = 18 :=
by
  sorry

end number_of_graphic_novels_l2396_239611


namespace problem_1_problem_2_l2396_239621

variables (α : ℝ) (h : Real.tan α = 3)

theorem problem_1 : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by
  -- Proof is skipped
  sorry

theorem problem_2 : Real.sin α * Real.sin α + Real.sin α * Real.cos α + 3 * Real.cos α * Real.cos α = 3 / 2 :=
by
  -- Proof is skipped
  sorry

end problem_1_problem_2_l2396_239621


namespace three_layers_coverage_l2396_239619

/--
Three table runners have a combined area of 208 square inches. 
By overlapping the runners to cover 80% of a table of area 175 square inches, 
the area that is covered by exactly two layers of runner is 24 square inches. 
Prove that the area of the table that is covered with three layers of runner is 22 square inches.
--/
theorem three_layers_coverage :
  ∀ (A T two_layers total_table_coverage : ℝ),
  A = 208 ∧ total_table_coverage = 0.8 * 175 ∧ two_layers = 24 →
  A = (total_table_coverage - two_layers - T) + 2 * two_layers + 3 * T →
  T = 22 :=
by
  intros A T two_layers total_table_coverage h1 h2
  sorry

end three_layers_coverage_l2396_239619


namespace part1_part2_l2396_239652
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l2396_239652


namespace find_side_length_l2396_239626

theorem find_side_length
  (X : ℕ)
  (h1 : 3 + 2 + X + 4 = 12) :
  X = 3 :=
by
  sorry

end find_side_length_l2396_239626


namespace gold_tetrahedron_volume_l2396_239625

theorem gold_tetrahedron_volume (side_length : ℝ) (h : side_length = 8) : 
  volume_of_tetrahedron_with_gold_vertices = 170.67 := 
by 
  sorry

end gold_tetrahedron_volume_l2396_239625


namespace isosceles_triangle_legs_length_l2396_239654

theorem isosceles_triangle_legs_length 
  (P : ℝ) (base : ℝ) (leg_length : ℝ) 
  (hp : P = 26) 
  (hb : base = 11) 
  (hP : P = 2 * leg_length + base) : 
  leg_length = 7.5 := 
by 
  sorry

end isosceles_triangle_legs_length_l2396_239654


namespace drying_time_correct_l2396_239679

theorem drying_time_correct :
  let short_haired_dog_drying_time := 10
  let full_haired_dog_drying_time := 2 * short_haired_dog_drying_time
  let num_short_haired_dogs := 6
  let num_full_haired_dogs := 9
  let total_short_haired_dogs_time := num_short_haired_dogs * short_haired_dog_drying_time
  let total_full_haired_dogs_time := num_full_haired_dogs * full_haired_dog_drying_time
  let total_drying_time_in_minutes := total_short_haired_dogs_time + total_full_haired_dogs_time
  let total_drying_time_in_hours := total_drying_time_in_minutes / 60
  total_drying_time_in_hours = 4 := 
by
  sorry

end drying_time_correct_l2396_239679


namespace trivia_team_total_points_l2396_239650

/-- Given the points scored by the 5 members who showed up in a trivia team game,
    prove that the total points scored by the team is 29. -/
theorem trivia_team_total_points 
  (points_first : ℕ := 5) 
  (points_second : ℕ := 9) 
  (points_third : ℕ := 7) 
  (points_fourth : ℕ := 5) 
  (points_fifth : ℕ := 3) 
  (total_points : ℕ := points_first + points_second + points_third + points_fourth + points_fifth) :
  total_points = 29 :=
by
  sorry

end trivia_team_total_points_l2396_239650


namespace digit_1035_is_2_l2396_239696

noncomputable def sequence_digits (n : ℕ) : ℕ :=
  -- Convert the sequence of numbers from 1 to n to digits and return a specific position.
  sorry

theorem digit_1035_is_2 : sequence_digits 500 = 2 :=
  sorry

end digit_1035_is_2_l2396_239696


namespace four_digit_numbers_with_8_or_3_l2396_239628

theorem four_digit_numbers_with_8_or_3 :
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  total_four_digit_numbers - numbers_without_8_or_3 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  sorry

end four_digit_numbers_with_8_or_3_l2396_239628


namespace period_fraction_sum_nines_l2396_239669

theorem period_fraction_sum_nines (q : ℕ) (p : ℕ) (N N1 N2 : ℕ) (n : ℕ) (t : ℕ) 
  (hq_prime : Nat.Prime q) (hq_gt_5 : q > 5) (hp_lt_q : p < q)
  (ht_eq_2n : t = 2 * n) (h_period : 10^t ≡ 1 [MOD q])
  (hN_eq_concat : (N = N1 * 10^n + N2) ∧ (N % 10^n = N2))
  : N1 + N2 = (10^n - 1) := 
sorry

end period_fraction_sum_nines_l2396_239669


namespace circle_equation_range_l2396_239636

theorem circle_equation_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a + 1 = 0) → a < 4 := 
by 
  sorry

end circle_equation_range_l2396_239636


namespace fraction_identity_l2396_239648

theorem fraction_identity (a b : ℝ) (hb : b ≠ 0) (h : a / b = 3 / 2) : (a + b) / b = 2.5 :=
by
  sorry

end fraction_identity_l2396_239648


namespace range_a_l2396_239638

def A : Set ℝ :=
  {x | x^2 + 5 * x + 6 ≤ 0}

def B : Set ℝ :=
  {x | -3 ≤ x ∧ x ≤ 5}

def C (a : ℝ) : Set ℝ :=
  {x | a < x ∧ x < a + 1}

theorem range_a (a : ℝ) : ((A ∪ B) ∩ C a = ∅) → (a ≥ 5 ∨ a ≤ -4) :=
  sorry

end range_a_l2396_239638


namespace more_stable_scores_l2396_239697

-- Define the variances for Student A and Student B
def variance_A : ℝ := 38
def variance_B : ℝ := 15

-- Formulate the theorem
theorem more_stable_scores : variance_A > variance_B → "B" = "B" :=
by
  intro h
  sorry

end more_stable_scores_l2396_239697


namespace household_savings_regression_l2396_239617

-- Define the problem conditions in Lean
def n := 10
def sum_x := 80
def sum_y := 20
def sum_xy := 184
def sum_x2 := 720

-- Define the averages
def x_bar := sum_x / n
def y_bar := sum_y / n

-- Define the lxx and lxy as per the solution
def lxx := sum_x2 - n * x_bar^2
def lxy := sum_xy - n * x_bar * y_bar

-- Define the regression coefficients
def b_hat := lxy / lxx
def a_hat := y_bar - b_hat * x_bar

-- State the theorem to be proved
theorem household_savings_regression :
  (∀ (x: ℝ), y = b_hat * x + a_hat) :=
by
  sorry -- skip the proof

end household_savings_regression_l2396_239617


namespace keith_picked_0_pears_l2396_239618

structure Conditions where
  apples_total : ℕ
  apples_mike : ℕ
  apples_nancy : ℕ
  apples_keith : ℕ
  pears_keith : ℕ

theorem keith_picked_0_pears (c : Conditions) (h_total : c.apples_total = 16)
 (h_mike : c.apples_mike = 7) (h_nancy : c.apples_nancy = 3)
 (h_keith : c.apples_keith = 6) : c.pears_keith = 0 :=
by
  sorry

end keith_picked_0_pears_l2396_239618


namespace first_operation_result_l2396_239675

def pattern (x y : ℕ) : ℕ :=
  if (x, y) = (3, 7) then 27
  else if (x, y) = (4, 5) then 32
  else if (x, y) = (5, 8) then 60
  else if (x, y) = (6, 7) then 72
  else if (x, y) = (7, 8) then 98
  else 26

theorem first_operation_result : pattern 2 3 = 26 := by
  sorry

end first_operation_result_l2396_239675


namespace point_of_tangency_is_correct_l2396_239639

theorem point_of_tangency_is_correct : 
  (∃ (x y : ℝ), y = x^2 + 20 * x + 63 ∧ x = y^2 + 56 * y + 875 ∧ x = -19 / 2 ∧ y = -55 / 2) :=
by
  sorry

end point_of_tangency_is_correct_l2396_239639


namespace tan_sub_sin_eq_sq3_div2_l2396_239612

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l2396_239612


namespace water_tank_capacity_l2396_239682

theorem water_tank_capacity :
  ∃ (x : ℝ), 0.9 * x - 0.4 * x = 30 → x = 60 :=
by
  sorry

end water_tank_capacity_l2396_239682


namespace gcd_problem_l2396_239659

def a := 47^11 + 1
def b := 47^11 + 47^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := 
by
  sorry

end gcd_problem_l2396_239659


namespace emily_sixth_score_l2396_239660

theorem emily_sixth_score:
  ∀ (s₁ s₂ s₃ s₄ s₅ sᵣ : ℕ),
  s₁ = 88 →
  s₂ = 90 →
  s₃ = 85 →
  s₄ = 92 →
  s₅ = 97 →
  (s₁ + s₂ + s₃ + s₄ + s₅ + sᵣ) / 6 = 91 →
  sᵣ = 94 :=
by intros s₁ s₂ s₃ s₄ s₅ sᵣ h₁ h₂ h₃ h₄ h₅ h₆;
   rw [h₁, h₂, h₃, h₄, h₅] at h₆;
   sorry

end emily_sixth_score_l2396_239660


namespace triangle_is_right_triangle_l2396_239680

theorem triangle_is_right_triangle (a b c : ℕ) (h_ratio : a = 3 * (36 / 12)) (h_perimeter : 3 * (36 / 12) + 4 * (36 / 12) + 5 * (36 / 12) = 36) :
  a^2 + b^2 = c^2 :=
by
  -- sorry for skipping the proof.
  sorry

end triangle_is_right_triangle_l2396_239680


namespace probability_of_each_suit_in_five_draws_with_replacement_l2396_239604

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end probability_of_each_suit_in_five_draws_with_replacement_l2396_239604


namespace distance_from_point_to_x_axis_l2396_239678

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_from_point_to_x_axis :
  let p := (-2, -Real.sqrt 5)
  distance_to_x_axis p = Real.sqrt 5 := by
  sorry

end distance_from_point_to_x_axis_l2396_239678


namespace fred_earnings_l2396_239684
noncomputable def start := 111
noncomputable def now := 115
noncomputable def earnings := now - start

theorem fred_earnings : earnings = 4 :=
by
  sorry

end fred_earnings_l2396_239684


namespace determine_x_l2396_239683

theorem determine_x : ∃ (x : ℕ), 
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7) ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ ¬(x < 120) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∧
  x = 10 :=
sorry

end determine_x_l2396_239683


namespace removed_term_sequence_l2396_239667

theorem removed_term_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (k : ℕ) :
  (∀ n, S n = 2 * n^2 - n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (S 21 - a k = 40 * 20) →
  a k = 4 * k - 3 →
  k = 16 :=
by
  intros hs ha h_avg h_ak
  sorry

end removed_term_sequence_l2396_239667


namespace train_length_calculation_l2396_239670

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l2396_239670


namespace determine_sanity_l2396_239693

-- Defining the conditions for sanity based on responses to a specific question

-- Define possible responses
inductive Response
| ball : Response
| yes : Response

-- Define sanity based on logical interpretation of an illogical question
def is_sane (response : Response) : Prop :=
  response = Response.ball

-- The theorem stating asking the specific question determines sanity
theorem determine_sanity (response : Response) : is_sane response ↔ response = Response.ball :=
by
  sorry

end determine_sanity_l2396_239693


namespace remainder_seven_times_quotient_l2396_239649

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r = 7 * q ∧ 0 ≤ r ∧ r < 23) ↔ (n = 30 ∨ n = 60 ∨ n = 90) :=
by 
  sorry

end remainder_seven_times_quotient_l2396_239649


namespace equal_roots_quadratic_l2396_239634

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

/--
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots,
then the value of a is ±4.
-/
theorem equal_roots_quadratic (a : ℝ) (h : quadratic_discriminant 2 (-a) 2 = 0) :
  a = 4 ∨ a = -4 :=
sorry

end equal_roots_quadratic_l2396_239634


namespace johns_total_earnings_per_week_l2396_239614

def small_crab_baskets_monday := 3
def medium_crab_baskets_monday := 2
def large_crab_baskets_thursday := 4
def jumbo_crab_baskets_thursday := 1

def crabs_per_small_basket := 4
def crabs_per_medium_basket := 3
def crabs_per_large_basket := 5
def crabs_per_jumbo_basket := 2

def price_per_small_crab := 3
def price_per_medium_crab := 4
def price_per_large_crab := 5
def price_per_jumbo_crab := 7

def total_weekly_earnings :=
  (small_crab_baskets_monday * crabs_per_small_basket * price_per_small_crab) +
  (medium_crab_baskets_monday * crabs_per_medium_basket * price_per_medium_crab) +
  (large_crab_baskets_thursday * crabs_per_large_basket * price_per_large_crab) +
  (jumbo_crab_baskets_thursday * crabs_per_jumbo_basket * price_per_jumbo_crab)

theorem johns_total_earnings_per_week : total_weekly_earnings = 174 :=
by sorry

end johns_total_earnings_per_week_l2396_239614


namespace marble_ratio_l2396_239686

-- Let Allison, Angela, and Albert have some number of marbles denoted by variables.
variable (Albert Angela Allison : ℕ)

-- Given conditions.
axiom h1 : Angela = Allison + 8
axiom h2 : Allison = 28
axiom h3 : Albert + Allison = 136

-- Prove that the ratio of the number of marbles Albert has to the number of marbles Angela has is 3.
theorem marble_ratio : Albert / Angela = 3 := by
  sorry

end marble_ratio_l2396_239686


namespace somu_one_fifth_age_back_l2396_239623

theorem somu_one_fifth_age_back {S F Y : ℕ}
  (h1 : S = 16)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 8 :=
by
  sorry

end somu_one_fifth_age_back_l2396_239623


namespace geom_sequence_general_formula_l2396_239655

theorem geom_sequence_general_formula :
  ∃ (a : ℕ → ℝ) (a₁ q : ℝ), 
  (∀ n, a n = a₁ * q ^ n ∧ abs (q) < 1 ∧ ∑' i, a i = 3 ∧ ∑' i, (a i)^2 = (9 / 2)) →
  (∀ n, a n = 2 * ((1 / 3) ^ (n - 1))) :=
by sorry

end geom_sequence_general_formula_l2396_239655


namespace eight_bags_weight_l2396_239666

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end eight_bags_weight_l2396_239666


namespace temperature_on_fifth_day_l2396_239624

theorem temperature_on_fifth_day (T : ℕ → ℝ) (x : ℝ)
  (h1 : (T 1 + T 2 + T 3 + T 4) / 4 = 58)
  (h2 : (T 2 + T 3 + T 4 + T 5) / 4 = 59)
  (h3 : T 1 / T 5 = 7 / 8) :
  T 5 = 32 := 
sorry

end temperature_on_fifth_day_l2396_239624


namespace simplify_fraction_l2396_239600

-- Define the given variables and their assigned values.
variable (b : ℕ)
variable (b_eq : b = 2)

-- State the theorem we want to prove
theorem simplify_fraction (b : ℕ) (h : b = 2) : 
  15 * b ^ 4 / (75 * b ^ 3) = 2 / 5 :=
by
  -- sorry indicates where the proof would be written.
  sorry

end simplify_fraction_l2396_239600


namespace elsa_emma_spending_ratio_l2396_239687

theorem elsa_emma_spending_ratio
  (E : ℝ)
  (h_emma : ∃ (x : ℝ), x = 58)
  (h_elizabeth : ∃ (y : ℝ), y = 4 * E)
  (h_total : 58 + E + 4 * E = 638) :
  E / 58 = 2 :=
by
  sorry

end elsa_emma_spending_ratio_l2396_239687


namespace min_dominos_in_2x2_l2396_239630

/-- A 100 × 100 square is divided into 2 × 2 squares.
Then it is divided into dominos (rectangles 1 × 2 and 2 × 1).
Prove that the minimum number of dominos within the 2 × 2 squares is 100. -/
theorem min_dominos_in_2x2 (N : ℕ) (hN : N = 100) :
  ∃ d : ℕ, d = 100 :=
sorry

end min_dominos_in_2x2_l2396_239630


namespace triangle_other_side_length_l2396_239632

theorem triangle_other_side_length (a b : ℝ) (c : ℝ) (h_a : a = 3) (h_b : b = 4) (h_right_angle : c * c = a * a + b * b ∨ a * a = c * c + b * b):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end triangle_other_side_length_l2396_239632


namespace translate_parabola_up_one_unit_l2396_239605

theorem translate_parabola_up_one_unit (x : ℝ) :
  let y := 3 * x^2
  (y + 1) = 3 * x^2 + 1 :=
by
  -- Proof omitted
  sorry

end translate_parabola_up_one_unit_l2396_239605


namespace bella_more_than_max_l2396_239673

noncomputable def num_students : ℕ := 10
noncomputable def bananas_eaten_by_bella : ℕ := 7
noncomputable def bananas_eaten_by_max : ℕ := 1

theorem bella_more_than_max : 
  bananas_eaten_by_bella - bananas_eaten_by_max = 6 :=
by
  sorry

end bella_more_than_max_l2396_239673


namespace total_price_of_purchases_l2396_239601

def price_of_refrigerator := 4275
def price_difference := 1490
def price_of_washing_machine := price_of_refrigerator - price_difference
def total_price := price_of_refrigerator + price_of_washing_machine

theorem total_price_of_purchases : total_price = 7060 :=
by
  rfl  -- This is just a placeholder; you need to solve the proof.

end total_price_of_purchases_l2396_239601


namespace bus_stops_for_minutes_per_hour_l2396_239689

theorem bus_stops_for_minutes_per_hour (speed_no_stops speed_with_stops : ℕ)
  (h1 : speed_no_stops = 60) (h2 : speed_with_stops = 45) : 
  (60 * (speed_no_stops - speed_with_stops) / speed_no_stops) = 15 :=
by
  sorry

end bus_stops_for_minutes_per_hour_l2396_239689


namespace angle_triple_complement_l2396_239653

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l2396_239653


namespace choose_amber_bronze_cells_l2396_239694

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end choose_amber_bronze_cells_l2396_239694


namespace least_xy_value_l2396_239642

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (1/x : ℚ) + 1/(2*y) = 1/8) :
  xy ≥ 128 :=
sorry

end least_xy_value_l2396_239642


namespace price_of_adult_ticket_l2396_239633

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end price_of_adult_ticket_l2396_239633


namespace center_of_symmetry_l2396_239664

theorem center_of_symmetry (k : ℤ) : ∀ (k : ℤ), ∃ x : ℝ, 
  (x = (k * Real.pi / 6 - Real.pi / 9) ∨ x = - (Real.pi / 18)) → False :=
by
  sorry

end center_of_symmetry_l2396_239664


namespace line_equation_through_point_parallel_to_lines_l2396_239635

theorem line_equation_through_point_parallel_to_lines (L L1 L2 : ℝ → ℝ → Prop) :
  (∀ x, L1 x (y: ℝ) ↔ 3 * x + y - 6 = 0) →
  (∀ x, L2 x (y: ℝ) ↔ 3 * x + y + 3 = 0) →
  (L 1 0) →
  (∀ x1 y1 x2 y2, L1 x1 y1 → L1 x2 y2 → (y2 - y1) / (x2 - x1) = -3) →
  ∃ A B C, (A = 1 ∧ B = -3 ∧ C = -3) ∧ (∀ x y, L x y ↔ A * x + B * y + C = 0) :=
by sorry

end line_equation_through_point_parallel_to_lines_l2396_239635


namespace sum_abc_eq_neg_ten_thirds_l2396_239646

variable (a b c d y : ℝ)

-- Define the conditions
def condition_1 : Prop := a + 2 = y
def condition_2 : Prop := b + 3 = y
def condition_3 : Prop := c + 4 = y
def condition_4 : Prop := d + 5 = y
def condition_5 : Prop := a + b + c + d + 6 = y

-- State the theorem
theorem sum_abc_eq_neg_ten_thirds
    (h1 : condition_1 a y)
    (h2 : condition_2 b y)
    (h3 : condition_3 c y)
    (h4 : condition_4 d y)
    (h5 : condition_5 a b c d y) :
    a + b + c + d = -10 / 3 :=
sorry

end sum_abc_eq_neg_ten_thirds_l2396_239646


namespace sum_at_simple_interest_l2396_239613

theorem sum_at_simple_interest (P R : ℝ) (h1 : P * R * 3 / 100 - P * (R + 3) * 3 / 100 = -90) : P = 1000 :=
sorry

end sum_at_simple_interest_l2396_239613


namespace find_a_c_l2396_239665

theorem find_a_c (a c : ℝ) (h1 : a + c = 35) (h2 : a < c)
  (h3 : ∀ x : ℝ, a * x^2 + 30 * x + c = 0 → ∃! x, a * x^2 + 30 * x + c = 0) :
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by
  sorry

end find_a_c_l2396_239665


namespace total_value_of_coins_l2396_239644

theorem total_value_of_coins (q d : ℕ) (total_value original_value swapped_value : ℚ)
  (h1 : q + d = 30)
  (h2 : total_value = 4.50)
  (h3 : original_value = 25 * q + 10 * d)
  (h4 : swapped_value = 10 * q + 25 * d)
  (h5 : swapped_value = original_value + 1.50) :
  total_value = original_value / 100 :=
sorry

end total_value_of_coins_l2396_239644


namespace arithmetic_sum_l2396_239668

theorem arithmetic_sum (a₁ an n : ℕ) (h₁ : a₁ = 5) (h₂ : an = 32) (h₃ : n = 10) :
  (n * (a₁ + an)) / 2 = 185 :=
by
  sorry

end arithmetic_sum_l2396_239668


namespace number_of_sides_of_regular_polygon_l2396_239606

theorem number_of_sides_of_regular_polygon (P s n : ℕ) (hP : P = 150) (hs : s = 15) (hP_formula : P = n * s) : n = 10 :=
  by {
    -- proof goes here
    sorry
  }

end number_of_sides_of_regular_polygon_l2396_239606


namespace negation_correct_l2396_239695

theorem negation_correct (x : ℝ) : -(3 * x - 2) = -3 * x + 2 := 
by sorry

end negation_correct_l2396_239695


namespace total_cost_is_72_l2396_239627

-- Definitions based on conditions
def adults (total_people : ℕ) (kids : ℕ) : ℕ := total_people - kids
def cost_per_adult_meal (cost_per_meal : ℕ) (adults : ℕ) : ℕ := cost_per_meal * adults
def total_cost (total_people : ℕ) (kids : ℕ) (cost_per_meal : ℕ) : ℕ := 
  cost_per_adult_meal cost_per_meal (adults total_people kids)

-- Given values
def total_people := 11
def kids := 2
def cost_per_meal := 8

-- Theorem statement
theorem total_cost_is_72 : total_cost total_people kids cost_per_meal = 72 := by
  sorry

end total_cost_is_72_l2396_239627


namespace add_solution_y_to_solution_x_l2396_239657

theorem add_solution_y_to_solution_x
  (x_volume : ℝ) (x_percent : ℝ) (y_percent : ℝ) (desired_percent : ℝ) (final_volume : ℝ)
  (x_alcohol : ℝ := x_volume * x_percent / 100) (y : ℝ := final_volume - x_volume) :
  (x_percent = 10) → (y_percent = 30) → (desired_percent = 15) → (x_volume = 300) →
  (final_volume = 300 + y) →
  ((x_alcohol + y * y_percent / 100) / final_volume = desired_percent / 100) →
  y = 100 := by
    intros h1 h2 h3 h4 h5 h6
    sorry

end add_solution_y_to_solution_x_l2396_239657


namespace function_nonnegative_l2396_239620

noncomputable def f (x : ℝ) := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem function_nonnegative (x : ℝ) : 
  (f x ≥ 0) ↔ (0 ≤ x ∧ x ≤ (1 / 7)) ∨ (3 ≤ x) :=
sorry

end function_nonnegative_l2396_239620


namespace problem1_problem2_l2396_239688

-- Given conditions
def A : Set ℝ := { x | x^2 - 2 * x - 15 > 0 }
def B : Set ℝ := { x | x < 6 }
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Statements to prove
theorem problem1 (m : ℝ) : p m → m ∈ { x | x < -3 } ∪ { x | x > 5 } :=
sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ { x | x < -3 } :=
sorry

end problem1_problem2_l2396_239688


namespace smallest_circle_area_l2396_239681

noncomputable def function_y (x : ℝ) : ℝ := 6 / x - 4 * x / 3

theorem smallest_circle_area :
  ∃ r : ℝ, (∀ x : ℝ, r * r = x^2 + (function_y x)^2) → r^2 * π = 4 * π :=
sorry

end smallest_circle_area_l2396_239681


namespace students_answered_both_correctly_l2396_239676

theorem students_answered_both_correctly
  (enrolled : ℕ)
  (did_not_take_test : ℕ)
  (answered_q1_correctly : ℕ)
  (answered_q2_correctly : ℕ)
  (total_students_answered_both : ℕ) :
  enrolled = 29 →
  did_not_take_test = 5 →
  answered_q1_correctly = 19 →
  answered_q2_correctly = 24 →
  total_students_answered_both = 19 :=
by
  intros
  sorry

end students_answered_both_correctly_l2396_239676


namespace gwen_money_remaining_l2396_239645

def gwen_money (initial : ℝ) (spent1 : ℝ) (earned : ℝ) (spent2 : ℝ) : ℝ :=
  initial - spent1 + earned - spent2

theorem gwen_money_remaining :
  gwen_money 5 3.25 1.5 0.7 = 2.55 :=
by
  sorry

end gwen_money_remaining_l2396_239645


namespace original_ratio_l2396_239661

theorem original_ratio (x y : ℤ)
  (h1 : y = 48)
  (h2 : (x + 12) * 2 = y) :
  x * 4 = y := sorry

end original_ratio_l2396_239661


namespace number_of_pairs_l2396_239616

theorem number_of_pairs : 
  (∃ (m n : ℤ), m + n = mn - 3) → ∃! (count : ℕ), count = 6 := by
  sorry

end number_of_pairs_l2396_239616


namespace part1_part2_l2396_239698

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end part1_part2_l2396_239698


namespace only_k_equal_1_works_l2396_239603

-- Define the first k prime numbers product
def prime_prod (k : ℕ) : ℕ :=
  Nat.recOn k 1 (fun n prod => prod * (Nat.factorial (n + 1) - Nat.factorial n))

-- Define a predicate for being the sum of two positive cubes
def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a^3 + b^3

-- The theorem statement
theorem only_k_equal_1_works :
  ∀ k : ℕ, (prime_prod k = 2 ↔ k = 1) :=
by
  sorry

end only_k_equal_1_works_l2396_239603


namespace probability_excellent_probability_good_or_better_l2396_239677

noncomputable def total_selections : ℕ := 10
noncomputable def total_excellent_selections : ℕ := 1
noncomputable def total_good_or_better_selections : ℕ := 7
noncomputable def P_excellent : ℚ := 1 / 10
noncomputable def P_good_or_better : ℚ := 7 / 10

theorem probability_excellent (total_selections total_excellent_selections : ℕ) :
  (total_excellent_selections : ℚ) / total_selections = 1 / 10 := by
  sorry

theorem probability_good_or_better (total_selections total_good_or_better_selections : ℕ) :
  (total_good_or_better_selections : ℚ) / total_selections = 7 / 10 := by
  sorry

end probability_excellent_probability_good_or_better_l2396_239677


namespace line_of_intersection_in_standard_form_l2396_239651

noncomputable def plane1 (x y z : ℝ) := 3 * x + 4 * y - 2 * z = 5
noncomputable def plane2 (x y z : ℝ) := 2 * x + 3 * y - z = 3

theorem line_of_intersection_in_standard_form :
  (∃ x y z : ℝ, plane1 x y z ∧ plane2 x y z ∧ (∀ t : ℝ, (x, y, z) = 
  (3 + 2 * t, -1 - t, t))) :=
by {
  sorry
}

end line_of_intersection_in_standard_form_l2396_239651


namespace midpoint_trajectory_of_intersecting_line_l2396_239671

theorem midpoint_trajectory_of_intersecting_line 
    (h₁ : ∀ x y, x^2 + 2 * y^2 = 4) 
    (h₂ : ∀ M: ℝ × ℝ, M = (4, 6)) :
    ∃ x y, (x-2)^2 / 22 + (y-3)^2 / 11 = 1 :=
sorry

end midpoint_trajectory_of_intersecting_line_l2396_239671


namespace otimes_2_3_eq_23_l2396_239692

-- Define the new operation
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- The proof statement
theorem otimes_2_3_eq_23 : otimes 2 3 = 23 := 
  by 
  sorry

end otimes_2_3_eq_23_l2396_239692


namespace problem_statement_l2396_239658

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem problem_statement : f (g 5) - g (f 5) = 63 :=
by
  sorry

end problem_statement_l2396_239658


namespace size_of_smaller_package_l2396_239637

theorem size_of_smaller_package
  (total_coffee : ℕ)
  (n_ten_ounce_packages : ℕ)
  (extra_five_ounce_packages : ℕ)
  (size_smaller_package : ℕ)
  (h1 : total_coffee = 115)
  (h2 : size_smaller_package = 5)
  (h3 : n_ten_ounce_packages = 7)
  (h4 : extra_five_ounce_packages = 2)
  (h5 : total_coffee = n_ten_ounce_packages * 10 + (n_ten_ounce_packages + extra_five_ounce_packages) * size_smaller_package) :
  size_smaller_package = 5 :=
by 
  sorry

end size_of_smaller_package_l2396_239637


namespace total_beats_together_in_week_l2396_239640

theorem total_beats_together_in_week :
  let samantha_beats_per_min := 250
  let samantha_hours_per_day := 3
  let michael_beats_per_min := 180
  let michael_hours_per_day := 2.5
  let days_per_week := 5

  let samantha_beats_per_day := samantha_beats_per_min * 60 * samantha_hours_per_day
  let samantha_beats_per_week := samantha_beats_per_day * days_per_week
  let michael_beats_per_day := michael_beats_per_min * 60 * michael_hours_per_day
  let michael_beats_per_week := michael_beats_per_day * days_per_week
  let total_beats_per_week := samantha_beats_per_week + michael_beats_per_week

  total_beats_per_week = 360000 := 
by
  -- The proof will go here
  sorry

end total_beats_together_in_week_l2396_239640


namespace sufficient_but_not_necessary_l2396_239641

variable (x y : ℝ)

theorem sufficient_but_not_necessary (x_gt_y_gt_zero : x > y ∧ y > 0) : (x / y > 1) :=
by
  sorry

end sufficient_but_not_necessary_l2396_239641


namespace minimum_balls_same_color_minimum_balls_two_white_l2396_239699

-- Define the number of black and white balls.
def num_black_balls : Nat := 100
def num_white_balls : Nat := 100

-- Problem 1: Ensure at least 2 balls of the same color.
theorem minimum_balls_same_color (n_black n_white : Nat) (h_black : n_black = num_black_balls) (h_white : n_white = num_white_balls) : 
  3 ≥ 2 :=
by
  sorry

-- Problem 2: Ensure at least 2 white balls.
theorem minimum_balls_two_white (n_black n_white : Nat) (h_black: n_black = num_black_balls) (h_white: n_white = num_white_balls) :
  102 ≥ 2 :=
by
  sorry

end minimum_balls_same_color_minimum_balls_two_white_l2396_239699


namespace total_sugar_in_all_candy_l2396_239643

-- definitions based on the conditions
def chocolateBars : ℕ := 14
def sugarPerChocolateBar : ℕ := 10
def lollipopSugar : ℕ := 37

-- proof statement
theorem total_sugar_in_all_candy :
  (chocolateBars * sugarPerChocolateBar + lollipopSugar) = 177 := 
by
  sorry

end total_sugar_in_all_candy_l2396_239643


namespace tangent_line_relation_l2396_239622

noncomputable def proof_problem (x1 x2 : ℝ) : Prop :=
  ((∃ (P Q : ℝ × ℝ),
    P = (x1, Real.log x1) ∧
    Q = (x2, Real.exp x2) ∧
    ∀ k : ℝ, Real.exp x2 = k ↔ k * (x2 - x1) = Real.log x1 - Real.exp x2) →
    (((x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0))))


theorem tangent_line_relation (x1 x2 : ℝ) (h : proof_problem x1 x2) : 
  (x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0) :=
sorry

end tangent_line_relation_l2396_239622


namespace ratio_of_square_sides_l2396_239685

theorem ratio_of_square_sides
  (a b : ℝ) 
  (h1 : ∃ square1 : ℝ, square1 = 2 * a)
  (h2 : ∃ square2 : ℝ, square2 = 2 * b)
  (h3 : a ^ 2 - 4 * a * b - 5 * b ^ 2 = 0) :
  2 * a / 2 * b = 5 :=
by
  sorry

end ratio_of_square_sides_l2396_239685


namespace topsoil_cost_correct_l2396_239662

noncomputable def topsoilCost (price_per_cubic_foot : ℝ) (yard_to_foot : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) (volume_in_yards : ℝ) : ℝ :=
  let volume_in_feet := volume_in_yards * yard_to_foot
  let cost_without_discount := volume_in_feet * price_per_cubic_foot
  if volume_in_feet > discount_threshold then
    cost_without_discount * (1 - discount_rate)
  else
    cost_without_discount

theorem topsoil_cost_correct:
  topsoilCost 8 27 100 0.10 7 = 1360.8 :=
by
  sorry

end topsoil_cost_correct_l2396_239662


namespace correct_method_eliminates_y_l2396_239629

def eliminate_y_condition1 (x y : ℝ) : Prop :=
  5 * x + 2 * y = 20

def eliminate_y_condition2 (x y : ℝ) : Prop :=
  4 * x - y = 8

theorem correct_method_eliminates_y (x y : ℝ) :
  eliminate_y_condition1 x y ∧ eliminate_y_condition2 x y →
  5 * x + 2 * y + 2 * (4 * x - y) = 36 :=
by
  sorry

end correct_method_eliminates_y_l2396_239629


namespace sqrt_sum_of_roots_l2396_239647

theorem sqrt_sum_of_roots :
  (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30).sqrt
  = (Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3) :=
by
  sorry

end sqrt_sum_of_roots_l2396_239647


namespace bob_grade_is_35_l2396_239674

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l2396_239674


namespace dolls_total_l2396_239602

theorem dolls_total (dina_dolls ivy_dolls casey_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : (2 / 3 : ℚ) * ivy_dolls = 20)
  (h3 : casey_dolls = 5 * 20) :
  dina_dolls + ivy_dolls + casey_dolls = 190 :=
by sorry

end dolls_total_l2396_239602


namespace find_line_equation_l2396_239609

-- define the condition of passing through the point (-3, -1)
def passes_through (x y : ℝ) (a b : ℝ) := (a = -3) ∧ (b = -1)

-- define the condition of being parallel to the line x - 3y - 1 = 0
def is_parallel (m n c : ℝ) := (m = 1) ∧ (n = -3)

-- theorem statement
theorem find_line_equation (a b : ℝ) (c : ℝ) :
  passes_through a b (-3) (-1) →
  is_parallel 1 (-3) c →
  (a - 3 * b + c = 0) :=
sorry

end find_line_equation_l2396_239609


namespace distribution_value_l2396_239610

def standard_deviation := 2
def mean := 51

theorem distribution_value (x : ℝ) (hx : x < 45) : (mean - 3 * standard_deviation) > x :=
by
  -- Provide the statement without proof
  sorry

end distribution_value_l2396_239610


namespace rectangle_dimension_area_l2396_239672

theorem rectangle_dimension_area (x : ℝ) 
  (h_dim : (3 * x - 5) * (x + 7) = 14 * x - 35) : 
  x = 0 :=
by
  sorry

end rectangle_dimension_area_l2396_239672
