import Mathlib

namespace vertical_asymptote_l2403_240333

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

theorem vertical_asymptote (c : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → ((x^2 - x + c) ≠ 0)) ∨
  (∀ x : ℝ, ((x^2 - x + c) = 0) ↔ (x = 2) ∨ (x = 4)) →
  c = -2 ∨ c = -12 :=
sorry

end vertical_asymptote_l2403_240333


namespace product_of_two_digit_numbers_is_not_five_digits_l2403_240393

theorem product_of_two_digit_numbers_is_not_five_digits :
  ∀ (a b c d : ℕ), (10 ≤ 10 * a + b) → (10 * a + b ≤ 99) → (10 ≤ 10 * c + d) → (10 * c + d ≤ 99) → 
    (10 * a + b) * (10 * c + d) < 10000 :=
by
  intros a b c d H1 H2 H3 H4
  -- proof steps would go here
  sorry

end product_of_two_digit_numbers_is_not_five_digits_l2403_240393


namespace abs_a_gt_abs_b_l2403_240364

variable (a b : Real)

theorem abs_a_gt_abs_b (h1 : a > 0) (h2 : b < 0) (h3 : a + b > 0) : |a| > |b| :=
by
  sorry

end abs_a_gt_abs_b_l2403_240364


namespace min_x_plus_y_l2403_240382

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_l2403_240382


namespace slope_of_chord_l2403_240398

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (P : ℝ × ℝ)
    (hp : P = (3, 2))
    (h1 : 4 * x1 ^ 2 + 9 * y1 ^ 2 = 144)
    (h2 : 4 * x2 ^ 2 + 9 * y2 ^ 2 = 144)
    (h3 : (x1 + x2) / 2 = 3)
    (h4 : (y1 + y2) / 2 = 2) : 
    (y1 - y2) / (x1 - x2) = -2 / 3 :=
by
  sorry

end slope_of_chord_l2403_240398


namespace sequence_formula_l2403_240300

def seq (n : ℕ) : ℕ := 
  match n with
  | 0     => 1
  | (n+1) => 2 * seq n + 3

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) : 
  seq n = 2^n + 1 - 3 :=
sorry

end sequence_formula_l2403_240300


namespace calculate_total_calories_l2403_240309

-- Definition of variables and conditions
def total_calories (C : ℝ) : Prop :=
  let FDA_recommended_intake := 25
  let consumed_calories := FDA_recommended_intake + 5
  (3 / 4) * C = consumed_calories

-- Theorem statement
theorem calculate_total_calories : ∃ C : ℝ, total_calories C ∧ C = 40 :=
by
  sorry  -- Proof will be provided here

end calculate_total_calories_l2403_240309


namespace gcd_372_684_l2403_240318

theorem gcd_372_684 : Int.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l2403_240318


namespace bianca_next_day_run_l2403_240381

-- Define the conditions
variable (miles_first_day : ℕ) (total_miles : ℕ)

-- Set the conditions for Bianca's run
def conditions := miles_first_day = 8 ∧ total_miles = 12

-- State the proposition we need to prove
def miles_next_day (miles_first_day total_miles : ℕ) : ℕ := total_miles - miles_first_day

-- The theorem stating the problem to prove
theorem bianca_next_day_run (h : conditions 8 12) : miles_next_day 8 12 = 4 := by
  unfold conditions at h
  simp [miles_next_day] at h
  sorry

end bianca_next_day_run_l2403_240381


namespace arithmetic_sequence_common_difference_l2403_240369

/--
Given an arithmetic sequence $\{a_n\}$ and $S_n$ being the sum of the first $n$ terms, 
with $a_1=1$ and $S_3=9$, prove that the common difference $d$ is equal to $2$.
-/
theorem arithmetic_sequence_common_difference :
  ∃ (d : ℝ), (∀ (n : ℕ), aₙ = 1 + (n - 1) * d) ∧ S₃ = a₁ + (a₁ + d) + (a₁ + 2 * d) ∧ a₁ = 1 ∧ S₃ = 9 → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l2403_240369


namespace solve_inequality_l2403_240327

theorem solve_inequality (x : ℝ) : 
  (0 < (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6))) ↔ 
  (x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x) :=
by 
  sorry

end solve_inequality_l2403_240327


namespace unique_nat_pair_l2403_240307

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end unique_nat_pair_l2403_240307


namespace flight_duration_l2403_240399

theorem flight_duration :
  ∀ (h m : ℕ),
  3 * 60 + 42 = 15 * 60 + 57 →
  0 < m ∧ m < 60 →
  h + m = 18 :=
by
  intros h m h_def hm_bound
  sorry

end flight_duration_l2403_240399


namespace fraction_draw_l2403_240332

theorem fraction_draw (john_wins : ℚ) (mike_wins : ℚ) (h_john : john_wins = 4 / 9) (h_mike : mike_wins = 5 / 18) :
    1 - (john_wins + mike_wins) = 5 / 18 :=
by
    rw [h_john, h_mike]
    sorry

end fraction_draw_l2403_240332


namespace largest_base7_three_digit_is_342_l2403_240363

-- Definition of the base-7 number 666
def base7_666 : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

-- The largest decimal number represented by a three-digit base-7 number is 342
theorem largest_base7_three_digit_is_342 : base7_666 = 342 := by
  sorry

end largest_base7_three_digit_is_342_l2403_240363


namespace exists_indices_for_sequences_l2403_240361

theorem exists_indices_for_sequences 
  (a b c : ℕ → ℕ) :
  ∃ (p q : ℕ), p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_indices_for_sequences_l2403_240361


namespace modulo_sum_of_99_plus_5_l2403_240376

theorem modulo_sum_of_99_plus_5 : let s_n := (99 / 2) * (2 * 1 + (99 - 1) * 1)
                                 let sum_with_5 := s_n + 5
                                 sum_with_5 % 7 = 6 :=
by
  sorry

end modulo_sum_of_99_plus_5_l2403_240376


namespace find_pairs_l2403_240366

theorem find_pairs (m n : ℕ) (h : m > 0 ∧ n > 0 ∧ m^2 = n^2 + m + n + 2018) :
  (m, n) = (1010, 1008) ∨ (m, n) = (506, 503) :=
by sorry

end find_pairs_l2403_240366


namespace gcd_is_18_l2403_240370

-- Define gcdX that represents the greatest common divisor of X and Y.
noncomputable def gcdX (X Y : ℕ) : ℕ := Nat.gcd X Y

-- Given conditions
def cond_lcm (X Y : ℕ) : Prop := Nat.lcm X Y = 180
def cond_ratio (X Y : ℕ) : Prop := ∃ k : ℕ, X = 2 * k ∧ Y = 5 * k

-- Theorem to prove that the gcd of X and Y is 18
theorem gcd_is_18 {X Y : ℕ} (h1 : cond_lcm X Y) (h2 : cond_ratio X Y) : gcdX X Y = 18 :=
by
  sorry

end gcd_is_18_l2403_240370


namespace triangle_ctg_inequality_l2403_240379

noncomputable def ctg (x : Real) := Real.cos x / Real.sin x

theorem triangle_ctg_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  ctg α ^ 2 + ctg β ^ 2 + ctg γ ^ 2 ≥ 1 :=
sorry

end triangle_ctg_inequality_l2403_240379


namespace monotonically_increasing_interval_l2403_240324

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem monotonically_increasing_interval : 
  ∀ x ∈ Set.Icc (-Real.pi) 0, 
  x ∈ Set.Icc (-Real.pi/6) 0 ↔ deriv f x = 0 := sorry

end monotonically_increasing_interval_l2403_240324


namespace theta_range_l2403_240394

noncomputable def f (x θ : ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_range (θ : ℝ) (k : ℤ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x θ > 0) →
  θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
sorry

end theta_range_l2403_240394


namespace largest_possible_3_digit_sum_l2403_240362

theorem largest_possible_3_digit_sum (X Y Z : ℕ) (h_diff : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
(h_digit_X : 0 ≤ X ∧ X ≤ 9) (h_digit_Y : 0 ≤ Y ∧ Y ≤ 9) (h_digit_Z : 0 ≤ Z ∧ Z ≤ 9) :
  (100 * X + 10 * X + X) + (10 * Y + X) + X = 994 → (X, Y, Z) = (8, 9, 0) := by
  sorry

end largest_possible_3_digit_sum_l2403_240362


namespace min_max_value_l2403_240372

-- Definition of the function to be minimized and maximized
def f (x y : ℝ) : ℝ := |x^3 - x * y^2|

-- Conditions
def x_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def y_condition (y : ℝ) : Prop := true

-- Goal: Prove the minimum of the maximum value
theorem min_max_value :
  ∃ y : ℝ, (∀ x : ℝ, x_condition x → f x y ≤ 8) ∧ (∀ y' : ℝ, (∀ x : ℝ, x_condition x → f x y' ≤ 8) → y' = y) :=
sorry

end min_max_value_l2403_240372


namespace f_2015_equals_2_l2403_240326

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_2015_equals_2 (f_even : ∀ x : ℝ, f (-x) = f x)
    (f_shift : ∀ x : ℝ, f (-x) = f (2 + x))
    (f_log : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log (3 * x + 1) / Real.log 2) :
    f 2015 = 2 :=
sorry

end f_2015_equals_2_l2403_240326


namespace meanScore_is_91_666_l2403_240342

-- Define Jane's quiz scores
def janesScores : List ℕ := [85, 88, 90, 92, 95, 100]

-- Define the total sum of Jane's quiz scores
def sumScores (scores : List ℕ) : ℕ := scores.foldl (· + ·) 0

-- The number of Jane's quiz scores
def numberOfScores (scores : List ℕ) : ℕ := scores.length

-- Define the mean of Jane's quiz scores
def meanScore (scores : List ℕ) : ℚ := sumScores scores / numberOfScores scores

-- The theorem to be proven
theorem meanScore_is_91_666 (h : janesScores = [85, 88, 90, 92, 95, 100]) :
  meanScore janesScores = 91.66666666666667 := by 
  sorry

end meanScore_is_91_666_l2403_240342


namespace meetings_percentage_l2403_240319

def workday_hours := 10
def first_meeting_minutes := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_workday_minutes := workday_hours * 60
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

theorem meetings_percentage :
    (total_meeting_minutes / total_workday_minutes) * 100 = 40 :=
by
  sorry

end meetings_percentage_l2403_240319


namespace goods_train_length_is_280_l2403_240347

noncomputable def length_of_goods_train (passenger_speed passenger_speed_kmh: ℝ) 
                                       (goods_speed goods_speed_kmh: ℝ) 
                                       (time_to_pass: ℝ) : ℝ :=
  let kmh_to_ms := (1000 : ℝ) / (3600 : ℝ)
  let passenger_speed_ms := passenger_speed * kmh_to_ms
  let goods_speed_ms     := goods_speed * kmh_to_ms
  let relative_speed     := passenger_speed_ms + goods_speed_ms
  relative_speed * time_to_pass

theorem goods_train_length_is_280 :
  length_of_goods_train 70 70 42 42 9 = 280 :=
by
  sorry

end goods_train_length_is_280_l2403_240347


namespace compute_fraction_product_l2403_240310

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l2403_240310


namespace xyz_sum_l2403_240375

theorem xyz_sum (x y z : ℝ) 
  (h1 : y + z = 17 - 2 * x) 
  (h2 : x + z = 1 - 2 * y) 
  (h3 : x + y = 8 - 2 * z) : 
  x + y + z = 6.5 :=
sorry

end xyz_sum_l2403_240375


namespace population_increase_time_l2403_240359

theorem population_increase_time (persons_added : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (time_for_one_person : ℕ) :
  persons_added = 160 →
  time_minutes = 40 →
  seconds_per_minute = 60 →
  total_seconds = time_minutes * seconds_per_minute →
  time_for_one_person = total_seconds / persons_added →
  time_for_one_person = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_increase_time_l2403_240359


namespace solve_inequality_l2403_240315

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem solve_inequality (x : ℝ) : (otimes (x-2) (x+2) < 2) ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by
  sorry

end solve_inequality_l2403_240315


namespace perpendicular_vectors_l2403_240335

theorem perpendicular_vectors (x y : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (2 + x, 1 - y)) 
  (hperp : (a.1 * b.1 + a.2 * b.2 = 0)) : 2 * y - x = 4 :=
sorry

end perpendicular_vectors_l2403_240335


namespace calculation_result_l2403_240374

theorem calculation_result : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end calculation_result_l2403_240374


namespace fifth_term_of_geometric_sequence_l2403_240340

theorem fifth_term_of_geometric_sequence (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h_a : a = 5) (h_fourth_term : a * r^3 = 405) :
  a * r^4 = 405 := by
  sorry

end fifth_term_of_geometric_sequence_l2403_240340


namespace woman_needs_butter_l2403_240311

noncomputable def butter_needed (cost_package : ℝ) (cost_8oz : ℝ) (cost_4oz : ℝ) 
                                (discount : ℝ) (lowest_price : ℝ) : ℝ :=
  if lowest_price = cost_8oz + 2 * (cost_4oz * discount / 100) then 8 + 2 * 4 else 0

theorem woman_needs_butter 
  (cost_single_package : ℝ := 7) 
  (cost_8oz_package : ℝ := 4) 
  (cost_4oz_package : ℝ := 2)
  (discount_4oz_package : ℝ := 50) 
  (lowest_price_payment : ℝ := 6) :
  butter_needed cost_single_package cost_8oz_package cost_4oz_package discount_4oz_package lowest_price_payment = 16 := 
by
  sorry

end woman_needs_butter_l2403_240311


namespace yellow_peaches_l2403_240308

theorem yellow_peaches (red_peaches green_peaches total_green_yellow_peaches : ℕ)
  (h1 : red_peaches = 5)
  (h2 : green_peaches = 6)
  (h3 : total_green_yellow_peaches = 20) :
  (total_green_yellow_peaches - green_peaches) = 14 :=
by
  sorry

end yellow_peaches_l2403_240308


namespace num_fixed_last_two_digits_l2403_240389

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits_l2403_240389


namespace election_at_least_one_past_officer_l2403_240343

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem election_at_least_one_past_officer : 
  let total_candidates := 16
  let past_officers := 7
  let officer_positions := 5
  choose total_candidates officer_positions - choose (total_candidates - past_officers) officer_positions = 4242 :=
by
  sorry

end election_at_least_one_past_officer_l2403_240343


namespace calculate_expr_l2403_240386

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l2403_240386


namespace problem_solution_exists_l2403_240312

theorem problem_solution_exists {x : ℝ} :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 =
    a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 +
    a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
    a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7)
  → a_2 = 56 :=
sorry

end problem_solution_exists_l2403_240312


namespace union_eq_M_l2403_240301

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def S : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem union_eq_M : M ∪ S = M := by
  /- this part is for skipping the proof -/
  sorry

end union_eq_M_l2403_240301


namespace find_a_value_l2403_240396

theorem find_a_value (a x y : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x - 3 * y = 3) : a = 6 :=
by
  rw [h1, h2] at h3 -- Substitute x and y values into the equation
  sorry -- The proof is omitted as per instructions.

end find_a_value_l2403_240396


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l2403_240380

theorem problem_1 (x y z : ℝ) (h : z = (x + y) / 2) : z = (x + y) / 2 :=
sorry

theorem problem_2 (x y w : ℝ) (h1 : w = x + y) : w = x + y :=
sorry

theorem problem_3 (x w y : ℝ) (h1 : w = x + y) (h2 : y = w - x) : y = w - x :=
sorry

theorem problem_4 (x z v : ℝ) (h1 : z = (x + y) / 2) (h2 : v = 2 * z) : v = 2 * (x + (x + y) / 2) :=
sorry

theorem problem_5 (x z u : ℝ) (h : u = - (x + z) / 5) : x + z + 5 * u = 0 :=
sorry

theorem problem_6 (y z t : ℝ) (h : t = (6 + y + z) / 2) : t = (6 + y + z) / 2 :=
sorry

theorem problem_7 (y z s : ℝ) (h : y + z + 4 * s - 10 = 0) : y + z + 4 * s - 10 = 0 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l2403_240380


namespace average_class_weight_l2403_240377

theorem average_class_weight :
  let students_A := 50
  let weight_A := 60
  let students_B := 60
  let weight_B := 80
  let students_C := 70
  let weight_C := 75
  let students_D := 80
  let weight_D := 85
  let total_students := students_A + students_B + students_C + students_D
  let total_weight := students_A * weight_A + students_B * weight_B + students_C * weight_C + students_D * weight_D
  (total_weight / total_students : ℝ) = 76.35 :=
by
  sorry

end average_class_weight_l2403_240377


namespace volume_of_box_l2403_240387

variable (width length height : ℝ)
variable (Volume : ℝ)

-- Given conditions
def w : ℝ := 9
def l : ℝ := 4
def h : ℝ := 7

-- The statement to prove
theorem volume_of_box : Volume = l * w * h := by
  sorry

end volume_of_box_l2403_240387


namespace infinitely_many_singular_pairs_l2403_240316

def largestPrimeFactor (n : ℕ) : ℕ := sorry -- definition of largest prime factor

def isSingularPair (p q : ℕ) : Prop :=
  p ≠ q ∧ ∀ (n : ℕ), n ≥ 2 → largestPrimeFactor n * largestPrimeFactor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs : ∃ (S : ℕ → (ℕ × ℕ)), ∀ i, isSingularPair (S i).1 (S i).2 :=
sorry

end infinitely_many_singular_pairs_l2403_240316


namespace cost_of_meal_l2403_240384

noncomputable def total_cost (hamburger_cost fry_cost drink_cost : ℕ) (num_hamburgers num_fries num_drinks : ℕ) (discount_rate : ℕ) : ℕ :=
  let initial_cost := (hamburger_cost * num_hamburgers) + (fry_cost * num_fries) + (drink_cost * num_drinks)
  let discount := initial_cost * discount_rate / 100
  initial_cost - discount

theorem cost_of_meal :
  total_cost 5 3 2 3 4 6 10 = 35 := by
  sorry

end cost_of_meal_l2403_240384


namespace part_a_l2403_240317

theorem part_a (m : ℕ) (A B : ℕ) (hA : A = (10^(2 * m) - 1) / 9) (hB : B = 4 * ((10^m - 1) / 9)) :
  ∃ k : ℕ, A + B + 1 = k^2 :=
sorry

end part_a_l2403_240317


namespace general_term_formula_sum_of_2_pow_an_l2403_240350

variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

axiom S5_eq_30 : S 5 = 30
axiom a1_a6_eq_14 : a 1 + a 6 = 14

theorem general_term_formula : ∀ n, a n = 2 * n :=
sorry

theorem sum_of_2_pow_an (n : ℕ) : T n = (4^(n + 1)) / 3 - 4 / 3 :=
sorry

end general_term_formula_sum_of_2_pow_an_l2403_240350


namespace phone_prices_purchase_plans_l2403_240395

noncomputable def modelA_price : ℝ := 2000
noncomputable def modelB_price : ℝ := 1000

theorem phone_prices :
  (∀ x y : ℝ, (2 * x + y = 5000 ∧ 3 * x + 2 * y = 8000) → x = modelA_price ∧ y = modelB_price) :=
by
    intro x y
    intro h
    have h1 := h.1
    have h2 := h.2
    -- We would provide the detailed proof here
    sorry

theorem purchase_plans :
  (∀ a : ℕ, (4 ≤ a ∧ a ≤ 6) ↔ (24000 ≤ 2000 * a + 1000 * (20 - a) ∧ 2000 * a + 1000 * (20 - a) ≤ 26000)) :=
by
    intro a
    -- We would provide the detailed proof here
    sorry

end phone_prices_purchase_plans_l2403_240395


namespace ring_width_l2403_240304

noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def outerCircumference : ℝ := 528 / 7

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem ring_width :
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  r_outer - r_inner = 4 :=
by
  -- Definitions for inner and outer radius
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  -- Proof goes here
  sorry

end ring_width_l2403_240304


namespace chocolate_syrup_per_glass_l2403_240306

-- Definitions from the conditions
def each_glass_volume : ℝ := 8
def milk_per_glass : ℝ := 6.5
def total_milk : ℝ := 130
def total_chocolate_syrup : ℝ := 60
def total_chocolate_milk : ℝ := 160

-- Proposition and statement to prove
theorem chocolate_syrup_per_glass : 
  (total_chocolate_milk / each_glass_volume) * milk_per_glass = total_milk → 
  (each_glass_volume - milk_per_glass = 1.5) := 
by 
  sorry

end chocolate_syrup_per_glass_l2403_240306


namespace age_of_oldest_sibling_l2403_240322

theorem age_of_oldest_sibling (Kay_siblings : ℕ) (Kay_age : ℕ) (youngest_sibling_age : ℕ) (oldest_sibling_age : ℕ) 
  (h1 : Kay_siblings = 14) (h2 : Kay_age = 32) (h3 : youngest_sibling_age = Kay_age / 2 - 5) 
  (h4 : oldest_sibling_age = 4 * youngest_sibling_age) : oldest_sibling_age = 44 := 
sorry

end age_of_oldest_sibling_l2403_240322


namespace molecular_weight_of_compound_l2403_240354

-- Define the atomic weights for Hydrogen, Chlorine, and Oxygen
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_O : ℝ := 15.999

-- Define the molecular weight of the compound
def molecular_weight (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

-- The proof problem statement
theorem molecular_weight_of_compound :
  molecular_weight atomic_weight_H atomic_weight_Cl atomic_weight_O = 68.456 :=
sorry

end molecular_weight_of_compound_l2403_240354


namespace arithmetic_sequence_properties_l2403_240323

noncomputable def general_term_formula (a₁ : ℕ) (S₃ : ℕ) (n : ℕ) (d : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def sum_of_double_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (2 * (a₁ + (n - 1) * d)) * n / 2

theorem arithmetic_sequence_properties
  (a₁ : ℕ) (S₃ : ℕ)
  (h₁ : a₁ = 2)
  (h₂ : S₃ = 9) :
  general_term_formula a₁ S₃ n (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) = n + 1 ∧
  sum_of_double_sequence a₁ (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) n = 2^(n+2) - 4 :=
by
  sorry

end arithmetic_sequence_properties_l2403_240323


namespace simplify_polynomial_l2403_240392

variable (x : ℝ)

theorem simplify_polynomial :
  (6*x^10 + 8*x^9 + 3*x^7) + (2*x^12 + 3*x^10 + x^9 + 5*x^7 + 4*x^4 + 7*x + 6) =
  2*x^12 + 9*x^10 + 9*x^9 + 8*x^7 + 4*x^4 + 7*x + 6 :=
by
  sorry

end simplify_polynomial_l2403_240392


namespace quadratic_solution_transform_l2403_240345

theorem quadratic_solution_transform (a b c : ℝ) (hA : 0 = a * (-3)^2 + b * (-3) + c) (hB : 0 = a * 4^2 + b * 4 + c) :
  (∃ x1 x2 : ℝ, a * (x1 - 1)^2 + b * (x1 - 1) + c = 0 ∧ a * (x2 - 1)^2 + b * (x2 - 1) + c = 0 ∧ x1 = -2 ∧ x2 = 5) :=
  sorry

end quadratic_solution_transform_l2403_240345


namespace cube_root_of_64_eq_two_pow_m_l2403_240321

theorem cube_root_of_64_eq_two_pow_m (m : ℕ) (h : (64 : ℝ) ^ (1 / 3) = (2 : ℝ) ^ m) : m = 2 := 
sorry

end cube_root_of_64_eq_two_pow_m_l2403_240321


namespace evaporation_amount_l2403_240341

variable (E : ℝ)

def initial_koolaid_powder : ℝ := 2
def initial_water : ℝ := 16
def final_percentage : ℝ := 0.04

theorem evaporation_amount :
  (initial_koolaid_powder = 2) →
  (initial_water = 16) →
  (0.04 * (initial_koolaid_powder + 4 * (initial_water - E)) = initial_koolaid_powder) →
  E = 4 :=
by
  intros h1 h2 h3
  sorry

end evaporation_amount_l2403_240341


namespace find_purple_balls_count_l2403_240371

theorem find_purple_balls_count (k : ℕ) (h : ∃ k > 0, (21 - 3 * k) = (3 / 4) * (7 + k)) : k = 4 :=
sorry

end find_purple_balls_count_l2403_240371


namespace rahul_batting_average_l2403_240331

theorem rahul_batting_average:
  ∃ (A : ℝ), A = 46 ∧
  (∀ (R : ℝ), R = 138 → R = 54 * 4 - 78 → A = R / 3) ∧
  ∃ (n_matches : ℕ), n_matches = 3 :=
by
  sorry

end rahul_batting_average_l2403_240331


namespace solve_arithmetic_sequence_sum_l2403_240303

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l2403_240303


namespace members_didnt_show_up_l2403_240337

theorem members_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  total_members = 14 →
  points_per_member = 5 →
  total_points = 35 →
  total_members - (total_points / points_per_member) = 7 :=
by
  intros
  sorry

end members_didnt_show_up_l2403_240337


namespace laura_walk_distance_l2403_240373

theorem laura_walk_distance 
  (east_blocks : ℕ) 
  (north_blocks : ℕ) 
  (block_length_miles : ℕ → ℝ) 
  (h_east_blocks : east_blocks = 8) 
  (h_north_blocks : north_blocks = 14) 
  (h_block_length_miles : ∀ b : ℕ, b = 1 → block_length_miles b = 1 / 4) 
  : (east_blocks + north_blocks) * block_length_miles 1 = 5.5 := 
by 
  sorry

end laura_walk_distance_l2403_240373


namespace average_first_15_even_numbers_l2403_240334

theorem average_first_15_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30) / 15 = 16 :=
by 
  sorry

end average_first_15_even_numbers_l2403_240334


namespace calculate_expression_l2403_240385

theorem calculate_expression : (-1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0) = Real.sqrt 2 - 3 :=
by
  sorry

end calculate_expression_l2403_240385


namespace max_quotient_l2403_240352

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  ∃ max_val, max_val = 225 ∧ ∀ (x y : ℝ), (100 ≤ x ∧ x ≤ 300) ∧ (500 ≤ y ∧ y ≤ 1500) → (y^2 / x^2) ≤ max_val := 
by
  use 225
  sorry

end max_quotient_l2403_240352


namespace tylenol_pill_mg_l2403_240338

noncomputable def tylenol_dose_per_pill : ℕ :=
  let mg_per_dose := 1000
  let hours_per_dose := 6
  let days := 14
  let pills := 112
  let doses_per_day := 24 / hours_per_dose
  let total_doses := doses_per_day * days
  let total_mg := total_doses * mg_per_dose
  total_mg / pills

theorem tylenol_pill_mg :
  tylenol_dose_per_pill = 500 := by
  sorry

end tylenol_pill_mg_l2403_240338


namespace div_by_64_l2403_240349

theorem div_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (5^n - 8*n^2 + 4*n - 1) :=
sorry

end div_by_64_l2403_240349


namespace probability_second_year_not_science_l2403_240378

def total_students := 2000

def first_year := 600
def first_year_science := 300
def first_year_arts := 200
def first_year_engineering := 100

def second_year := 450
def second_year_science := 250
def second_year_arts := 150
def second_year_engineering := 50

def third_year := 550
def third_year_science := 300
def third_year_arts := 200
def third_year_engineering := 50

def postgraduate := 400
def postgraduate_science := 200
def postgraduate_arts := 100
def postgraduate_engineering := 100

def not_third_year_not_science :=
  (first_year_arts + first_year_engineering) +
  (second_year_arts + second_year_engineering) +
  (postgraduate_arts + postgraduate_engineering)

def second_year_not_science := second_year_arts + second_year_engineering

theorem probability_second_year_not_science :
  (second_year_not_science / not_third_year_not_science : ℚ) = (2 / 7 : ℚ) :=
by
  let total := (first_year_arts + first_year_engineering) + (second_year_arts + second_year_engineering) + (postgraduate_arts + postgraduate_engineering)
  have not_third_year_not_science : total = 300 + 200 + 200 := by sorry
  have second_year_not_science_eq : second_year_not_science = 200 := by sorry
  sorry

end probability_second_year_not_science_l2403_240378


namespace intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l2403_240367

variable (P Q : Prop)

theorem intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false
  (h : P ∧ Q = False) : (P ∨ Q = False) ↔ (P ∧ Q = False) := 
by 
  sorry

end intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l2403_240367


namespace find_x_l2403_240328

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l2403_240328


namespace binomial_square_expression_l2403_240360

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l2403_240360


namespace problem_1_problem_2_l2403_240305

-- Statements for our proof problems
theorem problem_1 (a b : ℝ) : a^2 + b^2 ≥ 2 * (2 * a - b) - 5 :=
sorry

theorem problem_2 (a b : ℝ) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) ∧ (a = b ↔ a^a * b^b = (a * b)^((a + b) / 2)) :=
sorry

end problem_1_problem_2_l2403_240305


namespace largest_rectangle_area_l2403_240330

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l2403_240330


namespace largest_four_digit_number_last_digit_l2403_240355

theorem largest_four_digit_number_last_digit (n : ℕ) (n' : ℕ) (m r a b : ℕ) :
  (1000 * m + 100 * r + 10 * a + b = n) →
  (100 * m + 10 * r + a = n') →
  (n % 9 = 0) →
  (n' % 4 = 0) →
  b = 3 :=
by
  sorry

end largest_four_digit_number_last_digit_l2403_240355


namespace balance_balls_l2403_240353

variable (G B Y W R : ℕ)

theorem balance_balls :
  (4 * G = 8 * B) →
  (3 * Y = 7 * B) →
  (8 * B = 5 * W) →
  (2 * R = 6 * B) →
  (5 * G + 3 * Y + 3 * R = 26 * B) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end balance_balls_l2403_240353


namespace correct_sampling_methods_l2403_240388

/-- 
Given:
1. A group of 500 senior year students with the following blood type distribution: 200 with blood type O,
125 with blood type A, 125 with blood type B, and 50 with blood type AB.
2. A task to select a sample of 20 students to study the relationship between blood type and color blindness.
3. A high school soccer team consisting of 11 players, and the need to draw 2 players to investigate their study load.
4. Sampling methods: I. Random sampling, II. Systematic sampling, III. Stratified sampling.

Prove:
The correct sampling methods are: Stratified sampling (III) for the blood type-color blindness study and
Random sampling (I) for the soccer team study.
-/ 

theorem correct_sampling_methods (students : Finset ℕ) (blood_type_O blood_type_A blood_type_B blood_type_AB : ℕ)
  (sample_size_students soccer_team_size draw_size_soccer_team : ℕ)
  (sampling_methods : Finset ℕ) : 
  (students.card = 500) →
  (blood_type_O = 200) →
  (blood_type_A = 125) →
  (blood_type_B = 125) →
  (blood_type_AB = 50) →
  (sample_size_students = 20) →
  (soccer_team_size = 11) →
  (draw_size_soccer_team = 2) →
  (sampling_methods = {1, 2, 3}) →
  (s = (3, 1)) :=
by
  sorry

end correct_sampling_methods_l2403_240388


namespace y_explicit_and_range_l2403_240368

theorem y_explicit_and_range (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2*(m-1)*x1 + m + 1 = 0) (h2 : x2^2 - 2*(m-1)*x2 + m + 1 = 0) :
  x1 + x2 = 2*(m-1) ∧ x1 * x2 = m + 1 ∧ (x1^2 + x2^2 = 4*m^2 - 10*m + 2) 
  ∧ ∀ (y : ℝ), (∃ m, y = 4*m^2 - 10*m + 2) → y ≥ 6 :=
by
  sorry

end y_explicit_and_range_l2403_240368


namespace trigonometric_identity_l2403_240302

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 :=
by
  sorry

end trigonometric_identity_l2403_240302


namespace box_dimensions_sum_l2403_240339

theorem box_dimensions_sum (X Y Z : ℝ) (hXY : X * Y = 18) (hXZ : X * Z = 54) (hYZ : Y * Z = 36) (hX_pos : X > 0) (hY_pos : Y > 0) (hZ_pos : Z > 0) :
  X + Y + Z = 11 := 
sorry

end box_dimensions_sum_l2403_240339


namespace find_g_l2403_240357

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := sorry -- We will define this later in the statement

theorem find_g :
  (∀ x : ℝ, g (x + 2) = f x) →
  (∀ x : ℝ, g x = 2 * x - 1) :=
by
  intros h
  sorry

end find_g_l2403_240357


namespace prime_p_satisfies_condition_l2403_240383

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_p_satisfies_condition {p : ℕ} (hp : is_prime p) (hp2_8 : is_prime (p^2 + 8)) : p = 3 :=
sorry

end prime_p_satisfies_condition_l2403_240383


namespace unique_solution_l2403_240346

theorem unique_solution :
  ∀ (x y z n : ℕ), n ≥ 2 → z ≤ 5 * 2^(2 * n) → (x^ (2 * n + 1) - y^ (2 * n + 1) = x * y * z + 2^(2 * n + 1)) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  intros x y z n hn hzn hxyz
  sorry

end unique_solution_l2403_240346


namespace set_union_intersection_l2403_240325

-- Definitions
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}
def C : Set ℤ := {1, 2}

-- Theorem statement
theorem set_union_intersection : (A ∩ B ∪ C) = {0, 1, 2} :=
by
  sorry

end set_union_intersection_l2403_240325


namespace eval_expr_at_sqrt3_minus_3_l2403_240320

noncomputable def expr (a : ℝ) : ℝ :=
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2))

theorem eval_expr_at_sqrt3_minus_3 : expr (Real.sqrt 3 - 3) = -Real.sqrt 3 / 6 := 
  by sorry

end eval_expr_at_sqrt3_minus_3_l2403_240320


namespace three_digit_odd_number_is_803_l2403_240397

theorem three_digit_odd_number_is_803 :
  ∃ (a b c : ℕ), 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ c % 2 = 1 ∧
  100 * a + 10 * b + c = 803 ∧ (100 * a + 10 * b + c) / 11 = a^2 + b^2 + c^2 :=
by {
  sorry
}

end three_digit_odd_number_is_803_l2403_240397


namespace solution_correct_l2403_240365

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 ≤ |x - 3| ∧ |x - 3| ≤ 5
def condition2 (x : ℝ) : Prop := (x - 3) ^ 2 ≤ 16

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7)

-- Prove that the solution set is correct given the conditions
theorem solution_correct (x : ℝ) : condition1 x ∧ condition2 x ↔ solution_set x :=
by
  sorry

end solution_correct_l2403_240365


namespace pairings_equal_l2403_240351

-- Definitions for City A
def A_girls (n : ℕ) : Type := Fin n
def A_boys (n : ℕ) : Type := Fin n
def A_knows (n : ℕ) (g : A_girls n) (b : A_boys n) : Prop := True

-- Definitions for City B
def B_girls (n : ℕ) : Type := Fin n
def B_boys (n : ℕ) : Type := Fin (2 * n - 1)
def B_knows (n : ℕ) (i : Fin n) (j : Fin (2 * n - 1)) : Prop :=
  j.val < 2 * (i.val + 1)

-- Function to count the number of ways to pair r girls and r boys in city A
noncomputable def A (n r : ℕ) : ℕ := 
  if h : r ≤ n then 
    Nat.choose n r * Nat.choose n r * (r.factorial)
  else 0

-- Recurrence relation for city B
noncomputable def B (n r : ℕ) : ℕ :=
  if r = 0 then 1 else if r > n then 0 else
  if n < 2 then if r = 1 then (2 - 1) * 2 else 0 else
  B (n - 1) r + (2 * n - r) * B (n - 1) (r - 1)

-- We want to prove that number of pairings in city A equals number of pairings in city B for any r <= n
theorem pairings_equal (n r : ℕ) (h : r ≤ n) : A n r = B n r := sorry

end pairings_equal_l2403_240351


namespace rational_iff_arithmetic_progression_l2403_240344

theorem rational_iff_arithmetic_progression (x : ℝ) : 
  (∃ (i j k : ℤ), i < j ∧ j < k ∧ (x + i) + (x + k) = 2 * (x + j)) ↔ 
  (∃ n d : ℤ, d ≠ 0 ∧ x = n / d) := 
sorry

end rational_iff_arithmetic_progression_l2403_240344


namespace circles_intersect_iff_l2403_240313

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end circles_intersect_iff_l2403_240313


namespace time_to_cross_platform_is_correct_l2403_240391

noncomputable def speed_of_train := 36 -- speed in km/h
noncomputable def time_to_cross_pole := 12 -- time in seconds
noncomputable def time_to_cross_platform := 49.996960243180546 -- time in seconds

theorem time_to_cross_platform_is_correct : time_to_cross_platform = 49.996960243180546 := by
  sorry

end time_to_cross_platform_is_correct_l2403_240391


namespace find_functions_l2403_240356

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def domain (f g : ℝ → ℝ) : Prop := ∀ x, x ≠ 1 → x ≠ -1 → true

theorem find_functions
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_domain : domain f g)
  (h_eq : ∀ x, x ≠ 1 → x ≠ -1 → f x + g x = 1 / (x - 1)) :
  (∀ x, x ≠ 1 → x ≠ -1 → f x = x / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 → x ≠ -1 → g x = 1 / (x^2 - 1)) := 
by
  sorry

end find_functions_l2403_240356


namespace intersection_set_l2403_240358

-- Definition of the sets A and B
def setA : Set ℝ := { x | -2 < x ∧ x < 2 }
def setB : Set ℝ := { x | x < 0.5 }

-- The main theorem: Finding the intersection A ∩ B
theorem intersection_set : { x : ℝ | -2 < x ∧ x < 0.5 } = setA ∩ setB := by
  sorry

end intersection_set_l2403_240358


namespace find_total_cost_l2403_240348

-- Define the cost per kg for flour
def F : ℕ := 21

-- Conditions in the problem
axiom cost_eq_mangos_rice (M R : ℕ) : 10 * M = 10 * R
axiom cost_eq_flour_rice (R : ℕ) : 6 * F = 2 * R

-- Define the cost calculations
def total_cost (M R F : ℕ) : ℕ := (4 * M) + (3 * R) + (5 * F)

-- Prove the total cost given the conditions
theorem find_total_cost (M R : ℕ) (h1 : 10 * M = 10 * R) (h2 : 6 * F = 2 * R) : total_cost M R F = 546 :=
sorry

end find_total_cost_l2403_240348


namespace group_photo_arrangements_l2403_240314

theorem group_photo_arrangements :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
    ∀ (M G H P1 P2 : ℕ),
    (M = G + 1 ∨ M + 1 = G) ∧ (M ≠ H - 1 ∧ M ≠ H + 1) →
    arrangements = 36 :=
by {
  sorry
}

end group_photo_arrangements_l2403_240314


namespace number_of_pieces_sold_on_third_day_l2403_240329

variable (m : ℕ)

def first_day_sales : ℕ := m
def second_day_sales : ℕ := (m / 2) - 3
def third_day_sales : ℕ := second_day_sales m + 5

theorem number_of_pieces_sold_on_third_day :
  third_day_sales m = (m / 2) + 2 := by sorry

end number_of_pieces_sold_on_third_day_l2403_240329


namespace complete_square_l2403_240390

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intro h
  sorry

end complete_square_l2403_240390


namespace average_growth_rate_l2403_240336

theorem average_growth_rate (x : ℝ) (hx : (1 + x)^2 = 1.44) : x < 0.22 :=
sorry

end average_growth_rate_l2403_240336
