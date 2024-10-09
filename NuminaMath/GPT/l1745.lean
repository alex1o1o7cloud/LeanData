import Mathlib

namespace monkey_reaches_top_in_19_minutes_l1745_174583

theorem monkey_reaches_top_in_19_minutes (pole_height : ℕ) (ascend_first_min : ℕ) (slip_every_alternate_min : ℕ) 
    (total_minutes : ℕ) (net_gain_two_min : ℕ) : 
    pole_height = 10 ∧ ascend_first_min = 2 ∧ slip_every_alternate_min = 1 ∧ net_gain_two_min = 1 ∧ total_minutes = 19 →
    (net_gain_two_min * (total_minutes - 1) / 2 + ascend_first_min = pole_height) := 
by
    intros
    sorry

end monkey_reaches_top_in_19_minutes_l1745_174583


namespace percentage_of_acid_in_original_mixture_l1745_174503

theorem percentage_of_acid_in_original_mixture
  (a w : ℚ)
  (h1 : a / (a + w + 2) = 18 / 100)
  (h2 : (a + 2) / (a + w + 4) = 30 / 100) :
  (a / (a + w)) * 100 = 29 := 
sorry

end percentage_of_acid_in_original_mixture_l1745_174503


namespace initial_reading_times_per_day_l1745_174551

-- Definitions based on the conditions

/-- Number of pages Jessy plans to read initially in each session is 6. -/
def session_pages : ℕ := 6

/-- Jessy needs to read 140 pages in one week. -/
def total_pages : ℕ := 140

/-- Jessy reads an additional 2 pages per day to achieve her goal. -/
def additional_daily_pages : ℕ := 2

/-- Days in a week -/
def days_in_week : ℕ := 7

-- Proving Jessy's initial plan for reading times per day
theorem initial_reading_times_per_day (x : ℕ) (h : days_in_week * (session_pages * x + additional_daily_pages) = total_pages) : 
    x = 3 := by
  -- skipping the proof itself
  sorry

end initial_reading_times_per_day_l1745_174551


namespace trig_identity_l1745_174586

theorem trig_identity (f : ℝ → ℝ) (ϕ : ℝ) (h₁ : ∀ x, f x = 2 * Real.sin (2 * x + ϕ)) (h₂ : 0 < ϕ) (h₃ : ϕ < π) (h₄ : f 0 = 1) :
  f ϕ = 2 :=
sorry

end trig_identity_l1745_174586


namespace sqrt_2023_irrational_l1745_174523

theorem sqrt_2023_irrational : ¬ ∃ (r : ℚ), r^2 = 2023 := by
  sorry

end sqrt_2023_irrational_l1745_174523


namespace cats_left_in_store_l1745_174502

theorem cats_left_in_store 
  (initial_siamese : ℕ := 25)
  (initial_persian : ℕ := 18)
  (initial_house : ℕ := 12)
  (initial_maine_coon : ℕ := 10)
  (sold_siamese : ℕ := 6)
  (sold_persian : ℕ := 4)
  (sold_maine_coon : ℕ := 3)
  (sold_house : ℕ := 0)
  (remaining_siamese : ℕ := 19)
  (remaining_persian : ℕ := 14)
  (remaining_house : ℕ := 12)
  (remaining_maine_coon : ℕ := 7) : 
  initial_siamese - sold_siamese = remaining_siamese ∧
  initial_persian - sold_persian = remaining_persian ∧
  initial_house - sold_house = remaining_house ∧
  initial_maine_coon - sold_maine_coon = remaining_maine_coon :=
by sorry

end cats_left_in_store_l1745_174502


namespace cylinder_ratio_max_volume_l1745_174514

theorem cylinder_ratio_max_volume 
    (l w : ℝ) 
    (r : ℝ) 
    (h : ℝ)
    (H_perimeter : 2 * l + 2 * w = 12)
    (H_length_circumference : l = 2 * π * r)
    (H_width_height : w = h) :
    (∀ V : ℝ, V = π * r^2 * h) →
    (∀ r : ℝ, r = 2 / π) →
    ((2 * π * r) / h = 2) :=
sorry

end cylinder_ratio_max_volume_l1745_174514


namespace find_a_b_sum_l1745_174596

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x - 6

theorem find_a_b_sum (a b : ℝ)
  (h1 : f a = 1)
  (h2 : f b = -5) :
  a + b = 2 :=
  sorry

end find_a_b_sum_l1745_174596


namespace Kindergarten_Students_l1745_174572

theorem Kindergarten_Students (X : ℕ) (h1 : 40 * X + 40 * 10 + 40 * 11 = 1200) : X = 9 :=
by
  sorry

end Kindergarten_Students_l1745_174572


namespace sea_lions_at_zoo_l1745_174565

def ratio_sea_lions_to_penguins (S P : ℕ) : Prop := P = 11 * S / 4
def ratio_sea_lions_to_flamingos (S F : ℕ) : Prop := F = 7 * S / 4
def penguins_more_sea_lions (S P : ℕ) : Prop := P = S + 84
def flamingos_more_penguins (P F : ℕ) : Prop := F = P + 42

theorem sea_lions_at_zoo (S P F : ℕ)
  (h1 : ratio_sea_lions_to_penguins S P)
  (h2 : ratio_sea_lions_to_flamingos S F)
  (h3 : penguins_more_sea_lions S P)
  (h4 : flamingos_more_penguins P F) :
  S = 42 :=
sorry

end sea_lions_at_zoo_l1745_174565


namespace taxi_ride_cost_l1745_174561

theorem taxi_ride_cost (initial_cost : ℝ) (cost_first_3_miles : ℝ) (rate_first_3_miles : ℝ) (rate_after_3_miles : ℝ) (total_miles : ℝ) (remaining_miles : ℝ) :
  initial_cost = 2.00 ∧ rate_first_3_miles = 0.30 ∧ rate_after_3_miles = 0.40 ∧ total_miles = 8 ∧ total_miles - 3 = remaining_miles →
  initial_cost + 3 * rate_first_3_miles + remaining_miles * rate_after_3_miles = 4.90 :=
sorry

end taxi_ride_cost_l1745_174561


namespace evaluate_polynomial_103_l1745_174517

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l1745_174517


namespace rationalize_denominator_correct_l1745_174542

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct_l1745_174542


namespace range_of_x_l1745_174543

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end range_of_x_l1745_174543


namespace find_m_l1745_174576

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 + 3 * x2 = 5) : m = 7 / 4 :=
  sorry

end find_m_l1745_174576


namespace eq_root_count_l1745_174528

theorem eq_root_count (p : ℝ) : 
  (∀ x : ℝ, (2 * x^2 - 3 * p * x + 2 * p = 0 → (9 * p^2 - 16 * p = 0))) →
  (∃! p1 p2 : ℝ, (9 * p1^2 - 16 * p1 = 0) ∧ (9 * p2^2 - 16 * p2 = 0) ∧ p1 ≠ p2) :=
sorry

end eq_root_count_l1745_174528


namespace correct_coefficient_l1745_174564

-- Definitions based on given conditions
def isMonomial (expr : String) : Prop := true

def coefficient (expr : String) : ℚ :=
  if expr = "-a/3" then -1/3 else 0

-- Statement to prove
theorem correct_coefficient : coefficient "-a/3" = -1/3 :=
by
  sorry

end correct_coefficient_l1745_174564


namespace curve_crosses_itself_l1745_174518

-- Definitions of the parametric equations
def x (t k : ℝ) : ℝ := t^2 + k
def y (t k : ℝ) : ℝ := t^3 - k * t + 5

-- The main theorem statement
theorem curve_crosses_itself (k : ℝ) (ha : ℝ) (hb : ℝ) :
  ha ≠ hb →
  x ha k = x hb k →
  y ha k = y hb k →
  k = 9 ∧ x ha k = 18 ∧ y ha k = 5 :=
by
  sorry

end curve_crosses_itself_l1745_174518


namespace problem_statement_l1745_174513

variable {x y : ℤ}

def is_multiple_of_5 (n : ℤ) : Prop := ∃ m : ℤ, n = 5 * m
def is_multiple_of_10 (n : ℤ) : Prop := ∃ m : ℤ, n = 10 * m

theorem problem_statement (hx : is_multiple_of_5 x) (hy : is_multiple_of_10 y) :
  (is_multiple_of_5 (x + y)) ∧ (x + y ≥ 15) :=
sorry

end problem_statement_l1745_174513


namespace power_expression_result_l1745_174525

theorem power_expression_result : (-2)^2004 + (-2)^2005 = -2^2004 :=
by
  sorry

end power_expression_result_l1745_174525


namespace four_star_three_l1745_174578

def star (a b : ℕ) : ℕ := a^2 - a * b + b^2 + 2 * a * b

theorem four_star_three : star 4 3 = 37 :=
by
  -- here we would normally provide the proof steps
  sorry

end four_star_three_l1745_174578


namespace geometric_sequence_formula_and_sum_l1745_174530

theorem geometric_sequence_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h_arith : a 1 = 2 ∧ 2 * (a 3 + 1) = a 1 + a 4)
  (h_b : ∀ n, b n = Nat.log2 (a n)) :
  (∀ n, a n = 2 ^ n) ∧ (S n = (n * (n + 1)) / 2) := 
by 
  sorry

end geometric_sequence_formula_and_sum_l1745_174530


namespace ineq_10_3_minus_9_5_l1745_174574

variable {a b c : ℝ}

/-- Given \(a, b, c\) are positive real numbers and \(a + b + c = 1\), prove \(10(a^3 + b^3 + c^3) - 9(a^5 + b^5 + c^5) \geq 1\). -/
theorem ineq_10_3_minus_9_5 (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := 
sorry

end ineq_10_3_minus_9_5_l1745_174574


namespace percent_of_x_l1745_174501

theorem percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : z = 2 * x) :
  y = 0.75 * x :=
sorry

end percent_of_x_l1745_174501


namespace at_least_one_false_l1745_174554

theorem at_least_one_false (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
  by
  sorry

end at_least_one_false_l1745_174554


namespace Tamara_is_95_inches_l1745_174560

/- Defining the basic entities: Kim's height (K), Tamara's height, Gavin's height -/
def Kim_height (K : ℝ) := K
def Tamara_height (K : ℝ) := 3 * K - 4
def Gavin_height (K : ℝ) := 2 * K + 6

/- Combined height equation -/
def combined_height (K : ℝ) := (Tamara_height K) + (Kim_height K) + (Gavin_height K) = 200

/- Given that Kim's height satisfies the combined height condition,
   proving that Tamara's height is 95 inches -/
theorem Tamara_is_95_inches (K : ℝ) (h : combined_height K) : Tamara_height K = 95 :=
by
  sorry

end Tamara_is_95_inches_l1745_174560


namespace percentage_cross_pollinated_l1745_174545

-- Definitions and known conditions:
variables (F C T : ℕ)
variables (h1 : F + C = 221)
variables (h2 : F = 3 * T / 4)
variables (h3 : T = F + 39 + C)

-- Theorem statement for the percentage of cross-pollinated trees
theorem percentage_cross_pollinated : ((C : ℚ) / T) * 100 = 10 :=
by sorry

end percentage_cross_pollinated_l1745_174545


namespace derivative_of_gx_eq_3x2_l1745_174552

theorem derivative_of_gx_eq_3x2 (f : ℝ → ℝ) : (∀ x : ℝ, f x = (x + 1) * (x^2 - x + 1)) → (∀ x : ℝ, deriv f x = 3 * x^2) :=
by
  intro h
  sorry

end derivative_of_gx_eq_3x2_l1745_174552


namespace swimming_speed_l1745_174537

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 14) 
  (h3 : time = 3.5) 
  (h4 : distance = (v - water_speed) * time) : 
  v = 6 := 
by
  sorry

end swimming_speed_l1745_174537


namespace value_of_k_l1745_174559

open Nat

def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)
def comb (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem value_of_k : ∃ k : ℕ, perm 32 6 = k * comb 32 6 ∧ k = 720 := by
  use 720
  unfold perm comb
  sorry

end value_of_k_l1745_174559


namespace sum_max_min_a_l1745_174598

theorem sum_max_min_a (a : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x - 20 * a^2 < 0)
  (h2 : ∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 → x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) :
    -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → (1 + -1) = 0 :=
by
  sorry

end sum_max_min_a_l1745_174598


namespace frac_pattern_2_11_frac_pattern_general_l1745_174595

theorem frac_pattern_2_11 :
  (2 / 11) = (1 / 6) + (1 / 66) :=
sorry

theorem frac_pattern_general (n : ℕ) (hn : n ≥ 3) :
  (2 / (2 * n - 1)) = (1 / n) + (1 / (n * (2 * n - 1))) :=
sorry

end frac_pattern_2_11_frac_pattern_general_l1745_174595


namespace factorial_division_l1745_174547

theorem factorial_division (N : Nat) (h : N ≥ 2) : 
  (Nat.factorial (2 * N)) / ((Nat.factorial (N + 2)) * (Nat.factorial (N - 2))) = 
  (List.prod (List.range' (N + 3) (2 * N - (N + 2) + 1))) / (Nat.factorial (N - 1)) :=
sorry

end factorial_division_l1745_174547


namespace original_price_of_stamp_l1745_174590

theorem original_price_of_stamp (original_price : ℕ) (h : original_price * (1 / 5 : ℚ) = 6) : original_price = 30 :=
by
  sorry

end original_price_of_stamp_l1745_174590


namespace total_crayons_l1745_174505

-- We're given the conditions
def crayons_per_child : ℕ := 6
def number_of_children : ℕ := 12

-- We need to prove the total number of crayons.
theorem total_crayons (c : ℕ := crayons_per_child) (n : ℕ := number_of_children) : (c * n) = 72 := by
  sorry

end total_crayons_l1745_174505


namespace jo_page_an_hour_ago_l1745_174557

variables (total_pages current_page hours_left : ℕ)
variables (steady_reading_rate : ℕ)
variables (page_an_hour_ago : ℕ)

-- Conditions
def conditions := 
  steady_reading_rate * hours_left = total_pages - current_page ∧
  total_pages = 210 ∧
  current_page = 90 ∧
  hours_left = 4 ∧
  page_an_hour_ago = current_page - steady_reading_rate

-- Theorem to prove that Jo was on page 60 an hour ago
theorem jo_page_an_hour_ago (h : conditions total_pages current_page hours_left steady_reading_rate page_an_hour_ago) : 
  page_an_hour_ago = 60 :=
sorry

end jo_page_an_hour_ago_l1745_174557


namespace product_eq_one_of_abs_log_eq_l1745_174592

theorem product_eq_one_of_abs_log_eq (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := 
sorry

end product_eq_one_of_abs_log_eq_l1745_174592


namespace find_number_l1745_174536

-- Define given numbers
def a : ℕ := 555
def b : ℕ := 445

-- Define given conditions
def sum : ℕ := a + b
def difference : ℕ := a - b
def quotient : ℕ := 2 * difference
def remainder : ℕ := 30

-- Define the number we're looking for
def number := sum * quotient + remainder

-- The theorem to prove
theorem find_number : number = 220030 := by
  -- Use the let expressions to simplify the calculation for clarity
  let sum := a + b
  let difference := a - b
  let quotient := 2 * difference
  let number := sum * quotient + remainder
  show number = 220030
  -- Placeholder for proof
  sorry

end find_number_l1745_174536


namespace mean_square_sum_l1745_174531

theorem mean_square_sum (x y z : ℝ) 
  (h1 : x + y + z = 27)
  (h2 : x * y * z = 216)
  (h3 : x * y + y * z + z * x = 162) : 
  x^2 + y^2 + z^2 = 405 :=
by
  sorry

end mean_square_sum_l1745_174531


namespace find_a_l1745_174599

theorem find_a 
  (x : ℤ) 
  (a : ℤ) 
  (h1 : x = 2) 
  (h2 : y = a) 
  (h3 : 2 * x - 3 * y = 5) : a = -1 / 3 := 
by 
  sorry

end find_a_l1745_174599


namespace min_cos_for_sqrt_l1745_174527

theorem min_cos_for_sqrt (x : ℝ) (h : 2 * Real.cos x - 1 ≥ 0) : Real.cos x ≥ 1 / 2 := 
by
  sorry

end min_cos_for_sqrt_l1745_174527


namespace find_some_number_l1745_174573

def simplify_expr (x : ℚ) : Prop :=
  1 / 2 + ((2 / 3 * (3 / 8)) + x) - (8 / 16) = 4.25

theorem find_some_number :
  ∃ x : ℚ, simplify_expr x ∧ x = 4 :=
by
  sorry

end find_some_number_l1745_174573


namespace intersect_at_one_point_l1745_174512

theorem intersect_at_one_point (a : ℝ) : 
  (a * (4 * 4) + 4 * 4 * 6 = 0) -> a = 2 / (3: ℝ) :=
by sorry

end intersect_at_one_point_l1745_174512


namespace framed_painting_ratio_l1745_174571

-- Define the conditions and the problem
theorem framed_painting_ratio:
  ∀ (x : ℝ),
    (30 + 2 * x) * (20 + 4 * x) = 1500 →
    (20 + 4 * x) / (30 + 2 * x) = 4 / 5 := 
by sorry

end framed_painting_ratio_l1745_174571


namespace sequence_general_term_l1745_174508

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) = 2^n * a n) : 
  ∀ n, a n = 2^((n-1)*n / 2) := sorry

end sequence_general_term_l1745_174508


namespace third_number_in_first_set_is_42_l1745_174579

theorem third_number_in_first_set_is_42 (x y : ℕ) :
  (28 + x + y + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 42 :=
by { sorry }

end third_number_in_first_set_is_42_l1745_174579


namespace initial_people_in_castle_l1745_174540

theorem initial_people_in_castle (P : ℕ) (provisions : ℕ → ℕ → ℕ) :
  (provisions P 90) - (provisions P 30) = provisions (P - 100) 90 ↔ P = 300 :=
by
  sorry

end initial_people_in_castle_l1745_174540


namespace harry_terry_difference_l1745_174544

theorem harry_terry_difference :
  let H := 12 - (3 * 4)
  let T := 12 - (3 * 4) -- Correcting Terry's mistake
  H - T = 0 := by
  sorry

end harry_terry_difference_l1745_174544


namespace weight_of_b_is_37_l1745_174529

variables {a b c : ℝ}

-- Conditions
def average_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def average_bc (b c : ℝ) : Prop := (b + c) / 2 = 46

-- Statement to prove
theorem weight_of_b_is_37 (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 37 :=
by {
  sorry
}

end weight_of_b_is_37_l1745_174529


namespace polynomial_remainder_l1745_174532

theorem polynomial_remainder (p : Polynomial ℝ) :
  (p.eval 2 = 3) → (p.eval 3 = 9) → ∃ q : Polynomial ℝ, p = (Polynomial.X - 2) * (Polynomial.X - 3) * q + (6 * Polynomial.X - 9) :=
by
  sorry

end polynomial_remainder_l1745_174532


namespace soft_lenses_more_than_hard_l1745_174581

-- Define the problem conditions as Lean definitions
def total_sales (S H : ℕ) : Prop := 150 * S + 85 * H = 1455
def total_pairs (S H : ℕ) : Prop := S + H = 11

-- The theorem we need to prove
theorem soft_lenses_more_than_hard (S H : ℕ) (h1 : total_sales S H) (h2 : total_pairs S H) : S - H = 5 :=
by
  sorry

end soft_lenses_more_than_hard_l1745_174581


namespace binomial_last_three_terms_sum_l1745_174582

theorem binomial_last_three_terms_sum (n : ℕ) :
  (1 + n + (n * (n - 1)) / 2 = 79) → n = 12 :=
by
  sorry

end binomial_last_three_terms_sum_l1745_174582


namespace smallest_7_heavy_three_digit_number_l1745_174550

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_7_heavy_three_digit_number :
  ∃ n : ℕ, is_three_digit n ∧ is_7_heavy n ∧ (∀ m : ℕ, is_three_digit m ∧ is_7_heavy m → n ≤ m) ∧
  n = 103 := 
by
  sorry

end smallest_7_heavy_three_digit_number_l1745_174550


namespace intersecting_lines_l1745_174575

theorem intersecting_lines (m : ℝ) :
  (∃ (x y : ℝ), y = 2 * x ∧ x + y = 3 ∧ m * x + 2 * y + 5 = 0) ↔ (m = -9) :=
by
  sorry

end intersecting_lines_l1745_174575


namespace largest_minus_smallest_l1745_174541

-- Define the given conditions
def A : ℕ := 10 * 2 + 9
def B : ℕ := A - 16
def C : ℕ := B * 3

-- Statement to prove
theorem largest_minus_smallest : C - B = 26 := by
  sorry

end largest_minus_smallest_l1745_174541


namespace sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l1745_174526

open Real

theorem sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) :
  sqrt (x * y * z) ≥ sqrt x + sqrt y + sqrt z :=
by
  sorry

end sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l1745_174526


namespace cows_with_no_spot_l1745_174570

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l1745_174570


namespace evaluate_expression_l1745_174553

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end evaluate_expression_l1745_174553


namespace inradius_semicircle_relation_l1745_174594

theorem inradius_semicircle_relation 
  (a b c : ℝ)
  (h_acute: a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (S : ℝ)
  (p : ℝ)
  (r : ℝ)
  (ra rb rc : ℝ)
  (h_def_semi_perim : p = (a + b + c) / 2)
  (h_area : S = p * r)
  (h_ra : ra = (2 * S) / (b + c))
  (h_rb : rb = (2 * S) / (a + c))
  (h_rc : rc = (2 * S) / (a + b)) :
  2 / r = 1 / ra + 1 / rb + 1 / rc :=
by
  sorry

end inradius_semicircle_relation_l1745_174594


namespace water_percentage_l1745_174521

theorem water_percentage (P : ℕ) : 
  let initial_volume := 300
  let final_volume := initial_volume + 100
  let desired_water_percentage := 70
  let water_added := 100
  let final_water_amount := desired_water_percentage * final_volume / 100
  let current_water_amount := P * initial_volume / 100

  current_water_amount + water_added = final_water_amount → 
  P = 60 :=
by sorry

end water_percentage_l1745_174521


namespace simplify_expression_l1745_174549

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l1745_174549


namespace M_is_even_l1745_174509

def sum_of_digits (n : ℕ) : ℕ := -- Define the digit sum function
  sorry

theorem M_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  M % 2 = 0 :=
sorry

end M_is_even_l1745_174509


namespace problem_statement_l1745_174589

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem problem_statement (x : ℝ) (h : x ≠ 0) : f x > 0 :=
by sorry

end problem_statement_l1745_174589


namespace leon_older_than_aivo_in_months_l1745_174533

theorem leon_older_than_aivo_in_months
    (jolyn therese aivo leon : ℕ)
    (h1 : jolyn = therese + 2)
    (h2 : therese = aivo + 5)
    (h3 : jolyn = leon + 5) :
    leon = aivo + 2 := 
sorry

end leon_older_than_aivo_in_months_l1745_174533


namespace molecular_weight_one_mole_of_AlOH3_l1745_174507

variable (MW_7_moles : ℕ) (MW : ℕ)

theorem molecular_weight_one_mole_of_AlOH3 (h : MW_7_moles = 546) : MW = 78 :=
by
  sorry

end molecular_weight_one_mole_of_AlOH3_l1745_174507


namespace coefficient_of_x3y0_l1745_174522

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def f (m n : ℕ) : ℕ :=
  binomial_coeff 6 m * binomial_coeff 4 n

theorem coefficient_of_x3y0 :
  f 3 0 = 20 :=
by
  sorry

end coefficient_of_x3y0_l1745_174522


namespace ratio_A_BC_1_to_4_l1745_174584

/-
We will define the conditions and prove the ratio.
-/

def A := 20
def total := 100

-- defining the conditions
variables (B C : ℝ)
def condition1 := A + B + C = total
def condition2 := B = 3 / 5 * (A + C)

-- the theorem to prove
theorem ratio_A_BC_1_to_4 (h1 : condition1 B C) (h2 : condition2 B C) : A / (B + C) = 1 / 4 :=
by
  sorry

end ratio_A_BC_1_to_4_l1745_174584


namespace program_output_for_six_l1745_174546

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- The theorem we want to prove
theorem program_output_for_six : factorial 6 = 720 := by
  sorry

end program_output_for_six_l1745_174546


namespace complement_union_intersection_l1745_174588

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem complement_union_intersection :
  (compl (A ∪ B) = {x | x ≤ 2 ∨ 9 ≤ x}) ∧
  (compl (A ∩ B) = {x | x < 3 ∨ 5 ≤ x}) :=
by
  sorry

end complement_union_intersection_l1745_174588


namespace problem_statement_l1745_174566

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ (ab < 1) :=
by
  sorry

end problem_statement_l1745_174566


namespace smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l1745_174504

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_f : ∃ k > 0, ∀ x, f (x + k) = f x := 
sorry

theorem max_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = Real.sqrt 2 :=
sorry

theorem min_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = -1 :=
sorry

end smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l1745_174504


namespace max_sum_of_inequalities_l1745_174519

theorem max_sum_of_inequalities (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) :
  x + y ≤ 31 / 11 :=
sorry

end max_sum_of_inequalities_l1745_174519


namespace cube_red_face_probability_l1745_174535

theorem cube_red_face_probability :
  let faces_total := 6
  let red_faces := 3
  let probability_red := red_faces / faces_total
  probability_red = 1 / 2 :=
by
  sorry

end cube_red_face_probability_l1745_174535


namespace simplify_and_evaluate_l1745_174539

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) : 
  (1 - 1 / (a + 1)) * ((a^2 + 2 * a + 1) / a) = Real.sqrt 2 := 
by {
  sorry
}

end simplify_and_evaluate_l1745_174539


namespace campers_rowing_afternoon_l1745_174562

theorem campers_rowing_afternoon (morning_rowing morning_hiking total : ℕ) 
  (h1 : morning_rowing = 41) 
  (h2 : morning_hiking = 4) 
  (h3 : total = 71) : 
  total - (morning_rowing + morning_hiking) = 26 :=
by
  sorry

end campers_rowing_afternoon_l1745_174562


namespace simplify_trig_identity_l1745_174511

theorem simplify_trig_identity (α : ℝ) :
  (Real.cos (Real.pi / 3 + α) + Real.sin (Real.pi / 6 + α)) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l1745_174511


namespace area_of_triangle_is_2_l1745_174585

-- Define the conditions of the problem
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles in radians

-- Conditions for the triangle ABC
variable (sin_A : ℝ) (sin_C : ℝ)
variable (c2sinA_eq_5sinC : c^2 * sin_A = 5 * sin_C)
variable (a_plus_c_squared_eq_16_plus_b_squared : (a + c)^2 = 16 + b^2)
variable (ac_eq_5 : a * c = 5)
variable (cos_B : ℝ)
variable (sin_B : ℝ)

-- Sine and Cosine law results
variable (cos_B_def : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))
variable (sin_B_def : sin_B = Real.sqrt (1 - cos_B^2))

-- Area of the triangle
noncomputable def area_triangle_ABC := (1/2) * a * c * sin_B

-- Theorem to prove the area
theorem area_of_triangle_is_2 :
  area_triangle_ABC a c sin_B = 2 :=
by
  rw [area_triangle_ABC]
  sorry

end area_of_triangle_is_2_l1745_174585


namespace g_recursion_relation_l1745_174516

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2)^n

theorem g_recursion_relation (n : ℕ) : g (n + 1) - 2 * g n + g (n - 1) = 0 :=
  sorry

end g_recursion_relation_l1745_174516


namespace dan_picked_more_apples_l1745_174548

-- Define the number of apples picked by Benny and Dan
def apples_picked_by_benny := 2
def apples_picked_by_dan := 9

-- Lean statement to prove the given condition
theorem dan_picked_more_apples :
  apples_picked_by_dan - apples_picked_by_benny = 7 := 
sorry

end dan_picked_more_apples_l1745_174548


namespace factor_expression_l1745_174567

theorem factor_expression (x y : ℝ) :
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) :=
by
  sorry

end factor_expression_l1745_174567


namespace km_to_leaps_l1745_174569

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps_l1745_174569


namespace intersection_A_B_l1745_174515

-- Define set A
def A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

-- Define set B
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.sqrt x }

-- Prove that the intersection of sets A and B is [0, +∞)
theorem intersection_A_B : A ∩ B = { x | 0 ≤ x } :=
by
  sorry

end intersection_A_B_l1745_174515


namespace trapezoid_area_l1745_174558

theorem trapezoid_area
  (AD BC AC BD : ℝ)
  (h1 : AD = 24)
  (h2 : BC = 8)
  (h3 : AC = 13)
  (h4 : BD = 5 * Real.sqrt 17) : 
  ∃ (area : ℝ), area = 80 :=
by
  let area := (1 / 2) * (AD + BC) * 5
  existsi area
  sorry

end trapezoid_area_l1745_174558


namespace novel_cost_l1745_174577

-- Given conditions
variable (N : ℕ) -- cost of the novel
variable (lunch_cost : ℕ) -- cost of lunch

-- Conditions
axiom gift_amount : N + lunch_cost + 29 = 50
axiom lunch_cost_eq : lunch_cost = 2 * N

-- Question and answer tuple as a theorem
theorem novel_cost : N = 7 := 
by
  sorry -- Proof estaps are to be filled in.

end novel_cost_l1745_174577


namespace field_area_is_243_l1745_174506

noncomputable def field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : ℝ :=
  w * l

theorem field_area_is_243 (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : field_area w l h1 h2 = 243 :=
  sorry

end field_area_is_243_l1745_174506


namespace fraction_students_above_eight_l1745_174556

theorem fraction_students_above_eight (total_students S₈ : ℕ) (below_eight_percent : ℝ)
    (num_below_eight : total_students * below_eight_percent = 10) 
    (total_equals : total_students = 50) 
    (students_eight : S₈ = 24) :
    (total_students - (total_students * below_eight_percent + S₈)) / S₈ = 2 / 3 := 
by 
  -- Solution steps can go here 
  sorry

end fraction_students_above_eight_l1745_174556


namespace amount_of_juice_p_in_a_l1745_174568

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a_l1745_174568


namespace divisible_by_11_l1745_174555

theorem divisible_by_11 (n : ℤ) : (11 ∣ (n^2001 - n^4)) ↔ (n % 11 = 0 ∨ n % 11 = 1) :=
by
  sorry

end divisible_by_11_l1745_174555


namespace probability_even_sum_is_correct_l1745_174510

noncomputable def probability_even_sum : ℚ :=
  let p_even_first := (2 : ℚ) / 5
  let p_odd_first := (3 : ℚ) / 5
  let p_even_second := (1 : ℚ) / 4
  let p_odd_second := (3 : ℚ) / 4

  let p_both_even := p_even_first * p_even_second
  let p_both_odd := p_odd_first * p_odd_second

  p_both_even + p_both_odd

theorem probability_even_sum_is_correct : probability_even_sum = 11 / 20 := by
  sorry

end probability_even_sum_is_correct_l1745_174510


namespace path_inequality_l1745_174538

theorem path_inequality
  (f : ℕ → ℕ → ℝ) :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
sorry

end path_inequality_l1745_174538


namespace quadratics_root_k_value_l1745_174563

theorem quadratics_root_k_value :
  (∀ k : ℝ, (∀ x : ℝ, x^2 + k * x + 6 = 0 → (x = 2 ∨ ∃ x1 : ℝ, x1 * 2 = 6 ∧ x1 + 2 = k)) → 
  (x = 2 → ∃ x1 : ℝ, x1 = 3 ∧ k = -5)) := 
sorry

end quadratics_root_k_value_l1745_174563


namespace line_ellipse_intersect_l1745_174597

theorem line_ellipse_intersect (m k : ℝ) (h₀ : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) : m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersect_l1745_174597


namespace smallest_value_of_N_l1745_174534

theorem smallest_value_of_N (l m n : ℕ) (N : ℕ) (h1 : (l-1) * (m-1) * (n-1) = 270) (h2 : N = l * m * n): 
  N = 420 :=
sorry

end smallest_value_of_N_l1745_174534


namespace solution_set_abs_inequality_l1745_174591

theorem solution_set_abs_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_abs_inequality_l1745_174591


namespace nasadkas_in_barrel_l1745_174593

def capacity (B N V : ℚ) :=
  (B + 20 * V = 3 * B) ∧ (19 * B + N + 15.5 * V = 20 * B + 8 * V)

theorem nasadkas_in_barrel (B N V : ℚ) (h : capacity B N V) : B / N = 4 :=
by
  sorry

end nasadkas_in_barrel_l1745_174593


namespace michael_will_meet_two_times_l1745_174500

noncomputable def michael_meetings : ℕ :=
  let michael_speed := 6 -- feet per second
  let pail_distance := 300 -- feet
  let truck_speed := 12 -- feet per second
  let truck_stop_time := 20 -- seconds
  let initial_distance := pail_distance -- feet
  let michael_position (t: ℕ) := michael_speed * t
  let truck_position (cycle: ℕ) := pail_distance * cycle
  let truck_cycle_time := pail_distance / truck_speed + truck_stop_time -- seconds per cycle
  let truck_position_at_time (t: ℕ) := 
    let cycle := t / truck_cycle_time
    let remaining_time := t % truck_cycle_time
    if remaining_time < (pail_distance / truck_speed) then 
      truck_position cycle + truck_speed * remaining_time
    else 
      truck_position cycle + pail_distance
  let distance_between := 
    λ (t: ℕ) => truck_position_at_time t - michael_position t
  let meet_time := 
    λ (t: ℕ) => if distance_between t = 0 then 1 else 0
  let total_meetings := 
    (List.range 300).map meet_time -- estimating within 300 seconds
    |> List.sum
  total_meetings

theorem michael_will_meet_two_times : michael_meetings = 2 :=
  sorry

end michael_will_meet_two_times_l1745_174500


namespace car_speed_l1745_174587

variable (Distance : ℕ) (Time : ℕ)
variable (h1 : Distance = 495)
variable (h2 : Time = 5)

theorem car_speed (Distance Time : ℕ) (h1 : Distance = 495) (h2 : Time = 5) : 
  Distance / Time = 99 :=
by
  sorry

end car_speed_l1745_174587


namespace fifth_term_of_sequence_l1745_174520

theorem fifth_term_of_sequence :
  let a_n (n : ℕ) := (-1:ℤ)^(n+1) * (n^2 + 1)
  ∃ x : ℤ, a_n 5 * x^5 = 26 * x^5 :=
by
  sorry

end fifth_term_of_sequence_l1745_174520


namespace ptolemys_inequality_l1745_174580

variable {A B C D : Type} [OrderedRing A]
variable (AB BC CD DA AC BD : A)

/-- Ptolemy's inequality for a quadrilateral -/
theorem ptolemys_inequality 
  (AB_ BC_ CD_ DA_ AC_ BD_ : A) :
  AC * BD ≤ AB * CD + BC * AD :=
  sorry

end ptolemys_inequality_l1745_174580


namespace josh_bottle_caps_l1745_174524

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l1745_174524
