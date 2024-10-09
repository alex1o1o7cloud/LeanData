import Mathlib

namespace systematic_sampling_result_l2349_234967

theorem systematic_sampling_result :
  ∀ (total_students sample_size selected1_16 selected33_48 : ℕ),
  total_students = 800 →
  sample_size = 50 →
  selected1_16 = 11 →
  selected33_48 = selected1_16 + 32 →
  selected33_48 = 43 := by
  intros
  sorry

end systematic_sampling_result_l2349_234967


namespace cats_left_correct_l2349_234917

-- Define initial conditions
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def sold_cats : ℕ := 10

-- Define the total number of cats initially
def total_cats_initial : ℕ := siamese_cats + house_cats

-- Define the number of cats left after the sale
def cats_left : ℕ := total_cats_initial - sold_cats

-- Prove the number of cats left is 8
theorem cats_left_correct : cats_left = 8 :=
by 
  sorry

end cats_left_correct_l2349_234917


namespace ceil_neg_sqrt_frac_l2349_234953

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l2349_234953


namespace cos_pi_minus_double_alpha_l2349_234976

theorem cos_pi_minus_double_alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_double_alpha_l2349_234976


namespace find_y_l2349_234993

noncomputable def x : Real := 2.6666666666666665

theorem find_y (y : Real) (h : (x * y) / 3 = x^2) : y = 8 :=
sorry

end find_y_l2349_234993


namespace second_sheet_width_l2349_234972

theorem second_sheet_width :
  ∃ w : ℝ, (286 = 22 * w + 100) ∧ w = 8.5 :=
by
  -- Proof goes here
  sorry

end second_sheet_width_l2349_234972


namespace sector_area_l2349_234939

theorem sector_area (r θ : ℝ) (h₁ : θ = 2) (h₂ : r * θ = 4) : (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end sector_area_l2349_234939


namespace total_value_of_coins_l2349_234916

variable (numCoins : ℕ) (coinsValue : ℕ) 

theorem total_value_of_coins : 
  numCoins = 15 → 
  (∀ n: ℕ, n = 5 → coinsValue = 12) → 
  ∃ totalValue : ℕ, totalValue = 36 :=
  by
    sorry

end total_value_of_coins_l2349_234916


namespace range_of_k_l2349_234980

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*y^2 = 2 ∧ 
  (∀ e : ℝ, (x^2 / 2 + y^2 / (2 / e) = 1 → (2 / e) > 2))) → 
  0 < k ∧ k < 1 :=
by 
sorry

end range_of_k_l2349_234980


namespace jellybean_addition_l2349_234969

-- Definitions related to the problem
def initial_jellybeans : ℕ := 37
def removed_jellybeans_initial : ℕ := 15
def added_jellybeans (x : ℕ) : ℕ := x
def removed_jellybeans_again : ℕ := 4
def final_jellybeans : ℕ := 23

-- Prove that the number of jellybeans added back (x) is 5
theorem jellybean_addition (x : ℕ) 
  (h1 : initial_jellybeans - removed_jellybeans_initial + added_jellybeans x - removed_jellybeans_again = final_jellybeans) : 
  x = 5 :=
sorry

end jellybean_addition_l2349_234969


namespace alcohol_percentage_new_mixture_l2349_234904

/--
Given:
1. The initial mixture has 15 liters.
2. The mixture contains 20% alcohol.
3. 5 liters of water is added to the mixture.

Prove:
The percentage of alcohol in the new mixture is 15%.
-/
theorem alcohol_percentage_new_mixture :
  let initial_mixture_volume := 15 -- in liters
  let initial_alcohol_percentage := 20 / 100
  let initial_alcohol_volume := initial_alcohol_percentage * initial_mixture_volume
  let added_water_volume := 5 -- in liters
  let new_total_volume := initial_mixture_volume + added_water_volume
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 15 := 
by
  -- Proof steps go here
  sorry

end alcohol_percentage_new_mixture_l2349_234904


namespace solution_set_inequality_l2349_234959

theorem solution_set_inequality
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → ax^2 + bx + c > 0) :
  ∃ s : Set ℝ, s = {x | (1/2) < x ∧ x < 1} ∧ ∀ x : ℝ, x ∈ s → cx^2 + bx + a > 0 := by
sorry

end solution_set_inequality_l2349_234959


namespace angle_AOC_is_minus_150_l2349_234966

-- Define the conditions.
def rotate_counterclockwise (angle1 : Int) (angle2 : Int) : Int :=
  angle1 + angle2

-- The initial angle starts at 0°, rotates 120° counterclockwise, and then 270° clockwise
def angle_OA := 0
def angle_OB := rotate_counterclockwise angle_OA 120
def angle_OC := rotate_counterclockwise angle_OB (-270)

-- The theorem stating the resulting angle between OA and OC.
theorem angle_AOC_is_minus_150 : angle_OC = -150 := by
  sorry

end angle_AOC_is_minus_150_l2349_234966


namespace largest_divisor_is_one_l2349_234937

theorem largest_divisor_is_one (p q : ℤ) (hpq : p > q) (hp : p % 2 = 1) (hq : q % 2 = 0) :
  ∀ d : ℤ, (∀ p q : ℤ, p > q → p % 2 = 1 → q % 2 = 0 → d ∣ (p^2 - q^2)) → d = 1 :=
sorry

end largest_divisor_is_one_l2349_234937


namespace find_number_of_students_l2349_234977

open Nat

theorem find_number_of_students :
  ∃ n : ℕ, 35 < n ∧ n < 70 ∧ n % 6 = 3 ∧ n % 8 = 1 ∧ n = 57 :=
by
  use 57
  sorry

end find_number_of_students_l2349_234977


namespace system_solution_y_greater_than_five_l2349_234935

theorem system_solution_y_greater_than_five (m x y : ℝ) :
  (y = (m + 1) * x + 2) → 
  (y = (3 * m - 2) * x + 5) → 
  y > 5 ↔ 
  m ≠ 3 / 2 := 
sorry

end system_solution_y_greater_than_five_l2349_234935


namespace family_boys_girls_l2349_234913

theorem family_boys_girls (B G : ℕ) 
  (h1 : B - 1 = G) 
  (h2 : B = 2 * (G - 1)) : 
  B = 4 ∧ G = 3 := 
by {
  sorry
}

end family_boys_girls_l2349_234913


namespace p_sufficient_but_not_necessary_for_q_l2349_234979

def condition_p (x : ℝ) : Prop := x^2 - 9 > 0
def condition_q (x : ℝ) : Prop := x^2 - (5 / 6) * x + (1 / 6) > 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x, condition_p x → condition_q x) ∧ ¬(∀ x, condition_q x → condition_p x) :=
sorry

end p_sufficient_but_not_necessary_for_q_l2349_234979


namespace find_abc_l2349_234990

theorem find_abc :
  ∃ a b c : ℝ, (∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - a) * (x - b) / (x - c) ≤ 0)) ∧ a < b ∧ a + 2 * b + 3 * c = 74 :=
by
  sorry

end find_abc_l2349_234990


namespace solve_equation_l2349_234956

theorem solve_equation (x : ℤ) : x * (x + 2) + 1 = 36 ↔ x = 5 :=
by sorry

end solve_equation_l2349_234956


namespace money_given_to_last_set_l2349_234986

theorem money_given_to_last_set (total first second third fourth last : ℝ) 
  (h_total : total = 4500) 
  (h_first : first = 725) 
  (h_second : second = 1100) 
  (h_third : third = 950) 
  (h_fourth : fourth = 815) 
  (h_sum: total = first + second + third + fourth + last) : 
  last = 910 :=
sorry

end money_given_to_last_set_l2349_234986


namespace total_roses_planted_l2349_234918

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l2349_234918


namespace total_pieces_gum_is_correct_l2349_234973

-- Define the number of packages and pieces per package
def packages : ℕ := 27
def pieces_per_package : ℕ := 18

-- Define the total number of pieces of gum Robin has
def total_pieces_gum : ℕ :=
  packages * pieces_per_package

-- State the theorem and proof obligation
theorem total_pieces_gum_is_correct : total_pieces_gum = 486 := by
  -- Proof omitted
  sorry

end total_pieces_gum_is_correct_l2349_234973


namespace intersection_x_val_l2349_234957

theorem intersection_x_val (x y : ℝ) (h1 : y = 3 * x - 24) (h2 : 5 * x + 2 * y = 102) : x = 150 / 11 :=
by
  sorry

end intersection_x_val_l2349_234957


namespace total_people_on_boats_l2349_234930

theorem total_people_on_boats (boats : ℕ) (people_per_boat : ℕ) (h_boats : boats = 5) (h_people : people_per_boat = 3) : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l2349_234930


namespace original_board_length_before_final_cut_l2349_234911

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l2349_234911


namespace consumer_installment_credit_l2349_234975

theorem consumer_installment_credit (A C : ℝ) 
  (h1 : A = 0.36 * C) 
  (h2 : 57 = 1 / 3 * A) : 
  C = 475 := 
by 
  sorry

end consumer_installment_credit_l2349_234975


namespace monomial_sum_mn_l2349_234950

theorem monomial_sum_mn (m n : ℤ) 
  (h1 : m + 6 = 1) 
  (h2 : 2 * n + 1 = 7) : 
  m * n = -15 := by
  sorry

end monomial_sum_mn_l2349_234950


namespace correct_value_two_decimal_places_l2349_234907

theorem correct_value_two_decimal_places (x : ℝ) 
  (h1 : 8 * x + 8 = 56) : 
  (x / 8) + 7 = 7.75 :=
sorry

end correct_value_two_decimal_places_l2349_234907


namespace number_of_strings_is_multiple_of_3_l2349_234978

theorem number_of_strings_is_multiple_of_3 (N : ℕ) :
  (∀ (avg_total avg_one_third avg_two_third : ℚ), 
    avg_total = 80 ∧ avg_one_third = 70 ∧ avg_two_third = 85 →
    (∃ k : ℕ, N = 3 * k)) :=
by
  intros avg_total avg_one_third avg_two_third h
  sorry

end number_of_strings_is_multiple_of_3_l2349_234978


namespace tank_length_l2349_234910

theorem tank_length (W D : ℝ) (cost_per_sq_m total_cost : ℝ) (L : ℝ):
  W = 12 →
  D = 6 →
  cost_per_sq_m = 0.70 →
  total_cost = 520.8 →
  total_cost = cost_per_sq_m * ((2 * (W * D)) + (2 * (L * D)) + (L * W)) →
  L = 25 :=
by
  intros hW hD hCostPerSqM hTotalCost hEquation
  sorry

end tank_length_l2349_234910


namespace air_conditioner_sales_l2349_234909

-- Definitions based on conditions
def ratio_air_conditioners_refrigerators : ℕ := 5
def ratio_refrigerators_air_conditioners : ℕ := 3
def difference_in_sales : ℕ := 54

-- The property to be proven: 
def number_of_air_conditioners : ℕ := 135

theorem air_conditioner_sales
  (r_ac : ℕ := ratio_air_conditioners_refrigerators) 
  (r_ref : ℕ := ratio_refrigerators_air_conditioners) 
  (diff : ℕ := difference_in_sales) 
  : number_of_air_conditioners = 135 := sorry

end air_conditioner_sales_l2349_234909


namespace b_2_pow_100_value_l2349_234922

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n > 0, b (2 * n) = 2 * n * b n

theorem b_2_pow_100_value
  (b : ℕ → ℕ)
  (h_seq : seq b) :
  b (2^100) = 2^5050 * 3 :=
by
  sorry

end b_2_pow_100_value_l2349_234922


namespace average_price_of_pen_l2349_234981

theorem average_price_of_pen (c_total : ℝ) (n_pens n_pencils : ℕ) (p_pencil : ℝ)
  (h1 : c_total = 450) (h2 : n_pens = 30) (h3 : n_pencils = 75) (h4 : p_pencil = 2) :
  (c_total - (n_pencils * p_pencil)) / n_pens = 10 :=
by
  sorry

end average_price_of_pen_l2349_234981


namespace optimal_order_for_ostap_l2349_234997

variable (p1 p2 p3 : ℝ) (hp1 : 0 < p3) (hp2 : 0 < p1) (hp3 : 0 < p2) (h3 : p3 < p1) (h1 : p1 < p2)

theorem optimal_order_for_ostap :
  (∀ order : List ℝ, ∃ p4, order = [p1, p4, p3] ∨ order = [p3, p4, p1] ∨ order = [p2, p2, p2]) →
  (p4 = p2) :=
by
  sorry

end optimal_order_for_ostap_l2349_234997


namespace aaronTotalOwed_l2349_234949

def monthlyPayment : ℝ := 100
def numberOfMonths : ℕ := 12
def interestRate : ℝ := 0.1

def totalCostWithoutInterest : ℝ := monthlyPayment * (numberOfMonths : ℝ)
def interestAmount : ℝ := totalCostWithoutInterest * interestRate
def totalAmountOwed : ℝ := totalCostWithoutInterest + interestAmount

theorem aaronTotalOwed : totalAmountOwed = 1320 := by
  sorry

end aaronTotalOwed_l2349_234949


namespace gcd_g102_g103_l2349_234940

def g (x : ℕ) : ℕ := x^2 - x + 2007

theorem gcd_g102_g103 : 
  Nat.gcd (g 102) (g 103) = 3 :=
by
  sorry

end gcd_g102_g103_l2349_234940


namespace solve_for_x_l2349_234996

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end solve_for_x_l2349_234996


namespace inequality_proof_l2349_234983

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l2349_234983


namespace cookies_left_at_end_of_week_l2349_234970

def trays_baked_each_day : List Nat := [2, 3, 4, 5, 3, 4, 4]
def cookies_per_tray : Nat := 12
def cookies_eaten_by_frank : Nat := 2 * 7
def cookies_eaten_by_ted : Nat := 3 + 5
def cookies_eaten_by_jan : Nat := 5
def cookies_eaten_by_tom : Nat := 8
def cookies_eaten_by_neighbours_kids : Nat := 20

def total_cookies_baked : Nat :=
  (trays_baked_each_day.map (λ trays => trays * cookies_per_tray)).sum

def total_cookies_eaten : Nat :=
  cookies_eaten_by_frank + cookies_eaten_by_ted + cookies_eaten_by_jan +
  cookies_eaten_by_tom + cookies_eaten_by_neighbours_kids

def cookies_left : Nat := total_cookies_baked - total_cookies_eaten

theorem cookies_left_at_end_of_week : cookies_left = 245 :=
by
  sorry

end cookies_left_at_end_of_week_l2349_234970


namespace tom_total_dimes_l2349_234958

-- Define the original and additional dimes Tom received.
def original_dimes : ℕ := 15
def additional_dimes : ℕ := 33

-- Define the total number of dimes Tom has now.
def total_dimes : ℕ := original_dimes + additional_dimes

-- Statement to prove that the total number of dimes Tom has is 48.
theorem tom_total_dimes : total_dimes = 48 := by
  sorry

end tom_total_dimes_l2349_234958


namespace fraction_eq_zero_iff_x_eq_2_l2349_234944

theorem fraction_eq_zero_iff_x_eq_2 (x : ℝ) : (x - 2) / (x + 2) = 0 ↔ x = 2 := by sorry

end fraction_eq_zero_iff_x_eq_2_l2349_234944


namespace age_of_25th_student_l2349_234921

-- Definitions derived from problem conditions
def averageAgeClass (totalAge : ℕ) (totalStudents : ℕ) : ℕ := totalAge / totalStudents
def totalAgeGivenAverage (numStudents : ℕ) (averageAge : ℕ) : ℕ := numStudents * averageAge

-- Given conditions
def totalAgeOfAllStudents := 25 * 24
def totalAgeOf8Students := totalAgeGivenAverage 8 22
def totalAgeOf10Students := totalAgeGivenAverage 10 20
def totalAgeOf6Students := totalAgeGivenAverage 6 28
def totalAgeOf24Students := totalAgeOf8Students + totalAgeOf10Students + totalAgeOf6Students

-- The proof that the age of the 25th student is 56 years
theorem age_of_25th_student : totalAgeOfAllStudents - totalAgeOf24Students = 56 := by
  sorry

end age_of_25th_student_l2349_234921


namespace pears_sold_in_a_day_l2349_234948

-- Define the conditions
variable (morning_pears afternoon_pears : ℕ)
variable (h1 : afternoon_pears = 2 * morning_pears)
variable (h2 : afternoon_pears = 320)

-- Lean theorem statement to prove the question answer
theorem pears_sold_in_a_day :
  (morning_pears + afternoon_pears = 480) :=
by
  -- Insert proof here
  sorry

end pears_sold_in_a_day_l2349_234948


namespace fourth_root_expression_l2349_234915

-- Define a positive real number y
variable (y : ℝ) (hy : 0 < y)

-- State the problem in Lean
theorem fourth_root_expression : 
  Real.sqrt (Real.sqrt (y^2 * Real.sqrt y)) = y^(5/8) := sorry

end fourth_root_expression_l2349_234915


namespace simplify_radical_l2349_234920

theorem simplify_radical (x : ℝ) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) :=
by
  sorry

end simplify_radical_l2349_234920


namespace calculate_total_amount_l2349_234902

theorem calculate_total_amount
  (price1 discount1 price2 discount2 additional_discount : ℝ)
  (h1 : price1 = 76) (h2 : discount1 = 25)
  (h3 : price2 = 85) (h4 : discount2 = 15)
  (h5 : additional_discount = 10) :
  price1 - discount1 + price2 - discount2 - additional_discount = 111 :=
by {
  sorry
}

end calculate_total_amount_l2349_234902


namespace find_function_expression_l2349_234928

theorem find_function_expression (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = x^2 - 3 * x) → 
  (∀ x : ℝ, f x = x^2 - x - 2) :=
by
  sorry

end find_function_expression_l2349_234928


namespace fuel_tank_capacity_l2349_234951

theorem fuel_tank_capacity
  (ethanol_A_fraction : ℝ)
  (ethanol_B_fraction : ℝ)
  (ethanol_total : ℝ)
  (fuel_A_volume : ℝ)
  (C : ℝ)
  (h1 : ethanol_A_fraction = 0.12)
  (h2 : ethanol_B_fraction = 0.16)
  (h3 : ethanol_total = 28)
  (h4 : fuel_A_volume = 99.99999999999999)
  (h5 : 0.12 * 99.99999999999999 + 0.16 * (C - 99.99999999999999) = 28) :
  C = 200 := 
sorry

end fuel_tank_capacity_l2349_234951


namespace pants_price_l2349_234932

theorem pants_price (P B : ℝ) 
  (condition1 : P + B = 70.93)
  (condition2 : P = B - 2.93) : 
  P = 34.00 :=
by
  sorry

end pants_price_l2349_234932


namespace correct_expression_l2349_234901

variables {a b c : ℝ}

theorem correct_expression :
  -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b :=
by
  sorry

end correct_expression_l2349_234901


namespace circles_intersect_l2349_234906

-- Definition of the first circle
def circle1 (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Definition of the second circle
def circle2 (x y : ℝ) (r : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Statement proving the range of r for which the circles intersect
theorem circles_intersect (r : ℝ) (h : r > 0) : (∃ x y : ℝ, circle1 x y r ∧ circle2 x y r) → (2 ≤ r ∧ r ≤ 12) :=
by
  -- Definition of the distance between centers and conditions for intersection
  sorry

end circles_intersect_l2349_234906


namespace least_num_to_divisible_l2349_234985

theorem least_num_to_divisible (n : ℕ) : (1056 + n) % 27 = 0 → n = 24 :=
by
  sorry

end least_num_to_divisible_l2349_234985


namespace negation_universal_proposition_l2349_234962

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, |x| + x^4 ≥ 0) ↔ ∃ x₀ : ℝ, |x₀| + x₀^4 < 0 :=
by
  sorry

end negation_universal_proposition_l2349_234962


namespace value_of_a_8_l2349_234919

noncomputable def S (n : ℕ) : ℕ := n^2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then S n else S n - S (n - 1)

theorem value_of_a_8 : a 8 = 15 := 
by
  sorry

end value_of_a_8_l2349_234919


namespace range_of_x_satisfying_inequality_l2349_234987

def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x_satisfying_inequality :
  { x : ℝ | otimes x (x - 2) < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end range_of_x_satisfying_inequality_l2349_234987


namespace solve_eq_1_solve_eq_2_l2349_234942

open Real

theorem solve_eq_1 :
  ∃ x : ℝ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -2.5 :=
by
  sorry

theorem solve_eq_2 :
  ∃ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39 / 35 :=
by
  sorry

end solve_eq_1_solve_eq_2_l2349_234942


namespace negation_of_proposition_l2349_234988

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1 :=
by sorry

end negation_of_proposition_l2349_234988


namespace ellipse_foci_y_axis_range_l2349_234963

theorem ellipse_foci_y_axis_range (k : ℝ) : 
  (2*k - 1 > 2 - k) → (2 - k > 0) → (1 < k ∧ k < 2) := 
by 
  intros h1 h2
  -- We use the assumptions to derive the target statement.
  sorry

end ellipse_foci_y_axis_range_l2349_234963


namespace simplify_expression_l2349_234991

theorem simplify_expression (x : ℝ) :
  (3 * x)^5 + (4 * x^2) * (3 * x^2) = 243 * x^5 + 12 * x^4 :=
by
  sorry

end simplify_expression_l2349_234991


namespace garden_ratio_l2349_234960

theorem garden_ratio (L W : ℕ) (h1 : L = 50) (h2 : 2 * L + 2 * W = 150) : L / W = 2 :=
by
  sorry

end garden_ratio_l2349_234960


namespace minimum_positive_period_of_f_l2349_234974

noncomputable def f (x : ℝ) : ℝ := (1 + (Real.sqrt 3) * Real.tan x) * Real.cos x

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end minimum_positive_period_of_f_l2349_234974


namespace angles_on_line_y_eq_x_l2349_234924

-- Define a predicate representing that an angle has its terminal side on the line y = x
def angle_on_line_y_eq_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- The goal is to prove that the set of all such angles is as stated
theorem angles_on_line_y_eq_x :
  { α : ℝ | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 } = { α : ℝ | angle_on_line_y_eq_x α } :=
sorry

end angles_on_line_y_eq_x_l2349_234924


namespace customers_served_total_l2349_234965

theorem customers_served_total :
  let Ann_hours := 8
  let Ann_rate := 7
  let Becky_hours := 7
  let Becky_rate := 8
  let Julia_hours := 6
  let Julia_rate := 6
  let lunch_break := 0.5
  let Ann_customers := (Ann_hours - lunch_break) * Ann_rate
  let Becky_customers := (Becky_hours - lunch_break) * Becky_rate
  let Julia_customers := (Julia_hours - lunch_break) * Julia_rate
  Ann_customers + Becky_customers + Julia_customers = 137 := by
  sorry

end customers_served_total_l2349_234965


namespace geometric_sequence_general_term_l2349_234914

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∃ q : ℝ, (a n = 3 * q ^ (n - 1)) := by
  sorry

end geometric_sequence_general_term_l2349_234914


namespace problem_l2349_234905

def f (n : ℕ) : ℤ := 3 ^ (2 * n) - 32 * n ^ 2 + 24 * n - 1

theorem problem (n : ℕ) (h : 0 < n) : 512 ∣ f n := sorry

end problem_l2349_234905


namespace convex_polygon_triangle_count_l2349_234946

theorem convex_polygon_triangle_count {n : ℕ} (h : n ≥ 5) :
  ∃ T : ℕ, T ≤ n * (2 * n - 5) / 3 :=
by
  sorry

end convex_polygon_triangle_count_l2349_234946


namespace difference_of_squares_is_149_l2349_234998

-- Definitions of the conditions
def are_consecutive (n m : ℤ) : Prop := m = n + 1
def sum_less_than_150 (n : ℤ) : Prop := (n + (n + 1)) < 150

-- The difference of their squares
def difference_of_squares (n m : ℤ) : ℤ := (m * m) - (n * n)

-- Stating the problem where the answer expected is 149
theorem difference_of_squares_is_149 :
  ∀ n : ℤ, 
  ∀ m : ℤ,
  are_consecutive n m →
  sum_less_than_150 n →
  difference_of_squares n m = 149 :=
by
  sorry

end difference_of_squares_is_149_l2349_234998


namespace bob_eats_10_apples_l2349_234984

variable (B C : ℕ)
variable (h1 : B + C = 30)
variable (h2 : C = 2 * B)

theorem bob_eats_10_apples : B = 10 :=
by sorry

end bob_eats_10_apples_l2349_234984


namespace primes_satisfying_equation_l2349_234968

theorem primes_satisfying_equation :
  ∀ (p q : ℕ), p.Prime ∧ q.Prime → 
    (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ 
    (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
by
  sorry

end primes_satisfying_equation_l2349_234968


namespace minimum_value_expression_l2349_234952

theorem minimum_value_expression 
  (a b c : ℝ) 
  (h1 : 3 * a + 2 * b + c = 5) 
  (h2 : 2 * a + b - 3 * c = 1) 
  (h3 : 0 ≤ a) 
  (h4 : 0 ≤ b) 
  (h5 : 0 ≤ c) : 
  ∃(c : ℝ), (c ≥ 3/7 ∧ c ≤ 7/11) ∧ (3 * a + b - 7 * c = -5/7) :=
sorry 

end minimum_value_expression_l2349_234952


namespace discount_store_purchase_l2349_234994

theorem discount_store_purchase (n x y : ℕ) (hn : 2 * n + (x + y) = 2 * n) 
(h1 : 8 * x + 9 * y = 172) (hx : 0 ≤ x) (hy : 0 ≤ y): 
x = 8 ∧ y = 12 :=
sorry

end discount_store_purchase_l2349_234994


namespace square_root_combination_l2349_234912

theorem square_root_combination (a : ℝ) (h : 1 + a = 4 - 2 * a) : a = 1 :=
by
  -- proof goes here
  sorry

end square_root_combination_l2349_234912


namespace repeating_decimal_sum_l2349_234923

theorem repeating_decimal_sum :
  (0.12121212 + 0.003003003 + 0.0000500005 : ℚ) = 124215 / 999999 :=
by 
  have h1 : (0.12121212 : ℚ) = (0.12 + 0.0012) := sorry
  have h2 : (0.003003003 : ℚ) = (0.003 + 0.000003) := sorry
  have h3 : (0.0000500005 : ℚ) = (0.00005 + 0.0000000005) := sorry
  sorry


end repeating_decimal_sum_l2349_234923


namespace zengshan_suanfa_tongzong_l2349_234947

-- Definitions
variables (x y : ℝ)
variables (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5)

-- Theorem
theorem zengshan_suanfa_tongzong :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  -- Starting with the given hypotheses
  exact ⟨h1, h2⟩

end zengshan_suanfa_tongzong_l2349_234947


namespace concentric_spheres_volume_l2349_234938

theorem concentric_spheres_volume :
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  volume r3 - volume r2 = 876 * Real.pi := 
by
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  show volume r3 - volume r2 = 876 * Real.pi
  sorry

end concentric_spheres_volume_l2349_234938


namespace B_work_days_l2349_234955

-- Define work rates and conditions
def A_work_rate : ℚ := 1 / 18
def B_work_rate : ℚ := 1 / 15
def A_days_after_B_left : ℚ := 6
def total_work : ℚ := 1

-- Theorem statement
theorem B_work_days : ∃ x : ℚ, (x * B_work_rate + A_days_after_B_left * A_work_rate = total_work) → x = 10 := by
  sorry

end B_work_days_l2349_234955


namespace average_mark_second_class_l2349_234989

theorem average_mark_second_class
  (avg_mark_class1 : ℝ)
  (num_students_class1 : ℕ)
  (num_students_class2 : ℕ)
  (combined_avg_mark : ℝ) 
  (total_students : ℕ)
  (total_marks_combined : ℝ) :
  avg_mark_class1 * num_students_class1 + x * num_students_class2 = total_marks_combined →
  num_students_class1 + num_students_class2 = total_students →
  combined_avg_mark * total_students = total_marks_combined →
  avg_mark_class1 = 40 →
  num_students_class1 = 30 →
  num_students_class2 = 50 →
  combined_avg_mark = 58.75 →
  total_students = 80 →
  total_marks_combined = 4700 →
  x = 70 :=
by
  intros
  sorry

end average_mark_second_class_l2349_234989


namespace calculation_division_l2349_234908

theorem calculation_division :
  ((27 * 0.92 * 0.85) / (23 * 1.7 * 1.8)) = 0.3 :=
by
  sorry

end calculation_division_l2349_234908


namespace sacksPerSectionDaily_l2349_234964

variable (totalSacks : ℕ) (sections : ℕ) (sacksPerSection : ℕ)

-- Conditions from the problem
variables (h1 : totalSacks = 360) (h2 : sections = 8)

-- The theorem statement
theorem sacksPerSectionDaily : sacksPerSection = 45 :=
by
  have h3 : totalSacks / sections = 45 := by sorry
  have h4 : sacksPerSection = totalSacks / sections := by sorry
  exact Eq.trans h4 h3

end sacksPerSectionDaily_l2349_234964


namespace box_width_l2349_234934

variable (l h vc : ℕ)
variable (nc : ℕ)
variable (v : ℕ)

-- Given
def length_box := 8
def height_box := 5
def volume_cube := 10
def num_cubes := 60
def volume_box := num_cubes * volume_cube

-- To Prove
theorem box_width : (volume_box = l * h * w) → w = 15 :=
by
  intro h1
  sorry

end box_width_l2349_234934


namespace victoria_worked_weeks_l2349_234933

-- Definitions for given conditions
def hours_worked_per_day : ℕ := 9
def total_hours_worked : ℕ := 315
def days_in_week : ℕ := 7

-- Main theorem to prove
theorem victoria_worked_weeks : total_hours_worked / hours_worked_per_day / days_in_week = 5 :=
by
  sorry

end victoria_worked_weeks_l2349_234933


namespace garden_breadth_l2349_234900

-- Problem statement conditions
def perimeter : ℝ := 600
def length : ℝ := 205

-- Translate the problem into Lean:
theorem garden_breadth (breadth : ℝ) (h1 : 2 * (length + breadth) = perimeter) : breadth = 95 := 
by sorry

end garden_breadth_l2349_234900


namespace square_side_percentage_increase_l2349_234936

theorem square_side_percentage_increase (s : ℝ) (p : ℝ) :
  (s * (1 + p / 100)) ^ 2 = 1.44 * s ^ 2 → p = 20 :=
by
  sorry

end square_side_percentage_increase_l2349_234936


namespace interest_rate_is_5_percent_l2349_234941

noncomputable def interest_rate_1200_loan (R : ℝ) : Prop :=
  let time := 3.888888888888889
  let principal_1000 := 1000
  let principal_1200 := 1200
  let rate_1000 := 0.03
  let total_interest := 350
  principal_1000 * rate_1000 * time + principal_1200 * (R / 100) * time = total_interest

theorem interest_rate_is_5_percent :
  interest_rate_1200_loan 5 :=
by
  sorry

end interest_rate_is_5_percent_l2349_234941


namespace walmart_total_sales_l2349_234943

-- Define the constants for the prices
def thermometer_price : ℕ := 2
def hot_water_bottle_price : ℕ := 6

-- Define the quantities and relationships
def hot_water_bottles_sold : ℕ := 60
def thermometer_ratio : ℕ := 7
def thermometers_sold : ℕ := thermometer_ratio * hot_water_bottles_sold

-- Define the total sales for thermometers and hot-water bottles
def thermometer_sales : ℕ := thermometers_sold * thermometer_price
def hot_water_bottle_sales : ℕ := hot_water_bottles_sold * hot_water_bottle_price

-- Define the total sales amount
def total_sales : ℕ := thermometer_sales + hot_water_bottle_sales

-- Theorem statement
theorem walmart_total_sales : total_sales = 1200 := by
  sorry

end walmart_total_sales_l2349_234943


namespace total_cost_of_aquarium_l2349_234995

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l2349_234995


namespace count_ways_to_choose_4_cards_l2349_234903

-- A standard deck has 4 suits
def suits : Finset ℕ := {1, 2, 3, 4}

-- Each suit has 6 even cards: 2, 4, 6, 8, 10, and Queen (12)
def even_cards_per_suit : Finset ℕ := {2, 4, 6, 8, 10, 12}

-- Define the problem in Lean: 
-- Total number of ways to choose 4 cards such that all cards are of different suits and each is an even card.
theorem count_ways_to_choose_4_cards : (suits.card = 4 ∧ even_cards_per_suit.card = 6) → (1 * 6^4 = 1296) :=
by
  intros h
  have suits_distinct : suits.card = 4 := h.1
  have even_cards_count : even_cards_per_suit.card = 6 := h.2
  sorry

end count_ways_to_choose_4_cards_l2349_234903


namespace parabola_eqn_min_distance_l2349_234931

theorem parabola_eqn (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0) :
  (∀ x : ℝ,  y = a * x^2 + b * x) ↔ (∀ x : ℝ, y = (1/3) * x^2 - (2/3) * x) :=
by
  sorry

theorem min_distance (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0)
  (line_eq : ∀ x, (y : ℝ) = x - 25/4) :
  (∀ P : ℝ × ℝ, ∃ P_min : ℝ × ℝ, P_min = (5/2, 5/12)) :=
by
  sorry

end parabola_eqn_min_distance_l2349_234931


namespace find_x_l2349_234961

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_a_minus_b (x : ℝ) : ℝ × ℝ := ((1 - x), (4))

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition of perpendicular vectors
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The theorem to prove
theorem find_x : ∃ x : ℝ, is_perpendicular vector_a (vector_a_minus_b x) ∧ x = 9 :=
by {
  -- Sorry statement used to skip proof
  sorry
}

end find_x_l2349_234961


namespace sufficient_not_necessary_l2349_234926

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) : (a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) :=
by
  sorry

end sufficient_not_necessary_l2349_234926


namespace find_other_solution_l2349_234925

theorem find_other_solution (x : ℚ) :
  (72 * x ^ 2 + 43 = 113 * x - 12) → (x = 3 / 8) → (x = 43 / 36 ∨ x = 3 / 8) :=
by
  sorry

end find_other_solution_l2349_234925


namespace alex_charge_per_trip_l2349_234954

theorem alex_charge_per_trip (x : ℝ)
  (savings_needed : ℝ) (n_trips : ℝ) (worth_groceries : ℝ) (charge_per_grocery_percent : ℝ) :
  savings_needed = 100 → 
  n_trips = 40 →
  worth_groceries = 800 →
  charge_per_grocery_percent = 0.05 →
  n_trips * x + charge_per_grocery_percent * worth_groceries = savings_needed →
  x = 1.5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end alex_charge_per_trip_l2349_234954


namespace inequality_proof_l2349_234982

variable (a b c d : ℝ)
variable (habcda : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ab + bc + cd + da = 1)

theorem inequality_proof :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ (ab + bc + cd + da = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by sorry

end inequality_proof_l2349_234982


namespace find_b_in_triangle_l2349_234971

theorem find_b_in_triangle
  (a b c A B C : ℝ)
  (cos_A : ℝ) (cos_C : ℝ)
  (ha : a = 1)
  (hcos_A : cos_A = 4 / 5)
  (hcos_C : cos_C = 5 / 13) :
  b = 21 / 13 :=
by
  sorry

end find_b_in_triangle_l2349_234971


namespace exterior_angle_regular_octagon_l2349_234929

theorem exterior_angle_regular_octagon : 
  (∃ n : ℕ, n = 8 ∧ ∀ (i : ℕ), i < n → true) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end exterior_angle_regular_octagon_l2349_234929


namespace parabola_vertex_l2349_234945

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, y = 1 / 2 * (x + 1) ^ 2 - 1 / 2) →
    (h = -1 ∧ k = -1 / 2) :=
by
  sorry

end parabola_vertex_l2349_234945


namespace negation_of_existence_statement_l2349_234992

theorem negation_of_existence_statement :
  (¬ ∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_existence_statement_l2349_234992


namespace calories_consumed_in_week_l2349_234999

-- Define the calorie content of each type of burger
def calorie_A := 350
def calorie_B := 450
def calorie_C := 550

-- Define Dimitri's burger consumption over the 7 days
def consumption_day1 := (2 * calorie_A) + (1 * calorie_B)
def consumption_day2 := (1 * calorie_A) + (2 * calorie_B) + (1 * calorie_C)
def consumption_day3 := (1 * calorie_A) + (1 * calorie_B) + (2 * calorie_C)
def consumption_day4 := (3 * calorie_B)
def consumption_day5 := (1 * calorie_A) + (1 * calorie_B) + (1 * calorie_C)
def consumption_day6 := (2 * calorie_A) + (3 * calorie_C)
def consumption_day7 := (1 * calorie_B) + (2 * calorie_C)

-- Define the total weekly calorie consumption
def total_weekly_calories :=
  consumption_day1 + consumption_day2 + consumption_day3 +
  consumption_day4 + consumption_day5 + consumption_day6 + consumption_day7

-- State and prove the main theorem
theorem calories_consumed_in_week :
  total_weekly_calories = 11450 := 
by
  sorry

end calories_consumed_in_week_l2349_234999


namespace value_of_k_l2349_234927

theorem value_of_k (k : ℤ) (h : (∀ x : ℤ, (x^2 - k * x - 6) = (x - 2) * (x + 3))) : k = -1 := by
  sorry

end value_of_k_l2349_234927
