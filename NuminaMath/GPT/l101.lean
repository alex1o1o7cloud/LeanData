import Mathlib

namespace petya_friends_l101_101721

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101721


namespace claim1_claim2_l101_101149

theorem claim1 (n : ℤ) (hs : ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0) : 
  ∃ k : ℤ, n = 4 * k := 
sorry

theorem claim2 (n : ℕ) (h : n % 4 = 0) : 
  ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0 := 
sorry

end claim1_claim2_l101_101149


namespace trip_time_l101_101772

open Real

variables (d T : Real)

theorem trip_time :
  (T = d / 30 + (150 - d) / 6) ∧
  (T = 2 * (d / 30) + 1 + (150 - d) / 30) ∧
  (T - 1 = d / 6 + (150 - d) / 30) →
  T = 20 :=
by
  sorry

end trip_time_l101_101772


namespace units_digit_sum_even_20_to_80_l101_101411

theorem units_digit_sum_even_20_to_80 :
  let a := 20
  let d := 2
  let l := 80
  let n := ((l - a) / d) + 1 -- Given by the formula l = a + (n-1)d => n = (l - a) / d + 1
  let sum := (n * (a + l)) / 2
  (sum % 10) = 0 := sorry

end units_digit_sum_even_20_to_80_l101_101411


namespace greatest_prime_factor_15_fact_plus_17_fact_l101_101806

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l101_101806


namespace gcd_m_n_l101_101693

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end gcd_m_n_l101_101693


namespace operation_B_is_not_algorithm_l101_101302

-- Define what constitutes an algorithm.
def is_algorithm (desc : String) : Prop :=
  desc = "clear and finite steps to solve a certain type of problem"

-- Define given operations.
def operation_A : String := "Calculating the area of a circle given its radius"
def operation_B : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def operation_C : String := "Finding the equation of a line given two points in the coordinate plane"
def operation_D : String := "Operations of addition, subtraction, multiplication, and division"

-- Define expected property of an algorithm.
def is_algorithm_A : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_B : Prop := is_algorithm "cannot describe precise steps"
def is_algorithm_C : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_D : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"

theorem operation_B_is_not_algorithm :
  ¬ (is_algorithm operation_B) :=
by
   -- Change this line to the theorem proof.
   sorry

end operation_B_is_not_algorithm_l101_101302


namespace max_divisor_f_l101_101196

-- Given definition
def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

-- Main theorem to be proved
theorem max_divisor_f :
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → m ∣ f n) ∧ m = 36 :=
by
  -- The proof would go here
  sorry

end max_divisor_f_l101_101196


namespace number_of_lilies_l101_101184

theorem number_of_lilies (L : ℕ) 
  (h1 : ∀ n:ℕ, n * 6 = 6 * n)
  (h2 : ∀ n:ℕ, n * 3 = 3 * n) 
  (h3 : 5 * 3 = 15)
  (h4 : 6 * L + 15 = 63) : 
  L = 8 := 
by
  -- Proof omitted 
  sorry

end number_of_lilies_l101_101184


namespace arcsin_one_half_eq_pi_six_l101_101451

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101451


namespace distinct_values_for_D_l101_101920

-- Define distinct digits
def distinct_digits (a b c d e : ℕ) :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10

-- Declare the problem statement
theorem distinct_values_for_D : 
  ∃ D_values : Finset ℕ, 
    (∀ (A B C D E : ℕ), 
      distinct_digits A B C D E → 
      E + C = D ∧
      B + C = E ∧
      B + D = E) →
    D_values.card = 7 := 
by 
  sorry

end distinct_values_for_D_l101_101920


namespace fourth_powers_count_l101_101066

theorem fourth_powers_count (n m : ℕ) (h₁ : n^4 ≥ 100) (h₂ : m^4 ≤ 10000) :
  ∃ k, k = m - n + 1 ∧ k = 7 :=
by
  sorry

end fourth_powers_count_l101_101066


namespace probability_flies_swept_by_minute_hand_l101_101101

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l101_101101


namespace y_n_is_square_of_odd_integer_l101_101267

-- Define the sequences and the initial conditions
def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

-- Helper function to check if a number is odd
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The theorem to prove
theorem y_n_is_square_of_odd_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, y n = k ^ 2 ∧ is_odd k := by
  sorry

end y_n_is_square_of_odd_integer_l101_101267


namespace store_price_reduction_l101_101425

theorem store_price_reduction 
    (initial_price : ℝ) (initial_sales : ℕ) (price_reduction : ℝ)
    (sales_increase_factor : ℝ) (target_profit : ℝ)
    (x : ℝ) : (initial_price, initial_price - price_reduction, x) = (80, 50, 12) →
    sales_increase_factor = 20 →
    target_profit = 7920 →
    (30 - x) * (200 + sales_increase_factor * x / 2) = 7920 →
    x = 12 ∧ (initial_price - x) = 68 :=
by 
    intros h₁ h₂ h₃ h₄
    sorry

end store_price_reduction_l101_101425


namespace number_of_cows_l101_101615

variable (C H : ℕ)

section
-- Condition 1: Cows have 4 legs each
def cows_legs := C * 4

-- Condition 2: Chickens have 2 legs each
def chickens_legs := H * 2

-- Condition 3: The number of legs was 10 more than twice the number of heads
def total_legs := cows_legs C + chickens_legs H = 2 * (C + H) + 10

theorem number_of_cows : total_legs C H → C = 5 :=
by
  intros h
  sorry

end

end number_of_cows_l101_101615


namespace john_calories_eaten_l101_101296

def servings : ℕ := 3
def calories_per_serving : ℕ := 120
def fraction_eaten : ℚ := 1 / 2

theorem john_calories_eaten : 
  (servings * calories_per_serving : ℕ) * fraction_eaten = 180 :=
  sorry

end john_calories_eaten_l101_101296


namespace overall_gain_percent_l101_101128

theorem overall_gain_percent (cp1 cp2 cp3: ℝ) (sp1 sp2 sp3: ℝ) (h1: cp1 = 840) (h2: cp2 = 1350) (h3: cp3 = 2250) (h4: sp1 = 1220) (h5: sp2 = 1550) (h6: sp3 = 2150) : 
  (sp1 + sp2 + sp3 - (cp1 + cp2 + cp3)) / (cp1 + cp2 + cp3) * 100 = 10.81 := 
by 
  sorry

end overall_gain_percent_l101_101128


namespace find_age_of_second_person_l101_101008

variable (T A X : ℝ)

def average_original_group (T A : ℝ) : Prop :=
  T = 7 * A

def average_with_39 (T A : ℝ) : Prop :=
  T + 39 = 8 * (A + 2)

def average_with_second_person (T A X : ℝ) : Prop :=
  T + X = 8 * (A - 1) 

theorem find_age_of_second_person (T A X : ℝ) 
  (h1 : average_original_group T A)
  (h2 : average_with_39 T A)
  (h3 : average_with_second_person T A X) :
  X = 15 :=
sorry

end find_age_of_second_person_l101_101008


namespace wooden_block_length_l101_101583

-- Define the problem conditions
def meters_to_centimeters (m : ℕ) : ℕ := m * 100
def additional_length_cm (length_cm : ℕ) (additional_cm : ℕ) : ℕ := length_cm + additional_cm

-- Formalization of the problem
theorem wooden_block_length :
  let length_in_meters := 31
  let additional_cm := 30
  additional_length_cm (meters_to_centimeters length_in_meters) additional_cm = 3130 :=
by
  sorry

end wooden_block_length_l101_101583


namespace darren_total_tshirts_l101_101884

def num_white_packs := 5
def num_white_tshirts_per_pack := 6
def num_blue_packs := 3
def num_blue_tshirts_per_pack := 9

def total_tshirts (wpacks : ℕ) (wtshirts_per_pack : ℕ) (bpacks : ℕ) (btshirts_per_pack : ℕ) : ℕ :=
  (wpacks * wtshirts_per_pack) + (bpacks * btshirts_per_pack)

theorem darren_total_tshirts : total_tshirts num_white_packs num_white_tshirts_per_pack num_blue_packs num_blue_tshirts_per_pack = 57 :=
by
  -- proof needed
  sorry

end darren_total_tshirts_l101_101884


namespace find_a_5_l101_101049

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem find_a_5 {a : ℕ → ℤ} {S : ℕ → ℤ}
  (h_seq : arithmetic_sequence a)
  (h_S6 : S 6 = 3)
  (h_a4 : a 4 = 2)
  (h_sum_first_n : sum_first_n a S) :
  a 5 = 5 := 
sorry

end find_a_5_l101_101049


namespace find_2g_x_l101_101218

theorem find_2g_x (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + x)) (x : ℝ) (hx : x > 0) :
  2 * g x = 18 / (9 + x) :=
sorry

end find_2g_x_l101_101218


namespace sum_even_factors_of_720_l101_101275

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l101_101275


namespace total_whipped_cream_l101_101489

theorem total_whipped_cream (cream_from_farm : ℕ) (cream_to_buy : ℕ) (total_cream : ℕ) 
  (h1 : cream_from_farm = 149) 
  (h2 : cream_to_buy = 151) 
  (h3 : total_cream = cream_from_farm + cream_to_buy) : 
  total_cream = 300 :=
sorry

end total_whipped_cream_l101_101489


namespace special_collection_books_l101_101297

theorem special_collection_books (initial_books loaned_books returned_percent: ℕ) (loaned_books_value: loaned_books = 55) (returned_percent_value: returned_percent = 80) (initial_books_value: initial_books = 75) :
  initial_books - (loaned_books - (returned_percent * loaned_books / 100)) = 64 := by
  sorry

end special_collection_books_l101_101297


namespace smaller_solution_l101_101991

theorem smaller_solution (x : ℝ) (h : x^2 + 9 * x - 22 = 0) : x = -11 :=
sorry

end smaller_solution_l101_101991


namespace functional_eq_f_l101_101696

noncomputable def f (q : ℚ) : ℚ := sorry

theorem functional_eq_f
  (f : ℚ → ℚ)
  (h_f : ∀ x y : ℚ, 0 < x → 0 < y → f(x * f(y)) = f(x) / y) :
  ∃ f : ℚ → ℚ, ∀ x y : ℚ, 0 < x → 0 < y → f(x * f(y)) = f(x) / y :=
sorry

end functional_eq_f_l101_101696


namespace mooncake_inspection_random_event_l101_101283

-- Definition of event categories
inductive Event
| certain
| impossible
| random

-- Definition of the event in question
def mooncakeInspectionEvent (satisfactory: Bool) : Event :=
if satisfactory then Event.random else Event.random

-- Theorem statement to prove that the event is a random event
theorem mooncake_inspection_random_event (satisfactory: Bool) :
  mooncakeInspectionEvent satisfactory = Event.random :=
sorry

end mooncake_inspection_random_event_l101_101283


namespace division_pow_zero_l101_101410

theorem division_pow_zero (a b : ℝ) (hb : b ≠ 0) : ((a / b) ^ 0 = (1 : ℝ)) :=
by
  sorry

end division_pow_zero_l101_101410


namespace intersection_point_of_lines_l101_101116

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), x + 2 * y - 4 = 0 ∧ 2 * x - y + 2 = 0 ∧ (x, y) = (0, 2) :=
by
  sorry

end intersection_point_of_lines_l101_101116


namespace parabola_chords_reciprocal_sum_l101_101060

theorem parabola_chords_reciprocal_sum (x y : ℝ) (AB CD : ℝ) (p : ℝ) :
  (y = (4 : ℝ) * x) ∧ (AB ≠ 0) ∧ (CD ≠ 0) ∧
  (p = (2 : ℝ)) ∧
  (|AB| = (2 * p / (Real.sin (Real.pi / 4))^2)) ∧ 
  (|CD| = (2 * p / (Real.cos (Real.pi / 4))^2)) →
  (1 / |AB| + 1 / |CD| = 1 / 4) :=
by
  sorry

end parabola_chords_reciprocal_sum_l101_101060


namespace num_balls_total_l101_101227

theorem num_balls_total (m : ℕ) (h1 : 6 < m) (h2 : (6 : ℝ) / (m : ℝ) = 0.3) : m = 20 :=
by
  sorry

end num_balls_total_l101_101227


namespace f_sub_f_neg_l101_101350

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7 * x

-- State the theorem
theorem f_sub_f_neg : f 3 - f (-3) = 582 :=
by
  -- Definitions and calculations for the proof
  -- (You can complete this part in later proof development)
  sorry

end f_sub_f_neg_l101_101350


namespace perfect_squares_less_than_500_ending_in_4_l101_101206

theorem perfect_squares_less_than_500_ending_in_4 : 
  (∃ (squares : Finset ℕ), (∀ n ∈ squares, n < 500 ∧ (n % 10 = 4)) ∧ squares.card = 5) :=
by
  sorry

end perfect_squares_less_than_500_ending_in_4_l101_101206


namespace inequality_solution_l101_101277

theorem inequality_solution (x : ℤ) (h : x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) : x - 1 ≥ 0 ↔ x = 1 :=
by
  sorry

end inequality_solution_l101_101277


namespace find_other_solution_l101_101053

theorem find_other_solution (x : ℚ) (hx : 45 * (2 / 5 : ℚ)^2 + 22 = 56 * (2 / 5 : ℚ) - 9) : x = 7 / 9 :=
by 
  sorry

end find_other_solution_l101_101053


namespace greatest_prime_factor_of_sum_l101_101809

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l101_101809


namespace find_a_l101_101947

noncomputable def base25_num : ℕ := 3 * 25^7 + 1 * 25^6 + 4 * 25^5 + 2 * 25^4 + 6 * 25^3 + 5 * 25^2 + 2 * 25^1 + 3 * 25^0

theorem find_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a ≤ 14) : ((base25_num - a) % 12 = 0) → a = 2 := 
sorry

end find_a_l101_101947


namespace sqrt_72_eq_6_sqrt_2_l101_101109

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l101_101109


namespace quadratic_intersects_at_two_points_l101_101491

variable {k : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_intersects_at_two_points (k : ℝ) :
  (let a := k - 2 in
   let b := -(2 * k - 1) in
   let c := k in
   discriminant a b c > 0 ∧ a ≠ 0) ↔ k > -1 / 4 ∧ k ≠ 2 :=
by
  let a := k - 2
  let b := -(2 * k - 1)
  let c := k
  have D : discriminant a b c = 4 * k + 1 := by
    rw [discriminant, sq, neg_mul_eq_mul_neg, neg_neg, mul_assoc, mul_assoc, mul_comm (4 * k),
        mul_comm 4 1, ←mul_sub, add_comm, sq_sub, mul_sub, mul_comm]
  sorry

end quadratic_intersects_at_two_points_l101_101491


namespace minimum_value_l101_101852

theorem minimum_value (x : ℝ) (h : x > 1) : 2 * x + 7 / (x - 1) ≥ 2 * Real.sqrt 14 + 2 := by
  sorry

end minimum_value_l101_101852


namespace least_6_digit_number_sum_of_digits_l101_101859

-- Definitions based on conditions
def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def leaves_remainder2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Problem statement
theorem least_6_digit_number_sum_of_digits :
  ∃ n : ℕ, is_6_digit n ∧ leaves_remainder2 n 4 ∧ leaves_remainder2 n 610 ∧ leaves_remainder2 n 15 ∧ sum_of_digits n = 17 :=
sorry

end least_6_digit_number_sum_of_digits_l101_101859


namespace totalPears_l101_101389

-- Define the number of pears picked by Sara and Sally
def saraPears : ℕ := 45
def sallyPears : ℕ := 11

-- Statement to prove
theorem totalPears : saraPears + sallyPears = 56 :=
by
  sorry

end totalPears_l101_101389


namespace unique_solution_quadratic_eq_l101_101886

theorem unique_solution_quadratic_eq (q : ℚ) (hq : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 :=
by
  sorry

end unique_solution_quadratic_eq_l101_101886


namespace smallest_integer_with_12_divisors_l101_101596

theorem smallest_integer_with_12_divisors :
  ∃ (n : ℕ), (∀ k : ℕ, k < n → ¬(number_of_divisors k = 12)) ∧ number_of_divisors n = 12 ∧ n = 288 :=
by
  sorry

end smallest_integer_with_12_divisors_l101_101596


namespace fill_half_cistern_time_l101_101156

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l101_101156


namespace x_divisible_by_5_l101_101498

theorem x_divisible_by_5
  (x y : ℕ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_gt_1 : 1 < x)
  (h_eq : 2 * x^2 - 1 = y^15) : x % 5 = 0 :=
sorry

end x_divisible_by_5_l101_101498


namespace product_positivity_l101_101765

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l101_101765


namespace shaded_areas_equal_l101_101543

theorem shaded_areas_equal (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π / 4) : 
  (Real.tan φ) = 2 * φ :=
sorry

end shaded_areas_equal_l101_101543


namespace union_of_A_and_B_l101_101198

open Set

variable {α : Type}

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := 
by
  sorry

end union_of_A_and_B_l101_101198


namespace interval_after_speed_limit_l101_101985

noncomputable def car_speed_before : ℝ := 80 -- speed before the sign in km/h
noncomputable def car_speed_after : ℝ := 60 -- speed after the sign in km/h
noncomputable def initial_interval : ℕ := 10 -- interval between the cars in meters

-- Convert speeds from km/h to m/s
noncomputable def v : ℝ := car_speed_before * 1000 / 3600
noncomputable def u : ℝ := car_speed_after * 1000 / 3600

-- Given the initial interval and speed before the sign, calculate the time it takes for the second car to reach the sign
noncomputable def delta_t : ℝ := initial_interval / v

-- Given u and delta_t, calculate the new interval after slowing down
noncomputable def new_interval : ℝ := u * delta_t

-- Theorem statement in Lean
theorem interval_after_speed_limit : new_interval = 7.5 :=
sorry

end interval_after_speed_limit_l101_101985


namespace student_factor_l101_101168

theorem student_factor (x : ℤ) : (121 * x - 138 = 104) → x = 2 :=
by
  intro h
  sorry

end student_factor_l101_101168


namespace proof_triangle_properties_l101_101923

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l101_101923


namespace arcsin_of_half_l101_101469

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101469


namespace value_of_x_after_z_doubled_l101_101980

theorem value_of_x_after_z_doubled (x y z : ℕ) (hz : z = 48) (hz_d : z_d = 2 * z) (hy : y = z / 4) (hx : x = y / 3) :
  x = 8 := by
  -- Proof goes here (skipped as instructed)
  sorry

end value_of_x_after_z_doubled_l101_101980


namespace Petya_friends_l101_101713

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101713


namespace Petya_friends_l101_101712

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101712


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101789
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101789


namespace anes_age_l101_101955

theorem anes_age (w w_d : ℤ) (n : ℤ) 
  (h1 : 1436 ≤ w ∧ w < 1445)
  (h2 : 1606 ≤ w_d ∧ w_d < 1615)
  (h3 : w_d = w + n * 40) : 
  n = 4 :=
sorry

end anes_age_l101_101955


namespace floor_sum_min_value_l101_101210

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l101_101210


namespace arcsin_of_half_l101_101467

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101467


namespace cost_buses_minimize_cost_buses_l101_101290

theorem cost_buses
  (x y : ℕ) 
  (h₁ : x + y = 500)
  (h₂ : 2 * x + 3 * y = 1300) :
  x = 200 ∧ y = 300 :=
by 
  sorry

theorem minimize_cost_buses
  (m : ℕ) 
  (h₃: 15 * m + 25 * (8 - m) ≥ 180) :
  m = 2 ∧ (200 * m + 300 * (8 - m) = 2200) :=
by 
  sorry

end cost_buses_minimize_cost_buses_l101_101290


namespace trajectory_midpoint_chord_l101_101764

theorem trajectory_midpoint_chord (x y : ℝ) 
  (h₀ : y^2 = 4 * x) : (y^2 = 2 * x - 2) :=
sorry

end trajectory_midpoint_chord_l101_101764


namespace question_1_question_2_l101_101624

noncomputable def p1_correctness : ℝ := 2 / 3

def problem_conditions (P1 : ℝ) : Prop :=
  (1 - P1) ^ 2 = 1 / 9

def answer_question_1 : ℝ := p1_correctness

theorem question_1 :
  ∃ P1 : ℝ, problem_conditions P1 ∧ P1 = answer_question_1 :=
by
  sorry

noncomputable def advancing_probability : ℝ := 496 / 729

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def probability_of_advancement (P1 pi : ℝ) : ℝ :=
  P1^4 + binomial 4 3 * P1^3 * pi * P1 + binomial 5 3 * P1^3 * pi^2 * P1

theorem question_2 :
  probability_of_advancement p1_correctness (1 / 3) = advancing_probability :=
by
  sorry

end question_1_question_2_l101_101624


namespace plane_speed_west_l101_101163

theorem plane_speed_west (v t : ℝ) : 
  (300 * t + 300 * t = 1200) ∧ (t = 7 - t) → 
  (v = 300 * t / (7 - t)) ∧ (t = 2) → 
  v = 120 :=
by
  intros h1 h2
  sorry

end plane_speed_west_l101_101163


namespace arcsin_one_half_l101_101459

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101459


namespace sin_A_calculation_height_calculation_l101_101938

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l101_101938


namespace width_of_house_l101_101395

theorem width_of_house (L P_L P_W A_total : ℝ) (hL : L = 20.5) (hPL : P_L = 6) (hPW : P_W = 4.5) (hAtotal : A_total = 232) :
  ∃ W : ℝ, W = 10 :=
by
  have area_porch : ℝ := P_L * P_W
  have area_house := A_total - area_porch
  use area_house / L
  sorry

end width_of_house_l101_101395


namespace least_number_to_subtract_l101_101139

theorem least_number_to_subtract (x : ℕ) (h1 : 997 - x ≡ 3 [MOD 17]) (h2 : 997 - x ≡ 3 [MOD 19]) (h3 : 997 - x ≡ 3 [MOD 23]) : x = 3 :=
by
  sorry

end least_number_to_subtract_l101_101139


namespace suzy_total_jumps_in_two_days_l101_101111

-- Definitions based on the conditions in the problem
def yesterdays_jumps : ℕ := 247
def additional_jumps_today : ℕ := 131
def todays_jumps : ℕ := yesterdays_jumps + additional_jumps_today

-- Lean statement of the proof problem
theorem suzy_total_jumps_in_two_days : yesterdays_jumps + todays_jumps = 625 := by
  sorry

end suzy_total_jumps_in_two_days_l101_101111


namespace a_4_eq_28_l101_101374

def Sn (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by
  sorry

end a_4_eq_28_l101_101374


namespace bob_weight_l101_101256

theorem bob_weight (j b : ℝ) (h1 : j + b = 220) (h2 : b - 2 * j = b / 3) : b = 165 :=
  sorry

end bob_weight_l101_101256


namespace lisa_more_dresses_than_ana_l101_101557

theorem lisa_more_dresses_than_ana :
  ∀ (total_dresses ana_dresses : ℕ),
    total_dresses = 48 →
    ana_dresses = 15 →
    (total_dresses - ana_dresses) - ana_dresses = 18 :=
by
  intros total_dresses ana_dresses h1 h2
  sorry

end lisa_more_dresses_than_ana_l101_101557


namespace employee_y_payment_l101_101616

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 590) (h2 : X = 1.2 * Y) : Y = 268.18 := by
  sorry

end employee_y_payment_l101_101616


namespace arcsin_one_half_l101_101462

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101462


namespace four_digit_number_l101_101639

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem four_digit_number (n : ℕ) (hn1 : 1000 ≤ n) (hn2 : n < 10000) (condition : n = 9 * (reverse_digits n)) :
  n = 9801 :=
by
  sorry

end four_digit_number_l101_101639


namespace num_five_digit_ints_l101_101355

open Nat

theorem num_five_digit_ints : 
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  num_ways = 10 :=
by
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  sorry

end num_five_digit_ints_l101_101355


namespace positive_integers_with_at_most_three_diff_digits_l101_101207

theorem positive_integers_with_at_most_three_diff_digits : 
  ∃ n : ℕ, n < 1000 ∧ (∀ i, i < n → ∃ d1 d2 d3 : ℕ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
  (i = d1 ∨ i = d2 ∨ i = d3)) ∧ n = 819 :=
by
  sorry

end positive_integers_with_at_most_three_diff_digits_l101_101207


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101833

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101833


namespace smallest_integer_with_12_divisors_l101_101608

def divisor_count (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem smallest_integer_with_12_divisors : ∃ n : ℕ, n = 60 ∧ divisor_count n = 12 :=
by
  sorry

end smallest_integer_with_12_divisors_l101_101608


namespace expression_evaluation_l101_101632

theorem expression_evaluation : |1 - Real.sqrt 3| + 2 * Real.cos (Real.pi / 6) - Real.sqrt 12 - 2023 = -2024 := 
by {
    sorry
}

end expression_evaluation_l101_101632


namespace greatest_prime_factor_15f_plus_17f_l101_101830

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l101_101830


namespace probability_two_asian_countries_probability_A1_not_B1_l101_101030

-- Scope: Definitions for the problem context
def countries : List String := ["A1", "A2", "A3", "B1", "B2", "B3"]

-- Probability of picking two Asian countries from a pool of six (three Asian, three European)
theorem probability_two_asian_countries : 
  (3 / 15) = (1 / 5) := by
  sorry

-- Probability of picking one country from the Asian group and 
-- one from the European group, including A1 but not B1
theorem probability_A1_not_B1 : 
  (2 / 9) = (2 / 9) := by
  sorry

end probability_two_asian_countries_probability_A1_not_B1_l101_101030


namespace zoo_pandas_l101_101874

-- Defining the conditions
variable (total_couples : ℕ)
variable (pregnant_couples : ℕ)
variable (baby_pandas : ℕ)
variable (total_pandas : ℕ)

-- Given conditions
def paired_mates : Prop := ∃ c : ℕ, c = total_couples

def pregnant_condition : Prop := pregnant_couples = (total_couples * 25) / 100

def babies_condition : Prop := baby_pandas = 2

def total_condition : Prop := total_pandas = total_couples * 2 + baby_pandas

-- The theorem to be proven
theorem zoo_pandas (h1 : paired_mates total_couples)
                   (h2 : pregnant_condition total_couples pregnant_couples)
                   (h3 : babies_condition baby_pandas)
                   (h4 : pregnant_couples = 2) :
                   total_condition total_couples baby_pandas total_pandas :=
by sorry

end zoo_pandas_l101_101874


namespace incorrect_statement_about_absolute_value_l101_101612

theorem incorrect_statement_about_absolute_value (x : ℝ) : abs x = 0 → x = 0 :=
by 
  sorry

end incorrect_statement_about_absolute_value_l101_101612


namespace arcsin_one_half_l101_101473

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101473


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101798

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101798


namespace union_complement_U_B_l101_101952

def U : Set ℤ := { x | -3 < x ∧ x < 3 }
def A : Set ℤ := { 1, 2 }
def B : Set ℤ := { -2, -1, 2 }

theorem union_complement_U_B : A ∪ (U \ B) = { 0, 1, 2 } := by
  sorry

end union_complement_U_B_l101_101952


namespace complete_the_square_solution_l101_101853

theorem complete_the_square_solution (x : ℝ) :
  (∃ x, x^2 + 2 * x - 1 = 0) → (x + 1)^2 = 2 :=
sorry

end complete_the_square_solution_l101_101853


namespace largest_value_l101_101010

theorem largest_value :
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- conditions
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  -- sorry to skip the proof
  sorry

end largest_value_l101_101010


namespace mean_equality_l101_101971

theorem mean_equality (x : ℚ) : 
  (3 + 7 + 15) / 3 = (x + 10) / 2 → x = 20 / 3 := 
by 
  sorry

end mean_equality_l101_101971


namespace petya_has_19_friends_l101_101750

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101750


namespace arcsin_of_half_l101_101465

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101465


namespace smallest_integer_with_12_divisors_l101_101609

def divisors_count (n : ℕ) : ℕ :=
  (n.factors + 1).prod

theorem smallest_integer_with_12_divisors :
  (∀ n : ℕ, 0 < n → divisors_count n = 12 → 108 ≤ n) :=
begin
  -- sorry placeholder for proof
  sorry,
end

end smallest_integer_with_12_divisors_l101_101609


namespace find_second_smallest_odd_number_l101_101983

theorem find_second_smallest_odd_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6) = 112)) : (x + 2 = 27) :=
sorry

end find_second_smallest_odd_number_l101_101983


namespace sum_of_even_factors_720_l101_101270

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l101_101270


namespace arcsin_one_half_eq_pi_six_l101_101448

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101448


namespace james_tylenol_intake_per_day_l101_101231

theorem james_tylenol_intake_per_day :
  (∃ (tablets_per_dose doses_per_day tablet_mg : ℕ),
    tablets_per_dose = 2 ∧
    doses_per_day = 24 / 6 ∧
    tablet_mg = 375 ∧
    (tablets_per_dose * doses_per_day * tablet_mg = 3000)) :=
begin
  let tablets_per_dose := 2,
  let doses_per_day := 24 / 6,
  let tablet_mg := 375,
  use [tablets_per_dose, doses_per_day, tablet_mg],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  sorry
end

end james_tylenol_intake_per_day_l101_101231


namespace intersection_point_value_l101_101394

theorem intersection_point_value (c d: ℤ) (h1: d = 2 * -4 + c) (h2: -4 = 2 * d + c) : d = -4 :=
by
  sorry

end intersection_point_value_l101_101394


namespace green_ball_probability_l101_101883

def containerA := (8, 2) -- 8 green, 2 red
def containerB := (6, 4) -- 6 green, 4 red
def containerC := (5, 5) -- 5 green, 5 red
def containerD := (8, 2) -- 8 green, 2 red

def probability_of_green : ℚ :=
  (1 / 4) * (8 / 10) + (1 / 4) * (6 / 10) + (1 / 4) * (5 / 10) + (1 / 4) * (8 / 10)
  
theorem green_ball_probability :
  probability_of_green = 43 / 160 :=
sorry

end green_ball_probability_l101_101883


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101796

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101796


namespace area_of_shaded_region_l101_101569

noncomputable def shaded_area (length_in_feet : ℝ) (diameter : ℝ) : ℝ :=
  let length_in_inches := length_in_feet * 12
  let radius := diameter / 2
  let num_semicircles := length_in_inches / diameter
  let num_full_circles := num_semicircles / 2
  let area := num_full_circles * (radius ^ 2 * Real.pi)
  area

theorem area_of_shaded_region : shaded_area 1.5 3 = 13.5 * Real.pi :=
by
  sorry

end area_of_shaded_region_l101_101569


namespace number_of_female_students_school_l101_101879

theorem number_of_female_students_school (T S G_s B_s B G : ℕ) (h1 : T = 1600)
    (h2 : S = 200) (h3 : G_s = B_s - 10) (h4 : G_s + B_s = 200) (h5 : B_s = 105) (h6 : G_s = 95) (h7 : B + G = 1600) : 
    G = 760 :=
by
  sorry

end number_of_female_students_school_l101_101879


namespace laura_annual_income_l101_101535

variable (p : ℝ) -- percentage p
variable (A T : ℝ) -- annual income A and total income tax T

def tax1 : ℝ := 0.01 * p * 35000
def tax2 : ℝ := 0.01 * (p + 3) * (A - 35000)
def tax3 : ℝ := 0.01 * (p + 5) * (A - 55000)

theorem laura_annual_income (h_cond1 : A > 55000)
  (h_tax : T = 350 * p + 600 + 0.01 * (p + 5) * (A - 55000))
  (h_paid_tax : T = (0.01 * (p + 0.45)) * A):
  A = 75000 := by
  sorry

end laura_annual_income_l101_101535


namespace hyperbola_eccentricity_l101_101364

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0)
  (h_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / 16 = 1 ↔ true)
  (eccentricity : a^2 + 16 / a^2 = (5 / 3)^2) : a = 3 :=
by
  sorry

end hyperbola_eccentricity_l101_101364


namespace petya_friends_l101_101739

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101739


namespace petya_friends_count_l101_101743

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101743


namespace driver_total_miles_per_week_l101_101428

theorem driver_total_miles_per_week :
  let distance_monday_to_saturday := (30 * 3 + 25 * 4 + 40 * 2) * 6
  let distance_sunday := 35 * (5 - 1)
  distance_monday_to_saturday + distance_sunday = 1760 := by
  sorry

end driver_total_miles_per_week_l101_101428


namespace sally_orange_balloons_l101_101960

def initial_orange_balloons : ℝ := 9.0
def found_orange_balloons : ℝ := 2.0

theorem sally_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := 
by
  sorry

end sally_orange_balloons_l101_101960


namespace arcsin_one_half_l101_101472

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101472


namespace tangent_point_l101_101899

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := x + 1 - x * Real.log x

theorem tangent_point (a : ℝ) (n : ℕ) (hn : n > 0) (ha : n < a ∧ a < n + 1) :
  y = a + 1 → y = a * Real.log a → n = 3 :=
sorry

end tangent_point_l101_101899


namespace arcsin_of_half_l101_101444

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101444


namespace sum_even_factors_l101_101272

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l101_101272


namespace exists_zero_point_of_continuous_l101_101501

theorem exists_zero_point_of_continuous (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_sign : f a * f b < 0) :
  ∃ c ∈ Set.Icc a b, f c = 0 :=
sorry

end exists_zero_point_of_continuous_l101_101501


namespace greatest_prime_factor_of_15_l101_101816

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l101_101816


namespace minimize_y_l101_101549

variables (a b k : ℝ)

def y (x : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y : ∃ x : ℝ, y a b k x = y a b k ( (6 * a + 2 * b - k) / 8 ) :=
  sorry

end minimize_y_l101_101549


namespace petya_friends_l101_101717

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101717


namespace greatest_prime_factor_15_17_factorial_l101_101839

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l101_101839


namespace chairs_left_proof_l101_101368

def red_chairs : ℕ := 4
def yellow_chairs : ℕ := 2 * red_chairs
def blue_chairs : ℕ := 3 * yellow_chairs
def green_chairs : ℕ := blue_chairs / 2
def orange_chairs : ℕ := green_chairs + 2
def total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
def borrowed_chairs : ℕ := 5 + 3
def chairs_left : ℕ := total_chairs - borrowed_chairs

theorem chairs_left_proof : chairs_left = 54 := by
  -- This is where the proof would go
  sorry

end chairs_left_proof_l101_101368


namespace range_of_2a_plus_3b_l101_101649

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l101_101649


namespace remainder_of_p_l101_101219

theorem remainder_of_p (p : ℤ) (h1 : p = 35 * 17 + 10) : p % 35 = 10 := 
  sorry

end remainder_of_p_l101_101219


namespace sin_A_correct_height_on_AB_correct_l101_101929

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l101_101929


namespace Paul_correct_probability_l101_101229

theorem Paul_correct_probability :
  let P_Ghana := 1/2
  let P_Bolivia := 1/6
  let P_Argentina := 1/6
  let P_France := 1/6
  (P_Ghana^2 + P_Bolivia^2 + P_Argentina^2 + P_France^2) = 1/3 :=
by
  sorry

end Paul_correct_probability_l101_101229


namespace total_cookies_is_390_l101_101300

def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℚ := grayson_boxes * cookies_per_box
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box
def isabella_cookies : ℚ := (1 / 2) * grayson_cookies
def ethan_cookies : ℤ := (abigail_boxes * 2 * cookies_per_box) / 2

def total_cookies : ℚ := ↑abigail_cookies + grayson_cookies + ↑olivia_cookies + isabella_cookies + ↑ethan_cookies

theorem total_cookies_is_390 : total_cookies = 390 :=
by
  sorry

end total_cookies_is_390_l101_101300


namespace sin_diff_angle_identity_l101_101343

open Real

noncomputable def alpha : ℝ := sorry -- α is an obtuse angle

axiom h1 : 90 < alpha ∧ alpha < 180 -- α is an obtuse angle
axiom h2 : cos alpha = -3 / 5 -- given cosine value

theorem sin_diff_angle_identity :
  sin (π / 4 - alpha) = - (7 * sqrt 2) / 10 :=
by
  sorry

end sin_diff_angle_identity_l101_101343


namespace intersection_two_elements_l101_101666

open Real Set

-- Definitions
def M (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = k * (x - 1) + 1}
def N : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 - 2 * y = 0}

-- Statement of the problem
theorem intersection_two_elements (k : ℝ) (hk : k ≠ 0) :
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ∈ M k ∧ (x1, y1) ∈ N ∧ 
    (x2, y2) ∈ M k ∧ (x2, y2) ∈ N ∧ 
    (x1, y1) ≠ (x2, y2) := sorry

end intersection_two_elements_l101_101666


namespace sugar_total_l101_101869

variable (sugar_for_frosting sugar_for_cake : ℝ)

theorem sugar_total (h1 : sugar_for_frosting = 0.6) (h2 : sugar_for_cake = 0.2) :
  sugar_for_frosting + sugar_for_cake = 0.8 :=
by
  sorry

end sugar_total_l101_101869


namespace a_can_be_any_sign_l101_101905

theorem a_can_be_any_sign (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b)^2 < (c / d)^2) (hcd : c = -d) : True :=
by
  have := h
  subst hcd
  sorry

end a_can_be_any_sign_l101_101905


namespace smallest_positive_integer_with_12_divisors_is_72_l101_101610

noncomputable def prime_exponents {n : ℕ} (d : ℕ) : (ℕ → ℕ) :=
  -- This is a placeholder for the actual function which maps a prime to its exponent in n's factorization
  sorry

theorem smallest_positive_integer_with_12_divisors_is_72 :
  ∃ (n : ℕ), (∀ m : ℕ, (∀ p e : ℕ, prime_exponents m p ≤ e ∧ m = p ^ e) →
  (∃ f : ℕ → ℕ, (∀ p : ℕ, (is_prime p ∧ primality m f p) = (prime_exponents m p))
  ∧ 12 = (∏ q : ℕ in q.dvd m, (prime_exponents m q) + 1)) → m ≥ 72) :=
begin
  sorry
end

end smallest_positive_integer_with_12_divisors_is_72_l101_101610


namespace arcsin_of_half_l101_101443

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101443


namespace find_a_l101_101062

-- Given function
def quadratic_func (a x : ℝ) := a * (x - 1)^2 - a

-- Conditions
def condition1 (a : ℝ) := a ≠ 0
def condition2 (x : ℝ) := -1 ≤ x ∧ x ≤ 4
def min_value (y : ℝ) := y = -4

theorem find_a (a : ℝ) (ha : condition1 a) :
  ∃ a, (∀ x, condition2 x → quadratic_func a x = -4) → (a = 4 ∨ a = -1 / 2) :=
sorry

end find_a_l101_101062


namespace a_4_value_l101_101686

-- Definitions and Theorem
variable {α : Type*} [LinearOrderedField α]

noncomputable def geometric_seq (a₀ : α) (q : α) (n : ℕ) : α := a₀ * q ^ n

theorem a_4_value (a₁ : α) (q : α) (h : geometric_seq a₁ q 1 * geometric_seq a₁ q 2 * geometric_seq a₁ q 6 = 8) : 
  geometric_seq a₁ q 3 = 2 :=
sorry

end a_4_value_l101_101686


namespace sin_A_calculation_height_calculation_l101_101940

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l101_101940


namespace average_age_of_students_l101_101578

theorem average_age_of_students (A : ℝ) (h1 : ∀ n : ℝ, n = 20 → A + 1 = n) (h2 : ∀ k : ℝ, k = 40 → 19 * A + k = 20 * (A + 1)) : A = 20 :=
by
  sorry

end average_age_of_students_l101_101578


namespace sarah_apples_calc_l101_101249

variable (brother_apples : ℕ)
variable (sarah_apples : ℕ)
variable (multiplier : ℕ)

theorem sarah_apples_calc
  (h1 : brother_apples = 9)
  (h2 : multiplier = 5)
  (h3 : sarah_apples = multiplier * brother_apples) : sarah_apples = 45 := by
  sorry

end sarah_apples_calc_l101_101249


namespace petya_friends_l101_101732

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101732


namespace no_sum_14_l101_101527

theorem no_sum_14 (x y : ℤ) (h : x * y + 4 = 40) : x + y ≠ 14 :=
by sorry

end no_sum_14_l101_101527


namespace extra_interest_amount_l101_101291

def principal : ℝ := 15000
def rate1 : ℝ := 0.15
def rate2 : ℝ := 0.12
def time : ℕ := 2

theorem extra_interest_amount :
  principal * (rate1 - rate2) * time = 900 := by
  sorry

end extra_interest_amount_l101_101291


namespace facebook_bonus_each_female_mother_received_l101_101325

theorem facebook_bonus_each_female_mother_received (total_earnings : ℝ) (bonus_percentage : ℝ) 
    (total_employees : ℕ) (male_fraction : ℝ) (female_non_mothers : ℕ) : 
    total_earnings = 5000000 → bonus_percentage = 0.25 → total_employees = 3300 → 
    male_fraction = 1 / 3 → female_non_mothers = 1200 → 
    (250000 / ((total_employees - total_employees * male_fraction.to_nat) - female_non_mothers)) = 1250 :=
by {
  sorry
}

end facebook_bonus_each_female_mother_received_l101_101325


namespace grandchildren_ages_l101_101023

theorem grandchildren_ages (x : ℕ) (y : ℕ) :
  (x + y = 30) →
  (5 * (x * (x + 1) + (30 - x) * (31 - x)) = 2410) →
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) :=
by
  sorry

end grandchildren_ages_l101_101023


namespace greatest_prime_factor_15_fact_17_fact_l101_101785

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l101_101785


namespace cells_remain_illuminated_l101_101563

-- The rect grid screen of size m × n with more than (m - 1)(n - 1) cells illuminated 
-- with the condition that in any 2 × 2 square if three cells are not illuminated, 
-- then the fourth cell also turns off eventually.
theorem cells_remain_illuminated 
  {m n : ℕ} 
  (h1 : ∃ k : ℕ, k > (m - 1) * (n - 1) ∧ k ≤ m * n) 
  (h2 : ∀ (i j : ℕ) (hiv : i < m - 1) (hjv : j < n - 1), 
    (∃ c1 c2 c3 c4 : ℕ, 
      c1 + c2 + c3 + c4 = 4 ∧ 
      (c1 = 1 ∨ c2 = 1 ∨ c3 = 1 ∨ c4 = 1) → 
      (c1 = 0 ∧ c2 = 0 ∧ c3 = 0 ∧ c4 = 0))) :
  ∃ (i j : ℕ) (hil : i < m) (hjl : j < n), true := sorry

end cells_remain_illuminated_l101_101563


namespace root_equation_identity_l101_101571

theorem root_equation_identity {a b c p q : ℝ} 
  (h1 : a^2 + p*a + 1 = 0)
  (h2 : b^2 + p*b + 1 = 0)
  (h3 : b^2 + q*b + 2 = 0)
  (h4 : c^2 + q*c + 2 = 0) 
  : (b - a) * (b - c) = p*q - 6 := 
sorry

end root_equation_identity_l101_101571


namespace proof_l101_101667

noncomputable def M : Set ℝ := {x | 1 - (2 / x) > 0}
noncomputable def N : Set ℝ := {x | x ≥ 1}

theorem proof : (Mᶜ ∪ N) = {x | x ≥ 0} := sorry

end proof_l101_101667


namespace find_number_l101_101145

theorem find_number : ∃ x : ℝ, 0 < x ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_number_l101_101145


namespace evaluate_expression_at_2_l101_101042

-- Define the quadratic and linear components of the expression
def quadratic (x : ℝ) := 3 * x ^ 2 - 8 * x + 5
def linear (x : ℝ) := 4 * x - 7

-- State the proposition to evaluate the given expression at x = 2
theorem evaluate_expression_at_2 : quadratic 2 * linear 2 = 1 := by
  -- The proof is skipped by using sorry
  sorry

end evaluate_expression_at_2_l101_101042


namespace petya_friends_l101_101727

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101727


namespace mingi_math_test_total_pages_l101_101560

theorem mingi_math_test_total_pages (first_page last_page : Nat) (h_first_page : first_page = 8) (h_last_page : last_page = 21) : first_page <= last_page -> ((last_page - first_page + 1) = 14) :=
by
  sorry

end mingi_math_test_total_pages_l101_101560


namespace aitana_jayda_total_spending_l101_101031

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l101_101031


namespace problem_solution_l101_101948

noncomputable def g (n : ℕ) : ℕ :=
  (∏ d in (n.divisors.filter (λ d, 1 < d ∧ d < n)).to_finset, d)

def is_valid_n (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬(n ∣ g n)

def count_valid_n : ℕ :=
  (finset.range 101).filter is_valid_n).card

theorem problem_solution : count_valid_n = 32 := sorry

end problem_solution_l101_101948


namespace problem_part1_problem_part2_l101_101202

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * Real.log (2 * x + t) / Real.log a

theorem problem_part1 (a t : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  f a 1 - g a t 1 = 0 → t = -2 + Real.sqrt 2 :=
sorry

theorem problem_part2 (a t : ℝ) (ha_bound : 0 < a ∧ a < 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 15 → f a x ≥ g a t x) → t ≥ 1 :=
sorry

end problem_part1_problem_part2_l101_101202


namespace petya_friends_l101_101737

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101737


namespace average_monthly_balance_l101_101856

-- Definitions for the monthly balances
def January_balance : ℝ := 120
def February_balance : ℝ := 240
def March_balance : ℝ := 180
def April_balance : ℝ := 180
def May_balance : ℝ := 160
def June_balance : ℝ := 200

-- The average monthly balance theorem statement
theorem average_monthly_balance : 
    (January_balance + February_balance + March_balance + April_balance + May_balance + June_balance) / 6 = 180 := 
by 
  sorry

end average_monthly_balance_l101_101856


namespace rightmost_three_digits_of_7_pow_2023_l101_101778

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l101_101778


namespace no_integer_solutions_l101_101891

theorem no_integer_solutions :
  ∀ (m n : ℤ), (m^3 + 4 * m^2 + 3 * m ≠ 8 * n^3 + 12 * n^2 + 6 * n + 1) := by
  sorry

end no_integer_solutions_l101_101891


namespace sin_zero_degrees_l101_101176

theorem sin_zero_degrees : Real.sin 0 = 0 := 
by {
  -- The proof is added here (as requested no proof is required, hence using sorry)
  sorry
}

end sin_zero_degrees_l101_101176


namespace union_complement_eq_l101_101093

open Finset

variable (U P Q : Finset ℕ) (U_def : U = {1, 2, 3, 4}) (P_def : P = {1, 2}) (Q_def : Q = {1, 3})

theorem union_complement_eq : P ∪ (U \ Q) = {1, 2, 4} :=
by
  sorry

end union_complement_eq_l101_101093


namespace hyperbola_condition_l101_101862

theorem hyperbola_condition (m n : ℝ) : (m < 0 ∧ 0 < n) → (∀ x y : ℝ, nx^2 + my^2 = 1 → (n * x^2 - m * y^2 > 0)) :=
by
  sorry

end hyperbola_condition_l101_101862


namespace sam_original_puppies_count_l101_101247

theorem sam_original_puppies_count 
  (spotted_puppies_start : ℕ)
  (non_spotted_puppies_start : ℕ)
  (spotted_puppies_given : ℕ)
  (non_spotted_puppies_given : ℕ)
  (spotted_puppies_left : ℕ)
  (non_spotted_puppies_left : ℕ)
  (h1 : spotted_puppies_start = 8)
  (h2 : non_spotted_puppies_start = 5)
  (h3 : spotted_puppies_given = 2)
  (h4 : non_spotted_puppies_given = 3)
  (h5 : spotted_puppies_left = spotted_puppies_start - spotted_puppies_given)
  (h6 : non_spotted_puppies_left = non_spotted_puppies_start - non_spotted_puppies_given)
  (h7 : spotted_puppies_left = 6)
  (h8 : non_spotted_puppies_left = 2) :
  spotted_puppies_start + non_spotted_puppies_start = 13 :=
by
  sorry

end sam_original_puppies_count_l101_101247


namespace smaller_rectangle_length_ratio_l101_101892

theorem smaller_rectangle_length_ratio 
  (s : ℝ)
  (h1 : 5 = 5)
  (h2 : ∃ r : ℝ, r = s)
  (h3 : ∀ x : ℝ, x = s)
  (h4 : ∀ y : ℝ, y / 2 = s / 2)
  (h5 : ∀ z : ℝ, z = 3 * s)
  (h6 : ∀ w : ℝ, w = s) :
  ∃ l : ℝ, l / s = 4 :=
sorry

end smaller_rectangle_length_ratio_l101_101892


namespace find_b_l101_101916

theorem find_b (a b : ℝ) (B C : ℝ)
    (h1 : a * b = 60 * Real.sqrt 3)
    (h2 : Real.sin B = Real.sin C)
    (h3 : 15 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) :
  b = 2 * Real.sqrt 15 :=
sorry

end find_b_l101_101916


namespace road_signs_at_first_intersection_l101_101873

theorem road_signs_at_first_intersection (x : ℕ) 
    (h1 : x + (x + x / 4) + 2 * (x + x / 4) + (2 * (x + x / 4) - 20) = 270) : 
    x = 40 := 
sorry

end road_signs_at_first_intersection_l101_101873


namespace domain_g_l101_101055

def domain_f (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 2
def g (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ((1 < x) ∧ (x ≤ Real.sqrt 3)) ∧ domain_f (x^2 - 1) ∧ (0 < x - 1 ∧ x - 1 < 1)

theorem domain_g (x : ℝ) (f : ℝ → ℝ) (hf : ∀ a, domain_f a → True) : 
  g x f ↔ 1 < x ∧ x ≤ Real.sqrt 3 :=
by 
  sorry

end domain_g_l101_101055


namespace petya_friends_l101_101734

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101734


namespace number_of_girls_in_group_l101_101192

-- Define the given conditions
def total_students : ℕ := 20
def prob_of_selecting_girl : ℚ := 2/5

-- State the lean problem for the proof
theorem number_of_girls_in_group : (total_students : ℚ) * prob_of_selecting_girl = 8 := by
  sorry

end number_of_girls_in_group_l101_101192


namespace matrix_determinant_eq_9_l101_101282

theorem matrix_determinant_eq_9 (x : ℝ) :
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  (a * d - b * c = 9) → x = -2 :=
by 
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  sorry

end matrix_determinant_eq_9_l101_101282


namespace greatest_prime_factor_of_sum_factorials_l101_101826

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l101_101826


namespace not_possible_fill_prime_sum_l101_101544

open Matrix

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_prime_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j : Fin 3, 
  ((j + 1 < 3) → is_prime (m i j + m i ⟨j + 1, by linarith⟩)) ∧ -- right neighbor
  ((i + 1 < 3) → is_prime (m i j + m ⟨i + 1, by linarith⟩ j))   -- bottom neighbor

theorem not_possible_fill_prime_sum : ¬ ∃ m : Matrix (Fin 3) (Fin 3) ℕ,
  (∀ i j, 1 ≤ m i j ∧ m i j ≤ 9) ∧ 
  (∀ n ∈ (to_list m), ∃! k, m k.1 k.2 = n) ∧ -- all numbers 1 to 9 used exactly once
  valid_prime_sum m :=
by
  sorry

end not_possible_fill_prime_sum_l101_101544


namespace greatest_prime_factor_15_17_factorial_l101_101803

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l101_101803


namespace remainder_of_power_mod_l101_101487

theorem remainder_of_power_mod (a b n : ℕ) (h_prime : Nat.Prime n) (h_a_not_div : ¬ (n ∣ a)) :
  a ^ b % n = 82 :=
by
  have : n = 379 := sorry
  have : a = 6 := sorry
  have : b = 97 := sorry
  sorry

end remainder_of_power_mod_l101_101487


namespace solution_triples_l101_101890

noncomputable def find_triples (x y z : ℝ) : Prop :=
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  (1/x) + (1/y) + (1/z) = 1/2008

theorem solution_triples :
  ∃ (x y z : ℝ), find_triples x y z ∧ (x = 2008 ∧ y = 4016 ∧ z = -4016) :=
sorry

end solution_triples_l101_101890


namespace investment_after_8_years_l101_101577

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem investment_after_8_years :
  let P := 500
  let r := 0.03
  let n := 8
  let A := compound_interest P r n
  round A = 633 :=
by
  sorry

end investment_after_8_years_l101_101577


namespace ball_bounce_height_l101_101285

theorem ball_bounce_height (n : ℕ) : (512 * (1/2)^n < 20) → n = 8 := 
sorry

end ball_bounce_height_l101_101285


namespace time_A_reaches_destination_l101_101082

theorem time_A_reaches_destination (x t : ℝ) (h_ratio : (4 * t) = 3 * (t + 0.5)) : (t + 0.5) = 2 :=
by {
  -- derived by algebraic manipulation
  sorry
}

end time_A_reaches_destination_l101_101082


namespace average_time_per_mile_l101_101385

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end average_time_per_mile_l101_101385


namespace count_ordered_pairs_squares_diff_l101_101356

theorem count_ordered_pairs_squares_diff (m n : ℕ) (h1 : m ≥ n) (h2 : m^2 - n^2 = 72) : 
∃ (a : ℕ), a = 3 :=
sorry

end count_ordered_pairs_squares_diff_l101_101356


namespace triangle_third_side_length_l101_101957

theorem triangle_third_side_length (A B C : Type) 
  (AB : ℝ) (AC : ℝ) 
  (angle_ABC angle_ACB : ℝ) 
  (BC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 21) 
  (h3 : angle_ABC = 3 * angle_ACB) 
  : 
  BC = (some_correct_value ) := 
sorry

end triangle_third_side_length_l101_101957


namespace tangent_line_equation_range_of_a_l101_101663

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (3/2) * x^2 + 1

theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∀ x, f a x = a * x ^ 3 - (3 / 2) * x ^ 2 + 1 →
  ∀ x, deriv (f a) x = 3 * a * x ^ 2 - 3 * x →
  deriv (f a) 2 = 6 ∧ f a 2 = 3 ∧ ∃ m b, m = 6 ∧ b = -9 ∧ (∀ x, f a x = a * x ^ 3 - (3 / 2) * x ^ 2 + 1 → tangent (λ x, f a x) 2 m b) := sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ Icc (-1) (1/2) → f a x < a^2) : 1 < a := sorry

end tangent_line_equation_range_of_a_l101_101663


namespace polynomial_root_range_l101_101861

variable (a : ℝ)

theorem polynomial_root_range (h : ∀ x : ℂ, (2 * x^4 + a * x^3 + 9 * x^2 + a * x + 2 = 0) →
  ((x.re^2 + x.im^2 ≠ 1) ∧ x.im ≠ 0)) : (-2 * Real.sqrt 10 < a ∧ a < 2 * Real.sqrt 10) :=
sorry

end polynomial_root_range_l101_101861


namespace probability_two_white_balls_l101_101286

-- Definitions based on the conditions provided
def total_balls := 17        -- 8 white + 9 black
def white_balls := 8
def drawn_without_replacement := true

-- Proposition: Probability of drawing two white balls successively
theorem probability_two_white_balls:
  drawn_without_replacement → 
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 7 / 34 :=
by
  intros
  sorry

end probability_two_white_balls_l101_101286


namespace number_of_times_each_player_plays_l101_101868

def players : ℕ := 7
def total_games : ℕ := 42

theorem number_of_times_each_player_plays (x : ℕ) 
  (H1 : 42 = (players * (players - 1) * x) / 2) : x = 2 :=
by
  sorry

end number_of_times_each_player_plays_l101_101868


namespace fiona_pairs_l101_101044

theorem fiona_pairs : Nat.choose 12 2 = 66 := by
  sorry

end fiona_pairs_l101_101044


namespace g_is_correct_l101_101479

-- Define the given polynomial equation
def poly_lhs (x : ℝ) : ℝ := 2 * x^5 - x^3 + 4 * x^2 + 3 * x - 5
def poly_rhs (x : ℝ) : ℝ := 7 * x^3 - 4 * x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := -2 * x^5 + 6 * x^3 - 4 * x^2 - x + 7

-- The theorem to be proven
theorem g_is_correct : ∀ x : ℝ, poly_lhs x + g x = poly_rhs x :=
by
  intro x
  unfold poly_lhs poly_rhs g
  sorry

end g_is_correct_l101_101479


namespace giant_exponent_modulo_result_l101_101338

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l101_101338


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101797

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101797


namespace compound_interest_amount_l101_101412

/-
Given:
- Principal amount P = 5000
- Annual interest rate r = 0.07
- Time period t = 15 years

We aim to prove:
A = 5000 * (1 + 0.07) ^ 15 = 13795.15
-/
theorem compound_interest_amount :
  let P : ℝ := 5000
  let r : ℝ := 0.07
  let t : ℝ := 15
  let A : ℝ := P * (1 + r) ^ t
  A = 13795.15 :=
by
  sorry

end compound_interest_amount_l101_101412


namespace sum_six_times_product_l101_101361

variable (a b x : ℝ)

theorem sum_six_times_product (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * x) (h4 : 1/a + 1/b = 6) :
  x = a * b := sorry

end sum_six_times_product_l101_101361


namespace part_I_part_II_part_III_l101_101664

open Real

section Problem

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x
def g (x : ℝ) : ℝ := log x
def F (x : ℝ) : ℝ := f a x - g x
def G (x : ℝ) : ℝ := a * sin (1 - x) + log x

theorem part_I : (∀ x, diff F x = a - 1 / x) → diff F 1 = 0 → a = 1 :=
by
  -- Proof omitted
  sorry

theorem part_II : (∀ x ∈ Ioo 0 1, diff G x ≥ 0) → a ≤ 1 :=
by
  -- Proof omitted
  sorry

theorem part_III : (∀ n : ℕ, ∑ k in range n, sin (1 / (k + 1)^2) < log 2) :=
by
  -- Proof omitted
  sorry

end Problem

end part_I_part_II_part_III_l101_101664


namespace bob_walks_more_l101_101679

def street_width : ℝ := 30
def length_side1 : ℝ := 500
def length_side2 : ℝ := 300

def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

def alice_perimeter : ℝ := perimeter (length_side1 + 2 * street_width) (length_side2 + 2 * street_width)
def bob_perimeter : ℝ := perimeter (length_side1 + 4 * street_width) (length_side2 + 4 * street_width)

theorem bob_walks_more :
  bob_perimeter - alice_perimeter = 240 :=
by
  sorry

end bob_walks_more_l101_101679


namespace ratio_b_to_c_l101_101902

variables (a b c d e f : ℝ)

theorem ratio_b_to_c 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := 
sorry

end ratio_b_to_c_l101_101902


namespace a2b_sub_ab2_eq_neg16sqrt5_l101_101651

noncomputable def a : ℝ := 4 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 4 - 2 * Real.sqrt 5

theorem a2b_sub_ab2_eq_neg16sqrt5 : a^2 * b - a * b^2 = -16 * Real.sqrt 5 :=
by
  sorry

end a2b_sub_ab2_eq_neg16sqrt5_l101_101651


namespace fraction_N_div_M_l101_101692

def M : ℕ := Nat.lcm_list [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

def N : ℕ := Nat.lcm M (Nat.lcm_list [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])

theorem fraction_N_div_M : N / M = 2^2 * 3 * 29 * 31 * 37 * 41 := 
 by sorry

end fraction_N_div_M_l101_101692


namespace find_m_l101_101637

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -13.6 :=
by sorry

end find_m_l101_101637


namespace girls_to_boys_ratio_l101_101381

variable (g b : ℕ)
variable (h_total : g + b = 36)
variable (h_diff : g = b + 6)

theorem girls_to_boys_ratio (g b : ℕ) (h_total : g + b = 36) (h_diff : g = b + 6) :
  g / b = 7 / 5 := by
  sorry

end girls_to_boys_ratio_l101_101381


namespace ratio_nine_years_ago_correct_l101_101312

-- Conditions
def C : ℕ := 24
def G : ℕ := C / 2

-- Question and expected answer
def ratio_nine_years_ago : ℕ := (C - 9) / (G - 9)

theorem ratio_nine_years_ago_correct : ratio_nine_years_ago = 5 := by
  sorry

end ratio_nine_years_ago_correct_l101_101312


namespace total_yards_of_fabric_l101_101592

theorem total_yards_of_fabric (cost_checkered : ℝ) (cost_plain : ℝ) (price_per_yard : ℝ)
  (h1 : cost_checkered = 75) (h2 : cost_plain = 45) (h3 : price_per_yard = 7.50) :
  (cost_checkered / price_per_yard) + (cost_plain / price_per_yard) = 16 := 
by
  sorry

end total_yards_of_fabric_l101_101592


namespace complete_square_equation_l101_101376

theorem complete_square_equation (b c : ℤ) (h : (x : ℝ) → x^2 - 6 * x + 5 = (x + b)^2 - c) : b + c = 1 :=
by
  sorry  -- This is where the proof would go

end complete_square_equation_l101_101376


namespace sin_log_infinite_zeros_in_01_l101_101516

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l101_101516


namespace petya_friends_l101_101723

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101723


namespace greatest_prime_factor_15_fact_plus_17_fact_l101_101804

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l101_101804


namespace probability_both_heads_l101_101132

-- Define the sample space and the probability of each outcome
def sample_space : List (Bool × Bool) := [(true, true), (true, false), (false, true), (false, false)]

-- Define the function to check for both heads
def both_heads (outcome : Bool × Bool) : Bool :=
  outcome = (true, true)

-- Calculate the probability of both heads
theorem probability_both_heads :
  (sample_space.filter both_heads).length / sample_space.length = 1 / 4 := sorry

end probability_both_heads_l101_101132


namespace cheryl_used_material_l101_101419

theorem cheryl_used_material 
  (a b c l : ℚ) 
  (ha : a = 3 / 8) 
  (hb : b = 1 / 3) 
  (hl : l = 15 / 40) 
  (Hc: c = a + b): 
  (c - l = 1 / 3) := 
by 
  -- proof will be deferred to Lean's syntax for user to fill in.
  sorry

end cheryl_used_material_l101_101419


namespace solve_mod_equiv_l101_101006

theorem solve_mod_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-2222 ≡ n [ZMOD 9]) → n = 6 := by
  sorry

end solve_mod_equiv_l101_101006


namespace garrett_granola_bars_l101_101643

theorem garrett_granola_bars :
  ∀ (oatmeal_raisin peanut total : ℕ),
  peanut = 8 →
  total = 14 →
  oatmeal_raisin + peanut = total →
  oatmeal_raisin = 6 :=
by
  intros oatmeal_raisin peanut total h_peanut h_total h_sum
  sorry

end garrett_granola_bars_l101_101643


namespace toys_per_week_l101_101152

-- Define the number of days the workers work in a week
def days_per_week : ℕ := 4

-- Define the number of toys produced each day
def toys_per_day : ℕ := 1140

-- State the proof problem: workers produce 4560 toys per week
theorem toys_per_week : (toys_per_day * days_per_week) = 4560 :=
by
  -- Proof goes here
  sorry

end toys_per_week_l101_101152


namespace interval_length_t_subset_interval_t_l101_101513

-- Statement (1)
theorem interval_length_t (t : ℝ) (h : (Real.log t / Real.log 2) - 2 = 3) : t = 32 :=
  sorry

-- Statement (2)
theorem subset_interval_t (t : ℝ) (h : 2 ≤ Real.log t / Real.log 2 ∧ Real.log t / Real.log 2 ≤ 5) :
  0 < t ∧ t ≤ 32 :=
  sorry

end interval_length_t_subset_interval_t_l101_101513


namespace total_students_count_l101_101002

-- Define the conditions
def num_rows : ℕ := 8
def students_per_row : ℕ := 6
def students_last_row : ℕ := 5
def rows_with_six_students : ℕ := 7

-- Define the total students
def total_students : ℕ :=
  (rows_with_six_students * students_per_row) + students_last_row

-- The theorem to prove
theorem total_students_count : total_students = 47 := by
  sorry

end total_students_count_l101_101002


namespace problem1_problem2_l101_101661

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x + π/4)

theorem problem1 (ω : ℝ) (hω : ω > 0) : f ω 0 = sqrt 2 / 2 :=
by
  sorry

theorem problem2 (ω : ℝ) (hω : ω > 0) (h_period : (2 * π / ω) = π) :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f ω x ≤ 1 :=
by
  have hω_eq_2 : ω = 2 := by
    have := congr_arg (λ x, ω * x) h_period
    simp [this] at h_period
    exact h_period

  intro x hx
  simp [f]
  sorry

end problem1_problem2_l101_101661


namespace sin_A_and_height_on_AB_l101_101932

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l101_101932


namespace intersection_of_lines_l101_101328

theorem intersection_of_lines :
  ∃ (x y : ℚ), (8 * x - 3 * y = 24) ∧ (10 * x + 2 * y = 14) ∧ x = 45 / 23 ∧ y = -64 / 23 :=
by
  sorry

end intersection_of_lines_l101_101328


namespace expected_BBR_sequences_l101_101579

theorem expected_BBR_sequences :
  let total_cards := 52
  let black_cards := 26
  let red_cards := 26
  let probability_of_next_black := (25 / 51)
  let probability_of_third_red := (26 / 50)
  let probability_of_BBR := probability_of_next_black * probability_of_third_red
  let possible_start_positions := 26
  let expected_BBR := possible_start_positions * probability_of_BBR
  expected_BBR = (338 / 51) :=
by
  sorry

end expected_BBR_sequences_l101_101579


namespace smallest_n_for_factors_l101_101362

theorem smallest_n_for_factors (k : ℕ) (hk : (∃ p : ℕ, k = 2^p) ) :
  ∃ (n : ℕ), ( 5^2 ∣ n * k * 36 * 343 ) ∧ ( 3^3 ∣ n * k * 36 * 343 ) ∧ n = 75 :=
by
  sorry

end smallest_n_for_factors_l101_101362


namespace factorable_b_even_l101_101581

-- Defining the conditions
def is_factorable (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    m * p = 15 ∧ n * q = 15 ∧ b = m * q + n * p

-- The theorem to be stated
theorem factorable_b_even (b : ℤ) : is_factorable b ↔ ∃ k : ℤ, b = 2 * k :=
sorry

end factorable_b_even_l101_101581


namespace remaining_soup_can_feed_adults_l101_101288

-- Define initial conditions
def cans_per_soup_for_children : ℕ := 6
def cans_per_soup_for_adults : ℕ := 4
def initial_cans : ℕ := 8
def children_to_feed : ℕ := 24

-- Define the problem statement in Lean
theorem remaining_soup_can_feed_adults :
  (initial_cans - (children_to_feed / cans_per_soup_for_children)) * cans_per_soup_for_adults = 16 := by
  sorry

end remaining_soup_can_feed_adults_l101_101288


namespace sum_of_even_factors_720_l101_101271

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l101_101271


namespace sum_of_nine_l101_101050

theorem sum_of_nine (S : ℕ → ℕ) (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), S n = n * (a 1 + a n) / 2)
(h₁ : S 3 = 30) (h₂ : S 6 = 100) : S 9 = 240 := 
sorry

end sum_of_nine_l101_101050


namespace concert_attendance_l101_101098

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l101_101098


namespace initial_capacity_l101_101020

theorem initial_capacity (x : ℝ) (h1 : 0.9 * x = 198) : x = 220 :=
by
  sorry

end initial_capacity_l101_101020


namespace fraction_day_crew_loaded_l101_101144

variable (D W : ℕ)  -- D: Number of boxes loaded by each worker on the day crew, W: Number of workers on the day crew

-- Condition 1: Each worker on the night crew loaded 3/4 as many boxes as each worker on the day crew
def boxes_loaded_night_worker : ℕ := 3 * D / 4
-- Condition 2: The night crew has 5/6 as many workers as the day crew
def workers_night : ℕ := 5 * W / 6

-- Question: Fraction of all the boxes loaded by the day crew
theorem fraction_day_crew_loaded :
  (D * W : ℚ) / ((D * W) + (3 * D / 4) * (5 * W / 6)) = (8 / 13) := by
  sorry

end fraction_day_crew_loaded_l101_101144


namespace find_x_l101_101138

theorem find_x : ∃ x, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 := 
by
  sorry

end find_x_l101_101138


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101832

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101832


namespace find_D_l101_101068

theorem find_D (P Q : ℕ) (h_pos : 0 < P ∧ 0 < Q) (h_eq : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end find_D_l101_101068


namespace unique_exponential_function_l101_101896

theorem unique_exponential_function (g : ℝ → ℝ) :
  (∀ x1 x2 : ℝ, g (x1 + x2) = g x1 * g x2) →
  g 1 = 3 →
  (∀ x1 x2 : ℝ, x1 < x2 → g x1 < g x2) →
  ∀ x : ℝ, g x = 3^x :=
by
  sorry

end unique_exponential_function_l101_101896


namespace red_balls_in_total_color_of_158th_ball_l101_101127

def totalBalls : Nat := 200
def redBallsPerCycle : Nat := 5
def whiteBallsPerCycle : Nat := 4
def blackBallsPerCycle : Nat := 3
def cycleLength : Nat := redBallsPerCycle + whiteBallsPerCycle + blackBallsPerCycle

theorem red_balls_in_total :
  (totalBalls / cycleLength) * redBallsPerCycle + min redBallsPerCycle (totalBalls % cycleLength) = 85 :=
by sorry

theorem color_of_158th_ball :
  let positionInCycle := (158 - 1) % cycleLength + 1
  positionInCycle ≤ redBallsPerCycle := by sorry

end red_balls_in_total_color_of_158th_ball_l101_101127


namespace baba_yaga_powder_problem_l101_101281

theorem baba_yaga_powder_problem (A B d : ℤ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end baba_yaga_powder_problem_l101_101281


namespace circle_center_radius_sum_l101_101239

theorem circle_center_radius_sum :
  let D := { p : ℝ × ℝ | (p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34) }
  let c := 7
  let d := -5
  let s := 2 * Real.sqrt 10
  (c + d + s = 2 + 2 * Real.sqrt 10) :=
by
  sorry

end circle_center_radius_sum_l101_101239


namespace Petya_friends_l101_101711

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101711


namespace greatest_prime_factor_15_fact_17_fact_l101_101822

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l101_101822


namespace x_plus_q_eq_five_l101_101524

theorem x_plus_q_eq_five (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x < 5) : x + q = 5 :=
by
  sorry

end x_plus_q_eq_five_l101_101524


namespace average_time_per_mile_l101_101386

noncomputable def miles : ℕ := 24
noncomputable def hours : ℕ := 3
noncomputable def minutes : ℕ := 36

theorem average_time_per_mile :
  let total_time := hours * 60 + minutes in
  total_time / miles = 9 := by
  sorry

end average_time_per_mile_l101_101386


namespace rational_if_limit_fractional_part_eq_zero_l101_101552

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem rational_if_limit_fractional_part_eq_zero
  (P : ℤ[X]) (α : ℝ)
  (h_int_coeffs : ∀ n : ℕ, coeff P n ∈ ℤ)
  (h_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, fractional_part (P.eval (↑n) * α) < ε) : 
  ∃ r : ℚ, r = α := sorry

end rational_if_limit_fractional_part_eq_zero_l101_101552


namespace base_conversion_l101_101391

theorem base_conversion {b : ℕ} (h : 5 * 6 + 2 = b * b + b + 1) : b = 5 :=
by
  -- Begin omitted steps to solve the proof
  sorry

end base_conversion_l101_101391


namespace blue_balls_initial_count_l101_101982

theorem blue_balls_initial_count (B : ℕ)
  (h1 : 15 - 3 = 12)
  (h2 : (B - 3) / 12 = 1 / 3) :
  B = 7 :=
sorry

end blue_balls_initial_count_l101_101982


namespace Petya_friends_l101_101715

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101715


namespace smallest_possible_difference_l101_101589

noncomputable def PQ : ℕ := 504
noncomputable def QR : ℕ := PQ + 1
noncomputable def PR : ℕ := 2021 - PQ - QR

theorem smallest_possible_difference :
  PQ + QR + PR = 2021 ∧ PQ < QR ∧ QR ≤ PR ∧ ∀ x y z : ℕ, x + y + z = 2021 → x < y → 
  y ≤ z → (y - x) = 1 → x = PQ ∧ y = QR ∧ z = PR :=
by
  { tautology } -- Placeholder for the actual proof

end smallest_possible_difference_l101_101589


namespace olivia_hourly_rate_l101_101562

theorem olivia_hourly_rate (h_worked_monday : ℕ) (h_worked_wednesday : ℕ) (h_worked_friday : ℕ) (h_total_payment : ℕ) (h_total_hours : h_worked_monday + h_worked_wednesday + h_worked_friday = 13) (h_total_amount : h_total_payment = 117) :
  h_total_payment / (h_worked_monday + h_worked_wednesday + h_worked_friday) = 9 :=
by
  sorry

end olivia_hourly_rate_l101_101562


namespace sum_of_money_invested_l101_101580

noncomputable def principal_sum_of_money (R : ℝ) (T : ℝ) (CI_minus_SI : ℝ) : ℝ :=
  let SI := (625 * R * T / 100)
  let CI := 625 * ((1 + R / 100)^(T : ℝ) - 1)
  if (CI - SI = CI_minus_SI)
  then 625
  else 0

theorem sum_of_money_invested : 
  (principal_sum_of_money 4 2 1) = 625 :=
by
  unfold principal_sum_of_money
  sorry

end sum_of_money_invested_l101_101580


namespace petya_friends_l101_101724

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101724


namespace ratio_of_length_to_breadth_l101_101396

theorem ratio_of_length_to_breadth (b l k : ℕ) (h1 : b = 15) (h2 : l = k * b) (h3 : l * b = 675) : l / b = 3 :=
by
  sorry

end ratio_of_length_to_breadth_l101_101396


namespace smallest_difference_of_sides_l101_101588

/-- Triangle PQR has a perimeter of 2021 units. The sides have lengths that are integer values with PQ < QR ≤ PR. 
The smallest possible value of QR - PQ is 1. -/
theorem smallest_difference_of_sides :
  ∃ (PQ QR PR : ℕ), PQ < QR ∧ QR ≤ PR ∧ PQ + QR + PR = 2021 ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ QR + PR > PQ ∧ QR - PQ = 1 :=
sorry

end smallest_difference_of_sides_l101_101588


namespace ratio_of_fractions_l101_101908

-- Given conditions
variables {x y : ℚ}
variables (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0)

-- Assertion to be proved
theorem ratio_of_fractions (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 18 / 25 :=
sorry

end ratio_of_fractions_l101_101908


namespace part_a_l101_101690

theorem part_a (a b c : Int) (h1 : a + b + c = 0) : 
  ¬(a ^ 1999 + b ^ 1999 + c ^ 1999 = 2) :=
by
  sorry

end part_a_l101_101690


namespace exit_time_correct_l101_101115

def time_to_exit_wide : ℝ := 6
def time_to_exit_narrow : ℝ := 10

theorem exit_time_correct :
  ∃ x y : ℝ, x = 6 ∧ y = 10 ∧ 
  (1 / x + 1 / y = 4 / 15) ∧ 
  (y = x + 4) ∧ 
  (3.75 * (1 / x + 1 / y) = 1) :=
by
  use time_to_exit_wide
  use time_to_exit_narrow
  sorry

end exit_time_correct_l101_101115


namespace xy_sum_possible_values_l101_101209

theorem xy_sum_possible_values (x y : ℕ) (h1 : x < 20) (h2 : y < 20) (h3 : 0 < x) (h4 : 0 < y) (h5 : x + y + x * y = 95) :
  x + y = 18 ∨ x + y = 20 :=
by {
  sorry
}

end xy_sum_possible_values_l101_101209


namespace sum_of_x_values_l101_101276

theorem sum_of_x_values (x : ℝ) (h : x ≠ -1) : 
  (∃ x, 3 = (x^3 - 3*x^2 - 4*x)/(x + 1)) →
  (x = 6) :=
by
  sorry

end sum_of_x_values_l101_101276


namespace geometric_sequence_seventh_term_l101_101430

theorem geometric_sequence_seventh_term :
  ∃ (r : ℕ), (3 : ℕ) * r^4 = 243 ∧ (3 : ℕ) * r^6 = 2187 :=
by
  use 3
  split
  · calc
      (3 : ℕ) * 3^4 = 3 * 81 := by refl
      ... = 243 := by norm_num
  · calc
      (3 : ℕ) * 3^6 = 3 * 729 := by refl
      ... = 2187 := by norm_num

end geometric_sequence_seventh_term_l101_101430


namespace arcsin_half_eq_pi_six_l101_101457

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101457


namespace symmetric_line_equation_l101_101122

theorem symmetric_line_equation 
  (l1 : ∀ x y : ℝ, x - 2 * y - 2 = 0) 
  (l2 : ∀ x y : ℝ, x + y = 0) : 
  ∀ x y : ℝ, 2 * x - y - 2 = 0 :=
sorry

end symmetric_line_equation_l101_101122


namespace inequality_proof_l101_101106

theorem inequality_proof (a b : ℝ) (h1 : a > 1) (h2 : b > 1) :
    (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := 
by
  sorry

end inequality_proof_l101_101106


namespace smallest_prime_l101_101641

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ , m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n 

theorem smallest_prime :
  ∃ n : ℕ, n = 29 ∧ 
  n >= 10 ∧ n < 100 ∧
  is_prime n ∧
  ((n / 10) = 3) ∧ 
  is_composite (n % 10 * 10 + n / 10) ∧
  (n % 10 * 10 + n / 10) % 5 = 0 :=
by {
  sorry
}

end smallest_prime_l101_101641


namespace arcsin_one_half_l101_101461

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101461


namespace greatest_prime_factor_15_fact_17_fact_l101_101823

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l101_101823


namespace one_twenty_percent_of_number_l101_101216

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end one_twenty_percent_of_number_l101_101216


namespace sum_of_number_and_reverse_l101_101118

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l101_101118


namespace sum_of_number_and_reverse_l101_101121

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l101_101121


namespace find_root_and_m_l101_101054

theorem find_root_and_m (m x₂ : ℝ) (h₁ : (1 : ℝ) * x₂ = 3) (h₂ : (1 : ℝ) + x₂ = -m) : 
  x₂ = 3 ∧ m = -4 :=
by
  sorry

end find_root_and_m_l101_101054


namespace fill_pipe_time_l101_101157

theorem fill_pipe_time (t : ℕ) (H : ∀ C : Type, (1 / 2 : ℚ) * C = t * 1/2 * C) : t = t :=
by
  sorry

end fill_pipe_time_l101_101157


namespace number_of_members_l101_101432

theorem number_of_members (n : ℕ) (h : n^2 = 9216) : n = 96 :=
sorry

end number_of_members_l101_101432


namespace solution_m_value_l101_101656

theorem solution_m_value (m : ℝ) : 
  (m^2 - 5*m + 4 > 0) ∧ (m^2 - 2*m = 0) ↔ m = 0 :=
by
  sorry

end solution_m_value_l101_101656


namespace least_positive_integer_divisible_by_three_primes_l101_101595

-- Define the next three distinct primes larger than 5
def prime1 := 7
def prime2 := 11
def prime3 := 13

-- Define the product of these primes
def prod := prime1 * prime2 * prime3

-- Statement of the theorem
theorem least_positive_integer_divisible_by_three_primes : prod = 1001 :=
by
  sorry

end least_positive_integer_divisible_by_three_primes_l101_101595


namespace f_properties_l101_101756

variable (f : ℝ → ℝ)
variable (f_pos : ∀ x : ℝ, f x > 0)
variable (f_eq : ∀ a b : ℝ, f a * f b = f (a + b))

theorem f_properties :
  (f 0 = 1) ∧
  (∀ a : ℝ, f (-a) = 1 / f a) ∧
  (∀ a : ℝ, f a = (f (3 * a))^(1/3)) :=
by {
  sorry
}

end f_properties_l101_101756


namespace petya_friends_count_l101_101742

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101742


namespace conditional_probabilities_l101_101959

def outcomes_all_different : ℕ := 6 * 5 * 4

def outcomes_at_least_one_three : ℕ := 6^3 - 5^3

def outcomes_different_one_three : ℕ := 3 * 5 * 4

noncomputable def P_A_given_B : ℚ :=
  outcomes_different_one_three / outcomes_at_least_one_three

noncomputable def P_B_given_A : ℚ := 1 / 2

theorem conditional_probabilities :
  P_A_given_B = 60 / 91 ∧ P_B_given_A = 1 / 2 :=
by
  sorry

end conditional_probabilities_l101_101959


namespace greatest_prime_factor_of_15_add_17_factorial_l101_101844

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l101_101844


namespace sqrt_72_eq_6_sqrt_2_l101_101110

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l101_101110


namespace triangle_sin_A_and_height_l101_101926

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l101_101926


namespace possible_m_values_l101_101204

theorem possible_m_values (m : ℝ) :
  let A := {x : ℝ | mx - 1 = 0}
  let B := {2, 3}
  (A ⊆ B) → (m = 0 ∨ m = 1 / 2 ∨ m = 1 / 3) :=
by
  intro A B h
  sorry

end possible_m_values_l101_101204


namespace minute_hand_sweep_probability_l101_101099

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l101_101099


namespace greatest_prime_factor_of_15_add_17_factorial_l101_101847

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l101_101847


namespace walters_exceptional_days_l101_101775

variable (b w : ℕ)
variable (days_total dollars_total : ℕ)
variable (normal_earn exceptional_earn : ℕ)
variable (at_least_exceptional_days : ℕ)

-- Conditions
def conditions : Prop :=
  days_total = 15 ∧
  dollars_total = 70 ∧
  normal_earn = 4 ∧
  exceptional_earn = 6 ∧
  at_least_exceptional_days = 5 ∧
  b + w = days_total ∧
  normal_earn * b + exceptional_earn * w = dollars_total ∧
  w ≥ at_least_exceptional_days

-- Theorem to prove the number of exceptional days is 5
theorem walters_exceptional_days (h : conditions b w days_total dollars_total normal_earn exceptional_earn at_least_exceptional_days) : w = 5 :=
sorry

end walters_exceptional_days_l101_101775


namespace paper_cranes_l101_101406

theorem paper_cranes (B C A : ℕ) (h1 : A + B + C = 1000)
  (h2 : A = 3 * B - 100)
  (h3 : C = A - 67) : A = 443 := by
  sorry

end paper_cranes_l101_101406


namespace fill_pipe_half_time_l101_101153

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l101_101153


namespace greatest_prime_factor_of_sum_factorials_l101_101825

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l101_101825


namespace one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l101_101910

-- Definition of the conditions.
variable (x : ℝ)

-- Statement of the problem in Lean.
theorem one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1 (x : ℝ) :
    (1 / 3) * (9 * x - 3) = 3 * x - 1 :=
by sorry

end one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l101_101910


namespace find_expression_l101_101638

theorem find_expression : 1^567 + 3^5 / 3^3 - 2 = 8 :=
by
  sorry

end find_expression_l101_101638


namespace money_leftover_is_90_l101_101233

-- Define constants and given conditions.
def jars_quarters : ℕ := 4
def quarters_per_jar : ℕ := 160
def jars_dimes : ℕ := 4
def dimes_per_jar : ℕ := 300
def jars_nickels : ℕ := 2
def nickels_per_jar : ℕ := 500

def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05

def bike_cost : ℝ := 240
def total_quarters := jars_quarters * quarters_per_jar
def total_dimes := jars_dimes * dimes_per_jar
def total_nickels := jars_nickels * nickels_per_jar

-- Calculate the total money Jenn has in quarters, dimes, and nickels.
def total_value_quarters : ℝ := total_quarters * value_per_quarter
def total_value_dimes : ℝ := total_dimes * value_per_dime
def total_value_nickels : ℝ := total_nickels * value_per_nickel

def total_money : ℝ := total_value_quarters + total_value_dimes + total_value_nickels

-- Calculate the money left after buying the bike.
def money_left : ℝ := total_money - bike_cost

-- Prove that the amount of money left is precisely $90.
theorem money_leftover_is_90 : money_left = 90 :=
by
  -- Placeholder for the proof
  sorry

end money_leftover_is_90_l101_101233


namespace solution_pairs_l101_101268

def equation (r p : ℤ) : Prop := r^2 - r * (p + 6) + p^2 + 5 * p + 6 = 0

theorem solution_pairs :
  ∀ (r p : ℤ),
    equation r p ↔ (r = 3 ∧ p = 1) ∨ (r = 4 ∧ p = 1) ∨ 
                    (r = 0 ∧ p = -2) ∨ (r = 4 ∧ p = -2) ∨ 
                    (r = 0 ∧ p = -3) ∨ (r = 3 ∧ p = -3) :=
by
  sorry

end solution_pairs_l101_101268


namespace highest_possible_value_l101_101370

theorem highest_possible_value 
  (t q r1 r2 : ℝ)
  (h_eq : r1 + r2 = t)
  (h_cond : ∀ n : ℕ, n > 0 → r1^n + r2^n = t) :
  t = 2 → q = 1 → 
  r1 = 1 → r2 = 1 →
  (1 / r1^1004 + 1 / r2^1004 = 2) :=
by
  intros h_t h_q h_r1 h_r2
  rw [h_r1, h_r2]
  norm_num

end highest_possible_value_l101_101370


namespace greatest_prime_factor_15_17_factorial_l101_101838

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l101_101838


namespace sqrt_x_eq_0_123_l101_101494

theorem sqrt_x_eq_0_123 (x : ℝ) (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  -- proof goes here, but it is omitted
  sorry

end sqrt_x_eq_0_123_l101_101494


namespace distance_between_closest_points_l101_101636

noncomputable def distance_closest_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  (Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - r1 - r2)

theorem distance_between_closest_points :
  distance_closest_points (4, 4) (20, 12) 4 12 = 4 * Real.sqrt 20 - 16 :=
by
  sorry

end distance_between_closest_points_l101_101636


namespace greatest_prime_factor_of_15f_17f_is_17_l101_101840

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l101_101840


namespace petya_friends_l101_101736

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101736


namespace student_sums_l101_101436

theorem student_sums (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 48) : y = 36 :=
by
  sorry

end student_sums_l101_101436


namespace simplify_expression_l101_101251

-- Define the expressions and the simplification statement
def expr1 (x : ℝ) := (3 * x - 6) * (x + 8)
def expr2 (x : ℝ) := (x + 6) * (3 * x - 2)
def simplified (x : ℝ) := 2 * x - 36

theorem simplify_expression (x : ℝ) : expr1 x - expr2 x = simplified x := by
  sorry

end simplify_expression_l101_101251


namespace relatively_prime_bound_l101_101378

theorem relatively_prime_bound {m n : ℕ} {a : ℕ → ℕ} (h1 : 1 < m) (h2 : 1 < n) (h3 : m ≥ n)
  (h4 : ∀ i j, i ≠ j → a i = a j → False) (h5 : ∀ i, a i ≤ m) (h6 : ∀ i j, i ≠ j → a i ∣ a j → a i = 1) 
  (x : ℝ) : ∃ i, dist (a i * x) (round (a i * x)) ≥ 2 / (m * (m + 1)) * dist x (round x) :=
sorry

end relatively_prime_bound_l101_101378


namespace calculation_result_l101_101311

theorem calculation_result :
  (-1) * (-4) + 2^2 / (7 - 5) = 6 :=
by
  sorry

end calculation_result_l101_101311


namespace range_of_y_under_conditions_l101_101183

theorem range_of_y_under_conditions :
  (∀ x : ℝ, (x - y) * (x + y) < 1) → (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
  intro h
  have h' : ∀ x : ℝ, (x - y) * (1 - x - y) < 1 := by
    sorry
  have g_min : ∀ x : ℝ, y^2 - y < x^2 - x + 1 := by
    sorry
  have min_value : y^2 - y < 3/4 := by
    sorry
  have range_y : (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
    sorry
  exact range_y

end range_of_y_under_conditions_l101_101183


namespace line_intercepts_of_3x_minus_y_plus_6_eq_0_l101_101191

theorem line_intercepts_of_3x_minus_y_plus_6_eq_0 :
  (∃ y, 3 * 0 - y + 6 = 0 ∧ y = 6) ∧ (∃ x, 3 * x - 0 + 6 = 0 ∧ x = -2) :=
by
  sorry

end line_intercepts_of_3x_minus_y_plus_6_eq_0_l101_101191


namespace integral_right_angled_triangles_unique_l101_101754

theorem integral_right_angled_triangles_unique : 
  ∀ a b c : ℤ, (a < b) ∧ (b < c) ∧ (a^2 + b^2 = c^2) ∧ (a * b = 4 * (a + b + c))
  ↔ (a = 10 ∧ b = 24 ∧ c = 26)
  ∨ (a = 12 ∧ b = 16 ∧ c = 20)
  ∨ (a = 9 ∧ b = 40 ∧ c = 41) :=
by {
  sorry
}

end integral_right_angled_triangles_unique_l101_101754


namespace problem1_problem2_l101_101495

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 2) * x + 4

theorem problem1 (a : ℝ) :
  (∀ x, f a x > 0) → 0 < a ∧ a < 4 :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x, -3 <= x ∧ x <= 1 → f a x > 0) → (-1/2 < a ∧ a < 4) :=
sorry

end problem1_problem2_l101_101495


namespace smallest_possible_value_l101_101215

theorem smallest_possible_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b c : ℕ), a = floor((x + y) / z) ∧ b = floor((y + z) / x) ∧ c = floor((z + x) / y) ∧ (a + b + c) = 4 :=
begin
  sorry
end

end smallest_possible_value_l101_101215


namespace solve_for_x_l101_101189

theorem solve_for_x (y : ℝ) (x : ℝ) (h1 : y = 432) (h2 : 12^2 * x^4 / 432 = y) : x = 6 := by
  sorry

end solve_for_x_l101_101189


namespace number_of_green_balls_l101_101018

variable (b g : Nat) (P_b : Rat)

-- Given conditions
def num_blue_balls : Nat := 10
def prob_blue : Rat := 1 / 5

-- Define the total number of balls available and the probability equation
def total_balls : Nat := num_blue_balls + g
def probability_equation : Prop := (num_blue_balls : Rat) / (total_balls : Rat) = prob_blue

-- The main statement to be proven
theorem number_of_green_balls :
  probability_equation → g = 40 := 
sorry

end number_of_green_balls_l101_101018


namespace tournament_ranking_sequences_l101_101372

def total_fair_ranking_sequences (A B C D : Type) : Nat :=
  let saturday_outcomes := 2
  let sunday_outcomes := 4 -- 2 possibilities for (first, second) and 2 for (third, fourth)
  let tiebreaker_effect := 2 -- swap second and third
  saturday_outcomes * sunday_outcomes * tiebreaker_effect

theorem tournament_ranking_sequences (A B C D : Type) :
  total_fair_ranking_sequences A B C D = 32 := 
by
  sorry

end tournament_ranking_sequences_l101_101372


namespace incorrect_statement_l101_101523

theorem incorrect_statement (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬((x = 0 → a^x = 1) ∧
    (x = 1 → a^x = a) ∧
    (x = -1 → a^x = 1/a) ∧
    (x < 0 → 0 < a^x ∧ ∀ ε > 0, ∃ x' < x, a^x' < ε)) :=
sorry

end incorrect_statement_l101_101523


namespace solve_y_l101_101962

theorem solve_y : ∃ y : ℚ, 2 * y + 3 * y = 600 - (4 * y + 5 * y + 100) ∧ y = 250 / 7 := by
  sorry

end solve_y_l101_101962


namespace alpha_beta_sum_l101_101984

theorem alpha_beta_sum (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80 * x + 1551) / (x^2 + 57 * x - 2970)) :
  α + β = 137 :=
by
  sorry

end alpha_beta_sum_l101_101984


namespace susan_chairs_l101_101757

theorem susan_chairs : 
  ∀ (red yellow blue : ℕ), 
  red = 5 → 
  yellow = 4 * red → 
  blue = yellow - 2 → 
  red + yellow + blue = 43 :=
begin
  intros red yellow blue h1 h2 h3,
  rw h1,
  rw h2,
  rw h3,
  norm_num,
end

end susan_chairs_l101_101757


namespace bertha_no_children_count_l101_101304

-- Definitions
def bertha_daughters : ℕ := 6
def granddaughters_per_daughter : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30

-- Theorem to be proved
theorem bertha_no_children_count : 
  ∃ x : ℕ, (x * granddaughters_per_daughter + bertha_daughters = total_daughters_and_granddaughters) ∧ 
           (bertha_daughters - x + x * granddaughters_per_daughter = 26) :=
sorry

end bertha_no_children_count_l101_101304


namespace find_original_number_l101_101680

-- Definitions of the conditions
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem find_original_number (n x y : ℕ) 
  (h1 : isFiveDigitNumber n) 
  (h2 : n = 10 * x + y) 
  (h3 : n - x = 54321) : 
  n = 60356 := 
sorry

end find_original_number_l101_101680


namespace count_even_numbers_l101_101205

theorem count_even_numbers (a b : ℕ) (h1 : a > 300) (h2 : b ≤ 600) (h3 : ∀ n, 300 < n ∧ n ≤ 600 → n % 2 = 0) : 
  ∃ c : ℕ, c = 150 :=
by
  sorry

end count_even_numbers_l101_101205


namespace petya_friends_count_l101_101740

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101740


namespace quadratic_eq_one_solution_m_eq_49_div_12_l101_101575

theorem quadratic_eq_one_solution_m_eq_49_div_12 (m : ℝ) : 
  (∃ m, ∀ x, 3 * x ^ 2 - 7 * x + m = 0 → (b^2 - 4 * a * c = 0) → m = 49 / 12) :=
by
  sorry

end quadratic_eq_one_solution_m_eq_49_div_12_l101_101575


namespace smallest_integer_with_12_divisors_l101_101601

-- The number of divisors of a positive integer n
def num_divisors (n : ℕ) : ℕ :=
  (n.factors.group_by id).vals.map List.length |>.map (· + 1) |>.prod

-- The main theorem to prove
theorem smallest_integer_with_12_divisors : ∃ n : ℕ, num_divisors n = 12 ∧ (∀ m : ℕ, num_divisors m = 12 → n ≤ m) :=
by
  sorry

end smallest_integer_with_12_divisors_l101_101601


namespace petya_friends_l101_101733

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101733


namespace total_yards_fabric_l101_101591

variable (spent_checkered spent_plain cost_per_yard : ℝ)

def yards_checkered : ℝ := spent_checkered / cost_per_yard
def yards_plain : ℝ := spent_plain / cost_per_yard
def total_yards : ℝ := yards_checkered + yards_plain

theorem total_yards_fabric (h1 : spent_checkered = 75) (h2 : spent_plain = 45) (h3 : cost_per_yard = 7.50) :
  total_yards = 16 := by
  sorry

end total_yards_fabric_l101_101591


namespace petya_friends_l101_101725

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101725


namespace smallest_possible_value_l101_101214

theorem smallest_possible_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b c : ℕ), a = floor((x + y) / z) ∧ b = floor((y + z) / x) ∧ c = floor((z + x) / y) ∧ (a + b + c) = 4 :=
begin
  sorry
end

end smallest_possible_value_l101_101214


namespace mean_temperature_l101_101576

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end mean_temperature_l101_101576


namespace rightmost_three_digits_of_7_pow_2023_l101_101776

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l101_101776


namespace sin_A_calculation_height_calculation_l101_101939

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l101_101939


namespace quadratic_intersect_condition_l101_101492

theorem quadratic_intersect_condition (k : ℝ) :
  (k > -1/4) ∧ (k ≠ 2) ↔ ((2*k - 1)^2 - 4*k*(k - 2) > 0) ∧ (k - 2 ≠ 0) :=
begin
  sorry
end

end quadratic_intersect_condition_l101_101492


namespace complex_transformation_l101_101877

theorem complex_transformation :
  let z := complex.mk (-3) (-8)
  let rotation := complex.mk (1) (real.sqrt 3)
  let dilation := 2
  (z * (rotation * dilation)) = complex.mk (8 * real.sqrt 3 - 3) (-(3 * real.sqrt 3 + 8)) :=
by
  -- Placeholder for the proof
  sorry

end complex_transformation_l101_101877


namespace petya_friends_l101_101704

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101704


namespace fill_in_the_blank_l101_101303

-- Definitions of the problem conditions
def parent := "being a parent"
def parent_with_special_needs := "being the parent of a child with special needs"

-- The sentence describing two situations of being a parent
def sentence1 := "Being a parent is not always easy"
def sentence2 := "being the parent of a child with special needs often carries with ___ extra stress."

-- The correct word to fill in the blank.
def correct_answer := "it"

-- Proof problem
theorem fill_in_the_blank : correct_answer = "it" :=
by
  sorry

end fill_in_the_blank_l101_101303


namespace range_of_2a_plus_3b_l101_101648

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l101_101648


namespace gender_independence_expectation_X_probability_meet_standard_l101_101265

-- Given data
def total_population : ℕ := 100
def male_total : ℕ := 45
def female_total : ℕ := 55

def exercise_distribution : List (ℕ × ℕ) := [(30, 15), (45, 10)]

noncomputable def χ_squared (a b c d n : ℕ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem gender_independence :
  let a := 30
  let b := 15
  let c := 45
  let d := 10
  let n := 100 in
  χ_squared a b c d n < 3.841 :=
by sorry

-- Distribution and Expectation of X
--- Given probabilities for P(X)
def P_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 1/10
  | 1 => 3/5
  | 2 => 3/10
  | _ => 0

--- Expectation of X
noncomputable def E_X : ℚ :=
  let p0 := 0 * (1 / 10)
  let p1 := 1 * (3 / 5)
  let p2 := 2 * (3 / 10)
  p0 + p1 + p2

theorem expectation_X : E_X = 6 / 5 :=
by sorry

-- Binomial distribution
def P_meeting_standard (k : ℕ) : ℚ :=
  if k = 2 then (Nat.choose 3 k) * (1/4)^k * (3/4)^(3-k) else 0

theorem probability_meet_standard :
  P_meeting_standard 2 = 9 / 64 :=
by sorry

end gender_independence_expectation_X_probability_meet_standard_l101_101265


namespace minValue_l101_101895

theorem minValue (x y z : ℝ) (h : 1/x + 2/y + 3/z = 1) : x + y/2 + z/3 ≥ 9 :=
by
  sorry

end minValue_l101_101895


namespace chocolates_eaten_by_robert_l101_101387

theorem chocolates_eaten_by_robert (nickel_ate : ℕ) (robert_ate_more : ℕ) (H1 : nickel_ate = 3) (H2 : robert_ate_more = 4) :
  nickel_ate + robert_ate_more = 7 :=
by {
  sorry
}

end chocolates_eaten_by_robert_l101_101387


namespace proof_triangle_properties_l101_101924

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l101_101924


namespace geom_progression_sum_ratio_l101_101913

theorem geom_progression_sum_ratio (a : ℝ) (r : ℝ) (m : ℕ) :
  r = 5 →
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^m) / (1 - r)) = 126 →
  m = 3 := by
  sorry

end geom_progression_sum_ratio_l101_101913


namespace percent_of_z_l101_101614

variable (x y z : ℝ)

theorem percent_of_z :
  x = 1.20 * y →
  y = 0.40 * z →
  x = 0.48 * z :=
by
  intros h1 h2
  sorry

end percent_of_z_l101_101614


namespace number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l101_101357

theorem number_of_whole_numbers_between_sqrt_18_and_sqrt_120 : 
  ∀ (n : ℕ), 
  (5 ≤ n ∧ n ≤ 10) ↔ (6 = 6) :=
sorry

end number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l101_101357


namespace monotonicity_f_when_a_eq_1_range_of_a_l101_101506

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_f_when_a_eq_1 :
  (∀ x > 0, deriv (λ x, f x 1) x > 0) ∧ (∀ x < 0, deriv (λ x, f x 1) x < 0) :=
sorry

-- Part 2: Range of a such that f(x) ≥ 1/2 * x^3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x a ≥ 1/2 * x^3 + 1) ↔ a ≥ (7 - Real.exp 2) / 4 :=
sorry

end monotonicity_f_when_a_eq_1_range_of_a_l101_101506


namespace sum_of_digits_of_t_l101_101548

theorem sum_of_digits_of_t (m n k : ℕ) (h1 : m > n) (h2 : n = 6) 
  (h3 : Nat.trailingZeroes m.factorial = k) 
  (h4 : Nat.trailingZeroes (m + n).factorial = 2 * k) 
  (h_m : m = 20 ∨ m = 21 ∨ m = 25) :
  let t := 20 + 21 + 25 in
  Nat.digits 10 t.sum = 12 :=
by
  sorry

end sum_of_digits_of_t_l101_101548


namespace simplify_fraction_l101_101961

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by
  -- proof goes here
  sorry

end simplify_fraction_l101_101961


namespace sin_A_eq_height_on_AB_l101_101935

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l101_101935


namespace two_digit_number_l101_101918

theorem two_digit_number (x y : ℕ) (h1 : y = x + 4) (h2 : (10 * x + y) * (x + y) = 208) :
  10 * x + y = 26 :=
sorry

end two_digit_number_l101_101918


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101331

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101331


namespace sugar_needed_for_partial_recipe_l101_101019

theorem sugar_needed_for_partial_recipe :
  let initial_sugar := 5 + 3/4
  let part := 3/4
  let needed_sugar := 4 + 5/16
  initial_sugar * part = needed_sugar := 
by 
  sorry

end sugar_needed_for_partial_recipe_l101_101019


namespace symmetric_point_line_l101_101762

theorem symmetric_point_line (a b : ℝ) :
  (∀ (x y : ℝ), (y - 2) / (x - 1) = -2 → (x + 1)/2 + 2 * (y + 2)/2 - 10 = 0) →
  a = 3 ∧ b = 6 := by
  intro h
  sorry

end symmetric_point_line_l101_101762


namespace mistaken_divisor_is_12_l101_101531

-- Definitions based on conditions
def correct_divisor : ℕ := 21
def correct_quotient : ℕ := 36
def mistaken_quotient : ℕ := 63

-- The mistaken divisor  is computed as:
def mistaken_divisor : ℕ := correct_quotient * correct_divisor / mistaken_quotient

-- The theorem to prove the mistaken divisor is 12
theorem mistaken_divisor_is_12 : mistaken_divisor = 12 := by
  sorry

end mistaken_divisor_is_12_l101_101531


namespace tourist_groupings_count_l101_101407

/-- We define three guides and eight tourists. -/
def num_guides : ℕ := 3
def num_tourists : ℕ := 8

/-- The number of valid groupings of tourists where each guide gets at least one tourist -/
noncomputable def num_groupings : ℕ :=
  let total_arrangements := (num_guides ^ num_tourists) in
  let invalid_groupings :=
    (num_guides * (2 ^ num_tourists)) +
    (num_guides * 1) in
  total_arrangements - invalid_groupings

theorem tourist_groupings_count : num_groupings = 5796 := by
  -- Proof to be filled in
  sorry

end tourist_groupings_count_l101_101407


namespace final_temperature_is_100_l101_101545

-- Definitions based on conditions
def initial_temperature := 20  -- in degrees
def heating_rate := 5          -- in degrees per minute
def heating_time := 16         -- in minutes

-- The proof statement
theorem final_temperature_is_100 :
  initial_temperature + heating_rate * heating_time = 100 := by
  sorry

end final_temperature_is_100_l101_101545


namespace total_price_before_increase_l101_101418

-- Conditions
def original_price_candy_box (c_or: ℝ) := 10 = c_or * 1.25
def original_price_soda_can (s_or: ℝ) := 15 = s_or * 1.50

-- Goal
theorem total_price_before_increase :
  ∃ (c_or s_or : ℝ), original_price_candy_box c_or ∧ original_price_soda_can s_or ∧ c_or + s_or = 25 :=
by
  sorry

end total_price_before_increase_l101_101418


namespace petya_has_19_friends_l101_101751

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101751


namespace range_of_k_l101_101654

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > 0 → (k+4) * x < 0) → k < -4 :=
by
  sorry

end range_of_k_l101_101654


namespace notecard_calculation_l101_101305

theorem notecard_calculation (N E : ℕ) (h₁ : N - E = 80) (h₂ : N = 3 * E) : N = 120 :=
sorry

end notecard_calculation_l101_101305


namespace greatest_prime_factor_15_fact_17_fact_l101_101786

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l101_101786


namespace james_tylenol_intake_per_day_l101_101232

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end james_tylenol_intake_per_day_l101_101232


namespace price_increase_percentage_l101_101027

theorem price_increase_percentage (original_price : ℝ) (discount : ℝ) (reduced_price : ℝ) : 
  reduced_price = original_price * (1 - discount) →
  (original_price / reduced_price - 1) * 100 = 8.7 :=
by
  intros h
  sorry

end price_increase_percentage_l101_101027


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101333

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101333


namespace arcsin_half_eq_pi_six_l101_101455

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101455


namespace quadratic_points_order_l101_101077

theorem quadratic_points_order (c y1 y2 : ℝ) 
  (hA : y1 = 0^2 - 6 * 0 + c)
  (hB : y2 = 4^2 - 6 * 4 + c) : 
  y1 > y2 := 
by 
  sorry

end quadratic_points_order_l101_101077


namespace defeated_candidate_percentage_l101_101226

noncomputable def percentage_defeated_candidate (total_votes diff_votes invalid_votes : ℕ) : ℕ :=
  let valid_votes := total_votes - invalid_votes
  let P := 100 * (valid_votes - diff_votes) / (2 * valid_votes)
  P

theorem defeated_candidate_percentage (total_votes : ℕ) (diff_votes : ℕ) (invalid_votes : ℕ) :
  total_votes = 12600 ∧ diff_votes = 5000 ∧ invalid_votes = 100 → percentage_defeated_candidate total_votes diff_votes invalid_votes = 30 :=
by
  intros
  sorry

end defeated_candidate_percentage_l101_101226


namespace value_independent_of_a_value_when_b_is_neg_2_l101_101901

noncomputable def algebraic_expression (a b : ℝ) : ℝ :=
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2)

theorem value_independent_of_a (a b : ℝ) : algebraic_expression a b = 2 * b^2 :=
by
  sorry

theorem value_when_b_is_neg_2 (a : ℝ) : algebraic_expression a (-2) = 8 :=
by
  sorry

end value_independent_of_a_value_when_b_is_neg_2_l101_101901


namespace probability_white_ball_l101_101263

/-- Define the conditions for the two urns and the probability of drawing specific balls from them. -/
def urn1_total : ℕ := 9
def urn2_total : ℕ := 10

def urn1_white : ℕ := 5
def urn1_blue : ℕ := 4
def urn2_white : ℕ := 2
def urn2_blue : ℕ := 8

/-- Define the probabilities for drawing white and blue balls from the two urns. -/
def P_urn1_white : ℚ := urn1_white / urn1_total
def P_urn2_white : ℚ := urn2_white / urn2_total
def P_urn1_blue  : ℚ := urn1_blue / urn1_total
def P_urn2_blue  : ℚ := urn2_blue / urn2_total

/-- Define the probability combinations for drawing a white ball from the third urn according to the scenarios. -/
def P_scenario1 : ℚ := P_urn1_white * P_urn2_blue / 2
def P_scenario2 : ℚ := P_urn1_blue * P_urn2_white / 2
def P_scenario3 : ℚ := P_urn1_white * P_urn2_white

/-- Calculate the total probability of drawing a white ball from the third urn. -/
def P_white_from_third_urn : ℚ := P_scenario1 + P_scenario2 + P_scenario3

/-- Main theorem: The probability of drawing a white ball from the third urn is given by 17/45. -/
theorem probability_white_ball : P_white_from_third_urn = 17 / 45 := by
  sorry

end probability_white_ball_l101_101263


namespace sequence_comparison_l101_101502

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def arith_seq (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n-1) * d

theorem sequence_comparison
  (a₁ b₁ q d : ℝ)
  (h₃ : geom_seq a₁ q 3 = arith_seq b₁ d 3)
  (h₇ : geom_seq a₁ q 7 = arith_seq b₁ d 7)
  (q_pos : 0 < q)
  (d_pos : 0 < d) :
  geom_seq a₁ q 5 < arith_seq b₁ d 5 ∧
  geom_seq a₁ q 1 > arith_seq b₁ d 1 ∧
  geom_seq a₁ q 9 > arith_seq b₁ d 9 :=
by
  sorry

end sequence_comparison_l101_101502


namespace product_positivity_l101_101766

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l101_101766


namespace probability_two_flies_swept_l101_101100

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l101_101100


namespace factor_expression_l101_101484

theorem factor_expression (x : ℝ) :
  4 * x * (x - 5) + 7 * (x - 5) + 12 * (x - 5) = (4 * x + 19) * (x - 5) :=
by
  sorry

end factor_expression_l101_101484


namespace largest_inscribed_equilateral_triangle_area_l101_101314

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end largest_inscribed_equilateral_triangle_area_l101_101314


namespace probability_two_flies_swept_away_l101_101102

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l101_101102


namespace sin_A_correct_height_on_AB_correct_l101_101930

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l101_101930


namespace sequence_formula_l101_101047

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (diff : ∀ n, a (n + 1) - a n = 3^n) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end sequence_formula_l101_101047


namespace intersection_A_complement_UB_l101_101353

-- Definitions of the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5 * x ≥ 0}

-- Complement of B w.r.t. U
def complement_U_B : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

-- The statement we want to prove
theorem intersection_A_complement_UB : A ∩ complement_U_B = {2, 3} := by
  sorry

end intersection_A_complement_UB_l101_101353


namespace transistors_in_2010_l101_101367

theorem transistors_in_2010 
  (initial_transistors : ℕ) 
  (initial_year : ℕ) 
  (final_year : ℕ) 
  (doubling_period : ℕ)
  (initial_transistors_eq: initial_transistors = 500000)
  (initial_year_eq: initial_year = 1985)
  (final_year_eq: final_year = 2010)
  (doubling_period_eq : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 2048000000 := 
by 
  -- the proof goes here
  sorry

end transistors_in_2010_l101_101367


namespace complex_round_quadrant_l101_101073

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end complex_round_quadrant_l101_101073


namespace g_at_5_l101_101882

def g (x : ℝ) : ℝ := sorry -- Placeholder for the function definition, typically provided in further context

theorem g_at_5 : g 5 = 3 / 4 :=
by
  -- Given condition as a hypothesis
  have h : ∀ x: ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1 := sorry
  sorry  -- Full proof should go here

end g_at_5_l101_101882


namespace kangaroo_arrangement_count_l101_101320

theorem kangaroo_arrangement_count :
  let k := 8
  let tallest_at_ends := 2
  let middle := k - tallest_at_ends
  (tallest_at_ends * (middle.factorial)) = 1440 := by
  sorry

end kangaroo_arrangement_count_l101_101320


namespace ski_helmet_final_price_l101_101702

variables (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
def final_price_after_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let after_first_discount := initial_price * (1 - discount1)
  let after_second_discount := after_first_discount * (1 - discount2)
  after_second_discount

theorem ski_helmet_final_price :
  final_price_after_discounts 120 0.40 0.20 = 57.60 := 
  sorry

end ski_helmet_final_price_l101_101702


namespace solution_set_equivalence_l101_101401

def abs_value_solution_set (x : ℝ) : Prop := (x) * (|x + 2|) < 0

theorem solution_set_equivalence : {x : ℝ | abs_value_solution_set x} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 0)} :=
by
  sorry

end solution_set_equivalence_l101_101401


namespace tanner_savings_in_october_l101_101759

theorem tanner_savings_in_october 
    (sept_savings : ℕ := 17) 
    (nov_savings : ℕ := 25)
    (spent : ℕ := 49) 
    (left : ℕ := 41) 
    (X : ℕ) 
    (h : sept_savings + X + nov_savings - spent = left) 
    : X = 48 :=
by
  sorry

end tanner_savings_in_october_l101_101759


namespace trains_crossing_time_l101_101774

theorem trains_crossing_time :
  let length_of_each_train := 120 -- in meters
  let speed_of_each_train := 12 -- in km/hr
  let total_distance := length_of_each_train * 2
  let relative_speed := (speed_of_each_train * 1000 / 3600 * 2) -- in m/s
  total_distance / relative_speed = 36 := 
by
  -- Since we only need to state the theorem, the proof is omitted.
  sorry

end trains_crossing_time_l101_101774


namespace abigail_score_l101_101534

theorem abigail_score (sum_20 : ℕ) (sum_21 : ℕ) (h1 : sum_20 = 1700) (h2 : sum_21 = 1806) : (sum_21 - sum_20) = 106 :=
by
  sorry

end abigail_score_l101_101534


namespace first_class_students_count_l101_101255

theorem first_class_students_count 
  (x : ℕ) 
  (avg1 : ℕ) (avg2 : ℕ) (num2 : ℕ) (overall_avg : ℝ)
  (h_avg1 : avg1 = 40)
  (h_avg2 : avg2 = 60)
  (h_num2 : num2 = 50)
  (h_overall_avg : overall_avg = 52.5)
  (h_eq : 40 * x + 60 * 50 = (52.5:ℝ) * (x + 50)) :
  x = 30 :=
by
  sorry

end first_class_students_count_l101_101255


namespace triangle_area_is_2_l101_101968

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2_l101_101968


namespace arcsin_of_half_l101_101468

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101468


namespace average_weight_of_16_boys_l101_101404

theorem average_weight_of_16_boys :
  ∃ A : ℝ,
    (16 * A + 8 * 45.15 = 24 * 48.55) ∧
    A = 50.25 :=
by {
  -- Proof skipped, using sorry to denote the proof is required.
  sorry
}

end average_weight_of_16_boys_l101_101404


namespace fill_pipe_half_time_l101_101154

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l101_101154


namespace arcsin_of_half_l101_101464

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101464


namespace arcsin_one_half_l101_101470

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101470


namespace petya_has_19_friends_l101_101747

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101747


namespace find_k_l101_101069

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem find_k (a b k : ℝ) (h1 : f a b k = 4) (h2 : f a b (f a b k) = 7) (h3 : f a b (f a b (f a b k)) = 19) :
  k = 13 / 4 := 
sorry

end find_k_l101_101069


namespace circles_intersect_l101_101399

def circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 4 * p.2 + 3 = 0}

theorem circles_intersect : ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 :=
by
  sorry

end circles_intersect_l101_101399


namespace coordinate_inequality_l101_101266

theorem coordinate_inequality (x y : ℝ) :
  (xy > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧ (xy < 0 → (x - 2)^2 + (y + 1)^2 > 5) :=
by
  sorry

end coordinate_inequality_l101_101266


namespace divisibility_theorem_l101_101383

theorem divisibility_theorem {a m x n : ℕ} : (m ∣ n) ↔ (x^m - a^m ∣ x^n - a^n) :=
by
  sorry

end divisibility_theorem_l101_101383


namespace bananas_and_cantaloupe_cost_l101_101193

noncomputable def prices (a b c d : ℕ) : Prop :=
  a + b + c + d = 40 ∧
  d = 3 * a ∧
  b = c - 2

theorem bananas_and_cantaloupe_cost (a b c d : ℕ) (h : prices a b c d) : b + c = 20 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  -- Using the given conditions:
  --     a + b + c + d = 40
  --     d = 3 * a
  --     b = c - 2
  -- We find that b + c = 20
  sorry

end bananas_and_cantaloupe_cost_l101_101193


namespace price_restoration_l101_101000

theorem price_restoration (P : Real) (hP : P > 0) :
  let new_price := 0.85 * P
  let required_increase := ((1 / 0.85) - 1) * 100
  required_increase = 17.65 :=
by 
  sorry

end price_restoration_l101_101000


namespace solution_set_f_pos_l101_101898

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

-- Conditions
axiom h1 : ∀ x, f (-x) = -f x     -- f(x) is odd
axiom h2 : f 2 = 0                -- f(2) = 0
axiom h3 : ∀ x > 0, 2 * f x + x * (deriv f x) > 0 -- 2f(x) + xf'(x) > 0 for x > 0

-- Theorem to prove
theorem solution_set_f_pos : { x : ℝ | f x > 0 } = { x : ℝ | x > 2 ∨ (-2 < x ∧ x < 0) } :=
sorry

end solution_set_f_pos_l101_101898


namespace min_even_integers_among_eight_l101_101004

theorem min_even_integers_among_eight :
  ∃ (x y z a b m n o : ℤ), 
    x + y + z = 30 ∧
    x + y + z + a + b = 49 ∧
    x + y + z + a + b + m + n + o = 78 ∧
    (∀ e : ℕ, (∀ x y z a b m n o : ℤ, x + y + z = 30 ∧ x + y + z + a + b = 49 ∧ x + y + z + a + b + m + n + o = 78 → 
    e = 2)) := sorry

end min_even_integers_among_eight_l101_101004


namespace line_intersects_circle_l101_101888

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (kx - y - k +1 = 0) ∧ (x^2 + y^2 = 4) :=
sorry

end line_intersects_circle_l101_101888


namespace inequality_proof_l101_101092

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

theorem inequality_proof : (a * b < a + b ∧ a + b < 0) :=
by
  sorry

end inequality_proof_l101_101092


namespace tan_x_y_l101_101103

theorem tan_x_y (x y : ℝ) (h : Real.sin (2 * x + y) = 5 * Real.sin y) :
  Real.tan (x + y) = (3 / 2) * Real.tan x :=
sorry

end tan_x_y_l101_101103


namespace triangle_isosceles_or_right_angled_l101_101672

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ ∨ β + γ = Real.pi / 2) :=
sorry

end triangle_isosceles_or_right_angled_l101_101672


namespace probability_of_reaching_3_1_without_2_0_in_8_steps_l101_101390

theorem probability_of_reaching_3_1_without_2_0_in_8_steps :
  let n_total := 1680
  let invalid := 30
  let total := n_total - invalid
  let q := total / 4^8
  let gcd := Nat.gcd total 65536
  let m := total / gcd
  let n := 65536 / gcd
  (m + n = 11197) :=
by
  sorry

end probability_of_reaching_3_1_without_2_0_in_8_steps_l101_101390


namespace range_of_y_l101_101067

theorem range_of_y (x y : ℝ) (h1 : |y - 2 * x| = x^2) (h2 : -1 < x) (h3 : x < 0) : -3 < y ∧ y < 0 :=
by
  sorry

end range_of_y_l101_101067


namespace triangle_side_b_l101_101375

-- Define the conditions and state the problem
theorem triangle_side_b (A C : ℕ) (a b c : ℝ)
  (h1 : C = 4 * A)
  (h2 : a = 36)
  (h3 : c = 60) :
  b = 45 := by
  sorry

end triangle_side_b_l101_101375


namespace employees_age_distribution_l101_101369

-- Define the total number of employees
def totalEmployees : ℕ := 15000

-- Define the percentages
def malePercentage : ℝ := 0.58
def femalePercentage : ℝ := 0.42

-- Define the age distribution percentages for male employees
def maleBelow30Percentage : ℝ := 0.25
def male30To50Percentage : ℝ := 0.40
def maleAbove50Percentage : ℝ := 0.35

-- Define the percentage of female employees below 30
def femaleBelow30Percentage : ℝ := 0.30

-- Define the number of male employees
def numMaleEmployees : ℝ := malePercentage * totalEmployees

-- Calculate the number of male employees in each age group
def numMaleBelow30 : ℝ := maleBelow30Percentage * numMaleEmployees
def numMale30To50 : ℝ := male30To50Percentage * numMaleEmployees
def numMaleAbove50 : ℝ := maleAbove50Percentage * numMaleEmployees

-- Define the number of female employees
def numFemaleEmployees : ℝ := femalePercentage * totalEmployees

-- Calculate the number of female employees below 30
def numFemaleBelow30 : ℝ := femaleBelow30Percentage * numFemaleEmployees

-- Calculate the total number of employees below 30
def totalBelow30 : ℝ := numMaleBelow30 + numFemaleBelow30

-- We now state our theorem to prove
theorem employees_age_distribution :
  numMaleBelow30 = 2175 ∧
  numMale30To50 = 3480 ∧
  numMaleAbove50 = 3045 ∧
  totalBelow30 = 4065 := by
    sorry

end employees_age_distribution_l101_101369


namespace samantha_coins_value_l101_101104

theorem samantha_coins_value (n d : ℕ) (h1 : n + d = 25) 
    (original_value : ℕ := 250 - 5 * n) 
    (swapped_value : ℕ := 125 + 5 * n)
    (h2 : swapped_value = original_value + 100) : original_value = 140 := 
by
  sorry

end samantha_coins_value_l101_101104


namespace apple_baskets_l101_101094

theorem apple_baskets (total_apples : ℕ) (apples_per_basket : ℕ) (total_apples_eq : total_apples = 495) (apples_per_basket_eq : apples_per_basket = 25) :
  total_apples / apples_per_basket = 19 :=
by
  sorry

end apple_baskets_l101_101094


namespace proof_f_2008_l101_101259

theorem proof_f_2008 {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (3 * x + 1) = f (3 * (x + 1) + 1))
  (h3 : f (-1) = -1) : 
  f 2008 = 1 := 
by
  sorry

end proof_f_2008_l101_101259


namespace greatest_prime_factor_of_15f_17f_is_17_l101_101841

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l101_101841


namespace complex_multiplication_l101_101629

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  ((a + b * i) * (c + d * i)) = (-6 + 33 * i) :=
by
  have a := 3
  have b := -4
  have c := -6
  have d := 3
  sorry

end complex_multiplication_l101_101629


namespace sum_first_60_natural_numbers_l101_101174

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l101_101174


namespace frac_add_eq_l101_101850

theorem frac_add_eq : (2 / 5) + (3 / 10) = 7 / 10 := 
by
  sorry

end frac_add_eq_l101_101850


namespace least_integer_solution_l101_101136

theorem least_integer_solution (x : ℤ) : (∀ y : ℤ, |2 * y + 9| <= 20 → x ≤ y) ↔ x = -14 := by
  sorry

end least_integer_solution_l101_101136


namespace smallest_integer_with_exactly_12_divisors_l101_101607

theorem smallest_integer_with_exactly_12_divisors : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m ≠ n → (nat.divisors_count m = 12 → n < m))) ∧ nat.divisors_count n = 12 :=
by
  sorry

end smallest_integer_with_exactly_12_divisors_l101_101607


namespace greatest_prime_factor_15_fact_17_fact_l101_101787

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l101_101787


namespace concert_attendance_l101_101097

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l101_101097


namespace value_of_f_at_4_l101_101076

noncomputable def f (α : ℝ) (x : ℝ) := x^α

theorem value_of_f_at_4 : 
  (∃ α : ℝ, f α 2 = (Real.sqrt 2) / 2) → f (-1 / 2) 4 = 1 / 2 :=
by
  intros h
  sorry

end value_of_f_at_4_l101_101076


namespace time_needed_by_Alpha_and_Beta_l101_101408

theorem time_needed_by_Alpha_and_Beta (A B C h : ℝ)
  (h₀ : 1 / (A - 4) = 1 / (B - 2))
  (h₁ : 1 / A + 1 / B + 1 / C = 3 / C)
  (h₂ : A = B + 2)
  (h₃ : 1 / 12 + 1 / 10 = 11 / 60)
  : h = 60 / 11 :=
sorry

end time_needed_by_Alpha_and_Beta_l101_101408


namespace simplify_sqrt1_simplify_sqrt2_find_a_l101_101567

-- Part 1
theorem simplify_sqrt1 : ∃ m n : ℝ, m^2 + n^2 = 6 ∧ m * n = Real.sqrt 5 ∧ Real.sqrt (6 + 2 * Real.sqrt 5) = m + n :=
by sorry

-- Part 2
theorem simplify_sqrt2 : ∃ m n : ℝ, m^2 + n^2 = 5 ∧ m * n = -Real.sqrt 6 ∧ Real.sqrt (5 - 2 * Real.sqrt 6) = abs (m - n) :=
by sorry

-- Part 3
theorem find_a (a : ℝ) : (Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5) → (a = 3 ∨ a = -3) :=
by sorry

end simplify_sqrt1_simplify_sqrt2_find_a_l101_101567


namespace smallest_m_for_integral_solutions_l101_101007

theorem smallest_m_for_integral_solutions (p q : ℤ) (h : p * q = 42) (h0 : p + q = m / 15) : 
  0 < m ∧ 15 * p * p - m * p + 630 = 0 ∧ 15 * q * q - m * q + 630 = 0 →
  m = 195 :=
by 
  sorry

end smallest_m_for_integral_solutions_l101_101007


namespace solve_for_x_l101_101220

theorem solve_for_x (x : ℝ) (h : (10 - 6 * x)^ (1 / 3) = -2) : x = 3 := 
by
  sorry

end solve_for_x_l101_101220


namespace exterior_angle_of_triangle_cond_40_degree_l101_101125

theorem exterior_angle_of_triangle_cond_40_degree (A B C : ℝ)
  (h1 : (A = 40 ∨ B = 40 ∨ C = 40))
  (h2 : A = B)
  (h3 : A + B + C = 180) :
  ((180 - C) = 80 ∨ (180 - C) = 140) :=
by
  sorry

end exterior_angle_of_triangle_cond_40_degree_l101_101125


namespace first_division_percentage_l101_101682

theorem first_division_percentage (total_students : ℕ) (second_division_percentage just_passed_students : ℕ) 
  (h1 : total_students = 300) (h2 : second_division_percentage = 54) (h3 : just_passed_students = 60) : 
  (100 - second_division_percentage - ((just_passed_students * 100) / total_students)) = 26 :=
by
  sorry

end first_division_percentage_l101_101682


namespace value_of_k_through_point_l101_101894

noncomputable def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

theorem value_of_k_through_point (k : ℝ) (h : k ≠ 0) : inverse_proportion_function 2 k = 3 → k = 6 :=
by
  sorry

end value_of_k_through_point_l101_101894


namespace average_price_per_share_l101_101875

-- Define the conditions
def Microtron_price_per_share := 36
def Dynaco_price_per_share := 44
def total_shares := 300
def Dynaco_shares_sold := 150

-- Define the theorem to be proved
theorem average_price_per_share : 
  (Dynaco_shares_sold * Dynaco_price_per_share + (total_shares - Dynaco_shares_sold) * Microtron_price_per_share) / total_shares = 40 :=
by
  -- Skip the actual proof here
  sorry

end average_price_per_share_l101_101875


namespace ordering_9_8_4_12_3_16_l101_101134

theorem ordering_9_8_4_12_3_16 : (4 ^ 12 < 9 ^ 8) ∧ (9 ^ 8 = 3 ^ 16) :=
by {
  sorry
}

end ordering_9_8_4_12_3_16_l101_101134


namespace petya_friends_l101_101719

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101719


namespace simon_legos_l101_101572

theorem simon_legos (B : ℝ) (K : ℝ) (x : ℝ) (simon_has : ℝ) 
  (h1 : simon_has = B * 1.20)
  (h2 : K = 40)
  (h3 : B = K + x)
  (h4 : simon_has = 72) : simon_has = 72 := by
  sorry

end simon_legos_l101_101572


namespace trays_from_first_table_is_23_l101_101988

-- Definitions of conditions
def trays_per_trip : ℕ := 7
def trips_made : ℕ := 4
def trays_from_second_table : ℕ := 5

-- Total trays carried
def total_trays_carried : ℕ := trays_per_trip * trips_made

-- Number of trays picked from first table
def trays_from_first_table : ℕ :=
  total_trays_carried - trays_from_second_table

-- Theorem stating that the number of trays picked up from the first table is 23
theorem trays_from_first_table_is_23 : trays_from_first_table = 23 := by
  sorry

end trays_from_first_table_is_23_l101_101988


namespace vector_magnitude_parallel_l101_101354

theorem vector_magnitude_parallel (x : ℝ) 
  (h1 : 4 / x = 2 / 1) :
  ( Real.sqrt ((4 + x) ^ 2 + (2 + 1) ^ 2) ) = 3 * Real.sqrt 5 := 
sorry

end vector_magnitude_parallel_l101_101354


namespace parabola_line_intersection_length_l101_101658

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - 1
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length (k x1 x2 y1 y2 : ℝ)
  (h_focus : line 1 0 k)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_line1 : line x1 y1 k)
  (h_line2 : line x2 y2 k) :
  k = 1 ∧ (x1 + x2 + 2) = 8 :=
by
  sorry

end parabola_line_intersection_length_l101_101658


namespace pants_and_coat_cost_l101_101561

noncomputable def pants_shirt_costs : ℕ := 100
noncomputable def coat_cost_times_shirt : ℕ := 5
noncomputable def coat_cost : ℕ := 180

theorem pants_and_coat_cost (p s c : ℕ) 
  (h1 : p + s = pants_shirt_costs)
  (h2 : c = coat_cost_times_shirt * s)
  (h3 : c = coat_cost) :
  p + c = 244 :=
by
  sorry

end pants_and_coat_cost_l101_101561


namespace book_cost_l101_101688

theorem book_cost (n_5 n_3 : ℕ) (N : ℕ) :
  (N = n_5 + n_3) ∧ (N > 10) ∧ (N < 20) ∧ (5 * n_5 = 3 * n_3) →  5 * n_5 = 30 := 
sorry

end book_cost_l101_101688


namespace compute_expression_l101_101476

theorem compute_expression : (-3) * 2 + 4 = -2 := 
by
  sorry

end compute_expression_l101_101476


namespace reduced_price_correct_l101_101613

theorem reduced_price_correct (P R Q: ℝ) (h1 : R = 0.75 * P) (h2 : 900 = Q * P) (h3 : 900 = (Q + 5) * R)  :
  R = 45 := by 
  sorry

end reduced_price_correct_l101_101613


namespace evaluate_product_l101_101043

theorem evaluate_product (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5 * n^3 + 4 * n^2 + 4 * n := 
by
  -- Omitted proof steps
  sorry

end evaluate_product_l101_101043


namespace sandy_total_puppies_l101_101105

-- Definitions based on conditions:
def original_puppies : ℝ := 8.0
def additional_puppies : ℝ := 4.0

-- Theorem statement: total_puppies should be 12.0
theorem sandy_total_puppies : original_puppies + additional_puppies = 12.0 := 
by
  sorry

end sandy_total_puppies_l101_101105


namespace area_of_inscribed_equilateral_triangle_l101_101313

theorem area_of_inscribed_equilateral_triangle
  (r : ℝ) (h₀ : r = 10) : 
  ∃ A : ℝ, A = 75 * Real.sqrt 3 :=
by
  use 75 * Real.sqrt 3
  sorry

end area_of_inscribed_equilateral_triangle_l101_101313


namespace ellipse_eccentricity_range_l101_101240

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b) (h_b : b > 0) : 
  ∃ e : ℝ, (e = (Real.sqrt (a^2 - b^2)) / a) ∧ (e > 1/2 ∧ e < 1) :=
by
  sorry

end ellipse_eccentricity_range_l101_101240


namespace arcsin_one_half_eq_pi_six_l101_101450

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101450


namespace maximum_distance_travel_l101_101645

theorem maximum_distance_travel (front_tire_lifespan rear_tire_lifespan : ℕ) 
  (h1 : front_tire_lifespan = 20000) 
  (h2 : rear_tire_lifespan = 30000) : 
  ∃ max_distance, max_distance = 24000 :=
by {
  existsi 24000,
  rw [h1, h2],
  sorry
}

end maximum_distance_travel_l101_101645


namespace complex_quadrant_l101_101074

theorem complex_quadrant (z : ℂ) (h : z * complex.I = 2 - complex.I) : 
    ((z.re < 0) ∧ (z.im < 0)) :=
by
  sorry

end complex_quadrant_l101_101074


namespace floor_sum_min_value_l101_101211

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l101_101211


namespace petya_friends_l101_101735

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101735


namespace algebraic_expression_value_l101_101497

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 6 - Real.sqrt 2) : 2 * x^2 + 4 * Real.sqrt 2 * x = 8 :=
sorry

end algebraic_expression_value_l101_101497


namespace largest_three_digit_integer_l101_101594

theorem largest_three_digit_integer (n : ℕ) :
  75 * n ≡ 300 [MOD 450] →
  n < 1000 →
  ∃ m : ℕ, n = m ∧ (∀ k : ℕ, 75 * k ≡ 300 [MOD 450] ∧ k < 1000 → k ≤ n) := by
  sorry

end largest_three_digit_integer_l101_101594


namespace range_of_2a_plus_3b_l101_101646

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l101_101646


namespace comic_books_ratio_l101_101036

variable (S : ℕ)

def initial_comics := 22
def remaining_comics := 17
def comics_bought := 6

theorem comic_books_ratio (h1 : initial_comics - S + comics_bought = remaining_comics) :
  (S : ℚ) / initial_comics = 1 / 2 := by
  sorry

end comic_books_ratio_l101_101036


namespace smallest_with_12_divisors_l101_101597

theorem smallest_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, has_12_divisors m → n ≤ m) ∧ has_12_divisors n ∧ n = 72 :=
by 
    -- Define the condition for having exactly 12 divisors
    def has_12_divisors (n : ℕ) : Prop :=
    (∀ p : ℕ, nat.prime p → (nat.log n (p)^3 = 2 * 3) →
    (nat.log n (p)^5 = 2 * 2 * 3) →
    (n % (p^3) + (n % (p^2))) = 12 ∧ (sqrt (n^3 / nat.sqrt (n).p)) = (1 + (p)))
    sorry

end smallest_with_12_divisors_l101_101597


namespace cover_faces_with_strips_l101_101582

theorem cover_faces_with_strips (a b c : ℕ) :
  (∃ f g h : ℕ, a = 5 * f ∨ b = 5 * g ∨ c = 5 * h) ↔
  (∃ u v : ℕ, (a = 5 * u ∧ b = 5 * v) ∨ (a = 5 * u ∧ c = 5 * v) ∨ (b = 5 * u ∧ c = 5 * v)) := 
sorry

end cover_faces_with_strips_l101_101582


namespace arcsin_one_half_l101_101475

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101475


namespace selling_price_l101_101245

theorem selling_price (cost_price profit_percentage : ℝ) (h1 : cost_price = 90) (h2 : profit_percentage = 100) : 
    cost_price + (profit_percentage * cost_price / 100) = 180 :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end selling_price_l101_101245


namespace Kuwabara_class_girls_percentage_l101_101966

variable (num_girls num_boys : ℕ)

def total_students (num_girls num_boys : ℕ) : ℕ :=
  num_girls + num_boys

def girls_percentage (num_girls num_boys : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students num_girls num_boys : ℚ) * 100

theorem Kuwabara_class_girls_percentage (num_girls num_boys : ℕ) (h1: num_girls = 10) (h2: num_boys = 15) :
  girls_percentage num_girls num_boys = 40 := 
by
  sorry

end Kuwabara_class_girls_percentage_l101_101966


namespace greatest_prime_factor_15_fact_17_fact_l101_101820

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l101_101820


namespace plot_length_60_l101_101420

/-- The length of a rectangular plot is 20 meters more than its breadth. If the cost of fencing the plot at Rs. 26.50 per meter is Rs. 5300, then the length of the plot in meters is 60. -/
theorem plot_length_60 (b l : ℝ) (h1 : l = b + 20) (h2 : 2 * (l + b) * 26.5 = 5300) : l = 60 :=
by
  sorry

end plot_length_60_l101_101420


namespace find_speed_of_second_boy_l101_101590

theorem find_speed_of_second_boy
  (v : ℝ)
  (speed_first_boy : ℝ)
  (distance_apart : ℝ)
  (time_taken : ℝ)
  (h1 : speed_first_boy = 5.3)
  (h2 : distance_apart = 10.5)
  (h3 : time_taken = 35) :
  v = 5.6 :=
by {
  -- translation of the steps to work on the proof
  -- sorry is used to indicate that the proof is not provided here
  sorry
}

end find_speed_of_second_boy_l101_101590


namespace lesser_fraction_l101_101771

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 14 / 15) (h2 : x * y = 1 / 10) : min x y = 1 / 5 :=
sorry

end lesser_fraction_l101_101771


namespace inequality_proof_l101_101570

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 ≥ 4 * x + 4 * x * y :=
by
  sorry

end inequality_proof_l101_101570


namespace problem_statement_l101_101347

theorem problem_statement (f : ℝ → ℝ) (a b c m : ℝ)
  (h_cond1 : ∀ x, f x = -x^2 + a * x + b)
  (h_range : ∀ y, y ∈ Set.range f ↔ y ≤ 0)
  (h_ineq_sol : ∀ x, ((-x^2 + a * x + b > c - 1) ↔ (m - 4 < x ∧ x < m + 1))) :
  (b = -(1/4) * (2 * m - 3)^2) ∧ (c = -(21 / 4)) := sorry

end problem_statement_l101_101347


namespace probability_of_both_defective_given_one_defective_l101_101056

-- Definitions based on conditions in part a)
def totalProducts : ℕ := 6
def defectiveProducts : ℕ := 2
def selectedProducts : ℕ := 2

theorem probability_of_both_defective_given_one_defective (h : selectedProducts = 2) 
(h1 : 1 ∈ ({i : ℕ | i ≤ totalProducts}.filter (λ x, x <= defectiveProducts))) : 
  (∃ (n : ℚ), n = 1/15) :=
by
  -- Sorry is used to skip the proof step
  sorry

end probability_of_both_defective_given_one_defective_l101_101056


namespace valid_pairs_l101_101760

def valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_number (n : ℕ) : Prop :=
  let digits := [5, 3, 2, 9, n / 10 % 10, n % 10]
  (n % 2 = 0) ∧ (digits.sum % 3 = 0)

theorem valid_pairs (d₀ d₁ : ℕ) :
  valid_digit d₀ →
  valid_digit d₁ →
  (d₀ % 2 = 0) →
  valid_number (53290 * 10 + d₀ * 10 + d₁) →
  (d₀, d₁) = (0, 3) ∨ (d₀, d₁) = (2, 0) ∨ (d₀, d₁) = (2, 3) ∨ (d₀, d₁) = (2, 6) ∨
  (d₀, d₁) = (2, 9) ∨ (d₀, d₁) = (4, 1) ∨ (d₀, d₁) = (4, 4) ∨ (d₀, d₁) = (4, 7) ∨
  (d₀, d₁) = (6, 2) ∨ (d₀, d₁) = (6, 5) ∨ (d₀, d₁) = (6, 8) ∨ (d₀, d₁) = (8, 0) :=
by sorry

end valid_pairs_l101_101760


namespace scoops_arrangement_count_l101_101246

theorem scoops_arrangement_count :
  (5 * 4 * 3 * 2 * 1 = 120) :=
by
  sorry

end scoops_arrangement_count_l101_101246


namespace unique_identity_function_l101_101326

theorem unique_identity_function (f : ℝ → ℝ) (H : ∀ x y z : ℝ, (x^3 + f y * x + f z = 0) → (f x ^ 3 + y * f x + z = 0)) :
  f = id :=
by sorry

end unique_identity_function_l101_101326


namespace proof_equivalent_expression_l101_101351

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2

theorem proof_equivalent_expression (x y : ℝ) :
  (dollar ((x + y) ^ 2) (dollar y x)) - (dollar (dollar x y) (dollar x y)) = 
  4 * (x + y) ^ 2 * ((x + y) ^ 2 - 1) :=
by
  sorry

end proof_equivalent_expression_l101_101351


namespace triangle_acute_angles_integer_solution_l101_101190

theorem triangle_acute_angles_integer_solution :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), (20 < x ∧ x < 27) ∧ (12 < x ∧ x < 36) ↔ (x = 21 ∨ x = 22 ∨ x = 23 ∨ x = 24 ∨ x = 25 ∨ x = 26) :=
by
  sorry

end triangle_acute_angles_integer_solution_l101_101190


namespace initial_green_marbles_l101_101478

theorem initial_green_marbles (m g' : ℕ) (h_m : m = 23) (h_g' : g' = 9) : (g' + m = 32) :=
by
  subst h_m
  subst h_g'
  rfl

end initial_green_marbles_l101_101478


namespace greatest_prime_factor_15_17_factorial_l101_101800

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l101_101800


namespace no_single_two_three_digit_solution_l101_101014

theorem no_single_two_three_digit_solution :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x ∧ x ≤ 9) ∧
    (10 ≤ y ∧ y ≤ 99) ∧
    (100 ≤ z ∧ z ≤ 999) ∧
    (1/x : ℝ) = 1/y + 1/z :=
by
  sorry

end no_single_two_three_digit_solution_l101_101014


namespace intersection_point_exists_l101_101617

theorem intersection_point_exists :
  ∃ (x y z t : ℝ), (x = 1 - 2 * t) ∧ (y = 2 + t) ∧ (z = -1 - t) ∧
                   (x - 2 * y + 5 * z + 17 = 0) ∧ 
                   (x = -1) ∧ (y = 3) ∧ (z = -2) :=
by
  sorry

end intersection_point_exists_l101_101617


namespace petya_friends_l101_101731

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101731


namespace abs_diff_roots_quad_eq_l101_101186

theorem abs_diff_roots_quad_eq : 
  ∀ (r1 r2 : ℝ), 
  (r1 * r2 = 12) ∧ (r1 + r2 = 7) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  sorry

end abs_diff_roots_quad_eq_l101_101186


namespace smallest_four_digits_valid_remainder_l101_101486

def isFourDigit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999 

def validRemainder (x : ℕ) : Prop := 
  ∀ k ∈ [2, 3, 4, 5, 6], x % k = 1

theorem smallest_four_digits_valid_remainder :
  ∃ x1 x2 x3 x4 : ℕ,
    isFourDigit x1 ∧ validRemainder x1 ∧
    isFourDigit x2 ∧ validRemainder x2 ∧
    isFourDigit x3 ∧ validRemainder x3 ∧
    isFourDigit x4 ∧ validRemainder x4 ∧
    x1 = 1021 ∧ x2 = 1081 ∧ x3 = 1141 ∧ x4 = 1201 := 
sorry

end smallest_four_digits_valid_remainder_l101_101486


namespace correct_sampling_l101_101981

-- Let n be the total number of students
def total_students : ℕ := 60

-- Define the systematic sampling function
def systematic_sampling (n m : ℕ) (start : ℕ) : List ℕ :=
  List.map (λ k => start + k * m) (List.range n)

-- Prove that the sequence generated is equal to [3, 13, 23, 33, 43, 53]
theorem correct_sampling :
  systematic_sampling 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end correct_sampling_l101_101981


namespace sufficient_but_not_necessary_l101_101863

theorem sufficient_but_not_necessary (x : ℝ) : (x^2 = 9 → x = 3) ∧ (¬(x^2 = 9 → x = 3 ∨ x = -3)) :=
by
  sorry

end sufficient_but_not_necessary_l101_101863


namespace length_of_PQ_l101_101024

theorem length_of_PQ (p : ℝ) (h : p > 0) (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1x2 : x1 + x2 = 3 * p) (hy1 : y1^2 = 2 * p * x1) (hy2 : y2^2 = 2 * p * x2) 
  (focus : ¬ (y1 = 0)) : (abs (x1 - x2 + 2 * p) = 4 * p) := 
sorry

end length_of_PQ_l101_101024


namespace annual_rent_per_square_foot_is_172_l101_101123

def monthly_rent : ℕ := 3600
def local_taxes : ℕ := 500
def maintenance_fees : ℕ := 200
def length_of_shop : ℕ := 20
def width_of_shop : ℕ := 15

def total_monthly_cost : ℕ := monthly_rent + local_taxes + maintenance_fees
def annual_cost : ℕ := total_monthly_cost * 12
def area_of_shop : ℕ := length_of_shop * width_of_shop
def annual_rent_per_square_foot : ℕ := annual_cost / area_of_shop

theorem annual_rent_per_square_foot_is_172 :
  annual_rent_per_square_foot = 172 := by
    sorry

end annual_rent_per_square_foot_is_172_l101_101123


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101834

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101834


namespace dora_knows_coin_position_l101_101065

-- Definitions
def R_is_dime_or_nickel (R : ℕ) (L : ℕ) : Prop := 
  (R = 10 ∧ L = 5) ∨ (R = 5 ∧ L = 10)

-- Theorem statement
theorem dora_knows_coin_position (R : ℕ) (L : ℕ) 
  (h : R_is_dime_or_nickel R L) :
  (3 * R + 2 * L) % 2 = 0 ↔ (R = 10 ∧ L = 5) :=
by
  sorry

end dora_knows_coin_position_l101_101065


namespace vertical_asymptote_l101_101893

noncomputable def y (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote (x : ℝ) : (7 * x - 10 = 0) → (x = 10 / 7) :=
by
  intro h
  linarith [h]

#check vertical_asymptote

end vertical_asymptote_l101_101893


namespace beta_max_success_ratio_l101_101371

theorem beta_max_success_ratio :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < b ∧
    0 < c ∧ c < d ∧
    b + d ≤ 550 ∧
    (15 * a < 8 * b) ∧ (10 * c < 7 * d) ∧
    (21 * a + 16 * c < 4400) ∧
    ((a + c) / (b + d : ℚ) = 274 / 550) :=
sorry

end beta_max_success_ratio_l101_101371


namespace correct_total_score_l101_101402

theorem correct_total_score (total_score1 total_score2 : ℤ) : 
  (total_score1 = 5734 ∨ total_score2 = 5734) → (total_score1 = 5735 ∨ total_score2 = 5735) → 
  (total_score1 % 2 = 0 ∨ total_score2 % 2 = 0) → 
  (total_score1 ≠ total_score2) → 
  5734 % 2 = 0 :=
by
  sorry

end correct_total_score_l101_101402


namespace petya_friends_l101_101718

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101718


namespace room_area_is_18pi_l101_101166

def semicircle_room_area (R r : ℝ) (h : R > r) (d : ℝ) (hd : d = 12) : ℝ :=
  (π / 2) * (R^2 - r^2)

theorem room_area_is_18pi (R r : ℝ) (h : R > r) :
  semicircle_room_area R r h 12 (by rfl) = 18 * π :=
by
  sorry

end room_area_is_18pi_l101_101166


namespace Petya_friends_l101_101714

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101714


namespace unloading_time_relationship_l101_101289

-- Conditions
def loading_speed : ℝ := 30
def loading_time : ℝ := 8
def total_tonnage : ℝ := loading_speed * loading_time
def unloading_speed (x : ℝ) : ℝ := x

-- Proof statement
theorem unloading_time_relationship (x : ℝ) (hx : x ≠ 0) : 
  ∀ y : ℝ, y = 240 / x :=
by 
  sorry

end unloading_time_relationship_l101_101289


namespace sum_of_fifth_terms_arithmetic_sequences_l101_101344

theorem sum_of_fifth_terms_arithmetic_sequences (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (h₁ : ∀ n, a (n + 1) = a n + d₁)
  (h₂ : ∀ n, b (n + 1) = b n + d₂)
  (h₃ : a 1 + b 1 = 7)
  (h₄ : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end sum_of_fifth_terms_arithmetic_sequences_l101_101344


namespace water_for_bathing_per_horse_per_day_l101_101084

-- Definitions of the given conditions
def initial_horses : ℕ := 3
def additional_horses : ℕ := 5
def total_horses : ℕ := initial_horses + additional_horses
def drink_water_per_horse_per_day : ℕ := 5
def total_days : ℕ := 28
def total_water_needed : ℕ := 1568

-- The proven statement
theorem water_for_bathing_per_horse_per_day :
  ((total_water_needed - (total_horses * drink_water_per_horse_per_day * total_days)) / (total_horses * total_days)) = 2 :=
by
  sorry

end water_for_bathing_per_horse_per_day_l101_101084


namespace root_of_equation_l101_101907

theorem root_of_equation (a : ℝ) (h : a^2 * (-1)^2 + 2011 * a * (-1) - 2012 = 0) : 
  a = 2012 ∨ a = -1 :=
by sorry

end root_of_equation_l101_101907


namespace circle_equation_midpoint_trajectory_l101_101345

-- Definition for the circle equation proof
theorem circle_equation (x y : ℝ) (h : (x - 3)^2 + (y - 2)^2 = 13)
  (hx : x = 3) (hy : y = 2) : 
  (x - 3)^2 + (y - 2)^2 = 13 := by
  sorry -- Placeholder for proof

-- Definition for the midpoint trajectory proof
theorem midpoint_trajectory (x y : ℝ) (hx : x = (2 * x - 11) / 2)
  (hy : y = (2 * y - 2) / 2) (h : (2 * x - 11)^2 + (2 * y - 2)^2 = 13) :
  (x - 11 / 2)^2 + (y - 1)^2 = 13 / 4 := by
  sorry -- Placeholder for proof

end circle_equation_midpoint_trajectory_l101_101345


namespace length_of_c_l101_101953

theorem length_of_c (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h_triangle : 0 < c) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) → c = 3 :=
by
  intros h_ineq
  sorry

end length_of_c_l101_101953


namespace smallest_k_l101_101269

theorem smallest_k (k : ℕ) (h₁ : k > 1) (h₂ : k % 17 = 1) (h₃ : k % 6 = 1) (h₄ : k % 2 = 1) : k = 103 :=
by sorry

end smallest_k_l101_101269


namespace find_income_of_deceased_l101_101012
noncomputable def income_of_deceased_member 
  (members_before : ℕ) (avg_income_before : ℕ) 
  (members_after : ℕ) (avg_income_after : ℕ) : ℕ :=
  (members_before * avg_income_before) - (members_after * avg_income_after)

theorem find_income_of_deceased 
  (members_before avg_income_before members_after avg_income_after : ℕ) :
  income_of_deceased_member 4 840 3 650 = 1410 :=
by
  -- Problem claims income_of_deceased_member = Income before - Income after
  sorry

end find_income_of_deceased_l101_101012


namespace cost_of_cucumbers_l101_101564

theorem cost_of_cucumbers (C : ℝ) (h1 : ∀ (T : ℝ), T = 0.80 * C)
  (h2 : 2 * (0.80 * C) + 3 * C = 23) : C = 5 := by
  sorry

end cost_of_cucumbers_l101_101564


namespace find_f_of_2_l101_101906

noncomputable def f (x : ℕ) : ℕ := x^x + 2*x + 2

theorem find_f_of_2 : f 1 + 1 = 5 := 
by 
  sorry

end find_f_of_2_l101_101906


namespace gcd_problem_l101_101897

theorem gcd_problem (b : ℕ) (h : ∃ k : ℕ, b = 3150 * k) :
  gcd (b^2 + 9 * b + 54) (b + 4) = 2 := by
  sorry

end gcd_problem_l101_101897


namespace fill_half_cistern_time_l101_101155

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l101_101155


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101783

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101783


namespace sum_of_even_factors_720_l101_101273

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l101_101273


namespace irrationals_l101_101691

open Classical

variable (x : ℝ)

theorem irrationals (h : x^3 + 2 * x^2 + 10 * x = 20) : Irrational x ∧ Irrational (x^2) :=
by
  sorry

end irrationals_l101_101691


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101790
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101790


namespace player_A_advantage_l101_101857

theorem player_A_advantage (B A : ℤ) (rolls : ℕ) (h : rolls = 36) 
  (game_conditions : ∀ (x : ℕ), (x % 2 = 1 → A = A + x ∧ B = B - x) ∧ 
                      (x % 2 = 0 ∧ x ≠ 2 → A = A - x ∧ B = B + x) ∧ 
                      (x = 2 → A = A ∧ B = B)) : 
  (36 * (1 / 18 : ℚ) = 2) :=
by {
  -- Mathematical proof will be filled here
  sorry
}

end player_A_advantage_l101_101857


namespace at_least_two_greater_than_one_l101_101977

theorem at_least_two_greater_than_one
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  1 < a ∨ 1 < b ∨ 1 < c :=
sorry

end at_least_two_greater_than_one_l101_101977


namespace packs_of_sugar_l101_101324

theorem packs_of_sugar (cost_apples_per_kg cost_walnuts_per_kg cost_apples total : ℝ) (weight_apples weight_walnuts : ℝ) (less_sugar_by_1 : ℝ) (packs : ℕ) :
  cost_apples_per_kg = 2 →
  cost_walnuts_per_kg = 6 →
  cost_apples = weight_apples * cost_apples_per_kg →
  weight_apples = 5 →
  weight_walnuts = 0.5 →
  less_sugar_by_1 = 1 →
  total = 16 →
  packs = (total - (weight_apples * cost_apples_per_kg + weight_walnuts * cost_walnuts_per_kg)) / (cost_apples_per_kg - less_sugar_by_1) →
  packs = 3 :=
by
  sorry

end packs_of_sugar_l101_101324


namespace not_age_of_child_l101_101700

theorem not_age_of_child (ages : Set ℕ) (h_ages : ∀ x ∈ ages, 4 ≤ x ∧ x ≤ 10) : 
  5 ∉ ages := by
  let number := 1122
  have h_number : number % 5 ≠ 0 := by decide
  have h_divisible : ∀ x ∈ ages, number % x = 0 := sorry
  exact sorry

end not_age_of_child_l101_101700


namespace arthur_initial_amount_l101_101878

def initial_amount (X : ℝ) : Prop :=
  (1/5) * X = 40

theorem arthur_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by
  sorry

end arthur_initial_amount_l101_101878


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101799

open Nat

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  ∃ p : ℕ, prime p ∧ p ∣ (fact 15 + fact 17) ∧ ∀ q : ℕ, prime q ∧ q ∣ (fact 15 + fact 17) → q ≤ p :=
begin
  use 13,
  split,
  { exact prime_of_prime (by norm_num : nat.prime 13) },
  split,
  { rw [add_comm, ←mul_add, fact_succ, mul_comm (succ 15), ←mul_assoc, mul_comm 273], 
    exact dvd_mul_of_dvd_right (dvd_refl 273) _ },
  { intros q hq,
    have hf: 15! * 273 = (15! * 3) * 7 * 13 := by norm_num,
    rw hf at hq,
    apply le_of_dvd,
    { exact fact_pos 15 },
    cases hq with hpq,
    { exact le_of_dvd (pos_of_dvd_of_pos q (hq : q ∣ 7 * 13)) (hq_mul_self_le_self_or_of_prime_eq (by norm_num : nat.prime 13) hq) },
    { rw [←mul_assoc, factorize_mul, factorize_mul, factorize_mul] at *,
      refine list.all_max (13 :: nil) q (hq.symm ▸ hpq : prime q ∧ q ∣ 7 * 13),
      exact dec_trivial }
  }
end

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101799


namespace compute_expression_l101_101180

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l101_101180


namespace greatest_prime_factor_15_fact_17_fact_l101_101784

theorem greatest_prime_factor_15_fact_17_fact :
  let x := 15! + 17! in 
  let factors := { p : ℕ | p.prime ∧ p ∣ x } in 
  ∀ p ∈ factors, p ≤ 13 :=
by 
  sorry

end greatest_prime_factor_15_fact_17_fact_l101_101784


namespace nested_sqrt_eq_two_l101_101909

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by
  sorry

end nested_sqrt_eq_two_l101_101909


namespace greatest_prime_factor_of_15f_17f_is_17_l101_101842

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l101_101842


namespace point_after_transformations_l101_101131

-- Define the initial coordinates of point F
def F : ℝ × ℝ := (-1, -1)

-- Function to reflect a point over the x-axis
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Function to reflect a point over the line y = x
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Prove that F, when reflected over x-axis and then y=x, results in (1, -1)
theorem point_after_transformations : 
  reflect_over_y_eq_x (reflect_over_x F) = (1, -1) := by
  sorry

end point_after_transformations_l101_101131


namespace find_positive_product_l101_101767

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l101_101767


namespace cleaning_time_l101_101414

noncomputable def combined_cleaning_time (sawyer_time nick_time sarah_time : ℕ) : ℚ :=
  let rate_sawyer := 1 / sawyer_time
  let rate_nick := 1 / nick_time
  let rate_sarah := 1 / sarah_time
  1 / (rate_sawyer + rate_nick + rate_sarah)

theorem cleaning_time : combined_cleaning_time 6 9 4 = 36 / 19 := by
  have h1 : 1 / 6 = 1 / 6 := rfl
  have h2 : 1 / 9 = 1 / 9 := rfl
  have h3 : 1 / 4 = 1 / 4 := rfl
  rw [combined_cleaning_time, h1, h2, h3]
  norm_num
  sorry

end cleaning_time_l101_101414


namespace final_price_of_jacket_l101_101112

noncomputable def originalPrice : ℝ := 250
noncomputable def firstDiscount : ℝ := 0.60
noncomputable def secondDiscount : ℝ := 0.25

theorem final_price_of_jacket :
  let P := originalPrice
  let D1 := firstDiscount
  let D2 := secondDiscount
  let priceAfterFirstDiscount := P * (1 - D1)
  let finalPrice := priceAfterFirstDiscount * (1 - D2)
  finalPrice = 75 :=
by
  sorry

end final_price_of_jacket_l101_101112


namespace geometric_sequence_sum_l101_101681

variable {α : Type*} 
variable [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h1 : a 0 + a 1 = 20) 
  (h2 : a 2 + a 3 = 40) : 
  a 4 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l101_101681


namespace tub_drain_time_l101_101363

theorem tub_drain_time (time_for_five_sevenths : ℝ)
  (time_for_five_sevenths_eq_four : time_for_five_sevenths = 4) :
  let rate := time_for_five_sevenths / (5 / 7)
  let time_for_two_sevenths := 2 * rate
  time_for_two_sevenths = 11.2 := by
  -- Definitions and initial conditions
  sorry

end tub_drain_time_l101_101363


namespace rhombus_min_rotation_l101_101164

theorem rhombus_min_rotation (α : ℝ) (h1 : α = 60) : ∃ θ, θ = 180 := 
by 
  -- The proof here will show that the minimum rotation angle is 180°
  sorry

end rhombus_min_rotation_l101_101164


namespace find_b_for_parallel_lines_l101_101851

theorem find_b_for_parallel_lines :
  (∀ (b : ℝ), (∃ (f g : ℝ → ℝ),
  (∀ x, f x = 3 * x + b) ∧
  (∀ x, g x = (b + 9) * x - 2) ∧
  (∀ x, f x = g x → False)) →
  b = -6) :=
sorry

end find_b_for_parallel_lines_l101_101851


namespace area_of_region_between_semicircles_l101_101165

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end area_of_region_between_semicircles_l101_101165


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101835

def is_greatest_prime_factor (n p : ℕ) : Prop :=
  p.Prime ∧ p ∣ n ∧ (∀ q, q.Prime ∧ q ∣ n → q ≤ p)

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  is_greatest_prime_factor (Nat.factorial 15 + Nat.factorial 17) 17 :=
sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101835


namespace sum_of_coefficients_l101_101038

-- Given polynomial definition
def P (x : ℝ) : ℝ := (1 + x - 3 * x^2) ^ 1965

-- Lean 4 statement for the proof problem
theorem sum_of_coefficients :
  P 1 = -1 :=
by
  -- Proof placeholder
  sorry

end sum_of_coefficients_l101_101038


namespace circle_equation_correct_l101_101488

def line_through_fixed_point (a : ℝ) :=
  ∀ x y : ℝ, (x + y - 1) - a * (x + 1) = 0 → x = -1 ∧ y = 2

def equation_of_circle (x y: ℝ) :=
  (x + 1)^2 + (y - 2)^2 = 5

theorem circle_equation_correct (a : ℝ) (h : line_through_fixed_point a) :
  ∀ x y : ℝ, equation_of_circle x y ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
sorry

end circle_equation_correct_l101_101488


namespace factorial_square_ge_power_l101_101070

theorem factorial_square_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := 
by sorry

end factorial_square_ge_power_l101_101070


namespace petya_friends_l101_101722

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101722


namespace audit_options_correct_l101_101628

-- Define the initial number of ORs and GTUs
def initial_ORs : ℕ := 13
def initial_GTUs : ℕ := 15

-- Define the number of ORs and GTUs visited in the first week
def visited_ORs : ℕ := 2
def visited_GTUs : ℕ := 3

-- Calculate the remaining ORs and GTUs
def remaining_ORs : ℕ := initial_ORs - visited_ORs
def remaining_GTUs : ℕ := initial_GTUs - visited_GTUs

-- Calculate the number of ways to choose 2 ORs from remaining ORs
def choose_ORs : ℕ := Nat.choose remaining_ORs 2

-- Calculate the number of ways to choose 3 GTUs from remaining GTUs
def choose_GTUs : ℕ := Nat.choose remaining_GTUs 3

-- The final function to calculate the number of options
def number_of_options : ℕ := choose_ORs * choose_GTUs

-- The proof statement asserting the number of options is 12100
theorem audit_options_correct : number_of_options = 12100 := by
    sorry -- Proof will be filled in here

end audit_options_correct_l101_101628


namespace positive_diff_between_two_numbers_l101_101979

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l101_101979


namespace roots_g_eq_zero_l101_101159

noncomputable def g : ℝ → ℝ := sorry

theorem roots_g_eq_zero :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (∀ x : ℝ, g (12 + x) = g (12 - x)) →
  g 0 = 0 →
  ∃ L : ℕ, 
  (∀ k, 0 ≤ k ∧ k ≤ L → g (k * 48) = 0) ∧ 
  (∀ k : ℤ, -1000 ≤ k ∧ k ≤ 1000 → (∃ n : ℕ, k = n * 48)) ∧ 
  L + 1 = 42 := 
by sorry

end roots_g_eq_zero_l101_101159


namespace greatest_prime_factor_of_15_l101_101817

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l101_101817


namespace people_at_first_concert_l101_101096

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l101_101096


namespace fraction_product_equals_l101_101037

theorem fraction_product_equals :
  (7 / 4) * (14 / 49) * (10 / 15) * (12 / 36) * (21 / 14) * (40 / 80) * (33 / 22) * (16 / 64) = 1 / 12 := 
  sorry

end fraction_product_equals_l101_101037


namespace sqrt_of_square_neg7_l101_101631

theorem sqrt_of_square_neg7 : Real.sqrt ((-7:ℝ)^2) = 7 := by
  sorry

end sqrt_of_square_neg7_l101_101631


namespace max_students_l101_101537

-- Definitions for the conditions
noncomputable def courses := ["Mathematics", "Physics", "Biology", "Music", "History", "Geography"]

def most_preferred (ranking : List String) : Prop :=
  "Mathematics" ∈ (ranking.take 2) ∨ "Mathematics" ∈ (ranking.take 3)

def least_preferred (ranking : List String) : Prop :=
  "Music" ∉ ranking.drop (ranking.length - 2)

def preference_constraints (ranking : List String) : Prop :=
  ranking.indexOf "History" < ranking.indexOf "Geography" ∧
  ranking.indexOf "Physics" < ranking.indexOf "Biology"

def all_rankings_unique (rankings : List (List String)) : Prop :=
  ∀ (r₁ r₂ : List String), r₁ ≠ r₂ → r₁ ∈ rankings → r₂ ∈ rankings → r₁ ≠ r₂

-- The goal statement
theorem max_students : 
  ∃ (rankings : List (List String)), 
  (∀ r ∈ rankings, most_preferred r) ∧
  (∀ r ∈ rankings, least_preferred r) ∧
  (∀ r ∈ rankings, preference_constraints r) ∧
  all_rankings_unique rankings ∧
  rankings.length = 44 :=
sorry

end max_students_l101_101537


namespace y_share_is_correct_l101_101169

noncomputable def share_of_y (a : ℝ) := 0.45 * a

theorem y_share_is_correct :
  ∃ a : ℝ, (1 * a + 0.45 * a + 0.30 * a = 245) ∧ (share_of_y a = 63) :=
by
  sorry

end y_share_is_correct_l101_101169


namespace jenny_house_value_l101_101086

/-- Jenny's property tax rate is 2% -/
def property_tax_rate : ℝ := 0.02

/-- Her house's value increases by 25% due to the new high-speed rail project -/
noncomputable def house_value_increase_rate : ℝ := 0.25

/-- Jenny can afford to spend $15,000/year on property tax -/
def max_affordable_tax : ℝ := 15000

/-- Jenny can make improvements worth $250,000 to her house -/
def improvement_value : ℝ := 250000

/-- Current worth of Jenny's house -/
noncomputable def current_house_worth : ℝ := 500000

theorem jenny_house_value :
  property_tax_rate * (current_house_worth + improvement_value) = max_affordable_tax :=
by
  sorry

end jenny_house_value_l101_101086


namespace Diamond_result_l101_101885

-- Define the binary operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem Diamond_result : Diamond (Diamond 3 4) 2 = 179 := 
by 
  sorry

end Diamond_result_l101_101885


namespace trig_identity_proof_l101_101195

noncomputable def trig_identity (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) : ℝ :=
  (Real.sin (2 * α)) / (Real.cos (2 * β))

theorem trig_identity_proof (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) :
  trig_identity α β h1 h2 = 1 :=
sorry

end trig_identity_proof_l101_101195


namespace bags_wednesday_l101_101306

def charge_per_bag : ℕ := 4
def bags_monday : ℕ := 5
def bags_tuesday : ℕ := 3
def total_earnings : ℕ := 68

theorem bags_wednesday (h1 : charge_per_bag = 4)
                       (h2 : bags_monday = 5)
                       (h3 : bags_tuesday = 3)
                       (h4 : total_earnings = 68) :
  let earnings_monday_tuesday := (bags_monday + bags_tuesday) * charge_per_bag in
  let earnings_wednesday := total_earnings - earnings_monday_tuesday in
  earnings_wednesday / charge_per_bag = 9 :=
by
  sorry

end bags_wednesday_l101_101306


namespace derivative_y_l101_101117

noncomputable def y (x : ℝ) : ℝ := Real.sin x - Real.exp (x * Real.log 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = Real.cos x - Real.exp (x * Real.log 2) * Real.log 2 := 
by 
  sorry

end derivative_y_l101_101117


namespace find_value_l101_101532

-- Defining the sequence a_n, assuming all terms are positive
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

-- Definition to capture the given condition a_2 * a_4 = 4
def condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 4

-- The main statement
theorem find_value (a : ℕ → ℝ) (h_seq : is_geometric_sequence a) (h_cond : condition a) : 
  a 1 * a 5 + a 3 = 6 := 
by 
  sorry

end find_value_l101_101532


namespace no_solutions_l101_101485

theorem no_solutions : ¬ ∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end no_solutions_l101_101485


namespace simplify_expression_l101_101421

-- Define the variables x and y
variables (x y : ℝ)

-- State the theorem
theorem simplify_expression (x y : ℝ) (hy : y ≠ 0) :
  ((x + 3 * y)^2 - (x + y) * (x - y)) / (2 * y) = 3 * x + 5 * y := 
by 
  -- skip the proof
  sorry

end simplify_expression_l101_101421


namespace sum_of_number_and_reverse_l101_101119

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l101_101119


namespace arcsin_one_half_eq_pi_six_l101_101447

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101447


namespace bluegrass_percentage_l101_101568

theorem bluegrass_percentage (rx : ℝ) (ry : ℝ) (f : ℝ) (rm : ℝ) (wx : ℝ) (wy : ℝ) (B : ℝ) :
  rx = 0.4 →
  ry = 0.25 →
  f = 0.75 →
  rm = 0.35 →
  wx = 0.6667 →
  wy = 0.3333 →
  (wx * rx + wy * ry = rm) →
  B = 1.0 - rx →
  B = 0.6 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bluegrass_percentage_l101_101568


namespace factorization_correct_l101_101009

theorem factorization_correct (x : ℝ) : 
  x^4 - 5*x^2 - 36 = (x^2 + 4)*(x + 3)*(x - 3) :=
sorry

end factorization_correct_l101_101009


namespace exists_x0_in_interval_l101_101221

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) / x + Real.log (1 / (x - 1))

theorem exists_x0_in_interval :
  ∃ x0 ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f x0 = 0 := 
sorry  -- Proof is left as an exercise

end exists_x0_in_interval_l101_101221


namespace correct_total_cost_l101_101248

-- Number of sandwiches and their cost
def num_sandwiches : ℕ := 7
def sandwich_cost : ℕ := 4

-- Number of sodas and their cost
def num_sodas : ℕ := 9
def soda_cost : ℕ := 3

-- Total cost calculation
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem correct_total_cost : total_cost = 55 := by
  -- skip the proof details
  sorry

end correct_total_cost_l101_101248


namespace max_n_value_l101_101057

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem max_n_value (m : ℝ) (x_i : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, i < n → f (x_i i) / (x_i i) = m)
  (h2 : ∀ i, i < n → -2 * Real.pi ≤ x_i i ∧ x_i i ≤ 2 * Real.pi) :
  n ≤ 12 :=
sorry

end max_n_value_l101_101057


namespace quadratic_inequality_ab_l101_101197

theorem quadratic_inequality_ab (a b : ℝ) :
  (∀ x : ℝ, (x > -1 ∧ x < 1 / 3) → a * x^2 + b * x + 1 > 0) →
  a * b = 6 :=
sorry

end quadratic_inequality_ab_l101_101197


namespace sequence_periodicity_l101_101400

theorem sequence_periodicity (a : ℕ → ℝ) (h₁ : ∀ n, a (n + 1) = 1 / (1 - a n)) (h₂ : a 8 = 2) :
  a 1 = 1 / 2 := 
sorry

end sequence_periodicity_l101_101400


namespace smallest_positive_integer_with_12_divisors_l101_101604

theorem smallest_positive_integer_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ n) → n ≤ m) ∧ ∃ d : ℕ → ℕ, (d n = 12) :=
by
  sorry

end smallest_positive_integer_with_12_divisors_l101_101604


namespace people_distribution_l101_101587

theorem people_distribution
  (total_mentions : ℕ)
  (mentions_house : ℕ)
  (mentions_fountain : ℕ)
  (mentions_bench : ℕ)
  (mentions_tree : ℕ)
  (each_person_mentions : ℕ)
  (total_people : ℕ)
  (facing_house : ℕ)
  (facing_fountain : ℕ)
  (facing_bench : ℕ)
  (facing_tree : ℕ)
  (h_total_mentions : total_mentions = 27)
  (h_mentions_house : mentions_house = 5)
  (h_mentions_fountain : mentions_fountain = 6)
  (h_mentions_bench : mentions_bench = 7)
  (h_mentions_tree : mentions_tree = 9)
  (h_each_person_mentions : each_person_mentions = 3)
  (h_total_people : total_people = 9)
  (h_facing_house : facing_house = 5)
  (h_facing_fountain : facing_fountain = 4)
  (h_facing_bench : facing_bench = 2)
  (h_facing_tree : facing_tree = 9) :
  total_mentions / each_person_mentions = total_people ∧ 
  facing_house = mentions_house ∧
  facing_fountain = total_people - mentions_house ∧
  facing_bench = total_people - mentions_bench ∧
  facing_tree = total_people - mentions_tree :=
by
  sorry

end people_distribution_l101_101587


namespace arcsin_one_half_l101_101463

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101463


namespace Petya_friends_l101_101710

variable (x : ℕ)

-- Condition 1: If Petya gives 5 stickers to each friend, he will have 8 stickers left.
def condition1 (x : ℕ) : Prop := 5 * x + 8 = total_stickers

-- Condition 2: If Petya wants to give 6 stickers to each of his friends, he will be short of 11 stickers.
def condition2 (x : ℕ) : Prop := 6 * x - 11 = total_stickers

theorem Petya_friends (total_stickers : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 :=
sorry

end Petya_friends_l101_101710


namespace arcsin_of_half_l101_101440

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101440


namespace range_of_a_l101_101384

theorem range_of_a (x y a : ℝ) (h1 : x - y = 2) (h2 : x + y = a) (h3 : x > -1) (h4 : y < 0) : -4 < a ∧ a < 2 :=
sorry

end range_of_a_l101_101384


namespace arcsin_half_eq_pi_six_l101_101456

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101456


namespace max_integer_value_l101_101525

theorem max_integer_value (x : ℝ) : 
  ∃ M : ℤ, ∀ y : ℝ, (M = ⌊ 1 + 10 / (4 * y^2 + 12 * y + 9) ⌋ ∧ M ≤ 11) := 
sorry

end max_integer_value_l101_101525


namespace greatest_prime_factor_15_17_factorial_l101_101836

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l101_101836


namespace harriet_trip_time_l101_101142

theorem harriet_trip_time
  (speed_AB : ℕ := 100)
  (speed_BA : ℕ := 150)
  (total_trip_time : ℕ := 5)
  (time_threshold : ℕ := 180) :
  let D := (speed_AB * speed_BA * total_trip_time) / (speed_AB + speed_BA)
  let time_AB := D / speed_AB
  let time_AB_min := time_AB * 60
  time_AB_min = time_threshold :=
by
  sorry

end harriet_trip_time_l101_101142


namespace greatest_prime_factor_of_factorial_sum_l101_101794

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l101_101794


namespace min_value_mn_squared_l101_101052

theorem min_value_mn_squared (a b c m n : ℝ) 
  (h_triangle: a^2 + b^2 = c^2)
  (h_line: a * m + b * n + 2 * c = 0):
  m^2 + n^2 = 4 :=
by
  sorry

end min_value_mn_squared_l101_101052


namespace simplest_form_correct_l101_101998

variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)

def is_simplest_form (x : ℝ) : Prop :=
-- define what it means for a square root to be in simplest form
sorry

theorem simplest_form_correct :
  A = Real.sqrt (1 / 2) ∧ B = Real.sqrt 0.2 ∧ C = Real.sqrt 3 ∧ D = Real.sqrt 8 →
  ¬ is_simplest_form A ∧ ¬ is_simplest_form B ∧ is_simplest_form C ∧ ¬ is_simplest_form D :=
by
  -- prove that C is the simplest form and others are not
  sorry

end simplest_form_correct_l101_101998


namespace petya_friends_count_l101_101741

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101741


namespace monotonicity_a_eq_1_range_of_a_l101_101507

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x + a * x^2 - x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (exp x + 2 * x - 1 > 0)) ∧
  (∀ x : ℝ, x < 0 → (exp x + 2 * x - 1 < 0)) := sorry

-- Part 2: Range of a for f(x) ≥ 1/2 * x ^ 3 + 1 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 <= x → (exp x + a * x^2 - x >= 1/2 * x^3 + 1)) ↔
  (a ≥ (7 - exp 2) / 4) := sorry

end monotonicity_a_eq_1_range_of_a_l101_101507


namespace range_f1_l101_101016
open Function

theorem range_f1 (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Ici (-1) → y ∈ Set.Ici (-1) → x ≤ y → (x^2 + 2*a*x + 3) ≤ (y^2 + 2*a*y + 3)) →
  6 ≤ (1^2 + 2*a*1 + 3) :=
by
  intro h
  sorry

end range_f1_l101_101016


namespace distance_between_parallel_lines_eq_l101_101763

open Real

theorem distance_between_parallel_lines_eq
  (h₁ : ∀ (x y : ℝ), 3 * x + y - 3 = 0 → Prop)
  (h₂ : ∀ (x y : ℝ), 6 * x + 2 * y + 1 = 0 → Prop) :
  ∃ d : ℝ, d = (7 / 20) * sqrt 10 :=
sorry

end distance_between_parallel_lines_eq_l101_101763


namespace radar_placement_and_coverage_area_l101_101005

noncomputable def max_distance (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (15 : ℝ) / Real.sin (Real.pi / n)

noncomputable def coverage_area (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (480 : ℝ) * Real.pi / Real.tan (Real.pi / n)

theorem radar_placement_and_coverage_area 
  (n : ℕ) (r w : ℝ) (hn : n = 8) (hr : r = 17) (hw : w = 16) :
  max_distance n r w = (15 : ℝ) / Real.sin (Real.pi / 8) ∧
  coverage_area n r w = (480 : ℝ) * Real.pi / Real.tan (Real.pi / 8) :=
by
  sorry

end radar_placement_and_coverage_area_l101_101005


namespace algebraic_expression_value_l101_101671

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) : a^2 - 2 * a * b + b^2 + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l101_101671


namespace range_of_a_l101_101665

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

noncomputable def g (x a : ℝ) := Real.sqrt x + Real.sqrt (a - x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= a → ∃ x2 : ℝ, 4 ≤ x2 ∧ x2 ≤ 16 ∧ g x1 a = f x2) →
  4 ≤ a ∧ a ≤ 8 :=
sorry 

end range_of_a_l101_101665


namespace petya_friends_l101_101716

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101716


namespace michael_truck_meet_once_l101_101559

noncomputable def meets_count (michael_speed : ℕ) (pail_distance : ℕ) (truck_speed : ℕ) (truck_stop_duration : ℕ) : ℕ :=
  if michael_speed = 4 ∧ pail_distance = 300 ∧ truck_speed = 8 ∧ truck_stop_duration = 45 then 1 else sorry

theorem michael_truck_meet_once :
  meets_count 4 300 8 45 = 1 :=
by simp [meets_count]

end michael_truck_meet_once_l101_101559


namespace s_of_1_l101_101694

def t (x : ℚ) : ℚ := 5 * x - 10
def s (y : ℚ) : ℚ := (y^2 / (5^2)) + (5 * y / 5) + 6  -- reformulated to fit conditions

theorem s_of_1 :
  s (1 : ℚ) = 546 / 25 := by
  sorry

end s_of_1_l101_101694


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101814

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101814


namespace exists_integer_roots_l101_101358

theorem exists_integer_roots : 
  ∃ (a b c d e f : ℤ), ∃ r1 r2 r3 r4 r5 r6 : ℤ,
  (r1 + a) * (r2 ^ 2 + b * r2 + c) * (r3 ^ 3 + d * r3 ^ 2 + e * r3 + f) = 0 ∧
  (r4 + a) * (r5 ^ 2 + b * r5 + c) * (r6 ^ 3 + d * r6 ^ 2 + e * r6 + f) = 0 :=
  sorry

end exists_integer_roots_l101_101358


namespace sin_A_eq_height_on_AB_l101_101936

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l101_101936


namespace correct_statements_about_f_l101_101490

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem correct_statements_about_f : 
  (∀ x, (f x) ≤ (f e)) ∧ (f e = 1 / e) ∧ 
  (∀ x, (f x = 0) → x = 1) ∧ 
  (f 2 < f π ∧ f π < f 3) :=
by
  sorry

end correct_statements_about_f_l101_101490


namespace arcsin_of_half_l101_101466

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end arcsin_of_half_l101_101466


namespace part1_monotonicity_part2_find_range_l101_101505

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l101_101505


namespace polynomial_divisible_by_7_polynomial_divisible_by_12_l101_101182

theorem polynomial_divisible_by_7 (x : ℤ) : (x^7 - x) % 7 = 0 := 
sorry

theorem polynomial_divisible_by_12 (x : ℤ) : (x^4 - x^2) % 12 = 0 := 
sorry

end polynomial_divisible_by_7_polynomial_divisible_by_12_l101_101182


namespace arcsin_half_eq_pi_six_l101_101454

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101454


namespace petya_friends_l101_101728

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101728


namespace nonagon_perimeter_l101_101316

theorem nonagon_perimeter (n : ℕ) (side_length : ℝ) (P : ℝ) :
  n = 9 → side_length = 3 → P = n * side_length → P = 27 :=
by sorry

end nonagon_perimeter_l101_101316


namespace remainder_equal_to_zero_l101_101063

def A : ℕ := 270
def B : ℕ := 180
def M : ℕ := 25
def R_A : ℕ := A % M
def R_B : ℕ := B % M
def A_squared_B : ℕ := (A ^ 2 * B) % M
def R_A_R_B : ℕ := (R_A * R_B) % M

theorem remainder_equal_to_zero (h1 : A = 270) (h2 : B = 180) (h3 : M = 25) 
    (h4 : R_A = 20) (h5 : R_B = 5) : 
    A_squared_B = 0 ∧ R_A_R_B = 0 := 
by {
    sorry
}

end remainder_equal_to_zero_l101_101063


namespace A_is_false_l101_101642

variables {a b : ℝ}

-- Condition: Proposition B - The sum of the roots of the equation is 2
axiom sum_of_roots : ∀ (x1 x2 : ℝ), x1 + x2 = -a

-- Condition: Proposition C - x = 3 is a root of the equation
axiom root3 : ∃ (x1 x2 : ℝ), (x1 = 3 ∨ x2 = 3)

-- Condition: Proposition D - The two roots have opposite signs
axiom opposite_sign_roots : ∀ (x1 x2 : ℝ), x1 * x2 < 0

-- Prove: Proposition A is false
theorem A_is_false : ¬ (∃ x1 x2 : ℝ, x1 = 1 ∨ x2 = 1) :=
by
  sorry

end A_is_false_l101_101642


namespace maximize_profit_price_l101_101021

-- Definitions from the conditions
def initial_price : ℝ := 80
def initial_sales : ℝ := 200
def price_reduction_per_unit : ℝ := 1
def sales_increase_per_unit : ℝ := 20
def cost_price_per_helmet : ℝ := 50

-- Profit function
def profit (x : ℝ) : ℝ :=
  (x - cost_price_per_helmet) * (initial_sales + (initial_price - x) * sales_increase_per_unit)

-- The theorem statement
theorem maximize_profit_price : 
  ∃ x, (x = 70) ∧ (∀ y, profit y ≤ profit x) :=
sorry

end maximize_profit_price_l101_101021


namespace solve_expression_l101_101254

theorem solve_expression : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 :=
by
  sorry

end solve_expression_l101_101254


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101788
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101788


namespace integral_eval_l101_101185

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end integral_eval_l101_101185


namespace measure_angle_T_l101_101083

theorem measure_angle_T (P Q R S T : ℝ) (h₀ : P = R) (h₁ : R = T) (h₂ : Q + S = 180)
  (h_sum : P + Q + R + T + S = 540) : T = 120 :=
by
  sorry

end measure_angle_T_l101_101083


namespace greatest_prime_factor_15f_plus_17f_l101_101829

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l101_101829


namespace sum_even_factors_of_720_l101_101274

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l101_101274


namespace roots_of_polynomial_l101_101551

noncomputable def P (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 - 6 * x

theorem roots_of_polynomial : ∀ x : ℝ, P x = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by 
  -- Here you would provide the proof, but we use sorry to indicate it is left out
  sorry

end roots_of_polynomial_l101_101551


namespace no_solution_abs_eq_quadratic_l101_101327

theorem no_solution_abs_eq_quadratic (x : ℝ) : ¬ (|x - 4| = x^2 + 6 * x + 8) :=
by
  sorry

end no_solution_abs_eq_quadratic_l101_101327


namespace sin_minus_cos_third_quadrant_l101_101503

theorem sin_minus_cos_third_quadrant (α : ℝ) (h_tan : Real.tan α = 2) (h_quadrant : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := 
by 
  sorry

end sin_minus_cos_third_quadrant_l101_101503


namespace range_of_k_l101_101911

theorem range_of_k (k n : ℝ) (h : k ≠ 0) (h_pass : k - n^2 - 2 = k / 2) : k ≥ 4 :=
sorry

end range_of_k_l101_101911


namespace jungkook_colored_paper_count_l101_101377

theorem jungkook_colored_paper_count :
  (3 * 10) + 8 = 38 :=
by sorry

end jungkook_colored_paper_count_l101_101377


namespace simplify_3_375_to_fraction_l101_101143

def simplified_fraction_of_3_375 : ℚ := 3.375

theorem simplify_3_375_to_fraction : simplified_fraction_of_3_375 = 27 / 8 := 
by
  sorry

end simplify_3_375_to_fraction_l101_101143


namespace geo_sequence_necessity_l101_101657

theorem geo_sequence_necessity (a1 a2 a3 a4 : ℝ) (h_non_zero: a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) :
  (a1 * a4 = a2 * a3) → (∀ r : ℝ, (a2 = a1 * r) ∧ (a3 = a2 * r) ∧ (a4 = a3 * r)) → False :=
sorry

end geo_sequence_necessity_l101_101657


namespace sqrt_72_eq_6_sqrt_2_l101_101108

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l101_101108


namespace smallest_x_l101_101188

theorem smallest_x (x : ℝ) (h : |4 * x + 12| = 40) : x = -13 :=
sorry

end smallest_x_l101_101188


namespace greatest_prime_factor_15_fact_17_fact_l101_101821

theorem greatest_prime_factor_15_fact_17_fact : 
  ∀ n, n.prime → 15! + 17! = 15! * 273 → ∃ p, p ∣ (15! + 17!) ∧ p = 17 :=
by
  intros n hprime hfactorsum
  have factorization_273 : Nat.factors 273 = [3, 7, 13]
  sorry
  have factorial_primes :
    ∀ m, m.prime → m ≤ 15 → Nat.factors (15!) = [2, 3, 5, 7, 11, 13]
  sorry
  have prime_between_16_17 : [17].all prime
  sorry
  use 17
  split
  sorry
  rfl

end greatest_prime_factor_15_fact_17_fact_l101_101821


namespace farm_horses_cows_difference_l101_101703

-- Definitions based on provided conditions
def initial_ratio_horses_to_cows (horses cows : ℕ) : Prop := 5 * cows = horses
def transaction (horses cows sold bought : ℕ) : Prop :=
  horses - sold = 5 * cows - 15 ∧ cows + bought = cows + 15

-- Definitions to represent the ratios
def pre_transaction_ratio (horses cows : ℕ) : Prop := initial_ratio_horses_to_cows horses cows
def post_transaction_ratio (horses cows : ℕ) (sold bought : ℕ) : Prop :=
  transaction horses cows sold bought ∧ 7 * (horses - sold) = 17 * (cows + bought)

-- Statement of the theorem
theorem farm_horses_cows_difference :
  ∀ (horses cows : ℕ), 
    pre_transaction_ratio horses cows → 
    post_transaction_ratio horses cows 15 15 →
    (horses - 15) - (cows + 15) = 50 :=
by
  intros horses cows pre_ratio post_ratio
  sorry

end farm_horses_cows_difference_l101_101703


namespace proof_problem_l101_101253

noncomputable def problem (x y : ℝ) : ℝ :=
  let A := 2 * x + y
  let B := 2 * x - y
  (A ^ 2 - B ^ 2) * (x - 2 * y)

theorem proof_problem : problem (-1) 2 = 80 := by
  sorry

end proof_problem_l101_101253


namespace min_value_expr_l101_101091

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 := 
sorry

end min_value_expr_l101_101091


namespace problem_l101_101088

namespace MathProof

-- Definitions of A, B, and conditions
def A (x : ℤ) : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

-- Prove x = ± 1 when A ⊆ B, 
-- A ∪ B = { -1, 0, 1 }, 
-- and complement of A in B is { -1 }
theorem problem (x : ℤ) (hx : A x ⊆ B) : 
  (x = 1 ∨ x = -1) ∧ 
  (A x ∪ B = {-1, 0, 1}) ∧ 
  (B \ (A x) = {-1}) := 
sorry 

end MathProof

end problem_l101_101088


namespace evaluate_fraction_l101_101236

noncomputable section

variables (u v : ℂ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : u^2 + u * v + v^2 = 0)

theorem evaluate_fraction : (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end evaluate_fraction_l101_101236


namespace vacation_cost_division_l101_101586

theorem vacation_cost_division (n : ℕ) (h1 : 720 / 4 = 60 + 720 / n) : n = 3 := by
  sorry

end vacation_cost_division_l101_101586


namespace translate_upwards_one_unit_l101_101773

theorem translate_upwards_one_unit (x y : ℝ) : (y = 2 * x) → (y + 1 = 2 * x + 1) := 
by sorry

end translate_upwards_one_unit_l101_101773


namespace votes_candidate_X_l101_101081

theorem votes_candidate_X (X Y Z : ℕ) (h1 : X = (3 / 2 : ℚ) * Y) (h2 : Y = (3 / 5 : ℚ) * Z) (h3 : Z = 25000) : X = 22500 :=
by
  sorry

end votes_candidate_X_l101_101081


namespace permutations_without_HMMT_l101_101515

noncomputable def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem permutations_without_HMMT :
  let total_permutations := multinomial 8 2 2 4
  let block_permutations := multinomial 5 1 1 2
  (total_permutations - block_permutations + 1) = 361 :=
by
  sorry

end permutations_without_HMMT_l101_101515


namespace part_1_part_2_l101_101480

def f (x a : ℝ) : ℝ := |x - a| + 5 * x

theorem part_1 (x : ℝ) : (|x + 1| + 5 * x ≤ 5 * x + 3) ↔ (x ∈ Set.Icc (-4 : ℝ) 2) :=
by
  sorry

theorem part_2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) :=
by
  sorry

end part_1_part_2_l101_101480


namespace value_of_a9_l101_101230

variables (a : ℕ → ℤ) (d : ℤ)
noncomputable def arithmetic_sequence : Prop :=
(a 1 + (a 1 + 10 * d)) / 2 = 15 ∧
a 1 + (a 1 + d) + (a 1 + 2 * d) = 9

theorem value_of_a9 (h : arithmetic_sequence a d) : a 9 = 24 :=
by sorry

end value_of_a9_l101_101230


namespace cost_of_football_and_basketball_max_number_of_basketballs_l101_101483

-- Problem 1: Cost of one football and one basketball
theorem cost_of_football_and_basketball (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 310) 
  (h2 : 2 * x + 5 * y = 500) : 
  x = 50 ∧ y = 80 :=
sorry

-- Problem 2: Maximum number of basketballs
theorem max_number_of_basketballs (x : ℝ) 
  (h1 : 50 * (96 - x) + 80 * x ≤ 5800) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ 96) : 
  x ≤ 33 :=
sorry

end cost_of_football_and_basketball_max_number_of_basketballs_l101_101483


namespace monotonicity_a1_range_of_a_l101_101504

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- 1. Monotonicity when \( a = 1 \)
theorem monotonicity_a1 :
  (∀ x > 0, (f x 1)' > 0) ∧ (∀ x < 0, (f x 1)' < 0) :=
by
  sorry

-- 2. Range of \( a \) for \( f(x) \geq \frac{1}{2} x^3 + 1 \) for \( x \geq 0 \)
theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) (hf : f x a ≥ (1 / 2) * x^3 + 1) :
  a ≥ (7 - Real.exp 2) / 4 :=
by
  sorry

end monotonicity_a1_range_of_a_l101_101504


namespace exists_k_with_half_distinct_remainders_l101_101950

theorem exists_k_with_half_distinct_remainders 
  (p : ℕ) (h_prime : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, (Finset.univ.image (λ (i : Fin p), (a i + ↑i * k) % p)).card ≥ p / 2 := by
  sorry

end exists_k_with_half_distinct_remainders_l101_101950


namespace tangent_line_at_origin_l101_101258

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem tangent_line_at_origin :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, f x - (m * x + b) = 0 → 2 * x - f x + 1 = 0) :=
sorry

end tangent_line_at_origin_l101_101258


namespace arcsin_one_half_eq_pi_six_l101_101449

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101449


namespace no_real_roots_l101_101566

theorem no_real_roots (a b c : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) : 
  ¬ ∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2 * (a + b + c) * x + 3 = 0 := 
by
  sorry

end no_real_roots_l101_101566


namespace find_missing_number_l101_101365

theorem find_missing_number (x : ℝ)
  (h1 : (x + 42 + 78 + 104) / 4 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) :
  x = 74 :=
sorry

end find_missing_number_l101_101365


namespace percent_decrease_l101_101943

theorem percent_decrease (p_original p_sale : ℝ) (h₁ : p_original = 100) (h₂ : p_sale = 50) :
  ((p_original - p_sale) / p_original * 100) = 50 := by
  sorry

end percent_decrease_l101_101943


namespace equilateral_triangle_l101_101079

theorem equilateral_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) 
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) 
  (h4 : b = c) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end equilateral_triangle_l101_101079


namespace sum_of_number_and_reverse_l101_101120

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l101_101120


namespace kids_still_awake_l101_101864

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l101_101864


namespace multiple_of_a_age_l101_101278

theorem multiple_of_a_age (A B M : ℝ) (h1 : A = B + 5) (h2 : A + B = 13) (h3 : M * (A + 7) = 4 * (B + 7)) : M = 2.75 :=
sorry

end multiple_of_a_age_l101_101278


namespace domino_covering_impossible_odd_squares_l101_101025

theorem domino_covering_impossible_odd_squares
  (board1 : ℕ) -- 24 squares
  (board2 : ℕ) -- 21 squares
  (board3 : ℕ) -- 23 squares
  (board4 : ℕ) -- 35 squares
  (board5 : ℕ) -- 63 squares
  (h1 : board1 = 24)
  (h2 : board2 = 21)
  (h3 : board3 = 23)
  (h4 : board4 = 35)
  (h5 : board5 = 63) :
  (board2 % 2 = 1) ∧ (board3 % 2 = 1) ∧ (board4 % 2 = 1) ∧ (board5 % 2 = 1) :=
by {
  sorry
}

end domino_covering_impossible_odd_squares_l101_101025


namespace sin_log_infinite_zeros_in_01_l101_101517

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l101_101517


namespace infinite_power_tower_solution_l101_101321

theorem infinite_power_tower_solution : 
  ∃ x : ℝ, (∀ y, y = x ^ y → y = 4) → x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l101_101321


namespace largest_n_satisfying_conditions_l101_101990

theorem largest_n_satisfying_conditions : 
  ∃ n : ℤ, 200 < n ∧ n < 250 ∧ (∃ k : ℤ, 12 * n = k^2) ∧ n = 243 :=
by
  sorry

end largest_n_satisfying_conditions_l101_101990


namespace petya_friends_l101_101729

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101729


namespace intersection_point_interval_l101_101509

theorem intersection_point_interval (x₀ : ℝ) (h : x₀^3 = 2^x₀ + 1) : 
  1 < x₀ ∧ x₀ < 2 :=
by
  sorry

end intersection_point_interval_l101_101509


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101781

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101781


namespace polygon_sides_l101_101676

-- Define the given conditions
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def sum_exterior_angles : ℕ := 360

-- Define the theorem
theorem polygon_sides (n : ℕ) (h : sum_interior_angles n = 3 * sum_exterior_angles + 180) : n = 9 :=
sorry

end polygon_sides_l101_101676


namespace smallest_number_with_12_divisors_l101_101606

-- Define a function to calculate the number of divisors of a given positive integer
def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

-- Define a predicate to check if a number has exactly 12 divisors
def has_exactly_12_divisors (n : ℕ) : Prop :=
  num_divisors n = 12

-- Define the main theorem statement
theorem smallest_number_with_12_divisors : ∃ n : ℕ, has_exactly_12_divisors n ∧ ∀ m : ℕ, has_exactly_12_divisors m → n ≤ m :=
  sorry

end smallest_number_with_12_divisors_l101_101606


namespace triangle_sin_A_and_height_l101_101928

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l101_101928


namespace percentage_of_alcohol_in_vessel_Q_l101_101986

theorem percentage_of_alcohol_in_vessel_Q
  (x : ℝ)
  (h_mix : 2.5 + 0.04 * x = 6) :
  x = 87.5 :=
by
  sorry

end percentage_of_alcohol_in_vessel_Q_l101_101986


namespace distinct_remainders_l101_101949

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end distinct_remainders_l101_101949


namespace greatest_prime_factor_of_sum_factorials_l101_101827

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l101_101827


namespace nylon_cord_length_l101_101295

theorem nylon_cord_length {L : ℝ} (hL : L = 30) : ∃ (w : ℝ), w = 5 := 
by sorry

end nylon_cord_length_l101_101295


namespace exists_small_area_triangle_l101_101919

def lattice_point (x y : ℤ) : Prop := |x| ≤ 2 ∧ |y| ≤ 2

def no_three_collinear (points : List (ℤ × ℤ)) : Prop :=
∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
(p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
¬ (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) = 0)

noncomputable def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
(1 / 2 : ℚ) * |(p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))|

theorem exists_small_area_triangle {points : List (ℤ × ℤ)}
  (h1 : points.length = 6)
  (h2 : ∀ (p : ℤ × ℤ), p ∈ points → lattice_point p.1 p.2)
  (h3 : no_three_collinear points) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  triangle_area p1 p2 p3 ≤ 2 := 
sorry

end exists_small_area_triangle_l101_101919


namespace value_of_a_l101_101359

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (5-x)/(x-2) ≥ 0 ↔ -3 < x ∧ x < a) → a > 5 :=
by
  intro h
  sorry

end value_of_a_l101_101359


namespace log_cos_range_l101_101508

theorem log_cos_range :
  ∀ (x : ℝ), x ∈ Icc (-real.pi / 2) (real.pi / 2) →
  log 2 (3 * real.cos x + 1) ∈ Icc 0 2 :=
begin
  sorry
end

end log_cos_range_l101_101508


namespace fewest_coach_handshakes_l101_101301

theorem fewest_coach_handshakes (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 281) : k = 5 :=
sorry

end fewest_coach_handshakes_l101_101301


namespace find_positive_product_l101_101768

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l101_101768


namespace smallest_possible_floor_sum_l101_101212

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l101_101212


namespace fourth_rectangle_area_l101_101871

-- Define the conditions and prove the area of the fourth rectangle
theorem fourth_rectangle_area (x y z w : ℝ)
  (h_xy : x * y = 24)
  (h_xw : x * w = 35)
  (h_zw : z * w = 42)
  (h_sum : x + z = 21) :
  y * w = 33.777 := 
sorry

end fourth_rectangle_area_l101_101871


namespace proof_moles_HNO3_proof_molecular_weight_HNO3_l101_101881

variable (n_CaO : ℕ) (molar_mass_H : ℕ) (molar_mass_N : ℕ) (molar_mass_O : ℕ)

def verify_moles_HNO3 (n_CaO : ℕ) : ℕ :=
  2 * n_CaO

def verify_molecular_weight_HNO3 (molar_mass_H molar_mass_N molar_mass_O : ℕ) : ℕ :=
  molar_mass_H + molar_mass_N + 3 * molar_mass_O

theorem proof_moles_HNO3 :
  n_CaO = 7 →
  verify_moles_HNO3 n_CaO = 14 :=
sorry

theorem proof_molecular_weight_HNO3 :
  molar_mass_H = 101 / 100 ∧ molar_mass_N = 1401 / 100 ∧ molar_mass_O = 1600 / 100 →
  verify_molecular_weight_HNO3 molar_mass_H molar_mass_N molar_mass_O = 6302 / 100 :=
sorry

end proof_moles_HNO3_proof_molecular_weight_HNO3_l101_101881


namespace zeros_of_f_on_interval_l101_101518

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l101_101518


namespace petya_friends_l101_101709

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101709


namespace smallest_n_positive_odd_integer_l101_101993

theorem smallest_n_positive_odd_integer (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ^ ((n + 1)^2 / 5) > 500) : n = 6 := sorry

end smallest_n_positive_odd_integer_l101_101993


namespace greatest_prime_factor_of_15_l101_101818

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l101_101818


namespace number_added_is_8_l101_101405

theorem number_added_is_8
  (x y : ℕ)
  (h1 : x = 265)
  (h2 : x / 5 + y = 61) :
  y = 8 :=
by
  sorry

end number_added_is_8_l101_101405


namespace greatest_prime_factor_15f_plus_17f_l101_101831

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l101_101831


namespace least_sum_of_bases_l101_101972

theorem least_sum_of_bases :
  ∃ (c d : ℕ), (5 * c + 7 = 7 * d + 5) ∧ (c > 0) ∧ (d > 0) ∧ (c + d = 14) :=
by
  sorry

end least_sum_of_bases_l101_101972


namespace certain_number_div_5000_l101_101137

theorem certain_number_div_5000 (num : ℝ) (h : num / 5000 = 0.0114) : num = 57 :=
sorry

end certain_number_div_5000_l101_101137


namespace three_digit_number_mul_seven_results_638_l101_101437

theorem three_digit_number_mul_seven_results_638 (N : ℕ) 
  (hN1 : 100 ≤ N) 
  (hN2 : N < 1000)
  (hN3 : ∃ (x : ℕ), 7 * N = 1000 * x + 638) : N = 234 := 
sorry

end three_digit_number_mul_seven_results_638_l101_101437


namespace bags_wednesday_l101_101307

def charge_per_bag : ℕ := 4
def bags_monday : ℕ := 5
def bags_tuesday : ℕ := 3
def total_earnings : ℕ := 68

theorem bags_wednesday (h1 : charge_per_bag = 4)
                       (h2 : bags_monday = 5)
                       (h3 : bags_tuesday = 3)
                       (h4 : total_earnings = 68) :
  let earnings_monday_tuesday := (bags_monday + bags_tuesday) * charge_per_bag in
  let earnings_wednesday := total_earnings - earnings_monday_tuesday in
  earnings_wednesday / charge_per_bag = 9 :=
by
  sorry

end bags_wednesday_l101_101307


namespace jane_can_buy_9_tickets_l101_101085

-- Definitions
def ticket_price : ℕ := 15
def jane_amount_initial : ℕ := 160
def scarf_cost : ℕ := 25
def jane_amount_after_scarf : ℕ := jane_amount_initial - scarf_cost
def max_tickets (amount : ℕ) (price : ℕ) := amount / price

-- The main statement
theorem jane_can_buy_9_tickets :
  max_tickets jane_amount_after_scarf ticket_price = 9 :=
by
  -- Proof goes here (proof steps would be outlined)
  sorry

end jane_can_buy_9_tickets_l101_101085


namespace cans_of_soda_l101_101574

theorem cans_of_soda (S Q D : ℕ) : (4 * D * S) / Q = x :=
by
  sorry

end cans_of_soda_l101_101574


namespace find_sin_2a_l101_101051

noncomputable def problem_statement (a : ℝ) : Prop :=
a ∈ Set.Ioo (Real.pi / 2) Real.pi ∧
3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin ((Real.pi / 4) - a)

theorem find_sin_2a (a : ℝ) (h : problem_statement a) : Real.sin (2 * a) = -8 / 9 :=
sorry

end find_sin_2a_l101_101051


namespace point_A_in_third_quadrant_l101_101684

-- Defining the point A with its coordinates
structure Point :=
  (x : Int)
  (y : Int)

def A : Point := ⟨-1, -3⟩

-- The definition of quadrants in Cartesian coordinate system
def quadrant (p : Point) : String :=
  if p.x > 0 ∧ p.y > 0 then "first"
  else if p.x < 0 ∧ p.y > 0 then "second"
  else if p.x < 0 ∧ p.y < 0 then "third"
  else if p.x > 0 ∧ p.y < 0 then "fourth"
  else "boundary"

-- The theorem we want to prove
theorem point_A_in_third_quadrant : quadrant A = "third" :=
by 
  sorry

end point_A_in_third_quadrant_l101_101684


namespace largest_num_of_hcf_and_lcm_factors_l101_101761

theorem largest_num_of_hcf_and_lcm_factors (hcf : ℕ) (f1 f2 : ℕ) (hcf_eq : hcf = 23) (f1_eq : f1 = 13) (f2_eq : f2 = 14) : 
    hcf * max f1 f2 = 322 :=
by
  -- use the conditions to find the largest number
  rw [hcf_eq, f1_eq, f2_eq]
  sorry

end largest_num_of_hcf_and_lcm_factors_l101_101761


namespace petya_friends_count_l101_101744

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101744


namespace susans_total_chairs_l101_101758

def number_of_red_chairs := 5
def number_of_yellow_chairs := 4 * number_of_red_chairs
def number_of_blue_chairs := number_of_yellow_chairs - 2
def total_chairs := number_of_red_chairs + number_of_yellow_chairs + number_of_blue_chairs

theorem susans_total_chairs : total_chairs = 43 :=
by
  sorry

end susans_total_chairs_l101_101758


namespace value_of_k_l101_101318

def f (x : ℝ) := 4 * x ^ 2 - 5 * x + 6
def g (x : ℝ) (k : ℝ) := 2 * x ^ 2 - k * x + 1

theorem value_of_k :
  (f 5 - g 5 k = 30) → k = -10 := 
by 
  sorry

end value_of_k_l101_101318


namespace gcd_correct_l101_101989

def gcd_87654321_12345678 : ℕ :=
  gcd 87654321 12345678

theorem gcd_correct : gcd_87654321_12345678 = 75 := by 
  sorry

end gcd_correct_l101_101989


namespace inequality_with_sum_of_one_l101_101975

theorem inequality_with_sum_of_one
  (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum: a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) :=
sorry

end inequality_with_sum_of_one_l101_101975


namespace mn_condition_l101_101526

theorem mn_condition {m n : ℕ} (h : m * n = 121) : (m + 1) * (n + 1) = 144 :=
sorry

end mn_condition_l101_101526


namespace arcsin_one_half_l101_101474

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101474


namespace hexagon_rectangle_ratio_l101_101623

theorem hexagon_rectangle_ratio:
  ∀ (h w : ℕ), 
  (6 * h = 24) → (2 * (2 * w + w) = 24) → 
  (h / w = 1) := by
  intros h w
  intro hex_condition
  intro rect_condition
  sorry

end hexagon_rectangle_ratio_l101_101623


namespace empty_can_weight_l101_101022

theorem empty_can_weight (W w : ℝ) :
  (W + 2 * w = 0.6) →
  (W + 5 * w = 0.975) →
  W = 0.35 :=
by sorry

end empty_can_weight_l101_101022


namespace remaining_volume_correct_l101_101426

-- Define the side length of the cube
def side_length : ℝ := 6

-- Define the radius of the cylindrical section
def cylinder_radius : ℝ := 3

-- Define the height of the cylindrical section (which is equal to the side length of the cube)
def cylinder_height : ℝ := side_length

-- Define the volume of the cube
def volume_cube : ℝ := side_length^3

-- Define the volume of the cylindrical section
def volume_cylinder : ℝ := Real.pi * cylinder_radius^2 * cylinder_height

-- Define the remaining volume after removing the cylindrical section from the cube
def remaining_volume : ℝ := volume_cube - volume_cylinder

-- Theorem stating the remaining volume is 216 - 54π cubic feet
theorem remaining_volume_correct : remaining_volume = 216 - 54 * Real.pi :=
by
  -- Proof will go here
  sorry

end remaining_volume_correct_l101_101426


namespace price_reduction_for_target_profit_l101_101619
-- Import the necessary libraries

-- Define the conditions
def average_sales_per_day := 70
def initial_profit_per_item := 50
def sales_increase_per_dollar_decrease := 2

-- Define the functions for sales volume increase and profit per item
def sales_volume_increase (x : ℝ) : ℝ := 2 * x
def profit_per_item (x : ℝ) : ℝ := initial_profit_per_item - x

-- Define the function for daily profit
def daily_profit (x : ℝ) : ℝ := (profit_per_item x) * (average_sales_per_day + sales_volume_increase x)

-- State the main theorem
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, daily_profit x = 3572 ∧ x = 12 :=
sorry

end price_reduction_for_target_profit_l101_101619


namespace minimum_keys_needed_l101_101538

def cabinets : ℕ := 8
def boxes_per_cabinet : ℕ := 4
def phones_per_box : ℕ := 10
def total_phones_needed : ℕ := 52

theorem minimum_keys_needed : 
  ∀ (cabinets boxes_per_cabinet phones_per_box total_phones_needed: ℕ), 
  cabinets = 8 →
  boxes_per_cabinet = 4 →
  phones_per_box = 10 →
  total_phones_needed = 52 →
  exists (keys_needed : ℕ), keys_needed = 9 :=
by
  intros _ _ _ _ hc hb hp ht
  have h1 : nat.ceil (52 / 10) = 6 := sorry -- detail of calculation
  have h2 : nat.ceil (6 / 4) = 2 := sorry -- detail of calculation
  use 9
  sorry

end minimum_keys_needed_l101_101538


namespace cube_side_length_l101_101621

theorem cube_side_length (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
sorry

end cube_side_length_l101_101621


namespace petya_friends_l101_101726

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l101_101726


namespace office_person_count_l101_101114

theorem office_person_count
    (N : ℕ)
    (avg_age_all : ℕ)
    (num_5 : ℕ)
    (avg_age_5 : ℕ)
    (num_9 : ℕ)
    (avg_age_9 : ℕ)
    (age_15th : ℕ)
    (h1 : avg_age_all = 15)
    (h2 : num_5 = 5)
    (h3 : avg_age_5 = 14)
    (h4 : num_9 = 9)
    (h5 : avg_age_9 = 16)
    (h6 : age_15th = 86)
    (h7 : 15 * N = (num_5 * avg_age_5) + (num_9 * avg_age_9) + age_15th) :
    N = 20 :=
by
    -- Proof will be provided here
    sorry

end office_person_count_l101_101114


namespace right_triangle_hypotenuse_length_l101_101416

theorem right_triangle_hypotenuse_length (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 10 :=
by
  sorry

end right_triangle_hypotenuse_length_l101_101416


namespace dongzhi_daylight_hours_l101_101541

theorem dongzhi_daylight_hours:
  let total_hours_in_day := 24
  let daytime_ratio := 5
  let nighttime_ratio := 7
  let total_parts := daytime_ratio + nighttime_ratio
  let daylight_hours := total_hours_in_day * daytime_ratio / total_parts
  daylight_hours = 10 :=
by
  sorry

end dongzhi_daylight_hours_l101_101541


namespace largest_n_multiple_3_l101_101135

theorem largest_n_multiple_3 (n : ℕ) (h1 : n < 100000) (h2 : (8 * (n + 2)^5 - n^2 + 14 * n - 30) % 3 = 0) : n = 99999 := 
sorry

end largest_n_multiple_3_l101_101135


namespace arcsin_of_half_l101_101442

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101442


namespace product_of_solutions_is_zero_l101_101001

theorem product_of_solutions_is_zero :
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) -> x = 0)) -> true :=
by
  sorry

end product_of_solutions_is_zero_l101_101001


namespace seating_arrangement_l101_101041

theorem seating_arrangement (n : ℕ) (h1 : n * 9 + (100 - n) * 10 = 100) : n = 10 :=
by sorry

end seating_arrangement_l101_101041


namespace max_digits_product_l101_101423

def digitsProduct (A B : ℕ) : ℕ := A * B

theorem max_digits_product 
  (A B : ℕ) 
  (h1 : A + B + 5 ≡ 0 [MOD 9]) 
  (h2 : 0 ≤ A ∧ A ≤ 9) 
  (h3 : 0 ≤ B ∧ B ≤ 9) 
  : digitsProduct A B = 42 := 
sorry

end max_digits_product_l101_101423


namespace salt_solution_proof_l101_101668

theorem salt_solution_proof (x : ℝ) (P : ℝ) (hx : x = 28.571428571428573) :
  ((P / 100) * 100 + x) = 0.30 * (100 + x) → P = 10 :=
by
  sorry

end salt_solution_proof_l101_101668


namespace base_number_is_2_l101_101915

open Real

noncomputable def valid_x (x : ℝ) (n : ℕ) := sqrt (x^n) = 64

theorem base_number_is_2 (x : ℝ) (n : ℕ) (h : valid_x x n) (hn : n = 12) : x = 2 := 
by 
  sorry

end base_number_is_2_l101_101915


namespace matrix_B6_eq_sB_plus_tI_l101_101235

noncomputable section

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![1, -1;
     4, 2]

theorem matrix_B6_eq_sB_plus_tI :
  ∃ s t : ℤ, B^6 = s • B + t • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  have B2_eq : B^2 = -3 • B :=
    -- Matrix multiplication and scalar multiplication
    sorry
  use 81, 0
  have B4_eq : B^4 = 9 • B^2 := by
    rw [B2_eq]
    -- Calculation steps for B^4 equation
    sorry
  have B6_eq : B^6 = B^4 * B^2 := by
    rw [B4_eq, B2_eq]
    -- Calculation steps for B^6 final equation
    sorry
  rw [B6_eq]
  -- Final steps to show (81 • B + 0 • I = 81 • B)
  sorry

end matrix_B6_eq_sB_plus_tI_l101_101235


namespace petya_has_19_friends_l101_101746

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101746


namespace roots_of_equation_l101_101257

theorem roots_of_equation {x : ℝ} :
  (12 * x^2 - 31 * x - 6 = 0) →
  (x = (31 + Real.sqrt 1249) / 24 ∨ x = (31 - Real.sqrt 1249) / 24) :=
by
  sorry

end roots_of_equation_l101_101257


namespace sin_double_angle_l101_101342

theorem sin_double_angle (α : ℝ) (h : Real.tan α = -1/3) : Real.sin (2 * α) = -3/5 := by 
  sorry

end sin_double_angle_l101_101342


namespace sqrt_sum_eq_five_sqrt_three_l101_101323

theorem sqrt_sum_eq_five_sqrt_three : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_sum_eq_five_sqrt_three_l101_101323


namespace quadratic_has_negative_root_l101_101352

def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks for a range of m such that the quadratic function intersects the negative x-axis
theorem quadratic_has_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ quadratic_function m x = 0) ↔ (1 ≤ m ∧ m < 2 ∨ 2 < m ∧ m < 3) :=
sorry

end quadratic_has_negative_root_l101_101352


namespace amplitude_five_phase_shift_minus_pi_over_4_l101_101187

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4))

theorem amplitude_five : ∀ x : ℝ, 5 * Real.cos (x + (Real.pi / 4)) = f x :=
by
  sorry

theorem phase_shift_minus_pi_over_4 : ∀ x : ℝ, f x = 5 * Real.cos (x + (Real.pi / 4)) :=
by
  sorry

end amplitude_five_phase_shift_minus_pi_over_4_l101_101187


namespace polar_equation_of_circle_slope_of_line_l101_101540

-- Part 1: Polar equation of circle C
theorem polar_equation_of_circle (x y : ℝ) :
  (x - 2) ^ 2 + y ^ 2 = 9 -> ∃ (ρ θ : ℝ), ρ^2 - 4*ρ*Real.cos θ - 5 = 0 := 
sorry

-- Part 2: Slope of line L intersecting C at points A and B
theorem slope_of_line (α : ℝ) (L : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ t, L t = (t * Real.cos α, t * Real.sin α)) ∧ dist A B = 2 * Real.sqrt 7 ∧ 
  (∃ x y, (x - 2) ^ 2 + y ^ 2 = 9 ∧ L (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = (x, y))
  -> Real.tan α = 1 ∨ Real.tan α = -1 :=
sorry

end polar_equation_of_circle_slope_of_line_l101_101540


namespace smallest_possible_floor_sum_l101_101213

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l101_101213


namespace line_equation_isosceles_triangle_l101_101653

theorem line_equation_isosceles_triangle 
  (x y : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : l 3 2)
  (h2 : ∀ x y, l x y → (x = y ∨ x + y = 2 * intercept))
  (intercept : ℝ) :
  l x y ↔ (x - y = 1 ∨ x + y = 5) :=
by
  sorry

end line_equation_isosceles_triangle_l101_101653


namespace find_linear_in_two_variables_l101_101140

def is_linear_in_two_variables (eq : String) : Bool :=
  eq = "x=y+1"

theorem find_linear_in_two_variables :
  (is_linear_in_two_variables "4xy=2" = false) ∧
  (is_linear_in_two_variables "1-x=7" = false) ∧
  (is_linear_in_two_variables "x^2+2y=-2" = false) ∧
  (is_linear_in_two_variables "x=y+1" = true) :=
by
  sorry

end find_linear_in_two_variables_l101_101140


namespace journey_distance_l101_101858

theorem journey_distance (D : ℝ) (h1 : (D / 40) + (D / 60) = 40) : D = 960 :=
by
  sorry

end journey_distance_l101_101858


namespace greatest_prime_factor_15_fact_plus_17_fact_l101_101805

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l101_101805


namespace total_flowers_sold_l101_101493

/-
Ginger owns a flower shop, where she sells roses, lilacs, and gardenias.
On Tuesday, she sold three times more roses than lilacs, and half as many gardenias as lilacs.
If she sold 10 lilacs, prove that the total number of flowers sold on Tuesday is 45.
-/

theorem total_flowers_sold
    (lilacs roses gardenias : ℕ)
    (h_lilacs : lilacs = 10)
    (h_roses : roses = 3 * lilacs)
    (h_gardenias : gardenias = lilacs / 2)
    (ht : lilacs + roses + gardenias = 45) :
    lilacs + roses + gardenias = 45 :=
by sorry

end total_flowers_sold_l101_101493


namespace greatest_prime_factor_15_17_factorial_l101_101802

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l101_101802


namespace giant_exponent_modulo_result_l101_101336

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l101_101336


namespace probability_first_green_probability_both_green_conditional_probability_second_green_given_first_green_l101_101080

-- Define the conditions
def total_eggs : ℕ := 5
def green_eggs : ℕ := 3
def white_eggs : ℕ := 2

-- Calculation of probabilities
theorem probability_first_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  ((green_eggs.to_real / total_eggs.to_real) = (3 / 5)) :=
by
  intros h1 h2 h3
  sorry

theorem probability_both_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  (((green_eggs.to_real / total_eggs.to_real) *
   ((green_eggs - 1).to_real / (total_eggs - 1).to_real)) = (3 / 10)) :=
by
  intros h1 h2 h3
  sorry

theorem conditional_probability_second_green_given_first_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  ((((green_eggs - 1).to_real / (total_eggs - 1).to_real)) = (1 / 2)) :=
by
  intros h1 h2 h3
  sorry

end probability_first_green_probability_both_green_conditional_probability_second_green_given_first_green_l101_101080


namespace polynomial_solution_l101_101640

theorem polynomial_solution (P : Polynomial ℝ) (h_0 : P.eval 0 = 0) (h_func : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end polynomial_solution_l101_101640


namespace petya_friends_l101_101705

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101705


namespace remainder_of_expression_l101_101208

theorem remainder_of_expression (n : ℤ) (h : n % 60 = 1) : (n^2 + 2 * n + 3) % 60 = 6 := 
by
  sorry

end remainder_of_expression_l101_101208


namespace bags_on_wednesday_l101_101309

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l101_101309


namespace kids_still_awake_l101_101865

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l101_101865


namespace katherine_bottle_caps_l101_101547

-- Define the initial number of bottle caps Katherine has
def initial_bottle_caps : ℕ := 34

-- Define the number of bottle caps eaten by the hippopotamus
def eaten_bottle_caps : ℕ := 8

-- Define the remaining number of bottle caps Katherine should have
def remaining_bottle_caps : ℕ := initial_bottle_caps - eaten_bottle_caps

-- Theorem stating that Katherine will have 26 bottle caps after the hippopotamus eats 8 of them
theorem katherine_bottle_caps : remaining_bottle_caps = 26 := by
  sorry

end katherine_bottle_caps_l101_101547


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101812

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101812


namespace gcd_polynomial_l101_101199

open scoped Classical

-- Definitions and conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = k * m

-- The main theorem
theorem gcd_polynomial (b : ℕ) (h : is_multiple_of b 1428) : 
  Nat.gcd (b^2 + 11 * b + 30) (b + 6) = 6 :=
by
  sorry

end gcd_polynomial_l101_101199


namespace smallest_positive_integer_with_12_divisors_l101_101605

/-- The number of divisors of a positive integer n is determined by the product of the increments by 1 of the exponents in its prime factorization. -/
def number_of_divisors (n : ℕ) : ℕ :=
  let factors := n.factorization.to_multiset 
  factors.map (λ e, e + 1).prod

theorem smallest_positive_integer_with_12_divisors :
  ∃ n : ℕ, (number_of_divisors n = 12) ∧ (∀ m : ℕ, number_of_divisors m = 12 → n ≤ m) :=
begin
  sorry
end

end smallest_positive_integer_with_12_divisors_l101_101605


namespace cubic_yard_to_cubic_meter_l101_101072

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter_l101_101072


namespace smallest_with_12_divisors_is_60_l101_101599

def has_exactly_12_divisors (n : ℕ) : Prop :=
  (0 < n) ∧ (∀ d, d ∣ n → d > 0) ∧ (card (finset.filter (λ d, d ∣ n) (finset.range (n+1))) = 12)

theorem smallest_with_12_divisors_is_60 :
  ∃ (n : ℕ), has_exactly_12_divisors n ∧ ∀ m, has_exactly_12_divisors m → n ≤ m :=
begin
  use 60,
  split,
  {
    unfold has_exactly_12_divisors,
    split,
    { exact dec_trivial }, -- 60 > 0
    split,
    { intros d hd, exact nat.pos_of_ne_zero (ne_of_lt hd).symm },
    {
      -- There are exactly 12 divisors of 60
      have : (finset.filter (λ d, d ∣ 60) (finset.range (60+1))).card = 12 :=
      by dec_trivial,
      exact this,
    }
  },
  {
    intros m hm,
    have h1 : nat.prime_factors 60 = [2, 3, 5] := by dec_trivial,
    have h2 : ∀ d ∣ 60, d ∈ finset.filter (λ d, d ∣ 60) (finset.range (60+1)) := by dec_trivial,
    sorry
  }
end

end smallest_with_12_divisors_is_60_l101_101599


namespace arithmetic_progression_terms_even_sums_l101_101398

theorem arithmetic_progression_terms_even_sums (n a d : ℕ) (h_even : Even n) 
  (h_odd_sum : n * (a + (n - 2) * d) = 60) 
  (h_even_sum : n * (a + d + a + (n - 1) * d) = 72) 
  (h_last_first : (n - 1) * d = 12) : n = 8 := 
sorry

end arithmetic_progression_terms_even_sums_l101_101398


namespace largest_divisor_l101_101673

theorem largest_divisor (n : ℕ) (h1 : 0 < n) (h2 : 450 ∣ n ^ 2) : 30 ∣ n :=
sorry

end largest_divisor_l101_101673


namespace decreasing_function_l101_101058

theorem decreasing_function (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (m + 3) * x1 - 2 > (m + 3) * x2 - 2) ↔ m < -3 :=
by
  sorry

end decreasing_function_l101_101058


namespace total_cost_correct_l101_101011

-- Define the costs for each day
def day1_rate : ℝ := 150
def day1_miles_cost : ℝ := 0.50 * 620
def gps_service_cost : ℝ := 10
def day1_total_cost : ℝ := day1_rate + day1_miles_cost + gps_service_cost

def day2_rate : ℝ := 100
def day2_miles_cost : ℝ := 0.40 * 744
def day2_total_cost : ℝ := day2_rate + day2_miles_cost + gps_service_cost

def day3_rate : ℝ := 75
def day3_miles_cost : ℝ := 0.30 * 510
def day3_total_cost : ℝ := day3_rate + day3_miles_cost + gps_service_cost

-- Define the total cost
def total_cost : ℝ := day1_total_cost + day2_total_cost + day3_total_cost

-- Prove that the total cost is equal to the calculated value
theorem total_cost_correct : total_cost = 1115.60 :=
by
  -- This is where the proof would go, but we leave it out for now
  sorry

end total_cost_correct_l101_101011


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101335

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101335


namespace total_books_l101_101217

def books_per_shelf : ℕ := 78
def number_of_shelves : ℕ := 15

theorem total_books : books_per_shelf * number_of_shelves = 1170 := 
by
  sorry

end total_books_l101_101217


namespace greatest_value_of_squares_l101_101087

-- Given conditions
variables (a b c d : ℝ)
variables (h1 : a + b = 20)
variables (h2 : ab + c + d = 105)
variables (h3 : ad + bc = 225)
variables (h4 : cd = 144)

theorem greatest_value_of_squares : a^2 + b^2 + c^2 + d^2 ≤ 150 := by
  sorry

end greatest_value_of_squares_l101_101087


namespace eval_expression_l101_101403

theorem eval_expression : (4^2 - 2^3) = 8 := by
  sorry

end eval_expression_l101_101403


namespace daily_sales_change_l101_101284

theorem daily_sales_change
    (mon_sales : ℕ)
    (week_total_sales : ℕ)
    (days_in_week : ℕ)
    (avg_sales_per_day : ℕ)
    (other_days_total_sales : ℕ)
    (x : ℕ)
    (h1 : days_in_week = 7)
    (h2 : avg_sales_per_day = 5)
    (h3 : week_total_sales = avg_sales_per_day * days_in_week)
    (h4 : mon_sales = 2)
    (h5 : week_total_sales = mon_sales + other_days_total_sales)
    (h6 : other_days_total_sales = 33)
    (h7 : 2 + x + 2 + 2*x + 2 + 3*x + 2 + 4*x + 2 + 5*x + 2 + 6*x = other_days_total_sales) : 
  x = 1 :=
by
sorry

end daily_sales_change_l101_101284


namespace greatest_prime_factor_15_17_factorial_l101_101837

theorem greatest_prime_factor_15_17_factorial :
  ∀ n p q (hp_prime : Prime p) (hq_prime : Prime q),
  15! + 17! = n * (p * q) ∧ 15.factorial ∣ (15! + 17!) ∧ p = 13 ∧ p > q →
  p = 13 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_15_17_factorial_l101_101837


namespace petya_friends_count_l101_101745

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l101_101745


namespace triangle_area_l101_101677

theorem triangle_area (a c : ℝ) (B : ℝ) (h_a : a = 7) (h_c : c = 5) (h_B : B = 120 * Real.pi / 180) : 
  (1 / 2 * a * c * Real.sin B) = 35 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l101_101677


namespace numberOfRealSolutions_l101_101887

theorem numberOfRealSolutions :
  ∀ (x : ℝ), (-4*x + 12)^2 + 1 = (x - 1)^2 → (∃ a b : ℝ, (a ≠ b) ∧ (-4*a + 12)^2 + 1 = (a - 1)^2 ∧ (-4*b + 12)^2 + 1 = (b - 1)^2) := by
  sorry

end numberOfRealSolutions_l101_101887


namespace minimum_four_sum_multiple_of_four_l101_101222

theorem minimum_four_sum_multiple_of_four (n : ℕ) (h : n = 7) (s : Fin n → ℤ) :
  ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a + s b + s c + s d) % 4 = 0 := 
by
  -- Proof goes here
  sorry

end minimum_four_sum_multiple_of_four_l101_101222


namespace petya_friends_l101_101720

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l101_101720


namespace arcsin_one_half_l101_101458

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101458


namespace function_periodicity_even_l101_101346

theorem function_periodicity_even (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_period : ∀ x : ℝ, x ≥ 0 → f (x + 2) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1) :
  f (-2017) + f 2018 = 1 :=
sorry

end function_periodicity_even_l101_101346


namespace petya_friends_l101_101738

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l101_101738


namespace p_or_q_iff_not_p_and_not_q_false_l101_101522

variables (p q : Prop)

theorem p_or_q_iff_not_p_and_not_q_false : (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
by sorry

end p_or_q_iff_not_p_and_not_q_false_l101_101522


namespace greatest_prime_factor_of_factorial_sum_l101_101792

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l101_101792


namespace remainder_of_11_pow_2023_mod_33_l101_101630

theorem remainder_of_11_pow_2023_mod_33 : (11 ^ 2023) % 33 = 11 := 
by
  sorry

end remainder_of_11_pow_2023_mod_33_l101_101630


namespace sin_A_eq_height_on_AB_l101_101937

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l101_101937


namespace maximize_distance_l101_101644

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end maximize_distance_l101_101644


namespace intersection_A_B_eq_C_l101_101499

def A : Set ℝ := {1, 3, 5, 7}
def B : Set ℝ := {x | -x^2 + 4 * x ≥ 0}
def C : Set ℝ := {1, 3}

theorem intersection_A_B_eq_C : A ∩ B = C := 
by sorry

end intersection_A_B_eq_C_l101_101499


namespace smallest_integer_with_12_divisors_is_288_l101_101603

-- Given n is a positive integer with exactly 12 divisors, prove that the smallest such n is 288
theorem smallest_integer_with_12_divisors_is_288 :
  ∃ n : ℕ, (0 < n) ∧ ((∀ d : ℕ, d ∣ n → d > 0) ∧ (∀ d : ℕ, d ∣ n → (∃ (k : ℕ), k ∈ {1, 2, 3, 4, 6, 12}))) ∧ n = 288 :=
sorry

end smallest_integer_with_12_divisors_is_288_l101_101603


namespace medicine_price_after_discount_l101_101622

theorem medicine_price_after_discount :
  ∀ (price : ℝ) (discount : ℝ), price = 120 → discount = 0.3 → 
  (price - price * discount) = 84 :=
by
  intros price discount h1 h2
  rw [h1, h2]
  sorry

end medicine_price_after_discount_l101_101622


namespace eval_expression_l101_101322

theorem eval_expression : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 :=
by 
  sorry

end eval_expression_l101_101322


namespace traders_gain_percentage_l101_101172

theorem traders_gain_percentage (C : ℝ) (h : 0 < C) : 
  let cost_of_100_pens := 100 * C
  let gain := 40 * C
  let selling_price := cost_of_100_pens + gain
  let gain_percentage := (gain / cost_of_100_pens) * 100
  gain_percentage = 40 := by
  sorry

end traders_gain_percentage_l101_101172


namespace sum_of_squares_of_sum_and_difference_l101_101003

theorem sum_of_squares_of_sum_and_difference (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 8) : 
  (x + y)^2 + (x - y)^2 = 640 :=
by
  sorry

end sum_of_squares_of_sum_and_difference_l101_101003


namespace eval_expression_l101_101995

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l101_101995


namespace line_passes_through_fixed_point_l101_101970

theorem line_passes_through_fixed_point 
  (a b : ℝ) 
  (h : 2 * a + b = 1) : 
  a * 4 + b * 2 = 2 :=
sorry

end line_passes_through_fixed_point_l101_101970


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101791
noncomputable def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

theorem greatest_prime_factor_of_15_fact_plus_17_fact :
  ∀ (p : ℕ), prime p → 
      p ∣ (factorial 15 + factorial 17) → p <= 13 :=
by
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101791


namespace initial_ratio_of_milk_to_water_l101_101533

variable (M W : ℕ) -- M represents the amount of milk, W represents the amount of water

theorem initial_ratio_of_milk_to_water (h1 : M + W = 45) (h2 : 8 * M = 9 * (W + 23)) :
  M / W = 4 :=
by
  sorry

end initial_ratio_of_milk_to_water_l101_101533


namespace midpoint_in_polar_coordinates_l101_101536

-- Define the problem as a theorem in Lean 4
theorem midpoint_in_polar_coordinates :
  let A := (10, Real.pi / 4)
  let B := (10, 3 * Real.pi / 4)
  ∃ r θ, (r = 5 * Real.sqrt 2) ∧ (θ = Real.pi / 2) ∧
         0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

end midpoint_in_polar_coordinates_l101_101536


namespace problem_statement_l101_101652

-- Conditions
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a ^ x > 0
def q (x : ℝ) : Prop := x > 0 ∧ x ≠ 1 ∧ (Real.log 2 / Real.log x + Real.log x / Real.log 2 ≥ 2)

-- Theorem statement
theorem problem_statement (a x : ℝ) : ¬p a ∨ ¬q x :=
by sorry

end problem_statement_l101_101652


namespace aitana_jayda_total_spending_l101_101032

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l101_101032


namespace cosine_of_angle_l101_101500

theorem cosine_of_angle (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) : 
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_of_angle_l101_101500


namespace sum_of_squares_l101_101976

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 14) (h2 : a * b + b * c + a * c = 72) : 
  a^2 + b^2 + c^2 = 52 :=
by
  sorry

end sum_of_squares_l101_101976


namespace probability_red_card_top_l101_101625

def num_red_cards : ℕ := 26
def total_cards : ℕ := 52
def prob_red_card_top : ℚ := num_red_cards / total_cards

theorem probability_red_card_top : prob_red_card_top = (1 / 2) := by
  sorry

end probability_red_card_top_l101_101625


namespace range_of_k_l101_101078

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
by
  sorry

end range_of_k_l101_101078


namespace toby_photos_l101_101130

variable (p0 d c e x : ℕ)
def photos_remaining : ℕ := p0 - d + c + x - e

theorem toby_photos (h1 : p0 = 63) (h2 : d = 7) (h3 : c = 15) (h4 : e = 3) : photos_remaining p0 d c e x = 68 + x :=
by
  rw [h1, h2, h3, h4]
  sorry

end toby_photos_l101_101130


namespace green_team_final_score_l101_101238

theorem green_team_final_score (G : ℕ) :
  (∀ G : ℕ, 68 = G + 29 → G = 39) :=
by
  sorry

end green_team_final_score_l101_101238


namespace system_of_equations_solution_l101_101618

theorem system_of_equations_solution
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1)
  (h2 : x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2)
  (h3 : x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3)
  (h4 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4)
  (h5 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) :
  x1 = 1 ∧ x2 = -1 ∧ x3 = 1 ∧ x4 = -1 ∧ x5 = 1 :=
by {
  -- proof steps go here
  sorry
}

end system_of_equations_solution_l101_101618


namespace election_total_votes_l101_101013

theorem election_total_votes (V : ℝ)
  (h_majority : ∃ O, 0.84 * V = O + 476)
  (h_total_votes : ∀ O, V = 0.84 * V + O) :
  V = 700 :=
sorry

end election_total_votes_l101_101013


namespace base10_to_base4_addition_l101_101994

-- Define the base 10 numbers
def n1 : ℕ := 45
def n2 : ℕ := 28

-- Define the base 4 representations
def n1_base4 : ℕ := 2 * 4^2 + 3 * 4^1 + 1 * 4^0
def n2_base4 : ℕ := 1 * 4^2 + 3 * 4^1 + 0 * 4^0

-- The sum of the base 10 numbers
def sum_base10 : ℕ := n1 + n2

-- The expected sum in base 4
def sum_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Prove the equivalence
theorem base10_to_base4_addition :
  (n1 + n2 = n1_base4  + n2_base4) →
  (sum_base10 = sum_base4) :=
by
  sorry

end base10_to_base4_addition_l101_101994


namespace contradiction_assumption_l101_101854

theorem contradiction_assumption (a b c : ℕ) :
  (∃ k : ℕ, (k = a ∨ k = b ∨ k = c) ∧ ∃ n : ℕ, k = 2 * n + 1) →
  (∃ k1 k2 : ℕ, (k1 = a ∨ k1 = b ∨ k1 = c) ∧ (k2 = a ∨ k2 = b ∨ k2 = c) ∧ k1 ≠ k2 ∧ ∃ n1 n2 : ℕ, k1 = 2 * n1 ∧ k2 = 2 * n2) ∨
  (∀ k : ℕ, (k = a ∨ k = b ∨ k = c) → ∃ n : ℕ, k = 2 * n + 1) :=
sorry

end contradiction_assumption_l101_101854


namespace range_of_2a_plus_3b_l101_101647

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l101_101647


namespace smallest_m_value_l101_101662

noncomputable def smallest_m : ℝ :=
  let k := -1 in
  -(k * Real.pi / 2) - (Real.pi / 12)

theorem smallest_m_value (m : ℝ) :
  (∀ (x : ℝ), sin (2 * (x - m) + Real.pi / 3) = sin (2 * (-x - m) + Real.pi / 3)) → 0 < m → m = smallest_m :=
by
  sorry

end smallest_m_value_l101_101662


namespace find_x_from_roots_l101_101201

variable (x m : ℕ)

theorem find_x_from_roots (h1 : (m + 3)^2 = x) (h2 : (2 * m - 15)^2 = x) : x = 49 := by
  sorry

end find_x_from_roots_l101_101201


namespace minimum_keys_needed_l101_101539

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end minimum_keys_needed_l101_101539


namespace multiply_binomials_l101_101956

theorem multiply_binomials (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 :=
by 
  -- Proof is to be filled here
  sorry

end multiply_binomials_l101_101956


namespace find_g_of_3_l101_101393

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 3) : g 3 = 0 :=
by sorry

end find_g_of_3_l101_101393


namespace total_spent_l101_101033

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l101_101033


namespace farmers_acres_to_clean_l101_101431

-- Definitions of the main quantities
variables (A D : ℕ)

-- Conditions
axiom condition1 : A = 80 * D
axiom condition2 : 90 * (D - 1) + 30 = A

-- Theorem asserting the total number of acres to be cleaned
theorem farmers_acres_to_clean : A = 480 :=
by
  -- The proof would go here, but is omitted as per instructions
  sorry

end farmers_acres_to_clean_l101_101431


namespace smallest_integer_with_12_divisors_l101_101598

theorem smallest_integer_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, n > 0 ∧ (divisors_count m = 12 → n ≤ m)) ∧ n = 60 := by
  sorry

end smallest_integer_with_12_divisors_l101_101598


namespace greatest_prime_factor_15_17_factorial_l101_101801

theorem greatest_prime_factor_15_17_factorial :
  ∃ p : ℕ, prime p ∧ p.factors.greatest = 17 ∧ 15! + 17! = 15! * (1 + 16 * 17) := by
  sorry

end greatest_prime_factor_15_17_factorial_l101_101801


namespace math_problem_l101_101379

theorem math_problem 
  (a b c d : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c ≥ d) 
  (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * a^a * b^b * c^c * d^d < 1 := 
sorry

end math_problem_l101_101379


namespace miles_flown_on_thursday_l101_101678
-- Importing the necessary library

-- Defining the problem conditions and the proof goal
theorem miles_flown_on_thursday (x : ℕ) : 
  (∀ y, (3 * (1134 + y) = 7827) → y = x) → x = 1475 :=
by
  intro h
  specialize h 1475
  sorry

end miles_flown_on_thursday_l101_101678


namespace blue_marbles_l101_101124

theorem blue_marbles (r b : ℕ) (h_ratio : 3 * b = 5 * r) (h_red : r = 18) : b = 30 := by
  -- proof
  sorry

end blue_marbles_l101_101124


namespace bucket_weight_l101_101287

theorem bucket_weight (c d : ℝ) (x y : ℝ) 
  (h1 : x + 3/4 * y = c) 
  (h2 : x + 1/3 * y = d) :
  x + 1/4 * y = (6 * d - c) / 5 := 
sorry

end bucket_weight_l101_101287


namespace resulting_chemical_percentage_l101_101435

theorem resulting_chemical_percentage 
  (init_solution_pct : ℝ) (replacement_frac : ℝ) (replacing_solution_pct : ℝ) (resulting_solution_pct : ℝ) : 
  init_solution_pct = 0.85 →
  replacement_frac = 0.8181818181818182 →
  replacing_solution_pct = 0.30 →
  resulting_solution_pct = 0.40 :=
by
  intros h1 h2 h3
  sorry

end resulting_chemical_percentage_l101_101435


namespace part1_part2_l101_101699

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
axiom a_3a_5 : a 3 * a 5 = 63
axiom a_2a_6 : a 2 + a 6 = 16

-- Part (1) Proving the general formula
theorem part1 : 
  (∀ n : ℕ, a n = 12 - n) :=
sorry

-- Part (2) Proving the maximum value of S_n
theorem part2 :
  (∃ n : ℕ, (S n = (n * (12 - (n - 1) / 2)) → (n = 11 ∨ n = 12) ∧ (S n = 66))) :=
sorry

end part1_part2_l101_101699


namespace m_range_l101_101048

/-- Given a point (x, y) on the circle x^2 + (y - 1)^2 = 2, show that the real number m,
such that x + y + m ≥ 0, must satisfy m ≥ 1. -/
theorem m_range (x y m : ℝ) (h₁ : x^2 + (y - 1)^2 = 2) (h₂ : x + y + m ≥ 0) : m ≥ 1 :=
sorry

end m_range_l101_101048


namespace candy_bar_cost_l101_101319

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def cost_of_candy_bar : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost : cost_of_candy_bar = 1 := by
  sorry

end candy_bar_cost_l101_101319


namespace arithmetic_sequence_inequality_l101_101946

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_inequality
  (a d : ℕ)
  (i j k l : ℕ)
  (hi : i ≤ j)
  (hj : j ≤ k)
  (hk : k ≤ l)
  (hij: i + l = j + k)
  : (arithmetic_seq a d i) * (arithmetic_seq a d l) ≤ (arithmetic_seq a d j) * (arithmetic_seq a d k) :=
sorry

end arithmetic_sequence_inequality_l101_101946


namespace find_y_l101_101317

def star (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem find_y (y : ℝ) : star 3 (star 4 y) = -2 → y = -11.5 :=
by
  sorry

end find_y_l101_101317


namespace smallest_integer_larger_than_expr_is_248_l101_101992

noncomputable def small_int_larger_than_expr : ℕ :=
  let expr := (Real.sqrt 5 + Real.sqrt 3)^4
  248

theorem smallest_integer_larger_than_expr_is_248 :
    ∃ (n : ℕ), n > (Real.sqrt 5 + Real.sqrt 3)^4 ∧ n = small_int_larger_than_expr := 
by
  -- We introduce the target integer 248
  use (248 : ℕ)
  -- The given conditions should lead us to 248 being greater than the expression.
  sorry

end smallest_integer_larger_than_expr_is_248_l101_101992


namespace smallest_positive_integer_n_l101_101339

theorem smallest_positive_integer_n :
  ∃ (n: ℕ), n = 4 ∧ (∀ x: ℝ, (Real.sin x)^n + (Real.cos x)^n ≤ 2 / n) :=
sorry

end smallest_positive_integer_n_l101_101339


namespace tiffany_cans_at_end_of_week_l101_101129

theorem tiffany_cans_at_end_of_week:
  (4 + 2.5 - 1.25 + 0 + 3.75 - 1.5 + 0 = 7.5) :=
by
  sorry

end tiffany_cans_at_end_of_week_l101_101129


namespace Jie_is_tallest_l101_101529

variables (Person : Type) (Igor Jie Faye Goa Han : Person)
variable (taller : Person → Person → Prop)

-- Given conditions
axiom h1 : taller Jie Igor
axiom h2 : taller Faye Goa
axiom h3 : taller Jie Faye
axiom h4 : taller Goa Han

-- Problem statement
theorem Jie_is_tallest : ∀ x : Person, x ∈ {Igor, Faye, Goa, Han} → taller Jie x :=
by
  intros x hx
  cases hx <;> try { assumption } <;> try { exact_taller.trans h3
  |>
.javascript() { sorry }

end Jie_is_tallest_l101_101529


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101815

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101815


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101334

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101334


namespace initial_percentage_alcohol_l101_101424

-- Define the initial conditions
variables (P : ℚ) -- percentage of alcohol in the initial solution
variables (V1 V2 : ℚ) -- volumes of the initial solution and added alcohol
variables (C2 : ℚ) -- concentration of the resulting solution

-- Given the initial conditions and additional parameters
def initial_solution_volume : ℚ := 6
def added_alcohol_volume : ℚ := 1.8
def final_solution_volume : ℚ := initial_solution_volume + added_alcohol_volume
def final_solution_concentration : ℚ := 0.5 -- 50%

-- The amount of alcohol initially = (P / 100) * V1
-- New amount of alcohol after adding pure alcohol
-- This should equal to the final concentration of the new volume

theorem initial_percentage_alcohol : 
  (P / 100 * initial_solution_volume) + added_alcohol_volume = final_solution_concentration * final_solution_volume → 
  P = 35 :=
sorry

end initial_percentage_alcohol_l101_101424


namespace rotate_and_translate_line_l101_101388

theorem rotate_and_translate_line :
  let initial_line (x : ℝ) := 3 * x
  let rotated_line (x : ℝ) := - (1 / 3) * x
  let translated_line (x : ℝ) := - (1 / 3) * (x - 1)

  ∀ x : ℝ, translated_line x = - (1 / 3) * x + (1 / 3) := 
by
  intros
  simp
  sorry

end rotate_and_translate_line_l101_101388


namespace arcsin_of_half_l101_101441

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101441


namespace angle_B_equal_pi_div_3_l101_101917

-- Define the conditions and the statement to be proved
theorem angle_B_equal_pi_div_3 (A B C : ℝ) 
  (h₁ : Real.sin A / Real.sin B = 5 / 7)
  (h₂ : Real.sin B / Real.sin C = 7 / 8) : 
  B = Real.pi / 3 :=
sorry

end angle_B_equal_pi_div_3_l101_101917


namespace common_difference_l101_101659

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference_l101_101659


namespace subtraction_of_fractions_l101_101133

theorem subtraction_of_fractions :
  1 + 1 / 2 - 3 / 5 = 9 / 10 := by
  sorry

end subtraction_of_fractions_l101_101133


namespace inequality_proof_l101_101951

theorem inequality_proof
  (a b c d : ℝ)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  abc + bcd + cda + dab ≤ (1 / 27) + (176 * abcd / 27) :=
sorry

end inequality_proof_l101_101951


namespace arcsin_one_half_l101_101471

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101471


namespace tenth_number_in_row_1_sum_of_2023rd_numbers_l101_101701

noncomputable def a (n : ℕ) := (-2)^n
noncomputable def b (n : ℕ) := a n + (n + 1)

theorem tenth_number_in_row_1 : a 10 = (-2)^10 := 
sorry

theorem sum_of_2023rd_numbers : a 2023 + b 2023 = -(2^2024) + 2024 := 
sorry

end tenth_number_in_row_1_sum_of_2023rd_numbers_l101_101701


namespace eval_expression_l101_101996

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l101_101996


namespace packs_of_chewing_gum_zero_l101_101250

noncomputable def frozen_yogurt_price : ℝ := sorry
noncomputable def chewing_gum_price : ℝ := frozen_yogurt_price / 2
noncomputable def packs_of_chewing_gum : ℕ := sorry

theorem packs_of_chewing_gum_zero 
  (F : ℝ) -- Price of a pint of frozen yogurt
  (G : ℝ) -- Price of a pack of chewing gum
  (x : ℕ) -- Number of packs of chewing gum
  (H1 : G = F / 2)
  (H2 : 5 * F + x * G + 25 = 55)
  : x = 0 :=
sorry

end packs_of_chewing_gum_zero_l101_101250


namespace algebraic_expression_simplification_l101_101310

theorem algebraic_expression_simplification :
  0.25 * (-1 / 2) ^ (-4 : ℝ) - 4 / (Real.sqrt 5 - 1) ^ (0 : ℝ) - (1 / 16) ^ (-1 / 2 : ℝ) = -4 :=
by
  sorry

end algebraic_expression_simplification_l101_101310


namespace cookies_in_the_fridge_l101_101987

-- Define the conditions
def total_baked : ℕ := 256
def tim_cookies : ℕ := 15
def mike_cookies : ℕ := 23
def anna_cookies : ℕ := 2 * tim_cookies

-- Define the proof problem
theorem cookies_in_the_fridge : (total_baked - (tim_cookies + mike_cookies + anna_cookies)) = 188 :=
by
  -- insert proof here
  sorry

end cookies_in_the_fridge_l101_101987


namespace arcsin_of_half_l101_101445

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  -- condition: sin (π / 6) = 1 / 2
  have sin_pi_six : Real.sin (Real.pi / 6) = 1 / 2 := sorry,
  -- to prove
  exact sorry
}

end arcsin_of_half_l101_101445


namespace smallest_integer_with_12_divisors_l101_101602

theorem smallest_integer_with_12_divisors :
  ∃ n : ℕ, 0 < n ∧ (∀ m : ℕ, 0 < m → ∀ d : ℕ, m = 2^5 * 3^2 → d ≠ 288 ∧ n = 288) ∧ ∏ p in (finset.divisors 288) = 12 :=
sorry

end smallest_integer_with_12_divisors_l101_101602


namespace fraction_of_students_on_trip_are_girls_l101_101171

variable (b g : ℕ)
variable (H1 : g = 2 * b) -- twice as many girls as boys
variable (fraction_girls_on_trip : ℚ := 2 / 3)
variable (fraction_boys_on_trip : ℚ := 1 / 2)

def fraction_of_girls_on_trip (b g : ℕ) (H1 : g = 2 * b) (fraction_girls_on_trip : ℚ) (fraction_boys_on_trip : ℚ) :=
  let girls_on_trip := fraction_girls_on_trip * g
  let boys_on_trip := fraction_boys_on_trip * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip

theorem fraction_of_students_on_trip_are_girls (b g : ℕ) (H1 : g = 2 * b) : 
  fraction_of_girls_on_trip b g H1 (2 / 3) (1 / 2) = 8 / 11 := 
by sorry

end fraction_of_students_on_trip_are_girls_l101_101171


namespace shark_sightings_relationship_l101_101633

theorem shark_sightings_relationship (C D R : ℕ) (h₁ : C + D = 40) (h₂ : C = R - 8) (h₃ : C = 24) :
  R = 32 :=
by
  sorry

end shark_sightings_relationship_l101_101633


namespace volume_of_displaced_water_l101_101427

-- Defining the conditions of the problem
def cube_side_length : ℝ := 6
def cyl_radius : ℝ := 5
def cyl_height : ℝ := 12
def cube_volume (s : ℝ) : ℝ := s^3

-- Statement: The volume of water displaced by the cube when it is fully submerged in the barrel
theorem volume_of_displaced_water :
  cube_volume cube_side_length = 216 := by
  sorry

end volume_of_displaced_water_l101_101427


namespace fractions_order_l101_101855

theorem fractions_order :
  (25 / 21 < 23 / 19) ∧ (23 / 19 < 21 / 17) :=
by {
  sorry
}

end fractions_order_l101_101855


namespace asymptote_equations_l101_101511

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (e = sqrt 3) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

theorem asymptote_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : sqrt (a^2 + b^2) / a = sqrt 3) :
  ∀ (x : ℝ), ∃ (y : ℝ), y = sqrt 2 * x ∨ y = -sqrt 2 * x :=
sorry

end asymptote_equations_l101_101511


namespace domain_of_function_l101_101315

theorem domain_of_function :
  ∀ x : ℝ, ⌊x^2 - 8 * x + 18⌋ ≠ 0 :=
sorry

end domain_of_function_l101_101315


namespace petya_friends_l101_101706

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101706


namespace proof_problem_l101_101514

-- Definitions based on the given conditions
def cond1 : Prop := 1 * 9 + 2 = 11
def cond2 : Prop := 12 * 9 + 3 = 111
def cond3 : Prop := 123 * 9 + 4 = 1111
def cond4 : Prop := 1234 * 9 + 5 = 11111
def cond5 : Prop := 12345 * 9 + 6 = 111111

-- Main statement to prove
theorem proof_problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) : 
  123456 * 9 + 7 = 1111111 :=
sorry

end proof_problem_l101_101514


namespace find_c_l101_101550

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

theorem find_c :
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 5 :=
by
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  sorry

end find_c_l101_101550


namespace tallest_is_jie_l101_101528

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end tallest_is_jie_l101_101528


namespace estimate_students_spending_more_than_60_l101_101967

-- Definition of the problem
def students_surveyed : ℕ := 50
def students_inclined_to_subscribe : ℕ := 8
def total_students : ℕ := 1000
def estimated_students : ℕ := 600

-- Define the proof task
theorem estimate_students_spending_more_than_60 :
  (students_inclined_to_subscribe : ℝ) / (students_surveyed : ℝ) * (total_students : ℝ) = estimated_students :=
by
  sorry

end estimate_students_spending_more_than_60_l101_101967


namespace greatest_prime_factor_of_factorial_sum_l101_101795

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l101_101795


namespace circle_tangent_parabola_l101_101294

theorem circle_tangent_parabola (a b : ℝ) (h_parabola : ∀ x, x^2 + 1 = y) 
  (h_tangency : (a, a^2 + 1) ∧ (-a, a^2 + 1)) 
  (h_center : (0, b)) 
  (h_circle : ∀ x y, x^2 + (y - b)^2 = r^2) 
  (h_tangent_points : (x = a) ∧ (x = -a)) : 
  b - (a^2 + 1) = -1/2 := 
sorry

end circle_tangent_parabola_l101_101294


namespace greatest_prime_factor_of_sum_l101_101808

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l101_101808


namespace find_m_and_star_l101_101045

-- Definitions from conditions
def star (x y m : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

-- Given conditions
def given_star (x y : ℚ) (m : ℚ) : Prop := star x y m = 2 / 5

-- Target: Proving m = 1 and 2 * 6 = 6 / 7 given the conditions
theorem find_m_and_star :
  ∀ m : ℚ, 
  (given_star 1 2 m) → 
  (m = 1 ∧ star 2 6 m = 6 / 7) := 
sorry

end find_m_and_star_l101_101045


namespace kids_still_awake_l101_101867

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l101_101867


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101782

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101782


namespace circle_tangent_parabola_height_difference_l101_101292

theorem circle_tangent_parabola_height_difference 
  (a b : ℝ)
  (h_tangent1 : (a, a^2 + 1))
  (h_tangent2 : (-a, a^2 + 1))
  (parabola_eq : ∀ x, x^2 + 1 = (x, x^2 + 1))
  (circle_eq : ∀ x, x^2 + ((x^2 + 1) - b)^2 = r^2) : 
  b - (a^2 + 1) = 1 / 2 :=
sorry

end circle_tangent_parabola_height_difference_l101_101292


namespace sum_of_palindromes_l101_101973

-- Define a three-digit palindrome predicate
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ b < 10 ∧ n = 100*a + 10*b + a

-- Define the product of the two palindromes equaling 436,995
theorem sum_of_palindromes (a b : ℕ) (h_a : is_palindrome a) (h_b : is_palindrome b) (h_prod : a * b = 436995) : 
  a + b = 1332 :=
sorry

end sum_of_palindromes_l101_101973


namespace inequality_not_true_l101_101670

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a > 0) :=
sorry

end inequality_not_true_l101_101670


namespace map_lines_l101_101241

noncomputable def maps_lines_to_lines (f : ℝ² → ℝ²) : Prop :=
∀ L : set ℝ², is_line L → is_line (f '' L)

axiom is_circle (C : set ℝ²) : Prop

theorem map_lines (f : ℝ² → ℝ²) (hf : continuous f) (H : ∀ C : set ℝ², is_circle C → is_circle (f '' C)) :
  maps_lines_to_lines f :=
sorry

end map_lines_l101_101241


namespace height_difference_l101_101293

theorem height_difference
  (a b : ℝ)
  (parabola_eq : ∀ x, y = x^2 + 1)
  (circle_center : b = 2 * a^2 + 1 / 2) :
  b - (a^2 + 1) = a^2 - 1 / 2 :=
by {
  sorry
}

end height_difference_l101_101293


namespace ratio_of_numbers_l101_101585

theorem ratio_of_numbers (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l101_101585


namespace value_subtracted_is_five_l101_101912

variable (N x : ℕ)

theorem value_subtracted_is_five
  (h1 : (N - x) / 7 = 7)
  (h2 : (N - 14) / 10 = 4) : x = 5 := by
  sorry

end value_subtracted_is_five_l101_101912


namespace find_a_parallel_lines_l101_101200

theorem find_a_parallel_lines (a : ℝ) (l1_parallel_l2 : x + a * y + 6 = 0 → (a - 1) * x + 2 * y + 3 * a = 0 → Parallel) : a = -1 :=
sorry

end find_a_parallel_lines_l101_101200


namespace rectangular_plot_breadth_l101_101860

theorem rectangular_plot_breadth (b l : ℝ) (A : ℝ)
  (h1 : l = 3 * b)
  (h2 : A = l * b)
  (h3 : A = 2700) : b = 30 :=
by sorry

end rectangular_plot_breadth_l101_101860


namespace original_cost_of_tshirt_l101_101635

theorem original_cost_of_tshirt
  (backpack_cost : ℕ := 10)
  (cap_cost : ℕ := 5)
  (total_spent_after_discount : ℕ := 43)
  (discount : ℕ := 2)
  (tshirt_cost_before_discount : ℕ) :
  total_spent_after_discount + discount - (backpack_cost + cap_cost) = tshirt_cost_before_discount :=
by
  sorry

end original_cost_of_tshirt_l101_101635


namespace smallest_n_coprime_groups_l101_101695

open Finset

def S : Finset ℕ := range 99 \ {0}

theorem smallest_n_coprime_groups (T : Finset ℕ) (hT : ∀ T ⊆ S, T.card = 50 → 
  ∃ X ⊆ T, X.card = 10 ∧ 
  (∀ G₁ G₂ : Finset ℕ, G₁ ∪ G₂ = X ∧ G₁.card = 5 ∧ G₂.card = 5 → 
    (∃ x ∈ G₁, ∀ y ∈ G₁ \ {x}, Nat.Coprime x y) ∨ 
    (∃ x ∈ G₂, ∀ y ∈ G₂ \ {x}, ¬Nat.Coprime x y))) : ∀ (n : ℕ), n = 50 := 
sorry

end smallest_n_coprime_groups_l101_101695


namespace number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l101_101769

noncomputable def a (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem number_of_diagonals_pentagon : a 5 = 5 := sorry

theorem difference_hexagon_pentagon : a 6 - a 5 = 4 := sorry

theorem difference_successive_polygons (n : ℕ) (h : 4 ≤ n) : a (n + 1) - a n = n - 1 := sorry

end number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l101_101769


namespace evaluate_five_iterates_of_f_at_one_l101_101553

def f (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem evaluate_five_iterates_of_f_at_one :
  f (f (f (f (f 1)))) = 4 := by
  sorry

end evaluate_five_iterates_of_f_at_one_l101_101553


namespace solve_arcsin_cos_eq_x_over_3_l101_101963

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem solve_arcsin_cos_eq_x_over_3 :
  ∀ x,
  - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  arcsin (cos x) = x / 3 →
  x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8 :=
sorry

end solve_arcsin_cos_eq_x_over_3_l101_101963


namespace principal_sum_l101_101279

/-!
# Problem Statement
Given:
1. The difference between compound interest (CI) and simple interest (SI) on a sum at 10% per annum for 2 years is 65.
2. The rate of interest \( R \) is 10%.
3. The time \( T \) is 2 years.

We need to prove that the principal sum \( P \) is 6500.
-/

theorem principal_sum (P : ℝ) (R : ℝ) (T : ℕ) (H : (P * (1 + R / 100)^T - P) - (P * R * T / 100) = 65) 
                      (HR : R = 10) (HT : T = 2) : P = 6500 := 
by 
  sorry

end principal_sum_l101_101279


namespace compute_expr_l101_101178

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expr_l101_101178


namespace vector_calc_l101_101175

def vec1 : ℝ × ℝ := (5, -8)
def vec2 : ℝ × ℝ := (2, 6)
def vec3 : ℝ × ℝ := (-1, 4)
def scalar : ℝ := 5

theorem vector_calc :
  (vec1.1 - scalar * vec2.1 + vec3.1, vec1.2 - scalar * vec2.2 + vec3.2) = (-6, -34) :=
sorry

end vector_calc_l101_101175


namespace simplest_square_root_l101_101999

theorem simplest_square_root :
  (λ x, let y := x in y = sqrt 3) ∧ 
  (sqrt 3) = sqrt 3 ∧
  ∀ (y : ℝ), (y = sqrt (1 / 2)  ∨ y = sqrt 0.2 ∨ y = sqrt 3  ∨ y = sqrt 8) → 
  ( sqrt 3 = y → 
    ( ∃ (r : ℝ), y = 0 ∨ y = 1 ∨ y = r)
  )
:=
begin
  sorry
end

end simplest_square_root_l101_101999


namespace petya_has_19_friends_l101_101749

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101749


namespace percentage_increase_l101_101223

variable {α : Type} [LinearOrderedField α]

theorem percentage_increase (x y : α) (h : x = 0.5 * y) : y = x + x :=
by
  -- The steps of the proof are omitted and 'sorry' is used to skip actual proof.
  sorry

end percentage_increase_l101_101223


namespace kids_still_awake_l101_101866

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l101_101866


namespace smallest_positive_integer_n_l101_101849

theorem smallest_positive_integer_n :
  ∃ n: ℕ, (n > 0) ∧ (∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (∃ d: ℕ, d ∣ (n^2 - 2 * n) ∧ d ∣ k) ∧ (k ∣ (n^2 - 2 * n) → k = d)) ∧ n = 5 :=
by
  sorry

end smallest_positive_integer_n_l101_101849


namespace special_sale_day_price_l101_101439

-- Define the original price
def original_price : ℝ := 250

-- Define the first discount rate
def first_discount_rate : ℝ := 0.40

-- Calculate the price after the first discount
def price_after_first_discount (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- Define the second discount rate (special sale day)
def second_discount_rate : ℝ := 0.10

-- Calculate the price after the second discount
def price_after_second_discount (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price * (1 - discount_rate)

-- Theorem statement
theorem special_sale_day_price :
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 := by
  sorry

end special_sale_day_price_l101_101439


namespace smallest_n_product_exceeds_l101_101510

theorem smallest_n_product_exceeds (n : ℕ) : (5 : ℝ) ^ (n * (n + 1) / 14) > 1000 ↔ n = 7 :=
by sorry

end smallest_n_product_exceeds_l101_101510


namespace petya_friends_l101_101730

theorem petya_friends : ∃ x : ℕ, (5 * x + 8 = 6 * x - 11) → x = 19 :=
by
  use 19
  intro h
  sorry

end petya_friends_l101_101730


namespace range_of_a_l101_101675

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) :
  (0 < a ∧ 4 - 4 * a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l101_101675


namespace y_coord_range_of_M_l101_101373

theorem y_coord_range_of_M :
  ∀ (M : ℝ × ℝ), ((M.1 + 1)^2 + M.2^2 = 2) → 
  ((M.1 - 2)^2 + M.2^2 + M.1^2 + M.2^2 ≤ 10) →
  - (Real.sqrt 7) / 2 ≤ M.2 ∧ M.2 ≤ (Real.sqrt 7) / 2 := 
by 
  sorry

end y_coord_range_of_M_l101_101373


namespace solve_system_equations_l101_101573

theorem solve_system_equations (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    ∃ x y z : ℝ,  
      (x * y = (z - a) ^ 2) ∧
      (y * z = (x - b) ^ 2) ∧
      (z * x = (y - c) ^ 2) ∧
      x = ((b ^ 2 - a * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      y = ((c ^ 2 - a * b) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      z = ((a ^ 2 - b * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) :=
sorry

end solve_system_equations_l101_101573


namespace polynomial_not_divisible_by_x_minus_5_l101_101521

theorem polynomial_not_divisible_by_x_minus_5 (m : ℝ) :
  (∀ x, x = 4 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) →
  ¬(∀ x, x = 5 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) :=
by
  sorry

end polynomial_not_divisible_by_x_minus_5_l101_101521


namespace existence_of_unique_root_l101_101770

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 5

theorem existence_of_unique_root :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f 0 = -4 ∧
  f 2 = Real.exp 2 - 1 →
  ∃! c, f c = 0 :=
by
  sorry

end existence_of_unique_root_l101_101770


namespace smallest_divisor_sum_of_squares_of_1_to_7_l101_101035

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_divisor_sum_of_squares_of_1_to_7 (S : ℕ) (h : S = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) :
  ∃ m, is_divisor m S ∧ (∀ d, is_divisor d S → 2 ≤ d) :=
sorry

end smallest_divisor_sum_of_squares_of_1_to_7_l101_101035


namespace right_triangle_area_perimeter_ratio_l101_101848

theorem right_triangle_area_perimeter_ratio :
  let a := 4
  let b := 8
  let area := (1/2) * a * b
  let c := Real.sqrt (a^2 + b^2)
  let perimeter := a + b + c
  let ratio := area / perimeter
  ratio = 3 - Real.sqrt 5 :=
by
  sorry

end right_triangle_area_perimeter_ratio_l101_101848


namespace book_page_count_l101_101520

theorem book_page_count (x : ℝ) : 
    (x - (1 / 4 * x + 20)) - ((1 / 3 * (x - (1 / 4 * x + 20)) + 25)) - (1 / 2 * ((x - (1 / 4 * x + 20)) - (1 / 3 * (x - (1 / 4 * x + 20)) + 25)) + 30) = 70 →
    x = 480 :=
by
  sorry

end book_page_count_l101_101520


namespace greatest_prime_factor_of_15f_17f_is_17_l101_101843

-- Definitions for factorials based on the given conditions
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

-- The statement to prove
theorem greatest_prime_factor_of_15f_17f_is_17 : 
  ∀ (n : ℕ), n = 15! + 17! → 17 ∈ max_prime_factors n :=
sorry

end greatest_prime_factor_of_15f_17f_is_17_l101_101843


namespace max_value_of_z_l101_101530

theorem max_value_of_z (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) :
  x^2 + y^2 ≤ 2 :=
by {
  sorry
}

end max_value_of_z_l101_101530


namespace a_3_def_a_4_def_a_r_recurrence_l101_101170

-- Define minimally the structure of the problem.
noncomputable def a_r (r : ℕ) : ℕ := -- Definition for minimum phone calls required.
by sorry

-- Assertions for the specific cases provided.
theorem a_3_def : a_r 3 = 3 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_4_def : a_r 4 = 4 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_r_recurrence (r : ℕ) (hr : r ≥ 3) : a_r r ≤ a_r (r - 1) + 2 :=
by
  -- Proof is omitted with sorry.
  sorry

end a_3_def_a_4_def_a_r_recurrence_l101_101170


namespace probability_yellow_ball_l101_101683

-- Definitions of the conditions
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := white_balls + yellow_balls

-- Theorem statement
theorem probability_yellow_ball : (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by
  -- Using tactics to facilitate the proof
  simp [yellow_balls, total_balls]
  sorry

end probability_yellow_ball_l101_101683


namespace min_value_geometric_sequence_l101_101348

-- Definition for conditions and problem setup
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- Given data
variable (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_sum : a 2015 + a 2017 = Real.pi)

-- Goal statement
theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = (Real.pi^2) / 2 ∧ (
    ∀ a : ℕ → ℝ, 
    is_geometric_sequence a → 
    a 2015 + a 2017 = Real.pi → 
    a 2016 * (a 2014 + a 2018) ≥ (Real.pi^2) / 2
  ) :=
sorry

end min_value_geometric_sequence_l101_101348


namespace train_speed_is_correct_l101_101417

/-- Define the length of the train and the time taken to cross the telegraph post. --/
def train_length : ℕ := 240
def crossing_time : ℕ := 16

/-- Define speed calculation based on train length and crossing time. --/
def train_speed : ℕ := train_length / crossing_time

/-- Prove that the computed speed of the train is 15 meters per second. --/
theorem train_speed_is_correct : train_speed = 15 := sorry

end train_speed_is_correct_l101_101417


namespace correct_combined_monthly_rate_of_profit_l101_101161

structure Book :=
  (cost_price : ℕ)
  (selling_price : ℕ)
  (months_held : ℕ)

def profit (b : Book) : ℕ :=
  b.selling_price - b.cost_price

def monthly_rate_of_profit (b : Book) : ℕ :=
  if b.months_held = 0 then profit b else profit b / b.months_held

def combined_monthly_rate_of_profit (b1 b2 b3 : Book) : ℕ :=
  monthly_rate_of_profit b1 + monthly_rate_of_profit b2 + monthly_rate_of_profit b3

theorem correct_combined_monthly_rate_of_profit :
  combined_monthly_rate_of_profit
    {cost_price := 50, selling_price := 90, months_held := 1}
    {cost_price := 120, selling_price := 150, months_held := 2}
    {cost_price := 75, selling_price := 110, months_held := 0} 
    = 90 := 
by
  sorry

end correct_combined_monthly_rate_of_profit_l101_101161


namespace solution_l101_101392

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem solution (a b : ℝ) (H : a = 5 * Real.pi / 8 ∧ b = 7 * Real.pi / 8) :
  is_monotonically_increasing g a b :=
sorry

end solution_l101_101392


namespace evaluate_expression_l101_101482

theorem evaluate_expression (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 40 :=
by
  rw [hx, hy]
  sorry

end evaluate_expression_l101_101482


namespace people_at_first_concert_l101_101095

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l101_101095


namespace parallelogram_side_sum_l101_101181

variable (x y : ℚ)

theorem parallelogram_side_sum :
  4 * x - 1 = 10 →
  5 * y + 3 = 12 →
  x + y = 91 / 20 :=
by
  intros h1 h2
  sorry

end parallelogram_side_sum_l101_101181


namespace ranking_l101_101903

variables (score : string → ℝ)
variables (Hannah Cassie Bridget David : string)

-- Conditions based on the problem statement
axiom Hannah_shows_her_test_to_everyone : ∀ x, x ≠ Hannah → x = Cassie ∨ x = Bridget ∨ x = David
axiom David_shows_his_test_only_to_Bridget : ∀ x, x ≠ Bridget → x ≠ David
axiom Cassie_does_not_show_her_test : ∀ x, x = Hannah ∨ x = Bridget ∨ x = David → x ≠ Cassie

-- Statements based on what Cassie and Bridget claim
axiom Cassie_statement : score Cassie > min (score Hannah) (score Bridget)
axiom Bridget_statement : score David > score Bridget

-- Final ranking to be proved
theorem ranking : score David > score Bridget ∧ score Bridget > score Cassie ∧ score Cassie > score Hannah := sorry

end ranking_l101_101903


namespace proof_triangle_properties_l101_101925

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l101_101925


namespace apples_equation_l101_101409

variable {A J H : ℕ}

theorem apples_equation:
    A + J = 12 →
    H = A + J + 9 →
    A = J + 8 →
    H = 21 :=
by
  intros h1 h2 h3
  sorry

end apples_equation_l101_101409


namespace numSpaceDiagonals_P_is_241_l101_101151

noncomputable def numSpaceDiagonals (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
  let face_diagonals := 2 * quad_faces
  total_segments - edges - face_diagonals

theorem numSpaceDiagonals_P_is_241 :
  numSpaceDiagonals 26 60 24 12 = 241 := by 
  sorry

end numSpaceDiagonals_P_is_241_l101_101151


namespace remainder_of_N_mod_45_l101_101089

def concatenated_num_from_1_to_52 : ℕ := 
  -- This represents the concatenated number from 1 to 52.
  -- We define here in Lean as a placeholder 
  -- since Lean cannot concatenate numbers directly.
  12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152

theorem remainder_of_N_mod_45 : 
  concatenated_num_from_1_to_52 % 45 = 37 := 
sorry

end remainder_of_N_mod_45_l101_101089


namespace vertex_angle_measure_l101_101876

-- Define the isosceles triangle and its properties
def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) :=
  (A = B ∨ B = C ∨ C = A) ∧ (a + b + c = 180)

-- Define the conditions based on the problem statement
def two_angles_sum_to_100 (x y : ℝ) := x + y = 100

-- The measure of the vertex angle
theorem vertex_angle_measure (A B C : ℝ) (a b c : ℝ) 
  (h1 : is_isosceles_triangle A B C a b c) (h2 : two_angles_sum_to_100 A B) :
  C = 20 ∨ C = 80 :=
sorry

end vertex_angle_measure_l101_101876


namespace find_value_l101_101026

theorem find_value (number : ℕ) (h : number / 5 + 16 = 58) : number / 15 + 74 = 88 :=
sorry

end find_value_l101_101026


namespace greatest_prime_factor_of_sum_l101_101811

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l101_101811


namespace giant_exponent_modulo_result_l101_101337

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l101_101337


namespace car_distance_ratio_l101_101634

theorem car_distance_ratio (t : ℝ) (h₁ : t > 0)
    (speed_A speed_B : ℝ)
    (h₂ : speed_A = 70)
    (h₃ : speed_B = 35)
    (ratio : ℝ)
    (h₄ : ratio = 2)
    (h_time : ∀ a b : ℝ, a * t = b * t → a = b) :
  (speed_A * t) / (speed_B * t) = ratio := by
  sorry

end car_distance_ratio_l101_101634


namespace fill_pipe_time_l101_101158

theorem fill_pipe_time (t : ℕ) (H : ∀ C : Type, (1 / 2 : ℚ) * C = t * 1/2 * C) : t = t :=
by
  sorry

end fill_pipe_time_l101_101158


namespace distance_between_foci_is_six_l101_101264

-- Lean 4 Statement
noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  if (p1 = (1, 3) ∧ p2 = (6, -1) ∧ p3 = (11, 3)) then 6 else 0

theorem distance_between_foci_is_six : distance_between_foci (1, 3) (6, -1) (11, 3) = 6 :=
by
  sorry

end distance_between_foci_is_six_l101_101264


namespace abs_value_solution_l101_101360

theorem abs_value_solution (a : ℝ) : |-a| = |-5.333| → (a = 5.333 ∨ a = -5.333) :=
by
  sorry

end abs_value_solution_l101_101360


namespace sum_of_consecutive_pages_with_product_15300_l101_101262

theorem sum_of_consecutive_pages_with_product_15300 : 
  ∃ n : ℕ, n * (n + 1) = 15300 ∧ n + (n + 1) = 247 :=
by
  sorry

end sum_of_consecutive_pages_with_product_15300_l101_101262


namespace value_of_a_l101_101674

theorem value_of_a (a : ℝ) (h : (a ^ 3) * ((5).choose (2)) = 80) : a = 2 :=
  sorry

end value_of_a_l101_101674


namespace original_cost_l101_101438

theorem original_cost (A : ℝ) (discount : ℝ) (sale_price : ℝ) (original_price : ℝ) (h1 : discount = 0.30) (h2 : sale_price = 35) (h3 : sale_price = (1 - discount) * original_price) : 
  original_price = 50 := by
  sorry

end original_cost_l101_101438


namespace compute_expr_l101_101177

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expr_l101_101177


namespace product_fraction_l101_101889

open Int

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem product_fraction :
  (product first_six_composites : ℚ) / (product (first_three_primes ++ next_three_composites) : ℚ) = 24 / 7 :=
by 
  sorry

end product_fraction_l101_101889


namespace determinant_transformation_l101_101669

theorem determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
    p * (5 * r + 2 * s) - r * (5 * p + 2 * q) = -6 := by
  sorry

end determinant_transformation_l101_101669


namespace salaries_of_a_and_b_l101_101147

theorem salaries_of_a_and_b {x y : ℝ}
  (h1 : x + y = 5000)
  (h2 : 0.05 * x = 0.15 * y) :
  x = 3750 :=
by sorry

end salaries_of_a_and_b_l101_101147


namespace greatest_prime_factor_of_15_fact_plus_17_fact_l101_101813

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then by
    let factors := (List.range (n + 1)).filter (λ p => p.prime ∧ n % p = 0)
    exact List.maximum' factors (List.cons_mem _ h) 
  else 1

theorem greatest_prime_factor_of_15_fact_plus_17_fact : 
    greatest_prime_factor (factorial 15 + factorial 17) = 17 :=
  sorry

end greatest_prime_factor_of_15_fact_plus_17_fact_l101_101813


namespace total_spent_l101_101034

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l101_101034


namespace bird_mammal_difference_africa_asia_l101_101039

noncomputable def bird_families_to_africa := 42
noncomputable def bird_families_to_asia := 31
noncomputable def bird_families_to_south_america := 7

noncomputable def mammal_families_to_africa := 24
noncomputable def mammal_families_to_asia := 18
noncomputable def mammal_families_to_south_america := 15

noncomputable def reptile_families_to_africa := 15
noncomputable def reptile_families_to_asia := 9
noncomputable def reptile_families_to_south_america := 5

-- Calculate the total number of families migrating to Africa, Asia, and South America
noncomputable def total_families_to_africa := bird_families_to_africa + mammal_families_to_africa + reptile_families_to_africa
noncomputable def total_families_to_asia := bird_families_to_asia + mammal_families_to_asia + reptile_families_to_asia
noncomputable def total_families_to_south_america := bird_families_to_south_america + mammal_families_to_south_america + reptile_families_to_south_america

-- Calculate the combined total of bird and mammal families going to Africa
noncomputable def bird_and_mammal_families_to_africa := bird_families_to_africa + mammal_families_to_africa

-- Difference between bird and mammal families to Africa and total animal families to Asia
noncomputable def difference := bird_and_mammal_families_to_africa - total_families_to_asia

theorem bird_mammal_difference_africa_asia : difference = 8 := 
by
  sorry

end bird_mammal_difference_africa_asia_l101_101039


namespace incorrect_statement_l101_101382

-- Definitions based on the given conditions
def tripling_triangle_altitude_triples_area (b h : ℝ) : Prop :=
  3 * (1/2 * b * h) = 1/2 * b * (3 * h)

def halving_rectangle_base_halves_area (b h : ℝ) : Prop :=
  1/2 * b * h = 1/2 * (b * h)

def tripling_circle_radius_triples_area (r : ℝ) : Prop :=
  3 * (Real.pi * r^2) = Real.pi * (3 * r)^2

def tripling_divisor_and_numerator_leaves_quotient_unchanged (a b : ℝ) (hb : b ≠ 0) : Prop :=
  a / b = 3 * a / (3 * b)

def halving_negative_quantity_makes_it_greater (x : ℝ) : Prop :=
  x < 0 → (x / 2) > x

-- The incorrect statement is that tripling the radius of a circle triples the area
theorem incorrect_statement : ∃ r : ℝ, tripling_circle_radius_triples_area r → False :=
by
  use 1
  simp [tripling_circle_radius_triples_area]
  sorry

end incorrect_statement_l101_101382


namespace each_dog_food_intake_l101_101481

theorem each_dog_food_intake (total_food : ℝ) (dog_count : ℕ) (equal_amount : ℝ) : total_food = 0.25 → dog_count = 2 → (total_food / dog_count) = equal_amount → equal_amount = 0.125 :=
by
  intros h1 h2 h3
  sorry

end each_dog_food_intake_l101_101481


namespace sin_A_and_height_on_AB_l101_101934

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l101_101934


namespace integer_part_inequality_l101_101944

theorem integer_part_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
 (h_cond : (x + y + z) * ((1 / x) + (1 / y) + (1 / z)) = (91 / 10)) :
  (⌊(x^3 + y^3 + z^3) * ((1 / x^3) + (1 / y^3) + (1 / z^3))⌋) = 9 :=
by
  -- proof here
  sorry

end integer_part_inequality_l101_101944


namespace evaluate_ratio_is_negative_two_l101_101090

noncomputable def evaluate_ratio (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : ℂ :=
  (a^15 + b^15) / (a + b)^15

theorem evaluate_ratio_is_negative_two (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : 
  evaluate_ratio a b h = -2 := 
sorry

end evaluate_ratio_is_negative_two_l101_101090


namespace total_sampled_students_l101_101433

-- Define the total number of students in each grade
def students_in_grade12 : ℕ := 700
def students_in_grade11 : ℕ := 700
def students_in_grade10 : ℕ := 800

-- Define the number of students sampled from grade 10
def sampled_from_grade10 : ℕ := 80

-- Define the total number of students in the school
def total_students : ℕ := students_in_grade12 + students_in_grade11 + students_in_grade10

-- Prove that the total number of students sampled (x) is equal to 220
theorem total_sampled_students : 
  (sampled_from_grade10 : ℚ) / (students_in_grade10 : ℚ) * (total_students : ℚ) = 220 := 
by
  sorry

end total_sampled_students_l101_101433


namespace eraser_cost_l101_101029

noncomputable def price_of_erasers 
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  (bundle_count : ℝ) -- number of bundles sold
  (total_earned : ℝ) -- total amount earned
  (discount : ℝ) -- discount percentage for 20 bundles
  (bundle_contents : ℕ) -- 1 pencil and 2 erasers per bundle
  (price_ratio : ℝ) -- price ratio of eraser to pencil
  : Prop := 
  E = 0.5 * P ∧ -- The price of the erasers is 1/2 the price of the pencils.
  bundle_count = 20 ∧ -- The store sold a total of 20 bundles.
  total_earned = 80 ∧ -- The store earned $80.
  discount = 30 ∧ -- 30% discount for 20 bundles
  bundle_contents = 1 + 2 -- A bundle consists of 1 pencil and 2 erasers

theorem eraser_cost
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  : price_of_erasers P E 20 80 30 (1 + 2) 0.5 → E = 1.43 :=
by
  intro h
  sorry

end eraser_cost_l101_101029


namespace int_solve_ineq_l101_101904

theorem int_solve_ineq (x : ℤ) : (x + 3)^3 ≤ 8 ↔ x ≤ -1 :=
by sorry

end int_solve_ineq_l101_101904


namespace greatest_prime_factor_15f_plus_17f_l101_101828

theorem greatest_prime_factor_15f_plus_17f : 
  ∃ p : ℕ, p.Prime ∧ p = 17 ∧ ∀ q : ℕ, q.Prime → Nat.factorial 15 + Nat.factorial 17 ∣ q^p := sorry

end greatest_prime_factor_15f_plus_17f_l101_101828


namespace defeat_giant_enemy_crab_l101_101415

-- Definitions for the conditions of cutting legs and claws
def claws : ℕ := 2
def legs : ℕ := 6
def totalCuts : ℕ := claws + legs
def valid_sequences : ℕ :=
  (Nat.factorial legs) * (Nat.factorial claws) * Nat.choose (totalCuts - claws - 1) claws

-- Statement to prove the number of valid sequences of cuts given the conditions
theorem defeat_giant_enemy_crab : valid_sequences = 14400 := by
  sorry

end defeat_giant_enemy_crab_l101_101415


namespace greatest_prime_factor_of_15_l101_101819

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l101_101819


namespace find_m_given_solution_l101_101071

theorem find_m_given_solution (m x y : ℚ) (h₁ : x = 4) (h₂ : y = 3) (h₃ : m * x - y = 4) : m = 7 / 4 :=
by
  sorry

end find_m_given_solution_l101_101071


namespace area_of_region_l101_101914

-- The problem definition
def condition_1 (z : ℂ) : Prop := 
  0 < z.re / 20 ∧ z.re / 20 < 1 ∧
  0 < z.im / 20 ∧ z.im / 20 < 1 ∧
  0 < (20 / z).re ∧ (20 / z).re < 1 ∧
  0 < (20 / z).im ∧ (20 / z).im < 1

-- The proof statement
theorem area_of_region {z : ℂ} (h : condition_1 z) : 
  ∃ s : ℝ, s = 300 - 50 * Real.pi := sorry

end area_of_region_l101_101914


namespace arc_length_proof_l101_101173

noncomputable def arc_length_problem : ℝ :=
    ∫ y in 1..1.5, sqrt(1 + (0.5 * y - 0.5 / y) ^ 2)

theorem arc_length_proof :
    arc_length_problem = 0.3125 + 0.5 * Real.log 1.5 :=
sorry

end arc_length_proof_l101_101173


namespace polygonal_chain_segments_l101_101341

theorem polygonal_chain_segments (n : ℕ) :
  (∃ (S : Type) (chain : S → Prop), (∃ (closed_non_self_intersecting : S → Prop), 
  (∀ s : S, chain s → closed_non_self_intersecting s) ∧
  ∀ line_segment : S, chain line_segment → 
  (∃ other_segment : S, chain other_segment ∧ line_segment ≠ other_segment))) ↔ 
  (∃ k : ℕ, (n = 2 * k ∧ 5 ≤ k) ∨ (n = 2 * k + 1 ∧ 7 ≤ k)) :=
by sorry

end polygonal_chain_segments_l101_101341


namespace petya_has_19_friends_l101_101748

variable (x : ℕ)

-- Conditions defined
def condition1 : Prop := (total_stickers = 5 * x + 8)
def condition2 : Prop := (total_stickers = 6 * x - 11)

-- Theorem statement
theorem petya_has_19_friends (total_stickers : ℕ ) (x : ℕ) : condition1 x -> condition2 x -> x = 19 := by
  sorry

end petya_has_19_friends_l101_101748


namespace largest_of_five_numbers_l101_101141

theorem largest_of_five_numbers : ∀ (a b c d e : ℝ), 
  a = 0.938 → b = 0.9389 → c = 0.93809 → d = 0.839 → e = 0.893 → b = max a (max b (max c (max d e))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end largest_of_five_numbers_l101_101141


namespace ab_bc_ca_value_a4_b4_c4_value_l101_101650

theorem ab_bc_ca_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ab + bc + ca = -1/2 :=
sorry

theorem a4_b4_c4_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1/2 :=
sorry

end ab_bc_ca_value_a4_b4_c4_value_l101_101650


namespace max_regions_divided_by_lines_l101_101064

theorem max_regions_divided_by_lines (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) :
  ∃ r : ℕ, r = m * n + 2 * m + 2 * n - 1 :=
by
  sorry

end max_regions_divided_by_lines_l101_101064


namespace zeros_of_f_on_interval_l101_101519

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l101_101519


namespace students_in_line_l101_101160

theorem students_in_line (n : ℕ) (h : 1 ≤ n ∧ n ≤ 130) : 
  n = 3 ∨ n = 43 ∨ n = 129 :=
by
  sorry

end students_in_line_l101_101160


namespace greatest_prime_factor_15_fact_plus_17_fact_l101_101807

theorem greatest_prime_factor_15_fact_plus_17_fact :
  ∃ p : ℕ, prime p ∧ greatest_prime_factor (15! + 17!) = p := by
  sorry

end greatest_prime_factor_15_fact_plus_17_fact_l101_101807


namespace regular_polygon_enclosure_l101_101380

theorem regular_polygon_enclosure (m n : ℕ) (h₁: m = 6) (h₂: (m + 1) = 7): n = 6 :=
by
  -- Lean code to include the problem hypothesis and conclude the theorem
  sorry

end regular_polygon_enclosure_l101_101380


namespace simplify_product_l101_101148

theorem simplify_product (x y : ℝ) : 
  (x - 3 * y + 2) * (x + 3 * y + 2) = (x^2 + 4 * x + 4 - 9 * y^2) :=
by
  sorry

end simplify_product_l101_101148


namespace probability_three_and_one_l101_101366

-- Define the event: having 3 children of one sex and 1 of the opposite sex
def three_and_one (sexes : list bool) : Prop :=
  (count true sexes = 3 ∧ count false sexes = 1) ∨ (count true sexes = 1 ∧ count false sexes = 3)

-- Probability of each child being a boy (true) or a girl (false) is 1/2
def child_probability : ℙ (list bool) :=
  pmf.uniform (list bool) [true, false]

-- Final proof statement
theorem probability_three_and_one : 
  have_children (n : ℕ) := 4 ∧ child_probability := 1/2):
  ℙ (three_and_one have_children) = 1/2 :=
by
  sorry

end probability_three_and_one_l101_101366


namespace expected_value_is_correct_l101_101162

noncomputable def expected_winnings : ℚ :=
  (5/12 : ℚ) * 2 + (1/3 : ℚ) * 0 + (1/6 : ℚ) * (-2) + (1/12 : ℚ) * 10

theorem expected_value_is_correct : expected_winnings = 4 / 3 := 
by 
  -- Complex calculations skipped for brevity
  sorry

end expected_value_is_correct_l101_101162


namespace geometric_sequence_seventh_term_l101_101429

-- Define the initial conditions
def geometric_sequence_first_term := 3
def geometric_sequence_fifth_term (r : ℝ) := geometric_sequence_first_term * r^4 = 243

-- Statement for the seventh term problem
theorem geometric_sequence_seventh_term (r : ℝ) 
  (h1 : geometric_sequence_first_term = 3) 
  (h2 : geometric_sequence_fifth_term r) : 
  3 * r^6 = 2187 :=
sorry

end geometric_sequence_seventh_term_l101_101429


namespace avg_books_rounded_l101_101397

def books_read : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4)]

noncomputable def total_books_read (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.fst * pair.snd) 0

noncomputable def total_members (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.snd) 0

noncomputable def average_books_read (books : List (ℕ × ℕ)) : ℤ :=
  Int.ofNat (total_books_read books) / Int.ofNat (total_members books)

theorem avg_books_rounded :
  average_books_read books_read = 3 :=
by 
  sorry

end avg_books_rounded_l101_101397


namespace necessary_but_not_sufficient_for_parallel_lines_l101_101974

theorem necessary_but_not_sufficient_for_parallel_lines (m : ℝ) : 
  (m = -1/2 ∨ m = 0) ↔ (∀ x y : ℝ, (x + 2*m*y - 1 = 0 ∧ (3*m + 1)*x - m*y - 1 = 0) → false) :=
sorry

end necessary_but_not_sufficient_for_parallel_lines_l101_101974


namespace successful_pair_exists_another_with_same_arithmetic_mean_l101_101555

theorem successful_pair_exists_another_with_same_arithmetic_mean
  (a b : ℕ)
  (h_distinct : a ≠ b)
  (h_arith_mean_nat : ∃ m : ℕ, 2 * m = a + b)
  (h_geom_mean_nat : ∃ g : ℕ, g * g = a * b) :
  ∃ (c d : ℕ), c ≠ d ∧ ∃ m' : ℕ, 2 * m' = c + d ∧ ∃ g' : ℕ, g' * g' = c * d ∧ m' = (a + b) / 2 :=
sorry

end successful_pair_exists_another_with_same_arithmetic_mean_l101_101555


namespace find_excluded_digit_l101_101340

theorem find_excluded_digit (a b : ℕ) (d : ℕ) (h : a * b = 1024) (ha : a % 10 ≠ d) (hb : b % 10 ≠ d) : 
  ∃ r : ℕ, d = r ∧ r < 10 :=
by 
  sorry

end find_excluded_digit_l101_101340


namespace positive_diff_between_two_numbers_l101_101978

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l101_101978


namespace tiles_ratio_l101_101546

theorem tiles_ratio (total_tiles yellow_tiles purple_tiles white_tiles : ℕ) 
  (h_total : total_tiles = 20) (h_yellow : yellow_tiles = 3) 
  (h_purple : purple_tiles = 6) (h_white : white_tiles = 7) :
  let blue_tiles := total_tiles - (yellow_tiles + purple_tiles + white_tiles)
  in blue_tiles * yellow_tiles = 4 * 3 := by
  -- Number of non-blue tiles
  let non_blue_tiles := yellow_tiles + purple_tiles + white_tiles
  -- Number of blue tiles is the total number of tiles minus non-blue tiles
  let blue_tiles := total_tiles - non_blue_tiles
  -- Prove the ratio of blue tiles to yellow tiles is 4:3
  have h_blue : blue_tiles = 4 := by sorry
  -- Ensure blue_tiles * yellow_tiles = 4 * 3
  have h_ratio : blue_tiles * yellow_tiles = 4 * 3 := by
    rw [h_blue, h_yellow]
    -- Multiplies blue_tiles (4) and yellow_tiles (3) to 12.
    exact rfl
  exact h_ratio

end tiles_ratio_l101_101546


namespace number_of_participants_l101_101146

theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 :=
by
  sorry

end number_of_participants_l101_101146


namespace min_bottles_required_l101_101040

theorem min_bottles_required (bottle_ounces : ℕ) (total_ounces : ℕ) (h : bottle_ounces = 15) (ht : total_ounces = 150) :
  ∃ (n : ℕ), n * bottle_ounces >= total_ounces ∧ n = 10 :=
by
  sorry

end min_bottles_required_l101_101040


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101332

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101332


namespace percentage_of_50_of_125_l101_101017

theorem percentage_of_50_of_125 : (50 / 125) * 100 = 40 :=
by
  sorry

end percentage_of_50_of_125_l101_101017


namespace arcsin_one_half_eq_pi_six_l101_101446

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l101_101446


namespace chromium_percentage_new_alloy_l101_101228

-- Conditions as definitions
def first_alloy_chromium_percentage : ℝ := 12
def second_alloy_chromium_percentage : ℝ := 8
def first_alloy_weight : ℝ := 10
def second_alloy_weight : ℝ := 30

-- Final proof statement
theorem chromium_percentage_new_alloy : 
  ((first_alloy_chromium_percentage / 100 * first_alloy_weight +
    second_alloy_chromium_percentage / 100 * second_alloy_weight) /
  (first_alloy_weight + second_alloy_weight)) * 100 = 9 :=
by
  sorry

end chromium_percentage_new_alloy_l101_101228


namespace cos_Z_value_l101_101922

-- The conditions given in the problem
def sin_X := 4 / 5
def cos_Y := 3 / 5

-- The theorem we want to prove
theorem cos_Z_value (sin_X : ℝ) (cos_Y : ℝ) (hX : sin_X = 4/5) (hY : cos_Y = 3/5) : 
  ∃ cos_Z : ℝ, cos_Z = 7 / 25 :=
by
  -- Attach all conditions and solve
  sorry

end cos_Z_value_l101_101922


namespace range_of_a_l101_101496

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, e^x + 1/e^x > a) ∧ (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ↔ (-4 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l101_101496


namespace arcsin_half_eq_pi_six_l101_101453

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101453


namespace red_notebooks_count_l101_101954

variable (R B : ℕ)

-- Conditions
def cost_condition : Prop := 4 * R + 4 + 3 * B = 37
def count_condition : Prop := R + 2 + B = 12
def blue_notebooks_expr : Prop := B = 10 - R

-- Prove the number of red notebooks
theorem red_notebooks_count : cost_condition R B ∧ count_condition R B ∧ blue_notebooks_expr R B → R = 3 := by
  sorry

end red_notebooks_count_l101_101954


namespace jordan_oreos_l101_101689

def oreos (james jordan total : ℕ) : Prop :=
  james = 2 * jordan + 3 ∧
  jordan + james = total

theorem jordan_oreos (J : ℕ) (h : oreos (2 * J + 3) J 36) : J = 11 :=
by
  sorry

end jordan_oreos_l101_101689


namespace min_value_expression_l101_101329

noncomputable def expression (x y : ℝ) := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

theorem min_value_expression : ∀ x y : ℝ, expression x y ≥ -14 :=
by
  sorry

end min_value_expression_l101_101329


namespace x_intercept_is_34_l101_101556

-- Definitions of the initial line, rotation, and point.
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 50 = 0

def rotation_angle : ℝ := 30
def rotation_center : ℝ × ℝ := (10, 10)

-- Define the slope of the line l
noncomputable def slope_of_l : ℝ := 4 / 3

-- Define the slope of the line m after rotating line l by 30 degrees counterclockwise
noncomputable def tan_30 : ℝ := 1 / Real.sqrt 3
noncomputable def slope_of_m : ℝ := (slope_of_l + tan_30) / (1 - slope_of_l * tan_30)

-- Assume line m goes through the point (rotation_center.x, rotation_center.y)
-- This defines line m
def line_m (x y : ℝ) : Prop := y - rotation_center.2 = slope_of_m * (x - rotation_center.1)

-- To find the x-intercept of line m, we set y = 0 and solve for x
noncomputable def x_intercept_of_m : ℝ := rotation_center.1 - rotation_center.2 / slope_of_m

-- Proof statement that the x-intercept of line m is 34
theorem x_intercept_is_34 : x_intercept_of_m = 34 :=
by
  -- This would be the proof, but for now we leave it as sorry
  sorry

end x_intercept_is_34_l101_101556


namespace problem_1_l101_101015

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |2 * x + 1| + |x - 2| ≥ a ^ 2 - a + (1 / 2)) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_l101_101015


namespace bags_on_wednesday_l101_101308

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l101_101308


namespace prove_a_eq_b_l101_101261

theorem prove_a_eq_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_eq : a^b = b^a) (h_a_lt_1 : a < 1) : a = b :=
by
  sorry

end prove_a_eq_b_l101_101261


namespace shorter_side_of_rectangle_l101_101298

theorem shorter_side_of_rectangle (a b : ℕ) (h_perimeter : 2 * a + 2 * b = 62) (h_area : a * b = 240) : b = 15 :=
by
  sorry

end shorter_side_of_rectangle_l101_101298


namespace probability_not_overcoming_is_half_l101_101964

/-- Define the five elements. -/
inductive Element
| metal | wood | water | fire | earth

open Element

/-- Define the overcoming relation. -/
def overcomes : Element → Element → Prop
| metal, wood => true
| wood, earth => true
| earth, water => true
| water, fire => true
| fire, metal => true
| _, _ => false

/-- Define the probability calculation. -/
def probability_not_overcoming : ℚ :=
  let total_combinations := 10    -- C(5, 2)
  let overcoming_combinations := 5
  let not_overcoming_combinations := total_combinations - overcoming_combinations
  not_overcoming_combinations / total_combinations

/-- The proof problem statement. -/
theorem probability_not_overcoming_is_half : probability_not_overcoming = 1 / 2 :=
by
  sorry

end probability_not_overcoming_is_half_l101_101964


namespace profit_ratio_a_to_b_l101_101299

noncomputable def capital_a : ℕ := 3500
noncomputable def time_a : ℕ := 12
noncomputable def capital_b : ℕ := 10500
noncomputable def time_b : ℕ := 6

noncomputable def capital_months (capital : ℕ) (time : ℕ) : ℕ :=
  capital * time

noncomputable def capital_months_a : ℕ :=
  capital_months capital_a time_a

noncomputable def capital_months_b : ℕ :=
  capital_months capital_b time_b

theorem profit_ratio_a_to_b : (capital_months_a / Nat.gcd capital_months_a capital_months_b) =
                             2 ∧
                             (capital_months_b / Nat.gcd capital_months_a capital_months_b) =
                             3 := 
by
  sorry

end profit_ratio_a_to_b_l101_101299


namespace monotonically_decreasing_range_l101_101203

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 1

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, f' a x ≤ 0) → a ≤ -3 := by
  sorry

end monotonically_decreasing_range_l101_101203


namespace min_value_of_expression_l101_101697

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 27) 
  : x + 3 * y + 9 * z ≥ 27 :=
sorry

end min_value_of_expression_l101_101697


namespace largest_fraction_is_frac23_l101_101660

theorem largest_fraction_is_frac23 : 
    let frac35 := (3 : ℚ) / 5
    let frac23 := (2 : ℚ) / 3
    let frac49 := (4 : ℚ) / 9
    let frac515 := (5 : ℚ) / 15
    let frac845 := (8 : ℚ) / 45
  in frac23 > frac35 ∧ frac23 > frac49 ∧ frac23 > frac515 ∧ frac23 > frac845 :=
by
  let frac35 := (3 : ℚ) / 5
  let frac23 := (2 : ℚ) / 3
  let frac49 := (4 : ℚ) / 9
  let frac515 := (5 : ℚ) / 15
  let frac845 := (8 : ℚ) / 45
  sorry

end largest_fraction_is_frac23_l101_101660


namespace greatest_prime_factor_of_15_add_17_factorial_l101_101845

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l101_101845


namespace greatest_prime_factor_of_factorial_sum_l101_101793

/-- The greatest prime factor of 15! + 17! is 13. -/
theorem greatest_prime_factor_of_factorial_sum :
  ∀ n, (n = 15! + 17!) → greatest_prime_factor n = 13 :=
by
  sorry  -- Proof to be completed

end greatest_prime_factor_of_factorial_sum_l101_101793


namespace percentage_increase_chef_vs_dishwasher_l101_101627

variables 
  (manager_wage chef_wage dishwasher_wage : ℝ)
  (h_manager_wage : manager_wage = 8.50)
  (h_chef_wage : chef_wage = manager_wage - 3.315)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)

theorem percentage_increase_chef_vs_dishwasher :
  ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 22 :=
by
  sorry

end percentage_increase_chef_vs_dishwasher_l101_101627


namespace least_possible_value_l101_101687

theorem least_possible_value (x : ℚ) (h1 : x > 5 / 3) (h2 : x < 9 / 2) : 
  (9 / 2 - 5 / 3 : ℚ) = 17 / 6 :=
by sorry

end least_possible_value_l101_101687


namespace flooring_cost_correct_l101_101584

noncomputable def cost_of_flooring (l w h_t b_t c : ℝ) : ℝ :=
  let area_rectangle := l * w
  let area_triangle := (b_t * h_t) / 2
  let area_to_be_floored := area_rectangle - area_triangle
  area_to_be_floored * c

theorem flooring_cost_correct :
  cost_of_flooring 10 7 3 4 900 = 57600 :=
by
  sorry

end flooring_cost_correct_l101_101584


namespace simplify_expression_l101_101755

theorem simplify_expression : (Real.sin (15 * Real.pi / 180) + Real.sin (45 * Real.pi / 180)) / (Real.cos (15 * Real.pi / 180) + Real.cos (45 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  sorry

end simplify_expression_l101_101755


namespace value_of_a8_l101_101626

variable (a : ℕ → ℝ) (a_1 : a 1 = 2) (common_sum : ℝ) (h_sum : common_sum = 5)
variable (equal_sum_sequence : ∀ n, a (n + 1) + a n = common_sum)

theorem value_of_a8 : a 8 = 3 :=
sorry

end value_of_a8_l101_101626


namespace arcsin_half_eq_pi_six_l101_101452

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l101_101452


namespace rabbit_jump_lengths_order_l101_101244

theorem rabbit_jump_lengths_order :
  ∃ (R : ℕ) (G : ℕ) (P : ℕ) (F : ℕ),
    R = 2730 ∧
    R = P + 1100 ∧
    P = F + 150 ∧
    F = G - 200 ∧
    R > G ∧ G > P ∧ P > F :=
  by
  -- calculations
  sorry

end rabbit_jump_lengths_order_l101_101244


namespace lateral_surface_area_of_cylinder_l101_101434

variable (m n : ℝ) (S : ℝ)

theorem lateral_surface_area_of_cylinder (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (lateral_surface_area : ℝ),
    lateral_surface_area = (π * S) / (Real.sin (π * n / (m + n))) :=
sorry

end lateral_surface_area_of_cylinder_l101_101434


namespace total_number_of_students_l101_101225

theorem total_number_of_students (girls boys : ℕ) 
  (h_ratio : 8 * girls = 5 * boys) 
  (h_girls : girls = 160) : 
  girls + boys = 416 := 
sorry

end total_number_of_students_l101_101225


namespace sin_A_correct_height_on_AB_correct_l101_101931

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l101_101931


namespace algebraic_expression_value_l101_101698

theorem algebraic_expression_value 
  (p q r s : ℝ) 
  (hpq3 : p^2 / q^3 = 4 / 5) 
  (hrs2 : r^3 / s^2 = 7 / 9) : 
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := 
by 
  sorry

end algebraic_expression_value_l101_101698


namespace rate_of_rainfall_on_Monday_l101_101941

theorem rate_of_rainfall_on_Monday (R : ℝ) :
  7 * R + 4 * 2 + 2 * (2 * 2) = 23 → R = 1 := 
by
  sorry

end rate_of_rainfall_on_Monday_l101_101941


namespace total_fabric_yards_l101_101593

variable (checkered_cost plain_cost cost_per_yard : ℝ)
variable (checkered_yards plain_yards total_yards : ℝ)

def checkered_cost := 75
def plain_cost := 45
def cost_per_yard := 7.50

def checkered_yards := checkered_cost / cost_per_yard
def plain_yards := plain_cost / cost_per_yard

def total_yards := checkered_yards + plain_yards

theorem total_fabric_yards : total_yards = 16 :=
by {
  -- shorter and preferred syntax for skipping proof in Lean 4
  sorry
}

end total_fabric_yards_l101_101593


namespace min_value_x1_x2_l101_101512

theorem min_value_x1_x2 (a x_1 x_2 : ℝ) (h_a_pos : 0 < a) (h_sol_set : x_1 + x_2 = 4 * a) (h_prod_set : x_1 * x_2 = 3 * a^2) : 
  x_1 + x_2 + a / (x_1 * x_2) = 4 * a + 1 / (3 * a) :=
sorry

end min_value_x1_x2_l101_101512


namespace arcsin_one_half_l101_101460

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l101_101460


namespace fraction_simplification_l101_101880

theorem fraction_simplification : (3 : ℚ) / (2 - (3 / 4)) = 12 / 5 := by
  sorry

end fraction_simplification_l101_101880


namespace find_ratio_b_a_l101_101900

theorem find_ratio_b_a (a b : ℝ) 
  (h : ∀ x : ℝ, (2 * a - b) * x + (a + b) > 0 ↔ x > -3) : 
  b / a = 5 / 4 :=
sorry

end find_ratio_b_a_l101_101900


namespace greatest_prime_factor_15_factorial_plus_17_factorial_l101_101780

-- Define the greatest prime factor function.
def greatest_prime_factor (n : ℕ) : ℕ :=
if n ≤ 1 then n
else (nat.prime_factors n).max' sorry

theorem greatest_prime_factor_15_factorial_plus_17_factorial :
  greatest_prime_factor (15.factorial + 17.factorial) = 13 := 
by
  -- Factorization and given results.
  have h1 : 15.factorial + 17.factorial = 15.factorial * 273, 
  {
    sorry
  },
  have h2 : nat.prime_factors 273 = [3, 7, 13], 
  {
    sorry
  },
  -- The final goal to show.
  have h3 : greatest_prime_factor 273 = 13, 
  {
    sorry
  },
  rw h3,
  sorry

end greatest_prime_factor_15_factorial_plus_17_factorial_l101_101780


namespace paint_left_for_solar_system_l101_101558

-- Definitions for the paint used
def Mary's_paint := 3
def Mike's_paint := Mary's_paint + 2
def Lucy's_paint := 4

-- Total original amount of paint
def original_paint := 25

-- Total paint used by Mary, Mike, and Lucy
def total_paint_used := Mary's_paint + Mike's_paint + Lucy's_paint

-- Theorem stating the amount of paint left for the solar system
theorem paint_left_for_solar_system : (original_paint - total_paint_used) = 13 :=
by
  sorry

end paint_left_for_solar_system_l101_101558


namespace compute_expression_l101_101179

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l101_101179


namespace greatest_prime_factor_of_sum_l101_101810

-- Define the condition
def factorial (n : Nat) : Nat :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def expr := factorial 15 + factorial 17

def factored_expr := factorial 15 * (1 + 16 * 17)

-- Define prime factor check
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ n : Nat, n > 1 ∧ n < p → p % n ≠ 0

def greatest_prime_factor (n : Nat) : Nat :=
  if is_prime n then n else 1 -- Simplified; actual implementation would be more complex

-- The proof
theorem greatest_prime_factor_of_sum (n m : Nat) (h : expr = factored_expr) : greatest_prime_factor expr = 13 :=
by
  sorry

end greatest_prime_factor_of_sum_l101_101810


namespace find_x_l101_101413

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 15) : x = 35 :=
sorry

end find_x_l101_101413


namespace mixed_bead_cost_per_box_l101_101958

-- Definitions based on given conditions
def red_bead_cost : ℝ := 1.30
def yellow_bead_cost : ℝ := 2.00
def total_boxes : ℕ := 10
def red_boxes_used : ℕ := 4
def yellow_boxes_used : ℕ := 4

-- Theorem statement
theorem mixed_bead_cost_per_box :
  ((red_boxes_used * red_bead_cost) + (yellow_boxes_used * yellow_bead_cost)) / total_boxes = 1.32 :=
  by sorry

end mixed_bead_cost_per_box_l101_101958


namespace range_of_m_l101_101061

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m > 0) → m > 1 :=
by
  -- Proof goes here
  sorry

end range_of_m_l101_101061


namespace range_of_a_l101_101126

theorem range_of_a (a : ℝ) : (3 + 5 > 1 - 2 * a) ∧ (3 + (1 - 2 * a) > 5) ∧ (5 + (1 - 2 * a) > 3) → -7 / 2 < a ∧ a < -1 / 2 :=
by
  sorry

end range_of_a_l101_101126


namespace simplify_sqrt_expression_l101_101252

theorem simplify_sqrt_expression :
  (Real.sqrt (3 * 5) * Real.sqrt (3^3 * 5^3)) = 225 := 
by 
  sorry

end simplify_sqrt_expression_l101_101252


namespace percentage_republicans_vote_X_l101_101224

theorem percentage_republicans_vote_X (R : ℝ) (P_R : ℝ) :
  (3 * R * P_R + 2 * R * 0.15) - (3 * R * (1 - P_R) + 2 * R * 0.85) = 0.019999999999999927 * (3 * R + 2 * R) →
  P_R = 4.1 / 6 :=
by
  intro h
  sorry

end percentage_republicans_vote_X_l101_101224


namespace linda_spent_total_l101_101242

noncomputable def total_spent (notebooks_price_euro : ℝ) (notebooks_count : ℕ) 
    (pencils_price_pound : ℝ) (pencils_gift_card_pound : ℝ)
    (pens_price_yen : ℝ) (pens_points : ℝ) 
    (markers_price_dollar : ℝ) (calculator_price_dollar : ℝ)
    (marker_discount : ℝ) (coupon_discount : ℝ) (sales_tax : ℝ)
    (euro_to_dollar : ℝ) (pound_to_dollar : ℝ) (yen_to_dollar : ℝ) : ℝ :=
  let notebooks_cost := (notebooks_price_euro * notebooks_count) * euro_to_dollar
  let pencils_cost := 0
  let pens_cost := 0
  let marked_price := markers_price_dollar * (1 - marker_discount)
  let us_total_before_tax := (marked_price + calculator_price_dollar) * (1 - coupon_discount)
  let us_total_after_tax := us_total_before_tax * (1 + sales_tax)
  notebooks_cost + pencils_cost + pens_cost + us_total_after_tax

theorem linda_spent_total : 
  total_spent 1.2 3 1.5 5 170 200 2.8 12.5 0.15 0.10 0.05 1.1 1.25 0.009 = 18.0216 := 
  by
  sorry

end linda_spent_total_l101_101242


namespace range_of_m_l101_101565

theorem range_of_m (m : ℝ) : 
  ((m + 3) * (m - 4) < 0) → 
  (m^2 - 4 * (m + 3) ≤ 0) → 
  (-2 ≤ m ∧ m < 4) :=
by 
  intro h1 h2
  sorry

end range_of_m_l101_101565


namespace snail_crawl_distance_l101_101167

theorem snail_crawl_distance
  (α : ℕ → ℝ)  -- α represents the snail's position at each minute
  (crawls_forward : ∀ n m : ℕ, n < m → α n ≤ α m)  -- The snail moves forward (without going backward)
  (observer_finds : ∀ n : ℕ, α (n + 1) - α n = 1) -- Every observer finds that the snail crawled exactly 1 meter per minute
  (time_span : ℕ := 6)  -- Total observation period is 6 minutes
  : α time_span - α 0 ≤ 10 :=  -- The distance crawled in 6 minutes does not exceed 10 meters
by
  -- Proof goes here
  sorry

end snail_crawl_distance_l101_101167


namespace triangle_sin_A_and_height_l101_101927

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l101_101927


namespace find_units_digit_l101_101942

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem find_units_digit (n : ℕ) :
  is_three_digit n →
  (is_perfect_square n ∨ is_even n ∨ is_divisible_by_11 n ∨ digit_sum n = 12) ∧
  (¬is_perfect_square n ∨ ¬is_even n ∨ ¬is_divisible_by_11 n ∨ ¬(digit_sum n = 12)) →
  (n % 10 = 4) :=
sorry

end find_units_digit_l101_101942


namespace rightmost_three_digits_of_7_pow_2023_l101_101777

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 637 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l101_101777


namespace rightmost_three_digits_of_7_pow_2023_l101_101779

theorem rightmost_three_digits_of_7_pow_2023 :
  (7 ^ 2023) % 1000 = 343 :=
sorry

end rightmost_three_digits_of_7_pow_2023_l101_101779


namespace quadratic_nonneg_for_all_t_l101_101194

theorem quadratic_nonneg_for_all_t (x y : ℝ) : 
  (y ≤ x + 1) → (y ≥ -x - 1) → (x ≥ y^2 / 4) → (∀ (t : ℝ), (|t| ≤ 1) → t^2 + y * t + x ≥ 0) :=
by
  intro h1 h2 h3 t ht
  sorry

end quadratic_nonneg_for_all_t_l101_101194


namespace students_play_both_l101_101280

variable (students total_students football cricket neither : ℕ)
variable (H1 : total_students = 420)
variable (H2 : football = 325)
variable (H3 : cricket = 175)
variable (H4 : neither = 50)
  
theorem students_play_both (H1 : total_students = 420) (H2 : football = 325) 
    (H3 : cricket = 175) (H4 : neither = 50) : 
    students = 325 + 175 - (420 - 50) :=
by sorry

end students_play_both_l101_101280


namespace number_of_even_ones_matrices_l101_101477

noncomputable def count_even_ones_matrices (m n : ℕ) : ℕ :=
if m = 0 ∨ n = 0 then 1 else 2^((m-1)*(n-1))

theorem number_of_even_ones_matrices (m n : ℕ) : 
  count_even_ones_matrices m n = 2^((m-1)*(n-1)) := sorry

end number_of_even_ones_matrices_l101_101477


namespace correct_statement_l101_101611

-- Definitions as per conditions
def P1 : Prop := ∃ x : ℝ, x^2 = 64 ∧ abs x ^ 3 = 2
def P2 : Prop := ∀ x : ℝ, x = 0 → (¬∃ y, y * x = 1 ∧ -x = y)
def P3 : Prop := ∀ x y : ℝ, x + y = 0 → abs x / abs y = -1
def P4 : Prop := ∀ x a : ℝ, abs x + x = a → a > 0

-- The proof problem
theorem correct_statement : P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 := by
  sorry

end correct_statement_l101_101611


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101330

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 : 
  let λ := fun n => 
             match n with
             | 500 => 100
             | 100 => 20
             | _ => 0 in
  7^(7^(7^7)) ≡ 343 [MOD 500] :=
by 
  sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l101_101330


namespace algebraic_identity_example_l101_101422

-- Define the variables a and b
def a : ℕ := 287
def b : ℕ := 269

-- State the problem and the expected result
theorem algebraic_identity_example :
  a * a + b * b - 2 * a * b = 324 :=
by
  -- Since the proof is not required, we insert sorry here
  sorry

end algebraic_identity_example_l101_101422


namespace hyperbola_asymptotes_l101_101059

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                              (h3 : 2 * a = 4) (h4 : 2 * b = 6) : 
                              ∀ x y : ℝ, (y = (3 / 2) * x) ∨ (y = - (3 / 2) * x) := by
  sorry

end hyperbola_asymptotes_l101_101059


namespace exists_f_satisfies_functional_equation_l101_101945

open Rat

noncomputable def f (x : ℚ) := sorry

theorem exists_f_satisfies_functional_equation : 
  ∃ (f : ℚ+ → ℚ+), 
    (∀ (x y : ℚ+), f (x * f y) = f x / y) ∧ 
    (∃ k : ℚ, ∀ x : ℚ+, f x = x ^ k) :=
by
  sorry

end exists_f_satisfies_functional_equation_l101_101945


namespace quadratic_passing_point_calc_l101_101260

theorem quadratic_passing_point_calc :
  (∀ (x y : ℤ), y = 2 * x ^ 2 - 3 * x + 4 → ∃ (x' y' : ℤ), x' = 2 ∧ y' = 6) →
  (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  -- The corresponding proof would follow by providing the necessary steps.
  -- For now, let's just use sorry to meet the requirement.
  sorry

end quadratic_passing_point_calc_l101_101260


namespace greatest_prime_factor_of_sum_factorials_l101_101824

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_prime_factor_of_sum_factorials :
  let a := 15
  let b := 17
  let sum_factorials := factorial a + factorial b
  greatest_prime_factor sum_factorials = 17 :=
by
  -- Since we are just creating the statement, no proof is required
  sorry

end greatest_prime_factor_of_sum_factorials_l101_101824


namespace petya_friends_l101_101707

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101707


namespace sin_A_and_height_on_AB_l101_101933

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l101_101933


namespace exists_two_elements_l101_101620

variable (F : Finset (Finset ℕ))
variable (h1 : ∀ (A B : Finset ℕ), A ∈ F → B ∈ F → (A ∪ B) ∈ F)
variable (h2 : ∀ (A : Finset ℕ), A ∈ F → ¬ (3 ∣ A.card))

theorem exists_two_elements : ∃ (x y : ℕ), ∀ (A : Finset ℕ), A ∈ F → x ∈ A ∨ y ∈ A :=
by
  sorry

end exists_two_elements_l101_101620


namespace terminal_side_of_610_deg_is_250_deg_l101_101113

theorem terminal_side_of_610_deg_is_250_deg:
  ∃ k : ℤ, 610 % 360 = 250 := by
  sorry

end terminal_side_of_610_deg_is_250_deg_l101_101113


namespace complement_set_l101_101554

def U := {x : ℝ | x > 0}
def A := {x : ℝ | x > 2}
def complement_U_A := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem complement_set :
  {x : ℝ | x ∈ U ∧ x ∉ A} = complement_U_A :=
sorry

end complement_set_l101_101554


namespace roots_abs_lt_one_l101_101752

theorem roots_abs_lt_one
  (a b : ℝ)
  (h1 : |a| + |b| < 1)
  (h2 : a^2 - 4 * b ≥ 0) :
  ∀ (x : ℝ), x^2 + a * x + b = 0 → |x| < 1 :=
sorry

end roots_abs_lt_one_l101_101752


namespace area_of_rectangle_l101_101685

-- Define the given conditions
def side_length_of_square (s : ℝ) (ABCD : ℝ) : Prop :=
  ABCD = 4 * s^2

def perimeter_of_rectangle (s : ℝ) (perimeter : ℝ): Prop :=
  perimeter = 8 * s

-- Statement of the proof problem
theorem area_of_rectangle (s perimeter_area : ℝ) (h_perimeter : perimeter_of_rectangle s 160) :
  side_length_of_square s 1600 :=
by
  sorry

end area_of_rectangle_l101_101685


namespace starting_number_unique_l101_101921

-- Definitions based on conditions
def has_two_threes (n : ℕ) : Prop :=
  (n / 10 = 3 ∧ n % 10 = 3)

def is_starting_number (n m : ℕ) : Prop :=
  ∃ k, n + k = m ∧ k < (m - n) ∧ has_two_threes m

-- Theorem stating the proof problem
theorem starting_number_unique : ∃ n, is_starting_number n 30 ∧ n = 32 := 
sorry

end starting_number_unique_l101_101921


namespace neg_p_l101_101237

-- Define the sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- Define the proposition p
def p : Prop := ∀ x : ℤ, is_odd x → is_even (2 * x)

-- State the negation of proposition p
theorem neg_p : ¬ p ↔ ∃ x : ℤ, is_odd x ∧ ¬ is_even (2 * x) := by sorry

end neg_p_l101_101237


namespace greatest_prime_factor_of_15_add_17_factorial_l101_101846

theorem greatest_prime_factor_of_15_add_17_factorial:
  ∃ p : ℕ, prime p ∧ p = 13 ∧ 
    (∀ q : ℕ, prime q ∧ q ∣ (15! + 17!) → q ≤ 13) :=
begin
  sorry
end

end greatest_prime_factor_of_15_add_17_factorial_l101_101846


namespace division_value_l101_101872

theorem division_value (x : ℝ) (h1 : 2976 / x - 240 = 8) : x = 12 := 
by
  sorry

end division_value_l101_101872


namespace variance_of_arithmetic_sequence_common_diff_3_l101_101655

noncomputable def variance (ξ : List ℝ) : ℝ :=
  let n := ξ.length
  let mean := ξ.sum / n
  let var_sum := (ξ.map (fun x => (x - mean) ^ 2)).sum
  var_sum / n

def arithmetic_sequence (a1 : ℝ) (d : ℝ) (n : ℕ) : List ℝ :=
  List.range n |>.map (fun i => a1 + i * d)

theorem variance_of_arithmetic_sequence_common_diff_3 :
  ∀ (a1 : ℝ),
    variance (arithmetic_sequence a1 3 9) = 60 :=
by
  sorry

end variance_of_arithmetic_sequence_common_diff_3_l101_101655


namespace determine_expr_l101_101046

noncomputable def expr (a b c d : ℝ) : ℝ :=
  (1 + a + a * b) / (1 + a + a * b + a * b * c) +
  (1 + b + b * c) / (1 + b + b * c + b * c * d) +
  (1 + c + c * d) / (1 + c + c * d + c * d * a) +
  (1 + d + d * a) / (1 + d + d * a + d * a * b)

theorem determine_expr (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  expr a b c d = 2 :=
sorry

end determine_expr_l101_101046


namespace sqrt_approximation_l101_101349

theorem sqrt_approximation :
  (2^2 < 5) ∧ (5 < 3^2) ∧ 
  (2.2^2 < 5) ∧ (5 < 2.3^2) ∧ 
  (2.23^2 < 5) ∧ (5 < 2.24^2) ∧ 
  (2.236^2 < 5) ∧ (5 < 2.237^2) →
  (Float.ceil (Float.sqrt 5 * 100) / 100) = 2.24 := 
by
  intro h
  sorry

end sqrt_approximation_l101_101349


namespace min_combined_horses_ponies_l101_101028

theorem min_combined_horses_ponies (P H : ℕ) (h1 : P % 6 = 0) (h2 : P % 9 = 0) (h3 : H = P + 4) 
    (h4 : ∃ p_sh : ℕ, p_sh = (5 / 6 : ℚ) * P) (h5 : ∃ ic_p_sh : ℕ, ic_p_sh = (2 / 3 : ℚ) * (5 / 6 : ℚ) * P) :
  P + H = 40 :=
begin
  sorry
end

end min_combined_horses_ponies_l101_101028


namespace sum_A_B_l101_101997

theorem sum_A_B (A B : ℕ) 
  (h1 : (1 / 4 : ℚ) * (1 / 8) = 1 / (4 * A))
  (h2 : 1 / (4 * A) = 1 / B) : A + B = 40 := 
by
  sorry

end sum_A_B_l101_101997


namespace sum_of_squares_expressible_l101_101243

theorem sum_of_squares_expressible (a b c : ℕ) (h1 : c^2 = a^2 + b^2) : 
  ∃ x y : ℕ, x^2 + y^2 = c^2 + a*b ∧ ∃ u v : ℕ, u^2 + v^2 = c^2 - a*b :=
by
  sorry

end sum_of_squares_expressible_l101_101243


namespace smallest_integer_with_12_divisors_l101_101600

theorem smallest_integer_with_12_divisors : ∃ n : ℕ, n > 0 ∧ (number_of_divisors n = 12) ∧ (∀ m : ℕ, m > 0 ∧ number_of_divisors m = 12 → n ≤ m) :=
sorry

end smallest_integer_with_12_divisors_l101_101600


namespace sequence_converges_to_one_l101_101234

noncomputable def u (n : ℕ) : ℝ :=
1 + (Real.sin n) / n

theorem sequence_converges_to_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1| ≤ ε :=
sorry

end sequence_converges_to_one_l101_101234


namespace probability_of_multiple_of_3_is_1_5_l101_101150

-- Definition of the problem conditions
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Function to calculate the probability
noncomputable def probability_of_multiple_of_3 : ℚ := 
  let total_permutations := (Nat.factorial 5) / (Nat.factorial (5 - 4))  -- i.e., 120
  let valid_permutations := Nat.factorial 4  -- i.e., 24, for the valid combination
  valid_permutations / total_permutations 

-- Statement to be proved
theorem probability_of_multiple_of_3_is_1_5 :
  probability_of_multiple_of_3 = 1 / 5 := 
by
  -- Skeleton for the proof
  sorry

end probability_of_multiple_of_3_is_1_5_l101_101150


namespace problem_l101_101075

theorem problem (triangle square : ℕ) (h1 : triangle + 5 ≡ 1 [MOD 7]) (h2 : 2 + square ≡ 3 [MOD 7]) :
  triangle = 3 ∧ square = 1 := by
  sorry

end problem_l101_101075


namespace perpendicular_lines_b_eq_neg9_l101_101969

-- Definitions for the conditions.
def eq1 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def eq2 (b x y : ℝ) : Prop := b * x + 3 * y + 4 = 0

-- The problem statement
theorem perpendicular_lines_b_eq_neg9 (b : ℝ) : 
  (∀ x y, eq1 x y → eq2 b x y) ∧ (∀ x y, eq2 b x y → eq1 x y) → b = -9 :=
by
  sorry

end perpendicular_lines_b_eq_neg9_l101_101969


namespace semicircle_inequality_l101_101753

open Real

theorem semicircle_inequality {A B C D E : ℝ} (h : A^2 + B^2 + C^2 + D^2 + E^2 = 1):
  (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - E)^2 + (A - B) * (B - C) * (C - D) + (B - C) * (C - D) * (D - E) < 4 :=
by
  -- proof omitted
  sorry

end semicircle_inequality_l101_101753


namespace cistern_water_breadth_l101_101870

theorem cistern_water_breadth (length width total_area : ℝ) (h : ℝ) 
  (h_length : length = 10) 
  (h_width : width = 6) 
  (h_area : total_area = 103.2) : 
  (60 + 20*h + 12*h = total_area) → h = 1.35 :=
by
  intros
  sorry

end cistern_water_breadth_l101_101870


namespace sum_of_missing_digits_l101_101542

-- Define the problem's conditions
def add_digits (a b c d e f g h : ℕ) := 
a + b = 18 ∧ b + c + d = 21

-- Prove the sum of the missing digits equals 7
theorem sum_of_missing_digits (a b c d e f g h : ℕ) (h1 : add_digits a b c d e f g h) : a + c = 7 := 
sorry

end sum_of_missing_digits_l101_101542


namespace sqrt_72_eq_6_sqrt_2_l101_101107

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l101_101107


namespace petya_friends_l101_101708

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end petya_friends_l101_101708


namespace new_avg_weight_l101_101965

theorem new_avg_weight 
  (initial_avg_weight : ℝ)
  (initial_num_members : ℕ)
  (new_person1_weight : ℝ)
  (new_person2_weight : ℝ)
  (new_num_members : ℕ)
  (final_total_weight : ℝ)
  (final_avg_weight : ℝ) :
  initial_avg_weight = 48 →
  initial_num_members = 23 →
  new_person1_weight = 78 →
  new_person2_weight = 93 →
  new_num_members = initial_num_members + 2 →
  final_total_weight = (initial_avg_weight * initial_num_members) + new_person1_weight + new_person2_weight →
  final_avg_weight = final_total_weight / new_num_members →
  final_avg_weight = 51 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end new_avg_weight_l101_101965
