import Mathlib

namespace gcd_of_18_and_30_l266_266248

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266248


namespace largest_divisor_of_expression_l266_266909

theorem largest_divisor_of_expression (n : ℤ) : ∃ k : ℤ, k = 6 ∧ (n^3 - n + 15) % k = 0 := 
by
  use 6
  sorry

end largest_divisor_of_expression_l266_266909


namespace min_value_of_a_l266_266686

theorem min_value_of_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y))) : 
  a ≥ Real.sqrt 2 :=
sorry -- Proof is omitted

end min_value_of_a_l266_266686


namespace least_multiple_of_21_gt_380_l266_266924

theorem least_multiple_of_21_gt_380 : ∃ n : ℕ, (21 * n > 380) ∧ (21 * n = 399) :=
sorry

end least_multiple_of_21_gt_380_l266_266924


namespace proposition_is_false_l266_266820

noncomputable def false_proposition : Prop :=
¬(∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), Real.sin x + Real.cos x ≥ 2)

theorem proposition_is_false : false_proposition :=
by
  sorry

end proposition_is_false_l266_266820


namespace greatest_xy_value_l266_266392

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l266_266392


namespace persimmons_count_l266_266458

variables {P T : ℕ}

-- Conditions from the problem
axiom total_eq : P + T = 129
axiom diff_eq : P = T - 43

-- Theorem to prove that there are 43 persimmons
theorem persimmons_count : P = 43 :=
by
  -- Putting the proof placeholder
  sorry

end persimmons_count_l266_266458


namespace least_prime_factor_five_power_difference_l266_266925

theorem least_prime_factor_five_power_difference : 
  ∃ p : ℕ, (Nat.Prime p ∧ p ∣ (5^4 - 5^3)) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (5^4 - 5^3) → p ≤ q) := 
sorry

end least_prime_factor_five_power_difference_l266_266925


namespace geometric_series_sum_l266_266963

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1/3) :
  (∑' n : ℕ, a * r^n) = 3/2 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l266_266963


namespace gcd_18_30_l266_266277

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266277


namespace area_enclosed_by_curve_l266_266061

theorem area_enclosed_by_curve : 
  let curve_eq (x y : ℝ) := abs (x - 1) + abs (y - 1) = 1 in
  (area of the region enclosed by curve_eq equals 2) :=
begin
  sorry
end

end area_enclosed_by_curve_l266_266061


namespace hundredth_power_remainders_l266_266933

theorem hundredth_power_remainders (a : ℤ) : 
  (a % 5 = 0 → a^100 % 125 = 0) ∧ (a % 5 ≠ 0 → a^100 % 125 = 1) :=
by
  sorry

end hundredth_power_remainders_l266_266933


namespace smallest_sum_of_three_integers_l266_266069

theorem smallest_sum_of_three_integers (a b c : ℕ) (h1: a ≠ b) (h2: b ≠ c) (h3: a ≠ c) (h4: a * b * c = 72) :
  a + b + c = 13 :=
sorry

end smallest_sum_of_three_integers_l266_266069


namespace books_in_special_collection_at_beginning_of_month_l266_266641

theorem books_in_special_collection_at_beginning_of_month
  (loaned_out_real : Real)
  (loaned_out_books : Int)
  (returned_ratio : Real)
  (books_at_end : Int)
  (B : Int)
  (h1 : loaned_out_real = 49.99999999999999)
  (h2 : loaned_out_books = 50)
  (h3 : returned_ratio = 0.70)
  (h4 : books_at_end = 60)
  (h5 : loaned_out_books = Int.floor loaned_out_real)
  (h6 : ∀ (loaned_books : Int), loaned_books ≤ loaned_out_books → returned_ratio * loaned_books + (loaned_books - returned_ratio * loaned_books) = loaned_books)
  : B = 75 :=
by
  sorry

end books_in_special_collection_at_beginning_of_month_l266_266641


namespace greatest_xy_l266_266396

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l266_266396


namespace gcf_of_48_180_120_l266_266617

theorem gcf_of_48_180_120 : Nat.gcd (Nat.gcd 48 180) 120 = 12 := by
  sorry

end gcf_of_48_180_120_l266_266617


namespace find_b_l266_266992

noncomputable def point (x y : Float) : Float × Float := (x, y)

def line_y_eq_b_plus_x (b x : Float) : Float := b + x

def intersects_y_axis (b : Float) : Float × Float := (0, b)

def intersects_x_axis (b : Float) : Float × Float := (-b, 0)

def intersects_x_eq_5 (b : Float) : Float × Float := (5, b + 5)

def area_triangle_qrs (b : Float) : Float :=
  0.5 * (5 + b) * (b + 5)

def area_triangle_qop (b : Float) : Float :=
  0.5 * b * b

theorem find_b (b : Float) (h : b > 0) (h_area_ratio : area_triangle_qrs b / area_triangle_qop b = 4 / 9) : b = 5 :=
by
  sorry

end find_b_l266_266992


namespace cos_value_of_tan_third_quadrant_l266_266680

theorem cos_value_of_tan_third_quadrant (x : ℝ) (h1 : Real.tan x = 4 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -3 / 5 := 
sorry

end cos_value_of_tan_third_quadrant_l266_266680


namespace sufficient_but_not_necessary_condition_l266_266170

def vectors_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

def vector_a (x : ℝ) : ℝ × ℝ := (2, x - 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 1, 4)

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  x = 3 → vectors_parallel (vector_a x) (vector_b x) ∧
  vectors_parallel (vector_a 3) (vector_b 3) :=
by
  sorry

end sufficient_but_not_necessary_condition_l266_266170


namespace smallest_sum_of_squares_l266_266749

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l266_266749


namespace remainder_101_pow_37_mod_100_l266_266198

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l266_266198


namespace remainder_101_pow_37_mod_100_l266_266200

theorem remainder_101_pow_37_mod_100 :
  (101: ℤ) ≡ 1 [MOD 100] →
  (101: ℤ)^37 ≡ 1 [MOD 100] :=
by
  sorry

end remainder_101_pow_37_mod_100_l266_266200


namespace gcd_18_30_is_6_l266_266270

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266270


namespace robert_saves_5_dollars_l266_266793

theorem robert_saves_5_dollars :
  let original_price := 50
  let promotion_c_discount (price : ℕ) := price * 20 / 100
  let promotion_d_discount (price : ℕ) := 15
  let cost_promotion_c := original_price + (original_price - promotion_c_discount original_price)
  let cost_promotion_d := original_price + (original_price - promotion_d_discount original_price)
  (cost_promotion_c - cost_promotion_d) = 5 :=
by
  sorry

end robert_saves_5_dollars_l266_266793


namespace draw_is_unfair_ensure_fair_draw_l266_266151

open ProbabilityTheory MeasureTheory

-- Definitions for the given conditions:
def Card := {rank : ℕ // 6 ≤ rank ∧ rank ≤ 14} -- Ranks 6 to Ace (6 to 14)
def Deck := Finset (Fin 36) -- 36 unique cards
noncomputable def suit_high_rank_count (d : Deck) (v_card : Fin 36) (m_card : Fin 36) : ℕ := 
  -- Count how many cards are higher than Volodya's card
  card.count (λ c, c.val > v_card.val) d

-- Volodya draws first, then Masha draws:
variables (d : Deck) (v_card m_card : Fin 36)

-- Masha wins if she draws a card with a higher rank than Volodya’s card
def masha_wins := ∃ (m_card : Fin 36), (m_card ∈ d) ∧ (m_card.val > v_card.val)

-- Volodya wins if Masha doesn't win (Masha loses)
def volodya_wins := ¬ masha_wins

theorem draw_is_unfair (d : Deck) (v_card m_card : Fin 36) :
  (volodya_wins d v_card m_card) → ¬ (masha_wins d v_card) := sorry

-- To make it fair, we can introduce a suit hierarchy:
def suits := {"Hearts", "Diamonds", "Clubs", "Spades"}
def suit_order : suits → suits → Prop
| "Spades" "Hearts" := true
| "Hearts" "Diamonds" := true
| "Diamonds" "Clubs" := true
| "Clubs" "Spades" := false
| _, _ := false

-- A fair draw means using the suit_order to rank otherwise equal cards:
def fair_draw :=
  ∀ (c1 c2 : Card), (c1.rank = c2.rank → suit_order c1.suit c2.suit)

theorem ensure_fair_draw : fair_draw := sorry

end draw_is_unfair_ensure_fair_draw_l266_266151


namespace probability_sum_divisible_by_3_l266_266139

theorem probability_sum_divisible_by_3:
  ∀ (n a b c : ℕ), a + b + c = n →
  4 * (a^3 + b^3 + c^3 + 6 * a * b * c) ≥ (a + b + c)^3 :=
by 
  intros n a b c habc_eq_n
  sorry

end probability_sum_divisible_by_3_l266_266139


namespace final_segment_position_correct_l266_266638

def initial_segment : ℝ × ℝ := (1, 6)
def rotate_180_about (p : ℝ) (x : ℝ) : ℝ := p - (x - p)
def first_rotation_segment : ℝ × ℝ := (rotate_180_about 2 6, rotate_180_about 2 1)
def second_rotation_segment : ℝ × ℝ := (rotate_180_about 1 3, rotate_180_about 1 (-2))

theorem final_segment_position_correct :
  second_rotation_segment = (-1, 4) :=
by
  -- This is a placeholder for the actual proof.
  sorry

end final_segment_position_correct_l266_266638


namespace iodine_initial_amount_l266_266871

theorem iodine_initial_amount (half_life : ℕ) (days_elapsed : ℕ) (final_amount : ℕ) (initial_amount : ℕ) :
  half_life = 8 → days_elapsed = 24 → final_amount = 2 → initial_amount = final_amount * 2 ^ (days_elapsed / half_life) → initial_amount = 16 :=
by
  intros h_half_life h_days_elapsed h_final_amount h_initial_exp
  rw [h_half_life, h_days_elapsed, h_final_amount] at h_initial_exp
  norm_num at h_initial_exp
  exact h_initial_exp

end iodine_initial_amount_l266_266871


namespace prime_factorization_min_x_l266_266718

-- Define the conditions
variable (x y : ℕ) (a b e f : ℕ)

-- Given conditions: x and y are positive integers, and 5x^7 = 13y^11
axiom condition1 : 0 < x ∧ 0 < y
axiom condition2 : 5 * x^7 = 13 * y^11

-- Prove the mathematical equivalence
theorem prime_factorization_min_x (a b e f : ℕ) 
    (hx : 5 * x^7 = 13 * y^11)
    (h_prime : a = 13 ∧ b = 5 ∧ e = 6 ∧ f = 1) :
    a + b + e + f = 25 :=
sorry

end prime_factorization_min_x_l266_266718


namespace percentage_loss_l266_266639

theorem percentage_loss (CP SP : ℝ) (h₁ : CP = 1400) (h₂ : SP = 1232) :
  ((CP - SP) / CP) * 100 = 12 :=
by
  sorry

end percentage_loss_l266_266639


namespace sum_three_distinct_zero_l266_266039

variable {R : Type} [Field R]

theorem sum_three_distinct_zero
  (a b c x y : R)
  (h1 : a ^ 3 + a * x + y = 0)
  (h2 : b ^ 3 + b * x + y = 0)
  (h3 : c ^ 3 + c * x + y = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  a + b + c = 0 := by
  sorry

end sum_three_distinct_zero_l266_266039


namespace solution_set_of_inequality_l266_266766

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 - 7*x + 12 < 0 ↔ 3 < x ∧ x < 4 :=
by {
  sorry
}

end solution_set_of_inequality_l266_266766


namespace cauliflower_production_proof_l266_266940

theorem cauliflower_production_proof (x y : ℕ) 
  (h1 : y^2 - x^2 = 401)
  (hx : x > 0)
  (hy : y > 0) :
  y^2 = 40401 :=
by
  sorry

end cauliflower_production_proof_l266_266940


namespace simplify_fraction_l266_266901

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l266_266901


namespace nancy_games_this_month_l266_266885

-- Define the variables and conditions from the problem
def went_games_last_month : ℕ := 8
def plans_games_next_month : ℕ := 7
def total_games : ℕ := 24

-- Let's calculate the games this month and state the theorem
def games_last_and_next : ℕ := went_games_last_month + plans_games_next_month
def games_this_month : ℕ := total_games - games_last_and_next

-- The theorem statement
theorem nancy_games_this_month : games_this_month = 9 := by
  -- Proof is omitted for the sake of brevity
  sorry

end nancy_games_this_month_l266_266885


namespace total_cost_maria_l266_266580

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end total_cost_maria_l266_266580


namespace det_matrix_example_l266_266366

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem det_matrix_example : det_2x2 4 5 2 3 = 2 :=
by
  sorry

end det_matrix_example_l266_266366


namespace testing_schemes_count_l266_266075

theorem testing_schemes_count :
  let genuine_products := 5
  let defective_products := 4
  let total_tests := 10
  let required_tests := 6
  /*
    The number of ways to choose and arrange the products given the conditions.
    1. Choosing 1 defective product for the 6th test
    2. Choosing 2 out of 5 genuine products
    3. Arranging remaining 3 defective and 2 genuine in first 5 tests
  */
  (choose defective_products 1) * (choose genuine_products 2) * (factorial 5) = 4800 := sorry

end testing_schemes_count_l266_266075


namespace bus_trip_distance_l266_266210

-- Defining the problem variables
variables (x D : ℝ) -- x: speed in mph, D: total distance in miles

-- Main theorem stating the problem
theorem bus_trip_distance
  (h1 : 0 < x) -- speed of the bus is positive
  (h2 : (2 * x + 3 * (D - 2 * x) / (2 / 3 * x) / 2 + 0.75) - 2 - 4 = 0)
  -- The first scenario summarising the travel and delays
  (h3 : ((2 * x + 120) / x + 3 * (D - (2 * x + 120)) / (2 / 3 * x) / 2 + 0.75) - 3 = 0)
  -- The second scenario summarising the travel and delays; accident 120 miles further down
  : D = 720 := sorry

end bus_trip_distance_l266_266210


namespace smallest_possible_sum_of_squares_l266_266741

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l266_266741


namespace mary_sheep_purchase_l266_266171

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end mary_sheep_purchase_l266_266171


namespace smallest_sum_of_squares_l266_266751

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l266_266751


namespace calc_value_l266_266229

theorem calc_value : (3000 * (3000 ^ 2999) * 2 = 2 * 3000 ^ 3000) := 
by
  sorry

end calc_value_l266_266229


namespace exam_question_correct_count_l266_266706

theorem exam_question_correct_count (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 110) : C = 34 :=
by
  sorry

end exam_question_correct_count_l266_266706


namespace john_drove_total_distance_l266_266875

-- Define different rates and times for John's trip
def rate1 := 45 -- mph
def rate2 := 55 -- mph
def time1 := 2 -- hours
def time2 := 3 -- hours

-- Define the distances for each segment of the trip
def distance1 := rate1 * time1
def distance2 := rate2 * time2

-- Define the total distance
def total_distance := distance1 + distance2

-- The theorem to prove that John drove 255 miles in total
theorem john_drove_total_distance : total_distance = 255 :=
by
  sorry

end john_drove_total_distance_l266_266875


namespace buicks_count_l266_266711

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l266_266711


namespace remainder_b94_mod_55_eq_29_l266_266879

theorem remainder_b94_mod_55_eq_29 :
  (5^94 + 7^94) % 55 = 29 := 
by
  -- conditions: local definitions for bn, modulo, etc.
  sorry

end remainder_b94_mod_55_eq_29_l266_266879


namespace intersecting_lines_a_plus_b_l266_266910

theorem intersecting_lines_a_plus_b :
  ∃ (a b : ℝ), (∀ x y : ℝ, (x = 1 / 3 * y + a) ∧ (y = 1 / 3 * x + b) → (x = 3 ∧ y = 4)) ∧ a + b = 14 / 3 :=
sorry

end intersecting_lines_a_plus_b_l266_266910


namespace students_on_field_trip_l266_266886

theorem students_on_field_trip (vans: ℕ) (capacity_per_van: ℕ) (adults: ℕ) 
  (H_vans: vans = 3) 
  (H_capacity_per_van: capacity_per_van = 5) 
  (H_adults: adults = 3) : 
  (vans * capacity_per_van - adults = 12) :=
by
  sorry

end students_on_field_trip_l266_266886


namespace fractional_inequality_solution_l266_266073

theorem fractional_inequality_solution (x : ℝ) :
  (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end fractional_inequality_solution_l266_266073


namespace closest_number_l266_266594

theorem closest_number
  (a b c : ℝ)
  (h₀ : a = Real.sqrt 5)
  (h₁ : b = 3)
  (h₂ : b = (a + c) / 2) :
  abs (c - 3.5) ≤ abs (c - 2) ∧ abs (c - 3.5) ≤ abs (c - 2.5) ∧ abs (c - 3.5) ≤ abs (c - 3)  :=
by
  sorry

end closest_number_l266_266594


namespace prime_sequence_constant_l266_266127

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Condition: There exists a constant sequence of primes such that the given recurrence relation holds.
theorem prime_sequence_constant (p : ℕ) (k : ℤ) (n : ℕ) 
  (h1 : 1 ≤ n)
  (h2 : ∀ m ≥ 1, is_prime (p + m))
  (h3 : p + k = p + p + k) :
  ∀ m ≥ 1, p + m = p :=
sorry

end prime_sequence_constant_l266_266127


namespace replace_with_30_digit_nat_number_l266_266597

noncomputable def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

theorem replace_with_30_digit_nat_number (a : Fin 10 → ℕ) (h : ∀ i, is_three_digit (a i)) :
  ∃ b : ℕ, (b < 10^30 ∧ ∃ x : ℤ, (a 9) * x^9 + (a 8) * x^8 + (a 7) * x^7 + (a 6) * x^6 + (a 5) * x^5 + 
           (a 4) * x^4 + (a 3) * x^3 + (a 2) * x^2 + (a 1) * x + (a 0) = b) :=
by
  sorry

end replace_with_30_digit_nat_number_l266_266597


namespace find_pointA_coordinates_l266_266868

-- Define point B
def pointB : ℝ × ℝ := (4, -1)

-- Define the symmetry condition with respect to the x-axis
def symmetricWithRespectToXAxis (p₁ p₂ : ℝ × ℝ) : Prop :=
  p₁.1 = p₂.1 ∧ p₁.2 = -p₂.2

-- Theorem statement: Prove the coordinates of point A given the conditions
theorem find_pointA_coordinates :
  ∃ A : ℝ × ℝ, symmetricWithRespectToXAxis pointB A ∧ A = (4, 1) :=
by
  sorry

end find_pointA_coordinates_l266_266868


namespace range_of_omega_for_zeros_in_interval_l266_266845

noncomputable def f (ω : ℝ) (x : ℝ) := Real.cos (ω * x) - 1

theorem range_of_omega_for_zeros_in_interval (ω : ℝ) (hω_positve : ω > 0) :
  (∀ x ∈ set.Icc 0 (2 * Real.pi), f ω x = 0 → 2 ≤ ω ∧ ω < 3) :=
sorry

end range_of_omega_for_zeros_in_interval_l266_266845


namespace varies_fix_l266_266697

variable {x y z : ℝ}

theorem varies_fix {k j : ℝ} 
  (h1 : x = k * y^4)
  (h2 : y = j * z^(1/3)) : x = (k * j^4) * z^(4/3) := by
  sorry

end varies_fix_l266_266697


namespace find_k_l266_266433

open_locale big_operators

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : ∑' n, (7 * n - 3) / k^n = 2) :
  k = 2 + 3 * real.sqrt 2 / 2 :=
sorry

end find_k_l266_266433


namespace convex_quadrilateral_inequality_l266_266003

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

theorem convex_quadrilateral_inequality
    (AB CD BC AD AC BD : ℝ)
    (h : AB * CD + BC * AD >= AC * BD)
    (convex_quadrilateral : Prop) :
  AB * CD + BC * AD >= AC * BD :=
by
  sorry

end convex_quadrilateral_inequality_l266_266003


namespace smallest_sum_of_squares_l266_266745

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l266_266745


namespace total_money_calculation_l266_266640

theorem total_money_calculation (N50 N500 Total_money : ℕ) 
( h₁ : N50 = 37 ) 
( h₂ : N50 + N500 = 54 ) :
Total_money = N50 * 50 + N500 * 500 ↔ Total_money = 10350 := 
by 
  sorry

end total_money_calculation_l266_266640


namespace evaluate_expression_l266_266821

theorem evaluate_expression :
  (3 ^ (1 ^ (0 ^ 8)) + ( (3 ^ 1) ^ 0 ) ^ 8) = 4 :=
by
  sorry

end evaluate_expression_l266_266821


namespace range_of_omega_l266_266846

theorem range_of_omega :
  ∀ (ω : ℝ), 
  (0 < ω) → 
  (∀ x, x ∈ set.Icc (0 : ℝ) (2 * Real.pi) → cos (ω * x) - 1 = 0 → x ∈ {0, 2 * Real.pi, 4 * Real.pi}) →
  (2 ≤ ω ∧ ω < 3) :=
by
  intros ω hω_pos hzeros
  sorry

end range_of_omega_l266_266846


namespace gcd_of_18_and_30_l266_266289

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266289


namespace evaluate_g_g_g_25_l266_266575

def g (x : ℤ) : ℤ :=
  if x < 10 then x^2 - 9 else x - 20

theorem evaluate_g_g_g_25 : g (g (g 25)) = -4 := by
  sorry

end evaluate_g_g_g_25_l266_266575


namespace blocks_per_box_l266_266049

theorem blocks_per_box (total_blocks : ℕ) (boxes : ℕ) (h1 : total_blocks = 16) (h2 : boxes = 8) : total_blocks / boxes = 2 :=
by
  sorry

end blocks_per_box_l266_266049


namespace cylinder_volume_ratio_l266_266786

theorem cylinder_volume_ratio
  (h : ℝ)
  (r1 : ℝ)
  (r3 : ℝ := 3 * r1)
  (V1 : ℝ := 40) :
  let V2 := π * r3^2 * h
  (π * r1^2 * h = V1) → 
  V2 = 360 := by
{
  sorry
}

end cylinder_volume_ratio_l266_266786


namespace function_properties_l266_266847

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l266_266847


namespace adult_ticket_cost_l266_266102

-- Definitions from the conditions
def total_amount : ℕ := 35
def child_ticket_cost : ℕ := 3
def num_children : ℕ := 9

-- The amount spent on children’s tickets
def total_child_ticket_cost : ℕ := num_children * child_ticket_cost

-- The remaining amount after purchasing children’s tickets
def remaining_amount : ℕ := total_amount - total_child_ticket_cost

-- The adult ticket cost should be equal to the remaining amount
theorem adult_ticket_cost : remaining_amount = 8 :=
by sorry

end adult_ticket_cost_l266_266102


namespace max_value_fractions_l266_266720

noncomputable def maxFractions (a b c : ℝ) : ℝ :=
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c)

theorem max_value_fractions (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
    (h_sum : a + b + c = 2) :
    maxFractions a b c ≤ 1 ∧ 
    (a = 2 / 3 ∧ b = 2 / 3 ∧ c = 2 / 3 → maxFractions a b c = 1) := 
  by
    sorry

end max_value_fractions_l266_266720


namespace recurrence_relation_l266_266406

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l266_266406


namespace recurrence_relation_p_series_l266_266405

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l266_266405


namespace simplify_fraction_l266_266900

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l266_266900


namespace find_value_of_f_eq_l266_266355

noncomputable def f (ω ω φ x : ℝ) : ℝ := sin (ω * x + φ)

theorem find_value_of_f_eq :
  ∀ (ω φ : ℝ), 
    (∀ x, x ∈ set.Ioo (π / 6) (2 * π / 3) → (f ω φ x) ≤ (f ω φ (x + 1e-10))) → -- monotonically increasing
    (ω * ∓ (π / 6) + φ) = (φ + ω * (2 * π / 3)) → -- symmetric axes
    f ω φ (-(5 * π / 12)) = sqrt 3 / 2 :=
by
  sorry

end find_value_of_f_eq_l266_266355


namespace gcd_18_30_l266_266251

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266251


namespace anya_initial_seat_l266_266865

theorem anya_initial_seat (V G D E A : ℕ) (A' : ℕ) 
  (h1 : V + G + D + E + A = 15)
  (h2 : V + 1 ≠ A')
  (h3 : G - 3 ≠ A')
  (h4 : (D = A' → E ≠ A') ∧ (E = A' → D ≠ A'))
  (h5 : A = 3 + 2)
  : A = 3 := by
  sorry

end anya_initial_seat_l266_266865


namespace derivative_correct_l266_266604

noncomputable def derivative_of_composite_function (x : ℝ) : Prop :=
  let y := (5 * x - 3) ^ 3
  let dy_dx := 3 * (5 * x - 3) ^ 2 * 5
  dy_dx = 15 * (5 * x - 3) ^ 2

theorem derivative_correct (x : ℝ) : derivative_of_composite_function x :=
by
  sorry

end derivative_correct_l266_266604


namespace gcd_18_30_l266_266314

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266314


namespace more_volunteers_needed_l266_266437

theorem more_volunteers_needed
    (required_volunteers : ℕ)
    (students_per_class : ℕ)
    (num_classes : ℕ)
    (teacher_volunteers : ℕ)
    (total_volunteers : ℕ) :
    required_volunteers = 50 →
    students_per_class = 5 →
    num_classes = 6 →
    teacher_volunteers = 13 →
    total_volunteers = (students_per_class * num_classes) + teacher_volunteers →
    (required_volunteers - total_volunteers) = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end more_volunteers_needed_l266_266437


namespace space_between_trees_l266_266549

theorem space_between_trees (tree_count : ℕ) (tree_space : ℕ) (road_length : ℕ)
  (h1 : tree_space = 1) (h2 : tree_count = 13) (h3 : road_length = 157) :
  (road_length - tree_count * tree_space) / (tree_count - 1) = 12 := by
  sorry

end space_between_trees_l266_266549


namespace gcd_18_30_l266_266253

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266253


namespace greatest_xy_l266_266394

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l266_266394


namespace simplify_and_evaluate_l266_266446

noncomputable def x : ℕ := 2023
noncomputable def y : ℕ := 2

theorem simplify_and_evaluate :
  (x + 2 * y)^2 - ((x^3 + 4 * x^2 * y) / x) = 16 :=
by
  sorry

end simplify_and_evaluate_l266_266446


namespace inequality_div_two_l266_266538

theorem inequality_div_two (x y : ℝ) (h : x > y) : x / 2 > y / 2 := sorry

end inequality_div_two_l266_266538


namespace geometric_sum_s5_l266_266986

-- Definitions of the geometric sequence and its properties
variable {α : Type*} [Field α] (a : α)

-- The common ratio of the geometric sequence
def common_ratio : α := 2

-- The n-th term of the geometric sequence
def a_n (n : ℕ) : α := a * common_ratio ^ n

-- The sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : α := (a * (1 - common_ratio ^ n)) / (1 - common_ratio)

-- Define the arithmetic sequence property
def aro_seq_property (a_1 a_2 a_5 : α) : Prop := 2 * a_2 = 6 + a_5

-- Define a_2 and a_5 in terms of a
def a2 := a * common_ratio
def a5 := a * common_ratio ^ 4

-- State the main proof problem
theorem geometric_sum_s5 : 
  aro_seq_property a (a2 a) (a5 a) → 
  S_n a 5 = -31 / 2 :=
by
  sorry

end geometric_sum_s5_l266_266986


namespace coins_in_bag_l266_266938

theorem coins_in_bag (x : ℝ) (h : x + 0.5 * x + 0.25 * x = 140) : x = 80 :=
by sorry

end coins_in_bag_l266_266938


namespace gcd_18_30_l266_266254

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266254


namespace factor_1000000000001_l266_266772

theorem factor_1000000000001 : ∃ a b c : ℕ, 1000000000001 = a * b * c ∧ a = 73 ∧ b = 137 ∧ c = 99990001 :=
by {
  sorry
}

end factor_1000000000001_l266_266772


namespace second_smallest_prime_perimeter_l266_266520

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def scalene_triangle (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def prime_perimeter (a b c : ℕ) : Prop := 
  is_prime (a + b + c)

def different_primes (a b c : ℕ) : Prop := 
  is_prime a ∧ is_prime b ∧ is_prime c

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ), 
  scalene_triangle a b c ∧ 
  different_primes a b c ∧ 
  prime_perimeter a b c ∧ 
  a + b + c = 29 := 
sorry

end second_smallest_prime_perimeter_l266_266520


namespace rahul_matches_played_l266_266600

-- Define the conditions of the problem
variable (m : ℕ) -- number of matches Rahul has played so far
variable (runs_before : ℕ := 51 * m) -- total runs before today's match
variable (runs_today : ℕ := 69) -- runs scored today
variable (new_average : ℕ := 54) -- new batting average after today's match

-- The equation derived from the conditions
def batting_average_equation : Prop :=
  new_average * (m + 1) = runs_before + runs_today

-- The problem: prove that m = 5 given the conditions
theorem rahul_matches_played (h : batting_average_equation m) : m = 5 :=
  sorry

end rahul_matches_played_l266_266600


namespace simplify_fraction_l266_266898

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem simplify_fraction : (180 = 2^2 * 3^2 * 5) ∧ (270 = 2 * 3^3 * 5) ∧ (gcd 180 270 = 90) →
  180 / 270 = 2 / 3 :=
by
  intro h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  sorry -- Proof is omitted

end simplify_fraction_l266_266898


namespace find_x_l266_266552

theorem find_x
  (PQR_straight : ∀ x y : ℝ, x + y = 76 → 3 * x + 2 * y = 180)
  (h : x + y = 76) :
  x = 28 :=
by
  sorry

end find_x_l266_266552


namespace candy_received_l266_266126

theorem candy_received (pieces_eaten : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h_eaten : pieces_eaten = 12) (h_piles : piles = 4) (h_pieces_per_pile : pieces_per_pile = 5) :
  pieces_eaten + piles * pieces_per_pile = 32 := 
by
  sorry

end candy_received_l266_266126


namespace total_cupcakes_l266_266459

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (h1 : children = 8) (h2 : cupcakes_per_child = 12) : children * cupcakes_per_child = 96 :=
by
  sorry

end total_cupcakes_l266_266459


namespace probability_of_getting_specific_clothing_combination_l266_266363

def total_articles := 21

def ways_to_choose_4_articles : ℕ := Nat.choose total_articles 4

def ways_to_choose_2_shirts_from_6 : ℕ := Nat.choose 6 2

def ways_to_choose_1_pair_of_shorts_from_7 : ℕ := Nat.choose 7 1

def ways_to_choose_1_pair_of_socks_from_8 : ℕ := Nat.choose 8 1

def favorable_outcomes := 
  ways_to_choose_2_shirts_from_6 * 
  ways_to_choose_1_pair_of_shorts_from_7 * 
  ways_to_choose_1_pair_of_socks_from_8

def probability := (favorable_outcomes : ℚ) / (ways_to_choose_4_articles : ℚ)

theorem probability_of_getting_specific_clothing_combination : 
  probability = 56 / 399 := by
  sorry

end probability_of_getting_specific_clothing_combination_l266_266363


namespace parts_processed_per_hour_before_innovation_l266_266493

variable (x : ℝ) (h : 1500 / x - 1500 / (2.5 * x) = 18)

theorem parts_processed_per_hour_before_innovation : x = 50 :=
by
  sorry

end parts_processed_per_hour_before_innovation_l266_266493


namespace beads_currently_have_l266_266811

-- Definitions of the conditions
def friends : Nat := 6
def beads_per_bracelet : Nat := 8
def additional_beads_needed : Nat := 12

-- Theorem statement
theorem beads_currently_have : (beads_per_bracelet * friends - additional_beads_needed) = 36 := by
  sorry

end beads_currently_have_l266_266811


namespace cars_at_2023_cars_less_than_15_l266_266495

def a_recurrence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 0.9 * a n + 8

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 300

theorem cars_at_2023 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a) :
  a 4 = 240 :=
sorry

def shifted_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - 80 = 0.9 * (a n - 80)

theorem cars_less_than_15 (a : ℕ → ℝ)
  (h_recurrence : a_recurrence a)
  (h_initial : initial_condition a)
  (h_geom_seq : shifted_geom_seq a) :
  ∃ n, n ≥ 12 ∧ a n < 15 :=
sorry

end cars_at_2023_cars_less_than_15_l266_266495


namespace part_one_solution_set_part_two_range_a_l266_266530

noncomputable def f (x a : ℝ) := |x - a| + x

theorem part_one_solution_set (x : ℝ) :
  f x 3 ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7) :=
by sorry

theorem part_two_range_a (a : ℝ) :
  (∀ x, (1 ≤ x ∧ x ≤ 3) → f x a ≥ 2 * a^2) ↔ (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end part_one_solution_set_part_two_range_a_l266_266530


namespace ratio_of_cats_l266_266494

-- Definitions from conditions
def total_animals_anthony := 12
def fraction_cats_anthony := 2 / 3
def extra_dogs_leonel := 7
def total_animals_both := 27

-- Calculate number of cats and dogs Anthony has
def cats_anthony := fraction_cats_anthony * total_animals_anthony
def dogs_anthony := total_animals_anthony - cats_anthony

-- Calculate number of dogs Leonel has
def dogs_leonel := dogs_anthony + extra_dogs_leonel

-- Calculate number of cats Leonel has
def cats_leonel := total_animals_both - (cats_anthony + dogs_anthony + dogs_leonel)

-- Prove the desired ratio
theorem ratio_of_cats : (cats_leonel / cats_anthony) = (1 / 2) := by
  -- Insert proof steps here
  sorry

end ratio_of_cats_l266_266494


namespace exists_valid_sequence_l266_266625

def valid_sequence (s : ℕ → ℝ) : Prop :=
  (∀ i < 18, s i + s (i + 1) + s (i + 2) > 0) ∧  -- 18 to ensure the last 2 sequentials are covered in the 20 values
  (∑ i in Finset.range 20, s i) < 0

theorem exists_valid_sequence :
  ∃ s : ℕ → ℝ, valid_sequence s :=
by
  let s : ℕ → ℝ := λ i, if i % 3 == 2 then 6.5 else -3
  use s
  sorry

end exists_valid_sequence_l266_266625


namespace sandy_took_200_l266_266734

variable (X : ℝ)

/-- Given that Sandy had $140 left after spending 30% of the money she took for shopping,
we want to prove that Sandy took $200 for shopping. -/
theorem sandy_took_200 (h : 0.70 * X = 140) : X = 200 :=
by
  sorry

end sandy_took_200_l266_266734


namespace find_a_of_parallel_lines_l266_266017

theorem find_a_of_parallel_lines :
  ∀ (a : ℝ), (∀ (x y : ℝ), (x + 2 * y - 3 = 0) → (2 * x - a * y + 3 = 0)) → a = -4 :=
by
  intros a h
  have slope_l1 := (-1 : ℝ) / 2
  have slope_l2 := (-2 : ℝ) / a
  have slopes_equal := slope_l1 = slope_l2
  calc a = -4 : sorry

end find_a_of_parallel_lines_l266_266017


namespace cardinality_bound_l266_266573

theorem cardinality_bound {m n : ℕ} (hm : m > 1) (hn : n > 1)
  (S : Finset ℕ) (hS : S.card = n)
  (A : Fin m → Finset ℕ)
  (h : ∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ∃ i, (x ∈ A i ∧ y ∉ (A i)) ∨ (x ∉ (A i) ∧ y ∈ A i)) :
  n ≤ 2^m :=
sorry

end cardinality_bound_l266_266573


namespace solution_positive_iff_k_range_l266_266916

theorem solution_positive_iff_k_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (k / (2 * x - 4) - 1 = x / (x - 2))) ↔ (k > -4 ∧ k ≠ 4) := 
sorry

end solution_positive_iff_k_range_l266_266916


namespace vector_parallel_condition_l266_266998

def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (2 * m, m + 1)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_parallel_condition (m : ℝ) (h_parallel : AB OA OB = (3, 1) ∧ 
    (∀ k : ℝ, 2*m = 3*k ∧ m + 1 = k)) : m = -3 :=
by
  sorry

end vector_parallel_condition_l266_266998


namespace problem_solution_correct_l266_266131

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x = 1

def proposition_q : Prop :=
  {x : ℝ | x^2 - 3 * x + 2 < 0} = {x : ℝ | 1 < x ∧ x < 2}

theorem problem_solution_correct :
  (proposition_p ∧ proposition_q) ∧
  (proposition_p ∧ ¬proposition_q) = false ∧
  (¬proposition_p ∨ proposition_q) ∧
  (¬proposition_p ∨ ¬proposition_q) = false :=
by
  sorry

end problem_solution_correct_l266_266131


namespace compound_interest_calculation_l266_266943

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let A := P * ((1 + r / (n : ℝ)) ^ (n * t))
  A - P

theorem compound_interest_calculation :
  compoundInterest 500 0.05 1 5 = 138.14 := by
  sorry

end compound_interest_calculation_l266_266943


namespace greatest_xy_l266_266397

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l266_266397


namespace mashed_potatoes_vs_tomatoes_l266_266652

theorem mashed_potatoes_vs_tomatoes :
  let m := 144
  let t := 79
  m - t = 65 :=
by 
  repeat { sorry }

end mashed_potatoes_vs_tomatoes_l266_266652


namespace cement_amount_l266_266532

theorem cement_amount
  (originally_had : ℕ)
  (bought : ℕ)
  (total : ℕ)
  (son_brought : ℕ)
  (h1 : originally_had = 98)
  (h2 : bought = 215)
  (h3 : total = 450)
  (h4 : originally_had + bought + son_brought = total) :
  son_brought = 137 :=
by
  sorry

end cement_amount_l266_266532


namespace rectangle_area_l266_266937

def length : ℝ := 15
def width : ℝ := 0.9 * length
def area : ℝ := length * width

theorem rectangle_area : area = 202.5 := by
  sorry

end rectangle_area_l266_266937


namespace jakes_weight_l266_266856

theorem jakes_weight (J S B : ℝ) 
  (h1 : 0.8 * J = 2 * S)
  (h2 : J + S = 168)
  (h3 : B = 1.25 * (J + S))
  (h4 : J + S + B = 221) : 
  J = 120 :=
by
  sorry

end jakes_weight_l266_266856


namespace find_CM_of_trapezoid_l266_266425

noncomputable def trapezoid_CM (AD BC : ℝ) (M : ℝ) : ℝ :=
  if (AD = 12) ∧ (BC = 8) ∧ (M = 2.4)
  then M
  else 0

theorem find_CM_of_trapezoid (trapezoid_ABCD : Type) (AD BC CM : ℝ) (AM_divides_eq_areas : Prop) :
  AD = 12 → BC = 8 → AM_divides_eq_areas → CM = 2.4 := 
by
  intros h1 h2 h3
  have : AD = 12 := h1
  have : BC = 8 := h2
  have : CM = 2.4 := sorry
  exact this

end find_CM_of_trapezoid_l266_266425


namespace sin_function_value_l266_266349

noncomputable def f (x : ℝ) : ℝ := sin(2 * x - 5 * π / 6)

theorem sin_function_value :
  f (-5 * π / 12) = sqrt 3 / 2 := by
  sorry

end sin_function_value_l266_266349


namespace robert_birth_year_l266_266176

theorem robert_birth_year (n : ℕ) (h1 : (n + 1)^2 - n^2 = 89) : n = 44 ∧ n^2 = 1936 :=
by {
  sorry
}

end robert_birth_year_l266_266176


namespace inequality_solution_set_l266_266123

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1 / 3 ≤ x ∧ x < 1 / 2 :=
by
  sorry

end inequality_solution_set_l266_266123


namespace recurrence_relation_l266_266416

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l266_266416


namespace gcd_18_30_l266_266304

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266304


namespace gcd_18_30_l266_266260

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266260


namespace terminal_side_of_angle_y_eq_neg_one_l266_266842
/-
Given that the terminal side of angle θ lies on the line y = -x,
prove that y = -1 where y = sin θ / |sin θ| + |cos θ| / cos θ + tan θ / |tan θ|.
-/


noncomputable def y (θ : ℝ) : ℝ :=
  (Real.sin θ / |Real.sin θ|) + (|Real.cos θ| / Real.cos θ) + (Real.tan θ / |Real.tan θ|)

theorem terminal_side_of_angle_y_eq_neg_one (θ : ℝ) (k : ℤ) (h : θ = k * Real.pi - (Real.pi / 4)) :
  y θ = -1 :=
by
  sorry

end terminal_side_of_angle_y_eq_neg_one_l266_266842


namespace abc_relationship_l266_266672

noncomputable def a : ℝ := Real.log 5 - Real.log 3
noncomputable def b : ℝ := (2/5) * Real.exp (2/3)
noncomputable def c : ℝ := 2/3

theorem abc_relationship : b > c ∧ c > a :=
by
  sorry

end abc_relationship_l266_266672


namespace vote_proportion_inequality_l266_266862

theorem vote_proportion_inequality
  (a b k : ℕ)
  (hb_odd : b % 2 = 1)
  (hb_min : 3 ≤ b)
  (vote_same : ∀ (i j : ℕ) (hi hj : i ≠ j) (votes : ℕ → ℕ), ∃ (k_max : ℕ), ∀ (cont : ℕ), votes cont ≤ k_max) :
  (k : ℚ) / a ≥ (b - 1) / (2 * b) := sorry

end vote_proportion_inequality_l266_266862


namespace determine_value_of_a_l266_266110

theorem determine_value_of_a :
  ∃ b, (∀ x : ℝ, (4 * x^2 + 12 * x + (b^2)) = (2 * x + b)^2) :=
sorry

end determine_value_of_a_l266_266110


namespace probability_recurrence_relation_l266_266398

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l266_266398


namespace maximum_pizzas_baked_on_Friday_l266_266001

def george_bakes := 
  let total_pizzas : ℕ := 1000
  let monday_pizzas := total_pizzas * 7 / 10
  let tuesday_pizzas := if monday_pizzas * 4 / 5 < monday_pizzas * 9 / 10 
                        then monday_pizzas * 4 / 5 
                        else monday_pizzas * 9 / 10
  let wednesday_pizzas := if tuesday_pizzas * 4 / 5 < tuesday_pizzas * 9 / 10 
                          then tuesday_pizzas * 4 / 5 
                          else tuesday_pizzas * 9 / 10
  let thursday_pizzas := if wednesday_pizzas * 4 / 5 < wednesday_pizzas * 9 / 10 
                         then wednesday_pizzas * 4 / 5 
                         else wednesday_pizzas * 9 / 10
  let friday_pizzas := if thursday_pizzas * 4 / 5 < thursday_pizzas * 9 / 10 
                       then thursday_pizzas * 4 / 5 
                       else thursday_pizzas * 9 / 10
  friday_pizzas

theorem maximum_pizzas_baked_on_Friday : george_bakes = 2 := by
  sorry

end maximum_pizzas_baked_on_Friday_l266_266001


namespace gcd_of_18_and_30_l266_266290

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266290


namespace undefined_expr_iff_l266_266830

theorem undefined_expr_iff (a : ℝ) : (∃ x, x = (a^2 - 9) ∧ x = 0) ↔ (a = -3 ∨ a = 3) :=
by
  sorry

end undefined_expr_iff_l266_266830


namespace area_of_square_not_covered_by_circles_l266_266977

theorem area_of_square_not_covered_by_circles :
  let side : ℝ := 10
  let radius : ℝ := 5
  (side^2 - 4 * (π * radius^2) + 4 * (π * (radius^2) / 2)) = (100 - 50 * π) := 
sorry

end area_of_square_not_covered_by_circles_l266_266977


namespace sum_infinite_geometric_series_l266_266962

theorem sum_infinite_geometric_series (a r : ℝ) (h1 : a = 1) (h2 : r = 1/3) (h3 : |r| < 1) :
  ∑' n : ℕ, a * r^n = 3 / 2 :=
by
  have series_sum : ∑' n : ℕ, a * r^n = a / (1 - r) := by sorry
  rw [h1, h2] at series_sum
  rw [series_sum]
  norm_num

end sum_infinite_geometric_series_l266_266962


namespace greatest_xy_value_l266_266390

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l266_266390


namespace polynomial_has_zero_of_given_form_l266_266642

open Complex Polynomial

theorem polynomial_has_zero_of_given_form :
  ∃ (P : Polynomial ℂ), 
    P.degree = 4 ∧ 
    P.leadingCoeff = 1 ∧ 
    (∃ r s : ℤ, P.has_root r ∧ P.has_root s ∧ 
    ∃ α β : ℤ, P = Polynomial.real_coeff_of_degree 4 1 (x - r) (x - s) (x^2 + α * x + β) ∧ 
    (Complex.ofReal (2 : ℚ) + Complex.i * Complex.sqrt (8 : ℚ)) / 3) = 0 :=
begin
  sorry
end

end polynomial_has_zero_of_given_form_l266_266642


namespace sara_pumpkins_l266_266890

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l266_266890


namespace product_of_8_dice_is_divisible_by_8_l266_266809

open ProbabilityMeasure
open Classical

-- Define a standard 6-sided die
inductive Die
| face1 | face2 | face3 | face4 | face5 | face6

namespace Die

instance : Inhabited Die := ⟨Die.face1⟩

def eq_classes : Finset (Fin 6) := { ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩, ⟨6, by decide⟩ }

def Roll : Die → ℕ
| Die.face1 => 1
| Die.face2 => 2
| Die.face3 => 3
| Die.face4 => 4
| Die.face5 => 5
| Die.face6 => 6

def probability_space : ProbabilityMeasure (Finset Die) := sorry -- Construed probability space for the 8 rolls

-- Event indicating the product of the rolls
def event_production_divisible_by_8 (outcome: Fin 8 → Die) : Prop :=
  (List.prod (List.ofFn (λ i => (Roll (outcome i : Die)))) % 8 = 0)

-- Define measure for this event
def production_divisible_by_8_measure : ℚ := 
  probability_space.measure { outcome | event_production_divisible_by_8 outcome }

-- Statement of the main theorem
theorem product_of_8_dice_is_divisible_by_8 : production_divisible_by_8_measure = 35 / 36 := 
  sorry -- Proof is omitted

end product_of_8_dice_is_divisible_by_8_l266_266809


namespace maximum_xy_value_l266_266373

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l266_266373


namespace find_fx_value_l266_266354

noncomputable def f (x : Real) : Real :=
  sin (2 * x - 5 * Real.pi / 6)

theorem find_fx_value :
  (∀ x, f x = sin (2 * x - 5 * Real.pi / 6)) ∧ 
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) ∧ 
  (∀ x1 x2, (Real.pi / 6 < x1 ∧ x1 < 2 * Real.pi / 3 ∧ x1 < x2 ∧ x2 < 2 * Real.pi / 3) -> f x1 < f x2) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 := by
  sorry

end find_fx_value_l266_266354


namespace total_area_of_field_l266_266214

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end total_area_of_field_l266_266214


namespace intersection_points_count_l266_266606

theorem intersection_points_count : 
  ∃ n : ℕ, n = 2 ∧
  (∀ x ∈ (Set.Icc 0 (2 * Real.pi)), (1 + Real.sin x = 3 / 2) → n = 2) :=
sorry

end intersection_points_count_l266_266606


namespace orchid_bushes_total_l266_266457

def current_orchid_bushes : ℕ := 22
def new_orchid_bushes : ℕ := 13

theorem orchid_bushes_total : current_orchid_bushes + new_orchid_bushes = 35 := 
by 
  sorry

end orchid_bushes_total_l266_266457


namespace find_line_eq_l266_266983

noncomputable def line_perpendicular (p : ℝ × ℝ) (a b c: ℝ) : Prop :=
  ∃ (m: ℝ) (k: ℝ), k ≠ 0 ∧ (b * m = -a) ∧ p = (m, (c - a * m) / b) ∧
  (∀ x y : ℝ, y = m * x + ((c - a * m) / b) ↔ b * y = -a * x - c)

theorem find_line_eq (p : ℝ × ℝ) (a b c : ℝ) (p_eq : p = (-3, 0)) (perpendicular_eq : a = 2 ∧ b = -1 ∧ c = 3) :
  ∃ (m k : ℝ), (k ≠ 0 ∧ (-1 * (b / a)) = m ∧ line_perpendicular p a b c) ∧ (b * m = -a) ∧ ((k = (-a * m) / b) ∧ (b * k * 0 - (-a * 3)) = c) := sorry

end find_line_eq_l266_266983


namespace ab_value_l266_266008

theorem ab_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 30) (h4 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 :=
sorry

end ab_value_l266_266008


namespace dot_product_AB_AC_dot_product_AB_BC_l266_266032

-- The definition of equilateral triangle with side length 6
structure EquilateralTriangle (A B C : Type*) :=
  (side_len : ℝ)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_CAB : ℝ)
  (AB_len : ℝ)
  (AC_len : ℝ)
  (BC_len : ℝ)
  (AB_eq_AC : AB_len = AC_len)
  (AB_eq_BC : AB_len = BC_len)
  (cos_ABC : ℝ)
  (cos_BCA : ℝ)
  (cos_CAB : ℝ)

-- Given an equilateral triangle with side length 6 where the angles are defined,
-- we can define the specific triangle
noncomputable def triangleABC (A B C : Type*) : EquilateralTriangle A B C :=
{ side_len := 6,
  angle_ABC := 120,
  angle_BCA := 60,
  angle_CAB := 60,
  AB_len := 6,
  AC_len := 6,
  BC_len := 6,
  AB_eq_AC := rfl,
  AB_eq_BC := rfl,
  cos_ABC := -0.5,
  cos_BCA := 0.5,
  cos_CAB := 0.5 }

-- Prove the dot product of vectors AB and AC
theorem dot_product_AB_AC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.AC_len * T.cos_BCA) = 18 :=
by sorry

-- Prove the dot product of vectors AB and BC
theorem dot_product_AB_BC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.BC_len * T.cos_ABC) = -18 :=
by sorry

end dot_product_AB_AC_dot_product_AB_BC_l266_266032


namespace sum_of_edges_l266_266645

theorem sum_of_edges (a r : ℝ) 
  (h_volume : (a^3 = 512))
  (h_surface_area : (2 * (a^2 / r + a^2 + a^2 * r) = 384))
  (h_geometric_progression : true) :
  (4 * ((a / r) + a + (a * r)) = 96) :=
by
  -- It is only necessary to provide the theorem statement
  sorry

end sum_of_edges_l266_266645


namespace proportion_of_boys_geq_35_percent_l266_266191

variables (a b c d n : ℕ)

axiom room_constraint : 2 * (b + d) ≥ n
axiom girl_constraint : 3 * a ≥ 8 * b

theorem proportion_of_boys_geq_35_percent : (3 * c + 4 * d : ℚ) / (3 * a + 4 * b + 3 * c + 4 * d : ℚ) ≥ 0.35 :=
by 
  sorry

end proportion_of_boys_geq_35_percent_l266_266191


namespace f_at_2_l266_266979

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x ^ 2017 + a * x ^ 3 - b / x - 8

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by sorry

end f_at_2_l266_266979


namespace probability_one_letter_from_each_l266_266224

theorem probability_one_letter_from_each
  (total_cards : ℕ)
  (adam_cards : ℕ)
  (brian_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : adam_cards = 4)
  (h3 : brian_cards = 6)
  : (4/12 * 6/11) + (6/12 * 4/11) = 4/11 := by
  sorry

end probability_one_letter_from_each_l266_266224


namespace probability_two_more_sons_or_daughters_l266_266884

theorem probability_two_more_sons_or_daughters (n : ℕ) (p : ℚ) (k : ℕ) :
  n = 8 → p = 1 / 2 → k = 2 →
  (prob_at_least_two_more_sons_or_daughters n p k) = 37 / 128 :=
by
  sorry

-- Definitions used
def prob_at_least_two_more_sons_or_daughters (n : ℕ) (p : ℚ) (k : ℕ) : rat :=
  let total_combinations := (bit0 256 : rat)
  let equal_sons_daughters := nat.choose n (n / 2)
  let one_more_or_less_son_daughter := nat.choose n (n / 2 + 1)
  let non_favorable_cases := equal_sons_daughters + 2 * one_more_or_less_son_daughter
  (total_combinations - non_favorable_cases) / total_combinations

end probability_two_more_sons_or_daughters_l266_266884


namespace heating_time_correct_l266_266558

structure HeatingProblem where
  initial_temp : ℕ
  final_temp : ℕ
  heating_rate : ℕ

def time_to_heat (hp : HeatingProblem) : ℕ :=
  (hp.final_temp - hp.initial_temp) / hp.heating_rate

theorem heating_time_correct (hp : HeatingProblem) (h1 : hp.initial_temp = 20) (h2 : hp.final_temp = 100) (h3 : hp.heating_rate = 5) :
  time_to_heat hp = 16 :=
by
  sorry

end heating_time_correct_l266_266558


namespace digits_solution_exists_l266_266469

theorem digits_solution_exists (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : a = (b * (10 * b)) / (10 - b)) : a = 5 ∧ b = 2 :=
by
  sorry

end digits_solution_exists_l266_266469


namespace right_triangle_ineq_l266_266732

-- Definitions based on conditions in (a)
variables {a b c m f : ℝ}
variable (h_a : a ≥ 0)
variable (h_b : b ≥ 0)
variable (h_c : c > 0)
variable (h_a_b : a ≤ b)
variable (h_triangle : c = Real.sqrt (a^2 + b^2))
variable (h_m : m = a * b / c)
variable (h_f : f = (Real.sqrt 2 * a * b) / (a + b))

-- Proof goal based on the problem in (c)
theorem right_triangle_ineq : m + f ≤ c :=
sorry

end right_triangle_ineq_l266_266732


namespace intersection_M_N_l266_266137

theorem intersection_M_N :
  let M := {x | x^2 < 36}
  let N := {2, 4, 6, 8}
  M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l266_266137


namespace batsman_average_after_11th_inning_l266_266083

variable (x : ℝ) -- The average before the 11th inning
variable (new_average : ℝ) -- The average after the 11th inning
variable (total_runs : ℝ) -- Total runs scored after 11 innings

-- Given conditions
def condition1 := total_runs = 11 * (x + 5)
def condition2 := total_runs = 10 * x + 110

theorem batsman_average_after_11th_inning : 
  ∀ (x : ℝ), 
    (x = 55) → (x + 5 = 60) :=
by
  intros
  sorry

end batsman_average_after_11th_inning_l266_266083


namespace no_x_satisfies_inequality_l266_266849

def f (x : ℝ) : ℝ := x^2 + x

theorem no_x_satisfies_inequality : ¬ ∃ x : ℝ, f (x - 2) + f x < 0 :=
by 
  unfold f 
  sorry

end no_x_satisfies_inequality_l266_266849


namespace factorial_mod_11_l266_266668

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_11 : (factorial 13) % 11 = 0 := by
  sorry

end factorial_mod_11_l266_266668


namespace find_value_l266_266351

-- Define the function $f(x)$
def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

-- Define the problem statement in Lean 4
theorem find_value :
  (∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 0 ≤ 2 * x - 5 * Real.pi / 6 ∧ 2 * x - 5 * Real.pi / 6 ≤ Real.pi) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by
  -- A proper rigorous Lean proof should go here
  sorry

end find_value_l266_266351


namespace original_pumpkins_count_l266_266892

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l266_266892


namespace roll_four_fair_dice_l266_266775
noncomputable def roll_four_fair_dice_prob : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 6
  let prob_all_same : ℚ := favorable_outcomes / total_outcomes
  let prob_not_all_same : ℚ := 1 - prob_all_same
  prob_not_all_same

theorem roll_four_fair_dice :
  roll_four_fair_dice_prob = 215 / 216 :=
by
  sorry

end roll_four_fair_dice_l266_266775


namespace min_digs_is_three_l266_266467

/-- Represents an 8x8 board --/
structure Board :=
(dim : ℕ := 8)

/-- Each cell either contains the treasure or a plaque indicating minimum steps --/
structure Cell :=
(content : CellContent)

/-- Possible content of a cell --/
inductive CellContent
| Treasure
| Plaque (steps : ℕ)

/-- Function that returns the minimum number of cells to dig to find the treasure --/
def min_digs_to_find_treasure (board : Board) : ℕ := 3

/-- The main theorem stating the minimum number of cells needed to find the treasure on an 8x8 board --/
theorem min_digs_is_three : 
  ∀ board : Board, min_digs_to_find_treasure board = 3 := 
by 
  intro board
  sorry

end min_digs_is_three_l266_266467


namespace reach_any_composite_from_4_l266_266709

/-- 
Prove that starting from the number \( 4 \), it is possible to reach any given composite number 
through repeatedly adding one of its divisors, different from itself and one. 
-/
theorem reach_any_composite_from_4:
  ∀ n : ℕ, Prime (n) → n ≥ 4 → (∃ k d : ℕ, d ∣ k ∧ k = k + d ∧ k = n) := 
by 
  sorry


end reach_any_composite_from_4_l266_266709


namespace chocolate_bars_l266_266941

theorem chocolate_bars (num_small_boxes : ℕ) (num_bars_per_box : ℕ) (total_bars : ℕ) (h1 : num_small_boxes = 20) (h2 : num_bars_per_box = 32) (h3 : total_bars = num_small_boxes * num_bars_per_box) :
  total_bars = 640 :=
by
  sorry

end chocolate_bars_l266_266941


namespace two_digit_numbers_l266_266880

theorem two_digit_numbers :
  ∃ (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ x < y ∧ 2000 + x + y = x * y := 
sorry

end two_digit_numbers_l266_266880


namespace ellipse_focus_coordinates_l266_266758

theorem ellipse_focus_coordinates (a b c : ℝ) (x1 y1 x2 y2 : ℝ) 
  (major_axis_length : 2 * a = 20) 
  (focal_relationship : c^2 = a^2 - b^2)
  (focus1_location : x1 = 3 ∧ y1 = 4) 
  (focus_c_calculation : c = Real.sqrt (x1^2 + y1^2)) :
  (x2 = -3 ∧ y2 = -4) := by
  sorry

end ellipse_focus_coordinates_l266_266758


namespace problem_statement_l266_266684

noncomputable def x : ℕ := 4
noncomputable def y : ℤ := 3  -- alternatively, we could define y as -3 and the equality would still hold

theorem problem_statement : x^2 + y^2 + x + 2023 = 2052 := by
  sorry  -- Proof goes here

end problem_statement_l266_266684


namespace arun_weight_lower_limit_l266_266422

variable {W B : ℝ}

theorem arun_weight_lower_limit
  (h1 : 64 < W ∧ W < 72)
  (h2 : B < W ∧ W < 70)
  (h3 : W ≤ 67)
  (h4 : (64 + 67) / 2 = 66) :
  64 < B :=
by sorry

end arun_weight_lower_limit_l266_266422


namespace diamonds_in_F20_l266_266064

def F (n : ℕ) : ℕ :=
  -- Define recursively the number of diamonds in figure F_n
  match n with
  | 1 => 1
  | 2 => 9
  | n + 1 => F n + 4 * (n + 1)

theorem diamonds_in_F20 : F 20 = 761 :=
by sorry

end diamonds_in_F20_l266_266064


namespace total_earthworms_in_box_l266_266728

-- Definitions of the conditions
def applesPaidByOkeydokey := 5
def applesPaidByArtichokey := 7
def earthwormsReceivedByOkeydokey := 25
def ratio := earthwormsReceivedByOkeydokey / applesPaidByOkeydokey -- which should be 5

-- Theorem statement proving the total number of earthworms in the box
theorem total_earthworms_in_box :
  (applesPaidByOkeydokey + applesPaidByArtichokey) * ratio = 60 :=
by
  sorry

end total_earthworms_in_box_l266_266728


namespace cubic_roots_sum_of_cubes_l266_266503

def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem cubic_roots_sum_of_cubes :
  let α := cube_root 17
  let β := cube_root 73
  let γ := cube_root 137
  ∀ (a b c : ℝ),
    (a - α) * (a - β) * (a - γ) = 1/2 ∧
    (b - α) * (b - β) * (b - γ) = 1/2 ∧
    (c - α) * (c - β) * (c - γ) = 1/2 →
    a^3 + b^3 + c^3 = 228.5 :=
by {
  sorry
}

end cubic_roots_sum_of_cubes_l266_266503


namespace sum_of_first_and_fourth_l266_266767

theorem sum_of_first_and_fourth (x : ℤ) (h : x + (x + 6) = 156) : (x + 2) = 77 :=
by {
  -- This block represents the assumptions and goal as expressed above,
  -- but the proof steps are omitted.
  sorry
}

end sum_of_first_and_fourth_l266_266767


namespace find_f_of_3_l266_266523

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x + 2) = x) : f 3 = 1 := 
sorry

end find_f_of_3_l266_266523


namespace negate_exactly_one_even_l266_266727

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even :
  ¬(is_even a ∧ is_odd b ∧ is_odd c ∨ is_odd a ∧ is_even b ∧ is_odd c ∨ is_odd a ∧ is_odd b ∧ is_even c) ↔
  (is_even a ∧ is_even b ∨ is_even a ∧ is_even c ∨ is_even b ∧ is_even c ∨ is_odd a ∧ is_odd b ∧ is_odd c) := sorry

end negate_exactly_one_even_l266_266727


namespace largest_difference_l266_266167

noncomputable def A := 3 * (1003 ^ 1004)
noncomputable def B := 1003 ^ 1004
noncomputable def C := 1002 * (1003 ^ 1003)
noncomputable def D := 3 * (1003 ^ 1003)
noncomputable def E := 1003 ^ 1003
noncomputable def F := 1003 ^ 1002

theorem largest_difference : 
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l266_266167


namespace student_solved_18_correctly_l266_266954

theorem student_solved_18_correctly (total_problems : ℕ) (correct : ℕ) (wrong : ℕ) 
  (h1 : total_problems = 54) (h2 : wrong = 2 * correct) (h3 : total_problems = correct + wrong) :
  correct = 18 :=
by
  sorry

end student_solved_18_correctly_l266_266954


namespace smallest_sum_of_squares_l266_266744

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l266_266744


namespace algebra_expression_value_l266_266130

theorem algebra_expression_value (x y : ℝ) (h1 : x * y = 3) (h2 : x - y = -2) : x^2 * y - x * y^2 = -6 := 
by
  sorry

end algebra_expression_value_l266_266130


namespace talia_father_age_l266_266867

theorem talia_father_age 
  (t tf tm ta : ℕ) 
  (h1 : t + 7 = 20)
  (h2 : tm = 3 * t)
  (h3 : tf + 3 = tm)
  (h4 : ta = (tm - t) / 2)
  (h5 : ta + 2 = tf + 5) : 
  tf = 36 :=
by
  sorry

end talia_father_age_l266_266867


namespace more_girls_than_boys_l266_266912

theorem more_girls_than_boys (num_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ) (total_students : ℕ) (total_students_eq : num_students = 42) (ratio_eq : boys_ratio = 3 ∧ girls_ratio = 4) : (4 * 6) - (3 * 6) = 6 := by
  sorry

end more_girls_than_boys_l266_266912


namespace solve_equation1_solve_equation2_l266_266182

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end solve_equation1_solve_equation2_l266_266182


namespace gcd_of_18_and_30_l266_266250

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266250


namespace gcd_18_30_is_6_l266_266264

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266264


namespace smallest_sum_of_squares_l266_266750

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l266_266750


namespace set_points_quadrants_l266_266189

theorem set_points_quadrants (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → 
  (y > 0 ∧ x > 0) ∨ (y > 0 ∧ x < 0) :=
by 
  sorry

end set_points_quadrants_l266_266189


namespace new_profit_percentage_l266_266226

theorem new_profit_percentage (P : ℝ) (h1 : 1.10 * P = 990) (h2 : 0.90 * P * (1 + 0.30) = 1053) : 0.30 = 0.30 :=
by sorry

end new_profit_percentage_l266_266226


namespace unfair_draw_l266_266150

-- Define the types for suits and ranks
inductive Suit
| hearts | diamonds | clubs | spades

inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

-- Define a card as a combination of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Function to determine if a card is higher in rank
def higher_rank (r1 r2 : Rank) : Prop :=
  match r1, r2 with
  | Rank.Six, _ | Rank.Seven, Rank.Six | Rank.Eight, (Rank.Six | Rank.Seven) | Rank.Nine, (Rank.Six | Rank.Seven | Rank.Eight)
  | Rank.Ten, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine) | Rank.Jack, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten)
  | Rank.Queen, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack)
  | Rank.King, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen)
  | Rank.Ace, (Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King)
    => true
  | _, _ => false

-- Problem statement to prove unfairness of the draw
theorem unfair_draw :
  ∀ (vCard mCard : Card), (∃ (deck : List Card), 
  deck.length = 36 ∧ ∀ c, c ∈ deck →
  match c.rank with 
  | Rank.Six | Rank.Seven | Rank.Eight | Rank.Nine | Rank.Ten | Rank.Jack | Rank.Queen | Rank.King | Rank.Ace => true 
  | _ => false) →
  (∃ (vCard mCard : Card), 
    vCard ∈ deck ∧ mCard ∈ (deck.erase vCard) ∧ higher_rank vCard.rank mCard.rank) →
  ¬fair :=
sorry

end unfair_draw_l266_266150


namespace recurrence_relation_l266_266411

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l266_266411


namespace robin_candy_consumption_l266_266125

theorem robin_candy_consumption (x : ℕ) : 23 - x + 21 = 37 → x = 7 :=
by
  intros h
  sorry

end robin_candy_consumption_l266_266125


namespace gcd_18_30_l266_266275

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266275


namespace even_function_expression_l266_266934

theorem even_function_expression (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≥ 0 → f x = x^2 - 3 * x + 4)
  (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = if x < 0 then x^2 + 3 * x + 4 else x^2 - 3 * x + 4 :=
by {
  sorry
}

end even_function_expression_l266_266934


namespace sum_of_h_values_l266_266420

variable (f h : ℤ → ℤ)

-- Function definition for f and h
def f_def : ∀ x, 0 ≤ x → f x = f (x + 2) := sorry
def h_def : ∀ x, x < 0 → h x = f x := sorry

-- Symmetry condition for f being odd
def f_odd : ∀ x, f (-x) = -f x := sorry

-- Given value
def f_at_5 : f 5 = 1 := sorry

-- The proof statement we need:
theorem sum_of_h_values :
  h (-2022) + h (-2023) + h (-2024) = -1 :=
sorry

end sum_of_h_values_l266_266420


namespace area_of_circumscribed_circle_eq_48pi_l266_266089

noncomputable def side_length := 12
noncomputable def radius := (2/3) * (side_length / 2) * (Real.sqrt 3)
noncomputable def area := Real.pi * radius^2

theorem area_of_circumscribed_circle_eq_48pi :
  area = 48 * Real.pi :=
by
  sorry

end area_of_circumscribed_circle_eq_48pi_l266_266089


namespace number_of_marked_points_l266_266593

theorem number_of_marked_points (S S' : ℤ) (n : ℤ) 
  (h1 : S = 25) 
  (h2 : S' = S - 5 * n) 
  (h3 : S' = -35) : 
  n = 12 := 
  sorry

end number_of_marked_points_l266_266593


namespace gcd_18_30_l266_266282

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266282


namespace geometric_sequence_term_formula_l266_266343

theorem geometric_sequence_term_formula (a n : ℕ) (a_seq : ℕ → ℕ)
  (h1 : a_seq 0 = a - 1) (h2 : a_seq 1 = a + 1) (h3 : a_seq 2 = a + 4)
  (geometric_seq : ∀ n, a_seq (n + 1) = a_seq n * ((a_seq 1) / (a_seq 0))) :
  a = 5 ∧ a_seq n = 4 * (3 / 2) ^ (n - 1) :=
by
  sorry

end geometric_sequence_term_formula_l266_266343


namespace ten_crates_probability_l266_266904

theorem ten_crates_probability (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  let num_crates := 10
  let crate_dimensions := [3, 4, 6]
  let target_height := 41

  -- Definition of the generating function coefficients and constraints will be complex,
  -- so stating the specific problem directly.
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ m = 190 ∧ n = 2187 →
  let probability := (m : ℚ) / (n : ℚ)
  probability = (190 : ℚ) / 2187 := 
by
  sorry

end ten_crates_probability_l266_266904


namespace maximum_x1_x2_x3_l266_266043

theorem maximum_x1_x2_x3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
  x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
  x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
  x1 + x2 + x3 ≤ 61 := 
by sorry

end maximum_x1_x2_x3_l266_266043


namespace smallest_sum_of_squares_l266_266747

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l266_266747


namespace intersection_sets_l266_266015

noncomputable def set1 (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0
noncomputable def set2 (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem intersection_sets :
  { x : ℝ | set1 x } ∩ { x : ℝ | set2 x } = { x | (-1 : ℝ) < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_l266_266015


namespace sufficient_but_not_necessary_condition_l266_266607

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≤ -2) ↔ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a)) ∧ ¬ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a) → (a ≤ -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l266_266607


namespace exists_non_intersecting_segments_l266_266564

open Set

variable {S : Set (Point)} (N : ℕ) (c : Point → Point → Color) 
variable [finite S] [N ≥ 3]

-- Assuming no three points are collinear as a separate definition
def no_three_collinear (S : Set (Point)) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ S → p2 ∈ S → p3 ∈ S → 
  ¬collinear p1 p2 p3

-- Defining segments
def segment (p1 p2 : Point) : Type := {p : Point | p = p1 ∨ p = p2}

-- Ensuring each segment is colored
def colored_segments (S : Set (Point)) (c : Point → Point → Color) : Prop :=
  ∀ (p1 p2 : Point), p1 ∈ S → p2 ∈ S → p1 ≠ p2 → 
  (c p1 p2 = Color.red ∨ c p1 p2 = Color.blue)

theorem exists_non_intersecting_segments : 
  ∀ (S : Set (Point)) (N : ℕ) (c : Point → Point → Color), 
  finite S → N ≥ 3 → no_three_collinear S → 
  colored_segments S c → 
  ∃ (T : Set (segment)), 
    ∀ s1 s2 ∈ T, s1 ≠ s2 → (s1 ∩ s2 = ∅) ∧ (∃! e, e ∈ T) ∧
    no_polygon_subset T
by
  sorry

end exists_non_intersecting_segments_l266_266564


namespace simplify_fraction_l266_266781

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l266_266781


namespace custom_operation_correct_l266_266763

def custom_operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem custom_operation_correct : custom_operation 6 3 = 27 :=
by {
  sorry
}

end custom_operation_correct_l266_266763


namespace three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l266_266778

theorem three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3 :
  ∃ x1 x2 x3 : ℕ, ((x1 = 414 ∧ x2 = 444 ∧ x3 = 474) ∧ 
  (∀ n, (100 * 4 + 10 * n + 4 = x1 ∨ 100 * 4 + 10 * n + 4 = x2 ∨ 100 * 4 + 10 * n + 4 = x3) 
  → (100 * 4 + 10 * n + 4) % 3 = 0)) :=
by
  sorry

end three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l266_266778


namespace parking_lot_perimeter_l266_266096

theorem parking_lot_perimeter (x y: ℝ) 
  (h1: x = (2 / 3) * y)
  (h2: x^2 + y^2 = 400)
  (h3: x * y = 120) :
  2 * (x + y) = 20 * Real.sqrt 5 :=
by
  sorry

end parking_lot_perimeter_l266_266096


namespace candy_bar_cost_l266_266874

-- Define the conditions
def cost_gum_over_candy_bar (C G : ℝ) : Prop :=
  G = (1/2) * C

def total_cost (C G : ℝ) : Prop :=
  2 * G + 3 * C = 6

-- Define the proof problem
theorem candy_bar_cost (C G : ℝ) (h1 : cost_gum_over_candy_bar C G) (h2 : total_cost C G) : C = 1.5 :=
by
  sorry

end candy_bar_cost_l266_266874


namespace smallest_n_l266_266328

-- Define the conditions as predicates
def condition1 (n : ℕ) : Prop := (n + 2018) % 2020 = 0
def condition2 (n : ℕ) : Prop := (n + 2020) % 2018 = 0

-- The main theorem statement using these conditions
theorem smallest_n (n : ℕ) : 
  (∃ n, condition1 n ∧ condition2 n ∧ (∀ m, condition1 m ∧ condition2 m → n ≤ m)) ↔ n = 2030102 := 
by 
    sorry

end smallest_n_l266_266328


namespace integer_solutions_of_inequality_l266_266760

theorem integer_solutions_of_inequality :
  {x : ℤ | 3 ≤ 5 - 2 * x ∧ 5 - 2 * x ≤ 9} = {-2, -1, 0, 1} :=
by
  sorry

end integer_solutions_of_inequality_l266_266760


namespace field_area_l266_266218

theorem field_area (L W : ℝ) (h1: L = 20) (h2 : 2 * W + L = 41) : L * W = 210 :=
by
  sorry

end field_area_l266_266218


namespace maximize_wz_xy_zx_l266_266169

-- Variables definition
variables {w x y z : ℝ}

-- Main statement
theorem maximize_wz_xy_zx (h_sum : w + x + y + z = 200) (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (w * z + x * y + z * x) ≤ 7500 :=
sorry

end maximize_wz_xy_zx_l266_266169


namespace gray_region_area_l266_266870

-- Definitions based on given conditions
def radius_inner (r : ℝ) := r
def radius_outer (r : ℝ) := r + 3

-- Statement to prove: the area of the gray region
theorem gray_region_area (r : ℝ) : 
  (π * (radius_outer r)^2 - π * (radius_inner r)^2) = 6 * π * r + 9 * π := by
  sorry

end gray_region_area_l266_266870


namespace largest_multiple_of_7_whose_negation_greater_than_neg80_l266_266194

theorem largest_multiple_of_7_whose_negation_greater_than_neg80 : ∃ (n : ℤ), n = 77 ∧ (∃ (k : ℤ), n = k * 7) ∧ (-n > -80) :=
by
  sorry

end largest_multiple_of_7_whose_negation_greater_than_neg80_l266_266194


namespace integral_cos_neg_one_l266_266823

theorem integral_cos_neg_one: 
  ∫ x in (Set.Icc (Real.pi / 2) Real.pi), Real.cos x = -1 :=
by
  sorry

end integral_cos_neg_one_l266_266823


namespace henry_age_l266_266608

theorem henry_age (H J : ℕ) (h1 : H + J = 43) (h2 : H - 5 = 2 * (J - 5)) : H = 27 :=
by
  -- This is where we would prove the theorem based on the given conditions
  sorry

end henry_age_l266_266608


namespace gcd_18_30_l266_266258

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266258


namespace total_area_of_figure_l266_266211

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

def side_length_of_square (d : ℝ) : ℝ := d

def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def total_area (d : ℝ) : ℝ := area_of_square d + area_of_circle (radius_of_circle d)

theorem total_area_of_figure (d : ℝ) (h : d = 6) : total_area d = 36 + 9 * Real.pi :=
by
  -- skipping proof with sorry
  sorry

end total_area_of_figure_l266_266211


namespace parallelogram_area_l266_266044

-- Defining the vectors u and z
def u : ℝ × ℝ := (4, -1)
def z : ℝ × ℝ := (9, -3)

-- Computing the area of parallelogram formed by vectors u and z
def area_parallelogram (u z : ℝ × ℝ) : ℝ :=
  abs (u.1 * (z.2 + u.2) - u.2 * (z.1 + u.1))

-- Lean statement asserting that the area of the parallelogram is 3
theorem parallelogram_area : area_parallelogram u z = 3 := by
  sorry

end parallelogram_area_l266_266044


namespace minimum_guests_at_banquet_l266_266085

theorem minimum_guests_at_banquet (total_food : ℝ) (max_food_per_guest : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411) (h2 : max_food_per_guest = 2.5) : min_guests = 165 :=
by
  -- Proof omitted
  sorry

end minimum_guests_at_banquet_l266_266085


namespace value_divided_by_is_three_l266_266800

theorem value_divided_by_is_three (x : ℝ) (h : 72 / x = 24) : x = 3 := 
by
  sorry

end value_divided_by_is_three_l266_266800


namespace min_value_of_quadratic_fun_min_value_is_reached_l266_266679

theorem min_value_of_quadratic_fun (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1) :
  (3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 ≥ (15 / 782)) :=
sorry

theorem min_value_is_reached (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1)
  (h2 : 3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 = (15 / 782)) :
  true :=
sorry

end min_value_of_quadratic_fun_min_value_is_reached_l266_266679


namespace product_eval_at_3_l266_266824

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l266_266824


namespace probability_three_defective_phones_l266_266647

theorem probability_three_defective_phones :
  let total_smartphones := 380
  let defective_smartphones := 125
  let P_def_1 := (defective_smartphones : ℝ) / total_smartphones
  let P_def_2 := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let P_def_3 := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let P_all_three_def := P_def_1 * P_def_2 * P_def_3
  abs (P_all_three_def - 0.0351) < 0.001 := 
by
  sorry

end probability_three_defective_phones_l266_266647


namespace min_sum_xy_l266_266338

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (pos_x : 0 < x) (pos_y : 0 < y)
  (h : (1 : ℚ) / x + 1 / y = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l266_266338


namespace equation_solutions_l266_266082

theorem equation_solutions :
  ∀ x y : ℤ, x^2 + x * y + y^2 + x + y - 5 = 0 → (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -3) ∨ (x = -3 ∧ y = 1) :=
by
  intro x y h
  sorry

end equation_solutions_l266_266082


namespace rice_mixed_grain_amount_l266_266905

theorem rice_mixed_grain_amount (total_rice : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) (proportion : ℚ) 
    (h1 : total_rice = 1536) 
    (h2 : sample_size = 256)
    (h3 : mixed_in_sample = 18)
    (h4 : proportion = mixed_in_sample / sample_size) : 
    total_rice * proportion = 108 :=
  sorry

end rice_mixed_grain_amount_l266_266905


namespace smallest_x_y_sum_l266_266005

theorem smallest_x_y_sum (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y)
                        (h4 : (1 / (x : ℝ)) + (1 / (y : ℝ)) = (1 / 20)) :
    x + y = 81 :=
sorry

end smallest_x_y_sum_l266_266005


namespace MrC_loses_240_after_transactions_l266_266090

theorem MrC_loses_240_after_transactions :
  let house_initial_value := 12000
  let first_transaction_loss_percent := 0.15
  let second_transaction_gain_percent := 0.20
  let house_value_after_first_transaction :=
    house_initial_value * (1 - first_transaction_loss_percent)
  let house_value_after_second_transaction :=
    house_value_after_first_transaction * (1 + second_transaction_gain_percent)
  house_value_after_second_transaction - house_initial_value = 240 :=
by
  sorry

end MrC_loses_240_after_transactions_l266_266090


namespace final_balance_is_103_5_percent_of_initial_l266_266436

/-- Define Megan's initial balance. -/
def initial_balance : ℝ := 125

/-- Define the balance after 25% increase from babysitting. -/
def after_babysitting (balance : ℝ) : ℝ :=
  balance + (balance * 0.25)

/-- Define the balance after 20% decrease from buying shoes. -/
def after_shoes (balance : ℝ) : ℝ :=
  balance - (balance * 0.20)

/-- Define the balance after 15% increase by investing in stocks. -/
def after_stocks (balance : ℝ) : ℝ :=
  balance + (balance * 0.15)

/-- Define the balance after 10% decrease due to medical expenses. -/
def after_medical_expense (balance : ℝ) : ℝ :=
  balance - (balance * 0.10)

/-- Define the final balance. -/
def final_balance : ℝ :=
  let b1 := after_babysitting initial_balance
  let b2 := after_shoes b1
  let b3 := after_stocks b2
  after_medical_expense b3

/-- Prove that the final balance is 103.5% of the initial balance. -/
theorem final_balance_is_103_5_percent_of_initial :
  final_balance / initial_balance = 1.035 :=
by
  unfold final_balance
  unfold initial_balance
  unfold after_babysitting
  unfold after_shoes
  unfold after_stocks
  unfold after_medical_expense
  sorry

end final_balance_is_103_5_percent_of_initial_l266_266436


namespace relationship_between_m_and_n_l266_266431

variable (a b m n : ℝ)

axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : m = Real.sqrt a - Real.sqrt b
axiom h4 : n = Real.sqrt (a - b)

theorem relationship_between_m_and_n : m < n :=
by
  -- Lean requires 'sorry' to be used as a placeholder for the proof
  sorry

end relationship_between_m_and_n_l266_266431


namespace smallest_x_satisfying_expression_l266_266777

theorem smallest_x_satisfying_expression :
  ∃ x : ℤ, (∃ k : ℤ, x^2 + x + 7 = k * (x - 2)) ∧ (∀ y : ℤ, (∃ k' : ℤ, y^2 + y + 7 = k' * (y - 2)) → y ≥ x) ∧ x = -11 :=
by
  sorry

end smallest_x_satisfying_expression_l266_266777


namespace recurrence_relation_l266_266409

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l266_266409


namespace gcd_18_30_l266_266296

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266296


namespace student_B_more_stable_than_A_student_B_more_stable_l266_266220

-- Define students A and B.
structure Student :=
  (average_score : ℝ)
  (variance : ℝ)

-- Given data for both students.
def studentA : Student :=
  { average_score := 90, variance := 51 }

def studentB : Student :=
  { average_score := 90, variance := 12 }

-- The theorem that student B has more stable performance than student A.
theorem student_B_more_stable_than_A (A B : Student) (h_avg : A.average_score = B.average_score) :
  A.variance > B.variance → B.variance < A.variance :=
by
  intro h
  linarith

-- Specific instance of the theorem with given data for students A and B.
theorem student_B_more_stable : studentA.variance > studentB.variance → studentB.variance < studentA.variance :=
  student_B_more_stable_than_A studentA studentB rfl

end student_B_more_stable_than_A_student_B_more_stable_l266_266220


namespace fraction_value_l266_266334

theorem fraction_value
  (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 :=
by
  sorry

end fraction_value_l266_266334


namespace transport_connectivity_l266_266423

-- Define the condition that any two cities are connected by either an air route or a canal.
-- We will formalize this with an inductive type to represent the transport means: AirRoute or Canal.
inductive TransportMeans
| AirRoute : TransportMeans
| Canal : TransportMeans

open TransportMeans

-- Represent cities as a type 'City'
universe u
variable (City : Type u)

-- Connect any two cities by a transport means
variable (connected : City → City → TransportMeans)

-- We want to prove that for any set of cities, 
-- there exists a means of transport such that starting from any city,
-- it is possible to reach any other city using only that means of transport.
theorem transport_connectivity (n : ℕ) (h2 : n ≥ 2) : 
  ∃ (T : TransportMeans), ∀ (c1 c2 : City), connected c1 c2 = T :=
by
  sorry

end transport_connectivity_l266_266423


namespace remainder_101_pow_37_mod_100_l266_266197

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l266_266197


namespace solve_abs_system_eq_l266_266602

theorem solve_abs_system_eq (x y : ℝ) :
  (|x + y| + |1 - x| = 6) ∧ (|x + y + 1| + |1 - y| = 4) ↔ x = -2 ∧ y = -1 :=
by sorry

end solve_abs_system_eq_l266_266602


namespace possible_arrangements_l266_266077

-- Define the three girls: Anya, Sanya, and Tanya
inductive Girl | A | S | T deriving DecidableEq

open Girl

-- Define the proof goal
theorem possible_arrangements : 
  {list.permutations [A, S, T]} = 
  { [[A, S, T], [A, T, S], [S, A, T], [S, T, A], [T, A, S], [T, S, A]] } :=
by
  sorry

end possible_arrangements_l266_266077


namespace mingming_actual_height_l266_266591

def mingming_height (h : ℝ) : Prop := 1.495 ≤ h ∧ h < 1.505

theorem mingming_actual_height : ∃ α : ℝ, mingming_height α :=
by
  use 1.50
  sorry

end mingming_actual_height_l266_266591


namespace gcd_18_30_l266_266283

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266283


namespace infinite_danish_numbers_l266_266643

-- Definitions translated from problem conditions
def is_danish (n : ℕ) : Prop :=
  ∃ k, n = 3 * k ∨ n = 2 * 4 ^ k

theorem infinite_danish_numbers :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, is_danish n ∧ is_danish (2^n + n) := sorry

end infinite_danish_numbers_l266_266643


namespace boxes_neither_pens_nor_pencils_l266_266441

def total_boxes : ℕ := 10
def pencil_boxes : ℕ := 6
def pen_boxes : ℕ := 3
def both_boxes : ℕ := 2

theorem boxes_neither_pens_nor_pencils : (total_boxes - (pencil_boxes + pen_boxes - both_boxes)) = 3 :=
by
  sorry

end boxes_neither_pens_nor_pencils_l266_266441


namespace neg_exists_equiv_forall_l266_266994

theorem neg_exists_equiv_forall (p : ∃ n : ℕ, 2^n > 1000) :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ ∀ n : ℕ, 2^n ≤ 1000 := 
sorry

end neg_exists_equiv_forall_l266_266994


namespace probability_combined_event_l266_266193

def box_A := finset.range 21 \ {0}  -- {1, 2, ..., 20}
def box_B := finset.range 40 \ finset.range 10  -- {10, 11, ..., 39}

def tiles_A_less_10 := {n ∈ box_A | n < 10}.card
def tiles_B_odd_or_greater_35 := {n ∈ box_B | n % 2 = 1 ∨ n > 35}.card

theorem probability_combined_event : 
  (tiles_A_less_10 / box_A.card : ℚ) * (tiles_B_odd_or_greater_35 / box_B.card : ℚ) = (51 / 200 : ℚ) :=
by
  have h1 : tiles_A_less_10 = 9 := by sorry
  have h2 : tiles_B_odd_or_greater_35 = 17 := by sorry
  rw [h1, h2]
  norm_num
  sorry

end probability_combined_event_l266_266193


namespace part1_minimum_value_part2_max_k_l266_266012

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1_minimum_value : ∃ x₀ : ℝ, x₀ = Real.exp (-2) ∧ f x₀ = -Real.exp (-2) := 
by
  use Real.exp (-2)
  sorry

theorem part2_max_k (k : ℤ) : (∀ x > 1, f x > k * (x - 1)) → k ≤ 3 := 
by
  sorry

end part1_minimum_value_part2_max_k_l266_266012


namespace trig_identity_l266_266340

open Real

theorem trig_identity (α : ℝ) (h : 2 * sin α + cos α = 0) : 
  2 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = -12 / 5 :=
sorry

end trig_identity_l266_266340


namespace g_five_l266_266187

variable (g : ℝ → ℝ)

-- Given conditions
axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_three : g 3 = 4

-- Prove g(5) = 16 * (1 / 4)^(1/3)
theorem g_five : g 5 = 16 * (1 / 4)^(1/3) := by
  sorry

end g_five_l266_266187


namespace kasha_pasha_truth_difference_l266_266876

-- Define the conditions for Katya and Pasha
def katya_binomial_distribution := Binomial(4, 2/3)
def pasha_binomial_distribution := Binomial(4, 3/5)

theorem kasha_pasha_truth_difference :
  (∑ x in Finset.range 5,
    (katya_binomial_distribution.prob x) * (pasha_binomial_distribution.prob (x + 2))) = 48 / 625 :=
by
  sorry

end kasha_pasha_truth_difference_l266_266876


namespace tips_fraction_l266_266804

theorem tips_fraction {S T I : ℚ} (h1 : T = (7/4) * S) (h2 : I = S + T) : (T / I) = 7 / 11 :=
by
  sorry

end tips_fraction_l266_266804


namespace problem_1_problem_2_problem_3_l266_266531

open Set

-- Define the universal set U
def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

-- Problem Statements
theorem problem_1 : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

theorem problem_2 : (A ∩ B) ∩ C = ∅ := by
  sorry

theorem problem_3 : (U \ A) ∩ (U \ B) = {0, 3} := by
  sorry

end problem_1_problem_2_problem_3_l266_266531


namespace sum_abcd_l266_266124

theorem sum_abcd (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 :=
sorry

end sum_abcd_l266_266124


namespace leo_weight_l266_266144

theorem leo_weight 
  (L K E : ℝ)
  (h1 : L + 10 = 1.5 * K)
  (h2 : L + 10 = 0.75 * E)
  (h3 : L + K + E = 210) :
  L = 63.33 := 
sorry

end leo_weight_l266_266144


namespace P_has_real_root_l266_266838

def P : ℝ → ℝ := sorry
variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom a1_nonzero : a1 ≠ 0
axiom a2_nonzero : a2 ≠ 0
axiom a3_nonzero : a3 ≠ 0

axiom functional_eq (x : ℝ) :
  P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem P_has_real_root :
  ∃ x : ℝ, P x = 0 :=
sorry

end P_has_real_root_l266_266838


namespace sum_of_squares_eq_229_l266_266756

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l266_266756


namespace find_arithmetic_sequence_l266_266463

theorem find_arithmetic_sequence (a d : ℝ) : 
(a - d) + a + (a + d) = 6 ∧ (a - d) * a * (a + d) = -10 → 
  (a = 2 ∧ d = 3 ∨ a = 2 ∧ d = -3) :=
by
  sorry

end find_arithmetic_sequence_l266_266463


namespace find_integer_pairs_l266_266972

theorem find_integer_pairs (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) :=
by
  sorry

end find_integer_pairs_l266_266972


namespace f_neg1_gt_f_1_l266_266010

-- Definition of the function f and its properties.
variable {f : ℝ → ℝ}
variable (df : Differentiable ℝ f)
variable (eq_f : ∀ x : ℝ, f x = x^2 + 2 * x * f' 2)

-- The problem statement to prove f(-1) > f(1).
theorem f_neg1_gt_f_1 (h_deriv : ∀ x : ℝ, deriv f x = 2 * x - 8):
  f (-1) > f 1 :=
by
  sorry

end f_neg1_gt_f_1_l266_266010


namespace power_of_four_l266_266855

theorem power_of_four (x : ℕ) (h : 5^29 * 4^x = 2 * 10^29) : x = 15 := by
  sorry

end power_of_four_l266_266855


namespace smallest_base10_integer_l266_266928

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l266_266928


namespace ages_of_three_persons_l266_266185

theorem ages_of_three_persons (y m e : ℕ) 
  (h1 : e = m + 16)
  (h2 : m = y + 8)
  (h3 : e - 6 = 3 * (y - 6))
  (h4 : e - 6 = 2 * (m - 6)) :
  y = 18 ∧ m = 26 ∧ e = 42 := 
by 
  sorry

end ages_of_three_persons_l266_266185


namespace total_price_purchase_l266_266203

variable (S T : ℝ)

theorem total_price_purchase (h1 : 2 * S + T = 2600) (h2 : 900 = 1200 * 0.75) : 2600 + 900 = 3500 := by
  sorry

end total_price_purchase_l266_266203


namespace range_of_omega_l266_266843

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ zeros : ℝ, (f(x) = cos (ω * x) - 1) and (count_zeros (f(x),  [0, 2 * π]) = 3)) ↔ (2 ≤ ω ∧ ω < 3) := 
sorry

end range_of_omega_l266_266843


namespace tan_C_in_triangle_l266_266426

theorem tan_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : Real.tan A = 1) (h₃ : Real.tan B = 2) :
  Real.tan C = 3 :=
sorry

end tan_C_in_triangle_l266_266426


namespace decimal_equivalent_of_fraction_l266_266635

theorem decimal_equivalent_of_fraction :
  (16 : ℚ) / 50 = 32 / 100 :=
by sorry

end decimal_equivalent_of_fraction_l266_266635


namespace smallest_integer_a_l266_266477

theorem smallest_integer_a (a : ℤ) (b : ℤ) (h1 : a < 21) (h2 : 20 ≤ b) (h3 : b < 31) (h4 : (a : ℝ) / b < 2 / 3) : 13 < a :=
sorry

end smallest_integer_a_l266_266477


namespace sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l266_266592

open Real

theorem sqrt_1_plus_inv_squares_4_5 :
  sqrt (1 + 1/4^2 + 1/5^2) = 1 + 1/20 :=
by
  sorry

theorem sqrt_1_plus_inv_squares_general (n : ℕ) (h : 0 < n) :
  sqrt (1 + 1/n^2 + 1/(n+1)^2) = 1 + 1/(n * (n + 1)) :=
by
  sorry

theorem sqrt_101_100_plus_1_121 :
  sqrt (101/100 + 1/121) = 1 + 1/110 :=
by
  sorry

end sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l266_266592


namespace alpha_tan_beta_gt_beta_tan_alpha_l266_266439

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) 
: α * Real.tan β > β * Real.tan α := 
sorry

end alpha_tan_beta_gt_beta_tan_alpha_l266_266439


namespace omega_range_l266_266844

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l266_266844


namespace gcd_18_30_is_6_l266_266265

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266265


namespace find_a2023_l266_266004

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

theorem find_a2023 (a : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_cond1 : a 2 + a 7 = a 8 + 1)
  (h_cond2 : (a 4)^2 = a 2 * a 8) :
  a 2023 = 2023 := 
sorry

end find_a2023_l266_266004


namespace max_xy_l266_266717

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy <= 81 :=
by {
  sorry
}

end max_xy_l266_266717


namespace probability_statements_l266_266572

theorem probability_statements :
  let M N : Type → Prop
  let P : (Type → Prop) → ℚ
  (P(M) = 1/5 ∧ P(N) = 1/4 ∧ P(M ∪ N) = 9/20) ∧
  (P(M) = 1/2 ∧ P(N) = 1/3 ∧ P(M ∩ N) = 1/6 ∧ P(M ∩ N) = P(M) * P(N)) ∧
  (P(¬M) = 1/2 ∧ P(N) = 1/3 ∧ P(M ∩ N) = 1/6 ∧ P(M ∩ N) = P(M) * P(N)) ∧
  (P(M) = 1/2 ∧ P(¬N) = 1/3 ∧ P(M ∩ N) = 1/6 ∧ P(M ∩ N) ≠ P(M) * P(N)) ∧
  (P(M) = 1/2 ∧ P(N) = 1/3 ∧ P(¬(M ∩ N)) = 5/6 ∧ P(M ∩ N) = P(M) * P(N))
  → 4 = 4 :=
by
  sorry

end probability_statements_l266_266572


namespace george_hourly_rate_l266_266331

theorem george_hourly_rate (total_hours : ℕ) (total_amount : ℕ) (h1 : total_hours = 7 + 2)
  (h2 : total_amount = 45) : 
  total_amount / total_hours = 5 := 
by sorry

end george_hourly_rate_l266_266331


namespace prime_factor_of_T_l266_266204

-- Define constants and conditions
def x : ℕ := 2021
def T : ℕ := Nat.sqrt ((x + x) + (x - x) + (x * x) + (x / x))

-- Define what needs to be proved
theorem prime_factor_of_T : ∃ p : ℕ, Nat.Prime p ∧ Nat.factorization T p > 0 ∧ (∀ q : ℕ, Nat.Prime q ∧ Nat.factorization T q > 0 → q ≤ p) :=
sorry

end prime_factor_of_T_l266_266204


namespace find_f_2n_l266_266676

variable (f : ℤ → ℤ)
variable (n : ℕ)

axiom axiom1 {x y : ℤ} : f (x + y) = f x + f y + 2 * x * y + 1
axiom axiom2 : f (-2) = 1

theorem find_f_2n (n : ℕ) (h : n > 0) : f (2 * n) = 4 * n^2 + 2 * n - 1 := sorry

end find_f_2n_l266_266676


namespace power_of_two_contains_k_as_substring_l266_266834

theorem power_of_two_contains_k_as_substring (k : ℕ) (h1 : 1000 ≤ k) (h2 : k < 10000) : 
  ∃ n < 20000, ∀ m, 10^m * k ≤ 2^n ∧ 2^n < 10^(m+4) * (k+1) :=
sorry

end power_of_two_contains_k_as_substring_l266_266834


namespace race_distance_l266_266866

theorem race_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 30 → D / 30 = D / t)
                      (h2 : ∀ t : ℝ, t = 45 → D / 45 = D / t)
                      (h3 : ∀ d : ℝ, d = 33.333333333333336 → D - (D / 45) * 30 = d) :
  D = 100 :=
sorry

end race_distance_l266_266866


namespace polygonal_number_8_8_l266_266067

-- Definitions based on conditions
def triangular_number (n : ℕ) : ℕ := (n^2 + n) / 2
def square_number (n : ℕ) : ℕ := n^2
def pentagonal_number (n : ℕ) : ℕ := (3 * n^2 - n) / 2
def hexagonal_number (n : ℕ) : ℕ := (4 * n^2 - 2 * n) / 2

-- General formula for k-sided polygonal number
def polygonal_number (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

-- The proposition to be proved
theorem polygonal_number_8_8 : polygonal_number 8 8 = 176 := by
  sorry

end polygonal_number_8_8_l266_266067


namespace three_digit_numbers_excluding_adjacent_same_digits_is_correct_l266_266694

def num_valid_three_digit_numbers_exclude_adjacent_same_digits : Nat :=
  let total_numbers := 900
  let excluded_numbers_AAB := 81
  let excluded_numbers_BAA := 81
  total_numbers - (excluded_numbers_AAB + excluded_numbers_BAA)

theorem three_digit_numbers_excluding_adjacent_same_digits_is_correct :
  num_valid_three_digit_numbers_exclude_adjacent_same_digits = 738 := by
  sorry

end three_digit_numbers_excluding_adjacent_same_digits_is_correct_l266_266694


namespace proof_problem_l266_266958

-- Define the propositions as Lean terms
def prop1 : Prop := ∀ (l1 l2 : ℝ) (h1 : l1 ≠ 0 ∧ l2 ≠ 0), (l1 * l2 = -1) → (l1 ≠ l2)  -- Two perpendicular lines must intersect (incorrect definition)
def prop2 : Prop := ∀ (l : ℝ), ∃! (m : ℝ), (l * m = -1)  -- There is only one perpendicular line (incorrect definition)
def prop3 : Prop := (∀ (α β γ : ℝ), α = β → γ = 90 → α + γ = β + γ)  -- Equal corresponding angles when intersecting a third (incorrect definition)
def prop4 : Prop := ∀ (A B C : ℝ), (A = B ∧ B = C) → (A = C)  -- Transitive property of parallel lines

-- The statement that only one of these propositions is true, and it is the fourth one
theorem proof_problem (h1 : ¬ prop1) (h2 : ¬ prop2) (h3 : ¬ prop3) (h4 : prop4) : 
  ∃! (i : ℕ), i = 4 := 
by
  sorry

end proof_problem_l266_266958


namespace james_farmer_walk_distance_l266_266710

theorem james_farmer_walk_distance (d : ℝ) :
  ∃ d : ℝ,
    (∀ w : ℝ, (w = 300 + 50 → d = 20) ∧ 
             (w' = w * 1.30 ∧ w'' = w' * 1.20 → w'' = 546)) :=
by
  sorry

end james_farmer_walk_distance_l266_266710


namespace area_of_sector_l266_266342

theorem area_of_sector {R θ: ℝ} (hR: R = 2) (hθ: θ = (2 * Real.pi) / 3) :
  (1 / 2) * R^2 * θ = (4 / 3) * Real.pi :=
by
  simp [hR, hθ]
  norm_num
  linarith

end area_of_sector_l266_266342


namespace smallest_integer_repr_CCCD8_l266_266929

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l266_266929


namespace gcd_of_18_and_30_l266_266244

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266244


namespace gcd_of_18_and_30_l266_266288

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266288


namespace problem1_problem2_l266_266482

noncomputable def f (a x : ℝ) := a - (2 / x)

theorem problem1 (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → (f a x1 < f a x2)) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x < 2 * x)) → a ≤ 3 :=
sorry

end problem1_problem2_l266_266482


namespace speed_conversion_l266_266099

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph_expected : ℝ) :
  speed_mps = 35.0028 →
  conversion_factor = 3.6 →
  speed_kmph_expected = 126.01008 →
  speed_mps * conversion_factor = speed_kmph_expected :=
by
  intros h_mps h_cf h_kmph
  rw [h_mps, h_cf, h_kmph]
  sorry

end speed_conversion_l266_266099


namespace width_of_jordan_rectangle_l266_266480

def carol_length := 5
def carol_width := 24
def jordan_length := 2
def jordan_area := carol_length * carol_width

theorem width_of_jordan_rectangle : ∃ (w : ℝ), jordan_length * w = jordan_area ∧ w = 60 :=
by
  use 60
  simp [carol_length, carol_width, jordan_length, jordan_area]
  sorry

end width_of_jordan_rectangle_l266_266480


namespace marathon_problem_l266_266864

-- Defining the given conditions in the problem.
def john_position_right := 28
def john_position_left := 42
def mike_ahead := 10

-- Define total participants.
def total_participants := john_position_right + john_position_left - 1

-- Define Mike's positions based on the given conditions.
def mike_position_left := john_position_left - mike_ahead
def mike_position_right := john_position_right - mike_ahead

-- Proposition combining all the facts.
theorem marathon_problem :
  total_participants = 69 ∧ mike_position_left = 32 ∧ mike_position_right = 18 := by 
     sorry

end marathon_problem_l266_266864


namespace Atlantic_Call_additional_charge_is_0_20_l266_266921

def United_Telephone_base_rate : ℝ := 7.00
def United_Telephone_rate_per_minute : ℝ := 0.25
def Atlantic_Call_base_rate : ℝ := 12.00
def United_Telephone_total_charge_100_minutes : ℝ := United_Telephone_base_rate + 100 * United_Telephone_rate_per_minute
def Atlantic_Call_total_charge_100_minutes (x : ℝ) : ℝ := Atlantic_Call_base_rate + 100 * x

theorem Atlantic_Call_additional_charge_is_0_20 :
  ∃ x : ℝ, United_Telephone_total_charge_100_minutes = Atlantic_Call_total_charge_100_minutes x ∧ x = 0.20 :=
by {
  -- Since United_Telephone_total_charge_100_minutes = 32.00, we need to prove:
  -- Atlantic_Call_total_charge_100_minutes 0.20 = 32.00
  sorry
}

end Atlantic_Call_additional_charge_is_0_20_l266_266921


namespace quadratic_function_value_2_l266_266840

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_value_2 :
  f a b 2 = 3 :=
by
  -- Definitions and assumptions to be used
  sorry

end quadratic_function_value_2_l266_266840


namespace river_width_l266_266646

theorem river_width
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) (flow_rate_m_per_min : ℝ)
  (H_depth : depth = 5)
  (H_flow_rate_kmph : flow_rate_kmph = 4)
  (H_volume_per_minute : volume_per_minute = 6333.333333333333)
  (H_flow_rate_m_per_min : flow_rate_m_per_min = 66.66666666666667) :
  volume_per_minute / (depth * flow_rate_m_per_min) = 19 :=
by
  -- proof goes here
  sorry

end river_width_l266_266646


namespace no_negative_roots_of_P_l266_266533

def P (x : ℝ) : ℝ := x^4 - 5 * x^3 + 3 * x^2 - 7 * x + 1

theorem no_negative_roots_of_P : ∀ x : ℝ, P x = 0 → x ≥ 0 := 
by 
    sorry

end no_negative_roots_of_P_l266_266533


namespace smallest_integer_repr_CCCD8_l266_266930

theorem smallest_integer_repr_CCCD8 (C D : ℕ) (hC : C < 6) (hD : D < 8)
    (h_eq : 7 * C = 9 * D) : ∃ n : ℕ, (n = 7 * C) ∧ (n = 9 * D) ∧ (7 * C = 63) :=
by {
  existsi 63,
  split,
  { simp [←h_eq, mul_comm, mul_assoc, Nat.mul_div_cancel_left, Nat.gcd_eq, Nat.lcm_eq_gcd_mul] },
  { exact h_eq }
}

end smallest_integer_repr_CCCD8_l266_266930


namespace probability_two_boxes_three_same_color_l266_266863

def students : ℕ := 3
def block_colors : ℕ := 6
def boxes : ℕ := 6

axiom independent_placement : ∀ (s : ℕ), s = students → True  -- Placeholder for independence assumption

theorem probability_two_boxes_three_same_color :
  (students = 3) →
  (block_colors = 6) →
  (boxes = 6) →
  True → -- Placeholder for more detailed conditions about block placements
  (probability (exactly_two_boxes_receive_three_blocks_same_color students block_colors boxes) = 625 / 729) :=
by
  intros h_students h_colors h_boxes h_independent
  sorry

end probability_two_boxes_three_same_color_l266_266863


namespace usual_time_is_12_l266_266630

variable (S T : ℕ)

theorem usual_time_is_12 (h1: S > 0) (h2: 5 * (T + 3) = 4 * T) : T = 12 := 
by 
  sorry

end usual_time_is_12_l266_266630


namespace maximum_xy_value_l266_266376

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l266_266376


namespace max_m_ratio_l266_266681

theorem max_m_ratio (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∀ a b, (4 / a + 1 / b) ≥ m / (a + 4 * b)) :
  (m = 16) → (b / a = 1 / 4) :=
by sorry

end max_m_ratio_l266_266681


namespace find_f2_l266_266678

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def a : ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f x
axiom even_g : ∀ x, g (-x) = g x
axiom fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom g2_a : g 2 = a
axiom a_pos : a > 0
axiom a_ne1 : a ≠ 1

theorem find_f2 : f 2 = 15 / 4 := 
by sorry

end find_f2_l266_266678


namespace unique_function_satisfying_conditions_l266_266517

open Nat

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (f 1 = 1) ∧ (∀ n, f n * f (n + 2) = (f (n + 1))^2 + 1997)

theorem unique_function_satisfying_conditions :
  (∃! f : ℕ → ℕ, satisfies_conditions f) :=
sorry

end unique_function_satisfying_conditions_l266_266517


namespace maria_total_cost_l266_266586

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end maria_total_cost_l266_266586


namespace find_pos_integers_A_B_l266_266827

noncomputable def concat (A B : ℕ) : ℕ :=
  let b := Nat.log 10 B + 1
  A * 10 ^ b + B

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfiesConditions (A B : ℕ) : Prop :=
  isPerfectSquare (concat A B) ∧ concat A B = 2 * A * B

theorem find_pos_integers_A_B :
  ∃ (A B : ℕ), A = (5 ^ b + 1) / 2 ∧ B = 2 ^ b * A * 100 ^ m ∧ b % 2 = 1 ∧ ∀ m : ℕ, satisfiesConditions A B :=
sorry

end find_pos_integers_A_B_l266_266827


namespace math_problem_l266_266723

variable (a b c d : ℝ)

theorem math_problem 
    (h1 : a + b + c + d = 6)
    (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
    36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
    4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := 
by
    sorry

end math_problem_l266_266723


namespace max_xy_l266_266379

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l266_266379


namespace percentage_cost_for_overhead_l266_266070

theorem percentage_cost_for_overhead
  (P M N : ℝ)
  (hP : P = 48)
  (hM : M = 50)
  (hN : N = 12) :
  (P + M - P - N) / P * 100 = 79.17 := by
  sorry

end percentage_cost_for_overhead_l266_266070


namespace common_difference_is_3_l266_266000

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop := 
  ∀ n, a_n n = a1 + (n - 1) * d

-- Conditions
def a2_eq : a 2 = 3 := sorry
def a5_eq : a 5 = 12 := sorry

-- Theorem to prove the common difference is 3
theorem common_difference_is_3 :
  ∀ {a : ℕ → ℝ} {a1 d : ℝ},
  (arithmetic_sequence a a1 d)
  → a 2 = 3 
  → a 5 = 12 
  → d = 3 :=
  by
  intros a a1 d h_seq h_a2 h_a5
  sorry

end common_difference_is_3_l266_266000


namespace find_k_n_l266_266657

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l266_266657


namespace gcd_of_18_and_30_l266_266284

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266284


namespace probability_of_event_l266_266444

noncomputable def interval_probability : ℝ :=
  if 0 ≤ 1 ∧ 1 ≤ 1 then (1 - (1/3)) / (1 - 0) else 0

theorem probability_of_event :
  interval_probability = 2 / 3 :=
by
  rw [interval_probability]
  sorry

end probability_of_event_l266_266444


namespace find_a_l266_266722

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then -(Real.log (-x) / Real.log 2) + a else 0

theorem find_a (a : ℝ) :
  (f a (-2) + f a (-4) = 1) → a = 2 :=
by
  sorry

end find_a_l266_266722


namespace savings_fraction_l266_266805

theorem savings_fraction 
(P : ℝ) 
(f : ℝ) 
(h1 : P > 0) 
(h2 : 12 * f * P = 5 * (1 - f) * P) : 
    f = 5 / 17 :=
by
  sorry

end savings_fraction_l266_266805


namespace final_percentage_of_alcohol_l266_266213

theorem final_percentage_of_alcohol (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
  (removed_alcohol : ℝ) (added_water : ℝ) :
  initial_volume = 15 → initial_alcohol_percentage = 25 →
  removed_alcohol = 2 → added_water = 3 →
  ( ( (initial_alcohol_percentage / 100 * initial_volume - removed_alcohol) / 
    (initial_volume - removed_alcohol + added_water) ) * 100 = 10.9375) :=
by
  intros
  sorry

end final_percentage_of_alcohol_l266_266213


namespace find_f_value_l266_266350

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l266_266350


namespace mary_take_home_pay_l266_266590

def hourly_wage : ℝ := 8
def regular_hours : ℝ := 20
def first_overtime_hours : ℝ := 10
def second_overtime_hours : ℝ := 10
def third_overtime_hours : ℝ := 10
def remaining_overtime_hours : ℝ := 20
def social_security_tax_rate : ℝ := 0.08
def medicare_tax_rate : ℝ := 0.02
def insurance_premium : ℝ := 50

def regular_earnings := regular_hours * hourly_wage
def first_overtime_earnings := first_overtime_hours * (hourly_wage * 1.25)
def second_overtime_earnings := second_overtime_hours * (hourly_wage * 1.5)
def third_overtime_earnings := third_overtime_hours * (hourly_wage * 1.75)
def remaining_overtime_earnings := remaining_overtime_hours * (hourly_wage * 2)

def total_earnings := 
    regular_earnings + 
    first_overtime_earnings + 
    second_overtime_earnings + 
    third_overtime_earnings + 
    remaining_overtime_earnings

def social_security_tax := total_earnings * social_security_tax_rate
def medicare_tax := total_earnings * medicare_tax_rate
def total_taxes := social_security_tax + medicare_tax

def earnings_after_taxes := total_earnings - total_taxes
def earnings_take_home := earnings_after_taxes - insurance_premium

theorem mary_take_home_pay : earnings_take_home = 706 := by
  sorry

end mary_take_home_pay_l266_266590


namespace interest_earned_l266_266882

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : 
  P = 2000 → r = 0.05 → n = 5 → 
  A = P * (1 + r)^n → 
  A - P = 552.56 :=
by
  intro hP hr hn hA
  rw [hP, hr, hn] at hA
  sorry

end interest_earned_l266_266882


namespace max_xy_value_l266_266387

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l266_266387


namespace gcd_18_30_is_6_l266_266262

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266262


namespace map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l266_266616

theorem map_x_eq_3_and_y_eq_2_under_z_squared_to_uv :
  (∀ (z : ℂ), (z = 3 + I * z.im) → ((z^2).re = 9 - (9*z.im^2) / 36)) ∧
  (∀ (z : ℂ), (z = z.re + I * 2) → ((z^2).re = (4*z.re^2) / 16 - 4)) :=
by 
  sorry

end map_x_eq_3_and_y_eq_2_under_z_squared_to_uv_l266_266616


namespace probability_multiple_of_100_is_zero_l266_266541

def singleDigitMultiplesOf5 : Set ℕ := {5}
def primeNumbersLessThan50 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
def isMultipleOf100 (n : ℕ) : Prop := 100 ∣ n

theorem probability_multiple_of_100_is_zero :
  (∀ m ∈ singleDigitMultiplesOf5, ∀ p ∈ primeNumbersLessThan50, ¬ isMultipleOf100 (m * p)) →
  r = 0 :=
sorry

end probability_multiple_of_100_is_zero_l266_266541


namespace probability_correct_l266_266919

-- Definitions corresponding to the problem's conditions
def boxA_tiles : Finset ℕ := Finset.range 31  -- tiles numbered 1 through 30
def boxB_tiles : Finset ℕ := Finset.range' 21 50  -- tiles numbered 21 through 50

-- Definition of the events in the problem
def eventA (n : ℕ) : Prop := n < 20

def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def eventB (n : ℕ) : Prop := is_prime n ∨ is_odd n

-- Definitions of probabilities for each event
def probability_eventA : ℚ :=
  (boxA_tiles.filter eventA).card /. boxA_tiles.card

def probability_eventB : ℚ :=
  (boxB_tiles.filter eventB).card /. boxB_tiles.card

-- Combined probability
def combined_probability : ℚ :=
  probability_eventA * probability_eventB

-- The proof problem statement
theorem probability_correct : combined_probability = 19 / 60 :=
by
  -- Sorry placeholder for proof
  sorry

end probability_correct_l266_266919


namespace best_fitting_model_l266_266619

-- Define the \(R^2\) values for each model
def R2_Model1 : ℝ := 0.75
def R2_Model2 : ℝ := 0.90
def R2_Model3 : ℝ := 0.25
def R2_Model4 : ℝ := 0.55

-- State that Model 2 is the best fitting model
theorem best_fitting_model : R2_Model2 = max (max R2_Model1 R2_Model2) (max R2_Model3 R2_Model4) :=
by -- Proof skipped
  sorry

end best_fitting_model_l266_266619


namespace remainder_101_pow_37_mod_100_l266_266199

theorem remainder_101_pow_37_mod_100 :
  (101: ℤ) ≡ 1 [MOD 100] →
  (101: ℤ)^37 ≡ 1 [MOD 100] :=
by
  sorry

end remainder_101_pow_37_mod_100_l266_266199


namespace stream_current_rate_l266_266497

theorem stream_current_rate (r w : ℝ) : 
  (15 / (r + w) + 5 = 15 / (r - w)) → 
  (15 / (2 * r + w) + 1 = 15 / (2 * r - w)) →
  w = 2 := 
by
  sorry

end stream_current_rate_l266_266497


namespace gcd_18_30_l266_266303

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266303


namespace find_x_l266_266140

theorem find_x (x : ℝ) 
  (a : ℝ × ℝ := (2*x - 1, x + 3)) 
  (b : ℝ × ℝ := (x, 2*x + 1))
  (c : ℝ × ℝ := (1, 2))
  (h : (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0) :
  x = 3 :=
  sorry

end find_x_l266_266140


namespace max_f_l266_266434

noncomputable def S_n (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℚ :=
  S_n n / ((n + 32) * S_n (n + 1))

theorem max_f (n : ℕ) : f n ≤ 1 / 50 := sorry

-- Verify the bound is achieved for n = 8
example : f 8 = 1 / 50 := by
  unfold f S_n
  norm_num

end max_f_l266_266434


namespace color_circles_with_four_colors_l266_266178

theorem color_circles_with_four_colors (n : ℕ) (circles : Fin n → (ℝ × ℝ)) (radius : ℝ):
  (∀ i j, i ≠ j → dist (circles i) (circles j) ≥ 2 * radius) →
  ∃ f : Fin n → Fin 4, ∀ i j, dist (circles i) (circles j) < 2 * radius → f i ≠ f j :=
by
  sorry

end color_circles_with_four_colors_l266_266178


namespace find_kn_l266_266655

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l266_266655


namespace M_union_N_eq_l266_266691

open Set

def M : Set ℝ := { x | x^2 - 4 * x < 0 }
def N : Set ℝ := { x | abs x ≤ 2 }

theorem M_union_N_eq : M ∪ N = Ico (-2 : ℝ) 4 := by
  sorry

end M_union_N_eq_l266_266691


namespace cost_of_items_l266_266440

variable (e t d : ℝ)

noncomputable def ques :=
  5 * e + 5 * t + 2 * d

axiom cond1 : 3 * e + 4 * t = 3.40
axiom cond2 : 4 * e + 3 * t = 4.00
axiom cond3 : 5 * e + 4 * t + 3 * d = 7.50

theorem cost_of_items : ques e t d = 6.93 :=
by
  sorry

end cost_of_items_l266_266440


namespace gcd_of_18_and_30_l266_266243

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266243


namespace train_crossing_time_l266_266491

-- Define the conditions
def length_of_train : ℕ := 200  -- in meters
def speed_of_train_kmph : ℕ := 90  -- in km per hour
def length_of_tunnel : ℕ := 2500  -- in meters

-- Conversion of speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Define the total distance to be covered (train length + tunnel length)
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Define the expected time to cross the tunnel (in seconds)
def expected_time : ℕ := 108

-- The theorem statement to prove
theorem train_crossing_time : (total_distance / speed_of_train_mps) = expected_time := 
by
  sorry

end train_crossing_time_l266_266491


namespace A_scores_2_points_B_scores_at_least_2_points_l266_266551

-- Define the probabilities of outcomes.
def prob_A_win := 0.5
def prob_A_lose := 0.3
def prob_A_draw := 0.2

-- Calculate the probability of A scoring 2 points.
theorem A_scores_2_points : 
    (prob_A_win * prob_A_lose + prob_A_lose * prob_A_win + prob_A_draw * prob_A_draw) = 0.34 :=
by
  sorry

-- Calculate the probability of B scoring at least 2 points.
theorem B_scores_at_least_2_points : 
    (1 - (prob_A_win * prob_A_win + (prob_A_win * prob_A_draw + prob_A_draw * prob_A_win))) = 0.55 :=
by
  sorry

end A_scores_2_points_B_scores_at_least_2_points_l266_266551


namespace number_of_ordered_pairs_l266_266518

noncomputable def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k

theorem number_of_ordered_pairs :
  (∃ (n : ℕ), n = 29 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ x ≤ 2020 ∧ y ≤ 2020 →
    is_power_of_prime (3 * x^2 + 10 * x * y + 3 * y^2) → n = 29) :=
by
  sorry

end number_of_ordered_pairs_l266_266518


namespace multiply_658217_99999_l266_266330

theorem multiply_658217_99999 : 658217 * 99999 = 65821034183 := 
by
  sorry

end multiply_658217_99999_l266_266330


namespace quadruple_problem_l266_266166

open Finset

def quadruple_set := (finset.range 5).product (finset.range 5).product (finset.range 5).product (finset.range 5)

def count_even_sequences := 
  quadruple_set.filter 
    (λ quadruple, (let (a, (b, (c, d))) := quadruple in (a * d - b * c + 1) % 2 = 0)).card

theorem quadruple_problem : count_even_sequences = 136 := 
  by
  -- Proof is not needed, so we'll use sorry here.
  sorry

end quadruple_problem_l266_266166


namespace recurrence_relation_p_series_l266_266403

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l266_266403


namespace percentage_error_computation_l266_266489

theorem percentage_error_computation (x : ℝ) (h : 0 < x) : 
  let correct_result := 8 * x
  let erroneous_result := x / 8
  let error := |correct_result - erroneous_result|
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 :=
by
  sorry

end percentage_error_computation_l266_266489


namespace candy_division_l266_266649

theorem candy_division (total_candy : ℕ) (students : ℕ) (per_student : ℕ) 
  (h1 : total_candy = 344) (h2 : students = 43) : 
  total_candy / students = per_student ↔ per_student = 8 := 
by 
  sorry

end candy_division_l266_266649


namespace claire_balloons_l266_266501

def initial_balloons : ℕ := 50
def balloons_lost : ℕ := 12
def balloons_given_away : ℕ := 9
def balloons_received : ℕ := 11

theorem claire_balloons : initial_balloons - balloons_lost - balloons_given_away + balloons_received = 40 :=
by
  sorry

end claire_balloons_l266_266501


namespace solution_set_of_inequality_l266_266454

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l266_266454


namespace sum_of_squares_of_roots_l266_266967

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 9) (h2 : s₁ * s₂ = 14) :
  s₁^2 + s₂^2 = 53 :=
by
  sorry

end sum_of_squares_of_roots_l266_266967


namespace set_intersection_l266_266006

def A (x : ℝ) : Prop := -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3
def B (x : ℝ) : Prop := (x + 1) / x ≤ 0
def C_x_B (x : ℝ) : Prop := x < -1 ∨ x ≥ 0

theorem set_intersection :
  {x : ℝ | A x} ∩ {x : ℝ | C_x_B x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_l266_266006


namespace binom_15_12_eq_455_l266_266106

theorem binom_15_12_eq_455 : Nat.choose 15 12 = 455 := 
by sorry

end binom_15_12_eq_455_l266_266106


namespace simple_interest_problem_l266_266479

theorem simple_interest_problem (P : ℝ) (R : ℝ) (T : ℝ) : T = 10 → 
  ((P * R * T) / 100 = (4 / 5) * P) → R = 8 :=
by
  intros hT hsi
  sorry

end simple_interest_problem_l266_266479


namespace line_intersects_circle_l266_266993

noncomputable def line_eqn (a : ℝ) (x y : ℝ) : ℝ := a * x - y - a + 3
noncomputable def circle_eqn (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x - 2 * y - 4

-- Given the line l passes through M(1, 3)
def passes_through_M (a : ℝ) : Prop := line_eqn a 1 3 = 0

-- Given M(1, 3) is inside the circle
def M_inside_circle : Prop := circle_eqn 1 3 < 0

-- To prove the line intersects the circle
theorem line_intersects_circle (a : ℝ) (h1 : passes_through_M a) (h2 : M_inside_circle) : 
  ∃ p : ℝ × ℝ, line_eqn a p.1 p.2 = 0 ∧ circle_eqn p.1 p.2 = 0 :=
sorry

end line_intersects_circle_l266_266993


namespace sum_of_possible_n_values_l266_266034

theorem sum_of_possible_n_values (m n : ℕ) 
  (h : 0 < m ∧ 0 < n)
  (eq1 : 1/m + 1/n = 1/5) : 
  n = 6 ∨ n = 10 ∨ n = 30 → 
  m = 30 ∨ m = 10 ∨ m = 6 ∨ m = 5 ∨ m = 25 ∨ m = 1 →
  (6 + 10 + 30 = 46) := 
by 
  sorry

end sum_of_possible_n_values_l266_266034


namespace solve_quadratic_eq_solve_cubic_eq_l266_266207

-- Problem 1: Solve (x-1)^2 = 9
theorem solve_quadratic_eq (x : ℝ) (h : (x - 1) ^ 2 = 9) : x = 4 ∨ x = -2 := 
by 
  sorry

-- Problem 2: Solve (x+3)^3 / 3 - 9 = 0
theorem solve_cubic_eq (x : ℝ) (h : (x + 3) ^ 3 / 3 - 9 = 0) : x = 0 := 
by 
  sorry

end solve_quadratic_eq_solve_cubic_eq_l266_266207


namespace correct_proportion_expression_l266_266472

def is_fraction_correctly_expressed (numerator denominator : ℕ) (expression : String) : Prop :=
  -- Define the property of a correctly expressed fraction in English
  expression = "three-fifths"

theorem correct_proportion_expression : 
  is_fraction_correctly_expressed 3 5 "three-fifths" :=
by
  sorry

end correct_proportion_expression_l266_266472


namespace expression_positive_intervals_l266_266819

theorem expression_positive_intervals :
  {x : ℝ | (x + 2) * (x - 3) > 0} = {x | x < -2} ∪ {x | x > 3} :=
by
  sorry

end expression_positive_intervals_l266_266819


namespace arithmetic_geometric_sequences_l266_266987

noncomputable def geometric_sequence_sum (a q n : ℝ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequences (a : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  S 5 = geometric_sequence_sum a q 5 →
  2 * a * q = 6 + a * q^4 →
  S 5 = -31 / 2 :=
by
  intros hq1 hS5 hAR
  sorry

end arithmetic_geometric_sequences_l266_266987


namespace sum_of_squares_eq_229_l266_266754

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l266_266754


namespace normal_distribution_probability_l266_266669


open ProbabilityTheory MeasureTheory

noncomputable def P_6_lt_X_lt_7 : ℝ :=
  (cdf (normal 5 1) 7 - cdf (normal 5 1) 6)

theorem normal_distribution_probability :
  P_6_lt_X_lt_7 = 0.1359 :=
by
  sorry

end normal_distribution_probability_l266_266669


namespace math_expression_identity_l266_266814

theorem math_expression_identity :
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 :=
by
  sorry

end math_expression_identity_l266_266814


namespace problem_I_problem_II_l266_266135

-- Problem (I): Proving the inequality solution set
theorem problem_I (x : ℝ) : |x - 5| + |x + 6| ≤ 12 ↔ -13/2 ≤ x ∧ x ≤ 11/2 :=
by
  sorry

-- Problem (II): Proving the range of m
theorem problem_II (m : ℝ) : (∀ x : ℝ, |x - m| + |x + 6| ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end problem_I_problem_II_l266_266135


namespace gcd_of_18_and_30_l266_266317

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266317


namespace find_k_value_l266_266685

theorem find_k_value (x k : ℝ) (h : x = 2) (h_sol : (k / (x - 3)) - (1 / (3 - x)) = 1) : k = -2 :=
by
  -- sorry to suppress the actual proof
  sorry

end find_k_value_l266_266685


namespace min_value_x_plus_y_l266_266002

theorem min_value_x_plus_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
sorry

end min_value_x_plus_y_l266_266002


namespace remainder_of_101_pow_37_mod_100_l266_266196

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l266_266196


namespace sum_of_squares_eq_229_l266_266757

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l266_266757


namespace turtle_marathon_time_l266_266907

/-- Given a marathon distance of 42 kilometers and 195 meters and a turtle's speed of 15 meters per minute,
prove that the turtle will reach the finish line in 1 day, 22 hours, and 53 minutes. -/
theorem turtle_marathon_time :
  let speed := 15 -- meters per minute
  let distance_km := 42 -- kilometers
  let distance_m := 195 -- meters
  let total_distance := distance_km * 1000 + distance_m -- total distance in meters
  let time_min := total_distance / speed -- time to complete the marathon in minutes
  let hours := time_min / 60 -- time to complete the marathon in hours (division and modulus)
  let minutes := time_min % 60 -- remaining minutes after converting total minutes to hours
  let days := hours / 24 -- time to complete the marathon in days (division and modulus)
  let remaining_hours := hours % 24 -- remaining hours after converting total hours to days
  (days, remaining_hours, minutes) = (1, 22, 53) -- expected result
:= 
sorry

end turtle_marathon_time_l266_266907


namespace sunil_total_amount_l266_266629

noncomputable def principal (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  CI / ((1 + R / 100) ^ T - 1)

noncomputable def total_amount (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  let P := principal CI R T
  P + CI

theorem sunil_total_amount (CI : ℝ) (R : ℝ) (T : ℕ) :
  CI = 420 → R = 10 → T = 2 → total_amount CI R T = 2420 := by
  intros hCI hR hT
  rw [hCI, hR, hT]
  sorry

end sunil_total_amount_l266_266629


namespace perpendicular_line_through_point_l266_266239

noncomputable def is_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_line_through_point
  (line : ℝ → ℝ)
  (P : ℝ × ℝ)
  (h_line_eq : ∀ x, line x = 3 * x + 8)
  (hP : P = (2,1)) :
  ∃ a b c : ℝ, a * (P.1) + b * (P.2) + c = 0 ∧ is_perpendicular 3 (-b / a) ∧ a * 1 + b * 3 + c = 0 :=
sorry

end perpendicular_line_through_point_l266_266239


namespace nonnegative_interval_l266_266831

theorem nonnegative_interval (x : ℝ) : 
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3) ≥ 0 ↔ (x ≥ 0 ∧ x < 3) :=
by sorry

end nonnegative_interval_l266_266831


namespace existence_condition_l266_266430

variables {M : Type*} (A B C : Set M)

theorem existence_condition (M : Type*) (A B C : Set M) :
  (A \cap Bᶜ \cap Cᶜ = ∅ ∧ Aᶜ \cap B \cap C = ∅) ↔ ∃ (X : Set M), (X ∪ A) \ B = C :=
by sorry

end existence_condition_l266_266430


namespace multiply_of_Mari_buttons_l266_266714

-- Define the variables and constants from the problem
def Mari_buttons : ℕ := 8
def Sue_buttons : ℕ := 22
def Kendra_buttons : ℕ := 2 * Sue_buttons

-- Statement that we need to prove
theorem multiply_of_Mari_buttons : ∃ (x : ℕ), Kendra_buttons = 8 * x + 4 ∧ x = 5 := by
  sorry

end multiply_of_Mari_buttons_l266_266714


namespace determine_phi_l266_266524

theorem determine_phi (f : ℝ → ℝ) (φ : ℝ): 
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + 3 * φ)) ∧ 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∃ k : ℤ, φ = k * Real.pi / 3) :=
by 
  sorry

end determine_phi_l266_266524


namespace simplify_fraction_l266_266899

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l266_266899


namespace smallest_value_of_N_l266_266095

theorem smallest_value_of_N (l m n : ℕ) (N : ℕ) (h1 : (l-1) * (m-1) * (n-1) = 270) (h2 : N = l * m * n): 
  N = 420 :=
sorry

end smallest_value_of_N_l266_266095


namespace gcd_of_18_and_30_l266_266294

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266294


namespace max_length_small_stick_l266_266076

theorem max_length_small_stick (a b c : ℕ) 
  (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd (Nat.gcd a b) c = 4 :=
by
  rw [ha, hb, hc]
  -- At this point, the gcd calculus will be omitted, filing it with sorry
  sorry

end max_length_small_stick_l266_266076


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l266_266344

variable {n : ℕ}

-- Defining sequences and sums
def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry
def T (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Given conditions
axiom h1 : 2 * S n = 3 * a n - 3
axiom h2 : b 1 = a 1
axiom h3 : b 7 = b 1 * b 2
axiom a1_value : a 1 = 3
axiom d_value : ∃ d : ℕ, b 2 = b 1 + d ∧ b 7 = b 1 + 6 * d

theorem geometric_sequence_general_term : a n = 3 ^ n :=
by sorry

theorem arithmetic_sequence_sum : T n = n^2 + 2*n :=
by sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l266_266344


namespace sara_pumpkins_l266_266891

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l266_266891


namespace max_product_l266_266370

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l266_266370


namespace heating_time_correct_l266_266559

structure HeatingProblem where
  initial_temp : ℕ
  final_temp : ℕ
  heating_rate : ℕ

def time_to_heat (hp : HeatingProblem) : ℕ :=
  (hp.final_temp - hp.initial_temp) / hp.heating_rate

theorem heating_time_correct (hp : HeatingProblem) (h1 : hp.initial_temp = 20) (h2 : hp.final_temp = 100) (h3 : hp.heating_rate = 5) :
  time_to_heat hp = 16 :=
by
  sorry

end heating_time_correct_l266_266559


namespace recurrence_relation_l266_266412

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l266_266412


namespace solve_fraction_eq_l266_266513

theorem solve_fraction_eq (x : ℝ) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6) ↔ 
  (x = 7 ∨ x = -2) := 
by
  sorry

end solve_fraction_eq_l266_266513


namespace gcd_18_30_l266_266297

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266297


namespace y_mul_k_is_perfect_square_l266_266653

-- Defining y as given in the problem with its prime factorization
def y : Nat := 3^4 * (2^2)^5 * 5^6 * (2 * 3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Since the question asks for an integer k (in this case 75) such that y * k is a perfect square
def k : Nat := 75

-- The statement that needs to be proved
theorem y_mul_k_is_perfect_square : ∃ n : Nat, (y * k) = n^2 := 
by
  sorry

end y_mul_k_is_perfect_square_l266_266653


namespace difference_is_minus_four_l266_266729

def percentage_scoring_60 : ℝ := 0.15
def percentage_scoring_75 : ℝ := 0.25
def percentage_scoring_85 : ℝ := 0.40
def percentage_scoring_95 : ℝ := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85)

def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

def mean_score : ℝ :=
  (percentage_scoring_60 * score_60) +
  (percentage_scoring_75 * score_75) +
  (percentage_scoring_85 * score_85) +
  (percentage_scoring_95 * score_95)

def median_score : ℝ := score_85

def difference_mean_median : ℝ := mean_score - median_score

theorem difference_is_minus_four : difference_mean_median = -4 :=
by
  sorry

end difference_is_minus_four_l266_266729


namespace percentage_earth_fresh_water_l266_266611

theorem percentage_earth_fresh_water :
  let portion_land := 3 / 10
  let portion_water := 1 - portion_land
  let percent_salt_water := 97 / 100
  let percent_fresh_water := 1 - percent_salt_water
  100 * (portion_water * percent_fresh_water) = 2.1 :=
by
  sorry

end percentage_earth_fresh_water_l266_266611


namespace gcd_polynomial_multiple_528_l266_266526

-- Definition of the problem
theorem gcd_polynomial_multiple_528 (k : ℕ) : 
  gcd (3 * (528 * k) ^ 3 + (528 * k) ^ 2 + 4 * (528 * k) + 66) (528 * k) = 66 :=
by
  sorry

end gcd_polynomial_multiple_528_l266_266526


namespace volume_of_S_l266_266719

-- Define the region S in terms of the conditions
def region_S (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1.5 ∧ 
  abs x + abs y ≤ 1 ∧ 
  abs z ≤ 0.5

-- Define the volume calculation function
noncomputable def volume_S : ℝ :=
  sorry -- This is where the computation/theorem proving for volume would go

-- The theorem stating the volume of S
theorem volume_of_S : volume_S = 2 / 3 :=
  sorry

end volume_of_S_l266_266719


namespace printer_Z_time_l266_266598

theorem printer_Z_time (T_Z : ℝ) (h1 : (1.0 / 15.0 : ℝ) = (15.0 * ((1.0 / 12.0) + (1.0 / T_Z))) / 2.0833333333333335) : 
  T_Z = 18.0 :=
sorry

end printer_Z_time_l266_266598


namespace joe_time_to_store_l266_266873

theorem joe_time_to_store :
  ∀ (r_w : ℝ) (r_r : ℝ) (t_w t_r t_total : ℝ), 
   (r_r = 2 * r_w) → (t_w = 10) → (t_r = t_w / 2) → (t_total = t_w + t_r) → (t_total = 15) := 
by
  intros r_w r_r t_w t_r t_total hrw hrw_eq hr_tw hr_t_total
  sorry

end joe_time_to_store_l266_266873


namespace given_condition_required_solution_l266_266505

-- Define the polynomial f.
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

-- Given condition
theorem given_condition (x : ℝ) : f (x^2 + 2) = x^4 + 5 * x^2 := by sorry

-- Proving the required equivalence
theorem required_solution (x : ℝ) : f (x^2 - 2) = x^4 - 3 * x^2 - 4 := by sorry

end given_condition_required_solution_l266_266505


namespace maximum_xy_value_l266_266375

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l266_266375


namespace cos_alpha_l266_266696

theorem cos_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 :=
by
  sorry

end cos_alpha_l266_266696


namespace total_cost_maria_l266_266579

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end total_cost_maria_l266_266579


namespace remainder_3_pow_17_mod_5_l266_266471

theorem remainder_3_pow_17_mod_5 :
  (3^17) % 5 = 3 :=
by
  have h : 3^4 % 5 = 1 := by norm_num
  sorry

end remainder_3_pow_17_mod_5_l266_266471


namespace gcd_18_30_l266_266310

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266310


namespace total_wheels_l266_266769

theorem total_wheels (n_bicycles n_tricycles n_unicycles n_four_wheelers : ℕ)
                     (w_bicycle w_tricycle w_unicycle w_four_wheeler : ℕ)
                     (h1 : n_bicycles = 16)
                     (h2 : n_tricycles = 7)
                     (h3 : n_unicycles = 10)
                     (h4 : n_four_wheelers = 5)
                     (h5 : w_bicycle = 2)
                     (h6 : w_tricycle = 3)
                     (h7 : w_unicycle = 1)
                     (h8 : w_four_wheeler = 4)
  : (n_bicycles * w_bicycle + n_tricycles * w_tricycle
     + n_unicycles * w_unicycle + n_four_wheelers * w_four_wheeler) = 83 := by
  sorry

end total_wheels_l266_266769


namespace maria_total_cost_l266_266587

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end maria_total_cost_l266_266587


namespace add_neg3_and_2_mul_neg3_and_2_l266_266105

theorem add_neg3_and_2 : -3 + 2 = -1 := 
by
  sorry

theorem mul_neg3_and_2 : (-3) * 2 = -6 := 
by
  sorry

end add_neg3_and_2_mul_neg3_and_2_l266_266105


namespace gcd_of_18_and_30_l266_266320

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266320


namespace range_of_m_l266_266673

theorem range_of_m (y : ℝ) (x : ℝ) (xy_ne_zero : x * y ≠ 0) :
  (x^2 + 4 * y^2 = (m^2 + 3 * m) * x * y) → -4 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l266_266673


namespace sum_of_data_l266_266944

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) : a + b + c = 96 :=
by
  sorry

end sum_of_data_l266_266944


namespace second_offset_l266_266120

theorem second_offset (d : ℝ) (h1 : ℝ) (A : ℝ) (h2 : ℝ) : 
  d = 28 → h1 = 9 → A = 210 → h2 = 6 :=
by
  sorry

end second_offset_l266_266120


namespace range_alpha_minus_beta_l266_266365

theorem range_alpha_minus_beta (α β : ℝ) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π / 2) :
  - (3 * π) / 2 ≤ α - β ∧ α - β ≤ 0 :=
sorry

end range_alpha_minus_beta_l266_266365


namespace area_of_black_parts_l266_266058

theorem area_of_black_parts (x y : ℕ) (h₁ : x + y = 106) (h₂ : x + 2 * y = 170) : y = 64 :=
sorry

end area_of_black_parts_l266_266058


namespace gino_popsicle_sticks_left_l266_266978

-- Define the initial number of popsicle sticks Gino has
def initial_popsicle_sticks : ℝ := 63.0

-- Define the number of popsicle sticks Gino gives away
def given_away_popsicle_sticks : ℝ := 50.0

-- Expected number of popsicle sticks Gino has left
def expected_remaining_popsicle_sticks : ℝ := 13.0

-- Main theorem to be proven
theorem gino_popsicle_sticks_left :
  initial_popsicle_sticks - given_away_popsicle_sticks = expected_remaining_popsicle_sticks := 
by
  -- This is where the proof would go, but we leave it as 'sorry' for now
  sorry

end gino_popsicle_sticks_left_l266_266978


namespace problem_statement_l266_266570

-- Define the function f
def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Conditions given in the problem
variables (φ : ℝ)
variables (hφ : |φ| ≤ Real.pi / 2)
variables (hx1 : f φ (Real.pi / 6) = 1 / 2)
variables (hx2 : f φ (5 * Real.pi / 6) = 1 / 2)

-- The statement we need to prove
theorem problem_statement : f φ (3 * Real.pi / 4) = 0 :=
sorry

end problem_statement_l266_266570


namespace find_first_number_in_list_l266_266966

theorem find_first_number_in_list
  (x : ℕ)
  (h1 : x < 10)
  (h2 : ∃ n : ℕ, 2012 = x + 9 * n)
  : x = 5 :=
by
  sorry

end find_first_number_in_list_l266_266966


namespace simplify_and_evaluate_l266_266180

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  ((m ^ 2 - 9) / (m ^ 2 - 6 * m + 9) - 3 / (m - 3)) / (m ^ 2 / (m - 3)) = Real.sqrt 2 / 2 :=
by {
  -- Proof goes here
  sorry
}

end simplify_and_evaluate_l266_266180


namespace not_algorithm_is_C_l266_266621

-- Definitions based on the conditions recognized in a)
def option_A := "To go from Zhongshan to Beijing, first take a bus, then take a train."
def option_B := "The steps to solve a linear equation are to eliminate the denominator, remove the brackets, transpose terms, combine like terms, and make the coefficient 1."
def option_C := "The equation x^2 - 4x + 3 = 0 has two distinct real roots."
def option_D := "When solving the inequality ax + 3 > 0, the first step is to transpose terms, and the second step is to discuss the sign of a."

-- Problem statement
theorem not_algorithm_is_C : 
  (option_C ≠ "algorithm for solving a problem") ∧ 
  (option_A = "algorithm for solving a problem") ∧ 
  (option_B = "algorithm for solving a problem") ∧ 
  (option_D = "algorithm for solving a problem") :=
  by 
  sorry

end not_algorithm_is_C_l266_266621


namespace andy_late_duration_l266_266808

theorem andy_late_duration :
  let start_time := 7 * 60 + 15 in -- 7:15 AM converted to minutes
  let school_start := 8 * 60 in -- 8:00 AM converted to minutes
  let normal_travel_time := 30 in
  let red_light_stops := 3 * 4 in
  let construction_delay := 10 in
  let total_travel_time := normal_travel_time + red_light_stops + construction_delay in
  total_travel_time - (school_start - start_time) = 7 :=
by
  sorry

end andy_late_duration_l266_266808


namespace maximum_xy_value_l266_266374

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l266_266374


namespace find_z_l266_266448

theorem find_z (y z : ℝ) (k : ℝ) 
  (h1 : y = 3) (h2 : z = 16) (h3 : y ^ 2 * (z ^ (1 / 4)) = k)
  (h4 : k = 18) (h5 : y = 6) : z = 1 / 16 := by
  sorry

end find_z_l266_266448


namespace remaining_seeds_l266_266724

def initial_seeds : Nat := 54000
def seeds_per_zone : Nat := 3123
def number_of_zones : Nat := 7

theorem remaining_seeds (initial_seeds seeds_per_zone number_of_zones : Nat) : 
  initial_seeds - (seeds_per_zone * number_of_zones) = 32139 := 
by 
  sorry

end remaining_seeds_l266_266724


namespace total_cost_maria_l266_266581

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end total_cost_maria_l266_266581


namespace value_of_a_minus_b_l266_266853

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) : a - b = 7 ∨ a - b = 3 :=
sorry

end value_of_a_minus_b_l266_266853


namespace gcd_of_18_and_30_l266_266327

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266327


namespace zero_squared_sum_l266_266175

theorem zero_squared_sum (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := 
by 
  sorry

end zero_squared_sum_l266_266175


namespace additional_grazing_area_l266_266791

open Real

theorem additional_grazing_area :
  let initial_rope_length := 12
  let extended_rope_length := 21
  let angle_fraction := 3 / 4
  let area (r : ℝ) := angle_fraction * π * r^2
  let A1 := area initial_rope_length
  let A2 := area extended_rope_length
  let ΔA := A2 - A1
  ΔA ≈ 699.9 := 
  by 
    sorry

end additional_grazing_area_l266_266791


namespace simplify_fraction_l266_266782

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l266_266782


namespace triangle_area_tangent_log2_l266_266062

open Real

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem triangle_area_tangent_log2 :
  let y := log_base_2
  let f := fun x : ℝ => y x
  let deriv := (deriv f 1)
  let tangent_line := fun x : ℝ => deriv * (x - 1) + f 1
  let x_intercept := 1
  let y_intercept := tangent_line 0
  
  (1 : ℝ) * (abs y_intercept) / 2 = 1 / (2 * log 2) := by
  sorry

end triangle_area_tangent_log2_l266_266062


namespace rectangle_perimeters_l266_266092

theorem rectangle_perimeters (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 3 * (2 * a + 2 * b)) : 
  2 * (a + b) = 36 ∨ 2 * (a + b) = 28 :=
by sorry

end rectangle_perimeters_l266_266092


namespace chloe_sold_strawberries_l266_266230

noncomputable section

def cost_per_dozen : ℕ := 50
def sale_price_per_half_dozen : ℕ := 30
def total_profit : ℕ := 500
def profit_per_half_dozen := sale_price_per_half_dozen - (cost_per_dozen / 2)
def half_dozens_sold := total_profit / profit_per_half_dozen

theorem chloe_sold_strawberries : half_dozens_sold / 2 = 50 :=
by
  -- proof would go here
  sorry

end chloe_sold_strawberries_l266_266230


namespace arithmetic_seq_sum_l266_266184

theorem arithmetic_seq_sum(S : ℕ → ℝ) (d : ℝ) (h1 : S 5 < S 6) 
    (h2 : S 6 = S 7) (h3 : S 7 > S 8) : S 9 < S 5 := 
sorry

end arithmetic_seq_sum_l266_266184


namespace majors_selection_l266_266236

theorem majors_selection (majors : Finset ℕ) (A B : ℕ) (h : A ∈ majors) (h' : B ∈ majors) (h_card : majors.card = 7) :
  let S := majors.erase A;
  let T := majors.erase B;
  (majors.card.choose 3 - (2.choose 2) * (5.choose 1)) * 3.factorial = 180 :=
by
  sorry

end majors_selection_l266_266236


namespace rosie_pies_l266_266056

-- Definition of known conditions
def apples_per_pie (apples_pies_ratio : ℕ × ℕ) : ℕ :=
  apples_pies_ratio.1 / apples_pies_ratio.2

def pies_from_apples (total_apples : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples / apples_per_pie

-- Theorem statement
theorem rosie_pies (apples_pies_ratio : ℕ × ℕ) (total_apples : ℕ) :
  apples_pies_ratio = (12, 3) →
  total_apples = 36 →
  pies_from_apples total_apples (apples_per_pie apples_pies_ratio) = 9 :=
by
  intros h_ratio h_apples
  rw [h_ratio, h_apples]
  sorry

end rosie_pies_l266_266056


namespace sin_add_arcsin_arctan_l266_266654

theorem sin_add_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  Real.sin (a + b) = (2 + 3 * Real.sqrt 3) / 10 :=
by
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  sorry

end sin_add_arcsin_arctan_l266_266654


namespace simplest_form_fraction_C_l266_266081

def fraction_A (x : ℤ) (y : ℤ) : ℚ := (2 * x + 4) / (6 * x + 8)
def fraction_B (x : ℤ) (y : ℤ) : ℚ := (x + y) / (x^2 - y^2)
def fraction_C (x : ℤ) (y : ℤ) : ℚ := (x^2 + y^2) / (x + y)
def fraction_D (x : ℤ) (y : ℤ) : ℚ := (x^2 - y^2) / (x^2 - 2 * x * y + y^2)

theorem simplest_form_fraction_C (x y : ℤ) :
  ¬ (∃ (A : ℚ), A ≠ fraction_C x y ∧ (A = fraction_C x y)) :=
by
  intros
  sorry

end simplest_form_fraction_C_l266_266081


namespace largest_square_multiple_of_18_under_500_l266_266773

theorem largest_square_multiple_of_18_under_500 : 
  ∃ n : ℕ, n * n < 500 ∧ n * n % 18 = 0 ∧ (∀ m : ℕ, m * m < 500 ∧ m * m % 18 = 0 → m * m ≤ n * n) → 
  n * n = 324 :=
by
  sorry

end largest_square_multiple_of_18_under_500_l266_266773


namespace parts_processed_per_hour_before_innovation_l266_266492

theorem parts_processed_per_hour_before_innovation 
    (x : ℕ) 
    (h1 : ∀ x, (∃ x, x > 0)) 
    (h2 : 2.5 * x > x) 
    (h3 : ∀ x, 1500 / x - 1500 / (2.5 * x) = 18): 
    x = 50 := 
sorry

end parts_processed_per_hour_before_innovation_l266_266492


namespace gcd_of_18_and_30_l266_266286

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266286


namespace triangle_side_lengths_exist_l266_266527

theorem triangle_side_lengths_exist 
  (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  ∃ (x y z : ℝ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
  (a = y + z) ∧ (b = x + z) ∧ (c = x + y) :=
by
  let x := (a - b + c) / 2
  let y := (a + b - c) / 2
  let z := (-a + b + c) / 2
  have hx : x > 0 := sorry
  have hy : y > 0 := sorry
  have hz : z > 0 := sorry
  have ha : a = y + z := sorry
  have hb : b = x + z := sorry
  have hc : c = x + y := sorry
  exact ⟨x, y, z, hx, hy, hz, ha, hb, hc⟩

end triangle_side_lengths_exist_l266_266527


namespace sequence_satisfies_conditions_l266_266626

theorem sequence_satisfies_conditions :
  ∃ (S : Fin 20 → ℝ),
    (∀ i, i < 18 → S i + S (i + 1) + S (i + 2) > 0) ∧
    (∑ i, S i < 0) :=
by
  let S : Fin 20 → ℝ := 
    fun n => match n.1 with
             | 0 => -3
             | 1 => -3
             | 2 => 6.5
             | 3 => -3
             | 4 => -3
             | 5 => 6.5
             | 6 => -3
             | 7 => -3
             | 8 => 6.5
             | 9 => -3
             | 10 => -3
             | 11 => 6.5
             | 12 => -3
             | 13 => -3
             | 14 => 6.5
             | 15 => -3
             | 16 => -3
             | 17 => 6.5
             | 18 => -3
             | 19 => -3
  use S
  split
  {
    intro i
    intro h
    -- We will skip the detailed proof here
    sorry
  }
  {
    -- We will skip the detailed proof here
    sorry
  }

end sequence_satisfies_conditions_l266_266626


namespace tan_value_l266_266695

theorem tan_value (α : ℝ) 
  (h : (2 * Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) - 1) / (Real.sqrt 2 * Real.sin (2 * α + π / 4)) = 4) : 
  Real.tan (2 * α + π / 4) = 1 / 4 :=
by
  sorry

end tan_value_l266_266695


namespace number_of_zeros_l266_266068

noncomputable def f (x : Real) : Real :=
if x > 0 then -1 + Real.log x
else 3 * x + 4

theorem number_of_zeros : (∃ a b : Real, f a = 0 ∧ f b = 0 ∧ a ≠ b) := 
sorry

end number_of_zeros_l266_266068


namespace inclination_angle_l266_266689

theorem inclination_angle (α : ℝ) :
  (∃ t1 t2 : ℝ, t1 + t2 = 2 * real.sin α ∧ t1 * t2 = -3 ∧ (t1 - t2).abs = real.sqrt 15) →
  (α = π / 3 ∨ α = 2 * π / 3) :=
by
  sorry

end inclination_angle_l266_266689


namespace max_xy_value_l266_266385

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l266_266385


namespace william_probability_l266_266622

def probability_of_correct_answer (p : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  1 - q^n

theorem william_probability :
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  probability_of_correct_answer p q n = 11529 / 15625 :=
by
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  unfold probability_of_correct_answer
  sorry

end william_probability_l266_266622


namespace possible_triangle_perimeters_l266_266545

theorem possible_triangle_perimeters :
  {p | ∃ (a b c : ℝ), ((a = 3 ∨ a = 6) ∧ (b = 3 ∨ b = 6) ∧ (c = 3 ∨ c = 6)) ∧
                        (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
                        p = a + b + c} = {9, 15, 18} :=
by
  sorry

end possible_triangle_perimeters_l266_266545


namespace jim_profit_percentage_l266_266713

theorem jim_profit_percentage (S C : ℝ) (H1 : S = 670) (H2 : C = 536) :
  ((S - C) / C) * 100 = 25 :=
by
  sorry

end jim_profit_percentage_l266_266713


namespace trigonometric_identity_l266_266671

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
  sorry

end trigonometric_identity_l266_266671


namespace gcd_of_18_and_30_l266_266285

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266285


namespace sequences_equal_l266_266851

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2018 / (n + 1)) * a (n + 1) + a n

noncomputable def b : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2020 / (n + 1)) * b (n + 1) + b n

theorem sequences_equal :
  (a 1010) / 1010 = (b 1009) / 1009 :=
sorry

end sequences_equal_l266_266851


namespace stratified_sampling_height_group_selection_l266_266217

theorem stratified_sampling_height_group_selection :
  let total_students := 100
  let group1 := 20
  let group2 := 50
  let group3 := 30
  let total_selected := 18
  group1 + group2 + group3 = total_students →
  (group3 : ℝ) / total_students * total_selected = 5.4 →
  round ((group3 : ℝ) / total_students * total_selected) = 3 :=
by
  intros total_students group1 group2 group3 total_selected h1 h2
  sorry

end stratified_sampling_height_group_selection_l266_266217


namespace carl_first_to_roll_six_l266_266225

-- Definitions based on problem conditions
def prob_six := 1 / 6
def prob_not_six := 5 / 6

-- Define geometric series sum formula for the given context
theorem carl_first_to_roll_six :
  ∑' n : ℕ, (prob_not_six^(3*n+1) * prob_six) = 25 / 91 :=
by
  sorry

end carl_first_to_roll_six_l266_266225


namespace math_team_selection_l266_266104

/-- At Clearview High School, the math team is being selected from a club that includes 
four girls and six boys. Prove that the number of different teams consisting of three girls 
and two boys can be formed is 180 if one of the girls must be the team captain. -/
theorem math_team_selection (G B : ℕ) (choose : ℕ → ℕ → ℕ) (girls captains remaining_boys : ℕ) :
  G = 4 → B = 6 → 
  girls = choose G 1 * choose (G - 1) 2 → 
  captains = choose B 2 →
  remaining_boys = girls * captains →
  remaining_boys = 180 := 
by 
  intro hG hB hgirls hcaptains hresult
  rw [hG, hB] at *
  simp only [Nat.choose] at *
  sorry

end math_team_selection_l266_266104


namespace gcd_18_30_l266_266279

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266279


namespace inequality_proof_l266_266878

/-- Given a and b are positive and satisfy the inequality ab > 2007a + 2008b,
    prove that a + b > (sqrt 2007 + sqrt 2008)^2 -/
theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 :=
by
  sorry

end inequality_proof_l266_266878


namespace recurrence_relation_l266_266410

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l266_266410


namespace chord_length_of_circle_and_line_intersection_l266_266188

theorem chord_length_of_circle_and_line_intersection :
  ∀ (x y : ℝ), (x - 2 * y = 3) → ((x - 2)^2 + (y + 3)^2 = 9) → ∃ chord_length : ℝ, (chord_length = 4) :=
by
  intros x y hx hy
  sorry

end chord_length_of_circle_and_line_intersection_l266_266188


namespace smallest_integer_CC6_DD8_l266_266931

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l266_266931


namespace gcd_18_30_l266_266316

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266316


namespace triangle_side_length_l266_266525

theorem triangle_side_length (a b c : ℝ) (A : ℝ) 
  (h_a : a = 2) (h_c : c = 2) (h_A : A = 30) :
  b = 2 * Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l266_266525


namespace simplify_and_evaluate_l266_266057

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 - (1 / (x - 1))) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)) = (2 / 5) :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l266_266057


namespace find_ab_l266_266700

theorem find_ab
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36)
  : a * b = -15 :=
by
  sorry

end find_ab_l266_266700


namespace remainder_of_101_pow_37_mod_100_l266_266195

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l266_266195


namespace line_parabola_midpoint_l266_266059

theorem line_parabola_midpoint (a b : ℝ) 
  (r s : ℝ) 
  (intersects_parabola : ∀ x, x = r ∨ x = s → ax + b = x^2)
  (midpoint_cond : (r + s) / 2 = 5 ∧ (r^2 + s^2) / 2 = 101) :
  a + b = -41 :=
sorry

end line_parabola_midpoint_l266_266059


namespace people_lost_l266_266498

-- Define the given constants
def win_ratio : ℕ := 4
def lose_ratio : ℕ := 1
def people_won : ℕ := 28

-- The statement to prove that 7 people lost
theorem people_lost (win_ratio lose_ratio people_won : ℕ) (H : win_ratio * 7 = people_won * lose_ratio) : 7 = people_won * lose_ratio / win_ratio :=
by { sorry }

end people_lost_l266_266498


namespace geom_seq_sum_l266_266836

variable {a : ℕ → ℝ}

theorem geom_seq_sum (h : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 ∨ a 5 + a 7 = -6 := by
  sorry

end geom_seq_sum_l266_266836


namespace area_enclosed_by_curve_l266_266060

theorem area_enclosed_by_curve :
  ∃ (area : ℝ), (∀ (x y : ℝ), |x - 1| + |y - 1| = 1 → area = 2) :=
sorry

end area_enclosed_by_curve_l266_266060


namespace gcd_18_30_l266_266298

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266298


namespace arithmetic_formula_geometric_formula_comparison_S_T_l266_266159

noncomputable def a₁ : ℕ := 16
noncomputable def d : ℤ := -3

def a_n (n : ℕ) : ℤ := -3 * (n : ℤ) + 19
def b_n (n : ℕ) : ℤ := 4^(3 - n)

def S_n (n : ℕ) : ℚ := (-3 * (n : ℚ)^2 + 35 * n) / 2
def T_n (n : ℕ) : ℤ := -n^2 + 3 * n

theorem arithmetic_formula (n : ℕ) : a_n n = -3 * n + 19 :=
sorry

theorem geometric_formula (n : ℕ) : b_n n = 4^(3 - n) :=
sorry

theorem comparison_S_T (n : ℕ) :
  if n = 29 then S_n n = (T_n n : ℚ)
  else if n < 29 then S_n n > (T_n n : ℚ)
  else S_n n < (T_n n : ℚ) :=
sorry

end arithmetic_formula_geometric_formula_comparison_S_T_l266_266159


namespace exponential_inequality_l266_266522

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (Real.exp a * Real.exp c > Real.exp b * Real.exp d) :=
by sorry

end exponential_inequality_l266_266522


namespace max_xy_l266_266380

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l266_266380


namespace cubs_more_home_runs_than_cardinals_l266_266661

theorem cubs_more_home_runs_than_cardinals 
(h1 : 2 + 1 + 2 = 5) 
(h2 : 1 + 1 = 2) : 
5 - 2 = 3 :=
by sorry

end cubs_more_home_runs_than_cardinals_l266_266661


namespace recurrence_relation_l266_266417

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l266_266417


namespace draw_is_unfair_suit_hierarchy_makes_fair_l266_266149

structure Card where
  suit : ℕ -- 4 suits numbered from 0 to 3
  rank : ℕ -- 9 ranks numbered from 0 to 8

def deck : List Card :=
  List.join (List.map (λ s, List.map (λ r, ⟨s, r⟩) (List.range 9)) (List.range 4))

def DrawFair? : (deck : List Card) → Prop := sorry

-- Part (a): Prove that the draw is unfair
theorem draw_is_unfair : ¬ DrawFair? deck := sorry

-- Part (b): Prove that introducing a suit hierarchy can make the draw fair
def suit_hierarchy : Card → Card → Prop :=
λ c1 c2, (c1.rank < c2.rank) ∨ (c1.rank = c2.rank ∧ c1.suit < c2.suit)

theorem suit_hierarchy_makes_fair : ∃ h : Card → Card → Prop, h = suit_hierarchy ∧ DrawFair? deck[h] := sorry

end draw_is_unfair_suit_hierarchy_makes_fair_l266_266149


namespace area_of_WIN_sector_l266_266487

theorem area_of_WIN_sector (r : ℝ) (p : ℝ) (A_circ : ℝ) (A_WIN : ℝ) 
    (h_r : r = 15) 
    (h_p : p = 1 / 3) 
    (h_A_circ : A_circ = π * r^2) 
    (h_A_WIN : A_WIN = p * A_circ) :
    A_WIN = 75 * π := 
sorry

end area_of_WIN_sector_l266_266487


namespace ratio_final_to_initial_l266_266547

def initial_amount (P : ℝ) := P
def interest_rate := 4 / 100
def time_period := 25

def simple_interest (P : ℝ) := P * interest_rate * time_period

def final_amount (P : ℝ) := P + simple_interest P

theorem ratio_final_to_initial (P : ℝ) (hP : P > 0) :
  final_amount P / initial_amount P = 2 := by
  sorry

end ratio_final_to_initial_l266_266547


namespace johns_contribution_l266_266699

-- Definitions
variables (A J : ℝ)
axiom h1 : 1.5 * A = 75
axiom h2 : (2 * A + J) / 3 = 75

-- Statement of the proof problem
theorem johns_contribution : J = 125 :=
by
  sorry

end johns_contribution_l266_266699


namespace product_percent_x_l266_266540

variables {x y z w : ℝ}
variables (h1 : 0.45 * z = 1.2 * y) 
variables (h2 : y = 0.75 * x) 
variables (h3 : z = 0.8 * w)

theorem product_percent_x :
  (w * y) / x = 1.875 :=
by 
  sorry

end product_percent_x_l266_266540


namespace smallest_possible_value_l266_266785

-- Definitions and conditions provided
def x_plus_4_y_minus_4_eq_zero (x y : ℝ) : Prop := (x + 4) * (y - 4) = 0

-- Main theorem to state
theorem smallest_possible_value (x y : ℝ) (h : x_plus_4_y_minus_4_eq_zero x y) : x^2 + y^2 = 32 :=
sorry

end smallest_possible_value_l266_266785


namespace arithmetic_sequence_a14_eq_41_l266_266705

theorem arithmetic_sequence_a14_eq_41 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 5) 
  (h_a6 : a 6 = 17) : 
  a 14 = 41 :=
sorry

end arithmetic_sequence_a14_eq_41_l266_266705


namespace current_time_is_208_l266_266554

def minute_hand_position (t : ℝ) : ℝ := 6 * t
def hour_hand_position (t : ℝ) : ℝ := 0.5 * t

theorem current_time_is_208 (t : ℝ) (h1 : 0 < t) (h2 : t < 60) 
  (h3 : minute_hand_position (t + 8) + 60 = hour_hand_position (t + 5)) : 
  t = 8 :=
by sorry

end current_time_is_208_l266_266554


namespace cost_price_proof_l266_266801

noncomputable def cost_price_per_bowl : ℚ := 1400 / 103

theorem cost_price_proof
  (total_bowls: ℕ) (sold_bowls: ℕ) (selling_price_per_bowl: ℚ)
  (percentage_gain: ℚ) 
  (total_bowls_eq: total_bowls = 110)
  (sold_bowls_eq: sold_bowls = 100)
  (selling_price_per_bowl_eq: selling_price_per_bowl = 14)
  (percentage_gain_eq: percentage_gain = 300 / 11) :
  (selling_price_per_bowl * sold_bowls - (sold_bowls + 3) * (selling_price_per_bowl / (3 * percentage_gain / 100))) = cost_price_per_bowl :=
by
  sorry

end cost_price_proof_l266_266801


namespace smallest_sum_of_squares_l266_266743

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l266_266743


namespace vertices_after_cut_off_four_corners_l266_266429

-- Definitions for the conditions
def regular_tetrahedron.num_vertices : ℕ := 4

def new_vertices_per_cut : ℕ := 3

def total_vertices_after_cut : ℕ := 
  regular_tetrahedron.num_vertices + regular_tetrahedron.num_vertices * new_vertices_per_cut

-- The theorem to prove the question
theorem vertices_after_cut_off_four_corners :
  total_vertices_after_cut = 12 :=
by
  -- sorry is used to skip the proof steps, as per instructions
  sorry

end vertices_after_cut_off_four_corners_l266_266429


namespace simplify_fraction_l266_266897

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem simplify_fraction : (180 = 2^2 * 3^2 * 5) ∧ (270 = 2 * 3^3 * 5) ∧ (gcd 180 270 = 90) →
  180 / 270 = 2 / 3 :=
by
  intro h
  cases h with h1 h2h3
  cases h2h3 with h2 h3
  sorry -- Proof is omitted

end simplify_fraction_l266_266897


namespace simplify_fraction_l266_266445

theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := by
  sorry

end simplify_fraction_l266_266445


namespace gcd_18_30_l266_266313

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266313


namespace quadratic_equation_equivalence_l266_266022

theorem quadratic_equation_equivalence
  (a_0 a_1 a_2 : ℝ)
  (r s : ℝ)
  (h_roots : a_0 + a_1 * r + a_2 * r^2 = 0 ∧ a_0 + a_1 * s + a_2 * s^2 = 0)
  (h_a2_nonzero : a_2 ≠ 0) :
  (∀ x, a_0 ≠ 0 ↔ a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s)) :=
sorry

end quadratic_equation_equivalence_l266_266022


namespace entrepreneurs_not_attending_any_session_l266_266810

theorem entrepreneurs_not_attending_any_session 
  (total_entrepreneurs : ℕ) 
  (digital_marketing_attendees : ℕ) 
  (e_commerce_attendees : ℕ) 
  (both_sessions_attendees : ℕ)
  (h1 : total_entrepreneurs = 40)
  (h2 : digital_marketing_attendees = 22) 
  (h3 : e_commerce_attendees = 18) 
  (h4 : both_sessions_attendees = 8) : 
  total_entrepreneurs - (digital_marketing_attendees + e_commerce_attendees - both_sessions_attendees) = 8 :=
by sorry

end entrepreneurs_not_attending_any_session_l266_266810


namespace smallest_possible_sum_of_squares_l266_266739

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l266_266739


namespace max_xy_value_l266_266384

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l266_266384


namespace max_product_l266_266371

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l266_266371


namespace greatest_xy_l266_266395

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l266_266395


namespace complex_expression_l266_266968

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_expression (i : ℂ) (h : imaginary_unit i) :
  (1 - i) ^ 2016 + (1 + i) ^ 2016 = 2 ^ 1009 :=
by
  sorry

end complex_expression_l266_266968


namespace find_A_l266_266233

def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

theorem find_A (A : ℝ) (h : diamond A 5 = 82) : A = 12 :=
by
  unfold diamond at h
  sorry

end find_A_l266_266233


namespace max_product_l266_266372

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l266_266372


namespace gcd_of_18_and_30_l266_266293

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266293


namespace certain_event_l266_266011

-- Definitions for a line and plane
inductive Line
| mk : Line

inductive Plane
| mk : Plane

-- Definitions for parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p₁ p₂ : Plane) : Prop := sorry

-- Given conditions and the proof statement
theorem certain_event (l : Line) (α β : Plane) (h1 : perpendicular l α) (h2 : perpendicular l β) : plane_parallel α β :=
sorry

end certain_event_l266_266011


namespace sum_of_inradii_eq_height_l266_266227

variables (a b c h b1 a1 : ℝ)
variables (r r1 r2 : ℝ)

-- Assume CH is the height of the right-angled triangle ABC from the vertex of the right angle.
-- r, r1, r2 are the radii of the incircles of triangles ABC, AHC, and BHC respectively.
-- Given definitions:
-- BC = a
-- AC = b
-- AB = c
-- AH = b1
-- BH = a1
-- CH = h

-- Formulas for the radii of the respective triangles:
-- r : radius of incircle of triangle ABC = (a + b - h) / 2
-- r1 : radius of incircle of triangle AHC = (h + b1 - b) / 2
-- r2 : radius of incircle of triangle BHC = (h + a1 - a) / 2

theorem sum_of_inradii_eq_height 
  (H₁ : r = (a + b - h) / 2)
  (H₂ : r1 = (h + b1 - b) / 2) 
  (H₃ : r2 = (h + a1 - a) / 2) 
  (H₄ : b1 = b - h) 
  (H₅ : a1 = a - h) : 
  r + r1 + r2 = h :=
by
  sorry

end sum_of_inradii_eq_height_l266_266227


namespace find_integers_divisible_by_18_in_range_l266_266510

theorem find_integers_divisible_by_18_in_range :
  ∃ n : ℕ, (n % 18 = 0) ∧ (n ≥ 900) ∧ (n ≤ 930) ∧ (n = 900 ∨ n = 918) :=
sorry

end find_integers_divisible_by_18_in_range_l266_266510


namespace number_of_terms_arithmetic_sequence_l266_266109

theorem number_of_terms_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -36 → d = 6 → l = 66 → ∃ n, l = a + (n-1) * d ∧ n = 18 :=
by
  intros a d l ha hd hl
  exists 18
  rw [ha, hd, hl]
  sorry

end number_of_terms_arithmetic_sequence_l266_266109


namespace tom_candy_pieces_l266_266192

/-!
# Problem Statement
Tom bought 14 boxes of chocolate candy, 10 boxes of fruit candy, and 8 boxes of caramel candy. 
He gave 8 chocolate boxes and 5 fruit boxes to his little brother. 
If each chocolate box has 3 pieces inside, each fruit box has 4 pieces, and each caramel box has 5 pieces, 
prove that Tom still has 78 pieces of candy.
-/

theorem tom_candy_pieces 
  (chocolate_boxes : ℕ := 14)
  (fruit_boxes : ℕ := 10)
  (caramel_boxes : ℕ := 8)
  (gave_away_chocolate_boxes : ℕ := 8)
  (gave_away_fruit_boxes : ℕ := 5)
  (chocolate_pieces_per_box : ℕ := 3)
  (fruit_pieces_per_box : ℕ := 4)
  (caramel_pieces_per_box : ℕ := 5)
  : chocolate_boxes * chocolate_pieces_per_box + 
    fruit_boxes * fruit_pieces_per_box + 
    caramel_boxes * caramel_pieces_per_box - 
    (gave_away_chocolate_boxes * chocolate_pieces_per_box + 
     gave_away_fruit_boxes * fruit_pieces_per_box) = 78 :=
by
  sorry

end tom_candy_pieces_l266_266192


namespace number_of_uncertain_events_is_three_l266_266452

noncomputable def cloudy_day_will_rain : Prop := sorry
noncomputable def fair_coin_heads : Prop := sorry
noncomputable def two_students_same_birth_month : Prop := sorry
noncomputable def olympics_2008_in_beijing : Prop := true

def is_uncertain (event: Prop) : Prop :=
  event ∧ ¬(event = true ∨ event = false)

theorem number_of_uncertain_events_is_three :
  is_uncertain cloudy_day_will_rain ∧
  is_uncertain fair_coin_heads ∧
  is_uncertain two_students_same_birth_month ∧
  ¬is_uncertain olympics_2008_in_beijing →
  3 = 3 :=
by sorry

end number_of_uncertain_events_is_three_l266_266452


namespace sequence_fifth_term_l266_266153

theorem sequence_fifth_term (a b c : ℕ) :
  (a = (2 + b) / 3) →
  (b = (a + 34) / 3) →
  (34 = (b + c) / 3) →
  c = 89 :=
by
  intros ha hb hc
  sorry

end sequence_fifth_term_l266_266153


namespace bob_needs_8_additional_wins_to_afford_puppy_l266_266960

variable (n : ℕ) (grand_prize_per_win : ℝ) (total_cost : ℝ)

def bob_total_wins_to_afford_puppy : Prop :=
  total_cost = 1000 ∧ grand_prize_per_win = 100 ∧ n = (total_cost / grand_prize_per_win) - 2

theorem bob_needs_8_additional_wins_to_afford_puppy :
  bob_total_wins_to_afford_puppy 8 100 1000 :=
by {
  sorry
}

end bob_needs_8_additional_wins_to_afford_puppy_l266_266960


namespace pentagon_coloring_count_l266_266461

-- Define the three colors
inductive Color
| Red
| Yellow
| Green

open Color

-- Define the pentagon coloring problem
def adjacent_different (color1 color2 : Color) : Prop :=
color1 ≠ color2

-- Define a coloring for the pentagon
structure PentagonColoring :=
(A B C D E : Color)
(adjAB : adjacent_different A B)
(adjBC : adjacent_different B C)
(adjCD : adjacent_different C D)
(adjDE : adjacent_different D E)
(adjEA : adjacent_different E A)

-- The main statement to prove
theorem pentagon_coloring_count :
  ∃ (colorings : Finset PentagonColoring), colorings.card = 30 := sorry

end pentagon_coloring_count_l266_266461


namespace probability_sum_of_two_dice_is_4_l266_266920

noncomputable def fair_dice_probability_sum_4 : ℚ :=
  let total_outcomes := 6 * 6 -- Total outcomes for two dice
  let favorable_outcomes := 3 -- Outcomes that sum to 4: (1, 3), (3, 1), (2, 2)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_two_dice_is_4 : fair_dice_probability_sum_4 = 1 / 12 := 
by
  sorry

end probability_sum_of_two_dice_is_4_l266_266920


namespace problem_solution_l266_266161

-- Definition of parametric equations for C1
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Definition of the curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop :=
  ∃ α : ℝ, x = 2 * Real.cos α ∧ y = 2 + 2 * Real.sin α

-- Definition of the point P related to the curve C1
def P (x y : ℝ) : Prop :=
  ∃ α : ℝ, x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

-- Cartesian equation of curve C2
def C2_Cartesian (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

-- Define the polar coordinates of curve C1 and C2
def C1_polar (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

-- Function to find intersection points and |AB|
noncomputable def AB_distance (θ : ℝ) : ℝ :=
  let ρ1 := 4 * Real.sin θ in
  let ρ2 := 8 * Real.sin θ in
  Real.abs (ρ2 - ρ1)

-- Main theorem statement verifying the solution components
theorem problem_solution :
  (∀ x y, C1 x y → P x y → C2_Cartesian x y) ∧
  (AB_distance (π / 3) = 2 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l266_266161


namespace gcd_of_18_and_30_l266_266249

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266249


namespace range_of_first_term_in_geometric_sequence_l266_266989

theorem range_of_first_term_in_geometric_sequence (q a₁ : ℝ)
  (h_q : |q| < 1)
  (h_sum : a₁ / (1 - q) = q) :
  -2 < a₁ ∧ a₁ ≤ 0.25 ∧ a₁ ≠ 0 :=
by
  sorry

end range_of_first_term_in_geometric_sequence_l266_266989


namespace ratio_of_x_y_l266_266703

theorem ratio_of_x_y (x y : ℚ) (h : (2 * x - y) / (x + y) = 2 / 3) : x / y = 5 / 4 :=
sorry

end ratio_of_x_y_l266_266703


namespace intersection_eq_T_l266_266025

noncomputable def S : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 3 * x + 2 }
noncomputable def T : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x ^ 2 - 1 }

theorem intersection_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_eq_T_l266_266025


namespace solution_x_y_zero_l266_266145

theorem solution_x_y_zero (x y : ℤ) (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 :=
by
sorry

end solution_x_y_zero_l266_266145


namespace intersection_eq_l266_266789

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_eq : A ∩ B = {-1, 3} := 
by
  sorry

end intersection_eq_l266_266789


namespace smallest_sum_of_squares_l266_266752

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l266_266752


namespace smallest_x_satisfies_eq_l266_266829

theorem smallest_x_satisfies_eq : ∃ x : ℝ, (1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))) ∧ x = 7 - Real.sqrt 6 :=
by
  -- The proof steps would go here, but we're skipping them with sorry for now.
  sorry

end smallest_x_satisfies_eq_l266_266829


namespace icing_two_sides_on_Jack_cake_l266_266164

noncomputable def Jack_cake_icing_two_sides (cake_size : ℕ) : ℕ :=
  let side_cubes := 4 * (cake_size - 2) * 3
  let vertical_edge_cubes := 4 * (cake_size - 2)
  side_cubes + vertical_edge_cubes

-- The statement to be proven
theorem icing_two_sides_on_Jack_cake : Jack_cake_icing_two_sides 5 = 96 :=
by
  sorry

end icing_two_sides_on_Jack_cake_l266_266164


namespace exists_sequence_satisfying_conditions_l266_266627

theorem exists_sequence_satisfying_conditions :
  ∃ seq : array ℝ 20, 
  (∀ i : ℕ, i < 18 → (seq[i] + seq[i+1] + seq[i+2] > 0)) ∧ 
  (Finset.univ.sum (fun i => seq[i]) < 0) :=
  sorry

end exists_sequence_satisfying_conditions_l266_266627


namespace gcd_18_30_is_6_l266_266269

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266269


namespace opposite_blue_face_is_white_l266_266895

-- Define colors
inductive Color
| Red
| Blue
| Orange
| Purple
| Green
| Yellow
| White

-- Define the positions of colors on the cube
structure CubeConfig :=
(top : Color)
(front : Color)
(bottom : Color)
(back : Color)
(left : Color)
(right : Color)

-- The given conditions
def cube_conditions (c : CubeConfig) : Prop :=
  c.top = Color.Purple ∧
  c.front = Color.Green ∧
  c.bottom = Color.Yellow ∧
  c.back = Color.Orange ∧
  c.left = Color.Blue ∧
  c.right = Color.White

-- The statement we need to prove
theorem opposite_blue_face_is_white (c : CubeConfig) (h : cube_conditions c) :
  c.right = Color.White :=
by
  -- Proof placeholder
  sorry

end opposite_blue_face_is_white_l266_266895


namespace lcm_nuts_bolts_l266_266112

theorem lcm_nuts_bolts : Nat.lcm 13 8 = 104 := 
sorry

end lcm_nuts_bolts_l266_266112


namespace find_fourth_intersection_point_l266_266869

-- Conditions: the equation xy = 1 and known intersection points
def hyperbola_eq (x y : ℝ) : Prop := x * y = 1

def known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/5, 5)]

-- The Fourth Point we need to prove
def fourth_point (p : ℝ × ℝ) : Prop := p = (-5/12, -12/5)

-- Main theorem: Given the known intersection points, prove the fourth intersection point
theorem find_fourth_intersection_point :
  ∃ (x y : ℝ), (hyperbola_eq x y) ∧ (∀ (a b : ℝ) (h : (a, b) ∈ known_points), hyperbola_eq a b) ∧ (fourth_point (x, y)) :=
by
  exists (-5/12)
  exists (-12/5)
  split
  · -- show that x * y = 1 for the fourth point
    sorry
  split
  · -- verify that the known points satisfy x * y = 1
    intros a b h
    rcases h with ⟨hx, hy⟩
    cases h_1
    · exact (by norm_num : hyperbola_eq 3 (1/3))
    · exact (by norm_num : hyperbola_eq (-4) (-1/4))
    · exact (by norm_num : hyperbola_eq (1/5) 5)
  · -- show that the fourth point matches our proposed value
    exact rfl
  sorry


end find_fourth_intersection_point_l266_266869


namespace periodic_modulo_h_l266_266976

open Nat

-- Defining the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Defining the sequence as per the problem
def x_seq (n : ℕ) : ℕ :=
  binom (2 * n) n

-- The main theorem stating the required condition
theorem periodic_modulo_h (h : ℕ) (h_gt_one : h > 1) :
  (∃ N, ∀ n ≥ N, x_seq n % h = x_seq (n + 1) % h) ↔ h = 2 :=
by
  sorry

end periodic_modulo_h_l266_266976


namespace hens_not_laying_eggs_l266_266771

def chickens_on_farm := 440
def number_of_roosters := 39
def total_eggs := 1158
def eggs_per_hen := 3

theorem hens_not_laying_eggs :
  (chickens_on_farm - number_of_roosters) - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end hens_not_laying_eggs_l266_266771


namespace propane_tank_and_burner_cost_l266_266360

theorem propane_tank_and_burner_cost
(Total_money: ℝ)
(Sheet_cost: ℝ)
(Rope_cost: ℝ)
(Helium_cost_per_oz: ℝ)
(Lift_per_oz: ℝ)
(Max_height: ℝ)
(ht: Total_money = 200)
(hs: Sheet_cost = 42)
(hr: Rope_cost = 18)
(hh: Helium_cost_per_oz = 1.50)
(hlo: Lift_per_oz = 113)
(hm: Max_height = 9492)
:
(Total_money - (Sheet_cost + Rope_cost) 
 - (Max_height / Lift_per_oz * Helium_cost_per_oz) 
 = 14) :=
by
  sorry

end propane_tank_and_burner_cost_l266_266360


namespace gcd_18_30_l266_266257

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266257


namespace max_xy_l266_266382

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l266_266382


namespace g_domain_l266_266234

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

theorem g_domain : {x : ℝ | -1 < x ∧ x < 1} = Set {x | ∃ y, g x = y} :=
by
  sorry

end g_domain_l266_266234


namespace greatest_xy_value_l266_266389

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l266_266389


namespace melanie_books_bought_l266_266174

def books_before_yard_sale : ℝ := 41.0
def books_after_yard_sale : ℝ := 128
def books_bought : ℝ := books_after_yard_sale - books_before_yard_sale

theorem melanie_books_bought : books_bought = 87 := by
  sorry

end melanie_books_bought_l266_266174


namespace valid_addends_l266_266468

noncomputable def is_valid_addend (n : ℕ) : Prop :=
  ∃ (X Y : ℕ), (100 * 9 + 10 * X + 4) = n ∧ (30 + Y) ∈ [36, 30, 20, 10]

theorem valid_addends :
  ∀ (n : ℕ),
  is_valid_addend n ↔ (n = 964 ∨ n = 974 ∨ n = 984 ∨ n = 994) :=
by
  sorry

end valid_addends_l266_266468


namespace probability_recurrence_relation_l266_266399

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l266_266399


namespace greatest_xy_value_l266_266391

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l266_266391


namespace profit_percent_approx_l266_266953

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 30
noncomputable def selling_price : ℝ := 300

noncomputable def cost_price : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit / cost_price) * 100

theorem profit_percent_approx :
  purchase_price = 225 ∧ 
  overhead_expenses = 30 ∧ 
  selling_price = 300 → 
  abs (profit_percent - 17.65) < 0.01 := 
by 
  -- Proof omitted
  sorry

end profit_percent_approx_l266_266953


namespace celebration_women_count_l266_266651

theorem celebration_women_count (num_men : ℕ) (num_pairs : ℕ) (pairs_per_man : ℕ) (pairs_per_woman : ℕ) 
  (hm : num_men = 15) (hpm : pairs_per_man = 4) (hwp : pairs_per_woman = 3) (total_pairs : num_pairs = num_men * pairs_per_man) : 
  num_pairs / pairs_per_woman = 20 :=
by
  sorry

end celebration_women_count_l266_266651


namespace gcd_of_18_and_30_l266_266245

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266245


namespace problem_statement_l266_266682

noncomputable def a : ℝ := 13 / 2
noncomputable def b : ℝ := -4

theorem problem_statement :
  ∀ k : ℝ, ∃ x : ℝ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 ↔ x = 1 :=
by
  sorry

end problem_statement_l266_266682


namespace num_readers_sci_fiction_l266_266548

theorem num_readers_sci_fiction (T L B S: ℕ) (hT: T = 250) (hL: L = 88) (hB: B = 18) (hTotal: T = S + L - B) : 
  S = 180 := 
by 
  sorry

end num_readers_sci_fiction_l266_266548


namespace new_volume_proof_l266_266764

variable (r h : ℝ)
variable (π : ℝ := Real.pi) -- Lean's notation for π
variable (original_volume : ℝ := 15) -- given original volume

-- Define original volume of the cylinder
def V := π * r^2 * h

-- Define new volume of the cylinder using new dimensions
def new_V := π * (3 * r)^2 * (2 * h)

-- Prove that new_V is 270 when V = 15
theorem new_volume_proof (hV : V = 15) : new_V = 270 :=
by
  -- Proof will go here
  sorry

end new_volume_proof_l266_266764


namespace modular_inverse_of_2_mod_199_l266_266117

theorem modular_inverse_of_2_mod_199 : (2 * 100) % 199 = 1 := 
by sorry

end modular_inverse_of_2_mod_199_l266_266117


namespace find_y_l266_266736

theorem find_y (y : ℝ) (h : (15 + 25 + y) / 3 = 23) : y = 29 :=
sorry

end find_y_l266_266736


namespace no_solution_exists_l266_266118

theorem no_solution_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x ^ 2 + f y) = 2 * x - f y :=
by
  sorry

end no_solution_exists_l266_266118


namespace minimum_distance_to_recover_cost_l266_266633

theorem minimum_distance_to_recover_cost 
  (initial_consumption : ℝ) (modification_cost : ℝ) (modified_consumption : ℝ) (gas_cost : ℝ) : 
  22000 < (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 ∧ 
  (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 < 26000 :=
by
  let initial_consumption := 8.4
  let modified_consumption := 6.3
  let modification_cost := 400.0
  let gas_cost := 0.80
  sorry

end minimum_distance_to_recover_cost_l266_266633


namespace jimin_has_most_candy_left_l266_266872

-- Definitions based on conditions
def fraction_jimin_ate := 1 / 9
def fraction_taehyung_ate := 1 / 3
def fraction_hoseok_ate := 1 / 6

-- The goal to prove
theorem jimin_has_most_candy_left : 
  (1 - fraction_jimin_ate) > (1 - fraction_taehyung_ate) ∧ (1 - fraction_jimin_ate) > (1 - fraction_hoseok_ate) :=
by
  -- The actual proof steps are omitted here.
  sorry

end jimin_has_most_candy_left_l266_266872


namespace sequence_solution_existence_l266_266624

noncomputable def sequence_exists : Prop :=
  ∃ s : Fin 20 → ℝ,
    (∀ i : Fin 18, s i + s (i+1) + s (i+2) > 0) ∧
    (Finset.univ.sum (λ i : Fin 20, s i) < 0)

theorem sequence_solution_existence : sequence_exists :=
  sorry

end sequence_solution_existence_l266_266624


namespace gcd_18_30_l266_266299

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266299


namespace sufficient_not_necessary_l266_266481

theorem sufficient_not_necessary (x : ℝ) : (x < 1 → x < 2) ∧ (¬(x < 2 → x < 1)) :=
by
  sorry

end sufficient_not_necessary_l266_266481


namespace gcd_18_30_l266_266300

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266300


namespace find_sum_l266_266833

variable (a b c d : ℝ)

theorem find_sum :
  (ab + bc + cd + da = 20) →
  (b + d = 4) →
  (a + c = 5) := by
  sorry

end find_sum_l266_266833


namespace probZ_eq_1_4_l266_266223

noncomputable def probX : ℚ := 1/4
noncomputable def probY : ℚ := 1/3
noncomputable def probW : ℚ := 1/6

theorem probZ_eq_1_4 :
  let probZ : ℚ := 1 - (probX + probY + probW)
  probZ = 1/4 :=
by
  sorry

end probZ_eq_1_4_l266_266223


namespace segment_AC_length_l266_266035

-- Define segments AB and BC
def AB : ℝ := 4
def BC : ℝ := 3

-- Define segment AC in terms of the conditions given
def AC_case1 : ℝ := AB - BC
def AC_case2 : ℝ := AB + BC

-- The proof problem statement
theorem segment_AC_length : AC_case1 = 1 ∨ AC_case2 = 7 := by
  sorry

end segment_AC_length_l266_266035


namespace complement_of_M_l266_266721

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 2*x > 0 }
def complement (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

theorem complement_of_M :
  complement U M = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_of_M_l266_266721


namespace gcd_18_30_is_6_l266_266266

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266266


namespace gcd_18_30_l266_266255

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266255


namespace blue_pigment_percentage_l266_266486

-- Define weights and pigments in the problem
variables (S G : ℝ)
-- Conditions
def sky_blue_paint := 0.9 * S = 4.5
def total_weight := S + G = 10
def sky_blue_blue_pigment := 0.1
def green_blue_pigment := 0.7

-- Prove the percentage of blue pigment in brown paint is 40%
theorem blue_pigment_percentage :
  sky_blue_paint S →
  total_weight S G →
  (0.1 * (4.5 / 0.9) + 0.7 * (10 - (4.5 / 0.9))) / 10 * 100 = 40 :=
by
  intros h1 h2
  sorry

end blue_pigment_percentage_l266_266486


namespace total_pencils_in_drawer_l266_266456

-- Definitions based on conditions from the problem
def initial_pencils : ℕ := 138
def pencils_by_Nancy : ℕ := 256
def pencils_by_Steven : ℕ := 97

-- The theorem proving the total number of pencils in the drawer
theorem total_pencils_in_drawer : initial_pencils + pencils_by_Nancy + pencils_by_Steven = 491 :=
by
  -- This statement is equivalent to the mathematical problem given
  sorry

end total_pencils_in_drawer_l266_266456


namespace smallest_sum_of_squares_l266_266742

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l266_266742


namespace find_angle_D_l266_266946

theorem find_angle_D 
  (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50) :
  D = 25 := 
by
  sorry

end find_angle_D_l266_266946


namespace daniela_total_spent_l266_266818

-- Step d) Rewrite the math proof problem
theorem daniela_total_spent
    (shoe_price : ℤ) (dress_price : ℤ) (shoe_discount : ℤ) (dress_discount : ℤ)
    (shoe_count : ℤ)
    (shoe_original_price : shoe_price = 50)
    (dress_original_price : dress_price = 100)
    (shoe_discount_rate : shoe_discount = 40)
    (dress_discount_rate : dress_discount = 20)
    (shoe_total_count : shoe_count = 2)
    : shoe_count * (shoe_price - (shoe_price * shoe_discount / 100)) + (dress_price - (dress_price * dress_discount / 100)) = 140 := by 
    sorry

end daniela_total_spent_l266_266818


namespace collinear_points_x_value_l266_266087

theorem collinear_points_x_value :
  (∀ A B C : ℝ × ℝ, A = (-1, 1) → B = (2, -4) → C = (x, -9) → 
                    (∃ x : ℝ, x = 5)) :=
by sorry

end collinear_points_x_value_l266_266087


namespace statement_C_is_incorrect_l266_266023

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem statement_C_is_incorrect : g (-2) ≠ 0 :=
by
  sorry

end statement_C_is_incorrect_l266_266023


namespace slope_of_line_l266_266926

theorem slope_of_line : ∀ x y : ℝ, 3 * y + 2 * x = 6 * x - 9 → ∃ m b : ℝ, y = m * x + b ∧ m = -4 / 3 :=
by
  -- Sorry to skip proof
  sorry

end slope_of_line_l266_266926


namespace necessary_and_sufficient_condition_l266_266132

theorem necessary_and_sufficient_condition (x : ℝ) (h : x > 0) : (x + 1/x ≥ 2) ↔ (x > 0) :=
sorry

end necessary_and_sufficient_condition_l266_266132


namespace number_of_integer_pairs_l266_266677

theorem number_of_integer_pairs (n : ℕ) : 
  ∃ (count : ℕ), count = 2 * n^2 + 2 * n + 1 ∧ 
  ∀ x y : ℤ, abs x + abs y ≤ n ↔
  count = 2 * n^2 + 2 * n + 1 :=
by
  sorry

end number_of_integer_pairs_l266_266677


namespace m_and_n_relationship_l266_266146

-- Define the function f
def f (x m : ℝ) := x^2 - 4*x + 4 + m

-- State the conditions and required proof
theorem m_and_n_relationship (m n : ℝ) (h_domain : ∀ x, 2 ≤ x ∧ x ≤ n → 2 ≤ f x m ∧ f x m ≤ n) :
  m^n = 8 :=
by
  -- Placeholder for the actual proof
  sorry

end m_and_n_relationship_l266_266146


namespace solve_for_x_l266_266537

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 7 * x = 140) : x = 28 := by
  sorry

end solve_for_x_l266_266537


namespace base_seven_sum_of_product_l266_266228

def base_seven_to_decimal (d1 d0 : ℕ) : ℕ :=
  7 * d1 + d0

def decimal_to_base_seven (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d3 := n / (7 ^ 3)
  let r3 := n % (7 ^ 3)
  let d2 := r3 / (7 ^ 2)
  let r2 := r3 % (7 ^ 2)
  let d1 := r2 / 7
  let d0 := r2 % 7
  (d3, d2, d1, d0)

def sum_of_base_seven_digits (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 + d2 + d1 + d0

theorem base_seven_sum_of_product :
  let n1 := base_seven_to_decimal 3 5
  let n2 := base_seven_to_decimal 4 2
  let product := n1 * n2
  let (d3, d2, d1, d0) := decimal_to_base_seven product
  sum_of_base_seven_digits d3 d2 d1 d0 = 18 :=
  by
    sorry

end base_seven_sum_of_product_l266_266228


namespace kim_trip_time_l266_266051

-- Definitions
def distance_freeway : ℝ := 120
def distance_mountain : ℝ := 25
def speed_ratio : ℝ := 4
def time_mountain : ℝ := 75

-- The problem statement
theorem kim_trip_time : ∃ t_freeway t_total : ℝ,
  t_freeway = distance_freeway / (speed_ratio * (distance_mountain / time_mountain)) ∧
  t_total = time_mountain + t_freeway ∧
  t_total = 165 := by
  sorry

end kim_trip_time_l266_266051


namespace reciprocal_of_2_l266_266914

theorem reciprocal_of_2 : 1 / 2 = 1 / (2 : ℝ) := by
  sorry

end reciprocal_of_2_l266_266914


namespace gcd_18_30_is_6_l266_266272

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266272


namespace rearrange_raven_no_consecutive_vowels_l266_266142

theorem rearrange_raven_no_consecutive_vowels :
  let letters := ["R", "A", "V", "E", "N"]
  let vowels := ["A", "E"]
  let consonants := ["R", "V", "N"]
  (letters.permutations.length - (consonants.permutations.length * 2)) = 72 :=
by
  sorry

end rearrange_raven_no_consecutive_vowels_l266_266142


namespace clothes_in_total_l266_266578

-- Define the conditions as constants since they are fixed values
def piecesInOneLoad : Nat := 17
def numberOfSmallLoads : Nat := 5
def piecesPerSmallLoad : Nat := 6

-- Noncomputable for definition involving calculation
noncomputable def totalClothes : Nat :=
  piecesInOneLoad + (numberOfSmallLoads * piecesPerSmallLoad)

-- The theorem to prove Luke had 47 pieces of clothing in total
theorem clothes_in_total : totalClothes = 47 := by
  sorry

end clothes_in_total_l266_266578


namespace value_of_f_neg_5π_over_12_l266_266352

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem value_of_f_neg_5π_over_12 :
  ∀ (ω φ : ℝ), 
    (∀ x y : ℝ, (x < y ∧ x ∈ Ioo (π / 6) (2 * π / 3) ∧ y ∈ Ioo (π / 6) (2 * π / 3)) → f x ω φ < f y ω φ) ∧ 
    f (π / 6) ω φ = f (2 * π / 3) ω φ → 
    f (-5 * π / 12) ω φ = √3 / 2 :=
by
  sorry

end value_of_f_neg_5π_over_12_l266_266352


namespace employee_discount_percentage_l266_266952

theorem employee_discount_percentage (wholesale_cost retail_price employee_price discount_percentage : ℝ) 
  (h1 : wholesale_cost = 200)
  (h2 : retail_price = wholesale_cost * 1.2)
  (h3 : employee_price = 204)
  (h4 : discount_percentage = ((retail_price - employee_price) / retail_price) * 100) :
  discount_percentage = 15 :=
by
  sorry

end employee_discount_percentage_l266_266952


namespace find_a_l266_266997

def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_a (a : ℝ) : 
  are_perpendicular a (a + 2) → 
  a = -1 :=
by
  intro h
  unfold are_perpendicular at h
  have h_eq : a * (a + 2) = -1 := h
  have eq_zero : a * a + 2 * a + 1 = 0 := by linarith
  sorry

end find_a_l266_266997


namespace product_eval_at_3_l266_266825

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l266_266825


namespace run_to_grocery_store_time_l266_266534

theorem run_to_grocery_store_time
  (running_time: ℝ)
  (grocery_distance: ℝ)
  (friend_distance: ℝ)
  (half_way : friend_distance = grocery_distance / 2)
  (constant_pace : running_time / grocery_distance = (25 : ℝ) / 3)
  : (friend_distance * (25 / 3)) + (friend_distance * (25 / 3)) = 25 :=
by
  -- Given proofs for the conditions can be filled here
  sorry

end run_to_grocery_store_time_l266_266534


namespace weight_of_milk_l266_266173

def max_bag_capacity : ℕ := 20
def green_beans : ℕ := 4
def carrots : ℕ := 2 * green_beans
def fit_more : ℕ := 2
def current_weight : ℕ := max_bag_capacity - fit_more
def total_weight_of_green_beans_and_carrots : ℕ := green_beans + carrots

theorem weight_of_milk : (current_weight - total_weight_of_green_beans_and_carrots) = 6 := by
  -- Proof to be written here
  sorry

end weight_of_milk_l266_266173


namespace percent_enclosed_by_hexagons_l266_266911

variable (b : ℝ) -- side length of smaller squares

def area_of_small_square : ℝ := b^2
def area_of_large_square : ℝ := 16 * area_of_small_square b
def area_of_hexagon : ℝ := 3 * area_of_small_square b
def total_area_of_hexagons : ℝ := 2 * area_of_hexagon b

theorem percent_enclosed_by_hexagons :
  (total_area_of_hexagons b / area_of_large_square b) * 100 = 37.5 :=
by
  -- Proof omitted
  sorry

end percent_enclosed_by_hexagons_l266_266911


namespace marian_balance_proof_l266_266588

noncomputable def marian_new_balance : ℝ :=
  let initial_balance := 126.00
  let uk_purchase := 50.0
  let uk_discount := 0.10
  let uk_rate := 1.39
  let france_purchase := 70.0
  let france_discount := 0.15
  let france_rate := 1.18
  let japan_purchase := 10000.0
  let japan_discount := 0.05
  let japan_rate := 0.0091
  let towel_return := 45.0
  let interest_rate := 0.015
  let uk_usd := (uk_purchase * (1 - uk_discount)) * uk_rate
  let france_usd := (france_purchase * (1 - france_discount)) * france_rate
  let japan_usd := (japan_purchase * (1 - japan_discount)) * japan_rate
  let gas_usd := (uk_purchase / 2) * uk_rate
  let balance_before_interest := initial_balance + uk_usd + france_usd + japan_usd + gas_usd - towel_return
  let interest := balance_before_interest * interest_rate
  balance_before_interest + interest

theorem marian_balance_proof :
  abs (marian_new_balance - 340.00) < 1 :=
by
  sorry

end marian_balance_proof_l266_266588


namespace total_bales_stored_l266_266462

theorem total_bales_stored 
  (initial_bales : ℕ := 540) 
  (new_bales : ℕ := 2) : 
  initial_bales + new_bales = 542 :=
by
  sorry

end total_bales_stored_l266_266462


namespace length_of_platform_is_correct_l266_266942

-- Given conditions:
def length_of_train : ℕ := 250
def speed_of_train_kmph : ℕ := 72
def time_to_cross_platform : ℕ := 20

-- Convert speed from kmph to m/s
def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600

-- Distance covered in 20 seconds
def distance_covered : ℕ := speed_of_train_mps * time_to_cross_platform

-- Length of the platform
def length_of_platform : ℕ := distance_covered - length_of_train

-- The proof statement
theorem length_of_platform_is_correct :
  length_of_platform = 150 := by
  -- This proof would involve the detailed calculations and verifications as laid out in the solution steps.
  sorry

end length_of_platform_is_correct_l266_266942


namespace volume_of_convex_polyhedron_l266_266675

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron_l266_266675


namespace part_a_part_b_l266_266595

variable {A B C A₁ B₁ C₁ : Prop}
variables {a b c a₁ b₁ c₁ S S₁ : ℝ}

-- Assume basic conditions of triangles
variable (h1 : IsTriangle A B C)
variable (h2 : IsTriangleWithCentersAndSquares A B C A₁ B₁ C₁ a b c a₁ b₁ c₁ S S₁)
variable (h3 : IsExternalSquaresConstructed A B C A₁ B₁ C₁)

-- Part (a)
theorem part_a : a₁^2 + b₁^2 + c₁^2 = a^2 + b^2 + c^2 + 6 * S := 
sorry

-- Part (b)
theorem part_b : S₁ - S = (a^2 + b^2 + c^2) / 8 := 
sorry

end part_a_part_b_l266_266595


namespace smallest_sum_of_squares_l266_266753

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l266_266753


namespace purely_imaginary_a_eq_1_fourth_quadrant_a_range_l266_266991

-- Definitions based on given conditions
def z (a : ℝ) := (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I

-- Purely imaginary proof statement
theorem purely_imaginary_a_eq_1 (a : ℝ) 
  (hz : (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I = (0 : ℂ) + (a^2 - 5 * a - 6) * Complex.I) :
  a = 1 := by 
  sorry

-- Fourth quadrant proof statement
theorem fourth_quadrant_a_range (a : ℝ) 
  (hz1 : a^2 - 7 * a + 6 > 0) 
  (hz2 : a^2 - 5 * a - 6 < 0) : 
  -1 < a ∧ a < 1 := by 
  sorry

end purely_imaginary_a_eq_1_fourth_quadrant_a_range_l266_266991


namespace base_9_perfect_square_b_l266_266026

theorem base_9_perfect_square_b (b : ℕ) (a : ℕ) 
  (h0 : 0 < b) (h1 : b < 9) (h2 : a < 9) : 
  ∃ n, n^2 ≡ 729 * b + 81 * a + 54 [MOD 81] :=
sorry

end base_9_perfect_square_b_l266_266026


namespace expand_product_l266_266662

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9 * x + 18 := 
by sorry

end expand_product_l266_266662


namespace correct_operation_l266_266779

theorem correct_operation (x : ℝ) : (-x^3)^2 = x^6 :=
by sorry

end correct_operation_l266_266779


namespace ratio_chest_of_drawers_to_treadmill_l266_266614

theorem ratio_chest_of_drawers_to_treadmill :
  ∀ (C T TV : ℕ),
  T = 100 →
  TV = 3 * 100 →
  100 + C + TV = 600 →
  C / T = 2 :=
by
  intros C T TV ht htv heq
  sorry

end ratio_chest_of_drawers_to_treadmill_l266_266614


namespace distance_traveled_l266_266206

-- Define constants for speed and time
def speed : ℝ := 60
def time : ℝ := 5

-- Define the expected distance
def expected_distance : ℝ := 300

-- Theorem statement
theorem distance_traveled : speed * time = expected_distance :=
by
  sorry

end distance_traveled_l266_266206


namespace second_worker_time_l266_266957

theorem second_worker_time 
  (first_worker_rate : ℝ)
  (combined_rate : ℝ)
  (x : ℝ)
  (h1 : first_worker_rate = 1 / 6)
  (h2 : combined_rate = 1 / 2.4) :
  (1 / 6) + (1 / x) = combined_rate → x = 4 := 
by 
  intros h
  sorry

end second_worker_time_l266_266957


namespace project_completion_days_l266_266939

-- Define the work rates and the total number of days to complete the project
variables (a_rate b_rate : ℝ) (days_to_complete : ℝ)
variable (a_quit_before_completion : ℝ)

-- Define the conditions
def A_rate := 1 / 20
def B_rate := 1 / 20
def quit_before_completion := 10 

-- The total work done in the project as 1 project 
def total_work := 1

-- Define the equation representing the amount of work done by A and B
def total_days := 
  A_rate * (days_to_complete - a_quit_before_completion) + B_rate * days_to_complete

-- The theorem statement
theorem project_completion_days :
  A_rate = a_rate → 
  B_rate = b_rate → 
  quit_before_completion = a_quit_before_completion → 
  total_days = total_work → 
  days_to_complete = 15 :=
by 
  -- placeholders for the conditions
  intros h1 h2 h3 h4
  sorry

end project_completion_days_l266_266939


namespace nathan_and_parents_total_cost_l266_266730

/-- Define the total number of people -/
def num_people := 3

/-- Define the cost per object -/
def cost_per_object := 11

/-- Define the number of objects per person -/
def objects_per_person := 2 + 2 + 1

/-- Define the total number of objects -/
def total_objects := num_people * objects_per_person

/-- Define the total cost -/
def total_cost := total_objects * cost_per_object

/-- The main theorem to prove the total cost -/
theorem nathan_and_parents_total_cost : total_cost = 165 := by
  sorry

end nathan_and_parents_total_cost_l266_266730


namespace evaluate_fraction_l266_266971

theorem evaluate_fraction : (1 - 1/4) / (1 - 1/3) = 9/8 :=
by
  sorry

end evaluate_fraction_l266_266971


namespace find_integers_divisible_by_18_in_range_l266_266511

theorem find_integers_divisible_by_18_in_range :
  ∃ n : ℕ, (n % 18 = 0) ∧ (n ≥ 900) ∧ (n ≤ 930) ∧ (n = 900 ∨ n = 918) :=
sorry

end find_integers_divisible_by_18_in_range_l266_266511


namespace smallest_natural_number_with_condition_l266_266122

theorem smallest_natural_number_with_condition {N : ℕ} :
  (N % 10 = 6) ∧ (4 * N = (6 * 10 ^ ((Nat.digits 10 (N / 10)).length) + (N / 10))) ↔ N = 153846 :=
by
  sorry

end smallest_natural_number_with_condition_l266_266122


namespace probability_recurrence_relation_l266_266400

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l266_266400


namespace gcd_of_18_and_30_l266_266287

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266287


namespace general_term_formula_l266_266687

theorem general_term_formula (f : ℕ → ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ x, f x = 1 - 2^x) →
  (∀ n, f n = S n) →
  (∀ n, S n = 1 - 2^n) →
  (∀ n, n = 1 → a n = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = -2^(n-1)) :=
by
  sorry

end general_term_formula_l266_266687


namespace quilt_block_shading_fraction_l266_266231

theorem quilt_block_shading_fraction :
  (fraction_shaded : ℚ) → 
  (quilt_block_size : ℕ) → 
  (fully_shaded_squares : ℕ) → 
  (half_shaded_squares : ℕ) → 
  quilt_block_size = 16 →
  fully_shaded_squares = 6 →
  half_shaded_squares = 4 →
  fraction_shaded = 1/2 :=
by 
  sorry

end quilt_block_shading_fraction_l266_266231


namespace fiona_shirt_number_l266_266078

def is_two_digit_prime (n : ℕ) : Prop := 
  (n ≥ 10 ∧ n < 100 ∧ Nat.Prime n)

theorem fiona_shirt_number (d e f : ℕ) 
  (h1 : is_two_digit_prime d)
  (h2 : is_two_digit_prime e)
  (h3 : is_two_digit_prime f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) : 
  f = 19 := 
sorry

end fiona_shirt_number_l266_266078


namespace sum_first_8_terms_64_l266_266990

-- Define the problem conditions
def isArithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def isGeometricSeq (a : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → n < k → (a n)^2 = a m * a k

-- Given arithmetic sequence with a common difference 2
def arithmeticSeqWithDiff2 (a : ℕ → ℤ) : Prop :=
  isArithmeticSeq a ∧ (∃ d : ℤ, d = 2 ∧ ∀ (n : ℕ), a (n + 1) = a n + d)

-- Given a₁, a₂, a₅ form a geometric sequence
def a1_a2_a5_formGeometricSeq (a: ℕ → ℤ) : Prop :=
  (a 2)^2 = (a 1) * (a 5)

-- Sum of the first 8 terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a (n - 1)) / 2

-- Main statement
theorem sum_first_8_terms_64 (a : ℕ → ℤ) (h1 : arithmeticSeqWithDiff2 a) (h2 : a1_a2_a5_formGeometricSeq a) : 
  sum_of_first_n_terms a 8 = 64 := 
sorry

end sum_first_8_terms_64_l266_266990


namespace maria_total_cost_l266_266582

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end maria_total_cost_l266_266582


namespace probability_two_white_balls_l266_266959

def bagA := [1, 1]
def bagB := [2, 1]

def total_outcomes := 6
def favorable_outcomes := 2

theorem probability_two_white_balls : (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by
  sorry

end probability_two_white_balls_l266_266959


namespace total_fruit_count_l266_266052

theorem total_fruit_count :
  let gerald_apple_bags := 5
  let gerald_orange_bags := 4
  let apples_per_gerald_bag := 30
  let oranges_per_gerald_bag := 25
  let pam_apple_bags := 6
  let pam_orange_bags := 4
  let sue_apple_bags := 2 * gerald_apple_bags
  let sue_orange_bags := gerald_orange_bags / 2
  let apples_per_sue_bag := apples_per_gerald_bag - 10
  let oranges_per_sue_bag := oranges_per_gerald_bag + 5
  
  let gerald_apples := gerald_apple_bags * apples_per_gerald_bag
  let gerald_oranges := gerald_orange_bags * oranges_per_gerald_bag
  
  let pam_apples := pam_apple_bags * (3 * apples_per_gerald_bag)
  let pam_oranges := pam_orange_bags * (2 * oranges_per_gerald_bag)
  
  let sue_apples := sue_apple_bags * apples_per_sue_bag
  let sue_oranges := sue_orange_bags * oranges_per_sue_bag

  let total_apples := gerald_apples + pam_apples + sue_apples
  let total_oranges := gerald_oranges + pam_oranges + sue_oranges
  total_apples + total_oranges = 1250 :=

by
  sorry

end total_fruit_count_l266_266052


namespace greatest_xy_l266_266393

theorem greatest_xy (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 7 * x + 4 * y = 140) : xy ≤ 168 :=
begin
  sorry
end

example : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ xy = 168 :=
begin
  use [8, 21],
  split, exact dec_trivial,
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end greatest_xy_l266_266393


namespace Tina_independent_work_hours_l266_266623

-- Defining conditions as Lean constants
def Tina_work_rate := 1 / 12
def Ann_work_rate := 1 / 9
def Ann_work_hours := 3

-- Declaring the theorem to be proven
theorem Tina_independent_work_hours : 
  (Ann_work_hours * Ann_work_rate = 1/3) →
  ((1 : ℚ) - (Ann_work_hours * Ann_work_rate)) / Tina_work_rate = 8 :=
by {
  sorry
}

end Tina_independent_work_hours_l266_266623


namespace num_small_boxes_l266_266637

-- Conditions
def chocolates_per_small_box := 25
def total_chocolates := 400

-- Claim: Prove that the number of small boxes is 16
theorem num_small_boxes : (total_chocolates / chocolates_per_small_box) = 16 := 
by sorry

end num_small_boxes_l266_266637


namespace new_average_after_increase_and_bonus_l266_266603

theorem new_average_after_increase_and_bonus 
  (n : ℕ) (initial_avg : ℝ) (k : ℝ) (bonus : ℝ) 
  (h1: n = 37) 
  (h2: initial_avg = 73) 
  (h3: k = 1.65) 
  (h4: bonus = 15) 
  : (initial_avg * k) + bonus = 135.45 := 
sorry

end new_average_after_increase_and_bonus_l266_266603


namespace smallest_lucky_number_theorem_specific_lucky_number_theorem_l266_266701

-- Definitions based on the given conditions
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), (M = A * B) ∧
               (A ≥ B) ∧
               (A ≥ 10 ∧ A ≤ 99) ∧
               (B ≥ 10 ∧ B ≤ 99) ∧
               (A / 10 = B / 10) ∧
               (A % 10 + B % 10 = 6)

def smallest_lucky_number : ℕ :=
  165

def P (M A B : ℕ) := A + B
def Q (M A B : ℕ) := A - B

def specific_lucky_number (M A B : ℕ) : Prop :=
  M = A * B ∧ (P M A B) / (Q M A B) % 7 = 0

-- Theorems to prove
theorem smallest_lucky_number_theorem :
  ∃ M, is_lucky_number M ∧ M = smallest_lucky_number := by
  sorry

theorem specific_lucky_number_theorem :
  ∃ M A B, is_lucky_number M ∧ specific_lucky_number M A B ∧ M = 3968 := by
  sorry

end smallest_lucky_number_theorem_specific_lucky_number_theorem_l266_266701


namespace gcd_18_30_l266_266311

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266311


namespace binom_15_12_eq_455_l266_266107

theorem binom_15_12_eq_455 : Nat.choose 15 12 = 455 := 
by sorry

end binom_15_12_eq_455_l266_266107


namespace area_of_garden_l266_266715

theorem area_of_garden :
  ∃ (short_posts long_posts : ℕ), short_posts + long_posts - 4 = 24 → long_posts = 3 * short_posts →
  ∃ (short_length long_length : ℕ), short_length = (short_posts - 1) * 5 → long_length = (long_posts - 1) * 5 →
  (short_length * long_length = 3000) :=
by {
  sorry
}

end area_of_garden_l266_266715


namespace part1_part2_l266_266546

open Real

variable (A B C a b c : ℝ)

-- Conditions
variable (h1 : b * sin A = a * cos B)
variable (h2 : b = 3)
variable (h3 : sin C = 2 * sin A)

theorem part1 : B = π / 4 := 
  sorry

theorem part2 : ∃ a c, c = 2 * a ∧ 9 = a^2 + c^2 - 2 * a * c * cos (π / 4) := 
  sorry

end part1_part2_l266_266546


namespace nell_more_ace_cards_than_baseball_l266_266050

-- Definitions based on conditions
def original_baseball_cards : ℕ := 239
def original_ace_cards : ℕ := 38
def current_ace_cards : ℕ := 376
def current_baseball_cards : ℕ := 111

-- The statement we need to prove
theorem nell_more_ace_cards_than_baseball :
  current_ace_cards - current_baseball_cards = 265 :=
by
  -- Add the proof here
  sorry

end nell_more_ace_cards_than_baseball_l266_266050


namespace find_k_n_l266_266658

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l266_266658


namespace arithmetic_geometric_sequences_l266_266988

noncomputable def geometric_sequence_sum (a q n : ℝ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequences (a : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  S 5 = geometric_sequence_sum a q 5 →
  2 * a * q = 6 + a * q^4 →
  S 5 = -31 / 2 :=
by
  intros hq1 hS5 hAR
  sorry

end arithmetic_geometric_sequences_l266_266988


namespace parallel_lines_l266_266692

theorem parallel_lines (m : ℝ) 
  (h : 3 * (m - 2) + m * (m + 2) = 0) 
  : m = 1 ∨ m = -6 := 
by 
  sorry

end parallel_lines_l266_266692


namespace simplify_fraction_l266_266539

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) :
  (x - 1/y) / (y - 1/x) = x / y :=
sorry

end simplify_fraction_l266_266539


namespace cos_alpha_plus_5pi_over_4_eq_16_over_65_l266_266674

theorem cos_alpha_plus_5pi_over_4_eq_16_over_65
  (α β : ℝ)
  (hα : -π / 4 < α ∧ α < 0)
  (hβ : π / 2 < β ∧ β < π)
  (hcos_sum : Real.cos (α + β) = -4/5)
  (hcos_diff : Real.cos (β - π / 4) = 5/13) :
  Real.cos (α + 5 * π / 4) = 16/65 :=
by
  sorry

end cos_alpha_plus_5pi_over_4_eq_16_over_65_l266_266674


namespace transformed_curve_is_circle_l266_266133

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * cos θ^2 + 4 * sin θ^2)

def cartesian_curve (x y: ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 2 ∧ y' = y * sqrt (3 / 3)

theorem transformed_curve_is_circle (x y x' y' : ℝ) 
  (h1: cartesian_curve x y) (h2: transformation x y x' y') : 
  (x'^2 + y'^2 = 1) :=
sorry

end transformed_curve_is_circle_l266_266133


namespace no_counterexample_exists_l266_266168

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_exists : ∀ n : ℕ, sum_of_digits n % 9 = 0 → n % 9 = 0 :=
by
  intro n h
  sorry

end no_counterexample_exists_l266_266168


namespace required_speed_remaining_l266_266091

theorem required_speed_remaining (total_distance : ℕ) (total_time : ℕ) (initial_speed : ℕ) (initial_time : ℕ) 
  (h1 : total_distance = 24) (h2 : total_time = 8) (h3 : initial_speed = 4) (h4 : initial_time = 4) :
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

end required_speed_remaining_l266_266091


namespace gcd_18_30_l266_266312

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266312


namespace gcd_18_30_l266_266309

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266309


namespace meet_time_approx_l266_266945

noncomputable def length_of_track : ℝ := 1800 -- in meters
noncomputable def speed_first_woman : ℝ := 10 * 1000 / 3600 -- in meters per second
noncomputable def speed_second_woman : ℝ := 20 * 1000 / 3600 -- in meters per second
noncomputable def relative_speed : ℝ := speed_first_woman + speed_second_woman

theorem meet_time_approx (ε : ℝ) (hε : ε = 216.048) :
  ∃ t : ℝ, t = length_of_track / relative_speed ∧ abs (t - ε) < 0.001 :=
by
  sorry

end meet_time_approx_l266_266945


namespace min_gx1_gx2_l266_266013

noncomputable def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x - (a / 2) * Real.log x

theorem min_gx1_gx2 (x1 x2 a : ℝ) (h1 : 0 < x1 ∧ x1 < Real.exp 1) (h2 : 0 < x2) (hx1x2: x1 * x2 = 1) (ha : a > 0) :
  f x1 a = 0 ∧ f x2 a = 0 →
  g x1 a - g x2 a = -2 / Real.exp 1 :=
by sorry

end min_gx1_gx2_l266_266013


namespace range_S₁₂_div_d_l266_266336

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem range_S₁₂_div_d (a₁ d : α) (h_a₁_pos : a₁ > 0) (h_d_neg : d < 0) 
  (h_max_S_8 : ∀ n, arithmetic_sequence_sum a₁ d n ≤ arithmetic_sequence_sum a₁ d 8) :
  -30 < (arithmetic_sequence_sum a₁ d 12) / d ∧ (arithmetic_sequence_sum a₁ d 12) / d < -18 :=
by
  have h1 : -8 < a₁ / d := by sorry
  have h2 : a₁ / d < -7 := by sorry
  have h3 : (arithmetic_sequence_sum a₁ d 12) / d = 12 * (a₁ / d) + 66 := by sorry
  sorry

end range_S₁₂_div_d_l266_266336


namespace shift_line_one_unit_left_l266_266156

theorem shift_line_one_unit_left : ∀ (x y : ℝ), (y = x) → (y - 1 = (x + 1) - 1) :=
by
  intros x y h
  sorry

end shift_line_one_unit_left_l266_266156


namespace max_b_plus_c_triangle_l266_266037

theorem max_b_plus_c_triangle (a b c : ℝ) (A : ℝ) 
  (h₁ : a = 4) (h₂ : A = Real.pi / 3) (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b + c ≤ 8 :=
by
  -- sorry is added to skip the proof for now.
  sorry

end max_b_plus_c_triangle_l266_266037


namespace smallest_integer_CC6_DD8_l266_266932

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end smallest_integer_CC6_DD8_l266_266932


namespace ab_eq_zero_l266_266453

theorem ab_eq_zero (a b : ℤ) (h : ∀ m n : ℕ, ∃ k : ℤ, a * (m^2 : ℤ) + b * (n^2 : ℤ) = k^2) : a * b = 0 :=
by
  sorry

end ab_eq_zero_l266_266453


namespace gcd_of_18_and_30_l266_266324

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266324


namespace madeline_needs_work_hours_l266_266046

def rent : ℝ := 1200
def groceries : ℝ := 400
def medical_expenses : ℝ := 200
def utilities : ℝ := 60
def emergency_savings : ℝ := 200
def hourly_wage : ℝ := 15

def total_expenses : ℝ := rent + groceries + medical_expenses + utilities + emergency_savings

noncomputable def total_hours_needed : ℝ := total_expenses / hourly_wage

theorem madeline_needs_work_hours :
  ⌈total_hours_needed⌉ = 138 := by
  sorry

end madeline_needs_work_hours_l266_266046


namespace area_of_region_l266_266948

noncomputable def area : ℝ :=
  ∫ x in Set.Icc (-2 : ℝ) 0, (2 - (x + 1)^2 / 4) +
  ∫ x in Set.Icc (0 : ℝ) 2, (2 - x - (x + 1)^2 / 4)

theorem area_of_region : area = 5 / 3 := 
sorry

end area_of_region_l266_266948


namespace find_k_for_solutions_l266_266667

theorem find_k_for_solutions (k : ℝ) :
  (∀ x: ℝ, x = 3 ∨ x = 5 → k * x^2 - 8 * x + 15 = 0) → k = 1 :=
by
  sorry

end find_k_for_solutions_l266_266667


namespace white_balls_in_bag_l266_266949

open BigOperators

theorem white_balls_in_bag (N : ℕ) (N_green : ℕ) (N_yellow : ℕ) (N_red : ℕ) (N_purple : ℕ)
  (prob_not_red_nor_purple : ℝ) (W : ℕ)
  (hN : N = 100)
  (hN_green : N_green = 30)
  (hN_yellow : N_yellow = 10)
  (hN_red : N_red = 47)
  (hN_purple : N_purple = 3)
  (h_prob_not_red_nor_purple : prob_not_red_nor_purple = 0.5) :
  W = 10 :=
sorry

end white_balls_in_bag_l266_266949


namespace sum_of_cubes_of_consecutive_even_integers_l266_266190

theorem sum_of_cubes_of_consecutive_even_integers (x : ℤ) (h : x^2 + (x+2)^2 + (x+4)^2 = 2960) :
  x^3 + (x + 2)^3 + (x + 4)^3 = 90117 :=
sorry

end sum_of_cubes_of_consecutive_even_integers_l266_266190


namespace dividend_calculation_l266_266923

theorem dividend_calculation :
  ∀ (divisor quotient remainder : ℝ), 
  divisor = 37.2 → 
  quotient = 14.61 → 
  remainder = 0.67 → 
  (divisor * quotient + remainder) = 544.042 :=
by
  intros divisor quotient remainder h_div h_qt h_rm
  sorry

end dividend_calculation_l266_266923


namespace intersect_complement_A_and_B_l266_266359

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersect_complement_A_and_B : (Set.compl A ∩ B) = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end intersect_complement_A_and_B_l266_266359


namespace find_rate_percent_l266_266776

-- Given conditions as definitions
def SI : ℕ := 128
def P : ℕ := 800
def T : ℕ := 4

-- Define the formula for Simple Interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Define the rate percent we need to prove
def rate_percent : ℕ := 4

-- The theorem statement we need to prove
theorem find_rate_percent (h1 : simple_interest P rate_percent T = SI) : rate_percent = 4 := 
by sorry

end find_rate_percent_l266_266776


namespace simplify_fraction_expression_l266_266783

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l266_266783


namespace speed_of_train_l266_266956

-- Conditions
def train_length : ℝ := 180
def total_length : ℝ := 195
def time_cross_bridge : ℝ := 30

-- Conversion factor for units (1 m/s = 3.6 km/hr)
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem speed_of_train : 
  (total_length - train_length) / time_cross_bridge * conversion_factor = 23.4 :=
sorry

end speed_of_train_l266_266956


namespace positive_integers_sum_reciprocal_l266_266074

theorem positive_integers_sum_reciprocal (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_sum : a + b + c = 2010) (h_recip : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c = 1/58) :
  (a = 1740 ∧ b = 180 ∧ c = 90) ∨ 
  (a = 1740 ∧ b = 90 ∧ c = 180) ∨ 
  (a = 180 ∧ b = 90 ∧ c = 1740) ∨ 
  (a = 180 ∧ b = 1740 ∧ c = 90) ∨ 
  (a = 90 ∧ b = 1740 ∧ c = 180) ∨ 
  (a = 90 ∧ b = 180 ∧ c = 1740) := 
sorry

end positive_integers_sum_reciprocal_l266_266074


namespace complex_quadrant_l266_266129

theorem complex_quadrant (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 1 / (1 - i)) : 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_quadrant_l266_266129


namespace interval_of_monotonic_increase_l266_266908

noncomputable def powerFunction (k n x : ℝ) : ℝ := k * x ^ n

variable {k n : ℝ}

theorem interval_of_monotonic_increase
    (h : ∃ k n : ℝ, powerFunction k n 4 = 2) :
    (∀ x y : ℝ, 0 < x ∧ x < y → powerFunction k n x < powerFunction k n y) ∨
    (∀ x y : ℝ, 0 ≤ x ∧ x < y → powerFunction k n x ≤ powerFunction k n y) := sorry

end interval_of_monotonic_increase_l266_266908


namespace ads_on_first_web_page_l266_266922

theorem ads_on_first_web_page 
  (A : ℕ)
  (second_page_ads : ℕ := 2 * A)
  (third_page_ads : ℕ := 2 * A + 24)
  (fourth_page_ads : ℕ := 3 * A / 2)
  (total_ads : ℕ := 68 * 3 / 2)
  (sum_of_ads : A + 2 * A + (2 * A + 24) + 3 * A / 2 = total_ads) :
  A = 12 := 
by
  sorry

end ads_on_first_web_page_l266_266922


namespace find_algebraic_expression_l266_266536

-- Definitions as per the conditions
variable (a b : ℝ)

-- Given condition
def given_condition (σ : ℝ) : Prop := σ * (2 * a * b) = 4 * a^2 * b

-- The statement to prove
theorem find_algebraic_expression (σ : ℝ) (h : given_condition a b σ) : σ = 2 * a := 
sorry

end find_algebraic_expression_l266_266536


namespace yellow_tint_percentage_l266_266799

theorem yellow_tint_percentage (V₀ : ℝ) (P₀Y : ℝ) (V_additional : ℝ) 
  (hV₀ : V₀ = 40) (hP₀Y : P₀Y = 0.35) (hV_additional : V_additional = 8) : 
  (100 * ((V₀ * P₀Y + V_additional) / (V₀ + V_additional)) = 45.83) :=
by
  sorry

end yellow_tint_percentage_l266_266799


namespace cos_F_l266_266163

theorem cos_F (D E F : ℝ) (hDEF : D + E + F = 180)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = -16 / 65 :=
  sorry

end cos_F_l266_266163


namespace lindsay_dolls_problem_l266_266576

theorem lindsay_dolls_problem :
  let blonde_dolls := 6
  let brown_dolls := 3 * blonde_dolls
  let black_dolls := brown_dolls / 2
  let red_dolls := 2 * black_dolls
  let combined_dolls := black_dolls + brown_dolls + red_dolls
  combined_dolls - blonde_dolls = 39 :=
by
  sorry

end lindsay_dolls_problem_l266_266576


namespace max_positive_n_l266_266158

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

noncomputable def sequence_condition (a : ℕ → ℤ) : Prop :=
a 1010 / a 1009 < -1

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * (a 1 + a n) / 2

theorem max_positive_n (a : ℕ → ℤ) (h1 : is_arithmetic_sequence a) 
    (h2 : sequence_condition a) : n = 2018 ∧ sum_of_first_n_terms a 2018 > 0 := sorry

end max_positive_n_l266_266158


namespace fraction_irreducible_l266_266055

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l266_266055


namespace carter_reading_pages_l266_266815

theorem carter_reading_pages (c l o : ℕ)
  (h1: c = l / 2)
  (h2: l = o + 20)
  (h3: o = 40) : c = 30 := by
  sorry

end carter_reading_pages_l266_266815


namespace max_xy_value_l266_266383

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l266_266383


namespace certain_event_at_least_one_good_product_l266_266128

-- Define the number of products and their types
def num_products := 12
def num_good_products := 10
def num_defective_products := 2
def num_selected_products := 3

-- Statement of the problem
theorem certain_event_at_least_one_good_product :
  ∀ (selected : Finset (Fin num_products)),
  selected.card = num_selected_products →
  ∃ p ∈ selected, p.val < num_good_products :=
sorry

end certain_event_at_least_one_good_product_l266_266128


namespace gcd_of_18_and_30_l266_266322

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266322


namespace abs_diff_of_m_and_n_l266_266571

theorem abs_diff_of_m_and_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 :=
sorry

end abs_diff_of_m_and_n_l266_266571


namespace max_xy_l266_266378

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l266_266378


namespace smallest_b_l266_266506

theorem smallest_b (b : ℕ) : 
  (b % 3 = 2) ∧ (b % 4 = 3) ∧ (b % 5 = 4) ∧ (b % 7 = 6) ↔ b = 419 :=
by sorry

end smallest_b_l266_266506


namespace recurrence_relation_l266_266408

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l266_266408


namespace staircase_toothpicks_l266_266704

theorem staircase_toothpicks (n : ℕ) (total_toothpicks : ℕ) :
  (∑ k in finset.range(n + 1), 3 * k) = total_toothpicks →
  (∑ k in finset.range(6), 3 * k) = 90 →
  total_toothpicks = 300 →
  n = 13 :=
by
  intros h_general h_base h_300
  sorry

end staircase_toothpicks_l266_266704


namespace prove_river_improvement_l266_266631

def river_improvement_equation (x : ℝ) : Prop :=
  4800 / x - 4800 / (x + 200) = 4

theorem prove_river_improvement (x : ℝ) (h : x > 0) : river_improvement_equation x := by
  sorry

end prove_river_improvement_l266_266631


namespace gcd_18_30_is_6_l266_266263

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266263


namespace probability_absolute_value_l266_266670

noncomputable theory

-- Define the random variable and its normal distribution
def X : ℝ → ℝ := sorry -- X is normally distributed

axiom X_norm : ∀ x : ℝ, P(X ≤ x) = sorry -- X ~ N(4, σ^2)

-- Given conditions
axiom cond1 : X ∼ N(4, σ^2)
axiom cond2 : P(2 ≤ X ≤ 6) ≈ 0.6827

-- Reference data as axioms (approximations can be treated as axioms for practical purposes)
axiom ref1 : P(4 - σ ≤ X ≤ 4 + σ) ≈ 0.6827
axiom ref2 : P(4 - 2 * σ ≤ X ≤ 4 + 2 * σ) ≈ 0.9545
axiom ref3 : P(4 - 3 * σ ≤ X ≤ 4 + 3 * σ) ≈ 0.9973

-- The proof statement
theorem probability_absolute_value (σ : ℝ) : P(abs (X - 2) ≤ 4) = 0.84 :=
sorry

end probability_absolute_value_l266_266670


namespace gcd_18_30_l266_266281

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266281


namespace recurrence_relation_l266_266413

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l266_266413


namespace laptop_final_price_l266_266797

theorem laptop_final_price (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  initial_price = 500 → first_discount = 10 → second_discount = 20 →
  (initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)) = initial_price * 0.72 :=
by
  sorry

end laptop_final_price_l266_266797


namespace smallest_part_is_correct_l266_266628

-- Conditions
def total_value : ℕ := 360
def proportion1 : ℕ := 5
def proportion2 : ℕ := 7
def proportion3 : ℕ := 4
def proportion4 : ℕ := 8
def total_parts := proportion1 + proportion2 + proportion3 + proportion4
def value_per_part := total_value / total_parts
def smallest_proportion : ℕ := proportion3

-- Theorem to prove
theorem smallest_part_is_correct : value_per_part * smallest_proportion = 60 := by
  dsimp [total_value, total_parts, value_per_part, smallest_proportion]
  norm_num
  sorry

end smallest_part_is_correct_l266_266628


namespace find_variable_l266_266947

def expand : ℤ → ℤ := 3*2*6
    
theorem find_variable (a n some_variable : ℤ) (h : (3 - 7 + a = 3)):
  some_variable = -17 :=
sorry

end find_variable_l266_266947


namespace part1_part2_probability_part2_expected_value_l266_266449

-- Definitions based on conditions from the problem
def male_students (n : ℕ) := 10 * n
def female_students (n : ℕ) := 10 * n
def K_squared (n : ℕ) := 4.040

-- Given the total male and female students surveyed is 10n
-- Given K^2 ≈ 4.040 from the table
-- Proof of n = 20 assuming K_squared is accurate
theorem part1 (n : ℕ) (h1 : K_squared n = 4.040)
(h2 : 20 * n / 99 = 4.040) : n = 20 := sorry

-- Definitions based on conditions in problem 2.1
def total_students_not_understand := 9
def total_females_not_understand := 5
def total_males_not_understand := 4
noncomputable def prob_at_least_one_female_selected : ℚ :=
1 - (Nat.choose 4 3 : ℚ) / (Nat.choose 9 3)

-- Proof that the probability of selecting at least one female is 20/21
theorem part2_probability (h3 : prob_at_least_one_female_selected = 20/21) :
prob_at_least_one_female_selected = 20 / 21 := sorry

-- Definitions based on conditions in problem 2.2
noncomputable def expected_value (n : ℚ) := 10 * (11/20 : ℚ)

-- Proof that the expected value of X is 11/2
theorem part2_expected_value : expected_value 10 = 11 / 2 := sorry

end part1_part2_probability_part2_expected_value_l266_266449


namespace product_of_distinct_solutions_l266_266839

theorem product_of_distinct_solutions (x y : ℝ) (h₁ : x ≠ y) (h₂ : x ≠ 0) (h₃ : y ≠ 0) (h₄ : x - 2 / x = y - 2 / y) :
  x * y = -2 :=
sorry

end product_of_distinct_solutions_l266_266839


namespace quad_root_values_count_l266_266566

noncomputable def quad_root_values : Finset ℤ :=
  {20, -20, 12, -12}

theorem quad_root_values_count :
  let roots (α β : ℤ) := 2 * x^2 - m * x + 18 in
  let condition := ∃ α β : ℤ, α * β = 9 ∧ α + β = m / 2 in
  quad_root_values.card = 4 :=
by
  sorry

end quad_root_values_count_l266_266566


namespace smallest_number_starts_with_four_and_decreases_four_times_l266_266828

theorem smallest_number_starts_with_four_and_decreases_four_times :
  ∃ (X : ℕ), ∃ (A n : ℕ), (X = 4 * 10^n + A ∧ X = 4 * (10 * A + 4)) ∧ X = 410256 := 
by
  sorry

end smallest_number_starts_with_four_and_decreases_four_times_l266_266828


namespace range_of_a_l266_266435

noncomputable def satisfies_system (a b c : ℝ) : Prop :=
  (a^2 - b * c - 8 * a + 7 = 0) ∧ (b^2 + c^2 + b * c - 6 * a + 6 = 0)

theorem range_of_a (a b c : ℝ) 
  (h : satisfies_system a b c) : 1 ≤ a ∧ a ≤ 9 :=
by
  sorry

end range_of_a_l266_266435


namespace sum_of_distances_l266_266419

theorem sum_of_distances (P : ℤ × ℤ) (hP : P = (-1, -2)) :
  abs P.1 + abs P.2 = 3 :=
sorry

end sum_of_distances_l266_266419


namespace smallest_base10_integer_l266_266927

def is_valid_digit_base_6 (C : ℕ) : Prop := C ≤ 5
def is_valid_digit_base_8 (D : ℕ) : Prop := D ≤ 7

def CC_6_to_base10 (C : ℕ) : ℕ := 7 * C
def DD_8_to_base10 (D : ℕ) : ℕ := 9 * D

theorem smallest_base10_integer : ∃ C D : ℕ, 
  is_valid_digit_base_6 C ∧ 
  is_valid_digit_base_8 D ∧ 
  CC_6_to_base10 C = DD_8_to_base10 D ∧
  CC_6_to_base10 C = 63 := 
begin
  sorry
end

end smallest_base10_integer_l266_266927


namespace decreasing_interval_and_minimum_value_range_of_k_l266_266835

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x + k / x

theorem decreasing_interval_and_minimum_value (k : ℝ) (hk : k = Real.exp 1) :
  (∀ x > 0, x < Real.exp 1 → (f x k).deriv < 0) ∧ (f (Real.exp 1) k = 2) :=
by
  sorry

theorem range_of_k (k : ℝ) :
  (∀ x1 x2 > 0, x1 > x2 → f x1 k - f x2 k < x1 - x2) ↔ k ∈ Set.Ici (1 / 4) :=
by
  sorry

end decreasing_interval_and_minimum_value_range_of_k_l266_266835


namespace more_students_than_rabbits_l266_266969

theorem more_students_than_rabbits :
  let number_of_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let total_students := students_per_classroom * number_of_classrooms
  let total_rabbits := rabbits_per_classroom * number_of_classrooms
  total_students - total_rabbits = 95 := by
  sorry

end more_students_than_rabbits_l266_266969


namespace sequence_sum_eq_ten_implies_n_eq_120_l266_266136

theorem sequence_sum_eq_ten_implies_n_eq_120 :
  (∀ (a : ℕ → ℝ), (∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1))) →
    (∃ n, (Finset.sum (Finset.range n) a) = 10 → n = 120)) :=
by
  intro a h
  use 120
  intro h_sum
  sorry

end sequence_sum_eq_ten_implies_n_eq_120_l266_266136


namespace percentage_of_loss_l266_266488

theorem percentage_of_loss (CP SP : ℕ) (h1 : CP = 1750) (h2 : SP = 1610) : 
  (CP - SP) * 100 / CP = 8 := by
  sorry

end percentage_of_loss_l266_266488


namespace find_k_l266_266466

theorem find_k (k : ℝ) (x : ℝ) :
  x^2 + k * x + 1 = 0 ∧ x^2 - x - k = 0 → k = 2 := 
sorry

end find_k_l266_266466


namespace number_of_roots_l266_266759

-- Definitions for the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y

-- Main theorem to prove
theorem number_of_roots (f : ℝ → ℝ) (a : ℝ) (h1 : 0 < a) 
  (h2 : is_even_function f) (h3 : is_monotonic_in_interval f a) 
  (h4 : f 0 * f a < 0) : ∃ x0 > 0, f x0 = 0 ∧ ∃ x1 < 0, f x1 = 0 :=
sorry

end number_of_roots_l266_266759


namespace recurrence_relation_p_series_l266_266402

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l266_266402


namespace gcd_of_18_and_30_l266_266241

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266241


namespace probability_of_winning_at_least_10_rubles_l266_266768

-- Definitions based on conditions
def total_tickets : ℕ := 100
def win_20_rubles_tickets : ℕ := 5
def win_15_rubles_tickets : ℕ := 10
def win_10_rubles_tickets : ℕ := 15
def win_2_rubles_tickets : ℕ := 25
def win_nothing_tickets : ℕ := total_tickets - (win_20_rubles_tickets + win_15_rubles_tickets + win_10_rubles_tickets + win_2_rubles_tickets)

-- Probability calculations
def prob_win_20_rubles : ℚ := win_20_rubles_tickets / total_tickets
def prob_win_15_rubles : ℚ := win_15_rubles_tickets / total_tickets
def prob_win_10_rubles : ℚ := win_10_rubles_tickets / total_tickets

-- Prove the probability of winning at least 10 rubles
theorem probability_of_winning_at_least_10_rubles : 
  prob_win_20_rubles + prob_win_15_rubles + prob_win_10_rubles = 0.30 := by
  sorry

end probability_of_winning_at_least_10_rubles_l266_266768


namespace find_coprime_pairs_l266_266512

theorem find_coprime_pairs :
  ∀ (x y : ℕ), x > 0 → y > 0 → x.gcd y = 1 →
    (x ∣ y^2 + 210) →
    (y ∣ x^2 + 210) →
    (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) ∨ 
    (∃ n : ℕ, n > 0 ∧ n = 1 ∧ n = 1 ∧ 
      (x = 212*n - n - 1 ∨ y = 212*n - n - 1)) := sorry

end find_coprime_pairs_l266_266512


namespace binomial_probability_l266_266567

open Probability

-- Given conditions
variables {Ω : Type*} [probability_space Ω]
variables (n : ℕ) (X : Ω → ℕ)

-- Statement of the problem
theorem binomial_probability {X : Ω → ℕ} (hX : binomial X n (1/3)) (hE : 2 = n * (1 / 3)) :
  P (λ ω, X ω = 2) = 80 / 243 :=
sorry

end binomial_probability_l266_266567


namespace find_f_neg_5pi_12_l266_266356

-- Conditions
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

variables {ω φ : ℝ}
def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

axiom monotonicity_condition : is_monotonically_increasing f (π / 6) (2 * π / 3)
axiom symmetry_condition_1 : f (π / 6) = f (2 * π / 3)
axiom symmetry_condition_2 : f (-π / 6) = f (π / 2)

-- Goal
theorem find_f_neg_5pi_12 : f (- 5 * π / 12) = sqrt 3 / 2 := 
sorry

end find_f_neg_5pi_12_l266_266356


namespace fraction_color_films_l266_266950

variables {x y : ℕ} (h₁ : y ≠ 0) (h₂ : x ≠ 0)

theorem fraction_color_films (h₃ : 30 * x > 0) (h₄ : 6 * y > 0) :
  (6 * y : ℚ) / ((3 * y / 10) + 6 * y) = 20 / 21 := by
  sorry

end fraction_color_films_l266_266950


namespace production_rate_equation_l266_266093

theorem production_rate_equation (x : ℝ) (h : x > 0) :
  3000 / x - 3000 / (2 * x) = 5 :=
sorry

end production_rate_equation_l266_266093


namespace calculation_proof_l266_266813
  
  open Real

  theorem calculation_proof :
    sqrt 27 - 2 * cos (30 * (π / 180)) + (1 / 2)^(-2) - abs (1 - sqrt 3) = sqrt 3 + 5 :=
  by
    sorry
  
end calculation_proof_l266_266813


namespace john_expenditure_l266_266496

theorem john_expenditure (X : ℝ) (h : (1/2) * X + (1/3) * X + (1/10) * X + 8 = X) : X = 120 :=
by
  sorry

end john_expenditure_l266_266496


namespace cone_volume_ratio_l266_266080

theorem cone_volume_ratio (r_C h_C r_D h_D : ℝ) (h_rC : r_C = 20) (h_hC : h_C = 40) 
  (h_rD : r_D = 40) (h_hD : h_D = 20) : 
  (1 / 3 * pi * r_C^2 * h_C) / (1 / 3 * pi * r_D^2 * h_D) = 1 / 2 :=
by
  rw [h_rC, h_hC, h_rD, h_hD]
  sorry

end cone_volume_ratio_l266_266080


namespace minimum_value_of_abs_phi_l266_266028

theorem minimum_value_of_abs_phi (φ : ℝ) :
  (∃ k : ℤ, φ = k * π - (13 * π) / 6) → 
  ∃ φ_min : ℝ, 0 ≤ φ_min ∧ φ_min = abs φ ∧ φ_min = π / 6 :=
by
  sorry

end minimum_value_of_abs_phi_l266_266028


namespace sufficient_but_not_necessary_condition_l266_266881

def P (x : ℝ) : Prop := 0 < x ∧ x < 5
def Q (x : ℝ) : Prop := |x - 2| < 3

theorem sufficient_but_not_necessary_condition
  (x : ℝ) : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end sufficient_but_not_necessary_condition_l266_266881


namespace solution_of_az_eq_b_l266_266134

theorem solution_of_az_eq_b (a b z x y : ℝ) :
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬ ∃ y, 2 + y = (b + 1) * y) →
  az = b →
  z = 0 :=
by
  intros h1 h2 h3
  -- proof starts here
  sorry

end solution_of_az_eq_b_l266_266134


namespace sickness_temperature_increase_l266_266770

theorem sickness_temperature_increase :
  ∀ (normal_temp fever_threshold current_temp : ℕ), normal_temp = 95 → fever_threshold = 100 →
  current_temp = fever_threshold + 5 → (current_temp - normal_temp = 10) :=
by
  intros normal_temp fever_threshold current_temp h1 h2 h3
  sorry

end sickness_temperature_increase_l266_266770


namespace log_sum_identity_l266_266970

theorem log_sum_identity :
  (\frac{2}{Real.log 5000 / Real.log 8} + \frac{3}{Real.log 5000 / Real.log 9}) = 1 :=
by
  -- This is where the proof would go
  sorry

end log_sum_identity_l266_266970


namespace gcd_of_18_and_30_l266_266321

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266321


namespace distance_covered_by_center_of_circle_l266_266148

-- Definition of the sides of the triangle
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Definition of the circle's radius
def radius : ℕ := 2

-- Define a function that calculates the perimeter of the smaller triangle
noncomputable def smallerTrianglePerimeter (s1 s2 hyp r : ℕ) : ℕ :=
  (s1 - 2 * r) + (s2 - 2 * r) + (hyp - 2 * r)

-- Main theorem statement
theorem distance_covered_by_center_of_circle :
  smallerTrianglePerimeter side1 side2 hypotenuse radius = 18 :=
by
  sorry

end distance_covered_by_center_of_circle_l266_266148


namespace eq_value_l266_266698

theorem eq_value (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 3 ^ x * 4 ^ y = 19683 := by
  sorry

end eq_value_l266_266698


namespace monotonic_intervals_and_extreme_points_l266_266345

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x

theorem monotonic_intervals_and_extreme_points (a : ℝ) (h : 1 < a) :
  ∃ x1 x2, x1 = 1 ∧ x2 = a ∧ x1 < x2 ∧ f x2 a < - (3 / 2) * x1 :=
by
  sorry

end monotonic_intervals_and_extreme_points_l266_266345


namespace original_pumpkins_count_l266_266893

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l266_266893


namespace recurrence_relation_l266_266407

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l266_266407


namespace train_speed_is_54_kmh_l266_266221

noncomputable def train_length_m : ℝ := 285
noncomputable def train_length_km : ℝ := train_length_m / 1000
noncomputable def time_seconds : ℝ := 19
noncomputable def time_hours : ℝ := time_seconds / 3600
noncomputable def speed : ℝ := train_length_km / time_hours

theorem train_speed_is_54_kmh :
  speed = 54 := by
sorry

end train_speed_is_54_kmh_l266_266221


namespace recurrence_relation_l266_266414

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l266_266414


namespace clea_escalator_time_l266_266428

theorem clea_escalator_time (x y k : ℕ) (h1 : 90 * x = y) (h2 : 30 * (x + k) = y) :
  (y / k) = 45 := by
  sorry

end clea_escalator_time_l266_266428


namespace replace_asterisks_l266_266620

theorem replace_asterisks (x : ℝ) (h : (x / 20) * (x / 80) = 1) : x = 40 :=
sorry

end replace_asterisks_l266_266620


namespace computation_l266_266877

def g (x : ℕ) : ℕ := 7 * x - 3

theorem computation : g (g (g (g 1))) = 1201 := by
  sorry

end computation_l266_266877


namespace geometric_sum_s5_l266_266985

-- Definitions of the geometric sequence and its properties
variable {α : Type*} [Field α] (a : α)

-- The common ratio of the geometric sequence
def common_ratio : α := 2

-- The n-th term of the geometric sequence
def a_n (n : ℕ) : α := a * common_ratio ^ n

-- The sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : α := (a * (1 - common_ratio ^ n)) / (1 - common_ratio)

-- Define the arithmetic sequence property
def aro_seq_property (a_1 a_2 a_5 : α) : Prop := 2 * a_2 = 6 + a_5

-- Define a_2 and a_5 in terms of a
def a2 := a * common_ratio
def a5 := a * common_ratio ^ 4

-- State the main proof problem
theorem geometric_sum_s5 : 
  aro_seq_property a (a2 a) (a5 a) → 
  S_n a 5 = -31 / 2 :=
by
  sorry

end geometric_sum_s5_l266_266985


namespace parabola_decreasing_m_geq_neg2_l266_266014

theorem parabola_decreasing_m_geq_neg2 (m : ℝ) :
  (∀ x ≥ 2, ∃ y, y = -5 * (x + m)^2 - 3 ∧ (∀ x1 y1, x1 ≥ 2 → y1 = -5 * (x1 + m)^2 - 3 → y1 ≤ y)) →
  m ≥ -2 := 
by
  intro h
  sorry

end parabola_decreasing_m_geq_neg2_l266_266014


namespace average_employees_per_week_l266_266634

theorem average_employees_per_week (x : ℝ)
  (h1 : ∀ (x : ℝ), ∃ y : ℝ, y = x + 200)
  (h2 : ∀ (x : ℝ), ∃ z : ℝ, z = x + 150)
  (h3 : ∀ (x : ℝ), ∃ w : ℝ, w = 2 * (x + 150))
  (h4 : ∀ (w : ℝ), w = 400) :
  (250 + 50 + 200 + 400) / 4 = 225 :=
by 
  sorry

end average_employees_per_week_l266_266634


namespace bolts_per_box_l266_266521

def total_bolts_and_nuts_used : Nat := 113
def bolts_left_over : Nat := 3
def nuts_left_over : Nat := 6
def boxes_of_bolts : Nat := 7
def boxes_of_nuts : Nat := 3
def nuts_per_box : Nat := 15

theorem bolts_per_box :
  let total_bolts_and_nuts := total_bolts_and_nuts_used + bolts_left_over + nuts_left_over
  let total_nuts := boxes_of_nuts * nuts_per_box
  let total_bolts := total_bolts_and_nuts - total_nuts
  let bolts_per_box := total_bolts / boxes_of_bolts
  bolts_per_box = 11 := by
  sorry

end bolts_per_box_l266_266521


namespace max_product_l266_266369

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l266_266369


namespace quadratic_function_distinct_zeros_l266_266858

theorem quadratic_function_distinct_zeros (a : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) ↔ (a ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 0) := 
by
  sorry

end quadratic_function_distinct_zeros_l266_266858


namespace maximum_xy_value_l266_266377

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l266_266377


namespace no_integer_roots_of_polynomial_l266_266119

theorem no_integer_roots_of_polynomial :
  ¬ ∃ (x : ℤ), x^3 - 3 * x^2 - 10 * x + 20 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l266_266119


namespace find_kn_l266_266656

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end find_kn_l266_266656


namespace gcd_18_30_l266_266252

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266252


namespace beautiful_point_coordinates_l266_266543

-- Define a "beautiful point"
def is_beautiful_point (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = P.1 * P.2

theorem beautiful_point_coordinates (M : ℝ × ℝ) : 
  is_beautiful_point M ∧ abs M.1 = 2 → 
  (M = (2, 2) ∨ M = (-2, 2/3)) :=
by sorry

end beautiful_point_coordinates_l266_266543


namespace delete_middle_divides_l266_266664

def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := n / 10000
  let b := (n % 10000) / 1000
  let c := (n % 1000) / 100
  let d := (n % 100) / 10
  let e := n % 10
  (a, b, c, d, e)

def delete_middle_digit (n : ℕ) : ℕ :=
  let (a, b, c, d, e) := digits n
  1000 * a + 100 * b + 10 * d + e

theorem delete_middle_divides (n : ℕ) (hn : 10000 ≤ n ∧ n < 100000) :
  (delete_middle_digit n) ∣ n :=
sorry

end delete_middle_divides_l266_266664


namespace no_factors_of_p_l266_266660

open Polynomial

noncomputable def p : Polynomial ℝ := X^4 - 4 * X^2 + 16
noncomputable def optionA : Polynomial ℝ := X^2 + 4
noncomputable def optionB : Polynomial ℝ := X + 2
noncomputable def optionC : Polynomial ℝ := X^2 - 4*X + 4
noncomputable def optionD : Polynomial ℝ := X^2 - 4

theorem no_factors_of_p (h : Polynomial ℝ) : h ≠ p / optionA ∧ h ≠ p / optionB ∧ h ≠ p / optionC ∧ h ≠ p / optionD := by
  sorry

end no_factors_of_p_l266_266660


namespace problem1_l266_266632

theorem problem1 (k : ℝ) : (∃ x : ℝ, k*x^2 + (2*k + 1)*x + (k - 1) = 0) → k ≥ -1/8 := 
sorry

end problem1_l266_266632


namespace remitted_amount_is_correct_l266_266802

-- Define the constants and conditions of the problem
def total_sales : ℝ := 32500
def commission_rate1 : ℝ := 0.05
def commission_limit : ℝ := 10000
def commission_rate2 : ℝ := 0.04

-- Define the function to calculate the remitted amount
def remitted_amount (total_sales commission_rate1 commission_limit commission_rate2 : ℝ) : ℝ :=
  let commission1 := commission_rate1 * commission_limit
  let remaining_sales := total_sales - commission_limit
  let commission2 := commission_rate2 * remaining_sales
  total_sales - (commission1 + commission2)

-- Lean statement to prove the remitted amount
theorem remitted_amount_is_correct :
  remitted_amount total_sales commission_rate1 commission_limit commission_rate2 = 31100 :=
by
  sorry

end remitted_amount_is_correct_l266_266802


namespace expression_increase_fraction_l266_266424

theorem expression_increase_fraction (x y : ℝ) :
  let x' := 1.4 * x
  let y' := 1.4 * y
  let original := x * y^2
  let increased := x' * y'^2
  increased - original = (1744/1000) * original := by
sorry

end expression_increase_fraction_l266_266424


namespace total_pictures_painted_l266_266806

def pictures_painted_in_june : ℕ := 2
def pictures_painted_in_july : ℕ := 2
def pictures_painted_in_august : ℕ := 9

theorem total_pictures_painted : 
  pictures_painted_in_june + pictures_painted_in_july + pictures_painted_in_august = 13 :=
by
  sorry

end total_pictures_painted_l266_266806


namespace max_val_y_l266_266432

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + x - 1

theorem max_val_y (a b : ℝ) (ha : 0 ≤ a) (hb : b ≤ 1)
    (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a ∨ f x = b) :
  ∃ z, f z + f⁻¹ z = 2 := by
  sorry

end max_val_y_l266_266432


namespace find_ab_bc_value_l266_266020

theorem find_ab_bc_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 := by
sorry

end find_ab_bc_value_l266_266020


namespace lcm_gcd_product_l266_266519

theorem lcm_gcd_product (n m : ℕ) (h1 : n = 9) (h2 : m = 10) : 
  Nat.lcm n m * Nat.gcd n m = 90 := by
  sorry

end lcm_gcd_product_l266_266519


namespace gcd_18_30_l266_266295

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266295


namespace jean_more_trips_than_bill_l266_266499

variable (b j : ℕ)

theorem jean_more_trips_than_bill
  (h1 : b + j = 40)
  (h2 : j = 23) :
  j - b = 6 := by
  sorry

end jean_more_trips_than_bill_l266_266499


namespace additional_hours_needed_l266_266238

-- Define the conditions
def speed : ℕ := 5  -- kilometers per hour
def total_distance : ℕ := 30 -- kilometers
def hours_walked : ℕ := 3 -- hours

-- Define the statement to prove
theorem additional_hours_needed : total_distance / speed - hours_walked = 3 := 
by
  sorry

end additional_hours_needed_l266_266238


namespace max_value_phi_l266_266688

theorem max_value_phi (φ : ℝ) (hφ : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) :
  (∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2 - Real.pi / 3) →
  φ = Real.pi / 6 :=
by 
  intro h
  sorry

end max_value_phi_l266_266688


namespace fraction_problem_l266_266666

-- Definitions of x and y based on the given conditions
def x : ℚ := 3 / 5
def y : ℚ := 7 / 9

-- The theorem stating the mathematical equivalence to be proven
theorem fraction_problem : (5 * x + 9 * y) / (45 * x * y) = 10 / 21 :=
by
  sorry

end fraction_problem_l266_266666


namespace factorization_correct_l266_266826

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by
  sorry

end factorization_correct_l266_266826


namespace find_a_l266_266086

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + x

-- Define the derivative of the function f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 1

-- The main theorem: if the tangent at x = 1 is parallel to the line y = 2x, then a = 1
theorem find_a (a : ℝ) : f' 1 a = 2 → a = 1 :=
by
  intro h
  -- The proof is skipped
  sorry

end find_a_l266_266086


namespace Eleanor_books_l266_266822

theorem Eleanor_books (h p : ℕ) : 
    h + p = 12 ∧ 28 * h + 18 * p = 276 → h = 6 :=
by
  intro hp
  sorry

end Eleanor_books_l266_266822


namespace ratio_of_incomes_l266_266913

theorem ratio_of_incomes 
  (E1 E2 I1 I2 : ℕ)
  (h1 : E1 / E2 = 3 / 2)
  (h2 : E1 = I1 - 1200)
  (h3 : E2 = I2 - 1200)
  (h4 : I1 = 3000) :
  I1 / I2 = 5 / 4 :=
sorry

end ratio_of_incomes_l266_266913


namespace john_has_22_quarters_l266_266038

-- Definitions based on conditions
def number_of_quarters (Q : ℕ) : ℕ := Q
def number_of_dimes (Q : ℕ) : ℕ := Q + 3
def number_of_nickels (Q : ℕ) : ℕ := Q - 6

-- Total number of coins condition
def total_number_of_coins (Q : ℕ) : Prop := 
  (number_of_quarters Q) + (number_of_dimes Q) + (number_of_nickels Q) = 63

-- Goal: Proving the number of quarters is 22
theorem john_has_22_quarters : ∃ Q : ℕ, total_number_of_coins Q ∧ Q = 22 :=
by
  -- Proof skipped 
  sorry

end john_has_22_quarters_l266_266038


namespace students_not_invited_count_l266_266030

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students not invited to the event
def not_invited_students : ℕ := 14

-- Define the sets representing different levels of friends of Anna
-- This demonstrates that the total invited students can be derived from given conditions

def anna_immediate_friends : ℕ := 4
def anna_second_level_friends : ℕ := (12 - anna_immediate_friends)
def anna_third_level_friends : ℕ := (16 - 12)

-- Define total invited students
def invited_students : ℕ := 
  anna_immediate_friends + 
  anna_second_level_friends +
  anna_third_level_friends

-- Prove that the number of not invited students is 14
theorem students_not_invited_count : (total_students - invited_students) = not_invited_students :=
by
  sorry

end students_not_invited_count_l266_266030


namespace existence_of_B_l266_266859

theorem existence_of_B (a b : ℝ) (A : ℝ) (H_a : a = 1) (H_b : b = Real.sqrt 3) (H_A : A = Real.pi / 6) :
  ∃ B : ℝ, (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) ∧ a / Real.sin A = b / Real.sin B :=
by
  sorry

end existence_of_B_l266_266859


namespace binomial_expansion_of_110_minus_1_l266_266201

theorem binomial_expansion_of_110_minus_1:
  110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 109^5 :=
by
  -- We will use the binomial theorem: (a - b)^n = ∑ (k in range(n+1)), C(n, k) * a^(n-k) * (-b)^k
  -- where C(n, k) are the binomial coefficients.
  sorry

end binomial_expansion_of_110_minus_1_l266_266201


namespace max_hours_is_70_l266_266048

-- Define the conditions
def regular_hourly_rate : ℕ := 8
def first_20_hours : ℕ := 20
def max_weekly_earnings : ℕ := 660
def overtime_rate_multiplier : ℕ := 25

-- Define the overtime hourly rate
def overtime_hourly_rate : ℕ := regular_hourly_rate + (regular_hourly_rate * overtime_rate_multiplier / 100)

-- Define the earnings for the first 20 hours
def earnings_first_20_hours : ℕ := regular_hourly_rate * first_20_hours

-- Define the maximum overtime earnings
def max_overtime_earnings : ℕ := max_weekly_earnings - earnings_first_20_hours

-- Define the maximum overtime hours
def max_overtime_hours : ℕ := max_overtime_earnings / overtime_hourly_rate

-- Define the maximum total hours
def max_total_hours : ℕ := first_20_hours + max_overtime_hours

-- Theorem to prove that the maximum number of hours is 70
theorem max_hours_is_70 : max_total_hours = 70 :=
by
  sorry

end max_hours_is_70_l266_266048


namespace max_value_y_l266_266702

noncomputable def max_y (a b c d : ℝ) : ℝ :=
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2

theorem max_value_y {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 10) : max_y a b c d = 40 := 
  sorry

end max_value_y_l266_266702


namespace max_xy_value_l266_266386

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l266_266386


namespace find_tangent_parallel_to_x_axis_l266_266514

theorem find_tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), y = x^2 - 3 * x ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) := 
by
  sorry

end find_tangent_parallel_to_x_axis_l266_266514


namespace new_average_age_l266_266186

theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) :
  avg_age = 40 →
  num_people = 8 →
  leaving_age = 25 →
  remaining_people = 7 →
  (avg_age * num_people - leaving_age) / remaining_people = 42 :=
by
  sorry

end new_average_age_l266_266186


namespace cricket_team_number_of_players_l266_266918

theorem cricket_team_number_of_players 
  (throwers : ℕ)
  (total_right_handed : ℕ)
  (frac_left_handed_non_thrower : ℚ)
  (right_handed_throwers : throwers = 37)
  (all_throwers_right_handed : throwers * 1 = throwers)
  (total_right_handed_players : total_right_handed = 55)
  (left_handed_non_thrower_fraction : frac_left_handed_non_thrower = 1 / 3)
  (right_handed_non_throwers : total_right_handed - throwers = 18) :
  let non_throwers := (total_right_handed - throwers) * (1 / frac_left_handed_non_thrower - 1) in
  let total_players := throwers + non_throwers.to_nat in
  total_players = 64 :=
by 
  let non_throwers := 18 * (3 / 2) in 
  let total_players := 37 + non_throwers in
  have : total_players = 64 := rfl
  sorry

end cricket_team_number_of_players_l266_266918


namespace angle_DGO_is_50_degrees_l266_266036

theorem angle_DGO_is_50_degrees
  (triangle_DOG : Type)
  (D G O : triangle_DOG)
  (angle_DOG : ℝ)
  (angle_DGO : ℝ)
  (angle_OGD : ℝ)
  (bisect : Prop) :

  angle_DGO = 50 := 
by
  -- Conditions
  have h1 : angle_DGO = angle_DOG := sorry
  have h2 : angle_DOG = 40 := sorry
  have h3 : bisect := sorry
  -- Goal
  sorry

end angle_DGO_is_50_degrees_l266_266036


namespace principal_amount_l266_266147

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 5)

theorem principal_amount :
  ((P * R * T) / 100 = P - 2000) → P = 2500 :=
by
  sorry

end principal_amount_l266_266147


namespace gcd_18_30_l266_266305

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266305


namespace gcd_of_18_and_30_l266_266323

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266323


namespace chocolates_bought_l266_266737

theorem chocolates_bought (C S N : ℕ) (h1 : 4 * C = 7 * (S - C)) (h2 : N * C = 77 * S) :
  N = 121 :=
by
  sorry

end chocolates_bought_l266_266737


namespace solution_set_inequality_x0_1_solution_set_inequality_x0_half_l266_266346

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem solution_set_inequality_x0_1 : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f 1 ≥ c * (x - 1)) ↔ c ∈ Set.Icc (-1) 1 := 
by
  sorry

theorem solution_set_inequality_x0_half : 
  ∀ (c : ℝ), (∀ x, 0 < x → f x - f (1 / 2) ≥ c * (x - 1 / 2)) ↔ c = -2 :=
by
  sorry

end solution_set_inequality_x0_1_solution_set_inequality_x0_half_l266_266346


namespace radius_of_sphere_l266_266018

theorem radius_of_sphere (R : ℝ) (shots_count : ℕ) (shot_radius : ℝ) :
  shots_count = 125 →
  shot_radius = 1 →
  (shots_count : ℝ) * (4 / 3 * Real.pi * shot_radius^3) = 4 / 3 * Real.pi * R^3 →
  R = 5 :=
by
  intros h1 h2 h3
  sorry

end radius_of_sphere_l266_266018


namespace find_y_minus_x_l266_266609

theorem find_y_minus_x (x y : ℕ) (hx : x + y = 540) (hxy : (x : ℚ) / (y : ℚ) = 7 / 8) : y - x = 36 :=
by
  sorry

end find_y_minus_x_l266_266609


namespace no_real_solution_for_pairs_l266_266852

theorem no_real_solution_for_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 1 / (a + b)) :=
by
  sorry

end no_real_solution_for_pairs_l266_266852


namespace quadratic_intersection_l266_266980

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  ∃ x y : ℝ, (y = a * x^2 + b * x + c) ∧ (y = a * (x - h)^2 + b * (x - h) + d)
    ∧ x = (d - c) / b
    ∧ y = a * (d - c)^2 / b^2 + d :=
by {
  sorry
}

end quadratic_intersection_l266_266980


namespace rate_of_discount_l266_266484

theorem rate_of_discount (Marked_Price Selling_Price : ℝ) (h_marked : Marked_Price = 80) (h_selling : Selling_Price = 68) : 
  ((Marked_Price - Selling_Price) / Marked_Price) * 100 = 15 :=
by
  -- Definitions from conditions
  rw [h_marked, h_selling]
  -- Substitute the values and simplify
  sorry

end rate_of_discount_l266_266484


namespace smallest_sum_of_squares_l266_266748

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l266_266748


namespace expression_divisible_by_24_l266_266889

theorem expression_divisible_by_24 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end expression_divisible_by_24_l266_266889


namespace fraction_increase_by_two_l266_266542

theorem fraction_increase_by_two (x y : ℝ) : 
  (3 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * (3 * x * y) / (x + y) :=
by
  sorry

end fraction_increase_by_two_l266_266542


namespace value_of_3Y5_l266_266504

def Y (a b : ℤ) : ℤ := b + 10 * a - a^2 - b^2

theorem value_of_3Y5 : Y 3 5 = 1 := sorry

end value_of_3Y5_l266_266504


namespace find_A_l266_266183

theorem find_A (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 2 * x ^ 2 - 13 * x + 10 ≠ 0 → 1 / (x ^ 3 - 2 * x ^ 2 - 13 * x + 10) = A / (x + 2) + B / (x - 1) + C / (x - 1) ^ 2)
  → A = 1 / 9 := 
sorry

end find_A_l266_266183


namespace triangle_sides_ratios_l266_266733

theorem triangle_sides_ratios (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) :
  a / (b + c) = b / (a + c) + c / (a + b) :=
sorry

end triangle_sides_ratios_l266_266733


namespace student_question_choices_l266_266644

-- Definitions based on conditions
def partA_questions := 10
def partB_questions := 10
def choose_from_partA := 8
def choose_from_partB := 5

-- The proof problem statement
theorem student_question_choices :
  (Nat.choose partA_questions choose_from_partA) * (Nat.choose partB_questions choose_from_partB) = 11340 :=
by
  sorry

end student_question_choices_l266_266644


namespace distinct_real_nums_condition_l266_266042

theorem distinct_real_nums_condition 
  (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 :=
by
  sorry

end distinct_real_nums_condition_l266_266042


namespace max_product_l266_266368

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l266_266368


namespace gcd_of_18_and_30_l266_266325

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266325


namespace smallest_yummy_is_minus_2013_l266_266443

-- Define a yummy integer
def is_yummy (A : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (a : ℤ), (a <= A) ∧ (a + k = A) ∧ ((k + 1) * A - k*(k + 1)/2 = 2014)

-- Define the smallest yummy integer
def smallest_yummy : ℤ :=
  -2013

-- The Lean theorem to state the proof problem
theorem smallest_yummy_is_minus_2013 : ∀ A : ℤ, is_yummy A → (-2013 ≤ A) :=
by
  sorry

end smallest_yummy_is_minus_2013_l266_266443


namespace cookie_count_l266_266165

theorem cookie_count (C : ℕ) 
  (h1 : 3 * C / 4 + 1 * (C / 4) / 5 + 1 * (C / 4) * 4 / 20 = 10) 
  (h2: 1 * (5 * 4 / 20) / 10 = 1): 
  C = 100 :=
by 
sorry

end cookie_count_l266_266165


namespace equation_of_ellipse_equation_of_line_AB_l266_266337

-- Step 1: Given conditions for the ellipse and related hyperbola.
def condition_eccentricity (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c / a = Real.sqrt 2 / 2

def condition_distance_focus_asymptote (c : ℝ) : Prop :=
  abs c / Real.sqrt (1 + 2) = Real.sqrt 3 / 3

-- Step 2: Given conditions for the line AB.
def condition_line_A_B (k m : ℝ) : Prop :=
  k < 0 ∧ m^2 = 4 / 5 * (1 + k^2) ∧
  ∃ (x1 x2 y1 y2 : ℝ), 
  (1 + 2 * k^2) * x1^2 + 4 * k * m * x1 + 2 * m^2 - 2 = 0 ∧ 
  (1 + 2 * k^2) * x2^2 + 4 * k * m * x2 + 2 * m^2 - 2 = 0 ∧
  x1 + x2 = -4 * k * m / (1 + 2*k^2) ∧ 
  x1 * x2 = (2 * m^2 - 2) / (1 + 2*k^2)

def condition_circle_passes_F2 (x1 x2 k m : ℝ) : Prop :=
  (1 + k^2) * x1 * x2 + (k * m - 1) * (x1 + x2) + m^2 + 1 = 0

noncomputable def problem_data : Prop :=
  ∃ (a b c k m x1 x2 : ℝ),
    condition_eccentricity a b c ∧
    condition_distance_focus_asymptote c ∧
    condition_line_A_B k m ∧
    condition_circle_passes_F2 x1 x2 k m

-- Step 3: Statements to be proven.
theorem equation_of_ellipse : problem_data → 
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) :=
by sorry

theorem equation_of_line_AB : problem_data → 
  ∃ (k m : ℝ), m = 1 ∧ k = -1/2 ∧ ∀ x y : ℝ, (y = k * x + m) ↔ (y = -0.5 * x + 1) :=
by sorry

end equation_of_ellipse_equation_of_line_AB_l266_266337


namespace greatest_xy_value_l266_266388

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l266_266388


namespace points_count_l266_266216

theorem points_count :
  let A := (1, 0)
  let line1 := λ x, x = -1
  let parabola P := (P.1^2 / 4 = P.2)
  let distance_to_line P l := abs(P.1 - P.2) / real.sqrt 2
  in  [P | parabola P ∧ distance_to_line P (λ x, x = x) = real.sqrt 2 / 2].length = 3 :=
sorry

end points_count_l266_266216


namespace f_of_1789_l266_266605

-- Definitions as per conditions
def f : ℕ → ℕ := sorry -- This will be the function definition satisfying the conditions

axiom f_f_n (n : ℕ) (h : n > 0) : f (f n) = 4 * n + 9
axiom f_2_k (k : ℕ) : f (2^k) = 2^(k+1) + 3

-- Prove f(1789) = 3581 given the conditions.
theorem f_of_1789 : f 1789 = 3581 := 
sorry

end f_of_1789_l266_266605


namespace cookie_cost_l266_266589

theorem cookie_cost
  (classes3 : ℕ) (students_per_class3 : ℕ)
  (classes4 : ℕ) (students_per_class4 : ℕ)
  (classes5 : ℕ) (students_per_class5 : ℕ)
  (hamburger_cost : ℝ) (carrot_cost : ℝ) (total_lunch_cost : ℝ) (cookie_cost : ℝ)
  (h1 : classes3 = 5) (h2 : students_per_class3 = 30)
  (h3 : classes4 = 4) (h4 : students_per_class4 = 28)
  (h5 : classes5 = 4) (h6 : students_per_class5 = 27)
  (h7 : hamburger_cost = 2.10) (h8 : carrot_cost = 0.50)
  (h9 : total_lunch_cost = 1036):
  ((classes3 * students_per_class3) + (classes4 * students_per_class4) + (classes5 * students_per_class5)) * (cookie_cost + hamburger_cost + carrot_cost) = total_lunch_cost → 
  cookie_cost = 0.20 := 
by 
  sorry

end cookie_cost_l266_266589


namespace commission_percentage_l266_266636

-- Define the given conditions
def cost_of_item : ℝ := 17
def observed_price : ℝ := 25.50
def desired_profit_percentage : ℝ := 0.20

-- Calculate the desired profit in dollars
def desired_profit : ℝ := desired_profit_percentage * cost_of_item

-- Calculate the total desired price for the distributor
def total_desired_price : ℝ := cost_of_item + desired_profit

-- Calculate the commission in dollars
def commission_in_dollars : ℝ := observed_price - total_desired_price

-- Prove that commission percentage taken by the online store is 20%
theorem commission_percentage :
  (commission_in_dollars / observed_price) * 100 = 20 := 
by
  -- This is the placeholder for the proof
  sorry

end commission_percentage_l266_266636


namespace ratio_new_average_to_original_l266_266100

theorem ratio_new_average_to_original (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / scores.length : ℝ)
  let new_sum := scores.sum + 2 * A
  let new_avg := new_sum / (scores.length + 2)
  new_avg / A = 1 := 
by
  sorry

end ratio_new_average_to_original_l266_266100


namespace tim_original_vocab_l266_266465

theorem tim_original_vocab (days_in_year : ℕ) (years : ℕ) (learned_per_day : ℕ) (vocab_increase : ℝ) :
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  original_vocab = 14600 :=
by
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  show original_vocab = 14600
  sorry

end tim_original_vocab_l266_266465


namespace problem_statement_l266_266357

-- Definition of our function f
def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi / 6))

-- Theorem to prove that f(-5π/12) = √3/2
theorem problem_statement : 
  f(-5 * Real.pi / 12) = sqrt 3 / 2 := 
sorry

end problem_statement_l266_266357


namespace smallest_value_of_3a_plus_1_l266_266364

theorem smallest_value_of_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 2 = 2) : 3 * a + 1 = -5/4 :=
by
  sorry

end smallest_value_of_3a_plus_1_l266_266364


namespace Ben_more_new_shirts_than_Joe_l266_266650

theorem Ben_more_new_shirts_than_Joe :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
    alex_shirts = 4 →
    joe_shirts = alex_shirts + 3 →
    ben_shirts = 15 →
    ben_shirts - joe_shirts = 8 :=
by
  intros alex_shirts joe_shirts ben_shirts
  intros h_alex h_joe h_ben
  sorry

end Ben_more_new_shirts_than_Joe_l266_266650


namespace find_angle_y_l266_266033

-- Definitions of the angles in the triangle
def angle_ACD : ℝ := 90
def angle_DEB : ℝ := 58

-- Theorem proving the value of angle DCE (denoted as y)
theorem find_angle_y (angle_sum_property : angle_ACD + y + angle_DEB = 180) : y = 32 :=
by sorry

end find_angle_y_l266_266033


namespace sequence_inequality_l266_266205

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+2) => F (n+1) + F n

theorem sequence_inequality (n : ℕ) :
  (F (n+1) : ℝ)^(1 / n) ≥ 1 + 1 / ((F n : ℝ)^(1 / n)) :=
by
  sorry

end sequence_inequality_l266_266205


namespace find_optimal_price_and_units_l266_266951

noncomputable def price_and_units (x : ℝ) : Prop := 
  let cost_price := 40
  let initial_units := 500
  let profit_goal := 8000
  50 ≤ x ∧ x ≤ 70 ∧ (x - cost_price) * (initial_units - 10 * (x - 50)) = profit_goal

theorem find_optimal_price_and_units : 
  ∃ x units, price_and_units x ∧ units = 500 - 10 * (x - 50) ∧ x = 60 ∧ units = 400 := 
sorry

end find_optimal_price_and_units_l266_266951


namespace sum_of_squares_eq_229_l266_266755

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l266_266755


namespace set_intersection_complement_l266_266690

-- Definitions of the sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Statement of the problem for Lean 4
theorem set_intersection_complement :
  A ∩ (Set.compl B) = {1, 5, 7} := 
sorry

end set_intersection_complement_l266_266690


namespace smallest_sum_of_squares_l266_266746

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) : ∃ (x' y' : ℤ), x'^2 - y'^2 = 221 ∧ x'^2 + y'^2 ≤ x^2 + y^2 ∧ x'^2 + y'^2 = 229 :=
by
  -- Conditions and remaining goals to be proved
  sorry

end smallest_sum_of_squares_l266_266746


namespace percentage_of_whole_is_10_l266_266483

def part : ℝ := 0.01
def whole : ℝ := 0.1

theorem percentage_of_whole_is_10 : (part / whole) * 100 = 10 := by
  sorry

end percentage_of_whole_is_10_l266_266483


namespace a_eq_zero_iff_purely_imaginary_l266_266982

open Complex

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem a_eq_zero_iff_purely_imaginary (a b : ℝ) :
  (a = 0) ↔ purely_imaginary (a + b * Complex.I) :=
by
  sorry

end a_eq_zero_iff_purely_imaginary_l266_266982


namespace flowers_are_55_percent_daisies_l266_266796

noncomputable def percent_daisies (F : ℝ) (yellow : ℝ) (white_daisies : ℝ) (yellow_daisies : ℝ) : ℝ :=
  (yellow_daisies + white_daisies) / F * 100

theorem flowers_are_55_percent_daisies (F : ℝ) (yellow_t : ℝ) (yellow_d : ℝ) (white : ℝ) (white_d : ℝ) :
    yellow_t = 0.5 * yellow →
    yellow_d = yellow - yellow_t →
    white_d = (2 / 3) * white →
    yellow = (7 / 10) * F →
    white = F - yellow →
    percent_daisies F yellow white_d yellow_d = 55 :=
by
  sorry

end flowers_are_55_percent_daisies_l266_266796


namespace inequality_proof_equality_case_l266_266053

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) ≥ c / b - (c^2) / a) :=
sorry

theorem equality_case (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) = c / b - c^2 / a) ↔ (a = b * c) :=
sorry

end inequality_proof_equality_case_l266_266053


namespace skew_lines_angle_range_l266_266071

theorem skew_lines_angle_range (θ : ℝ) (h_skew : θ > 0 ∧ θ ≤ 90) :
  0 < θ ∧ θ ≤ 90 :=
sorry

end skew_lines_angle_range_l266_266071


namespace fifth_term_binomial_expansion_l266_266648

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_term_binomial_expansion (b x : ℝ) :
  let term := (binomial 7 4) * ((b / x)^(7 - 4)) * ((-x^2 * b)^4)
  term = -35 * b^7 * x^5 := 
by
  sorry

end fifth_term_binomial_expansion_l266_266648


namespace maria_total_cost_l266_266584

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end maria_total_cost_l266_266584


namespace find_constant_c_l266_266418

theorem find_constant_c : ∃ (c : ℝ), (∀ n : ℤ, c * (n:ℝ)^2 ≤ 3600) ∧ (∀ n : ℤ, n ≤ 5) ∧ (c = 144) :=
by
  sorry

end find_constant_c_l266_266418


namespace original_savings_l266_266084

theorem original_savings (tv_cost : ℚ) (fraction_on_tv : ℚ) (original_savings : ℚ) :
  (tv_cost = 220) → (fraction_on_tv = 1 / 4) → (original_savings * fraction_on_tv = tv_cost) →
  original_savings = 880 :=
by
  intros h_tv_cost h_fraction_on_tv h_equal
  sorry

end original_savings_l266_266084


namespace simplify_fraction_expression_l266_266784

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l266_266784


namespace no_such_pairs_exist_l266_266973

theorem no_such_pairs_exist : ¬ ∃ (n m : ℕ), n > 1 ∧ (∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n) ∧ 
                                    (∀ d : ℕ, d ≠ n → d ∣ n → d + 1 ∣ m ∧ d + 1 ≠ m ∧ d + 1 ≠ 1) :=
by
  sorry

end no_such_pairs_exist_l266_266973


namespace brenda_bought_stones_l266_266961

-- Given Conditions
def n_bracelets : ℕ := 3
def n_stones_per_bracelet : ℕ := 12

-- Problem Statement: Prove Betty bought the correct number of stone-shaped stars
theorem brenda_bought_stones :
  let n_total_stones := n_bracelets * n_stones_per_bracelet
  n_total_stones = 36 := 
by 
  -- proof goes here, but we omit it with sorry
  sorry

end brenda_bought_stones_l266_266961


namespace janet_additional_money_needed_is_1225_l266_266557

def savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def months_required : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

noncomputable def total_rent : ℕ := rent_per_month * months_required
noncomputable def total_upfront_cost : ℕ := total_rent + deposit + utility_deposit + moving_costs
noncomputable def additional_money_needed : ℕ := total_upfront_cost - savings

theorem janet_additional_money_needed_is_1225 : additional_money_needed = 1225 :=
by
  sorry

end janet_additional_money_needed_is_1225_l266_266557


namespace range_of_m_l266_266027

theorem range_of_m (m : ℝ) :
  (∃ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 0 ∧
    (∃ ρ₀ θ₀ : ℝ, ∀ ρ θ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 
      m * ρ₀ * (Real.cos θ₀)^2 + 3 * ρ₀ * (Real.sin θ₀)^2 - 6 * (Real.cos θ₀))) →
  m > 0 ∧ m ≠ 3 := sorry

end range_of_m_l266_266027


namespace correct_choices_l266_266474

theorem correct_choices :
  (∃ u : ℝ × ℝ, (2 * u.1 + u.2 + 3 = 0) → u = (1, -2)) ∧
  ¬ (∀ a : ℝ, (a = -1 ↔ a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0) → a = -1) ∧
  ((∃ (l : ℝ) (P : ℝ × ℝ), l = x + y - 6 → P = (2, 4) → 2 + 4 = l) → x + y - 6 = 0) ∧
  ((∃ (m b : ℝ), y = m * x + b → b = -2) → y = 3 * x - 2) :=
sorry

end correct_choices_l266_266474


namespace value_of_a_l266_266857

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := (2 : ℝ) * x - y - 1 = 0

def line2 (x y a : ℝ) : Prop := (2 : ℝ) * x + (a + 1) * y + 2 = 0

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 x y) → (line2 x y a)

-- The theorem to be proved
theorem value_of_a (a : ℝ) : parallel_lines a → a = -2 :=
sorry

end value_of_a_l266_266857


namespace series_sum_eq_five_l266_266816

open Nat Real

noncomputable def sum_series : ℝ := ∑' (n : ℕ), (2 * n ^ 2 - n) / (n * (n + 1) * (n + 2))

theorem series_sum_eq_five : sum_series = 5 :=
sorry

end series_sum_eq_five_l266_266816


namespace unique_solution_quadratic_l266_266663

theorem unique_solution_quadratic (q : ℚ) :
  (∃ x : ℚ, q ≠ 0 ∧ q * x^2 - 16 * x + 9 = 0) ∧ (∀ y z : ℚ, (q * y^2 - 16 * y + 9 = 0 ∧ q * z^2 - 16 * z + 9 = 0) → y = z) → q = 64 / 9 :=
by
  sorry

end unique_solution_quadratic_l266_266663


namespace sequence_periodic_l266_266787

noncomputable def exists_N (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n+2) = abs (a (n+1)) - a n

theorem sequence_periodic (a : ℕ → ℝ) (h : exists_N a) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a (n+9) = a n :=
sorry

end sequence_periodic_l266_266787


namespace ratio_of_a_to_b_l266_266854

theorem ratio_of_a_to_b (a b : ℝ) (h1 : 0.5 / 100 * a = 85) (h2 : 0.75 / 100 * b = 150) : a / b = 17 / 20 :=
by {
  -- Proof will go here
  sorry
}

end ratio_of_a_to_b_l266_266854


namespace new_barbell_cost_l266_266555

theorem new_barbell_cost (old_barbell_cost new_barbell_cost : ℝ) 
  (h1 : old_barbell_cost = 250)
  (h2 : new_barbell_cost = old_barbell_cost * 1.3) :
  new_barbell_cost = 325 := by
  sorry

end new_barbell_cost_l266_266555


namespace andy_late_l266_266807

theorem andy_late
  (school_start : ℕ := 480) -- 8:00 AM in minutes since midnight
  (normal_travel_time : ℕ := 30)
  (red_lights : ℕ := 4)
  (red_light_wait_time : ℕ := 3)
  (construction_wait_time : ℕ := 10)
  (departure_time : ℕ := 435) -- 7:15 AM in minutes since midnight
  : ((school_start - departure_time) < (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time)) →
    school_start + (normal_travel_time + (red_lights * red_light_wait_time) + construction_wait_time - (school_start - departure_time)) = school_start + 7 :=
by
  -- This skips the proof part
  sorry

end andy_late_l266_266807


namespace walking_speed_proof_l266_266215

-- Definitions based on the problem's conditions
def rest_time_per_period : ℕ := 5
def distance_per_rest : ℕ := 10
def total_distance : ℕ := 50
def total_time : ℕ := 320

-- The man's walking speed
def walking_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The main statement to be proved
theorem walking_speed_proof : 
  walking_speed total_distance ((total_time - ((total_distance / distance_per_rest) * rest_time_per_period)) / 60) = 10 := 
by
  sorry

end walking_speed_proof_l266_266215


namespace negation_of_universal_prop_l266_266731

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end negation_of_universal_prop_l266_266731


namespace student_can_create_16_valid_programs_l266_266101

open Finset

variable (Courses : Finset String := {"English", "Algebra", "Geometry", "History", "Art", "Latin", "Biology"})
variable (MathCourses : Finset String := {"Algebra", "Geometry"})

def valid_programs (sel : Finset String) : Prop :=
  "English" ∈ sel ∧ (sel ∩ MathCourses).nonempty ∧ sel.card = 4

noncomputable def count_valid_programs : ℕ :=
  (Courses.erase "English").powerset.filter (λ c, (c ∩ MathCourses).nonempty ∧ c.card = 3).card

theorem student_can_create_16_valid_programs :
  count_valid_programs = 16 :=
sorry

end student_can_create_16_valid_programs_l266_266101


namespace find_x0_l266_266333

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x_0 : ℝ) (h : f' x_0 = 2) : x_0 = Real.exp 1 :=
by
  sorry

end find_x0_l266_266333


namespace boxes_needed_l266_266045

def num_red_pencils := 45
def num_yellow_pencils := 80
def num_pencils_per_red_box := 15
def num_pencils_per_blue_box := 25
def num_pencils_per_yellow_box := 10
def num_pencils_per_green_box := 30

def num_blue_pencils (x : Nat) := 3 * x + 6
def num_green_pencils (red : Nat) (blue : Nat) := 2 * (red + blue)

def total_boxes_needed : Nat :=
  let red_boxes := num_red_pencils / num_pencils_per_red_box
  let blue_boxes := (num_blue_pencils num_red_pencils) / num_pencils_per_blue_box + 
                    if ((num_blue_pencils num_red_pencils) % num_pencils_per_blue_box) = 0 then 0 else 1
  let yellow_boxes := num_yellow_pencils / num_pencils_per_yellow_box
  let green_boxes := (num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) / num_pencils_per_green_box + 
                     if ((num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) % num_pencils_per_green_box) = 0 then 0 else 1
  red_boxes + blue_boxes + yellow_boxes + green_boxes

theorem boxes_needed : total_boxes_needed = 30 := sorry

end boxes_needed_l266_266045


namespace recurrence_relation_l266_266415

variables {n k : ℕ}

def p : ℕ → ℕ → ℚ := sorry

theorem recurrence_relation (n k : ℕ) (hnk : n ≥ k) :
  p n k = p (n-1) k - (1 / (2^k)) * p (n-k) k + (1 / (2^k)) :=
begin
  sorry
end

end recurrence_relation_l266_266415


namespace gcd_18_30_l266_266278

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266278


namespace minimum_sum_of_original_numbers_l266_266577

theorem minimum_sum_of_original_numbers 
  (m n : ℕ) 
  (h1 : m < n) 
  (h2 : 23 * m - 20 * n = 460) 
  (h3 : ∀ m n, 23 * m - 20 * n = 460 → m < n):
  m + n = 321 :=
sorry

end minimum_sum_of_original_numbers_l266_266577


namespace tablets_taken_l266_266177

theorem tablets_taken (total_time interval_time : ℕ) (h1 : total_time = 60) (h2 : interval_time = 15) : total_time / interval_time = 4 :=
by
  sorry

end tablets_taken_l266_266177


namespace find_integer_b_l266_266063

theorem find_integer_b (z : ℝ) : ∃ b : ℝ, (z^2 - 6*z + 17 = (z - 3)^2 + b) ∧ b = 8 :=
by
  -- The proof would go here
  sorry

end find_integer_b_l266_266063


namespace gcd_18_30_is_6_l266_266267

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266267


namespace gcd_18_30_l266_266276

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266276


namespace ellipse_problem_part1_ellipse_problem_part2_l266_266040

-- Statement of the problem
theorem ellipse_problem_part1 :
  ∃ k : ℝ, (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 → (
    (∃ t > 0, x = t * y + 1) → k = (Real.sqrt 2) / 2)) :=
sorry

theorem ellipse_problem_part2 :
  ∃ S_max : ℝ, ∀ (t : ℝ), (t > 0 → (S_max = (4 * (t^2 + 1)^2) / ((t^2 + 2) * (2 * t^2 + 1)))) → t^2 = 1 → S_max = 16 / 9 :=
sorry

end ellipse_problem_part1_ellipse_problem_part2_l266_266040


namespace part_i_l266_266208

theorem part_i (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end part_i_l266_266208


namespace gcd_of_18_and_30_l266_266247

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266247


namespace intersection_M_N_l266_266138

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l266_266138


namespace gcd_of_18_and_30_l266_266319

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266319


namespace math_problem_l266_266535

variable (a b c d : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
variable (h1 : a^3 + b^3 + 3 * a * b = 1)
variable (h2 : c + d = 1)

theorem math_problem :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 + (d + 1 / d)^3 ≥ 40 := sorry

end math_problem_l266_266535


namespace maria_total_cost_l266_266583

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end maria_total_cost_l266_266583


namespace factorize_expression_l266_266114

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l266_266114


namespace simplify_fraction_l266_266902

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end simplify_fraction_l266_266902


namespace gcd_18_30_l266_266315

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266315


namespace lowest_total_points_l266_266975

-- Five girls and their respective positions
inductive Girl where
  | Fiona
  | Gertrude
  | Hannah
  | India
  | Janice
  deriving DecidableEq, Repr, Inhabited

open Girl

-- Initial position mapping
def initial_position : Girl → Nat
  | Fiona => 1
  | Gertrude => 2
  | Hannah => 3
  | India => 4
  | Janice => 5

-- Final position mapping
def final_position : Girl → Nat
  | Fiona => 3
  | Gertrude => 2
  | Hannah => 5
  | India => 1
  | Janice => 4

-- Define a function to calculate points for given initial and final positions
def points_awarded (g : Girl) : Nat :=
  initial_position g - final_position g

-- Define a function to calculate the total number of points
def total_points : Nat :=
  points_awarded Fiona + points_awarded Gertrude + points_awarded Hannah + points_awarded India + points_awarded Janice

theorem lowest_total_points : total_points = 5 :=
by
  -- Placeholder to skip the proof steps
  sorry

end lowest_total_points_l266_266975


namespace triangle_median_difference_l266_266162

theorem triangle_median_difference
    (A B C D E : Type)
    (BC_len : BC = 10)
    (AD_len : AD = 6)
    (BE_len : BE = 7.5) :
    ∃ X_max X_min : ℝ, 
    X_max = AB^2 + AC^2 + BC^2 ∧ 
    X_min = AB^2 + AC^2 + BC^2 ∧ 
    (X_max - X_min) = 56.25 :=
by
  sorry

end triangle_median_difference_l266_266162


namespace max_m_n_sq_l266_266659

theorem max_m_n_sq (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 1981) (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m * n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_sq_l266_266659


namespace new_barbell_cost_l266_266556

variable (P_old : ℝ) (percentage_increase : ℝ)

theorem new_barbell_cost (h1 : P_old = 250) (h2 : percentage_increase = 0.30) : 
  let P_new := P_old + percentage_increase * P_old in 
  P_new = 325 :=
by
  -- Definitions and statement are correct and the proof is not required.
  sorry

end new_barbell_cost_l266_266556


namespace geometric_sequence_a5_l266_266708

-- Definitions from the conditions
def a1 : ℕ := 2
def a9 : ℕ := 8

-- The statement we need to prove
theorem geometric_sequence_a5 (q : ℝ) (h1 : a1 = 2) (h2 : a9 = a1 * q ^ 8) : a1 * q ^ 4 = 4 := by
  have h_q4 : q ^ 4 = 2 := sorry
  -- Proof continues...
  sorry

end geometric_sequence_a5_l266_266708


namespace evaluate_expression_l266_266367

def my_star (A B : ℕ) : ℕ := (A + B) / 2
def my_hash (A B : ℕ) : ℕ := A * B + 1

theorem evaluate_expression : my_hash (my_star 4 6) 5 = 26 := 
by
  sorry

end evaluate_expression_l266_266367


namespace cosine_identity_l266_266021

variable (α : ℝ)

theorem cosine_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) : 
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cosine_identity_l266_266021


namespace triplet_D_sum_not_one_l266_266476

def triplet_sum_not_equal_to_one : Prop :=
  (1.2 + -0.2 + 0.0 ≠ 1)

theorem triplet_D_sum_not_one : triplet_sum_not_equal_to_one := 
  by
    sorry

end triplet_D_sum_not_one_l266_266476


namespace apples_picked_correct_l266_266562

-- Define the conditions as given in the problem
def apples_given_to_Melanie : ℕ := 27
def apples_left : ℕ := 16

-- Define the problem statement
def total_apples_picked := apples_given_to_Melanie + apples_left

-- Prove that the total apples picked is equal to 43 given the conditions
theorem apples_picked_correct : total_apples_picked = 43 := by
  sorry

end apples_picked_correct_l266_266562


namespace smallest_possible_sum_of_squares_l266_266738

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l266_266738


namespace gcd_18_30_l266_266301

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266301


namespace find_integer_divisible_by_18_and_sqrt_between_30_and_30_5_l266_266509

theorem find_integer_divisible_by_18_and_sqrt_between_30_and_30_5 :
  ∃ x : ℕ, (30^2 ≤ x) ∧ (x ≤ 30.5^2) ∧ (x % 18 = 0) ∧ (x = 900) :=
by
  sorry

end find_integer_divisible_by_18_and_sqrt_between_30_and_30_5_l266_266509


namespace line_equation_l266_266065

variable (x y : ℝ)

theorem line_equation (x1 y1 m : ℝ) (h : x1 = -2 ∧ y1 = 3 ∧ m = 2) :
    -2 * x + y = 1 := by
  sorry

end line_equation_l266_266065


namespace maria_total_cost_l266_266585

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end maria_total_cost_l266_266585


namespace gcd_of_18_and_30_l266_266318

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266318


namespace gcd_18_30_l266_266261

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266261


namespace lower_tap_used_earlier_l266_266209

-- Define the conditions given in the problem
def capacity : ℕ := 36
def midway_capacity : ℕ := capacity / 2
def lower_tap_rate : ℕ := 4  -- minutes per litre
def upper_tap_rate : ℕ := 6  -- minutes per litre

def lower_tap_draw (minutes : ℕ) : ℕ := minutes / lower_tap_rate  -- litres drawn by lower tap
def beer_left_after_draw (initial_amount litres_drawn : ℕ) : ℕ := initial_amount - litres_drawn

-- Define the assistant's drawing condition
def assistant_draw_min : ℕ := 16
def assistant_draw_litres : ℕ := lower_tap_draw assistant_draw_min

-- Define proof statement
theorem lower_tap_used_earlier :
  let initial_amount := capacity
  let litres_when_midway := midway_capacity
  let litres_beer_left := beer_left_after_draw initial_amount assistant_draw_litres
  let additional_litres := litres_beer_left - litres_when_midway
  let time_earlier := additional_litres * upper_tap_rate
  time_earlier = 84 := 
by
  sorry

end lower_tap_used_earlier_l266_266209


namespace gasohol_problem_l266_266794

noncomputable def initial_gasohol_volume (x : ℝ) : Prop :=
  let ethanol_in_initial_mix := 0.05 * x
  let ethanol_to_add := 2
  let total_ethanol := ethanol_in_initial_mix + ethanol_to_add
  let total_volume := x + 2
  0.1 * total_volume = total_ethanol

theorem gasohol_problem (x : ℝ) : initial_gasohol_volume x → x = 36 := by
  intro h
  sorry

end gasohol_problem_l266_266794


namespace divisor_of_44404_l266_266072

theorem divisor_of_44404: ∃ k : ℕ, 2 * 11101 = k ∧ k ∣ (44402 + 2) :=
by
  sorry

end divisor_of_44404_l266_266072


namespace possible_integer_radii_l266_266964

theorem possible_integer_radii (r : ℕ) (h : r < 140) : 
  (3 * 2 * r * π = 2 * 140 * π) → ∃ rs : Finset ℕ, rs.card = 10 := by
  sorry

end possible_integer_radii_l266_266964


namespace gcd_of_18_and_30_l266_266240

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266240


namespace max_k_mono_incr_binom_l266_266332

theorem max_k_mono_incr_binom :
  ∀ (k : ℕ), (k ≤ 11) → 
  (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ k → (Nat.choose 10 (i - 1) < Nat.choose 10 (j - 1))) →
  k = 6 :=
by sorry

end max_k_mono_incr_binom_l266_266332


namespace least_k_l266_266894

noncomputable def u : ℕ → ℝ
| 0 => 1 / 8
| (n + 1) => 3 * u n - 3 * (u n) ^ 2

theorem least_k :
  ∃ k : ℕ, |u k - (1 / 3)| ≤ 1 / 2 ^ 500 ∧ ∀ m < k, |u m - (1 / 3)| > 1 / 2 ^ 500 :=
by
  sorry

end least_k_l266_266894


namespace find_asterisk_l266_266473

theorem find_asterisk : ∃ (x : ℕ), (63 / 21) * (x / 189) = 1 ∧ x = 63 :=
by
  sorry

end find_asterisk_l266_266473


namespace sum_of_inserted_numbers_eq_12_l266_266427

theorem sum_of_inserted_numbers_eq_12 (a b : ℝ) (r d : ℝ) 
  (h1 : a = 2 * r) 
  (h2 : b = 2 * r^2) 
  (h3 : b = a + d) 
  (h4 : 12 = b + d) : 
  a + b = 12 :=
by
  sorry

end sum_of_inserted_numbers_eq_12_l266_266427


namespace total_pieces_in_10_row_triangle_l266_266812

open Nat

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem total_pieces_in_10_row_triangle : 
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  unit_rods + connectors = 231 :=
by
  let unit_rods := arithmetic_sequence_sum 3 3 10
  let connectors := triangular_number 11
  show unit_rods + connectors = 231
  sorry

end total_pieces_in_10_row_triangle_l266_266812


namespace sin_double_angle_l266_266841

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end sin_double_angle_l266_266841


namespace num_of_integers_abs_leq_six_l266_266141

theorem num_of_integers_abs_leq_six (x : ℤ) : 
  (|x - 3| ≤ 6) → ∃ (n : ℕ), n = 13 := 
by 
  sorry

end num_of_integers_abs_leq_six_l266_266141


namespace tilted_rectangle_l266_266094

theorem tilted_rectangle (VWYZ : Type) (YW ZV : ℝ) (ZY VW : ℝ) (W_above_horizontal : ℝ) (Z_height : ℝ) (x : ℝ) :
  YW = 100 → ZV = 100 → ZY = 150 → VW = 150 → W_above_horizontal = 20 → Z_height = (100 + x) →
  x = 67 :=
by
  sorry

end tilted_rectangle_l266_266094


namespace mans_speed_against_current_l266_266798

theorem mans_speed_against_current (V_with_current V_current V_against : ℝ) (h1 : V_with_current = 21) (h2 : V_current = 4.3) : 
  V_against = V_with_current - 2 * V_current := 
sorry

end mans_speed_against_current_l266_266798


namespace inequality_abc_l266_266528

variable {a b c : ℝ}

theorem inequality_abc
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end inequality_abc_l266_266528


namespace transformed_eq_l266_266485

theorem transformed_eq (a b c : ℤ) (h : a > 0) :
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 → (a * x + b)^2 = c) →
  a + b + c = 64 :=
by
  sorry

end transformed_eq_l266_266485


namespace chessboard_problem_proof_l266_266599

variable (n : ℕ)

noncomputable def chessboard_problem : Prop :=
  ∀ (colors : Fin (2 * n) → Fin (2 * n) → Fin n),
  ∃ i₁ i₂ j₁ j₂,
    i₁ ≠ i₂ ∧
    j₁ ≠ j₂ ∧
    colors i₁ j₁ = colors i₁ j₂ ∧
    colors i₂ j₁ = colors i₂ j₂

/-- Given a 2n x 2n chessboard colored with n colors, there exist 2 tiles in either the same column 
or row such that if the colors of both tiles are swapped, then there exists a rectangle where all 
its four corner tiles have the same color. -/
theorem chessboard_problem_proof (n : ℕ) : chessboard_problem n :=
sorry

end chessboard_problem_proof_l266_266599


namespace heating_time_l266_266561

def T_initial: ℝ := 20
def T_final: ℝ := 100
def rate: ℝ := 5

theorem heating_time : (T_final - T_initial) / rate = 16 := by
  sorry

end heating_time_l266_266561


namespace expression_value_l266_266202

theorem expression_value (x y z : ℤ) (hx : x = 26) (hy : y = 3 * x / 2) (hz : z = 11) :
  x - (y - z) - ((x - y) - z) = 22 := 
by
  -- problem statement here
  -- simplified proof goes here
  sorry

end expression_value_l266_266202


namespace monotonic_criteria_l266_266848

noncomputable def monotonic_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → 
  (-2 * x₁^2 + m * x₁ + 1) ≤ (-2 * x₂^2 + m * x₂ + 1)

theorem monotonic_criteria (m : ℝ) : 
  (m ≤ -4 ∨ m ≥ 16) ↔ monotonic_interval m := 
sorry

end monotonic_criteria_l266_266848


namespace smallest_possible_sum_of_squares_l266_266740

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l266_266740


namespace max_bishops_on_chessboard_l266_266774

theorem max_bishops_on_chessboard (N : ℕ) (N_pos: 0 < N) : 
  ∃ max_number : ℕ, max_number = 2 * N - 2 :=
sorry

end max_bishops_on_chessboard_l266_266774


namespace triangle_is_isosceles_l266_266860

-- lean statement
theorem triangle_is_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) : 
  ∃ k : ℝ, a = k ∧ b = k := 
sorry

end triangle_is_isosceles_l266_266860


namespace gcd_18_30_l266_266273

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266273


namespace gcd_of_18_and_30_l266_266326

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l266_266326


namespace unfair_draw_fair_draw_with_suit_hierarchy_l266_266152

noncomputable def deck := {suit : String, rank : ℕ // suit ∈ {"hearts", "diamonds", "clubs", "spades"} ∧ rank ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14}}
def prob_V (v : deck) : ℚ := 1 / 36
def prob_M_given_V (v m : deck) : ℚ := 1 / 35
def higher_rank (v m : deck) : Prop := m.rank > v.rank

-- Prove the draw is unfair
theorem unfair_draw : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank v m then prob_M_given_V v m else 0) < 
  (∑ m in (deck \ {v}), if ¬higher_rank v m then prob_M_given_V v m else 0)) :=
sorry

-- Making the draw fair by introducing suit hierarchy
def suit_order : String → ℕ
| "spades" := 4
| "hearts" := 3
| "diamonds" := 2
| "clubs" := 1
| _ := 0

def higher_rank_with_suit (v m : deck) : Prop :=
  if v.rank = m.rank then suit_order m.suit > suit_order v.suit else m.rank > v.rank

-- Prove introducing suit hierarchy can make the draw fair
theorem fair_draw_with_suit_hierarchy : 
  (∀ v : deck, (∑ m in (deck \ {v}), if higher_rank_with_suit v m then prob_M_given_V v m else 0) = 
  (∑ m in (deck \ {v}), if ¬higher_rank_with_suit v m then prob_M_given_V v m else 0)) :=
sorry

end unfair_draw_fair_draw_with_suit_hierarchy_l266_266152


namespace statement1_statement2_statement3_l266_266237

variable (P_W P_Z : ℝ)

/-- The conditions of the problem: -/
def conditions : Prop :=
  P_W = 0.4 ∧ P_Z = 0.2

/-- Proof of the first statement -/
theorem statement1 (h : conditions P_W P_Z) : 
  P_W * P_Z = 0.08 := 
by sorry

/-- Proof of the second statement -/
theorem statement2 (h : conditions P_W P_Z) :
  P_W * (1 - P_Z) + (1 - P_W) * P_Z = 0.44 := 
by sorry

/-- Proof of the third statement -/
theorem statement3 (h : conditions P_W P_Z) :
  1 - P_W * P_Z = 0.92 := 
by sorry

end statement1_statement2_statement3_l266_266237


namespace gcd_18_30_l266_266308

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266308


namespace factorize_expression_l266_266115

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l266_266115


namespace problem_statement_l266_266353

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - 5 * Real.pi / 6)

theorem problem_statement :
  (∀ (x : ℝ), (∀ a b : ℝ, a < b → x ∈ (Set.Ioc (Real.pi / 6) (2 * Real.pi / 3)) → f(x) < f(b)) → x = (Real.pi / 6) ∨ x = (2 * Real.pi / 3)) →
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
by sorry

end problem_statement_l266_266353


namespace hurdle_distance_l266_266596

theorem hurdle_distance (d : ℝ) : 
  50 + 11 * d + 55 = 600 → d = 45 := by
  sorry

end hurdle_distance_l266_266596


namespace average_score_bounds_l266_266553

/-- Problem data definitions -/
def n_100 : ℕ := 2
def n_90_99 : ℕ := 9
def n_80_89 : ℕ := 17
def n_70_79 : ℕ := 28
def n_60_69 : ℕ := 36
def n_50_59 : ℕ := 7
def n_48 : ℕ := 1

def sum_scores_min : ℕ := (100 * n_100 + 90 * n_90_99 + 80 * n_80_89 + 70 * n_70_79 + 60 * n_60_69 + 50 * n_50_59 + 48)
def sum_scores_max : ℕ := (100 * n_100 + 99 * n_90_99 + 89 * n_80_89 + 79 * n_70_79 + 69 * n_60_69 + 59 * n_50_59 + 48)
def total_people : ℕ := n_100 + n_90_99 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_48

/-- Prove the minimum and maximum average scores. -/
theorem average_score_bounds :
  (sum_scores_min / total_people : ℚ) = 68.88 ∧
  (sum_scores_max / total_people : ℚ) = 77.61 :=
by
  sorry

end average_score_bounds_l266_266553


namespace weight_of_triangular_piece_l266_266803

noncomputable def density_factor (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def square_weight (side_length : ℝ) (weight : ℝ) : ℝ := weight

noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (side_length ^ 2 * Real.sqrt 3) / 4

theorem weight_of_triangular_piece :
  let side_square := 4
  let weight_square := 16
  let side_triangle := 6
  let area_square := side_square ^ 2
  let area_triangle := triangle_area side_triangle
  let density_square := density_factor weight_square area_square
  let weight_triangle := area_triangle * density_square
  abs weight_triangle - 15.59 < 0.01 :=
by
  sorry

end weight_of_triangular_piece_l266_266803


namespace buicks_count_l266_266712

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l266_266712


namespace find_q_from_min_y_l266_266232

variables (a p q m : ℝ)
variable (a_nonzero : a ≠ 0)
variable (min_y : ∀ x : ℝ, a*x^2 + p*x + q ≥ m)

theorem find_q_from_min_y :
  q = m + p^2 / (4 * a) :=
sorry

end find_q_from_min_y_l266_266232


namespace men_in_first_group_l266_266903

theorem men_in_first_group (M : ℕ) (h1 : M * 35 = 7 * 50) : M = 10 := by
  sorry

end men_in_first_group_l266_266903


namespace solution_set_arcsin_inequality_l266_266447

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3

theorem solution_set_arcsin_inequality :
  (∀ x, x ∈ set.Icc (-1 : ℝ) 1 → monotone f) →
  (∀ x, f x > f 0) →
  {x : ℝ | f x > 0} = set.Ioc 0 1 :=
by 
  intros h_mono h_gt
  sorry

end solution_set_arcsin_inequality_l266_266447


namespace gcd_of_18_and_30_l266_266242

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266242


namespace shift_sine_graph_l266_266896

theorem shift_sine_graph (x : ℝ) : 
  (∃ θ : ℝ, θ = (5 * Real.pi) / 4 ∧ 
  y = Real.sin (x - Real.pi / 4) → y = Real.sin (x + θ) 
  ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := sorry

end shift_sine_graph_l266_266896


namespace equivalent_statements_l266_266788

-- Definitions based on the problem
def is_not_negative (x : ℝ) : Prop := x >= 0
def is_not_positive (x : ℝ) : Prop := x <= 0
def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

-- The main theorem statement
theorem equivalent_statements (x : ℝ) : 
  (is_not_negative x → is_not_positive (x^2)) ↔ (is_positive (x^2) → is_negative x) :=
by
  sorry

end equivalent_statements_l266_266788


namespace problem_statement_l266_266568

def S : Set Nat := {x | x ∈ Finset.range 13 \ Finset.range 1}

def n : Nat :=
  4^12 - 3 * 3^12 + 3 * 2^12

theorem problem_statement : n % 1000 = 181 :=
by
  sorry

end problem_statement_l266_266568


namespace find_integer_divisible_by_18_and_sqrt_between_30_and_30_5_l266_266508

theorem find_integer_divisible_by_18_and_sqrt_between_30_and_30_5 :
  ∃ x : ℕ, (30^2 ≤ x) ∧ (x ≤ 30.5^2) ∧ (x % 18 = 0) ∧ (x = 900) :=
by
  sorry

end find_integer_divisible_by_18_and_sqrt_between_30_and_30_5_l266_266508


namespace profit_8000_l266_266618

noncomputable def profit (selling_price increase : ℝ) : ℝ :=
  (selling_price - 40 + increase) * (500 - 10 * increase)

theorem profit_8000 (increase : ℝ) :
  profit 50 increase = 8000 →
  ((increase = 10 ∧ (50 + increase = 60) ∧ (500 - 10 * increase = 400)) ∨ 
   (increase = 30 ∧ (50 + increase = 80) ∧ (500 - 10 * increase = 200))) :=
by
  sorry

end profit_8000_l266_266618


namespace largest_divisor_60_36_divisible_by_3_l266_266601

theorem largest_divisor_60_36_divisible_by_3 : 
  ∃ x, (x ∣ 60) ∧ (x ∣ 36) ∧ (3 ∣ x) ∧ (∀ y, (y ∣ 60) → (y ∣ 36) → (3 ∣ y) → y ≤ x) ∧ x = 12 :=
sorry

end largest_divisor_60_36_divisible_by_3_l266_266601


namespace polynomial_identity_l266_266179

open Polynomial

-- Definition of the non-zero polynomial of interest
noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.C a * (Polynomial.X ^ 3 - Polynomial.X)

-- Theorem stating that, for all x, the given equation holds for the polynomial p
theorem polynomial_identity (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, (x - 1) * (p a).eval (x + 1) - (x + 2) * (p a).eval x = 0 :=
by
  sorry

end polynomial_identity_l266_266179


namespace a_increasing_l266_266765

noncomputable def a : ℕ → ℝ
| 0     := 1 / 5
| (n+1) := 2^n - 3 * a n

theorem a_increasing (n : ℕ) : a (n + 1) > a n := 
by {
  sorry
}

end a_increasing_l266_266765


namespace factorize_expression_l266_266116

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorize_expression_l266_266116


namespace average_salary_of_technicians_l266_266031

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (avg_salary_all_workers : ℕ)
  (total_technicians : ℕ)
  (avg_salary_non_technicians : ℕ)
  (h1 : total_workers = 18)
  (h2 : avg_salary_all_workers = 8000)
  (h3 : total_technicians = 6)
  (h4 : avg_salary_non_technicians = 6000) :
  (72000 / total_technicians) = 12000 := 
  sorry

end average_salary_of_technicians_l266_266031


namespace game_points_l266_266550

noncomputable def total_points (total_enemies : ℕ) (red_enemies : ℕ) (blue_enemies : ℕ) 
  (enemies_defeated : ℕ) (points_per_enemy : ℕ) (bonus_points : ℕ) 
  (hits_taken : ℕ) (points_lost_per_hit : ℕ) : ℕ :=
  (enemies_defeated * points_per_enemy + if enemies_defeated > 0 ∧ enemies_defeated < total_enemies then bonus_points else 0) - (hits_taken * points_lost_per_hit)

theorem game_points (h : total_points 6 3 3 4 3 5 2 2 = 13) : Prop := sorry

end game_points_l266_266550


namespace min_value_l266_266335

-- Define the conditions
variables (x y : ℝ)
-- Assume x and y are in the positive real numbers
axiom pos_x : 0 < x
axiom pos_y : 0 < y
-- Given equation
axiom eq1 : x + 2 * y = 2 * x * y

-- The goal is to prove that the minimum value of 3x + 4y is 5 + 2sqrt(6)
theorem min_value (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (eq1 : x + 2 * y = 2 * x * y) : 
  3 * x + 4 * y ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end min_value_l266_266335


namespace max_possible_percent_error_in_garden_area_l266_266725

open Real

theorem max_possible_percent_error_in_garden_area :
  ∃ (error_max : ℝ), error_max = 21 :=
by
  -- Given conditions
  let accurate_diameter := 30
  let max_error_percent := 10

  -- Defining lower and upper bounds for the diameter
  let lower_diameter := accurate_diameter - accurate_diameter * (max_error_percent / 100)
  let upper_diameter := accurate_diameter + accurate_diameter * (max_error_percent / 100)

  -- Calculating the exact and potential extreme areas
  let exact_area := π * (accurate_diameter / 2) ^ 2
  let lower_area := π * (lower_diameter / 2) ^ 2
  let upper_area := π * (upper_diameter / 2) ^ 2

  -- Calculating the percent errors
  let lower_error_percent := ((exact_area - lower_area) / exact_area) * 100
  let upper_error_percent := ((upper_area - exact_area) / exact_area) * 100

  -- We need to show the maximum error is 21%
  use upper_error_percent -- which should be 21% according to the problem statement
  sorry -- proof goes here

end max_possible_percent_error_in_garden_area_l266_266725


namespace sin_monotonic_increasing_f_symmetric_axes_find_value_l266_266347

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi * 5 / 6)

theorem sin_monotonic_increasing (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) :
  Monotone (λ x, f x) (Ioo a b) := 
sorry

theorem f_symmetric_axes (a b : ℝ) (h1 : a = Real.pi / 6) (h2 : b = 2 * Real.pi / 3) 
  (x y : ℝ) (hx : x = a + (b - a) * k) (hy : y = a + (b - a) * (-k)) : 
  f x = f y :=
sorry

theorem find_value :
  f (-5 * Real.pi / 12) = Real.sqrt 3 / 2 :=
sorry

end sin_monotonic_increasing_f_symmetric_axes_find_value_l266_266347


namespace find_f_of_neg_5_pi_over_12_l266_266348

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - (5 * π / 6))

theorem find_f_of_neg_5_pi_over_12 :
  (∀ x ∈ (set.Ioo (π / 6) (2 * π / 3)), (0 : ℝ) < (f x - f (x - 1))) ∧
  (∀ x, f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  f (-5 * π / 12) = √3 / 2 :=
by
  sorry

end find_f_of_neg_5_pi_over_12_l266_266348


namespace factorize_expression_l266_266113

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l266_266113


namespace number_of_salads_bought_l266_266615

variable (hot_dogs_cost : ℝ := 5 * 1.50)
variable (initial_money : ℝ := 2 * 10)
variable (change_given_back : ℝ := 5)
variable (total_spent : ℝ := initial_money - change_given_back)
variable (salad_cost : ℝ := 2.50)

theorem number_of_salads_bought : (total_spent - hot_dogs_cost) / salad_cost = 3 := 
by 
  sorry

end number_of_salads_bought_l266_266615


namespace sufficient_but_not_necessary_condition_l266_266451

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → (m = 1) ∨ (m = -2)) → (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → false) :=
by
  intros hm h_perp h
  sorry

end sufficient_but_not_necessary_condition_l266_266451


namespace largest_of_three_l266_266464

theorem largest_of_three (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : ab + ac + bc = -8) 
  (h3 : abc = -20) : 
  max a (max b c) = (1 + Real.sqrt 41) / 2 := 
by 
  sorry

end largest_of_three_l266_266464


namespace probability_recurrence_relation_l266_266401

theorem probability_recurrence_relation (n k : ℕ) (h : k < n) :
  ∀ (p : ℕ → ℕ → ℝ), p n k = p (n-1) k - (1 / (2:ℝ)^k) * p (n-k) k + 1 / (2:ℝ)^k := 
sorry

end probability_recurrence_relation_l266_266401


namespace height_of_parallelogram_l266_266516

theorem height_of_parallelogram (area base height : ℝ) (h1 : area = 240) (h2 : base = 24) : height = 10 :=
by
  sorry

end height_of_parallelogram_l266_266516


namespace fraction_absent_l266_266917

theorem fraction_absent (total_students present_students : ℕ) (h1 : total_students = 28) (h2 : present_students = 20) : 
  (total_students - present_students) / total_students = 2 / 7 :=
by
  sorry

end fraction_absent_l266_266917


namespace gcd_of_18_and_30_l266_266246

-- Define the numbers
def num1 := 18
def num2 := 30

-- State the GCD property
theorem gcd_of_18_and_30 : Nat.gcd num1 num2 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266246


namespace gcd_of_18_and_30_l266_266292

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266292


namespace find_integer_a_l266_266612

-- Definitions based on the conditions
def in_ratio (x y z : ℕ) := ∃ k : ℕ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k
def satisfies_equation (z : ℕ) (a : ℕ) := z = 30 * a - 15

-- The proof problem statement
theorem find_integer_a (x y z : ℕ) (a : ℕ) :
  in_ratio x y z →
  satisfies_equation z a →
  (∃ a : ℕ, a = 4) :=
by
  intros h1 h2
  sorry

end find_integer_a_l266_266612


namespace shaded_area_floor_l266_266790

noncomputable def area_of_white_quarter_circle : ℝ := Real.pi / 4

noncomputable def area_of_white_per_tile : ℝ := 4 * area_of_white_quarter_circle

noncomputable def area_of_tile : ℝ := 4

noncomputable def shaded_area_per_tile : ℝ := area_of_tile - area_of_white_per_tile

noncomputable def number_of_tiles : ℕ := by
  have floor_area : ℝ := 12 * 15
  have tile_area : ℝ := 2 * 2
  exact Nat.floor (floor_area / tile_area)

noncomputable def total_shaded_area (num_tiles : ℕ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_floor : total_shaded_area number_of_tiles = 180 - 45 * Real.pi := by
  sorry

end shaded_area_floor_l266_266790


namespace distinct_necklaces_count_l266_266103

open Classical

/-- Using 5 beads and 3 different colors, the total number of distinct necklaces,
  considering the dihedral group D5 symmetries, is 39. -/
theorem distinct_necklaces_count : 
  ∃ N : ℕ, N = 39 ∧
    ∀ beaded_necklaces : fin 3 ^ 5, 
    true := -- Assuming we have some definite combinatorial structure beaded_necklaces 
  begin
    let D5 := dihedralGroup 5,
    have H : |D5| = 10 := by sorry,
    let count_fixed_points := λ (g : D5), (fin 3) ^ (fixedPoints g),
    let total_count := (1 / |D5|.toReal) * ∑ g in D5, count_fixed_points g,
    exact ⟨39, rfl, sorry⟩
  end

end distinct_necklaces_count_l266_266103


namespace problem_1_problem_2_problem_3_l266_266007

variable (α : ℝ)
variable (tan_alpha_two : Real.tan α = 2)

theorem problem_1 : (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α + Real.sin α) = 8 / 5 :=
by
  sorry

theorem problem_2 : (Real.cos α ^ 2 + Real.sin α * Real.cos α) / (2 * Real.sin α * Real.cos α + Real.sin α ^ 2) = 3 / 8 :=
by
  sorry

theorem problem_3 : (Real.sin α ^ 2 - Real.sin α * Real.cos α + 2) = 12 / 5 :=
by
  sorry

end problem_1_problem_2_problem_3_l266_266007


namespace heating_time_l266_266560

def T_initial: ℝ := 20
def T_final: ℝ := 100
def rate: ℝ := 5

theorem heating_time : (T_final - T_initial) / rate = 16 := by
  sorry

end heating_time_l266_266560


namespace melanie_missed_games_l266_266726

-- Define the total number of soccer games played and the number attended by Melanie
def total_games : ℕ := 64
def attended_games : ℕ := 32

-- Statement to be proven
theorem melanie_missed_games : total_games - attended_games = 32 := by
  -- Placeholder for the proof
  sorry

end melanie_missed_games_l266_266726


namespace second_container_clay_l266_266212

theorem second_container_clay :
  let h1 := 3
  let w1 := 5
  let l1 := 7
  let clay1 := 105
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let V1 := h1 * w1 * l1
  let V2 := h2 * w2 * l2
  V1 = clay1 →
  V2 = 6 * V1 →
  V2 = 630 :=
by
  intros
  sorry

end second_container_clay_l266_266212


namespace flowerbed_seeds_l266_266888

theorem flowerbed_seeds (n_fbeds n_seeds_per_fbed total_seeds : ℕ)
    (h1 : n_fbeds = 8)
    (h2 : n_seeds_per_fbed = 4) :
    total_seeds = n_fbeds * n_seeds_per_fbed := by
  sorry

end flowerbed_seeds_l266_266888


namespace find_marks_in_chemistry_l266_266108

theorem find_marks_in_chemistry
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_physics : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (marks_english_eq : marks_english = 86)
  (marks_math_eq : marks_math = 85)
  (marks_physics_eq : marks_physics = 92)
  (marks_biology_eq : marks_biology = 95)
  (average_marks_eq : average_marks = 89)
  (num_subjects_eq : num_subjects = 5) : 
  ∃ marks_chemistry : ℕ, marks_chemistry = 87 :=
by
  sorry

end find_marks_in_chemistry_l266_266108


namespace min_sum_intercepts_of_line_l266_266024

theorem min_sum_intercepts_of_line (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : a + b = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_intercepts_of_line_l266_266024


namespace gcd_18_30_l266_266307

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266307


namespace sin_eq_sqrt3_div_2_l266_266832

open Real

theorem sin_eq_sqrt3_div_2 (theta : ℝ) :
  sin theta = (sqrt 3) / 2 ↔ (∃ k : ℤ, theta = π/3 + 2*k*π ∨ theta = 2*π/3 + 2*k*π) :=
by
  sorry

end sin_eq_sqrt3_div_2_l266_266832


namespace minimum_percentage_increase_in_mean_replacing_with_primes_l266_266470

def mean (S : List ℤ) : ℚ :=
  (S.sum : ℚ) / S.length

noncomputable def percentage_increase (original new : ℚ) : ℚ :=
  ((new - original) / original) * 100

theorem minimum_percentage_increase_in_mean_replacing_with_primes :
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  percentage_increase (mean F) (mean G) = 100 :=
by {
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  sorry 
}

end minimum_percentage_increase_in_mean_replacing_with_primes_l266_266470


namespace sum_of_dimensions_l266_266490

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 30) (h2 : A * C = 60) (h3 : B * C = 90) : A + B + C = 24 := 
sorry

end sum_of_dimensions_l266_266490


namespace solve_eq_l266_266735

theorem solve_eq (x : ℝ) : x^6 - 19*x^3 = 216 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_eq_l266_266735


namespace find_triplets_satisfying_equation_l266_266665

theorem find_triplets_satisfying_equation :
  ∃ (x y z : ℕ), x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧ (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triplets_satisfying_equation_l266_266665


namespace correct_value_wrongly_copied_l266_266761

theorem correct_value_wrongly_copied 
  (mean_initial : ℕ)
  (mean_correct : ℕ)
  (wrong_value : ℕ) 
  (n : ℕ) 
  (initial_mean : mean_initial = 250)
  (correct_mean : mean_correct = 251)
  (wrongly_copied : wrong_value = 135)
  (number_of_values : n = 30) : 
  ∃ x : ℕ, x = 165 := 
by
  use (wrong_value + (mean_correct - mean_initial) * n / n)
  sorry

end correct_value_wrongly_copied_l266_266761


namespace sarahs_trip_length_l266_266442

noncomputable def sarahsTrip (x : ℝ) : Prop :=
  x / 4 + 15 + x / 3 = x

theorem sarahs_trip_length : ∃ x : ℝ, sarahsTrip x ∧ x = 36 := by
  -- There should be a proof here, but it's omitted as per the task instructions
  sorry

end sarahs_trip_length_l266_266442


namespace equation_of_parallel_line_l266_266515

theorem equation_of_parallel_line (A : ℝ × ℝ) (c : ℝ) : 
  A = (-1, 0) → (∀ x y, 2 * x - y + 1 = 0 → 2 * x - y + c = 0) → 
  2 * (-1) - 0 + c = 0 → c = 2 :=
by
  intros A_coord parallel_line point_on_line
  sorry

end equation_of_parallel_line_l266_266515


namespace geom_seq_sum_five_terms_l266_266455

theorem geom_seq_sum_five_terms (a : ℕ → ℝ) (q : ℝ) 
    (h_pos : ∀ n, 0 < a n)
    (h_a2 : a 2 = 8) 
    (h_arith : 2 * a 4 - a 3 = a 3 - 4 * a 5) :
    a 1 * (1 - q^5) / (1 - q) = 31 :=
by
    sorry

end geom_seq_sum_five_terms_l266_266455


namespace final_result_l266_266121

def a : ℕ := 2548
def b : ℕ := 364
def hcd := Nat.gcd a b
def result := hcd + 8 - 12

theorem final_result : result = 360 := by
  sorry

end final_result_l266_266121


namespace second_number_more_than_first_l266_266421

-- Definitions of A and B based on the given ratio
def A : ℚ := 7 / 56
def B : ℚ := 8 / 56

-- Proof statement
theorem second_number_more_than_first : ((B - A) / A) * 100 = 100 / 7 :=
by
  -- skipped the proof
  sorry

end second_number_more_than_first_l266_266421


namespace exponentiation_power_rule_l266_266935

theorem exponentiation_power_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_power_rule_l266_266935


namespace inequality_x_y_z_l266_266054

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := 
by sorry

end inequality_x_y_z_l266_266054


namespace fatous_lemma_inequality_fatous_lemma_finite_measure_probability_measure_inequality_continuity_property_l266_266569

noncomputable theory

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] (μ P : Measure Ω) 
  (A : ℕ → Set Ω) [Countable ℕ]

-- Problem 1: Fatou's Lemma for measures (a)
theorem fatous_lemma_inequality (μ : Measure Ω) (A : ℕ → Set Ω) :
  μ (Filter.UnderLim A) ≤ Filter.UnderLim (fun n => μ (A n)) := sorry

-- Problem 2: Fatou's Lemma for measures (b)
theorem fatous_lemma_finite_measure (μ : Measure Ω) (A : ℕ → Set Ω) [finite_measure μ] :
  μ (Filter.OverLim A) ≥ Filter.OverLim (fun n => μ (A n)) := sorry

-- Problem 3: Probability measure inequality
theorem probability_measure_inequality (P : Measure Ω) [ProbabilityMeasure P] 
  (A : ℕ → Set Ω) :
  P (Filter.UnderLim A) ≤ Filter.UnderLim (fun n => P (A n)) ∧
  Filter.UnderLim (fun n => P (A n)) ≤ Filter.OverLim (fun n => P (A n)) ∧
  Filter.OverLim (fun n => P (A n)) ≤ P (Filter.OverLim A) := sorry

-- Problem 4: Continuity property for probability measures
theorem continuity_property (P : Measure Ω) [ProbabilityMeasure P] 
  (A : ℕ → Set Ω) (B : Set Ω) (h : Filter.OverLim A = B) (h' : Filter.UnderLim A = B) :
  P B = Filter.Lim (fun n => P (A n)) := sorry

end fatous_lemma_inequality_fatous_lemma_finite_measure_probability_measure_inequality_continuity_property_l266_266569


namespace distinct_solutions_difference_l266_266574

theorem distinct_solutions_difference (r s : ℝ) (hr : (r - 5) * (r + 5) = 25 * r - 125)
  (hs : (s - 5) * (s + 5) = 25 * s - 125) (neq : r ≠ s) (hgt : r > s) : r - s = 15 := by
  sorry

end distinct_solutions_difference_l266_266574


namespace max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l266_266341

theorem max_lg_sum_eq_one {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ u, u = Real.log x + Real.log y → u ≤ 1 :=
sorry

theorem min_inv_sum_eq_specific_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ v, v = (1 / x) + (1 / y) → v ≥ (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l266_266341


namespace rate_of_change_l266_266792

noncomputable def radius : ℝ := 12
noncomputable def θ (t : ℝ) : ℝ := (38 + 5 * t) * (Real.pi / 180)
noncomputable def area (t : ℝ) : ℝ := (1/2) * radius^2 * θ t

theorem rate_of_change (t : ℝ) : deriv area t = 2 * Real.pi :=
by
  sorry

end rate_of_change_l266_266792


namespace gcd_18_30_l266_266259

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266259


namespace gcd_18_30_l266_266306

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l266_266306


namespace gcd_18_30_is_6_l266_266268

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266268


namespace max_xy_l266_266381

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l266_266381


namespace gcd_18_30_l266_266280

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266280


namespace gcd_18_30_l266_266256

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266256


namespace no_such_function_l266_266111

theorem no_such_function :
  ¬ (∃ f : ℕ → ℕ, ∀ n ≥ 2, f (f (n - 1)) = f (n + 1) - f (n)) :=
sorry

end no_such_function_l266_266111


namespace geo_arith_sequences_sum_first_2n_terms_l266_266160

variables (n : ℕ)

-- Given conditions in (a)
def common_ratio : ℕ := 3
def arithmetic_diff : ℕ := 2

-- The sequences provided in the solution (b)
def a_n (n : ℕ) : ℕ := common_ratio ^ n
def b_n (n : ℕ) : ℕ := 2 * n + 1

-- Sum formula for geometric series up to 2n terms
def S_2n (n : ℕ) : ℕ := (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n

theorem geo_arith_sequences :
  a_n n = common_ratio ^ n
  ∨ b_n n = 2 * n + 1 := sorry

theorem sum_first_2n_terms :
  S_2n n = (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n := sorry

end geo_arith_sequences_sum_first_2n_terms_l266_266160


namespace flowers_total_l266_266936

theorem flowers_total (yoojung_flowers : ℕ) (namjoon_flowers : ℕ)
 (h1 : yoojung_flowers = 32)
 (h2 : yoojung_flowers = 4 * namjoon_flowers) :
  yoojung_flowers + namjoon_flowers = 40 := by
  sorry

end flowers_total_l266_266936


namespace tangent_line_at_P_l266_266984

/-- Define the center of the circle as the origin and point P --/
def center : ℝ × ℝ := (0, 0)

def P : ℝ × ℝ := (1, 2)

/-- Define the circle with radius squared r², where the radius passes through point P leading to r² = 5 --/
def circle_equation (x y : ℝ) : Prop := x * x + y * y = 5

/-- Define the condition that point P lies on the circle centered at the origin --/
def P_on_circle : Prop := circle_equation P.1 P.2

/-- Define what it means for a line to be the tangent at point P --/
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem tangent_line_at_P : P_on_circle → ∃ x y, tangent_line x y :=
by {
  sorry
}

end tangent_line_at_P_l266_266984


namespace remaining_shoes_to_sell_l266_266883

def shoes_goal : Nat := 80
def shoes_sold_last_week : Nat := 27
def shoes_sold_this_week : Nat := 12

theorem remaining_shoes_to_sell : shoes_goal - (shoes_sold_last_week + shoes_sold_this_week) = 41 :=
by
  sorry

end remaining_shoes_to_sell_l266_266883


namespace shoveling_driveway_time_l266_266780

theorem shoveling_driveway_time (S : ℝ) (Wayne_rate : ℝ) (combined_rate : ℝ) :
  (S = 1 / 7) → (Wayne_rate = 6 * S) → (combined_rate = Wayne_rate + S) → (combined_rate = 1) :=
by { sorry }

end shoveling_driveway_time_l266_266780


namespace car_miles_per_tankful_in_city_l266_266088

-- Define constants for the given values
def miles_per_tank_on_highway : ℝ := 462
def fewer_miles_per_gallon : ℝ := 15
def miles_per_gallon_in_city : ℝ := 40

-- Prove the car traveled 336 miles per tankful in the city
theorem car_miles_per_tankful_in_city :
  (miles_per_tank_on_highway / (miles_per_gallon_in_city + fewer_miles_per_gallon)) * miles_per_gallon_in_city = 336 := 
by
  sorry

end car_miles_per_tankful_in_city_l266_266088


namespace gcd_18_30_l266_266274

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l266_266274


namespace compound_p_and_q_false_l266_266981

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1) /- The function y = a^x is monotonically decreasing. -/
def q : Prop := (a > 1/2) /- The function y = log(ax^2 - x + a) has the range R. -/

theorem compound_p_and_q_false : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ (a > 1) :=
by {
  -- this part will contain the proof steps, omitted here.
  sorry
}

end compound_p_and_q_false_l266_266981


namespace beautiful_point_coordinates_l266_266544

-- Define a "beautiful point"
def is_beautiful_point (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = P.1 * P.2

theorem beautiful_point_coordinates (M : ℝ × ℝ) : 
  is_beautiful_point M ∧ abs M.1 = 2 → 
  (M = (2, 2) ∨ M = (-2, 2/3)) :=
by sorry

end beautiful_point_coordinates_l266_266544


namespace total_weight_correct_l266_266047

def Marco_strawberry_weight : ℕ := 15
def Dad_strawberry_weight : ℕ := 22
def total_strawberry_weight : ℕ := Marco_strawberry_weight + Dad_strawberry_weight

theorem total_weight_correct :
  total_strawberry_weight = 37 :=
by
  sorry

end total_weight_correct_l266_266047


namespace range_m_l266_266995

open Set

noncomputable def A : Set ℝ := { x : ℝ | -5 ≤ x ∧ x ≤ 3 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m + 3 }

theorem range_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ m ≤ 0 :=
by
  sorry

end range_m_l266_266995


namespace leila_armchairs_l266_266563

theorem leila_armchairs :
  ∀ {sofa_price armchair_price coffee_table_price total_invoice armchairs : ℕ},
  sofa_price = 1250 →
  armchair_price = 425 →
  coffee_table_price = 330 →
  total_invoice = 2430 →
  1 * sofa_price + armchairs * armchair_price + 1 * coffee_table_price = total_invoice →
  armchairs = 2 :=
by
  intros sofa_price armchair_price coffee_table_price total_invoice armchairs
  intros h1 h2 h3 h4 h_eq
  sorry

end leila_armchairs_l266_266563


namespace family_members_to_pay_l266_266974

theorem family_members_to_pay :
  (∃ (n : ℕ), 
    5 * 12 = 60 ∧ 
    60 * 2 = 120 ∧ 
    120 / 10 = 12 ∧ 
    12 * 2 = 24 ∧ 
    24 / 4 = n ∧ 
    n = 6) :=
by
  sorry

end family_members_to_pay_l266_266974


namespace ninth_term_of_sequence_is_4_l266_266817

-- Definition of the first term and common ratio
def a1 : ℚ := 4
def r : ℚ := 1

-- Definition of the nth term of a geometric sequence
def a (n : ℕ) : ℚ := a1 * r^(n-1)

-- Proof that the ninth term of the sequence is 4
theorem ninth_term_of_sequence_is_4 : a 9 = 4 := by
  sorry

end ninth_term_of_sequence_is_4_l266_266817


namespace find_number_l266_266361

theorem find_number (x : ℕ) (h : x + 56 = 110) : x = 54 :=
sorry

end find_number_l266_266361


namespace gcd_18_30_is_6_l266_266271

def gcd_18_30 : ℕ :=
  gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 :=
by {
  -- The step here will involve using properties of gcd and prime factorization,
  -- but we are given the result directly for the purpose of this task.
  sorry
}

end gcd_18_30_is_6_l266_266271


namespace sufficient_not_necessary_condition_l266_266683

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h : a > |b|) : a^2 > b^2 :=
by 
  sorry

end sufficient_not_necessary_condition_l266_266683


namespace letters_with_line_not_dot_l266_266861

-- Defining the conditions
def num_letters_with_dot_and_line : ℕ := 9
def num_letters_with_dot_only : ℕ := 7
def total_letters : ℕ := 40

-- Proving the number of letters with a straight line but not a dot
theorem letters_with_line_not_dot :
  (num_letters_with_dot_and_line + num_letters_with_dot_only + x = total_letters) → x = 24 :=
by
  intros h
  sorry

end letters_with_line_not_dot_l266_266861


namespace complement_U_M_l266_266016

noncomputable def U : Set ℝ := {x : ℝ | x > 0}

noncomputable def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}

theorem complement_U_M : (U \ M) = {x : ℝ | x ≥ 2} := 
by
  sorry

end complement_U_M_l266_266016


namespace loss_percentage_l266_266362

/-
Books Problem:
Determine the loss percentage on the first book given:
1. The cost of the first book (C1) is Rs. 280.
2. The total cost of two books is Rs. 480.
3. The second book is sold at a gain of 19%.
4. Both books are sold at the same price.
-/

theorem loss_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 = 280)
  (h2 : C1 + C2 = 480) (h3 : SP2 = C2 * 1.19) (h4 : SP1 = SP2) : 
  (C1 - SP1) / C1 * 100 = 15 := 
by
  sorry

end loss_percentage_l266_266362


namespace domain_tan_x_plus_pi_over_3_l266_266906

open Real Set

theorem domain_tan_x_plus_pi_over_3 :
  ∀ x : ℝ, ¬ (∃ k : ℤ, x = k * π + π / 6) ↔ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = k * π + π / 6} :=
by {
  sorry
}

end domain_tan_x_plus_pi_over_3_l266_266906


namespace required_sand_volume_is_five_l266_266097

noncomputable def length : ℝ := 10
noncomputable def depth_cm : ℝ := 50
noncomputable def depth_m : ℝ := depth_cm / 100  -- converting cm to m
noncomputable def width : ℝ := 2
noncomputable def total_volume : ℝ := length * depth_m * width
noncomputable def current_volume : ℝ := total_volume / 2
noncomputable def additional_sand : ℝ := total_volume - current_volume

theorem required_sand_volume_is_five : additional_sand = 5 :=
by sorry

end required_sand_volume_is_five_l266_266097


namespace find_six_digit_numbers_l266_266098

variable (m n : ℕ)

-- Definition that the original number becomes six-digit when multiplied by 4
def is_six_digit (x : ℕ) : Prop := x ≥ 100000 ∧ x < 1000000

-- Conditions
def original_number := 100 * m + n
def new_number := 10000 * n + m
def satisfies_conditions (m n : ℕ) : Prop :=
  is_six_digit (100 * m + n) ∧
  is_six_digit (10000 * n + m) ∧
  4 * (100 * m + n) = 10000 * n + m

-- Theorem statement
theorem find_six_digit_numbers (h₁ : satisfies_conditions 1428 57)
                               (h₂ : satisfies_conditions 1904 76)
                               (h₃ : satisfies_conditions 2380 95) :
  ∃ m n, satisfies_conditions m n :=
  sorry -- Proof omitted

end find_six_digit_numbers_l266_266098


namespace central_angle_of_sector_in_unit_circle_with_area_1_is_2_l266_266154

theorem central_angle_of_sector_in_unit_circle_with_area_1_is_2 :
  ∀ (θ : ℝ), (∀ (r : ℝ), (r = 1) → (1 / 2 * r^2 * θ = 1) → θ = 2) :=
by
  intros θ r hr h
  sorry

end central_angle_of_sector_in_unit_circle_with_area_1_is_2_l266_266154


namespace value_ranges_l266_266009

theorem value_ranges 
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_eq1 : 3 * a + 2 * b + c = 5)
  (h_eq2 : 2 * a + b - 3 * c = 1) :
  (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧ 
  (-5 / 7 ≤ (3 * a + b - 7 * c) ∧ (3 * a + b - 7 * c) ≤ -1 / 11) :=
by 
  sorry

end value_ranges_l266_266009


namespace solve_x_l266_266029

theorem solve_x (x : ℝ) (h : 2 - 2 / (1 - x) = 2 / (1 - x)) : x = -2 := 
by
  sorry

end solve_x_l266_266029


namespace right_rectangular_prism_volume_l266_266610

theorem right_rectangular_prism_volume
    (a b c : ℝ)
    (H1 : a * b = 56)
    (H2 : b * c = 63)
    (H3 : a * c = 72)
    (H4 : c = 3 * a) :
    a * b * c = 2016 * Real.sqrt 6 :=
by
  sorry

end right_rectangular_prism_volume_l266_266610


namespace correct_exponentiation_l266_266475

theorem correct_exponentiation (a : ℝ) : (-2 * a^3) ^ 4 = 16 * a ^ 12 :=
by sorry

end correct_exponentiation_l266_266475


namespace largest_integer_solution_l266_266079

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 3 + 4 / 5 < 5 / 3) : x ≤ 2 :=
sorry

end largest_integer_solution_l266_266079


namespace solve_for_y_l266_266500

def G (a y c d : ℕ) := 3 ^ y + 6 * d

theorem solve_for_y (a c d : ℕ) (h1 : G a 2 c d = 735) : 2 = 2 := 
by
  sorry

end solve_for_y_l266_266500


namespace find_principal_l266_266955

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h1 : SI = 4020.75) (h2 : R = 0.0875) (h3 : T = 5.5) (h4 : SI = P * R * T) : 
  P = 8355.00 :=
sorry

end find_principal_l266_266955


namespace vector_subtraction_l266_266996

open Real

def vector_a : (ℝ × ℝ) := (3, 2)
def vector_b : (ℝ × ℝ) := (0, -1)

theorem vector_subtraction : 
  3 • vector_b - vector_a = (-3, -5) :=
by 
  -- Proof needs to be written here.
  sorry

end vector_subtraction_l266_266996


namespace speed_of_stream_l266_266478

theorem speed_of_stream :
  ∃ (v : ℝ), (∀ (swim_speed : ℝ), swim_speed = 1.5 → 
    (∀ (time_upstream : ℝ) (time_downstream : ℝ), 
      time_upstream = 2 * time_downstream → 
      (1.5 + v) / (1.5 - v) = 2)) → v = 0.5 :=
sorry

end speed_of_stream_l266_266478


namespace family_groups_correct_l266_266915

structure Child where
  name : String
  eyeColor : String
  hairColor : String

def Liam   := Child.mk "Liam" "Green" "Black"
def Mia    := Child.mk "Mia" "Brown" "Red"
def Noah   := Child.mk "Noah" "Green" "Red"
def Eva    := Child.mk "Eva" "Brown" "Black"
def Oliver := Child.mk "Oliver" "Green" "Black"
def Lucy   := Child.mk "Lucy" "Brown" "Red"
def Jack   := Child.mk "Jack" "Green" "Red"

def family1 : Set Child := {Liam, Oliver}
def family2 : Set Child := {Mia, Lucy}
def family3 : Set Child := {Noah, Jack}

theorem family_groups_correct :
  (∀ f ∈ {family1, family2, family3}, ∃ c1 ∈ f, ∃ c2 ∈ f, 
      (c1.eyeColor = c2.eyeColor) ∨ (c1.hairColor = c2.hairColor)) ∧
  (∀ c ∈ {Liam, Mia, Noah, Eva, Oliver, Lucy, Jack},
      c ∈ family1 ∨ c ∈ family2 ∨ c ∈ family3) :=
by
  -- Proof skipped
  sorry

end family_groups_correct_l266_266915


namespace Leonard_is_11_l266_266716

def Leonard_age (L N J P T: ℕ) : Prop :=
  (L = N - 4) ∧
  (N = J / 2) ∧
  (P = 2 * L) ∧
  (T = P - 3) ∧
  (L + N + J + P + T = 75)

theorem Leonard_is_11 (L N J P T : ℕ) (h : Leonard_age L N J P T) : L = 11 :=
by {
  sorry
}

end Leonard_is_11_l266_266716


namespace gcd_of_18_and_30_l266_266291

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end gcd_of_18_and_30_l266_266291


namespace third_term_of_arithmetic_sequence_l266_266157

variable (a : ℕ → ℤ)
variable (a1_eq_2 : a 1 = 2)
variable (a2_eq_8 : a 2 = 8)
variable (arithmetic_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))

theorem third_term_of_arithmetic_sequence :
  a 3 = 14 :=
by
  sorry

end third_term_of_arithmetic_sequence_l266_266157


namespace sheep_problem_l266_266172

theorem sheep_problem (mary_sheep : ℕ) (bob_sheep : ℕ) (mary_sheep_initial : mary_sheep = 300)
    (bob_sheep_calculated : bob_sheep = (2 * mary_sheep) + 35) :
    (mary_sheep + 266 = bob_sheep - 69) :=
begin
  sorry
end

end sheep_problem_l266_266172


namespace alpha_in_first_quadrant_l266_266143

theorem alpha_in_first_quadrant (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 2) < 0) 
  (h2 : Real.tan (Real.pi + α) > 0) : 
  (0 < α ∧ α < Real.pi / 2) ∨ (2 * Real.pi < α ∧ α < 5 * Real.pi / 2) := 
by
  sorry

end alpha_in_first_quadrant_l266_266143


namespace eccentricity_is_two_l266_266358

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 3) : 
  eccentricity_of_hyperbola a b h1 h2 h3 = 2 := 
  sorry

end eccentricity_is_two_l266_266358


namespace lowest_n_for_K6_l266_266887

theorem lowest_n_for_K6 (G : SimpleGraph (Fin 1991)) (h : ∀ v : Fin 1991, G.degree v ≥ 1593) :
  ∃ (S : Finset (Fin 1991)), S.card = 6 ∧ (∀ u v ∈ S, G.Adj u v) :=
by
  sorry

end lowest_n_for_K6_l266_266887


namespace max_x_for_lcm_l266_266066

open Nat

-- Define the condition for the least common multiple function for three numbers
def lcm3 (a b c : ℕ) : ℕ := lcm (lcm a b) c

theorem max_x_for_lcm (x : ℕ) : lcm3 x 12 15 = 180 -> x = 180 := by
  sorry

end max_x_for_lcm_l266_266066


namespace find_x_l266_266329

theorem find_x (x : ℕ) (h : (85 + 32 / x : ℝ) * x = 9637) : x = 113 :=
sorry

end find_x_l266_266329


namespace distance_from_P_to_focus_l266_266837

-- Definition of a parabola y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Definition of distance from P to y-axis
def distance_to_y_axis (x : ℝ) : ℝ := abs x

-- Definition of the focus of the parabola y^2 = 8x
def focus : (ℝ × ℝ) := (2, 0)

-- Definition of Euclidean distance
def euclidean_distance (P₁ P₂ : ℝ × ℝ) : ℝ :=
  (P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 

theorem distance_from_P_to_focus (x y : ℝ) (h₁ : parabola x y) (h₂ : distance_to_y_axis x = 4) :
  abs (euclidean_distance (x, y) focus) = 6 :=
sorry

end distance_from_P_to_focus_l266_266837


namespace max_profit_l266_266795

noncomputable def maximum_profit : ℤ := 
  21000

theorem max_profit (x y : ℕ) 
  (h1 : 4 * x + 8 * y ≤ 8000)
  (h2 : 2 * x + y ≤ 1300)
  (h3 : 15 * x + 20 * y ≤ maximum_profit) : 
  15 * x + 20 * y = maximum_profit := 
sorry

end max_profit_l266_266795


namespace number_of_valid_5_digit_numbers_l266_266693

def is_multiple_of_16 (n : Nat) : Prop := 
  n % 16 = 0

theorem number_of_valid_5_digit_numbers : Nat := 
  sorry

example : number_of_valid_5_digit_numbers = 90 :=
  sorry

end number_of_valid_5_digit_numbers_l266_266693


namespace claire_final_balloons_l266_266502

noncomputable def initial_balloons : ℕ := 50
noncomputable def lost_floated_balloons : ℕ := 1 + 12
noncomputable def given_away_balloons : ℕ := 9
noncomputable def gained_balloons : ℕ := 11
noncomputable def final_balloons : ℕ := initial_balloons - (lost_floated_balloons + given_away_balloons) + gained_balloons

theorem claire_final_balloons : final_balloons = 39 :=
by
  unfold final_balloons, initial_balloons, lost_floated_balloons, given_away_balloons, gained_balloons
  simp
  norm_num
  sorry

end claire_final_balloons_l266_266502


namespace number_of_numbers_l266_266450

theorem number_of_numbers (N : ℕ) (h_avg : (18 * N + 40) / N = 22) : N = 10 :=
by
  sorry

end number_of_numbers_l266_266450


namespace sum_first_nine_terms_l266_266707

open ArithmeticSequence

theorem sum_first_nine_terms 
  (a : ℕ → ℕ) 
  (h1 : a 1 + a 4 + a 7 = 39) 
  (h2 : a 3 + a 6 + a 9 = 27) 
  (arith_seq : arithmetic_sequence a) :
  sum_first_nine_terms arith_seq = 99 := 
sorry

end sum_first_nine_terms_l266_266707


namespace vector_parallel_condition_l266_266999

def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (2 * m, m + 1)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_parallel_condition (m : ℝ) (h_parallel : AB OA OB = (3, 1) ∧ 
    (∀ k : ℝ, 2*m = 3*k ∧ m + 1 = k)) : m = -3 :=
by
  sorry

end vector_parallel_condition_l266_266999


namespace cody_initial_marbles_l266_266965

theorem cody_initial_marbles (M : ℕ) (h1 : (2 / 3 : ℝ) * M - (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M) - (2 * (1 / 4 : ℝ) * ((2 / 3 : ℝ) * M)) = 7) : M = 42 := 
  sorry

end cody_initial_marbles_l266_266965


namespace paper_folding_holes_l266_266219

def folded_paper_holes (folds: Nat) (holes: Nat) : Nat :=
  match folds with
  | 0 => holes
  | n+1 => 2 * folded_paper_holes n holes

theorem paper_folding_holes : folded_paper_holes 3 1 = 8 :=
by
  -- sorry to skip the proof
  sorry

end paper_folding_holes_l266_266219


namespace gcd_18_30_l266_266302

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := 
by
  sorry

end gcd_18_30_l266_266302


namespace polygon_sides_eq_five_l266_266762

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five_l266_266762


namespace solve_for_x_l266_266181

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.07 * (25 + x) = 15.1) : x = 111.25 :=
by
  sorry

end solve_for_x_l266_266181


namespace parallelogram_A2B2C2D2_l266_266155

theorem parallelogram_A2B2C2D2
  (A B C D A1 B1 C1 D1 A2 B2 C2 D2 : Point)
  (hABCD_parallelogram : parallelogram A B C D)
  (hA1_on_AB : lies_on_line_segment A1 A B)
  (hB1_on_BC : lies_on_line_segment B1 B C)
  (hC1_on_CD : lies_on_line_segment C1 C D)
  (hD1_on_DA : lies_on_line_segment D1 D A)
  (hA2_on_A1B1 : lies_on_line_segment A2 A1 B1)
  (hB2_on_B1C1 : lies_on_line_segment B2 B1 C1)
  (hC2_on_C1D1 : lies_on_line_segment C2 C1 D1)
  (hD2_on_D1A1 : lies_on_line_segment D2 D1 A1)
  (h_ratios : (segment_ratio A A1 B A1)
            = (segment_ratio B B1 C B1)
            = (segment_ratio C C1 D C1)
            = (segment_ratio D D1 A D1)
            = (segment_ratio A1 D2 D1 D2)
            = (segment_ratio D1 C2 C1 C2)
            = (segment_ratio C1 B2 B1 B2)
            = (segment_ratio B1 A2 A1 A2))
  : parallelogram A2 B2 C2 D2
    ∧ parallel A2 B2 A B
    ∧ parallel B2 C2 B C
    ∧ parallel C2 D2 C D
    ∧ parallel D2 A2 D A := 
sorry

end parallelogram_A2B2C2D2_l266_266155


namespace book_arrangement_l266_266019

-- Define the books as distinct elements
def math_books := fin 3
def chinese_books := fin 3

/-- Prove that the number of ways to arrange 3 different math books and 3 different Chinese books 
such that no two books of the same subject are adjacent is 72. -/
theorem book_arrangement :
  let ways_to_arrange := 2 * nat.factorial 3 * nat.factorial 3  in
  ways_to_arrange = 72 :=
by
  sorry

end book_arrangement_l266_266019


namespace animal_count_in_hollow_l266_266438

theorem animal_count_in_hollow (heads legs : ℕ) (animals_with_odd_legs animals_with_even_legs : ℕ) :
  heads = 18 →
  legs = 24 →
  (∀ n, n % 2 = 1 → animals_with_odd_legs * 2 = heads - 2 * n) →
  (∀ m, m % 2 = 0 → animals_with_even_legs * 1 = heads - m) →
  (animals_with_odd_legs + animals_with_even_legs = 10 ∨
   animals_with_odd_legs + animals_with_even_legs = 12 ∨
   animals_with_odd_legs + animals_with_even_legs = 14) :=
sorry

end animal_count_in_hollow_l266_266438


namespace intersection_of_A_and_B_l266_266339

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := { x | ∃ m : ℕ, x = 2 * m }

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by sorry

end intersection_of_A_and_B_l266_266339


namespace divisible_bc_ad_l266_266565

open Int

theorem divisible_bc_ad (a b c d m : ℤ) (hm : 0 < m)
  (h1 : m ∣ a * c)
  (h2 : m ∣ b * d)
  (h3 : m ∣ (b * c + a * d)) :
  m ∣ b * c ∧ m ∣ a * d :=
by
  sorry

end divisible_bc_ad_l266_266565


namespace no_such_function_exists_l266_266235

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ n > 2, f (f (n - 1)) = f (n + 1) - f n :=
by {
  sorry
}

end no_such_function_exists_l266_266235


namespace total_distance_from_A_through_B_to_C_l266_266460

noncomputable def distance_A_B_map : ℝ := 120
noncomputable def distance_B_C_map : ℝ := 70
noncomputable def map_scale : ℝ := 10 -- km per cm

noncomputable def distance_A_B := distance_A_B_map * map_scale -- Distance from City A to City B in km
noncomputable def distance_B_C := distance_B_C_map * map_scale -- Distance from City B to City C in km
noncomputable def total_distance := distance_A_B + distance_B_C -- Total distance in km

theorem total_distance_from_A_through_B_to_C :
  total_distance = 1900 := by
  sorry

end total_distance_from_A_through_B_to_C_l266_266460


namespace two_digit_number_eq_27_l266_266222

theorem two_digit_number_eq_27 (A : ℕ) (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
    (h : A = 10 * x + y) (hcond : A = 3 * (x + y)) : A = 27 :=
by
  sorry

end two_digit_number_eq_27_l266_266222


namespace inequality_m_2n_l266_266850

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

lemma max_f : ∃ x : ℝ, f x = 2 :=
sorry

theorem inequality_m_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 1/(2*n) = 2) : m + 2*n ≥ 2 :=
sorry

end inequality_m_2n_l266_266850


namespace math_problem_l266_266041

variable (a b c d : ℝ)

-- The initial condition provided in the problem
def given_condition : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7

-- The statement that needs to be proven
theorem math_problem 
  (h : given_condition a b c d) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := 
by 
  sorry

end math_problem_l266_266041


namespace minimize_surface_area_l266_266613

-- Define the problem conditions
def volume (x y : ℝ) : ℝ := 2 * x^2 * y
def surface_area (x y : ℝ) : ℝ := 2 * (2 * x^2 + 2 * x * y + x * y)

theorem minimize_surface_area :
  ∃ (y : ℝ), 
  (∀ (x : ℝ), volume x y = 72) → 
  1 * 2 * y = 4 :=
by
  sorry

end minimize_surface_area_l266_266613


namespace f_neg_val_is_minus_10_l266_266529
-- Import the necessary Lean library

-- Define the function f with the given conditions
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 3

-- Define the specific values
def x_val : ℝ := 2023
def x_neg_val : ℝ := -2023
def f_pos_val : ℝ := 16

-- Theorem to prove
theorem f_neg_val_is_minus_10 (a b : ℝ)
  (h : f a b x_val = f_pos_val) : 
  f a b x_neg_val = -10 :=
by
  -- Sorry placeholder for proof
  sorry

end f_neg_val_is_minus_10_l266_266529


namespace recurrence_relation_p_series_l266_266404

noncomputable def p_series (n k : ℕ) : ℝ :=
if k < n then (p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k))
else 0

-- Statement of the theorem
theorem recurrence_relation_p_series (n k : ℕ) (h : k < n) :
  p_series n k = p_series (n - 1) k - (1 / (2 : ℝ)^k) * p_series (n - k) k + (1 / (2 : ℝ)^k) :=
sorry

end recurrence_relation_p_series_l266_266404


namespace evaluate_expression_is_15_l266_266507

noncomputable def sumOfFirstNOddNumbers (n : ℕ) : ℕ :=
  n^2

noncomputable def simplifiedExpression : ℕ :=
  sumOfFirstNOddNumbers 1 +
  sumOfFirstNOddNumbers 2 +
  sumOfFirstNOddNumbers 3 +
  sumOfFirstNOddNumbers 4 +
  sumOfFirstNOddNumbers 5

theorem evaluate_expression_is_15 : simplifiedExpression = 15 := by
  sorry

end evaluate_expression_is_15_l266_266507
