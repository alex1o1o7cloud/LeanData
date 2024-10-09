import Mathlib

namespace length_of_best_day_l380_38062

theorem length_of_best_day
  (len_raise_the_roof : Nat)
  (len_rap_battle : Nat)
  (len_best_day : Nat)
  (total_ride_duration : Nat)
  (playlist_count : Nat)
  (total_songs_length : Nat)
  (h_len_raise_the_roof : len_raise_the_roof = 2)
  (h_len_rap_battle : len_rap_battle = 3)
  (h_total_ride_duration : total_ride_duration = 40)
  (h_playlist_count : playlist_count = 5)
  (h_total_songs_length : len_raise_the_roof + len_rap_battle + len_best_day = total_songs_length)
  (h_playlist_length : total_ride_duration / playlist_count = total_songs_length) :
  len_best_day = 3 := 
sorry

end length_of_best_day_l380_38062


namespace ratio_of_area_l380_38094

noncomputable def area_of_triangle_ratio (AB CD height : ℝ) (h : CD = 2 * AB) : ℝ :=
  let ABCD_area := (AB + CD) * height / 2
  let EAB_area := ABCD_area / 3
  EAB_area / ABCD_area

theorem ratio_of_area (AB CD : ℝ) (height : ℝ) (h1 : AB = 10) (h2 : CD = 20) (h3 : height = 5) : 
  area_of_triangle_ratio AB CD height (by rw [h1, h2]; ring) = 1 / 3 :=
sorry

end ratio_of_area_l380_38094


namespace students_in_first_bus_l380_38082

theorem students_in_first_bus (total_buses : ℕ) (avg_students_per_bus : ℕ) 
(avg_remaining_students : ℕ) (num_remaining_buses : ℕ) 
(h1 : total_buses = 6) 
(h2 : avg_students_per_bus = 28) 
(h3 : avg_remaining_students = 26) 
(h4 : num_remaining_buses = 5) :
  (total_buses * avg_students_per_bus - num_remaining_buses * avg_remaining_students = 38) :=
by
  sorry

end students_in_first_bus_l380_38082


namespace no_nat_numbers_satisfy_l380_38063

theorem no_nat_numbers_satisfy (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k := 
sorry

end no_nat_numbers_satisfy_l380_38063


namespace garrett_cats_count_l380_38067

def number_of_cats_sheridan : ℕ := 11
def difference_in_cats : ℕ := 13

theorem garrett_cats_count (G : ℕ) (h : G - number_of_cats_sheridan = difference_in_cats) : G = 24 :=
by
  sorry

end garrett_cats_count_l380_38067


namespace seahorse_penguin_ratio_l380_38013

theorem seahorse_penguin_ratio :
  ∃ S P : ℕ, S = 70 ∧ P = S + 85 ∧ Nat.gcd 70 (S + 85) = 5 ∧ 70 / Nat.gcd 70 (S + 85) = 14 ∧ (S + 85) / Nat.gcd 70 (S + 85) = 31 :=
by
  sorry

end seahorse_penguin_ratio_l380_38013


namespace sufficient_but_not_necessary_l380_38001

theorem sufficient_but_not_necessary (a : ℝ) : (a > 6 → a^2 > 36) ∧ ¬(a^2 > 36 → a > 6) := 
by
  sorry

end sufficient_but_not_necessary_l380_38001


namespace surface_area_implies_side_length_diagonal_l380_38003

noncomputable def cube_side_length_diagonal (A : ℝ) := 
  A = 864 → ∃ s d : ℝ, s = 12 ∧ d = 12 * Real.sqrt 3

theorem surface_area_implies_side_length_diagonal : 
  cube_side_length_diagonal 864 := by
  sorry

end surface_area_implies_side_length_diagonal_l380_38003


namespace find_side_b_l380_38084

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
    (hC : C = Real.pi / 3) (hA : A = Real.pi / 6) (hB : B = Real.pi / 2) : b = 4 := by
  sorry

end find_side_b_l380_38084


namespace solution_set_for_absolute_value_inequality_l380_38023

theorem solution_set_for_absolute_value_inequality :
  {x : ℝ | |2 * x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by 
  sorry

end solution_set_for_absolute_value_inequality_l380_38023


namespace find_a3_l380_38012

variable (a_n : ℕ → ℤ) (a1 a4 a5 : ℤ)
variable (d : ℤ := -2)

-- Conditions
axiom h1 : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom h2 : a4 = a1 + 3 * d
axiom h3 : a5 = a1 + 4 * d
axiom h4 : a4 * a4 = a1 * a5

-- Question to prove
theorem find_a3 : (a_n 3) = 5 := by
  sorry

end find_a3_l380_38012


namespace find_n_coins_l380_38092

def num_coins : ℕ := 5

theorem find_n_coins (n : ℕ) (h : (n^2 + n + 2) = 2^n) : n = num_coins :=
by {
  -- Proof to be filled in
  sorry
}

end find_n_coins_l380_38092


namespace product_of_slopes_hyperbola_l380_38075

theorem product_of_slopes_hyperbola (a b x0 y0 : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : (x0, y0) ≠ (-a, 0)) (h4 : (x0, y0) ≠ (a, 0)) 
(h5 : x0^2 / a^2 - y0^2 / b^2 = 1) : 
(y0 / (x0 + a) * (y0 / (x0 - a)) = b^2 / a^2) :=
sorry

end product_of_slopes_hyperbola_l380_38075


namespace power_inequality_l380_38035

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_ineq : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19)

theorem power_inequality :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by
  sorry

end power_inequality_l380_38035


namespace chessboard_tiling_impossible_l380_38017

theorem chessboard_tiling_impossible :
  ¬ ∃ (cover : (Fin 5 × Fin 7 → Prop)), 
    (cover (0, 3) = false) ∧
    (∀ i j, (cover (i, j) → cover (i + 1, j) ∨ cover (i, j + 1)) ∧
             ∀ x y z w, cover (x, y) → cover (z, w) → (x ≠ z ∨ y ≠ w)) :=
sorry

end chessboard_tiling_impossible_l380_38017


namespace maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l380_38077

-- Part (a): One blue cube
theorem maximum_amount_one_blue_cube : 
  ∃ (B : ℕ → ℚ) (P : ℕ → ℕ), (B 1 = 2) ∧ (∀ m > 1, B m = 2^m / P m) ∧ (P 1 = 1) ∧ (∀ m > 1, P m = m) ∧ B 100 = 2^100 / 100 :=
by
  sorry

-- Part (b): Exactly n blue cubes
theorem maximum_amount_n_blue_cubes (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 100) : 
  ∃ (B : ℕ × ℕ → ℚ) (P : ℕ × ℕ → ℕ), (B (1, 0) = 2) ∧ (B (1, 1) = 2) ∧ (∀ m > 1, B (m, 0) = 2^m) ∧ (P (1, 0) = 1) ∧ (P (1, 1) = 1) ∧ (∀ m > 1, P (m, 0) = 1) ∧ B (100, n) = 2^100 / Nat.choose 100 n :=
by
  sorry

end maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l380_38077


namespace parallelogram_altitude_base_ratio_l380_38095

theorem parallelogram_altitude_base_ratio 
  (area base : ℕ) (h : ℕ) 
  (h_base : base = 9)
  (h_area : area = 162)
  (h_area_eq : area = base * h) : 
  h / base = 2 := 
by 
  -- placeholder for the proof
  sorry

end parallelogram_altitude_base_ratio_l380_38095


namespace problem1_problem2_problem3_l380_38064

noncomputable def U : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
noncomputable def A : Set ℝ := {x | x < 1 ∨ x > 3}
noncomputable def B : Set ℝ := {x | x < 1 ∨ x > 2}

theorem problem1 : A ∩ B = {x | x < 1 ∨ x > 3} := 
  sorry

theorem problem2 : A ∩ (U \ B) = ∅ := 
  sorry

theorem problem3 : U \ (A ∪ B) = {1, 2} := 
  sorry

end problem1_problem2_problem3_l380_38064


namespace pictures_vertical_l380_38065

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l380_38065


namespace cost_of_senior_ticket_l380_38021

theorem cost_of_senior_ticket (x : ℤ) (total_tickets : ℤ) (cost_regular_ticket : ℤ) (total_sales : ℤ) (senior_tickets_sold : ℤ) (regular_tickets_sold : ℤ) :
  total_tickets = 65 →
  cost_regular_ticket = 15 →
  total_sales = 855 →
  senior_tickets_sold = 24 →
  regular_tickets_sold = total_tickets - senior_tickets_sold →
  total_sales = senior_tickets_sold * x + regular_tickets_sold * cost_regular_ticket →
  x = 10 :=
by
  sorry

end cost_of_senior_ticket_l380_38021


namespace problem_statement_l380_38006

variable (a b : Type) [LinearOrder a] [LinearOrder b]
variable (α β : Type) [LinearOrder α] [LinearOrder β]

-- Given conditions
def line_perpendicular_to_plane (l : Type) (p : Type) [LinearOrder l] [LinearOrder p] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

def lines_parallel (l1 : Type) (l2 : Type) [LinearOrder l1] [LinearOrder l2] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

theorem problem_statement (a b α : Type) [LinearOrder a] [LinearOrder b] [LinearOrder α]
(val_perp1 : line_perpendicular_to_plane a α)
(val_perp2 : line_perpendicular_to_plane b α)
: lines_parallel a b :=
sorry

end problem_statement_l380_38006


namespace parabola_directrix_distance_l380_38066

theorem parabola_directrix_distance (a : ℝ) : 
  (abs (a / 4 + 1) = 2) → (a = -12 ∨ a = 4) := 
by
  sorry

end parabola_directrix_distance_l380_38066


namespace college_student_ticket_cost_l380_38076

theorem college_student_ticket_cost 
    (total_visitors : ℕ)
    (nyc_residents: ℕ)
    (college_students_nyc: ℕ)
    (total_money_received : ℕ) :
    total_visitors = 200 →
    nyc_residents = total_visitors / 2 →
    college_students_nyc = (nyc_residents * 30) / 100 →
    total_money_received = 120 →
    (total_money_received / college_students_nyc) = 4 := 
sorry

end college_student_ticket_cost_l380_38076


namespace nature_of_graph_l380_38031

theorem nature_of_graph :
  ∀ (x y : ℝ), (x^2 - 3 * y) * (x - y + 1) = (y^2 - 3 * x) * (x - y + 1) →
    (y = -x - 3 ∨ y = x ∨ y = x + 1) ∧ ¬( (y = -x - 3) ∧ (y = x) ∧ (y = x + 1) ) :=
by
  intros x y h
  sorry

end nature_of_graph_l380_38031


namespace periodic_sequence_a2019_l380_38042

theorem periodic_sequence_a2019 :
  (∃ (a : ℕ → ℤ),
    a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ 
    (∀ n : ℕ, n ≥ 4 → a n = a (n-1) * a (n-3)) ∧
    a 2019 = -1) :=
sorry

end periodic_sequence_a2019_l380_38042


namespace jellybeans_condition_l380_38073

theorem jellybeans_condition (n : ℕ) (h1 : n ≥ 150) (h2 : n % 15 = 14) : n = 164 :=
sorry

end jellybeans_condition_l380_38073


namespace meters_conversion_equivalence_l380_38038

-- Define the conditions
def meters_to_decimeters (m : ℝ) : ℝ := m * 10
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- State the problem
theorem meters_conversion_equivalence :
  7.34 = 7 + (meters_to_decimeters 0.3) / 10 + (meters_to_centimeters 0.04) / 100 :=
sorry

end meters_conversion_equivalence_l380_38038


namespace part1_even_function_part2_two_distinct_zeros_l380_38016

noncomputable def f (x a : ℝ) : ℝ := (4^x + a) / 2^x
noncomputable def g (x a : ℝ) : ℝ := f x a - (a + 1)

theorem part1_even_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) ↔ a = 1 :=
sorry

theorem part2_two_distinct_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (a ∈ Set.Icc (1/2) 1 ∪ Set.Icc 1 2) :=
sorry

end part1_even_function_part2_two_distinct_zeros_l380_38016


namespace cos_beta_eq_sqrt10_over_10_l380_38008

-- Define the conditions and the statement
theorem cos_beta_eq_sqrt10_over_10 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = 2)
  (h_sin_sum : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 :=
sorry

end cos_beta_eq_sqrt10_over_10_l380_38008


namespace nadia_flower_shop_l380_38079

theorem nadia_flower_shop (roses lilies cost_per_rose cost_per_lily cost_roses cost_lilies total_cost : ℕ)
  (h1 : roses = 20)
  (h2 : lilies = 3 * roses / 4)
  (h3 : cost_per_rose = 5)
  (h4 : cost_per_lily = 2 * cost_per_rose)
  (h5 : cost_roses = roses * cost_per_rose)
  (h6 : cost_lilies = lilies * cost_per_lily)
  (h7 : total_cost = cost_roses + cost_lilies) :
  total_cost = 250 :=
by
  sorry

end nadia_flower_shop_l380_38079


namespace compare_logs_l380_38030

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem compare_logs (a b c : ℝ) (h1 : a = log_base 4 1.25) (h2 : b = log_base 5 1.2) (h3 : c = log_base 4 8) :
  c > a ∧ a > b :=
by
  sorry

end compare_logs_l380_38030


namespace total_cars_l380_38052

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end total_cars_l380_38052


namespace p_sufficient_but_not_necessary_for_q_l380_38024

def proposition_p (x : ℝ) := x - 1 = 0
def proposition_q (x : ℝ) := (x - 1) * (x + 2) = 0

theorem p_sufficient_but_not_necessary_for_q :
  ( (∀ x, proposition_p x → proposition_q x) ∧ ¬(∀ x, proposition_p x ↔ proposition_q x) ) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l380_38024


namespace playground_children_count_l380_38036

theorem playground_children_count (boys girls : ℕ) (h_boys : boys = 27) (h_girls : girls = 35) : boys + girls = 62 := by
  sorry

end playground_children_count_l380_38036


namespace problem1_problem2_l380_38025

variable (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) ^ 2 + x * (x - 2 * y) = 2 * x ^ 2 + y ^ 2 := by
  sorry

variable (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) -- to ensure the denominators are non-zero

-- Problem 2
theorem problem2 : (x ^ 2 - 6 * x + 9) / (x - 2) / (x + 2 - (3 * x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end problem1_problem2_l380_38025


namespace smallest_t_for_sine_polar_circle_l380_38086

theorem smallest_t_for_sine_polar_circle :
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → ∃ r : ℝ, r = Real.sin θ) ∧
           (∀ θ : ℝ, (θ = t) → ∃ r : ℝ, r = 0) ∧
           (∀ t' : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t') → ∃ r : ℝ, r = Real.sin θ) →
                       (∀ θ : ℝ, (θ = t') → ∃ r : ℝ, r = 0) → t' ≥ t) :=
by
  sorry

end smallest_t_for_sine_polar_circle_l380_38086


namespace price_each_clock_is_correct_l380_38083

-- Definitions based on the conditions
def numberOfDolls := 3
def numberOfClocks := 2
def numberOfGlasses := 5
def pricePerDoll := 5
def pricePerGlass := 4
def totalCost := 40
def profit := 25

-- The total revenue from selling dolls and glasses
def revenueFromDolls := numberOfDolls * pricePerDoll
def revenueFromGlasses := numberOfGlasses * pricePerGlass
def totalRevenueNeeded := totalCost + profit
def revenueFromDollsAndGlasses := revenueFromDolls + revenueFromGlasses

-- The required revenue from clocks
def revenueFromClocks := totalRevenueNeeded - revenueFromDollsAndGlasses

-- The price per clock
def pricePerClock := revenueFromClocks / numberOfClocks

-- Statement to prove
theorem price_each_clock_is_correct : pricePerClock = 15 := sorry

end price_each_clock_is_correct_l380_38083


namespace vacation_animals_total_l380_38028

noncomputable def lisa := 40
noncomputable def alex := lisa / 2
noncomputable def jane := alex + 10
noncomputable def rick := 3 * jane
noncomputable def tim := 2 * rick
noncomputable def you := 5 * tim
noncomputable def total_animals := lisa + alex + jane + rick + tim + you

theorem vacation_animals_total : total_animals = 1260 := by
  sorry

end vacation_animals_total_l380_38028


namespace cost_of_two_other_puppies_l380_38009

theorem cost_of_two_other_puppies (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) (remaining_puppies_cost : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  remaining_puppies_cost = (total_cost - num_sale_puppies * sale_price) →
  (remaining_puppies_cost / (num_puppies - num_sale_puppies)) = 175 :=
by
  intros
  sorry

end cost_of_two_other_puppies_l380_38009


namespace product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l380_38041

theorem product_two_smallest_one_digit_primes_and_largest_three_digit_prime :
  2 * 3 * 997 = 5982 :=
by
  sorry

end product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l380_38041


namespace solve_bank_account_problem_l380_38045

noncomputable def bank_account_problem : Prop :=
  ∃ (A E Z : ℝ),
    A > E ∧
    Z > A ∧
    A - E = (1/12) * (A + E) ∧
    Z - A = (1/10) * (Z + A) ∧
    1.10 * A = 1.20 * E + 20 ∧
    1.10 * A + 30 = 1.15 * Z ∧
    E = 2000 / 23

theorem solve_bank_account_problem : bank_account_problem :=
sorry

end solve_bank_account_problem_l380_38045


namespace problem_l380_38018

theorem problem (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 1) 
  (h3 : a + c + d = 16) 
  (h4 : b + c + d = 9) : 
  a * b + c * d = 734 / 9 := 
by 
  sorry

end problem_l380_38018


namespace solutions_of_quadratic_l380_38046

theorem solutions_of_quadratic (c : ℝ) (h : ∀ α β : ℝ, 
  (α^2 - 3*α + c = 0 ∧ β^2 - 3*β + c = 0) → 
  ( (-α)^2 + 3*(-α) - c = 0 ∨ (-β)^2 + 3*(-β) - c = 0 ) ) :
  ∃ α β : ℝ, (α = 0 ∧ β = 3) ∨ (α = 3 ∧ β = 0) :=
by
  sorry

end solutions_of_quadratic_l380_38046


namespace find_distance_from_origin_l380_38078

-- Define the conditions as functions
def point_distance_from_x_axis (y : ℝ) : Prop := abs y = 15
def distance_from_point (x y : ℝ) (x₀ y₀ : ℝ) (d : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = d^2

-- Define the proof problem
theorem find_distance_from_origin (x y : ℝ) (n : ℝ) (hx : x = 2 + Real.sqrt 105) (hy : point_distance_from_x_axis y) (hx_gt : x > 2) (hdist : distance_from_point x y 2 7 13) :
  n = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end find_distance_from_origin_l380_38078


namespace lowest_exam_score_l380_38026

theorem lowest_exam_score 
  (first_exam_score : ℕ := 90) 
  (second_exam_score : ℕ := 108) 
  (third_exam_score : ℕ := 102) 
  (max_score_per_exam : ℕ := 120) 
  (desired_average : ℕ := 100) 
  (total_exams : ℕ := 5) 
  (total_score_needed : ℕ := desired_average * total_exams) : 
  ∃ (lowest_score : ℕ), lowest_score = 80 :=
by
  sorry

end lowest_exam_score_l380_38026


namespace initial_deposit_l380_38059

theorem initial_deposit (P R : ℝ) (h1 : 8400 = P + (P * R * 2) / 100) (h2 : 8760 = P + (P * (R + 4) * 2) / 100) : 
  P = 2250 :=
  sorry

end initial_deposit_l380_38059


namespace find_a2016_l380_38098

theorem find_a2016 (S : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h₁ : S 1 = 6)
  (h₂ : S 2 = 4)
  (h₃ : ∀ n, S n > 0)
  (h₄ : ∀ n, (S (2 * n - 1))^2 = S (2 * n) * S (2 * n + 2))
  (h₅ : ∀ n, 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1))
  : a 2016 = -1009 := 
  sorry

end find_a2016_l380_38098


namespace set_intersection_l380_38011

open Set

def U := {x : ℝ | True}
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x - 1 ≥ 0}
def complement (U B : Set ℝ) := {x : ℝ | x ∉ B}
def intersection (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection :
  intersection A (complement U B) = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end set_intersection_l380_38011


namespace mark_paintable_area_l380_38019

theorem mark_paintable_area :
  let num_bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let area_excluded := 70
  let area_wall_one_bedroom := 2 * (length * height) + 2 * (width * height) - area_excluded 
  (area_wall_one_bedroom * num_bedrooms) = 1520 :=
by
  sorry

end mark_paintable_area_l380_38019


namespace danny_total_bottle_caps_l380_38005

def danny_initial_bottle_caps : ℕ := 37
def danny_found_bottle_caps : ℕ := 18

theorem danny_total_bottle_caps : danny_initial_bottle_caps + danny_found_bottle_caps = 55 := by
  sorry

end danny_total_bottle_caps_l380_38005


namespace inequality_proof_l380_38049

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) : 
  1 / a + 4 / b ≥ 9 / 4 :=
by
  sorry

end inequality_proof_l380_38049


namespace compute_expression_l380_38002

theorem compute_expression : 2 * ((3 + 7) ^ 2 + (3 ^ 2 + 7 ^ 2)) = 316 := 
by
  sorry

end compute_expression_l380_38002


namespace Carmen_candle_burn_time_l380_38074

theorem Carmen_candle_burn_time
  (night_to_last_candle_first_scenario : ℕ := 8)
  (hours_per_night_second_scenario : ℕ := 2)
  (nights_second_scenario : ℕ := 24)
  (candles_second_scenario : ℕ := 6) :
  ∃ T : ℕ, (night_to_last_candle_first_scenario * T = hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) ∧ T = 1 :=
by
  let T := (hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) / night_to_last_candle_first_scenario
  have : T = 1 := by sorry
  use T
  exact ⟨ by sorry, this⟩

end Carmen_candle_burn_time_l380_38074


namespace problem1_problem2_l380_38022

-- Problem 1
theorem problem1 : -9 + (-4 * 5) = -29 :=
by
  sorry

-- Problem 2
theorem problem2 : (-(6) * -2) / (2 / 3) = -18 :=
by
  sorry

end problem1_problem2_l380_38022


namespace top_leftmost_rectangle_is_B_l380_38037

-- Definitions for the side lengths of each rectangle
def A_w : ℕ := 6
def A_x : ℕ := 2
def A_y : ℕ := 7
def A_z : ℕ := 10

def B_w : ℕ := 2
def B_x : ℕ := 1
def B_y : ℕ := 4
def B_z : ℕ := 8

def C_w : ℕ := 5
def C_x : ℕ := 11
def C_y : ℕ := 6
def C_z : ℕ := 3

def D_w : ℕ := 9
def D_x : ℕ := 7
def D_y : ℕ := 5
def D_z : ℕ := 9

def E_w : ℕ := 11
def E_x : ℕ := 4
def E_y : ℕ := 9
def E_z : ℕ := 1

-- The problem statement to prove
theorem top_leftmost_rectangle_is_B : 
  (B_w = 2 ∧ B_y = 4) ∧ 
  (A_w = 6 ∨ D_w = 9 ∨ C_w = 5 ∨ E_w = 11) ∧
  (A_y = 7 ∨ D_y = 5 ∨ C_y = 6 ∨ E_y = 9) → 
  (B_w = 2 ∧ ∀ w : ℕ, w = 6 ∨ w = 5 ∨ w = 9 ∨ w = 11 → B_w < w) :=
by {
  -- skipping the proof
  sorry
}

end top_leftmost_rectangle_is_B_l380_38037


namespace total_pages_of_book_l380_38047

-- Definitions for the conditions
def firstChapterPages : Nat := 66
def secondChapterPages : Nat := 35
def thirdChapterPages : Nat := 24

-- Theorem stating the main question and answer
theorem total_pages_of_book : firstChapterPages + secondChapterPages + thirdChapterPages = 125 := by
  -- Proof will be provided here
  sorry

end total_pages_of_book_l380_38047


namespace translate_upwards_one_unit_l380_38080

theorem translate_upwards_one_unit (x y : ℝ) : (y = 2 * x) → (y + 1 = 2 * x + 1) := 
by sorry

end translate_upwards_one_unit_l380_38080


namespace number_of_correct_propositions_l380_38055

theorem number_of_correct_propositions : 
    (∀ a b : ℝ, a < b → ¬ (a^2 < b^2)) ∧ 
    (∀ a : ℝ, (∀ x : ℝ, |x + 1| + |x - 1| ≥ a ↔ a ≤ 2)) ∧ 
    (¬ (∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0) → 
    1 = 1 := 
by
  sorry

end number_of_correct_propositions_l380_38055


namespace travel_time_K_l380_38093

/-
Given that:
1. K's speed is x miles per hour.
2. M's speed is x - 1 miles per hour.
3. K takes 1 hour less than M to travel 60 miles (i.e., 60/x hours).
Prove that K's time to travel 60 miles is 6 hours.
-/
theorem travel_time_K (x : ℝ)
  (h1 : x > 0)
  (h2 : x ≠ 1)
  (h3 : 60 / (x - 1) - 60 / x = 1) :
  60 / x = 6 :=
sorry

end travel_time_K_l380_38093


namespace determine_day_from_statements_l380_38033

/-- Define the days of the week as an inductive type. -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving DecidableEq, Repr

open Day

/-- Define the properties of the lion lying on specific days. -/
def lion_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Define the properties of the lion telling the truth on specific days. -/
def lion_truth (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday ∨ d = Sunday

/-- Define the properties of the unicorn lying on specific days. -/
def unicorn_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

/-- Define the properties of the unicorn telling the truth on specific days. -/
def unicorn_truth (d : Day) : Prop :=
  d = Sunday ∨ d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Function to determine the day before a given day. -/
def yesterday (d : Day) : Day :=
  match d with
  | Monday    => Sunday
  | Tuesday   => Monday
  | Wednesday => Tuesday
  | Thursday  => Wednesday
  | Friday    => Thursday
  | Saturday  => Friday
  | Sunday    => Saturday

/-- Define the lion's statement: "Yesterday was a day when I lied." -/
def lion_statement (d : Day) : Prop :=
  lion_lies (yesterday d)

/-- Define the unicorn's statement: "Yesterday was a day when I lied." -/
def unicorn_statement (d : Day) : Prop :=
  unicorn_lies (yesterday d)

/-- Prove that today must be Thursday given the conditions and statements. -/
theorem determine_day_from_statements (d : Day) :
    lion_statement d ∧ unicorn_statement d → d = Thursday := by
  sorry

end determine_day_from_statements_l380_38033


namespace AB_plus_C_eq_neg8_l380_38085

theorem AB_plus_C_eq_neg8 (A B C : ℤ) (g : ℝ → ℝ)
(hf : ∀ x > 3, g x > 0.5)
(heq : ∀ x, g x = x^2 / (A * x^2 + B * x + C))
(hasymp_vert : ∀ x, (A * (x + 3) * (x - 2) = 0 → x = -3 ∨ x = 2))
(hasymp_horiz : (1 : ℝ) / (A : ℝ) < 1) :
A + B + C = -8 :=
sorry

end AB_plus_C_eq_neg8_l380_38085


namespace Yankees_to_Mets_ratio_l380_38056

theorem Yankees_to_Mets_ratio : 
  ∀ (Y M R : ℕ), M = 88 → (M + R + Y = 330) → (4 * R = 5 * M) → (Y : ℚ) / M = 3 / 2 :=
by
  intros Y M R hm htotal hratio
  sorry

end Yankees_to_Mets_ratio_l380_38056


namespace cars_and_tourists_l380_38053

theorem cars_and_tourists (n t : ℕ) (h : n * t = 737) : n = 11 ∧ t = 67 ∨ n = 67 ∧ t = 11 :=
by
  sorry

end cars_and_tourists_l380_38053


namespace rectangle_perimeter_eq_30sqrt10_l380_38020

theorem rectangle_perimeter_eq_30sqrt10 (A : ℝ) (l : ℝ) (w : ℝ) 
  (hA : A = 500) (hlw : l = 2 * w) (hArea : A = l * w) : 
  2 * (l + w) = 30 * Real.sqrt 10 :=
by
  sorry

end rectangle_perimeter_eq_30sqrt10_l380_38020


namespace minimum_area_triangle_AOB_l380_38014

theorem minimum_area_triangle_AOB : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) ∧ (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) → (1/2 * a * b ≥ 12)) := 
sorry

end minimum_area_triangle_AOB_l380_38014


namespace find_B_l380_38097

def is_prime_203B21 (B : ℕ) : Prop :=
  2 ≤ B ∧ B < 10 ∧ Prime (200000 + 3000 + 100 * B + 20 + 1)

theorem find_B : ∃ B, is_prime_203B21 B ∧ ∀ B', is_prime_203B21 B' → B' = 5 := by
  sorry

end find_B_l380_38097


namespace arithmetic_sequence_count_l380_38087

-- Define the initial conditions
def a1 : ℤ := -3
def d : ℤ := 3
def an : ℤ := 45

-- Proposition stating the number of terms n in the arithmetic sequence
theorem arithmetic_sequence_count :
  ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 17 :=
by
  -- Skip the proof
  sorry

end arithmetic_sequence_count_l380_38087


namespace sum_of_digits_a_l380_38007

def a : ℕ := 10^10 - 47

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_a : sum_of_digits a = 81 := 
  by 
    sorry

end sum_of_digits_a_l380_38007


namespace probability_of_inverse_proportion_l380_38099

def points : List (ℝ × ℝ) :=
  [(0.5, -4.5), (1, -4), (1.5, -3.5), (2, -3), (2.5, -2.5), (3, -2), (3.5, -1.5),
   (4, -1), (4.5, -0.5), (5, 0)]

def inverse_proportion_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0.5, -4.5), (4.5, -0.5)), ((1, -4), (4, -1)), ((1.5, -3.5), (3.5, -1.5)), ((2, -3), (3, -2))]

theorem probability_of_inverse_proportion:
  let num_pairs := List.length points * (List.length points - 1)
  let favorable_pairs := 2 * List.length inverse_proportion_pairs
  favorable_pairs / num_pairs = (4 : ℚ) / 45 := by
  sorry

end probability_of_inverse_proportion_l380_38099


namespace value_of_a2_l380_38044

theorem value_of_a2 
  (a1 a2 a3 : ℝ)
  (h_seq : ∃ d : ℝ, (-8) = -8 + d * 0 ∧ a1 = -8 + d * 1 ∧ 
                     a2 = -8 + d * 2 ∧ a3 = -8 + d * 3 ∧ 
                     10 = -8 + d * 4) :
  a2 = 1 :=
by {
  sorry
}

end value_of_a2_l380_38044


namespace locus_of_points_l380_38096

-- Define points A and B
variable {A B : (ℝ × ℝ)}
-- Define constant d
variable {d : ℝ}

-- Definition of the distances
def distance_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem locus_of_points (A B : (ℝ × ℝ)) (d : ℝ) :
  ∀ M : (ℝ × ℝ), distance_sq M A - distance_sq M B = d ↔ 
  ∃ x : ℝ, ∃ y : ℝ, (M.1, M.2) = (x, y) ∧ 
  x = ((B.1 - A.1)^2 + d) / (2 * (B.1 - A.1)) :=
by
  sorry

end locus_of_points_l380_38096


namespace total_flowers_eaten_l380_38015

-- Definitions based on conditions
def num_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Statement asserting the total number of flowers eaten
theorem total_flowers_eaten : num_bugs * flowers_per_bug = 6 := by
  sorry

end total_flowers_eaten_l380_38015


namespace min_value_expression_l380_38004

theorem min_value_expression (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) :
  xy + (1 / xy) = 17 / 4 :=
sorry

end min_value_expression_l380_38004


namespace max_pqrs_squared_l380_38029

theorem max_pqrs_squared (p q r s : ℝ)
  (h1 : p + q = 18)
  (h2 : pq + r + s = 85)
  (h3 : pr + qs = 190)
  (h4 : rs = 120) :
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
sorry

end max_pqrs_squared_l380_38029


namespace box_height_l380_38034

theorem box_height (x : ℝ) (hx : x + 5 = 10)
  (surface_area : 2*x^2 + 4*x*(x + 5) ≥ 150) : x + 5 = 10 :=
sorry

end box_height_l380_38034


namespace Lucy_total_groceries_l380_38010

theorem Lucy_total_groceries :
  let packs_of_cookies := 12
  let packs_of_noodles := 16
  let boxes_of_cereals := 5
  let packs_of_crackers := 45
  (packs_of_cookies + packs_of_noodles + packs_of_crackers + boxes_of_cereals) = 78 :=
by
  sorry

end Lucy_total_groceries_l380_38010


namespace power_function_result_l380_38039
noncomputable def f (x : ℝ) (k : ℝ) (n : ℝ) : ℝ := k * x ^ n

theorem power_function_result (k n : ℝ) (h1 : f 27 k n = 3) : f 8 k (1/3) = 2 :=
by 
  sorry

end power_function_result_l380_38039


namespace eight_odot_six_eq_ten_l380_38000

-- Define the operation ⊙ as given in the problem statement
def operation (a b : ℕ) : ℕ := a + (3 * a) / (2 * b)

-- State the theorem to prove
theorem eight_odot_six_eq_ten : operation 8 6 = 10 :=
by
  -- Here you will provide the proof, but we skip it with sorry
  sorry

end eight_odot_six_eq_ten_l380_38000


namespace general_term_formula_l380_38057

theorem general_term_formula (n : ℕ) :
  ∀ (S : ℕ → ℝ), (∀ k : ℕ, S k = 1 - 2^k) → 
  (∀ a : ℕ → ℝ, a 1 = (S 1) ∧ (∀ m : ℕ, m > 1 → a m = S m - S (m - 1)) → 
  a n = -2 ^ (n - 1)) :=
by
  intro S hS a ha
  sorry

end general_term_formula_l380_38057


namespace greatest_integer_gcd_6_l380_38088

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l380_38088


namespace percentage_less_than_l380_38054

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l380_38054


namespace probability_same_unit_l380_38043

theorem probability_same_unit
  (units : ℕ) (people : ℕ) (same_unit_cases total_cases : ℕ)
  (h_units : units = 4)
  (h_people : people = 2)
  (h_total_cases : total_cases = units * units)
  (h_same_unit_cases : same_unit_cases = units) :
  (same_unit_cases :  ℝ) / total_cases = 1 / 4 :=
by sorry

end probability_same_unit_l380_38043


namespace value_of_a_l380_38068

-- Definitions based on conditions
def cond1 (a : ℝ) := |a| - 1 = 0
def cond2 (a : ℝ) := a + 1 ≠ 0

-- The main proof problem
theorem value_of_a (a : ℝ) : (cond1 a ∧ cond2 a) → a = 1 :=
by
  sorry

end value_of_a_l380_38068


namespace minimize_shoes_l380_38040

-- Definitions for inhabitants, one-legged inhabitants, and shoe calculations
def total_inhabitants := 10000
def P (percent_one_legged : ℕ) := (percent_one_legged * total_inhabitants) / 100
def non_one_legged (percent_one_legged : ℕ) := total_inhabitants - (P percent_one_legged)
def non_one_legged_with_shoes (percent_one_legged : ℕ) := (non_one_legged percent_one_legged) / 2
def shoes_needed (percent_one_legged : ℕ) := 
  (P percent_one_legged) + 2 * (non_one_legged_with_shoes percent_one_legged)

-- Theorem to prove that 100% one-legged minimizes the shoes required
theorem minimize_shoes : ∀ (percent_one_legged : ℕ), shoes_needed percent_one_legged = total_inhabitants → percent_one_legged = 100 :=
by
  intros percent_one_legged h
  sorry

end minimize_shoes_l380_38040


namespace no_solution_values_l380_38089

theorem no_solution_values (m : ℝ) :
  (∀ x : ℝ, x ≠ 5 → x ≠ -5 → (1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25))) ↔
  m = -1 ∨ m = 5 ∨ m = -5 / 11 :=
by
  sorry

end no_solution_values_l380_38089


namespace h_value_l380_38058

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem h_value :
  ∃ (h : ℝ → ℝ), (h 0 = 7)
  ∧ (∃ (a b c : ℝ), (f a = 0) ∧ (f b = 0) ∧ (f c = 0) ∧ (h (-8) = (1/49) * (-8 - a^3) * (-8 - b^3) * (-8 - c^3))) 
  ∧ h (-8) = -1813 := by
  sorry

end h_value_l380_38058


namespace arith_seq_sum_7_8_9_l380_38070

noncomputable def S_n (a : Nat → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n.succ).sum a

def arith_seq (a : Nat → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

theorem arith_seq_sum_7_8_9 (a : Nat → ℝ) (h_arith : arith_seq a)
    (h_S3 : S_n a 3 = 8) (h_S6 : S_n a 6 = 7) : 
  (a 7 + a 8 + a 9) = 1 / 8 := 
  sorry

end arith_seq_sum_7_8_9_l380_38070


namespace certain_number_x_l380_38090

theorem certain_number_x (p q x : ℕ) (hp : p > 1) (hq : q > 1)
  (h_eq : x * (p + 1) = 21 * (q + 1)) 
  (h_sum : p + q = 36) : x = 245 := 
by 
  sorry

end certain_number_x_l380_38090


namespace circles_intersect_l380_38032

theorem circles_intersect (R r d: ℝ) (hR: R = 7) (hr: r = 4) (hd: d = 8) : (R - r < d) ∧ (d < R + r) :=
by
  rw [hR, hr, hd]
  exact ⟨by linarith, by linarith⟩

end circles_intersect_l380_38032


namespace abs_triangle_inequality_l380_38027

theorem abs_triangle_inequality {a : ℝ} (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 :=
sorry

end abs_triangle_inequality_l380_38027


namespace quadratic_inequality_iff_abs_a_le_2_l380_38050

theorem quadratic_inequality_iff_abs_a_le_2 (a : ℝ) :
  (|a| ≤ 2) ↔ (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) :=
sorry

end quadratic_inequality_iff_abs_a_le_2_l380_38050


namespace trapezoid_triangle_area_ratio_l380_38091

/-- Given a trapezoid with triangles ABC and ADC such that the ratio of their areas is 4:1 and AB + CD = 150 cm.
Prove that the length of segment AB is 120 cm. --/
theorem trapezoid_triangle_area_ratio
  (h ABC_area ADC_area : ℕ)
  (AB CD : ℕ)
  (h_ratio : ABC_area / ADC_area = 4)
  (area_ABC : ABC_area = AB * h / 2)
  (area_ADC : ADC_area = CD * h / 2)
  (h_sum : AB + CD = 150) :
  AB = 120 := 
sorry

end trapezoid_triangle_area_ratio_l380_38091


namespace stratified_sampling_l380_38071

-- Definitions
def total_staff : ℕ := 150
def senior_titles : ℕ := 45
def intermediate_titles : ℕ := 90
def clerks : ℕ := 15
def sample_size : ℕ := 10

-- Ratios for stratified sampling
def senior_sample : ℕ := (senior_titles * sample_size) / total_staff
def intermediate_sample : ℕ := (intermediate_titles * sample_size) / total_staff
def clerks_sample : ℕ := (clerks * sample_size) / total_staff

-- Theorem statement
theorem stratified_sampling :
  senior_sample = 3 ∧ intermediate_sample = 6 ∧ clerks_sample = 1 :=
by
  sorry

end stratified_sampling_l380_38071


namespace circle_condition_l380_38069

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_l380_38069


namespace both_A_and_B_are_Gnomes_l380_38081

inductive Inhabitant
| Elf
| Gnome

open Inhabitant

def lies_about_gold (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def tells_truth_about_others (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def A_statement : Prop := ∀ i : Inhabitant, lies_about_gold i → i = Gnome
def B_statement : Prop := ∀ i : Inhabitant, tells_truth_about_others i → i = Gnome

theorem both_A_and_B_are_Gnomes (A_statement_true : A_statement) (B_statement_true : B_statement) :
  ∀ i : Inhabitant, (lies_about_gold i ∧ tells_truth_about_others i) → i = Gnome :=
by
  sorry

end both_A_and_B_are_Gnomes_l380_38081


namespace probability_first_prize_both_distribution_of_X_l380_38072

-- Definitions for the conditions
def total_students : ℕ := 500
def male_students : ℕ := 200
def female_students : ℕ := 300

def male_first_prize : ℕ := 10
def female_first_prize : ℕ := 25

def male_second_prize : ℕ := 15
def female_second_prize : ℕ := 25

def male_third_prize : ℕ := 15
def female_third_prize : ℕ := 40

-- Part (1): Prove the probability that both selected students receive the first prize is 1/240.
theorem probability_first_prize_both :
  (male_first_prize / male_students : ℚ) * (female_first_prize / female_students : ℚ) = 1 / 240 := 
sorry

-- Part (2): Prove the distribution of X.
def P_male_award : ℚ := (male_first_prize + male_second_prize + male_third_prize) / male_students
def P_female_award : ℚ := (female_first_prize + female_second_prize + female_third_prize) / female_students

theorem distribution_of_X :
  ∀ X : ℕ, X = 0 ∧ ((1 - P_male_award) * (1 - P_female_award) = 28 / 50) ∨ 
           X = 1 ∧ ((1 - P_male_award) * (1 - P_female_award) + (P_male_award * (1 - P_female_award)) + ((1 - P_male_award) * P_female_award) = 19 / 50) ∨ 
           X = 2 ∧ (P_male_award * P_female_award = 3 / 50) :=
sorry

end probability_first_prize_both_distribution_of_X_l380_38072


namespace percentage_decrease_of_original_number_is_30_l380_38061

theorem percentage_decrease_of_original_number_is_30 :
  ∀ (original_number : ℕ) (difference : ℕ) (percent_increase : ℚ) (percent_decrease : ℚ),
  original_number = 40 →
  percent_increase = 0.25 →
  difference = 22 →
  original_number + percent_increase * original_number - (original_number - percent_decrease * original_number) = difference →
  percent_decrease = 0.30 :=
by
  intros original_number difference percent_increase percent_decrease h_original h_increase h_diff h_eq
  sorry

end percentage_decrease_of_original_number_is_30_l380_38061


namespace betty_cookies_brownies_l380_38060

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l380_38060


namespace problem_l380_38051

-- Define the problem conditions and the statement that needs to be proved
theorem problem:
  ∀ (x : ℝ), (x ∈ Set.Icc (-1) m) ∧ ((1 - (-1)) / (m - (-1)) = 2 / 5) → m = 4 := by
  sorry

end problem_l380_38051


namespace sqrt_12_estimate_l380_38048

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l380_38048
