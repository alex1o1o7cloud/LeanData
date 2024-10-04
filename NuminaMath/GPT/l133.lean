import Mathlib

namespace integer_fraction_condition_l133_133188

theorem integer_fraction_condition (p : ℕ) (h_pos : 0 < p) :
  (∃ k : ℤ, k > 0 ∧ (5 * p + 15) = k * (3 * p - 9)) ↔ (4 ≤ p ∧ p ≤ 19) :=
by
  sorry

end integer_fraction_condition_l133_133188


namespace fraction_unchanged_when_multiplied_by_3_l133_133723

variable (x y : ℚ)

theorem fraction_unchanged_when_multiplied_by_3 (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x) / (3 * (3 * x + y)) = x / (3 * x + y) :=
by
  sorry

end fraction_unchanged_when_multiplied_by_3_l133_133723


namespace range_of_m_l133_133030

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 1 > m

def proposition_q (m : ℝ) : Prop :=
  3 - m > 1

theorem range_of_m (m : ℝ) (p_false : ¬proposition_p m) (q_true : proposition_q m) (pq_false : ¬(proposition_p m ∧ proposition_q m)) (porq_true : proposition_p m ∨ proposition_q m) : 
  1 ≤ m ∧ m < 2 := 
sorry

end range_of_m_l133_133030


namespace find_t_l133_133078

theorem find_t (p q r s t : ℤ)
  (h₁ : p - q - r + s - t = -t)
  (h₂ : p - (q - (r - (s - t))) = -4 + t) :
  t = 2 := 
sorry

end find_t_l133_133078


namespace find_a_l133_133589

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) (ha_pos : a > 0) (ha_neq1 : a ≠ 1) : a = 2 :=
sorry

end find_a_l133_133589


namespace find_digits_l133_133998

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l133_133998


namespace equivalent_expr_l133_133238

theorem equivalent_expr (a y : ℝ) (ha : a ≠ 0) (hy : y ≠ a ∧ y ≠ -a) :
  ( (a / (a + y) + y / (a - y)) / ( y / (a + y) - a / (a - y)) ) = -1 :=
by
  sorry

end equivalent_expr_l133_133238


namespace man_rowing_speed_l133_133952

noncomputable def rowing_speed_in_still_water : ℝ :=
  let distance := 0.1   -- kilometers
  let time := 20 / 3600 -- hours
  let current_speed := 3 -- km/hr
  let downstream_speed := distance / time
  downstream_speed - current_speed

theorem man_rowing_speed :
  rowing_speed_in_still_water = 15 :=
  by
    -- Proof comes here
    sorry

end man_rowing_speed_l133_133952


namespace loaves_on_friday_l133_133365

theorem loaves_on_friday
  (bread_wed : ℕ)
  (bread_thu : ℕ)
  (bread_sat : ℕ)
  (bread_sun : ℕ)
  (bread_mon : ℕ)
  (inc_wed_thu : bread_thu - bread_wed = 2)
  (inc_sat_sun : bread_sun - bread_sat = 5)
  (inc_sun_mon : bread_mon - bread_sun = 6)
  (pattern : ∀ n : ℕ, bread_wed + (2 + n) + n = bread_thu + n)
  : bread_thu + 3 = 10 := 
sorry

end loaves_on_friday_l133_133365


namespace probability_rain_all_days_l133_133368

noncomputable def probability_rain_each_day := 
  ((2 / 5) : ℚ, (1 / 2) : ℚ, (3 / 10) : ℚ)

theorem probability_rain_all_days (p_fri p_sat p_sun : ℚ) (h_fri : p_fri = 2 / 5) (h_sat : p_sat = 1 / 2) (h_sun : p_sun = 3 / 10) :
  (p_fri * p_sat * p_sun * 100 : ℚ) = 6 :=
by 
  calc
    (p_fri * p_sat * p_sun * 100 : ℚ) 
        = (2 / 5 * 1 / 2 * 3 / 10 * 100 : ℚ) : by rw [h_fri, h_sat, h_sun]
    ... = 6 : by norm_num

end probability_rain_all_days_l133_133368


namespace Elle_in_seat_2_given_conditions_l133_133840

theorem Elle_in_seat_2_given_conditions
    (seats : Fin 4 → Type) -- Representation of the seating arrangement.
    (Garry Elle Fiona Hank : Type)
    (seat_of : Type → Fin 4)
    (h1 : seat_of Garry = 0) -- Garry is in seat #1 (index 0)
    (h2 : ¬ (seat_of Elle = seat_of Hank + 1 ∨ seat_of Elle = seat_of Hank - 1)) -- Elle is not next to Hank
    (h3 : ¬ (seat_of Fiona > seat_of Garry ∧ seat_of Fiona < seat_of Hank) ∧ ¬ (seat_of Fiona < seat_of Garry ∧ seat_of Fiona > seat_of Hank)) -- Fiona is not between Garry and Hank
    : seat_of Elle = 1 :=  -- Conclusion: Elle is in seat #2 (index 1)
    sorry

end Elle_in_seat_2_given_conditions_l133_133840


namespace symmetric_function_l133_133166

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def symmetric_about_axis (f : ℤ → ℤ) (axis : ℤ) : Prop :=
  ∀ x : ℤ, f (axis - x) = f (axis + x)

theorem symmetric_function (a : ℕ → ℤ) (d : ℤ) (f : ℤ → ℤ) (a1 a2 : ℤ) (axis : ℤ) :
  (∀ x, f x = |x - a1| + |x - a2|) →
  arithmetic_sequence a d →
  d ≠ 0 →
  axis = (a1 + a2) / 2 →
  symmetric_about_axis f axis :=
by
  -- Proof goes here
  sorry

end symmetric_function_l133_133166


namespace maddie_episodes_friday_l133_133880

theorem maddie_episodes_friday :
  let total_episodes : ℕ := 8
  let episode_duration : ℕ := 44
  let monday_time : ℕ := 138
  let thursday_time : ℕ := 21
  let weekend_time : ℕ := 105
  let total_time : ℕ := total_episodes * episode_duration
  let non_friday_time : ℕ := monday_time + thursday_time + weekend_time
  let friday_time : ℕ := total_time - non_friday_time
  let friday_episodes : ℕ := friday_time / episode_duration
  friday_episodes = 2 :=
by
  sorry

end maddie_episodes_friday_l133_133880


namespace pentagon_area_eq_half_l133_133195

variables {A B C D E : Type*} -- Assume A, B, C, D, E are some points in a plane

-- Assume the given conditions in the problem
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)
variables (pentagon_area : ℝ)

-- Assume the constraints from the problem statement
axiom angle_A_eq_90 : angle_A = 90
axiom angle_C_eq_90 : angle_C = 90
axiom AB_eq_AE : AB = AE
axiom BC_eq_CD : BC = CD
axiom AC_eq_1 : AC = 1

theorem pentagon_area_eq_half : pentagon_area = 1 / 2 :=
sorry

end pentagon_area_eq_half_l133_133195


namespace days_in_month_l133_133951

theorem days_in_month 
  (S : ℕ) (D : ℕ) (h1 : 150 * S + 120 * D = (S + D) * 125) (h2 : S = 5) :
  S + D = 30 :=
by
  sorry

end days_in_month_l133_133951


namespace cookies_difference_l133_133357

-- Define the initial conditions
def initial_cookies : ℝ := 57
def cookies_eaten : ℝ := 8.5
def cookies_bought : ℝ := 125.75

-- Problem statement
theorem cookies_difference (initial_cookies cookies_eaten cookies_bought : ℝ) : 
  cookies_bought - cookies_eaten = 117.25 := 
sorry

end cookies_difference_l133_133357


namespace least_positive_linear_combination_24_18_l133_133936

theorem least_positive_linear_combination_24_18 (x y : ℤ) :
  ∃ (a : ℤ) (b : ℤ), 24 * a + 18 * b = 6 :=
by
  use 1
  use -1
  sorry

end least_positive_linear_combination_24_18_l133_133936


namespace magic_square_y_value_l133_133595

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end magic_square_y_value_l133_133595


namespace percentage_reduction_is_20_l133_133899

noncomputable def reduction_in_length (L W : ℝ) (x : ℝ) := 
  (L * (1 - x / 100)) * (W * 1.25) = L * W

theorem percentage_reduction_is_20 (L W : ℝ) : 
  reduction_in_length L W 20 := 
by 
  unfold reduction_in_length
  sorry

end percentage_reduction_is_20_l133_133899


namespace merchant_mixture_l133_133657

theorem merchant_mixture :
  ∃ (x y z : ℤ), x + y + z = 560 ∧ 70 * x + 64 * y + 50 * z = 33600 := by
  sorry

end merchant_mixture_l133_133657


namespace selection_options_l133_133507

theorem selection_options (group1 : Fin 5) (group2 : Fin 4) : (group1.1 + group2.1 + 1 = 9) :=
sorry

end selection_options_l133_133507


namespace total_pencils_correct_l133_133147

variable (donna_pencils marcia_pencils cindi_pencils : ℕ)

-- Given conditions translated into Lean
def condition1 : Prop := donna_pencils = 3 * marcia_pencils
def condition2 : Prop := marcia_pencils = 2 * cindi_pencils
def condition3 : Prop := cindi_pencils = 30 / 0.5

-- The proof statement
theorem total_pencils_correct : 
  condition1 ∧ condition2 ∧ condition3 → donna_pencils + marcia_pencils = 480 :=
begin
  -- Placeholder for the actual proof
  sorry
end

end total_pencils_correct_l133_133147


namespace Mr_A_financial_outcome_l133_133610

def home_worth : ℝ := 200000
def profit_percent : ℝ := 0.15
def loss_percent : ℝ := 0.05

def selling_price := (1 + profit_percent) * home_worth
def buying_price := (1 - loss_percent) * selling_price

theorem Mr_A_financial_outcome : 
  selling_price - buying_price = 11500 :=
by
  sorry

end Mr_A_financial_outcome_l133_133610


namespace find_integer_n_l133_133300

noncomputable def cubic_expr_is_pure_integer (n : ℤ) : Prop :=
  (729 * n ^ 6 - 540 * n ^ 4 + 240 * n ^ 2 - 64 : ℂ).im = 0

theorem find_integer_n :
  ∃! n : ℤ, cubic_expr_is_pure_integer n := 
sorry

end find_integer_n_l133_133300


namespace minimum_value_inverse_sum_l133_133179

variables {m n : ℝ}

theorem minimum_value_inverse_sum 
  (hm : m > 0) 
  (hn : n > 0) 
  (hline : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1)
  (hchord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ m * x2 + n * y2 + 2 = 0 → 
    (x1 - x2)^2 + (y1 - y2)^2 = 4) : 
  ∃ m n : ℝ, 3 * m + n = 2 ∧ m > 0 ∧ n > 0 ∧ 
    (∀ m' n' : ℝ, 3 * m' + n' = 2 → m' > 0 → n' > 0 → 
      (1 / m' + 3 / n' ≥ 6)) :=
sorry

end minimum_value_inverse_sum_l133_133179


namespace rachel_fathers_age_when_rachel_is_25_l133_133474

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l133_133474


namespace find_c_plus_d_l133_133187

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l133_133187


namespace playground_area_l133_133240

theorem playground_area (B : ℕ) (L : ℕ) (playground_area : ℕ) 
  (h1 : L = 8 * B) 
  (h2 : L = 240) 
  (h3 : playground_area = (1 / 6) * (L * B)) : 
  playground_area = 1200 :=
by
  sorry

end playground_area_l133_133240


namespace angles_on_line_y_eq_x_l133_133909

-- Define a predicate representing that an angle has its terminal side on the line y = x
def angle_on_line_y_eq_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- The goal is to prove that the set of all such angles is as stated
theorem angles_on_line_y_eq_x :
  { α : ℝ | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 } = { α : ℝ | angle_on_line_y_eq_x α } :=
sorry

end angles_on_line_y_eq_x_l133_133909


namespace ellipse_hyperbola_equation_l133_133867

-- Definitions for the Ellipse and Hyperbola
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / 10 + (y^2) / m = 1
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2) - (y^2) / b = 1

-- Conditions
def same_foci (c1 c2 : ℝ) : Prop := c1 = c2
def intersection_at_p (x y : ℝ) : Prop := x = (Real.sqrt 10) / 3 ∧ (ellipse x y 1 ∧ hyperbola x y 8)

-- Theorem stating the mathematically equivalent proof problem
theorem ellipse_hyperbola_equation :
  ∀ (m b : ℝ) (x y : ℝ), ellipse x y m ∧ hyperbola x y b ∧ same_foci (Real.sqrt (10 - m)) (Real.sqrt (1 + b)) ∧ intersection_at_p x y
  → (m = 1) ∧ (b = 8) := 
by
  intros m b x y h
  sorry

end ellipse_hyperbola_equation_l133_133867


namespace tens_digit_of_3_pow_100_l133_133923

-- Definition: The cyclic behavior of the last two digits of 3^n.
def last_two_digits_cycle : List ℕ := [03, 09, 27, 81, 43, 29, 87, 61, 83, 49, 47, 41, 23, 69, 07, 21, 63, 89, 67, 01]

-- Condition: The length of the cycle of the last two digits of 3^n.
def cycle_length : ℕ := 20

-- Assertion: The last two digits of 3^20 is 01.
def last_two_digits_3_pow_20 : ℕ := 1

-- Given n = 100, the tens digit of 3^n when n is expressed in decimal notation
theorem tens_digit_of_3_pow_100 : (3 ^ 100 / 10) % 10 = 0 := by
  let n := 100
  let position_in_cycle := (n % cycle_length)
  have cycle_repeat : (n % cycle_length = 0) := rfl
  have digits_3_pow_20 : (3^20 % 100 = 1) := by sorry
  show (3 ^ 100 / 10) % 10 = 0
  sorry

end tens_digit_of_3_pow_100_l133_133923


namespace flower_pattern_perimeter_l133_133059

theorem flower_pattern_perimeter (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 45) : 
    let arc_length := (360 - θ) / 360 * 2 * π * r
    let total_perimeter := arc_length + 2 * r
    total_perimeter = (21 / 4 * π) + 6 := 
by
  -- Definitions from conditions
  let arc_length := (360 - θ) / 360 * 2 * π * r
  let total_perimeter := arc_length + 2 * r

  -- Assertions to reach the target conclusion
  have h_arc_length: arc_length = (21 / 4 * π) :=
    by
      sorry

  -- Incorporate the radius
  have h_total: total_perimeter = (21 / 4 * π) + 6 :=
    by
      sorry

  exact h_total

end flower_pattern_perimeter_l133_133059


namespace jennie_speed_difference_l133_133600

noncomputable def average_speed_difference : ℝ :=
  let distance := 200
  let time_heavy_traffic := 5
  let construction_delay := 0.5
  let rest_stops_heavy := 0.5
  let time_no_traffic := 4
  let rest_stops_no_traffic := 1 / 3
  let actual_driving_time_heavy := time_heavy_traffic - construction_delay - rest_stops_heavy
  let actual_driving_time_no := time_no_traffic - rest_stops_no_traffic
  let average_speed_heavy := distance / actual_driving_time_heavy
  let average_speed_no := distance / actual_driving_time_no
  average_speed_no - average_speed_heavy

theorem jennie_speed_difference :
  average_speed_difference = 4.5 :=
sorry

end jennie_speed_difference_l133_133600


namespace find_f_2009_l133_133173

-- Defining the function f and specifying the conditions
variable (f : ℝ → ℝ)
axiom h1 : f 3 = -Real.sqrt 3
axiom h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x

-- Proving the desired statement
theorem find_f_2009 : f 2009 = 2 + Real.sqrt 3 :=
sorry

end find_f_2009_l133_133173


namespace find_g2_l133_133096

variable (g : ℝ → ℝ)

def condition (x : ℝ) : Prop :=
  g x - 2 * g (1 / x) = 3^x

theorem find_g2 (h : ∀ x ≠ 0, condition g x) : g 2 = -3 - (4 * Real.sqrt 3) / 9 :=
  sorry

end find_g2_l133_133096


namespace polynomials_with_sum_of_abs_values_and_degree_eq_4_l133_133562

-- We define the general structure and conditions of the problem.
def polynomial_count : ℕ := 
  let count_0 := 1 -- For n = 0
  let count_1 := 6 -- For n = 1
  let count_2 := 9 -- For n = 2
  let count_3 := 1 -- For n = 3
  count_0 + count_1 + count_2 + count_3

theorem polynomials_with_sum_of_abs_values_and_degree_eq_4 : polynomial_count = 17 := 
by
  unfold polynomial_count
  -- The detailed proof steps for the count would go here
  sorry

end polynomials_with_sum_of_abs_values_and_degree_eq_4_l133_133562


namespace sufficient_not_necessary_condition_l133_133319

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ (a : ℝ), a = 2 → (-(a) * (a / 4) = -1)) ∧ ∀ (a : ℝ), (-(a) * (a / 4) = -1 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_not_necessary_condition_l133_133319


namespace gcd_182_98_l133_133016

theorem gcd_182_98 : Nat.gcd 182 98 = 14 :=
by
  -- Provide the proof here, but as per instructions, we'll use sorry to skip it.
  sorry

end gcd_182_98_l133_133016


namespace more_cats_than_dogs_l133_133505

-- Define the number of cats and dogs
def c : ℕ := 23
def d : ℕ := 9

-- The theorem we need to prove
theorem more_cats_than_dogs : c - d = 14 := by
  sorry

end more_cats_than_dogs_l133_133505


namespace maximum_piles_l133_133772

theorem maximum_piles (n : ℕ) (h : n = 660) : 
  ∃ m, m = 30 ∧ 
       ∀ (piles : Finset ℕ), (piles.sum id = n) →
       (∀ x ∈ piles, ∀ y ∈ piles, x ≤ y → y < 2 * x) → 
       (piles.card ≤ m) :=
by
  sorry

end maximum_piles_l133_133772


namespace max_piles_l133_133776

open Finset

-- Define the condition for splitting and constraints
def valid_pile_splitting (initial_pile : ℕ) : Prop :=
  ∃ (piles : Finset ℕ), 
    (∑ x in piles, x = initial_pile) ∧ 
    (∀ x ∈ piles, ∀ y ∈ piles, x ≠ y → x < 2 * y) 

-- Define the theorem stating the maximum number of piles
theorem max_piles (initial_pile : ℕ) (h : initial_pile = 660) : 
  ∃ (n : ℕ) (piles : Finset ℕ), valid_pile_splitting initial_pile ∧ pile.card = 30 := 
sorry

end max_piles_l133_133776


namespace mary_gave_becky_green_crayons_l133_133463

-- Define the initial conditions
def initial_green_crayons : Nat := 5
def initial_blue_crayons : Nat := 8
def given_blue_crayons : Nat := 1
def remaining_crayons : Nat := 9

-- Define the total number of crayons initially
def total_initial_crayons : Nat := initial_green_crayons + initial_blue_crayons

-- Define the number of crayons given away
def given_crayons : Nat := total_initial_crayons - remaining_crayons

-- The crux of the problem
def given_green_crayons : Nat :=
  given_crayons - given_blue_crayons

-- Formal statement of the theorem
theorem mary_gave_becky_green_crayons
  (h_initial_green : initial_green_crayons = 5)
  (h_initial_blue : initial_blue_crayons = 8)
  (h_given_blue : given_blue_crayons = 1)
  (h_remaining : remaining_crayons = 9) :
  given_green_crayons = 3 :=
by {
  -- This should be the body of the proof, but we'll skip it for now
  sorry
}

end mary_gave_becky_green_crayons_l133_133463


namespace sandy_gain_percent_l133_133532

def gain_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  (gain * 100) / total_cost

theorem sandy_gain_percent :
  gain_percent 900 300 1260 = 5 :=
by
  sorry

end sandy_gain_percent_l133_133532


namespace third_side_not_one_l133_133870

theorem third_side_not_one (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c ≠ 1) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end third_side_not_one_l133_133870


namespace candy_problem_l133_133259

theorem candy_problem
  (x y m : ℤ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hxy : x + y = 176)
  (hcond : x - m * (y - 16) = 47)
  (hm : m > 1) :
  x ≥ 131 := 
sorry

end candy_problem_l133_133259


namespace calculation_l133_133401

theorem calculation : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 :=
by
  -- We add "sorry" here to indicate where the proof would go.
  sorry

end calculation_l133_133401


namespace interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l133_133312

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi →
    ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧
      (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem axis_of_symmetry (k : ℤ) :
  ∃ x : ℝ, x = Real.pi / 3 + k * (Real.pi / 2) := sorry

theorem max_and_min_values :
  ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      ((f x = 2 ∧ x = Real.pi / 3) ∨ (f x = -1 ∧ x = 0))) := sorry

end interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l133_133312


namespace sin_identity_l133_133026

variable (α : ℝ)
axiom alpha_def : α = Real.pi / 7

theorem sin_identity : (Real.sin (3 * α)) ^ 2 - (Real.sin α) ^ 2 = Real.sin (2 * α) * Real.sin (3 * α) := 
by 
  sorry

end sin_identity_l133_133026


namespace second_pipe_fills_in_15_minutes_l133_133921

theorem second_pipe_fills_in_15_minutes :
  ∀ (x : ℝ),
  (∀ (x : ℝ), (1 / 2 + (7.5 / x)) = 1 → x = 15) :=
by
  intros
  sorry

end second_pipe_fills_in_15_minutes_l133_133921


namespace total_yards_run_l133_133350

-- Define the yardages and games for each athlete
def Malik_yards_per_game : ℕ := 18
def Malik_games : ℕ := 5

def Josiah_yards_per_game : ℕ := 22
def Josiah_games : ℕ := 7

def Darnell_yards_per_game : ℕ := 11
def Darnell_games : ℕ := 4

def Kade_yards_per_game : ℕ := 15
def Kade_games : ℕ := 6

-- Prove that the total yards run by the four athletes is 378
theorem total_yards_run :
  (Malik_yards_per_game * Malik_games) +
  (Josiah_yards_per_game * Josiah_games) +
  (Darnell_yards_per_game * Darnell_games) +
  (Kade_yards_per_game * Kade_games) = 378 :=
by
  sorry

end total_yards_run_l133_133350


namespace max_value_of_a2b3c4_l133_133457

open Real

theorem max_value_of_a2b3c4
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683 / 472392 :=
sorry

end max_value_of_a2b3c4_l133_133457


namespace find_k_l133_133752

-- Definitions of the conditions as given in the problem
def total_amount (A B C : ℕ) : Prop := A + B + C = 585
def c_share (C : ℕ) : Prop := C = 260
def equal_shares (A B C k : ℕ) : Prop := 4 * A = k * C ∧ 6 * B = k * C

-- The theorem we need to prove
theorem find_k (A B C k : ℕ) (h_tot: total_amount A B C)
  (h_c: c_share C) (h_eq: equal_shares A B C k) : k = 3 := by 
  sorry

end find_k_l133_133752


namespace sale_price_is_207_l133_133648

-- Definitions for the conditions given
def price_at_store_P : ℝ := 200
def regular_price_at_store_Q (price_P : ℝ) : ℝ := price_P * 1.15
def sale_price_at_store_Q (regular_price_Q : ℝ) : ℝ := regular_price_Q * 0.90

-- Goal: Prove the sale price of the bicycle at Store Q is 207
theorem sale_price_is_207 : sale_price_at_store_Q (regular_price_at_store_Q price_at_store_P) = 207 :=
by
  sorry

end sale_price_is_207_l133_133648


namespace total_pencils_l133_133286

theorem total_pencils (initial_additional1 initial_additional2 : ℕ) (h₁ : initial_additional1 = 37) (h₂ : initial_additional2 = 17) : (initial_additional1 + initial_additional2) = 54 :=
by sorry

end total_pencils_l133_133286


namespace other_group_less_garbage_l133_133462

theorem other_group_less_garbage :
  387 + (735 - 387) = 735 :=
by
  sorry

end other_group_less_garbage_l133_133462


namespace determine_m_l133_133838

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem determine_m (m : ℝ) : 3 * f 5 m = 2 * g 5 m → m = 10 / 7 := 
by sorry

end determine_m_l133_133838


namespace polygon_sides_l133_133499

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l133_133499


namespace Pria_drove_372_miles_l133_133470

theorem Pria_drove_372_miles (advertisement_mileage : ℕ) (tank_capacity : ℕ) (mileage_difference : ℕ) 
(h1 : advertisement_mileage = 35) 
(h2 : tank_capacity = 12) 
(h3 : mileage_difference = 4) : 
(advertisement_mileage - mileage_difference) * tank_capacity = 372 :=
by sorry

end Pria_drove_372_miles_l133_133470


namespace total_arrangements_excluding_zhang_for_shooting_event_l133_133629

theorem total_arrangements_excluding_zhang_for_shooting_event
  (students : Fin 5) 
  (events : Fin 3)
  (shooting : events ≠ 0) : 
  ∃ arrangements, arrangements = 48 := 
sorry

end total_arrangements_excluding_zhang_for_shooting_event_l133_133629


namespace find_x_l133_133172

-- Define the vectors and the condition of them being parallel
def vector_a : (ℝ × ℝ) := (3, 1)
def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -1)
def parallel (a b : (ℝ × ℝ)) := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The theorem to prove
theorem find_x (x : ℝ) (h : parallel (3, 1) (x, -1)) : x = -3 :=
by
  sorry

end find_x_l133_133172


namespace ratio_and_equation_imp_value_of_a_l133_133052

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l133_133052


namespace increasing_function_l133_133929

def fA (x : ℝ) : ℝ := -x
def fB (x : ℝ) : ℝ := (2 / 3) ^ x
def fC (x : ℝ) : ℝ := x ^ 2
def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function (x y : ℝ) (h : x < y) : fD x < fD y := sorry

end increasing_function_l133_133929


namespace price_of_bracelets_max_type_a_bracelets_l133_133957

-- Part 1: Proving the prices of the bracelets
theorem price_of_bracelets :
  ∃ (x y : ℝ), (3 * x + y = 128 ∧ x + 2 * y = 76) ∧ (x = 36 ∧ y = 20) :=
sorry

-- Part 2: Proving the maximum number of type A bracelets they can buy within the budget
theorem max_type_a_bracelets :
  ∃ (m : ℕ), 36 * m + 20 * (100 - m) ≤ 2500 ∧ m = 31 :=
sorry

end price_of_bracelets_max_type_a_bracelets_l133_133957


namespace solution_no_triangle_l133_133725

noncomputable def problem : Prop :=
  ∀ (A B C : ℝ) (a b c : ℝ), b = 4 ∧ c = 2 ∧ C = 60 → ¬ ∃ (A B : ℝ), (a / Real.sin A) = (b / Real.sin B) ∧ (a / Real.sin A) = (c / Real.sin C)

theorem solution_no_triangle (h : problem) : True := sorry

end solution_no_triangle_l133_133725


namespace ticket_price_reduction_l133_133250

-- Definitions of the problem constants and variables
def original_price : ℝ := 50
def increase_fraction : ℝ := 1 / 3
def revenue_increase_fraction : ℝ := 1 / 4

-- New number of tickets sold after price reduction
def new_number_of_tickets_sold (x : ℝ) : ℝ := x * (1 + increase_fraction)

-- New price per ticket after reduction
def new_price_per_ticket (reduction : ℝ) : ℝ := original_price - reduction

-- Original revenue
def original_revenue (x : ℝ) : ℝ := x * original_price

-- New revenue after price reduction
def new_revenue (x reduction : ℝ) : ℝ := new_number_of_tickets_sold x * new_price_per_ticket reduction

-- The equation relating new revenue to the original revenue with the given increase
def revenue_relation (x reduction : ℝ) : Prop :=
  new_revenue x reduction = (1 + revenue_increase_fraction) * original_revenue x

-- The goal is to find the reduction in price per ticket (reduction) such that the revenue_relation holds
theorem ticket_price_reduction :
  ∃ y : ℝ, ∀ x > 0, revenue_relation x y ∧ y = 25 / 2 :=
begin
  sorry -- Proof goes here
end

end ticket_price_reduction_l133_133250


namespace parabola_vertex_sum_l133_133557

theorem parabola_vertex_sum 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x + 3)^2 + 4))
  (h2 : (a * 49 + 4) = -2)
  : a + b + c = 100 / 49 :=
by
  sorry

end parabola_vertex_sum_l133_133557


namespace similar_triangles_x_value_l133_133659

-- Define the conditions of the problem
variables (x : ℝ) (h₁ : 10 / x = 8 / 5)

-- State the theorem/proof problem
theorem similar_triangles_x_value : x = 6.25 :=
by
  -- Proof goes here
  sorry

end similar_triangles_x_value_l133_133659


namespace show_length_50_l133_133848

def Gina_sSis_three_as_often (G S : ℕ) : Prop := G = 3 * S
def sister_total_shows (G S : ℕ) : Prop := G + S = 24
def Gina_total_minutes (G : ℕ) (minutes : ℕ) : Prop := minutes = 900
def length_of_each_show (minutes shows length : ℕ) : Prop := length = minutes / shows

theorem show_length_50 (G S : ℕ) (length : ℕ) :
  Gina_sSis_three_as_often G S →
  sister_total_shows G S →
  Gina_total_minutes G 900 →
  length_of_each_show 900 G length →
  length = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end show_length_50_l133_133848


namespace pencils_bought_l133_133146

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l133_133146


namespace product_mod_32_l133_133072

def product_of_all_odd_primes_less_than_32 : ℕ :=
  3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  (product_of_all_odd_primes_less_than_32) % 32 = 9 :=
sorry

end product_mod_32_l133_133072


namespace total_cost_computers_l133_133955

theorem total_cost_computers (B T : ℝ) 
  (cA : ℝ := 1.4 * B) 
  (cB : ℝ := B) 
  (tA : ℝ := T) 
  (tB : ℝ := T + 20) 
  (total_cost_A : ℝ := cA * tA)
  (total_cost_B : ℝ := cB * tB):
  total_cost_A = total_cost_B → 70 * B = total_cost_A := 
by
  sorry

end total_cost_computers_l133_133955


namespace debby_pictures_l133_133932

theorem debby_pictures : 
  let zoo_pics := 24
  let museum_pics := 12
  let pics_deleted := 14
  zoo_pics + museum_pics - pics_deleted = 22 := 
by
  sorry

end debby_pictures_l133_133932


namespace total_jumps_correct_l133_133380

-- Define Ronald's jumps
def Ronald_jumps : ℕ := 157

-- Define the difference in jumps between Rupert and Ronald
def difference : ℕ := 86

-- Define Rupert's jumps
def Rupert_jumps : ℕ := Ronald_jumps + difference

-- Define the total number of jumps
def total_jumps : ℕ := Ronald_jumps + Rupert_jumps

-- State the main theorem we want to prove
theorem total_jumps_correct : total_jumps = 400 := 
by sorry

end total_jumps_correct_l133_133380


namespace monthly_cost_per_person_is_1000_l133_133200

noncomputable def john_pays : ℝ := 32000
noncomputable def initial_fee_per_person : ℝ := 4000
noncomputable def total_people : ℝ := 4
noncomputable def john_pays_half : Prop := true

theorem monthly_cost_per_person_is_1000 :
  john_pays_half →
  (john_pays * 2 - (initial_fee_per_person * total_people)) / (total_people * 12) = 1000 :=
by
  intro h
  sorry

end monthly_cost_per_person_is_1000_l133_133200


namespace cost_of_each_steak_meal_l133_133753

variable (x : ℝ)

theorem cost_of_each_steak_meal :
  (2 * x + 2 * 3.5 + 3 * 2 = 99 - 38) → x = 24 := 
by
  intro h
  sorry

end cost_of_each_steak_meal_l133_133753


namespace billiard_expected_reflections_l133_133062

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_expected_reflections :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end billiard_expected_reflections_l133_133062


namespace H_function_is_f_x_abs_x_l133_133868

-- Definition: A function f is odd if ∀ x ∈ ℝ, f(-x) = -f(x)
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition: A function f is strictly increasing if ∀ x1, x2 ∈ ℝ, x1 < x2 implies f(x1) < f(x2)
def is_strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- Define the function f(x) = x * |x|
def f (x : ℝ) : ℝ := x * abs x

-- The main theorem which states that f(x) = x * |x| is an "H function"
theorem H_function_is_f_x_abs_x : is_odd f ∧ is_strictly_increasing f :=
  sorry

end H_function_is_f_x_abs_x_l133_133868


namespace option_D_is_correct_l133_133273

variable (a b : ℝ)

theorem option_D_is_correct :
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^2 + 3 * a ≠ 4 * a^2) ∧
  ((a + 2) * (a - 2) ≠ a^2 - 2) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end option_D_is_correct_l133_133273


namespace domain_of_function_l133_133487

-- Define the setting and the constants involved
variables {f : ℝ → ℝ}
variable {c : ℝ}

-- The statement about the function's domain
theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≤ 0 ∧ x ≠ -c) :=
sorry

end domain_of_function_l133_133487


namespace program_output_l133_133926

theorem program_output (a : ℕ) (h : a = 3) : (if a < 10 then 2 * a else a * a) = 6 :=
by
  rw [h]
  norm_num

end program_output_l133_133926


namespace equal_areas_of_quadrilaterals_l133_133075

open EuclideanGeometry

variables {A B C D E F K : Point}
variables {AB AD : Line}
variables {S : Real}

-- Conditions
def is_parallelogram (A B C D : Point) : Prop := is_parallel (A - B) (C - D) ∧ is_parallel (A - D) (B - C)
def lies_on (P : Point) (L : Line) : Prop := on_line P L
def segment_condition (P Q R : Point) : Prop := between Q P R

-- Problem statement
theorem equal_areas_of_quadrilaterals
  (h1 : is_parallelogram A B C D)
  (h2 : lies_on E AB)
  (h3 : lies_on F AD)
  (h4 : segment_condition A B E)
  (h5 : segment_condition A D F)
  (h6 : ∃ K, intersection_point (line_through E D) (line_through F B) K) :
    area_quadrilateral A B K D = area_quadrilateral C E K F :=
begin
  sorry -- Proof goes here
end

end equal_areas_of_quadrilaterals_l133_133075


namespace complex_div_equation_l133_133688

theorem complex_div_equation (z : ℂ) (h : z / (1 - 2 * complex.I) = complex.I) : 
  z = 2 + complex.I :=
sorry

end complex_div_equation_l133_133688


namespace maximum_piles_l133_133773

theorem maximum_piles (n : ℕ) (h : n = 660) : 
  ∃ m, m = 30 ∧ 
       ∀ (piles : Finset ℕ), (piles.sum id = n) →
       (∀ x ∈ piles, ∀ y ∈ piles, x ≤ y → y < 2 * x) → 
       (piles.card ≤ m) :=
by
  sorry

end maximum_piles_l133_133773


namespace popcorn_kernels_needed_l133_133536

theorem popcorn_kernels_needed
  (h1 : 2 * 4 = 4 * 1) -- Corresponds to "2 tablespoons make 4 cups"
  (joanie : 3) -- Joanie wants 3 cups
  (mitchell : 4) -- Mitchell wants 4 cups
  (miles_davis : 6) -- Miles and Davis together want 6 cups
  (cliff : 3) -- Cliff wants 3 cups
  : 2 * (joanie + mitchell + miles_davis + cliff) / 4 = 8 :=
by sorry

end popcorn_kernels_needed_l133_133536


namespace angle_bisectors_geq_nine_times_inradius_l133_133493

theorem angle_bisectors_geq_nine_times_inradius 
  (r : ℝ) (f_a f_b f_c : ℝ) 
  (h_triangle : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ r = (1 / 2) * (a + b + c) * r ∧ 
      f_a ≥ (2 * a * b / (a + b) + 2 * a * c / (a + c)) / 2 ∧ 
      f_b ≥ (2 * b * a / (b + a) + 2 * b * c / (b + c)) / 2 ∧ 
      f_c ≥ (2 * c * a / (c + a) + 2 * c * b / (c + b)) / 2)
  : f_a + f_b + f_c ≥ 9 * r :=
sorry

end angle_bisectors_geq_nine_times_inradius_l133_133493


namespace age_of_teacher_l133_133756

theorem age_of_teacher (S T : ℕ) (avg_students avg_total : ℕ) (num_students num_total : ℕ)
  (h1 : num_students = 50)
  (h2 : avg_students = 14)
  (h3 : num_total = 51)
  (h4 : avg_total = 15)
  (h5 : S = avg_students * num_students)
  (h6 : S + T = avg_total * num_total) :
  T = 65 := 
by {
  sorry
}

end age_of_teacher_l133_133756


namespace mod11_residue_l133_133263

theorem mod11_residue :
  (305 % 11 = 8) →
  (44 % 11 = 0) →
  (176 % 11 = 0) →
  (18 % 11 = 7) →
  (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end mod11_residue_l133_133263


namespace solve_fractional_equation_l133_133481

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) : 
  (2 * x) / (x - 1) = x / (3 * (x - 1)) + 1 ↔ x = -3 / 2 :=
by sorry

end solve_fractional_equation_l133_133481


namespace find_ratio_of_square_to_circle_radius_l133_133441

def sector_circle_ratio (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) : Prop :=
  (R = (5 * a * sqrt2) / 2) →
  (r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) →
  (a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2))

theorem find_ratio_of_square_to_circle_radius
  (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) (h1 : R = (5 * a * sqrt2) / 2)
  (h2 : r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) :
  a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2) :=
  sorry

end find_ratio_of_square_to_circle_radius_l133_133441


namespace find_x_l133_133145

theorem find_x (x : ℝ) : (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) → x = -5 :=
by
  sorry

end find_x_l133_133145


namespace line_passes_fixed_point_max_distance_eqn_l133_133578

-- Definition of the line equation
def line_eq (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Point P
def point_P : ℝ × ℝ :=
  (3, 4)

-- Fixed point that the line passes through
def fixed_point : ℝ × ℝ :=
  (-2, 3)

-- Statement that the line passes through the fixed point
theorem line_passes_fixed_point (a b : ℝ) :
  line_eq a b (-2) 3 :=
sorry

-- Equation of the line when distance from point P to line is maximized
def line_max_distance (a b : ℝ) : Prop :=
  5 * 3 + 4 + 7 = 0

-- Statement that the equation of the line is as given when distance is maximized
theorem max_distance_eqn (a b : ℝ) :
  line_max_distance a b :=
sorry

end line_passes_fixed_point_max_distance_eqn_l133_133578


namespace find_expression_roots_l133_133171

-- Define the roots of the given quadratic equation
def is_root (α : ℝ) : Prop := α ^ 2 - 2 * α - 1 = 0

-- Define the main statement to be proven
theorem find_expression_roots (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  5 * α ^ 4 + 12 * β ^ 3 = 169 := sorry

end find_expression_roots_l133_133171


namespace max_lateral_surface_area_cylinder_optimizes_l133_133559

noncomputable def max_lateral_surface_area_cylinder (r m : ℝ) : ℝ × ℝ :=
  let r_c := r / 2
  let h_c := m / 2
  (r_c, h_c)

theorem max_lateral_surface_area_cylinder_optimizes {r m : ℝ} (hr : 0 < r) (hm : 0 < m) :
  let (r_c, h_c) := max_lateral_surface_area_cylinder r m
  r_c = r / 2 ∧ h_c = m / 2 :=
sorry

end max_lateral_surface_area_cylinder_optimizes_l133_133559


namespace proof_problem_l133_133455

-- Definitions of arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r^n

-- Lean statement of the problem
theorem proof_problem 
  (a b : ℕ → ℝ)
  (h_a_arithmetic : is_arithmetic_sequence a)
  (h_b_geometric : is_geometric_sequence b)
  (h_condition : a 1 - (a 7)^2 + a 13 = 0)
  (h_b7_a7 : b 7 = a 7) :
  b 3 * b 11 = 4 :=
sorry

end proof_problem_l133_133455


namespace typing_time_together_l133_133376

theorem typing_time_together 
  (jonathan_time : ℝ)
  (susan_time : ℝ)
  (jack_time : ℝ)
  (document_pages : ℝ)
  (combined_time : ℝ) :
  jonathan_time = 40 →
  susan_time = 30 →
  jack_time = 24 →
  document_pages = 10 →
  combined_time = document_pages / ((document_pages / jonathan_time) + (document_pages / susan_time) + (document_pages / jack_time)) →
  combined_time = 10 :=
by sorry

end typing_time_together_l133_133376


namespace min_max_solution_A_l133_133254

theorem min_max_solution_A (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 5 * x + 8 * y + 9 * z = 700) 
                           (h₃ : 0 ≤ x ∧ x ≤ 60) (h₄ : 0 ≤ y ∧ y ≤ 60) (h₅ : 0 ≤ z ∧ z ≤ 47) :
    35 ≤ x ∧ x ≤ 49 :=
by
  sorry

end min_max_solution_A_l133_133254


namespace calculate_expression_l133_133826

theorem calculate_expression : (3.75 - 1.267 + 0.48 = 2.963) :=
by
  sorry

end calculate_expression_l133_133826


namespace largest_digit_divisible_by_6_l133_133519

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133519


namespace imaginary_unit_cube_l133_133800

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
by
  sorry

end imaginary_unit_cube_l133_133800


namespace sum_of_digits_of_x_squared_eq_36_l133_133133

noncomputable def base_r_representation_sum (r : ℕ) (x : ℕ) := ∃ (p q : ℕ), 
  r <= 36 ∧
  x = p * (r^3 + r^2) + q * (r + 1) ∧
  2 * q = 5 * p ∧
  ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a ∧
  b = 9 ∧
  a + b + c = 18

theorem sum_of_digits_of_x_squared_eq_36 (r x : ℕ) :
  base_r_representation_sum r x → ∑ d in (digits r (x^2)), d = 36 :=
sorry

end sum_of_digits_of_x_squared_eq_36_l133_133133


namespace line_tangent_to_circle_perpendicular_l133_133573

theorem line_tangent_to_circle_perpendicular 
  (l₁ l₂ : String)
  (C : String)
  (h1 : l₂ = "4 * x - 3 * y + 1 = 0")
  (h2 : C = "x^2 + y^2 + 2 * y - 3 = 0") :
  (l₁ = "3 * x + 4 * y + 14 = 0" ∨ l₁ = "3 * x + 4 * y - 6 = 0") :=
by
  sorry

end line_tangent_to_circle_perpendicular_l133_133573


namespace sum_of_reciprocals_is_3_over_8_l133_133369

theorem sum_of_reciprocals_is_3_over_8 (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  (1 / x + 1 / y) = 3 / 8 := 
by 
  sorry

end sum_of_reciprocals_is_3_over_8_l133_133369


namespace calculation_equality_l133_133291

theorem calculation_equality : ((8^5 / 8^2) * 4^4) = 2^17 := by
  sorry

end calculation_equality_l133_133291


namespace value_of_Y_l133_133045

-- Definitions for the conditions in part a)
def M := 2021 / 3
def N := M / 4
def Y := M + N

-- The theorem stating the question and its correct answer
theorem value_of_Y : Y = 843 := by
  sorry

end value_of_Y_l133_133045


namespace cookies_left_l133_133212

-- Define the conditions
def pounds_of_flour_used_per_batch : ℕ := 2
def batches_per_bakery_bag_of_flour : ℕ := 5
def total_bags_used : ℕ := 4
def cookies_per_batch : ℕ := 12
def cookies_eaten_by_jim : ℕ := 15

-- Calculate the total pounds of flour used
def total_pounds_of_flour := total_bags_used * batches_per_bakery_bag_of_flour

-- Calculate the total number of batches
def total_batches := total_pounds_of_flour / pounds_of_flour_used_per_batch

-- Calculate the total number of cookies cooked
def total_cookies := total_batches * cookies_per_batch

-- Calculate the number of cookies left
theorem cookies_left :
  let total_cookies := total_batches * cookies_per_batch in 
  total_cookies - cookies_eaten_by_jim = 105 :=
by
  sorry

end cookies_left_l133_133212


namespace cos_270_eq_zero_l133_133976

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l133_133976


namespace trigonometric_values_l133_133032

theorem trigonometric_values (α β : ℝ) (h1 : real.cos α = 1 / 7)
  (h2 : real.cos (α - β) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < real.pi / 2) :
  real.tan (2 * α) = -8 * real.sqrt 3 / 47 ∧ β = real.pi / 3 :=
by { sorry }

end trigonometric_values_l133_133032


namespace grasshopper_jump_distance_l133_133366

-- Definitions based on conditions
def frog_jump : ℤ := 39
def higher_jump_distance : ℤ := 22
def grasshopper_jump : ℤ := frog_jump - higher_jump_distance

-- The statement we need to prove
theorem grasshopper_jump_distance :
  grasshopper_jump = 17 :=
by
  -- Here, proof would be provided but we skip with sorry
  sorry

end grasshopper_jump_distance_l133_133366


namespace lines_perpendicular_l133_133054

theorem lines_perpendicular
  (k₁ k₂ : ℝ)
  (h₁ : k₁^2 - 3*k₁ - 1 = 0)
  (h₂ : k₂^2 - 3*k₂ - 1 = 0) :
  k₁ * k₂ = -1 → 
  (∃ l₁ l₂: ℝ → ℝ, 
    ∀ x, l₁ x = k₁ * x ∧ l₂ x = k₂ * x → 
    ∃ m, m = -1) := 
sorry

end lines_perpendicular_l133_133054


namespace cats_in_studio_count_l133_133824

theorem cats_in_studio_count :
  (70 + 40 + 30 + 50
  - 25 - 15 - 20 - 28
  + 5 + 10 + 12
  - 8
  + 12) = 129 :=
by sorry

end cats_in_studio_count_l133_133824


namespace number_of_episodes_last_season_more_than_others_l133_133451

-- Definitions based on conditions
def episodes_per_other_season : ℕ := 22
def initial_seasons : ℕ := 9
def duration_per_episode : ℚ := 0.5
def total_hours_after_last_season : ℚ := 112

-- Derived definitions based on conditions (not solution steps)
def total_hours_first_9_seasons := initial_seasons * episodes_per_other_season * duration_per_episode
def additional_hours_last_season := total_hours_after_last_season - total_hours_first_9_seasons
def episodes_last_season := additional_hours_last_season / duration_per_episode

-- Proof problem statement
theorem number_of_episodes_last_season_more_than_others : 
  episodes_last_season = episodes_per_other_season + 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_episodes_last_season_more_than_others_l133_133451


namespace real_number_x_equal_2_l133_133181

theorem real_number_x_equal_2 (x : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2 * i) * (x + i) = 4 - 3 * i → x = 2 :=
by
  sorry

end real_number_x_equal_2_l133_133181


namespace binomial_prob_1_l133_133040

noncomputable def bernoulli_trial (n : ℕ) (p : ℚ) : Measure ℕ :=
  Measure.dirac (binomial_distribution n p)

theorem binomial_prob_1 : (Probability := bernoulli_trial 3 (1/3)) (ProbabilityEvent.Exactly 1) = 4/9 :=
sorry

end binomial_prob_1_l133_133040


namespace lcm_of_48_and_14_is_56_l133_133373

theorem lcm_of_48_and_14_is_56 :
  ∀ n : ℕ, (n = 48 ∧ Nat.gcd n 14 = 12) → Nat.lcm n 14 = 56 :=
by
  intro n h
  sorry

end lcm_of_48_and_14_is_56_l133_133373


namespace baylor_final_amount_l133_133288

def CDA := 4000
def FCP := (1 / 2) * CDA
def SCP := FCP + (2 / 5) * FCP
def TCP := 2 * (FCP + SCP)
def FDA := CDA + FCP + SCP + TCP

theorem baylor_final_amount : FDA = 18400 := by
  sorry

end baylor_final_amount_l133_133288


namespace cos_270_eq_zero_l133_133975

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l133_133975


namespace math_problem_l133_133671

theorem math_problem : (3 ^ 456) + (9 ^ 5 / 9 ^ 3) = 82 := 
by 
  sorry

end math_problem_l133_133671


namespace donation_problem_l133_133193

theorem donation_problem
  (A B C D : Prop)
  (h1 : ¬A ↔ (B ∨ C ∨ D))
  (h2 : B ↔ D)
  (h3 : C ↔ ¬B) 
  (h4 : D ↔ ¬B): A := 
by
  sorry

end donation_problem_l133_133193


namespace jina_mascots_l133_133199

variables (x y z x_new Total : ℕ)

def mascots_problem :=
  (y = 3 * x) ∧
  (x_new = x + 2 * y) ∧
  (z = 2 * y) ∧
  (Total = x_new + y + z) →
  Total = 16 * x

-- The statement only, no proof is required
theorem jina_mascots : mascots_problem x y z x_new Total := sorry

end jina_mascots_l133_133199


namespace cos_100_eq_neg_sqrt_l133_133422

theorem cos_100_eq_neg_sqrt (a : ℝ) (h : Real.sin (80 * Real.pi / 180) = a) : 
  Real.cos (100 * Real.pi / 180) = -Real.sqrt (1 - a^2) := 
sorry

end cos_100_eq_neg_sqrt_l133_133422


namespace Marias_score_l133_133058

def total_questions := 30
def points_per_correct_answer := 20
def points_deducted_per_incorrect_answer := 5
def total_answered := total_questions
def correct_answers := 19
def incorrect_answers := total_questions - correct_answers
def score := (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect_answer)

theorem Marias_score : score = 325 := by
  -- proof goes here
  sorry

end Marias_score_l133_133058


namespace inequality_proof_l133_133690

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by
  sorry

end inequality_proof_l133_133690


namespace cost_to_fill_pool_l133_133067

/-- Definition of the pool dimensions and constants --/
def pool_length := 20
def pool_width := 6
def pool_depth := 10
def cubic_feet_to_liters := 25
def liter_cost := 3

/-- Calculating the cost to fill the pool --/
def pool_volume := pool_length * pool_width * pool_depth
def total_liters := pool_volume * cubic_feet_to_liters
def total_cost := total_liters * liter_cost

/-- Theorem stating that the total cost to fill the pool is $90,000 --/
theorem cost_to_fill_pool : total_cost = 90000 := by
  sorry

end cost_to_fill_pool_l133_133067


namespace find_a_l133_133700

theorem find_a {a : ℝ} (h : ∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5 → x ≤ 3) : a = 8 :=
sorry

end find_a_l133_133700


namespace five_algorithmic_statements_l133_133306

-- Define the five types of algorithmic statements in programming languages
inductive AlgorithmicStatement : Type
| input : AlgorithmicStatement
| output : AlgorithmicStatement
| assignment : AlgorithmicStatement
| conditional : AlgorithmicStatement
| loop : AlgorithmicStatement

-- Theorem: Every programming language contains these five basic types of algorithmic statements
theorem five_algorithmic_statements : 
  ∃ (s : List AlgorithmicStatement), 
    (s.length = 5) ∧ 
    ∀ x, x ∈ s ↔
    x = AlgorithmicStatement.input ∨
    x = AlgorithmicStatement.output ∨
    x = AlgorithmicStatement.assignment ∨
    x = AlgorithmicStatement.conditional ∨
    x = AlgorithmicStatement.loop :=
by
  sorry

end five_algorithmic_statements_l133_133306


namespace polar_to_rectangular_coordinates_l133_133339

theorem polar_to_rectangular_coordinates :
  let r := 2
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (1, Real.sqrt 3) :=
by
  sorry

end polar_to_rectangular_coordinates_l133_133339


namespace Donna_and_Marcia_total_pencils_l133_133148

def DonnaPencils (CindiPencils MarciaPencils DonnaPencils : ℕ) : Prop :=
  DonnaPencils = 3 * MarciaPencils

def MarciaPencils (CindiPencils MarciaPencils : ℕ) : Prop :=
  MarciaPencils = 2 * CindiPencils

def CindiPencils (CindiSpent CindiPencilCost CindiPencils : ℕ) : Prop :=
  CindiPencils = CindiSpent / CindiPencilCost

theorem Donna_and_Marcia_total_pencils (CindiSpent CindiPencilCost : ℕ) (DonnaPencils MarciaPencils CindiPencils : ℕ)
  (hCindi : CindiPencils CindiSpent CindiPencilCost CindiPencils)
  (hMarcia : MarciaPencils CindiPencils MarciaPencils)
  (hDonna : DonnaPencils CindiPencils MarciaPencils DonnaPencils) :
  DonnaPencils + MarciaPencils = 480 := 
sorry

end Donna_and_Marcia_total_pencils_l133_133148


namespace evaluate_expression_l133_133060

variable (a b c d e : ℝ)

-- The equivalent proof problem statement
theorem evaluate_expression 
  (h : (a / b * c - d + e = a / (b * c - d - e))) : 
  a / b * c - d + e = a / (b * c - d - e) :=
by 
  exact h

-- Placeholder for the proof
#check evaluate_expression

end evaluate_expression_l133_133060


namespace unique_function_solution_l133_133012

theorem unique_function_solution :
  ∀ f : ℕ+ → ℕ+, (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end unique_function_solution_l133_133012


namespace length_of_bridge_l133_133377

theorem length_of_bridge 
    (length_of_train : ℕ)
    (speed_of_train_km_per_hr : ℕ)
    (time_to_cross_seconds : ℕ)
    (bridge_length : ℕ) 
    (h_train_length : length_of_train = 130)
    (h_speed_train : speed_of_train_km_per_hr = 54)
    (h_time_cross : time_to_cross_seconds = 30)
    (h_bridge_length : bridge_length = 320) : 
    bridge_length = 320 :=
by sorry

end length_of_bridge_l133_133377


namespace length_of_AB_l133_133704

noncomputable def hyperbola_conditions (a b : ℝ) (hac : a > 0) (hbc : b = 2 * a) :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def circle_intersection_condition (A B : ℝ × ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
  (A = (x1, y1)) ∧ (B = (x2, y2)) ∧ ((x1 - 2)^2 + (y1 - 3)^2 = 1 ∧ y1 = 2 * x1) ∧
  ((x2 - 2)^2 + (y2 - 3)^2 = 1 ∧ y2 = 2 * x2)

theorem length_of_AB {a b : ℝ} (hac : a > 0) (hb : b = 2 * a) :
  (hyperbola_conditions a b hac hb) →
  ∃ (A B : ℝ × ℝ), circle_intersection_condition A B → 
  dist A B = (4 * Real.sqrt 5) / 5 :=
by
  sorry

end length_of_AB_l133_133704


namespace current_algae_plants_l133_133882

def original_algae_plants : ℕ := 809
def additional_algae_plants : ℕ := 2454

theorem current_algae_plants :
  original_algae_plants + additional_algae_plants = 3263 := by
  sorry

end current_algae_plants_l133_133882


namespace middle_income_sample_count_l133_133594

def total_households : ℕ := 600
def high_income_families : ℕ := 150
def middle_income_families : ℕ := 360
def low_income_families : ℕ := 90
def sample_size : ℕ := 80

theorem middle_income_sample_count : 
  (middle_income_families / total_households) * sample_size = 48 := 
by
  sorry

end middle_income_sample_count_l133_133594


namespace total_books_l133_133879

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end total_books_l133_133879


namespace smallest_a_l133_133068

theorem smallest_a (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) : a = 17 :=
sorry

end smallest_a_l133_133068


namespace third_part_of_division_l133_133276

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division_l133_133276


namespace cauliflower_sales_l133_133081

theorem cauliflower_sales :
  let total_earnings := 500
  let b_sales := 57
  let c_sales := 2 * b_sales
  let s_sales := (c_sales / 2) + 16
  let t_sales := b_sales + s_sales
  let ca_sales := total_earnings - (b_sales + c_sales + s_sales + t_sales)
  ca_sales = 126 := by
  sorry

end cauliflower_sales_l133_133081


namespace nathan_tokens_used_is_18_l133_133353

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l133_133353


namespace min_value_of_reciprocals_l133_133315

open Real

theorem min_value_of_reciprocals (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) :
  (1 / a) + (1 / (b + 1)) ≥ 2 :=
sorry

end min_value_of_reciprocals_l133_133315


namespace find_q_l133_133244

-- Given polynomial Q(x) with coefficients p, q, d
variables {p q d : ℝ}

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 + p * x^2 + q * x + d

-- Assume the conditions of the problem
theorem find_q (h1 : d = 5)                   -- y-intercept is 5
    (h2 : (-p / 3) = -d)                    -- mean of zeros = product of zeros
    (h3 : (-p / 3) = 1 + p + q + d)          -- mean of zeros = sum of coefficients
    : q = -26 := 
    sorry

end find_q_l133_133244


namespace new_job_larger_than_original_l133_133804

theorem new_job_larger_than_original (original_workers original_days new_workers new_days : ℕ) 
  (h_original_workers : original_workers = 250)
  (h_original_days : original_days = 16)
  (h_new_workers : new_workers = 600)
  (h_new_days : new_days = 20) :
  (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry

end new_job_larger_than_original_l133_133804


namespace probability_of_not_shorter_than_one_meter_l133_133621

noncomputable def probability_of_event_A : ℝ := 
  let length_of_rope : ℝ := 3
  let event_A_probability : ℝ := 1 / 3
  event_A_probability

theorem probability_of_not_shorter_than_one_meter (l : ℝ) (h_l : l = 3) : 
    probability_of_event_A = 1 / 3 :=
sorry

end probability_of_not_shorter_than_one_meter_l133_133621


namespace minimum_bail_rate_l133_133372

theorem minimum_bail_rate 
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (bail_rate : ℝ)
  (time_to_shore : ℝ) :
  distance = 2 ∧
  leak_rate = 15 ∧
  max_water = 60 ∧
  rowing_speed = 3 ∧
  time_to_shore = distance / rowing_speed * 60 →
  bail_rate = (leak_rate * time_to_shore - max_water) / time_to_shore →
  bail_rate = 13.5 :=
by
  intros
  sorry

end minimum_bail_rate_l133_133372


namespace number_of_blocks_needed_to_form_cube_l133_133785

-- Define the dimensions of the rectangular block
def block_length : ℕ := 5
def block_width : ℕ := 4
def block_height : ℕ := 3

-- Define the side length of the cube
def cube_side_length : ℕ := 60

-- The expected number of rectangular blocks needed
def expected_number_of_blocks : ℕ := 3600

-- Statement to prove the number of rectangular blocks needed to form the cube
theorem number_of_blocks_needed_to_form_cube
  (l : ℕ) (w : ℕ) (h : ℕ) (cube_side : ℕ) (expected_count : ℕ)
  (h_l : l = block_length)
  (h_w : w = block_width)
  (h_h : h = block_height)
  (h_cube_side : cube_side = cube_side_length)
  (h_expected : expected_count = expected_number_of_blocks) :
  (cube_side ^ 3) / (l * w * h) = expected_count :=
sorry

end number_of_blocks_needed_to_form_cube_l133_133785


namespace largest_digit_divisible_by_6_l133_133520

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133520


namespace largest_digit_divisible_by_6_l133_133522

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l133_133522


namespace power_first_digits_l133_133224

theorem power_first_digits (n : ℕ) (h1 : ∀ k : ℕ, n ≠ 10^k) : ∃ j k : ℕ, 1973 ≤ n^j / 10^k ∧ n^j / 10^k < 1974 := by
  sorry

end power_first_digits_l133_133224


namespace flower_options_l133_133214

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end flower_options_l133_133214


namespace savings_percentage_correct_l133_133002

-- Definitions based on conditions
def food_per_week : ℕ := 100
def num_weeks : ℕ := 4
def rent : ℕ := 1500
def video_streaming : ℕ := 30
def cell_phone : ℕ := 50
def savings : ℕ := 198

-- Total spending calculations based on the conditions
def food_total : ℕ := food_per_week * num_weeks
def total_spending : ℕ := food_total + rent + video_streaming + cell_phone

-- Calculation of the percentage
def savings_percentage (savings total_spending : ℕ) : ℕ :=
  (savings * 100) / total_spending

-- The statement to prove
theorem savings_percentage_correct : savings_percentage savings total_spending = 10 := by
  sorry

end savings_percentage_correct_l133_133002


namespace students_taking_chem_or_phys_not_both_l133_133634

def students_taking_both : ℕ := 12
def students_taking_chemistry : ℕ := 30
def students_taking_only_physics : ℕ := 18

theorem students_taking_chem_or_phys_not_both : 
  (students_taking_chemistry - students_taking_both) + students_taking_only_physics = 36 := 
by
  sorry

end students_taking_chem_or_phys_not_both_l133_133634


namespace time_spent_cutting_hair_l133_133447

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l133_133447


namespace frank_cookies_l133_133021

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l133_133021


namespace fruit_vendor_total_l133_133886

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l133_133886


namespace cos_270_eq_zero_l133_133979

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l133_133979


namespace floor_equation_solution_l133_133862

theorem floor_equation_solution (a b : ℝ) :
  (∀ x y : ℝ, ⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋) → (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) := by
  sorry

end floor_equation_solution_l133_133862


namespace angle_KMT_l133_133784

-- Let A, B, C, T, K, and M be points such that:
variables {A B C T K M : Type}
-- Triangles ABT and ACK are constructed externally with respect to sides AB and AC of triangle ABC
-- such that ∠ATB = 90°, ∠AKC = 90°, ∠ABT = 30°, ∠ACK = 30°.
variable [EuclideanGeometry]

-- Define the triangle ABC
axiom ABC : Triangle A B C

-- ∠ATB = ∠AKC = 90°
axiom angle_ATB : ∠ A T B = 90
axiom angle_AKC : ∠ A K C = 90

-- ∠ABT = ∠ACK = 30°
axiom angle_ABT : ∠ A B T = 30
axiom angle_ACK : ∠ A C K = 30

-- BM = MC
axiom midpoint_M : midpoint M B C

-- We need to prove that the measure of ∠KMT = 60°
theorem angle_KMT : ∠ K M T = 60 := 
sorry

end angle_KMT_l133_133784


namespace correct_algorithm_description_l133_133641

def conditions_about_algorithms (desc : String) : Prop :=
  (desc = "A" → false) ∧
  (desc = "B" → false) ∧
  (desc = "C" → true) ∧
  (desc = "D" → false)

theorem correct_algorithm_description : ∃ desc : String, 
  conditions_about_algorithms desc :=
by
  use "C"
  unfold conditions_about_algorithms
  simp
  sorry

end correct_algorithm_description_l133_133641


namespace no_positive_integral_solution_l133_133844

theorem no_positive_integral_solution :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ p : ℕ, Prime p ∧ n^2 - 45 * n + 520 = p :=
by {
  -- Since we only need the statement, we'll introduce the necessary steps without the full proof
  sorry
}

end no_positive_integral_solution_l133_133844


namespace correct_exponentiation_l133_133640

variable (a : ℝ)

theorem correct_exponentiation : (a^2)^3 = a^6 := by
  sorry

end correct_exponentiation_l133_133640


namespace first_train_speed_l133_133805

noncomputable def speed_of_first_train (length_train1 : ℕ) (speed_train2 : ℕ) (length_train2 : ℕ) (time_cross : ℕ) : ℕ :=
  let relative_speed_m_s := (500 : ℕ) / time_cross
  let relative_speed_km_h := relative_speed_m_s * 18 / 5
  relative_speed_km_h - speed_train2

theorem first_train_speed :
  speed_of_first_train 270 80 230 9 = 920 := by
  sorry

end first_train_speed_l133_133805


namespace max_num_piles_l133_133770

/-- Maximum number of piles can be formed from 660 stones -/
theorem max_num_piles (total_stones : ℕ) (h : total_stones = 660) :
  ∃ (max_piles : ℕ), max_piles = 30 ∧ 
  ∀ (piles : list ℕ), (piles.sum = total_stones) → 
                      (∀ (x y : ℕ), x ∈ piles → y ∈ piles → 
                                  (x ≤ 2 * y ∧ y ≤ 2 * x)) → 
                      (piles.length ≤ max_piles) :=
by
  sorry

end max_num_piles_l133_133770


namespace fraction_human_habitable_surface_l133_133718

variable (fraction_water_coverage : ℚ)
variable (fraction_inhabitable_remaining_land : ℚ)
variable (fraction_reserved_for_agriculture : ℚ)

def fraction_inhabitable_land (f_water : ℚ) (f_inhabitable : ℚ) : ℚ :=
  (1 - f_water) * f_inhabitable

def fraction_habitable_land (f_inhabitable_land : ℚ) (f_reserved : ℚ) : ℚ :=
  f_inhabitable_land * (1 - f_reserved)

theorem fraction_human_habitable_surface 
  (h1 : fraction_water_coverage = 3/5)
  (h2 : fraction_inhabitable_remaining_land = 2/3)
  (h3 : fraction_reserved_for_agriculture = 1/2) :
  fraction_habitable_land 
    (fraction_inhabitable_land fraction_water_coverage fraction_inhabitable_remaining_land)
    fraction_reserved_for_agriculture = 2/15 :=
by {
  sorry
}

end fraction_human_habitable_surface_l133_133718


namespace cos_alpha_minus_pi_over_6_l133_133318

theorem cos_alpha_minus_pi_over_6 (α : Real) 
  (h1 : Real.pi / 2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (α - Real.pi / 6) = (3 * Real.sqrt 3 - 4) / 10 := 
by 
  sorry

end cos_alpha_minus_pi_over_6_l133_133318


namespace garden_area_increase_l133_133814

theorem garden_area_increase :
  let length := 80
  let width := 20
  let additional_fence := 60
  let original_area := length * width
  let original_perimeter := 2 * (length + width)
  let total_fence := original_perimeter + additional_fence
  let side_of_square := total_fence / 4
  let square_area := side_of_square * side_of_square
  square_area - original_area = 2625 :=
by
  sorry

end garden_area_increase_l133_133814


namespace part_1_prob_excellent_part_2_rounds_pvalues_l133_133057

-- Definition of the probability of an excellent pair
def prob_excellent (p1 p2 : ℚ) : ℚ :=
  2 * p1 * (1 - p1) * p2 * p2 + p1 * p1 * 2 * p2 * (1 - p2) + p1 * p1 * p2 * p2

-- Part (1) statement: Prove the probability that they achieve "excellent pair" status in the first round
theorem part_1_prob_excellent (p1 p2 : ℚ) (hp1 : p1 = 3/4) (hp2 : p2 = 2/3) :
  prob_excellent p1 p2 = 2/3 := by
  rw [hp1, hp2]
  sorry

-- Part (2) statement: Prove the minimum number of rounds and values of p1 and p2
theorem part_2_rounds_pvalues (n : ℕ) (p1 p2 : ℚ) (h_sum : p1 + p2 = 4/3)
  (h_goal : n * prob_excellent p1 p2 ≥ 16) :
  (n = 27) ∧ (p1 = 2/3) ∧ (p2 = 2/3) := by
  sorry

end part_1_prob_excellent_part_2_rounds_pvalues_l133_133057


namespace least_positive_int_to_multiple_of_3_l133_133116

theorem least_positive_int_to_multiple_of_3 (x : ℕ) (h : 575 + x ≡ 0 [MOD 3]) : x = 1 := 
by
  sorry

end least_positive_int_to_multiple_of_3_l133_133116


namespace star_value_example_l133_133411

def my_star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

theorem star_value_example : my_star 3 5 = 68 := 
by
  sorry

end star_value_example_l133_133411


namespace quadratic_has_distinct_real_roots_l133_133591

theorem quadratic_has_distinct_real_roots (m : ℝ) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = m - 1 ∧ (b^2 - 4 * a * c > 0) → (m < 2) :=
by
  sorry

end quadratic_has_distinct_real_roots_l133_133591


namespace people_per_table_l133_133968

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l133_133968


namespace swimming_speed_l133_133543

theorem swimming_speed (s v : ℝ) (h_s : s = 4) (h_time : 1 / (v - s) = 2 * (1 / (v + s))) : v = 12 := 
by
  sorry

end swimming_speed_l133_133543


namespace find_number_of_sides_l133_133497

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l133_133497


namespace donuts_per_student_l133_133466

theorem donuts_per_student (total_donuts : ℕ) (students : ℕ) (percentage_likes_donuts : ℚ) 
    (H1 : total_donuts = 4 * 12) 
    (H2 : students = 30) 
    (H3 : percentage_likes_donuts = 0.8) 
    (H4 : ∃ (likes_donuts : ℕ), likes_donuts = students * percentage_likes_donuts) : 
    (∃ (donuts_per_student : ℚ), donuts_per_student = total_donuts / (students * percentage_likes_donuts)) → donuts_per_student = 2 :=
by
    sorry

end donuts_per_student_l133_133466


namespace sum_of_three_numbers_l133_133901

theorem sum_of_three_numbers : ∃ (a b c : ℝ), a ≤ b ∧ b ≤ c ∧ b = 8 ∧ 
  (a + b + c) / 3 = a + 8 ∧ (a + b + c) / 3 = c - 20 ∧ a + b + c = 60 :=
sorry

end sum_of_three_numbers_l133_133901


namespace hyperbola_eccentricity_l133_133859

variables (a b e : ℝ) (F1 F2 P : ℝ × ℝ)

-- The hyperbola assumption
def hyperbola : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / a^2 - y^2 / b^2 = 1
-- a > 0 and b > 0
def positive_a_b : Prop := a > 0 ∧ b > 0
-- Distance between foci
def distance_foci : Prop := dist F1 F2 = 12
-- Distance PF2
def distance_p_f2 : Prop := dist P F2 = 5
-- To be proven, eccentricity of the hyperbola
def eccentricity : Prop := e = 3 / 2

theorem hyperbola_eccentricity : hyperbola a b P ∧ positive_a_b a b ∧ distance_foci F1 F2 ∧ distance_p_f2 P F2 → eccentricity e :=
by
  sorry

end hyperbola_eccentricity_l133_133859


namespace balls_in_boxes_l133_133469

theorem balls_in_boxes : 
  let boxes := {box1, box2, box3}
  let balls := {1, 2, 3, 4, 5, 6}
  ∃ (arrangement : Set (Set ℕ)), arrangement.card = 3 ∧ 
  (∀ b ∈ arrangement, b.card = 2) ∧ 
  ({1, 2} ∈ arrangement) ∧ 
  (∀ b ∈ arrangement, b ⊆ balls) ∧ 
  ∀ b₁ b₂ ∈ arrangement, b₁ ≠ b₂ → b₁ ∩ b₂ = ∅ 
  → arrangement.count = 18 :=
sorry

end balls_in_boxes_l133_133469


namespace width_of_room_l133_133489

theorem width_of_room 
  (length : ℝ) 
  (cost : ℝ) 
  (rate : ℝ) 
  (h_length : length = 6.5) 
  (h_cost : cost = 10725) 
  (h_rate : rate = 600) 
  : (cost / rate) / length = 2.75 :=
by
  rw [h_length, h_cost, h_rate]
  norm_num

end width_of_room_l133_133489


namespace calculate_max_marks_l133_133080

theorem calculate_max_marks (shortfall_math : ℕ) (shortfall_science : ℕ) 
                            (shortfall_literature : ℕ) (shortfall_social_studies : ℕ)
                            (required_math : ℕ) (required_science : ℕ)
                            (required_literature : ℕ) (required_social_studies : ℕ)
                            (max_math : ℕ) (max_science : ℕ)
                            (max_literature : ℕ) (max_social_studies : ℕ) :
                            shortfall_math = 40 ∧ required_math = 95 ∧ max_math = 800 ∧
                            shortfall_science = 35 ∧ required_science = 92 ∧ max_science = 438 ∧
                            shortfall_literature = 30 ∧ required_literature = 90 ∧ max_literature = 300 ∧
                            shortfall_social_studies = 25 ∧ required_social_studies = 88 ∧ max_social_studies = 209 :=
by
  sorry

end calculate_max_marks_l133_133080


namespace icosahedron_minimal_rotation_l133_133670

structure Icosahedron :=
  (faces : ℕ)
  (is_regular : Prop)
  (face_shape : Prop)

def icosahedron := Icosahedron.mk 20 (by sorry) (by sorry)

def theta (θ : ℝ) : Prop :=
  ∃ θ > 0, ∀ h : Icosahedron, 
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72

theorem icosahedron_minimal_rotation :
  ∃ θ > 0, ∀ h : Icosahedron,
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72 :=
by sorry

end icosahedron_minimal_rotation_l133_133670


namespace number_of_foons_correct_l133_133811

-- Define the conditions
def area : ℝ := 5  -- Area in cm^2
def thickness : ℝ := 0.5  -- Thickness in cm
def total_volume : ℝ := 50  -- Total volume in cm^3

-- Define the proof problem
theorem number_of_foons_correct :
  (total_volume / (area * thickness) = 20) :=
by
  -- The necessary computation would go here, but for now we'll use sorry to indicate the outcome
  sorry

end number_of_foons_correct_l133_133811


namespace exercise_l133_133239

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise_l133_133239


namespace little_john_spent_on_sweets_l133_133349

theorem little_john_spent_on_sweets
  (initial_amount : ℝ)
  (amount_per_friend : ℝ)
  (friends_count : ℕ)
  (amount_left : ℝ)
  (spent_on_sweets : ℝ) :
  initial_amount = 10.50 →
  amount_per_friend = 2.20 →
  friends_count = 2 →
  amount_left = 3.85 →
  spent_on_sweets = initial_amount - (amount_per_friend * friends_count) - amount_left →
  spent_on_sweets = 2.25 :=
by
  intros h_initial h_per_friend h_friends_count h_left h_spent
  sorry

end little_john_spent_on_sweets_l133_133349


namespace find_age_of_D_l133_133765

theorem find_age_of_D
(Eq1 : a + b + c + d = 108)
(Eq2 : a - b = 12)
(Eq3 : c - (a - 34) = 3 * (d - (a - 34)))
: d = 13 := 
sorry

end find_age_of_D_l133_133765


namespace find_x_l133_133714

variable (x y : ℚ)

-- Condition
def condition : Prop :=
  (x / (x - 2)) = ((y^3 + 3 * y - 2) / (y^3 + 3 * y - 5))

-- Assertion to prove
theorem find_x (h : condition x y) : x = ((2 * y^3 + 6 * y - 4) / 3) :=
sorry

end find_x_l133_133714


namespace triangle_interior_angle_l133_133202

open EuclideanGeometry

noncomputable def problem_statement (A B C D M : Point) : Prop :=
  ∃ (A B C D M : Point),
  (triangle A B C) ∧
  (angle B A C = 2 * angle C A B) ∧
  (angle B A C > 90°) ∧
  (line_contains A B D) ∧
  (perpendicular (line C D) (line A C)) ∧
  (midpoint M B C) ∧
  (angle A M B = angle D M C)

-- The theorem stating the problem
theorem triangle_interior_angle :
  ∀ (A B C D M : Point), problem_statement A B C D M :=
begin
  -- Given conditions (stated in problem_statement)
  sorry
end

end triangle_interior_angle_l133_133202


namespace work_efficiency_ratio_l133_133281

theorem work_efficiency_ratio
  (A B : ℝ)
  (h1 : A + B = 1 / 18)
  (h2 : B = 1 / 27) :
  A / B = 1 / 2 := 
by
  sorry

end work_efficiency_ratio_l133_133281


namespace intersection_of_M_and_N_l133_133077

def set_M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def set_N : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def intersection_M_N : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := 
by sorry

end intersection_of_M_and_N_l133_133077


namespace largest_digit_divisible_by_6_l133_133521

theorem largest_digit_divisible_by_6 :
  ∃ (N : ℕ), N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0 ∧ (∀ m ∈ {N | N ∈ {0, 2, 4, 6, 8} ∧ (26 + N) % 3 = 0}, m ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133521


namespace total_baseball_fans_l133_133056

-- Conditions given
def ratio_YM (Y M : ℕ) : Prop := 2 * Y = 3 * M
def ratio_MR (M R : ℕ) : Prop := 4 * R = 5 * M
def M_value : ℕ := 88

-- Prove total number of baseball fans
theorem total_baseball_fans (Y M R : ℕ) (h1 : ratio_YM Y M) (h2 : ratio_MR M R) (hM : M = M_value) :
  Y + M + R = 330 :=
sorry

end total_baseball_fans_l133_133056


namespace evaluate_expression_l133_133988

-- Define the expression as given in the problem
def expr1 : ℤ := |9 - 8 * (3 - 12)|
def expr2 : ℤ := |5 - 11|

-- Define the mathematical equivalence
theorem evaluate_expression : (expr1 - expr2) = 75 := by
  sorry

end evaluate_expression_l133_133988


namespace three_lines_form_triangle_l133_133780

/-- Theorem to prove that for three lines x + y = 0, x - y = 0, and x + ay = 3 to form a triangle, the value of a cannot be ±1. -/
theorem three_lines_form_triangle (a : ℝ) : ¬ (a = 1 ∨ a = -1) :=
sorry

end three_lines_form_triangle_l133_133780


namespace trains_time_to_clear_each_other_l133_133940

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

noncomputable def speed_to_m_s (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def total_length (l1 l2 : ℝ) : ℝ :=
  l1 + l2

theorem trains_time_to_clear_each_other :
  ∀ (l1 l2 : ℝ) (v1_kmph v2_kmph : ℝ),
    l1 = 100 → l2 = 280 →
    v1_kmph = 42 → v2_kmph = 30 →
    (total_length l1 l2) / (speed_to_m_s (relative_speed v1_kmph v2_kmph)) = 19 :=
by
  intros l1 l2 v1_kmph v2_kmph h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end trains_time_to_clear_each_other_l133_133940


namespace parallel_lines_k_l133_133856

theorem parallel_lines_k (k : ℝ) :
  (∃ (x y : ℝ), (k-3) * x + (4-k) * y + 1 = 0 ∧ 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 3 ∨ k = 5) :=
by
  sorry

end parallel_lines_k_l133_133856


namespace isolating_and_counting_bacteria_process_l133_133906

theorem isolating_and_counting_bacteria_process
  (soil_sampling : Prop)
  (spreading_dilution_on_culture_medium : Prop)
  (decompose_urea : Prop) :
  (soil_sampling ∧ spreading_dilution_on_culture_medium ∧ decompose_urea) →
  (Sample_dilution ∧ Selecting_colonies_that_can_grow ∧ Identification) :=
sorry

end isolating_and_counting_bacteria_process_l133_133906


namespace probability_of_one_failure_l133_133918

theorem probability_of_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  (p1 * (1 - p2) + (1 - p1) * p2) = 0.14 :=
by
  rw [h1, h2]
  -- Additional leaning code can be inserted here to finalize the proof if this was complete
  sorry

end probability_of_one_failure_l133_133918


namespace equilateral_cannot_be_obtuse_l133_133120

-- Additional definitions for clarity and mathematical rigor.
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a
def is_obtuse (A B C : ℝ) : Prop := 
    (A > 90 ∧ B < 90 ∧ C < 90) ∨ 
    (B > 90 ∧ A < 90 ∧ C < 90) ∨
    (C > 90 ∧ A < 90 ∧ B < 90)

-- Theorem statement
theorem equilateral_cannot_be_obtuse (a b c : ℝ) (A B C : ℝ) :
  is_equilateral a b c → 
  (A + B + C = 180) → 
  (A = B ∧ B = C) → 
  ¬ is_obtuse A B C :=
by { sorry } -- Proof is not necessary as per instruction.

end equilateral_cannot_be_obtuse_l133_133120


namespace complex_number_on_ray_is_specific_l133_133851

open Complex

theorem complex_number_on_ray_is_specific (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) 
  (h₂ : a = b) (h₃ : abs z = 1) : 
  z = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * I :=
by
  sorry

end complex_number_on_ray_is_specific_l133_133851


namespace lcm_value_count_l133_133568

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l133_133568


namespace segment_ratios_correct_l133_133757

noncomputable def compute_segment_ratios : (ℕ × ℕ) :=
  let ratio := 20 / 340;
  let gcd := Nat.gcd 1 17;
  if (ratio = 1 / 17) ∧ (gcd = 1) then (1, 17) else (0, 0) 

theorem segment_ratios_correct : 
  compute_segment_ratios = (1, 17) := 
by
  sorry

end segment_ratios_correct_l133_133757


namespace cos_270_eq_zero_l133_133974

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l133_133974


namespace no_integer_solutions_other_than_zero_l133_133618

theorem no_integer_solutions_other_than_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  sorry

end no_integer_solutions_other_than_zero_l133_133618


namespace least_y_solution_l133_133525

theorem least_y_solution :
  (∃ y : ℝ, 3 * y^2 + 5 * y + 2 = 4 ∧ ∀ z : ℝ, 3 * z^2 + 5 * z + 2 = 4 → y ≤ z) →
  ∃ y : ℝ, y = -2 :=
by
  sorry

end least_y_solution_l133_133525


namespace cos_270_eq_zero_l133_133978

theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 := by
  sorry

end cos_270_eq_zero_l133_133978


namespace find_a_b_sum_l133_133456

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_b_sum (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 24) : a + b = 6 :=
  sorry

end find_a_b_sum_l133_133456


namespace fraction_calculation_l133_133293

theorem fraction_calculation : (4 / 9 + 1 / 9) / (5 / 8 - 1 / 8) = 10 / 9 := by
  sorry

end fraction_calculation_l133_133293


namespace find_digits_l133_133996

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l133_133996


namespace rent_increase_l133_133345

theorem rent_increase (monthly_rent_first_3_years : ℕ) (months_first_3_years : ℕ) 
  (total_paid : ℕ) (total_years : ℕ) (months_in_a_year : ℕ) (new_monthly_rent : ℕ) :
  monthly_rent_first_3_years * (months_in_a_year * 3) + new_monthly_rent * (months_in_a_year * (total_years - 3)) = total_paid →
  new_monthly_rent = 350 :=
by
  intros h
  -- proof development
  sorry

end rent_increase_l133_133345


namespace fruit_vendor_total_l133_133887

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end fruit_vendor_total_l133_133887


namespace train_cross_pole_time_l133_133138

noncomputable def time_to_cross_pole : ℝ :=
  let speed_km_hr := 60
  let speed_m_s := speed_km_hr * 1000 / 3600
  let length_of_train := 50
  length_of_train / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole = 3 := 
by
  sorry

end train_cross_pole_time_l133_133138


namespace last_three_digits_of_8_pow_104_l133_133684

def last_three_digits_of_pow (x n : ℕ) : ℕ :=
  (x ^ n) % 1000

theorem last_three_digits_of_8_pow_104 : last_three_digits_of_pow 8 104 = 984 := 
by
  sorry

end last_three_digits_of_8_pow_104_l133_133684


namespace n_is_power_of_p_l133_133321

-- Given conditions as definitions
variables {x y p n k l : ℕ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < p) (h4 : 0 < n) (h5 : 0 < k)
variables (h6 : x^n + y^n = p^k) (h7 : odd n) (h8 : n > 1) (h9 : prime p) (h10 : odd p)

-- The theorem to be proved
theorem n_is_power_of_p : ∃ l : ℕ, n = p^l :=
  sorry

end n_is_power_of_p_l133_133321


namespace max_num_piles_l133_133771

/-- Maximum number of piles can be formed from 660 stones -/
theorem max_num_piles (total_stones : ℕ) (h : total_stones = 660) :
  ∃ (max_piles : ℕ), max_piles = 30 ∧ 
  ∀ (piles : list ℕ), (piles.sum = total_stones) → 
                      (∀ (x y : ℕ), x ∈ piles → y ∈ piles → 
                                  (x ≤ 2 * y ∧ y ≤ 2 * x)) → 
                      (piles.length ≤ max_piles) :=
by
  sorry

end max_num_piles_l133_133771


namespace square_side_length_l133_133136

theorem square_side_length (s : ℝ) (h : s^2 = 1/9) : s = 1/3 :=
sorry

end square_side_length_l133_133136


namespace minimize_MA_dot_MB_cosine_AMB_l133_133853

open Real

def coord_O := (0, 0)
def vector_OA := (1, 7)
def vector_OB := (5, 1)
def vector_OP := (2, 1)
def OM (M : ℝ × ℝ) : Prop := ∃ k : ℝ, M = (2 * k, k)

-- Condition (I): \overrightarrow{OM} = (4, 2)
def condition_OM := (4, 2)

theorem minimize_MA_dot_MB (M : ℝ × ℝ) (h : OM M ∧ M = condition_OM) : 
  let MA := (1 - (M.1), 7 - (M.2))
  let MB := (5 - (M.1), 1 - (M.2))
  (MA.1 * MB.1 + MA.2 * MB.2) = -8 := 
by 
  sorry

theorem cosine_AMB (M : ℝ × ℝ) (h : OM M ∧ M = condition_OM) :
  let MA := (1 - (M.1), 7 - (M.2))
  let MB := (5 - (M.1), 1 - (M.2))
  cosangle MA MB = - (4 * sqrt 17) / 17 :=
by 
  sorry

end minimize_MA_dot_MB_cosine_AMB_l133_133853


namespace integer_roots_7_values_of_a_l133_133836

theorem integer_roots_7_values_of_a :
  (∃ a : ℝ, (∀ r s : ℤ, (r + s = -a ∧ (r * s = 8 * a))) ∧ (∃ n : ℕ, n = 7)) :=
sorry

end integer_roots_7_values_of_a_l133_133836


namespace comparison_of_square_roots_l133_133201

theorem comparison_of_square_roots (P Q : ℝ) (hP : P = Real.sqrt 2) (hQ : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q :=
by
  sorry

end comparison_of_square_roots_l133_133201


namespace car_price_l133_133620

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end car_price_l133_133620


namespace cookies_left_correct_l133_133211

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end cookies_left_correct_l133_133211


namespace maximize_sector_area_l133_133424

noncomputable def sector_radius_angle (r l α : ℝ) : Prop :=
  2 * r + l = 40 ∧ α = l / r

theorem maximize_sector_area :
  ∃ r α : ℝ, sector_radius_angle r 20 α ∧ r = 10 ∧ α = 2 :=
by
  sorry

end maximize_sector_area_l133_133424


namespace center_of_conic_l133_133541

-- Define the conic equation
def conic_equation (p q r α β γ : ℝ) : Prop :=
  p * α * β + q * α * γ + r * β * γ = 0

-- Define the barycentric coordinates of the center
def center_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (r * (p + q - r), q * (p + r - q), p * (r + q - p))

-- Theorem to prove that the barycentric coordinates of the center are as expected
theorem center_of_conic (p q r α β γ : ℝ) (h : conic_equation p q r α β γ) :
  center_coordinates p q r = (r * (p + q - r), q * (p + r - q), p * (r + q - p)) := 
sorry

end center_of_conic_l133_133541


namespace smallest_positive_n_common_factor_l133_133264

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∃ d : ℕ, d > 1 ∧ d ∣ (8 * n - 3) ∧ d ∣ (6 * n + 4)) ∧ n = 1 :=
by
  sorry

end smallest_positive_n_common_factor_l133_133264


namespace solve_for_a_l133_133050

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l133_133050


namespace consecutive_numbers_difference_l133_133910

theorem consecutive_numbers_difference :
  ∃ (n : ℕ), (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → (n + 5 - n = 5) :=
by {
  sorry
}

end consecutive_numbers_difference_l133_133910


namespace min_trams_spy_sees_l133_133374

/-- 
   Vasya stood at a bus stop for some time and saw 1 bus and 2 trams.
   Buses run every hour.
   After Vasya left, a spy stood at the bus stop for 10 hours and saw 10 buses.
   Given these conditions, the minimum number of trams that the spy could have seen is 5.
-/
theorem min_trams_spy_sees (bus_interval tram_interval : ℕ) 
  (vasya_buses vasya_trams spy_buses spy_hours min_trams : ℕ) 
  (h1 : bus_interval = 1)
  (h2 : vasya_buses = 1)
  (h3 : vasya_trams = 2)
  (h4 : spy_buses = spy_hours)
  (h5 : spy_buses = 10)
  (h6 : spy_hours = 10)
  (h7 : ∀ t : ℕ, t * tram_interval ≤ 2 → 2 * bus_interval ≤ 2)
  (h8 : min_trams = 5) :
  min_trams = 5 := 
sorry

end min_trams_spy_sees_l133_133374


namespace speed_of_man_rowing_upstream_l133_133282

-- Define conditions
def V_m : ℝ := 20 -- speed of the man in still water (kmph)
def V_downstream : ℝ := 25 -- speed of the man rowing downstream (kmph)
def V_s : ℝ := V_downstream - V_m -- calculate the speed of the stream

-- Define the theorem to prove the speed of the man rowing upstream
theorem speed_of_man_rowing_upstream 
  (V_m : ℝ) (V_downstream : ℝ) (V_s : ℝ := V_downstream - V_m) : 
  V_upstream = V_m - V_s :=
by
  sorry

end speed_of_man_rowing_upstream_l133_133282


namespace KeatonAnnualEarnings_l133_133734

-- Keaton's conditions for oranges
def orangeHarvestInterval : ℕ := 2
def orangeSalePrice : ℕ := 50

-- Keaton's conditions for apples
def appleHarvestInterval : ℕ := 3
def appleSalePrice : ℕ := 30

-- Annual earnings calculation
def annualEarnings (monthsInYear : ℕ) : ℕ :=
  let orangeEarnings := (monthsInYear / orangeHarvestInterval) * orangeSalePrice
  let appleEarnings := (monthsInYear / appleHarvestInterval) * appleSalePrice
  orangeEarnings + appleEarnings

-- Prove the total annual earnings is 420
theorem KeatonAnnualEarnings : annualEarnings 12 = 420 :=
  by 
    -- We skip the proof details here.
    sorry

end KeatonAnnualEarnings_l133_133734


namespace circle_through_two_points_on_y_axis_l133_133683

theorem circle_through_two_points_on_y_axis :
  ∃ (b : ℝ), (∀ (x y : ℝ), (x + 1)^2 + (y - 4)^2 = (x - 3)^2 + (y - 2)^2 → b = 1) ∧ 
  (∀ (x y : ℝ), (x - 0)^2 + (y - b)^2 = 10) := 
sorry

end circle_through_two_points_on_y_axis_l133_133683


namespace partA_partB_partC_partD_l133_133258

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l133_133258


namespace partA_partB_partC_partD_l133_133257

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end partA_partB_partC_partD_l133_133257


namespace Jazmin_strips_width_l133_133344

theorem Jazmin_strips_width (w1 w2 g : ℕ) (h1 : w1 = 44) (h2 : w2 = 33) (hg : g = Nat.gcd w1 w2) : g = 11 := by
  -- Markdown above outlines:
  -- w1, w2 are widths of the construction paper
  -- h1: w1 = 44
  -- h2: w2 = 33
  -- hg: g = gcd(w1, w2)
  -- Prove g == 11
  sorry

end Jazmin_strips_width_l133_133344


namespace max_piles_660_l133_133774

noncomputable def max_piles (initial_piles : ℕ) : ℕ :=
  if initial_piles = 660 then 30 else 0

theorem max_piles_660 (initial_piles : ℕ)
  (h : initial_piles = 660) :
  ∃ n, max_piles initial_piles = n ∧ n = 30 :=
begin
  use 30,
  split,
  { rw [max_piles, if_pos h], },
  { refl, },
end

end max_piles_660_l133_133774


namespace perp_line_through_point_l133_133310

variable (x y c : ℝ)

def line_perpendicular (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

def perpendicular_line (x y c : ℝ) : Prop :=
  2*x + y + c = 0

theorem perp_line_through_point :
  (line_perpendicular x y) ∧ (perpendicular_line (-2) 3 1) :=
by
  -- The first part asserts that the given line equation holds
  have h1 : line_perpendicular x y := sorry
  -- The second part asserts that our calculated line passes through the point (-2, 3) and is perpendicular
  have h2 : perpendicular_line (-2) 3 1 := sorry
  exact ⟨h1, h2⟩

end perp_line_through_point_l133_133310


namespace five_million_squared_l133_133674

theorem five_million_squared : (5 * 10^6)^2 = 25 * 10^12 := by
  sorry

end five_million_squared_l133_133674


namespace pandas_bamboo_consumption_l133_133963

/-- Given:
  1. An adult panda can eat 138 pounds of bamboo each day.
  2. A baby panda can eat 50 pounds of bamboo a day.
Prove: the total pounds of bamboo eaten by both pandas in a week is 1316 pounds. -/
theorem pandas_bamboo_consumption :
  let adult_daily_bamboo := 138
  let baby_daily_bamboo := 50
  let days_in_week := 7
  (adult_daily_bamboo * days_in_week) + (baby_daily_bamboo * days_in_week) = 1316 := by
  sorry

end pandas_bamboo_consumption_l133_133963


namespace exists_three_distinct_integers_in_A_l133_133158

noncomputable def A (m n : ℤ) : Set ℤ := { x^2 + m * x + n | x : ℤ }

theorem exists_three_distinct_integers_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a = b * c :=
by
  sorry

end exists_three_distinct_integers_in_A_l133_133158


namespace yellow_chips_are_one_l133_133643

-- Definitions based on conditions
def yellow_chip_points : ℕ := 2
def blue_chip_points : ℕ := 4
def green_chip_points : ℕ := 5

variables (Y B G : ℕ)

-- Given conditions
def point_product_condition : Prop := (yellow_chip_points^Y * blue_chip_points^B * green_chip_points^G = 16000)
def equal_blue_green : Prop := (B = G)

-- Theorem to prove the number of yellow chips
theorem yellow_chips_are_one (Y B G : ℕ) (hprod : point_product_condition Y B G) (heq : equal_blue_green B G) : Y = 1 :=
by {
    sorry -- Proof omitted
}

end yellow_chips_are_one_l133_133643


namespace num_distinct_triangles_in_octahedron_l133_133582

theorem num_distinct_triangles_in_octahedron : ∃ n : ℕ, n = 48 ∧ ∀ (V : Finset (Fin 8)), 
  V.card = 3 → (∀ {a b c : Fin 8}, a ∈ V ∧ b ∈ V ∧ c ∈ V → 
  ¬((a = 0 ∧ b = 1 ∧ c = 2) ∨ (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 6 ∧ b = 7 ∧ c = 8)
  ∨ (a = 7 ∧ b = 0 ∧ c = 1) ∨ (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 6 ∧ c = 7))) :=
by sorry

end num_distinct_triangles_in_octahedron_l133_133582


namespace strawberry_candies_count_l133_133253

theorem strawberry_candies_count (S G : ℕ) (h1 : S + G = 240) (h2 : G = S - 2) : S = 121 :=
by
  sorry

end strawberry_candies_count_l133_133253


namespace increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l133_133928

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l133_133928


namespace find_length_PQ_l133_133326

noncomputable def length_of_PQ (PQ PR : ℝ) (ST SU : ℝ) (angle_PQPR angle_STSU : ℝ) : ℝ :=
if (angle_PQPR = 120 ∧ angle_STSU = 120 ∧ PR / SU = 8 / 9) then 
  2 
else 
  0

theorem find_length_PQ :
  let PQ := 4 
  let PR := 8
  let ST := 9
  let SU := 18
  let PQ_crop := 2
  let angle_PQPR := 120
  let angle_STSU := 120
  length_of_PQ PQ PR ST SU angle_PQPR angle_STSU = PQ_crop :=
by
  sorry

end find_length_PQ_l133_133326


namespace apples_left_is_ten_l133_133914

noncomputable def appleCost : ℝ := 0.80
noncomputable def orangeCost : ℝ := 0.50
def initialApples : ℕ := 50
def initialOranges : ℕ := 40
def totalEarnings : ℝ := 49
def orangesLeft : ℕ := 6

theorem apples_left_is_ten (A : ℕ) :
  (50 - A) * appleCost + (40 - orangesLeft) * orangeCost = 49 → A = 10 :=
by
  sorry

end apples_left_is_ten_l133_133914


namespace part_A_part_B_part_D_l133_133256

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l133_133256


namespace B_can_finish_work_in_18_days_l133_133650

theorem B_can_finish_work_in_18_days : 
  ∃ B_days : ℚ, 
    B_days = 18 ↔ 
    let A_work_rate := (1 : ℚ) / 12,
        B_work_rate := (1 : ℚ) / B_days,
        total_work := 1,
        A_work_done := 2 * A_work_rate,
        remaining_work := total_work - A_work_done,
        combined_work_rate := A_work_rate + B_work_rate in
    6 * combined_work_rate = remaining_work := 
begin
  sorry
end

end B_can_finish_work_in_18_days_l133_133650


namespace problem_l133_133412

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem (h₁ : f a b c 0 = f a b c 4) (h₂ : f a b c 4 > f a b c 1) : a > 0 ∧ 4 * a + b = 0 :=
by 
  sorry

end problem_l133_133412


namespace central_angle_of_unfolded_side_surface_l133_133628

theorem central_angle_of_unfolded_side_surface
  (radius : ℝ) (slant_height : ℝ) (arc_length : ℝ) (central_angle_deg : ℝ)
  (h_radius : radius = 1)
  (h_slant_height : slant_height = 3)
  (h_arc_length : arc_length = 2 * Real.pi) :
  central_angle_deg = 120 :=
by
  sorry

end central_angle_of_unfolded_side_surface_l133_133628


namespace juice_expense_l133_133655

theorem juice_expense (M P : ℕ) 
  (h1 : M + P = 17) 
  (h2 : 5 * M + 6 * P = 94) : 6 * P = 54 :=
by 
  sorry

end juice_expense_l133_133655


namespace most_appropriate_method_to_solve_4x2_minus_9_eq_0_l133_133761

theorem most_appropriate_method_to_solve_4x2_minus_9_eq_0 :
  (∀ x : ℤ, 4 * x^2 - 9 = 0 ↔ x = 3 / 2 ∨ x = -3 / 2) → true :=
by
  sorry

end most_appropriate_method_to_solve_4x2_minus_9_eq_0_l133_133761


namespace range_of_a_l133_133160

noncomputable def satisfies_condition (a : ℝ) : Prop :=
∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs ((1 / 2) * x^3 - a * x) ≤ 1

theorem range_of_a :
  {a : ℝ | satisfies_condition a} = {a : ℝ | - (1 / 2) ≤ a ∧ a ≤ (3 / 2)} :=
by
  sorry

end range_of_a_l133_133160


namespace percent_area_contained_l133_133815

-- Define the conditions as Lean definitions
def side_length_square (s : ℝ) : ℝ := s
def width_rectangle (s : ℝ) : ℝ := 2 * s
def length_rectangle (s : ℝ) : ℝ := 3 * (width_rectangle s)

-- Define areas based on definitions
def area_square (s : ℝ) : ℝ := (side_length_square s) ^ 2
def area_rectangle (s : ℝ) : ℝ := (length_rectangle s) * (width_rectangle s)

-- The main theorem stating the percentage of the rectangle's area contained within the square
theorem percent_area_contained (s : ℝ) (h : s ≠ 0) :
  (area_square s / area_rectangle s) * 100 = 8.33 := by
  sorry

end percent_area_contained_l133_133815


namespace functional_equation_zero_l133_133308

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_equation_zero_l133_133308


namespace ages_sum_is_71_l133_133290

def Beckett_age : ℕ := 12
def Olaf_age : ℕ := Beckett_age + 3
def Shannen_age : ℕ := Olaf_age - 2
def Jack_age : ℕ := 2 * Shannen_age + 5
def sum_of_ages : ℕ := Beckett_age + Olaf_age + Shannen_age + Jack_age

theorem ages_sum_is_71 : sum_of_ages = 71 := by
  unfold sum_of_ages Beckett_age Olaf_age Shannen_age Jack_age
  calc
    12 + (12 + 3) + (12 + 3 - 2) + (2 * (12 + 3 - 2) + 5)
      = 12 + 15 + 13 + 31 := by rfl
      ... = 71 := by rfl

end ages_sum_is_71_l133_133290


namespace janet_savings_l133_133197

def wall1_area := 5 * 8 -- wall 1 area
def wall2_area := 7 * 8 -- wall 2 area
def wall3_area := 6 * 9 -- wall 3 area
def total_area := wall1_area + wall2_area + wall3_area
def tiles_per_square_foot := 4
def total_tiles := total_area * tiles_per_square_foot

def turquoise_tile_cost := 13
def turquoise_labor_cost := 6
def total_cost_turquoise := (total_tiles * turquoise_tile_cost) + (total_area * turquoise_labor_cost)

def purple_tile_cost := 11
def purple_labor_cost := 8
def total_cost_purple := (total_tiles * purple_tile_cost) + (total_area * purple_labor_cost)

def orange_tile_cost := 15
def orange_labor_cost := 5
def total_cost_orange := (total_tiles * orange_tile_cost) + (total_area * orange_labor_cost)

def least_expensive_option := total_cost_purple
def most_expensive_option := total_cost_orange

def savings := most_expensive_option - least_expensive_option

theorem janet_savings : savings = 1950 := by
  sorry

end janet_savings_l133_133197


namespace locus_square_l133_133123

open Real

variables {x y c1 c2 d1 d2 : ℝ}

/-- The locus of points in a square -/
theorem locus_square (h_square: d1 < d2 ∧ c1 < c2) (h_x: d1 ≤ x ∧ x ≤ d2) (h_y: c1 ≤ y ∧ y ≤ c2) :
  |y - c1| + |y - c2| = |x - d1| + |x - d2| :=
by sorry

end locus_square_l133_133123


namespace pretty_number_characterization_l133_133846

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, k < n → ℓ < n → k > 0 → ℓ > 0 → 
    (n ∣ 2*k - ℓ ∨ n ∣ 2*ℓ - k)

theorem pretty_number_characterization :
  ∀ n : ℕ, is_pretty n ↔ (Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end pretty_number_characterization_l133_133846


namespace negation_of_p_l133_133706

open Real

-- Define the statement to be negated
def p := ∀ x : ℝ, -π/2 < x ∧ x < π/2 → tan x > 0

-- Define the negation of the statement
def not_p := ∃ x_0 : ℝ, -π/2 < x_0 ∧ x_0 < π/2 ∧ tan x_0 ≤ 0

-- Theorem stating that the negation of p is not_p
theorem negation_of_p : ¬ p ↔ not_p :=
sorry

end negation_of_p_l133_133706


namespace max_product_of_sum_2020_l133_133111

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l133_133111


namespace sufficient_but_not_necessary_l133_133534

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬((x < -1 ∨ x > 1) → (x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l133_133534


namespace tan_half_angle_l133_133203

theorem tan_half_angle (p q : ℝ) (h_cos : Real.cos p + Real.cos q = 3 / 5) (h_sin : Real.sin p + Real.sin q = 1 / 5) : Real.tan ((p + q) / 2) = 1 / 3 :=
sorry

end tan_half_angle_l133_133203


namespace total_books_97_l133_133878

variable (nDarryl nLamont nLoris : ℕ)

-- Conditions
def condition1 (nLoris nLamont : ℕ) : Prop := nLoris + 3 = nLamont
def condition2 (nLamont nDarryl : ℕ) : Prop := nLamont = 2 * nDarryl
def condition3 (nDarryl : ℕ) : Prop := nDarryl = 20

-- Theorem stating the total number of books is 97
theorem total_books_97 : nLoris + nLamont + nDarryl = 97 :=
by
  have h1 : nDarryl = 20 := condition3 nDarryl
  have h2 : nLamont = 2 * nDarryl := condition2 nLamont nDarryl
  have h3 : nLoris + 3 = nLamont := condition1 nLoris nLamont
  sorry

end total_books_97_l133_133878


namespace edric_hourly_rate_l133_133986

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end edric_hourly_rate_l133_133986


namespace hyperbola_condition_l133_133897

theorem hyperbola_condition (m : ℝ) : ((m - 2) * (m + 3) < 0) ↔ (-3 < m ∧ m < 0) := by
  sorry

end hyperbola_condition_l133_133897


namespace distance_parallel_lines_distance_point_line_l133_133579

def line1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point : ℝ × ℝ := (0, 2)

noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_parallel_lines : distance_between_lines 2 1 (-1) 1 = (2 * Real.sqrt 5) / 5 := by
  sorry

theorem distance_point_line : distance_point_to_line 2 1 (-1) 0 2 = (Real.sqrt 5) / 5 := by
  sorry

end distance_parallel_lines_distance_point_line_l133_133579


namespace student_score_l133_133341

theorem student_score (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 150) : c = 42 :=
by
-- Proof steps here, we skip by using sorry for now
sorry

end student_score_l133_133341


namespace exponent_equality_l133_133330

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l133_133330


namespace distance_foci_of_hyperbola_l133_133309

noncomputable def distance_between_foci : ℝ :=
  8 * Real.sqrt 5

theorem distance_foci_of_hyperbola :
  ∃ A B : ℝ, (9 * A^2 - 36 * A - B^2 + 4 * B = 40) → distance_between_foci = 8 * Real.sqrt 5 :=
sorry

end distance_foci_of_hyperbola_l133_133309


namespace width_of_metallic_sheet_l133_133544

-- Define the given conditions
def length_of_sheet : ℝ := 48
def side_of_square_cut : ℝ := 7
def volume_of_box : ℝ := 5236

-- Define the question as a Lean theorem
theorem width_of_metallic_sheet : ∃ (w : ℝ), w = 36 ∧
  volume_of_box = (length_of_sheet - 2 * side_of_square_cut) * (w - 2 * side_of_square_cut) * side_of_square_cut := by
  sorry

end width_of_metallic_sheet_l133_133544


namespace rob_has_12_pennies_l133_133890

def total_value_in_dollars (quarters dimes nickels pennies : ℕ) : ℚ :=
  (quarters * 25 + dimes * 10 + nickels * 5 + pennies) / 100

theorem rob_has_12_pennies
  (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ)
  (h1 : quarters = 7) (h2 : dimes = 3) (h3 : nickels = 5) 
  (h4 : total_value_in_dollars quarters dimes nickels pennies = 2.42) :
  pennies = 12 :=
by
  sorry

end rob_has_12_pennies_l133_133890


namespace find_primes_l133_133799

theorem find_primes (A B C : ℕ) (hA : A < 20) (hB : B < 20) (hC : C < 20)
  (hA_prime : Prime A) (hB_prime : Prime B) (hC_prime : Prime C)
  (h_sum : A + B + C = 30) : 
  (A = 2 ∧ B = 11 ∧ C = 17) ∨ (A = 2 ∧ B = 17 ∧ C = 11) ∨ 
  (A = 11 ∧ B = 2 ∧ C = 17) ∨ (A = 11 ∧ B = 17 ∧ C = 2) ∨ 
  (A = 17 ∧ B = 2 ∧ C = 11) ∨ (A = 17 ∧ B = 11 ∧ C = 2) :=
sorry

end find_primes_l133_133799


namespace inequality_l133_133027

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.pi)
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem inequality (h1: a = Real.sqrt 2) (h2: b = Real.log 3 / Real.log Real.pi) (h3: c = Real.log 0.5 / Real.log 2) : a > b ∧ b > c := 
by 
  sorry

end inequality_l133_133027


namespace floor_equation_l133_133584

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end floor_equation_l133_133584


namespace sweet_apples_percentage_is_75_l133_133552

noncomputable def percentage_sweet_apples 
  (price_sweet : ℝ) 
  (price_sour : ℝ) 
  (total_apples : ℕ) 
  (total_earnings : ℝ) 
  (percentage_sweet_expr : ℝ) :=
  price_sweet * percentage_sweet_expr + price_sour * (total_apples - percentage_sweet_expr) = total_earnings

theorem sweet_apples_percentage_is_75 :
  percentage_sweet_apples 0.5 0.1 100 40 75 :=
by
  unfold percentage_sweet_apples
  sorry

end sweet_apples_percentage_is_75_l133_133552


namespace max_difference_proof_l133_133802

-- Define the revenue function R(x)
def R (x : ℕ+) : ℝ := 3000 * (x : ℝ) - 20 * (x : ℝ) ^ 2

-- Define the cost function C(x)
def C (x : ℕ+) : ℝ := 500 * (x : ℝ) + 4000

-- Define the profit function P(x) as revenue minus cost
def P (x : ℕ+) : ℝ := R x - C x

-- Define the marginal function M
def M (f : ℕ+ → ℝ) (x : ℕ+) : ℝ := f (⟨x + 1, Nat.succ_pos x⟩) - f x

-- Define the marginal profit function MP(x)
def MP (x : ℕ+) : ℝ := M P x

-- Statement of the proof
theorem max_difference_proof : 
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → P x ≤ P x_max) → -- P achieves its maximum at some x_max within constraints
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → MP x ≤ MP x_max) → -- MP achieves its maximum at some x_max within constraints
  (P x_max - MP x_max = 71680) := 
sorry -- proof omitted

end max_difference_proof_l133_133802


namespace train_passing_platform_time_l133_133818

-- Conditions
variable (l t : ℝ) -- Length of the train and time to pass the pole
variable (v : ℝ) -- Velocity of the train
variable (n : ℝ) -- Multiple of t seconds to pass the platform
variable (d_platform : ℝ) -- Length of the platform

-- Theorem statement
theorem train_passing_platform_time (h1 : d_platform = 3 * l) (h2 : v = l / t) (h3 : n = (l + d_platform) / l) :
  n = 4 := by
  sorry

end train_passing_platform_time_l133_133818


namespace train_speed_l133_133545

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_taken : ℝ := 20
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train : ℝ := total_distance / time_taken

theorem train_speed : speed_of_train = 18.5 :=
  by sorry

end train_speed_l133_133545


namespace length_of_MN_l133_133393

theorem length_of_MN (b : ℝ) (h_focus : ∃ b : ℝ, (3/2, b).1 > 0 ∧ (3/2, b).2 * (3/2, b).2 = 6 * (3 / 2)) : 
  |2 * b| = 6 :=
by sorry

end length_of_MN_l133_133393


namespace exponent_equality_l133_133329

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l133_133329


namespace water_to_milk_ratio_l133_133278

theorem water_to_milk_ratio 
  (V : ℝ) 
  (hV : V > 0) 
  (milk_volume1 : ℝ := (3 / 5) * V) 
  (water_volume1 : ℝ := (2 / 5) * V) 
  (milk_volume2 : ℝ := (4 / 5) * V) 
  (water_volume2 : ℝ := (1 / 5) * V)
  (total_milk_volume : ℝ := milk_volume1 + milk_volume2)
  (total_water_volume : ℝ := water_volume1 + water_volume2) :
  total_water_volume / total_milk_volume = (3 / 7) := 
  sorry

end water_to_milk_ratio_l133_133278


namespace train_passing_time_l133_133277

theorem train_passing_time
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (conversion_factor : ℝ)
  (speed_in_mps : ℝ)
  (time : ℝ)
  (H1 : length_of_train = 65)
  (H2 : speed_in_kmph = 36)
  (H3 : conversion_factor = 5 / 18)
  (H4 : speed_in_mps = speed_in_kmph * conversion_factor)
  (H5 : time = length_of_train / speed_in_mps) :
  time = 6.5 :=
by
  sorry

end train_passing_time_l133_133277


namespace rebus_solution_l133_133995

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l133_133995


namespace triangle_side_split_l133_133246

theorem triangle_side_split
  (PQ QR PR : ℝ)  -- Triangle sides
  (PS SR : ℝ)     -- Segments of PR divided by angle bisector
  (h_ratio : PQ / QR = 3 / 4)
  (h_sum : PR = 15)
  (h_PS_SR : PS / SR = 3 / 4)
  (h_PR_split : PS + SR = PR) :
  SR = 60 / 7 :=
by
  sorry

end triangle_side_split_l133_133246


namespace fill_sink_time_l133_133409

theorem fill_sink_time {R1 R2 R T: ℝ} (h1: R1 = 1 / 210) (h2: R2 = 1 / 214) (h3: R = R1 + R2) (h4: T = 1 / R):
  T = 105.75 :=
by 
  sorry

end fill_sink_time_l133_133409


namespace unrepresentable_integers_l133_133834

theorem unrepresentable_integers :
    {n : ℕ | ∀ a b : ℕ, a > 0 → b > 0 → n ≠ (a * (b + 1) + (a + 1) * b) / (b * (b + 1)) } =
    {1} ∪ {n | ∃ m : ℕ, n = 2^m + 2} :=
by
    sorry

end unrepresentable_integers_l133_133834


namespace quadratic_eq_has_equal_roots_l133_133443

theorem quadratic_eq_has_equal_roots (q : ℚ) :
  (∃ x : ℚ, x^2 - 3 * x + q = 0 ∧ (x^2 - 3 * x + q = 0)) → q = 9 / 4 :=
by
  sorry

end quadratic_eq_has_equal_roots_l133_133443


namespace f_m_plus_1_positive_l133_133428

def f (a x : ℝ) := x^2 + x + a

theorem f_m_plus_1_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := 
  sorry

end f_m_plus_1_positive_l133_133428


namespace injectivity_of_composition_l133_133604

variable {R : Type*} [LinearOrderedField R]

def injective (f : R → R) := ∀ a b, f a = f b → a = b

theorem injectivity_of_composition {f g : R → R} (h : injective (g ∘ f)) : injective f :=
by
  sorry

end injectivity_of_composition_l133_133604


namespace sum_of_remainders_eq_11_mod_13_l133_133271

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l133_133271


namespace Ada_initial_seat_l133_133086

-- We have 6 seats
def Seats := Fin 6

-- Friends' movements expressed in terms of seat positions changes
variable (Bea Ceci Dee Edie Fred Ada : Seats)

-- Conditions about the movements
variable (beMovedRight : Bea.val + 1 = Ada.val)
variable (ceMovedLeft : Ceci.val = Ada.val + 2)
variable (deeMovedRight : Dee.val + 1 = Ada.val)
variable (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
  edie_new = Fred ∧ fred_new = Edie)

-- Ada returns to an end seat (1 or 6)
axiom adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩

-- Theorem to prove Ada's initial position
theorem Ada_initial_seat (Bea Ceci Dee Edie Fred Ada : Seats)
  (beMovedRight : Bea.val + 1 = Ada.val)
  (ceMovedLeft : Ceci.val = Ada.val + 2)
  (deeMovedRight : Dee.val + 1 = Ada.val)
  (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
    edie_new = Fred ∧ fred_new = Edie)
  (adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩) :
  Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩ := sorry

end Ada_initial_seat_l133_133086


namespace cost_of_4_bags_of_ice_l133_133141

theorem cost_of_4_bags_of_ice (
  cost_per_2_bags : ℝ := 1.46
) 
  (h : cost_per_2_bags / 2 = 0.73)
  :
  4 * (cost_per_2_bags / 2) = 2.92 :=
by 
  sorry

end cost_of_4_bags_of_ice_l133_133141


namespace same_fixed_point_l133_133556

open Finset Equiv.Perm

variables (n : ℕ)
-- Denote Sn as the group of permutations of the sequence (1, 2, ..., n)
noncomputable def S_n := equiv.perm (fin n)

-- Assume G is a subgroup of Sn
variable (G : subgroup (S_n n))

-- Assume for every non-identity element in G, there exists a unique k in {1,...,n} such that π(k) = k.
def condition (π : S_n n) : Prop :=
  ∃! k : fin n, π k = k

-- The main theorem to prove
theorem same_fixed_point : 
  (∀ π ∈ G, π ≠ 1 → condition n π) → 
  ∃ k : fin n, ∀ π ∈ G, π ≠ 1 → π k = k :=
  sorry

end same_fixed_point_l133_133556


namespace kostya_initially_planted_l133_133378

def bulbs_after_planting (n : ℕ) (stages : ℕ) : ℕ :=
  match stages with
  | 0 => n
  | k + 1 => 2 * bulbs_after_planting n k - 1

theorem kostya_initially_planted (n : ℕ) (stages : ℕ) :
  bulbs_after_planting n stages = 113 → n = 15 := 
sorry

end kostya_initially_planted_l133_133378


namespace solve_for_y_l133_133304

theorem solve_for_y (y : ℤ) : (4 + y) / (6 + y) = (2 + y) / (3 + y) → y = 0 := by 
  sorry

end solve_for_y_l133_133304


namespace binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l133_133479

-- Definitions and lemma statement
theorem binomial_coefficient_divisible_by_prime
  {p k : ℕ} (hp : Prime p) (hk : 0 < k) (hkp : k < p) :
  p ∣ Nat.choose p k := 
sorry

-- Theorem for k = 0 and k = p cases
theorem binomial_coefficient_extreme_cases {p : ℕ} (hp : Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l133_133479


namespace jackson_money_proof_l133_133728

noncomputable def jackson_money (W : ℝ) := 7 * W
noncomputable def lucy_money (W : ℝ) := 3 * W
noncomputable def ethan_money (W : ℝ) := 3 * W + 20

theorem jackson_money_proof : ∀ (W : ℝ), (W + 7 * W + 3 * W + (3 * W + 20) = 600) → jackson_money W = 290.01 :=
by 
  intros W h
  have total_eq := h
  sorry

end jackson_money_proof_l133_133728


namespace log_eq_l133_133587

theorem log_eq {a b : ℝ} (h₁ : a = Real.log 256 / Real.log 4) (h₂ : b = Real.log 27 / Real.log 3) : 
  a = (4 / 3) * b :=
by
  sorry

end log_eq_l133_133587


namespace product_of_two_numbers_l133_133896

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 :=
by
  sorry

end product_of_two_numbers_l133_133896


namespace gym_monthly_revenue_l133_133387

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l133_133387


namespace max_profit_l133_133140

noncomputable def initial_cost : ℝ := 10
noncomputable def cost_per_pot : ℝ := 0.0027
noncomputable def total_cost (x : ℝ) : ℝ := initial_cost + cost_per_pot * x

noncomputable def P (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 5.7 * x + 19
else 108 - 1000 / (3 * x)

noncomputable def r (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 3 * x + 9
else 98 - 1000 / (3 * x) - 27 * x / 10

theorem max_profit (x : ℝ) : r 10 = 39 :=
sorry

end max_profit_l133_133140


namespace sum_of_remainders_eq_11_mod_13_l133_133269

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l133_133269


namespace gym_monthly_income_l133_133392

theorem gym_monthly_income (bi_monthly_charge : ℕ) (members : ℕ) (monthly_income : ℕ) 
  (h1 : bi_monthly_charge = 18)
  (h2 : members = 300)
  (h3 : monthly_income = 10800) : 
  2 * bi_monthly_charge * members = monthly_income :=
by
  rw [h1, h2, h3]
  norm_num

end gym_monthly_income_l133_133392


namespace ratio_and_equation_imp_value_of_a_l133_133051

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l133_133051


namespace money_left_over_l133_133644

def initial_amount : ℕ := 120
def sandwich_fraction : ℚ := 1 / 5
def museum_ticket_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

theorem money_left_over :
  let sandwich_cost := initial_amount * sandwich_fraction
  let museum_ticket_cost := initial_amount * museum_ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  initial_amount - total_spent = 16 :=
by
  sorry

end money_left_over_l133_133644


namespace quadratic_polynomial_positive_a_l133_133416

variables {a b c n : ℤ}
def p (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem quadratic_polynomial_positive_a (h : a ≠ 0) 
    (hn : n < p n) (hn_ppn : p n < p (p n)) (hn_pppn : p (p n) < p (p (p n))) :
    a > 0 :=
by
    sorry

end quadratic_polynomial_positive_a_l133_133416


namespace num_k_values_lcm_l133_133564

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l133_133564


namespace range_of_values_includes_one_integer_l133_133410

theorem range_of_values_includes_one_integer (x : ℝ) (h : -1 < 2 * x + 3 ∧ 2 * x + 3 < 1) :
  ∃! n : ℤ, -7 < (2 * x - 3) ∧ (2 * x - 3) < -5 ∧ n = -6 :=
sorry

end range_of_values_includes_one_integer_l133_133410


namespace line_intersects_x_axis_between_A_and_B_l133_133037

theorem line_intersects_x_axis_between_A_and_B (a : ℝ) :
  (∀ x, (x = 1 ∨ x = 3) → (2 * x + (3 - a) = 0)) ↔ 5 ≤ a ∧ a ≤ 9 :=
by
  sorry

end line_intersects_x_axis_between_A_and_B_l133_133037


namespace a_power_2018_plus_b_power_2018_eq_2_l133_133324

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem a_power_2018_plus_b_power_2018_eq_2 (a b : ℝ) :
  (∀ x : ℝ, f x a b + f (1 / x) a b = 0) → a^2018 + b^2018 = 2 :=
by 
  sorry

end a_power_2018_plus_b_power_2018_eq_2_l133_133324


namespace value_of_a5_l133_133164

theorem value_of_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n * (n + 1)) (ha : ∀ n, a n = S n - S (n - 1)) :
  a 5 = 20 :=
by
  sorry

end value_of_a5_l133_133164


namespace total_points_combined_l133_133340

-- Definitions of the conditions
def Jack_points : ℕ := 8972
def Alex_Bella_points : ℕ := 21955

-- The problem statement to be proven
theorem total_points_combined : Jack_points + Alex_Bella_points = 30927 :=
by sorry

end total_points_combined_l133_133340


namespace fish_price_relation_l133_133825

variables (b_c m_c b_v m_v : ℝ)

axiom cond1 : 3 * b_c + m_c = 5 * b_v
axiom cond2 : 2 * b_c + m_c = 3 * b_v + m_v

theorem fish_price_relation : 5 * m_v = b_c + 2 * m_c :=
by
  sorry

end fish_price_relation_l133_133825


namespace has_minimum_value_iff_l133_133695

noncomputable def f (a x : ℝ) : ℝ :=
if x < a then -a * x + 4 else (x - 2) ^ 2

theorem has_minimum_value_iff (a : ℝ) : (∃ m, ∀ x, f a x ≥ m) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end has_minimum_value_iff_l133_133695


namespace no_common_perfect_squares_l133_133740

theorem no_common_perfect_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (∃ m n : ℕ, a^2 + 4 * b = m^2 ∧ b^2 + 4 * a = n^2) :=
by
  sorry

end no_common_perfect_squares_l133_133740


namespace loss_is_selling_price_of_16_pencils_l133_133220

theorem loss_is_selling_price_of_16_pencils
  (S : ℝ) -- Assume the selling price of one pencil is S
  (C : ℝ) -- Assume the cost price of one pencil is C
  (h₁ : 80 * C = 1.2 * 80 * S) -- The cost of 80 pencils is 1.2 times the selling price of 80 pencils
  : (80 * C - 80 * S) = 16 * S := -- The loss for selling 80 pencils equals the selling price of 16 pencils
  sorry

end loss_is_selling_price_of_16_pencils_l133_133220


namespace largest_c_in_range_of_f_l133_133151

theorem largest_c_in_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 - 6 * x + c = 2) -> c ≤ 11 :=
by
  sorry

end largest_c_in_range_of_f_l133_133151


namespace find_k_value_l133_133574

theorem find_k_value (S : ℕ → ℕ) (a : ℕ → ℕ) (k : ℤ) 
  (hS : ∀ n, S n = 5 * n^2 + k * n)
  (ha2 : a 2 = 18) :
  k = 3 := 
sorry

end find_k_value_l133_133574


namespace area_of_triangle_DEF_l133_133563

-- Definitions of the given conditions
def angle_D : ℝ := 45
def DF : ℝ := 4
def DE : ℝ := DF -- Because it's a 45-45-90 triangle

-- Leam statement proving the area of the triangle
theorem area_of_triangle_DEF : 
  (1 / 2) * DE * DF = 8 := by
  -- Since DE = DF = 4, the area of the triangle can be computed
  sorry

end area_of_triangle_DEF_l133_133563


namespace sin_sub_pi_over_3_eq_neg_one_third_l133_133697

theorem sin_sub_pi_over_3_eq_neg_one_third {x : ℝ} (h : Real.cos (x + (π / 6)) = 1 / 3) :
  Real.sin (x - (π / 3)) = -1 / 3 := 
  sorry

end sin_sub_pi_over_3_eq_neg_one_third_l133_133697


namespace finite_cuboid_blocks_l133_133472

/--
Prove that there are only finitely many cuboid blocks with integer dimensions a, b, c
such that abc = 2(a - 2)(b - 2)(c - 2) and c ≤ b ≤ a.
-/
theorem finite_cuboid_blocks :
  ∃ (S : Finset (ℤ × ℤ × ℤ)), ∀ (a b c : ℤ), (abc = 2 * (a - 2) * (b - 2) * (c - 2)) → (c ≤ b) → (b ≤ a) → (a, b, c) ∈ S := 
by
  sorry

end finite_cuboid_blocks_l133_133472


namespace third_root_of_cubic_equation_l133_133675

-- Definitions
variable (a b : ℚ) -- We use rational numbers due to the fractions involved
def cubic_equation (x : ℚ) : ℚ := a * x^3 + (a + 3 * b) * x^2 + (2 * b - 4 * a) * x + (10 - a)

-- Conditions
axiom h1 : cubic_equation a b (-1) = 0
axiom h2 : cubic_equation a b 4 = 0

-- The theorem we aim to prove
theorem third_root_of_cubic_equation : ∃ (c : ℚ), c = -62 / 19 ∧ cubic_equation a b c = 0 :=
sorry

end third_root_of_cubic_equation_l133_133675


namespace largest_3_digit_base9_divisible_by_7_l133_133107

def is_three_digit_base9 (n : ℕ) : Prop :=
  n < 9^3

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def base9_to_base10 (n : ℕ) : ℕ :=
  let digits := [n / 81 % 9, n / 9 % 9, n % 9] in
  digits[0] * 81 + digits[1] * 9 + digits[2]

theorem largest_3_digit_base9_divisible_by_7 :
  ∃ n : ℕ, is_three_digit_base9 n ∧ is_divisible_by (base9_to_base10 n) 7 ∧ base9_to_base10 n = 728 ∧ n = 888 :=
sorry

end largest_3_digit_base9_divisible_by_7_l133_133107


namespace simplify_sqrt_expression_l133_133142

theorem simplify_sqrt_expression :
  2 * Real.sqrt 12 - Real.sqrt 27 - (Real.sqrt 3 * Real.sqrt (1 / 9)) = (2 * Real.sqrt 3) / 3 := 
by
  sorry

end simplify_sqrt_expression_l133_133142


namespace pet_store_total_birds_l133_133942

def total_birds_in_pet_store (bird_cages parrots_per_cage parakeets_per_cage : ℕ) : ℕ :=
  bird_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_total_birds :
  total_birds_in_pet_store 4 8 2 = 40 :=
by
  sorry

end pet_store_total_birds_l133_133942


namespace orchard_total_mass_l133_133597

def num_gala_trees := 20
def yield_gala_tree := 120
def num_fuji_trees := 10
def yield_fuji_tree := 180
def num_redhaven_trees := 30
def yield_redhaven_tree := 55
def num_elberta_trees := 15
def yield_elberta_tree := 75

def total_mass_gala := num_gala_trees * yield_gala_tree
def total_mass_fuji := num_fuji_trees * yield_fuji_tree
def total_mass_redhaven := num_redhaven_trees * yield_redhaven_tree
def total_mass_elberta := num_elberta_trees * yield_elberta_tree

def total_mass_fruit := total_mass_gala + total_mass_fuji + total_mass_redhaven + total_mass_elberta

theorem orchard_total_mass : total_mass_fruit = 6975 := by
  sorry

end orchard_total_mass_l133_133597


namespace extremum_at_one_and_value_at_two_l133_133426

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_one_and_value_at_two (a b : ℝ) (h_deriv : 3 + 2*a + b = 0) (h_value : 1 + a + b + a^2 = 10) : 
  f 2 a b = 18 := 
by 
  sorry

end extremum_at_one_and_value_at_two_l133_133426


namespace mark_initial_money_l133_133210

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l133_133210


namespace molecular_weight_bleach_l133_133902

theorem molecular_weight_bleach :
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  molecular_weight = 74.44
:=
by
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  sorry

end molecular_weight_bleach_l133_133902


namespace value_of_expr_l133_133528

theorem value_of_expr (x : ℤ) (h : x = 3) : (2 * x + 6) ^ 2 = 144 := by
  sorry

end value_of_expr_l133_133528


namespace solve_equation_l133_133764

theorem solve_equation (x : ℝ) : 2 * x - 4 = 0 ↔ x = 2 :=
by sorry

end solve_equation_l133_133764


namespace fraction_division_result_l133_133260

theorem fraction_division_result :
  (5/6) / (-9/10) = -25/27 := 
by
  sorry

end fraction_division_result_l133_133260


namespace probability_of_s_in_statistics_l133_133871

theorem probability_of_s_in_statistics :
  let totalLetters := 10
  let count_s := 3
  (count_s / totalLetters : ℚ) = 3 / 10 := by
  sorry

end probability_of_s_in_statistics_l133_133871


namespace sum_of_g1_l133_133069

noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition : ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y := sorry

theorem sum_of_g1 : g 1 = 1 := 
by
  -- Provide the necessary proof steps to show g(1) = 1
  sorry

end sum_of_g1_l133_133069


namespace solve_eq_l133_133763

theorem solve_eq : ∀ x : ℝ, -2 * (x - 1) = 4 → x = -1 := 
by
  intro x
  intro h
  sorry

end solve_eq_l133_133763


namespace probability_four_heads_before_three_tails_l133_133070

theorem probability_four_heads_before_three_tails :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ (q = m / n) ∧ m + n = 61 :=
by
  sorry

end probability_four_heads_before_three_tails_l133_133070


namespace cube_of_composite_as_diff_of_squares_l133_133889

theorem cube_of_composite_as_diff_of_squares (n : ℕ) (h : ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    n^3 = A₁^2 - B₁^2 ∧ 
    n^3 = A₂^2 - B₂^2 ∧ 
    n^3 = A₃^2 - B₃^2 ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) := sorry

end cube_of_composite_as_diff_of_squares_l133_133889


namespace keaton_earns_yearly_l133_133732

/-- Keaton's total yearly earnings from oranges and apples given the harvest cycles and prices. -/
theorem keaton_earns_yearly : 
  let orange_harvest_cycle := 2
  let orange_harvest_price := 50
  let apple_harvest_cycle := 3
  let apple_harvest_price := 30
  let months_in_a_year := 12
  
  let orange_harvests_per_year := months_in_a_year / orange_harvest_cycle
  let apple_harvests_per_year := months_in_a_year / apple_harvest_cycle
  
  let orange_yearly_earnings := orange_harvests_per_year * orange_harvest_price
  let apple_yearly_earnings := apple_harvests_per_year * apple_harvest_price
    
  orange_yearly_earnings + apple_yearly_earnings = 420 :=
by
  sorry

end keaton_earns_yearly_l133_133732


namespace total_money_shared_l133_133666

theorem total_money_shared (rA rB rC : ℕ) (pA : ℕ) (total : ℕ) 
  (h_ratio : rA = 1 ∧ rB = 2 ∧ rC = 7) 
  (h_A_money : pA = 20) 
  (h_total : total = pA * rA + pA * rB + pA * rC) : 
  total = 200 := by 
  sorry

end total_money_shared_l133_133666


namespace greatest_three_digit_base_nine_divisible_by_seven_l133_133106

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l133_133106


namespace line_equation_l133_133418

theorem line_equation (l : ℝ → ℝ → Prop) (a b : ℝ) 
  (h1 : ∀ x y, l x y ↔ y = - (b / a) * x + b) 
  (h2 : l 2 1) 
  (h3 : a + b = 0) : 
  l x y ↔ y = x - 1 ∨ y = x / 2 := 
by
  sorry

end line_equation_l133_133418


namespace ryan_learning_hours_l133_133149

theorem ryan_learning_hours (H_E : ℕ) (H_C : ℕ) (h1 : H_E = 6) (h2 : H_C = 2) : H_E - H_C = 4 := by
  sorry

end ryan_learning_hours_l133_133149


namespace circle_numbers_contradiction_l133_133820

theorem circle_numbers_contradiction :
  ¬ ∃ (f : Fin 25 → Fin 25), ∀ i : Fin 25, 
  let a := f i
  let b := f ((i + 1) % 25)
  (b = a + 10 ∨ b = a - 10 ∨ ∃ k : Int, b = a * k) :=
by
  sorry

end circle_numbers_contradiction_l133_133820


namespace rachel_fathers_age_when_rachel_is_25_l133_133475

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l133_133475


namespace waiter_tables_l133_133961

theorem waiter_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  initial_customers = 62 → 
  customers_left = 17 → 
  people_per_table = 9 → 
  remaining_customers = initial_customers - customers_left →
  tables = remaining_customers / people_per_table →
  tables = 5 :=
by
  intros hinitial hleft hpeople hremaining htables
  rw [hinitial, hleft, hpeople] at *
  simp at *
  sorry

end waiter_tables_l133_133961


namespace solve_combinations_l133_133494

-- This function calculates combinations
noncomputable def C (n k : ℕ) : ℕ := if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combinations (x : ℤ) :
  C 16 (x^2 - x).natAbs = C 16 (5*x - 5).natAbs → x = 1 ∨ x = 3 :=
by
  sorry

end solve_combinations_l133_133494


namespace intersection_equal_l133_133570

-- Define the sets M and N based on given conditions
def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Define the intersection of M and N
def intersection : Set ℝ := {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)}

-- The statement to be proved
theorem intersection_equal : M ∩ N = intersection :=
by 
  sorry -- Skipping the proof

end intersection_equal_l133_133570


namespace charlie_steps_in_running_session_l133_133828

variables (m_steps_3km : ℕ) (times_field : ℕ)
variables (distance_1_field : ℕ) (steps_per_km : ℕ)

-- Conditions
def charlie_steps : ℕ := 5350
def field_distance : ℕ := 3
def run_times : ℚ := 2.5

-- Statement we need to prove
theorem charlie_steps_in_running_session : 
  let distance_ran := run_times * field_distance in
  let total_steps := (charlie_steps * distance_ran) / field_distance in
  total_steps = 13375 := 
by simp [charlie_steps, field_distance, run_times]; sorry

end charlie_steps_in_running_session_l133_133828


namespace part1_part2_l133_133701

open Set

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1 / a) * x + 1

theorem part1 (x : ℝ) : f 2 (2^x) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

theorem part2 (a x : ℝ) (h : a > 2) : f a x ≥ 0 ↔ x ∈ (Iic (1/a) ∪ Ici a) :=
by sorry

end part1_part2_l133_133701


namespace sum_of_possible_values_of_N_l133_133328

variable (N S : ℝ) (hN : N ≠ 0)

theorem sum_of_possible_values_of_N : 
  (3 * N + 5 / N = S) → 
  ∀ N1 N2 : ℝ, (3 * N1^2 - S * N1 + 5 = 0) ∧ (3 * N2^2 - S * N2 + 5 = 0) → 
  N1 + N2 = S / 3 :=
by 
  intro hS hRoots
  sorry

end sum_of_possible_values_of_N_l133_133328


namespace cream_strawberry_prices_l133_133832

noncomputable def price_flavor_B : ℝ := 30
noncomputable def price_flavor_A : ℝ := 40

theorem cream_strawberry_prices (x y : ℝ) 
  (h1 : y = x + 10) 
  (h2 : 800 / y = 600 / x) : 
  x = price_flavor_B ∧ y = price_flavor_A :=
by 
  sorry

end cream_strawberry_prices_l133_133832


namespace barker_high_school_team_count_l133_133400

theorem barker_high_school_team_count (students_total : ℕ) (baseball_team : ℕ) (hockey_team : ℕ) 
  (both_sports : ℕ) : 
  students_total = 36 → baseball_team = 25 → hockey_team = 19 → both_sports = (baseball_team + hockey_team - students_total) → both_sports = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end barker_high_school_team_count_l133_133400


namespace inhabitant_50_statement_l133_133985

-- Definitions
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

def tells_truth (inh: Inhabitant) (statement: Bool) : Bool :=
  match inh with
  | Inhabitant.knight => statement
  | Inhabitant.liar => not statement

noncomputable def inhabitant_at_position (pos: Nat) : Inhabitant :=
  if (pos % 2) = 1 then
    if pos % 4 = 1 then Inhabitant.knight else Inhabitant.liar
  else
    if pos % 4 = 0 then Inhabitant.knight else Inhabitant.liar

def neighbor (pos: Nat) : Nat := (pos % 50) + 1

-- Theorem statement
theorem inhabitant_50_statement : tells_truth (inhabitant_at_position 50) (inhabitant_at_position (neighbor 50) = Inhabitant.knight) = true :=
by
  -- Proof would go here
  sorry

end inhabitant_50_statement_l133_133985


namespace gen_sequence_term_l133_133417

theorem gen_sequence_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ k, a (k + 1) = 3 * a k + 1) :
  a n = (3^n - 1) / 2 := by
  sorry

end gen_sequence_term_l133_133417


namespace range_of_m_l133_133590

theorem range_of_m (m : ℝ) (h1 : m + 3 > 0) (h2 : m - 1 < 0) : -3 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l133_133590


namespace jesse_pencils_l133_133066

def initial_pencils : ℕ := 78
def pencils_given : ℕ := 44
def final_pencils : ℕ := initial_pencils - pencils_given

theorem jesse_pencils :
  final_pencils = 34 :=
by
  -- Proof goes here
  sorry

end jesse_pencils_l133_133066


namespace validate_model_and_profit_range_l133_133943

noncomputable def is_exponential_model_valid (x y : ℝ) : Prop :=
  ∃ T a : ℝ, T > 0 ∧ a > 1 ∧ y = T * a^x

noncomputable def is_profitable_for_at_least_one_billion (x : ℝ) : Prop :=
  (∃ T a : ℝ, T > 0 ∧ a > 1 ∧ 1/5 * (Real.sqrt 2)^x ≥ 10 ∧ 0 < x ∧ x ≤ 12) ∨
  (-0.2 * (x - 12) * (x - 17) + 12.8 ≥ 10 ∧ x > 12)

theorem validate_model_and_profit_range :
  (is_exponential_model_valid 2 0.4) ∧
  (is_exponential_model_valid 4 0.8) ∧
  (is_exponential_model_valid 12 12.8) ∧
  is_profitable_for_at_least_one_billion 11.3 ∧
  is_profitable_for_at_least_one_billion 19 :=
by
  sorry

end validate_model_and_profit_range_l133_133943


namespace two_digit_numbers_div_by_7_with_remainder_1_l133_133662

theorem two_digit_numbers_div_by_7_with_remainder_1 :
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (10 * a + b) % 7 = 1 ∧ (10 * b + a) % 7 = 1} 
  = {22, 29, 92, 99} := 
by
  sorry

end two_digit_numbers_div_by_7_with_remainder_1_l133_133662


namespace find_f7_l133_133698

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f (x)

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = 2 * x^2

theorem find_f7 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : periodic_function f 4)
  (h3 : specific_values f) :
  f 7 = -2 :=
by
  sorry

end find_f7_l133_133698


namespace find_original_cost_price_l133_133642

theorem find_original_cost_price (C S C_new S_new : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) (h3 : S_new = S - 16.80) (h4 : S_new = 1.04 * C_new) : C = 80 :=
by
  sorry

end find_original_cost_price_l133_133642


namespace find_a_l133_133034

-- Define the circle equation and the line equation as conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1
def line_eq (x y a : ℝ) : Prop := y = x + a
def chord_length (l : ℝ) : Prop := l = 2

-- State the main problem
theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, circle_eq x y → ∃ y', line_eq x y' a ∧ chord_length 2) :
  a = -2 :=
sorry

end find_a_l133_133034


namespace product_of_reciprocals_is_9_over_4_l133_133501

noncomputable def product_of_reciprocals (a b : ℝ) : ℝ :=
  (1 / a) * (1 / b)

theorem product_of_reciprocals_is_9_over_4 (a b : ℝ) (h : a + b = 3 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  product_of_reciprocals a b = 9 / 4 :=
sorry

end product_of_reciprocals_is_9_over_4_l133_133501


namespace find_other_endpoint_l133_133243

theorem find_other_endpoint (mx my x₁ y₁ x₂ y₂ : ℤ) 
  (h1 : mx = (x₁ + x₂) / 2) 
  (h2 : my = (y₁ + y₂) / 2) 
  (h3 : mx = 3) 
  (h4 : my = 4) 
  (h5 : x₁ = -2) 
  (h6 : y₁ = -5) : 
  x₂ = 8 ∧ y₂ = 13 := 
by
  sorry

end find_other_endpoint_l133_133243


namespace simplify_fraction_l133_133891

theorem simplify_fraction (h1 : 222 = 2 * 3 * 37) (h2 : 8888 = 8 * 11 * 101) :
  (222 / 8888) * 22 = 1 / 2 :=
by
  sorry

end simplify_fraction_l133_133891


namespace polygon_length_l133_133234

noncomputable def DE : ℝ := 3
noncomputable def EF : ℝ := 6
noncomputable def DE_plus_EF : ℝ := DE + EF

theorem polygon_length 
  (area_ABCDEF : ℝ)
  (AB BC FA : ℝ)
  (A B C D E F : ℝ × ℝ) :
  area_ABCDEF = 60 →
  AB = 10 →
  BC = 7 →
  FA = 6 →
  A = (0, 10) →
  B = (10, 10) →
  C = (10, 0) →
  D = (6, 0) →
  E = (6, 3) →
  F = (0, 3) →
  DE_plus_EF = 9 :=
by
  intros
  sorry

end polygon_length_l133_133234


namespace sequence_x21_zero_l133_133754

theorem sequence_x21_zero (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≤ 10000) (h4 : x2 ≤ 10000) :
  let x3 := |x1 - x2| in
  let x4 := min (|x1 - x2|) (min (|x1 - x3|) (|x2 - x3|)) in
  let seq := fun n => 
    match n with
    | 1 => x1
    | 2 => x2
    | 3 => x3
    | 4 => x4
    | n + 5 => min (|seq (n + 1) - seq (n + 2)|) (min (|seq (n + 1) - seq (n + 3)|) (|seq (n + 2) - seq (n + 3)|)) in
  seq 21 = 0 := sorry

end sequence_x21_zero_l133_133754


namespace base12_remainder_div_7_l133_133789

-- Define the base-12 number 2543 in decimal form
def n : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0

-- Theorem statement: the remainder when n is divided by 7 is 6
theorem base12_remainder_div_7 : n % 7 = 6 := by
  sorry

end base12_remainder_div_7_l133_133789


namespace greatest_base9_3_digit_divisible_by_7_l133_133109

def base9_to_decimal (n : Nat) : Nat :=
  match n with
  | 0     => 0
  | n + 1 => (n % 10) * Nat.pow 9 (n / 10)

def decimal_to_base9 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n => let rec aux (n acc : Nat) :=
              if n = 0 then acc
              else aux (n / 9) ((acc * 10) + (n % 9))
         in aux n 0

theorem greatest_base9_3_digit_divisible_by_7 :
  ∃ (n : Nat), n < Nat.pow 9 3 ∧ (n % 7 = 0) ∧ n = 8 * 81 + 8 * 9 + 8 :=
begin
  sorry -- Proof would go here
end

end greatest_base9_3_digit_divisible_by_7_l133_133109


namespace diag_AC_gt_diag_BD_l133_133359

namespace QuadrilateralProof

-- Define the quadrilateral type with vertices and internal angles
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h_sum : angle_A + angle_B + angle_C + angle_D = 2 * π)

-- Define the conditions for the problem
variables {q : Quadrilateral}
(h_A_acute : 0 < q.angle_A ∧ q.angle_A < π / 2) 
(h_B_obtuse : π / 2 < q.angle_B ∧ q.angle_B < π)
(h_C_obtuse : π / 2 < q.angle_C ∧ q.angle_C < π)
(h_D_obtuse : π / 2 < q.angle_D ∧ q.angle_D < π)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the problem to prove AC > BD
theorem diag_AC_gt_diag_BD : 
  distance q.A q.C > distance q.B q.D :=
sorry

end QuadrilateralProof

end diag_AC_gt_diag_BD_l133_133359


namespace geometric_sequence_a4_l133_133724

-- Define the geometric sequence and known conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ) (q : ℝ)

-- Given conditions:
def a2_eq_4 : Prop := a 2 = 4
def a6_eq_16 : Prop := a 6 = 16

-- The goal is to show a 4 = 8 given the conditions
theorem geometric_sequence_a4 (h_seq : geometric_sequence a q)
  (h_a2 : a2_eq_4 a)
  (h_a6 : a6_eq_16 a) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l133_133724


namespace greatest_product_l133_133113

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l133_133113


namespace flower_shop_options_l133_133213

theorem flower_shop_options :
  {n : ℕ // n = {xy : ℕ × ℕ // 2 * xy.1 + 3 * xy.2 = 20}.card} = 3 :=
by
  sorry

end flower_shop_options_l133_133213


namespace inv_38_mod_53_l133_133317

theorem inv_38_mod_53 (h : 15 * 31 % 53 = 1) : ∃ x : ℤ, 38 * x % 53 = 1 ∧ (x % 53 = 22) :=
by
  sorry

end inv_38_mod_53_l133_133317


namespace relationship_y1_y2_l133_133715

theorem relationship_y1_y2 (x1 x2 y1 y2 : ℝ) 
  (h1: x1 > 0) 
  (h2: 0 > x2) 
  (h3: y1 = 2 / x1)
  (h4: y2 = 2 / x2) : 
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l133_133715


namespace p_minus_q_eq_16_sqrt_2_l133_133204

theorem p_minus_q_eq_16_sqrt_2 (p q : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 28 * x - 84 → x = p ∨ x = q)
  (h_distinct : p ≠ q) (h_p_gt_q : p > q) : p - q = 16 * Real.sqrt 2 :=
sorry

end p_minus_q_eq_16_sqrt_2_l133_133204


namespace exponent_equality_l133_133331

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l133_133331


namespace proof_of_value_of_6y_plus_3_l133_133863

theorem proof_of_value_of_6y_plus_3 (y : ℤ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 :=
by
  sorry

end proof_of_value_of_6y_plus_3_l133_133863


namespace monotonic_intervals_extremum_values_l133_133858

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

theorem monotonic_intervals :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, x > 2 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → deriv f x < 0) := sorry

theorem extremum_values :
  ∃ a b : ℝ, (a = -12) ∧ (b = 15) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≥ b → f x = b) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≤ a → f x = a) := sorry

end monotonic_intervals_extremum_values_l133_133858


namespace incorrect_statement_B_is_wrong_l133_133794

variable (number_of_students : ℕ) (sample_size : ℕ) (population : Set ℕ) (sample : Set ℕ)

-- Conditions
def school_population_is_4000 := number_of_students = 4000
def sample_selected_is_400 := sample_size = 400
def valid_population := population = { x | x < 4000 }
def valid_sample := sample = { x | x < 400 }

-- Incorrect statement (as per given solution)
def incorrect_statement_B := ¬(∀ student ∈ population, true)

theorem incorrect_statement_B_is_wrong 
  (h1 : school_population_is_4000 number_of_students)
  (h2 : sample_selected_is_400 sample_size)
  (h3 : valid_population population)
  (h4 : valid_sample sample)
  : incorrect_statement_B population :=
sorry

end incorrect_statement_B_is_wrong_l133_133794


namespace no_C_makes_2C7_even_and_multiple_of_5_l133_133835

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem no_C_makes_2C7_even_and_multiple_of_5 : ∀ C : ℕ, ¬(C < 10) ∨ ¬(is_even (2 * 100 + C * 10 + 7) ∧ is_multiple_of_5 (2 * 100 + C * 10 + 7)) :=
by
  intro C
  sorry

end no_C_makes_2C7_even_and_multiple_of_5_l133_133835


namespace max_k_no_real_roots_max_integer_value_k_no_real_roots_l133_133048

-- Define the quadratic equation with the condition on the discriminant.
theorem max_k_no_real_roots : ∀ k : ℤ, (4 + 4 * (k : ℝ) < 0) ↔ k < -1 := sorry

-- Prove that the maximum integer value of k satisfying this condition is -2.
theorem max_integer_value_k_no_real_roots : ∃ k_max : ℤ, k_max ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } ∧ ∀ k' : ℤ, k' ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } → k' ≤ k_max :=
sorry

end max_k_no_real_roots_max_integer_value_k_no_real_roots_l133_133048


namespace number_added_multiplied_l133_133129

theorem number_added_multiplied (x : ℕ) (h : (7/8 : ℚ) * x = 28) : ((x + 16) * (5/16 : ℚ)) = 15 :=
by
  sorry

end number_added_multiplied_l133_133129


namespace bouquet_combinations_l133_133540

/--
Given a budget of $60, roses costing $4 each, and carnations costing $2 each, prove that 
there are 16 different combinations of roses and carnations that sum up to exactly $60.
-/
theorem bouquet_combinations : ∃ (r c : ℕ), 4 * r + 2 * c = 60 ∧ finite_combinations 16 :=
by
  sorry

end bouquet_combinations_l133_133540


namespace hyperbola_center_l133_133006

theorem hyperbola_center : 
  (∃ x y : ℝ, (4 * y + 6)^2 / 16 - (5 * x - 3)^2 / 9 = 1) →
  (∃ h k : ℝ, h = 3 / 5 ∧ k = -3 / 2 ∧ 
    (∀ x' y', (4 * y' + 6)^2 / 16 - (5 * x' - 3)^2 / 9 = 1 → x' = h ∧ y' = k)) :=
sorry

end hyperbola_center_l133_133006


namespace find_parabola_equation_l133_133131

noncomputable def parabola_equation (a : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ), 
    F.1 = a / 4 ∧ F.2 = 0 ∧
    A.1 = 0 ∧ A.2 = a / 2 ∧
    (abs (F.1 * A.2) / 2) = 4

theorem find_parabola_equation :
  ∀ (a : ℝ), parabola_equation a → a = 8 ∨ a = -8 :=
by
  sorry

end find_parabola_equation_l133_133131


namespace sum_greater_than_product_l133_133682

theorem sum_greater_than_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
by { sorry }

end sum_greater_than_product_l133_133682


namespace rectangle_length_fraction_of_circle_radius_l133_133098

noncomputable def square_side (area : ℕ) : ℕ :=
  Nat.sqrt area

noncomputable def rectangle_length (breadth area : ℕ) : ℕ :=
  area / breadth

theorem rectangle_length_fraction_of_circle_radius
  (square_area : ℕ)
  (rectangle_breadth : ℕ)
  (rectangle_area : ℕ)
  (side := square_side square_area)
  (radius := side)
  (length := rectangle_length rectangle_breadth rectangle_area) :
  square_area = 4761 →
  rectangle_breadth = 13 →
  rectangle_area = 598 →
  length / radius = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end rectangle_length_fraction_of_circle_radius_l133_133098


namespace combined_salary_ABC_and_E_l133_133908

def salary_D : ℕ := 7000
def avg_salary : ℕ := 9000
def num_individuals : ℕ := 5

theorem combined_salary_ABC_and_E :
  (avg_salary * num_individuals - salary_D) = 38000 :=
by
  -- proof goes here
  sorry

end combined_salary_ABC_and_E_l133_133908


namespace charlie_steps_proof_l133_133829

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l133_133829


namespace find_unknown_rate_l133_133285

theorem find_unknown_rate :
  ∃ x : ℝ, (300 + 750 + 2 * x) / 10 = 170 ↔ x = 325 :=
by
    sorry

end find_unknown_rate_l133_133285


namespace Rachel_father_age_when_Rachel_is_25_l133_133476

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l133_133476


namespace solve_x_squared_solve_x_cubed_l133_133407

-- Define the first problem with its condition and prove the possible solutions
theorem solve_x_squared {x : ℝ} (h : (x + 1)^2 = 9) : x = 2 ∨ x = -4 :=
sorry

-- Define the second problem with its condition and prove the possible solution
theorem solve_x_cubed {x : ℝ} (h : -2 * (x^3 - 1) = 18) : x = -2 :=
sorry

end solve_x_squared_solve_x_cubed_l133_133407


namespace probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l133_133314

noncomputable def p_a : ℝ := 0.18
noncomputable def p_b : ℝ := 0.5
noncomputable def p_b_given_a : ℝ := 0.2
noncomputable def p_c : ℝ := 0.3
noncomputable def p_c_given_a : ℝ := 0.4
noncomputable def p_c_given_b : ℝ := 0.6

noncomputable def p_a_and_b : ℝ := p_a * p_b_given_a
noncomputable def p_a_and_b_and_c : ℝ := p_c_given_a * p_a_and_b
noncomputable def p_a_and_b_given_c : ℝ := p_a_and_b_and_c / p_c
noncomputable def p_a_and_c_given_b : ℝ := p_a_and_b_and_c / p_b

theorem probability_a_and_b_and_c : p_a_and_b_and_c = 0.0144 := by
  sorry

theorem probability_a_and_b_given_c : p_a_and_b_given_c = 0.048 := by
  sorry

theorem probability_a_and_c_given_b : p_a_and_c_given_b = 0.0288 := by
  sorry

end probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l133_133314


namespace cos_gamma_l133_133875

theorem cos_gamma (Q : ℝ × ℝ × ℝ) (h : 0 < Q.1 ∧ 0 < Q.2 ∧ 0 < Q.3)
  (α β γ : ℝ) (cos_alpha_eq : real.cos α = 2 / 5) (cos_beta_eq : real.cos β = 1 / 4) :
  real.cos γ = real.sqrt 311 / 20 :=
by
  sorry

end cos_gamma_l133_133875


namespace length_of_AB_l133_133702

theorem length_of_AB (a b : ℝ) (ha : a > 0) (hb : b = 2 * a)
  (eccentricity_eq : sqrt (1 + (b^2) / (a^2)) = sqrt 5) 
  (A B : ℝ × ℝ)
  (hA : (2 * A.fst - A.snd = 0) ∧ ((A.fst - 2)^2 + (A.snd - 3)^2 = 1))
  (hB : (2 * B.fst - B.snd = 0) ∧ ((B.fst - 2)^2 + (B.snd - 3)^2 = 1)) :
  dist A B = (4 * sqrt 5) / 5 := by sorry

end length_of_AB_l133_133702


namespace solve_for_x_l133_133864

theorem solve_for_x (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 3 ^ (5 / 2) ∨ x = -3 ^ (5 / 2) :=
by
  sorry

end solve_for_x_l133_133864


namespace shaded_area_of_overlap_l133_133922

structure Rectangle where
  width : ℕ
  height : ℕ

structure Parallelogram where
  base : ℕ
  height : ℕ

def area_of_rectangle (r : Rectangle) : ℕ :=
  r.width * r.height

def area_of_parallelogram (p : Parallelogram) : ℕ :=
  p.base * p.height

def overlapping_area_square (side : ℕ) : ℕ :=
  side * side

theorem shaded_area_of_overlap 
  (r : Rectangle)
  (p : Parallelogram)
  (overlapping_side : ℕ)
  (h1 : r.width = 4)
  (h2 : r.height = 12)
  (h3 : p.base = 10)
  (h4 : p.height = 4)
  (h5 : overlapping_side = 4) :
  area_of_rectangle r + area_of_parallelogram p - overlapping_area_square overlapping_side = 72 :=
by
  sorry

end shaded_area_of_overlap_l133_133922


namespace intersection_M_N_l133_133041

def M (x : ℝ) : Prop := abs (x - 1) ≥ 2

def N (x : ℝ) : Prop := x^2 - 4 * x ≥ 0

def P (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4

theorem intersection_M_N (x : ℝ) : (M x ∧ N x) → P x :=
by
  sorry

end intersection_M_N_l133_133041


namespace females_with_advanced_degrees_l133_133937

noncomputable def total_employees := 200
noncomputable def total_females := 120
noncomputable def total_advanced_degrees := 100
noncomputable def males_college_degree_only := 40

theorem females_with_advanced_degrees :
  (total_employees - total_females) - males_college_degree_only = 
  total_employees - total_females - males_college_degree_only ∧ 
  total_females = 120 ∧ 
  total_advanced_degrees = 100 ∧ 
  total_employees = 200 ∧ 
  males_college_degree_only = 40 ∧
  total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60 :=
sorry

end females_with_advanced_degrees_l133_133937


namespace find_k_l133_133015

theorem find_k (k : ℝ) :
  ∃ k, ∀ x : ℝ, (3 * x^3 + k * x^2 - 8 * x + 52) % (3 * x + 4) = 7 :=
by
-- The proof would go here, we insert sorry to acknowledge the missing proof
sorry

end find_k_l133_133015


namespace expression_is_integer_l133_133751

theorem expression_is_integer (m : ℕ) (hm : 0 < m) :
  ∃ k : ℤ, k = (m^4 / 24 + m^3 / 4 + 11*m^2 / 24 + m / 4 : ℚ) :=
by
  sorry

end expression_is_integer_l133_133751


namespace sum_q_p_values_l133_133577

def p (x : ℤ) : ℤ := x^2 - 4

def q (x : ℤ) : ℤ := -abs x

theorem sum_q_p_values : 
  (q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3))) = -20 :=
by
  sorry

end sum_q_p_values_l133_133577


namespace units_digit_k_squared_plus_two_to_k_is_7_l133_133458

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_k_squared_plus_two_to_k_is_7 :
  let k := (2008^2 + 2^2008) in
  units_digit (k^2 + 2^k) = 7 :=
by
  let k := (2008^2 + 2^2008)
  sorry

end units_digit_k_squared_plus_two_to_k_is_7_l133_133458


namespace lcm_value_count_l133_133569

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l133_133569


namespace gumballs_problem_l133_133001

theorem gumballs_problem 
  (L x : ℕ)
  (h1 : 19 ≤ (17 + L + x) / 3 ∧ (17 + L + x) / 3 ≤ 25)
  (h2 : ∃ x_min x_max, x_max - x_min = 18 ∧ x_min = 19 ∧ x = x_min ∨ x = x_max) : 
  L = 21 :=
sorry

end gumballs_problem_l133_133001


namespace sum_of_transformed_numbers_l133_133247

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  a'' + b'' = 3 * S + 24 := 
by
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  sorry

end sum_of_transformed_numbers_l133_133247


namespace alex_buys_15_pounds_of_rice_l133_133364

theorem alex_buys_15_pounds_of_rice (r b : ℝ) 
  (h1 : r + b = 30)
  (h2 : 75 * r + 35 * b = 1650) : 
  r = 15.0 := sorry

end alex_buys_15_pounds_of_rice_l133_133364


namespace find_n_from_binomial_term_l133_133035

noncomputable def binomial_coefficient (n r : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem find_n_from_binomial_term :
  (∃ n : ℕ, 3^2 * binomial_coefficient n 2 = 54) ↔ n = 4 :=
by
  sorry

end find_n_from_binomial_term_l133_133035


namespace problem1_problem2a_problem2b_problem3_l133_133413

noncomputable def f (a x : ℝ) := -x^2 + a * x - 2
noncomputable def g (x : ℝ) := x * Real.log x

-- Problem 1
theorem problem1 {a : ℝ} : (∀ x : ℝ, 0 < x → g x ≥ f a x) → a ≤ 3 :=
sorry

-- Problem 2 
theorem problem2a (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1 / Real.exp 1) :
  ∃ xmin : ℝ, g (1 / Real.exp 1) = -1 / Real.exp 1 ∧ 
  ∃ xmax : ℝ, g (m + 1) = (m + 1) * Real.log (m + 1) :=
sorry

theorem problem2b (m : ℝ) (h₀ : 1 / Real.exp 1 ≤ m) :
  ∃ xmin ymax : ℝ, xmin = g m ∧ ymax = g (m + 1) :=
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : 0 < x) : 
  Real.log x + (2 / (Real.exp 1 * x)) ≥ 1 / Real.exp x :=
sorry

end problem1_problem2a_problem2b_problem3_l133_133413


namespace find_smaller_angle_l133_133782

theorem find_smaller_angle (h : 4 * x + 3 * x = 90) : 3 * (90 / 7) ≈ 38.57 :=
by
  sorry

end find_smaller_angle_l133_133782


namespace find_length_of_PC_l133_133128

theorem find_length_of_PC (P A B C D : ℝ × ℝ) (h1 : (P.1 - A.1)^2 + (P.2 - A.2)^2 = 25)
                            (h2 : (P.1 - D.1)^2 + (P.2 - D.2)^2 = 36)
                            (h3 : (P.1 - B.1)^2 + (P.2 - B.2)^2 = 49)
                            (square_ABCD : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2) :
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = 38 :=
by
  sorry

end find_length_of_PC_l133_133128


namespace find_y_l133_133155

theorem find_y (k p y : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) 
  (h : (y - 2 * k)^2 - (y - 3 * k)^2 = 4 * k^2 - p) : 
  y = -(p + k^2) / (2 * k) :=
sorry

end find_y_l133_133155


namespace S13_equals_26_l133_133419

open Nat

variable (a : Nat → ℕ)

-- Define the arithmetic sequence property
def arithmetic_sequence (d a₁ : Nat → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a₁ + n * d

-- Define the summation property
def sum_of_first_n_terms (S : Nat → ℕ) (a₁ : ℕ) (d : ℕ) : Prop :=
   ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2

-- The given condition
def condition (a₁ d : ℕ) : Prop :=
  2 * (a₁ + 4 * d) + 3 * (a₁ + 6 * d) + 2 * (a₁ + 8 * d) = 14

-- The Lean statement for the proof problem
theorem S13_equals_26 (a₁ d : ℕ) (S : Nat → ℕ) 
  (h_seq : arithmetic_sequence a d a₁) 
  (h_sum : sum_of_first_n_terms S a₁ d)
  (h_cond : condition a₁ d) : 
  S 13 = 26 := 
sorry

end S13_equals_26_l133_133419


namespace identify_quadratic_equation_l133_133927

/-- Proving which equation is a quadratic equation from given options -/
def is_quadratic_equation (eq : String) : Prop :=
  eq = "sqrt(x^2)=2" ∨ eq = "x^2 - x - 2" ∨ eq = "1/x^2 - 2=0" ∨ eq = "x^2=0"

theorem identify_quadratic_equation :
  ∀ (eq : String), is_quadratic_equation eq → eq = "x^2=0" :=
by
  intro eq h
  -- add proof steps here
  sorry

end identify_quadratic_equation_l133_133927


namespace KeatonAnnualEarnings_l133_133735

-- Keaton's conditions for oranges
def orangeHarvestInterval : ℕ := 2
def orangeSalePrice : ℕ := 50

-- Keaton's conditions for apples
def appleHarvestInterval : ℕ := 3
def appleSalePrice : ℕ := 30

-- Annual earnings calculation
def annualEarnings (monthsInYear : ℕ) : ℕ :=
  let orangeEarnings := (monthsInYear / orangeHarvestInterval) * orangeSalePrice
  let appleEarnings := (monthsInYear / appleHarvestInterval) * appleSalePrice
  orangeEarnings + appleEarnings

-- Prove the total annual earnings is 420
theorem KeatonAnnualEarnings : annualEarnings 12 = 420 :=
  by 
    -- We skip the proof details here.
    sorry

end KeatonAnnualEarnings_l133_133735


namespace trapezoid_diagonals_perpendicular_iff_geometric_mean_l133_133225

structure Trapezoid :=
(a b c d e f : ℝ) -- lengths of sides a, b, c, d, and diagonals e, f.
(right_angle : d^2 = a^2 + c^2) -- Condition that makes it a right-angled trapezoid.

theorem trapezoid_diagonals_perpendicular_iff_geometric_mean (T : Trapezoid) :
  (T.e * T.e + T.f * T.f = T.a * T.a + T.b * T.b + T.c * T.c + T.d * T.d) ↔ 
  (T.d * T.d = T.a * T.c) := 
sorry

end trapezoid_diagonals_perpendicular_iff_geometric_mean_l133_133225


namespace max_product_of_sum_2020_l133_133110

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end max_product_of_sum_2020_l133_133110


namespace keaton_earns_yearly_l133_133733

/-- Keaton's total yearly earnings from oranges and apples given the harvest cycles and prices. -/
theorem keaton_earns_yearly : 
  let orange_harvest_cycle := 2
  let orange_harvest_price := 50
  let apple_harvest_cycle := 3
  let apple_harvest_price := 30
  let months_in_a_year := 12
  
  let orange_harvests_per_year := months_in_a_year / orange_harvest_cycle
  let apple_harvests_per_year := months_in_a_year / apple_harvest_cycle
  
  let orange_yearly_earnings := orange_harvests_per_year * orange_harvest_price
  let apple_yearly_earnings := apple_harvests_per_year * apple_harvest_price
    
  orange_yearly_earnings + apple_yearly_earnings = 420 :=
by
  sorry

end keaton_earns_yearly_l133_133733


namespace sum_of_coefficients_l133_133696

theorem sum_of_coefficients (a₅ a₄ a₃ a₂ a₁ a₀ : ℤ)
  (h₀ : (x - 2)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  (h₁ : a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = -1)
  (h₂ : a₀ = -32) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 :=
sorry

end sum_of_coefficients_l133_133696


namespace max_product_of_sum_2020_l133_133115

theorem max_product_of_sum_2020 : 
  ∃ x y : ℤ, x + y = 2020 ∧ (x * y) ≤ 1020100 ∧ (∀ a b : ℤ, a + b = 2020 → a * b ≤ x * y) :=
begin
  sorry
end

end max_product_of_sum_2020_l133_133115


namespace christine_speed_l133_133971

def distance : ℕ := 20
def time : ℕ := 5

theorem christine_speed :
  (distance / time) = 4 := 
sorry

end christine_speed_l133_133971


namespace relation_1_relation_2_relation_3_general_relationship_l133_133989

theorem relation_1 (a b : ℝ) (h1: a = 3) (h2: b = 3) : a^2 + b^2 = 2 * a * b :=
by 
  have h : a = 3 := h1
  have h' : b = 3 := h2
  sorry

theorem relation_2 (a b : ℝ) (h1: a = 2) (h2: b = 1/2) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = 2 := h1
  have h' : b = 1/2 := h2
  sorry

theorem relation_3 (a b : ℝ) (h1: a = -2) (h2: b = 3) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = -2 := h1
  have h' : b = 3 := h2
  sorry

theorem general_relationship (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b :=
by
  sorry

end relation_1_relation_2_relation_3_general_relationship_l133_133989


namespace find_X_value_l133_133717

-- Given definitions and conditions
def X (n : ℕ) : ℕ := 3 + 2 * (n - 1)
def S (n : ℕ) : ℕ := n * (n + 2)

-- Proposition we need to prove
theorem find_X_value : ∃ n : ℕ, S n ≥ 10000 ∧ X n = 201 :=
by
  -- Placeholder for proof
  sorry

end find_X_value_l133_133717


namespace question_1_question_2_l133_133168

-- Condition: The coordinates of point P are given by the equations x = -3a - 4, y = 2 + a

-- Question 1: Prove coordinates when P lies on the x-axis
theorem question_1 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hy0 : y = 0) :
  a = -2 ∧ x = 2 ∧ y = 0 :=
sorry

-- Question 2: Prove coordinates when PQ is parallel to the y-axis
theorem question_2 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hx5 : x = 5) :
  a = -3 ∧ x = 5 ∧ y = -1 :=
sorry

end question_1_question_2_l133_133168


namespace earnings_pool_cleaning_correct_l133_133934

-- Definitions of the conditions
variable (Z : ℕ) -- Number of times Zoe babysat Zachary
variable (earnings_total : ℝ := 8000) 
variable (earnings_Zachary : ℝ := 600)
variable (earnings_per_session : ℝ := earnings_Zachary / Z)
variable (sessions_Julie : ℕ := 3 * Z)
variable (sessions_Chloe : ℕ := 5 * Z)

-- Calculation of earnings from babysitting
def earnings_Julie : ℝ := sessions_Julie * earnings_per_session
def earnings_Chloe : ℝ := sessions_Chloe * earnings_per_session
def earnings_babysitting_total : ℝ := earnings_Zachary + earnings_Julie + earnings_Chloe

-- Calculation of earnings from pool cleaning
def earnings_pool_cleaning : ℝ := earnings_total - earnings_babysitting_total

-- The theorem we are interested in
theorem earnings_pool_cleaning_correct :
  earnings_pool_cleaning Z = 2600 := by
  sorry

end earnings_pool_cleaning_correct_l133_133934


namespace proof_problem_l133_133408

noncomputable def f (x : ℝ) := Real.tan (x + (Real.pi / 4))

theorem proof_problem :
  (- (3 * Real.pi) / 4 < 1 - Real.pi ∧ 1 - Real.pi < -1 ∧ -1 < 0 ∧ 0 < Real.pi / 4) →
  f 0 > f (-1) ∧ f (-1) > f 1 := by
  sorry

end proof_problem_l133_133408


namespace min_value_4x_plus_3y_l133_133169

theorem min_value_4x_plus_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_plus_3y_l133_133169


namespace blocks_for_fort_l133_133727

theorem blocks_for_fort :
  let length := 15 
  let width := 12 
  let height := 6
  let thickness := 1
  let V_original := length * width * height
  let interior_length := length - 2 * thickness
  let interior_width := width - 2 * thickness
  let interior_height := height - thickness
  let V_interior := interior_length * interior_width * interior_height
  let V_blocks := V_original - V_interior
  V_blocks = 430 :=
by
  sorry

end blocks_for_fort_l133_133727


namespace gym_monthly_income_l133_133390

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l133_133390


namespace find_reals_abc_d_l133_133676

theorem find_reals_abc_d (a b c d : ℝ)
  (h1 : a * b * c + a * b + b * c + c * a + a + b + c = 1)
  (h2 : b * c * d + b * c + c * d + d * b + b + c + d = 9)
  (h3 : c * d * a + c * d + d * a + a * c + c + d + a = 9)
  (h4 : d * a * b + d * a + a * b + b * d + d + a + b = 9) :
  a = b ∧ b = c ∧ c = (2 : ℝ)^(1/3) - 1 ∧ d = 5 * (2 : ℝ)^(1/3) - 1 :=
sorry

end find_reals_abc_d_l133_133676


namespace violet_children_count_l133_133512

theorem violet_children_count 
  (family_pass_cost : ℕ := 120)
  (adult_ticket_cost : ℕ := 35)
  (child_ticket_cost : ℕ := 20)
  (separate_ticket_total_cost : ℕ := 155)
  (adult_count : ℕ := 1) : 
  ∃ c : ℕ, 35 + 20 * c = 155 ∧ c = 6 :=
by
  sorry

end violet_children_count_l133_133512


namespace average_age_of_dance_group_l133_133337

theorem average_age_of_dance_group
  (avg_age_children : ℕ)
  (avg_age_adults : ℕ)
  (num_children : ℕ)
  (num_adults : ℕ)
  (total_num_members : ℕ)
  (total_sum_ages : ℕ)
  (average_age : ℚ)
  (h_children : avg_age_children = 12)
  (h_adults : avg_age_adults = 40)
  (h_num_children : num_children = 8)
  (h_num_adults : num_adults = 12)
  (h_total_members : total_num_members = 20)
  (h_total_ages : total_sum_ages = 576)
  (h_average_age : average_age = 28.8) :
  average_age = (total_sum_ages : ℚ) / total_num_members :=
by
  sorry

end average_age_of_dance_group_l133_133337


namespace tiling_rect_divisible_by_4_l133_133813

theorem tiling_rect_divisible_by_4 (m n : ℕ) (h : ∃ k l : ℕ, m = 4 * k ∧ n = 4 * l) : 
  (∃ a : ℕ, m = 4 * a) ∧ (∃ b : ℕ, n = 4 * b) :=
by 
  sorry

end tiling_rect_divisible_by_4_l133_133813


namespace geometric_series_sum_l133_133681

-- Define the first term and common ratio
def a : ℚ := 5 / 3
def r : ℚ := -1 / 6

-- Prove the sum of the infinite geometric series
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 10 / 7 := by
  sorry

end geometric_series_sum_l133_133681


namespace range_of_f_lt_0_l133_133436

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

variable (f : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_decreasing : decreasing_on f (Set.Iic 0))
variable (h_at_2 : f 2 = 0)

theorem range_of_f_lt_0 : ∀ x, x ∈ (Set.Ioo (-2) 0 ∪ Set.Ioi 2) → f x < 0 := by
  sorry

end range_of_f_lt_0_l133_133436


namespace tom_age_l133_133508

theorem tom_age (S T : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 := by
  sorry

end tom_age_l133_133508


namespace cos_of_sin_given_l133_133033

theorem cos_of_sin_given (θ : ℝ) (h : Real.sin (88 * Real.pi / 180 + θ) = 2 / 3) :
  Real.cos (178 * Real.pi / 180 + θ) = - (2 / 3) :=
by
  sorry

end cos_of_sin_given_l133_133033


namespace jogger_ahead_distance_l133_133656

/-- The jogger is running at a constant speed of 9 km/hr, the train at a speed of 45 km/hr,
    it is 210 meters long and passes the jogger in 41 seconds.
    Prove the jogger is 200 meters ahead of the train. -/
theorem jogger_ahead_distance 
  (v_j : ℝ) (v_t : ℝ) (L : ℝ) (t : ℝ) (d : ℝ) 
  (hv_j : v_j = 9) (hv_t : v_t = 45) (hL : L = 210) (ht : t = 41) :
  d = 200 :=
by {
  -- The conditions and the final proof step, 
  -- actual mathematical proofs steps are not necessary according to the problem statement.
  sorry
}

end jogger_ahead_distance_l133_133656


namespace fraction_conversion_integer_l133_133143

theorem fraction_conversion_integer (x : ℝ) :
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 →
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end fraction_conversion_integer_l133_133143


namespace math_proof_problem_l133_133076

theorem math_proof_problem
  (a b c : ℝ)
  (h : a ≠ b)
  (h1 : b ≠ c)
  (h2 : c ≠ a)
  (h3 : (a / (2 * (b - c))) + (b / (2 * (c - a))) + (c / (2 * (a - b))) = 0) :
  (a / (b - c)^3) + (b / (c - a)^3) + (c / (a - b)^3) = 0 := 
by
  sorry

end math_proof_problem_l133_133076


namespace power_equality_l133_133332

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l133_133332


namespace portia_high_school_students_l133_133358

theorem portia_high_school_students (P L : ℕ) (h1 : P = 4 * L) (h2 : P + L = 2500) : P = 2000 := by
  sorry

end portia_high_school_students_l133_133358


namespace smallest_digit_divisibility_l133_133017

theorem smallest_digit_divisibility : 
  ∃ d : ℕ, (d < 10) ∧ (∃ k1 k2 : ℤ, 5 + 2 + 8 + d + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d + 7 + 4 = 3 * k2) ∧ (∀ d' : ℕ, (d' < 10) ∧ 
  (∃ k1 k2 : ℤ, 5 + 2 + 8 + d' + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d' + 7 + 4 = 3 * k2) → d ≤ d') :=
by
  sorry

end smallest_digit_divisibility_l133_133017


namespace find_principal_6400_l133_133486

theorem find_principal_6400 (CI SI P : ℝ) (R T : ℝ) 
  (hR : R = 5) (hT : T = 2) 
  (hSI : SI = P * R * T / 100) 
  (hCI : CI = P * (1 + R / 100) ^ T - P) 
  (hDiff : CI - SI = 16) : 
  P = 6400 := 
by 
  sorry

end find_principal_6400_l133_133486


namespace problem1_l133_133381

theorem problem1 (x y : ℤ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 :=
sorry

end problem1_l133_133381


namespace gym_monthly_income_l133_133389

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l133_133389


namespace problem_solution_l133_133435

theorem problem_solution (a b c d e f g : ℝ) 
  (h1 : a + b + e = 7)
  (h2 : b + c + f = 10)
  (h3 : c + d + g = 6)
  (h4 : e + f + g = 9) : 
  a + d + g = 6 := 
sorry

end problem_solution_l133_133435


namespace find_y_l133_133588

theorem find_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 64) : y = 2 / 3 :=
by
  -- Proof omitted
  sorry

end find_y_l133_133588


namespace original_price_of_apples_l133_133548

-- Define variables and conditions
variables (P : ℝ)

-- The conditions of the problem
def price_increase_condition := 1.25 * P * 8 = 64

-- The theorem stating the original price per pound of apples
theorem original_price_of_apples (h : price_increase_condition P) : P = 6.40 :=
sorry

end original_price_of_apples_l133_133548


namespace andy_cavity_per_candy_cane_l133_133399

theorem andy_cavity_per_candy_cane 
  (cavities_per_candy_cane : ℝ)
  (candy_caned_from_parents : ℝ := 2)
  (candy_caned_each_teacher : ℝ := 3)
  (num_teachers : ℝ := 4)
  (allowance_factor : ℝ := 1/7)
  (total_cavities : ℝ := 16) :
  let total_given_candy : ℝ := candy_caned_from_parents + candy_caned_each_teacher * num_teachers
  let total_bought_candy : ℝ := allowance_factor * total_given_candy
  let total_candy : ℝ := total_given_candy + total_bought_candy
  total_candy / total_cavities = cavities_per_candy_cane :=
by
  sorry

end andy_cavity_per_candy_cane_l133_133399


namespace cos_270_eq_zero_l133_133972

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l133_133972


namespace remainder_division_l133_133526

theorem remainder_division (n : ℕ) :
  n = 2345678901 →
  n % 102 = 65 :=
by sorry

end remainder_division_l133_133526


namespace largest_possible_number_of_pencils_in_a_box_l133_133355

/-- Olivia bought 48 pencils -/
def olivia_pencils : ℕ := 48
/-- Noah bought 60 pencils -/
def noah_pencils : ℕ := 60
/-- Liam bought 72 pencils -/
def liam_pencils : ℕ := 72

/-- The GCD of the number of pencils bought by Olivia, Noah, and Liam is 12 -/
theorem largest_possible_number_of_pencils_in_a_box :
  gcd olivia_pencils (gcd noah_pencils liam_pencils) = 12 :=
by {
  sorry
}

end largest_possible_number_of_pencils_in_a_box_l133_133355


namespace frank_has_3_cookies_l133_133023

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l133_133023


namespace sum_mod_13_l133_133267

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l133_133267


namespace find_quadratic_function_l133_133575

theorem find_quadratic_function (a h k x y : ℝ) (vertex_y : ℝ) (intersect_y : ℝ)
    (hv : h = 1 ∧ k = 2)
    (hi : x = 0 ∧ y = 3) :
    (∀ x, y = a * (x - h) ^ 2 + k) → vertex_y = h ∧ intersect_y = k →
    y = x^2 - 2 * x + 3 :=
by
  sorry

end find_quadratic_function_l133_133575


namespace starting_player_can_ensure_integer_roots_l133_133484

theorem starting_player_can_ensure_integer_roots :
  ∃ (a b c : ℤ), ∀ (x : ℤ), (x^3 + a * x^2 + b * x + c = 0) →
  (∃ r1 r2 r3 : ℤ, x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

end starting_player_can_ensure_integer_roots_l133_133484


namespace remainder_when_divided_l133_133159

theorem remainder_when_divided (k : ℕ) (h_pos : 0 < k) (h_rem : 80 % k = 8) : 150 % (k^2) = 69 := by 
  sorry

end remainder_when_divided_l133_133159


namespace rebus_solution_l133_133992

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l133_133992


namespace least_number_to_add_for_divisibility_by_11_l133_133265

theorem least_number_to_add_for_divisibility_by_11 : ∃ k : ℕ, 11002 + k ≡ 0 [MOD 11] ∧ k = 9 := by
  sorry

end least_number_to_add_for_divisibility_by_11_l133_133265


namespace max_piles_660_l133_133775

noncomputable def max_piles (initial_piles : ℕ) : ℕ :=
  if initial_piles = 660 then 30 else 0

theorem max_piles_660 (initial_piles : ℕ)
  (h : initial_piles = 660) :
  ∃ n, max_piles initial_piles = n ∧ n = 30 :=
begin
  use 30,
  split,
  { rw [max_piles, if_pos h], },
  { refl, },
end

end max_piles_660_l133_133775


namespace dodecahedron_interior_diagonals_l133_133710

-- Definition of a dodecahedron based on given conditions
structure Dodecahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (vertices_per_face : ℕ)
  (faces_per_vertex : ℕ)
  (interior_diagonals : ℕ)

-- Conditions provided in the problem
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    interior_diagonals := 130 }

-- The theorem to prove that given a dodecahedron structure, it has the correct number of interior diagonals
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : d.interior_diagonals = 130 := by
  sorry

end dodecahedron_interior_diagonals_l133_133710


namespace brad_reads_more_pages_l133_133180

-- Definitions based on conditions
def greg_pages_per_day : ℕ := 18
def brad_pages_per_day : ℕ := 26

-- Statement to prove
theorem brad_reads_more_pages : brad_pages_per_day - greg_pages_per_day = 8 :=
by
  -- sorry is used here to indicate the absence of a proof
  sorry

end brad_reads_more_pages_l133_133180


namespace ticket_price_reduction_l133_133249

theorem ticket_price_reduction
    (original_price : ℝ := 50)
    (increase_in_tickets : ℝ := 1 / 3)
    (increase_in_revenue : ℝ := 1 / 4)
    (x : ℝ)
    (reduced_price : ℝ)
    (new_tickets : ℝ := x * (1 + increase_in_tickets))
    (original_revenue : ℝ := x * original_price)
    (new_revenue : ℝ := new_tickets * reduced_price) :
    new_revenue = (1 + increase_in_revenue) * original_revenue →
    reduced_price = original_price - (original_price / 2) :=
    sorry

end ticket_price_reduction_l133_133249


namespace part_A_part_B_part_D_l133_133255

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l133_133255


namespace graph_is_two_lines_l133_133301

theorem graph_is_two_lines : ∀ (x y : ℝ), (x ^ 2 - 25 * y ^ 2 - 20 * x + 100 = 0) ↔ (x = 10 + 5 * y ∨ x = 10 - 5 * y) := 
by 
  intro x y
  sorry

end graph_is_two_lines_l133_133301


namespace remainder_of_sums_modulo_l133_133153

theorem remainder_of_sums_modulo :
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 :=
by
  sorry

end remainder_of_sums_modulo_l133_133153


namespace equation_of_perpendicular_line_intersection_l133_133237

theorem equation_of_perpendicular_line_intersection  :
  ∃ (x y : ℝ), 4 * x + 2 * y + 5 = 0 ∧ 3 * x - 2 * y + 9 = 0 ∧ 
               (∃ (m : ℝ), m = 2 ∧ 4 * x - 2 * y + 11 = 0) := 
sorry

end equation_of_perpendicular_line_intersection_l133_133237


namespace function_increasing_interval_l133_133491

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x ^ 2) / Real.log 2

def domain (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem function_increasing_interval : 
  ∀ x, domain x → 0 < x ∧ x < 1 → ∀ y, domain y → 0 < y ∧ y < 1 → x < y → f x < f y :=
by 
  intros x hx h0 y hy h1 hxy
  sorry

end function_increasing_interval_l133_133491


namespace chord_triangle_count_l133_133850

theorem chord_triangle_count (n : ℕ) (h : n ≥ 6) :
  let S := (Nat.choose n 3) + 4 * (Nat.choose n 4) + 5 * (Nat.choose n 5) + (Nat.choose n 6) in
  ∃ (points : finset (ℤ × ℤ)), 
    points.card = n ∧
    (∀ (p1 p2 p3 : (ℤ × ℤ)), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points → (chord_intersection p1 p2 p3)) ∧
    (∀ (c1 c2 c3 : chord), ¬three_intersect c1 c2 c3) →
    (∃ (triangles : finset (finset (ℤ × ℤ))), triangles.card = S) := sorry

end chord_triangle_count_l133_133850


namespace money_lent_years_l133_133539

noncomputable def compound_interest_time (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem money_lent_years :
  compound_interest_time 740 671.2018140589569 0.05 1 = 2 := by
  sorry

end money_lent_years_l133_133539


namespace eq1_solution_eq2_solution_eq3_solution_eq4_solution_l133_133298

-- Equation 1: 3x^2 - 2x - 1 = 0
theorem eq1_solution (x : ℝ) : 3 * x ^ 2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) :=
by sorry

-- Equation 2: (y + 1)^2 - 4 = 0
theorem eq2_solution (y : ℝ) : (y + 1) ^ 2 - 4 = 0 ↔ (y = 1 ∨ y = -3) :=
by sorry

-- Equation 3: t^2 - 6t - 7 = 0
theorem eq3_solution (t : ℝ) : t ^ 2 - 6 * t - 7 = 0 ↔ (t = 7 ∨ t = -1) :=
by sorry

-- Equation 4: m(m + 3) - 2m = 0
theorem eq4_solution (m : ℝ) : m * (m + 3) - 2 * m = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

end eq1_solution_eq2_solution_eq3_solution_eq4_solution_l133_133298


namespace find_s_of_2_l133_133071

-- Define t and s as per the given conditions
def t (x : ℚ) : ℚ := 4 * x - 9
def s (x : ℚ) : ℚ := x^2 + 4 * x - 5

-- The theorem that we need to prove
theorem find_s_of_2 : s 2 = 217 / 16 := by
  sorry

end find_s_of_2_l133_133071


namespace squirrels_more_than_nuts_l133_133913

theorem squirrels_more_than_nuts (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : nuts = 2) : squirrels - nuts = 2 := by
  sorry

end squirrels_more_than_nuts_l133_133913


namespace area_of_quadrilateral_EFGM_l133_133063

noncomputable def area_ABMJ := 1.8 -- Given area of quadrilateral ABMJ

-- Conditions described in a more abstract fashion:
def is_perpendicular (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of each adjacent pairs being perpendicular
  sorry

def is_congruent (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of all sides except AL and GF being congruent
  sorry

def are_segments_intersecting (B G E L : Point) (M : Point) : Prop :=
  -- Description of segments BG and EL intersecting at point M
  sorry

def area_ratio (tri1 tri2 : Finset Triangle) : ℝ :=
  -- Function that returns the ratio of areas covered by the triangles
  sorry

theorem area_of_quadrilateral_EFGM 
  (A B C D E F G H I J K L M : Point)
  (h1 : is_perpendicular A B C D E F G H I J K L)
  (h2 : is_congruent A B C D E F G H I J K L)
  (h3 : are_segments_intersecting B G E L M)
  : 7 / 3 * area_ABMJ = 4.2 :=
by
  -- Proof of the theorem that area EFGM == 4.2 using the conditions
  sorry

end area_of_quadrilateral_EFGM_l133_133063


namespace average_stoppage_time_per_hour_l133_133917

theorem average_stoppage_time_per_hour :
    ∀ (v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl : ℝ),
    v1_excl = 54 → v1_incl = 36 →
    v2_excl = 72 → v2_incl = 48 →
    v3_excl = 90 → v3_incl = 60 →
    ( ((54 / v1_excl - 54 / v1_incl) + (72 / v2_excl - 72 / v2_incl) + (90 / v3_excl - 90 / v3_incl)) / 3 = 0.5 ) := 
by
    intros v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl
    sorry

end average_stoppage_time_per_hour_l133_133917


namespace fraction_eq_zero_l133_133678

theorem fraction_eq_zero {x : ℝ} (h : (6 * x) ≠ 0) : (x - 5) / (6 * x) = 0 ↔ x = 5 := 
by
  sorry

end fraction_eq_zero_l133_133678


namespace find_constant_c_l133_133434

theorem find_constant_c (c : ℝ) (h : (x + 7) ∣ (c*x^3 + 19*x^2 - 3*c*x + 35)) : c = 3 := by
  sorry

end find_constant_c_l133_133434


namespace antonio_age_in_months_l133_133599

-- Definitions based on the conditions
def is_twice_as_old (isabella_age antonio_age : ℕ) : Prop :=
  isabella_age = 2 * antonio_age

def future_age (current_age months_future : ℕ) : ℕ :=
  current_age + months_future

-- Given the conditions
variables (isabella_age antonio_age : ℕ)
variables (future_age_18months target_age : ℕ)

-- Conditions
axiom condition1 : is_twice_as_old isabella_age antonio_age
axiom condition2 : future_age_18months = 18
axiom condition3 : target_age = 10 * 12

-- Assertion that we need to prove
theorem antonio_age_in_months :
  ∃ (antonio_age : ℕ), future_age isabella_age future_age_18months = target_age → antonio_age = 51 :=
by
  sorry

end antonio_age_in_months_l133_133599


namespace f_at_one_is_zero_f_is_increasing_range_of_x_l133_133852

open Function

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x > 1, f x > 0)
variable (h2 : ∀ x y, f (x * y) = f x + f y)

-- Problem Statements
theorem f_at_one_is_zero : f 1 = 0 := 
sorry

theorem f_is_increasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ > x₂) : 
  f x₁ > f x₂ := 
sorry

theorem range_of_x (f3_eq_1 : f 3 = 1) (x : ℝ) (h3 : x ≥ 1 + Real.sqrt 10) : 
  f x - f (1 / (x - 2)) ≥ 2 := 
sorry

end f_at_one_is_zero_f_is_increasing_range_of_x_l133_133852


namespace retirement_fund_increment_l133_133816

theorem retirement_fund_increment (k y : ℝ) (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27) : k * Real.sqrt y = 810 := by
  sorry

end retirement_fund_increment_l133_133816


namespace find_unique_number_l133_133845

def is_three_digit_number (N : ℕ) : Prop := 100 ≤ N ∧ N < 1000

def nonzero_digits (A B C : ℕ) : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0

def digits_of_number (N A B C : ℕ) : Prop := N = 100 * A + 10 * B + C

def product (N A B : ℕ) := N * (10 * A + B) * A

def divides (n m : ℕ) := ∃ k, n * k = m

theorem find_unique_number (N A B C : ℕ) (h1 : is_three_digit_number N)
    (h2 : nonzero_digits A B C) (h3 : digits_of_number N A B C)
    (h4 : divides 1000 (product N A B)) : N = 875 :=
sorry

end find_unique_number_l133_133845


namespace find_duration_l133_133781

noncomputable def machine_times (x : ℝ) : Prop :=
  let tP := x + 5
  let tQ := x + 3
  let tR := 2 * (x * (x + 3) / 3)
  (1 / tP + 1 / tQ + 1 / tR = 1 / x) ∧ (tP > 0) ∧ (tQ > 0) ∧ (tR > 0)

theorem find_duration {x : ℝ} (h : machine_times x) : x = 3 :=
sorry

end find_duration_l133_133781


namespace quadratic_identity_l133_133471

theorem quadratic_identity
  (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c) / ((a - b) * (a - c))) +
  (b^2 * (x - a) * (x - c) / ((b - a) * (b - c))) +
  (c^2 * (x - a) * (x - b) / ((c - a) * (c - b))) =
  x^2 :=
sorry

end quadratic_identity_l133_133471


namespace dave_apps_left_l133_133144

theorem dave_apps_left (initial_apps deleted_apps remaining_apps : ℕ)
  (h_initial : initial_apps = 23)
  (h_deleted : deleted_apps = 18)
  (h_calculation : remaining_apps = initial_apps - deleted_apps) :
  remaining_apps = 5 := 
by 
  sorry

end dave_apps_left_l133_133144


namespace warehouse_length_l133_133827

theorem warehouse_length (L W : ℕ) (times supposed_times : ℕ) (total_distance : ℕ)
  (h1 : W = 400)
  (h2 : supposed_times = 10)
  (h3 : times = supposed_times - 2)
  (h4 : total_distance = times * (2 * L + 2 * W))
  (h5 : total_distance = 16000) :
  L = 600 := by
  sorry

end warehouse_length_l133_133827


namespace x_plus_q_eq_2q_minus_3_l133_133189

theorem x_plus_q_eq_2q_minus_3 (x q : ℝ) (h1: |x + 3| = q) (h2: x > -3) :
  x + q = 2q - 3 :=
sorry

end x_plus_q_eq_2q_minus_3_l133_133189


namespace part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l133_133857

theorem part1_condition_represents_line (m : ℝ) :
  (m^2 - 2 * m - 3 ≠ 0) ∧ (2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
sorry

theorem part2_slope_does_not_exist (m : ℝ) :
  (m = 1 / 2) ↔ (m^2 - 2 * m - 3 = 0 ∧ (2 * m^2 + m - 1 = 0) ∧ ((1 * x = (4 / 3)))) :=
sorry

theorem part3_x_intercept (m : ℝ) :
  (2 * m - 6) / (m^2 - 2 * m - 3) = -3 ↔ m = -5 / 3 :=
sorry

theorem part4_angle_condition (m : ℝ) :
  -((m^2 - 2 * m - 3) / (2 * m^2 + m - 1)) = 1 ↔ m = 4 / 3 :=
sorry

end part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l133_133857


namespace value_of_x_l133_133089

theorem value_of_x (x y : ℕ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := sorry

end value_of_x_l133_133089


namespace minimum_gumballs_needed_l133_133287

/-- Alex wants to buy at least 150 gumballs,
    and have exactly 14 gumballs left after dividing evenly among 17 people.
    Determine the minimum number of gumballs Alex should buy. -/
theorem minimum_gumballs_needed (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 14) : n = 150 :=
sorry

end minimum_gumballs_needed_l133_133287


namespace recurring_fraction_division_l133_133841

-- Define the values
def x : ℚ := 8 / 11
def y : ℚ := 20 / 11

-- The theorem statement function to prove x / y = 2 / 5
theorem recurring_fraction_division :
  (x / y = (2 : ℚ) / 5) :=
by 
  -- Skip the proof
  sorry

end recurring_fraction_division_l133_133841


namespace triangle_side_lengths_l133_133433

theorem triangle_side_lengths (a b c : ℝ) 
  (h1 : a + b + c = 18) 
  (h2 : a + b = 2 * c) 
  (h3 : b = 2 * a):
  a = 4 ∧ b = 8 ∧ c = 6 := 
by
  sorry

end triangle_side_lengths_l133_133433


namespace find_value_of_m_l133_133623

def ellipse_condition (x y : ℝ) (m : ℝ) : Prop :=
  x^2 + m * y^2 = 1

theorem find_value_of_m (m : ℝ) 
  (h1 : ∀ (x y : ℝ), ellipse_condition x y m)
  (h2 : ∀ a b : ℝ, (a^2 = 1/m ∧ b^2 = 1) ∧ (a = 2 * b)) : 
  m = 1/4 :=
by
  sorry

end find_value_of_m_l133_133623


namespace third_grade_parts_in_batch_l133_133807

-- Define conditions
variable (x y s : ℕ) (h_first_grade : 24 = 24) (h_second_grade : 36 = 36)
variable (h_sample_size : 20 = 20) (h_sample_third_grade : 10 = 10)

-- The problem: Prove the total number of third-grade parts in the batch is 60 and the number of second-grade parts sampled is 6
open Nat

theorem third_grade_parts_in_batch
  (h_total_parts : x - y = 60)
  (h_third_grade_proportion : y = (1 / 2) * x)
  (h_second_grade_proportion : s = (36 / 120) * 20) :
  y = 60 ∧ s = 6 := by
  sorry

end third_grade_parts_in_batch_l133_133807


namespace cost_of_toaster_l133_133729

-- Definitions based on the conditions
def initial_spending : ℕ := 3000
def tv_return : ℕ := 700
def returned_bike_cost : ℕ := 500
def sold_bike_cost : ℕ := returned_bike_cost + (returned_bike_cost / 5)
def selling_price : ℕ := (4 * sold_bike_cost) / 5
def total_out_of_pocket : ℕ := 2020

-- Proving the cost of the toaster
theorem cost_of_toaster : initial_spending - (tv_return + returned_bike_cost) + selling_price - total_out_of_pocket = 260 := by
  sorry

end cost_of_toaster_l133_133729


namespace remainder_division_l133_133612

theorem remainder_division (exists_quotient : ∃ q r : ℕ, r < 5 ∧ N = 5 * 5 + r)
    (exists_quotient_prime : ∃ k : ℕ, N = 11 * k + 3) :
  ∃ r : ℕ, r = 0 ∧ N % 5 = r := 
sorry

end remainder_division_l133_133612


namespace find_monthly_growth_rate_l133_133944

-- Define all conditions.
variables (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ)

-- The conditions from the given problem
def initial_sales (March_sales : ℝ) : Prop := March_sales = 4 * 10^6
def final_sales (May_sales : ℝ) : Prop := May_sales = 9 * 10^6
def growth_occurred (March_sales May_sales : ℝ) (monthly_growth_rate : ℝ) : Prop :=
  May_sales = March_sales * (1 + monthly_growth_rate)^2

-- The Lean 4 theorem to be proven.
theorem find_monthly_growth_rate 
  (h1 : initial_sales March_sales) 
  (h2 : final_sales May_sales) 
  (h3 : growth_occurred March_sales May_sales monthly_growth_rate) : 
  400 * (1 + monthly_growth_rate)^2 = 900 := 
sorry

end find_monthly_growth_rate_l133_133944


namespace khalil_paid_correct_amount_l133_133746

-- Defining the charges for dogs and cats
def cost_per_dog : ℕ := 60
def cost_per_cat : ℕ := 40

-- Defining the number of dogs and cats Khalil took to the clinic
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

-- The total amount Khalil paid
def total_amount_paid : ℕ := 3600

-- The theorem to prove the total amount Khalil paid
theorem khalil_paid_correct_amount :
  (cost_per_dog * num_dogs + cost_per_cat * num_cats) = total_amount_paid :=
by
  sorry

end khalil_paid_correct_amount_l133_133746


namespace marvelous_class_student_count_l133_133215

theorem marvelous_class_student_count (g : ℕ) (jb : ℕ) (jg : ℕ) (j_total : ℕ) (jl : ℕ) (init_jb : ℕ) : 
  jb = g + 3 →  -- Number of boys
  jg = 2 * g + 1 →  -- Jelly beans received by each girl
  init_jb = 726 →  -- Initial jelly beans
  jl = 4 →  -- Leftover jelly beans
  j_total = init_jb - jl →  -- Jelly beans distributed
  (jb * jb + g * jg = j_total) → -- Total jelly beans distributed equation
  2 * g + 1 + g + jb = 31 := -- Total number of students
by
  sorry

end marvelous_class_student_count_l133_133215


namespace astronaut_total_days_l133_133888

-- Definitions of the regular and leap seasons.
def regular_season_days := 49
def leap_season_days := 51

-- Definition of the number of days in different types of years.
def days_in_regular_year := 2 * regular_season_days + 3 * leap_season_days
def days_in_first_3_years := 2 * regular_season_days + 3 * (leap_season_days + 1)
def days_in_years_7_to_9 := 2 * regular_season_days + 3 * (leap_season_days + 2)

-- Calculation for visits.
def first_visit := regular_season_days
def second_visit := 2 * regular_season_days + 3 * (leap_season_days + 1)
def third_visit := 3 * (2 * regular_season_days + 3 * (leap_season_days + 1))
def fourth_visit := 4 * days_in_regular_year + 3 * days_in_first_3_years + 3 * days_in_years_7_to_9

-- Total days spent.
def total_days := first_visit + second_visit + third_visit + fourth_visit

-- The proof statement.
theorem astronaut_total_days : total_days = 3578 :=
by
  -- We place a sorry here to skip the proof.
  sorry

end astronaut_total_days_l133_133888


namespace probability_allison_greater_l133_133665

theorem probability_allison_greater (A D S : ℕ) (prob_derek_less_than_4 : ℚ) (prob_sophie_less_than_4 : ℚ) : 
  (A > D) ∧ (A > S) → prob_derek_less_than_4 = 1 / 2 ∧ prob_sophie_less_than_4 = 2 / 3 → 
  (1 / 2 : ℚ) * (2 / 3 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_allison_greater_l133_133665


namespace problem_l133_133073

def f (x: ℝ) := 3 * x - 4
def g (x: ℝ) := 2 * x + 3

theorem problem (x : ℝ) : f (2 + g 3) = 29 :=
by
  sorry

end problem_l133_133073


namespace defective_rate_worker_y_l133_133561

theorem defective_rate_worker_y (d_x d_y : ℝ) (f_y : ℝ) (total_defective_rate : ℝ) :
  d_x = 0.005 → f_y = 0.8 → total_defective_rate = 0.0074 → 
  (0.2 * d_x + f_y * d_y = total_defective_rate) → d_y = 0.008 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end defective_rate_worker_y_l133_133561


namespace triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l133_133726

theorem triangle_angle_ratio_arbitrary (k1 k2 k3 : ℕ) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ (A / B = k1 / k2) ∧ (A / C = k1 / k3) :=
  sorry

theorem convex_quadrilateral_angle_ratio_not_arbitrary (k1 k2 k3 k4 : ℕ) :
  ¬(∃ (A B C D : ℝ), A + B + C + D = 360 ∧
  A < B + C + D ∧
  B < A + C + D ∧
  C < A + B + D ∧
  D < A + B + C) :=
  sorry

theorem convex_pentagon_angle_ratio_not_arbitrary (k1 k2 k3 k4 k5 : ℕ) :
  ¬(∃ (A B C D E : ℝ), A + B + C + D + E = 540 ∧
  A < (B + C + D + E) / 2 ∧
  B < (A + C + D + E) / 2 ∧
  C < (A + B + D + E) / 2 ∧
  D < (A + B + C + E) / 2 ∧
  E < (A + B + C + D) / 2) :=
  sorry

end triangle_angle_ratio_arbitrary_convex_quadrilateral_angle_ratio_not_arbitrary_convex_pentagon_angle_ratio_not_arbitrary_l133_133726


namespace sin_cos_identity_l133_133689

variables (α : ℝ)

def tan_pi_add_alpha (α : ℝ) : Prop := Real.tan (Real.pi + α) = 3

theorem sin_cos_identity (h : tan_pi_add_alpha α) : 
  Real.sin (-α) * Real.cos (Real.pi - α) = 3 / 10 :=
sorry

end sin_cos_identity_l133_133689


namespace percent_is_250_l133_133384

def part : ℕ := 150
def whole : ℕ := 60
def percent := (part : ℚ) / (whole : ℚ) * 100

theorem percent_is_250 : percent = 250 := 
by 
  sorry

end percent_is_250_l133_133384


namespace probability_of_winning_l133_133916

variable (P_A P_B P_C P_M_given_A P_M_given_B P_M_given_C : ℝ)

theorem probability_of_winning :
  P_A = 0.6 →
  P_B = 0.3 →
  P_C = 0.1 →
  P_M_given_A = 0.1 →
  P_M_given_B = 0.2 →
  P_M_given_C = 0.3 →
  (P_A * P_M_given_A + P_B * P_M_given_B + P_C * P_M_given_C) = 0.15 :=
by sorry

end probability_of_winning_l133_133916


namespace y_coordinates_difference_l133_133872

theorem y_coordinates_difference {m n k : ℤ}
  (h1 : m = 2 * n + 5)
  (h2 : m + 4 = 2 * (n + k) + 5) :
  k = 2 :=
by
  sorry

end y_coordinates_difference_l133_133872


namespace casey_nail_decorating_time_l133_133296

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l133_133296


namespace problem1_problem2_l133_133382

theorem problem1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

theorem problem2 (a b : ℝ) (h : (|a - 2| + |b + 2| = 0) ∨ (|a - 2| * |b + 2| < 0)) : 3a + 2b = 2 := by
  sorry

end problem1_problem2_l133_133382


namespace find_x_l133_133790

theorem find_x (a b c d x : ℕ) 
  (h1 : x = a + 7) 
  (h2 : a = b + 12) 
  (h3 : b = c + 15) 
  (h4 : c = d + 25) 
  (h5 : d = 95) : 
  x = 154 := 
by 
  sorry

end find_x_l133_133790


namespace slices_per_birthday_l133_133478

-- Define the conditions: 
-- k is the age, the number of candles, starting from 3.
variable (k : ℕ) (h : k ≥ 3)

-- Define the function for the number of triangular slices
def number_of_slices (k : ℕ) : ℕ := 2 * k - 5

-- State the theorem to prove that the number of slices is 2k - 5
theorem slices_per_birthday (k : ℕ) (h : k ≥ 3) : 
    number_of_slices k = 2 * k - 5 := 
by
  sorry

end slices_per_birthday_l133_133478


namespace num_dinosaur_dolls_l133_133055

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end num_dinosaur_dolls_l133_133055


namespace balls_distribution_ways_l133_133560

theorem balls_distribution_ways : 
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end balls_distribution_ways_l133_133560


namespace tangent_line_at_five_l133_133038

variable {f : ℝ → ℝ}

theorem tangent_line_at_five 
  (h_tangent : ∀ x, f x = -x + 8)
  (h_tangent_deriv : deriv f 5 = -1) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by sorry

end tangent_line_at_five_l133_133038


namespace compare_star_l133_133004

def star (m n : ℤ) : ℤ := (m + 2) * 3 - n

theorem compare_star : star 2 (-2) > star (-2) 2 := 
by sorry

end compare_star_l133_133004


namespace Glenn_total_expenditure_l133_133745

-- Define initial costs and discounts
def ticket_cost_Monday : ℕ := 5
def ticket_cost_Wednesday : ℕ := 2 * ticket_cost_Monday
def ticket_cost_Saturday : ℕ := 5 * ticket_cost_Monday
def discount_Wednesday (cost : ℕ) : ℕ := cost * 90 / 100
def additional_expense_Saturday : ℕ := 7

-- Define number of attendees
def attendees_Wednesday : ℕ := 4
def attendees_Saturday : ℕ := 2

-- Calculate total costs
def total_cost_Wednesday : ℕ :=
  attendees_Wednesday * discount_Wednesday ticket_cost_Wednesday
def total_cost_Saturday : ℕ :=
  attendees_Saturday * ticket_cost_Saturday + additional_expense_Saturday

-- Calculate the total money spent by Glenn
def total_spent : ℕ :=
  total_cost_Wednesday + total_cost_Saturday

-- Combine all conditions and conclusions into proof statement
theorem Glenn_total_expenditure : total_spent = 93 := by
  sorry

end Glenn_total_expenditure_l133_133745


namespace saly_needs_10_eggs_per_week_l133_133351

theorem saly_needs_10_eggs_per_week :
  let Saly_needs_per_week := S
  let Ben_needs_per_week := 14
  let Ked_needs_per_week := Ben_needs_per_week / 2
  let total_eggs_in_month := 124
  let weeks_per_month := 4
  let Ben_needs_per_month := Ben_needs_per_week * weeks_per_month
  let Ked_needs_per_month := Ked_needs_per_week * weeks_per_month
  let Saly_needs_per_month := total_eggs_in_month - (Ben_needs_per_month + Ked_needs_per_month)
  let S := Saly_needs_per_month / weeks_per_month
  Saly_needs_per_week = 10 :=
by
  sorry

end saly_needs_10_eggs_per_week_l133_133351


namespace find_n_l133_133583

theorem find_n : ∃ n : ℤ, (n^2 / 4).toFloor - (n / 2).toFloor^2 = 5 ∧ n = 11 :=
by
  sorry

end find_n_l133_133583


namespace part1_part2_l133_133074

-- Part 1
theorem part1 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

-- Part 2
theorem part2 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

end part1_part2_l133_133074


namespace digit_A_in_comb_60_15_correct_l133_133194

-- Define the combination function
def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main theorem we want to prove
theorem digit_A_in_comb_60_15_correct : 
  ∃ (A : ℕ), (660 * 10^9 + A * 10^8 + B * 10^7 + 5 * 10^6 + A * 10^4 + 640 * 10^1 + A) = comb 60 15 ∧ A = 6 :=
by
  sorry

end digit_A_in_comb_60_15_correct_l133_133194


namespace ages_sum_l133_133289

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end ages_sum_l133_133289


namespace regular_tickets_sold_l133_133542

variables (S R : ℕ) (h1 : S + R = 65) (h2 : 10 * S + 15 * R = 855)

theorem regular_tickets_sold : R = 41 :=
sorry

end regular_tickets_sold_l133_133542


namespace solution_correct_l133_133900

def mascot_options := ["A Xiang", "A He", "A Ru", "A Yi", "Le Yangyang"]

def volunteer_options := ["A", "B", "C", "D", "E"]

noncomputable def count_valid_assignments (mascots : List String) (volunteers : List String) : Nat :=
  let all_assignments := mascots.permutations
  let valid_assignments := all_assignments.filter (λ p =>
    (p.get! 0 = "A Xiang" ∨ p.get! 1 = "A Xiang") ∧ p.get! 2 ≠ "Le Yangyang")
  valid_assignments.length

theorem solution_correct :
  count_valid_assignments mascot_options volunteer_options = 36 :=
by
  sorry

end solution_correct_l133_133900


namespace ratio_of_c_to_d_l133_133553

theorem ratio_of_c_to_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
    (h1 : 9 * x - 6 * y = c) (h2 : 15 * x - 10 * y = d) :
    c / d = -2 / 5 :=
by
  sorry

end ratio_of_c_to_d_l133_133553


namespace square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l133_133495

theorem square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive :
  (∀ n : ℝ, 0 ≤ n → n^2 ≤ 0 → False) ↔ (∀ m : ℝ, m < 0 → m^2 ≤ 0) := 
sorry

end square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l133_133495


namespace range_of_m_value_of_x_l133_133177

noncomputable def a : ℝ := 3 / 2

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement for the range of m
theorem range_of_m :
  ∀ m : ℝ, f (3 * m - 2) < f (2 * m + 5) ↔ (2 / 3) < m ∧ m < 7 :=
by
  intro m
  sorry

-- Value of x
theorem value_of_x :
  ∃ x : ℝ, f (x - 2 / x) = Real.log (7 / 2) / Real.log (3 / 2) ∧ x > 0 ∧ x = 4 :=
by
  use 4
  sorry

end range_of_m_value_of_x_l133_133177


namespace find_C_and_D_l133_133980

noncomputable def C : ℚ := 51 / 10
noncomputable def D : ℚ := 29 / 10

theorem find_C_and_D (x : ℚ) (h1 : x^2 - 4*x - 21 = (x - 7)*(x + 3))
  (h2 : (8*x - 5) / ((x - 7)*(x + 3)) = C / (x - 7) + D / (x + 3)) :
  C = 51 / 10 ∧ D = 29 / 10 :=
by
  sorry

end find_C_and_D_l133_133980


namespace total_books_borrowed_lunchtime_correct_l133_133861

def shelf_A_borrowed (X : ℕ) : Prop :=
  110 - X = 60 ∧ X = 50

def shelf_B_borrowed (Y : ℕ) : Prop :=
  150 - 50 + 20 - Y = 80 ∧ Y = 80

def shelf_C_borrowed (Z : ℕ) : Prop :=
  210 - 45 = 165 ∧ 165 - 130 = Z ∧ Z = 35

theorem total_books_borrowed_lunchtime_correct :
  ∃ (X Y Z : ℕ),
    shelf_A_borrowed X ∧
    shelf_B_borrowed Y ∧
    shelf_C_borrowed Z ∧
    X + Y + Z = 165 :=
by
  sorry

end total_books_borrowed_lunchtime_correct_l133_133861


namespace vacation_cost_division_l133_133251

theorem vacation_cost_division (n : ℕ) (total_cost : ℕ) 
  (cost_difference : ℕ)
  (cost_per_person_5 : ℕ) :
  total_cost = 1000 → 
  cost_difference = 50 → 
  cost_per_person_5 = total_cost / 5 →
  (total_cost / n) = cost_per_person_5 + cost_difference → 
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end vacation_cost_division_l133_133251


namespace balls_into_boxes_arrangement_l133_133615

theorem balls_into_boxes_arrangement : ∃ n, n = 10 ∧ 
  (∑ x in {1,2,3}, x ≥ 1 ∧ x ∈ { 
      arrangements | multiset.card arrangements = 6 ∧ arrangements.card ≤ 6 ∧ 
      multiset.card arrangements ≥ 1 ∧  arrangements.card ≥ 3 } = 1) :=
begin
  use 10,
  sorry,
end

end balls_into_boxes_arrangement_l133_133615


namespace speed_of_other_person_l133_133229

-- Definitions related to the problem conditions
def pooja_speed : ℝ := 3  -- Pooja's speed in km/hr
def time : ℝ := 4  -- Time in hours
def distance : ℝ := 20  -- Distance between them after 4 hours in km

-- Define the unknown speed S as a parameter to be solved
variable (S : ℝ)

-- Define the relative speed when moving in opposite directions
def relative_speed (S : ℝ) : ℝ := S + pooja_speed

-- Create a theorem to encapsulate the problem and to be proved
theorem speed_of_other_person 
  (h : distance = relative_speed S * time) : S = 2 := 
  sorry

end speed_of_other_person_l133_133229


namespace sector_perimeter_ratio_l133_133053

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) 
  (h1 : α > 0) 
  (h2 : r > 0) 
  (h3 : R > 0) 
  (h4 : (1/2) * α * r^2 / ((1/2) * α * R^2) = 1/4) :
  (2 * r + α * r) / (2 * R + α * R) = 1 / 2 := 
sorry

end sector_perimeter_ratio_l133_133053


namespace carolyn_sum_correct_l133_133893

def initial_sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_removes : List ℕ := [4, 8, 10, 9]

theorem carolyn_sum_correct : carolyn_removes.sum = 31 :=
by
  sorry

end carolyn_sum_correct_l133_133893


namespace algebraic_identity_l133_133383

theorem algebraic_identity (a b : ℝ) : a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by
  sorry

end algebraic_identity_l133_133383


namespace tory_needs_to_raise_more_l133_133019

variable (goal : ℕ) (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
variable (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ)

def remainingAmount (goal : ℕ) 
                    (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
                    (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ) : ℕ :=
  let profitFromChocolateChip := soldChocolateChip * pricePerChocolateChip
  let profitFromOatmealRaisin := soldOatmealRaisin * pricePerOatmealRaisin
  let profitFromSugarCookie := soldSugarCookie * pricePerSugarCookie
  let totalProfit := profitFromChocolateChip + profitFromOatmealRaisin + profitFromSugarCookie
  goal - totalProfit

theorem tory_needs_to_raise_more : 
  remainingAmount 250 6 5 4 5 10 15 = 110 :=
by
  -- Proof omitted 
  sorry

end tory_needs_to_raise_more_l133_133019


namespace incorrect_statement_C_l133_133468

theorem incorrect_statement_C :
  (∀ b h : ℕ, (2 * b) * h = 2 * (b * h)) ∧
  (∀ b h : ℕ, (1 / 2) * b * (2 * h) = 2 * ((1 / 2) * b * h)) ∧
  (∀ r : ℕ, (π * (2 * r) ^ 2 ≠ 2 * (π * r ^ 2))) ∧
  (∀ a b : ℕ, (a / 2) / (2 * b) ≠ a / b) ∧
  (∀ x : ℤ, x < 0 -> 2 * x < x) →
  false :=
by
  intros h
  sorry

end incorrect_statement_C_l133_133468


namespace log_tangent_ratio_l133_133691

open Real

theorem log_tangent_ratio (α β : ℝ) 
  (h1 : sin (α + β) = 1 / 2) 
  (h2 : sin (α - β) = 1 / 3) : 
  log 5 * (tan α / tan β) = 1 := 
sorry

end log_tangent_ratio_l133_133691


namespace cuboid_volume_l133_133946

theorem cuboid_volume (P h : ℝ) (P_eq : P = 32) (h_eq : h = 9) :
  ∃ (s : ℝ), 4 * s = P ∧ s * s * h = 576 :=
by
  sorry

end cuboid_volume_l133_133946


namespace fiftieth_statement_l133_133983

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l133_133983


namespace min_value_expression_l133_133191

theorem min_value_expression (a b: ℝ) (h : 2 * a + b = 1) : (a - 1) ^ 2 + (b - 1) ^ 2 = 4 / 5 :=
sorry

end min_value_expression_l133_133191


namespace remainder_div_x_minus_4_l133_133924

def f (x : ℕ) : ℕ := x^5 - 8 * x^4 + 16 * x^3 + 25 * x^2 - 50 * x + 24

theorem remainder_div_x_minus_4 : 
  (f 4) = 224 := 
by 
  -- Proof goes here
  sorry

end remainder_div_x_minus_4_l133_133924


namespace sum_first_sequence_terms_l133_133496

theorem sum_first_sequence_terms 
  (S : ℕ → ℕ) 
  (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n - S (n - 1) = 2 * n - 1)
  (h2 : S 2 = 3) 
  : a 1 + a 3 = 5 :=
sorry

end sum_first_sequence_terms_l133_133496


namespace only_rational_root_is_one_l133_133013

-- Define the polynomial
def polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 (x : ℚ) : ℚ :=
  3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

-- The main theorem stating that 1 is the only rational root
theorem only_rational_root_is_one : 
  ∀ x : ℚ, polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 x = 0 ↔ x = 1 :=
by
  sorry

end only_rational_root_is_one_l133_133013


namespace point_in_second_quadrant_l133_133061

theorem point_in_second_quadrant (m n : ℝ)
  (h_translation : ∃ A' : ℝ × ℝ, A' = (m+2, n+3) ∧ (A'.1 < 0) ∧ (A'.2 > 0)) :
  m < -2 ∧ n > -3 :=
by
  sorry

end point_in_second_quadrant_l133_133061


namespace calculate_molecular_weight_l133_133787

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def num_atoms_C := 3
def num_atoms_H := 6
def num_atoms_O := 1

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  (nC * wC) + (nH * wH) + (nO * wO)

theorem calculate_molecular_weight :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 58.078 :=
by
  sorry

end calculate_molecular_weight_l133_133787


namespace spaghetti_tortellini_ratio_l133_133646

theorem spaghetti_tortellini_ratio (students_surveyed : ℕ)
                                    (spaghetti_lovers : ℕ)
                                    (tortellini_lovers : ℕ)
                                    (h1 : students_surveyed = 850)
                                    (h2 : spaghetti_lovers = 300)
                                    (h3 : tortellini_lovers = 200) :
  spaghetti_lovers / tortellini_lovers = 3 / 2 :=
by
  sorry

end spaghetti_tortellini_ratio_l133_133646


namespace total_passengers_landed_l133_133602

theorem total_passengers_landed (on_time late : ℕ) (h_on_time : on_time = 14507) (h_late : late = 213) :
  on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l133_133602


namespace five_year_salary_increase_l133_133883

noncomputable def salary_growth (S : ℝ) := S * (1.08)^5

theorem five_year_salary_increase (S : ℝ) : 
  salary_growth S = S * 1.4693 := 
sorry

end five_year_salary_increase_l133_133883


namespace find_TU_square_l133_133363

-- Definitions
variables (P Q R S T U : ℝ × ℝ)
variable (side : ℝ)
variable (QT RU PT SU PQ : ℝ)

-- Setting the conditions
variables (side_eq_10 : side = 10)
variables (QT_eq_7 : QT = 7)
variables (RU_eq_7 : RU = 7)
variables (PT_eq_24 : PT = 24)
variables (SU_eq_24 : SU = 24)
variables (PQ_eq_10 : PQ = 10)

-- The theorem statement
theorem find_TU_square : TU^2 = 1150 :=
by
  -- Proof to be done here.
  sorry

end find_TU_square_l133_133363


namespace discriminant_eq_13_l133_133335

theorem discriminant_eq_13 (m : ℝ) (h : (3)^2 - 4*1*(-m) = 13) : m = 1 :=
sorry

end discriminant_eq_13_l133_133335


namespace length_of_train_l133_133241

noncomputable def train_length : ℕ := 1200

theorem length_of_train 
  (L : ℝ) 
  (speed_km_per_hr : ℝ) 
  (time_min : ℕ) 
  (speed_m_per_s : ℝ) 
  (time_sec : ℕ) 
  (distance : ℝ) 
  (cond1 : L = L)
  (cond2 : speed_km_per_hr = 144) 
  (cond3 : time_min = 1)
  (cond4 : speed_m_per_s = speed_km_per_hr * 1000 / 3600)
  (cond5 : time_sec = time_min * 60)
  (cond6 : distance = speed_m_per_s * time_sec)
  (cond7 : 2 * L = distance)
  : L = train_length := 
sorry

end length_of_train_l133_133241


namespace projections_concyclic_l133_133874

open EuclideanGeometry

variables {A B C D A' C' B' D' : Point}
variables {BD AC : Line}

-- Given conditions
variable (h1 : OnCircle A B C D)
variable (h2 : Proj A BD A')
variable (h3 : Proj C BD C')
variable (h4 : Proj B AC B')
variable (h5 : Proj D AC D')

-- Proof goal
theorem projections_concyclic :
  Concyclic A' B' C' D' := sorry

end projections_concyclic_l133_133874


namespace sales_tax_per_tire_l133_133920

def cost_per_tire : ℝ := 7
def number_of_tires : ℕ := 4
def final_total_cost : ℝ := 30

theorem sales_tax_per_tire :
  (final_total_cost - number_of_tires * cost_per_tire) / number_of_tires = 0.5 :=
sorry

end sales_tax_per_tire_l133_133920


namespace calc_pairs_count_l133_133649

theorem calc_pairs_count :
  ∃! (ab : ℤ × ℤ), (ab.1 + ab.2 = ab.1 * ab.2) :=
by
  sorry

end calc_pairs_count_l133_133649


namespace ratio_debt_manny_to_annika_l133_133509

-- Define the conditions
def money_jericho_has : ℕ := 30
def debt_to_annika : ℕ := 14
def remaining_money_after_debts : ℕ := 9

-- Define the amount Jericho owes Manny
def debt_to_manny : ℕ := money_jericho_has - debt_to_annika - remaining_money_after_debts

-- Prove the ratio of amount Jericho owes Manny to the amount he owes Annika is 1:2
theorem ratio_debt_manny_to_annika :
  debt_to_manny * 2 = debt_to_annika :=
by
  -- Proof goes here
  sorry

end ratio_debt_manny_to_annika_l133_133509


namespace cos_270_eq_zero_l133_133973

theorem cos_270_eq_zero : ∀ (θ : ℝ), θ = 270 → (∀ (p : ℝ × ℝ), p = (0,1) → cos θ = p.fst) → cos 270 = 0 :=
by
  intros θ hθ h
  sorry

end cos_270_eq_zero_l133_133973


namespace student_percentage_first_subject_l133_133137

theorem student_percentage_first_subject
  (P : ℝ)
  (h1 : (P + 60 + 70) / 3 = 60) : P = 50 :=
  sorry

end student_percentage_first_subject_l133_133137


namespace complex_sum_equals_one_l133_133205

noncomputable def main (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1))

theorem complex_sum_equals_one (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : main x h1 h2 = 1 := by
  sorry

end complex_sum_equals_one_l133_133205


namespace minimum_sum_of_distances_squared_l133_133421

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared_l133_133421


namespace find_sum_of_variables_l133_133712

theorem find_sum_of_variables (x y : ℚ) (h1 : 5 * x - 3 * y = 17) (h2 : 3 * x + 5 * y = 1) : x + y = 21 / 17 := 
  sorry

end find_sum_of_variables_l133_133712


namespace store_breaks_even_l133_133945

-- Defining the conditions based on the problem statement.
def cost_price_piece1 (profitable : ℝ → Prop) : Prop :=
  ∃ x, profitable x ∧ 1.5 * x = 150

def cost_price_piece2 (loss : ℝ → Prop) : Prop :=
  ∃ y, loss y ∧ 0.75 * y = 150

def profitable (x : ℝ) : Prop := x + 0.5 * x = 150
def loss (y : ℝ) : Prop := y - 0.25 * y = 150

-- Store breaks even if the total cost price equals the total selling price
theorem store_breaks_even (x y : ℝ)
  (P1 : cost_price_piece1 profitable)
  (P2 : cost_price_piece2 loss) :
  (x + y = 100 + 200) → (150 + 150) = 300 :=
by
  sorry

end store_breaks_even_l133_133945


namespace required_run_rate_l133_133938

theorem required_run_rate (run_rate_first_10_overs : ℝ) (target_runs total_overs first_overs : ℕ) :
  run_rate_first_10_overs = 4.2 ∧ target_runs = 282 ∧ total_overs = 50 ∧ first_overs = 10 →
  (target_runs - run_rate_first_10_overs * first_overs) / (total_overs - first_overs) = 6 :=
by
  sorry

end required_run_rate_l133_133938


namespace max_band_members_l133_133954

variable (r x m : ℕ)

noncomputable def band_formation (r x m: ℕ) :=
  m = r * x + 4 ∧
  m = (r - 3) * (x + 2) ∧
  m < 100

theorem max_band_members (r x m : ℕ) (h : band_formation r x m) : m = 88 :=
by
  sorry

end max_band_members_l133_133954


namespace minimum_value_ab_l133_133046

theorem minimum_value_ab (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a * b - 2 * a - b = 0) :
  8 ≤ a * b :=
by sorry

end minimum_value_ab_l133_133046


namespace ratio_of_areas_l133_133397

-- Definitions and conditions
variables (s r : ℝ)
variables (h1 : 4 * s = 4 * π * r)

-- Statement to prove
theorem ratio_of_areas (h1 : 4 * s = 4 * π * r) : s^2 / (π * r^2) = π := by
  sorry

end ratio_of_areas_l133_133397


namespace games_attended_this_month_l133_133731

theorem games_attended_this_month 
  (games_last_month games_next_month total_games games_this_month : ℕ)
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44)
  (h4 : games_last_month + games_this_month + games_next_month = total_games) : 
  games_this_month = 11 := by
  sorry

end games_attended_this_month_l133_133731


namespace sequence_properties_l133_133157

def seq (n : ℕ) : ℕ → ℝ := λ i, 2 * i

def G (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in range n, (2 : ℝ) ^ i * a (i + 1)) / n

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, a (i + 1)

theorem sequence_properties:
  let a := seq n in
  (∀ n, G a n = 2^n) →
  (a (n+1) = n+1) ∧
  (S a 2023 / 2023 = 1013) ∧
  let b := λ n, (9 / 10) ^ n * a (n + 1) in
  ∃ n, ∀ m, b n ≥ b (n + m) := sorry

end sequence_properties_l133_133157


namespace ratio_Ryn_Nikki_l133_133601

def Joyce_movie_length (M : ℝ) : ℝ := M + 2
def Nikki_movie_length (M : ℝ) : ℝ := 3 * M
def Ryn_movie_fraction (F : ℝ) (Nikki_movie_length : ℝ) : ℝ := F * Nikki_movie_length

theorem ratio_Ryn_Nikki 
  (M : ℝ) 
  (Nikki_movie_is_30 : Nikki_movie_length M = 30) 
  (total_movie_hours_is_76 : M + Joyce_movie_length M + Nikki_movie_length M + Ryn_movie_fraction F (Nikki_movie_length M) = 76) 
  : F = 4 / 5 := 
by 
  sorry

end ratio_Ryn_Nikki_l133_133601


namespace inequality_of_ab_l133_133346

theorem inequality_of_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 :=
by
  sorry

end inequality_of_ab_l133_133346


namespace simplify_and_evaluate_l133_133085

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_l133_133085


namespace part1_part2_case1_part2_case2_part2_case3_l133_133161

namespace InequalityProof

variable {a x : ℝ}

def f (a x : ℝ) := a * x^2 + x - a

theorem part1 (h : a = 1) : (x > 1 ∨ x < -2) → f a x > 1 :=
by sorry

theorem part2_case1 (h1 : a < 0) (h2 : a < -1/2) : (- (a + 1) / a) < x ∧ x < 1 → f a x > 1 :=
by sorry

theorem part2_case2 (h1 : a < 0) (h2 : a = -1/2) : x ≠ 1 → f a x > 1 :=
by sorry

theorem part2_case3 (h1 : a < 0) (h2 : 0 > a) (h3 : a > -1/2) : 1 < x ∧ x < - (a + 1) / a → f a x > 1 :=
by sorry

end InequalityProof

end part1_part2_case1_part2_case2_part2_case3_l133_133161


namespace largest_digit_divisible_by_6_l133_133516

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l133_133516


namespace find_increasing_function_l133_133930

-- Define each function
def fA (x : ℝ) := -x
def fB (x : ℝ) := (2 / 3) ^ x
def fC (x : ℝ) := x ^ 2
def fD (x : ℝ) := x ^ (1 / 3)

-- Define the statement that fD is the only increasing function among the options
theorem find_increasing_function (f : ℝ → ℝ) (hf : f = fD) :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fA x < fA y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fB x < fB y) ∧ 
  (¬ ∀ x y : ℝ, x < y → fC x < fC y) :=
by {
  sorry
}

end find_increasing_function_l133_133930


namespace sum_of_remainders_eq_11_mod_13_l133_133270

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l133_133270


namespace percentage_increase_l133_133970

theorem percentage_increase (use_per_six_months : ℝ) (new_annual_use : ℝ) : 
  use_per_six_months = 90 →
  new_annual_use = 216 →
  ((new_annual_use - 2 * use_per_six_months) / (2 * use_per_six_months)) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_increase_l133_133970


namespace markup_percentage_l133_133283

theorem markup_percentage 
  (CP : ℝ) (x : ℝ) (MP : ℝ) (SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (x / 100) * CP)
  (h3 : SP = MP - (10 / 100) * MP)
  (h4 : SP = CP + (35 / 100) * CP) :
  x = 50 :=
by sorry

end markup_percentage_l133_133283


namespace no_infinite_lines_satisfying_conditions_l133_133009

theorem no_infinite_lines_satisfying_conditions :
  ¬ ∃ (l : ℕ → ℝ → ℝ → Prop)
      (k : ℕ → ℝ)
      (a b : ℕ → ℝ),
    (∀ n, l n 1 1) ∧
    (∀ n, k (n + 1) = a n - b n) ∧
    (∀ n, k n * k (n + 1) ≥ 0) := 
sorry

end no_infinite_lines_satisfying_conditions_l133_133009


namespace motorcycles_count_l133_133721

/-- In a parking lot, there are cars and motorcycles. 
    Each car has 5 wheels (including one spare) and each motorcycle has 2 wheels. 
    There are 19 cars in the parking lot. 
    Altogether all vehicles have 117 wheels. 
    Prove that there are 11 motorcycles in the parking lot. -/
theorem motorcycles_count 
  (C M : ℕ)
  (hc : C = 19)
  (total_wheels : ℕ)
  (total_wheels_eq : total_wheels = 117)
  (car_wheels : ℕ)
  (car_wheels_eq : car_wheels = 5 * C)
  (bike_wheels : ℕ)
  (bike_wheels_eq : bike_wheels = total_wheels - car_wheels)
  (wheels_per_bike : ℕ)
  (wheels_per_bike_eq : wheels_per_bike = 2):
  M = bike_wheels / wheels_per_bike :=
by
  sorry

end motorcycles_count_l133_133721


namespace highest_temperature_l133_133822

theorem highest_temperature
  (initial_temp : ℝ := 60)
  (final_temp : ℝ := 170)
  (heating_rate : ℝ := 5)
  (cooling_rate : ℝ := 7)
  (total_time : ℝ := 46) :
  ∃ T : ℝ, (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time ∧ T = 240 :=
by
  sorry

end highest_temperature_l133_133822


namespace soda_cost_l133_133104

variable (b s : ℕ)

theorem soda_cost (h1 : 2 * b + s = 210) (h2 : b + 2 * s = 240) : s = 90 := by
  sorry

end soda_cost_l133_133104


namespace sum_of_numbers_l133_133279

theorem sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) : a + b + c = 22 :=
by
  sorry

end sum_of_numbers_l133_133279


namespace find_special_number_l133_133638

-- Define the function to reverse the digits of a natural number.
def reverseDigits (n : Nat) : Nat :=
  n.toDigits.reverse.foldl (λ acc d => acc * 10 + d) 0

-- Define the proof problem statement.
theorem find_special_number : ∃ n : Nat, 1 ≤ n ∧ n ≤ 10000 ∧ reverseDigits n = Nat.ceil (n / 2) ∧ n = 7993 :=
by sorry

end find_special_number_l133_133638


namespace incorrect_judgment_l133_133162

variable (p q : Prop)
variable (hyp_p : p = (3 + 3 = 5))
variable (hyp_q : q = (5 > 2))

theorem incorrect_judgment : 
  (¬ (p ∧ q) ∧ ¬p) = false :=
by
  sorry

end incorrect_judgment_l133_133162


namespace max_value_frac_x1_x2_et_l133_133029

theorem max_value_frac_x1_x2_et (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x * Real.exp x)
  (hg : ∀ x, g x = - (Real.log x) / x)
  (x1 x2 t : ℝ)
  (hx1 : f x1 = t)
  (hx2 : g x2 = t)
  (ht_pos : t > 0) :
  ∃ x1 x2, (f x1 = t ∧ g x2 = t) ∧ (∀ u v, (f u = t ∧ g v = t → u / (v * Real.exp t) ≤ 1 / Real.exp 1)) :=
by
  sorry

end max_value_frac_x1_x2_et_l133_133029


namespace age_of_youngest_child_l133_133280

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) :
  x = 4 :=
sorry

end age_of_youngest_child_l133_133280


namespace vincent_correct_answer_l133_133636

theorem vincent_correct_answer (y : ℕ) (h : (y - 7) / 5 = 23) : (y - 5) / 7 = 17 :=
by
  sorry

end vincent_correct_answer_l133_133636


namespace rebus_solution_l133_133993

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l133_133993


namespace probability_of_same_color_balls_l133_133633

-- Definitions of the problem
def total_balls_bag_A := 8 + 4
def total_balls_bag_B := 6 + 6
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

def P (event: Nat -> Bool) (total: Nat) : Nat :=
  let favorable := (List.range total).filter event |>.length
  favorable / total

-- Probability of drawing a white ball from bag A
def P_A := P (λ n => n < white_balls_bag_A) total_balls_bag_A

-- Probability of drawing a red ball from bag A
def P_not_A := P (λ n => n >= white_balls_bag_A && n < total_balls_bag_A) total_balls_bag_A

-- Probability of drawing a white ball from bag B
def P_B := P (λ n => n < white_balls_bag_B) total_balls_bag_B

-- Probability of drawing a red ball from bag B
def P_not_B := P (λ n => n >= white_balls_bag_B && n < total_balls_bag_B) total_balls_bag_B

-- Independence assumption (product rule for independent events)
noncomputable def P_same_color := P_A * P_B + P_not_A * P_not_B

-- Final theorem to prove
theorem probability_of_same_color_balls :
  P_same_color = 1 / 2 := by
    sorry

end probability_of_same_color_balls_l133_133633


namespace number_of_integers_satisfying_inequalities_l133_133005

theorem number_of_integers_satisfying_inequalities :
  ∃ (count : ℕ), count = 3 ∧
    (∀ x : ℤ, -4 * x ≥ x + 10 → -3 * x ≤ 15 → -5 * x ≥ 3 * x + 24 → 2 * x ≤ 18 →
      x = -5 ∨ x = -4 ∨ x = -3) :=
sorry

end number_of_integers_satisfying_inequalities_l133_133005


namespace area_of_quadrilateral_ABFG_l133_133812

/-- 
Given conditions:
1. Rectangle with dimensions AC = 40 and AE = 24.
2. Points B and F are midpoints of sides AC and AE, respectively.
3. G is the midpoint of DE.
Prove that the area of quadrilateral ABFG is 600 square units.
-/
theorem area_of_quadrilateral_ABFG (AC AE : ℝ) (B F G : ℤ) 
  (hAC : AC = 40) (hAE : AE = 24) (hB : B = 1/2 * AC) (hF : F = 1/2 * AE) (hG : G = 1/2 * AE):
  area_of_ABFG = 600 :=
by
  sorry

end area_of_quadrilateral_ABFG_l133_133812


namespace taylor_family_reunion_l133_133966

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l133_133966


namespace train_speed_l133_133959

theorem train_speed (x : ℝ) (v : ℝ) 
  (h1 : (x / 50) + (2 * x / v) = 3 * x / 25) : v = 20 :=
by
  sorry

end train_speed_l133_133959


namespace find_a_if_odd_l133_133036

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem find_a_if_odd (a : ℝ) : (∀ x : ℝ, f (-x) a = -f x a) → a = 0 := by
  intro h
  have h0 : f 0 a = 0 := by
    simp [f]
    specialize h 0
    simp [f] at h
    exact h
  sorry

end find_a_if_odd_l133_133036


namespace complementary_angle_ratio_l133_133783

noncomputable def smaller_angle_measure (x : ℝ) : ℝ := 
  3 * (90 / 7)

theorem complementary_angle_ratio :
  ∀ (A B : ℝ), (B = 4 * (90 / 7)) → (A = 3 * (90 / 7)) → 
  (A + B = 90) → A = 38.57142857142857 :=
by
  intros A B hB hA hSum
  sorry

end complementary_angle_ratio_l133_133783


namespace compound_interest_comparison_l133_133948

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l133_133948


namespace paco_initial_cookies_l133_133082

theorem paco_initial_cookies (x : ℕ) (h : x - 2 + 36 = 2 + 34) : x = 2 :=
by
-- proof steps will be filled in here
sorry

end paco_initial_cookies_l133_133082


namespace people_per_table_l133_133967

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l133_133967


namespace parking_lot_motorcycles_l133_133720

theorem parking_lot_motorcycles :
  ∀ (C M : ℕ), (∀ (n : ℕ), C = 19 ∧ (5 * C + 2 * M = 117) → M = 11) := 
by
  intros C M h
  cases h with hC hWheels
  have hCeq : C = 19 := by sorry
  have hWeq : 5 * 19 + 2 * M = 117 := by sorry
  have hM : M = 11 := by sorry
  exact hM

end parking_lot_motorcycles_l133_133720


namespace find_y_l133_133327

variables (x y : ℝ)

theorem find_y (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 :=
by
  sorry

end find_y_l133_133327


namespace a_b_sum_possible_values_l133_133608

theorem a_b_sum_possible_values (a b : ℝ) 
  (h1 : a^3 - 12 * a^2 + 9 * a - 18 = 0)
  (h2 : 9 * b^3 - 135 * b^2 + 450 * b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 :=
sorry

end a_b_sum_possible_values_l133_133608


namespace triangle_inequality_l133_133739

variable {α β γ a b c: ℝ}

theorem triangle_inequality (h1 : α + β + γ = π)
  (h2 : α > 0) (h3 : β > 0) (h4 : γ > 0)
  (h5 : a > 0) (h6 : b > 0) (h7 : c > 0)
  (h8 : (α > β ∧ a > b) ∨ (α = β ∧ a = b) ∨ (α < β ∧ a < b))
  (h9 : (β > γ ∧ b > c) ∨ (β = γ ∧ b = c) ∨ (β < γ ∧ b < c))
  (h10 : (γ > α ∧ c > a) ∨ (γ = α ∧ c = a) ∨ (γ < α ∧ c < a)) :
  (π / 3) ≤ (a * α + b * β + c * γ) / (a + b + c) ∧
  (a * α + b * β + c * γ) / (a + b + c) < (π / 2) :=
sorry

end triangle_inequality_l133_133739


namespace nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l133_133323

theorem nat_no_solution_x3_plus_5y_eq_y3_plus_5x (x y : ℕ) (h₁ : x ≠ y) : 
  x^3 + 5 * y ≠ y^3 + 5 * x :=
sorry

theorem positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5 * y = y^3 + 5 * x :=
sorry

end nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l133_133323


namespace casey_nail_decorating_time_l133_133297

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l133_133297


namespace find_b_perpendicular_lines_l133_133097

theorem find_b_perpendicular_lines (b : ℚ)
  (line1 : (3 : ℚ) * x + 4 * y - 6 = 0)
  (line2 : b * x + 4 * y - 6 = 0)
  (perpendicular : ( - (3 : ℚ) / 4 ) * ( - (b / 4) ) = -1) :
  b = - (16 : ℚ) / 3 := 
sorry

end find_b_perpendicular_lines_l133_133097


namespace pentagon_rectangle_ratio_l133_133284

theorem pentagon_rectangle_ratio :
  let p : ℝ := 60  -- Perimeter of both the pentagon and the rectangle
  let length_side_pentagon : ℝ := 12
  let w : ℝ := 10
  p / 5 = length_side_pentagon ∧ p/6 = w ∧ length_side_pentagon / w = 6/5 :=
sorry

end pentagon_rectangle_ratio_l133_133284


namespace square_of_sum_opposite_l133_133760

theorem square_of_sum_opposite (a b : ℝ) : (-(a) + b)^2 = (-a + b)^2 :=
by
  sorry

end square_of_sum_opposite_l133_133760


namespace cost_of_mens_t_shirt_l133_133964

-- Definitions based on conditions
def womens_price : ℕ := 18
def womens_interval : ℕ := 30
def mens_interval : ℕ := 40
def shop_open_hours_per_day : ℕ := 12
def total_earnings_per_week : ℕ := 4914

-- Auxiliary definitions based on conditions
def t_shirts_sold_per_hour (interval : ℕ) : ℕ := 60 / interval
def t_shirts_sold_per_day (interval : ℕ) : ℕ := shop_open_hours_per_day * t_shirts_sold_per_hour interval
def t_shirts_sold_per_week (interval : ℕ) : ℕ := t_shirts_sold_per_day interval * 7

def weekly_earnings_womens : ℕ := womens_price * t_shirts_sold_per_week womens_interval
def weekly_earnings_mens : ℕ := total_earnings_per_week - weekly_earnings_womens
def mens_price : ℚ := weekly_earnings_mens / t_shirts_sold_per_week mens_interval

-- The statement to be proved
theorem cost_of_mens_t_shirt : mens_price = 15 := by
  sorry

end cost_of_mens_t_shirt_l133_133964


namespace value_of_a_8_l133_133348

noncomputable def S (n : ℕ) : ℕ := n^2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then S n else S n - S (n - 1)

theorem value_of_a_8 : a 8 = 15 := 
by
  sorry

end value_of_a_8_l133_133348


namespace juice_difference_proof_l133_133911

def barrel_initial_A := 10
def barrel_initial_B := 8
def transfer_amount := 3

def barrel_final_A := barrel_initial_A + transfer_amount
def barrel_final_B := barrel_initial_B - transfer_amount

def juice_difference := barrel_final_A - barrel_final_B

theorem juice_difference_proof : juice_difference = 8 := by
  sorry

end juice_difference_proof_l133_133911


namespace no_a_b_not_divide_bn_minus_n_l133_133679

theorem no_a_b_not_divide_bn_minus_n :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (n : ℕ), 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end no_a_b_not_divide_bn_minus_n_l133_133679


namespace arithmetic_sequence_common_difference_l133_133571

/--
Given an arithmetic sequence $\{a_n\}$ and $S_n$ being the sum of the first $n$ terms, 
with $a_1=1$ and $S_3=9$, prove that the common difference $d$ is equal to $2$.
-/
theorem arithmetic_sequence_common_difference :
  ∃ (d : ℝ), (∀ (n : ℕ), aₙ = 1 + (n - 1) * d) ∧ S₃ = a₁ + (a₁ + d) + (a₁ + 2 * d) ∧ a₁ = 1 ∧ S₃ = 9 → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l133_133571


namespace casey_nail_decorating_time_l133_133295

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l133_133295


namespace max_piles_l133_133769

theorem max_piles (n : ℕ) (hn : n = 660) :
  ∃ (k : ℕ), (∀ (piles : list ℕ),
    (sum piles = n) →
    (∀ (x y : ℕ), x ∈ piles → y ∈ piles → x ≤ 2 * y ∧ y ≤ 2 * x) →
    list.length piles ≤ k) ∧ k = 30 :=
sorry

end max_piles_l133_133769


namespace abc_cubic_sum_identity_l133_133603

theorem abc_cubic_sum_identity (a b c : ℂ) 
  (M : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : M = fun i j => if i = 0 then (if j = 0 then a else if j = 1 then b else c)
                      else if i = 1 then (if j = 0 then b else if j = 1 then c else a)
                      else (if j = 0 then c else if j = 1 then a else b))
  (h2 : M ^ 3 = 1)
  (h3 : a * b * c = -1) :
  a^3 + b^3 + c^3 = 4 := sorry

end abc_cubic_sum_identity_l133_133603


namespace min_value_l133_133127

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2*y = 2) : 
  ∃ c : ℝ, c = 2 ∧ ∀ z, (z = (x^2 / (2*y) + 4*(y^2) / x)) → z ≥ c :=
by
  sorry

end min_value_l133_133127


namespace angle_APB_l133_133444

-- Define the problem conditions
variables (XY : Π X Y : ℝ, XY = X + Y) -- Line XY is a straight line
          (semicircle_XAZ : Π X A Z : ℝ, semicircle_XAZ = X + Z - A) -- Semicircle XAZ
          (semicircle_ZBY : Π Z B Y : ℝ, semicircle_ZBY = Z + Y - B) -- Semicircle ZBY
          (PA_tangent_XAZ_at_A : Π P A X Z : ℝ, PA_tangent_XAZ_at_A = P + A + X - Z) -- PA tangent to XAZ at A
          (PB_tangent_ZBY_at_B : Π P B Z Y : ℝ, PB_tangent_ZBY_at_B = P + B + Z - Y) -- PB tangent to ZBY at B
          (arc_XA : ℝ := 45) -- Arc XA is 45 degrees
          (arc_BY : ℝ := 60) -- Arc BY is 60 degrees

-- Main theorem to prove
theorem angle_APB : ∀ P A B: ℝ, 
  540 - 90 - 135 - 120 - 90 = 105 := by 
  -- Proof goes here
  sorry

end angle_APB_l133_133444


namespace largest_digit_divisible_by_6_l133_133515

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133515


namespace find_complex_z_l133_133687

theorem find_complex_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z / (1 - 2 * i) = i) :
  z = 2 + i :=
sorry

end find_complex_z_l133_133687


namespace water_tank_capacity_l133_133663

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end water_tank_capacity_l133_133663


namespace part1_part2_l133_133576

noncomputable section
open Real

section
variables {x A a b c : ℝ}
variables {k : ℤ}

def f (x : ℝ) : ℝ := sin (2 * x - (π / 6)) + 2 * cos x ^ 2 - 1

theorem part1 (k : ℤ) : 
  ∀ x : ℝ, 
  k * π - (π / 3) ≤ x ∧ x ≤ k * π + (π / 6) → 
    ∀ x₁ x₂, 
      k * π - (π / 3) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + (π / 6) → 
        f x₁ < f x₂ := sorry

theorem part2 {A a b c : ℝ} 
  (h_a_seq : 2 * a = b + c) 
  (h_dot : b * c * cos A = 9) 
  (h_A_fA : f A = 1 / 2) 
  : 
  a = 3 * sqrt 2 := sorry

end

end part1_part2_l133_133576


namespace kristin_runs_around_l133_133736

-- Definitions of the conditions.
def kristin_runs_faster (v_k v_s : ℝ) : Prop := v_k = 3 * v_s
def sarith_runs_times (S : ℕ) : Prop := S = 8
def field_length (c_field a_field : ℝ) : Prop := c_field = a_field / 2

-- The question is to prove Kristin runs around the field 12 times.
def kristin_runs_times (K : ℕ) : Prop := K = 12

-- The main theorem statement combining conditions to prove the question.
theorem kristin_runs_around :
  ∀ (v_k v_s c_field a_field : ℝ) (S K : ℕ),
    kristin_runs_faster v_k v_s →
    sarith_runs_times S →
    field_length c_field a_field →
    K = (S : ℝ) * (3 / 2) →
    kristin_runs_times K :=
by sorry

end kristin_runs_around_l133_133736


namespace janet_total_pills_l133_133730

-- Define number of days per week
def days_per_week : ℕ := 7

-- Define pills per day for each week
def pills_first_2_weeks :=
  let multivitamins := 2 * days_per_week * 2
  let calcium := 3 * days_per_week * 2
  let magnesium := 5 * days_per_week * 2
  multivitamins + calcium + magnesium

def pills_third_week :=
  let multivitamins := 2 * days_per_week
  let calcium := 1 * days_per_week
  let magnesium := 0 * days_per_week
  multivitamins + calcium + magnesium

def pills_fourth_week :=
  let multivitamins := 3 * days_per_week
  let calcium := 2 * days_per_week
  let magnesium := 3 * days_per_week
  multivitamins + calcium + magnesium

def total_pills := pills_first_2_weeks + pills_third_week + pills_fourth_week

theorem janet_total_pills : total_pills = 245 := by
  -- Lean will generate a proof goal here with the left-hand side of the equation
  -- equal to an evaluated term, and we say that this equals 245 based on the problem's solution.
  sorry

end janet_total_pills_l133_133730


namespace find_sum_of_smallest_multiples_l133_133184

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l133_133184


namespace arithmetic_sequence_ratio_l133_133163

variable {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variable (d : ℤ)
variable (a1 a3 a4 : ℤ)
variable (h_geom : a3^2 = a1 * a4)
variable (h_seq : ∀ n, a_n (n+1) = a_n n + d)
variable (h_sum : ∀ n, S_n n = (n * (2 * a1 + (n - 1) * d)) / 2)

theorem arithmetic_sequence_ratio :
  (S_n 3 - S_n 2) / (S_n 5 - S_n 3) = 2 :=
by 
  sorry

end arithmetic_sequence_ratio_l133_133163


namespace students_not_picked_correct_l133_133252

-- Define the total number of students and the number of students picked for the team
def total_students := 17
def students_picked := 3 * 4

-- Define the number of students who didn't get picked based on the conditions
noncomputable def students_not_picked : ℕ := total_students - students_picked

-- The theorem stating the problem
theorem students_not_picked_correct : students_not_picked = 5 := 
by 
  sorry

end students_not_picked_correct_l133_133252


namespace fruit_vendor_sold_fruits_l133_133884

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l133_133884


namespace train_passes_jogger_in_time_l133_133810

def jogger_speed_kmh : ℝ := 8
def train_speed_kmh : ℝ := 60
def initial_distance_m : ℝ := 360
def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m
noncomputable def passing_time_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_time :
  passing_time_s = 38.75 := by
  sorry

end train_passes_jogger_in_time_l133_133810


namespace time_for_b_and_d_together_l133_133275

theorem time_for_b_and_d_together :
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  (∃ B_rate C_rate : ℚ,
    B_rate + C_rate = 1 / 3 ∧
    A_rate + C_rate = 1 / 2 ∧
    1 / (B_rate + D_rate) = 2.4) :=
  
by
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  use 1 / 6, 1 / 6
  sorry

end time_for_b_and_d_together_l133_133275


namespace perimeter_rectangles_l133_133686

theorem perimeter_rectangles (a b : ℕ) (p_rect1 p_rect2 : ℕ) (p_photo : ℕ) (h1 : 2 * (a + b) = p_photo) (h2 : a + b = 10) (h3 : p_rect1 = 40) (h4 : p_rect2 = 44) : 
p_rect1 ≠ p_rect2 -> (p_rect1 = 40 ∧ p_rect2 = 44) := 
by 
  sorry

end perimeter_rectangles_l133_133686


namespace probability_more_than_60000_l133_133218

def boxes : List ℕ := [8, 800, 8000, 40000, 80000]

def probability_keys (keys : ℕ) : ℚ :=
  1 / keys

def probability_winning (n : ℕ) : ℚ :=
  if n = 4 then probability_keys 5 + probability_keys 5 * probability_keys 4 else 0

theorem probability_more_than_60000 : 
  probability_winning 4 = 1/4 := sorry

end probability_more_than_60000_l133_133218


namespace carol_initial_cupcakes_l133_133847

/--
For the school bake sale, Carol made some cupcakes. She sold 9 of them and then made 28 more.
Carol had 49 cupcakes. We need to show that Carol made 30 cupcakes initially.
-/
theorem carol_initial_cupcakes (x : ℕ) 
  (h1 : x - 9 + 28 = 49) : 
  x = 30 :=
by 
  -- The proof is not required as per instruction.
  sorry

end carol_initial_cupcakes_l133_133847


namespace mark_money_l133_133208

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l133_133208


namespace cone_volume_l133_133503

theorem cone_volume (V_cyl : ℝ) (d : ℝ) (π : ℝ) (V_cyl_eq : V_cyl = 81 * π) (h_eq : 2 * (d / 2) = 2 * d) :
  ∃ (V_cone : ℝ), V_cone = 27 * π * (6 ^ (1/3)) :=
by 
  sorry

end cone_volume_l133_133503


namespace max_piles_660_stones_l133_133779

-- Define the conditions in Lean
def initial_stones := 660

def valid_pile_sizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → a ≤ b → b < 2 * a

-- Define the goal statement in Lean
theorem max_piles_660_stones :
  ∃ (piles : List ℕ), (piles.length = 30) ∧ (piles.sum = initial_stones) ∧ valid_pile_sizes piles :=
sorry

end max_piles_660_stones_l133_133779


namespace find_inverse_sum_l133_133230

variable {R : Type*} [OrderedRing R]

-- Define the function f and its inverse
variable (f : R → R)
variable (f_inv : R → R)

-- Conditions
axiom f_inverse : ∀ y, f (f_inv y) = y
axiom f_prop : ∀ x, f x + f (1 - x) = 2

-- The theorem we need to prove
theorem find_inverse_sum (x : R) : f_inv (x - 2) + f_inv (4 - x) = 1 :=
by
  sorry

end find_inverse_sum_l133_133230


namespace donuts_per_student_l133_133467

theorem donuts_per_student 
    (dozens_of_donuts : ℕ)
    (students_in_class : ℕ)
    (percentage_likes_donuts : ℕ)
    (students_who_like_donuts : ℕ)
    (total_donuts : ℕ)
    (donuts_per_student : ℕ) :
    dozens_of_donuts = 4 →
    students_in_class = 30 →
    percentage_likes_donuts = 80 →
    students_who_like_donuts = (percentage_likes_donuts * students_in_class) / 100 →
    total_donuts = dozens_of_donuts * 12 →
    donuts_per_student = total_donuts / students_who_like_donuts →
    donuts_per_student = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end donuts_per_student_l133_133467


namespace initial_mean_corrected_l133_133242

theorem initial_mean_corrected (M : ℝ) (H : 30 * M + 30 = 30 * 151) : M = 150 :=
sorry

end initial_mean_corrected_l133_133242


namespace maximum_revenue_l133_133953

def ticket_price (x : ℕ) (y : ℤ) : Prop :=
  (6 ≤ x ∧ x ≤ 10 ∧ y = 1000 * x - 5750) ∨
  (10 < x ∧ x ≤ 38 ∧ y = -30 * x^2 + 1300 * x - 5750)

theorem maximum_revenue :
  ∃ x y, ticket_price x y ∧ y = 8830 ∧ x = 22 :=
by {
  sorry
}

end maximum_revenue_l133_133953


namespace frog_reaches_seven_l133_133877

theorem frog_reaches_seven (p q : ℕ) (h_rel_prime : Nat.coprime p q) (h_complete_probability : p = 43 ∧ q = 64) :
  p + q = 107 :=
by
  sorry

end frog_reaches_seven_l133_133877


namespace value_range_of_m_for_equation_l133_133502

theorem value_range_of_m_for_equation 
    (x : ℝ) 
    (cos_x : ℝ) 
    (h1: cos_x = Real.cos x) :
    ∃ (m : ℝ), (0 ≤ m ∧ m ≤ 8) ∧ (4 * cos_x + Real.sin x ^ 2 + m - 4 = 0) := sorry

end value_range_of_m_for_equation_l133_133502


namespace sum_of_digits_smallest_N_l133_133605

/-- Define the probability Q(N) -/
def Q (N : ℕ) : ℚ :=
  ((2 * N) / 3 + 1) / (N + 1)

/-- Main mathematical statement to be proven in Lean 4 -/

theorem sum_of_digits_smallest_N (N : ℕ) (h1 : N > 9) (h2 : N % 6 = 0) (h3 : Q N < 7 / 10) : 
  (N.digits 10).sum = 3 :=
  sorry

end sum_of_digits_smallest_N_l133_133605


namespace tangent_line_of_cubic_at_l133_133758

theorem tangent_line_of_cubic_at (x y : ℝ) (h : y = x^3) (hx : x = 1) (hy : y = 1) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_of_cubic_at_l133_133758


namespace proof_problem_l133_133626

noncomputable def h : Polynomial ℝ := Polynomial.X^3 - Polynomial.X^2 - 4 * Polynomial.X + 4
noncomputable def p : Polynomial ℝ := Polynomial.X^3 + 12 * Polynomial.X^2 - 13 * Polynomial.X - 64

theorem proof_problem : 
  (∀ x : ℝ, h.eval x = 0 → p.eval (x^3) = 0) ∧ 
  (∀ a b c : ℝ, (p = Polynomial.X^3 + a * Polynomial.X^2 + b * Polynomial.X + c) → 
  ((a, b, c) = (12, -13, -64))) :=
sorry

end proof_problem_l133_133626


namespace min_balls_to_guarantee_18_l133_133537

noncomputable def min_balls_needed {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) : ℕ :=
  95

theorem min_balls_to_guarantee_18 {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) :
  min_balls_needed h_red h_green h_yellow h_blue h_white h_black = 95 :=
  by
  -- Placeholder for the actual proof
  sorry

end min_balls_to_guarantee_18_l133_133537


namespace taylor_family_reunion_l133_133965

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l133_133965


namespace markup_percentage_l133_133394

theorem markup_percentage (PP SP SaleP : ℝ) (M : ℝ) (hPP : PP = 60) (h1 : SP = 60 + M * SP)
  (h2 : SaleP = SP * 0.8) (h3 : 4 = SaleP - PP) : M = 0.25 :=
by 
  sorry

end markup_percentage_l133_133394


namespace sum_of_digits_palindrome_l133_133132

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end sum_of_digits_palindrome_l133_133132


namespace expected_rolls_of_six_correct_l133_133084

noncomputable def die_rolls : ℕ := 100
noncomputable def probability_of_six : ℚ := 1 / 6
noncomputable def expected_rolls_of_six : ℚ := 50 / 3

theorem expected_rolls_of_six_correct :
  let X := ProbabilityMassFunction.binomial die_rolls probability_of_six in
  ProbabilityMassFunction.expectedValue X = expected_rolls_of_six :=
sorry

end expected_rolls_of_six_correct_l133_133084


namespace find_constant_l133_133585

theorem find_constant (c : ℝ) (f : ℝ → ℝ)
  (h : f x = c * x^3 + 19 * x^2 - 4 * c * x + 20)
  (hx : f (-7) = 0) :
  c = 3 :=
sorry

end find_constant_l133_133585


namespace main_problem_l133_133176

-- Defining the function f(x) given a
def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

-- Tangent condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (1 / x₀ - a = 1) ∧ (x₀ - 1 - log 2 = log x₀ - a * x₀)

-- Inequality condition
def holds_inequality (a : ℝ) : Prop :=
  ∀ x > 0, (x + 1) * (log x - a * x) ≤ log x - (x / exp 1)

-- Main theorem combining both parts
theorem main_problem :
  (∃ a : ℝ, is_tangent a ∧ holds_inequality a) ↔
  ∃ a : ℝ, a = 1 ∧ a ∈ (set.Ici (1 / exp 1)) :=
by sorry

end main_problem_l133_133176


namespace max_value_of_expression_l133_133533

theorem max_value_of_expression (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) : x^2 + y^2 ≤ 10 :=
sorry

end max_value_of_expression_l133_133533


namespace differentiable_increasing_necessary_but_not_sufficient_l133_133869

variable {f : ℝ → ℝ}

theorem differentiable_increasing_necessary_but_not_sufficient (h_diff : ∀ x : ℝ, DifferentiableAt ℝ f x) :
  (∀ x : ℝ, 0 < deriv f x) → ∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) ∧ ¬ (∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) → ∀ x : ℝ, 0 < deriv f x) := 
sorry

end differentiable_increasing_necessary_but_not_sufficient_l133_133869


namespace calculation_is_correct_l133_133925

-- Define the numbers involved in the calculation
def a : ℝ := 12.05
def b : ℝ := 5.4
def c : ℝ := 0.6

-- Expected result of the calculation
def expected_result : ℝ := 65.67

-- Prove that the calculation is correct
theorem calculation_is_correct : (a * b + c) = expected_result :=
by
  sorry

end calculation_is_correct_l133_133925


namespace length_of_BC_l133_133873

theorem length_of_BC (AB AC AM : ℝ) (hAB : AB = 5) (hAC : AC = 8) (hAM : AM = 4.5) : 
  ∃ BC, BC = Real.sqrt 97 :=
by
  sorry

end length_of_BC_l133_133873


namespace students_divided_into_groups_l133_133370

theorem students_divided_into_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) (n_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : not_picked = 36) 
  (h3 : students_per_group = 7) 
  (h4 : total_students - not_picked = 28) 
  (h5 : 28 / students_per_group = 4) :
  n_groups = 4 :=
by
  sorry

end students_divided_into_groups_l133_133370


namespace final_price_after_discounts_l133_133386

theorem final_price_after_discounts (m : ℝ) : (0.8 * m - 10) = selling_price :=
by
  sorry

end final_price_after_discounts_l133_133386


namespace second_discarded_number_l133_133093

theorem second_discarded_number (S : ℝ) (X : ℝ) :
  (S = 50 * 44) →
  ((S - 45 - X) / 48 = 43.75) →
  X = 55 :=
by
  intros h1 h2
  -- The proof steps would go here, but we leave it unproved
  sorry

end second_discarded_number_l133_133093


namespace probability_all_correct_l133_133935

noncomputable def probability_mcq : ℚ := 1 / 3
noncomputable def probability_true_false : ℚ := 1 / 2

theorem probability_all_correct :
  (probability_mcq * probability_true_false * probability_true_false) = (1 / 12) :=
by
  sorry

end probability_all_correct_l133_133935


namespace count_multiples_3_or_4_but_not_6_l133_133711

def multiples_between (m n k : Nat) : Nat :=
  (k / m) + (k / n) - (k / (m * n))

theorem count_multiples_3_or_4_but_not_6 :
  let count_multiples (d : Nat) := (3000 / d)
  let multiples_of_3 := count_multiples 3
  let multiples_of_4 := count_multiples 4
  let multiples_of_6 := count_multiples 6
  multiples_of_3 + multiples_of_4 - multiples_of_6 = 1250 := by
  sorry

end count_multiples_3_or_4_but_not_6_l133_133711


namespace problem_min_x_plus_2y_l133_133031

theorem problem_min_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) : 
  x + 2 * y ≥ -2 * Real.sqrt 2 - 1 :=
sorry

end problem_min_x_plus_2y_l133_133031


namespace geometric_sequence_sum_l133_133440

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_common_ratio : ∀ n, a (n + 1) = 2 * a n)
    (h_sum : a 1 + a 2 + a 3 = 21) : a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l133_133440


namespace largest_digit_divisible_by_6_l133_133524

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l133_133524


namespace baking_powder_difference_l133_133274

-- Define the known quantities
def baking_powder_yesterday : ℝ := 0.4
def baking_powder_now : ℝ := 0.3

-- Define the statement to prove, i.e., the difference in baking powder
theorem baking_powder_difference : baking_powder_yesterday - baking_powder_now = 0.1 :=
by
  -- Proof omitted
  sorry

end baking_powder_difference_l133_133274


namespace seafoam_azure_ratio_l133_133473

-- Define the conditions
variables (P S A : ℕ) 

-- Purple Valley has one-quarter as many skirts as Seafoam Valley
axiom h1 : P = S / 4

-- Azure Valley has 60 skirts
axiom h2 : A = 60

-- Purple Valley has 10 skirts
axiom h3 : P = 10

-- The goal is to prove the ratio of Seafoam Valley skirts to Azure Valley skirts is 2 to 3
theorem seafoam_azure_ratio : S / A = 2 / 3 :=
by 
  sorry

end seafoam_azure_ratio_l133_133473


namespace partition_no_infinite_arith_prog_l133_133196

theorem partition_no_infinite_arith_prog :
  ∃ (A B : Set ℕ), 
  (∀ n ∈ A, n ∈ B → False) ∧ 
  (∀ (a b : ℕ) (d : ℕ), (a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b) % d = 0) → False) ∧
  (∀ (a b : ℕ) (d : ℕ), (a ∈ B ∧ b ∈ B ∧ a ≠ b ∧ (a - b) % d = 0) → False) :=
sorry

end partition_no_infinite_arith_prog_l133_133196


namespace no_solution_m_l133_133716

theorem no_solution_m {
  m : ℚ
  } (h : ∀ x : ℚ, x ≠ 3 → (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) ≠ -1) : 
  m = 1 ∨ m = 5 / 3 :=
sorry

end no_solution_m_l133_133716


namespace oak_trees_remaining_is_7_l133_133506

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7_l133_133506


namespace cos_270_eq_zero_l133_133977

theorem cos_270_eq_zero : Real.cos (270 * (π / 180)) = 0 :=
by
  -- Conditions given in the problem
  have rotation_def : ∀ (θ : ℝ), ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = Real.cos θ ∧ y = Real.sin θ := by
    intros θ
    use [Real.cos θ, Real.sin θ]
    split
    · exact Real.cos_sq_add_sin_sq θ
    split
    · rfl
    · rfl

  have rotation_270 : ∃ (x y : ℝ), x = 0 ∧ y = -1 := by
    use [0, -1]
    split
    · rfl
    · rfl

  -- Goal to be proved
  have result := Real.cos (270 * (π / 180))
  show result = 0
  sorry

end cos_270_eq_zero_l133_133977


namespace binary_division_correct_l133_133307

def b1100101 := 0b1100101
def b1101 := 0b1101
def b101 := 0b101
def expected_result := 0b11111010

theorem binary_division_correct : ((b1100101 * b1101) / b101) = expected_result :=
by {
  sorry
}

end binary_division_correct_l133_133307


namespace system1_solution_system2_solution_l133_133361

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 4 * x + 8 * y = 12) (h2 : 3 * x - 2 * y = 5) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : (1/2) * x - (y + 1) / 3 = 1) (h2 : 6 * x + 2 * y = 10) :
  x = 2 ∧ y = -1 := by
  sorry

end system1_solution_system2_solution_l133_133361


namespace mean_daily_profit_l133_133759

theorem mean_daily_profit 
  (mean_first_15_days : ℝ) 
  (mean_last_15_days : ℝ) 
  (n : ℝ) 
  (m1_days : ℝ) 
  (m2_days : ℝ) : 
  (mean_first_15_days = 245) → 
  (mean_last_15_days = 455) → 
  (m1_days = 15) → 
  (m2_days = 15) → 
  (n = 30) →
  (∀ P, P = (245 * 15 + 455 * 15) / 30) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mean_daily_profit_l133_133759


namespace max_ab_perpendicular_l133_133044

theorem max_ab_perpendicular (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + b = 3) : ab <= (9 / 8) := 
sorry

end max_ab_perpendicular_l133_133044


namespace fraction_simplifies_l133_133673

theorem fraction_simplifies :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end fraction_simplifies_l133_133673


namespace b_work_time_l133_133121

theorem b_work_time (W : ℝ) (days_A days_combined : ℝ)
  (hA : W / days_A = W / 16)
  (h_combined : W / days_combined = W / (16 / 3)) :
  ∃ days_B, days_B = 8 :=
by
  sorry

end b_work_time_l133_133121


namespace greatest_3_digit_base9_divisible_by_7_l133_133108

theorem greatest_3_digit_base9_divisible_by_7 :
  ∃ (n : ℕ), n < 729 ∧ n ≥ 81 ∧ n % 7 = 0 ∧ n = 8 * 81 + 8 * 9 + 8 := 
by 
  use 728
  split
  {
    exact nat.pred_lt (ne_of_lt (by norm_num))
  }
  split
  {
    exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.zero_le 7))) 
  }
  split
  {
    norm_num
  }
  norm_num

end greatest_3_digit_base9_divisible_by_7_l133_133108


namespace triangle_formation_l133_133668

-- Problem interpretation and necessary definitions
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def stick1 : ℕ := 4
def stick2 : ℕ := 9
def options : List ℕ := [4, 5, 9, 14]
def answer : ℕ := 9

-- The proof problem
theorem triangle_formation : can_form_triangle stick1 stick2 answer :=
by
  -- Utilizing the triangle inequality theorem to validate the formation
  unfold can_form_triangle
  split
  -- The constraints for the side lengths will follow as stated in the proof problem.
  { sorry }

end triangle_formation_l133_133668


namespace cos_double_angle_zero_l133_133708

theorem cos_double_angle_zero
  (θ : ℝ)
  (a : ℝ×ℝ := (1, -Real.cos θ))
  (b : ℝ×ℝ := (1, 2 * Real.cos θ))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.cos (2 * θ) = 0 :=
by sorry

end cos_double_angle_zero_l133_133708


namespace carnations_count_l133_133546

-- Define the conditions 
def vase_capacity : Nat := 9
def number_of_vases : Nat := 3
def number_of_roses : Nat := 23
def total_flowers : Nat := number_of_vases * vase_capacity

-- Define the number of carnations
def number_of_carnations : Nat := total_flowers - number_of_roses

-- Assertion that should be proved
theorem carnations_count : number_of_carnations = 4 := by
  sorry

end carnations_count_l133_133546


namespace prime_not_fourth_power_l133_133011

theorem prime_not_fourth_power (p : ℕ) (hp : p > 5) (prime : Prime p) : 
  ¬ ∃ a : ℕ, p = a^4 + 4 :=
by
  sorry

end prime_not_fourth_power_l133_133011


namespace least_number_subtracted_divisible_by_six_l133_133791

theorem least_number_subtracted_divisible_by_six :
  ∃ d : ℕ, d = 6 ∧ (427398 - 6) % d = 0 := by
sorry

end least_number_subtracted_divisible_by_six_l133_133791


namespace relationship_m_n_l133_133028

variables {a b : ℝ}

theorem relationship_m_n (h1 : |a| ≠ |b|) (m : ℝ) (n : ℝ)
  (hm : m = (|a| - |b|) / |a - b|)
  (hn : n = (|a| + |b|) / |a + b|) :
  m ≤ n :=
by sorry

end relationship_m_n_l133_133028


namespace seating_arrangement_l133_133354

-- Define the problem in Lean
theorem seating_arrangement :
  let n := 9   -- Total number of people
  let r := 7   -- Number of seats at the circular table
  let combinations := Nat.choose n 2  -- Ways to select 2 people not seated
  let factorial (k : ℕ) := Nat.recOn k 1 (λ k' acc => (k' + 1) * acc)
  let arrangements := factorial (r - 1)  -- Ways to seat 7 people around a circular table
  combinations * arrangements = 25920 :=
by
  -- In Lean, sorry is used to indicate that we skip the proof for now.
  sorry

end seating_arrangement_l133_133354


namespace parrot_arrangement_l133_133192

theorem parrot_arrangement : 
  ∃ arrangements : Finset (Perm (Fin 8)), 
  (∀ σ ∈ arrangements, (σ 0 = 0 ∨ σ 0 = 1) ∧ (σ 7 = 0 ∨ σ 7 = 1) ∧ σ 3 = 7) ∧ 
  arrangements.card = 240 :=
sorry

end parrot_arrangement_l133_133192


namespace algebraic_identity_l133_133375

theorem algebraic_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

example : (2011 : ℝ)^2 - (2010 : ℝ)^2 = 4021 := 
by
  have h := algebraic_identity 2011 2010
  rw [h]
  norm_num

end algebraic_identity_l133_133375


namespace necessary_but_not_sufficient_ellipse_l133_133598

def is_ellipse (m : ℝ) : Prop := 
  1 < m ∧ m < 3 ∧ m ≠ 2

theorem necessary_but_not_sufficient_ellipse (m : ℝ) :
  (1 < m ∧ m < 3) → (m ≠ 2) → is_ellipse m :=
by
  intros h₁ h₂
  have h : 1 < m ∧ m < 3 ∧ m ≠ 2 := ⟨h₁.left, h₁.right, h₂⟩
  exact h

end necessary_but_not_sufficient_ellipse_l133_133598


namespace scientific_notation_110_billion_l133_133547

theorem scientific_notation_110_billion :
  ∃ (n : ℝ) (e : ℤ), 110000000000 = n * 10 ^ e ∧ 1 ≤ n ∧ n < 10 ∧ n = 1.1 ∧ e = 11 :=
by
  sorry

end scientific_notation_110_billion_l133_133547


namespace max_product_of_sum_2020_l133_133114

theorem max_product_of_sum_2020 : 
  ∃ x y : ℤ, x + y = 2020 ∧ (x * y) ≤ 1020100 ∧ (∀ a b : ℤ, a + b = 2020 → a * b ≤ x * y) :=
begin
  sorry
end

end max_product_of_sum_2020_l133_133114


namespace number_of_whole_numbers_in_intervals_l133_133431

theorem number_of_whole_numbers_in_intervals : 
  let interval_start := (5 / 3 : ℝ)
  let interval_end := 2 * Real.pi
  ∃ n : ℕ, interval_start < ↑n ∧ ↑n < interval_end ∧ (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) ∧ 
  (∀ m : ℕ, interval_start < ↑m ∧ ↑m < interval_end → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6)) :=
sorry

end number_of_whole_numbers_in_intervals_l133_133431


namespace KellyGamesLeft_l133_133452

def initialGames : ℕ := 121
def gamesGivenAway : ℕ := 99

theorem KellyGamesLeft : initialGames - gamesGivenAway = 22 := by
  sorry

end KellyGamesLeft_l133_133452


namespace set_intersection_l133_133743

open Set

def U : Set ℤ := univ
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {-1, 1}
def C_U_B : Set ℤ := U \ B

theorem set_intersection :
  A ∩ C_U_B = {2} := 
by
  sorry

end set_intersection_l133_133743


namespace sum_of_first_12_terms_geometric_sequence_l133_133322

variable {α : Type*} [Field α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (Finset.range n).sum a

theorem sum_of_first_12_terms_geometric_sequence
  (a : ℕ → α)
  (h_geo : geometric_sequence a)
  (h_sum1 : sum_first_n_terms a 3 = 4)
  (h_sum2 : sum_first_n_terms a 6 - sum_first_n_terms a 3 = 8) :
  sum_first_n_terms a 12 = 60 := 
sorry

end sum_of_first_12_terms_geometric_sequence_l133_133322


namespace euler_totient_inequality_l133_133741

variable {n : ℕ}
def even (n : ℕ) := ∃ k : ℕ, n = 2 * k
def positive (n : ℕ) := n > 0

theorem euler_totient_inequality (h_even : even n) (h_positive : positive n) : 
  Nat.totient n ≤ n / 2 :=
sorry

end euler_totient_inequality_l133_133741


namespace blocks_fit_into_box_l133_133808

theorem blocks_fit_into_box :
  let box_height := 8
  let box_width := 10
  let box_length := 12
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let box_volume := box_height * box_width * box_length
  let block_volume := block_height * block_width * block_length
  let num_blocks := box_volume / block_volume
  num_blocks = 40 :=
by
  sorry

end blocks_fit_into_box_l133_133808


namespace cost_of_tax_free_items_l133_133796

/-- 
Daniel went to a shop and bought items worth Rs 25, including a 30 paise sales tax on taxable items
with a tax rate of 10%. Prove that the cost of tax-free items is Rs 22.
-/
theorem cost_of_tax_free_items (total_spent taxable_amount sales_tax rate : ℝ)
  (h1 : total_spent = 25)
  (h2 : sales_tax = 0.3)
  (h3 : rate = 0.1)
  (h4 : taxable_amount = sales_tax / rate) :
  (total_spent - taxable_amount = 22) :=
by
  sorry

end cost_of_tax_free_items_l133_133796


namespace root_reciprocals_identity_l133_133299

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  (a + b + c = 12) ∧ (a * b + b * c + c * a = 20) ∧ (a * b * c = -5)

theorem root_reciprocals_identity (a b c : ℝ) (h : cubic_roots a b c) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 20.8 :=
by
  sorry

end root_reciprocals_identity_l133_133299


namespace difference_of_integers_l133_133356

theorem difference_of_integers :
  ∀ (x y : ℤ), (x = 32) → (y = 5*x + 2) → (y - x = 130) :=
by
  intros x y hx hy
  sorry

end difference_of_integers_l133_133356


namespace possible_values_of_n_l133_133881

theorem possible_values_of_n (n : ℕ) (h_pos : 0 < n) (h_prime_n : Nat.Prime n) (h_prime_double_sub1 : Nat.Prime (2 * n - 1)) (h_prime_quad_sub1 : Nat.Prime (4 * n - 1)) :
  n = 2 ∨ n = 3 :=
by
  sorry

end possible_values_of_n_l133_133881


namespace farm_needs_horse_food_per_day_l133_133549

-- Definition of conditions
def ratio_sheep_to_horses := 4 / 7
def food_per_horse := 230
def number_of_sheep := 32

-- Number of horses based on ratio
def number_of_horses := (number_of_sheep * 7) / 4

-- Proof Statement
theorem farm_needs_horse_food_per_day :
  (number_of_horses * food_per_horse) = 12880 :=
by
  -- skipping the proof steps
  sorry

end farm_needs_horse_food_per_day_l133_133549


namespace cookie_distribution_l133_133103

theorem cookie_distribution (b m l : ℕ)
  (h1 : b + m + l = 30)
  (h2 : m = 2 * b)
  (h3 : l = b + m) :
  b = 5 ∧ m = 10 ∧ l = 15 := 
by 
  sorry

end cookie_distribution_l133_133103


namespace pentagon_area_pq_sum_l133_133460

theorem pentagon_area_pq_sum 
  (p q : ℤ) 
  (hp : 0 < q ∧ q < p) 
  (harea : 5 * p * q - q * q = 700) : 
  ∃ sum : ℤ, sum = p + q :=
by
  sorry

end pentagon_area_pq_sum_l133_133460


namespace inhabitant_50th_statement_l133_133984

-- Definition of inhabitant types
inductive InhabitantType
| Knight
| Liar

-- Predicate for the statement of inhabitants
def says (inhabitant : InhabitantType) (statement : InhabitantType) : Bool :=
  match inhabitant with
  | InhabitantType.Knight => true
  | InhabitantType.Liar => false

-- Conditions from the problem
axiom inhabitants : Fin 50 → InhabitantType
axiom statements : ∀ i : Fin 50, i.val % 2 = 0 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Liar = false)
axiom statements' : ∀ i : Fin 50, i.val % 2 = 1 → (says (inhabitants ((i + 1) % 50)) InhabitantType.Knight = true)

-- Goal to prove
theorem inhabitant_50th_statement : says (inhabitants 49) InhabitantType.Knight := by
  sorry

end inhabitant_50th_statement_l133_133984


namespace abs_neg_2023_l133_133232

theorem abs_neg_2023 : |(-2023)| = 2023 :=
by
  sorry

end abs_neg_2023_l133_133232


namespace subset_range_l133_133535

open Set

-- Definitions of sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- The statement of the problem
theorem subset_range (a : ℝ) (h : A ⊆ B a) : 2 ≤ a :=
sorry -- Skipping the proof

end subset_range_l133_133535


namespace complex_inequality_l133_133047

theorem complex_inequality (m : ℝ) : 
  (m - 3 ≥ 0 ∧ m^2 - 9 = 0) → m = 3 := 
by
  sorry

end complex_inequality_l133_133047


namespace exactly_one_red_probability_l133_133806

open Finset

/-- Given a bag containing 2 red balls and 2 white balls,
    the probability of drawing exactly one red ball when 2 balls are drawn at random
    is 2/3.
-/
noncomputable def probability_exactly_one_red : ℚ :=
  let red_draws := 2
  let white_draws := 2
  let total_draws := red_draws + white_draws
  let draw_two := (total_draws.choose 2)
  let favorable_draws := (red_draws.choose 1) * (white_draws.choose 1)
  let probability := favorable_draws / draw_two
  probability

theorem exactly_one_red_probability :
  probability_exactly_one_red = 2/3 :=
by
  unfold probability_exactly_one_red
  sorry

end exactly_one_red_probability_l133_133806


namespace solve_basketball_points_l133_133437

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points_l133_133437


namespace compare_compound_interest_l133_133950

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l133_133950


namespace mark_initial_money_l133_133209

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end mark_initial_money_l133_133209


namespace greatest_product_l133_133112

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l133_133112


namespace mul_582964_99999_l133_133530

theorem mul_582964_99999 : 582964 * 99999 = 58295817036 := by
  sorry

end mul_582964_99999_l133_133530


namespace smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l133_133685

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f_is_pi : 
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ ε > 0, ε < Real.pi → ∃ x, f (x + ε) ≠ f x) :=
by
  sorry

theorem f_at_pi_over_2_not_sqrt_3_over_2 : f (Real.pi / 2) ≠ Real.sqrt 3 / 2 :=
by
  sorry

theorem max_value_of_f_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 6 → f x ≤ 1 :=
by
  sorry

end smallest_positive_period_of_f_is_pi_f_at_pi_over_2_not_sqrt_3_over_2_max_value_of_f_on_interval_l133_133685


namespace unique_xy_exists_l133_133480

theorem unique_xy_exists (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y) ^ 2 + 3 * x + y) / 2 := 
sorry

end unique_xy_exists_l133_133480


namespace smallest_n_square_smallest_n_cube_l133_133118

theorem smallest_n_square (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 2) ↔ n = 3 := 
by sorry

theorem smallest_n_cube (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 3) ↔ n = 2 := 
by sorry

end smallest_n_square_smallest_n_cube_l133_133118


namespace hawkeye_remaining_balance_l133_133581

theorem hawkeye_remaining_balance
  (cost_per_charge : ℝ) (number_of_charges : ℕ) (initial_budget : ℝ) : 
  cost_per_charge = 3.5 → number_of_charges = 4 → initial_budget = 20 → 
  initial_budget - (number_of_charges * cost_per_charge) = 6 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end hawkeye_remaining_balance_l133_133581


namespace area_sin_transformed_l133_133738

noncomputable def sin_transformed (x : ℝ) : ℝ := 4 * Real.sin (x - Real.pi)

theorem area_sin_transformed :
  ∫ x in Real.pi..3 * Real.pi, |sin_transformed x| = 16 :=
by
  sorry

end area_sin_transformed_l133_133738


namespace measure_of_angle_Q_l133_133338

-- Given conditions
variables (α β γ δ : ℝ)
axiom h1 : α = 130
axiom h2 : β = 95
axiom h3 : γ = 110
axiom h4 : δ = 104

-- Statement of the problem
theorem measure_of_angle_Q (Q : ℝ) (h5 : Q + α + β + γ + δ = 540) : Q = 101 := 
sorry

end measure_of_angle_Q_l133_133338


namespace no_such_a_and_sequence_exists_l133_133839

theorem no_such_a_and_sequence_exists :
  ¬∃ (a : ℝ) (a_pos : 0 < a ∧ a < 1) (a_seq : ℕ → ℝ), (∀ n : ℕ, 0 < a_seq n) ∧ (∀ n : ℕ, 1 + a_seq (n + 1) ≤ a_seq n + (a / (n + 1)) * a_seq n) :=
by
  sorry

end no_such_a_and_sequence_exists_l133_133839


namespace largest_digit_divisible_by_6_l133_133514

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133514


namespace sum_mod_13_l133_133266

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l133_133266


namespace martha_correct_guess_probability_l133_133609

namespace MarthaGuess

-- Definitions for the conditions
def height_guess_child_accurate : ℚ := 4 / 5
def height_guess_adult_accurate : ℚ := 5 / 6
def weight_guess_tight_clothing_accurate : ℚ := 3 / 4
def weight_guess_loose_clothing_accurate : ℚ := 7 / 10

-- Probabilities of incorrect guesses
def height_guess_child_inaccurate : ℚ := 1 - height_guess_child_accurate
def height_guess_adult_inaccurate : ℚ := 1 - height_guess_adult_accurate
def weight_guess_tight_clothing_inaccurate : ℚ := 1 - weight_guess_tight_clothing_accurate
def weight_guess_loose_clothing_inaccurate : ℚ := 1 - weight_guess_loose_clothing_accurate

-- Combined probability of guessing incorrectly for each case
def incorrect_prob_child_loose : ℚ := height_guess_child_inaccurate * weight_guess_loose_clothing_inaccurate
def incorrect_prob_adult_tight : ℚ := height_guess_adult_inaccurate * weight_guess_tight_clothing_inaccurate
def incorrect_prob_adult_loose : ℚ := height_guess_adult_inaccurate * weight_guess_loose_clothing_inaccurate

-- Total probability of incorrect guesses for all three cases
def total_incorrect_prob : ℚ := incorrect_prob_child_loose * incorrect_prob_adult_tight * incorrect_prob_adult_loose

-- Probability of at least one correct guess
def correct_prob_at_least_once : ℚ := 1 - total_incorrect_prob

-- Main theorem stating the final result
theorem martha_correct_guess_probability : correct_prob_at_least_once = 7999 / 8000 := by
  sorry

end MarthaGuess

end martha_correct_guess_probability_l133_133609


namespace min_value_fraction_l133_133461

theorem min_value_fraction (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + 2 * b + 3 * c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 := 
sorry

end min_value_fraction_l133_133461


namespace circumcircle_eqn_l133_133043

def point := ℝ × ℝ

def A : point := (-1, 5)
def B : point := (5, 5)
def C : point := (6, -2)

def circ_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circumcircle_eqn :
  ∃ D E F : ℝ, (∀ (p : point), p ∈ [A, B, C] → circ_eq D E F p.1 p.2) ∧
              circ_eq (-4) (-2) (-20) = circ_eq D E F := by
  sorry

end circumcircle_eqn_l133_133043


namespace minimum_possible_value_of_S_l133_133606

open Finset

def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, x ∈ (range 16 \ {0})) ∧      -- S is a subset of {1, 2, ..., 15}
  (S.card = 7) ∧                          -- S has exactly 7 elements
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬ (a ∣ b ∨ b ∣ a)). -- No a is multiple/factor of b

theorem minimum_possible_value_of_S :
  ∃ S : Finset ℕ, valid_set S ∧ S.min' ⟨_, by linarith⟩ = 3 :=
sorry

end minimum_possible_value_of_S_l133_133606


namespace abs_neg_two_l133_133231

theorem abs_neg_two : abs (-2) = 2 := 
by 
  sorry

end abs_neg_two_l133_133231


namespace eq_circle_value_of_k_l133_133694

noncomputable def circle_center : Prod ℝ ℝ := (2, 3)
noncomputable def circle_radius := 2
noncomputable def line_equation (k : ℝ) : ℝ → ℝ := fun x => k * x - 1
noncomputable def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

theorem eq_circle (x y : ℝ) : 
  circle_equation x y ↔ (x - 2)^2 + (y - 3)^2 = 4 := 
by sorry

theorem value_of_k (k : ℝ) : 
  (∀ M N : Prod ℝ ℝ, 
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 ∧ 
  line_equation k M.1 = M.2 ∧ line_equation k N.1 = N.2 ∧ 
  M ≠ N ∧ 
  (circle_center.1 - M.1) * (circle_center.1 - N.1) + 
  (circle_center.2 - M.2) * (circle_center.2 - N.2) = 0) → 
  (k = 1 ∨ k = 7) := 
by sorry

end eq_circle_value_of_k_l133_133694


namespace average_age_of_cricket_team_l133_133235

theorem average_age_of_cricket_team :
  let captain_age := 28
  let ages_sum := 28 + (28 + 4) + (28 - 2) + (28 + 6)
  let remaining_players := 15 - 4
  let total_sum := ages_sum + remaining_players * (A - 1)
  let total_players := 15
  total_sum / total_players = 27.25 := 
by 
  sorry

end average_age_of_cricket_team_l133_133235


namespace weather_forecast_probability_l133_133627

noncomputable def binomial {n : ℕ} (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

theorem weather_forecast_probability :
  binomial 3 0.8 2 = 0.384 :=
by sorry

end weather_forecast_probability_l133_133627


namespace intersection_eq_l133_133705

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 - x ≤ 0}

theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l133_133705


namespace complex_multiplication_l133_133699

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l133_133699


namespace necessary_condition_l133_133981

theorem necessary_condition :
  (∀ x : ℝ, (1 / x < 3) → (x > 1 / 3)) → (∀ x : ℝ, (1 / x < 3) ↔ (x > 1 / 3)) → False :=
by
  sorry

end necessary_condition_l133_133981


namespace remainder_of_m_div_1000_l133_133607

   -- Define the set T
   def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

   -- Define the computation of m
   noncomputable def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

   -- Statement for the proof problem
   theorem remainder_of_m_div_1000 : m % 1000 = 625 := by
     sorry
   
end remainder_of_m_div_1000_l133_133607


namespace mass_percentage_B_in_H3BO3_l133_133262

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_B : ℝ := 10.81
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_H3BO3 : ℝ := 3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

theorem mass_percentage_B_in_H3BO3 : (atomic_mass_B / molar_mass_H3BO3) * 100 = 17.48 :=
by
  sorry

end mass_percentage_B_in_H3BO3_l133_133262


namespace exists_N_for_sqrt_expressions_l133_133645

theorem exists_N_for_sqrt_expressions 
  (p q n : ℕ) (hp : 0 < p) (hq : 0 < q) (hn : 0 < n) (h_q_le_p2 : q ≤ p^2) :
  ∃ N : ℕ, 
    (N > 0) ∧ 
    ((p - Real.sqrt (p^2 - q))^n = N - Real.sqrt (N^2 - q^n)) ∧ 
    ((p + Real.sqrt (p^2 - q))^n = N + Real.sqrt (N^2 - q^n)) :=
sorry

end exists_N_for_sqrt_expressions_l133_133645


namespace correct_operation_l133_133931

variable {a b : ℝ}

theorem correct_operation : (3 * a^2 * b - 3 * b * a^2 = 0) :=
by sorry

end correct_operation_l133_133931


namespace ellipse_y_axis_intersection_l133_133669

open Real

/-- Defines an ellipse with given foci and a point on the ellipse,
    and establishes the coordinate of the other y-axis intersection. -/
theorem ellipse_y_axis_intersection :
  ∃ y : ℝ, (dist (0, y) (1, -1) + dist (0, y) (-2, 2) = 3 * sqrt 2) ∧ y = sqrt ((9 * sqrt 2 - 4) / 2) :=
sorry

end ellipse_y_axis_intersection_l133_133669


namespace quadratic_equation_m_l133_133336

theorem quadratic_equation_m (m b : ℝ) (h : (m - 2) * x ^ |m| - b * x - 1 = 0) : m = -2 :=
by
  sorry

end quadratic_equation_m_l133_133336


namespace triangle_parallelograms_l133_133982

def f (n : ℕ) : ℕ := 3 * (n + 2).choose 4

theorem triangle_parallelograms (n : ℕ) : 
  ∃ f : ℕ → ℕ, f = λ n, 3 * (n + 2).choose 4 ∧ f(n) = 3 * (n + 2).choose 4 :=
by
  sorry

end triangle_parallelograms_l133_133982


namespace Luke_piles_of_quarters_l133_133079

theorem Luke_piles_of_quarters (Q : ℕ) (h : 6 * Q = 30) : Q = 5 :=
by
  sorry

end Luke_piles_of_quarters_l133_133079


namespace price_reduction_example_l133_133464

def original_price_per_mango (P : ℝ) : Prop :=
  (115 * P = 383.33)

def number_of_mangoes (P : ℝ) (n : ℝ) : Prop :=
  (n * P = 360)

def new_number_of_mangoes (n : ℝ) (R : ℝ) : Prop :=
  ((n + 12) * R = 360)

def percentage_reduction (P R : ℝ) (reduction : ℝ) : Prop :=
  (reduction = ((P - R) / P) * 100)

theorem price_reduction_example : 
  ∃ P R reduction, original_price_per_mango P ∧
    (∃ n, number_of_mangoes P n ∧ new_number_of_mangoes n R) ∧ 
    percentage_reduction P R reduction ∧ 
    reduction = 9.91 :=
by
  sorry

end price_reduction_example_l133_133464


namespace triangle_internal_angles_external_angle_theorem_l133_133596

theorem triangle_internal_angles {A B C : ℝ}
 (mA : A = 64) (mB : B = 33) (mC_ext : C = 120) :
  180 - A - B = 83 :=
by
  sorry

theorem external_angle_theorem {A C D : ℝ}
 (mA : A = 64) (mC_ext : C = 120) :
  C = A + D → D = 56 :=
by
  sorry

end triangle_internal_angles_external_angle_theorem_l133_133596


namespace determine_integer_n_l133_133007

theorem determine_integer_n (n : ℤ) :
  (n + 15 ≥ 16) ∧ (-5 * n < -10) → n = 3 :=
by
  sorry

end determine_integer_n_l133_133007


namespace problem_solution_l133_133174

variables {f : ℝ → ℝ}

-- f is monotonically decreasing on [1, 3]
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- f(x+3) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = f (3 - x)

-- Given conditions
axiom mono_dec : monotone_decreasing_on f 1 3
axiom even_f : even_function f

-- To prove: f(π) < f(2) < f(5)
theorem problem_solution : f π < f 2 ∧ f 2 < f 5 :=
by
  sorry

end problem_solution_l133_133174


namespace complementary_event_equivalence_l133_133405

-- Define the event E: hitting the target at least once in two shots.
-- Event E complementary: missing the target both times.

def eventE := "hitting the target at least once"
def complementaryEvent := "missing the target both times"

theorem complementary_event_equivalence :
  (complementaryEvent = "missing the target both times") ↔ (eventE = "hitting the target at least once") :=
by
  sorry

end complementary_event_equivalence_l133_133405


namespace system_equations_solution_l133_133482

theorem system_equations_solution (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 3) ∧ 
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) ∧ 
  (1 / (x * y * z) = 1) → 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end system_equations_solution_l133_133482


namespace triangle_shape_l133_133572

theorem triangle_shape (a b c : ℝ) (h : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_shape_l133_133572


namespace length_of_AB_l133_133703

theorem length_of_AB 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : sqrt (1 + (b^2 / a^2)) = sqrt 5)
  (x y : ℝ) (h4 : -(x - 2)^2 + -(y - 3)^2 = 1) : 
  ∃ (A B : ℝ × ℝ), |(B.1 - A.1, B.2 - A.2)| = (4 * sqrt 5 / 5) :=
begin
  sorry
end

end length_of_AB_l133_133703


namespace initial_people_in_gym_l133_133969

variables (W A S : ℕ)

theorem initial_people_in_gym (h1 : (W - 3 + 2 - 3 + 4 - 2 + 1 = W + 1))
                              (h2 : (A + 2 - 1 + 3 - 3 + 1 = A + 2))
                              (h3 : (S + 1 - 2 + 1 + 3 - 2 + 2 = S + 3))
                              (final_total : (W + 1) + (A + 2) + (S + 3) + 2 = 30) :
  W + A + S = 22 :=
by 
  sorry

end initial_people_in_gym_l133_133969


namespace obtuse_probability_l133_133750

-- Define the vertices of the pentagon
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨0, 2⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨2*real.pi + 1, 0⟩
def D : Point := ⟨2*real.pi + 1, 4⟩
def E : Point := ⟨0, 4⟩

-- Define the center of the semicircle and its radius
def center : Point := ⟨2, 1⟩
def radius : ℝ := real.sqrt 5

-- Define the conditions for angle APB to be obtuse
def is_obtuse (P : Point) : Prop := 
  ∠((A.x, A.y), (P.x, P.y), (B.x, B.y)) > real.pi / 2

-- Define the probability calculation
noncomputable def area_pentagon : ℝ := 8 * real.pi
noncomputable def area_semicircle : ℝ := (5/2) * real.pi
noncomputable def probability_obtuse : ℝ := area_semicircle / area_pentagon

-- The theorem to prove:
theorem obtuse_probability : probability_obtuse = 5 / 16 :=
  sorry

end obtuse_probability_l133_133750


namespace spending_record_l133_133833

-- Definitions based on conditions
def deposit_record (x : ℤ) : ℤ := x
def spend_record (x : ℤ) : ℤ := -x

-- Theorem statement
theorem spending_record (x : ℤ) (hx : x = 500) : spend_record x = -500 := by
  sorry

end spending_record_l133_133833


namespace find_m_l133_133039

theorem find_m (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 < 2) ∧ (m^2 + m = 2) → m = -2 :=
by
  sorry

end find_m_l133_133039


namespace mark_money_l133_133207

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l133_133207


namespace ellipse_equation_max_area_line_eqn_l133_133420

-- Definitions of ellipse and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b c : ℝ) : Prop := c / a = sqrt 3 / 2 ∧ c^2 = a^2 - b^2

-- Given point M
def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Line l intersects ellipse at points P and Q with |AP| = |AQ|
def line_l_intersect (P Q A : ℝ × ℝ) (C : ℝ → ℝ → Prop) : Prop := 
  C P.1 P.2 ∧ C Q.1 Q.2 ∧ A.2 = -1 ∧ (A.1 = 0 → (P.1 = Q.1 ∧ A.1^2 + A.2^2 = (P.1 - A.2)^2))

-- Given the origin O(0, 0)
def origin (O : ℝ × ℝ) : Prop := O.1 = 0 ∧ O.2 = 0

-- Problems to be proven
theorem ellipse_equation : 
  ∀ (a b : ℝ), 0 < b ∧ b < a -> 
    point_M M.1 M.2 ->
    eccentricity a b sqrt(a^2 - b^2) →
    ellipse a b 2 1 →
    ellipse a b 0 (-1) →
    ellipse a b x y ↔ (x^2 / 8) + (y^2 / 2) = 1 :=
by sorry

theorem max_area_line_eqn :
  ∀ (a b : ℝ), 0 < b ∧ b < a →
    point_M M.1 M.2 →
    eccentricity a b sqrt(a^2 - b^2) →
    ellipse a b 2 1 →
    origin O →
  ∃ (k m x y : ℝ), line_l_intersect (x, y) (k*x + m, k*y + m) (0, -1) (ellipse 2 1) ∧ 
      max_area (triangle O (x, y) (k*x + m, k*y + m)) →
        m = 3 ∧ (k = 1 ∨ k = -1 ∨ k = sqrt 2 ∨ k = -sqrt 2) :=
by sorry

end ellipse_equation_max_area_line_eqn_l133_133420


namespace symmetric_points_parabola_l133_133554

theorem symmetric_points_parabola (x1 x2 y1 y2 m : ℝ) (h1 : y1 = 2 * x1^2) (h2 : y2 = 2 * x2^2)
    (h3 : x1 * x2 = -3 / 4) (h_sym: (y2 - y1) / (x2 - x1) = -1)
    (h_mid: (y2 + y1) / 2 = (x2 + x1) / 2 + m) :
    m = 2 := sorry

end symmetric_points_parabola_l133_133554


namespace probability_points_one_unit_apart_l133_133483

theorem probability_points_one_unit_apart :
  let points := 10
  let rect_length := 3
  let rect_width := 2
  let total_pairs := (points * (points - 1)) / 2
  let favorable_pairs := 10  -- derived from solution steps
  (favorable_pairs / total_pairs : ℚ) = (2 / 9 : ℚ) :=
by
  sorry

end probability_points_one_unit_apart_l133_133483


namespace average_speed_is_69_l133_133651

-- Definitions for the conditions
def distance_hr1 : ℕ := 90
def distance_hr2 : ℕ := 30
def distance_hr3 : ℕ := 60
def distance_hr4 : ℕ := 120
def distance_hr5 : ℕ := 45
def total_distance : ℕ := distance_hr1 + distance_hr2 + distance_hr3 + distance_hr4 + distance_hr5
def total_time : ℕ := 5

-- The theorem to be proven
theorem average_speed_is_69 :
  (total_distance / total_time) = 69 :=
by
  sorry

end average_speed_is_69_l133_133651


namespace edricHourlyRateIsApproximatelyCorrect_l133_133987

-- Definitions as per conditions
def edricMonthlySalary : ℝ := 576
def edricHoursPerDay : ℝ := 8
def edricDaysPerWeek : ℝ := 6
def weeksPerMonth : ℝ := 4.33

-- Calculation as per the proof problem
def edricWeeklyHours (hoursPerDay daysPerWeek : ℝ) : ℝ := hoursPerDay * daysPerWeek

def edricMonthlyHours (weeklyHours weeksPerMonth : ℝ) : ℝ := weeklyHours * weeksPerMonth

def edricHourlyRate (monthlySalary monthlyHours : ℝ) : ℝ := monthlySalary / monthlyHours

-- The theorem to prove
theorem edricHourlyRateIsApproximatelyCorrect : (edricHourlyRate edricMonthlySalary (edricMonthlyHours (edricWeeklyHours edricHoursPerDay edricDaysPerWeek) weeksPerMonth)) ≈ 2.77 :=
by
  sorry

end edricHourlyRateIsApproximatelyCorrect_l133_133987


namespace quadratic_value_at_5_l133_133624

-- Define the conditions provided in the problem
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Create a theorem that states that if a quadratic with given conditions has its vertex at (2, 7) and passes through (0, -7), then passing through (5, n) means n = -24.5
theorem quadratic_value_at_5 (a b c n : ℝ)
  (h1 : quadratic a b c 2 = 7)
  (h2 : quadratic a b c 0 = -7)
  (h3 : quadratic a b c 5 = n) :
  n = -24.5 :=
by
  sorry

end quadratic_value_at_5_l133_133624


namespace quadratic_inequality_solution_l133_133150

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 18 < 0} = {x : ℝ | -3 < x ∧ x < 6} :=
by
  sorry

end quadratic_inequality_solution_l133_133150


namespace emma_missing_coins_l133_133010

theorem emma_missing_coins (x : ℤ) (h₁ : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  let missing := x - remaining
  missing / x = 1 / 9 :=
by
  sorry

end emma_missing_coins_l133_133010


namespace rearrange_digits_2552_l133_133709

theorem rearrange_digits_2552 : 
    let digits := [2, 5, 5, 2]
    let factorial := fun n => Nat.factorial n
    let permutations := (factorial 4) / (factorial 2 * factorial 2)
    permutations = 6 :=
by
  sorry

end rearrange_digits_2552_l133_133709


namespace age_of_17th_student_l133_133125

theorem age_of_17th_student (avg_age_17 : ℕ) (total_students : ℕ) (avg_age_5 : ℕ) (students_5 : ℕ) (avg_age_9 : ℕ) (students_9 : ℕ)
  (h1 : avg_age_17 = 17) (h2 : total_students = 17) (h3 : avg_age_5 = 14) (h4 : students_5 = 5) (h5 : avg_age_9 = 16) (h6 : students_9 = 9) :
  ∃ age_17th_student : ℕ, age_17th_student = 75 :=
by
  sorry

end age_of_17th_student_l133_133125


namespace period_of_sine_plus_cosine_l133_133788

noncomputable def period_sine_cosine_sum (b : ℝ) : ℝ :=
  2 * Real.pi / b

theorem period_of_sine_plus_cosine (b : ℝ) (hb : b = 3) :
  period_sine_cosine_sum b = 2 * Real.pi / 3 :=
by
  rw [hb]
  apply rfl

end period_of_sine_plus_cosine_l133_133788


namespace reflection_eq_l133_133898

theorem reflection_eq (x y : ℝ) : 
    let line_eq (x y : ℝ) := 2 * x + 3 * y - 5 = 0 
    let reflection_eq (x y : ℝ) := 3 * x + 2 * y - 5 = 0 
    (∀ (x y : ℝ), line_eq x y ↔ reflection_eq y x) →
    reflection_eq x y :=
by
    sorry

end reflection_eq_l133_133898


namespace average_squares_of_first_10_multiples_of_7_correct_l133_133154

def first_10_multiples_of_7 : List ℕ := List.map (fun n => 7 * n) (List.range 10)

def squares (l : List ℕ) : List ℕ := List.map (fun n => n * n) l

def sum (l : List ℕ) : ℕ := List.foldr (· + ·) 0 l

theorem average_squares_of_first_10_multiples_of_7_correct :
  (sum (squares first_10_multiples_of_7) / 10 : ℚ) = 1686.5 :=
by
  sorry

end average_squares_of_first_10_multiples_of_7_correct_l133_133154


namespace base_edge_length_l133_133371

theorem base_edge_length (x : ℕ) :
  (∃ (x : ℕ), 
    (∀ (sum_edges : ℕ), sum_edges = 6 * x + 48 → sum_edges = 120) →
    x = 12) := 
sorry

end base_edge_length_l133_133371


namespace max_pieces_in_8x8_grid_l133_133442

theorem max_pieces_in_8x8_grid : 
  ∃ m n : ℕ, (m = 8) ∧ (n = 9) ∧ 
  (∀ H V : ℕ, (H ≤ n) → (V ≤ n) → 
   (H + V + 1 ≤ 16)) := sorry

end max_pieces_in_8x8_grid_l133_133442


namespace apricot_tea_calories_l133_133465

theorem apricot_tea_calories :
  let apricot_juice_weight := 150
  let apricot_juice_calories_per_100g := 30
  let honey_weight := 50
  let honey_calories_per_100g := 304
  let water_weight := 300
  let apricot_tea_weight := apricot_juice_weight + honey_weight + water_weight
  let apricot_juice_calories := apricot_juice_weight * apricot_juice_calories_per_100g / 100
  let honey_calories := honey_weight * honey_calories_per_100g / 100
  let total_calories := apricot_juice_calories + honey_calories
  let caloric_density := total_calories / apricot_tea_weight
  let tea_weight := 250
  let calories_in_250g_tea := tea_weight * caloric_density
  calories_in_250g_tea = 98.5 := by
  sorry

end apricot_tea_calories_l133_133465


namespace average_income_l133_133122

/-- The daily incomes of the cab driver over 5 days. --/
def incomes : List ℕ := [400, 250, 650, 400, 500]

/-- Prove that the average income of the cab driver over these 5 days is $440. --/
theorem average_income : (incomes.sum / incomes.length) = 440 := by
  sorry

end average_income_l133_133122


namespace John_overall_profit_l133_133450

theorem John_overall_profit :
  let CP_grinder := 15000
  let Loss_percentage_grinder := 0.04
  let CP_mobile_phone := 8000
  let Profit_percentage_mobile_phone := 0.10
  let CP_refrigerator := 24000
  let Profit_percentage_refrigerator := 0.08
  let CP_television := 12000
  let Loss_percentage_television := 0.06
  let SP_grinder := CP_grinder * (1 - Loss_percentage_grinder)
  let SP_mobile_phone := CP_mobile_phone * (1 + Profit_percentage_mobile_phone)
  let SP_refrigerator := CP_refrigerator * (1 + Profit_percentage_refrigerator)
  let SP_television := CP_television * (1 - Loss_percentage_television)
  let Total_CP := CP_grinder + CP_mobile_phone + CP_refrigerator + CP_television
  let Total_SP := SP_grinder + SP_mobile_phone + SP_refrigerator + SP_television
  let Overall_profit := Total_SP - Total_CP
  Overall_profit = 1400 := by
  sorry

end John_overall_profit_l133_133450


namespace largest_number_l133_133962

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = 0) (h3 : c = 1) (h4 : d = -9) :
  max (max a b) (max c d) = c :=
by
  sorry

end largest_number_l133_133962


namespace focus_of_parabola_l133_133236

theorem focus_of_parabola (focus : ℝ × ℝ) : 
  (∃ p : ℝ, y = p * x^2 / 2 → focus = (0, 1 / 2)) :=
by
  sorry

end focus_of_parabola_l133_133236


namespace Tyler_age_l133_133206

variable (T B S : ℕ) -- Assuming ages are non-negative integers

theorem Tyler_age (h1 : T = B - 3) (h2 : T + B + S = 25) (h3 : S = B + 2) : T = 6 := by
  sorry

end Tyler_age_l133_133206


namespace equilateral_A1C1E1_l133_133219

variables {A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Type*}

-- Defining the convex hexagon and the equilateral triangles.
def is_convex_hexagon (A B C D E F : Type*) : Prop := sorry

def is_equilateral (P Q R : Type*) : Prop := sorry

-- Given conditions
variable (h_hexagon : is_convex_hexagon A B C D E F)
variable (h_eq_triangles :
  is_equilateral A B C₁ ∧ is_equilateral B C D₁ ∧ is_equilateral C D E₁ ∧
  is_equilateral D E F₁ ∧ is_equilateral E F A₁ ∧ is_equilateral F A B₁)
variable (h_B1D1F1 : is_equilateral B₁ D₁ F₁)

-- Statement to be proved
theorem equilateral_A1C1E1 :
  is_equilateral A₁ C₁ E₁ :=
sorry

end equilateral_A1C1E1_l133_133219


namespace bus_return_trip_fraction_l133_133809

theorem bus_return_trip_fraction :
  (3 / 4 * 200 + x * 200 = 310) → (x = 4 / 5) := by
  sorry

end bus_return_trip_fraction_l133_133809


namespace distance_at_2_point_5_l133_133248

def distance_data : List (ℝ × ℝ) :=
  [(0, 0), (1, 10), (2, 40), (3, 90), (4, 160), (5, 250)]

def quadratic_relation (t s k : ℝ) : Prop :=
  s = k * t^2

theorem distance_at_2_point_5 :
  ∃ k : ℝ, (∀ (t s : ℝ), (t, s) ∈ distance_data → quadratic_relation t s k) ∧ quadratic_relation 2.5 62.5 k :=
by
  sorry

end distance_at_2_point_5_l133_133248


namespace ab_is_10_pow_116_l133_133616

noncomputable def ab (a b : ℝ) : ℝ :=
  if 2 * (Real.sqrt (Real.log a) + Real.sqrt (Real.log b)) + Real.log (Real.sqrt a) + Real.log (Real.sqrt b) = 108
  then a * b
  else 0

theorem ab_is_10_pow_116 (a b : ℝ) 
  (hPosa : 0 < a) (hPosb : 0 < b) 
  (h : 2 * (Real.sqrt (Real.log a) + Real.sqrt (Real.log b)) + Real.log (Real.sqrt a) + Real.log (Real.sqrt b) = 108) :
  ab a b = 10^116 :=
by
  sorry

end ab_is_10_pow_116_l133_133616


namespace horner_polynomial_rewrite_polynomial_value_at_5_l133_133511

def polynomial (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 6 * x^3 - 2 * x^2 - 5 * x - 2

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_polynomial_rewrite :
  polynomial = horner_polynomial := 
sorry

theorem polynomial_value_at_5 :
  polynomial 5 = 7548 := 
sorry

end horner_polynomial_rewrite_polynomial_value_at_5_l133_133511


namespace arithmetic_sequence_ratio_l133_133245

/-- 
  Given the ratio of the sum of the first n terms of two arithmetic sequences,
  prove the ratio of the 11th terms of these sequences.
-/
theorem arithmetic_sequence_ratio (S T : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 1 : ℚ) / (4 * n + 2)) : 
  S 21 / T 21 = 74 / 43 :=
sorry

end arithmetic_sequence_ratio_l133_133245


namespace total_students_l133_133216

-- Define the conditions
def chocolates_distributed (y z : ℕ) : ℕ :=
  y * y + z * z

-- Define the main theorem to be proved
theorem total_students (y z : ℕ) (h : z = y + 3) (chocolates_left: ℕ) (initial_chocolates: ℕ)
  (h_chocolates: chocolates_distributed y z = initial_chocolates - chocolates_left) : 
  y + z = 33 :=
by
  sorry

end total_students_l133_133216


namespace find_c_plus_d_l133_133186

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l133_133186


namespace charlie_steps_l133_133830

theorem charlie_steps (steps_per_run : ℕ) (runs : ℝ) (expected_steps : ℕ) :
  steps_per_run = 5350 →
  runs = 2.5 →
  expected_steps = 13375 →
  runs * steps_per_run = expected_steps :=
by intros; linarith; sorry

end charlie_steps_l133_133830


namespace range_of_m_for_circle_l133_133095

theorem range_of_m_for_circle (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0)  ↔ m < -2*Real.sqrt 3 ∨ m > 2*Real.sqrt 3 :=
by 
  sorry

end range_of_m_for_circle_l133_133095


namespace which_is_linear_l133_133272

-- Define what it means to be a linear equation in two variables
def is_linear_equation_in_two_vars (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- Define each of the given equations
def equation_A (x y : ℝ) : Prop := x / 2 + 3 * y = 2
def equation_B (x y : ℝ) : Prop := x / 2 + 1 = 3 * x * y
def equation_C (x y : ℝ) : Prop := 2 * x + 1 = 3 * x
def equation_D (x y : ℝ) : Prop := 3 * x + 2 * y^2 = 1

-- Theorem stating which equation is linear in two variables
theorem which_is_linear : 
  is_linear_equation_in_two_vars equation_A ∧ 
  ¬ is_linear_equation_in_two_vars equation_B ∧ 
  ¬ is_linear_equation_in_two_vars equation_C ∧ 
  ¬ is_linear_equation_in_two_vars equation_D := 
by 
  sorry

end which_is_linear_l133_133272


namespace watch_arrangement_count_l133_133941

noncomputable def number_of_satisfying_watch_arrangements : Nat :=
  let dial_arrangements := Nat.factorial 2
  let strap_arrangements := Nat.factorial 3
  dial_arrangements * strap_arrangements

theorem watch_arrangement_count :
  number_of_satisfying_watch_arrangements = 12 :=
by
-- Proof omitted
sorry

end watch_arrangement_count_l133_133941


namespace candy_count_correct_l133_133156

-- Define initial count of candy
def initial_candy : ℕ := 47

-- Define number of pieces of candy eaten
def eaten_candy : ℕ := 25

-- Define number of pieces of candy received
def received_candy : ℕ := 40

-- The final count of candy is what we are proving
theorem candy_count_correct : initial_candy - eaten_candy + received_candy = 62 :=
by
  sorry

end candy_count_correct_l133_133156


namespace number_of_machines_in_first_scenario_l133_133088

noncomputable def machine_work_rate (R : ℝ) (hours_per_job : ℝ) : Prop :=
  (6 * R * 8 = 1)

noncomputable def machines_first_scenario (M : ℝ) (R : ℝ) (hours_per_job_first : ℝ) : Prop :=
  (M * R * hours_per_job_first = 1)

theorem number_of_machines_in_first_scenario (M : ℝ) (R : ℝ) :
  machine_work_rate R 8 ∧ machines_first_scenario M R 6 -> M = 8 :=
sorry

end number_of_machines_in_first_scenario_l133_133088


namespace power_equality_l133_133333

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l133_133333


namespace fish_ranking_l133_133083

def ranks (P V K T : ℕ) : Prop :=
  P < K ∧ K < T ∧ T < V

theorem fish_ranking (P V K T : ℕ) (h1 : K < T) (h2 : P + V = K + T) (h3 : P + T < V + K) : ranks P V K T :=
by
  sorry

end fish_ranking_l133_133083


namespace moe_cannot_finish_on_time_l133_133744

theorem moe_cannot_finish_on_time (lawn_length lawn_width : ℝ) (swath : ℕ) (overlap : ℕ) (speed : ℝ) (available_time : ℝ) :
  lawn_length = 120 ∧ lawn_width = 180 ∧ swath = 30 ∧ overlap = 6 ∧ speed = 4000 ∧ available_time = 2 →
  (lawn_width / (swath - overlap) * lawn_length / speed) > available_time :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end moe_cannot_finish_on_time_l133_133744


namespace greatest_possible_individual_award_l133_133134

variable (prize : ℕ)
variable (total_winners : ℕ)
variable (min_award : ℕ)
variable (fraction_prize : ℚ)
variable (fraction_winners : ℚ)

theorem greatest_possible_individual_award 
  (h1 : prize = 2500)
  (h2 : total_winners = 25)
  (h3 : min_award = 50)
  (h4 : fraction_prize = 3/5)
  (h5 : fraction_winners = 2/5) :
  ∃ award, award = 1300 := by
  sorry

end greatest_possible_individual_award_l133_133134


namespace area_ratio_of_squares_l133_133550

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * (4 * b) = 4 * a) : (a * a) / (b * b) = 16 :=
by
  sorry

end area_ratio_of_squares_l133_133550


namespace average_speed_l133_133538

theorem average_speed (v1 v2 t1 t2 total_time total_distance : ℝ)
  (h1 : v1 = 50)
  (h2 : t1 = 4)
  (h3 : v2 = 80)
  (h4 : t2 = 4)
  (h5 : total_time = t1 + t2)
  (h6 : total_distance = v1 * t1 + v2 * t2) :
  (total_distance / total_time = 65) :=
by
  sorry

end average_speed_l133_133538


namespace degree_of_divisor_polynomial_l133_133658

theorem degree_of_divisor_polynomial (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15)
  (hq : q.degree = 9)
  (hr : r.degree = 4)
  (hfdqr : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_polynomial_l133_133658


namespace probability_correct_l133_133492

-- Define the set of segment lengths
def segment_lengths : List ℕ := [1, 3, 5, 7, 9]

-- Define the triangle inequality condition
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of favorable outcomes, i.e., sets that can form a triangle
def favorable_sets : List (ℕ × ℕ × ℕ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

-- Define the total number of ways to select three segments out of five
def total_combinations : ℕ :=
  10

-- Define the number of favorable sets
def number_of_favorable_sets : ℕ :=
  favorable_sets.length

-- Calculate the probability of selecting three segments that form a triangle
def probability_of_triangle : ℚ :=
  number_of_favorable_sets / total_combinations

-- The theorem to prove
theorem probability_correct : probability_of_triangle = 3 / 10 :=
  by {
    -- Placeholder for the proof
    sorry
  }

end probability_correct_l133_133492


namespace geometric_probability_l133_133396

noncomputable def probability_point_within_rectangle (l w : ℝ) (A_rectangle A_circle : ℝ) : ℝ :=
  A_rectangle / A_circle

theorem geometric_probability (l w : ℝ) (r : ℝ) (A_rectangle : ℝ) (h_length : l = 4) 
  (h_width : w = 3) (h_radius : r = 2.5) (h_area_rectangle : A_rectangle = 12) :
  A_rectangle / (Real.pi * r^2) = 48 / (25 * Real.pi) :=
by
  sorry

end geometric_probability_l133_133396


namespace cauchy_schwarz_inequality_l133_133855

theorem cauchy_schwarz_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by
  sorry

end cauchy_schwarz_inequality_l133_133855


namespace problem_lean_version_l133_133798

theorem problem_lean_version (n : ℕ) : 
  (n > 0) ∧ (6^n - 1 ∣ 7^n - 1) ↔ ∃ k : ℕ, n = 4 * k :=
by
  sorry

end problem_lean_version_l133_133798


namespace min_expression_value_l133_133302

theorem min_expression_value (a b c : ℝ) (h_sum : a + b + c = -1) (h_abc : a * b * c ≤ -3) :
  3 ≤ (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) :=
sorry

end min_expression_value_l133_133302


namespace nathan_tokens_used_is_18_l133_133352

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l133_133352


namespace remainder_when_divided_by_10_l133_133117

theorem remainder_when_divided_by_10 :
  (4219 * 2675 * 394082 * 5001) % 10 = 0 :=
sorry

end remainder_when_divided_by_10_l133_133117


namespace at_least_three_heads_in_ten_flips_l133_133652

theorem at_least_three_heads_in_ten_flips : 
  let total_sequences := 2^10
  let fewer_than_three_heads := nat.choose 10 0 + nat.choose 10 1 + nat.choose 10 2
  total_sequences - fewer_than_three_heads = 968 :=
by
  let total_sequences := 2^10
  let fewer_than_three_heads := nat.choose 10 0 + nat.choose 10 1 + nat.choose 10 2
  show total_sequences - fewer_than_three_heads = 968 from sorry

end at_least_three_heads_in_ten_flips_l133_133652


namespace expected_number_of_males_in_sample_l133_133817

theorem expected_number_of_males_in_sample : 
  let total_athletes := 48 + 36
  let male_ratio := 48 / total_athletes.to_rat
  let sample_size := 21
  let expected_males := male_ratio * sample_size
  expected_males = 12 :=
by
  let total_athletes := 48 + 36
  let male_ratio := 48 / total_athletes.to_rat
  let sample_size := 21
  let expected_males := male_ratio * sample_size
  have h : total_athletes = 84 := rfl
  have r : male_ratio = 4 / 7 := by norm_num [total_athletes, male_ratio]
  have s : expected_males = (4 / 7) * 21 := by simp [male_ratio, sample_size]
  have result : expected_males = 12 := by norm_num [s]
  exact result

end expected_number_of_males_in_sample_l133_133817


namespace total_members_in_club_l133_133719

theorem total_members_in_club (females : ℕ) (males : ℕ) (total : ℕ) : 
  (females = 12) ∧ (females = 2 * males) ∧ (total = females + males) → total = 18 := 
by
  sorry

end total_members_in_club_l133_133719


namespace speed_of_sisters_sailboat_l133_133065

variable (v_j : ℝ) (d : ℝ) (t_wait : ℝ)

-- Conditions
def janet_speed : Prop := v_j = 30
def lake_distance : Prop := d = 60
def janet_wait_time : Prop := t_wait = 3

-- Question to Prove
def sister_speed (v_s : ℝ) : Prop :=
  janet_speed v_j ∧ lake_distance d ∧ janet_wait_time t_wait →
  v_s = 12

-- The main theorem
theorem speed_of_sisters_sailboat (v_j d t_wait : ℝ) (h1 : janet_speed v_j) (h2 : lake_distance d) (h3 : janet_wait_time t_wait) :
  ∃ v_s : ℝ, sister_speed v_j d t_wait v_s :=
by
  sorry

end speed_of_sisters_sailboat_l133_133065


namespace isosceles_triangle_base_length_l133_133895

theorem isosceles_triangle_base_length
  (a : ℕ) (b : ℕ)
  (ha : a = 7) 
  (p : ℕ)
  (hp : p = a + a + b) 
  (hp_perimeter : p = 21) : b = 7 :=
by 
  -- The actual proof will go here, using the provided conditions
  sorry

end isosceles_triangle_base_length_l133_133895


namespace simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l133_133429

def A (a b : ℝ) := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) := 4 * a^2 + 6 * a * b + 8 * a

theorem simplify_2A_minus_B {a b : ℝ} :
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a :=
by
  sorry

theorem twoA_minusB_value_when_a_neg2_b_1 :
  2 * A (-2) (1) - B (-2) (1) = 54 :=
by
  sorry

theorem twoA_minusB_independent_of_a {b : ℝ} :
  (∀ a : ℝ, 2 * A a b - B a b = 6 * b - 8 * a) → b = -1 / 2 :=
by
  sorry

end simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l133_133429


namespace largest_digit_divisible_by_6_l133_133518

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l133_133518


namespace cube_surface_area_l133_133797

noncomputable def volume_of_cube (s : ℝ) := s ^ 3
noncomputable def surface_area_of_cube (s : ℝ) := 6 * (s ^ 2)

theorem cube_surface_area (s : ℝ) (h : volume_of_cube s = 1728) : surface_area_of_cube s = 864 :=
  sorry

end cube_surface_area_l133_133797


namespace range_of_a_l133_133178

variable (x a : ℝ)

theorem range_of_a (h1 : ∀ x, x ≤ a → x < 2) (h2 : ∀ x, x < 2) : a ≥ 2 :=
sorry

end range_of_a_l133_133178


namespace car_price_l133_133619

theorem car_price (down_payment : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) 
    (h_down_payment : down_payment = 5000) 
    (h_monthly_payment : monthly_payment = 250)
    (h_loan_years : loan_years = 5) : 
    down_payment + monthly_payment * loan_years * 12 = 20000 := 
by
  rw [h_down_payment, h_monthly_payment, h_loan_years]
  norm_num
  sorry

end car_price_l133_133619


namespace find_f_l133_133423

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : ∀ x : ℝ, f x = x + 1 :=
by
  sorry

end find_f_l133_133423


namespace petya_vasya_sum_equality_l133_133749

theorem petya_vasya_sum_equality : ∃ (k m : ℕ), 2^(k+1) * 1023 = m * (m + 1) :=
by
  sorry

end petya_vasya_sum_equality_l133_133749


namespace solution_set_of_inequality_l133_133630

theorem solution_set_of_inequality : {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x^2 + 2 * x < 3} :=
sorry

end solution_set_of_inequality_l133_133630


namespace prove_positive_a_l133_133415

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end prove_positive_a_l133_133415


namespace equality_conditions_l133_133558

theorem equality_conditions (a b c d : ℝ) :
  a + bcd = (a + b) * (a + c) * (a + d) ↔ a = 0 ∨ a^2 + a * (b + c + d) + bc + bd + cd = 1 :=
by
  sorry

end equality_conditions_l133_133558


namespace sum_of_converted_2016_is_correct_l133_133222

theorem sum_of_converted_2016_is_correct :
  (20.16 + 20.16 + 20.16 + 201.6 + 201.6 + 201.6 = 463.68 ∨
   2.016 + 2.016 + 2.016 + 20.16 + 20.16 + 20.16 = 46.368) :=
by
  sorry

end sum_of_converted_2016_is_correct_l133_133222


namespace find_digits_l133_133999

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l133_133999


namespace second_pipe_fill_time_l133_133510

theorem second_pipe_fill_time :
  ∃ x : ℝ, x ≠ 0 ∧ (1 / 10 + 1 / x - 1 / 20 = 1 / 7.5) ∧ x = 60 :=
by
  sorry

end second_pipe_fill_time_l133_133510


namespace ab_value_l133_133170

theorem ab_value (a b : ℝ) (h1 : a = Real.exp (2 - a)) (h2 : 1 + Real.log b = Real.exp (1 - Real.log b)) : 
  a * b = Real.exp 1 :=
sorry

end ab_value_l133_133170


namespace graph_of_eq_hyperbola_l133_133793

theorem graph_of_eq_hyperbola (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 1 → ∃ a b : ℝ, a * b = x * y ∧ a * b = 1/2 := by
  sorry

end graph_of_eq_hyperbola_l133_133793


namespace volleyball_team_starters_l133_133613

/--
There are 18 players in the school's girls volleyball team, including a set of quadruplets: Beth, Barbara, Bonnie, and Brenda.
- Beth, Barbara, Bonnie, and Brenda must be in the starting lineup.
We want to determine the number of ways to choose 8 starters given this condition.
--/
theorem volleyball_team_starters :
  let total_players := 18
  let quadruplets := 4
  let remaining_positions := 8 - quadruplets
  choose (total_players - quadruplets) remaining_positions = 1001 :=
by
  sorry

end volleyball_team_starters_l133_133613


namespace greene_family_admission_cost_l133_133090

theorem greene_family_admission_cost (x : ℝ) (h1 : ∀ y : ℝ, y = x - 13) (h2 : ∀ z : ℝ, z = x + (x - 13)) :
  x = 45 :=
by
  sorry

end greene_family_admission_cost_l133_133090


namespace max_piles_l133_133768

theorem max_piles (n : ℕ) (hn : n = 660) :
  ∃ (k : ℕ), (∀ (piles : list ℕ),
    (sum piles = n) →
    (∀ (x y : ℕ), x ∈ piles → y ∈ piles → x ≤ 2 * y ∧ y ≤ 2 * x) →
    list.length piles ≤ k) ∧ k = 30 :=
sorry

end max_piles_l133_133768


namespace find_missing_number_l133_133490

theorem find_missing_number (x : ℕ) (h1 : (1 + 22 + 23 + 24 + x + 26 + 27 + 2) = 8 * 20) : x = 35 :=
  sorry

end find_missing_number_l133_133490


namespace total_votes_is_132_l133_133439

theorem total_votes_is_132 (T : ℚ) 
  (h1 : 1 / 4 * T + 1 / 3 * T = 77) : 
  T = 132 := 
  sorry

end total_votes_is_132_l133_133439


namespace max_piles_660_stones_l133_133767

theorem max_piles_660_stones (init_stones : ℕ) (A : finset ℕ) :
  init_stones = 660 →
  (∀ x ∈ A, x > 0) →
  (∀ x y ∈ A, x ≤ y → y < 2 * x) →
  A.sum id = init_stones →
  A.card ≤ 30 :=
sorry

end max_piles_660_stones_l133_133767


namespace initial_crayons_per_box_l133_133611

-- Define the initial total number of crayons in terms of x
def total_initial_crayons (x : ℕ) : ℕ := 4 * x

-- Define the crayons given to Mae
def crayons_to_Mae : ℕ := 5

-- Define the crayons given to Lea
def crayons_to_Lea : ℕ := 12

-- Define the remaining crayons
def remaining_crayons : ℕ := 15

-- Prove that the initial number of crayons per box is 8 given the conditions
theorem initial_crayons_per_box (x : ℕ) : total_initial_crayons x - crayons_to_Mae - crayons_to_Lea = remaining_crayons → x = 8 :=
by
  intros h
  sorry

end initial_crayons_per_box_l133_133611


namespace negation_of_universal_l133_133903

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by
  sorry

end negation_of_universal_l133_133903


namespace steak_entree_cost_l133_133198

theorem steak_entree_cost
  (total_guests : ℕ)
  (steak_factor : ℕ)
  (chicken_entree_cost : ℕ)
  (total_budget : ℕ)
  (H1 : total_guests = 80)
  (H2 : steak_factor = 3)
  (H3 : chicken_entree_cost = 18)
  (H4 : total_budget = 1860) :
  ∃ S : ℕ, S = 25 := by
  -- Proof steps omitted
  sorry

end steak_entree_cost_l133_133198


namespace probability_correct_l133_133385

structure Bag :=
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

def marbles_drawn_sequence (bag : Bag) : ℚ :=
  let total_marbles := bag.blue + bag.green + bag.yellow
  let prob_blue_first := ↑bag.blue / total_marbles
  let prob_green_second := ↑bag.green / (total_marbles - 1)
  let prob_yellow_third := ↑bag.yellow / (total_marbles - 2)
  prob_blue_first * prob_green_second * prob_yellow_third

theorem probability_correct (bag : Bag) (h : bag = ⟨4, 6, 5⟩) : 
  marbles_drawn_sequence bag = 20 / 455 :=
by
  sorry

end probability_correct_l133_133385


namespace division_remainder_false_l133_133803

theorem division_remainder_false :
  ¬(1700 / 500 = 17 / 5 ∧ (1700 % 500 = 3 ∧ 17 % 5 = 2)) := by
  sorry

end division_remainder_false_l133_133803


namespace chocolate_bars_in_large_box_l133_133795

def num_small_boxes : ℕ := 17
def chocolate_bars_per_small_box : ℕ := 26
def total_chocolate_bars : ℕ := 17 * 26

theorem chocolate_bars_in_large_box :
  total_chocolate_bars = 442 :=
by
  sorry

end chocolate_bars_in_large_box_l133_133795


namespace age_proof_l133_133432

theorem age_proof (y d : ℕ)
  (h1 : y = 4 * d)
  (h2 : y - 7 = 11 * (d - 7)) :
  y = 48 ∧ d = 12 :=
by
  -- The proof is omitted
  sorry

end age_proof_l133_133432


namespace gcd_5670_9800_l133_133261

-- Define the two given numbers
def a := 5670
def b := 9800

-- State that the GCD of a and b is 70
theorem gcd_5670_9800 : Int.gcd a b = 70 := by
  sorry

end gcd_5670_9800_l133_133261


namespace triangle_area_correct_l133_133292

/-- Define the points of the triangle -/
def x1 : ℝ := -4
def y1 : ℝ := 2
def x2 : ℝ := 2
def y2 : ℝ := 8
def x3 : ℝ := -2
def y3 : ℝ := -2

/-- Define the area calculation function -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Define the area of the given triangle -/
def given_triangle_area : ℝ :=
  triangle_area x1 y1 x2 y2 x3 y3

/-- The goal is to prove that the area of the given triangle is 22 square units -/
theorem triangle_area_correct : given_triangle_area = 22 := by
  sorry

end triangle_area_correct_l133_133292


namespace determine_e_l133_133101

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the problem statement
theorem determine_e (d e f : ℝ)
  (h1 : f = 9)
  (h2 : (d * (d + 9)) - 168 = 0)
  (h3 : d^2 - 6 * e = 12 + d + e)
  : e = -24 ∨ e = 20 :=
by
  sorry

end determine_e_l133_133101


namespace playground_area_l133_133904

theorem playground_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 84 ∧ l = 3 * w ∧ l * w = 330.75 :=
by
  sorry

end playground_area_l133_133904


namespace total_crayons_is_12_l133_133915

-- Definitions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Goal to prove
theorem total_crayons_is_12 : initial_crayons + added_crayons = 12 :=
by
  sorry

end total_crayons_is_12_l133_133915


namespace jack_sees_color_change_l133_133958

noncomputable def traffic_light_cycle := 95    -- Total duration of the traffic light cycle
noncomputable def change_window := 15          -- Duration window where color change occurs
def observation_interval := 5                  -- Length of Jack's observation interval

/-- Probability that Jack sees the color change during his observation. -/
def probability_of_observing_change (cycle: ℕ) (window: ℕ) : ℚ :=
  window / cycle

theorem jack_sees_color_change :
  probability_of_observing_change traffic_light_cycle change_window = 3 / 19 :=
by
  -- We only need the statement for verification
  sorry

end jack_sees_color_change_l133_133958


namespace trig_expression_eval_l133_133551

theorem trig_expression_eval :
  (cos (30 * Real.pi / 180) * tan (60 * Real.pi / 180) - cos (45 * Real.pi / 180) ^ 2 + tan (45 * Real.pi / 180)) = 2 :=
by
  have h1 : cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2 := sorry
  have h2 : tan (60 * Real.pi / 180) = Real.sqrt 3 := sorry
  have h3 : cos (45 * Real.pi / 180) = (Real.sqrt 2) / 2 := sorry
  have h4 : tan (45 * Real.pi / 180) = 1 := sorry
  sorry

end trig_expression_eval_l133_133551


namespace compare_compound_interest_l133_133949

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l133_133949


namespace cost_of_orange_juice_l133_133637

theorem cost_of_orange_juice (O : ℝ) (H1 : ∀ (apple_juice_cost : ℝ), apple_juice_cost = 0.60 ):
  let total_bottles := 70
  let total_cost := 46.20
  let orange_juice_bottles := 42
  let apple_juice_bottles := total_bottles - orange_juice_bottles
  let equation := (orange_juice_bottles * O + apple_juice_bottles * 0.60 = total_cost)
  equation -> O = 0.70 := by
  sorry

end cost_of_orange_juice_l133_133637


namespace floor_equation_solution_l133_133990

theorem floor_equation_solution (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (⌊ (a^2 : ℝ) / b ⌋ + ⌊ (b^2 : ℝ) / a ⌋ = ⌊ (a^2 + b^2 : ℝ) / (a * b) ⌋ + a * b) ↔
    (∃ n : ℕ, a = n ∧ b = n^2 + 1) ∨ (∃ n : ℕ, a = n^2 + 1 ∧ b = n) :=
sorry

end floor_equation_solution_l133_133990


namespace find_certain_number_l133_133866

theorem find_certain_number (x certain_number : ℕ) (h1 : certain_number + x = 13200) (h2 : x = 3327) : certain_number = 9873 :=
by
  sorry

end find_certain_number_l133_133866


namespace number_of_unique_combinations_l133_133223

-- Define the inputs and the expected output.
def n := 8
def r := 3
def expected_combinations := 56

-- We state our theorem indicating that the combination of 8 toppings chosen 3 at a time
-- equals 56.
theorem number_of_unique_combinations :
  (Nat.choose n r = expected_combinations) :=
by
  sorry

end number_of_unique_combinations_l133_133223


namespace value_subtracted_l133_133485

theorem value_subtracted (n v : ℝ) (h1 : 2 * n - v = -12) (h2 : n = -10.0) : v = -8 :=
by
  sorry

end value_subtracted_l133_133485


namespace power_equality_l133_133334

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l133_133334


namespace rebus_solution_l133_133994

open Nat

theorem rebus_solution
  (A B C: ℕ)
  (hA: A ≠ 0)
  (hB: B ≠ 0)
  (hC: C ≠ 0)
  (distinct: A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (rebus: A * 101 + B * 110 + C * 11 + (A * 100 + B * 10 + 6) + (A * 100 + C * 10 + C) = 1416) :
  A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l133_133994


namespace find_a1_l133_133167

theorem find_a1 (a b : ℕ → ℝ) (h1 : ∀ n ≥ 1, a (n + 1) + b (n + 1) = (a n + b n) / 2) 
  (h2 : ∀ n ≥ 1, a (n + 1) * b (n + 1) = (a n * b n) ^ (1/2)) 
  (hb2016 : b 2016 = 1) (ha1_pos : a 1 > 0) :
  a 1 = 2^2015 :=
sorry

end find_a1_l133_133167


namespace inequality_solution_empty_l133_133592

theorem inequality_solution_empty {a x: ℝ} : 
  (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0 → 
  (-2 < a) ∧ (a < 6 / 5) :=
sorry

end inequality_solution_empty_l133_133592


namespace min_value_of_sum_l133_133347

theorem min_value_of_sum (a b : ℤ) (h : a * b = 150) : a + b = -151 :=
  sorry

end min_value_of_sum_l133_133347


namespace minimum_value_l133_133320

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) :
  (1 / x + 1 / y) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end minimum_value_l133_133320


namespace polygon_sides_l133_133500

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l133_133500


namespace Dana_pencils_equals_combined_l133_133555

-- Definitions based on given conditions
def pencils_Jayden : ℕ := 20
def pencils_Marcus (pencils_Jayden : ℕ) : ℕ := pencils_Jayden / 2
def pencils_Dana (pencils_Jayden : ℕ) : ℕ := pencils_Jayden + 15
def pencils_Ella (pencils_Marcus : ℕ) : ℕ := 3 * pencils_Marcus - 5
def combined_pencils (pencils_Marcus : ℕ) (pencils_Ella : ℕ) : ℕ := pencils_Marcus + pencils_Ella

-- Theorem to prove:
theorem Dana_pencils_equals_combined (pencils_Jayden : ℕ := 20) : 
  pencils_Dana pencils_Jayden = combined_pencils (pencils_Marcus pencils_Jayden) (pencils_Ella (pencils_Marcus pencils_Jayden)) := by
  sorry

end Dana_pencils_equals_combined_l133_133555


namespace arithmetic_sequence_sum_l133_133165

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 3 = 7) (h2 : a 5 + a 7 = 26) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((2 * n + 1)^2 - 1)) ∧
  (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_sum_l133_133165


namespace least_actual_square_area_l133_133762

theorem least_actual_square_area :
  let side_measured := 7
  let lower_bound := 6.5
  let actual_area := lower_bound * lower_bound
  actual_area = 42.25 :=
by
  sorry

end least_actual_square_area_l133_133762


namespace largest_of_consecutive_odds_l133_133907

-- Defining the six consecutive odd numbers
def consecutive_odd_numbers (a b c d e f : ℕ) : Prop :=
  (a = b + 2) ∧ (b = c + 2) ∧ (c = d + 2) ∧ (d = e + 2) ∧ (e = f + 2)

-- Defining the product condition
def product_of_odds (a b c d e f : ℕ) : Prop :=
  a * b * c * d * e * f = 135135

-- Defining the odd numbers greater than zero
def positive_odds (a b c d e f : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1)

-- Theorem
theorem largest_of_consecutive_odds (a b c d e f : ℕ) 
  (h1 : consecutive_odd_numbers a b c d e f)
  (h2 : product_of_odds a b c d e f)
  (h3 : positive_odds a b c d e f) : 
  a = 13 :=
sorry

end largest_of_consecutive_odds_l133_133907


namespace slower_pipe_filling_time_l133_133747

theorem slower_pipe_filling_time
  (t : ℝ)
  (H1 : ∀ (time_slow : ℝ), time_slow = t)
  (H2 : ∀ (time_fast : ℝ), time_fast = t / 3)
  (H3 : 1 / t + 1 / (t / 3) = 1 / 40) :
  t = 160 :=
sorry

end slower_pipe_filling_time_l133_133747


namespace range_of_a_l133_133175

theorem range_of_a (a : ℝ) (x : ℝ) : (∃ x, x^2 - a*x - a ≤ -3) → (a ≤ -6 ∨ a ≥ 2) :=
sorry

end range_of_a_l133_133175


namespace cost_of_6_bottle_caps_l133_133305

-- Define the cost of each bottle cap
def cost_per_bottle_cap : ℕ := 2

-- Define how many bottle caps we are buying
def number_of_bottle_caps : ℕ := 6

-- Define the total cost of the bottle caps
def total_cost : ℕ := 12

-- The proof statement to prove that the total cost is as expected
theorem cost_of_6_bottle_caps :
  cost_per_bottle_cap * number_of_bottle_caps = total_cost :=
by
  sorry

end cost_of_6_bottle_caps_l133_133305


namespace count_library_books_l133_133748

theorem count_library_books (initial_library_books : ℕ) 
  (books_given_away : ℕ) (books_added_from_source : ℕ) (books_donated : ℕ) 
  (h1 : initial_library_books = 125)
  (h2 : books_given_away = 42)
  (h3 : books_added_from_source = 68)
  (h4 : books_donated = 31) : 
  initial_library_books - books_given_away - books_donated = 52 :=
by sorry

end count_library_books_l133_133748


namespace ratio_of_ducks_l133_133227

theorem ratio_of_ducks (lily_ducks lily_geese rayden_geese rayden_ducks : ℕ) 
  (h1 : lily_ducks = 20) 
  (h2 : lily_geese = 10) 
  (h3 : rayden_geese = 4 * lily_geese) 
  (h4 : rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70) : 
  rayden_ducks / lily_ducks = 3 :=
by
  sorry

end ratio_of_ducks_l133_133227


namespace system1_solution_l133_133087

variable (x y : ℝ)

theorem system1_solution :
  (3 * x - y = -1) ∧ (x + 2 * y = 9) ↔ (x = 1) ∧ (y = 4) := by
  sorry

end system1_solution_l133_133087


namespace solution_set_of_inequality_l133_133311

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 12 < 0 } = { x : ℝ | -4 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l133_133311


namespace compound_interest_comparison_l133_133947

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l133_133947


namespace pablo_mother_pays_each_page_l133_133614

-- Definitions based on the conditions in the problem
def pages_per_book := 150
def number_books_read := 12
def candy_cost := 15
def money_leftover := 3
def total_money := candy_cost + money_leftover
def total_pages := number_books_read * pages_per_book
def amount_paid_per_page := total_money / total_pages

-- The theorem to be proven
theorem pablo_mother_pays_each_page
    (pages_per_book : ℝ)
    (number_books_read : ℝ)
    (candy_cost : ℝ)
    (money_leftover : ℝ)
    (total_money := candy_cost + money_leftover)
    (total_pages := number_books_read * pages_per_book)
    (amount_paid_per_page := total_money / total_pages) :
    amount_paid_per_page = 0.01 :=
by
  sorry

end pablo_mother_pays_each_page_l133_133614


namespace systematic_sampling_result_l133_133504

-- Define the set of bags numbered from 1 to 30
def bags : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the systematic sampling function
def systematic_sampling (n k interval : ℕ) : List ℕ :=
  List.range k |> List.map (λ i => n + i * interval)

-- Specific parameters for the problem
def number_of_bags := 30
def bags_drawn := 6
def interval := 5
def expected_samples := [2, 7, 12, 17, 22, 27]

-- Statement of the theorem
theorem systematic_sampling_result : 
  systematic_sampling 2 bags_drawn interval = expected_samples :=
by
  sorry

end systematic_sampling_result_l133_133504


namespace original_radius_eq_n_div_3_l133_133622

theorem original_radius_eq_n_div_3 (r n : ℝ) (h : (r + n)^2 = 4 * r^2) : r = n / 3 :=
by
  sorry

end original_radius_eq_n_div_3_l133_133622


namespace monotonic_intervals_and_non_negative_f_l133_133425

noncomputable def f (m x : ℝ) : ℝ := m / x - m + Real.log x

theorem monotonic_intervals_and_non_negative_f (m : ℝ) : 
  (∀ x > 0, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end monotonic_intervals_and_non_negative_f_l133_133425


namespace greatest_b_not_in_range_l133_133786

theorem greatest_b_not_in_range : ∃ b : ℤ, b = 10 ∧ ∀ x : ℝ, x^2 + (b:ℝ) * x + 20 ≠ -7 := sorry

end greatest_b_not_in_range_l133_133786


namespace successful_combinations_l133_133664

def herbs := 4
def gems := 6
def incompatible_combinations := 3

theorem successful_combinations : herbs * gems - incompatible_combinations = 21 := by
  sorry

end successful_combinations_l133_133664


namespace degrees_to_radians_neg_210_l133_133403

theorem degrees_to_radians_neg_210 :
  -210 * (Real.pi / 180) = - (7 / 6) * Real.pi :=
by
  sorry

end degrees_to_radians_neg_210_l133_133403


namespace probability_at_least_one_each_color_in_bag_l133_133018

open BigOperators

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_at_least_one_each_color : ℚ :=
  let total_ways := num_combinations 9 5
  let favorable_ways := 27 + 27 + 27 -- 3 scenarios (2R+1B+2G, 2B+1R+2G, 2G+1R+2B)
  favorable_ways / total_ways

theorem probability_at_least_one_each_color_in_bag :
  prob_at_least_one_each_color = 9 / 14 :=
by
  sorry

end probability_at_least_one_each_color_in_bag_l133_133018


namespace C_recurrence_S_recurrence_l133_133379

noncomputable def C (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def C_n (n : ℕ) (x : ℝ) : ℝ := 2 * Real.cos (n * x)
noncomputable def S_n (n : ℕ) (x : ℝ) : ℝ := Real.sin (n * x) / Real.sin x

theorem C_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  C_n n x = C x * C_n (n - 1) x - C_n (n - 2) x := sorry

theorem S_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  S_n n x = C x * S_n (n - 1) x - S_n (n - 2) x := sorry

end C_recurrence_S_recurrence_l133_133379


namespace units_digit_of_k_squared_plus_2_k_l133_133459

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end units_digit_of_k_squared_plus_2_k_l133_133459


namespace geometric_sequence_m_solution_l133_133182

theorem geometric_sequence_m_solution (m : ℝ) (h : ∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 4 ∧ a * c = b^2) :
  m = 2 ∨ m = -2 :=
by
  sorry

end geometric_sequence_m_solution_l133_133182


namespace words_lost_due_to_prohibition_l133_133531

-- Define the conditions given in the problem.
def number_of_letters := 64
def forbidden_letter := 7
def total_one_letter_words := number_of_letters
def total_two_letter_words := number_of_letters * number_of_letters

-- Define the forbidden letter loss calculation.
def one_letter_words_lost := 1
def two_letter_words_lost := number_of_letters + number_of_letters - 1

-- Define the total words lost calculation.
def total_words_lost := one_letter_words_lost + two_letter_words_lost

-- State the theorem to prove the number of words lost is 128.
theorem words_lost_due_to_prohibition : total_words_lost = 128 :=
by sorry

end words_lost_due_to_prohibition_l133_133531


namespace remaining_fruits_l133_133130

theorem remaining_fruits (initial_apples initial_oranges initial_mangoes taken_apples twice_taken_apples taken_mangoes) : 
  initial_apples = 7 → 
  initial_oranges = 8 → 
  initial_mangoes = 15 → 
  taken_apples = 2 → 
  twice_taken_apples = 2 * taken_apples → 
  taken_mangoes = 2 * initial_mangoes / 3 → 
  initial_apples - taken_apples + initial_oranges - twice_taken_apples + initial_mangoes - taken_mangoes = 14 :=
by
  sorry

end remaining_fruits_l133_133130


namespace number_of_k_values_l133_133567

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l133_133567


namespace area_ratio_greater_than_two_ninths_l133_133905

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]

def area_triangle (A B C : α) : α := sorry -- Placeholder for the area function
noncomputable def triangle_division (A B C P Q R : α) : Prop :=
  -- Placeholder for division condition
  -- Here you would check that P, Q, and R divide the perimeter of triangle ABC into three equal parts
  sorry

theorem area_ratio_greater_than_two_ninths (A B C P Q R : α) :
  triangle_division A B C P Q R → area_triangle P Q R > (2 / 9) * area_triangle A B C :=
by
  sorry -- The proof goes here

end area_ratio_greater_than_two_ninths_l133_133905


namespace iron_needed_for_hydrogen_l133_133152

-- Conditions of the problem
def reaction (Fe H₂SO₄ FeSO₄ H₂ : ℕ) : Prop :=
  Fe + H₂SO₄ = FeSO₄ + H₂

-- Given data
def balanced_equation : Prop :=
  reaction 1 1 1 1
 
def produced_hydrogen : ℕ := 2
def produced_from_sulfuric_acid : ℕ := 2
def needed_iron : ℕ := 2

-- Problem statement to be proved
theorem iron_needed_for_hydrogen (H₂SO₄ H₂ : ℕ) (h1 : produced_hydrogen = H₂) (h2 : produced_from_sulfuric_acid = H₂SO₄) (balanced_eq : balanced_equation) :
  needed_iron = 2 := by
sorry

end iron_needed_for_hydrogen_l133_133152


namespace calc_area_of_quadrilateral_l133_133960

-- Define the terms and conditions using Lean definitions
noncomputable def triangle_areas : ℕ × ℕ × ℕ := (6, 9, 15)

-- State the theorem
theorem calc_area_of_quadrilateral (a b c d : ℕ) (area1 area2 area3 : ℕ):
  area1 = 6 →
  area2 = 9 →
  area3 = 15 →
  a + b + c + d = area1 + area2 + area3 →
  d = 65 :=
  sorry

end calc_area_of_quadrilateral_l133_133960


namespace number_of_digits_in_N_l133_133625

noncomputable def N : ℕ := 2^12 * 5^8

theorem number_of_digits_in_N : (Nat.digits 10 N).length = 10 := by
  sorry

end number_of_digits_in_N_l133_133625


namespace how_many_peaches_l133_133064

-- Define the variables
variables (Jake Steven : ℕ)

-- Conditions
def has_fewer_peaches : Prop := Jake = Steven - 7
def jake_has_9_peaches : Prop := Jake = 9

-- The theorem that proves Steven's number of peaches
theorem how_many_peaches (Jake Steven : ℕ) (h1 : has_fewer_peaches Jake Steven) (h2 : jake_has_9_peaches Jake) : Steven = 16 :=
by
  -- Proof goes here
  sorry

end how_many_peaches_l133_133064


namespace cube_division_l133_133453

theorem cube_division (n : ℕ) (hn1 : 6 ≤ n) (hn2 : n % 2 = 0) : 
  ∃ m : ℕ, (n = 2 * m) ∧ (∀ a : ℕ, ∀ b : ℕ, ∀ c: ℕ, a = m^3 - (m - 1)^3 + 1 → b = 3 * m * (m - 1) + 2 → a = b) :=
by
  sorry

end cube_division_l133_133453


namespace sum_of_interior_angles_l133_133094

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1980) :
    180 * ((n + 3) - 2) = 2520 :=
by
  sorry

end sum_of_interior_angles_l133_133094


namespace original_cost_of_article_l133_133398

theorem original_cost_of_article (x: ℝ) (h: 0.76 * x = 320) : x = 421.05 :=
sorry

end original_cost_of_article_l133_133398


namespace problem_statement_l133_133183

theorem problem_statement
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023 * (c * d) - (a + b) / 20 = 2024 := 
by 
  sorry

end problem_statement_l133_133183


namespace order_of_numbers_l133_133801

noncomputable def a : ℝ := 60.7
noncomputable def b : ℝ := 0.76
noncomputable def c : ℝ := Real.log 0.76

theorem order_of_numbers : (c < b) ∧ (b < a) :=
by
  have h1 : c = Real.log 0.76 := rfl
  have h2 : b = 0.76 := rfl
  have h3 : a = 60.7 := rfl
  have hc : c < 0 := sorry
  have hb : 0 < b := sorry
  have ha : 1 < a := sorry
  sorry 

end order_of_numbers_l133_133801


namespace election_candidate_a_votes_l133_133124

theorem election_candidate_a_votes :
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  candidate_a_votes = 333200 :=
by
  let total_votes : ℕ := 560000
  let invalid_percentage : ℚ := 15 / 100
  let candidate_a_percentage : ℚ := 70 / 100
  let total_valid_votes := total_votes * (1 - invalid_percentage)
  let candidate_a_votes := total_valid_votes * candidate_a_percentage
  show candidate_a_votes = 333200
  sorry

end election_candidate_a_votes_l133_133124


namespace compute_fraction_sum_l133_133003

theorem compute_fraction_sum :
  8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 :=
by
  sorry

end compute_fraction_sum_l133_133003


namespace triangle_with_sticks_l133_133667

theorem triangle_with_sticks (c : ℕ) (h₁ : 4 + 9 > c) (h₂ : 9 - 4 < c) :
  c = 9 :=
by
  sorry

end triangle_with_sticks_l133_133667


namespace find_N_l133_133713

theorem find_N :
  ∃ N : ℕ,
  (5 + 6 + 7 + 8 + 9) / 5 = (2005 + 2006 + 2007 + 2008 + 2009) / (N : ℝ) ∧ N = 1433 :=
sorry

end find_N_l133_133713


namespace find_a_l133_133854

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x * x - 4 <= 0) → (2 * x + a <= 0)) ↔ (a = -4) := by
  sorry

end find_a_l133_133854


namespace minimum_value_F_l133_133639

noncomputable def minimum_value_condition (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 10 * (x + y)

noncomputable def F (x y : ℝ) : ℝ :=
  6 * y + 8 * x - 9

theorem minimum_value_F :
  (∃ x y : ℝ, minimum_value_condition x y) → ∃ x y : ℝ, minimum_value_condition x y ∧ F x y = 11 :=
sorry

end minimum_value_F_l133_133639


namespace gym_monthly_income_l133_133391

theorem gym_monthly_income (bi_monthly_charge : ℕ) (members : ℕ) (monthly_income : ℕ) 
  (h1 : bi_monthly_charge = 18)
  (h2 : members = 300)
  (h3 : monthly_income = 10800) : 
  2 * bi_monthly_charge * members = monthly_income :=
by
  rw [h1, h2, h3]
  norm_num

end gym_monthly_income_l133_133391


namespace problem1_problem2_problem3_problem4_l133_133000

-- Problem 1
theorem problem1 : (-3 / 8) + ((-5 / 8) * (-6)) = 27 / 8 :=
by sorry

-- Problem 2
theorem problem2 : 12 + (7 * (-3)) - (18 / (-3)) = -3 :=
by sorry

-- Problem 3
theorem problem3 : -((2:ℤ)^2) - (4 / 7) * (2:ℚ) - (-((3:ℤ)^2:ℤ) : ℤ) = -99 / 7 :=
by sorry

-- Problem 4
theorem problem4 : -(((-1) ^ 2020 : ℤ)) + ((6 : ℚ) / (-(2 : ℤ) ^ 3)) * (-1 / 3) = -3 / 4 :=
by sorry

end problem1_problem2_problem3_problem4_l133_133000


namespace combination_sum_l133_133672

theorem combination_sum : Nat.choose 10 3 + Nat.choose 10 4 = 330 := 
by
  sorry

end combination_sum_l133_133672


namespace average_pages_per_day_l133_133933

variable (total_pages : ℕ := 160)
variable (pages_read : ℕ := 60)
variable (days_left : ℕ := 5)

theorem average_pages_per_day : (total_pages - pages_read) / days_left = 20 := by
  sorry

end average_pages_per_day_l133_133933


namespace inequality_always_holds_l133_133693

theorem inequality_always_holds (a b : ℝ) (h : a * b > 0) : (b / a + a / b) ≥ 2 :=
sorry

end inequality_always_holds_l133_133693


namespace find_sum_of_smallest_multiples_l133_133185

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l133_133185


namespace wire_division_l133_133119

theorem wire_division (L leftover total_length : ℝ) (seg1 seg2 : ℝ)
  (hL : L = 120 * 2)
  (hleftover : leftover = 2.4)
  (htotal : total_length = L + leftover)
  (hseg1 : seg1 = total_length / 3)
  (hseg2 : seg2 = total_length / 3) :
  seg1 = 80.8 ∧ seg2 = 80.8 := by
  sorry

end wire_division_l133_133119


namespace right_triangle_area_l133_133190

theorem right_triangle_area (a b c : ℝ) (h : c = 5) (h1 : a = 3) (h2 : c^2 = a^2 + b^2) : 
  1 / 2 * a * b = 6 :=
by
  sorry

end right_triangle_area_l133_133190


namespace no_real_a_values_l133_133991

noncomputable def polynomial_with_no_real_root (a : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 ≠ 0
  
theorem no_real_a_values :
  ∀ a : ℝ, (∃ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 = 0) → false :=
by sorry

end no_real_a_values_l133_133991


namespace function_cannot_be_decreasing_if_f1_lt_f2_l133_133427

variable (f : ℝ → ℝ)

theorem function_cannot_be_decreasing_if_f1_lt_f2
  (h : f 1 < f 2) : ¬ (∀ x y, x < y → f y < f x) :=
by
  sorry

end function_cannot_be_decreasing_if_f1_lt_f2_l133_133427


namespace common_ratio_is_two_l133_133414

-- Geometric sequence definition
noncomputable def common_ratio (n : ℕ) (a : ℕ → ℝ) : ℝ :=
a 2 / a 1

-- The sequence has 10 terms
def ten_terms (a : ℕ → ℝ) : Prop :=
∀ n, 1 ≤ n ∧ n ≤ 10

-- The product of the odd terms is 2
def product_of_odd_terms (a : ℕ → ℝ) : Prop :=
(a 1) * (a 3) * (a 5) * (a 7) * (a 9) = 2

-- The product of the even terms is 64
def product_of_even_terms (a : ℕ → ℝ) : Prop :=
(a 2) * (a 4) * (a 6) * (a 8) * (a 10) = 64

-- The problem statement to prove that the common ratio q is 2
theorem common_ratio_is_two (a : ℕ → ℝ) (q : ℝ) (h1 : ten_terms a) 
(h2 : product_of_odd_terms a) (h3 : product_of_even_terms a) : q = 2 :=
by {
  sorry
}

end common_ratio_is_two_l133_133414


namespace find_b_l133_133823

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3 * x^2
noncomputable def tangent_line (x b : ℝ) : ℝ := -3 * x + b

theorem find_b
  (b : ℝ)
  (h : ∃ x : ℝ, curve x = tangent_line x b ∧ deriv curve x = -3) :
  b = 1 :=
by
  sorry

end find_b_l133_133823


namespace max_angle_MPN_is_pi_over_2_l133_133831

open Real

noncomputable def max_angle_MPN (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : ℝ :=
  sorry

theorem max_angle_MPN_is_pi_over_2 (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : 
  max_angle_MPN θ P hP = π / 2 :=
sorry

end max_angle_MPN_is_pi_over_2_l133_133831


namespace linear_system_sum_l133_133313

theorem linear_system_sum (x y : ℝ) 
  (h1: x - y = 2) 
  (h2: y = 2): 
  x + y = 6 := 
sorry

end linear_system_sum_l133_133313


namespace casey_nail_decorating_time_l133_133294

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l133_133294


namespace find_a6_l133_133860

theorem find_a6 (a : ℕ → ℚ) (h₁ : ∀ n, a (n + 1) = 2 * a n - 1) (h₂ : a 8 = 16) : a 6 = 19 / 4 :=
sorry

end find_a6_l133_133860


namespace cube_and_fourth_power_remainders_l133_133737

theorem cube_and_fourth_power_remainders (
  b : Fin 2018 → ℕ) 
  (h1 : StrictMono b) 
  (h2 : (Finset.univ.sum b) = 2018^3) :
  ((Finset.univ.sum (λ i => b i ^ 3)) % 5 = 3) ∧
  ((Finset.univ.sum (λ i => b i ^ 4)) % 5 = 1) := 
sorry

end cube_and_fourth_power_remainders_l133_133737


namespace time_spent_cutting_hair_l133_133448

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l133_133448


namespace num_k_values_lcm_l133_133565

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l133_133565


namespace Rachel_father_age_when_Rachel_is_25_l133_133477

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l133_133477


namespace books_left_over_after_repacking_l133_133438

theorem books_left_over_after_repacking :
  ((1335 * 39) % 40) = 25 :=
sorry

end books_left_over_after_repacking_l133_133438


namespace max_piles_660_stones_l133_133778

-- Define the conditions in Lean
def initial_stones := 660

def valid_pile_sizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → a ≤ b → b < 2 * a

-- Define the goal statement in Lean
theorem max_piles_660_stones :
  ∃ (piles : List ℕ), (piles.length = 30) ∧ (piles.sum = initial_stones) ∧ valid_pile_sizes piles :=
sorry

end max_piles_660_stones_l133_133778


namespace initial_people_count_l133_133362

theorem initial_people_count (C : ℝ) (n : ℕ) (h : n > 1) :
  ((C / (n - 1)) - (C / n) = 0.125) →
  n = 8 := by
  sorry

end initial_people_count_l133_133362


namespace smallest_rectangles_required_l133_133527

theorem smallest_rectangles_required :
  ∀ (r h : ℕ) (area_square length_square : ℕ),
  r = 3 → h = 4 →
  (∀ k, (k: ℕ) ∣ (r * h) → (k: ℕ) = r * h) →
  length_square = 12 →
  area_square = length_square * length_square →
  (area_square / (r * h) = 12) :=
by
  intros
  /- The mathematical proof steps will be filled here -/
  sorry

end smallest_rectangles_required_l133_133527


namespace james_ride_time_l133_133449

theorem james_ride_time :
  let distance := 80 
  let speed := 16 
  distance / speed = 5 := 
by
  -- sorry to skip the proof
  sorry

end james_ride_time_l133_133449


namespace hawkeye_remaining_money_l133_133580

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end hawkeye_remaining_money_l133_133580


namespace find_x_l133_133792

theorem find_x (x : ℝ) : (x / 18) * (36 / 72) = 1 → x = 36 :=
by
  intro h
  sorry

end find_x_l133_133792


namespace birches_planted_l133_133912

variable 
  (G B X : ℕ) -- G: number of girls, B: number of boys, X: number of birches

-- Conditions:
variable
  (h1 : G + B = 24) -- Total number of students
  (h2 : 3 * G + X = 24) -- Total number of plants
  (h3 : X = B / 3) -- Birches planted by boys

-- Proof statement:
theorem birches_planted : X = 6 :=
by 
  sorry

end birches_planted_l133_133912


namespace graph_does_not_pass_through_quadrant_II_l133_133488

noncomputable def linear_function (x : ℝ) : ℝ := 3 * x - 4

def passes_through_quadrant_I (x : ℝ) : Prop := x > 0 ∧ linear_function x > 0
def passes_through_quadrant_II (x : ℝ) : Prop := x < 0 ∧ linear_function x > 0
def passes_through_quadrant_III (x : ℝ) : Prop := x < 0 ∧ linear_function x < 0
def passes_through_quadrant_IV (x : ℝ) : Prop := x > 0 ∧ linear_function x < 0

theorem graph_does_not_pass_through_quadrant_II :
  ¬(∃ x : ℝ, passes_through_quadrant_II x) :=
sorry

end graph_does_not_pass_through_quadrant_II_l133_133488


namespace joe_spent_255_minutes_l133_133445

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l133_133445


namespace greatest_3_digit_base9_div_by_7_l133_133105

def base9_to_decimal (n : ℕ) : ℕ :=
  let d2 := n / 81
  let d1 := (n % 81) / 9
  let d0 := n % 9
  d2 * 81 + d1 * 9 + d0

def greatest_base9_3_digit_div_by_7 (n : ℕ) : Prop :=
  n < 9 * 9 * 9 ∧ 7 ∣ (base9_to_decimal n)

theorem greatest_3_digit_base9_div_by_7 :
  ∃ n, greatest_base9_3_digit_div_by_7 n ∧ n = 888 :=
begin
  sorry
end

end greatest_3_digit_base9_div_by_7_l133_133105


namespace hanoi_moves_minimal_l133_133139

theorem hanoi_moves_minimal (n : ℕ) : ∃ m, 
  (∀ move : ℕ, move = 2^n - 1 → move = m) := 
by
  sorry

end hanoi_moves_minimal_l133_133139


namespace series_converges_to_half_l133_133402

noncomputable def series_value : ℝ :=
  ∑' (n : ℕ), (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

theorem series_converges_to_half : series_value = 1 / 2 :=
  sorry

end series_converges_to_half_l133_133402


namespace monotonicity_of_f_l133_133404

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → 0 < a → f a x1 < f a x2) ∧
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → a < 0 → f a x1 > f a x2) :=
by {
  sorry
}

end monotonicity_of_f_l133_133404


namespace solve_for_a_l133_133049

theorem solve_for_a (a b : ℝ) (h₁ : b = 4 * a) (h₂ : b = 20 - 7 * a) : a = 20 / 11 :=
by
  sorry

end solve_for_a_l133_133049


namespace total_lunch_bill_l133_133228

theorem total_lunch_bill (hotdog salad : ℝ) (h1 : hotdog = 5.36) (h2 : salad = 5.10) : hotdog + salad = 10.46 := 
by
  rw [h1, h2]
  norm_num
  

end total_lunch_bill_l133_133228


namespace complex_fraction_sum_zero_l133_133849

section complex_proof
open Complex

theorem complex_fraction_sum_zero (z1 z2 : ℂ) (hz1 : z1 = 1 + I) (hz2 : z2 = 1 - I) :
  (z1 / z2) + (z2 / z1) = 0 := by
  sorry
end complex_proof

end complex_fraction_sum_zero_l133_133849


namespace fencing_rate_3_rs_per_meter_l133_133233

noncomputable def rate_per_meter (A_hectares : ℝ) (total_cost : ℝ) : ℝ := 
  let A_m2 := A_hectares * 10000
  let r := Real.sqrt (A_m2 / Real.pi)
  let C := 2 * Real.pi * r
  total_cost / C

theorem fencing_rate_3_rs_per_meter : rate_per_meter 17.56 4456.44 = 3.00 :=
by 
  sorry

end fencing_rate_3_rs_per_meter_l133_133233


namespace volume_of_prism_l133_133894

variables (a b : ℝ) (α β : ℝ)
  (h1 : a > b)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)

noncomputable def volume_prism : ℝ :=
  (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β

theorem volume_of_prism (a b α β : ℝ) (h1 : a > b) (h2 : 0 < α ∧ α < π / 2) (h3 : 0 < β ∧ β < π / 2) :
  volume_prism a b α β = (a^2 - b^2) * (a - b) / 8 * (Real.tan α)^2 * Real.tan β := by
  sorry

end volume_of_prism_l133_133894


namespace parabola_y_intercepts_l133_133430

theorem parabola_y_intercepts :
  let f : ℝ → ℝ := λ y, 3 * y^2 - 5 * y + 2
  ∃ y1 y2 : ℝ, f y1 = 0 ∧ f y2 = 0 ∧ y1 ≠ y2 :=
by
  sorry

end parabola_y_intercepts_l133_133430


namespace part_a_l133_133135

theorem part_a (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, a (n + 2) = a (n + 1) * a n + 1) :
  ∀ n, ¬ (4 ∣ a n) :=
by
  sorry

end part_a_l133_133135


namespace root_equation_l133_133742

theorem root_equation (p q : ℝ) (hp : 3 * p^2 - 5 * p - 7 = 0)
                                  (hq : 3 * q^2 - 5 * q - 7 = 0) :
            (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := 
by sorry

end root_equation_l133_133742


namespace williams_farm_tax_l133_133842

variables (T : ℝ)
variables (tax_collected : ℝ := 3840)
variables (percentage_williams_land : ℝ := 0.5)
variables (percentage_taxable_land : ℝ := 0.25)

theorem williams_farm_tax : (percentage_williams_land * tax_collected) = 1920 := by
  sorry

end williams_farm_tax_l133_133842


namespace max_piles_660_stones_l133_133766

theorem max_piles_660_stones (init_stones : ℕ) (A : finset ℕ) :
  init_stones = 660 →
  (∀ x ∈ A, x > 0) →
  (∀ x y ∈ A, x ≤ y → y < 2 * x) →
  A.sum id = init_stones →
  A.card ≤ 30 :=
sorry

end max_piles_660_stones_l133_133766


namespace first_pipe_fill_time_l133_133395

theorem first_pipe_fill_time 
  (T : ℝ)
  (h1 : 48 * (1 / T - 1 / 24) + 18 * (1 / T) = 1) :
  T = 22 :=
by
  sorry

end first_pipe_fill_time_l133_133395


namespace students_without_A_l133_133593

theorem students_without_A 
  (total_students : ℕ) 
  (A_in_literature : ℕ) 
  (A_in_science : ℕ) 
  (A_in_both : ℕ) 
  (h_total_students : total_students = 35)
  (h_A_in_literature : A_in_literature = 10)
  (h_A_in_science : A_in_science = 15)
  (h_A_in_both : A_in_both = 5) :
  total_students - (A_in_literature + A_in_science - A_in_both) = 15 :=
by {
  sorry
}

end students_without_A_l133_133593


namespace problem1_problem2_l133_133020

def count_good_subsets (n : ℕ) : ℕ := 
if n % 2 = 1 then 2^(n - 1) 
else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2)

def sum_f_good_subsets (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2)

theorem problem1 (n : ℕ)  :
  (count_good_subsets n = (if n % 2 = 1 then 2^(n - 1) else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2))) :=
sorry

theorem problem2 (n : ℕ) :
  (sum_f_good_subsets n = (if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
  else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2))) := 
sorry

end problem1_problem2_l133_133020


namespace log_50_between_consecutive_integers_l133_133632

theorem log_50_between_consecutive_integers :
    (∃ (m n : ℤ), m < n ∧ m < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3) :=
by
  have log_10_eq_1 : Real.log 10 / Real.log 10 = 1 := by sorry
  have log_100_eq_2 : Real.log 100 / Real.log 10 = 2 := by sorry
  have log_increasing : ∀ (x y : ℝ), x < y → Real.log x / Real.log 10 < Real.log y / Real.log 10 := by sorry
  have interval : 10 < 50 ∧ 50 < 100 := by sorry
  use 1
  use 2
  sorry

end log_50_between_consecutive_integers_l133_133632


namespace multiple_of_area_l133_133092

-- Define the given conditions
def perimeter (s : ℝ) : ℝ := 4 * s
def area (s : ℝ) : ℝ := s * s

theorem multiple_of_area (m s a p : ℝ) 
  (h1 : p = perimeter s)
  (h2 : a = area s)
  (h3 : m * a = 10 * p + 45)
  (h4 : p = 36) : m = 5 :=
by 
  sorry

end multiple_of_area_l133_133092


namespace farmer_land_area_l133_133654

theorem farmer_land_area
  (A : ℝ)
  (h1 : A / 3 + A / 4 + A / 5 + 26 = A) : A = 120 :=
sorry

end farmer_land_area_l133_133654


namespace derivative_y_l133_133843

noncomputable def y (x : ℝ) : ℝ := 
  Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

variable {x : ℝ}

theorem derivative_y :
  2 * x + 3 > 0 → 
  HasDerivAt y (4 * Real.sqrt (x^2 + 3 * x + 2) / (2 * x + 3)) x :=
by 
  sorry

end derivative_y_l133_133843


namespace percent_problem_l133_133586

theorem percent_problem (x : ℝ) (h : 0.35 * 400 = 0.20 * x) : x = 700 :=
by sorry

end percent_problem_l133_133586


namespace find_m_l133_133042

-- Definitions of the given vectors and their properties
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Condition that vectors a and b are parallel
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 = 0

-- Goal: Find the value of m such that vectors a and b are parallel
theorem find_m (m : ℝ) : 
  are_parallel a (b m) → m = 6 :=
by
  sorry

end find_m_l133_133042


namespace largest_digit_divisible_by_6_l133_133523

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l133_133523


namespace pauls_plumbing_hourly_charge_l133_133221

theorem pauls_plumbing_hourly_charge :
  ∀ P : ℕ,
  (55 + 4 * P = 75 + 4 * 30) → 
  P = 35 :=
by
  intros P h
  sorry

end pauls_plumbing_hourly_charge_l133_133221


namespace greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l133_133126

theorem greatest_divisor_of_product_of_5_consecutive_multiples_of_4 :
  let n1 := 4
  let n2 := 8
  let n3 := 12
  let n4 := 16
  let n5 := 20
  let spf1 := 2 -- smallest prime factor of 4
  let spf2 := 2 -- smallest prime factor of 8
  let spf3 := 2 -- smallest prime factor of 12
  let spf4 := 2 -- smallest prime factor of 16
  let spf5 := 2 -- smallest prime factor of 20
  let p1 := n1^spf1
  let p2 := n2^spf2
  let p3 := n3^spf3
  let p4 := n4^spf4
  let p5 := n5^spf5
  let product := p1 * p2 * p3 * p4 * p5
  product % (2^24) = 0 :=
by 
  sorry

end greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l133_133126


namespace elisa_improvement_l133_133406

theorem elisa_improvement (cur_laps cur_minutes prev_laps prev_minutes : ℕ) 
  (h1 : cur_laps = 15) (h2 : cur_minutes = 30) 
  (h3 : prev_laps = 20) (h4 : prev_minutes = 50) : 
  ((prev_minutes / prev_laps : ℚ) - (cur_minutes / cur_laps : ℚ) = 0.5) :=
by
  sorry

end elisa_improvement_l133_133406


namespace rhombus_area_l133_133367

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 24 :=
by {
  sorry
}

end rhombus_area_l133_133367


namespace ranking_of_scores_l133_133722

-- Let the scores of Ann, Bill, Carol, and Dick be A, B, C, and D respectively.

variables (A B C D : ℝ)

-- Conditions
axiom cond1 : B + D = A + C
axiom cond2 : C + B > D + A
axiom cond3 : C > A + B

-- Statement of the problem
theorem ranking_of_scores : C > D ∧ D > B ∧ B > A :=
by
  -- Placeholder for proof (proof steps aren't required)
  sorry

end ranking_of_scores_l133_133722


namespace volume_of_prism_l133_133635

theorem volume_of_prism 
  (a b c : ℝ) 
  (h₁ : a * b = 51) 
  (h₂ : b * c = 52) 
  (h₃ : a * c = 53) 
  : (a * b * c) = 374 :=
by sorry

end volume_of_prism_l133_133635


namespace total_amount_l133_133956

-- Define the conditions in Lean
variables (X Y Z: ℝ)
variable (h1 : Y = 0.75 * X)
variable (h2 : Z = (2/3) * X)
variable (h3 : Y = 48)

-- The theorem stating that the total amount of money is Rs. 154.67
theorem total_amount (X Y Z : ℝ) (h1 : Y = 0.75 * X) (h2 : Z = (2/3) * X) (h3 : Y = 48) : 
  X + Y + Z = 154.67 := 
by
  sorry

end total_amount_l133_133956


namespace non_deg_ellipse_b_l133_133303

theorem non_deg_ellipse_b (b : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = b ∧ (∀ x y : ℝ, (x - 3)^2 + 9*(y + 3/2)^2 = b + 145/4)) → b > -145/4 :=
sorry

end non_deg_ellipse_b_l133_133303


namespace polynomial_identity_l133_133755

theorem polynomial_identity 
  (P : Polynomial ℤ)
  (a b : ℤ) 
  (h_distinct : a ≠ b)
  (h_eq : P.eval a * P.eval b = -(a - b) ^ 2) : 
  P.eval a + P.eval b = 0 := 
by
  sorry

end polynomial_identity_l133_133755


namespace problem_statement_l133_133631

noncomputable def middle_of_three_consecutive (x : ℕ) : ℕ :=
  let y := x + 1
  let z := x + 2
  y

theorem problem_statement :
  ∃ x : ℕ, 
    (x + (x + 1) = 18) ∧ 
    (x + (x + 2) = 20) ∧ 
    ((x + 1) + (x + 2) = 23) ∧ 
    (middle_of_three_consecutive x = 7) :=
by
  sorry

end problem_statement_l133_133631


namespace find_digits_l133_133997

theorem find_digits (A B C : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : B ≠ C) :
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ↔ (A = 4 ∧ B = 7 ∧ C = 6) :=
begin
  sorry
end

end find_digits_l133_133997


namespace find_number_of_sides_l133_133498

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l133_133498


namespace smallest_k_divides_polynomial_l133_133837

theorem smallest_k_divides_polynomial :
  ∃ (k : ℕ), k > 0 ∧ (∀ z : ℂ, z ≠ 0 → 
    (z ^ 11 + z ^ 9 + z ^ 7 + z ^ 6 + z ^ 5 + z ^ 2 + 1) ∣ (z ^ k - 1)) ∧ k = 11 := by
  sorry

end smallest_k_divides_polynomial_l133_133837


namespace real_solution_exists_l133_133014

theorem real_solution_exists : ∃ x : ℝ, x^3 + (x+1)^4 + (x+2)^3 = (x+3)^4 :=
sorry

end real_solution_exists_l133_133014


namespace minimum_value_of_f_l133_133099

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 4 * y * z + 2 * z * x - 6 * x - 10 * y - 12 * z

theorem minimum_value_of_f : ∃ x y z : ℝ, f x y z = -14 :=
by
  sorry

end minimum_value_of_f_l133_133099


namespace bank_teller_rolls_of_coins_l133_133680

theorem bank_teller_rolls_of_coins (tellers : ℕ) (coins_per_roll : ℕ) (total_coins : ℕ) (h_tellers : tellers = 4) (h_coins_per_roll : coins_per_roll = 25) (h_total_coins : total_coins = 1000) : 
  (total_coins / tellers) / coins_per_roll = 10 :=
by 
  sorry

end bank_teller_rolls_of_coins_l133_133680


namespace perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l133_133707

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Problem statement for part (I)
theorem perp_bisector_eq : ∃ (k m: ℝ), 3 * k - 4 * m - 23 = 0 :=
sorry

-- Problem statement for part (II)
theorem parallel_line_eq : ∃ (k m: ℝ), 4 * k + 3 * m + 1 = 0 :=
sorry

-- Problem statement for part (III)
theorem reflected_ray_eq : ∃ (k m: ℝ), 11 * k + 27 * m + 74 = 0 :=
sorry

end perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l133_133707


namespace determine_sum_of_squares_l133_133008

theorem determine_sum_of_squares
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : x * y * z = 72)
  (h3 : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := 
sorry

end determine_sum_of_squares_l133_133008


namespace tangent_line_at_1_1_of_x_pow_x_l133_133025

theorem tangent_line_at_1_1_of_x_pow_x :
  ∀ x : ℝ, 0 < x →
  let y := x^x in
  let f := λ x : ℝ, x in
  let φ := λ x : ℝ, x in
  deriv y 1 = 1 ∧ (y = x := f 1, φ 1) :=
by
  sorry

end tangent_line_at_1_1_of_x_pow_x_l133_133025


namespace calculate_mirror_area_l133_133217

def outer_frame_width : ℝ := 65
def outer_frame_height : ℝ := 85
def frame_width : ℝ := 15

def mirror_width : ℝ := outer_frame_width - 2 * frame_width
def mirror_height : ℝ := outer_frame_height - 2 * frame_width
def mirror_area : ℝ := mirror_width * mirror_height

theorem calculate_mirror_area : mirror_area = 1925 := by
  sorry

end calculate_mirror_area_l133_133217


namespace g_g_2_equals_226_l133_133865

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

theorem g_g_2_equals_226 : g (g 2) = 226 := by
  sorry

end g_g_2_equals_226_l133_133865


namespace largest_digit_divisible_by_6_l133_133517

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l133_133517


namespace frank_cookies_l133_133022

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l133_133022


namespace number_of_k_values_l133_133566

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l133_133566


namespace total_voters_in_districts_l133_133102

theorem total_voters_in_districts : 
  ∀ (D1 D2 D3 : ℕ),
  (D1 = 322) →
  (D2 = D3 - 19) →
  (D3 = 2 * D1) →
  (D1 + D2 + D3 = 1591) :=
by
  intros D1 D2 D3 h1 h2 h3
  sorry

end total_voters_in_districts_l133_133102


namespace daily_chicken_loss_l133_133660

/--
A small poultry farm has initially 300 chickens, 200 turkeys, and 80 guinea fowls. Every day, the farm loses some chickens, 8 turkeys, and 5 guinea fowls. After one week (7 days), there are 349 birds left in the farm. Prove the number of chickens the farmer loses daily.
-/
theorem daily_chicken_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss days total_birds_left : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : days = 7)
  (h7 : total_birds_left = 349)
  (h8 : initial_chickens + initial_turkeys + initial_guinea_fowls
       - (daily_turkey_loss * days + daily_guinea_fowl_loss * days + (initial_chickens - total_birds_left)) = total_birds_left) :
  initial_chickens - (total_birds_left + daily_turkey_loss * days + daily_guinea_fowl_loss * days) / days = 20 :=
by {
    -- Proof goes here
    sorry
}

end daily_chicken_loss_l133_133660


namespace sally_total_expense_l133_133360

-- Definitions based on the problem conditions
def peaches_price_after_coupon : ℝ := 12.32
def peaches_coupon : ℝ := 3.00
def cherries_weight : ℝ := 2.00
def cherries_price_per_kg : ℝ := 11.54
def apples_weight : ℝ := 4.00
def apples_price_per_kg : ℝ := 5.00
def apples_discount_percentage : ℝ := 0.15
def oranges_count : ℝ := 6.00
def oranges_price_per_unit : ℝ := 1.25
def oranges_promotion : ℝ := 3.00 -- Buy 2, get 1 free means she pays for 4 out of 6

-- Calculation of the total expense
def total_expense : ℝ :=
  (peaches_price_after_coupon + peaches_coupon) + 
  (cherries_weight * cherries_price_per_kg) + 
  ((apples_weight * apples_price_per_kg) * (1 - apples_discount_percentage)) +
  (4 * oranges_price_per_unit)

-- Statement to verify total expense
theorem sally_total_expense : total_expense = 60.40 := by
  sorry

end sally_total_expense_l133_133360


namespace sin_double_angle_identity_l133_133677

theorem sin_double_angle_identity: 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_double_angle_identity_l133_133677


namespace range_of_a_l133_133325

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 → f a x1 ≥ g x2) ↔ a ≥ -2 :=
by
  sorry

end range_of_a_l133_133325


namespace Lucas_age_in_3_years_l133_133091

variable (Gladys Billy Lucas : ℕ)

theorem Lucas_age_in_3_years :
  Gladys = 30 ∧ Billy = Gladys / 3 ∧ Gladys = 2 * (Billy + Lucas) →
  Lucas + 3 = 8 :=
by
  intro h
  cases h with h1 h2
  cases h2 with hBilly h3
  sorry

end Lucas_age_in_3_years_l133_133091


namespace find_e_l133_133876

theorem find_e (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + b = 32)
  (h6 : a + c = 36)
  (h7 : b + c = 37)
  (h8 : c + e = 48)
  (h9 : d + e = 51) : e = 55 / 2 :=
  sorry

end find_e_l133_133876


namespace max_value_of_k_l133_133454

theorem max_value_of_k (m : ℝ) (k : ℝ) (h1 : 0 < m) (h2 : m < 1/2) 
  (h3 : ∀ m, 0 < m → m < 1/2 → (1 / m + 2 / (1 - 2 * m) ≥ k)) : k = 8 :=
sorry

end max_value_of_k_l133_133454


namespace vector_t_solution_l133_133342

theorem vector_t_solution (t : ℝ) :
  ∃ t, (∃ (AB AC BC : ℝ × ℝ), 
         AB = (t, 1) ∧ AC = (2, 2) ∧ BC = (2 - t, 1) ∧ 
         (AC.1 - AB.1) * AC.1 + (AC.2 - AB.2) * AC.2 = 0 ) → 
         t = 3 :=
by {
  sorry -- proof content omitted as per instructions
}

end vector_t_solution_l133_133342


namespace fibonacci_coprime_l133_133226

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_coprime (n : ℕ) (hn : n ≥ 1) :
  Nat.gcd (fibonacci n) (fibonacci (n - 1)) = 1 := by
  sorry

end fibonacci_coprime_l133_133226


namespace largest_of_given_numbers_l133_133529

theorem largest_of_given_numbers :
  (0.99 > 0.9099) ∧
  (0.99 > 0.9) ∧
  (0.99 > 0.909) ∧
  (0.99 > 0.9009) →
  ∀ (x : ℝ), (x = 0.99 ∨ x = 0.9099 ∨ x = 0.9 ∨ x = 0.909 ∨ x = 0.9009) → 
  x ≤ 0.99 :=
by
  sorry

end largest_of_given_numbers_l133_133529


namespace algebraic_expression_simplification_l133_133692

theorem algebraic_expression_simplification (x y : ℝ) (h : x + y = 1) : x^3 + y^3 + 3 * x * y = 1 := 
by
  sorry

end algebraic_expression_simplification_l133_133692


namespace average_disk_space_per_minute_l133_133653

theorem average_disk_space_per_minute 
  (days : ℕ := 15) 
  (disk_space : ℕ := 36000) 
  (minutes_per_day : ℕ := 1440) 
  (total_minutes := days * minutes_per_day) 
  (average_space_per_minute := disk_space / total_minutes) :
  average_space_per_minute = 2 :=
sorry

end average_disk_space_per_minute_l133_133653


namespace largest_digit_divisible_by_6_l133_133513

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l133_133513


namespace percentage_increase_pay_rate_l133_133819

theorem percentage_increase_pay_rate (r t c e : ℕ) (h_reg_rate : r = 10) (h_total_surveys : t = 100) (h_cellphone_surveys : c = 60) (h_total_earnings : e = 1180) : 
  (13 - 10) / 10 * 100 = 30 :=
by
  sorry

end percentage_increase_pay_rate_l133_133819


namespace simplify_polynomial_l133_133892

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 :=
by {
  sorry
}

end simplify_polynomial_l133_133892


namespace number_of_pupils_l133_133919

-- Define the number of total people
def total_people : ℕ := 803

-- Define the number of parents
def parents : ℕ := 105

-- We need to prove the number of pupils is 698
theorem number_of_pupils : (total_people - parents) = 698 := 
by
  -- Skip the proof steps
  sorry

end number_of_pupils_l133_133919


namespace fruit_vendor_sold_fruits_l133_133885

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l133_133885


namespace exists_x_for_every_n_l133_133617

theorem exists_x_for_every_n (n : ℕ) (hn : 0 < n) : ∃ x : ℤ, 2^n ∣ (x^2 - 17) :=
sorry

end exists_x_for_every_n_l133_133617


namespace men_left_the_job_l133_133647

theorem men_left_the_job
    (work_rate_20men : 20 * 4 = 30)
    (work_rate_remaining : 6 * 6 = 36) :
    4 = 20 - (20 * 4) / (6 * 6)  :=
by
  sorry

end men_left_the_job_l133_133647


namespace five_digit_numbers_arithmetic_sequence_count_l133_133821

open Finset

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b ∧ a < b ∧ b < c

def five_digit_numbers_count : ℕ :=
  card (filter (λ n : ℕ,
    let d1 := n / 10000, d2 := n / 1000 % 10, d3 := n / 100 % 10, d4 := n / 10 % 10, d5 := n % 10 in
    n >= 10000 ∧ n < 100000 ∧
    nodup [d1, d2, d3, d4, d5] ∧
    is_arithmetic_sequence d2 d3 d4)
  (range 100000))

theorem five_digit_numbers_arithmetic_sequence_count :
  five_digit_numbers_count = 744 :=
  sorry

end five_digit_numbers_arithmetic_sequence_count_l133_133821


namespace height_of_pole_l133_133661

-- Defining the constants according to the problem statement
def AC := 5.0 -- meters
def AD := 4.0 -- meters
def DE := 1.7 -- meters

-- We need to prove that the height of the pole AB is 8.5 meters
theorem height_of_pole (AB : ℝ) (hAC : AC = 5) (hAD : AD = 4) (hDE : DE = 1.7) :
  AB = 8.5 := by
  sorry

end height_of_pole_l133_133661


namespace max_piles_l133_133777

open Finset

-- Define the condition for splitting and constraints
def valid_pile_splitting (initial_pile : ℕ) : Prop :=
  ∃ (piles : Finset ℕ), 
    (∑ x in piles, x = initial_pile) ∧ 
    (∀ x ∈ piles, ∀ y ∈ piles, x ≠ y → x < 2 * y) 

-- Define the theorem stating the maximum number of piles
theorem max_piles (initial_pile : ℕ) (h : initial_pile = 660) : 
  ∃ (n : ℕ) (piles : Finset ℕ), valid_pile_splitting initial_pile ∧ pile.card = 30 := 
sorry

end max_piles_l133_133777


namespace opposite_of_three_l133_133100

theorem opposite_of_three :
  ∃ x : ℤ, 3 + x = 0 ∧ x = -3 :=
by
  sorry

end opposite_of_three_l133_133100


namespace gym_monthly_revenue_l133_133388

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l133_133388


namespace intersection_of_A_and_B_l133_133316

variable (x y : ℝ)

def A := {y : ℝ | ∃ x > 1, y = Real.log x / Real.log 2}
def B := {y : ℝ | ∃ x > 1, y = (1 / 2) ^ x}

theorem intersection_of_A_and_B :
  (A ∩ B) = {y : ℝ | 0 < y ∧ y < 1 / 2} :=
by sorry

end intersection_of_A_and_B_l133_133316


namespace joe_spent_255_minutes_l133_133446

-- Define the time taken to cut hair for women, men, and children
def time_per_woman : Nat := 50
def time_per_man : Nat := 15
def time_per_child : Nat := 25

-- Define the number of haircuts for each category
def women_haircuts : Nat := 3
def men_haircuts : Nat := 2
def children_haircuts : Nat := 3

-- Compute the total time spent cutting hair
def total_time_spent : Nat :=
  (women_haircuts * time_per_woman) +
  (men_haircuts * time_per_man) +
  (children_haircuts * time_per_child)

-- The theorem stating the total time spent is equal to 255 minutes
theorem joe_spent_255_minutes : total_time_spent = 255 := by
  sorry

end joe_spent_255_minutes_l133_133446


namespace frank_has_3_cookies_l133_133024

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l133_133024


namespace james_profit_correct_l133_133343

noncomputable def jamesProfit : ℝ :=
  let tickets_bought := 200
  let cost_per_ticket := 2
  let winning_ticket_percentage := 0.20
  let percentage_one_dollar := 0.50
  let percentage_three_dollars := 0.30
  let percentage_four_dollars := 0.20
  let percentage_five_dollars := 0.80
  let grand_prize_ticket_count := 1
  let average_remaining_winner := 15
  let tax_percentage := 0.10
  let total_cost := tickets_bought * cost_per_ticket
  let winning_tickets := tickets_bought * winning_ticket_percentage
  let tickets_five_dollars := winning_tickets * percentage_five_dollars
  let other_winning_tickets := winning_tickets - tickets_five_dollars - grand_prize_ticket_count
  let total_winnings_before_tax := (tickets_five_dollars * 5) + (grand_prize_ticket_count * 5000) + (other_winning_tickets * average_remaining_winner)
  let total_tax := total_winnings_before_tax * tax_percentage
  let total_winnings_after_tax := total_winnings_before_tax - total_tax
  total_winnings_after_tax - total_cost

theorem james_profit_correct : jamesProfit = 4338.50 := by
  sorry

end james_profit_correct_l133_133343


namespace sum_mod_13_l133_133268

theorem sum_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 :=
by {
  sorry
}

end sum_mod_13_l133_133268


namespace farmer_total_land_l133_133939

noncomputable def total_land_owned_by_farmer (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) : ℝ :=
  let cleared_land := cleared_percentage
  let total_clearance_with_tomato := cleared_land_with_tomato
  let unused_cleared_percentage := 1 - grape_percentage - potato_percentage
  let total_cleared_land := total_clearance_with_tomato / unused_cleared_percentage
  total_cleared_land / cleared_land

theorem farmer_total_land (cleared_land_with_tomato : ℝ) (cleared_percentage : ℝ) (grape_percentage : ℝ) (potato_percentage : ℝ) :
  (cleared_land_with_tomato = 450) →
  (cleared_percentage = 0.90) →
  (grape_percentage = 0.10) →
  (potato_percentage = 0.80) →
  total_land_owned_by_farmer cleared_land_with_tomato 90 10 80 = 1666.6667 :=
by
  intro h1 h2 h3 h4
  sorry

end farmer_total_land_l133_133939
