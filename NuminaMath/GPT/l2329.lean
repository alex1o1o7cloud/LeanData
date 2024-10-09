import Mathlib

namespace school_population_l2329_232954

variable (b g t a : ℕ)

theorem school_population (h1 : b = 2 * g) (h2 : g = 4 * t) (h3 : a = t / 2) : 
  b + g + t + a = 27 * b / 16 := by
  sorry

end school_population_l2329_232954


namespace part1_part2_l2329_232965

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 2 * a + 3)

theorem part1 (x : ℝ) : f x 2 ≤ 9 ↔ -2 ≤ x ∧ x ≤ 4 :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ Set.Iic (-2 / 3) ∪ Set.Ici (14 / 3) :=
by sorry

end part1_part2_l2329_232965


namespace rectangle_area_change_l2329_232947

theorem rectangle_area_change
  (L B : ℝ)
  (hL : L > 0)
  (hB : B > 0)
  (new_L : ℝ := 1.25 * L)
  (new_B : ℝ := 0.85 * B):
  (new_L * new_B = 1.0625 * (L * B)) :=
by
  sorry

end rectangle_area_change_l2329_232947


namespace geom_seq_common_ratio_l2329_232914

theorem geom_seq_common_ratio (S_3 S_6 : ℕ) (h1 : S_3 = 7) (h2 : S_6 = 63) : 
  ∃ q : ℕ, q = 2 := 
by
  sorry

end geom_seq_common_ratio_l2329_232914


namespace door_height_is_eight_l2329_232943

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l2329_232943


namespace equivalence_of_equation_and_conditions_l2329_232924

open Real
open Set

-- Definitions for conditions
def condition1 (t : ℝ) : Prop := cos t ≠ 0
def condition2 (t : ℝ) : Prop := sin t ≠ 0
def condition3 (t : ℝ) : Prop := cos (2 * t) ≠ 0

-- The main statement to be proved
theorem equivalence_of_equation_and_conditions (t : ℝ) :
  ((sin t / cos t - cos t / sin t + 2 * (sin (2 * t) / cos (2 * t))) * (1 + cos (3 * t))) = 4 * sin (3 * t) ↔
  ((∃ k l : ℤ, t = (π / 5) * (2 * k + 1) ∧ k ≠ 5 * l + 2) ∨ (∃ n l : ℤ, t = (π / 3) * (2 * n + 1) ∧ n ≠ 3 * l + 1))
    ∧ condition1 t
    ∧ condition2 t
    ∧ condition3 t :=
by
  sorry

end equivalence_of_equation_and_conditions_l2329_232924


namespace percentage_of_students_owning_birds_l2329_232986

theorem percentage_of_students_owning_birds
    (total_students : ℕ) 
    (students_owning_birds : ℕ) 
    (h_total_students : total_students = 500) 
    (h_students_owning_birds : students_owning_birds = 75) : 
    (students_owning_birds * 100) / total_students = 15 := 
by 
    sorry

end percentage_of_students_owning_birds_l2329_232986


namespace ratio_of_A_to_B_l2329_232957

theorem ratio_of_A_to_B (A B C : ℝ) (hB : B = 270) (hBC : B = (1 / 4) * C) (hSum : A + B + C = 1440) : A / B = 1 / 3 :=
by
  -- The proof is omitted for this example
  sorry

end ratio_of_A_to_B_l2329_232957


namespace distance_points_l2329_232920

theorem distance_points : 
  let P1 := (2, -1)
  let P2 := (7, 6)
  dist P1 P2 = Real.sqrt 74 :=
by
  sorry

end distance_points_l2329_232920


namespace natural_numbers_equal_power_l2329_232944

theorem natural_numbers_equal_power
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n :=
by
  sorry

end natural_numbers_equal_power_l2329_232944


namespace length_FD_of_folded_square_l2329_232925

theorem length_FD_of_folded_square :
  let A := (0, 0)
  let B := (8, 0)
  let D := (0, 8)
  let C := (8, 8)
  let E := (6, 0)
  let F := (8, 8 - (FD : ℝ))
  (ABCD_square : ∀ {x y : ℝ}, (x = 0 ∨ x = 8) ∧ (y = 0 ∨ y = 8)) →  
  let DE := (6 - 0 : ℝ)
  let Pythagorean_statement := (8 - FD) ^ 2 = FD ^ 2 + 6 ^ 2
  ∃ FD : ℝ, FD = 7 / 4 :=
sorry

end length_FD_of_folded_square_l2329_232925


namespace compare_numbers_l2329_232966

theorem compare_numbers :
  3 * 10^5 < 2 * 10^6 ∧ -2 - 1 / 3 > -3 - 1 / 2 := by
  sorry

end compare_numbers_l2329_232966


namespace problem_statement_l2329_232983

/-!
The problem states:
If |a-2| and |m+n+3| are opposite numbers, then a + m + n = -1.
-/

theorem problem_statement (a m n : ℤ) (h : |a - 2| = -|m + n + 3|) : a + m + n = -1 :=
by {
  sorry
}

end problem_statement_l2329_232983


namespace sum_of_cubes_l2329_232948

theorem sum_of_cubes (a b : ℕ) (h1 : 2 * x = a) (h2 : 3 * x = b) (h3 : b - a = 3) : a^3 + b^3 = 945 := by
  sorry

end sum_of_cubes_l2329_232948


namespace mike_total_spending_is_correct_l2329_232972

-- Definitions for the costs of the items
def cost_marbles : ℝ := 9.05
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52
def cost_toy_car : ℝ := 3.75
def cost_puzzle : ℝ := 8.99
def cost_stickers : ℝ := 1.25

-- Definitions for the discounts
def discount_puzzle : ℝ := 0.15
def discount_toy_car : ℝ := 0.10

-- Definition for the coupon
def coupon_amount : ℝ := 5.00

-- Total spent by Mike on toys
def total_spent : ℝ :=
  cost_marbles + 
  cost_football + 
  cost_baseball + 
  (cost_toy_car - cost_toy_car * discount_toy_car) + 
  (cost_puzzle - cost_puzzle * discount_puzzle) + 
  cost_stickers - 
  coupon_amount

-- Proof statement
theorem mike_total_spending_is_correct : 
  total_spent = 27.7865 :=
by
  sorry

end mike_total_spending_is_correct_l2329_232972


namespace min_PM_PN_min_PM_squared_PN_squared_l2329_232903

noncomputable def min_value_PM_PN := 3 * Real.sqrt 5

noncomputable def min_value_PM_squared_PN_squared := 229 / 10

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 5⟩
def N : Point := ⟨-2, 4⟩

def on_line (P : Point) : Prop :=
  P.x - 2 * P.y + 3 = 0

theorem min_PM_PN {P : Point} (h : on_line P) :
  dist (P.x, P.y) (M.x, M.y) + dist (P.x, P.y) (N.x, N.y) = min_value_PM_PN := sorry

theorem min_PM_squared_PN_squared {P : Point} (h : on_line P) :
  (dist (P.x, P.y) (M.x, M.y))^2 + (dist (P.x, P.y) (N.x, N.y))^2 = min_value_PM_squared_PN_squared := sorry

end min_PM_PN_min_PM_squared_PN_squared_l2329_232903


namespace distinct_nonzero_digits_sum_l2329_232952

theorem distinct_nonzero_digits_sum
  (x y z w : Nat)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hxw : x ≠ w)
  (hyz : y ≠ z)
  (hyw : y ≠ w)
  (hzw : z ≠ w)
  (h1 : w + x = 10)
  (h2 : y + w = 9)
  (h3 : z + x = 9) :
  x + y + z + w = 18 :=
sorry

end distinct_nonzero_digits_sum_l2329_232952


namespace product_of_first_three_terms_is_960_l2329_232927

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l2329_232927


namespace min_value_eq_9_l2329_232987

-- Defining the conditions
variable (a b : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (h_eq : a - 2 * b = 0)

-- The goal is to prove the minimum value of (1/a) + (4/b) is 9
theorem min_value_eq_9 (ha : a > 0) (hb : b > 0) (h_eq : a - 2 * b = 0) 
  : ∃ (m : ℝ), m = 9 ∧ (∀ x, x = 1/a + 4/b → x ≥ m) :=
sorry

end min_value_eq_9_l2329_232987


namespace part1_part2_l2329_232969

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part (1) 
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ a) → -2 ≤ a ∧ a ≤ 1 := by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l2329_232969


namespace gcd_lcm_of_45_and_150_l2329_232919

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem gcd_lcm_of_45_and_150 :
  GCD 45 150 = 15 ∧ LCM 45 150 = 450 :=
by
  sorry

end gcd_lcm_of_45_and_150_l2329_232919


namespace sum_first_n_terms_l2329_232917

-- Define the sequence a_n
def geom_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Define the main conditions from the problem
axiom a7_cond (a : ℕ → ℕ) : a 7 = 8 * a 4
axiom arithmetic_seq_cond (a : ℕ → ℕ) : (1 / 2 : ℝ) * a 2 < (a 3 - 4) ∧ (a 3 - 4) < (a 4 - 12)

-- Define the sequences a_n and b_n using the conditions
def a_n (n : ℕ) : ℕ := 2^(n + 1)
def b_n (n : ℕ) : ℤ := (-1)^n * (Int.ofNat (n + 1))

-- Define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) : ℤ :=
  (Finset.range n).sum b_n

-- Main theorem statement
theorem sum_first_n_terms (k : ℕ) : |T_n k| = 20 → k = 40 ∨ k = 37 :=
sorry

end sum_first_n_terms_l2329_232917


namespace acute_angle_range_l2329_232975

theorem acute_angle_range (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : Real.sin α < Real.cos α) : 0 < α ∧ α < π / 4 :=
sorry

end acute_angle_range_l2329_232975


namespace yoongi_age_l2329_232984

theorem yoongi_age (Y H : ℕ) (h1 : Y + H = 16) (h2 : Y = H + 2) : Y = 9 :=
by
  sorry

end yoongi_age_l2329_232984


namespace science_club_election_l2329_232956

theorem science_club_election :
  let total_candidates := 20
  let past_officers := 10
  let non_past_officers := total_candidates - past_officers
  let positions := 6
  let total_ways := Nat.choose total_candidates positions
  let no_past_officer_ways := Nat.choose non_past_officers positions
  let exactly_one_past_officer_ways := past_officers * Nat.choose non_past_officers (positions - 1)
  total_ways - no_past_officer_ways - exactly_one_past_officer_ways = 36030 := by
    sorry

end science_club_election_l2329_232956


namespace tim_total_expenditure_l2329_232963

def apple_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 6
def chocolate_price : ℕ := 10

def apple_quantity : ℕ := 8
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 3
def flour_quantity : ℕ := 3
def chocolate_quantity : ℕ := 1

def discounted_pineapple_price : ℕ := pineapple_price / 2
def discounted_milk_price : ℕ := milk_price - 1
def coupon_discount : ℕ := 10
def discount_threshold : ℕ := 50

def total_cost_before_coupon : ℕ :=
  (apple_quantity * apple_price) +
  (milk_quantity * discounted_milk_price) +
  (pineapple_quantity * discounted_pineapple_price) +
  (flour_quantity * flour_price) +
  chocolate_price

def final_price : ℕ :=
  if total_cost_before_coupon >= discount_threshold
  then total_cost_before_coupon - coupon_discount
  else total_cost_before_coupon

theorem tim_total_expenditure : final_price = 40 := by
  sorry

end tim_total_expenditure_l2329_232963


namespace opposite_of_2023_l2329_232926

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l2329_232926


namespace relatively_prime_2n_plus_1_4n2_plus_1_l2329_232959

theorem relatively_prime_2n_plus_1_4n2_plus_1 (n : ℕ) (h : n > 0) : 
  Nat.gcd (2 * n + 1) (4 * n^2 + 1) = 1 := 
by
  sorry

end relatively_prime_2n_plus_1_4n2_plus_1_l2329_232959


namespace bus_speed_kmph_l2329_232912

theorem bus_speed_kmph : 
  let distance := 600.048 
  let time := 30
  (distance / time) * 3.6 = 72.006 :=
by
  sorry

end bus_speed_kmph_l2329_232912


namespace total_journey_time_l2329_232909

def distance_to_post_office : ℝ := 19.999999999999996
def speed_to_post_office : ℝ := 25
def speed_back : ℝ := 4

theorem total_journey_time : 
  (distance_to_post_office / speed_to_post_office) + (distance_to_post_office / speed_back) = 5.8 :=
by
  sorry

end total_journey_time_l2329_232909


namespace find_M_l2329_232999

variable (M : ℕ)

theorem find_M (h : (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M) : M = 1003 :=
sorry

end find_M_l2329_232999


namespace pies_sold_each_day_l2329_232976

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end pies_sold_each_day_l2329_232976


namespace quadratic_equation_has_real_root_l2329_232967

theorem quadratic_equation_has_real_root
  (a c m n : ℝ) :
  ∃ x : ℝ, c * x^2 + m * x - a = 0 ∨ ∃ y : ℝ, a * y^2 + n * y + c = 0 :=
by
  -- Proof omitted
  sorry

end quadratic_equation_has_real_root_l2329_232967


namespace P_and_Q_equivalent_l2329_232995

def P (x : ℝ) : Prop := 3 * x - x^2 ≤ 0
def Q (x : ℝ) : Prop := |x| ≤ 2
def P_intersection_Q (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

theorem P_and_Q_equivalent : ∀ x, (P x ∧ Q x) ↔ P_intersection_Q x :=
by {
  sorry
}

end P_and_Q_equivalent_l2329_232995


namespace largest_consecutive_odd_sum_l2329_232973

theorem largest_consecutive_odd_sum (x : ℤ) (h : 20 * (x + 19) = 8000) : x + 38 = 419 := 
by
  sorry

end largest_consecutive_odd_sum_l2329_232973


namespace lcm_153_180_560_l2329_232945

theorem lcm_153_180_560 : Nat.lcm (Nat.lcm 153 180) 560 = 85680 :=
by
  sorry

end lcm_153_180_560_l2329_232945


namespace students_wearing_other_colors_l2329_232935

variable (total_students blue_percentage red_percentage green_percentage : ℕ)
variable (h_total : total_students = 600)
variable (h_blue : blue_percentage = 45)
variable (h_red : red_percentage = 23)
variable (h_green : green_percentage = 15)

theorem students_wearing_other_colors :
  (total_students * (100 - (blue_percentage + red_percentage + green_percentage)) / 100 = 102) :=
by
  sorry

end students_wearing_other_colors_l2329_232935


namespace derivative_at_neg_one_l2329_232994

noncomputable def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_neg_one (a b c : ℝ) (h : (4 * a * 1^3 + 2 * b * 1) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := 
sorry

end derivative_at_neg_one_l2329_232994


namespace correct_dispersion_statements_l2329_232939

def statement1 (make_use_of_data : Prop) : Prop :=
make_use_of_data = true

def statement2 (multi_numerical_values : Prop) : Prop :=
multi_numerical_values = true

def statement3 (dispersion_large_value_small : Prop) : Prop :=
dispersion_large_value_small = false

theorem correct_dispersion_statements
  (make_use_of_data : Prop)
  (multi_numerical_values : Prop)
  (dispersion_large_value_small : Prop)
  (h1 : statement1 make_use_of_data)
  (h2 : statement2 multi_numerical_values)
  (h3 : statement3 dispersion_large_value_small) :
  (make_use_of_data ∧ multi_numerical_values ∧ ¬ dispersion_large_value_small) = true :=
by
  sorry

end correct_dispersion_statements_l2329_232939


namespace brother_to_madeline_ratio_l2329_232978

theorem brother_to_madeline_ratio (M B T : ℕ) (hM : M = 48) (hT : T = 72) (hSum : M + B = T) : B / M = 1 / 2 := by
  sorry

end brother_to_madeline_ratio_l2329_232978


namespace find_y_l2329_232913

-- Declare the variables and conditions
variable (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 1.5 * x = 0.3 * y
def condition2 : Prop := x = 20

-- State the theorem that given these conditions, y must be 100
theorem find_y (h1 : condition1 x y) (h2 : condition2 x) : y = 100 :=
by sorry

end find_y_l2329_232913


namespace triangle_sides_inequality_l2329_232953

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : a + b + c ≤ 2) :
  -3 < (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) ∧ 
  (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) < 3 :=
by sorry

end triangle_sides_inequality_l2329_232953


namespace jessica_balloon_count_l2329_232949

theorem jessica_balloon_count :
  (∀ (joan_initial_balloon_count sally_popped_balloon_count total_balloon_count: ℕ),
  joan_initial_balloon_count = 9 →
  sally_popped_balloon_count = 5 →
  total_balloon_count = 6 →
  ∃ (jessica_balloon_count: ℕ),
    jessica_balloon_count = total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count) →
    jessica_balloon_count = 2) :=
by
  intros joan_initial_balloon_count sally_popped_balloon_count total_balloon_count j1 j2 t1
  use total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count)
  sorry

end jessica_balloon_count_l2329_232949


namespace polynomial_expansion_l2329_232934

theorem polynomial_expansion :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a_1 * x^4 + a_2 * x^3 + a_3 * x^2 + 16 * x + 4) :=
by
  sorry

end polynomial_expansion_l2329_232934


namespace abc_less_than_one_l2329_232950

variables {a b c : ℝ}

theorem abc_less_than_one (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1: a^2 < b) (h2: b^2 < c) (h3: c^2 < a) : a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end abc_less_than_one_l2329_232950


namespace proportion_of_salt_correct_l2329_232941

def grams_of_salt := 50
def grams_of_water := 1000
def total_solution := grams_of_salt + grams_of_water
def proportion_of_salt : ℚ := grams_of_salt / total_solution

theorem proportion_of_salt_correct :
  proportion_of_salt = 1 / 21 := 
  by {
    sorry
  }

end proportion_of_salt_correct_l2329_232941


namespace total_buttons_l2329_232910

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l2329_232910


namespace common_ratio_geom_series_l2329_232961

theorem common_ratio_geom_series 
  (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 4 / 7) 
  (h₂ : a₂ = 20 / 21) :
  ∃ r : ℚ, r = 5 / 3 ∧ a₂ / a₁ = r := 
sorry

end common_ratio_geom_series_l2329_232961


namespace infinite_rel_prime_set_of_form_2n_minus_3_l2329_232932

theorem infinite_rel_prime_set_of_form_2n_minus_3 : ∃ S : Set ℕ, (∀ x ∈ S, ∃ n : ℕ, x = 2^n - 3) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧ S.Infinite := 
by
  sorry

end infinite_rel_prime_set_of_form_2n_minus_3_l2329_232932


namespace solve_eq1_solve_eq2_l2329_232921

theorem solve_eq1 (y : ℝ) : 6 - 3 * y = 15 + 6 * y ↔ y = -1 := by
  sorry

theorem solve_eq2 (x : ℝ) : (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 2 ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l2329_232921


namespace inequality_solution_set_l2329_232989

variable {f : ℝ → ℝ}

-- Conditions
def neg_domain : Set ℝ := {x | x < 0}
def pos_domain : Set ℝ := {x | x > 0}
def f_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f_property_P (f : ℝ → ℝ) := ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

-- Translate question and correct answer into a proposition in Lean
theorem inequality_solution_set (h1 : ∀ x, f (-x) = -f x)
                                (h2 : ∀ x1 x2, (0 < x1) → (0 < x2) → (x1 ≠ x1) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)) :
  {x | f (x - 2) < f (x^2 - 4) / (x + 2)} = {x | x < -3} ∪ {x | -1 < x ∧ x < 2} := 
sorry

end inequality_solution_set_l2329_232989


namespace simplify_polynomial_expression_l2329_232979

theorem simplify_polynomial_expression (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) = r^3 - 4 * r^2 + 2 * r + 3 :=
by
  sorry

end simplify_polynomial_expression_l2329_232979


namespace sum_of_coordinates_after_reflections_l2329_232916

theorem sum_of_coordinates_after_reflections :
  let A := (3, 2)
  let B := (9, 18)
  let N := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflect_y (P : ℤ × ℤ) := (-P.1, P.2)
  let reflect_x (P : ℤ × ℤ) := (P.1, -P.2)
  let N' := reflect_y N
  let N'' := reflect_x N'
  N''.1 + N''.2 = -16 := by sorry

end sum_of_coordinates_after_reflections_l2329_232916


namespace smallest_omega_l2329_232991

theorem smallest_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ k : ℤ, (2 / 3) * ω = 2 * k) -> ω = 3 :=
by
  sorry

end smallest_omega_l2329_232991


namespace find_a_l2329_232971

theorem find_a : (a : ℕ) = 103 * 97 * 10009 → a = 99999919 := by
  intro h
  sorry

end find_a_l2329_232971


namespace trig_problem_l2329_232955

-- Translate the conditions and problems into Lean 4:
theorem trig_problem (α : ℝ) (h1 : Real.tan α = 2) :
    (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end trig_problem_l2329_232955


namespace max_xy_l2329_232928

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 16) : 
  xy ≤ 32 :=
sorry

end max_xy_l2329_232928


namespace value_of_A_is_18_l2329_232931

theorem value_of_A_is_18
  (A B C D : ℕ)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : A ≠ D)
  (h4 : B ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A * B = 72)
  (h8 : C * D = 72)
  (h9 : A - B = C + D) : A = 18 :=
sorry

end value_of_A_is_18_l2329_232931


namespace triangle_area_correct_l2329_232930

def line1 (x : ℝ) : ℝ := 8
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the intersection points
def intersection1 : ℝ × ℝ := (6, line1 6)
def intersection2 : ℝ × ℝ := (-6, line1 (-6))
def intersection3 : ℝ × ℝ := (0, line2 0)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_correct :
  triangle_area intersection1 intersection2 intersection3 = 36 :=
by
  sorry

end triangle_area_correct_l2329_232930


namespace sum_of_squares_five_consecutive_not_perfect_square_l2329_232904

theorem sum_of_squares_five_consecutive_not_perfect_square 
  (x : ℤ) : ¬ ∃ k : ℤ, (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 = k^2 :=
by 
  sorry

end sum_of_squares_five_consecutive_not_perfect_square_l2329_232904


namespace train_cross_time_l2329_232933

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 255.03
noncomputable def train_speed_ms : ℝ := 12.5
noncomputable def distance_to_travel : ℝ := train_length + bridge_length
noncomputable def expected_time : ℝ := 30.0024

theorem train_cross_time :
  (distance_to_travel / train_speed_ms) = expected_time :=
by sorry

end train_cross_time_l2329_232933


namespace minimum_discount_l2329_232985

open Real

theorem minimum_discount (CP MP SP_min : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  CP = 800 ∧ MP = 1200 ∧ SP_min = 960 ∧ profit_margin = 0.20 ∧
  MP * (1 - discount / 100) ≥ SP_min → discount = 20 :=
by
  intros h
  rcases h with ⟨h_cp, h_mp, h_sp_min, h_profit_margin, h_selling_price⟩
  simp [h_cp, h_mp, h_sp_min, h_profit_margin, sub_eq_self, div_eq_self] at *
  sorry

end minimum_discount_l2329_232985


namespace part_I_part_II_l2329_232964

noncomputable def f (x : ℝ) : ℝ :=
  |x - (1/2)| + |x + (1/2)|

def solutionSetM : Set ℝ :=
  { x : ℝ | -1 < x ∧ x < 1 }

theorem part_I :
  { x : ℝ | f x < 2 } = solutionSetM := 
sorry

theorem part_II (a b : ℝ) (ha : a ∈ solutionSetM) (hb : b ∈ solutionSetM) :
  |a + b| < |1 + a * b| :=
sorry

end part_I_part_II_l2329_232964


namespace necessary_but_not_sufficient_condition_l2329_232940

variables (p q : Prop)

theorem necessary_but_not_sufficient_condition
  (h : ¬p → q) (hn : ¬q → p) : 
  (p → ¬q) ∧ ¬(¬q → p) :=
sorry

end necessary_but_not_sufficient_condition_l2329_232940


namespace largest_perfect_square_factor_4410_l2329_232902

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end largest_perfect_square_factor_4410_l2329_232902


namespace common_tangent_slope_l2329_232923

theorem common_tangent_slope (a m : ℝ) : 
  ((∃ a, ∃ m, l = (2 * a) ∧ l = (3 * m^2) ∧ a^2 = 2 * m^3) → (l = 0 ∨ l = 64 / 27)) := 
sorry

end common_tangent_slope_l2329_232923


namespace solve_abs_inequality_l2329_232901

theorem solve_abs_inequality (x : ℝ) (h : 1 < |x - 1| ∧ |x - 1| < 4) : (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5) :=
by
  sorry

end solve_abs_inequality_l2329_232901


namespace range_of_x_for_f_ln_x_gt_f_1_l2329_232905

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def is_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x_for_f_ln_x_gt_f_1
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_dec : is_decreasing_on_nonneg f)
  (hf_condition : ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e) :
  ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e := sorry

end range_of_x_for_f_ln_x_gt_f_1_l2329_232905


namespace correct_equation_l2329_232998

theorem correct_equation :
  ¬ (7^3 * 7^3 = 7^9) ∧ 
  (-3^7 / 3^2 = -3^5) ∧ 
  ¬ (2^6 + (-2)^6 = 0) ∧ 
  ¬ ((-3)^5 / (-3)^3 = -3^2) :=
by 
  sorry

end correct_equation_l2329_232998


namespace choose_one_from_ten_l2329_232918

theorem choose_one_from_ten :
  Nat.choose 10 1 = 10 :=
by
  sorry

end choose_one_from_ten_l2329_232918


namespace exists_n0_find_N_l2329_232980

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Definition of the sequence {a_n}
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = f (a n)

-- Problem (1): Existence of n0
theorem exists_n0 (a : ℕ → ℝ) (h_seq : seq a) (h_a1 : a 1 = 3) : 
  ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n :=
  sorry

-- Problem (2): Smallest N
theorem find_N (a : ℕ → ℝ) (h_seq : seq a) (m : ℕ) (h_m : m > 1) 
  (h_a1 : 1 + 1 / (m : ℝ) < a 1 ∧ a 1 < m / (m - 1)) : 
  ∃ N : ℕ, ∀ n ≥ N, 0 < a n ∧ a n < 1 :=
  sorry

end exists_n0_find_N_l2329_232980


namespace production_rate_equation_l2329_232996

theorem production_rate_equation (x : ℝ) (h : x > 0) :
  3000 / x - 3000 / (2 * x) = 5 :=
sorry

end production_rate_equation_l2329_232996


namespace find_cost_price_per_meter_l2329_232960

/-- Given that a shopkeeper sells 200 meters of cloth for Rs. 12000 at a loss of Rs. 6 per meter,
we want to find the cost price per meter of cloth. Specifically, we need to prove that the
cost price per meter is Rs. 66. -/
theorem find_cost_price_per_meter
  (total_meters : ℕ := 200)
  (selling_price : ℕ := 12000)
  (loss_per_meter : ℕ := 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 :=
sorry

end find_cost_price_per_meter_l2329_232960


namespace cost_per_pack_is_correct_l2329_232970

def total_amount_spent : ℝ := 120
def num_packs_bought : ℕ := 6
def expected_cost_per_pack : ℝ := 20

theorem cost_per_pack_is_correct :
  total_amount_spent / num_packs_bought = expected_cost_per_pack :=
  by 
    -- here would be the proof
    sorry

end cost_per_pack_is_correct_l2329_232970


namespace positive_slope_asymptote_l2329_232974

-- Define the foci points A and B and the given equation of the hyperbola
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-3, 1)
def hyperbola_eqn (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- State the theorem about the positive slope of the asymptote
theorem positive_slope_asymptote (x y : ℝ) (h : hyperbola_eqn x y) : 
  ∃ b a : ℝ, b = Real.sqrt 5 ∧ a = 2 ∧ (b / a) = Real.sqrt 5 / 2 :=
by
  sorry

end positive_slope_asymptote_l2329_232974


namespace dreamy_bookstore_sales_l2329_232922

theorem dreamy_bookstore_sales :
  let total_sales_percent := 100
  let notebooks_percent := 45
  let bookmarks_percent := 25
  let neither_notebooks_nor_bookmarks_percent := total_sales_percent - (notebooks_percent + bookmarks_percent)
  neither_notebooks_nor_bookmarks_percent = 30 :=
by {
  sorry
}

end dreamy_bookstore_sales_l2329_232922


namespace putnam_inequality_l2329_232962

variable (a x : ℝ)

theorem putnam_inequality (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3 * a * (a - x)^5 +
  5 / 2 * a^2 * (a - x)^4 -
  1 / 2 * a^4 * (a - x)^2 < 0 :=
by
  sorry

end putnam_inequality_l2329_232962


namespace rectangle_area_unchanged_l2329_232977

theorem rectangle_area_unchanged
  (x y : ℝ)
  (h1 : x * y = (x + 3) * (y - 1))
  (h2 : x * y = (x - 3) * (y + 1.5)) :
  x * y = 31.5 :=
sorry

end rectangle_area_unchanged_l2329_232977


namespace simplify_and_evaluate_l2329_232906

theorem simplify_and_evaluate (a : ℕ) (h : a = 2023) : (a + 1) / a / (a - 1 / a) = 1 / 2022 :=
by
  sorry

end simplify_and_evaluate_l2329_232906


namespace abs_value_product_l2329_232958

theorem abs_value_product (x : ℝ) (h : |x - 5| - 4 = 0) : ∃ y z, (y - 5 = 4 ∨ y - 5 = -4) ∧ (z - 5 = 4 ∨ z - 5 = -4) ∧ y * z = 9 :=
by 
  sorry

end abs_value_product_l2329_232958


namespace distinct_pos_real_ints_l2329_232907

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distinct_pos_real_ints (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : ∀ n : ℕ, (floor (n * a)) ∣ (floor (n * b))) : ∃ k l : ℤ, a = k ∧ b = l :=
by
  sorry

end distinct_pos_real_ints_l2329_232907


namespace domain_of_tan_l2329_232990

theorem domain_of_tan :
    ∀ k : ℤ, ∀ x : ℝ,
    (x > (k * π / 2 - π / 8) ∧ x < (k * π / 2 + 3 * π / 8)) ↔
    2 * x - π / 4 ≠ k * π + π / 2 :=
by
  intro k x
  sorry

end domain_of_tan_l2329_232990


namespace infinitely_many_n_squared_plus_one_no_special_divisor_l2329_232981

theorem infinitely_many_n_squared_plus_one_no_special_divisor :
  ∃ (f : ℕ → ℕ), (∀ n, f n ≠ 0) ∧ ∀ n, ∀ k, f n^2 + 1 ≠ k^2 + 1 ∨ k^2 + 1 = 1 :=
by
  sorry

end infinitely_many_n_squared_plus_one_no_special_divisor_l2329_232981


namespace pure_imaginary_m_value_l2329_232968

theorem pure_imaginary_m_value (m : ℝ) (h₁ : m ^ 2 + m - 2 = 0) (h₂ : m ^ 2 - 1 ≠ 0) : m = -2 := by
  sorry

end pure_imaginary_m_value_l2329_232968


namespace middle_number_of_ratio_l2329_232911

theorem middle_number_of_ratio (x : ℝ) (h : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862) : 2 * x = 14 :=
sorry

end middle_number_of_ratio_l2329_232911


namespace complement_A_inter_B_l2329_232993

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def A : Set ℤ := { x | x * (x - 1) = 0 }
def B : Set ℤ := { x | -1 < x ∧ x < 2 }

theorem complement_A_inter_B {U A B : Set ℤ} :
  A ⊆ U → B ⊆ U → 
  (A ∩ B) ⊆ (U ∩ A ∩ B) → 
  (U \ (A ∩ B)) = { -1, 2 } :=
by 
  sorry

end complement_A_inter_B_l2329_232993


namespace num_students_59_l2329_232900

theorem num_students_59 (apples : ℕ) (taken_each : ℕ) (students : ℕ) 
  (h_apples : apples = 120) 
  (h_taken_each : taken_each = 2) 
  (h_students_divisors : ∀ d, d = 59 → d ∣ (apples / taken_each)) : students = 59 :=
sorry

end num_students_59_l2329_232900


namespace prob_sum_divisible_by_4_l2329_232992

-- Defining the set and its properties
def set : Finset ℕ := {1, 2, 3, 4, 5}

def isDivBy4 (n : ℕ) : Prop := n % 4 = 0

-- Defining a function to calculate combinations
def combinations (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Defining the successful outcomes and the total combinations
def successfulOutcomes : ℕ := 3
def totalOutcomes : ℕ := combinations 5 3

-- Defining the probability
def probability : ℚ := successfulOutcomes / ↑totalOutcomes

-- The proof problem
theorem prob_sum_divisible_by_4 : probability = 3 / 10 := by
  sorry

end prob_sum_divisible_by_4_l2329_232992


namespace speed_of_the_stream_l2329_232938

theorem speed_of_the_stream (d v_s : ℝ) :
  (∀ (t_up t_down : ℝ), t_up = d / (57 - v_s) ∧ t_down = d / (57 + v_s) ∧ t_up = 2 * t_down) →
  v_s = 19 := by
  sorry

end speed_of_the_stream_l2329_232938


namespace integer_divisibility_l2329_232988

theorem integer_divisibility
  (x y z : ℤ)
  (h : 11 ∣ (7 * x + 2 * y - 5 * z)) :
  11 ∣ (3 * x - 7 * y + 12 * z) :=
sorry

end integer_divisibility_l2329_232988


namespace price_difference_l2329_232942

theorem price_difference (P F : ℝ) (h1 : 0.85 * P = 78.2) (h2 : F = 78.2 * 1.25) : F - P = 5.75 :=
by
  sorry

end price_difference_l2329_232942


namespace reporters_not_covering_politics_l2329_232908

theorem reporters_not_covering_politics (P_X P_Y P_Z intlPol otherPol econOthers : ℝ)
  (h1 : P_X = 0.15) (h2 : P_Y = 0.10) (h3 : P_Z = 0.08)
  (h4 : otherPol = 0.50) (h5 : intlPol = 0.05) (h6 : econOthers = 0.02) :
  (1 - (P_X + P_Y + P_Z + intlPol + otherPol + econOthers)) = 0.10 := by
  sorry

end reporters_not_covering_politics_l2329_232908


namespace tip_percentage_is_20_l2329_232997

noncomputable def total_bill : ℕ := 16 + 14
noncomputable def james_share : ℕ := total_bill / 2
noncomputable def james_paid : ℕ := 21
noncomputable def tip_amount : ℕ := james_paid - james_share
noncomputable def tip_percentage : ℕ := (tip_amount * 100) / total_bill 

theorem tip_percentage_is_20 :
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_is_20_l2329_232997


namespace new_average_score_l2329_232951

theorem new_average_score (avg_score : ℝ) (num_students : ℕ) (dropped_score : ℝ) (new_num_students : ℕ) :
  num_students = 16 →
  avg_score = 61.5 →
  dropped_score = 24 →
  new_num_students = num_students - 1 →
  (avg_score * num_students - dropped_score) / new_num_students = 64 :=
by
  sorry

end new_average_score_l2329_232951


namespace union_comm_union_assoc_inter_distrib_union_l2329_232936

variables {α : Type*} (A B C : Set α)

theorem union_comm : A ∪ B = B ∪ A := sorry

theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry

theorem inter_distrib_union : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry

end union_comm_union_assoc_inter_distrib_union_l2329_232936


namespace gcd_1037_425_l2329_232937

theorem gcd_1037_425 : Int.gcd 1037 425 = 17 :=
by
  sorry

end gcd_1037_425_l2329_232937


namespace calculate_expression_l2329_232929

theorem calculate_expression :
  (16^16 * 8^8) / 4^32 = 16777216 := by
  sorry

end calculate_expression_l2329_232929


namespace find_m_l2329_232915

theorem find_m (x m : ℤ) (h : x = -1 ∧ x - 2 * m = 9) : m = -5 :=
sorry

end find_m_l2329_232915


namespace quadratic_root_range_l2329_232946

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l2329_232946


namespace inequality_solution_l2329_232982

theorem inequality_solution (x : ℝ) : 3 * x^2 - 8 * x + 3 < 0 ↔ (1 / 3 < x ∧ x < 3) := by
  sorry

end inequality_solution_l2329_232982
