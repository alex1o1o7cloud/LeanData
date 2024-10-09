import Mathlib

namespace horizontal_shift_equivalence_l1788_178893

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)
noncomputable def resulting_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, resulting_function x = original_function (x + Real.pi / 3) :=
by sorry

end horizontal_shift_equivalence_l1788_178893


namespace option_a_option_d_l1788_178848

theorem option_a (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m = Nat.choose n (n - m) := 
sorry

theorem option_d (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m := 
sorry

end option_a_option_d_l1788_178848


namespace percent_round_trip_tickets_l1788_178881

-- Define the main variables
variables (P R : ℝ)

-- Define the conditions based on the problem statement
def condition1 : Prop := 0.3 * P = 0.3 * R
 
-- State the theorem to prove
theorem percent_round_trip_tickets (h1 : condition1 P R) : R / P * 100 = 30 := by sorry

end percent_round_trip_tickets_l1788_178881


namespace greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l1788_178860

-- Define the given conditions
def totalOranges : ℕ := 81
def totalCookies : ℕ := 65
def numberOfChildren : ℕ := 7

-- Define the floor division for children
def orangesPerChild : ℕ := totalOranges / numberOfChildren
def cookiesPerChild : ℕ := totalCookies / numberOfChildren

-- Calculate leftover (donated) quantities
def orangesLeftover : ℕ := totalOranges % numberOfChildren
def cookiesLeftover : ℕ := totalCookies % numberOfChildren

-- Statements to prove
theorem greatest_number_of_donated_oranges : orangesLeftover = 4 := by {
    sorry
}

theorem greatest_number_of_donated_cookies : cookiesLeftover = 2 := by {
    sorry
}

end greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l1788_178860


namespace correct_option_l1788_178815

variable (a : ℝ)

theorem correct_option (h1 : 5 * a^2 - 4 * a^2 = a^2)
                       (h2 : a^7 / a^4 = a^3)
                       (h3 : (a^3)^2 = a^6)
                       (h4 : a^2 * a^3 = a^5) : 
                       a^7 / a^4 = a^3 := 
by
  exact h2

end correct_option_l1788_178815


namespace no_solution_exists_l1788_178823

theorem no_solution_exists :
  ¬ ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧ n % 8 = 0 ∧ n % 10 = 6 ∧ n % 7 = 6 := 
by
  sorry

end no_solution_exists_l1788_178823


namespace Carla_servings_l1788_178891

-- Define the volumes involved
def volume_watermelon : ℕ := 500
def volume_cream : ℕ := 100
def volume_per_serving : ℕ := 150

-- The total volume is the sum of the watermelon and cream volumes
def total_volume : ℕ := volume_watermelon + volume_cream

-- The number of servings is the total volume divided by the volume per serving
def n_servings : ℕ := total_volume / volume_per_serving

-- The theorem to prove that Carla can make 4 servings of smoothies
theorem Carla_servings : n_servings = 4 := by
  sorry

end Carla_servings_l1788_178891


namespace find_a_value_l1788_178859

theorem find_a_value (a x y : ℝ) :
  (|y| + |y - x| ≤ a - |x - 1| ∧ (y - 4) * (y + 3) ≥ (4 - x) * (3 + x)) → a = 7 :=
by
  sorry

end find_a_value_l1788_178859


namespace lily_jog_time_l1788_178865

theorem lily_jog_time :
  (∃ (max_time : ℕ) (lily_miles_max : ℕ) (max_distance : ℕ) (lily_time_ratio : ℕ) (distance_wanted : ℕ)
      (expected_time : ℕ),
    max_time = 36 ∧
    lily_miles_max = 4 ∧
    max_distance = 6 ∧
    lily_time_ratio = 3 ∧
    distance_wanted = 7 ∧
    expected_time = 21 ∧
    lily_miles_max * lily_time_ratio = max_time ∧
    max_distance * lily_time_ratio = distance_wanted * expected_time) := 
sorry

end lily_jog_time_l1788_178865


namespace original_price_l1788_178868

theorem original_price (P S : ℝ) (h1 : S = 1.25 * P) (h2 : S - P = 625) : P = 2500 := by
  sorry

end original_price_l1788_178868


namespace remainder_sum_is_74_l1788_178844

-- Defining the values from the given conditions
def num1 : ℕ := 1234567
def num2 : ℕ := 890123
def divisor : ℕ := 256

-- We state the theorem to capture the main problem
theorem remainder_sum_is_74 : (num1 + num2) % divisor = 74 := 
sorry

end remainder_sum_is_74_l1788_178844


namespace maximum_value_of_expression_l1788_178890

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end maximum_value_of_expression_l1788_178890


namespace code_transformation_l1788_178807

def old_to_new_encoding (s : String) : String := sorry

theorem code_transformation :
  old_to_new_encoding "011011010011" = "211221121" := sorry

end code_transformation_l1788_178807


namespace parabola_distance_ratio_l1788_178889

open Real

theorem parabola_distance_ratio (p : ℝ) (M N : ℝ × ℝ)
  (h1 : p = 4)
  (h2 : M.snd ^ 2 = 2 * p * M.fst)
  (h3 : N.snd ^ 2 = 2 * p * N.fst)
  (h4 : (M.snd - 2 * N.snd) * (M.snd + 2 * N.snd) = 48) :
  |M.fst + 2| = 4 * |N.fst + 2| := sorry

end parabola_distance_ratio_l1788_178889


namespace not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l1788_178803

def vector_a : ℝ × ℝ := (3, 2)
def vector_vA : ℝ × ℝ := (3, -2)
def vector_vB : ℝ × ℝ := (2, 3)
def vector_vD : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem not_perpendicular_to_vA : dot_product vector_a vector_vA ≠ 0 := by sorry
theorem not_perpendicular_to_vB : dot_product vector_a vector_vB ≠ 0 := by sorry
theorem not_perpendicular_to_vD : dot_product vector_a vector_vD ≠ 0 := by sorry

end not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l1788_178803


namespace find_T_b_plus_T_neg_b_l1788_178833

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem find_T_b_plus_T_neg_b (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 :=
sorry

end find_T_b_plus_T_neg_b_l1788_178833


namespace find_angle_B_l1788_178885

def is_triangle (A B C : ℝ) : Prop :=
A + B > C ∧ B + C > A ∧ C + A > B

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Defining the problem conditions
lemma given_condition : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a := sorry
-- A triangle with sides a, b, c
lemma triangle_property : is_triangle a b c := sorry

-- The equivalent proof problem
theorem find_angle_B (h_triangle : is_triangle a b c) (h_cond : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) : 
    B = π / 6 := sorry

end find_angle_B_l1788_178885


namespace green_marbles_l1788_178838

theorem green_marbles :
  ∀ (total: ℕ) (blue: ℕ) (red: ℕ) (yellow: ℕ), 
  total = 164 →
  blue = total / 2 →
  red = total / 4 →
  yellow = 14 →
  (total - (blue + red + yellow)) = 27 :=
by
  intros total blue red yellow h_total h_blue h_red h_yellow
  sorry

end green_marbles_l1788_178838


namespace therapy_charge_l1788_178877

-- Define the charges
def first_hour_charge (S : ℝ) : ℝ := S + 50
def subsequent_hour_charge (S : ℝ) : ℝ := S

-- Define the total charge before service fee for 8 hours
def total_charge_8_hours_before_fee (F S : ℝ) : ℝ := F + 7 * S

-- Define the total charge including the service fee for 8 hours
def total_charge_8_hours (F S : ℝ) : ℝ := 1.10 * (F + 7 * S)

-- Define the total charge before service fee for 3 hours
def total_charge_3_hours_before_fee (F S : ℝ) : ℝ := F + 2 * S

-- Define the total charge including the service fee for 3 hours
def total_charge_3_hours (F S : ℝ) : ℝ := 1.10 * (F + 2 * S)

theorem therapy_charge (S F : ℝ) :
  (F = S + 50) → (1.10 * (F + 7 * S) = 900) → (1.10 * (F + 2 * S) = 371.87) :=
by {
  sorry
}

end therapy_charge_l1788_178877


namespace contradiction_assumption_l1788_178801

theorem contradiction_assumption (a : ℝ) (h : a < |a|) : ¬(a ≥ 0) :=
by 
  sorry

end contradiction_assumption_l1788_178801


namespace find_other_root_l1788_178899

theorem find_other_root (m : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, (x = -6 → (x^2 + m * x - 6 = 0))) → (x^2 + m * x - 6 = (x + 6) * (x - 1)) → (∀ x : ℝ, (x^2 + 5 * x - 6 = 0) → (x = -6 ∨ x = 1))) :=
sorry

end find_other_root_l1788_178899


namespace samson_fuel_calculation_l1788_178846

def total_fuel_needed (main_distance : ℕ) (fuel_rate : ℕ) (hilly_distance : ℕ) (hilly_increase : ℚ)
                      (detours : ℕ) (detour_distance : ℕ) : ℚ :=
  let normal_distance := main_distance - hilly_distance
  let normal_fuel := (fuel_rate / 70) * normal_distance
  let hilly_fuel := (fuel_rate / 70) * hilly_distance * hilly_increase
  let detour_fuel := (fuel_rate / 70) * (detours * detour_distance)
  normal_fuel + hilly_fuel + detour_fuel

theorem samson_fuel_calculation :
  total_fuel_needed 140 10 30 1.2 2 5 = 22.28 :=
by sorry

end samson_fuel_calculation_l1788_178846


namespace analytic_expression_on_1_2_l1788_178827

noncomputable def f : ℝ → ℝ :=
  sorry

theorem analytic_expression_on_1_2 (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  f x = Real.logb (1 / 2) (x - 1) :=
sorry

end analytic_expression_on_1_2_l1788_178827


namespace find_k_l1788_178873

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem find_k (k : ℝ) :
  let a : vector := (2, 3)
  let b : vector := (1, 4)
  let c : vector := (k, 3)
  orthogonal (a.1 + b.1, a.2 + b.2) c → k = -7 :=
by
  intros
  sorry

end find_k_l1788_178873


namespace product_representation_count_l1788_178802

theorem product_representation_count :
  let n := 1000000
  let distinct_ways := 139
  (∃ (a b c d e f : ℕ), 2^(a+b+c) * 5^(d+e+f) = n ∧ 
    a + b + c = 6 ∧ d + e + f = 6 ) → 
    139 = distinct_ways := 
by {
  sorry
}

end product_representation_count_l1788_178802


namespace difference_between_hit_and_unreleased_l1788_178869

-- Define the conditions as constants
def hit_songs : Nat := 25
def top_100_songs : Nat := hit_songs + 10
def total_songs : Nat := 80

-- Define the question, conditional on the definitions above
theorem difference_between_hit_and_unreleased : 
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5 :=
by
  sorry

end difference_between_hit_and_unreleased_l1788_178869


namespace Maria_ate_2_cookies_l1788_178863

theorem Maria_ate_2_cookies : 
  ∀ (initial_cookies given_to_friend given_to_family remaining_after_eating : ℕ),
  initial_cookies = 19 →
  given_to_friend = 5 →
  given_to_family = (initial_cookies - given_to_friend) / 2 →
  remaining_after_eating = initial_cookies - given_to_friend - given_to_family - 2 →
  remaining_after_eating = 5 →
  2 = 2 := by
  intros
  sorry

end Maria_ate_2_cookies_l1788_178863


namespace housewife_money_left_l1788_178800

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end housewife_money_left_l1788_178800


namespace max_z_value_l1788_178841

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x * y + y * z + z * x = -3) : z ≤ 2 := sorry

end max_z_value_l1788_178841


namespace greatest_value_x_is_correct_l1788_178898

noncomputable def greatest_value_x : ℝ :=
-8 + Real.sqrt 6

theorem greatest_value_x_is_correct :
  ∀ x : ℝ, (x ≠ 9) → ((x^2 - x - 90) / (x - 9) = 2 / (x + 6)) → x ≤ greatest_value_x :=
by
  sorry

end greatest_value_x_is_correct_l1788_178898


namespace initial_red_balls_l1788_178822

-- Define all the conditions as given in part (a)
variables (R : ℕ)  -- Initial number of red balls
variables (B : ℕ)  -- Number of blue balls
variables (Y : ℕ)  -- Number of yellow balls

-- The conditions
def conditions (R B Y total : ℕ) : Prop :=
  B = 2 * R ∧
  Y = 32 ∧
  total = (R - 6) + B + Y

-- The target statement proving R = 16 given the conditions
theorem initial_red_balls (R: ℕ) (B: ℕ) (Y: ℕ) (total: ℕ) 
  (h : conditions R B Y total): 
  total = 74 → R = 16 :=
by 
  sorry

end initial_red_balls_l1788_178822


namespace gift_combinations_l1788_178830

theorem gift_combinations (wrapping_paper_count ribbon_count card_count : ℕ)
  (restricted_wrapping : ℕ)
  (restricted_ribbon : ℕ)
  (total_combinations := wrapping_paper_count * ribbon_count * card_count)
  (invalid_combinations := card_count)
  (valid_combinations := total_combinations - invalid_combinations) :
  wrapping_paper_count = 10 →
  ribbon_count = 4 →
  card_count = 5 →
  restricted_wrapping = 10 →
  restricted_ribbon = 1 →
  valid_combinations = 195 :=
by
  intros
  sorry

end gift_combinations_l1788_178830


namespace candies_remaining_l1788_178851

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l1788_178851


namespace eq1_solutions_eq2_solutions_l1788_178855

theorem eq1_solutions (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ (x = 3 + Real.sqrt 6) ∨ (x = 3 - Real.sqrt 6) :=
by {
  sorry
}

theorem eq2_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ (x = 2) ∨ (x = 1) :=
by {
  sorry
}

end eq1_solutions_eq2_solutions_l1788_178855


namespace manuscript_age_in_decimal_l1788_178892

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end manuscript_age_in_decimal_l1788_178892


namespace sin_690_eq_neg_0_5_l1788_178872

theorem sin_690_eq_neg_0_5 : Real.sin (690 * Real.pi / 180) = -0.5 := by
  sorry

end sin_690_eq_neg_0_5_l1788_178872


namespace smallest_possible_b_l1788_178812

-- Definition of the polynomial Q(x)
def Q (x : ℤ) : ℤ := sorry -- Polynomial with integer coefficients

-- Initial conditions for b and Q
variable (b : ℤ) (hb : b > 0)
variable (hQ1 : Q 2 = b)
variable (hQ2 : Q 4 = b)
variable (hQ3 : Q 6 = b)
variable (hQ4 : Q 8 = b)
variable (hQ5 : Q 1 = -b)
variable (hQ6 : Q 3 = -b)
variable (hQ7 : Q 5 = -b)
variable (hQ8 : Q 7 = -b)

theorem smallest_possible_b : b = 315 :=
by
  sorry

end smallest_possible_b_l1788_178812


namespace alcohol_by_volume_l1788_178896

/-- Solution x is 10% alcohol by volume and is 50 ml.
    Solution y is 30% alcohol by volume and is 150 ml.
    We must prove the final solution is 25% alcohol by volume. -/
theorem alcohol_by_volume (vol_x vol_y : ℕ) (conc_x conc_y : ℕ) (vol_mix : ℕ) (conc_mix : ℕ) :
  vol_x = 50 →
  conc_x = 10 →
  vol_y = 150 →
  conc_y = 30 →
  vol_mix = vol_x + vol_y →
  conc_mix = 100 * (vol_x * conc_x + vol_y * conc_y) / vol_mix →
  conc_mix = 25 :=
by
  intros h1 h2 h3 h4 h5 h_cons
  sorry

end alcohol_by_volume_l1788_178896


namespace series_sum_equals_9_over_4_l1788_178814

noncomputable def series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

theorem series_sum_equals_9_over_4 :
  series_sum = 9 / 4 :=
sorry

end series_sum_equals_9_over_4_l1788_178814


namespace two_common_points_with_x_axis_l1788_178840

noncomputable def func (x d : ℝ) : ℝ := x^3 - 3 * x + d

theorem two_common_points_with_x_axis (d : ℝ) :
(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func x1 d = 0 ∧ func x2 d = 0) ↔ (d = 2 ∨ d = -2) :=
by
  sorry

end two_common_points_with_x_axis_l1788_178840


namespace cream_ratio_l1788_178837

theorem cream_ratio (john_coffee_initial jane_coffee_initial : ℕ)
  (john_drank john_added_cream jane_added_cream jane_drank : ℕ) :
  john_coffee_initial = 20 →
  jane_coffee_initial = 20 →
  john_drank = 3 →
  john_added_cream = 4 →
  jane_added_cream = 3 →
  jane_drank = 5 →
  john_added_cream / (jane_added_cream * 18 / (23 * 1)) = (46 / 27) := 
by
  intros
  sorry

end cream_ratio_l1788_178837


namespace pick_three_different_cards_in_order_l1788_178853

theorem pick_three_different_cards_in_order :
  (52 * 51 * 50) = 132600 :=
by
  sorry

end pick_three_different_cards_in_order_l1788_178853


namespace elixir_concentration_l1788_178810

theorem elixir_concentration (x a : ℝ) 
  (h1 : (x * 100) / (100 + a) = 9) 
  (h2 : (x * 100 + a * 100) / (100 + 2 * a) = 23) : 
  x = 11 :=
by 
  sorry

end elixir_concentration_l1788_178810


namespace power_relationship_l1788_178829

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l1788_178829


namespace smallest_base_for_80_l1788_178857

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l1788_178857


namespace strawberries_for_mom_l1788_178850

-- Define the conditions as Lean definitions
def dozen : ℕ := 12
def strawberries_picked : ℕ := 2 * dozen
def strawberries_eaten : ℕ := 6

-- Define the statement to be proven
theorem strawberries_for_mom : (strawberries_picked - strawberries_eaten) = 18 := by
  sorry

end strawberries_for_mom_l1788_178850


namespace trajectory_of_A_l1788_178818

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A_l1788_178818


namespace tommy_profit_l1788_178819

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit_l1788_178819


namespace percentage_below_cost_l1788_178884

variable (CP SP : ℝ)

-- Given conditions
def cost_price : ℝ := 5625
def more_for_profit : ℝ := 1800
def profit_percentage : ℝ := 0.16
def expected_SP : ℝ := cost_price + (cost_price * profit_percentage)
def actual_SP : ℝ := expected_SP - more_for_profit

-- Statement to prove
theorem percentage_below_cost (h1 : CP = cost_price) (h2 : SP = actual_SP) :
  (CP - SP) / CP * 100 = 16 := by
sorry

end percentage_below_cost_l1788_178884


namespace problem_part_1_problem_part_2_problem_part_3_l1788_178835

open Set

-- Definitions for the given problem conditions
def U : Set ℕ := { x | x > 0 ∧ x < 10 }
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6}
def D : Set ℕ := B ∩ C

-- Prove each part of the problem
theorem problem_part_1 :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

theorem problem_part_2 :
  D = {3, 4} ∧
  (∀ (s : Set ℕ), s ⊆ D ↔ s = ∅ ∨ s = {3} ∨ s = {4} ∨ s = {3, 4}) := by
  sorry

theorem problem_part_3 :
  (U \ D) = {1, 2, 5, 6, 7, 8, 9} := by
  sorry

end problem_part_1_problem_part_2_problem_part_3_l1788_178835


namespace inequality_x_n_l1788_178825

theorem inequality_x_n (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : n ≥ 2) : (1 - x)^n + (1 + x)^n < 2^n := 
sorry

end inequality_x_n_l1788_178825


namespace sin_beta_value_l1788_178805

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < β ∧ β < π / 2)
variable (h₂ : Real.sin α = 3 / 5)
variable (h₃ : Real.cos (β - α) = 12 / 13)

theorem sin_beta_value : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_value_l1788_178805


namespace max_f_alpha_side_a_l1788_178879

noncomputable def a_vec (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)
noncomputable def b_vec (α : ℝ) : ℝ × ℝ := (6 * Real.sin α + Real.cos α, 7 * Real.sin α - 2 * Real.cos α)

noncomputable def f (α : ℝ) : ℝ := (a_vec α).1 * (b_vec α).1 + (a_vec α).2 * (b_vec α).2

theorem max_f_alpha : ∀ α : ℝ, f α ≤ 4 * Real.sqrt 2 + 2 :=
by
sorry

theorem side_a (A : ℝ) (b c : ℝ) (h1 : f A = 6) (h2 : 1/2 * b * c * Real.sin A = 3) (h3 : b + c = 2 + 3 * Real.sqrt 2) : 
  ∃ a : ℝ, a = Real.sqrt 10 :=
by
sorry

end max_f_alpha_side_a_l1788_178879


namespace Sam_has_most_pages_l1788_178826

theorem Sam_has_most_pages :
  let pages_per_inch_miles := 5
  let inches_miles := 240
  let pages_per_inch_daphne := 50
  let inches_daphne := 25
  let pages_per_inch_sam := 30
  let inches_sam := 60

  let pages_miles := inches_miles * pages_per_inch_miles
  let pages_daphne := inches_daphne * pages_per_inch_daphne
  let pages_sam := inches_sam * pages_per_inch_sam
  pages_sam = 1800 ∧ pages_sam > pages_miles ∧ pages_sam > pages_daphne :=
by
  sorry

end Sam_has_most_pages_l1788_178826


namespace point_coordinates_l1788_178878

theorem point_coordinates (M : ℝ × ℝ) 
  (hx : abs M.2 = 3) 
  (hy : abs M.1 = 2) 
  (h_first_quadrant : 0 < M.1 ∧ 0 < M.2) : 
  M = (2, 3) := 
sorry

end point_coordinates_l1788_178878


namespace area_enclosed_by_graph_l1788_178828

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l1788_178828


namespace log_expression_is_zero_l1788_178821

noncomputable def log_expr : ℝ := (Real.logb 2 3 + Real.logb 2 27) * (Real.logb 4 4 + Real.logb 4 (1/4))

theorem log_expression_is_zero : log_expr = 0 :=
by
  sorry

end log_expression_is_zero_l1788_178821


namespace convert_to_spherical_l1788_178870

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then 3 * Real.pi / 2
           else if x > 0 then Real.arctan (y / x)
           else if y >= 0 then Real.arctan (y / x) + Real.pi
           else Real.arctan (y / x) - Real.pi
  (ρ, θ, φ)

theorem convert_to_spherical :
  rectangular_to_spherical (3 * Real.sqrt 2) (-4) 5 =
  (Real.sqrt 59, 2 * Real.pi + Real.arctan ((-4) / (3 * Real.sqrt 2)), Real.arccos (5 / Real.sqrt 59)) :=
by
  sorry

end convert_to_spherical_l1788_178870


namespace range_of_a_l1788_178834

open Real

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * exp x1 - a = 0) ∧ (x2 * exp x2 - a = 0)) ↔ -1 / exp 1 < a ∧ a < 0 :=
sorry

end range_of_a_l1788_178834


namespace chess_tournament_winner_l1788_178817

theorem chess_tournament_winner :
  ∀ (x : ℕ) (P₉ P₁₀ : ℕ),
  (x > 0) →
  (9 * x) = 4 * P₃ →
  P₉ = (x * (x - 1)) / 2 + 9 * x^2 →
  P₁₀ = (9 * x * (9 * x - 1)) / 2 →
  (9 * x^2 - x) * 2 ≥ 81 * x^2 - 9 * x →
  x = 1 →
  P₃ = 9 :=
by
  sorry

end chess_tournament_winner_l1788_178817


namespace fliers_left_l1788_178816

theorem fliers_left (initial_fliers : ℕ) (fraction_morning : ℕ) (fraction_afternoon : ℕ) :
  initial_fliers = 2000 → 
  fraction_morning = 1 / 10 → 
  fraction_afternoon = 1 / 4 → 
  (initial_fliers - initial_fliers * fraction_morning - 
  (initial_fliers - initial_fliers * fraction_morning) * fraction_afternoon) = 1350 := by
  intros initial_fliers_eq fraction_morning_eq fraction_afternoon_eq
  sorry

end fliers_left_l1788_178816


namespace determine_a_l1788_178811

theorem determine_a (x : ℝ) (n : ℕ) (h : x > 0) (h_ineq : x + a / x^n ≥ n + 1) : a = n^n := by
  sorry

end determine_a_l1788_178811


namespace part_i_l1788_178836

theorem part_i (n : ℕ) (h₁ : n ≥ 1) (h₂ : n ∣ (2^n - 1)) : n = 1 :=
sorry

end part_i_l1788_178836


namespace geometric_sequence_property_l1788_178854

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∀ m n : ℕ, a (m + n) = a m * a n / a 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h : geometric_sequence a) 
    (h4 : a 4 = 5) 
    (h8 : a 8 = 6) : 
    a 2 * a 10 = 30 :=
by
  sorry

end geometric_sequence_property_l1788_178854


namespace enrollment_difference_l1788_178867

theorem enrollment_difference 
  (Varsity_enrollment : ℕ)
  (Northwest_enrollment : ℕ)
  (Central_enrollment : ℕ)
  (Greenbriar_enrollment : ℕ) 
  (h1 : Varsity_enrollment = 1300) 
  (h2 : Northwest_enrollment = 1500)
  (h3 : Central_enrollment = 1800)
  (h4 : Greenbriar_enrollment = 1600) : 
  Varsity_enrollment < Northwest_enrollment ∧ 
  Northwest_enrollment < Greenbriar_enrollment ∧ 
  Greenbriar_enrollment < Central_enrollment → 
    (Greenbriar_enrollment - Varsity_enrollment = 300) :=
by
  sorry

end enrollment_difference_l1788_178867


namespace polygon_side_count_l1788_178824

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l1788_178824


namespace find_sum_l1788_178852

theorem find_sum (a b : ℝ) 
  (h₁ : (a + Real.sqrt b) + (a - Real.sqrt b) = -8) 
  (h₂ : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) : 
  a + b = 8 := 
sorry

end find_sum_l1788_178852


namespace inequality1_inequality2_l1788_178849

theorem inequality1 (x : ℝ) : 2 * x - 1 > x - 3 → x > -2 := by
  sorry

theorem inequality2 (x : ℝ) : 
  (x - 3 * (x - 2) ≥ 4) ∧ ((x - 1) / 5 < (x + 1) / 2) → -7 / 3 < x ∧ x ≤ 1 := by
  sorry

end inequality1_inequality2_l1788_178849


namespace sum_first_twelve_arithmetic_divisible_by_6_l1788_178809

theorem sum_first_twelve_arithmetic_divisible_by_6 
  (a d : ℕ) (h1 : a > 0) (h2 : d > 0) : 
  6 ∣ (12 * a + 66 * d) := 
by
  sorry

end sum_first_twelve_arithmetic_divisible_by_6_l1788_178809


namespace beta_max_success_ratio_l1788_178832

theorem beta_max_success_ratio :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < b ∧
    0 < c ∧ c < d ∧
    b + d ≤ 550 ∧
    (15 * a < 8 * b) ∧ (10 * c < 7 * d) ∧
    (21 * a + 16 * c < 4400) ∧
    ((a + c) / (b + d : ℚ) = 274 / 550) :=
sorry

end beta_max_success_ratio_l1788_178832


namespace complex_expression_l1788_178842

theorem complex_expression (z : ℂ) (h : z = (i + 1) / (i - 1)) : z^2 + z + 1 = -i := 
by 
  sorry

end complex_expression_l1788_178842


namespace find_percentage_l1788_178861

noncomputable def percentage_solve (x : ℝ) : Prop :=
  0.15 * 40 = (x / 100) * 16 + 2

theorem find_percentage (x : ℝ) (h : percentage_solve x) : x = 25 :=
by
  sorry

end find_percentage_l1788_178861


namespace shelves_of_picture_books_l1788_178862

theorem shelves_of_picture_books
   (total_books : ℕ)
   (books_per_shelf : ℕ)
   (mystery_shelves : ℕ)
   (mystery_books : ℕ)
   (total_mystery_books : mystery_books = mystery_shelves * books_per_shelf)
   (total_books_condition : total_books = 32)
   (mystery_books_condition : mystery_books = 5 * books_per_shelf) :
   (total_books - mystery_books) / books_per_shelf = 3 :=
by
  sorry

end shelves_of_picture_books_l1788_178862


namespace interval_of_increase_inequality_for_large_x_l1788_178831

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + log x

theorem interval_of_increase :
  ∀ x > 0, ∀ y > x, f y > f x :=
by
  sorry

theorem inequality_for_large_x (x : ℝ) (hx : x > 1) :
  (1/2) * x^2 + log x < (2/3) * x^3 :=
by
  sorry

end interval_of_increase_inequality_for_large_x_l1788_178831


namespace question1_question2_question3_l1788_178866

variables {a x1 x2 : ℝ}

-- Definition of the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Conditions
axiom a_positive : a > 0
axiom roots_exist : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0
axiom roots_real : x1 + x2 = -1 / a ∧ x1 * x2 = 1 / a

-- Question 1
theorem question1 : (1 + x1) * (1 + x2) = 1 :=
sorry

-- Question 2
theorem question2 : x1 < -1 ∧ x2 < -1 :=
sorry

-- Additional condition for question 3
axiom ratio_in_range : x1 / x2 ∈ Set.Icc (1 / 10 : ℝ) 10

-- Question 3
theorem question3 : a <= 1 / 4 :=
sorry

end question1_question2_question3_l1788_178866


namespace company_members_and_days_l1788_178883

theorem company_members_and_days {t n : ℕ} (h : t = 6) :
    n = (t * (t - 1)) / 2 → n = 15 :=
by
  intro hn
  rw [h] at hn
  simp at hn
  exact hn

end company_members_and_days_l1788_178883


namespace age_of_youngest_child_l1788_178847

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : x = 6 :=
sorry

end age_of_youngest_child_l1788_178847


namespace valid_four_digit_numbers_count_l1788_178845

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l1788_178845


namespace combined_weight_of_candles_l1788_178839

theorem combined_weight_of_candles (candles : ℕ) (weight_per_candle : ℕ) (total_weight : ℕ) :
  candles = 10 - 3 →
  weight_per_candle = 8 + 1 →
  total_weight = candles * weight_per_candle →
  total_weight = 63 :=
by
  intros
  subst_vars
  sorry

end combined_weight_of_candles_l1788_178839


namespace pen_price_ratio_l1788_178894

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l1788_178894


namespace arithmetic_mean_a8_a11_l1788_178856

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_a8_a11 {a : ℕ → ℝ} (h1 : geometric_sequence a (-2)) 
    (h2 : a 2 * a 6 = 4 * a 3) :
  ((a 7 + a 10) / 2) = -56 :=
sorry

end arithmetic_mean_a8_a11_l1788_178856


namespace ending_number_l1788_178813

theorem ending_number (h : ∃ n, 3 * n = 99 ∧ n = 33) : ∃ m, m = 99 :=
by
  sorry

end ending_number_l1788_178813


namespace select_student_B_l1788_178843

-- Define the average scores for the students A, B, C, D
def avg_A : ℝ := 85
def avg_B : ℝ := 90
def avg_C : ℝ := 90
def avg_D : ℝ := 85

-- Define the variances for the students A, B, C, D
def var_A : ℝ := 50
def var_B : ℝ := 42
def var_C : ℝ := 50
def var_D : ℝ := 42

-- Theorem stating the selected student should be B
theorem select_student_B (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ)
  (h_avg_A : avg_A = 85) (h_avg_B : avg_B = 90) (h_avg_C : avg_C = 90) (h_avg_D : avg_D = 85)
  (h_var_A : var_A = 50) (h_var_B : var_B = 42) (h_var_C : var_C = 50) (h_var_D : var_D = 42) :
  (avg_B = 90 ∧ avg_C = 90 ∧ avg_B ≥ avg_A ∧ avg_B ≥ avg_D ∧ var_B < var_C) → 
  (select_student = "B") :=
by
  sorry

end select_student_B_l1788_178843


namespace square_of_binomial_l1788_178875

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l1788_178875


namespace total_pages_book_l1788_178874

-- Define the conditions
def reading_speed1 : ℕ := 10 -- pages per day for first half
def reading_speed2 : ℕ := 5 -- pages per day for second half
def total_days : ℕ := 75 -- total days spent reading

-- This is the main theorem we seek to prove:
theorem total_pages_book (P : ℕ) 
  (h1 : ∃ D1 D2 : ℕ, D1 + D2 = total_days ∧ D1 * reading_speed1 = P / 2 ∧ D2 * reading_speed2 = P / 2) : 
  P = 500 :=
by
  sorry

end total_pages_book_l1788_178874


namespace average_a_b_l1788_178858

-- Defining the variables A, B, C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- The theorem stating that the average weight of a and b is 40 kg
theorem average_a_b (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : (A + B) / 2 = 40 :=
sorry

end average_a_b_l1788_178858


namespace calculate_expression_l1788_178876

theorem calculate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + (1 / 6)) = 57 :=
by
  sorry

end calculate_expression_l1788_178876


namespace remainder_3042_div_29_l1788_178808

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end remainder_3042_div_29_l1788_178808


namespace lines_intersect_at_point_l1788_178806

/-
Given two lines parameterized as:
Line 1: (x, y) = (2, 0) + s * (3, -4)
Line 2: (x, y) = (6, -10) + v * (5, 3)
Prove that these lines intersect at (242/29, -248/29).
-/

def parametric_line_1 (s : ℚ) : ℚ × ℚ :=
  (2 + 3 * s, -4 * s)

def parametric_line_2 (v : ℚ) : ℚ × ℚ :=
  (6 + 5 * v, -10 + 3 * v)

theorem lines_intersect_at_point :
  ∃ (s v : ℚ), parametric_line_1 s = parametric_line_2 v ∧ parametric_line_1 s = (242 / 29, -248 / 29) :=
sorry

end lines_intersect_at_point_l1788_178806


namespace smallest_geometric_number_l1788_178871

noncomputable def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_smallest_geometric_number (n : ℕ) : Prop :=
  n = 261

theorem smallest_geometric_number :
  ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10)) ∧
  (n / 100 = 2) ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  is_smallest_geometric_number n :=
by
  sorry

end smallest_geometric_number_l1788_178871


namespace large_beds_l1788_178895

theorem large_beds {L : ℕ} {M : ℕ} 
    (h1 : M = 2) 
    (h2 : ∀ (x : ℕ), 100 <= x → L = (320 - 60 * M) / 100) : 
  L = 2 :=
by
  sorry

end large_beds_l1788_178895


namespace train_length_proof_l1788_178887

noncomputable def length_of_train : ℝ := 450.09

theorem train_length_proof
  (speed_kmh : ℝ := 60)
  (time_s : ℝ := 27) :
  (speed_kmh * (5 / 18) * time_s = length_of_train) :=
by
  sorry

end train_length_proof_l1788_178887


namespace pipe_q_fills_cistern_in_15_minutes_l1788_178804

theorem pipe_q_fills_cistern_in_15_minutes :
  ∃ T : ℝ, 
    (1/12 * 2 + 1/T * 2 + 1/T * 10.5 = 1) → 
    T = 15 :=
by {
  -- Assume the conditions and derive T = 15
  sorry
}

end pipe_q_fills_cistern_in_15_minutes_l1788_178804


namespace proof_x_bounds_l1788_178897

noncomputable def x : ℝ :=
  1 / Real.logb (1 / 3) (1 / 2) +
  1 / Real.logb (1 / 3) (1 / 4) +
  1 / Real.logb 7 (1 / 8)

theorem proof_x_bounds : 3 < x ∧ x < 3.5 := 
by
  sorry

end proof_x_bounds_l1788_178897


namespace find_k_l1788_178882

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2 * k, 3)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem find_k : ∃ k : ℝ, dot_product a (vector_add (scalar_mult 2 a) (b k)) = 0 ∧ k = -8 :=
by
  sorry

end find_k_l1788_178882


namespace Hillary_sunday_minutes_l1788_178880

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end Hillary_sunday_minutes_l1788_178880


namespace solution_exists_l1788_178888

noncomputable def find_p_q : Prop :=
  ∃ p q : ℕ, (p^q - q^p = 1927) ∧ (p = 2611) ∧ (q = 11)

theorem solution_exists : find_p_q :=
sorry

end solution_exists_l1788_178888


namespace total_employees_l1788_178886

theorem total_employees (x : Nat) (h1 : x < 13) : 13 + 6 * x = 85 :=
by
  sorry

end total_employees_l1788_178886


namespace simplify_fraction_l1788_178820

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1788_178820


namespace result_of_operation_given_y_l1788_178864

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem result_of_operation_given_y :
  ∀ (y : ℤ), y = 11 → operation y 10 = 90 :=
by
  intros y hy
  rw [hy]
  show operation 11 10 = 90
  sorry

end result_of_operation_given_y_l1788_178864
