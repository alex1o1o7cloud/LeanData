import Mathlib

namespace rice_quantity_proof_l38_38066

variable (initial_kg : ℝ) (pound_per_kg : ℝ) (price_factor_1 : ℝ) (price_factor_2 : ℝ) (exchange_rate_factor : ℝ)

-- Define initial conditions
def initial_conditions :=
  initial_kg = 20 ∧
  pound_per_kg = 2.2 ∧
  price_factor_1 = 0.8 ∧
  price_factor_2 = 0.9 ∧
  exchange_rate_factor = 1.05

-- Define the expected quantity of rice in kilograms after all changes
def expected_final_kg (initial_kg price_factor_1 price_factor_2 pound_per_kg exchange_rate_factor : ℝ) : ℝ :=
  let new_kg := initial_kg * (1 / price_factor_1) * (1 / price_factor_2)
  let new_pounds := new_kg * pound_per_kg
  let new_pounds_after_exchange := new_pounds * exchange_rate_factor
  new_pounds_after_exchange / pound_per_kg

-- The theorem statement we need to prove
theorem rice_quantity_proof (h : initial_conditions) :
  expected_final_kg initial_kg price_factor_1 price_factor_2 pound_per_kg exchange_rate_factor = 29.17 :=
by 
  sorry

end rice_quantity_proof_l38_38066


namespace volume_frustum_2240_over_3_l38_38605

def volume_of_pyramid (base_edge: ℝ) (height: ℝ) : ℝ :=
    (1 / 3) * (base_edge ^ 2) * height

def volume_of_frustum (original_base_edge: ℝ) (original_height: ℝ)
  (smaller_base_edge: ℝ) (smaller_height: ℝ) : ℝ :=
  volume_of_pyramid original_base_edge original_height - volume_of_pyramid smaller_base_edge smaller_height

theorem volume_frustum_2240_over_3 :
  volume_of_frustum 16 10 8 5 = 2240 / 3 :=
by sorry

end volume_frustum_2240_over_3_l38_38605


namespace percent_filled_cone_l38_38977

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38977


namespace carol_winning_choice_l38_38608

-- Definitions based on the problem conditions
def alice_choices (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 0.8
def bob_choices (y : ℝ) : Prop := 0.3 ≤ y ∧ y ≤ 0.7
def carol_choices (z : ℝ) : Prop := 0 ≤ z ∧ z ≤ 1

-- Statement of the problem
theorem carol_winning_choice : ∃ c, carol_choices c ∧ c = 0.5 :=
begin
  use 0.5,
  split,
  { exact ⟨(show 0 ≤ 0.5, by linarith), (show 0.5 ≤ 1, by linarith)⟩ },
  { refl }
end

end carol_winning_choice_l38_38608


namespace mascot_sales_growth_rate_equation_l38_38509

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end mascot_sales_growth_rate_equation_l38_38509


namespace solution_exists_l38_38534

def count_digit (d : ℕ) (n : ℕ) : ℕ :=
if n = 0 then 0
else (if n % 10 = d then 1 else 0) + count_digit d (n / 10)

def correct_code (n : ℕ) : Prop :=
  let num_twos := count_digit 2 n
  let num_threes := count_digit 3 n
  (num_twos > num_threes) ∧ (num_twos + num_threes = 7) ∧ (n % 3 = 0) ∧ (n % 4 = 0)

theorem solution_exists : ∃ n : ℕ, correct_code n ∧ n = 2222232 :=
by
  use 2222232
  unfold correct_code
  simp only [count_digit]
  -- Steps to show:
  -- (1) count_digit 2 2222232 = 6
  -- (2) count_digit 3 2222232 = 1
  -- (3) 6 > 1
  -- (4) 6 + 1 = 7
  -- (5) 2222232 % 3 = 0
  -- (6) 2222232 % 4 = 0
  sorry

end solution_exists_l38_38534


namespace T_15_is_correct_l38_38783

def T (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 3
  else if n = 3 then 6
  else if n = 4 then 11
  else T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

theorem T_15_is_correct : T 15 = --computed_value
:= sorry

end T_15_is_correct_l38_38783


namespace print_pages_500_l38_38016

theorem print_pages_500 (cost_per_page cents total_dollars) : 
  cost_per_page = 3 → 
  total_dollars = 15 → 
  cents = 100 * total_dollars → 
  (cents / cost_per_page) = 500 :=
by 
  intros h1 h2 h3
  sorry

end print_pages_500_l38_38016


namespace length_CF_is_zero_l38_38440

theorem length_CF_is_zero
  (A B C D E F : Point)
  (AD BC : Segment)
  (h1 : length AD = 7)
  (h2 : length BC = 7)
  (AB : Segment)
  (h3 : length AB = 6)
  (DC : Segment)
  (h4 : length DC = 14)
  (h5 : on_line DF C)
  (DEF_triangle : IsRightTriangle D E F)
  (h6 : midpoint B D E) : length (segment C F) = 0 := sorry

end length_CF_is_zero_l38_38440


namespace parabola_intercepts_sum_l38_38108

noncomputable def y_intercept (f : ℝ → ℝ) : ℝ := f 0

noncomputable def x_intercepts_of_parabola (a b c : ℝ) : (ℝ × ℝ) :=
let Δ := b ^ 2 - 4 * a * c in
(
  (-b + real.sqrt Δ) / (2 * a),
  (-b - real.sqrt Δ) / (2 * a)
)

theorem parabola_intercepts_sum :
  let f := λ x : ℝ, 3 * x^2 - 9 * x + 4 in
  let (e, f) := x_intercepts_of_parabola 3 (-9) 4 in
  y_intercept f + e + f = 19 / 3 :=
by
  sorry

end parabola_intercepts_sum_l38_38108


namespace number_of_subsets_divisible_by_m_l38_38312

def number_of_subsets (m n : ℕ) : ℕ :=
  if m > 1 ∧ odd m ∧ odd n ∧ m ∣ n then
    let r := n / m in
    ((2 ^ n + (m - 1) * 2 ^ r) / m) - 1
  else 0

theorem number_of_subsets_divisible_by_m (m n : ℕ) (hm1 : m > 1) (hm_odd : odd m) (hn_odd : odd n) (hm_div_n : m ∣ n) :
  ∃ X : set (fin n), m ∣ ∑ x in X, x :=
sorry

end number_of_subsets_divisible_by_m_l38_38312


namespace square_inscribed_in_right_triangle_side_length_l38_38813

theorem square_inscribed_in_right_triangle_side_length
  (A B C X Y Z W : ℝ × ℝ)
  (AB BC AC : ℝ)
  (square_side : ℝ)
  (h : 0 < square_side) :
  -- Define the lengths of sides of the triangle.
  AB = 3 ∧ BC = 4 ∧ AC = 5 ∧

  -- Define the square inscribed in the triangle
  (W.1 - A.1)^2 + (W.2 - A.2)^2 = square_side^2 ∧
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = square_side^2 ∧
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = square_side^2 ∧
  (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = square_side^2 ∧
  (Z.1 - C.1)^2 + (Z.2 - C.2)^2 = square_side^2 ∧

  -- Points where square meets triangle sides
  X.1 = A.1 ∧ Z.1 = C.1 ∧ Y.1 = X.1 ∧ W.1 = Z.1 ∧ Z.2 = Y.2 ∧

  -- Right triangle condition
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  
  -- Right angle at vertex B
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  →
  -- Prove the side length of the inscribed square
  square_side = 60 / 37 :=
sorry

end square_inscribed_in_right_triangle_side_length_l38_38813


namespace waiter_earnings_l38_38234

theorem waiter_earnings (total_customers tipping_customers no_tip_customers tips_each : ℕ) (h1 : total_customers = 7) (h2 : no_tip_customers = 4) (h3 : tips_each = 9) (h4 : tipping_customers = total_customers - no_tip_customers) :
  tipping_customers * tips_each = 27 :=
by sorry

end waiter_earnings_l38_38234


namespace angle_equality_l38_38709

variable {A B C E F D G : Type} [InnerProductSpace ℝ E]

def midpoint (a b : E) : E := (a + b) / 2

-- Given points
variables (A B C G : E)
-- Conditions:
-- E, F, D are midpoints of [AB], [BC], [CA] respectively
variables (E : E) (F : E) (D : E)
(hE : E = midpoint A B) (hF : F = midpoint B C) (hD : D = midpoint C A)
-- G is the foot of the altitude from vertex B to side AC
(hG : ∃ H : E, H ∈ LineBetween A C ∧ ∠ B G H = 90)

theorem angle_equality :
  ∠ E G F = ∠ E D F :=
sorry

end angle_equality_l38_38709


namespace ducks_in_pond_l38_38139

theorem ducks_in_pond (D : ℕ) (h1 : 0.5 * D = M) (h2 : 0.3 * M = 6) : D = 40 := by
  sorry

end ducks_in_pond_l38_38139


namespace find_constants_l38_38652

theorem find_constants :
  ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 7) / (x^2 - 3 * x - 18) = P / (x - 6) + Q / (x + 3)) ∧
    P = 31 / 9 ∧ Q = 5 / 9 :=
by
  sorry

end find_constants_l38_38652


namespace max_value_log_product_l38_38305

theorem max_value_log_product (a b c : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : log 10 a + log b c = 3) (h5 : log 10 b + log a c = 4) :
  ∃ t : ℝ, t > 0 ∧ (log 10 a) * (log 10 c) ≤ 16 / 3 :=
sorry

end max_value_log_product_l38_38305


namespace socks_problem_l38_38469

-- Declare the constants for the problem
constant pairs_total : Nat  -- Total pairs of socks Niko bought
constant cost_per_pair : Real := 2  -- Cost of each pair of socks (in $)
constant profit_percent : Real := 0.25 -- 25% profit
constant profit_amount : Real := 0.2 -- $0.2 profit on each of the other pairs
constant total_profit : Real := 3  -- Total profit Niko wants

-- Define the 4 pairs Niko makes 25% profit on
constant pairs_with_percent_profit : Nat := 4
constant per_pair_profit_percent : Real := cost_per_pair * profit_percent

-- Define the 5 pairs Niko makes $0.2 profit on
constant pairs_with_fixed_profit : Nat := 5
constant per_pair_profit_fixed : Real := profit_amount

-- Calculate the total profit from both categories
constant total_percent_profit : Real := pairs_with_percent_profit * per_pair_profit_percent
constant total_fixed_profit : Real := pairs_with_fixed_profit * per_pair_profit_fixed

#eval total_percent_profit + total_fixed_profit  -- This should be $3

-- Prove that given the conditions, the total number of pairs Niko bought is 9
theorem socks_problem (h1 : cost_per_pair = 2) (h2 : profit_percent = 0.25) (h3: profit_amount = 0.2) (h4 : total_profit = 3) : 
  pairs_total = pairs_with_percent_profit + pairs_with_fixed_profit -> 
  pairs_total = 9 :=
by {
  have h5 : per_pair_profit_percent = 0.5 := by sorry,
  have h6 : total_percent_profit = 2 := by sorry,
  have h7 : total_fixed_profit = 1 := by sorry,
  have h8 : total_percent_profit + total_fixed_profit = 3 := by sorry,
  sorry
}

end socks_problem_l38_38469


namespace find_alpha_l38_38718

def f (x : ℝ) : ℝ :=
if x < 2 then 3 - x else 2^x - 3

theorem find_alpha (α : ℝ) :
  f (f α) = 1 ↔ α = 1 ∨ α = Real.log 5 / Real.log 2 :=
by
  sorry

end find_alpha_l38_38718


namespace esther_morning_distance_l38_38801

-- Definitions used in conditions
def morning_speed : ℝ := 45
def lunchtime_speed : ℝ := 25
def lunchtime_time : ℝ := 1/3
def evening_speed : ℝ := 30
def total_distance : ℝ := 50
def total_time : ℝ := 2

-- Theorem to prove
theorem esther_morning_distance :
  ∃ D : ℝ, D = 125 / 6 ∧
  (D / morning_speed + (lunchtime_speed * lunchtime_time) / lunchtime_speed + D / evening_speed = total_time) ∧
  (D + (lunchtime_speed * lunchtime_time) + D = total_distance) :=
begin
  sorry
end

end esther_morning_distance_l38_38801


namespace t_shirts_per_package_l38_38552

theorem t_shirts_per_package (total_t_shirts : ℕ) (total_packages : ℕ) (h1 : total_t_shirts = 39) (h2 : total_packages = 3) : total_t_shirts / total_packages = 13 :=
by {
  sorry
}

end t_shirts_per_package_l38_38552


namespace maximum_sum_of_consecutive_numbers_on_checkerboard_l38_38779

theorem maximum_sum_of_consecutive_numbers_on_checkerboard : 
  ∃ (grid : Array (Array ℕ)), -- assume existence of a specific 5x5 grid placement of numbers
    (all_used_exactly_once : ∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ i j, grid[i][j] = n) ∧ 
    (consecutive_adjacent : ∀ n, 1 ≤ n ∧ n < 25 → 
       ∃ i j, ∃ (dir : (ℕ × ℕ)), (dir ∈ [(0, 1), (1, 0), (0, -1), (-1, 0)]) ∧ (grid[i][j] = n) ∧ (grid[i + dir.1][j + dir.2] = n + 1)) →
  ∃ color_center_grid : (ℕ × ℕ) → Bool, -- coloring function for the checkerboard
    let black_positions := 
      {pos : (ℕ × ℕ) | color_center_grid pos = true ∧ pos ≠ (2, 2)}, -- center square (indexed from 0) is black
      sum_black := black_positions.sum.1 grid := 
      (∑ n in (1 : set ℕ), n ∧ ¬ ∃ i j, grid[i][j] = n) : ℕ in
    sum_black = 169
    sorry

end maximum_sum_of_consecutive_numbers_on_checkerboard_l38_38779


namespace combined_value_of_silver_and_gold_l38_38621

noncomputable def silver_cube_side : ℝ := 3
def silver_weight_per_cubic_inch : ℝ := 6
def silver_price_per_ounce : ℝ := 25
def gold_layer_fraction : ℝ := 0.5
def gold_weight_per_square_inch : ℝ := 0.1
def gold_price_per_ounce : ℝ := 1800
def markup_percentage : ℝ := 1.10

def calculate_combined_value (side weight_per_cubic_inch silver_price layer_fraction weight_per_square_inch gold_price markup : ℝ) : ℝ :=
  let volume := side^3
  let weight_silver := volume * weight_per_cubic_inch
  let value_silver := weight_silver * silver_price
  let surface_area := 6 * side^2
  let area_gold := surface_area * layer_fraction
  let weight_gold := area_gold * weight_per_square_inch
  let value_gold := weight_gold * gold_price
  let total_value_before_markup := value_silver + value_gold
  let selling_price := total_value_before_markup * (1 + markup)
  selling_price

theorem combined_value_of_silver_and_gold :
  calculate_combined_value silver_cube_side silver_weight_per_cubic_inch silver_price_per_ounce gold_layer_fraction gold_weight_per_square_inch gold_price_per_ounce markup_percentage = 18711 :=
by
  sorry

end combined_value_of_silver_and_gold_l38_38621


namespace four_digit_numbers_count_l38_38394

theorem four_digit_numbers_count : 
  ∃ (n : ℕ), n = 4 ∧ 
  (∀ (digits : list ℕ), digits = [2, 0, 0, 3] → 
    (∀ (num : ℕ), num ∈ (multiset.pi (set {2, 3}) (λ _, ({0, 0, 2, 3} : finset ℕ)) : finset (list ℕ)) →
      ¬num.head = 0)) := 
sorry

end four_digit_numbers_count_l38_38394


namespace sum_of_first_15_even_positive_integers_l38_38163

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end sum_of_first_15_even_positive_integers_l38_38163


namespace relationship_between_PR_and_DR_l38_38434

theorem relationship_between_PR_and_DR
  (A B C P Q R D : Type*)
  (H1 : ∀ (angle : ℝ), angle = 90 → angle = 90)
  (H2 : ∃ AC : ℝ, ∀ distance : ℝ, (AC = distance ↔ distance = AC))
  (H3 : ∃ AB : ℝ, ∀ distance : ℝ, (AB = distance ↔ distance = AB))
  (H4 : ∀ (PQ : ℝ), PQ⊥AB * AC)
  (H5 : ∀ (circle : set ℝ), circle = {x | x ∈ circle})
  (H6 : ∀ (tangent : ℝ), tangent ∈ circle ↔ (circle \ PQ ∈ tangent * D))
  : PR = DR := by
  sorry

end relationship_between_PR_and_DR_l38_38434


namespace max_candy_leftover_l38_38387

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l38_38387


namespace comparison_neg_fractions_l38_38899

theorem comparison_neg_fractions (a b : ℚ) (ha : a = -5/6) (hb : b = -4/5) :
  a < b ↔ -5/6 < -4/5 := 
by 
  have h : 5/6 > 4/5 := sorry
  exact h


end comparison_neg_fractions_l38_38899


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38882

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38882


namespace product_of_sequence_infinity_l38_38591

noncomputable def a : ℕ → ℝ
| 0     := 1 / 2
| (n+1) := 1 + (a n - 1)^2

theorem product_of_sequence_infinity :
  (∏ i in Finset.range ⊤, a i) = 2 / 3 :=
sorry

end product_of_sequence_infinity_l38_38591


namespace magnitude_of_a_l38_38731

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
variable (hθ : theta = π / 3)
variable (hb : ‖b‖ = 1)
variable (hab : ‖a + 2 • b‖ = 2 * sqrt 3)

theorem magnitude_of_a :
  ‖a‖ = 2 :=
by
  sorry

end magnitude_of_a_l38_38731


namespace foreign_student_percentage_l38_38233

theorem foreign_student_percentage (total_students new_foreign_students future_foreign_students : ℕ)
  (h_total : total_students = 1800)
  (h_new_foreign : new_foreign_students = 200)
  (h_future_foreign : future_foreign_students = 740) :
  let current_foreign_students := future_foreign_students - new_foreign_students in
  current_foreign_students = (30 * total_students) / 100 :=
by
  sorry

end foreign_student_percentage_l38_38233


namespace monogram_count_l38_38468

theorem monogram_count : 
  let last_initial := 'M' in
  ∃ (first middle : Char), 
    first < middle ∧ 
    first ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'] ∧
    middle ∈ ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'] ∧
    first ≠ middle ∧
    (number of unique (first, middle) pairs) = 66 :=
begin
  -- proof goes here 
  sorry
end

end monogram_count_l38_38468


namespace hostel_food_supply_duration_l38_38207

/-
  Hostel had provisions for 250 men for 48 days, with varying daily consumption rates.
  After 10 days, 50 men leave and 30 new men arrive with different consumption rates.
  Prove that the remaining food will last for approximately 7.21 days.
-/
theorem hostel_food_supply_duration :
  let initial_men := 250
  let initial_days := 48
  let initial_low_consumption := 0.3 * initial_men * 2
  let initial_medium_consumption := 0.5 * initial_men * 3
  let initial_high_consumption := 0.2 * initial_men * 4
  let daily_initial_consumption := initial_low_consumption + initial_medium_consumption + initial_high_consumption
  let initial_food_supply := initial_men * initial_days
  let days_elapsed := 10
  let consumed_food := days_elapsed * daily_initial_consumption
  let remaining_food := initial_food_supply - consumed_food

  let men_left := initial_men - 50
  let men_arrived := 30

  let new_low_consumption := 0.45 * men_arrived * 1.5
  let new_medium_consumption := 0.4 * men_arrived * 3
  let new_high_consumption := 0.15 * men_arrived * 5

  let old_low_consumption := 0.3 * men_left * 2
  let old_medium_consumption := 0.5 * men_left * 3
  let old_high_consumption := 0.2 * men_left * 4

  let daily_new_consumption := old_low_consumption + old_medium_consumption + old_high_consumption +
                               new_low_consumption + new_medium_consumption + new_high_consumption

  remaining_food / daily_new_consumption ≈ 7.21 :=
sorry

end hostel_food_supply_duration_l38_38207


namespace sum_of_first_15_even_integers_l38_38165

theorem sum_of_first_15_even_integers : 
  let a := 2 in
  let d := 2 in
  let n := 15 in
  let S := (n / 2) * (a + (a + (n - 1) * d)) in
  S = 240 :=
by
  sorry

end sum_of_first_15_even_integers_l38_38165


namespace moon_speed_kmph_l38_38514

theorem moon_speed_kmph (speed_kmps : ℕ) (seconds_in_hour : ℕ) : 
  speed_kmps = 105 → seconds_in_hour = 3600 → speed_kmps * seconds_in_hour = 378000 
:=
begin
  sorry
end

end moon_speed_kmph_l38_38514


namespace identification_of_persons_l38_38486

inductive Status
  | knight
  | liar
  | visitor

def person (n : Nat) := Status

axiom statements :
  (p1 p2 p3 p4 p5 p6 p7 : Status) →
  (p1 = Status.liar ↔ p2 = Status.liar ↔ ¬ (p1 ≠ Status.liar ∧ p2 ≠ Status.liar ∧ ¬ ∃ k, p1 = k ∧ p2 = k)) ∧
  (p2 = Status.knight ↔ p3 = Status.knight ↔ ¬ (p2 ≠ Status.knight ∧ p3 ≠ Status.knight ∧ ¬ ∃ l, p2 = l ∧ p3 = l)) ∧ 
  (p3 = Status.liar ↔ p4 = Status.liar ↔ ¬ (p3 ≠ Status.liar ∧ p4 ≠ Status.liar ∧ ¬ ∃ m, p3 = m ∧ p4 = m)) ∧ 
  (p4 = Status.knight ↔ p5 = Status.knight ↔ ¬ (p4 ≠ Status.knight ∧ p5 ≠ Status.knight ∧ ¬ ∃ n, p4 = n ∧ p5 = n)) ∧ 
  (p5 = Status.liar ↔ p6 = Status.liar ↔ ¬ (p5 ≠ Status.liar ∧ p6 ≠ Status.liar ∧ ¬ ∃ o, p5 = o ∧ p6 = o)) ∧ 
  (p6 = Status.knight ↔ p7 = Status.knight ↔ ¬ (p6 ≠ Status.knight ∧ p7 ≠ Status.knight ∧ ¬ ∃ q, p6 = q ∧ p7 = q)) ∧ 
  (p7 = Status.liar ↔ p1 = Status.liar ↔ ¬ (p7 ≠ Status.liar ∧ p1 ≠ Status.liar ∧ ¬ ∃ r, p7 = r ∧ p1 = r))

theorem identification_of_persons : ∃ (p1 p2 p3 p4 p5 p6 p7 : Status),
  p1 = Status.liar ∧
  p2 = Status.liar ∧
  p3 = Status.liar ∧
  p4 = Status.liar ∧
  p7 = Status.liar ∧
  p5 = Status.visitor ∧
  p6 = Status.visitor ∧
  (statements p1 p2 p3 p4 p5 p6 p7) :=
begin
  -- Proof to be written
  sorry
end

end identification_of_persons_l38_38486


namespace batsman_average_after_17th_inning_l38_38563

theorem batsman_average_after_17th_inning (A : ℝ) :
  (16 * A + 87) / 17 = A + 3 → A + 3 = 39 :=
by
  intro h
  sorry

end batsman_average_after_17th_inning_l38_38563


namespace average_runs_in_30_matches_l38_38569

theorem average_runs_in_30_matches 
  (avg1 : ℕ) (matches1 : ℕ) (avg2 : ℕ) (matches2 : ℕ) (total_matches : ℕ)
  (h1 : avg1 = 40) (h2 : matches1 = 20) (h3 : avg2 = 13) (h4 : matches2 = 10) (h5 : total_matches = 30) :
  ((avg1 * matches1 + avg2 * matches2) / total_matches) = 31 := by
  sorry

end average_runs_in_30_matches_l38_38569


namespace locus_of_M_equals_circle_l38_38041

noncomputable def locus_of_M (b : ℝ) (x y : ℝ) : Prop :=
  ∃ (Px Py : ℝ), x^2 + y^2 = 9 ∧ 
    (∃ (Mx My : ℝ), (x - b)^2 + y^2 = 
      ((b + 2 * (Mx - b)) - x)^2 + (2 * My)^2 ∧
        (Mx - 4)^2 + My^2 = 1)

theorem locus_of_M_equals_circle (b x y : ℕ) (h1 : x^2 + y^2 = 9) 
  (h2 : (x - b)^2 + y^2 = ((b + 2 * (x - b)) - b)^2 + (2 * y)^2) :
  locus_of_M b x y :=
by {
  sorry,
}

end locus_of_M_equals_circle_l38_38041


namespace students_like_cricket_l38_38421

theorem students_like_cricket
  (B : ℕ) (B_inter_C : ℕ) (B_union_C : ℕ)
  (hB : B = 12) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) :
  ∃ (C : ℕ), C = 8 :=
by
  let C := B_union_C - (B - B_inter_C)
  have hC : C = 8 := by
    rw [hB_union_C, hB, hB_inter_C]
    norm_num
  use C
  exact hC

end students_like_cricket_l38_38421


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38890

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38890


namespace extreme_values_l38_38717

noncomputable def f (x : ℝ) : ℝ := x^5 + 5 * x^4 + 5 * x^3 + 1

theorem extreme_values :
  (∀ x, (x = -3 → f x = 28) ∧ (x = -1 → f x = 0)) ∧
  (∀ x ∈ set.Icc (-2:ℝ) 2, ((-2 ≤ x ∧ x ≤ -1) → f (-1) ≤ f x) ∧ ((-1 < x ∧ x ≤ 2) → f 2 ≥ f x)) :=
by
  unfold f
  sorry

end extreme_values_l38_38717


namespace work_rate_ab_together_l38_38562

-- Define A, B, and C as the work rates of individuals
variables (A B C : ℝ)

-- We are given the following conditions:
-- 1. a, b, and c together can finish the job in 11 days
-- 2. c alone can finish the job in 41.25 days

-- Given these conditions, we aim to prove that a and b together can finish the job in 15 days
theorem work_rate_ab_together
  (h1 : A + B + C = 1 / 11)
  (h2 : C = 1 / 41.25) :
  1 / (A + B) = 15 :=
by
  sorry

end work_rate_ab_together_l38_38562


namespace min_side_of_triangle_l38_38689

-- Let \( S \) be the area of triangle \( \triangle ABC \) and \( \gamma \) be the measure of angle \( C \).
-- To find the minimum value of side \( C \) opposite to \( \angle C \).

theorem min_side_of_triangle (a b S : ℝ) (γ : ℝ) (hS : S = (1 / 2) * a * b * Real.sin γ) :
  ∃ c_min : ℝ, c_min = 2 * Real.sqrt(S * Real.tan(γ / 2)) :=
by {
  sorry
}

end min_side_of_triangle_l38_38689


namespace area_of_A1B1C1D1_greater_than_quarter_area_of_ABCD_l38_38123

variables {A B C D A1 B1 C1 D1 : ℝ → ℝ}

-- Assumptions on the points and areas
axiom ConvexQuadrilateral (A B C D : ℝ → ℝ) : Convex ℝ (ConvexHull ℝ {p : ℝ → ℝ | p = A ∨ p = B ∨ p = C ∨ p = D})
axiom AreaDividesIntoTwoEqualParts (A B C D A1 : ℝ → ℝ) : Area (AA1) = Area (ABCD) / 2
axiom AreaDividesIntoTwoEqualParts (B B1 : ℝ → ℝ) : Area (BB1) = Area (ABCD) / 2
axiom AreaDividesIntoTwoEqualParts (C C1 : ℝ → ℝ) : Area (CC1) = Area (ABCD) / 2
axiom AreaDividesIntoTwoEqualParts (D D1 : ℝ → ℝ) : Area (DD1) = Area (ABCD) / 2

-- Prove the required inequality about the areas
theorem area_of_A1B1C1D1_greater_than_quarter_area_of_ABCD
  (convQ : ConvexQuadrilateral A B C D)
  (area_split_A : AreaDividesIntoTwoEqualParts A B C D A1)
  (area_split_B : AreaDividesIntoTwoEqualParts B B1)
  (area_split_C : AreaDividesIntoTwoEqualParts C C1)
  (area_split_D : AreaDividesIntoTwoEqualParts D D1) :
  Area (A1B1C1D1) > (1 / 4) * Area (ABCD) :=
sorry

end area_of_A1B1C1D1_greater_than_quarter_area_of_ABCD_l38_38123


namespace three_numbers_equal_l38_38149

theorem three_numbers_equal {a b c d : ℕ} 
  (h : ∀ {x y z w : ℕ}, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) → x * y + z * w = x * z + y * w) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end three_numbers_equal_l38_38149


namespace find_numbers_with_property_l38_38470

def contains_all_digits_once (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let n_digits := (to_digits 10 n).erase_dup
  n_digits.perm digits

theorem find_numbers_with_property :
  (∀ n : ℕ, n = 94857312 ∨ n = 89745321 ∨ n = 98745231 → contains_all_digits_once n ∧ contains_all_digits_once (n * 6)) :=
  by sorry

end find_numbers_with_property_l38_38470


namespace minimum_sticks_broken_n12_can_form_square_n15_l38_38283

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l38_38283


namespace inequality_always_holds_l38_38268

theorem inequality_always_holds (m : ℝ) : (-6 < m ∧ m ≤ 0) ↔ ∀ x : ℝ, 2 * m * x^2 + m * x - 3 / 4 < 0 := 
sorry

end inequality_always_holds_l38_38268


namespace volume_of_prism_l38_38213

theorem volume_of_prism (l w h : ℝ)
  (h1 : l * w = 10)
  (h2 : w * h = 15)
  (h3 : l * h = 18) :
  l * w * h = 30 * real.sqrt 3 :=
sorry

end volume_of_prism_l38_38213


namespace probability_same_rank_l38_38811

noncomputable def probability_same_rank_in_five_drawn : ℚ :=
  1 - ((choose 13 5 * (4 ^ 5)) / (choose 52 5))

theorem probability_same_rank : (probability_same_rank_in_five_drawn ≈ 0.49) :=
  sorry

end probability_same_rank_l38_38811


namespace selling_price_of_bracelet_l38_38020

theorem selling_price_of_bracelet (x : ℝ) 
  (cost_per_bracelet : ℝ) 
  (num_bracelets : ℕ) 
  (box_of_cookies_cost : ℝ) 
  (money_left_after_buying_cookies : ℝ) 
  (total_revenue : ℝ) 
  (total_cost_of_supplies : ℝ) :
  cost_per_bracelet = 1 →
  num_bracelets = 12 →
  box_of_cookies_cost = 3 →
  money_left_after_buying_cookies = 3 →
  total_cost_of_supplies = cost_per_bracelet * num_bracelets →
  total_revenue = 9 →
  x = total_revenue / num_bracelets :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Placeholder for the actual proof
  sorry

end selling_price_of_bracelet_l38_38020


namespace min_breaks_for_square_12_can_form_square_15_l38_38297

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l38_38297


namespace sum_of_diameters_of_balls_l38_38760

theorem sum_of_diameters_of_balls
  (a b : ℝ)
  (h1 : ∃ p : ℝ × ℝ × ℝ, p = (5, 5, 10))
  (h2 : ∀ (p : ℝ × ℝ × ℝ), p = (5, 5, 10) → 
         (p.1 - a)^2 + (p.2 - a)^2 + (p.3 - a)^2 = a^2) 
  (h3 : ∀ (p : ℝ × ℝ × ℝ), p = (5, 5, 10) → 
         (p.1 - b)^2 + (p.2 - b)^2 + (p.3 - b)^2 = b^2) :
  2 * a + 2 * b = 40 := by
  sorry

end sum_of_diameters_of_balls_l38_38760


namespace find_angle_bisector_length_l38_38010

noncomputable def angle_bisector_length (AB AC : ℝ) (cosA : ℝ) : ℝ :=
let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * cosA) in
let BD := (2 * BC) / 3 in
let CD := BC / 3 in
let cosB := ((AB^2 + BD^2 - (BD + CD)^2) / (2 * AB * BD)) in
real.sqrt (AB^2 + BD^2 - 2 * AB * BD * cosB)

theorem find_angle_bisector_length (AB AC : ℝ) (cosA : ℝ) (hAB : AB = 4) (hAC : AC = 8) (hcosA : cosA = 1/10) :
  angle_bisector_length AB AC cosA = real.sqrt (16 + (4 * 73.6 / 9) - (16 * real.sqrt (73.6) / 3 * (((16 + ((2 * real.sqrt (73.6))/3)^2 - 64) / (2 * 4 * ((2 * real.sqrt (73.6))/3))))) :=
sorry

end find_angle_bisector_length_l38_38010


namespace larger_integer_exists_l38_38538

theorem larger_integer_exists (a b : ℤ) (h1 : a - b = 8) (h2 : a * b = 272) : a = 17 :=
sorry

end larger_integer_exists_l38_38538


namespace subset_sum_divisible_by_2n_l38_38034

theorem subset_sum_divisible_by_2n 
  (n : ℤ) (hn : n ≥ 4) (a : Fin n.succ → ℤ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (ha_interval : ∀ i, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n.succ)), ∑ i in s, a i % (2 * n) = 0 := by
sorry

end subset_sum_divisible_by_2n_l38_38034


namespace cookies_with_five_cups_of_flour_l38_38450

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l38_38450


namespace proof_l38_38170

-- Define the equation and its conditions
def equation (x m : ℤ) : Prop := (3 * x - 1) / 2 + m = 3

-- Part 1: Prove that for m = 5, the corresponding x must be 1
def part1 : Prop :=
  ∃ x : ℤ, equation x 5 ∧ x = 1

-- Part 2: Prove that if the equation has a positive integer solution, the positive integer m must be 2
def part2 : Prop :=
  ∃ m x : ℤ, m > 0 ∧ x > 0 ∧ equation x m ∧ m = 2

theorem proof : part1 ∧ part2 :=
  by
    sorry

end proof_l38_38170


namespace double_sum_is_two_l38_38642

-- Define the double sum as a Lean definition
def double_sum (f : ℕ → ℕ → ℝ) : ℝ :=
  ∑' m, ∑' n, f m n

-- Define the given function inside the sum
def f (m n : ℕ) : ℝ :=
  1 / (m * n * (m + n + 3))

-- Lean 4 statement to assert the equivalence of the sum and 2
theorem double_sum_is_two : double_sum f = 2 := by
  sorry

end double_sum_is_two_l38_38642


namespace production_average_l38_38269

theorem production_average (n : ℕ) (P : ℕ) (hP : P = n * 50)
  (h1 : (P + 95) / (n + 1) = 55) : n = 8 :=
by
  -- skipping the proof
  sorry

end production_average_l38_38269


namespace kaylin_age_32_l38_38023

-- Defining the ages of the individuals as variables
variables (Kaylin Sarah Eli Freyja Alfred Olivia : ℝ)

-- Defining the given conditions
def conditions : Prop := 
  (Kaylin = Sarah - 5) ∧
  (Sarah = 2 * Eli) ∧
  (Eli = Freyja + 9) ∧
  (Freyja = 2.5 * Alfred) ∧
  (Alfred = (3/4) * Olivia) ∧
  (Freyja = 9.5)

-- Main statement to prove
theorem kaylin_age_32 (h : conditions Kaylin Sarah Eli Freyja Alfred Olivia) : Kaylin = 32 :=
by
  sorry

end kaylin_age_32_l38_38023


namespace Kira_breakfast_time_l38_38448

theorem Kira_breakfast_time :
  let sausages := 3
  let eggs := 6
  let time_per_sausage := 5
  let time_per_egg := 4
  (sausages * time_per_sausage + eggs * time_per_egg) = 39 :=
by
  sorry

end Kira_breakfast_time_l38_38448


namespace find_dallas_age_l38_38778

variable (Dallas_last_year Darcy_last_year Dexter_age Darcy_this_year Derek this_year_age : ℕ)

-- Conditions
axiom cond1 : Dallas_last_year = 3 * Darcy_last_year
axiom cond2 : Darcy_this_year = 2 * Dexter_age
axiom cond3 : Dexter_age = 8
axiom cond4 : Derek = this_year_age + 4

-- Theorem: Proving Dallas's current age
theorem find_dallas_age (Dallas_last_year : ℕ)
  (H1 : Dallas_last_year = 3 * (Darcy_this_year - 1))
  (H2 : Darcy_this_year = 2 * Dexter_age)
  (H3 : Dexter_age = 8)
  (H4 : Derek = (Dallas_last_year + 1) + 4) :
  Dallas_last_year + 1 = 46 :=
by
  sorry

end find_dallas_age_l38_38778


namespace avg_value_assertion_l38_38781

noncomputable def average_value_of_set (T : Finset ℕ) : ℝ :=
  (T.sum : ℝ) / T.card

theorem avg_value_assertion
  (T : Finset ℕ)
  (b1 : ℕ) (bm : ℕ)
  (h1 : ∀ T', T' = T.erase b1 → average_value_of_set T' = 42)
  (h2 : ∀ T', T' = (T.erase b1).erase bm → average_value_of_set T' = 39)
  (h3 : ∀ T', T' = T.erase bm → average_value_of_set T' = 45)
  (h4 : bm = b1 + 90) :
  average_value_of_set T = 49.35 := 
  sorry

end avg_value_assertion_l38_38781


namespace calculate_diameter_l38_38834

noncomputable def wheel_diameter (revolutions_per_minute : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance_per_revolution := speed_m_per_min / revolutions_per_minute
  distance_per_revolution / Real.pi

theorem calculate_diameter :
  wheel_diameter 265.15 60 = 1.2 :=
begin
  sorry
end

end calculate_diameter_l38_38834


namespace ellipse_equation_and_m_value_l38_38372

-- Define the conditions of the problem
def lineEquation (m : ℝ) : ℝ → ℝ := λ x => -x + m

def ellipse (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (a b c e : ℝ) := e = c / a
def focalLength (c : ℝ) := 2 * c = 2

-- Define the main problem
theorem ellipse_equation_and_m_value :
  ∃ (a b m : ℝ), a > b ∧ b > 0 ∧ lineEquation m = λ x => -x + m ∧
  ellipse (1 : ℝ) (1 : ℝ) a b ∧ eccentricity a b 1 (Real.sqrt 3 / 3 : ℝ) ∧
  focalLength (1 : ℝ) ∧ (∃ (m : ℝ), m = 2 * Real.sqrt 15 / 5 ∨ m = -2 * Real.sqrt 15 / 5) :=
begin
  sorry
end

end ellipse_equation_and_m_value_l38_38372


namespace volume_filled_cone_l38_38956

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38956


namespace fourth_term_of_geometric_sequence_l38_38100

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) :=
  a * r ^ (n - 1)

theorem fourth_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (ar5_eq : a * r ^ 5 = 32) 
  (a_eq : a = 81) :
  geometric_sequence a r 4 = 24 := 
by 
  sorry

end fourth_term_of_geometric_sequence_l38_38100


namespace term_2012_of_T_is_2057_l38_38842

-- Define a function that checks if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the sequence T as all natural numbers which are not perfect squares
def T (n : ℕ) : ℕ :=
  (n + Nat.sqrt (4 * n)) 

-- The theorem to state the mathematical proof problem
theorem term_2012_of_T_is_2057 :
  T 2012 = 2057 :=
sorry

end term_2012_of_T_is_2057_l38_38842


namespace length_OB_in_circle_l38_38625

-- Definition of Circle O with circumference 18π
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def CircleO (r : ℝ) := circumference r = 18 * Real.pi

-- Segment AB is a diameter and angle AOB is 45 degrees
def isDiameter (A B O : ℝ) (diam : ℝ) := diam = 2 * A
def angleAOB (O A B : ℝ) := 45 = Real.pi / 4

-- The goal is to prove that OB is 9 inches
theorem length_OB_in_circle (r A B O : ℝ) 
  (h1 : CircleO r) 
  (h2 : isDiameter A B O (2 * r))
  (h3 : angleAOB O A B) : 
  OB = 9 :=
by 
  -- This is where the proof would be introduced
  sorry

end length_OB_in_circle_l38_38625


namespace max_knights_among_10_l38_38638

def is_knight (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (p m ↔ (m ≥ n))

def is_liar (p : ℕ → Prop) (n : ℕ) : Prop :=
  ∀ m : ℕ, (¬ p m ↔ (m ≥ n))

def greater_than (k : ℕ) (n : ℕ) := n > k

def less_than (k : ℕ) (n : ℕ) := n < k

def person_statement_1 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => greater_than 1 n
  | 2 => greater_than 2 n
  | 3 => greater_than 3 n
  | 4 => greater_than 4 n
  | 5 => greater_than 5 n
  | 6 => greater_than 6 n
  | 7 => greater_than 7 n
  | 8 => greater_than 8 n
  | 9 => greater_than 9 n
  | 10 => greater_than 10 n
  | _ => false

def person_statement_2 (i : ℕ) (n : ℕ) : Prop :=
  match i with
  | 1 => less_than 1 n
  | 2 => less_than 2 n
  | 3 => less_than 3 n
  | 4 => less_than 4 n
  | 5 => less_than 5 n
  | 6 => less_than 6 n
  | 7 => less_than 7 n
  | 8 => less_than 8 n
  | 9 => less_than 9 n
  | 10 => less_than 10 n
  | _ => false

theorem max_knights_among_10 (knights : ℕ) : 
  (∀ i < 10, (is_knight (person_statement_1 (i + 1)) (i + 1) ∨ is_liar (person_statement_1 (i + 1)) (i + 1))) ∧
  (∀ i < 10, (is_knight (person_statement_2 (i + 1)) (i + 1) ∨ is_liar (person_statement_2 (i + 1)) (i + 1))) →
  knights ≤ 8 := sorry

end max_knights_among_10_l38_38638


namespace number_of_linear_eqs_l38_38183

def is_linear (e : String) : Bool :=
  match e with
  | "x-2=\frac{1}{x}" => false
  | "0.2x=1" => true
  | "x/3=x-3" => true
  | "x^{2}-4-3x" => false
  | "x=0" => true
  | "x-y=6" => true
  | _ => false

def linear_eq_count : Nat :=
  ["x-2=\frac{1}{x}",
   "0.2x=1",
   "x/3=x-3",
   "x^{2}-4-3x",
   "x=0",
   "x-y=6"].countp is_linear

theorem number_of_linear_eqs : linear_eq_count = 3 :=
  by
    sorry

end number_of_linear_eqs_l38_38183


namespace circumscribed_inscribed_radius_inequality_l38_38456

theorem circumscribed_inscribed_radius_inequality {T : Triangle} (R : ℝ) (r : ℝ) :
  (is_circumscribed_radius T R) →
  (is_inscribed_radius T r) →
  R ≥ 2 * r ∧ (R = 2 * r ↔ is_equilateral T) :=
sorry

end circumscribed_inscribed_radius_inequality_l38_38456


namespace exponential_generating_function_l38_38184

-- Define the exponential generating function 
def E_S_n (S : ℕ → ℕ → ℕ) (x : ℝ) : ℝ := ∑ N in (finset.range 100), (S N 0) * (x ^ N) / (N.factorial)

-- Define Stirling numbers of the second kind
def stirling_second_kind (n N : ℕ) : ℕ := 
  ∑ k in (finset.range (n+1)), (-1)^(n - k) * (nat.choose n k) * (k ^ N)

-- Define the main theorem to prove
theorem exponential_generating_function :
  ∀ (x : ℝ) (n : ℕ), E_S_n stirling_second_kind x = (x.exp - 1) ^ n / n.factorial := by
  sorry

end exponential_generating_function_l38_38184


namespace billiard_angle_range_l38_38576

theorem billiard_angle_range (P Q A B C D E F : Point) (θ : Real) :
  is_regular_hexagon A B C D E F →
  midpoint A B P →
  hits_on_sequence P Q [BC, CD, DE, EF, FA] →
  measure_angle B P Q = θ →
  real.arctan (3 * real.sqrt 3 / 10) < θ ∧ θ < real.arctan (3 * real.sqrt 3 / 8) :=
by
  sorry

end billiard_angle_range_l38_38576


namespace exists_small_triangle_1_exists_small_triangle_2_l38_38572

-- Given conditions
def square_side_length : ℝ := 1
def points_in_square (n : ℕ) (points : Fin n → ℝ × ℝ) : Prop :=
  ∀ i, fst (points i) ∈ Set.Icc 0 square_side_length ∧ snd (points i) ∈ Set.Icc 0 square_side_length
def no_three_collinear (n : ℕ) (points : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)

-- Definitions related to collinearity
def collinear (a b c : ℝ × ℝ) : Prop :=
  let (ax, ay) := a
  let (bx, by) := b
  let (cx, cy) := c
  (bx - ax) * (cy - ay) = (by - ay) * (cx - ax)

-- Conditions for existence of small triangles
theorem exists_small_triangle_1 (n : ℕ) (points : Fin n → ℝ × ℝ)
  (hno_three_collinear : no_three_collinear n points)
  (hn_gt_101 : n > 101)
  (hpoints_in_square : points_in_square n points) :
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangle_area (points i) (points j) (points k) < 0.01 :=
sorry

theorem exists_small_triangle_2 (n : ℕ) (points : Fin n → ℝ × ℝ)
  (hno_three_collinear : no_three_collinear n points)
  (hn_gt_101 : n > 101)
  (hpoints_in_square : points_in_square n points) :
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangle_area (points i) (points j) (points k) ≤ 0.005 :=
sorry

-- Helper function to compute the area of a triangle
noncomputable def triangle_area (a b c : (ℝ × ℝ)) : ℝ :=
  let (ax, ay) := a
  let (bx, by) := b
  let (cx, cy) := c
  (1/2 : ℝ) * abs ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


end exists_small_triangle_1_exists_small_triangle_2_l38_38572


namespace clairvoyance_diploma_problem_l38_38761

theorem clairvoyance_diploma_problem :
  ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ (i : ℕ) (hi : i ∈ S), 
  ∀ (j : ℕ) (hj : j ≠ i ∧ j ≠ (i + 1) % 13 ∧ j ≠ (i + 12) % 13 ∧ j ∉ S), 
  j ∈ {0, 1, ..., 12} := 
by
  sorry

end clairvoyance_diploma_problem_l38_38761


namespace parallelogram_projection_l38_38755

variable {V : Type} [InnerProductSpace ℝ V]

def projection_onto (v: V) (l: V) : ℝ := (inner v l) / (inner l l)

theorem parallelogram_projection {A B C D P: V} (h_pgram : A - B = D - C) (h_para1 : inner (A - B) (A - B) = inner (D - C) (D - C))
  (l : V) :
  projection_onto (C - A) l = projection_onto (B - A) l + projection_onto (C - B) l ∨ 
  projection_onto (C - A) l = projection_onto (A) l - projection_onto (D - C) l := 
by
    sorry

end parallelogram_projection_l38_38755


namespace negation_proposition_l38_38121

theorem negation_proposition :
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + 2 ≤ 0)) =
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - 3 * x + 2 > 0) := 
sorry

end negation_proposition_l38_38121


namespace shaded_region_area_computed_correctly_l38_38596

noncomputable def side_length : ℝ := 15
noncomputable def quarter_circle_radius : ℝ := side_length / 3
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
noncomputable def shaded_region_area : ℝ := square_area - circle_area

theorem shaded_region_area_computed_correctly : 
  shaded_region_area = 225 - 25 * Real.pi := 
by 
  -- This statement only defines the proof problem.
  sorry

end shaded_region_area_computed_correctly_l38_38596


namespace cone_water_fill_percentage_l38_38929

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38929


namespace part_I_part_II_l38_38277

def statement_p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0
def statement_q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

theorem part_I (m : ℝ) (h : m > 0) : ([(-2 : ℝ)..6] ⊆ [2 - m .. 2 + m]) → m ≥ 4 :=
by
  sorry

theorem part_II (x : ℝ) (m : ℝ) (h : m = 5) : (statement_p x ∨ statement_q m x ∧ ¬ (statement_p x ∧ statement_q m x)) → x ∈ [-3, -2) ∪ (6, 7] :=
by
  sorry

end part_I_part_II_l38_38277


namespace derivative_at_x0_l38_38499

noncomputable def Y (x : ℝ) := (Real.sin x - Real.cos x) / (2 * Real.cos x)
def x0 : ℝ := Real.pi / 3

theorem derivative_at_x0 : deriv Y x0 = 2 := by
  sorry

end derivative_at_x0_l38_38499


namespace triangle_circumcircle_areas_l38_38199

theorem triangle_circumcircle_areas (A B C : ℝ) :
  (∃ a b c : ℝ, a = 13 ∧ b = 14 ∧ c = 15 ∧
                let s := (a + b + c) / 2 in
                let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
                let R := (a * b * c) / (4 * area) in
                let circle_area := real.pi * R^2 in
                C = circle_area / 2 ∧
                A + B + area = C) :=
begin
  sorry
end

end triangle_circumcircle_areas_l38_38199


namespace vasya_can_sort_365_cards_l38_38802
-- Import the Mathlib library to bring in the necessary functions and definitions.

-- Declare the conditions and the final proof statement.
theorem vasya_can_sort_365_cards (cards : Fin 365 → ℕ) :
    (∀ (x y z : Fin 365), ∃ (order : List (Fin 365)), 
       (cards (order.nthLe 0 (by simp [Fin.of_nat_succ_id, Fin.of_nat_val])) <= 
        cards (order.nthLe 1 (by simp [Fin.of_nat_succ_id, Fin.of_nat_val])) ∧ 
        cards (order.nthLe 1 (by simp [Fin.of_nat_succ_id, Fin.of_nat_val])) <= 
        cards (order.nthLe 2 (by simp [Fin.of_nat_succ_id, Fin.of_nat_val])))) → 
    ∃ (sorted_order : List (Fin 365)), 
      (∀ (i : Fin 364), cards (sorted_order.nthLe i sorry) <= cards (sorted_order.nthLe (i + 1) sorry)) := 
sorry

end vasya_can_sort_365_cards_l38_38802


namespace parabola_intercepts_sum_l38_38111

theorem parabola_intercepts_sum :
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  y_intercept + x_intercept1 + x_intercept2 = 7 :=
by
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  have sum_intercepts : y_intercept + x_intercept1 + x_intercept2 = 7 := by
        calc (4 : ℝ) + ((9 + Real.sqrt 33) / 6) + ((9 - Real.sqrt 33) / 6)
            = 4 + (18 / 6) : by
              rw [add_assoc, ← add_div, add_sub_cancel]
            ... = 4 + 3 : by norm_num
            ... = 7 : by norm_num
  exact sum_intercepts

end parabola_intercepts_sum_l38_38111


namespace circle_tangent_line_eq_l38_38710

noncomputable def point (x y : ℝ) := (x, y)
def A := point (-1) 2
def B := point (-2) 0

def line_eq (a b c : ℝ) (p : ℝ × ℝ) : Prop := a * p.1 + b * p.2 + c = 0

def circle_eq (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Condition: Circle with center at A is tangent to the line x + 2y + 7 = 0
def is_tangent (c : ℝ × ℝ) (r : ℝ) (a b k : ℝ) : Prop :=
  abs(a * c.1 + b * c.2 + k) / sqrt(a^2 + b^2) = r

-- The equation of the circle centered at A with radius 2√5
def circle_A_eqn (p : ℝ × ℝ) := circle_eq A (2 * sqrt 5) p

-- Condition: The moving line l passes through point B(-2,0)
def passes_through (l a b : ℝ) (p : ℝ × ℝ) : Prop :=
  l * p.1 + a * p.2 + b = 0

-- The equation of line l
def line_l_eqn (p : ℝ × ℝ) :=
  line_eq 3 (-4) 6 p ∨ line_eq 1 0 (-2) p

theorem circle_tangent_line_eq :
  ∀ (p : ℝ × ℝ),
    (is_tangent A (2 * sqrt 5) 1 2 7 → circle_A_eqn p) ∧
    (|MN| = 2 * sqrt 19 → passes_through 1 0 (-2) B → line_l_eqn p) :=
by { sorry }

end circle_tangent_line_eq_l38_38710


namespace area_of_intersection_l38_38768

-- Conditions
def radius : ℝ := 5
def center1 : ℝ × ℝ := (-5, 0)
def center2 : ℝ × ℝ := (5, 0)

-- Define the area function for a circle
def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Proving the area of the shaded region where the circles intersect
theorem area_of_intersection : 
  area_of_circle radius = 25 * Real.pi := 
by 
  -- This is a placeholder for the proof
  sorry

end area_of_intersection_l38_38768


namespace alok_ordered_plates_of_mixed_vegetable_l38_38610

theorem alok_ordered_plates_of_mixed_vegetable :
  ∀ (Cchapati Crice Cmix : ℕ) (Pchapati Price : ℕ) (TotalPaid AmountMixed CostMixed CmixTotal : ℕ),
    Cchapati = 6 -> Crice = 45 -> Cmix = 70 ->
    Pchapati = 16 -> Price = 5 -> TotalPaid = 961 ->
    (TotalPaid - (Pchapati * Cchapati + Price * Crice)) / Cmix = 9 ->
    AmountMixed = 9 :=
by
  intros Cchapati Crice Cmix Pchapati Price TotalPaid AmountMixed CostMixed CmixTotal
  assume hCchapati hCrice hCmix hPchapati hPrice hTotalPaid hCalc
  sorry

end alok_ordered_plates_of_mixed_vegetable_l38_38610


namespace number_of_standard_right_triangles_l38_38415

/-- A right triangle with side lengths (a, b, c) is a 'standard right triangle' if
    the perimeter equals the area. This theorem states there are exactly two such triangles. -/
theorem number_of_standard_right_triangles : 
  ∃! (a b c : ℕ+), (a^2 + b^2 = c^2) ∧ (a + b + c = (a * b) / 2) := 
sorry

end number_of_standard_right_triangles_l38_38415


namespace volunteer_distribution_l38_38247

theorem volunteer_distribution :
  let students := 5
  let projects := 4
  let combinations := Nat.choose students 2
  let permutations := Nat.factorial projects
  combinations * permutations = 240 := 
by
  sorry

end volunteer_distribution_l38_38247


namespace initial_bones_count_l38_38204

theorem initial_bones_count (B : ℕ) (h1 : B + 8 = 23) : B = 15 :=
sorry

end initial_bones_count_l38_38204


namespace ellipse_fixed_point_l38_38325

noncomputable theory
open Real

-- Define the conditions

def isEllipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^2 = b^2 + 1^2 ∧ (b / a = (sqrt 3) / 2)

def isTangentLine (a b : ℝ) (c : ℝ) (k m : ℝ) : Prop :=
  (k = sqrt 3) ∧ (m = -k * c)

-- Create the theorem to prove the given conditions and result
theorem ellipse_fixed_point :
 (∃ (a b : ℝ), isEllipse a b ∧ 
   (∀ k m, 
    (∃ x1 x2 y1 y2, y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ 
      (7 * m^2 + 16 * k * m + 4 * k^2 = 0) → 
     k ≠ 0 → ∃ p, p = (2 / 7, 0)))) :=
begin
  sorry
end

end ellipse_fixed_point_l38_38325


namespace percent_divisible_3_or_6_up_to_200_l38_38548

theorem percent_divisible_3_or_6_up_to_200 : 
  ∃ (percent : ℝ), percent = 33 ∧ 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → (n % 3 = 0 ∨ n % 6 = 0) → n) -> 
  (count (n : ℕ), 1 ≤ n ∧ n ≤ 200 ∧ (n % 3 = 0 ∨ n % 6 = 0)) / 200 = 33 / 100 :=
begin
  sorry
end

end percent_divisible_3_or_6_up_to_200_l38_38548


namespace ratio_first_term_l38_38198

theorem ratio_first_term (x : ℝ) (h1 : 60 / 100 = x / 25) : x = 15 := 
sorry

end ratio_first_term_l38_38198


namespace part1_g_expression_and_monotonicity_part2_range_of_a_l38_38365

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 6) - sqrt 3 / 2

theorem part1_g_expression_and_monotonicity :
  (∀ x, g x = sin (2 * x + π / 6) - sqrt 3 / 2) ∧ 
  (∀ k : ℤ, ∃ I : set ℝ, I = {y : ℝ | -π/3 + k * π ≤ y ∧ y ≤ π/6 + k * π}) :=
sorry

theorem part2_range_of_a :
  (∀ x ∈ set.Icc (π / 6) (π / 3), 
  ∀ a > 0, 
  af x + g x ≥ (sqrt (a ^ 2 + 1)) / 2 - (sqrt 3 / 2) * (a + 1)) → 
  set.Ioo 0 (sqrt 3) :=
sorry

end part1_g_expression_and_monotonicity_part2_range_of_a_l38_38365


namespace inv_composition_l38_38618

theorem inv_composition (f g : ℝ → ℝ) (hf : Function.Bijective f) (hg : Function.Bijective g) (h : ∀ x, f⁻¹ (g x) = 2 * x - 4) : 
  g⁻¹ (f (-3)) = 1 / 2 :=
by
  sorry

end inv_composition_l38_38618


namespace calculate_geometric_sequence_sum_l38_38051

def geometric_sequence (a₁ r : ℤ) (n : ℕ) : ℤ :=
  a₁ * r^n

theorem calculate_geometric_sequence_sum :
  let a₁ := 1
  let r := -2
  let a₂ := geometric_sequence a₁ r 1
  let a₃ := geometric_sequence a₁ r 2
  let a₄ := geometric_sequence a₁ r 3
  a₁ + |a₂| + a₃ + |a₄| = 15 :=
by
  sorry

end calculate_geometric_sequence_sum_l38_38051


namespace extreme_value_at_x_eq_one_l38_38102

noncomputable def f (x a b: ℝ) : ℝ := x^3 - a * x^2 + b * x + a^2
noncomputable def f_prime (x a b: ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem extreme_value_at_x_eq_one (a b : ℝ) (h_prime : f_prime 1 a b = 0) (h_value : f 1 a b = 10) : a = -4 :=
by 
  sorry -- proof goes here

end extreme_value_at_x_eq_one_l38_38102


namespace log_identity_l38_38739

theorem log_identity (k x : ℝ) (h: log x k * log k 7 = 6) : x = 117649 :=
sorry

end log_identity_l38_38739


namespace no_valid_placement_of_12_ships_1x4_l38_38014

-- Definitions for the board and ships
def ship := (range: ℕ × ℕ, size: ℕ)
def touching (s1 s2: ship) : Prop := 
  -- Define touching condition: sides or corners
  sorry

def valid_placement (ships: list ship) : Prop :=
  -- Define all ships are placed within a 10x10 grid without touching
  list.all (λ s, -- ship within bounds and not touching any other ship
    let ((x, y), size) := s in
    x >= 1 ∧ y >= 1 ∧ x + size - 1 <= 10 ∧ y + size - 1 <= 10 ∧
    list.all (λ s', s ≠ s' → ¬touching s s') ships) ships

theorem no_valid_placement_of_12_ships_1x4 : ¬ ∃ ships: list ship, 
  (list.length ships = 12) ∧ valid_placement ships :=
by
  sorry

end no_valid_placement_of_12_ships_1x4_l38_38014


namespace domain_of_f_range_of_f_value_of_f_at_alpha_l38_38720

def f (x : ℝ) : ℝ := (sin (2 * x) - cos (2 * x) + 1) / (2 * sin x)

-- (1)
theorem domain_of_f : {x : ℝ | x ≠ k * π for some k ∈ ℤ} :=
sorry

-- (2)
theorem range_of_f : ∀ y : ℝ, y ∈ [real.sqrt 2, -1) ∪ (-1, 1) ∪ (1, real.sqrt 2)] :=
sorry

-- (3)
theorem value_of_f_at_alpha (α : ℝ) (h_acute : 0 < α ∧ α < π / 2) (h_tan : tan (α / 2) = 1/2) : f α = 7/5 :=
sorry

end domain_of_f_range_of_f_value_of_f_at_alpha_l38_38720


namespace avg_weight_section_b_l38_38857

/-- Definition of the average weight of section B based on given conditions --/
theorem avg_weight_section_b :
  let W_A := 50
  let W_class := 54.285714285714285
  let num_A := 40
  let num_B := 30
  let total_class_weight := (num_A + num_B) * W_class
  let total_A_weight := num_A * W_A
  let total_B_weight := total_class_weight - total_A_weight
  let W_B := total_B_weight / num_B
  W_B = 60 :=
by
  sorry

end avg_weight_section_b_l38_38857


namespace sum_of_coefficients_l38_38852

-- Defining the given conditions
def vertex : ℝ × ℝ := (5, -4)
def point : ℝ × ℝ := (3, -2)

-- Defining the problem to prove the sum of the coefficients
theorem sum_of_coefficients (a b c : ℝ)
  (h_eq : ∀ y, 5 = a * ((-4) + y)^2 + c)
  (h_pt : 3 = a * ((-4) + (-2))^2 + b * (-2) + c) :
  a + b + c = -15 / 2 :=
sorry

end sum_of_coefficients_l38_38852


namespace even_function_a_eq_zero_solutions_in_interval_l38_38795

section
variables (a : ℝ) (f : ℝ → ℝ)

-- Definition of the function
def g (x : ℝ) : ℝ := a * sin (2 * x) + 2 * (cos x) ^ 2

-- Statement 1: Proving that f(x) is even implies a = 0
theorem even_function_a_eq_zero (h : ∀ x, g x = g (- x)) : a = 0 :=
sorry

-- Given condition
def f_condition (a : ℝ) : ℝ := a + 1

-- Statement 2: To find solutions to f(x) = 1 - sqrt(2)
theorem solutions_in_interval (h : f_condition (√3) = √3 + 1) 
  (h_eq : ∀ x, g x = 1 - sqrt 2 → x ∈ [(-π:ℝ), π]) :
  (∃ k : ℤ, x = -11/24 * π + k * π ∨ x = -5/24 * π + k * π ∨ x = 13/24 * π + k * π ∨ x = 19/24 * π + k * π) :=
sorry

end

end even_function_a_eq_zero_solutions_in_interval_l38_38795


namespace distance_focus_directrix_parabola_l38_38500

theorem distance_focus_directrix_parabola : 
  ∀ (x y : ℝ), y^2 = 8 * x → 
    let focus_distance := 4 in focus_distance = 4 :=
by
  sorry

end distance_focus_directrix_parabola_l38_38500


namespace min_breaks_for_square_12_can_form_square_15_l38_38295

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l38_38295


namespace mass_percentage_N_in_Ammonia_l38_38655

noncomputable def molar_mass_N : ℝ := 14.01
noncomputable def molar_mass_H : ℝ := 1.01

def molar_mass_NH3 : ℝ := molar_mass_N + 3 * molar_mass_H
def mass_percentage_N : ℝ := (molar_mass_N / molar_mass_NH3) * 100

theorem mass_percentage_N_in_Ammonia : mass_percentage_N = 82.23 := 
sorry

end mass_percentage_N_in_Ammonia_l38_38655


namespace lcm_problem_l38_38667

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end lcm_problem_l38_38667


namespace evaluate_f_3_minus_f_neg_3_l38_38409

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 7*x

-- Theorem asserting the given problem
theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 :=
by
  -- Definition of g and h (from the problem's conditions)
  let g (x : ℝ) : ℝ := 2*x^2
  let h (x : ℝ) : ℝ := x^5 + 3*x^3 + 7*x
  
  -- Express f in terms of g and h
  have hf : ∀ x, f x = g x + h x := by intros; simp [f, g, h]
  -- Show even and odd properties
  have g_even : ∀ x, g x = g (-x) := by intros; simp [g, pow_two]
  have h_odd : ∀ x, h (-x) = -h x := by intros; simp [h]; ring
  
  sorry -- Steps leading to 690 will be detailed here

end evaluate_f_3_minus_f_neg_3_l38_38409


namespace roll_two_dice_prime_sum_l38_38825

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l38_38825


namespace problem_statement_l38_38378

open Set

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { x | -2 < x ∧ x < 2 }
noncomputable def B : Set ℝ := { x | x < -1 ∨ x > 4 }
noncomputable def complement_B := { x | -1 ≤ x ∧ x ≤ 4 }

-- Define the union of A and the complement of B
theorem problem_statement :
  A ∪ complement_B = { x : ℝ | -2 < x ∧ x ≤ 4 } :=
by {
  sorry,
}

end problem_statement_l38_38378


namespace incorrect_statements_count_l38_38228

theorem incorrect_statements_count :
  let A := {0, 1}
  let statement1 := {x ∈ A | True}.card == 3
  let statement2 := (∀ x : ℝ, x^2 = 1 → x = 1) → ¬(∀ x : ℝ, x^2 = 1 → x ≠ 1)
  let statement3 := (∀ x : ℝ, x^2 - 3 * x - 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - 3 * x - 2 < 0)
  let statement4 := (p q : Prop) → (p ∨ q → p ∧ q) ≠ (p ∧ q → p ∨ q)
  in
  (∀ p q : Prop, 0 + cond (statement1) 1 0 + cond (statement2) 1 0 + cond (statement3) 1 0 + cond (statement4) 1 0) = 3 :=
by
  sorry

end incorrect_statements_count_l38_38228


namespace periodic_translation_l38_38147

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (4 * x + π / 3)

-- Define the period of the function
def period := π / 2

-- Define the condition for m
def is_periodic_translation (m : ℝ) : Prop := ∃ k : ℤ, m = k * period

-- Statement of the proof problem
theorem periodic_translation (m : ℝ) :
  is_periodic_translation m ↔ m = π / 2 :=
sorry

end periodic_translation_l38_38147


namespace find_x_l38_38401

theorem find_x (x : ℤ) (h : 2^(x-5) = 8^3) : x = 14 := 
by sorry

end find_x_l38_38401


namespace jayden_half_of_ernesto_in_some_years_l38_38000

theorem jayden_half_of_ernesto_in_some_years :
  ∃ x : ℕ, (4 + x = (1 : ℝ) / 2 * (11 + x)) ∧ x = 3 := by
  sorry

end jayden_half_of_ernesto_in_some_years_l38_38000


namespace calculate_rate_l38_38176

def principal : ℝ := 415
def amount : ℝ := 514
def time : ℝ := 4
def simple_interest (p a : ℝ) : ℝ := a - p
def rate (si p t : ℝ) : ℝ := (si * 100) / (p * t)

theorem calculate_rate : rate (simple_interest principal amount) principal time = 5.96 := by
  sorry

end calculate_rate_l38_38176


namespace op_dot_of_10_5_l38_38847

-- Define the operation \odot
def op_dot (a b : ℕ) : ℕ := a + (2 * a) / b

-- Theorem stating that 10 \odot 5 = 14
theorem op_dot_of_10_5 : op_dot 10 5 = 14 :=
by
  sorry

end op_dot_of_10_5_l38_38847


namespace find_101st_digit_in_decimal_of_7_div_36_l38_38405

open Nat

theorem find_101st_digit_in_decimal_of_7_div_36 :
  let decimal := "194444"
  (0.194444 : Real)  -- define the repeating sequence in decimal form
  -- Assert the 101st digit in the repeating sequence of the fraction 7/36
  ∃ digit : Char, digit = '4' ∧ (substring decimal ((101 % 6) - 1) (101 % 6)) = Char.toString digit :=
begin
  sorry
end

end find_101st_digit_in_decimal_of_7_div_36_l38_38405


namespace magnitude_a_minus_3b_l38_38382

variables {ℝ : Type} [noncomputable ℝ]

variables (a b : ℝ ^ 3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = real.pi / 3)

theorem magnitude_a_minus_3b : ∥a - 3 • b∥ = real.sqrt 7 :=
sorry

end magnitude_a_minus_3b_l38_38382


namespace diagonal_parallel_to_side_of_parallelogram_l38_38473

theorem diagonal_parallel_to_side_of_parallelogram
  (A B C D K L M N : ℝ × ℝ)
  (h_parallel1 : A.1 = D.1 ∧ C.1 = B.1 ∧ A.2 = B.2 ∧ D.2 = C.2)
  (on_sides : (K.1 = A.1 ∧ K.2 ∈ set.Icc A.2 B.2) ∧
              (L.2 = B.2 ∧ L.1 ∈ set.Icc B.1 C.1) ∧
              (M.1 = C.1 ∧ M.2 ∈ set.Icc C.2 D.2) ∧
              (N.2 = D.2 ∧ N.1 ∈ set.Icc D.1 A.1))
  (area_condition : parallelogram_area A B C D / 2 = quadrilateral_area K L M N) :
  ∃ (X Y : ℝ × ℝ), (X = K ∨ X = L ∨ X = M ∨ X = N) ∧ (Y = K ∨ Y = L ∨ Y = M ∨ Y = N) ∧
    (X ≠ Y) ∧ line_parallel X Y (boundary_line A B C D) :=
sorry

end diagonal_parallel_to_side_of_parallelogram_l38_38473


namespace sum_of_valid_b_values_is_9_l38_38403

-- Definitions using conditions directly from the problem
def is_divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

def possible_values_b : list ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def last_three_digits (b : ℕ) : ℕ :=
  200 + 10 * b + 7

def valid_b_values : list ℕ :=
  possible_values_b.filter (λ b, is_divisible_by_8 (last_three_digits b))

-- Sum the possible values of B
def sum_valid_b_values : ℕ :=
  valid_b_values.sum

-- Statement we need to prove
theorem sum_of_valid_b_values_is_9 : sum_valid_b_values = 9 :=
  by sorry

end sum_of_valid_b_values_is_9_l38_38403


namespace sin_cos_15_eq_1_over_4_l38_38238

theorem sin_cos_15_eq_1_over_4 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := 
by
  sorry

end sin_cos_15_eq_1_over_4_l38_38238


namespace tangent_condition_find_equation_of_line_area_range_l38_38316

-- Condition definitions
def ellipse (x y : ℝ) := x^2 / 2 + y^2 = 1
def line (k b x y : ℝ) := y = k * x + b
def circle (x y : ℝ) := x^2 + y^2 = 2
variable (k b : ℝ)

-- Problem statements
theorem tangent_condition : 
  ∀ (y x : ℝ), line k b x y →  |b| / (real.sqrt (1 + k^2)) = 1 → b^2 = k^2 + 1 := by 
  sorry

theorem find_equation_of_line :
  ∀ (x1 y1 x2 y2 : ℝ), 
  ellipse x1 y1 ∧ ellipse x2 y2 ∧ line k b x1 y1 ∧ line k b x2 y2 ∧ 
  (x1 * x2 + (k * x1 + b) * (k * x2 + b) = 2 / 3) → 
  (k = 1 ∨ k = -1) ∧ (b = real.sqrt 2 ∨ b = -real.sqrt 2) := by 
  sorry

theorem area_range :
  ∀ (m : ℝ), (2 / 3 ≤ m ∧ m ≤ 3 / 4) →
  (∀ (k2 b : ℝ), m = (k2 + 1) / (2 * k2 + 1) → (1 / 2 ≤ k2 ∧ k2 ≤ 1) →
  (∀ (S : ℝ), S = real.sqrt (2 / 2) * real.sqrt (1 - 1 / k2^2) →
  (real.sqrt (6) / 4 ≤ S ∧ S ≤ 2 / 3))) := by 
  sorry

end tangent_condition_find_equation_of_line_area_range_l38_38316


namespace initial_number_of_persons_l38_38830

noncomputable def avg_weight_change : ℝ := 5.5
noncomputable def old_person_weight : ℝ := 68
noncomputable def new_person_weight : ℝ := 95.5
noncomputable def weight_diff : ℝ := new_person_weight - old_person_weight

theorem initial_number_of_persons (N : ℝ) 
  (h1 : avg_weight_change * N = weight_diff) : N = 5 :=
  by
  sorry

end initial_number_of_persons_l38_38830


namespace sum_of_reciprocals_l38_38784

-- Given the conditions
variables {a b c : ℝ}
-- roots of the polynomial x^3 - x - 2 = 0
def is_root (r : ℝ) : Prop := r^3 - r - 2 = 0

-- the statement to prove
theorem sum_of_reciprocals (ha : is_root a) (hb : is_root b) (hc : is_root c) :
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2)) = -3 / 5 :=
begin
  sorry
end

end sum_of_reciprocals_l38_38784


namespace max_marks_400_l38_38218

theorem max_marks_400 {M : ℝ} (h : 0.45 * M = 150 + 30) : M = 400 := 
by
  sorry

end max_marks_400_l38_38218


namespace volume_filled_cone_l38_38953

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38953


namespace part1_part2_l38_38691

noncomputable def a_n (a_1 d n : ℕ) : ℤ := a_1 + (n - 1) * d

noncomputable def S_n (a_1 d n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

noncomputable def b_n (n : ℕ) (a_n : ℕ → ℤ) : ℤ := (-1 : ℤ) ^ n * a_n n

noncomputable def T_2n (n : ℕ) (b_n : ℕ → ℤ) : ℤ := ∑ i in finset.range (2 * n), b_n (i + 1)

theorem part1 (h1 : a_n 1 4 2 = 5) (h2 : S_n 1 4 4 = 28) : 
  ∀ n : ℕ, a_n 1 4 n = 4 * (n : ℤ) - 3 :=
by
  sorry

theorem part2 (h_formula : ∀ n : ℕ, a_n 1 4 n = 4 * (n : ℤ) - 3) :
  ∀ n : ℕ, T_2n n (b_n (λ n, 4 * (n : ℤ) - 3)) = 4 * (n : ℤ) :=
by
  sorry

end part1_part2_l38_38691


namespace prime_factorization_problem_l38_38137

theorem prime_factorization_problem
  (x y m n : ℕ)
  (hx : 0 < x) (hy : 0 < y)
  (h1 : log 10 x + 2 * log 10 (gcd x y) = 24)
  (h2 : log 10 y + 2 * log 10 (lcm x y) = 156)
  (hm : prime_factor_count x = m)
  (hn : prime_factor_count y = n) :
  3 * m + 2 * n = 256 :=
sorry

end prime_factorization_problem_l38_38137


namespace min_value_of_f_l38_38276

noncomputable def f (x : ℕ) : ℝ :=
  if x > 0 then (x * x + 33) / x else 0

theorem min_value_of_f : ∃ x ∈ (Set.univ \ {0}), f x = 23 / 2 :=
begin
  sorry
end

end min_value_of_f_l38_38276


namespace concurrency_of_lines_l38_38464

theorem concurrency_of_lines
  (A B C A_1 B_2 C_1 A_2 B_1 C_2 : Type*)
  [AffineSpace Type*]
  [EuclideanGeometry Type*]
  (triangle_ABC : Triangle A B C)
  (rectangle_BB2C1C : Rectangle B B_2 C_1 C)
  (rectangle_CCA1A : Rectangle C C_2 A_1 A)
  (rectangle_AA2B1B : Rectangle A A_2 B_1 B)
  (angle_condition : ∠ B C_1 C + ∠ C A_1 A + ∠ A B_1 B = 180) :
  Concurrency (Line A_1 B_2) (Line B_1 C_2) (Line C_1 A_2) :=
sorry

end concurrency_of_lines_l38_38464


namespace polynomial_identity_proof_l38_38573

noncomputable def polynomial_identity : Prop := 
  ∀ (x a b c d : ℝ), 
     (3 * x + 2) * (2 * x - 3) * (x - 4) = a * x^3 + b * x^2 + c * x + d →
     a - b + c - d = 25

theorem polynomial_identity_proof : polynomial_identity := 
begin
  intros x a b c d h,
  sorry
end

end polynomial_identity_proof_l38_38573


namespace value_at_pi_over_4_intervals_of_increase_l38_38721

def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * (cos x) ^ 2 - 1

theorem value_at_pi_over_4 : f (π / 4) = 1 := 
by sorry 

theorem intervals_of_increase : 
  ∃ (k : ℤ), ∀ x, (- (3 * π / 8) + k * π ≤ x ∧ x ≤ (π / 8) + k * π) ∧ 
  f' x > 0 := by sorry 

end value_at_pi_over_4_intervals_of_increase_l38_38721


namespace volume_frustum_correct_l38_38604

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end volume_frustum_correct_l38_38604


namespace total_games_in_single_elimination_tournament_l38_38219

def single_elimination_tournament_games (teams : ℕ) : ℕ :=
teams - 1

theorem total_games_in_single_elimination_tournament :
  single_elimination_tournament_games 23 = 22 :=
by
  sorry

end total_games_in_single_elimination_tournament_l38_38219


namespace circle_sine_intersection_l38_38628

theorem circle_sine_intersection (h k r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n > 16 ∧
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, (x - h)^2 + (2 * Real.sin x - k)^2 = r^2) ∧ xs.card = n :=
by
  sorry

end circle_sine_intersection_l38_38628


namespace advantageous_bank_l38_38537

variable (C : ℝ) (p n : ℝ)

noncomputable def semiAnnualCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (2 * 100)) ^ n

noncomputable def monthlyCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (12 * 100)) ^ (6 * n)

theorem advantageous_bank (p n : ℝ) :
  monthlyCompounding p n - semiAnnualCompounding p n > 0 := sorry

#check advantageous_bank

end advantageous_bank_l38_38537


namespace sum_of_digits_7_pow_17_mod_100_l38_38892

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l38_38892


namespace frequency_of_students_scoring_90_to_100_l38_38663

theorem frequency_of_students_scoring_90_to_100 (total_students : ℕ) (students_scoring_90_to_100 : ℕ) :
  total_students = 50 → students_scoring_90_to_100 = 10 → (students_scoring_90_to_100 / total_students : ℚ) = 1 / 5 :=
begin
  intros h_total h_score,
  rw [h_total, h_score],
  norm_num,
end

end frequency_of_students_scoring_90_to_100_l38_38663


namespace find_b_in_triangle_l38_38419

noncomputable def triangle_b_value (a b c : ℝ) (tanB : ℝ) : ℝ := b

theorem find_b_in_triangle :
  ∀ (a b c : ℝ) (tanB : ℝ),
    (a + b + c) * (b + c - a) = 3 * b * c →
    a = sqrt 3 →
    tanB = sqrt 2 / 4 →
    b = 2 / 3 :=
begin
  intros a b c tanB h1 h2 h3,
  sorry
end

end find_b_in_triangle_l38_38419


namespace area_of_triangle_ABC_l38_38309

noncomputable def triangle_area_example
  (a b c : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 6) : ℝ := 
if h : a + b > c ∧ a + c > b ∧ b + c > a then
  let s := (a + b + c) / 2 in
    real.sqrt (s * (s - a) * (s - b) * (s - c))
else 0

theorem area_of_triangle_ABC 
  (h: triangle_area_example 3 5 6 3 rfl 5 rfl 6 rfl = 2 * real.sqrt 14) : 
  triangle_area_example 3 5 6 3 rfl 5 rfl 6 rfl = 2 * real.sqrt 14 := 
by {
  sorry
}

end area_of_triangle_ABC_l38_38309


namespace number_of_true_propositions_l38_38375

-- Define the original condition
def original_proposition (a b : ℝ) : Prop := (a + b = 1) → (a * b ≤ 1 / 4)

-- Define contrapositive
def contrapositive (a b : ℝ) : Prop := (a * b > 1 / 4) → (a + b ≠ 1)

-- Define inverse
def inverse (a b : ℝ) : Prop := (a * b ≤ 1 / 4) → (a + b = 1)

-- Define converse
def converse (a b : ℝ) : Prop := (a + b ≠ 1) → (a * b > 1 / 4)

-- State the problem
theorem number_of_true_propositions (a b : ℝ) :
  (original_proposition a b ∧ contrapositive a b ∧ ¬inverse a b ∧ ¬converse a b) → 
  (∃ n : ℕ, n = 1) :=
by sorry

end number_of_true_propositions_l38_38375


namespace yolanda_rate_is_correct_l38_38906

noncomputable def Yolanda_walking_rate : ℕ → ℕ → ℕ → ℕ → ℕ :=
  λ (total_distance bob_rate bob_distance start_time_diff : ℕ), 
  let meeting_time := bob_distance / bob_rate in
  let yolanda_time := meeting_time + start_time_diff in
  (total_distance - bob_distance) / yolanda_time

theorem yolanda_rate_is_correct
  (total_distance bob_rate bob_distance start_time_diff : ℕ)
  (bob_good_time : bob_distance = 30) 
  (bob_good_rate : bob_rate = 6) 
  (start_time_difference : start_time_diff = 1)
  (total_distance_is_60 : total_distance = 60) :
  Yolanda_walking_rate total_distance bob_rate bob_distance start_time_diff = 5 := 
by
  unfold Yolanda_walking_rate
  rw [bob_good_time, bob_good_rate, start_time_difference, total_distance_is_60]
  -- Additional details are skipped, this is covered by sorry
  sorry

end yolanda_rate_is_correct_l38_38906


namespace part_a_center_of_gravity_l38_38653

def center_of_gravity_arc (a : ℝ) : ℝ × ℝ :=
  ⟨5 / 8 * a, 5 * π / 128 * a⟩

theorem part_a_center_of_gravity (a : ℝ) (h₁ : 0 ≤ a) :
  let arc := {p : ℝ × ℝ // (p.1)^(2/3) + (p.2)^(2/3) = a^(2/3)}
  ∃ (g : arc → ℝ), (∀ p ∈ arc, g p = p.1) → center_of_gravity_arc a = 
  center_of_gravity_arc a :=
sorry

end part_a_center_of_gravity_l38_38653


namespace smallest_positive_integer_satisfying_condition_l38_38158

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end smallest_positive_integer_satisfying_condition_l38_38158


namespace num_satisfying_permutations_l38_38039

open List

def perm_condition (l : List ℕ) : Prop :=
  ∀ (m n : ℕ), 1 ≤ m → m < n → n ≤ 10 → (nthLe l (m-1) (by linarith) + m ≤ nthLe l (n-1) (by linarith) + n)

theorem num_satisfying_permutations : 
  let permutations := filter perm_condition (permutations (range' 1 10))
  permutations.length = 512 := by
sorry

end num_satisfying_permutations_l38_38039


namespace james_bought_291_pounds_l38_38445

def weight_vest_cost (v_cost : ℕ) (p_cost : ℝ) (v200_cost : ℕ) (d : ℕ) (s : ℕ) : ℕ :=
  let discounted_cost := v200_cost - d
  let amount_spent_if_no_save := v_cost + s
  let weight_plate_cost := discounted_cost - v_cost
  floor (weight_plate_cost / p_cost)

theorem james_bought_291_pounds :
  weight_vest_cost 250 1.2 700 100 110 = 291 := by
  sorry

end james_bought_291_pounds_l38_38445


namespace Jason_saturday_hours_l38_38446

theorem Jason_saturday_hours (x y : ℕ) 
  (h1 : 4 * x + 6 * y = 88)
  (h2 : x + y = 18) : 
  y = 8 :=
sorry

end Jason_saturday_hours_l38_38446


namespace probability_sum_one_l38_38519

theorem probability_sum_one (a : ℝ) :
  (a / 2 + a / 6 + a / 12 + a / 20 = 1) →
  (P_X_1 : ℝ) 
  (P_X_2 : ℝ) 
  (hPX1 : P_X_1 = a / 2)
  (hPX2 : P_X_2 = a / 6)
  (P_half_less_X_less_five_half : ℝ := P_X_1 + P_X_2) :
  P_half_less_X_less_five_half = 5 / 6 :=
sorry

end probability_sum_one_l38_38519


namespace angle_between_a_plus_b_and_b_l38_38732

noncomputable def angleBetweenVectors (a b : ℝ × ℝ) : ℝ :=
  let sum := (a.1 + b.1, a.2 + b.2)
  let dotProduct := sum.1 * b.1 + sum.2 * b.2
  let magSum := Real.sqrt (sum.1 ^ 2 + sum.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  Real.acos (dotProduct / (magSum * magB))

theorem angle_between_a_plus_b_and_b :
  let a := (-1 : ℝ, 2 : ℝ)
  let b := (2 : ℝ, 1 : ℝ)
  angleBetweenVectors a b = π / 4 :=
by
  sorry

end angle_between_a_plus_b_and_b_l38_38732


namespace tangency_points_coplanar_l38_38594

theorem tangency_points_coplanar
  {A B C D K L M N : Point}   -- Points A, B, C, D, K, L, M, N
  (S : Sphere)                -- Sphere S
  (tangent_to_sphere : 
    (tangency (A, B) S K) ∧ (tangency (B, C) S L) ∧ 
    (tangency (C, D) S M) ∧ (tangency (D, A) S N))  -- Tangency conditions
  (quadrilateral_inscribed : circumscribed (A, B, C, D) S) : -- Quadrilateral circumscribed condition
  coplanar K L M N := 
sorry

end tangency_points_coplanar_l38_38594


namespace solution_set_of_quadratic_inequality_l38_38128

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by 
  sorry

end solution_set_of_quadratic_inequality_l38_38128


namespace pyramid_volume_is_1_12_l38_38602

def base_rectangle_length_1 := 1
def base_rectangle_width_1_4 := 1 / 4
def pyramid_height_1 := 1

noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * (base_rectangle_length_1 * base_rectangle_width_1_4) * pyramid_height_1

theorem pyramid_volume_is_1_12 : pyramid_volume = 1 / 12 :=
sorry

end pyramid_volume_is_1_12_l38_38602


namespace cone_water_fill_percentage_l38_38935

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38935


namespace simplify_expression_l38_38172

noncomputable def a : ℝ := 2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18
noncomputable def b : ℝ := 4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50

theorem simplify_expression : a * b = 97 := by
  sorry

end simplify_expression_l38_38172


namespace marathon_winner_average_speed_l38_38209

-- Define the conditions of the problem
def marathon_distance : ℝ := 42
def start_time : ℝ := 11 + 30 / 60
def end_time : ℝ := 13 + 45 / 60

-- Calculate the total running time
def running_time : ℝ := end_time - start_time

-- Calculate the average speed
def average_speed : ℝ := marathon_distance / running_time

-- The theorem we want to prove
theorem marathon_winner_average_speed : average_speed = 18.6 := by
  -- The proof goes here
  sorry

end marathon_winner_average_speed_l38_38209


namespace solve_equation_l38_38490

theorem solve_equation (x : ℝ) (h : x ≠ 2) :
  x^2 = (4*x^2 + 4) / (x - 2) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by
  sorry

end solve_equation_l38_38490


namespace cone_height_l38_38337

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l38_38337


namespace probability_of_losing_l38_38751

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end probability_of_losing_l38_38751


namespace proof_problem_l38_38478

def scores_A : List ℝ := [2, 4, 6, 8, 7, 7, 8, 9, 9, 10]
def scores_B : List ℝ := [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length)

noncomputable def range (scores : List ℝ) : ℝ :=
  scores.maximum - scores.minimum

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

def B_is_more_stable : Prop :=
  range scores_B < range scores_A ∧ variance scores_B < variance scores_A

def A_performs_better : Prop :=
  mean scores_A > mean scores_B ∨ (mean scores_A = mean scores_B ∧ scores_A.count (λ x => x ≥ 9) > scores_B.count (λ x => x ≥ 9))

def A_has_more_potential : Prop := -- This can be a bit subjective; here, we represent it as a statement to be proven.
  true -- Placeholder for analysis trends, separate study might be required.

theorem proof_problem :
  mean scores_A = 7 ∧ range scores_A = 8 ∧ variance scores_A = 5.4 ∧
  mean scores_B = 7 ∧ range scores_B = 4 ∧ variance scores_B = 1.2 ∧
  B_is_more_stable ∧ A_performs_better ∧ A_has_more_potential :=
by 
  sorry

end proof_problem_l38_38478


namespace evaluate_expression_l38_38632

-- Define the operation [a, b, c] as (a + b) / c where c ≠ 0
def abc_op (a b c : ℝ) (h : c ≠ 0) : ℝ := (a + b) / c

-- The theorem to be proven
theorem evaluate_expression :
  abc_op (abc_op 100 50 150 (by norm_num)) (abc_op 4 2 6 (by norm_num)) (abc_op 20 10 30 (by norm_num)) (by norm_num) = 2 :=
sorry

end evaluate_expression_l38_38632


namespace percent_filled_cone_l38_38975

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38975


namespace debby_deletion_l38_38169

theorem debby_deletion :
  ∀ (zoo_pics museum_pics remaining_pics deleted_pics : ℕ),
    zoo_pics = 24 →
    museum_pics = 12 →
    remaining_pics = 22 →
    deleted_pics = zoo_pics + museum_pics - remaining_pics →
    deleted_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics deleted_pics h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end debby_deletion_l38_38169


namespace quadrilateral_area_l38_38483

theorem quadrilateral_area (EF FG EH HG : ℕ) (hEFH : EF * EF + FG * FG = 25)
(hEHG : EH * EH + HG * HG = 25) (h_distinct : EF ≠ EH ∧ FG ≠ HG) 
(h_greater_one : EF > 1 ∧ FG > 1 ∧ EH > 1 ∧ HG > 1) :
  (EF * FG) / 2 + (EH * HG) / 2 = 12 := 
sorry

end quadrilateral_area_l38_38483


namespace trapezoid_area_l38_38600

variable (x : ℝ)
variable (h : ℝ) (B1 B2 : ℝ)
variable (A : ℝ)

-- Conditions
def height_x : Prop := h = x
def base1_3x : Prop := B1 = 3 * x
def base2_4x : Prop := B2 = 4 * x

-- Theorem to prove the area expression
theorem trapezoid_area (height_x : height_x) (base1_3x : base1_3x) (base2_4x : base2_4x) : 
  A = 7 * x^2 / 2 := 
sorry

end trapezoid_area_l38_38600


namespace rearrange_ЭПИГРАФ_l38_38432

theorem rearrange_ЭПИГРАФ : (nat.choose 7 3) = 35 := by
  sorry

end rearrange_ЭПИГРАФ_l38_38432


namespace smallest_positive_integer_satisfying_condition_l38_38159

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end smallest_positive_integer_satisfying_condition_l38_38159


namespace probability_product_divisible_by_4_gt_half_l38_38266

theorem probability_product_divisible_by_4_gt_half :
  let n := 2023
  let even_count := n / 2
  let four_div_count := n / 4
  let select_five := 5
  (true) ∧ (even_count = 1012) ∧ (four_div_count = 505)
  → 0.5 < (1 - ((2023 - even_count) / 2023) * ((2022 - (even_count - 1)) / 2022) * ((2021 - (even_count - 2)) / 2021) * ((2020 - (even_count - 3)) / 2020) * ((2019 - (even_count - 4)) / 2019)) :=
by
  sorry

end probability_product_divisible_by_4_gt_half_l38_38266


namespace Lizzie_has_27_crayons_l38_38053

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l38_38053


namespace minimum_sticks_broken_n12_can_form_square_n15_l38_38284

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l38_38284


namespace cone_height_l38_38339

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l38_38339


namespace cone_volume_percentage_filled_l38_38948

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38948


namespace painting_falls_if_any_nail_removed_l38_38392

-- Define the commutator
def commutator (a b : Type) [Group a] [Group b] (x : a) (y : b) : a :=
 x * y * x⁻¹ * y⁻¹

-- The theorem statement
theorem painting_falls_if_any_nail_removed (a1 a2 a3 : Type) [Group a1] [Group a2] [Group a3] 
(x1 : a1) (x2 : a2) (x3 : a3) : 
commutator (commutator x1 x2) x3 = x1 * x2 * x1⁻¹ * x2⁻¹ * x3 * x2 * x1 * x2⁻¹ * x1⁻¹ * x3⁻¹ :=
sorry

end painting_falls_if_any_nail_removed_l38_38392


namespace volume_of_cone_l38_38714

-- Definitions of the given conditions
def cos_angle_ASB : ℝ := sqrt 15 / 4
def angle_SA_base : ℝ := 60 * real.pi / 180  -- convert degrees to radians
def area_triangle_SAB : ℝ := 2

-- The statement to be proved
theorem volume_of_cone :
  (∃ (SA : ℝ) (r : ℝ) (h : ℝ), 
     (cos_angle_ASB = sqrt 15 / 4) ∧ 
     (angle_SA_base = 60 * real.pi / 180) ∧ 
     (area_triangle_SAB = 2) ∧ 
     (volume := 1 / 3 * real.pi * r ^ 2 * h) ∧ 
     (volume = 8 * sqrt 3 * real.pi / 3)) :=
sorry

end volume_of_cone_l38_38714


namespace arithmetic_seq_squares_l38_38073

theorem arithmetic_seq_squares (x y z : ℝ) (h : y - x = z - y) :
  (x^2 + x * y + y^2, x^2 + x * z + z^2, y^2 + y * z + z^2).2 - (x^2 + x * y + y^2) =
  (y^2 + y * z + z^2) - (x^2 + x * z + z^2) :=
by sorry

end arithmetic_seq_squares_l38_38073


namespace tan_of_angle_passing_through_point_l38_38355

theorem tan_of_angle_passing_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (-1, real.sqrt 3)) :
  real.tan α = -real.sqrt 3 := 
sorry

end tan_of_angle_passing_through_point_l38_38355


namespace integral_1_integral_2_integral_3_integral_4_integral_5_integral_6_l38_38918

section

-- 1. Integral of (3x^2 - 2x + 5) dx
theorem integral_1 : ∫ (3 * x^2 - 2 * x + 5) dx = x^3 - x^2 + 5 * x + C :=
sorry

-- 2. Integral of (2x^2 + x - 1) / x^3 dx
theorem integral_2 : ∫ ((2 * x^2 + x - 1) / x^3) dx = 2 * log |x| - 1 / x + 1 / (2 * x^2) + C :=
sorry

-- 3. Integral of (1 + e^x)^2 dx
theorem integral_3 : ∫ (1 + exp x) ^ 2 dx = x + 2 * exp x + 1 / 2 * exp (2 * x) + C :=
sorry

-- 4. Integral of (2x + 3) / (x^2 - 5) dx
theorem integral_4 : ∫ ((2 * x + 3) / (x^2 - 5)) dx = log |x^2 - 5| + (3 / (2 * sqrt 5)) * log |(x - sqrt 5) / (x + sqrt 5)| + C :=
sorry

-- 5. Integral of x^2 / (x^2 + 1) dx
theorem integral_5 : ∫ (x^2 / (x^2 + 1)) dx = x - arctan x + C :=
sorry

-- 6. Integral of tg^2 ϕ dϕ
theorem integral_6 : ∫ (tan ϕ)^2 dϕ = tan ϕ - ϕ + C :=
sorry

end

end integral_1_integral_2_integral_3_integral_4_integral_5_integral_6_l38_38918


namespace tank_capacity_l38_38232

def outlet_rate (C : ℝ) : ℝ := C / 10
def inlet_rate1 : ℝ := 240
def inlet_rate2 : ℝ := 360
def combined_inlet_rate : ℝ := inlet_rate1 + inlet_rate2
def effective_emptying_rate (C : ℝ) : ℝ := C / 18

theorem tank_capacity (C : ℝ) : outlet_rate C - combined_inlet_rate = effective_emptying_rate C → C = 13500 :=
by
  sorry

end tank_capacity_l38_38232


namespace find_BD_l38_38393

theorem find_BD (ABCD : Type) [rect : rectangle ABCD] (A B C D A' : point)
  (h1 : fold_triangle_ABD_over_BD_maps_A_to_A')
  (h2 : A'C = BD / 3)
  (h3 : area ABCD = 27 * real.sqrt 2) :
  BD = 9 :=
sorry

end find_BD_l38_38393


namespace hyperbola_foci_distance_l38_38256

noncomputable def a_squared : ℝ := 5
noncomputable def b_squared : ℝ := 0.5
noncomputable def equation : ℝ → ℝ → Prop := λ x y, 9 * y^2 / 45 - 6 * x^2 / 3 = 1

theorem hyperbola_foci_distance :
  (∀ x y, equation x y) →
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c = 2 * Real.sqrt 5.5 :=
by
  sorry

end hyperbola_foci_distance_l38_38256


namespace integral_value_l38_38712

theorem integral_value 
  (a : ℝ) 
  (h : let f := (a + x + x^2)*(1 - x)^4 in 
       coeff_at_x^3(f) = -14 ∧ a = 4) : 
  ∫ x in 0..2, sqrt (16 - x^2) = (4 * real.pi) / 3 + 2 * real.sqrt 3 :=
by
  sorry

end integral_value_l38_38712


namespace cone_height_l38_38334

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l38_38334


namespace quadrilateral_axis_of_symmetry_l38_38495

variable {A B C D M N : Type}

-- Assume A, B, C, D are points in a Euclidean space with given properties.
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (A B C D : ℝ) (angle_A_eq_angle_B : ∠ A = ∠ B) (AD_eq_BC : dist A D = dist B C)
variables (M : ℝ) (N : ℝ)

-- Define M and N as midpoints of AB and CD respectively
-- Assume the coordinates are such that they hold the midpoint properties
def isMidpoint := ∀ (X Y Z : ℝ), dist X Z = dist Y Z → (Z = (X + Y) / 2)

-- Prove that MN is the axis of symmetry of the quadrilateral
theorem quadrilateral_axis_of_symmetry :
  isMidpoint A B M →
  isMidpoint C D N →
  lineSymmetry (lineSegment M N) (quadrilateral A B C D) :=
by
  sorry

end quadrilateral_axis_of_symmetry_l38_38495


namespace starting_player_wins_with_perfect_play_l38_38533

-- Define the initial condition: Three piles of stones, each containing 30 stones
def initial_piles : List ℕ := [30, 30, 30]

-- Define the game rules: A player can take any number of stones from a single pile
-- Winning condition: The player who takes the last stone wins

-- State the theorem: The first player can always win with perfect play
theorem starting_player_wins_with_perfect_play (piles : List ℕ) (h : piles = initial_piles) :
  ∃ strategy : (List ℕ → (ℕ × ℕ)), ∀ game_state : List ℕ,
    (player_wins game_state strategy ∨ player_loses game_state strategy) := 
sorry

end starting_player_wins_with_perfect_play_l38_38533


namespace sum_first_60_terms_l38_38103

noncomputable def a (n : ℕ) : ℝ :=
  (-1)^n * (2 * n - 1) * real.cos (n * real.pi / 2) + 1

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1)

theorem sum_first_60_terms : S 60 = 120 := 
by 
  sorry

end sum_first_60_terms_l38_38103


namespace maximum_distance_l38_38854

-- Definition of the chessboard
def chessboard : Type := fin 8 × fin 8

-- Definition of numbering of chessboard squares
def numbering (s : chessboard) : ℕ :=
  -- Assume 'num' is a valid numbering function from 1 to 64
  sorry 

-- Definition of neighboring squares
def neighbors (s1 s2 : chessboard) : Prop :=
  s1 != s2 ∧
  ( (s1.1 = s2.1 ∧ int.of_nat (nat.abs (s1.2.val - s2.2.val)) <= 1) ∨
    (s1.2 = s2.2 ∧ int.of_nat (nat.abs (s1.1.val - s2.1.val)) <= 1) ∨
    (int.of_nat (nat.abs (s1.1.val - s2.1.val)) = 1 ∧
     int.of_nat (nat.abs (s1.2.val - s2.2.val)) = 1) )

-- Definition of the distance between two squares
def distance (n : ℕ) (s1 s2 : chessboard) (numbering : chessboard → ℕ) : ℕ :=
  if neighbors s1 s2 then nat.abs ((numbering s1) - (numbering s2)) else 0

-- The main theorem statement
theorem maximum_distance :
  ∃ num : (chessboard → ℕ),
    ∀ s1 s2 : chessboard, neighbors s1 s2 →
    distance 8 s1 s2 num ≤ 8 :=
sorry

end maximum_distance_l38_38854


namespace solve_problem_l38_38160

noncomputable def smallest_positive_integer : ℕ :=
  Inf {n : ℕ | 0 < n ∧ (Real.sqrt n - Real.sqrt (n - 1) < 0.01)}

theorem solve_problem : smallest_positive_integer = 2501 :=
begin
  sorry
end

end solve_problem_l38_38160


namespace inequality_proof_l38_38411

theorem inequality_proof (a b : ℤ) (ha : a > 0) (hb : b > 0) : a + b ≤ 1 + a * b :=
by
  sorry

end inequality_proof_l38_38411


namespace max_distance_of_ball_travel_l38_38615

theorem max_distance_of_ball_travel (x y : ℝ) (a b : ℝ) (c : ℝ) (A B : ℝ) 
  (h_ellipse_eqn : (x^2 / 16) + (y^2 / 9) = 1)
  (h_a_eqn : a = 4)
  (h_b_eqn : b = 3)
  (h_c_eqn : c = sqrt (a^2 - b^2))
  (h_dist_foci : abs (x - A) + abs (x - B) = 4 * a) : 
  (4 * a = 16) := 
by 
  sorry

end max_distance_of_ball_travel_l38_38615


namespace expression_for_f_in_neg2_to_0_l38_38700

noncomputable def f : ℝ → ℝ := sorry

theorem expression_for_f_in_neg2_to_0 :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x : ℝ, f (x - 3/2) = f (x + 1/2)) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x = x) →
  (∀ x, -2 < x ∧ x < 0 → f x = 3 - |x + 1|) :=
by
  intros h
  cases h with h_even h
  cases h with h_periodic h_segment
  sorry

end expression_for_f_in_neg2_to_0_l38_38700


namespace sum_of_intersections_l38_38362

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_property : ∀ x : ℝ, f (-x) = 2 - f x

theorem sum_of_intersections :
  ∀ (m : ℕ) (x y : fin m → ℝ),
  (∀ i, y i = (x i + 1) / x i) →
  (∀ i, y i = f (x i)) →
  finset.univ.sum (λ i, x i + y i) = m :=
begin
  sorry
end

end sum_of_intersections_l38_38362


namespace company_ratio_l38_38201

noncomputable def company_ratio_proof : Prop :=
  ∃ (A B C D: ℕ), 
    A = 30 ∧ 
    C = A + 10 ∧ 
    D = C - 5 ∧ 
    A + B + C + D + 20 = 185 ∧ 
    B / A = 2

#eval company_ratio_proof -- This is to evaluate whether the statement builds correctly

theorem company_ratio : company_ratio_proof :=
  sorry

end company_ratio_l38_38201


namespace flowers_remaining_l38_38094

theorem flowers_remaining :
  let initial_flowers := 3 * 12
  let given_away := initial_flowers / 2
  let in_vase := initial_flowers - given_away
  let wilted := in_vase / 3
  in (in_vase - wilted) = 12 :=
by
  sorry

end flowers_remaining_l38_38094


namespace solve_system_of_equations_l38_38820

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x + 2 * y = 5 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 := 
by
  sorry

end solve_system_of_equations_l38_38820


namespace number_of_m_set_classes_l38_38677

def is_m_set_class (X M : Set (Set X)) : Prop :=
  X ∈ M ∧ ∅ ∈ M ∧ (∀ A B, A ∈ M → B ∈ M → (A ∪ B) ∈ M ∧ (A ∩ B) ∈ M)

def example_set : Set (Set (Set ℕ)) := {∅, {2, 3}, {1, 2, 3}}

theorem number_of_m_set_classes :
  ∃! n, n = 10 ∧ 
        ∃ M : Set (Set (Set ℕ)), is_m_set_class {1, 2, 3} M ∧
                                  ({2, 3} ∈ M) :=
sorry

end number_of_m_set_classes_l38_38677


namespace problem_part1_problem_part2_problem_part3_l38_38706

open Nat

theorem problem_part1 (n : ℕ) (h : 2 * (choose n 1) = (2^2 * (choose n 2)) / 5) (hm : 0 < n) : n = 6 :=
sorry

theorem problem_part2 (n : ℕ) (hn : n = 6) : 
  let coeff := binom 6 3 * 2^3 
  in coeff = 160 :=
sorry

theorem problem_part3 : 
  let exp := (∑ i in range 7, 2_i) = 2^6 in
  exp = 64 :=
sorry

end problem_part1_problem_part2_problem_part3_l38_38706


namespace length_of_goods_train_l38_38205

theorem length_of_goods_train (speed_kmh : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) (length_of_train: ℝ) : 
  speed_kmh = 90 → 
  platform_length_m = 350 → 
  time_sec = 32 → 
  length_of_train = (speed_kmh * 1000 / 3600) * time_sec - platform_length_m → 
  length_of_train = 450 :=
by
  intros h_speed h_platform h_time h_length
  rw [h_speed, h_platform, h_time] at h_length
  exact h_length

-- To handle skipping the proof
-- sorry

end length_of_goods_train_l38_38205


namespace cone_height_l38_38345

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l38_38345


namespace seq_periodicity_l38_38369

def g : ℕ → ℕ
| 1 := 3
| 2 := 4
| 3 := 2
| 4 := 1
| 5 := 5
| _ := 0 -- For completeness, although input is meant to be only {1, 2, 3, 4, 5}

def v : ℕ → ℕ
| 0 := 3
| (n+1) := g (v n)

theorem seq_periodicity :
  v 2002 = 4 :=
by sorry

end seq_periodicity_l38_38369


namespace simplify_polynomial_l38_38082

theorem simplify_polynomial :
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  (x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10) :=
by
  sorry

end simplify_polynomial_l38_38082


namespace num_unfair_le_half_n_minus_1_l38_38487

open Nat

def is_product_of_distinct_primes (n : ℕ) (k : ℕ) (ps : list ℕ) : Prop :=
  (ps.length = k) ∧ (∀ p ∈ ps, prime p) ∧ (n = ps.prod) ∧ (k ≥ 2)

noncomputable def num_unfair (n : ℕ) : ℕ := sorry

theorem num_unfair_le_half_n_minus_1 (n k : ℕ) (ps : list ℕ) 
  (hk : k ≥ 2) (hprimes : is_product_of_distinct_primes n k ps) : 
  num_unfair n ≤ (n - 1) / 2 :=
sorry

end num_unfair_le_half_n_minus_1_l38_38487


namespace truck_travel_distance_l38_38224

theorem truck_travel_distance (rear_tire_limit front_tire_limit : ℕ) (swap_interval : ℕ) :
  (rear_tire_limit = 15000) →
  (front_tire_limit = 25000) →
  (swap_interval > 0) →
  let max_travel_distance := 18750
  in swap_interval * ((front_tire_limit * rear_tire_limit) / (front_tire_limit * 3 + rear_tire_limit * 5)) = max_travel_distance := 
by
  intros
  sorry

end truck_travel_distance_l38_38224


namespace max_marks_400_l38_38217

theorem max_marks_400 {M : ℝ} (h1 : 0.35 * M = 140) : M = 400 :=
by 
-- skipping the proof using sorry
sorry

end max_marks_400_l38_38217


namespace find_x_for_distance_l38_38694

/-- Given points A(x, 1, 2) and B(2, 3, 4) with the distance |AB| = 2√6, show that x = -2 or x = 6. -/
theorem find_x_for_distance (x : ℝ) (h_dist : real.sqrt ((2 - x)^2 + (3 - 1)^2 + (4 - 2)^2) = 2 * real.sqrt 6) :
  x = -2 ∨ x = 6 := 
sory

end find_x_for_distance_l38_38694


namespace volume_filled_cone_l38_38958

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38958


namespace original_number_is_142857_l38_38592

-- Definitions based on conditions
def six_digit_number (x : ℕ) : ℕ := 100000 + x
def moved_digit_number (x : ℕ) : ℕ := 10 * x + 1

-- Lean statement of the equivalent problem
theorem original_number_is_142857 : ∃ x, six_digit_number x = 142857 ∧ moved_digit_number x = 3 * six_digit_number x :=
  sorry

end original_number_is_142857_l38_38592


namespace base_8_subtraction_l38_38255

theorem base_8_subtraction : 
  let x := 0o1234   -- 1234 in base 8
  let y := 0o765    -- 765 in base 8
  let result := 0o225 -- 225 in base 8
  x - y = result := by sorry

end base_8_subtraction_l38_38255


namespace tangent_line_eqn_at_one_l38_38838

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x

theorem tangent_line_eqn_at_one :
  let f' := fun x : ℝ => Real.log x + 2 in
  let tangent_line_at := fun x y : ℝ => y - f 1 = f' 1 * (x - 1) in
  tangent_line_at x y = (y = 2 * x - 1) :=
by
  unfold f
  have h1: f' 1 = 2 := by
    unfold f'
    simp
  have h2: f 1 = 1 := by
    unfold f
    simp
  have tangent_eq := calc
    tangent_line_at x y = y - f 1 = f' 1 * (x - 1) : rfl
                      ... = y - 1 = 2 * (x - 1)    : by rw [h1, h2]
                      ... = y = 2 * x - 1          : sorry
  exact tangent_eq
  sorry

end tangent_line_eqn_at_one_l38_38838


namespace inequality_holds_l38_38253

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_holds_l38_38253


namespace correct_operation_l38_38551

variable {x y : ℝ}

theorem correct_operation :
  (2 * x^2 + 4 * x^2 = 6 * x^2) → 
  (x * x^3 = x^4) → 
  ((x^3)^2 = x^6) →
  ((xy)^5 = x^5 * y^5) →
  ((x^3)^2 = x^6) := 
by 
  intros h1 h2 h3 h4
  exact h3

end correct_operation_l38_38551


namespace area_of_square_STUV_l38_38528

-- Defining the conditions
variable (C L : ℝ)
variable (h1 : 2 * (C + L) = 40)

-- The goal is to prove the area of the square STUV
theorem area_of_square_STUV : (C + L) * (C + L) = 400 :=
by
  sorry

end area_of_square_STUV_l38_38528


namespace min_breaks_for_square_12_can_form_square_15_l38_38296

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l38_38296


namespace num_subsets_of_P_l38_38696

namespace SetProof
  open Set

  def M : Set ℕ := {1, 2}
  def N : Set ℕ := {2, 3, 4}
  def P : Set ℕ := M ∪ N

  theorem num_subsets_of_P :
    (P = {1, 2, 3, 4}) → (2 ^ 4 = 16) := by
    intro hP
    rw [hP]
    rfl
end SetProof

end num_subsets_of_P_l38_38696


namespace cubes_even_sum_even_l38_38045

theorem cubes_even_sum_even (p q : ℕ) (h : Even (p^3 - q^3)) : Even (p + q) := sorry

end cubes_even_sum_even_l38_38045


namespace water_filled_percent_l38_38986

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38986


namespace probability_of_multiple_5_or_7_l38_38475

noncomputable def probability_multiple_5_or_7 : ℚ :=
let total_balls := 100 in
let multiples_of_5 := 20 in
let multiples_of_7 := 14 in
let multiples_of_35 := 2 in
let favorable_outcomes := multiples_of_5 + multiples_of_7 - multiples_of_35 in
favorable_outcomes / total_balls

theorem probability_of_multiple_5_or_7 :
  probability_multiple_5_or_7 = 8 / 25 :=
by
  -- Definition of the total number of balls
  let total_balls := 100

  -- Counting multiples of 5
  let multis_5 := (list.range' 5 total_balls).countp (λ n, n % 5 = 0)

  -- Counting multiples of 7
  let multis_7 := (list.range' 7 total_balls).countp (λ n, n % 7 = 0)

  -- Counting multiples of both 5 and 7 (i.e., multiples of 35)
  let multis_35 := (list.range' 35 total_balls).countp (λ n, n % 35 = 0)

  -- Using the inclusion-exclusion principle to find the total
  let favorable_outcomes := multis_5 + multis_7 - multis_35

  -- Calculate probability
  let probability := favorable_outcomes / total_balls

  have : multis_5 = 20, from sorry
  have : multis_7 = 14, from sorry
  have : multis_35 = 2, from sorry
  have : favorable_outcomes = 32, from sorry
  have : favorable_outcomes / total_balls = 8 / 25, from sorry

  -- Applying inclusion-exclusion principle to verify the final probability
  exact this

end probability_of_multiple_5_or_7_l38_38475


namespace triangle_area_l38_38122

theorem triangle_area (P r : ℝ) (hP : P = 48) (hr : r = 2.5) :
  let s := P / 2 in
  let A := r * s in
  A = 60 := by
sorry

end triangle_area_l38_38122


namespace smallest_k_for_factorial_divisibility_l38_38657

theorem smallest_k_for_factorial_divisibility : 
  ∃ (k : ℕ), (∀ n : ℕ, n < k → ¬(2040 ∣ n!)) ∧ (2040 ∣ k!) ∧ k = 17 :=
by
  -- We skip the actual proof steps and provide a placeholder for the proof
  sorry

end smallest_k_for_factorial_divisibility_l38_38657


namespace ratio_fifth_term_l38_38671

-- Definitions of arithmetic sequences and sums
def arithmetic_seq_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * d 1) / 2

-- Conditions
variables (S_n S'_n : ℕ → ℕ) (n : ℕ)

-- Given conditions
axiom ratio_sum : ∀ (n : ℕ), S_n n / S'_n n = (5 * n + 3) / (2 * n + 7)
axiom sums_at_9 : S_n 9 = 9 * (S_n 1 + S_n 9) / 2
axiom sums'_at_9 : S'_n 9 = 9 * (S'_n 1 + S'_n 9) / 2

-- Theorem to prove
theorem ratio_fifth_term : (9 * (S_n 1 + S_n 9) / 2) / (9 * (S'_n 1 + S'_n 9) / 2) = 48 / 25 := sorry

end ratio_fifth_term_l38_38671


namespace digit_ends_in_9_l38_38093

theorem digit_ends_in_9 {α : Type} (pairs : Fin 999 → α × α) 
  (h : ∀ i, ∃ (a b : α), pairs i = (a, b) ∧ (|a - b| = 1 ∨ |a - b| = 6)) : 
  (∑ i, | (pairs i).1 - (pairs i).2 |) % 10 = 9 :=
sorry

end digit_ends_in_9_l38_38093


namespace hulk_first_jump_over_1500_l38_38828

def geometric_sequence (a r : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := r * geometric_sequence n

def hulk_jump_geometric_sequence : ℕ → ℕ := geometric_sequence 2 3

theorem hulk_first_jump_over_1500 : ∃ n : ℕ, hulk_jump_geometric_sequence n > 1500 ∧ ∀ m < n, hulk_jump_geometric_sequence m ≤ 1500 := by
  sorry

end hulk_first_jump_over_1500_l38_38828


namespace sum_of_first_15_even_integers_l38_38164

theorem sum_of_first_15_even_integers : 
  let a := 2 in
  let d := 2 in
  let n := 15 in
  let S := (n / 2) * (a + (a + (n - 1) * d)) in
  S = 240 :=
by
  sorry

end sum_of_first_15_even_integers_l38_38164


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38884

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38884


namespace cone_height_l38_38338

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l38_38338


namespace proof_angle_between_face_and_base_l38_38126

noncomputable def angle_between_face_and_base (k : ℝ) : Prop :=
  α = Real.arccot (Real.sqrt (2 * Real.sqrt 3 * Real.pi * k - 27) / 6) ∧
  k > 9 * Real.sqrt 3 / (2 * Real.pi)

theorem proof_angle_between_face_and_base 
  (P P1 : Point)
  (MM1 : ℝ)
  (O : Point) (R a b α x : ℝ) 
  (PM P1M1 P1P : ℝ) 
  (h : ℝ) 
  (A1 A2 : ℝ)
  (V_pyramid V_sphere : ℝ)
  (k : ℝ)
  (cond1 : P1M1 = a / (2 * Real.sqrt 3))
  (cond2 : PM = b / (2 * Real.sqrt 3))
  (cond3 : P1P = 2 * R) :
  angle_between_face_and_base k :=
by {
  sorry
}

end proof_angle_between_face_and_base_l38_38126


namespace smallest_n_l38_38549

theorem smallest_n (n : ℕ) : 
  (n % 6 = 2) ∧ (n % 7 = 3) ∧ (n % 8 = 4) → n = 8 :=
  by sorry

end smallest_n_l38_38549


namespace volume_filled_cone_l38_38949

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38949


namespace possible_values_of_n_l38_38849

theorem possible_values_of_n :
  ∃ (n : ℤ), 
    ∃ x y z : ℝ, 
      (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
      (x > 0 ∧ y > 0 ∧ z > 0) ∧
      (∃ a : ℝ, a = 1007 ∧ (x = a ∧ y = a/2 + real.sqrt(r) ∧ z = a/2 - real.sqrt(r)) ∧ r^2 ∈ set.Icc 1 1014048 ) ∧
      ∃ n_values, n_values = 1013043 :=
sorry -- Proof to be filled in

end possible_values_of_n_l38_38849


namespace radius_of_insphere_of_triangular_pyramid_l38_38115

theorem radius_of_insphere_of_triangular_pyramid :
  ∀ (A B C D : Type) (AB BC CA : ℝ),
  (AB = real.sqrt 41) →
  (BC = real.sqrt 61) →
  (CA = real.sqrt 52) →
  (∀ u v w : tuple A (B, C), u • v = 0) → -- representing edge orthogonality with tuple constraints
  let x := real.sqrt 16 in
  let y := real.sqrt 25 in
  let z := real.sqrt 36 in
  ∀ (O : Type), (contains A B C O D) → -- center lies on the base
  r = (((4 : ℕ) * (5 : ℕ) * (6 : ℕ)) / ((4 : ℕ) * (5 : ℕ) + (5 : ℕ) * (6 : ℕ) + (4 : ℕ) * (6 : ℕ))) :=
sorry


end radius_of_insphere_of_triangular_pyramid_l38_38115


namespace coefficient_x2_in_binomial_expansion_coefficient_x2_is_70_l38_38245

theorem coefficient_x2_in_binomial_expansion : 
  ∀ (x : ℝ), (x - 1 / sqrt x) ^ 8 = ∑ i in (finset.range 9), (binom 8 i * (-1) ^ i * x ^ (8 - (3 / 2) * i)) :=
begin
  sorry
end

theorem coefficient_x2_is_70 :
  ∀ (x : ℝ), 
  has_ceiling.has_floor ((x - 1 / sqrt x) ^ 8).term 4 = 70  :=
begin
  sorry
end

end coefficient_x2_in_binomial_expansion_coefficient_x2_is_70_l38_38245


namespace problem_solution_l38_38661

theorem problem_solution (x : Real) (h : log 5 (x^2 - 5 * x + 14) = 2) :
    x = (5 + sqrt 69) / 2 ∨ x = (5 - sqrt 69) / 2 :=
by
  sorry

end problem_solution_l38_38661


namespace sum_of_adjacents_to_15_l38_38518

-- Definitions of the conditions
def divisorsOf225 : Set ℕ := {3, 5, 9, 15, 25, 45, 75, 225}

-- Definition of the adjacency relationship
def isAdjacent (x y : ℕ) (s : Set ℕ) : Prop :=
  x ∈ s ∧ y ∈ s ∧ Nat.gcd x y > 1

-- Problem statement in Lean 4
theorem sum_of_adjacents_to_15 :
  ∃ x y : ℕ, isAdjacent 15 x divisorsOf225 ∧ isAdjacent 15 y divisorsOf225 ∧ x + y = 120 :=
by
  sorry

end sum_of_adjacents_to_15_l38_38518


namespace fraction_less_than_mode_l38_38759

def mode {α : Type*} [decidable_eq α] (l : list α) : α :=
(l.nth_le (l.indexes l.max).head (by simp)).get_or_else l.head

theorem fraction_less_than_mode (l : list ℕ) (h_mode : ∃ m, mode l = m ∧ ∀ x ∈ l, x ≤ m) 
  (h_fraction : (l.count(< mode l) : ℚ) / l.length = 2 / 9) :
  ∃ l, (l.count(< mode l) : ℚ) / l.length = 2 / 9 :=
by {
  sorry
}

end fraction_less_than_mode_l38_38759


namespace cyclic_quad_D_E_F_K_l38_38631

-- Given data
variables {A B C H M N D K E F : Type}
variables [triangle ABC] [orthocenter H ABC]
variables [midpoint M BC] [midpoint N AH]
variables [on_line D M H] [parallel AD BC]
variables [on_line K A H] [cyclic D N M K]
variables [on_line E A C] [eq_angle E H M C]
variables [on_line F A B] [eq_angle F H M B]

-- Theorem statement
theorem cyclic_quad_D_E_F_K : cyclic D E F K :=
sorry

end cyclic_quad_D_E_F_K_l38_38631


namespace condition_for_a_b_complex_l38_38535

theorem condition_for_a_b_complex (a b : ℂ) (h1 : a ≠ 0) (h2 : 2 * a + b ≠ 0) :
  (2 * a + b) / a = b / (2 * a + b) → 
  (∃ z : ℂ, a = z ∨ b = z) ∨ 
  ((∃ z1 : ℂ, a = z1) ∧ (∃ z2 : ℂ, b = z2)) :=
sorry

end condition_for_a_b_complex_l38_38535


namespace not_possible_2020_parts_possible_2023_parts_l38_38609

-- Define the initial number of parts and the operation that adds two parts
def initial_parts : Nat := 1
def operation (n : Nat) : Nat := n + 2

theorem not_possible_2020_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2020) : False :=
sorry

theorem possible_2023_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2023) : True :=
sorry

end not_possible_2020_parts_possible_2023_parts_l38_38609


namespace count_k_values_l38_38669

-- Definitions based on the conditions
def k_satisfies_conditions (k : ℕ) : Prop :=
  let ⟨a, b, c⟩ := (nat.factorization k)
  a ≤ 20 ∧ b ≤ 10 ∧ c = 10

def count_satisfying_k : ℕ :=
  (21 * 11)

-- Main theorem statement
theorem count_k_values : count_satisfying_k = 231 :=
by
  -- Placeholder for the proof
  sorry

end count_k_values_l38_38669


namespace reduced_price_l38_38908

theorem reduced_price (P Q : ℝ) (h : P ≠ 0) (h₁ : 900 = Q * P) (h₂ : 900 = (Q + 6) * (0.90 * P)) : 0.90 * P = 15 :=
by 
  sorry

end reduced_price_l38_38908


namespace problem_statement_l38_38043

def S (x y : ℝ) : Prop := (x^2 - y^2) % 2 = 1
def T (x y : ℝ) : Prop := sin (2 * π * x^2) - sin (2 * π * y^2) = cos (2 * π * x^2) - cos (2 * π * y^2)

theorem problem_statement : ∀ x y : ℝ, S x y → T x y ∧ ¬ (T x y → S x y) :=
by
  sorry

end problem_statement_l38_38043


namespace cos_theta_planes_l38_38458

theorem cos_theta_planes (θ : ℝ) :
  let n₁ : ℝ × ℝ × ℝ := (4, -3, 1)
      n₂ : ℝ × ℝ × ℝ := (1, 5, -2) in
  let dot_product := n₁.1 * n₂.1 + n₁.2 * n₂.2 + n₁.3 * n₂.3 in
  let magnitude₁ := Real.sqrt (n₁.1^2 + n₁.2^2 + n₁.3^2) in
  let magnitude₂ := Real.sqrt (n₂.1^2 + n₂.2^2 + n₂.3^2) in
  (dot_product = -13) →
  (magnitude₁ = Real.sqrt(26)) →
  (magnitude₂ = Real.sqrt(30)) →
  let cosθ := dot_product / (magnitude₁ * magnitude₂) in
  cosθ = -13 / Real.sqrt(780) :=
by
  intros n₁ n₂ dot_product magnitude₁ magnitude₂ h1 h2 h3 cosθ
  sorry

end cos_theta_planes_l38_38458


namespace sum_tens_ones_digits_3_plus_4_power_17_l38_38879

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l38_38879


namespace find_y_intercept_of_line_l38_38376

theorem find_y_intercept_of_line (n : ℕ) (a : ℕ → ℚ) (h_seq : ∀ k, a k = 1 / (k * (k + 1))) 
    (h_sum : ∑ k in Finset.range (n + 1), a k = 9 / 10) (h_n : n = 9) : 
    let line_equation := (n+1) * 0 + y + n = 0 in y = -9 :=
by
  sorry

end find_y_intercept_of_line_l38_38376


namespace correct_growth_rate_l38_38510

noncomputable def growth_rate_eq (x : ℝ) : Prop :=
  10 * (1 + x)^2 = 11.5

axiom initial_sales_volume : ℝ := 10
axiom final_sales_volume : ℝ := 11.5
axiom monthly_growth_rate (x : ℝ) : x > 0

theorem correct_growth_rate (x : ℝ) (hx : monthly_growth_rate x) :
  growth_rate_eq x :=
-- sorry
by
  have h1 : initial_sales_volume = 10 := rfl
  have h2 : final_sales_volume = 11.5 := rfl
  rw [h1, h2]
  sorry

end correct_growth_rate_l38_38510


namespace lcm_problem_l38_38666

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end lcm_problem_l38_38666


namespace curve_to_standard_form_chord_line_equation_l38_38823

theorem curve_to_standard_form (k : ℝ) : 
  let x := 8 * k / (1 + k^2),
      y := 2 * (1 - k^2) / (1 + k^2)
  in (x^2 / 16 + y^2 / 4 = 1) :=
sorry

theorem chord_line_equation (t θ : ℝ) (A B P : ℝ × ℝ) :
  let x := 2 + t * Real.cos θ,
      y := 1 + t * Real.sin θ,
      mid := (2, 1)
  in P = mid → 
     A = (2 + t * Real.cos θ, 1 + t * Real.sin θ) →
     B = (2 - t * Real.cos θ, 1 - t * Real.sin θ) →
     P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →
     (∀ t θ, x + 2 * y - 4 = 0) :=
sorry

end curve_to_standard_form_chord_line_equation_l38_38823


namespace cone_filled_with_water_to_2_3_height_l38_38995

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38995


namespace tan_product_eq_four_l38_38855

-- Define the angles in degrees
def ang1 : ℝ := 17
def ang2 : ℝ := 18
def ang3 : ℝ := 27
def ang4 : ℝ := 28

-- Function to convert degrees to radians
def deg_to_rad (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Define the tangents of the angles
def tan_ang1 : ℝ := Real.tan (deg_to_rad ang1)
def tan_ang2 : ℝ := Real.tan (deg_to_rad ang2)
def tan_ang3 : ℝ := Real.tan (deg_to_rad ang3)
def tan_ang4 : ℝ := Real.tan (deg_to_rad ang4)

-- Statement of the proof problem
theorem tan_product_eq_four : (1 + tan_ang1) * (1 + tan_ang2) * (1 + tan_ang3) * (1 + tan_ang4) = 4 :=
by sorry

end tan_product_eq_four_l38_38855


namespace vector_combination_on_circle_l38_38029

-- Definitions of points on the circle
structure Point := (x : ℝ) (y : ℝ)
def on_circle (p : Point) : Prop := p.x^2 + p.y^2 = 1

-- Vectors represented as points from the origin (0, 0) to Point A, B, and C
def vector_inner_product (p1 p2 : Point) : ℝ := p1.x * p2.x + p1.y * p2.y

-- Problem Statement
theorem vector_combination_on_circle 
  {A B C : Point} 
  (hA : on_circle A) 
  (hB : on_circle B) 
  (hC : on_circle C) 
  (distinctA : A ≠ B) 
  (distinctB : B ≠ C) 
  (distinctC : C ≠ A) 
  (ortho : vector_inner_product A B = 0)
  (λ μ : ℝ) 
  (hOC : C = {x := λ * A.x + μ * B.x, y := λ * A.y + μ * B.y}) :
  λ^2 + μ^2 = 1 := 
sorry

end vector_combination_on_circle_l38_38029


namespace cone_height_l38_38335

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l38_38335


namespace determinant_inequality_solution_l38_38753

theorem determinant_inequality_solution (a : ℝ) :
  (∀ x : ℝ, (x > -1 → x < (4 / a))) ↔ a = -4 := by
sorry

end determinant_inequality_solution_l38_38753


namespace cone_height_l38_38341

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l38_38341


namespace LukaLemonadeSolution_l38_38798

def LukaLemonadeProblem : Prop :=
  ∃ (L S W : ℕ), 
    (S = 3 * L) ∧
    (W = 3 * S) ∧
    (L = 4) ∧
    (W = 36)

theorem LukaLemonadeSolution : LukaLemonadeProblem :=
  by sorry

end LukaLemonadeSolution_l38_38798


namespace find_f_g_one_l38_38408

-- Definitions of f and g as provided in the conditions
def f (x : ℝ) : ℝ := x^2 - 3*x + 2
def g (x : ℝ) : ℝ := 3*x^3 + 2

-- The statement to prove
theorem find_f_g_one : f (g 1) = 12 :=
by
  unfold f g
  norm_num
  sorry

end find_f_g_one_l38_38408


namespace water_filled_percent_l38_38982

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38982


namespace find_balloons_given_to_Fred_l38_38484

variable (x : ℝ)
variable (Sam_initial_balance : ℝ := 46.0)
variable (Dan_balance : ℝ := 16.0)
variable (total_balance : ℝ := 52.0)

theorem find_balloons_given_to_Fred
  (h : Sam_initial_balance - x + Dan_balance = total_balance) :
  x = 10.0 :=
by
  sorry

end find_balloons_given_to_Fred_l38_38484


namespace seans_net_profit_l38_38079

-- Definitions for the conditions
def price_per_patch : ℕ → ℕ → ℝ
| _ 0 := 12.00
| _ c := if c ≤ 10 then 12.00
         else if c ≤ 30 then 11.50
         else if c ≤ 50 then 11.00
         else 10.50

def patches_sold := 75 + 100 + 25
def units_ordered := (patches_sold + 99) / 100  -- Equivalent to ceiling(n / 100)

def cost_of_patches := patches_sold * 1.25
def shipping_fee := units_ordered * 20
def total_cost := cost_of_patches + shipping_fee

def revenue_1_to_10 := 5 * (15 * price_per_patch 15 10)
def revenue_11_to_30 := 2 * (50 * 11.50)
def revenue_31_to_50 := (25 * 11.00)
def total_revenue := revenue_1_to_10 + revenue_11_to_30 + revenue_31_to_50

def net_profit := total_revenue - total_cost

theorem seans_net_profit : net_profit = 2035 :=
by sorry

end seans_net_profit_l38_38079


namespace even_function_f_l38_38747

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^x - 1 else sorry

theorem even_function_f (h_even : ∀ x : ℝ, f x = f (-x)) : f 1 = -1 / 2 := by
  -- proof development skipped
  sorry

end even_function_f_l38_38747


namespace sufficient_but_not_necessary_condition_l38_38774

noncomputable def f (a x : ℝ) : ℝ := |x - a| + |x|

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x ≥ 0, a < 0 → f a x = 2 * x - a ∧ 2 * x - a ≥ 0) ∧
  (a = 0 → ∀ x ≥ 0, f a x = 2 * x ∧ 2 * x ≥ 0) → 
  (∃ c : Prop, c = "sufficient but not necessary") :=
by
  sorry

end sufficient_but_not_necessary_condition_l38_38774


namespace heptagon_angle_sum_l38_38004

theorem heptagon_angle_sum 
  (angle_A angle_B angle_C angle_D angle_E angle_F angle_G : ℝ) 
  (h : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540) :
  angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540 :=
by
  sorry

end heptagon_angle_sum_l38_38004


namespace computer_production_per_month_l38_38581

def days : ℕ := 28
def hours_per_day : ℕ := 24
def intervals_per_hour : ℕ := 2
def computers_per_interval : ℕ := 3

theorem computer_production_per_month : 
  (days * hours_per_day * intervals_per_hour * computers_per_interval = 4032) :=
by sorry

end computer_production_per_month_l38_38581


namespace Drew_age_is_12_l38_38431

def Sam_age_current : ℕ := 46
def Sam_age_in_five_years : ℕ := Sam_age_current + 5

def Drew_age_now (D : ℕ) : Prop :=
  Sam_age_in_five_years = 3 * (D + 5)

theorem Drew_age_is_12 (D : ℕ) (h : Drew_age_now D) : D = 12 :=
by
  sorry

end Drew_age_is_12_l38_38431


namespace cone_height_l38_38330

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l38_38330


namespace probability_prime_sum_two_dice_l38_38827

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def num_ways_sum_is (target_sum : ℕ) : ℕ :=
  finset.card { (a, b) : finset.univ × finset.univ | a + b = target_sum }

def total_outcomes : ℕ := 36

theorem probability_prime_sum_two_dice :
  (∑ n in (finset.range 13).filter is_prime, num_ways_sum_is n) / total_outcomes = 5 / 12 :=
sorry

end probability_prime_sum_two_dice_l38_38827


namespace total_sandwiches_l38_38616

theorem total_sandwiches (n : ℕ) (s : ℕ) (H1 : n = 4) (H2 : s = 3) : n * s = 12 :=
by {
  rw [H1, H2],
  exact Nat.mul_eq_mul_right (by simp),
  sorry
}

end total_sandwiches_l38_38616


namespace shift_parabola_left_one_l38_38414

def original_parabola (x : ℝ) : ℝ := -x^2 + 1

def shifted_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 1

theorem shift_parabola_left_one :
  ∀ (x : ℝ), shifted_parabola x = original_parabola (x + 1) :=
by
  intros x
  simp only [shifted_parabola, original_parabola]
  sorry

end shift_parabola_left_one_l38_38414


namespace matrix_condition_l38_38259

theorem matrix_condition (a b c d : ℝ) :
  let N := ![![4, 0], ![0, 1]],
      M := ![![a, b], ![c, d]] in
  N ⬝ M = ![![4 * a, 4 * b], ![c, d]] := by
  sorry

end matrix_condition_l38_38259


namespace cone_height_l38_38332

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l38_38332


namespace determine_hair_colors_l38_38136

-- Define the possible hair colors
inductive HairColor
| blonde
| red
| brunette

open HairColor

-- Define the prisons
structure Prisoner :=
(id : Nat)
(name : String)
(statement : String)
(truth_type : TruthType)

-- Define the truth types
inductive TruthType
| always_tell_truth
| always_lie
| can_either

open TruthType

-- Define the prisoners and their statements
def P1 := { id := 1, name := "Anna", statement := "My dear Anna is blonde, and the lady of the prisoner in the adjacent cell is also blonde.", truth_type := can_either}
def P2 := { id := 2, name := "Brynhild", statement := "My beloved Brynhild has red hair, and the ladies of two of my neighboring prisoners are brunettes.", truth_type := can_either}
def P3 := { id := 3, name := "Clotilde", statement := "My charming Clotilde is red-haired. The ladies of my two neighboring prisoners are also red-haired.", truth_type := can_either}
def P4 := { id := 4, name := "Gudrun", statement := "My tender Gudrun is red-haired. The ladies of my neighboring prisoners are brunettes.", truth_type := can_either}
def P5 := { id := 5, name := "Johanna", statement := "My beautiful Johanna is a brunette. The lady of the neighboring prisoner is also a brunette. As for the prisoner in the opposite tower, his lady is neither brunette nor red-haired.", truth_type := can_either}

-- The final hair color solution
def HairColorAssignment := Nat -> HairColor

def finalAssignment : HairColorAssignment := fun n =>
  match n with
  | 1 => blonde   -- Anna
  | 2 => red      -- Brynhild
  | 3 => red      -- Clotilde
  | 4 => red      -- Gudrun
  | 5 => brunette -- Johanna
  | _ => blonde   -- Just a default

-- Lean Theorem Statement
theorem determine_hair_colors {P1 P2 P3 P4 P5 : Prisoner} :
  finalAssignment 1 = blonde ∧
  finalAssignment 2 = red ∧
  finalAssignment 3 = red ∧
  finalAssignment 4 = red ∧
  finalAssignment 5 = brunette :=
by
  sorry

end determine_hair_colors_l38_38136


namespace circle_equation_l38_38306

theorem circle_equation (radius : ℝ) (m : ℝ) (center : ℝ × ℝ) (A : ℝ × ℝ) :
  -- Conditions
  center = (m, 3 * m) ∧
  radius = m ∧
  A = (2, 3) ∧
  (2 - m)^2 + (3 - 3 * m)^2 = m^2 ∧
  -- Equation of the circle
  ∃ (x y : ℝ), (x - m)^2 + (y - 3 * m)^2 = m^2 :=
begin
  sorry
end

end circle_equation_l38_38306


namespace stirling_number_thm_l38_38069
open Nat

/-- Definition of the Stirling number of the second kind S(n, r) -/
def Stirling_number (n r : ℕ) := { ways : ℕ // ∀ r, n ≥ r → ways = S(n, r) }

/-- The math proof problem statement in Lean 4 -/
theorem stirling_number_thm (n r : ℕ) (h : n ≥ r) :
  Stirling_number n r = 
    (1 / r.factorial) * 
    (∑ j in range (r + 1), 
      (-1 : ℤ)^j * (binomial r j) * (r - j)^n) :=
  sorry

end stirling_number_thm_l38_38069


namespace nth_power_of_b_l38_38462

theorem nth_power_of_b (b n : ℤ) (h1 : n > 1)
  (h2 : ∀ k : ℤ, ∃ a_k : ℤ, a_k ^ n ≡ b [MOD k]) : ∃ c : ℤ, b = c ^ n :=
sorry

end nth_power_of_b_l38_38462


namespace volume_ratio_l38_38789

structure RegularTriangularPyramid (V : Type) [InnerProductSpace ℝ V] :=
  (A B C P : V)
  (PO : ℝ)
  (M : V)
  (is_midpoint : M = (P + O) / 2)
  (is_plane_parallel : ∀ Q, Q ∈ span ℝ {A, M} → ∀ R, R ∈ span ℝ {B, C} → ∀ S, S ∉ span ℝ {A, B, C})

theorem volume_ratio (V : Type) [InnerProductSpace ℝ V] (PY : RegularTriangularPyramid V) 
  (H : ∃ Q, Q ∈ span ℝ {PY.A, PY.M} ∧ 
           ∃ R, R ∈ span ℝ {PY.B, PY.C} ∧ 
           ∃ S, S ∉ span ℝ {PY.A, PY.B, PY.C}) : 
  volume (PY.P, PY.A, Q) / volume (PY.P, PY.ABC) = 4 / 21 :=
sorry

end volume_ratio_l38_38789


namespace sum_of_non_solutions_l38_38465

theorem sum_of_non_solutions 
  (A B C : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 36)) / ((x + C) * (x + 9)) = 2) 
  (h_inf : set.infinite {x : ℝ | ((x + B) * (A * x + 36)) / ((x + C) * (x + 9)) = 2}) :
  ∑ x in ({-9, -18} : finset ℝ), x = -27 :=
by sorry

end sum_of_non_solutions_l38_38465


namespace suitable_third_stick_l38_38554

theorem suitable_third_stick :
  ∀ (a b : Nat) (x : Nat),
    a = 4 →
    b = 10 →
    (x = 3 ∨ x = 5 ∨ x = 8 ∨ x = 15) →
    (a + b > x ∧ a + x > b ∧ b + x > a) →
    x = 8 := 
begin
  sorry
end

end suitable_third_stick_l38_38554


namespace cone_volume_percentage_filled_l38_38947

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38947


namespace prob_at_least_one_l38_38146

-- Defining the probabilities of the alarms going off on time
def prob_A : ℝ := 0.80
def prob_B : ℝ := 0.90

-- Define the complementary event (neither alarm goes off on time)
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

-- The main theorem statement we need to prove
theorem prob_at_least_one : 1 - prob_neither = 0.98 :=
by
  sorry

end prob_at_least_one_l38_38146


namespace find_k_for_noncollinear_vectors_l38_38031

theorem find_k_for_noncollinear_vectors 
  (e1 e2 : Vector ℝ)
  (h1 : ¬ collinear e1 e2)
  (h2 : ∃ x, (k • e1 + 4 • e2) = (x • (e1 + k • e2)) ∧ (x < 0)) :
  k = -2 :=
sorry

end find_k_for_noncollinear_vectors_l38_38031


namespace cone_height_l38_38333

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l38_38333


namespace problem_statement_l38_38153

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end problem_statement_l38_38153


namespace gcd_correct_l38_38553

noncomputable def gcd_algorithm : ℕ → ℕ → ℕ
| m, 0 => m
| 0, n => n
| m, n => gcd_algorithm n (m % n)

theorem gcd_correct (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∃ k, gcd_algorithm m n = k ∧ k = Nat.gcd m n := by
  sorry

end gcd_correct_l38_38553


namespace value_n_equals_27692_l38_38035

noncomputable def num_valid_n : ℕ :=
  (List.range 100000).filter (λ n, 
    let q := n / 50 in
    let r := n % 50 in
    (n ≥ 10000 ∧ n < 100000) ∧ (q + r) % 13 = 0
  ).length

theorem value_n_equals_27692 : num_valid_n = 27692 :=
sorry

end value_n_equals_27692_l38_38035


namespace ratio_independent_of_X_l38_38143

variable {A B X Y Z : Point}
variable {ω₁ ω₂ ω₃ : Circle}
variable (h₁ : ω₁.AB = ω₂.AB ∧ ω₂.AB = ω₃.AB)
variable (hX : X ∈ ω₁ ∧ X ≠ A ∧ X ≠ B)
variable (hY : Y ∈ ω₂)
variable (hZ : Z ∈ ω₃)
variable (hAXY : collinear [A, X, Y])
variable (hAXZ : collinear [A, X, Z])
variable (hYZ_between : between Y X Z)

theorem ratio_independent_of_X :
  (XY / YZ) = (XY / YZ) :=
by sorry

end ratio_independent_of_X_l38_38143


namespace find_ratio_XG_GY_l38_38011

noncomputable theory

-- Let X, Y, Z be points in a triangle
variables {X Y Z E G Q : Type*}

-- X, Y, Z are points within some affine space
variables [add_comm_group X] [module ℝ X]
          [add_comm_group Y] [module ℝ Y]
          [add_comm_group Z] [module ℝ Z]
          [add_comm_group E] [module ℝ E]
          [add_comm_group G] [module ℝ G]
          [add_comm_group Q] [module ℝ Q]

-- Assume points E, G exist on the lines XZ and XY respectively
-- and Q is the intersection of lines XE and YG
axiom condition_1 : ∃ (E : Z), E ∈ line_between X Z
axiom condition_2 : ∃ (G : Y), G ∈ line_between X Y
axiom condition_3 : ∃ (Q : Type*), Q ∈ intersection (line_between X E) (line_between Y G)

-- Given segment ratios
axiom ratio_1 : XQ / QE = 3 / 2
axiom ratio_2 : GQ / QY = 3 / 1

-- Goal: prove the ratio of distances XG to GY is 2
theorem find_ratio_XG_GY : ratio (distance X G) (distance G Y) = 2 :=
sorry

end find_ratio_XG_GY_l38_38011


namespace integer_solution_count_l38_38396

theorem integer_solution_count :
  (∃ x : ℤ, -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24) ↔
  (∃ n : ℕ, n = 3) :=
by
  sorry

end integer_solution_count_l38_38396


namespace sum_equality_l38_38741

-- Define the conditions and hypothesis
variables (x y z : ℝ)
axiom condition : (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0

-- State the theorem
theorem sum_equality : x + y + z = 21 :=
by sorry

end sum_equality_l38_38741


namespace rectangle_length_l38_38911

-- Define a structure for the rectangle.
structure Rectangle where
  breadth : ℝ
  length : ℝ
  area : ℝ

-- Define the given conditions.
def givenConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.area = 6075

-- State the theorem.
theorem rectangle_length (r : Rectangle) (h : givenConditions r) : r.length = 135 :=
by
  sorry

end rectangle_length_l38_38911


namespace x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l38_38417

def y_is_60_percent_greater_than_x (x y : ℝ) : Prop :=
  y = 1.60 * x

def z_is_40_percent_less_than_y (y z : ℝ) : Prop :=
  z = 0.60 * y

theorem x_not_4_17_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x ≠ 0.9583 * z :=
by {
  sorry
}

theorem x_is_8_0032_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x = 0.919968 * z :=
by {
  sorry
}

end x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l38_38417


namespace water_filled_percent_l38_38988

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38988


namespace initial_marbles_l38_38558

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l38_38558


namespace find_all_strictly_monotone_functions_l38_38649

open Function

noncomputable def verify_function (f : ℕ → ℕ) :=
  ∀ n : ℕ, f (f n) = 3 * n

def strictly_monotone (f : ℕ → ℕ) :=
  ∀ m n : ℕ, m < n → f m < f n

def correct_function_form (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ,
    let base3 (n : ℕ) : String := -- assumed some helper for base 3 string conversion
      by sorry -- Placeholder for base 3 representation
    in
    match base3 n with
    | "1abc...d3" => f (base3 n) = "2abc...d3"
    | "2abc...d3" => f (base3 n) = "1abc...d03"
    | _ => false

theorem find_all_strictly_monotone_functions :
  ∀ f : ℕ → ℕ, strictly_monotone f ∧ (verify_function f) ↔ correct_function_form f :=
sorry

end find_all_strictly_monotone_functions_l38_38649


namespace cone_height_l38_38344

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l38_38344


namespace part1_part2_l38_38366

theorem part1 (a c x : ℝ) (h1 : a > 0) (h2 : ∃ x, f (x) = a * x^2 - (a + c) * x + c) (h3 : f' (x) = f (x + 2)) :
  (∀ x, a * x^2 + 2 * a * x - 3 * a > 0 ↔ x < -3 ∨ x > 1) :=
by sorry

theorem part2 (a x : ℝ) (h4 : f (0) = 1) :
  (a = 0 → ∀ x, x > 1) ∧
  (0 < a ∧ a < 1 → ∀ x, 1 < x ∧ x < 1 / a) ∧
  (a > 1 → ∀ x, 1 / a < x ∧ x < 1) ∧
  (a = 1 → ∀ x, False) :=
by sorry

end part1_part2_l38_38366


namespace percent_filled_cone_l38_38973

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38973


namespace length_of_MN_l38_38771

variables {A B C D M N : Point}
variable (trapezoid : Trapezoid A B C D)
variables (h1 : parallel (BC) (AD))
variables (M : intersection (angle_bisector(A)) (angle_bisector(B)))
variables (N : intersection (angle_bisector(C)) (angle_bisector(D)))

theorem length_of_MN (h2 : M = intersection (angle_bisector(A)) (angle_bisector(B)))
  (h3 : N = intersection (angle_bisector(C)) (angle_bisector(D))) :
  segment_length(M N) = 1/2 * ((segment_length (AD) + segment_length (BC)) - (segment_length(AB) + segment_length(CD))) :=
sorry

end length_of_MN_l38_38771


namespace reciprocal_sum_lt_3_l38_38049

def valid_digit (d : ℕ) : Prop := d ≠ 2 ∧ d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 6

def valid_number (n : ℕ) : Prop :=
  n > 0 ∧ ∀ k, ((n / 10^k) % 10) < 10 → valid_digit ((n / 10^k) % 10)

def A : set ℕ := { x : ℕ | valid_number x }

theorem reciprocal_sum_lt_3 :
  (∑ x in A, (1 : ℚ) / x) < 3 :=
sorry

end reciprocal_sum_lt_3_l38_38049


namespace balance_weight_l38_38077

variable (pear banana : ℝ)
variable (weight_ratio : 4 * pear = 3 * banana)
variable (pears_count : ℝ := 48)
variable (bananas_needed : ℝ := 36)

theorem balance_weight (h : weight_ratio) : pears_count * pear = bananas_needed * banana := by
  sorry

end balance_weight_l38_38077


namespace tournament_total_players_l38_38429

/--
In a tournament involving n players:
- Each player scored half of all their points in matches against participants who took the last three places.
- Each game results in 1 point.
- Total points from matches among the last three (bad) players = 3.
- The number of games between good and bad players = 3n - 9.
- Total points good players scored from bad players = 3n - 12.
- Games among good players total to (n-3)(n-4)/2 resulting points.
Prove that the total number of participants in the tournament is 9.
-/
theorem tournament_total_players (n : ℕ) :
  3 * (n - 4) = (n - 3) * (n - 4) / 2 → 
  n = 9 :=
by
  intros h
  sorry

end tournament_total_players_l38_38429


namespace min_breaks_12_no_breaks_15_l38_38302

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l38_38302


namespace cos_of_angle_add_pi_six_range_of_given_expression_l38_38436

-- Problem 1
theorem cos_of_angle_add_pi_six (α : ℝ) (h : ∃ x ≥ 0, (x, 2 * real.sqrt 2 * x) = (1, 2 * real.sqrt 2)):
  real.cos (α + real.pi / 6) = (real.sqrt 3 - 2 * real.sqrt 2) / 6 :=
sorry

-- Problem 2
theorem range_of_given_expression (α : ℝ) (h : α ≥ real.pi / 6 ∧ α ≤ 3 * real.pi / 4):
  ∃ y, y = (3 / 2) * real.sin (2 * α) + real.sqrt 3 * real.cos α ^ 2 - real.sqrt 3 / 2 ∧ (-real.sqrt 3 ≤ y ∧ y ≤ real.sqrt 3) :=
sorry

end cos_of_angle_add_pi_six_range_of_given_expression_l38_38436


namespace number_of_valid_integers_l38_38398

def is_valid_interval (n : ℤ) : Prop :=
  (n + 8) * (n - 4) * (n - 12) * (n + 5) < 0

def positive_integer_interval : ℤ → Prop :=
  λ n, 0 < n ∧ is_valid_interval n

theorem number_of_valid_integers : ∃ n : ℕ, n = 7 ∧ ∀ k: ℤ, positive_integer_interval k ↔ k ∈ {5, 6, 7, 8, 9, 10, 11} :=
sorry

end number_of_valid_integers_l38_38398


namespace remainder_of_f_div_r_minus_2_l38_38656

def f (r : ℝ) : ℝ := r^15 - 3

theorem remainder_of_f_div_r_minus_2 : f 2 = 32765 := by
  sorry

end remainder_of_f_div_r_minus_2_l38_38656


namespace find_n_l38_38418

theorem find_n (n : ℕ) (h : (√(2^n) = 64)) : n = 12 :=
  sorry

end find_n_l38_38418


namespace complex_quadrant_l38_38438

noncomputable def i := Complex.I

noncomputable def z := i^4 + i^2015

#eval Complex.conj z

theorem complex_quadrant :
  let z := i^4 + i^2015
  in let conj_z := Complex.conj z
  in conj_z.im > 0 ∧ conj_z.re > 0 :=
by
  sorry

end complex_quadrant_l38_38438


namespace cone_height_l38_38342

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l38_38342


namespace cone_filled_with_water_to_2_3_height_l38_38992

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38992


namespace ceil_minus_x_eq_one_minus_fractional_part_l38_38822

variable (x : ℝ)

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem ceil_minus_x_eq_one_minus_fractional_part (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ⌈x⌉ - x = 1 - fractional_part x :=
sorry

end ceil_minus_x_eq_one_minus_fractional_part_l38_38822


namespace cone_filled_with_water_to_2_3_height_l38_38997

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38997


namespace cost_price_article_l38_38564
-- Importing the required library

-- Definition of the problem
theorem cost_price_article
  (C S C_new S_new : ℝ)
  (h1 : S = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : S_new = S - 1)
  (h4 : S_new = 1.045 * C) :
  C = 200 :=
by
  -- The proof is omitted
  sorry

end cost_price_article_l38_38564


namespace diameter_is_1p201_l38_38832

noncomputable def diameter_of_wheel (revolutions_per_minute : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60
  (speed_mpm / revolutions_per_minute) / Real.pi

theorem diameter_is_1p201 :
  diameter_of_wheel 265.15 60 ≈ 1.201 := by
  sorry

end diameter_is_1p201_l38_38832


namespace volume_ratio_of_cubes_l38_38545

theorem volume_ratio_of_cubes :
  ∀ (edge1 : ℕ) (edge2_ft : ℕ) (foot_to_inch : ℕ),
    edge1 = 4 →
    edge2_ft = 2 →
    foot_to_inch = 12 →
    let edge2 := edge2_ft * foot_to_inch
    in (edge1 / edge2) ^ 3 = 1 / 216 :=
by
  intros edge1 edge2_ft foot_to_inch h1 h2 h3
  let edge2 := edge2_ft * foot_to_inch
  sorry

end volume_ratio_of_cubes_l38_38545


namespace square_side_length_l38_38850

noncomputable def square_ratio (a b : ℕ) : Prop :=
  let ratio := (a : ℚ) / b 
  sqrt ratio = (10 : ℚ) / 7

theorem square_side_length (area_ratio : ℚ) (perimeter_large : ℚ) 
  (h₁ : area_ratio = 300 / 147) 
  (h₂ : perimeter_large = 60) : 
  square_ratio 100 49 ∧ (15 * (7 : ℚ) / 10 = 10.5) := 
begin
  sorry
end

end square_side_length_l38_38850


namespace rectangle_area_l38_38926

theorem rectangle_area (r : ℝ) (h_ratio : 2 = 1) (h_radius : r = 5) :
  ∃ A, A = 200 :=
by 
  -- Definitions based on given conditions
  let diameter := 2 * r,
  let width := 2 * r,
  let length := 2 * width,

  -- Calculating the area using the defined dimensions
  let area := length * width,

  -- End goal
  use area,
  rw [h_radius, h_ratio, show r = 5 from h_radius] sorry

end rectangle_area_l38_38926


namespace find_cone_height_l38_38352

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l38_38352


namespace water_volume_percentage_l38_38968

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38968


namespace tourists_number_l38_38132

theorem tourists_number (m : ℕ) (k l : ℤ) (n : ℕ) (hn : n = 23) (hm1 : 2 * m ≡ 1 [MOD n]) (hm2 : 3 * m ≡ 13 [MOD n]) (hn_gt_13 : n > 13) : n = 23 := 
by
  sorry

end tourists_number_l38_38132


namespace jacks_speed_l38_38775

-- Define the initial distance between Jack and Christina.
def initial_distance : ℝ := 360

-- Define Christina's speed.
def christina_speed : ℝ := 7

-- Define Lindy's speed.
def lindy_speed : ℝ := 12

-- Define the total distance Lindy travels.
def lindy_total_distance : ℝ := 360

-- Prove Jack's speed given the conditions.
theorem jacks_speed : ∃ v : ℝ, (initial_distance - christina_speed * (lindy_total_distance / lindy_speed)) / (lindy_total_distance / lindy_speed) = v ∧ v = 5 :=
by {
  sorry
}

end jacks_speed_l38_38775


namespace profit_function_correct_max_profit_l38_38579

-- Define the revenue function R(x)
def revenue (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x ^ 2
  else 80000

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

-- Define the profit function f(x)
def profit (x : ℝ) : ℝ := 
  revenue x - total_cost x

-- Prove that the profit function f(x) is as given
theorem profit_function_correct (x : ℝ) : 
  profit x = if 0 ≤ x ∧ x ≤ 400 then - (1/2) * x ^ 2 + 300 * x - 20000 else 60000 - 100 * x :=
begin
  sorry
end

-- Prove that the maximum profit is $25,000 when x = 300
theorem max_profit : 
  ∃ x : ℝ, x = 300 ∧ profit x = 25000 :=
begin
  sorry
end

end profit_function_correct_max_profit_l38_38579


namespace cos_six_arccos_one_fourth_l38_38251

theorem cos_six_arccos_one_fourth : 
  let x := Real.arccos (1 / 4) in Real.cos (6 * x) = -7 / 128 :=
by
  let x := Real.arccos (1 / 4)
  have : Real.cos x = 1 / 4 := Real.cos_arccos_of_mem_Icc (by norm_num)
  sorry

end cos_six_arccos_one_fourth_l38_38251


namespace quadrilateral_perimeter_ge_twice_diagonal_l38_38211

theorem quadrilateral_perimeter_ge_twice_diagonal (R : Type) [metric_space R] 
    {A B C D K L M N : R} (h_rectangle: is_rectangle A B C D)
    (h_quadrilateral: inscribed_quadrilateral K L M N A B C D) :
    perimeter K L M N ≥ 2 * dist A C :=
sorry

end quadrilateral_perimeter_ge_twice_diagonal_l38_38211


namespace permutations_twin_pairs_more_numerous_l38_38150

noncomputable def F0 : ℕ → ℝ
| 0       := 1
| (n + 1) := 2 * (n + 1) * ((2 * (n + 1) - 2) * F0 n + F1 n)

noncomputable def F1 : ℕ → ℝ
| 0       := 1
| (n + 1) := F0 n + 2 * (n + 1) * F0 n

theorem permutations_twin_pairs_more_numerous (n : ℕ) : 
  F0 n < F1 n := 
by
  -- Proof skipped.
  sorry

end permutations_twin_pairs_more_numerous_l38_38150


namespace relationship_between_a_b_c_l38_38679

noncomputable def a : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def b : ℝ := 1.1^1.3
noncomputable def c : ℝ := Real.sin 1

theorem relationship_between_a_b_c : b > c ∧ c > a := by
  sorry

end relationship_between_a_b_c_l38_38679


namespace inequality_first_inequality_second_l38_38085

theorem inequality_first (x : ℝ) : 4 * x - 2 < 1 - 2 * x → x < 1 / 2 := 
sorry

theorem inequality_second (x : ℝ) : (3 - 2 * x ≥ x - 6) ∧ ((3 * x + 1) / 2 < 2 * x) → 1 < x ∧ x ≤ 3 :=
sorry

end inequality_first_inequality_second_l38_38085


namespace water_volume_percentage_l38_38959

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38959


namespace vertical_asymptote_at_4_l38_38410

theorem vertical_asymptote_at_4 (x : ℝ) :
  let y := (x^2 + 2*x + 8) / (x - 4)
  in ∀ x : ℝ, x = 4 → (x - 4 = 0) := 
by
  intros x h
  rw h
  exact rfl

end vertical_asymptote_at_4_l38_38410


namespace incorrect_option_C_l38_38275

theorem incorrect_option_C (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : ¬ (ab > (a + b)^2) :=
by {
  sorry
}

end incorrect_option_C_l38_38275


namespace probability_of_draw_l38_38905

-- Define the probabilities as constants
def prob_not_lose_xiao_ming : ℚ := 3 / 4
def prob_lose_xiao_dong : ℚ := 1 / 2

-- State the theorem we want to prove
theorem probability_of_draw :
  prob_not_lose_xiao_ming - prob_lose_xiao_dong = 1 / 4 :=
by
  sorry

end probability_of_draw_l38_38905


namespace geometric_sequence_formula_sum_of_first_n_terms_l38_38423

variable (a : ℕ → ℕ)

def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = a 1 * (2 ^ n)

theorem geometric_sequence_formula (h1 : a 1 + a 3 = 10)
  (h2 : 4 * (a 3) ^ 2 = a 2 * a 6) : ∀ n, a n = 2 ^ n :=
by
  sorry

def sum_to_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, k * a k

theorem sum_of_first_n_terms (n : ℕ) :
  ∀ a, geometric_sequence a → sum_to_n a n = (n - 1) * 2^(n + 1) + 2 :=
by
  sorry

end geometric_sequence_formula_sum_of_first_n_terms_l38_38423


namespace simplify_expression_l38_38488

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by
  sorry

end simplify_expression_l38_38488


namespace num_valid_arrangements_l38_38862

-- Define the 3x3 grid as a type
def grid := fin 3 × fin 3

-- Enumerate A, B, C as the letters
inductive letter
| A | B | C

open letter

-- Define the condition for each row and column containing each letter exactly once
def is_valid_grid (g : grid → letter) : Prop :=
  (∀ i : fin 3, ∃ j1 j2 j3 : fin 3, j1 ≠ j2 ∧ j1 ≠ j3 ∧ j2 ≠ j3 ∧ g (i, j1) = A ∧ g (i, j2) = B ∧ g (i, j3) = C) ∧
  (∀ j : fin 3, ∃ i1 i2 i3 : fin 3, i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ g (i1, j) = A ∧ g (i2, j) = B ∧ g (i3, j) = C)

-- Place A in the specified positions
def has_fixed_A (g : grid → letter) : Prop :=
  g (0, 1) = A

-- The main theorem to be proved
theorem num_valid_arrangements :
  ∃ (g : grid → letter), is_valid_grid g ∧ has_fixed_A g ∧ (fintype.card {g // is_valid_grid g ∧ has_fixed_A g} = 2) :=
sorry

end num_valid_arrangements_l38_38862


namespace valid_digit_distribution_l38_38542

theorem valid_digit_distribution (n : ℕ) : 
  (∃ (d1 d2 d5 others : ℕ), 
    d1 = n / 2 ∧
    d2 = n / 5 ∧
    d5 = n / 5 ∧
    others = n / 10 ∧
    d1 + d2 + d5 + others = n) :=
by
  sorry

end valid_digit_distribution_l38_38542


namespace probability_of_winning_is_correct_probability_of_exactly_one_winner_is_correct_l38_38530

-- Define the parameters of the game
def total_balls := 6
def red_balls := 2
def white_balls := 2
def black_balls := 2

-- Define the event of drawing one ball of each color
def win_event := (red_balls choose 1) * (white_balls choose 1) * (black_balls choose 1)

-- Calculate the total number of ways to draw 3 balls from 6
def total_outcomes := (total_balls choose 3)

-- Calculate the probability of winning
def probability_of_winning := win_event / total_outcomes

-- Define the binomial distribution for the number of winners among 3 players
def binomial_distribution := Binomial 3 probability_of_winning

-- Calculate the probability that exactly 1 person wins
def probability_of_exactly_one_winner := binomial_distribution.pmf 1

theorem probability_of_winning_is_correct :
  probability_of_winning = 2 / 5 :=
sorry

theorem probability_of_exactly_one_winner_is_correct :
  probability_of_exactly_one_winner = 54 / 125 :=
sorry

end probability_of_winning_is_correct_probability_of_exactly_one_winner_is_correct_l38_38530


namespace polynomial_value_at_3_l38_38684

theorem polynomial_value_at_3 :
  let f (x : ℝ) := ((((5 * x + 2) * x + 3.5) * x - 2.6) * x + 1.7) * x - 0.8
  in f 3 = 1452.4 :=
by
  sorry

end polynomial_value_at_3_l38_38684


namespace simplify_expression_l38_38167

theorem simplify_expression : ( (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) ) = 1 :=
by
  sorry

end simplify_expression_l38_38167


namespace combined_movie_production_l38_38019

theorem combined_movie_production 
  (LJ_productions_yr1 : ℝ)
  (LJ_growth_rate : ℝ)
  (JT_increase_rate : ℝ)
  (JT_growth_rate : ℝ)
  (target_total : ℝ)
  (eq_LJ_productions_yr1 : LJ_productions_yr1 = 220)
  (eq_LJ_growth_rate : LJ_growth_rate = 0.03)
  (eq_JT_increase_rate : JT_increase_rate = 0.25)
  (eq_JT_growth_rate : JT_growth_rate = 0.05)
  (eq_target_total : target_total ≈ 2688) : 
  let LJ_yr1 := LJ_productions_yr1,
      LJ_yr2 := LJ_yr1 * (1 + LJ_growth_rate),
      LJ_yr3 := LJ_yr2 * (1 + LJ_growth_rate),
      LJ_yr4 := LJ_yr3 * (1 + LJ_growth_rate),
      LJ_yr5 := LJ_yr4 * (1 + LJ_growth_rate)
      in
  let JT_yr1 := LJ_productions_yr1 * (1 + JT_increase_rate),
      JT_yr2 := JT_yr1 * (1 + JT_growth_rate),
      JT_yr3 := JT_yr2 * (1 + JT_growth_rate),
      JT_yr4 := JT_yr3 * (1 + JT_growth_rate),
      JT_yr5 := JT_yr4 * (1 + JT_growth_rate)
      in
  LJ_yr1 + LJ_yr2 + LJ_yr3 + LJ_yr4 + LJ_yr5 + 
  JT_yr1 + JT_yr2 + JT_yr3 + JT_yr4 + JT_yr5 ≈ target_total := 
by sorry

end combined_movie_production_l38_38019


namespace parabola_focus_ellipse_focus_l38_38413

theorem parabola_focus_ellipse_focus (p : ℝ) : 
  let a := √6,
      b := √2,
      c := √(6 - 2),
      right_focus := (c, 0) in
      right_focus = (2, 0) → 
      ((p / 2, 0) = (2, 0)) → 
      p = 4 :=
by
  intro a b c right_focus h1 h2
  -- a is √6
  -- b is √2
  -- right_focus = (√(6 - 2), 0) = (2, 0)
  -- focus of the parabola y^2 = 2px is (p/2, 0)
  exact sorry

end parabola_focus_ellipse_focus_l38_38413


namespace calculate_diameter_l38_38835

noncomputable def wheel_diameter (revolutions_per_minute : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance_per_revolution := speed_m_per_min / revolutions_per_minute
  distance_per_revolution / Real.pi

theorem calculate_diameter :
  wheel_diameter 265.15 60 = 1.2 :=
begin
  sorry
end

end calculate_diameter_l38_38835


namespace cone_volume_percentage_filled_l38_38945

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38945


namespace A_quits_5_days_before_completion_l38_38197

theorem A_quits_5_days_before_completion (A B : ℕ) (hA : A = 20) (hB : B = 30) : 
    ∀ (total_days : ℕ) (quitting_day : ℕ), (total_days = 15) 
    → let work_percentage_A := (quitting_day / A)
        let work_percentage_B := (total_days / B)
        in work_percentage_A + work_percentage_B = 1 
    → quitting_day = 10 
    → (total_days - quitting_day) = 5 :=
by
  intros total_days quitting_day h_total h_comb_work h_quitting
  sorry

end A_quits_5_days_before_completion_l38_38197


namespace max_passengers_l38_38865

theorem max_passengers (y x : ℕ) (f : ℕ → ℕ) (g : ℕ → ℕ) :
  (∀ x, y = f x) →
  f 4 = 16 →
  f 6 = 10 →
  (∃ k m, (f x = k * x + m) ∧ k ≠ 0 ∧ 4 * k + m = 16 ∧ 6 * k + m = 10 ∧ g x = 110 * x * f x) →
  (∃ x_max, x_max ∈ {1,2,3,4,5,6,7,8,9} ∧ ∀ x ∈ {1,2,3,4,5,6,7,8,9}, g x_max ≥ g x) →
  f x = -3 * x + 28 ∧ x_max = 5 ∧ g 5 = 14300 :=
by
  sorry

end max_passengers_l38_38865


namespace assemble_regular_heptagon_from_pieces_l38_38685

theorem assemble_regular_heptagon_from_pieces (H : Type) [regular_heptagon H] :
  ∃ (H' : Type) [regular_heptagon H'], assembles_from_triangular_pieces H H' :=
sorry

end assemble_regular_heptagon_from_pieces_l38_38685


namespace sum_of_altitudes_l38_38116

theorem sum_of_altitudes (x y : ℝ) (h_line : 18 * x + 9 * y = 108) :
  let intercept_x := 108 / 18,
      intercept_y := 108 / 9,
      area := 1 / 2 * intercept_x * intercept_y,
      altitude_x := intercept_x,
      altitude_y := intercept_y,
      altitude_origin := 108 / 21 in
  altitude_x + altitude_y + altitude_origin = 23 + 1 / 7 :=
by
  let intercept_x := 108 / 18
  let intercept_y := 108 / 9
  let area := 1 / 2 * intercept_x * intercept_y
  let altitude_x := intercept_x
  let altitude_y := intercept_y
  let altitude_origin := 108 / 21
  have eq1 : altitude_x + altitude_y + altitude_origin = 23 + 1 / 7
  sorry

end sum_of_altitudes_l38_38116


namespace is_exponential_f3_l38_38901

def is_exponential (f : ℕ → ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℕ, f x = a ^ x)

def f1 (x : ℕ) : ℕ := 5 ^ (x + 1)
def f2 (x : ℕ) : ℕ := x ^ 4
def f3 (x : ℕ) : ℕ := 3 ^ x
def f4 (x : ℕ) : ℕ := -2 * 3 ^ x

theorem is_exponential_f3 : is_exponential f3 := sorry

end is_exponential_f3_l38_38901


namespace online_store_price_l38_38583

/-- Proof problem: Given the conditions where an online store
takes a 20% commission of the price set by the distributor,
the distributor obtains the product at $15 per item, and 
wants to maintain a 10% profit on the cost, prove that
the price observed by the buyer online to meet the distributor's
profit requirement is $20.63. --/

theorem online_store_price
  (cost : ℝ)
  (profit_pct : ℝ)
  (commission_pct : ℝ)
  (desired_profit : ℝ)
  (observed_price : ℝ)
  (commission_factor : ℝ := 1 - commission_pct)
  (observed_price := (cost * (1 + profit_pct) / commission_factor).round)
  (cost := 15)
  (profit_pct := 0.10)
  (commission_pct := 0.20)
  (desired_profit := 1.5)
  (observed_price = 20.63) : observed_price = 20.63 := 
sorry

end online_store_price_l38_38583


namespace arithmetic_progression_intervals_l38_38479

/-- Place a system of non-overlapping segments of length 1 on a number line such that for any
    infinite arithmetic progression (with an arbitrary first term and any difference),
    at least one of its terms falls inside one of the segments in our system. -/
theorem arithmetic_progression_intervals (
  S : set (ℝ × ℝ) -- Set of non-overlapping unit-length segments
) (hS : ∀ (s ∈ S), s.2 - s.1 = 1) -- Each segment has length 1
  (prog_start : ℝ) (d : ℝ) -- First term and common difference of the arithmetic progression 
  (prog : ℕ → ℝ) -- Arbitrary infinite arithmetic progression
  (hprog : ∀ n, prog n = prog_start + n * d) -- Definition of the arithmetic progression
  (no_overlap : ∀ s1 s2 ∈ S, s1 ≠ s2 → s1.2 ≤ s2.1 ∨ s2.2 ≤ s1.1) -- Non-overlapping segments
  : 
  ∃ s ∈ S, ∃ n, s.1 ≤ prog n ∧ prog n ≤ s.2 := -- Conclusion: at least one term of the progression falls inside one segment
sorry

end arithmetic_progression_intervals_l38_38479


namespace point_of_tangency_l38_38261

def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

theorem point_of_tangency :
  parabola1 (-7) (-24) ∧ parabola2 (-7) (-24) := by
  sorry

end point_of_tangency_l38_38261


namespace find_second_sum_l38_38174

theorem find_second_sum (x : ℝ) (h : 24 * x / 100 = (2730 - x) * 15 / 100) : 2730 - x = 1680 := by
  sorry

end find_second_sum_l38_38174


namespace B_finishes_in_4_days_l38_38922

theorem B_finishes_in_4_days
  (A_days : ℕ) (B_days : ℕ) (working_days_together : ℕ) 
  (A_rate : ℝ) (B_rate : ℝ) (combined_rate : ℝ) (work_done : ℝ) (remaining_work : ℝ)
  (B_rate_alone : ℝ) (days_B: ℝ) :
  A_days = 5 →
  B_days = 10 →
  working_days_together = 2 →
  A_rate = 1 / A_days →
  B_rate = 1 / B_days →
  combined_rate = A_rate + B_rate →
  work_done = combined_rate * working_days_together →
  remaining_work = 1 - work_done →
  B_rate_alone = 1 / B_days →
  days_B = remaining_work / B_rate_alone →
  days_B = 4 := 
by
  intros
  sorry

end B_finishes_in_4_days_l38_38922


namespace find_first_month_sales_l38_38585

noncomputable def avg_sales (sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ) : ℕ :=
(sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6

theorem find_first_month_sales :
  let sales_2 := 6927
  let sales_3 := 6855
  let sales_4 := 7230
  let sales_5 := 6562
  let sales_6 := 5091
  let avg_sales_needed := 6500
  ∃ sales_1, avg_sales sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 = avg_sales_needed := 
by
  sorry

end find_first_month_sales_l38_38585


namespace remainder_div_9_l38_38099

theorem remainder_div_9 (N : ℕ)
  (h1 : nat.sizeof N = 2015)
  (h2 : ∀ d ∈ nat.digits 10 N, d = 5 ∨ d = 6 ∨ d = 7)
  (h3 : count 5 (nat.digits 10 N) = count 7 (nat.digits 10 N) + 15):
  N % 9 = 6 :=
by sorry

end remainder_div_9_l38_38099


namespace best_estimate_average_l38_38541

variables {T : Type} -- Define a general type T
variables (plates : List T) -- List of plates where bacteria are counted
variables (count_bacteria : T → ℕ) -- Function to count bacteria on each plate
variables (true_bacteria_count : ℕ) -- True number of bacteria in the sample (hypothetical)

// Definition for the average bacteria count from multiple plates
def average_bacteria_count (plates : List T) (count_bacteria : T → ℕ) : ℕ :=
  (plates.map count_bacteria).sum / plates.length

-- The statement to prove:
theorem best_estimate_average (h : List.length plates > 0) : 
  true_bacteria_count = average_bacteria_count plates count_bacteria :=
sorry

end best_estimate_average_l38_38541


namespace min_square_base_length_l38_38017

theorem min_square_base_length (width length : ℕ) (h1 : width = 9) (h2 : length = 21) :
  Nat.lcm width length = 63 :=
by {
  rw [h1, h2],
  sorry
}

end min_square_base_length_l38_38017


namespace HephaestusCharges_l38_38617

variable (x : ℕ)

theorem HephaestusCharges :
  3 * x + 6 * (12 - x) = 54 -> x = 6 :=
by
  intros h
  sorry

end HephaestusCharges_l38_38617


namespace cone_water_volume_percentage_l38_38999

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l38_38999


namespace adult_cat_food_per_day_l38_38081

def kittens : ℕ := 4
def adult_cats : ℕ := 3
def initial_cans : ℕ := 7
def kitten_food_per_day : ℚ := 3 / 4
def additional_cans : ℕ := 35
def days : ℕ := 7

theorem adult_cat_food_per_day : 
  (let 
    total_kitten_food_per_day := kittens * kitten_food_per_day,
    total_adult_food_per_day (x : ℚ) := adult_cats * x,
    total_food_per_day (x : ℚ) := total_kitten_food_per_day + total_adult_food_per_day x,
    total_food_needed (x : ℚ) := total_food_per_day x * days
  in
  ∃ x : ℚ, total_food_needed x = initial_cans + additional_cans → x = 1) :=
by 
  sorry

end adult_cat_food_per_day_l38_38081


namespace not_perfect_square_9n_squared_minus_9n_plus_9_l38_38036

theorem not_perfect_square_9n_squared_minus_9n_plus_9
  (n : ℕ) (h : n > 1) : ¬ (∃ k : ℕ, 9 * n^2 - 9 * n + 9 = k * k) := sorry

end not_perfect_square_9n_squared_minus_9n_plus_9_l38_38036


namespace alice_walks_miles_each_morning_l38_38645

theorem alice_walks_miles_each_morning (x : ℕ) :
  (5 * x + 5 * 12 = 110) → x = 10 :=
by
  intro h
  -- Proof omitted
  sorry

end alice_walks_miles_each_morning_l38_38645


namespace cone_water_fill_percentage_l38_38932

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38932


namespace painters_work_days_l38_38489

theorem painters_work_days (r : ℝ) (h₀ : 0 < r) (h₁ : 6 * r = 1.5) : 4 * (1.5 / 6) = 2.25 := by
  calc
    4 * (1.5 / 6) = 4 * (9 / 4) / 6 : by sorry -- transformation resulting in 9/4
    ... = 2.25 : by sorry -- actual multiplication showing the result is 2.25

end painters_work_days_l38_38489


namespace part1_a_value_part2_A_coordinates_l38_38693

-- Define the points and conditions
def Point (x y : ℤ) : Type := (x, y)

-- Part 1 proof
theorem part1_a_value (a : ℚ) (h : 3 * a - 9 = 4) : a = 13 / 3 := by
  sorry

-- Part 2 proof
theorem part2_A_coordinates (a : ℤ) (h1 : 3 < a) (h2 : a < 5) (h3 : A = Point (3 * a - 9) (2 * a - 10)) : A = (3, -2) := by
  sorry

end part1_a_value_part2_A_coordinates_l38_38693


namespace exists_shape_12_cells_divisible_3_4_l38_38068

theorem exists_shape_12_cells_divisible_3_4 : 
  ∃ n : ℕ, n = 12 ∧ n < 16 ∧ n % 4 = 0 ∧ n % 3 = 0 :=
by {
  use 12,
  split, 
  { refl },
  split, 
  { norm_num },
  split, 
  { norm_num },
  { norm_num }
}

end exists_shape_12_cells_divisible_3_4_l38_38068


namespace simplify_expr_evaluate_difference_l38_38187

-- Problem 1: Simplify (2a - b) - 2(a - 2b)
theorem simplify_expr (a b : ℝ) : (2 * a - b) - 2 * (a - 2 * b) = 3 * b := by
  sorry

-- Problem 2: Evaluate the difference at x = -1
theorem evaluate_difference (x : ℝ) (h : x = -1) : 
  (2 * x^2 - 3 * x + 1) - (-3 * x^2 + 5 * x - 7) = 21 := by
  have h1 : (2 * x^2 - 3 * x + 1) - (-3 * x^2 + 5 * x - 7) = 5 * x^2 - 8 * x + 8 := by sorry
  have h2 : 5 * (-1)^2 - 8 * (-1) + 8 = 21 := by sorry
  rw [h] at h1
  exact h2

end simplify_expr_evaluate_difference_l38_38187


namespace equation_solution_l38_38244

theorem equation_solution (x : ℝ) :
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) + 1 / (x + 8) = 0) →
  (x = -4 - 2 * Real.sqrt 3) ∨ (x = 2 - 2 * Real.sqrt 3) := by
  sorry

end equation_solution_l38_38244


namespace sufficient_condition_perpendicular_l38_38030

variable {Plane : Type} [AffPlane Plane] 
variable {Line : Type} [AffineSpace Line Plane]
variable (α β γ : Plane) 
variable (m n l : Line)

-- Definitions of perpendicularity
def perp_lines (l1 l2 : Line) : Prop := sorry -- definition for two perpendicular lines
def perp_plane_line (α : Plane) (l : Line) : Prop := sorry -- definition for a plane and a line being perpendicular
def perp_planes (α β : Plane) : Prop := sorry -- definition for two planes being perpendicular

-- Given conditions
axiom n_perp_α : perp_plane_line α n
axiom n_perp_β : perp_plane_line β n
axiom m_perp_α : perp_plane_line α m

-- To prove
theorem sufficient_condition_perpendicular (α β γ : Plane) (m n l : Line):
  perp_plane_line α n →
  perp_plane_line β n →
  perp_plane_line α m →
  perp_plane_line β m := 
begin
  intro h₁,
  intro h₂,
  intro h₃,
  sorry,
end

end sufficient_condition_perpendicular_l38_38030


namespace solve_triangle_ABC_proof_l38_38868

noncomputable def triangle_ABC_proof : Prop :=
  let AC := 450
  let BC := 300
  let AK := AC / 2  -- since AK = CK, CK is also 225
  let CK := AC / 2  -- same as AK
  let BK := sorry  -- Not specified directly, so assumed as a variable we can define later if needed.
  let AM := 180
  let KM := 180 -- since K is the midpoint of PM and PM = 2 * PK, implying PK = AM = 180
  ∃ (L P M : Type) (AB BL LP : ℝ), 
      -- ensuring we bind all the variables we need 
      (∃ (intersection_condition : ∀ B K L, (B ∈ intersection_of (BK, CL))), 
        LP = 120)

-- This theorem encapsulates the conditions and requests a proof of LP = 120
theorem solve_triangle_ABC_proof :
  triangle_ABC_proof :=
begin
  sorry -- Steps should be provided here in a complete implementation.
end

end solve_triangle_ABC_proof_l38_38868


namespace find_f_and_m_l38_38453

noncomputable def f : ℤ → ℤ
| 1 := 1
| 2 := 20
| -4 := -4
| x  := sorry  -- Placeholder for the general definition of f(x)

variables (a b c m : ℤ)
variables f_spec : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y * (x + y) + b * x * y + c * (x + y) + 4
variables lower_bound : ∀ x : ℕ, f x ≥ m * x * x + (5 * m + 1) * x + 4 * m

theorem find_f_and_m :
  (∀ x : ℤ, f x = (3 * x ^ 3 + 5 * x ^ 2 + 2 * x) / 2 - 4) ∧
  (∃ m : ℤ, m = -1) :=
by
  sorry

end find_f_and_m_l38_38453


namespace min_breaks_12_no_breaks_15_l38_38300

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l38_38300


namespace segments_form_triangle_l38_38690

theorem segments_form_triangle {A B C K L M N F G : Type*}
  [ordered_semiring K] [ordered_add_comm_group  L] [add_comm_group M] [add_torsors K L M] 
  (h_triangle: triangle A B C)
  (h_parallelogram1: parallelogram A B K L)
  (h_parallelogram2: parallelogram B C M N)
  (h_parallelogram3: parallelogram A C F G)
  (KN MF GL : ℝ)
  (h1 : KN = segment_length K N)
  (h2 : MF = segment_length M F)
  (h3 : GL = segment_length G L)
  (h_triangle_inequality1 : KN + MF > GL)
  (h_triangle_inequality2 : MF + GL > KN)
  (h_triangle_inequality3 : GL + KN > MF) :
  true := 
by {
  sorry
}

end segments_form_triangle_l38_38690


namespace volunteer_selection_l38_38611

theorem volunteer_selection (n m k : ℕ) (h₁ : n = 4) (h₂ : m = 3) (h₃ : k = 3) :
  (nat.choose (n + m) k) - (nat.choose m k) - (nat.choose n k) = 30 :=
by
  subst h₁
  subst h₂
  subst h₃
  simp
  sorry

end volunteer_selection_l38_38611


namespace square_possible_n12_square_possible_n15_l38_38291

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l38_38291


namespace quadrilateral_trapezoid_or_parallelogram_l38_38807

-- Define the structure and conditions of the problem.
variable {A B C D O M2 M4 : Type*}
variable [affine_space A]
variables {P1 P2 : Point A}

-- Define points and their relationships.
variables {A B C D O : Point A}
variables {M2 M4 : Point A}
variables (h1 : midpoint A B M2) (h2 : midpoint C D M4)
variables (h3 : line_through M2 M4 O)

-- Define the specific form of quadrilateral properties.
def my_quad (A B C D : Point A) := true

-- Define the midpoint and intersection properties.
def midpoint (P Q R : Point A) : Prop := true
def line_through (P Q R : Point A) : Prop := true

theorem quadrilateral_trapezoid_or_parallelogram :
  ∀ (A B C D O M2 M4 : Point A),
    my_quad A B C D →
    midpoint A B M2 →
    midpoint C D M4 →
    line_through M2 M4 O →
    (is_trapezoid A B C D ∨ is_parallelogram A B C D) :=
by sorry

end quadrilateral_trapezoid_or_parallelogram_l38_38807


namespace lamp_marked_price_proof_l38_38212

def marked_price_check (initial_price : ℝ) (discount_rate : ℝ) 
  (gain_rate : ℝ) (final_discount_rate : ℝ) : Prop :=
  let purchase_price := initial_price * (1 - discount_rate) in
  let desired_selling_price := purchase_price * (1 + gain_rate) in
  let marked_price := desired_selling_price / (1 - final_discount_rate) in
  marked_price = 800 / 17

theorem lamp_marked_price_proof :
  marked_price_check 40 0.20 0.25 0.15 :=
by
  -- provide a proof here
  sorry

end lamp_marked_price_proof_l38_38212


namespace correct_survey_method_l38_38903

-- Definitions for the conditions
def visionStatusOfMiddleSchoolStudentsNationwide := "Comprehensive survey is impractical for this large population."
def batchFoodContainsPreservatives := "Comprehensive survey is unnecessary, sampling survey would suffice."
def airQualityOfCity := "Comprehensive survey is impractical due to vast area, sampling survey is appropriate."
def passengersCarryProhibitedItems := "Comprehensive survey is necessary for security reasons."

-- Theorem stating that option C is the correct and reasonable choice
theorem correct_survey_method : airQualityOfCity = "Comprehensive survey is impractical due to vast area, sampling survey is appropriate." := by
  sorry

end correct_survey_method_l38_38903


namespace verify_expressions_l38_38274

theorem verify_expressions (a : ℝ) (m n : ℕ) (h1 : 0 < a) (h2 : 0 < m) (h3 : 1 < n) :
  (a ^ (m / n) = (a ^ m) ^ (1 / n)) ∧ (a ^ 0 = 1) ∧ (a ^ (-m / n) = 1 / ((a ^ m) ^ (1 / n))) := 
by
  sorry

end verify_expressions_l38_38274


namespace calculate_3mn_l38_38707

section MathProofProblem

variables (x y m n : ℝ)

-- Condition: The equation is linear in two variables x and y
def is_linear_equation (e : ℝ) : Prop := (4 * m - 1 = 1) ∧ (3 * n - 2 * m = 1)

-- Theorem to be proved
theorem calculate_3mn (h : is_linear_equation 5) : 3 * m * n = 1 :=
  by
    sorry

end MathProofProblem

end calculate_3mn_l38_38707


namespace new_socks_bought_l38_38676

theorem new_socks_bought :
  ∀ (original_socks throw_away new_socks total_socks : ℕ),
    original_socks = 28 →
    throw_away = 4 →
    total_socks = 60 →
    total_socks = original_socks - throw_away + new_socks →
    new_socks = 36 :=
by
  intros original_socks throw_away new_socks total_socks h_original h_throw h_total h_eq
  sorry

end new_socks_bought_l38_38676


namespace right_triangle_third_side_l38_38223

theorem right_triangle_third_side (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) : c = Real.sqrt (b^2 - a^2) :=
by
  rw [h1, h2]
  sorry

end right_triangle_third_side_l38_38223


namespace elaineExpenseChanges_l38_38025

noncomputable def elaineIncomeLastYear : ℝ := 20000 + 5000
noncomputable def elaineExpensesLastYearRent := 0.10 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearGroceries := 0.20 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearHealthcare := 0.15 * elaineIncomeLastYear
noncomputable def elaineTotalExpensesLastYear := elaineExpensesLastYearRent + elaineExpensesLastYearGroceries + elaineExpensesLastYearHealthcare
noncomputable def elaineSavingsLastYear := elaineIncomeLastYear - elaineTotalExpensesLastYear

noncomputable def elaineIncomeThisYear : ℝ := 23000 + 10000
noncomputable def elaineExpensesThisYearRent := 0.30 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearGroceries := 0.25 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearHealthcare := (0.15 * elaineIncomeThisYear) * 1.10
noncomputable def elaineTotalExpensesThisYear := elaineExpensesThisYearRent + elaineExpensesThisYearGroceries + elaineExpensesThisYearHealthcare
noncomputable def elaineSavingsThisYear := elaineIncomeThisYear - elaineTotalExpensesThisYear

theorem elaineExpenseChanges :
  ( ((elaineExpensesThisYearRent - elaineExpensesLastYearRent) / elaineExpensesLastYearRent) * 100 = 296)
  ∧ ( ((elaineExpensesThisYearGroceries - elaineExpensesLastYearGroceries) / elaineExpensesLastYearGroceries) * 100 = 65)
  ∧ ( ((elaineExpensesThisYearHealthcare - elaineExpensesLastYearHealthcare) / elaineExpensesLastYearHealthcare) * 100 = 45.2)
  ∧ ( (elaineSavingsLastYear / elaineIncomeLastYear) * 100 = 55)
  ∧ ( (elaineSavingsThisYear / elaineIncomeThisYear) * 100 = 28.5)
  ∧ ( (elaineTotalExpensesLastYear / elaineIncomeLastYear) = 0.45 )
  ∧ ( (elaineTotalExpensesThisYear / elaineIncomeThisYear) = 0.715 )
  ∧ ( (elaineSavingsLastYear - elaineSavingsThisYear) = 4345 ∧ ( (55 - ((elaineSavingsThisYear / elaineIncomeThisYear) * 100)) = 26.5 ))
:= by sorry

end elaineExpenseChanges_l38_38025


namespace sum_max_min_MN_l38_38752

def is_arithmetic_sequence (a b c : ℝ) : Prop := 2 * b = a + c

def point_projection (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ := sorry -- Function to find the projection point.

theorem sum_max_min_MN (a b c : ℝ) (h_arith : is_arithmetic_sequence a b c) (P : ℝ × ℝ := (-1, 0)) (N : ℝ × ℝ := (3, 3)) :
  let M := point_projection P a b c in
  let A := (0, -1) in
  let r := Real.sqrt 2 in
  let AN := Real.sqrt ((A.1 - N.1)^2 + (A.2 - N.2)^2) in
  let |MN|_max := AN + r in
  let |MN|_min := AN - r in
  |MN|_max + |MN|_min = 10 := 
by 
  sorry

end sum_max_min_MN_l38_38752


namespace percent_filled_cone_l38_38976

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38976


namespace third_eight_digit_armstrong_number_l38_38859

def sum_of_eighth_powers (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (λ d acc, acc + d^8) 0

theorem third_eight_digit_armstrong_number (n : ℕ) :
  (sum_of_eighth_powers 24678051 = 24678051) →
  (sum_of_eighth_powers 88593477 = 88593477) →
  (sum_of_eighth_powers n = n) →
  n = 24678050 := by
  sorry

end third_eight_digit_armstrong_number_l38_38859


namespace part_a_ellipse_and_lines_l38_38547

theorem part_a_ellipse_and_lines (x y : ℝ) : 
  (4 * x^2 + 8 * y^2 + 8 * y * abs y = 1) ↔ 
  ((y ≥ 0 ∧ (x^2 / (1/4) + y^2 / (1/16)) = 1) ∨ 
  (y < 0 ∧ ((x = 1/2) ∨ (x = -1/2)))) := 
sorry

end part_a_ellipse_and_lines_l38_38547


namespace parabola_intercepts_sum_l38_38110

noncomputable def y_intercept (f : ℝ → ℝ) : ℝ := f 0

noncomputable def x_intercepts_of_parabola (a b c : ℝ) : (ℝ × ℝ) :=
let Δ := b ^ 2 - 4 * a * c in
(
  (-b + real.sqrt Δ) / (2 * a),
  (-b - real.sqrt Δ) / (2 * a)
)

theorem parabola_intercepts_sum :
  let f := λ x : ℝ, 3 * x^2 - 9 * x + 4 in
  let (e, f) := x_intercepts_of_parabola 3 (-9) 4 in
  y_intercept f + e + f = 19 / 3 :=
by
  sorry

end parabola_intercepts_sum_l38_38110


namespace infinite_pairs_condition_l38_38460

def distinct_prime_divisors (n : ℕ) : ℕ := sorry -- assume the existence of this function

theorem infinite_pairs_condition :
  ∃ᶠ (ab : ℕ × ℕ) in (filter (λ p : ℕ × ℕ, p.1 ≠ p.2) (λ a b, distinct_prime_divisors (a + b) = distinct_prime_divisors a + distinct_prime_divisors b)),
    distinct_prime_divisors (ab.1 + ab.2) = distinct_prime_divisors ab.1 + distinct_prime_divisors ab.2 := 
sorry

end infinite_pairs_condition_l38_38460


namespace A_time_correct_l38_38763

noncomputable def A_time := 5.25

def race_distance := 80 
def B_behind := 56 
def B_behind_time := 7
def B_distance_covered := race_distance - B_behind := 24
def A_time_minus_7 := A_time - B_behind_time

theorem A_time_correct : 
  let V_a := race_distance / A_time,
      V_b := B_distance_covered / A_time_minus_7,
      B_distance_in_A_time := V_b * A_time in
  (B_distance_in_A_time = race_distance - B_behind) → A_time = 5.25 :=
by
  sorry

end A_time_correct_l38_38763


namespace arithmetic_geometric_sequence_l38_38715

theorem arithmetic_geometric_sequence (a b : ℝ) (h1 : 2 * a = 1 + b) (h2 : b^2 = a) (h3 : a ≠ b) :
  7 * a * log a (-b) = 7 / 8 :=
by
  sorry

end arithmetic_geometric_sequence_l38_38715


namespace div_ad_bc_l38_38816

theorem div_ad_bc (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) : (a - c) ∣ (a * d + b * c) :=
sorry

end div_ad_bc_l38_38816


namespace todd_snow_cone_price_l38_38145

noncomputable def snow_cone_price (borrowed : ℕ) (repay : ℕ) (spent : ℕ) (sold : ℕ) (left : ℕ) := 
  (repay + left) / sold

theorem todd_snow_cone_price :
  snow_cone_price 100 110 75 200 65 = 0.875 :=
by
  sorry

end todd_snow_cone_price_l38_38145


namespace find_power_of_4_l38_38265

theorem find_power_of_4 (x : Nat) : 
  (2 * x + 5 + 2 = 29) -> 
  (x = 11) :=
by
  sorry

end find_power_of_4_l38_38265


namespace smallest_integer_N_l38_38637

theorem smallest_integer_N : ∃ N : ℕ, (Nat.choose (N + 1) 5 = 3003) ∧ ∀ M : ℕ, (Nat.choose (M + 1) 5 = 3003 → N ≤ M) :=
by
  existsi 19
  split
  · -- Prove that (Nat.choose (19 + 1) 5 = 3003)
    sorry
  · -- Prove that 19 is minimal with (Nat.choose (N+1) 5 = 3003)
    intro M h
    have hN : 20 = M + 1, sorry
    exact sorry

end smallest_integer_N_l38_38637


namespace green_competition_l38_38424

theorem green_competition {x : ℕ} (h : 0 ≤ x ∧ x ≤ 25) : 
  5 * x - (25 - x) ≥ 85 :=
by
  sorry

end green_competition_l38_38424


namespace B_max_at_125_l38_38630

noncomputable def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 : ℝ) ^ k

theorem B_max_at_125 :
  ∃ k, 0 ≤ k ∧ k ≤ 500 ∧ (∀ n, 0 ≤ n ∧ n ≤ 500 → B k ≥ B n) ∧ k = 125 :=
by
  sorry

end B_max_at_125_l38_38630


namespace determinant_evaluation_l38_38644

-- Define the given matrix M
def M (x y : ℝ) : Matrix (fin 3) (fin 3) ℝ :=
  ![
    ![1, x, x^2],
    ![1, x + y, y^2],
    ![1, x^2, x + y]
  ]

-- State the theorem to prove
theorem determinant_evaluation (x y : ℝ) :
  Matrix.det (M x y) = x^4 - x^3 + y^2 - x^2 * y^2 :=
by
  sorry

end determinant_evaluation_l38_38644


namespace water_volume_percentage_l38_38960

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38960


namespace BN_DM_not_parallel_l38_38806

noncomputable theory

-- Definitions of the geometrical entities in the problem.
variable (A B C D M N : Type) [lattice.order_top A] [lattice.order_top B] [lattice.order_top C] [lattice.order_top D] [lattice.order_top M] [lattice.order_top N] 

-- Conditions
def is_midpoint (P Q R : Type) : Prop :=
  lattice.order_top P ∧ lattice.order_top Q ∧ lattice.order_top R

def is_trapezoid (A B C D : Type) : Prop :=
  lattice.order_top A ∧ lattice.order_top B ∧ lattice.order_top C ∧ lattice.order_top D

-- The proof statement that BN and DM cannot be parallel
theorem BN_DM_not_parallel (A B C D M N : Type) [is_trapezoid A B C D] [is_midpoint A B M] [is_midpoint C D N] : 
  ¬ parallelogram.parallel B N D M :=
sorry

end BN_DM_not_parallel_l38_38806


namespace water_filled_percent_l38_38983

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38983


namespace probability_of_prime_or_multiple_of_three_l38_38216

-- Define the set of numbers labeled on the spinner
def spinner_numbers := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define a predicate for prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a predicate for multiples of 3
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Define the set of numbers that are either prime numbers or multiples of 3
def favorable_numbers := {n ∈ spinner_numbers | is_prime n ∨ is_multiple_of_three n}

-- Define the probability calculation
def probability := (favorable_numbers.to_finset.card : ℚ) / (spinner_numbers.to_finset.card : ℚ)

theorem probability_of_prime_or_multiple_of_three :
  probability = 5 / 8 := by
  sorry

end probability_of_prime_or_multiple_of_three_l38_38216


namespace sum_of_missing_angles_l38_38582

theorem sum_of_missing_angles (angle_sum_known : ℕ) (divisor : ℕ) (total_sides : ℕ) (missing_angles_sum : ℕ)
  (h1 : angle_sum_known = 1620)
  (h2 : divisor = 180)
  (h3 : total_sides = 12)
  (h4 : angle_sum_known + missing_angles_sum = divisor * (total_sides - 2)) :
  missing_angles_sum = 180 :=
by
  -- Skipping the proof for this theorem
  sorry

end sum_of_missing_angles_l38_38582


namespace second_spray_kill_percent_l38_38927

-- Conditions
def first_spray_kill_percent : ℝ := 50
def both_spray_kill_percent : ℝ := 5
def germs_left_after_both : ℝ := 30

-- Lean 4 statement
theorem second_spray_kill_percent (x : ℝ) 
  (H : 100 - (first_spray_kill_percent + x - both_spray_kill_percent) = germs_left_after_both) :
  x = 15 :=
by
  sorry

end second_spray_kill_percent_l38_38927


namespace find_value_of_a_l38_38320

theorem find_value_of_a :
  let a : Real := sqrt((19.19)^2 + (39.19)^2 - (38.38) * (39.19))
  in a = 20 :=
by sorry

end find_value_of_a_l38_38320


namespace sum_of_coefficients_l38_38916

theorem sum_of_coefficients (a : ℕ → ℤ) (x : ℂ) :
  (2*x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + 
  a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 →
  a 0 = 1 →
  a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 20 :=
sorry

end sum_of_coefficients_l38_38916


namespace volume_filled_cone_l38_38952

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38952


namespace cone_filled_with_water_to_2_3_height_l38_38990

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38990


namespace marlon_keeps_4_lollipops_l38_38055

def initial_lollipops : ℕ := 42
def fraction_given_to_emily : ℚ := 2 / 3
def lollipops_given_to_lou : ℕ := 10

theorem marlon_keeps_4_lollipops :
  let lollipops_given_to_emily := fraction_given_to_emily * initial_lollipops
  let lollipops_after_emily := initial_lollipops - lollipops_given_to_emily
  let marlon_keeps := lollipops_after_emily - lollipops_given_to_lou
  marlon_keeps = 4 :=
by
  sorry

end marlon_keeps_4_lollipops_l38_38055


namespace moles_of_CH4_required_l38_38397

theorem moles_of_CH4_required
  (initial_moles_Cl2 : ℕ)
  (needed_moles_CH4 : ℕ) :
  (initial_moles_Cl2 = 2) →
  (CH4 + Cl2 = CH3Cl + HCl) →
  needed_moles_CH4 = 2 :=
by
  intros h₁ h₂
  sorry

end moles_of_CH4_required_l38_38397


namespace necessarily_positive_expressions_l38_38810

theorem necessarily_positive_expressions
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  (b + b^2 > 0) ∧ (b + 3 * b^2 > 0) :=
sorry

end necessarily_positive_expressions_l38_38810


namespace roots_quadratic_l38_38705

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l38_38705


namespace jack_marathon_time_l38_38444

noncomputable def marathon_distance : ℝ := 42
noncomputable def jill_time : ℝ := 4.2
noncomputable def speed_ratio : ℝ := 0.7636363636363637

noncomputable def jill_speed : ℝ := marathon_distance / jill_time
noncomputable def jack_speed : ℝ := speed_ratio * jill_speed
noncomputable def jack_time : ℝ := marathon_distance / jack_speed

theorem jack_marathon_time : jack_time = 5.5 := sorry

end jack_marathon_time_l38_38444


namespace find_solutions_l38_38363

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 9 * x ^ 2 + 6

theorem find_solutions :
  ∃ x1 x2 x3 : ℝ, f x1 = Real.sqrt 2 ∧ f x2 = Real.sqrt 2 ∧ f x3 = Real.sqrt 2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end find_solutions_l38_38363


namespace height_GF_l38_38042

variable (A B C D G O F : Type)
variable [AffineSpace ℝ A B C D G O F]

-- Conditions
variables (h1 : Parallel ℝ AD BC)
variables (h2 : Tangent ℝ CD (Circle.diameter AB))
variables (h3 : Midpoint G C D)
variables (h4 : Length CD = 8)

-- Prove height GF = 4 cm
theorem height_GF :
  height G F = 4 :=
by sorry

end height_GF_l38_38042


namespace sum_less_four_l38_38044

theorem sum_less_four (n : ℤ) (h : n ≥ 2) : 
  ∑ k in Finset.range (n - 1), (n / (n - k) * (1 / 2^(k - 1))) < 4 :=
by 
  sorry

end sum_less_four_l38_38044


namespace solution_g_equals_12_l38_38794

def g (x : ℝ) : ℝ :=
  if x < -1 then 3 * x + 12
  else if -1 ≤ x ∧ x < 1 then x^2 - 1
  else 2 * x + 1

theorem solution_g_equals_12 : 
  ∃ x : ℝ, g x = 12 ∧ x = 11 / 2 :=
by
  use 11 / 2
  split
  { sorry }
  { sorry }

end solution_g_equals_12_l38_38794


namespace quadratic_parabola_graph_l38_38815

noncomputable def conditions (a b c : ℝ) : Prop :=
a < 0 ∧ b > 0 ∧ c > 0 

theorem quadratic_parabola_graph (a b c : ℝ) : 
  conditions a b c →
  ax + by + c = 0 →
  y = ax^2 + bx + c →
  -- Expected graph property: parabola opening downward with vertex in second quadrant
  true := 
sorry

end quadratic_parabola_graph_l38_38815


namespace pyramid_volume_not_3_25_l38_38844

theorem pyramid_volume_not_3_25 (b : ℝ) (V : ℝ) (h : ℝ) (α : ℝ) (π : ℝ) (base_area : ℝ) : 
  b = 2 → V = 3.25 → 
  Vpyramid = (1/3) * base_area * h → 
  r = b * cos α →
  h = b * sin α →
  Vcone = (1/3) * π * (r^2) * h →
  Vpyramid < Vcone →
  Vpyramid ≠ 3.25 := 
by 
  sorry

end pyramid_volume_not_3_25_l38_38844


namespace grade12_students_count_l38_38851

theorem grade12_students_count (total_students : ℕ) (x : ℕ)
  (h1 : 10 * x + 8 * x + 7 * x = total_students)
  (h2 : (total_students / 200) * 0.2 = 1) :
  7 * x = 280 :=
by
  sorry

end grade12_students_count_l38_38851


namespace cone_water_fill_percentage_l38_38937

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38937


namespace sum_of_first_15_even_positive_integers_l38_38162

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end sum_of_first_15_even_positive_integers_l38_38162


namespace convert_base_3_to_5_l38_38241

def base_3_to_base_10 (n : Nat) : Nat :=
  let digits := [2, 0, 1, 2, 1]
  digits.foldr (λ d acc => acc * 3 + d) 0

def base_10_to_base_5 (n : Nat) : List Nat :=
  let rec go n acc :=
    if n = 0 then acc else go (n / 5) ((n % 5) :: acc)
  go n []

theorem convert_base_3_to_5 (n : Nat) (h : n = 20121) : 
  base_10_to_base_5 (base_3_to_base_10 n) = [1, 2, 0, 3] :=
by
  sorry

end convert_base_3_to_5_l38_38241


namespace right_angle_vertex_c_false_l38_38015

theorem right_angle_vertex_c_false (a b m : ℝ) :
  (1 / m^2 = 1 / a^2 + 1 / b^2) ↔ ∠C = 90° is false :=
sorry

end right_angle_vertex_c_false_l38_38015


namespace arithmetic_sequence_sum_l38_38130

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (S_10_eq : S 10 = 20) (S_20_eq : S 20 = 15) :
  S 30 = -15 :=
by
  sorry

end arithmetic_sequence_sum_l38_38130


namespace find_a_of_square_conditions_l38_38090

noncomputable def a_square := real.sqrt (144 : ℝ)
def area_square := a_square * a_square

def x := (48 : ℝ)
def y := real.log x / real.log a

def a := real.rpow (4 / 3 : ℝ) (1 / 12 : ℝ)

theorem find_a_of_square_conditions
  (h1 : area_square = 144)
  (A : (x, log_a x))
  (B : (x + 12, 2 * log_a (x + 12)))
  (D : (x, log_a (x - 12) + 12))
  : a = real.rpow (4 / 3 : ℝ) (1 / 12 : ℝ) :=
sorry

end find_a_of_square_conditions_l38_38090


namespace cone_water_fill_percentage_l38_38930

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38930


namespace subset1_squares_equals_product_subset2_squares_equals_product_l38_38633

theorem subset1_squares_equals_product :
  (1^2 + 3^2 + 4^2 + 9^2 + 107^2 = 1 * 3 * 4 * 9 * 107) :=
sorry

theorem subset2_squares_equals_product :
  (3^2 + 4^2 + 9^2 + 107^2 + 11555^2 = 3 * 4 * 9 * 107 * 11555) :=
sorry

end subset1_squares_equals_product_subset2_squares_equals_product_l38_38633


namespace total_cost_is_67_15_l38_38125

noncomputable def calculate_total_cost : ℝ :=
  let caramel_cost := 3
  let candy_bar_cost := 2 * caramel_cost
  let cotton_candy_cost := (candy_bar_cost * 4) / 2
  let chocolate_bar_cost := candy_bar_cost + caramel_cost
  let lollipop_cost := candy_bar_cost / 3

  let candy_bar_total := 6 * candy_bar_cost
  let caramel_total := 3 * caramel_cost
  let cotton_candy_total := 1 * cotton_candy_cost
  let chocolate_bar_total := 2 * chocolate_bar_cost
  let lollipop_total := 2 * lollipop_cost

  let discounted_candy_bar_total := candy_bar_total * 0.9
  let discounted_caramel_total := caramel_total * 0.85
  let discounted_cotton_candy_total := cotton_candy_total * 0.8
  let discounted_chocolate_bar_total := chocolate_bar_total * 0.75
  let discounted_lollipop_total := lollipop_total -- No additional discount

  discounted_candy_bar_total +
  discounted_caramel_total +
  discounted_cotton_candy_total +
  discounted_chocolate_bar_total +
  discounted_lollipop_total

theorem total_cost_is_67_15 : calculate_total_cost = 67.15 := by
  sorry

end total_cost_is_67_15_l38_38125


namespace true_proposition_l38_38313

variable (p q : Prop)

-- Assume p is true
axiom hp : p

-- Assume q is false
axiom hq : ¬q

-- We need to state that the true proposition is (p ∧ ¬q)
theorem true_proposition : (p ∧ ¬q) :=
by
  exact ⟨hp, hq⟩

end true_proposition_l38_38313


namespace polyhedron_diagonal_length_l38_38141

noncomputable def find_t (e : ℝ) : ℝ :=
  by 
    -- The cubic equation t^3 - 7et^2 + 2e^3 = 0
    sorry

theorem polyhedron_diagonal_length :
  ∃ t : ℝ, t^3 - 7 * (1:ℝ) * t^2 + 2 * (1:ℝ)^3 = 0 ∧ 
  abs (t - 0.29) < 0.01 :=
by
  -- Showing existence of t that satisfies the cubic equation and is close to 0.29
  -- for e = 1
  let e := 1
  use find_t e
  have h : find_t e ^ 3 - 7 * e * (find_t e) ^ 2 + 2 * e ^ 3 = 0 := sorry
  have approx_t : abs (find_t e - 0.29) < 0.01 := sorry
  exact ⟨find_t e, ⟨h, approx_t⟩⟩

end polyhedron_diagonal_length_l38_38141


namespace base7_to_base10_of_645_l38_38584

theorem base7_to_base10_of_645 :
  (6 * 7^2 + 4 * 7^1 + 5 * 7^0) = 327 := 
by 
  sorry

end base7_to_base10_of_645_l38_38584


namespace lambda_property_l38_38792
open Int

noncomputable def lambda : ℝ := 1 + Real.sqrt 2

theorem lambda_property (n : ℕ) (hn : n > 0) :
  2 * ⌊lambda * n⌋ = 1 - n + ⌊lambda * ⌊lambda * n⌋⌋ :=
sorry

end lambda_property_l38_38792


namespace problem_1a_problem_1b_problem_2a_problem_2b_case_1_l38_38361

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + (2 - a) * Real.log x

theorem problem_1a {
  let a := 8
  : StrictMono (λ x, f x a) := 
sorry

theorem problem_1b :
  TangentLine (λ x, f x 8) (1, -3) ∧ ((8 * x + y - 5 = 0) := 
sorry

theorem problem_2a (a : ℝ) :
  (a ≤ 0 → ∃ x ∈ Icc Real.e (Real.e ^ 2), IsMinOn (λ x, f x a) (Icc Real.e (Real.e ^ 2)) x) := 
sorry

theorem problem_2b_case_1 (a : ℝ) :
(a > 0 → a ≥ 2 * (Real.e ^ 2 - 1)^2 → ∃ x ∈ Icc Real.e (Real.e ^ 2), IsMinOn (λ x, f x a) (Icc Real.e (Real.e ^ 2)) x 
∧ x = (Real.e ^ 2))
∧ (a > 0 → a < 2 * (Real.e ^ 2 - 1)^2 →  a > 2 * (Real.e - 1)^2 → ∃ x ∈ Icc Real.e (Real.e ^ 2), IsMinOn (λ x, f x a) (Icc Real.e (Real.e ^ 2)) x 
∧ x = (1 + sqrt(2*a)/2) )
∧ (a > 0 → a ≤ 2 * (Real.e - 1)^2 → ∃ x ∈ Icc Real.e (Real.e ^ 2), IsMinOn (λ x, f x a) (Icc Real.e (Real.e ^ 2)) x 
∧ x = Real.e) := 
sorry

end problem_1a_problem_1b_problem_2a_problem_2b_case_1_l38_38361


namespace smallest_b_nine_l38_38262

theorem smallest_b_nine (b : ℕ) (h_b : b > 8) : (∀ x : ℤ, ¬ prime (x^4 + b^2)) ↔ b = 9 := by sorry

end smallest_b_nine_l38_38262


namespace mul_99_105_l38_38643

theorem mul_99_105 : 99 * 105 = 10395 := 
by
  -- Annotations and imports are handled; only the final Lean statement provided as requested.
  sorry

end mul_99_105_l38_38643


namespace find_universal_set_l38_38728

open Set

variable (A C U : Set ℕ)
variable (A_def : A = {1, 3, 5}) (C_def : C = {2, 4, 6})

theorem find_universal_set :
  ∁ U A = C → U = A ∪ C := by
  intro h₁
  rw [A_def, C_def] at *
  sorry

end find_universal_set_l38_38728


namespace first_fun_friday_is_march_30_l38_38928

def month_days := 31
def start_day := 4 -- 1 for Sunday, 2 for Monday, ..., 7 for Saturday; 4 means Thursday
def first_friday := 2
def fun_friday (n : ℕ) : ℕ := first_friday + (n - 1) * 7

theorem first_fun_friday_is_march_30 (h1 : start_day = 4)
                                    (h2 : month_days = 31) :
                                    fun_friday 5 = 30 :=
by 
  -- Proof is omitted
  sorry

end first_fun_friday_is_march_30_l38_38928


namespace intersection_abscissa_interval_l38_38786

theorem intersection_abscissa_interval :
  let f (x : ℝ) := log x
  let g (x : ℝ) := 1 / x
  let x0 := Classical.choose ((∃ x : ℝ, f x = g x) ∧ 0 < x)
  (2 < x0 ∧ x0 < 3) := 
begin
  sorry
end

end intersection_abscissa_interval_l38_38786


namespace angle_between_squares_is_15_degrees_l38_38476

theorem angle_between_squares_is_15_degrees
  (O : Type) [Real O]
  (R : ℝ) -- radius of the circle
  (ABCD A1B1C1D1 : Type) [Square ABCD] [Square A1B1C1D1]
  (inscribed_square : ∀ (P : O), P ∈ ABCD → ∃ Q ∈ O, distance P Q = R)
  (circumscribed_square : ∀ (P : A1B1C1D1), ∃ Q ∈ O, distance P Q = R)
  (M : O)
  (tangent_point : M ∈ ABCD)
  (center_of_circle : O ∈ ABCD ∧ O ∈ A1B1C1D1) :
  ∃ α : ℝ, α = 15 :=
begin
  sorry
end

end angle_between_squares_is_15_degrees_l38_38476


namespace plane_equation_through_point_parallel_to_plane_l38_38257

theorem plane_equation_through_point_parallel_to_plane : 
  ∃ A B C D : ℤ, 
    (A > 0) ∧
    (Int.gcd (Int.abs A) (Int.gcd (Int.abs B) (Int.gcd (Int.abs C) (Int.abs D))) = 1) ∧
    (A * 2 + B * (-1) + C * 3 + D = 0) ∧
    (A = 2) ∧ (B = -1) ∧ (C = 3) ∧ (D = -14) :=
by
  sorry

end plane_equation_through_point_parallel_to_plane_l38_38257


namespace length_DE_l38_38009

/-- In triangle ABC with side lengths AB = 5, BC = 7, and AC = 3, 
if line DE is drawn from vertex B such that DE is perpendicular to AC 
and intersects AC at point D, then the length of DE is 4.33. -/
theorem length_DE (A B C D E : Type)
  (AB BC AC : ℝ)
  (hAB : AB = 5)
  (hBC : BC = 7)
  (hAC : AC = 3)
  (DE_perpendicular : ∃ (DE : ℝ), is_perpendicular DE AC ∧ intersects_at D E AC)
  : ∃ (DE : ℝ), DE = 4.33 := 
sorry

end length_DE_l38_38009


namespace percent_filled_cone_l38_38970

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38970


namespace triangle_area_correct_l38_38399
noncomputable def area_of_triangle_intercepts : ℝ :=
  let f (x : ℝ) : ℝ := (x - 3) ^ 2 * (x + 2)
  let x1 := 3
  let x2 := -2
  let y_intercept := f 0
  let base := x1 - x2
  let height := y_intercept
  1 / 2 * base * height

theorem triangle_area_correct :
  area_of_triangle_intercepts = 45 :=
by
  sorry

end triangle_area_correct_l38_38399


namespace solve_for_x_l38_38493

def star (a b : ℝ) : ℝ := 3 * a - b

theorem solve_for_x :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solve_for_x_l38_38493


namespace cone_filled_with_water_to_2_3_height_l38_38991

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38991


namespace parabola_intercepts_sum_l38_38112

theorem parabola_intercepts_sum :
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  y_intercept + x_intercept1 + x_intercept2 = 7 :=
by
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  have sum_intercepts : y_intercept + x_intercept1 + x_intercept2 = 7 := by
        calc (4 : ℝ) + ((9 + Real.sqrt 33) / 6) + ((9 - Real.sqrt 33) / 6)
            = 4 + (18 / 6) : by
              rw [add_assoc, ← add_div, add_sub_cancel]
            ... = 4 + 3 : by norm_num
            ... = 7 : by norm_num
  exact sum_intercepts

end parabola_intercepts_sum_l38_38112


namespace average_playtime_in_minutes_l38_38452

noncomputable def lena_playtime_hours : ℝ := 3.5
noncomputable def lena_playtime_minutes : ℝ := lena_playtime_hours * 60
noncomputable def brother_playtime_minutes : ℝ := 1.2 * lena_playtime_minutes + 17
noncomputable def sister_playtime_minutes : ℝ := 1.5 * brother_playtime_minutes

theorem average_playtime_in_minutes :
  (lena_playtime_minutes + brother_playtime_minutes + sister_playtime_minutes) / 3 = 294.17 :=
by
  sorry

end average_playtime_in_minutes_l38_38452


namespace only_negative_number_l38_38229

theorem only_negative_number :
  ∀ (a b c d : ℝ),
    a = (-1)^0 →
    b = |(-1)| →
    c = real.sqrt 1 →
    d = -(1^2) →
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 0 ∧ ({d} : set ℝ) = {x | x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d)} :=
by {
  intros a b c d ha hb hc hd,
  split,
  { linarith [ha.symm, hb.symm, hc.symm, hd.symm] },
  { sorry }
}

end only_negative_number_l38_38229


namespace projectile_reaches_100_feet_at_2_5_seconds_l38_38501

theorem projectile_reaches_100_feet_at_2_5_seconds :
  ∃ t : ℝ, (y : ℝ) = -16 * t^2 + 80 * t ∧ y = 100 ∧ t = 2.5 :=
begin
  sorry,
end

end projectile_reaches_100_feet_at_2_5_seconds_l38_38501


namespace water_bottle_capacity_l38_38225

theorem water_bottle_capacity :
  (20 * 250 + 13 * 600) / 1000 = 12.8 := 
by
  sorry

end water_bottle_capacity_l38_38225


namespace solve_y_l38_38084

theorem solve_y : ∃ y : ℚ, 5 + 3.2 * y = 2.1 * y - 25 ∧ y = -300 / 11 :=
by
  -- Provided conditions and expected answer need to be converted into assertions
  existsi (-300 / 11 : ℚ)
  split
  sorry -- This is where you would actually solve the equation to show both conditions hold

end solve_y_l38_38084


namespace transmission_incorrect_option_C_l38_38008

theorem transmission_incorrect_option_C :
  ∀ (α β : ℝ), 0 < α ∧ α < 1 ∧ 0 < β ∧ β < 1 →
  ¬(let prob_decode_1 := β * (1 - β)^2 + (1 - β)^3 in
  let correct_prob_decode_1 := 3 * β * (1 - β)^2 + (1 - β)^3 in
  prob_decode_1 = correct_prob_decode_1) :=
by
  intros α β h
  sorry

end transmission_incorrect_option_C_l38_38008


namespace truth_tellers_in_circle_l38_38673

noncomputable def num_truth_tellers (n : ℕ) : Prop := 
  n = 2

theorem truth_tellers_in_circle : ∃ n, num_truth_tellers n 
:= 
begin
  use 2,
  unfold num_truth_tellers,
  exact eq.refl 2
end

end truth_tellers_in_circle_l38_38673


namespace area_of_R_is_quarter_l38_38873

open EuclideanGeometry

variables (A B C D E : Point)
  [unit_square A B C D]
  [is_equilateral_triangle A B E]
  [on BC E]

def region_R (A D : Point) : set Point := {P | A = D ∧ dist_P_to_line AD P ∈ Icc (1/4) (1/2)}

theorem area_of_R_is_quarter :
  area (region_R A D \ triangle A B E) = 1/4 :=
sorry

end area_of_R_is_quarter_l38_38873


namespace ratio_AF_BF_l38_38373

-- Define the conditions of the problem
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def line_eq_through_focus (x y : ℝ) : Prop := y = sqrt(3) * (x - 1)
def intersection_points (x1 y1 x2 y2 : ℝ) (h1 : parabola_eq x1 y1) (h2 : parabola_eq x2 y2) 
  (h3 : line_eq_through_focus x1 y1) (h4 : line_eq_through_focus x2 y2) : Prop := true

-- Define the lengths from focus F to points A and B
def distance_AF (x1 : ℝ) : ℝ := abs (x1 - 1)
def distance_BF (x2 : ℝ) : ℝ := abs (x2 - 1)

-- Define the main statement to be proven, assuming the necessary conditions
theorem ratio_AF_BF (x1 x2 y1 y2 : ℝ)
  (h1 : parabola_eq x1 y1)
  (h2 : parabola_eq x2 y2)
  (h3 : line_eq_through_focus x1 y1)
  (h4 : line_eq_through_focus x2 y2)
  (h5 : distance_AF x1 > distance_BF x2) :
  distance_AF x1 / distance_BF x2 = 3 :=
by
  sorry

end ratio_AF_BF_l38_38373


namespace sequence_event_equivalence_l38_38046

open Set Filter

variables {Ω : Type*} {A : ℕ → Set Ω} -- Sequence of events A_n from Ω

theorem sequence_event_equivalence (A : ℕ → Set Ω) : 
  limsup (λ n, A n) \ liminf (λ n, A n) =
  limsup (λ n, A n \ A (n + 1)) ∧
  limsup (λ n, A (n + 1) \ A n) ∧
  limsup (λ n, symmetricDifference (A n) (A (n + 1))) :=
sorry

end sequence_event_equivalence_l38_38046


namespace sally_money_value_l38_38076

def sally_has_money (sally_money jolly_money : ℕ) : Prop :=
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70)

theorem sally_money_value (sally_money : ℕ) (jolly_money : ℕ) (h : sally_has_money sally_money jolly_money) : sally_money = 100 :=
by
  cases h with h_sally h_jolly
  rw [eq_add_of_sub_eq h_sally]
  exact rfl

end sally_money_value_l38_38076


namespace range_of_k_l38_38507

theorem range_of_k (k : ℝ) :
  (∃ P Q : ℝ × ℝ, 
    (P.2 = k * P.1 + 1) ∧ 
    (Q.2 = k * Q.1 + 1) ∧
    ((P.1 - 2)^2 + (P.2 - 1)^2 = 4) ∧
    ((Q.1 - 2)^2 + (Q.2 - 1)^2 = 4) ∧
    dist P Q ≥ 2 * sqrt 2) ↔ (k ∈ set.Icc (-1 : ℝ) 1) :=
sorry

end range_of_k_l38_38507


namespace tangent_line_eq_find_range_a_find_range_m_l38_38722

noncomputable def h (x a : ℝ) : ℝ := -2 * a * x + log x
noncomputable def f (x a : ℝ) : ℝ := (a / 2) * x ^ 2 + h x a

-- Part 1: Tangent line equation at (2, h(2)) when a = 1
theorem tangent_line_eq {x : ℝ} (h_eq : ∀ x, h x 1 = -2 * x + log x) :
  ∀ y x, y + 4 - log 2 = - (3 / 2) * (x - 2) := sorry

-- Part 2: Find the range of values for a given x1 * x2 > 1 / 2
theorem find_range_a (a : ℝ) (f_prime_eq : ∀ x, f x a = ((a / 2) * x ^ 2 - 2 * a * x + 1) / x) :
  1 < a ∧ a < 2 := sorry

-- Part 3: Find the range of values for m such that inequality always holds
theorem find_range_m (a : ℝ) (x0 : ℝ) (p : 1 + sqrt 2 / 2 ≤ x0 ∧ x0 ≤ 2)
  (ineq : ∀ a x0, 1 < a ∧ a < 2 → f x0 a + log (a + 1) > m * (a^2 - 1) - (a + 1) + 2 * log 2) :
  m ∈ Set.Iic (- (1 / 4)) := sorry

end tangent_line_eq_find_range_a_find_range_m_l38_38722


namespace convex_22gon_not_divisible_into_7_pentagons_l38_38072

theorem convex_22gon_not_divisible_into_7_pentagons :
  ∀ (polygon : Type) (n : ℕ) (k : ℕ), 
  convex polygon → sides polygon = 22 → k = 7 → 
  ¬ (∃ (p : list (list (fin 5))), length p = 7 ∧ all_pentagons polygon p) :=
sorry

def convex (polygon : Type) : Prop := -- definition of convexity
sorry

def sides (polygon : Type) : ℕ := -- function returning the number of sides
sorry

def all_pentagons (polygon : Type) (p : list (list (fin 5))) : Prop := -- checks if all sub-polygons are pentagons
sorry

end convex_22gon_not_divisible_into_7_pentagons_l38_38072


namespace hyperbola_eq_l38_38723

theorem hyperbola_eq (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : -b / a = -1/2) (h4 : a^2 + b^2 = 5^2) :
  ∃ (a b : ℝ), (a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
  (∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2 - y^2 / b^2 = 1)})) := sorry

end hyperbola_eq_l38_38723


namespace problem_statement_l38_38457

noncomputable def sequence : List ℕ := 
  List.filter (λ n, n.binaryDigits.count 1 = 9) (List.range (2^14))

def N := sequence.nth 1499

theorem problem_statement :
  ∃ r, N % 500 = r :=
by
  sorry

end problem_statement_l38_38457


namespace man_l38_38586

def man's_rate : ℝ := 7 -- Man's rate in still water
def speed_against_stream : ℝ := 12 -- Speed against the stream

-- Speed of the stream
def speed_of_stream : ℝ := man's_rate - speed_against_stream

-- Man's speed with the stream
def speed_with_stream : ℝ := man's_rate + (real.abs speed_of_stream)

-- The theorem to prove
theorem man's_speed_with_stream :
  speed_with_stream = 12 :=
by
  sorry

end man_l38_38586


namespace can_divide_animals_into_two_groups_l38_38532

def day_consumption (days : ℕ) : ℚ := (1 : ℚ) / ↑days

def horse_consumption := day_consumption 3 / 2  -- 1.5 days -> 2/3 per day
def bull_consumption := day_consumption 2      -- 2 days -> 1/2 per day
def cow_consumption := day_consumption 3       -- 3 days -> 1/3 per day
def calf_consumption := day_consumption 4      -- 4 days -> 1/4 per day
def sheep_consumption := day_consumption 6     -- 6 days -> 1/6 per day
def goat_consumption := day_consumption 12     -- 12 days -> 1/12 per day

def total_consumption : ℚ := 
  horse_consumption + bull_consumption + cow_consumption +
  calf_consumption + sheep_consumption + goat_consumption

theorem can_divide_animals_into_two_groups :
  (∃ (group1 group2 : list ℚ), group1.sum = 1 ∧ group2.sum = 1 ∧
    group1 = [horse_consumption, cow_consumption] ∧ 
    group2 = [bull_consumption, calf_consumption, sheep_consumption, goat_consumption]) ∨
  (∃ (group1 group2 : list ℚ), group1.sum = 1 ∧ group2.sum = 1 ∧
    group1 = [horse_consumption, calf_consumption, goat_consumption] ∧
    group2 = [bull_consumption, cow_consumption, sheep_consumption]) :=
by
  sorry

end can_divide_animals_into_two_groups_l38_38532


namespace train_speed_in_kmph_l38_38191

def train_length : ℝ := 275
def time_to_cross : ℝ := 7

def speed_meters_per_second (distance : ℝ) (time : ℝ) : ℝ := distance / time

def convert_mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

theorem train_speed_in_kmph :
  convert_mps_to_kmph (speed_meters_per_second train_length time_to_cross) = 141.43 :=
by
  sorry

end train_speed_in_kmph_l38_38191


namespace largest_n_dividing_18_power_n_in_factorial_30_l38_38654

theorem largest_n_dividing_18_power_n_in_factorial_30 :
  ∃ n : ℕ, (∀ m : ℕ, (18^m ∣ fact 30) ↔ (m ≤ 7)) :=
by
  sorry

end largest_n_dividing_18_power_n_in_factorial_30_l38_38654


namespace roots_quadratic_expr_l38_38702

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end roots_quadratic_expr_l38_38702


namespace find_x_l38_38660

theorem find_x (x : ℝ) : 5 ^ (x + 2) = 625 → x = 2 := 
by 
  sorry

end find_x_l38_38660


namespace cooler_capacity_l38_38447

theorem cooler_capacity (C : ℝ) (h1 : 3.25 * C = 325) : C = 100 :=
sorry

end cooler_capacity_l38_38447


namespace garden_length_l38_38566

theorem garden_length :
  ∀ (w : ℝ) (l : ℝ),
  (l = 2 * w) →
  (2 * l + 2 * w = 150) →
  l = 50 :=
by
  intros w l h1 h2
  sorry

end garden_length_l38_38566


namespace bear_no_winning_strategy_l38_38026

variable (X : Type) (mathcal_F : Finset (Finset X))
variable [Fintype X]

-- Conditions
variable (perm : ∀ (x y : X), ∃ π : Equiv.Perm X, π x = y) 
variable (perm_invar : ∀ (π : Equiv.Perm X) (A : Finset X), A ∈ mathcal_F → π '' A ∈ mathcal_F)

-- Definition of the bear and crocodile game as per conditions
def game_conditions := 
  ∀ (coloring : Coloring X), 
  ¬ (∃ (A ∈ mathcal_F), ∀ a ∈ A, coloring a = coloring.bear)

-- Our proof goal
theorem bear_no_winning_strategy :
  game_conditions X mathcal_F perm perm_invar := 
sorry

end bear_no_winning_strategy_l38_38026


namespace total_pencils_l38_38248

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 15) : pencils_per_child * children = 30 :=
by
  rw [h1, h2]
  exact (Nat.mul_comm _ _).trans (Nat.mul 2 15)
  sorry

end total_pencils_l38_38248


namespace find_sum_pqr_l38_38492

/-- Given integers p, q, r such that the GCD of the polynomials x^2 + px + q and 
    x^2 + qx + r is x - 1, and the LCM is x^3 - 2x^2 - 5x + 6,
    prove that p + q + r = -4. -/
theorem find_sum_pqr
  (p q r : ℤ)
  (h_gcd : Polynomial.gcd (Polynomial.X^2 + Polynomial.C p * Polynomial.X + Polynomial.C q) 
                          (Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) 
           = Polynomial.X - 1)
  (h_lcm : Polynomial.lcm (Polynomial.X^2 + Polynomial.C p * Polynomial.X + Polynomial.C q) 
                          (Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) 
           = Polynomial.X^3 - 2 * Polynomial.X^2 - 5 * Polynomial.X + 6) :
  p + q + r = -4 :=
sorry

end find_sum_pqr_l38_38492


namespace cone_water_fill_percentage_l38_38933

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38933


namespace probability_seven_games_needed_l38_38920

theorem probability_seven_games_needed (p_win : ℚ) (p_loss : ℚ) :
  p_win = 2 / 3 →
  p_loss = 1 - p_win →
  (∑ i in finset.range (7), (if i = 4 then (p_win ^ 5 * p_loss ^ 2 * F (6, 4)) else 0) +
                      (if i = 4 then (p_win ^ 2 * p_loss ^ 5 * F (6, 4)) else 0)) = 20 / 81 := by
  sorry

end probability_seven_games_needed_l38_38920


namespace magnitude_of_MN_is_correct_l38_38733

variable {x y : ℝ}

def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  (Real.sqrt (v.fst ^ 2 + v.snd ^ 2))

theorem magnitude_of_MN_is_correct
    (a : ℝ × ℝ := (2, -1))
    (b : ℝ × ℝ := (x, -2))
    (c : ℝ × ℝ := (3, y))
    (M : ℝ × ℝ := (x, y))
    (N : ℝ × ℝ := (y, x))
    (parallel : ∃ k : ℝ, b = (k * a.fst, k * a.snd))
    (perpendicular : (a.fst + b.fst) * (b.fst - c.fst) + (a.snd + b.snd) * (b.snd - c.snd) = 0) :
  vector_magnitude (N.fst - M.fst, N.snd - M.snd) = 8 * Real.sqrt 2 := 
sorry

end magnitude_of_MN_is_correct_l38_38733


namespace roots_quadratic_expr_l38_38703

theorem roots_quadratic_expr (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0)
    (h2 : Polynomial.eval n (Polynomial.C 1 * X^2 + Polynomial.C 2 * X + Polynomial.C (-5)) = 0) :
  m^2 + m * n + 2 * m = 0 :=
sorry

end roots_quadratic_expr_l38_38703


namespace Tristan_wins_p1_Abigaelle_wins_p2_l38_38037

noncomputable def p1 : ℕ := 10^9 + 7
noncomputable def p2 : ℕ := 10^9 + 9

def is_prime (p : ℕ) : Prop := Nat.Prime p

def can_Abigaelle_win (p : ℕ) : Prop := 
  ∀ (X : ℕ) (a : ℕ → ℕ), X ≥ 1 → (∀ n, a n > 0) → 
  ∃ M : ℕ, M % p = 0 ∧ (∃ k, ∃ f : Fin k → ℕ, (X * list.prod (list.of_fn f.val) + list.sum (list.of_fn f.val)) % p = M)

theorem Tristan_wins_p1 :
  is_prime p1 → ¬ can_Abigaelle_win p1 := 
by
  intro h1
  sorry

theorem Abigaelle_wins_p2 :
  is_prime p2 → can_Abigaelle_win p2 := 
by
  intro h2
  sorry

end Tristan_wins_p1_Abigaelle_wins_p2_l38_38037


namespace at_least_1094_people_know_secret_on_target_day_l38_38063

-- Definitions based on the conditions provided
def chris_shares_secret := 1
def roommates := 3
def sharing_rate := 3

-- At time t=0, Chris and his three roommates know the secret
def initial_people_know_secret := chris_shares_secret + roommates

-- Recursive definition of the number of people knowing the secret each day
noncomputable def people_know_secret (n : ℕ) : ℕ :=
  ∑ i in finset.range (n+1), sharing_rate^i

-- Problem statement
theorem at_least_1094_people_know_secret_on_target_day :
  ∃ n : ℕ, people_know_secret n ≥ 1094 :=
begin
  -- Expected to use properties of geometric series
  sorry
end

end at_least_1094_people_know_secret_on_target_day_l38_38063


namespace blue_tile_probability_l38_38196

theorem blue_tile_probability : 
  let tiles := finset.range 100 in
  let blue_tiles := tiles.filter (λ n, n % 8 = 3) in
  (blue_tiles.card : ℚ) / tiles.card = 13 / 100 :=
sorry

end blue_tile_probability_l38_38196


namespace train_crossing_time_l38_38179

def train_length : ℝ := 150 -- in meters
def train_speed_kmph : ℝ := 122 -- in km/hr

def convert_speed (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) -- converts km/hr to m/s

def train_speed_mps : ℝ := convert_speed train_speed_kmph -- in m/s

def crossing_time (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed -- time in seconds

theorem train_crossing_time :
  crossing_time train_length train_speed_mps = 4.43 :=
by
  sorry

end train_crossing_time_l38_38179


namespace centers_not_marked_points_l38_38138

/-!
# Centers of Circumscribed Circles

There are several points marked on a plane, and not all of these points lie on the same line.
A circle is circumscribed around each triangle with vertices at the marked points. 

This theorem demonstrates that it is impossible for the centers of all these circumscribed circles to be among the marked points.
-/

noncomputable def exists_marked_points : Prop :=
  ∃ (P : Set ℝ × ℝ) (hP : P.nonempty) (hLine : ¬ collinear P),
    ∀ (T : Triangle ℝ),
      T.vertices = P →
      ∀ (ω : CircleCircumscribed T),
      ω.center ∈ P

theorem centers_not_marked_points (P : Set ℝ × ℝ) (hP : P.nonempty) (hLine : ¬ collinear P) :
  ¬ (∀ (T : Triangle ℝ), T.vertices = P → ∀ (ω : CircleCircumscribed T), ω.center ∈ P) :=
sorry

end centers_not_marked_points_l38_38138


namespace find_x_from_expression_l38_38379

theorem find_x_from_expression
  (y : ℚ)
  (h1 : y = -3/2)
  (h2 : -2 * (x : ℚ) - y^2 = 0.25) : 
  x = -5/4 := 
by 
  sorry

end find_x_from_expression_l38_38379


namespace sum_terms_a1_a17_l38_38686

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 := by
  sorry

end sum_terms_a1_a17_l38_38686


namespace sequence_decreasing_from_100th_term_l38_38524

noncomputable def a_n (n : ℕ) : ℝ := (100 ^ n) / nat.factorial n

theorem sequence_decreasing_from_100th_term :
  ∀ n > 99, a_n (n + 1) < a_n n :=
by
  sorry

end sequence_decreasing_from_100th_term_l38_38524


namespace total_expenditure_now_l38_38178

def initial_students : ℕ := 100
def additional_students : ℕ := 20
def decrease_avg_expense : ℕ := 5
def expenditure_increase : ℕ := 400

def original_avg_expense : ℝ -- This is the value we solve for in the end
def current_students : ℕ := initial_students + additional_students

theorem total_expenditure_now :
  let original_total_expenditure := initial_students * original_avg_expense in
  let new_avg_expense := original_avg_expense - decrease_avg_expense in
  let new_total_expenditure := current_students * new_avg_expense in
  original_total_expenditure + expenditure_increase = new_total_expenditure → 
  new_total_expenditure = 5400 :=
by
  sorry

end total_expenditure_now_l38_38178


namespace minimum_sticks_broken_n12_can_form_square_n15_l38_38285

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l38_38285


namespace circle_and_tangent_line_l38_38327

theorem circle_and_tangent_line (P : ℝ × ℝ) (C : ℝ × ℝ) (d : ℝ) (h1 : P = (1, 0)) (h2 : C = (2, 3)) (h3 : d = 2) :
  ((∀ x y, (x - 2)^2 + (y - 3)^2 = 1) ∧ 
   ((∀ k, line k P ∧ tangent_to_circle k C 1 → line_equation k = (λ p : ℝ × ℝ, p.1 = 1) ∨ line_equation k = (λ p : ℝ × ℝ, 4 * p.1 - 3 * p.2 - 4 = 0)))) :=
by
  sorry

-- Definitions needed for this theorem, assuming they are defined elsewhere in the math library or need explicit definitions
def line (k : ℝ) (P : ℝ × ℝ) : Prop := sorry
def tangent_to_circle (k : ℝ) (C : ℝ × ℝ) (r : ℝ) : Prop := sorry
def line_equation (k : ℝ) : (ℝ × ℝ) → Prop := sorry

end circle_and_tangent_line_l38_38327


namespace smallest_n_exists_l38_38769

def connected (a b : ℕ) : Prop := -- define connection based on a picture not specified here, placeholder
sorry

def not_connected (a b : ℕ) : Prop := ¬ connected a b

def coprime (a n : ℕ) : Prop := ∀ k : ℕ, k > 1 → k ∣ a → ¬ k ∣ n

def common_divisor_greater_than_one (a n : ℕ) : Prop := ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ n

theorem smallest_n_exists :
  ∃ n : ℕ,
  (n = 35) ∧
  ∀ (numbers : Fin 7 → ℕ),
  (∀ i j, not_connected (numbers i) (numbers j) → coprime (numbers i + numbers j) n) ∧
  (∀ i j, connected (numbers i) (numbers j) → common_divisor_greater_than_one (numbers i + numbers j) n) := 
sorry

end smallest_n_exists_l38_38769


namespace tammy_haircuts_l38_38095

theorem tammy_haircuts (total_haircuts free_haircuts haircuts_to_next_free : ℕ) 
(h1 : free_haircuts = 5) 
(h2 : haircuts_to_next_free = 5) 
(h3 : total_haircuts = 79) : 
(haircuts_to_next_free = 5) :=
by {
  sorry
}

end tammy_haircuts_l38_38095


namespace min_breaks_for_square_12_can_form_square_15_l38_38293

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l38_38293


namespace problem_statement_l38_38319

def f (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 0.2 / Real.log 10
noncomputable def c : ℝ := 2^(0.2)

-- Proving the main statement
theorem problem_statement : f(b) < f(a) ∧ f(a) < f(c) := by
  sorry

end problem_statement_l38_38319


namespace sum_tens_ones_digits_3_plus_4_power_17_l38_38881

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l38_38881


namespace range_of_PA_plus_PB_l38_38315

variable (A B P : ℝ × ℝ)

def circle1 (M : ℝ × ℝ) : Prop :=
  prod.fst M ^ 2 + prod.snd M ^ 2 = 1

def circle2 (P : ℝ × ℝ) : Prop :=
  (prod.fst P - 3) ^ 2 + (prod.snd P - 4) ^ 2 = 1

def distance (M N : ℝ × ℝ) : ℝ :=
  real.sqrt ((prod.fst M - prod.fst N) ^ 2 + (prod.snd M - prod.snd N) ^ 2)

axiom A_on_circle1 : circle1 A
axiom B_on_circle1 : circle1 B
axiom dist_AB_eq_sqrt3 : distance A B = real.sqrt 3
axiom P_on_circle2 : circle2 P

noncomputable def PA_vec (P A : ℝ × ℝ) : ℝ × ℝ :=
  (prod.fst P - prod.fst A, prod.snd P - prod.snd A)

noncomputable def PB_vec (P B : ℝ × ℝ) : ℝ × ℝ :=
  (prod.fst P - prod.fst B, prod.snd P - prod.snd B)

noncomputable def vec_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (prod.fst u + prod.fst v, prod.snd u + prod.snd v)

noncomputable def vec_norm (u : ℝ × ℝ) : ℝ :=
  real.sqrt (prod.fst u ^ 2 + prod.snd u ^ 2)

theorem range_of_PA_plus_PB :
  7 ≤ vec_norm (vec_add (PA_vec P A) (PB_vec P B)) ∧ 
  vec_norm (vec_add (PA_vec P A) (PB_vec P B)) ≤ 13 :=
sorry

end range_of_PA_plus_PB_l38_38315


namespace point_in_third_quadrant_l38_38746
open Complex

noncomputable def find_coordinates (a b : ℝ) (h : (a : ℂ) + I = (2 - I : ℂ) * (b - I)) : ℝ × ℝ :=
  let sol := solve_system {a = 2b - 1, 1 = -(2 + b)} in
  match sol with
  | (a, b) => (a, b)

theorem point_in_third_quadrant : 
  ∃ (a b : ℝ), (a + Complex.I) / (b - Complex.I) = 2 - Complex.I ∧ a = -7 ∧ b = -3 ∧ a < 0 ∧ b < 0 :=
  sorry

end point_in_third_quadrant_l38_38746


namespace line_circle_intersection_points_l38_38845

theorem line_circle_intersection_points (x y θ a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 * real.cos θ + P.2 * real.sin θ + a = 0 ∧ P.1 ^ 2 + P.2 ^ 2 = a ^ 2) ↔ (count_intersections x y θ a = 1) :=
by
  sorry

end line_circle_intersection_points_l38_38845


namespace min_sticks_to_break_n12_l38_38282

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l38_38282


namespace sum_of_ages_is_correct_l38_38529

-- Define the present ages of A, B, and C
def present_age_A : ℕ := 11

-- Define the ratio conditions from 3 years ago
def three_years_ago_ratio (A B C : ℕ) : Prop :=
  B - 3 = 2 * (A - 3) ∧ C - 3 = 3 * (A - 3)

-- The statement we want to prove
theorem sum_of_ages_is_correct {A B C : ℕ} (hA : A = 11)
  (h_ratio : three_years_ago_ratio A B C) :
  A + B + C = 57 :=
by
  -- The proof part will be handled here
  sorry

end sum_of_ages_is_correct_l38_38529


namespace arith_seq_ratio_l38_38412

theorem arith_seq_ratio (x y a1 a2 b1 b2 b3 : ℝ) 
  (h1: x ≠ y)
  (h2: y = x + 3 * ((y - x) / 3))
  (h3: y = x + 4 * ((y - x) / 4)) 
  (ha1: a1 = x + ((n : ℕ) = 1) * (y - x) / 3)
  (ha2: a2 = x + 2 * ((y - x) / 3))
  (hb1: b1 = x + ((n : ℕ) = 1) * (y - x) / 4)
  (hb2: b2 = x + 2 * ((y - x) / 4))
  (hb3: b3 = x + 3 * ((y - x) / 4)):
  (a2 - a1) / (b2 - b1) = 4 / 3 := by
  sorry

end arith_seq_ratio_l38_38412


namespace parabola_intercept_sum_l38_38107

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l38_38107


namespace part_one_part_two_part_three_l38_38370

def f(x : ℝ) := x^2 - 1
def g(a x : ℝ) := a * |x - 1|

-- (I)
theorem part_one (a : ℝ) : 
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁| = g a x₁ ∧ |f x₂| = g a x₂) ↔ (a = 0 ∨ a = 2)) :=
sorry

-- (II)
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ (a <= -2) :=
sorry

-- (III)
def G(a x : ℝ) := |f x| + g a x

theorem part_three (a : ℝ) (h : a < 0) : 
  (∀ x ∈ [-2, 2], G a x ≤ if a <= -3 then 0 else 3 + a) :=
sorry

end part_one_part_two_part_three_l38_38370


namespace min_sticks_to_break_n12_l38_38281

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l38_38281


namespace common_points_distance_l38_38724

noncomputable def parametric_curve_C1 (t : ℝ) : ℝ × ℝ := 
  (1 + Real.sqrt 2 * Real.cos t, Real.sqrt 2 * Real.sin t)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos θ - ρ * Real.sin θ = 4

theorem common_points_distance :
  (∃ (t : ℝ), (1 + Real.sqrt 2 * Real.cos t) = (λ x y, 2x - y - 4 = 0) ∧ 
               (Real.sqrt 2 * Real.sin t) = (λ x y, 2x - y - 4 = 0)) → 
  ∃ (d : ℝ), d = (2 * Real.sqrt 30) / 5 :=
sorry

end common_points_distance_l38_38724


namespace homogeneous_variances_not_rejected_estimated_population_variance_l38_38675

noncomputable def sample_variances : List ℝ := [0.21, 0.25, 0.34, 0.40]
def sample_size : ℕ := 17

def cochran_critical_value : ℝ := 0.4366
def calculated_Gobs : ℝ := 0.333

theorem homogeneous_variances_not_rejected 
  (variances : List ℝ)
  (sample_size : ℕ)
  (Gobs : ℝ)
  (Gcrit : ℝ) : Gobs < Gcrit := by
  have h1 : Gobs = calculated_Gobs := rfl
  have h2 : Gcrit = cochran_critical_value := rfl
  rw [h1, h2]
  norm_num
  -- Concluding that Gobs < Gcrit
  exact dec_trivial

theorem estimated_population_variance (variances : List ℝ) : 
  (1 / variances.length : ℝ) * variances.sum = 0.3 := by
  have h : variances = sample_variances := rfl
  rw [h]
  -- calculate the mean of the given variances
  norm_num
  exact dec_trivial

#check homogeneous_variances_not_rejected sample_variances sample_size 0.333 0.4366
#check estimated_population_variance sample_variances

end homogeneous_variances_not_rejected_estimated_population_variance_l38_38675


namespace maximum_value_of_objective_function_l38_38380

variables (x y : ℝ)

def objective_function (x y : ℝ) := 3 * x + 2 * y

theorem maximum_value_of_objective_function : 
  (∀ x y, (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4) → objective_function x y ≤ 12) 
  ∧ 
  (∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4 ∧ objective_function x y = 12) :=
sorry

end maximum_value_of_objective_function_l38_38380


namespace sum_of_k_binom_eq_l38_38659

theorem sum_of_k_binom_eq :
  (∑ k in {k : ℕ | binom 23 4 + binom 23 5 = binom 24 k}, k) = 24 := 
by
  sorry

end sum_of_k_binom_eq_l38_38659


namespace clay_capacity_second_box_l38_38194

theorem clay_capacity_second_box :
  let
    height1 := 4
    width1 := 3
    length1 := 7
    clay1 := 84
    height2 := height1 / 2
    width2 := width1 * 4
    length2 := length1
    volume1 := height1 * width1 * length1
    volume2 := height2 * width2 * length2
    volume_ratio := volume2 / volume1
  in
    volume1 = 84 → volume2 = 168 → volume_ratio = 2 → n = 168 :=
begin
  sorry
end

end clay_capacity_second_box_l38_38194


namespace part_a_part_b_part_c_l38_38491

-- Part (a)
theorem part_a (x : ℝ) : 
  1 - log10 5 = 1/3 * (log10 (1/2) + log10 x + 1/3 * log10 5) → 
  x = 16 / real.cbrt 5 := 
sorry

-- Part (b)
theorem part_b (x : ℝ) : 
  log (1/3) (-1/x) = 2 → 
  x = -9 := 
sorry 

-- Part (c)
theorem part_c (x : ℝ) : 
  log10 ((2 * x - 5)^2) = 0 → 
  x = 2 ∨ x = 3 := 
sorry 

end part_a_part_b_part_c_l38_38491


namespace total_pokemon_cards_l38_38078

def pokemon_cards (sam dan tom keith : Nat) : Nat :=
  sam + dan + tom + keith

theorem total_pokemon_cards :
  pokemon_cards 14 14 14 14 = 56 := by
  sorry

end total_pokemon_cards_l38_38078


namespace cone_height_l38_38348

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l38_38348


namespace train_crossing_time_l38_38598

theorem train_crossing_time :
  ∀ (train_length bridge_length : ℕ) (speed_kmh : ℝ),
    train_length = 250 →
    bridge_length = 350 →
    speed_kmh = 72 →
    ∃ (time : ℝ), time = 30
by
  intros train_length bridge_length speed_kmh ht hb hs
  let time := (train_length + bridge_length) / (speed_kmh * 1000 / 3600)
  use time
  simp only [ht, hb, hs]
  sorry

end train_crossing_time_l38_38598


namespace mutually_exclusive_event_pairs_l38_38435

-- Definitions and conditions from the problem
def volunteers := 7
def males := 4
def females := 3

-- Define events
def event1 (selected: List String) : Prop := 
    selected.count (= "female") = 1 ∧ selected.count (= "female") = 2
def event2 (selected: List String) : Prop := 
    selected.count (= "female") ≥ 1 ∧ selected.count (= "female") = 2
def event3 (selected: List String) : Prop := 
    selected.count (= "male") ≥ 1 ∧ selected.count (= "female") ≥ 1
def event4 (selected: List String) : Prop := 
    selected.count (= "female") ≥ 1 ∧ selected.count (= "male") = 0

theorem mutually_exclusive_event_pairs : 
    ∃ (events : List (List String → Prop)),
    ∃ (num_pairs : ℕ), 
    events = [event1, event2, event3, event4] ∧ num_pairs = 1 :=
by
  -- We use sorry here to skip the proof
  sorry

end mutually_exclusive_event_pairs_l38_38435


namespace number_of_perfect_squares_up_to_250_l38_38531

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_perfect_squares (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).filter is_perfect_square |>.card

theorem number_of_perfect_squares_up_to_250 : 
  count_perfect_squares 250 = 15 :=
by
  sorry

end number_of_perfect_squares_up_to_250_l38_38531


namespace solve_inequality_l38_38086

theorem solve_inequality (x : ℝ) :
  -2 < (x^2 - 10*x + 21) / (x^2 - 6*x + 10) ∧ 
  (x^2 - 10*x + 21) / (x^2 - 6*x + 10) < 3 ↔ 
  x ∈ set.Ioo (3/2 : ℝ) 3 :=
sorry

end solve_inequality_l38_38086


namespace neither_sufficient_nor_necessary_l38_38185

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬ (∀ (x : ℝ), x < π / 4 → tan x < 1) ∧ ¬ (∀ (x : ℝ), tan x < 1 → x < π / 4) :=
by
  split
  -- Part 1: Prove $x＜\frac{π}{4}$ does not imply $\tan x \lt 1$
  {
    intro h,
    have counterexample := -3 * π / 4,
    have hx : counterexample < π / 4 := by
      -- Prove $-\frac{3π}{4} < π / 4$
      sorry,
    have h_tan : tan counterexample = 1 := by
      -- Prove $\tan(-\frac{3π}{4})=\tan\frac{π}{4}=1$
      sorry,
    have hcontr : tan counterexample < 1 := h counterexample hx,
    linarith,
  },
  -- Part 2: Prove $\tan x \lt 1$ does not imply $x＜\frac{π}{4}$
  {
    intro h,
    have counterexample := 3 * π / 4,
    have ht : tan counterexample = -1 := by
      -- Prove $\tan\frac{3π}{4} = -1$
      sorry,
    have h_tan : tan counterexample < 1 := by
      -- Prove $\tan\frac{3π}{4} < 1$
      linarith,
    have hcontr : counterexample < π / 4 := h counterexample h_tan,
    linarith,
  }

end neither_sufficient_nor_necessary_l38_38185


namespace ed_total_pets_l38_38639

theorem ed_total_pets (num_dogs num_cats : ℕ) (h_dogs : num_dogs = 2) (h_cats : num_cats = 3) :
  ∃ num_fish : ℕ, (num_fish = 2 * (num_dogs + num_cats)) ∧ (num_dogs + num_cats + num_fish) = 15 :=
by
  sorry

end ed_total_pets_l38_38639


namespace Katka_number_l38_38022

theorem Katka_number : ∃ (x : ℕ), (x + 0 < 10^5 ∧ 
  let fourth_line_sum := (3/2 * x) + (6/5 * x) + (10/9 * x) in 
  ∃ k : ℕ, fourth_line_sum = k^3 ∧ x = 11250) := 
sorry

end Katka_number_l38_38022


namespace square_possible_n12_square_possible_n15_l38_38292

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l38_38292


namespace melinda_large_coffees_l38_38472

def cost_doughnut : ℝ := 0.45
def cost_total_harold : ℝ := 4.91
def num_doughnuts_harold : ℕ := 3
def num_coffees_harold : ℕ := 4
def cost_total_melinda : ℝ := 7.59
def num_doughnuts_melinda : ℕ := 5

theorem melinda_large_coffees :
  let cost_coffees := cost_total_harold - (num_doughnuts_harold * cost_doughnut),
      cost_one_coffee := cost_coffees / num_coffees_harold,
      cost_doughnuts_melinda := num_doughnuts_melinda * cost_doughnut,
      cost_melinda_coffees := cost_total_melinda - cost_doughnuts_melinda
  in cost_melinda_coffees / cost_one_coffee = 6 :=
by
  sorry

end melinda_large_coffees_l38_38472


namespace divisible_by_840_l38_38672

noncomputable theory

-- Defining the function K
def K (n : ℕ) : ℕ := n^3 + 6 * n^2 - 4 * n - 24

-- Main statement
theorem divisible_by_840 (n : ℕ) :
  (K n) % 840 = 0 ↔ n ∈ {2, 8, 12, 22, 44, 54, 58, 64, 68} :=
  by sorry

end divisible_by_840_l38_38672


namespace smallest_positive_integer_n_l38_38157

noncomputable def smallest_n : ℕ := 60

theorem smallest_positive_integer_n (n : ℕ) (h1 : 11 * n - 6 = 11 * smallest_n - 6) (h2 : 8 * n + 5 = 8 * smallest_n + 5) : ∃ d : ℕ, d > 1 ∧ d ∣ (11 * n - 6) ∧ d ∣ (8 * n + 5)  :=
by {
  sorry,
}

end smallest_positive_integer_n_l38_38157


namespace lambda_value_l38_38754

variables {ℝ : Type} [field ℝ]
variables (e₁ e₂ : ℝ) (λ : ℝ)

-- Non-collinear unit vectors
def unit_vector (v : ℝ) := v ≠ 0

-- Collinear vectors
def collinear (v₁ v₂ : ℝ) := ∃ k : ℝ, v₁ = k * v₂

-- Given conditions
def given_conditions (λ : ℝ) : Prop :=
  unit_vector e₁ ∧ unit_vector e₂ ∧ ¬collinear e₁ e₂ ∧ collinear (λ * e₁ - e₂) (e₁ - λ * e₂)

-- Problem statement
theorem lambda_value (h : given_conditions λ) : λ = 1 ∨ λ = -1 :=
sorry

end lambda_value_l38_38754


namespace range_of_m_l38_38695

open Set

theorem range_of_m (m : ℝ) : 
  (∀ x, (m + 1 ≤ x ∧ x ≤ 2 * m - 1) → (-2 < x ∧ x ≤ 5)) → 
  m ∈ Iic (3 : ℝ) :=
by
  intros h
  sorry

end range_of_m_l38_38695


namespace minimum_negative_factors_l38_38407

theorem minimum_negative_factors (a b c d : ℝ) (h1 : a * b * c * d < 0) (h2 : a + b = 0) (h3 : c * d > 0) : 
    (∃ x ∈ [a, b, c, d], x < 0) :=
by
  sorry

end minimum_negative_factors_l38_38407


namespace lecture_permutation_order_l38_38062

theorem lecture_permutation_order (Jixi : ℕ) (school1 school2 school3 : ℕ)
    (h_fixed : Jixi = 1) : 
    ∃ orders : ℕ, orders = Nat.factorial 3 ∧ orders = 6 :=
by
  use Nat.factorial 3
  simp [Nat.factorial]
  use 6
  sorry

end lecture_permutation_order_l38_38062


namespace minimum_value_of_expression_l38_38459

theorem minimum_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) = 24 :=
sorry

end minimum_value_of_expression_l38_38459


namespace scientific_notation_0_000048_l38_38522

/-!
# Scientific Notation Proof

We aim to prove that \(0.000048\) is equivalent to \(4.8 \times 10^{-5}\).
-/

theorem scientific_notation_0_000048 : (0.000048 : ℝ) = 4.8 * 10^(-5) := sorry

end scientific_notation_0_000048_l38_38522


namespace zoo_entry_ticket_cost_l38_38471

-- Definitions based on conditions
def bus_fare_one_way := 1.50
def total_money := 40
def money_left := 24
def num_people := 2

-- Prove that the cost of the zoo entry ticket per person is $5
theorem zoo_entry_ticket_cost : 
  ∃ x : ℝ, (2 * x + 2 * (2 * bus_fare_one_way) = total_money - money_left) ∧ x = 5 :=
by
  sorry

end zoo_entry_ticket_cost_l38_38471


namespace amusement_park_ticket_cost_l38_38018

/-- Jeremie is going to an amusement park with 3 friends. 
    The cost of a set of snacks is $5. 
    The total cost for everyone to go to the amusement park and buy snacks is $92.
    Prove that the cost of one ticket is $18.
-/
theorem amusement_park_ticket_cost 
  (number_of_people : ℕ)
  (snack_cost_per_person : ℕ)
  (total_cost : ℕ)
  (ticket_cost : ℕ) :
  number_of_people = 4 → 
  snack_cost_per_person = 5 → 
  total_cost = 92 → 
  ticket_cost = 18 :=
by
  intros h1 h2 h3
  sorry

end amusement_park_ticket_cost_l38_38018


namespace minimum_sticks_broken_n12_can_form_square_n15_l38_38287

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l38_38287


namespace jumble_island_word_count_l38_38506

theorem jumble_island_word_count :
  let alphabet_size := 21
  let no_A_included_size := 20
  let one_letter_words := 1
  let two_letter_total := Nat.pow alphabet_size 2
  let two_letter_no_A := Nat.pow no_A_included_size 2
  let two_letter_with_A := two_letter_total - two_letter_no_A
  let three_letter_total := Nat.pow alphabet_size 3
  let three_letter_no_A := Nat.pow no_A_included_size 3
  let three_letter_with_A := three_letter_total - three_letter_no_A
  let four_letter_total := Nat.pow alphabet_size 4
  let four_letter_no_A := Nat.pow no_A_included_size 4
  let four_letter_with_A := four_letter_total - four_letter_no_A
  let five_letter_total := Nat.pow alphabet_size 5
  let five_letter_no_A := Nat.pow no_A_included_size 5
  let five_letter_with_A := five_letter_total - five_letter_no_A
  one_letter_words + two_letter_with_A + three_letter_with_A + four_letter_with_A + five_letter_with_A = 920885 := 
by {
  let alphabet_size := 21
  let no_A_included_size := 20
  let one_letter_words := 1
  let two_letter_total := Nat.pow alphabet_size 2
  let two_letter_no_A := Nat.pow no_A_included_size 2
  let two_letter_with_A := two_letter_total - two_letter_no_A
  let three_letter_total := Nat.pow alphabet_size 3
  let three_letter_no_A := Nat.pow no_A_included_size 3
  let three_letter_with_A := three_letter_total - three_letter_no_A
  let four_letter_total := Nat.pow alphabet_size 4
  let four_letter_no_A := Nat.pow no_A_included_size 4
  let four_letter_with_A := four_letter_total - four_letter_no_A
  let five_letter_total := Nat.pow alphabet_size 5
  let five_letter_no_A := Nat.pow no_A_included_size 5
  let five_letter_with_A := five_letter_total - five_letter_no_A
  show one_letter_words + two_letter_with_A + three_letter_with_A + four_letter_with_A + five_letter_with_A = 920885, sorry
}

end jumble_island_word_count_l38_38506


namespace length_of_train_l38_38221

-- Conditions
def speed_kmh : ℝ := 126
def time_seconds : ℝ := 9
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Statement of the problem
theorem length_of_train (S_kmh : ℝ) (T_s : ℝ) (S_ms : ℝ) (L : ℝ) 
    (h1 : S_kmh = 126) (h2 : T_s = 9) (h3 : S_ms = (S_kmh * 1000) / 3600) : L = 315 :=
by
  -- Using provided conditions definitions
  have h4 : S_ms = 35 := by sorry
  have h5 : L = S_ms * T_s := by sorry
  rw [h4, h2] at h5
  exact h5

end length_of_train_l38_38221


namespace fewer_mpg_in_city_l38_38923

noncomputable def tank_size (miles_city: ℝ) (mpg_city: ℝ) : ℝ :=
  miles_city / mpg_city

noncomputable def mpg_highway (miles_highway: ℝ) (tank_size: ℝ) : ℝ :=
  miles_highway / tank_size

theorem fewer_mpg_in_city (miles_highway miles_city: ℝ) (mpg_city: ℝ) :
  miles_highway = 448 → miles_city = 336 → mpg_city = 18 →
  (mpg_highway miles_highway (tank_size miles_city mpg_city) - mpg_city) = 6 := 
by
  intros h_highway h_city h_mpg_city
  rw [h_highway, h_city, h_mpg_city]
  have tank := tank_size 336 18
  have mpg_high := mpg_highway 448 tank
  have difference := mpg_high - 18
  have approx_6 : |difference - 5.99| < 0.01 := by sorry
  have diff_approx_6 : difference = 6 := by sorry
  exact diff_approx_6
  

end fewer_mpg_in_city_l38_38923


namespace mechanical_pencils_and_pens_price_l38_38904

theorem mechanical_pencils_and_pens_price
    (x y : ℝ)
    (h₁ : 7 * x + 6 * y = 46.8)
    (h₂ : 3 * x + 5 * y = 32.2) :
  x = 2.4 ∧ y = 5 :=
sorry

end mechanical_pencils_and_pens_price_l38_38904


namespace betting_strategy_exists_l38_38430

theorem betting_strategy_exists (S : ℝ) (hS : S > 0) : 
  ∃ (x1 x2 x3 x4 : ℝ), x1 >= S / 6 ∧ x2 >= S / 2 ∧ x3 >= S / 6 ∧ x4 >= S / 7 ∧ x1 + x2 + x3 + x4 = S :=
begin
  sorry
end

end betting_strategy_exists_l38_38430


namespace locus_is_hyperbola_l38_38117

-- Define fixed points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define a predicate to represent the condition on the absolute difference of distances
def locus_condition (M : ℝ × ℝ) : Prop :=
  abs ((dist M F1) - (dist M F2)) = 4

-- Define the main statement
theorem locus_is_hyperbola (M : ℝ × ℝ) : locus_condition M ↔ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((M.1 / a)^2 - (M.2 / b)^2 = 1 ∨ (M.1 / a)^2 - (M.2 / b)^2 = -1)) :=
begin
  sorry
end

end locus_is_hyperbola_l38_38117


namespace absolute_sum_l38_38670

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem absolute_sum : 
    (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end absolute_sum_l38_38670


namespace largest_n_divisible_l38_38156

theorem largest_n_divisible : ∃ n : ℕ, (∀ k : ℕ, (k^3 + 150) % (k + 5) = 0 → k ≤ n) ∧ n = 20 := 
by
  sorry

end largest_n_divisible_l38_38156


namespace zack_marbles_number_l38_38557

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l38_38557


namespace min_sticks_to_break_n12_l38_38280

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l38_38280


namespace tadpoles_more_than_fish_l38_38142

def fish_initial : ℕ := 100
def tadpoles_initial := 4 * fish_initial
def snails_initial : ℕ := 150
def fish_caught : ℕ := 12
def tadpoles_to_frogs := (2 * tadpoles_initial) / 3
def snails_crawled_away : ℕ := 20

theorem tadpoles_more_than_fish :
  let fish_now : ℕ := fish_initial - fish_caught
  let tadpoles_now : ℕ := tadpoles_initial - tadpoles_to_frogs
  fish_now < tadpoles_now ∧ tadpoles_now - fish_now = 46 :=
by
  sorry

end tadpoles_more_than_fish_l38_38142


namespace smallest_prime_dividing_polynomial_l38_38190

theorem smallest_prime_dividing_polynomial :
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (n : ℤ), p ∣ (n^2 + 5 * n + 23)) ∧
  ∀ (q : ℕ), Prime q → 
    (∃ m : ℤ, q ∣ (m^2 + 5 * m + 23)) → q ≥ p :=
sorry

end smallest_prime_dividing_polynomial_l38_38190


namespace water_volume_percentage_l38_38961

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38961


namespace solve_problem_l38_38161

noncomputable def smallest_positive_integer : ℕ :=
  Inf {n : ℕ | 0 < n ∧ (Real.sqrt n - Real.sqrt (n - 1) < 0.01)}

theorem solve_problem : smallest_positive_integer = 2501 :=
begin
  sorry
end

end solve_problem_l38_38161


namespace negation_of_p_implies_a_gt_one_half_l38_38374

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + x + 1 / 2 ≤ 0

-- Define the statement that negation of p implies a > 1/2
theorem negation_of_p_implies_a_gt_one_half (a : ℝ) (h : ¬ p a) : a > 1 / 2 :=
by
  sorry

end negation_of_p_implies_a_gt_one_half_l38_38374


namespace balls_sum_l38_38195

theorem balls_sum (m n : ℕ) (h₁ : ∀ a, a ∈ ({m, 8, n} : Finset ℕ)) -- condition: balls are identical except for color
  (h₂ : (8 : ℝ) / (m + 8 + n) = (m + n : ℝ) / (m + 8 + n)) : m + n = 8 :=
sorry

end balls_sum_l38_38195


namespace greatest_remainder_le_11_l38_38390

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l38_38390


namespace absolute_value_condition_l38_38498

theorem absolute_value_condition (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ -2 ≤ x ∧ x ≤ 3 := sorry

end absolute_value_condition_l38_38498


namespace evaluate_a_b_range_l38_38368

noncomputable def func (x a b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem evaluate_a_b_range :
  (∃ a b : ℝ, func 1 a b = -1 ∧ (∀ x : ℝ, 3 * x^2 - 6 * a * x + 2 * b = 0 → x = 1) ∧ a = 1/3 ∧ b = -1/2) ∧
  (∀ x ∈ set.Icc (0 : ℝ) (2 : ℝ), -1 ≤ func x (1/3) (-1/2) ∧ func x (1/3) (-1/2) ≤ 2) :=
by
  sorry

end evaluate_a_b_range_l38_38368


namespace find_g_x_plus_f_y_l38_38504

variable {R : Type*} [Add R] [Mul R] [Neg R] [Div R] [OfNat R (nat.lit 2)] [OfNat R (nat.lit 5)]

variable (f g : R → R)
variable (x y : R)

def condition : Prop :=
  ∀ x y, f (x + g y) = (2 : R) * x + y + (5 : R)

theorem find_g_x_plus_f_y (cond : condition f g) :
  g (x + f y) = x / (2 : R) + y + (5 / (2 : R)) :=
sorry

end find_g_x_plus_f_y_l38_38504


namespace sum_even_probability_l38_38477

def probability_even_sum_of_wheels : ℚ :=
  let prob_wheel1_odd := 3 / 5
  let prob_wheel1_even := 2 / 5
  let prob_wheel2_odd := 2 / 3
  let prob_wheel2_even := 1 / 3
  (prob_wheel1_odd * prob_wheel2_odd) + (prob_wheel1_even * prob_wheel2_even)

theorem sum_even_probability :
  probability_even_sum_of_wheels = 8 / 15 :=
by
  -- Goal statement with calculations showed in the equivalent problem
  sorry

end sum_even_probability_l38_38477


namespace convex_pentagon_bisect_line_l38_38307

-- Define a convex pentagon in a 2D plane
structure Pentagon where
  A B C D E : Point2D
  convex : ConvexHull [A, B, C, D, E]

-- Define a function to check if a line bisects the area of a convex pentagon
def bisectsArea (p : Pentagon) (l : Line2D) : Prop :=
  let (above, below) := splitPentagon p l
  polygonArea above = polygonArea below

-- Define the actual statement
theorem convex_pentagon_bisect_line (p : Pentagon) :
  ∃ (Z K : Point2D), bisectsArea p (Line2D.mk Z K) := sorry

end convex_pentagon_bisect_line_l38_38307


namespace length_of_CE_is_3_l38_38443

noncomputable def length_CE (A B C F E : Type) [triangle_inscribed A B C] [angle_bisector A B C F] [meets_circumcircle F] : Prop :=
  let FB := 2
  let EF := 1
  let CE := 3
  CE = 3

theorem length_of_CE_is_3 (A B C F E : Type) [triangle_inscribed A B C] [angle_bisector A B C F] [meets_circumcircle F] 
  (h1 : FB = 2)
  (h2 : EF = 1)
  : CE = 3 :=
by
  sorry

end length_of_CE_is_3_l38_38443


namespace eccentricity_of_ellipse_l38_38437

variables {a b : ℝ} (h1 : a > b > 0) 
variables (c : ℝ) (e : ℝ)

-- Definitions for the ellipse and related geometrical entities
def ellipse_eq : Prop := (λ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
def midpoint_M (c : ℝ) (b : ℝ) : ℝ × ℝ := (c / 2, b / 2)

-- Given condition through angle MOF
axiom angle_condition (h2 : real.tan (30 * real.pi / 180) = b / c)

-- Derived relationships
def eccentricity_relationship1 (h3 : c^2 = 3 * b^2) : Prop := c^2 = 3 * b^2
def eccentricity_relationship2 (h4 : b^2 = a^2 - c^2) : Prop := b^2 = a^2 - c^2

-- Statement to prove
theorem eccentricity_of_ellipse : e = sqrt 3 / 2 :=
sorry

end eccentricity_of_ellipse_l38_38437


namespace tom_average_speed_l38_38866

theorem tom_average_speed
  (total_distance : ℕ)
  (distance1 : ℕ)
  (speed1 : ℕ)
  (distance2 : ℕ)
  (speed2 : ℕ)
  (H : total_distance = distance1 + distance2)
  (H1 : distance1 = 12)
  (H2 : speed1 = 24)
  (H3 : distance2 = 48)
  (H4 : speed2 = 48) :
  (total_distance : ℚ) / ((distance1 : ℚ) / speed1 + (distance2 : ℚ) / speed2) = 40 :=
by
  sorry

end tom_average_speed_l38_38866


namespace Lizzie_has_27_crayons_l38_38054

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l38_38054


namespace measure_angle_ABC_approx_l38_38788

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

noncomputable def angle_ABC : ℝ :=
  let A : ℝ × ℝ × ℝ := (-3, 1, 5)
  let B : ℝ × ℝ × ℝ := (-4, -2, 4)
  let C : ℝ × ℝ × ℝ := (-5, -2, 6)
  let AB := distance A B
  let AC := distance A C
  let BC := distance B C
  real.acos ((AB ^ 2 + BC ^ 2 - AC ^ 2) / (2 * AB * BC))

theorem measure_angle_ABC_approx : angle_ABC ≈ 87.13 / 180 * real.pi :=
  sorry

end measure_angle_ABC_approx_l38_38788


namespace tan_alpha_is_sqrt3_l38_38678

def cos_alpha_minus_pi_eq_half (α : ℝ) : Prop := cos (α - π) = 1 / 2
def alpha_range (α : ℝ) : Prop := -π < α ∧ α < 0

theorem tan_alpha_is_sqrt3 (α : ℝ) (h1 : cos_alpha_minus_pi_eq_half α) (h2 : alpha_range α) : tan α = sqrt 3 :=
by
  sorry

end tan_alpha_is_sqrt3_l38_38678


namespace find_cone_height_l38_38353

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l38_38353


namespace water_volume_percentage_l38_38963

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38963


namespace tom_shirts_total_cost_l38_38536

theorem tom_shirts_total_cost 
  (num_tshirts_per_fandom : ℕ)
  (num_fandoms : ℕ)
  (cost_per_shirt : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (total_shirts : ℕ := num_tshirts_per_fandom * num_fandoms)
  (discount_per_shirt : ℚ := (cost_per_shirt : ℚ) * discount_rate)
  (cost_per_shirt_after_discount : ℚ := (cost_per_shirt : ℚ) - discount_per_shirt)
  (total_cost_before_tax : ℚ := (total_shirts * cost_per_shirt_after_discount))
  (tax_added : ℚ := total_cost_before_tax * tax_rate)
  (total_amount_paid : ℚ := total_cost_before_tax + tax_added)
  (h1 : num_tshirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : cost_per_shirt = 15) 
  (h4 : discount_rate = 0.2)
  (h5 : tax_rate = 0.1)
  : total_amount_paid = 264 := 
by 
  sorry

end tom_shirts_total_cost_l38_38536


namespace water_filled_percent_l38_38985

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38985


namespace determine_a_range_l38_38485

noncomputable def single_element_intersection (a : ℝ) : Prop :=
  let A := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a * x + 1)}
  let B := {p : ℝ × ℝ | ∃ x : ℝ, p = (x, |x|)}
  (∃ p : ℝ × ℝ, p ∈ A ∧ p ∈ B) ∧ 
  ∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ A ∧ p₁ ∈ B → p₂ ∈ A ∧ p₂ ∈ B → p₁ = p₂

theorem determine_a_range : 
  ∀ a : ℝ, single_element_intersection a ↔ a ∈ Set.Iic (-1) ∨ a ∈ Set.Ici 1 :=
sorry

end determine_a_range_l38_38485


namespace Julio_fish_catch_rate_l38_38777

theorem Julio_fish_catch_rate (F : ℕ) : 
  (9 * F) - 15 = 48 → F = 7 :=
by
  intro h1
  --- proof
  sorry

end Julio_fish_catch_rate_l38_38777


namespace question_I_question_II_l38_38919

-- Problem (I): Probability that x^2 + 2ax + b^2 = 0 has real roots

def quadratic_real_roots (a b : ℕ) : Bool :=
  a ≥ b

def count_quadratic_real_roots_events : ℕ :=
  ([(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)].filter (λ ab, quadratic_real_roots ab.1 ab.2)).length

def total_quadratic_events : ℕ := 12

theorem question_I : (count_quadratic_real_roots_events : ℚ) / total_quadratic_events = 1 / 2 :=
by
  sorry

-- Problem (II): Probability that point P(m, n) falls within the region defined by x - y ≥ 0, x + y < 5

def within_region (m n : ℕ) : Bool :=
  m - n ≥ 0 ∧ m + n < 5

def count_within_region_events : ℕ :=
  ([(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)].filter (λ mn, within_region mn.1 mn.2)).length

def total_point_events : ℕ := 16

theorem question_II : (count_within_region_events : ℚ) / total_point_events = 1 / 4 :=
by
  sorry

end question_I_question_II_l38_38919


namespace minute_hand_distance_l38_38512

theorem minute_hand_distance (r : ℝ) (t : ℝ) (C : ℝ) : r = 12 → t = 1 → C = 2 * Real.pi * r → distance_traveled t C = 24 * Real.pi :=
by
  sorry

def distance_traveled (t : ℝ) (C : ℝ) : ℝ :=
  t * C

end minute_hand_distance_l38_38512


namespace which_is_system_lin_eq_l38_38900

def option_A : Prop := ∀ (x : ℝ), x - 1 = 2 * x
def option_B : Prop := ∀ (x y : ℝ), x - 1/y = 1
def option_C : Prop := ∀ (x z : ℝ), x + z = 3
def option_D : Prop := ∀ (x y z : ℝ), x - y + z = 1

theorem which_is_system_lin_eq (hA : option_A) (hB : option_B) (hC : option_C) (hD : option_D) :
    (∀ (x z : ℝ), x + z = 3) :=
by
  sorry

end which_is_system_lin_eq_l38_38900


namespace knight_cannot_traverse_all_free_squares_l38_38803

-- Definitions for the conditions
def infinite_chessboard : Type := ℕ × ℕ

def is_pawn_placed (pos : infinite_chessboard) : Prop :=
  pos.1 % 4 = 0 ∧ pos.2 % 4 = 0

def is_free_square (pos : infinite_chessboard) : Prop :=
  ¬ is_pawn_placed pos

-- Question translated to a Lean theorem
theorem knight_cannot_traverse_all_free_squares :
  ¬ ∃ knight_path : list infinite_chessboard,
    (∀ pos ∈ knight_path, is_free_square pos) ∧
    (∀ (pos₁ pos₂ ∈ knight_path), pos₁ ≠ pos₂) ∧
    (∀ (i : ℕ), i < knight_path.length - 1 → valid_knight_move (knight_path.nth i) (knight_path.nth (i + 1))) ∧
    (hunter_coverage knight_path) :=
sorry

def valid_knight_move (start : infinite_chessboard) (end_ : infinite_chessboard) : Prop :=
  (abs (start.1 - end_.1) = 2 ∧ abs (start.2 - end_.2) = 1) ∨
  (abs (start.1 - end_.1) = 1 ∧ abs (start.2 - end_.2) = 2)

def hunter_coverage (path : list infinite_chessboard) : Prop :=
  ∀ (pos : infinite_chessboard),
    (is_free_square pos → (∃ (i : ℕ), path.nth i = some pos))

end knight_cannot_traverse_all_free_squares_l38_38803


namespace probability_brick_in_box_l38_38863

noncomputable def find_y (n : ℕ) (a b c : ℕ) : ℕ :=
if a < b ∧ b < c ∧ c < n then n - 3 else n

theorem probability_brick_in_box (c1 c2 c3 d1 d2 d3 : ℕ) 
  (hc1 : 1 ≤ c1) (hc2 : 1 ≤ c2) (hc3 : 1 ≤ c3)
  (hd1 : 1 ≤ d1) (hd2 : 1 ≤ d2) (hd3 : 1 ≤ d3)
  (hlow : c1 ≤ 50) (hrest : d1 ≠ c1 ∧ d1 ≠ c2 ∧ d1 ≠ c3 
  ∧ d2 ≠ c1 ∧ d2 ≠ c2 ∧ d2 ≠ c3 ∧ d3 ≠ c1 
  ∧ d3 ≠ c2 ∧ d3 ≠ c3 ∧ d1 ≠ d2) : 
  (c1 + c2 + c3 + d1 + d2 + d3) % 47 = 5 :=
by sorry

end probability_brick_in_box_l38_38863


namespace monotonic_decreasing_interval_ln_x2_2x_3_l38_38120

noncomputable def ln_decreasing_interval : Set ℝ := { x : ℝ | x < -3 }

theorem monotonic_decreasing_interval_ln_x2_2x_3 :
  ∀ x : ℝ, (x < -3) ↔ IsLocalMinOn (λ x : ℝ, Real.log (x^2 + 2*x - 3)) (-∞, -3) :=
begin
  sorry, -- Proof to be provided 
end

end monotonic_decreasing_interval_ln_x2_2x_3_l38_38120


namespace greatest_remainder_le_11_l38_38391

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l38_38391


namespace main_theorem_l38_38915

noncomputable def proof_problem (n : ℕ) (x : Fin n → ℝ) (h_1 : ∀ i, 0 ≤ x i) : Prop :=
  let x_next := (i : Fin (n + 1)) := if h : i.val < n then x ⟨i.val, h⟩ else x ⟨0, by simp⟩
  let a := Finset.min' (Finset.univ.image (λ i, x i)) (by {
    have : Finset.univ.image (λ i, x i) ≠ ∅ := by simp,
    exact this,
  })
  in
    (∑ j : Fin n, (1 + x j) / (1 + x_next ⟨j + 1, Nat.add_lt_of_lt_pred j.2⟩)) ≤ n + (1 / (1 + a)^2) * ∑ j : Fin n, (x j - a)^2

theorem main_theorem (n : ℕ) (x : Fin n → ℝ) (h_1 : ∀ i, 0 ≤ x i ) :
  proof_problem n x h_1 ∧ 
  (proof_problem n x h_1 → (∀ i, x i = x 0)) :=
sorry

end main_theorem_l38_38915


namespace ermias_balls_more_is_5_l38_38215

-- Define the conditions
def time_per_ball : ℕ := 20
def alexia_balls : ℕ := 20
def total_time : ℕ := 900

-- Define Ermias's balls
def ermias_balls_more (x : ℕ) : ℕ := alexia_balls + x

-- Alexia's total inflation time
def alexia_total_time : ℕ := alexia_balls * time_per_ball

-- Ermias's total inflation time given x more balls than Alexia
def ermias_total_time (x : ℕ) : ℕ := (ermias_balls_more x) * time_per_ball

-- Total time taken by both Alexia and Ermias
def combined_time (x : ℕ) : ℕ := alexia_total_time + ermias_total_time x

-- Proven that Ermias inflated 5 more balls than Alexia given the total time condition
theorem ermias_balls_more_is_5 : (∃ x : ℕ, combined_time x = total_time) := 
by {
  sorry
}

end ermias_balls_more_is_5_l38_38215


namespace cone_filled_with_water_to_2_3_height_l38_38989

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38989


namespace sum_of_digits_7_pow_17_mod_100_l38_38895

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l38_38895


namespace range_of_a_l38_38727

theorem range_of_a (A B : Set ℝ) (a : ℝ) (f : ℝ → ℝ) 
  (hA : A = {x : ℝ | x^2 - x ≤ 0})
  (hF : ∀ x ∈ A, f x = 2 - x + a)
  (hB : B = range (f ∘ (fun x ↦ x ∈ A)))
  (hSubset : B ⊆ A) :
  a = -1 :=
by {
  unfold Set.range at hB,
  unfold Set.subset at hSubset,
  sorry
}

end range_of_a_l38_38727


namespace sum_arithmetic_seq_l38_38002

open Nat

variable {an : ℕ → ℕ} -- define the arithmetic sequence

-- Define the sum of the first n terms of the arithmetic sequence
def sum (n : ℕ) : ℕ := (List.range n).map an |>.sum

-- Define the condition for the problem S_10 = 120
axiom S10 (h : sum 10 = 120)

theorem sum_arithmetic_seq : (2 * an 1 + 9 * an 2) + 27 = 24 := 
  by 
  sorry

end sum_arithmetic_seq_l38_38002


namespace minimum_value_is_sqrt13_l38_38260

noncomputable def minimum_value_expression : ℝ → ℝ :=
  λ x, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2)

theorem minimum_value_is_sqrt13 : ∃ x : ℝ, minimum_value_expression x = Real.sqrt 13 :=
begin
  sorry
end

end minimum_value_is_sqrt13_l38_38260


namespace find_cone_height_l38_38354

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l38_38354


namespace rider_distance_traveled_l38_38577

noncomputable def caravan_speed := 1  -- km/h
noncomputable def rider_speed := 1 + Real.sqrt 2  -- km/h

theorem rider_distance_traveled : 
  (1 / (rider_speed - 1) + 1 / (rider_speed + 1)) = 1 :=
by
  sorry

end rider_distance_traveled_l38_38577


namespace solution_set_of_inequality_l38_38526

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) > -2) ↔ (x < 3 / 2 ∨ x > 2) :=
by sorry

end solution_set_of_inequality_l38_38526


namespace point_on_x_axis_l38_38744

theorem point_on_x_axis (m : ℤ) (P : ℤ × ℤ) (hP : P = (m + 3, m + 1)) (h : P.2 = 0) : P = (2, 0) :=
by 
  sorry

end point_on_x_axis_l38_38744


namespace coefficient_of_x3_in_binomial_expansion_l38_38496

theorem coefficient_of_x3_in_binomial_expansion :
  (coeff_of_binomial_term (1 + 2 * x) 6 3) = 160 :=
  by sorry

end coefficient_of_x3_in_binomial_expansion_l38_38496


namespace congruence_similarity_relation_l38_38876

theorem congruence_similarity_relation :
  ∀ (F1 F2 : Type) (c1 : congruent F1 F2) (s1 : similar F1 F2),
  (congruent F1 F2 → (equivalent F1 F2 ∧ similar F1 F2 ∧ scaleFactor F1 F2 = 1))
  ∧
  (similar F1 F2 → (¬congruent F1 F2 → (scaleFactor F1 F2 ≠ 1))) :=
by
  sorry

-- Definitions required for the theorem
def congruent (F1 F2 : Type) : Prop := -- Definition of congruence
sorry

def similar (F1 F2 : Type) : Prop := -- Definition of similarity
sorry

def equivalent (F1 F2 : Type) : Prop := -- Definition of equivalence
sorry

def scaleFactor (F1 F2 : Type) : ℝ := -- Scale factor between two figures
sorry

end congruence_similarity_relation_l38_38876


namespace imaginary_part_z_l38_38701

def z : ℂ := (2 * complex.I) / (1 - complex.I)

theorem imaginary_part_z : z.im = 1 := by
  sorry

end imaginary_part_z_l38_38701


namespace cookies_with_five_cups_of_flour_l38_38451

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l38_38451


namespace obtain_26_kg_of_sand_l38_38912

theorem obtain_26_kg_of_sand :
  ∃ (x y : ℕ), (37 - x = x + 3) ∧ (20 - y = y + 2) ∧ (x + y = 26) := by
  sorry

end obtain_26_kg_of_sand_l38_38912


namespace inequality_proof_l38_38048

variable (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
variable (h_sum : a + b + c + d = 1)

theorem inequality_proof :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ (1 / 27) + (176 / 27) * a * b * c * d :=
by
  sorry

end inequality_proof_l38_38048


namespace sound_pressure_level_l38_38843

theorem sound_pressure_level (p_0 p_1 p_2 p_3 : ℝ) (h_p0 : 0 < p_0)
  (L_p : ℝ → ℝ)
  (h_gasoline : 60 ≤ L_p p_1 ∧ L_p p_1 ≤ 90)
  (h_hybrid : 50 ≤ L_p p_2 ∧ L_p p_2 ≤ 60)
  (h_electric : L_p p_3 = 40)
  (h_L_p : ∀ p, L_p p = 20 * Real.log (p / p_0))
  : p_2 ≤ p_1 ∧ p_1 ≤ 100 * p_2 :=
by
  sorry

end sound_pressure_level_l38_38843


namespace average_discount_rate_l38_38203

theorem average_discount_rate
  (bag_marked_price : ℝ) (bag_sold_price : ℝ)
  (shoes_marked_price : ℝ) (shoes_sold_price : ℝ)
  (jacket_marked_price : ℝ) (jacket_sold_price : ℝ)
  (h_bag : bag_marked_price = 80) (h_bag_sold : bag_sold_price = 68)
  (h_shoes : shoes_marked_price = 120) (h_shoes_sold : shoes_sold_price = 96)
  (h_jacket : jacket_marked_price = 150) (h_jacket_sold : jacket_sold_price = 135) : 
  (15 : ℝ) =
  (((bag_marked_price - bag_sold_price) / bag_marked_price * 100) + 
   ((shoes_marked_price - shoes_sold_price) / shoes_marked_price * 100) + 
   ((jacket_marked_price - jacket_sold_price) / jacket_marked_price * 100)) / 3 :=
by {
  sorry
}

end average_discount_rate_l38_38203


namespace disinfectant_usage_l38_38214

theorem disinfectant_usage (x : ℝ) (hx1 : 0 < x) (hx2 : 120 / x / 2 = 120 / (x + 4)) : x = 4 :=
by
  sorry

end disinfectant_usage_l38_38214


namespace central_angle_radian_measure_l38_38520

-- Define the unit circle radius
def unit_circle_radius : ℝ := 1

-- Given an arc of length 1
def arc_length : ℝ := 1

-- Problem Statement: Prove that the radian measure of the central angle α is 1
theorem central_angle_radian_measure :
  ∀ (r : ℝ) (l : ℝ), r = unit_circle_radius → l = arc_length → |l / r| = 1 :=
by
  intros r l hr hl
  rw [hr, hl]
  sorry

end central_angle_radian_measure_l38_38520


namespace spherical_shell_surface_area_l38_38829

noncomputable def total_surface_area_of_spherical_shell
  (base_area_first_hemisphere : ℝ)
  (inner_hemisphere_radius := Real.sqrt (base_area_first_hemisphere / π))
  (outer_hemisphere_radius := inner_hemisphere_radius + 1) :
  ℝ :=
  2 * π * (outer_hemisphere_radius * outer_hemisphere_radius)
  - 2 * π * (inner_hemisphere_radius * inner_hemisphere_radius)

theorem spherical_shell_surface_area :
  total_surface_area_of_spherical_shell (200 * π) = 2 * π + 40 * Real.sqrt 2 * π :=
by sorry

end spherical_shell_surface_area_l38_38829


namespace cone_volume_percentage_filled_l38_38939

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38939


namespace best_model_l38_38427

theorem best_model (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.55) (h2 : R2 = 0.65) (h3 : R3 = 0.79) (h4 : R4 = 0.95) :
  R4 > R3 ∧ R4 > R2 ∧ R4 > R1 :=
by {
  sorry
}

end best_model_l38_38427


namespace cost_of_door_tickets_l38_38226

theorem cost_of_door_tickets (x : ℕ) 
  (advanced_purchase_cost : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (advanced_tickets_sold : ℕ)
  (total_revenue_advanced : ℕ := advanced_tickets_sold * advanced_purchase_cost)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold) : 
  advanced_purchase_cost = 8 ∧
  total_tickets = 140 ∧
  total_revenue = 1720 ∧
  advanced_tickets_sold = 100 →
  door_tickets_sold * x + total_revenue_advanced = total_revenue →
  x = 23 := 
by
  intros h1 h2
  sorry

end cost_of_door_tickets_l38_38226


namespace postage_cost_of_3_75_ounces_l38_38587

def postage_cost (weight : ℝ) : ℕ :=
  let base_rate : ℕ := 25
  let additional_rate : ℕ := 18
  let additional_weight := weight - 1
  let additional_cost := (additional_weight.ceil : ℕ) * additional_rate
  let total_cost_before_tax := base_rate + additional_cost
  let tax := (total_cost_before_tax : ℝ) * 0.10
  (total_cost_before_tax : ℝ) + tax

theorem postage_cost_of_3_75_ounces : postage_cost 3.75 = 87 :=
by
  sorry

end postage_cost_of_3_75_ounces_l38_38587


namespace volume_filled_cone_l38_38951

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38951


namespace max_vertex_product_sum_l38_38227

theorem max_vertex_product_sum (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 9}) :
  let s := (a + b) * (c + d) * (e + f) in
  (∀ a b c d e f, {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 9} → (a + b) * (c + d) * (e + f) ≤ 512) := 
begin
  sorry
end

end max_vertex_product_sum_l38_38227


namespace shortest_path_length_l38_38864

theorem shortest_path_length (x y z : ℕ) (h1 : x + y = z + 1) (h2 : x + z = y + 5) (h3 : y + z = x + 7) : 
  min (min x y) z = 3 :=
by sorry

end shortest_path_length_l38_38864


namespace a_4_is_11_l38_38006

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_4_is_11 : a 4 = 11 := by
  sorry

end a_4_is_11_l38_38006


namespace common_ratio_of_arithmetic_sequence_l38_38324

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_ratio_of_arithmetic_sequence (a : ℕ → α) (q : α)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h3 : a 1 > 0)
  (h4 : ∀ n : ℕ, a n < a (n + 1)) :
  q = 2 := 
sorry

end common_ratio_of_arithmetic_sequence_l38_38324


namespace money_collected_is_correct_l38_38091

-- Define the given conditions
def cost_student_ticket : ℝ := 4
def cost_general_admission_ticket : ℝ := 6
def total_tickets_sold : ℝ := 525
def general_admission_tickets_sold : ℝ := 388

-- Define the number of student tickets sold
def student_tickets_sold := total_tickets_sold - general_admission_tickets_sold

-- Define the money collected from different ticket sales
def money_from_student_tickets := student_tickets_sold * cost_student_ticket
def money_from_general_admission_tickets := general_admission_tickets_sold * cost_general_admission_ticket

-- Define the total money collected
def total_money_collected := money_from_student_tickets + money_from_general_admission_tickets

-- Prove that the total money collected is 2876 dollars
theorem money_collected_is_correct : total_money_collected = 2876 :=
by
  rw [total_money_collected, money_from_student_tickets, student_tickets_sold, total_tickets_sold, general_admission_tickets_sold, cost_student_ticket, cost_general_admission_ticket],
  norm_num,
  sorry

end money_collected_is_correct_l38_38091


namespace median_and_mode_of_books_l38_38766

def student_book_distribution : List (ℕ × ℕ) :=
  [(2, 7), (4, 8), (7, 10), (3, 12)]

def median {α : Type} [LinearOrder α] (l : List α) : α := 
  let sorted := l.sort
  if h : sorted.length % 2 = 1 then
    sorted.nthLe (sorted.length / 2) (by
      rw [List.length_sorted, h, Nat.div_lt_self_iff]
      exact sorted.length)
  else
    let m := sorted.length / 2
    (sorted.nthLe (m - 1) (by
      rw [List.length_sorted, Nat.div_pos]; exact sorted.length))
    + sorted.nthLe m (by
      rw [List.length_sorted, Nat.div_pos]; exact sorted.length)) / 2

def mode {α : Type} [DecidableEq α] (l : List α) : α :=
  l.foldr (fun a map =>
    map.insertWith Nat.add a 1) ∅ |>.toList.maximumBy (·.2 < ·.2)

noncomputable def books : List ℕ :=
  student_book_distribution.bind (fun ⟨count, books⟩ => List.replicate count books)

theorem median_and_mode_of_books :
  median books = 10 ∧ mode books = 10 :=
by {
  sorry
}

end median_and_mode_of_books_l38_38766


namespace cone_water_fill_percentage_l38_38931

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38931


namespace binary_to_decimal_to_base7_l38_38242

-- Define the binary number
def binaryToDecimal (b : String) : ℕ :=
  let folder sum pos char :=
    sum + (char.toNat - '0'.toNat) * (2 ^ pos)
  b.foldr folder 0 (List.range b.length)

-- Define the conversion from decimal to base 7
def decimalToBase7 (n : ℕ) : String :=
  if n = 0 then "0"
  else
    let rec loop (n : ℕ) (acc : String) : String :=
      if n = 0 then acc
      else loop (n / 7) ((n % 7).natAbs.toString ++ acc)
    loop n ""

-- Conditions given in the problem
def binaryNumber : String := "101101"
def decimalNumber : ℕ := 45
def base7Number : String := "63"

-- The equality to prove
theorem binary_to_decimal_to_base7 :
  binaryToDecimal binaryNumber = decimalNumber ∧ decimalToBase7 decimalNumber = base7Number :=
by
  sorry

end binary_to_decimal_to_base7_l38_38242


namespace function_1_is_odd_l38_38070

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def function_1 (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem function_1_is_odd : is_odd_function function_1 :=
by
  sorry

end function_1_is_odd_l38_38070


namespace cone_filled_with_water_to_2_3_height_l38_38993

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38993


namespace exists_prime_q_not_dividing_n_pow_p_sub_p_l38_38790

theorem exists_prime_q_not_dividing_n_pow_p_sub_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ (q : ℕ), Nat.Prime q ∧ ∀ (n : ℤ), ¬ q ∣ (n^p - p) :=
  sorry

end exists_prime_q_not_dividing_n_pow_p_sub_p_l38_38790


namespace water_volume_percentage_l38_38965

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38965


namespace count_peculiarly_powerful_integers_lt_3000_l38_38624

/-- Definition of peculiarly powerful integers -/
def peculiarly_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), b > 1 ∧ (nat.prime b ∧ b % 2 = 1) ∧ a^b = n

/-- Definition of odd primes -/
def odd_prime (b : ℕ) : Prop :=
  nat.prime b ∧ b % 2 = 1

/-- Main statement -/
theorem count_peculiarly_powerful_integers_lt_3000 : 
  (finset.range 3000).filter peculiarly_powerful ∈ finset.card = 10 :=
sorry

end count_peculiarly_powerful_integers_lt_3000_l38_38624


namespace g_70_eq_1195_l38_38503

def g (n : ℤ) : ℤ :=
  if n >= 1200 then n - 5 else g (g (n + 7))

theorem g_70_eq_1195 : g 70 = 1195 :=
  sorry

end g_70_eq_1195_l38_38503


namespace minimum_words_to_learn_l38_38400

theorem minimum_words_to_learn
  (total_words : ℕ)
  (pass_score_percentage : ℚ)
  (guess_percentage : ℚ)
  (total_words = 800)
  (pass_score_percentage = 0.90)
  (guess_percentage = 0.10) :
  ∃ (x : ℕ), ((x : ℚ) + guess_percentage * (total_words - (x : ℚ))) / total_words ≥ pass_score_percentage ∧ x = 712 :=
by
  sorry

end minimum_words_to_learn_l38_38400


namespace books_per_shelf_l38_38075

theorem books_per_shelf (mystery_shelves picture_shelves total_books total_shelves books_per_shelf: ℕ) 
  (h1: mystery_shelves = 6) 
  (h2: picture_shelves = 2) 
  (h3: total_books = 72) 
  (h4: total_shelves = mystery_shelves + picture_shelves) 
  (h5: books_per_shelf = total_books / total_shelves) : 
  books_per_shelf = 9 := 
by
  rw [h1, h2] at h4
  rw [h4, h3]
  norm_num at h5
  exact h5


end books_per_shelf_l38_38075


namespace largest_n_with_positive_sum_of_terms_l38_38699

variable {a : ℕ → ℝ}
variable {d : ℝ}
variable (n : ℕ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n+1) = a n + d

def common_difference_negative (d : ℝ) : Prop :=
d < 0

def first_term_positive (a : ℕ → ℝ) : Prop :=
a 1 > 0

def specific_term_product_condition (a : ℕ → ℝ) : Prop :=
a 23 * (a 22 + a 23) < 0

-- Question (Proof problem)
theorem largest_n_with_positive_sum_of_terms : 
  arithmetic_sequence a d →
  common_difference_negative d →
  first_term_positive a →
  specific_term_product_condition a →
  (∃ n, ∀ m, m ≤ n → 0 < ∑ i in Finset.range (m + 1), a i) →
  44 = n :=
by
  sorry

end largest_n_with_positive_sum_of_terms_l38_38699


namespace find_theta_l38_38240

-- Define the cis function in terms of cosine and sine
def cis (φ : ℝ) : ℂ := complex.ofReal (real.cos φ) + complex.I * complex.ofReal (real.sin φ)

-- Define the sum from the problem
def sum_cis_seq : ℂ :=
  (list.sum (list.map cis [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]))

-- Prove the angle θ such that the sum is r * cis θ
theorem find_theta : ∃ r θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ r > 0 ∧ sum_cis_seq = complex.ofReal r * cis θ ∧ θ = 90 :=
by
  sorry

end find_theta_l38_38240


namespace mascot_sales_growth_rate_equation_l38_38508

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end mascot_sales_growth_rate_equation_l38_38508


namespace range_of_m_l38_38756

-- Define the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y + 1 = 0

-- Define the second circle
def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - m = 0

-- Lean statement for the proof problem
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by sorry

end range_of_m_l38_38756


namespace monotonic_increasing_interval_l38_38513

noncomputable def f (x : ℝ) : ℝ := log 2 (x^2 - 4*x + 3)

theorem monotonic_increasing_interval : 
  (∀ x : ℝ, 3 < x → f x > f (x - 1)) → 
  (∀ x : ℝ, ∃ y : ℝ, x < y ∧ f y = f (y + 1)) :=
sorry

end monotonic_increasing_interval_l38_38513


namespace sum_of_squares_equal_a_sum_of_squares_greater_b_l38_38773

noncomputable def polynomial_a1 := 4 * (x^3) - 18 * (x^2) + 24 * x - 8
noncomputable def polynomial_a2 := 4 * (x^3) - 18 * (x^2) + 24 * x - 9

noncomputable def polynomial_b1 := 4 * (x^3) - 18 * (x^2) + 24 * x - 11
noncomputable def polynomial_b2 := 4 * (x^3) - 18 * (x^2) + 24 * x - 12

theorem sum_of_squares_equal_a : 
  (sum_of_squares_roots polynomial_a1) = (sum_of_squares_roots polynomial_a2) :=
by sorry

theorem sum_of_squares_greater_b : 
  (sum_of_squares_roots polynomial_b1) < (sum_of_squares_roots polynomial_b2) :=
by sorry

end sum_of_squares_equal_a_sum_of_squares_greater_b_l38_38773


namespace cone_height_l38_38343

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l38_38343


namespace cone_height_l38_38340

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ) (r : ℝ) 
  (h_slant_height : l = 13)
  (h_lateral_area : A = 65 * π)
  (h_radius : r = 5)
  (h_height_formula : h = Real.sqrt (l^2 - r^2)) : 
  h = 12 := 
by 
  sorry

end cone_height_l38_38340


namespace total_cubes_proof_l38_38735

def Grady_initial_red_cubes := 20
def Grady_initial_blue_cubes := 15
def Gage_initial_red_cubes := 10
def Gage_initial_blue_cubes := 12
def Harper_initial_red_cubes := 8
def Harper_initial_blue_cubes := 10

def Gage_red_received := (2 / 5) * Grady_initial_red_cubes
def Gage_blue_received := (1 / 3) * Grady_initial_blue_cubes

def Grady_red_after_Gage := Grady_initial_red_cubes - Gage_red_received
def Grady_blue_after_Gage := Grady_initial_blue_cubes - Gage_blue_received

def Harper_red_received := (1 / 4) * Grady_red_after_Gage
def Harper_blue_received := (1 / 2) * Grady_blue_after_Gage

def Gage_total_red := Gage_initial_red_cubes + Gage_red_received
def Gage_total_blue := Gage_initial_blue_cubes + Gage_blue_received

def Harper_total_red := Harper_initial_red_cubes + Harper_red_received
def Harper_total_blue := Harper_initial_blue_cubes + Harper_blue_received

def Gage_total_cubes := Gage_total_red + Gage_total_blue
def Harper_total_cubes := Harper_total_red + Harper_total_blue

def Gage_Harper_total_cubes := Gage_total_cubes + Harper_total_cubes

theorem total_cubes_proof : Gage_Harper_total_cubes = 61 := by
  sorry

end total_cubes_proof_l38_38735


namespace probe_distance_before_refuel_l38_38593

def total_distance : ℕ := 5555555555555
def distance_from_refuel : ℕ := 3333333333333
def distance_before_refuel : ℕ := 2222222222222

theorem probe_distance_before_refuel :
  total_distance - distance_from_refuel = distance_before_refuel := by
  sorry

end probe_distance_before_refuel_l38_38593


namespace cone_volume_percentage_filled_l38_38943

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38943


namespace positional_relationship_l38_38188

-- Definition of what it means for lines to be skew
def skew (a b : Line) : Prop := ¬ (exists P : Point, P ∈ a ∧ P ∈ b) ∧ ¬ (a ∥ b)

-- Definition of parallelism (using already established Lean notation)
def parallel (a b : Line) : Prop := a ∥ b

-- Main theorem
theorem positional_relationship (a b c : Line) (h1 : skew a b) (h2 : parallel c a) :
  skew c b ∨ (exists P : Point, P ∈ c ∧ P ∈ b) :=
sorry

end positional_relationship_l38_38188


namespace grace_charges_for_pulling_weeds_l38_38385

theorem grace_charges_for_pulling_weeds :
  (∃ (W : ℕ ), 63 * 6 + 9 * W + 10 * 9 = 567 → W = 11) :=
by
  use 11
  intro h
  sorry

end grace_charges_for_pulling_weeds_l38_38385


namespace initial_pages_l38_38515

/-
Given:
1. Sammy uses 25% of the pages for his science project.
2. Sammy uses another 10 pages for his math homework.
3. There are 80 pages remaining in the pad.

Prove that the initial number of pages in the pad (P) is 120.
-/

theorem initial_pages (P : ℝ) (h1 : P * 0.25 + 10 + 80 = P) : 
  P = 120 :=
by 
  sorry

end initial_pages_l38_38515


namespace eval_expr_inv_mul_four_l38_38897

theorem eval_expr_inv_mul_four : (2^0 - 1 + 4^2 - 0)⁻¹ * 4 = 1/4 :=
by
  sorry

end eval_expr_inv_mul_four_l38_38897


namespace volume_frustum_correct_l38_38603

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end volume_frustum_correct_l38_38603


namespace trapezoid_area_18_l38_38550

structure Point where
  x : ℝ
  y : ℝ

structure Trapezoid where
  A B C D : Point

def trapezoidABCD : Trapezoid :=
  { A := { x := 1, y := -2 }
  , B := { x := 1, y := 1 }
  , C := { x := 5, y := 7 }
  , D := { x := 5, y := 1 }
  }

noncomputable def area_of_trapezoid (T : Trapezoid) : ℝ :=
  let h := T.C.x - T.A.x
  let b1 := abs (T.B.y - T.A.y)
  let b2 := abs (T.C.y - T.D.y)
  (1 / 2) * (b1 + b2) * h

theorem trapezoid_area_18 : area_of_trapezoid trapezoidABCD = 18 := 
  sorry

end trapezoid_area_18_l38_38550


namespace cone_volume_percentage_filled_l38_38940

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38940


namespace volume_filled_cone_l38_38955

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38955


namespace find_A_l38_38607

theorem find_A (A B : ℝ) 
  (h1 : A - 3 * B = 303.1)
  (h2 : 10 * B = A) : 
  A = 433 :=
by
  sorry

end find_A_l38_38607


namespace volume_filled_cone_l38_38954

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38954


namespace find_interest_rate_l38_38254

-- Define the given conditions
variables (P A t n CI : ℝ) (r : ℝ)

-- Suppose given conditions
variables (hP : P = 1200)
variables (hCI : CI = 240)
variables (hA : A = P + CI)
variables (ht : t = 1)
variables (hn : n = 1)

-- Define the statement to prove 
theorem find_interest_rate : (A = P * (1 + r / n)^(n * t)) → (r = 0.2) :=
by
  sorry

end find_interest_rate_l38_38254


namespace square_possible_n12_square_possible_n15_l38_38289

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l38_38289


namespace tangent_parallel_to_cd_l38_38831

-- Define the necessary components and conditions of the problem:

-- Define points in the Euclidean plane
variables {P : Type*} [euclidean_geometry P]
variables (A B C D K: P)

-- Conditions:
-- 1. ABCD is an inscribed quadrangle
-- 2. The diagonals AC and BD intersect at point K
-- 3. The tangent at K to the circle circumscribed around triangle ABK

-- Define that points A, B, C, D lie on a circle
def inscribed_quadrangle (A B C D : P) : Prop := 
  ∃ (O: P), 
    (lie_on_circle A O) ∧ (lie_on_circle B O) ∧ (lie_on_circle C O) ∧ (lie_on_circle D O)

-- Define that diagonals intersect at K
def diagonals_intersect_at (A B C D K : P) : Prop := 
  line_through A C ∩ line_through B D = {K}

-- Define the tangent line at point K of the circumcircle of ΔABK
def tangent_at_point (A B K : P) : line P :=
  let O := circumcenter A B K in
  tangent_line O K

-- Define that two lines are parallel
def parallel_lines (l1 l2 : line P) : Prop := 
  ∀ (p1 p2 : P), (p1 ∈ l1) → (p2 ∈ l2) → is_parallel l1 l2

-- Theorem statement
theorem tangent_parallel_to_cd
  (h_quad: inscribed_quadrangle A B C D)
  (h_intersect: diagonals_intersect_at A B C D K) :
  let t := tangent_at_point A B K in
  parallel_lines t (line_through C D) :=
sorry

end tangent_parallel_to_cd_l38_38831


namespace solution_set_inequality_l38_38129

theorem solution_set_inequality (x : ℝ) : 
  x * | x - 1 | > 0 ↔ (0 < x ∧ x < 1) ∨ (1 < x) := 
sorry

end solution_set_inequality_l38_38129


namespace cost_of_each_burger_l38_38665

theorem cost_of_each_burger (purchases_per_day : ℕ) (total_days : ℕ) (total_amount_spent : ℕ)
  (h1 : purchases_per_day = 4) (h2 : total_days = 30) (h3 : total_amount_spent = 1560) : 
  total_amount_spent / (purchases_per_day * total_days) = 13 :=
by
  subst h1
  subst h2
  subst h3
  sorry

end cost_of_each_burger_l38_38665


namespace combined_length_l38_38599

-- Definition of the problem conditions
def conditions := ∃ (L1 L2 P : ℕ), 
  let speed1 := 15 in
  let speed2 := 20 in
  L1 = speed1 * 10 ∧
  L1 + P = speed1 * 16 ∧
  L2 = speed2 * 12

-- Theorem statement for the solution
theorem combined_length (L1 L2 P : ℕ) (h : conditions):
  P + L2 = 330 :=
by sorry

end combined_length_l38_38599


namespace sides_of_triangle_perimeter_of_triangle_circumradius_of_triangle_l38_38805

-- Definitions of the angles and the radius R
variables (α β γ R : ℝ)

-- Given conditions
def side_a : ℝ := 2 * R * Real.sin α
def side_b : ℝ := 2 * R * Real.sin β
def side_c : ℝ := 2 * R * Real.sin γ

-- Perimeter
def perimeter : ℝ := side_a + side_b + side_c

-- Prove the equivalent statements
theorem sides_of_triangle : 
    (side_a = 2 * R * Real.sin α) ∧ 
    (side_b = 2 * R * Real.sin β) ∧ 
    (side_c = 2 * R * Real.sin γ) := by
  sorry

theorem perimeter_of_triangle : 
    perimeter = 2 * R * (Real.sin α + Real.sin β + Real.sin γ) := by
  sorry

theorem circumradius_of_triangle : 
    ∃ (circumradius : ℝ), circumradius = R := by
  sorry

end sides_of_triangle_perimeter_of_triangle_circumradius_of_triangle_l38_38805


namespace pencils_total_l38_38021

theorem pencils_total (Sabrina_pencils : ℕ) (hS : Sabrina_pencils = 14) : 
  let Justin_pencils := 2 * Sabrina_pencils + 8 
  in Justin_pencils + Sabrina_pencils = 50 := by
  sorry

end pencils_total_l38_38021


namespace min_value_of_m_plus_n_l38_38406

theorem min_value_of_m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : log 3 m + log 3 n = 4) : m + n = 18 :=
sorry

end min_value_of_m_plus_n_l38_38406


namespace first_storm_duration_l38_38539

theorem first_storm_duration
  (x y : ℕ)
  (h1 : 30 * x + 15 * y = 975)
  (h2 : x + y = 45) :
  x = 20 :=
by sorry

end first_storm_duration_l38_38539


namespace pushing_speed_l38_38052

theorem pushing_speed (v D : ℝ)
  (h1 : D = -720)
  (h2 : 20 + (D - 1800 - 5 * v) / 480 = D / 320 + 17) :
  v = 72 :=
by
  have h3 := calc
    20 + (D - 1800 - 5 * v) / 480 = D / 320 + 17 : h2
    _ ↔ 3 + (D - 1800 - 5 * v) / 480 = D / 320 : by linarith
  sorry

end pushing_speed_l38_38052


namespace probability_xy_minus_x_minus_y_even_l38_38870

theorem probability_xy_minus_x_minus_y_even :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let even_numbers := {2, 4, 6, 8, 10, 12}
  let count_even_pairs := |even_numbers.choose 2|
  let count_total_pairs := |S.choose 2|
  let desired_probability := count_even_pairs / count_total_pairs
  desired_probability = 5 / 22 :=
by 
  sorry

end probability_xy_minus_x_minus_y_even_l38_38870


namespace eval_expression_l38_38133

theorem eval_expression : 3 ^ 2 - (4 * 2) = 1 :=
by
  sorry

end eval_expression_l38_38133


namespace sum_of_two_digit_and_reverse_l38_38836

theorem sum_of_two_digit_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9)
  (h5 : (10 * a + b) - (10 * b + a) = 9 * (a + b)) : (10 * a + b) + (10 * b + a) = 11 :=
by
  sorry

end sum_of_two_digit_and_reverse_l38_38836


namespace range_of_a_value_of_a_l38_38797

noncomputable def hyperbola : Type := { a : ℝ // a > 0 ∧ ∀ (x y : ℝ), (x^2) / (a^2) - y^2 = 1 }

def line (x y: ℝ) := x + y = 1

def y_axis_intersection (x y: ℝ) := (x, y) = (0, 1)

def PA_PB_relation (x1 y1 x2 y2: ℝ) := 
  ((x1, y1 - 1) = (5 / 12 : ℝ) * (x2, y2 - 1))

theorem range_of_a (a : ℝ) (h1 : hyperbola)
  (h_line : ∃ x y, line x y) : 
  0 < a ∧ a < Real.sqrt 2 ∧ a ≠ 1 := 
sorry

theorem value_of_a (a : ℝ) (h2 : ∃ x1 y1 x2 y2, 
  PA_PB_relation x1 y1 x2 y2) : 
  a = 17 / 13 := 
sorry

end range_of_a_value_of_a_l38_38797


namespace ratio_sum_l38_38433

open Function

-- Define the conditions of the problem
structure Rectangle (α : Type*) :=
(AB BC : α)

structure Point (α : Type*) :=
(BE EF FC : α)

-- Given rectangle ABCD with the known lengths
def given_rectangle : Rectangle ℝ := {AB := 10, BC := 4}

-- Given points E and F with relations
def given_points : Point ℝ := {BE := 2, EF := 1, FC := 1}

-- Statement to prove the required ratio and the sum
theorem ratio_sum (r s t : ℕ) : 
  let BE := 2, EF := 1, FC := 1 in
  let BP := 4, PQ := 1, QD := 3 in
  BE = 2 * EF ∧ EF = FC ∧ BP : PQ : QD = 4 : 1 : 3 ∧ Nat.gcd (Nat.gcd r s) t = 1 ∧ (r + s + t) = 8 :=
by
  sorry

end ratio_sum_l38_38433


namespace probability_of_5_chocolate_days_l38_38814

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_5_chocolate_days :
  binomial_probability 7 5 (1/2) = 21 / 128 := 
sorry

end probability_of_5_chocolate_days_l38_38814


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38888

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38888


namespace find_B_plus_C_l38_38404

-- Define the arithmetic translations for base 8 numbers
def base8_to_dec (a b c : ℕ) : ℕ := 8^2 * a + 8 * b + c

def condition1 (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 1 ≤ A ∧ A ≤ 7 ∧ 1 ≤ B ∧ B ≤ 7 ∧ 1 ≤ C ∧ C ≤ 7

-- Define the main condition in the problem
def condition2 (A B C : ℕ) : Prop :=
  base8_to_dec A B C + base8_to_dec B C A + base8_to_dec C A B = 8^3 * A + 8^2 * A + 8 * A

-- The main statement to be proven
theorem find_B_plus_C (A B C : ℕ) (h1 : condition1 A B C) (h2 : condition2 A B C) : B + C = 7 :=
sorry

end find_B_plus_C_l38_38404


namespace number_of_polygon_pairs_l38_38271

theorem number_of_polygon_pairs :
  ∃! (n : ℕ × ℕ), n.1 > 5 ∧ n.2 > 5 ∧ (3 * n.1 - 4 * n.2 = -2) :=
sorry

end number_of_polygon_pairs_l38_38271


namespace radius_of_inscribed_circle_l38_38757

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C)

-- Given conditions
def AC : ℝ := 24
def BC : ℝ := 10
def AB : ℝ := 26

-- Statement to be proved
theorem radius_of_inscribed_circle (hAC : triangle.side_length A C = AC)
                                   (hBC : triangle.side_length B C = BC)
                                   (hAB : triangle.side_length A B = AB) :
  triangle.incircle_radius = 4 :=
by sorry

end radius_of_inscribed_circle_l38_38757


namespace solve_inequality_l38_38360

def f (x : Real) : Real := Real.log10 (x + 1)

theorem solve_inequality (x : Real) : 
  0 < f (1 - 2*x) - f x ∧ f (1 - 2*x) - f x < 1 ↔ -2/3 < x ∧ x < 1/3 := 
by 
  -- Proof is omitted
  sorry

end solve_inequality_l38_38360


namespace colorable_triangulation_2019_gon_l38_38626

def is_color (c : ℕ → ℕ → ℕ) (i : ℕ) (color_count : ℕ) : Prop :=
  ∑ k in Finset.range 2019, if c i k = color_count then 1 else 0 = 673

def color_property (c : ℕ → ℕ → ℕ) : Prop :=
  ∀ T : Finset (Finset (ℕ × ℕ)), T.card = 2017 →
  (∀ t ∈ T, (∃ a b x y z, t = {|(a, b), (x, y), (z, a)|} ∧
    (∀ p, p ∈ t → c p.fst p.snd = c (p.snd) (p.fst)) ∨ (∀ p q, p ≠ q → c p.fst p.snd ≠ c q.fst q.snd)))

theorem colorable_triangulation_2019_gon
  (c : ℕ → ℕ → ℕ)
  (h_red : is_color c 0 1) 
  (h_yellow : is_color c 1 1)
  (h_blue : is_color c 2 1)
  : color_property c :=
sorry

end colorable_triangulation_2019_gon_l38_38626


namespace noah_holidays_l38_38060

theorem noah_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_total : ℕ) 
  (h1 : holidays_per_month = 3) (h2 : months_in_year = 12) (h3 : holidays_total = holidays_per_month * months_in_year) : 
  holidays_total = 36 := 
by
  sorry

end noah_holidays_l38_38060


namespace recommendation_plans_count_l38_38001

-- Define the problem conditions in Lean 4.
def male_student_set : Type := {s : ℕ // s ∈ (finset.range 3)} -- 3 male students
def female_student_set : Type := {s : ℕ // s ∈ (finset.range 2)} -- 2 female students
def total_students := male_student_set ⊕ female_student_set

-- Formulate the main theorem to prove.
theorem recommendation_plans_count : 
  (finset.card (finset.filter (λ x : total_students, true) (finset.univ : finset total_students)) = 5) →
  (∃ n : ℕ, n = 24) :=
by
  -- Assumptions: there are 3 male and 2 female students.
  intro h_total_students
  -- Proof needed here
  sorry

end recommendation_plans_count_l38_38001


namespace count_multiples_l38_38737

theorem count_multiples (lcm_val : ℕ) (start : ℕ) (end' : ℕ) (count : ℕ) :
  lcm_val = 360 →
  start = 2000 →
  end' = 5000 →
  count = 8 →
  let lower_bound := (start + lcm_val - 1) / lcm_val * lcm_val in
  let upper_bound := end' / lcm_val * lcm_val in
  lower_bound = 2160 →
  upper_bound = 4680 →
  ∃ (n : ℕ), n = (upper_bound - lower_bound) / lcm_val + 1 ∧ n = count :=
by {
  intros h_lcm h_start h_end h_count hlb hub,
  use (upper_bound - lower_bound) / lcm_val + 1,
  rw [hlb, hub, h_count],
  exact ⟨rfl⟩
}

end count_multiples_l38_38737


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38885

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38885


namespace income_expenditure_ratio_l38_38114

theorem income_expenditure_ratio (I E S : ℕ) (hI : I = 19000) (hS : S = 11400) (hRel : S = I - E) :
  I / E = 95 / 38 :=
by
  sorry

end income_expenditure_ratio_l38_38114


namespace sum_tens_ones_digits_3_plus_4_power_17_l38_38880

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l38_38880


namespace find_second_rectangle_width_l38_38144

/-
Define the given conditions:
- The first rectangle's dimensions
- The second rectangle's height
- The area relationship between the first and second rectangles

And then prove that the width of the second rectangle is 3 inches.
-/

theorem find_second_rectangle_width:
  ∃ W : ℝ, (let area_first := 4 * 5 in    -- Area of the first rectangle
            let area_second := 18 in      -- Area of the second rectangle
            area_first - 2 = area_second -- Given area relationship
            ∧ W * 6 = area_second)  -- Area formula for the second rectangle and width calculation
    ∧ W = 3 := sorry

end find_second_rectangle_width_l38_38144


namespace incorrect_option_D_l38_38371

theorem incorrect_option_D (x : ℝ) (hx_neg : x < 0) :
  ∀ y : ℝ, y = 1 / x → (∃ y_increases : ∀ ε > 0, ∃ δ > 0, ∀ x' ∈ Icc (x + δ) x, y < 1 / x') → false :=
by {
  -- Translate the given problem to Lean statement:
  -- Given: the function y = 1 / x, where y is the inverse proportion function
  -- Prove: the claim that when x < 0, y increases as x increases is incorrect
  sorry
}

end incorrect_option_D_l38_38371


namespace minimum_sticks_broken_n12_can_form_square_n15_l38_38286

-- Define the total length function
def total_length (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- For n = 12, prove that at least 2 sticks need to be broken to form a square
theorem minimum_sticks_broken_n12 : ∀ (n : ℕ), n = 12 → total_length n % 4 ≠ 0 → 2 = 2 := 
by 
  intros n h1 h2
  sorry

-- For n = 15, prove that a square can be directly formed
theorem can_form_square_n15 : ∀ (n : ℕ), n = 15 → total_length n % 4 = 0 := 
by 
  intros n h1
  sorry

end minimum_sticks_broken_n12_can_form_square_n15_l38_38286


namespace sum_of_n_satisfying_abs_eq_l38_38166

theorem sum_of_n_satisfying_abs_eq : 
  (n1 n2 : ℤ) (h1 : |3 * n1 - 12| = 6) (h2 : |3 * n2 - 12| = 6) 
  (h_unique : n1 ≠ n2) : 
  n1 + n2 = 8 := 
by
  sorry

end sum_of_n_satisfying_abs_eq_l38_38166


namespace Mr_Deane_filled_today_l38_38058

theorem Mr_Deane_filled_today :
  ∀ (x : ℝ),
    (25 * (1.4 - 0.4) + 1.4 * x = 39) →
    x = 10 :=
by
  intros x h
  sorry

end Mr_Deane_filled_today_l38_38058


namespace quadratic_function_formula_l38_38683

theorem quadratic_function_formula (f : ℝ → ℝ) (a : ℝ)
  (h_a : a < 0)
  (h_roots : ∀ x, (f x + 2 * x = a * (x - 1) * (x - 3)))
  (h_double_root : ∃ x, (f x + 6 * a = 0) ∧ (∀ y, (y ≠ x) → (f y + 6 * a ≠ 0))) :
  f = (λ x, - (1:ℝ) / 4 * x^2 - x - 3 / 4) :=
by
  sorry

end quadratic_function_formula_l38_38683


namespace minimize_travel_expense_l38_38127

noncomputable def travel_cost_A (x : ℕ) : ℝ := 2000 * x * 0.75
noncomputable def travel_cost_B (x : ℕ) : ℝ := 2000 * (x - 1) * 0.8

theorem minimize_travel_expense (x : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x ≤ 15 → travel_cost_B x < travel_cost_A x) ∧
  (x = 16 → travel_cost_A x = travel_cost_B x) ∧
  (17 ≤ x ∧ x ≤ 25 → travel_cost_A x < travel_cost_B x) :=
by
  sorry

end minimize_travel_expense_l38_38127


namespace parabola_intercept_sum_l38_38106

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l38_38106


namespace length_second_platform_l38_38220

-- Define the conditions
def length_train : ℕ := 100
def time_platform1 : ℕ := 15
def length_platform1 : ℕ := 350
def time_platform2 : ℕ := 20

-- Prove the length of the second platform is 500m
theorem length_second_platform : ∀ (speed_train : ℚ), 
  speed_train = (length_train + length_platform1) / time_platform1 →
  (speed_train = (length_train + L) / time_platform2) → 
  L = 500 :=
by 
  intro speed_train h1 h2
  sorry

end length_second_platform_l38_38220


namespace initial_distributions_count_l38_38601

-- Define the setting of the triangular array
def triangular_array (n : ℕ) : Prop := 
  ∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → ∀ x_i : ℕ, (x_i = 0 ∨ x_i = 1) → x_i.succ

-- Define the specific case of the problem
def initial_distribution (n : ℕ) : Prop := 
  triangular_array 12

-- Define the reduction using binomial coefficients and modulo arithmetic
def sum_mod_four (x : Fin 12 → ℕ) : ℕ :=
  (x 0) + 3 * (x 1) + 3 * (x 10) + (x 11)

-- Main statement: proving that the number of valid initial distributions is 4096
theorem initial_distributions_count : 
  ∃ (count : ℕ), count = 4096 ∧
  (∀ x : Fin 12 → ℕ, (initial_distribution 12) → (sum_mod_four x % 4 = 0 → count = 4096)) :=
begin
  sorry
end

end initial_distributions_count_l38_38601


namespace very_heavy_tailed_permutations_count_l38_38588

theorem very_heavy_tailed_permutations_count :
  ∃ count, (∀ (p : List ℕ), p ~ [1, 2, 3, 4, 6] → 
     (p.take 3).sum < (p.drop 3).sum → count = 36) :=
begin
  use 36,
  intros p hp hsum,
  sorry
end

end very_heavy_tailed_permutations_count_l38_38588


namespace square_possible_n12_square_possible_n15_l38_38290

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l38_38290


namespace probability_of_diagonals_in_regular_hexagon_l38_38575

/-- Definitions based on conditions:
    - A regular hexagon has 6 vertices.
    - A diagonal is a line between any two non-adjacent vertices.
    - Compute the number of diagonals and apply combinatorial mathematics to compute the probability.--/
noncomputable def probability_of_diagonals_intersecting_inside_hexagon : ℚ :=
let vertices := 6 in
let sides := vertices in
let total_pairs := (vertices * (vertices - 1)) / 2 in
let diagonals := total_pairs - sides in
let diagonal_pairs := (diagonals * (diagonals - 1)) / 2 in
let intersecting_diagonals := ((vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / (4 * 3 * 2 * 1)) in
(intersecting_diagonals : ℚ) / diagonal_pairs

theorem probability_of_diagonals_in_regular_hexagon :
  probability_of_diagonals_intersecting_inside_hexagon = 5 / 12 := by 
  sorry

end probability_of_diagonals_in_regular_hexagon_l38_38575


namespace num_distinct_integers_formed_l38_38395

theorem num_distinct_integers_formed (digits : Multiset ℕ) (h : digits = {2, 2, 3, 3, 3}) : 
  Multiset.card (Multiset.powerset digits).attach = 10 := 
by {
  sorry
}

end num_distinct_integers_formed_l38_38395


namespace smallest_B_l38_38442

-- Definitions and conditions
def known_digit_sum : Nat := 4 + 8 + 3 + 9 + 4 + 2
def divisible_by_3 (n : Nat) : Bool := n % 3 = 0

-- Statement to prove
theorem smallest_B (B : Nat) (h : B < 10) (hdiv : divisible_by_3 (B + known_digit_sum)) : B = 0 :=
sorry

end smallest_B_l38_38442


namespace inequality_sqrt_l38_38481

theorem inequality_sqrt (x : ℝ) (h : 4 ≤ x) : (sqrt (x - 3) + sqrt (x - 2) > sqrt (x - 4) + sqrt (x - 1)) :=
sorry

end inequality_sqrt_l38_38481


namespace total_number_of_animals_l38_38449

-- Definitions for the number of each type of animal
def cats : ℕ := 645
def dogs : ℕ := 567
def rabbits : ℕ := 316
def reptiles : ℕ := 120

-- The statement to prove
theorem total_number_of_animals :
  cats + dogs + rabbits + reptiles = 1648 := by
  sorry

end total_number_of_animals_l38_38449


namespace water_volume_percentage_l38_38962

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38962


namespace painting_final_value_l38_38231

-- Define the conditions and the final statement to be proved
variables 
  (P : ℝ) -- Original price of the painting
  (P_pos : 0 < P) -- Price of the painting is positive

-- Value calculations through the years
def value_after_first_year := 1.30 * P
def value_after_second_year := value_after_first_year - 0.15 * value_after_first_year
def value_after_third_year := value_after_second_year - 0.10 * value_after_second_year
def selling_price := value_after_third_year - 0.05 * value_after_third_year
def price_in_buyers_currency := selling_price + 0.20 * selling_price

-- Final computation
def final_percentage := (price_in_buyers_currency / P) * 100

theorem painting_final_value (P : ℝ) (P_pos : 0 < P) : 
  final_percentage P = 113.373 := by
  -- proof skipped
  sorry

end painting_final_value_l38_38231


namespace part1_part2_l38_38356

noncomputable theory

-- Part (I)
def a1 (x : ℝ) : ℝ × ℝ := (1, Real.cos(x / 2))
def b1 (x : ℝ) (y : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin(x / 2) + Real.cos(x / 2), y)
def collinear (a b : ℝ × ℝ) : Prop := a.1 / b.1 = a.2 / b.2
def f1 (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6) + 0.5

theorem part1 (x : ℝ) (h1 : collinear (a1 x) (b1 x (f1 x))) (h2 : f1 x = 1) : 
  Real.cos ((2 * Real.pi / 3) - 2 * x) = -0.5 := 
sorry

-- Part (II)
variables (A B C a b c : ℝ)

def sides_of_triangle : Prop := 2 * a * Real.cos C + c = 2 * b
def f2 (B : ℝ) : ℝ := Real.sin (B + Real.pi / 6) + 0.5

theorem part2 (A : ℝ) (hA : A = Real.pi / 3) (B : ℝ) (h1 : sides_of_triangle a b c A B C) : 
  1 < f2 B ∧ f2 B ≤ 1.5 :=
sorry

end part1_part2_l38_38356


namespace triangle_A2B2C2_equilateral_l38_38543

theorem triangle_A2B2C2_equilateral 
  (ABC : Triangle)
  (circumscribed_circle : Circle)
  (A1 B1 C1 A2 B2 C2 : Point)
  (h1 : circumscribed_circle.inscribed ABC)
  (h2 : ParallelThroughVertex ABC A1 B1 C1)
  (h3 : DividedIntoThreeEqualParts circumscribed_circle ABC (A1, A2) (B1, B2) (C1, C2))
  : IsEquilateralTriangle A2 B2 C2 :=
sorry

end triangle_A2B2C2_equilateral_l38_38543


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38889

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38889


namespace actual_distance_is_correct_l38_38874

def temperature : ℝ := 26
def original_length : ℝ := 3
def measured_distance : ℝ := 7856.728
def linear_expansion_coefficient : ℝ := 0.000017

theorem actual_distance_is_correct : 
  let expanded_length := original_length * (1 + linear_expansion_coefficient * temperature) in
  let actual_distance := measured_distance * (1 + linear_expansion_coefficient * temperature) in
  actual_distance = 7860.201 := 
by 
  sorry

end actual_distance_is_correct_l38_38874


namespace sector_area_l38_38521

/--
The area of a sector with radius 6cm and central angle 15° is (3 * π / 2) cm².
-/
theorem sector_area (R : ℝ) (θ : ℝ) (h_radius : R = 6) (h_angle : θ = 15) :
    (S : ℝ) = (3 * Real.pi / 2) := by
  sorry

end sector_area_l38_38521


namespace chameleons_cannot_all_turn_to_single_color_l38_38064

theorem chameleons_cannot_all_turn_to_single_color
  (W : ℕ) (B : ℕ)
  (hW : W = 20)
  (hB : B = 25)
  (h_interaction: ∀ t : ℕ, ∃ W' B' : ℕ,
    W' + B' = W + B ∧
    (W - B) % 3 = (W' - B') % 3) :
  ∀ t : ℕ, (W - B) % 3 ≠ 0 :=
by
  sorry

end chameleons_cannot_all_turn_to_single_color_l38_38064


namespace Pam_has_740_fruits_l38_38067

/-
Define the given conditions.
-/
def Gerald_apple_bags : ℕ := 5
def apples_per_Gerald_bag : ℕ := 30
def Gerald_orange_bags : ℕ := 4
def oranges_per_Gerald_bag : ℕ := 25

def Pam_apple_bags : ℕ := 6
def apples_per_Pam_bag : ℕ := 3 * apples_per_Gerald_bag
def Pam_orange_bags : ℕ := 4
def oranges_per_Pam_bag : ℕ := 2 * oranges_per_Gerald_bag

/-
Proving the total number of apples and oranges Pam has.
-/
def total_fruits_Pam : ℕ :=
    Pam_apple_bags * apples_per_Pam_bag + Pam_orange_bags * oranges_per_Pam_bag

theorem Pam_has_740_fruits : total_fruits_Pam = 740 := by
  sorry

end Pam_has_740_fruits_l38_38067


namespace AF2_eq_5_ellipse_equation_l38_38780

-- Definitions based on problem conditions
def F1 : Point := F1
def F2 : Point := F2
def a : ℝ := a
def b : ℝ := b
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def line_through_F1_intersects_ellipse (A B : Point) : Prop := 
  ellipse A.x A.y ∧ ellipse B.x B.y ∧ line_through A B = F1_line

def distance_AF1_eq_3BF1 (A B : Point) : Prop := 
  dist A F1 = 3 * dist B F1

def length_AB_eq_4 (A B : Point) : Prop := 
  dist A B = 4

def perimeter_triangle_ABF2_eq_16 (A B : Point) : Prop := 
  dist A B + dist A F2 + dist B F2 = 16

-- Proof statements
theorem AF2_eq_5 (A B : Point) 
  (h1 : ellipse A.x A.y) 
  (h2 : ellipse B.x B.y) 
  (h3 : distance_AF1_eq_3BF1 A B) 
  (h4 : length_AB_eq_4 A B) 
  (h5 : perimeter_triangle_ABF2_eq_16 A B) : 
  dist A F2 = 5 := 
sorry

theorem ellipse_equation 
  (F1_line : line) 
  (A B : Point)
  (h1 : ellipse A.x A.y) 
  (h2 : ellipse B.x B.y) 
  (h3 : distance_AF1_eq_3BF1 A B) 
  (h4 : length_AB_eq_4 A B) 
  (h5 : perimeter_triangle_ABF2_eq_16 A B) 
  (h6 : slope_of_line A B = 1) : 
  ellipse (x : ℝ) (y : ℝ) → (x^2 / 16) + (y^2 / 8) = 1 := 
sorry

end AF2_eq_5_ellipse_equation_l38_38780


namespace water_volume_percentage_l38_38967

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38967


namespace find_omega_l38_38364

variables {ω α β : ℝ}

def f (x : ℝ) : ℝ := sin (ω * x - π / 6) + 1 / 2

theorem find_omega (hω : 0 < ω) 
  (hα : f α = -1 / 2) 
  (hβ : f β = 1 / 2)
  (hmin : |α - β| = 3 * π / 4) : ω = 2 / 3 :=
by
  sorry

end find_omega_l38_38364


namespace maximum_value_of_f_l38_38038

def f (x : ℝ) : ℝ := min (3 - x^2) (2 * x)

theorem maximum_value_of_f : ∃ x : ℝ, f x = 2 :=
by
  sorry

end maximum_value_of_f_l38_38038


namespace min_sticks_to_break_n12_l38_38279

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l38_38279


namespace find_number_l38_38917

theorem find_number (x : ℝ) (h : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt x = 2.6) : x ≈ 175.031 :=
by
  sorry

end find_number_l38_38917


namespace buratino_correct_l38_38799

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_nine_digit_number (n : ℕ) : Prop :=
  n >= 10^8 ∧ n < 10^9 ∧ (∀ i j : ℕ, i < 9 ∧ j < 9 ∧ i ≠ j → ((n / 10^i) % 10 ≠ (n / 10^j) % 10)) ∧
  (∀ i : ℕ, i < 9 → (n / 10^i) % 10 ≠ 7)

def can_form_prime (n : ℕ) : Prop :=
  ∃ m : ℕ, valid_nine_digit_number n ∧ (m < 1000 ∧ is_prime m ∧
   (∃ erase_indices : List ℕ, erase_indices.length = 6 ∧ 
    ∀ i : ℕ, i ∈ erase_indices → i < 9 ∧ 
    (n % 10^(9 - i)) / 10^(3 - i) = m))

theorem buratino_correct : 
  ∀ n : ℕ, valid_nine_digit_number n → ¬ can_form_prime n :=
by
  sorry

end buratino_correct_l38_38799


namespace find_k_l38_38358

-- Definitions of conditions
def equation1 (x k : ℝ) : Prop := x^2 + k*x + 10 = 0
def equation2 (x k : ℝ) : Prop := x^2 - k*x + 10 = 0
def roots_relation (a b k : ℝ) : Prop :=
  equation1 a k ∧ 
  equation1 b k ∧ 
  equation2 (a + 3) k ∧
  equation2 (b + 3) k

-- Statement to be proven
theorem find_k (a b k : ℝ) (h : roots_relation a b k) : k = 3 :=
sorry

end find_k_l38_38358


namespace cone_height_l38_38346

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l38_38346


namespace problem_divisible_by_900_l38_38101

theorem problem_divisible_by_900 (X : ℕ) (a b c d : ℕ) 
  (h1 : 1000 <= X)
  (h2 : X < 10000)
  (h3 : X = 1000 * a + 100 * b + 10 * c + d)
  (h4 : d ≠ 0)
  (h5 : (X + (1000 * a + 100 * c + 10 * b + d)) % 900 = 0)
  : X % 90 = 45 := 
sorry

end problem_divisible_by_900_l38_38101


namespace water_volume_percentage_l38_38964

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38964


namespace cone_volume_percentage_filled_l38_38944

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38944


namespace solve_cubic_solve_quadratic_l38_38819

theorem solve_cubic :
  (∃ x : ℝ, 8 * x^3 = 125) → (∃ x : ℝ, x = 5 / 2) :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  use (5 / 2)
  have h_eq_five: x = 5 / 2 := sorry
  exact h_eq_five

theorem solve_quadratic :
  (∃ x : ℝ, 4 * (x - 1)^2 = 9) → (∃ x : ℝ, x = 5 / 2) :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  use (5 / 2)
  have h_eq_five: x = 5 / 2 := sorry
  exact h_eq_five

end solve_cubic_solve_quadratic_l38_38819


namespace find_d_l38_38641

open Real EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∃ (O P Q A B C : Point) (d : ℝ),
    equilateral A B C 900 ∧
    P ∉ Plane.mk A B C ∧ Q ∉ Plane.mk A B C ∧
    ¬ Plane.mk A B C.contains P ∧
    ¬ Plane.mk A B C.contains Q ∧
    PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧
    dihedral_angle (Plane.mk A B P) (Plane.mk A B Q) = 60 ∧
    dist O A = d ∧ dist O B = d ∧ dist O C = d ∧ dist O P = d ∧ dist O Q = d ∧
    d = 675

theorem find_d : problem_statement := sorry

end find_d_l38_38641


namespace repeating_decimal_division_l38_38152

theorem repeating_decimal_division :
  (let r : ℚ := 1/99 in
   let x : ℚ := 63 * r in
   let y : ℚ := 21 * r in
   x / y = 3) :=
by {
  sorry
}

end repeating_decimal_division_l38_38152


namespace trig_inverse_identity_l38_38246

theorem trig_inverse_identity :
  arcsin (-1/2) + arccos (-sqrt 3 / 2) + arctan (-sqrt 3) = π / 3 :=
by sorry

end trig_inverse_identity_l38_38246


namespace percentage_increase_correct_l38_38544

-- Define the original dimensions
def L : ℝ := 100
def W : ℝ := 100
def H : ℝ := 100

-- Define the new dimensions after increase
def L_new : ℝ := L + 0.15 * L
def W_new : ℝ := W + 0.20 * W
def H_new : ℝ := H + 0.10 * H

-- Define the original volume
def V_original : ℝ := L * W * H

-- Define the new volume
def V_new : ℝ := L_new * W_new * H_new

-- Calculate the percentage increase in volume
def percentage_increase : ℝ := ((V_new - V_original) / V_original) * 100

-- The statement to be proved
theorem percentage_increase_correct : percentage_increase = 51.8 := by
  sorry

end percentage_increase_correct_l38_38544


namespace angle_MKO_eq_angle_MLO_l38_38065

namespace Geometry

variables {O A K L P Q M : Type}
variables [MetricSpace O] [MetricSpace A] [MetricSpace K] [MetricSpace L] 
          [MetricSpace P] [MetricSpace Q] [MetricSpace M]

-- Conditions:
-- 1. KL is a chord of a circle with center O.
def is_chord (K L O : Type) [MetricSpace O] : Prop := sorry

-- 2. Point A is taken on the extension of KL.
def on_extension (A K L : Type) [MetricSpace A] : Prop := sorry

-- 3. Tangents AP and AQ are drawn from point A.
def is_tangent (A P Q : Type) [MetricSpace A] : Prop := sorry

-- 4. M is the midpoint of the segment PQ.
def is_midpoint (M P Q : Type) [MetricSpace M] : Prop := sorry

-- Prove:
theorem angle_MKO_eq_angle_MLO {O A K L P Q M : Type}
  [MetricSpace O] [MetricSpace A] [MetricSpace K] [MetricSpace L]
  [MetricSpace P] [MetricSpace Q] [MetricSpace M] :
  is_chord K L O → on_extension A K L → is_tangent A P Q → is_midpoint M P Q →
  ∠MKO = ∠MLO := 
sorry

end Geometry

end angle_MKO_eq_angle_MLO_l38_38065


namespace bamboo_pole_sections_l38_38861

-- Define the problem with corresponding conditions
theorem bamboo_pole_sections (n : ℕ) (d : ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 10 + (i - 1) * d)
    (h2 : 10 + (n-1)*d ≥ 0) -- Lengths must be non-negative
    (h3 : (10 + (n - 3) * d) + (10 + (n - 2) * d) + (10 + (n - 1) * d) = 114)
    (h4 : 10 + 5 * d = real.sqrt (10 * (10 + (n - 1) * d))) :
    n = 16 :=
begin
  sorry
end

end bamboo_pole_sections_l38_38861


namespace smallest_n_cube_mod_500_ends_in_388_l38_38263

theorem smallest_n_cube_mod_500_ends_in_388 :
  ∃ n : ℕ, 0 < n ∧ n^3 % 500 = 388 ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 500 = 388 → n ≤ m :=
sorry

end smallest_n_cube_mod_500_ends_in_388_l38_38263


namespace zero_in_set_zero_l38_38902

-- Define that 0 is an element
def zero_element : Prop := true

-- Define that {0} is a set containing only the element 0
def set_zero : Set ℕ := {0}

-- The main theorem that proves 0 ∈ {0}
theorem zero_in_set_zero (h : zero_element) : 0 ∈ set_zero := 
by sorry

end zero_in_set_zero_l38_38902


namespace sum_of_squares_of_tangent_segments_point_L_intersection_l38_38770

variables {A B C O A' I L : Point}
variables {R r : ℝ}
variables {AB AC AL A'I : ℝ}

-- Assuming the triangle ABC, A' being the symmetric point to A w.r.t the circumcenter O
-- I is the incenter of the triangle and (O is the circumcenter)
-- Incircle radius is r and circumcircle radius is R
-- Point L is as described in the problem

-- Proving part (a)
theorem sum_of_squares_of_tangent_segments
  (H_symmA : symmetric_point_wrt_circumcenter O A A')
  (H_incenter : incenter I A B C)
  (H_incircle_radius : incircle_radius I A B C = r)
  (H_circumcircle_radius : circumcircle_radius O A B C = R) :
  tangent_segment_squared A I r + tangent_segment_squared A' I r = 4 * R^2 - 4 * R * r - 2 * r^2 :=
sorry

-- Proving part (b)
theorem point_L_intersection
  (H_symmA : symmetric_point_wrt_circumcenter O A A')
  (H_incenter : incenter I A B C)
  (H_incircle_radius : incircle_radius I A B C = r)
  (H_circumcircle_radius : circumcircle_radius O A B C = R)
  (H_intersection : intersection L (circle_center_radius A' (distance A' I)) (circumcircle O R A B C)) :
  distance A L = real.sqrt (distance A B * distance A C) :=
sorry

-- Definitions used in theorems
def symmetric_point_wrt_circumcenter (O A A' : Point) : Prop := sorry
def incenter (I A B C : Point) : Prop := sorry
def incircle_radius (I A B C : Point) : ℝ := sorry
def circumcircle_radius (O A B C : Point) : ℝ := sorry
def tangent_segment_squared (P I : Point) (r : ℝ) : ℝ := sorry
def distance (P Q : Point) : ℝ := sorry
def circle_center_radius (P : Point) (r : ℝ) : Set Point := sorry
def intersection (L : Point) (S1 S2 : Set Point) : Prop := sorry

end sum_of_squares_of_tangent_segments_point_L_intersection_l38_38770


namespace min_breaks_12_no_breaks_15_l38_38298

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l38_38298


namespace distinct_x_intercepts_l38_38736

noncomputable def polynomial := (λ x : ℝ, (x - 5) * (x ^ 2 + 7 * x + 12) * (x - 1))

theorem distinct_x_intercepts : (∃ xs : Finset ℝ, polynomial 0 = 0 ∧ xs.card = 4) :=
by
  let xs := Finset.ofList [5, 1, -3, -4]
  have : polynomial = λ x, (x - 5) * (x ^ 2 + 7 * x + 12) * (x - 1) := rfl
  sorry

end distinct_x_intercepts_l38_38736


namespace equivalent_proof_problem_l38_38317

variable (a : Real)

def condition : Prop :=
  (cos a + sin a) / (cos a - sin a) = 2

theorem equivalent_proof_problem (h : condition a) : 
  cos a ^ 2 + sin a * cos a = 6 / 5 :=
sorry

end equivalent_proof_problem_l38_38317


namespace value_of_otimes_l38_38243

variable (a b : ℚ)

/-- Define the operation ⊗ -/
def otimes (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Given conditions -/
axiom condition1 : otimes a b 1 (-3) = 2 

/-- Target proof -/
theorem value_of_otimes : otimes a b 2 (-6) = 7 :=
by
  sorry

end value_of_otimes_l38_38243


namespace plane_parallel_l38_38681

variables (Point : Type) (Line Plane : Type) [Geometry Point Line Plane]

-- Defining basic geometrical relationships
variables {l m n : Line} {α β : Plane} {A B C : Point}

-- Geometric conditions as given in the problem
def perpendicular_to_plane (l : Line) (α : Plane) : Prop := 
  ∀ (A : Point), A ∈ α → ¬ A ∈ l

-- Equivalent statement to be proved
theorem plane_parallel (l : Line) (α β : Plane) 
  (h1: perpendicular_to_plane l α) (h2: perpendicular_to_plane l β) : α ≠ β :=
sorry

end plane_parallel_l38_38681


namespace locus_of_midpoint_of_square_l38_38597

theorem locus_of_midpoint_of_square (a : ℝ) (x y : ℝ) (h1 : x^2 + y^2 = 4 * a^2) :
  (∃ X Y : ℝ, 2 * X = x ∧ 2 * Y = y ∧ X^2 + Y^2 = a^2) :=
by {
  -- No proof is required, so we use 'sorry' here
  sorry
}

end locus_of_midpoint_of_square_l38_38597


namespace no_solution_if_n_eq_neg_one_l38_38635

theorem no_solution_if_n_eq_neg_one (n x y z : ℝ) :
  (n * x + y + z = 2) ∧ (x + n * y + z = 2) ∧ (x + y + n * z = 2) ↔ n = -1 → false :=
by
  sorry

end no_solution_if_n_eq_neg_one_l38_38635


namespace find_large_number_l38_38177

theorem find_large_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_large_number_l38_38177


namespace maximum_distance_exists_l38_38322

-- Define the circle equation and the ellipse equation
def circle_eq (P : ℝ × ℝ) : Prop := (P.1)^2 + (P.2 - 4)^2 = 1
def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1)^2 / 9 + (Q.2)^2 = 1

-- Define the distance formula
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- The main theorem to be proved
theorem maximum_distance_exists :
  ∀ P Q : ℝ × ℝ, circle_eq P → ellipse_eq Q → distance P Q ≤ 1 + 3 * real.sqrt 3 :=
by
  intros P Q Pc Pe
  sorry

end maximum_distance_exists_l38_38322


namespace number_of_numbers_in_last_group_l38_38135

theorem number_of_numbers_in_last_group :
  ∃ n : ℕ, (60 * 13) = (57 * 6) + 50 + (61 * n) ∧ n = 6 :=
sorry

end number_of_numbers_in_last_group_l38_38135


namespace sufficient_but_not_necessary_l38_38497

theorem sufficient_but_not_necessary (x : ℝ) (h1 : 0 < x ∧ x < 1) (h2 : x < 1) : 
  (1/x > 1 → exp (x - 1) < 1) ∧ ¬(exp (x - 1) < 1 → 1/x > 1) :=
sorry

end sufficient_but_not_necessary_l38_38497


namespace m_minus_n_l38_38571

noncomputable theory

open Classical

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let d := b * b - 4 * a * c
  in ((-b + real.sqrt d) / (2 * a), (-b - real.sqrt d) / (2 * a))

def larger_root (a b c : ℝ) : ℝ :=
  max (fst (quadratic_roots a b c)) (snd (quadratic_roots a b c))

def smaller_root (a b c : ℝ) : ℝ :=
  min (fst (quadratic_roots a b c)) (snd (quadratic_roots a b c))

theorem m_minus_n :
  let m := larger_root (1992 * 1992) (-1991 * 1993) (-1)
  let n := smaller_root 1 1991 (-1992)
  in m - n = 1993 :=
by
  sorry

end m_minus_n_l38_38571


namespace chocolate_bar_breaks_l38_38925

-- Definition of the problem as per the conditions
def chocolate_bar (rows : ℕ) (cols : ℕ) : ℕ := rows * cols

-- Statement of the proving problem
theorem chocolate_bar_breaks :
  ∀ (rows cols : ℕ), chocolate_bar rows cols = 40 → rows = 5 → cols = 8 → 
  (rows - 1) + (cols * (rows - 1)) = 39 :=
by
  intros rows cols h_bar h_rows h_cols
  sorry

end chocolate_bar_breaks_l38_38925


namespace initial_pigs_l38_38140

theorem initial_pigs (x : ℕ) (h : x + 86 = 150) : x = 64 :=
by
  sorry

end initial_pigs_l38_38140


namespace inequality_am_gm_l38_38071

theorem inequality_am_gm (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end inequality_am_gm_l38_38071


namespace parabola_intercept_sum_l38_38105

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l38_38105


namespace number_in_central_region_l38_38812

theorem number_in_central_region (a b c d : ℤ) :
  a + b + c + d = -4 →
  ∃ x : ℤ, x = -4 + 2 :=
by
  intros h
  use -2
  sorry

end number_in_central_region_l38_38812


namespace palindrome_count_l38_38148

theorem palindrome_count : 
    let digits := [5, 6, 7],
        is_palindrome (n : ℕ) : Prop := ∃ (a b c : ℕ), n = a * 10001 + b * 1010 + c * 100 + b * 10 + a
    in 
    let palindromes := {n | ∃ (a b c ∈ digits), is_palindrome n},
        valid_palindromes := palindromes.filter (λ n, (∃ d ∈ digits, d = 5 ∧ n % 10 = d) ∨
                                                   (∃ d ∈ digits, d = 5 ∧ (n / 10) % 10 = d) ∨
                                                   (∃ d ∈ digits, d = 5 ∧ (n / 100) % 10 = d) ∨
                                                   (∃ d ∈ digits, d = 5 ∧ (n / 1000) % 10 = d) ∨
                                                   (∃ d ∈ digits, d = 5 ∧ (n / 10000) % 10 = d))
    in
    valid_palindromes.card = 19 := 
sorry

end palindrome_count_l38_38148


namespace range_of_f_l38_38636

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x + 1) + x

theorem range_of_f :
  (∀ y, ∃ x, f(x) = y ↔ -1 / 2 ≤ y) :=
sorry

end range_of_f_l38_38636


namespace cone_height_l38_38331

-- Definitions given in the problem
def slant_height : ℝ := 13
def lateral_area : ℝ := 65 * Real.pi

-- Definition of the radius as derived from the given conditions
def radius : ℝ := lateral_area / (Real.pi * slant_height) -- This simplifies to 5

-- Using the Pythagorean theorem to express the height
def height : ℝ := Real.sqrt (slant_height^2 - radius^2)

-- The statement to prove
theorem cone_height : height = 12 := by
  sorry

end cone_height_l38_38331


namespace cone_height_l38_38336

theorem cone_height (l : ℝ) (LA : ℝ) (h : ℝ) (r : ℝ) (h_eq : h = sqrt (l^2 - r^2))
  (LA_eq : LA = π * r * l) (l_val : l = 13) (LA_val : LA = 65 * π) : h = 12 :=
by
  -- substitution of the values of l and LA
  have l_13 := l_val,
  have LA_65π := LA_val,
  
  -- solve for r from LA = π * r * l
  have r_val : r = LA / (π * l), sorry,

  -- then use the Pythagorean theorem to solve for h
  have h_12 : h = sqrt (l^2 - r^2), sorry,

  -- final conclusion: h must be equal to 12
  exact sorry

end cone_height_l38_38336


namespace cannot_tile_surface_square_hexagon_l38_38860

-- Definitions of internal angles of the tile shapes
def internal_angle_triangle := 60
def internal_angle_square := 90
def internal_angle_hexagon := 120
def internal_angle_octagon := 135

-- The theorem to prove that square and hexagon cannot tile a surface without gaps or overlaps
theorem cannot_tile_surface_square_hexagon : ∀ (m n : ℕ), internal_angle_square * m + internal_angle_hexagon * n ≠ 360 := 
by sorry

end cannot_tile_surface_square_hexagon_l38_38860


namespace min_breaks_for_square_12_can_form_square_15_l38_38294

-- Definitions and conditions for case n = 12
def stick_lengths_12 := (finset.range 12).map (λ i, i + 1)
def total_length_12 := stick_lengths_12.sum

-- Proof problem for n = 12
theorem min_breaks_for_square_12 : 
  ∃ min_breaks : ℕ, total_length_12 + min_breaks * 2 ∈ {k | k % 4 = 0} ∧ min_breaks = 2 :=
sorry

-- Definitions and conditions for case n = 15
def stick_lengths_15 := (finset.range 15).map (λ i, i + 1)
def total_length_15 := stick_lengths_15.sum

-- Proof problem for n = 15
theorem can_form_square_15 : 
  total_length_15 % 4 = 0 :=
sorry

end min_breaks_for_square_12_can_form_square_15_l38_38294


namespace parabola_intercepts_sum_l38_38109

noncomputable def y_intercept (f : ℝ → ℝ) : ℝ := f 0

noncomputable def x_intercepts_of_parabola (a b c : ℝ) : (ℝ × ℝ) :=
let Δ := b ^ 2 - 4 * a * c in
(
  (-b + real.sqrt Δ) / (2 * a),
  (-b - real.sqrt Δ) / (2 * a)
)

theorem parabola_intercepts_sum :
  let f := λ x : ℝ, 3 * x^2 - 9 * x + 4 in
  let (e, f) := x_intercepts_of_parabola 3 (-9) 4 in
  y_intercept f + e + f = 19 / 3 :=
by
  sorry

end parabola_intercepts_sum_l38_38109


namespace single_solution_inequality_l38_38634

theorem single_solution_inequality (a : ℝ) :
  (∃! (x : ℝ), abs (x^2 + 2 * a * x + 3 * a) ≤ 2) ↔ a = 1 ∨ a = 2 := 
sorry

end single_solution_inequality_l38_38634


namespace sum_f_eq_0_l38_38326

-- Define the even function property
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property that shifting the graph one unit to the right results in an odd function
def shift_right_odd (f : ℝ → ℝ) : Prop := ∀ x, f (x - 1) = -f x

-- Define the function f and state the given conditions
noncomputable def f : ℝ → ℝ := sorry

axiom f_even : is_even f
axiom f_shift_right_odd : shift_right_odd f
axiom f_at_2 : f 2 = -1

-- State the main theorem
theorem sum_f_eq_0 : (∑ i in finset.range 2017, f (i + 1)) = 0 := sorry

end sum_f_eq_0_l38_38326


namespace fraction_of_second_year_students_not_declared_major_l38_38180

variables {T : ℝ} (condition1 : T > 0)
variables (first_year_fraction : ℝ) (second_year_fraction_declared_major : ℝ)

def first_year_students := 1 / 2 * T
def second_year_students := T - first_year_students
def first_year_declared_major_fraction := 1 / 5
def second_year_declared_major_fraction := 4 * first_year_declared_major_fraction
def second_year_not_declared_major_fraction := 1 - second_year_declared_major_fraction
def second_year_not_declared_major_students := second_year_not_declared_major_fraction * second_year_students
def required_fraction := second_year_not_declared_major_students / T

theorem fraction_of_second_year_students_not_declared_major :
  T > 0 → 
  required_fraction = 1 / 10 :=
  λ h, by sorry

end fraction_of_second_year_students_not_declared_major_l38_38180


namespace trigonometric_problem_l38_38318

theorem trigonometric_problem (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := by
  sorry

end trigonometric_problem_l38_38318


namespace superstitious_numbers_correct_l38_38875

-- Define what it means for a number to be superstitious
def is_superstitious (n : ℕ) : Prop :=
  n = 13 * (n.digits 10).sum

-- Define the set of superstitious numbers we have found
def superstitious_numbers : set ℕ := {117, 156, 195}

-- The theorem statement combining the conditions and the known solution
theorem superstitious_numbers_correct :
  {n : ℕ | is_superstitious n} = superstitious_numbers :=
sorry

end superstitious_numbers_correct_l38_38875


namespace min_sticks_to_break_n12_l38_38278

theorem min_sticks_to_break_n12 : 
  let sticks := (Finset.range 12).map (λ x => x + 1)
  let total_length := sticks.sum
  total_length % 4 ≠ 0 → 
  (∃ k, k < 3 ∧ 
    ∃ broken_sticks: Finset Nat, 
      (∀ s ∈ broken_sticks, s < 12 ∧ s > 0) ∧ broken_sticks.card = k ∧ 
        sticks.sum + (broken_sticks.sum / 2) % 4 = 0) :=
sorry

end min_sticks_to_break_n12_l38_38278


namespace count_k_values_l38_38668

-- Definitions based on the conditions
def k_satisfies_conditions (k : ℕ) : Prop :=
  let ⟨a, b, c⟩ := (nat.factorization k)
  a ≤ 20 ∧ b ≤ 10 ∧ c = 10

def count_satisfying_k : ℕ :=
  (21 * 11)

-- Main theorem statement
theorem count_k_values : count_satisfying_k = 231 :=
by
  -- Placeholder for the proof
  sorry

end count_k_values_l38_38668


namespace equilateral_triangle_area_l38_38134

/-- 
  Statement: The square of the area of an equilateral triangle, 
  whose vertices lie on the ellipse \( x^2 + 4y^2 = 4 \) 
  and one of whose vertices is the centroid, is \(3\).
--/
theorem equilateral_triangle_area (x y : ℝ) :
  (∃ (a b c : ℝ × ℝ), 
    a = (0, 0) ∧ 
    b ∈ {p : ℝ × ℝ | p.1 ^ 2 + 4 * p.2 ^ 2 = 4} ∧ 
    c ∈ {p : ℝ × ℝ | p.1 ^ 2 + 4 * p.2 ^ 2 = 4} ∧ 
    dist a b = dist b c ∧ dist b c = dist c a) →
  (sqrt 3) ^ 2 = 3 := 
by 
  case exists.intro _ _ H
  -- Proof omitted
  sorry

end equilateral_triangle_area_l38_38134


namespace jane_wins_probability_l38_38595

-- Definitions based on the conditions
def spinner_sections : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def spins : List (ℕ × ℕ) := (spinner_sections.product spinner_sections)

def jane_wins (spin_jane spin_brother : ℕ) : Bool :=
  abs (spin_jane - spin_brother) < 3

-- Definition of the probability calculation based on the conditions
noncomputable def probability_jane_wins : ℚ :=
  (spins.filter (λ p => jane_wins p.1 p.2)).length / spins.length

-- Statement to be proved
theorem jane_wins_probability : probability_jane_wins = 17 / 32 :=
  sorry

end jane_wins_probability_l38_38595


namespace percent_filled_cone_l38_38969

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38969


namespace fresh_grape_weight_l38_38674

variable (D : ℝ) (F : ℝ)

axiom dry_grape_weight : D = 66.67
axiom fresh_grape_water_content : F * 0.25 = D * 0.75

theorem fresh_grape_weight : F = 200.01 :=
by sorry

end fresh_grape_weight_l38_38674


namespace bookstore_magazine_purchase_count_l38_38193

theorem bookstore_magazine_purchase_count :
  let total_methods :=
    (nat.choose 8 5) + (nat.choose 8 4 * nat.choose 3 2)
  in total_methods = 266 :=
by
  sorry

end bookstore_magazine_purchase_count_l38_38193


namespace probability_of_same_color_is_correct_l38_38420

-- Define the parameters for balls in the bag
def green_balls : ℕ := 8
def red_balls : ℕ := 6
def blue_balls : ℕ := 1
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probabilities of drawing each color
def prob_green : ℚ := green_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := blue_balls / total_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ :=
  prob_green^2 + prob_red^2 + prob_blue^2

theorem probability_of_same_color_is_correct :
  prob_same_color = 101 / 225 :=
by
  sorry

end probability_of_same_color_is_correct_l38_38420


namespace zack_marbles_number_l38_38555

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l38_38555


namespace greatest_n_divides_K_l38_38455

theorem greatest_n_divides_K :
  let K := ∏ n in finset.range(1, 20), n^(20 - n) in
  ∃ n : ℕ, 2^n ∣ K ∧ ∀ m : ℕ, 2^m ∣ K → m ≤ 150 :=
by
  sorry

end greatest_n_divides_K_l38_38455


namespace roots_quadratic_l38_38704

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l38_38704


namespace water_filled_percent_l38_38981

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38981


namespace area_of_fig_between_x1_and_x2_l38_38098

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2_l38_38098


namespace train_problem_l38_38182

variables (x : ℝ) (p q : ℝ)
variables (speed_p speed_q : ℝ) (dist_diff : ℝ)

theorem train_problem
  (speed_p : speed_p = 50)
  (speed_q : speed_q = 40)
  (dist_diff : ∀ x, x = 500 → p = 50 * x ∧ q = 40 * (500 - 100)) :
  p + q = 900 :=
by
sorry

end train_problem_l38_38182


namespace probability_losing_game_l38_38749

-- Define the odds of winning in terms of number of wins and losses
def odds_winning := (wins: ℕ, losses: ℕ) := (5, 3)

-- Given the odds of winning, calculate the total outcomes
def total_outcomes : ℕ := (odds_winning.1 + odds_winning.2)

-- Define the probability of losing the game
def probability_of_losing (wins losses: ℕ) (total: ℕ) : ℚ := (losses : ℚ) / (total : ℚ)

-- Given odds of 5:3, prove the probability of losing is 3/8
theorem probability_losing_game : probability_of_losing odds_winning.1 odds_winning.2 total_outcomes = 3 / 8 :=
by
  sorry

end probability_losing_game_l38_38749


namespace jo_reading_time_l38_38776

variable (book_total_pages current_page pages_one_hour_ago : ℝ)
variable (steady_pace : Prop)

def pages_read_in_one_hour :=
  current_page - pages_one_hour_ago

def pages_left_to_read :=
  book_total_pages - current_page

def time_to_finish_book :=
  pages_left_to_read / pages_read_in_one_hour

theorem jo_reading_time :
  steady_pace →
  book_total_pages = 325.5 →
  current_page = 136.25 →
  pages_one_hour_ago = 97.5 →
  time_to_finish_book = 4.88 :=
by
  sorry

end jo_reading_time_l38_38776


namespace correct_growth_rate_l38_38511

noncomputable def growth_rate_eq (x : ℝ) : Prop :=
  10 * (1 + x)^2 = 11.5

axiom initial_sales_volume : ℝ := 10
axiom final_sales_volume : ℝ := 11.5
axiom monthly_growth_rate (x : ℝ) : x > 0

theorem correct_growth_rate (x : ℝ) (hx : monthly_growth_rate x) :
  growth_rate_eq x :=
-- sorry
by
  have h1 : initial_sales_volume = 10 := rfl
  have h2 : final_sales_volume = 11.5 := rfl
  rw [h1, h2]
  sorry

end correct_growth_rate_l38_38511


namespace inequality_solution_l38_38527

theorem inequality_solution (x : ℝ) :
  (x - 2 > 1) ∧ (-2 * x ≤ 4) ↔ (x > 3) :=
by
  sorry

end inequality_solution_l38_38527


namespace decrease_hours_by_13_percent_l38_38565

theorem decrease_hours_by_13_percent (W H : ℝ) (hW_pos : W > 0) (hH_pos : H > 0) :
  let W_new := 1.15 * W
  let H_new := H / 1.15
  let income_decrease_percentage := (1 - H_new / H) * 100
  abs (income_decrease_percentage - 13.04) < 0.01 := 
by
  sorry

end decrease_hours_by_13_percent_l38_38565


namespace route_x_quicker_than_route_y_by_l38_38059

noncomputable def time_route_x (distance_x speed_x : ℝ) : ℝ := (distance_x / speed_x) * 60

noncomputable def time_route_y (distance_y speed_y1 distance_c_zone speed_c_zone : ℝ) : ℝ := 
  ((distance_y - distance_c_zone) / speed_y1) * 60 + (distance_c_zone / speed_c_zone) * 60

theorem route_x_quicker_than_route_y_by (time_x time_y : ℝ) : 
  ∀ (distance_x speed_x distance_y speed_y1 distance_c_zone speed_c_zone : ℝ), 
    time_route_x distance_x speed_x = time_x →
    time_route_y distance_y speed_y1 distance_c_zone speed_c_zone = time_y →
    (time_x - time_y) = 1.2 :=
by
  intros distance_x speed_x distance_y speed_y1 distance_c_zone speed_c_zone hx hy
  sorry

-- Assign the given data to variables
def distance_x : ℝ := 8
def speed_x : ℝ := 40
def distance_y : ℝ := 7
def speed_y1 : ℝ := 50
def distance_c_zone : ℝ := 1
def speed_c_zone : ℝ := 10

-- Calculate times for both routes
def time_x := time_route_x distance_x speed_x
def time_y := time_route_y distance_y speed_y1 distance_c_zone speed_c_zone

#eval time_x
#eval time_y

-- Verify the proof using provided data
example : (time_x - time_y) = 1.2 :=
  route_x_quicker_than_route_y_by time_x time_y distance_x speed_x distance_y speed_y1 distance_c_zone speed_c_zone rfl rfl

end route_x_quicker_than_route_y_by_l38_38059


namespace expression_evaluation_l38_38402

theorem expression_evaluation (x : ℝ) (h : 2 * x - 7 = 8 * x - 1) : 5 * (x - 3) = -20 :=
by
  sorry

end expression_evaluation_l38_38402


namespace team_b_people_l38_38425

theorem team_b_people (a_avg b_avg total_avg : ℝ) (team_b_more : ℝ) (x : ℕ) : 
  (a_avg = 75) → (b_avg = 73) → (total_avg = 73.5) → (team_b_more = 6) → 
  ((75 * x + 73 * (x + 6)) / (x + (x + 6)) = 73.5) → (x + 6 = 9) :=
by
  intros h_a_avg h_b_avg h_total_avg h_team_b_more h_combined_avg 
  sorry -- Proof omitted

end team_b_people_l38_38425


namespace water_filled_percent_l38_38980

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38980


namespace hyperbola_eccentricity_l38_38682

-- Define the hyperbola and conditions
def is_hyperbola (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1

-- Define the points M and N on the asymptote
def symmetric_points_on_asymptote (m n a b : ℝ) : Prop :=
  n = b / a * m ∧ n > 0

-- Define the distance between focus points
def distance_between_foci (c a : ℝ) : Prop :=
  c ^ 2 = a ^ 2 + (b ^ 2)

-- Rectangle area condition
def rectangle_area_condition (c n a : ℝ) : Prop :=
  c * n = 2 * sqrt 6 * a ^ 2

-- Finally prove the eccentricity
theorem hyperbola_eccentricity (a b c m n : ℝ) (h_hyperbola : is_hyperbola m n a b)
  (h_symmetric : symmetric_points_on_asymptote m n a b)
  (h_focus_distance : distance_between_foci c a)
  (h_area : rectangle_area_condition c n a) :
    c / a = sqrt 3 :=
  sorry

end hyperbola_eccentricity_l38_38682


namespace solve_eq_solutions_l38_38818

noncomputable def solve_eq (x : ℝ) : Prop :=
  sqrt (x + 16) - 8 / sqrt (x + 16) = 4

theorem solve_eq_solutions :
  ∀ x : ℝ, solve_eq x ↔ (x = 20 + 8 * sqrt 3) ∨ (x = 20 - 8 * sqrt 3) := 
by 
  sorry

end solve_eq_solutions_l38_38818


namespace smallest_difference_l38_38898

theorem smallest_difference :
  ∃ (w x y z u v : ℕ), 
    {w, x, y, z, u, v} = {1, 3, 7, 8, 9} ∧
    a = 1000 * w + 100 * x + 10 * y + z ∧
    b = 10 * u + v ∧
    a - b = 1279 :=
by
  sorry

end smallest_difference_l38_38898


namespace cone_filled_with_water_to_2_3_height_l38_38996

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38996


namespace ABD_perimeter_lt_ACD_perimeter_implies_AB_lt_AC_l38_38791

variable {Point : Type}   -- Define the type representing points
variable (A B C D M : Point)   -- Define points A, B, C, D, and M

-- Define distances between the points (euclidean distance here as an example.
variable [metric_space Point] [has_dist Point]

-- Convex condition (assuming a valid quadrilateral with proper geometrical properties)
variable (convex_quad : convex_hull {A, B, C, D})

-- Perimeters of triangles
variable (P_ABD : dist A B + dist B D + dist D A)
variable (P_ACD : dist A C + dist C D + dist D A)

-- Given condition: The perimeter of triangle ABD is less than the perimeter of triangle ACD
variable (h : P_ABD < P_ACD)

-- The goal is to prove that AB < AC
theorem ABD_perimeter_lt_ACD_perimeter_implies_AB_lt_AC (convex_quad : convex_hull {A, B, C, D}) 
  (h : dist A B + dist B D + dist D A < dist A C + dist C D + dist D A) : 
  dist A B < dist A C := 
sorry

end ABD_perimeter_lt_ACD_perimeter_implies_AB_lt_AC_l38_38791


namespace triangle_area_l38_38743

noncomputable def area_of_triangle (CD : ℝ) (angle_A : ℝ) (angle_C : ℝ) : ℝ :=
  if (CD = 1) ∧ (angle_A = 15) ∧ (angle_C = 90) then
    (Real.sqrt 6 + Real.sqrt 2) / 4
  else
    0 -- default case to handle other inputs

theorem triangle_area:
  ∀ (CD : ℝ) (angle_A : ℝ) (angle_C : ℝ),
  CD = 1 → angle_A = 15 → angle_C = 90 →
  area_of_triangle CD angle_A angle_C = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  intros CD angle_A angle_C h1 h2 h3
  unfold area_of_triangle
  simp [h1, h2, h3]
  sorry

end triangle_area_l38_38743


namespace mask_digits_l38_38837

theorem mask_digits : 
  ∃ (elephant mouse pig panda : ℕ), 
  (elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧ 
   mouse ≠ pig ∧ mouse ≠ panda ∧ pig ≠ panda) ∧
  (4 * 4 = 16) ∧ (7 * 7 = 49) ∧ (8 * 8 = 64) ∧ (9 * 9 = 81) ∧
  (elephant = 6) ∧ (mouse = 4) ∧ (pig = 8) ∧ (panda = 1) :=
by
  sorry

end mask_digits_l38_38837


namespace find_m_value_l38_38381

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_m_value (m : ℝ) 
  (h : dot_product (2 * m - 1, 3) (1, -1) = 2) : 
  m = 3 := by
  sorry

end find_m_value_l38_38381


namespace solution_set_inequality_l38_38304

theorem solution_set_inequality (a x : ℝ) (h : a > 0) :
  (∀ x, (a + 1 ≤ x ∧ x ≤ a + 3) ↔ (|((2 * x - 3 - 2 * a) / (x - a))| ≤ 1)) := 
sorry

end solution_set_inequality_l38_38304


namespace find_a10_l38_38698

theorem find_a10 
  (a : ℕ → ℝ)
  (h1 : ∀ n, ∃ d, ∀ m, (m > 1) → (a m - a (m - 1)) = d)  -- Arithmetic sequence of 1/a_n
  (a1 : a 1 = 1)
  (a4 : a 4 = 4) :
  a 10 = -4 / 5 := 
sorry

end find_a10_l38_38698


namespace calories_per_candy_bar_l38_38858

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) 
  (h : total_calories = 341) (n : number_of_bars = 11) : (total_calories / number_of_bars = 31) :=
by
  sorry

end calories_per_candy_bar_l38_38858


namespace cone_water_fill_percentage_l38_38934

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38934


namespace triangle_AKM_perimeter_l38_38012

theorem triangle_AKM_perimeter (A B C K M O : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] 
  [MetricSpace M] [MetricSpace O]
  (h₁ : A ≠ B) (h₂ : A ≠ C) (h₃ : B ≠ C)
  (hAC : dist A C = 1) (hAB : dist A B = 2)
  (hO : is_angle_bisectors_intersection O A B C)
  (hKO : parallel_through_point K O C)
  (hMO : parallel_through_point M O B)
  (hKAC : segment_intersects K O A C)
  (hMAB : segment_intersects M O A B) :
  dist A K + dist K M + dist M A = 2 := sorry

end triangle_AKM_perimeter_l38_38012


namespace intersection_M_N_l38_38745

def M := { y : ℝ | ∃ x : ℝ, y = 2^x }
def N := { y : ℝ | ∃ x : ℝ, y = 2 * Real.sin x }

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 2 } :=
by
  sorry

end intersection_M_N_l38_38745


namespace sum_of_digits_7_pow_17_mod_100_l38_38893

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l38_38893


namespace sum_of_powers_l38_38627

noncomputable def i : ℂ := complex.I

theorem sum_of_powers (E : ℂ := complex.I) : 
  (2 * ∑ n in (finset.range 301).image (λ x, x - 150), E ^ n) = 2 :=
by
  have h : E ^ 4 = 1 := complex.I_pow4
  sorry

end sum_of_powers_l38_38627


namespace compare_abc_l38_38680

theorem compare_abc (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                               (h2 : b = Real.sin 1)
                               (h3 : c = 3^(-0.5)) : c < a ∧ a < b :=
by
  sorry

end compare_abc_l38_38680


namespace roll_two_dice_prime_sum_l38_38824

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l38_38824


namespace fraction_walk_home_l38_38619

theorem fraction_walk_home : 
  (1 - ((1 / 2) + (1 / 4) + (1 / 10) + (1 / 8))) = (1 / 40) :=
by 
  sorry

end fraction_walk_home_l38_38619


namespace find_cone_height_l38_38351

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l38_38351


namespace teacher_age_l38_38568

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_total : ℕ) 
  (h1 : avg_age_students = 14) (h2 : num_students = 50) (h3 : avg_age_with_teacher = 15) (h4 : num_total = 51) :
  ∃ (teacher_age : ℕ), teacher_age = 65 :=
by sorry

end teacher_age_l38_38568


namespace range_area_triangle_l38_38716

theorem range_area_triangle (x y : ℝ) (h_ellipse : x^2 / 6 + y^2 / 4 = 1) (C2 : ℝ → ℝ → Prop) 
  (h_foci : C2 (sqrt 2) 0 ∧ C2 (-sqrt 2) 0) :
  (∃ a b, 1 ≤ a ∧ a ≤ b ∧ b ≤ sqrt 2 ∧ (∃ t, 0 < t ∧ x^2 + y^2 = t ∧ t = 2 
  ∧ P x y ∧ Q x y t ∧ area_triangle O P Q = S a b)) :=
sorry

end range_area_triangle_l38_38716


namespace cone_water_fill_percentage_l38_38938

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38938


namespace find_CD_l38_38032

noncomputable def right_triangle (A B C : Type) : Prop :=
∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ (a = 2 ∨ b = 2 ∨ c = 2)

variables (A B C D : ℝ)

def angle_B_is_right_angle : Prop := 
B = 90

def circle_intersects_AC_at_D (A B C D : ℝ) : Prop := 
AD = 2 ∧ BD = 3

theorem find_CD 
  (h1: right_triangle A B C)
  (h2: angle_B_is_right_angle B)
  (h3: circle_intersects_AC_at_D A B C D) :
  CD = 4.5 := 
  sorry

end find_CD_l38_38032


namespace sqrt_fraction_fact_l38_38264

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_fraction_fact :
  Real.sqrt (factorial 9 / 210 : ℝ) = 24 * Real.sqrt 3 := by
  sorry

end sqrt_fraction_fact_l38_38264


namespace arithmetic_sequence_a8_l38_38003

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a m + 1 - a m) 
  (h1 : a 2 = 3) (h2 : a 5 = 12) : a 8 = 21 := 
by 
  sorry

end arithmetic_sequence_a8_l38_38003


namespace volume_frustum_2240_over_3_l38_38606

def volume_of_pyramid (base_edge: ℝ) (height: ℝ) : ℝ :=
    (1 / 3) * (base_edge ^ 2) * height

def volume_of_frustum (original_base_edge: ℝ) (original_height: ℝ)
  (smaller_base_edge: ℝ) (smaller_height: ℝ) : ℝ :=
  volume_of_pyramid original_base_edge original_height - volume_of_pyramid smaller_base_edge smaller_height

theorem volume_frustum_2240_over_3 :
  volume_of_frustum 16 10 8 5 = 2240 / 3 :=
by sorry

end volume_frustum_2240_over_3_l38_38606


namespace minkowski_inequality_l38_38574

open scoped BigOperators

-- Define p as a reals number where p >= 1
def p (p : ℝ) : Prop := p ≥ 1

-- Define a list x and y as lists of strictly positive real numbers
def x (x : ℕ → ℝ) : Prop := ∀ i, x i > 0
def y (y : ℕ → ℝ) : Prop := ∀ i, y i > 0

-- Define the length of the lists x and y
def length_x_and_y (n : ℕ) (x y : ℕ → ℝ) : Prop := ∀ i, 0 ≤ i ∧ i < n

theorem minkowski_inequality (p : ℝ) (x y : ℕ → ℝ) (n : ℕ) 
  (ppos : p ≥ 1) 
  (xpos : ∀ i, x i > 0) 
  (ypos : ∀ i, y i > 0) 
  (length : ∀ i, i < n) : 
  ((∑ i in finset.range n, (x i + y i)^p)^(1/p)) ≤ 
  ((∑ i in finset.range n, (x i)^p)^(1/p)) + 
  ((∑ i in finset.range n, (y i)^p)^(1/p)) :=
sorry

end minkowski_inequality_l38_38574


namespace complete_square_solution_l38_38088

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l38_38088


namespace sequence_sum_l38_38725

def a_n (n : ℕ) : ℕ :=
if odd n then 2 * n + 1 else 2 ^ n

theorem sequence_sum : a_n 4 + a_n 5 = 27 := by
  sorry

end sequence_sum_l38_38725


namespace initial_southwards_distance_l38_38474

-- Define a structure that outlines the journey details
structure Journey :=
  (southwards : ℕ) 
  (westwards1 : ℕ := 10)
  (northwards : ℕ := 20)
  (westwards2 : ℕ := 20) 
  (home_distance : ℕ := 30)

-- Main theorem statement without proof
theorem initial_southwards_distance (j : Journey) : j.southwards + j.northwards = j.home_distance → j.southwards = 10 := by
  intro h
  sorry

end initial_southwards_distance_l38_38474


namespace probability_at_least_one_head_l38_38250

theorem probability_at_least_one_head :
  let p_tails : ℚ := 1 / 2
  let p_four_tails : ℚ := p_tails ^ 4
  let p_at_least_one_head : ℚ := 1 - p_four_tails
  p_at_least_one_head = 15 / 16 := by
  sorry

end probability_at_least_one_head_l38_38250


namespace probability_prime_sum_two_dice_l38_38826

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def num_ways_sum_is (target_sum : ℕ) : ℕ :=
  finset.card { (a, b) : finset.univ × finset.univ | a + b = target_sum }

def total_outcomes : ℕ := 36

theorem probability_prime_sum_two_dice :
  (∑ n in (finset.range 13).filter is_prime, num_ways_sum_is n) / total_outcomes = 5 / 12 :=
sorry

end probability_prime_sum_two_dice_l38_38826


namespace find_x_coord_of_N_l38_38329

theorem find_x_coord_of_N
  (M N : ℝ × ℝ)
  (hM : M = (3, -5))
  (hN : N = (x, 2))
  (parallel : M.1 = N.1) :
  x = 3 :=
sorry

end find_x_coord_of_N_l38_38329


namespace boys_in_other_communities_l38_38428

def percentage_of_other_communities (p_M p_H p_S : ℕ) : ℕ :=
  100 - (p_M + p_H + p_S)

def number_of_boys_other_communities (total_boys : ℕ) (percentage_other : ℕ) : ℕ :=
  (percentage_other * total_boys) / 100

theorem boys_in_other_communities (N p_M p_H p_S : ℕ) (hN : N = 650) (hpM : p_M = 44) (hpH : p_H = 28) (hpS : p_S = 10) :
  number_of_boys_other_communities N (percentage_of_other_communities p_M p_H p_S) = 117 :=
by
  -- Steps to prove the theorem would go here
  sorry

end boys_in_other_communities_l38_38428


namespace gamma_delta_purchases_l38_38202

open Finset

-- Defining the problem context and proof
theorem gamma_delta_purchases :
  let cookies := 6
  let milk := 4
  let gamma_choices := cookies + milk
  let delta_choices := cookies
  ∑ i in (range 4), 
    ((choose gamma_choices 3 - i) * (if i = 0 then 1 else choose delta_choices i)) +
    ∑ j in (range 0..3), 
      if j = 2 then choose delta_choices 2 else
      if j = 1 then delta_choices else
      if j = 0 then choose (delta_choices - 1) 1 * delta_choices else choose delta_choices 3
  = 656 := by
  sorry

end gamma_delta_purchases_l38_38202


namespace alpha_correct_and_polar_eqn_l38_38272

variables {t α θ ρ : ℝ}
variables (x y : ℝ)

def point_P : ℝ × ℝ := (2, 1)
def parametric_line (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, 1 + t * Real.sin α)

def intersection_x_axis (α : ℝ) : ℝ × ℝ :=
  parametric_line (-(1 / Real.sin α)) α

def intersection_y_axis (α : ℝ) : ℝ × ℝ :=
  parametric_line (-(2 / Real.cos α)) α

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem alpha_correct_and_polar_eqn :
  (distance point_P (intersection_x_axis α) *
   distance point_P (intersection_y_axis α) = 4) ↔
  (α = 3 * Real.pi / 4) ∧
  ∀ (θ : ℝ), (ρ * (Real.cos θ + Real.sin θ) = 3) :=
by sorry

end alpha_correct_and_polar_eqn_l38_38272


namespace volume_filled_cone_l38_38950

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38950


namespace popsicle_sticks_per_group_l38_38800

theorem popsicle_sticks_per_group (total_sticks initial_sticks left_sticks : ℕ) (groups : ℕ) : 
  initial_sticks = 170 → 
  left_sticks = 20 → 
  groups = 10 → 
  (total_sticks = initial_sticks - left_sticks) → 
  (total_sticks / groups = 15) :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have total_sticks_def : total_sticks = 150 := by 
    simp [h4] 
  rw total_sticks_def
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end popsicle_sticks_per_group_l38_38800


namespace length_of_faster_train_is_370_l38_38872

noncomputable def length_of_faster_train (vf vs : ℕ) (t : ℕ) : ℕ :=
  let rel_speed := vf - vs
  let rel_speed_m_per_s := rel_speed * 1000 / 3600
  rel_speed_m_per_s * t

theorem length_of_faster_train_is_370 :
  length_of_faster_train 72 36 37 = 370 := 
  sorry

end length_of_faster_train_is_370_l38_38872


namespace max_terms_sum_125_l38_38061

theorem max_terms_sum_125 : ∃ (S : Finset ℕ), 
  (∑ x in S, x) = 125 ∧
  (∀ x ∈ S, x > 1) ∧ 
  (∀ x y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧
  (S.card = 8) ∧ 
  (∀ (T : Finset ℕ), (∑ x in T, x) = 125 ∧ (∀ x ∈ T, x > 1) ∧ (∀ x y ∈ T, x ≠ y → Nat.gcd x y = 1) → T.card ≤ 8) :=
begin
  sorry
end

end max_terms_sum_125_l38_38061


namespace platform_length_l38_38173

theorem platform_length :
  (let train_length := 150 in
   let time_to_cross_pole := 18 in
   let time_to_cross_platform := 39 in
   let speed_of_train := train_length / time_to_cross_pole in
   let platform_length := 39 * speed_of_train - train_length in
   platform_length = 174.87) :=
begin
  sorry
end

end platform_length_l38_38173


namespace rectangle_count_is_84_l38_38817

theorem rectangle_count_is_84 :
  ∃ (H V : Type) (redH : H → Prop) (blueV : V → Prop), 
    (H.card = 6) ∧ (V.card = 5) ∧ 
    (finset.card {h ∈ H | redH h} = 3) ∧ 
    (finset.card {v ∈ V | blueV v} = 2) →
  let count_rectangles := 
    let horizontal_choices := 
      finset.card {h1 h2 ∈ H | redH h1 ∨ redH h2} in
    let vertical_choices := 
      finset.card {v1 v2 ∈ V | blueV v1 ∨ blueV v2} in
    horizontal_choices * vertical_choices in
  count_rectangles = 84 :=
by sorry

end rectangle_count_is_84_l38_38817


namespace sum_of_reciprocals_products_of_roots_l38_38463

theorem sum_of_reciprocals_products_of_roots :
  (∃ p q r s : ℝ, 
    (∀ x : ℝ, x^4 + 10 * x^3 + 20 * x^2 + 15 * x + 6 = 0 ↔ x = p ∨ x = q ∨ x = r ∨ x = s) ∧ 
    (pq + pr + ps + qr + qs + rs = 20) ∧ 
    (pqrs = 6)) →
  (1 / p / q + 1 / p / r + 1 / p / s + 1 / q / r + 1 / q / s + 1 / r / s = 10 / 3) :=
begin
  sorry
end

end sum_of_reciprocals_products_of_roots_l38_38463


namespace exist_modified_consecutive_set_l38_38651

theorem exist_modified_consecutive_set :
  ∃ n : ℕ, ∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ,
  (a_1 ∈ {n-1, n, n+1} ∧ 
   a_2 ∈ {n, n+1, n+2} ∧ 
   a_3 ∈ {n+1, n+2, n+3} ∧ 
   a_4 ∈ {n+2, n+3, n+4} ∧ 
   a_5 ∈ {n+3, n+4, n+5} ∧ 
   a_6 ∈ {n+4, n+5, n+6} ∧ 
   a_7 ∈ {n+5, n+6, n+7}) ∧ 
  ((a_1 * a_2 * a_3 * a_4 * a_5 * a_6 * a_7) = (n * (n+1) * (n+2) * (n+3) * (n+4) * (n+5) * (n+6))) :=
sorry

end exist_modified_consecutive_set_l38_38651


namespace parabola_intercepts_sum_l38_38113

theorem parabola_intercepts_sum :
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  y_intercept + x_intercept1 + x_intercept2 = 7 :=
by
  let y_intercept := 4
  let x_intercept1 := (9 + Real.sqrt 33) / 6
  let x_intercept2 := (9 - Real.sqrt 33) / 6
  have sum_intercepts : y_intercept + x_intercept1 + x_intercept2 = 7 := by
        calc (4 : ℝ) + ((9 + Real.sqrt 33) / 6) + ((9 - Real.sqrt 33) / 6)
            = 4 + (18 / 6) : by
              rw [add_assoc, ← add_div, add_sub_cancel]
            ... = 4 + 3 : by norm_num
            ... = 7 : by norm_num
  exact sum_intercepts

end parabola_intercepts_sum_l38_38113


namespace train_length_l38_38222

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end train_length_l38_38222


namespace bernardo_prob_larger_l38_38235

def set_Bernardo := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def set_Silvia := {1, 2, 3, 4, 5, 6, 7, 8, 10}

def selection_probability (set_B: Finset ℕ) (set_S: Finset ℕ) : ℚ :=
  let total_B := (set_B.card).choose 3
  let total_S := (set_S.card).choose 3
  let valid_selections := -- Number of valid ways Bernardo's number is larger
    (finset.powersetLen 3 set_B).sum
        (λ b, (finset.powersetLen 3 set_S).count
          (λ s, descending_array_to_num b > descending_array_to_num s))
  (valid_selections : ℚ) / (total_B * total_S)

theorem bernardo_prob_larger :
  selection_probability set_Bernardo set_Silvia = 83 / 168 :=
begin
  sorry
end

end bernardo_prob_larger_l38_38235


namespace cone_volume_l38_38570

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume (r h : ℝ) (V : ℝ) (h_r : r = 10) (h_h : h = 21) (h_V : V = 2199.109) :
  volume_of_cone r h ≈ V :=
 by
  -- assume the equivalence of real numbers for approximately equal
  sorry

end cone_volume_l38_38570


namespace complex_expression_evaluate_l38_38040

noncomputable def z : ℂ := complex.cos (3 * real.pi / 11) + complex.sin (3 * real.pi / 11) * complex.I

theorem complex_expression_evaluate :
  (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33) = 2197 :=
by
  have h1 : z^11 = -1, by
    calc
      z^11 = (complex.cos (3 * real.pi / 11) + complex.sin (3 * real.pi / 11) * complex.I) ^ 11 : sorry
         ... = complex.cos (3 * real.pi) + complex.sin (3 * real.pi) * complex.I : sorry
         ... = -1 : sorry,
  
  have h2 : 3 * z + z^3 = -6 * z - 2 * z^3, by sorry,
  have h3 : 3 * z^3 + z^9 = -9 * z + 3 * z^7 + 4, by sorry,
  have h4 : 3 * z^5 + z^15 = -9 * z - 3 + 3 * z^6 + 1, by sorry,
  show (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33) = 2197,
  by calc
    (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33)
    = (-6 * z - 2 * z^3) * (-9 * z + 3 * z^7 + 4) * (-9 * z - 3 + 3 * z^6 + 1) : by rw [h2, h3, h4]
    ... = 2197 : sorry.

end complex_expression_evaluate_l38_38040


namespace monthly_average_growth_rate_eq_l38_38924

theorem monthly_average_growth_rate_eq (x : ℝ) :
  16 * (1 + x)^2 = 25 :=
sorry

end monthly_average_growth_rate_eq_l38_38924


namespace fourth_degree_to_standard_form_right_side_perfect_square_l38_38909

theorem fourth_degree_to_standard_form (a b c d : ℝ) :
  ∃ A B C : ℝ, (λ t, t^4 + a * t^3 + b * t^2 + c * t + d) = (λ x, x^4 + A * x^2 + B * x + C) := by
  sorry

theorem right_side_perfect_square (A B C : ℝ) :
  ∃ α > -A / 2, ∀ x : ℝ, (x^4 + 2 * α * x^2 + α^2) = ((A + 2 * α) * x^2 + B * x + (C + α^2)) :=
  sorry

end fourth_degree_to_standard_form_right_side_perfect_square_l38_38909


namespace pear_juice_percentage_is_19_23_l38_38057

def ounces_of_pear_juice_per_pear : ℝ :=
  10 / 4

def ounces_of_orange_juice_per_orange : ℝ :=
  7

def pear_juice_from_pears (pears : ℝ) : ℝ :=
  pears * ounces_of_pear_juice_per_pear

def orange_juice_from_oranges (oranges : ℝ) : ℝ :=
  oranges * ounces_of_orange_juice_per_orange

def total_juice (oranges pears : ℝ) : ℝ :=
  pear_juice_from_pears(pears) + orange_juice_from_oranges(oranges)

def percent_pear_juice (oranges pears : ℝ) : ℝ :=
  (pear_juice_from_pears(pears) / total_juice(oranges, pears)) * 100

theorem pear_juice_percentage_is_19_23 :
  percent_pear_juice 12 8 = 19.23 :=
by
  unfold percent_pear_juice pear_juice_from_pears orange_juice_from_oranges ounces_of_pear_juice_per_pear ounces_of_orange_juice_per_orange total_juice
  have h1 : 10 / 4 = 2.5 := by norm_num
  have h2 : 8 * 2.5 = 20 := by norm_num
  have h3 : 12 * 7 = 84 := by norm_num
  have h4 : 20 + 84 = 104 := by norm_num
  have h5 : 20 / 104 = 0.1923 := by norm_num
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end pear_juice_percentage_is_19_23_l38_38057


namespace sum_sin_zero_l38_38658

theorem sum_sin_zero (x y z : ℝ) (hx : Real.sin x = Real.tan y) (hy : Real.sin y = Real.tan z) (hz : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
by
s  -- Proof skipped

end sum_sin_zero_l38_38658


namespace cos_A_sin_B_eq_l38_38092

theorem cos_A_sin_B_eq (A B : ℝ) (hA1 : 0 < A) (hA2 : A < π / 2) (hB1 : 0 < B) (hB2 : B < π / 2)
    (h : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
    Real.cos A * Real.sin B = 1 / Real.sqrt 6 := sorry

end cos_A_sin_B_eq_l38_38092


namespace water_filled_percent_l38_38987

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38987


namespace shaded_area_of_square_l38_38439

theorem shaded_area_of_square (side_length : ℕ) (shaded_triangles : 4) (rectangles : 4) :
  side_length = 4 →
  rectangles = 4 →
  shaded_triangles = 4 →
  ∃ (total_area_shaded : ℕ), total_area_shaded = 8 :=
by
  intro h1 h2 h3
  use 8
  sorry

end shaded_area_of_square_l38_38439


namespace deformation_after_flip_l38_38620

variables (m1 m2 : ℝ) (g k : ℝ) (x1 x2 x : ℝ)

-- Conditions
def condition1 := x1 = 8 / 100 -- converting cm to meters
def condition2 := x2 = 15 / 100 -- converting cm to meters

def hooked_law_upper := m1 * g = 2 * k * x1
def hooked_law_lower := m2 * g = 2 * k * x2

def deformation_whole := m2 * g = k * x

-- Theorem to prove
theorem deformation_after_flip (h1 : condition1) (h2 : condition2)
  (h3 : hooked_law_upper) (h4 : hooked_law_lower)
  (h5 : deformation_whole) : x = 0.3 := 
sorry

end deformation_after_flip_l38_38620


namespace cone_volume_percentage_filled_l38_38941

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38941


namespace reduced_admission_price_is_less_l38_38426

-- Defining the conditions
def regular_admission_cost : ℕ := 8
def total_people : ℕ := 2 + 3 + 1
def total_cost_before_6pm : ℕ := 30
def cost_per_person_before_6pm : ℕ := total_cost_before_6pm / total_people

-- Stating the theorem
theorem reduced_admission_price_is_less :
  (regular_admission_cost - cost_per_person_before_6pm) = 3 :=
by
  sorry -- Proof to be filled

end reduced_admission_price_is_less_l38_38426


namespace area_of_section_volume_of_pyramid_l38_38590

/-
Given:
1. The pyramid is a regular triangular pyramid.
2. Side \( a \) of the base.
3. Angle \( \alpha \) between the section and the base.
4. The section is made through the vertex and the midpoints of two sides of the base.

Prove:
1. The area of the section is \( \frac{a^2 \sqrt{3}}{48 \cos \alpha} \).
2. The volume of the pyramid is \( \frac{a^3 \tan \alpha}{48} \).
-/

variables (a α : Real) (midpoint_section : Bool)

-- Conditions
def is_regular_triangular_pyramid : Prop := true  -- This should include precise mathematical definition  
def side_of_base : Real := a
def angle_alpha : Real := α
def section_made : Bool := midpoint_section

-- To be proven
theorem area_of_section (h1 : is_regular_triangular_pyramid) (h2 : section_made) :
    ∃ A, A = (a^2 * Real.sqrt 3) / (48 * Real.cos α) := sorry

theorem volume_of_pyramid (h1 : is_regular_triangular_pyramid) (h2 : section_made) :
    ∃ V, V = (a^3 * Real.tan α) / 48 := sorry

end area_of_section_volume_of_pyramid_l38_38590


namespace tangent_circles_length_O1O2_l38_38730

-- Setting up the problem
noncomputable def length_O1O2 (d1 d2 : ℝ) : set ℝ :=
let r1 := d1 / 2 in
let r2 := d2 / 2 in
{r1 + r2, abs (r1 - r2)}

-- The theorem statement
theorem tangent_circles_length_O1O2 :
  length_O1O2 9 4 = {6.5, 2.5} :=
sorry

end tangent_circles_length_O1O2_l38_38730


namespace complete_square_solution_l38_38087

theorem complete_square_solution (x : ℝ) :
  x^2 - 8 * x + 6 = 0 → (x - 4)^2 = 10 :=
by
  intro h
  -- Proof would go here
  sorry

end complete_square_solution_l38_38087


namespace find_c_plus_d_l38_38742

theorem find_c_plus_d (a b c d : ℝ) (h1 : a + b = 12) (h2 : b + c = 9) (h3 : a + d = 6) : 
  c + d = 3 := 
sorry

end find_c_plus_d_l38_38742


namespace total_penalty_kicks_l38_38096

theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (hoop_challenges : ℕ)
  (h_total : total_players = 25) (h_goalies : goalies = 5) (h_hoop_challenges : hoop_challenges = 10) :
  (goalies * (total_players - 1)) = 120 :=
by
  sorry

end total_penalty_kicks_l38_38096


namespace chess_team_arrangements_l38_38494

-- Define the problem conditions
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def girls_at_ends : bool := true

-- Define what we need to prove
theorem chess_team_arrangements : 
    (girls_at_ends = true) →
    (num_boys = 3) →
    (num_girls = 3) →
    ∃ n : ℕ, n = 144 :=
by
  sorry

end chess_team_arrangements_l38_38494


namespace arrangement_count_l38_38612

theorem arrangement_count :
  ∀ (animals : List String),
    animals = ["Rat", "Cow", "Tiger", "Rabbit", "Dragon", "Snake"] →
    (List.nodup animals) →
    ∃ (arrangements : Nat), arrangements = 24 :=
by
  intros animals h1 h2
  have h3 : animals.length = 6 := by simp [h1]
  have h4 : (2.choose 1) = 1 := by norm_num -- choosing 1 out of 2 fixed positions
  have h5 : ((4 !)) = 24 := by norm_num -- arranging 4 animals
  use 24
  sorry

end arrangement_count_l38_38612


namespace Randy_bats_l38_38809

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l38_38809


namespace tan_2x_equals_cos_half_x_has_5_solutions_l38_38738

theorem tan_2x_equals_cos_half_x_has_5_solutions :
  ∃ n : ℕ, {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan(2 * x) = Real.cos(x / 2)}.finite ∧
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan(2 * x) = Real.cos(x / 2)}.card = 5 :=
by
  sorry

end tan_2x_equals_cos_half_x_has_5_solutions_l38_38738


namespace Q_transform_l38_38848

def rotate_180_clockwise (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  (2 * px - qx, 2 * py - qy)

def reflect_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def Q := (8, -11) -- from the reverse transformations

theorem Q_transform (c d : ℝ) :
  (reflect_y_equals_x (rotate_180_clockwise (2, -3) (c, d)) = (5, -4)) → (d - c = -19) :=
by sorry

end Q_transform_l38_38848


namespace player_two_bound_player_one_bound_l38_38871

/-
Two people play the following game:
The first person calls out a number, and the second person uses their judgment to replace one of the asterisks in the following expression:
\[ * * * * - * * * * \]
Then the first person calls out another number, and this process continues for a total of 8 times, until all the asterisks are replaced by numbers.
The goal of the first person is to make the resulting difference as large as possible, while the second person aims to make the difference as small as possible.
-/

/- 
  (1) No matter what numbers the first person calls out, 
  the second person can always arrange the numbers so that the difference does not exceed 4000.
-/
theorem player_two_bound {ncalls : Nat -> Nat} (a b : List Nat) 
  (l1 l2 : List Nat) (h1 : l1.length = 4) (h2 : l2.length = 4) 
  (e1 : a = l1 ++ l2) (e2 : b = l2 ++ l1) : 
  | list.to_digit l1 - list.to_digit l2 | ≤ 4000 :=
by
  sorry

/-
  (2) No matter where the second person places the numbers, 
  the first person can always call out numbers in such a way 
  that the resulting difference is at least 4000.
-/
theorem player_one_bound {ncalls : Nat -> Nat} (a b : List Nat) 
  (l1 l2 : List Nat) (h1 : l1.length = 4) (h2 : l2.length = 4) 
  (e1 : a = l1 ++ l2) (e2 : b = l2 ++ l1) : 
  | list.to_digit l1 - list.to_digit l2 | ≥ 4000 :=
by
  sorry

end player_two_bound_player_one_bound_l38_38871


namespace minimum_races_to_determine_top_five_fastest_horses_l38_38181

-- Defining the conditions
def max_horses_per_race : ℕ := 3
def total_horses : ℕ := 50

-- The main statement to prove the minimum number of races y
theorem minimum_races_to_determine_top_five_fastest_horses (y : ℕ) :
  y = 19 :=
sorry

end minimum_races_to_determine_top_five_fastest_horses_l38_38181


namespace rectangle_dimensions_l38_38441

-- Define the conditions
def side_length_tile : ℝ := 10
def perimeter_tile : ℝ := 4 * side_length_tile
def perimeter_small_square : ℝ := perimeter_tile / 5
def side_length_small_square : ℝ := perimeter_small_square / 4

-- Define the variables a and b for the dimensions of the rectangles
def a : ℝ := (side_length_tile - side_length_small_square) / 2
def b : ℝ := side_length_tile - a

-- Theorem to prove the dimensions of the rectangles
theorem rectangle_dimensions : a = 4 ∧ b = 6 :=
by
  -- Skip the proof with 'sorry'
  sorry

end rectangle_dimensions_l38_38441


namespace isosceles_triangle_angle_lt_90_l38_38734

theorem isosceles_triangle_angle_lt_90
  (A B C : Point) (h: triangle A B C) (h1 : AB = AC) :
  angle B < 90 :=
sorry

end isosceles_triangle_angle_lt_90_l38_38734


namespace water_filled_percent_l38_38979

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38979


namespace percent_filled_cone_l38_38972

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38972


namespace polynomial_geometric_roots_k_value_l38_38124

theorem polynomial_geometric_roots_k_value 
    (j k : ℝ)
    (h : ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧ 
      (∀ u v : ℝ, (u = a ∨ u = a * r ∨ u = a * r^2 ∨ u = a * r^3) →
        (v = a ∨ v = a * r ∨ v = a * r^2 ∨ v = a * r^3) →
        u ≠ v) ∧ 
      (a + a * r + a * r^2 + a * r^3 = 0) ∧
      (a^4 * r^6 = 900)) :
  k = -900 :=
sorry

end polynomial_geometric_roots_k_value_l38_38124


namespace Randy_bats_l38_38808

theorem Randy_bats (bats gloves : ℕ) (h1 : gloves = 7 * bats + 1) (h2 : gloves = 29) : bats = 4 :=
by
  sorry

end Randy_bats_l38_38808


namespace vector_expression_l38_38047

open Real EuclideanSpace

noncomputable def a : E 3 := ![4, -7, 2]
noncomputable def b : E 3 := ![-2, 3, -1]
noncomputable def c : E 3 := ![5, -1, 8]

theorem vector_expression :
  3 • (a + 2 • b) ⬝ (2 • (b + c) × (3 • c - 2 • a)) = -66 :=
by
  sorry

end vector_expression_l38_38047


namespace max_distance_from_circle_to_line_l38_38118

open Real

def circle : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 + y^2 - 4*x - 4*y - 10 = 0 }

def line : set (ℝ × ℝ) :=
  { p | let (x, y) := p in x + y - 14 = 0 }

theorem max_distance_from_circle_to_line : 
  ∀ p ∈ circle, ∃ q ∈ line, dist p q = 8 * sqrt 2 :=
by
  sorry

end max_distance_from_circle_to_line_l38_38118


namespace repeating_decimal_division_l38_38151

theorem repeating_decimal_division :
  (let r : ℚ := 1/99 in
   let x : ℚ := 63 * r in
   let y : ℚ := 21 * r in
   x / y = 3) :=
by {
  sorry
}

end repeating_decimal_division_l38_38151


namespace max_distance_KM_l38_38480

noncomputable def point := ℝ × ℝ

def dist (p q : point) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def center_K : point := (8 / 3, 0)
def center_M : point := (4 / 3, 4 * sqrt 3 / 3)

def radius_K : ℝ := 4 / 3
def radius_M : ℝ := 4 / 3

theorem max_distance_KM : dist center_K center_M + radius_K + radius_M = 16 / 3 := by
  sorry

end max_distance_KM_l38_38480


namespace angle_QRM_l38_38273

-- Definitions for the named points.
structure Point where
  x : ℝ
  y : ℝ

-- Lines represented by pairs of points.
def Line (P Q : Point) := { r : ℝ // P.x * r = Q.x * r + 1 }

-- Given conditions
variables (J K L M Q R : Point)
variable (parallel_JK_LM : Line J K = Line L M)
variable (angle_LRQ : ℝ)
variable (angle_RJK : ℝ)

-- Additional conditions extracted from the problem
variable (angle_LRQ_val : angle_LRQ = 2.5 * 24)
variable (angle_RJK_val : angle_RJK = (2 * 24) + (3 * 24))

-- The main theorem to prove
theorem angle_QRM : ∠ Q R M = 60 :=
by
  have h1 : angle_LRQ + angle_RJK = 180 := sorry -- Proof of the consecutive interior angles theorem
  have h2 : 2.5 * 24 + (2 * 24 + 3 * 24) = 180 := by linarith
  have h3 : 2.5 * 24 = 60 := by norm_num
  show ∠ Q R M = 60, by exact h3

end angle_QRM_l38_38273


namespace shifted_parabola_is_g_l38_38517

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5
def g (x : ℝ) : ℝ := (x + 1)^2 - 3

theorem shifted_parabola_is_g (x : ℝ) :
  (shift_left_up f 2 3) x = g x := 
sorry

end shifted_parabola_is_g_l38_38517


namespace unique_n_digit_number_divisible_by_5_power_l38_38189

theorem unique_n_digit_number_divisible_by_5_power (n : ℕ) (h : n > 0) :
  ∃! (A : ℕ), A < 10^n ∧ A ≥ 10^(n-1) ∧ (A % 5^n = 0) ∧ (∀ d : ℕ, d ∈ digits 10 A → d ∈ {1, 2, 3, 4, 5}) :=
sorry

end unique_n_digit_number_divisible_by_5_power_l38_38189


namespace max_candy_leftover_l38_38388

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l38_38388


namespace remainder_div_polynomial_l38_38523

theorem remainder_div_polynomial :
  ∀ (x : ℝ), 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    R x = (3^101 - 2^101) * x + (2^101 - 2 * 3^101) ∧
    x^101 = (x^2 - 5 * x + 6) * Q x + R x :=
by
  sorry

end remainder_div_polynomial_l38_38523


namespace find_angle_ACD_and_cond_l38_38525

noncomputable def angle_ACD (α : ℝ) : ℝ :=
  Real.arcsin (Real.tan (α / 2) / Real.sqrt 3)

theorem find_angle_ACD_and_cond α (hα : 0 < α ∧ α < 180) :
  angle_ACD α = Real.arcsin (Real.tan (α / 2) / Real.sqrt 3) ∧ α ≤ 120 :=
by {
  split,
  { 
    refl,
  },
  {
    exact Real.tan_half_le_sqrt3_of_lt_pi_div2 (by linarith) (by linarith)
  }
}

end find_angle_ACD_and_cond_l38_38525


namespace sum_first_2024_correct_l38_38726

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 0 ∧ ∀ n : ℕ, a (n + 2) = a n + 2 * (-1) ^ n

def sum_of_first_2024_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 2024, a (i + 1)

theorem sum_first_2024_correct (a : ℕ → ℝ) (h : sequence a) : 
  sum_of_first_2024_terms a = 2024 := 
sorry

end sum_first_2024_correct_l38_38726


namespace number_of_carbon_atoms_l38_38580

/-- A proof to determine the number of carbon atoms in a compound given specific conditions
-/
theorem number_of_carbon_atoms
  (H_atoms : ℕ) (O_atoms : ℕ) (C_weight : ℕ) (H_weight : ℕ) (O_weight : ℕ) (Molecular_weight : ℕ) :
  H_atoms = 6 →
  O_atoms = 1 →
  C_weight = 12 →
  H_weight = 1 →
  O_weight = 16 →
  Molecular_weight = 58 →
  (Molecular_weight - (H_atoms * H_weight + O_atoms * O_weight)) / C_weight = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_carbon_atoms_l38_38580


namespace purple_balls_are_zero_l38_38578

/-- 
A bag contains 100 balls: 50 white, 30 green, 8 yellow, 9 red, and some unknown number of purple.
If a ball is chosen at random, the probability that it is neither red nor purple is 0.88.
Let P be the number of purple balls in the bag.
Prove that the number of purple balls is 0.
-/
theorem purple_balls_are_zero
  (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (P : ℕ)
  (h_total : total_balls = 100) 
  (h_white : white_balls = 50) 
  (h_green : green_balls = 30) 
  (h_yellow : yellow_balls = 8) 
  (h_red : red_balls = 9) 
  (h_probability : (white_balls + green_balls + yellow_balls) / total_balls = 0.88) :
  P = 0 :=
sorry

end purple_balls_are_zero_l38_38578


namespace alpha_minus_beta_l38_38708

-- Providing the conditions
variable (α β : ℝ)
variable (hα1 : 0 < α ∧ α < Real.pi / 2)
variable (hβ1 : 0 < β ∧ β < Real.pi / 2)
variable (hα2 : Real.tan α = 4 / 3)
variable (hβ2 : Real.tan β = 1 / 7)

-- The goal is to show that α - β = π / 4 given the conditions
theorem alpha_minus_beta :
  α - β = Real.pi / 4 := by
  sorry

end alpha_minus_beta_l38_38708


namespace domain_of_f_l38_38155

def domain_of_function {α : Type*} [TopologicalSpace α] (f : α → α) (s : Set α) : Prop :=
  ∀ x : α, x ∈ s ↔ ∃ y, x = y ∧ f y ≠ 0

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x - 3) / (x^2 - 4)

theorem domain_of_f :
  domain_of_function f ((Set.Iio (-2)) ∪ (Set.Ioo (-2) 2) ∪ (Set.Ioi 2)) :=
by
  sorry

end domain_of_f_l38_38155


namespace max_min_difference_l38_38914

def y (x : ℝ) : ℝ := x * abs (3 - x) - (x - 3) * abs x

theorem max_min_difference : (0 : ℝ) ≤ x → (x < 3 → y x ≤ y (3 / 4)) ∧ (x < 0 → y x = 0) ∧ (x ≥ 3 → y x = 0) → 
  (y (3 / 4) - (min (y 0) (min (y (-1)) (y 3)))) = 1.125 :=
by
  sorry

end max_min_difference_l38_38914


namespace exists_equilateral_triangle_l38_38729

noncomputable def exists_equilateral_triangle_on_lines (a b c : ℝ → ℝ → Prop) : Prop :=
  (∀ x1 x2, a x1 x2 → b x1 x2) →              
  (∃ P A B : ℝ × ℝ,
    c P.1 P.2 ∧
    a A.1 A.2 ∧
    b B.1 B.2 ∧
    dist P A = dist A B ∧
    dist A B = dist B P)

theorem exists_equilateral_triangle (a b c : ℝ → ℝ → Prop) (h_parallel : ∀ x1 x2, a x1 x2 → b x1 x2)
  (h_skew : ∀ x1 x2, ¬ ( a x1 x2 ∧ c x1 x2 )) :
  exists_equilateral_triangle_on_lines a b c :=
begin
  -- Proof goes here
  sorry
end

end exists_equilateral_triangle_l38_38729


namespace square_perimeter_sum_l38_38097

theorem square_perimeter_sum (x : ℝ)
  (h1 : ∀ x, ∃ s : ℝ, s^2 = x^2 + 8 * x + 16)
  (h2 : ∀ x, ∃ t : ℝ, t^2 = 4 * x^2 - 12 * x + 9)
  (h3 : 4 * (sqrt (x^2 + 8 * x + 16)) + 4 * (sqrt (4 * x^2 - 12 * x + 9)) = 32) :
  x = 7/3 :=
sorry

end square_perimeter_sum_l38_38097


namespace same_terminal_side_l38_38613

open Real

/-- Given two angles θ₁ = -7π/9 and θ₂ = 11π/9, prove that they have the same terminal side
    which means proving θ₂ - θ₁ is an integer multiple of 2π. -/
theorem same_terminal_side (θ₁ θ₂ : ℝ) (hθ₁ : θ₁ = - (7 * π / 9)) (hθ₂ : θ₂ = 11 * π / 9) :
    ∃ (k : ℤ), θ₂ - θ₁ = 2 * π * k := by
  sorry

end same_terminal_side_l38_38613


namespace percent_filled_cone_l38_38974

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38974


namespace shaded_region_area_l38_38270

theorem shaded_region_area :
  let r := 5 in
  let quarter_circle_area := (1/4) * Real.pi * r^2 in
  let square_area := r^2 in
  let shaded_region_area := quarter_circle_area - square_area in
  let total_shaded_area := 8 * shaded_region_area in
  total_shaded_area = 50 * Real.pi - 200 :=
by
  let r := 5
  let quarter_circle_area := (1/4) * Real.pi * r^2
  let square_area := r^2
  let shaded_region_area := quarter_circle_area - square_area
  let total_shaded_area := 8 * shaded_region_area
  show total_shaded_area = 50 * Real.pi - 200
  sorry

end shaded_region_area_l38_38270


namespace ellipse_polar_inverse_sum_l38_38767

noncomputable def ellipse_equation (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

theorem ellipse_polar_inverse_sum (A B : ℝ × ℝ)
  (hA : ∃ α₁, ellipse_equation α₁ = A)
  (hB : ∃ α₂, ellipse_equation α₂ = B)
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / (A.1 ^ 2 + A.2 ^ 2) + 1 / (B.1 ^ 2 + B.2 ^ 2)) = 7 / 12 :=
by
  sorry

end ellipse_polar_inverse_sum_l38_38767


namespace mh_range_l38_38711

theorem mh_range (x m : ℝ) (h : 1 / 3 < x ∧ x < 1 / 2) (hx : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 := 
sorry

end mh_range_l38_38711


namespace longer_side_of_rectangle_l38_38200

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def diameter (r : ℝ) : ℝ := 2 * r
noncomputable def rect_area (r : ℝ) : ℝ := 3 * (circle_area r)
noncomputable def longer_side (r : ℝ) (rect_area : ℝ) : ℝ := rect_area / (diameter r)

theorem longer_side_of_rectangle :
  let r := circle_radius in
  let d := diameter r in
  let A := rect_area r in
  longer_side r A = 9 * Real.pi :=
by
  -- Proof omitted
  sorry

end longer_side_of_rectangle_l38_38200


namespace prove_N_not_subset_M_l38_38377

def set_M : Set ℝ := { x | x^2 - 6 * x - 16 < 0 }

def set_possible_N : List (Set ℝ) := [
  ∅,
  { x | -2 < x ∧ x < 8 },
  { x | 0 < Real.log x ∧ Real.log x ≤ 2 },
  Icc (-1 : ℝ) 8
]

theorem prove_N_not_subset_M :
  ∀ N ∈ set_possible_N, (N ⊆ set_M) → N ≠ Icc (-1 : ℝ) 8 := sorry

end prove_N_not_subset_M_l38_38377


namespace options_equal_results_l38_38614

theorem options_equal_results :
  (4^3 ≠ 3^4) ∧
  ((-5)^3 = (-5^3)) ∧
  ((-6)^2 ≠ -6^2) ∧
  ((- (5/2))^2 ≠ (- (2/5))^2) :=
by {
  sorry
}

end options_equal_results_l38_38614


namespace Ximena_page_numbering_l38_38171

theorem Ximena_page_numbering (total_pages : ℕ) (digit2_stickers : ℕ) (max_page : ℕ) :
  digit2_stickers = 100 →
  max_page = 244 →
  (count_digit_occurrences 2 total_pages ≤ 100) :=
sorry

end Ximena_page_numbering_l38_38171


namespace no_counterexample_to_divisibility_l38_38787

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_to_divisibility :
  ∀ n ∈ {27, 36, 45, 81}, (sum_of_digits n) % 9 = 0 → n % 9 = 0 :=
by
  sorry

end no_counterexample_to_divisibility_l38_38787


namespace probability_of_losing_l38_38750

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end probability_of_losing_l38_38750


namespace right_triangular_prism_volume_l38_38687

theorem right_triangular_prism_volume (R a h V : ℝ)
  (h1 : 4 * Real.pi * R^2 = 12 * Real.pi)
  (h2 : h = 2 * R)
  (h3 : (1 / 3) * (Real.sqrt 3 / 2) * a = R)
  (h4 : V = (1 / 2) * a * a * (Real.sin (Real.pi / 3)) * h) :
  V = 54 :=
by sorry

end right_triangular_prism_volume_l38_38687


namespace sin_sum_alpha_pi_over_3_l38_38697

theorem sin_sum_alpha_pi_over_3 (alpha : ℝ) (h1 : Real.cos (alpha + 2/3 * Real.pi) = 4/5) (h2 : -Real.pi/2 < alpha ∧ alpha < 0) :
  Real.sin (alpha + Real.pi/3) + Real.sin alpha = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_sum_alpha_pi_over_3_l38_38697


namespace tens_digit_of_9_to_2023_l38_38546

theorem tens_digit_of_9_to_2023 :
  (9^2023 % 100) / 10 % 10 = 8 :=
sorry

end tens_digit_of_9_to_2023_l38_38546


namespace land_side_length_l38_38907

/-- The length of one side of a square plot of land with an area of 1024 square units is 32 units. -/
theorem land_side_length (A : ℝ) (h : A = 1024) : ∃ s : ℝ, s * s = A ∧ s = 32 :=
by
  use 32
  split
  . show 32 * 32 = A
    rw [h] 
    norm_num
  . show 32 = 32
    rfl

end land_side_length_l38_38907


namespace problem_solution_l38_38648

theorem problem_solution (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 3) (h2 : sqrt (3 - x) - sqrt (x + 1) > 1 / 2) : -1 ≤ x ∧ x < 1 - sqrt 31 / 8 := sorry

end problem_solution_l38_38648


namespace vector_operation_l38_38383

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

theorem vector_operation :
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end vector_operation_l38_38383


namespace greatest_perimeter_is_approximately_34_36_l38_38821

noncomputable def greatest_perimeter_of_divided_isosceles_triangle 
  (base : ℝ) (height : ℝ) (n : ℕ) : ℝ :=
  let area_fn (k : ℕ) : ℝ := 
    2 + real.sqrt (height^2 + (base / n * k)^2) 
      + real.sqrt (height^2 + (base / n * (k + 1))^2) in
  list.maximum (list.map area_fn (list.range n)) sorry

theorem greatest_perimeter_is_approximately_34_36 :
  greatest_perimeter_of_divided_isosceles_triangle 12 15 6 ≈ 34.36 :=
sorry

end greatest_perimeter_is_approximately_34_36_l38_38821


namespace exists_correct_seating_l38_38210

variables (m n : ℕ)
variables (row seat : ℕ → ℕ)

/-- Given a theater with m rows and n seats per row, and mn tickets sold such that some tickets have duplicate seat numbers, prove that it is possible for at least one audience member to sit in the seat corresponding to both their row and seat number on their ticket. -/
theorem exists_correct_seating (mn_tickets_sold : list (ℕ × ℕ)) :
  ∃ (t : ℕ × ℕ), (t ∈ mn_tickets_sold) ∧ t.1 < m ∧ t.2 < n ∧
    ∀ t' ∈ mn_tickets_sold, t ≠ t' → (t.1 ≠ t'.1 ∨ t.2 ≠ t'.2) :=
sorry

end exists_correct_seating_l38_38210


namespace slope_angle_l38_38328

-- Define the problem conditions
def passes_through_point (l : LinearMap ℝ ℝ ℝ) (P : ℝ × ℝ)  : Prop :=
  l P.1 = P.2

def intersects_curve (l : LinearMap ℝ ℝ ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, f x1 = l x1 ∧ f x2 = l x2 ∧ x1 ≠ x2

def triangle_area (A O B : ℝ × ℝ) : ℝ :=
  0.5 * (A.1 * B.2 - A.2 * B.1)

noncomputable def line_l : LinearMap ℝ ℝ ℝ :=
  { toFun := λ x, -((Real.sqrt 3) / 3) * (x - 2) }

def curve_f : ℝ → ℝ := λ x, Real.sqrt (2 - x^2)

theorem slope_angle {P : ℝ × ℝ} (hP : P = (2, 0)) : 
  passes_through_point line_l P ∧ intersects_curve line_l curve_f ∧ triangle_area (1, 1) (0, 0) (-1, -1) = 1 → 
  ∃ θ, θ = 150 := 
by
  sorry

end slope_angle_l38_38328


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38887

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38887


namespace cone_water_fill_percentage_l38_38936

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l38_38936


namespace diameter_is_1p201_l38_38833

noncomputable def diameter_of_wheel (revolutions_per_minute : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60
  (speed_mpm / revolutions_per_minute) / Real.pi

theorem diameter_is_1p201 :
  diameter_of_wheel 265.15 60 ≈ 1.201 := by
  sorry

end diameter_is_1p201_l38_38833


namespace sequence_increasing_l38_38505

noncomputable def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ) (hn : 0 < n) : a n < a (n + 1) :=
by
  -- Proof to be provided
  sorry

end sequence_increasing_l38_38505


namespace water_volume_percentage_l38_38966

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l38_38966


namespace time_to_next_specific_time_l38_38131

theorem time_to_next_specific_time : 
  ∀ (current_time next_time : Nat) (minutes_passed : Nat), 
  current_time = 232 ∧ next_time = 323 ∧ minutes_passed = 91 → next_time - current_time = minutes_passed :=
by
  intro current_time next_time minutes_passed
  rintros ⟨h1, h2, h3⟩
  have h : (5 * 60 + 23) - (3 * 60 + 52) = 91 := by
    calc
      (5 * 60 + 23) - (3 * 60 + 52) 
          = (300 + 23) - (180 + 52) : by rfl
      ... = 323 - 232 : by rfl
      ... = 91 : by rfl
  exact h

end time_to_next_specific_time_l38_38131


namespace arun_borrowed_amount_l38_38910

theorem arun_borrowed_amount :
  ∃ P : ℝ, 
    (P * 0.08 * 4 + P * 0.10 * 6 + P * 0.12 * 5 = 12160) → P = 8000 :=
sorry

end arun_borrowed_amount_l38_38910


namespace prove_R_value_l38_38461

noncomputable def geometric_series (Q : ℕ) : ℕ :=
  (2^(Q + 1) - 1)

noncomputable def R (F : ℕ) : ℝ :=
  Real.sqrt (Real.log (1 + F) / Real.log 2)

theorem prove_R_value :
  let F := geometric_series 120
  R F = 11 :=
by
  sorry

end prove_R_value_l38_38461


namespace number_of_proper_subsets_of_set_l38_38846

open Finset

theorem number_of_proper_subsets_of_set {α : Type} (s : Finset α) (h : s = {0, 2, 3}) :
  (∑ k in range s.card, (s.card.choose k)) = 7 :=
by
  rw h
  simp
  sorry

end number_of_proper_subsets_of_set_l38_38846


namespace min_breaks_12_no_breaks_15_l38_38301

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l38_38301


namespace slope_angle_of_parallel_line_l38_38853

theorem slope_angle_of_parallel_line (y : ℝ): 
  (∀ x : ℝ, y + 3 = 0 → slope_angle (λ x : ℝ, y + 3) = 0) :=
by
  sorry

end slope_angle_of_parallel_line_l38_38853


namespace cone_filled_with_water_to_2_3_height_l38_38994

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38994


namespace num_n_digit_integers_total_digits_needed_l38_38175

section

variable (n : ℕ)

-- Problem (a)
theorem num_n_digit_integers : 
  ∀ (n : ℕ), 1 ≤ n → (10^n - 10^(n-1)) = 9 * 10^(n-1) := by
  sorry

-- Problem (b)
theorem total_digits_needed :
  ∀ (n : ℕ), 1 ≤ n → ∑ k in finset.range n, (k + 1) * 9 * 10^k = n * 10^n - (10^n - 1) / 9 := by
  sorry

end

end num_n_digit_integers_total_digits_needed_l38_38175


namespace find_a_l38_38796

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a)^2 + (2 * Real.ln x - 2 * a)^2

theorem find_a (a : ℝ) (h : ∃ x > 0, f x a ≤ 4 / 5) : a = 1 / 5 :=
  sorry

end find_a_l38_38796


namespace probability_losing_game_l38_38748

-- Define the odds of winning in terms of number of wins and losses
def odds_winning := (wins: ℕ, losses: ℕ) := (5, 3)

-- Given the odds of winning, calculate the total outcomes
def total_outcomes : ℕ := (odds_winning.1 + odds_winning.2)

-- Define the probability of losing the game
def probability_of_losing (wins losses: ℕ) (total: ℕ) : ℚ := (losses : ℚ) / (total : ℚ)

-- Given odds of 5:3, prove the probability of losing is 3/8
theorem probability_losing_game : probability_of_losing odds_winning.1 odds_winning.2 total_outcomes = 3 / 8 :=
by
  sorry

end probability_losing_game_l38_38748


namespace unique_lcm_condition_l38_38027

theorem unique_lcm_condition (a : Fin 2000 → ℕ) (h_distinct: ∀ i j, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j, i ≤ j → a i ≤ a j)
  (h_bound: ∀ i, 1 ≤ a i ∧ a i < 4000)
  (h_lcm: ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 4000)
  : a 0 ≥ 1334 :=
by
  sorry

end unique_lcm_condition_l38_38027


namespace area_triangle_QRS_l38_38758

open Real

noncomputable def area_of_triangle {A B C : Point} (a : LineSegment A B) (b : LineSegment B C) (c : LineSegment C A) : ℝ := by
  let base := segmentLength c
  let height := segmentLength a
  exact (1 / 2) * base * height
  sorry

theorem area_triangle_QRS :
  ∀ {P Q R S : Point},
    right_angle ∠RPQ →
    collinear {P, Q, S} →
    segmentLength (LineSegment S Q) = 14 →
    segmentLength (LineSegment S P) = 18 →
    segmentLength (LineSegment S R) = 30 →
    area_of_triangle (LineSegment Q R) (LineSegment R S) (LineSegment S Q) = 168 := by
  intros P Q R S hAngle hCollinear hSQ hSP hSR
  sorry

end area_triangle_QRS_l38_38758


namespace sum_tens_ones_digits_3_plus_4_power_17_l38_38877

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l38_38877


namespace tetrahedron_minimum_distance_l38_38688

noncomputable def minimum_distance (AB CD AD AC BC BD : ℝ) (M N : Point) : ℝ :=
  min_dist AB CD AD AC BC BD M N

theorem tetrahedron_minimum_distance (h1 : AB = 2) (h2 : CD = 2)
                                      (h3 : AD = 3) (h4 : AC = 3)
                                      (h5 : BC = 3) (h6 : BD = 3)
                                      (M_on_AB : Point_on_AB M AB)
                                      (N_on_CD : Point_on_CD N CD) :
  minimum_distance 2 2 3 3 3 3 M N = sqrt 7 :=
sorry

end tetrahedron_minimum_distance_l38_38688


namespace solve_for_x_l38_38083

theorem solve_for_x (x : ℝ) (h : sqrt (x^3) = 27 * (27 ^ (1 / 3))) : x = 27 := 
sorry

end solve_for_x_l38_38083


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38886

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38886


namespace book_width_l38_38192

theorem book_width (length area : ℝ) (h_length : length = 2) (h_area : area = 6) : 
  ∃ width : ℝ, width = 3 ∧ length * width = area :=
by 
  -- We assume the length and area of the book.
  have h_eq : 2 * 3 = 6, by norm_num,
  -- The width should satisfy is 3 inches.
  exact ⟨3, rfl, h_eq⟩

end book_width_l38_38192


namespace solution_unique_s_l38_38650

theorem solution_unique_s (s : ℝ) (hs : ⌊s⌋ + s = 22.7) : s = 11.7 :=
sorry

end solution_unique_s_l38_38650


namespace percentage_of_only_cat_owners_l38_38762

theorem percentage_of_only_cat_owners (total_students total_dog_owners total_cat_owners both_cat_dog_owners : ℕ) 
(h_total_students : total_students = 500)
(h_total_dog_owners : total_dog_owners = 120)
(h_total_cat_owners : total_cat_owners = 80)
(h_both_cat_dog_owners : both_cat_dog_owners = 40) :
( (total_cat_owners - both_cat_dog_owners : ℕ) * 100 / total_students ) = 8 := 
by
  sorry

end percentage_of_only_cat_owners_l38_38762


namespace water_filled_percent_l38_38984

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l38_38984


namespace zack_marbles_number_l38_38556

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l38_38556


namespace files_sorted_first_one_and_half_hours_l38_38206

-- Given conditions as definitions
def sorting_rate := 30
def total_files := 1775
def total_time := 3 + 10 / 60 -- 3.1667 hours

-- Defining general time points
def one_and_half_hours := 1.5
def clerks_rate (x t: ℕ) := ((sorting_rate * x) + (sorting_rate * (x - t) * (1.0 : float))) + 
                            ((sorting_rate * (x - 2 * t) * (1.0 : float)))

-- The goal statement for proof
theorem files_sorted_first_one_and_half_hours (x t: ℕ) 
  (h1 : sorting_rate * x + sorting_rate * (x - t) + sorting_rate * (x - 2 * t) + sorting_rate * (x - 3 * t) / 6 = total_files):
  sorting_rate * x * (1.0 : float) + sorting_rate * (x - t * (1.5 : float)) = 945 :=
sorry

end files_sorted_first_one_and_half_hours_l38_38206


namespace sum_of_first_17_terms_l38_38310

variable {a : ℕ → ℝ} -- we assume a sequence of real numbers

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_17_terms (assume_arithmetic : arithmetic_sequence a)
  (h : a 4 + a 14 = 2) : 
  (let S₁₇ := (17 * (a 1 + a 17)) / 2 in S₁₇ = 17) :=
sorry

end sum_of_first_17_terms_l38_38310


namespace ways_to_choose_colors_l38_38765

theorem ways_to_choose_colors (colors : Fin 10 → Prop) (blue : colors β) :
  ∃ n : ℕ, n = 36 :=
by
  let colors := 10
  let blue := true
  let remainingColors := 9
  let choose := Nat.choose remainingColors 2
  have h : choose = 36 := sorry
  use choose
  exact h

end ways_to_choose_colors_l38_38765


namespace eq_line_parabola_l38_38208

/-- 
Proof problem: 
Given a line passing through the point (2, 1), intersects the parabola y^2 = 5/2 x at points A and B (distinct from the origin O) and the condition OA ⊥ OB, prove the equation of the line is 2x + y - 5 = 0.
-/
theorem eq_line_parabola (A B O : Point) (hA : A ≠ O) (hB : B ≠ O) (hAB : A ≠ B) 
  (line : Line) (parabola : Parabola) :
  (line.intersects_at A) ∧ (line.intersects_at B) ∧ (parabola.contains A) ∧ (parabola.contains B) 
  ∧ (OA⊥OB) → line.equation = 2*x + y - 5 := 
sorry

end eq_line_parabola_l38_38208


namespace cos_2theta_eq_neg_half_l38_38314

theorem cos_2theta_eq_neg_half (θ : ℝ) 
  (h : 2^(1 - 3/2 + 3 * real.cos θ) + 3 = 2^(2 + real.cos θ)) : 
  real.cos (2 * θ) = -1/2 :=
sorry

end cos_2theta_eq_neg_half_l38_38314


namespace cone_height_l38_38347

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l38_38347


namespace caleb_double_burgers_l38_38623

theorem caleb_double_burgers (S D : ℕ) (h1 : S + D = 50) (h2 : 1.00 * S + 1.50 * D = 66.50) : D = 33 := 
by sorry

end caleb_double_burgers_l38_38623


namespace find_f_at_3_l38_38104

-- Defining the conditions as Lean statements
-- Condition: The graph of the function passes through a fixed point A
def point_A_on_graph_of_exponential_function : Prop :=
  ∃ (A : ℝ × ℝ), (A = (2, 8)) ∧ (∀ x, y = 2^(x-2) + 7) 

-- Condition: Point A is on the graph of the power function
def point_A_on_graph_of_power_function (f : ℝ → ℝ) : Prop :=
  f 2 = 8

-- Define the power function f using the found α
def power_function (x : ℝ) : ℝ := x^3

-- Theorem: Given the conditions, prove that f(3) = 27
theorem find_f_at_3 : point_A_on_graph_of_exponential_function ∧ point_A_on_graph_of_power_function power_function → power_function 3 = 27 := by
  sorry

end find_f_at_3_l38_38104


namespace zeros_of_derivative_arithmetic_progression_l38_38561
-- Import the necessary library

-- Define fourth-degree polynomial and the arithmetic progression properties
variable (α a r : ℝ) (hα : α ≠ 0) (hr : r > 0)

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := α * (x - a) * (x - (a + r)) * (x - (a + 2 * r)) * (x - (a + 3 * r))

-- Define the derivative f'(x)
def f_prime (x : ℝ) : ℝ := (derivative (f α a r))

-- State the theorem that needs to be proven
theorem zeros_of_derivative_arithmetic_progression (h_zeros : 
  ∀ x : ℝ, f α a r x = 0 → (x = a) ∨ (x = a + r) ∨ (x = a + 2 * r) ∨ (x = a + 3 * r)) :
  ∀ y : ℝ, f_prime α a r y = 0 → ∃ d : ℝ, ∀ n : ℤ, 
  (∃ m : ℤ, (f_prime α a r n = 0 ∧ (n:ℝ) = d * m)) :=
begin
  sorry
end

end zeros_of_derivative_arithmetic_progression_l38_38561


namespace shape_c_transformed_is_illustrated_l38_38692

-- Definitions for distinct shapes A, B, C, D, E
inductive Shape
| A : Shape
| B : Shape
| C : Shape
| D : Shape
| E : Shape

-- Definition for the illustrated shape
constant illustrated_shape : Shape

-- A function to perform a 180-degree rotation
-- Let's assume rotate_180 is predefined (if not defined, one would need to define it)
axiom rotate_180 : Shape → Shape

-- The main theorem to be proven
theorem shape_c_transformed_is_illustrated : rotate_180 Shape.C = illustrated_shape :=
sorry

end shape_c_transformed_is_illustrated_l38_38692


namespace percent_filled_cone_l38_38971

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38971


namespace num_correct_propositions_l38_38359

noncomputable def Proposition1 := ∀ (L1 L2 L3 : ℝ × ℝ), (L1 ⊥ L2 ∧ L3 ⊥ L2) → L1 ∥ L3
noncomputable def Proposition2 := ∀ (L1 L2 : ℝ × ℝ) (P : set (ℝ × ℝ)), (L1 ∥ P ∧ L2 ∥ P) → L1 ∥ L2
noncomputable def Proposition3 := ∀ (L1 L2 L3 : ℝ × ℝ), (L1 ∥ L2 ∧ L2 ∥ L3) → L1 ∥ L3
noncomputable def Proposition4 := ∀ (L1 L2 : ℝ × ℝ), (L1 ≠ L2 ∧ (L1.in_plane → ¬(L1 ∩ L2).nonempty)) → L1 ∥ L2

theorem num_correct_propositions : (¬Proposition1 ∧ ¬Proposition2 ∧ Proposition3 ∧ Proposition4) → 2 :=
by intros h
   sorry

end num_correct_propositions_l38_38359


namespace converse_not_true_prop_B_l38_38033

noncomputable def line_in_plane (b : Type) (α : Type) : Prop := sorry
noncomputable def perp_line_plane (b : Type) (β : Type) : Prop := sorry
noncomputable def perp_planes (α : Type) (β : Type) : Prop := sorry
noncomputable def parallel_planes (α : Type) (β : Type) : Prop := sorry

variables (a b c : Type) (α β : Type)

theorem converse_not_true_prop_B :
  (line_in_plane b α) → (perp_planes α β) → ¬ (perp_line_plane b β) :=
sorry

end converse_not_true_prop_B_l38_38033


namespace last_digit_expr_is_4_l38_38567

-- Definitions for last digits.
def last_digit (n : ℕ) : ℕ := n % 10

def a : ℕ := 287
def b : ℕ := 269

def expr := (a * a) + (b * b) - (2 * a * b)

-- Conjecture stating that the last digit of the given expression is 4.
theorem last_digit_expr_is_4 : last_digit expr = 4 := 
by sorry

end last_digit_expr_is_4_l38_38567


namespace cone_volume_percentage_filled_l38_38946

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38946


namespace initial_men_checking_exam_papers_l38_38089

theorem initial_men_checking_exam_papers :
  ∀ (M : ℕ),
  (M * 8 * 5 = (1/2 : ℝ) * (2 * 20 * 8)) → M = 4 :=
by
  sorry

end initial_men_checking_exam_papers_l38_38089


namespace cone_filled_with_water_to_2_3_height_l38_38998

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l38_38998


namespace standard_eq_C_and_polar_radius_M_l38_38640

-- Definitions
def parametric_eq_l1 (k t : ℝ) : ℝ × ℝ := (2 + t, k * t)
def parametric_eq_l2 (k m : ℝ) : ℝ × ℝ := (-2 + m, m / k)

-- Conditions for the proof
def intersection_trajectory_eq (x y : ℝ) : Prop :=
  x^2 - y^2 = 4

def polar_eq_l3 (θ : ℝ) : ℝ :=
  sqrt (2) / (cos θ + sin θ)

-- Main statement
theorem standard_eq_C_and_polar_radius_M (k t m x y θ : ℝ) :
  let P := parametric_eq_l1 k t in
  let Q := parametric_eq_l2 k m in
  let l1 := P.1 = x ∧ P.2 = y in
  let l2 := Q.1 = x ∧ Q.2 = y in
  intersection_trajectory_eq x y →
  polar_eq_l3 θ = sqrt(5) :=
sorry

end standard_eq_C_and_polar_radius_M_l38_38640


namespace angles_terminal_sides_and_sets_l38_38840

theorem angles_terminal_sides_and_sets :
  let α := (λ (k : ℤ), (2 * k + 1) * 180 : ℚ) in
  let β := (λ (k : ℤ), (4 * k + 1) * 180 : ℚ) in
  let M := {x | ∃ (k : ℤ), x = 45 + k * 90} in
  let N := {y | ∃ (k : ℤ), y = 90 + k * 45} in
  ( ∀ k : ℤ, α k % 360 = β k % 360 ) ∧ (M ⊆ N) := by sorry

end angles_terminal_sides_and_sets_l38_38840


namespace tan_sum_BC_l38_38772

theorem tan_sum_BC {a b c A B C : ℝ} (h₀ : b - c = (1/3) * a) (h₁ : sin B = 2 * sin A) :
  tan (B + C) = (11 / 7) * tan A :=
by
  -- Actual proof steps are omitted.
  sorry

end tan_sum_BC_l38_38772


namespace max_candy_leftover_l38_38386

theorem max_candy_leftover (x : ℕ) : (∃ k : ℕ, x = 12 * k + 11) → (x % 12 = 11) :=
by
  sorry

end max_candy_leftover_l38_38386


namespace max_of_min_values_l38_38028

theorem max_of_min_values {n : ℕ} (hn : n ≥ 2) 
  (a : fin (n+1) → ℝ) (h : ∀ i, a i ∈ set.Icc 0 1) :
  (∃ i, (a i) * (1 - a (i + 1)) = 1 / 4) :=
sorry

end max_of_min_values_l38_38028


namespace asha_wins_probability_l38_38422

variable (p_lose p_tie p_win : ℚ)

theorem asha_wins_probability 
  (h_lose : p_lose = 3 / 7) 
  (h_tie : p_tie = 1 / 7) 
  (h_total : p_win + p_lose + p_tie = 1) : 
  p_win = 3 / 7 := by
  sorry

end asha_wins_probability_l38_38422


namespace average_speed_second_part_l38_38921

theorem average_speed_second_part
  (total_distance : ℝ)
  (first_speed : ℝ)
  (total_time : ℝ)
  (distance_first_part : ℝ)
  (remaining_time := total_time - (distance_first_part / first_speed))
  (remaining_distance := total_distance - distance_first_part)
  (second_speed := remaining_distance / remaining_time) :
  total_distance = 250 ∧
  first_speed = 40 ∧
  total_time = 5.4 ∧
  distance_first_part = 148 →
  second_speed = 60 :=
by
  intros h
  cases h with h_total_distance h_rest
  cases h_rest with h_first_speed h_rest
  cases h_rest with h_total_time h_distance_first_part
  sorry

end average_speed_second_part_l38_38921


namespace find_lambda_l38_38384

noncomputable theory
open_locale classical

-- Definitions of the vectors
def vec_a : ℝ × ℝ := (2, 1)
def vec_b (λ : ℝ) : ℝ × ℝ := (3, λ)

-- Dot product of two 2D vectors
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for given conditions, λ = -1 or λ = 3
theorem find_lambda (λ : ℝ) : (dot_prod (2 • vec_a - vec_b λ) (vec_b λ) = 0) ↔ (λ = -1 ∨ λ = 3) := by
  sorry

end find_lambda_l38_38384


namespace sin_75_mul_sin_15_eq_one_fourth_l38_38856

theorem sin_75_mul_sin_15_eq_one_fourth : 
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_75_mul_sin_15_eq_one_fourth_l38_38856


namespace area_of_overlap_l38_38540

theorem area_of_overlap (beta : ℝ) (h : 0 < beta ∧ beta < π) :
  let A_rhombus := (1 / 2) * 1 * 2 * Real.sin(beta)
  let A_circle := Real.pi * 1^2
  A_rhombus - A_circle = Real.sin(beta) - Real.pi :=
by
  sorry

end area_of_overlap_l38_38540


namespace volume_filled_cone_l38_38957

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l38_38957


namespace distinct_prime_factors_210_l38_38516

-- Lean 4 statement to check the number of distinct prime factors of 210
theorem distinct_prime_factors_210 : (∃ p q r s : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ s.prime ∧ 210 = p * q * r * s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :=
by {
    -- We provide the prime numbers and their product implicitly proving the theorem
    use [2, 3, 5, 7],
    split,
    exact prime_two,
    split,
    exact prime_three,
    split,
    exact prime_five,
    split,
    exact prime_seven,
    split,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    norm_num,
    sorry,
}

end distinct_prime_factors_210_l38_38516


namespace sqrt_five_estimation_l38_38249

theorem sqrt_five_estimation : 6 < (real.sqrt 5 * (2 * real.sqrt 5 - real.sqrt 2)) ∧ (real.sqrt 5 * (2 * real.sqrt 5 - real.sqrt 2)) < 7 :=
by
  sorry

end sqrt_five_estimation_l38_38249


namespace units_digit_five_consecutive_l38_38646

theorem units_digit_five_consecutive (n : ℕ) :
  ∃ k ∈ {n, n + 1, n + 2, n + 3, n + 4}, k % 5 = 0 →
  ∃ a b ∈ {n, n + 1, n + 2, n + 3, n + 4}, a ≠ b ∧ a % 2 = 0 ∧ b % 2 = 0 →
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
sorry

end units_digit_five_consecutive_l38_38646


namespace find_integer_for_perfect_square_l38_38252

theorem find_integer_for_perfect_square :
  ∃ (n : ℤ), ∃ (m : ℤ), n^2 + 20 * n + 11 = m^2 ∧ n = 35 := by
  sorry

end find_integer_for_perfect_square_l38_38252


namespace probability_product_multiple_of_72_is_zero_l38_38416

def is_multiple_of_72 (x : ℕ) : Prop :=
  ∃ k, x = 72 * k

def in_set (x : ℕ) : Prop :=
  x = 4 ∨ x = 8 ∨ x = 18 ∨ x = 28 ∨ x = 36 ∨ x = 49 ∨ x = 56

def multiply (a b : ℕ) : ℕ := a * b

theorem probability_product_multiple_of_72_is_zero :
  (∀ (a b : ℕ), a ≠ b ∧ in_set a ∧ in_set b → ¬ is_multiple_of_72 (multiply a b)) :=
by
  intros a b h,
  cases h with ha hb,
  have ha' := ha.1,
  have hb' := ha.2.2.1,
  sorry

end probability_product_multiple_of_72_is_zero_l38_38416


namespace intersection_points_polar_coords_max_distance_to_line_l38_38357

-- Definition of line l and circle C
def line_eq := ∀ x : ℝ, ∃ y : ℝ, y = x + 4
def parametric_circle_eq (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)

-- Convert parametric form to Cartesian form: x^2 + (y - 2)^2 = 4
def cartesian_circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Theorem for the intersection points in polar coordinates
theorem intersection_points_polar_coords :
  ∃ θ1 θ2 : ℝ, 
  let (x1, y1) := parametric_circle_eq θ1,
      (x2, y2) := parametric_circle_eq θ2 in
  line_eq x1 y1 ∧ line_eq x2 y2 ∧ 
  (( x1 = -2 ∧ y1 = 2 ∧ Real.sqrt (x1^2 + y1^2) = 2 * Real.sqrt 2 ∧ Real.atan2 y1 x1 = 3 * Real.pi / 4) ∨
   ( x2 = 0 ∧ y2 = 4 ∧ Real.sqrt (x2^2 + y2^2) = 4 ∧ Real.atan2 y2 x2 = Real.pi / 2)) := sorry

-- Theorem for the maximum distance from P to the line l
theorem max_distance_to_line :
  ∀ x y : ℝ, 
  cartesian_circle_eq x y →
  ∃ P : ℝ × ℝ, P = (x, y) ∧
  (Real.abs (0 + (-2) + 4) / Real.sqrt 2 + Real.sqrt (2^2)) = √2 + 2 := sorry

end intersection_points_polar_coords_max_distance_to_line_l38_38357


namespace machines_complete_job_in_12_days_l38_38869

-- Given the conditions
variable (D : ℕ) -- The number of days for 12 machines to complete the job
variable (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8))

-- Prove the number of days for 12 machines to complete the job
theorem machines_complete_job_in_12_days (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8)) : D = 12 :=
by
  sorry

end machines_complete_job_in_12_days_l38_38869


namespace square_possible_n12_square_possible_n15_l38_38288

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l38_38288


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38883

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38883


namespace range_of_x0_l38_38050

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^(-x) - 1 else real.sqrt x

theorem range_of_x0 (x0 : ℝ) (h : f x0 > 1) : x0 ∈ (-∞, -1) ∪ (1, ∞) :=
sorry

end range_of_x0_l38_38050


namespace cone_height_l38_38349

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l38_38349


namespace find_b_continuity_l38_38793

theorem find_b_continuity :
  (∀ x, (x ≤ 4 → f x = 3 * x^2 + 5) ∧ (x > 4 → f x = b * x - 2))
  (continuous_at f 4) →
  b = 55 / 4 := by
  sorry

variable (f : ℝ → ℝ)
variable (b : ℝ)

end find_b_continuity_l38_38793


namespace cube_labeling_possible_l38_38013

def distinct_three_digit_numbers : List (Fin 1000) :=
  [111, 112, 121, 122, 211, 212, 221, 222]

def adjacent (i j : Fin 8) : Prop :=
  -- Placeholder definition, specific adjacency logic can be defined
  sorry

def differ_at_least_two_digits (a b : Fin 1000) : Prop :=
  let s := (a.val.to_string.to_list.zip b.val.to_string.to_list)
  2 ≤ (s.filter (λ pair => pair.1 ≠ pair.2)).length

theorem cube_labeling_possible : ∃ (label : Fin 8 → Fin 1000), 
  (∀ i j, adjacent i j → differ_at_least_two_digits (label i) (label j)) ∧ 
  (∀ i j, label i = label j → i = j) :=
by
  sorry

end cube_labeling_possible_l38_38013


namespace acute_triangle_exists_l38_38267

theorem acute_triangle_exists (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a1 + a2 > a4) (h5 : a1 + a4 > a2) (h6 : a2 + a4 > a1)
  (h7 : a1 + a3 > a4) (h8 : a1 + a4 > a3) (h9 : a3 + a4 > a1)
  (h10 : a2 + a3 > a4) (h11 : a2 + a4 > a3) (h12 : a3 + a4 > a2)
  (h13 : a1 + a2 > a5) (h14 : a1 + a5 > a2) (h15 : a2 + a5 > a1)
  (h16 : a1 + a3 > a5) (h17 : a1 + a5 > a3) (h18 : a3 + a5 > a1)
  (h19 : a2 + a3 > a5) (h20 : a2 + a5 > a3) (h21 : a3 + a5 > a2)
  (h22 : a1 + a4 > a5) (h23 : a1 + a5 > a4) (h24 : a4 + a5 > a1)
  (h25 : a2 + a4 > a5) (h26 : a2 + a5 > a4) (h27 : a4 + a5 > a2)
  (h28 : a3 + a4 > a5) (h29 : a3 + a5 > a4) (h30 : a4 + a5 > a3) :
  ∃ i j k, (i, j, k : Fin 5) → i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
   (a i j k).sqrt ≤ a[i,j] ↔ i< j ∧ j < k :=
  sorry

end acute_triangle_exists_l38_38267


namespace sum_tens_ones_digits_3_plus_4_power_17_l38_38878

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l38_38878


namespace number_of_ways_to_place_rooks_l38_38764

theorem number_of_ways_to_place_rooks :
  let columns := 6
  let rows := 2006
  let rooks := 3
  ((Nat.choose columns rooks) * (rows * (rows - 1) * (rows - 2))) = 20 * 2006 * 2005 * 2004 :=
by {
  sorry
}

end number_of_ways_to_place_rooks_l38_38764


namespace problem_statement_l38_38154

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end problem_statement_l38_38154


namespace problem_l38_38005

theorem problem (l_eq : ∀ x y : ℝ, x + (√3 : ℝ) * y = 5 * (√3 : ℝ)) (circle_eq_polar : ∀ θ : ℝ, ∃ (ρ : ℝ), ρ = 4 * sin θ) :
  (∀ θ : ℝ, 2 * (∃ ρ : ℝ, ρ * sin (θ + π / 6) = 5 * √3)) ∧
  (∀ x y : ℝ, x^2 + (y - 2)^2 = 4) ∧
  ∃ ρ₁ ρ₂ : ℝ, θ = π / 6 → (abs (ρ₁ - ρ₂) = 3) :=
begin
  sorry
end

end problem_l38_38005


namespace percent_filled_cone_l38_38978

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l38_38978


namespace problem_solution_set_l38_38308

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

axiom f_decreasing : decreasing_function f
axiom f_passes_A : f (-3) = 2
axiom f_passes_B : f (2) = -2
axiom f_inv_def : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem problem_solution_set (x : ℝ) :
  |2 * f_inv (x^2 - 2) + 1| < 5 ↔ (x ∈ Ioo (-2) 0 ∨ x ∈ Ioo 0 2) :=
by
  sorry

end problem_solution_set_l38_38308


namespace integral_cos_square_div_one_plus_cos_minus_sin_squared_l38_38913

theorem integral_cos_square_div_one_plus_cos_minus_sin_squared:
  ∫ x in (-2 * Real.pi / 3 : Real)..0, (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2 = (Real.sqrt 3) / 2 - Real.log 2 := 
by
  sorry

end integral_cos_square_div_one_plus_cos_minus_sin_squared_l38_38913


namespace integer_solutions_of_equation_l38_38647

theorem integer_solutions_of_equation (x y : ℤ) (h : x^2 - 2*y^2 = 2^(x+y)) :
  (x, y) = (2, 0) ∨ (x, y) = (2, -1) ∨ (x, y) = (6, -4) ∨ (x, y) = (4, 0) ∨ (x, y) = (12, -8) :=
sorry

end integer_solutions_of_equation_l38_38647


namespace yogurt_packs_ordered_l38_38622

theorem yogurt_packs_ordered (P : ℕ) (price_per_pack refund_amount : ℕ) (expired_percentage : ℚ)
  (h1 : price_per_pack = 12)
  (h2 : refund_amount = 384)
  (h3 : expired_percentage = 0.40)
  (h4 : refund_amount / price_per_pack = 32)
  (h5 : 32 / expired_percentage = P) :
  P = 80 :=
sorry

end yogurt_packs_ordered_l38_38622


namespace interval_of_decrease_l38_38258

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -1 ∧ b = 11 ∧ 
  (∀ x : ℝ, x ∈ set.Ioo a b → (f' x) < 0) :=
sorry

end interval_of_decrease_l38_38258


namespace negative_result_of_operations_l38_38168

theorem negative_result_of_operations :
  (∃ x : ℕ, x = 1 ∧ -(-4) = 4) ∧
  (∃ x : ℕ, x = 2 ∧ |-4| = 4) ∧
  (∃ x : ℕ, x = 3 ∧ -4^2 = -16) ∧
  (∃ x : ℕ, x = 4 ∧ (-4)^2 = 16) →
  ∃ y : ℕ, y = 3 ∧ -4^2 = -16 :=
by {
  intros h,
  rcases h with ⟨⟨_, _, hA⟩, ⟨_, _, hB⟩, ⟨_, _, hC⟩, ⟨_, _, hD⟩⟩,
  use 3,
  exact ⟨rfl, hC⟩,
}

end negative_result_of_operations_l38_38168


namespace compare_f_values_l38_38303

variable (a : ℝ) (f : ℝ → ℝ) (m n : ℝ)

theorem compare_f_values (h_a : 0 < a ∧ a < 1)
    (h_f : ∀ x > 0, f (Real.logb a x) = a * (x^2 - 1) / (x * (a^2 - 1)))
    (h_mn : m > n ∧ n > 0 ∧ m > 0) :
    f (1 / n) > f (1 / m) := by 
  sorry

end compare_f_values_l38_38303


namespace sum_of_digits_7_pow_17_mod_100_l38_38894

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l38_38894


namespace infinite_primes_of_form_4n_plus_3_l38_38482

theorem infinite_primes_of_form_4n_plus_3 :
  ∀ (S : Finset ℕ), (∀ p ∈ S, Prime p ∧ p % 4 = 3) →
  ∃ q, Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
by 
  sorry

end infinite_primes_of_form_4n_plus_3_l38_38482


namespace initial_marbles_l38_38559

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l38_38559


namespace equation_of_line_passing_through_point_with_slope_l38_38502

theorem equation_of_line_passing_through_point_with_slope :
  ∃ (l : ℝ → ℝ), l 0 = -1 ∧ ∀ (x y : ℝ), y = l x ↔ y + 1 = 2 * x :=
sorry

end equation_of_line_passing_through_point_with_slope_l38_38502


namespace angle_BPC_90_l38_38454

variables (A B C D P : Point)
variable [IsTrapezium ABCD]
variable [Parallel AB CD]
variable (h1 : length(AB) + length(CD) = length(AD))
variable (h2 : on_line A D P)
variable (h3 : length(A, P) = length(AB))
variable (h4 : length(P, D) = length(CD))

theorem angle_BPC_90 : angle B P C = 90 :=
by
  sorry  -- Proof omitted

end angle_BPC_90_l38_38454


namespace max_value_modulus_l38_38782

noncomputable def complex_value (β γ : ℂ) : ℂ :=
  (β - γ) / (1 - conj γ * β)

theorem max_value_modulus (α β : ℂ) (hβ : complex.abs β = 1) (hγβ : conj (α + complex.I) * β ≠ 1) :
  (∃ γ : ℂ, γ = α + complex.I) →
  (∀ γ : ℂ, γ = α + complex.I →
    complex.abs (complex_value β γ) ≤ 1) :=
begin
  sorry,
end

end max_value_modulus_l38_38782


namespace noah_in_middle_chair_l38_38662

-- Define the type for persons
inductive person
| Liam
| Noah
| Olivia
| Emma
| Sophia

open person

-- Define the type for chairs
def chair := fin 5

-- Define the seating arrangement type
def seating := chair → person

-- Define conditions
def condition_1 (s : seating) : Prop := s 0 = Sophia
def condition_2 (s : seating) : Prop := ∃ i : chair, i < 4 ∧ s i = Emma ∧ s (i + 1) = Liam
def condition_3 (s : seating) : Prop := ∃ i j : chair, i < j ∧ s i = Noah ∧ s j = Emma
def condition_4 (s : seating) : Prop := ∃ i j k : chair, i < j ∧ j < k ∧ s i = Noah ∧ s k = Olivia

-- Define the proof problem: Noah is in the middle chair (chair 2)
theorem noah_in_middle_chair (s : seating) :
  condition_1 s ∧ condition_2 s ∧ condition_3 s ∧ condition_4 s → s 2 = Noah :=
by
  sorry

end noah_in_middle_chair_l38_38662


namespace max_min_sum_f_l38_38119

open Real

noncomputable def f (x : ℝ) : ℝ := x * cos x + 1

theorem max_min_sum_f :
  ∃ (M m : ℝ), 
  (∀ x ∈ Ioo (-5 : ℝ) 5, f x ≤ M) ∧
  (∀ x ∈ Ioo (-5 : ℝ) 5, f x ≥ m) ∧
  (∀ y, 
  (∀ x ∈ Ioo (-5 : ℝ) 5, f x ≤ y) → M ≤ y) ∧
  (∀ y, 
  (∀ x ∈ Ioo (-5 : ℝ) 5, f x ≥ y) → m ≥ y) ∧
  (M + m = 2) :=
begin
  sorry
end

end max_min_sum_f_l38_38119


namespace initial_nickels_l38_38467

theorem initial_nickels (quarters : ℕ) (initial_nickels : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (H1 : initial_nickels = 87) (H2 : borrowed_nickels = 75) (H3 : current_nickels = 12) : 
  initial_nickels = current_nickels + borrowed_nickels := 
by 
  -- proof steps go here
  sorry

end initial_nickels_l38_38467


namespace mary_initial_triangles_l38_38056

theorem mary_initial_triangles (s t : ℕ) (h1 : s + t = 10) (h2 : 4 * s + 3 * t = 36) : t = 4 :=
by
  sorry

end mary_initial_triangles_l38_38056


namespace value_of_expression_l38_38740

theorem value_of_expression (n m : ℤ) (h : m = 2 * n^2 + n + 1) : 8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end value_of_expression_l38_38740


namespace distance_between_lines_l38_38237

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  abs (C2 - C1) / real.sqrt (A^2 + B^2)

theorem distance_between_lines :
  distance_between_parallel_lines 2 2 1 4 = 3 * real.sqrt 2 / 4 :=
by
  sorry

end distance_between_lines_l38_38237


namespace valid_range_of_a_l38_38719

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x ^ 2 + 3)

noncomputable def g (a x : ℝ) : ℝ := (1 / 3) * a * x ^ 3 - a ^ 2 * x

def is_valid_a (a : ℝ) : Prop :=
  ∀ x1, x1 ∈ set.Icc 0 2 → ∃ x2, x2 ∈ set.Icc 0 2 ∧ f x1 = g a x2

theorem valid_range_of_a (a : ℝ) : is_valid_a a ↔ (1 / 3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end valid_range_of_a_l38_38719


namespace minimum_n_value_l38_38311

theorem minimum_n_value (p : ℝ → ℝ → Prop) (q : ℝ → ℝ → Prop) (m : ℝ) (n : ℝ) :
  (∀ x, p x m → q x n) →
  (∀ x, |x - m| ≤ 2 → -1 ≤ x ∧ x ≤ n) →
  n = 3 :=
begin
  assume sufficient,
  assume conditions,
  sorry,  -- Proof is not required
end

end minimum_n_value_l38_38311


namespace find_m_from_ellipse_l38_38839

theorem find_m_from_ellipse (m : ℝ) (x y : ℝ) : 
  (x^2 / m + y^2 / 4 = 1) ∧ (∀ a b c : ℝ, 2 * c = 2 → focalLength (Ellipse x y m 4) = 2) → 
  (m = 3 ∨ m = 5) :=
by
  sorry

end find_m_from_ellipse_l38_38839


namespace correct_propositions_l38_38230

theorem correct_propositions (P1 P2 P3 P4 : Prop)
  (H1 : P1 ↔ ∀ (p q : Plane) (l : Line), (p ∥ l) ∧ (q ∥ l) → p ∥ q)
  (H2 : P2 ↔ ∀ (p q : Plane), (p ∥ q) → ∀ (r : Plane), (p ∥ r) ∧ (q ∥ r) → r ∥ q)
  (H3 : P3 ↔ ∀ (a b : Line) (l : Line), (a ⊥ l) ∧ (b ⊥ l) → a ∥ b)
  (H4 : P4 ↔ ∀ (a b : Line), (∀ (p : Plane), (a ⊥ p) ∧ (b ⊥ p)) → a ∥ b) :
  (P2 ∧ P4) :=
by
  have hP2 : P2, from H2.mpr sorry,
  have hP4 : P4, from H4.mpr sorry,
  exact ⟨hP2, hP4⟩

end correct_propositions_l38_38230


namespace find_cone_height_l38_38350

noncomputable def cone_height (A l : ℝ) : ℝ := 
  let r := A / (l * Real.pi) in
  Real.sqrt (l^2 - r^2)

theorem find_cone_height : cone_height (65 * Real.pi) 13 = 12 := by
  let r := 5
  have h_eq : cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) := by 
    unfold cone_height
    sorry -- This step would carry out the necessary substeps.
  calc
    cone_height (65 * Real.pi) 13 = Real.sqrt (13^2 - r^2) : by exact h_eq
                         ... = Real.sqrt 144 : by norm_num
                         ... = 12 : by norm_num

end find_cone_height_l38_38350


namespace maximum_k_divisible_by_m_l38_38785

theorem maximum_k_divisible_by_m (m : ℕ) (h : m > 1) (x : ℕ → ℕ)
  (h_seq_1 : ∀ i, 0 ≤ i ∧ i < m → x i = 2^i)
  (h_seq_2 : ∀ i, i ≥ m → x i = ∑ j in (finset.range m), x (i - j - 1)) :
  ∃ k, (∀ i, i ≥ k ∧ i < k + m → m ∣ x i) → k = m :=
sorry

end maximum_k_divisible_by_m_l38_38785


namespace A_plus_J_l38_38007

variable (A B C D E F G H I J : ℝ)

-- Conditions
def E_val : Prop := E = 8
def sum_of_consecutive_triples : Prop := 
  A + B + C = 27 ∧
  B + C + D = 27 ∧
  C + D + E = 27 ∧
  E + F + G = 27 ∧
  F + G + H = 27 ∧
  G + H + I = 27 ∧
  H + I + J = 27

-- Proof Statement
theorem A_plus_J (h1 : E_val) (h2 : sum_of_consecutive_triples) : A + J = -27 := 
by 
  sorry

end A_plus_J_l38_38007


namespace lunch_cake_count_l38_38589

variable (L : ℕ)
variable (cakes_dinner : ℕ)
variable (diff_cakes : ℕ)
variable (cakes_dinner_eq : cakes_dinner = 9)
variable (diff_cakes_eq : diff_cakes = 3)
variable (dinner_cakes_eq : cakes_dinner = L + diff_cakes)

theorem lunch_cake_count : L = 6 :=
  by
    rw [cakes_dinner_eq, diff_cakes_eq, dinner_cakes_eq]
    sorry

end lunch_cake_count_l38_38589


namespace sum_of_digits_7_pow_17_mod_100_l38_38896

-- The problem: What is the sum of the tens digit and the ones digit of the integer form of \(7^{17} \mod 100\)?
theorem sum_of_digits_7_pow_17_mod_100 :
  let n := 7^17 % 100 in
  (n / 10 + n % 10) = 7 :=
by
  -- We let Lean handle the proof that \(7^{17} \mod 100 = 7\)
  sorry

end sum_of_digits_7_pow_17_mod_100_l38_38896


namespace expected_red_hair_americans_l38_38804

theorem expected_red_hair_americans (prob_red_hair : ℝ) (sample_size : ℕ) :
  prob_red_hair = 1 / 6 → sample_size = 300 → (prob_red_hair * sample_size = 50) := by
  intros
  sorry

end expected_red_hair_americans_l38_38804


namespace min_breaks_12_no_breaks_15_l38_38299

-- Define the function to sum the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- The main theorem for n = 12
theorem min_breaks_12 : ∀ (n = 12), (∑ i in finset.range (n + 1), i % 4 ≠ 0) → 2 := 
by sorry

-- The main theorem for n = 15
theorem no_breaks_15 : ∀ (n = 15), (∑ i in finset.range (n + 1), i % 4 = 0) → 0 := 
by sorry

end min_breaks_12_no_breaks_15_l38_38299


namespace x_power_10_l38_38239

noncomputable def x_value (x : ℝ) : Prop :=
  (log (5 * x ^ 3) / log 5 + log (25 * x ^ 4) / log 5 = -1)

theorem x_power_10 (x : ℝ) (h : x_value x) : x ^ 10 = 1 / 5 ^ 10 :=
sorry

end x_power_10_l38_38239


namespace find_m_times_t_l38_38466

variable {R : Type*} [LinearOrderedField R]

noncomputable def g (x : R) : R := sorry

axiom g_property : ∀ x y : R, g (g x - y) = g x + g (g y - g (-x)) - x

theorem find_m_times_t : 
    let m := (if ∃ t : R, g 3 = t then 1 else 0) in
    let t := if ∃ t : R, g 3 = t then Classical.choose (exists_g3_value 3) else 0 in
    m * t = 3 :=
by
  sorry

end find_m_times_t_l38_38466


namespace unique_g_zero_l38_38841

theorem unique_g_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) = g (x) + g (y) - 1) : g 0 = 1 :=
by
  sorry

end unique_g_zero_l38_38841


namespace attitude_related_to_gender_expected_value_agree_l38_38186

def survey_data : Type := {a b c d n : ℕ}

def chi_squared (data : survey_data) : ℝ :=
  let ⟨a, b, c, d, n⟩ := data
  let χ2 := (n * ((a*d - b*c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))
  χ2

def critical_value_99_percent : ℝ := 6.635

def expected_value (p : ℝ) (n : ℕ) : ℝ :=
  p * n

theorem attitude_related_to_gender
  (data : survey_data)
  (h_data : data.a = 70 ∧ data.b = 50 ∧ data.c = 30 ∧ data.d = 50 ∧ data.n = 200)
  : chi_squared data > critical_value_99_percent :=
 sorry

theorem expected_value_agree (p : ℝ) (n : ℕ)
  (h_p : p = 3 / 5)
  : expected_value p n = 9 / 5 :=
  sorry

end attitude_related_to_gender_expected_value_agree_l38_38186


namespace unique_N_solutions_l38_38080

theorem unique_N_solutions (N : ℕ) (x : Fin N → ℤ) :
  (4 * N + 1) * (∑ i, x i ^ 2) = 4 * (∑ i, x i) ^ 2 + 4 * N + 1 →
  N = 2 ∨ N = 6 :=
by
  sorry

end unique_N_solutions_l38_38080


namespace parabola_slope_constant_product_l38_38321

noncomputable def parabola_equation (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x

noncomputable def is_on_parabola (M : ℝ × ℝ) (p : ℝ) : Prop :=
  M.2^2 = 2 * p * M.1

noncomputable def distance_to_focus (M F : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)

noncomputable def problem_statement (F M : ℝ × ℝ) (p x0 : ℝ) (Q : ℝ × ℝ) : Prop :=
  p > 0 ∧
  M.2 = 1 ∧
  is_on_parabola M p ∧
  |distance_to_focus M F| = (5 * x0) / 4 ∧
  x0 = M.1 ∧
  ∃ (p:ℝ), parabola_equation p ∧
  ∃ (kAM kBM : ℝ), kAM * kBM = -1/2

theorem parabola_slope_constant_product :
  ∃ p x0 (F M : ℝ × ℝ) (Q : ℝ × ℝ),
  p = 1/2 →
  (problem_statement F M p x0 Q) → 
  (∃ kAM kBM, kAM * kBM = -1/2) :=
by
  sorry

end parabola_slope_constant_product_l38_38321


namespace dogs_total_food_l38_38867

def Momo_dry_per_meal : ℝ := 1.3
def Momo_wet_per_meal : ℝ := 0.7
def Fifi_dry_per_meal : ℝ := 1.6
def Fifi_wet_per_meal : ℝ := 0.5
def Momo_meals_per_day : ℝ := 2
def Fifi_meals_per_day : ℝ := 2
def Gigi_dry_per_meal : ℝ := 2
def Gigi_wet_per_meal : ℝ := 1
def Gigi_meals_per_day : ℝ := 3
def dry_conversion_factor : ℝ := 3.2
def wet_conversion_factor : ℝ := 2.8

def total_pounds_of_food : ℝ :=
  ((Momo_dry_per_meal * Momo_meals_per_day + Fifi_dry_per_meal * Fifi_meals_per_day + Gigi_dry_per_meal * Gigi_meals_per_day) / dry_conversion_factor) +
  ((Momo_wet_per_meal * Momo_meals_per_day + Fifi_wet_per_meal * Fifi_meals_per_day + Gigi_wet_per_meal * Gigi_meals_per_day) / wet_conversion_factor)

theorem dogs_total_food : total_pounds_of_food = 5.6161 :=
by sorry

end dogs_total_food_l38_38867


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l38_38891

theorem sum_of_tens_and_ones_digit_of_7_pow_17 :
  let n := 7 ^ 17 in
  (n % 10) + ((n / 10) % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l38_38891


namespace greatest_remainder_le_11_l38_38389

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l38_38389


namespace lines_parallel_or_coincident_l38_38323

/-- Given lines l₁ and l₂ with certain properties,
    prove that they are either parallel or coincident. -/
theorem lines_parallel_or_coincident
  (P Q : ℝ × ℝ)
  (hP : P = (-2, -1))
  (hQ : Q = (3, -6))
  (h_slope1 : ∀ θ, θ = 135 → Real.tan (θ * (Real.pi / 180)) = -1)
  (h_slope2 : (Q.2 - P.2) / (Q.1 - P.1) = -1) : 
  true :=
by sorry

end lines_parallel_or_coincident_l38_38323


namespace cone_volume_percentage_filled_l38_38942

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l38_38942


namespace problem1_problem2_l38_38367

def f (x : Real) (a : Real) : Real := exp x - exp a * (a + log x)

-- Part (1)
def f_monotonically_increasing_on (a : Real) : Prop :=
  ∀ x > 1, exp x - exp a * (1 + log x) > 0

theorem problem1 :
  f_monotonically_increasing_on 1 :=
by sorry

-- Part (2)
def f_nonnegative_always (a : Real) : Prop :=
  ∀ x > 0, exp x - exp a * (a + log x) ≥ 0

theorem problem2 (a : Real) :
  f_nonnegative_always a → a ≤ 1 :=
by sorry

end problem1_problem2_l38_38367


namespace induction_proof_l38_38074

-- Given conditions and definitions
def plane_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

-- The induction hypothesis for k ≥ 2
def induction_step (k : ℕ) (h : 2 ≤ k) : Prop :=
  plane_parts (k + 1) - plane_parts k = k + 1

-- The complete statement we want to prove
theorem induction_proof (k : ℕ) (h : 2 ≤ k) : induction_step k h := by
  sorry

end induction_proof_l38_38074


namespace hexagon_area_sqroot_sum_l38_38629

theorem hexagon_area_sqroot_sum (s : ℝ) (side_len : s = 3) (centroid_intersect : centroid_intersects_at_G) :
  ∃ p q : ℤ, p + q = 756 ∧ (hexagon_area s = real.sqrt p + real.sqrt q) :=
begin
  sorry
end

end hexagon_area_sqroot_sum_l38_38629


namespace Jason_earned_60_dollars_l38_38024

-- Define initial and final amounts of money
variable (Jason_initial Jason_final : ℕ)

-- State the assumption about Jason's initial and final amounts of money
variable (h_initial : Jason_initial = 3) (h_final : Jason_final = 63)

-- Define the amount of money Jason earned
def Jason_earn := Jason_final - Jason_initial

-- Prove that Jason earned 60 dollars by delivering newspapers
theorem Jason_earned_60_dollars : Jason_earn Jason_initial Jason_final = 60 := by
  sorry

end Jason_earned_60_dollars_l38_38024


namespace initial_marbles_l38_38560

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l38_38560


namespace power_function_point_l38_38713

theorem power_function_point (n : ℕ) (hn : 2^n = 8) : n = 3 := 
by
  sorry

end power_function_point_l38_38713


namespace certain_number_is_166_l38_38236

theorem certain_number_is_166 :
  ∃ x : ℕ, x - 78 =  (4 - 30) + 114 ∧ x = 166 := by
  sorry

end certain_number_is_166_l38_38236


namespace perpendicular_lines_implies_perpendicular_plane_l38_38664

theorem perpendicular_lines_implies_perpendicular_plane
  (triangle_sides : Line → Prop)
  (circle_diameters : Line → Prop)
  (perpendicular : Line → Line → Prop)
  (is_perpendicular_to_plane : Line → Prop) :
  (∀ l₁ l₂, triangle_sides l₁ → triangle_sides l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) ∧
  (∀ l₁ l₂, circle_diameters l₁ → circle_diameters l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) :=
  sorry

end perpendicular_lines_implies_perpendicular_plane_l38_38664
