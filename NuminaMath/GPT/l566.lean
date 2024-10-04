import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Linear.Equiv
import Mathlib.Algebra.Order.Sqrt
import Mathlib.Algebra.Order.Sub
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Probability.Finite
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.NumberTheory.Cyclotomic.Basic
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.InfiniteSum
import Mathlib.Topology.Basic
import Mathlibimportant_emails :=
import Mathlibimportant_emails =
import analysis.special_functions.trigonometric

namespace numbers_satisfy_conditions_l566_566555

theorem numbers_satisfy_conditions :
  ∃ (nums : list ℤ), 
    (nums.length = 2017) ∧
    (∀ (subset : list ℤ), subset.length = 7 → (subset.sum (λ n, n^2)) = 7) ∧
    (∀ (subset : list ℤ), subset.length = 11 → (subset.sum id > 0)) ∧
    (nums.sum id % 9 = 0) ∧
    (nums.count (-1) = 5) ∧
    (nums.count 1 = 2012) :=
by
  sorry

end numbers_satisfy_conditions_l566_566555


namespace probability_two_red_balls_l566_566950

def total_balls : ℕ := 15
def red_balls_initial : ℕ := 7
def blue_balls_initial : ℕ := 8
def red_balls_after_first_draw : ℕ := 6
def remaining_balls_after_first_draw : ℕ := 14

theorem probability_two_red_balls :
  (red_balls_initial / total_balls) *
  (red_balls_after_first_draw / remaining_balls_after_first_draw) = 1 / 5 :=
by sorry

end probability_two_red_balls_l566_566950


namespace youngest_sibling_age_l566_566502

def ages_correct (O M S Y : ℝ) : Prop :=
  -- The sum of the ages is 100
  O + M + S + Y = 100 ∧
  -- The youngest sibling's age is the difference between one-third of the middle sibling's age and one-fifth of the oldest sibling's age
  Y = (1/3) * M - (1/5) * O ∧
  -- The middle sibling is 8 years younger than the oldest sibling
  M = O - 8 ∧
  -- The second youngest sibling is 6.5 years older than half of the youngest sibling's age
  S = (1/2) * Y + 6.5

theorem youngest_sibling_age :
  ∃ Y : ℝ, ∀ O M S : ℝ, ages_correct O M S Y → abs (Y - 3.83) < 0.01 :=
begin
  sorry
end

end youngest_sibling_age_l566_566502


namespace minimum_value_of_expression_l566_566056

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_constraint : x * y + z ^ 2 = 8) :
  (∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y + z ^ 2 = 8 ∧ ((x+y)/z + (y+z)/x^2 + (z+x)/y^2) = 4 :=
begin
  sorry
end

end minimum_value_of_expression_l566_566056


namespace geometric_sequence_sum_l566_566019

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l566_566019


namespace minimum_positive_period_of_f_l566_566488

-- Define the function f(x)
noncomputable def f : ℝ → ℝ := 
  λ x, (sqrt 3 * sin x + cos x) * (sqrt 3 * cos x - sin x)

-- State the theorem
theorem minimum_positive_period_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π) :=
sorry

end minimum_positive_period_of_f_l566_566488


namespace polynomial_has_close_roots_l566_566054

open Complex Polynomial -- Opens necessary namespaces

theorem polynomial_has_close_roots (d : ℕ) (hd : d ≥ 13) (P : Polynomial ℂ)
  (hdeg : P.degree = d) 
  (hcoeff : ∀ n, 0 ≤ n ∧ n ≤ d → P.coeff n = P.coeff (d - n))
  (hnodouble : ∀ (z : ℂ), (P.derivative.eval z ≠ 0 → P.eval z ≠ 0)) :
  ∃ z1 z2 : ℂ, z1 ≠ z2 ∧ |z1 - z2| < 1 := 
sorry

end polynomial_has_close_roots_l566_566054


namespace max_area_ABC_l566_566015

theorem max_area_ABC (a : ℝ) (ha : 2 * sqrt 2 - 2 < a ∧ a < 2 * sqrt 2 + 2) :
  let b := sqrt 2 * a in
  let c := 2 in
  ∃ S, ∀ a (b := sqrt 2 * a) (c := 2),
    S = (sqrt 2 / 2) * a^2 * (sqrt ((-a^4 + 24 * a^2 - 16) / 8)) →
    S ≤ 2 * sqrt 2 :=
sorry

end max_area_ABC_l566_566015


namespace lines_intersecting_skew_lines_l566_566377

theorem lines_intersecting_skew_lines 
  (a b : ℝ) -- two skew lines
  (α : ℝ) -- an acute angle
  (h_skew_perpendicular : a ≠ b ∧ (a * b = 0)) -- skew lines are perpendicular
  (h_acute : 0 < α ∧ α < π / 2) -- α is acute
  : (if α ≤ π / 4 then 0 else 4) = number_of_intersecting_lines a b α :=
sorry

end lines_intersecting_skew_lines_l566_566377


namespace area_of_stripe_l566_566218

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end area_of_stripe_l566_566218


namespace extreme_values_at_1_and_2_range_of_c_l566_566803

def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_at_1_and_2 (a b c : ℝ) 
  (h1 : f' a b 1 = 0) (h2 : f' a b 2 = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem range_of_c (c : ℝ) 
  (h : ∀ x ∈ Icc (0 : ℝ) 3, f -3 4 c x < c^2) : 
  c < -1 ∨ 9 < c :=
by sorry

end extreme_values_at_1_and_2_range_of_c_l566_566803


namespace number_of_liars_l566_566407

theorem number_of_liars {n : ℕ} (h1 : n ≥ 1) (h2 : n ≤ 200) (h3 : ∃ k : ℕ, k < n ∧ k ≥ 1) : 
  (∃ l : ℕ, l = 199 ∨ l = 200) := 
sorry

end number_of_liars_l566_566407


namespace range_of_a_l566_566943

noncomputable def f (a x : ℝ) : ℝ := log a (2 - a * x)

theorem range_of_a (a : ℝ) : (∀ x y ∈ Iic (1 : ℝ), x < y → f a y < f a x) → (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l566_566943


namespace martin_berry_expenditure_l566_566806

theorem martin_berry_expenditure : 
  (let daily_consumption := 1 / 2
       berry_price := 2
       days := 30 in
   daily_consumption * days * berry_price = 30) :=
by
  sorry

end martin_berry_expenditure_l566_566806


namespace inequality_in_interval_l566_566686

theorem inequality_in_interval (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) : x^2 + (cos x)^2 > 1 :=
sorry

end inequality_in_interval_l566_566686


namespace max_value_sqrt_cubed_l566_566429

theorem max_value_sqrt_cubed (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  real.cbrt (a * b * c) + real.cbrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 := 
sorry

end max_value_sqrt_cubed_l566_566429


namespace problem_statement_l566_566329

theorem problem_statement (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  2014 = (a^2 + b^2) * (c^3 - d^3) :=
by
  -- Definitions from the problem
  let a := 5
  let b := 9
  let c := 3
  let d := 2
  
  -- Assertions from the problem conditions
  have h1 : 5^2 + 9^2 = 106 := by norm_num
  have h2 : 3^3 - 2^3 = 19 := by norm_num
  have h3 : 106 * 19 = 2014 := by norm_num

  -- Final assertion proving the given statement
  exact h3

end problem_statement_l566_566329


namespace number_of_multiples_of_15_l566_566725

theorem number_of_multiples_of_15 (a b : ℕ) (h₁ : a = 15) (h₂ : b = 305) : 
  ∃ n : ℕ, n = 20 ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → (15 * k) ≥ a ∧ (15 * k) ≤ b := by
  sorry

end number_of_multiples_of_15_l566_566725


namespace max_value_sqrt_cubed_l566_566430

theorem max_value_sqrt_cubed (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  real.cbrt (a * b * c) + real.cbrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 := 
sorry

end max_value_sqrt_cubed_l566_566430


namespace part1_part2_l566_566342

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 - 5x + 4 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

-- Part (1): Show the range of m where A ∩ B = B is m ≤ 1.
theorem part1 (m : ℝ) : (A ∩ B m = B m) → m ≤ 1 := 
sorry

-- Part (2): Show the range of m where q is necessary but not sufficient for p is m ≥ 2.
theorem part2 (m : ℝ) (p : ℝ → Prop) (q : ℝ → Prop) (h_p : ∀ x, p x ↔ x ∈ A) (h_q : ∀ x, q x ↔ x ∈ B m) :
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) → m ≥ 2 :=
sorry

end part1_part2_l566_566342


namespace tom_age_is_19_l566_566898

-- Define the ages of Carla, Tom, Dave, and Emily
variable (C : ℕ) -- Carla's age

-- Conditions
def tom_age := 2 * C - 1
def dave_age := C + 3
def emily_age := C / 2

-- Sum of their ages equating to 48
def total_age := C + tom_age C + dave_age C + emily_age C

-- Theorem to be proven
theorem tom_age_is_19 (h : total_age C = 48) : tom_age C = 19 := 
by {
  sorry
}

end tom_age_is_19_l566_566898


namespace second_solution_sugar_content_l566_566934

theorem second_solution_sugar_content :
  ∀ (W : ℝ) (S : ℝ), (1/4 * W)*(10/100) + (3/4 * W)*(S/100) = 0.14 * W → S = 26 := 
by 
  intros W S h
  have h1 : 10 / 100 * W / 4 + S / 100 * W / 4 = 0.14 * W
  {
    calc
      (1/4 * W) * (10 / 100) + (3/4 * W) * (S / 100) = 0.075 * W + (S/400) * W : sorry
       ...  = (1 / 4) * W * 0.10 + (1/4 * W) * (S / 100 * 100/400) : sorry
       ... = 0.095 : sorry
  },
  exact sorry

end second_solution_sugar_content_l566_566934


namespace least_possible_value_z_minus_x_l566_566932

theorem least_possible_value_z_minus_x (x y z : ℤ) (h1 : Even x) (h2 : Odd y) (h3 : Odd z) (h4 : x < y) (h5 : y < z) (h6 : y - x > 5) : z - x = 9 := 
sorry

end least_possible_value_z_minus_x_l566_566932


namespace liam_drinks_17_glasses_l566_566452

def minutes_in_hours (h : ℕ) : ℕ := h * 60

def total_time_in_minutes (hours : ℕ) (extra_minutes : ℕ) : ℕ := 
  minutes_in_hours hours + extra_minutes

def rate_of_drinking (drink_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / drink_interval

theorem liam_drinks_17_glasses : 
  rate_of_drinking 20 (total_time_in_minutes 5 40) = 17 :=
by
  sorry

end liam_drinks_17_glasses_l566_566452


namespace sum_of_possible_values_l566_566797

variable {a b c d : ℝ}

-- conditions
def condition1 : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = -3 / 4

-- prove that
theorem sum_of_possible_values 
  (h : condition1) : 
  (a - d) * (b - c) / ((a - b) * (c - d)) = 3 / 4 :=
by 
  sorry

end sum_of_possible_values_l566_566797


namespace angle_of_line_eq_60_degrees_l566_566150

-- Define the slope of the line
def slope (line : ℝ → ℝ) : ℝ := 3.sqrt

-- Define the angle of inclination as an equality to the arc tangent of the slope
def angle_of_inclination (slope : ℝ) : ℝ := Real.arctan slope

-- Prove that the angle of inclination for the given line is 60 degrees (π/3 radians)
theorem angle_of_line_eq_60_degrees (θ : ℝ) (h : θ = Real.arctan (3.sqrt)) : θ = Real.pi / 3 :=
by
  sorry

end angle_of_line_eq_60_degrees_l566_566150


namespace molly_more_minutes_than_xanthia_l566_566922

-- Define the constants: reading speeds and book length
def xanthia_speed := 80  -- pages per hour
def molly_speed := 40    -- pages per hour
def book_length := 320   -- pages

-- Define the times taken to read the book in hours
def xanthia_time := book_length / xanthia_speed
def molly_time := book_length / molly_speed

-- Define the time difference in minutes
def time_difference_minutes := (molly_time - xanthia_time) * 60

theorem molly_more_minutes_than_xanthia : time_difference_minutes = 240 := 
by {
  -- Here the proof would go, but we'll leave it as a sorry for now.
  sorry
}

end molly_more_minutes_than_xanthia_l566_566922


namespace find_digits_number_l566_566770

theorem find_digits_number :
  ∀ (s : List ℕ), 
  (s.length = 1000 → 
  s.take 1 = [2] → 
  s.take 11.drop = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] → 
  s.take 311.drop 21 = (List.range 200 300) →
  ( ∃ (n : ℕ), 216 = n ∧ (s.nth 997 = n / 100) ∧ (s.nth 998 = (n % 100) / 10) ∧ (s.nth 999 = n % 10))) :=
begin
  assume s,
  assume hlen h1 h2 h3,
  have h4 : ∀ i, 0 < i → i < 200 -> ((s.nth i).get_or_else 0).digit_1 = 2,
  from sorry,
  have h5 : ∀ i, 200 ≤ i → i < 300 -> (s.nth i).get_or_else 0 / 100 = 2,
  from sorry,
  have h6 : ∀ i, 800 ≤ i → i < 3000 -> (s.nth i).get_or_else 0 / 1000 = 2,
  from sorry,
  use 216,
  split; [refl, split; [from sorry, from sorry]],
  end

end find_digits_number_l566_566770


namespace stickers_per_page_l566_566515

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l566_566515


namespace octal_to_decimal_5374_l566_566982

-- Define the octal number as a list of digits in octal
def octal_to_decimal (l : List ℕ) : ℕ :=
  l.foldr (λ (d : ℕ) (acc : ℕ) → acc * 8 + d) 0

-- Define the given octal number 5374_8 in terms of its digits
def num_octal := [5, 3, 7, 4]

-- The statement we want to prove: converting the octal 5374_8 to decimal should equal 2812.
theorem octal_to_decimal_5374 :
  octal_to_decimal num_octal = 2812 :=
  sorry

end octal_to_decimal_5374_l566_566982


namespace probability_grunters_win_all_5_games_l566_566475

noncomputable def probability_grunters_win_game : ℚ := 4 / 5

theorem probability_grunters_win_all_5_games :
  (probability_grunters_win_game ^ 5) = 1024 / 3125 := 
  by 
  sorry

end probability_grunters_win_all_5_games_l566_566475


namespace book_recipients_sequences_count_l566_566995

theorem book_recipients_sequences_count :
  let students := 15
  let meetings := 3
  ∀ (choices : ℕ → ℕ), 
  (choices 1 = 15) →
  (choices 2 = 14) →
  (choices 3 = 13) →
  ∏ i in (finset.range 3).map (λ x, x + 1), choices i = 2730 :=
by
  sorry

end book_recipients_sequences_count_l566_566995


namespace greatest_possible_value_of_a_l566_566124

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l566_566124


namespace min_y_value_l566_566426

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 56*y) : 
  ∃ y_min, y_min = 28 - 2*Real.sqrt 212 ∧ ∀ y_val, x_val ∈ ℝ, (x_val^2 + y_val^2 = 16*x_val + 56*y_val) → y_val ≥ y_min :=
by
  sorry

end min_y_value_l566_566426


namespace compute_expression_l566_566800

theorem compute_expression :
  let a := (4 : ℚ) / 7
  let b := (5 : ℚ) / 9
  let c := 1 / a
  a^3 * b^⁻⁴ * c = 295936 / 857500 :=
by
  sorry

end compute_expression_l566_566800


namespace store_makes_profit_l566_566238

def store_transaction (selling_price : ℕ) (profit_percentage1 : ℚ) (loss_percentage2 : ℚ) : ℤ :=
  let purchase_price1 := selling_price / (1 + profit_percentage1)
  let purchase_price2 := selling_price / (1 - loss_percentage2)
  let total_revenue := 2 * selling_price
  let total_cost := purchase_price1 + purchase_price2
  total_revenue - total_cost

theorem store_makes_profit (selling_price : ℕ) (profit_percentage1 : ℚ) (loss_percentage2 : ℚ) :
  selling_price = 64 →
  profit_percentage1 = 0.6 →
  loss_percentage2 = 0.2 →
  store_transaction selling_price profit_percentage1 loss_percentage2 = 8 :=
by
  intros
  sorry

end store_makes_profit_l566_566238


namespace sum_of_coordinates_l566_566024

structure Point :=
  (x : ℝ)
  (y : ℝ)

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

noncomputable def line_equation (P Q : Point) : (ℝ × ℝ) :=
  let m := (Q.y - P.y) / (Q.x - P.x) in
  let b := P.y - m * P.x in
  (m, b)

noncomputable def intersection (m1 b1 m2 b2 : ℝ) : Point :=
  let x := (b2 - b1) / (m1 - m2) in
  let y := m1 * x + b1 in
  ⟨x, y⟩

theorem sum_of_coordinates :
  let A := ⟨0, 6⟩
  let B := ⟨0, 0⟩
  let C := ⟨10, 0⟩
  let D := midpoint A B
  let E := midpoint B C
  let (m1, b1) := line_equation A E
  let (m2, b2) := line_equation C D
  let F := intersection m1 b1 m2 b2
  F.x + F.y = 5 :=
by
  sorry

end sum_of_coordinates_l566_566024


namespace problem_f_inequality_l566_566393

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * deriv (f 2) * x + 3

theorem problem_f_inequality 
  (h_diff : differentiable ℝ f) 
  (h_def : ∀ x : ℝ, f x = x^2 + 2 * deriv (f 2) * x + 3) :
  f 0 > f 6 :=
by 
  sorry

end problem_f_inequality_l566_566393


namespace initial_money_amount_l566_566925

theorem initial_money_amount (M : ℝ)
  (h_clothes : M * (1 / 3) = c)
  (h_food : (M - c) * (1 / 5) = f)
  (h_travel : (M - c - f) * (1 / 4) = t)
  (h_remaining : M - c - f - t = 600) : M = 1500 := by
  sorry

end initial_money_amount_l566_566925


namespace problem_solution_l566_566243

def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, y x = a * x^2 + b * x + c

def not_quadratic_func := 
  let yA := fun x => -2 * x^2
  let yB := fun x => 2 * (x - 1)^2 + 1
  let yC := fun x => (x - 3)^2 - x^2
  let yD := fun a => a * (8 - a)
  (¬ is_quadratic yC) ∧ (is_quadratic yA) ∧ (is_quadratic yB) ∧ (is_quadratic yD)

theorem problem_solution : not_quadratic_func := 
sorry

end problem_solution_l566_566243


namespace max_newsstands_l566_566760

theorem max_newsstands (n : ℕ) (h : n = 6) : 
  let num_intersections := (n * (n - 1)) / 2 in
  num_intersections = 15 := 
by 
  sorry

end max_newsstands_l566_566760


namespace find_a_l566_566842

theorem find_a
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f(x) = x / 3 + 2)
  (h₁ : ∀ x, g(x) = 5 - 2 * x)
  (h₂ : f(g(a)) = 4) : 
  a = -1 / 2 := 
sorry

end find_a_l566_566842


namespace angle_between_c_and_a_plus_b_l566_566432

variables {ℝ : Type} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]

-- Define the given vectors and their properties
variables (a b c : euclidean_space ℝ (fin 3))

-- Given conditions:
-- \|a\| = \|b\| > \|a + b\|
-- \|c\| = \|a + b\|
-- We are required to prove the angle between c and (a + b) is 0 degrees.
theorem angle_between_c_and_a_plus_b 
  (ha : ∥a∥ = ∥b∥)
  (hab : ∥a∥ > ∥a + b∥)
  (hc : ∥c∥ = ∥a + b∥) :
  real.angle (inner_product_geometry.angle c (a + b)) = 0 :=
sorry

end angle_between_c_and_a_plus_b_l566_566432


namespace slope_angle_of_inclination_l566_566147

theorem slope_angle_of_inclination : 
  (∃ θ : ℝ, tan θ = √3 ∧ θ = 60 * real.pi / 180) :=
begin
  sorry
end

end slope_angle_of_inclination_l566_566147


namespace order_of_numbers_l566_566281

noncomputable def a : ℝ := (3 / 4) ^ (-1 / 3)
noncomputable def b : ℝ := (3 / 4) ^ (-1 / 4)
noncomputable def c : ℝ := (3 / 2) ^ (-1 / 4)

theorem order_of_numbers : a > b ∧ b > c :=
by
  sorry

end order_of_numbers_l566_566281


namespace max_points_l566_566507

/-- Definition of the initial conditions for the problem. -/
def initialPiles : List (Fin 400) := List.replicate 100 400

/-- Definition of the rules for stone removal and scoring. -/
def score (piles : List (Fin 400)) (i j : Fin 100) : ℤ :=
  abs (piles[i] - piles[j])

/-- Definition of the requirement to remove all stones. -/
def canRemoveAllStones : List (Fin 400) → Prop
  | [] => True
  | h :: t => False

/-- Definition of the total score Petya can achieve. -/
def totalScore : List (Fin 400) → ℤ
  | [] => 0
  | (h :: t) => sorry

/-- Proof statement: Given the initial conditions and the rules of the game,
    Petya can score a maximum of 3920000 points. -/
theorem max_points : ∃ (piles : List (Fin 400)), canRemoveAllStones piles → totalScore piles = 3920000 := by
  sorry

end max_points_l566_566507


namespace OHara_triple_example_l566_566897

theorem OHara_triple_example : (∃ x : ℝ, (sqrt 49 + sqrt 16 = x) ∧ x = 11) :=
by
  use 11
  split
  · norm_num
  · rfl

end OHara_triple_example_l566_566897


namespace negation_sin_x_x_minus_1_l566_566138

theorem negation_sin_x_x_minus_1 :
  ¬ (∀ x : ℝ, sin x ≠ x - 1) ↔ ∃ x : ℝ, sin x = x - 1 :=
by
  sorry

end negation_sin_x_x_minus_1_l566_566138


namespace collision_probability_l566_566873

def time := ℝ  -- representing time in hours

-- Define the time intervals as mentioned in the conditions
noncomputable def trainA_arrival_time : set time := {t | 9 ≤ t ∧ t ≤ 14.5}
noncomputable def trainB_arrival_time : set time := {t | 9.5 ≤ t ∧ t ≤ 12.5}
noncomputable def intersection_clear_time := 45 / 60 -- in hours

-- Define the event space
def is_collision (a b : time) : Prop :=
  abs (a - b) < intersection_clear_time

-- Define the probability function
noncomputable def uniform_prob (s : set time) : ℝ := sorry

-- Conditions:
-- 1. Train A arrives between 9:00 AM and 2:30 PM.
-- 2. Train B arrives between 9:30 AM and 12:30 PM.
-- 3. Each train takes 45 minutes to clear the intersection.
def prob_collision : ℝ :=
  (uniform_prob trainA_arrival_time) *
  (∫ a in trainA_arrival_time, ∫ b in trainB_arrival_time, indicator is_collision a b)

theorem collision_probability : prob_collision = 13 / 48 :=
  sorry

end collision_probability_l566_566873


namespace number_of_six_digit_numbers_l566_566752

theorem number_of_six_digit_numbers :
  let valid_digits : List ℕ := [1, 3, 5, 7, 9]
  let is_valid (digits : List ℕ) : Prop := 
    ∀ (i : ℕ), (i < 5) → (digits.nth i = some 1 → digits.nth (i + 1) ≠ some 1)
  by sorry

  ∑ (digs : List ℕ) in Finset.filter (λ digs, ∀ d ∈ digs, d ∈ valid_digits ∧ is_valid digs) (Finset.univ : Finset (List ℕ)),
  1 = 13056 :=
sorry

end number_of_six_digit_numbers_l566_566752


namespace not_common_period_f_and_g_l566_566769

-- Definitions used in the proof
def is_periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x : ℝ, f (x + T) = f x
def f : ℝ → ℝ := λ x, if ∃ q1 q2 : ℚ, x = q1 + q2 * real.sqrt 2 then 1/7 else 0
def g : ℝ → ℝ := λ x, 3 * (if ∃ q1 q2 : ℚ, x = q1 * real.sqrt 3 + q2 then 1/7 else 0)

-- Question statement: proving that f and g do not necessarily have a common period
theorem not_common_period_f_and_g : 
  is_periodic f ∧ is_periodic g ∧ is_periodic (λ x, f x + g x) → ¬ ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ g (x + T) = g x :=
sorry

end not_common_period_f_and_g_l566_566769


namespace construct_parallelogram_l566_566335

-- Definitions based on problem conditions
variables {k : Type*} [Field k]
variables (l1 l2 l3 l4 : AffineSubspace k (k × k)) -- Four pairwise non-parallel lines
variable (O : k × k) -- Point O not lying on any of these lines

-- Define non-parallelism and point not lying on any of the lines
def non_parallel_lines := l1 ≠ l2 ∧ l1 ≠ l3 ∧ l1 ≠ l4 ∧ l2 ≠ l3 ∧ l2 ≠ l4 ∧ l3 ≠ l4
def point_not_on_lines := (O ∉ l1) ∧ (O ∉ l2) ∧ (O ∉ l3) ∧ (O ∉ l4)

-- Statement of the proof problem
theorem construct_parallelogram (h1 : non_parallel_lines l1 l2 l3 l4) 
                                (h2 : point_not_on_lines l1 l2 l3 l4 O) :
  ∃ (A B C D : k × k), 
    A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3 ∧ D ∈ l4 ∧ 
    (let M := (A + C + B + D) / 4 in 
     M = O ∧ 
     A - B = D - C ∧ A - D = B - C) := sorry

end construct_parallelogram_l566_566335


namespace problem_solution_l566_566084

open Nat

def is_valid_condition1 (x y z : Nat) : Prop :=
  100 <= 100*x + 10*y + z ∧ 100*x + 10*y + z < 1000 ∧ (100*x + 10*y + z) % 5 = 0 ∧ x > y ∧ y > z

def is_valid_condition2 (x y z : Nat) : Prop :=
  100 <= 100*x + 10*y + z ∧ 100*x + 10*y + z < 1000 ∧ (100*x + 10*y + z) % 5 = 0 ∧ x < y ∧ y < z

def A : Nat :=
  (∑ k in Finset.range 9, Finset.range k).card

def B : Nat :=
  (∑ k in Finset.range 4, Finset.range k).card

theorem problem_solution : ((A > B) ∧ ¬(B > 10) ∧ ¬(A + B > 100) ∧ ¬(A < 10) ∧ ¬(¬(A > B) ∧ ¬(B > 10) ∧ ¬(A + B > 100) ∧ ¬(A < 10))) := by
    have h_correctA : A = 42 := sorry
    have h_correctB : B = 6 := sorry
    sorry

end problem_solution_l566_566084


namespace solution_to_eq_l566_566864

def eq1 (x y z t : ℕ) : Prop := x * y - x * z + y * t = 182
def cond_numbers (n : ℕ) : Prop := n = 12 ∨ n = 14 ∨ n = 37 ∨ n = 65

theorem solution_to_eq 
  (x y z t : ℕ) 
  (hx : cond_numbers x) 
  (hy : cond_numbers y) 
  (hz : cond_numbers z) 
  (ht : cond_numbers t) 
  (h : eq1 x y z t) : 
  (x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14) ∨ 
  (x = 37 ∧ y = 12 ∧ z = 14 ∧ t = 65) := 
sorry

end solution_to_eq_l566_566864


namespace Jisha_walked_62_miles_l566_566045

theorem Jisha_walked_62_miles :
  ∃ (T1 T2 T3 : ℕ) (S1 S2 S3 : ℕ),
    T1 = 18 / 3 ∧
    T2 = T1 - 1 ∧
    S2 = S1 + 1 ∧
    S1 = 3 ∧
    S3 = S2 ∧
    T3 = T1 ∧
    (18 + S2 * T2 + S3 * T3 = 62) :=
  by
  have T1: ℕ := 6
  have T2: ℕ := 5
  have T3: ℕ := 6
  have S1: ℕ := 3
  have S2: ℕ := 4
  have S3: ℕ := S2
  use T1, T2, T3, S1, S2, S3
  split, exact rfl, split, exact rfl, split, exact rfl, split, exact rfl, split, exact rfl, split, exact rfl
  exact rfl

end Jisha_walked_62_miles_l566_566045


namespace solution_concentration_l566_566213

noncomputable def concentration_replaced_solution 
  (initial_concentration : ℝ := 0.4) 
  (new_concentration : ℝ := 0.35) 
  (fraction_replaced : ℝ := 1/3) : ℝ :=
  let C := (new_concentration - (2/3) * initial_concentration) / (1/3) in C

theorem solution_concentration (Q : ℝ) :
  let initial_concentration := 0.4 in
  let new_concentration := 0.35 in
  let fraction_replaced := 1/3 in
  concentration_replaced_solution initial_concentration new_concentration fraction_replaced = 0.25 :=
by
  unfold concentration_replaced_solution
  sorry

end solution_concentration_l566_566213


namespace important_emails_l566_566456

theorem important_emails (total_emails : ℕ) (spam_fraction : ℚ) (promo_fraction : ℚ) (important_emails : ℕ)
  (h_total : total_emails = 400)
  (h_spam_fraction : spam_fraction = 1/4)
  (h_promo_fraction : promo_fraction = 2/5)
  (h_important : important_emails = 180) :
by
  simp [h_total, h_spam_fraction, h_promo_fraction, h_important]
  sorry

end important_emails_l566_566456


namespace sufficient_but_not_necessary_condition_l566_566151

theorem sufficient_but_not_necessary_condition : ∀ (y : ℝ), (y = 2 → y^2 = 4) ∧ (y^2 = 4 → (y = 2 ∨ y = -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l566_566151


namespace infinitely_many_n_l566_566095

theorem infinitely_many_n (p : ℕ) (hp : p.Prime) (hp2 : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ n * 2^n + 1 :=
sorry

end infinitely_many_n_l566_566095


namespace vectors_cross_zero_l566_566386

-- Define vectors u, v, and w in three-dimensional space
variables (u v w : ℝ × ℝ × ℝ)

-- Define the cross product
def cross_prod (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, 
   a.2.0 * b.2.2 - a.2.2 * b.2.0,
   a.2.1 * b.2.0 - a.2.0 * b.2.1)

-- Given conditions
axiom u_cross_v : cross_prod u v = (3, -1, 2)
axiom v_cross_w : cross_prod v w = (-4, 2, -3)

-- Theorem statement
theorem vectors_cross_zero : cross_prod (u + v + w) (u + v + w) = (0, 0, 0) :=
sorry

end vectors_cross_zero_l566_566386


namespace num_divisors_of_2_exp_prod_primes_plus_one_l566_566443

theorem num_divisors_of_2_exp_prod_primes_plus_one (n : ℕ) (p : Fin n → ℕ) (h : ∀ i, Prime (p i) ∧ Odd (p i)) :
  ∃ k, k ≥ 2^(2^(n-1)) ∧ ∃ d : List ℕ, (∀ x ∈ d, x ∣ 2^(List.prod (List.ofFn p))) + 1 ∧ List.length d = k := by
  sorry

end num_divisors_of_2_exp_prod_primes_plus_one_l566_566443


namespace polynomial_fixed_points_upper_bound_l566_566069

noncomputable def P (x : ℝ) : ℝ := sorry  -- Assume P is a polynomial function with specified degree and integer coefficients
noncomputable def Q (P : ℝ → ℝ) (k : ℕ) (x : ℝ) : ℝ :=
  (nat.iterate P k) x -- Q_k(x) is defined as P iterated k times

theorem polynomial_fixed_points_upper_bound (n : ℕ) (hP_deg : P.degree = n) (hP_int : ∀ (c : ℝ), c ∈ P.coefficients → c ∈ ℤ) (k : ℕ) :
  ∃ t_set : set ℤ, t_set.card ≤ n ∧ ∀ t ∈ t_set, Q P k t = t :=
begin
  sorry -- the proof goes here
end

end polynomial_fixed_points_upper_bound_l566_566069


namespace measure_of_angle_Y_is_45_l566_566175

-- Define the conditions
def triangle (A B C : Type) := (A B C : Type)

variables (X Y Z : Type)
variable (angle_X : ℝ)
variable (isosceles_right_triangle_XYZ : Prop)

-- Conditions
axiom X_is_45_degrees : angle_X = 45
axiom XYZ_is_isosceles_right_triangle : isosceles_right_triangle_XYZ

-- The problem statement
theorem measure_of_angle_Y_is_45 (h1 : XYZ_is_isosceles_right_triangle) (h2 : X_is_45_degrees) : 
  ∃ angle_Y : ℝ, angle_Y = 45 :=
sorry

end measure_of_angle_Y_is_45_l566_566175


namespace sum_of_coefficients_proof_l566_566291

-- Problem statement: Define the expressions and prove the sum of the coefficients
def expr1 (c : ℝ) : ℝ := -(3 - c) * (c + 2 * (3 - c))
def expanded_form (c : ℝ) : ℝ := -c^2 + 9 * c - 18
def sum_of_coefficients (p : ℝ) := -1 + 9 - 18

theorem sum_of_coefficients_proof (c : ℝ) : sum_of_coefficients (expr1 c) = -10 := by
  sorry

end sum_of_coefficients_proof_l566_566291


namespace triangle_XYZ_area_l566_566900

theorem triangle_XYZ_area (X Y Z W : Point)
  (h_right_angle : ∠Y = 90°)
  (hW : W is_foot_of_altitude_from Y_on (XZ))
  (hXW : dist X W = 5)
  (hWZ : dist W Z = 3) :
  area_of_triangle X Y Z = 4 * real.sqrt 15 :=
sorry

end triangle_XYZ_area_l566_566900


namespace focus_of_parabola_y_eq_2x2_l566_566111

def parabola_focus_coordinates : Prop :=
    let y : ℝ → ℝ := λ x, 2 * x^2
    ∃ (x_coord y_coord : ℝ), 
        (x_coord = 0 ∧ y_coord = 1 / 8) ∧ 
        ∃ (p : ℝ), (2 * p = 1 / 2) ∧ 
        (p = y_coord)

theorem focus_of_parabola_y_eq_2x2 : parabola_focus_coordinates :=
    by
    sorry

end focus_of_parabola_y_eq_2x2_l566_566111


namespace PQ_length_l566_566053

noncomputable def gamma1 : Circle := { center := O1, radius := 3 }
noncomputable def gamma2 : Circle := { center := O2, radius := 4 }
noncomputable def gamma3 : Circle := { center := O3, radius := 9 }

def areTangentExternally (c1 c2 : Circle) : Prop := 
  dist c1.center c2.center = c1.radius + c2.radius

def areTangentInternally (c1 c2 : Circle) : Prop := 
  dist c1.center c2.center = abs (c1.radius - c2.radius)

axiom O1O2_distance : dist O1 O2 = 7
axiom O1O3_distance : dist O1 O3 = 12
axiom O2O3_distance : dist O2 O3 = 13

theorem PQ_length : length_PQ gamma1 gamma2 gamma3 = 72 * sqrt 3 / 7 :=
sorry

end PQ_length_l566_566053


namespace find_numbers_l566_566558

theorem find_numbers :
  ∃ (nums : Fin 2017 → ℤ),
    (∀ i, nums i = 1 ∨ nums i = -1) ∧
    (∀ s : Finset (Fin 2017), s.card = 7 → (∑ i in s, nums i ^ 2) = 7) ∧
    (∀ s : Finset (Fin 2017), s.card = 11 → 0 < (∑ i in s, nums i)) ∧
    (∑ i, nums i) % 9 = 0 ∧
    (∑ i, if nums i = -1 then 1 else 0) = 5 :=
by
  sorry

end find_numbers_l566_566558


namespace percent_notebooks_staplers_clips_l566_566845

def percent_not_special (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) : ℝ :=
  100 - (n + s + c)

theorem percent_notebooks_staplers_clips (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) :
  percent_not_special n s c h_n h_s h_c = 25 :=
by
  unfold percent_not_special
  rw [h_n, h_s, h_c]
  norm_num

end percent_notebooks_staplers_clips_l566_566845


namespace final_match_odd_numbers_l566_566817

theorem final_match_odd_numbers (n : ℕ) (hn : n % 2 = 1) : 
  ∃ g b, g ∈ {1, 2, ..., n} ∧ b ∈ {1, 2, ..., n} ∧ (∃ m, last_game_pair n m = (g, b)) ∧ g % 2 = 1 ∧ b % 2 = 1 :=
sorry

end final_match_odd_numbers_l566_566817


namespace tangent_line_at_e_l566_566126

noncomputable def f : ℝ → ℝ := λ x => x * Real.log x

theorem tangent_line_at_e (h_e : Real.exp 1 = e) :
  ∀ x y : ℝ, (y = 2 * x - e) ↔ (y - e = 2 * (x - e) ∧ x = e ∧ f(e) = e) :=
sorry

end tangent_line_at_e_l566_566126


namespace max_value_f_l566_566132

-- Define the function f(x) with the given conditions
def f (x : ℝ) : ℝ := Math.sin (2 * x - (Real.pi / 3))

-- Define the interval [0, π/4]
def interval := Set.Icc 0 (Real.pi / 4)

-- Prove that the maximum value of f(x) in the interval is 1/2
theorem max_value_f :
  (Set.maxImage f interval) = 1 / 2 := 
sorry

end max_value_f_l566_566132


namespace greatest_possible_remainder_l566_566378

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 9 ∧ x % 9 = r ∧ r = 8 :=
by
  use 8
  sorry -- Proof to be filled in

end greatest_possible_remainder_l566_566378


namespace polynomial_symmetry_l566_566231

noncomputable def polynomial := ∀ x: ℝ, x^10 + a_9 * x^9 + a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem polynomial_symmetry (f : polynomial) 
    (h1 : f(1) = f(-1)) (h2 : f(2) = f(-2)) (h3 : f(3) = f(-3)) 
    (h4 : f(4) = f(-4)) (h5 : f(5) = f(-5)) : 
    ∀ x : ℝ, f(x) = f(-x) := 
by 
sorry

end polynomial_symmetry_l566_566231


namespace no_triples_l566_566993

theorem no_triples (a b c : ℤ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  ¬ (∃ p : ℤ, Prime p ∧ p = (a-2)*(b-2)*(c-2) + 12 ∧ p ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) ∧ 0 < p < a^2 + b^2 + c^2 + a*b*c - 2017) :=
sorry

end no_triples_l566_566993


namespace average_hamburgers_per_day_l566_566977

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l566_566977


namespace confidence_in_relation_of_age_hidden_item_distribution_l566_566211

noncomputable def chi_square_test (a b c d n : ℕ) : ℚ :=
  ((n:ℚ) * ((a*d - b*c)^2)) / ((a+b)*(c+d)*(a+c)*(b+d))

def k_squared_confidence (a b c d n : ℕ) (k_0 : ℚ) : Prop :=
  chi_square_test a b c d n > k_0

theorem confidence_in_relation_of_age : k_squared_confidence 18 30 22 10 80 6.635 := by
  sorry

open Nat

noncomputable def binomial_distribution (n k : ℕ) (p : ℚ) : ℚ :=
  (choose n k : ℚ) * (p^k) * ((1-p)^(n-k))

def distribution_table (ξ : ℕ → ℚ) : Prop :=
  ξ 0 = 1 / 8 ∧ ξ 1 = 3 / 8 ∧ ξ 2 = 3 / 8 ∧ ξ 3 = 1 / 8

def expectation (ξ : ℕ → ℚ) : ℚ :=
  list.sum (list.map (λ k, k * ξ k) [0,1,2,3])

def expected_value (ξ : ℕ → ℚ) (e : ℚ) : Prop :=
  expectation ξ = e

theorem hidden_item_distribution : 
  ∃ ξ : ℕ → ℚ, distribution_table ξ ∧ expected_value ξ (3 / 2) := by
  sorry

end confidence_in_relation_of_age_hidden_item_distribution_l566_566211


namespace total_distance_correct_l566_566783

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l566_566783


namespace largest_integer_n_neg_l566_566295

theorem largest_integer_n_neg {
  let p := λ n : ℤ, n^2 - 13 * n + 40 < 0,
  5 < n,
  ∀ m, 5 < m -> m < 7 -> (n < m)
} : n = 7 :=
by {
  sorry
}

end largest_integer_n_neg_l566_566295


namespace sum_of_terms_in_arithmetic_sequence_eq_l566_566034

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms_in_arithmetic_sequence_eq :
  arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 36) →
  (a 3 + a 10 = 18) :=
by
  intros h_seq h_sum
  -- Proof placeholder
  sorry

end sum_of_terms_in_arithmetic_sequence_eq_l566_566034


namespace total_driving_time_is_40_l566_566963

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l566_566963


namespace three_digit_numbers_divisible_by_5_l566_566085

theorem three_digit_numbers_divisible_by_5 (A B : ℕ) :
  (A = (∑ k in range 1 (8+1), k + ∑ k in range 1 (3+1), k)) →
  (B = (∑ k in range 1 (3+1), k)) →
  A > B :=
by {
  -- A is calculated by first sum and B by second sum for clarity
  intro hA,
  intro hB,
  rw [hA, hB],
  sorry -- Proof steps are omitted as per the requirements
}

end three_digit_numbers_divisible_by_5_l566_566085


namespace smallest_x_solution_l566_566839

noncomputable def satisfies_equation (x : ℚ) : Prop :=
  7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 35)

theorem smallest_x_solution : ∀ x : ℚ, satisfies_equation x → x ≥ -7/3 :=
begin
  sorry -- Proof to be filled
end

end smallest_x_solution_l566_566839


namespace tangent_line_at_x1_l566_566366

noncomputable def f (x : ℝ) : ℝ := 1 / x + 2 * x

theorem tangent_line_at_x1 :
  let m := diff f 1 in
  let y1 := f 1 in
  let x1 := 1 in
  m = 1 ∧ y1 = 3 → (λ x y, x - y + 2 = 0) (1 : ℝ) (f 1) :=
by -- Define f, calculate derivative and show it equals 1 at x=1, then use point-slope form to show eq of tangent line is x - y + 2 = 0
  sorry

end tangent_line_at_x1_l566_566366


namespace function_domain_l566_566293

def domain_condition (x : ℝ) : Prop :=
  x - 1 ≥ 0

theorem function_domain (x : ℝ) : domain_condition x ↔ (x ∈ set.Ici 1) :=
by {
  sorry
}

end function_domain_l566_566293


namespace gnollish_valid_sentences_count_is_50_l566_566474

def gnollish_words : List String := ["splargh", "glumph", "amr", "blort"]

def is_valid_sentence (sentence : List String) : Prop :=
  match sentence with
  | [_, "splargh", "glumph"] => False
  | ["splargh", "glumph", _] => False
  | [_, "blort", "amr"] => False
  | ["blort", "amr", _] => False
  | _ => True

def count_valid_sentences (n : Nat) : Nat :=
  (List.replicate n gnollish_words).mapM id |>.length

theorem gnollish_valid_sentences_count_is_50 : count_valid_sentences 3 = 50 :=
by 
  sorry

end gnollish_valid_sentences_count_is_50_l566_566474


namespace perfect_square_trinomial_l566_566746

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a : ℝ), (x^2 + mx + 1) = (x + a)^2) ↔ (m = 2 ∨ m = -2) := sorry

end perfect_square_trinomial_l566_566746


namespace ball_height_less_than_10_after_16_bounces_l566_566208

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end ball_height_less_than_10_after_16_bounces_l566_566208


namespace initial_birds_l566_566519

-- Define the initial number of birds (B) and the fact that 13 more birds flew up to the tree
-- Define that the total number of birds after 13 more birds joined is 42
theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
by
  sorry

end initial_birds_l566_566519


namespace solution_set_of_inequality_l566_566701

def f (x : ℝ) : ℝ := log x / log 2 + x ^ 2

theorem solution_set_of_inequality :
  {x : ℝ | let y := log x in f y + f (-y) < 2} = { x : ℝ | (1 / real.exp 1 < x ∧ x < 1) ∨ (1 < x ∧ x < real.exp 1) } :=
by
  sorry

end solution_set_of_inequality_l566_566701


namespace min_sum_labels_diagonal_l566_566610

theorem min_sum_labels_diagonal :
  let labels := λ (i j : ℕ), 1 / (i + j + 1 : ℝ)
  let sum_diagonal := ∑ k in Finset.range 9, labels (k + 1) (k + 1)
  sum_diagonal = (1 / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13 + 1 / 15 + 1 / 17 + 1 / 19) := 
by
  sorry

end min_sum_labels_diagonal_l566_566610


namespace average_hamburgers_per_day_l566_566979

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l566_566979


namespace f_odd_and_increasing_l566_566554

-- Define the function f : ℝ → ℝ
def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
  sorry

end f_odd_and_increasing_l566_566554


namespace common_tangents_to_circles_l566_566139

def circle_eq (x y r : ℝ) : (ℝ → ℝ → Prop) := λ (a b : ℝ), (a - x) ^ 2 + (b - y) ^ 2 = r ^ 2

def center (x y r : ℝ) : (ℝ × ℝ) := (x, y)

def num_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ :=
-- This is a helper definition to be used in the condition below
sorry

theorem common_tangents_to_circles :
  let O1 := circle_eq 1 0 1 in
  let O2 := circle_eq 2 0 2 in
  num_common_tangents O1 O2 = 1 :=
sorry

end common_tangents_to_circles_l566_566139


namespace one_third_way_l566_566185

theorem one_third_way (x₁ x₂ : ℚ) (w₁ w₂ : ℕ) (h₁ : x₁ = 1/4) (h₂ : x₂ = 3/4) (h₃ : w₁ = 2) (h₄ : w₂ = 1) : 
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5 / 12 :=
by 
  rw [h₁, h₂, h₃, h₄]
  -- Simplification of the weighted average to get 5/12
  sorry

end one_third_way_l566_566185


namespace min_spotted_and_blue_eyed_l566_566996

-- Definitions of sets and their cardinalities
variable {U : Type} [Fintype U] {A B : Set U}

def spotted_rabbits := finset.univ.filter (λ x, x ∈ A) 
def blue_eyed_rabbits := finset.univ.filter (λ x, x ∈ B) 

-- Given conditions
axiom total_rabbits (U : Type) [Fintype U] (A B : Set U) : (spotted_rabbits ∪ blue_eyed_rabbits).card ≤ 100
axiom spotted_card : spotted_rabbits.card = 53
axiom blue_eyed_card : blue_eyed_rabbits.card = 73

-- Proof goal
theorem min_spotted_and_blue_eyed (U : Type) [Fintype U] (A B : Set U) :
  (spotted_rabbits ∩ blue_eyed_rabbits).card ≥ 26 :=
sorry

end min_spotted_and_blue_eyed_l566_566996


namespace hamiltonian_cycle_two_colors_l566_566750

-- Define a monochromatic Hamiltonian cycle in a complete graph with edge coloring.
theorem hamiltonian_cycle_two_colors (n : ℕ) (G : Type) [graph G] (color : G → G → ℕ):
    (∀ u v, u ≠ v → (color u v = 1 ∨ color u v = 2)) →
    ∃ (cycle : list G), (∀ (u v : G), (u, v) ∈ cycle.edges → (color u v = 1 ∨ color u v = 2)) ∧
    (cycle.nodup ∧ (∀ (u v : G), (u, v) ∈ cycle.edges → color u v = color u cycle.head ∨ color u v = color (cycle.head) v)) ∧
    (cycle.head = cycle.last ∧ cycle.length = n + 1) :=
begin
  sorry
end

end hamiltonian_cycle_two_colors_l566_566750


namespace base_six_equal_base_b_l566_566108

theorem base_six_equal_base_b (b : ℝ) : (5 * 6^1 + 4 * 6^0) = (1 * b^2 + 2 * b + 1) → b = -1 + real.sqrt 34 :=
by
  sorry

end base_six_equal_base_b_l566_566108


namespace TylerWeightDifference_l566_566904

-- Define the problem conditions
def PeterWeight : ℕ := 65
def SamWeight : ℕ := 105
def TylerWeight := 2 * PeterWeight

-- State the theorem
theorem TylerWeightDifference : (TylerWeight - SamWeight = 25) :=
by
  -- proof goes here
  sorry

end TylerWeightDifference_l566_566904


namespace find_n_l566_566011

-- Define the operation ø
def op (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- Prove that n operating with 2 and then 1 equals 8 implies n = 3
theorem find_n (n : ℕ) (H : op (op n 2) 1 = 8) : n = 3 :=
by
  -- Proof will be provided later
  sorry

end find_n_l566_566011


namespace largest_cyclic_decimal_l566_566255

def digits_on_circle := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def max_cyclic_decimal : ℕ := sorry

theorem largest_cyclic_decimal :
  max_cyclic_decimal = 957913 :=
sorry

end largest_cyclic_decimal_l566_566255


namespace orthocenter_to_vertex_twice_circumcenter_to_side_l566_566828

theorem orthocenter_to_vertex_twice_circumcenter_to_side
  (A B C : Point)
  (O : Point) 
  (O1 : Point) 
  (M : Point) 
  (h1 : IsCircumcenter O A B C)
  (h2 : IsOrthocenter O1 A B C)
  (h3 : IsMidpoint M B C) :
  distance A O1 = 2 * distance O M := 
sorry

end orthocenter_to_vertex_twice_circumcenter_to_side_l566_566828


namespace vector_addition_dot_product_l566_566718

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vector_addition :
  let c := (1, 2) + (3, 1)
  c = (4, 3) := by
  sorry

theorem dot_product :
  let d := (1 * 3 + 2 * 1)
  d = 5 := by
  sorry

end vector_addition_dot_product_l566_566718


namespace tetrahedron_pythagorean_theorem_l566_566412

noncomputable section

variables {a b c : ℝ} {S_ABC S_VAB S_VBC S_VAC : ℝ}

-- Conditions
def is_right_triangle (a b c : ℝ) := c^2 = a^2 + b^2
def is_right_tetrahedron (S_ABC S_VAB S_VBC S_VAC : ℝ) := 
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2

-- Theorem Statement
theorem tetrahedron_pythagorean_theorem (a b c S_ABC S_VAB S_VBC S_VAC : ℝ) 
  (h1 : is_right_triangle a b c)
  (h2 : S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2) :
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2 := 
by sorry

end tetrahedron_pythagorean_theorem_l566_566412


namespace wire_length_correct_l566_566945

noncomputable def wire_length (V : ℝ) (d : ℝ) : ℝ :=
  let r := d / 2 / 1000 -- convert diameter to radius in meters
  let V_m3 := V * (10 ^ (-6)) -- convert volume to cubic meters
  V_m3 / (π * r^2)

theorem wire_length_correct :
  wire_length 22 1 ≈ 28.01 := -- ≈ denotes approximate equality
by 
  unfold wire_length 
  sorry

end wire_length_correct_l566_566945


namespace rotated_and_shifted_line_eq_l566_566466

theorem rotated_and_shifted_line_eq :
  let rotate_line_90 (x y : ℝ) := ( -y, x )
  let shift_right (x y : ℝ) := (x + 1, y)
  ∃ (new_a new_b new_c : ℝ), 
  (∀ (x y : ℝ), (y = 3 * x → x * new_a + y * new_b + new_c = 0)) ∧ 
  (new_a = 1) ∧ (new_b = 3) ∧ (new_c = -1) := by
  sorry

end rotated_and_shifted_line_eq_l566_566466


namespace rachel_bike_lock_code_l566_566460

/-- Rachel's bike lock code problem:
She has a four-digit code using digits from 1 to 4, where each even digit is followed by an odd digit and each odd digit is followed by an even digit. Prove that the number of possible codes she needs to try is 32. -/
theorem rachel_bike_lock_code : 
    (∃ count : ℕ,
        (∀ (code : Fin 10000) (d1 d2 d3 d4 : ℕ), 
            code = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧
            d1 ∈ {1, 2, 3, 4} ∧
            d2 ∈ {1, 2, 3, 4} ∧
            d3 ∈ {1, 2, 3, 4} ∧
            d4 ∈ {1, 2, 3, 4} ∧
            ((d1 % 2 = 0 → d2 % 2 = 1) ∧ (d1 % 2 = 1 → d2 % 2 = 0)) ∧
            ((d2 % 2 = 0 → d3 % 2 = 1) ∧ (d2 % 2 = 1 → d3 % 2 = 0)) ∧
            ((d3 % 2 = 0 → d4 % 2 = 1) ∧ (d3 % 2 = 1 → d4 % 2 = 0))
        ) → count = 32) :=
by {
  -- Proof is to be filled in
  sorry
}

end rachel_bike_lock_code_l566_566460


namespace parallelogram_sum_l566_566612

open Real

def distance (p q : ℝ × ℝ) : ℝ :=
    sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

def parallelogram_perimeter_and_area (A B C : ℝ × ℝ) : ℝ :=
  let D := (3, 3) -- derived using midpoint formula
  let side1 := distance A B
  let side2 := distance A C
  let perimeter := 2 * (side1 + side2)
  let base := side2
  let height := 4 -- assumed in solution
  let area := base * height
  perimeter + area

theorem parallelogram_sum (A B C : ℝ × ℝ) : 
  (A = (2, 3)) →
  (B = (5, 7)) →
  (C = (0, -1)) →
  parallelogram_perimeter_and_area A B C = 10 + 12 * sqrt 5 :=
by 
  intros hA hB hC
  rw [hA, hB, hC]
  unfold parallelogram_perimeter_and_area
  -- proving derived D as (3, 3)
  have hD : (3, 3) = (3, 3) := by rfl
  rw hD
  -- distance between (2, 3) and (5, 7)
  have distAB : distance (2, 3) (5, 7) = 5 := by sorry
  -- distance between (2, 3) and (0, -1)
  have distAC : distance (2, 3) (0, -1) = sqrt 20 := by sorry
  rw [distAB, distAC]
  -- simplify perimeter and area
  have perim := 2 * (5 + sqrt 20)
  have area := sqrt 20 * 4
  -- final simplification to match 10 + 12 * sqrt 5
  calc
    10 + 12 * sqrt 5 = perim + area := by sorry

end parallelogram_sum_l566_566612


namespace equal_segments_YX_ZX_l566_566074

open EuclideanGeometry

variables {K I A O Y Z X : Point}
variables (h1 : midpoint K A O) (h2 : perp Y I (bisector I O K))
variables (h3 : perp Z A (bisector A O K)) (h4 : intersection X K O Y Z)

theorem equal_segments_YX_ZX : dist X Y = dist X Z := 
by
  sorry

end equal_segments_YX_ZX_l566_566074


namespace equilateral_triangle_area_l566_566090

theorem equilateral_triangle_area (A B C M : Type) [point A] [point B] [point C] [point M]
  (AM BM : ℝ) (AM_eq : AM = 2) (BM_eq : BM = 2) (CM : ℝ) (CM_eq : CM = 1) : 
  ∃ x : ℝ, 
    let area := (Math.sqrt(3) * (7 + Math.sqrt(13))) / 8 in
    A ∉ line B C ∧ B ∉ line A C ∧ C ∉ line A B 
    ∧ dist A B = x ∧ dist A C = x ∧ dist B C = x 
    ∧ dist A M = AM ∧ dist B M = BM ∧ dist C M = CM → 
    (√3 / 4) * x^2 = area := 
begin
  sorry
end

end equilateral_triangle_area_l566_566090


namespace greatest_possible_value_of_a_l566_566117

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l566_566117


namespace range_of_x_l566_566627

theorem range_of_x :
  ∀ (a : Fin 15 → ℝ), (∀ i, a i ∈ {0, 1, 3}) → 0 ≤ (∑ i, a i / 4^(i+1)) ∧ (∑ i, a i / 4^(i+1)) < 1 :=
by
  intro a ha
  sorry

end range_of_x_l566_566627


namespace part_one_part_two_l566_566324

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 - (a + 1/a) * x + 1

theorem part_one (x : ℝ) : f x (1/2) ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part_two (x a : ℝ) (h : a > 0) : 
  ((a < 1) → (f x a ≤ 0 ↔ (a ≤ x ∧ x ≤ 1/a))) ∧
  ((a > 1) → (f x a ≤ 0 ↔ (1/a ≤ x ∧ x ≤ a))) ∧
  ((a = 1) → (f x a ≤ 0 ↔ (x = 1))) :=
by
  sorry

end part_one_part_two_l566_566324


namespace complement_intersection_eq_l566_566712

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 2, 5}) (hB : B = {1, 3, 4})

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_eq_l566_566712


namespace raja_monthly_income_l566_566461

theorem raja_monthly_income (X : ℝ) 
  (h1 : 0.1 * X = 5000) : X = 50000 :=
sorry

end raja_monthly_income_l566_566461


namespace travelers_on_liner_l566_566888

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566888


namespace bob_grade_is_35_l566_566774

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l566_566774


namespace find_num_students_B_l566_566163

-- Given conditions as definitions
def num_students_A : ℕ := 24
def avg_weight_A : ℚ := 40
def avg_weight_B : ℚ := 35
def avg_weight_class : ℚ := 38

-- The total weight for sections A and B
def total_weight_A : ℚ := num_students_A * avg_weight_A
def total_weight_B (x: ℕ) : ℚ := x * avg_weight_B

-- The number of students in section B
noncomputable def num_students_B : ℕ := 16

-- The proof problem: Prove that number of students in section B is 16
theorem find_num_students_B (x: ℕ) (h: (total_weight_A + total_weight_B x) / (num_students_A + x) = avg_weight_class) : 
  x = 16 :=
by
  sorry

end find_num_students_B_l566_566163


namespace average_age_of_group_is_correct_l566_566847

-- Conditions
def number_of_sixth_graders : ℕ := 40
def avg_age_sixth_graders : ℝ := 12
def number_of_parents : ℕ := 60
def avg_age_parents : ℝ := 35

-- Total Age Calculations
def total_age_sixth_graders : ℝ := number_of_sixth_graders * avg_age_sixth_graders
def total_age_parents : ℝ := number_of_parents * avg_age_parents

-- Combined Count and Total Age
def total_number_of_individuals : ℕ := number_of_sixth_graders + number_of_parents
def combined_total_age : ℝ := total_age_sixth_graders + total_age_parents

-- Average Age of the Group
def avg_age_group : ℝ := combined_total_age / total_number_of_individuals

-- Theorem to prove the average age
theorem average_age_of_group_is_correct : avg_age_group = 25.8 := by
  sorry

end average_age_of_group_is_correct_l566_566847


namespace modulus_of_z_l566_566444

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3 * complex.I) = 6 + 4 * complex.I) : complex.abs z = 2 :=
by {
  sorry
}

end modulus_of_z_l566_566444


namespace difference_between_numbers_l566_566575

theorem difference_between_numbers : 
  ∃ (a : ℕ), a + 10 * a = 30000 → 9 * a = 24543 := 
by 
  sorry

end difference_between_numbers_l566_566575


namespace determine_n_l566_566628

open Function

noncomputable def coeff_3 (n : ℕ) : ℕ :=
  2^(n-2) * Nat.choose n 2

noncomputable def coeff_4 (n : ℕ) : ℕ :=
  2^(n-3) * Nat.choose n 3

theorem determine_n (n : ℕ) (b3_eq_2b4 : coeff_3 n = 2 * coeff_4 n) : n = 5 :=
  sorry

end determine_n_l566_566628


namespace num_expr_div_by_10_l566_566091

theorem num_expr_div_by_10 : (11^11 + 12^12 + 13^13) % 10 = 0 := by
  sorry

end num_expr_div_by_10_l566_566091


namespace frac_val_of_x_y_l566_566066

theorem frac_val_of_x_y (x y : ℝ) (h: (4 : ℝ) < (2 * x - 3 * y) / (2 * x + 3 * y) ∧ (2 * x - 3 * y) / (2 * x + 3 * y) < 8) (ht: ∃ t : ℤ, x = t * y) : x / y = -2 := 
by
  sorry

end frac_val_of_x_y_l566_566066


namespace ages_problem_l566_566728

theorem ages_problem (x y : ℕ) (h1 : 2 * x = y) (h2 : 2 * x + y = 35) : x = 10 ∧ y = 15 :=
by
  have h3 : y = 3 * x / 2 := by
    calc
      y = 3 * x / 2 : by sorry
  have h4 : 2 * x + 3 * x / 2 = 35 := by
    calc
      2 * x + 3 * x / 2 = 35 : by sorry
  have h5 : 7 * x / 2 = 35 := by
    calc
      7 * x / 2 = 35 : by sorry
  have h6 : 7 * x = 70 := by
    calc
      7 * x = 70 : by sorry
  have h7 : x = 10 := by
      calc
        x = 10 : by sorry
  have h8 : y = 15 := by
    calc
      y = 15 : by sorry
  exact ⟨h7, h8⟩ 

end ages_problem_l566_566728


namespace dozen_pen_cost_l566_566112

-- Definitions based on the conditions
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x
def total_cost (x : ℝ) (y : ℝ) : ℝ := 3 * cost_of_pen x + y * cost_of_pencil x

open Classical
noncomputable def cost_dozen_pens (x : ℝ) : ℝ := 12 * cost_of_pen x

theorem dozen_pen_cost (x y : ℝ) (h : total_cost x y = 150) : cost_dozen_pens x = 60 * x :=
by
  sorry

end dozen_pen_cost_l566_566112


namespace simplify_fraction_l566_566835

theorem simplify_fraction (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  12 * x * y^3 / (9 * x^2 * y^2) = 8 / 9 :=
by {
  rw [h1, h2], -- Substitute x = 3 and y = 2
  -- Simplify the fraction
  simp,
  exact sorry,
}

end simplify_fraction_l566_566835


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566318

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566318


namespace stickers_per_page_l566_566513

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l566_566513


namespace trig_identity_l566_566305

-- Proving the equality (we state the problem here)
theorem trig_identity :
  Real.sin (40 * Real.pi / 180) * (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) = -8 / 3 :=
by
  sorry

end trig_identity_l566_566305


namespace snail_max_distance_300_meters_l566_566986
-- Import required library

-- Define the problem statement
theorem snail_max_distance_300_meters 
  (n : ℕ) (left_turns : ℕ) (right_turns : ℕ) 
  (total_distance : ℕ)
  (h1 : n = 300)
  (h2 : left_turns = 99)
  (h3 : right_turns = 200)
  (h4 : total_distance = n) : 
  ∃ d : ℝ, d = 100 * Real.sqrt 2 :=
by
  sorry

end snail_max_distance_300_meters_l566_566986


namespace B_interval_l566_566609

noncomputable def g : ℕ → ℝ 
| 2 := 2
| (n+1) := log (n + 1 + g n)

def B := g 2050

theorem B_interval : log 2053 < B ∧ B < log 2054 := by
  sorry

end B_interval_l566_566609


namespace minimum_n_l566_566172

theorem minimum_n (n : ℕ) (h1 : ∀ x : ℕ, x ∈ { 1, 2, 3 } → ∀ y : ℕ, y ∈ { 1, 2, ... , n } → (x, y) :ℚ )
  (h2 : ∀ i j : ℕ, i ∈ { 1, 2, 3 } → j ∈ { 1, 2, ... , n } → ¬ ∃ k : ℕ, k ∈ { 1, 2, ..., 3n } ∧ (∃ i' j' : ℕ, (i', j') ∈ { 1, 2, 3 } × { 1, 2, ... , n } ∧ i = i' ∧ j = j'))
  (h3 : ∀ g : fin n → fin 3 → ℕ, ∀ i : fin n, max (tall (g i 0)) (max (tall (g i 1)) (tall (g i 2))) = tall (g i 0)) 
  (h4 : ∀ (i j k : ℕ), i ∈ {1, 2, ... , n} → j = 10 → ∃ (g : fin n → fin 3 → ℕ), tall (g (i - 1 + j) 2) = tall (g i 0))
  : n = 40 := sorry

end minimum_n_l566_566172


namespace money_total_l566_566389

theorem money_total (s j m : ℝ) (h1 : 3 * s = 80) (h2 : j / 2 = 70) (h3 : 2.5 * m = 100) :
  s + j + m = 206.67 :=
sorry

end money_total_l566_566389


namespace symmetric_ratios_equal_squared_ratios_l566_566077

variables {A B C M N : Type*} [Euclidean_Space ℝ] 
  (triangle : triangle A B C)
  {AM AN : Line ℝ} 
  (h_symmetry : symmetric_wrt_angle_bisector triangle AM AN A)
  (h_M_on_BC : M ∈ line_segment B C)
  (h_N_on_BC : N ∈ line_segment B C)

theorem symmetric_ratios_equal_squared_ratios 
  (BM BN : ℝ) (CM CN : ℝ) (b c : ℝ) :
  BM * BN / (CM * CN) = c^2 / b^2 :=
  sorry

end symmetric_ratios_equal_squared_ratios_l566_566077


namespace max_value_at_x_eq_2_l566_566312

def quadratic_function (x : ℝ) : ℝ :=
  - (1 / 4) * x^2 + x - 4

theorem max_value_at_x_eq_2 :
  quadratic_function 2 = -3 :=
sorry

end max_value_at_x_eq_2_l566_566312


namespace parabola_axis_of_symmetry_is_x_eq_1_l566_566656

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end parabola_axis_of_symmetry_is_x_eq_1_l566_566656


namespace center_of_circle_param_eq_l566_566492

theorem center_of_circle_param_eq (θ : ℝ) : 
  (∃ c : ℝ × ℝ, ∀ θ, 
    ∃ (x y : ℝ), 
      (x = 2 + 2 * Real.cos θ) ∧ 
      (y = 2 * Real.sin θ) ∧ 
      (x - c.1)^2 + y^2 = 4) 
  ↔ 
  c = (2, 0) :=
by
  sorry

end center_of_circle_param_eq_l566_566492


namespace length_segment_XY_l566_566494

-- Given the conditions
variables (ABC : Triangle) (ω : Circle)
variable (BC tangent_AB at P : Line)
variable (mid_AB mid_AC common_set_line : Line)
variable (circumcircle_APQ : Circle)

-- Conditions
variable (perimeter_ABC_1 : ABC.perimeter = 1)
variable (e_tangent_BC : ω.tangent BC)
variable (e_tangent_AB : ω.tangent_at_line_extension P AB)
variable (e_tangent_AC : ω.tangent_at_line_extension Q AC)
variable (e_common_set_mid_AB_mid_AC : line_through_midpoint AB AC intersects circumcircle_APQ at X Y)

-- Theorem statement
theorem length_segment_XY : length(XY) = 1 / 2 :=
by
  sorry

end length_segment_XY_l566_566494


namespace exists_inscribed_hex_with_area_at_least_two_thirds_l566_566392

variable (W V : Set (Set Point))
variable (is_convex_polygon : Set Point → Prop)
variable (is_symmetric_hexagon : Set Point → Prop)
variable (unit_area : Set Point → Prop)
variable (inscription : Set Point → Set Point → Prop)

theorem exists_inscribed_hex_with_area_at_least_two_thirds :
  is_convex_polygon W → unit_area W →
  (∃ V, is_symmetric_hexagon V ∧ inscription V W ∧ area V ≥ 2 / 3) :=
by
  -- Proof will go here
  sorry

end exists_inscribed_hex_with_area_at_least_two_thirds_l566_566392


namespace part1_part2_l566_566550

-- Define the sets and necessary assumptions
def A : set ℕ := { n | ∃ k : ℕ, k > 0 ∧ n = 2^k }
def N_pos : set ℕ := { n | n > 0 }
def A_complement : set ℕ := N_pos \ A

-- Statement for part (1)
theorem part1 (a : ℕ) (b : ℕ) (h : a ∈ A) (hb : b ∈ N_pos) (h_cond : b < 2 * a - 1) : ¬(2 * a ∣ b * (b + 1)) :=
sorry

-- Statement for part (2)
theorem part2 (a : ℕ) (h : a ∈ A_complement) (h_ne : a ≠ 1) : 
  ∃ b : ℕ, b ∈ N_pos ∧ b < 2 * a - 1 ∧ 2 * a ∣ b * (b + 1) :=
sorry

end part1_part2_l566_566550


namespace remainder_of_least_N_divided_by_1000_l566_566651

noncomputable def f (n : ℕ) : ℕ :=
  (n.digits 4).sum

noncomputable def g (n : ℕ) : ℕ :=
  (f n).digits 8 .sum

def N : ℕ :=
  Nat.find (λ n, (g n).digits 16.any (λ d, d ≥ 10))

theorem remainder_of_least_N_divided_by_1000 : N % 1000 = 151 := by
  sorry

end remainder_of_least_N_divided_by_1000_l566_566651


namespace boat_speed_l566_566562

theorem boat_speed 
  (lighthouse_distance : ℝ)
  (time : ℝ)
  (direction1 direction2 : ℝ)
  (lighthouse_distance = 80)
  (time = 1)
  (direction1 = southwest)
  (direction2 = southsouthwest) : 
  boat_speed lighthouse_distance time direction1 direction2 = 40 * (2 + Real.sqrt 2) := 
sorry

end boat_speed_l566_566562


namespace length_of_AB_l566_566178

-- Definitions based on the conditions provided.
def side_length_smaller_square : ℝ := 1
def side_length_larger_square : ℝ := 7

-- Define the points of interest demonstrating the translated problem.
def AC : ℝ := side_length_smaller_square + side_length_larger_square
def BC : ℝ := side_length_larger_square - side_length_smaller_square

-- The target declaration we need to prove.
theorem length_of_AB : real.sqrt (AC^2 + BC^2) = 10 :=
by
  sorry

end length_of_AB_l566_566178


namespace not_possible_consecutive_results_l566_566819

theorem not_possible_consecutive_results 
  (dot_counts : ℕ → ℕ)
  (h_identical_conditions : ∀ (i : ℕ), dot_counts i = 1 ∨ dot_counts i = 2 ∨ dot_counts i = 3) 
  (h_correct_dot_distribution : ∀ (i j : ℕ), (i ≠ j → dot_counts i ≠ dot_counts j))
  : ¬ (∃ (consecutive : ℕ → ℕ), 
        (∀ (k : ℕ), k < 6 → consecutive k = dot_counts (4 * k) + dot_counts (4 * k + 1) 
                         + dot_counts (4 * k + 2) + dot_counts (4 * k + 3))
        ∧ (∀ (k : ℕ), k < 5 → consecutive (k + 1) = consecutive k + 1)) := sorry

end not_possible_consecutive_results_l566_566819


namespace min_area_OAB_min_area_MAB_l566_566067

noncomputable def parabola_min_area_OAB (p : ℝ) (hp : p > 0) : ℝ :=
  if h : p > 0 then p^2 / 2 else 0

theorem min_area_OAB (p : ℝ) (hp : p > 0) :
  let parabola_min_area_OAB (p : ℝ) (hp : p > 0) : ℝ :=
    if h : p > 0 then p^2 / 2 else 0
  in parabola_min_area_OAB p hp = p^2 / 2 := by sorry

noncomputable def parabola_min_area_MAB (p : ℝ) (hp : p > 0) : ℝ :=
  if h : p > 0 then p^2 else 0

theorem min_area_MAB (p : ℝ) (hp : p > 0) :
  let parabola_min_area_MAB (p : ℝ) (hp : p > 0) : ℝ :=
    if h : p > 0 then p^2 else 0
  in parabola_min_area_MAB p hp = p^2 := by sorry

end min_area_OAB_min_area_MAB_l566_566067


namespace sin_60_deg_l566_566605

theorem sin_60_deg : 2 * Real.sin (Real.pi / 3) = Real.sqrt 3 :=
by
  have h : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
    sorry -- This comes from a standard trigonometric identity which can be assumed to be true.
  calc
    2 * Real.sin (Real.pi / 3) = 2 * (Real.sqrt 3 / 2) : by rw [h]
    ... = (2 * Real.sqrt 3) / 2                : by ring
    ... = Real.sqrt 3                          : by ring
  sorry

end sin_60_deg_l566_566605


namespace integral_sin_x_squared_series_integral_sqrt_x_e_x_series_integral_sqrt_one_minus_x_cubed_series_l566_566600

open Real

-- Problem 1
theorem integral_sin_x_squared_series:
  (∫ x in 0..a, sin(x^2)) = (a^3 / 3) - (a^7 / 42) + (a^11 / 3960) - (a^15 / 113400) + C :=
sorry

-- Problem 2
theorem integral_sqrt_x_e_x_series:
  (∫ x in 0..a, sqrt x * exp x) = (2 / 3) * a^(3/2) + (2 / 5) * a^(5/2) + (2 * a^(7/2)) / 14 + (2 * a^(9/2)) / (factorial 3 * 9) + C :=
sorry

-- Problem 3
theorem integral_sqrt_one_minus_x_cubed_series:
  (∫ x in 0..a, sqrt(1 - x^3)) = a - (a^4 / 8) - (a^7 / 56) - (a^10 / 480) + C :=
sorry

end integral_sin_x_squared_series_integral_sqrt_x_e_x_series_integral_sqrt_one_minus_x_cubed_series_l566_566600


namespace average_age_of_group_is_correct_l566_566848

-- Conditions
def number_of_sixth_graders : ℕ := 40
def avg_age_sixth_graders : ℝ := 12
def number_of_parents : ℕ := 60
def avg_age_parents : ℝ := 35

-- Total Age Calculations
def total_age_sixth_graders : ℝ := number_of_sixth_graders * avg_age_sixth_graders
def total_age_parents : ℝ := number_of_parents * avg_age_parents

-- Combined Count and Total Age
def total_number_of_individuals : ℕ := number_of_sixth_graders + number_of_parents
def combined_total_age : ℝ := total_age_sixth_graders + total_age_parents

-- Average Age of the Group
def avg_age_group : ℝ := combined_total_age / total_number_of_individuals

-- Theorem to prove the average age
theorem average_age_of_group_is_correct : avg_age_group = 25.8 := by
  sorry

end average_age_of_group_is_correct_l566_566848


namespace sequence_a_n_l566_566763

theorem sequence_a_n (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h3 : ∀ n : ℕ, 1 < n → a (n + 1) = a n + a (n - 1)) :
  a 7 = 1 :=
sorry

end sequence_a_n_l566_566763


namespace stratified_sampling_l566_566585

theorem stratified_sampling (teachers male_students female_students total_pop sample_female_students proportion_total n : ℕ)
    (h_teachers : teachers = 200)
    (h_male_students : male_students = 1200)
    (h_female_students : female_students = 1000)
    (h_total_pop : total_pop = teachers + male_students + female_students)
    (h_sample_female_students : sample_female_students = 80)
    (h_proportion_total : proportion_total = female_students / total_pop)
    (h_proportion_equation : sample_female_students = proportion_total * n) :
  n = 192 :=
by
  sorry

end stratified_sampling_l566_566585


namespace average_age_l566_566849

theorem average_age (n_students : ℕ) (n_parents : ℕ) (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  n_students = 40 →
  n_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 35 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents) = 25.8 :=
by 
  intros h_students h_parents h_avg_students h_avg_parents
  rw [h_students, h_parents, h_avg_students, h_avg_parents]
  norm_num
  sorry

end average_age_l566_566849


namespace fibonacci_sum_inequality_l566_566374

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => (fib (n+1)) + (fib n)

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem fibonacci_sum_inequality (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range (n + 1), comb n k * fib k) < (2*n + 2)^n / Nat.factorial n :=
by
  sorry

end fibonacci_sum_inequality_l566_566374


namespace maximum_k_for_books_l566_566257

theorem maximum_k_for_books (n_books : ℕ) (k : ℕ) (n_books = 1300) :
  (∃ k : ℕ, ∀ rearrangement : finset (finset ℕ), 
    (∀ s ∈ rearrangement, s.card ≤ n_books / k) → 
    ∃ shelf : finset ℕ, ∃ t ∈ rearrangement, t.card ≥ 5 ∧ shelf.card = t.card) ↔ k = 18 :=
by sorry

end maximum_k_for_books_l566_566257


namespace find_max_area_ABM_l566_566039

noncomputable theory

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 * (1 + 3 * (Real.sin θ)^2) = 16

def point_P_on_C1 (ρ θ : ℝ) : Prop :=
  polar_curve (2 * ρ) θ

def curve_C2 (x y : ℝ) : Prop :=
  (x / 2)^2 + y^2 = 1

def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = t

def standard_line_l (x y : ℝ) : Prop :=
  x - y - 1 = 0

def point_A : (ℝ × ℝ) := (0, -1)
def point_B : (ℝ × ℝ) := (1, 0)

def dist_AB : ℝ := Real.sqrt 2

def point_M_on_C2 (x y : ℝ) : Prop :=
  curve_C2 x y

def dist_M_to_l (x y : ℝ) : ℝ :=
  (abs (2 * Real.cos x - Real.sin y - 1)) / Real.sqrt 2

def max_area_ABM : ℝ :=
  1 / 2 * dist_AB * ((Real.sqrt 5 + 1) / Real.sqrt 2)

theorem find_max_area_ABM : max_area_ABM = (Real.sqrt 5 + 1) / 2 := 
  by sorry

end find_max_area_ABM_l566_566039


namespace computer_repair_cost_l566_566607

theorem computer_repair_cost :
  ∃ (C : ℝ), (5 * 11) + (2 * 15) + (2 * C) = 121 ∧ C = 18 :=
by
  use 18
  split
  · calc
      5 * 11 + 2 * 15 + 2 * 18 = 55 + 30 + 36 : by rw [mul_assoc, mul_assoc, mul_assoc]
      ... = 55 + 30 + 36 : sorry -- provide steps matching tuples in solution
      ... = 121 : by norm_num
  · exact rfl

end computer_repair_cost_l566_566607


namespace decimal_places_of_fraction_l566_566618

/-
  Statement: Evaluate the number of decimal places in the decimal representation
  of the fraction $\frac{123456789}{2^{26} \times 5^{4}}$ to show that it equals 26.
-/

theorem decimal_places_of_fraction : 
  ∀ k : ℕ, k = 26 ↔ (∃ n: ℝ, n = 123456789 / (2^26 * 5^4) ∧ (decimal_places n) = k) :=
begin
  sorry
end

-- Definition of the function to count decimal places
def decimal_places (x: ℝ) : ℕ := sorry

end decimal_places_of_fraction_l566_566618


namespace least_number_with_remainder_l566_566534

theorem least_number_with_remainder (N : ℕ) : (∃ k : ℕ, N = 12 * k + 4) → N = 256 :=
by
  intro h
  sorry

end least_number_with_remainder_l566_566534


namespace f_is_odd_l566_566482

-- Define the odd function and its properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- The given function when x > 0
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^3 + x + 1
  else x^3 + x - 1

-- Prove the function is odd
theorem f_is_odd : is_odd_function f :=
by
  intros x
  -- Decompose the cases when x > 0 and x <= 0
  cases (lt_or_le x 0) with h_pos h_non_pos
  · rw [if_pos h_pos]       -- the case x > 0
    rw [if_neg (neg_nonneg_of_nonpos h_pos.le)]
    -- Apply the odd function property
    sorry
  · rw [if_neg h_non_pos] -- the case x ≤ 0
    rw [if_pos (neg_pos_of_neg h_non_pos)]
    -- Apply the odd function property
    sorry

end f_is_odd_l566_566482


namespace bob_grade_is_35_l566_566772

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l566_566772


namespace decreasing_function_solution_set_l566_566435

theorem decreasing_function_solution_set {f : ℝ → ℝ} (h : ∀ x y, x < y → f y < f x) :
  {x : ℝ | f 2 < f (2*x + 1)} = {x : ℝ | x < 1/2} :=
by
  sorry

end decreasing_function_solution_set_l566_566435


namespace problem_conditions_l566_566348

noncomputable def minimum_xy : ℝ := 9
noncomputable def minimum_x_plus_y : ℝ := 6

theorem problem_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : log x + log y = log (x + y + 3)) :
    (min_xy : ℝ := x * y) ≥ minimum_xy ∧ 
    (min_x_plus_y : ℝ := x + y) ≥ minimum_x_plus_y :=
by
  sorry

end problem_conditions_l566_566348


namespace rain_thunder_both_days_l566_566867

-- Define the given probabilities
def probRainMonday := 0.40
def probRainTuesday := 0.30
def probThunderGivenRain := 0.10

-- Define the independence
def independentEvents (pA pB : ℝ) : Prop := pA * pB

-- Define the problem statement
theorem rain_thunder_both_days : 
  independentEvents probRainMonday probRainTuesday 
  → independentEvents probThunderGivenRain probThunderGivenRain
  → (probRainMonday * probRainTuesday * probThunderGivenRain * probThunderGivenRain) * 100 = 0.12 :=
by
  sorry

end rain_thunder_both_days_l566_566867


namespace point_C_coordinates_x_plus_y_value_l566_566038

/-- Lean statement for part 1 -/
theorem point_C_coordinates :
  let B := (-(2 : ℝ), 1 : ℝ)
  (∣B.1∣^2 + B.2^2 = 5)
  rotate(B, π / 2) = (1, 2)
:= sorry

/-- Lean statement for part 2 -/
theorem x_plus_y_value :
  let cos_theta := (√10 / 10)
  let sin_theta := (3 * √10 / 10)
  let OP := (2 * √10)
  let P := (2, 6)
  let OA := (1, 0)
  let OB := (-1, 2)
  (angle ☓OP, OA☿ = angle (π / 4))
  (angle ☓OP, OB☿ = angle θ)
  (OP = x * ☓OA☿ + y * ☓OB☿)
  (x := 5)
  (y := 3)
  x + y = 8
:= sorry

end point_C_coordinates_x_plus_y_value_l566_566038


namespace percent_of_juniors_l566_566170

theorem percent_of_juniors (total_students : ℕ) 
  (students_participating_in_sports : ℕ) 
  (percent_participating_in_sports : ℝ)
  (juniors_in_sports : ℝ) : 
  total_students = 500 ∧ 
  percent_participating_in_sports = 0.70 ∧ 
  juniors_in_sports = 140 → 
  (students_participating_in_sports = (juniors_in_sports / percent_participating_in_sports).toNat → 
  ((students_participating_in_sports.toNat : ℝ) / total_students.toNat * 100 = 40)
  ) :=
begin
  sorry
end

end percent_of_juniors_l566_566170


namespace natural_solutions_count_l566_566381

theorem natural_solutions_count : (finset.card 
  (finset.filter (λ (p : ℕ × ℕ), p.1 < 76 ∧ p.2 < 71 ∧ (71 * p.1 + 76 * p.2 < 76 * 71))
    ((finset.range 76).product (finset.range 71)))) = 2625 :=
  sorry

end natural_solutions_count_l566_566381


namespace probability_adjacent_abby_brigdet_diff_row_charlie_l566_566833

/-- Seven students (A, B, C, D, E, F, G) are randomly seated in two rows of four seats each.
  Define the following:
  - Abby (A) and Bridget (B) must be adjacent in the same row or column.
  - Charlie (C) must be seated in a different row from Abby (A).
  Prove the probability of these conditions is 5/21. -/
theorem probability_adjacent_abby_brigdet_diff_row_charlie :
  let students := ["A", "B", "C", "D", "E", "F", "G"]
  let rows := 2
  let cols := 4
  let total_seats := rows * cols
  let total_arrangements := total_seats * factorial (total_seats - 1)
  let adjacent_row_pairs := rows * (cols - 1)
  let adjacent_column_pairs := cols * (rows - 1)
  let total_adjacent_pairs := adjacent_row_pairs + adjacent_column_pairs
  let favorable_AB_arrangements := total_adjacent_pairs * 2 * 4
  let remaining_arrangements := factorial (total_seats - 3)
  let favorable_outcomes := favorable_AB_arrangements * remaining_arrangements
  (favorable_outcomes / total_arrangements : ℚ) = 5 / 21 :=
sorry

end probability_adjacent_abby_brigdet_diff_row_charlie_l566_566833


namespace y_intercepts_count_l566_566721

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l566_566721


namespace collision_probability_l566_566872

def time := ℝ  -- representing time in hours

-- Define the time intervals as mentioned in the conditions
noncomputable def trainA_arrival_time : set time := {t | 9 ≤ t ∧ t ≤ 14.5}
noncomputable def trainB_arrival_time : set time := {t | 9.5 ≤ t ∧ t ≤ 12.5}
noncomputable def intersection_clear_time := 45 / 60 -- in hours

-- Define the event space
def is_collision (a b : time) : Prop :=
  abs (a - b) < intersection_clear_time

-- Define the probability function
noncomputable def uniform_prob (s : set time) : ℝ := sorry

-- Conditions:
-- 1. Train A arrives between 9:00 AM and 2:30 PM.
-- 2. Train B arrives between 9:30 AM and 12:30 PM.
-- 3. Each train takes 45 minutes to clear the intersection.
def prob_collision : ℝ :=
  (uniform_prob trainA_arrival_time) *
  (∫ a in trainA_arrival_time, ∫ b in trainB_arrival_time, indicator is_collision a b)

theorem collision_probability : prob_collision = 13 / 48 :=
  sorry

end collision_probability_l566_566872


namespace monotonic_intervals_specific_case_l566_566368

def f (x a : ℝ) : ℝ := (real.log x) + (2 * a / (x + 1))

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 2 → ∀ x > 0, differentiable_at ℝ (λ x, f x a) x → monotone_on (λ x, f x a) (set.Ioi 0)) ∧
  (a > 2 → (∀ x > 0, differentiable_at ℝ (λ x, f x a) x → 
    ((monotone_on (λ x, f x a) (set.Ioo 0 (a - 1 - real.sqrt (a^2 - 2*a))))
     ∧ (monotone_on (λ x, f x a) (set.Ioi (a - 1 + real.sqrt (a^2 - 2*a))))
     ∧ (antitone_on (λ x, f x a) (set.Ioo (a - 1 - real.sqrt (a^2 - 2*a)) (a - 1 + real.sqrt (a^2 - 2*a))))))) :=
by
  sorry

theorem specific_case (x : ℝ) (h_pos : x > 0) : 
  f x 1 ≤ (x + 1) / 2 :=
by
  sorry

end monotonic_intervals_specific_case_l566_566368


namespace max_value_of_xing_l566_566036

-- Define the conditions
def consecutive_nonzero_naturals (x : ℕ) : list ℕ := list.map (λ n, n + x) (list.range 11)

def represents_unique_characters (chars : list ℕ) : Prop :=
  chars.nodup

def idiom_sum_to_21 (idiom : list ℕ) : Prop :=
  idiom.sum = 21

def system_of_idioms (idioms : list (list ℕ)) : Prop :=
  (∀ idiom ∈ idioms, idiom_sum_to_21 idiom) ∧ 
  represents_unique_characters (idioms.join)

-- Prove the maximum value of character '行' can be 8
theorem max_value_of_xing (x : ℕ) (idioms : list (list ℕ)) :
  system_of_idioms idioms →
  let 行 := (idioms.join.filter (λ n, n = x)).head 8 in
  行 ≤ 8 :=
begin
  sorry
end

end max_value_of_xing_l566_566036


namespace maximum_distance_l566_566414

open Real

noncomputable def line1 (k : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), k * x - y + 2 = 0
noncomputable def line2 (k : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), x + k * y - 2 = 0
noncomputable def line (x y : ℝ) : ℝ := x - y - 4

def point_M : ℝ × ℝ := (0, 2)

def distance_point_to_line (point : ℝ × ℝ) (line: ℝ × ℝ → ℝ) : ℝ :=
  abs (line point) / sqrt (1^2 + (-1)^2)

theorem maximum_distance (k : ℝ) :
  distance_point_to_line point_M line = 3 * sqrt 2 :=
by
  sorry

end maximum_distance_l566_566414


namespace distance_focus_directrix_parabola_l566_566372

theorem distance_focus_directrix_parabola (x y : ℝ) :
  (x^2 = (1 / 4) * y) →
  let focus := (0 : ℝ, 1 / 16 : ℝ)
  let directrix := (λ x, -1 / 16) in
  dist focus (0, directrix 0) = 1 / 8 :=
by
  intros h
  let focus := (0 : ℝ, 1 / 16 : ℝ)
  let directrix := (λ x, -1 / 16) 
  -- the distance formula for vertical points dist(a,b) = (a - b).nat_abs
  sorry

end distance_focus_directrix_parabola_l566_566372


namespace quadratic_to_vertex_form_l566_566868

theorem quadratic_to_vertex_form : 
  ∃ a b c : ℝ, (∀ x : ℝ, 12 * x^2 + 72 * x + 300 = a * (x + b)^2 + c) ∧ a + b + c = 207 :=
by 
  use 12, 3, 192
  split
  { intro x
    calc 
      12 * x^2 + 72 * x + 300
          = 12 * (x^2 + 6 * x + 25) : by ring
      ... = 12 * ((x + 3)^2 + 16) : by rw [←add_assoc, ←nat.cast_two, Nat.cast_add, pow_two, bit0, add_mul]
      ... = 12 * (x + 3)^2 + 192 : by ring },
  { exact add_add_add_comm.mpr rfl }

/- The statement rewrites the given problem as a Lean 4 theorem. It states that there exist constants a, b, and c such that the quadratic expression can be expressed in the required form, and their sum equals 207. -/

end quadratic_to_vertex_form_l566_566868


namespace minimum_tangent_length_l566_566487

theorem minimum_tangent_length :
  let L := line (λ x, x + 1)
  let C := circle { center := (3, 0), radius := 1 } in
  ∀ P : Point, L.contains(P) → tangent_length(C, P) = real.sqrt 7 := by
  sorry

end minimum_tangent_length_l566_566487


namespace combined_mpg_rate_l566_566462

-- Conditions of the problem
def ray_mpg : ℝ := 48
def tom_mpg : ℝ := 24
def ray_distance (s : ℝ) : ℝ := 2 * s
def tom_distance (s : ℝ) : ℝ := s

-- Theorem to prove the combined rate of miles per gallon
theorem combined_mpg_rate (s : ℝ) (h : s > 0) : 
  let total_distance := tom_distance s + ray_distance s
  let ray_gas_usage := ray_distance s / ray_mpg
  let tom_gas_usage := tom_distance s / tom_mpg
  let total_gas_usage := ray_gas_usage + tom_gas_usage
  total_distance / total_gas_usage = 36 := 
by
  sorry

end combined_mpg_rate_l566_566462


namespace tan_half_angles_l566_566385

theorem tan_half_angles (a b : ℝ) (ha : 3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0) :
  ∃ z : ℝ, z = Real.tan (a / 2) * Real.tan (b / 2) ∧ (z = Real.sqrt (6 / 13) ∨ z = -Real.sqrt (6 / 13)) :=
by
  sorry

end tan_half_angles_l566_566385


namespace solve_for_y_l566_566694

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y_l566_566694


namespace F_8_not_true_F_6_might_be_true_l566_566201

variable {n : ℕ}

-- Declare the proposition F
variable (F : ℕ → Prop)

-- Placeholder conditions
axiom condition1 : ¬ F 7
axiom condition2 : ∀ k : ℕ, k > 0 → (F k → F (k + 1))

-- Proof statements
theorem F_8_not_true : ¬ F 8 :=
by {
  sorry
}

theorem F_6_might_be_true : ¬ ¬ F 6 :=
by {
  sorry
}

end F_8_not_true_F_6_might_be_true_l566_566201


namespace sin_add_half_pi_alpha_eq_identical_symmetry_axes_domain_of_y_correct_proposition_l566_566204

-- Problem 1
theorem sin_add_half_pi_alpha_eq : 
  ∀ α : ℝ, (Real.sin (π / 6 + α) = 3 / 5) → (Real.cos (α - π / 3) = 3 / 5) :=
by
  intros α h
  sorry

-- Problem 2
theorem identical_symmetry_axes :
  ∀ φ : ℝ, (|φ| < π / 2 → φ = -π / 3) :=
by
  intro φ hyp
  have h_sym : |φ| < π / 2 := hyp
  sorry

-- Problem 3
theorem domain_of_y :
  ∃ (k : ℤ) (x : ℝ) , (x ∈ [π / 6 + k * π, π / 3 + k * π)) ∧ y = Real.sqrt (Real.sin (2 * x - π / 3)) + Real.log (Real.tan (x + π / 6)) :=
by
  use k
  sorry

-- Problem 4
theorem correct_proposition : 
  ∀ (a b : ℝ → ℝ), (∀ x y : ℝ, |a + b| ≤ |a| + |b|) :=
by
  intros a b
  sorry

end sin_add_half_pi_alpha_eq_identical_symmetry_axes_domain_of_y_correct_proposition_l566_566204


namespace dolphins_to_be_trained_next_month_l566_566511

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end dolphins_to_be_trained_next_month_l566_566511


namespace MOREMOM_arrangements_l566_566626

theorem MOREMOM_arrangements : 
  let total_letters := 7
  let frequency_M := 3
  let frequency_O := 2
  let arrangements := Nat.factorial total_letters / (Nat.factorial frequency_M * Nat.factorial frequency_O)
  in arrangements = 420 :=
by 
  sorry

end MOREMOM_arrangements_l566_566626


namespace slope_angle_of_inclination_l566_566148

theorem slope_angle_of_inclination : 
  (∃ θ : ℝ, tan θ = √3 ∧ θ = 60 * real.pi / 180) :=
begin
  sorry
end

end slope_angle_of_inclination_l566_566148


namespace photo_trip_ratio_l566_566804

theorem photo_trip_ratio
  (initial_photos : ℕ)
  (total_photos_after_trip : ℕ)
  (photos_first_day : ℕ)
  (photos_second_day_more: ℕ)
  (photos_second_day := photos_first_day + photos_second_day_more)
  (total_photos_day_1_and_2 := photos_first_day + photos_second_day)
  (initial_photos : ℕ := 400)
  (total_photos_after_trip : ℕ := 920)
  (photos_second_day_more : ℕ := 120)
  (h : initial_photos + total_photos_day_1_and_2 = total_photos_after_trip) : 
  (photos_first_day : initial_photos) = 1:2 := 
by
  -- Using the given assumptions
  sorry

end photo_trip_ratio_l566_566804


namespace matrix_mul_correct_l566_566321

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := 
![
  ![3, -1],
  ![5,  7]
]

noncomputable def matrix_B : Matrix (Fin 2) (Fin 3) ℤ :=
![
  ![2,  1,  4],
  ![1,  0, -2]
]

noncomputable def matrix_C : Matrix (Fin 2) (Fin 3) ℤ :=
![
  ![5,  3, 14],
  ![17,  5,  6]
]

theorem matrix_mul_correct : matrix_A ⬝ matrix_B = matrix_C :=
  by {
    -- Usual place for the proof
    sorry
  }

end matrix_mul_correct_l566_566321


namespace least_possible_remainder_l566_566588

open Nat

def deviation (p : ℕ) (S : Finset ℕ) : ℤ :=
  let M := Finset.range p \ {0}
  let Π_S := ∏ x in S, (x : ℤ)
  let Π_S_complement := ∏ x in (M \ S), (x : ℤ)
  Π_S - Π_S_complement

theorem least_possible_remainder (p : ℕ) (S : Finset ℕ) (h₁ : nat.prime p) (h₂ : ∃ n, p = 12 * n + 11) (h₃ : S.card = (p - 1) / 2) (h₄ : let M := Finset.range p \ {0} in 
  (∏ x in S, x) ≥ (∏ x in (M \ S), x)) : 
  ∃ q : ℤ, q % p = 2 ∧ q = deviation p S :=
sorry

end least_possible_remainder_l566_566588


namespace fractions_of_120_equals_2_halves_l566_566910

theorem fractions_of_120_equals_2_halves :
  (1 / 6) * (1 / 4) * (1 / 5) * 120 = 2 / 2 := 
by
  sorry

end fractions_of_120_equals_2_halves_l566_566910


namespace ellipse_params_sum_l566_566799

theorem ellipse_params_sum :
  let F1 : ℝ × ℝ := (0, 0)
  let F2 : ℝ × ℝ := (6, 0)
  let P (h k a b : ℝ) := ∃ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1
  (PF1 PF2 : ℝ → ℝ × ℝ)
  (∑ (PF1 : ℝ) (PF2 : ℝ), (λ t, dist PF1 F1 + dist PF2 F2))= 10
  in (3 : ℝ) + (0 : ℝ) + (5 : ℝ) + (4 : ℝ) = 12 := 
sorry

end ellipse_params_sum_l566_566799


namespace infinite_non_expressible_integers_l566_566470

theorem infinite_non_expressible_integers :
  ∃ (S : Set ℤ), S.Infinite ∧ (∀ n ∈ S, ∀ a b c : ℕ, n ≠ 2^a + 3^b - 5^c) :=
sorry

end infinite_non_expressible_integers_l566_566470


namespace divisible_by_six_l566_566004

theorem divisible_by_six (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 6 ∣ n :=
sorry

end divisible_by_six_l566_566004


namespace sum_even_powers_eq_7_pow_5_l566_566000

theorem sum_even_powers_eq_7_pow_5 (a : Fin 12 → ℤ) :
  (∀ x : ℤ, (x + 1) * (2 * x^2 - 1)^5 = ∑ i in Finset.range 12, a i * x^i) →
  a 0 + 2^2 * a 2 + 2^4 * a 4 + 2^6 * a 6 + 2^8 * a 8 + 2^10 * a 10 = 7^5 :=
by sorry

end sum_even_powers_eq_7_pow_5_l566_566000


namespace closest_axis_of_symmetry_l566_566744

def original_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

def shifted_function (x : ℝ) : ℝ :=
  original_function (x - Real.pi / 6)

def axis_of_symmetry (x : ℝ) : ℝ :=
  2 * x - Real.pi / 12

theorem closest_axis_of_symmetry : ∃ (x : ℝ), axis_of_symmetry x = - 5 * Real.pi / 24 :=
  sorry

end closest_axis_of_symmetry_l566_566744


namespace quadratic_roots_proof_l566_566653

theorem quadratic_roots_proof (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c = 0 ↔ (x = 1 ∨ x = -2)) → (b = 1 ∧ c = -2) :=
by
  sorry

end quadratic_roots_proof_l566_566653


namespace angle_of_line_eq_60_degrees_l566_566149

-- Define the slope of the line
def slope (line : ℝ → ℝ) : ℝ := 3.sqrt

-- Define the angle of inclination as an equality to the arc tangent of the slope
def angle_of_inclination (slope : ℝ) : ℝ := Real.arctan slope

-- Prove that the angle of inclination for the given line is 60 degrees (π/3 radians)
theorem angle_of_line_eq_60_degrees (θ : ℝ) (h : θ = Real.arctan (3.sqrt)) : θ = Real.pi / 3 :=
by
  sorry

end angle_of_line_eq_60_degrees_l566_566149


namespace contrapositive_l566_566110

theorem contrapositive (x y : ℝ) : (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by
  intro h
  sorry

end contrapositive_l566_566110


namespace total_rainfall_2006_l566_566017

theorem total_rainfall_2006 (avg_rainfall_2005 : ℕ) (increase_2006 : ℕ) (months : ℕ) :
  avg_rainfall_2005 = 40 →
  increase_2006 = 3 →
  months = 12 →
  (avg_rainfall_2005 + increase_2006) * months = 516 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  show (40 + 3) * 12 = 516
  sorry

end total_rainfall_2006_l566_566017


namespace introduces_new_solutions_l566_566601

variable {α : Type*} [LinearOrder α] {F1 F2 f : α → ℝ}

theorem introduces_new_solutions (h1 : ∀ x, F1(x) > F2(x)) (h2 : ∀ x, f(x) < 0) :
  ∃ x, (F1(x) < F2(x)) → (f(x) * F1(x) < f(x) * F2(x)) :=
sorry

end introduces_new_solutions_l566_566601


namespace exists_unique_decomposition_l566_566708

theorem exists_unique_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ),
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n, z n ≥ z (n-1)) ∧
    (∀ n, y n * (z n - z (n-1)) = 0) ∧
    z 0 = 0 :=
sorry

end exists_unique_decomposition_l566_566708


namespace ratio_areas_l566_566586

-- Define the perimeter P
variable (P : ℝ) (hP : P > 0)

-- Define the side lengths
noncomputable def side_length_square := P / 4
noncomputable def side_length_triangle := P / 3

-- Define the radius of the circumscribed circle for the square
noncomputable def radius_square := (P * Real.sqrt 2) / 8
-- Define the area of the circumscribed circle for the square
noncomputable def area_circle_square := Real.pi * (radius_square P)^2

-- Define the radius of the circumscribed circle for the equilateral triangle
noncomputable def radius_triangle := (P * Real.sqrt 3) / 9 
-- Define the area of the circumscribed circle for the equilateral triangle
noncomputable def area_circle_triangle := Real.pi * (radius_triangle P)^2

-- Prove the ratio of the areas is 27/32
theorem ratio_areas (P : ℝ) (hP : P > 0) : 
  (area_circle_square P / area_circle_triangle P) = (27 / 32) := by
  sorry

end ratio_areas_l566_566586


namespace slytherin_postcards_l566_566168

theorem slytherin_postcards (n : ℕ) (h₁ : n = 30) (h₂ : ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ z ≠ x →
  ¬ (friend x y ∧ friend y z ∧ friend z x)) :
  (max_postcards n = 450) :=
sorry

variable {α : Type*} (friend : α → α → Prop)

variable (max_postcards : ℕ → ℕ)

def max_postcards := λ n, if n = 30 then 450 else 0  -- This is an auxiliary definition for 
                                                    -- demonstration. In real cases, it might
                                                    -- depend on actual computation.

end slytherin_postcards_l566_566168


namespace f_eq_correct_l566_566613

-- Define the function f(n) as per the given triangular structure
def f : ℕ → ℕ
| 0     := 0
| (n+1) := 4 * 2^n - 2 * (n + 1)

-- The main theorem to prove
theorem f_eq_correct : f 100 = 4 * 2^99 - 200 :=
by
  sorry

end f_eq_correct_l566_566613


namespace parametric_eq_C2_general_eq_l1_l566_566416

-- Part (I): Prove the parametric equation of C2 given the conditions.
theorem parametric_eq_C2 : 
  ∀ θ : ℝ, ∃ x y : ℝ, (x = 2 * cos θ ∧ y = sin θ) ∧ (x^2 / 4 + y^2 = 1) :=
by
  sorry

-- Part (II): Prove the general equation of line l1 given the conditions.
theorem general_eq_l1 : 
  ∀ (A B C D : ℝ), 
  (A = 2 * cos θ ∧ B = sin θ ∧ C = 2 * cos φ ∧ D = sin φ) → 
  (maximize (perimeter_quadrilateral A B C D)) → 
  (line_through_origin_symmetric_y L1 L2) → 
  (L1 : y = 1/4 * x) :=
by
  sorry

end parametric_eq_C2_general_eq_l1_l566_566416


namespace problem_A_l566_566665

theorem problem_A (α β : Real) (h1 : π/4 ≤ α ∧ α ≤ π) (h2 : π ≤ β ∧ β ≤ 3*π/2)
(h3 : sin(2*α) = 4/5) (h4 : cos(α + β) = -Real.sqrt 2 / 10) :
  sin(α) - cos(α) = Real.sqrt 5 / 5 ∧ β - α = 3*π/4 := 
sorry

end problem_A_l566_566665


namespace area_square_II_l566_566802

noncomputable def square_diagonal_to_side_length (d : ℝ) : ℝ :=
  d / Real.sqrt 2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

theorem area_square_II (a b: ℝ) : 
  let d_I := 2 * a + 3 * b in
  let s_I := square_diagonal_to_side_length d_I in
  let A_I := square_area s_I in
  let A_II := 3 * A_I in
  A_II = (3 * (2 * a + 3 * b)^2) / 2 :=
by
  sorry

end area_square_II_l566_566802


namespace largest_multiple_of_11_lt_neg150_l566_566533

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l566_566533


namespace general_term_sequence_l566_566707

def a_n (n : ℕ) : ℚ := (2 * n - 1) / (n^2)

theorem general_term_sequence :
  ∀ n : ℕ, a_n n = (2 * n - 1) / (n^2) :=
by {
  intro n,
  -- skipping proof
  sorry
}

end general_term_sequence_l566_566707


namespace isosceles_triangle_cos_solutions_l566_566500

theorem isosceles_triangle_cos_solutions (x : ℝ) :
  (0 < x ∧ x < 90) ∧ 
  (cos x = cos x) ∧ 
  (cos x = cos x) ∧ 
  (cos 9x = cos 9x) ∧ 
  (3 * x = 3 * x) -> 
  (x = 37.5 ∨ x = 45 ∨ x = 75) :=
by sorry

end isosceles_triangle_cos_solutions_l566_566500


namespace arithmetic_sequence_sum_l566_566753

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a n = a m + d * (n - m)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n * (a 1 + a n)) / 2

-- Given conditions
def given_conditions (a : ℕ → ℝ) : Prop :=
a 1 - a 4 + a 8 - a 12 + a 15 = 2

-- Statement to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : given_conditions a) :
  sum_arithmetic_sequence a 15 = -15 :=
sorry

end arithmetic_sequence_sum_l566_566753


namespace acute_isosceles_cos_sine_l566_566146

theorem acute_isosceles_cos_sine (x : ℝ) (hx : 0 < x ∧ x < 90) :
  (cos (5 * x) = sin (4 * x)) → (x = 10 ∨ x = 50) :=
by
  sorry

end acute_isosceles_cos_sine_l566_566146


namespace statement_A_statement_B_statement_C_statement_D_correct_statements_l566_566349

-- Define the vectors and the condition that they form a basis (are linearly independent)
variables {V : Type*} [add_comm_group V] [module ℝ V]
variables {a b c : V}

-- Abstracting the statement that a, b, c form a basis (are linearly independent)
def is_basis (a b c : V) : Prop :=
  linear_independent ℝ ![a, b, c]

-- Given condition: a, b, c form a basis of V
axiom basis_condition : is_basis a b c

-- Proof obligations:
theorem statement_A : (∀ (x y z : ℝ), x • a + y • b + z • c = 0 → x = 0 ∧ y = 0 ∧ z = 0) :=
by {
  -- basis_condition will be used here
  sorry
}

theorem statement_B : ¬(linear_dependent ℝ ![a, b, c]) :=
by {
  -- contradiction argument based on the definition of basis_condition
  sorry
}

theorem statement_C : ¬(∃ (x y : ℝ), a = x • b + y • c) :=
by {
  -- basis_condition will be used here to show impossibility
  sorry
}

theorem statement_D : linear_independent ℝ ![a + b, b - c, c + 2 • a] :=
by {
  -- basis_condition will be used here to show linear independence
  sorry
}

-- From the above theorems, we can list out the correct statements
theorem correct_statements : (statement_A ∧ statement_D ∧ ¬ statement_B ∧ ¬ statement_C) :=
by {
  split, exact statement_A,
  split, exact statement_D,
  split, exact statement_B,
  exact statement_C,
}

end statement_A_statement_B_statement_C_statement_D_correct_statements_l566_566349


namespace smallest_value_among_exponentiations_l566_566191

theorem smallest_value_among_exponentiations (x : ℝ) (h : x = 2016) :
  x^(-1) < x^(-1/2) ∧ x^(-1) < x^(0) ∧ x^(-1) < x^(1/2) ∧ x^(-1) < x^(1) :=
by
  sorry

end smallest_value_among_exponentiations_l566_566191


namespace students_who_like_both_l566_566379

/-- Define the variables according to the problem's conditions --/
variables (total_students ramen_lovers bread_lovers neither_lovers : ℕ)

/-- The main theorem to prove the number of students who like both ramen and bread --/
theorem students_who_like_both (h1 : total_students = 500)
                               (h2 : ramen_lovers = 289)
                               (h3 : bread_lovers = 337)
                               (h4 : neither_lovers = 56)
                               : ramen_lovers + bread_lovers - total_students - neither_lovers = 182 :=
by 
  sorry

end students_who_like_both_l566_566379


namespace death_rate_calculation_l566_566113

theorem death_rate_calculation
  (birth_rate : ℕ)
  (net_growth_rate : ℝ)
  (initial_population : ℕ)
  (death_rate : ℕ) :
  birth_rate = 52 →
  net_growth_rate = 0.012 →
  initial_population = 3000 →
  0.012 * 3000 = 52 - death_rate →
  death_rate = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact eq.symm h4

end death_rate_calculation_l566_566113


namespace number_sum_20_eq_30_l566_566152

theorem number_sum_20_eq_30 : ∃ x : ℤ, 20 + x = 30 → x = 10 :=
by {
  sorry
}

end number_sum_20_eq_30_l566_566152


namespace cost_of_one_dozen_pens_l566_566854

-- Define the conditions
def cost_of_pen (x : ℕ) : ℕ := 5 * x
def cost_of_pencil (x : ℕ) : ℕ := x

-- Given the first equation from the problem
def equation1 (x : ℕ) : Prop := 3 * (cost_of_pen x) + 5 * (cost_of_pencil x) = 260

-- Suppose x is the cost of one pencil
def pencil_cost := 13

-- Define the cost of one pen using the ratio given in the conditions
def pen_cost := cost_of_pen pencil_cost

-- Prove the cost of one dozen pens
def cost_of_dozen_pens : ℕ := 12 * pen_cost

-- The main theorem stating the cost of one dozen pens
theorem cost_of_one_dozen_pens : cost_of_dozen_pens = 780 :=
by
  -- Main steps follow directly from the given and calculations
  have cost_condition : equation1 pencil_cost, from rfl, -- This follows from the equation we defined.
  -- Cost of one pen
  have pen_cost_eq : pen_cost = 65, by { unfold pen_cost, unfold pencil_cost, unfold cost_of_pen, rfl },
  -- Calculate the cost of one dozen pens
  have dozen_cost_eq : cost_of_dozen_pens = 12 * 65, by { unfold cost_of_dozen_pens, rw pen_cost_eq },
  -- Simplify to get the final cost
  rw dozen_cost_eq,
  norm_num,
  rfl

end cost_of_one_dozen_pens_l566_566854


namespace S_2016_eq_l566_566703

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

-- Define the derivative of the function f
def f_prime (x : ℝ) (a : ℝ) : ℝ := 2 * x - a

-- Define the tangent condition
def is_perpendicular (a : ℝ) : Prop := 
  (f_prime 1 a) * (-(1 / 3)) = -1

-- Define the sequence term
def seq_term (n : ℕ) : ℝ := 1 / (n * (n + 1))

-- Define the sum of the sequence term
def S (n : ℕ) : ℝ := 
  ∑ i in Finset.range n, seq_term (i + 1)

-- Primary theorem statement
theorem S_2016_eq : 
  let a := -1 in 
  is_perpendicular a ∧ S 2016 = (2016 / 2017) := 
begin
  sorry
end

end S_2016_eq_l566_566703


namespace midpoint_of_B_l566_566866

/-- Define the initial points B, I, G -/
def B : ℝ × ℝ := (1, 1)
def I : ℝ × ℝ := (2, 4)
def G : ℝ × ℝ := (5, 1)

/-- Define the result of rotating each point 90 degrees clockwise -/
def rotate90 (p: ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

/-- Apply the rotation to each point -/
def B' : ℝ × ℝ := rotate90 B
def I' : ℝ × ℝ := rotate90 I
def G' : ℝ × ℝ := rotate90 G

/-- Define the translation by three units left and four units up -/
def translate (p: ℝ × ℝ) : ℝ × ℝ := (p.1 - 3, p.2 + 4)

/-- Apply the translation to each rotated point -/
def B'' : ℝ × ℝ := translate B'
def I'' : ℝ × ℝ := translate I'
def G'' : ℝ × ℝ := translate G'

/-- Define the function to calculate the midpoint of two points -/
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- State the theorem to be proved -/
theorem midpoint_of_B''_G'' : midpoint B'' G'' = (-2, 1) := sorry

end midpoint_of_B_l566_566866


namespace problem1_problem2_l566_566748

theorem problem1 (a b c : ℝ) (A B C : ℝ) (h1 : A + B + C = π) (h2 : a = b / (cos C)) (h3 : b = c / (cos A)) (h4 : c = a / (cos B)) :
  b * cos C + c * cos B = 2 * a * cos A → A = π / 3 :=
begin
  sorry
end

theorem problem2 (a b c : ℝ) (A : ℝ) (h1 : a = 3 * √2) (h2 : b + c = 6) (h3 : A = π / 3) :
  |(b, c)| = √30 :=
begin
  sorry
end

end problem1_problem2_l566_566748


namespace average_birth_rate_l566_566406

theorem average_birth_rate (B : ℕ) 
  (death_rate : ℕ := 3)
  (daily_net_increase : ℕ := 86400) 
  (intervals_per_day : ℕ := 86400 / 2) 
  (net_increase : ℕ := (B - death_rate) * intervals_per_day) : 
  net_increase = daily_net_increase → 
  B = 5 := 
sorry

end average_birth_rate_l566_566406


namespace parabola_y_intercepts_zero_l566_566723

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l566_566723


namespace exists_triangle_BM_MN_ND_decrease_radius_when_MN_decreases_l566_566332

-- Condition Definitions
variables {A B C D E F M N : Type}
variables [parallelogram ABCD] -- ABCD is a parallelogram
variables [point_on_side E BC] -- E is a point on side BC
variables [point_on_side F CD] -- F is a point on side CD
variables [same_area (triangle ABE) (triangle BCF)] -- triangles ABE and BCF have the same area

-- Intersections of diagonals with lines
variables [intersect_at BD AE M] -- Diagonal BD intersects AE at M
variables [intersect_at BD AF N] -- Diagonal BD intersects AF at N

-- Prove existence of a triangle with sides BM, MN, ND
theorem exists_triangle_BM_MN_ND :
  ∃ (T : triangle) (B M N D : Type), 
    sides_of_triangle T = (BM, MN, ND) :=
sorry

-- Prove circumradius condition
theorem decrease_radius_when_MN_decreases :
  decreasing_side MN → decreasing_circumradius (triangle_with_sides BM MN ND) :=
sorry

end exists_triangle_BM_MN_ND_decrease_radius_when_MN_decreases_l566_566332


namespace problem1_problem2_l566_566362

noncomputable def a_k (n : ℕ) (k : ℕ) (x : ℝ) : ℝ := (n.choose (k-1)) * (1 / 2 * x)^(k-1)

noncomputable def F (n : ℕ) (x : ℝ) : ℝ := ∑ i in Finset.range (n + 1), (i + 1) * a_k n (i + 1) x

open Finset

theorem problem1 (n : ℕ) :
  if (a_k n 1 0 = 1) ∧ (a_k n 2 0 = n / 2) ∧ (a_k n 3 0 = n * (n - 1) / 8)
  then n = 8 :=
sorry

theorem problem2 (n : ℕ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) :
  |F n x1 - F n x2| ≤ 2^(n-1) * (n+2) - 1 :=
sorry

end problem1_problem2_l566_566362


namespace travelers_on_liner_l566_566887

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566887


namespace pow_mod_eq_l566_566103

theorem pow_mod_eq :
  (13 ^ 7) % 11 = 7 :=
by
  sorry

end pow_mod_eq_l566_566103


namespace min_value_of_expression_l566_566733

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + (1 / (a - 1)) ≥ 3 :=
by sorry

end min_value_of_expression_l566_566733


namespace angle_properties_l566_566664

namespace angle_problem

def angle := -1920 * Real.pi / 180
def quadrant := 3
def candidate_thetas : List ℝ := [-2 * Real.pi / 3, -8 * Real.pi / 3]

theorem angle_properties :
  ∃ beta k, alpha = beta + 2 * k * Real.pi ∧ 0 ≤ beta ∧ beta < 2 * Real.pi ∧ (quadrant = 3) ∧ 
  ∃ theta, theta ∈ candidate_thetas ∧ -4 * Real.pi ≤ theta ∧ theta < 0 := 
by
  sorry

end angle_problem

end angle_properties_l566_566664


namespace hoseok_subtraction_result_l566_566228

theorem hoseok_subtraction_result:
  ∃ x : ℤ, 15 * x = 45 ∧ x - 1 = 2 :=
by
  sorry

end hoseok_subtraction_result_l566_566228


namespace Clark_Travel_Time_Comparison_l566_566411

noncomputable def speed (s : ℝ) (t_1 t_2 : ℝ) : Prop :=
  t_1 = 120 / s ∧
  t_2 = 240 / (2 * s) ∧
  t_1 = t_2

theorem Clark_Travel_Time_Comparison (s : ℝ) (t_1 t_2 : ℝ) :
  speed s t_1 t_2 → t_1 = t_2 :=
by
  intro h
  cases h with h1 h2
  exact h2.2

end Clark_Travel_Time_Comparison_l566_566411


namespace area_percent_less_l566_566935

theorem area_percent_less 
  (r1 r2 : ℝ)
  (h : r1 / r2 = 3 / 10) 
  : 1 - (π * (r1:ℝ)^2 / (π * (r2:ℝ)^2)) = 0.91 := 
by 
  sorry

end area_percent_less_l566_566935


namespace gg_has_exactly_three_distinct_real_roots_l566_566060

noncomputable def g (x d : ℝ) := x^2 + 4 * x + d

theorem gg_has_exactly_three_distinct_real_roots (d : ℝ) :
  (∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ g (g x1) d = 0 ∧ g (g x2) d = 0 ∧ g (g x3) d = 0) ↔ d = 0 :=
by
  sorry

end gg_has_exactly_three_distinct_real_roots_l566_566060


namespace euler_formula_quadrant_l566_566788

theorem euler_formula_quadrant (θ : ℝ) (h : θ = 2) : 
  let z := complex.exp (complex.I * θ)
  in -1 < z.re ∧ z.re < 0 ∧ 0 < z.im ∧ z.im < 1 :=
by
  sorry

end euler_formula_quadrant_l566_566788


namespace important_emails_count_l566_566453

def total_emails : ℕ := 400
def spam_fraction : ℚ := 1 / 4
def promotional_fraction : ℚ := 2 / 5

theorem important_emails_count :
  let spam_emails := spam_fraction * total_emails
      non_spam_emails := total_emails - (spam_emails:ℕ)
      promotional_emails := promotional_fraction * non_spam_emails
  in important_emails = 180 :=
by
  sorry

end important_emails_count_l566_566453


namespace sqrt_fraction_sum_l566_566602

theorem sqrt_fraction_sum : 
    Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30 := 
by
  sorry

end sqrt_fraction_sum_l566_566602


namespace ratio_of_areas_l566_566463

theorem ratio_of_areas 
  (lenA : ℕ) (brdA : ℕ) (lenB : ℕ) (brdB : ℕ)
  (h_lenA : lenA = 48) 
  (h_brdA : brdA = 30)
  (h_lenB : lenB = 60) 
  (h_brdB : brdB = 35) :
  (lenA * brdA : ℚ) / (lenB * brdB) = 24 / 35 :=
by
  sorry

end ratio_of_areas_l566_566463


namespace find_a_l566_566741

noncomputable def a : ℝ := sqrt 3

theorem find_a 
  (a_pos : a > 0)
  (hyperbola : ∀ x y : ℝ, (x^2 / a^2) - y^2 = 1)
  (distance : ∀ P : ℝ × ℝ, (P = (2, 0)) → ∀ line : ℝ → ℝ, (line = λ x, x / a) → 
  (dist : ℝ) , dist = (|2| / sqrt (1 + a^2)) → dist = 1):
  a = sqrt 3 := sorry

end find_a_l566_566741


namespace parabola_y_intercepts_zero_l566_566722

-- Define the quadratic equation
def quadratic (a b c y: ℝ) : ℝ := a * y^2 + b * y + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Condition: equation of the parabola and discriminant calculation
def parabola_equation : Prop := 
  let a := 3
  let b := -4
  let c := 5
  discriminant a b c < 0

-- Statement to prove
theorem parabola_y_intercepts_zero : 
  (parabola_equation) → (∀ y : ℝ, quadratic 3 (-4) 5 y ≠ 0) :=
by
  intro h
  sorry

end parabola_y_intercepts_zero_l566_566722


namespace probability_each_player_has_3_l566_566896

noncomputable def three_friends_game_probability : ℚ :=
  sorry

theorem probability_each_player_has_3 :
  three_friends_game_probability = 1 / 4 :=
  sorry

end probability_each_player_has_3_l566_566896


namespace complex_magnitude_l566_566355

noncomputable def is_purely_imaginary (x : ℂ) : Prop :=
  x.re = 0

noncomputable def magnitude (z : ℂ) : ℝ :=
  complex.abs z

theorem complex_magnitude (a : ℝ) 
    (h1 : is_purely_imaginary ((2 - complex.i) / (a + complex.i))) :
    magnitude ((2 * a + 1) + real.sqrt 2 * complex.i) = real.sqrt 6 :=
by
  sorry

end complex_magnitude_l566_566355


namespace total_cost_of_fruit_l566_566192

theorem total_cost_of_fruit (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 58) 
  (h2 : 3 * x + 2 * y = 72) : 
  3 * x + 3 * y = 78 := 
by
  sorry

end total_cost_of_fruit_l566_566192


namespace probability_three_heads_in_seven_tosses_l566_566573

theorem probability_three_heads_in_seven_tosses :
  (Nat.choose 7 3 : ℝ) / (2 ^ 7 : ℝ) = 35 / 128 :=
by
  sorry

end probability_three_heads_in_seven_tosses_l566_566573


namespace ball_rebound_original_height_l566_566948

theorem ball_rebound_original_height (H : ℝ) (H_positive : 0 < H)
  (total_distance : ℝ) (h_total_distance : total_distance = 320) :
  let first_rebound := 0.5 * H,
      second_rebound := 0.4 * first_rebound,
      third_rebound := 0.3 * second_rebound,
      fourth_rebound := 0.2 * third_rebound,
      fifth_rebound := 0.1 * fourth_rebound,
      total_travel := H + first_rebound + first_rebound + second_rebound + second_rebound +
                      third_rebound + third_rebound + fourth_rebound + fourth_rebound +
                      fifth_rebound
    in total_travel ≈ 2.5452 * H :=
  sorry

end ball_rebound_original_height_l566_566948


namespace twenty_four_game_l566_566413

-- Definition of the cards' values
def card2 : ℕ := 2
def card5 : ℕ := 5
def cardJ : ℕ := 11
def cardQ : ℕ := 12

-- Theorem stating the proof
theorem twenty_four_game : card2 * (cardJ - card5) + cardQ = 24 :=
by
  sorry

end twenty_four_game_l566_566413


namespace find_C_l566_566580

theorem find_C (A B C : ℕ) :
  (8 + 5 + 6 + 3 + 2 + A + B) % 3 = 0 →
  (4 + 3 + 7 + 5 + A + B + C) % 3 = 0 →
  C = 2 :=
by
  intros h1 h2
  sorry

end find_C_l566_566580


namespace travelers_on_liner_l566_566886

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566886


namespace inequality_proof_l566_566459

theorem inequality_proof
  (x1 x2 x3 x4 x5 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ x1 * (x2 + x3 + x4 + x5) :=
by
  sorry

end inequality_proof_l566_566459


namespace find_b10_l566_566135

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = b (n + 1) + b n

theorem find_b10 (b : ℕ → ℕ) (h0 : ∀ n, b n > 0) (h1 : b 9 = 544) (h2 : sequence_b b) : b 10 = 883 :=
by
  -- We could provide steps of the proof here, but we use 'sorry' to omit the proof content
  sorry

end find_b10_l566_566135


namespace pyramid_surface_area_l566_566852

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7) / 2

theorem pyramid_surface_area (a : ℝ) :
  let hexagon_base_area := 3 * a^2 * Real.sqrt 3 / 2
  let triangle_area_1 := a^2 / 2
  let triangle_area_2 := a^2
  let triangle_area_3 := a^2 * Real.sqrt 7 / 4
  let lateral_area := 2 * (triangle_area_1 + triangle_area_2 + triangle_area_3)
  total_surface_area a = hexagon_base_area + lateral_area := 
sorry

end pyramid_surface_area_l566_566852


namespace total_points_l566_566082

noncomputable def Noa_score : ℕ := 30
noncomputable def Phillip_score : ℕ := 2 * Noa_score
noncomputable def Lucy_score : ℕ := (3 / 2) * Phillip_score

theorem total_points : 
  Noa_score + Phillip_score + Lucy_score = 180 := 
by
  sorry

end total_points_l566_566082


namespace area_ratio_IJKL_WXYZ_l566_566101

-- Definitions based on the conditions
def WXYZ_side := 10
def WI := WXYZ_side / 3
def IX := 2 * WI
def IJKL_side := (Real.sqrt 5 * WXYZ_side) / 3
def area_WXYZ := WXYZ_side ^ 2
def area_IJKL := IJKL_side ^ 2

-- Problem statement
theorem area_ratio_IJKL_WXYZ :
  (area_IJKL / area_WXYZ) = 10 / 9 :=
sorry

end area_ratio_IJKL_WXYZ_l566_566101


namespace measure_of_angle_ADB_l566_566754

def is_equilateral (A B C : Type) [is_semigroup A] : Prop :=
∀ a b c, congruent A B C

variables {A B C D : Type} [is_semigroup A]

theorem measure_of_angle_ADB 
  (h1 : is_equilateral A B C)
  (h2 : points_on_line D A C)
  (h3 : angle A B D = 70) :
  angle A D B = 50 :=
sorry

end measure_of_angle_ADB_l566_566754


namespace part_one_part_two_l566_566669

def f (a x : ℝ) : ℝ := abs (x - a ^ 2) + abs (x + 2 * a + 3)

theorem part_one (a x : ℝ) : f a x ≥ 2 :=
by 
  sorry

noncomputable def f_neg_three_over_two (a : ℝ) : ℝ := f a (-3/2)

theorem part_two (a : ℝ) (h : f_neg_three_over_two a < 3) : -1 < a ∧ a < 0 :=
by 
  sorry

end part_one_part_two_l566_566669


namespace distinct_n_values_1995_l566_566615

open Int

noncomputable def countDistinctValues (k: Int) : Int :=
  if 1 ≤ k ∧ k ≤ 2012 then
    (k^3 / 2012).floor
  else
    0

theorem distinct_n_values_1995 : 
  (Finset.image countDistinctValues (Finset.Icc 1 2012)).card = 1995 := 
sorry

end distinct_n_values_1995_l566_566615


namespace find_q_zero_l566_566064

-- Assuming the polynomials p, q, and r are defined, and their relevant conditions are satisfied.

def constant_term (f : ℕ → ℝ) : ℝ := f 0

theorem find_q_zero (p q r : ℕ → ℝ)
  (h : p * q = r)
  (h_p_const : constant_term p = 5)
  (h_r_const : constant_term r = -10) :
  q 0 = -2 :=
sorry

end find_q_zero_l566_566064


namespace range_of_a_l566_566693

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc (2 : ℝ) 3, x^2 - 2*x + a > 0) → a > 0 :=
by
  intro h
  /- Proof omitted -/
  sorry

end range_of_a_l566_566693


namespace range_of_k_for_obtuse_angle_l566_566397

variable (k : ℝ)
def vector_a := (k, 3)
def vector_b := (1, 4)
def vector_c := (2, 1)
def v := (2 * k - 3, -6)
def dot_product (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem range_of_k_for_obtuse_angle :
  (dot_product (2 * (vector_a k) - 3 * vector_b) vector_c < 0) →
  (2 * k - 3 ≠ 2 * (-6)) →
  (-∞ < k ∧ k < - 9 / 2) ∨ (- 9 / 2 < k ∧ k < 3) :=
by
  sorry

end range_of_k_for_obtuse_angle_l566_566397


namespace KC_eq_AB_l566_566818

variables (A B C K L: Type*)
variables (hTriangle : ∃ (α β γ : ℝ), α + β + γ = π)
variables {L_mid_AK : L = (1 / 2) • A + (1 / 2) • K}
variables {BK_bisects_LBC : is_angle_bisector_of BK (angle L B C)}
variables {BC_eq_2BL : BC = 2 * BL}

theorem KC_eq_AB (A B C K L: Type*) (hTriangle: ∃ (α β γ : ℝ), α + β + γ = π)
    (L_mid_AK : L = (1 / 2) • A + (1 / 2) • K)
    (BK_bisects_LBC : is_angle_bisector_of BK (angle L B C))
    (BC_eq_2BL : BC = 2 * BL) : KC = AB := 
sorry

end KC_eq_AB_l566_566818


namespace pyramid_volume_correct_l566_566142

-- Defining the rectangle and its properties
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (AB BC : ℝ)
  (AB_eq : AB = 6 * real.sqrt 3)
  (BC_eq : BC = 13 * real.sqrt 3)
  (A_coords : A = (0, 0))
  (B_coords : B = (6 * real.sqrt 3, 0))
  (C_coords : C = (6 * real.sqrt 3, 13 * real.sqrt 3))
  (D_coords : D = (0, 13 * real.sqrt 3))

-- Defining the midpoint P of the diagonal AC
def midpoint_AC (r : Rectangle) : ℝ × ℝ :=
  let (x₁, y₁) := r.A in
  let (x₂, y₂) := r.C in
  ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- The volume V of the pyramid formed
noncomputable def volume_pyramid (r : Rectangle) : ℝ :=
  let base_area := (1 / 2) * (r.AB * r.BC) in
  let P := midpoint_AC r in
  let z := -- Calculation for z based on distances PA = PB = PC omitted for brevity in the def
           sorry in
  (1 / 3) * base_area * z

-- The theorem statement to prove
theorem pyramid_volume_correct (r : Rectangle) :
  volume_pyramid r = (1 / 3) * (39 * real.sqrt 3) * sorry :=
sorry

end pyramid_volume_correct_l566_566142


namespace RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l566_566787

-- Definitions representing the conditions
def TotalCrates : ℕ := 50
def PriceGrapes : ℕ := 15
def PriceMangoes : ℕ := 20
def PricePassionFruits : ℕ := 25
def CratesGrapes : ℕ := 13
def CratesMangoes : ℕ := 20
def CratesPassionFruits : ℕ := TotalCrates - CratesGrapes - CratesMangoes

def RevenueWithoutDiscounts : ℕ :=
  (CratesGrapes * PriceGrapes) +
  (CratesMangoes * PriceMangoes) +
  (CratesPassionFruits * PricePassionFruits)

def DiscountGrapes : Float := if CratesGrapes > 10 then 0.10 else 0.0
def DiscountMangoes : Float := if CratesMangoes > 15 then 0.15 else 0.0
def DiscountPassionFruits : Float := if CratesPassionFruits > 5 then 0.20 else 0.0

def DiscountedPrice (price : ℕ) (discount : Float) : Float := 
  price.toFloat * (1.0 - discount)

def RevenueWithDiscounts : Float :=
  (CratesGrapes.toFloat * DiscountedPrice PriceGrapes DiscountGrapes) +
  (CratesMangoes.toFloat * DiscountedPrice PriceMangoes DiscountMangoes) +
  (CratesPassionFruits.toFloat * DiscountedPrice PricePassionFruits DiscountPassionFruits)

-- Proof problems
theorem RevenueWithoutDiscounts_is_1020 : RevenueWithoutDiscounts = 1020 := sorry
theorem RevenueWithDiscounts_is_855_5 : RevenueWithDiscounts = 855.5 := sorry
theorem Difference_is_164_5 : (RevenueWithoutDiscounts.toFloat - RevenueWithDiscounts) = 164.5 := sorry

end RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l566_566787


namespace dexter_boxes_l566_566629

theorem dexter_boxes (x : ℕ)
  (total_cards : ℕ) 
  (cards_per_other_type : ℕ) 
  (cards_per_football : ℕ) : 
  total_cards = 255 ∧ 
  cards_per_other_type = 15 ∧ 
  cards_per_football = 20 ∧ 
  total_cards = cards_per_other_type * x + cards_per_football * (x - 3) →
  x = 9 :=
by {
  intros,
  sorry,
}

end dexter_boxes_l566_566629


namespace six_digit_mod_27_l566_566841

theorem six_digit_mod_27 (X : ℕ) (hX : 100000 ≤ X ∧ X < 1000000) (Y : ℕ) (hY : ∃ a b : ℕ, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ X = 1000 * a + b ∧ Y = 1000 * b + a) :
  X % 27 = Y % 27 := 
by
  sorry

end six_digit_mod_27_l566_566841


namespace clock_rings_in_a_day_l566_566660

-- Define the conditions
def rings_every_3_hours : ℕ := 3
def first_ring : ℕ := 1 -- This is 1 A.M. in our problem
def total_hours_in_day : ℕ := 24

-- Define the theorem
theorem clock_rings_in_a_day (n_rings : ℕ) : 
  (∀ n : ℕ, n_rings = total_hours_in_day / rings_every_3_hours + 1) :=
by
  -- use sorry to skip the proof
  sorry

end clock_rings_in_a_day_l566_566660


namespace trajectory_of_moving_circle_tangent_to_fixed_circles_l566_566176

-- Define the conditions
variables (O1 O2 O : Point)
variables (r1 r2 R : ℝ)
variables (h_diff_radii : r1 ≠ r2)
variables (h_no_coincide : O1 ≠ O2)
variables (h_no_intersect : (dist O1 O2) > (r1 + r2))

-- Define the nature of the trajectory
theorem trajectory_of_moving_circle_tangent_to_fixed_circles :
  ∃ (P : Point → Prop), (∀ p, P p → (dist p O1 - dist p O2 = r1 - r2) ∨ (dist p O1 + dist p O2 = r1 + r2)) :=
sorry

end trajectory_of_moving_circle_tangent_to_fixed_circles_l566_566176


namespace at_least_two_triangles_with_two_good_sides_l566_566071

def is_good_side (j k: ℕ) : Prop := j ∈ Nat ∧ k ∈ Nat

structure Rectangle :=
  (m n : ℕ)
  (m_odd : Odd m)
  (n_odd : Odd n)
  (vertices : List (ℕ × ℕ))
  (well_formed : vertices = [(0,0), (0, m), (n, m), (n, 0)])

structure Triangle :=
  (vertices : List (ℕ × ℕ))
  (good_side : (ℕ × ℕ) → Prop)
  (bad_side : (ℕ × ℕ) → Prop)
  (one_good_side : ∃ side, good_side side ∧ (height_from_side side = 1))
  (bad_side_common : ∀ side, bad_side side → ∃ tri1 tri2, tri1 ≠ tri2 ∧ side ∈ tri1 ∧ side ∈ tri2)

noncomputable def height_from_side (side : (ℕ × ℕ)) : ℕ := sorry

theorem at_least_two_triangles_with_two_good_sides
  (r : Rectangle)
  (triangles : List Triangle)
  (good_sides_condition : ∀ triangle ∈ triangles, triangle.one_good_side)
  (bad_sides_common_condition : ∀ triangle ∈ triangles, ∀ side, triangle.bad_side side → triangle.bad_side_common side) :
  ∃ t1 t2 ∈ triangles, t1 ≠ t2 ∧
  ((∃ side1 side2, t1.good_side side1 ∧ t1.good_side side2 ∧ side1 ≠ side2) ∧
  (∃ side1 side2, t2.good_side side1 ∧ t2.good_side side2 ∧ side1 ≠ side2)) :=
sorry

end at_least_two_triangles_with_two_good_sides_l566_566071


namespace value_of_f_at_6_value_of_f_f_of_0_l566_566692

def f (x : ℝ) : ℝ := if x ≥ 0 then log (x + 2) / log 2 - 3 else log (-x + 2) / log 2 - 3

lemma even_f {x : ℝ} : f (-x) = f x := 
by simp [f, log (-x + 2) / log 2, log (x + 2) / log 2]

theorem value_of_f_at_6 : f 6 = 0 := 
by sorry

theorem value_of_f_f_of_0 : f (f 0) = -1 :=
by sorry

end value_of_f_at_6_value_of_f_f_of_0_l566_566692


namespace bob_grade_is_35_l566_566777

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l566_566777


namespace savings_calculation_l566_566546

theorem savings_calculation (x : ℕ) (h1 : 15 * x = 15000) : (15000 - 8 * x = 7000) :=
sorry

end savings_calculation_l566_566546


namespace total_valid_votes_l566_566933

theorem total_valid_votes (V : ℝ)
  (h1 : ∃ c1 c2 : ℝ, c1 = 0.70 * V ∧ c2 = 0.30 * V)
  (h2 : ∀ c1 c2, c1 - c2 = 182) : V = 455 :=
sorry

end total_valid_votes_l566_566933


namespace problem_i_problem_ii_problem_iii_l566_566073

variables (a b c : ℝ)

-- Question (i)
theorem problem_i (a b c : ℝ) (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  (a + b - c) * (b + c - a) * (c + a - b) ≥ 0 := sorry

-- Question (ii)
theorem problem_ii (a b c : ℝ) (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) := sorry

-- Question (iii)
theorem problem_iii : ∃ (a b c : ℝ), a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) ∧ 
                      a^4 + b^4 + c^4 > 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
begin
  use [1, 1, 3],
  split,
  { norm_num, linarith, },
  { norm_num, linarith, },
end

end problem_i_problem_ii_problem_iii_l566_566073


namespace remainder_calculation_l566_566187

theorem remainder_calculation 
  (x : ℤ) (y : ℝ)
  (hx : 0 < x)
  (hy : y = 70.00000000000398)
  (hx_div_y : (x : ℝ) / y = 86.1) :
  x % y = 7 :=
by
  sorry

end remainder_calculation_l566_566187


namespace sum_series_l566_566303

-- Define the series term
def series_term (n : ℤ) : ℤ := 2 * (-1)^n

-- Define the series sum from -15 to 15
def series_sum : ℤ := ∑ i in finset.Icc (-15 : ℤ) 15, series_term i

theorem sum_series : series_sum = 2 := by
  sorry

end sum_series_l566_566303


namespace number_of_multiples_840_in_range_l566_566726

theorem number_of_multiples_840_in_range :
  ∃ n, n = 1 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 2500 ∧ (840 ∣ x) → x = 1680 :=
by
  sorry

end number_of_multiples_840_in_range_l566_566726


namespace symmetry_axis_of_translated_sine_function_l566_566319

theorem symmetry_axis_of_translated_sine_function :
  ∀ (f g : ℝ → ℝ),
    (∀ x, f x = 2 * sin (2 * x + π / 6)) →
    (∀ x, g x = f (x - π / 6)) →
    (∃ k : ℤ, ∀ x, g x = 2 * sin (2 * x - π / 6) ∧ x = k * (π / 2) + π / 3) :=
by
  intro f g hf hg
  sorry

end symmetry_axis_of_translated_sine_function_l566_566319


namespace vegetables_sold_ratio_l566_566574

def totalMassInstalled (carrots zucchini broccoli : ℕ) : ℕ := carrots + zucchini + broccoli

def massSold (soldMass : ℕ) : ℕ := soldMass

def vegetablesSoldRatio (carrots zucchini broccoli soldMass : ℕ) : ℚ :=
  soldMass / (carrots + zucchini + broccoli)

theorem vegetables_sold_ratio
  (carrots zucchini broccoli soldMass : ℕ)
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8)
  (h_soldMass : soldMass = 18) :
  vegetablesSoldRatio carrots zucchini broccoli soldMass = 1 / 2 := by
  sorry

end vegetables_sold_ratio_l566_566574


namespace price_first_oil_l566_566944

theorem price_first_oil (P : ℝ) (h1 : 10 * P + 5 * 66 = 15 * 58.67) : P = 55.005 :=
sorry

end price_first_oil_l566_566944


namespace time_after_four_classes_Alexander_time_after_science_l566_566593

theorem time_after_four_classes (start_time : ℕ) (classes_completed : ℕ) (class_duration : ℕ) : ℕ :=
  if start_time = 12 ∧ classes_completed = 4 ∧ class_duration = 45 then 
    12 + (4 * 45 / 60)
  else 
    sorry

theorem Alexander_time_after_science : time_after_four_classes 12 4 45 = 3 := by
  simp only [time_after_four_classes]
  sorry

end time_after_four_classes_Alexander_time_after_science_l566_566593


namespace dolphins_to_be_trained_next_month_l566_566510

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end dolphins_to_be_trained_next_month_l566_566510


namespace number_of_pairs_is_2533_l566_566621

open Int

def count_pairs : ℕ :=
  Nat.card { p : ℕ × ℕ // (1 ≤ p.1 ∧ p.1 ≤ 4018) ∧ (3^p.2 < 2^p.1 ∧ 2^p.1 < 2^(p.1 + 3) ∧ 2^(p.1 + 3) < 3^(p.2 + 1)) }

theorem number_of_pairs_is_2533 : count_pairs = 2533 := 
by
  sorry

end number_of_pairs_is_2533_l566_566621


namespace students_passed_finals_l566_566878

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end students_passed_finals_l566_566878


namespace cost_option_1_cost_option_2_most_cost_effective_l566_566566

-- Definitions for conditions
def amplifier_price : ℕ := 300
def pen_price : ℕ := 50
def discount_rate : ℚ := 0.9
def amplifiers_bought : ℕ := 20

-- Problem 1: Cost under Option 1
theorem cost_option_1 (x : ℕ) (h : x > 20) : 
  20 * amplifier_price + (x - 20) * pen_price = 50 * x + 5000 :=
by sorry

-- Problem 2: Cost under Option 2
theorem cost_option_2 (x : ℕ) (h : x > 20) :
  20 * (amplifier_price * discount_rate).toInt + x * (pen_price * discount_rate).toInt = 45 * x + 5400 :=
by sorry

-- Problem 3: Cost-effective option for x = 30
theorem most_cost_effective (x : ℕ) (h : x = 30) :
  20 * amplifier_price + 10 * (pen_price * discount_rate).toInt + 6000 = 6450 :=
by sorry

end cost_option_1_cost_option_2_most_cost_effective_l566_566566


namespace belize_homes_l566_566505

theorem belize_homes (H : ℝ) 
  (h1 : (3 / 5) * (3 / 4) * H = 240) : 
  H = 400 :=
sorry

end belize_homes_l566_566505


namespace triangle_AD_eq_DU_l566_566241

theorem triangle_AD_eq_DU
  (A B C U D : Point)
  (h1 : circumcenter U A B C)
  (h2 : angle B C A = 60)
  (h3 : angle C B U = 45)
  (h4 : ∃ (L D : Point), intersection_point L B U = D ∧ intersection_point L A C = D) :
  dist A D = dist D U :=
sorry

end triangle_AD_eq_DU_l566_566241


namespace point_translation_right_l566_566089

theorem point_translation_right :
  ∀ (A B : ℤ), A = -3 ∧ B = A + 7 → B = 4 :=
by
  intros A B
  intro h
  have h1 : A = -3 := h.1
  have h2 : B = A + 7 := h.2
  rw h1 at h2
  simp at h2
  exact h2

end point_translation_right_l566_566089


namespace altitude_of_triangle_on_rectangle_diagonal_l566_566674

theorem altitude_of_triangle_on_rectangle_diagonal
  (a b : ℝ) 
  (h_rect_area : a * b = (1/2) * (sqrt (a^2 + b^2) * (2 * a * b / (sqrt (a^2 + b^2))))) :
  (2 * a * b) / (sqrt (a^2 + b^2)) = (2 * a * b) / (sqrt (a^2 + b^2)) :=
by 
  sorry

end altitude_of_triangle_on_rectangle_diagonal_l566_566674


namespace surface_area_cuboid_l566_566641

-- Define the dimensions of the cuboid.
def length : ℕ := 8
def breadth : ℕ := 6
def height : ℕ := 9

-- Compute the surface area of the cuboid.
def surface_area (l b h : ℕ) : ℕ := 2 * (l * h) + 2 * (l * b) + 2 * (b * h)

-- The theorem to prove the surface area is equal to 348 square centimeters.
theorem surface_area_cuboid : 
  surface_area length breadth height = 348 := by
  sorry

end surface_area_cuboid_l566_566641


namespace tan_series_eq_sin_2x_tan_zero_or_kpi_l566_566923

theorem tan_series_eq_sin_2x (x : ℝ) (h : |tan x| < 1) :
  (8.407 * ((1 : ℝ) + tan x + (tan x)^2 + ... + (tan x)^n + ...) /
  ((1 : ℝ) - tan x + (tan x)^2 - ... + (-1)^n * (tan x)^n + ...)) = (1 + sin (2 * x)) :=
sorry

theorem tan_zero_or_kpi (x : ℝ) (hx : (8.407 * ((1 : ℝ) + tan x + (tan x)^2 + ... + (tan x)^n + ...) /
  ((1 : ℝ) - tan x + (tan x)^2 - ... + (-1)^n * (tan x)^n + ...)) = (1 + sin (2 * x)) ∧ |tan x| < 1) :
  ∃ k : ℤ, x = k * π :=
sorry

end tan_series_eq_sin_2x_tan_zero_or_kpi_l566_566923


namespace number_of_homologous_functions_l566_566391

/-- Definition of homologous functions --/
def homologous_function (f g : ℝ → ℝ) (range_f : set ℝ) : Prop :=
  (∀ x, f x = g x) ∧ (set.range f = range_f)

noncomputable def count_homologous_functions : ℕ :=
  {g : ℝ → ℝ // homologous_function (λ x, x^2) g {0, 1}}.finset.card

theorem number_of_homologous_functions :
  count_homologous_functions = 3 := by
  -- proof goes here
  sorry

end number_of_homologous_functions_l566_566391


namespace janet_hourly_wage_l566_566044

variable (x : ℝ) -- Janet's hourly wage

-- Conditions
def weekly_hours : ℝ := 52
def regular_hours : ℝ := 40
def weekly_overtime_hours : ℝ := weekly_hours - regular_hours
def overtime_rate : ℝ := 1.5 * x
def normal_weekly_pay : ℝ := regular_hours * x
def overtime_weekly_pay : ℝ := weekly_overtime_hours * overtime_rate
def total_weekly_pay : ℝ := normal_weekly_pay + overtime_weekly_pay 
def car_cost : ℝ := 4640
def weeks_to_work : ℝ := 4
def total_earnings : ℝ := weeks_to_work * total_weekly_pay

-- The proposition
theorem janet_hourly_wage : x = 20 :=
by
  have : 4 * (40 * x + 12 * (1.5 * x)) = 4640 := sorry -- Setup equation
  have : 4 * 58 * x = 4640 := sorry -- Simplify equation
  have : 232 * x = 4640 := sorry -- Further simplify
  have : x = 4640 / 232 := sorry -- Solve for x
  have : x = 20 := sorry -- Simplify division
  sorry

end janet_hourly_wage_l566_566044


namespace sum_f_values_l566_566131

-- Define the function f and the conditions
variable {f : ℝ → ℝ}
axiom symm_about_point : ∀ x, f(x) = -f(-x - (3/2))
axiom shift_property : ∀ x, f(x) = -f(x + (3/2))
axiom f_one : f(1) = 1
axiom f_zero : f(0) = -2

-- Define the theorem to prove
theorem sum_f_values : (∑ i in (finset.range 2009).map nat.cast ∘ list.to_finset, f (1 + i)) = 2 :=
by
  -- Proof goes here
  sorry

end sum_f_values_l566_566131


namespace equilateral_triangle_ellipse_ratio_l566_566250

noncomputable def AB (a b : ℝ) : ℝ := 3
noncomputable def F1F2 (a b : ℝ) : ℝ := 2 * Real.sqrt(5)

theorem equilateral_triangle_ellipse_ratio
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (h_triangle : ∃ B C, 
    B = (0, 2) ∧ 
    (∃ A, ABC_eq A B C ∧
      AC_parallel_x A C))
    :
  (AB a b) / (F1F2 a b) = 3 / (2 * Real.sqrt(5)) :=
by
  sorry

end equilateral_triangle_ellipse_ratio_l566_566250


namespace find_m_value_l566_566346

theorem find_m_value (m : ℤ) : 
  (∀ x : ℤ, (x + 2) * (x + 3) = x^2 + m * x + 6) → m = 5 := 
by
  intro h
  have h_eq : (x + 2) * (x + 3) = x^2 + 5 * x + 6 := 
    by ring_exp
  specialize h x
  rw h_eq at h
  simp at h
  exact_mod_cast h
  sorry

end find_m_value_l566_566346


namespace factory_output_equation_l566_566212

-- Define the average growth rate as a variable
variable (x : ℝ)

-- Define the output in January
def january_output := 500

-- Define the output in February
def february_output := january_output * (1 + x)

-- Define the output in March
def march_output := february_output * (1 + x)

-- Define the given condition for March output
def march_output_condition := 720

-- The main theorem stating the equation that represents the situation
theorem factory_output_equation : january_output * (1 + x) * (1 + x) = march_output_condition := by
  sorry

end factory_output_equation_l566_566212


namespace number_of_painted_cells_l566_566577

def grid_width : ℕ := 2000
def grid_height : ℕ := 70

def lcm (a b : ℕ) : ℕ := (a / Nat.gcd a b) * b

theorem number_of_painted_cells : lcm grid_width grid_height = 14000 := by
  -- Converting the LCM proof to Lean statement
  have gcd_calculation : Nat.gcd grid_width grid_height = 10 := by
    -- Verifying GCD calculation
    sorry
  rw [Nat.lcm_eq, gcd_calculation]
  norm_num

end number_of_painted_cells_l566_566577


namespace min_expression_value_l566_566352

theorem min_expression_value 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b = 1) :
  a = (sqrt 3 - 1) / 2 → 
  ∀ x y : ℝ, 
  x = (2*a^2 + 1) / (a * b) - 2 → 
  x ≥ 2 * sqrt 3 := 
sorry

end min_expression_value_l566_566352


namespace length_of_GH_l566_566762

theorem length_of_GH
  (AC BC : ℝ)
  (angle_ACB : AC > 0 ∧ BC > 0 ∧ angle_equality)
  (square_DEFG square_GHIJ : is_square)
  (point_E_on_AC point_I_on_BC : point_on_side)
  (mid_J : J_is_midpoint)
  : length of GH = 60 / 77 :=
by {
    let AB := sqrt (AC^2 + BC^2),
    have h1 : AB = 5 := by linarith,
    have h2 : GH = 60 / 77 := by linarith,
    exact h2,
}

-- Definitions and assumptions used
def angle_equality : Prop := ∠ ACB = 90
def is_square (q : quadrilateral) : Prop := q.has_four_equal_sides ∧ q.has_four_right_angles
def point_on_side (p : point, s : line_segment) : Prop := p ∈ s
def J_is_midpoint : Prop := J = midpoint(DG)

end length_of_GH_l566_566762


namespace combined_population_l566_566813

-- Defining the conditions
def population_New_England : ℕ := 2100000

def population_New_York (p_NE : ℕ) : ℕ := (2 / 3 : ℚ) * p_NE

-- The theorem to be proven
theorem combined_population (p_NE : ℕ) (h1 : p_NE = population_New_England) : 
  population_New_York p_NE + p_NE = 3500000 :=
by
  sorry

end combined_population_l566_566813


namespace range_of_m_l566_566341

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2m - 1}
def B : Set ℝ := {x | x^2 - 7 * x + 10 ≤ 0}

theorem range_of_m (m : ℝ) (hA : ∀ x, x ∈ A m → x ∈ B) : 2 ≤ m ∧ m ≤ 3 :=
by 
  sorry

end range_of_m_l566_566341


namespace triangle_area_relation_l566_566756

/-- In an acute-angled triangle \(\triangle ABC\), the areas of the orthic triangle,
tangential triangle, contact triangle, excentral triangle, and medial triangle satisfy the
relationship \(S_H \leq S_K = S_J \leq S_I \leq S_G\). Equality holds if and only if the
original triangle is an equilateral triangle. -/
theorem triangle_area_relation (ABC : Type*)
  [acute_angled_triangle ABC]
  (S_H S_K S_J S_I S_G : ℝ) :
  S_H ≤ S_K ∧ S_K = S_J ∧ S_J ≤ S_I ∧ S_I ≤ S_G ∧
  (S_H = S_K ↔ is_equilateral_triangle ABC) :=
sorry

end triangle_area_relation_l566_566756


namespace rectangles_covered_by_hexagons_is_50_percent_l566_566975

noncomputable def percentage_hexagons_cover (rect_width rect_height num_squares square_side_length : ℕ) : ℚ :=
  let total_area := rect_width * rect_height
  let square_area := num_squares * square_side_length^2
  let area_covered_by_hexagons := total_area - square_area
  (area_covered_by_hexagons * 100 : ℚ) / total_area

theorem rectangles_covered_by_hexagons_is_50_percent :
  (rectangle_width rectangle_height num_squares square_side_length : ℕ) :
  rectangle_width = 4 ∧ rectangle_height = 3 ∧ num_squares = 6 ∧ square_side_length = 1 →
  percentage_hexagons_cover rectangle_width rectangle_height num_squares square_side_length = 50 := 
by
  sorry

end rectangles_covered_by_hexagons_is_50_percent_l566_566975


namespace slytherin_postcards_l566_566167

theorem slytherin_postcards (n : ℕ) (h₁ : n = 30) (h₂ : ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ z ≠ x →
  ¬ (friend x y ∧ friend y z ∧ friend z x)) :
  (max_postcards n = 450) :=
sorry

variable {α : Type*} (friend : α → α → Prop)

variable (max_postcards : ℕ → ℕ)

def max_postcards := λ n, if n = 30 then 450 else 0  -- This is an auxiliary definition for 
                                                    -- demonstration. In real cases, it might
                                                    -- depend on actual computation.

end slytherin_postcards_l566_566167


namespace parallel_planes_of_parallel_skew_lines_l566_566596

-- Definitions for planes and skew lines (assumed as existing structures)
variables (α β : Type) [plane α] [plane β]
variables (m n : Type) [line m] [line n]

-- Skew lines do not intersect and are not parallel
def skew (l1 l2 : Type) [line l1] [line l2] : Prop :=
  ¬intersect l1 l2 ∧ ¬parallel l1 l2

axiom parallel (p1 p2 : Type) [plane p1] [plane p2] : Prop
axiom intersect (l1 l2 : Type) [line l1] : Prop

-- Define parallel property for lines and planes
axiom line_parallel_to_plane (l : Type) [line l] (p : Type) [plane p] : Prop

-- Main theorem statement
theorem parallel_planes_of_parallel_skew_lines (h_skew: skew m n)
  (h_mα: line_parallel_to_plane m α) (h_nα: line_parallel_to_plane n α)
  (h_mβ: line_parallel_to_plane m β) (h_nβ: line_parallel_to_plane n β) :
  parallel α β :=
  sorry

end parallel_planes_of_parallel_skew_lines_l566_566596


namespace best_choice_is_C_optionA_is_not_minimal_optionB_is_not_minimal_optionD_is_not_logic_l566_566659

variable (W B K N E R : ℕ)

-- Conditions
axiom duration_washing_brushing : W = 5
axiom duration_cleaning_kettle : B = 2
axiom duration_boiling_water : K = 8
axiom duration_soaking_noodles : N = 3
axiom duration_eating : E = 10
axiom duration_listening_radio : R = 8

-- Define total times for each option
def optionA_time : ℕ := W + B + K + N + E + R
def optionB_time : ℕ := B + K + N + E + R
def optionC_time : ℕ := B + K + N + E
def optionD_time : ℕ := E + R + N + K + B

theorem best_choice_is_C : optionC_time W B K N E R = 23 :=
by
  have hW : W = 5 := duration_washing_brushing
  have hB : B = 2 := duration_cleaning_kettle
  have hK : K = 8 := duration_boiling_water
  have hN : N = 3 := duration_soaking_noodles
  have hE : E = 10 := duration_eating
  have hR : R = 8 := duration_listening_radio
  rw [optionC_time, hW, hB, hK, hN, hE]
  norm_num
  done

theorem optionA_is_not_minimal : optionA_time W B K N E R ≠ 23 :=
sorry

theorem optionB_is_not_minimal : optionB_time W B K N E R ≠ 23 :=
sorry

theorem optionD_is_not_logic : ¬ (N < K) :=
sorry

end best_choice_is_C_optionA_is_not_minimal_optionB_is_not_minimal_optionD_is_not_logic_l566_566659


namespace travelers_on_liner_l566_566891

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566891


namespace vector_subtraction_l566_566940

def a : ℝ × ℝ := (3, 5)
def b : ℝ × ℝ := (-2, 1)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)

theorem vector_subtraction : (a.1 - two_b.1, a.2 - two_b.2) = (7, 3) := by
  sorry

end vector_subtraction_l566_566940


namespace students_passed_finals_l566_566879

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end students_passed_finals_l566_566879


namespace range_of_x_l566_566075

noncomputable def f (x : ℝ) : ℝ := real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x (x : ℝ) (h : 1/3 < x ∧ x < 1) : f x > f (2 * x - 1) :=
sorry

end range_of_x_l566_566075


namespace evaluate_f_f_4_l566_566363

def f (x : ℝ) : ℝ := 
  if x ≥ 0 then -2^x else Real.logb 4 (Real.abs x)

theorem evaluate_f_f_4 : f (f 4) = 2 := 
by sorry

end evaluate_f_f_4_l566_566363


namespace bob_grade_is_35_l566_566776

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l566_566776


namespace coefficient_c_nonzero_l566_566133

-- We are going to define the given polynomial and its conditions
def P (x : ℝ) (a b c d e : ℝ) : ℝ :=
  x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Given conditions
def five_x_intercepts (P : ℝ → ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧ P x5 = 0

def double_root_at_zero (P : ℝ → ℝ) : Prop :=
  P 0 = 0 ∧ deriv P 0 = 0

-- Equivalent proof problem
theorem coefficient_c_nonzero (a b c d e : ℝ)
  (h1 : P 0 a b c d e = 0)
  (h2 : deriv (P · a b c d e) 0 = 0)
  (h3 : ∀ x, P x a b c d e = x^2 * (x - 1) * (x - 2) * (x - 3))
  (h4 : ∀ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) : 
  c ≠ 0 := 
sorry

end coefficient_c_nonzero_l566_566133


namespace bus_speed_excluding_stoppages_l566_566636

theorem bus_speed_excluding_stoppages 
  (v_s : ℕ) -- Speed including stoppages in kmph
  (stop_duration_minutes : ℕ) -- Duration of stoppages in minutes per hour
  (stop_duration_fraction : ℚ := stop_duration_minutes / 60) -- Fraction of hour stopped
  (moving_fraction : ℚ := 1 - stop_duration_fraction) -- Fraction of hour moving
  (distance_per_hour : ℚ := v_s) -- Distance traveled per hour including stoppages
  (v : ℚ) -- Speed excluding stoppages
  
  (h1 : v_s = 50)
  (h2 : stop_duration_minutes = 10)
  
  -- Equation representing the total distance equals the distance traveled moving
  (h3 : v * moving_fraction = distance_per_hour)
: v = 60 := sorry

end bus_speed_excluding_stoppages_l566_566636


namespace tax_rate_correct_l566_566953

/-- The tax rate in dollars per $100.00 is $82.00, given that the tax rate as a percent is 82%. -/
theorem tax_rate_correct (x : ℝ) (h : x = 82) : (x / 100) * 100 = 82 :=
by
  rw [h]
  sorry

end tax_rate_correct_l566_566953


namespace problem_solution_l566_566014

-- Sequence and sum definitions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Given condition
def seq_condition := ∀ n : ℕ, a n + 2 * S n = 3

-- First statement: No three terms form an arithmetic sequence
def no_arithmetic_seq (h : seq_condition) : Prop :=
  ∀ (p q r : ℕ), p < q -> q < r -> ¬(2 * a q = a p + a r)

-- Second statement: No geometric sequence that satisfies the sum condition
def no_geometric_seq (h : seq_condition) : Prop :=
  ¬(∃ (m n k : ℕ), ∀ (x : ℕ), (m + n * x < k) →
      (\(S(k) = (a (m + n * k) - a m)/ 1-(a (n)- a n)) ∧ 
      (9/160 < S(k) ∧ S(k) < 1/13))

-- Main theorem combining both statements
theorem problem_solution (h : seq_condition) : no_arithmetic_seq h ∧ no_geometric_seq h := sorry

end problem_solution_l566_566014


namespace max_shelves_with_5_same_books_l566_566265

theorem max_shelves_with_5_same_books (k : ℕ) : k ≤ 18 → ∃ books : fin 1300 → fin k, 
  ∀ (rearranged_books : fin 1300 → fin k), 
    ∃ (shelf : fin k), (books '' {n | books n = shelf}).card ≥ 5 ∧ ∀ (n : fin 1300), rearranged_books n = books n → books n = shelf :=
begin
  sorry
end

end max_shelves_with_5_same_books_l566_566265


namespace find_point_P_l566_566715

/-- 
Given two points A and B, find the coordinates of point P that lies on the line AB
and satisfies that the distance from A to P is half the vector from A to B.
-/
theorem find_point_P 
  (A B : ℝ × ℝ) 
  (hA : A = (3, -4)) 
  (hB : B = (-9, 2)) 
  (P : ℝ × ℝ) 
  (hP : P.1 - A.1 = (1/2) * (B.1 - A.1) ∧ P.2 - A.2 = (1/2) * (B.2 - A.2)) : 
  P = (-3, -1) := 
sorry

end find_point_P_l566_566715


namespace ratio_of_square_areas_l566_566587

noncomputable def ratio_of_areas (s : ℝ) : ℝ := s^2 / (4 * s^2)

theorem ratio_of_square_areas (s : ℝ) (h : s ≠ 0) : ratio_of_areas s = 1 / 4 := 
by
  sorry

end ratio_of_square_areas_l566_566587


namespace tile_reduction_l566_566984

-- Define the operation that removes perfect square tiles and renumbers the rest
def removePerfectSquares (n : ℕ) : ℕ :=
  n - (List.range (n + 1)).countp (λ x => ∃ k, k * k = x)

-- Recursive function to apply the operation until only one tile remains
def countOperations (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else 1 + countOperations (removePerfectSquares n)

-- The main theorem stating that starting from 49 we need 3 operations to reduce to 1 tile
theorem tile_reduction : countOperations 49 = 3 := by
  sorry

end tile_reduction_l566_566984


namespace John_pays_first_year_l566_566047

theorem John_pays_first_year
  (num_family_members : ℕ)
  (joining_fee : ℕ)
  (monthly_fee : ℕ)
  (months_in_year : ℕ)
  (total_cost_people : ℕ)
  (johns_share : ℕ) :
  num_family_members = 4 →
  joining_fee = 4000 →
  monthly_fee = 1000 →
  months_in_year = 12 →
  (total_cost_people = (num_family_members * joining_fee) + 
                      (num_family_members * monthly_fee * months_in_year)) →
  johns_share = total_cost_people / 2 →
  johns_share = 32000 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h5
  have : total_cost_people = 4 * 4000 + 4 * 1000 * 12,
    simp [h5]
  rw this at h6
  simp [h6]
  sorry

end John_pays_first_year_l566_566047


namespace num_n_cool_perms_l566_566624

-- Definition of the sum of the first k natural numbers
def sum_first_n (k : ℕ) : ℕ := k * (k + 1) / 2

-- Definition of an n-cool permutation
def is_n_cool_perm (n : ℕ) (a : List ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → ∃ r, 0 ≤ r ∧ r ≤ n - k ∧ 
    sum_first_n k = (List.sum (List.drop r (List.take (r + k) a)))

-- The main theorem stating that the number of n-cool permutations is 2^(n-1)
theorem num_n_cool_perms (n : ℕ) : 
  (Finset.filter (is_n_cool_perm n) (Finset.univ : Finset (List.perm (List.range n.succ)))).card = 2^(n-1) :=
sorry

end num_n_cool_perms_l566_566624


namespace right_triangle_median_lines_num_valid_n_correct_l566_566043

noncomputable def num_valid_n : ℕ :=
  2

theorem right_triangle_median_lines (n : ℝ) :
  ∃ (a b : ℝ), 
    let Q := (a, b + 2 * (b+4*a+1- b) / 4)
        R := (a - 2 * (b+4*a+1-b) / (4*(b+4*a+1-b)), b) in
    (∃ P : ℝ × ℝ, P = (a, b) ∧ (Q.2 - P.2)/(Q.1 - P.1) = 4 ∧ (R.2 - P.2)/(R.1 - P.1) = n ∧ (n = 1 ∨ n = 16)) :=
begin
  sorry
end

theorem num_valid_n_correct : ∃ (n_set : set ℝ), 
  -- This set contains exactly the n values for which the conditions hold
  (∀ n ∈ n_set, right_triangle_median_lines n) ∧
  n_set.card = num_valid_n :=
begin
  sorry
end

end right_triangle_median_lines_num_valid_n_correct_l566_566043


namespace height_of_cuboid_l566_566157

-- Define the conditions for the problem
def width : ℝ := 7
def length : ℝ := 8
def surface_area : ℝ := 442

-- Define the height variable to be determined
def height := 11

-- Formulate the proof problem
theorem height_of_cuboid (h : ℝ) (A : ℝ) (l : ℝ) (w : ℝ) :
  w = width → l = length → A = 2 * l * w + 2 * l * h + 2 * w * h → A = surface_area → h = height := by
  intros hw hl HA precise_A
  -- Using the conditions to reason about h, the proof goes here
  sorry

end height_of_cuboid_l566_566157


namespace volume_Q3_l566_566675

structure TetrahedronSeq (V₀ : ℝ) :=
(vol : ℕ → ℝ)
(vol_0 : vol 0 = V₀)
(vol_succ : ∀ n, 
  vol (n + 1) = 
    vol n + 
    4 * (2/3)^(3 * (n + 1)) * vol 0 
      * (∏ i in finset.range (n + 1), 4 * (2/3)^(3 * i)))

theorem volume_Q3 : 
  (TetrahedronSeq.mk 2).vol 3 = 2028/243 := sorry

end volume_Q3_l566_566675


namespace martin_berry_expenditure_l566_566807

theorem martin_berry_expenditure : 
  (let daily_consumption := 1 / 2
       berry_price := 2
       days := 30 in
   daily_consumption * days * berry_price = 30) :=
by
  sorry

end martin_berry_expenditure_l566_566807


namespace custom_license_plates_count_l566_566384

theorem custom_license_plates_count : 
  let letters := 26
      digits := 10
      letter_or_digit := 36 in
  (26 * 1 * 36 * 10) + (26 * 36 * 1 * 10) + (26 * 36 * 1 * 1) = 19656 :=
by {
  let letters := 26;
  let digits := 10;
  let letter_or_digit := 36;
  have h1 : (26 * 1 * 36 * 10) = 9360 := by sorry,
  have h2 : (26 * 36 * 1 * 10) = 9360 := by sorry,
  have h3 : (26 * 36 * 1 * 1) = 936 := by sorry,
  calc
    (26 * 1 * 36 * 10) + (26 * 36 * 1 * 10) + (26 * 36 * 1 * 1)
    = 9360 + 9360 + 936 : by rw [h1, h2, h3]
    ... = 19656 : by norm_num
}

end custom_license_plates_count_l566_566384


namespace age_difference_l566_566195

variable (A B C : ℕ)

-- Conditions
def ages_total_condition (a b c : ℕ) : Prop :=
  a + b = b + c + 11

-- Proof problem statement
theorem age_difference (a b c : ℕ) (h : ages_total_condition a b c) : a - c = 11 :=
by
  sorry

end age_difference_l566_566195


namespace gcd_polynomial_l566_566354

theorem gcd_polynomial (b : ℤ) (k : ℤ) (hk : k % 2 = 1) (h_b : b = 1193 * k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := 
  sorry

end gcd_polynomial_l566_566354


namespace incenter_inside_equilateral_of_acute_l566_566156

variable {A B C X Y Z : Point}
variable {ABC XYZ : Triangle}
variable (I O : Point) -- I is incenter of ABC, O is incenter of XYZ

-- Definitions of being vertices lying on the sides of other triangles:
def on_side (P Q R : Point) (abc : Triangle) : Prop :=
  ∃ (P' Q' R' : Point), (P' ∈ Segment Q R) ∧ (Q' ∈ Segment R P) ∧ (R' ∈ Segment P Q)

-- Definitions of triangles being equilateral and acute-angled:
def is_equilateral (T : Triangle) : Prop :=
  let ⟨a, b, c⟩ := T in dist a b = dist b c ∧ dist b c = dist c a

def is_acute (T : Triangle) : Prop :=
  ∀ A B C, angle A B C < π / 2

-- The proof statement:
theorem incenter_inside_equilateral_of_acute (hXYZ: is_equilateral XYZ) (hABC: is_acute ABC)
  (hX : X ∈ Segment B C) (hY : Y ∈ Segment C A) (hZ : Z ∈ Segment A B)
  (hI : I = incenter ABC) (hO : O = incenter XYZ) :
  inside_triangle I XYZ :=
  sorry

end incenter_inside_equilateral_of_acute_l566_566156


namespace find_different_mass_part_l566_566822

-- Definitions for the parts a1, a2, a3, a4 and their masses
variable {α : Type}
variables (a₁ a₂ a₃ a₄ : α)
variable [LinearOrder α]

-- Definition of the problem conditions
def different_mass_part (a₁ a₂ a₃ a₄ : α) : Prop :=
  (a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₁ ≠ a₄ ∨ a₂ ≠ a₃ ∨ a₂ ≠ a₄ ∨ a₃ ≠ a₄)

-- Theorem statement assuming we can identify the differing part using two weighings on a pan balance
theorem find_different_mass_part (h : different_mass_part a₁ a₂ a₃ a₄) :
  ∃ (part : α), part = a₁ ∨ part = a₂ ∨ part = a₃ ∨ part = a₄ :=
sorry

end find_different_mass_part_l566_566822


namespace geometric_sequence_smallest_n_l566_566958

theorem geometric_sequence_smallest_n (x : ℝ) (h1 : x > 0) (h2 : ∀ n : ℕ, a_1 = Real.exp x ∧ a_2 = x ∧ a_3 = Real.log x)
    (h3 : ∀ n : ℕ, a_{n+1} = (x / (Real.exp x)) * a_n) :
  ∃ n : ℕ, a_n = 2 * x ∧ n = 8 := 
sorry

end geometric_sequence_smallest_n_l566_566958


namespace total_fish_weight_l566_566288

noncomputable def trout_count : ℕ := 4
noncomputable def catfish_count : ℕ := 3
noncomputable def bluegill_count : ℕ := 5

noncomputable def trout_weight : ℝ := 2
noncomputable def catfish_weight : ℝ := 1.5
noncomputable def bluegill_weight : ℝ := 2.5

theorem total_fish_weight :
  let total_trout_weight := trout_count * trout_weight,
      total_catfish_weight := catfish_count * catfish_weight,
      total_bluegill_weight := bluegill_count * bluegill_weight,
      total_weight := total_trout_weight + total_catfish_weight + total_bluegill_weight
  in total_weight = 25 := 
  by 
  sorry

end total_fish_weight_l566_566288


namespace sum_of_digits_l566_566737

theorem sum_of_digits (x y z w : ℕ) 
  (hxz : z + x = 10) 
  (hyz : y + z = 9) 
  (hxw : x + w = 9) 
  (hx_ne_hy : x ≠ y)
  (hx_ne_hz : x ≠ z)
  (hx_ne_hw : x ≠ w)
  (hy_ne_hz : y ≠ z)
  (hy_ne_hw : y ≠ w)
  (hz_ne_hw : z ≠ w) :
  x + y + z + w = 19 := by
  sorry

end sum_of_digits_l566_566737


namespace equation_solution_count_l566_566280

noncomputable def f (x : ℝ) : ℝ :=
  x^(-2) - (1/2)^x

theorem equation_solution_count : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) :=
sorry

end equation_solution_count_l566_566280


namespace geom_seq_sum_l566_566021

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l566_566021


namespace find_angle_B_find_side_b_l566_566334

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}
variable {dot_product_max : ℝ}

-- Conditions
def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin A + c * Real.sin C - b * Real.sin B = Real.sqrt 2 * a * Real.sin C

def vectors (m n : ℝ × ℝ) := 
  m = (Real.cos A, Real.cos (2 * A)) ∧ n = (12, -5)

def side_length_a (a : ℝ) := 
  a = 4

-- Questions and Proof Problems
theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : triangle_condition a b c A B C) : 
  B = π / 4 :=
sorry

theorem find_side_b (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (max_dot_product_condition : Real.cos A = 3 / 5) 
  (ha : side_length_a a) (hb : b = a * Real.sin B / Real.sin A) : 
  b = 5 * Real.sqrt 2 / 2 :=
sorry

end find_angle_B_find_side_b_l566_566334


namespace quadratic_res_cos_sum_square_l566_566801

open Real

def is_quadratic_residue_mod (p : ℕ) (a : ℕ) : Prop :=
  ∃ x : ℕ, (x^2) % p = a % p

def set_A (p : ℕ) (S : Set ℕ) : Set ℕ :=
  {a ∈ S | is_quadratic_residue_mod p a}

def set_B (p : ℕ) (S : Set ℕ) : Set ℕ :=
  {b ∈ S | ¬ is_quadratic_residue_mod p b}

theorem quadratic_res_cos_sum_square (n : ℕ) (hn : n > 0) (p : ℕ) (hp : p = 4 * n + 1) (hp_prime : nat.prime p) :
  let S := {x | x ∈ finset.range(p) ∧ x % 2 = 1} in
  (∑ a in set_A p S, cos (a * π / p))^2 + (∑ b in set_B p S, cos (b * π / p))^2 = 2 :=
by sorry

end quadratic_res_cos_sum_square_l566_566801


namespace medium_supermarkets_in_sample_l566_566569

-- Definitions of the conditions
def total_supermarkets : ℕ := 200 + 400 + 1400
def prop_medium_supermarkets : ℚ := 400 / total_supermarkets
def sample_size : ℕ := 100

-- Problem: Prove that the number of medium-sized supermarkets in the sample is 20.
theorem medium_supermarkets_in_sample : 
  (sample_size * prop_medium_supermarkets) = 20 :=
by
  sorry

end medium_supermarkets_in_sample_l566_566569


namespace year_with_same_calendar_1992_2024_l566_566874

/-
Problem: Prove that the year with the same calendar as 1992 is 2024 given that the year next to 1991 having the same calendar is 1992.
-/

/-- 
Theorem: The year with the same calendar as 1992 is 2024.
-/
theorem year_with_same_calendar_1992_2024 :
  ∃ y : ℕ, y = 2024 ∧ 
  (∀ x : ℕ, (x = 1992 → y = x + 28 - 4 → ∃ n : ℕ, x + n = y ∧ ∀ k ≥ 1, x + k ∉ {100, 200, 300, 400}) 
  → exists x_next : ℕ, x_next = 2020 → x = 1992 → y = 2024) :=
by
  sorry

end year_with_same_calendar_1992_2024_l566_566874


namespace max_value_sqrt_cubed_max_value_sqrt_cubed_achieved_l566_566428

theorem max_value_sqrt_cubed (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) :
  (∛(a * b * c) + ∛((2 - a) * (2 - b) * (2 - c))) ≤ 2 :=
sorry

theorem max_value_sqrt_cubed_achieved :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
  (∛(a * b * c) + ∛((2 - a) * (2 - b) * (2 - c)) = 2) :=
sorry

end max_value_sqrt_cubed_max_value_sqrt_cubed_achieved_l566_566428


namespace convergence_l566_566792

open Real

variable (f : ℝ → ℝ) (u : ℕ → ℝ)

-- Condition 1: f is a non-decreasing function
axiom non_decreasing : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)

-- Condition 2: f(y) - f(x) < y - x for all real x and y > x
axiom condition2 : ∀ x y : ℝ, x < y → f(y) - f(x) < y - x

-- Condition 3: Sequence condition u_{n+2} = f(u_{n+1}) - f(u_n)
axiom sequence_condition : ∀ n : ℕ, u (n + 2) = f (u (n + 1)) - f (u n)

-- Theorem to prove: For any ε > 0, there exists a positive integer N such that for all n ≥ N, |u(n)| < ε
theorem convergence (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ+, ∀ n : ℕ, n ≥ N → |u n| < ε := sorry

end convergence_l566_566792


namespace pentagon_increasing_arithmetic_sequences_l566_566625

theorem pentagon_increasing_arithmetic_sequences : 
  (∃ (angles : Fin 5 → ℕ), 
    (∃ d,  angles = λ i => 108 - 2 * d + d * i) ∧ 
    (∀ i, angles i < 120) ∧ 
    (angles 0 > 0) ∧ 
    (∀ i j : Fin 5, i < j → angles i < angles j)) 
    ↔ 
    (∃ (d : ℕ), 1 ≤ d ∧ d ≤ 5) := 
sorry

end pentagon_increasing_arithmetic_sequences_l566_566625


namespace prime_iff_exists_a_l566_566063

variables (k h : ℕ) (n : ℕ) (a : ℤ)

-- Let n = 2^k * h + 1
def n_def : ℕ := 2^k * h + 1

-- Assuming the conditions 0 ≤ h < 2^k and h is odd
def h_cond (h : ℕ) : Prop := 0 ≤ h ∧ h < 2^k ∧ h % 2 = 1

theorem prime_iff_exists_a (h_nonneg : 0 ≤ h) (h_lt : h < 2^k) (h_odd : h % 2 = 1) :
  nat.prime (2^k * h + 1) ↔ ∃ a : ℤ, (a^((2^k * h) / 2)) % (2^k * h + 1) = -1 :=
sorry

end prime_iff_exists_a_l566_566063


namespace total_apple_trees_is_800_l566_566242

variable (T P A : ℕ) -- Total number of trees, peach trees, and apple trees respectively
variable (samples_peach samples_apple : ℕ) -- Sampled peach trees and apple trees respectively
variable (sampled_percentage : ℕ) -- Percentage of total trees sampled

-- Given conditions
axiom H1 : sampled_percentage = 10
axiom H2 : samples_peach = 50
axiom H3 : samples_apple = 80

-- Theorem to prove the number of apple trees
theorem total_apple_trees_is_800 : A = 800 :=
by sorry

end total_apple_trees_is_800_l566_566242


namespace find_n_l566_566643

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : Float.cos n = Float.cos 812) : n = 88 :=
sorry

end find_n_l566_566643


namespace total_fish_catch_l566_566100

noncomputable def Johnny_fishes : ℕ := 8
noncomputable def Sony_fishes : ℕ := 4 * Johnny_fishes
noncomputable def total_fishes : ℕ := Sony_fishes + Johnny_fishes

theorem total_fish_catch : total_fishes = 40 := by
  sorry

end total_fish_catch_l566_566100


namespace jason_messages_l566_566420

theorem jason_messages :
  ∃ M : ℕ, (M + M / 2 + 150) / 5 = 96 ∧ M = 220 := by
  sorry

end jason_messages_l566_566420


namespace incorrect_option_C_l566_566130

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 3 + φ)

theorem incorrect_option_C (φ : ℝ) (h1 : |φ| < π / 2) : 
  ¬ (∀ x ∈ Ioo (-3 * π / 4) (-π / 4), strict_mono (f x φ)) :=
by
  sorry

end incorrect_option_C_l566_566130


namespace fraction_susan_can_eat_l566_566472

theorem fraction_susan_can_eat
  (v t n nf : ℕ)
  (h₁ : v = 6)
  (h₂ : n = 4)
  (h₃ : 1/3 * t = v)
  (h₄ : nf = v - n) :
  nf / t = 1 / 9 :=
sorry

end fraction_susan_can_eat_l566_566472


namespace number_before_star_is_five_l566_566010

theorem number_before_star_is_five (n : ℕ) (h1 : n % 72 = 0) (h2 : n % 10 = 0) (h3 : ∃ k, n = 400 + 10 * k) : (n / 10) % 10 = 5 :=
sorry

end number_before_star_is_five_l566_566010


namespace total_driving_time_is_40_l566_566962

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l566_566962


namespace tablet_battery_life_l566_566079

noncomputable def battery_life_remaining
  (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) : ℝ :=
  let no_use_consumption_rate := 1 / no_use_life
  let use_consumption_rate := 1 / use_life
  let no_use_time := total_on_time - use_time
  let total_battery_used := no_use_time * no_use_consumption_rate + use_time * use_consumption_rate
  let remaining_battery := 1 - total_battery_used
  remaining_battery / no_use_consumption_rate

theorem tablet_battery_life (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) :
  battery_life_remaining no_use_life use_life total_on_time use_time = 6 :=
by
  -- The proof will go here, we use sorry for now to skip the proof step.
  sorry

end tablet_battery_life_l566_566079


namespace area_expression_of_quadrilateral_l566_566755

-- Definitions corresponding to the conditions
def is_right_triangle (A B C : Point) : Prop :=
  ∃ a b c : ℝ, a ^ 2 + b ^ 2 = c ^ 2 ∧ a = dist A B ∧ b = dist B C ∧ c = dist A C

def quadrilateral (A B C D : Point) : Prop :=
  dist A B = 8 ∧ dist B C = 4 ∧ dist C D = 10 ∧ dist D A = 10 ∧
  ∃ E : Point, is_right_triangle D C A

-- The proof goal corresponding to the problem statement
theorem area_expression_of_quadrilateral (A B C D : Point)
  (h1 : quadrilateral A B C D) :
  ∃ a b c : ℝ, (a = 0 ∧ c = 0 ∧ b = 62) ∧ a + b + c = 62 :=
by
  sorry

end area_expression_of_quadrilateral_l566_566755


namespace find_value_of_2S_l566_566072

open BigOperators

theorem find_value_of_2S : 
  let f := λ m n, ∏ i in finset.range (n - m + 1), (1 - (1:ℚ) / (m + i))
  let S := ∑ m in finset.range (2008 - 2 + 1) + 2, f m 2008
  2 * S = 2007 :=
sorry

end find_value_of_2S_l566_566072


namespace unique_n_value_l566_566055

theorem unique_n_value (n : ℕ) (a b : ℕ) (a_1 a_2 a_3 : ℤ) (terms_sum : ℤ) (dec_expansion : (ℕ → ℤ)) :
  (n ≥ 2) →
  (n = a_1 + a_2 + a_3 + terms_sum) →
  (∃ (k : ℕ), dec_expansion k = if k = 1 then a_1 else if k = 2 then a_2 else if k = 3 then a_3 else _) →
  (n = 2^a * 5^b) →
  (a_1 = 1 ∧ a_2 = 2 ∧ a_3 = 5 ∧ terms_sum = 0) →
  (n = 8) :=
by sorry

end unique_n_value_l566_566055


namespace polygon_points_distance_l566_566972

open Real

theorem polygon_points_distance (S a : ℝ) (hS_pos : 0 < S) (ha_pos : 0 < a)
  (polygon : set (ℝ × ℝ)) (h_polygon_area : ∃ k, set.measure_theory.measure_of (polygon) k ∧ k = S)
  (h_polygon_in_square : ∀ p ∈ polygon, 0 ≤ prod.fst p ∧ prod.fst p ≤ a ∧ 0 ≤ prod.snd p ∧ prod.snd p ≤ a) :
  ∃ p1 p2 ∈ polygon, dist p1 p2 ≥ S / a :=
sorry

end polygon_points_distance_l566_566972


namespace product_value_l566_566314

noncomputable def product_of_integers (A B C D : ℕ) : ℕ :=
  A * B * C * D

theorem product_value :
  ∃ (A B C D : ℕ), A + B + C + D = 72 ∧ 
                    A + 2 = B - 2 ∧ 
                    A + 2 = C * 2 ∧ 
                    A + 2 = D / 2 ∧ 
                    product_of_integers A B C D = 64512 :=
by
  sorry

end product_value_l566_566314


namespace part_i_solution_set_part_ii_minimum_value_l566_566704

-- Part (I)
theorem part_i_solution_set :
  (∀ (x : ℝ), 1 = 1 ∧ 2 = 2 → |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) :=
by { sorry }

-- Part (II)
theorem part_ii_minimum_value (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 2 * a * b) :
  |x - a| + |x + b| ≥ 9 / 2 :=
by { sorry }

end part_i_solution_set_part_ii_minimum_value_l566_566704


namespace find_a_l566_566681

def point (ℝ) : Type :=
  ℝ × ℝ

def collinear (A B C : point ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a :
  ∃ a : ℝ, collinear (3, 2) (-2, a) (8, 12) ∧ a = -8 :=
by
  use -8
  apply And.intro
  all_goals { sorry }

end find_a_l566_566681


namespace correct_operation_result_l566_566087

-- Define the conditions
def original_number : ℤ := 231
def incorrect_result : ℤ := 13

-- Define the two incorrect operations and the intended corrections
def reverse_subtract : ℤ := incorrect_result + 20
def reverse_division : ℤ := reverse_subtract * 7

-- Define the intended operations
def intended_multiplication : ℤ := original_number * 7
def intended_addition : ℤ := intended_multiplication + 20

-- The theorem we need to prove
theorem correct_operation_result :
  original_number = reverse_division →
  intended_addition > 1100 :=
by
  intros h
  sorry

end correct_operation_result_l566_566087


namespace martins_spending_l566_566808

-- Define the conditions:
def dailyBerryConsumption : ℚ := 1 / 2
def costPerCup : ℚ := 2
def days : ℕ := 30

-- Define the main theorem:
theorem martins_spending : (dailyBerryConsumption * days * costPerCup) = 30 := by
  -- This is where the proof would go.
  sorry

end martins_spending_l566_566808


namespace valid_clay_millennium_problem_submission_l566_566542

def valid_submission (s : String) : Prop :=
  s = "Birch and Swinnerton-Dyer Conjecture" ∨
  s = "Hodge Conjecture" ∨
  s = "Navier-Stokes Existence and Smoothness" ∨
  s = "P vs NP Problem" ∨
  s = "Poincaré Conjecture" ∨
  s = "Riemann Hypothesis" ∨
  s = "Yang-Mills Existence and Mass Gap"

theorem valid_clay_millennium_problem_submission 
  (s : String) 
  (h : valid_submission s) : s ∈ {"Birch and Swinnerton-Dyer Conjecture", 
                                   "Hodge Conjecture", 
                                   "Navier-Stokes Existence and Smoothness", 
                                   "P vs NP Problem", 
                                   "Poincaré Conjecture", 
                                   "Riemann Hypothesis", 
                                   "Yang-Mills Existence and Mass Gap"} := 
by {
  -- Given that s is a valid submission, 
  -- it must be one of the seven problems.
  exact h,
}

#check valid_clay_millennium_problem_submission

end valid_clay_millennium_problem_submission_l566_566542


namespace problem_l566_566861

def table : Type := fin 3 → fin 3 → ℕ

def condition (t : table) : Prop :=
(t 0 0 = 1) ∧ (t 1 1 = 2) ∧
(∀ i, ∃ j, t i j = 1) ∧ 
(∀ i, ∃ j, t i j = 2) ∧ 
(∀ i, ∃ j, t i j = 3) ∧ 
(∀ j, ∃ i, t i j = 1) ∧ 
(∀ j, ∃ i, t i j = 2) ∧ 
(∀ j, ∃ i, t i j = 3)

variables (A B : ℕ)

def table_filled_with_A_B_assigned : table :=
λ i j, match i, j with
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 2
| 1, 1 => 2
| 1, 2 => A
| 2, 2 => B
| _, _ => sorry -- remaining entries are determined to fulfill the conditions
end

theorem problem : condition (table_filled_with_A_B_assigned A B) → A + B = 4 := by
sorry

end problem_l566_566861


namespace arithmetic_seq_75th_term_difference_l566_566248

theorem arithmetic_seq_75th_term_difference :
  ∃ (d : ℝ), 300 * (50 + d) = 15000 ∧ -30 / 299 ≤ d ∧ d ≤ 30 / 299 ∧
  let L := 50 - 225 * (30 / 299)
  let G := 50 + 225 * (30 / 299)
  G - L = 13500 / 299 := by
sorry

end arithmetic_seq_75th_term_difference_l566_566248


namespace initial_bird_count_l566_566768

theorem initial_bird_count (B : ℕ) (h₁ : B + 7 = 12) : B = 5 :=
by
  sorry

end initial_bird_count_l566_566768


namespace solve_system_l566_566640

-- Define the conditions
def condition1 (x y : ℚ) : Prop := 7 * x + 3 * y = -10
def condition2 (x y : ℚ) : Prop := 4 * x - 6 * y = -38

-- Define the ordered pair that solves the system
def solution_pair : ℚ × ℚ := (-(29 / 9), -(113 / 27))

-- State the theorem
theorem solve_system : 
  let (x, y) := solution_pair in 
  condition1 x y ∧ condition2 x y :=
by
  let (x, y) := solution_pair
  dsimp [solution_pair]
  split
  · dsimp [condition1, solution_pair] -- for the first condition
    sorry
  · dsimp [condition2, solution_pair] -- for the second condition
    sorry

end solve_system_l566_566640


namespace line_through_point_equidistant_l566_566239

open Real

structure Point where
  x : ℝ
  y : ℝ

def line_equation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def equidistant (p1 p2 : Point) (l : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := l
  let dist_from_p1 := abs (a * p1.x + b * p1.y + c) / sqrt (a^2 + b^2)
  let dist_from_p2 := abs (a * p2.x + b * p2.y + c) / sqrt (a^2 + b^2)
  dist_from_p1 = dist_from_p2

theorem line_through_point_equidistant (a b c : ℝ)
  (P : Point) (A : Point) (B : Point) :
  (P = ⟨1, 2⟩) →
  (A = ⟨2, 2⟩) →
  (B = ⟨4, -6⟩) →
  line_equation a b c P →
  equidistant A B (a, b, c) →
  (a = 2 ∧ b = 1 ∧ c = -4) :=
by
  sorry

end line_through_point_equidistant_l566_566239


namespace monotone_increasing_interval_l566_566369

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ) + sqrt 3 * cos (ω * x + φ)

variables (ω : ℝ) (φ : ℝ)
variables (ω_pos : ω > 0) (φ_bound : abs φ < π / 2)
variables (period_cond : ∀ x, f ω φ (x + π / ω) = f ω φ x)
variables (even_func : ∀ x, f ω φ (-x) = f ω φ x)

theorem monotone_increasing_interval :
  {x : ℝ | ∃ k : ℤ, k * π - π / 2 ≤ x ∧ x ≤ k * π} :=
sorry

end monotone_increasing_interval_l566_566369


namespace total_fish_weight_l566_566287

noncomputable def trout_count : ℕ := 4
noncomputable def catfish_count : ℕ := 3
noncomputable def bluegill_count : ℕ := 5

noncomputable def trout_weight : ℝ := 2
noncomputable def catfish_weight : ℝ := 1.5
noncomputable def bluegill_weight : ℝ := 2.5

theorem total_fish_weight :
  let total_trout_weight := trout_count * trout_weight,
      total_catfish_weight := catfish_count * catfish_weight,
      total_bluegill_weight := bluegill_count * bluegill_weight,
      total_weight := total_trout_weight + total_catfish_weight + total_bluegill_weight
  in total_weight = 25 := 
  by 
  sorry

end total_fish_weight_l566_566287


namespace greatest_a_for_integer_solutions_l566_566120

theorem greatest_a_for_integer_solutions :
  ∃ a : ℕ, 
    (∀ x : ℤ, x^2 + a * x = -21 → ∃ y : ℤ, y * (y + a) = -21) ∧ 
    ∀ b : ℕ, (∀ x : ℤ, x^2 + b * x = -21 → ∃ y : ℤ, y * (y + b) = -21) → b ≤ a :=
begin
  -- Proof goes here
  sorry
end

end greatest_a_for_integer_solutions_l566_566120


namespace digit_makes_multiple_of_six_l566_566473

theorem digit_makes_multiple_of_six :
  ∃ (digit : ℕ), digit ∈ {0, 2, 4, 6, 8} ∧ 85 * 10000 + 67 * 10 + digit % 2 = 0 ∧ (26 + digit) % 3 = 0 ∧ digit = 6 :=
by
  sorry

end digit_makes_multiple_of_six_l566_566473


namespace kamal_average_marks_l566_566050

theorem kamal_average_marks :
  (76 / 120) * 0.2 + 
  (60 / 110) * 0.25 + 
  (82 / 100) * 0.15 + 
  (67 / 90) * 0.2 + 
  (85 / 100) * 0.15 + 
  (78 / 95) * 0.05 = 0.70345 :=
by 
  sorry

end kamal_average_marks_l566_566050


namespace pythagorean_triple_3_4_hypotenuse_l566_566673

open Nat -- we work with natural numbers

theorem pythagorean_triple_3_4_hypotenuse :
  ∃ x : ℕ, 3^2 + 4^2 = x^2 ∧ x = 5 := by
  exists 5
  constructor
  · exact rfl
  · rfl

end pythagorean_triple_3_4_hypotenuse_l566_566673


namespace exists_line_l_l566_566713

-- Define a structure for pairwise skew lines
structure IsSkew (a b : Line) : Prop :=
(skew : ∀ (p : Point), ¬ (p ∈ a ∧ p ∈ b))

-- Define pairwise skew lines
variables (a b c : Line)
variable [IsSkew a b]
variable [IsSkew a c]
variable [IsSkew b c]

-- Define the planes alpha and beta
axiom exists_plane_alpha : ∃ α : Plane, a ⊆ α ∧ Parallel α c
axiom exists_plane_beta : ∃ β : Plane, b ⊆ β ∧ Parallel β c

-- Define the condition for the existence of the line l
theorem exists_line_l : (∃ α β : Plane, a ⊆ α ∧ Parallel α c ∧ b ⊆ β ∧ Parallel β c ∧ ∃ l : Line, l ∈ (α ∩ β) ∧ Parallel l c ∧ l ∈ a ∧ l ∈ b) → 
  (∃ l : Line, l ∈ (α ∩ β) ∧ Parallel l c ∧ (l ∈ a) ∧ (l ∈ b)) :=
by 
  sorry

end exists_line_l_l566_566713


namespace limit_calculation_l566_566323

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

theorem limit_calculation :
  (Real.exp (-1) * Real.exp 0 - Real.exp (-1) * Real.exp 0) / 0 = -3 / Real.exp 1 := by
  sorry

end limit_calculation_l566_566323


namespace positive_difference_l566_566594

noncomputable def alice_initial := 6000
noncomputable def charlie_initial := 8000
noncomputable def alice_rate := 0.05
noncomputable def charlie_rate := 0.045
noncomputable def years := 15

noncomputable def alice_balance := alice_initial * (1 + alice_rate) ^ years
noncomputable def charlie_balance := charlie_initial * (1 + charlie_rate) ^ years
noncomputable def difference := abs (charlie_balance - alice_balance)

theorem positive_difference : abs (charlie_balance - alice_balance) ≈ 2945 := 
by
  sorry

end positive_difference_l566_566594


namespace james_overall_average_speed_l566_566418

def overall_average_speed
    (time_cycled_min : ℝ) (speed_cycled_mph : ℝ)
    (time_jogged_min : ℝ) (speed_jogged_mph : ℝ) : ℝ :=
  let time_cycled_hrs := time_cycled_min / 60
  let distance_cycled := speed_cycled_mph * time_cycled_hrs
  let time_jogged_hrs := time_jogged_min / 60
  let distance_jogged := speed_jogged_mph * time_jogged_hrs
  let total_distance := distance_cycled + distance_jogged
  let total_time := time_cycled_hrs + time_jogged_hrs
  total_distance / total_time

theorem james_overall_average_speed :
  overall_average_speed 45 12 75 6 = 8.25 := by
  sorry

end james_overall_average_speed_l566_566418


namespace lambda_value_l566_566716

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem lambda_value 
  (a b : V) 
  (h_not_parallel : ¬ ∃ k : ℝ, a = k • b) 
  (h_parallel : ∃ μ : ℝ, (λ : ℝ) • a + b = μ • (a + 2 • b)) : λ = 1 / 2 :=
by sorry

end lambda_value_l566_566716


namespace max_unique_solution_l566_566279

theorem max_unique_solution (x y : ℕ) (m : ℕ) (h : 2005 * x + 2007 * y = m) : 
  m = 2 * 2005 * 2007 ↔ ∃! (x y : ℕ), 2005 * x + 2007 * y = m :=
sorry

end max_unique_solution_l566_566279


namespace maximum_value_attains_maximum_value_l566_566434

theorem maximum_value
  (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c = 1) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 / 2 :=
sorry

theorem attains_maximum_value :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) = 1 / 2 :=
sorry

end maximum_value_attains_maximum_value_l566_566434


namespace surface_area_circumscribed_sphere_l566_566333

-- Define the conditions as Lean definitions
variables (A B C D : Type) 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AD BC CD : ℝ)

-- Given conditions
axiom h1 : AD = 2
axiom h2 : BD = 4
axiom h3 : ⊥  ∠(D, A, B, C)  -- AD is perpendicular to plane BCD
axiom h4 : ⊥ ∠(B, C, D)      -- BC is perpendicular to CD

-- Formulate the mathematical problem 
theorem surface_area_circumscribed_sphere : 
    ∃ (s : Sphere ABCD), s.surface_area = 20 * π := sorry

end surface_area_circumscribed_sphere_l566_566333


namespace dog_catches_rabbit_l566_566974

-- Defining the conditions stated in problem part a)
def rabbit_start_gap := 100
def rabbit_steps_to_dog_steps_distance := 8 / 3
def dog_steps_to_rabbit_steps_time := 4 / 9

-- The statement we need to prove
theorem dog_catches_rabbit :
  -- condition 1: initial gap
  rabbit_start_gap = 100 →
  -- condition 2: distance proportion
  rabbit_steps_to_dog_steps_distance = 8 / 3 →
  -- condition 3: time proportion
  dog_steps_to_rabbit_steps_time = 4 / 9 →
  -- conclusion: dog needs to run at least 240 steps
  True := 
begin
  sorry
end

end dog_catches_rabbit_l566_566974


namespace average_age_increase_l566_566160

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end average_age_increase_l566_566160


namespace problem_statement_l566_566862

open Nat

noncomputable def binomial_sum_mod (n : ℕ) : ℕ :=
  (Nat.sum (Finset.range 65) (λ k, Nat.choose 2024 k)) % n

theorem problem_statement : prime 2027 ∧ binomial_sum_mod 2027 = 1089 :=
by
  -- We assert the divisibility and proceed.
  constructor
  · exact prime_def 2027 sorry -- assuming primality of 2027
  · exact sorry -- proof of the sum.

end problem_statement_l566_566862


namespace dvd_cd_ratio_l566_566016

theorem dvd_cd_ratio (total_sales : ℕ) (dvd_sales : ℕ) (cd_sales : ℕ) (h1 : total_sales = 273) (h2 : dvd_sales = 168) (h3 : cd_sales = total_sales - dvd_sales) : (dvd_sales / Nat.gcd dvd_sales cd_sales) = 8 ∧ (cd_sales / Nat.gcd dvd_sales cd_sales) = 5 :=
by
  sorry

end dvd_cd_ratio_l566_566016


namespace cruise_liner_travelers_l566_566884

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l566_566884


namespace shifted_parabola_expression_l566_566105

theorem shifted_parabola_expression (x : ℝ) :
  let y_original := x^2
  let y_shifted_right := (x - 1)^2
  let y_shifted_up := y_shifted_right + 2
  y_shifted_up = (x - 1)^2 + 2 :=
by
  sorry

end shifted_parabola_expression_l566_566105


namespace joan_savings_l566_566046

def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25

theorem joan_savings : quarters_to_cents 6 = 150 :=
by
  rw [quarters_to_cents]
  norm_num
  sorry

end joan_savings_l566_566046


namespace teachers_by_car_l566_566029

noncomputable def total_teachers (teachers_by_bicycle : ℕ) (percent_by_bicycle : ℝ) : ℝ :=
  teachers_by_bicycle / percent_by_bicycle

theorem teachers_by_car 
  (teachers_by_bicycle : ℕ) 
  (percent_by_bicycle percent_by_car : ℝ) 
  (hb : percent_by_bicycle = 0.60) 
  (hc : percent_by_car = 0.12) 
  (h : teachers_by_bicycle = 45) : 
  (0.12 * total_teachers teachers_by_bicycle 0.60).toNat = 9 :=
by
  sorry

end teachers_by_car_l566_566029


namespace employee_weekly_earnings_l566_566249

theorem employee_weekly_earnings :
  let 
    task_A_hours_day1_3 := 6 * 3, -- 6 hours/day for 3 days
    task_A_hours_day4_5 := 6 * 2 * 2, -- 12 hours/day for 2 days
    total_task_A_hours := task_A_hours_day1_3 + task_A_hours_day4_5, -- total hours in Task A

    task_B_hours_day1_3 := 4 * 3, -- 4 hours/day for 3 days
    task_B_hours_day4_5 := 3 * 2, -- 3 hours/day for 2 days
    total_task_B_hours := task_B_hours_day1_3 + task_B_hours_day4_5, -- total hours in Task B

    rate_task_A := 30,
    overtime_rate_task_A := 1.5 * rate_task_A,
    base_hours_task_A := 40,

    earnings_task_A_base := min total_task_A_hours base_hours_task_A * rate_task_A,
    earnings_task_A_overtime := max (total_task_A_hours - base_hours_task_A) 0 * overtime_rate_task_A,

    total_earnings_task_A := earnings_task_A_base + earnings_task_A_overtime,

    rate_task_B := 40,
    total_earnings_task_B := total_task_B_hours * rate_task_B,

    total_earnings_before_commission := total_earnings_task_A + total_earnings_task_B,

    commission_rate := if total_task_B_hours >= 10 then 0.10 else 0,
    commission := commission_rate * total_earnings_before_commission,

    total_earnings := total_earnings_before_commission + commission
  in total_earnings = 2211 := sorry

end employee_weekly_earnings_l566_566249


namespace find_angle_B_l566_566688

open_locale real

variables {α : Type*} [inner_product_space ℝ α]

def is_orthocenter {A B C H : α} : Prop := 
∃ (AH BH CH : α), 
  AH = A - H ∧ 
  BH = B - H ∧ 
  CH = C - H ∧ 
  orthocenter_condition H A B C

noncomputable def orthocenter_condition (H A B C : α) : Prop := 
(A - H) + 2 • (B - H) + 6 • (C - H) = 0

theorem find_angle_B {A B C H : α} (h₁ : is_orthocenter A B C H) 
  (h₂ : orthocenter_condition H A B C) : 
  ∃ (angle_B : ℝ), angle_B = π / 3 :=
sorry

end find_angle_B_l566_566688


namespace part1_solution_set_part2_range_of_a_l566_566671

def f (x a : ℝ) := abs (2*x + 4) + x - 2*a + a^2

theorem part1_solution_set (x : ℝ) : 
  let a := 2 in f x a ≥ 6 ↔ x ∈ Set.Iic (-10) ∪ Set.Ici (2 / 3) := sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 10 - abs (2 - x)) ↔ a ∈ Set.Iic (-2) ∪ Set.Ici 4 := sorry

end part1_solution_set_part2_range_of_a_l566_566671


namespace largest_angle_of_pentagon_l566_566137

theorem largest_angle_of_pentagon (a b c d e : ℝ) (h : a / b = 3 / 3 ∧ b / c = 3 / 3 ∧ c / d = 3 / 4 ∧ d / e = 4 / 5) (sum_angles : a + b + c + d + e = 540) : 
  ∃ largest_angle, largest_angle = (5 / (3 + 3 + 3 + 4 + 5)) * 540 := 
by
  -- let x be the common factor
  let x := 540 / 18,
  -- the largest angle is 5x
  exact 
    have : a = 3 * x ∧ b = 3 * x ∧ c = 3 * x ∧ d = 4 * x ∧ e = 5 * x,
    from sorry,
    ⟨5 * x, by
      calculate
      have h : x = 30 := by {
        field_proof := sorry,
        let x' := 30,
        exact x'
      },
      let largest_angle := 5 * x,
      show largest_angle = 150 := by
        let largest_angle' := 150,
        exact largest_angle'⟩

end largest_angle_of_pentagon_l566_566137


namespace problem_1a_problem_1b_problem_2_problem_3_l566_566761

-- Define the division of powers with the same base
def div_powers_same_base (a : ℚ) (m n : ℕ) : ℚ :=
if m > n then a^(m-n) else if m = n then 1 else (a^(n-m))⁻¹

-- Problem 1(a): Prove (\frac{1}{3})^4 \div (\frac{1}{3})^2 = \frac{1}{9}
theorem problem_1a : div_powers_same_base (1/3) 4 2 = 1/9 := by
  sorry

-- Problem 1(b): Prove 2^3 \div 2^7 = \frac{1}{16}
theorem problem_1b : div_powers_same_base 2 3 7 = 1/16 := by
  sorry

-- Problem 2: Prove x = \frac{1}{3} given 3 \div 3^{3x + 4} = \frac{1}{81}
theorem problem_2 (x : ℚ) (h : 3 / 3^(3 * x + 4) = 1 / 81) : x = 1/3 := by
  sorry

-- Problem 3: Prove the possible values of x are 4, 2, 3 given (5 - 2x)^{3x - 1} \div (5 - 2x)^{x + 7} = 1
theorem problem_3 (x : ℚ) (h : (5 - 2 * x)^(3 * x - 1) / (5 - 2 * x)^(x + 7) = 1) :
  x = 4 ∨ x = 2 ∨ x = 3 := by
  sorry

end problem_1a_problem_1b_problem_2_problem_3_l566_566761


namespace problem_part_I_problem_part_II_l566_566415

-- Define the arithmetic sequence and common difference
def arithmetic_seq (n : ℕ) (a₁ : ℤ) (d : ℤ) : ℤ := a₁ + (n - 1) * d

-- Given conditions in the problem
def a₁ : ℤ := 3
def d : ℤ := 2
def geometric_seq (x y z : ℤ) : Prop := y * y = x * z

-- Summation for arithmetic series
def S (n : ℕ) : ℤ := n * (n + 2)

theorem problem_part_I (n : ℕ) : arithmetic_seq n a₁ d = 2 * n + 1 :=
by {
  sorry
}

theorem problem_part_II (n : ℕ) :
  (finset.range n).sum (λ k, 1 / S (k + 1)) = 3 / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2)) :=
by {
  sorry
}

end problem_part_I_problem_part_II_l566_566415


namespace correct_statement_about_parabola_l566_566655

theorem correct_statement_about_parabola (x : ℝ) : 
  let y := -2 * (x - 1)^2 + 3 in
  (∀ x, y = -2 * (x - 1)^2 + 3 → ∃ S : Prop, S = "The axis of symmetry is the line x = 1") :=

sorry

end correct_statement_about_parabola_l566_566655


namespace equilibrium_concentrations_and_mass_correct_l566_566274

noncomputable def equilibrium_concentrations_and_mass (initial_BaCl2 : ℝ) (Kc : ℝ) (initial_volume : ℝ) : ℝ × ℝ × ℝ × ℝ :=
let x := 10 - (1 / Kc) in
let [BaCl2_eq, BaSO4_eq, NaCl_eq] := [10 - x, x, 2 * x] in
let molar_mass_BaSO4 := 137.33 + 32.07 + 64.00 in
let mass_BaSO4 := x * molar_mass_BaSO4 in
(BaCl2_eq, BaSO4_eq, NaCl_eq, mass_BaSO4)

theorem equilibrium_concentrations_and_mass_correct :
    equilibrium_concentrations_and_mass 10 (5 * 10^6) 1 = (0, 10, 20, 2334) :=
by
  sorry

end equilibrium_concentrations_and_mass_correct_l566_566274


namespace population_growth_patterns_correct_l566_566540

-- Definitions based on conditions
def europe_has_low_growth (european_countries: Set Country) (growth_rate: Country → Real) : Prop :=
  ∀ c ∈ european_countries, growth_rate c < 3

def some_african_countries_high_growth (african_countries: Set Country) (growth_rate: Country → Real) : Prop :=
  ∃ c ∈ african_countries, growth_rate c ≥ 3

def us_has_positive_growth (us: Country) (growth_rate: Country → Real) : Prop :=
  growth_rate us > 0

def mortality_due_to_aging (developed_countries: Set Country) (mortality_rate: Country → Real) : Prop :=
  ∀ c ∈ developed_countries, high_mortality_rate c ↔ aging_population_structure c

-- Theorem statement combining conditions and correct answer
theorem population_growth_patterns_correct :
  (∀ (european_countries african_countries: Set Country) (us: Country) (developed_countries: Set Country)
   (growth_rate: Country → Real) (high_mortality_rate aging_population_structure: Country → Prop),
    europe_has_low_growth european_countries growth_rate →
    some_african_countries_high_growth african_countries growth_rate →
    us_has_positive_growth us growth_rate →
    mortality_due_to_aging developed_countries mortality_rate aging_population_structure →
    (∃ b : Bool, b = true)) :=
sorry

end population_growth_patterns_correct_l566_566540


namespace price_of_basic_computer_l566_566561

-- Definitions for the prices
variables (C_b P M K C_e : ℝ)

-- Conditions
axiom h1 : C_b + P + M + K = 2500
axiom h2 : C_e + P + M + K = 3100
axiom h3 : P = (3100 / 6)
axiom h4 : M = (3100 / 5)
axiom h5 : K = (3100 / 8)
axiom h6 : C_e = C_b + 600

-- Theorem stating the price of the basic computer
theorem price_of_basic_computer : C_b = 975.83 :=
by {
  sorry
}

end price_of_basic_computer_l566_566561


namespace equation_of_line_passing_origin_l566_566576

def line_through_origin_eq (k : ℝ) : Prop := ∀ x : ℝ, y = k * x

def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

def chord_length (L : ℝ → ℝ) (length : ℝ) : Prop := 
  ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ 
  (circle_eq P.1 P.2) ∧ (circle_eq Q.1 Q.2) ∧ 
  (P.2 = L P.1) ∧ (Q.2 = L Q.1) ∧ 
  (dist P Q = length)

theorem equation_of_line_passing_origin (L : ℝ → ℝ) :
  (∀ x, L x = 2 * x) →
  chord_length L 2 →
  ∃ k, ∀ x, L x = k * x :=
sorry

end equation_of_line_passing_origin_l566_566576


namespace total_distance_correct_l566_566782

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end total_distance_correct_l566_566782


namespace find_BC_length_l566_566689

noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  1/2 * A * B * C

theorem find_BC_length (A B C : ℝ) (angleA : ℝ)
  (h1 : area_triangle 5 A (Real.sin (π / 6)) = 5 * Real.sqrt 3)
  (h2 : B = 5)
  (h3 : angleA = π / 6) :
  C = Real.sqrt 13 :=
by
  sorry

end find_BC_length_l566_566689


namespace stripe_area_l566_566220

theorem stripe_area :
  ∀ (diameter height width revolutions : ℝ), 
    diameter = 40 → 
    height = 100 → 
    width = 4 → 
    revolutions = 3 → 
    let circumference := Real.pi * diameter in
    let total_length := circumference * revolutions in
    let area := width * total_length in
    area = 480 * Real.pi :=
by
  intros diameter height width revolutions h1 h2 h3 h4 circumference total_length area
  sorry

end stripe_area_l566_566220


namespace supplement_of_double_complement_l566_566528

def angle : ℝ := 30

def complement (θ : ℝ) : ℝ :=
  90 - θ

def double_complement (θ : ℝ) : ℝ :=
  2 * (complement θ)

def supplement (θ : ℝ) : ℝ :=
  180 - θ

theorem supplement_of_double_complement (θ : ℝ) (h : θ = angle) : supplement (double_complement θ) = 60 :=
by
  sorry

end supplement_of_double_complement_l566_566528


namespace angle_double_complement_l566_566179

theorem angle_double_complement (x : ℝ) (h₁ : x + (90 - x) = 90) (h₂ : x = 2 * (90 - x)) : x = 60 := 
begin
  sorry
end

end angle_double_complement_l566_566179


namespace abs_sum_of_roots_l566_566652

theorem abs_sum_of_roots 
  (a b c m : ℤ) 
  (h1 : a + b + c = 0)
  (h2 : ab + bc + ca = -2023)
  : |a| + |b| + |c| = 102 := 
sorry

end abs_sum_of_roots_l566_566652


namespace sum_interior_angles_polygon_l566_566114

theorem sum_interior_angles_polygon (n : ℕ) (h : 180 * (n - 2) = 1440) :
  180 * ((n + 3) - 2) = 1980 := by
  sorry

end sum_interior_angles_polygon_l566_566114


namespace solve_A_range_f_l566_566353

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (π / 3)) * (Real.cos x) ^ 2 - (Real.sin (π / 6)) ^ 2 * (Real.sin (2 * x))

theorem solve_A {a b : ℝ} (A B : ℝ) (h : b * (Real.sin A) ^ 2 = sqrt 3 * a * (Real.cos A) * (Real.sin B)) :
  A = π / 3 := by
  sorry

theorem range_f {x : ℝ} (hx : 0 ≤ x ∧ x ≤ π / 2) :
  (sqrt 3 - 2) / 4 ≤ f x ∧ f x ≤ sqrt 3 / 2 := by
  sorry

end solve_A_range_f_l566_566353


namespace problem_solution_l566_566083

open Nat

def is_valid_condition1 (x y z : Nat) : Prop :=
  100 <= 100*x + 10*y + z ∧ 100*x + 10*y + z < 1000 ∧ (100*x + 10*y + z) % 5 = 0 ∧ x > y ∧ y > z

def is_valid_condition2 (x y z : Nat) : Prop :=
  100 <= 100*x + 10*y + z ∧ 100*x + 10*y + z < 1000 ∧ (100*x + 10*y + z) % 5 = 0 ∧ x < y ∧ y < z

def A : Nat :=
  (∑ k in Finset.range 9, Finset.range k).card

def B : Nat :=
  (∑ k in Finset.range 4, Finset.range k).card

theorem problem_solution : ((A > B) ∧ ¬(B > 10) ∧ ¬(A + B > 100) ∧ ¬(A < 10) ∧ ¬(¬(A > B) ∧ ¬(B > 10) ∧ ¬(A + B > 100) ∧ ¬(A < 10))) := by
    have h_correctA : A = 42 := sorry
    have h_correctB : B = 6 := sorry
    sorry

end problem_solution_l566_566083


namespace part1_part2_l566_566677

section Problem

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom pos_terms {n : ℕ} : 0 < a n
axiom sum_condition {n : ℕ} : 2 * S n = a n + (1 / a n)

-- Question 1: Prove that {S_n^2} forms an arithmetic progression
theorem part1 (n : ℕ) : (S 1) ^ 2 = 1 ∧ (∀ n ≥ 1, (S (n + 1)) ^ 2 - (S n) ^ 2 = 1) :=
sorry

-- Sequence bn
variable (b : ℕ → ℝ)
axiom b1 {n : ℕ} : b 1 = 1
axiom bn_condition {n : ℕ} : b n / (b (n + 1) - b n) = S n ^ 2 - (1/2)

-- Question 2: Prove the sum of the first n terms of the given sequence
theorem part2 (n : ℕ) : 
  ∑ i in finset.range n, (-1)^(i + 1) * (4 * (S i) ^ 2 / (b i * b (i + 1))) = -1 + ((-1) ^ n) / (2 * n + 1) :=
sorry

end Problem

end part1_part2_l566_566677


namespace age_difference_l566_566501

theorem age_difference (sum_ages : ℕ) (eldest_age : ℕ) (age_diff : ℕ) 
(h1 : sum_ages = 50) (h2 : eldest_age = 14) :
  14 + (14 - age_diff) + (14 - 2 * age_diff) + (14 - 3 * age_diff) + (14 - 4 * age_diff) = 50 → age_diff = 2 := 
by
  intro h
  sorry

end age_difference_l566_566501


namespace train_speed_l566_566545

theorem train_speed (L V : ℝ) (h1 : L = V * 10) (h2 : L + 500 = V * 35) : V = 20 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end train_speed_l566_566545


namespace product_of_squares_l566_566140

theorem product_of_squares (a_1 a_2 a_3 b_1 b_2 b_3 : ℕ) (N : ℕ) (h1 : (a_1 * b_1)^2 = N) (h2 : (a_2 * b_2)^2 = N) (h3 : (a_3 * b_3)^2 = N) 
: (a_1^2 * b_1^2) = 36 ∨  (a_2^2 * b_2^2) = 36 ∨ (a_3^2 * b_3^2) = 36:= 
sorry

end product_of_squares_l566_566140


namespace three_digit_numbers_divisible_by_5_l566_566086

theorem three_digit_numbers_divisible_by_5 (A B : ℕ) :
  (A = (∑ k in range 1 (8+1), k + ∑ k in range 1 (3+1), k)) →
  (B = (∑ k in range 1 (3+1), k)) →
  A > B :=
by {
  -- A is calculated by first sum and B by second sum for clarity
  intro hA,
  intro hB,
  rw [hA, hB],
  sorry -- Proof steps are omitted as per the requirements
}

end three_digit_numbers_divisible_by_5_l566_566086


namespace cube_assignment_exists_l566_566928

def face1 := {A, B, C, D}
def face2 := {E, F, G, H}
def face3 := {A, B, E, F}
def face4 := {B, C, F, G}
def face5 := {C, D, G, H}
def face6 := {D, A, H, E}
def cube_faces : list (set ℕ) := [face1, face2, face3, face4, face5, face6]

theorem cube_assignment_exists (A B C D E F G H : int) :
  (A ∈ {1, -1}) ∧ (B ∈ {1, -1}) ∧ (C ∈ {1, -1}) ∧ (D ∈ {1, -1}) ∧
  (E ∈ {1, -1}) ∧ (F ∈ {1, -1}) ∧ (G ∈ {1, -1}) ∧ (H ∈ {1, -1}) ∧
  ∀ face ∈ cube_faces, ∏ v in face, v = -1 := sorry

end cube_assignment_exists_l566_566928


namespace find_set_of_x_l566_566730

noncomputable def exponential_inequality_solution (x : ℝ) : Prop :=
  1 < Real.exp x ∧ Real.exp x < 2

theorem find_set_of_x (x : ℝ) :
  exponential_inequality_solution x ↔ 0 < x ∧ x < Real.log 2 :=
by
  sorry

end find_set_of_x_l566_566730


namespace greatest_possible_value_of_a_l566_566122

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l566_566122


namespace probability_conditional_l566_566465

def event_A (dice1 dice2 : ℕ) : Prop := dice1 ≠ dice2
def event_B (dice1 dice2 : ℕ) : Prop := dice1 = 5 ∨ dice2 = 5

theorem probability_conditional :
  (∃ (A_prob B_prob : ℕ), 
    A_prob = 10 ∧ B_prob = 11 ∧ 
    ∀ (P_conditional : ℝ), 
      P_conditional = A_prob / B_prob → P_conditional = 10 / 11) :=
begin
  use 10,
  use 11,
  sorry,
end

end probability_conditional_l566_566465


namespace find_selling_price_l566_566572

def cost_price : ℝ := 900
def gain_percentage : ℝ := 40
def profit (cp gp : ℝ) : ℝ := (gp / 100) * cp
def selling_price (cp pr : ℝ) : ℝ := cp + pr

theorem find_selling_price : 
  selling_price cost_price (profit cost_price gain_percentage) = 1260 := by
  sorry

end find_selling_price_l566_566572


namespace line_equation_through_point_and_area_l566_566965

theorem line_equation_through_point_and_area (k b : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4/3, 2)) ∧
  (∀ (A B : ℝ × ℝ), A = (- b / k, 0) ∧ B = (0, b) → 
  1 / 2 * abs ((- b / k) * b) = 6) →
  (y = k * x + b ↔ (y = -3/4 * x + 3 ∨ y = -3 * x + 6)) :=
by
  sorry

end line_equation_through_point_and_area_l566_566965


namespace correct_statement_about_parabola_l566_566654

theorem correct_statement_about_parabola (x : ℝ) : 
  let y := -2 * (x - 1)^2 + 3 in
  (∀ x, y = -2 * (x - 1)^2 + 3 → ∃ S : Prop, S = "The axis of symmetry is the line x = 1") :=

sorry

end correct_statement_about_parabola_l566_566654


namespace value_of_a_b_l566_566001

-- Define the variables and conditions
variables (a b : ℝ) (i : ℂ)
axiom imaginary_unit : i = complex.I
axiom condition1 : b + (a - 2) * i = 1 + i

-- State the theorem
theorem value_of_a_b : a + b = 4 :=
by sorry

end value_of_a_b_l566_566001


namespace website_generates_suggestions_l566_566266

def website_suggestions (x : ℕ) : Prop :=
  4 * x + 5 = 65

theorem website_generates_suggestions :
  ∃ x : ℕ, website_suggestions x ∧ x = 15 :=
by
  use 15
  split
  · exact rfl   -- This confirms 4 * 15 + 5 = 65
  · rfl        -- This confirms x = 15
  sorry        -- Final proof step to be completed

end website_generates_suggestions_l566_566266


namespace gray_circle_product_l566_566969

theorem gray_circle_product :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ),
  let P := (x1 * x2 * x3 * x4 : ℕ),
      Q := (x2 * x3 * x5 * x6 : ℕ),
      R := (x3 * x4 * x7 * x8 : ℕ),
      S := (x1 * x5 * x7 * x9 : ℕ),
      T := (x2 * x6 * x8 * x9 : ℕ) in
  let grey_product := (P * Q * S * T) / (R * R) in
  P = 10 → Q = 4 → R = 12 → S = 6 → T = 24 → grey_product = 40 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 x8 x9 P Q R S T grey_product hP hQ hR hS hT
  rw [hP, hQ, hR, hS, hT]
  sorry

end gray_circle_product_l566_566969


namespace find_radius_yz_l566_566758

-- Define the setup for the centers of the circles and their radii
def circle_with_center (c : Type*) (radius : ℝ) : Prop := sorry
def tangent_to (c₁ c₂ : Type*) : Prop := sorry

-- Given conditions
variable (O X Y Z : Type*)
variable (r : ℝ)
variable (Xe_radius : circle_with_center X 1)
variable (O_radius : circle_with_center O 2)
variable (XtangentO : tangent_to X O)
variable (YtangentO : tangent_to Y O)
variable (YtangentX : tangent_to Y X)
variable (YtangentZ : tangent_to Y Z)
variable (ZtangentO : tangent_to Z O)
variable (ZtangentX : tangent_to Z X)
variable (ZtangentY : tangent_to Z Y)

-- The theorem to prove
theorem find_radius_yz :
  r = 8 / 9 := sorry

end find_radius_yz_l566_566758


namespace find_number_l566_566924

def sum := 555 + 445
def difference := 555 - 445
def quotient := 2 * difference
def remainder := 30
def N : ℕ := 220030

theorem find_number (N : ℕ) : 
  N = sum * quotient + remainder :=
  by
    sorry

end find_number_l566_566924


namespace calculate_tax_l566_566174

theorem calculate_tax
  (price_cheeseburger : ℝ)
  (price_milkshake : ℝ)
  (price_coke : ℝ)
  (price_fries : ℝ)
  (price_cookie : ℝ)
  (toby_money : ℝ)
  (toby_change : ℝ)
  (num_cheeseburgers : ℕ)
  (num_cookies : ℕ) :
  price_cheeseburger = 3.65 →
  price_milkshake = 2 →
  price_coke = 1 →
  price_fries = 4 →
  price_cookie = 0.5 →
  toby_money = 15 →
  toby_change = 7 →
  num_cheeseburgers = 2 →
  num_cookies = 3 →
  let subtotal := (num_cheeseburgers * price_cheeseburger) +
                  price_milkshake +
                  price_coke +
                  price_fries +
                  (num_cookies * price_cookie) in
  let total_paid := toby_money - toby_change + (toby_money - toby_change) in
  let tax := total_paid - subtotal in
  tax = 0.20 :=
by
  intros
  sorry

end calculate_tax_l566_566174


namespace carols_cupcakes_l566_566313

theorem carols_cupcakes (initial_cupcakes sold_cupcakes total_cupcakes made_more : ℕ) 
  (h1 : initial_cupcakes = 30) 
  (h2 : sold_cupcakes = 9) 
  (h3 : total_cupcakes = 49) :
  made_more = total_cupcakes - (initial_cupcakes - sold_cupcakes) :=
by {
  rw [h1, h2, h3],
  sorry
}

end carols_cupcakes_l566_566313


namespace pieces_picked_by_olivia_l566_566450

-- Define the conditions
def picked_by_edward : ℕ := 3
def total_picked : ℕ := 19

-- Prove the number of pieces picked up by Olivia
theorem pieces_picked_by_olivia (O : ℕ) (h : O + picked_by_edward = total_picked) : O = 16 :=
by sorry

end pieces_picked_by_olivia_l566_566450


namespace jill_water_jars_l566_566781

theorem jill_water_jars (x : ℕ) (h : x * (1 / 4 + 1 / 2 + 1) = 28) : 3 * x = 48 :=
by
  sorry

end jill_water_jars_l566_566781


namespace angle_of_inclination_l566_566911

theorem angle_of_inclination (x y : ℝ) (θ : ℝ) :
  (x - y - 1 = 0) → θ = 45 :=
by
  sorry

end angle_of_inclination_l566_566911


namespace range_of_a_l566_566012

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then (a-1)*x + 1 else x^2 - 2*a*x + 6

theorem range_of_a (a : ℝ) : (∀ y ∈ set.univ, ∃ x : ℝ, f a x = y) ↔ 2 ≤ a := 
begin 
  sorry 
end

end range_of_a_l566_566012


namespace platform_length_l566_566946

noncomputable def length_of_platform (L : ℝ) : Prop :=
  ∃ (a : ℝ), 
    -- Train starts from rest
    (0 : ℝ) * 24 + (1/2) * a * 24^2 = 300 ∧
    -- Train crosses a platform in 39 seconds
    (0 : ℝ) * 39 + (1/2) * a * 39^2 = 300 + L ∧
    -- Constant acceleration found
    a = (25 : ℝ) / 24

-- Claim that length of platform should be 492.19 meters
theorem platform_length : length_of_platform 492.19 :=
sorry

end platform_length_l566_566946


namespace weng_total_earnings_l566_566908

noncomputable def weng_earnings_usd : ℝ :=
  let usd_per_hr_job1 : ℝ := 12
  let eur_per_hr_job2 : ℝ := 13
  let gbp_per_hr_job3 : ℝ := 9
  let hr_job1 : ℝ := 2 + 15 / 60
  let hr_job2 : ℝ := 1 + 40 / 60
  let hr_job3 : ℝ := 3 + 10 / 60
  let usd_to_eur : ℝ := 0.85
  let usd_to_gbp : ℝ := 0.76
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.32
  let earnings_job1 : ℝ := usd_per_hr_job1 * hr_job1
  let earnings_job2_eur : ℝ := eur_per_hr_job2 * hr_job2
  let earnings_job2_usd : ℝ := earnings_job2_eur * eur_to_usd
  let earnings_job3_gbp : ℝ := gbp_per_hr_job3 * hr_job3
  let earnings_job3_usd : ℝ := earnings_job3_gbp * gbp_to_usd
  earnings_job1 + earnings_job2_usd + earnings_job3_usd

theorem weng_total_earnings : weng_earnings_usd = 90.19 :=
by
  sorry

end weng_total_earnings_l566_566908


namespace inequality_proof_l566_566327

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + y ≤ (y^2 / x) + (x^2 / y) :=
sorry

end inequality_proof_l566_566327


namespace h_one_value_l566_566051

def f (x : ℝ) : ℝ := 3 * x^2 + 2

def g (x : ℝ) : ℝ := real.sqrt (f x) - 3

def h (x : ℝ) : ℝ := f (g x)

theorem h_one_value : h 1 = 44 - 18 * real.sqrt 5 :=
  sorry

end h_one_value_l566_566051


namespace points_in_circle_l566_566875

theorem points_in_circle (points : Fin 110 → ℝ × ℝ) 
  (h_points_in_unit_square : ∀ i, 0 ≤ points i.1 ∧ points i.1 ≤ 1 ∧ 0 ≤ points i.2 ∧ points i.2 ≤ 1) :
  ∃ (i j k l : Fin 110), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
    dist (points i) (points j) < (1/8) ∧
    dist (points i) (points k) < (1/8) ∧ 
    dist (points i) (points l) < (1/8) :=
sorry

end points_in_circle_l566_566875


namespace coin_difference_is_eight_l566_566824

theorem coin_difference_is_eight :
  let min_coins := 2  -- two 25-cent coins
  let max_coins := 10 -- ten 5-cent coins
  max_coins - min_coins = 8 :=
by
  sorry

end coin_difference_is_eight_l566_566824


namespace problem_a_problem_b_problem_c_problem_d_l566_566387

variable {a b : ℝ}

-- conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + b = 1

theorem problem_a : (a + 1/a) * (b + 1/b) > 4 := sorry

theorem problem_b : ∃ a b, a + b = 1 ∧ a > 0 ∧ b > 0 ∧ (sqrt (1 + a) + sqrt (1 + b) ≤ sqrt 6) 
               ∧ (∀ c d, c + d = 1 ∧ c > 0 ∧ d > 0 → (sqrt (1 + c) + sqrt (1 + d) = sqrt 6)) := sorry

theorem problem_c : ¬ ∃ a b, a + b = 1 ∧ a > 0 ∧ b > 0 ∧ 3 + 2 * Real.sqrt 2 = (1/a + 2/b) := sorry

theorem problem_d : ∃ a b, a + b = 1 ∧ a > 0 ∧ b > 0 ∧ ∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → (ab + 4a + b) / (4a + b) = 10 / 9 := sorry


end problem_a_problem_b_problem_c_problem_d_l566_566387


namespace coefficient_c_for_factor_l566_566184

def P (x : ℝ) (c : ℝ) : ℝ := x^3 + 4*x^2 + c*x - 20

theorem coefficient_c_for_factor (c : ℝ) : (x - 5).is_factor (P x c) ↔ c = -41 :=
by sorry

end coefficient_c_for_factor_l566_566184


namespace b_contribution_l566_566544

/-- A starts business with Rs. 3500.
    After 9 months, B joins as a partner.
    After a year, the profit is divided in the ratio 2:3.
    Prove that B's contribution to the capital is Rs. 21000. -/
theorem b_contribution (a_capital : ℕ) (months_a : ℕ) (b_time : ℕ) (profit_ratio_num : ℕ) (profit_ratio_den : ℕ)
  (h_a_capital : a_capital = 3500)
  (h_months_a : months_a = 12)
  (h_b_time : b_time = 3)
  (h_profit_ratio : profit_ratio_num = 2 ∧ profit_ratio_den = 3) :
  (21000 * b_time * profit_ratio_num) / (3 * profit_ratio_den) = 3500 * months_a := by
  sorry

end b_contribution_l566_566544


namespace f_neither_odd_nor_even_l566_566493

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x ≤ 1 then (1 + x) * Real.sqrt((1 - x) / (1 + x)) else 0

theorem f_neither_odd_nor_even :
  ¬(∀ x, -1 < x ∧ x ≤ 1 → f (-x) = f x) ∧ ¬(∀ x, -1 < x ∧ x ≤ 1 → f (-x) = -f x) :=
by
  sorry

end f_neither_odd_nor_even_l566_566493


namespace cafeteria_arrangement_l566_566764

/-- In the seventh week, Ba Shu High School will arrange 5 student council members from the sophomore year to maintain order in the cafeteria. The arrangement requires one person per day from Monday to Friday, with no repeats. 
Conditions: 
1) Student A cannot be arranged on Monday
2) Student B cannot be arranged on Friday
3) Student C cannot be arranged adjacent to Student A.
Prove that there are 46 different valid arrangements. -/
theorem cafeteria_arrangement :
  let students := ["A", "B", "C", "D", "E"] in
  ∃ arrangements : List (List String), arrangements.length = 46 ∧ 
    ∀ arrangement ∈ arrangements, 
      (arrangement.length = 5 ∧ 
      arrangement[0] ≠ "A" ∧ 
      arrangement[4] ≠ "B" ∧ 
      (∀ i, 0 ≤ i ∧ i < 4 → (arrangement[i] = "A" → arrangement[i+1] ≠ "C") ∧ (arrangement[i+1] = "A" → arrangement[i] ≠ "C"))) := 
sorry

end cafeteria_arrangement_l566_566764


namespace least_x_divisible_by_3_l566_566914

theorem least_x_divisible_by_3 (x : ℕ) (h : 57 % 3 = 0) : ∃ (x₀ : ℕ), x₀ = 0 ∧ (x₀ * 57) % 3 = 0 :=
by
  use 0
  split
  · refl
  · rw [zero_mul, zero_mod, h]
  · sorry
end

end least_x_divisible_by_3_l566_566914


namespace liz_car_percentage_sale_l566_566805

theorem liz_car_percentage_sale (P : ℝ) (h1 : 30000 = P - 2500) (h2 : 26000 = P * (80 / 100)) : 80 = 80 :=
by 
  sorry

end liz_car_percentage_sale_l566_566805


namespace find_other_number_l566_566446

theorem find_other_number (c d : ℤ) (h1 : 3 * c + 4 * d = 161) (h2 : c = 17 ∨ d = 17) : c = 31 ∨ d = 31 :=
by
  cases h2 with
  | inl hc =>
    -- Substitute c = 17 in the equation
    have h3 : 3 * 17 + 4 * d = 161 := by rw [hc] at h1; exact h1
    -- Solve for d
    have h4 : d = 27.5 := by linarith
    -- Since d must be an integer, c = 17 is not valid
    contradiction
  | inr hd =>
    -- Substitute d = 17 in the equation
    have h3 : 3 * c + 4 * 17 = 161 := by rw [hd] at h1; exact h1
    -- Solve for c
    have h4 : 3 * c = 161 - 68 := by linarith
    have h5 : 3 * c = 93 := by linarith
    have h6 : c = 31 := by linarith
    -- c = 31 is valid
    exact Or.inl h6
  sorry

end find_other_number_l566_566446


namespace problem_1_problem_2_l566_566711

-- Define the universal set U
def U := Set.univ

-- Define sets A and B based on the given conditions
def A : Set ℝ := {x : ℝ | x^2 - 3 * x - 18 ≥ 0}
def B : Set ℝ := {x : ℝ | (x + 5) / (x - 14) ≤ 0}

-- Define set C based on the given conditions
def C (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Statement (1): Prove the intersection of the complement of B in U with A
theorem problem_1 : (U \ B) ∩ A = (Set.Iio (-5) ∪ Set.Ici 14) ∩ A := by
  sorry

-- Statement (2): Prove the range of the real number a
theorem problem_2 : ∀ a : ℝ, (B ∩ C a = C a) ↔ a ≥ -5 / 2 := by
  sorry

end problem_1_problem_2_l566_566711


namespace problem_3_div_27_l566_566662

theorem problem_3_div_27 (a b : ℕ) (h : 2^a = 8^(b + 1)) : 3^a / 27^b = 27 := by
  -- proof goes here
  sorry

end problem_3_div_27_l566_566662


namespace exists_two_solutions_l566_566658

noncomputable def system_two_solutions (a : ℝ) : Prop :=
  let eq1 := ∀ (x y : ℝ), x * Real.sin a - (y - 6) * Real.cos a = 0
  let eq2 := ∀ (x y : ℝ), ((x - 3) ^ 2 + (y - 3) ^ 2 - 1) * ((x - 3) ^ 2 + (y - 3) ^ 2 - 9) = 0
  ∃ n : ℤ, (a > π / 2 + π * n) ∧ (a < 3 * π / 4 - Real.asin (√2 / 6) + π * n) ∨
            (a > 3 * π / 4 + Real.asin (√2 / 6) + π * n) ∧ (a < π + π * n)

theorem exists_two_solutions :
  ∀ a : ℝ, system_two_solutions a := sorry

end exists_two_solutions_l566_566658


namespace positive_x_is_approx_5_46_l566_566710

noncomputable def xyz_system_pos_solution (x y z : ℝ) : Prop :=
  x > 0 ∧
  y > 0 ∧
  z > 0 ∧
  (xy = 8 - 3 * x - 2 * y) ∧
  (yz = 8 - 3 * y - 3 * z) ∧
  (xz = 40 - 5 * x - 4 * z)

noncomputable def x_value_expr (x : ℝ) : ℝ :=
  sqrt (14 * 17 * 60) / 17 - 2

theorem positive_x_is_approx_5_46 :
  ∃ x y z : ℝ, xyz_system_pos_solution x y z ∧ abs (x - 5.46) < 0.01 :=
  sorry

end positive_x_is_approx_5_46_l566_566710


namespace max_shelves_within_rearrangement_l566_566262

theorem max_shelves_within_rearrangement (k : ℕ) :
  (∀ books : finset ℕ, books.card = 1300 →
    (∃ shelf_assignment_before shelf_assignment_after : finset ℕ → ℕ,
      (∀ book, shelf_assignment_before book ≤ k ∧ shelf_assignment_after book ≤ k) ∧
      (∃ shelf, (books.filter (λ book, shelf_assignment_before book = shelf)).card ≥ 5 ∧
                (books.filter (λ book, shelf_assignment_after book = shelf)).card ≥ 5)
    )
  ) → k ≤ 18 := sorry

end max_shelves_within_rearrangement_l566_566262


namespace find_d_l566_566486

-- Conditions
def line_eq (x y : ℝ) : Prop :=
  y = (2 * x + 1) / 3

def param_eq (v d : ℝ × ℝ) (t x y : ℝ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2) ∧ x ≥ 4

def distance_cond (x y t : ℝ) : Prop :=
  real.sqrt ((x - 4) ^ 2 + (y - 2) ^ 2) = t

-- Given conditions
def conditions (v d : ℝ × ℝ) : Prop :=
  v = (4, 2) ∧ d = (3 / real.sqrt 13, 2 / real.sqrt 13)

-- The proof goal
theorem find_d (v d : ℝ × ℝ) (t x y : ℝ) :
  param_eq v d t x y ∧ distance_cond x y t →
  conditions v d :=
by
  sorry

end find_d_l566_566486


namespace problem_monotonicity_problem_zeros_range_l566_566365

noncomputable def f (a x : ℝ) : ℝ := (x - 1) * exp(x) - (1 / 2) * a * x^2

theorem problem_monotonicity (a x : ℝ) : 
  (f a x ≤ f a (x + ε) ∧ ε > 0 ∧ (a ≤ 0 ∨ (0 < a ∧ a < 1 ∨ a > 1))) → 
  (f a x > f a (x - ε) ∧ ε > 0) :=
sorry

theorem problem_zeros_range (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ x1 ≠ x2) ↔ a ≤ 0 :=
sorry

end problem_monotonicity_problem_zeros_range_l566_566365


namespace abs_diff_sub_l566_566549

theorem abs_diff_sub : |8 - 3| - |4 - 6| = 3 := 
by {
  -- Evaluating the absolute value differences.
  have h1 : |8 - 3| = 5, by exact abs_of_nonneg (by norm_num),
  have h2 : |4 - 6| = 2, by exact abs_of_neg (by norm_num),
  rw [h1, h2],
  norm_num,
  sorry
}

end abs_diff_sub_l566_566549


namespace sum_of_divisors_2310_l566_566536

theorem sum_of_divisors_2310 :
  let N := 2310
  let sum_divisors (n : ℕ) : ℕ :=
    ∑ d in (list.range (n + 1)).filter (λ d, n % d = 0), d
  sum_divisors N = 6912 := sorry

end sum_of_divisors_2310_l566_566536


namespace modified_pyramid_volume_l566_566973

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) 
  (hV : V = 1/3 * s^2 * h) (hV_eq : V = 72) :
  (1/3) * (3 * s)^2 * (2 * h) = 1296 := by
  sorry

end modified_pyramid_volume_l566_566973


namespace minimally_intersecting_triples_count_correct_l566_566277

open Finset

def minimally_intersecting_triples_count : Nat :=
  let universe := range 8
  (universe.card.choose 3) * (universe.erase x.card.choose 2) * (universe.erase y.card.choose 1) % 1000

theorem minimally_intersecting_triples_count_correct :
  minimally_intersecting_triples_count = 80 := by
  sorry

end minimally_intersecting_triples_count_correct_l566_566277


namespace hyperbola_equation_and_points_l566_566672

open Real

variable (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y c : ℝ) (F : ℝ × ℝ→ℝ)

noncomputable def hyperbola_asymptote_ratio := b = √3 * a

noncomputable def hyperbola_focus_dist := c = 2

theorem hyperbola_equation_and_points
  (hyperbola_eq : x^2 / a^2 - y^2 / b^2 = 1)
  (asymptote_eq : hyperbola_asymptote_ratio a b)
  (focus_eq : hyperbola_focus_dist c)
  (focus_pt : F 2 0) :
  (x^2 - y^2 / 3 = 1) ∧ 
  (∃ (E₁ E₂ : ℝ × ℝ), E₁ = (-4, 0) ∧ E₂ = (4, 0) ∧ ∀ (M : ℝ × ℝ), abs (dist M E₁ - dist M E₂) = 4) :=
sorry

end hyperbola_equation_and_points_l566_566672


namespace range_of_a_l566_566667

theorem range_of_a
  (a x : ℝ)
  (p : -4 < x - a ∧ x - a < 4)
  (q : (x - 2) * (3 - x) > 0)
  (suff_cond : ∀ x, q → p) : -1 ≤ a ∧ a ≤ 6 := sorry

end range_of_a_l566_566667


namespace density_comparison_l566_566991

variables (m_CT m_M V_CT V_M : ℝ)
noncomputable def rho_CT := m_CT / V_CT
noncomputable def rho_M := m_M / V_M

theorem density_comparison 
  (h1 : m_M = 2 * m_CT)
  (h2 : V_M = 10 * V_CT) :
  rho_M m_CT m_M V_CT V_M = 0.2 * rho_CT m_CT V_CT :=
sorry

end density_comparison_l566_566991


namespace simplify_and_evaluate_l566_566837

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l566_566837


namespace two_digit_number_representation_l566_566590

theorem two_digit_number_representation (a b : ℕ) (ha : a < 10) (hb : b < 10) : 10 * b + a = d :=
  sorry

end two_digit_number_representation_l566_566590


namespace ha_unique_expression_l566_566826

variable {α β γ a b p : ℝ}
variable {h_a : ℝ}

-- Assumptions
variable (triangle_inequality : (0 < α ∧ α < π) ∧ (0 < β ∧ β < π) ∧ (0 < γ ∧ γ < π))
variable (semiperimeter_def : p = (a + b + √(a*b)) / 2)
variable (relation1 : (p - a) * (Math.cos (β / 2)) * (Math.sin (α / 2)) = (p - b) * (Math.cos (α / 2)) * (Math.sin (β / 2)))
variable (relation2 : a * h_a = 2 * (p - a) * ((a * (Math.cos (β / 2) * Math.cos (γ / 2))) / (Math.cos (α / 2))))

theorem ha_unique_expression :
  h_a = (2 * (p - a) * (Math.cos (β / 2) * Math.cos (γ / 2)) / (Math.cos (α / 2))) ∧
  h_a = (2 * (p - b) * (Math.sin (β / 2) * Math.cos (γ / 2)) / (Math.sin (α / 2))) := sorry

end ha_unique_expression_l566_566826


namespace monotonicity_of_f_range_of_a_l566_566698

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x ^ 2 + (1 - a) * x - a * Real.log x

-- I. Monotonicity of f(x)
theorem monotonicity_of_f (a x : ℝ) (h₁ : 0 < x) : 
  (a ≤ 0 → (∀ x, 0 < x -> deriv (f a) x > 0)) ∧ 
  (a > 0 → (∀ x, 0 < x ∧ x < a -> deriv (f a) x < 0) ∧ (∀ x, a < x -> deriv (f a) x > 0)) :=
by 
  sorry

-- II. Range of a such that |f(x1) - f(x2)| ≥ 4|x1 - x2|
theorem range_of_a (a : ℝ) (h : a < 0) :
  (∀ x1 x2 : ℝ, (0 < x1) ∧ (0 < x2) → |f a x1 - f a x2| ≥ 4 * |x1 - x2|) → a ≤ -1 :=
by 
  sorry

end monotonicity_of_f_range_of_a_l566_566698


namespace Simplify_and_evaluate_expression_l566_566838

theorem Simplify_and_evaluate_expression (x : ℝ) (h : x^2 + x - 5 = 0) :
    (let expr := (x - 2) / ((x - 2)^2) / (x + 2 - (x^2 + x - 4) / (x - 2)) + 1 / (x + 1) in
    expr) = -1 / 5 := 
sorry

end Simplify_and_evaluate_expression_l566_566838


namespace acute_triangle_sides_l566_566638

theorem acute_triangle_sides (n : ℕ) (h : n ≥ 13) :
  ∀ (a : Fin n → ℝ), (∀ i, a i > 0) →
    (Finset.max' (Finset.univ.image a) (by
      intro h
      simp at h)) ≤ n * Finset.min' (Finset.univ.image a) (by 
        intro h
        simp at h) →
    ∃ i j k : Fin n, 
      i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
      let x := a i, y := a j, z := a k in 
      (x, y, z) = if x ≤ y then if y ≤ z then (x, y, z) else if x ≤ z then (x, z, y) else (z, x, y) else if x ≤ z then (y, x, z) else if y ≤ z then (y, z, x) else (z, y, x) ∧
      match x, y, z with 
      | (x, y, z) => (x^2 + y^2 > z^2) :=
  sorry

end acute_triangle_sides_l566_566638


namespace find_c_share_l566_566467

noncomputable def shares (a b c d : ℝ) : Prop :=
  (5 * a = 4 * c) ∧ (7 * b = 4 * c) ∧ (2 * d = 4 * c) ∧ (a + b + c + d = 1200)

theorem find_c_share (A B C D : ℝ) (h : shares A B C D) : C = 275 :=
  by
  sorry

end find_c_share_l566_566467


namespace pauline_bought_2_pounds_of_meat_l566_566088

theorem pauline_bought_2_pounds_of_meat :
  ∀ (cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent : ℝ) 
    (num_bell_peppers : ℕ),
  cost_taco_shells = 5 →
  cost_bell_pepper = 1.5 →
  cost_meat_per_pound = 3 →
  total_spent = 17 →
  num_bell_peppers = 4 →
  (total_spent - (cost_taco_shells + (num_bell_peppers * cost_bell_pepper))) / cost_meat_per_pound = 2 :=
by
  intros cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent num_bell_peppers 
         h1 h2 h3 h4 h5
  sorry

end pauline_bought_2_pounds_of_meat_l566_566088


namespace bob_grade_is_35_l566_566771

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l566_566771


namespace stickers_per_page_l566_566512

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end stickers_per_page_l566_566512


namespace quadratic_points_range_l566_566336

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end quadratic_points_range_l566_566336


namespace find_possible_numbers_l566_566207

theorem find_possible_numbers (N : ℕ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9)
  (cond : nat.num_digits N = 200 ∧ N = 10^199 * 5 * a ∧ 10^199 * 5 * a < 10^200) :
  (∃ a, N = 125 * a * 10^197 ∧ (1 ≤ a ∧ a ≤ 3)) :=
by
  sorry

end find_possible_numbers_l566_566207


namespace find_cos_4_theta_l566_566734

noncomputable def cos_4_theta : ℂ → ℝ :=
  λ θ, real.cos (4 * θ)

theorem find_cos_4_theta (θ : ℂ) (h : complex.exp (complex.I * θ) = (1 - complex.I * real.sqrt 8) / 3) :
  cos_4_theta θ = 17 / 81 :=
by
  sorry

end find_cos_4_theta_l566_566734


namespace smallest_k_l566_566832

noncomputable def sequence_v : ℕ → ℝ
| 0       := -1/8
| (n + 1) := 1/2 * sequence_v n * (1 - sequence_v n)

theorem smallest_k (M : ℝ) (hM : M = 0) :
  ∃ k : ℕ, k = 497 ∧ |sequence_v k - M| ≤ 1 / 2^500 :=
begin
  use 497,
  have h_seq : ∀ k, sequence_v k = (-1/8) * (1/2)^k,
  { intro k,
    induction k with k ih,
    { refl },
    { simp [sequence_v, ih],
      ring } },
  rw [h_seq 497, hM],
  simp,
  norm_num,
end

end smallest_k_l566_566832


namespace max_value_fraction_modulus_l566_566794

open Complex

noncomputable def max_value_fraction_modulus (α β : ℂ) (θ : ℝ) (hβ1 : abs β = 2) (hβ2 : β = 2 * exp (I * θ)) (h_ne : conj α * β ≠ 1) : 
  ℝ := 
  sorry -- The definition of the expression we'd like to maximize

theorem max_value_fraction_modulus (α β : ℂ) (θ : ℝ) (hβ1 : abs β = 2) (hβ2 : β = 2 * exp (I * θ)) (h_ne : conj α * β ≠ 1) :
  max_value_fraction_modulus α β θ hβ1 hβ2 h_ne ≤ 1 :=
sorry

end max_value_fraction_modulus_l566_566794


namespace average_hamburgers_per_day_l566_566976

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l566_566976


namespace magnitude_fourth_power_l566_566645

open Complex

noncomputable def complex_magnitude_example : ℂ := 4 + 3 * Real.sqrt 3 * Complex.I

theorem magnitude_fourth_power :
  ‖complex_magnitude_example ^ 4‖ = 1849 := by
  sorry

end magnitude_fourth_power_l566_566645


namespace derivative_not_in_second_quadrant_l566_566739

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 4

-- Given condition: Axis of symmetry is x = 2
def axis_of_symmetry (b : ℝ) : Prop := b = -4

-- Additional condition: behavior of the derivative and quadrant check
def not_in_second_quadrant (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f' x < 0

-- The main theorem to be proved
theorem derivative_not_in_second_quadrant (b c : ℝ) (h : axis_of_symmetry b) :
  not_in_second_quadrant f_derivative :=
by {
  sorry
}

end derivative_not_in_second_quadrant_l566_566739


namespace dice_probability_never_two_l566_566188

open Probability

variable (a b c : Fin 6 → ℕ)  -- Three random variables representing the dice rolls, each taking values in {1, 2, 3, 4, 5, 6}

/-- The mathematical statement to prove the probability that (a-2)(b-2)(c-2) ≠ 0 given three standard dice rolls is 8/27. -/
theorem dice_probability_never_two {Ω : Type*} [ProbabilitySpace Ω] :
  let D : Set (Fin 6 → ℕ) := {x | x 0 ≠ 2 ∧ x 1 ≠ 2 ∧ x 2 ≠ 2} in
  Prob(D) = 8 / 27 :=
sorry

end dice_probability_never_two_l566_566188


namespace martins_spending_l566_566809

-- Define the conditions:
def dailyBerryConsumption : ℚ := 1 / 2
def costPerCup : ℚ := 2
def days : ℕ := 30

-- Define the main theorem:
theorem martins_spending : (dailyBerryConsumption * days * costPerCup) = 30 := by
  -- This is where the proof would go.
  sorry

end martins_spending_l566_566809


namespace cobbler_hours_per_day_l566_566215

-- Defining some conditions based on our problem statement
def cobbler_rate : ℕ := 3  -- pairs of shoes per hour
def friday_hours : ℕ := 3  -- number of hours worked on Friday
def friday_pairs : ℕ := cobbler_rate * friday_hours  -- pairs mended on Friday
def weekly_pairs : ℕ := 105  -- total pairs mended in a week
def mon_thu_pairs : ℕ := weekly_pairs - friday_pairs  -- pairs mended from Monday to Thursday
def mon_thu_hours : ℕ := mon_thu_pairs / cobbler_rate  -- total hours worked from Monday to Thursday

-- Thm statement: If a cobbler works h hours daily from Mon to Thu, then h = 8 implies total = 105 pairs
theorem cobbler_hours_per_day (h : ℕ) : (4 * h = mon_thu_hours) ↔ (h = 8) :=
by
  sorry

end cobbler_hours_per_day_l566_566215


namespace infinite_series_tan_eq_l566_566827

theorem infinite_series_tan_eq (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π / 4) :
    (1 - ∑' n : ℕ, (tan ϕ) ^ n) = (sqrt 2 * cos ϕ) / (2 * sin (π / 4 + ϕ)) := sorry

end infinite_series_tan_eq_l566_566827


namespace angle_AKM_eq_angle_CDN_l566_566632

-- Define point and angle representations, assume the given square ABCD and divisions
variables (A B C D K M N : Type) [square ABCD]
variables [div_three A B C D] 
variables [connected_points K M N]

-- State the theorem
theorem angle_AKM_eq_angle_CDN 
  (h1 : is_square ABCD)
  (h2 : divided_by_three_parts ABCD)
  (h3 : connected_points_opp_sides ABCD) :
  ∠AKM = ∠CDN := 
sorry

end angle_AKM_eq_angle_CDN_l566_566632


namespace mom_has_enough_money_l566_566081

def original_price : ℝ := 268
def discount_rate : ℝ := 0.2
def money_brought : ℝ := 230
def discounted_price := original_price * (1 - discount_rate)

theorem mom_has_enough_money : money_brought ≥ discounted_price := by
  sorry

end mom_has_enough_money_l566_566081


namespace coke_cost_l566_566520

-- Define conditions
def cheeseburger_cost : ℝ := 3.65
def milkshake_cost : ℝ := 2
def large_fries_cost : ℝ := 4
def cookie_cost : ℝ := 0.5
def tax : ℝ := 0.2
def toby_initial_money : ℝ := 15
def toby_change : ℝ := 7

-- Define what we need to show
theorem coke_cost :
  let total_meal_cost := 2 * cheeseburger_cost + milkshake_cost + large_fries_cost + 3 * cookie_cost + tax,
      toby_share := toby_initial_money - toby_change,
      total_bill := 2 * toby_share in
  total_bill - total_meal_cost = 1 := sorry

end coke_cost_l566_566520


namespace range_of_m_l566_566939

noncomputable def f (x m : ℝ) := Real.exp x * (Real.log x + (1 / 2) * x ^ 2 - m * x)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → ((Real.exp x * ((1 / x) + x - m)) > 0)) → m < 2 := by
  sorry

end range_of_m_l566_566939


namespace logs_sawed_l566_566141

/-- Given the conditions described, we need to find the number of logs (L) -/
theorem logs_sawed (number_of_cuts number_of_pieces : ℕ) (H1 : number_of_cuts = 10) (H2 : number_of_pieces = 16) : 
  let L := number_of_pieces - number_of_cuts in
  L = 6 := by
  sorry

end logs_sawed_l566_566141


namespace area_of_stripe_l566_566219

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end area_of_stripe_l566_566219


namespace FA_equals_AB_l566_566052

variable (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]

-- Define the pentagon properties
variable (AB AE BC CD DE BD CE : ℝ)

-- Define the conditions as stated in the problem
variable (pentagon : ∀ (A B C D E : Type), -- Pentagon 
  (AB = AE) ∧ -- AB equals AE
  (BC = CD) ∧ -- BC equals CD
  (CD = DE) ∧ -- CD equals DE
  (RightAngle (angle E B F)) ∧ -- Right angle at B
  (RightAngle (angle A E B)) ∧ -- Right angle at E
  (intersect BD CE = F) -- Diagonals BD and CE intersect at F
)

-- Prove the required result
theorem FA_equals_AB
  (pentagon : ∀ (A B C D E : Type), 
    (AB = AE) ∧ 
    (BC = CD) ∧ 
    (CD = DE) ∧ 
    (RightAngle (angle E B F)) ∧ 
    (RightAngle (angle A E B)) ∧ 
    (intersect BD CE = F)
  )
  : (FA = AB) :=
  by sorry

end FA_equals_AB_l566_566052


namespace number_of_ways_to_select_president_and_vice_president_l566_566399

-- Define the given conditions
def num_candidates : Nat := 4

-- Define the problem to prove
theorem number_of_ways_to_select_president_and_vice_president : (num_candidates * (num_candidates - 1)) = 12 :=
by
  -- This is where the proof would go, but we are skipping it
  sorry

end number_of_ways_to_select_president_and_vice_president_l566_566399


namespace coin_flip_probability_l566_566931

theorem coin_flip_probability (P : ℕ → ℕ → ℚ) (n : ℕ) :
  (∀ k, P k 0 = 1/2) →
  (∀ k, P k 1 = 1/2) →
  (∀ k m, P k m = 1/2) →
  n = 3 →
  P 0 0 * P 1 1 * P 2 1 = 1/8 :=
by
  intros h0 h1 h_indep hn
  sorry

end coin_flip_probability_l566_566931


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566317

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566317


namespace right_triangle_area_hypotenuse_30_deg_l566_566584

theorem right_triangle_area_hypotenuse_30_deg
  (h : Real)
  (θ : Real)
  (A : Real)
  (H1 : θ = 30)
  (H2 : h = 12)
  : A = 18 * Real.sqrt 3 := by
  sorry

end right_triangle_area_hypotenuse_30_deg_l566_566584


namespace constant_term_in_expansion_l566_566853

theorem constant_term_in_expansion :
  ∃ (c : ℕ), c = 70 ∧ (∀ x : ℚ, x ≠ 0 → constant_term_in_expansion (1/x^2 + x^2 + 2)^4 = c) :=
sorry

end constant_term_in_expansion_l566_566853


namespace equation_has_only_solution_l566_566639

theorem equation_has_only_solution (x : ℝ) : 
  (∛(17 * x - 1) + ∛(11 * x + 1) = 2 * ∛x) → (x = 0) := by
  -- Add proof steps here
  sorry

end equation_has_only_solution_l566_566639


namespace symmetric_curve_eq_l566_566294

-- Define the original curve equation and line of symmetry
def original_curve (x y : ℝ) : Prop := y^2 = 4 * x
def line_of_symmetry (x : ℝ) : Prop := x = 2

-- The equivalent Lean 4 statement
theorem symmetric_curve_eq (x y : ℝ) (hx : line_of_symmetry 2) :
  (∀ (x' y' : ℝ), original_curve (4 - x') y' → y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_eq_l566_566294


namespace incorrect_statement_l566_566920

def line := ℝ × ℝ × ℝ
def plane := ℝ × ℝ × ℝ × ℝ

-- Definitions of conditions
def condition_A (L₁ : line) (P₂ : plane) : Prop :=
  (∀ (L ∈ P₂), ⊥_L₁ L) → ⊥_P₁ P₂

def condition_B (L : line) (P₁ P₂ : plane) : Prop :=
  (‖ L ∈ P₁ ∧ P₂) → ∥_P₁ P₂

def condition_C (L : line) (P₁ P₂ : plane) : Prop :=
  (‖ L ∈ P₁ ∧ ⋂ P₁, L ∥ Lin P₁ P₂)

def condition_D (L₁ L₂ : line) (P : plane) : Prop :=
  (⊥_L₁ P₁ ∧ ⊥_L₂ P₁) → ⊥_L₁ L₂

-- The theorem to be proven
theorem incorrect_statement : ¬condition_B :=
sorry

end incorrect_statement_l566_566920


namespace total_number_of_bottles_l566_566209

def water_bottles := 2 * 12
def orange_juice_bottles := (7 / 4) * 12
def apple_juice_bottles := water_bottles + 6
def total_bottles := water_bottles + orange_juice_bottles + apple_juice_bottles

theorem total_number_of_bottles :
  total_bottles = 75 :=
by
  sorry

end total_number_of_bottles_l566_566209


namespace root_eq_neg_l566_566013

theorem root_eq_neg {a : ℝ} (h : 3 * a - 9 < 0) : (a - 4) * (a - 5) > 0 :=
by
  sorry

end root_eq_neg_l566_566013


namespace maximize_happy_monkeys_l566_566967

theorem maximize_happy_monkeys (pears bananas peaches mandarins : ℕ)
  (h1 : pears = 20) (h2 : bananas = 30) (h3 : peaches = 40) (h4 : mandarins = 50) :
  ∃ (max_monkeys : ℕ), max_monkeys = 45 :=
by 
  use 45
  sorry

end maximize_happy_monkeys_l566_566967


namespace intersection_in_fourth_quadrant_l566_566537

theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (y = log a x) ∧ (y = (1 - a) * x) ∧ (x > 0) ∧ (y < 0) :=
sorry

end intersection_in_fourth_quadrant_l566_566537


namespace angle_between_vectors_l566_566717

-- We state the conditions from part a)
variables {a b : EuclideanSpace ℝ (Fin 3)}

open Real

-- Definitions based on given conditions
def vector_lengths (v : EuclideanSpace ℝ (Fin 3)) : ℝ := ∥v∥

def perp_condition (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  b ⬝ (2 • a + b) = 0

def lengths_condition (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥a∥ = 2 ∧ ∥b∥ = 2

-- Prove that the angle between vector a and b is 2π/3 given the conditions
theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3))
  (h_perp : perp_condition a b)
  (h_lengths : lengths_condition a b) :
  angle a b = 2 * π / 3 :=
sorry

end angle_between_vectors_l566_566717


namespace tangent_line_parallel_curve_l566_566361

def curve (x : ℝ) : ℝ := x^4

def line_parallel_to_curve (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x0 y0 : ℝ, l x0 y0 ∧ curve x0 = y0 ∧ ∀ (x : ℝ), l x (curve x)

theorem tangent_line_parallel_curve :
  ∃ (l : ℝ → ℝ → Prop), line_parallel_to_curve l ∧ ∀ x y, l x y ↔ 8 * x + 16 * y + 3 = 0 :=
by
  sorry

end tangent_line_parallel_curve_l566_566361


namespace Fedya_third_l566_566094

/-- Definitions for order of children's arrival -/
inductive Child
| Roman | Fedya | Liza | Katya | Andrew

open Child

def arrival_order (order : Child → ℕ) : Prop :=
  order Liza > order Roman ∧
  order Katya < order Liza ∧
  order Fedya = order Katya + 1 ∧
  order Katya ≠ 1

/-- Theorem stating that Fedya is third based on the given conditions -/
theorem Fedya_third (order : Child → ℕ) (H : arrival_order order) : order Fedya = 3 :=
sorry

end Fedya_third_l566_566094


namespace find_slope_of_intersecting_line_l566_566447

-- Define the conditions
def line_p (x : ℝ) : ℝ := 2 * x + 3
def line_q (x : ℝ) (m : ℝ) : ℝ := m * x + 1

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (4, 11)

-- Prove that the slope m of line q such that both lines intersect at (4, 11) is 2.5
theorem find_slope_of_intersecting_line (m : ℝ) :
  line_q 4 m = 11 → m = 2.5 :=
by
  intro h
  sorry

end find_slope_of_intersecting_line_l566_566447


namespace polynomials_product_even_not_4_l566_566497

-- Define the polynomial P
noncomputable def P (n : ℕ) (a : ℕ → ℕ) (x : ℕ) : ℕ :=
  ∑ i in finset.range (n+1), a i * x^i

-- Define the polynomial Q
noncomputable def Q (m : ℕ) (b : ℕ → ℕ) (x : ℕ) : ℕ :=
  ∑ j in finset.range (m+1), b j * x^j

-- Define the product of P and Q
noncomputable def product (n m : ℕ) (a b : ℕ → ℕ) (x : ℕ) : ℕ :=
  (P n a x) * (Q m b x)

-- Define even property for coefficients
def all_even (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i < n + 1, even (a i)

-- Define not all divisible by 4 property
def not_all_divisible_by_4 (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ i < n + 1, ¬ (4 ∣ a i)

-- Main theorem statement
theorem polynomials_product_even_not_4
  (n m : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ)
  (h_even : all_even (n + m) (λ k, (product n m a b k)))
  (h_not_4 : not_all_divisible_by_4 (n + m) (λ k, (product n m a b k))) :
  (all_even n a ∧ (∃ j < m + 1, odd (b j))) ∨ (all_even m b ∧ (∃ i < n + 1, odd (a i))) :=
begin
  sorry -- Proof here
end

end polynomials_product_even_not_4_l566_566497


namespace general_term_formula_l566_566843

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 4
  else sequence (n - 2) - 2 * sequence (n - 1)

theorem general_term_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h : ∀ n : ℕ, 0 < n → a n - 2 * a (n + 1) + a (n + 2) = 0) :
  ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end general_term_formula_l566_566843


namespace find_side_b_l566_566375

theorem find_side_b
  (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 6)
  (ha : a = sqrt 3)
  (hc : c = 1)
  (cos_rule : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : 
  b = 1 :=
by sorry

end find_side_b_l566_566375


namespace smallest_m_for_partition_l566_566061

def T (m : ℕ) : Set ℕ := { n | 2 ≤ n ∧ n ≤ m }

theorem smallest_m_for_partition (m : ℕ) (h : m ≥ 256) : 
  ∃ partition : T(m) → Bool, ∀ a b c : ℕ, 
  T(m) a → T(m) b → T(m) c → ¬(partition a = partition b ∧ partition b = partition c) → ab ≠ c :=
sorry

end smallest_m_for_partition_l566_566061


namespace dolphins_to_be_trained_next_month_l566_566508

-- Definition of conditions
def total_dolphins : ℕ := 20
def fraction_fully_trained := 1 / 4
def fraction_currently_training := 2 / 3

-- Lean 4 statement for the proof problem
theorem dolphins_to_be_trained_next_month :
  let fully_trained := total_dolphins * fraction_fully_trained
  let remaining := total_dolphins - fully_trained
  let currently_training := remaining * fraction_currently_training
  remaining - currently_training = 5 := by
begin
  -- Calculation core based on the given conditions
  let fully_trained := total_dolphins * fraction_fully_trained,
  let remaining := total_dolphins - fully_trained,
  let currently_training := remaining * fraction_currently_training,
  show remaining - currently_training = 5,
  sorry  -- Proof should go here
end

end dolphins_to_be_trained_next_month_l566_566508


namespace irrational_sqrt_3_l566_566245

theorem irrational_sqrt_3 : ¬ ∃ (q : ℚ), (q : ℝ) = Real.sqrt 3 := by
  sorry

end irrational_sqrt_3_l566_566245


namespace find_x2_plus_y2_l566_566292

theorem find_x2_plus_y2 
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h1 : x * y + x + y = 117) 
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := 
sorry

end find_x2_plus_y2_l566_566292


namespace sum_of_a_for_unique_solution_in_quadratic_eq_l566_566517

theorem sum_of_a_for_unique_solution_in_quadratic_eq : 
  (sum_of_as_for_quadratic_has_unique_solution eqn1) = -12 :=
by
  -- Define the conditions
  let eqn1 := λ (a : ℝ) => 2 * x^2 + (a + 6) * x + 7 = 0
  -- State the theorem with the given problem conditions
  sorry

end sum_of_a_for_unique_solution_in_quadratic_eq_l566_566517


namespace proof_firstExpr_proof_secondExpr_l566_566604

noncomputable def firstExpr : ℝ :=
  Real.logb 2 (Real.sqrt (7 / 48)) + Real.logb 2 12 - (1 / 2) * Real.logb 2 42 - 1

theorem proof_firstExpr :
  firstExpr = -3 / 2 :=
by
  sorry

noncomputable def secondExpr : ℝ :=
  (Real.logb 10 2) ^ 2 + Real.logb 10 (2 * Real.logb 10 50 + Real.logb 10 25)

theorem proof_secondExpr :
  secondExpr = 0.0906 + Real.logb 10 5.004 :=
by
  sorry

end proof_firstExpr_proof_secondExpr_l566_566604


namespace inequality_abc_ge_1_sqrt_abcd_l566_566425

theorem inequality_abc_ge_1_sqrt_abcd
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) :=
by
  sorry

end inequality_abc_ge_1_sqrt_abcd_l566_566425


namespace annual_growth_rate_l566_566223

variable { P : ℝ }
variable { x : ℝ }

theorem annual_growth_rate (h : P * (1 + x)^2 = 1.2 * P) : x < 0.1 :=
by
  have h1 : (1 + x)^2 = 1.2 := by
    sorry
  have h2 : x = Real.sqrt 1.2 - 1 := by
    sorry
  have h3 : Real.sqrt 1.2 < 1.1 := by
    sorry
  have h4 : Real.sqrt 1.2 - 1 < 0.1 := by
    sorry
  exact h4

end annual_growth_rate_l566_566223


namespace bus_driver_limit_of_hours_l566_566951

theorem bus_driver_limit_of_hours (r o T H L : ℝ)
  (h_reg_rate : r = 16)
  (h_ot_rate : o = 1.75 * r)
  (h_total_comp : T = 752)
  (h_hours_worked : H = 44)
  (h_equation : r * L + o * (H - L) = T) :
  L = 40 :=
  sorry

end bus_driver_limit_of_hours_l566_566951


namespace macey_weeks_to_save_l566_566449

theorem macey_weeks_to_save :
  ∀ (total_cost amount_saved weekly_savings : ℝ),
    total_cost = 22.45 →
    amount_saved = 7.75 →
    weekly_savings = 1.35 →
    ⌈(total_cost - amount_saved) / weekly_savings⌉ = 11 :=
by
  intros total_cost amount_saved weekly_savings h_total_cost h_amount_saved h_weekly_savings
  sorry

end macey_weeks_to_save_l566_566449


namespace sum_of_first_n_primes_eq_41_l566_566649

theorem sum_of_first_n_primes_eq_41 : 
  ∃ (n : ℕ) (primes : List ℕ), 
    primes = [2, 3, 5, 7, 11, 13] ∧ primes.sum = 41 ∧ primes.length = n := 
by 
  sorry

end sum_of_first_n_primes_eq_41_l566_566649


namespace greatest_possible_value_of_a_l566_566118

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l566_566118


namespace function_roots_in_interval_l566_566702

theorem function_roots_in_interval 
  (a b : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0)
  (symmetric : ∀ x, (a * sin (2 * (x + π / 6)) + b * cos (2 * (x + π / 6))) 
                      = (a * sin (2 * x) + b * cos (2 * x)))
  : ∃ x1 x2 ∈ [0, 2 * π], (a * sin (2 * x1) + b * cos (2 * x1)) = 2 * b ∧ (a * sin (2 * x2) + b * cos (2 * x2)) = 2 * b ∧ x1 ≠ x2 :=
begin
  sorry
end

end function_roots_in_interval_l566_566702


namespace smallest_four_digit_divisible_by_9_l566_566915

theorem smallest_four_digit_divisible_by_9 
    (n : ℕ) 
    (h1 : 1000 ≤ n ∧ n < 10000) 
    (h2 : n % 9 = 0)
    (h3 : n % 10 % 2 = 1)
    (h4 : (n / 1000) % 2 = 1)
    (h5 : (n / 10) % 10 % 2 = 0)
    (h6 : (n / 100) % 10 % 2 = 0) :
  n = 3609 :=
sorry

end smallest_four_digit_divisible_by_9_l566_566915


namespace garden_ratio_l566_566582

theorem garden_ratio (L W : ℕ) (h1 : L = 50) (h2 : 2 * L + 2 * W = 150) : L / W = 2 :=
by
  sorry

end garden_ratio_l566_566582


namespace point_outside_circle_l566_566395

-- Define the conditions
def Point := ℝ × ℝ

def Circle (center : Point) (radius : ℝ) : set Point :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the specific circle and point
def center : Point := (a, -2)
def radius : ℝ := 5
def P : Point := (-1, 1)

-- Define the proof problem
theorem point_outside_circle (a : ℝ) :
  ¬(((-1 - a)^2 + (1 + 2)^2) ≤ radius^2) → (a > 3 ∨ a < -5) :=
sorry

end point_outside_circle_l566_566395


namespace ken_change_l566_566599

theorem ken_change (cost_per_pound : ℕ) (quantity : ℕ) (amount_paid : ℕ) (total_cost : ℕ) (change : ℕ) 
(h1 : cost_per_pound = 7)
(h2 : quantity = 2)
(h3 : amount_paid = 20)
(h4 : total_cost = cost_per_pound * quantity)
(h5 : change = amount_paid - total_cost) : change = 6 :=
by 
  sorry

end ken_change_l566_566599


namespace dolphins_to_be_trained_next_month_l566_566509

-- Definition of conditions
def total_dolphins : ℕ := 20
def fraction_fully_trained := 1 / 4
def fraction_currently_training := 2 / 3

-- Lean 4 statement for the proof problem
theorem dolphins_to_be_trained_next_month :
  let fully_trained := total_dolphins * fraction_fully_trained
  let remaining := total_dolphins - fully_trained
  let currently_training := remaining * fraction_currently_training
  remaining - currently_training = 5 := by
begin
  -- Calculation core based on the given conditions
  let fully_trained := total_dolphins * fraction_fully_trained,
  let remaining := total_dolphins - fully_trained,
  let currently_training := remaining * fraction_currently_training,
  show remaining - currently_training = 5,
  sorry  -- Proof should go here
end

end dolphins_to_be_trained_next_month_l566_566509


namespace mean_home_runs_l566_566129

theorem mean_home_runs :
  let n_5 := 3
  let n_8 := 5
  let n_9 := 3
  let n_11 := 1
  let total_home_runs := 5 * n_5 + 8 * n_8 + 9 * n_9 + 11 * n_11
  let total_players := n_5 + n_8 + n_9 + n_11
  let mean := total_home_runs / total_players
  mean = 7.75 :=
by
  sorry

end mean_home_runs_l566_566129


namespace resultant_force_correct_l566_566522

-- Define the conditions
def P1 : ℝ := 80
def P2 : ℝ := 130
def distance : ℝ := 12.035
def theta1 : ℝ := 125
def theta2 : ℝ := 135.1939

-- Calculate the correct answer
def result_magnitude : ℝ := 209.299
def result_direction : ℝ := 131.35

-- The goal statement to be proved
theorem resultant_force_correct :
  ∃ (R : ℝ) (theta_R : ℝ), 
    R = result_magnitude ∧ theta_R = result_direction := 
sorry

end resultant_force_correct_l566_566522


namespace combined_resistance_l566_566193

theorem combined_resistance (x y r : ℝ) (hx : x = 5) (hy : y = 7) (h_parallel : 1 / r = 1 / x + 1 / y) : 
  r = 35 / 12 := 
by 
  sorry

end combined_resistance_l566_566193


namespace triangle_similarity_l566_566154

variables (A B C O T S B1 C1 : Point)
variables (circleO : circle O)
variables (h_tangentB : tangent circleO B)
variables (h_tangentC : tangent circleO C)
variables (h_intersectT : ∀ (P : Point), on_line (line_through B C) P ↔ P = T)
variables (h_S_on_BC : on_ray B C S)
variables (h_AS_perpendicular_AT : perp (line_through A S) (line_through A T))
variables (h_B1_C1_on_ST : on_ray S T B1 ∧ on_ray S T C1 ∧ BT = B1T ∧ BT = C1T ∧ ∃ S, between C1 B1 S)

theorem triangle_similarity :
  similar (triangle A B C) (triangle A B1 C1) :=
sorry

end triangle_similarity_l566_566154


namespace disinfectant_purchasing_plan_l566_566938

/-- Given conditions:
    - 20 bottles of brand A and 10 bottles of brand B cost 1300 yuan
    - 10 bottles of brand A and 10 bottles of brand B cost 800 yuan
    - The company's budget for disinfectants is not exceeding 1900 yuan for 50 bottles
    - The quantity of brand A disinfectant should be no less than half of the quantity of brand B disinfectant
   Show:
    - The unit price of brand A is 50 yuan and brand B is 30 yuan
    - There are 4 different integer purchasing plans for given constraints.
-/
theorem disinfectant_purchasing_plan :
  ∃ (x y : ℕ), 
    20 * x + 10 * y = 1300 ∧ 
    10 * x + 10 * y = 800 ∧ 
    (∀ a : ℕ, ((50 * a + 30 * (50 - a) ≤ 1900) ∧ (2 * a ≥ 50 - a) → a ∈ {17, 18, 19, 20})) :=
by
  sorry

end disinfectant_purchasing_plan_l566_566938


namespace find_constants_l566_566789

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constants (a : ℝ) (x : ℝ) (h : x ≠ -2) :
  f a (f a x) = x ∧ a = -4 :=
by
  sorry

end find_constants_l566_566789


namespace find_n_l566_566731

theorem find_n (n : ℚ) (h : 7^(5 * n) = (1/7)^(3 * n - 18)) : n = 9 / 4 :=
by 
    sorry

end find_n_l566_566731


namespace total_driving_time_is_40_l566_566964

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l566_566964


namespace product_ABC_sol_l566_566098

theorem product_ABC_sol (A B C : ℚ) : 
  (∀ x : ℚ, x^2 - 20 = A * (x + 2) * (x - 3) + B * (x - 2) * (x - 3) + C * (x - 2) * (x + 2)) → 
  A * B * C = 2816 / 35 := 
by 
  intro h
  sorry

end product_ABC_sol_l566_566098


namespace count_valid_integers_eq_337_l566_566310

open Complex

noncomputable def valid_integer_count : ℕ :=
  finset.card {n | 1 ≤ n ∧ n ≤ 2023 ∧ 
                  ∏ k in finset.range n, ((1 + exp (2 * π * I * (k : ℂ) / n))^n + 1)^2 = 0}

theorem count_valid_integers_eq_337 :
  valid_integer_count = 337 := 
sorry

end count_valid_integers_eq_337_l566_566310


namespace book_problem_part1_book_problem_part2_l566_566541

variables (costA costB : ℝ) (x y W : ℝ)

theorem book_problem_part1 (h1 : costA = 1.5 * costB) 
  (h2 : 540 / costA + 3 = 450 / costB) :
  costA = 45 ∧ costB = 30 :=
by
  sorry

theorem book_problem_part2 (costA costB : ℝ) (x y : ℝ)
  (h1 : costA = 1.5 * costB)
  (h2 : 540 / costA + 3 = 450 / costB)
  (h3 : x + y = 50)
  (h4 : x ≥ y + 6)
  (h5 : W = costA * x + costB * y) :
  x = 28 ∧ y = 22 ∧ W = 1920 :=
by
  sorry

end book_problem_part1_book_problem_part2_l566_566541


namespace quadratic_points_range_l566_566337

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end quadratic_points_range_l566_566337


namespace sum_of_perimeters_l566_566251

theorem sum_of_perimeters (a : ℝ) : 
    ∑' n : ℕ, (3 * a) * (1/3)^n = 9 * a / 2 :=
by sorry

end sum_of_perimeters_l566_566251


namespace ellipse_focal_distance_l566_566990

theorem ellipse_focal_distance :
  let a := 9
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 4 * Real.sqrt 14 :=
by
  sorry

end ellipse_focal_distance_l566_566990


namespace rahim_books_bought_l566_566830

theorem rahim_books_bought (x : ℕ) 
  (first_shop_cost second_shop_cost total_books : ℕ)
  (avg_price total_spent : ℕ)
  (h1 : first_shop_cost = 1500)
  (h2 : second_shop_cost = 340)
  (h3 : total_books = x + 60)
  (h4 : avg_price = 16)
  (h5 : total_spent = first_shop_cost + second_shop_cost)
  (h6 : avg_price = total_spent / total_books) :
  x = 55 :=
by
  sorry

end rahim_books_bought_l566_566830


namespace no_real_or_imaginary_solution_l566_566127

def no_solution (t : ℝ) : Prop := sqrt (36 - t^2) + 6 = 0

theorem no_real_or_imaginary_solution : ∀ t : ℝ, no_solution t → false := 
by 
  intros t h,
  sorry

end no_real_or_imaginary_solution_l566_566127


namespace smallest_n_equals_67_l566_566795

theorem smallest_n_equals_67 :
  let a := Real.pi / 2010
  in (∃ n : ℕ, 0 < n ∧ 2 * ∑ k in Finset.range (n+1), Real.cos (k^2 * a) * Real.sin (k * a) = 0 ∧ n = 67) :=
sorry

end smallest_n_equals_67_l566_566795


namespace multiples_of_15_not_45_count_l566_566603

theorem multiples_of_15_not_45_count :
  let nums_15 := {n | 100 ≤ n ∧ n ≤ 999 ∧ n % 15 = 0},
      nums_45 := {n | 100 ≤ n ∧ n ≤ 999 ∧ n % 45 = 0} in
  (nums_15.card - nums_45.card = 40) :=
by
  -- Definitions of sequences
  let nums_15 := finset.range (999 - 105 + 1).filter (λ n, (n + 105) % 15 = 0),
      nums_45 := finset.range (999 - 135 + 1).filter (λ n, (n + 135) % 45 = 0)
  
  -- Calculation for nums_15 (multiples of 15)
  have nums_15_length : (nums_15.card = 60),
  {
    -- Number of terms in sequence starting at 105 and ending at 990 with common difference 15
    sorry
  },
  
  -- Calculation for nums_45 (multiples of 45)
  have nums_45_length : (nums_45.card = 20),
  {
    -- Number of terms in sequence starting at 135 and ending at 990 with common difference 45
    sorry
  },

  -- Complete proof
  show (nums_15.card - nums_45.card) = 40,
  from calc
    nums_15.card - nums_45.card = 60 - 20 : by { rw [nums_15_length, nums_45_length] }
                         ... = 40 : by norm_num

end multiples_of_15_not_45_count_l566_566603


namespace minimal_sum_of_initial_terms_l566_566983

theorem minimal_sum_of_initial_terms (a b : ℕ)
  (h : ∃ n : ℕ, x n = 1000) 
  (x : ℕ → ℕ) 
  (h1 : x 1 = a) 
  (h2 : x 2 = b) 
  (h3 : ∀ n : ℕ, x (n + 2) = x n + x (n + 1)) : 
  a + b = 10 :=
by 
  sorry

end minimal_sum_of_initial_terms_l566_566983


namespace overlapping_parts_length_l566_566186

def length_of_each_overlapping_part (L l : ℕ) (n : ℕ) : ℕ :=
  (l - L) / (n - 1)

theorem overlapping_parts_length (L l : ℕ) (n : ℕ) (h1 : l = 4 * 250) (h2 : L = 925) (h3 : n = 4) :
  length_of_each_overlapping_part L l n = 25 :=
by
  unfold length_of_each_overlapping_part
  rw [h1, h2, h3]
  simp
  sorry

end overlapping_parts_length_l566_566186


namespace count_distinct_digits_l566_566581

theorem count_distinct_digits (n : ℕ) (h1 : ∃ (n : ℕ), n^3 = 125) : 
  n = 5 :=
by
  sorry

end count_distinct_digits_l566_566581


namespace exists_int_l_l566_566441

theorem exists_int_l (m n : ℕ)
  (h : ∀ k : ℕ, Nat.gcd (11 * k - 1) m = Nat.gcd (11 * k - 1) n) :
  ∃ l : ℤ, m = 11^l * n := 
by
  sorry

end exists_int_l_l566_566441


namespace log_sum_eq_ten_l566_566037

noncomputable theory
open_locale noncomputable

-- Definitions for the problem
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable h_geom : ∀ n, a (n + 1) = r * a n
variable h_pos : ∀ n, a n > 0
variable h_condition : a 5 * a 6 + a 4 * a 7 = 18

-- Goal to prove
theorem log_sum_eq_ten :
  ∑ i in finset.range 10, real.logb 3 (a (i + 1)) = 10 :=
sorry

end log_sum_eq_ten_l566_566037


namespace thickness_of_layer_l566_566237

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem thickness_of_layer (radius_sphere radius_cylinder : ℝ) (volume_sphere volume_cylinder : ℝ) (h : ℝ) : 
  radius_sphere = 3 → 
  radius_cylinder = 10 →
  volume_sphere = volume_of_sphere radius_sphere →
  volume_cylinder = volume_of_cylinder radius_cylinder h →
  volume_sphere = volume_cylinder → 
  h = 9 / 25 :=
by
  intros
  sorry

end thickness_of_layer_l566_566237


namespace monthly_rent_of_new_apartment_l566_566422

theorem monthly_rent_of_new_apartment 
  (former_rent_per_sqft : ℝ) 
  (former_sqft : ℝ) 
  (yearly_savings : ℝ) 
  (former_rent_per_sqft = 2) 
  (former_sqft = 750) 
  (yearly_savings = 1200) : 
  (former_rent_per_sqft * former_sqft * 12 - yearly_savings) / 12 = 1400 :=
by
  sorry

end monthly_rent_of_new_apartment_l566_566422


namespace fraction_of_upgraded_sensors_is_one_seventh_l566_566926

theorem fraction_of_upgraded_sensors_is_one_seventh
  (N U : ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < 24 → nonUpgradedSensors i = N)
  (h2 : N = U / 4) :
  (U / (24 * N + U)) = 1 / 7 :=
by
  sorry

end fraction_of_upgraded_sensors_is_one_seventh_l566_566926


namespace domain_of_f_l566_566480

theorem domain_of_f :
  {x : ℝ | 5 - x > 0 ∧ 2^x - 1 ≥ 0} = Set.Ico 0 5 :=
by
  sorry

end domain_of_f_l566_566480


namespace units_digit_sum_l566_566506

theorem units_digit_sum :
  let prod_2016_2017 := (Prod (fun _ => 2017) (Finset.range 2016)),
      prod_2017_2016 := (Prod (fun _ => 2016) (Finset.range 2017)) in
  (prod_2016_2017 % 10 + prod_2017_2016 % 10) % 10 = 7 :=
by
  let prod_2016_2017 := (List.prod (List.replicate 2016 2017))
  let prod_2017_2016 := (List.prod (List.replicate 2017 2016))
  have h1 : prod_2016_2017 % 10 = 1 := sorry
  have h2 : prod_2017_2016 % 10 = 6 := sorry
  exact (h1 + h2) % 10


end units_digit_sum_l566_566506


namespace fixed_point_exists_l566_566901

open EuclideanGeometry

def exists_equal_distance_point (circle1 : Circle) (circle2 : Circle) (A : Point) (t : ℝ) (P1 : ℝ → Point) (P2 : ℝ → Point) : Prop :=
  (∀ t, (P1 t ∈ circle1 ∧ P2 t ∈ circle2) ∧ dist A P1(t) = dist A P2(t)) →
  ∃ P : Point, ∀ t, dist P (P1 t) = dist P (P2 t)

theorem fixed_point_exists (circle1 circle2 : Circle) (A : Point) (t : ℝ) (P1 P2 : ℝ → Point) :
  (P1 0 = A) ∧ (P2 0 = A) ∧ (∀ t, P1 t ∈ circle1) ∧ (∀ t, P2 t ∈ circle2) ∧ 
  (∃ T, ∀ t, P1(t + T) = P1(t) ∧ P2(t + T) = P2(t)) →
  exists_equal_distance_point circle1 circle2 A t P1 P2 :=
sorry

end fixed_point_exists_l566_566901


namespace tank_capacity_l566_566225

variable (C : ℝ)

noncomputable def leak_rate := C / 6 -- litres per hour
noncomputable def inlet_rate := 6 * 60 -- litres per hour
noncomputable def net_emptying_rate := C / 12 -- litres per hour

theorem tank_capacity : 
  (360 - leak_rate C = net_emptying_rate C) → 
  C = 1440 :=
by 
  sorry

end tank_capacity_l566_566225


namespace cyclic_quadrilateral_angle_difference_l566_566276

theorem cyclic_quadrilateral_angle_difference {A B C D P Q : Type*} 
  [CyclicQuadrilateral ABCD]
  (h1 : ∠ ABD = 70°)
  (h2 : ∠ ADB = 50°)
  (h3 : BC = CD)
  (h4 : ∀ P, ∃ P, P ∈ AB ∩ CD)
  (h5 : ∀ Q, ∃ Q, Q ∈ AD ∩ BC) :
  ∠ APQ - ∠ AQP = 20° := 
sorry

end cyclic_quadrilateral_angle_difference_l566_566276


namespace solve_change_in_rate_of_profit_l566_566929

noncomputable def change_in_rate_of_profit (P r R : ℝ) : Prop :=
  let capital_a := 10000
  let original_profit_share_a := (2 / 3) * P
  let new_profit_share_a := (2 / 3) * P + 200
  let original_amount := capital_a * r
  let new_amount := capital_a * (r + R)
  original_profit_share_a = original_amount ∧ new_profit_share_a = new_amount ∧ R = 0.02

theorem solve_change_in_rate_of_profit :
  ∀ (P r R : ℝ), change_in_rate_of_profit P r R :=
by
  intros
  let capital_a := 10000
  let original_profit_share_a := (2 / 3) * P
  let new_profit_share_a := (2 / 3) * P + 200
  let original_amount := capital_a * r
  let new_amount := capital_a * (r + R)
  have h1 : original_profit_share_a = original_amount := sorry
  have h2 : new_profit_share_a = new_amount := sorry
  have h3 : R = 0.02 := sorry
  exact ⟨h1, h2, h3⟩

end solve_change_in_rate_of_profit_l566_566929


namespace condition_sufficient_and_necessary_l566_566326

theorem condition_sufficient_and_necessary (x1 x2 : ℝ) :
  (f x1 * f x2 < 1) ↔ (x1 + x2 > 0) :=
by
  let f := λ x : ℝ, (1 / 2) ^ x
  sorry

end condition_sufficient_and_necessary_l566_566326


namespace expected_winnings_l566_566970

theorem expected_winnings (roll_1_2: ℝ) (roll_3_4: ℝ) (roll_5_6: ℝ) (p1_2 p3_4 p5_6: ℝ) :
    roll_1_2 = 2 →
    roll_3_4 = 4 →
    roll_5_6 = -6 →
    p1_2 = 1 / 8 →
    p3_4 = 1 / 4 →
    p5_6 = 1 / 8 →
    (2 * p1_2 + 2 * p1_2 + 4 * p3_4 + 4 * p3_4 + roll_5_6 * p5_6 + roll_5_6 * p5_6) = 1 := by
  intros
  sorry

end expected_winnings_l566_566970


namespace find_possible_numbers_l566_566206

theorem find_possible_numbers (N : ℕ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9)
  (cond : nat.num_digits N = 200 ∧ N = 10^199 * 5 * a ∧ 10^199 * 5 * a < 10^200) :
  (∃ a, N = 125 * a * 10^197 ∧ (1 ≤ a ∧ a ≤ 3)) :=
by
  sorry

end find_possible_numbers_l566_566206


namespace max_hardcover_books_l566_566403

-- Define the conditions as provided in the problem
def total_books : ℕ := 36
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n

-- The logical statement we need to prove
theorem max_hardcover_books :
  ∃ h : ℕ, (∃ c : ℕ, is_composite c ∧ 2 * h + c = total_books) ∧ 
  ∀ h' c', is_composite c' ∧ 2 * h' + c' = total_books → h' ≤ h :=
sorry

end max_hardcover_books_l566_566403


namespace number_of_solutions_l566_566382

theorem number_of_solutions (x y : ℕ) : 
  (∃ n : ℕ, n = 2625 ∧ (x, y) ∈ { (a, b) : ℤ × ℤ | 1 ≤ a ∧ a ≤ 75 ∧ 1 ≤ b ∧ b ≤ 70 ∧ 76 * b + 71 * a < 76 * 71} ∧ 1 ≤ x ∧ x ≤ 75 ∧ 1 ≤ y ∧ y ≤ 70 ∧ 76 * y + 71 * x < 76 * 71)
  ∧ (∀ (x y : ℕ), 76 * y + 71 * x < 76 * 71)
:= by
  sorry

end number_of_solutions_l566_566382


namespace max_shelves_within_rearrangement_l566_566261

theorem max_shelves_within_rearrangement (k : ℕ) :
  (∀ books : finset ℕ, books.card = 1300 →
    (∃ shelf_assignment_before shelf_assignment_after : finset ℕ → ℕ,
      (∀ book, shelf_assignment_before book ≤ k ∧ shelf_assignment_after book ≤ k) ∧
      (∃ shelf, (books.filter (λ book, shelf_assignment_before book = shelf)).card ≥ 5 ∧
                (books.filter (λ book, shelf_assignment_after book = shelf)).card ≥ 5)
    )
  ) → k ≤ 18 := sorry

end max_shelves_within_rearrangement_l566_566261


namespace smallest_positive_angle_cos_eq_l566_566300

theorem smallest_positive_angle_cos_eq:
  let θ : ℝ := 21 in
  cos θ = (sin 45 + cos 60 - sin 30 - cos 24) :=
  by
    sorry

end smallest_positive_angle_cos_eq_l566_566300


namespace evaluate_replaced_expression_at_2_l566_566035

/-- Definition of the initial expression before replacement --/
def initial_expression (x : ℝ) : ℝ := (x + 2) / (x - 2)

/-- Definition of the replaced expression, where x is replaced by the initial expression --/
def replaced_expression (x : ℝ) : ℝ := (initial_expression x + 2) / (initial_expression x - 2)

/-- Theorem: Evaluating the replaced expression at x = 2 results in 1 --/
theorem evaluate_replaced_expression_at_2 : replaced_expression 2 = 1 :=
by sorry

end evaluate_replaced_expression_at_2_l566_566035


namespace probability_of_same_color_l566_566158

noncomputable theory
open ProbabilityTheory

-- Define the setup of the problem
def bagA : Finset (String × ℕ) := {("white", 1), ("red", 2), ("black", 3)}
def bagB : Finset (String × ℕ) := {("white", 2), ("red", 3), ("black", 1)}

-- Define the event of drawing a ball of each color
def event (c : String) : Finset (String × String) :=
  ({c} ×ˢ Finset.image (λ (b : String × ℕ), b.1) bagA) ∩ ({c} ×ˢ Finset.image (λ (b : String × ℕ), b.1) bagB)

-- Define the possible outcomes
def possible_outcomes : Finset (String × String) :=
  Finset.product (Finset.image (λ (b : String × ℕ), b.1) bagA) (Finset.image (λ (b : String × ℕ), b.1) bagB)

-- Define the event of same color outcome
def same_color_event : Finset (String × String) :=
  event "white" ∪ event "red" ∪ event "black"

-- Calculate the probability of the same color event
def prob_same_color : ℚ :=
  (same_color_event.card : ℚ) / (possible_outcomes.card : ℚ)

-- Proof statement
theorem probability_of_same_color : prob_same_color = 11 / 36 := by
  sorry

end probability_of_same_color_l566_566158


namespace linear_function_solution_l566_566203

theorem linear_function_solution (f : ℝ → ℝ) (h1 : ∀ x, f (f x) = 16 * x - 15) :
  (∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5) :=
sorry

end linear_function_solution_l566_566203


namespace area_excluding_garden_proof_l566_566405

noncomputable def area_land_excluding_garden (length width r : ℝ) : ℝ :=
  let area_rec := length * width
  let area_circle := Real.pi * (r ^ 2)
  area_rec - area_circle

theorem area_excluding_garden_proof :
  area_land_excluding_garden 8 12 3 = 96 - 9 * Real.pi :=
by
  unfold area_land_excluding_garden
  sorry

end area_excluding_garden_proof_l566_566405


namespace range_of_m_common_tangents_with_opposite_abscissas_l566_566798

section part1
variable {x : ℝ}

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def h (m : ℝ) (x : ℝ) := m * f x / Real.sin x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 0 Real.pi, h m x ≥ Real.sqrt 2) ↔ m ∈ Set.Ici (Real.sqrt 2 / Real.exp (Real.pi / 4)) := 
by
  sorry
end part1

section part2
variable {x : ℝ}

noncomputable def g (x : ℝ) := Real.log x
noncomputable def f_tangent_line_at (x₁ : ℝ) (x : ℝ) := Real.exp x₁ * x + (1 - x₁) * Real.exp x₁
noncomputable def g_tangent_line_at (x₂ : ℝ) (x : ℝ) := x / x₂ + Real.log x₂ - 1

theorem common_tangents_with_opposite_abscissas :
  ∃ x₁ x₂ : ℝ, (f_tangent_line_at x₁ = g_tangent_line_at (Real.exp (-x₁))) ∧ (x₁ = -x₂) :=
by
  sorry
end part2

end range_of_m_common_tangents_with_opposite_abscissas_l566_566798


namespace greatest_a_for_integer_solutions_l566_566119

theorem greatest_a_for_integer_solutions :
  ∃ a : ℕ, 
    (∀ x : ℤ, x^2 + a * x = -21 → ∃ y : ℤ, y * (y + a) = -21) ∧ 
    ∀ b : ℕ, (∀ x : ℤ, x^2 + b * x = -21 → ∃ y : ℤ, y * (y + b) = -21) → b ≤ a :=
begin
  -- Proof goes here
  sorry
end

end greatest_a_for_integer_solutions_l566_566119


namespace ab_is_zero_l566_566684

-- Define that a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b - 2

-- The main theorem to prove
theorem ab_is_zero (a b : ℝ) (h_odd : is_odd (f a b)) : a * b = 0 := 
sorry

end ab_is_zero_l566_566684


namespace can_accommodate_20_chickens_80_chicks_cannot_accommodate_30_chickens_100_chicks_max_chickens_and_chicks_max_chicks_l566_566214

-- Define the constraints
def area_constraint (x y : ℕ) : Prop :=
  2 * x + y = 120

def feed_constraint (x y : ℕ) : Prop :=
  4 * x + y ≤ 200

-- Prove the given conditions
theorem can_accommodate_20_chickens_80_chicks : ∃ (x y : ℕ), x = 20 ∧ y = 80 ∧ area_constraint x y ∧ feed_constraint x y :=
  by { use [20, 80], simp [area_constraint, feed_constraint], norm_num }

theorem cannot_accommodate_30_chickens_100_chicks : ¬ ∃ (x y : ℕ), x = 30 ∧ y = 100 ∧ area_constraint x y :=
  by { simp [area_constraint], intro h, cases h with x hx, cases hx with y hy, 
        cases hy with h1 h2, cases h2 with h3 h4, 
        rw [h1, h1] at h3, norm_num at h3, contradiction }

theorem max_chickens_and_chicks : ∃ (x y : ℕ), x = 40 ∧ y = 40 ∧ area_constraint x y ∧ feed_constraint x y :=
  by { use [40, 40], simp [area_constraint, feed_constraint], norm_num }

theorem max_chicks : ∃ (x y : ℕ), x = 0 ∧ y = 120 ∧ area_constraint x y ∧ feed_constraint x y :=
  by { use [0, 120], simp [area_constraint, feed_constraint], norm_num }

end can_accommodate_20_chickens_80_chicks_cannot_accommodate_30_chickens_100_chicks_max_chickens_and_chicks_max_chicks_l566_566214


namespace water_per_drop_l566_566857

def drips_per_minute : ℕ := 10
def time_in_minutes : ℕ := 60
def total_water_in_hour : ℕ := 30

theorem water_per_drop : 
  (drips_per_minute * time_in_minutes = 600) → 
  (total_water_in_hour / 600 = 0.05) := 
by
  sorry

end water_per_drop_l566_566857


namespace bankers_gain_is_126_l566_566851

-- Define the given conditions
def present_worth : ℝ := 600
def interest_rate : ℝ := 0.10
def time_period : ℕ := 2

-- Define the formula for compound interest to find the amount due A
def amount_due (PW : ℝ) (R : ℝ) (T : ℕ) : ℝ := PW * (1 + R) ^ T

-- Define the banker's gain as the difference between the amount due and the present worth
def bankers_gain (A : ℝ) (PW : ℝ) : ℝ := A - PW

-- The theorem to prove that the banker's gain is Rs. 126 given the conditions
theorem bankers_gain_is_126 : bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 126 := by
  sorry

end bankers_gain_is_126_l566_566851


namespace total_waiting_time_l566_566814

def t1 : ℕ := 20
def t2 : ℕ := 4 * t1 + 14
def T : ℕ := t1 + t2

theorem total_waiting_time : T = 114 :=
by {
  -- Preliminary calculations and justification would go here
  sorry
}

end total_waiting_time_l566_566814


namespace find_n_l566_566062

   theorem find_n (n : ℕ) : 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (n = 34 ∨ n = 37) :=
   by
     intros
     sorry
   
end find_n_l566_566062


namespace radius_of_largest_sphere_l566_566987

-- Definitions for the conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 5
def circle_center : ℝ × ℝ × ℝ := (4, 0, 1)
def circle_radius : ℝ := 1
def sphere_center : ℝ × ℝ × ℝ := (0, 0, 4)

-- Proof statement
theorem radius_of_largest_sphere : 
  (∀ (r : ℝ), r = 4 ↔ 
    (dist ((4:ℝ),0,1) (0,0,r) ^ 2 = dist 4 (r - 1) ^ 2 + dist (r + 1) ^ 2)) :=
by
  sorry

end radius_of_largest_sphere_l566_566987


namespace collinear_A_B_G_l566_566687

-- Definitions for circles, given two points of intersections
variable {O₁ O₂ O₃ : Point}
variable {A B C D E F G : Point}

-- Given conditions
axiom circles_intersection : ∀ (O₁ O₂ : Circle) (A B : Point), A ∈ O₁ ∧ A ∈ O₂ ∧ B ∈ O₁ ∧ B ∈ O₂
axiom circle_tangent_point1 : ∀ (O₁ O₃ : Circle) (C : Point), C ∈ O₁ ∧ C ∉ O₃ ∧ ¬Collinear O₁ C O₁
axiom circle_tangent_point2 : ∀ (O₂ O₃ : Circle) (D : Point), D ∈ O₂ ∧ D ∉ O₃ ∧ ¬Collinear O₂ D O₂
axiom line_tangent_point1 : ∀ (O₁ : Circle) (E : Point) (F : Point) (EF : Line), E ∈ O₁ ∧ F ∈ O₁ ∧ E ∈ EF ∧ F ∈ EF
axiom line_tangent_point2 : ∀ (O₂ : Circle) (E : Point) (F : Point) (EF : Line), F ∈ O₂ ∧ E ∈ EF ∧ F ∈ EF
axiom intersection_lines : ∀ (C E D F G : Point), G ∈ line C E ∧ G ∈ line D F

-- The theorem to prove
theorem collinear_A_B_G :
  ∀ (O₁ O₂ : Circle) (A B C D E F G : Point),
  circles_intersection O₁ O₂ A B →
  circle_tangent_point1 O₁ O₃ C →
  circle_tangent_point2 O₂ O₃ D →
  line_tangent_point1 O₁ E F EF →
  line_tangent_point2 O₂ E F EF →
  intersection_lines C E D F G →
  Collinear A B G := by
  sorry

end collinear_A_B_G_l566_566687


namespace piggy_bank_exceed_five_dollars_l566_566424

noncomputable def sequence_sum (n : ℕ) : ℕ := 2^n - 1

theorem piggy_bank_exceed_five_dollars (n : ℕ) (start_day : Nat) (day_of_week : Fin 7) :
  ∃ (n : ℕ), sequence_sum n > 500 ∧ n = 9 ∧ (start_day + n) % 7 = 2 := 
sorry

end piggy_bank_exceed_five_dollars_l566_566424


namespace problem_statement_l566_566328

theorem problem_statement (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  2014 = (a^2 + b^2) * (c^3 - d^3) :=
by
  -- Definitions from the problem
  let a := 5
  let b := 9
  let c := 3
  let d := 2
  
  -- Assertions from the problem conditions
  have h1 : 5^2 + 9^2 = 106 := by norm_num
  have h2 : 3^3 - 2^3 = 19 := by norm_num
  have h3 : 106 * 19 = 2014 := by norm_num

  -- Final assertion proving the given statement
  exact h3

end problem_statement_l566_566328


namespace total_number_of_trees_l566_566516

variable {T : ℕ} -- Define T as a natural number (total number of trees)
variable (h1 : 70 / 100 * T + 105 = T) -- Indicates 30% of T is 105

theorem total_number_of_trees (h1 : 70 / 100 * T + 105 = T) : T = 350 :=
by
sorry

end total_number_of_trees_l566_566516


namespace sum_x_coords_T_l566_566230

variable (Q : List (ℝ × ℝ))
variable (x_coords : List ℝ)
variable (vertices : ℕ)

-- Given conditions
def condition_1 : Prop := vertices = 39
def condition_2 : Prop := x_coords.length = vertices
def sum_x_coords_condition : Prop := x_coords.sum = 117

-- Derived midpoint vertices (using midpoints logic applied iteratively three times)
def midpoint_vertices (coords : List ℝ) : List ℝ :=
  List.map₂ (fun x1 x2 => (x1 + x2) / 2) coords (List.tail coords ++ [List.head coords])

def R_x_coords : List ℝ := midpoint_vertices x_coords
def S_x_coords : List ℝ := midpoint_vertices R_x_coords
def T_x_coords : List ℝ := midpoint_vertices S_x_coords

-- Theorem: Sum of x-coordinates of vertices of T is 117
theorem sum_x_coords_T (h1 : condition_1) (h2 : condition_2) (h3 : sum_x_coords_condition) : T_x_coords.sum = 117 := 
  sorry

end sum_x_coords_T_l566_566230


namespace find_phase_and_vertical_shift_l566_566299

def phase_shift (C B : ℝ) := C / B

def vertical_shift (D : ℝ) := D

theorem find_phase_and_vertical_shift (A B C D : ℝ) (h: A = 3 ∧ B = 3 ∧ C = -π/4 ∧ D = 1) :
  phase_shift C B = -π/12 ∧ vertical_shift D = 1 :=
by
  sorry

end find_phase_and_vertical_shift_l566_566299


namespace exists_angles_of_inclination_l566_566846

variables (α0 β0 γ0 α1 β1 γ1 θa θb θc : ℝ)
variables (a0 b0 c0 a1 b1 c1 : ℝ)

-- Assume the angles of the original triangle. 
axiom angles_original : α0 + β0 + γ0 = 180

-- Assume the angles of the projected triangle.
axiom angles_projection : α1 + β1 + γ1 = 180

-- Definition of the angles of inclination
def angles_of_inclination (α0 β0 γ0 α1 β1 γ1 θa θb θc : ℝ) : Prop := 
  ∃ θa θb θc, 
    α0 = α1 ∧ 
    β0 = β1 ∧ 
    γ0 = γ1

theorem exists_angles_of_inclination (α0 β0 γ0 α1 β1 γ1 : ℝ) :
  angles_of_inclination α0 β0 γ0 α1 β1 γ1 θa θb θc :=
sorry

end exists_angles_of_inclination_l566_566846


namespace range_of_a_l566_566009

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x > x ^ 2 + a) → a < -8 :=
by sorry

end range_of_a_l566_566009


namespace sum_of_series_l566_566269

theorem sum_of_series :
  3 * ∑ n in Finset.range(31).map((+(-15)).toEmbedding), (-1)^n = 3 :=
by
  sorry

end sum_of_series_l566_566269


namespace can_measure_all_weights_l566_566518

theorem can_measure_all_weights (a b c : ℕ) 
  (h_sum : a + b + c = 10) 
  (h_unique : (a = 1 ∧ b = 2 ∧ c = 7) ∨ (a = 1 ∧ b = 3 ∧ c = 6)) : 
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 10 → 
    ∃ (k l m : ℤ), w = k * a + l * b + m * c ∨ w = k * -a + l * -b + m * -c :=
  sorry

end can_measure_all_weights_l566_566518


namespace cruise_liner_travelers_l566_566885

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l566_566885


namespace hexagon_side_length_l566_566026

theorem hexagon_side_length (d : ℝ) (h_hex : d = 20) : ∃ s : ℝ, s = 10 :=
by
  use 10
  sorry

end hexagon_side_length_l566_566026


namespace thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l566_566200

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l566_566200


namespace final_position_of_bug_l566_566563

/-- Define the function x that represents the horizontal movement of the bug -/
noncomputable def bug_horizontal_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

/-- Define the function y that represents the vertical movement of the bug -/
noncomputable def bug_vertical_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

/-- Prove that the bug's final position converges to (4/5, 2/5) given its movement pattern -/
theorem final_position_of_bug : 
  let hx := bug_horizontal_sum 1 (-1/4)
  let hy := bug_vertical_sum 1/2 (-1/4)
  (hx, hy) = (4/5, 2/5) :=
begin
  have hx_eq  := bug_horizontal_sum 1 (-1/4),
  have hy_eq := bug_vertical_sum 1/2 (-1/4),
  simp [bug_horizontal_sum, bug_vertical_sum] at hx_eq hy_eq,
  sorry,
end

end final_position_of_bug_l566_566563


namespace ratio_of_members_l566_566994

theorem ratio_of_members (r p : ℕ) (h1 : 5 * r + 12 * p = 8 * (r + p)) : (r / p : ℚ) = 4 / 3 := by
  sorry -- This is a placeholder for the actual proof.

end ratio_of_members_l566_566994


namespace probability_one_hits_l566_566524

theorem probability_one_hits 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 1 / 2) (hB : p_B = 1 / 3):
  p_A * (1 - p_B) + (1 - p_A) * p_B = 1 / 2 := by
  sorry

end probability_one_hits_l566_566524


namespace sales_worth_l566_566980

def old_scheme_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_scheme_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)
def remuneration_difference (S : ℝ) : ℝ := new_scheme_remuneration S - old_scheme_remuneration S

theorem sales_worth (S : ℝ) (h : remuneration_difference S = 600) : S = 24000 :=
by
  sorry

end sales_worth_l566_566980


namespace georgina_teaches_2_phrases_per_week_l566_566661

theorem georgina_teaches_2_phrases_per_week
    (total_phrases : ℕ) 
    (initial_phrases : ℕ) 
    (days_owned : ℕ)
    (phrases_per_week : ℕ):
    total_phrases = 17 → 
    initial_phrases = 3 → 
    days_owned = 49 → 
    phrases_per_week = (total_phrases - initial_phrases) / (days_owned / 7) → 
    phrases_per_week = 2 := 
by
  intros h_total h_initial h_days h_calc
  rw [h_total, h_initial, h_days] at h_calc
  sorry  -- Proof to be filled

end georgina_teaches_2_phrases_per_week_l566_566661


namespace number_of_subsets_modulo_2048_is_2_pow_1994_l566_566298

def S : set ℕ := { x | 1 ≤ x ∧ x ≤ 2005 }
def A : set ℕ := { x | ∃ n, n ≤ 10 ∧ x = 2^n }

theorem number_of_subsets_modulo_2048_is_2_pow_1994 :
  {B : set ℕ // B ⊆ S ∧ (∑ b in B, b) % 2048 = 2006}.card = 2^1994 :=
sorry

end number_of_subsets_modulo_2048_is_2_pow_1994_l566_566298


namespace find_all_k_l566_566791

variables {R : Type*} [CommRing R]
variables {P : R[X]} {n : ℕ} {a_n : R} {a_i : R}
variables {α : Fin n.succ → R} {k : R}

-- Conditions: 
-- P has degree n ≥ 5 with integer coefficients and n different integer roots {0, α_2, ..., α_n}
noncomputable def P_form (P : R[X]) (α : Fin n.succ → R) (h_coeff : ∀ i, a_i ∈ ℤ) (h_deg : P.degree = n) (h_roots : ∀ i, P.eval (α i) = 0) :=
  -- P(x) = a_n * x * (x - α_2) * (x - α_3) * ... * (x - α_n)
  ∀ x, P.eval x = a_n * x * ∏ i in Fin.range' 1 n, (x - C (α i))

-- Theorem statement: 
-- Find all k such that P(P(k)) = 0 given the polynomial's properties and roots
theorem find_all_k (h_form : P_form P α (λ i, a_i ∈ ℤ) n h_roots) :
  (P.eval (P.eval k) = 0) ↔ (k ∈ {0, α 1, α 2, ..., α n}) := 
begin
  sorry
end

end find_all_k_l566_566791


namespace Theo_wins_game_l566_566598

theorem Theo_wins_game (p : ℕ) (hp : prime p) (hp2 : p > 2) :
  ∃ strategy : (ℕ → ℕ) → ℕ → ℕ, ∀ a : ℕ, a > 0 → ∃ l : ℕ, 
  (∃ b : ℕ, b > a ∧ s_l (a, b) = a^b + b^a ∧ divides p (S_l (l, s_l))) → 
  (strategy = Theo_winning_strategy) :=
sorry

noncomputable def s_l (a b : ℕ) : ℕ := a^b + b^a

noncomputable def S_l (l : ℕ) (s_l : ℕ → ℕ × ℕ → ℕ) : ℕ := 
  ∑ i in range (l + 1), (i + 1) * s_l i

noncomputable def Theo_winning_strategy : (ℕ → ℕ) → ℕ → ℕ := sorry

end Theo_wins_game_l566_566598


namespace sum_of_solutions_l566_566918

theorem sum_of_solutions :
  (∀ x : ℝ, (8 * x / 40 = 7 / x) → x ∈ {√35, -√35}) →
  √35 + (-√35) = 0 := by
  sorry

end sum_of_solutions_l566_566918


namespace centers_of_rectangles_on_intersecting_circles_l566_566714

noncomputable def set_of_centers_of_rectangles (O1 O2 : ℝ × ℝ) (R1 R2 : ℝ) : set (ℝ × ℝ) :=
let A := (* compute point A from O1 and O2 and radii R1 R2 *)
let B := (* compute point B from O1 and O2 and radii R1 R2 *)
let C := (* compute point C from O1 and O2 and radii R1 R2 *)
let D := (* compute point D from O1 and O2 and radii R1 R2 *)
let midpoint (p1 p2 : ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
let interval1 := set.Icc (midpoint A B) (midpoint A D)
let interval2 := set.Icc (midpoint B C) (midpoint C D)
let interval3 := set.Icc (midpoint O1 O2) (midpoint O1 O2) -- excluding the common chord midpoint
(interval1 ∪ interval2 ∪ interval3) \ {midpoint O1 O2}

theorem centers_of_rectangles_on_intersecting_circles (O1 O2 : ℝ × ℝ) (R1 R2 : ℝ) :
    ∃ (s : set (ℝ × ℝ)), s = set_of_centers_of_rectangles O1 O2 R1 R2 :=
begin
  sorry
end

end centers_of_rectangles_on_intersecting_circles_l566_566714


namespace smallest_solution_x4_minus_40x2_plus_400_eq_zero_l566_566917

theorem smallest_solution_x4_minus_40x2_plus_400_eq_zero :
  ∃ x : ℝ, (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_40x2_plus_400_eq_zero_l566_566917


namespace minimum_value_of_fx_l566_566538

theorem minimum_value_of_fx : ∀ x : ℝ, (0 < x ∧ x < π / 4) → 
  (∃ f, f = (cos x)^2 / (cos x * sin x - (sin x)^2) ∧ 4 ≤ f) :=
by
  sorry

end minimum_value_of_fx_l566_566538


namespace median_inequalities_l566_566858

-- Define the context and variables for the problem
variables {a b c s_a s_b : ℝ}

-- Define the conditions
-- Triangle is a right triangle with legs 'a', 'b', and hypotenuse 'c'
axiom right_triangle (h : a^2 + b^2 = c^2)

-- Median lengths to the legs 'a' and 'b' are 's_a' and 's_b' respectively
axiom median_to_leg_a (h₁ : s_a = sqrt ((3/4) * a^2 + b^2))
axiom median_to_leg_b (h₂ : s_b = sqrt ((3/4) * b^2 + a^2))

-- Theorem statement
theorem median_inequalities : 3 / 2 * c < s_a + s_b ∧ s_a + s_b <= sqrt (10) / 2 * c :=
by
  sorry

end median_inequalities_l566_566858


namespace probability_solved_l566_566548

theorem probability_solved (pA pB pA_and_B : ℚ) :
  pA = 2 / 3 → pB = 3 / 4 → pA_and_B = (2 / 3) * (3 / 4) →
  pA + pB - pA_and_B = 11 / 12 :=
by
  intros hA hB hA_and_B
  rw [hA, hB, hA_and_B]
  sorry

end probability_solved_l566_566548


namespace triangle_hypotenuse_segments_l566_566484

theorem triangle_hypotenuse_segments :
  ∀ (x : ℝ) (BC AC : ℝ),
  BC / AC = 3 / 7 →
  ∃ (h : ℝ) (BD AD : ℝ),
    h = 42 ∧
    BD * AD = h^2 ∧
    BD / AD = 9 / 49 ∧
    BD = 18 ∧
    AD = 98 :=
by
  sorry

end triangle_hypotenuse_segments_l566_566484


namespace bad_numbers_less_than_100_l566_566232

def is_bad (n : ℕ) : Prop :=
  n = 1 ∨ (∀ p q : ℕ, (p > 1 ∧ q > 1 ∧ p * q = n) → p = q)

def bad_numbers_count (limit : ℕ) : ℕ :=
  Nat.card (Set.filter is_bad (Finset.range limit).val)

theorem bad_numbers_less_than_100 : bad_numbers_count 100 = 30 :=
by {
  sorry
}

end bad_numbers_less_than_100_l566_566232


namespace greatest_remainder_when_dividing_by_10_l566_566719

theorem greatest_remainder_when_dividing_by_10 (x : ℕ) : 
  ∃ r : ℕ, r < 10 ∧ r = x % 10 ∧ r = 9 :=
by
  sorry

end greatest_remainder_when_dividing_by_10_l566_566719


namespace number_of_solutions_l566_566383

theorem number_of_solutions (x y : ℕ) : 
  (∃ n : ℕ, n = 2625 ∧ (x, y) ∈ { (a, b) : ℤ × ℤ | 1 ≤ a ∧ a ≤ 75 ∧ 1 ≤ b ∧ b ≤ 70 ∧ 76 * b + 71 * a < 76 * 71} ∧ 1 ≤ x ∧ x ≤ 75 ∧ 1 ≤ y ∧ y ≤ 70 ∧ 76 * y + 71 * x < 76 * 71)
  ∧ (∀ (x y : ℕ), 76 * y + 71 * x < 76 * 71)
:= by
  sorry

end number_of_solutions_l566_566383


namespace grazing_area_proof_l566_566985

noncomputable def grazing_area (s r : ℝ) : ℝ :=
  let A_circle := 3.14 * r^2
  let A_sector := (300 / 360) * A_circle
  let A_triangle := (1.732 / 4) * s^2
  let A_triangle_part := A_triangle / 3
  let A_grazing := A_sector - A_triangle_part
  3 * A_grazing

theorem grazing_area_proof : grazing_area 5 7 = 136.59 :=
  by
  sorry

end grazing_area_proof_l566_566985


namespace bulb_arrangement_l566_566881

theorem bulb_arrangement (B R W : ℕ) (hB : B = 7) (hR : R = 7) (hW : W = 12) :
  let n_blue_red := B + R,
      slots := B + R + 1 in
  (nat.choose n_blue_red B) * (nat.choose slots W) = 1561560 :=
by
  have h1 : nat.choose 14 7 = 3432, by sorry,
  have h2 : nat.choose 15 12 = 455, by sorry,
  rw [h1, h2],
  norm_num

end bulb_arrangement_l566_566881


namespace bus_travel_time_l566_566564

theorem bus_travel_time (D1 D2: ℝ) (T: ℝ) (h1: D1 + D2 = 250) (h2: D1 >= 0) (h3: D2 >= 0) :
  T = D1 / 40 + D2 / 60 ↔ D1 + D2 = 250 := 
by
  sorry

end bus_travel_time_l566_566564


namespace sum_always_even_l566_566006

def sum_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_always_even :
  ∀ f : ℕ → ℤ, (∀ n, f n = n ∨ f n = -n) →
  (1 ≤ n → n ≤ 1995) →
  (∑ i in Finset.range(1995 + 1), f i) % 2 = 0 := sorry

end sum_always_even_l566_566006


namespace stickers_per_page_l566_566514

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l566_566514


namespace area_of_triangle_l566_566057

noncomputable def hyperbolaAreaProblem : ℝ :=
let F1 := (-2, 0) in
let F2 := (2, 0) in
let hyperbola : {P : ℝ × ℝ // (P.1 * P.1) / 4 - (P.2 * P.2) = 1} :=
  ⟨(P.1, P.2), _⟩ in
let pointP : {P : ℝ × ℝ // (P.1 * P.1) / 4 - (P.2 * P.2) = 1} in
let angleF1PF2 := 90 in
4

theorem area_of_triangle (F1 F2 : ℝ × ℝ) (P : {P : ℝ × ℝ // (P.1 * P.1) / 4 - (P.2 * P.2) = 1}) (angleF1PF2 : ℝ) (h : angleF1PF2 = 90) : hyperbolaAreaProblem = 4 :=
sorry

end area_of_triangle_l566_566057


namespace probability_three_blocks_same_color_l566_566253

theorem probability_three_blocks_same_color :
  let colors := ["red", "blue", "yellow", "white", "green"]
  let boxes := [1, 2, 3, 4, 5, 6]
  let people := ["Ang", "Ben", "Jasmin"]
  let choose_boxes := (choose 6 5)
  (∑ (colour : String) in colors, ∑ in boxes, (1 / 6) ^ 3) = 5 / 216 :=
sorry

end probability_three_blocks_same_color_l566_566253


namespace swimmer_speed_in_still_water_l566_566240

-- Define the various given conditions as constants in Lean
def swimmer_distance : ℝ := 3
def river_current_speed : ℝ := 1.7
def time_taken : ℝ := 2.3076923076923075

-- Define what we need to prove: the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) :
  swimmer_distance = (v - river_current_speed) * time_taken → 
  v = 3 := by
  sorry

end swimmer_speed_in_still_water_l566_566240


namespace determine_sanity_l566_566271

-- Define the Transylvanian's response based on their mental state and type.
inductive MentalState
| Sane
| Insane

-- Definitions to capture the behavior of sane and insane Transylvanians.
def response (state: MentalState): Bool :=
  match state with
  | MentalState.Sane => true  -- "Yes, I am a person."
  | MentalState.Insane => false  -- "No, I am not a person."

-- Given the definitions above, state the main theorem.
theorem determine_sanity (state: MentalState) : 
  (response state = true ↔ state = MentalState.Sane) ∧ 
  (response state = false ↔ state = MentalState.Insane) :=
by
  split
  { intro h
    cases state
    { simp [response] at h
      assumption }
    { simp [response] at h
      contradiction } }
  { intro h
    cases state
    { simp [response] at h
      contradiction }
    { simp [response] at h
      assumption } }

end determine_sanity_l566_566271


namespace average_hamburgers_per_day_l566_566978

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l566_566978


namespace projectile_first_reaches_70_feet_l566_566855

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ, t = 7/4 ∧ 0 < t ∧ ∀ s : ℝ, s < t → -16 * s^2 + 80 * s < 70 :=
by 
  sorry

end projectile_first_reaches_70_feet_l566_566855


namespace volume_of_region_l566_566306

theorem volume_of_region:
  let region := { p : ℝ × ℝ × ℝ | let (x, y, z) := p; 
                                 |x + 2*y + z| + |x + 2*y - z| ≤ 12 ∧ 
                                 x ≥ 0 ∧ y ≥ 0 ∧ z ≥ -2 } 
  in 
  let volume := measure_theory.measure_space.volume.of_set region 
  in 
  volume = 72 := 
sorry

end volume_of_region_l566_566306


namespace distinct_points_count_l566_566623

open Real

theorem distinct_points_count :
  let C1 := {p : ℝ × ℝ | (p.1 + 2 * p.2 - 4) * (3 * p.1 - p.2 + 6) = 0}
  let C2 := {p : ℝ × ℝ | (p.1 - p.2 + 3) * (4 * p.1 + 3 * p.2 - 15) = 0}
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ C1 ∧ (x₁, y₁) ∈ C2 ∧ 
   (x₂, y₂) ∈ C1 ∧ (x₂, y₂) ∈ C2 ∧ (x₁, y₁) ≠ (x₂, y₂) ∧ 
   ∀ (z w : ℝ × ℝ), (z ∈ C1 ∧ z ∈ C2) → (z = (x₁, y₁) ∨ z = (x₂, y₂)))
   ∧ ¬∃ (u v : ℝ × ℝ), (u ∈ C1 ∧ u ∈ C2 ∧ v ∈ C1 ∧ v ∈ C2 ∧ u ≠ v ∧ u ≠ (x₁, y₁) ∧ u ≠ (x₂, y₂) ∧ v ≠ (x₁, y₁) ∧ v ≠ (x₂, y₂))) := sorry

end distinct_points_count_l566_566623


namespace initial_earning_members_l566_566476

theorem initial_earning_members (average_income_before: ℝ) (average_income_after: ℝ) (income_deceased: ℝ) (n: ℝ)
    (H1: average_income_before = 735)
    (H2: average_income_after = 650)
    (H3: income_deceased = 990)
    (H4: n * average_income_before - (n - 1) * average_income_after = income_deceased)
    : n = 4 := 
by 
  rw [H1, H2, H3] at H4
  linarith


end initial_earning_members_l566_566476


namespace find_digits_exists_l566_566759

theorem find_digits_exists :
  ∃ (a b c d e f g h i j k l m n o p q r s t : ℕ),
    (a ≠ 0 ∧ d ≠ 0 ∧ k ≠ 0 ∧ n ≠ 0 ∧ -- numbers cannot start with zero
    a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    e ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    f ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    g ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    h ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    i ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    j ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    k ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    l ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    o ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    p ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    q ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    r ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    s ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    t ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (∀ {x y z w},
      x ∈ {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t} →
      y ∈ {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t} →
      z ∈ {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t} →
      w ∈ {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t} →
      x ≠ y → y ≠ z → z ≠ w → w ≠ x →
      ∀ {x1 x2},
      x1 = x →
      x2 ∈ {1} → true
    ) ∧
    ((a * 10 + b + c) * (d * 10 + e + f) = g * 1000 + h * 100 + i * 10 + j) ∧
    ((k * 10 + l + m) * (n * 10 + o + p) = q * 1000 + r * 100 + s * 10 + t)) :=
by
  sorry

end find_digits_exists_l566_566759


namespace smaller_cuboid_length_l566_566724

theorem smaller_cuboid_length
  (width_sm : ℝ)
  (height_sm : ℝ)
  (length_lg : ℝ)
  (width_lg : ℝ)
  (height_lg : ℝ)
  (num_sm : ℝ)
  (h1 : width_sm = 2)
  (h2 : height_sm = 3)
  (h3 : length_lg = 18)
  (h4 : width_lg = 15)
  (h5 : height_lg = 2)
  (h6 : num_sm = 18) :
  ∃ (length_sm : ℝ), (108 * length_sm = 540) ∧ (length_sm = 5) :=
by
  -- proof logic will be here
  sorry

end smaller_cuboid_length_l566_566724


namespace total_distance_is_correct_l566_566785

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l566_566785


namespace log_25_between_consecutive_integers_l566_566155

noncomputable def log_base_10 (x : ℝ) := real.log x / real.log 10

theorem log_25_between_consecutive_integers :
  ∃ a b : ℕ, a + b = 3 ∧ (a : ℝ) < log_base_10 25 ∧ log_base_10 25 < (b : ℝ) := by
  have h1 : log_base_10 10 = 1 := by sorry
  have h2 : log_base_10 100 = 2 := by sorry
  have h3 : 1 < log_base_10 25 := by sorry
  have h4 : log_base_10 25 < 2 := by sorry
  use (1 : ℕ), (2 : ℕ)
  constructor
  · exact rfl
  constructor
  · exact h3
  · exact h4

end log_25_between_consecutive_integers_l566_566155


namespace speed_in_still_water_l566_566966

-- Given conditions
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 41

-- Question: Prove the speed of the man in still water is 33 kmph.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 33 := 
by 
  sorry

end speed_in_still_water_l566_566966


namespace sum_of_first_four_terms_l566_566503

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

variables (a : ℕ → ℤ)
variable (h1 : a 2 = 1)
variable (h2 : a 3 = 3)

theorem sum_of_first_four_terms :
  let d := a 3 - a 2 in
  let a1 := a 2 - d in
  let a4 := a 3 + d in
  a1 + a 2 + a 3 + a4 = 8 :=
by
  sorry

end sum_of_first_four_terms_l566_566503


namespace complex_fraction_result_l566_566439

variables (z : ℂ) (i : ℂ)

theorem complex_fraction_result : z = 1 - i → (i / conj z) = (1 / 2) + (1 / 2) * i := 
by
  intro h
  subst h
  sorry

end complex_fraction_result_l566_566439


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566315

-- Part 1: Total unique three-digit numbers
theorem unique_three_digit_numbers : 
  (finset.univ.card) * (finset.univ.erase 0).card * (finset.univ.erase 0).erase 9.card = 648 :=
sorry

-- Part 2: Total unique three-digit odd numbers
theorem unique_three_digit_odd_numbers :
  5 * ((finset.univ.erase 5).erase 0).card * ((finset.univ.erase 5).erase 0).erase 9.card = 320 :=
sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566315


namespace bob_grade_is_35_l566_566778

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l566_566778


namespace parabola_standard_eq_l566_566742

theorem parabola_standard_eq (p : ℝ) (x y : ℝ) :
  (∃ x y, 3 * x - 4 * y - 12 = 0) →
  ( (p = 6 ∧ x^2 = -12 * y ∧ y = -3) ∨ (p = 8 ∧ y^2 = 16 * x ∧ x = 4)) :=
sorry

end parabola_standard_eq_l566_566742


namespace condition_for_complex_sum_squares_l566_566909

theorem condition_for_complex_sum_squares {n : ℕ} (h : n ≥ 2) 
  (x : Fin n.succ → ℝ) : 
  (∑ k in Finset.range n.succ, x k ^ 2) ≥ x 0 ^ 2 := 
sorry

end condition_for_complex_sum_squares_l566_566909


namespace inverse_matrix_fc_l566_566614

open Matrix

noncomputable def matrix_inverse_own_pair_count : Prop :=
  let A := !![a, 4; -9, d]
  in ∃ a d : ℝ, A * A = 1 ∧ ∀ (a d : ℝ), A * A = 1 → (a, d) ∈ [{(Real.sqrt 37, -Real.sqrt 37), (-Real.sqrt 37, Real.sqrt 37)}]

theorem inverse_matrix_fc : matrix_inverse_own_pair_count := 
by
  sorry

end inverse_matrix_fc_l566_566614


namespace evaluate_product_l566_566290

theorem evaluate_product :
  (∏ n in Finset.range 98, ((n+1)*(n+3)+(n+1)) / (n+2)^2) = 9800 / 9801 :=
sorry

end evaluate_product_l566_566290


namespace quadratic_inequality_iff_abs_a_le_2_l566_566650

theorem quadratic_inequality_iff_abs_a_le_2 (a : ℝ) :
  (|a| ≤ 2) ↔ (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) :=
sorry

end quadratic_inequality_iff_abs_a_le_2_l566_566650


namespace kids_played_on_tuesday_l566_566786

-- Define the total number of kids Julia played with
def total_kids : ℕ := 18

-- Define the number of kids Julia played with on Monday
def monday_kids : ℕ := 4

-- Define the number of kids Julia played with on Tuesday
def tuesday_kids : ℕ := total_kids - monday_kids

-- The proof goal:
theorem kids_played_on_tuesday : tuesday_kids = 14 :=
by sorry

end kids_played_on_tuesday_l566_566786


namespace problem1_problem2_l566_566668

theorem problem1 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : Real.exp x + Real.exp y > 2 * Real.exp 1 :=
by {
  sorry -- Proof goes here
}

theorem problem2 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : x * Real.log x + y * Real.log y > 0 :=
by {
  sorry -- Proof goes here
}

end problem1_problem2_l566_566668


namespace max_postcards_sent_l566_566165

-- Define the problem conditions
def num_students : ℕ := 30
def mutual_friendship (x y : ℕ) : Prop := x ≠ y → ¬(x = y)
def no_triangle (P : finset (ℕ × ℕ)) : Prop := ∀ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ¬((a, b) ∈ P ∧ (b, c) ∈ P ∧ (a, c) ∈ P)

-- The proof statement that verifies the maximum number of postcards sent
theorem max_postcards_sent : ∃ (P : finset (ℕ × ℕ)), |P| = 450 ∧ no_triangle P :=
sorry

end max_postcards_sent_l566_566165


namespace number_of_correct_statements_l566_566490

theorem number_of_correct_statements :
∀ (S1 S2 S3 S4 : Prop),
(S1 ↔ ∀ (P : Type) [metric_space P] (pt : P) (l : set P), (∃ h : ∃ (p : P), p ∉ l, ∀ (q : P), q ∈ l → dist pt q ≥ dist pt h.some → ∀ (q : P), q ∈ l ∧ q ≠ h.some → dist pt q > dist pt h.some)) ∧
(S2 ↔ ∀ (P : Type) [metric_space P] (l1 l2 : set P), (∃ t : set P, l1 ∩ t ≠ ∅ ∧ l2 ∩ t ≠ ∅ → ∀ (a1 a2 : set P), a1 ∩ t = ∅ → a2 ∩ t = ∅ → (l1 ∩ l2 ≠ ∅))) ∧
(S3 ↔ ∀ (P : Type) [metric_space P] (l1 l2 : set P), ∃ t : set P, ∀ (p1 p2 : P), p1 ∈ l1 → p2 ∈ l2 → angle p1 t = angle t p2 → Euclidean_geometry.parallel l1 l2) ∧
(S4 ↔ ∀ (P : Type) [metric_space P] (l1 l2 : set P), l1 ≠ l2 ∧ ¬(l1 ∩ l2 = ∅) ↔ Euclidean_geometry.parallel l1 l2 ∨ (l1 ∩ l2 ≠ ∅)) →
count (λ S, S = true) [S1, S2, S3, S4] = 3 := by sorry

end number_of_correct_statements_l566_566490


namespace sum_max_min_values_l566_566032

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 32 / x

theorem sum_max_min_values :
  y 1 = 34 ∧ y 2 = 24 ∧ y 4 = 40 → ((y 4 + y 2) = 64) :=
by
  sorry

end sum_max_min_values_l566_566032


namespace surface_area_circle_segment_surface_area_loop_of_curve_l566_566304

-- 1. Surface area of the circle segment
theorem surface_area_circle_segment (x y b R y1 y2 : ℝ) : 
  (x ^ 2 + (y - b) ^ 2 = R ^ 2) → 
  y1 ≤ y → y ≤ y2 → 
  2 * π * R * (y2 - y1) := 
sorry

-- 2. Surface area of the loop of the curve
theorem surface_area_loop_of_curve (x y a : ℝ) : 
  (9 * a * x ^ 2 = y * (3 * a - y) ^ 2) → 
  0 ≤ y → y ≤ 3 * a →
  3 * π * a ^ 2 := 
sorry

end surface_area_circle_segment_surface_area_loop_of_curve_l566_566304


namespace complex_number_in_third_quadrant_l566_566109

def z : ℂ := complex.I * (-2 - complex.I)

theorem complex_number_in_third_quadrant (z : ℂ) (h : z = complex.I * (-2 - complex.I)) : 
  z.re < 0 ∧ z.im < 0 :=
by
  simp [h]
  sorry

end complex_number_in_third_quadrant_l566_566109


namespace important_emails_l566_566455

theorem important_emails (total_emails : ℕ) (spam_fraction : ℚ) (promo_fraction : ℚ) (important_emails : ℕ)
  (h_total : total_emails = 400)
  (h_spam_fraction : spam_fraction = 1/4)
  (h_promo_fraction : promo_fraction = 2/5)
  (h_important : important_emails = 180) :
by
  simp [h_total, h_spam_fraction, h_promo_fraction, h_important]
  sorry

end important_emails_l566_566455


namespace minimum_additional_squares_for_symmetry_l566_566400

def square := prod ℕ ℕ

def is_initially_shaded (p : square) : Prop :=
  p = (2, 1) ∨ p = (3, 4) ∨ p = (4, 3)

def is_additional_shaded (p : square) : Prop :=
  p = (3, 1) ∨ p = (1, 3) ∨ p = (4, 2) ∨ p = (1, 2)

def is_final_shaded (p : square) : Prop :=
  is_initially_shaded p ∨ is_additional_shaded p

def is_vertically_symmetric (s : square → Prop) : Prop :=
  ∀ p, s p → s (p.1, 7 - p.2)

def is_horizontally_symmetric (s : square → Prop) : Prop :=
  ∀ p, s p → s (5 - p.1, p.2)

theorem minimum_additional_squares_for_symmetry :
  (∀ p, is_final_shaded p) →
  is_vertically_symmetric is_final_shaded →
  is_horizontally_symmetric is_final_shaded →
  ∃ p₁ p₂ p₃ p₄,
    is_additional_shaded p₁ ∧
    is_additional_shaded p₂ ∧
    is_additional_shaded p₃ ∧
    is_additional_shaded p₄ ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₄ ∧
    p₄ ≠ p₁ ∧ p₄ ≠ p₂ ∧ p₃ ≠ p₁ :=
sorry

end minimum_additional_squares_for_symmetry_l566_566400


namespace stratified_sampling_correct_l566_566216

-- Definitions for the conditions
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_size : ℕ := 15
def sampling_proportion : ℚ := sample_size / total_employees

-- Statement to prove
theorem stratified_sampling_correct :
  (young_employees * sampling_proportion = 7) ∧
  (middle_aged_employees * sampling_proportion = 5) ∧
  (elderly_employees * sampling_proportion = 3) :=
by
  sorry

end stratified_sampling_correct_l566_566216


namespace latte_cost_l566_566078

theorem latte_cost (L : ℝ) 
  (latte_days : ℝ := 5)
  (iced_coffee_cost : ℝ := 2)
  (iced_coffee_days : ℝ := 3)
  (weeks_in_year : ℝ := 52)
  (spending_reduction : ℝ := 0.25)
  (savings : ℝ := 338) 
  (current_annual_spending : ℝ := 4 * savings)
  (weekly_spending : ℝ := latte_days * L + iced_coffee_days * iced_coffee_cost)
  (annual_spending_eq : weeks_in_year * weekly_spending = current_annual_spending) :
  L = 4 := 
sorry

end latte_cost_l566_566078


namespace distance_of_route_l566_566606

theorem distance_of_route (Vq : ℝ) (Vy : ℝ) (D : ℝ) (h1 : Vy = 1.5 * Vq) (h2 : D = Vq * 2) (h3 : D = Vy * 1.3333333333333333) : D = 1.5 :=
by
  sorry

end distance_of_route_l566_566606


namespace smallest_y_exists_l566_566870

theorem smallest_y_exists (M : ℤ) (y : ℕ) (h : 2520 * y = M ^ 3) : y = 3675 :=
by
  have h_factorization : 2520 = 2^3 * 3^2 * 5 * 7 := sorry
  sorry

end smallest_y_exists_l566_566870


namespace line_perpendicular_to_plane_l566_566732

open Plane Geometry

-- Defining the geometric context and the necessary relationships
variables {α β : Plane} {l : Line}

-- The main statement
theorem line_perpendicular_to_plane
  (h1 : l ⟂ β)
  (h2 : α ∥ β) :
  l ⟂ α :=
sorry

end line_perpendicular_to_plane_l566_566732


namespace parabola_vertex_coordinates_l566_566942

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), y = (x - 2)^2 ∧ (x, y) = (2, 0) :=
sorry

end parabola_vertex_coordinates_l566_566942


namespace ratio_cece_to_abe_l566_566592

def abe_ants : ℕ := 4
def beth_ants : ℕ := 1.5 * abe_ants
def duke_ants : ℕ := 0.5 * abe_ants
def total_ants : ℕ := 20
def cece_ants : ℕ := total_ants - (abe_ants + beth_ants + duke_ants)

theorem ratio_cece_to_abe (abe_ants beth_ants duke_ants cece_ants total_ants : ℕ) :
  abe_ants = 4 → beth_ants = 1.5 * abe_ants → duke_ants = 0.5 * abe_ants →
  total_ants = 20 → cece_ants = total_ants - (abe_ants + beth_ants + duke_ants) →
  (cece_ants : ℚ) / abe_ants = 2 :=
by {
  sorry
}

end ratio_cece_to_abe_l566_566592


namespace range_of_b_l566_566033

-- Definitions of circles O and O_1
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x-4)^2 + y^2 = 4

-- Definition of the line equation for P
def line_P (x y b : ℝ) : Prop := x + sqrt 3 * y = b

-- Problem statement
theorem range_of_b (b : ℝ) :
  (∃ P : ℝ × ℝ, (line_P P.1 P.2 b) ∧ ∃ A, ∃ B,
    tangent_to_point P A circle_O ∧ tangent_to_point P B circle_O1 ∧
    dist P B = 2 * dist P A) ↔ 
  (-20 / 3 < b ∧ b < 4) := sorry

end range_of_b_l566_566033


namespace natural_solutions_count_l566_566380

theorem natural_solutions_count : (finset.card 
  (finset.filter (λ (p : ℕ × ℕ), p.1 < 76 ∧ p.2 < 71 ∧ (71 * p.1 + 76 * p.2 < 76 * 71))
    ((finset.range 76).product (finset.range 71)))) = 2625 :=
  sorry

end natural_solutions_count_l566_566380


namespace simplify_and_evaluate_l566_566836

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate_l566_566836


namespace statement_1_statement_2_statement_3_statement_4_statement_5_statement_6_l566_566283

-- Assumptions
variable {N_A : Nat}

-- Conditions
axiom condition_1 : ∀ (n : Nat), (n = 0.1 * N_A) → false
axiom condition_2 : ∀ (n_H2 : Nat) (n_N2 : Nat), (n_H2 = 1.5 * N_A ∧ n_N2 = 0.5 * N_A) → false
axiom condition_3 : ∃ (x y : Nat), (x + y = 0.1 * N_A)
axiom condition_4 : ∀ (n_SO2 : Nat) (n_O2 : Nat), (n_SO2 = 2 * N_A ∧ n_O2 = 1 * N_A) → false
axiom condition_5 : ∀ (n_NO2 : Nat), (n_NO2 = N_A) → false
axiom condition_6 : ∀ (n_NH3 : Nat), (n_NH3 = N_A) → false

-- Proof statements
theorem statement_1 : ¬ ∃ (n : Nat), n = 0.1 * N_A := by
  intros
  apply condition_1
  assumption

theorem statement_2 : ¬ ∃ (n_H2 n_N2 : Nat), n_H2 = 1.5 * N_A ∧ n_N2 = 0.5 * N_A := by
  intros
  apply condition_2
  tauto

theorem statement_3 : ∃ (x y : Nat), (x + y = 0.1 * N_A) := condition_3

theorem statement_4 : ¬ ∃ (n_SO2 n_O2 : Nat), n_SO2 = 2 * N_A ∧ n_O2 = 1 * N_A := by
  intros
  apply condition_4
  tauto

theorem statement_5 : ¬ ∃ (n_NO2 : Nat), n_NO2 = N_A := by
  intros
  apply condition_5
  assumption

theorem statement_6 : ¬ ∃ (n_NH3 : Nat), n_NH3 = N_A := by
  intros
  apply condition_6
  assumption

end statement_1_statement_2_statement_3_statement_4_statement_5_statement_6_l566_566283


namespace two_mul_seven_pow_n_plus_one_divisible_by_three_l566_566829

-- Definition of natural numbers
variable (n : ℕ)

-- Statement of the problem in Lean
theorem two_mul_seven_pow_n_plus_one_divisible_by_three (n : ℕ) : 3 ∣ (2 * 7^n + 1) := 
sorry

end two_mul_seven_pow_n_plus_one_divisible_by_three_l566_566829


namespace average_age_of_women_l566_566106

theorem average_age_of_women (A : ℝ) (W1 W2 : ℝ)
  (cond1 : 10 * (A + 6) - 10 * A = 60)
  (cond2 : W1 + W2 = 60 + 40) :
  (W1 + W2) / 2 = 50 := 
by
  sorry

end average_age_of_women_l566_566106


namespace problem_solution_l566_566709

def seq (a : ℕ → ℝ) (a1 : a 1 = 0) (rec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : Prop :=
  a 6 = Real.sqrt 3

theorem problem_solution (a : ℕ → ℝ) (h1 : a 1 = 0) (hrec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : 
  seq a h1 hrec :=
by
  sorry

end problem_solution_l566_566709


namespace max_shelves_with_5_same_books_l566_566263

theorem max_shelves_with_5_same_books (k : ℕ) : k ≤ 18 → ∃ books : fin 1300 → fin k, 
  ∀ (rearranged_books : fin 1300 → fin k), 
    ∃ (shelf : fin k), (books '' {n | books n = shelf}).card ≥ 5 ∧ ∀ (n : fin 1300), rearranged_books n = books n → books n = shelf :=
begin
  sorry
end

end max_shelves_with_5_same_books_l566_566263


namespace localization_strategy_pros_cons_l566_566458

theorem localization_strategy_pros_cons
  (dense_placement : Bool)
  (economic_advantages : dense_placement → List String)
  (disadvantages : dense_placement → List String) :
  (dense_placement = true →
   economic_advantages dense_placement =
     ["Creation of Market Monopoly",
      "Reduction in Logistics Costs",
      "Franchise Ownership Structure"] ∧
   disadvantages dense_placement =
     ["Cannibalization Effect",
      "Antitrust Laws"]) :=
by
  intros
  sorry

end localization_strategy_pros_cons_l566_566458


namespace part1_part2_part3_l566_566003

-- Definition of harmonious algebraic expression
def harmonious (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x_max x_min : ℝ, a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧ 
                     (∀ x ∈ set.Icc a b, f x ≤ f x_max) ∧ 
                     (∀ x ∈ set.Icc a b, f x_min ≤ f x)

-- Part 1
theorem part1 : 
  let f := λ x : ℝ, abs (x - 1) in
  ∀ (a b : ℝ), a = -2 ∧ b = 2 → 
               (∀ x ∈ set.Icc a b, f x ≤ 3 ∧ 0 ≤ f x) ∧ 
               ¬ harmonious f a b :=
by
  sorry

-- Part 2 - For each given algebraic expression, determine if they are harmonious
theorem part2 : 
  (let f1 := λ x : ℝ, -x + 1 in
   let f2 := λ x : ℝ, -x^2 + 2 in
   let f3 := λ x : ℝ, x^2 + abs x - 4 in
   harmonious f2 (-2) 2) ∧ 
  ¬ harmonious (λ x : ℝ, -x + 1) (-2) 2 ∧
  ¬ harmonious (λ x : ℝ, x^2 + abs x - 4) (-2) 2 :=
by
  sorry

-- Part 3
theorem part3 :
  let f := λ x : ℝ, (λ a : ℝ, a / (abs x + 1) - 2) in
  ∀ (a b x : ℝ), a = -2 ∧ b = 2 → 
               (∃ (max_a min_a : ℝ), 0 ≤ max_a ∧ max_a ≤ 4 ∧ 0 ≤ min_a ∧ min_a ≤ 4 ∧ 
                harmonious (f max_a) a b ∧ harmonious (f min_a) a b) :=
by
  sorry

end part1_part2_part3_l566_566003


namespace sum_a_b_formula_l566_566358

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the geometric sequence
def b (n : ℕ) : ℕ := 2^(n-1)

-- Define the sum of a_{b_1} + a_{b_2} + ... + a_{b_n}
def sum_a_b (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a (b (i + 1)))

-- State the final theorem
theorem sum_a_b_formula (n : ℕ) : 
  sum_a_b n = 2^(n + 1) - n - 2 :=
by
  sorry

end sum_a_b_formula_l566_566358


namespace solve_for_a_l566_566347

-- Given conditions
def x : ℕ := 2
def y : ℕ := 2
def equation (a : ℚ) : Prop := a * x + y = 5

-- Our goal is to prove that "a = 3/2" given the conditions
theorem solve_for_a : ∃ a : ℚ, equation a ∧ a = 3 / 2 :=
by
  sorry

end solve_for_a_l566_566347


namespace num_squares_in_6x6_grid_l566_566284

-- Define the parameters
def grid_lines : ℕ := 6

-- Function to compute the number of squares of each size in the grid
def num_squares_in_grid (n : ℕ) : ℕ :=
  let max_square_size := n - 1 in
  (List.range (max_square_size)).sum (λ size, (max_square_size - size) ^ 2)

-- The property stating that in a 6x6 grid, there are 55 different squares.
theorem num_squares_in_6x6_grid :
  num_squares_in_grid grid_lines = 55 :=
by
  -- [Skipping the proof steps]
  sorry

end num_squares_in_6x6_grid_l566_566284


namespace dogs_prevent_wolf_escaping_square_l566_566757

variable {v : ℝ} -- Define the maximum speed v of the wolf
variable (dogs_speed : ℝ) -- Define the speed of the dogs

-- Conditions
axiom wolf_speed_defined : v > 0
axiom dogs_faster_than_wolf : dogs_speed = 1.5 * v
axiom dogs_run_boundary : true  -- Placeholder for dogs' restriction to boundary
axiom wolf_can_kill_single_dog : true -- Placeholder for wolf can kill a single dog
axiom two_dogs_can_kill_wolf : true -- Placeholder for two dogs killing the wolf
axiom dogs_initial_position : true -- Placeholder for dogs' initial positions

-- Prove that with these conditions, the dogs can prevent the wolf from escaping the square
theorem dogs_prevent_wolf_escaping_square :
  ∀ (wolf_position : ℝ × ℝ), wolf_position ∈ (set.Icc 0 1 × set.Icc 0 1) →
  dogs_speed > v * sqrt 2 →
  ∃ strategy, (strategy_prevent_escape: true) := -- Placeholder for the actual strategy
sorry

end dogs_prevent_wolf_escaping_square_l566_566757


namespace propositions_correct_l566_566340

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, log (2 : ℝ) x > 0
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- The overall theorem to be proved based on p and q
theorem propositions_correct :
  (p ∧ q) ∧ (p ∧ ¬q → False) ∧ ((¬p ∨ q)) ∧ ((¬p ∨ ¬q) → False) :=
begin
  -- Placeholder for proof, not needed in this statement task
  sorry,
end

end propositions_correct_l566_566340


namespace unique_line_through_point_5_4_with_x_intercept_and_prime_y_intercept_l566_566031

open Nat

/-- 
Given a line passing through the point (5, 4) with the equation (x / a) + (y / b) = 1,
where a is a positive integer and b is a positive prime number, 
prove that there is exactly one such line.
-/
theorem unique_line_through_point_5_4_with_x_intercept_and_prime_y_intercept :
  ∃! (a b : ℕ), a > 0 ∧ Prime b ∧ (5 / a + 4 / b = 1) :=
sorry

end unique_line_through_point_5_4_with_x_intercept_and_prime_y_intercept_l566_566031


namespace parallel_lines_in_space_are_coplanar_l566_566498

theorem parallel_lines_in_space_are_coplanar :
  ∀ (l1 l2 : ℝ → ℝ³), 
  parallel l1 l2 → coplanar l1 l2 ∧ ¬(∃ p, l1 p = l2 p) :=
by
  sorry

end parallel_lines_in_space_are_coplanar_l566_566498


namespace chapters_ratio_l566_566998

theorem chapters_ratio
  (c1 : ℕ) (c2 : ℕ) (total : ℕ) (x : ℕ)
  (h1 : c1 = 20)
  (h2 : c2 = 15)
  (h3 : total = 75)
  (h4 : x = (c1 + 2 * c2) / 2)
  (h5 : c1 + 2 * c2 + x = total) :
  (x : ℚ) / (c1 + 2 * c2 : ℚ) = 1 / 2 :=
by
  sorry

end chapters_ratio_l566_566998


namespace jenny_better_choice_sales_l566_566780

-- Define the conditions
def homesA : Nat := 10
def boxesPerHomeA : Nat := 2
def homesB : Nat := 5
def boxesPerHomeB : Nat := 5
def pricePerBox : Nat := 2

-- Define the total sales calculations for each neighborhood
def salesA := homesA * boxesPerHomeA * pricePerBox
def salesB := homesB * boxesPerHomeB * pricePerBox

-- Prove that Jenny makes more money in the better neighborhood
theorem jenny_better_choice_sales : max salesA salesB = salesB :=
by
  have hA : salesA = homesA * boxesPerHomeA * pricePerBox := rfl
  have hB : salesB = homesB * boxesPerHomeB * pricePerBox := rfl
  have h_salesA : salesA = 10 * 2 * 2 := rfl
  have h_salesB : salesB = 5 * 5 * 2 := rfl
  rw [h_salesA, h_salesB]
  -- 10 * 2 * 2 = 40
  calc max 40 50 = 50 := by
    sorry

end jenny_better_choice_sales_l566_566780


namespace nine_fifths_sum_l566_566102

open Real

theorem nine_fifths_sum (a b: ℝ) (ha: a > 0) (hb: b > 0)
    (h1: a * (sqrt a) + b * (sqrt b) = 183) 
    (h2: a * (sqrt b) + b * (sqrt a) = 182) : 
    9 / 5 * (a + b) = 657 := 
by 
    sorry

end nine_fifths_sum_l566_566102


namespace domain_of_v_l566_566529

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (Real.cos x)

theorem domain_of_v :
  (∀ x : ℝ, (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < x ∧ x < 2 * n * Real.pi + Real.pi / 2) ↔ 
    ∀ x : ℝ, ∀ x_in_domain : ℝ, (0 < Real.cos x ∧ 1 / Real.sqrt (Real.cos x) = x_in_domain)) :=
sorry

end domain_of_v_l566_566529


namespace box_volume_l566_566992

-- Define the dimensions and the cut size
def length := 16
def width := 12
def cut_size (x : ℝ) := x

-- Define the new dimensions after folding
def new_length (x : ℝ) := length - 2 * cut_size x
def new_width (x : ℝ) := width - 2 * cut_size x
def height (x : ℝ) := cut_size x

-- Define the volume of the constructed box
def volume (x : ℝ) := new_length x * new_width x * height x

-- Statement to prove
theorem box_volume (x : ℝ) : volume x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end box_volume_l566_566992


namespace domain_of_f_l566_566479

noncomputable def f (x : ℝ) : ℝ := 3 / real.sqrt (real.log x)

theorem domain_of_f : {x : ℝ | f x = f x} = {x : ℝ | 1 < x} :=
by
  sorry

end domain_of_f_l566_566479


namespace T_perimeter_is_20_l566_566136

-- Define the perimeter of a rectangle given its length and width
def perimeter_rectangle (length width : ℝ) : ℝ :=
  2 * length + 2 * width

-- Given conditions
def rect1_length : ℝ := 1
def rect1_width : ℝ := 4
def rect2_length : ℝ := 2
def rect2_width : ℝ := 5
def overlap_height : ℝ := 1

-- Calculate the perimeter of each rectangle
def perimeter_rect1 : ℝ := perimeter_rectangle rect1_length rect1_width
def perimeter_rect2 : ℝ := perimeter_rectangle rect2_length rect2_width

-- Calculate the overlap adjustment
def overlap_adjustment : ℝ := 2 * overlap_height

-- The total perimeter of the T shape
def perimeter_T : ℝ := perimeter_rect1 + perimeter_rect2 - overlap_adjustment

-- The proof statement that we need to show
theorem T_perimeter_is_20 : perimeter_T = 20 := by
  sorry

end T_perimeter_is_20_l566_566136


namespace correct_statements_l566_566128

def statement_1 : Prop :=
  ∀ (data : List ℝ), sample_variance data ≥ 0

def statement_2 (m n : ℕ) (a b : ℝ) : Prop :=
  ∀ (avg : ℝ), avg = (m * a + n * b) / (m + n) 

def statement_3 : Prop :=
  ∃ (x : ℕ), 503 = 16 * 31 + x ∧ x = 7

def number_of_correct_statements (stm1 stm2 stm3 : Prop) : ℕ :=
  [stm1, stm2, stm3].count (λ s => s = True) 

theorem correct_statements :
  (statement_1 ∧ statement_2 ∧ statement_3) → number_of_correct_statements statement_1 statement_2 statement_3 = 2 :=
by sorry

end correct_statements_l566_566128


namespace rectangle_area_tangent_circle_l566_566954

/-- Given a rectangle ABCD with a circle of radius r that is tangent to sides AB, AD, and CD, and passes through the midpoint of diagonal AC, the area of the rectangle is 8r^2. -/
theorem rectangle_area_tangent_circle (ABCD : rectangle) (r : ℝ) (A B C D M : point) 
  (h_tangent_AB : circle_tangent_to r AB ABCD)
  (h_tangent_AD : circle_tangent_to r AD ABCD)
  (h_tangent_CD : circle_tangent_to r CD ABCD)
  (h_midpoint : midpoint AC M)
  (h_pass_through_midpoint : circle_passes_through r M ABCD) :
  area ABCD = 8 * r^2 := 
sorry

end rectangle_area_tangent_circle_l566_566954


namespace matrix_eq_sum_35_l566_566433

theorem matrix_eq_sum_35 (a b c d : ℤ) (h1 : 2 * a = 14 * a - 15 * b)
  (h2 : 2 * b = 9 * a - 10 * b)
  (h3 : 3 * c = 14 * c - 15 * d)
  (h4 : 3 * d = 9 * c - 10 * d) :
  a + b + c + d = 35 :=
sorry

end matrix_eq_sum_35_l566_566433


namespace smallest_sum_of_nine_consecutive_l566_566916

theorem smallest_sum_of_nine_consecutive :
  ∃ n : ℕ, (∑ i in finset.range 9, (n + i)) % 10^7 = 3040102 ∧ (∑ i in finset.range 9, (n + i)) = 83040102 := 
sorry

end smallest_sum_of_nine_consecutive_l566_566916


namespace bottles_per_person_l566_566882

theorem bottles_per_person
  (boxes : ℕ)
  (bottles_per_box : ℕ)
  (bottles_eaten : ℕ)
  (people : ℕ)
  (total_bottles : ℕ := boxes * bottles_per_box)
  (remaining_bottles : ℕ := total_bottles - bottles_eaten)
  (bottles_per_person : ℕ := remaining_bottles / people) :
  boxes = 7 → bottles_per_box = 9 → bottles_eaten = 7 → people = 8 → bottles_per_person = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end bottles_per_person_l566_566882


namespace problem1_problem2_l566_566941

/-- Proof statement for the first mathematical problem -/
theorem problem1 (x : ℝ) (h : (x - 2) ^ 2 = 9) : x = 5 ∨ x = -1 :=
by {
  -- Proof goes here
  sorry
}

/-- Proof statement for the second mathematical problem -/
theorem problem2 (x : ℝ) (h : 27 * (x + 1) ^ 3 + 8 = 0) : x = -5 / 3 :=
by {
  -- Proof goes here
  sorry
}

end problem1_problem2_l566_566941


namespace largest_multiple_of_11_less_than_minus_150_l566_566531

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l566_566531


namespace snowfall_on_friday_l566_566448

def snowstorm (snow_wednesday snow_thursday total_snow : ℝ) : ℝ :=
  total_snow - (snow_wednesday + snow_thursday)

theorem snowfall_on_friday :
  snowstorm 0.33 0.33 0.89 = 0.23 := 
by
  -- (Conditions)
  -- snow_wednesday = 0.33
  -- snow_thursday = 0.33
  -- total_snow = 0.89
  -- (Conclusion) snowstorm 0.33 0.33 0.89 = 0.23
  sorry

end snowfall_on_friday_l566_566448


namespace solve_inequalities_l566_566840

theorem solve_inequalities :
  {x : ℝ | -3 < x ∧ x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 2} =
  { x : ℝ | (5 / (x + 3) ≥ 1) ∧ (x^2 + x - 2 ≥ 0) } :=
sorry

end solve_inequalities_l566_566840


namespace initial_books_count_l566_566893

theorem initial_books_count (x : ℕ) (h : x + 10 = 48) : x = 38 := 
by
  sorry

end initial_books_count_l566_566893


namespace probability_of_winning_l566_566196

def roll_is_seven (d1 d2 : ℕ) : Prop :=
  d1 + d2 = 7

theorem probability_of_winning (d1 d2 : ℕ) (h : roll_is_seven d1 d2) :
  (1/6 : ℚ) = 1/6 :=
by
  sorry

end probability_of_winning_l566_566196


namespace track_length_is_320_l566_566997

noncomputable def length_of_track (x : ℝ) : Prop :=
  (∃ v_b v_s : ℝ, (v_b > 0 ∧ v_s > 0 ∧ v_b + v_s = x / 2 ∧ -- speeds of Brenda and Sally must sum up to half the track length against each other
                    80 / v_b = (x / 2 - 80) / v_s ∧ -- First meeting condition
                    120 / v_s + 80 / v_b = (x / 2 + 40) / v_s + (x - 80) / v_b -- Second meeting condition
                   )) ∧ x = 320

theorem track_length_is_320 : ∃ x : ℝ, length_of_track x :=
by
  use 320
  unfold length_of_track
  simp
  sorry

end track_length_is_320_l566_566997


namespace point_in_second_quadrant_coordinates_l566_566005

variable (x y : ℝ)
variable (P : ℝ × ℝ)
variable (h1 : P.1 = x)
variable (h2 : P.2 = y)

def isInSecondQuadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def distanceToXAxis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distanceToYAxis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem point_in_second_quadrant_coordinates (h1 : isInSecondQuadrant P)
    (h2 : distanceToXAxis P = 2)
    (h3 : distanceToYAxis P = 1) :
    P = (-1, 2) :=
by 
  sorry

end point_in_second_quadrant_coordinates_l566_566005


namespace industrial_lubricants_percentage_l566_566567

theorem industrial_lubricants_percentage :
  let a := 12   -- percentage for microphotonics
  let b := 24   -- percentage for home electronics
  let c := 15   -- percentage for food additives
  let d := 29   -- percentage for genetically modified microorganisms
  let angle_basic_astrophysics := 43.2 -- degrees for basic astrophysics
  let total_angle := 360              -- total degrees in a circle
  let total_budget := 100             -- total budget in percentage
  let e := (angle_basic_astrophysics / total_angle) * total_budget -- percentage for basic astrophysics
  a + b + c + d + e = 92 → total_budget - (a + b + c + d + e) = 8 :=
by
  intros
  sorry

end industrial_lubricants_percentage_l566_566567


namespace travelers_on_liner_l566_566889

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566889


namespace total_driving_time_l566_566961

theorem total_driving_time
    (TotalCattle : ℕ) (Distance : ℝ) (TruckCapacity : ℕ) (Speed : ℝ)
    (hcattle : TotalCattle = 400)
    (hdistance : Distance = 60)
    (hcapacity : TruckCapacity = 20)
    (hspeed : Speed = 60) :
    let Trips := TotalCattle / TruckCapacity,
        OneWayTime := Distance / Speed,
        RoundTripTime := 2 * OneWayTime,
        TotalTime := Trips * RoundTripTime
    in TotalTime = 40 :=
by
  sorry

end total_driving_time_l566_566961


namespace XY_point_collinear_with_P_l566_566790

noncomputable def triangleABC := {A B C : Point | 
  ∃ B : Point,
  right_angle ∠A B C ∧ -- \(ABC \triangle \rightarrow A\) 
  circumcircle \triangleABC ∧ -- circumcircle containing \triangleABC in the plane 
  (exists D : Point, D = midpoint (shorter_arc A B) (circumcircle \triangleABC)) ∧ -- \(D\) is midpoint 
  (exists P : Point, P ∈ Line AB ∧ CP = CD) ∧ -- \(P\) on line AB, such that \(CP = CD\) 
  distinct X Y ∈ (Points circumcircle \triangleABC)  ∧ --  distinct \(X\) and \(Y\) points on the circumcircle
  (AX = AY = PD) -- \(AX = AY = PD\)
}

theorem XY_point_collinear_with_P {A B C D P X Y : Point} 
  (h₁ : right_angle ∠ A B C)
  (circumcircle : circle \triangleABC)
  (h₂ : D = midpoint (shorter_arc A B) circumcircle)
  (h₃ : P ∈ Line AB ∧ CP = CD)
  (h₄ : distinct X Y ∈ Points circumcircle)
  (h₅ : AX = AY = PD) :
  collinear [X, Y, P] :=
sorry

end XY_point_collinear_with_P_l566_566790


namespace max_shelves_with_5_same_books_l566_566264

theorem max_shelves_with_5_same_books (k : ℕ) : k ≤ 18 → ∃ books : fin 1300 → fin k, 
  ∀ (rearranged_books : fin 1300 → fin k), 
    ∃ (shelf : fin k), (books '' {n | books n = shelf}).card ≥ 5 ∧ ∀ (n : fin 1300), rearranged_books n = books n → books n = shelf :=
begin
  sorry
end

end max_shelves_with_5_same_books_l566_566264


namespace main_proof_l566_566816

def total_models := 25
def red_ext_brown_int := 12
def blue_ext_brown_int := 8
def red_ext_beige_int := 2
def blue_ext_beige_int := 3

def total_brown_int := red_ext_brown_int + blue_ext_brown_int
def total_red_ext := red_ext_brown_int + red_ext_beige_int
def total_red_ext_and_brown_int := red_ext_brown_int

def P_B := total_brown_int / total_models
def P_A := total_red_ext / total_models
def P_AB := total_red_ext_and_brown_int / total_models
def P_B_given_A := P_AB / P_A

theorem main_proof : 
  P_B = 4 / 5 ∧ 
  P_B_given_A = 6 / 7 ∧ 
  P_A * P_B ≠ P_AB ∧ 
  (let distribution_X := 
    ([150, 300, 600], [1/2, 49/150, 13/75]))
    ∧
    (E_X = 150 * 1/2 + 300 * 49/150 + 600 * 13/75 = 277) := 
by sorry

end main_proof_l566_566816


namespace average_age_increase_l566_566162

noncomputable def average_increase (ages : List ℕ) := 
  let total_ages := ages.sum 
  let n := ages.length
  total_ages / n

theorem average_age_increase :
  ∀ (s1 : Finset ℕ) (s2 : Finset ℕ) (t : ℕ),
  s1.card = 9 → s2.card = 1 →
  t ∈ s2 →
  s1.sum (λ x, x) / 9 = 8 →
  28 = t →
  average_increase (s1.val ++ s2.val) - average_increase (s1.val) = 2 := by
  sorry

end average_age_increase_l566_566162


namespace find_u_values_l566_566685

namespace MathProof

variable (u v : ℝ)
variable (h1 : u ≠ 0) (h2 : v ≠ 0)
variable (h3 : u + 1/v = 8) (h4 : v + 1/u = 16/3)

theorem find_u_values : u = 4 + Real.sqrt 232 / 4 ∨ u = 4 - Real.sqrt 232 / 4 :=
by {
  sorry
}

end MathProof

end find_u_values_l566_566685


namespace compute_ratio_PX_PS_l566_566766

open_locale big_operators

structure Point : Type :=
(x : ℝ) (y : ℝ)

def distance (a b : Point) : ℝ :=
real.sqrt ((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2)

structure Triangle :=
(P Q R : Point)

variables (P Q R M N X S : Point)

noncomputable def ratio_PX_PS {Δ : Triangle} (hM : distance Δ.P M = 2) (hMQ : distance M Δ.Q = 6)
  (hN : distance Δ.P N = 3) (hNR : distance N Δ.R = 9) (hA : ∀ (v : ℝ), ∀ (u : ℝ), v / u = 2 / 3) :
  ℝ :=
(floating-point division gives the angle bisector partition) sorry

-- The theorem to prove
theorem compute_ratio_PX_PS (Δ : Triangle) (hM : distance Δ.P M = 2) (hMQ : distance M Δ.Q = 6)
  (hN : distance Δ.P N = 3) (hNR : distance N Δ.R = 9) (hA : ∀ (v : ℝ), ∀ (u : ℝ), v / u = 2 / 3) :
  ratio_PX_PS Δ hM hMQ hN hNR hA = 1 / 4 :=
sorry

end compute_ratio_PX_PS_l566_566766


namespace inequality1_solution_inequality2_solution_l566_566097

open Real

-- First problem: proving the solution set for x + |2x + 3| >= 2
theorem inequality1_solution (x : ℝ) : x + abs (2 * x + 3) >= 2 ↔ (x <= -5 ∨ x >= -1/3) := 
sorry

-- Second problem: proving the solution set for |x - 1| - |x - 5| < 2
theorem inequality2_solution (x : ℝ) : abs (x - 1) - abs (x - 5) < 2 ↔ x < 4 :=
sorry

end inequality1_solution_inequality2_solution_l566_566097


namespace partition_inequality_equality_even_l566_566068

variables {X : Type} {A B : finset (finset X)}

def partitions (A B : finset (finset X)) (X : finset X) :=
  (∀ a ∈ A, ∃ (xa : finset X), xa ⊆ X ∧ a = xa) ∧
  (∀ b ∈ B, ∃ (xb : finset X), xb ⊆ X ∧ b = xb)

def disjoint_unions (A B : finset (finset X)) :=
  ∀ a ∈ A, ∀ b ∈ B, a ∩ b = ∅

def condition (A B : finset (finset X)) (n : ℕ) :=
  ∀ a ∈ A, ∀ b ∈ B, a ∩ b = ∅ → (a ∪ b).card ≥ n

theorem partition_inequality {A B : finset (finset X)} (X : finset X) (n : ℕ) 
  (hpartitions : partitions A B X) (hdisjoint : disjoint_unions A B) (hcondition : condition A B n) :
  X.card ≥ n ^ 2 / 2 :=
sorry

theorem equality_even {A B : finset (finset X)} (X : finset X) (n : ℕ) (hneven : n % 2 = 0) 
  (hpartitions : partitions A B X) (hdisjoint : disjoint_unions A B) (hcondition : condition A B n) :
  X.card = n ^ 2 / 2 :=
sorry

end partition_inequality_equality_even_l566_566068


namespace percentage_deducted_from_list_price_l566_566989

def cost_price : ℝ := 66.5
def marked_price : ℝ := 87.5
def profit_percent : ℝ := 25
def selling_price : ℝ := cost_price + cost_price * (profit_percent / 100)

theorem percentage_deducted_from_list_price (D : ℝ) :
  selling_price = marked_price - (D / 100) * marked_price → D = 5 :=
by
  have selling_price_eq : selling_price = 66.5 + 0.25 * 66.5 := rfl
  rw selling_price_eq
  norm_num -- Calculation shows selling_price = 83.125 directly
  intro h
  have eq : 83.125 = 87.5 - (D / 100) * 87.5 := h
  linarith
  sorry

end percentage_deducted_from_list_price_l566_566989


namespace power_expansion_l566_566268

theorem power_expansion (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := 
by 
  sorry

end power_expansion_l566_566268


namespace tanC_div_tanA_plus_tanC_div_tanB_eq_four_l566_566410

theorem tanC_div_tanA_plus_tanC_div_tanB_eq_four
  (a b c A B C : ℝ)
  (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : a = b * (Real.cos C))
  (h5 : a * a + b * b = 3 * c * c / 2)
  (h6 : Real.tan A = (Real.sin A) / (Real.cos A))
  (h7 : Real.tan B = (Real.sin B) / (Real.cos B))
  (h8 : Real.tan C = (Real.sin C) / (Real.cos C)) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
  sorry

end tanC_div_tanA_plus_tanC_div_tanB_eq_four_l566_566410


namespace find_a_l566_566699

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = 3 * x^(a-2) - 2) (h_cond : f 2 = 4) : a = 3 :=
by
  sorry

end find_a_l566_566699


namespace weed_pulling_ratio_l566_566468

variable (W : ℕ)

theorem weed_pulling_ratio
  (hwednesday : W) 
  (htotal : 25 + W + (1 / 5 : ℚ) * W + ((1 / 5 : ℚ) * W - 10) = 120) :
  (W / 25 : ℚ) = 3 :=
by sorry

end weed_pulling_ratio_l566_566468


namespace fibonacci_geometric_progression_l566_566676

noncomputable def is_geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, b (n + 1) = r * b n

def is_fibonacci_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 2) = b (n + 1) + b n

def golden_ratio := (1 + Real.sqrt 5) / 2

theorem fibonacci_geometric_progression (b : ℕ → ℝ) :
  (is_fibonacci_sequence b ∧
   (b 0 = golden_ratio * b 1 ∨ b 1 = golden_ratio * b 0)) →
  ∃ r : ℝ, is_geometric_sequence b r :=
sorry

end fibonacci_geometric_progression_l566_566676


namespace problem_statements_l566_566597

theorem problem_statements :
  (¬(∀ (A B : ℝ), (A > B → sin A > sin B) ∧ ¬(sin A > sin B) ↔ A > B)) ∧
  (∀ (x : ℝ), x^2 - x + 1 >= 0 ↔ ¬(∃ x : ℝ, x^2 - x + 1 < 0)) ∧
  (∀ {p q : Prop}, (¬(p ∨ q)) ↔ (¬p ∧ ¬q)) ∧
  (∀ (x : ℝ), (x > 2 → x^2 - 3x + 2 > 0) ∧ ¬(x^2 - 3x + 2 > 0 → x > 2)) :=
sorry

end problem_statements_l566_566597


namespace square_area_EFGH_l566_566751

theorem square_area_EFGH (AB BP : ℝ) (h1 : AB = Real.sqrt 72) (h2 : BP = 2) (x : ℝ)
  (h3 : AB + BP = 2 * x + 2) : x^2 = 18 :=
by
  sorry

end square_area_EFGH_l566_566751


namespace asymptotes_hyperbola_l566_566351

theorem asymptotes_hyperbola (a b c : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) 
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1) 
  (hyperbola_eq : ∀ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1) 
  (ecc_product : (Real.sqrt (1 - (b^2)/(a^2))) * (Real.sqrt (1 + (b^2)/(a^2))) = (2 * Real.sqrt 2) / 3) : 
  ∀ x y : ℝ, (x ± (Real.sqrt 3)*y = 0) :=
sorry

end asymptotes_hyperbola_l566_566351


namespace pizza_cost_l566_566831

theorem pizza_cost
  (P T : ℕ)
  (hT : T = 1)
  (h_total : 3 * P + 4 * T + 5 = 39) :
  P = 10 :=
by
  sorry

end pizza_cost_l566_566831


namespace y_coordinate_third_vertex_eq_l566_566252

theorem y_coordinate_third_vertex_eq (A B : ℝ × ℝ) (hA : A = (1, 3)) (hB : B = (7, 3)) : 
  ∃ C : ℝ × ℝ, (equilateral_triangle A B C ∧ C.2 = 3 + 3 * Real.sqrt 3 ∧ in_first_quadrant C) :=
by 
  sorry

-- Additional relevant definitions that may be required
structure Point := (x : ℝ) (y : ℝ)

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def dist (A B : Point) : ℝ :=
  sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def in_first_quadrant (C : Point) : Prop :=
  0 < C.x ∧ 0 < C.y

variable A B : Point

end y_coordinate_third_vertex_eq_l566_566252


namespace find_general_formula_sum_b_n_less_than_two_l566_566678

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def S_n (n : ℕ) : ℚ := (n^2 + n) / 2

noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

theorem find_general_formula (n : ℕ) : b_n n = 2 / (n^2 + n) := by 
  sorry

theorem sum_b_n_less_than_two (n : ℕ) :
  Finset.sum (Finset.range n) (λ k => b_n (k + 1)) < 2 :=
by 
  sorry

end find_general_formula_sum_b_n_less_than_two_l566_566678


namespace positive_difference_coords_P_l566_566521

-- Definitions of the points A, B, C
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (12, 0)

-- Definitions for the intersection points P and Q on the specific lines
def point_P (x : ℝ) : ℝ × ℝ := (x, -5/6 * x + 10)
def point_Q (x : ℝ) : ℝ × ℝ := (x, 0)

-- Function to calculate the area of triangle PQC
def area_PQC (x : ℝ) : ℝ :=
  1/2 * abs (12 - x) * abs (-5/6 * x + 10)

-- The proof statement
theorem positive_difference_coords_P (x : ℝ) (h₁ : area_PQC x = 16) : abs (x - (-5/6 * x + 10)) = 1 :=
by
  sorry

end positive_difference_coords_P_l566_566521


namespace fuel_cost_is_50_cents_l566_566234

-- Define the capacities of the tanks
def small_tank_capacity : ℕ := 60
def large_tank_capacity : ℕ := 60 * 3 / 2 -- 50% larger than small tank

-- Define the number of planes
def number_of_small_planes : ℕ := 2
def number_of_large_planes : ℕ := 2

-- Define the service charge per plane
def service_charge_per_plane : ℕ := 100
def total_service_charge : ℕ :=
  service_charge_per_plane * (number_of_small_planes + number_of_large_planes)

-- Define the total cost to fill all planes
def total_cost : ℕ := 550

-- Define the total fuel capacity
def total_fuel_capacity : ℕ :=
  number_of_small_planes * small_tank_capacity + number_of_large_planes * large_tank_capacity

-- Define the total fuel cost
def total_fuel_cost : ℕ := total_cost - total_service_charge

-- Define the fuel cost per liter
def fuel_cost_per_liter : ℕ :=
  total_fuel_cost / total_fuel_capacity

theorem fuel_cost_is_50_cents :
  fuel_cost_per_liter = 50 / 100 := by
sorry

end fuel_cost_is_50_cents_l566_566234


namespace max_value_sqrt_cubed_max_value_sqrt_cubed_achieved_l566_566427

theorem max_value_sqrt_cubed (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) :
  (∛(a * b * c) + ∛((2 - a) * (2 - b) * (2 - c))) ≤ 2 :=
sorry

theorem max_value_sqrt_cubed_achieved :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
  (∛(a * b * c) + ∛((2 - a) * (2 - b) * (2 - c)) = 2) :=
sorry

end max_value_sqrt_cubed_max_value_sqrt_cubed_achieved_l566_566427


namespace locus_of_centers_is_circle_l566_566644

noncomputable theory

open EuclideanGeometry

-- Define a triangle in Euclidean space
variables {P : Type*} [EuclideanSpace P] {A B C : P}

-- Define the concept of an equilateral triangle circumscribed around another triangle
def is_circumscribed_equilateral_triangle (T : EuclideanTriangle P) (ABC : EuclideanTriangle P) : Prop :=
  ∃ T, is_equilateral T ∧ T.circumscribes ABC

-- Define the locus of points
def locus_of_centers_of_circumscribed_equilateral_triangles (ABC : EuclideanTriangle P) : set P :=
  { O : P | ∃ (T : EuclideanTriangle P), is_circumscribed_equilateral_triangle T ABC ∧ circumcenter T = O }

-- The main theorem
theorem locus_of_centers_is_circle (ABC : EuclideanTriangle P) :
  is_circle (locus_of_centers_of_circumscribed_equilateral_triangles ABC) :=
sorry

end locus_of_centers_is_circle_l566_566644


namespace determine_a_values_l566_566700

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (3 - a) ^ x
  else log a (x - 1) + 3

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

theorem determine_a_values :
  ∀ a : ℝ, is_monotonically_increasing (f a) ↔ a ∈ set.Ico (3 - real.sqrt 3) 2 :=
sorry

end determine_a_values_l566_566700


namespace students_who_did_not_receive_an_A_l566_566408

def total_students : ℕ := 40
def a_in_literature : ℕ := 10
def a_in_science : ℕ := 18
def a_in_both : ℕ := 6

theorem students_who_did_not_receive_an_A :
  total_students - ((a_in_literature + a_in_science) - a_in_both) = 18 :=
by
  sorry

end students_who_did_not_receive_an_A_l566_566408


namespace product_ratio_l566_566278

theorem product_ratio (A B : ℝ) (m n : ℕ) (hA : A = ∏ i in Ico 2 (nat.succ n), (1 - 1 / i ^ 3))
    (hB : B = ∏ i in Ico 1 (nat.succ n), (1 + 1 / (i * (i + 1))))
    (hRatio : A / B = m / n) 
    (hmn : Nat.coprime m n) 
    (hFinal : A / B = 1 / 3) : 100 * m + n = 103 :=
by
  sorry

end product_ratio_l566_566278


namespace set_intersection_l566_566445

noncomputable def A : Set ℝ := { x | x / (x - 1) < 0 }
noncomputable def B : Set ℝ := { x | 0 < x ∧ x < 3 }
noncomputable def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem set_intersection (x : ℝ) : (x ∈ A ∧ x ∈ B) ↔ x ∈ expected_intersection :=
by
  sorry

end set_intersection_l566_566445


namespace mother_daughter_ages_l566_566227

theorem mother_daughter_ages :
  ∃ (x y : ℕ), (y = x + 22) ∧ (2 * x = (x + 22) - x) ∧ (x = 11) ∧ (y = 33) :=
by
  sorry

end mother_daughter_ages_l566_566227


namespace find_a_l566_566007

noncomputable def f (a x : ℝ) := 3*x^3 - 9*x + a
noncomputable def f' (x : ℝ) : ℝ := 9*x^2 - 9

theorem find_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  a = 6 ∨ a = -6 :=
by sorry

end find_a_l566_566007


namespace find_numbers_l566_566557

theorem find_numbers :
  ∃ (nums : Fin 2017 → ℤ),
    (∀ i, nums i = 1 ∨ nums i = -1) ∧
    (∀ s : Finset (Fin 2017), s.card = 7 → (∑ i in s, nums i ^ 2) = 7) ∧
    (∀ s : Finset (Fin 2017), s.card = 11 → 0 < (∑ i in s, nums i)) ∧
    (∑ i, nums i) % 9 = 0 ∧
    (∑ i, if nums i = -1 then 1 else 0) = 5 :=
by
  sorry

end find_numbers_l566_566557


namespace sufficient_but_not_necessary_perpendicular_l566_566202

theorem sufficient_but_not_necessary_perpendicular (a : ℝ) :
  (∃ a' : ℝ, a' = -1 ∧ (a' = -1 → (0 : ℝ) ≠ 3 * a' - 1)) ∨
  (∃ a' : ℝ, a' ≠ -1 ∧ (a' ≠ -1 → (0 : ℝ) ≠ 3 * a' - 1)) →
  (3 * a' - 1) * (a' - 3) = -1 := sorry

end sufficient_but_not_necessary_perpendicular_l566_566202


namespace total_driving_time_l566_566960

theorem total_driving_time
    (TotalCattle : ℕ) (Distance : ℝ) (TruckCapacity : ℕ) (Speed : ℝ)
    (hcattle : TotalCattle = 400)
    (hdistance : Distance = 60)
    (hcapacity : TruckCapacity = 20)
    (hspeed : Speed = 60) :
    let Trips := TotalCattle / TruckCapacity,
        OneWayTime := Distance / Speed,
        RoundTripTime := 2 * OneWayTime,
        TotalTime := Trips * RoundTripTime
    in TotalTime = 40 :=
by
  sorry

end total_driving_time_l566_566960


namespace fixed_point_for_any_a_local_minimum_a_neg1_max_m_increasing_function_l566_566275

section Problem

variables (a : ℝ) 

-- Condition 1: The function f(x)
def f (x : ℝ) : ℝ := x + (a * log x) / x

-- Question 1: f(1) = 1 for any a ∈ ℝ
theorem fixed_point_for_any_a : ∀ a : ℝ, f a 1 = 1 :=
sorry

end Problem

section Problem2

-- Function definition for a = -1
def f_neg1 (x : ℝ) : ℝ := x - (log x) / x

-- Question 2: Local minimum of f at x = 1 for a = -1
theorem local_minimum_a_neg1 : ∀ x : ℝ, ∃ x_min = 1, f_neg1 x_min = 1 :=
sorry

end Problem2

section Problem3

variables (m : ℝ)

-- Function for increasing condition
def h (x : ℝ) (a : ℝ) : ℝ := x^2 - a * log x + a

-- Question 3: Maximum value of m for which f is increasing for any a ∈ (0, m]
theorem max_m_increasing_function : ∃ m, ∀ a ∈ Icc (0:ℝ) m, ∀ x : ℝ, 0 < x → 0 ≤ h x a :=
sorry

end Problem3

end fixed_point_for_any_a_local_minimum_a_neg1_max_m_increasing_function_l566_566275


namespace max_q_value_for_closed_polyline_l566_566892

theorem max_q_value_for_closed_polyline :
  ∃ q : ℝ, 
    (∀ (r : ℝ) (i : ℕ), i ≤ 4 → let r_i := r * q^i in 
    (∃ A_i : ℝ, A_i ∈ sphere (0 : ℝ) r_i)) ∧ 
    q = (Real.sqrt 5 + 1) / 2 :=
sorry

end max_q_value_for_closed_polyline_l566_566892


namespace union_of_M_and_N_l566_566663

-- Define the sets M and N
def M : set ℝ := { y | ∃ x, y = x^2 }
def N : set ℝ := { y | ∃ x, y = 2^x ∧ x < 0 }

-- State the theorem that M ∪ N = [0, 1)
theorem union_of_M_and_N : M ∪ N = { y | 0 ≤ y ∧ y < 1 } := sorry

end union_of_M_and_N_l566_566663


namespace common_number_in_two_sets_l566_566611

theorem common_number_in_two_sets (f : Fin 5 → ℝ) (l : Fin 5 → ℝ) (c : ℝ)
    (h_f_avg : (∑ i, f i) / 5 = 7)
    (h_l_avg : (∑ i, l i) / 5 = 9)
    (h_avg_all : (∑ i, (if i < 5 then f ⟨i, Nat.lt_of_lt_of_le i.2 (Nat.le_succ 4)⟩ else l ⟨i - 5, by linarith[show i - 5 < 5 by linarith]⟩)) / 9 = 25/3) :
    c = 5 := by
  sorry

end common_number_in_two_sets_l566_566611


namespace stratified_sampling_result_l566_566981

-- Define the total number of students in each grade
def students_grade10 : ℕ := 1600
def students_grade11 : ℕ := 1200
def students_grade12 : ℕ := 800

-- Define the condition
def stratified_sampling (x : ℕ) : Prop :=
  (x / (students_grade10 + students_grade11 + students_grade12) = (20 / students_grade12))

-- The main statement to be proven
theorem stratified_sampling_result 
  (students_grade10 : ℕ)
  (students_grade11 : ℕ)
  (students_grade12 : ℕ)
  (sampled_from_grade12 : ℕ)
  (h_sampling : stratified_sampling 90)
  (h_sampled12 : sampled_from_grade12 = 20) :
  (90 - sampled_from_grade12 = 70) :=
  by
    sorry

end stratified_sampling_result_l566_566981


namespace total_distance_is_correct_l566_566784

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end total_distance_is_correct_l566_566784


namespace triangle_area_pqr_l566_566398

noncomputable def ratio (a b : ℝ) := a / b
noncomputable def area (a : ℝ) := a

variables {P Q R S T U : Type}
variable [Triangle P Q R S]
variable [Midpoint S P Q]
variable (h1 : ratio (area S P T) (area T U S) = 2)
variable (h2 : ratio (area P T R) (area S P Q) = 1/3)
variable (h3 : area T U S = 12)

theorem triangle_area_pqr : area (P Q R) = 96 :=
by
  sorry

end triangle_area_pqr_l566_566398


namespace cos_solution_differential_eq_l566_566906

theorem cos_solution_differential_eq (y : ℝ → ℝ) (h : ∀ x, y x = Real.cos x) :
  ∀ x, (deriv^[2] y) x + y x = 0 :=
by
  intro x
  rw [h x]
  have h1 : deriv (Real.cos) x = -Real.sin x := Real.deriv_cos x
  have h2 : deriv ((deriv (Real.cos)) x) = deriv (-Real.sin x) := congr_fun Real.deriv_cos x
  simp [h1, h2, deriv_neg, deriv_sin]
  rw [add_right_neg]
  exact eq.refl 0

end cos_solution_differential_eq_l566_566906


namespace find_other_endpoint_of_diameter_l566_566568

theorem find_other_endpoint_of_diameter 
    (center endpoint : ℝ × ℝ) 
    (h_center : center = (5, -2)) 
    (h_endpoint : endpoint = (2, 3))
    : (center.1 + (center.1 - endpoint.1), center.2 + (center.2 - endpoint.2)) = (8, -7) := 
by
  sorry

end find_other_endpoint_of_diameter_l566_566568


namespace sin_of_arccos_l566_566608

theorem sin_of_arccos :
  (sin (arccos (8 / 17)) = 15 / 17) :=
sorry

end sin_of_arccos_l566_566608


namespace similar_rhombuses_with_60_deg_angles_l566_566247

theorem similar_rhombuses_with_60_deg_angles :
  ∀ (R1 R2 : Type) [rhombus R1] [rhombus R2],
  (∃ (α1 α2 : ℝ), α1 = 60 ∧ α2 = 60) →
  similar R1 R2 :=
by
  sorry

end similar_rhombuses_with_60_deg_angles_l566_566247


namespace faster_pump_rate_ratio_l566_566525

theorem faster_pump_rate_ratio (S F : ℝ) 
  (h1 : S + F = 1/5) 
  (h2 : S = 1/12.5) : F / S = 1.5 :=
by
  sorry

end faster_pump_rate_ratio_l566_566525


namespace arctan_sum_pi_four_l566_566041

theorem arctan_sum_pi_four {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b^2 + c^2 = a^2) :
  arctan (b / (a + c)) + arctan (c / (a + b)) = π / 4 :=
sorry

end arctan_sum_pi_four_l566_566041


namespace dima_is_mistaken_l566_566631

theorem dima_is_mistaken :
  (∃ n : Nat, n > 0 ∧ ∀ n, 3 * n = 4 * n) → False :=
by
  intros h
  obtain ⟨n, hn1, hn2⟩ := h
  have hn := (hn2 n)
  linarith

end dima_is_mistaken_l566_566631


namespace sum_of_digits_of_y_coordinate_of_C_l566_566431

variable (a b c : ℝ)
variable (A B C : ℝ × ℝ)

-- Points A, B, C on the parabola y = x^2
hypothesis (hA : A = (a, a^2))
hypothesis (hB : B = (b, b^2))
hypothesis (hC1 : C.1 = b)
hypothesis (hC2 : C.2 = c)

-- A and B are distinct points and AB parallel to x-axis
hypothesis (hAB_parallel : A.2 = B.2)
hypothesis (h_distinct : a ≠ b)

-- Right triangle ∆ABC with B as the right angle and area 1004
hypothesis (h_right_angle : C = (b, c) ∧ c ≠ b^2)
hypothesis (h_area : 0.5 * |a - b| * |c - b^2| = 1004)

theorem sum_of_digits_of_y_coordinate_of_C : (c ≈ b^2 + 1004 ∨ c ≈ b^2 - 1004) → sum_of_digits c = 6 := 
by
  sorry

end sum_of_digits_of_y_coordinate_of_C_l566_566431


namespace construct_isosceles_triangle_l566_566616

theorem construct_isosceles_triangle (m_a m_b : ℝ) : m_a > (m_b / 2) → ∃ (A B C : Type) (heightA heightB : ℝ), 
heightA = m_a ∧ heightB = m_b ∧ is_isosceles_triangle A B C :=
sorry

end construct_isosceles_triangle_l566_566616


namespace exists_palindrome_l566_566229

-- Define what makes a pair of strings over "a" and "b" good
inductive GoodPair : String → String → Prop
| base : GoodPair "a" "b"
| form1 : ∀ u v, GoodPair u v → GoodPair (u ++ v) v
| form2 : ∀ u v, GoodPair u v → GoodPair u (u ++ v)

-- Define what is a palindrome
def is_palindrome (s : String) : Prop :=
  s = s.reverse

-- Define the main theorem
theorem exists_palindrome (alpha beta : String) (h : GoodPair alpha beta) : 
  ∃ gamma, is_palindrome gamma ∧ alpha ++ beta = "a" ++ gamma ++ "b" := 
sorry

end exists_palindrome_l566_566229


namespace smaller_tablet_diagonal_length_is_5_l566_566869

-- Define the areas of the larger and smaller tablets
def area_large (d : ℝ) := (d / Real.sqrt 2) ^ 2
def area_small (d : ℝ) := (d / Real.sqrt 2) ^ 2

-- Given conditions:
-- 1. The area of the larger tablet is 5.5 square inches greater than that of the smaller tablet
-- 2. The diagonal length of the larger tablet is 6 inches
def condition1 (d_small : ℝ) := area_large 6 = area_small d_small + 5.5

-- Prove that the diagonal length of the smaller tablet is 5 inches
theorem smaller_tablet_diagonal_length_is_5 :
  ∃ (d_small : ℝ), condition1 d_small ∧ d_small = 5 :=
by
  -- proof goes here
  sorry

end smaller_tablet_diagonal_length_is_5_l566_566869


namespace total_price_of_bananas_and_oranges_l566_566115

variable (price_orange price_pear price_banana : ℝ)

axiom total_cost_orange_pear : price_orange + price_pear = 120
axiom cost_pear : price_pear = 90
axiom diff_orange_pear_banana : price_orange - price_pear = price_banana

theorem total_price_of_bananas_and_oranges :
  let num_bananas := 200
  let num_oranges := 2 * num_bananas
  let cost_bananas := num_bananas * price_banana
  let cost_oranges := num_oranges * price_orange
  cost_bananas + cost_oranges = 24000 :=
by
  sorry

end total_price_of_bananas_and_oranges_l566_566115


namespace IMO_39_presel_l566_566937

theorem IMO_39_presel (n : ℕ) (r : Fin n → ℝ) (h : ∀ i, 1 ≤ r i) :
  ∑ i in Finset.univ, 1 / (r i + 1) ≥ n / (Real.geom_mean (Finset.univ.image r) + 1) :=
sorry

end IMO_39_presel_l566_566937


namespace problem_statement_l566_566070

def f (x : ℝ) : ℝ := x^6 + x^2 + 7 * x

theorem problem_statement : f 3 - f (-3) = 42 := by
  sorry

end problem_statement_l566_566070


namespace triangle_shape_area_l566_566499

theorem triangle_shape_area (a b : ℕ) (area_small area_middle area_large : ℕ) :
  a = 2 →
  b = 2 →
  area_small = (1 / 2) * a * b →
  area_middle = 2 * area_small →
  area_large = 2 * area_middle →
  area_small + area_middle + area_large = 14 :=
by
  intros
  sorry

end triangle_shape_area_l566_566499


namespace coefficient_x2_in_nested_polynomial_l566_566695

-- The problem involves finding the coefficient of x^2 in the polynomial
-- that results from the expression ((...((x-2)^2-2)^2-...-2)^2-2)^2 with k nested parentheses.

theorem coefficient_x2_in_nested_polynomial (k : ℕ) :
  let P_k (x : ℝ) := (((x - 2)^2 - 2) ^ 2 - 2) ^ (2 ^ (k - 1))
  ∃ C_k : ℝ, (C_k = 4^(k-1) * (4^k - 1) / 3) ∧ 
  (polynomial.coeff (polynomial.of_function (P_k)) 2) = C_k :=
by
  sorry

end coefficient_x2_in_nested_polynomial_l566_566695


namespace point_outside_circle_l566_566745

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a*x + b*y = 1 ∧ x^2 + y^2 = 1)) : a^2 + b^2 ≥ 1 :=
sorry

end point_outside_circle_l566_566745


namespace total_volume_stacked_dice_l566_566183

def die_volume (width length height : ℕ) : ℕ := 
  width * length * height

def total_dice (horizontal vertical layers : ℕ) : ℕ := 
  horizontal * vertical * layers

theorem total_volume_stacked_dice :
  let width := 1
  let length := 1
  let height := 1
  let horizontal := 7
  let vertical := 5
  let layers := 3
  let single_die_volume := die_volume width length height
  let num_dice := total_dice horizontal vertical layers
  single_die_volume * num_dice = 105 :=
by
  sorry  -- proof to be provided

end total_volume_stacked_dice_l566_566183


namespace average_age_increase_l566_566159

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end average_age_increase_l566_566159


namespace horner_eval_at_2_l566_566905

def poly (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_eval_at_2 : poly 2 = 373 := by
  sorry

end horner_eval_at_2_l566_566905


namespace calculate_pow_zero_l566_566270

theorem calculate_pow_zero: (2023 - Real.pi) ≠ 0 → (2023 - Real.pi)^0 = 1 := by
  -- Proof
  sorry

end calculate_pow_zero_l566_566270


namespace simplified_value_l566_566394

-- Define the operation ∗
def operation (m n p q : ℚ) : ℚ :=
  m * p * (n / q)

-- Prove that the simplified value of 5/4 ∗ 6/2 is 60
theorem simplified_value : operation 5 4 6 2 = 60 :=
by
  sorry

end simplified_value_l566_566394


namespace last_two_nonzero_digits_70_factorial_l566_566863

theorem last_two_nonzero_digits_70_factorial : 
  let N := 70
  (∀ N : ℕ, 0 < N → N % 2 ≠ 0 → N % 5 ≠ 0 → ∃ x : ℕ, x % 100 = N % (N + (N! / (2 ^ 16)))) →
  (N! / 10 ^ 16) % 100 = 68 :=
by
sorry

end last_two_nonzero_digits_70_factorial_l566_566863


namespace fg_eq_gf_condition_l566_566390

theorem fg_eq_gf_condition (m n p q : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := 
sorry

end fg_eq_gf_condition_l566_566390


namespace solution_set_l566_566743

noncomputable def f : ℝ → ℝ := sorry

axiom h0 : ∀ x : ℝ, f'(x) < f(x)
axiom h1 : ∀ x : ℝ, f(x + 1) = f(1 - x)
axiom h2 : f(2) = 1

theorem solution_set (x : ℝ) : f(x) < Real.exp x ↔ 0 < x :=
by
  sorry

end solution_set_l566_566743


namespace uncommon_card_cost_l566_566899

/--
Tom's deck contains 19 rare cards, 11 uncommon cards, and 30 common cards.
Each rare card costs $1.
Each common card costs $0.25.
The total cost of the deck is $32.
Prove that the cost of each uncommon card is $0.50.
-/
theorem uncommon_card_cost (x : ℝ): 
  let rare_count := 19
  let uncommon_count := 11
  let common_count := 30
  let rare_cost := 1
  let common_cost := 0.25
  let total_cost := 32
  (rare_count * rare_cost) + (common_count * common_cost) + (uncommon_count * x) = total_cost 
  → x = 0.5 :=
by
  sorry

end uncommon_card_cost_l566_566899


namespace positive_differences_sum_eq_68896_l566_566058

-- Definition of the set S as given in the problem
def S := {n : ℕ | ∃ k : ℕ, k ≤ 8 ∧ n = 3^k}

-- Definition and the theorem to prove
theorem positive_differences_sum_eq_68896 :
  let differences := {d | ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a < b ∧ d = b - a} in
  let N := ∑ d in differences, d in
  N = 68896 :=
by
  sorry

end positive_differences_sum_eq_68896_l566_566058


namespace sum_of_reciprocals_of_distances_l566_566143

theorem sum_of_reciprocals_of_distances :
  let e := (sqrt 5) / 3 in
  let directrix := (9:ℝ) / (sqrt 5) in
  let points := (fin 24).toList.map (λ i => (3 * cos (i * π / 12), 2 * sin (i * π / 12))) in
  let distances := points.map (λ p => abs (fst p - directrix)) in
  let reciprocals := distances.map (λ d => 1 / d) in
  (reciprocals.sum = 6 * sqrt 5) :=
sorry

end sum_of_reciprocals_of_distances_l566_566143


namespace increasing_function_range_minimum_value_of_function_l566_566364

-- Statement for the first problem
theorem increasing_function_range (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ≥ 2, f(x) = ln x + 2 * a / x) :
  (∀ x ≥ 2, f'(x) ≥ 0) → a ≤ 1 :=
sorry

-- Statement for the second problem
theorem minimum_value_of_function (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ (set.Icc 1 real.exp 1), f(x) = ln x + 2 * a / x) (h_min : f(x) = 3) :
  (∃ x ∈ (set.Icc 1 real.exp 1), f(x) = 3) → a = real.exp 1 :=
sorry

end increasing_function_range_minimum_value_of_function_l566_566364


namespace max_value_of_goods_purchased_l566_566236

def initial_amount : ℕ := 650
def cashback_per_spent : ℕ := 40
def spending_threshold : ℕ := 200

theorem max_value_of_goods_purchased (initial_amount cashback_per_spent spending_threshold : ℕ) 
  (amount := 650)
  : ∃ total_value, total_value = 770 :=
by {
  use 770,
  sorry
}

end max_value_of_goods_purchased_l566_566236


namespace trig_expr_correct_l566_566495

noncomputable def trig_expr : ℝ := Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
                                   Real.cos (160 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem trig_expr_correct : trig_expr = 1 / 2 := 
  sorry

end trig_expr_correct_l566_566495


namespace Dianne_keeps_in_sales_l566_566630

theorem Dianne_keeps_in_sales :
  let total_customers := 1000
  let return_rate := 0.37
  let book_cost := 15
  let customers_returning := total_customers * return_rate
  let customers_keeping := total_customers - customers_returning
  let total_sales := customers_keeping * book_cost in
  total_sales = 9450 :=
by
  let total_customers := 1000
  let return_rate := 0.37
  let book_cost := 15
  let customers_returning := total_customers * return_rate
  let customers_keeping := total_customers - customers_returning
  let total_sales := customers_keeping * book_cost
  show total_sales = 9450
  sorry

end Dianne_keeps_in_sales_l566_566630


namespace union_of_A_B_l566_566076

noncomputable def A : Set ℝ := {x | Real.log x / Real.log 2 < 0}
noncomputable def B : Set ℝ := {m | m^2 - 2 * m < 0}

theorem union_of_A_B :
  A ∪ B = Ioo 0 2 :=
by
  sorry

end union_of_A_B_l566_566076


namespace numbers_satisfy_conditions_l566_566556

theorem numbers_satisfy_conditions :
  ∃ (nums : list ℤ), 
    (nums.length = 2017) ∧
    (∀ (subset : list ℤ), subset.length = 7 → (subset.sum (λ n, n^2)) = 7) ∧
    (∀ (subset : list ℤ), subset.length = 11 → (subset.sum id > 0)) ∧
    (nums.sum id % 9 = 0) ∧
    (nums.count (-1) = 5) ∧
    (nums.count 1 = 2012) :=
by
  sorry

end numbers_satisfy_conditions_l566_566556


namespace sum_sin_alpha_ge_one_l566_566330

open Real

theorem sum_sin_alpha_ge_one {n : ℕ} {α : Fin n → ℝ} 
    (hα : ∀ i, 0 ≤ α i ∧ α i ≤ π)
    (hA : ∃ (M : ℕ), (∑ i in finset.univ, (1 + cos (α i)) = 2 * M + 1)) :
    ∑ i in finset.univ, sin (α i) ≥ 1 :=
by
  sorry

end sum_sin_alpha_ge_one_l566_566330


namespace Sn_bounds_l566_566350

-- Given that {a_n} is an arithmetic sequence with a_1 = 3 and a_4 = 12
def aₙ (n : ℕ) : ℕ := 3 + 3 * (n - 1)

-- The sequence {b_n} satisfies b_1 = 4, b_4 = 20, and {b_n - a_n} is a geometric sequence
def bₙ (n : ℕ) : ℕ := 3 * n + 2^(n-1)

-- The sum of the first n terms of the sequence { 2^(n-1) + 3 / b_n * b_(n+1) } is S_n
def Sₙ (n : ℕ) : ℝ := ∑ i in range n, (2^(i - 1) + 3 : ℝ) / (bₙ i * bₙ (i + 1))

theorem Sn_bounds (n : ℕ) : (1/8 : ℝ) ≤ Sₙ n ∧ Sₙ n < (1/4 : ℝ) :=
sorry

end Sn_bounds_l566_566350


namespace max_partition_l566_566491

def max_N (n : ℕ) : ℕ :=
  if n = 1 then 8 else 5 * n + 2

theorem max_partition (n : ℕ) (h : n ≥ 1) : 
  ∃ N, N = max_N n ∧ 
    ∃ A B : set ℕ, 
      A ∪ B = {i | n ≤ i ∧ i ≤ N} ∧ 
      A ∩ B = ∅ ∧ 
      (∀ x y z ∈ A, x + y ≠ z) ∧ 
      (∀ x y z ∈ B, x + y ≠ z) :=
sorry

end max_partition_l566_566491


namespace max_profit_l566_566222

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 
    -0.5 * x^2 + 3.5 * x - 0.5 
  else if x > 5 then 
    17 - 2.5 * x 
  else 
    0

theorem max_profit :
  ∀ x : ℝ, (annual_profit 3.5 = 5.625) :=
by
  -- Proof omitted
  sorry

end max_profit_l566_566222


namespace total_number_of_balls_l566_566171

def number_of_yellow_balls : Nat := 6
def probability_yellow_ball : Rat := 1 / 9

theorem total_number_of_balls (N : Nat) (h1 : number_of_yellow_balls = 6) (h2 : probability_yellow_ball = 1 / 9) :
    6 / N = 1 / 9 → N = 54 := 
by
  sorry

end total_number_of_balls_l566_566171


namespace smallest_multiple_of_6_and_15_l566_566647

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ b = 30 := 
by 
  use 30 
  sorry

end smallest_multiple_of_6_and_15_l566_566647


namespace divisibility_condition_l566_566559

theorem divisibility_condition
  (a p q : ℕ) (hpq : p ≤ q) (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) :=
by
  sorry

end divisibility_condition_l566_566559


namespace range_g_l566_566282

def g (x : ℝ) := (x + 1) / (x^2 + 1)

theorem range_g : set.range g = {1 / 2} :=
by
  sorry

end range_g_l566_566282


namespace total_driving_time_l566_566959

theorem total_driving_time
    (TotalCattle : ℕ) (Distance : ℝ) (TruckCapacity : ℕ) (Speed : ℝ)
    (hcattle : TotalCattle = 400)
    (hdistance : Distance = 60)
    (hcapacity : TruckCapacity = 20)
    (hspeed : Speed = 60) :
    let Trips := TotalCattle / TruckCapacity,
        OneWayTime := Distance / Speed,
        RoundTripTime := 2 * OneWayTime,
        TotalTime := Trips * RoundTripTime
    in TotalTime = 40 :=
by
  sorry

end total_driving_time_l566_566959


namespace total_female_officers_l566_566820

theorem total_female_officers
  (percent_female_on_duty : ℝ)
  (total_on_duty : ℝ)
  (half_of_total_on_duty : ℝ)
  (num_females_on_duty : ℝ) :
  percent_female_on_duty = 0.10 →
  total_on_duty = 200 →
  half_of_total_on_duty = total_on_duty / 2 →
  num_females_on_duty = half_of_total_on_duty →
  num_females_on_duty = percent_female_on_duty * (1000 : ℝ) :=
by
  intros h1 h2 h3 h4
  sorry

end total_female_officers_l566_566820


namespace common_ratio_value_l566_566865

theorem common_ratio_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (x + y + z = 0 ∨ x + y + z ≠ 0) → ((x + y) / z = -1 ∨ (x + y) / z = 2) :=
by
  sorry

end common_ratio_value_l566_566865


namespace sum_of_rectangle_areas_l566_566343

theorem sum_of_rectangle_areas :
  let widths := 3
  let lengths := [1, 9, 25, 49, 81, 121, 169]
  (∑ len in lengths, widths * len) = 1365 := 
by
  sorry

end sum_of_rectangle_areas_l566_566343


namespace arithmetic_progression_b_value_geometric_progression_b_value_l566_566322

variable (a b c : ℝ)

-- Given conditions
def a_value : ℝ := 5 + 2 * Real.sqrt 6
def c_value : ℝ := 5 - 2 * Real.sqrt 6

-- Prove that if a, b, c form an arithmetic progression, then b = 5
theorem arithmetic_progression_b_value (h₁ : a = a_value) (h₂ : c = c_value) (h₃ : 2 * b = a + c) : b = 5 := 
sorry

-- Prove that if a, b, c form a geometric progression, then b = ±1
theorem geometric_progression_b_value (h₁ : a = a_value) (h₂ : c = c_value) (h₃ : b^2 = a * c) : b = 1 ∨ b = -1 := 
sorry

end arithmetic_progression_b_value_geometric_progression_b_value_l566_566322


namespace div_mul_expression_l566_566527

theorem div_mul_expression (a b c d : ℕ) (h1 : a = 72) (h2 : b = 6) (h3 : c = 3) (h4 : d = 2) :
  (a / (b / c) * d) = 72 := 
by
  rw [h1, h2, h3, h4]
  simp
  sorry

end div_mul_expression_l566_566527


namespace mario_age_is_4_l566_566153

-- Define the conditions
def sum_of_ages (mario maria : ℕ) : Prop := mario + maria = 7
def mario_older_by_one (mario maria : ℕ) : Prop := mario = maria + 1

-- State the theorem to prove Mario's age is 4 given the conditions
theorem mario_age_is_4 (mario maria : ℕ) (h1 : sum_of_ages mario maria) (h2 : mario_older_by_one mario maria) : mario = 4 :=
sorry -- Proof to be completed later

end mario_age_is_4_l566_566153


namespace option_b_is_quadratic_l566_566189

theorem option_b_is_quadratic (a b c : ℝ) (x : ℝ) (hx : 3 * x^2 + 2 * x + 4 = 0) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ (3 * x^2 + 2 * x + 4 = a * x^2 + b * x + c)) :=
by
  use [3, 2, 4]
  split
  · exact three_ne_zero -- Lean knows that 3 ≠ 0
  · exact hx
  done

end option_b_is_quadratic_l566_566189


namespace gwen_money_received_from_dad_l566_566308

variables (D : ℕ)

-- Conditions
def mom_received := 8
def mom_more_than_dad := 3

-- Question and required proof
theorem gwen_money_received_from_dad : 
  (mom_received = D + mom_more_than_dad) -> D = 5 := 
by
  sorry

end gwen_money_received_from_dad_l566_566308


namespace optimal_distance_l566_566907

def transportation_cost_minimizer : ℝ :=
  let A_to_B := 183      -- distance from city A to city B in km
  let A_to_river := 33   -- distance from city A to river in km
  let cost_land := 1     -- cost per km on land (c)
  let cost_river := 0.5  -- cost per km on river (c/2)
  let x := 11 * real.sqrt 3
  let cost_function (x : ℝ) := cost_land * real.sqrt (A_to_river ^ 2 + x ^ 2) + cost_river * (A_to_B - x)
  x

theorem optimal_distance : transportation_cost_minimizer = 11 * real.sqrt 3 :=
by
  sorry

end optimal_distance_l566_566907


namespace find_least_number_l566_566535

theorem find_least_number (x : ℕ) :
  (∀ k, 24 ∣ k + 7 → 32 ∣ k + 7 → 36 ∣ k + 7 → 54 ∣ k + 7 → x = k) → 
  x + 7 = Nat.lcm (Nat.lcm (Nat.lcm 24 32) 36) 54 → x = 857 :=
by
  sorry

end find_least_number_l566_566535


namespace geometric_series_sum_l566_566634

theorem geometric_series_sum : 
  ∑' n : ℕ, (5 / 3) * (-1 / 3) ^ n = (5 / 4) := by
  sorry

end geometric_series_sum_l566_566634


namespace carina_total_coffee_l566_566273

def number_of_ten_ounce_packages : ℕ := 4
def number_of_five_ounce_packages : ℕ := number_of_ten_ounce_packages + 2
def ounces_in_each_ten_ounce_package : ℕ := 10
def ounces_in_each_five_ounce_package : ℕ := 5

def total_coffee_ounces : ℕ := 
  (number_of_ten_ounce_packages * ounces_in_each_ten_ounce_package) +
  (number_of_five_ounce_packages * ounces_in_each_five_ounce_package)

theorem carina_total_coffee : total_coffee_ounces = 70 := by
  -- proof to be provided
  sorry

end carina_total_coffee_l566_566273


namespace number_of_moles_of_electrons_gained_l566_566373

-- Definition of the redox reaction
def redox_reaction : string := "2Cu(IO_3)_2 + 24KI + 12H_2SO_4 = 2CuI + 13I_2 + 12K_2SO_4 + 12H_2O"

-- Definition of the changes in valence states in the reaction
def valence_Cu_initial : ℕ := 2
def valence_Cu_final : ℕ := 1
def valence_I_initial : ℕ := 5
def valence_I_final : ℕ := 0

-- Prove that the number of moles of electrons gained by 1 mole of oxidizing agent is 11 mol
theorem number_of_moles_of_electrons_gained : 
    (valence_Cu_initial - valence_Cu_final) + 2 * (valence_I_initial - valence_I_final) = 11 := by
  sorry

end number_of_moles_of_electrons_gained_l566_566373


namespace unique_prime_digit_l566_566145

def is_prime_digit : ℕ → ℕ → Prop 
| n B := prime (2 * 10^6 + 0 * 10^5 + 2 * 10^4 + 4 * 10^3 + 0 * 10^2 + 5 * 10 + B)

theorem unique_prime_digit : (∀ B, prime (202405B) → B = 1) : sorry 

end unique_prime_digit_l566_566145


namespace smallest_integer_x_l566_566181

theorem smallest_integer_x (x : ℤ) : 
  ( ∀ x : ℤ, ( 2 * (x : ℚ) / 5 + 3 / 4 > 7 / 5 → 2 ≤ x )) :=
by
  intro x
  sorry

end smallest_integer_x_l566_566181


namespace greening_task_equation_l566_566552

variable (x : ℝ)

theorem greening_task_equation (h1 : 600000 = 600 * 1000)
    (h2 : ∀ a b : ℝ, a * 1.25 = b -> b = a * (1 + 25 / 100)) :
  (60 * (1 + 25 / 100)) / x - 60 / x = 30 := by
  sorry

end greening_task_equation_l566_566552


namespace cruise_liner_travelers_l566_566883

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l566_566883


namespace car_speed_proof_l566_566543

-- Define the problem constants and hypothesis
def car_speed_problem : Prop :=
  ∃ v : ℝ,
    (1 / v - 1 / 36) = 20 / 3600 ∧ v = 30

-- State the theorem
theorem car_speed_proof : car_speed_problem :=
begin
  -- Proof goes here
  sorry
end

end car_speed_proof_l566_566543


namespace chord_length_l566_566402

-- Definitions based on conditions
variables {r : ℝ} {O A B C D P : EuclideanGeometry.Point}

-- Problem statement (the theorem to be proven)
theorem chord_length (hO : O = midpoint A B)
  (hdiam : AB = 4 * r)
  (hperp : is_perpendicular (line_through C D) (line_through A B))
  (hmid : is_midpoint P A B)
  (hCD : CP = PD) :
  chord_length CD = 2 * r * √3 :=
  sorry -- Proof goes here

end chord_length_l566_566402


namespace ratio_of_distances_l566_566272

-- Define the speeds and times for Car A and Car B
def speedA : ℝ := 70
def timeA : ℝ := 10
def speedB : ℝ := 35
def timeB : ℝ := 10

-- Define the distances by their respective formulas
def distanceA : ℝ := speedA * timeA
def distanceB : ℝ := speedB * timeB

-- Prove the ratio of distances covered by Car A and Car B is 2:1
theorem ratio_of_distances : distanceA / distanceB = 2 :=
by
  -- Using the defined distances
  let distanceA := distanceA
  let distanceB := distanceB
  -- Simplifying the ratio
  have h : distanceA / distanceB = (70 * 10) / (35 * 10) := sorry
  -- Further simplification
  have h1 : (70 * 10) / (35 * 10) = 2 := sorry
  show distanceA / distanceB = 2 from h1
  sorry

end ratio_of_distances_l566_566272


namespace distinct_five_digit_numbers_with_product_24_l566_566646

theorem distinct_five_digit_numbers_with_product_24 :
  number_of_distinct_positive_integers (digits_range := [1, 9]) (num_digits := 5) (digit_product := 24) = 120 :=
by
  sorry

end distinct_five_digit_numbers_with_product_24_l566_566646


namespace A_complete_job_in_12_hours_l566_566210

theorem A_complete_job_in_12_hours :
  let a : ℝ := 12 in -- time for A to complete the job
  let d : ℝ := 6 in  -- time for D to complete the job
  (1 / a + 1 / d = 1 / 4) → a = 12 :=
by
  intros h
  -- proof manually skipped
  sorry

end A_complete_job_in_12_hours_l566_566210


namespace tangent_line_at_origin_maximum_k_for_inequality_l566_566367

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem tangent_line_at_origin : ∀ x : ℝ, 
  let y := f 0 + (2 * (x - 0))
  in y = 2 * x := 
sorry

theorem maximum_k_for_inequality : ∀ k : ℝ,
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x > k * (x + (x^3 / 3))) ↔ k ≤ 2 :=
sorry

end tangent_line_at_origin_maximum_k_for_inequality_l566_566367


namespace travelers_on_liner_l566_566890

theorem travelers_on_liner (a : ℤ) :
  250 ≤ a ∧ a ≤ 400 ∧ 
  a % 15 = 7 ∧
  a % 25 = 17 →
  a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l566_566890


namespace integral_sin_pi_eq_two_l566_566244

theorem integral_sin_pi_eq_two :
  ∫ (x : ℝ) in 0..π, sin x = 2 :=
by
  sorry

end integral_sin_pi_eq_two_l566_566244


namespace max_shelves_within_rearrangement_l566_566260

theorem max_shelves_within_rearrangement (k : ℕ) :
  (∀ books : finset ℕ, books.card = 1300 →
    (∃ shelf_assignment_before shelf_assignment_after : finset ℕ → ℕ,
      (∀ book, shelf_assignment_before book ≤ k ∧ shelf_assignment_after book ≤ k) ∧
      (∃ shelf, (books.filter (λ book, shelf_assignment_before book = shelf)).card ≥ 5 ∧
                (books.filter (λ book, shelf_assignment_after book = shelf)).card ≥ 5)
    )
  ) → k ≤ 18 := sorry

end max_shelves_within_rearrangement_l566_566260


namespace greatest_possible_value_of_a_l566_566116

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l566_566116


namespace probability_jason_emily_in_the_picture_l566_566419

-- Definitions for the conditions
def completes_lap_in (time: ℕ) (runner: ℕ -> ℕ) (t: ℕ): Prop :=
  ∀ n, runner (n * time) = runner 0

def position (runner: ℕ -> ℕ) (time: ℕ) : ℚ :=
  runner time % 1

def within_camera_range (camera_coverage: ℕ) (runner_pos: ℚ) : Prop :=
  runner_pos ≤ 1 / camera_coverage ∨ runner_pos ≥ 1 - 1 / camera_coverage

-- Definitions for Jason and Emily
def Jason (t: ℕ) : ℚ := (t / 100.0) % 1
def Emily (t: ℕ) : ℚ := (t / 120.0) % 1

-- Theorem stating that the probability both are in the picture in any given minute is 2 / 3.
theorem probability_jason_emily_in_the_picture
  (camera_coverage: ℕ := 2)
  (Jason_loop_time: ℕ := 100)
  (Emily_loop_time: ℕ := 120)
  (minute: ℕ := 60) :
  completes_lap_in Jason_loop_time Jason minute ∧
  completes_lap_in Emily_loop_time Emily minute →
  (∑ t in (range minute), 
    by { have h := within_camera_range camera_coverage (position Jason t) ∧ 
            within_camera_range camera_coverage (position Emily t),
         exact if h then 1 else 0 }) / minute = 2 / 3 :=
sorry

end probability_jason_emily_in_the_picture_l566_566419


namespace percentage_saving_l566_566955

theorem percentage_saving 
  (p_coat p_pants : ℝ)
  (d_coat d_pants : ℝ)
  (h_coat : p_coat = 100)
  (h_pants : p_pants = 50)
  (h_d_coat : d_coat = 0.30)
  (h_d_pants : d_pants = 0.40) :
  (p_coat * d_coat + p_pants * d_pants) / (p_coat + p_pants) = 0.333 :=
by
  sorry

end percentage_saving_l566_566955


namespace ball_hits_ground_l566_566856

theorem ball_hits_ground 
  (y : ℝ → ℝ) 
  (height_eq : ∀ t, y t = -3 * t^2 - 6 * t + 90) :
  ∃ t : ℝ, y t = 0 ∧ t = 5.00 :=
by
  sorry

end ball_hits_ground_l566_566856


namespace combined_population_of_New_England_and_New_York_l566_566811

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end combined_population_of_New_England_and_New_York_l566_566811


namespace students_not_making_cut_l566_566551

theorem students_not_making_cut
  (girls boys called_back : ℕ) 
  (h1 : girls = 39) 
  (h2 : boys = 4) 
  (h3 : called_back = 26) :
  (girls + boys) - called_back = 17 := 
by sorry

end students_not_making_cut_l566_566551


namespace maximum_k_for_books_l566_566259

theorem maximum_k_for_books (n_books : ℕ) (k : ℕ) (n_books = 1300) :
  (∃ k : ℕ, ∀ rearrangement : finset (finset ℕ), 
    (∀ s ∈ rearrangement, s.card ≤ n_books / k) → 
    ∃ shelf : finset ℕ, ∃ t ∈ rearrangement, t.card ≥ 5 ∧ shelf.card = t.card) ↔ k = 18 :=
by sorry

end maximum_k_for_books_l566_566259


namespace find_point_C_l566_566339

variables (x1 y1 x2 y2 λ : ℝ)

theorem find_point_C (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) 
  (hB : B = (x2, y2)) 
  (hABC : (λ * (B.1 - A.1), λ * (B.2 - A.2)) = (\ -A.1, -A.2)) :
  ∃ C : ℝ × ℝ, C = (λ * B.1 - A.1 / (λ - 1), λ * B.2 - A.2 / (λ - 1)) :=
sorry

end find_point_C_l566_566339


namespace quadratic_has_two_distinct_real_solutions_solutions_are_non_positive_integers_and_b_lt_2a_l566_566199

/-- Given the quadratic equation x^2 + (a + b - 1)x + ab - a - b = 0, where a and b are positive
integers and a ≤ b, we need to prove that:
1. The equation has two distinct real solutions.
2. If one of the solutions is an integer, then both solutions are non-positive integers and b < 2a.
-/
theorem quadratic_has_two_distinct_real_solutions (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : a ≤ b) :
  let Δ := (a - b)^2 + 2 * (a + b) + 1 in
  Δ > 0 := 
by
  sorry

theorem solutions_are_non_positive_integers_and_b_lt_2a (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : a ≤ b)
  (h4: ∃ r : ℤ, r * r + r * (a + b - 1) + (ab - a - b) = 0) :
  (∀ r : ℤ, (r * r + r * (a + b - 1) + (ab - a - b) = 0) → ∃ s : ℤ, (s * s + s * (a + b - 1) + (ab - a - b) = 0) ∧ s = -r - (a + b - 1)) ∧ b < 2 * a :=
by
  sorry

end quadratic_has_two_distinct_real_solutions_solutions_are_non_positive_integers_and_b_lt_2a_l566_566199


namespace area_of_triangle_DEF_l566_566589

-- Definitions and premises
variables {D E F : ℝ} -- assuming the angles D, E, and F are real numbers
variable r : ℝ := 2
variable R : ℝ := 9
axiom cos_sum_eq : 2 * real.cos E = real.cos D + real.cos F
axiom sum_of_cosines : real.cos D + real.cos E + real.cos F = 1 + r / R

-- The goal is to prove that the area of the triangle DEF is 54
theorem area_of_triangle_DEF : 
  (let s := 3 * R in r * s = 54) := sorry

end area_of_triangle_DEF_l566_566589


namespace kaleb_boxes_required_l566_566423

/-- Kaleb's Games Packing Problem -/
theorem kaleb_boxes_required (initial_games sold_games box_capacity : ℕ) (h1 : initial_games = 76) (h2 : sold_games = 46) (h3 : box_capacity = 5) :
  ((initial_games - sold_games) / box_capacity) = 6 :=
by
  -- Skipping the proof
  sorry

end kaleb_boxes_required_l566_566423


namespace not_proposition_statement_A_l566_566190

-- Definitions of the statements
def statement_A : Prop := ¬(true ∨ false) -- "It may rain tomorrow" is not purely true or false
def statement_B : Prop := ∀ (α β : Type), vertical_angles α β -> α = β -- "Vertical angles are equal" is a proposition
def statement_C {A : Type} [HasAngles A] (angle_A : A) : Prop := is_acute angle_A -- "∠A is an acute angle" is a proposition
def statement_D : Prop := ¬("China has the largest population in the world" ≡ true ∨ false) = false -- "China has the largest population" is a proposition

-- Theorem to prove that statement A is not a proposition
theorem not_proposition_statement_A : statement_A := sorry

end not_proposition_statement_A_l566_566190


namespace salary_restoration_percentage_l566_566936

theorem salary_restoration_percentage (S : ℝ) (h : S > 0) :
  let reduced_salary := S * 0.85,
      P := (1 / 0.85 - 1) * 100
  in (reduced_salary * (1 + P / 100) = S) :=
by
  intro S h
  let reduced_salary := S * 0.85
  let P := (1 / 0.85 - 1) * 100
  have : reduced_salary * (1 + P / 100) = S := sorry
  exact this

end salary_restoration_percentage_l566_566936


namespace find_number_l566_566296

noncomputable def mean_proportional (x y : ℝ) : ℝ := real.sqrt (x * y)

theorem find_number (m y : ℝ) (h1 : m = 56.5) (h2 : y = 64) :
  let x := (m ^ 2) / y in x = 49.87890625 :=
by
  have : mean_proportional x y = m := by sorry
  have : m = real.sqrt (x * y) := by sorry
  sorry

end find_number_l566_566296


namespace boys_in_choir_l566_566880

theorem boys_in_choir
  (h1 : 20 + 2 * 20 + 16 + b = 88)
  : b = 12 :=
by
  sorry

end boys_in_choir_l566_566880


namespace printer_a_time_l566_566825

theorem printer_a_time :
  ∀ (A B : ℕ), 
  B = A + 4 → 
  A + B = 12 → 
  (480 / A = 120) :=
by 
  intros A B hB hAB
  sorry

end printer_a_time_l566_566825


namespace parallelogram_proof_l566_566059

noncomputable def parallelogram_area (l w : ℝ) : ℝ := l * w

structure Parallelogram :=
  (area : ℝ)
  (P Q R S : ℝ)
  (diagonal : ℝ)

theorem parallelogram_proof :
  ∃ (m n p : ℕ), m + n + p = 211 ∧
  ∀ (W X Y Z : ℝ) (P Q R S : ℝ),
    let area := 24 in
    let PQ := 9 in
    let RS := 10 in
    let d := X - Z in
    let d_sq := d * d in
    d_sq = m + n * real.sqrt p ∧
    n ∈ ℕ ∧
    p ∈ ℕ ∧
    p % 1 ≠ 0 :=
begin
  sorry
end

end parallelogram_proof_l566_566059


namespace Georg_can_identify_counterfeit_coins_l566_566633

-- Define the assumptions and conditions as given in part (a)
def coins : Nat := 100
def min_coins_shown : Nat := 10
def max_coins_shown : Nat := 20
def exaggerated_by (actual counterfeit_count : Nat) (exaggeration : Nat) : Nat := actual + exaggeration

-- State the theorem to be proven
theorem Georg_can_identify_counterfeit_coins 
  (B : finset (fin coins) → Nat)
  (h_exaggerate : ∃ k : ℕ, ∀ s : finset (fin coins), B s = exaggerated_by (s.filter (is_counterfeit_coin)).card k) 
  : ∃ f : fin (coins) → bool, (∀ i : fin coins, if f i then is_counterfeit_coin i else ¬ is_counterfeit_coin i) :=
sorry

end Georg_can_identify_counterfeit_coins_l566_566633


namespace misoo_must_deliver_amount_l566_566080

theorem misoo_must_deliver_amount :
  ∀ (total_milk: ℕ) (difference: ℕ), 
  total_milk = 2100 ∧ difference = 200 → 
  let joohee_milk := (total_milk - difference) / 2 in
  let misoo_milk := joohee_milk + difference in
  misoo_milk = 1150 :=
by
  intros total_milk difference h
  cases h with ht hd
  let joohee_milk := (total_milk - difference) / 2
  let misoo_milk := joohee_milk + difference
  sorry -- No proof provided as instructed.

end misoo_must_deliver_amount_l566_566080


namespace number_of_possible_measures_for_A_l566_566860

theorem number_of_possible_measures_for_A :
  ∃ A B : ℕ, (A > 0) ∧ (B > 0) ∧ (A + B = 90) ∧ (∃ k : ℕ, (k ≥ 1) ∧ (A = k * B)) ∧
  (set_of k, B, number_of_possible_measures := 11) :=
sorry

end number_of_possible_measures_for_A_l566_566860


namespace circle_area_below_line_l566_566912

noncomputable def area_below_line (f : ℝ → ℝ) : ℝ :=
  let radius := 5
  let center := (3, 8)
  let circle_area := real.pi * radius^2
  let half_circle_area := circle_area / 2
  let intersection_distance := center.2 - 7
  let theta := 2 * real.acos (intersection_distance / radius)
  let segment_area := half_circle_area - 
                      (0.5 * radius^2 * real.sin(theta))
  segment_area

-- The problem is to show the segment area is approximated to (25 * real.pi) / 2 - 6.31
theorem circle_area_below_line : 
  area_below_line (λ y, 7) ≈ (25 * real.pi) / 2 - 6.31 :=
sorry

end circle_area_below_line_l566_566912


namespace area_of_quadrilateral_is_correct_l566_566690

noncomputable def areaOfQuadrilateral {A B C B1 B2 C1 C2 : Type*} 
(area_ABC : ℝ)
(is_trisection_AB : trisection_points A B B1 B2)
(is_trisection_AC : trisection_points A C C1 C2)
(line_B1C : line_segment B1 C)
(line_B2C : line_segment B2 C)
(line_BC1 : line_segment B C1)
(line_BC2 : line_segment B C2) :
  ℝ :=
sorry

theorem area_of_quadrilateral_is_correct 
  {A B C B1 B2 C1 C2 : Type*} 
  (h_area_ABC : areaOfTriangle A B C = 1)
  (h_trisection_AB : trisection_points A B B1 B2)
  (h_trisection_AC : trisection_points A C C1 C2)
  (h_line_B1C : line_segment B1 C)
  (h_line_B2C : line_segment B2 C)
  (h_line_BC1 : line_segment B C1)
  (h_line_BC2 : line_segment B C2) :
  areaOfQuadrilateral 1 h_trisection_AB h_trisection_AC h_line_B1C h_line_B2C h_line_BC1 h_line_BC2 = 9 / 70 :=
sorry

end area_of_quadrilateral_is_correct_l566_566690


namespace binomial_theorem_l566_566093

theorem binomial_theorem (x y : ℕ) (n : ℕ) : 
  (x + y) ^ n = ∑ k in finset.range (n + 1), (Nat.choose n k) * x ^ (n - k) * y ^ k := 
sorry

end binomial_theorem_l566_566093


namespace evaluate_expression_l566_566289

theorem evaluate_expression (y : ℝ) : 
  (2 ^ (4 * y - 1)) / (5 ^ (-1) + 3 ^ (-1) + 7 ^ (-1)) = 2 ^ (4 * y - 1) * (105 / 71) := 
by
  sorry

end evaluate_expression_l566_566289


namespace problem_solution_l566_566338

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

def A := point (-2) 4
def B := point 3 (-1)
def C := point (-3) (-4)

def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
    (p1.1 - p2.1, p1.2 - p2.2)

def vector_scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
    (a * v.1, a * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
    (v1.1 + v2.1, v1.2 + v2.2)

def CA := vector_sub A C
def CB := vector_sub B C

def CM := vector_scalar_mul 3 CA
def CN := vector_scalar_mul 2 CB

def M := vector_add CM C
def N := vector_add CN C

def MN := vector_sub N M

def distance (v : ℝ × ℝ) : ℝ :=
    Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem problem_solution :
  M = (0, 20) ∧ N = (9, 2) ∧ MN = (9, -18) ∧ distance MN = 9 * Real.sqrt 5 :=
by
    sorry

end problem_solution_l566_566338


namespace muffins_sugar_l566_566233

theorem muffins_sugar (cups_muffins_ratio : 24 * 3 = 72 * s / 9) : s = 9 := by
  sorry

end muffins_sugar_l566_566233


namespace allocation_schemes_l566_566256

theorem allocation_schemes :
  ∃ schemes : ℕ, schemes = 36 ∧ 
  ∀ (interns : ℕ) (classes : ℕ), 
    interns = 4 → classes = 3 → 
    (∃ (assign : fin interns → fin classes), 
      (∀ c : fin classes, (∃ i : fin interns, assign i = c))) → schemes = 6 * 6 :=
by
  sorry

end allocation_schemes_l566_566256


namespace casper_entry_exit_ways_correct_l566_566565

-- Define the total number of windows
def num_windows : Nat := 8

-- Define the number of ways Casper can enter and exit through different windows
def casper_entry_exit_ways (num_windows : Nat) : Nat :=
  num_windows * (num_windows - 1)

-- Create a theorem to state the problem and its solution
theorem casper_entry_exit_ways_correct : casper_entry_exit_ways num_windows = 56 := by
  sorry

end casper_entry_exit_ways_correct_l566_566565


namespace largest_multiple_of_11_lt_neg150_l566_566532

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l566_566532


namespace supremum_g_div_f_l566_566442

noncomputable def f (S : Finset ℝ) (n : ℕ) : ℝ :=
  (∑ a in S, a)^n

noncomputable def g (S : Finset ℝ) (n : ℕ) : ℝ :=
  ∑ t in S.powersetLen n, ∏ i in t, i

theorem supremum_g_div_f (n : ℕ) (h : 0 < n) :
  ∀ S : Finset ℝ, ∀ (a ∈ S), 0 < a →
  (∀ (t : Finset ℝ), t.card = n → t ⊆ S) →
  Sup { g S n / f S n | S : Finset ℝ } = 1 / n! :=
sorry

end supremum_g_div_f_l566_566442


namespace gen_term_a_correct_sum_b_seq_correct_l566_566360

-- Definitions
def is_arithmetic_seq (a : ℕ → ℤ ) : Prop := 
  ∀ n : ℕ, n > 0 → a n + 2 * a (n + 1) = 6*n + 1 

def gen_term_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2*n - 1 

def b_seq (n : ℕ) (a : ℕ → ℤ) : ℤ := 2 * a n * (cos (n * π / 2))^2 

def sum_b_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T (2*n) = 4*n^2 + 2*n

-- Proof statements
theorem gen_term_a_correct (a : ℕ → ℤ) (h : is_arithmetic_seq a) : gen_term_a a := 
sorry 

theorem sum_b_seq_correct (a : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ) (h1 : is_arithmetic_seq a) 
  (h2 : ∀ n : ℕ, b n = b_seq n a) : sum_b_seq T b := 
sorry

end gen_term_a_correct_sum_b_seq_correct_l566_566360


namespace stripe_area_l566_566221

theorem stripe_area :
  ∀ (diameter height width revolutions : ℝ), 
    diameter = 40 → 
    height = 100 → 
    width = 4 → 
    revolutions = 3 → 
    let circumference := Real.pi * diameter in
    let total_length := circumference * revolutions in
    let area := width * total_length in
    area = 480 * Real.pi :=
by
  intros diameter height width revolutions h1 h2 h3 h4 circumference total_length area
  sorry

end stripe_area_l566_566221


namespace hyperbola_eccentricity_l566_566344

-- Definition of the condition
def is_asymptote (h : ℝ) (λ : ℝ) (a : ℝ) (b : ℝ) : Prop :=
  (b / a) = 2

-- Prove the given statement
theorem hyperbola_eccentricity (λ a b : ℝ) (h_asymp: 2 * a + b = 0) (h : is_asymptote h λ a b) : 
  sqrt (1 + (b / a)^2) = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l566_566344


namespace faces_not_necessarily_congruent_faces_necessarily_regular_l566_566571

-- Definitions from the conditions.
variable (P : Type*) [polyhedron P]
variable (S : sphere)

-- Assume the conditions of the problem
axiom intersects_each_edge_at_two_points (e : edge P) : ∃ p1 p2 : point, p1 ∈ S ∧ p2 ∈ S ∧ divides_into_three_equal_segments e p1 p2

-- Proving the necessity and non-necessity conditions
theorem faces_not_necessarily_congruent (P : Type*) [polyhedron P] (S : sphere) :
  (∀ e : edge P, ∃ p1 p2 : point, p1 ∈ S ∧ p2 ∈ S ∧ divides_into_three_equal_segments e p1 p2) →
  ¬(∀ f1 f2 : face P, f1 ≅ f2) := sorry

theorem faces_necessarily_regular (P : Type*) [polyhedron P] (S : sphere) :
  (∀ e : edge P, ∃ p1 p2 : point, p1 ∈ S ∧ p2 ∈ S ∧ divides_into_three_equal_segments e p1 p2) →
  ∀ f : face P, regular_polygon f := sorry

end faces_not_necessarily_congruent_faces_necessarily_regular_l566_566571


namespace decimal_to_base13_185_l566_566617

theorem decimal_to_base13_185 : 
  ∀ n : ℕ, n = 185 → 
      ∃ a b c : ℕ, a * 13^2 + b * 13 + c = n ∧ 0 ≤ a ∧ a < 13 ∧ 0 ≤ b ∧ b < 13 ∧ 0 ≤ c ∧ c < 13 ∧ (a, b, c) = (1, 1, 3) := 
by
  intros n hn
  use 1, 1, 3
  sorry

end decimal_to_base13_185_l566_566617


namespace value_of_x_plus_4_l566_566736

theorem value_of_x_plus_4 (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 :=
by
  sorry

end value_of_x_plus_4_l566_566736


namespace exactly_one_correct_l566_566404

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end exactly_one_correct_l566_566404


namespace strawberries_harvest_l566_566919

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest_l566_566919


namespace nondegenerate_triangle_integer_area_l566_566793

theorem nondegenerate_triangle_integer_area {p : ℕ} (hp : Nat.Prime p ∧ Odd p) :
  (∀ (a b c : ℕ), a + b + c = 4 * p → a < b + c ∧ b < a + c ∧ c < a + b → 
  (∃ (A : ℕ), A * A = (2 * p) * (2 * p - a) * (2 * p - b) * (2 * p - c) ∧ Nat.isSquare A)) ∧
  (p % 8 = 1 ∨ p % 8 = 3) := by
  sorry

end nondegenerate_triangle_integer_area_l566_566793


namespace domain_of_log_function_l566_566478

theorem domain_of_log_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ∀ x : ℝ, f(x) = log a (2^x - 1) → x ∈ set.Ioi 0 :=
by
  sorry

end domain_of_log_function_l566_566478


namespace find_solution_set_l566_566648

noncomputable def solution_set : Set ℕ := { x | (x^2 - 5 * x + 5 = 1) ∨ 
                                          ((x^2 - 5 * x + 5 = -1) ∧ (even (x^2 - 9 * x + 20))) ∨ 
                                          (x^2 - 9 * x + 20 = 0) }

theorem find_solution_set : solution_set = {1, 2, 3, 4, 5} := 
by 
  sorry

end find_solution_set_l566_566648


namespace problem_c_problem_d_l566_566670

open Complex

noncomputable def z1 : ℂ := 2 / (-1 + I)

theorem problem_c : (z1^4 = -4) := by
  sorry

theorem problem_d : ∀ (z : ℂ), (‖z‖ = ‖z1‖) → (z.re^2 + z.im^2 = 2) := by
  intro z h
  have mag_eq : ‖z‖ = √2 := h
  sorry

end problem_c_problem_d_l566_566670


namespace extreme_value_at_1_l566_566325

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

noncomputable def df_dx (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  let fx := f x a b
  let df := D (fun x => fx) x
  df

theorem extreme_value_at_1 (a b : ℝ) 
  (h1 : df_dx 1 a b = 0) 
  (h2 : f 1 a b = 10) :
  a + b = -7 :=
sorry

end extreme_value_at_1_l566_566325


namespace find_c_find_area_l566_566765

-- Define variables
variables (a b c : ℝ) (cosC : ℝ)

-- Define known conditions
def known_conditions : Prop :=
  (a = 1) ∧ (b = 2) ∧ (cosC = 1 / 4)

-- Define the Law of Cosines application
def law_of_cosines (a b cosC c : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosC

-- Prove that c = 2
theorem find_c (h : known_conditions a b cosC) : c = 2 :=
  by
  sorry

-- Define the Pythagorean identity for sine
def sinC_from_cosC (cosC : ℝ) : ℝ :=
  sqrt (1 - cosC^2)

-- Define the area formula
def area_formula (a b sinC : ℝ) : ℝ :=
  1 / 2 * a * b * sinC

-- Prove that the area is (sqrt 15) / 4
theorem find_area (h : known_conditions a b cosC) : area_formula a b (sinC_from_cosC cosC) = sqrt(15) / 4 :=
  by
  sorry

end find_c_find_area_l566_566765


namespace geometric_sequence_sum_l566_566020

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l566_566020


namespace transformed_function_constants_l566_566706

def f (x : ℝ) : ℝ :=
if -4 ≤ x ∧ x ≤ -1 then -x - 3
else if -1 < x ∧ x < 2 then x^2 - 1
else if 2 ≤ x ∧ x ≤ 4 then 3 - x
else 0

def g (x : ℝ) : ℝ := 2 * f(3 * x) + 5

theorem transformed_function (x : ℝ) :
  g(x) = 2 * f(3 * x) + 5 :=
by
  sorry

theorem constants : ∃ (a b c : ℝ), g(x) = a * f(b * x) + c ∧ a = 2 ∧ b = 3 ∧ c = 5 :=
by
  use 2, 3, 5
  split
  exact transformed_function
  split
  rfl
  split
  rfl
  rfl

end transformed_function_constants_l566_566706


namespace age_problem_l566_566099

-- Definitions from conditions
variables (p q : ℕ) -- ages of p and q as natural numbers
variables (Y : ℕ) -- number of years ago p was half the age of q

-- Main statement
theorem age_problem :
  (p + q = 28) ∧ (p / q = 3 / 4) ∧ (p - Y = (q - Y) / 2) → Y = 8 :=
by
  sorry

end age_problem_l566_566099


namespace sum_of_first_n_terms_geometric_sequence_l566_566357

open_locale big_operators

-- Definition of the geometric sequence with given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, q > 0 ∧ (∀ n, a (n+1) = q * a n)

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := 2 ^ n

-- Sum of the first n terms of a geometric sequence
noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
∑ i in finset.range n, a i

-- Theorem statement
theorem sum_of_first_n_terms_geometric_sequence : 
  (∀ a : ℕ → ℝ, geometric_sequence a ∧ a 1 = 1 ∧ a 2 = 4 → sum_first_n_terms a n = 2^n - 1) :=
begin
  sorry
end

end sum_of_first_n_terms_geometric_sequence_l566_566357


namespace calculation_problem_equation_solving_problem_l566_566553

-- Part 1: Calculation problem
theorem calculation_problem :
  (nat.cbrt (-27) : ℝ) + real.sqrt ((-2)^2) + abs (1 - real.sqrt 2) = real.sqrt 2 - 2 :=
by
  sorry -- Proof is omitted here; verified in solution steps.

-- Part 2: Equation solving problem
theorem equation_solving_problem (x : ℝ) :
  4 * (x + 2)^2 - 16 = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry -- Proof is omitted here; verified in solution steps.

end calculation_problem_equation_solving_problem_l566_566553


namespace meeting_days_for_eu_committee_l566_566844

noncomputable def max_meeting_days (n : ℕ) : ℕ := 2^(n - 1)

theorem meeting_days_for_eu_committee (total_member_states : ℕ) (daily_meetings : ℕ)
  (condition_2 : ∀ (d : ℕ), d < daily_meetings → ∃ (represented_states : Finset ℕ), represented_states.nonempty)
  (condition_3 : ∀ (d1 d2 : ℕ), d1 < daily_meetings → d2 < daily_meetings → d1 ≠ d2 → 
                  ∃ (r1 r2 : Finset ℕ), r1 ≠ r2 ∧ r1 ∩ r2 ≠ ∅)
  (condition_4 : ∀ (n : ℕ), n < daily_meetings →
                  ∀ (k : ℕ), k < n → ∃ (rn rk : Finset ℕ), (rn ∩ rk).nonempty) :
  daily_meetings ≤ max_meeting_days total_member_states :=
sorry

end meeting_days_for_eu_committee_l566_566844


namespace function_values_l566_566311

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * x^2 + c

theorem function_values (a b c : ℝ) : 
  f a b c 1 = 1 ∧ f a b c (-1) = 1 := 
by
  sorry

end function_values_l566_566311


namespace bob_grade_is_35_l566_566775

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l566_566775


namespace probability_same_group_l566_566164

open ProbabilityTheory

theorem probability_same_group (p : ProbMassFunction (Fin 3)) :
  (∀ a b : Fin 3, p a = p b) → 
  P (λ (x : Fin 3 × Fin 3), x.fst = x.snd) = 1 / 3 :=
by
  sorry

end probability_same_group_l566_566164


namespace remainder_of_2857916_div_4_l566_566180

theorem remainder_of_2857916_div_4 :
  (2857916 % 4 = 0) :=
by
  let n := 2857916
  have h : (n % 100) = 16 := rfl
  have last_two_digits_divisible_by_4 : (16 % 4 = 0) := by norm_num
  sorry

end remainder_of_2857916_div_4_l566_566180


namespace minimum_width_for_fence_l566_566469

theorem minimum_width_for_fence (w : ℝ) (h : 0 ≤ 20) : 
  (w * (w + 20) ≥ 150) → w ≥ 10 :=
by
  sorry

end minimum_width_for_fence_l566_566469


namespace combined_population_of_New_England_and_New_York_l566_566810

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end combined_population_of_New_England_and_New_York_l566_566810


namespace f_g_x_eq_l566_566436

noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / 3
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem f_g_x_eq (x : ℝ) : f (g x) = (x^2 + 7*x + 12) / 3 := by
  sorry

end f_g_x_eq_l566_566436


namespace product_of_real_roots_l566_566496

theorem product_of_real_roots (x : ℝ) (h : x ^ real.logb 5 x = 5) 
  : x = 5 ∨ x = 1 / 5 → x = 5 → 1 = 1 :=
by sorry

end product_of_real_roots_l566_566496


namespace largest_multiple_of_11_less_than_minus_150_l566_566530

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end largest_multiple_of_11_less_than_minus_150_l566_566530


namespace priyas_driving_speed_l566_566457

noncomputable def distance (speed time : ℝ) : ℝ := speed * time

noncomputable def speed (distance time : ℝ) : ℝ := distance / time

theorem priyas_driving_speed :
  ∀ (speed_xz : ℝ) (time_xz : ℝ) (time_zy : ℝ) (speed_zy : ℝ),
  speed_xz = 50 →
  time_xz = 5 →
  time_zy = 2.0833333333333335 →
  distance speed_xz time_xz / 2 / time_zy ≈ 60 :=
by
  intros speed_xz time_xz time_zy speed_zy hxz txz tzy
  have dist_xz : ℝ := distance speed_xz time_xz
  have dist_zy : ℝ := dist_xz / 2
  have speed_zy := speed dist_zy time_zy
  sorry

end priyas_driving_speed_l566_566457


namespace compute_a_b_c_d_sum_l566_566747

-- Define the sides of the triangle
def AB : ℝ := 4
def BC : ℝ := 5
def CA : ℝ := 6

-- Define the circular arcs with 60 degree angles outside of the triangle
def arc_p (A B : ℝ) (angle : ℝ) := 
  angle = 60 ∧ A = 4 ∧ B = 5 -- Simplified notation for midpoint check

def arc_q (A C : ℝ) (angle : ℝ) := 
  angle = 60 ∧ A = 4 ∧ C = 6

def arc_r (B C : ℝ) (angle : ℝ) :=
  angle = 60 ∧ B = 5 ∧ C = 6

-- Define midpoints of the arcs
def X := midpoint of arc_p
def Y := midpoint of arc_q
def Z := midpoint of arc_r

-- Define the statement to be proved
theorem compute_a_b_c_d_sum : 
  ∃ a b c d : ℤ, (a > 0 ∧ b > 0 ∧ c ≥ 0 ∧ d > 0 ∧ gcd a c d = 1 ∧ 
  ¬ ∃ p : ℤ, prime p ∧ p^2 ∣ b ∧ sin (∠ X Z Y) = (a * sqrt b + c) / d ∧ (a + b + c + d = 72)) :=
  sorry

end compute_a_b_c_d_sum_l566_566747


namespace number_of_clips_after_k_steps_l566_566526

theorem number_of_clips_after_k_steps (k : ℕ) : 
  ∃ (c : ℕ), c = 2^(k-1) + 1 :=
by sorry

end number_of_clips_after_k_steps_l566_566526


namespace trains_crossing_time_l566_566903

theorem trains_crossing_time
  (length_train1 : ℕ)
  (time_train1 : ℕ)
  (length_train2 : ℕ)
  (time_train2 : ℕ)
  (h1 : length_train1 = 240)
  (h2 : time_train1 = 3)
  (h3 : length_train2 = 300)
  (h4 : time_train2 = 10) :
  (length_train1 + length_train2) / ((length_train1 / time_train1) + (length_train2 / time_train2)) = 4.91 :=
by
  sorry

end trains_crossing_time_l566_566903


namespace bricks_in_chimney_proof_l566_566999

noncomputable def bricks_in_chimney (h : ℕ) : Prop :=
  let brenda_rate := h / 8
  let brandon_rate := h / 12
  let combined_rate_with_decrease := (brenda_rate + brandon_rate) - 12
  (6 * combined_rate_with_decrease = h) 

theorem bricks_in_chimney_proof : ∃ h : ℕ, bricks_in_chimney h ∧ h = 288 :=
sorry

end bricks_in_chimney_proof_l566_566999


namespace ellipse_line_slope_l566_566691

theorem ellipse_line_slope (m n : ℝ) (A B : ℝ × ℝ)
  (hA_ellipse : m * A.fst^2 + n * A.snd^2 = 1)
  (hB_ellipse : m * B.fst^2 + n * B.snd^2 = 1)
  (hA_line : A.snd = 1 - A.fst)
  (hB_line : B.snd = 1 - B.fst)
  (h_slope : 2 * (1 - A.fst + 1 - B.fst) / (A.fst + B.fst) = sqrt 2) :
  m / n = sqrt 2 := sorry

end ellipse_line_slope_l566_566691


namespace angle_bisector_of_triangle_l566_566042

noncomputable def is_angle_bisector (P Q R : Point) (X : Point) : Prop :=
dist_from_side X P Q = dist_from_side X Q R

variables {A B C M : Point}

theorem angle_bisector_of_triangle (
  hA : is_angle_bisector A B C M,
  hB : is_angle_bisector B A C M
  ) : is_angle_bisector C A B M :=
sorry

end angle_bisector_of_triangle_l566_566042


namespace range_of_3a_minus_b_l566_566320

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 4) (h2 : -1 ≤ a - b ∧ a - b ≤ 2) :
  -1 ≤ (3 * a - b) ∧ (3 * a - b) ≤ 8 :=
sorry

end range_of_3a_minus_b_l566_566320


namespace intervals_of_monotonicity_range_of_a_range_of_k_l566_566697

-- Define f(x)
def f (x : ℝ) : ℝ := (1 + 2 * log x) / (x ^ 2)

-- Define the first task
theorem intervals_of_monotonicity :
  (∀ x ∈ Ioo 0 1, deriv f x > 0) ∧ (∀ x ∈ Ioi 1, deriv f x < 0) :=
sorry

-- Define g(x)
def g (a x : ℝ) : ℝ := a * x^2 - 2 * log x - 1

-- Define the second task
theorem range_of_a (a : ℝ) :
  (∃ x1 x2, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0) → (0 < a ∧ a < 1) :=
sorry

-- Define the third task condition
theorem range_of_k (x1 x2 k : ℝ) (hx : 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) :
  (f x1 - f x2) / (log x1 - log x2) ≤ k → k ∈ Ioi (-2 / Real.e) :=
sorry

end intervals_of_monotonicity_range_of_a_range_of_k_l566_566697


namespace age_problem_l566_566224

theorem age_problem (age x : ℕ) (h : age = 64) :
  (1 / 2 : ℝ) * (8 * (age + x) - 8 * (age - 8)) = age → x = 8 :=
by
  sorry

end age_problem_l566_566224


namespace diagonal_prism_length_l566_566583

theorem diagonal_prism_length (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 12) :
  ∃ d : ℝ, d = √433 ∧ ∀ p₁ p₂ p₃ : ℝ, p₁ = a * a ∧ p₂ = b * b ∧ p₃ = c * c → a*a + b*b + c*c = d*d :=
by
  use √433
  sorry

end diagonal_prism_length_l566_566583


namespace average_age_increase_l566_566161

noncomputable def average_increase (ages : List ℕ) := 
  let total_ages := ages.sum 
  let n := ages.length
  total_ages / n

theorem average_age_increase :
  ∀ (s1 : Finset ℕ) (s2 : Finset ℕ) (t : ℕ),
  s1.card = 9 → s2.card = 1 →
  t ∈ s2 →
  s1.sum (λ x, x) / 9 = 8 →
  28 = t →
  average_increase (s1.val ++ s2.val) - average_increase (s1.val) = 2 := by
  sorry

end average_age_increase_l566_566161


namespace problem1_values_of_ab_problem2_value_of_tan_alpha_l566_566871

theorem problem1_values_of_ab
  (x : ℝ) (a b : ℝ) (hx : (x^2 + a * x - 2) < 0 → (-1, b))
  (ha : a = -1) (hb : b = 2) :
  a = -1 ∧ b = 2 :=
by
  sorry

theorem problem2_value_of_tan_alpha
  (a b : ℝ) (z1 z2 : ℂ) (α : ℝ)
  (hz1 : z1 = a + b * complex.I)
  (hz2 : z2 = complex.cos α + complex.sin α * complex.I)
  (hz1z2_img : (z1 * z2).re = 0 ∧ (z1 * z2).im ≠ 0)
  (ha : a = -1) (hb : b = 2) :
  real.tan α = -1/2 :=
by
  sorry

end problem1_values_of_ab_problem2_value_of_tan_alpha_l566_566871


namespace imaginary_part_of_fraction_l566_566356

theorem imaginary_part_of_fraction (i : ℂ) (hi : i * i = -1) : (1 + i) / (1 - i) = 1 :=
by
  -- Skipping the proof
  sorry

end imaginary_part_of_fraction_l566_566356


namespace important_emails_count_l566_566454

def total_emails : ℕ := 400
def spam_fraction : ℚ := 1 / 4
def promotional_fraction : ℚ := 2 / 5

theorem important_emails_count :
  let spam_emails := spam_fraction * total_emails
      non_spam_emails := total_emails - (spam_emails:ℕ)
      promotional_emails := promotional_fraction * non_spam_emails
  in important_emails = 180 :=
by
  sorry

end important_emails_count_l566_566454


namespace shortTreesPlanted_l566_566169

-- Definitions based on conditions
def currentShortTrees : ℕ := 31
def tallTrees : ℕ := 32
def futureShortTrees : ℕ := 95

-- The proposition to be proved
theorem shortTreesPlanted :
  futureShortTrees - currentShortTrees = 64 :=
by
  sorry

end shortTreesPlanted_l566_566169


namespace find_smallest_number_l566_566815

variable (x : ℕ)

def second_number := 2 * x
def third_number := 4 * second_number x
def average := (x + second_number x + third_number x) / 3

theorem find_smallest_number (h : average x = 165) : x = 45 := by
  sorry

end find_smallest_number_l566_566815


namespace proof_expr_l566_566666

theorem proof_expr (a b c : ℤ) (h1 : a - b = 3) (h2 : b - c = 2) : (a - c)^2 + 3 * a + 1 - 3 * c = 41 := by {
  sorry
}

end proof_expr_l566_566666


namespace balance_scale_comparison_l566_566177

theorem balance_scale_comparison :
  (4 / 3) * Real.pi * (8 : ℝ)^3 > (4 / 3) * Real.pi * (3 : ℝ)^3 + (4 / 3) * Real.pi * (5 : ℝ)^3 :=
by
  sorry

end balance_scale_comparison_l566_566177


namespace sum_c_2017_l566_566040

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := a n + b n + real.sqrt (a n ^ 2 + b n ^ 2)

noncomputable def b : ℕ → ℝ
| 0     := 1
| (n+1) := a n + b n - real.sqrt (a n ^ 2 + b n ^ 2)

noncomputable def c (n : ℕ) : ℝ :=
1 / a n + 1 / b n

theorem sum_c_2017 :
  (∑ n in finset.range 2017, c (n + 1)) = 4034 :=
sorry

end sum_c_2017_l566_566040


namespace cubes_with_even_blue_faces_l566_566331

theorem cubes_with_even_blue_faces :
  let total_cubes := 6 * 3 * 2
  let corner_cubes := 8
  let edge_cubes := 4 * 4
  let face_cubes := 2 * (6 - 2)
  let internal_cubes := (6 - 2) * (3 - 2) * (2 - 2)
  total_cubes = 36 ∧
  corner_cubes = 8 ∧
  edge_cubes = 16 ∧
  face_cubes = 8 ∧
  internal_cubes = 4 →
  edge_cubes + internal_cubes = 20 :=
by
  intros total_cubes corner_cubes edge_cubes face_cubes internal_cubes
  exact ⟨rfl, rfl, rfl, rfl, rfl, by simp [edge_cubes, internal_cubes]; norm_num⟩

end cubes_with_even_blue_faces_l566_566331


namespace annie_age_when_anna_three_times_current_age_l566_566254

theorem annie_age_when_anna_three_times_current_age
  (anna_age : ℕ) (annie_age : ℕ)
  (h1 : anna_age = 13)
  (h2 : annie_age = 3 * anna_age) :
  annie_age + 2 * anna_age = 65 :=
by
  sorry

end annie_age_when_anna_three_times_current_age_l566_566254


namespace range_of_m_l566_566008

theorem range_of_m {x m : ℝ} 
  (h : ∀ x, x ≤ 1 → (1 / 2) ^ x + (1 / 3) ^ x - m ≥ 0) : 
  ∃ m, m ≤ 5 / 6 :=
begin
  sorry
end

end range_of_m_l566_566008


namespace geometric_then_sum_geometric_l566_566683

variable {a b c d : ℝ}

def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def forms_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

theorem geometric_then_sum_geometric (h : geometric_sequence a b c d) :
  forms_geometric_sequence (a + b) (b + c) (c + d) :=
sorry

end geometric_then_sum_geometric_l566_566683


namespace y_intercepts_count_l566_566720

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l566_566720


namespace purely_imaginary_complex_l566_566740

theorem purely_imaginary_complex (a : ℝ) 
  (h₁ : a^2 + 2 * a - 3 = 0)
  (h₂ : a + 3 ≠ 0) : a = 1 := by
  sorry

end purely_imaginary_complex_l566_566740


namespace ratio_surface_areas_cube_tetrahedron_l566_566217

-- Definitions to set up the problem
def cube_side_length : ℝ := 2
def tetrahedron_vertices : list (ℝ × ℝ × ℝ) := [(0,0,0), (2,2,0), (2,0,2), (0,2,2)]

-- Function to calculate the distance between two 3D points
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Function to calculate the surface area of a cube given its side length
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- Function to calculate the surface area of a regular tetrahedron given its side length
def surface_area_tetrahedron (a : ℝ) : ℝ := real.sqrt 3 * a^2

-- Statement of the problem
theorem ratio_surface_areas_cube_tetrahedron :
  let s := cube_side_length
  let t := distance (tetrahedron_vertices.head) (tetrahedron_vertices.tail.head)
  surface_area_cube s / surface_area_tetrahedron t = real.sqrt 3 := by
  sorry

end ratio_surface_areas_cube_tetrahedron_l566_566217


namespace integral_x_plus_one_over_x_l566_566635

theorem integral_x_plus_one_over_x :
  ∫ x in 1..2, (x + 1/x) = (3/2 + Real.log 2) :=
by
  sorry

end integral_x_plus_one_over_x_l566_566635


namespace compare_gardens_l566_566595

def area (length width : ℝ) : ℝ := length * width

def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem compare_gardens :
  let area_Alice := area 30 50 in
  let area_Bob := area 35 45 in
  let perimeter_Alice := perimeter 30 50 in
  let perimeter_Bob := perimeter 35 45 in
  area_Bob - area_Alice = 75 ∧ perimeter_Alice = perimeter_Bob :=
by
  sorry

end compare_gardens_l566_566595


namespace problem_theorem_l566_566539

theorem problem_theorem (x y z : ℤ) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1)
  (h3 : 3 * x = 5 * z + 2) : 
  11 * y - x + 7 * z = 219 := 
by
  sorry

end problem_theorem_l566_566539


namespace math_proof_l566_566309

noncomputable def proof_problem (f : ℝ → ℝ) : Prop :=
  differentiable ℝ f ∧
  deriv f 1 = 0 ∧
  ∀ x : ℝ, (x - 1) * deriv f x > 0 →
  f(0) + f(2) > 2 * f(1)

  theorem math_proof : ∀ (f : ℝ → ℝ), proof_problem f := sorry

end math_proof_l566_566309


namespace maximum_k_for_books_l566_566258

theorem maximum_k_for_books (n_books : ℕ) (k : ℕ) (n_books = 1300) :
  (∃ k : ℕ, ∀ rearrangement : finset (finset ℕ), 
    (∀ s ∈ rearrangement, s.card ≤ n_books / k) → 
    ∃ shelf : finset ℕ, ∃ t ∈ rearrangement, t.card ≥ 5 ∧ shelf.card = t.card) ↔ k = 18 :=
by sorry

end maximum_k_for_books_l566_566258


namespace each_friend_pays_amount_l566_566205

def total_bill : ℝ := 100
def num_friends : ℝ := 5
def coupon_discount : ℝ := 6 / 100

def discounted_bill := total_bill * (1 - coupon_discount)
def amount_per_friend := discounted_bill / num_friends

theorem each_friend_pays_amount : amount_per_friend = 18.80 := by
  sorry

end each_friend_pays_amount_l566_566205


namespace unique_positive_integer_with_digits_2_5_divisible_by_2_pow_2005_l566_566092

def is_composed_of_2s_and_5s (x : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (digits 10 x) → d = 2 ∨ d = 5

def has_exactly_2005_digits (x : ℕ) : Prop :=
  (digits 10 x).length = 2005

theorem unique_positive_integer_with_digits_2_5_divisible_by_2_pow_2005 :
  ∃! (x : ℕ), is_composed_of_2s_and_5s x ∧ has_exactly_2005_digits x ∧ x % 2^2005 = 0 :=
sorry

end unique_positive_integer_with_digits_2_5_divisible_by_2_pow_2005_l566_566092


namespace roger_reading_weeks_l566_566464

theorem roger_reading_weeks (total_books books_per_week : ℕ) (h_total : total_books = 30) (h_weekly : books_per_week = 6) : total_books / books_per_week = 5 := by
  rw [h_total, h_weekly]
  norm_num

end roger_reading_weeks_l566_566464


namespace students_playing_both_l566_566749

theorem students_playing_both (T F L N B : ℕ)
  (hT : T = 39)
  (hF : F = 26)
  (hL : L = 20)
  (hN : N = 10)
  (hTotal : (F + L - B) + N = T) :
  B = 17 :=
by
  sorry

end students_playing_both_l566_566749


namespace circle_properties_intercept_length_l566_566679

theorem circle_properties (a r : ℝ) (h1 : a^2 + 16 = r^2) (h2 : (6 - a)^2 + 16 = r^2) (h3 : r > 0) :
  a = 3 ∧ r = 5 :=
by
  sorry

theorem intercept_length (m : ℝ) (h : |24 + m| / 5 = 3) :
  m = -4 ∨ m = -44 :=
by
  sorry

end circle_properties_intercept_length_l566_566679


namespace breadth_of_rectangular_plot_l566_566547

theorem breadth_of_rectangular_plot (b l A : ℕ) (h1 : A = 20 * b) (h2 : l = b + 10) 
    (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l566_566547


namespace evaluate_sum_l566_566345

variable {a b c : ℝ}

theorem evaluate_sum 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by
  sorry

end evaluate_sum_l566_566345


namespace part1_part2_l566_566705

noncomputable def f (a x : ℝ) := x + a^2 / x
def g (x : ℝ) := x + Real.log x
noncomputable def h (a x : ℝ) := f a x + g x

theorem part1 (a : ℝ) (h_extremum : ∃ x, x = 1 ∧ deriv (h a) x = 0) :
  a = Real.sqrt 3 :=
sorry

theorem part2 (a : ℝ) (h_condition : ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) :
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end part1_part2_l566_566705


namespace greatest_possible_value_of_a_l566_566123

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end greatest_possible_value_of_a_l566_566123


namespace average_bc_l566_566107

variables (A B C : ℝ)

-- Conditions
def average_abc := (A + B + C) / 3 = 45
def average_ab := (A + B) / 2 = 40
def weight_b := B = 31

-- Proof statement
theorem average_bc (A B C : ℝ) (h_avg_abc : average_abc A B C) (h_avg_ab : average_ab A B) (h_b : weight_b B) :
  (B + C) / 2 = 43 :=
sorry

end average_bc_l566_566107


namespace total_fish_weight_is_25_l566_566285

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end total_fish_weight_is_25_l566_566285


namespace distinct_real_roots_sum_l566_566002

theorem distinct_real_roots_sum (p r_1 r_2 : ℝ) (h_eq : ∀ x, x^2 + p * x + 18 = 0)
  (h_distinct : r_1 ≠ r_2) (h_root1 : x^2 + p * x + 18 = 0)
  (h_root2 : x^2 + p * x + 18 = 0) : |r_1 + r_2| > 6 :=
sorry

end distinct_real_roots_sum_l566_566002


namespace combined_population_l566_566812

-- Defining the conditions
def population_New_England : ℕ := 2100000

def population_New_York (p_NE : ℕ) : ℕ := (2 / 3 : ℚ) * p_NE

-- The theorem to be proven
theorem combined_population (p_NE : ℕ) (h1 : p_NE = population_New_England) : 
  population_New_York p_NE + p_NE = 3500000 :=
by
  sorry

end combined_population_l566_566812


namespace orthocentric_tetrahedron_equivalence_l566_566198

def isOrthocentricTetrahedron 
  (sums_of_squares_of_opposite_edges_equal : Prop) 
  (products_of_cosines_of_opposite_dihedral_angles_equal : Prop)
  (angles_between_opposite_edges_equal : Prop) : Prop :=
  sums_of_squares_of_opposite_edges_equal ∨
  products_of_cosines_of_opposite_dihedral_angles_equal ∨
  angles_between_opposite_edges_equal

theorem orthocentric_tetrahedron_equivalence
  (sums_of_squares_of_opposite_edges_equal 
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal : Prop) :
  isOrthocentricTetrahedron
    sums_of_squares_of_opposite_edges_equal
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal :=
sorry

end orthocentric_tetrahedron_equivalence_l566_566198


namespace range_of_b_l566_566735

open Real

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := - (1/2) * x^2 + b * log (x + 2)

-- Define the derivative of f
def f' (b : ℝ) (x : ℝ) : ℝ := -x + b / (x + 2)

-- Prove that if f is decreasing on (-1, +∞), then b ≤ -1
theorem range_of_b (b : ℝ) : (∀ x : ℝ, x > -1 → f' b x ≤ 0) → b ≤ -1 :=
by {
  intro h, 
  sorry
}

end range_of_b_l566_566735


namespace inequality_solution_equality_condition_l566_566834

theorem inequality_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_solution_equality_condition_l566_566834


namespace number_of_classes_l566_566952

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 := by
  sorry

end number_of_classes_l566_566952


namespace surface_area_black_cube_diff_l566_566421

theorem surface_area_black_cube_diff
  (X Y : ℕ)
  (hX : X = 6)
  (hY : Y = 13) :
  Y - X = 7 :=
by
  rw [hX, hY]
  norm_num

end surface_area_black_cube_diff_l566_566421


namespace total_fish_weight_is_25_l566_566286

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end total_fish_weight_is_25_l566_566286


namespace number_of_valid_guesses_equals_4536_l566_566957

-- Define the digits available.
def available_digits : List ℕ := [1, 1, 2, 2, 2, 3, 3, 4, 4]

-- Define the constraint that the length of each number must be at most 4.
def valid_partition (l : List ℕ) : Prop :=
  l.length ≤ 4

-- Define the conditions.
def condition (d1 d2 d3 : List ℕ) : Prop :=
  valid_partition d1 ∧ valid_partition d2 ∧ valid_partition d3 ∧
  (d1 ++ d2 ++ d3).length = 9 ∧
  (d1 ++ d2 ++ d3) ~ available_digits

-- Prove the total number of valid guesses is 4536 given the conditions.
theorem number_of_valid_guesses_equals_4536 : 
∃ d1 d2 d3 : List ℕ, condition d1 d2 d3 ∧ (d1.product (d2.product d3)).length = 4536 :=
sorry

end number_of_valid_guesses_equals_4536_l566_566957


namespace symmetric_circle_eq_a_l566_566125

theorem symmetric_circle_eq_a :
  ∀ (a : ℝ), (∀ x y : ℝ, (x^2 + y^2 - a * x + 2 * y + 1 = 0) ↔ (∃ x y : ℝ, (x - y = 1) ∧ ( x^2 + y^2 = 1))) → a = 2 :=
by
  sorry

end symmetric_circle_eq_a_l566_566125


namespace max_sum_ge_ordered_sum_l566_566376

theorem max_sum_ge_ordered_sum (n : ℕ) (a b : Fin n → ℝ) (A B : Fin n → ℝ) 
  (ha : ∀ i j : Fin n, i ≤ j → A i ≤ A j)
  (hA : ∃ (σ : Equiv.Perm (Fin n)), ∀ i : Fin n, A i = a (σ i))
  (hb : ∀ i j : Fin n, i ≤ j → B i ≥ B j)
  (hB : ∃ (τ : Equiv.Perm (Fin n)), ∀ i : Fin n, B i = b (τ i)) :
  Finset.max' (Finset.image (λ i : Fin n, a i + b i) Finset.univ)
    (Finset.nonempty_image_iff.mpr Finset.univ_nonempty) ≥ 
  Finset.max' (Finset.image (λ i : Fin n, A i + B i) Finset.univ)
    (Finset.nonempty_image_iff.mpr Finset.univ_nonempty) := sorry

end max_sum_ge_ordered_sum_l566_566376


namespace profit_percent_l566_566578

theorem profit_percent (P : ℝ) : 
    let cost_price := 46 * P
    let selling_price_per_pen := (99 / 100) * P
    let total_selling_price := 58 * selling_price_per_pen
    let profit := total_selling_price - cost_price
    let profit_percent := (profit / cost_price) * 100
in profit_percent = 24.65 :=
by
  sorry

end profit_percent_l566_566578


namespace measure_angle4_l566_566027

-- Define the angles and conditions
def angle1 : ℝ := 85
def angle2 : ℝ := 34
def angle3 : ℝ := 20
def angle5 : ℝ := sorry -- placeholder
def angle6 : ℝ := sorry -- placeholder

-- Define the sum of angles in a triangle
def triangle_angle_sum (a b c : ℝ) : Prop := a + b + c = 180

-- State the theorem
theorem measure_angle4 (h1 : angle1 = 85) (h2 : angle2 = 34) (h3 : angle3 = 20) :
  (∃ angle5 angle6, triangle_angle_sum angle1 angle2 angle3 ∧ 
                    triangle_angle_sum angle5 angle6 41 ∧ 
                    triangle_angle_sum 139 angle5 angle6) :=
begin
  sorry
end

end measure_angle4_l566_566027


namespace grouping_people_l566_566028

theorem grouping_people :
  let men := 4,
      women := 4,
      total := men + women,
      group1 := 2,
      group2 := 2,
      group3 := 4,
      ways_group1 := Nat.choose men 1 * Nat.choose women 1,
      ways_group2 := Nat.choose (men - 1) 1 * Nat.choose (women - 1) 1,
      ways_group3 := 1,
      total_ways := (ways_group1 * ways_group2 * ways_group3) / 2 in
  total_ways = 72 := by
  sorry

end grouping_people_l566_566028


namespace sqrt_range_real_l566_566396

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end sqrt_range_real_l566_566396


namespace proper_subsets_count_l566_566727

open Finset

theorem proper_subsets_count {α : Type*} {S : Finset α} (h : S = {2, 4, 6, 8} : Finset ℕ) :
  (S.powerset.filter (λ s, s ≠ S)).card = 15 :=
by sorry

end proper_subsets_count_l566_566727


namespace sum_of_midpoint_coordinates_l566_566182

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (8, 16)
  let (x2, y2) := (2, -8)
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 :=
by
  sorry

end sum_of_midpoint_coordinates_l566_566182


namespace greatest_a_for_integer_solutions_l566_566121

theorem greatest_a_for_integer_solutions :
  ∃ a : ℕ, 
    (∀ x : ℤ, x^2 + a * x = -21 → ∃ y : ℤ, y * (y + a) = -21) ∧ 
    ∀ b : ℕ, (∀ x : ℤ, x^2 + b * x = -21 → ∃ y : ℤ, y * (y + b) = -21) → b ≤ a :=
begin
  -- Proof goes here
  sorry
end

end greatest_a_for_integer_solutions_l566_566121


namespace probability_problem_l566_566560

noncomputable def prob_at_least_two_less_than_ten : ℚ :=
  let prob_less_than_ten := (9 : ℚ) / 20
  let prob_not_less_than_ten := (11 : ℚ) / 20
  let prob := λ k, nat.choose 5 k * (prob_less_than_ten ^ k) * (prob_not_less_than_ten ^ (5 - k))
  prob 2 + prob 3 + prob 4 + prob 5

theorem probability_problem :
  prob_at_least_two_less_than_ten = 157439 / 20000 := sorry

end probability_problem_l566_566560


namespace number_of_matches_in_first_set_l566_566949

theorem number_of_matches_in_first_set
  (avg_next_13_matches : ℕ := 15)
  (total_matches : ℕ := 35)
  (avg_all_matches : ℚ := 23.17142857142857)
  (x : ℕ := total_matches - 13) :
  x = 22 := by
  sorry

end number_of_matches_in_first_set_l566_566949


namespace centroid_expression_l566_566417

variables {V : Type*} [inner_product_space ℝ V]

-- Given vectors a and b
variables (a b : V)

-- Define the vectors AB and AC
def AB : V := a
def AC : V := b

-- Define the centroid property for the triangle ABC
def is_centroid (G A B C : V) : Prop :=
  G = (A + B + C) / 3

-- The main theorem to prove
theorem centroid_expression (G A B C : V) (h1 : AB = a) (h2 : AC = b) (hG_centroid : is_centroid G A B C) :
  G - A = (a + b) / 3 :=
by sorry

end centroid_expression_l566_566417


namespace factorize_expression_l566_566637

variable (b : ℝ)

theorem factorize_expression : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end factorize_expression_l566_566637


namespace band_member_contribution_l566_566307

def number_of_people : ℕ := 500
def ticket_price : ℝ := 30.0
def band_share_percentage : ℝ := 0.70
def number_of_band_members : ℕ := 4

def total_revenue : ℝ := number_of_people * ticket_price
def band_share : ℝ := band_share_percentage * total_revenue
def amount_per_band_member : ℝ := band_share / number_of_band_members

theorem band_member_contribution : amount_per_band_member = 2625 := by
  -- Proof goes here
  sorry

end band_member_contribution_l566_566307


namespace find_least_n_to_determine_p_l566_566235

theorem find_least_n_to_determine_p : ∃ n ∈ {1, 2, 3, 4}, 
  ∀ (p : ℕ) (seq : ℕ → ℕ),
    (1 ≤ p ∧ p ≤ 10 ∧ ∀ n, seq (n + p) = seq n) → 
    ∃ (indices : fin n → ℕ), 
      ((∀ i j, seq (indices i) = seq (indices j) ↔ (i % p = j % p) ∨ i % p = 0) ∧
       p = (argmin (λ q, ∀ (a b : fin 4), (seq (indices a) ≠ seq (indices b)) ↔ (a % q ≠ b % q)) (1 : ℕ) 10)) :=
  sorry

end find_least_n_to_determine_p_l566_566235


namespace no_solutions_Y_l566_566048

theorem no_solutions_Y (Y : ℕ) : 2 * Y + Y + 3 * Y = 14 ↔ false :=
by 
  sorry

end no_solutions_Y_l566_566048


namespace acute_angle_rhombus_l566_566173

theorem acute_angle_rhombus (α : ℝ) (acute : α > 0 ∧ α < (π / 2)):
  let β := 2 * arc_tan (2 * cos α) in
  0 < β ∧ β < π / 2 := 
sorry

end acute_angle_rhombus_l566_566173


namespace eleven_does_not_divide_choose_1000_500_l566_566921

-- Definition of the exponent of prime p in n!
def prime_exponent_in_factorial (p n : ℕ) : ℕ :=
  (List.range n).sum (λ k, n / p ^ (k + 1))

-- Definition of binomial coefficient (n choose k)
def binomial (n k : ℕ) : ℕ := n.choose k

-- Statement asserting that 11 does not divide binomial(1000, 500)
theorem eleven_does_not_divide_choose_1000_500 :
  ¬ (11 ∣ binomial 1000 500) := by
  sorry

end eleven_does_not_divide_choose_1000_500_l566_566921


namespace solve_eq_l566_566302

theorem solve_eq :
  { x : ℝ | (14 * x - x^2) / (x + 2) * (x + (14 - x) / (x + 2)) = 48 } =
  {4, (1 + Real.sqrt 193) / 2, (1 - Real.sqrt 193) / 2} :=
by
  sorry

end solve_eq_l566_566302


namespace pond_algae_coverage_l566_566023

/-- 
In a local pond, the amount of algae triples each day.
The pond was completely covered with algae on day 20.
Prove that on day 18, the pond was approximately 10% covered in algae,
implying it was about 90% algae-free.
-/
theorem pond_algae_coverage :
  ∃ k : ℕ, k = 18 ∧ 
  let coverage : ℕ → ℝ := λ n, 1 / (3 : ℝ)^((20 - n) : ℕ) in
  abs (coverage k - 0.1) < 0.01 :=
sorry

end pond_algae_coverage_l566_566023


namespace intersection_point_l566_566018

def polar_l1 (rho θ : ℝ) : Prop := rho * sin (θ - (Real.pi / 4)) = Real.sqrt 2 / 2
def parametric_l2 (t : ℝ) (x y : ℝ) : Prop := x = 1 - 2*t ∧ y = 2*t + 2

theorem intersection_point :
  (∃ (x y : ℝ), (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi ∧ polar_l1 ρ θ ∧ x = ρ * cos θ ∧ y = ρ * sin θ) ∧
  (∃ (t : ℝ), parametric_l2 t x y)) →
  (1, 2) :=
by
  sorry

end intersection_point_l566_566018


namespace family_ages_l566_566104

theorem family_ages :
  ∃ (x j b m F M : ℕ), 
    (b = j - x) ∧
    (m = j - 2 * x) ∧
    (j * b = F) ∧
    (b * m = M) ∧
    (j + b + m + F + M = 90) ∧
    (F = M + x ∨ F = M - x) ∧
    (j = 6) ∧ 
    (b = 6) ∧ 
    (m = 6) ∧ 
    (F = 36) ∧ 
    (M = 36) :=
sorry

end family_ages_l566_566104


namespace circumcircle_intersects_euler_line_in_two_points_l566_566440

-- Define a structure for points in a plane
structure Point := 
(x : ℝ) (y : ℝ)

-- Define the concept of a triangle and its properties
structure Triangle :=
(A : Point) (B : Point) (C : Point)
(is_acute : ∀ (P Q R : Point), (P = A ∧ Q = B ∧ R = C) → acute_angle(A, B, C))
(is_scalene : ∀ (P Q R S T : Point), (P = A ∧ Q = B ∧ R = C ∧ S ≠ T))

-- Define the incenter of a triangle
def incenter (T : Triangle) : Point := sorry

-- Define the Euler line of a triangle
def euler_line (T : Triangle) : set Point := sorry

-- Define circumcircle of a triangle
def circumcircle (A B C : Point) : set Point := sorry

-- Define the function to determine if two sets intersect at two distinct points
def intersects_at_two_points (S₁ S₂ : set Point) : Prop :=
∃ (P Q : Point), P ≠ Q ∧ P ∈ S₁ ∧ P ∈ S₂ ∧ Q ∈ S₁ ∧ Q ∈ S₂

-- The main theorem statement
theorem circumcircle_intersects_euler_line_in_two_points (T : Triangle) 
  (h_acute : T.is_acute T.A T.B T.C)
  (h_scalene : T.is_scalene T.A T.B T.C T.A T.B) :
  intersects_at_two_points (circumcircle T.B (incenter T) T.C) (euler_line T) :=
sorry

end circumcircle_intersects_euler_line_in_two_points_l566_566440


namespace sum_c_k_squared_l566_566620

noncomputable def c_k (k : ℕ) : ℝ :=
  k + (1 / (3 * k + (1 / (3 * k + (1 / (3 * k + ...)))))

theorem sum_c_k_squared : 
  ∑ k in finset.range 15, (c_k k) ^ 2 = 1255 :=
sorry

end sum_c_k_squared_l566_566620


namespace infinite_series_equivalence_l566_566438

theorem infinite_series_equivalence (x y : ℝ) (hy : y ≠ 0 ∧ y ≠ 1) 
  (series_cond : ∑' n : ℕ, x / (y^(n+1)) = 3) :
  ∑' n : ℕ, x / ((x + 2*y)^(n+1)) = 3 * (y - 1) / (5*y - 4) := 
by
  sorry

end infinite_series_equivalence_l566_566438


namespace series_remainder_is_zero_l566_566267

theorem series_remainder_is_zero :
  let a : ℕ := 4
  let d : ℕ := 6
  let n : ℕ := 17
  let l : ℕ := a + d * (n - 1) -- last term
  let S : ℕ := n * (a + l) / 2 -- sum of the series
  S % 17 = 0 := by
  sorry

end series_remainder_is_zero_l566_566267


namespace parabola_axis_of_symmetry_is_x_eq_1_l566_566657

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end parabola_axis_of_symmetry_is_x_eq_1_l566_566657


namespace max_integer_radius_l566_566738

theorem max_integer_radius (r : ℝ) (h : π * r^2 < 90 * π) : ∃ n : ℕ, n = 9 ∧ ↑n ≤ r ∧ r < ↑(n + 1) :=
by {
  sorry,
}

end max_integer_radius_l566_566738


namespace cone_submersion_height_eq_l566_566570

-- Define specific gravities, heights, and radius as parameters
variables (s s' m r : ℝ)

-- Define height of the sinking cone
def submerged_height (s s' m : ℝ) : ℝ :=
  m * (1 - (Real.cbrt ((s' - s) / s')))

-- Prove that the height to which the cone will sink equals the computed submerged_height
theorem cone_submersion_height_eq :
  s < s' →
  ∀ {h : ℝ}, h = submerged_height s s' m →
  h = m * (1 - (Real.cbrt ((s' - s) / s'))) :=
by {
  intros,
  sorry
}

end cone_submersion_height_eq_l566_566570


namespace train_crossing_time_approx_l566_566767

noncomputable def train_time_to_cross_pole
  (train_length : ℝ)
  (train_speed_kmph : ℝ) : ℝ :=
  let speed_m_per_s := train_speed_kmph * 1000 / 3600
  in train_length / speed_m_per_s

theorem train_crossing_time_approx
  (train_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 135 → train_speed_kmph = 56.5 →
  abs (train_time_to_cross_pole train_length train_speed_kmph - 8.6) < 0.1 :=
by
  intros h_length h_speed
  rw [h_length, h_speed]
  compute
  sorry

end train_crossing_time_approx_l566_566767


namespace geometric_sequence_monotonically_decreasing_inequality_l566_566144

-- Conditions
def a : ℕ+ → ℤ
def b : ℕ+ → ℤ

axiom a1 : a 1 = 1
axiom b1 : b 1 = 1
axiom a_recur (n : ℕ+) : a (n + 1) = a n + 2 * b n
axiom b_recur (n : ℕ+) : b (n + 1) = a n + b n

-- Question (I)
theorem geometric_sequence (n : ℕ+) :
  {a (n : ℕ) ^ 2 - 2 * b (n : ℕ) ^ 2} is_geometric_sequence_with_common_ratio (-1) := sorry

-- Question (II)
def c (n : ℕ+) := abs (↑(a n) / ↑(b n) - real.sqrt 2)
theorem monotonically_decreasing (n : ℕ+) :
  ∀ m ≥ n, c (m + 1) < c m := sorry

–- Question (III)
theorem inequality (n : ℕ+) :
  ∑ i in finset.range n.succ, 1/(i+2)^2 * ∑ j in finset.range (i+1), j^2 / a (j + 1) <
  ∑ i in finset.range n.succ, i / a (i + 1) := sorry

end geometric_sequence_monotonically_decreasing_inequality_l566_566144


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566316

-- Part 1: Total unique three-digit numbers
theorem unique_three_digit_numbers : 
  (finset.univ.card) * (finset.univ.erase 0).card * (finset.univ.erase 0).erase 9.card = 648 :=
sorry

-- Part 2: Total unique three-digit odd numbers
theorem unique_three_digit_odd_numbers :
  5 * ((finset.univ.erase 5).erase 0).card * ((finset.univ.erase 5).erase 0).erase 9.card = 320 :=
sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l566_566316


namespace ratio_proof_l566_566523

noncomputable def ratio_of_two_numbers (x y : ℝ): ℝ :=
  x / y

theorem ratio_proof (x y : ℝ) (h : x^2 = 8 * y^2 - 224) : ratio_of_two_numbers x y = sqrt(8 - 224/(y^2)) :=
by
  sorry

end ratio_proof_l566_566523


namespace ratio_a_d_l566_566930

theorem ratio_a_d (a b c d : ℕ) 
  (hab : a * 4 = b * 3) 
  (hbc : b * 9 = c * 7) 
  (hcd : c * 7 = d * 5) : 
  a * 12 = d :=
sorry

end ratio_a_d_l566_566930


namespace max_effective_time_min_b_for_effectiveness_l566_566729

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then (4 + x) / (4 - x)
  else if 2 < x ∧ x ≤ 5 then 5 - x
  else 0

theorem max_effective_time (a : ℝ) (x : ℝ) : 
  1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ x ∧ x ≤ 5 ∧ 4 * f x ≥ 4 → 0 ≤ x ∧ x ≤ 4 :=
sorry

theorem min_b_for_effectiveness (b x : ℝ) :
  0 ≤ x ∧ x ≤ 2 ∧ 4 - 2 * x + b * ((4 + x) / (4 - x)) ≥ 4 → b ≥ 24 - 16 * Real.sqrt 2 :=
sorry

end max_effective_time_min_b_for_effectiveness_l566_566729


namespace time_for_B_work_alone_l566_566947

def work_rate_A : ℚ := 1 / 6
def work_rate_combined : ℚ := 1 / 3
def work_share_C : ℚ := 1 / 8

theorem time_for_B_work_alone : 
  ∃ x : ℚ, (work_rate_A + 1 / x = work_rate_combined - work_share_C) → x = 24 := 
sorry

end time_for_B_work_alone_l566_566947


namespace length_of_chord_l566_566226

noncomputable def parabola : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

noncomputable def focus : ℝ × ℝ := (1, 0)

noncomputable def slope : ℝ := real.tan (real.pi / 6) -- tan 30 degrees in radians

-- Define the line passing through the focus with the calculated slope
noncomputable def line (p : ℝ × ℝ) : set (ℝ × ℝ) := 
  {q : ℝ × ℝ | q.2 = slope * (q.1 - p.1)}

-- Define the line passing through the focus
noncomputable def line_through_focus : set (ℝ × ℝ) :=
  line focus

theorem length_of_chord :
  line_through_focus ∩ parabola ≠ ∅ →
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ parabola ∧ B ∈ parabola ∧
  dist A B = 16 :=
by
  sorry

end length_of_chord_l566_566226


namespace matrix_inverse_multiplication_l566_566680

variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![![2, 0], ![0, 1]])
variable (B : Matrix (Fin 2) (Fin 2) ℝ := !![![1, 2], ![-1, 5]])

theorem matrix_inverse_multiplication :
  A⁻¹ ⬝ B = !![![1 / 2, 1], ![-1 / 2, 5]] :=
by
  sorry

end matrix_inverse_multiplication_l566_566680


namespace map_area_l566_566591

def length : ℕ := 5
def width : ℕ := 2
def area_of_map (length width : ℕ) : ℕ := length * width

theorem map_area : area_of_map length width = 10 := by
  sorry

end map_area_l566_566591


namespace part_I_part_II_l566_566371

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2

noncomputable def h_deriv (a : ℝ) (x : ℝ) : ℝ := 
  deriv (λ x : ℝ, log x + a * x^2) x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  g(a, x) + h_deriv(a, x)

theorem part_I (x : ℝ) (hx : x > 0) : 
  f 0 (1 / 2) = 2 - 2 * log 2 :=
sorry

theorem part_II (a x1 x2 m : ℝ) 
  (h₀ : -8 < a ∧ a < -2) 
  (h₁ : 1 ≤ x1 ∧ x1 ≤ 3) 
  (h₂ : 1 ≤ x2 ∧ x2 ≤ 3)
  (h₃ : |f(a, x1) - f(a, x2)| > (m + log 3) * a - 2 * log 3 + (2 / 3) * log (-a)) : 
  m ∈ Ioi ((2 / (3 * exp 2)) - 4) :=
sorry

end part_I_part_II_l566_566371


namespace students_passed_correct_l566_566876

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l566_566876


namespace range_of_a_if_f_zero_le_one_monotonicity_of_f_number_of_zeros_of_f_plus_abs_l566_566796

noncomputable def f (a x : ℝ) : ℝ :=
  (x - a)^2 + abs (x - a) - a * (a - 1)

-- Proof Problem 1: Range of values for a if f(0) ≤ 1
theorem range_of_a_if_f_zero_le_one (a : ℝ) (h : f a 0 ≤ 1) :
  a ∈ set.Iic (1 / 2) :=
sorry

-- Proof Problem 2: Monotonicity of f(x)
theorem monotonicity_of_f (a x : ℝ) :
  (∀ x ∈ set.Ioi a, monotone (f a x)) ∧ (∀ x ∈ set.Iio a, antimonotone (f a x)) :=
sorry

-- Proof Problem 3: Number of zeros of f(x) + |x| when a > 2
theorem number_of_zeros_of_f_plus_abs (a : ℝ) (h : a > 2) :
  ∃ n : ℕ, n = 2 ∧ (∀ z : ℝ, f a z + abs z = 0 → z ∈ finset.fin_range n) :=
sorry

end range_of_a_if_f_zero_le_one_monotonicity_of_f_number_of_zeros_of_f_plus_abs_l566_566796


namespace probability_of_multiples_of_6_or_8_l566_566096

-- Each ball is numbered from 1 to 60.
def balls : Finset ℕ := Finset.range 61 -- this defines the set {0, 1, 2, ..., 60}

-- The subset of balls that are divisible by 6.
def divisible_by_6 : Finset ℕ := balls.filter (λ n, n % 6 = 0)

-- The subset of balls that are divisible by 8.
def divisible_by_8 : Finset ℕ := balls.filter (λ n, n % 8 = 0)

-- The subset of balls that are divisible by both 6 and 8.
def divisible_by_24 : Finset ℕ := balls.filter (λ n, n % 24 = 0)

-- Computing the probability
def probability_divisible_by_6_or_8 := 
  (divisible_by_6.card + divisible_by_8.card - divisible_by_24.card) / balls.card

-- Theorem statement
theorem probability_of_multiples_of_6_or_8 : probability_divisible_by_6_or_8 = 1 / 4 :=
by sorry

end probability_of_multiples_of_6_or_8_l566_566096


namespace total_savings_percentage_l566_566956

theorem total_savings_percentage
  (original_coat_price : ℕ) (original_pants_price : ℕ)
  (coat_discount_percent : ℚ) (pants_discount_percent : ℚ)
  (original_total_price : ℕ) (total_savings : ℕ)
  (savings_percentage : ℚ) :
  original_coat_price = 120 →
  original_pants_price = 60 →
  coat_discount_percent = 0.30 →
  pants_discount_percent = 0.60 →
  original_total_price = original_coat_price + original_pants_price →
  total_savings = original_coat_price * coat_discount_percent + original_pants_price * pants_discount_percent →
  savings_percentage = (total_savings / original_total_price) * 100 →
  savings_percentage = 40 := 
by
  intros
  sorry

end total_savings_percentage_l566_566956


namespace part1_cos_B_part2_A_l566_566409

variables (A B C : ℝ) (a b c : ℝ)
hypothesis h₁ : A + B + C = π
hypothesis h₂ : A < π / 2 ∧ B < π / 2 ∧ C < π / 2 -- Acute triangle
hypothesis h₃ : b = 5
hypothesis h₄ : b = c -- From part (1) conclusion that B = C
hypothesis h₅ : cos A = 3 / 4
hypothesis h₆ : cos A = 1 / 3
hypothesis h₇ : cos B = cos C -- Derived from given condition

noncomputable def cos_B_part1 : ℝ :=
  if cos A = 1 / 3 then cos B else 0

theorem part1_cos_B (h : cos_A = 1 / 3) : cos_B_part1 = sqrt 3 / 3 :=
sorry

noncomputable def a_part2 : ℝ :=
  if b = 5 ∧ cos A = 3 / 4 then sqrt ((b^2 + b^2 - 2 * b * b * cos A) / 2) else 0
  -- Note: sqrt((_)/2) to correct typo where 50 is 25+25 divided later

theorem part2_A (h : b = 5 ∧ cos_A = 3 / 4) : a_part2 = 5 * sqrt 2 / 2 :=
sorry

end part1_cos_B_part2_A_l566_566409


namespace complex_power_sum_eq_five_l566_566065

noncomputable def w : ℂ := sorry

theorem complex_power_sum_eq_five (h : w^3 + w^2 + 1 = 0) : 
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 :=
sorry

end complex_power_sum_eq_five_l566_566065


namespace symmetry_center_of_f_l566_566504

def f (x : ℝ) : ℝ := (1/2) * Real.cos (2 * x) + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_center_of_f : f (-Real.pi / 12) = 0 :=
by
  sorry

end symmetry_center_of_f_l566_566504


namespace tablespoons_per_mocktail_l566_566049

-- Define the parameters and conditions given in the problem
def limes_per_dollar := 3
def lime_cost := 1.0
def total_spent := 5.0
def juice_per_lime := 2 -- tablespoons
def days := 30

-- Define the theorem stating the problem
theorem tablespoons_per_mocktail : 
  let limes_bought := (total_spent / (lime_cost / limes_per_dollar)).to_int in
  let total_juice := limes_bought * juice_per_lime in
  let mocktails := days in
  (total_juice / mocktails) = 1 :=
by
  -- Definitions
  let limes_bought := (total_spent / (lime_cost / limes_per_dollar)).to_int
  let total_juice := limes_bought * juice_per_lime
  let mocktails := days
  show (total_juice / mocktails) = 1, by sorry

end tablespoons_per_mocktail_l566_566049


namespace proposition_problem_l566_566246

open real

theorem proposition_problem :
  (∀ x : ℝ, ¬(x^2 + 1 > 3 * x) ↔ x^2 + 1 ≤ 3 * x) ∧
  (∀ a : ℝ, (cos (2 * a * x) = cos (π - 2 * a * x)) → a = 1 ∨ a = -1) ∧
  (¬(∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + 2 * x ≥ a * x ↔ (min (x^2 + 2 * x) ≥ max (a * x)))) ∧
  (∀ A B : ℝ (A > B ↔ sin A > sin B)) :=
by
  sorry

end proposition_problem_l566_566246


namespace find_alpha_find_range_l566_566682

noncomputable theory

variable {α : ℝ}

theorem find_alpha (h1 : sin α * tan α = 3 / 2) (h2 : 0 < α ∧ α < π) : α = π / 3 :=
sorry

theorem find_range (α : ℝ) (hα : α = π / 3) :
  set.range (λ x, 4 * cos x * cos (x - α)) = set.Icc 2 3 :=
sorry

end find_alpha_find_range_l566_566682


namespace trapezium_area_l566_566642

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 285 :=
by {
  sorry
}

end trapezium_area_l566_566642


namespace smallest_multiple_of_4_and_13_l566_566895

theorem smallest_multiple_of_4_and_13 : ∃ n : ℕ, n > 0 ∧ (4 ∣ n) ∧ (13 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (4 ∣ m) ∧ (13 ∣ m)) → n ≤ m :=
begin
  use 52,
  split,
  { exact nat.pos_of_ne_zero (by norm_num), }, -- n > 0
  split,
  { exact dvd.intro 13 (by norm_num), }, -- 4 ∣ 52
  split,
  { exact dvd.intro 4 (by norm_num), }, -- 13 ∣ 52
  { intros m hm,
    cases hm with hm_pos hm',
    cases hm' with hm4 hm13,
    rw nat.dvd_lcm_left 4 13 at hm4,
    rw nat.dvd_lcm_right 4 13 at hm13,
    exact le_of_dvd hm_pos (nat.lcm_dvd hm4 hm13), -- n ≤ m
  }
end

end smallest_multiple_of_4_and_13_l566_566895


namespace geom_seq_sum_l566_566022

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l566_566022


namespace bob_grade_is_35_l566_566779

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l566_566779


namespace monday_dressing_time_l566_566821

theorem monday_dressing_time 
  (Tuesday_time Wednesday_time Thursday_time Friday_time Old_average_time : ℕ)
  (H_tuesday : Tuesday_time = 4)
  (H_wednesday : Wednesday_time = 3)
  (H_thursday : Thursday_time = 4)
  (H_friday : Friday_time = 2)
  (H_average : Old_average_time = 3) :
  ∃ Monday_time : ℕ, Monday_time = 2 :=
by
  let Total_time_5_days := Old_average_time * 5
  let Total_time := 4 + 3 + 4 + 2
  let Monday_time := Total_time_5_days - Total_time
  exact ⟨Monday_time, sorry⟩

end monday_dressing_time_l566_566821


namespace keith_receives_two_messages_l566_566030

noncomputable def messages_received_by_Keith : ℕ := 8 * (27 / 94.5 : ℝ)

theorem keith_receives_two_messages :
  messages_received_by_Keith = 2 :=
by
  -- Define L as the number of messages Juan sends to Laurence
  let L := (27 / 94.5 : ℝ) in

  -- Calculate the number of messages Keith receives from Juan
  have h : ℝ := 8 * L,

  -- Show that the result is equal to 2 using real division and multiplication
  have h_eq : h = 2 := by
    calc
      h = 8 * (27 / 94.5) : by rfl
      ... = (8 * 27) / 94.5 : by ring
      ... = 216 / 94.5 : by rfl
      ... = 2 : by norm_num,

  -- Conclude that messages_received_by_Keith equals 2
  show messages_received_by_Keith = 2, by
    rw [h_eq],
    rfl

end keith_receives_two_messages_l566_566030


namespace new_student_weight_l566_566477

theorem new_student_weight (W_new : ℝ) (W : ℝ) (avg_decrease : ℝ) (num_students : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_decrease = 5 → old_weight = 86 → num_students = 8 →
  W_new = W - old_weight + new_weight → W_new = W - avg_decrease * num_students →
  new_weight = 46 :=
by
  intros avg_decrease_eq old_weight_eq num_students_eq W_new_eq avg_weight_decrease_eq
  rw [avg_decrease_eq, old_weight_eq, num_students_eq] at *
  sorry

end new_student_weight_l566_566477


namespace number_of_integers_l566_566297

theorem number_of_integers (n : ℤ) : 
  (16 < n^2) → (n^2 < 121) → n = -10 ∨ n = -9 ∨ n = -8 ∨ n = -7 ∨ n = -6 ∨ n = -5 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 := 
by
  sorry

end number_of_integers_l566_566297


namespace min_ab_l566_566485

theorem min_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 1 / b = 1) : ab = 4 :=
  sorry

end min_ab_l566_566485


namespace probability_tiles_A_B_l566_566894

def is_favorable_draw_A (n : ℕ) : Prop := n < 20
def is_favorable_draw_B (m : ℕ) : Prop := (m % 2 = 1) ∨ (m > 45)

def tiles_A := Finset.range 31 -- tiles 0 to 30 (numbers 1 to 30)
def tiles_B := Finset.range' 21 51 -- tiles 21 to 50

def count_A_favorable := (tiles_A.filter is_favorable_draw_A).card
def count_B_favorable := (tiles_B.filter is_favorable_draw_B).card
def total_A := tiles_A.card
def total_B := tiles_B.card

def prob_A := count_A_favorable / total_A
def prob_B := count_B_favorable / total_B

theorem probability_tiles_A_B:
  count_A_favorable = 19 ∧
  count_B_favorable = 19 ∧
  total_A = 30 ∧
  total_B = 30 →
  prob_A * prob_B = (361 : ℚ) / (900 : ℚ) := by
  sorry

end probability_tiles_A_B_l566_566894


namespace triangle_area_l566_566622

def Point := ℝ × ℝ

def area_of_parallelogram (u v : Point) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

def area_of_triangle (a b c : Point) : ℝ :=
  let u := (fst a - fst c, snd a - snd c)
  let v := (fst b - fst c, snd b - snd c)
  (area_of_parallelogram u v) / 2

theorem triangle_area :
  area_of_triangle (2, -3) (8, 6) (5, 1) = 1.5 :=
by
  sorry

end triangle_area_l566_566622


namespace machine_parts_probabilities_l566_566902

-- Define the yield rates for the two machines
def yield_rate_A : ℝ := 0.8
def yield_rate_B : ℝ := 0.9

-- Define the probabilities of defectiveness for each machine
def defective_probability_A := 1 - yield_rate_A
def defective_probability_B := 1 - yield_rate_B

theorem machine_parts_probabilities :
  (defective_probability_A * defective_probability_B = 0.02) ∧
  (((yield_rate_A * defective_probability_B) + (defective_probability_A * yield_rate_B)) = 0.26) ∧
  (defective_probability_A * defective_probability_B + (1 - (defective_probability_A * defective_probability_B)) = 1) :=
by
  sorry

end machine_parts_probabilities_l566_566902


namespace digits_not_distinct_l566_566968

theorem digits_not_distinct (n : ℕ) (hn : 1000000000 ≤ n^2 + 1 ∧ n^2 + 1 < 10000000000) : 
  ¬ list.nodup (nat.digits 10 (n^2 + 1)) :=
by sorry

end digits_not_distinct_l566_566968


namespace train_cross_platform_l566_566927

variable (train_length: ℝ) (time_to_cross_tree: ℝ) (platform_length: ℝ)

def train_speed (train_length: ℝ) (time_to_cross_tree: ℝ) : ℝ := train_length / time_to_cross_tree

def total_distance (train_length: ℝ) (platform_length: ℝ) : ℝ := train_length + platform_length

def time_to_pass_platform (train_length: ℝ) (time_to_cross_tree: ℝ) (platform_length: ℝ) : ℝ :=
  (train_length + platform_length) / (train_length / time_to_cross_tree)

theorem train_cross_platform :
  train_time_to_cross_platform 1200 120 700 = 190 := by
  sorry

end train_cross_platform_l566_566927


namespace domain_of_g_l566_566913

noncomputable def g (x : ℝ) : ℝ := (x - 2) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | ∃ (u v : ℝ), u < x < 2 ∨ 3 < x < v} = set_of g := 
sorry

end domain_of_g_l566_566913


namespace zero_all_entries_in_table_l566_566471

theorem zero_all_entries_in_table 
  (m n : ℕ) 
  (A : ℕ → ℕ → ℤ)
  (op : (ℕ → ℕ) → int → A → A)
  (h_base : ∀ (A₃ : ℕ → ℕ → ℤ), (∀ i j, i < 3 → j < 3 → A₃ i j = 0)) :
  ∃ (op_seq : list (ℕ → ℕ) × int), ∀ i j, 0 ≤ i < m → 0 ≤ j < n → (op_seq.foldl (λ A op_i, op (fst op_i) (snd op_i) A) A) i j = 0 := 
by
  sorry

end zero_all_entries_in_table_l566_566471


namespace max_postcards_sent_l566_566166

-- Define the problem conditions
def num_students : ℕ := 30
def mutual_friendship (x y : ℕ) : Prop := x ≠ y → ¬(x = y)
def no_triangle (P : finset (ℕ × ℕ)) : Prop := ∀ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ¬((a, b) ∈ P ∧ (b, c) ∈ P ∧ (a, c) ∈ P)

-- The proof statement that verifies the maximum number of postcards sent
theorem max_postcards_sent : ∃ (P : finset (ℕ × ℕ)), |P| = 450 ∧ no_triangle P :=
sorry

end max_postcards_sent_l566_566166


namespace students_passed_correct_l566_566877

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l566_566877


namespace total_cost_sandwiches_and_sodas_l566_566197

theorem total_cost_sandwiches_and_sodas :
  let price_sandwich : Real := 2.49
  let price_soda : Real := 1.87
  let quantity_sandwich : ℕ := 2
  let quantity_soda : ℕ := 4
  (quantity_sandwich * price_sandwich + quantity_soda * price_soda) = 12.46 := 
by
  sorry

end total_cost_sandwiches_and_sodas_l566_566197


namespace cap_partition_l566_566483

theorem cap_partition : 
  ∃ (red blue : Set String), 
    red ≠ ∅ ∧ blue ≠ ∅ ∧ 
    red ∪ blue = {'Šárka', 'Světlana', 'Marta', 'Maruška', 'Monika'} ∧ 
    red ∩ blue = ∅ ∧ 
    (∑ girl in red, match girl with 
                          | "Šárka" => 20 
                          | "Světlana" => 29 
                          | "Marta" => 31 
                          | "Maruška" => 49 
                          | "Monika" => 51
                          | _ => 0 
                      end = 60) ∧ 
    (∑ girl in blue, match girl with 
                          | "Šárka" => 20 
                          | "Světlana" => 29 
                          | "Marta" => 31 
                          | "Maruška" => 49 
                          | "Monika" => 51
                          | _ => 0 
                      end = 120) :=
by
  sorry

end cap_partition_l566_566483


namespace find_a_for_extraneous_roots_find_a_for_no_solution_l566_566696

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution_l566_566696


namespace average_age_l566_566850

theorem average_age (n_students : ℕ) (n_parents : ℕ) (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  n_students = 40 →
  n_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 35 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents) = 25.8 :=
by 
  intros h_students h_parents h_avg_students h_avg_parents
  rw [h_students, h_parents, h_avg_students, h_avg_parents]
  norm_num
  sorry

end average_age_l566_566850


namespace books_left_over_l566_566401

def total_books (box_count : ℕ) (books_per_box : ℤ) : ℤ :=
  box_count * books_per_box

theorem books_left_over
  (box_count : ℕ)
  (books_per_box : ℤ)
  (new_box_capacity : ℤ)
  (books_total : ℤ := total_books box_count books_per_box) :
  box_count = 1500 →
  books_per_box = 35 →
  new_box_capacity = 43 →
  books_total % new_box_capacity = 40 :=
by
  intros
  sorry

end books_left_over_l566_566401


namespace zoltan_incorrect_answers_l566_566579

/-- Total number of questions: 50, Points for each correct answer: +4, Points for each incorrect answer: -1, Points for each unanswered question: 0, Zoltan answered 45 questions, Zoltan's total score was 135 points. Prove that the number of questions Zoltan answered incorrectly is 9. -/
theorem zoltan_incorrect_answers (total_questions answered_questions points_per_correct points_per_incorrect points_per_unanswered total_score : ℕ)
    (h1 : total_questions = 50) 
    (h2 : points_per_correct = 4) 
    (h3 : points_per_incorrect = (-1)) 
    (h4 : points_per_unanswered = 0) 
    (h5 : answered_questions = 45) 
    (h6 : total_score = 135) 
    : ∃ incorrect_answers : ℕ, incorrect_answers = 9 :=
by
  sorry

end zoltan_incorrect_answers_l566_566579


namespace sum_ALGEBRA_equals_5_l566_566859

def letter_value (ch : Char) : ℤ :=
  let pos := ch.toNat - 'A'.toNat + 1 in
  match pos % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 0 => -3
  | _ => 0 -- For completeness, though unreachable

noncomputable def sum_of_word_values (word : String) : ℤ :=
  word.foldl (λ acc ch => acc + (letter_value ch)) 0

theorem sum_ALGEBRA_equals_5 : sum_of_word_values "ALGEBRA" = 5 := by
  sorry

end sum_ALGEBRA_equals_5_l566_566859


namespace max_single_player_salary_is_426000_l566_566025

noncomputable def max_single_player_salary (total_salary_cap : ℤ) (min_salary : ℤ) (num_players : ℤ) : ℤ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_single_player_salary_is_426000 :
  ∃ y, max_single_player_salary 800000 17000 23 = y ∧ y = 426000 :=
by
  sorry

end max_single_player_salary_is_426000_l566_566025


namespace smallest_positive_integer_no_diff_sets_l566_566301

theorem smallest_positive_integer_no_diff_sets :
  let N := 2^2023 + 2^2022
  in ∀ (A B : set ℕ), 
  A ≠ B →
  A ⊆ { x | ∃ k, x = 2^k ∧ k ≤ 2023 ∨ x = N } →
  B ⊆ { x | ∃ k, x = 2^k ∧ k ≤ 2023 ∨ x = N } →
  ∃ (n1 n2 : ℕ), n1 = n2 → (∃ m1, A = set.to_finset (finset.range (2^m1).succ)) ∧
  ∃ m2, B = set.to_finset (finset.range (2^m2).succ) →
  finset.card (set.to_finset A) = finset.card (set.to_finset B) →
  finset.sum (set.to_finset A) id = finset.sum (set.to_finset B) id →
  False :=
by sorry

end smallest_positive_integer_no_diff_sets_l566_566301


namespace mayo_bottle_count_l566_566194

-- Define the given ratio and the number of ketchup bottles
def ratio_ketchup : ℕ := 3
def ratio_mustard : ℕ := 3
def ratio_mayo : ℕ := 2
def num_ketchup_bottles : ℕ := 6

-- Define the proof problem: The number of mayo bottles
theorem mayo_bottle_count :
  (num_ketchup_bottles / ratio_ketchup) * ratio_mayo = 4 :=
by sorry

end mayo_bottle_count_l566_566194


namespace part1_part2_l566_566370
noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem part1 : {x : ℝ | f x ≥ 3} = {x | x ≤ 0} ∪ {x | x ≥ 3} :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x ≤ -a^2 + a + 7) ↔ -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end part1_part2_l566_566370


namespace greatest_perimeter_of_strips_l566_566823

theorem greatest_perimeter_of_strips :
  let base := 10
  let height := 12
  let half_base := base / 2
  let right_triangle_area := (base / 2 * height) / 2
  let number_of_pieces := 10
  let sub_area := right_triangle_area / (number_of_pieces / 2)
  let h1 := (2 * sub_area) / half_base
  let hypotenuse := Real.sqrt (h1^2 + (half_base / 2)^2)
  let perimeter := half_base + 2 * hypotenuse
  perimeter = 11.934 :=
by
  sorry

end greatest_perimeter_of_strips_l566_566823


namespace negation_of_universal_l566_566489

theorem negation_of_universal :
  ¬ (∀ x : ℝ, 2 * x ^ 2 + x - 1 ≤ 0) ↔ ∃ x : ℝ, 2 * x ^ 2 + x - 1 > 0 := 
by 
  sorry

end negation_of_universal_l566_566489


namespace proof_problem_l566_566437

variables (l m n : Type) (α β γ : Type)
variables [Perp l α] [Perp m α] [ParallelProp l m]
variables [Subset m β] [Perp m l] [Projection n l β] [PerpProp m n]

theorem proof_problem :
  (∀ (l m : Type) [Perp l α] [Perp m α], Parallel l m) ∧
  (∀ (m n l : Type) [Subset m β] [Projection n l β] [Perp l m], Perp m n) :=
begin
  split,
  { intros l m Plα Pmα,
    exact ParallelProp l m, },
  { intros m n l SmB Pnlβ Plm,
    exact PerpProp m n, }
end

end proof_problem_l566_566437


namespace bob_grade_is_35_l566_566773

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l566_566773


namespace normal_trading_cards_l566_566134

variable (x : Int)

-- Conditions
axiom h1 : ∀ (x : Int), (x + 3922) + x = 46166

-- Goal
theorem normal_trading_cards : x = 21122 :=
by
  have h2 : 2 * x + 3922 = 46166 := h1 x
  have h3 : 2 * x = 46166 - 3922 := by linarith
  have h4 : 2 * x = 42244 := by norm_num at h3
  have h5 : x = 42244 / 2 := by linarith
  have h6 : x = 21122 := by norm_num at h5
  exact h6

end normal_trading_cards_l566_566134


namespace trapezoid_area_l566_566988

-- Define the bases and height
variables (x : ℝ)

-- Define the bases and height using given conditions
def base1 := 3 * x
def base2 := 5 * x
def height := x

-- Prove the area of the trapezoid is 4 * x^2
theorem trapezoid_area (x : ℝ) :
    (height * ((base1 + base2) / 2)) = 4 * x^2 :=
by 
  -- We can include the mathematical proof here in practice, but we'll use sorry for now
  sorry

end trapezoid_area_l566_566988


namespace find_f_minus1_find_max_min_interval_l566_566388

noncomputable def polynomial := λ x : ℝ, x^2 - 4*x + 3

theorem find_f_minus1 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 - 4*x + 3)
  (h1 : f 1 = 0)
  (h3 : f 3 = 0) : 
  f (-1) = 8 :=
by 
  sorry

theorem find_max_min_interval 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 - 4*x + 3)
  (h1 : f 1 = 0)
  (h3 : f 3 = 0) :
  ∃ (max_val min_val : ℝ),
    max_val = 3 ∧ min_val = -1 ∧ 
    (∀ x ∈ set.Icc 2 4, min_val ≤ f x ∧ f x ≤ max_val) :=
by
  sorry

end find_f_minus1_find_max_min_interval_l566_566388


namespace geom_seq_cosine_l566_566359

theorem geom_seq_cosine (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_cond : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * real.pi) :
  real.cos (a 2 * a 12) = 1 / 2 :=
by sorry

end geom_seq_cosine_l566_566359


namespace acute_triangle_LMN_l566_566481

theorem acute_triangle_LMN (A B C L M N : Type) 
  (hABC : is_triangle A B C) 
  (hLMN : is_intersection_of_external_angle_bisectors A B C L M N) : 
  is_acute_triangle L M N := 
sorry

end acute_triangle_LMN_l566_566481


namespace find_segment_lengths_l566_566451

variable (A B C : ℝ)
variable (ab ac bc : ℝ)

axiom collinear : A - B + B - C = A - C
axiom ab_eq_5 : ab = 5
axiom ac_eq_three_half_bc : ac = 3 / 2 * bc
axiom ab_def : ab = (B - A).abs
axiom ac_def : ac = (C - A).abs
axiom bc_def : bc = (C - B).abs

theorem find_segment_lengths (A B C ab ac bc : ℝ)
  (collinear : A - B + B - C = A - C)
  (ab_eq_5 : ab = 5)
  (ac_eq_three_half_bc : ac = 3 / 2 * bc)
  (ab_def : ab = (B - A).abs)
  (ac_def : ac = (C - A).abs)
  (bc_def : bc = (C - B).abs) :
  (bc = 10 ∧ ac = 15) ∨ (bc = 2 ∧ ac = 3) :=
by
  sorry

end find_segment_lengths_l566_566451


namespace area_of_L_shaped_sheetrock_is_28_l566_566971

-- Define the given conditions
def length_body : ℝ := 6  -- Length of the main rectangular body in feet
def width_body_inches : ℝ := 60  -- Width of the main rectangular body in inches
def length_cutout : ℝ := 2  -- Length of the cutout in feet
def width_cutout_inches : ℝ := 12  -- Width of the cutout in inches
def inches_in_foot : ℝ := 12  -- Conversion factor

-- Convert widths to feet
def width_body : ℝ := width_body_inches / inches_in_foot
def width_cutout : ℝ := width_cutout_inches / inches_in_foot

-- Calculate areas
def area_body : ℝ := length_body * width_body
def area_cutout : ℝ := length_cutout * width_cutout

-- Calculate the area of the L-shaped sheetrock
def area_L_shaped_sheetrock : ℝ := area_body - area_cutout

-- The target statement to prove
theorem area_of_L_shaped_sheetrock_is_28 : area_L_shaped_sheetrock = 28 := by
  sorry


end area_of_L_shaped_sheetrock_is_28_l566_566971


namespace first_digit_after_decimal_point_is_4_l566_566619

noncomputable def series_sum : Real :=
  ∑ k in (Finset.range 1005), (1 / ((4 * k + 1) * (4 * k + 2)) - 1 / ((4 * k + 3) * (4 * k + 4)))

theorem first_digit_after_decimal_point_is_4 :
  let first_digit := Real.floor (series_sum * 10) % 10
  in first_digit = 4 := by
  sorry

end first_digit_after_decimal_point_is_4_l566_566619
