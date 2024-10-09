import Mathlib

namespace rhombus_perimeter_l2409_240921

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l2409_240921


namespace fraction_halfway_between_l2409_240948

theorem fraction_halfway_between : 
  ∃ (x : ℚ), (x = (1 / 6 + 1 / 4) / 2) ∧ x = 5 / 24 :=
by
  sorry

end fraction_halfway_between_l2409_240948


namespace equal_roots_of_quadratic_l2409_240962

theorem equal_roots_of_quadratic (k : ℝ) : 
  (∃ x, (x^2 + 2 * x + k = 0) ∧ (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end equal_roots_of_quadratic_l2409_240962


namespace gwendolyn_reading_time_l2409_240918

/--
Gwendolyn can read 200 sentences in 1 hour. 
Each paragraph has 10 sentences. 
There are 20 paragraphs per page. 
The book has 50 pages. 
--/
theorem gwendolyn_reading_time : 
  let sentences_per_hour := 200
  let sentences_per_paragraph := 10
  let paragraphs_per_page := 20
  let pages := 50
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  (total_sentences / sentences_per_hour) = 50 := 
by
  let sentences_per_hour : ℕ := 200
  let sentences_per_paragraph : ℕ := 10
  let paragraphs_per_page : ℕ := 20
  let pages : ℕ := 50
  let sentences_per_page : ℕ := sentences_per_paragraph * paragraphs_per_page
  let total_sentences : ℕ := sentences_per_page * pages
  have h : (total_sentences / sentences_per_hour) = 50 := by sorry
  exact h

end gwendolyn_reading_time_l2409_240918


namespace isosceles_triangle_angles_l2409_240941

theorem isosceles_triangle_angles (α β γ : ℝ) 
  (h1 : α = 50)
  (h2 : α + β + γ = 180)
  (isosceles : (α = β ∨ α = γ ∨ β = γ)) :
  (β = 50 ∧ γ = 80) ∨ (γ = 50 ∧ β = 80) :=
by
  sorry

end isosceles_triangle_angles_l2409_240941


namespace eval_expression_l2409_240980

theorem eval_expression : -20 + 12 * ((5 + 15) / 4) = 40 :=
by
  sorry

end eval_expression_l2409_240980


namespace min_value_f_at_3_f_increasing_for_k_neg4_l2409_240928

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + k / (x - 1)

-- Problem (1): If k = 4, find the minimum value of f(x) and the corresponding value of x.
theorem min_value_f_at_3 : ∃ x > 1, @f x 4 = 5 ∧ x = 3 :=
  sorry

-- Problem (2): If k = -4, prove that f(x) is an increasing function for x > 1.
theorem f_increasing_for_k_neg4 : ∀ ⦃x y : ℝ⦄, 1 < x → x < y → f x (-4) < f y (-4) :=
  sorry

end min_value_f_at_3_f_increasing_for_k_neg4_l2409_240928


namespace Ben_sales_value_l2409_240950

noncomputable def value_of_sale (old_salary new_salary commission_ratio sales_required : ℝ) (diff_salary: ℝ) :=
  ∃ x : ℝ, 0.15 * x * sales_required = diff_salary ∧ x = 750

theorem Ben_sales_value (old_salary new_salary commission_ratio sales_required diff_salary: ℝ)
  (h1: old_salary = 75000)
  (h2: new_salary = 45000)
  (h3: commission_ratio = 0.15)
  (h4: sales_required = 266.67)
  (h5: diff_salary = old_salary - new_salary) :
  value_of_sale old_salary new_salary commission_ratio sales_required diff_salary :=
by
  sorry

end Ben_sales_value_l2409_240950


namespace count_integers_congruent_mod_l2409_240923

theorem count_integers_congruent_mod (n : ℕ) (h₁ : n < 1200) (h₂ : n ≡ 3 [MOD 7]) : 
  ∃ (m : ℕ), (m = 171) :=
by
  sorry

end count_integers_congruent_mod_l2409_240923


namespace intersection_M_N_l2409_240944

-- Define the universal set U, and subsets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- Prove that the intersection of M and N is as stated
theorem intersection_M_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 1} :=
by
  -- This is where the proof would go
  sorry

end intersection_M_N_l2409_240944


namespace cost_to_marked_price_ratio_l2409_240964

variables (p : ℝ) (discount : ℝ := 0.20) (cost_ratio : ℝ := 0.60)

theorem cost_to_marked_price_ratio :
  (cost_ratio * (1 - discount) * p) / p = 0.48 :=
by sorry

end cost_to_marked_price_ratio_l2409_240964


namespace algebra_or_drafting_not_both_l2409_240912

theorem algebra_or_drafting_not_both {A D : Finset ℕ} (h1 : (A ∩ D).card = 10) (h2 : A.card = 24) (h3 : D.card - (A ∩ D).card = 11) : (A ∪ D).card - (A ∩ D).card = 25 := by
  sorry

end algebra_or_drafting_not_both_l2409_240912


namespace principal_amount_l2409_240927

theorem principal_amount (P : ℝ) (h : (P * 0.1236) - (P * 0.12) = 36) : P = 10000 := 
sorry

end principal_amount_l2409_240927


namespace train_speed_is_45_km_per_hr_l2409_240967

/-- 
  Given the length of the train (135 m), the time to cross a bridge (30 s),
  and the length of the bridge (240 m), we want to prove that the speed of the 
  train is 45 km/hr.
--/

def length_of_train : ℕ := 135
def time_to_cross_bridge : ℕ := 30
def length_of_bridge : ℕ := 240
def speed_of_train_in_km_per_hr (L_t t L_b : ℕ) : ℕ := 
  ((L_t + L_b) * 36 / 10) / t

theorem train_speed_is_45_km_per_hr : 
  speed_of_train_in_km_per_hr length_of_train time_to_cross_bridge length_of_bridge = 45 :=
by 
  -- Assuming the calculations are correct, the expected speed is provided here directly
  sorry

end train_speed_is_45_km_per_hr_l2409_240967


namespace payment_proof_l2409_240934

theorem payment_proof (X Y : ℝ) 
  (h₁ : X + Y = 572) 
  (h₂ : X = 1.20 * Y) 
  : Y = 260 := 
by 
  sorry

end payment_proof_l2409_240934


namespace tax_percentage_l2409_240958

theorem tax_percentage (C T : ℝ) (h1 : C + 10 = 90) (h2 : 1 = 90 - C - T * 90) : T = 0.1 := 
by 
  -- We provide the conditions using sorry to indicate the steps would go here
  sorry

end tax_percentage_l2409_240958


namespace swans_count_l2409_240908

def numberOfSwans : Nat := 12

theorem swans_count (y : Nat) (x : Nat) (h1 : y = 5) (h2 : ∃ n m : Nat, x = 2 * n + 2 ∧ x = 3 * m - 3) : x = numberOfSwans := 
  by 
    sorry

end swans_count_l2409_240908


namespace wrongly_noted_mark_l2409_240919

theorem wrongly_noted_mark (n : ℕ) (avg_wrong avg_correct correct_mark : ℝ) (x : ℝ)
  (h1 : n = 30)
  (h2 : avg_wrong = 60)
  (h3 : avg_correct = 57.5)
  (h4 : correct_mark = 15)
  (h5 : n * avg_wrong - n * avg_correct = x - correct_mark)
  : x = 90 :=
sorry

end wrongly_noted_mark_l2409_240919


namespace surface_area_of_circumscribed_sphere_l2409_240973

/-- 
  Problem: Determine the surface area of the sphere circumscribed about a cube with edge length 2.

  Given:
  - The edge length of the cube is 2.
  - The space diagonal of a cube with edge length \(a\) is given by \(d = \sqrt{3} \cdot a\).
  - The diameter of the circumscribed sphere is equal to the space diagonal of the cube.
  - The surface area \(S\) of a sphere with radius \(R\) is given by \(S = 4\pi R^2\).

  To Prove:
  - The surface area of the sphere circumscribed about the cube is \(12\pi\).
-/
theorem surface_area_of_circumscribed_sphere (a : ℝ) (π : ℝ) (h1 : a = 2) 
  (h2 : ∀ a, d = Real.sqrt 3 * a) (h3 : ∀ d, R = d / 2) (h4 : ∀ R, S = 4 * π * R^2) : 
  S = 12 * π := 
by
  sorry

end surface_area_of_circumscribed_sphere_l2409_240973


namespace min_angle_B_l2409_240983

-- Definitions using conditions from part a)
def triangle (A B C : ℝ) : Prop := A + B + C = Real.pi
def arithmetic_sequence_prop (A B C : ℝ) : Prop := 
  Real.tan A + Real.tan C = 2 * (1 + Real.sqrt 2) * Real.tan B

-- Main theorem to prove
theorem min_angle_B (A B C : ℝ) (h1 : triangle A B C) (h2 : arithmetic_sequence_prop A B C) :
  B ≥ Real.pi / 4 :=
sorry

end min_angle_B_l2409_240983


namespace range_of_c_l2409_240987

theorem range_of_c (x y c : ℝ) (h1 : x^2 + (y - 2)^2 = 1) (h2 : x^2 + y^2 + c ≤ 0) : c ≤ -9 :=
by
  -- Proof goes here
  sorry

end range_of_c_l2409_240987


namespace height_of_parabolic_arch_l2409_240902

theorem height_of_parabolic_arch (a : ℝ) (x : ℝ) (k : ℝ) (h : ℝ) (s : ℝ) :
  k = 20 →
  s = 30 →
  a = - 4 / 45 →
  x = 3 →
  k = h →
  y = a * x^2 + k →
  h = 20 → 
  y = 19.2 :=
by
  -- Given the conditions, we'll prove using provided Lean constructs
  sorry

end height_of_parabolic_arch_l2409_240902


namespace find_x_l2409_240943

theorem find_x (x : ℝ) : 0.6 * x = (x / 3) + 110 → x = 412.5 := 
by
  intro h
  sorry

end find_x_l2409_240943


namespace solve_for_x_l2409_240900

theorem solve_for_x (x : ℝ) (h : (6 * x ^ 2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1) : x = -18 :=
sorry

end solve_for_x_l2409_240900


namespace find_multiplier_l2409_240931

-- Define the variables x and y
variables (x y : ℕ)

-- Define the conditions
def condition1 := (x / 6) * y = 12
def condition2 := x = 6

-- State the theorem to prove
theorem find_multiplier (h1 : condition1 x y) (h2 : condition2 x) : y = 12 :=
sorry

end find_multiplier_l2409_240931


namespace sum_a_b_eq_neg2_l2409_240924

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : f a + f b = 20) : a + b = -2 :=
by
  sorry

end sum_a_b_eq_neg2_l2409_240924


namespace distance_from_P_to_focus_l2409_240960

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

end distance_from_P_to_focus_l2409_240960


namespace farmer_planting_problem_l2409_240949

theorem farmer_planting_problem (total_acres : ℕ) (flax_acres : ℕ) (sunflower_acres : ℕ)
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : sunflower_acres = total_acres - flax_acres) :
  sunflower_acres - flax_acres = 80 := by
  sorry

end farmer_planting_problem_l2409_240949


namespace composite_sum_l2409_240993

open Nat

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) : ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + c = x * y :=
by
  sorry

end composite_sum_l2409_240993


namespace alice_journey_duration_l2409_240933
noncomputable def journey_duration (start_hour start_minute end_hour end_minute : ℕ) : ℕ :=
  let start_in_minutes := start_hour * 60 + start_minute
  let end_in_minutes := end_hour * 60 + end_minute
  if end_in_minutes >= start_in_minutes then end_in_minutes - start_in_minutes
  else end_in_minutes + 24 * 60 - start_in_minutes
  
theorem alice_journey_duration :
  ∃ start_hour start_minute end_hour end_minute,
  (7 ≤ start_hour ∧ start_hour < 8 ∧ start_minute = 38) ∧
  (16 ≤ end_hour ∧ end_hour < 17 ∧ end_minute = 35) ∧
  journey_duration start_hour start_minute end_hour end_minute = 537 :=
by {
  sorry
}

end alice_journey_duration_l2409_240933


namespace percent_increase_sales_l2409_240970

theorem percent_increase_sales (sales_this_year sales_last_year : ℝ) (h1 : sales_this_year = 460) (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 :=
by
  sorry

end percent_increase_sales_l2409_240970


namespace MoneyDivision_l2409_240920

theorem MoneyDivision (w x y z : ℝ)
  (hw : y = 0.5 * w)
  (hx : x = 0.7 * w)
  (hz : z = 0.3 * w)
  (hy : y = 90) :
  w + x + y + z = 450 := by
  sorry

end MoneyDivision_l2409_240920


namespace PropositionA_necessary_not_sufficient_l2409_240946

variable (a : ℝ)

def PropositionA : Prop := a < 2
def PropositionB : Prop := a^2 < 4

theorem PropositionA_necessary_not_sufficient : 
  (PropositionA a → PropositionB a) ∧ ¬ (PropositionB a → PropositionA a) :=
sorry

end PropositionA_necessary_not_sufficient_l2409_240946


namespace scouts_attended_l2409_240974

def chocolate_bar_cost : ℝ := 1.50
def total_spent : ℝ := 15
def sections_per_bar : ℕ := 3
def smores_per_scout : ℕ := 2

theorem scouts_attended (bars : ℝ) (sections : ℕ) (smores : ℕ) (scouts : ℕ) :
  bars = total_spent / chocolate_bar_cost →
  sections = bars * sections_per_bar →
  smores = sections →
  scouts = smores / smores_per_scout →
  scouts = 15 :=
by
  intro h1 h2 h3 h4
  sorry

end scouts_attended_l2409_240974


namespace incorrect_proposition3_l2409_240911

open Real

-- Definitions from the problem
def prop1 (x : ℝ) := 2 * sin (2 * x - π / 3) = 2
def prop2 (x y : ℝ) := tan x + tan (π - x) = 0
def prop3 (x1 x2 : ℝ) (k : ℤ) := x1 - x2 = (k : ℝ) * π → k % 2 = 1
def prop4 (x : ℝ) := cos x ^ 2 + sin x >= -1

-- Incorrect proposition proof
theorem incorrect_proposition3 (x1 x2 : ℝ) (k : ℤ) :
  sin (2 * x1 - π / 4) = 0 →
  sin (2 * x2 - π / 4) = 0 →
  x1 - x2 ≠ (k : ℝ) * π := sorry

end incorrect_proposition3_l2409_240911


namespace annual_growth_rate_l2409_240995

-- definitions based on the conditions in the problem
def FirstYear : ℝ := 400
def ThirdYear : ℝ := 625
def n : ℕ := 2

-- the main statement to prove the corresponding equation
theorem annual_growth_rate (x : ℝ) : 400 * (1 + x)^2 = 625 :=
sorry

end annual_growth_rate_l2409_240995


namespace total_participants_l2409_240982

theorem total_participants (x : ℕ) (h1 : 800 / x + 60 = 800 / (x - 3)) : x = 8 :=
sorry

end total_participants_l2409_240982


namespace sum_mod_9_l2409_240930

theorem sum_mod_9 :
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := 
by sorry

end sum_mod_9_l2409_240930


namespace domain_of_h_l2409_240979

-- Definition of the function domain of f(x) and h(x)
def f_domain := Set.Icc (-10: ℝ) 6
def h_domain := Set.Icc (-2: ℝ) (10/3)

-- Definition of f and h
def f (x: ℝ) : ℝ := sorry  -- f is assumed to be defined on the interval [-10, 6]
def h (x: ℝ) : ℝ := f (-3 * x)

-- Theorem statement: Given the domain of f(x), the domain of h(x) is as follows
theorem domain_of_h :
  (∀ x, x ∈ f_domain ↔ (-3 * x) ∈ h_domain) :=
sorry

end domain_of_h_l2409_240979


namespace systematic_sampling_l2409_240942

theorem systematic_sampling (N : ℕ) (k : ℕ) (interval : ℕ) (seq : List ℕ) : 
  N = 70 → k = 7 → interval = 10 → 
  seq = [3, 13, 23, 33, 43, 53, 63] := 
by 
  intros hN hk hInt;
  sorry

end systematic_sampling_l2409_240942


namespace ratio_of_share_l2409_240939

/-- A certain amount of money is divided amongst a, b, and c. 
The share of a is $122, and the total amount of money is $366. 
Prove that the ratio of a's share to the combined share of b and c is 1 / 2. -/
theorem ratio_of_share (a b c : ℝ) (total share_a : ℝ) (h1 : a + b + c = total) 
  (h2 : total = 366) (h3 : share_a = 122) : share_a / (total - share_a) = 1 / 2 := by
  sorry

end ratio_of_share_l2409_240939


namespace roger_left_money_correct_l2409_240922

noncomputable def roger_left_money (P : ℝ) (q : ℝ) (E : ℝ) (r1 : ℝ) (C : ℝ) (r2 : ℝ) : ℝ :=
  let feb_expense := q * P
  let after_feb := P - feb_expense
  let mar_expense := E * r1
  let after_mar := after_feb - mar_expense
  let mom_gift := C * r2
  after_mar + mom_gift

theorem roger_left_money_correct :
  roger_left_money 45 0.35 20 1.2 46 0.8 = 42.05 :=
by
  sorry

end roger_left_money_correct_l2409_240922


namespace sum_of_digits_l2409_240953

variable (a b c d e f : ℕ)

theorem sum_of_digits :
  ∀ (a b c d e f : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 →
    a + b + c + d + e + f = 28 := 
by
  intros a b c d e f h
  sorry

end sum_of_digits_l2409_240953


namespace nancy_balloons_l2409_240925

variable (MaryBalloons : ℝ) (NancyBalloons : ℝ)

theorem nancy_balloons (h1 : NancyBalloons = 4 * MaryBalloons) (h2 : MaryBalloons = 1.75) : 
  NancyBalloons = 7 := 
by 
  sorry

end nancy_balloons_l2409_240925


namespace bear_cubs_count_l2409_240975

theorem bear_cubs_count (total_meat : ℕ) (meat_per_cub : ℕ) (rabbits_per_day : ℕ) (weeks_days : ℕ) (meat_per_rabbit : ℕ)
  (mother_total_meat : ℕ) (number_of_cubs : ℕ) : 
  total_meat = 210 →
  meat_per_cub = 35 →
  rabbits_per_day = 10 →
  weeks_days = 7 →
  meat_per_rabbit = 5 →
  mother_total_meat = rabbits_per_day * weeks_days * meat_per_rabbit →
  meat_per_cub * number_of_cubs + mother_total_meat = total_meat →
  number_of_cubs = 4 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end bear_cubs_count_l2409_240975


namespace geometric_seq_ad_eq_2_l2409_240907

open Real

def geometric_sequence (a b c d : ℝ) : Prop :=
∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r 

def is_max_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
f x = y ∧ ∀ z : ℝ, z ≠ x → f x ≥ f z

theorem geometric_seq_ad_eq_2 (a b c d : ℝ) :
  geometric_sequence a b c d →
  is_max_point (λ x => 3 * x - x ^ 3) b c →
  a * d = 2 :=
by
  sorry

end geometric_seq_ad_eq_2_l2409_240907


namespace simplify_expression_l2409_240937

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x ^ 2 + 10 - (5 - 4 * x + 8 * x ^ 2) = -16 * x ^ 2 + 8 * x + 5 :=
by
  sorry

end simplify_expression_l2409_240937


namespace tan_315_degrees_l2409_240938

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l2409_240938


namespace rabbit_time_2_miles_l2409_240909

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end rabbit_time_2_miles_l2409_240909


namespace jogs_per_day_l2409_240999

-- Definitions of conditions
def weekdays_per_week : ℕ := 5
def total_weeks : ℕ := 3
def total_miles : ℕ := 75

-- Define the number of weekdays in total weeks
def total_weekdays : ℕ := total_weeks * weekdays_per_week

-- Theorem to prove Damien jogs 5 miles per day on weekdays
theorem jogs_per_day : total_miles / total_weekdays = 5 := by
  sorry

end jogs_per_day_l2409_240999


namespace tan_half_sum_l2409_240992

theorem tan_half_sum (p q : ℝ)
  (h1 : Real.cos p + Real.cos q = (1:ℝ)/3)
  (h2 : Real.sin p + Real.sin q = (8:ℝ)/17) :
  Real.tan ((p + q) / 2) = (24:ℝ)/17 := 
sorry

end tan_half_sum_l2409_240992


namespace conner_ties_sydney_l2409_240991

def sydney_initial_collect := 837
def conner_initial_collect := 723

def sydney_collect_day_one := 4
def conner_collect_day_one := 8 * sydney_collect_day_one / 2

def sydney_collect_day_two := (sydney_initial_collect + sydney_collect_day_one) - ((sydney_initial_collect + sydney_collect_day_one) / 10)
def conner_collect_day_two := conner_initial_collect + conner_collect_day_one + 123

def sydney_collect_day_three := sydney_collect_day_two + 2 * conner_collect_day_one
def conner_collect_day_three := (conner_collect_day_two - (123 / 4))

theorem conner_ties_sydney :
  sydney_collect_day_three <= conner_collect_day_three :=
by
  sorry

end conner_ties_sydney_l2409_240991


namespace find_x_condition_l2409_240954

theorem find_x_condition (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := 
by
  sorry

end find_x_condition_l2409_240954


namespace quadratic_solution_eq_l2409_240968

theorem quadratic_solution_eq (c d : ℝ) 
  (h_eq : ∀ x : ℝ, x^2 - 6*x + 11 = 25 ↔ (x = c ∨ x = d))
  (h_order : c ≥ d) :
  c + 2*d = 9 - Real.sqrt 23 :=
sorry

end quadratic_solution_eq_l2409_240968


namespace GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l2409_240959

noncomputable def GCD (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCD_17_51 : GCD 17 51 = 17 := by
  sorry

theorem LCM_17_51 : LCM 17 51 = 51 := by
  sorry

theorem GCD_6_8 : GCD 6 8 = 2 := by
  sorry

theorem LCM_8_9 : LCM 8 9 = 72 := by
  sorry

end GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l2409_240959


namespace xy_value_l2409_240994

variable (x y : ℕ)

def condition1 : Prop := 8^x / 4^(x + y) = 16
def condition2 : Prop := 16^(x + y) / 4^(7 * y) = 256

theorem xy_value (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 48 := by
  sorry

end xy_value_l2409_240994


namespace andrei_stamps_l2409_240986

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l2409_240986


namespace min_fraction_l2409_240910

theorem min_fraction (x A C : ℝ) (hx : x > 0) (hA : A = x^2 + 1/x^2) (hC : C = x + 1/x) :
  ∃ m, m = 2 * Real.sqrt 2 ∧ ∀ B, B > 0 → x^2 + 1/x^2 = B → x + 1/x = C → B / C ≥ m :=
by
  sorry

end min_fraction_l2409_240910


namespace find_2a_plus_b_l2409_240945

open Real

theorem find_2a_plus_b (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
    (h1 : 4 * (cos a)^3 - 3 * (cos b)^3 = 2) 
    (h2 : 4 * cos (2 * a) + 3 * cos (2 * b) = 1) : 
    2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l2409_240945


namespace interest_difference_l2409_240901

noncomputable def principal := 63100
noncomputable def rate := 10 / 100
noncomputable def time := 2

noncomputable def simple_interest := principal * rate * time
noncomputable def compound_interest := principal * (1 + rate)^time - principal

theorem interest_difference :
  (compound_interest - simple_interest) = 671 := by
  sorry

end interest_difference_l2409_240901


namespace total_amount_shared_l2409_240972

-- Define the initial conditions
def ratioJohn : ℕ := 2
def ratioJose : ℕ := 4
def ratioBinoy : ℕ := 6
def JohnShare : ℕ := 2000
def partValue : ℕ := JohnShare / ratioJohn

-- Define the shares based on the ratio and part value
def JoseShare := ratioJose * partValue
def BinoyShare := ratioBinoy * partValue

-- Prove the total amount shared is Rs. 12000
theorem total_amount_shared : (JohnShare + JoseShare + BinoyShare) = 12000 :=
  by
  sorry

end total_amount_shared_l2409_240972


namespace bicyclist_speed_remainder_l2409_240940

theorem bicyclist_speed_remainder (total_distance first_distance remainder_distance first_speed avg_speed remainder_speed time_total time_first time_remainder : ℝ) 
  (H1 : total_distance = 350)
  (H2 : first_distance = 200)
  (H3 : remainder_distance = total_distance - first_distance)
  (H4 : first_speed = 20)
  (H5 : avg_speed = 17.5)
  (H6 : time_total = total_distance / avg_speed)
  (H7 : time_first = first_distance / first_speed)
  (H8 : time_remainder = time_total - time_first)
  (H9 : remainder_speed = remainder_distance / time_remainder) :
  remainder_speed = 15 := 
sorry

end bicyclist_speed_remainder_l2409_240940


namespace other_root_of_quadratic_l2409_240904

theorem other_root_of_quadratic 
  (a b c: ℝ) 
  (h : a * (b - c - d) * (1:ℝ)^2 + b * (c - a + d) * (1:ℝ) + c * (a - b - d) = 0) : 
  ∃ k: ℝ, k = c * (a - b - d) / (a * (b - c - d)) :=
sorry

end other_root_of_quadratic_l2409_240904


namespace carlton_outfits_l2409_240903

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l2409_240903


namespace jessica_coins_worth_l2409_240977

theorem jessica_coins_worth :
  ∃ (n d : ℕ), n + d = 30 ∧ 5 * (30 - d) + 10 * d = 165 :=
by {
  sorry
}

end jessica_coins_worth_l2409_240977


namespace right_triangle_congruence_l2409_240966

theorem right_triangle_congruence (A B C D : Prop) :
  (A → true) → (C → true) → (D → true) → (¬ B) → B :=
by
sorry

end right_triangle_congruence_l2409_240966


namespace find_volume_of_pyramid_l2409_240926

noncomputable def volume_of_pyramid
  (a : ℝ) (α : ℝ)
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) : ℝ :=
  (a ^ 3 * abs (Real.cos α)) / 3

--and the theorem to prove the statement
theorem find_volume_of_pyramid
  (a α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) :
  volume_of_pyramid a α h1 h2 h3 = (a ^ 3 * abs (Real.cos α)) / 3 :=
sorry

end find_volume_of_pyramid_l2409_240926


namespace find_f_l2409_240965

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f (x - 1/x) = x^2 + 1/x^2 - 4) :
  ∀ x : ℝ, f x = x^2 - 2 :=
by
  intros x
  sorry

end find_f_l2409_240965


namespace second_hand_distance_l2409_240917

theorem second_hand_distance (r : ℝ) (t : ℝ) (π : ℝ) (hand_length_6cm : r = 6) (time_15_min : t = 15) : 
  ∃ d : ℝ, d = 180 * π :=
by
  sorry

end second_hand_distance_l2409_240917


namespace find_C_l2409_240996

theorem find_C
  (A B C D : ℕ)
  (h1 : 0 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 4 * 1000 + A * 100 + 5 * 10 + B + (C * 1000 + 2 * 100 + D * 10 + 7) = 8070) :
  C = 3 :=
by
  sorry

end find_C_l2409_240996


namespace line_intersects_x_axis_at_10_0_l2409_240969

theorem line_intersects_x_axis_at_10_0 :
  let x1 := 9
  let y1 := 1
  let x2 := 5
  let y2 := 5
  let slope := (y2 - y1) / (x2 - x1)
  let y := 0
  ∃ x, (x - x1) * slope = y - y1 ∧ y = 0 → x = 10 := by
  sorry

end line_intersects_x_axis_at_10_0_l2409_240969


namespace inequality_not_always_hold_l2409_240978

variable (a b c : ℝ)

theorem inequality_not_always_hold (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) : ¬ (∀ (a b : ℝ), |a - b| + 1 / (a - b) ≥ 2) :=
by
  sorry

end inequality_not_always_hold_l2409_240978


namespace sum_of_coordinates_point_D_l2409_240988

theorem sum_of_coordinates_point_D 
(M : ℝ × ℝ) (C D : ℝ × ℝ) 
(hM : M = (3, 5)) 
(hC : C = (1, 10)) 
(hmid : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
: D.1 + D.2 = 5 :=
sorry

end sum_of_coordinates_point_D_l2409_240988


namespace Mr_Deane_filled_today_l2409_240957

theorem Mr_Deane_filled_today :
  ∀ (x : ℝ),
    (25 * (1.4 - 0.4) + 1.4 * x = 39) →
    x = 10 :=
by
  intros x h
  sorry

end Mr_Deane_filled_today_l2409_240957


namespace gcd_of_lcm_l2409_240936

noncomputable def gcd (A B C : ℕ) : ℕ := Nat.gcd (Nat.gcd A B) C
noncomputable def lcm (A B C : ℕ) : ℕ := Nat.lcm (Nat.lcm A B) C

theorem gcd_of_lcm (A B C : ℕ) (LCM_ABC : ℕ) (Product_ABC : ℕ) :
  lcm A B C = LCM_ABC →
  A * B * C = Product_ABC →
  gcd A B C = 20 :=
by
  intros lcm_eq product_eq
  sorry

end gcd_of_lcm_l2409_240936


namespace projection_cardinal_inequality_l2409_240981

variables {Point : Type} [Fintype Point] [DecidableEq Point]

def projection_Oyz (S : Finset Point) : Finset Point := sorry
def projection_Ozx (S : Finset Point) : Finset Point := sorry
def projection_Oxy (S : Finset Point) : Finset Point := sorry

theorem projection_cardinal_inequality
  (S : Finset Point)
  (S_x := projection_Oyz S)
  (S_y := projection_Ozx S)
  (S_z := projection_Oxy S)
  : (Finset.card S)^2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) :=
sorry

end projection_cardinal_inequality_l2409_240981


namespace find_divisor_l2409_240971

theorem find_divisor (d : ℕ) : 15 = (d * 4) + 3 → d = 3 := by
  intros h
  have h1 : 15 - 3 = 4 * d := by
    linarith
  have h2 : 12 = 4 * d := by
    linarith
  have h3 : d = 3 := by
    linarith
  exact h3

end find_divisor_l2409_240971


namespace factor_expression_l2409_240947

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l2409_240947


namespace integer_for_finitely_many_n_l2409_240956

theorem integer_for_finitely_many_n (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, N < n → ¬ ∃ k : ℤ, (a + 1 / 2) ^ n + (b + 1 / 2) ^ n = k := 
sorry

end integer_for_finitely_many_n_l2409_240956


namespace bonus_tasks_l2409_240951

-- Definition for earnings without bonus
def earnings_without_bonus (tasks : ℕ) : ℕ := tasks * 2

-- Definition for calculating the total bonus received
def total_bonus (tasks : ℕ) (earnings : ℕ) : ℕ := earnings - earnings_without_bonus tasks

-- Definition for the number of bonuses received given the total bonus and a single bonus amount
def number_of_bonuses (total_bonus : ℕ) (bonus_amount : ℕ) : ℕ := total_bonus / bonus_amount

-- The theorem we want to prove
theorem bonus_tasks (tasks : ℕ) (earnings : ℕ) (bonus_amount : ℕ) (bonus_tasks : ℕ) :
  earnings = 78 →
  tasks = 30 →
  bonus_amount = 6 →
  bonus_tasks = tasks / (number_of_bonuses (total_bonus tasks earnings) bonus_amount) →
  bonus_tasks = 10 :=
by
  intros h_earnings h_tasks h_bonus_amount h_bonus_tasks
  sorry

end bonus_tasks_l2409_240951


namespace max_n_perfect_cube_l2409_240913

-- Definition for sum of squares
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Definition for sum of squares from (n+1) to 2n
def sum_of_squares_segment (n : ℕ) : ℕ :=
  2 * n * (2 * n + 1) * (4 * n + 1) / 6 - n * (n + 1) * (2 * n + 1) / 6

-- Definition for the product of the sums
def product_of_sums (n : ℕ) : ℕ :=
  (sum_of_squares n) * (sum_of_squares_segment n)

-- Predicate for perfect cube
def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y ^ 3 = x

-- The main theorem to be proved
theorem max_n_perfect_cube : ∃ (n : ℕ), n ≤ 2050 ∧ is_perfect_cube (product_of_sums n) ∧ ∀ m : ℕ, (m ≤ 2050 ∧ is_perfect_cube (product_of_sums m)) → m ≤ 2016 := 
sorry

end max_n_perfect_cube_l2409_240913


namespace solve_equation_1_solve_equation_2_l2409_240929

theorem solve_equation_1 (x : ℝ) (h₁ : x - 4 = -5) : x = -1 :=
sorry

theorem solve_equation_2 (x : ℝ) (h₂ : (1/2) * x + 2 = 6) : x = 8 :=
sorry

end solve_equation_1_solve_equation_2_l2409_240929


namespace prime_gt_five_condition_l2409_240914

theorem prime_gt_five_condition (p : ℕ) [Fact (Nat.Prime p)] (h : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - a^2) ∣ (p - b)^2 := 
sorry

end prime_gt_five_condition_l2409_240914


namespace library_pupils_count_l2409_240985

-- Definitions for the conditions provided in the problem
def num_rectangular_tables : Nat := 7
def num_pupils_per_rectangular_table : Nat := 10
def num_square_tables : Nat := 5
def num_pupils_per_square_table : Nat := 4

-- Theorem stating the problem's question and the required proof
theorem library_pupils_count :
  num_rectangular_tables * num_pupils_per_rectangular_table + 
  num_square_tables * num_pupils_per_square_table = 90 :=
sorry

end library_pupils_count_l2409_240985


namespace min_value_x_add_y_div_2_l2409_240990

theorem min_value_x_add_y_div_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 2 * x - y = 0) :
  ∃ x y, 0 < x ∧ 0 < y ∧ (x * y - 2 * x - y = 0 ∧ x + y / 2 = 4) :=
sorry

end min_value_x_add_y_div_2_l2409_240990


namespace point_M_first_quadrant_distances_length_of_segment_MN_l2409_240984

-- Proof problem 1
theorem point_M_first_quadrant_distances (m : ℝ) (h1 : 2 * m + 1 > 0) (h2 : m + 3 > 0) (h3 : m + 3 = 2 * (2 * m + 1)) :
  m = 1 / 3 :=
by
  sorry

-- Proof problem 2
theorem length_of_segment_MN (m : ℝ) (h4 : m + 3 = 1) :
  let Mx := 2 * m + 1
  let My := m + 3
  let Nx := 2
  let Ny := 1
  let distMN := abs (Nx - Mx)
  distMN = 5 :=
by
  sorry

end point_M_first_quadrant_distances_length_of_segment_MN_l2409_240984


namespace smallest_four_digit_divisible_by_35_l2409_240997

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l2409_240997


namespace values_of_a_l2409_240915

theorem values_of_a (a : ℝ) : 
  ∃a1 a2 : ℝ, 
  (∀ x y : ℝ, (y = 3 * x + a) ∧ (y = x^3 + 3 * a^2) → (x = 0) → (y = 3 * a^2)) →
  ((a = 0) ∨ (a = 1/3)) ∧ 
  ((a1 = 0) ∨ (a1 = 1/3)) ∧
  ((a2 = 0) ∨ (a2 = 1/3)) ∧ 
  (a ≠ a1 ∨ a ≠ a2) ∧ 
  (∃ n : ℤ, n = 2) :=
by sorry

end values_of_a_l2409_240915


namespace triplet_unique_solution_l2409_240976

theorem triplet_unique_solution {x y z : ℝ} :
  x^2 - 2*x - 4*z = 3 →
  y^2 - 2*y - 2*x = -14 →
  z^2 - 4*y - 4*z = -18 →
  (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry

end triplet_unique_solution_l2409_240976


namespace min_value_of_squares_find_p_l2409_240932

open Real

theorem min_value_of_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (eqn : a + sqrt 2 * b + sqrt 3 * c = 2 * sqrt 3) :
  a^2 + b^2 + c^2 = 2 :=
by sorry

theorem find_p (m : ℝ) (hm : m = 2) (p q : ℝ) :
  (∀ x, |x - 3| ≥ m ↔ x^2 + p * x + q ≥ 0) → p = -6 :=
by sorry

end min_value_of_squares_find_p_l2409_240932


namespace booksReadPerDay_l2409_240963

-- Mrs. Hilt read 14 books in a week.
def totalBooksReadInWeek : ℕ := 14

-- There are 7 days in a week.
def daysInWeek : ℕ := 7

-- We need to prove that the number of books read per day is 2.
theorem booksReadPerDay :
  totalBooksReadInWeek / daysInWeek = 2 :=
by
  sorry

end booksReadPerDay_l2409_240963


namespace angle_B_in_triangle_l2409_240905

/-- In triangle ABC, if BC = √3, AC = √2, and ∠A = π/3,
then ∠B = π/4. -/
theorem angle_B_in_triangle
  (BC AC : ℝ) (A B : ℝ)
  (hBC : BC = Real.sqrt 3)
  (hAC : AC = Real.sqrt 2)
  (hA : A = Real.pi / 3) :
  B = Real.pi / 4 :=
sorry

end angle_B_in_triangle_l2409_240905


namespace simplify_fraction_l2409_240955

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h_cond : y^3 - 1/x ≠ 0) :
  (x^3 - 1/y) / (y^3 - 1/x) = x / y :=
by
  sorry

end simplify_fraction_l2409_240955


namespace find_constants_a_b_l2409_240935

variables (x a b : ℝ)

theorem find_constants_a_b (h : (x - a) / (x + b) = (x^2 - 45 * x + 504) / (x^2 + 66 * x - 1080)) :
  a + b = 48 :=
sorry

end find_constants_a_b_l2409_240935


namespace operation_result_l2409_240952

-- Define the operation
def operation (a b : ℝ) : ℝ := (a - b) ^ 3

theorem operation_result (x y : ℝ) : operation ((x - y) ^ 3) ((y - x) ^ 3) = -8 * (y - x) ^ 9 := 
  sorry

end operation_result_l2409_240952


namespace largest_of_numbers_l2409_240916

theorem largest_of_numbers (a b c d : ℝ) (hₐ : a = 0) (h_b : b = -1) (h_c : c = -2) (h_d : d = Real.sqrt 3) :
  d = Real.sqrt 3 ∧ d > a ∧ d > b ∧ d > c :=
by
  -- Using sorry to skip the proof
  sorry

end largest_of_numbers_l2409_240916


namespace katya_solves_enough_l2409_240998

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l2409_240998


namespace fixed_point_exists_l2409_240906

theorem fixed_point_exists (m : ℝ) :
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 → x = 3 ∧ y = 1 :=
by
  sorry

end fixed_point_exists_l2409_240906


namespace sum_adjacent_to_49_l2409_240961

noncomputable def sum_of_adjacent_divisors : ℕ :=
  let divisors := [5, 7, 35, 49, 245]
  -- We assume an arrangement such that adjacent pairs to 49 are {35, 245}
  35 + 245

theorem sum_adjacent_to_49 : sum_of_adjacent_divisors = 280 := by
  sorry

end sum_adjacent_to_49_l2409_240961


namespace set_A_main_inequality_l2409_240989

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 2|
def A : Set ℝ := {x | f x < 3}

theorem set_A :
  A = {x | -2 / 3 < x ∧ x < 0} :=
sorry

theorem main_inequality (s t : ℝ) (hs : -2 / 3 < s ∧ s < 0) (ht : -2 / 3 < t ∧ t < 0) :
  |1 - t / s| < |t - 1 / s| :=
sorry

end set_A_main_inequality_l2409_240989
