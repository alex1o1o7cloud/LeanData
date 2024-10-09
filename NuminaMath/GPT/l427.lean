import Mathlib

namespace compare_combined_sums_l427_42783

def numeral1 := 7524258
def numeral2 := 523625072

def place_value_2_numeral1 := 200000 + 20
def place_value_5_numeral1 := 50000 + 500
def combined_sum_numeral1 := place_value_2_numeral1 + place_value_5_numeral1

def place_value_2_numeral2 := 200000000 + 20
def place_value_5_numeral2 := 500000 + 50
def combined_sum_numeral2 := place_value_2_numeral2 + place_value_5_numeral2

def difference := combined_sum_numeral2 - combined_sum_numeral1

theorem compare_combined_sums :
  difference = 200249550 := by
  sorry

end compare_combined_sums_l427_42783


namespace tank_capacity_l427_42702

theorem tank_capacity (x : ℝ) (h₁ : (3/4) * x = (1/3) * x + 18) : x = 43.2 := sorry

end tank_capacity_l427_42702


namespace four_consecutive_even_impossible_l427_42729

def is_four_consecutive_even_sum (S : ℕ) : Prop :=
  ∃ n : ℤ, S = 4 * n + 12

theorem four_consecutive_even_impossible :
  ¬ is_four_consecutive_even_sum 34 :=
by
  sorry

end four_consecutive_even_impossible_l427_42729


namespace count_semiprimes_expressed_as_x_cubed_minus_1_l427_42739

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_semiprime (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n

theorem count_semiprimes_expressed_as_x_cubed_minus_1 :
  (∃ S : Finset ℕ, 
    S.card = 4 ∧ 
    ∀ n ∈ S, n < 2018 ∧ 
    ∃ x : ℕ, x > 0 ∧ x^3 - 1 = n ∧ is_semiprime n) :=
sorry

end count_semiprimes_expressed_as_x_cubed_minus_1_l427_42739


namespace cucumber_new_weight_l427_42740

-- Definitions for the problem conditions
def initial_weight : ℝ := 100
def initial_water_percentage : ℝ := 0.99
def final_water_percentage : ℝ := 0.96
noncomputable def new_weight : ℝ := initial_weight * (1 - initial_water_percentage) / (1 - final_water_percentage)

-- The theorem stating the problem to be solved
theorem cucumber_new_weight : new_weight = 25 :=
by
  -- Skipping the proof for now
  sorry

end cucumber_new_weight_l427_42740


namespace area_of_square_l427_42741

theorem area_of_square (d : ℝ) (hd : d = 14 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 196 := by
  sorry

end area_of_square_l427_42741


namespace sin_func_even_min_period_2pi_l427_42753

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem sin_func_even_min_period_2pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end sin_func_even_min_period_2pi_l427_42753


namespace sufficient_not_necessary_l427_42772

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1 → x^2 - 2*x + 1 > 0) ∧ (¬(x^2 - 2*x + 1 > 0 → x > 1)) := by
  sorry

end sufficient_not_necessary_l427_42772


namespace fly_distance_to_ceiling_l427_42782

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ), 
  (x = 3) → 
  (y = 4) → 
  (z * z + 25 = 49) →
  z = 2 * Real.sqrt 6 :=
by
  sorry

end fly_distance_to_ceiling_l427_42782


namespace lamps_purchased_min_type_B_lamps_l427_42755

variables (x y m : ℕ)

def total_lamps := x + y = 50
def total_cost := 40 * x + 65 * y = 2500
def profit_type_A := 60 - 40 = 20
def profit_type_B := 100 - 65 = 35
def profit_requirement := 20 * (50 - m) + 35 * m ≥ 1400

theorem lamps_purchased (h₁ : total_lamps x y) (h₂ : total_cost x y) : 
  x = 30 ∧ y = 20 :=
  sorry

theorem min_type_B_lamps (h₃ : profit_type_A) (h₄ : profit_type_B) (h₅ : profit_requirement m) : 
  m ≥ 27 :=
  sorry

end lamps_purchased_min_type_B_lamps_l427_42755


namespace conclusion_l427_42735

-- Assuming U is the universal set and Predicates represent Mems, Ens, and Veens
variable (U : Type)
variable (Mem : U → Prop)
variable (En : U → Prop)
variable (Veen : U → Prop)

-- Hypotheses
variable (h1 : ∀ x, Mem x → En x)          -- Hypothesis I: All Mems are Ens
variable (h2 : ∀ x, En x → ¬Veen x)        -- Hypothesis II: No Ens are Veens

-- To be proven
theorem conclusion (x : U) : (Mem x → ¬Veen x) ∧ (Mem x → ¬Veen x) := sorry

end conclusion_l427_42735


namespace find_numbers_satisfying_conditions_l427_42743

theorem find_numbers_satisfying_conditions (x y z : ℝ)
(h1 : x + y + z = 11 / 18)
(h2 : 1 / x + 1 / y + 1 / z = 18)
(h3 : 2 / y = 1 / x + 1 / z) :
x = 1 / 9 ∧ y = 1 / 6 ∧ z = 1 / 3 :=
sorry

end find_numbers_satisfying_conditions_l427_42743


namespace fraction_of_b_eq_two_thirds_l427_42799

theorem fraction_of_b_eq_two_thirds (A B : ℝ) (x : ℝ) (h1 : A + B = 1210) (h2 : B = 484)
  (h3 : (2/3) * A = x * B) : x = 2/3 :=
by
  sorry

end fraction_of_b_eq_two_thirds_l427_42799


namespace Miles_trombones_count_l427_42701

theorem Miles_trombones_count :
  let fingers := 10
  let trumpets := fingers - 3
  let hands := 2
  let guitars := hands + 2
  let french_horns := guitars - 1
  let heads := 1
  let trombones := heads + 2
  trumpets + guitars + french_horns + trombones = 17 → trombones = 3 :=
by
  intros h
  sorry

end Miles_trombones_count_l427_42701


namespace sequence_strictly_increasing_from_14_l427_42789

def a (n : ℕ) : ℤ := n^4 - 20 * n^2 - 10 * n + 1

theorem sequence_strictly_increasing_from_14 :
  ∀ n : ℕ, n ≥ 14 → a (n + 1) > a n :=
by
  sorry

end sequence_strictly_increasing_from_14_l427_42789


namespace emily_saves_more_using_promotion_a_l427_42721

-- Definitions based on conditions
def price_per_pair : ℕ := 50
def promotion_a_cost : ℕ := price_per_pair + price_per_pair / 2
def promotion_b_cost : ℕ := price_per_pair + (price_per_pair - 20)

-- Statement to prove the savings
theorem emily_saves_more_using_promotion_a :
  promotion_b_cost - promotion_a_cost = 5 := by
  sorry

end emily_saves_more_using_promotion_a_l427_42721


namespace flight_duration_l427_42730

theorem flight_duration (h m : ℕ) (Hh : h = 2) (Hm : m = 32) : h + m = 34 := by
  sorry

end flight_duration_l427_42730


namespace probability_math_majors_consecutive_l427_42752

noncomputable def total_ways := Nat.choose 11 4 -- Number of ways to choose 5 persons out of 12 (fixing one)
noncomputable def favorable_ways := 12         -- Number of ways to arrange 5 math majors consecutively around a round table

theorem probability_math_majors_consecutive :
  (favorable_ways : ℚ) / total_ways = 2 / 55 :=
by
  sorry

end probability_math_majors_consecutive_l427_42752


namespace shares_difference_l427_42751

noncomputable def Faruk_share (V : ℕ) : ℕ := (3 * (V / 5))
noncomputable def Ranjith_share (V : ℕ) : ℕ := (7 * (V / 5))

theorem shares_difference {V : ℕ} (hV : V = 1500) : 
  Ranjith_share V - Faruk_share V = 1200 :=
by
  rw [Faruk_share, Ranjith_share]
  subst hV
  -- It's just a declaration of the problem and sorry is used to skip the proof.
  sorry

end shares_difference_l427_42751


namespace avg_wx_l427_42767

theorem avg_wx (w x y : ℝ) (h1 : 3 / w + 3 / x = 3 / y) (h2 : w * x = y) : (w + x) / 2 = 1 / 2 :=
by
  -- omitted proof
  sorry

end avg_wx_l427_42767


namespace pyramid_volume_l427_42710

theorem pyramid_volume (b : ℝ) (h₀ : b > 0) :
  let base_area := (b * b * (Real.sqrt 3)) / 4
  let height := b / 2
  let volume := (1 / 3) * base_area * height
  volume = (b^3 * (Real.sqrt 3)) / 24 :=
sorry

end pyramid_volume_l427_42710


namespace original_price_l427_42728

theorem original_price (price_paid original_price : ℝ) 
  (h₁ : price_paid = 5) 
  (h₂ : price_paid = original_price / 10) : 
  original_price = 50 := by
  sorry

end original_price_l427_42728


namespace thomas_spends_40000_in_a_decade_l427_42748

/-- 
Thomas spends 4k dollars every year on his car insurance.
One decade is 10 years.
-/
def spending_per_year : ℕ := 4000

def years_in_a_decade : ℕ := 10

/-- 
We need to prove that the total amount Thomas spends in a decade on car insurance equals $40,000.
-/
theorem thomas_spends_40000_in_a_decade : spending_per_year * years_in_a_decade = 40000 := by
  sorry

end thomas_spends_40000_in_a_decade_l427_42748


namespace ratio_of_perimeter_to_length_XY_l427_42704

noncomputable def XY : ℝ := 17
noncomputable def XZ : ℝ := 8
noncomputable def YZ : ℝ := 15
noncomputable def ZD : ℝ := 240 / 17

-- Defining the perimeter P
noncomputable def P : ℝ := 17 + 2 * (240 / 17)

-- Finally, the statement with the ratio in the desired form
theorem ratio_of_perimeter_to_length_XY : 
  (P / XY) = (654 / 289) :=
by
  sorry

end ratio_of_perimeter_to_length_XY_l427_42704


namespace device_prices_within_budget_l427_42742

-- Given conditions
def x : ℝ := 12 -- Price of each type A device in thousands of dollars
def y : ℝ := 10 -- Price of each type B device in thousands of dollars
def budget : ℝ := 110 -- The budget in thousands of dollars

-- Conditions as given equations and inequalities
def condition1 : Prop := 3 * x - 2 * y = 16
def condition2 : Prop := 3 * y - 2 * x = 6
def budget_condition (a : ℕ) : Prop := 12 * a + 10 * (10 - a) ≤ budget

-- Theorem to prove
theorem device_prices_within_budget :
  condition1 ∧ condition2 ∧
  (∀ a : ℕ, a ≤ 5 → budget_condition a) :=
by sorry

end device_prices_within_budget_l427_42742


namespace rectangle_same_color_exists_l427_42776

def color := ℕ -- We use ℕ as a stand-in for three colors {0, 1, 2}

def same_color_rectangle_exists (coloring : (Fin 4) → (Fin 82) → color) : Prop :=
  ∃ (i j : Fin 4) (k l : Fin 82), i ≠ j ∧ k ≠ l ∧
    coloring i k = coloring i l ∧
    coloring j k = coloring j l ∧
    coloring i k = coloring j k

theorem rectangle_same_color_exists :
  ∀ (coloring : (Fin 4) → (Fin 82) → color),
  same_color_rectangle_exists coloring :=
by
  sorry

end rectangle_same_color_exists_l427_42776


namespace volume_of_pyramid_l427_42780

theorem volume_of_pyramid (A B C : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (28, 0)) (hC : C = (12, 20))
  (D : ℝ × ℝ) (hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E : ℝ × ℝ) (hE : E = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (F : ℝ × ℝ) (hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (∃ h : ℝ, h = 10 ∧ ∃ V : ℝ, V = (1 / 3) * 70 * h ∧ V = 700 / 3) :=
by sorry

end volume_of_pyramid_l427_42780


namespace area_of_quadrilateral_l427_42786

theorem area_of_quadrilateral (θ : ℝ) (sin_θ : Real.sin θ = 4/5) (b1 b2 : ℝ) (h: ℝ) (base1 : b1 = 14) (base2 : b2 = 20) (height : h = 8) : 
  (1 / 2) * (b1 + b2) * h = 136 := by
  sorry

end area_of_quadrilateral_l427_42786


namespace number_of_fowls_l427_42708

theorem number_of_fowls (chickens : ℕ) (ducks : ℕ) (h1 : chickens = 28) (h2 : ducks = 18) : chickens + ducks = 46 :=
by
  sorry

end number_of_fowls_l427_42708


namespace max_distance_l427_42788

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end max_distance_l427_42788


namespace total_letters_sent_l427_42779

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l427_42779


namespace number_of_children_l427_42771

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l427_42771


namespace carol_first_toss_six_probability_l427_42765

theorem carol_first_toss_six_probability :
  let p := 1 / 6
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  (prob_carol_first_six / (1 - prob_cycle)) = 125 / 671 :=
by
  let p := (1 / 6:ℚ)
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  have sum_geo_series : prob_carol_first_six / (1 - prob_cycle) = 125 / 671 := sorry
  exact sum_geo_series

end carol_first_toss_six_probability_l427_42765


namespace minimum_police_officers_needed_l427_42736

def grid := (5, 8)
def total_intersections : ℕ := 54
def max_distance_to_police := 2

theorem minimum_police_officers_needed (min_police_needed : ℕ) :
  (min_police_needed = 6) := sorry

end minimum_police_officers_needed_l427_42736


namespace area_closed_figure_sqrt_x_x_cube_l427_42754

noncomputable def integral_diff_sqrt_x_cube (a b : ℝ) :=
∫ x in a..b, (Real.sqrt x - x^3)

theorem area_closed_figure_sqrt_x_x_cube :
  integral_diff_sqrt_x_cube 0 1 = 5 / 12 :=
by
  sorry

end area_closed_figure_sqrt_x_x_cube_l427_42754


namespace remainder_of_N_mod_D_l427_42766

/-- The given number N and the divisor 252 defined in terms of its prime factors. -/
def N : ℕ := 9876543210123456789
def D : ℕ := 252

/-- The remainders of N modulo 4, 9, and 7 as given in the solution -/
def N_mod_4 : ℕ := 1
def N_mod_9 : ℕ := 0
def N_mod_7 : ℕ := 6

theorem remainder_of_N_mod_D :
  N % D = 27 :=
by
  sorry

end remainder_of_N_mod_D_l427_42766


namespace trajectory_line_or_hyperbola_l427_42792

theorem trajectory_line_or_hyperbola
  (a b : ℝ)
  (ab_pos : a * b > 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b) :
  (∃ s t : ℝ, f (s-t) * f (s+t) = (f s)^2) →
  (∃ s t : ℝ, ((t = 0) ∨ (a * t^2 - 2 * a * s^2 + 2 * b = 0))) → true := sorry

end trajectory_line_or_hyperbola_l427_42792


namespace solve_equation_l427_42734

theorem solve_equation :
  ∀ x : ℝ, 
    (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 4)) ↔ 
      (x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2) := 
by
  intro x
  sorry

end solve_equation_l427_42734


namespace frustum_volume_correct_l427_42723

noncomputable def base_length := 20 -- cm
noncomputable def base_width := 10 -- cm
noncomputable def original_altitude := 12 -- cm
noncomputable def cut_height := 6 -- cm
noncomputable def base_area := base_length * base_width -- cm^2
noncomputable def original_volume := (1 / 3 : ℚ) * base_area * original_altitude -- cm^3
noncomputable def top_area := base_area / 4 -- cm^2
noncomputable def smaller_pyramid_volume := (1 / 3 : ℚ) * top_area * cut_height -- cm^3
noncomputable def frustum_volume := original_volume - smaller_pyramid_volume -- cm^3

theorem frustum_volume_correct :
  frustum_volume = 700 :=
by
  sorry

end frustum_volume_correct_l427_42723


namespace john_february_phone_bill_l427_42725

-- Define given conditions
def base_cost : ℕ := 30
def included_hours : ℕ := 50
def overage_cost_per_minute : ℕ := 15 -- costs per minute in cents
def hours_talked_in_February : ℕ := 52

-- Define conversion from dollars to cents
def cents_per_dollar : ℕ := 100

-- Define total cost calculation
def total_cost (base_cost : ℕ) (included_hours : ℕ) (overage_cost_per_minute : ℕ) (hours_talked : ℕ) : ℕ :=
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_cost := extra_minutes * overage_cost_per_minute
  base_cost * cents_per_dollar + extra_cost

-- State the theorem
theorem john_february_phone_bill : total_cost base_cost included_hours overage_cost_per_minute hours_talked_in_February = 4800 := by
  sorry

end john_february_phone_bill_l427_42725


namespace exists_line_intersecting_all_segments_l427_42718

theorem exists_line_intersecting_all_segments 
  (segments : List (ℝ × ℝ)) 
  (h1 : ∀ (P Q R : (ℝ × ℝ)), P ∈ segments → Q ∈ segments → R ∈ segments → ∃ (L : ℝ × ℝ → Prop), L P ∧ L Q ∧ L R) :
  ∃ (L : ℝ × ℝ → Prop), ∀ (S : (ℝ × ℝ)), S ∈ segments → L S :=
by
  sorry

end exists_line_intersecting_all_segments_l427_42718


namespace distance_karen_covers_l427_42703

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l427_42703


namespace radius_of_C3_correct_l427_42715

noncomputable def radius_of_C3
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ) : ℝ :=
if h1 : r1 = 2 ∧ r2 = 3
    ∧ (TA = 4) -- Conditions 1 and 2
   then 8
   else 0

-- Proof statement
theorem radius_of_C3_correct
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : TA = 4) :
  radius_of_C3 C1 C2 C3 r1 r2 A B T TA = 8 :=
by 
  sorry

end radius_of_C3_correct_l427_42715


namespace quincy_sold_more_than_jake_l427_42773

variables (T : ℕ) (Jake Quincy : ℕ)

def thors_sales (T : ℕ) := T
def jakes_sales (T : ℕ) := T + 10
def quincys_sales (T : ℕ) := 10 * T

theorem quincy_sold_more_than_jake (h1 : jakes_sales T = Jake) 
  (h2 : quincys_sales T = Quincy) (h3 : Quincy = 200) : 
  Quincy - Jake = 170 :=
by
  sorry

end quincy_sold_more_than_jake_l427_42773


namespace factor_of_increase_l427_42758

-- Define the conditions
def interest_rate : ℝ := 0.25
def time_period : ℕ := 4

-- Define the principal amount as a variable
variable (P : ℝ)

-- Define the simple interest formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * (T : ℝ)

-- Define the total amount function
def total_amount (P : ℝ) (SI : ℝ) : ℝ := P + SI

-- The theorem that we need to prove: The factor by which the sum of money increases is 2
theorem factor_of_increase :
  total_amount P (simple_interest P interest_rate time_period) = 2 * P := by
  sorry

end factor_of_increase_l427_42758


namespace intersection_M_N_l427_42706

def M := { y : ℝ | ∃ x : ℝ, y = 2^x }
def N := { y : ℝ | ∃ x : ℝ, y = 2 * Real.sin x }

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 2 } :=
by
  sorry

end intersection_M_N_l427_42706


namespace greatest_good_number_smallest_bad_number_l427_42720

def is_good (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ (a * d = b * c)

def is_good_iff_exists_xy (M : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≤ y ∧ M ≤ x * y ∧ (x + 1) * (y + 1) ≤ M + 49

theorem greatest_good_number : ∃ (M : ℕ), is_good M ∧ ∀ (N : ℕ), is_good N → N ≤ M :=
  by
    use 576
    sorry

theorem smallest_bad_number : ∃ (M : ℕ), ¬is_good M ∧ ∀ (N : ℕ), ¬is_good N → M ≤ N :=
  by
    use 443
    sorry

end greatest_good_number_smallest_bad_number_l427_42720


namespace theater_loss_l427_42761

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l427_42761


namespace sum_of_fractions_and_decimal_l427_42717

theorem sum_of_fractions_and_decimal :
  (3 / 10) + (5 / 100) + (7 / 1000) + 0.001 = 0.358 :=
by
  sorry

end sum_of_fractions_and_decimal_l427_42717


namespace oranges_count_l427_42795

noncomputable def initial_oranges (O : ℕ) : Prop :=
  let apples := 14
  let blueberries := 6
  let remaining_fruits := 26
  13 + (O - 1) + 5 = remaining_fruits

theorem oranges_count (O : ℕ) (h : initial_oranges O) : O = 9 :=
by
  have eq : 13 + (O - 1) + 5 = 26 := h
  -- Simplify the equation to find O
  sorry

end oranges_count_l427_42795


namespace function_passes_through_point_l427_42764

theorem function_passes_through_point (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, -1) ∧ ∀ x : ℝ, (y = a^(x-1) - 2) → y = -1 := by
  sorry

end function_passes_through_point_l427_42764


namespace sum_a_b_range_l427_42791

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem sum_a_b_range : let a := 0
                       let b := 3
                       a + b = 3 := by
  sorry

end sum_a_b_range_l427_42791


namespace smallest_ratio_is_three_l427_42722

theorem smallest_ratio_is_three (m n : ℕ) (a : ℕ) (h1 : 2^m + 1 = a * (2^n + 1)) (h2 : a > 1) : a = 3 :=
sorry

end smallest_ratio_is_three_l427_42722


namespace find_f6_l427_42796

-- Define the function f and the necessary properties
variable (f : ℕ+ → ℕ+)
variable (h1 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1)
variable (h2 : f 1 ≠ 1)

-- State the theorem to prove that f(6) = 5
theorem find_f6 : f 6 = 5 :=
sorry

end find_f6_l427_42796


namespace mod_exp_result_l427_42777

theorem mod_exp_result :
  (2 ^ 46655) % 9 = 1 :=
by
  sorry

end mod_exp_result_l427_42777


namespace cos_4theta_value_l427_42716

theorem cos_4theta_value (theta : ℝ) 
  (h : ∑' n : ℕ, (Real.cos theta)^(2 * n) = 8) : 
  Real.cos (4 * theta) = 1 / 8 := 
sorry

end cos_4theta_value_l427_42716


namespace age_of_new_person_l427_42778

theorem age_of_new_person 
    (n : ℕ) 
    (T : ℕ := n * 14) 
    (n_eq : n = 9) 
    (new_average : (T + A) / (n + 1) = 16) 
    (A : ℕ) : A = 34 :=
by
  sorry

end age_of_new_person_l427_42778


namespace jayda_spending_l427_42737

theorem jayda_spending
  (J A : ℝ)
  (h1 : A = J + (2/5) * J)
  (h2 : J + A = 960) :
  J = 400 :=
by
  sorry

end jayda_spending_l427_42737


namespace mango_rate_l427_42775

theorem mango_rate (x : ℕ) : 
  (sells_rate : ℕ) = 3 → 
  (profit_percent : ℕ) = 50 → 
  (buying_price : ℚ) = 2 := by
  sorry

end mango_rate_l427_42775


namespace jason_car_count_l427_42727

theorem jason_car_count :
  ∀ (red green purple total : ℕ),
  (green = 4 * red) →
  (red = purple + 6) →
  (purple = 47) →
  (total = purple + red + green) →
  total = 312 :=
by
  intros red green purple total h1 h2 h3 h4
  sorry

end jason_car_count_l427_42727


namespace sum_of_possible_values_l427_42745

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) :
    ∃ N₁ N₂ : ℝ, (N₁ + N₂ = 4 ∧ N₁ * (N₁ - 4) = -21 ∧ N₂ * (N₂ - 4) = -21) :=
sorry

end sum_of_possible_values_l427_42745


namespace correct_factorization_l427_42709

-- Definitions of the options given in the problem
def optionA (a : ℝ) := a^3 - a = a * (a^2 - 1)
def optionB (a b : ℝ) := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def optionC (a : ℝ) := a^2 - 2 * a - 8 = a * (a - 2) - 8
def optionD (a : ℝ) := a^2 - a + 1/4 = (a - 1/2)^2

-- Stating the proof problem
theorem correct_factorization : ∀ (a : ℝ), optionD a :=
by
  sorry

end correct_factorization_l427_42709


namespace incorrect_conclusion_l427_42798

theorem incorrect_conclusion
  (a b : ℝ) 
  (h₁ : 1/a < 1/b) 
  (h₂ : 1/b < 0) 
  (h₃ : a < 0) 
  (h₄ : b < 0) 
  (h₅ : a > b) : ¬ (|a| + |b| > |a + b|) := 
sorry

end incorrect_conclusion_l427_42798


namespace percent_decaf_coffee_l427_42757

variable (initial_stock new_stock decaf_initial_percent decaf_new_percent : ℝ)
variable (initial_stock_pos new_stock_pos : initial_stock > 0 ∧ new_stock > 0)

theorem percent_decaf_coffee :
    initial_stock = 400 → 
    decaf_initial_percent = 20 → 
    new_stock = 100 → 
    decaf_new_percent = 60 → 
    (100 * ((decaf_initial_percent / 100 * initial_stock + decaf_new_percent / 100 * new_stock) / (initial_stock + new_stock))) = 28 := 
by
  sorry

end percent_decaf_coffee_l427_42757


namespace latest_time_to_reach_80_degrees_l427_42768

theorem latest_time_to_reach_80_degrees :
  ∀ (t : ℝ), (-t^2 + 14 * t + 40 = 80) → t ≤ 10 :=
by
  sorry

end latest_time_to_reach_80_degrees_l427_42768


namespace downstream_speed_is_28_l427_42747

-- Define the speed of the man in still water
def speed_in_still_water : ℝ := 24

-- Define the speed of the man rowing upstream
def speed_upstream : ℝ := 20

-- Define the speed of the stream
def speed_stream : ℝ := speed_in_still_water - speed_upstream

-- Define the speed of the man rowing downstream
def speed_downstream : ℝ := speed_in_still_water + speed_stream

-- The main theorem stating that the speed of the man rowing downstream is 28 kmph
theorem downstream_speed_is_28 : speed_downstream = 28 := by
  sorry

end downstream_speed_is_28_l427_42747


namespace larger_triangle_perimeter_l427_42707

-- Given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

def similar (t1 t2 : Triangle) (k : ℝ) : Prop :=
  t1.a / t2.a = k ∧ t1.b / t2.b = k ∧ t1.c / t2.c = k

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define specific triangles based on the problem
def smaller_triangle : Triangle := {a := 12, b := 12, c := 15}
def larger_triangle_ratio : ℝ := 2
def larger_triangle : Triangle := {a := 12 * larger_triangle_ratio, b := 12 * larger_triangle_ratio, c := 15 * larger_triangle_ratio}

-- Main theorem statement
theorem larger_triangle_perimeter : perimeter larger_triangle = 78 :=
by 
  sorry

end larger_triangle_perimeter_l427_42707


namespace profit_percentage_example_l427_42738

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℕ) (sp_total : ℝ) (sp_count : ℕ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

theorem profit_percentage_example : profit_percentage 25 15 33 12 = 65 :=
by
  sorry

end profit_percentage_example_l427_42738


namespace ratio_goats_sold_to_total_l427_42760

-- Define the conditions
variables (G S : ℕ) (total_revenue goat_sold : ℕ)
-- The ratio of goats to sheep is 5:7
axiom ratio_goats_to_sheep : G = (5/7) * S
-- The total number of sheep and goats is 360
axiom total_animals : G + S = 360
-- Mr. Mathews makes $7200 from selling some goats and 2/3 of the sheep
axiom selling_conditions : 40 * goat_sold + 30 * (2/3) * S = 7200

-- Prove the ratio of the number of goats sold to the total number of goats
theorem ratio_goats_sold_to_total : goat_sold / G = 1 / 2 := by
  sorry

end ratio_goats_sold_to_total_l427_42760


namespace boys_count_l427_42784

/-
Conditions:
1. The total number of members in the chess team is 26.
2. 18 members were present at the last session.
3. One-third of the girls attended the session.
4. All of the boys attended the session.
-/
def TotalMembers : Nat := 26
def LastSessionAttendance : Nat := 18
def GirlsAttendance (G : Nat) : Nat := G / 3
def BoysAttendance (B : Nat) : Nat := B

/-
Main theorem statement:
Prove that the number of boys in the chess team is 14.
-/
theorem boys_count (B G : Nat) (h1 : B + G = TotalMembers) (h2 : GirlsAttendance G + BoysAttendance B = LastSessionAttendance) : B = 14 :=
by
  sorry

end boys_count_l427_42784


namespace simplify_and_evaluate_expression_l427_42714

theorem simplify_and_evaluate_expression (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5/(a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l427_42714


namespace max_value_4287_5_l427_42790

noncomputable def maximum_value_of_expression (x y : ℝ) := x * y * (105 - 2 * x - 5 * y)

theorem max_value_4287_5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 105) :
  maximum_value_of_expression x y ≤ 4287.5 :=
sorry

end max_value_4287_5_l427_42790


namespace min_distinct_integers_for_ap_and_gp_l427_42774

theorem min_distinct_integers_for_ap_and_gp (n : ℕ) :
  (∀ (b q a d : ℤ), b ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 →
    (∃ (i : ℕ), i < 5 → b * (q ^ i) = a + i * d) ∧ 
    (∃ (j : ℕ), j < 5 → b * (q ^ j) ≠ a + j * d) ↔ n ≥ 6) :=
by {
  sorry
}

end min_distinct_integers_for_ap_and_gp_l427_42774


namespace ara_height_l427_42731

theorem ara_height (shea_height_now : ℝ) (shea_growth_percent : ℝ) (ara_growth_fraction : ℝ)
    (height_now : shea_height_now = 75) (growth_percent : shea_growth_percent = 0.25) 
    (growth_fraction : ara_growth_fraction = (2/3)) : 
    ∃ ara_height_now : ℝ, ara_height_now = 70 := by
  sorry

end ara_height_l427_42731


namespace jake_more_balloons_than_allan_l427_42769

-- Define the initial and additional balloons for Allan
def initial_allan_balloons : Nat := 2
def additional_allan_balloons : Nat := 3

-- Total balloons Allan has in the park
def total_allan_balloons : Nat := initial_allan_balloons + additional_allan_balloons

-- Number of balloons Jake has
def jake_balloons : Nat := 6

-- The proof statement
theorem jake_more_balloons_than_allan : jake_balloons - total_allan_balloons = 1 := by
  sorry

end jake_more_balloons_than_allan_l427_42769


namespace trig_identity_problem_l427_42797

theorem trig_identity_problem {α : ℝ} (h : Real.tan α = 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trig_identity_problem_l427_42797


namespace solution_set_of_inequality_l427_42781

variable (m x : ℝ)

-- Defining the condition
def inequality (m x : ℝ) := x^2 - (2 * m - 1) * x + m^2 - m > 0

-- Problem statement
theorem solution_set_of_inequality (h : inequality m x) : x < m-1 ∨ x > m :=
  sorry

end solution_set_of_inequality_l427_42781


namespace function_range_cosine_identity_l427_42793

theorem function_range_cosine_identity
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₀ : 0 < ω)
  (h₁ : ∀ x, f x = (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x))
  (h₂ : ∀ x, f (x + π / ω) = f x) :
  Set.Icc (f (-π / 3)) (f (π / 6)) = Set.Icc (-1 / 2) 1 :=
by
  sorry

end function_range_cosine_identity_l427_42793


namespace larger_number_l427_42713

theorem larger_number (HCF LCM a b : ℕ) (h_hcf : HCF = 28) (h_factors: 12 * 15 * HCF = LCM) (h_prod : a * b = HCF * LCM) :
  max a b = 180 :=
sorry

end larger_number_l427_42713


namespace coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l427_42705

-- Definition of the conditions
def condition_1 (Px: ℝ) (Py: ℝ) : Prop := Px = 0

def condition_2 (Px: ℝ) (Py: ℝ) : Prop := Py = Px + 3

def condition_3 (Px: ℝ) (Py: ℝ) : Prop := 
  abs Py = 2 ∧ Px > 0 ∧ Py < 0

-- Proof problem for condition 1
theorem coordinate_P_condition_1 : ∃ (Px Py: ℝ), condition_1 Px Py ∧ Px = 0 ∧ Py = -7 := 
  sorry

-- Proof problem for condition 2
theorem coordinate_P_condition_2 : ∃ (Px Py: ℝ), condition_2 Px Py ∧ Px = 10 ∧ Py = 13 :=
  sorry

-- Proof problem for condition 3
theorem coordinate_P_condition_3 : ∃ (Px Py: ℝ), condition_3 Px Py ∧ Px = 5/2 ∧ Py = -2 :=
  sorry

end coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l427_42705


namespace dice_probability_theorem_l427_42759

def at_least_three_same_value_probability (num_dice : ℕ) (num_sides : ℕ) : ℚ :=
  if num_dice = 5 ∧ num_sides = 10 then
    -- Calculating the probability
    (81 / 10000) + (9 / 20000) + (1 / 10000)
  else
    0

theorem dice_probability_theorem :
  at_least_three_same_value_probability 5 10 = 173 / 20000 :=
by
  sorry

end dice_probability_theorem_l427_42759


namespace find_carl_age_l427_42700

variables (Alice Bob Carl : ℝ)

-- Conditions
def average_age : Prop := (Alice + Bob + Carl) / 3 = 15
def carl_twice_alice : Prop := Carl - 5 = 2 * Alice
def bob_fraction_alice : Prop := Bob + 4 = (3 / 4) * (Alice + 4)

-- Conjecture
theorem find_carl_age : average_age Alice Bob Carl ∧ carl_twice_alice Alice Carl ∧ bob_fraction_alice Alice Bob → Carl = 34.818 :=
by
  sorry

end find_carl_age_l427_42700


namespace temperature_difference_l427_42785

theorem temperature_difference :
  let T_midnight := -4
  let T_10am := 5
  T_10am - T_midnight = 9 :=
by
  let T_midnight := -4
  let T_10am := 5
  show T_10am - T_midnight = 9
  sorry

end temperature_difference_l427_42785


namespace range_of_m_l427_42724

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 0 → (y = 1 - 3 * m / x) → y > 0) ↔ (m > 1 / 3) :=
sorry

end range_of_m_l427_42724


namespace sandy_gain_percent_l427_42711

def gain_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  (gain * 100) / total_cost

theorem sandy_gain_percent :
  gain_percent 900 300 1260 = 5 :=
by
  sorry

end sandy_gain_percent_l427_42711


namespace exists_schoolchild_who_participated_in_all_competitions_l427_42719

theorem exists_schoolchild_who_participated_in_all_competitions
    (competitions : Fin 50 → Finset ℕ)
    (h_card : ∀ i, (competitions i).card = 30)
    (h_unique : ∀ i j, i ≠ j → competitions i ≠ competitions j)
    (h_intersect : ∀ S : Finset (Fin 50), S.card = 30 → 
      ∃ x, ∀ i ∈ S, x ∈ competitions i) :
    ∃ x, ∀ i, x ∈ competitions i :=
by
  sorry

end exists_schoolchild_who_participated_in_all_competitions_l427_42719


namespace total_distance_walked_l427_42750

noncomputable def desk_to_fountain_distance : ℕ := 30
noncomputable def number_of_trips : ℕ := 4

theorem total_distance_walked :
  2 * desk_to_fountain_distance * number_of_trips = 240 :=
by
  sorry

end total_distance_walked_l427_42750


namespace gcd_of_differences_is_10_l427_42770

theorem gcd_of_differences_is_10 (a b c : ℕ) (h1 : b > a) (h2 : c > b) (h3 : c > a)
  (h4 : b - a = 20) (h5 : c - b = 50) (h6 : c - a = 70) : Int.gcd (b - a) (Int.gcd (c - b) (c - a)) = 10 := 
sorry

end gcd_of_differences_is_10_l427_42770


namespace radius_wire_is_4_cm_l427_42733

noncomputable def radius_of_wire_cross_section (r_sphere : ℝ) (length_wire : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * r_sphere^3
  let volume_wire := volume_sphere / length_wire
  Real.sqrt (volume_wire / Real.pi)

theorem radius_wire_is_4_cm :
  radius_of_wire_cross_section 12 144 = 4 :=
by
  unfold radius_of_wire_cross_section
  sorry

end radius_wire_is_4_cm_l427_42733


namespace smallest_add_to_multiple_of_4_l427_42763

theorem smallest_add_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ (587 + n) % 4 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (587 + m) % 4 = 0 → n ≤ m :=
  sorry

end smallest_add_to_multiple_of_4_l427_42763


namespace number_of_equilateral_triangles_l427_42726

noncomputable def parabola_equilateral_triangles (y x : ℝ) : Prop :=
  y^2 = 4 * x

theorem number_of_equilateral_triangles : ∃ n : ℕ, n = 2 ∧
  ∀ (a b c d e : ℝ), 
    (parabola_equilateral_triangles (a - 1) b) ∧ 
    (parabola_equilateral_triangles (c - 1) d) ∧ 
    ((a = e ∧ b = 0) ∨ (c = e ∧ d = 0)) → n = 2 :=
by 
  sorry

end number_of_equilateral_triangles_l427_42726


namespace cube_volume_given_surface_area_l427_42762

/-- Surface area of a cube given the side length. -/
def surface_area (side_length : ℝ) := 6 * side_length^2

/-- Volume of a cube given the side length. -/
def volume (side_length : ℝ) := side_length^3

theorem cube_volume_given_surface_area :
  ∃ side_length : ℝ, surface_area side_length = 24 ∧ volume side_length = 8 :=
by
  sorry

end cube_volume_given_surface_area_l427_42762


namespace principal_trebled_after_5_years_l427_42787

theorem principal_trebled_after_5_years (P R: ℝ) (n: ℝ) :
  (P * R * 10 / 100 = 700) →
  ((P * R * n + 3 * P * R * (10 - n)) / 100 = 1400) →
  n = 5 :=
by
  intros h1 h2
  sorry

end principal_trebled_after_5_years_l427_42787


namespace turtles_remaining_on_log_l427_42749
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l427_42749


namespace at_least_one_fuse_blows_l427_42794

theorem at_least_one_fuse_blows (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.74) (independent : ∀ (A B : Prop), A ∧ B → ¬(A ∨ B)) :
  1 - (1 - pA) * (1 - pB) = 0.961 :=
by
  sorry

end at_least_one_fuse_blows_l427_42794


namespace south_movement_notation_l427_42756

/-- If moving north 8m is denoted as +8m, then moving south 5m is denoted as -5m. -/
theorem south_movement_notation (north south : ℤ) (h1 : north = 8) (h2 : south = -north) : south = -5 :=
by
  sorry

end south_movement_notation_l427_42756


namespace find_angle_l427_42712

-- Define the conditions
variables (x : ℝ)

-- Conditions given in the problem
def angle_complement_condition (x : ℝ) := (10 : ℝ) + 3 * x
def complementary_condition (x : ℝ) := x + angle_complement_condition x = 90

-- Prove that the angle x equals to 20 degrees
theorem find_angle : (complementary_condition x) → x = 20 := 
by
  -- Placeholder for the proof
  sorry

end find_angle_l427_42712


namespace factor_of_quadratic_expression_l427_42732

def is_factor (a b : ℤ) : Prop := ∃ k, b = k * a

theorem factor_of_quadratic_expression (m : ℤ) :
  is_factor (m - 8) (m^2 - 5 * m - 24) :=
sorry

end factor_of_quadratic_expression_l427_42732


namespace problem_l427_42746

def f (x a : ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem problem (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : f 1 a = 4 :=
sorry

end problem_l427_42746


namespace cos_three_pi_over_two_l427_42744

theorem cos_three_pi_over_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  -- Provided as correct by the solution steps role
  sorry

end cos_three_pi_over_two_l427_42744
