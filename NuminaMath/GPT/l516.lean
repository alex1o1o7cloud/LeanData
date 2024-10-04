import Mathlib

namespace frac_calculation_l516_516903

theorem frac_calculation : (3 / 4 + 7 / 3) / (3 / 2) = 37 / 18 :=
by
  -- simplifying mixed numbers
  have h1 : (3 : ℚ) / 4 + (2 + 1 / 3) = (3 / 4 + 7 / 3),
  -- simplifying the denominator
  have h2 : (1 + 1 / 2) = (3 / 2),
  -- performing the division
  calc (3 / 4 + 7 / 3) / (3 / 2)
      = (3 / 4 + 7 / 3) * 2 / 3 : by sorry
      ... = 37 / 18 : by sorry

end frac_calculation_l516_516903


namespace DistanceFromPtoAB_l516_516085

noncomputable def SquareABCD := {A : (ℝ × ℝ) // A = (0, 8)} ∧
                                 {B : (ℝ × ℝ) // B = (8, 8)} ∧
                                 {C : (ℝ × ℝ) // C = (8, 0)} ∧
                                 {D : (ℝ × ℝ) // D = (0, 0)}

def MidpointM (D C : (ℝ × ℝ)) := (D.1 + C.1) / 2, 0

def CircleM (x y : ℝ) := (x - 4)^2 + y^2 = 16

def CircleB (x y : ℝ) := (x - 8)^2 + (y - 8)^2 = 64

theorem DistanceFromPtoAB :
  ∀ (P : (ℝ × ℝ)), P ∈ { P : (ℝ × ℝ) // CircleM P.1 P.2 ∧ CircleB P.1 P.2 } → 
    ∃ y : ℝ, P.2 = y ∧ y is the distance from P to line x = 0 := 
  by
  sorry

end DistanceFromPtoAB_l516_516085


namespace square_AP_eq_AB_l516_516774

namespace Geometry

-- Definitions for the square
variable {A B C D E F P : Type}

-- Given conditions
variables [Square ABCD]
variables [Midpoint E CD] [Midpoint F DA]
variable [Intersection P BE CF]

-- Prove AP = AB
theorem square_AP_eq_AB
  (h_square : Square ABCD)
  (h_mid_E : Midpoint E CD)
  (h_mid_F : Midpoint F DA)
  (h_intersection_P : Intersection P (Line B E) (Line C F)) :
  Distance AP = Distance AB :=
sorry

end Geometry

end square_AP_eq_AB_l516_516774


namespace area_of_region_l516_516620

theorem area_of_region :
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  (∫ t in (Real.pi / 3)..(Real.pi / 2), (x t) * (deriv y t)) * 2 = 2 * Real.pi - 3 * Real.sqrt 3 := by
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  have h1 : ∫ t in (Real.pi / 3)..(Real.pi / 2), x t * deriv y t = 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2*t)) / 2 := sorry
  have h2 : 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2 * t)) / 2 = 2 * Real.pi - 3 * Real.sqrt 3 := sorry
  sorry

end area_of_region_l516_516620


namespace tangent_line_at_point_l516_516106

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

theorem tangent_line_at_point (x y : ℝ) (h : f 1 = 0) (h' : deriv f 1 = -2) :
  2 * x + y - 2 = 0 :=
sorry

end tangent_line_at_point_l516_516106


namespace service_providers_choice_l516_516644

theorem service_providers_choice:
  let num_providers := 20
  let children := 4
  (finset.range num_providers).card.factorial / ((finset.range (num_providers - children)).card.factorial) = 116280 := sorry

end service_providers_choice_l516_516644


namespace john_pays_2400_per_year_l516_516417

theorem john_pays_2400_per_year
  (hours_per_month : ℕ)
  (minutes_per_hour : ℕ)
  (songs_per_minute : ℕ)
  (cost_per_song : ℕ)
  (months_per_year : ℕ)
  (H1 : hours_per_month = 20)
  (H2 : minutes_per_hour = 60)
  (H3 : songs_per_minute = 3)
  (H4 : cost_per_song = 50)
  (H5 : months_per_year = 12) :
  let minutes_per_month := hours_per_month * minutes_per_hour,
      songs_per_month := minutes_per_month / songs_per_minute,
      cost_per_month := songs_per_month * cost_per_song in
  cost_per_month * months_per_year = 2400 := by
  sorry

end john_pays_2400_per_year_l516_516417


namespace cherry_pie_probability_l516_516566

noncomputable def probability_of_cherry_pie : Real :=
  let packets := ["KK", "KV", "VV"]
  let prob :=
    (1/3 * 1/4) + -- Case KK broken, then picking from KV or VV
    (1/6 * 1/2) + -- Case KV broken (cabbage found), picking cherry from KV
    (1/3 * 1) + -- Case VV broken (cherry found), remaining cherry picked
    (1/6 * 0) -- Case KV broken (cherry found), remaining cabbage
  prob

theorem cherry_pie_probability : probability_of_cherry_pie = 2 / 3 :=
  sorry

end cherry_pie_probability_l516_516566


namespace sum_binomial_identity_l516_516470

theorem sum_binomial_identity (n : ℕ) (h : n ≥ 1) : 
  ∑ k in finset.range (n / 2 + 1), (-1)^k * (nat.choose (n + 1) k) * (nat.choose (2 * n - 2 * k - 1) n) = (n * (n + 1)) / 2 := 
by
  sorry

end sum_binomial_identity_l516_516470


namespace Ak_largest_at_166_l516_516663

theorem Ak_largest_at_166 :
  let A : ℕ → ℝ := λ k, (Nat.choose 1000 k : ℝ) * (0.2 ^ k)
  A 166 > A 165 ∧ A 166 > A 167 ∧ ∀ k, k ≠ 166 → A 166 > A k :=
by
  sorry

end Ak_largest_at_166_l516_516663


namespace count_negative_numbers_l516_516994

-- Define the expressions
def expr1 : Int := -(-2)
def expr2 : Int := -(abs (-1))
def expr3 : Int := -(abs 1)
def expr4 : Int := (-3)^2
def expr5 : Int := -(2^2)
def expr6 : Int := (-3)^3

-- Define the main statement we want to prove
theorem count_negative_numbers:
  List.count (fun x => x < 0) [expr1, expr2, expr3, expr4, expr5, expr6] = 4 := by
  sorry

end count_negative_numbers_l516_516994


namespace total_customers_l516_516230

namespace math_proof

-- Definitions based on the problem's conditions.
def tables : ℕ := 9
def women_per_table : ℕ := 7
def men_per_table : ℕ := 3

-- The theorem stating the problem's question and correct answer.
theorem total_customers : tables * (women_per_table + men_per_table) = 90 := 
by
  -- This would be expanded into a proof, but we use sorry to bypass it here.
  sorry

end math_proof

end total_customers_l516_516230


namespace abs_x_minus_one_lt_two_iff_x_times_x_minus_three_lt_zero_not_x_times_x_minus_three_lt_zero_abs_x_minus_one_lt_two_l516_516231

theorem abs_x_minus_one_lt_two_iff_x_times_x_minus_three_lt_zero (x : ℝ) : 
  (|x - 1| < 2) → (x(x - 3) < 0) :=
begin
  -- Proof omitted
  sorry
end

theorem not_x_times_x_minus_three_lt_zero_abs_x_minus_one_lt_two (x : ℝ) :
  ¬(x(x - 3) < 0) → (|x - 1| ≥ 2) :=
begin
  -- Proof omitted
  sorry
end

end abs_x_minus_one_lt_two_iff_x_times_x_minus_three_lt_zero_not_x_times_x_minus_three_lt_zero_abs_x_minus_one_lt_two_l516_516231


namespace steak_cost_l516_516275

-- Let S be the cost of each steak
variable (S : ℝ)

-- Each orders a $5 drink
def drink_cost : ℝ := 5

-- Billy pays $8 in tips
def billy_tip_payment : ℝ := 8

-- The tip rate is 20%
def tip_rate : ℝ := 0.20

-- Billy wants to pay 80% of the tip
def billy_payment_rate : ℝ := 0.80

-- The equation representing the total cost of their meals and the given conditions
def meal_cost : ℝ := 2 * S + 2 * drink_cost

-- The equation representing the tip calculation for Billy
def billy_tip : ℝ := billy_payment_rate * tip_rate * meal_cost

theorem steak_cost : S = 20 :=
by
  have h1 : billy_tip = billy_tip_payment := sorry
  have h2 : S = 20 := sorry
  exact h2

end steak_cost_l516_516275


namespace factorize_expression_l516_516664

variable (m n : ℝ)

theorem factorize_expression : 12 * m^2 * n - 12 * m * n + 3 * n = 3 * n * (2 * m - 1)^2 :=
by
  sorry

end factorize_expression_l516_516664


namespace find_B_find_cos_C_l516_516386

-- Definitions based on conditions
variables (A B C a b c : ℝ)
axiom triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
axiom triangle_angle_sum : A + B + C = π
axiom sides_opposite_angles : ∀ x, (x = A ∨ x = B ∨ x = C) → (x ≠ A ↔ side x ≠ a) ∧ (x ≠ B ↔ side x ≠ b) ∧ (x ≠ C ↔ side x ≠ c)
axiom cosine_eq : cos C / cos B = (2 * a - c) / b
axiom tan_sum : tan(A + π / 4) = 7

-- Proof problems 
theorem find_B : B = π / 3 :=
by 
  sorry

theorem find_cos_C : cos C = (3 * sqrt 3 - 4) / 10 :=
by 
  sorry

end find_B_find_cos_C_l516_516386


namespace consecutive_odd_numbers_l516_516464

-- Define the three consecutive odd numbers
def x := 4.2
def second_number := x + 2
def third_number := x + 4

-- Define the equation
def equation (y : ℝ) := 9 * x = 2 * third_number + 2 * second_number + y

-- Declare the theorem to prove
theorem consecutive_odd_numbers (y : ℝ) (h : equation y) : y = 9 :=
by
  sorry

end consecutive_odd_numbers_l516_516464


namespace fixed_point_exists_l516_516862

noncomputable def f (a : ℝ) (x : ℝ) := log a (x - 1) + 1

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 1 :=
by
  sorry

end fixed_point_exists_l516_516862


namespace probability_no_rain_four_days_l516_516515

theorem probability_no_rain_four_days (p : ℚ) (p_rain : ℚ) 
  (h_p_rain : p_rain = 2 / 3) 
  (h_p_no_rain : p = 1 - p_rain) : 
  p ^ 4 = 1 / 81 :=
by
  have h_p : p = 1 / 3, sorry
  rw [h_p],
  norm_num

end probability_no_rain_four_days_l516_516515


namespace no_solution_inequality_l516_516124

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end no_solution_inequality_l516_516124


namespace candy_distribution_l516_516459

/--
Nine distinct pieces of candy are to be distributed among four bags (red, blue, white, and green), 
each bag must receive at least one piece of candy. Prove that the number of ways to distribute 
the candies such that each bag gets at least one piece of candy is 1024.
-/
theorem candy_distribution : 
  let candies : Finset (Fin 9) := Finset.univ in
  let bags := 4 in
  (∑ (f : (Fin 9) → Fin 4) in (candies.powerset.filter (λ s, (∀ i : Fin 4, ∃ a : Fin 9, f a = i))), 1) = 1024 :=
by sorry

end candy_distribution_l516_516459


namespace Mr_Kishore_Savings_l516_516987

noncomputable def total_expenses := 
  5000 + 1500 + 4500 + 2500 + 2000 + 6100 + 3500 + 2700

noncomputable def monthly_salary (S : ℝ) := 
  total_expenses + 0.10 * S = S

noncomputable def savings (S : ℝ) := 
  0.10 * S

theorem Mr_Kishore_Savings : 
  ∃ S : ℝ, monthly_salary S ∧ savings S = 3422.22 :=
by
  sorry

end Mr_Kishore_Savings_l516_516987


namespace exists_le_square_sum_l516_516937

theorem exists_le_square_sum (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 ≤ x i) :
    let s := ∑ i in Finset.range n, ∑ j in Finset.Ico i.succ n, x i * x j
    ∃ i, (x i)^2 ≤ (2 * s) / (n^2 - n) := 
sorry

end exists_le_square_sum_l516_516937


namespace stones_required_to_pave_hall_l516_516223

theorem stones_required_to_pave_hall :
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    (area_hall_dm2 / area_stone_dm2) = 3600 :=
by
    -- Definitions
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5

    -- Convert to decimeters
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    
    -- Calculate areas
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    
    -- Calculate number of stones 
    let number_of_stones := area_hall_dm2 / area_stone_dm2

    -- Prove the required number of stones
    have h : number_of_stones = 3600 := sorry
    exact h

end stones_required_to_pave_hall_l516_516223


namespace size_of_third_file_l516_516634

theorem size_of_third_file 
  (s : ℝ) (t : ℝ) (f1 : ℝ) (f2 : ℝ) (f3 : ℝ) 
  (h1 : s = 2) (h2 : t = 120) (h3 : f1 = 80) (h4 : f2 = 90) : 
  f3 = s * t - (f1 + f2) :=
by
  sorry

end size_of_third_file_l516_516634


namespace find_x_81_9_729_l516_516517

theorem find_x_81_9_729
  (x : ℝ)
  (h : (81 : ℝ)^(x-2) / (9 : ℝ)^(x-2) = (729 : ℝ)^(2*x-1)) :
  x = 1/5 :=
sorry

end find_x_81_9_729_l516_516517


namespace crucian_carps_count_l516_516135

/--
There are colored carps and crucian carps in the aquarium tank.
Both of them eat 3 bags of feed. 
To feed all these fish, the keepers prepare 60 bags of individual feed and 15 bags of 8 packets of feeds.
There are 52 colored carps in the tank. 
Prove that there are 8 crucian carps in the tank.
-/
theorem crucian_carps_count:
  (coloredCarps crucianCarps : ℕ) 
  (feedsEachFish : ℕ) 
  (individualBags preparedBags packetsPerBag : ℕ) :
  feedsEachFish = 3 →
  individualBags = 60 →
  preparedBags = 15 →
  packetsPerBag = 8 →
  coloredCarps = 52 →
  60 + (15 * 8) / 3 - coloredCarps = crucianCarps → 
  crucianCarps = 8 :=
by
  intros feedsEachFish_eq individualBags_eq preparedBags_eq packetsPerBag_eq coloredCarps_eq calculation_eq
  sorry

end crucian_carps_count_l516_516135


namespace spent_on_accessories_l516_516081

-- Definitions based on the conditions
def original_money : ℕ := 48
def money_on_snacks : ℕ := 8
def money_left_after_purchases : ℕ := (original_money / 2) + 4

-- Proving how much Sid spent on computer accessories
theorem spent_on_accessories : ℕ :=
  original_money - (money_left_after_purchases + money_on_snacks) = 12 :=
by
  sorry

end spent_on_accessories_l516_516081


namespace serving_time_correct_l516_516647
noncomputable theory

def ounces_per_bowl := 10
def bowls_per_minute := 5
def gallons_of_soup := 6
def ounces_per_gallon := 128

def total_ounces := gallons_of_soup * ounces_per_gallon
def serving_rate := ounces_per_bowl * bowls_per_minute

def serving_time := total_ounces / serving_rate

def rounded_serving_time := Int.floor (serving_time + 0.5)

theorem serving_time_correct : rounded_serving_time = 15 := by
  sorry

end serving_time_correct_l516_516647


namespace f_neg_five_pi_div_three_eq_l516_516753

noncomputable def f : ℝ → ℝ
| x := if x ∈ Set.Ico (-Real.pi / 2) 0 then Real.cos x else 
  if x ∈ Set.Ico 0 (Real.pi / 2) then -Real.cos x else f (x - Real.pi * Real.floor (x / Real.pi))

-- Proposition encapsulating all conditions and the problem statement
theorem f_neg_five_pi_div_three_eq:
  (f(-5 * Real.pi / 3) = -1 / 2) ∧ 
  (∀ x, f(-x) = -f(x)) ∧ 
  (∀ x, f(x + Real.pi) = f(x)) ∧ 
  (∀ x, x ∈ Set.Ico (-Real.pi / 2) 0 → f(x) = Real.cos x) :=
sorry

end f_neg_five_pi_div_three_eq_l516_516753


namespace equivalent_representation_l516_516013

def original_representation (n : ℕ) : Prop :=
  n = 2 * 4^5 + 0 * 4^4 + (-1) * 4^3 + 0 * 4^2 + 2 * 4^1 + (-1) * 1

def new_representation (n : ℕ) : Prop :=
  n = -(-2) * 4^5 + 0 * 4^4 + (-1) * 4^3 + 1 * 4^2 + (-2) * 4^1 + (-1) * 1

theorem equivalent_representation :
  ∀ n, original_representation n → new_representation n :=
by
  sorry

end equivalent_representation_l516_516013


namespace jose_task_completion_time_l516_516227

theorem jose_task_completion_time
    (J N : ℝ) -- Jose's and Jane's rates in tasks per day
    (H1 : J + N = 1 / 12) -- Combined rate can complete work in 12 days
    (H2 : J = 1 / 48) -- Jose’s rate deduced from completing half the task alone in 24 days
    (H3 : N = 1 / 16) -- Jane's rate deduced from Jose's rate and combined rate
    (H4 : N > J) : -- Jane is more efficient than Jose
    (T : ℝ) -- Total time José takes to complete the whole task alone
    (HT : J * T = 1) -- Jose’s rate multiplied by total time equals one whole task
    : T = 48 :=
begin
    sorry
end

end jose_task_completion_time_l516_516227


namespace interest_rate_supposed_to_be_invested_l516_516242

variable (P T : ℕ) (additional_interest interest_rate_15 interest_rate_R : ℚ)

def simple_interest (principal: ℚ) (time: ℚ) (rate: ℚ) : ℚ := (principal * time * rate) / 100

theorem interest_rate_supposed_to_be_invested :
  P = 15000 → T = 2 → additional_interest = 900 → interest_rate_15 = 15 →
  simple_interest P T interest_rate_15 = simple_interest P T interest_rate_R + additional_interest →
  interest_rate_R = 12 := by
  intros hP hT h_add h15 h_interest
  simp [simple_interest] at *
  sorry

end interest_rate_supposed_to_be_invested_l516_516242


namespace max_area_of_triangle_PDE_l516_516616

variables (r : ℝ) (chord : ℝ) (P : ℝ → ℝ → Prop)

/-- Given r = 5 and chord BC = 8,
    Prove the maximum area of triangle PDE is 48/25 -/
theorem max_area_of_triangle_PDE (h1 : r = 5) (h2 : chord = 8)
  (P_on_arc : ∀ (B C : ℝ), P B C → P B C) 
  (PD_perp_AB : ∀ (PD : ℝ -> ℝ), ∃ D : ℝ, P D PD)
  (PE_perp_AC : ∀ (PE : ℝ -> ℝ), ∃ E : ℝ, P E PE)
  (DE_connect_BC : ∀ (DE BC : ℝ), DE + BC = chord) :
  exists maximum_area : ℝ, maximum_area = 48 / 25 :=
by
  exists 48 / 25
  sorry

end max_area_of_triangle_PDE_l516_516616


namespace max_colors_404_max_colors_406_l516_516133

theorem max_colors_404 (n k : ℕ) (h1 : n = 404) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

theorem max_colors_406 (n k : ℕ) (h1 : n = 406) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

end max_colors_404_max_colors_406_l516_516133


namespace find_p_exists_imaginary_root_find_all_other_roots_no_positive_integer_n_l516_516791

variable {p : ℝ}
def P (x : ℂ) := x^4 + 3 * x^3 + 3 * x + p

theorem find_p_exists_imaginary_root (x₁ : ℂ) (hx₁ : |x₁| = 1) (hRe : 2 * x₁.re = (1 / 2) * (Real.sqrt 17 - 3)) :
  p = -1 - 3 * x₁^3 - 3 * x₁ := sorry

theorem find_all_other_roots :
  -- Statement to find all other roots
  ∃ x₁ : ℂ, |x₁| = 1 ∧ p = -1 - 3 * x₁^3 - 3 * x₁ → -- Prove the existence of x₁
  ∀ x : ℂ, x ∈ {x₁, complex.conj x₁, -x₁, -complex.conj x₁} :=
  sorry

theorem no_positive_integer_n (x₁ : ℂ) (hx₁ : |x₁| = 1) (hRe : 2 * x₁.re = (1 / 2) * (Real.sqrt 17 - 3)) :
  ¬∃ (n : ℕ), 0 < n ∧ x₁^n = 1 := sorry

end find_p_exists_imaginary_root_find_all_other_roots_no_positive_integer_n_l516_516791


namespace min_value_AP_l516_516407

variables (A B C D P : Type)
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ P]
variables (a b c d p : vector_space ℝ)

-- Conditions
def angle_BAC_eq_pi_div_3 (A B C : vector_space ℝ) : Prop := ∃ BAC, BAC = π / 3
def D_midpoint_AB (A B D : vector_space ℝ) : Prop := ∃ AB, D = (A + B) / 2
def P_on_CD (C D P : vector_space ℝ) (t : ℝ) : Prop := P = t * C + 1 / 3 * AB
def area_ABC (A B C : vector_space ℝ) : Prop := area(A, B, C) = (3 * sqrt 3) / 2

-- Main Goal
theorem min_value_AP (A B C D P : vector_space ℝ) (t : ℝ) 
  (h1 : angle_BAC_eq_pi_div_3 A B C) 
  (h2 : D_midpoint_AB A B D) 
  (h3 : P_on_CD C D P t) 
  (h4 : area_ABC A B C) : 
  ∃ m : ℝ, (∀ P : vector_space ℝ, abs(m * P) ≥ abs(1 / sqrt 2)) := 
sorry

end min_value_AP_l516_516407


namespace solution_set_of_fractional_inequality_l516_516523

theorem solution_set_of_fractional_inequality :
  {x : ℝ | (x + 1) / (x - 3) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_fractional_inequality_l516_516523


namespace cos_theta_eq_3_div_5_l516_516376

def P : ℝ × ℝ := (3, 4)
def OP := Real.sqrt (3^2 + 4^2)
def x := 3

theorem cos_theta_eq_3_div_5 :
  OP = 5 →
  Real.cos (Real.atan2 P.2 P.1) = x / OP :=
by
  intro hOP
  rw [hOP]
  norm_num
  sorry

end cos_theta_eq_3_div_5_l516_516376


namespace cinema_meeting_day_l516_516795

-- Define the cycles for Kolya, Seryozha, and Vanya.
def kolya_cycle : ℕ := 4
def seryozha_cycle : ℕ := 5
def vanya_cycle : ℕ := 6

-- The problem statement requiring proof.
theorem cinema_meeting_day : ∃ n : ℕ, n > 0 ∧ n % kolya_cycle = 0 ∧ n % seryozha_cycle = 0 ∧ n % vanya_cycle = 0 ∧ n = 60 := 
  sorry

end cinema_meeting_day_l516_516795


namespace sum_of_B_coordinates_l516_516704

theorem sum_of_B_coordinates 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (M : ℝ × ℝ)
  (midpoint_x : (A.1 + x) / 2 = M.1) 
  (midpoint_y : (A.2 + y) / 2 = M.2) 
  (A_conds : A = (7, -1))
  (M_conds : M = (4, 3)) :
  x + y = 8 :=
by 
  sorry

end sum_of_B_coordinates_l516_516704


namespace player2_wins_with_optimal_play_l516_516462

def board_size : ℕ := 8

def valid_move (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop :=
  -- Check if placing at pos creates no L-shaped configuration
  -- This part is left abstract as describing the exact condition requires further context.

def player_loses (board : ℕ × ℕ → bool) (player : ℕ) : Prop :=
  -- Player loses if there are no valid moves they can make
  ∀ pos : ℕ × ℕ, ¬ valid_move board pos

theorem player2_wins_with_optimal_play :
  ∃ board : ℕ × ℕ → bool, ∀ moves : list (ℕ × ℕ), (moves.length % 2 = 1) ∧ (∀ pos ∈ moves, valid_move board pos) → player_loses board 1 :=
begin
  sorry,
end

end player2_wins_with_optimal_play_l516_516462


namespace complex_root_equation_l516_516619

def ω := (-1 + complex.i * real.sqrt 3) / 2
def ω_star := (-1 - complex.i * real.sqrt 3) / 2

theorem complex_root_equation :
  (ω ^ 4 + ω_star ^ 4 - 2) = -3 :=
by
  sorry

end complex_root_equation_l516_516619


namespace disjoint_subsets_with_equal_sum_l516_516540

theorem disjoint_subsets_with_equal_sum (S : Finset ℕ) (h1 : S.card = 10) (h2 : ∀ x ∈ S, x < 100) :
  ∃ A B : Finset ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ A ∩ B = ∅ ∧ ∑ a in A, a = ∑ b in B, b := 
  sorry

end disjoint_subsets_with_equal_sum_l516_516540


namespace dilation_centred_at_neg1_plus_2i_with_factor4_l516_516101

noncomputable def dilation_image (c z : ℂ) (k : ℝ) : ℂ :=
  k * (z - c) + c

theorem dilation_centred_at_neg1_plus_2i_with_factor4
  (c : ℂ) (z : ℂ) (k : ℝ) (h_c : c = -1 + 2 * complex.I) 
  (h_z : z = 3 + 4 * complex.I) (h_k : k = 4) :
  dilation_image c z k = 15 + 10 * complex.I := by
  sorry

end dilation_centred_at_neg1_plus_2i_with_factor4_l516_516101


namespace steve_cookie_boxes_l516_516088

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end steve_cookie_boxes_l516_516088


namespace rationalize_denominator_eq_sum_l516_516474

open Real

theorem rationalize_denominator_eq_sum :
  ∃ (A B C : ℤ), (C > 0) ∧ (∀ (p : ℕ), prime p → ¬ (p^3 ∣ B)) ∧
  (3 / (2 * real.cbrt 5) = (A * real.cbrt B) / C) ∧ (A + B + C = 38) :=
sorry

end rationalize_denominator_eq_sum_l516_516474


namespace inequality_solution_set_l516_516524

theorem inequality_solution_set (x : ℝ) :
  2^(x^2 - 4 * x - 3) > 2^(-3 * (x - 1)) ↔ (x < -2 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l516_516524


namespace perpendicular_lines_slope_l516_516379

theorem perpendicular_lines_slope (k : ℝ) :
  let l1 := 3 * x + 2 * y - 7 = 0,
      l2 := 4 * x + k * y = 1 in
  (∀ x y : ℝ, (l1 ∧ l2) → (-(3/2) * (-(4/k)) = -1)) → k = -6 :=
by
  sorry

end perpendicular_lines_slope_l516_516379


namespace shifted_parabola_expr_l516_516494

theorem shifted_parabola_expr :
  ∀ (x : ℝ), (let y_orig := -x^2 in let y_shifted := -((x - 2)^2) in y_orig = -x^2 → y_shifted = -(x - 2)^2) :=
by
  intro x
  simp

end shifted_parabola_expr_l516_516494


namespace find_AB_l516_516000

-- Define the right triangle ABC with given properties
variables {A B C : Type} [Real] (AC BC AB : Real)
variables (tanA : Real) (angleC : Real)

-- Given conditions from the problem
axiom tan_A_eq : tanA = 3 / 4
axiom AC_eq : AC = 6
axiom angleC_eq : angleC = π / 2

-- Goal: Prove that AB is 7.5
theorem find_AB : AB = 7.5 :=
by
  sorry

end find_AB_l516_516000


namespace equivalent_pi_value_l516_516776

noncomputable def equivalent_pi (V : ℝ) : ℝ :=
  let r := (3 * V / (4 * real.pi))^(1 / 3)
  let d := 2 * r
  let ancient_d := (16 / 9 * V)^(1 / 3)
  if (d = ancient_d) then
    3.375
  else
    0 -- This represents an error, as the comparison implies an error in logic

theorem equivalent_pi_value (V : ℝ) : equivalent_pi V = 3.375 :=
  sorry

end equivalent_pi_value_l516_516776


namespace probability_at_least_one_defective_l516_516221

open ProbabilityTheory

/-- A box contains 21 electric bulbs, out of which 4 are defective. Two bulbs are chosen 
at random from this box. Prove that the probability that at least one of these is defective is 37/105. -/
theorem probability_at_least_one_defective :
  let total_bulbs := 21,
      defective_bulbs := 4,
      chosen_bulbs := 2,
      prob_at_least_one_def := (1 - (17 / 21) * (16 / 20)) = 37 / 105 in
  prob_at_least_one_def := sorry

end probability_at_least_one_defective_l516_516221


namespace k_eq_1_l516_516437

theorem k_eq_1 
  (n m k : ℕ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hk : k > 0) 
  (h : (n - 1) * n * (n + 1) = m^k) : 
  k = 1 := 
sorry

end k_eq_1_l516_516437


namespace one_true_proposition_l516_516468

def proposition1 : Prop := ∀ x : ℝ, ∃ y : ℝ, x = y
def proposition2 : Prop := ∀ x : ℝ, ¬(∃ y : ℚ, y * y = x) → ∃ z : ℚ, z = x
def proposition3 : Prop := ((∀ x : ℝ, x = 1 → x = real.sqrt 1) ∧ (∀ x : ℝ, x = 1 → x = x ^ (1/3)))
def proposition4 : Prop := real.sqrt 25 = 5 ∨ real.sqrt 25 = -5
def proposition5 : Prop := real.sqrt 81 = 9

theorem one_true_proposition : (proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 ∧ ¬proposition5) := 
by { sorry }

end one_true_proposition_l516_516468


namespace equal_variance_sequence_properties_l516_516782

-- Define the "equal variance sequence" property
def is_equal_variance_sequence (a : ℕ → ℤ) (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → (a n)^2 - (a (n - 1))^2 = p

-- Define the sequence to check if it's an equal variance sequence
def seq1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n

-- Prove the propositions
theorem equal_variance_sequence_properties (a : ℕ → ℤ) (p : ℤ) (d : ℤ) :
  is_equal_variance_sequence seq1 0 ∧
  (is_equal_variance_sequence a p → ∀ n ≥ 2, a n ^ 2 - a (n-1) ^ 2 = p) ∧
  (is_equal_variance_sequence a p → ∀ k : ℕ, k > 0 → is_equal_variance_sequence (λ n, a (k * n)) (k * p)) ∧
  (is_equal_variance_sequence a p → (∀ n ≥ 2, a n - a (n - 1) = d) → (∀ n, a n = a 1)) :=
by
  -- All proofs omitted
  sorry

end equal_variance_sequence_properties_l516_516782


namespace factorization_of_expression_l516_516302

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l516_516302


namespace investment_duration_l516_516669

-- Define the given conditions
def Principal : ℝ := 921.0526315789474
def Rate : ℝ := 9 -- percentage per annum
def Amount : ℝ := 1120

-- Define the Interest calculation
def Interest : ℝ := Amount - Principal

-- Define the formula to calculate Time
def Time : ℝ := (Interest * 100) / (Principal * Rate)

-- Prove that the calculated time is equal to 2.4 years
theorem investment_duration : Time = 2.4 := by
  sorry

end investment_duration_l516_516669


namespace John_pays_2400_per_year_l516_516421

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l516_516421


namespace range_of_omega_l516_516686

def f (ω x : ℝ) : ℝ := sin (ω * x) - cos (ω * x)

theorem range_of_omega (ω : ℝ) (h1 : ω > 1 / 4) :
  (∀ x : ℝ, (f ω x = 0) → ¬(2 * Real.pi < x ∧ x < 3 * Real.pi)) →
  (3 / 8 ≤ ω ∧ ω ≤ 7 / 12) ∨ (7 / 8 ≤ ω ∧ ω ≤ 11 / 12) :=
sorry

end range_of_omega_l516_516686


namespace projectile_reaches_40_at_first_time_l516_516857

theorem projectile_reaches_40_at_first_time : ∃ t : ℝ, 0 < t ∧ (40 = -16 * t^2 + 64 * t) ∧ (∀ t' : ℝ, 0 < t' ∧ t' < t → ¬ (40 = -16 * t'^2 + 64 * t')) ∧ t = 0.8 :=
by
  sorry

end projectile_reaches_40_at_first_time_l516_516857


namespace dice_same_number_probability_l516_516194

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516194


namespace total_digits_even_integers_l516_516211

theorem total_digits_even_integers (N : ℕ) (h : N = 4500) : 
  (let one_digit := 4 * 1 in
   let two_digits := (N / 100 - 1) * 9 * 2 - 10 * 1 + 2 in
   let three_digits := (N / 1000 - 1) * 9 * 3 - (N / 100 - 1) * 9 * 2 in
   let four_digits := (N - (N / 1000 * 1000)) * 4 + (N / 1000 * 10 * 3) + (N / 10000) * 1 + 1000 * 4 in
   one_digit + two_digits + three_digits + four_digits = 19444) :=
sorry

end total_digits_even_integers_l516_516211


namespace group_discussion_group_7_discusses_l516_516981

def group_sizes : List ℕ :=
  [2, 3, 5, 6, 7, 8, 11, 12, 13, 17, 20, 22, 24]

noncomputable def total_students (groups : List ℕ) : ℕ :=
  groups.sum

theorem group_discussion (a b x : ℕ) (h_a : a = 6 * b) (h_total : total_students group_sizes = 150) (h_attend : a + b + x = 150) : x ≡ 4 [MOD 7] :=
by
  have h1 : a + b + x = total_students group_sizes := by
    rw [h_total, h_attend]
  have h2 : 7 * b + x = 150 := by
    rw [← h_a, h_attend]
  have h3 : x = 150 - 7 * b := by
    linarith
  have h4 : 150 ≡ 4 [MOD 7] := by
    norm_num
  exact congr_arg (λ z, z ≡ 4 [MOD 7]) h3.trans h4

theorem group_7_discusses : 
  (x : ℕ) (x ∈ group_sizes)
  (h : x ≡ 4 [MOD 7]) :
  x = 7 :=
by
  fin_cases x with
  | 7 => exact rfl
  | _ => sorry

end group_discussion_group_7_discusses_l516_516981


namespace savings_calculation_l516_516458

theorem savings_calculation
  (video_game_cost : ℝ)
  (headset_cost : ℝ)
  (gift_cost : ℝ)
  (sales_tax : ℝ)
  (initial_savings_percent : ℝ)
  (initial_weeks : ℕ)
  (subsequent_savings_percent : ℝ)
  (weekly_allowance : ℝ)
  (save_subsequent_weeks : ℕ) :
  video_game_cost = 50 →
  headset_cost = 70 →
  gift_cost = 30 →
  sales_tax = 0.12 →
  initial_savings_percent = 0.33 →
  initial_weeks = 6 →
  subsequent_savings_percent = 0.5 →
  weekly_allowance = 10 →
  save_subsequent_weeks = ((total_cost : ℝ) → (total_gift_cost : ℝ) →
  (saved_in_initial_weeks : ℝ) → (remaining_cost : ℝ) → 
  (initial_total_savings : ℝ) → (further_sales : ℝ) →
  (required_total_savings_for_remainder : ℕ) →
  -- Definitions:
  total_cost = (video_game_cost + headset_cost + gift_cost) * (1 + sales_tax) →
  total_gift_cost = gift_cost * (1 + sales_tax) →
  saved_in_initial_weeks = weekly_allowance * initial_savings_percent * initial_weeks →
  initial_total_savings = saved_in_initial_weeks + weekly_allowance * initial_savings_percent →
  remaining_cost = total_gift_cost - initial_total_savings →

  -- Time to save for the gift:
  required_total_savings_for_remainder == (remaining_cost / (weekly_allowance * subsequent_savings_percent)).ceil) →

  -- Total weeks:
  save_subsequent_weeks = 9 + ((total_cost - total_gift_cost) / (weekly_allowance * subsequent_savings_percent)).ceil) = 36 :=
begin
  sorry
end

end savings_calculation_l516_516458


namespace sufficient_but_not_necessary_l516_516048

noncomputable def is_odd_func {f : ℝ → ℝ} : Prop :=
∀ x : ℝ, f (-x) = -f(x)

theorem sufficient_but_not_necessary (φ : ℝ) (hφ : φ = π / 2) : 
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f x = cos (x + φ)) → is_odd_func f) ∧ (¬ ∀ k : ℤ, φ = k * π + (π / 2)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l516_516048


namespace at_least_one_d_i_eq_one_l516_516936

theorem at_least_one_d_i_eq_one 
    (n : ℕ) 
    (d : Fin (n+1) → Fin 3) 
    (h1 : ∀ i, d i ∈ ({0, 1, 2} : Set (Fin 3))) 
    (h2 : ∃ k : ℕ, (∑ i in Finset.range (n + 1), (3^i) * (d i)) = k^2) : 
    ∃ i, d i = 1 := sorry

end at_least_one_d_i_eq_one_l516_516936


namespace dan_marbles_l516_516286

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  original_marbles = 128 →
  given_marbles = 32 →
  remaining_marbles = original_marbles - given_marbles →
  remaining_marbles = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dan_marbles_l516_516286


namespace find_side_b_l516_516789

theorem find_side_b (A B : ℝ) (a : ℝ) (hA : A = 30) (hB : B = 45) (ha : a = Real.sqrt 2): 
  ∃ b : ℝ, b = 2 :=
by
  use 2
  sorry

end find_side_b_l516_516789


namespace rescue_boat_area_and_sum_l516_516534

theorem rescue_boat_area_and_sum : 
  let time := 12 / 60 -- 12 minutes in hours
  let river_speed := 40 -- miles per hour
  let land_speed := 10 -- miles per hour
  let max_river_distance := river_speed * time -- Maximum distance along the river
  let max_land_distance := land_speed * time -- Maximum distance on land
  let area := 4 * (π * ∫ x in 0..max_river_distance, (2 - x / (4 * 10 / river_speed))^2) / 4
  let normalized_area := (232 * π) / 6 -- Normalized area calculation
  let sum_of_parts := 232 + 6
  area = normalized_area ∧ sum_of_parts = 238 :=
by
  -- Begin proof here
  sorry

end rescue_boat_area_and_sum_l516_516534


namespace cone_new_height_eq_sqrt_85_l516_516963

/-- A cone has a uniform circular base of radius 6 feet and a slant height of 13 feet.
    After the side breaks, the slant height reduces by 2 feet, making the new slant height 11 feet.
    We need to determine the new height from the base to the tip of the cone, and prove it is sqrt(85). -/
theorem cone_new_height_eq_sqrt_85 :
  let r : ℝ := 6
  let l : ℝ := 13
  let l' : ℝ := 11
  let h : ℝ := Real.sqrt (13^2 - 6^2)
  let H : ℝ := Real.sqrt (11^2 - 6^2)
  H = Real.sqrt 85 :=
by
  sorry


end cone_new_height_eq_sqrt_85_l516_516963


namespace sum_vectors_regular_ngon_eq_zero_l516_516838

open Complex

theorem sum_vectors_regular_ngon_eq_zero (n : ℕ) (h_n : 0 < n) : 
  (∑ k in Finset.range n, exp (2 * π * I * k / n)) = 0 := 
sorry

end sum_vectors_regular_ngon_eq_zero_l516_516838


namespace linear_function_l516_516436

open Real

-- Given function f that is twice differentiable
variable (f : ℝ → ℝ)
variable (h_f_diff : ∀ x, Differentiable ℝ f x ∧ Differentiable ℝ (deriv f) x)

-- Given condition of the integral form
variable (h_condition : ∀ x y, y > 0 → (1/(2*y)) * ∫ t in (x-y)..(x+y), f t = f x)

theorem linear_function (h_f_diff : ∀ x, Differentiable ℝ f x ∧ Differentiable ℝ (deriv f) x)
  (h_condition : ∀ x y, y > 0 → (1/(2*y)) * ∫ t in (x-y)..(x+y), f t = f x) :
  ∃ (a b : ℝ), ∀ x : ℝ, f x = a * x + b := 
sorry

end linear_function_l516_516436


namespace smallest_prime_with_digits_sum_22_l516_516208

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l516_516208


namespace find_n_l516_516107

theorem find_n (n : ℕ) (h1 : 6! / (6 - n)! = 120) : n = 3 :=
sorry

end find_n_l516_516107


namespace equal_sum_S_2008_l516_516751

def an (n : ℕ) : ℤ := sorry -- Define the sequence
def S (n : ℕ) : ℤ := ∑ i in finset.range n, an (i + 1)

axiom equal_sum_sequence (n : ℕ) : an n + an (n + 1) = -3
axiom initial_condition : an 1 = 1

theorem equal_sum_S_2008 : S 2008 = -3012 :=
by
  -- Adding the sorry to indicate the proof is omitted
  sorry

end equal_sum_S_2008_l516_516751


namespace smallest_t_for_circle_l516_516110

theorem smallest_t_for_circle (t : ℝ) :
  (∀ r θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) → t ≥ π :=
by sorry

end smallest_t_for_circle_l516_516110


namespace largest_six_digit_number_l516_516670

-- Define the six-digit number with digits a, b, c, d, e, f
def is_valid_six_digit_number (a b c d e f : ℕ) : Prop :=
  c = a + b ∧ d = a + 2 * b ∧ e = 2 * a + 3 * b ∧ f = 3 * a + 5 * b ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

-- Express the problem in terms of finding the largest six-digit number
theorem largest_six_digit_number : ∃ a b c d e f : ℕ, 
  is_valid_six_digit_number a b c d e f ∧ 
  (10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f = 303369) := 
by
  use 3, 0, 3, 3, 6, 9
  simp [is_valid_six_digit_number]
  split; repeat {norm_num}
  sorry

end largest_six_digit_number_l516_516670


namespace geometric_series_ratio_l516_516109

theorem geometric_series_ratio (a r : ℝ) 
  (h_series : ∑' n : ℕ, a * r^n = 18 )
  (h_odd_series : ∑' n : ℕ, a * r^(2*n + 1) = 8 ) : 
  r = 4 / 5 := 
sorry

end geometric_series_ratio_l516_516109


namespace trebled_resultant_is_correct_l516_516596

-- Definitions based on the conditions provided in step a)
def initial_number : ℕ := 5
def doubled_result : ℕ := initial_number * 2
def added_15_result : ℕ := doubled_result + 15
def trebled_resultant : ℕ := added_15_result * 3

-- We need to prove that the trebled resultant is equal to 75
theorem trebled_resultant_is_correct : trebled_resultant = 75 :=
by
  sorry

end trebled_resultant_is_correct_l516_516596


namespace probability_all_black_after_rotation_l516_516952

/-- A 4x4 grid has 16 unit squares each painted either white or black with equal probability.
    After rotating the grid by 90 degrees clockwise and painting each white square at a previously black position to black,
    the probability that the entire grid is black is 1/1048576. -/
theorem probability_all_black_after_rotation :
  ∀ (grid : Fin 16 → Bool), 
    (∀ i, grid i = tt ∨ grid i = ff) →
    -- Probability that a single unit square is black
    (∀ i, Prob (grid i = tt) = 1/2) →
    -- The probability that entire grid is black after rotation operation
    Prob (all_black_after_rotation grid) = 1 / 1048576 :=
by sorry

end probability_all_black_after_rotation_l516_516952


namespace same_number_probability_four_dice_l516_516170

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516170


namespace arithmetic_sequence_30th_term_difference_l516_516601

theorem arithmetic_sequence_30th_term_difference (d : ℚ) (a : ℕ → ℚ) :
  (∀ n, a n = 60 + (n - 75) * d) ∧ (∀ n, 20 ≤ a n ∧ a n ≤ 90) ∧ (finset.sum (finset.range 150) a = 9000) →
  (|d| ≤ 30/149) →
  (let L := 60 - 119 * d;
       G := 60 + 119 * d in
       G - L = 7140 / 149) := by
  intros h₁ h₂
  sorry

end arithmetic_sequence_30th_term_difference_l516_516601


namespace augmented_matrix_eq_l516_516737

variable {R : Type} [CommRing R]

def system_of_eqns_matrix : Matrix (Fin 2) (Fin 3) R :=
  ![![2, -7, -3], ![4, -1, 5]]

def system_of_eqns : Matrix (Fin 2) (Fin 2) R :=
  ![![2, -7], ![4, -1]]

def constants : Matrix (Fin 2) (Fin 1) R :=
  ![[-3], [5]]

theorem augmented_matrix_eq :
  aug system_of_eqns constants = system_of_eqns_matrix :=
by sorry

end augmented_matrix_eq_l516_516737


namespace inequality_solution_set_l516_516883

theorem inequality_solution_set {x : ℝ} : 2 * x^2 - x - 1 > 0 ↔ (x < -1 / 2 ∨ x > 1) := 
sorry

end inequality_solution_set_l516_516883


namespace farmer_feed_total_cost_l516_516222

/-- 
A farmer spent $35 on feed for chickens and goats. He spent 40% of the money on chicken feed, which he bought at a 50% discount off the full price, and spent the rest on goat feed, which he bought at full price. Prove that if the farmer had paid full price for both the chicken feed and the goat feed, he would have spent $49.
-/
theorem farmer_feed_total_cost
  (total_spent : ℝ := 35)
  (chicken_feed_fraction : ℝ := 0.40)
  (goat_feed_fraction : ℝ := 0.60)
  (discount : ℝ := 0.50)
  (chicken_feed_discounted : ℝ := chicken_feed_fraction * total_spent)
  (chicken_feed_full_price : ℝ := chicken_feed_discounted / (1 - discount))
  (goat_feed_full_price : ℝ := goat_feed_fraction * total_spent):
  chicken_feed_full_price + goat_feed_full_price = 49 := 
sorry

end farmer_feed_total_cost_l516_516222


namespace find_r_l516_516369

theorem find_r (r s : ℝ) (h_quadratic : ∀ y, y^2 - r * y - s = 0) (h_r_pos : r > 0) 
    (h_root_diff : ∀ (y₁ y₂ : ℝ), (y₁ = (r + Real.sqrt (r^2 + 4 * s)) / 2 
        ∧ y₂ = (r - Real.sqrt (r^2 + 4 * s)) / 2) → |y₁ - y₂| = 2) : r = 2 :=
sorry

end find_r_l516_516369


namespace binomial_sum_expression_l516_516860

open Nat

theorem binomial_sum_expression (n : ℕ) :
  ∑ k in range (n + 1), ((-2) ^ k) * (binom n k) = (-1 : ℤ) ^ n - 1 :=
by
  sorry

end binomial_sum_expression_l516_516860


namespace find_y_expression_check_point_on_graph_l516_516711

-- Given conditions
variables (x y k : ℝ)
variable h_prop : y + 3 = k * (x - 1)
variable h_cond : x = 2
variable h_cond2 : y = -2

-- Correct answers
def y_expression : ℝ := x - 4
def point_is_on_graph : Prop := y = x - 4

-- Statements to prove
theorem find_y_expression (h_prop : y + 3 = k * (x - 1)) (h_cond : x = 2) (h_cond2 : y = -2) : y = x - 4 := by
  sorry

theorem check_point_on_graph : point_is_on_graph (-1) (-5) := by
  sorry

end find_y_expression_check_point_on_graph_l516_516711


namespace number_of_empty_chests_l516_516973

theorem number_of_empty_chests (total_chests : ℕ) (non_empty_chests : ℕ) :
  non_empty_chests = 2006 →
  total_chests = (10 * non_empty_chests) + 1 →
  ∃ empty_chests : ℕ, empty_chests = total_chests - (1 + non_empty_chests) ∧ empty_chests = 18054 :=
begin
  intros h_non_empty_chests h_total_chests,
  use total_chests - (1 + non_empty_chests),
  split,
  { exact rfl },
  { rw [h_non_empty_chests, h_total_chests],
    norm_num, }
end

end number_of_empty_chests_l516_516973


namespace problem_tetrahedron_l516_516404

-- Let Tetra denote a tetrahedron with vertices S, A, B, C
structure Tetra (S A B C : Type)

-- Define properties
def is_perpendicular {T : Type} [inner_product_space ℝ T] (x y : T) : Prop := inner_product_space.orthogonal x y

def is_centroid {T : Type} (M : T) (A B C : T) : Prop := sorry  -- Definition of centroid

def is_midpoint {T : Type} (P D E : T) : Prop := sorry  -- Definition of midpoint

noncomputable def is_parallel {T : Type} [inner_product_space ℝ T] (DP SC : T) : Prop := sorry  -- Definition of parallel

-- Main proof statements
theorem problem_tetrahedron 
  {S A B C M D D' : Type}
  [inner_product_space ℝ S]
  [inner_product_space ℝ A]
  [inner_product_space ℝ B]
  [inner_product_space ℝ C]
  [inner_product_space ℝ M]
  [inner_product_space ℝ D]
  [inner_product_space ℝ D']
  (h1 : is_perpendicular SA SB)
  (h2 : is_perpendicular SB SC)
  (h3 : is_perpendicular SC SA)
  (h4 : is_centroid M A B C)
  (h5 : is_midpoint D A B)
  (h6 : is_parallel D SC) : 
  ∃ (D' : Type), sorry := 
sorry

end problem_tetrahedron_l516_516404


namespace min_value_x_squared_sub_3x_add_2023_max_value_neg_2x_squared_add_x_add_3_l516_516308

-- Part 1: Minimum value of x^2 - 3x + 2023
theorem min_value_x_squared_sub_3x_add_2023 (x : ℝ) : 
  x^2 - 3 * x + 2023 ≥ 2020.75 :=
begin
  sorry
end

-- Part 2: Maximum value of -2x^2 + x + 3
theorem max_value_neg_2x_squared_add_x_add_3 (x : ℝ) : 
  -2 * x^2 + x + 3 ≤ 25 / 8 :=
begin
  sorry
end

end min_value_x_squared_sub_3x_add_2023_max_value_neg_2x_squared_add_x_add_3_l516_516308


namespace factor_expression_l516_516300

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l516_516300


namespace part_a_part_b_l516_516944

-- Part a)
theorem part_a (a : Fin 100 → ℝ) (h_diff : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ (s : Fin 8 → Fin 100),
  ∀ (t : Fin 9 → Fin 100),
    (∃ i j, s i = t j) →
    (∑ i, a (s i)) / 8 ≠ (∑ j, a (t j)) / 9 :=
sorry

-- Part b)
theorem part_b (a : Fin 100 → ℤ)
  (h : ∀ (s : Fin 8 → Fin 100), ∃ (t : Fin 9 → Fin 100), 
       ((∑ i, a (s i)) / 8) = ((∑ j, a (t j)) / 9)) :
  ∀ i j, a i = a j :=
sorry

end part_a_part_b_l516_516944


namespace sum_of_specific_terms_in_arithmetic_sequence_l516_516035

theorem sum_of_specific_terms_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_S11 : S 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end sum_of_specific_terms_in_arithmetic_sequence_l516_516035


namespace same_face_probability_l516_516178

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516178


namespace a100_gt_two_pow_99_l516_516695

theorem a100_gt_two_pow_99 
  (a : ℕ → ℤ) 
  (h1 : a 1 > a 0)
  (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l516_516695


namespace ratio_area_BMNC_to_ABC_l516_516771

theorem ratio_area_BMNC_to_ABC (a α : ℝ) (h_isosceles : is_isosceles_triangle a α) (h_obtuse : α > 90)
  (BN CM : ℝ) (h_altitudes_BN_CM : altitudes BN CM a α) :
  (area_BMNC BN CM α) / (area_ABC a α) = 4 * (sin (α / 2))^4 :=
by
  sorry

end ratio_area_BMNC_to_ABC_l516_516771


namespace factor_expression_l516_516303

theorem factor_expression (x : ℝ) : 45 * x^2 + 135 * x = 45 * x * (x + 3) := 
by
  sorry

end factor_expression_l516_516303


namespace dice_same_number_probability_l516_516197

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516197


namespace M_eq_P_l516_516033

open Set

def M : Set ℝ := { x | ∃ k : ℤ, x = (3 * k - 2) * Real.pi }

def P : Set ℝ := { y | ∃ λ : ℤ, y = (3 * λ + 1) * Real.pi }

theorem M_eq_P : M = P :=
by {
  sorry
}

end M_eq_P_l516_516033


namespace same_face_probability_l516_516182

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516182


namespace rounding_increases_value_l516_516053

theorem rounding_increases_value (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (rounded_a : ℕ := a + 1)
  (rounded_b : ℕ := b - 1)
  (rounded_c : ℕ := c + 1)
  (rounded_d : ℕ := d + 1) :
  (rounded_a * rounded_d) / rounded_b + rounded_c > (a * d) / b + c := 
sorry

end rounding_increases_value_l516_516053


namespace maci_school_supplies_cost_l516_516452

theorem maci_school_supplies_cost :
  let blue_pen_cost := 0.10
  let red_pen_cost := 2 * blue_pen_cost
  let pencil_cost := red_pen_cost / 2
  let notebook_cost := 10 * blue_pen_cost
  let blue_pen_count := 10
  let red_pen_count := 15
  let pencil_count := 5
  let notebook_count := 3
  let total_pen_count := blue_pen_count + red_pen_count
  let total_cost_before_discount := 
      blue_pen_count * blue_pen_cost + 
      red_pen_count * red_pen_cost + 
      pencil_count * pencil_cost + 
      notebook_count * notebook_cost
  let pen_discount_rate := if total_pen_count > 12 then 0.10 else 0
  let notebook_discount_rate := if notebook_count > 4 then 0.20 else 0
  let pen_discount := pen_discount_rate * (blue_pen_count * blue_pen_cost + red_pen_count * red_pen_cost)
  let total_cost_after_discount := 
      total_cost_before_discount - pen_discount
  total_cost_after_discount = 7.10 :=
by
  sorry

end maci_school_supplies_cost_l516_516452


namespace initial_cards_eq_4_l516_516531

theorem initial_cards_eq_4 (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  sorry

end initial_cards_eq_4_l516_516531


namespace selection_methods_count_l516_516071

open Set

def U : Set ℕ := {0, 1, 2, 3}

theorem selection_methods_count : ∃ (A : Set (Set ℕ)), 
  {(∅ : Set ℕ), U} ⊆ A ∧
  (∀ A B ∈ A, A ⊆ B ∨ B ⊆ A) ∧ 
  A.card = 4 ∧ A = 36 :=
sorry

end selection_methods_count_l516_516071


namespace sum_of_constants_l516_516749

variable (a b c : ℝ)

theorem sum_of_constants (h :  2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) :
  a + b + c = 9 := 
sorry

end sum_of_constants_l516_516749


namespace probability_of_same_number_on_four_dice_l516_516163

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516163


namespace quadratic_root_exists_in_range_l516_516129

theorem quadratic_root_exists_in_range :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ x^2 + 3 * x - 5 = 0 := 
by
  sorry

end quadratic_root_exists_in_range_l516_516129


namespace no_rain_four_days_l516_516512

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l516_516512


namespace same_number_probability_four_dice_l516_516174

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516174


namespace same_number_probability_four_dice_l516_516176

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516176


namespace painted_area_perimeter_is_260_l516_516586

-- Define the dimensions of the outer frame
def outerFrameWidth : ℝ := 100
def outerFrameHeight : ℝ := 50

-- Define the width of the border/frame
def borderWidth : ℝ := 5

-- Calculate the dimensions of the painted area
def paintedAreaWidth : ℝ := outerFrameWidth - 2 * borderWidth
def paintedAreaHeight : ℝ := outerFrameHeight - 2 * borderWidth

-- Calculate the perimeter of the painted area
def paintedAreaPerimeter : ℝ := 2 * (paintedAreaWidth + paintedAreaHeight)

-- The main statement that needs to be proven
theorem painted_area_perimeter_is_260 :
  paintedAreaPerimeter = 260 := 
by 
sory 

end painted_area_perimeter_is_260_l516_516586


namespace ellipse_properties_l516_516339

theorem ellipse_properties : 
  (∃ b : ℝ, 
    (∀ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 →  
    (√(4 + b^2) = √6) → 
    (b = √2 ∧ (∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1 
    ∧ ( (∀ F1 F2 : ℝ × ℝ, F1 = (-√2, 0) ∧ F2 = (√2, 0) ) )
    ∧ (∀ (P : ℝ × ℝ), let x0 := P.1, y0 := P.2 
                          in ∀ M Q H : ℝ × ℝ, 
                            (H = ((x0 - 2) / 2, y0 / 2)) →
                            (M.1 = H.1) ∧ (M.2 = -H.2) →
                            (Q.1 = 0) ∧ (Q.2 = -y0 / 2) →
                            (x0 ≠ 2 ∧ y0 ≠ 0) →
                            (x0 = 2 / 3)
                         )
    ) ) ) ) := 
sorry

end ellipse_properties_l516_516339


namespace percent_area_square_of_rectangle_l516_516253

-- Assume the dimensions based on the given ratios
variables (s : ℝ) (w : ℝ) (l : ℝ)

-- Conditions from the problem
def square_side_length := s
def rectangle_width := 3 * s
def rectangle_length := (3 / 2) * (3 * s)

-- Areas computed based on the dimensions
def area_square := s^2
def area_rectangle := ((3 / 2) * (3 * s)) * (3 * s)

/-- Prove that the percentage of the rectangle's area that is inside the square is 7.41% -/
theorem percent_area_square_of_rectangle : ((area_square / area_rectangle) * 100 = 7.41) := 
by
  sorry

end percent_area_square_of_rectangle_l516_516253


namespace is_parallelogram_PQRS_l516_516031

-- Define the isosceles trapezoid and related points
variables (A B C D E P Q R S : Point)

-- Define assumptions for the isosceles trapezoid and intersection point
hypothesis (h1 : isosceles_trapezoid A B C D)
hypothesis (h2 : intersection_diagonals A C B D E)
-- Define circumcenters for the given triangles
hypothesis (hP : circumcenter_triangle P A B E)
hypothesis (hQ : circumcenter_triangle Q B C E)
hypothesis (hR : circumcenter_triangle R C D E)
hypothesis (hS : circumcenter_triangle S A D E)

-- The statement we need to prove
theorem is_parallelogram_PQRS : parallelogram P Q R S :=
sorry

end is_parallelogram_PQRS_l516_516031


namespace complex_addition_example_l516_516277

theorem complex_addition_example : (5 - 5 * complex.i) + ((-2) - 1 * complex.i) - (3 + 4 * complex.i) = -10 * complex.i := 
by 
-- Prove by separating real and imaginary parts manually (steps not shown)
sorry

end complex_addition_example_l516_516277


namespace ellipse_equation_constants_l516_516996

noncomputable def ellipse_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t),
  (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_constants :
  ∃ (A B C D E F : ℤ), ∀ (x y : ℝ),
  ((∃ t : ℝ, (x, y) = ellipse_parametric_eq t) → (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2502) :=
sorry

end ellipse_equation_constants_l516_516996


namespace perimeter_of_triangle_l516_516113

-- Define the side lengths of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 7

-- Define the third side of the triangle, which is an even number and satisfies the triangle inequality conditions
def side3 : ℕ := 6

-- Define the theorem to prove the perimeter of the triangle
theorem perimeter_of_triangle : side1 + side2 + side3 = 15 := by
  -- The proof is omitted for brevity
  sorry

end perimeter_of_triangle_l516_516113


namespace angle_BDC_correct_l516_516790

theorem angle_BDC_correct (A B C D : Type) 
  (angle_A : ℝ) (angle_B : ℝ) (angle_DBC : ℝ) : 
  angle_A = 60 ∧ angle_B = 70 ∧ angle_DBC = 40 → 
  ∃ angle_BDC : ℝ, angle_BDC = 100 := 
by
  intro h
  sorry

end angle_BDC_correct_l516_516790


namespace candy_block_production_l516_516414

theorem candy_block_production :
  (friends : ℕ) (necklaces_per_person : ℕ) (total_blocks : ℕ)
  (htotal_friends  : friends = 8)
  (hnecks_per_person  : necklaces_per_person = 1)
  (candy_per_necklace : ℕ)
  (h_candies_per_necklace : candy_per_necklace = 10)
  (htotal_blocks    : total_blocks = 3)
  (total_necks : ℕ) 
  (htotal_necklaces : total_necks = friends + necklaces_per_person) 
  (strictly_total_candies  : ℕ)
  (hstrictly_total_candies : strictly_total_candies = total_necks * candy_per_necklace) :
  (strictly_total_candies / total_blocks = 30) :=
by
  sorry

end candy_block_production_l516_516414


namespace valid_pairs_count_is_40_l516_516743

def count_valid_pairs : ℕ :=
  Set.card {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1^2 + p.2 < 50 ∧ p.2^2 + p.1 < 50}

theorem valid_pairs_count_is_40 : count_valid_pairs = 40 := 
  sorry

end valid_pairs_count_is_40_l516_516743


namespace solve_for_x_l516_516313

theorem solve_for_x : ∀ x : ℝ, sqrt (4 * x + 9) = 12 → x = 33.75 := by 
  sorry

end solve_for_x_l516_516313


namespace gumball_problem_l516_516264

def total_remaining_gumballs (a b m : ℕ) : ℕ := a + b + m

def alicia_gumballs : ℕ := 20
def pedro_gumballs : ℕ := 20 + (3 / 2 : ℝ) * 20 |> Int.floor |> fun n => n + (20 % 2) -- Instead of rounding, ensure integral value
def maria_gumballs : ℕ := (1 / 2 : ℝ) * pedro_gumballs |> Int.floor |> fun n => n + (pedro_gumballs % 2) -- Similarly ensure integral value

def alicia_remaining_gumballs : ℕ := alicia_gumballs - (alicia_gumballs / 3 |> Int.floor)
def pedro_remaining_gumballs : ℕ := pedro_gumballs - (pedro_gumballs / 3 |> Int.floor)
def maria_remaining_gumballs : ℕ := maria_gumballs - (maria_gumballs / 3 |> Int.floor)

theorem gumball_problem : total_remaining_gumballs alicia_remaining_gumballs pedro_remaining_gumballs maria_remaining_gumballs = 65 :=
by
  -- Proof to be completed
  sorry

end gumball_problem_l516_516264


namespace find_y_intercept_of_second_parabola_l516_516866

theorem find_y_intercept_of_second_parabola :
  ∃ D : ℝ × ℝ, D = (0, 9) ∧ 
    (∃ A : ℝ × ℝ, A = (10, 4) ∧ 
     ∃ B : ℝ × ℝ, B = (6, 0) ∧ 
     (∀ x y : ℝ, y = (-1/4) * x ^ 2 + 5 * x - 21 → A = (10, 4)) ∧ 
     (∀ x y : ℝ, y = (1/4) * (x - B.1) ^ 2 + B.2 ∧ y = 4 ∧ B = (6, 0) → A = (10, 4))) :=
  sorry

end find_y_intercept_of_second_parabola_l516_516866


namespace same_face_probability_l516_516183

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516183


namespace height_difference_A_B_l516_516112

variables (D A E F G H B : ℝ)

-- Given conditions
def height_diff_D_A := 3.3
def height_diff_E_D := -4.2
def height_diff_F_E := -0.5
def height_diff_G_F := 2.7
def height_diff_H_G := 3.9
def height_diff_B_H := -5.6

-- Proof problem statement
theorem height_difference_A_B :
  A - B = 0.4 :=
by
  have : B - A = (D - A) + (E - D) + (F - E) + (G - F) + (H - G) + (B - H),
    by sorry,
  rw [
    height_diff_D_A,
    height_diff_E_D,
    height_diff_F_E,
    height_diff_G_F,
    height_diff_H_G,
    height_diff_B_H,
    apply the expression simplifications
  ]
  sorry

end height_difference_A_B_l516_516112


namespace at_least_one_less_than_zero_l516_516752

theorem at_least_one_less_than_zero {a b : ℝ} (h: a + b < 0) : a < 0 ∨ b < 0 := 
by 
  sorry

end at_least_one_less_than_zero_l516_516752


namespace speed_of_A_l516_516953

-- Definitions based on the problem conditions
def start_time := 10 -- 10 am
def junction_time_A := 12 -- 12 noon
def total_time := 4 -- total travel time in hours (from 10 am to 2 pm)
def speed_B := 40 -- speed of B in kmph
def distance_at_2_pm := 160 -- distance between A and B at 2 pm in km

-- Statement to prove the speed of A
theorem speed_of_A (x : ℝ) (h₁ : (2 * x)^2 + 80^2 = 160^2) : 
  x = 40 * real.sqrt 3 :=
by
  sorry

end speed_of_A_l516_516953


namespace part1_part2_l516_516728

open Real

noncomputable def f (a x : ℝ) := a * log x - (2 * (x - 1)) / (x + 1)

-- Part 1
theorem part1 (a x : ℝ) (h : 0 < x) : a ≥ 1 ↔ (a / x - 4 / ((x + 1)^2) ≥ 0) :=
sorry

-- Part 2
theorem part2 (n : ℕ) (h : 0 < n) : 
    (Σ' (k : ℕ) (H : 0 < k ∧ k ≤ n), (sqrt (k + 1) - sqrt k) / (sqrt (k + 1) + sqrt k)) < 1 / 4 * log (n + 1) :=
sorry

end part1_part2_l516_516728


namespace smallest_prime_with_digit_sum_22_l516_516205

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem smallest_prime_with_digit_sum_22 :
  (∃ n : ℕ, is_prime n ∧ digit_sum n = 22 ∧ ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22) → n ≤ m) ∧
  ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22 ∧ m < 499) → false := 
sorry

end smallest_prime_with_digit_sum_22_l516_516205


namespace gcd_possible_values_l516_516549

theorem gcd_possible_values (a b : ℕ) (h : a * b = 360) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ d ∈ {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120} := 
by sorry

end gcd_possible_values_l516_516549


namespace eighty_percentile_is_10_8_l516_516518

-- Initial data set and relevant definitions
def data_set : List ℝ := [10.2, 9.7, 10.8, 9.1, 8.9, 8.6, 9.8, 9.6, 9.9, 11.2, 10.6, 11.7]

def percentile (p : ℝ) (data : List ℝ) : ℝ :=
  let sorted_data := data.qsort (λ a b => a < b)
  let pos := (p / 100) * (sorted_data.length : ℝ)
  if pos.frac = 0 then
    sorted_data.nth_le (nat.floor pos - 1) sorry
  else
    sorted_data.nth_le (nat.floor pos) sorry

theorem eighty_percentile_is_10_8 :
  percentile 80 data_set = 10.8 := sorry

end eighty_percentile_is_10_8_l516_516518


namespace min_steps_to_remove_pebbles_l516_516059

theorem min_steps_to_remove_pebbles :
  ∀ (piles : fin 100 → ℕ), 
  (∀ i, piles i = i + 1) →
  (∃ (steps : ℕ), steps = 7 ∧
    (∃ (f : fin 100 → ℕ → ℕ), 
      (∀ i, f i 0 = piles i) ∧ 
      (∀ i n, 
        (f i (n + 1) = f i n - if f i n > 2^(6-n) then 2^(6-n) else f i n) ∧
        (f i 7 = 0)))) := 
sorry

end min_steps_to_remove_pebbles_l516_516059


namespace eggs_in_each_basket_is_four_l516_516626

theorem eggs_in_each_basket_is_four 
  (n : ℕ)
  (h1 : n ∣ 16) 
  (h2 : n ∣ 28) 
  (h3 : n ≥ 2) : 
  n = 4 :=
sorry

end eggs_in_each_basket_is_four_l516_516626


namespace students_per_table_l516_516949

theorem students_per_table (
  initial_students_per_table : ℝ := 6.0
  number_of_tables : ℝ := 34.0
) : (initial_students_per_table * number_of_tables) / number_of_tables = initial_students_per_table := by
sorry

end students_per_table_l516_516949


namespace tess_jogs_single_peak_l516_516492

theorem tess_jogs_single_peak
  (AB CD AD BC : ℝ)
  (h_parallel : AB < CD)
  (h_unequal : AD ≠ BC) :
  ∃ (graph : ℝ → ℝ), 
    (∀ t, t ∈ Icc 0 1 → graph t = 0 ∨ (∃ t₁, graph t₁ = graph 1 - graph 0) ∨ (∀ t₂, graph t₂ ≤ graph t₁ → (t₁ = 0 ∨ t₁ = 1))) :=
by {
  sorry
}

end tess_jogs_single_peak_l516_516492


namespace no_solutions_l516_516309

/-- Prove that there are no pairs of positive integers (x, y) such that x² + y² + x = 2x³. -/
theorem no_solutions : ∀ x y : ℕ, 0 < x → 0 < y → (x^2 + y^2 + x = 2 * x^3) → false :=
by
  sorry

end no_solutions_l516_516309


namespace arithmetic_sequence_a20_l516_516783

theorem arithmetic_sequence_a20 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2)) → 
  (∃ a : ℕ → ℕ, a 20 = 39) :=
by
  sorry

end arithmetic_sequence_a20_l516_516783


namespace square_reflection_transforms_l516_516069

theorem square_reflection_transforms : 
  ∀ (x1 x2 y1 y2 : ℝ), 
  x1 ≠ x2 ∧ y1 ≠ y2 → 
  set.unique (λ (P : ℝ × ℝ), (P.1 + 2 * (x2 - x1), P.2 + 2 * (y2 - y1))) 4 :=
by
  sorry

end square_reflection_transforms_l516_516069


namespace find_smallest_multiples_sum_l516_516375
-- Import necessary libraries for the proof

-- Initial problem variables and conditions
def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_6 : ℕ := 102

-- Defining the problem and the solution
theorem find_smallest_multiples_sum :
  (∃ a b : ℕ, a = smallest_two_digit_multiple_of_5 ∧ b = smallest_three_digit_multiple_of_6 ∧ a + b = 112) :=
begin
  use 10,
  use 102,
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end find_smallest_multiples_sum_l516_516375


namespace polygon_diagonals_l516_516976

def exterior_angle_magnitude : ℝ := 10

theorem polygon_diagonals (n : ℕ) 
  (h1 : n * exterior_angle_magnitude = 360) : 
  let diag := n * (n - 3) / 2 in 
  diag = 594 :=
by
  sorry

end polygon_diagonals_l516_516976


namespace log_a2010_l516_516396

variable {R : Type*}
variable [Preorder R] [FloorSemiring R]

noncomputable def f (x : R) := x ^ 3 - 6 * x ^ 2 + 4 * x - 1

def is_arithmetic_sequence (a : List R) : Prop :=
  ∀ (i j : Nat), i < j ∧ j + 1 < a.length → a[j + 1] - a[j] = a[j] - a[j - 1]

def extreme_points_of (p : List R) (a b : R) : Prop :=
  a ∈ p ∧ b ∈ p ∧ deriv (f : R → R) a = 0 ∧ deriv (f : R → R) b = 0

theorem log_a2010 {a : List R} (ha : is_arithmetic_sequence a) (h_extr : extreme_points_of a (a !! 3) (a !! 4017)) :
  log (HasLog.log (4⁻¹)) (a !! 2010) = -1 / 2 :=
by
  sorry

end log_a2010_l516_516396


namespace same_number_probability_four_dice_l516_516172

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516172


namespace total_persimmons_l516_516130

-- Definitions based on conditions in a)
def totalWeight (kg : ℕ) := kg = 3
def weightPerFivePersimmons (kg : ℕ) := kg = 1

-- The proof problem
theorem total_persimmons (k : ℕ) (w : ℕ) (x : ℕ) (h1 : totalWeight k) (h2 : weightPerFivePersimmons w) : x = 15 :=
by
  -- With the definitions totalWeight and weightPerFivePersimmons given in the conditions
  -- we aim to prove that the number of persimmons, x, is 15.
  sorry

end total_persimmons_l516_516130


namespace problem_solution_l516_516854

theorem problem_solution : ∃ n : ℕ, (n > 0) ∧ (21 - 3 * n > 15) ∧ (∀ m : ℕ, (m > 0) ∧ (21 - 3 * m > 15) → m = n) :=
by
  sorry

end problem_solution_l516_516854


namespace probability_of_Q_section_l516_516127

theorem probability_of_Q_section (sections : ℕ) (Q_sections : ℕ) (h1 : sections = 6) (h2 : Q_sections = 2) :
  Q_sections / sections = 2 / 6 :=
by
  -- solution proof is skipped
  sorry

end probability_of_Q_section_l516_516127


namespace negation_of_existential_statement_l516_516117

theorem negation_of_existential_statement : 
  (¬∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_of_existential_statement_l516_516117


namespace find_ratio_b_c_l516_516409

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end find_ratio_b_c_l516_516409


namespace find_angles_of_isosceles_triangle_l516_516588

noncomputable def isosceles_triangle {A B C : Type*} [metric_space A] (triangle : triangle A B C) : Prop :=
triangle.side AB = triangle.side AC

def height_from_vertex {A B C : Type*} [metric_space A] (triangle : triangle A B C) (H : Type*) : Prop :=
H ∈ line A B ∧ H ∈ line A C ∧ point_on_height A C H

def segment_intersects_height {A B C P Q H : Type*} [metric_space A] [metric_space P] [metric_space Q] [metric_space H] 
  (triangle : triangle A B C) (segment : segment C P) : Prop :=
segment ∩ line A H = Q

def area_relation {A B C H P Q : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space H] [metric_space P]
  [metric_space Q] (triangle1 triangle2 : triangle) : Prop :=
area triangle1 = 4 * area triangle2

theorem find_angles_of_isosceles_triangle (A B C H P Q : Type*) [metric_space A] [metric_space B] [metric_space C]
  [metric_space H] [metric_space P] [metric_space Q]
  (triangle : triangle A B C)
  (h_isosceles : isosceles_triangle triangle)
  (h_height : height_from_vertex triangle H)
  (h_segment : segment_intersects_height triangle (segment C P) Q)
  (h_area : ∀ (triangle1 triangle2 : triangle B H Q), area_relation triangle1 triangle2) :
  angles triangle = (30°, 75°, 75°) :=
sorry

end find_angles_of_isosceles_triangle_l516_516588


namespace wuyang_volleyball_team_members_l516_516847

theorem wuyang_volleyball_team_members :
  (Finset.filter Nat.Prime (Finset.range 50)).card = 15 :=
by
  sorry

end wuyang_volleyball_team_members_l516_516847


namespace remainder_div_101_l516_516917

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l516_516917


namespace smallest_x_divisible_conditions_l516_516214

theorem smallest_x_divisible_conditions : ∃ x : ℕ, x > 0 ∧
  (x % 6 = 5) ∧
  (x % 7 = 6) ∧
  (x % 8 = 7) ∧
  (∀ y : ℕ, (y > 0 ∧
             (y % 6 = 5) ∧
             (y % 7 = 6) ∧
             (y % 8 = 7)) → y ≥ 167) :=
begin
  use 167,
  split,
  { -- Prove 167 is positive
    linarith },
  split,
  { -- Prove 167 % 6 = 5
    norm_num },
  split,
  { -- Prove 167 % 7 = 6
    norm_num },
  split,
  { -- Prove 167 % 8 = 7
    norm_num },
  { -- Prove that no smaller positive x satisfies the conditions
    intros y hy,
    cases hy with hy_pos hy_rem,
    rcases hy_rem with ⟨hy6, ⟨hy7, hy8⟩⟩,
    -- Perform modulus operations to derive the necessary inequality constraints...
    sorry
  }
end

end smallest_x_divisible_conditions_l516_516214


namespace r_plus_s_eq_12_l516_516114

-- Define the line equation
def line : ℝ → ℝ := λ x, -5/3 * x + 15

-- Define points P and Q
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 15)

-- Define point T on the line segment PQ
def T (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (r, s) = (t * 9, (1-t) * 15)

-- Define the areas of triangles
def area_POQ : ℝ := 1/2 * 9 * 15
def area_TOP (r s : ℝ) : ℝ := 1/2 * r * s

-- Condition that the area of ∆POQ is twice the ∆TOP
def condition_area (r s : ℝ) : Prop := area_TOP r s = area_POQ / 2

-- Prove that r + s = 12 given the conditions
theorem r_plus_s_eq_12 (r s : ℝ) (hT : T r s) (hArea : condition_area r s) : r + s = 12 :=
by
  sorry

end r_plus_s_eq_12_l516_516114


namespace sum_x_y_z_of_c_plus_d_cubed_l516_516038

noncomputable def c : ℝ := -real.sqrt (25 / 55)
noncomputable def d : ℝ := real.sqrt ((3 + real.sqrt 8)^2 / 12)

theorem sum_x_y_z_of_c_plus_d_cubed : 
  (c < 0) → (d > 0) → ∃ (x y z : ℕ), (c + d)^3 = (x * real.sqrt y) / z ∧ x + y + z = 3037 :=
by
  intros h_c_neg h_d_pos
  use [7, 30, 3000]
  sorry

end sum_x_y_z_of_c_plus_d_cubed_l516_516038


namespace gcd_36_n_eq_12_l516_516111

theorem gcd_36_n_eq_12 (n : ℕ) (h1 : 80 ≤ n) (h2 : n ≤ 100) (h3 : Int.gcd 36 n = 12) : n = 84 ∨ n = 96 :=
by
  sorry

end gcd_36_n_eq_12_l516_516111


namespace increase_by_percentage_l516_516950

theorem increase_by_percentage (original : ℝ) (percentage : ℝ) : 
  original = 700 → percentage = 0.85 → original * (1 + percentage) = 1295 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- proof goes here
  sorry

end increase_by_percentage_l516_516950


namespace collinear_points_x_value_l516_516345

theorem collinear_points_x_value (x : ℝ) :
  let A := (1 : ℝ, 1 : ℝ)
  let B := (-4 : ℝ, 5 : ℝ)
  let C := (x, 13 : ℝ)
  collinear A B C → x = -14 :=
by
  have collinear := sorry,
  exact collinear

end collinear_points_x_value_l516_516345


namespace building_height_l516_516984

def acceleration := 2.5
def constant_velocity := 5
def time_constant_velocity := 18
def time_acceleration := 4

theorem building_height : 
  let d_acc_dec := 2 * (1/2 * acceleration * time_acceleration^2) in
  let d_const := constant_velocity * time_constant_velocity in
  d_acc_dec + d_const = 100 :=
by
  have h1 : d_acc_dec = 2 * (1/2 * acceleration * time_acceleration^2) := rfl
  have h2 : d_const = constant_velocity * time_constant_velocity := rfl
  rw [h1, h2]
  have h3 : 2 * (1/2 * acceleration * 4^2) = 10 := by sorry
  have h4 : 5 * 18 = 90 := by sorry
  rw [h3, h4]
  exact rfl

end building_height_l516_516984


namespace calculate_total_area_l516_516840

noncomputable def rectangle_sides := (AB BC : ℝ)
noncomputable def AB := 3
noncomputable def BC := 4

noncomputable def D_radius := Real.sqrt (AB ^ 2 + BC ^ 2)
noncomputable def M_radius := Real.sqrt (AB ^ 2 + (BC / 2) ^ 2)

noncomputable def total_area_regions_II_and_III := 30.3

theorem calculate_total_area :
  D_radius = 5 ∧ M_radius = Real.sqrt 13 ∧
  rectangle_sides (AB BC) → 
  total_area_regions_II_and_III = 30.3 :=
by
  intros
  sorry

end calculate_total_area_l516_516840


namespace find_speed_of_B_l516_516980

-- Define the conditions
variables (d : ℝ) (x : ℝ) (t_diff : ℝ)

-- Let the distance be 12 km
def distance := d = 12

-- Let the speed of student B be x km/h and student A be 1.2 * x km/h
def speed_A := 1.2 * x
def speed_B := x

-- A arrives 10 minutes (1/6 hours) earlier than B
def time_difference := t_diff = 1 / 6

-- Relationship of times to travel the distance
def time_B := d / speed_B
def time_A := d / speed_A

-- The final proof problem
theorem find_speed_of_B (h1 : distance) (h2 : time_difference) : x = 12 :=
by
  have h3 : time_B - time_difference = time_A := sorry
  have h4 : (12 / x) - (1 / 6) = (12 / (1.2 * x)) := sorry
  have h5 : x = 12 := sorry
  exact h5

end find_speed_of_B_l516_516980


namespace average_speed_whole_journey_l516_516942

theorem average_speed_whole_journey (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 54
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  let V_avg := total_distance / total_time
  V_avg = 64.8 :=
by
  sorry

end average_speed_whole_journey_l516_516942


namespace grid_filled_correctly_l516_516304

noncomputable def filled_grid : matrix (fin 3) (fin 3) ℕ :=
![![1, 5, 3], 
  ![5, 3, 1], 
  ![3, 1, 5]]

theorem grid_filled_correctly :
  ∃ (grid : matrix (fin 3) (fin 3) ℕ), 
    grid 0 0 = 1 ∧ grid 1 1 = 3 ∧ grid 2 2 = 5 ∧
    (∀ i, list.perm (grid i) ![1, 3, 5]) ∧
    (∀ j, list.perm (vector.map (λ i, grid i j) ⟨![0,1,2], by decide⟩) ![1, 3, 5]) :=
begin
  use filled_grid,
  unfold filled_grid,
  split, repeat { split },
  all_goals { try { refl } },
  -- Remaining goals proving that each row and column contains 1,3,5 exactly are omitted
  sorry, sorry
end

end grid_filled_correctly_l516_516304


namespace sum_arithmetic_progression_l516_516557

theorem sum_arithmetic_progression (n : ℕ) (u₁ d : ℤ) : 
  let u := λ i : ℕ, u₁ + (i - 1) * d in
  Σ i in finset.range n, u (i + 1) = n * u₁ + (n * (n - 1) / 2) * d :=
by sorry

end sum_arithmetic_progression_l516_516557


namespace angle_KLM_eq_beta_l516_516405

theorem angle_KLM_eq_beta
  (A B C K L M : Type)
  (AB AC BC : ℝ)
  (hAC : AC = 1 / 2 * (AB + BC))
  (hBisector : ∀ (𝛽 : ℝ), ∃ (L : Type), ∠BAC • L = ∠BCA)
  (hK : K = midpoint A B)
  (hM : M = midpoint B C)
  (angle_ABC : ℝ)
  (hBeta : angle_ABC = β) :
  ∠KLM = β :=
sorry

end angle_KLM_eq_beta_l516_516405


namespace problem1_problem2_problem3_l516_516948

-- Define prime factor counting functions
def count_prime_factors (n : ℕ) (p : ℕ) : ℕ :=
  if n % p = 0 then 1 + count_prime_factors (n / p) p else 0

def f (n : ℕ) : ℕ :=
  ∑ p in Nat.primes.filter (λ p, p % 4 = 1), count_prime_factors n p

def g (n : ℕ) : ℕ :=
  ∑ p in Nat.primes.filter (λ p, p % 4 = 3), count_prime_factors n p

-- Problem 1
theorem problem1 (n : ℕ) (hn : 0 < n) : f(n) ≥ g(n) := 
  sorry

-- Problem 2
theorem problem2 : ∃ᶠ n in atTop, f(n) = g(n) := 
  sorry

-- Problem 3
theorem problem3 : ∃ᶠ n in atTop, f(n) > g(n) :=
  sorry

end problem1_problem2_problem3_l516_516948


namespace probability_odd_sum_l516_516019

def P := {1, 2, 3}
def Q := {1, 2, 4}
def R := {1, 3, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def probability_sum_odd : ℚ :=
  let outcomes := Prod.prod P Q R
  let odd_sum := {o | is_odd (o.1 + o.2 + o.3)}
  (outcomes ∩ odd_sum).card.to_rat / outcomes.card.to_rat

theorem probability_odd_sum :
  probability_sum_odd = 4 / 9 :=
sorry

end probability_odd_sum_l516_516019


namespace ab_leq_one_l516_516858

theorem ab_leq_one (a b x : ℝ) (h1 : (x + a) * (x + b) = 9) (h2 : x = a + b) : a * b ≤ 1 := 
sorry

end ab_leq_one_l516_516858


namespace determine_a_l516_516295

theorem determine_a (a : ℚ) (x : ℚ) : 
  (∃ r s : ℚ, (r*x + s)^2 = a*x^2 + 18*x + 16) → 
  a = 81/16 := 
sorry

end determine_a_l516_516295


namespace train_speed_in_kmh_l516_516604

-- Definitions from the conditions
def length_of_train : ℝ := 800 -- in meters
def time_to_cross_pole : ℝ := 20 -- in seconds
def conversion_factor : ℝ := 3.6 -- (km/h) per (m/s)

-- Statement to prove the train's speed in km/h
theorem train_speed_in_kmh :
  (length_of_train / time_to_cross_pole * conversion_factor) = 144 :=
  sorry

end train_speed_in_kmh_l516_516604


namespace find_f_l516_516444

-- Define the function f and assert the initial conditions
def f (x : ℝ) : ℝ := sorry

-- Theorem statement
theorem find_f (x y : ℝ)
  (h1 : f 0 = 1)
  (h2 : ∀ x y : ℝ, f(x - y) = f x - y * (2 * x - y + 1)) :
  f x = x^2 + x + 1 := 
sorry

end find_f_l516_516444


namespace sum_of_series_l516_516639

theorem sum_of_series : 
  (∑ n in Finset.range 100, 1 / ((2 * (n + 1) - 1) * (2 * (n + 1) + 1))) = 100 / 201 := 
sorry

end sum_of_series_l516_516639


namespace find_multiplier_l516_516853

/-- Define the number -/
def number : ℝ := -10.0

/-- Define the multiplier m -/
def m : ℝ := 0.4

/-- Given conditions and prove the correct multiplier -/
theorem find_multiplier (number : ℝ) (m : ℝ) 
  (h1 : ∃ m : ℝ, m * number - 8 = -12) 
  (h2 : number = -10.0) : m = 0.4 :=
by
  -- We skip the actual steps and provide the answer using sorry
  sorry

end find_multiplier_l516_516853


namespace find_a_l516_516347

theorem find_a (A B : set ℝ) (a : ℝ) : 
  (A = {x | x = 2^a ∨ x = 3}) ∧ (B = {x | x = 2 ∨ x = 3}) ∧ (A ∪ B = {x | x = 2 ∨ x = 3 ∨ x = 4}) → a = 2 :=
by
  sorry

end find_a_l516_516347


namespace inequation_proof_l516_516714

theorem inequation_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 1) :
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3 / 2) :=
by
  sorry

end inequation_proof_l516_516714


namespace equilateral_triangle_l516_516448

variables {A B C F E : Type} [PlanarGeometry A B C]

-- Assume given conditions
def is_triangle (ABC : Triangle A B C) := triangle A B C
def is_median (AF CE : Segment A F, Segment C E) := (median AF CE)
def angles_equal (angleBAF angleBCE : Angle A F B, Angle B C E) := (angle A F B = 30º ∧ angle B C E = 30º)

-- Prove that the triangle is equilateral
theorem equilateral_triangle 
  (H1 : is_triangle ABC) 
  (H2 : is_median AF CE) 
  (H3 : angles_equal (angle A F B) (angle B C E))
  : AB = BC ∧ BC = AC := 
sorry

end equilateral_triangle_l516_516448


namespace problem_solution_l516_516317

noncomputable def tau (n : ℕ) : ℕ := 
  (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card

noncomputable def S (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).sum tau

noncomputable def a : ℕ := 
  (Finset.filter (λ n, S n % 2 = 1) (Finset.range 1001)).card

noncomputable def b : ℕ := 
  (Finset.filter (λ n, S n % 2 = 0) (Finset.range 1001)).card

theorem problem_solution : abs (a - b) = 104 := 
  sorry

end problem_solution_l516_516317


namespace dense_local_minima_of_continuous_nowhere_monotone_l516_516435

noncomputable def nowhere_monotone_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ I (hI : I ⊆ s) (hI' : I ≠ ∅) (hI'' : I ≠ set.univ),
  ¬ (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y)
  ∧ ¬ (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≥ f y)

theorem dense_local_minima_of_continuous_nowhere_monotone
(f : ℝ → ℝ) (H_cont : continuous_on f (set.Icc 0 1))
(H_nowhere_monotone : nowhere_monotone_on f (set.Icc 0 1)) :
dense {x | ∃ U ∈ 𝓝 x, ∀ y ∈ U, f y ≥ f x} :=
sorry

end dense_local_minima_of_continuous_nowhere_monotone_l516_516435


namespace inscribed_circle_radius_DEF_l516_516293

noncomputable def radius_inscribed_circle (DE DF EF : ℕ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius_DEF :
  radius_inscribed_circle 26 16 20 = 5 * Real.sqrt 511.5 / 31 :=
by
  sorry

end inscribed_circle_radius_DEF_l516_516293


namespace number_of_mappings_satisfying_condition_l516_516349

open Function Set

theorem number_of_mappings_satisfying_condition :
  let A := {1, 2, 3}
  let B := {4, 5, 6, 7}
  ∃ (f : A → B), 
  (∀ x ∈ A, Odd (x + f x + x * f x)) → num_mappings (f : A → B) = 32 :=
by
  -- Definitions and assumptions
  let A := {1, 2, 3}
  let B := {4, 5, 6, 7}
  let mappings_satisfying_condition : (A → B) → Prop :=
    fun f => ∀ x ∈ A, Odd (x + f x + x * f x)
  -- The statement to be proven
  have num_mappings := (2 : ℕ) * (2 : ℕ) * (2 : ℕ) -- 2 choices for each element in A
  show ∃ (f : A → B), mappings_satisfying_condition f → num_mappings = 32
  sorry

end number_of_mappings_satisfying_condition_l516_516349


namespace remainder_div_101_l516_516914

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l516_516914


namespace arithmetic_sequence_sum_l516_516397

theorem arithmetic_sequence_sum
  (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (H1 : a1 = -2017)
  (H2 : (S 2013 : ℤ) / 2013 - (S 2011 : ℤ) / 2011 = 2)
  (H3 : ∀ n : ℕ, S n = n * a1 + (n * (n - 1) / 2) * d) :
  S 2017 = -2017 :=
by
  sorry

end arithmetic_sequence_sum_l516_516397


namespace collinear_points_m_eq_4_div_3_l516_516344

-- Definitions of points A, B, and C as per the problem conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 1⟩
def B (m : ℝ) : Point := ⟨m, 3⟩
def C : Point := ⟨4, 7⟩

-- The main theorem stating the proof goal
theorem collinear_points_m_eq_4_div_3 (m : ℝ) (h : ∃ k : ℝ, B m = A + k • (C - A)) : 
  m = 4 / 3 :=
sorry

end collinear_points_m_eq_4_div_3_l516_516344


namespace geometric_sequence_a_10_l516_516768

noncomputable def geometric_sequence := ℕ → ℝ

def a_3 (a r : ℝ) := a * r^2 = 3
def a_5_equals_8a_7 (a r : ℝ) := a * r^4 = 8 * a * r^6

theorem geometric_sequence_a_10 (a r : ℝ) (seq : geometric_sequence) (h₁ : a_3 a r) (h₂ : a_5_equals_8a_7 a r) :
  seq 10 = a * r^9 := by
  sorry

end geometric_sequence_a_10_l516_516768


namespace a_n_correct_b_n_correct_T_n_formula_l516_516693

-- Definitions from the conditions
def S (n : ℕ) : ℕ := 2 * (a n) - 2

-- Given condition that P(b_n, b_{n+1}) lies on the line y = x + 2
def P (n : ℕ) : Prop := b (n + 1) = b n + 2

-- Definitions for sequences {a_n} and {b_n}
def a : ℕ → ℕ := λ n, 2 ^ n
def b : ℕ → ℕ := λ n, 2 * n - 1

-- Proof that the sequence {a_n} is defined correctly
theorem a_n_correct (n : ℕ) : a n = 2 ^ n := 
by rwa [←nat.cast_pow, pow_succ]

-- Proof that the sequence {b_n} is defined correctly
theorem b_n_correct (n : ℕ) (h : b 1 = 1 ∧ ∀ n, P n) : b n = 2 * n - 1 := 
sorry

-- Definition for sequence {c_n} and its sum of first n terms
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := ∑ i in finset.range n, c (i + 1)

-- Proof of the sum of the first n terms of the sequence {c_n}
theorem T_n_formula (n : ℕ) : T n = (2n - 3) * 2^(n+1) + 6 :=
sorry

end a_n_correct_b_n_correct_T_n_formula_l516_516693


namespace same_face_probability_l516_516179

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516179


namespace locate_in_second_quadrant_l516_516007

def is_second_quadrant (z : ℂ) : Prop :=
    z.re < 0 ∧ z.im > 0

theorem locate_in_second_quadrant
    (z : ℂ)
    (hz : z = -1 + 2 * complex.i) :
    is_second_quadrant z :=
by
  -- placeholder for proof
  sorry

end locate_in_second_quadrant_l516_516007


namespace quadrilateral_rotation_l516_516244

variables (Z : Type) [point Z] (quad unshaded_quad : Type) [quadrilateral quad] [quadrilateral unshaded_quad]
           (rotate : point Z → quadrilateral quad → ℝ → quadrilateral unshaded_quad)

def rotation_270_degrees (quad quad_transformed : quadrilateral quad) : Prop :=
  rotate Z quad 270 = quad_transformed

theorem quadrilateral_rotation (quad_transformed : quadrilateral unshaded_quad) :
  rotation_270_degrees quad quad_transformed →
  quad_transformed = unshaded_quad :=
by
  sorry 

end quadrilateral_rotation_l516_516244


namespace stratified_sampling_appropriate_method_l516_516962

theorem stratified_sampling_appropriate_method 
    (total_employees : ℕ) 
    (senior_titles : ℕ) 
    (intermediate_titles : ℕ) 
    (general_staff : ℕ) 
    (sample_size : ℕ) :
    total_employees = 150 → 
    senior_titles = 15 → 
    intermediate_titles = 45 → 
    general_staff = 90 → 
    sample_size = 30 →
    stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size := sorry

end stratified_sampling_appropriate_method_l516_516962


namespace find_coefficients_l516_516213

theorem find_coefficients : ∃ a b : ℤ, 
  (∀ x : ℤ, (f x = a * x^3 - 6 * x^2 + b * x - 5) ∧ 
  (f 1 = -5) ∧ 
  (f (-2) = -53)) → (a = 7 ∧ b = -7) := 
sorry

end find_coefficients_l516_516213


namespace midpoint_of_segments_coincides_with_intersection_of_segments_of_opposite_midpoints_l516_516476

-- Definitions and assumptions based on the problem
variables {A B C D M N P Q O O' O'' : Type}
variables (isMidpoint : ∀ {X Y Z : Type}, X = (Z, Y) → Prop)
variables (isIntersection : ∀ {X Y Z O'' : Type}, X = (Z, Y) ∧ Y = (O'', Z) → Prop)

-- The quadrilateral ABCD with midpoints M, N, P, Q of sides AB, BC, CD, DA respectively
variables (ABCD : Type) (midpoints: M = (A, B) ∧ N = (B, C) ∧ P = (C, D) ∧ Q = (D, A))
variables (diagonals_midpoints : O = (A, C) ∧ O' = (B, D))

-- Theorem to prove
theorem midpoint_of_segments_coincides_with_intersection_of_segments_of_opposite_midpoints :
  isMidpoint O (A, C) →
  isMidpoint O' (B, D) →
  isIntersection O'' ((Q, M), (P, N)) →
  O'' = (O + O') / 2 :=
sorry

end midpoint_of_segments_coincides_with_intersection_of_segments_of_opposite_midpoints_l516_516476


namespace dot_product_values_l516_516441

variables {ℝ : Type*} [InnerProductSpace ℝ ℝ]

-- Definitions of vectors a, b, c
variables (a b c : ℝ) 

-- Norm conditions
def norm_a : ℝ := 5
def norm_b : ℝ := 8
def norm_c : ℝ := 6

-- Dot product condition
def dot_product_condition : Prop := (a * b + b * c = 0)

-- The theorem to prove
theorem dot_product_values (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : ∥a∥ = norm_a) 
  (h2 : ∥b∥ = norm_b) 
  (h3 : ∥c∥ = norm_c) 
  (h4 : dot_product_condition) : 
  (a * b >= -40 ∧  a * b <= 40) :=
sorry

end dot_product_values_l516_516441


namespace angles_ACP_and_QCB_equiv_or_supp_l516_516009

noncomputable theory
open_locale classical

variables (A B C D H P Q : Type)
variables [trapezoid : is_trapezoid A B C D] (AC_eq_BC : A.boundary C = B.boundary C) (H_mid_AB : (A.boundary midpoint B) = H)
variables (l_passes_H : line H P Q)

theorem angles_ACP_and_QCB_equiv_or_supp (A B C D H P Q : Type)
  [trapezoid : is_trapezoid A B C D] (AC_eq_BC : A.boundary C = B.boundary C) 
  (H_mid_AB : (A.boundary midpoint B) = H)
  (l_passes_H : line H P Q) :
  ∠(line A boundary C point P) = ∠(line Q boundary C point B) ∨ ∠(line A boundary C point P) + ∠(line Q boundary C point B) = 180 :=
sorry

end angles_ACP_and_QCB_equiv_or_supp_l516_516009


namespace correlation_index_l516_516008

variable (height_variation_weight_explained : ℝ)
variable (random_errors_contribution : ℝ)

def R_squared : ℝ := height_variation_weight_explained

theorem correlation_index (h1 : height_variation_weight_explained = 0.64) (h2 : random_errors_contribution = 0.36) : R_squared height_variation_weight_explained = 0.64 :=
by
  exact h1  -- Placeholder for actual proof, since only statement is required

end correlation_index_l516_516008


namespace Ben_win_probability_l516_516872

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l516_516872


namespace triangle_statement_four_incorrect_l516_516760

theorem triangle_statement_four_incorrect (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A + B + C = π) (h5 : A > π / 2) 
  (h6 : cos A < 0) : ¬ (sin B > cos C) :=
by
  sorry

end triangle_statement_four_incorrect_l516_516760


namespace toys_calculation_l516_516018

-- Define the number of toys each person has as variables
variables (Jason John Rachel : ℕ)

-- State the conditions
variables (h1 : Jason = 3 * John)
variables (h2 : John = Rachel + 6)
variables (h3 : Jason = 21)

-- Define the theorem to prove the number of toys Rachel has
theorem toys_calculation : Rachel = 1 :=
by {
  sorry
}

end toys_calculation_l516_516018


namespace probability_of_same_number_on_four_dice_l516_516167

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516167


namespace min_balls_to_ensure_15_same_color_l516_516578

theorem min_balls_to_ensure_15_same_color (red green yellow blue white black total: ℕ) (h_red : red = 28) (h_green : green = 20) (h_yellow : yellow = 12) (h_blue : blue = 20) (h_white : white = 10) (h_black : black = 10) (h_total : total = red + green + yellow + blue + white + black) : 
  total = 100 → ∃ n, n ≥ 75 ∧ (∀ balls : List ℕ, balls.length = n → balls.count 15 (.=== )) :=
by
  sorry

end min_balls_to_ensure_15_same_color_l516_516578


namespace problem_statement_l516_516499

-- Conditions
def f (x : ℝ) : ℝ := Real.sin x + x^3 + 1
def a : ℝ := f 1

-- Statement
theorem problem_statement : f (-1) = 2 - a :=
by
  -- The proof steps would go here, but we're skipping them.
  sorry

end problem_statement_l516_516499


namespace tina_more_than_katya_l516_516428

theorem tina_more_than_katya (katya_sales ricky_sales : ℕ) (tina_sales : ℕ) (
  h1 : katya_sales = 8) (h2 : ricky_sales = 9) (h3 : tina_sales = 2 * (katya_sales + ricky_sales)) :
  tina_sales - katya_sales = 26 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  rw [h3, h1]
  norm_num
  sorry

end tina_more_than_katya_l516_516428


namespace derivative_of_y_l516_516100

def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_y (x : ℝ) : deriv y x = -x * Real.sin x :=
by
  sorry

end derivative_of_y_l516_516100


namespace math_problem_l516_516818

open Real

theorem math_problem
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x + y + z = 1) :
  ( (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 ) :=
by {
  sorry
}

end math_problem_l516_516818


namespace max_y_difference_between_intersections_l516_516864

theorem max_y_difference_between_intersections :
  ∃ (x1 x2 : ℝ), (4 - 2 * x1^2 + x1^3 = 2 + 2 * x1^2 + x1^3) ∧
                 (4 - 2 * x2^2 + x2^3 = 2 + 2 * x2^2 + x2^3) ∧
                 (∀ (y1 y2 : ℝ), y1 = 4 - 2 * x1^2 + x1^3 → y2 = 4 - 2 * x2^2 + x2^3 →
                    |y1 - y2| = sqrt 2 / 2) :=
by sorry

end max_y_difference_between_intersections_l516_516864


namespace find_plane_height_l516_516946

-- Definitions of the conditions
def Rectangle (A B C D P : Type) [hABCD : ∀ A B C D : P, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A] : Prop :=
  true -- assume only type checking for a quadrilateral

def Pyramid (base : Type) (A P : Type) [is_rect : Rectangle A B C D P] : Prop :=
  true -- assume only type checking for a Pyramid on a rectangular base

structure Dimensions :=
(AB : ℝ)
(BC : ℝ)

structure Heights :=
(height_P : ℝ)
(height_plane : ℝ)
(height_P' : ℝ)

-- Given conditions in terms of dimensions and height
def dims : Dimensions := ⟨15, 20⟩
def heights : Heights := ⟨30, _, _⟩

-- Assertion regarding volumes and height ratios
def heightRatio (h k : ℝ) : Prop :=
  k^3 = 1 / 9 ∧ k = 1 / 3 ∧ h = k * heights.height_P

-- Formalize the problem statement
theorem find_plane_height :
  ∃ h_plane : ℝ, h_plane = 20 :=
by
  -- Given conditions
  have area_base := dims.AB * dims.BC
  have volume_P : ℝ := area_base * heights.height_P / 3
  have volume_P' := volume_P / 9
  have h' : ℝ := 1 / 3 * heights.height_P
  sorry -- Skip the proof steps and directly state the theorem

end find_plane_height_l516_516946


namespace same_number_on_four_dice_l516_516187

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516187


namespace sufficient_but_not_necessary_condition_sufficient_but_not_necessary_iff_l516_516806

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : |x + 1| < 1) :
  |x| < 2 :=
sorry

-- Provide a statement that acknowledges the condition should be sufficient but not proven necessary
example (x : ℝ) :
  (|x + 1| < 1) → (|x| < 2) :=
begin
  assume h : |x + 1| < 1,
  exact sufficient_but_not_necessary_condition x h,
end

theorem sufficient_but_not_necessary_iff (x : ℝ) :
  (∀ (h : |x + 1| < 1), |x| < 2) ∧ ¬(∀ (h : |x| < 2), |x + 1| < 1) :=
sorry

end sufficient_but_not_necessary_condition_sufficient_but_not_necessary_iff_l516_516806


namespace ratio_of_discretionary_income_l516_516646

theorem ratio_of_discretionary_income 
  (salary : ℝ) (D : ℝ)
  (h_salary : salary = 3500)
  (h_discretionary : 0.15 * D = 105) :
  D / salary = 1 / 5 :=
by
  sorry

end ratio_of_discretionary_income_l516_516646


namespace sum_of_angles_l516_516243

theorem sum_of_angles (W X Y Z : Type) (angle WZ XY : ℕ)
  (hWZ : angle WZ = 40) (hXY : angle XY = 20) :
  angle WXY + angle WZY = 120 := 
sorry

end sum_of_angles_l516_516243


namespace inequality_abc_l516_516074

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a) + (1/b) ≥ 4/(a + b) :=
by
  sorry

end inequality_abc_l516_516074


namespace solve_fx_eq_one_l516_516730

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc (-1 : ℝ) 2 then 3 - x ^ 2 else x - 3

theorem solve_fx_eq_one :
  {x : ℝ | f x = 1} = {real.sqrt 2, 4} :=
by
  sorry

end solve_fx_eq_one_l516_516730


namespace average_headcount_11600_l516_516544

theorem average_headcount_11600 : 
  let h02_03 := 11700
  let h03_04 := 11500
  let h04_05 := 11600
  (h02_03 + h03_04 + h04_05) / 3 = 11600 := 
by
  sorry

end average_headcount_11600_l516_516544


namespace joan_total_cost_is_correct_l516_516793

def year1_home_games := 6
def year1_away_games := 3
def year1_home_playoff_games := 1
def year1_away_playoff_games := 1

def year2_home_games := 2
def year2_away_games := 2
def year2_home_playoff_games := 1
def year2_away_playoff_games := 0

def home_game_ticket := 60
def away_game_ticket := 75
def home_playoff_ticket := 120
def away_playoff_ticket := 100

def friend_home_game_ticket := 45
def friend_away_game_ticket := 75

def home_game_transportation := 25
def away_game_transportation := 50

noncomputable def year1_total_cost : ℕ :=
  (year1_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year1_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def year2_total_cost : ℕ :=
  (year2_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year2_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def total_cost : ℕ := year1_total_cost + year2_total_cost

theorem joan_total_cost_is_correct : total_cost = 2645 := by
  sorry

end joan_total_cost_is_correct_l516_516793


namespace cube_root_power_l516_516658

theorem cube_root_power (a : ℝ) (h : a = 8) : (a^(1/3))^12 = 4096 := by
  rw [h]
  have h2 : 8 = 2^3 := rfl
  rw h2
  sorry

end cube_root_power_l516_516658


namespace problem_S_is_three_rays_l516_516034

/-- Define the set S based on the given conditions -/
def S : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | 
    (p.1 = 1 ∧ p.2 <= 7) ∨ 
    (p.1 >= 1 ∧ p.2 = p.1 + 6) ∨ 
    (p.2 = 7 ∧ p.1 <= 1)}

/-- The set S forms three rays intersecting at the point (1,7) -/
theorem problem_S_is_three_rays :
  S = {p : ℝ × ℝ | p = (1, 7) ∨ (p.1 = 1 ∧ p.2 <= 7) ∨ (p.1 >= 1 ∧ p.2 = p.1 + 6) ∨ (p.2 = 7 ∧ p.1 <= 1)} :=
sorry

end problem_S_is_three_rays_l516_516034


namespace computer_off_time_l516_516023

def initial_time := Friday 14 -- 2 p.m. on Friday, using 24-hour format
def duration := 30 -- duration in hours
def end_time := Saturday 20 -- 8 p.m. on Saturday, using 24-hour format

theorem computer_off_time : (initial_time + duration) = end_time :=
by
  sorry

end computer_off_time_l516_516023


namespace no_n_exists_l516_516297

theorem no_n_exists (n : ℕ) : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
by {
  sorry
}

end no_n_exists_l516_516297


namespace delta_5_is_zero_for_all_n_delta_k_nonzero_for_k_lt_5_l516_516677

def seq (n : ℕ) : ℤ := n^4 + n^2

def delta (f : ℕ → ℤ) (k n : ℕ) : ℤ :=
  match k with
  | 0 => f n
  | (k+1) => delta (λ m => delta f k (m + 1) - delta f k m) 1 n

theorem delta_5_is_zero_for_all_n (n : ℕ) : delta seq 5 n = 0 :=
by
  sorry

theorem delta_k_nonzero_for_k_lt_5 (k n : ℕ) (h : k < 5) : delta seq k n ≠ 0 :=
by
  sorry

end delta_5_is_zero_for_all_n_delta_k_nonzero_for_k_lt_5_l516_516677


namespace abs_diff_eq_l516_516835

-- Given conditions
variables (e a b : ℝ)
variable h1 : (a ≠ b)
variable h2 : a^2 + e^3 = 2*e*a + 4
variable h3 : b^2 + e^3 = 2*e*b + 4

-- The theorem statement
theorem abs_diff_eq : |a - b| = 2 * real.sqrt (e^2 - e^3 + 4) :=
sorry

end abs_diff_eq_l516_516835


namespace remainder_div_101_l516_516915

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l516_516915


namespace number_of_outcomes_l516_516140

def outcomes_for_four_throws_each_unique (A B C D : ℕ) : Prop :=
  ∃ numbers : Finset ℕ, numbers.card = 4 ∧ 
  (∀ i ∈ numbers, 1 ≤ i ∧ i ≤ 6) ∧ 
  (∀ i j ∈ numbers, i ≠ j → numbers.card = 4) ∧
  (∀ i ∈ numbers, i ≠ D → numbers.card = 3)

theorem number_of_outcomes 
  (A B C D : ℕ)
  (h1 : A ≠ B) 
  (h2 : A ≠ C)
  (h3 : B ≠ C)
  (h4 : A ≠ D)
  (h5 : B ≠ D)
  (h6 : C ≠ D) 
  : outcomes_for_four_throws_each_unique A B C D → 270 :=
sorry

end number_of_outcomes_l516_516140


namespace derivative_of_even_function_is_odd_l516_516460

variables {R : Type*}

-- Definitions and Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem derivative_of_even_function_is_odd (f g : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, deriv f x = g x) : odd_function g :=
sorry

end derivative_of_even_function_is_odd_l516_516460


namespace shape_is_cone_l516_516316

-- Definitions based on the problem conditions
variables (c : ℝ) -- c is a constant
variable (r θ φ : ℝ)

-- Predicate representing the condition φ = π/2 - c
def spherical_coordinates_condition (φ : ℝ) : Prop := φ = real.pi / 2 - c

-- Lean 4 statement to prove the shape is a cone
theorem shape_is_cone (c : ℝ) (h : spherical_coordinates_condition c φ) : 
  ∃ (r θ : ℝ), cone r θ φ :=
sorry

end shape_is_cone_l516_516316


namespace remainder_9876543210_mod_101_l516_516918

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l516_516918


namespace arithmetic_mean_eqn_l516_516056

theorem arithmetic_mean_eqn : 
  (3/5 + 6/7) / 2 = 51/70 :=
  by sorry

end arithmetic_mean_eqn_l516_516056


namespace antiderivative_constant_example_function_properties_l516_516433

-- Define the set of functions F
def func_set (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 * x) = f x

-- First part: prove that functions in F with antiderivative are constant
theorem antiderivative_constant (f : ℝ → ℝ) (F : ℝ → ℝ) (hF0 : F 0 = 0)
  (hf : func_set f) (hF : ∀ x, F.deriv x = f x) :
  ∀ x : ℝ, f x = f 0 :=
sorry

-- Second part: proving the existence of a function satisfying the integration condition
noncomputable def example_function : ℝ → ℝ :=
λ x, if (∃ k : ℤ, x = 2 ^ k) then 1 else 0

theorem example_function_properties :
  func_set example_function ∧
  (∀ a b : ℝ, ∫ x in a..b, example_function x = 0) ∧
  ∃ x y : ℝ, example_function x ≠ example_function y :=
sorry

end antiderivative_constant_example_function_properties_l516_516433


namespace spent_on_computer_accessories_l516_516078

theorem spent_on_computer_accessories :
  ∀ (x : ℕ), (original : ℕ) (snacks : ℕ) (remaining : ℕ),
  original = 48 →
  snacks = 8 →
  remaining = 4 + original / 2 →
  original - (x + snacks) = remaining →
  x = 12 :=
by
  intros x original snacks remaining
  intro h_original
  intro h_snacks
  intro h_remaining
  intro h_spent
  sorry

end spent_on_computer_accessories_l516_516078


namespace decagon_angle_Q_l516_516478

theorem decagon_angle_Q (n : ℕ) (Q : Type):
  regular_polygon n ∧ n = 10 ∧ extended_sides_meet_at Q →
  angle_measure Q = 72 :=
by
  intros
  sorry

end decagon_angle_Q_l516_516478


namespace frac_ineq_solution_set_l516_516122

theorem frac_ineq_solution_set (x : ℝ) : (x ≠ 3) → (((x - 2) / (3 - x)) ≤ 1) ↔ (x > 3 ∨ x ≤ 5 / 2) :=
by
  intro h
  sorry

end frac_ineq_solution_set_l516_516122


namespace hockey_league_teams_l516_516533

theorem hockey_league_teams (n : ℕ) 
  (H1 : ∀ i j: ℕ, i ≠ j → team_faces i j = 10)
  (H2 : total_games = 1200) : 
  n = 16 := 
by
  sorry

end hockey_league_teams_l516_516533


namespace dice_probability_same_face_l516_516155

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516155


namespace cosine_smaller_angle_l516_516522

theorem cosine_smaller_angle
  (h : ℝ) -- Distance between the parallel planes
  (l1 l2 : ℝ) -- Lengths of segments of two lines
  (α : ℝ) -- Smaller angle
  (ratio_seg : l1 / l2 = 5 / 9) -- Ratio of the segments
  (ratio_angle : α = 2 * α) -- Ratio of the angles
  : real.cos(α) = 0.9 := by
  sorry

end cosine_smaller_angle_l516_516522


namespace cuboid_volume_l516_516940

/-- Given a cuboid with edges 6 cm, 5 cm, and 6 cm, the volume of the cuboid
    is 180 cm³. -/
theorem cuboid_volume (a b c : ℕ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 6) :
  a * b * c = 180 := by
  sorry

end cuboid_volume_l516_516940


namespace product_of_000412_and_9243817_is_closest_to_3600_l516_516212

def product_closest_to (x y value: ℝ) : Prop := (abs (x * y - value) < min (abs (x * y - 350)) (min (abs (x * y - 370)) (min (abs (x * y - 3700)) (abs (x * y - 4000)))))

theorem product_of_000412_and_9243817_is_closest_to_3600 :
  product_closest_to 0.000412 9243817 3600 :=
by
  sorry

end product_of_000412_and_9243817_is_closest_to_3600_l516_516212


namespace area_of_circle_with_given_equation_l516_516289

open Real

theorem area_of_circle_with_given_equation :
  let eq := fun (x y : ℝ) => x^2 + y^2 - 6 * x + 8 * y + 9
  ∃ r : ℝ, (∀ x y : ℝ, eq x y = 16 ↔ (x-3)^2 + (y+4)^2 = r^2) ∧ r = 4 → 
    (π * r^2 = 16 * π) :=
by
  intro eq
  use 4
  split
  { intro r
    split
    { intro h
      sorry
    }
    { intro h
      sorry
    }
  }
  { norm_num }

end area_of_circle_with_given_equation_l516_516289


namespace volume_S_eq_two_l516_516882

def S := {p : ℝ × ℝ × ℝ | |p.1| + |p.2| ≤ 1 ∧ |p.1| + |p.3| ≤ 1 ∧ |p.2| + |p.3| ≤ 1}

theorem volume_S_eq_two : volume S = 2 := sorry

end volume_S_eq_two_l516_516882


namespace problem1_problem2_l516_516365

section

variables {x a : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a / x 

-- Problem (1): Prove that f(x) is even when a = 0
theorem problem1 (x : ℝ) : f x 0 = f (-x) 0 :=
by
  unfold f
  simp

-- Problem (2): Prove that f(x) is an increasing function in [2, +∞) implies a ≤ 16
theorem problem2 (a x : ℝ) (h1 : 2 ≤ x) (h2 : f (x+1) a ≥ f x a) : a ≤ 16 :=
by
  sorry

end

end problem1_problem2_l516_516365


namespace rectangle_length_to_width_ratio_l516_516674

variables (s : ℝ)

-- Given conditions
def small_square_side := s
def large_square_side := 3 * s
def rectangle_length := large_square_side
def rectangle_width := large_square_side - 2 * small_square_side

-- Theorem to prove the ratio of the length to the width of the rectangle
theorem rectangle_length_to_width_ratio : 
  ∃ (r : ℝ), r = rectangle_length s / rectangle_width s ∧ r = 3 := 
by
  sorry

end rectangle_length_to_width_ratio_l516_516674


namespace percentage_increase_in_expenses_l516_516974

theorem percentage_increase_in_expenses 
  (monthly_salary rs20000 : ℝ) 
  (original_savings_fraction : ℝ) 
  (current_savings : ℝ) 
  (original_savings : monthly_salary * original_savings_fraction = 2000)
  (current_savings_def : current_savings = 200)
  (salary_def : monthly_salary = 20000) :
  let increased_expense := (original_savings - current_savings)
  let original_expenses := (monthly_salary - original_savings)
  let percentage_increase := (increased_expense / original_expenses) * 100
  percentage_increase = 10 := 
by
  sorry

end percentage_increase_in_expenses_l516_516974


namespace norb_age_is_correct_l516_516091

open Nat

/-
Problem: Norb's age is to be identified among the given guesses such that:
1. At least 60% of the guesses are less than Norb's age.
2. Two of the guesses differ from Norb's age by exactly 1.
3. Norb's age is a prime number.
Given guesses are: [26, 31, 33, 35, 39, 41, 43, 46, 49, 53, 55, 57].

The solution identified that Norb's age must be 59.
-/

def guesses : List ℕ := [26, 31, 33, 35, 39, 41, 43, 46, 49, 53, 55, 57]

def norb_age : ℕ := 59

theorem norb_age_is_correct :
  (norb_age ∈ [47, 59]) ∧
  (∃ (a b : ℕ), a ∈ guesses ∧ b ∈ guesses ∧ a = norb_age - 1 ∧ b = norb_age + 1) ∧
  (norb_age ∈ primes) ∧
  (∃ n, n ≥ 0.6 * guesses.length ∧ n = guesses.filter (λ x, x < norb_age).length) :=
by
  sorry

end norb_age_is_correct_l516_516091


namespace hyperbola_focus_perpendicular_l516_516351

theorem hyperbola_focus_perpendicular 
  (F1 F2 P : ℝ × ℝ) (m : ℝ) 
  (on_hyperbola : P.1^2 / 9 - P.2^2 / m = 1)
  (perpendicular : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0)
  (directrix : F1 = (-4, 0) ∨ F2 = (-4, 0)) :
  (real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 14 :=
by
  sorry

end hyperbola_focus_perpendicular_l516_516351


namespace remainder_of_large_number_l516_516911

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l516_516911


namespace erin_serves_all_soup_in_15_minutes_l516_516649

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end erin_serves_all_soup_in_15_minutes_l516_516649


namespace divisibility_by_100_l516_516149

theorem divisibility_by_100 (n : ℕ) (k : ℕ) (h : n = 5 * k + 2) :
    100 ∣ (5^n + 12*n^2 + 12*n + 3) :=
sorry

end divisibility_by_100_l516_516149


namespace grace_is_14_l516_516741

def GraceAge (G F C E D : ℕ) : Prop :=
  G = F - 6 ∧ F = C + 2 ∧ E = C + 3 ∧ D = E - 4 ∧ D = 17

theorem grace_is_14 (G F C E D : ℕ) (h : GraceAge G F C E D) : G = 14 :=
by sorry

end grace_is_14_l516_516741


namespace lateral_surface_area_of_cone_l516_516720

theorem lateral_surface_area_of_cone (r h : ℝ) (r_is_4 : r = 4) (h_is_3 : h = 3) :
  ∃ A : ℝ, A = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_cone_l516_516720


namespace triangle_A_and_Area_l516_516762

theorem triangle_A_and_Area :
  ∀ (a b c A B C : ℝ), 
  (b - (1 / 2) * c = a * Real.cos C) 
  → (4 * (b + c) = 3 * b * c) 
  → (a = 2 * Real.sqrt 3)
  → (A = 60) ∧ (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) :=
by
  intros a b c A B C h1 h2 h3
  sorry

end triangle_A_and_Area_l516_516762


namespace probability_winning_l516_516874

-- Define the probability of losing
def P_lose : ℚ := 5 / 8

-- Define the total probability constraint
theorem probability_winning : P_lose = 5 / 8 → (1 - P_lose) = 3 / 8 := 
by
  intro h
  rw h
  norm_num
  sorry

end probability_winning_l516_516874


namespace negative_integers_abs_le_4_l516_516991

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l516_516991


namespace part1_part2_l516_516571
open Real

-- Part 1
theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ∧
  (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ≤ 8 := 
sorry

-- Part 2
theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ β > 0, β = 4 ∧ sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^2 / β :=
sorry

end part1_part2_l516_516571


namespace TP_bisects_angle_ATB_l516_516141

-- Define the given problem condition that two circles are tangent internally with a chord
variables (C1 C2 : Circle) (T A B P : Point)
(h1 : C2 ⊂ C1)     -- C2 is internally tangent to C1 at point T
(h2 : Tangent C1 C2 T)      -- Tangency condition of the circles
(h3 : Chord C1 A B P)        -- AB is a chord of C1
(h4 : AB ⊆ Tangent C2 P)     -- AB is tangent to C2 at P

-- Problem Statement: Prove TP bisects ∠ATB
theorem TP_bisects_angle_ATB :
  bisects (line T P) (angle A T B) :=
sorry

end TP_bisects_angle_ATB_l516_516141


namespace horner_correct_evaluation_l516_516539

def f (x : ℝ) : ℝ :=
  1 + 2 * x + 3 * x^2 + 4 * x^3 + 5 * x^4 + 6 * x^5

noncomputable def eval_horner (x : ℝ) : ℝ :=
  (((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 1

theorem horner_correct_evaluation : eval_horner 2 = f 2 :=
  sorry

end horner_correct_evaluation_l516_516539


namespace crickets_total_l516_516555

theorem crickets_total (initial_crickets : Real) (found_crickets : Real) : 
  initial_crickets = 7 ∧ found_crickets = 11 → initial_crickets + found_crickets = 18 := by
  intros h
  cases h with hi hf
  rw [hi, hf]
  norm_num

end crickets_total_l516_516555


namespace side_b_of_triangle_l516_516787

theorem side_b_of_triangle (A B : ℝ) (a b : ℝ) (hA : A = 30) (hB : B = 45) (ha : a = real.sqrt 2) :
  b = 2 :=
by
-- Including conditions for sine values for angles
have hsinA : real.sin (real.to_radians A) = 1 / 2, by sorry,
have hsinB : real.sin (real.to_radians B) = real.sqrt 2 / 2, by sorry,
-- Applying Law of Sines
have h : a / (real.sin (real.to_radians A)) = b / (real.sin (real.to_radians B)), by sorry,
-- Substituting given values and solving for b
sorry

end side_b_of_triangle_l516_516787


namespace min_value_alpha_beta_l516_516739

noncomputable def alpha_beta_vectors (α β : ℝ × ℝ) : Prop :=
  let vector_magnitude (v : ℝ × ℝ) := (v.1^2 + v.2^2)^(1/2)
  let dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
  let angle_in_radians (θ : ℝ) := θ * (Real.pi / 180)
  let angle_between (u v : ℝ × ℝ) := 
    let cos_theta := dot_product u v / (vector_magnitude u * vector_magnitude v)
    real.arccos cos_theta
  let α_plus_β := (α.1 + β.1, α.2 + β.2)
  let α_minus_2β := (α.1 - 2 * β.1, α.2 - 2 * β.2)
  in 
    vector_magnitude ((2 * β.1 - α.1, 2 * β.2 - α.2)) = sqrt 3 ∧
    angle_between α_plus_β α_minus_2β = angle_in_radians 150 ∧
    @real_min (fun t : ℝ => vector_magnitude ((t * α_plus_β.1 - (3/2) * β.1, t * α_plus_β.2 - (3/2) * β.2))) = sqrt 3 / 4

-- The statement of the theorem confirming the conditions and the minimum value.
theorem min_value_alpha_beta (α β : ℝ × ℝ) (h : alpha_beta_vectors α β) : 
  @real_min (fun t : ℝ => vector_magnitude ((t * (α.1 + β.1 - (3/2) * β.1), t * (α.2 + β.2 - (3/2) * β.2))) = sqrt 3 / 4 :=
sorry

end min_value_alpha_beta_l516_516739


namespace part_1_correct_part_2_correct_l516_516678

noncomputable theory

def ya_interval (T : ℝ) (m n : ℤ) : Prop := m < T ∧ T < n

axiom ya_interval_neg_sqrt_7 : ya_interval (-Real.sqrt 7) (-3) (-2)

theorem part_1_correct : ya_interval (-Real.sqrt 7) (-3) (-2) := 
ya_interval_neg_sqrt_7

variables (m n c x y : ℤ)

axiom condition1 : 0 < m + Real.sqrt n ∧ m + Real.sqrt n < 12
axiom condition2 : x = m ∧ y = Real.sqrt n
axiom condition3 : ∃ (m n x y : ℤ), (0 < m + Real.sqrt n ∧ m + Real.sqrt n < 12) ∧ 
                    (x = m ∧ y = Real.sqrt n) ∧ (c = m * x - n * y)

theorem part_2_correct : (c = 1 ∨ c = 37) :=
sorry

end part_1_correct_part_2_correct_l516_516678


namespace surface_area_is_36_l516_516893

def unit_cube : Type := ℕ  -- using ℕ to represent unit cubes

structure Solid :=
(base_layer : ℕ)
(second_layer : ℕ)

axiom base_layer_cubes : unit_cube := 8
axiom second_layer_cubes : unit_cube := 4

def total_cubes (s : Solid) : Prop :=
  s.base_layer = base_layer_cubes ∧ s.second_layer = second_layer_cubes

def visible_faces_base_layer (s : Solid) : ℕ := 8 + 4 + 12
def visible_faces_second_layer (s : Solid) : ℕ := 4 + 8

def total_surface_area (s : Solid) : ℕ :=
  visible_faces_base_layer s + visible_faces_second_layer s

theorem surface_area_is_36 : ∀ s: Solid, total_cubes s → total_surface_area s = 36 :=
by
  intros
  unfold total_surface_area
  unfold visible_faces_base_layer
  unfold visible_faces_second_layer
  unfold total_cubes
  sorry

end surface_area_is_36_l516_516893


namespace find_angle_B_l516_516406

variable {A B C : Type} [EuclideanSpace B]
variable {a b c : ℝ} (h : b^2 = a^2 + a * c + c^2)

theorem find_angle_B (A B C : B) (h : b^2 = a^2 + a * c + c^2) : ∠A B C = 120 :=
by
  sorry

end find_angle_B_l516_516406


namespace valid_paths_count_l516_516576

open Function

def is_valid_path (path : List (Int × Int)) : Prop :=
  (path.head = (-3, -3)) ∧
  (path.last = (3, 3)) ∧
  (∀ (p₁ p₂ : Int × Int), (p₁, p₂) ∈ (path.zip path.tail) → 
    ((p₁.1 + 1 = p₂.1 ∧ p₁.2 = p₂.2) ∨ (p₁.1 = p₂.1 ∧ p₁.2 + 1 = p₂.2))) ∧
  (∀ (p : Int × Int), p ∈ path → ¬(-1 ≤ p.1 ∧ p.1 ≤ 1 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1)) ∧
  (path.length = 13)

def count_valid_paths : Nat :=
  List.length (List.filter is_valid_path
    (List.replicate (13) (List.replicate 13 (0, 0)))) -- just a placeholder for actual path generation

theorem valid_paths_count :
  count_valid_paths = 238 :=
sorry

end valid_paths_count_l516_516576


namespace derivative_at_zero_of_even_function_l516_516355

theorem derivative_at_zero_of_even_function 
  (f : ℝ → ℝ) 
  (hf_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x) 
  (hf_deriv : ∀ x : ℝ, ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h ∈ set.Icc (-(h : ℝ)) (h : ℝ), abs h < δ → abs (f (x + h) - f x - l * h) / abs h < ε) 
  (hf_even : ∀ x : ℝ, f(-x) = f(x))
  : f'(0) = 0 :=
by 
  sorry

end derivative_at_zero_of_even_function_l516_516355


namespace num_pupils_is_40_l516_516939

-- given conditions
def incorrect_mark : ℕ := 83
def correct_mark : ℕ := 63
def mark_difference : ℕ := incorrect_mark - correct_mark
def avg_increase : ℚ := 1 / 2

-- the main problem statement to prove
theorem num_pupils_is_40 (n : ℕ) (h : (mark_difference : ℚ) / n = avg_increase) : n = 40 := 
sorry

end num_pupils_is_40_l516_516939


namespace evaluate_cube_root_power_l516_516653

theorem evaluate_cube_root_power (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^(3 : ℝ)) : (cbrt a)^12 = b^12 :=
by
  sorry

example : evaluate_cube_root_power 8 2 4096 (by rfl)

end evaluate_cube_root_power_l516_516653


namespace johns_fraction_l516_516021

noncomputable def fraction_spent_arcade (weekly_allowance : ℝ) (remaining_candy_store : ℝ) := 
  let f : ℝ := (3/5) in 
  let arcade_spend := f * weekly_allowance in 
  let remaining_post_arcade := weekly_allowance - arcade_spend in 
  let toy_store_spend := (1/3) * remaining_post_arcade in 
  let remaining_post_toy := remaining_post_arcade - toy_store_spend in
  remaining_post_toy = remaining_candy_store

theorem johns_fraction {weekly_allowance remaining_candy_store : ℝ} (h1 : weekly_allowance = 4.80) (h2 : remaining_candy_store = 1.28) : 
  fraction_spent_arcade weekly_allowance remaining_candy_store := 
by {
  rw [h1, h2],
  sorry
}

end johns_fraction_l516_516021


namespace specific_heat_capacity_l516_516050

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end specific_heat_capacity_l516_516050


namespace at_least_one_person_between_l516_516888

theorem at_least_one_person_between (six_people : Finset (Fin 6)) :
  let A := 0
  let B := 1
  A ∈ six_people ∧ B ∈ six_people → 
  ∃ arrangements : Finset (list (Fin 6)), arrangements.card = 720 :=
by
  sorry

end at_least_one_person_between_l516_516888


namespace class_strength_l516_516095

/-- The average age of an adult class is 40 years.
    12 new students with an average age of 32 years join the class,
    therefore decreasing the average by 4 years.
    What was the original strength of the class? -/
theorem class_strength (x : ℕ) (h1 : ∃ (x : ℕ), ∀ (y : ℕ), y ≠ x → y = 40) 
                       (h2 : 12 ≥ 0) (h3 : 32 ≥ 0) (h4 : (x + 12) * 36 = 40 * x + 12 * 32) : 
  x = 12 := 
sorry

end class_strength_l516_516095


namespace min_value_of_expression_l516_516334

theorem min_value_of_expression (a b : ℝ) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  42 + b^2 + 1 / (a * b) ≥ 17 / 2 := 
sorry

end min_value_of_expression_l516_516334


namespace minimum_t_for_fox_escape_l516_516899

def fox_escape_condition (t : ℤ) : Prop :=
  (1.23 : ℚ) * (1 - (t / 100 : ℚ)) < 1

theorem minimum_t_for_fox_escape : ∃ (t : ℤ), (t = 19) ∧ fox_escape_condition t :=
by {
  use 19,
  split,
  { rfl },
  { have h : (19 : ℚ) / 100 = 0.19 := by norm_num,
    rw [fox_escape_condition],
    norm_num,
    rw [of_nat_eq_coe'],
    exact lt_trans
      (by linarith) -- 1.23 * (1 - 0.19) = 1.23 * 0.81 = 0.9963 < 1
      (by linarith) -- checking 1.23 * (1 - t / 100) < 1 is less
  },
  sorry -- to be completed
}

end minimum_t_for_fox_escape_l516_516899


namespace algebraic_expression_value_l516_516359

namespace MathProof

variables {α β : ℝ} 

-- Given conditions
def is_root (a : ℝ) : Prop := a^2 - a - 1 = 0
def roots_of_quadratic (α β : ℝ) : Prop := is_root α ∧ is_root β

-- The proof problem statement
theorem algebraic_expression_value (h : roots_of_quadratic α β) : α^2 + α * (β^2 - 2) = 0 := 
by sorry

end MathProof

end algebraic_expression_value_l516_516359


namespace segment_intersection_l516_516057

theorem segment_intersection (n : ℕ) (segments : Fin (2 * n + 1) → set ℝ) 
  (h_intersects : ∀ i, (segments i).finite ∧ (|{j | j ≠ i ∧ segments i ∩ segments j ≠ ∅}| ≥ n)) 
  : ∃ k, ∀ l, k ≠ l → segments k ∩ segments l ≠ ∅ :=
sorry

end segment_intersection_l516_516057


namespace minimum_beard_hairs_l516_516502

theorem minimum_beard_hairs (A : Finset ℝ) (hA : ∀ a ∈ A, a > 0) (h_card : A.card = 100) :
  ∃ B : Finset ℝ, B.card = 101 ∧ (∀ s ⊆ B, s.sum ∈ A ∨ (∃ t ⊆ A, s.sum = t.sum)) := sorry

end minimum_beard_hairs_l516_516502


namespace polynomial_q_l516_516845

theorem polynomial_q (q : Polynomial ℝ) :
  q + Polynomial.Coeff [0, 0, 10, 0, 5, 0, 2] = Polynomial.Coeff [3, 5, 40, 30, 9, 0, 0] →
  q = Polynomial.Coeff [3, 5, 30, 30, 4, 0, -2] :=
begin
  sorry
end

end polynomial_q_l516_516845


namespace exists_nat_solution_for_A_415_l516_516680

theorem exists_nat_solution_for_A_415 : ∃ (m n : ℕ), 3 * m^2 * n = n^3 + 415 := by
  sorry

end exists_nat_solution_for_A_415_l516_516680


namespace weight_of_daughter_l516_516865

theorem weight_of_daughter 
  (M D C : ℝ)
  (h1 : M + D + C = 120)
  (h2 : D + C = 60)
  (h3 : C = (1 / 5) * M)
  : D = 48 :=
by
  sorry

end weight_of_daughter_l516_516865


namespace loss_percentage_proof_l516_516247

-- Define the cost price and selling price
def CP : ℝ := 1900
def SP : ℝ := 1558

-- Define the loss
def Loss : ℝ := CP - SP

-- Define the loss percentage
def LossPercentage : ℝ := (Loss / CP) * 100

-- The theorem to prove:
theorem loss_percentage_proof : LossPercentage = 18 := by
  sorry

end loss_percentage_proof_l516_516247


namespace correct_division_result_l516_516548

theorem correct_division_result : 
  ∀ (a b : ℕ),
  (1722 / (10 * b + a) = 42) →
  (10 * a + b = 14) →
  1722 / 14 = 123 :=
by
  intros a b h1 h2
  sorry

end correct_division_result_l516_516548


namespace same_number_on_four_dice_l516_516189

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516189


namespace fraction_of_three_quarters_is_two_ninths_l516_516901

theorem fraction_of_three_quarters_is_two_ninths :
  (∃ x : ℚ, x * (3/4) = (2/9) ∧ x = (8/27)) ∧ ((8 / 27 : ℚ) ≈ (29.6 / 100)) :=
by {
  sorry
}

end fraction_of_three_quarters_is_two_ninths_l516_516901


namespace triangle_area_is_32_l516_516150

def point : Type := ℝ × ℝ

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b

def y_eq_2x := {p : point | p.2 = 2 * p.1}
def y_eq_neg2x := {p : point | p.2 = -2 * p.1}
def y_eq_8 := {p : point | p.2 = 8}

def intersect (l1 l2 : point -> Prop) : set point :=
  {p : point | l1 p ∧ l2 p}

def vertices : set point :=
  (intersect y_eq_2x y_eq_8) ∪
  (intersect y_eq_neg2x y_eq_8) ∪
  ({p : point | p = (0, 0)})

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def triangle_area (a b c : point) : ℝ :=
  let base := distance a b in
  let height := abs (c.2 - a.2) in
  0.5 * base * height

theorem triangle_area_is_32 :
  triangle_area (4, 8) (-4, 8) (0, 0) = 32 := 
sorry

end triangle_area_is_32_l516_516150


namespace inequality_proof_l516_516698

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : 0 < c)
  : a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c :=
sorry

end inequality_proof_l516_516698


namespace circle_area_from_triangle_l516_516354

noncomputable def area_of_circle (π : ℝ) (d : ℝ) := π * (d / 2) ^ 2

theorem circle_area_from_triangle
  (a b : ℝ) (h1 : (1 / 2) * a * b = 5)
  (h2 : ∀ d, d = real.sqrt (a^2 + b^2))
  (π : ℝ) (hπ : π = 3.14) :
  ∃ d, area_of_circle π d = 25.12 :=
by
  have h3 : a * b = 10, from (mul_eq_of_eq_div (eq_div_of_mul_eq h1)).symm,
  have h4 : ∃ d, d = real.sqrt (a^2 + b^2), from ⟨_, h2⟩,
  sorry

end circle_area_from_triangle_l516_516354


namespace volunteer_allocation_scheme_l516_516617

-- Define the number of volunteers and service points
def volunteers : ℕ := 4
def service_points : ℕ := 3

-- Define the problem
theorem volunteer_allocation_scheme : 
  ∀ (volunteers : ℕ) (service_points : ℕ),
  volunteers = 4 → 
  service_points = 3 → 
  (∃ f : Fin volunteers → Fin service_points, 
     ∀ s : Fin service_points, ∃ v : Fin volunteers, f v = s) →
  (number_of_allocation_schemes volunteers service_points = 36) :=
sorry

end volunteer_allocation_scheme_l516_516617


namespace min_speed_A_l516_516148

theorem min_speed_A (V_B V_C V_A : ℕ) (d_AB d_AC wind extra_speed : ℕ) :
  V_B = 50 →
  V_C = 70 →
  d_AB = 40 →
  d_AC = 280 →
  wind = 5 →
  V_A > ((d_AB * (V_A + wind + extra_speed)) / (d_AC - d_AB) - wind) :=
sorry

end min_speed_A_l516_516148


namespace cannot_form_larger_square_l516_516554

theorem cannot_form_larger_square:
  let pieces := 2 * 4 + 3 * 3 + 2 * 2 + 3 * 1 in
  pieces = 24 → ¬∃ (n : ℕ), n * n = 24 :=
by
  intros pieces_eq h
  cases h with n hn
  have : ∀ k : ℕ, k * k ≠ 24 := 
    by sorry -- since 24 is not a perfect square, there is no integer k such that k * k = 24
  contradiction

end cannot_form_larger_square_l516_516554


namespace bob_cleaning_time_l516_516263

theorem bob_cleaning_time (alice_time : ℝ) (bob_fraction : ℝ) (bob_time : ℝ) 
  (h1 : alice_time = 15) (h2 : bob_fraction = 3 / 4) (h3 : bob_time = bob_fraction * alice_time) : 
  bob_time = 11.25 := by
  sorry

end bob_cleaning_time_l516_516263


namespace same_number_probability_four_dice_l516_516173

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516173


namespace relationship_m_p_n_l516_516036

variable (a : ℝ) (h_a : a > 1)

def m := Real.log (a^2 + 1) / Real.log a
def n := Real.log (a - 1) / Real.log a
def p := Real.log (2 * a) / Real.log a

theorem relationship_m_p_n : m a = Real.log (a^2 + 1) / Real.log a ∧ 
                             p a = Real.log (2 * a) / Real.log a ∧ 
                             n a = Real.log (a - 1) / Real.log a ∧ 
                             m a > p a ∧ p a > n a := by
  sorry

end relationship_m_p_n_l516_516036


namespace largest_repeated_basket_count_l516_516966

noncomputable def max_repeated_baskets 
  (total_baskets : ℕ) 
  (min_oranges_per_basket : ℕ) 
  (max_oranges_per_basket : ℕ) : ℕ := 
  let num_possible_counts := max_oranges_per_basket - min_oranges_per_basket + 1 in
  total_baskets / num_possible_counts + (if total_baskets % num_possible_counts = 0 then 0 else 1)

theorem largest_repeated_basket_count 
  (total_baskets = 150) 
  (min_oranges_per_basket = 80) 
  (max_oranges_per_basket = 130) : 
  max_repeated_baskets total_baskets min_oranges_per_basket max_oranges_per_basket = 3 := 
sorry

end largest_repeated_basket_count_l516_516966


namespace inverse_of_matrix_l516_516307

theorem inverse_of_matrix :
  let A := Matrix.of ![![5, -3], ![4, -2]]
  let A_inv := Matrix.of ![![-1, 1.5], ![-2, 2.5]]
  (A * A_inv = 1) := sorry

end inverse_of_matrix_l516_516307


namespace sin_arccos_curve_l516_516932

theorem sin_arccos_curve (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
  let y := Real.sin (Real.arccos x)
  in x^2 + y^2 = 1 ∧ y ≥ 0 :=
by
  let y := Real.sin (Real.arccos x)
  split
  · exact sorry -- prove x^2 + y^2 = 1
  · exact sorry -- prove y ≥ 0

end sin_arccos_curve_l516_516932


namespace volume_of_cuboid_l516_516856

theorem volume_of_cuboid (a b c : ℕ) (h_a : a = 2) (h_b : b = 5) (h_c : c = 8) : 
  a * b * c = 80 := 
by 
  sorry

end volume_of_cuboid_l516_516856


namespace a_finishes_remaining_work_in_3_over_4_day_l516_516556

theorem a_finishes_remaining_work_in_3_over_4_day
  (A B C : Type)
  (A_work_rate : ℚ)
  (B_work_rate : ℚ)
  (C_work_rate : ℚ)
  (B_days_worked : ℕ)
  (C_days_worked : ℕ) :
  (A_work_rate = 1 / 9) →
  (B_work_rate = 1 / 15) →
  (C_work_rate = 1 / 20) →
  (B_days_worked = 10) →
  (C_days_worked = 5) →
  let B_work := B_days_worked * B_work_rate in
  let C_work := C_days_worked * C_work_rate in
  let total_work := B_work + C_work in
  let remaining_work := 1 - total_work in
  let days_for_A := remaining_work / A_work_rate in
  days_for_A = 3 / 4 :=
by
  intros hA hB hC hBd hCd
  let B_work := B_days_worked * B_work_rate
  let C_work := C_days_worked * C_work_rate
  let total_work := B_work + C_work
  let remaining_work := 1 - total_work
  let days_for_A := remaining_work / A_work_rate
  sorry

end a_finishes_remaining_work_in_3_over_4_day_l516_516556


namespace value_of_ka_l516_516719

noncomputable def k_a_sum (k : ℝ) (a : ℝ) : ℝ :=
k + a

theorem value_of_ka (k a : ℝ)
  (h₁ : ∀ x : ℝ, f x = k * x^a)
  (h₂ : f (1/2) = 1/4) :
  k + a = 3 :=
sorry

end value_of_ka_l516_516719


namespace sphere_surface_area_l516_516820

theorem sphere_surface_area (P: ℝ) (r₁ r₂: ℝ) (α β: ℝ) (O: σ) (A: Type) (has_sphere_O_radii: ∀ r₁ r₂, section_radius α l O r₁ ∧ section_radius β l O r₂) (dihedral_angle_abc : ∀ α β, angle_between α β = 150) (line_l_sphere_O: ∀ l O, common_point l O = P) :
  surface_area sphere O = 112 * π :=
by
sorry

end sphere_surface_area_l516_516820


namespace Fr_zero_for_all_r_l516_516027

-- Define our variables
variables {x y z A B C : ℝ}
-- Define F_r
def F (r : ℕ) : ℝ := x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)
-- Define the conditions
axiom integral_multiple_of_pi : ∃ k : ℤ, A + B + C = k * Real.pi
axiom F1_zero : F 1 = 0
axiom F2_zero : F 2 = 0

-- Prove the main statement
theorem Fr_zero_for_all_r (r : ℕ) (hr : r > 0) : F r = 0 :=
sorry

end Fr_zero_for_all_r_l516_516027


namespace chocolates_per_small_box_l516_516591

/-- A large box contains 19 small boxes and each small box contains a certain number of chocolate bars.
There are 475 chocolate bars in the large box. --/
def number_of_chocolate_bars_per_small_box : Prop :=
  ∃ x : ℕ, 475 = 19 * x ∧ x = 25

theorem chocolates_per_small_box : number_of_chocolate_bars_per_small_box :=
by
  sorry -- proof is skipped

end chocolates_per_small_box_l516_516591


namespace can_fit_two_cross_shaped_pastries_l516_516583

-- Define the structure of a cross-shaped pastry made of five 1x1 squares
structure CrossShapedPastry :=
  (squares: Finset (ℕ × ℕ)) -- Set of coordinates of the 5 squares

-- Define the box with area 16 square units
def box := { coords : Finset (ℕ × ℕ) // coords.size = 16 }

-- The question is to prove that two cross-shaped pastries can fit into the box
theorem can_fit_two_cross_shaped_pastries
  (p1 p2 : CrossShapedPastry)
  (b : box)
  : ∃ (placement : Finset ((ℕ × ℕ) × (ℕ × ℕ))), placement.size = p1.squares.size + p2.squares.size ∧ placement ⊆ b.val :=
sorry -- Proof omitted

end can_fit_two_cross_shaped_pastries_l516_516583


namespace binom_20_10_l516_516707

theorem binom_20_10 : 
  ( ∀ (n k : ℕ), n ≥ k → k ≥ 0 → ∑ i in finset.range(k + 1), nat.choose n i = 2 ^ n ) →
  (nat.choose 18 9 = 48620) →
  (nat.choose 18 10 = 43758) →
  (nat.choose 19 9 = 92378) →
  nat.choose 20 10 = 184756 :=
by 
  intros _ h1 h2 h3
  sorry

end binom_20_10_l516_516707


namespace value_of_a_l516_516361

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then real.log x / real.log 2 else x^2

theorem value_of_a (a : ℝ) (h : f 4 = 2 * f a) : a = -1 ∨ a = 2 :=
  sorry

end value_of_a_l516_516361


namespace inequality_empty_solution_range_l516_516125

/-
Proof problem:
Prove that for the inequality |x-3| + |x-a| < 1 to have no solutions, the range of a must be (-∞, 2] ∪ [4, +∞).
-/

theorem inequality_empty_solution_range (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - a| < 1 → false) ↔ a ∈ set.Iic 2 ∪ set.Ici 4 := sorry

end inequality_empty_solution_range_l516_516125


namespace triangle_hypotenuse_sine_ratio_l516_516438

theorem triangle_hypotenuse_sine_ratio (n : ℕ) (h_n_pos : 0 < n)
  (A B C : Point) (h_right : right_triangle A B C)
  (D : Fin (2 * n) → Point)
  (α : Fin (2 * n + 1) → ℝ) 
  (h_angles : ∀ i : Fin (2 * n + 1), ∠ (D i.succ.pred) A (D i.succ) = α i)
  (h_D_eq : ∀ i : Fin (2 * n), dist (D i.succ.pred) (D i.succ) = dist (D i.succ) (D i.succ.succ))
  (h_boundary_b : D 0 = B) (h_boundary_c : D (2 * n) = C)
  : (∏ i in Finset.range (n + 1), Real.sin (α ⟨2 * i, sorry⟩)) / 
    (∏ i in Finset.range n, Real.sin (α ⟨2 * i + 1, sorry⟩)) = 1 / (2 * n + 1) := 
sorry

end triangle_hypotenuse_sine_ratio_l516_516438


namespace geometric_sequence_problem_l516_516400

noncomputable def geometric_sequence_solution (a_1 a_2 a_3 a_4 a_5 q : ℝ) : Prop :=
  (a_5 - a_1 = 15) ∧
  (a_4 - a_2 = 6) ∧
  (a_3 = 4 ∧ q = 2 ∨ a_3 = -4 ∧ q = 1/2)

theorem geometric_sequence_problem :
  ∃ a_1 a_2 a_3 a_4 a_5 q : ℝ, geometric_sequence_solution a_1 a_2 a_3 a_4 a_5 q :=
by
  sorry

end geometric_sequence_problem_l516_516400


namespace dice_probability_same_face_l516_516153

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516153


namespace exists_m_le_3n_not_divide_succ_l516_516691

noncomputable def integer_coefficient_polynomial_of_degree (n : ℕ) : Type :=
{ f : ℤ → ℤ // ∃ (c : ℤ) (coeffs : Fin n.succ → ℤ),
    c ≠ 0 ∧ f = λ x, (List.range n.succ).sum (λ k, coeffs ⟨k, (by linarith)⟩ * x^k) }

def has_no_integer_roots (f : ℤ → ℤ) : Prop :=
∀ (m : ℤ), f m ≠ 0

theorem exists_m_le_3n_not_divide_succ
  (n : ℕ) (hn : 0 < n)
  (f : integer_coefficient_polynomial_of_degree n)
  (hf : has_no_integer_roots f.val) :
  ∃ (m : ℕ), m ≤ 3 * n ∧ ¬(f.val m ∣ f.val (m + 1)) :=
sorry

end exists_m_le_3n_not_divide_succ_l516_516691


namespace math_problem_l516_516747

theorem math_problem (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1)
  (h4 : m = -1 ∨ m = 3) : (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ (a + b) * (c / d) + m * c * d + (b / a) = -2 :=
by
  sorry

end math_problem_l516_516747


namespace half_taking_function_range_l516_516108

noncomputable def isMonotonic {α : Type*} [Preorder α] (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x ≤ y → f x ≤ f y

def halfTakingFunction {α : Type*} [Preorder α] [Nontrivial α] (f : α → α) (a b : α) : Prop :=
  isMonotonic f ∧ 
  (∀ x ∈ set.Icc a b, f x ∈ set.Icc (a/2) (b/2))

theorem half_taking_function_range (c : ℝ) (t : ℝ) (a b : ℝ)
  (hc : c > 0) (hc_ne_one : c ≠ 1)
  (hf : halfTakingFunction (λ x, log c (c^x + t)) a b) :
  0 < t ∧ t < 1 / 4 :=
begin
  sorry
end

end half_taking_function_range_l516_516108


namespace sum_of_19_consecutive_integers_is_1007_l516_516573

theorem sum_of_19_consecutive_integers_is_1007 : 
  ∃ (m : ℤ), (∑ i in finset.range 19, (m - 9 + i)) = 1007 :=
by
  sorry

end sum_of_19_consecutive_integers_is_1007_l516_516573


namespace number_of_workers_in_original_scenario_l516_516014

-- Definitions based on the given conditions
def original_days := 70
def alternative_days := 42
def alternative_workers := 50

-- The statement we want to prove
theorem number_of_workers_in_original_scenario : 
  (∃ (W : ℕ), W * original_days = alternative_workers * alternative_days) → ∃ (W : ℕ), W = 30 :=
by
  sorry

end number_of_workers_in_original_scenario_l516_516014


namespace average_speed_ratio_l516_516558

theorem average_speed_ratio (t_E t_F : ℝ) (d_B d_C : ℝ) (htE : t_E = 3) (htF : t_F = 4) (hdB : d_B = 450) (hdC : d_C = 300) :
  (d_B / t_E) / (d_C / t_F) = 2 :=
by
  sorry

end average_speed_ratio_l516_516558


namespace combined_selling_price_l516_516602

theorem combined_selling_price 
  (cost_price1 cost_price2 cost_price3 : ℚ)
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℚ)
  (h1 : cost_price1 = 1200) (h2 : profit_percentage1 = 0.4)
  (h3 : cost_price2 = 800)  (h4 : profit_percentage2 = 0.3)
  (h5 : cost_price3 = 600)  (h6 : profit_percentage3 = 0.5) : 
  cost_price1 * (1 + profit_percentage1) +
  cost_price2 * (1 + profit_percentage2) +
  cost_price3 * (1 + profit_percentage3) = 3620 := by 
  sorry

end combined_selling_price_l516_516602


namespace t_perimeter_difference_l516_516025

def t (n : ℕ) : ℕ := sorry -- definition placeholder

theorem t_perimeter_difference (n : ℕ) (h : n ≥ 3) :
  t (2 * n - 1) - t (2 * n) = (nat.floor (6 / n) : ℕ) ∨
  t (2 * n - 1) - t (2 * n) = (nat.floor (6 / n) : ℕ) + 1 := sorry

end t_perimeter_difference_l516_516025


namespace serving_time_correct_l516_516648
noncomputable theory

def ounces_per_bowl := 10
def bowls_per_minute := 5
def gallons_of_soup := 6
def ounces_per_gallon := 128

def total_ounces := gallons_of_soup * ounces_per_gallon
def serving_rate := ounces_per_bowl * bowls_per_minute

def serving_time := total_ounces / serving_rate

def rounded_serving_time := Int.floor (serving_time + 0.5)

theorem serving_time_correct : rounded_serving_time = 15 := by
  sorry

end serving_time_correct_l516_516648


namespace inequality_l516_516440

variable (n : ℕ) (a x : fin n → ℝ)

# Conditions
def positive (v : fin n → ℝ) := ∀ i, 0 < v i
def sum_to_one (v : fin n → ℝ) := finset.univ.sum v = 1

# Problem statement
theorem inequality (ha : positive a) (hx : positive x) (ha_sum : sum_to_one a) (hx_sum : sum_to_one x) :
  2 * (finset.univ.subset pairs_univ).sum (λ ⟨i, j, hij⟩, x i * x j) ≤ (n - 2) / (n - 1) + finset.univ.sum (λ i, (a i * (x i) ^ 2) / (1 - (a i))) := 
sorry

# Helper Definitions
def pairs_univ := (finset.univ).off_diag_filter finset.univ


end inequality_l516_516440


namespace apples_taken_out_l516_516955

theorem apples_taken_out :
  ∀ (A : ℕ),
  let initial_apples := 7,
      initial_oranges := 8,
      initial_mangoes := 15,
      total_initial_fruits := initial_apples + initial_oranges + initial_mangoes,
      remaining_fruits := total_initial_fruits - 16,
      oranges_taken_out := 2 * A,
      mangoes_taken_out := 10 in
  initial_apples - A + initial_oranges - oranges_taken_out + initial_mangoes - mangoes_taken_out = remaining_fruits ↔ A = 2 :=
by
  intro A
  let initial_apples := 7
  let initial_oranges := 8
  let initial_mangoes := 15
  let total_initial_fruits := initial_apples + initial_oranges + initial_mangoes
  let remaining_fruits := total_initial_fruits - 16
  let oranges_taken_out := 2 * A
  let mangoes_taken_out := 10
  have : total_initial_fruits = 30 := rfl
  have : remaining_fruits = 14 := rfl
  have : A + oranges_taken_out + mangoes_taken_out = 16 := sorry
  split
  { intro h
    have h1 : A = 2 := sorry
    exact h1 }
  { intro h
    have h2 : 3 * A = 6 := sorry
    exact h2 }

end apples_taken_out_l516_516955


namespace quadratic_equality_l516_516745

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l516_516745


namespace parabola_focus_directrix_distance_l516_516855

theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y = (1 / 4) * x^2 → 
  (∃ p : ℝ, p = 2 ∧ x^2 = 4 * p * y) →
  ∃ d : ℝ, d = 2 :=
by
  sorry

end parabola_focus_directrix_distance_l516_516855


namespace integral_sqrt3_x_correct_l516_516575

-- Defining the integral problem
noncomputable def integral_sqrt3_x : ℝ :=
  ∫ x in (0 : ℝ)..(1 : ℝ), real.cbrt x

-- The expected answer
theorem integral_sqrt3_x_correct : integral_sqrt3_x = 3 / 4 :=
by
  sorry

end integral_sqrt3_x_correct_l516_516575


namespace concurrency_of_GI_HJ_B_symmedian_l516_516431

variable 
  (ω : Circle) 
  (A B C D E F G H I J : Point) 
  (h : InscribedQuadrilateral ω A B C D)
  (hE : Line (Segment A A) ∩ Line (Segment C D) = E)
  (hF : Line (Segment A A) ∩ Line (Segment B C) = F)
  (hG : Line (Segment B E) ∩ Circle ω = G)
  (hH : Line (Segment B E) ∩ Line (Segment A D) = H)
  (hI : Line (Segment D F) ∩ Circle ω = I)
  (hJ : Line (Segment D F) ∩ Line (Segment A B) = J)

theorem concurrency_of_GI_HJ_B_symmedian :
  Concurrent (Line (Segment G I))
             (Line (Segment H J))
             (B_Symmedian ω (A, B, C, D)) :=
sorry

end concurrency_of_GI_HJ_B_symmedian_l516_516431


namespace arrangement_scheme_count_l516_516072

def students := {A, B, C, D, E}
def jobs := {translator, tourGuide, etiquette, driver}

def A_can_do : jobs \ {driver}
def B_can_do : jobs \ {translator}
def C_can_do : jobs
def D_can_do : jobs
def E_can_do : jobs

def arrangement_schemes : ℕ :=
  let options_for_A := 3 -- A has 3 options (since he cannot drive)
  let options_for_B_if_A_does_translation := 3 -- B has 3 options if A does translation
  let options_for_B_if_A_does_not_translation := 2 -- B has 2 options if A does not translation
  let choices_for_remaining_two := 6 -- Choosing 2 jobs out of 3 people
  let total_if_A_B_participate := options_for_A + (options_for_B_if_A_does_translation + options_for_B_if_A_does_not_translation * options_for_A)
  let total_if_only_one_A_B_participate := 36 -- given that only one of A, B participates
  total_if_A_B_participate * choices_for_remaining_two + total_if_only_one_A_B_participate

theorem arrangement_scheme_count : arrangement_schemes = 78 := 
  by
    sorry

end arrangement_scheme_count_l516_516072


namespace semicircle_area_difference_l516_516681

theorem semicircle_area_difference 
  (A B C P D E F : Type) 
  (h₁ : S₅ - S₆ = 2) 
  (h₂ : S₁ - S₂ = 1) 
  : S₄ - S₃ = 3 :=
by
  -- Using Lean tactics to form the proof, place sorry for now.
  sorry

end semicircle_area_difference_l516_516681


namespace gcd_1458_1479_l516_516292

def a : ℕ := 1458
def b : ℕ := 1479
def gcd_ab : ℕ := 21

theorem gcd_1458_1479 : Nat.gcd a b = gcd_ab := sorry

end gcd_1458_1479_l516_516292


namespace fg_half_ab_l516_516029

theorem fg_half_ab
  (A B C U D E F G : Point)
  (hABC : triangle_right_angle_at A B C C)
  (hU : circumcenter A B C U)
  (hD : on AC D)
  (hE : on BC E)
  (hAngleEUD : ∠ E U D = 90)
  (hF : foot_perpendicular D A B F)
  (hG : foot_perpendicular E A B G) : 
  length FG = (1 / 2) * length AB := 
sorry

end fg_half_ab_l516_516029


namespace decipher_proof_l516_516568

noncomputable def decipher_message (n : ℕ) (hidden_message : String) :=
  if n = 2211169691162 then hidden_message = "Kiss me, dearest" else false

theorem decipher_proof :
  decipher_message 2211169691162 "Kiss me, dearest" = true :=
by
  -- Proof skipped
  sorry

end decipher_proof_l516_516568


namespace radius_of_spheres_in_cone_l516_516010

theorem radius_of_spheres_in_cone :
  ∀ (r : ℝ),
    let base_radius := 6
    let height := 15
    let distance_from_vertex := (2 * Real.sqrt 3 / 3) * r
    let total_height := height - r
    (total_height = distance_from_vertex) →
    r = 27 - 6 * Real.sqrt 3 :=
by
  intros r base_radius height distance_from_vertex total_height H
  sorry -- The proof of the theorem will be filled here.

end radius_of_spheres_in_cone_l516_516010


namespace spent_on_accessories_l516_516083

-- Definitions based on the conditions
def original_money : ℕ := 48
def money_on_snacks : ℕ := 8
def money_left_after_purchases : ℕ := (original_money / 2) + 4

-- Proving how much Sid spent on computer accessories
theorem spent_on_accessories : ℕ :=
  original_money - (money_left_after_purchases + money_on_snacks) = 12 :=
by
  sorry

end spent_on_accessories_l516_516083


namespace symmetric_matrix_eigenvalue_inequality_equality_condition_l516_516026

noncomputable def diagonal_matrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ (i j : Fin n), i ≠ j → A i j = 0

theorem symmetric_matrix_eigenvalue_inequality (n : ℕ) 
  (A : Matrix (Fin n) (Fin n) ℝ) (hA_symm : A.isSymm) 
  (λs : Fin n → ℝ) (hλs : A.eigenvalues = λs) :
  (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico (i + 1) n, A i i * A j j) >= 
  (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico (i + 1) n, λs i * λs j) :=
by
  sorry

theorem equality_condition (n : ℕ)
  (A : Matrix (Fin n) (Fin n) ℝ) (hA_symm : A.isSymm)
  (λs : Fin n → ℝ) (hλs : A.eigenvalues = λs) :
  (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico (i + 1) n, A i i * A j j)
  = (∑ i in Finset.Ico 0 n, ∑ j in Finset.Ico (i + 1) n, λs i * λs j) ↔ 
  ∀ (i j : Fin n), i ≠ j → A i j = 0 :=
by
  sorry

end symmetric_matrix_eigenvalue_inequality_equality_condition_l516_516026


namespace apples_total_l516_516062

/-- Pinky the Pig, Danny the Duck, and Benny the Bunny collectively have 157 apples after accounting for Lucy the Llama's sales. --/
theorem apples_total (pinky_apples : ℕ) (danny_apples : ℕ) (benny_apples : ℕ) (lucy_sales : ℕ) :
  pinky_apples = 36 →
  danny_apples = 73 →
  benny_apples = 48 →
  lucy_sales = 15 →
  pinky_apples + danny_apples + benny_apples = 157 :=
by
  intros h_p h_d h_b h_l
  rw [h_p, h_d, h_b]
  exact congr_arg2 (· + ·) (congr_arg2 (· + ·) rfl rfl) rfl
  sorry

end apples_total_l516_516062


namespace part1_part2_l516_516320

-- Define the quadratic equation
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 - (2*m - 3)*x + m^2 + 1 = 0

-- Part (1): If m is a real root of the quadratic equation
theorem part1 (m : ℝ) (h : quadratic_eq m m) : m = -1/3 := 
  sorry

-- Part (2): If m is negative, the situation of the roots of the equation
theorem part2 (m : ℝ) (h_m_neg : m < 0) : 
  let Δ := (2*m - 3)^2 - 4 * (m^2 + 1) in
  Δ > 0 :=
  sorry

end part1_part2_l516_516320


namespace projection_of_b_onto_a_l516_516147

def vector_a : (ℝ × ℝ) := (0, 1)
def vector_b : (ℝ × ℝ) := (-2, -8)

theorem projection_of_b_onto_a :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_a_squared := vector_a.1 ^ 2 + vector_a.2 ^ 2
  let projection := (dot_product / magnitude_a_squared) * vector_a
  projection = (0, -8) :=
by
  sorry

end projection_of_b_onto_a_l516_516147


namespace correct_option_is_A_l516_516933

variable (a b : ℤ)

-- Option A condition
def optionA : Prop := 3 * a^2 * b / b = 3 * a^2

-- Option B condition
def optionB : Prop := a^12 / a^3 = a^4

-- Option C condition
def optionC : Prop := (a + b)^2 = a^2 + b^2

-- Option D condition
def optionD : Prop := (-2 * a^2)^3 = 8 * a^6

theorem correct_option_is_A : 
  optionA a b ∧ ¬optionB a ∧ ¬optionC a b ∧ ¬optionD a :=
by
  sorry

end correct_option_is_A_l516_516933


namespace transformation_sequences_return_to_original_l516_516802

noncomputable def vertices : List (Int × Int) := [(1,1), (5,1), (1,4)]
noncomputable def transformations : List (ℝ × ℤ) := [(60,1), (120,1), (180,1), (1, -1), (-1, 1)]
noncomputable def sequences_of_three_transformations := 125

-- Prove the number of sequences of three transformations that return the triangle to its original position is 12
theorem transformation_sequences_return_to_original : 
  (count_valid_sequences vertices transformations sequences_of_three_transformations) = 12 := by
  sorry

end transformation_sequences_return_to_original_l516_516802


namespace area_ratio_l516_516895

-- Define the conditions given in the problem
variables (A A1 C C1 B E F : Point) -- Define points in space
variables (BC1 EF : ℝ) -- Define lengths as real numbers

-- Given conditions
axiom A1B : A1B = 2
axiom EF_condition : EF = 1
axiom BC1_condition : BC1 = 2
axiom AE_arithmetic_mean : AE = (BC1 + EF) / 2 

-- Main proof statement
theorem area_ratio :
  let AE := (BC1 + EF) / 2 in
  AE = ((2 + 1) / 2) →
  (1 + real.sqrt (5 / 2)) :=
begin
  sorry -- Proof to be completed
end

end area_ratio_l516_516895


namespace ellipse_equation_angle_relation_l516_516341

open Real

-- Definitions based on conditions
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_C (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def eccentricity (a b c : ℝ) : Prop :=
  c / a = sqrt(2) / 2 ∧ c = sqrt(a^2 - b^2)

def line_intersection (k : ℝ) (xa ya xb yb : ℝ) : Prop :=
  ya = k * xa - 1/3 ∧ yb = k * xb - 1/3

def midpoint (xa ya xb yb xm ym : ℝ) : Prop :=
  xm = (xa + xb) / 2 ∧ ym = (ya + yb) / 2

-- Theorem statements
theorem ellipse_equation (a b : ℝ) (h_ab : a > b ∧ b > 0)
  (h_C : point_C 0 1) (h_ecc : eccentricity a b (sqrt(a^2 - b^2))) :
  ellipse a b h_ab 0 1 → a^2 = 2 ∧ b^2 = 1 :=
sorry

theorem angle_relation (a b : ℝ) (h_ab : a > b ∧ b > 0)
  (h_C : point_C 0 1) (h_ecc : eccentricity a b (sqrt(a^2 - b^2)))
  (k xa ya xb yb : ℝ) (h_line : line_intersection k xa ya xb yb)
  (xm ym : ℝ) (h_mid : midpoint xa ya xb yb xm ym) :
  ∃ λ : ℝ, λ = 2 ∧ ∀ (A B M C : ℝ × ℝ),
  ∠(A, M, C) = λ * ∠(A, B, C) :=
sorry

end ellipse_equation_angle_relation_l516_516341


namespace polygon_perimeter_is_35_l516_516256

-- Define the concept of a regular polygon with given side length and exterior angle
def regular_polygon_perimeter (n : ℕ) (side_length : ℕ) : ℕ := 
  n * side_length

theorem polygon_perimeter_is_35 (side_length : ℕ) (exterior_angle : ℕ) (n : ℕ)
  (h1 : side_length = 7) (h2 : exterior_angle = 72) (h3 : 360 / exterior_angle = n) :
  regular_polygon_perimeter n side_length = 35 :=
by
  -- We skip the proof body as only the statement is required
  sorry

end polygon_perimeter_is_35_l516_516256


namespace pentagon_angle_x_l516_516255

theorem pentagon_angle_x (C : Type*) [has_center C] : 
  let x := 2 * (360 / 5 : ℝ) in
  x = 144 := 
by
  sorry

-- Definitions needed for the theorem
class has_center (C : Type*) : Prop := 
  (center : C)
  (vertices : set C)
  (is_regular_pentagon : true)

end pentagon_angle_x_l516_516255


namespace percentile_80_equals_10_8_l516_516520

def dataSet : List ℝ := [10.2, 9.7, 10.8, 9.1, 8.9, 8.6, 9.8, 9.6, 9.9, 11.2, 10.6, 11.7]

def percentile (k : ℕ) (data : List ℝ) : ℕ :=
  let sortedData := data.sorted
  let n := sortedData.length
  let pos := (k : ℝ) / 100 * (n : ℝ)
  if pos % 1 = 0 then ⌊pos⌋.toNat - 1 else ⌈pos⌉.toNat - 1

theorem percentile_80_equals_10_8 : percentile 80 dataSet = 10.8 :=
by
  sorry

end percentile_80_equals_10_8_l516_516520


namespace pure_imaginary_implies_a_neg_one_l516_516352

theorem pure_imaginary_implies_a_neg_one (a : ℝ) 
  (h_pure_imaginary : ∃ (y : ℝ), z = 0 + y * I) : 
  z = a + 1 - a * I → a = -1 :=
by
  sorry

end pure_imaginary_implies_a_neg_one_l516_516352


namespace count_nasty_functions_l516_516968

def is_nasty (f : Fin 5 → Fin 5) : Prop :=
  ∀ a b : Fin 5, a ≠ b → f a ≠ b ∨ f b ≠ a

theorem count_nasty_functions : Finset.card {f : (Fin 5 → Fin 5) // is_nasty f} = 1950 := 
sorry

end count_nasty_functions_l516_516968


namespace smallest_constant_c_equality_conditions_l516_516042

-- Statement of the problem
theorem smallest_constant_c (n : ℕ) (hn : 2 ≤ n) 
  (x : ℕ → ℝ) (hx : ∀ i, 0 ≤ x i) :
  (∑ i in finset.Ico 1 n, ∑ j in finset.Ico (i + 1) (n + 1), 
  x i * x j * (x i ^ 2 + x j ^ 2)) ≤ (1 / 8) * (∑ i in finset.Ico 1 (n + 1), x i) ^ 4 := sorry

theorem equality_conditions (n : ℕ) (hn : 2 ≤ n)
  (x : ℕ → ℝ) (hx : ∀ i, 0 ≤ x i) :
  (∑ i in finset.Ico 1 (n + 1), x i) ^ 2 = 4 * 
  (∑ i in finset.Ico 1 n, ∑ j in finset.Ico (i + 1) (n + 1), x i * x j)
  ∧ (∑ i in finset.Ico 1 n, ∑ j in finset.Ico (i + 1) (n + 1), 
  ∑ k in finset.Ico (j + 1) (n + 1), x i * x j * x k * (x i + x j + x k) = 0) 
  ↔ ∃ i j, x i = x j ∧ (∀ k ≠ i, ∀ k ≠ j, x k = 0) := sorry

end smallest_constant_c_equality_conditions_l516_516042


namespace correct_statement_l516_516935

theorem correct_statement :
  (∀ x y : ℝ, tan x > tan y) where
  x = Real.toRadians 46
  y = Real.toRadians 44 :=
sorry

end correct_statement_l516_516935


namespace remainder_of_9876543210_div_101_l516_516925

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l516_516925


namespace value_of_angle_BAC_maximum_sum_of_areas_l516_516781

variables {A B C D : Type*} [inner_product_space ℝ A]
variables (U V W X : A)

def points_opposite_side (A B C D : A) : Prop :=
det A B C * det A D C < 0

def quadrilateral_with_conditions (T : Type*) 
  [inner_product_space ℝ T] (A B C D : T) : Prop :=
(dist A B = 3) ∧ (dist B C = 5) ∧ 
(∃ B_angle D_angle : ℝ, cos B_angle = 3 / 5 ∧ cos D_angle = -3 / 5)

theorem value_of_angle_BAC
  (h : quadrilateral_with_conditions ℝ U V W X) :
  sorry := sorry

theorem maximum_sum_of_areas
  (h : quadrilateral_with_conditions ℝ U V W X) :
  sorry := sorry

end value_of_angle_BAC_maximum_sum_of_areas_l516_516781


namespace greatest_common_divisor_of_150_and_n_l516_516143

theorem greatest_common_divisor_of_150_and_n (n : ℕ) (h1 : (∀ d, d ∣ 150 ∧ d ∣ n → d = 1 ∨ d = 5 ∨ d = 25))
  : ∃ p : ℕ, nat.prime p ∧ (∀ d, d ∣ 150 ∧ d ∣ n → d = 1 ∨ d = p ∨ d = p^2) ∧ p = 5 :=
begin
  sorry
end

end greatest_common_divisor_of_150_and_n_l516_516143


namespace dragonfly_probability_three_moves_l516_516584

def tetrahedron_vertices := ({A, B, C, D} : set char)
def tetrahedron_edges := { (A, B), (A, C), (A, D), (B, C), (B, D), (C, D) } -- Assuming edges are undirected
def valid_move (u v : char) (visited : set char) : Prop := (u, v) ∈ tetrahedron_edges ∧ v ∉ visited

noncomputable def probability_visits_all_vertices (start : char) := 
  let valid_sequences : list (list char) := [
    -- Sequences starting from 'start', visiting each vertex exactly once
    [start, if start = A then B else if start = B then A else if start = C then A else D, C, D],
    [start, if start = A then B else if start = B then A else if start = C then A else D, D, C],
    [start, if start = A then C else if start = B then C else if start = C then B else A, B, D],
    [start, if start = A then C else if start = B then C else if start = C then B else A, D, B], 
    -- Additional sequences to cover all permutations for symmetry
  ] in
  (valid_sequences.length : ℝ) / 27

theorem dragonfly_probability_three_moves (start : char) :
  start ∈ tetrahedron_vertices → 
  probability_visits_all_vertices start = 4 / 27 := by
  sorry

end dragonfly_probability_three_moves_l516_516584


namespace MN2_minus_PQ2_eq_13_l516_516446

variables (A B C D M P N Q : Type) [convex_quadrilateral A B C D]
variables (AB BC CD DA : ℝ) 

-- Define the side lengths
variables (hAB : AB = 5) (hBC : BC = 6) (hCD : CD = 7) (hDA : DA = 8)

-- Define the midpoints
variables (M_mid : midpoint A B M) (P_mid : midpoint B C P) 
          (N_mid : midpoint C D N) (Q_mid : midpoint D A Q)

theorem MN2_minus_PQ2_eq_13 : 
  MN^2 - PQ^2 = 13 :=
sorry

end MN2_minus_PQ2_eq_13_l516_516446


namespace INPUT_is_input_statement_l516_516217

-- Define what constitutes each type of statement
def isOutputStatement (stmt : String) : Prop :=
  stmt = "PRINT"

def isInputStatement (stmt : String) : Prop :=
  stmt = "INPUT"

def isConditionalStatement (stmt : String) : Prop :=
  stmt = "THEN"

def isEndStatement (stmt : String) : Prop :=
  stmt = "END"

-- The main theorem
theorem INPUT_is_input_statement : isInputStatement "INPUT" := by
  sorry

end INPUT_is_input_statement_l516_516217


namespace ramu_profit_percent_l516_516068

def purchase_price := 48000
def mechanical_repairs := 6000
def bodywork := 4000
def interior_refurbishment := 3000
def taxes_and_fees := 2000
def selling_price := 72900

def total_cost := purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees
def profit := selling_price - total_cost
def profit_percent := (profit.toFloat / total_cost.toFloat) * 100

theorem ramu_profit_percent : profit_percent ≈ 15.71 := 
by
  sorry

end ramu_profit_percent_l516_516068


namespace number_of_possible_sets_P_l516_516450

universe u
variable {α : Type u}

def universal_set : Set Int := {x : Int | -5 < x ∧ x < 5}
def subset_S : Set Int := {-1, 1, 3}

theorem number_of_possible_sets_P :
  (#{P : Set Int // (∀ x, x ∈ universal_set ↔ x ∉ P) ∧ (P ⊆ universal_set) ∧ (Pᶜ ∩ universal_set) ⊆ subset_S}).card = 8 := 
sorry

end number_of_possible_sets_P_l516_516450


namespace ratio_of_kinetic_energies_l516_516594

def elastic_collision (m : ℝ) (v0 T_max : ℝ) : ℝ :=
  let vf := v0 / 2
  let KE_initial := 1 / 2 * m * v0^2
  let KE_final := 1 / 2 * (3 * m * vf^2 + m * (v0 - 3 * vf)^2)
  KE_final / KE_initial

theorem ratio_of_kinetic_energies (m v0 T_max : ℝ) (h : v0 > 0) :
  elastic_collision m v0 T_max = 1 / 2 :=
by
  sorry

end ratio_of_kinetic_energies_l516_516594


namespace quadratic_properties_l516_516692

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties 
  (a b c : ℝ) (ha : a ≠ 0) (h_passes_through : quadratic_function a b c 0 = 1) (h_unique_zero : quadratic_function a b c (-1) = 0) :
  quadratic_function a b c = quadratic_function 1 2 1 ∧ 
  (∀ k, ∃ g,
    (k ≤ -2 → g = k + 3) ∧ 
    (-2 < k ∧ k ≤ 6 → g = -((k^2 - 4*k) / 4)) ∧ 
    (6 < k → g = 9 - 2*k)) :=
sorry

end quadratic_properties_l516_516692


namespace find_machines_l516_516487

theorem find_machines (R : ℝ) : 
  (N : ℕ) -> 
  (H1 : N * R * 6 = 1) -> 
  (H2 : 4 * R * 12 = 1) -> 
  N = 8 :=
by
  sorry

end find_machines_l516_516487


namespace inverse_proportionality_l516_516570

theorem inverse_proportionality:
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = k / x) ∧ y = 1 ∧ x = 2 →
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = 2 / x :=
by
  sorry

end inverse_proportionality_l516_516570


namespace polynomial_transformation_l516_516045

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_transformation (x : ℝ) :
  (f (x^2 + 2) = x^4 + 6 * x^2 + 4) →
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  intro h
  sorry

end polynomial_transformation_l516_516045


namespace four_digit_sum_l516_516967

theorem four_digit_sum (n1 n2 n3 : ℕ) (h1 : n1 < 10000) (h2 : n2 < 10000) (h3 : n3 < 10000)
  (h4 : is_ascending n1) (h5 : is_descending n2) (h6 : is_permutation_of_digits n1 n3) 
  (h7 : n1 + n2 + n3 = 14364) : 
  n1 = 3456 := sorry

-- Definitions of conditions

-- Check if a number is in ascending order
def is_ascending (n : ℕ) : Prop := 
  let digits := (nat.digits 10 n)
  digits = list.sort digits

-- Check if a number is in descending order
def is_descending (n : ℕ) : Prop := 
  let digits := (nat.digits 10 n)
  digits = list.reverse (list.sort digits)

-- Check if one number is a permutation of the digits of another
def is_permutation_of_digits (n1 n2 : ℕ) : Prop :=
  let digits1 := (nat.digits 10 n1)
  let digits2 := (nat.digits 10 n2)
  digits1 ~ digits2

end four_digit_sum_l516_516967


namespace problem_statement_l516_516284

-- Declare x as a sequence limit
noncomputable def x : Real := 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + ...)) -- formalizing the infinite nested fraction

-- Our main theorem statement
theorem problem_statement :
  let frac : Real := 1 / ((x + 2) * (x - 3))
  ∃ A B C : Real, frac = (A + Real.sqrt B) / C ∧ abs A + abs B + abs C = 72 := 
by
  sorry

end problem_statement_l516_516284


namespace dice_probability_same_face_l516_516158

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516158


namespace M_eq_N_l516_516120

def M : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def N : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l516_516120


namespace find_value_of_S_l516_516779

noncomputable def value_of_S : ℝ :=
  let length_of_each_part := (2 / 8) in
  let S := 1.0 + length_of_each_part in
  S

-- Now we can state the theorem
theorem find_value_of_S (h1 : 0 ≤ 2) (h2 : ∀ i : ℕ, i ≤ 8 → 0 ≤ (i : ℝ) ∧ (i : ℝ) ≤ 2) : value_of_S = 1.25 :=
by
  suffices length_part : (2 / 8 : ℝ) = 0.25 by
  rw [value_of_S, length_part]
  norm_num

end find_value_of_S_l516_516779


namespace problem_solution_l516_516373

def f : ℕ → ℕ := sorry

axiom f_recursive (a b : ℕ) (ha : a > 0) (hb : b > 0) : f (a + b) = f a * f b
axiom f_base_case : f 1 = 2

theorem problem_solution : (∑ i in finset.range 1006, (f (2 * (i + 1)) / f (2 * i + 1))) = 2012 := 
by sorry

end problem_solution_l516_516373


namespace expression_value_at_2_l516_516930

theorem expression_value_at_2 : (2^2 + 3 * 2 - 4) = 6 :=
by 
  sorry

end expression_value_at_2_l516_516930


namespace number_of_foals_l516_516265

theorem number_of_foals (t f : ℕ) (h1 : t + f = 11) (h2 : 2 * t + 4 * f = 30) : f = 4 :=
by
  sorry

end number_of_foals_l516_516265


namespace integer_roots_l516_516599

-- Define the polynomial with integer coefficients
def polynomial (a2 a1 : ℤ) : ℤ[X] :=
  X^3 + a2 * X^2 + a1 * X - 11

-- The statement to prove
theorem integer_roots (a2 a1 : ℤ) :
  ∀ x : ℤ, polynomial a2 a1.eval x = 0 → x = -11 ∨ x = -1 ∨ x = 1 ∨ x = 11 :=
by
  sorry

end integer_roots_l516_516599


namespace conference_handshakes_l516_516273

theorem conference_handshakes :
  let n := 50
  let g1 := 30
  let g2 := 20
  let handshake_count := (g2 * (n - 1)) + (g2 * (g2 - 1) / 2)
  handshake_count = 1170 := 
by
  let n := 50
  let g1 := 30
  let g2 := 20
  let handshake_count := (g2 * (n - 1)) + (g2 * (g2 - 1) / 2)
  show handshake_count = 1170 from by sorry

end conference_handshakes_l516_516273


namespace man_speed_proof_l516_516593

noncomputable def man_speed_to_post_office (v : ℝ) : Prop :=
  let distance := 19.999999999999996
  let time_back := distance / 4
  let total_time := 5 + 48 / 60
  v > 0 ∧ distance / v + time_back = total_time

theorem man_speed_proof : ∃ v : ℝ, man_speed_to_post_office v ∧ v = 25 := by
  sorry

end man_speed_proof_l516_516593


namespace steve_bought_2_cookies_boxes_l516_516090

theorem steve_bought_2_cookies_boxes :
  ∀ (milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
     milk_qty cereal_qty banana_qty apple_qty cookies_qty : ℕ),
    milk_price = 3 →
    cereal_price_per_box = 3.5 →
    banana_price = 0.25 →
    apple_price = 0.5 →
    cookies_cost_per_box = 2 * milk_price →
    milk_qty = 1 →
    cereal_qty = 2 →
    banana_qty = 4 →
    apple_qty = 4 →
    (milk_price * milk_qty + cereal_price_per_box * cereal_qty + 
     banana_price * banana_qty + apple_price * apple_qty + 
     cookies_cost_per_box * cookies_qty = 25) →
    cookies_qty = 2 :=
by
  intros milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
         milk_qty cereal_qty banana_qty apple_qty cookies_qty
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 hsum
  have h_milk : milk_price = 3 := h1
  have h_cereal : cereal_price_per_box = 3.5 := h2
  have h_banana : banana_price = 0.25 := h3
  have h_apple : apple_price = 0.5 := h4
  have h_cookies_price : cookies_cost_per_box = 2 * milk_price := h5
  have h_milk_qty : milk_qty = 1 := h6
  have h_cereal_qty : cereal_qty = 2 := h7
  have h_banana_qty : banana_qty = 4 := h8
  have h_apple_qty : apple_qty = 4 := h9
  rw [h_milk, h_cereal, h_banana, h_apple, h_cookies_price, h_milk_qty, h_cereal_qty, h_banana_qty, h_apple_qty] at hsum
  sorry

end steve_bought_2_cookies_boxes_l516_516090


namespace complement_A_in_U_l516_516451

open Set

noncomputable def U := {x : ℕ | (Real.sqrt x : ℝ) ≤ 2}
def A := ({1, 2} : Set ℕ)
def complement_U_A := {0, 3, 4}

theorem complement_A_in_U : compl A ∩ U = complement_U_A := by
  sorry

end complement_A_in_U_l516_516451


namespace spade_problem_l516_516318

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 2 (spade 3 (spade 1 4)) = -46652 := 
by sorry

end spade_problem_l516_516318


namespace smallest_value_d_plus_e_l516_516702

theorem smallest_value_d_plus_e :
  ∃ (d e : ℕ), (2^10 * 3^6 * 5^4 = d^e) ∧ (d > 0) ∧ (e > 0) ∧ (d + e = 21602) :=
begin
  sorry
end

end smallest_value_d_plus_e_l516_516702


namespace triplet_solution_l516_516638

theorem triplet_solution (p q : ℕ) (n : ℕ) (hp : p.prime) (hq : q.prime)
  (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (hn : n > 1) :
  (q^(n+2) ≡ 3^(n+2) [MOD (p^n)]) ∧ (p^(n+2) ≡ 3^(n+2) [MOD (q^n)]) ↔ 
  (p = 3 ∧ q = 3 ∧ n ≥ 2) :=
sorry

end triplet_solution_l516_516638


namespace sphere_properties_l516_516528

theorem sphere_properties (V : ℝ) (π : ℝ) (r S d : ℝ) (hV : V = 288 * π)
    (hV_eq : V = (4 / 3) * π * r^3) :
    S = 4 * π * r^2 ∧ d = 2 * r :=
by
  -- standard definitions for radius-based derivations from volume
  have r_cube_eq : r^3 = (288 * π * 3) / (4 * π),
  { rw [hV, hV_eq], ring }
  -- solution can be completed as a proof later
  sorry

end sphere_properties_l516_516528


namespace no_solution_inequality_l516_516123

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end no_solution_inequality_l516_516123


namespace erin_serves_all_soup_in_15_minutes_l516_516650

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end erin_serves_all_soup_in_15_minutes_l516_516650


namespace dice_same_number_probability_l516_516200

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516200


namespace four_line_segments_max_planes_l516_516323

theorem four_line_segments_max_planes 
  (A B C D E : Point)
  (L1 : LineSegment A B)
  (L2 : LineSegment B C)
  (L3 : LineSegment C D)
  (L4 : LineSegment D E) :
  ∃ P1 P2 P3 P4 : Plane, 
    (L1 ⊆ P1) ∧ (L2 ⊆ P2) ∧ (L3 ⊆ P3) ∧ (L4 ⊆ P4) ∧
    P1 ≠ P2 ∧ P2 ≠ P3 ∧ P3 ≠ P4 :=
sorry

end four_line_segments_max_planes_l516_516323


namespace second_smallest_root_is_zero_l516_516500

noncomputable def poly (a b c d : ℝ) : Polynomial ℝ :=
  Polynomial.X^7 - 8 * Polynomial.X^6 + 20 * Polynomial.X^5 + 5 * Polynomial.X^4 - a * Polynomial.X^3 - 2 * Polynomial.X^2 + (b - d) * Polynomial.X + c + 5

theorem second_smallest_root_is_zero (a b c d : ℝ) (h : ∀ x : ℝ, poly a b c d x = 0) :
  (∃ xs : List ℝ, xs.nodup ∧ xs.length = 4 ∧ list.sorted List.Ord xs ∧ xs.nth 1 = 0) :=
sorry

end second_smallest_root_is_zero_l516_516500


namespace part_one_part_two_part_three_l516_516954

-- Define conditions and statements for each part of the problem
def total_balls := 6
def red_balls := 2
def green_balls := 4

-- Function to calculate combination (n choose k)
def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then (nat.factorial n) / (nat.factorial k * nat.factorial (n - k))
else 0

-- First: Probability of drawing a red ball on the second draw
noncomputable def probability_of_drawing_red_on_second_draw : ℚ :=
  (red_balls * (total_balls - 1)) / (total_balls * (total_balls - 1))

theorem part_one : probability_of_drawing_red_on_second_draw = 1 / 3 := sorry

-- Second: Probability of drawing two balls of the same color
noncomputable def probability_of_drawing_same_color : ℚ :=
  (combination red_balls 2 + combination green_balls 2) / combination total_balls 2

theorem part_two : probability_of_drawing_same_color = 7 / 15 := sorry

-- Third: Finding the value of n where P(two red balls) = 1/21
noncomputable def value_of_n_for_given_probability : ℚ → ℕ :=
  λ prob, let n := (21 * 2) in (nat.find (λ x, combination (2 + x) 2 / combination (x + 2) 2 = prob))

noncomputable def calculated_n : ℕ := value_of_n_for_given_probability (1 / 21)

theorem part_three : calculated_n = 5 := sorry

end part_one_part_two_part_three_l516_516954


namespace distinct_real_number_sum_and_square_sum_eq_l516_516043

theorem distinct_real_number_sum_and_square_sum_eq
  (a b c d : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3)
  (h_square_sum : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / (a - b) / (a - c) / (a - d)) + (b^5 / (b - a) / (b - c) / (b - d)) +
  (c^5 / (c - a) / (c - b) / (c - d)) + (d^5 / (d - a) / (d - b) / (d - c)) = -9 :=
by
  sorry

end distinct_real_number_sum_and_square_sum_eq_l516_516043


namespace Lyka_savings_l516_516828

def Smartphone_cost := 800
def Initial_savings := 200
def Gym_cost_per_month := 50
def Total_months := 4
def Weeks_per_month := 4
def Savings_per_week_initial := 50
def Savings_per_week_after_raise := 80

def Total_savings : Nat :=
  let initial_savings := Savings_per_week_initial * Weeks_per_month * 2
  let increased_savings := Savings_per_week_after_raise * Weeks_per_month * 2
  initial_savings + increased_savings

theorem Lyka_savings :
  (Initial_savings + Total_savings) = 1040 := by
  sorry

end Lyka_savings_l516_516828


namespace probability_winning_l516_516875

-- Define the probability of losing
def P_lose : ℚ := 5 / 8

-- Define the total probability constraint
theorem probability_winning : P_lose = 5 / 8 → (1 - P_lose) = 3 / 8 := 
by
  intro h
  rw h
  norm_num
  sorry

end probability_winning_l516_516875


namespace y_coord_of_third_vertex_of_equilateral_l516_516997

/-- Given two vertices of an equilateral triangle at (0, 6) and (10, 6), and the third vertex in the first quadrant,
    prove that the y-coordinate of the third vertex is 6 + 5 * sqrt 3. -/
theorem y_coord_of_third_vertex_of_equilateral (A B C : ℝ × ℝ)
  (hA : A = (0, 6)) (hB : B = (10, 6)) (hAB : dist A B = 10) (hC : C.2 > 6):
  C.2 = 6 + 5 * Real.sqrt 3 :=
sorry

end y_coord_of_third_vertex_of_equilateral_l516_516997


namespace smaller_integer_is_49_l516_516097

theorem smaller_integer_is_49 (m n : ℕ) (hm : 10 ≤ m ∧ m < 100) (hn : 10 ≤ n ∧ n < 100)
  (h : (m + n) / 2 = m + n / 100) : min m n = 49 :=
by
  sorry

end smaller_integer_is_49_l516_516097


namespace lambda_range_l516_516501

theorem lambda_range (a b λ : ℝ) : 
  (∀ a b : ℝ, a^2 + 8 * b^2 ≥ λ * b * (a + b)) ↔ (-8 ≤ λ ∧ λ ≤ 4) :=
sorry

end lambda_range_l516_516501


namespace no_rain_four_days_l516_516513

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l516_516513


namespace smallest_prime_with_digits_sum_22_l516_516209

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l516_516209


namespace most_likely_outcomes_l516_516673

open ProbabilityTheory

noncomputable def probability {n : ℕ} (k : ℕ) : ℝ :=
  (nat.choose n k) * (1 / 2) ^ n

theorem most_likely_outcomes :
  let n := 5,
      pA := probability n 0,
      pB := probability n 0,
      pC := probability n 3,
      pD := probability n 2,
      pE := 2 * probability n 1 in
  pC = pD ∧ pD = pE ∧ pC > pA ∧ pA = pB
:=
by
  sorry

end most_likely_outcomes_l516_516673


namespace single_elimination_game_count_l516_516259

theorem single_elimination_game_count (n : Nat) (h : n = 23) : n - 1 = 22 :=
by
  sorry

end single_elimination_game_count_l516_516259


namespace johns_yearly_music_cost_l516_516420

theorem johns_yearly_music_cost 
  (hours_per_month : ℕ := 20)
  (minutes_per_hour : ℕ := 60)
  (average_song_length : ℕ := 3)
  (cost_per_song : ℕ := 50) -- represented in cents to avoid decimals
  (months_per_year : ℕ := 12)
  : (hours_per_month * minutes_per_hour // average_song_length) * cost_per_song * months_per_year = 2400 * 100 := -- 2400 dollars (* 100 to represent cents)
  sorry

end johns_yearly_music_cost_l516_516420


namespace gumballs_initial_count_l516_516645

theorem gumballs_initial_count (x : ℝ) (h : (0.75 ^ 3) * x = 27) : x = 64 :=
by
  sorry

end gumballs_initial_count_l516_516645


namespace polygon_diagonals_l516_516979

theorem polygon_diagonals (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 10) :
    let n := 360 / exterior_angle in
    n * (n - 3) / 2 = 594 :=
by
  -- Proof to be filled in here.
  sorry

end polygon_diagonals_l516_516979


namespace number_of_configurations_l516_516951

-- Define the grid configuration
def grid_3x3 := fin 3 × fin 3

-- Define what it means for three points to form a line in a 3x3 grid
noncomputable def is_line (pts : set (fin 3 × fin 3)) : Prop :=
  ∃ m b : ℚ, ∀ pt ∈ pts, ↑pt.2 = m * ↑pt.1 + b

-- Define a configuration of the grid with triangles (t) and circles (c)
structure config :=
  (t : set grid_3x3)
  (c : set grid_3x3)
  (ht : is_line t)
  (hc : is_line c)
  (hdisjoint : t ∩ c = ∅)
  (htotal : t ∪ c = grid_3x3)

-- The number of such configurations is 84
theorem number_of_configurations : ∃ n : ℕ, n = 84 ∧ 
  (∃ cfgs : finset config, cfgs.card = n) :=
by
  sorry

end number_of_configurations_l516_516951


namespace same_face_probability_l516_516177

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516177


namespace solve_equation_l516_516843

theorem solve_equation (x : ℝ) (h1 : x ≠ 2 / 3) :
  (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_equation_l516_516843


namespace chi_square_test_probability_calculation_l516_516959

-- Definitions and conditions
def a := 15 -- Very interested and 35 years old and below
def b := 20 -- Very interested and above 35 years old
def c := 10 -- Not interested and 35 years old and below
def d := 15 -- Not interested and above 35 years old
def n := a + b + c + d -- Total sample size

-- Chi-square statistic calculation
def K_squared : ℝ := n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_square_test : K_squared ≈ 0.049 := by
  unfold K_squared
  norm_num
  sorry

-- Probability Calculation
def total_ways := Nat.choose 5 3
def favorable_ways := 6

def probability_exactly_one_not_interested := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_calculation : probability_exactly_one_not_interested = 3 / 5 := by
  unfold probability_exactly_one_not_interested total_ways favorable_ways
  norm_num
  sorry

end chi_square_test_probability_calculation_l516_516959


namespace domain_f_l516_516102

def f (x : ℝ) := log 2 (x^2 + 2*x - 3)

theorem domain_f :
  {x : ℝ | x^2 + 2*x - 3 > 0} = {x | x < -3} ∪ {x | x > 1} :=
by
  sorry

end domain_f_l516_516102


namespace quadratic_function_order_l516_516755

theorem quadratic_function_order (a b c : ℝ) (h_neg_a : a < 0) 
  (h_sym : ∀ x, (a * (x + 2)^2 + b * (x + 2) + c) = (a * (2 - x)^2 + b * (2 - x) + c)) :
  (a * (-1992)^2 + b * (-1992) + c) < (a * (1992)^2 + b * (1992) + c) ∧
  (a * (1992)^2 + b * (1992) + c) < (a * (0)^2 + b * (0) + c) :=
by
  sorry

end quadratic_function_order_l516_516755


namespace probability_no_rain_four_days_l516_516514

theorem probability_no_rain_four_days (p : ℚ) (p_rain : ℚ) 
  (h_p_rain : p_rain = 2 / 3) 
  (h_p_no_rain : p = 1 - p_rain) : 
  p ^ 4 = 1 / 81 :=
by
  have h_p : p = 1 / 3, sorry
  rw [h_p],
  norm_num

end probability_no_rain_four_days_l516_516514


namespace solve_fraction_eq_l516_516640

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 3) 
    (h₃ : 3 / (x - 2) = 6 / (x - 3)) : x = 1 :=
by 
  sorry

end solve_fraction_eq_l516_516640


namespace Pete_needs_20_bottles_l516_516892

def total_debt : ℝ := 90
def num_20_bills : ℕ := 2
def num_10_bills : ℕ := 4
def bottle_value : ℝ := 0.5

theorem Pete_needs_20_bottles :
  let total_amount := (num_20_bills * 20) + (num_10_bills * 10)
  let amount_needed := total_debt - total_amount
  amount_needed / bottle_value = 20 :=
by
  let total_amount := (num_20_bills * 20) + (num_10_bills * 10)
  have total_amount_eq : total_amount = 80 := by norm_num
  let amount_needed := total_debt - total_amount
  have amount_needed_eq : amount_needed = 10 := by norm_num
  show amount_needed / bottle_value = 20
  sorry

end Pete_needs_20_bottles_l516_516892


namespace part1_equation_of_ellipse_part2_ratio_is_constant_l516_516694

def eccentricity_is_constant (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (∃ c : ℝ, c = real.sqrt 3 ∧ c / a = real.sqrt 3 / 2)

def passes_through_point (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x = 1 ∧ y = real.sqrt 3 / 2 → (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)

theorem part1_equation_of_ellipse (a b : ℝ) (H1 : eccentricity_is_constant a b) (H2 : passes_through_point a b) :
  a = 2 ∧ b = 1 ∧ ∃ c : ℝ, c = real.sqrt 3 ∧ (x y : ℝ) (H: x ^ 2 / 4 + y ^ 2 = 1) :=
sorry

def line_intersects_ellipse (a b x y : ℝ) : Prop :=
  x = 1 ∧ y = 0 → (∃ m1 m2 : ℝ,
    (m1 + m2 ≠ 0 → (y1 y2 : ℝ, y1 + y2 = -2 * m1 / (m1 ^ 2 + 4) ∧ y1 * y2 = -3 / (m1 ^ 2 + 4)) ∧
    (y3 y4 : ℝ, y3 + y4 = -2 * m2 / (m2 ^ 2 + 4) ∧ y3 * y4 = -3 / (m2 ^ 2 + 4))

theorem part2_ratio_is_constant (a b : ℝ) (H1 : line_intersects_ellipse a b 1 0) : 
  ∀ (m1 m2 : ℝ), m1 + m2 ≠ 0 → ∃ MG MH : ℝ, MG / MH = 1 :=
sorry

end part1_equation_of_ellipse_part2_ratio_is_constant_l516_516694


namespace eccentricity_of_hyperbola_is_two_l516_516332

open_locale real

def point_on_terminal_side_of_angle (α : ℝ) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ α = real.pi / 3

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : point_on_terminal_side_of_angle (real.pi / 3) a b) : ℝ :=
  2

theorem eccentricity_of_hyperbola_is_two 
  (α : ℝ) (a b : ℝ) 
  (h : point_on_terminal_side_of_angle α a b) 
  (hab : α = real.pi / 3) : 
  hyperbola_eccentricity a b h = 2 :=
by {
  rw [hab],
  sorry
}

end eccentricity_of_hyperbola_is_two_l516_516332


namespace principal_calculation_l516_516559

noncomputable def principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem principal_calculation :
  principal_amount 1120 0.05 3 1 ≈ 967.68 :=
by
  calc
    principal_amount 1120 0.05 3 1
        = 1120 / (1 + 0.05 / 1) ^ (1 * 3) := rfl
    ... = 1120 / (1.05) ^ 3             := rfl
    ... = 1120 / 1.157625              := rfl
    ... ≈ 967.68                       := by norm_num

end principal_calculation_l516_516559


namespace smallest_prime_with_digit_sum_22_l516_516204

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_with_digit_sum_22 : ∃ p : ℕ, is_prime p ∧ digit_sum p = 22 ∧ 
  (∀ q : ℕ, is_prime q ∧ digit_sum q = 22 → p ≤ q) ∧ p = 499 :=
sorry

end smallest_prime_with_digit_sum_22_l516_516204


namespace A_investment_amount_l516_516970

def investment_share (capital : ℝ) (time : ℝ) (total_capital : ℝ) (total_time : ℝ) (total_profit : ℝ) : ℝ := 
  (capital * time) / (total_capital * total_time) * total_profit

theorem A_investment_amount :
  ∃ (x : ℝ), 
  let total_profit := 100 in
  let A_share := 75 in
  let B_investment := 200 in
  let A_time := 12 in
  let B_time := 6 in
  let total_time := 12 in
  let eqn := investment_share x A_time (x * A_time + B_investment * B_time) total_time total_profit  in
  eqn = A_share ∧ x = 300 :=
by
  sorry

end A_investment_amount_l516_516970


namespace gasoline_price_increase_l516_516381

theorem gasoline_price_increase
  (P Q : ℝ)
  (h1 : (P * Q) * 1.10 = P * (1 + X / 100) * Q * 0.88) :
  X = 25 :=
by
  -- proof here
  sorry

end gasoline_price_increase_l516_516381


namespace Bob_age_l516_516526

variable (B C : ℕ)

theorem Bob_age : B + C = 66 ∧ C = 3 * B + 2 → B = 16 := by
  assume h
  sorry

end Bob_age_l516_516526


namespace conditional_probability_example_l516_516119

open ProbabilityTheory

/-- Event A is defined as the component's service life exceeding 1 year and event B as the component's service life exceeding 2 years.
    Given that P(A) = 0.6 and P(B ∩ A) = 0.3, we need to show that P(B | A) = 0.5. -/
theorem conditional_probability_example (P : event → ℝ) 
  (A B : event) 
  (hA: P A = 0.6) 
  (hB_inter_A: P (B ∩ A) = 0.3) : 
  P(B | A) = 0.5 :=
sorry

end conditional_probability_example_l516_516119


namespace number_of_subsets_of_P_l516_516822

theorem number_of_subsets_of_P {P : Set ℕ} (h : P = {0, 1}) : (P.powerset.card = 4) :=
by 
  sorry

end number_of_subsets_of_P_l516_516822


namespace tangent_line_slope_at_pi_over_2_l516_516672

def slope_of_tangent_line (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  (deriv f) x

theorem tangent_line_slope_at_pi_over_2 : 
  slope_of_tangent_line (λ x : ℝ, Real.tan (x / 2)) (Real.pi / 2) = 1 :=
by 
  -- proof goes here
  sorry

end tangent_line_slope_at_pi_over_2_l516_516672


namespace trajectory_of_point_P_l516_516696

noncomputable def point := ℝ × ℝ

def A : point := (0, real.sqrt 3)

def circle_eq (x y : ℝ) : Prop := x^2 + (y + real.sqrt 3)^2 = 16

def is_on_circle (M : point) : Prop := circle_eq M.1 M.2

def radius_O1M (O1 M : point) := (M.1 - O1.1)^2 + (M.2 - O1.2)^2 = 16

theorem trajectory_of_point_P : 
    ∀ (P M : point),
    is_on_circle M ∧ (M ≠ (0, real.sqrt 3)) ∧ (M ≠ (0, -real.sqrt 3)) ∧ radius_O1M (0, -real.sqrt 3) M →
    let O1 := (0, -real.sqrt 3) in
    abs ((P.1 - A.1)^2 + (P.2 - A.2)^2 - ((P.1 - O1.1)^2 + (P.2 - O1.2)^2)) = 4 →
    P.1^2 + (P.2^2) / 4 = 1 :=
begin
  sorry
end

end trajectory_of_point_P_l516_516696


namespace same_number_on_four_dice_l516_516185

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516185


namespace find_length_of_DF_l516_516001

theorem find_length_of_DF (cos_E : ℝ) (DF : ℝ) (EF : ℝ)
    (h1 : cos_E = DF / EF)
    (h2 : cos_E = 8 * √115 / 115)
    (h3 : EF = √115) :
    DF = 8 := 
sorry

end find_length_of_DF_l516_516001


namespace remainder_9876543210_mod_101_l516_516919

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l516_516919


namespace count_valid_a_l516_516679

theorem count_valid_a :
  ∃ (S : Finset ℕ), (∀ a ∈ S, (1 ≤ a ∧ a ≤ 40) ∧ ∃ (x y : ℤ), x^2 + (3 * a + 2) * x + a^2 = 0 ∧ y^2 + (3 * a + 2) * y + a^2 = 0)
  ∧ S.card = 5 := 
sorry

end count_valid_a_l516_516679


namespace pizza_percent_increase_l516_516834

-- Define the diameters
def diameter1 : ℝ := 8
def diameter2 : ℝ := 14

-- Define the areas
def area (d : ℝ) : ℝ := Real.pi * (d / 2)^2

-- Compute percent increase
def percent_increase (A1 A2 : ℝ) : ℝ := ((A2 - A1) / A1) * 100

-- The main statement
theorem pizza_percent_increase : 
  percent_increase (area diameter1) (area diameter2) = 206.25 :=
by 
  -- This is where the proof would go
  sorry

end pizza_percent_increase_l516_516834


namespace susie_total_score_l516_516389

theorem susie_total_score :
  let c := 15 in
  let i := 10 in
  let u := 5 in
  let p_c := 2 in
  let p_i := -1 in
  let p_u := 0 in
  (c * p_c + i * p_i + u * p_u) = 20 :=
by
  sorry

end susie_total_score_l516_516389


namespace shirt_cost_is_ten_l516_516961

theorem shirt_cost_is_ten (S J : ℝ) (h1 : J = 2 * S) 
    (h2 : 20 * S + 10 * J = 400) : S = 10 :=
by
  -- proof skipped
  sorry

end shirt_cost_is_ten_l516_516961


namespace max_b_minus_a_l516_516353

noncomputable def f (a x : ℝ) := (1/3:ℝ) * x ^ 3 - 2 * a * x
noncomputable def g (b x : ℝ) := x ^ 2 + 2 * b * x

noncomputable def f' (a x : ℝ) := x ^ 2 - 2 * a
noncomputable def g' (b x : ℝ) := 2 * x + 2 * b

theorem max_b_minus_a :
  ∀ {a b : ℝ},
    a > 0 →
    (∀ x, a < x ∧ x < b → f' a x * g' b x ≤ 0) →
    b - a ≤ 1 / 2 :=
begin
  sorry
end

end max_b_minus_a_l516_516353


namespace students_checked_out_l516_516463

theorem students_checked_out (n l : ℕ) (h1 : n = 16) (h2 : l = 9) :
  n - l = 7 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_checked_out_l516_516463


namespace sum_balanced_numbers_mod_13_is_0_l516_516900

def is_balanced (n : ℕ) : Prop :=
  let digits := (List.of_digits 10 n) in
  1000 < n ∧ n < 1000000 ∧
  (List.sum (digits.drop 3) = List.sum (digits.take 3))

theorem sum_balanced_numbers_mod_13_is_0 :
  (List.sum (List.filter is_balanced (List.range 999999))) % 13 = 0 := 
sorry

end sum_balanced_numbers_mod_13_is_0_l516_516900


namespace same_face_probability_l516_516184

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516184


namespace dice_probability_sum_three_l516_516550

theorem dice_probability_sum_three (total_outcomes : ℕ := 36) (favorable_outcomes : ℕ := 2) :
  favorable_outcomes / total_outcomes = 1 / 18 :=
by
  sorry

end dice_probability_sum_three_l516_516550


namespace students_from_other_communities_l516_516392

theorem students_from_other_communities (total_students : ℕ) 
  (percent_muslims percent_hindus percent_sikhs percent_christians percent_buddhists : ℕ) 
  (h_tot : total_students = 1500)
  (h_mus : percent_muslims = 38)
  (h_hin : percent_hindus = 26)
  (h_sik : percent_sikhs = 12)
  (h_chr : percent_christians = 6)
  (h_bud : percent_buddhists = 4) :
  let percent_other_communities := 100 - (percent_muslims + percent_hindus + percent_sikhs + percent_christians + percent_buddhists)
  in let num_other_communities := percent_other_communities * total_students / 100
  in num_other_communities = 210 :=
by
  sorry

end students_from_other_communities_l516_516392


namespace A1B1C1D1_is_cyclic_l516_516852

-- Definitions as per the conditions
variables (A B C D N A1 B1 C1 D1 : Type) [cyclic_quadrilateral A B C D]
variables (h_diagonals_intersect : diagonals_intersect_at A B C D N)
variables (h_A1_B1 : intersections_of_circumcircle_ANB A D B C A1 B1 N)
variables (h_C1_D1 : intersections_of_circumcircle_CND C D B A C1 D1 N)

theorem A1B1C1D1_is_cyclic :
  is_cyclic_quadrilateral_centered_at A1 B1 C1 D1 N :=
sorry

end A1B1C1D1_is_cyclic_l516_516852


namespace colorable_map_l516_516058

theorem colorable_map (n : ℕ) (circles : fin n → set (ℝ × ℝ)) (chords : fin n → set (ℝ × ℝ)) 
(h_circles_disjoint : ∀ i j, i ≠ j → circles i ∩ circles j = ∅)
(h_chords_per_circle : ∀ i, set_of_chords (chords i) circles i)
(h_chords_intersect_once : ∀ i j, i ≠ j → (chords i ∩ chords j).card ≤ 1) :
  ∃ (coloring : (ℝ × ℝ) → fin 3),
  (∀ i j, adj (chords i) (chords j) → coloring i ≠ coloring j) :=
sorry

end colorable_map_l516_516058


namespace fraction_disliking_but_liking_l516_516618

-- Definitions based on conditions
def total_students : ℕ := 100
def like_dancing : ℕ := 70
def dislike_dancing : ℕ := total_students - like_dancing

def say_they_like_dancing (like_dancing : ℕ) : ℕ := (70 * like_dancing) / 100
def say_they_dislike_dancing (like_dancing : ℕ) : ℕ := like_dancing - say_they_like_dancing like_dancing

def dislike_and_say_dislike (dislike_dancing : ℕ) : ℕ := (80 * dislike_dancing) / 100
def say_dislike_but_like (like_dancing : ℕ) : ℕ := say_they_dislike_dancing like_dancing

def total_say_dislike : ℕ := dislike_and_say_dislike dislike_dancing + say_dislike_but_like like_dancing

noncomputable def fraction_like_but_say_dislike : ℚ := (say_dislike_but_like like_dancing : ℚ) / (total_say_dislike : ℚ)

theorem fraction_disliking_but_liking : fraction_like_but_say_dislike = 46.67 / 100 := 
by sorry

end fraction_disliking_but_liking_l516_516618


namespace range_of_f_l516_516635

noncomputable def star (a b : ℝ) : ℝ :=
if a ≤ b then a else b

def f (x : ℝ) : ℝ :=
star (Real.cos x) (Real.sin x)

theorem range_of_f :
  Set.range f = Set.Icc (-1 : ℝ) (Real.sqrt 2 / 2) :=
sorry

end range_of_f_l516_516635


namespace max_convex_quadrilaterals_l516_516687

-- Define the points on the plane and the conditions
variable (A : Fin 7 → (ℝ × ℝ))

-- Hypothesis that any 3 given points are not collinear
def not_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1)

-- Hypothesis that the convex hull of all points is \triangle A1 A2 A3
def convex_hull_triangle (A : Fin 7 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 7), i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)

-- The theorem to be proven
theorem max_convex_quadrilaterals :
  convex_hull_triangle A →
  (∀ i j k : Fin 7, i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)) →
  ∃ n, n = 17 := 
by
  sorry

end max_convex_quadrilaterals_l516_516687


namespace probability_of_same_number_on_four_dice_l516_516168

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516168


namespace tetrahedron_inequality_l516_516785

section tetrahedron_inequality

variables {T : Type} [MetricSpace T] [NormedSpace ℝ T]

structure Tetrahedron (T : Type) [MetricSpace T] [NormedSpace ℝ T] :=
  (A B C D : T)

noncomputable def edge_length_sum (T : Tetrahedron T) : ℝ :=
  dist T.A T.B + dist T.A T.C + dist T.A T.D + dist T.B T.C + dist T.B T.D + dist T.C T.D

noncomputable def tetrahedron_volume (T : Tetrahedron T) : ℝ := sorry

noncomputable def cross_section_area (T : Tetrahedron T) (X Y : T) : ℝ := sorry

theorem tetrahedron_inequality (T : Tetrahedron T) :
  let P := edge_length_sum T,
      V := tetrahedron_volume T
  in
  let S_CD := cross_section_area T T.C T.D,
      S_AB := cross_section_area T T.A T.B,
      S_BC := cross_section_area T T.B T.C,
      S_AD := cross_section_area T T.A T.D,
      S_BD := cross_section_area T T.B T.D,
      S_AC := cross_section_area T T.A T.C
  in
  1 / S_CD + 1 / S_AB + 1 / S_BC + 1 / S_AD + 1 / S_BD + 1 / S_AC ≤ P / (3 * V) :=
sorry

end tetrahedron_inequality

end tetrahedron_inequality_l516_516785


namespace trigonometric_identity_l516_516482

theorem trigonometric_identity (α : ℝ) :
    cos (π / 4 - α) - sin (π / 4 - α) = sqrt 2 * sin α :=  
  sorry

end trigonometric_identity_l516_516482


namespace executive_board_elections_l516_516998

noncomputable def num_candidates : ℕ := 18
noncomputable def num_positions : ℕ := 6
noncomputable def num_former_board_members : ℕ := 8

noncomputable def total_selections := Nat.choose num_candidates num_positions
noncomputable def no_former_board_members_selections := Nat.choose (num_candidates - num_former_board_members) num_positions

noncomputable def valid_selections := total_selections - no_former_board_members_selections

theorem executive_board_elections : valid_selections = 18354 :=
by sorry

end executive_board_elections_l516_516998


namespace probability_three_draws_l516_516579

/-- Given a box containing 7 chips numbered 1 through 7,
    chips are drawn randomly without replacement until the sum exceeds 9.
    Prove that the probability of requiring exactly 3 draws with the last chip 
    being an odd number that causes the sum to exceed 9 is 1/14. -/
theorem probability_three_draws (chips : Finset ℕ) (f : chips = {1, 2, 3, 4, 5, 6, 7}) :
  (prob_event (draws_exceeding_sum 9 chips 3 odd) = 1 / 14) :=
sorry

end probability_three_draws_l516_516579


namespace smallest_square_side_lengths_l516_516137

theorem smallest_square_side_lengths (x : ℕ) 
    (h₁ : ∀ (y : ℕ), y = x + 8) 
    (h₂ : ∀ (z : ℕ), z = 50) 
    (h₃ : ∀ (QS PS RT QT : ℕ), QS = 8 ∧ PS = x ∧ RT = 42 - x ∧ QT = x + 8 ∧ (8 / x) = ((42 - x) / (x + 8))) : 
  x = 2 ∨ x = 32 :=
by 
  sorry

end smallest_square_side_lengths_l516_516137


namespace quadratic_roots_expression_l516_516335

-- We start by stating that we presume noncomputable theory where needed
noncomputable theory

-- The main statement of the proof problem
theorem quadratic_roots_expression :
  ∀ (m n : ℝ), (x^2 + m * x + 3 = 0 ↔ (x = 1 ∨ x = n)) →
  (1 + n = -m) →
  (1 * n = 3) →
  (m + n) ^ 2023 = -1 :=
by
  intros m n h_eq h_vieta_sum h_vieta_prod
  -- Proof steps would go here
  sorry

end quadratic_roots_expression_l516_516335


namespace initial_average_runs_l516_516096

theorem initial_average_runs (A : ℕ) (h : 10 * A + 87 = 11 * (A + 5)) : A = 32 :=
by
  sorry

end initial_average_runs_l516_516096


namespace symmetric_yaxis_sin_eq_l516_516003

theorem symmetric_yaxis_sin_eq (α β : ℝ) (h : ∀ (P : ℝ × ℝ), 
  (∃ r : ℝ, P = (r * cos α, r * sin α)) → (∃ r' : ℝ, (P.1, P.2) = (r' * cos β, r' * sin β) ∧ P.1 = -r * cos α ∧ P.2 = r * sin α)) :
  sin α = sin β :=
by
  sorry

end symmetric_yaxis_sin_eq_l516_516003


namespace correct_statements_l516_516358

variable (a b : ℝ^2)

-- Define hypotheses
def side_length_square : Prop := ∃ (A B C D : ℝ^2), dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2

def ab_relation : Prop := b - 2 * a = a

def bc_relation : Prop := b = 2 * a + bc

-- Define the theorem to prove the statements B and D
theorem correct_statements (h₁ : side_length_square) (h₂ : ab_relation) (h₃ : bc_relation) : 
  (a ⬝ b = 2) ∧ (b - 4 * a) ⬝ b = 0 := by
-- Proof steps go here
sorry

end correct_statements_l516_516358


namespace remainder_of_large_number_l516_516908

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l516_516908


namespace dice_probability_same_face_l516_516156

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516156


namespace min_value_3y3_6y_neg2_l516_516811

-- Let y be a positive real number, find the minimum value of 3y^3 + 6y^{-2}.
theorem min_value_3y3_6y_neg2 (y : ℝ) (hy : 0 < y) : ∃ m, (∀ z : ℝ, 0 < z → 3 * z^3 + 6 * z^(-2) ≥ m) ∧ m = 9 := 
sorry

end min_value_3y3_6y_neg2_l516_516811


namespace father_l516_516585

variable {son_age : ℕ} -- Son's present age
variable {father_age : ℕ} -- Father's present age

-- Conditions
def father_is_four_times_son (son_age father_age : ℕ) : Prop := father_age = 4 * son_age
def sum_of_ages_ten_years_ago (son_age father_age : ℕ) : Prop := (son_age - 10) + (father_age - 10) = 60

-- Theorem statement
theorem father's_present_age 
  (son_age father_age : ℕ)
  (h1 : father_is_four_times_son son_age father_age) 
  (h2 : sum_of_ages_ten_years_ago son_age father_age) : 
  father_age = 64 :=
sorry

end father_l516_516585


namespace probability_win_l516_516867

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l516_516867


namespace figure_area_is_sqrt3_div2_l516_516775

/-- Given a plane figure with sides AD and BE parallel, and sides AF and CD parallel,
with all these sides having length 1, and 
∠FAD = ∠BCD = 30° and ∠DAF = ∠DBC = 30°, and 
in which each diagonal AC and BD intersects at point O and divides each other into equal parts.
Prove that the area of the entire figure is sqrt(3) / 2. -/
noncomputable def area_of_figure : ℝ :=
let angle := 30 / 180 * Real.pi in
let s := 1 in
2 * (s * s * Real.sin(angle * 2) / 4)

theorem figure_area_is_sqrt3_div2 :
  area_of_figure = Real.sqrt 3 / 2 :=
sorry

end figure_area_is_sqrt3_div2_l516_516775


namespace length_difference_l516_516827

def list_I := [3, 4, 8, 19]
def x : Nat

def list_II := x :: list_I

theorem length_difference : (list_II.length - list_I.length) = 1 :=
by
  sorry

end length_difference_l516_516827


namespace Carter_reads_30_pages_per_hour_l516_516624

theorem Carter_reads_30_pages_per_hour :
  ∀ (Carter_read_per_hour : ℕ) (Lucy_read_per_hour : ℕ) (Oliver_read_per_hour : ℕ),
    (Carter_read_per_hour = Lucy_read_per_hour / 2) →
    (Lucy_read_per_hour = Oliver_read_per_hour + 20) →
    (Oliver_read_per_hour = 40) →
    Carter_read_per_hour = 30 :=
by
  intros Carter_read_per_hour Lucy_read_per_hour Oliver_read_per_hour
  assume CarterCondition : Carter_read_per_hour = Lucy_read_per_hour / 2
  assume LucyCondition : Lucy_read_per_hour = Oliver_read_per_hour + 20
  assume OliverCondition : Oliver_read_per_hour = 40
  sorry

end Carter_reads_30_pages_per_hour_l516_516624


namespace polynomial_degree_and_leading_coeff_l516_516490

noncomputable def binom : ℕ → ℕ → ℕ
| n k := if h : k ≤ n then nat.choose n k else 0 -- binomial coefficient

noncomputable def P (x : ℤ) : ℤ :=
  ∑ i in finset.range 2022, binom 2021 i * x ^ i

noncomputable def Q (x : ℤ) : ℤ :=
  ∑ i in finset.range 2023, binom 2022 i * x ^ i

theorem polynomial_degree_and_leading_coeff (x : ℤ) :
  (degree (P x * Q x) = 4043) ∧ (leading_coeff (P x * Q x) = 1) :=
by {
  sorry,
}

end polynomial_degree_and_leading_coeff_l516_516490


namespace same_number_probability_four_dice_l516_516171

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516171


namespace fraction_evaluation_l516_516278

theorem fraction_evaluation :
  (7 / 18 * (9 / 2) + 1 / 6) / ((40 / 3) - (15 / 4) / (5 / 16)) * (23 / 8) =
  4 + 17 / 128 :=
by
  -- conditions based on mixed number simplification
  have h1 : 4 + 1 / 2 = (9 : ℚ) / 2 := by sorry
  have h2 : 13 + 1 / 3 = (40 : ℚ) / 3 := by sorry
  have h3 : 3 + 3 / 4 = (15 : ℚ) / 4 := by sorry
  have h4 : 2 + 7 / 8 = (23 : ℚ) / 8 := by sorry
  -- the main proof
  sorry

end fraction_evaluation_l516_516278


namespace sum_middle_three_of_alternating_sequence_l516_516742

theorem sum_middle_three_of_alternating_sequence :
  ∃ (arr : List ℕ), arr = [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1] ∧
                    (∀ i, (i % 2 = 0 → arr.nth i < arr.nth (i + 1)) → 
                           (i % 2 = 1 → arr.nth i ≤ arr.nth (i + 1))) ∧
                    (arr.nth 3 + arr.nth 4 + arr.nth 5 = 13) :=
begin
  -- the proof is omitted
  sorry
end

end sum_middle_three_of_alternating_sequence_l516_516742


namespace evaluate_cube_root_power_l516_516651

theorem evaluate_cube_root_power (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^(3 : ℝ)) : (cbrt a)^12 = b^12 :=
by
  sorry

example : evaluate_cube_root_power 8 2 4096 (by rfl)

end evaluate_cube_root_power_l516_516651


namespace position_of_2010_l516_516461

-- Define the set of possible digits
def digits : List ℕ := [0, 1, 2]

-- Define a predicate to check if a number can be formed using the given digits
def valid_number (n : ℕ) : Prop :=
  n.to_digits.all (λ d, d ∈ digits)

-- Define a predicate to check if a number is greater than 1000
def greater_than_1000 (n : ℕ) : Prop :=
  n > 1000

-- Define the sequence of numbers formed by the digits and greater than 1000
def number_sequence : List ℕ :=
  List.filter (λ n, valid_number n ∧ greater_than_1000 n) (List.range 10000)

-- Define the position of a number in the sequence
noncomputable def position (n : ℕ) : ℕ :=
  List.index_of n number_sequence + 1

-- The theorem statement that 2010 is in position 30
theorem position_of_2010 : position 2010 = 30 :=
  by
    sorry

end position_of_2010_l516_516461


namespace cover_condition_l516_516321

theorem cover_condition (n : ℕ) :
  (∃ (f : ℕ) (h1 : f = n^2), f % 2 = 0) ↔ (n % 2 = 0) := 
sorry

end cover_condition_l516_516321


namespace fraction_second_year_not_major_l516_516224

-- Define the conditions
def total_students (T : ℕ) : Prop := 
  let first_year_students := T / 2 in
  let second_year_students := T / 2 in
  let first_year_declared_major := 1 / 10 * T in
  let fraction_second_year_declared_major := 1 / 3 * (1 / 10) in
  let second_year_declared_major := fraction_second_year_declared_major * second_year_students in
  let second_year_not_declared_major := (1 - fraction_second_year_declared_major) * second_year_students in
  second_year_not_declared_major / T = 29 / 60

-- Theorem statement
theorem fraction_second_year_not_major {T : ℕ} (hT : total_students T) : 
  ( (1 - (1 / 3) * (1 / 10)) * (T / 2) ) / T = 29 / 60 := 
by
  sorry

end fraction_second_year_not_major_l516_516224


namespace grasshopper_ways_to_2014_l516_516969

/-- Given the grasshopper's jump rules, prove that the number of ways it can reach the line x + y = 2014 starting from (0, 0) is (3/4) * 3^2014 + (1/4). -/
theorem grasshopper_ways_to_2014 :
  let f : ℕ → ℚ := λ n, if n = 0 then 1 else if n = 1 then 2 else 2 * f (n - 1) + 3 * f (n - 2) in
  f 2014 = (3 / 4 : ℚ) * 3 ^ 2014 + (1 / 4 : ℚ) :=
sorry

end grasshopper_ways_to_2014_l516_516969


namespace find_angle_C_l516_516410

def cosine_rule (a b c : ℝ) : ℝ := 
(a^2 + b^2 - c^2) / (2 * a * b)

theorem find_angle_C (a b c : ℝ) (A B C : ℝ) :
  b + c = 2 * a →
  3 * a = 5 * b →
  C = Real.arccos (-1 / 2) :=
by
  intros h1 h2 
  have h3 : b = (3 / 5) * a := sorry 
  have h4 : c = (7 / 5) * a := sorry
  have h5 : cosine_rule a b c = -1 / 2 := sorry
  have h6 : C = Real.arccos (-1 / 2) := sorry
  assumption

end find_angle_C_l516_516410


namespace std_dev_sample_l516_516331

def x1 : ℕ := 4
def x2 : ℕ := 5
def x3 : ℕ := 6

noncomputable def mean (a b c : ℕ) : ℚ := (a + b + c) / 3

noncomputable def variance (a b c : ℕ) : ℚ := 
  (1 / 3 : ℚ) * (((a - mean a b c) ^ 2) + ((b - mean a b c) ^ 2) + ((c - mean a b c) ^ 2))

noncomputable def std_deviation (a b c : ℕ) : ℚ := real.sqrt (variance a b c)

theorem std_dev_sample :
  std_deviation x1 x2 x3 = (real.sqrt 6) / 3 :=
by
  unfold x1 x2 x3
  unfold mean
  unfold variance
  unfold std_deviation
  sorry

end std_dev_sample_l516_516331


namespace negative_integers_abs_le_4_l516_516988

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l516_516988


namespace shadow_projection_height_l516_516975

theorem shadow_projection_height :
  ∃ (x : ℝ), (∃ (shadow_area : ℝ), shadow_area = 192) ∧ 1000 * x = 25780 :=
by
  sorry

end shadow_projection_height_l516_516975


namespace remainder_div_101_l516_516913

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l516_516913


namespace sine_of_diagonal_angle_l516_516401

theorem sine_of_diagonal_angle :
  ∀ (A B C D O : Point) (S_ABO S_CDO BC : ℝ),
  (S_ABO = 3 / 2) →
  (S_CDO = 3 / 2) →
  (distance B C = 3 * sqrt 2) →
  (cos (angle A D C) = 3 / sqrt 10) →
  let AC := distance A C,
      BD := distance B D,
      α := angle A O C in
  sin α = 6 / sqrt 37 :=
by
  -- problem conditions setup, proof omitted
  sorry

end sine_of_diagonal_angle_l516_516401


namespace exist_2004_integers_sum_eq_product_l516_516615

theorem exist_2004_integers_sum_eq_product :
  ∃ (a : Fin 2004 → ℕ), (∀ (i : Fin 2004), a i > 0) ∧ (∑ i, a i = ∏ i, a i) :=
  sorry

end exist_2004_integers_sum_eq_product_l516_516615


namespace area_of_region_bounded_by_y_eq_xsq_y_eq_10_y_eq_5_y_axis_l516_516151

noncomputable def area_of_bounded_region : ℝ :=
  let f := λ x : ℝ, x^2 in
  let a := sqrt 5 in
  let b := sqrt 10 in
  2 * (∫ x in a..b, f x) / 3

theorem area_of_region_bounded_by_y_eq_xsq_y_eq_10_y_eq_5_y_axis :
  area_of_bounded_region = 20 * sqrt 10 / 3 - 10 * sqrt 5 / 3 :=
by
  sorry

end area_of_region_bounded_by_y_eq_xsq_y_eq_10_y_eq_5_y_axis_l516_516151


namespace height_difference_after_3_years_l516_516595

/-- Conditions for the tree's and boy's growth rates per season. --/
def tree_spring_growth : ℕ := 4
def tree_summer_growth : ℕ := 6
def tree_fall_growth : ℕ := 2
def tree_winter_growth : ℕ := 1

def boy_spring_growth : ℕ := 2
def boy_summer_growth : ℕ := 2
def boy_fall_growth : ℕ := 0
def boy_winter_growth : ℕ := 0

/-- Initial heights. --/
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24

/-- Length of each season in months. --/
def season_length : ℕ := 3

/-- Time period in years. --/
def years : ℕ := 3

/-- Prove the height difference between the tree and the boy after 3 years is 73 inches. --/
theorem height_difference_after_3_years :
    let tree_annual_growth := tree_spring_growth * season_length +
                             tree_summer_growth * season_length +
                             tree_fall_growth * season_length +
                             tree_winter_growth * season_length
    let tree_final_height := initial_tree_height + tree_annual_growth * years
    let boy_annual_growth := boy_spring_growth * season_length +
                            boy_summer_growth * season_length +
                            boy_fall_growth * season_length +
                            boy_winter_growth * season_length
    let boy_final_height := initial_boy_height + boy_annual_growth * years
    tree_final_height - boy_final_height = 73 :=
by sorry

end height_difference_after_3_years_l516_516595


namespace probability_win_l516_516868

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l516_516868


namespace minimum_colors_infinite_lattice_l516_516324

theorem minimum_colors_infinite_lattice (n : ℕ) (h : n > 1) :
  ∃ k, (∀ corner : set (ℕ × ℕ), 
    (∀ (i j : ℕ), 
      i < n → j < n-1 → ((i, 0) ∈ corner ∧ (0, j) ∈ corner → i ≠ j)) 
          → ∀ (c : ℕ × ℕ → ℕ), (∀ (i j : ℕ × ℕ), i ≠ j → c i ≠ c j) 
            → c (i, j) ≠ c (0, 0)) 
      → k = n^2 - 1 := 
sorry

end minimum_colors_infinite_lattice_l516_516324


namespace net_sag_calculation_l516_516995

open Real

noncomputable def sag_of_net (m1 m2 h1 h2 x1 : ℝ) : ℝ :=
  let g := 9.81
  let a := 28
  let b := -1.75
  let c := -50.75
  let D := b^2 - 4*a*c
  let sqrtD := sqrt D
  (1.75 + sqrtD) / (2 * a)

theorem net_sag_calculation :
  let m1 := 78.75
  let x1 := 1
  let h1 := 15
  let m2 := 45
  let h2 := 29
  sag_of_net m1 m2 h1 h2 x1 = 1.38 := 
by
  sorry

end net_sag_calculation_l516_516995


namespace polynomials_same_degree_l516_516700

open Polynomial

-- Definitions for the conditions
variables {R : Type*} [CommRing R] [IsDomain R] [IsPrincipalIdealRing R] [IsField R]
variables (P : ℕ → Polynomial R) (n : ℕ) (x : R)

-- Condition: P_i are monic polynomials with real coefficients
def monic_polynomials (P : ℕ → Polynomial ℝ) (n : ℕ) : Prop :=
  ∀ i, i < n → (P i).monic

-- Condition: S_y is the set of real numbers x such that y = P_i(x) for some i
def S (y : ℝ) : Set ℝ :=
  {x : ℝ | ∃ i, i < n ∧ eval x (P i) = y}

-- Condition: |S_{y1}| = |S_{y2}| for any two real numbers y1 and y2
def S_equal_size (P : ℕ → Polynomial ℝ) (n : ℕ) : Prop :=
  ∀ y1 y2 : ℝ, (S P y1 n).card = (S P y2 n).card

-- Theorem statement
theorem polynomials_same_degree (P : ℕ → Polynomial ℝ) (n : ℕ)
  (h1 : monic_polynomials P n)
  (h2 : S_equal_size P n) :
  ∀ i j, i < n → j < n → (P i).degree = (P j).degree :=
  sorry

end polynomials_same_degree_l516_516700


namespace negation_proposition_l516_516116

theorem negation_proposition : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l516_516116


namespace max_value_of_f_l516_516364

theorem max_value_of_f (a b c : ℝ) (h1 : 0 < a)
    (h2 : (λ x => -a * x^2 + (2 * a - b) * x + b - c) (-3) = 0)
    (h3 : (λ x => -a * x^2 + (2 * a - b) * x + b - c) 0 = 0)
    (h4 : (a * 9 - 3 * b + c = -e^6)) :
    (∀ x, f x = (x^2 + 5 * x + 5) / e^x -> -5 ≤ x -> 5 * e^5 ≤ f x) :=
by
  sorry

end max_value_of_f_l516_516364


namespace length_of_PX_l516_516047

-- Define points and their relationships as assumptions
variables {X Z P Q R : Type*}
variables [Point X] [Point Z] [Point P] [Point Q] [Point R]

-- Define lengths of line segments
def XR : ℝ := 3
def PQ : ℝ := 6
def XZ : ℝ := 7

-- Prove that PX = 14 given the conditions
theorem length_of_PX 
  (h1 : ¬(P ∈ (line_through X Z)))
  (h2 : Q ∈ (line_through X Z))
  (h3 : on_line P R (line_through P X))
  (h4 : perpendicular (line_through P Q) (line_through X Z))
  (h5 : perpendicular (line_through X R) (line_through P X))
  (h6 : XR = 3)
  (h7 : PQ = 6)
  (h8 : XZ = 7) 
  : PX = 14 :=
by
  sorry

end length_of_PX_l516_516047


namespace spent_on_accessories_l516_516082

-- Definitions based on the conditions
def original_money : ℕ := 48
def money_on_snacks : ℕ := 8
def money_left_after_purchases : ℕ := (original_money / 2) + 4

-- Proving how much Sid spent on computer accessories
theorem spent_on_accessories : ℕ :=
  original_money - (money_left_after_purchases + money_on_snacks) = 12 :=
by
  sorry

end spent_on_accessories_l516_516082


namespace min_value_inverse_sum_l516_516136

variable (m n : ℝ)
variable (hm : 0 < m)
variable (hn : 0 < n)
variable (b : ℝ) (hb : b = 2)
variable (hline : 3 * m + n = 1)

theorem min_value_inverse_sum : 
  (1 / m + 4 / n) = 7 + 4 * Real.sqrt 3 :=
  sorry

end min_value_inverse_sum_l516_516136


namespace bug_twelfth_move_l516_516956

theorem bug_twelfth_move (Q : ℕ → ℚ)
  (hQ0 : Q 0 = 1)
  (hQ1 : Q 1 = 0)
  (hQ2 : Q 2 = 1/2)
  (h_recursive : ∀ n, Q (n + 1) = 1/2 * (1 - Q n)) :
  let m := 683
  let n := 2048
  (Nat.gcd m n = 1) ∧ (m + n = 2731) :=
by
  sorry

end bug_twelfth_move_l516_516956


namespace union_of_sets_l516_516699

theorem union_of_sets (x : ℕ) (M N : set ℕ) (h1 : M = {0, x}) (h2 : N = {1, 2}) (h3 : M ∩ N = {2}) : M ∪ N = {0, 1, 2} := by
  sorry

end union_of_sets_l516_516699


namespace jafaris_candy_l516_516315

-- Define the conditions
variable (candy_total : Nat)
variable (taquon_candy : Nat)
variable (mack_candy : Nat)

-- Assume the conditions from the problem
axiom candy_total_def : candy_total = 418
axiom taquon_candy_def : taquon_candy = 171
axiom mack_candy_def : mack_candy = 171

-- Define the statement to be proved
theorem jafaris_candy : (candy_total - (taquon_candy + mack_candy)) = 76 :=
by
  -- Proof goes here
  sorry

end jafaris_candy_l516_516315


namespace notification_possible_l516_516766

-- Define the conditions
def side_length : ℝ := 2
def speed : ℝ := 3
def initial_time : ℝ := 12 -- noon
def arrival_time : ℝ := 19 -- 7 PM
def notification_time : ℝ := arrival_time - initial_time -- total available time for notification

-- Define the proof statement
theorem notification_possible :
  ∃ (partition : ℕ → ℝ) (steps : ℕ → ℝ), (∀ k, steps k * partition k < notification_time) ∧ 
  ∑' k, (steps k * partition k) ≤ 6 :=
by
  sorry

end notification_possible_l516_516766


namespace train_length_l516_516503

theorem train_length (speed_km_per_hr : ℝ) (time_min : ℝ) (distance_m : ℝ) (L : ℝ) 
  (h1 : speed_km_per_hr = 108)
  (h2 : time_min = 1)
  (h3 : distance_m = 1800)
  (h4 : 2 * L = distance_m) : 
  L = 900 :=
by
  have speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600),
  have speed_correct : speed_m_per_s = 30, by
    calc
      speed_km_per_hr * (1000 / 3600) 
      = 108 * (1000 / 3600) : by rw[h1]
      ... 
      = 108 * (5 / 18) : by norm_num
      ... 
      = 30 : by norm_num,
  have time_seconds : ℝ := time_min * 60,
  have time_correct : time_seconds = 60, by
    calc time_min * 60 
    = 1 * 60 : by rw[h2]
    ... 
    = 60 : by norm_num,
  have distance_correct : distance_m = speed_m_per_s * time_seconds, by
    calc distance_m 
    = 30 * 60 : by rw[speed_correct, time_correct]
    ... 
    = 1800 : by norm_num,
  exact sorry

end train_length_l516_516503


namespace inspectors_ratio_l516_516957

-- Define the variables involved
variables {x m a b c : ℝ}

-- Define the conditions and equations based on the problem statement
-- Define the production rate conditions
constant h_prod_1 : m = 3 * x
constant h_prod_2 : 18 * x + 3 * m = 6 * a * c
constant h_prod_3 : 2 * (7 / 4 * x) + 2 * m = 2 * b * c
constant h_prod_4 : 6 * (8 / 3 * x) + m = 4 * b * c

-- Prove the required ratio
theorem inspectors_ratio (h_prod1 : m = 3 * x) (h_prod2 : 18 * x + 3 * m = 6 * a * c)
  (h_prod3 : 2 * (7 / 4 * x) + 2 * m = 2 * b * c) (h_prod4 : 6 * (8 / 3 * x) + m = 4 * b * c) :
  a / b = 18 / 19 :=
by {
  -- The proof would go here
  sorry
}

end inspectors_ratio_l516_516957


namespace number_of_dice_l516_516958

theorem number_of_dice (n : ℕ) (h : (1 / 6 : ℝ) ^ (n - 1) = 0.0007716049382716049) : n = 5 :=
sorry

end number_of_dice_l516_516958


namespace min_n_eq_9_l516_516394

/-- In four-dimensional space, the distance between points A and B is defined as 
  d(A, B) = sqrt(sum (i = 1 to 4) (a_i - b_i)^2).
  Define the set I = { P(c_1, c_2, c_3, c_4) | c_i = 0 or 1, i = 1, 2, 3, 4 }.

  We want to determine the minimum value of n such that any n-subset of I contains 
  three points forming an equilateral triangle.

  The minimum value of n satisfying the condition is 9.
-/
theorem min_n_eq_9 :
  ∃ (n : ℕ), (∀ (Q : set (vector ℕ 4)), 
    Q.card = n → 
    (∃ (P1 P2 P3 : vector ℕ 4), 
    P1 ∈ Q ∧ P2 ∈ Q ∧ P3 ∈ Q ∧ 
    (P1 - P2).norm = (P2 - P3).norm ∧ (P2 - P3).norm = (P3 - P1).norm)) ∧ 
  n = 9 :=
sorry

end min_n_eq_9_l516_516394


namespace estimated_germination_probability_stable_l516_516218

structure ExperimentData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations

def experimentalData : List ExperimentData := [
  ⟨50, 47⟩,
  ⟨100, 89⟩,
  ⟨200, 188⟩,
  ⟨500, 461⟩,
  ⟨1000, 892⟩,
  ⟨2000, 1826⟩,
  ⟨3000, 2733⟩
]

def germinationFrequency (data : ExperimentData) : ℚ :=
  data.m / data.n

def closeTo (x y : ℚ) (ε : ℚ) : Prop :=
  |x - y| < ε

theorem estimated_germination_probability_stable :
  ∃ ε > 0, ∀ data ∈ experimentalData, closeTo (germinationFrequency data) 0.91 ε :=
by
  sorry

end estimated_germination_probability_stable_l516_516218


namespace polygon_diagonals_l516_516978

theorem polygon_diagonals (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 10) :
    let n := 360 / exterior_angle in
    n * (n - 3) / 2 = 594 :=
by
  -- Proof to be filled in here.
  sorry

end polygon_diagonals_l516_516978


namespace sequence_formula_l516_516402

def seq (n : ℕ) : ℕ := 
  match n with
  | 0     => 1
  | (n+1) => 2 * seq n + 3

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) : 
  seq n = 2^n + 1 - 3 :=
sorry

end sequence_formula_l516_516402


namespace sequence_values_l516_516055

-- Definitions of the sequence and the conditions
def sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then (sequence (n - 1) - 1) else 3 * sequence (n - 2)

-- Initial terms of the sequence are given explicitly
noncomputable def initial_terms : List ℕ :=
  [1, 3, 2, 6, 5, 15, 14]

-- Define x, y, z based on the sequence
def x := 3 * 14
def y := x - 1
def z := 3 * y

-- Theorem stating the exact values of x, y, z
theorem sequence_values : x = 42 ∧ y = 41 ∧ z = 123 :=
by
  unfold x y z
  exact And.intro rfl (And.intro rfl rfl)

end sequence_values_l516_516055


namespace quality_equations_count_l516_516801

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def is_quality_equation (a b c : ℕ) : Prop :=
  a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ (c * c) - a * c - b = 0

def count_quality_equations : ℕ :=
  Set.card { (a, b) | ∃ c, is_quality_equation a b c }

theorem quality_equations_count :
  count_quality_equations = 12 :=
sorry

end quality_equations_count_l516_516801


namespace division_problem_l516_516902

-- Define the involved constants and operations
def expr1 : ℚ := 5 / 2 * 3
def expr2 : ℚ := 100 / expr1

-- Formulate the final equality
theorem division_problem : expr2 = 40 / 3 :=
  by sorry

end division_problem_l516_516902


namespace geom_seq_product_l516_516390

noncomputable def geom_seq (a : ℕ → ℝ) := 
∀ n m: ℕ, ∃ r : ℝ, a (n + m) = a n * r ^ m

theorem geom_seq_product (a : ℕ → ℝ) 
  (h_seq : geom_seq a) 
  (h_pos : ∀ n, 0 < a n) 
  (h_log_sum : Real.log (a 3) + Real.log (a 6) + Real.log (a 9) = 3) : 
  a 1 * a 11 = 100 := 
sorry

end geom_seq_product_l516_516390


namespace g_property_g_400_g_800_l516_516816

noncomputable def g : ℝ+ → ℝ :=
sorry

theorem g_property (x y : ℝ+) : g (x * y) = g x * y :=
sorry

theorem g_400 : g 400 = 2 :=
sorry

theorem g_800 : g 800 = 4 :=
by
  -- Use the given conditions and derive that g(800) = 4
  have h₁ : g 400 = 2 := g_400
  have h₂ : ∀ x y : ℝ+, g (x * y) = g x * y := g_property
  sorry

end g_property_g_400_g_800_l516_516816


namespace same_number_on_four_dice_l516_516188

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516188


namespace eval_expression_solve_inequalities_l516_516232

-- Problem 1: Evaluation of the expression equals sqrt(2)
theorem eval_expression : (1 - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + |Real.sqrt 2 - 1|) = Real.sqrt 2 := 
by sorry

-- Problem 2: Solution set of the inequality system
theorem solve_inequalities (x : ℝ) : 
  ((3 * x + 1) / 2 ≥ (4 * x + 3) / 3 ∧ 2 * x + 7 ≥ 5 * x - 17) ↔ (3 ≤ x ∧ x ≤ 8) :=
by sorry

end eval_expression_solve_inequalities_l516_516232


namespace minimum_value_of_function_l516_516808

theorem minimum_value_of_function (y : ℝ) (hy : y > 0) : 
  3 * y^3 + 6 * y^(-2) ≥ 9 :=
sorry

end minimum_value_of_function_l516_516808


namespace swim_distance_l516_516607

theorem swim_distance (v d : ℝ) (c : ℝ := 2.5) :
  (8 = d / (v + c)) ∧ (8 = 24 / (v - c)) → d = 84 :=
by
  sorry

end swim_distance_l516_516607


namespace no_domino_tiling_possible_l516_516630

-- Definition of the modified chessboard
structure Chessboard :=
  (size : ℕ)
  (removed_corners : Finset (ℕ × ℕ)) -- Set of removed corner coordinates

-- Predicate to check if domino tiling is possible
def domino_tiling_possible (board : Chessboard) : Prop :=
  ∃ (f : (ℕ × ℕ) → (ℕ × ℕ)), -- a mapping function for tiling
    (∀ p, f p ≠ p) ∧ -- no square maps to itself
    (∀ p, (p.1 + p.2) % 2 ≠ ((f p).1 + (f p).2) % 2) ∧ -- adjacent squares of different colors
    (∀ p, (p.1 < board.size ∧ p.2 < board.size ∧ p ∉ board.removed_corners) → f (f p) = p)

-- A specific instance of the modified chessboard
def chessboard_with_removed_corners : Chessboard :=
  { size := 8,
    removed_corners := {(0, 0), (7, 7)}.to_finset }

-- Proof statement
theorem no_domino_tiling_possible :
  ¬ domino_tiling_possible chessboard_with_removed_corners :=
  sorry

end no_domino_tiling_possible_l516_516630


namespace corn_syrup_in_sport_formulation_l516_516784

theorem corn_syrup_in_sport_formulation
  (standard_ratio_flavoring_to_corn_syrup : ℕ := 1)
  (standard_ratio_corn_syrup_to_flavoring : ℕ := 12)
  (standard_ratio_flavoring_to_water : ℕ := 30)
  (sport_ratio_flavoring_to_corn_syrup : ℚ := 3 * (1 : ℚ) / (12 : ℚ))
  (sport_ratio_flavoring_to_water : ℚ := (1 : ℚ) / (2 * 30))
  (water_in_sport_formulation : ℚ := 120) :
  let corn_syrup_in_sport_formulation : ℚ :=
    (sport_ratio_flavoring_to_corn_syrup * water_in_sport_formulation) / sport_ratio_flavoring_to_water
  in corn_syrup_in_sport_formulation = 8 :=
by
  sorry

end corn_syrup_in_sport_formulation_l516_516784


namespace find_TU_squared_l516_516489

variables {P Q R S T U : Type}
variable [metric_space S]

def is_square (PQRS : set S) : Prop :=
∃ (P Q R S : S), 
  (dist P Q = 15) ∧ (dist Q R = 15) ∧ (dist R S = 15) ∧ (dist S P = 15) ∧ 
  (dist P R = dist Q S = 21.2132)  -- diagonal of square with side 15 is 15√2 ≈ 21.2132

def exterior_points_to_square (P Q R S T U : S) : Prop :=
  dist P T = 7 ∧ dist R U = 7 ∧ dist Q T = 17 ∧ dist S U = 17

theorem find_TU_squared {PQRS : set S} (h₁ : is_square PQRS) (h₂ : exterior_points_to_square P Q R S T U) : 
  (dist T U)^2 = 901 :=
sorry  -- Proof omitted for the sake of this task

end find_TU_squared_l516_516489


namespace y_intercept_of_line_b_l516_516826

/-
Conditions:
1. Line b is parallel to the line y = 2x + 4.
2. Line b passes through the point (3, 7).
-/

noncomputable def line1 : ℝ → ℝ := λ x, 2 * x + 4

noncomputable def is_parallel (line1 line2 : ℝ → ℝ) := ∀ x, line1 x - line2 x = constant

noncomputable def point_on_line (line : ℝ → ℝ) (p : ℝ × ℝ) := p.snd = line p.fst

/-
Proof problem:
Prove that the y-intercept of line b is y = 1, given the above conditions.
-/
theorem y_intercept_of_line_b :
  ∃ line_b : ℝ → ℝ, (is_parallel line1 line_b) ∧ (point_on_line line_b (3, 7)) ∧ (line_b 0 = 1) :=
by
  sorry

end y_intercept_of_line_b_l516_516826


namespace derivative_exp_of_complex_l516_516411

theorem derivative_exp_of_complex (x y : ℝ) (z : ℂ) (hz : z = x + iy) :
  deriv (λ z, complex.exp z) z = complex.exp z :=
by
  sorry

end derivative_exp_of_complex_l516_516411


namespace find_f_at_0_l516_516712

variable {f : ℝ → ℝ}

-- Define the conditions as hypotheses in Lean
-- The function y = f(x+1) is even
def is_even_function : Prop := ∀ x : ℝ, f(x+1) = f(-x+1)
-- Given f(2) = 1
def f_at_2 : Prop := f(2) = 1

-- The statement that we need to prove
theorem find_f_at_0 (h1 : is_even_function) (h2 : f_at_2) : f(0) = 1 :=
sorry

end find_f_at_0_l516_516712


namespace find_face_value_l516_516941

-- Given definitions as per the conditions
def TD : ℝ := 189
def R : ℝ := 16
def T : ℝ := 0.75

-- Calculate the Face Value in terms of the given conditions
def FV (TD R T : ℝ) : ℝ := (TD * (100 + R * T)) / (R * T)

-- The theorem statement to prove the face value is Rs. 1764
theorem find_face_value : FV TD R T = 1764 :=
by
  -- Skip the proof here
  sorry

end find_face_value_l516_516941


namespace find_G_16_l516_516439

-- Let G be a polynomial function
variable {G : ℝ → ℝ}

-- Given conditions
def condition1 : Polynomial ℝ
def condition2 : G 8 = 35
def condition3 : ∀ x : ℝ, x^2 + 8*x + 12 ≠ 0 → (G (4 * x) / G (x + 4)) = 16 - (80 * x + 100) / (x^2 + 8 * x + 12)

-- Problem statement
theorem find_G_16 (h1 : condition1)
                  (h2 : condition2)
                  (h3 : condition3) :
  G 16 = 117 := 
sorry

end find_G_16_l516_516439


namespace conic_sections_propositions_l516_516993

theorem conic_sections_propositions :
  let prop1 := (∀ (x : ℝ), 2 * x^2 - 5 * x + 2 = 0 → (x = 1 / 2 ∨ x = 2) ∧ (x = 1 / 2 → x < 1) ∧ (x = 2 → x > 1))
  let prop2 := (∀ (A B P : ℝ × ℝ) (k : ℝ), k > 0 → |A.1 - P.1| - |B.1 - P.1| = k → |A.2 - P.2| = 0 → |B.2 - P.2| = 0 → (|A.1 - B.1| ≥ k))
  let prop3 := (∀ (k : ℝ), 0 < k ∧ k < 4 → (∀ x y : ℝ, k * x^2 + (4 - k) * y^2 = 1 → false))
  let prop4 := (let foci_hyperbola := (sqrt (25 + 9), 0)
                let foci_ellipse := (sqrt (35 - 1), 0)
                in foci_hyperbola = foci_ellipse)
  prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by
  sorry

end conic_sections_propositions_l516_516993


namespace quadrilateral_inequality_l516_516233

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

-- Given a convex quadrilateral ABCD with vertices at (x1, y1), (x2, y2), (x3, y3), and (x4, y4), prove:
theorem quadrilateral_inequality :
  (distance_squared x1 y1 x3 y3 + distance_squared x2 y2 x4 y4) ≤
  (distance_squared x1 y1 x2 y2 + distance_squared x2 y2 x3 y3 +
   distance_squared x3 y3 x4 y4 + distance_squared x4 y4 x1 y1) :=
sorry

end quadrilateral_inequality_l516_516233


namespace ice_cream_stall_difference_l516_516061

theorem ice_cream_stall_difference (d : ℕ) 
  (h1 : ∃ d, 10 + (10 + d) + (10 + 2*d) + (10 + 3*d) + (10 + 4*d) = 90) : 
  d = 4 :=
by
  sorry

end ice_cream_stall_difference_l516_516061


namespace maximize_profit_l516_516262

def cost (x : ℝ) : ℝ :=
if x < 90 then 0.5 * x^2 + 60 * x
else 121 * x + 8100 / x - 2180

def revenue (x : ℝ) : ℝ := 1.2 * x

def profit (x : ℝ) : ℝ := revenue x - (cost x + 5)

theorem maximize_profit : ∃ x : ℝ, 0 < x ∧ profit x = 1500 ∧ ∀ y, 0 < y → profit y ≤ 1500 := sorry

end maximize_profit_l516_516262


namespace proportion_exists_x_l516_516738

theorem proportion_exists_x : ∃ x : ℕ, 1 * x = 3 * 4 :=
by
  sorry

end proportion_exists_x_l516_516738


namespace problem_statement_l516_516842

theorem problem_statement (n : ℕ) (hn : n > 0) : (122 ^ n - 102 ^ n - 21 ^ n) % 2020 = 2019 :=
by
  sorry

end problem_statement_l516_516842


namespace minimum_value_of_function_l516_516809

theorem minimum_value_of_function (y : ℝ) (hy : y > 0) : 
  3 * y^3 + 6 * y^(-2) ≥ 9 :=
sorry

end minimum_value_of_function_l516_516809


namespace base7_addition_correct_l516_516261

-- Define the numbers in base 7
def base7 : ℕ → ℕ := λ n, n

def num1 : ℕ := base7 12
def num2 : ℕ := base7 254
def expected_sum : ℕ := base7 306

-- Define the proof problem to show that the sum of num1 and num2 in base 7 equals expected_sum
theorem base7_addition_correct : (num1 + num2 = expected_sum) :=
sorry

end base7_addition_correct_l516_516261


namespace baron_max_n_l516_516132

theorem baron_max_n : ∃ n : ℕ, (2 ^ (n - 1) ≥ 80) ∧ (n = 7) :=
by {
  use 7,
  split,
  {
    exact Nat.pow_le_pow_of_le_right 2 (7 - 1) (log 80).round,
  },
  {
    exact rfl,
  },
}

end baron_max_n_l516_516132


namespace probability_no_rain_four_days_l516_516516

theorem probability_no_rain_four_days (p : ℚ) (p_rain : ℚ) 
  (h_p_rain : p_rain = 2 / 3) 
  (h_p_no_rain : p = 1 - p_rain) : 
  p ^ 4 = 1 / 81 :=
by
  have h_p : p = 1 / 3, sorry
  rw [h_p],
  norm_num

end probability_no_rain_four_days_l516_516516


namespace no_square_from_vertices_l516_516251

-- Define that the plane is divided into equilateral triangles
def equilateral_triangles (plane : Set Point) : Prop :=
  -- Placeholder definition for the division into equilateral triangles
  sorry

-- Define the notion of vertices of these triangles
def vertex_of_equilateral_triangle (p : Point) : Prop :=
  -- Placeholder definition for vertices within the described setup
  sorry

-- Prove that no four vertices form a square
theorem no_square_from_vertices (plane : Set Point) 
  (h : equilateral_triangles plane) (A B C D : Point) :
  vertex_of_equilateral_triangle A →
  vertex_of_equilateral_triangle B →
  vertex_of_equilateral_triangle C →
  vertex_of_equilateral_triangle D →
  ¬ (is_square A B C D) :=
sorry

end no_square_from_vertices_l516_516251


namespace sum_of_three_digit_positive_integers_l516_516928

noncomputable def sum_of_arithmetic_series (a l n : ℕ) : ℕ :=
  (a + l) / 2 * n

theorem sum_of_three_digit_positive_integers : 
  sum_of_arithmetic_series 100 999 900 = 494550 :=
by
  -- skipping the proof
  sorry

end sum_of_three_digit_positive_integers_l516_516928


namespace remainder_of_9876543210_div_101_l516_516924

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l516_516924


namespace num_palindrome_times_l516_516758

def is_valid_hour (h : Nat) : Prop :=
  0 ≤ h ∧ h < 24

def is_valid_minute (m : Nat) : Prop :=
  0 ≤ m ∧ m < 60

def is_palindrome_time (h m : Nat) : Prop :=
  let h_digits := if h < 10 then [0, h] else [h / 10, h % 10]
  let m_digits := [m / 10, m % 10]
  h_digits = m_digits.reverse

def count_palindrome_times : ℕ :=
  List.length ((List.range 24).product (List.range 60)).filter (λ ⟨h, m⟩ => is_palindrome_time h m)

theorem num_palindrome_times : count_palindrome_times = 60 := by
  sorry

end num_palindrome_times_l516_516758


namespace fourth_rectangle_perimeter_l516_516326

theorem fourth_rectangle_perimeter (P1 P2 P3 P4 : ℕ) (h1 : P1 = 16) (h2 : P2 = 18) (h3 : P3 = 24) 
  (h4 : P1 + P2 = P3 + P4) : P4 = 10 :=
by
  -- Introduce the hypotheses into the context
  rw [h1, h2, h3] at h4,
  -- Simplify the equation
  linarith,
  sorry

end fourth_rectangle_perimeter_l516_516326


namespace value_of_expression_l516_516330

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : a^2 - b^2 - 2 * b = 1 := 
by
  sorry

end value_of_expression_l516_516330


namespace curve_traced_by_expression_is_ellipse_l516_516850

theorem curve_traced_by_expression_is_ellipse (r : ℝ) (θ : ℝ) (r_pos : r = 3) :
  let z := r * complex.exp (complex.I * θ) in
  z^2 + (1 / z)^2 = 82 / 9 * cos (2 * θ) + complex.I * (80 / 9 * sin (2 * θ))
  ↔ ∃ u v : ℝ, u = 82 / 9 * cos (2 * θ) ∧ v = 80 / 9 * sin (2 * θ) ∧
  (u / (82 / 9))^2 + (v / (80 / 9))^2 = 1 :=
sorry

end curve_traced_by_expression_is_ellipse_l516_516850


namespace no_rain_four_days_l516_516511

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l516_516511


namespace calc_expression_l516_516279

theorem calc_expression :
  (-2^2 - (Real.cbrt (-8) + 8) / Real.sqrt ((-6)^2) - |Real.sqrt 7 - 3|) = -8 + Real.sqrt 7 :=
  by sorry

end calc_expression_l516_516279


namespace quadrant_of_angle_l516_516372

theorem quadrant_of_angle (θ : ℝ) (h1 : tan θ * sin θ < 0) (h2 : tan θ * cos θ > 0) : 
    π / 2 < θ ∧ θ < π :=
sorry

end quadrant_of_angle_l516_516372


namespace hyperbola_foci_problem_l516_516032

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

noncomputable def foci_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def foci_2 : ℝ × ℝ := (Real.sqrt 5, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def vector (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v1.1 + v2.2 * v2.2

noncomputable def orthogonal (P : ℝ × ℝ) : Prop :=
  dot_product (vector P foci_1) (vector P foci_2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def required_value (P : ℝ × ℝ) : ℝ :=
  distance P foci_1 * distance P foci_2

theorem hyperbola_foci_problem (P : ℝ × ℝ) : 
  point_on_hyperbola P → orthogonal P → required_value P = 2 := 
sorry

end hyperbola_foci_problem_l516_516032


namespace tan_alpha_minus_pi_over_4_sin_beta_l516_516706

variable (α β : ℝ)

-- Given conditions
def given_conditions : Prop :=
  (α ∈ (Real.pi/2, Real.pi)) ∧
  (β ∈ (0, Real.pi/2)) ∧
  (Real.sin α = 4/5) ∧
  (Real.cos (α - β) = Real.sqrt 10 / 10)

-- Prove tan(α - π/4) = 7
theorem tan_alpha_minus_pi_over_4 : given_conditions α β → Real.tan (α - Real.pi/4) = 7 := by
  sorry

-- Prove sin β = 13√10/50
theorem sin_beta : given_conditions α β → Real.sin β = (13 * Real.sqrt 10) / 50 := by
  sorry

end tan_alpha_minus_pi_over_4_sin_beta_l516_516706


namespace cirrus_clouds_count_l516_516881

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l516_516881


namespace evaluate_cube_root_power_l516_516652

theorem evaluate_cube_root_power (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^(3 : ℝ)) : (cbrt a)^12 = b^12 :=
by
  sorry

example : evaluate_cube_root_power 8 2 4096 (by rfl)

end evaluate_cube_root_power_l516_516652


namespace inequality_f_l516_516732

noncomputable def f (x : ℝ) : ℝ := 1 / x^2
def a : ℝ := Real.log 4 / Real.log 5
def b : ℝ := Real.log 3 / Real.log 5
def c : ℝ := 2^0.2

theorem inequality_f (h_decreasing : ∀ {x y : ℝ}, 0 < x → x < y → f(y) < f(x))
  (h_order : 0 < b ∧ b < a ∧ a < 1 ∧ 1 < c) :
  f(c) < f(a) ∧ f(a) < f(b) :=
by
  sorry

end inequality_f_l516_516732


namespace inequality_holds_for_positive_real_numbers_l516_516434

theorem inequality_holds_for_positive_real_numbers 
  (a b : ℝ) (h_pos_a : 0 < a) (h_a_lt_b : a < b) (n : ℕ) (h_pos_n : 0 < n) 
  (x : Fin n → ℝ) (h_x : ∀ i, a ≤ x i ∧ x i ≤ b) : 
  (∑ i in Finset.range n, abs (x i - x (i + 1 % n))) ≤ (2 * (b - a) / (b + a)) * (∑ i in Finset.range n, x i) :=
sorry

end inequality_holds_for_positive_real_numbers_l516_516434


namespace same_face_probability_l516_516180

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516180


namespace spent_on_computer_accessories_l516_516080

theorem spent_on_computer_accessories :
  ∀ (x : ℕ), (original : ℕ) (snacks : ℕ) (remaining : ℕ),
  original = 48 →
  snacks = 8 →
  remaining = 4 + original / 2 →
  original - (x + snacks) = remaining →
  x = 12 :=
by
  intros x original snacks remaining
  intro h_original
  intro h_snacks
  intro h_remaining
  intro h_spent
  sorry

end spent_on_computer_accessories_l516_516080


namespace joey_learn_swimming_time_l516_516609

variable (days_vacation_joey : ℚ)
variable (time_learned_jon_smith : ℚ)
variable (time_learned_alexa : ℚ)

-- Alexa was on vacation for 3/4ths of the time it took Ethan to learn 12 fencing tricks
h1 : time_learned_alexa = 3/4 * time_learned_jon_smith

-- Joey spent half as much time as Ethan spent to learn swimming
h2 : days_vacation_joey = (1/2) * time_learned_jon_smith

-- Alexa spent a week and 2 days on vacation
h3 : time_learned_alexa = 9

theorem joey_learn_swimming_time : days_vacation_joey = 6 := by
  sorry

end joey_learn_swimming_time_l516_516609


namespace max_product_60_l516_516682

-- Define the set of numbers we have
def numberSet : Set Int := {-3, -4, -1, 2, 5}

-- Define a function to calculate the product of three integers
def prod3 (a b c: Int) : Int := a * b * c

-- Define the maximal product
def maxProduct : Int :=
  let products := {prod3 x y z | x y z ∈ numberSet}
  Set.max products

-- Given the set of numbers {-3, -4, -1, 2, 5}
-- The maximum product achieved by taking any triplet is 60

theorem max_product_60 : maxProduct = 60 := sorry

end max_product_60_l516_516682


namespace rationalize_and_sum_l516_516472

theorem rationalize_and_sum (A B C : ℤ : C > 0 ∧ ¬ ∃ p: ℤ, p^3 ∣ B) :
  (3 : ℝ) / (2 * real.cbrt (5 : ℝ)) = (A : ℝ) * real.cbrt (B : ℝ) / (C : ℝ) →
  A + B + C = 38 :=
sorry

end rationalize_and_sum_l516_516472


namespace shaina_keeps_chocolate_l516_516022

theorem shaina_keeps_chocolate :
  let total_chocolate := (60 : ℚ) / 7
  let number_of_piles := 5
  let weight_per_pile := total_chocolate / number_of_piles
  let given_weight_back := (1 / 2) * weight_per_pile
  let kept_weight := weight_per_pile - given_weight_back
  kept_weight = 6 / 7 :=
by
  sorry

end shaina_keeps_chocolate_l516_516022


namespace same_face_probability_l516_516181

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l516_516181


namespace incorrect_conclusion_l516_516246

noncomputable def normal_density_part_one (x : ℝ) : ℝ := 
  (1 / (10 * real.sqrt (2 * real.pi))) * real.exp (-(x - 110)^2 / 200)

def mean_part_two : ℝ := 115
def variance_part_two : ℝ := 56.25
def stddev_part_two : ℝ := real.sqrt variance_part_two

axiom part_two_normal_distribution : ∃ ξ : ℝ, ξ ∼ real.normalDistribution mean_part_two stddev_part_two

axiom normal_probabilities_stddevs (μ : ℝ) (σ : ℝ) :
  ∀ (ξ : ℝ), ξ ∼ real.normalDistribution μ σ ->
  P(μ - σ < ξ ∧ ξ < μ + σ) = 0.6826 ∧
  P(μ - 2 * σ < ξ ∧ ξ < μ + 2 * σ) = 0.9544 ∧
  P(μ - 3 * σ < ξ ∧ ξ < μ + 3 * σ) = 0.9974

def students_part_one : ℕ := 750
def students_part_two : ℕ := 750

def mean_part_one : ℝ := 110
def stddev_part_one : ℝ := 10

theorem incorrect_conclusion (A_part_one: x ∼ real.normalDistribution 110 (10^2∕R)) :
  ∀ x, normal_density_part_one x = (1 / (10 * real.sqrt (2 * real.pi))) * real.exp (-(x - mean_part_one) ^ 2 / (2 * stddev_part_one ^ 2)) :=
sorry

end incorrect_conclusion_l516_516246


namespace probability_not_greater_than_4_arithmetic_sequence_l516_516886

theorem probability_not_greater_than_4_arithmetic_sequence :
  let a1 := 12
  let d := -2
  let n := 16
  let terms := List.range n + 1 -- List of indices from 1 to 16
  let sequence := terms.map (λ i => a1 + (i - 1) * d)
  let favorable := sequence.filter (λ i => i ≤ 4)
  (favorable.length : ℚ) / n = 3 / 4 :=
by
  sorry

end probability_not_greater_than_4_arithmetic_sequence_l516_516886


namespace macy_hit_ball_50_times_l516_516453

-- Definitions and conditions
def token_pitches : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def piper_hits : ℕ := 55
def missed_pitches : ℕ := 315

-- Calculation based on conditions
def total_pitches : ℕ := (macy_tokens + piper_tokens) * token_pitches
def total_hits : ℕ := total_pitches - missed_pitches
def macy_hits : ℕ := total_hits - piper_hits

-- Prove that Macy hit 50 times
theorem macy_hit_ball_50_times : macy_hits = 50 := 
by
  sorry

end macy_hit_ball_50_times_l516_516453


namespace oil_added_to_mixture_l516_516829

theorem oil_added_to_mixture (x : ℝ) :
  let initial_mixture_weight : ℝ := 8
      initial_oil_weight : ℝ := 0.2 * initial_mixture_weight
      initial_B_weight : ℝ := 0.8 * initial_mixture_weight
      final_total_weight : ℝ := 14 + x
      final_oil_weight : ℝ := 1.6 + x + 1.2
      final_B_weight : ℝ := 11.2 in
  final_B_weight = 0.7 * final_total_weight → x = 2 :=
begin
  sorry
end

end oil_added_to_mixture_l516_516829


namespace arithmetic_sequence_sum_first_8_terms_l516_516777

noncomputable def sum_of_first_8_terms (a : ℕ → ℝ) :=
  ∑ i in Finset.range 8, a (i + 1)

theorem arithmetic_sequence_sum_first_8_terms
  (a : ℕ → ℝ)
  (h₁ : a 3 = 5)
  (h₂ : a 4 + a 8 = 22) :
  sum_of_first_8_terms a = 64 :=
sorry

end arithmetic_sequence_sum_first_8_terms_l516_516777


namespace smallest_base_b_l516_516546

theorem smallest_base_b (k : ℕ) (hk : k = 7) : ∃ (b : ℕ), b = 64 ∧ b^k > 4^20 := by
  sorry

end smallest_base_b_l516_516546


namespace ellipse_minor_axis_length_l516_516342

theorem ellipse_minor_axis_length
  (semi_focal_distance : ℝ)
  (eccentricity : ℝ)
  (semi_focal_distance_eq : semi_focal_distance = 2)
  (eccentricity_eq : eccentricity = 2 / 3) :
  ∃ minor_axis_length : ℝ, minor_axis_length = 2 * Real.sqrt 5 :=
by
  sorry

end ellipse_minor_axis_length_l516_516342


namespace remainder_9876543210_mod_101_l516_516922

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l516_516922


namespace count_valid_arrangements_l516_516772

theorem count_valid_arrangements : 
  ∃ (n : ℕ), n = 96 ∧
  (let digits := [3, 6, 7, 2, 0] in 
  ∃ (arrangements : list (list ℕ)), 
    arrangements = list.permutations digits ∧
    ∀ (num : list ℕ), num ∈ arrangements → (num.head ≠ 0) → n = list.length arrangements) := by
  sorry

end count_valid_arrangements_l516_516772


namespace sid_spent_on_computer_accessories_l516_516077

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l516_516077


namespace cube_root_power_l516_516656

theorem cube_root_power (a : ℝ) (h : a = 8) : (a^(1/3))^12 = 4096 := by
  rw [h]
  have h2 : 8 = 2^3 := rfl
  rw h2
  sorry

end cube_root_power_l516_516656


namespace sum_of_first_n_odd_numbers_l516_516622

theorem sum_of_first_n_odd_numbers (n : ℕ) : 
    (∑ i in Finset.range (n + 1), (2 * i + 1)) = (n + 1) ^ 2 := 
by 
  sorry

end sum_of_first_n_odd_numbers_l516_516622


namespace train_A_start_time_l516_516538

theorem train_A_start_time :
  let distance := 155 -- km
  let speed_A := 20 -- km/h
  let speed_B := 25 -- km/h
  let start_B := 8 -- a.m.
  let meet_time := 11 -- a.m.
  let travel_time_B := meet_time - start_B -- time in hours for train B from 8 a.m. to 11 a.m.
  let distance_B := speed_B * travel_time_B -- distance covered by train B
  let distance_A := distance - distance_B -- remaining distance covered by train A
  let travel_time_A := distance_A / speed_A -- time for train A to cover its distance
  let start_A := meet_time - travel_time_A -- start time for train A
  start_A = 7 := by
  sorry

end train_A_start_time_l516_516538


namespace eighty_percentile_is_10_8_l516_516519

-- Initial data set and relevant definitions
def data_set : List ℝ := [10.2, 9.7, 10.8, 9.1, 8.9, 8.6, 9.8, 9.6, 9.9, 11.2, 10.6, 11.7]

def percentile (p : ℝ) (data : List ℝ) : ℝ :=
  let sorted_data := data.qsort (λ a b => a < b)
  let pos := (p / 100) * (sorted_data.length : ℝ)
  if pos.frac = 0 then
    sorted_data.nth_le (nat.floor pos - 1) sorry
  else
    sorted_data.nth_le (nat.floor pos) sorry

theorem eighty_percentile_is_10_8 :
  percentile 80 data_set = 10.8 := sorry

end eighty_percentile_is_10_8_l516_516519


namespace angle_slope_condition_l516_516715

theorem angle_slope_condition (α k : Real) (h₀ : k = Real.tan α) (h₁ : 0 ≤ α ∧ α < Real.pi) : 
  (α < Real.pi / 3) → (k < Real.sqrt 3) ∧ ¬((k < Real.sqrt 3) → (α < Real.pi / 3)) := 
sorry

end angle_slope_condition_l516_516715


namespace part1_maximum_value_part2_find_k_l516_516722

/-- Part 1: Given condition of the ellipse and point A, prove the maximum value of the expression -/
theorem part1_maximum_value
  (P : ℝ × ℝ)
  (A : ℝ × ℝ := (√2, 2))
  (ellipse : ℝ × ℝ → Prop := λ P, (P.1^2 / 3) + P.2^2 = 1)
  (focus1 focus2 : ℝ × ℝ)
  (h_f1 : focus1 = (-√3, 0))
  (h_f2 : focus2 = (√3, 0))
  (h_P_on_ellipse : ellipse P) :
  |(P.1 - A.1, P.2 - A.2)| + |(P.1 - focus1.1, P.2 - focus1.2)| ≤ 2 + 2 * √3 := sorry

/-- Part 2: Given condition of the ellipse and line, find the value of k -/
theorem part2_find_k
  (ellipse : ℝ × ℝ → Prop := λ P, (P.1^2 / 3) + P.2^2 = 1)
  (line : ℝ → ℝ := λ x, k * x + √2)
  (O : ℝ × ℝ := (0, 0))
  (A B : ℝ × ℝ)
  (h_A : ellipse A ∧ A.2 = k * A.1 + √2)
  (h_B : ellipse B ∧ B.2 = k * B.1 + √2)
  (h_intersect : A ≠ B)
  (h_dot_product : (A.1 * B.1) + (A.2 * B.2) = 1) :
  k = √6 / 3 ∨ k = -√6 / 3 := sorry

end part1_maximum_value_part2_find_k_l516_516722


namespace remainder_of_9876543210_div_101_l516_516927

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l516_516927


namespace john_pays_2400_per_year_l516_516415

theorem john_pays_2400_per_year
  (hours_per_month : ℕ)
  (minutes_per_hour : ℕ)
  (songs_per_minute : ℕ)
  (cost_per_song : ℕ)
  (months_per_year : ℕ)
  (H1 : hours_per_month = 20)
  (H2 : minutes_per_hour = 60)
  (H3 : songs_per_minute = 3)
  (H4 : cost_per_song = 50)
  (H5 : months_per_year = 12) :
  let minutes_per_month := hours_per_month * minutes_per_hour,
      songs_per_month := minutes_per_month / songs_per_minute,
      cost_per_month := songs_per_month * cost_per_song in
  cost_per_month * months_per_year = 2400 := by
  sorry

end john_pays_2400_per_year_l516_516415


namespace coordinate_sum_of_point_on_graph_l516_516380

theorem coordinate_sum_of_point_on_graph (g : ℕ → ℕ) (h : ℕ → ℕ)
  (h1 : g 2 = 8)
  (h2 : ∀ x, h x = 3 * (g x) ^ 2) :
  2 + h 2 = 194 :=
by
  sorry

end coordinate_sum_of_point_on_graph_l516_516380


namespace inequality_abc_l516_516836

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l516_516836


namespace solution_set_of_inequality_l516_516235

theorem solution_set_of_inequality (x : ℝ) : 
  |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 :=
sorry

end solution_set_of_inequality_l516_516235


namespace tina_more_than_katya_l516_516426

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end tina_more_than_katya_l516_516426


namespace max_perimeter_of_polygons_l516_516891

noncomputable def largest_possible_perimeter (sides1 sides2 sides3 : Nat) (len : Nat) : Nat :=
  (sides1 + sides2 + sides3) * len

theorem max_perimeter_of_polygons
  (a b c : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : 180 * (a - 2) / a + 180 * (b - 2) / b + 180 * (c - 2) / c = 360)
  (h5 : ∃ (p : ℕ), ∃ q : ℕ, (a = p ∧ c = p ∧ a = q ∨ a = q ∧ b = p ∨ b = q ∧ c = p))
  : largest_possible_perimeter a b c 2 = 24 := 
sorry

end max_perimeter_of_polygons_l516_516891


namespace product_even_or_odd_is_even_l516_516066

noncomputable theory

variable {X : Type} [has_neg X] [has_mul X]

-- Definitions for even and odd functions
def is_even (f : X → X) : Prop := ∀ x, f (-x) = f x
def is_odd (f : X → X) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating that the product of two even or two odd functions is even
theorem product_even_or_odd_is_even (f g : X → X) 
  (h_f : is_even f ∨ is_odd f) (h_g : is_even g ∨ is_odd g) : is_even (λ x, f x * g x) := 
by
  sorry

end product_even_or_odd_is_even_l516_516066


namespace probability_green_or_yellow_l516_516201

def total_marbles (green yellow red blue : Nat) : Nat :=
  green + yellow + red + blue

def marble_probability (green yellow red blue : Nat) : Rat :=
  (green + yellow) / (total_marbles green yellow red blue)

theorem probability_green_or_yellow :
  let green := 4
  let yellow := 3
  let red := 4
  let blue := 2
  marble_probability green yellow red blue = 7 / 13 := by
  sorry

end probability_green_or_yellow_l516_516201


namespace exradius_sum_eq_p_squared_l516_516837

-- Given conditions
variables {a b c p r_a r_b r_c : ℝ}

-- Definitions for the conditions
def exradius_product_1 := r_a * r_b = p * (p - c)
def exradius_product_2 := r_b * r_c = p * (p - a)
def exradius_product_3 := r_c * r_a = p * (p - b)
def semi_perimeter := p = (a + b + c) / 2

-- The theorem to prove
theorem exradius_sum_eq_p_squared 
  (h1 : exradius_product_1) 
  (h2 : exradius_product_2) 
  (h3 : exradius_product_3) 
  (h4 : semi_perimeter) : 
  r_a * r_b + r_b * r_c + r_c * r_a = p^2 :=
by
  sorry

end exradius_sum_eq_p_squared_l516_516837


namespace positive_terms_in_sequence_2015_l516_516705

-- Definition for S_n as given in the problem
def S (n : ℕ) : ℝ := ∑ i in finset.range n, real.cos (i * real.pi / 8)

-- The problem statement
theorem positive_terms_in_sequence_2015 : 
  (finset.filter (λ n, S (n + 1) > 0) (finset.range 2015)).card = 756 := 
sorry

end positive_terms_in_sequence_2015_l516_516705


namespace pie_chart_probability_l516_516250

theorem pie_chart_probability
  (P_W P_X P_Z : ℚ)
  (h_W : P_W = 1/4)
  (h_X : P_X = 1/3)
  (h_Z : P_Z = 1/6) :
  1 - P_W - P_X - P_Z = 1/4 :=
by
  -- The detailed proof steps are omitted as per the requirement.
  sorry

end pie_chart_probability_l516_516250


namespace intersection_S_T_l516_516823

def set_S : Set ℝ := { x | abs x < 5 }
def set_T : Set ℝ := { x | x^2 + 4*x - 21 < 0 }

theorem intersection_S_T :
  set_S ∩ set_T = { x | -5 < x ∧ x < 3 } :=
sorry

end intersection_S_T_l516_516823


namespace smaller_square_side_sum_l516_516488

theorem smaller_square_side_sum 
  (ABCD : ℝ)
  (side_length_ABCD : ABCD = 2)
  (E F : ℝ → ℝ)
  (EF_length : E(2) = 0 ∧ F(0) = 1)
  (right_triangle_AEF : ∀ (x y: ℝ), x^2 + y^2 = 5):
  let s := (4 * (sqrt 5 - 2)) / 3,
  ∃ (a b c : ℤ), s = (a + sqrt b) / c ∧ a + b + c = 15 := 
by
  sorry

end smaller_square_side_sum_l516_516488


namespace probability_of_same_number_on_four_dice_l516_516162

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516162


namespace probability_win_l516_516869

theorem probability_win (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose) = 3 / 8 :=
by
  rw [h]
  norm_num

end probability_win_l516_516869


namespace find_constants_and_min_value_l516_516729

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x * Real.log x
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b * Real.log x + b
noncomputable def g (a b x : ℝ) := f a b x - Real.exp 1 * x^2

theorem find_constants_and_min_value :
  (∀ (a b : ℝ),
    -- Condition for the derivative at x = 1 and the given tangent line slope
    (f' a b 1 = 2 * Real.exp 1) ∧
    -- Condition for the function value at x = 1
    (f a b 1 = Real.exp 1) →
    -- Expected results for a and b
    (a = 1 ∧ b = Real.exp 1)) ∧

  -- Evaluating the minimum value of the function g(x)
  (∀ (x : ℝ), 0 < x →
    -- Given the minimum occurs at x = 1
    g 1 (Real.exp 1) 1 = 0 ∧
    (∀ (x : ℝ), 0 < x →
      (g 1 (Real.exp 1) x ≥ 0))) :=
sorry

end find_constants_and_min_value_l516_516729


namespace initial_candles_count_l516_516266

section

variable (C : ℝ)
variable (h_Alyssa : C / 2 = C / 2)
variable (h_Chelsea : C / 2 - 0.7 * (C / 2) = 6)

theorem initial_candles_count : C = 40 := 
by sorry

end

end initial_candles_count_l516_516266


namespace quadratic_has_real_root_in_interval_l516_516065

theorem quadratic_has_real_root_in_interval (a b : ℝ) : 
  ∃ x ∈ set.Ioo 0 1, 3 * a * x^2 + 2 * b * x - (a + b) = 0 :=
sorry

end quadratic_has_real_root_in_interval_l516_516065


namespace abc_sum_is_32_l516_516748

theorem abc_sum_is_32 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c = 31) (h5 : b * c + a = 31) (h6 : a * c + b = 31) : 
  a + b + c = 32 := 
by
  -- Proof goes here
  sorry

end abc_sum_is_32_l516_516748


namespace swim_meeting_count_l516_516145

-- Definitions for the problem conditions
def pool_length : ℕ := 100
def rate_swimmer1 : ℕ := 4
def rate_swimmer2 : ℕ := 5
def time_minutes : ℕ := 15
def time_seconds : ℕ := time_minutes * 60

-- Problem statement
theorem swim_meeting_count :
  let swimmer1_time := pool_length / rate_swimmer1,
      swimmer2_time := pool_length / rate_swimmer2,
      round_trip_time1 := swimmer1_time * 2,
      round_trip_time2 := swimmer2_time * 2,
      lcm_time := Nat.lcm round_trip_time1 round_trip_time2,
      meet_interval := lcm_time / swimmer1_time,
      meets_in_lcm_time := lcm_time / meet_interval,
      total_meets := (time_seconds / lcm_time) * meets_in_lcm_time
  in total_meets = 36 := by
  sorry

end swim_meeting_count_l516_516145


namespace sum_of_tan_squared_S_eq_sqrt2_l516_516432

noncomputable def sum_of_tan_squared_over_S : ℝ :=
  let S := { x : ℝ | 0 < x ∧ x < π / 2 ∧ 
                  (∃ (a b c : ℝ), (a = Real.sin x ∨ b = Real.sin x ∨ c = Real.sin x) ∧
                                   (a = Real.cos x ∨ b = Real.cos x ∨ c = Real.cos x) ∧
                                   (a = Real.tan x ∨ b = Real.tan x ∨ c = Real.tan x) ∧
                                   (a^2 + b^2 = c^2)) } in
  ∑ x in S, Real.tan x ^ 2

theorem sum_of_tan_squared_S_eq_sqrt2 :
  sum_of_tan_squared_over_S = Real.sqrt 2 := 
  sorry

end sum_of_tan_squared_S_eq_sqrt2_l516_516432


namespace correct_option_l516_516268
-- We'll import the necessary libraries.

-- Define the conditions given in the problem.
variables (b A C : ℝ) (a1 c1 A1 : ℝ) (a2 c2 A2 : ℝ) (a3 c3 A3 : ℝ)

-- Define the conditions of the options in Lean
def option_A : Prop := b = 20 ∧ A = 45 ∧ C = 80
def option_B : Prop := a1 = 30 ∧ c1 = 28 ∧ A1 = 60
def option_C : Prop := a2 = 14 ∧ c2 = 16 ∧ A2 = 45
def option_D : Prop := a3 = 12 ∧ c3 = 10 ∧ A3 = 120

-- Define the theorem to express which option has two solutions
theorem correct_option 
  (hA : option_A)
  (hB : option_B)
  (hC : option_C)
  (hD : option_D) :
  (∃ (α : Triangle), α.solutions(a2, c2, A2) = 2) ∧ 
  (∀ (αβ : Triangle), αβ ≠ option_C → αβ.solutions(a2, c2, A2) ≠ 2) := 
sorry

end correct_option_l516_516268


namespace lower_interest_percentage_l516_516241

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem lower_interest_percentage :
  ∃ P : ℝ, (simple_interest 8400 15 2) - (simple_interest 8400 P 2) = 840 ∧ P = 10 :=
by
  have h1 : simple_interest 8400 15 2 = 2520 := by norm_num
  have h2 : (simple_interest 8400 P 2) = 168 * P := by sorry
  sorry

end lower_interest_percentage_l516_516241


namespace math_teachers_probability_expectation_of_math_teachers_l516_516240

-- Definition of the conditions
def total_teachers := 6
def chinese_teachers := 2
def math_teachers := 2
def english_teachers := 2
def selected_teachers := 3

-- Part (1): Prove that the probability of selecting more math teachers than Chinese teachers is 3/10
theorem math_teachers_probability : 
  let total_ways := Nat.choose total_teachers selected_teachers in
  let ways_A1 := Nat.choose math_teachers 1 * Nat.choose english_teachers 2 in
  let ways_A2 := Nat.choose math_teachers 2 * Nat.choose (total_teachers - math_teachers) 1 in
  let probability_A1 := (ways_A1 : ℝ) / total_ways in
  let probability_A2 := (ways_A2 : ℝ) / total_ways in
  (probability_A1 + probability_A2) = 3 / 10 :=
by
  sorry

-- Part (2): Prove that the expectation of the number of math teachers selected is 1
theorem expectation_of_math_teachers :
  let total_ways := Nat.choose total_teachers selected_teachers in
  let P_X0 := (Nat.choose math_teachers 0 * Nat.choose (total_teachers - math_teachers) (selected_teachers - 0) : ℝ) / total_ways in
  let P_X1 := (Nat.choose math_teachers 1 * Nat.choose (total_teachers - math_teachers) (selected_teachers - 1) : ℝ) / total_ways in
  let P_X2 := (Nat.choose math_teachers 2 * Nat.choose (total_teachers - 2) (selected_teachers - 2) : ℝ) / total_ways in
  (0 * P_X0 + 1 * P_X1 + 2 * P_X2) = 1 :=
by
  sorry

end math_teachers_probability_expectation_of_math_teachers_l516_516240


namespace sum_of_angles_is_55_l516_516466

noncomputable def arc_BR : ℝ := 60
noncomputable def arc_RS : ℝ := 50
noncomputable def arc_AC : ℝ := 0
noncomputable def arc_BS := arc_BR + arc_RS
noncomputable def angle_P := (arc_BS - arc_AC) / 2
noncomputable def angle_R := arc_AC / 2
noncomputable def sum_of_angles := angle_P + angle_R

theorem sum_of_angles_is_55 :
  sum_of_angles = 55 :=
by
  sorry

end sum_of_angles_is_55_l516_516466


namespace warden_issued_16_citations_l516_516249

-- Declarations of variables
variable (c_littering c_offleashDogs c_parkingFines total_citations : ℕ)

-- Conditions
def condition_1 : c_littering = 4 := by sorry
def condition_2 : c_offleashDogs = c_littering := by sorry
def condition_3 : c_parkingFines = 2 * c_littering := by sorry
def condition_4 : total_citations = c_littering + c_offleashDogs + c_parkingFines := by sorry

-- The theorem to be proved that total citations are 16
theorem warden_issued_16_citations' : total_citations = 16 := by
  rw [condition_1, condition_2, condition_3, condition_4]
  sorry

end warden_issued_16_citations_l516_516249


namespace dice_same_number_probability_l516_516198

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516198


namespace area_triangle_MON_l516_516569

variables (A B C D O P M N : Type)
variables [Segment A B] [Segment A C] [Segment A D] [Segment B C] [Segment B D] [Segment C D]
variables [Segment B P] [Segment C P] [Segment P M] [Segment P N]

-- Definitions of the points and conditions
def is_trapezoid (A B C D : Type) : Prop := true
def has_area (fig : Type) (area : ℚ) : Prop := true
def midpoint (B : Type) (A D : Type) : Type := P
def intersect_diagonal (A B C D O : Type) : Prop := true
def intersect_on_segments (mid : Type) (A : Type) (B : Type) (M N : Type) : Prop := true
def base_relations (AD BC : Type) (ratio : ℚ) : Prop := true

-- Given conditions for the problem
axiom trapezoid_ABCD : is_trapezoid A B C D
axiom area_ABCD : has_area (trapezoid_ABCD) 405
axiom midpoint_P : midpoint B A D = P
axiom diagonal_intersection_O : intersect_diagonal A B C D O
axiom segment_intersections : intersect_on_segments P A B M N
axiom base_relation : base_relations A D B C (2 : ℚ)

-- Theorem statement
theorem area_triangle_MON :
  ∃ (area : ℚ), area = 45 / 4 ∨ area = 36 / 5 := sorry

end area_triangle_MON_l516_516569


namespace calculate_fg2_l516_516804

def f (x : ℝ) : ℝ := 3 * real.sqrt x + 15 / real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

theorem calculate_fg2 : f (g 2) = 18 :=
by
  sorry

end calculate_fg2_l516_516804


namespace find_x3_l516_516896

theorem find_x3 
  (x1 x2 : ℝ) 
  (h1 : 0 < x1) 
  (h2 : x1 < x2) 
  (x1_val : x1 = 1) 
  (x2_val : x2 = 16) :
  ∃ x3 : ℝ, f(x3) = 3 ∧ x3 = 9 :=
by
  let f := λ x : ℝ, real.sqrt x
  sorry

end find_x3_l516_516896


namespace min_modulus_of_quadratic_equation_l516_516497

noncomputable def min_modulus_of_complex_coeff (m : ℂ) : ℝ :=
  if (has_real_roots : (m^2 - 4 - 8 * I).im = 0) then |m| else ∞

theorem min_modulus_of_quadratic_equation 
  (h : ∀ x : ℂ, x^2 + m*x + (1 + 2 * I) = 0 → x.im = 0) : 
  min_modulus_of_complex_coeff m = sqrt (2 + 2 * sqrt 5) :=
sorry

end min_modulus_of_quadratic_equation_l516_516497


namespace b_alone_completes_in_20_days_l516_516938

variable (work : Type) [CommRing work]

-- Definitions
def combined_work_rate (A B : work → ℝ) := (A + B) 1
def a_work_rate (A : work → ℝ) := A 1
def b_work_rate (B : work → ℝ) := B 1

-- Theorem statement
theorem b_alone_completes_in_20_days
  (A B : work → ℝ)
  (h1 : combined_work_rate A B = 1 / 10)
  (h2 : a_work_rate A = 1 / 20) :
  1 / b_work_rate B = 20 :=
by
  -- Using hypotheses to define work rates
  let combined_rate := combined_work_rate A B
  let a_rate := a_work_rate A
  let b_rate := b_work_rate B
  -- Showing combined rate, a's rate and b's rate calculations
  have hb : b_rate = combined_rate - a_rate, by sorry
  -- Simplifying hb and proving the original theorem statement
  sorry

end b_alone_completes_in_20_days_l516_516938


namespace solve_geographical_problem_l516_516276

def R : ℝ := 6370 -- Earth's radius in km
def φ₁ : ℝ := 47.5 -- Latitude of point B in degrees
def λ₁ : ℝ := 19.1 -- Longitude of point B in degrees
def φ₂ : ℝ := 30.5 -- Latitude of point A in degrees
def λ₂ : ℝ := -9.6 -- Longitude of point A in degrees

/-- 
Prove the following:
1. The distance from the equatorial plane and Earth's axis of rotation for points A and B.
2. The distance from the 0° and ±90° meridian planes for points A and B.
3. The straight-line distance (chord) between points A and B.
4. The distance on the Earth's surface along the shortest arc between points A and B.
5. The maximum depth of the chord AB below the Earth's surface.
6. The general formula for the cosine of the angle between any two points on the Earth's surface.
-/
theorem solve_geographical_problem :
  -- Distances from the equatorial plane for points A and B
  let z₁ := R * Real.sin (φ₁ * Real.pi / 180)
  let d₁ := R * Real.cos (φ₁ * Real.pi / 180)
  let z₂ := R * Real.sin (φ₂ * Real.pi / 180)
  let d₂ := R * Real.cos (φ₂ * Real.pi / 180),

  -- Distances from the 0° and ±90° meridian planes for points A and B
  let y₁ := d₁ * Real.sin (λ₁ * Real.pi / 180)
  let x₁ := d₁ * Real.cos (λ₁ * Real.pi / 180)
  let y₂ := d₂ * Real.sin (λ₂ * Real.pi / 180)
  let x₂ := d₂ * Real.cos (λ₂ * Real.pi / 180),

  -- Distance between A and B as a chord
  let a := x₂ - x₁
  let b := y₁ - y₂
  let c := z₁ - z₂
  let h := Real.sqrt (a^2 + b^2 + c^2),

  -- Shortest surface distance between A and B
  let cos_θ := 1 - (h^2 / (2 * R^2))
  let θ := Real.arccos cos_θ
  let surface_distance := R * θ,

  -- Maximum depth of the chord AB
  let cos_θ_half := Real.cos (θ / 2)
  let m := R * (1 - cos_θ_half),
  
  -- General formula for the cosine of the angle between two points on the Earth's surface
  let cos_distance := Real.sin (φ₁ * Real.pi / 180) * Real.sin (φ₂ * Real.pi / 180) + 
                       Real.cos (φ₁ * Real.pi / 180) * Real.cos (φ₂ * Real.pi / 180) * 
                       Real.cos ((λ₁ - λ₂) * Real.pi / 180),

  -- Assertions
  z₁ = 4696 ∧ d₁ = 4304 ∧
  z₂ = 3233 ∧ d₂ = 5488 ∧
  y₁ = 1408 ∧ x₁ = 4066 ∧
  y₂ = -915 ∧ x₂ = 5411 ∧
  h = 3057 ∧ 
  surface_distance = 3088 ∧ 
  m = 190 ∧ 
  cos_distance = Real.cos θ := 
sorry

end solve_geographical_problem_l516_516276


namespace radius_of_convergence_l516_516310

noncomputable def power_series (x : ℝ) : ℕ → ℝ := λ n, (3^n / n!) * x^n

theorem radius_of_convergence : ∀ x : ℝ, ∃ r = 1, power_series x ↦ 0 := 
by {
  sorry
}

end radius_of_convergence_l516_516310


namespace find_missing_number_l516_516721

theorem find_missing_number (n : ℝ) :
  (0.0088 * 4.5) / (0.05 * 0.1 * n) = 990 → n = 0.008 :=
by
  intro h
  sorry

end find_missing_number_l516_516721


namespace find_ratio_b_c_l516_516408

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end find_ratio_b_c_l516_516408


namespace chess_class_percentage_l516_516134

-- Given conditions as definitions
def total_students : ℕ := 2000
def students_in_chess_class (P : ℝ) := (P / 100) * total_students
def students_in_swimming_class (P : ℝ) := 0.5 * students_in_chess_class P
def students_attending_swimming : ℕ := 100

-- The main statement to prove
theorem chess_class_percentage (P : ℝ) :
    students_in_swimming_class P = students_attending_swimming → P = 10 :=
by
  -- Proof begins here
  sorry

end chess_class_percentage_l516_516134


namespace line_does_not_pass_through_third_quadrant_l516_516750

variable {a b c : ℝ}

theorem line_does_not_pass_through_third_quadrant
  (hac : a * c < 0) (hbc : b * c < 0) : ¬ ∃ x y, x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 :=
sorry

end line_does_not_pass_through_third_quadrant_l516_516750


namespace ellipse_proof_l516_516716

noncomputable theory

open_locale classical

def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 4 = 1

def point_A : Prop := (0, -2) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2}
def point_B : Prop := ((3/2 : ℝ), -1) ∈ {p : ℝ × ℝ | ellipse_equation p.1 p.2}
def point_K : Prop := (0, -2)

def passes_through_fixed_point (x : ℝ → ℝ) (fixed : ℝ × ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (t * p1.1 + (1 - t) * p2.1 = fixed.1) ∧ (t * p1.2 + (1 - t) * p2.2 = fixed.2)

theorem ellipse_proof (hA : point_A) (hB : point_B) :
  (∀ M N : ℝ × ℝ, passes_through_fixed_point (λ x, -2) point_K M N) :=
sorry

end ellipse_proof_l516_516716


namespace even_three_colored_vertices_l516_516228

theorem even_three_colored_vertices (V F E : Type) [Finite V] [Finite F] [Finite E] 
(incidence : V → Finset F) 
(face_colors : F → Fin3) -- where Fin3 = {red, yellow, blue}
(convex_polyhedron : ∀ v : V, (incidence v).card = 3) :
  Even (Finset.card {v : V | ∃ r y b : F, r ≠ y ∧ y ≠ b ∧ b ≠ r ∧
        face_colors r = 0 ∧ face_colors y = 1 ∧ face_colors b = 2 ∧
        {r, y, b} ⊆ incidence v}) :=
sorry

end even_three_colored_vertices_l516_516228


namespace domain_of_h_l516_516905

-- Defining the function h(x)
def h (x : ℝ) : ℝ := (5 * x + 3) / (x - 4)

-- Stating the theorem
theorem domain_of_h :
  ∀ x : ℝ, x ≠ 4 ↔ x ∈ {x : ℝ | x ≠ 4} := 
sorry

end domain_of_h_l516_516905


namespace problem1_problem2_problem3_l516_516362

noncomputable def f (x: Real) : Real := (x * Real.log x) / (x + 1)

theorem problem1 (h: (f 1 = 1 / 2) ∧ (∃ (m: Real), (m = -2) ∧ (∀ x y, 2*x + y - 2 = 0))) :
  (dist (0, 0) { p : Real × Real | 2 * p.1 + p.2 - 2 = 0 } = 2 * Real.sqrt 5 / 5) := 
sorry

theorem problem2 (h: ∀ x, 1 ≤ x → f x ≤ m * (x - 1)) :
  (m ≥ 1 / 2) :=
sorry

theorem problem3 (n: ℕ) (h: n > 0) :
  Real.log (42 * n + 1) < ∑ i in (Finset.range n).map Nat.succ, i / (4 * i^2 - 1) :=
sorry

end problem1_problem2_problem3_l516_516362


namespace sylvesters_problem_l516_516565

open Finset

theorem sylvesters_problem (S : Finset (ℝ × ℝ)) (h_finite : S.nonempty) :
  (∀ P1 P2 ∈ S, P1 ≠ P2 → ∃ P3 ∈ S, P3 ≠ P1 ∧ P3 ≠ P2 ∧ collinear ℝ {(P1), (P2), (P3)}) →
  ∃ l : ℝ × ℝ → Prop, ∀ P ∈ S, l P :=
begin
  sorry
end

end sylvesters_problem_l516_516565


namespace probability_of_same_number_on_four_dice_l516_516164

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516164


namespace circle_touches_AB_l516_516598

-- Definitions of points and their relationships
variables {ω : Type*} [circle ω] (A B C D O E F : Point ω)

-- Define the conditions as hypotheses
hypothesis (condition1 : on_chord C A B)
hypothesis (condition2 : midpoint D A C)
hypothesis (condition3 : center O ω)
hypothesis (condition4 : second_intersection (circumcircle B O D) ω E)
hypothesis (condition5 : second_intersection_line (O C) F)

-- The theorem to prove that circumcircle of triangle CEF touches AB
theorem circle_touches_AB : touches (circumcircle C E F) (line A B) :=
sorry

end circle_touches_AB_l516_516598


namespace rectangle_length_l516_516507

/--
The perimeter of a rectangle is 150 cm. The length is 15 cm greater than the width.
This theorem proves that the length of the rectangle is 45 cm under these conditions.
-/
theorem rectangle_length (P w l : ℝ) (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : l = 45 :=
by
  sorry

end rectangle_length_l516_516507


namespace remainder_of_9876543210_div_101_l516_516923

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l516_516923


namespace diff_of_two_numbers_l516_516128

theorem diff_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end diff_of_two_numbers_l516_516128


namespace cathy_wallet_left_money_l516_516625

noncomputable def amount_left_in_wallet (initial : ℝ) (dad_amount : ℝ) (book_cost : ℝ) (saving_percentage : ℝ) : ℝ :=
  let mom_amount := 2 * dad_amount
  let total_initial := initial + dad_amount + mom_amount
  let after_purchase := total_initial - book_cost
  let saved_amount := saving_percentage * after_purchase
  after_purchase - saved_amount

theorem cathy_wallet_left_money :
  amount_left_in_wallet 12 25 15 0.20 = 57.60 :=
by 
  sorry

end cathy_wallet_left_money_l516_516625


namespace proof_of_circle_equation_proof_of_point_Q_l516_516002

noncomputable def circle_equation : Prop :=
  ∃ (m n : ℝ), 
  (m < 0 ∧ n > 0) ∧ 
  ((x - m)^2 + (y - n)^2 = 8) ∧ 
  (|m - n| = 4) ∧ 
  (m^2 + n^2 = 8) ∧ 
  ∀ (x y : ℝ), 
  (x + 2)^2 + (y - 2)^2 = 8

noncomputable def point_Q : Prop :=
  ∃ (x y : ℝ), 
  (x + 2)^2 + (y - 2)^2 = 8 ∧ 
  ((x - 4)^2 + y^2 = 16) ∧ 
  (x ≠ 0 ∨ y ≠ 0) ∧ 
  (x = 4 / 5) ∧ 
  (y = 12 / 5)

theorem proof_of_circle_equation : circle_equation := sorry

theorem proof_of_point_Q : point_Q := sorry

end proof_of_circle_equation_proof_of_point_Q_l516_516002


namespace branches_number_l516_516015

-- Conditions (converted into Lean definitions)
def total_leaves : ℕ := 12690
def twigs_per_branch : ℕ := 90
def leaves_per_twig_percentage_4 : ℝ := 0.3
def leaves_per_twig_percentage_5 : ℝ := 0.7
def leaves_per_twig_4 : ℕ := 4
def leaves_per_twig_5 : ℕ := 5

-- The goal
theorem branches_number (B : ℕ) 
  (h1 : twigs_per_branch = 90) 
  (h2 : leaves_per_twig_percentage_4 = 0.3) 
  (h3 : leaves_per_twig_percentage_5 = 0.7) 
  (h4 : leaves_per_twig_4 = 4) 
  (h5 : leaves_per_twig_5 = 5) 
  (h6 : total_leaves = 12690) :
  B = 30 := 
sorry

end branches_number_l516_516015


namespace triangle_is_right_l516_516525

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := 2 * z1

theorem triangle_is_right (O A B : ℂ) (h1 : A = z1) (h2 : B = z2)
  (h3 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0)
  (h4 : O = 0) :
  ∃ (θ : ℝ), θ = real.pi / 2 := sorry

end triangle_is_right_l516_516525


namespace smallest_prime_with_digit_sum_22_l516_516207

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem smallest_prime_with_digit_sum_22 :
  (∃ n : ℕ, is_prime n ∧ digit_sum n = 22 ∧ ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22) → n ≤ m) ∧
  ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22 ∧ m < 499) → false := 
sorry

end smallest_prime_with_digit_sum_22_l516_516207


namespace integral_sqrt_x2_add_a_l516_516073

theorem integral_sqrt_x2_add_a (x a : ℝ) :
  ∫ (x : ℝ) in set.univ, (λ x, sqrt (x^2 + a)) =
  (1 / 2) * (x * sqrt (x^2 + a) + a * log (abs (x + sqrt (x^2 + a)))) + C :=
by
  sorry

end integral_sqrt_x2_add_a_l516_516073


namespace same_number_on_four_dice_l516_516190

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516190


namespace profit_sharing_l516_516535

-- Define constants and conditions
def Tom_investment : ℕ := 30000
def Tom_share : ℝ := 0.40

def Jose_investment : ℕ := 45000
def Jose_start_month : ℕ := 2
def Jose_share : ℝ := 0.30

def Sarah_investment : ℕ := 60000
def Sarah_start_month : ℕ := 5
def Sarah_share : ℝ := 0.20

def Ravi_investment : ℕ := 75000
def Ravi_start_month : ℕ := 8
def Ravi_share : ℝ := 0.10

def total_profit : ℕ := 120000

-- Define expected shares
def Tom_expected_share : ℕ := 48000
def Jose_expected_share : ℕ := 36000
def Sarah_expected_share : ℕ := 24000
def Ravi_expected_share : ℕ := 12000

-- Theorem statement
theorem profit_sharing :
  let Tom_contribution := Tom_investment * 12
  let Jose_contribution := Jose_investment * (12 - Jose_start_month)
  let Sarah_contribution := Sarah_investment * (12 - Sarah_start_month)
  let Ravi_contribution := Ravi_investment * (12 - Ravi_start_month)
  Tom_share * total_profit = Tom_expected_share ∧
  Jose_share * total_profit = Jose_expected_share ∧
  Sarah_share * total_profit = Sarah_expected_share ∧
  Ravi_share * total_profit = Ravi_expected_share := by {
    sorry
  }

end profit_sharing_l516_516535


namespace sum_first_2016_terms_l516_516733

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom seq_initial : a 1 = 1
axiom seq_recurrence : ∀ n : ℕ, a (n + 1) * a n = 2 ^ n

-- Definition of the sum of the sequence
def sum_seq (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

-- The main statement to prove
theorem sum_first_2016_terms :
  sum_seq a 2015 = 3 * 2 ^ 1008 - 3 :=
sorry

end sum_first_2016_terms_l516_516733


namespace quadratic_rewrite_constants_l516_516290

theorem quadratic_rewrite_constants (a b c : ℤ) 
    (h1 : -4 * (x - 2) ^ 2 + 144 = -4 * x ^ 2 + 16 * x + 128) 
    (h2 : a = -4)
    (h3 : b = -2)
    (h4 : c = 144) 
    : a + b + c = 138 := by
  sorry

end quadratic_rewrite_constants_l516_516290


namespace value_of_p_l516_516338

theorem value_of_p (a : ℕ → ℚ) (m : ℕ) (p : ℚ)
  (h1 : a 1 = 111)
  (h2 : a 2 = 217)
  (h3 : ∀ n : ℕ, 3 ≤ n ∧ n ≤ m → a n = a (n - 2) - (n - p) / a (n - 1))
  (h4 : m = 220) :
  p = 110 / 109 :=
by
  sorry

end value_of_p_l516_516338


namespace cirrus_clouds_count_l516_516880

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l516_516880


namespace probability_of_same_number_on_four_dice_l516_516165

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516165


namespace no_tiling_possible_with_given_dimensions_l516_516012

theorem no_tiling_possible_with_given_dimensions :
  ¬(∃ (n : ℕ), n * (2 * 2 * 1) = (3 * 4 * 5) ∧ 
   (∀ i j k : ℕ, i * 2 = 3 ∨ i * 2 = 4 ∨ i * 2 = 5) ∧
   (∀ i j k : ℕ, j * 2 = 3 ∨ j * 2 = 4 ∨ j * 2 = 5) ∧
   (∀ i j k : ℕ, k * 1 = 3 ∨ k * 1 = 4 ∨ k * 1 = 5)) :=
sorry

end no_tiling_possible_with_given_dimensions_l516_516012


namespace cyclist_speed_l516_516851

variable (x : ℝ)
variable (t_ab t_ba : ℝ)

def distance_ab := 60 -- distance from A to B in km

def time_from_a_to_b (x : ℝ) : ℝ := distance_ab / x -- time to travel from A to B

def time_first_part := 1 -- First hour of the return trip
def break_time := 20 / 60 -- Break time 20 minutes converted to hours

def remaining_distance (x : ℝ) : ℝ := distance_ab - x -- remaining distance after the first hour
def increased_speed (x : ℝ) : ℝ := x + 4 -- speed after the break
def time_remaining_distance (x : ℝ) : ℝ := remaining_distance x / increased_speed x -- time to cover the remaining distance

def total_time_back (x : ℝ) : ℝ := time_first_part + break_time + time_remaining_distance x -- Total time for return journey

theorem cyclist_speed : 
  time_from_a_to_b x = total_time_back x → 
  x = 20 := 
by 
  sorry

end cyclist_speed_l516_516851


namespace factorization_of_expression_l516_516301

theorem factorization_of_expression (a b c : ℝ) : 
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
by
  sorry

end factorization_of_expression_l516_516301


namespace IncorrectOptionC_l516_516552

def Material1 (A B C D : Prop) : Prop :=
  (A = "The 'Internet Post Comments Service Management Regulations' require post comment service providers to establish information security management systems, a 'review-before-post' system, checks for real identity authentication, etc., to create a healthy and civilized cyberspace") ∧
  (B = "From a global perspective, it is effective to supervise internet comments and other content according to the law, and it conforms to the trend of international internet governance development") ∧
  (C = "The United States, the United Kingdom, and Russia have all issued relatively complete legal and regulatory systems to strictly supervise internet post comments and other content, leading in the governance of the internet according to the law") ∧
  (D = "The 'Internet Post Comments Service Management Regulations' clarify the red lines of post comments, but the red lines are not equivalent to the moral bottom line, and its restraining force is limited for those without any moral bottom line")

theorem IncorrectOptionC (A B C D : Prop) (text : Prop) :
  Material1 A B C D →
  text =
    ("The United States and the United Kingdom have enacted some regulations, but only Russia has established a 'more complete legal and regulatory system.'") →
  ¬ C := 
by
  sorry

end IncorrectOptionC_l516_516552


namespace concyclic_points_l516_516537

-- Definitions of the concepts
variable {Point : Type}
-- Define the circles γ1 and γ2 with points M and N as intersections
variable (γ1 γ2 : Set Point) (M N : Point)
-- Points A, B, C, D, E, F
variable (A B C D E F : Point)
-- Lines AM, AN, DM, and DN
variable (line_AM line_AN line_DM line_DN : Set Point)
-- Order of points around γ1
variable (order_on_γ1 : List Point)

-- Noncomputable statement for conditions
noncomputable theory
open_locale classical

-- Assuming all conditions
axiom condition_1 : M ∈ γ1 ∧ M ∈ γ2
axiom condition_2 : N ∈ γ1 ∧ N ∈ γ2
axiom condition_3 : A ∈ γ1
axiom condition_4 : D ∈ γ2
axiom condition_5 : B ∈ γ2 ∧ B ≠ M ∧ B ≠ N ∧ B ∈ line_AM
axiom condition_6 : C ∈ γ2 ∧ C ≠ M ∧ C ≠ N ∧ C ∈ line_AN
axiom condition_7 : E ∈ γ1 ∧ E ≠ M ∧ E ≠ N ∧ E ∈ line_DM
axiom condition_8 : F ∈ γ1 ∧ F ≠ M ∧ F ≠ N ∧ F ∈ line_DN
axiom condition_9 : [M, N, F, A, E] = order_on_γ1
axiom condition_10 : dist A B = dist D E

-- Theorem statement
theorem concyclic_points :
  ∃ (circle : Set Point), (A ∈ circle) ∧ (F ∈ circle) ∧ (C ∈ circle) ∧ (D ∈ circle) ∧ (∀ (A₁ D₁ : Point), A₁ ∈ γ1 ∧ D₁ ∈ γ2 ∧ 
  dist A₁ (λ B₁ : Point, B₁ ∈ γ2 ∧ B₁ ≠ M ∧ B₁ ≠ N ∧ B₁ ∈ (λ x : Point, ∃ (line_AM₁ : Set Point), A₁ ∉ line_AM₁ ∧ x ∈ line_AM₁)) =
  dist D₁ (λ E₁ : Point, E₁ ∈ γ1 ∧ E₁ ≠ M ∧ E₁ ≠ N ∧ E₁ ∈ (λ x : Point, ∃ (line_DM₁ : Set Point), D₁ ∉ line_DM₁ ∧ x ∈ line_DM₁)) → 
  (A₁ ∈ circle) ∧ (D₁ ∈ circle)) :=
sorry

end concyclic_points_l516_516537


namespace cosine_rule_find_c_l516_516761

noncomputable def find_c (a b A : ℝ) : set ℝ :=
  {c | c = 3 * Real.sqrt 3 + 3 ∨ c = 3 * Real.sqrt 3 - 3}

theorem cosine_rule_find_c :
  let a := 3 * Real.sqrt 2,
      b := 6,
      A := Real.pi / 6,
      cos_A := Real.cos A
  in find_c a b A = 
     {c | c = 3 * Real.sqrt 3 + 3 ∨ c = 3 * Real.sqrt 3 - 3} :=
by
  sorry

end cosine_rule_find_c_l516_516761


namespace symmetric_point_Q_l516_516004

-- Definitions based on conditions
def P : ℝ × ℝ := (-3, 2)
def symmetric_with_respect_to_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.fst, -point.snd)

-- Theorem stating that the coordinates of point Q (symmetric to P with respect to the x-axis) are (-3, -2)
theorem symmetric_point_Q : symmetric_with_respect_to_x_axis P = (-3, -2) := 
sorry

end symmetric_point_Q_l516_516004


namespace midpoint_of_AB_minimize_PM_l516_516343

noncomputable def midpoint_coordinates_of_AB (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

theorem midpoint_of_AB (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (l = (λ x y, y = (Real.sqrt 3 / 3) * (x + 1))) →
  (C = (λ x y, (x - 2) ^ 2 + y ^ 2 = 3)) →
  midpoint_coordinates_of_AB l C = (5 / 4, 3 * Real.sqrt 3/4) :=
sorry

noncomputable def minimum_PM_coordinates (C : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

theorem minimize_PM (C : ℝ → ℝ → Prop) :
  (C = (λ x y, (x -2) ^2 + y ^ 2 =3 )) →
  minimum_PM_coordinates C = (-1 / 32, 1 / 8) :=
sorry

end midpoint_of_AB_minimize_PM_l516_516343


namespace find_cookies_per_box3_l516_516627

variable (cookies_per_box1 cookies_per_box2 total_boxes1 total_boxes2 total_boxes3 total_cookies : ℕ)

def compute_cookies_box3 (cookies_per_box3 : ℕ) : Prop :=
  total_boxes1 * cookies_per_box1 +
  total_boxes2 * cookies_per_box2 +
  total_boxes3 * cookies_per_box3 = total_cookies

theorem find_cookies_per_box3
  (h1 : cookies_per_box1 = 12)
  (h2 : cookies_per_box2 = 20)
  (h3 : total_boxes1 = 50)
  (h4 : total_boxes2 = 80)
  (h5 : total_boxes3 = 70)
  (h6 : total_cookies = 3320) :
  compute_cookies_box3 16 :=
by
  sorry

end find_cookies_per_box3_l516_516627


namespace geometric_progression_forth_term_eq_l516_516498

theorem geometric_progression_forth_term_eq :
  ∀ (a1 a2 a3 : ℝ), 
  a1 = real.sqrt 2 ∧ 
  a2 = real.root 4 2 ∧ 
  a3 = real.root 8 2 → 
  ∀ (a4 : ℝ), a4 = a3 * (a2 / a1) →
  a4 = 1 / real.root 8 2 :=
by
  intros a1 a2 a3 h a4 h_a4
  rw [h.1, h.2.1, h.2.2] at h_a4
  sorry

end geometric_progression_forth_term_eq_l516_516498


namespace probability_winning_l516_516873

-- Define the probability of losing
def P_lose : ℚ := 5 / 8

-- Define the total probability constraint
theorem probability_winning : P_lose = 5 / 8 → (1 - P_lose) = 3 / 8 := 
by
  intro h
  rw h
  norm_num
  sorry

end probability_winning_l516_516873


namespace certain_event_is_B_l516_516216

-- Definitions for conditions
def eventA (l1 l2 : Line) : Prop :=
  parallel l1 l2 ∧ ∀ (α β : Angle), interiorAnglesOnSameSide l1 l2 α β → α = β

def eventB (p : Point) (l : Line) : Prop :=
  ∃! (m : Line), m ≠ l ∧ parallel l m ∧ p ∉ l ∧ p ∈ m

def eventC (Δ1 Δ2 : Triangle) : Prop :=
  ∃ (s1 s2 : Side) (a1 a2 : Angle), equalSides Δ1 Δ2 s1 s2 ∧ equalAngle Δ1 Δ2 a1 a2 → congruent Δ1 Δ2

def eventD : Prop := 
  ∃ (x : ℕ), x ∈ {1, 2, 3, 4, 5, 6} ∧ x = 9

-- The proof problem
theorem certain_event_is_B (l : Line) (p : Point) :
  ∀ A B C D : Prop,
    (A = eventA l ∧ B = eventB p l ∧ C = eventC Δ1 Δ2 ∧ D = eventD) →
    (B = true) :=
by
  sorry

end certain_event_is_B_l516_516216


namespace area_triangle_PAB_range_l516_516726

noncomputable def f : ℝ → ℝ
| x => if 0 < x ∧ x < 1 then -Real.log x else Real.log x

theorem area_triangle_PAB_range (x1 x2 x : ℝ) (h1 : 0 < x1) (h2 : x1 < 1) (h3 : 1 < x2) (h4 : x2 = 1 / x1) (h5 : 1 < x) :
  ∃ S : ℝ, S = (x : ℕ) → 1 < S :=
sorry

end area_triangle_PAB_range_l516_516726


namespace multiplier_for_difference_l516_516271

variable (x y k : ℕ)
variable (h1 : x + y = 81)
variable (h2 : x^2 - y^2 = k * (x - y))
variable (h3 : x ≠ y)

theorem multiplier_for_difference : k = 81 := 
by
  sorry

end multiplier_for_difference_l516_516271


namespace choose_100_disjoint_chords_with_equal_sums_l516_516530

theorem choose_100_disjoint_chords_with_equal_sums :
  ∃ (chords : Finset (ℕ × ℕ)), chords.card = 100 ∧ 
    ∀ (c ∈ chords), ∀ (c' ∈ chords), c ≠ c' → (c.1 + c.2 = c'.1 + c'.2) :=
sorry

end choose_100_disjoint_chords_with_equal_sums_l516_516530


namespace same_number_probability_four_dice_l516_516169

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516169


namespace range_of_a_l516_516821

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (|x + a| + |2x - 1| ≤ |2x + 1|)) → (a ∈ set.Icc (-3 : ℝ) 0) :=
by
  sorry

end range_of_a_l516_516821


namespace quadrilateral_perimeter_l516_516152

noncomputable def perimeter_ABCD (AB DC BC : ℝ) : ℝ :=
let AD := Real.sqrt ((AB - DC) ^ 2 + BC ^ 2)
in AB + BC + DC + AD

theorem quadrilateral_perimeter (AB DC BC : ℝ)
  (h1 : AB = 12)
  (h2 : DC = 7)
  (h3 : BC = 15)
  (h4 : AB - DC > 0) :
  perimeter_ABCD AB DC BC = 34 + 5 * Real.sqrt 10 := by
  rw [perimeter_ABCD, h1, h2, h3]
  have AD := Real.sqrt ((h1 - h2) ^ 2 + h3 ^ 2)
  rw [h1, h2, h3, Real.sqrt_eq_rpow]
  sorry

end quadrilateral_perimeter_l516_516152


namespace numbers_containing_digit_1_more_frequent_l516_516992

theorem numbers_containing_digit_1_more_frequent :
  let total_numbers := 10^10 in
  let numbers_without_1 := 9^10 - 1 in
  let numbers_with_1 := total_numbers - numbers_without_1 in
  numbers_with_1 > numbers_without_1 :=
by
  sorry

end numbers_containing_digit_1_more_frequent_l516_516992


namespace eighteen_mnp_eq_P_np_Q_2mp_l516_516491

theorem eighteen_mnp_eq_P_np_Q_2mp (m n p : ℕ) (P Q : ℕ) (hP : P = 2 ^ m) (hQ : Q = 3 ^ n) :
  18 ^ (m * n * p) = P ^ (n * p) * Q ^ (2 * m * p) :=
by
  sorry

end eighteen_mnp_eq_P_np_Q_2mp_l516_516491


namespace prove_XCV_sum_correct_l516_516493

noncomputable def XCV_sum_correct : Prop :=
  ∃ (X C V : ℕ),
  XCV = 100 * X + 10 * C + V ∧
  XXV = 100 * X + 10 * X + V ∧
  CXX = 100 * C + 10 * X + X ∧
  XYN (X, C, V) ∧
  (100 * X + 10 * C + V) + (100 * X + 10 * X + V) = (100 * C + 10 * X + X)

theorem prove_XCV_sum_correct : XCV_sum_correct :=
sorry

end prove_XCV_sum_correct_l516_516493


namespace f_f_five_l516_516049

noncomputable def f : ℝ → ℝ
| x => if x ≤ 2 then 2^(x-2) else real.log2 (x-1)

theorem f_f_five : f (f 5) = 1 :=
by
  sorry

end f_f_five_l516_516049


namespace right_triangle_not_possible_l516_516269

theorem right_triangle_not_possible :
  ¬ (∃ (a₁ a₂ a₃ : ℝ), (a₁ = 2 ∧ a₂ = 5 ∧ a₃ = 6) ∧ (a₂^2 + a₁^2 = a₃^2)) :=
by
  apply not_exists.2
  intro a₁
  apply not_exists.2
  intro a₂
  apply not_exists.2
  intro a₃
  intro h
  cases h with ha h₁h₂
  cases h₁h₂ with hh₂
  rw [← hh₂.1, ← hh₂.2, pow_two, pow_two, pow_two] at ha
  contradiction
  -- Here we expect the result to derive a contradiction that shows 4 + 25 ≠ 36
  sorry

end right_triangle_not_possible_l516_516269


namespace John_pays_2400_per_year_l516_516422

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l516_516422


namespace triangle_side_length_l516_516384

theorem triangle_side_length (AB AC BC BX CX : ℕ)
  (h1 : AB = 86)
  (h2 : AC = 97)
  (h3 : BX + CX = BC)
  (h4 : AX = AB)
  (h5 : AX = 86)
  (h6 : AB * AB * CX + AC * AC * BX = BC * (BX * CX + AX * AX))
  : BC = 61 := 
sorry

end triangle_side_length_l516_516384


namespace f_monotonic_increasing_solve_inequality_l516_516363

-- Define the function f
def f (a : ℝ) (x : ℝ) := a / (Real.exp x + 1) + 1

-- Assume f is an odd function
axiom h_odd : ∀ x : ℝ, f -2 (-x) = -f -2 x

-- Theorem 1: f is monotonically increasing 
theorem f_monotonic_increasing : ∀ x : ℝ, f -2 x < f -2 (x + 1) :=
by sorry

-- Theorem 2: Solve the inequality
theorem solve_inequality (x : ℝ) :
  f -2 (Real.log 2 * Real.log 2 * x) + f -2 (Real.log (Real.sqrt 2) * x - 3) ≤ 0 →
  x ∈ Set.Icc (1 / 8) 2 :=
by sorry

end f_monotonic_increasing_solve_inequality_l516_516363


namespace steve_bought_2_cookies_boxes_l516_516089

theorem steve_bought_2_cookies_boxes :
  ∀ (milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
     milk_qty cereal_qty banana_qty apple_qty cookies_qty : ℕ),
    milk_price = 3 →
    cereal_price_per_box = 3.5 →
    banana_price = 0.25 →
    apple_price = 0.5 →
    cookies_cost_per_box = 2 * milk_price →
    milk_qty = 1 →
    cereal_qty = 2 →
    banana_qty = 4 →
    apple_qty = 4 →
    (milk_price * milk_qty + cereal_price_per_box * cereal_qty + 
     banana_price * banana_qty + apple_price * apple_qty + 
     cookies_cost_per_box * cookies_qty = 25) →
    cookies_qty = 2 :=
by
  intros milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
         milk_qty cereal_qty banana_qty apple_qty cookies_qty
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 hsum
  have h_milk : milk_price = 3 := h1
  have h_cereal : cereal_price_per_box = 3.5 := h2
  have h_banana : banana_price = 0.25 := h3
  have h_apple : apple_price = 0.5 := h4
  have h_cookies_price : cookies_cost_per_box = 2 * milk_price := h5
  have h_milk_qty : milk_qty = 1 := h6
  have h_cereal_qty : cereal_qty = 2 := h7
  have h_banana_qty : banana_qty = 4 := h8
  have h_apple_qty : apple_qty = 4 := h9
  rw [h_milk, h_cereal, h_banana, h_apple, h_cookies_price, h_milk_qty, h_cereal_qty, h_banana_qty, h_apple_qty] at hsum
  sorry

end steve_bought_2_cookies_boxes_l516_516089


namespace math_problem_l516_516701

theorem math_problem (t : ℝ) (k m n : ℕ) (hrel_prime: nat.coprime m n)
    (h1 : (1 + real.sin t) * (1 + real.cos t) = 9/4)
    (h2 : (1 - real.sin t) * (1 - real.cos t) = m / n - real.sqrt k) :
    k + m + n = 6 := by 
  sorry

end math_problem_l516_516701


namespace translate_and_double_cos_l516_516536

theorem translate_and_double_cos (x : ℝ) : 
  let f := λ x, Real.cos x in
  let f_translated := λ x, f (x + (π / 4)) in
  let f_result := λ x, f_translated (2 * x) in
  f_result x = Real.cos (2 * x + (π / 4)) :=
by
  sorry

end translate_and_double_cos_l516_516536


namespace tangent_circumcircles_at_midpoint_l516_516813

theorem tangent_circumcircles_at_midpoint
  (O M B H_A : Type) 
  (ABC : Triangle)
  (circumcenter : O)
  (midpoint : M)
  (line_BO : Line O)
  (altitude_B : Line B)
  (intersects : intersects line_BO altitude_B)
  (is_midpoint : is_midpoint M (side AC))
  (tangent_property : is_tangent (circumcircle BHA) (line AC) M)
  (tangent_property2 : is_tangent (circumcircle BHC) (line AC) M) :
  tangent_property ∧ tangent_property2 :=
by {
  sorry,
}

end tangent_circumcircles_at_midpoint_l516_516813


namespace painting_cost_in_euro_l516_516832

/-- A painting costs 140 Namibian dollars. Given the exchange rates:
    1 USD = 7 N$
    1 USD = 0.9 EUR
    Prove that the painting costs 18 Euros (EUR). --/
theorem painting_cost_in_euro (N_usd : ℝ) (E_usd : ℝ) (C_painting : ℝ) : 
  (N_usd = 7) → (E_usd = 0.9) → (C_painting = 140) → (C_painting / N_usd * E_usd = 18) :=
begin
  intros hN hE hC,
  rw [hN, hE, hC],
  norm_num
end

end painting_cost_in_euro_l516_516832


namespace expand_product_l516_516662

noncomputable def expand_poly (x : ℝ) : ℝ := (x + 3) * (x^2 + 2 * x + 4)

theorem expand_product (x : ℝ) : expand_poly x = x^3 + 5 * x^2 + 10 * x + 12 := 
by 
  -- This will be filled with the proof steps, but for now we use sorry.
  sorry

end expand_product_l516_516662


namespace determinant_of_A_l516_516660

variable (x : ℝ)

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, 
    match i, j with
    | 0, 0 => 2 * x + 3
    | 0, 1 => x
    | 0, 2 => x
    | 1, 0 => 2 * x
    | 1, 1 => 2 * x + 3
    | 1, 2 => x
    | 2, 0 => 2 * x
    | 2, 1 => x
    | 2, 2 => 2 * x + 3
    | _, _ => 0

theorem determinant_of_A : Matrix.det (A x) = 2 * x^3 + 27 * x^2 + 27 * x + 27 :=
  sorry

end determinant_of_A_l516_516660


namespace factorial_division_l516_516703

theorem factorial_division : 8.factorial = 40320 → 8.factorial / 4.factorial = 1680 := by
  intros h
  rw h
  norm_num

end factorial_division_l516_516703


namespace eventually_periodic_of_rational_cubic_l516_516814

noncomputable def is_rational_sequence (P : ℚ → ℚ) (q : ℕ → ℚ) :=
  ∀ n : ℕ, q (n + 1) = P (q n)

theorem eventually_periodic_of_rational_cubic (P : ℚ → ℚ) (q : ℕ → ℚ) (hP : ∃ a b c d : ℚ, ∀ x : ℚ, P x = a * x^3 + b * x^2 + c * x + d) (hq : is_rational_sequence P q) : 
  ∃ k ≥ 1, ∀ n ≥ 1, q (n + k) = q n := 
sorry

end eventually_periodic_of_rational_cubic_l516_516814


namespace total_value_of_assets_l516_516455

variable (value_expensive_stock : ℕ)
variable (shares_expensive_stock : ℕ)
variable (shares_other_stock : ℕ)
variable (value_other_stock : ℕ)

theorem total_value_of_assets
    (h1: value_expensive_stock = 78)
    (h2: shares_expensive_stock = 14)
    (h3: shares_other_stock = 26)
    (h4: value_other_stock = value_expensive_stock / 2) :
    shares_expensive_stock * value_expensive_stock + shares_other_stock * value_other_stock = 2106 := by
    sorry

end total_value_of_assets_l516_516455


namespace correct_size_balloons_count_l516_516099

-- Definitions of the given conditions
def initial_balloons : ℕ := 47
def additional_balloons : ℕ := 13
def correct_size_percentage : ℝ := 0.70

-- Statement to prove the correct size balloons count
theorem correct_size_balloons_count :
  0.70 * (initial_balloons + additional_balloons) = 42 := by
  sorry

end correct_size_balloons_count_l516_516099


namespace train_crosses_in_26_seconds_l516_516587

def speed_km_per_hr := 72
def length_of_train := 250
def length_of_platform := 270

def total_distance := length_of_train + length_of_platform

noncomputable def speed_m_per_s := (speed_km_per_hr * 1000 / 3600)  -- Convert km/hr to m/s

noncomputable def time_to_cross := total_distance / speed_m_per_s

theorem train_crosses_in_26_seconds :
  time_to_cross = 26 := 
sorry

end train_crosses_in_26_seconds_l516_516587


namespace angle_measure_Q_l516_516480

def decagon := sorry -- Define symbolically as a regular decagon

-- Sides BJ and EF are part of decagon lack specificity but we'll refer symbolically
-- In Lean, we'll define the intersection point of the extended sides as Q.

def sides_extended_intersect_at_Q (d : decagon) : Prop :=
  ∃ Q, isRegularDecagon d ∧ (sidesExtended d.overlineBJ d.overlineEF Q)

-- Given BJ and EF of decagon are extended to meet at Q, 
-- prove angle measure at Q is 72 degrees
theorem angle_measure_Q {d : decagon} (h1 : isRegularDecagon d) (h2 : sides_extended_intersect_at_Q d) :
  angleMeasureQ d = 72 := by sorry

end angle_measure_Q_l516_516480


namespace max_two_scoop_sundaes_l516_516613

-- Define the problem conditions
def num_ice_cream_types : ℕ := 8
def scoops_per_sundae : ℕ := 2

-- Proof statement in Lean 4
theorem max_two_scoop_sundaes :  
  ∀ n r : ℕ, (n = num_ice_cream_types) → (r = scoops_per_sundae) →
  ((nat.choose n r) = 28) := 
by
  intros n r h_n h_r
  rw [h_n, h_r]
  iterate 2 { rw nat.choose }
  sorry -- Proof goes here

end max_two_scoop_sundaes_l516_516613


namespace analytical_expression_when_x_in_5_7_l516_516718

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma symmetric_about_one (x : ℝ) : f (1 - x) = f (1 + x) := sorry
lemma values_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = x := sorry

theorem analytical_expression_when_x_in_5_7 (x : ℝ) (h : 5 < x ∧ x ≤ 7) :
  f x = 6 - x :=
sorry

end analytical_expression_when_x_in_5_7_l516_516718


namespace min_value_on_top_layer_l516_516945

-- Definitions reflecting conditions
def bottom_layer : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def block_value (layer : List ℕ) (i : ℕ) : ℕ :=
  layer.getD (i-1) 0 -- assuming 1-based indexing

def second_layer_values : List ℕ :=
  [block_value bottom_layer 1 + block_value bottom_layer 2 + block_value bottom_layer 3,
   block_value bottom_layer 2 + block_value bottom_layer 3 + block_value bottom_layer 4,
   block_value bottom_layer 4 + block_value bottom_layer 5 + block_value bottom_layer 6,
   block_value bottom_layer 5 + block_value bottom_layer 6 + block_value bottom_layer 7,
   block_value bottom_layer 7 + block_value bottom_layer 8 + block_value bottom_layer 9,
   block_value bottom_layer 8 + block_value bottom_layer 9 + block_value bottom_layer 10]

def third_layer_values : List ℕ :=
  [second_layer_values.getD 0 0 + second_layer_values.getD 1 0 + second_layer_values.getD 2 0,
   second_layer_values.getD 1 0 + second_layer_values.getD 2 0 + second_layer_values.getD 3 0,
   second_layer_values.getD 3 0 + second_layer_values.getD 4 0 + second_layer_values.getD 5 0]

def top_layer_value : ℕ :=
  third_layer_values.getD 0 0 + third_layer_values.getD 1 0 + third_layer_values.getD 2 0

theorem min_value_on_top_layer : top_layer_value = 114 :=
by
  have h0 := block_value bottom_layer 1 -- intentionally leaving this incomplete as we're skipping the actual proof
  sorry

end min_value_on_top_layer_l516_516945


namespace laura_running_speed_l516_516429

noncomputable def running_speed (x : ℝ) : ℝ := x^2 - 1

noncomputable def biking_speed (x : ℝ) : ℝ := 3 * x + 2

noncomputable def biking_time (x: ℝ) : ℝ := 30 / (biking_speed x)

noncomputable def running_time (x: ℝ) : ℝ := 5 / (running_speed x)

noncomputable def total_motion_time (x : ℝ) : ℝ := biking_time x + running_time x

-- Laura's total workout duration without transition time
noncomputable def required_motion_time : ℝ := 140 / 60

theorem laura_running_speed (x : ℝ) (hx : total_motion_time x = required_motion_time) :
  running_speed x = 83.33 :=
sorry

end laura_running_speed_l516_516429


namespace find_f_2015_l516_516709

variables {ℝ : Type*} [LinearOrderedField ℝ]

-- Define f as an odd function satisfying f(2-x) = f(x)
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def satisfies_periodicity (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f x

-- Define specific form of f in the interval (0, 2)
def specific_form (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 2 → f x = x * (2 - x)

-- Define our main theorem
theorem find_f_2015 (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_periodicity : satisfies_periodicity f)
  (h_specific : specific_form f) :
  f 2015 = -1 :=
sorry

end find_f_2015_l516_516709


namespace fraction_is_two_thirds_l516_516665

noncomputable def fraction_of_price_of_ballet_slippers (f : ℚ) : Prop :=
  let price_high_heels := 60
  let num_ballet_slippers := 5
  let total_cost := 260
  price_high_heels + num_ballet_slippers * f * price_high_heels = total_cost

theorem fraction_is_two_thirds : fraction_of_price_of_ballet_slippers (2 / 3) := by
  sorry

end fraction_is_two_thirds_l516_516665


namespace problem_solution_l516_516846

variable (x y : ℝ)

-- Conditions
axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : (4 * x - 3 * y) / (x + 4 * y) = 3

-- Goal
theorem problem_solution : (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 :=
by
  sorry

end problem_solution_l516_516846


namespace base_six_equality_l516_516098

theorem base_six_equality (b : ℕ) (h₁ : 5 * 6 + 3 = 33) (h₂ : 1 * b^2 + 1 * b^1 + 3 = 113_b): 
  1 * b^2 + 1 * b + 3 = 33 ->  b = 5 := 
by 
  sorry

end base_six_equality_l516_516098


namespace min_two_triangles_with_small_area_l516_516773

noncomputable def rectangle (A B C D : Point) : Prop := 
  -- Definition of rectangle with area 1
  area (A, B, C, D) = 1

def non_collinear (points : list Point) : Prop := 
  ∀ (P Q R : Point), P ≠ Q → Q ≠ R → P ≠ R → 
  (P, Q, R) ∈ combinatorial_combinations points 3 → ¬collinear P Q R

def triangle_area_leq (P Q R : Point) (x : ℝ) : Prop :=
  area (P, Q, R) ≤ x

theorem min_two_triangles_with_small_area 
  (A B C D : Point) 
  (h_rect : rectangle A B C D)
  (P1 P2 P3 P4 P5 : Point) 
  (h_bounds : ∀ Pi, Pi ∈ {P1, P2, P3, P4, P5} → inside_or_on_boundary A B C D Pi)
  (h_non_collinear : non_collinear [P1, P2, P3, P4, P5]) :
  ∃ (triangles : list (Point × Point × Point)), 
  (∀ (t : Point × Point × Point), t ∈ triangles → 
  triangle_area_leq t.1 t.2 t.3 (1 / 4)) ∧ 
  length triangles ≥ 2 :=
sorry

end min_two_triangles_with_small_area_l516_516773


namespace graph_properties_l516_516754

theorem graph_properties (a b : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) 
    (passes_quad_one : ∃ x1 : ℝ, f x1 > 0) 
    (passes_quad_two : ∃ x2 : ℝ, f x2 < 0 ∧ x2 < 0) 
    (passes_quad_four : ∃ x3 : ℝ, f x3 > 0 ∧ x3 < 0) 
    (not_pass_quad_three : ∀ x4 : ℝ, f x4 ≥ 0 ∨ x4 ≤ 0) : 
    (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) :=
begin
  sorry,
end

where f (x : ℝ) := a^x + b - 1

end graph_properties_l516_516754


namespace evaluateExpression_at_3_l516_516483

noncomputable def evaluateExpression (x : ℚ) : ℚ :=
  (x - 1 + (2 - 2 * x) / (x + 1)) / ((x * x - x) / (x + 1))

theorem evaluateExpression_at_3 : evaluateExpression 3 = 2 / 3 := by
  sorry

end evaluateExpression_at_3_l516_516483


namespace remainder_div_101_l516_516916

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l516_516916


namespace hyperbola_eccentricity_prob_l516_516603

def DiceRolls := {p : ℕ × ℕ // (1 ≤ p.1 ∧ p.1 ≤ 6) ∧ (1 ≤ p.2 ∧ p.2 ≤ 6)}

def hyperbola_eccentricity (a b : ℕ) : ℝ :=
  real.sqrt (a^2 + b^2) / a

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 3), (1, 4), (1, 5), (2, 5), (1, 6), (2, 6)}

theorem hyperbola_eccentricity_prob :
  ∑ x in (Finset.univ : Finset DiceRolls), 
    if hyperbola_eccentricity x.1.1 x.1.2 > real.sqrt 5 then 1 else 0 = 6 →
  (∑ x in (Finset.univ : Finset DiceRolls), 1) = 36 →
  (6/36 : ℝ) = (1/6 : ℝ) :=
sorry

end hyperbola_eccentricity_prob_l516_516603


namespace range_of_a_l516_516366

noncomputable def f (x : ℝ) : ℝ := x * abs (x^2 - 12)

theorem range_of_a (m : ℝ) (hm : 0 ≤ m) : 
  ∃ a : ℝ, a ∈ set.Ici (1 : ℝ) ∧ range (λ x, f x) ∩ set.Icc (0 : ℝ) (a * m^2) = range (λ x, f x) ∩ set.Icc (0 : ℝ) (a * m^2) :=
by {
  -- Begin proof
  sorry -- Proof steps
}

end range_of_a_l516_516366


namespace discriminant_game_l516_516642

theorem discriminant_game {n : ℕ} (a : Fin (n + 1) → ℕ) :
  ∃ N, (∀ (β1 β2 : ℕ), β1 ≠ β2 →
    ∑ k in Finset.range (n + 1), a k * β1^k ≠ ∑ k in Finset.range (n + 1), a k * β2^k) → N = 2 :=
by sorry

end discriminant_game_l516_516642


namespace dice_probability_same_face_l516_516160

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516160


namespace lambda_parallel_l516_516684

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (3, 3)

theorem lambda_parallel (λ : ℝ) :
  (λ • a + b) ∈ Submodule.span ℝ ({b - a} : Set (ℝ × ℝ)) ↔ λ = -1 :=
by
  sorry

end lambda_parallel_l516_516684


namespace nandan_gain_l516_516796

theorem nandan_gain (x t : ℝ) (nandan_gain krishan_gain total_gain : ℝ)
  (h1 : krishan_gain = 12 * x * t)
  (h2 : nandan_gain = x * t)
  (h3 : total_gain = nandan_gain + krishan_gain)
  (h4 : total_gain = 78000) :
  nandan_gain = 6000 :=
by
  -- Proof goes here
  sorry

end nandan_gain_l516_516796


namespace policeman_can_catch_gangster_l516_516006

-- Definitions based on conditions
structure Square :=
  (side : ℝ)

structure Position :=
  (x y : ℝ)

def center (sq : Square) : Position :=
  { x := sq.side / 2, y := sq.side / 2 }

def is_on_side (sq : Square) (p : Position) : Prop :=
  (p.x = 0 ∨ p.x = sq.side ∨ p.y = 0 ∨ p.y = sq.side)

-- Speeds: policeman's speed is half the gangster's speed
variables (v : ℝ) -- gangster's speed
def policeman_speed := v / 2

-- Hypotheses: Initial positions
variables (sq : Square)
variables (policeman : Position := center sq)
variables (gangster : Position) -- Gangster starts at one of the vertices

-- Ensuring gangster starts at a vertex of the square
axiom gangster_on_vertex : gangster = { x := 0, y := 0 }
                           ∨ gangster = { x := 0, y := sq.side }
                           ∨ gangster = { x := sq.side, y := 0 }
                           ∨ gangster = { x := sq.side, y := sq.side }

-- Theorem to be proven
theorem policeman_can_catch_gangster :
  ∃ p : Position, is_on_side sq p ∧ (∃ t : ℝ, p = { x := gangster.x + v * t / 2, y := gangster.y + v * t / 2}) := 
sorry

end policeman_can_catch_gangster_l516_516006


namespace work_efficiency_ratio_l516_516590

variable (A B : ℝ)
variable (h1 : A = 1 / 2 * B) 
variable (h2 : 1 / (A + B) = 13)
variable (h3 : B = 1 / 19.5)

theorem work_efficiency_ratio : A / B = 1 / 2 := by
  sorry

end work_efficiency_ratio_l516_516590


namespace similar_1995_digit_numbers_exist_l516_516543

theorem similar_1995_digit_numbers_exist :
  ∃ (a b c : ℕ), 
    (∀ (x : ℕ), (x = a ∨ x = b ∨ x = c) → 
      x.to_digits.to_finset ⊆ {4, 5, 9}) ∧
    a.digits.length = 1995 ∧ 
    b.digits.length = 1995 ∧
    c.digits.length = 1995 ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a + b = c :=
sorry

end similar_1995_digit_numbers_exist_l516_516543


namespace rotate_angle_l516_516115

theorem rotate_angle
  (mACB : ℝ)
  (h1 : mACB = 60)
  (rotation : ℝ)
  (h2 : rotation = 420) :
  let new_angle := (mACB + rotation) % 360
  in new_angle % 360 = 0 :=
by
  sorry

end rotate_angle_l516_516115


namespace markers_leftover_l516_516456

theorem markers_leftover :
  let total_markers := 154
  let num_packages := 13
  total_markers % num_packages = 11 :=
by
  sorry

end markers_leftover_l516_516456


namespace number_of_cirrus_clouds_l516_516879

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l516_516879


namespace minimize_expression_l516_516046

noncomputable def E (x1 x2 x3 x4 : ℝ) : ℝ :=
  (2 * sin x1 ^ 2 + 1 / sin x1 ^ 2) * 
  (2 * sin x2 ^ 2 + 1 / sin x2 ^ 2) * 
  (2 * sin x3 ^ 2 + 1 / sin x3 ^ 2) * 
  (2 * sin x4 ^ 2 + 1 / sin x4 ^ 2)

theorem minimize_expression :
  ∀ (x1 x2 x3 x4 : ℝ), 
  0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ x1 + x2 + x3 + x4 = real.pi →
  E x1 x2 x3 x4 ≥ 81 :=
by
  sorry

end minimize_expression_l516_516046


namespace expected_value_coin_flip_l516_516237

/-
Problem Statement:
A biased coin flips heads with a probability of 2/3 and tails with a probability of 1/3. 
Flipping a heads gains $5, flipping two consecutive tails in a row results in losing $15. 
Harris performs two independent coin flips. Prove that the expected value of Harris's gain is $10/3.
-/
theorem expected_value_coin_flip : 
  let P_H := 2/3,
      P_T := 1/3,
      gain_H := 5,
      gain_TT := -15 in
  let E := (P_H * P_H * (2 * gain_H) +
            P_H * P_T * gain_H +
            P_T * P_H * gain_H +
            P_T * P_T * gain_TT) in
  E = 10 / 3 :=
by
  sorry

end expected_value_coin_flip_l516_516237


namespace dice_same_number_probability_l516_516196

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516196


namespace find_m_values_l516_516736

def A : set ℝ := {x | x = -2 ∨ x = -3}
def B (m : ℝ) : set ℝ := {x | m * x + 1 = 0}

theorem find_m_values (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, 1/2, 1/3} : set ℝ) := by
  sorry

end find_m_values_l516_516736


namespace composite_divisible_by_factorial_l516_516668

theorem composite_divisible_by_factorial (n : ℕ) (hn : n > 1) : (n ∣ (n-1)!) ↔ (¬ Nat.Prime n ∨ n = 4) :=
by
  sorry

end composite_divisible_by_factorial_l516_516668


namespace harmonious_equations_have_real_roots_l516_516636

-- Definitions based on conditions
def is_harmonious (a b c : ℝ) : Prop := b = a + c

-- The main theorem to prove
theorem harmonious_equations_have_real_roots (a b c : ℝ) (h : a ≠ 0) (h_harmonious : is_harmonious a b c) : 
    let Δ := b^2 - 4 * a * c in Δ ≥ 0 :=
by
  sorry

end harmonious_equations_have_real_roots_l516_516636


namespace sid_spent_on_computer_accessories_l516_516076

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l516_516076


namespace remainder_of_9876543210_div_101_l516_516926

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l516_516926


namespace clear_time_approx_l516_516564

/-- Variables defining the problem -/
def length_train1 : ℕ := 110   -- in meters
def length_train2 : ℕ := 200   -- in meters
def speed_train1 : ℕ := 80     -- in km/h
def speed_train2 : ℕ := 65     -- in km/h

/-- Total distance the trains need to cover to be clear of each other -/
def total_distance : ℕ := length_train1 + length_train2

/-- Relative speed of the trains moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1 + speed_train2

/-- Convert relative speed to meters per second -/
def relative_speed_mps : ℝ := (relative_speed_kmh * 1000) / 3600

/-- Calculate the time for the trains to be completely clear of each other -/
def time_clear : ℝ := total_distance / relative_speed_mps

/-- Proving the time taken is approximately 7.69 seconds -/
theorem clear_time_approx : |time_clear - 7.69| < 0.01 := sorry

end clear_time_approx_l516_516564


namespace factor_expression_l516_516299

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l516_516299


namespace shaded_area_correct_l516_516778

def right_trapezoid (A B C D : Type) [euclidean_geometry] : Prop :=
  angle A D B = 90 ∧ angle B A C = 90

def rectangle_area (A D E F : Type) [euclidean_geometry] : ℝ :=
  6.36

def intersects_at_point (B E A D P : Type) [euclidean_geometry] : Prop :=
  line BE intersects line AD at P

noncomputable def area_shaded_region (A B C D E F P : Type) [euclidean_geometry] : ℝ :=
  if right_trapezoid A B C D ∧ rectangle_area A D E F = 6.36 ∧ intersects_at_point B E A D P then 3.18 else 0

theorem shaded_area_correct {A B C D E F P : Type} [euclidean_geometry] :
  right_trapezoid A B C D →
  rectangle_area A D E F = 6.36 →
  intersects_at_point B E A D P →
  area_shaded_region A B C D E F P = 3.18 :=
by
  intro h_trap h_rect h_inter
  unfold area_shaded_region
  rw [h_trap, h_rect, h_inter]
  simp 
  exact rfl
  sorry

end shaded_area_correct_l516_516778


namespace C1_equation_fixed_point_line_AC_fixed_point_l516_516717

-- Definitions of the conditions
def C1 (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : set (ℝ × ℝ) :=
  { p | p.1^2 / a^2 - p.2^2 / b^2 = 1 }

def C2 : set (ℝ × ℝ) :=
  { p | p.1^2 / 5 + p.2^2 / 3 = 1 }

def foci_same (F : ℝ × ℝ) (C : set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ C, ∃ q ∈ C, (p - q).norm = ((1 : ℝ), 0).norm

def eccentricity_scale (eC1 eC2 : ℝ) : Prop :=
  eC1 = (sqrt 5) * eC2

theorem C1_equation_fixed_point (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (C1_foci_C2_foci : foci_same (1, 0) (C1 a b h_a_pos h_b_pos))
  (eccentricity_relation : eccentricity_scale (sqrt (a^2 + b^2) / a) (sqrt 2)) :
  (C1 a b h_a_pos h_b_pos) = (λ p, p.1^2 - p.2^2 = 4) :=
sorry

theorem line_AC_fixed_point
  (A F B C : ℝ × ℝ)
  (right_branch_C1 : A ∈ C1 2 2 by norm_num by norm_num)
  (right_focus_F : F = (2, 0))
  (AF_intersects_C1 : B ∈ C1 2 2 by norm_num by norm_num)
  (BC_perpendicular : C = (sqrt 2 / 2, C.2)) :
  ∃ x, (0, x) = (3 * sqrt 2 / 4, 0) :=
sorry

end C1_equation_fixed_point_line_AC_fixed_point_l516_516717


namespace price_of_ice_cream_l516_516581

theorem price_of_ice_cream (x : ℝ) :
  (225 * x + 125 * 0.52 = 200) → (x = 0.60) :=
sorry

end price_of_ice_cream_l516_516581


namespace factorial_fraction_simplification_l516_516929

theorem factorial_fraction_simplification : 
  (11! * 7! * 3! : ℚ) / (10! * 8!) = 11 / 56 := 
by 
  sorry

end factorial_fraction_simplification_l516_516929


namespace healthcare_contribution_is_57_5_cents_l516_516413

def jack_earnings_per_hour_dollars := 25
def healthcare_contribution_rate := 0.023

def earnings_per_hour_cents := jack_earnings_per_hour_dollars * 100
def healthcare_contribution_per_hour_cents := earnings_per_hour_cents * healthcare_contribution_rate

theorem healthcare_contribution_is_57_5_cents :
  healthcare_contribution_per_hour_cents = 57.5 :=
by
  sorry

end healthcare_contribution_is_57_5_cents_l516_516413


namespace prob_three_cards_in_sequence_l516_516257

theorem prob_three_cards_in_sequence : 
  let total_cards := 52
  let spades_count := 13
  let hearts_count := 13
  let sequence_prob := (spades_count / total_cards) * (hearts_count / (total_cards - 1)) * ((spades_count - 1) / (total_cards - 2))
  sequence_prob = (78 / 5100) :=
by
  sorry

end prob_three_cards_in_sequence_l516_516257


namespace find_side_b_l516_516788

theorem find_side_b (A B : ℝ) (a : ℝ) (hA : A = 30) (hB : B = 45) (ha : a = Real.sqrt 2): 
  ∃ b : ℝ, b = 2 :=
by
  use 2
  sorry

end find_side_b_l516_516788


namespace side_b_of_triangle_l516_516786

theorem side_b_of_triangle (A B : ℝ) (a b : ℝ) (hA : A = 30) (hB : B = 45) (ha : a = real.sqrt 2) :
  b = 2 :=
by
-- Including conditions for sine values for angles
have hsinA : real.sin (real.to_radians A) = 1 / 2, by sorry,
have hsinB : real.sin (real.to_radians B) = real.sqrt 2 / 2, by sorry,
-- Applying Law of Sines
have h : a / (real.sin (real.to_radians A)) = b / (real.sin (real.to_radians B)), by sorry,
-- Substituting given values and solving for b
sorry

end side_b_of_triangle_l516_516786


namespace polygon_diagonals_l516_516977

def exterior_angle_magnitude : ℝ := 10

theorem polygon_diagonals (n : ℕ) 
  (h1 : n * exterior_angle_magnitude = 360) : 
  let diag := n * (n - 3) / 2 in 
  diag = 594 :=
by
  sorry

end polygon_diagonals_l516_516977


namespace construct_triangle_l516_516146

open Real
open Set

noncomputable def exists_construct_triangle (B C : Point) (angle_B : ℝ) (a d : ℝ) : Prop :=
  ∃ (A : Point), Triangle ABC ∧
  measure_angle B A C = angle_B ∧
  distance B C = a ∧ 
  distance B A - distance A C = d

theorem construct_triangle (B C : Point) (angle_B : ℝ) (a d : ℝ)
  (angle_condition : 0 < angle_B ∧ angle_B < π)
  (side_condition : 0 < a)
  (difference_condition : 0 < d) :
  exists_construct_triangle B C angle_B a d :=
sorry

end construct_triangle_l516_516146


namespace disease_probability_l516_516092

theorem disease_probability:
  (let D := event that a person has the disease,
       Dc := event that a person does not have the disease,
       T := event that a person tests positive,
       Pr_D := 1 / 200,  -- Probability of having the disease
       Pr_Dc := 1 - Pr_D,  -- Probability of not having the disease
       Pr_T_given_D := 1,  -- Probability of a positive test given the disease
       Pr_T_given_Dc := 0.05,  -- Probability of a positive test given no disease
       Pr_T := Pr_T_given_D * Pr_D + Pr_T_given_Dc * Pr_Dc)
  (Pr_D * Pr_T_given_D / Pr_T) = 20 / 219 :=
by sorry

end disease_probability_l516_516092


namespace probability_of_same_number_on_four_dice_l516_516161

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516161


namespace evaluate_cube_root_power_l516_516654

theorem evaluate_cube_root_power (a : ℝ) (b : ℝ) (c : ℝ) (h : a = b^(3 : ℝ)) : (cbrt a)^12 = b^12 :=
by
  sorry

example : evaluate_cube_root_power 8 2 4096 (by rfl)

end evaluate_cube_root_power_l516_516654


namespace simplify_expression_l516_516904

theorem simplify_expression (w : ℝ) :
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 :=
by
  sorry

end simplify_expression_l516_516904


namespace expected_final_set_size_l516_516454

theorem expected_final_set_size : 
  let collection := {s : Set ℕ // s ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧ s ≠ ∅}
  -- Initial number of distinct nonempty subsets of {1,2,3,4,5,6,7,8}
  (initial_size : ℕ := 255) 
  -- Number of operations
  (operations : ℕ := 254) 
  -- Final expected size of the set
  (expected_size : ℚ := 1024 / 255) 
  -- Expected size proved
  expected_size = 1024 / 255 := sorry

end expected_final_set_size_l516_516454


namespace debbie_total_tape_l516_516287

def large_box_tape : ℕ := 4
def medium_box_tape : ℕ := 2
def small_box_tape : ℕ := 1
def label_tape : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5

def total_tape_used : ℕ := 
  (large_boxes_packed * (large_box_tape + label_tape)) +
  (medium_boxes_packed * (medium_box_tape + label_tape)) +
  (small_boxes_packed * (small_box_tape + label_tape))

theorem debbie_total_tape : total_tape_used = 44 := by
  sorry

end debbie_total_tape_l516_516287


namespace simplify_expr1_simplify_expr2_l516_516628

noncomputable section

theorem simplify_expr1 :
  ( (-2 * Real.sqrt 3 * Complex.i + 1) / (1 + 2 * Real.sqrt 3 * Complex.i) + 
    (Real.sqrt 2) / (1 + Complex.i)) ^ 2000 + 
    (1 + Complex.i) / (3 - Complex.i)
  = (6 / 65 : ℂ) + (39 / 65 : ℂ) * Complex.i := 
sorry

theorem simplify_expr2 :
  (5 * (4 + Complex.i) ^ 2) / (Complex.i * (2 + Complex.i)) + 
    2 / (1 - Complex.i) ^ 2
  = (-1 : ℂ) + 39 * Complex.i := 
sorry

end simplify_expr1_simplify_expr2_l516_516628


namespace merchant_profit_percentage_l516_516248

-- Definitions related to the question
def markup_percentage := 0.20
def discount_percentage := 0.10
def cost_price := 100.0

-- Proof statement to be proved
theorem merchant_profit_percentage :
  let MP := cost_price + (markup_percentage * cost_price) in
  let SP := MP - (discount_percentage * MP) in
  let Profit := SP - cost_price in
  let Profit_Percentage := (Profit / cost_price) * 100 in
  Profit_Percentage = 8 :=
by 
  -- Proof will go here
  sorry

end merchant_profit_percentage_l516_516248


namespace triangle_properties_l516_516383

theorem triangle_properties 
  (a b c : ℝ) (cosC : ℝ) :
  a = 4 →
  b = 5 →
  cosC = 1 / 8 →
  c = Real.sqrt (a^2 + b^2 - 2 * a * b * cosC) ∧
  S_abc = (1/2) * a * b * Real.sqrt(1 - cosC^2) →
  c = 6 ∧ S_abc =  (15 * Real.sqrt 7) / 4 := 
by
  sorry

end triangle_properties_l516_516383


namespace john_pays_2400_per_year_l516_516416

theorem john_pays_2400_per_year
  (hours_per_month : ℕ)
  (minutes_per_hour : ℕ)
  (songs_per_minute : ℕ)
  (cost_per_song : ℕ)
  (months_per_year : ℕ)
  (H1 : hours_per_month = 20)
  (H2 : minutes_per_hour = 60)
  (H3 : songs_per_minute = 3)
  (H4 : cost_per_song = 50)
  (H5 : months_per_year = 12) :
  let minutes_per_month := hours_per_month * minutes_per_hour,
      songs_per_month := minutes_per_month / songs_per_minute,
      cost_per_month := songs_per_month * cost_per_song in
  cost_per_month * months_per_year = 2400 := by
  sorry

end john_pays_2400_per_year_l516_516416


namespace area_of_square_field_l516_516849

theorem area_of_square_field (s : ℕ) (area : ℕ) (cost_per_meter : ℕ) (total_cost : ℕ) (gate_width : ℕ) :
  (cost_per_meter = 3) →
  (total_cost = 1998) →
  (gate_width = 1) →
  (total_cost = cost_per_meter * (4 * s - 2 * gate_width)) →
  (area = s^2) →
  area = 27889 :=
by
  intros h_cost_per_meter h_total_cost h_gate_width h_cost_eq h_area_eq
  sorry

end area_of_square_field_l516_516849


namespace problem_solution_l516_516572

structure Point where
  x : ℝ
  y : ℝ

def SquareCenter (PS : ℝ) : Point :=
  ⟨PS / 2, PS / 2⟩

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

noncomputable def angle (A B C : Point) : ℝ :=
  real.acos ((B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) / (distance A B * distance A C))

def lengthPS : ℝ := 800
def P := ⟨0, lengthPS⟩
def Q := ⟨lengthPS, lengthPS⟩
def O := SquareCenter lengthPS

variable (G H : Point)
variable (PG QH : ℝ)

axiom G_between_P_H : P.x < G.x ∧ G.x < H.x ∧ H.x < Q.x
axiom angle_GOH_eq_60 : angle G O H = real.pi / 3
axiom GH_eq_350 : distance G H = 350
axiom PG_lt_QH : PG < QH
axiom QH_expr : QH = 225 + 75 * real.sqrt 3

theorem problem_solution : (225 + 75 + 3 = 303) :=
by sorry

end problem_solution_l516_516572


namespace remainder_of_large_number_l516_516910

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l516_516910


namespace ellipse_and_circle_existence_l516_516340

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (eccentricity a b = (Real.sqrt 2) / 2) ∧ 
  (focal_distance a b = 4) ∧ 
  (equation a b = (x, y : ℝ) → (x^2 / (8:ℝ) + y^2 / (4:ℝ)) = 1)

noncomputable def circle_equation : Prop :=
  (center_x, center_y : ℝ) → (center_x = 0 ∧ center_y = 0) ∧ 
  (equation_center circle_radius := (x^2 + y^2 = (8:ℝ)/3)) ∧ 
  ∃ (circle_radius > 0), 
  (intersect_tangent (a, b, E) := ∃ A B, tangent_line (center_x, center_y) A B) ∧
  (orthogonal_vectors (a, b, E) := ∀A B, 
  (vec_cross_product AO_vec OB_vec = 0))

theorem ellipse_and_circle_existence :
  ∀ a b : ℝ, ellipse_equation a b →
  (∃ (circle_radius : ℝ), 
  circle_equation ∧ 
  ∀ (A B : ℝ), (tangent AB → orthogonal AB) → 
  |AB| ≤ 2 * Real.sqrt 3) := by
  sorry

end ellipse_and_circle_existence_l516_516340


namespace no_1234_in_sequence_repeats_1975_in_sequence_l516_516403

def digit_sequence (s : List ℕ) : Prop :=
  ∀ n, 0 ≤ n → n + 4 < s.length →
    s.get (n + 4) = (s.get n + s.get (n + 1) + s.get (n + 2) + s.get (n + 3)) % 10

theorem no_1234_in_sequence :
  ∀ s, digit_sequence (1 :: 9 :: 7 :: 5 :: s) →
    ¬ (List.isPrefixOf [1, 2, 3, 4] (1 :: 9 :: 7 :: 5 :: s)) :=
by
  sorry

theorem repeats_1975_in_sequence :
  ∀ s, digit_sequence (1 :: 9 :: 7 :: 5 :: s) →
    ∃ t, List.isPrefixOf [1, 9, 7, 5] t ∧
      ∃ u v, (1 :: 9 :: 7 :: 5 :: s) = u ++ t ++ v :=
by
  sorry

end no_1234_in_sequence_repeats_1975_in_sequence_l516_516403


namespace integral_solution_l516_516621

noncomputable def integral_problem : Prop :=
  ∃ C : ℝ, ∀ x : ℝ, 
    -1 ≤ x ∧ x ≤ 1 →
    ∃ F : ℝ → ℝ, 
      (∀ x : ℝ, (F' x = some (λ x, (arccos x)^4 / 4 + arccos x)) ∧
       F x = - (arccos x)^4 / 4 - arccos x + C)

theorem integral_solution : integral_problem :=
sorry

end integral_solution_l516_516621


namespace find_m_and_a_l516_516104

def quadratic_eq_has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0

def eq_solution_set (a : ℝ) : Set ℝ :=
  {x | (x - a) * (x + a - 2) < 0}

theorem find_m_and_a :
  let M := {m : ℝ | quadratic_eq_has_solutions m}
  let N a := eq_solution_set a
  (M = Icc (-1 / 4 : ℝ) 2) ∧ (∀ a, M ⊆ N a → a ∈ Iio (-1 / 4) ∪ Ioi (9 / 4)) :=
by sorry

end find_m_and_a_l516_516104


namespace compute_S6_l516_516817

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ :=
  x^m + (1 / x^m)

theorem compute_S6 (x : ℝ) (h : x + 1/x = 5) : S_m x 6 = 511 := by
  have S2 := (x + 1/x) ^ 2 - 2
  have S3 := (x + 1/x) * ((x + 1/x) ^ 2 - 2) - (x + 1/x)
  have S4 := ((x + 1/x) ^ 2 - 2) ^ 2 - 2
  have S6 := S2 * S4 - S3
  exact S6
  sorry

end compute_S6_l516_516817


namespace remainder_of_large_number_l516_516912

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l516_516912


namespace monotonicity_and_extrema_of_f_l516_516725

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem monotonicity_and_extrema_of_f :
  (∀ (x_1 x_2 : ℝ), x_1 ∈ Set.Icc (-1 : ℝ) 2 → x_2 ∈ Set.Icc (-1 : ℝ) 2 → x_1 < x_2 → f x_1 < f x_2) ∧ 
  (f (-1) = -1) ∧ 
  (f 2 = 8) :=
by
  sorry

end monotonicity_and_extrema_of_f_l516_516725


namespace equal_floor_values_exist_l516_516887

theorem equal_floor_values_exist (n : ℕ) (hn : n > 3) (A : Fin n → ℕ) (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j)
  (h_bound : ∀ i : Fin n, A i < (n - 1)!) :
  ∃ i j k l : Fin n, i ≠ j ∧ k ≠ l ∧ (i < j) ∧ (k < l) ∧ ⌊(A j / A i : ℚ)⌋ = ⌊(A l / A k : ℚ)⌋ :=
by
  sorry

end equal_floor_values_exist_l516_516887


namespace variance_of_data_set_l516_516527

def data_set : List ℤ := [ -2, -1, 0, 3, 5 ]

def mean (l : List ℤ) : ℚ :=
  (l.sum / l.length)

def variance (l : List ℤ) : ℚ :=
  (1 / l.length) * (l.map (λ x => (x - mean l : ℚ)^2)).sum

theorem variance_of_data_set : variance data_set = 34 / 5 := by
  sorry

end variance_of_data_set_l516_516527


namespace correct_number_of_conclusions_l516_516689

variables (a b c : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

def passes_through (p : ℝ × ℝ) := parabola a b c p.1 = p.2

theorem correct_number_of_conclusions : 
  (0 < a) → (a < c) → (passes_through a b c (1, 0)) → 
  (2a + b < 0) ∧ ¬(∀ x > 1, parabola a b c x > parabola a b c 1) ∧ 
  (∃ x in ℝ, (ax^2 + bx + (b + c) = 0) ∧ has_two_distinct_real_roots) →
  number_of_correct_conclusions = 2
:= sorry

end correct_number_of_conclusions_l516_516689


namespace rate_of_first_batch_l516_516272

theorem rate_of_first_batch (x : ℝ) 
  (cost_second_batch : ℝ := 20 * 14.25)
  (total_cost : ℝ := 30 * x + 285)
  (weight_mixture : ℝ := 30 + 20)
  (selling_price_per_kg : ℝ := 15.12) :
  (total_cost * 1.20 / weight_mixture = selling_price_per_kg) → x = 11.50 :=
by
  sorry

end rate_of_first_batch_l516_516272


namespace negative_integers_abs_le_4_l516_516989

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l516_516989


namespace part_1_part_2_l516_516708

theorem part_1 (a b A B : ℝ)
  (h : b * (Real.sin A)^2 = Real.sqrt 3 * a * Real.cos A * Real.sin B) 
  (h_sine_law : b / Real.sin B = a / Real.sin A)
  (A_in_range: A ∈ Set.Ioo 0 Real.pi):
  A = Real.pi / 3 := 
sorry

theorem part_2 (x : ℝ)
  (A : ℝ := Real.pi / 3)
  (h_sin_cos : ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
                f x = (Real.sin A * (Real.cos x)^2) - (Real.sin (A / 2))^2 * (Real.sin (2 * x))) :
  Set.image f (Set.Icc 0 (Real.pi / 2)) = Set.Icc ((Real.sqrt 3 - 2)/4) (Real.sqrt 3 / 2) :=
sorry

end part_1_part_2_l516_516708


namespace cube_volume_is_125_l516_516965

def original_cube_volume (n m : ℕ) (cubeCounts: ℕ) : Prop :=
  n^3 = 98 + m^3 ∧ cubeCounts = 99 ∧ (∀i, i ≠ 98 → smaller_cube_edge_len i = 1)

theorem cube_volume_is_125 {n m : ℕ} (h : original_cube_volume n m 99) : n^3 = 125 := 
  by
    -- Given conditions
    sorry

end cube_volume_is_125_l516_516965


namespace range_of_a_l516_516382

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → a ∈ Icc (-8) 0 :=
by
  sorry

end range_of_a_l516_516382


namespace fhomas_milk_probability_correct_l516_516574

noncomputable def probability_enough_milk : ℚ :=
let total_volume := 2 * 2 * 2 in
let tetrahedron_volume := (1 / 6) * 2 * 2 * 2 in
1 - (tetrahedron_volume / total_volume)

theorem fhomas_milk_probability_correct :
  probability_enough_milk = 5 / 6 :=
sorry -- Proof is not required

end fhomas_milk_probability_correct_l516_516574


namespace intersection_of_P_and_Q_l516_516329

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2 ∧ x ∈ Set.univ}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {-2, -1, 0} :=
sorry

end intersection_of_P_and_Q_l516_516329


namespace subset_sum_divisible_by_k_l516_516469

theorem subset_sum_divisible_by_k (k : ℕ) (h : k > 0) (x : Fin k → ℤ) :
  ∃ (s : Finset (Fin k)), s.nonempty ∧ (s.sum x) % k = 0 :=
by
  sorry

end subset_sum_divisible_by_k_l516_516469


namespace incenter_closest_to_median_l516_516121

variables (a b c : ℝ) (s_a s_b s_c d_a d_b d_c : ℝ)

noncomputable def median_length (a b c : ℝ) : ℝ := 
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def distance_to_median (x y median_length : ℝ) : ℝ := 
  (y - x) / (2 * median_length)

theorem incenter_closest_to_median
  (h₀ : a = 4) (h₁ : b = 5) (h₂ : c = 8) 
  (h₃ : s_a = median_length a b c)
  (h₄ : s_b = median_length b a c)
  (h₅ : s_c = median_length c a b)
  (h₆ : d_a = distance_to_median b c s_a)
  (h₇ : d_b = distance_to_median a c s_b)
  (h₈ : d_c = distance_to_median a b s_c) : 
  d_a = d_c := 
sorry

end incenter_closest_to_median_l516_516121


namespace package_cheaper_than_per_person_l516_516885

theorem package_cheaper_than_per_person (x : ℕ) :
  (90 * 6 + 10 * x < 54 * x + 8 * 3 * x) ↔ x ≥ 8 :=
by
  sorry

end package_cheaper_than_per_person_l516_516885


namespace polynomial_tangents_cyclic_impossible_l516_516815

theorem polynomial_tangents_cyclic_impossible
  (P : ℝ → ℝ) (n : ℕ)
  (h1 : ∀ x, P x = -P (-x)) -- P(x) is odd function
  (h2 : ∀ x, P x ≠ 0) -- non-zero polynomial
  (h3 : ∃ l : list ℝ, l.length = n ∧ ∀ (i : fin n), P (l.nth_le i i.is_lt) ≠ 0 ∧ P (l.nth_le i i.is_lt) ≠ P 0) -- distinct points A1, A2, ..., An
  (h4 : ∀ (i : fin n), let t := l.nth_le i i.is_lt in ∇f (P t) goes through l.nth_le (i.succ % n) (lt_of_le_of_ne (nat.mod_le _ _) i_eq_zero)) -- tangents condition
  : false :=
begin
  sorry
end

end polynomial_tangents_cyclic_impossible_l516_516815


namespace additional_men_joined_l516_516861

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined_l516_516861


namespace johns_yearly_music_cost_l516_516419

theorem johns_yearly_music_cost 
  (hours_per_month : ℕ := 20)
  (minutes_per_hour : ℕ := 60)
  (average_song_length : ℕ := 3)
  (cost_per_song : ℕ := 50) -- represented in cents to avoid decimals
  (months_per_year : ℕ := 12)
  : (hours_per_month * minutes_per_hour // average_song_length) * cost_per_song * months_per_year = 2400 * 100 := -- 2400 dollars (* 100 to represent cents)
  sorry

end johns_yearly_music_cost_l516_516419


namespace cos_of_B_in_right_triangle_l516_516666

theorem cos_of_B_in_right_triangle :
  ∀ (A B C: ℝ × ℝ) (hBC: 8) (hBA: 10) (hCA: 6),
  (C.1 = 0) ∧ (C.2 = 6) ∧
  (A.1 = 0) ∧ (A.2 = 0) ∧
  (B.1 = 8) ∧ (B.2 = 0) →
  ∃ cos_B: ℝ, cos_B = 3 / 5 :=
by
  sorry

end cos_of_B_in_right_triangle_l516_516666


namespace bijective_function_exists_zero_a_l516_516306

theorem bijective_function_exists_zero_a (f : ℝ → ℝ) (bijective_f : Function.Bijective f) : 
  (∀ x : ℝ, f(f(x)) = x^2 * f(x) + a * x^2) ↔ a = 0 :=
by
sorry

end bijective_function_exists_zero_a_l516_516306


namespace sqrt_equiv_l516_516294

theorem sqrt_equiv (x : ℝ) (hx : x < -1) :
  sqrt ((x + 1) / (2 - (x + 2) / x)) = sqrt (abs (x^2 + x) / abs (x - 2)) :=
by
  sorry

end sqrt_equiv_l516_516294


namespace ribbon_cost_comparison_l516_516220

theorem ribbon_cost_comparison 
  (A : Type)
  (yellow_ribbon_cost blue_ribbon_cost : ℕ)
  (h1 : yellow_ribbon_cost = 24)
  (h2 : blue_ribbon_cost = 36) :
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n < blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n > blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n = blue_ribbon_cost / n) :=
sorry

end ribbon_cost_comparison_l516_516220


namespace find_circle_eq_l516_516333

theorem find_circle_eq
  (C1_eq : ∀ x y : ℝ, x^2 + y^2 - 3*x = 0)
  (center_x center_y : ℝ)
  (center_x_eq : center_x = 2)
  (center_y_eq : center_y = 1)
  (P_x P_y : ℝ)
  (P_coords : P_x = 5 ∧ P_y = -2) :
  ∃ r : ℝ, (x - center_x)^2 + (y - center_y)^2 = r^2 ∧ (x + 2*y - 5 + r^2 = 0 ∧ (P_x, P_y) ∈ (x + 2*y - 5 + r^2) → r^2 = 4) :=
sorry

end find_circle_eq_l516_516333


namespace probability_not_perfect_power_correct_l516_516506

-- Define a function to check if a number is a perfect power
def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x y : ℕ), (x > 1) ∧ (y > 1) ∧ (x^y = n)

-- Counting numbers from 1 to 150 that are not perfect powers
def count_not_perfect_powers : ℕ :=
  (Finset.range 151).filter (λ n, ¬is_perfect_power n).card

-- Define the total number of elements
def total_numbers : ℕ := 150

-- Define the probability of selecting a number that is not a perfect power
def probability_not_perfect_power : ℚ :=
  count_not_perfect_powers / total_numbers

-- The theorem to prove
theorem probability_not_perfect_power_correct :
  probability_not_perfect_power = 133 / 150 :=
by {
  -- Skip the proof steps with sorry
  sorry
}

end probability_not_perfect_power_correct_l516_516506


namespace find_omega_l516_516378

theorem find_omega (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ π / 3 → sin (ω * x) ≤ sin (ω * y))
  (h3 : ∀ x y : ℝ, π / 3 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin (ω * x) ≥ sin (ω * y)) :
  ω = 3 / 2 :=
sorry

end find_omega_l516_516378


namespace bicycle_trip_distance_l516_516608

theorem bicycle_trip_distance {x : ℝ} 
  (h : real.sqrt ((10 * x) ^ 2 + (5 * x) ^ 2) = 75) : 
  x = 3 * real.sqrt 5 :=
sorry

end bicycle_trip_distance_l516_516608


namespace dice_probability_same_face_l516_516154

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516154


namespace fisher_eligibility_l516_516138

theorem fisher_eligibility (A1 A2 S : ℕ) (hA1 : A1 = 84) (hS : S = 82) :
  (S ≥ 80) → (A1 + A2 ≥ 170) → (A2 = 86) :=
by
  sorry

end fisher_eligibility_l516_516138


namespace clockTimeAtMeeting_l516_516999

-- Define the conditions for the clock's behavior
def clockRunsFast (t : ℕ) : ℕ := t / 2
def clockRunsSlow (t : ℕ) : ℕ := t * 2

-- Define the total elapsed time for one real minute according to the broken clock
def timeElapsedInOneRealMinute : ℕ := clockRunsFast 30 + clockRunsSlow 30

-- Define the total elapsed time for a given number of real minutes according to the broken clock
def elapsedTime (realMinutes : ℕ) : ℕ := realMinutes * (timeElapsedInOneRealMinute / 60)

-- Define the initial time the clock broke and the meeting time
def initialTime : ℕ := 14 * 60 -- 14:00 in minutes
def meetingTime : ℕ := 40 -- 14:40 in minutes

-- Prove the clock's time at the moment of the meeting
theorem clockTimeAtMeeting : 
  let brokenClockTime := initialTime + elapsedTime meetingTime
  in brokenClockTime = 14 * 60 + 50 :=
by sorry

end clockTimeAtMeeting_l516_516999


namespace intercepts_sum_l516_516592

theorem intercepts_sum (x y : ℝ) : (y - 3 = -3 * (x - 6)) →
  (∃ x₀, y = 0 ∧ x₀ - 6 = 1) →
  (∃ y₀, x = 0 ∧ y₀ - 3 = 18) →
  (let x₀ := 7 in 
  let y₀ := 21 in 
  x₀ + y₀ = 28) :=
by
  intros h_line h_x_intercept h_y_intercept
  -- Proof goes here
  sorry

end intercepts_sum_l516_516592


namespace positive_difference_between_median_and_mode_l516_516545

def data : List ℕ := [36, 37, 37, 38, 40, 40, 40, 41, 42, 43, 54, 55, 57, 59, 61, 61, 65, 68, 69]

def mode (l : List ℕ) : ℕ :=
  l.groupBy id l.length 
  |> List.maximumBy (λ p, p.snd.length) 
  |> Option.getOrElse (0, 0) |> fun p => p.fst

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

noncomputable def positiveDifference  (a b : ℕ) : ℕ :=
  if a >= b then a - b else b - a

theorem positive_difference_between_median_and_mode :
  positiveDifference (median data) (mode data) = 2 :=
by
  sorry

end positive_difference_between_median_and_mode_l516_516545


namespace triangle_geometry_l516_516370

theorem triangle_geometry 
  (A : ℝ × ℝ) 
  (hA : A = (5,1))
  (median_CM : ∀ x y : ℝ, 2 * x - y - 5 = 0)
  (altitude_BH : ∀ x y : ℝ, x - 2 * y - 5 = 0):
  (∀ x y : ℝ, 2 * x + y - 11 = 0) ∧
  (4, 3) ∈ {(x, y) | 2 * x + y = 11 ∧ 2 * x - y = 5} :=
by
  sorry

end triangle_geometry_l516_516370


namespace prob_no_rain_four_days_l516_516508

noncomputable def prob_rain_one_day : ℚ := 2 / 3

noncomputable def prob_no_rain_one_day : ℚ := 1 - prob_rain_one_day

def independent_events (events : List (Unit → Prop)) : Prop :=
  -- A statement about independence of events
  sorry

theorem prob_no_rain_four_days :
  let days := 4
  let prob_no_rain := prob_no_rain_one_day
  independent_events (List.replicate days (fun _ => prob_no_rain)) →
  (prob_no_rain^days) = (1/81) := 
by
  sorry

end prob_no_rain_four_days_l516_516508


namespace area_of_parallelogram_l516_516723

theorem area_of_parallelogram 
  (x1 y1 x2 y2 : ℝ)
  (hA : x1^2 + 2*y1^2 = 1)
  (hC : x2^2 + 2*y2^2 = 1) :
  let l1 : ℝ → ℝ := λ x, (y1 / x1) * x in
  let dist := |y1 * x2 - x1 * y2| / sqrt (x1^2 + y1^2) in
  let S := 2 * |x1 * y2 - x2 * y1| in
  S = sqrt (2) → S = sqrt (2) := sorry

end area_of_parallelogram_l516_516723


namespace number_of_elements_in_M_inter_N_is_zero_or_one_l516_516367

noncomputable def y_eq_f_of_x (f : ℝ → ℝ) (a b : ℝ) :=
  { p : ℝ × ℝ | p.1 ∈ set.Icc a b ∧ p.2 = f p.1 }

noncomputable def x_eq_zero :=
  { p : ℝ × ℝ | p.1 = 0 }

theorem number_of_elements_in_M_inter_N_is_zero_or_one (f : ℝ → ℝ) (a b : ℝ) :
  (set.prod (set.Icc a b) (set.range f) ∩ {x : ℝ × ℝ | x.1 = 0}).finite ∧ (set.prod (set.Icc a b) (set.range f) ∩ {x : ℝ × ℝ | x.1 = 0}).card ≤ 1 :=
by
  sorry

end number_of_elements_in_M_inter_N_is_zero_or_one_l516_516367


namespace area_ln_shape_l516_516094

noncomputable def areaEnclosed_ln : ℝ := 
  - ∫ x in (1/e : ℝ)..1, Real.log x + ∫ x in 1..e, Real.log x

theorem area_ln_shape : 
  (∫ x in (1/e : ℝ)..e, Real.log x) = areaEnclosed_ln := 
begin
  -- sorry to skip the proof
  sorry
end

end area_ln_shape_l516_516094


namespace sum_first_n_terms_l516_516337

noncomputable theory

def seq (a : ℕ → ℕ) := (a 1 = 1) ∧ ∀ n, a (n + 1) - a n = 2^n

def Sn (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = ∑ i in finset.range n, a (i + 1)

theorem sum_first_n_terms (a S : ℕ → ℕ) (h_seq : seq a) (h_sum : Sn S a) :
  ∀ n, S n = 2^(n + 1) - 2 - n :=
sorry

end sum_first_n_terms_l516_516337


namespace CD_is_diameter_of_circle_omega1_l516_516063

theorem CD_is_diameter_of_circle_omega1
  (Q A B K C D : Type)
  (ω1 ω2 : Type)
  (hQ_outside_ω1 : ¬ (Q ∈ interior ω1))
  (hQA_tangent_ω1 : QTangent ω1 A)
  (hQB_tangent_ω1 : QTangent ω1 B)
  (hω2_center_Q : center ω2 = Q)
  (hω2_through_A : A ∈ ω2)
  (hω2_through_B : B ∈ ω2)
  (hK_on_arc_AB_ω2 : K ∈ arc AB inside ω2)
  (hAK_intersects_ω1_at_C : intersects_at_second_point AK ω1 C)
  (hBK_intersects_ω1_at_D : intersects_at_second_point BK ω1 D) :
  is_diameter CD ω1 :=
sorry

end CD_is_diameter_of_circle_omega1_l516_516063


namespace problem_statement_l516_516825

-- 1. General formula for the sequence
def seq (n : ℕ) : ℕ := 2 * n - 1

-- 2. Condition given for sequence
axiom seq_cond (n : ℕ) (hn : n ≥ 1) : 
  seq 1 + (∑ i in Finset.range (n - 1), (1 : ℚ) / (bit1 (i + 1) - 1) * seq (i + 2)) + (1 : ℚ) / (bit1 n - 1) * seq n = n

-- 3. Definition of sequence b_n
def b (n : ℕ) : ℚ := 2 / (Real.sqrt (seq (n + 1)) + Real.sqrt (seq n))

-- 4. Calculation of sum T_n
def T (n : ℕ) : ℚ := (∑ i in Finset.range n, b (i + 1))

-- 5. Statement of the problem:
theorem problem_statement :
  (seq n = 2 * n - 1) ∧  
  (T 60 = 10) :=
  by
  sorry  -- Proof to be filled

end problem_statement_l516_516825


namespace smallest_prime_with_digits_sum_22_l516_516210

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l516_516210


namespace decagon_angle_Q_l516_516479

theorem decagon_angle_Q (n : ℕ) (Q : Type):
  regular_polygon n ∧ n = 10 ∧ extended_sides_meet_at Q →
  angle_measure Q = 72 :=
by
  intros
  sorry

end decagon_angle_Q_l516_516479


namespace sample_points_correlation_l516_516391

-- Define the set of sample data points
def sample_points (n : ℕ) (xs ys : ℕ → ℝ) :=
  n ≥ 2 ∧ ¬∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → xs i = xs j ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (ys i = - (1 / 2) * (xs i) + 1)

-- Define the function to calculate the correlation coefficient
def correlation_coefficient (xs ys : ℕ → ℝ) : ℝ := sorry -- Assume there's a function calculating the correlation coefficient of the data points.

theorem sample_points_correlation (n : ℕ) (xs ys : ℕ → ℝ) (h : sample_points n xs ys) :
  correlation_coefficient xs ys = -1 := 
sorry

end sample_points_correlation_l516_516391


namespace sequence_general_term_l516_516734

theorem sequence_general_term :
  ∀ (n : ℕ), n > 0 → (∃ (x : ℕ → ℤ), x 1 = 6 ∧ x 2 = 4 ∧ (∀ n ≥ 1, x (n + 2) = (x (n + 1))^2 - 4) / x n ∧ 
  x n = 8 - 2 * n) := 
by {
  sorry,
}

#eval sequence_general_term

end sequence_general_term_l516_516734


namespace graphGn_planarity_l516_516282

open Nat

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isConnected (a b : ℕ) : Prop := isPrime (a + b)

def graphGn (n : ℕ) : SimpleGraph ℕ :=
{ adj := λ a b, a ≠ b ∧ isConnected a b,
  symm := λ a b ab, ⟨ab.1.symm, ab.2⟩,
  loopless := λ a aa, aa.1 rfl }

theorem graphGn_planarity (n : ℕ) : (graphGn n).planar ↔ n ≤ 8 := sorry

end graphGn_planarity_l516_516282


namespace arithmetic_sequence_geometric_condition_l516_516947

noncomputable def S (n : ℕ) : ℕ := n * (2 + n - 1) / 2

noncomputable def a (n : ℕ) : ℕ := 1 + (n - 1) * 1

theorem arithmetic_sequence_geometric_condition :
  S 5 = 15 → 
  (a 3) * (a 12) = (a 6) * (a 6) →
  (S 2023) / (a 2023) = 1012 := 
by
  sorry

end arithmetic_sequence_geometric_condition_l516_516947


namespace minimum_value_y_l516_516807

noncomputable def y (x : ℚ) : ℚ := |3 - x| + |x - 2| + |-1 + x|

theorem minimum_value_y : ∃ x : ℚ, y x = 2 :=
by
  sorry

end minimum_value_y_l516_516807


namespace remainder_9876543210_mod_101_l516_516920

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l516_516920


namespace same_functions_C_same_functions_D_l516_516934

noncomputable def fC (x : ℝ) : ℝ := x^2 + (x - 1)^0
noncomputable def gC (x : ℝ) : ℝ := (x^3 - x^2 + x - 1) / (x - 1)

noncomputable def fD (x : ℝ) : ℝ := sqrt x + 1 / x
noncomputable def gD (t : ℝ) : ℝ := sqrt t + 1 / t

theorem same_functions_C : ∀ x, x ≠ 1 → fC x = gC x := by
  sorry

theorem same_functions_D : ∀ t, t > 0 → fD t = gD t := by
  sorry

end same_functions_C_same_functions_D_l516_516934


namespace equal_distance_l516_516812

variables {A B C D E F P Q : Type}
variables [EuclideanGeometry A B C D E F P Q]

-- Conditions:
def acute_triangle (ABC : Triangle) : Prop := 
  ∀ θ ∈ Triangle.angles ABC, θ < 90

def feet_of_perpendiculars (A B C D E F : Point) (ABC : Triangle) : Prop :=
  is_foot_of_perpendicular A D BC ∧ 
  is_foot_of_perpendicular B E CA ∧ 
  is_foot_of_perpendicular C F AB

def lies_on_circumcircle (P : Point) (ABC : Triangle) (EF : Line) : Prop :=
  is_circumcircle_point P ABC ∧ 
  P ∈ EF

def intersection_of_lines (Q : Point) (BP DF : Line) : Prop :=
  Q ∈ (BP ∩ DF)

-- The statement to prove:
theorem equal_distance {ABC : Triangle}
  (h1 : acute_triangle ABC)
  (h2 : feet_of_perpendiculars A B C D E F ABC)
  (h3 : lies_on_circumcircle P ABC (Line.mk E F))
  (h4 : intersection_of_lines Q (Line.mk B P) (Line.mk D F)) :
  dist A P = dist A Q :=
sorry

end equal_distance_l516_516812


namespace min_ratio_AB_CD_l516_516447

theorem min_ratio_AB_CD (A B C D K : Point) (h_cyclic : cyclic A B C D) 
(h_K_on_AB : K ∈ line_through A B)
(h_bisect_BD_KC : Point bisects_line BD (segment K C))
(h_bisect_AC_KD : Point bisects_line AC (segment K D)) : 
  abs (distance A B / distance C D) = 1 := 
sorry

end min_ratio_AB_CD_l516_516447


namespace obtuse_triangle_l516_516385

theorem obtuse_triangle {A B C : ℝ} (h : ∀ (A B C : ℝ), ∠A + ∠B + ∠C = π → cos A * cos B > sin A * sin B) : ∠C > π / 2 :=
by
  sorry

end obtuse_triangle_l516_516385


namespace attendance_difference_is_85_l516_516889

def saturday_attendance : ℕ := 80
def monday_attendance : ℕ := saturday_attendance - 20
def wednesday_attendance : ℕ := monday_attendance + 50
def friday_attendance : ℕ := saturday_attendance + monday_attendance
def thursday_attendance : ℕ := 45
def expected_audience : ℕ := 350

def total_attendance : ℕ := 
  saturday_attendance + 
  monday_attendance + 
  wednesday_attendance + 
  friday_attendance + 
  thursday_attendance

def more_people_attended_than_expected : ℕ :=
  total_attendance - expected_audience

theorem attendance_difference_is_85 : more_people_attended_than_expected = 85 := 
by
  unfold more_people_attended_than_expected
  unfold total_attendance
  unfold saturday_attendance
  unfold monday_attendance
  unfold wednesday_attendance
  unfold friday_attendance
  unfold thursday_attendance
  unfold expected_audience
  exact sorry

end attendance_difference_is_85_l516_516889


namespace smallest_prime_with_digit_sum_22_l516_516203

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_with_digit_sum_22 : ∃ p : ℕ, is_prime p ∧ digit_sum p = 22 ∧ 
  (∀ q : ℕ, is_prime q ∧ digit_sum q = 22 → p ≤ q) ∧ p = 499 :=
sorry

end smallest_prime_with_digit_sum_22_l516_516203


namespace solve_system_l516_516084

theorem solve_system :
  ∃ x y : ℚ, 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 :=
by {
  sorry
}

end solve_system_l516_516084


namespace min_path_sum_l516_516387

/- 
  We assume points A, B, C, and D given as 
  A(-2, -3), B(4, -1), C(m, 0), D(n, n).
  We need to prove that the minimum value of 
  AB + BC + CD + AD is equal to 58 + 2 * sqrt(10).
-/

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

variables (m n : ℝ)

theorem min_path_sum :
  let A := (-2, -3) in
  let B := (4, -1) in
  let C := (m, 0) in
  let D := (n, n) in
  distance A B + distance B C + distance C D + distance D A = 58 + 2 * real.sqrt 10 := 
sorry

end min_path_sum_l516_516387


namespace operation_value_l516_516449

-- Define the operations as per the conditions.
def star (m n : ℤ) : ℤ := n^2 - m
def hash (m k : ℤ) : ℚ := (k + 2 * m) / 3

-- State the theorem we want to prove.
theorem operation_value : hash (star 3 3) (star 2 5) = 35 / 3 :=
  by
  sorry

end operation_value_l516_516449


namespace find_c_for_two_solutions_in_real_l516_516312

noncomputable def system_two_solutions (x y c : ℝ) : Prop := (|x + y| = 2007 ∧ |x - y| = c)

theorem find_c_for_two_solutions_in_real : ∃ c : ℝ, (∀ x y : ℝ, system_two_solutions x y c) ↔ (c = 0) :=
by
  sorry

end find_c_for_two_solutions_in_real_l516_516312


namespace sum_first_20_terms_l516_516724

theorem sum_first_20_terms (a : ℕ → ℝ) (h_rec : ∀ n, a n + 1 = (a (n + 1) + 1) / (2 * a (n + 1) + 3)) (h_init : a 1 = 1) :
  ∑ i in finset.range 20, (2 / (a i + 1)) = 780 :=
sorry

end sum_first_20_terms_l516_516724


namespace unicorn_rope_length_l516_516606

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 1500
noncomputable def c : ℕ := 3

theorem unicorn_rope_length : a + b + c = 1593 :=
by
  -- The steps to prove the theorem should go here, but as stated, we skip this with "sorry".
  sorry

end unicorn_rope_length_l516_516606


namespace janice_initial_sentences_l516_516017

theorem janice_initial_sentences (
    normal_speed : ℕ,
    first_duration : ℕ,
    increased_speed : ℕ,
    second_duration : ℕ,
    correction_duration : ℕ,
    incorrectly_typed : ℕ,
    slow_speed : ℕ,
    final_duration : ℕ,
    end_of_day_sentences : ℕ
) : 
    normal_speed = 6 →
    first_duration = 10 →
    increased_speed = 7 →
    second_duration = 10 →
    correction_duration = 15 →
    incorrectly_typed = 35 →
    slow_speed = 5 →
    final_duration = 18 →
    end_of_day_sentences = 536 →
    let first_period := normal_speed * first_duration,
        second_period := increased_speed * second_duration,
        third_period := increased_speed * correction_duration - incorrectly_typed,
        fourth_period := slow_speed * final_duration,
        total_typed := first_period + second_period + third_period + fourth_period in
    end_of_day_sentences - total_typed = 246 :=
by
  intros h_normal_speed h_first_duration h_increased_speed h_second_duration h_correction_duration h_incorrectly_typed h_slow_speed h_final_duration h_end_of_day_sentences
  simp [h_normal_speed, h_first_duration, h_increased_speed, h_second_duration, h_correction_duration, h_incorrectly_typed, h_slow_speed, h_final_duration, h_end_of_day_sentences]
  let first_period := (6 : ℕ) * 10
  let second_period := (7 : ℕ) * 10
  let third_period := (7 : ℕ) * 15 - 35
  let fourth_period := (5 : ℕ) * 18
  let total_typed := first_period + second_period + third_period + fourth_period
  -- You would normally continue the proof here.
  -- We put sorry to indicate that the proof is not complete.
  sorry

end janice_initial_sentences_l516_516017


namespace leah_total_coin_value_l516_516430

variable (p n : ℕ) -- Let p be the number of pennies and n be the number of nickels

-- Leah has 15 coins consisting of pennies and nickels
axiom coin_count : p + n = 15

-- If she had three more nickels, she would have twice as many pennies as nickels
axiom conditional_equation : p = 2 * (n + 3)

-- We want to prove that the total value of Leah's coins in cents is 27
theorem leah_total_coin_value : 5 * n + p = 27 := by
  sorry

end leah_total_coin_value_l516_516430


namespace six_points_concyclic_l516_516605

open EuclideanGeometry

variables {A B C O I D E F : Point}
variables {A1 A2 B1 B2 C1 C2 : Point}
variables {circle_O : Circle}
variables {circumcircle_ABC : Circle}
variables {circle_ID circle_IE circle_IF : Circle}

-- Data given in the condition
noncomputable def problem_conditions (hABC : InscribedTriangle A B C circle_O)
  (I_is_incenter : Incenter I A B C)
  (D_on_circumcircle : OnLine AI ∧ OnCircumcircle D circumcircle_ABC)
  (E_on_circumcircle : OnLine BI ∧ OnCircumcircle E circumcircle_ABC)
  (F_on_circumcircle : OnLine CI ∧ OnCircumcircle F circumcircle_ABC)
  (ha1a2_on_BC : ∀ (A1 A2 : Point), OnDiameterCircle A1 A2 circle_ID ∧ OnSquircle A1 A2 B C)
  (hb1b2_on_CA : ∀ (B1 B2 : Point), OnDiameterCircle B1 B2 circle_IE ∧ OnSquircle B1 B2 C A)
  (hc1c2_on_AB : ∀ (C1 C2 : Point), OnDiameterCircle C1 C2 circle_IF ∧ OnSquircle C1 C2 A B) : Prop :=
concyclic_points A1 A2 B1 B2 C1 C2

theorem six_points_concyclic {A B C O I D E F : Point}
  {A1 A2 B1 B2 C1 C2 : Point}
  (hABC : InscribedTriangle A B C circle_O)
  (I_is_incenter : Incenter I A B C)
  (D_on_circumcircle : OnLine AI ∧ OnCircumcircle D circumcircle_ABC)
  (E_on_circumcircle : OnLine BI ∧ OnCircumcircle E circumcircle_ABC)
  (F_on_circumcircle : OnLine CI ∧ OnCircumcircle F circumcircle_ABC)
  (ha1a2_on_BC : ∀ (A1 A2 : Point), OnDiameterCircle A1 A2 circle_ID ∧ OnSquircle A1 A2 B C)
  (hb1b2_on_CA : ∀ (B1 B2 : Point), OnDiameterCircle B1 B2 circle_IE ∧ OnSquircle B1 B2 C A)
  (hc1c2_on_AB : ∀ (C1 C2 : Point), OnDiameterCircle C1 C2 circle_IF ∧ OnSquircle C1 C2 A B) :
  concyclic_points A1 A2 B1 B2 C1 C2 :=
begin
  apply problem_conditions;
  sorry
end

end six_points_concyclic_l516_516605


namespace problem1_problem2_problem3_problem4_l516_516623

-- Problem 1
theorem problem1 : (1 : ℤ) * (-1 : ℤ) ^ 2018 - (3 - Real.pi) ^ 0 + (- (1 : ℚ) / 3) ^ (-2) = 9 := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) : (x + 1) * (x + 3) - (x - 2) ^ 2 = 8 * x - 1 := by
  sorry
  
-- Problem 3
theorem problem3 : (199 : ℤ) ^ 2 - 199 * 201 = -398 := by
  sorry

-- Problem 4
theorem problem4 : (∏ n in (finset.range 2022).image (λ n, n + 2), 1 - 1 / (n : ℚ) ^ 2) = 1012 / 2023 := by
  have h : ∏ n in (finset.range 2022).image (λ n, n + 2), 1 - 1 / (n : ℚ)^2 = 
    ∏ n in (finset.range 2022).image (λ n, n + 2), ((n + 1) / n) * ((n - 1) / n), sorry
    -- The detailed calculation and proof would be added here
  rw h,
  sorry

end problem1_problem2_problem3_problem4_l516_516623


namespace divide_students_l516_516296

theorem divide_students :
  let students := {A, B, C, D}
  let classes := {class1, class2}
  (∃ s1 s2 : set student, s1 ≠ ∅ ∧ s2 ≠ ∅ ∧ s1 ∩ s2 = ∅ ∧ s1 ∪ s2 = students) ∧
  (A ∈ s1 ∧ B ∈ s2 ∨ A ∈ s2 ∧ B ∈ s1) →
  ∃! (partition : student → class), 
    (∀ s ∈ partition, s ≠ ∅) ∧ 
    (∃ (f : student → class), ∀ (s t : student), s ≠ t → f s ≠ f t) ∧ 
    {(C, class1), (C, class2), (D, class1), (D, class2)}.card = 6 :=
sorry

end divide_students_l516_516296


namespace tower_count_l516_516582

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multinomialCoeff (n : Nat) (ks : List Nat) : Nat :=
  factorial n / List.foldr (fun k acc => acc * factorial k) 1 ks

theorem tower_count :
  let totalCubes := 9
  let usedCubes := 8
  let redCubes := 2
  let blueCubes := 3
  let greenCubes := 4
  multinomialCoeff totalCubes [redCubes, blueCubes, greenCubes] = 1260 :=
by
  sorry

end tower_count_l516_516582


namespace maximum_elephants_l516_516229

theorem maximum_elephants (e_1 e_2 : ℕ) :
  (∃ e_1 e_2 : ℕ, 28 * e_1 + 37 * e_2 = 1036 ∧ (∀ k, 28 * e_1 + 37 * e_2 = k → k ≤ 1036 )) → 
  28 * e_1 + 37 * e_2 = 1036 :=
sorry

end maximum_elephants_l516_516229


namespace geom_sequence_sum_l516_516541

noncomputable def sequence (n : ℕ) : ℕ := 
  if (n % 3 = 1) then 1
  else if (n % 3 = 2) then 2
  else 3

theorem geom_sequence_sum :
  sequence 1 = 1 ∧ sequence 2 = 2 ∧ (∀ n : ℕ, n > 0 → sequence n * sequence (n + 1) * sequence (n + 2) = 6) →
  ∑ i in finset.range 9, sequence (i + 1) = 18 :=
by
  sorry

end geom_sequence_sum_l516_516541


namespace sum_possible_values_of_g_34_l516_516039

def f (x : ℝ) : ℝ := 4 * x^2 + 2
def g (y : ℝ) : ℝ := ∃ x : ℝ, y = 4 * x^2 + 2 ∧ x^2 + x + 1

theorem sum_possible_values_of_g_34 : g 34 = 18 :=
by
  sorry

end sum_possible_values_of_g_34_l516_516039


namespace find_x_y_z_of_fold_points_l516_516690

noncomputable def area_of_fold_points_of_triangle (DE DF : ℝ) (angle_E : ℝ) : ℝ :=
  let radius_DE := DE / 2
  let radius_DF := DF / 2
  (1 / 2) * Real.pi * radius_DE^2

theorem find_x_y_z_of_fold_points (DE DF : ℝ) (angle_E : ℝ) (H_DE : DE = 20) (H_DF : DF = 40) (H_angle : angle_E = 90) :
  area_of_fold_points_of_triangle DE DF angle_E = 50 * Real.pi ∧ 50 + 0 + 1 = 51 := by
  sorry

end find_x_y_z_of_fold_points_l516_516690


namespace train_speed_proof_l516_516260

noncomputable def train_speed_kmph (length_train length_bridge : ℕ) (time_seconds : ℚ) : ℚ :=
  ((length_train + length_bridge) * 3.6) / time_seconds

theorem train_speed_proof :
  train_speed_kmph 100 200 29.997600191984642 ≈ 36.003 :=
sorry

end train_speed_proof_l516_516260


namespace average_height_13_year_old_boys_country_l516_516139

def total_height_north (num_boys_north : ℕ) (avg_height_north : ℝ) := num_boys_north * avg_height_north
def total_height_south (num_boys_south : ℕ) (avg_height_south : ℝ) := num_boys_south * avg_height_south
def total_boys (num_boys_north num_boys_south : ℕ) := num_boys_north + num_boys_south
def overall_average_height (total_height_north total_height_south : ℝ) (total_boys : ℕ) := (total_height_north + total_height_south) / total_boys

theorem average_height_13_year_old_boys_country :
  overall_average_height (total_height_north 300 1.6) (total_height_south 200 1.5) 500 = 1.56 :=
by
  sorry

end average_height_13_year_old_boys_country_l516_516139


namespace John_pays_2400_per_year_l516_516423

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l516_516423


namespace second_mechanic_charge_per_hour_is_85_l516_516142

noncomputable def second_mechanic_hourly_rate : ℕ :=
sorry

theorem second_mechanic_charge_per_hour_is_85 :
  ∃ x : ℕ, 
    (∀ (first_mechanic_rate : ℕ) (total_hours : ℕ) (total_charge : ℕ) (second_mechanic_hours : ℕ),
        first_mechanic_rate = 45 ∧
        total_hours = 20 ∧
        total_charge = 1100 ∧
        second_mechanic_hours = 5 →
        x = (total_charge - (first_mechanic_rate * (total_hours - second_mechanic_hours))) / second_mechanic_hours) ∧
    x = 85 :=
begin
  use 85,
  split,
  { intros first_mechanic_rate total_hours total_charge second_mechanic_hours,
    rintro ⟨h1, h2, h3, h4⟩,
    rw [h1, h2, h3, h4],
    norm_num },
  { refl }
end

end second_mechanic_charge_per_hour_is_85_l516_516142


namespace log_recurring_eq_l516_516641

theorem log_recurring_eq : 
  ∃ x : ℝ, 0 < x ∧ x = Real.log 3 (64 + Real.log 3 (64 + Real.log 3 (64 + ...))) ∧ x ≈ 4.2 := 
sorry

end log_recurring_eq_l516_516641


namespace sqrt_9_eq_3_sqrt_neg_4_sq_eq_4_cbrt_neg_8_eq_neg_2_l516_516281

theorem sqrt_9_eq_3 : sqrt 9 = 3 := by
  sorry

theorem sqrt_neg_4_sq_eq_4 : sqrt ((-4) ^ 2) = 4 := by
  sorry

theorem cbrt_neg_8_eq_neg_2 : real.cbrt (-8) = -2 := by
  sorry

end sqrt_9_eq_3_sqrt_neg_4_sq_eq_4_cbrt_neg_8_eq_neg_2_l516_516281


namespace work_together_days_l516_516225

theorem work_together_days (ravi_days prakash_days : ℕ) (hr : ravi_days = 50) (hp : prakash_days = 75) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 30 :=
sorry

end work_together_days_l516_516225


namespace harmonic_sum_is_integer_l516_516288

def harmonic_sum (n : ℕ) : ℚ :=
  (finset.range n).sum (λ i, 1 / (i + 1 : ℚ))

theorem harmonic_sum_is_integer (n : ℕ) : harmonic_sum n = 1 ↔ n = 1 :=
by
  sorry

end harmonic_sum_is_integer_l516_516288


namespace max_min_PA_l516_516360

noncomputable def curve_parametric : ℝ → (ℝ × ℝ) := 
  λ θ, (2 * Real.cos θ, 3 * Real.sin θ)

noncomputable def line_parametric (t : ℝ) : (ℝ × ℝ) :=
  (2 + t, 2 - 2 * t)

def line_standard (x y : ℝ) : Prop := 2 * x + y = 6

lemma distance_from_point_to_line (x y : ℝ) : ℝ :=
  Real.abs ((4 * x + 3 * y - 6) / Real.sqrt 5)

lemma parametric_point_on_curve (θ : ℝ) : 
  curve_parametric θ = (2 * Real.cos θ, 3 * Real.sin θ) :=
by sorry

lemma distance_from_P_to_l (θ : ℝ) : ℝ :=
  let P := curve_parametric θ in
  distance_from_point_to_line P.1 P.2

lemma calculate_PA (θ α : ℝ) : ℝ :=
  let d := distance_from_P_to_l θ in
  (d / Real.sin (π / 6))

theorem max_min_PA (θ α : ℝ) (h1 : ∀ α, 0 ≤ α ∧ α ≤ π / 2) :
  let PA := calculate_PA θ α in
  PA ≤ (22 * Real.sqrt 5 / 5) ∧ PA ≥ (2 * Real.sqrt 5 / 5) :=
by sorry

end max_min_PA_l516_516360


namespace inequality_empty_solution_range_l516_516126

/-
Proof problem:
Prove that for the inequality |x-3| + |x-a| < 1 to have no solutions, the range of a must be (-∞, 2] ∪ [4, +∞).
-/

theorem inequality_empty_solution_range (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - a| < 1 → false) ↔ a ∈ set.Iic 2 ∪ set.Ici 4 := sorry

end inequality_empty_solution_range_l516_516126


namespace parabola_expression_l516_516465

theorem parabola_expression (a b : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, (a*x*x + b*x + 2) = 2 ↔ x = 0) :
  ∃ P : ℝ → ℝ, (P = λ x, a*x^2 + b*x + 2) ∧ (∀ x, P x = x^2 + 2) :=
begin
  have : ∀ x, a = 1 → b = 0 → a * x^2 + b * x + 2 = x^2 + 2, by simp [h1],
  sorry,
end

end parabola_expression_l516_516465


namespace solve_for_a_l516_516357
noncomputable theory

-- Definitions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a^x else -(a^(-x))

axiom log_base_half_four : log (1/2) 4 = -2

-- Theorem Statement
theorem solve_for_a (a : ℝ) :
  odd_function (f a) ∧ f a (log_base_half_four) = -3 ∧ (∀ x, x > 0 → f a x = a^x) →
  a = sqrt 3 :=
begin
  sorry
end

end solve_for_a_l516_516357


namespace geometric_sequence_relationship_l516_516688

theorem geometric_sequence_relationship
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hpos : ∀ n, a n > 0)
  (hsum : ∀ n, S n = (∑ k in finset.range n, a k))
  (ha3a7 : a 3 * a 7 = 16 * a 5)
  (ha3a5 : a 3 + a 5 = 20) :
  ∀ n, S n = 2 * a n - 1 :=
sorry

end geometric_sequence_relationship_l516_516688


namespace pens_given_away_l516_516060

theorem pens_given_away (initial_pens : ℕ) (pens_left : ℕ) (n : ℕ) (h1 : initial_pens = 56) (h2 : pens_left = 34) (h3 : n = initial_pens - pens_left) : n = 22 := by
  -- The proof is omitted
  sorry

end pens_given_away_l516_516060


namespace total_cups_of_coffee_l516_516070

theorem total_cups_of_coffee (cups_sandra cups_marcie : ℕ) (h1 : cups_sandra = 6) (h2 : cups_marcie = 2) : cups_sandra + cups_marcie = 8 :=
by
  rw [h1, h2]
  simp

end total_cups_of_coffee_l516_516070


namespace steve_cookie_boxes_l516_516087

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end steve_cookie_boxes_l516_516087


namespace probability_product_zero_l516_516894

theorem probability_product_zero :
  let S := {-3, -1, 0, 0, 4, 5}
      (favorable : Finset (ℤ × ℤ)) 
        := { (-3, 0), (-1, 0), (0, 4), (0, 5), (0, -3), (0, -1), (0, 4), (0, 5) } 
      in
      let outcomes := (S.product S).filter (λ (p : ℤ × ℤ), p.fst ≠ p.snd) in
      (∃ (a b ∈ S, a ≠ b), 
        (0 < outcomes.card) ∧ 
        favorable.card = 8 ∧ 
        outcomes.card = 15 ∧ 
        favorable.card/to_float outcomes.card = 8/15)
        :=
begin
  let S : set ℤ := {-3, -1, 0, 0, 4, 5},
  let favorable : finset (ℤ × ℤ) := {(0, -3), (0, -1), (0, 4), (0, 5), 
                                    (-3, 0), (-1, 0), (4, 0), (5, 0)},
  let outcomes := finset.product (finset.filter (λ a, a ≠ 0) $ S.to_finset) 
                                (finset.filter (λ a, a ≠ 0) $ S.to_finset),
  
  have h1 : outcomes.card = 15 := sorry,
  have h2 : favorable.card = 8 := sorry,
  have h3 : ¬∅ := sorry,
  exact ⟨S, favorable, h1, h2, h3⟩,
end

end probability_product_zero_l516_516894


namespace book_difference_l516_516504

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18
def difference : ℕ := initial_books - borrowed_books

theorem book_difference : difference = 57 := by
  -- Proof will go here
  sorry

end book_difference_l516_516504


namespace average_speed_is_37_5_l516_516238

-- Define the conditions
def distance_local : ℕ := 60
def speed_local : ℕ := 30
def distance_gravel : ℕ := 10
def speed_gravel : ℕ := 20
def distance_highway : ℕ := 105
def speed_highway : ℕ := 60
def traffic_delay : ℚ := 15 / 60
def obstruction_delay : ℚ := 10 / 60

-- Define the total distance
def total_distance : ℕ := distance_local + distance_gravel + distance_highway

-- Define the total time
def total_time : ℚ :=
  (distance_local / speed_local) +
  (distance_gravel / speed_gravel) +
  (distance_highway / speed_highway) +
  traffic_delay +
  obstruction_delay

-- Define the average speed as distance divided by time
def average_speed : ℚ := total_distance / total_time

theorem average_speed_is_37_5 :
  average_speed = 37.5 := by sorry

end average_speed_is_37_5_l516_516238


namespace cos_A_value_triangle_shape_l516_516759

variables {A B C a b c : ℝ}

-- Condition: (sin B + sin C + sin A)(sin B + sin C - sin A) = (18 / 5) * sin B * sin C
def condition1 : Prop :=
  (Real.sin B + Real.sin C + Real.sin A) * (Real.sin B + Real.sin C - Real.sin A) = (18 / 5) * Real.sin B * Real.sin C

-- Condition: b and c are the roots of the equation x^2 - 9x + 25 * cos A = 0 with b > c
def condition2 : Prop :=
  (Polynomial.roots (Polynomial.X ^ 2 - 9 * Polynomial.X + 25 * Real.cos A) = {b, c}) ∧ b > c

-- Prove: cos A = 4 / 5 under the given conditions
theorem cos_A_value (h1 : condition1) (h2 : condition2) : Real.cos A = 4 / 5 := 
sorry

-- Prove: the shape of triangle ABC is a right-angle triangle
theorem triangle_shape (h1 : condition1) (h2 : condition2) (h3 : Real.cos A = 4 / 5) : 
  a^2 + c^2 = b^2 :=
sorry

end cos_A_value_triangle_shape_l516_516759


namespace slower_train_pass_time_l516_516563

noncomputable def calculate_time_to_pass (length_train : ℕ) (speed_train1 : ℕ) (speed_train2 : ℕ) 
  (conversion_factor : ℕ) : ℚ :=
  let speed1_m_s := speed_train1 * conversion_factor / 3600
  let speed2_m_s := speed_train2 * conversion_factor / 3600
  let relative_speed := speed1_m_s + speed2_m_s
  length_train / relative_speed

theorem slower_train_pass_time : 
  calculate_time_to_pass 500 45 30 1000 ≈ 24.01 := 
sorry

end slower_train_pass_time_l516_516563


namespace loan_amount_principal_l516_516841

-- Definitions based on conditions
def rate_of_interest := 3
def time_period := 3
def simple_interest := 108

-- Question translated to Lean 4 statement
theorem loan_amount_principal : ∃ P, (simple_interest = (P * rate_of_interest * time_period) / 100) ∧ P = 1200 :=
sorry

end loan_amount_principal_l516_516841


namespace expand_expression_l516_516298

theorem expand_expression (x : ℂ) :
  (x^15 - 4*x^8 + 2*x^(-3) - 9) * (3*x^3) = 3*x^18 - 12*x^11 - 27*x^3 + 6 := 
by 
  -- Sorry is placed here to indicate where the proof would go:
  sorry

end expand_expression_l516_516298


namespace multiplication_relationship_l516_516560

theorem multiplication_relationship (a b : ℝ) (h : a * b = 177) (ha : a = 2994) (hb : b = 14.5) :
    (a / 100) * (b / 10) = 0.177 :=
by
  have ha' : a / 100 = 29.94, by rw [ha]; exact (div_eq_inv_mul (2994 : ℝ) (100 : ℝ)).symm ▸ rfl
  have hb' : b / 10 = 1.45, by rw [hb]; exact (div_eq_inv_mul (14.5 : ℝ) (10 : ℝ)).symm ▸ rfl
  rw [ha', hb']
  calc 29.94 * 1.45 = 2994 / 100 * (14.5 / 10) : by rw [ha', hb']
            ... = (2994 * 14.5) / (100 * 10) : by exact (mul_div_left_comm (2994 : ℝ) (14.5 : ℝ) 100 10)
            ... = 177 / 1000               : by rw [h, mul_comm]
            ... = 0.177                    : by norm_num

end multiplication_relationship_l516_516560


namespace intersection_volume_correct_l516_516633

noncomputable def volume_intersection_spheres 
  (c1 c2 : ℝ × ℝ × ℝ) (r1 r2 : ℝ) (d : ℝ) : ℝ :=
  (π / (12 * d)) * ((r1 + r2 - d)^2) * (d^2 + 2 * d * (r1 + r2) - 3 * (r1 - r2)^2)

theorem intersection_volume_correct :
  (volume_intersection_spheres (0, 0, 10) (0, 0, 2) 5 4 8) = (205 * π / 96) :=
sorry

end intersection_volume_correct_l516_516633


namespace intersection_complement_l516_516803

open Set
open Classical

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def R := ℝ
def U_def := {x : ℝ | x ∈ R }
def A_def := {x : ℝ | x > 0}
def B_def := {x : ℝ | x > 1}

theorem intersection_complement :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x <= 1} :=
by
  sorry

end intersection_complement_l516_516803


namespace Nikita_is_mistaken_l516_516553

noncomputable theory
open_locale classical

def initial_black_pen_cost : ℕ := 9
def initial_blue_pen_cost : ℕ := 4
def new_black_pen_cost : ℕ := 4
def new_blue_pen_cost : ℕ := 9

def total_cost_initial (b s : ℕ) : ℕ := initial_black_pen_cost * b + initial_blue_pen_cost * s
def total_cost_new (b s : ℕ) : ℕ := new_black_pen_cost * b + new_blue_pen_cost * s
def cost_difference (b s : ℕ) : ℕ := total_cost_initial b s - total_cost_new b s

theorem Nikita_is_mistaken (b s : ℕ) (h : cost_difference b s = 49) : False :=
by 
   have : 5 * (b - s) = 49 := by sorry
   sorry

end Nikita_is_mistaken_l516_516553


namespace dice_probability_same_face_l516_516157

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516157


namespace rate_at_which_bowls_were_bought_l516_516597

theorem rate_at_which_bowls_were_bought 
    (total_bowls : ℕ) (sold_bowls : ℕ) (price_per_sold_bowl : ℝ) (remaining_bowls : ℕ) (percentage_gain : ℝ) 
    (total_bowls_eq : total_bowls = 115) 
    (sold_bowls_eq : sold_bowls = 104) 
    (price_per_sold_bowl_eq : price_per_sold_bowl = 20) 
    (remaining_bowls_eq : remaining_bowls = 11) 
    (percentage_gain_eq : percentage_gain = 0.4830917874396135) 
  : ∃ (R : ℝ), R = 18 :=
  sorry

end rate_at_which_bowls_were_bought_l516_516597


namespace problem1_problem2_prob_dist_problem2_expectation_l516_516144

noncomputable def probability_A_wins_match_B_wins_once (pA pB : ℚ) : ℚ :=
  (pB * pA * pA) + (pA * pB * pA * pA)

theorem problem1 : probability_A_wins_match_B_wins_once (2/3) (1/3) = 20/81 :=
  by sorry

noncomputable def P_X (x : ℕ) (pA pB : ℚ) : ℚ :=
  match x with
  | 2 => pA^2 + pB^2
  | 3 => pB * pA^2 + pA * pB^2
  | 4 => (pA * pB * pA * pA) + (pB * pA * pB * pB)
  | 5 => (pB * pA * pB * pA) + (pA * pB * pA * pB)
  | _ => 0

theorem problem2_prob_dist : 
  P_X 2 (2/3) (1/3) = 5/9 ∧
  P_X 3 (2/3) (1/3) = 2/9 ∧
  P_X 4 (2/3) (1/3) = 10/81 ∧
  P_X 5 (2/3) (1/3) = 8/81 :=
  by sorry

noncomputable def E_X (pA pB : ℚ) : ℚ :=
  2 * (P_X 2 pA pB) + 3 * (P_X 3 pA pB) + 
  4 * (P_X 4 pA pB) + 5 * (P_X 5 pA pB)

theorem problem2_expectation : E_X (2/3) (1/3) = 224/81 :=
  by sorry

end problem1_problem2_prob_dist_problem2_expectation_l516_516144


namespace min_area_triangle_ABO_eq_4_min_product_MA_MB_eq_4_l516_516368

-- Definition of line l under given conditions
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Coordinates of points A and B
def point_A (k : ℝ) : ℝ × ℝ := (-2 - (1 / k), 0)
def point_B (k : ℝ) : ℝ × ℝ := (0, 1 + 2 * k)

-- Area of triangle ABO
noncomputable def area_triangle_ABO (k : ℝ) : ℝ :=
  0.5 * (1 + 2 * k) * (2 + (1 / k))

-- Statement for minimum value of area S
theorem min_area_triangle_ABO_eq_4 :
  ∃ (k : ℝ) (h : k > 0), area_triangle_ABO k = 4 ∧ line_l k x y → x - 2*y + 4 = 0 :=
sorry

-- Vector distances for points A and B from point M
def vec_MA (k : ℝ) : ℝ × ℝ := (- (1 / k), -1)
def vec_MB (k : ℝ) : ℝ × ℝ := (2, 2 * k)

-- Dot product of vectors MA and MB
noncomputable def dot_product_MA_MB (k : ℝ) : ℝ :=
  -(vec_MA k).fst * (vec_MB k).fst - (vec_MA k).snd * (vec_MB k).snd

-- Statement for minimum value of |MA| * |MB|
theorem min_product_MA_MB_eq_4 :
  ∃ (k : ℝ) (h : k > 0), dot_product_MA_MB k = 4 :=
sorry

end min_area_triangle_ABO_eq_4_min_product_MA_MB_eq_4_l516_516368


namespace proof_problem_l516_516567

variables {n k : ℕ} (A : Finset ℕ) (b : ℕ)

-- Assume 1 < k ≤ n
variables (h1 : 1 < k) (h2 : k ≤ n)

-- Define the conditions
def condition1 : Prop :=
  ∀ S ⊆ A, S.card = k-1 → ¬ (b ∣ ∏ x in S, x)

def condition2 : Prop :=
  ∀ S ⊆ A, S.card = k → b ∣ ∏ x in S, x

def condition3 : Prop :=
  ∀ (a a' ∈ A), a ≠ a' → ¬ (a ∣ a')

-- Define the problem in terms of the above conditions
theorem proof_problem (hA1 : condition1 A b)
    (hA2 : condition2 A b)
    (hA3: condition3 A) :
    ∃ b A, A = {2 * p | p ∈ (Finset.range n).filter Prime} ∧ b = 2^k := 
sorry

end proof_problem_l516_516567


namespace sequence_is_geometric_not_arithmetic_l516_516757

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

def S_n (n : ℕ) : ℕ :=
  2^n - 1

theorem sequence_is_geometric_not_arithmetic (n : ℕ) : 
  (∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) ∧
  (a_n 1 = 1) ∧
  (∃ r : ℕ, r > 1 ∧ ∀ n ≥ 1, a_n (n + 1) = r * a_n n) ∧
  ¬(∃ d : ℤ, ∀ n, (a_n (n + 1) : ℤ) = a_n n + d) :=
by
  sorry

end sequence_is_geometric_not_arithmetic_l516_516757


namespace polar_to_rectangular_coords_l516_516285

theorem polar_to_rectangular_coords :
  ∀ (r θ : ℝ), r = 10 → θ = 5 * Real.pi / 4 →
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2, -5 * Real.sqrt 2) :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  have : Real.cos (5 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
    by sorry -- placeholder for the trigonometric calculation
  have : Real.sin (5 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
    by sorry -- placeholder for the trigonometric calculation
  rw [this, this]
  sorry -- placeholder for the remaining algebraic simplifications

end polar_to_rectangular_coords_l516_516285


namespace internal_angle_sum_l516_516824

noncomputable def f (A B x : ℝ) : ℝ :=
  (sin B / cos A) ^ x + (sin A / cos B) ^ x

theorem internal_angle_sum (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (h : ∀ x > 0, f A B x < 2) :
  0 < A + B ∧ A + B < π / 2 :=
by
  sorry

end internal_angle_sum_l516_516824


namespace number_of_cirrus_clouds_l516_516878

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l516_516878


namespace same_number_on_four_dice_l516_516192

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516192


namespace non_union_women_percentage_l516_516764

theorem non_union_women_percentage :
  let total_employees := 2000
  let percent_women := 0.52
  let percent_men := 0.48
  let department_A := 0.40
  let department_B := 0.35
  let department_C := 0.25
  let unionized_A := 0.60
  let unionized_B := 0.70
  let unionized_C := 0.80
  
  let total_women := total_employees * percent_women
  let total_men := total_employees * percent_men

  let employees_A := total_employees * department_A
  let employees_B := total_employees * department_B
  let employees_C := total_employees * department_C

  let union_employees_A := employees_A * unionized_A
  let union_employees_B := employees_B * unionized_B
  let union_employees_C := employees_C * unionized_C

  let non_union_employees_A := employees_A - union_employees_A
  let non_union_employees_B := employees_B - union_employees_B
  let non_union_employees_C := employees_C - union_employees_C

  let non_union_women_A := non_union_employees_A * percent_women
  let non_union_women_B := non_union_employees_B * percent_women
  let non_union_women_C := non_union_employees_C * percent_women

  let percent_non_union_women_A := (non_union_women_A / non_union_employees_A) * 100
  let percent_non_union_women_B := (non_union_women_B / non_union_employees_B) * 100
  let percent_non_union_women_C := (non_union_women_C / non_union_employees_C) * 100
  
  percent_non_union_women_A ≈ 52 ∧ 
  percent_non_union_women_B ≈ 52 ∧ 
  percent_non_union_women_C ≈ 52 :=
begin
  -- the proof would go here
  sorry
end

end non_union_women_percentage_l516_516764


namespace ned_sells_25_mice_per_day_l516_516457

section
variable (normal_mouse_cost : ℝ) (left_mouse_increment : ℝ) (weekly_earnings : ℝ)
variable (days_open_per_week : ℝ)

def left_mouse_cost (normal_mouse_cost : ℝ) (increment : ℝ) : ℝ :=
  normal_mouse_cost * (1 + increment)

def daily_earnings (weekly_earnings : ℝ) (days_open : ℝ) : ℝ :=
  weekly_earnings / days_open

def mice_sold_per_day (daily_earnings : ℝ) (mouse_cost : ℝ) : ℝ :=
  daily_earnings / mouse_cost

theorem ned_sells_25_mice_per_day :
  ∀ (normal_mouse_cost : ℝ) (left_mouse_increment : ℝ) (weekly_earnings : ℝ) (days_open_per_week : ℝ),
    normal_mouse_cost = 120 →
    left_mouse_increment = 0.3 →
    weekly_earnings = 15600 →
    days_open_per_week = 4 →
    mice_sold_per_day (daily_earnings weekly_earnings days_open_per_week) (left_mouse_cost normal_mouse_cost left_mouse_increment) = 25 :=
by
  intro normal_mouse_cost left_mouse_increment weekly_earnings days_open_per_week h1 h2 h3 h4
  unfold left_mouse_cost daily_earnings mice_sold_per_day
  rw [h1, h2, h3, h4]
  norm_num
  sorry
end

end ned_sells_25_mice_per_day_l516_516457


namespace find_principal_l516_516258

-- Define the conditions as given
variables (R P : ℝ) (t : ℝ := 10) (extra_interest : ℝ := 210)

-- The given conditions
def original_interest := P * R * t / 100
def increased_interest := P * (R + 3) * t / 100
def interest_difference := increased_interest - original_interest = extra_interest

-- The equivalent proof problem
theorem find_principal (h : interest_difference) : P = 700 :=
sorry

end find_principal_l516_516258


namespace part1_part2_l516_516348

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B (a : ℝ) := {x : ℝ | (x - a) * (x - a - 1) < 0}

theorem part1 (a : ℝ) : (1 ∈ set_B a) → 0 < a ∧ a < 1 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, x ∈ set_B a → x ∈ set_A) ∧ (∃ x, x ∉ set_B a ∧ x ∈ set_A) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l516_516348


namespace obtuse_angle_condition_l516_516371

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (λ : ℝ) : ℝ × ℝ := (-1, λ)

-- Definition of dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Statement about the obtuse angle condition
theorem obtuse_angle_condition (λ : ℝ) :
  dot_product a (b λ) < 0 ↔ (λ > -1/2 ∧ λ ≠ 2) :=
by
  sorry

end obtuse_angle_condition_l516_516371


namespace find_f4_f10_l516_516356

noncomputable def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, x ∈ set.univ

axiom symmetry_x1 : ∀ x : ℝ, f (2 - x) = f x

axiom symmetry_x2 : ∀ x : ℝ, f (4 - x) = f x

axiom initial_value : f 0 = 1

theorem find_f4_f10 : f 4 + f 10 = 2 :=
by sorry

end find_f4_f10_l516_516356


namespace fencing_required_l516_516254

-- Conditions
def L : ℕ := 20
def A : ℕ := 680

-- Statement to prove
theorem fencing_required : ∃ W : ℕ, A = L * W ∧ 2 * W + L = 88 :=
by
  -- Here you would normally need the logical steps to arrive at the proof
  sorry

end fencing_required_l516_516254


namespace sum_x_coordinates_eq_4_5_l516_516495

noncomputable def g : ℝ → ℝ
| x if -4 ≤ x ∧ x ≤ -2 := 2 * x
| x if -2 < x ∧ x ≤ -1 := -x - 1
| x if -1 < x ∧ x ≤ 1 := -x + 3
| x if 1 < x ∧ x ≤ 2 := x + 1
| x if 2 < x ∧ x ≤ 4 := 2 * x - 3
| _ := 0  -- default case to cover all other values

theorem sum_x_coordinates_eq_4_5 : 
  let x_coords := {x | g x = 2.5} in 
  (finset.sum (finset.filter (λ x, g x = 2.5) (finset.range (5))) id) = 4.5 :=
sorry

end sum_x_coordinates_eq_4_5_l516_516495


namespace angle_measure_Q_l516_516481

def decagon := sorry -- Define symbolically as a regular decagon

-- Sides BJ and EF are part of decagon lack specificity but we'll refer symbolically
-- In Lean, we'll define the intersection point of the extended sides as Q.

def sides_extended_intersect_at_Q (d : decagon) : Prop :=
  ∃ Q, isRegularDecagon d ∧ (sidesExtended d.overlineBJ d.overlineEF Q)

-- Given BJ and EF of decagon are extended to meet at Q, 
-- prove angle measure at Q is 72 degrees
theorem angle_measure_Q {d : decagon} (h1 : isRegularDecagon d) (h2 : sides_extended_intersect_at_Q d) :
  angleMeasureQ d = 72 := by sorry

end angle_measure_Q_l516_516481


namespace longest_chord_eq_l516_516859

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2 * x - 3 = 0

def point_p := (0, 1 : ℝ)

theorem longest_chord_eq :
  ∀ (x y : ℝ), (circle_eq x y → (x, y) = point_p) → x + y - 1 = 0 :=
by
  intros x y h
  sorry

end longest_chord_eq_l516_516859


namespace volume_truncated_cone_surface_area_lateral_truncated_cone_l516_516877

-- Definition of the conditions
variables {R r k : ℝ} (h : R > r) (h_k : k > 0)

-- Volume of the truncated cone
theorem volume_truncated_cone :
  (R > r) →
  let K := k * (R + r) ^ 2 * (R ^ 2 + R * r + r ^ 2) / (R * r * (R ^ 2 + 3 * R * r + r ^ 2))
  in K = K := 
sorry

-- Surface area of the lateral surface of the truncated cone
theorem surface_area_lateral_truncated_cone :
  (R > r) →
  let F := k * (R + r) * 
              sqrt ((R ^ 2 + 3 * R * r + r ^ 2) ^ 2 * ℝ.pi ^ 2 * R ^ 2 * r ^ 2 * (R - r) ^ 2 + 9 * k ^ 2 * (R + r) ^ 4) / 
              (R * r * (R ^ 2 + 3 * R * r + r ^ 2))
  in F = F :=
sorry

end volume_truncated_cone_surface_area_lateral_truncated_cone_l516_516877


namespace seating_arrangements_around_round_table_l516_516131

theorem seating_arrangements_around_round_table (n : ℕ) (h : n = 7) :
  (nat.factorial n) / n = 720 :=
by
  simp [nat.factorial, h]
  rw [h]
  sorry

end seating_arrangements_around_round_table_l516_516131


namespace hyperbola_center_l516_516589

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = -2) (h3 : x2 = 10) (h4 : y2 = 6) : 
  let xm := (x1 + x2) / 2 in
  let ym := (y1 + y2) / 2 in
  (xm, ym) = (8, 2) := by
  sorry

end hyperbola_center_l516_516589


namespace missing_digits_in_mean_sequence_l516_516283

def sequence (n : Nat) : Nat := 8 * (List.range (n + 1)).map (λ i => 10^i).foldl (+) 0

def mean_sequence : Nat := (List.range 9).map sequence |> List.sum / 9

theorem missing_digits_in_mean_sequence : 
  ∀ d ∈ [5, 6, 8], ¬(List.nodup (Nat.digits 10 mean_sequence) ∧ d ∈ Nat.digits 10 mean_sequence) :=
by sorry

end missing_digits_in_mean_sequence_l516_516283


namespace perfect_number_is_28_l516_516600

def is_perfect_number (n : ℕ) : Prop :=
  ∑ d in (finset.range (n + 1)).filter (λ d, n % d = 0), d = 2 * n

def sum_of_reciprocals (factors : finset ℕ) : ℚ :=
  ∑ d in factors, (1 : ℚ) / d

noncomputable def perfect_number_28 := 28

theorem perfect_number_is_28 :
  is_perfect_number perfect_number_28 ∧
  sum_of_reciprocals ((finset.range (perfect_number_28 + 1)).filter (λ d, perfect_number_28 % d = 0)) = 2 :=
by
  sorry

end perfect_number_is_28_l516_516600


namespace sin_A_of_triangle_area_and_geometric_mean_l516_516377

theorem sin_A_of_triangle_area_and_geometric_mean 
  (s r : ℝ) (A : ℝ) 
  (h1 : 1 / 2 * s * r * sin A = 50)
  (h2 : s * r = 225) : 
  sin A = 4 / 9 := 
  sorry

end sin_A_of_triangle_area_and_geometric_mean_l516_516377


namespace laptop_sale_price_l516_516971

theorem laptop_sale_price
  (original_price : ℝ) 
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (delivery_fee : ℝ) 
  (price_after_first_discount : ℝ) 
  (price_after_second_discount : ℝ)
  (final_sale_price : ℝ) : 
    original_price = 500 → 
    discount1 = 0.10 → 
    discount2 = 0.20 → 
    delivery_fee = 30 → 
    price_after_first_discount = original_price * (1 - discount1) → 
    price_after_second_discount = price_after_first_discount * (1 - discount2) → 
    final_sale_price = price_after_second_discount + delivery_fee → 
    final_sale_price = 390 :=
by
  intros hOrig hDisc1 hDisc2 hDelivery hPrice1 hPrice2 hFinal
  rw [hOrig, hDisc1, hDisc2, hDelivery] at *
  rw hPrice1
  rw hPrice2
  rw hFinal
  sorry

end laptop_sale_price_l516_516971


namespace solve_for_n_l516_516484

theorem solve_for_n (n : ℝ) (h : 9^(n^2) * 9^(n^2) * 9^(n^2) = 81^3) : n = Real.sqrt 2 ∨ n = -Real.sqrt 2 :=
sorry

end solve_for_n_l516_516484


namespace mean_temperature_is_correct_l516_516118

-- Define the list of temperatures
def temperatures : List ℝ := [85, 84, 85, 83, 82, 84, 86, 88, 90, 89]

-- Define the mean function
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Theorem stating that the mean temperature is 85.6
theorem mean_temperature_is_correct : mean temperatures = 85.6 := by
  sorry

end mean_temperature_is_correct_l516_516118


namespace no_valid_b_arithmetic_sequence_l516_516667

open Real

theorem no_valid_b_arithmetic_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) :
  ¬ (∀ b (0 < b ∧ b < 360), 
  (2 * cos (3*b/2) * sin (b/2) = 2 * cos (5*b/2) * sin (b/2)) ∧ 
   (2 * cos (5*b/2) * sin (b/2) = 2 * cos (7*b/2) * sin (b/2))) :=
sorry

end no_valid_b_arithmetic_sequence_l516_516667


namespace inverse_relation_a1600_inverse_relation_a400_l516_516067

variable (a b : ℝ)

def k := 400 

theorem inverse_relation_a1600 : (a * b = k) → (a = 1600) → (b = 0.25) :=
by
  sorry

theorem inverse_relation_a400 : (a * b = k) → (a = 400) → (b = 1) :=
by
  sorry

end inverse_relation_a1600_inverse_relation_a400_l516_516067


namespace circle_range_k_l516_516103

theorem circle_range_k (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y + 10 - k = 0) → k > 2 :=
by
  sorry

end circle_range_k_l516_516103


namespace triangle_third_side_range_l516_516897

theorem triangle_third_side_range {x : ℤ} : 
  (7 < x ∧ x < 17) → (4 ≤ x ∧ x ≤ 16) :=
by
  sorry

end triangle_third_side_range_l516_516897


namespace total_amount_shared_l516_516239

theorem total_amount_shared (J Jo B : ℝ) (r1 r2 r3 : ℝ)
  (H1 : r1 = 2) (H2 : r2 = 4) (H3 : r3 = 6) (H4 : J = 1600) (part_value : ℝ)
  (H5 : part_value = J / r1) (H6 : Jo = r2 * part_value) (H7 : B = r3 * part_value) :
  J + Jo + B = 9600 :=
sorry

end total_amount_shared_l516_516239


namespace male_students_not_exceed_928_l516_516529

-- Definitions for the conditions
def total_students := 81650
def rows := 22
def cols := 75
def max_pairs := 11

-- For each row i, let's define the number of male students in row i
def male_students (i : ℕ) := ℕ
def female_students (i : ℕ) := cols - male_students i

-- The condition about pairs of students in the same row
def pairs_condition (i : ℕ) :=
  (male_students i * (male_students i - 1) / 2) + (female_students i * (female_students i - 1) / 2) ≤ max_pairs

-- The statement to prove
theorem male_students_not_exceed_928 :
  (∀ i < rows, pairs_condition i) → (∑ i in finset.range rows, male_students i) ≤ 928 := 
  by
    sorry

end male_students_not_exceed_928_l516_516529


namespace max_min_sum_of_squares_l516_516713

theorem max_min_sum_of_squares:
  ∃ (x : Fin 10 → ℕ), (∀ i, 0 < x i) ∧ (∑ i, x i = 2005) ∧
    (∑ i, (x i)^2 = 3984025 ∨ ∑ i, (x i)^2 = 402005) :=
begin
  sorry
end

end max_min_sum_of_squares_l516_516713


namespace ellipse_chord_through_focus_l516_516105

-- Define the variables and parameters of the ellipse
def a : ℝ := 6
def b : ℝ := 4
def c : ℝ := Real.sqrt (a^2 - b^2)
def F : ℝ × ℝ := (2 * Real.sqrt 5, 0)

theorem ellipse_chord_through_focus (x y : ℝ) (A B : ℝ × ℝ) :
  (x = 2 * Real.sqrt 5 + 4 / Real.sqrt 5) →
  (y = 4 * 4 / 36) →
  (AF = 2) →
  (A = (x, y)) →
  (B = (x, -y)) →
  (A.dist F = 2) →
  (B.dist F = 2) →
  (B.dist F = 32 / 15) :=
by
  sorry

end ellipse_chord_through_focus_l516_516105


namespace total_students_in_class_l516_516884

-- Define the initial conditions
def num_students_in_row (a b: Nat) : Nat := a + 1 + b
def num_lines : Nat := 3
noncomputable def students_in_row : Nat := num_students_in_row 2 5 

-- Theorem to prove the total number of students in the class
theorem total_students_in_class : students_in_row * num_lines = 24 :=
by
  sorry

end total_students_in_class_l516_516884


namespace resort_total_cost_l516_516986

noncomputable def first_cabin_cost (P : ℝ) := P
noncomputable def second_cabin_cost (P : ℝ) := (1/2) * P
noncomputable def third_cabin_cost (P : ℝ) := (1/6) * P
noncomputable def land_cost (P : ℝ) := 4 * P
noncomputable def pool_cost (P : ℝ) := P

theorem resort_total_cost (P : ℝ) (h : P = 22500) :
  first_cabin_cost P + pool_cost P + second_cabin_cost P + third_cabin_cost P + land_cost P = 150000 :=
by
  sorry

end resort_total_cost_l516_516986


namespace percentile_80_equals_10_8_l516_516521

def dataSet : List ℝ := [10.2, 9.7, 10.8, 9.1, 8.9, 8.6, 9.8, 9.6, 9.9, 11.2, 10.6, 11.7]

def percentile (k : ℕ) (data : List ℝ) : ℕ :=
  let sortedData := data.sorted
  let n := sortedData.length
  let pos := (k : ℝ) / 100 * (n : ℝ)
  if pos % 1 = 0 then ⌊pos⌋.toNat - 1 else ⌈pos⌉.toNat - 1

theorem percentile_80_equals_10_8 : percentile 80 dataSet = 10.8 :=
by
  sorry

end percentile_80_equals_10_8_l516_516521


namespace range_of_derivative_max_value_of_a_l516_516805

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.cos x - (x - Real.pi / 2) * Real.sin x

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ :=
  -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

-- Part (1): Prove the range of the derivative when a = -1 is [0, π/2]
theorem range_of_derivative (x : ℝ) (h0 : 0 ≤ x) (hπ : x ≤ Real.pi / 2) :
  (0 ≤ f' (-1) x) ∧ (f' (-1) x ≤ Real.pi / 2) := 
sorry

-- Part (2): Prove the maximum value of 'a' when f(x) ≤ 0 always holds
theorem max_value_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 0) :
  a ≤ -1 := 
sorry

end range_of_derivative_max_value_of_a_l516_516805


namespace domain_of_sqrt_sin_l516_516291

theorem domain_of_sqrt_sin (x : ℝ) (k : ℤ) : 
  (sin x) ≥ 0 ↔ ∃ k : ℤ, (2 * k * Real.pi) ≤ x ∧ x ≤ (2 * k * Real.pi + Real.pi) := 
sorry

end domain_of_sqrt_sin_l516_516291


namespace find_abc_value_l516_516735

theorem find_abc_value (a b c : ℕ) 
  (h_set : {a, b, c} = {1, 2, 3})
  (h1 : a ≠ 3 ∨ b = 3 ∨ c ≠ 1)
  (h2 : ¬(a ≠ 3 ∧ b = 3) ∧ ¬(b = 3 ∧ c ≠ 1) ∧ ¬(a ≠ 3 ∧ c ≠ 1)) :
  100 * a + 10 * b + c = 312 := 
sorry

end find_abc_value_l516_516735


namespace tina_more_than_katya_l516_516425

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end tina_more_than_katya_l516_516425


namespace married_men_items_indeterminate_l516_516890

theorem married_men_items_indeterminate 
  (total_men : ℕ)
  (married_men : ℕ)
  (men_with_tv : ℕ)
  (men_with_radio : ℕ)
  (men_with_car : ℕ)
  (men_with_laptop : ℕ)
  (pct_married_with_tv : ℕ)
  (pct_married_with_radio : ℕ)
  (pct_married_with_car : ℕ)
  (pct_single_with_laptop : ℕ) :
  total_men = 500 ∧
  married_men = 330 ∧
  men_with_tv = 275 ∧
  men_with_radio = 385 ∧
  men_with_car = 270 ∧
  men_with_laptop = 225 ∧
  pct_married_with_tv = 60 ∧
  pct_married_with_radio = 70 ∧
  pct_married_with_car = 40 ∧
  pct_single_with_laptop = 30 →
  ∃ n : ℕ, unknown n :=
by
  sorry

end married_men_items_indeterminate_l516_516890


namespace tina_more_than_katya_l516_516427

theorem tina_more_than_katya (katya_sales ricky_sales : ℕ) (tina_sales : ℕ) (
  h1 : katya_sales = 8) (h2 : ricky_sales = 9) (h3 : tina_sales = 2 * (katya_sales + ricky_sales)) :
  tina_sales - katya_sales = 26 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  rw [h3, h1]
  norm_num
  sorry

end tina_more_than_katya_l516_516427


namespace total_arrangement_with_at_least_one_girl_l516_516475

theorem total_arrangement_with_at_least_one_girl :
  let boys := 4 in
  let girls := 3 in
  let people := boys + girls in
  let combinations (n k : ℕ) := Nat.choose n k in
  let arrangements (n : ℕ) := Nat.fact n in
  combinations people 3 * arrangements 3 - combinations boys 3 * arrangements 3 = 186 :=
by
  let boys := 4
  let girls := 3
  let people := boys + girls
  let combinations (n k : ℕ) := Nat.choose n k
  let arrangements (n : ℕ) := Nat.fact n
  -- sorry is used to skip the proof part
  calc combinations people 3 * arrangements 3 - combinations boys 3 * arrangements 3 = 186 : sorry

end total_arrangement_with_at_least_one_girl_l516_516475


namespace opponent_score_value_l516_516898

-- Define the given conditions
def total_points : ℕ := 720
def games_played : ℕ := 24
def average_score := total_points / games_played
def championship_score := average_score / 2 - 2
def opponent_score := championship_score + 2

-- Lean theorem statement to prove
theorem opponent_score_value : opponent_score = 15 :=
by
  -- Proof to be filled in
  sorry

end opponent_score_value_l516_516898


namespace prime_iff_totient_divisor_sum_l516_516064

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def euler_totient (n : ℕ) : ℕ := sorry  -- we assume implementation of Euler's Totient function
def divisor_sum (n : ℕ) : ℕ := sorry  -- we assume implementation of Divisor sum function

theorem prime_iff_totient_divisor_sum (n : ℕ) :
  (2 ≤ n) → (euler_totient n ∣ (n - 1)) → (n + 1 ∣ divisor_sum n) → is_prime n :=
  sorry

end prime_iff_totient_divisor_sum_l516_516064


namespace calculate_expression_l516_516443

def smallest_positive_two_digit_multiple_of_7 : ℕ := 14
def smallest_positive_three_digit_multiple_of_5 : ℕ := 100

theorem calculate_expression : 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  (c * d) - 100 = 1300 :=
by 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  sorry

end calculate_expression_l516_516443


namespace min_value_3y3_6y_neg2_l516_516810

-- Let y be a positive real number, find the minimum value of 3y^3 + 6y^{-2}.
theorem min_value_3y3_6y_neg2 (y : ℝ) (hy : 0 < y) : ∃ m, (∀ z : ℝ, 0 < z → 3 * z^3 + 6 * z^(-2) ≥ m) ∧ m = 9 := 
sorry

end min_value_3y3_6y_neg2_l516_516810


namespace vector_difference_dot_product_is_zero_l516_516740

def vector_a : ℝ × ℝ := (1, 0)
def vector_b : ℝ × ℝ := (1/2, 1/2)

theorem vector_difference_dot_product_is_zero :
  (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) • vector_b = 0 := sorry

end vector_difference_dot_product_is_zero_l516_516740


namespace Ben_win_probability_l516_516870

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l516_516870


namespace parallelgram_projection_equality_l516_516028

open EuclideanGeometry

variables {A B C D M E F : Point}
variables [parallelogram ABCD]
variables (HM : M ∈ lineThrough A C)
variables (HE : IsOrthoProj E M (lineThrough A B))
variables (HF : IsOrthoProj F M (lineThrough A D))

theorem parallelgram_projection_equality :
  dist M E * dist C D = dist B C * dist M F :=
  sorry

end parallelgram_projection_equality_l516_516028


namespace specific_heat_capacity_l516_516051

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end specific_heat_capacity_l516_516051


namespace trapezoid_area_is_correct_l516_516030

noncomputable def area_trapezoid (A B C D X Y : ℝ × ℝ) (u v : ℝ) : ℝ :=
  let BY := u
  let DX := v
  let AB := Real.sqrt (16 + u^2)
  let CD := Real.sqrt (9 + v^2)
  let SlopeBC := u / 2
  let SlopeAD := v / 3
  
  if (AB = CD) ∧ (SlopeBC = SlopeAD) then
    let AreaABC := (1/2) * 6 * BY
    let AreaADC := (1/2) * 6 * DX
    AreaABC + AreaADC
  else
    0

theorem trapezoid_area_is_correct :
  ∃ (A B C D X Y : ℝ × ℝ) (u v : ℝ),
    (A = (3, 0)) ∧ (C = (-3, 0)) ∧ (X = (0, 0)) ∧ (Y = (-1, 0)) ∧ 
    ((u = 2 * Real.sqrt(35) / 5) ∧ (v = 3 * Real.sqrt(35) / 5)) ∧ 
    (area_trapezoid A B C D X Y u v = 3 * Real.sqrt 35) := sorry

end trapezoid_area_is_correct_l516_516030


namespace Janet_gym_hours_l516_516016

theorem Janet_gym_hours :
  ∃ f : ℕ, (5 = 2 * (1.5 : ℤ) + 2 * f) ∧ (1 = f) :=
by
  sorry

end Janet_gym_hours_l516_516016


namespace remainder_of_large_number_l516_516909

theorem remainder_of_large_number : 
  (9876543210 : ℤ) % 101 = 73 := 
by
  unfold_coes
  unfold_norm_num
  sorry

end remainder_of_large_number_l516_516909


namespace series_divergence_l516_516797

theorem series_divergence (a : ℕ → ℝ) (hdiv : ¬ ∃ l, ∑' n, a n = l) (hpos : ∀ n, a n > 0) (hnoninc : ∀ n m, n ≤ m → a m ≤ a n) : 
  ¬ ∃ l, ∑' n, (a n / (1 + n * a n)) = l :=
by
  sorry

end series_divergence_l516_516797


namespace cricket_run_rate_l516_516780

theorem cricket_run_rate (r : ℝ) (o₁ T o₂ : ℕ) (r₁ : ℝ) (Rₜ : ℝ) : 
  r = 4.8 ∧ o₁ = 10 ∧ T = 282 ∧ o₂ = 40 ∧ r₁ = (T - r * o₁) / o₂ → Rₜ = 5.85 := 
by 
  intros h
  sorry

end cricket_run_rate_l516_516780


namespace semicircle_radius_l516_516833

theorem semicircle_radius 
  (A B C K O : Type) 
  [Geometry A B C K O]
  (isosceles_triangle : Triangle A B C)
  (base_BC : Base B C)
  (semicircle : Semicircle A K B) 
  (touches_leg : Touches semicircle AC)
  (divides_leg : Divides semicircle AB 5 4) :
  let r := radius semicircle in
  r = 15 / Real.sqrt 11 :=
sorry

end semicircle_radius_l516_516833


namespace summer_holiday_weather_l516_516983

theorem summer_holiday_weather (sunny_mornings sunny_afternoons rainy_total : ℕ) (h1 : sunny_mornings = 7) (h2 : sunny_afternoons = 5) (h3 : rainy_total = 8) (h4 : ∀ days, if (days = rainy_total) then sunny_mornings = 7 else True) : (7 + 5 + 8) / 2 = 10 := 
by have h : (7 + 5 + 8 = 20) := rfl
   have hl : 20 / 2 = 10 := rfl
   exact hl -- here is to fill in using h's result.

end summer_holiday_weather_l516_516983


namespace ryan_more_hours_english_than_chinese_l516_516661

-- Definitions for the time Ryan spends on subjects
def weekday_hours_english : ℕ := 6 * 5
def weekend_hours_english : ℕ := 2 * 2
def total_hours_english : ℕ := weekday_hours_english + weekend_hours_english

def weekday_hours_chinese : ℕ := 3 * 5
def weekend_hours_chinese : ℕ := 1 * 2
def total_hours_chinese : ℕ := weekday_hours_chinese + weekend_hours_chinese

-- Theorem stating the difference in hours spent on English vs Chinese
theorem ryan_more_hours_english_than_chinese :
  (total_hours_english - total_hours_chinese) = 17 := by
  sorry

end ryan_more_hours_english_than_chinese_l516_516661


namespace smallest_positive_integer_modulo_l516_516311

theorem smallest_positive_integer_modulo :
  ∃ n : ℕ, n ≡ 1 [MOD 5] ∧ n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 9] ∧ n ≡ 4 [MOD 11] ∧
  (∀ m : ℕ, (m ≡ 1 [MOD 5] ∧ m ≡ 2 [MOD 7] ∧ m ≡ 3 [MOD 9] ∧ m ≡ 4 [MOD 11]) → m ≥ n) :=
by {
    let n := 1731,
    use n,
    split,
    norm_num,
    split,
    norm_num,
    split,
    norm_num,
    split,
    norm_num,
    intro m,
    intro h,
    cases' h with h5 h7,
    cases' h7 with h9 h11,
    cases' h11 with h9 h11,
    sorry
}

end smallest_positive_integer_modulo_l516_516311


namespace problem1_problem2_problem3_l516_516328

theorem problem1 :
  (∃ n : ℕ, 2^n = 1024 ∧ n = 10) := by
  -- n = 10 and 2^10 = 1024
  existsi 10
  split
  { -- Prove that 2^10 = 1024
    sorry
  }
  { -- Prove that n = 10
    refl
  }

theorem problem2 :
  (∀ n = 10, 
    -- constant term in the expansion of (sqrt(x) - 2/x^2)^n is 180
    let constant_term := 
      finset.sum (finset.range (n+1)) (λ r, if r = 2 then nat.choose 10 r * (-2)^2 else 0)
    in constant_term = 180) := by
  intros n hn
  subst hn
  -- simplify to find the constant term
  have : finset.sum (finset.range 11) (λ r, if r = 2 then nat.choose 10 r * (-2)^2 else 0) = nat.choose 10 2 * 4 := by
    sorry
  rw this
  norm_num

theorem problem3 :
  (∀ n = 10, 
    -- number of rational terms in the expansion of (sqrt(x) - 2/x^2)^n is 6
    let rational_terms := 
      finset.card (finset.filter (λ (r : ℕ), (10 - 5 * r) % 2 = 0) (finset.range (n+1)))
    in rational_terms = 6) := by
  intros n hn
  subst hn
  -- simplify and check for rational terms
  have : finset.card (finset.filter (λ (r : ℕ), (10 - 5 * r) % 2 = 0) (finset.range 11)) = 6 := by
    sorry
  rw this
  refl

end problem1_problem2_problem3_l516_516328


namespace num_distinct_orders_of_targets_l516_516393

theorem num_distinct_orders_of_targets : 
  let total_targets := 10
  let column_A_targets := 4
  let column_B_targets := 4
  let column_C_targets := 2
  (Nat.factorial total_targets) / 
  ((Nat.factorial column_A_targets) * (Nat.factorial column_B_targets) * (Nat.factorial column_C_targets)) = 5040 := 
by
  sorry

end num_distinct_orders_of_targets_l516_516393


namespace remainder_when_divided_by_22_l516_516226

theorem remainder_when_divided_by_22 
    (y : ℤ) 
    (h : y % 264 = 42) :
    y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l516_516226


namespace num_bags_in_range_l516_516982

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

noncomputable def stddev (l : List ℝ) : ℝ :=
  real.sqrt (variance l)

def weights : List ℝ := [495, 500, 503, 508, 498, 500, 493, 500, 503, 500]

def bags_in_range (l : List ℝ) (μ σ : ℝ) : ℕ :=
  (l.filter (λ x => μ - σ ≤ x ∧ x ≤ μ + σ)).length

theorem num_bags_in_range : bags_in_range weights (mean weights) (stddev weights) = 7 := by
  let μ := mean weights
  let σ := stddev weights
  have h_mean : μ = 500 := by sorry
  have h_stddev : σ = 4 := by sorry
  rw [h_mean, h_stddev]
  show bags_in_range weights 500 4 = 7
  exact sorry

end num_bags_in_range_l516_516982


namespace triangle_area_proof_l516_516756

-- The sides of the triangle
def a : ℝ := 39
def b : ℝ := 36
def c : ℝ := 15

-- Semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Area using Heron's formula
def area : ℝ := (s * (s - a) * (s - b) * (s - c)).sqrt

-- Statement to prove
theorem triangle_area_proof : area = 270 := by
  sorry

end triangle_area_proof_l516_516756


namespace no_such_function_exists_l516_516643

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l516_516643


namespace find_K_l516_516794

-- Define the side length of the cube
def side_len := 3

-- Calculate the surface area of the cube
def cube_surface_area := 6 * (side_len ^ 2)

-- Assume the surface area of the sphere is the same as the cube
def sphere_surface_area := cube_surface_area

-- Calculate the radius of the sphere from its surface area
def sphere_radius := real.sqrt(sphere_surface_area / (4 * real.pi))

-- Define the volume of the sphere
def sphere_volume := (4 / 3) * real.pi * (sphere_radius ^ 3)

-- Define the target volume form involving K
def target_volume (K : ℝ) := (K * real.sqrt(6)) / real.sqrt(real.pi)

-- The theorem to prove
theorem find_K : ∃ (K : ℝ), sphere_volume = target_volume K :=
  sorry

end find_K_l516_516794


namespace parallel_by_perpendicular_l516_516041

variables (m n : Type) [IsLine m] [IsLine n]
variables (α β γ : Type) [IsPlane α] [IsPlane β] [IsPlane γ]

-- Given: m and n are different lines, α, β, γ are different planes
-- Proposition: If m ⊥ α and n ⊥ α, then m ∥ n

theorem parallel_by_perpendicular (h1 : Perp m α) (h2 : Perp n α) : Parallel m n :=
sorry

end parallel_by_perpendicular_l516_516041


namespace range_of_f_l516_516727

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem range_of_f : ∀ (a b : ℝ), (∀ x : ℝ, f a b x = f a b (-x)) ∧ (a - 3 ≤ 2 * a) 
  → set_of (λ y, ∃ x : ℝ, a - 3 ≤ x ∧ x ≤ 2 * a ∧ y = f a b x) = set.Icc (3 : ℝ) (7 : ℝ) :=
by
  intros a b h
  sorry

end range_of_f_l516_516727


namespace Ben_win_probability_l516_516871

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l516_516871


namespace quadratic_equality_l516_516744

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end quadratic_equality_l516_516744


namespace sum_of_distinct_integers_l516_516442

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prod : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) : a + b + c + d + e = 39 :=
by
  sorry

end sum_of_distinct_integers_l516_516442


namespace g_x_equation_g_3_value_l516_516863

noncomputable def g : ℝ → ℝ := sorry

theorem g_x_equation (x : ℝ) (hx : x ≠ 1/2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x := sorry

theorem g_3_value : g 3 = 31 / 8 :=
by
  -- Use the provided functional equation and specific input values to derive g(3)
  sorry

end g_x_equation_g_3_value_l516_516863


namespace sum_power_inequality_l516_516798

theorem sum_power_inequality (n : ℕ) (h₀ : n ≥ 3)
  (x : Fin n → ℝ) (h₁ : ∀ i, 0 < x i) (h₂ : ∑ i, x i = 1) :
  ∑ i, x i ^ (1 - x ((i + 1) % n)) < 2 :=
by
  sorry

end sum_power_inequality_l516_516798


namespace ratio_HC_JE_l516_516467

theorem ratio_HC_JE 
  (A B C D E F G H J: Type*)
  (dist_AB dist_BC dist_CD dist_DE dist_EF : ℝ)
  (segment_AF : List (ℝ) = [dist_AB, dist_BC, dist_CD, dist_DE, dist_EF])
  (dist_CD = 1)
  (dist_AD = dist_AB + dist_BC + dist_CD)
  (dist_EF = 1)
  (dist_AF = dist_AB + dist_BC + dist_CD + dist_DE + dist_EF)
  (H_on_GD : Set.in_seg B D H)
  (J_on_GF : Set.in_seg A F J)
  (AG_parallel_HC : is_parallel A G C H)
  (AG_parallel_JE : is_parallel A G E J)
  (AD_pos : 0 < dist_AD)
  (AF_pos : 0 < dist_AF) : 
  \(\frac{HC}{JE} = \frac{5}{3}\) :=
sorry

end ratio_HC_JE_l516_516467


namespace geom_seq_sum_l516_516675

noncomputable def geom_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
a₁ * r^(n-1)

theorem geom_seq_sum (a₁ r : ℝ) (h_pos : 0 < a₁) (h_pos_r : 0 < r)
  (h : a₁ * (geom_seq a₁ r 5) + 2 * (geom_seq a₁ r 3) * (geom_seq a₁ r 6) + a₁ * (geom_seq a₁ r 11) = 16) :
  (geom_seq a₁ r 3 + geom_seq a₁ r 6) = 4 :=
sorry

end geom_seq_sum_l516_516675


namespace logarithmic_growth_rate_slows_l516_516610

theorem logarithmic_growth_rate_slows (x : ℝ) (h : x > 0) :
  ∀ f : ℝ → ℝ, (f = logBase 2) -> (∀ x1 x2 : ℝ, x1 > x2 -> x2 > 0 -> (f' x2) > (f' x1)) :=
by
  sorry

end logarithmic_growth_rate_slows_l516_516610


namespace complex_division_result_l516_516710

theorem complex_division_result : (1 - Complex.i) / (3 + Complex.i) = (1 / 5 : ℂ) - (2 / 5 : ℂ) * Complex.i :=
by sorry

end complex_division_result_l516_516710


namespace remainder_when_sum_divided_by_30_l516_516931

theorem remainder_when_sum_divided_by_30 (x y z : ℕ) (hx : x % 30 = 14) (hy : y % 30 = 5) (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 :=
by
  sorry

end remainder_when_sum_divided_by_30_l516_516931


namespace inequality_proof_l516_516412

variable {f : ℝ → ℝ}

theorem inequality_proof
  (h : ∀ x ∈ Ioo 0 π, (deriv f x) * sin x > f x * cos x) :
  f (π / 4) > sqrt 2 * f (π / 6) :=
by
  sorry

end inequality_proof_l516_516412


namespace sum_of_non_palindrome_integers_between_100_200_become_palindrome_in_seven_steps_l516_516319

-- Definitions
def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString in 
  s = s.reverse

def reverse_and_add (n : ℕ) : ℕ := 
  n + (nat.ofDigits 10 (n.digits 10).reverse)

def takes_steps_to_palindrome (n : ℕ) (steps : ℕ) : Prop := 
  ∃ k, k = steps ∧ (∀ i < k, ¬ is_palindrome (nat.iterate reverse_and_add i n))
  ∧ is_palindrome (nat.iterate reverse_and_add k n)

-- Main theorem statement
theorem sum_of_non_palindrome_integers_between_100_200_become_palindrome_in_seven_steps :
  (∑ n in (set.Icc 100 200).filter (λ n, ¬ is_palindrome n ∧ takes_steps_to_palindrome n 7), n) = 374 := 
sorry

end sum_of_non_palindrome_integers_between_100_200_become_palindrome_in_seven_steps_l516_516319


namespace gcd_pow_of_subtraction_l516_516906

noncomputable def m : ℕ := 2^2100 - 1
noncomputable def n : ℕ := 2^1950 - 1

theorem gcd_pow_of_subtraction : Nat.gcd m n = 2^150 - 1 :=
by
  -- To be proven
  sorry

end gcd_pow_of_subtraction_l516_516906


namespace find_x_l516_516697

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 4 * x + 3 ≥ 0
def q (x : ℤ) : Prop := x ∈ Set.Univ

-- We need to prove x = -2 given the conditions
theorem find_x : ∀ (x : ℤ), p x ∧ q x ∧ ¬(p x ∧ q x) ∧ ¬¬q x → x = -2 := by
  intro x
  sorry

end find_x_l516_516697


namespace factorial_of_6_is_720_l516_516839

theorem factorial_of_6_is_720 : (Nat.factorial 6) = 720 := by
  sorry

end factorial_of_6_is_720_l516_516839


namespace sum_f_powers_of_two_l516_516685

noncomputable def f (x : ℝ) : ℝ := 4 * log (2 : ℝ) (3 : ℝ) * log (3 : ℝ) x + 233

-- Prove that ∑ from k = 1 to 8 of f(2^k) equals 2008
theorem sum_f_powers_of_two :
  (∑ k in range 1 to 9, f (2 ^ k)) = 2008 :=
sorry

end sum_f_powers_of_two_l516_516685


namespace dice_same_number_probability_l516_516199

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516199


namespace rationalize_denominator_eq_sum_l516_516473

open Real

theorem rationalize_denominator_eq_sum :
  ∃ (A B C : ℤ), (C > 0) ∧ (∀ (p : ℕ), prime p → ¬ (p^3 ∣ B)) ∧
  (3 / (2 * real.cbrt 5) = (A * real.cbrt B) / C) ∧ (A + B + C = 38) :=
sorry

end rationalize_denominator_eq_sum_l516_516473


namespace dice_same_number_probability_l516_516195

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516195


namespace distance_between_trees_l516_516388

theorem distance_between_trees (number_of_trees : ℕ) (total_length : ℝ) (number_of_gaps : ℕ) (distance_between_trees : ℝ):
  number_of_trees = 26 → 
  total_length = 600 → 
  number_of_gaps = number_of_trees - 1 → 
  distance_between_trees = total_length / number_of_gaps → 
  distance_between_trees = 24 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h4],
  norm_num
}

end distance_between_trees_l516_516388


namespace complex_division_example_l516_516547

theorem complex_division_example : (Complex.ofReal (-2) - Complex.i) / Complex.i = -1 + 2 * Complex.i := 
by
  sorry

end complex_division_example_l516_516547


namespace calculate_closing_oranges_l516_516274

open_locale classical

def initial_lemons := 50
def initial_oranges := 60
def closing_lemons := 20
def ratio_decrease_percent := 0.40

theorem calculate_closing_oranges : ∃ (x : ℕ), x = 40 ∧ 
  (closing_lemons : ℚ) / (x : ℚ) = (initial_lemons : ℚ) / (initial_oranges : ℚ) * (1 - ratio_decrease_percent) :=
sorry

end calculate_closing_oranges_l516_516274


namespace distance_to_origin_l516_516496

def A : ℝ × ℝ := (-1, -2)

theorem distance_to_origin : real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = real.sqrt 5 :=
by
  sorry

end distance_to_origin_l516_516496


namespace cube_volume_l516_516562

theorem cube_volume (lateral_surface_area : ℝ) (h : lateral_surface_area = 100) : 
  ∃ (v : ℝ), v = 125 := 
by 
  let side_length := Math.sqrt (lateral_surface_area / 4)
  have h_side : side_length = 5 :=
    by 
      rw [h]
      sorry
  let volume := side_length ^ 3
  have h_volume : volume = 125 :=
    by 
      rw [h_side]
      sorry
  use volume
  exact h_volume

end cube_volume_l516_516562


namespace proposition_false_at_4_l516_516252

theorem proposition_false_at_4 (P : ℕ → Prop) (hp : ∀ k : ℕ, k > 0 → (P k → P (k + 1))) (h4 : ¬ P 5) : ¬ P 4 :=
by {
    sorry
}

end proposition_false_at_4_l516_516252


namespace evaluate_expression_l516_516659

theorem evaluate_expression :
  2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 :=
by
  sorry

end evaluate_expression_l516_516659


namespace circumcenter_equality_l516_516800

open EuclideanGeometry

variables {A B C D O O1 O2 : Point}
variables {circumcircleABC circumcircleABD circumcircleACD : Circle}

-- Conditions:
def is_triangle (A B C : Point) : Prop := 
A ≠ B ∧ B ≠ C ∧ C ≠ A

def is_circumcenter (O : Point) (△ : Triangle) : Prop :=
O = triangle.circumcenter △

def is_angle_bisector_foot (D : Point) (A B C : Point) : Prop :=
∃ (E : Point), is_bisector E ∧ A = E ∧ C = D

-- Triangle centers
def is_circumcenter_of_triangleABD (O1 : Point) (△ABD : Triangle) : Prop :=
O1 = triangle.circumcenter △ABD

def is_circumcenter_of_triangleACD (O2 : Point) (△ACD : Triangle) : Prop :=
O2 = triangle.circumcenter △ACD

theorem circumcenter_equality
  (h1 : is_triangle A B C)
  (h2 : is_circumcenter O (Triangle.mk A B C))
  (h3 : is_angle_bisector_foot D A B C)
  (h4 : is_circumcenter O1 (Triangle.mk A B D))
  (h5 : is_circumcenter O2 (Triangle.mk A C D)) :
  dist O O1 = dist O O2 :=
sorry

end circumcenter_equality_l516_516800


namespace statement_C_correct_statements_l516_516551

section 

variable {x k a : ℝ}

-- Statement A
def statement_A : Prop := Monotone (λ x, log (6 + x - 2 * x ^ 2)) 

-- Statement B
def statement_B : Prop := ∀ x : ℝ, x ^ 2 + 2 * x + a > 0 ↔ a > 1

-- Statement C
def f (x : ℝ) (k : ℝ) : ℝ := (k - real.exp (x * real.log 3)) / (1 + real.exp (x * real.log 3)) 

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem statement_C : Prop := odd_function (λ x, f x 1)

-- Statement D
def g (x : ℝ) : ℝ := real.log (x + real.sqrt (x ^ 2 + 1))

def statement_D : Prop := odd_function g ∧ Monotone g

theorem correct_statements : Prop := statement_B ∧ statement_C ∧ statement_D

end

end statement_C_correct_statements_l516_516551


namespace dice_probability_same_face_l516_516159

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l516_516159


namespace cube_root_power_l516_516655

theorem cube_root_power (a : ℝ) (h : a = 8) : (a^(1/3))^12 = 4096 := by
  rw [h]
  have h2 : 8 = 2^3 := rfl
  rw h2
  sorry

end cube_root_power_l516_516655


namespace find_a_l516_516037

noncomputable def satisfies_conditions (a : ℝ) : Prop :=
  let z := (a + complex.i) ^ 2 * complex.i
  in complex.re z > 0 ∧ complex.im z = 0

theorem find_a : satisfies_conditions (-1) :=
by sorry

end find_a_l516_516037


namespace negative_integers_abs_le_4_l516_516990

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l516_516990


namespace collinearity_of_L_M_N_l516_516819

-- Define the right triangle and the points L, M, N.
variables {A B C L M N : Type} [Geometry]

-- A triangle is right-angled at A.
axiom right_triangle (A B C : Type) (h : ∠ BAC = 90°) : Triangle A B C

-- L is a point on segment BC
axiom point_on_segment (L : Type) (BC : Segment B C)

-- The circumcircle of triangle ABL intersects AC at M
axiom circumcircle_ABL_intersect_AC (A B C L M : Type) (circle_ABL : Circumcircle A B L) : Point M ∈ (circle_ABL ∩ [AC])

-- The circumcircle of triangle ACL intersects AB at N
axiom circumcircle_ACL_intersect_AB (A C L N : Type) (circle_ACL : Circumcircle A C L) : Point N ∈ (circle_ACL ∩ [AB])

-- The main theorem: the collinearity of points L, M, and N
theorem collinearity_of_L_M_N (A B C L M N : Type) 
    [right_triangle A B C]
    [point_on_segment L BC]
    [circumcircle_ABL_intersect_AC A B C L M]
    [circumcircle_ACL_intersect_AB A C L N] : Collinear L M N := 
begin
  sorry,
end

end collinearity_of_L_M_N_l516_516819


namespace same_number_probability_four_dice_l516_516175

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l516_516175


namespace find_a_l516_516731

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := (Real.sqrt (a^2 + 3)) / a

theorem find_a (a : ℝ) (h : a > 0) (hexp : hyperbola_eccentricity a = 2) : a = 1 :=
by
  sorry

end find_a_l516_516731


namespace pyramid_dihedral_angle_l516_516769

theorem pyramid_dihedral_angle 
  (k : ℝ) 
  (h_k_pos : 0 < k) :
  ∃ α : ℝ, α = 2 * Real.arccos (1 / Real.sqrt (Real.sqrt (4 * k))) :=
sorry

end pyramid_dihedral_angle_l516_516769


namespace cubic_root_equation_solution_l516_516485

theorem cubic_root_equation_solution (x : ℝ) : 
      (∛(30 * x + ∛(30 * x + 25)) = 15) ↔ (x = 335 / 3) := 
by 
  sorry

end cubic_root_equation_solution_l516_516485


namespace find_first_set_length_l516_516020

def length_of_second_set : ℤ := 20
def ratio := 5

theorem find_first_set_length (x : ℤ) (h1 : length_of_second_set = ratio * x) : x = 4 := 
sorry

end find_first_set_length_l516_516020


namespace regular_price_adult_ticket_l516_516314

theorem regular_price_adult_ticket : 
  ∀ (concessions_cost_children cost_adult1 cost_adult2 cost_adult3 cost_adult4 cost_adult5
       ticket_cost_child cost_discount1 cost_discount2 cost_discount3 total_cost : ℝ),
  (concessions_cost_children = 3) → 
  (cost_adult1 = 5) → 
  (cost_adult2 = 6) → 
  (cost_adult3 = 7) → 
  (cost_adult4 = 4) → 
  (cost_adult5 = 9) → 
  (ticket_cost_child = 7) → 
  (cost_discount1 = 3) → 
  (cost_discount2 = 2) → 
  (cost_discount3 = 1) → 
  (total_cost = 139) → 
  (∀ A : ℝ, total_cost = 
    (2 * concessions_cost_children + cost_adult1 + cost_adult2 + cost_adult3 + cost_adult4 + cost_adult5) + 
    (2 * ticket_cost_child + (2 * A + (A - cost_discount1) + (A - cost_discount2) + (A - cost_discount3))) → 
    5 * A - 6 = 88 →
    A = 18.80) :=
by
  intros
  sorry

end regular_price_adult_ticket_l516_516314


namespace function_characterization_l516_516637
open Int

def is_totally_multiplicative (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (a * b) = f a * f b

def is_non_negative (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n ≥ 0

def satisfies_division_algorithm_property (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, b ≠ 0 → ∃ q r : ℤ, a = q * b + r ∧ f r < f b

def initial_conditions (f : ℤ → ℤ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ f (-1) = 1 ∧ ∀ x : ℤ, f x = f (-x)

def behavior_for_positive_integers (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n > 0 → f n ≥ Int.log2 (n + 1)

theorem function_characterization :
  ∀ (f : ℤ → ℤ), is_totally_multiplicative f →
    is_non_negative f →
    satisfies_division_algorithm_property f →
    initial_conditions f →
    behavior_for_positive_integers f →
    ∃ c : ℤ, c ≠ 0 ∧ (∀ n : ℤ, f n = (n.natAbs : ℤ) ^ c) :=
by
  intro f h1 h2 h3 h4 h5
  sorry

end function_characterization_l516_516637


namespace tree_ratio_l516_516245

theorem tree_ratio (native_trees : ℕ) (total_planted : ℕ) (M : ℕ) 
  (h1 : native_trees = 30) 
  (h2 : total_planted = 80) 
  (h3 : total_planted = M + M / 3) :
  (native_trees + M) / native_trees = 3 :=
sorry

end tree_ratio_l516_516245


namespace collinear_MNQ_l516_516325

theorem collinear_MNQ
  (A B C P : Point)
  (hABC : equilateral_triangle A B C)
  (hP : on_circumcircle P A B C)
  (M N Q : Point)
  (hM : parallel (line P M) (line B C) ∧ on_line M (line C A))
  (hN : parallel (line P N) (line C A) ∧ on_line N (line A B))
  (hQ : parallel (line P Q) (line A B) ∧ on_line Q (line B C))
  : collinear M N Q :=
by
  sorry

end collinear_MNQ_l516_516325


namespace number_of_stations_between_hyderabad_and_bangalore_l516_516532

theorem number_of_stations_between_hyderabad_and_bangalore :
  ∃ n : ℕ, (n + 2) * (n + 1) / 2 = 132 ∧ n = 10 :=
by {
  use 10,
  split,
  calc
    (10 + 2) * (10 + 1) / 2 = 12 * 11 / 2 : by norm_num
                              ... = 132 : by norm_num,
  exact rfl
}

end number_of_stations_between_hyderabad_and_bangalore_l516_516532


namespace find_k_l516_516398

theorem find_k
  (AB AC : ℝ)
  (k : ℝ)
  (h1 : AB = AC)
  (h2 : AB = 8)
  (h3 : AC = 5 - k) : k = -3 :=
by
  sorry

end find_k_l516_516398


namespace quadratic_to_square_l516_516486

theorem quadratic_to_square (x : ℝ) :
  4 * x^2 - 8 * x - 128 = 0 → ∃ r s : ℝ, (x + r)^2 = s ∧ r = -1 ∧ s = 33 :=
by
  intro h
  use [-1, 33]
  sorry

end quadratic_to_square_l516_516486


namespace freshmen_count_l516_516848

theorem freshmen_count (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ↔ n = 574 := 
by sorry

end freshmen_count_l516_516848


namespace expression_value_l516_516629

theorem expression_value :
  (2 + 7/9)^(1/2) + (Real.log 5 / Real.log 10)^0 + (27/64)^(-1/3) = 4 :=
by
  sorry

end expression_value_l516_516629


namespace distinct_triangle_areas_l516_516799

variables (A B C D E F G : ℝ) (h : ℝ)
variables (AB BC CD EF FG AC BD AD EG : ℝ)

def is_valid_points := AB = 2 ∧ BC = 1 ∧ CD = 3 ∧ EF = 1 ∧ FG = 2 ∧ AC = AB + BC ∧ BD = BC + CD ∧ AD = AB + BC + CD ∧ EG = EF + FG

theorem distinct_triangle_areas (h_pos : 0 < h) (valid : is_valid_points AB BC CD EF FG AC BD AD EG) : 
  ∃ n : ℕ, n = 5 := 
by
  sorry

end distinct_triangle_areas_l516_516799


namespace ratio_area_IJKL_WXYZ_l516_516086

-- Definitions based on given conditions
def side_length_WXYZ (s : ℝ) := 12 * s
def WI (s : ℝ) := 9 * s
def IZ (s : ℝ) := 3 * s
def side_length_IJKL (s : ℝ) := 3 * Real.sqrt 2 * s

-- Areas of squares
def area_WXYZ (s : ℝ) := (12 * s) ^ 2
def area_IJKL (s : ℝ) := (3 * Real.sqrt 2 * s) ^ 2

-- Theorem that states the ratio of areas
theorem ratio_area_IJKL_WXYZ (s : ℝ) (h : WI s = 3 * IZ s) :
  area_IJKL s / area_WXYZ s = 1 / 8 := by
  sorry

end ratio_area_IJKL_WXYZ_l516_516086


namespace prob_no_rain_four_days_l516_516509

noncomputable def prob_rain_one_day : ℚ := 2 / 3

noncomputable def prob_no_rain_one_day : ℚ := 1 - prob_rain_one_day

def independent_events (events : List (Unit → Prop)) : Prop :=
  -- A statement about independence of events
  sorry

theorem prob_no_rain_four_days :
  let days := 4
  let prob_no_rain := prob_no_rain_one_day
  independent_events (List.replicate days (fun _ => prob_no_rain)) →
  (prob_no_rain^days) = (1/81) := 
by
  sorry

end prob_no_rain_four_days_l516_516509


namespace trigonometric_identity_l516_516234

theorem trigonometric_identity :
  (cos (10 * (π / 180)) / (2 * sin (10 * (π / 180))) - 2 * cos (10 * (π / 180)) = (sqrt 3) / 2) := by
  sorry

end trigonometric_identity_l516_516234


namespace area_relation_l516_516831

open EuclideanGeometry

noncomputable def S (Δ : Triangle) : ℝ := sorry

variable {A B C D E F : Point}
variable (ABC : Triangle A B C)

def point_on_line_segment (P A B : Point) : Prop := P ∈ segment A B

def parallel {a b : Line} : Prop := parallel a b

-- Definitions for specific lines
def DE_line : Line := Line.mk D E
def BC_line : Line := Line.mk B C
def EF_line : Line := Line.mk E F
def AB_line : Line := Line.mk A B

theorem area_relation
  (hE_on_AC : point_on_line_segment E A C)
  (hDE_parallel_BC : parallel DE_line BC_line)
  (hEF_parallel_AB : parallel EF_line AB_line)
  (hD_on_AC : point_on_line_segment D A C)
  (hF_on_AB : point_on_line_segment F A B) :
  S (Triangle.mk B D E F) = 2 * real.sqrt (S (Triangle.mk A D E) * S (Triangle.mk E F C)) :=
sorry

end area_relation_l516_516831


namespace uncovered_area_within_larger_circle_l516_516399

theorem uncovered_area_within_larger_circle (r_large : ℝ) (r_small : ℝ) (area_no_cover : ℝ) 
  (h_large : r_large = 10) 
  (h_small : r_small = r_large / 2) : 
  area_no_cover = π * r_large^2 - 3 * π * r_small^2 :=
by {
  -- Given conditions
  rw [h_large, h_small], 
  -- Compute areas and substitute values accordingly
  sorry
}

end uncovered_area_within_larger_circle_l516_516399


namespace largest_safe_n_l516_516270

theorem largest_safe_n (capacities : Finset ℕ) (occupancies : ℕ → ℕ) :
  capacities = Finset.range 101 200 ∧ (occupancies.sum = 8824) →
  ∃ (A B : ℕ), A ≠ B ∧ (A ∈ capacities) ∧ (B ∈ capacities) ∧ (occupancies A + occupancies B ≤ B) := sorry

end largest_safe_n_l516_516270


namespace jordan_novels_read_l516_516424

variable (J A : ℕ)

theorem jordan_novels_read (h1 : A = (1 / 10) * J)
                          (h2 : J = A + 108) :
                          J = 120 := 
by
  sorry

end jordan_novels_read_l516_516424


namespace probability_of_same_number_on_four_dice_l516_516166

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l516_516166


namespace average_of_first_150_terms_l516_516632

def sequence (n : ℕ) : ℤ := (-1)^n * (n + 1)

theorem average_of_first_150_terms :
  (∑ i in Finset.range 150, sequence i) / 150 = -0.5 :=
sorry

end average_of_first_150_terms_l516_516632


namespace spent_on_computer_accessories_l516_516079

theorem spent_on_computer_accessories :
  ∀ (x : ℕ), (original : ℕ) (snacks : ℕ) (remaining : ℕ),
  original = 48 →
  snacks = 8 →
  remaining = 4 + original / 2 →
  original - (x + snacks) = remaining →
  x = 12 :=
by
  intros x original snacks remaining
  intro h_original
  intro h_snacks
  intro h_remaining
  intro h_spent
  sorry

end spent_on_computer_accessories_l516_516079


namespace arithmetic_seq_a5_value_l516_516005

theorem arithmetic_seq_a5_value (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := 
sorry

end arithmetic_seq_a5_value_l516_516005


namespace find_k_l516_516040

noncomputable def f (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : f a b c 1 = 0) 
  (h2 : 50 < f a b c 7) (h2' : f a b c 7 < 60) 
  (h3 : 70 < f a b c 8) (h3' : f a b c 8 < 80) 
  (h4 : 5000 * k < f a b c 100) (h4' : f a b c 100 < 5000 * (k + 1)) : 
  k = 3 := 
sorry

end find_k_l516_516040


namespace prove_inequality_iff_l516_516322

theorem prove_inequality_iff (x : ℝ) : 
  (real.cbrt (5 * x + 2) - real.cbrt (x + 3) ≤ 1) ↔ (x ≤ 5) :=
sorry

end prove_inequality_iff_l516_516322


namespace number_of_correct_statements_zero_l516_516612

theorem number_of_correct_statements_zero :
  (∀ (f : ℝ → ℝ) (x : ℝ), f' x = 0 → ¬(x = is_extremum_point f x)) ∧
  (∀ (f : ℝ → ℝ) (a b : ℝ), largest_local_maximum f a b ≠ maximum_value f [a, b]) ∧
  (∀ (f : ℝ → ℝ) (x₁ x₂ : ℝ), local_maximum f x₁ ∧ local_minimum f x₂ → f x₁ ≤ f x₂) ∧
  (∀ (f : ℝ → ℝ), ¬(some_functions_two_minimum f)) ∧
  (∀ (f : ℝ → ℝ) (x : ℝ), is_extremum_point f x → ¬(∃ f' x = 0)) :=
sorry

end number_of_correct_statements_zero_l516_516612


namespace cube_root_power_l516_516657

theorem cube_root_power (a : ℝ) (h : a = 8) : (a^(1/3))^12 = 4096 := by
  rw [h]
  have h2 : 8 = 2^3 := rfl
  rw h2
  sorry

end cube_root_power_l516_516657


namespace sum_abs_sequence_eq_153_l516_516093

def sequence (n : ℕ) : ℤ := 2 * n - 7

def abs_sequence_sum (m : ℕ) : ℤ :=
  (Finset.range (m + 1)).sum (λ n, |sequence n|)

theorem sum_abs_sequence_eq_153 : abs_sequence_sum 15 = 153 :=
by sorry

end sum_abs_sequence_eq_153_l516_516093


namespace contradiction_assumption_l516_516215

theorem contradiction_assumption 
  (a b c : ℕ) :
  ¬ (∃ n : ℕ, (n = a ∨ n = b ∨ n = c) ∧ n % 2 = 0 → 
    (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) ↔ 
  ∃ n₁ n₂ : ℕ, n₁ ∈ {a, b, c} ∧ n₂ ∈ {a, b, c} ∧ 
    n₁ ≠ n₂ ∧ 
    n₁ % 2 = 0 ∧ n₂ % 2 = 0 := 
sorry

end contradiction_assumption_l516_516215


namespace find_x2_x1_add_x3_l516_516445

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := (10*x^3 - 210*x^2 + 3)

-- Statement including conditions and the question we need to prove
theorem find_x2_x1_add_x3 :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    polynomial x₁ = 0 ∧ 
    polynomial x₂ = 0 ∧ 
    polynomial x₃ = 0 ∧ 
    x₂ * (x₁ + x₃) = 21 :=
by sorry

end find_x2_x1_add_x3_l516_516445


namespace count_false_propositions_is_three_l516_516611

-- Define each proposition as a boolean statement
def proposition1 : Prop := ¬ (∀ (A B : ℝ), (A = B) → (∀ v w : ℝ, (v = w) → (A = v ∧ B = w)))
def proposition2 : Prop := ¬ (∀ l m t : ℝ, (l = m) → (t = l))
def proposition3 : Prop := ∀ (angle1 angle2 : ℝ), (angle1 = angle2) → (90 - angle1 = 90 - angle2)
def proposition4 : Prop := ¬ (∀ (x y : ℝ), (x^2 = y^2) → (x = y))

-- Define the problem of counting the number of false propositions
def num_false_propositions (props : List Prop) : Nat :=
  props.count (λ p, ¬p)

-- The list of propositions
def propositions : List Prop := [proposition1, proposition2, proposition3, proposition4]

-- The theorem to be proven
theorem count_false_propositions_is_three : num_false_propositions propositions = 3 := 
by
  sorry

end count_false_propositions_is_three_l516_516611


namespace next_two_equations_l516_516830

-- Definitions based on the conditions in the problem
def pattern1 (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Statement to prove the continuation of the pattern
theorem next_two_equations 
: pattern1 9 40 41 ∧ pattern1 11 60 61 :=
by
  sorry

end next_two_equations_l516_516830


namespace percent_of_a_is_4b_l516_516844

variable (a b : ℝ)

theorem percent_of_a_is_4b (hab : a = 1.8 * b) :
  (4 * b / a) * 100 = 222.22 := by
  sorry

end percent_of_a_is_4b_l516_516844


namespace omitted_angle_of_convex_polygon_l516_516765

theorem omitted_angle_of_convex_polygon (calculated_sum : ℕ) (omitted_angle : ℕ)
    (h₁ : calculated_sum = 2583) (h₂ : omitted_angle = 2700 - 2583) :
    omitted_angle = 117 :=
by
  sorry

end omitted_angle_of_convex_polygon_l516_516765


namespace minimal_polynomial_l516_516044

namespace ProofProblem

-- Define the polynomial f(x)
def f (x: ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

-- Define the polynomial g(x)
def g (x: ℝ) : ℝ := 3*x^2 - 5*x - 3

-- Define the conditions
axiom condition1 : f(2) = g(2)
axiom condition2 : f(2 - real.sqrt 2) = g(2 - real.sqrt 2)
axiom condition3 : f(2 + real.sqrt 2) = g(2 + real.sqrt 2)

-- State the theorem
theorem minimal_polynomial :
  ∀ x: ℝ, f(x) = g(x) :=
by
  -- Proof to be provided
  sorry

end ProofProblem

end minimal_polynomial_l516_516044


namespace kenny_pieces_used_l516_516327

-- Definitions based on conditions
def mushrooms_cut := 22
def pieces_per_mushroom := 4
def karla_pieces := 42
def remaining_pieces := 8
def total_pieces := mushrooms_cut * pieces_per_mushroom

-- Theorem to be proved
theorem kenny_pieces_used :
  total_pieces - (karla_pieces + remaining_pieces) = 38 := 
by 
  sorry

end kenny_pieces_used_l516_516327


namespace train_speed_correct_l516_516985

-- Define the necessary conditions
def train_crosses_pole (time_sec : ℕ) : Prop :=
  time_sec = 9

def train_length_meters (length_m : ℕ) : Prop :=
  length_m = 145

-- Convert length from meters to kilometers
def length_in_km (length_m : ℕ) : ℝ :=
  length_m / 1000.0

-- Convert time from seconds to hours
def time_in_hours (time_sec : ℕ) : ℝ :=
  time_sec / 3600.0

-- Define the speed calculation
def train_speed (length_m : ℕ) (time_sec : ℕ) : ℝ :=
  length_in_km length_m / time_in_hours time_sec 

-- The theorem to be proved
theorem train_speed_correct :
  ∀ (length_m time_sec : ℕ),
  train_crosses_pole time_sec →
  train_length_meters length_m →
  train_speed length_m time_sec = 58 :=
by 
  intros length_m time_sec h_time h_length
  simp [train_crosses_pole, train_length_meters] at h_time h_length
  rw [h_time, h_length]
  sorry

end train_speed_correct_l516_516985


namespace length_of_DE_l516_516052

theorem length_of_DE 
  (DP EQ DE: ℝ) 
  (DP_val: DP = 18) 
  (EQ_val: EQ = 24) 
  (medians_perpendicular: medians_perpendicular (\△ DEF) DP EQ)
  : DE = 20 := 
by 
  -- Proof steps would go here
  sorry

end length_of_DE_l516_516052


namespace final_result_for_seven_at_twentyone_l516_516746

-- Define the operation
def at (a b : ℕ) : ℝ := (a * b : ℝ) / (a + b + 2)

-- Conditions
lemma seven_at_twentyone_equals :
  (7 @ 21 : ℝ) = (147 / 30 : ℝ) := by
  sorry

lemma operation_addition :
  (147 / 30 + 3 : ℝ) = (79 / 10 : ℝ) := by
  sorry

-- Main theorem
theorem final_result_for_seven_at_twentyone :
  (7 @ 21 + 3 : ℝ) = (79 / 10 : ℝ) := by
  apply operation_addition

end final_result_for_seven_at_twentyone_l516_516746


namespace canoe_vs_kayak_l516_516542

theorem canoe_vs_kayak (
  C K : ℕ 
) (h1 : 14 * C + 15 * K = 288) 
  (h2 : C = (3 * K) / 2) : 
  C - K = 4 := 
sorry

end canoe_vs_kayak_l516_516542


namespace circle_tangent_problem_l516_516346

variable {P Q : Type} [EuclideanGroup P] [AffinePlane P]

variables ( ω : Circle P Q)
variables ( A B C : P) 
variables ( l : Line P) 
variables ( AC : Line P) 
variables ( D : P)

-- Conditions
variable (tangent_at_A : Tangent ω A l)
variable (farther_B : FartherFromLine B C l)
variable (AC_intersect : IntersectAt AC (parallelThrough l B) D)

-- Question (Prove this statement)
theorem circle_tangent_problem
  (hA : OnCircle ω A) 
  (hB : OnCircle ω B) 
  (hC : OnCircle ω C) 
  (hD : IntersectAt AC (parallelThrough l B) D)
  : distance A B ^ 2 = distance A C * distance A D := 
sorry

end circle_tangent_problem_l516_516346


namespace sum_g_eq_one_ninth_l516_516676

def g (n : ℕ) : ℝ :=
  ∑' (k : ℕ) in (Set.Ici 3), (1 : ℝ) / (k : ℝ)^n

-- State the theorem
theorem sum_g_eq_one_ninth : ∑' (n : ℕ) in (Set.Ici 3), g n = 1/9 := 
  sorry

end sum_g_eq_one_ninth_l516_516676


namespace probability_at_least_one_red_ball_l516_516267

theorem probability_at_least_one_red_ball :
  let red_balls := 2
  let white_balls := 3
  let total_balls := red_balls + white_balls
  let p_white := white_balls / total_balls
  let p_second_white := (white_balls - 1) / (total_balls - 1)
  let p_two_white := p_white * p_second_white
  in 1 - p_two_white = 7 / 10 := by
  sorry

end probability_at_least_one_red_ball_l516_516267


namespace greatest_power_two_factor_l516_516907

theorem greatest_power_two_factor : 
  let n := 1004 in 10^n - 4^(n//2) = 2^1007 * k → ∃ k : ℤ, k % 2 ≠ 0 :=
by
  sorry

end greatest_power_two_factor_l516_516907


namespace degrees_subtraction_l516_516280

theorem degrees_subtraction :
  (108 * 3600 + 18 * 60 + 25) - (56 * 3600 + 23 * 60 + 32) = (51 * 3600 + 54 * 60 + 53) :=
by sorry

end degrees_subtraction_l516_516280


namespace chocolate_bars_in_each_box_l516_516972

theorem chocolate_bars_in_each_box (total_bars small_boxes : ℕ) (h1 : total_bars = 375) (h2 : small_boxes = 15) :
  total_bars / small_boxes = 25 :=
by
  rw [h1, h2]
  -- add the necessary steps for the proof
  sorry

end chocolate_bars_in_each_box_l516_516972


namespace smallest_prime_with_digit_sum_22_l516_516202

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_with_digit_sum_22 : ∃ p : ℕ, is_prime p ∧ digit_sum p = 22 ∧ 
  (∀ q : ℕ, is_prime q ∧ digit_sum q = 22 → p ≤ q) ∧ p = 499 :=
sorry

end smallest_prime_with_digit_sum_22_l516_516202


namespace men_to_complete_work_l516_516236

theorem men_to_complete_work (x : ℕ) (h1 : 10 * 80 = x * 40) : x = 20 :=
by
  sorry

end men_to_complete_work_l516_516236


namespace michael_completes_in_50_days_l516_516561

theorem michael_completes_in_50_days :
  ∀ {M A W : ℝ},
    (W / M + W / A = W / 20) →
    (14 * W / 20 + 10 * W / A = W) →
    M = 50 :=
by
  sorry

end michael_completes_in_50_days_l516_516561


namespace W_is_basis_of_orthogonal_complement_l516_516011

open Real
open Matrix

def vector_space : Type := VectorSpace ℝ (Fin 4)

def v1 : vector_space := ![1, -1, -1, 1]
def v2 : vector_space := ![1, -1, 1, -1]

def subspace_V : Submodule ℝ vector_space :=
  Submodule.span ℝ ({v1, v2} : Set vector_space)

def is_orthogonal (v w : vector_space) : Prop := dot_product v w = 0

def orthogonal_complement_W (V : Submodule ℝ vector_space) : Submodule ℝ vector_space :=
  { carrier := {w | ∀ v ∈ V, is_orthogonal v w },
    zero_mem' := by simp [is_orthogonal],
    add_mem' := by
      intros w₁ w₂ hw₁ hw₂ v hv
      simp [is_orthogonal, dot_add]
      exact add_eq_zero_iff_eq_zero_and_eq_zero.mp (hw₁ v hv ▸ hw₂ v hv ▸ rfl),
    smul_mem' := by
      intros c w hw v hv
      simp [is_orthogonal, dot_smul]
      exact mul_eq_zero_iff.mp (hw v hv ▸ rfl) }

def W_basis : Set (vector_space) := {![1, 1, 0, 0], ![0, 0, 1, 1]}

theorem W_is_basis_of_orthogonal_complement :
  Submodule.span ℝ W_basis = orthogonal_complement_W subspace_V :=
  sorry

end W_is_basis_of_orthogonal_complement_l516_516011


namespace rationalize_and_sum_l516_516471

theorem rationalize_and_sum (A B C : ℤ : C > 0 ∧ ¬ ∃ p: ℤ, p^3 ∣ B) :
  (3 : ℝ) / (2 * real.cbrt (5 : ℝ)) = (A : ℝ) * real.cbrt (B : ℝ) / (C : ℝ) →
  A + B + C = 38 :=
sorry

end rationalize_and_sum_l516_516471


namespace sqrt_five_is_infinite_non_repeating_decimal_l516_516505

/- Definitions: -/
def rational (x : ℚ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q
def finite_decimal (x : ℝ) : Prop := ∃ n : ℕ, (n : ℝ) > x
def infinite_repeating_decimal (x : ℝ) : Prop := ∃ S : List ℕ, ∃ k : ℕ, x = S / k
def infinite_non_repeating_decimal (x : ℝ) : Prop := ∀ S : List ℕ, ∀ k : ℕ, x ≠ S / k

/- Main Lean 4 statement: -/
theorem sqrt_five_is_infinite_non_repeating_decimal :
  infinite_non_repeating_decimal (Real.sqrt 5) := 
sorry

end sqrt_five_is_infinite_non_repeating_decimal_l516_516505


namespace inv_256_mod_101_l516_516350

theorem inv_256_mod_101 (h : 16⁻¹ ≡ 31 [MOD 101]) : 256⁻¹ ≡ 52 [MOD 101] := 
by 
  sorry

end inv_256_mod_101_l516_516350


namespace assignment_plans_count_l516_516395

/-- A proof that the number of different assignment plans for the volunteers and tasks given the specific conditions is 36. -/
theorem assignment_plans_count :
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guiding", "etiquette", "driving"]
  let permissible_roles_zhang_zhao := ["translation", "tour guiding"]
  let permissible_roles_others := ["translation", "tour guiding", "etiquette", "driving"]
  (calculate_assignment_plans volunteers tasks permissible_roles_zhang_zhao permissible_roles_others) = 36 :=
  by sorry

noncomputable def calculate_assignment_plans (volunteers : List String) (tasks : List String) 
  (permissible_roles_zhang_zhao : List String) (permissible_roles_others : List String) : Nat := 
  -- A placeholder for the function implementation
  sorry

end assignment_plans_count_l516_516395


namespace find_positive_solutions_l516_516305

theorem find_positive_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_pos : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄ ∧ 0 < x₅)
    (h1 : x₁ + x₂ = x₃^2)
    (h2 : x₂ + x₃ = x₄^2)
    (h3 : x₃ + x₄ = x₅^2)
    (h4 : x₄ + x₅ = x₁^2)
    (h5 : x₅ + x₁ = x₂^2) :
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
    by {
        -- Proof goes here
        sorry
    }

end find_positive_solutions_l516_516305


namespace projection_of_vector_l516_516876

theorem projection_of_vector
  (v : ℝ × ℝ)
  (hv1 : (0, 2) • v = (6 / 13, -4 / 13) ) :
  let p := (3, -1)
      v' := (6, -4)
  in ∃ c : ℝ, (p • v' / c = (33 / 13, -22 / 13)) :=
sorry

end projection_of_vector_l516_516876


namespace problem_conditions_l516_516054

theorem problem_conditions (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
    (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
    (↑2 = ([(x^2 + y^2 < a^2 + b^2), (x^2 - y^2 < a^2 - b^2), 
             (x^2 * y^2 < a^2 * b^2), (x^2 / y^2 < a^2 / b^2)] 
             |>.count id)) := 
sorry

end problem_conditions_l516_516054


namespace prob_arrival_times_diff_at_least_10_minutes_l516_516477

noncomputable def probability_diff_at_least_10_minutes : ℚ :=
let total_area := (30 * 30 : ℚ),
    favorable_area := (2 * (1 / 2 * 20 * 20 : ℚ)) in
favorable_area / total_area

theorem prob_arrival_times_diff_at_least_10_minutes :
  probability_diff_at_least_10_minutes = 4 / 9 := 
sorry

end prob_arrival_times_diff_at_least_10_minutes_l516_516477


namespace initial_ratio_of_milk_to_water_l516_516767

theorem initial_ratio_of_milk_to_water (M W : ℕ) (H1 : M + W = 45)
  (H2 : M / (W + 11 : ℕ) = 1.8) : M / W = 4 := 
by 
  sorry

end initial_ratio_of_milk_to_water_l516_516767


namespace sid_spent_on_computer_accessories_l516_516075

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l516_516075


namespace log_identity_solution_l516_516374

theorem log_identity_solution (b x : ℝ) (h_b_pos : b > 0) (h_b_not_one : b ≠ 1) (h_x_not_one : x ≠ 1) 
    (h_log_eq : log b^3 x + log x^3 b = 1) :
    x = b ^ ((3 + sqrt 5) / 2) ∨ x = b ^ ((3 - sqrt 5) / 2) :=
sorry

end log_identity_solution_l516_516374


namespace same_number_on_four_dice_l516_516186

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516186


namespace exponential_growth_miller_town_l516_516763

def percentage_miller_town (year : ℕ) : ℝ :=
  if year = 2000 then 0.08
  else if year = 2005 then 0.12
  else if year = 2010 then 0.20
  else if year = 2015 then 0.40
  else 0

theorem exponential_growth_miller_town :
  ∃ f : ℝ → ℝ, (∀ t : ℕ, t ∈ {2000, 2005, 2010, 2015} → (∃ a b : ℝ, a > 0 ∧ b > 1 ∧ f t = a * b ^ (t - 2000))) :=
by
  sorry

end exponential_growth_miller_town_l516_516763


namespace contractor_days_engaged_l516_516964

theorem contractor_days_engaged :
  ∃ (x : ℕ), (∀ (y : ℕ), y = 8 → (25 * x - 7.5 * y = 490)) ∧ x = 22 :=
by
  sorry

end contractor_days_engaged_l516_516964


namespace remainder_9876543210_mod_101_l516_516921

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end remainder_9876543210_mod_101_l516_516921


namespace smallest_N_l516_516770

theorem smallest_N (N : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) :
  (c1 = 6 * c2 - 5) ∧
  (N = 35 * c2 - 35) ∧
  (c3 = 5 * c4 - 3) ∧
  (N = 30 * c4 - 41) ∧
  (c5 = 47) ∧
  (c6 = 47) ∧
  (x1 = y2) ∧
  (x2 = y1) ∧
  (x3 = y4) ∧
  (x4 = y5) ∧
  (x5 = y6) ∧
  (x6 = y3) →
  N = 180 :=
begin
  intros h,
  sorry
end

end smallest_N_l516_516770


namespace find_g_l516_516631

open Function

def linear_system (a b c d e f g : ℚ) :=
  a + b + c + d + e = 1 ∧
  b + c + d + e + f = 2 ∧
  c + d + e + f + g = 3 ∧
  d + e + f + g + a = 4 ∧
  e + f + g + a + b = 5 ∧
  f + g + a + b + c = 6 ∧
  g + a + b + c + d = 7

theorem find_g (a b c d e f g : ℚ) (h : linear_system a b c d e f g) : 
  g = 13 / 3 :=
sorry

end find_g_l516_516631


namespace bisector_of_angle_l516_516336

theorem bisector_of_angle
  (A B C D S T : Type)
  [EuclideanGeometry A B C D S T]
  (h1 : IsDiameter A B)
  (h2 : OnSemicircle C A B)
  (h3 : OnSemicircle D A B)
  (h4 : S = intersection (line_through A C) (line_through B D))
  (h5 : T = foot_of_perpendicular S (segment A B)) :
  IsAngleBisector (line_through S T) ∠CTD := 
sorry

end bisector_of_angle_l516_516336


namespace angle_sum_eq_pi_div_2_l516_516943

open Real

theorem angle_sum_eq_pi_div_2 (θ1 θ2 : ℝ) (h1 : 0 < θ1 ∧ θ1 < π / 2) (h2 : 0 < θ2 ∧ θ2 < π / 2)
  (h : (sin θ1)^2020 / (cos θ2)^2018 + (cos θ1)^2020 / (sin θ2)^2018 = 1) :
  θ1 + θ2 = π / 2 :=
sorry

end angle_sum_eq_pi_div_2_l516_516943


namespace Jill_talking_time_total_l516_516792

-- Definition of the sequence of talking times
def talking_time : ℕ → ℕ 
| 0 => 5
| (n+1) => 2 * talking_time n

-- The statement we need to prove
theorem Jill_talking_time_total :
  (talking_time 0) + (talking_time 1) + (talking_time 2) + (talking_time 3) + (talking_time 4) = 155 :=
by
  sorry

end Jill_talking_time_total_l516_516792


namespace smallest_prime_with_digit_sum_22_l516_516206

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem smallest_prime_with_digit_sum_22 :
  (∃ n : ℕ, is_prime n ∧ digit_sum n = 22 ∧ ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22) → n ≤ m) ∧
  ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22 ∧ m < 499) → false := 
sorry

end smallest_prime_with_digit_sum_22_l516_516206


namespace minimum_groups_needed_l516_516580

theorem minimum_groups_needed :
  ∃ (g : ℕ), g = 5 ∧ ∀ n k : ℕ, n = 30 → k ≤ 7 → n / k = g :=
by
  sorry

end minimum_groups_needed_l516_516580


namespace num_distinct_sums_l516_516683

def is_proper_fraction (n d : ℕ) : Prop :=
  d ∣ 1000 ∧ d ≠ 1 ∧ 0 < n ∧ n < d ∧ gcd n d = 1

def sum_proper_fraction (n d : ℕ) : ℕ := n + d

def unique_sums_proper_fractions : ℕ :=
  (finset.image (λ (nd : ℕ × ℕ), sum_proper_fraction nd.1 nd.2)
    (finset.filter (λ (nd : ℕ × ℕ), is_proper_fraction nd.1 nd.2)
      (finset.product (finset.range 1000) (finset.range 1000)))).card

theorem num_distinct_sums : unique_sums_proper_fractions = 863 := 
sorry

end num_distinct_sums_l516_516683


namespace find_number_l516_516219

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def XiaoQian_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ n < 5

def XiaoLu_statements (n : ℕ) : Prop :=
  n < 7 ∧ 10 ≤ n ∧ n < 100

def XiaoDai_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ ¬ (n < 5)

theorem find_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 99 ∧ 
    ( (XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ XiaoDai_statements n) ) ∧
    n = 9 :=
sorry

end find_number_l516_516219


namespace johns_yearly_music_cost_l516_516418

theorem johns_yearly_music_cost 
  (hours_per_month : ℕ := 20)
  (minutes_per_hour : ℕ := 60)
  (average_song_length : ℕ := 3)
  (cost_per_song : ℕ := 50) -- represented in cents to avoid decimals
  (months_per_year : ℕ := 12)
  : (hours_per_month * minutes_per_hour // average_song_length) * cost_per_song * months_per_year = 2400 * 100 := -- 2400 dollars (* 100 to represent cents)
  sorry

end johns_yearly_music_cost_l516_516418


namespace fundraiser_rate_per_hour_l516_516024
open Real -- Real numbers

theorem fundraiser_rate_per_hour
  (volunteers_last_week : ℕ) (hours_per_volunteer_last_week : ℕ) (rate_per_hour_last_week : ℝ) 
  (volunteers_this_week : ℕ) (hours_per_volunteer_this_week : ℕ) :
  volunteers_last_week = 8 →
  hours_per_volunteer_last_week = 40 →
  rate_per_hour_last_week = 18 →
  volunteers_this_week = 12 →
  hours_per_volunteer_this_week = 32 →
  let total_amount_last_week := volunteers_last_week * hours_per_volunteer_last_week * rate_per_hour_last_week in
  let total_amount_this_week := total_amount_last_week in
  let rate_per_hour_this_week := total_amount_last_week / (volunteers_this_week * hours_per_volunteer_this_week) in
  rate_per_hour_this_week = 15 :=
by
  intros
  have h1 : total_amount_last_week = 8 * 40 * 18 := by rw [H, H_1, H_2]; sorry
  have h2 : total_amount_this_week = 5760 := by rw [<- h1]; sorry
  have h3 : total_amount_this_week = 12 * 32 * (total_amount_last_week / (12 * 32)) := by sorry
  have h4 : rate_per_hour_this_week = 5760 / (12 * 32) := by sorry
  have h5 : rate_per_hour_this_week = 15 := by sorry
  exact h5

end fundraiser_rate_per_hour_l516_516024


namespace same_number_on_four_dice_l516_516191

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l516_516191


namespace cow_heavy_chicken_cow_heavy_duck_l516_516960

variables (weight_chicken weight_duck weight_cow : ℕ)

-- Given condition statements
def chicken_weight: weight_chicken = 3 := sorry
def duck_weight: weight_duck = 6 := sorry
def cow_weight: weight_cow = 624 := sorry

-- Proof statement for the cow being 208 times heavier than the chicken
theorem cow_heavy_chicken (h1 : weight_cow = 624) (h2 : weight_chicken = 3): weight_cow / weight_chicken = 208 :=
by 
  sorry

-- Proof statement for the cow being 104 times heavier than the duck
theorem cow_heavy_duck (h1 : weight_cow = 624) (h2 : weight_duck = 6): weight_cow / weight_duck = 104 :=
by 
  sorry

end cow_heavy_chicken_cow_heavy_duck_l516_516960


namespace positive_solution_count_l516_516671

theorem positive_solution_count (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
    (∃ y, y > 0 ∧ y ≤ 1 ∧ sin (arccos (tan (arcsin y))) = y) ∧ 
    (∀ z, z > 0 ∧ z ≤ 1 ∧ sin (arccos (tan (arcsin z))) = z → z = √(2 - √3)) :=
begin
  sorry
end

end positive_solution_count_l516_516671


namespace length_of_second_train_l516_516577

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (cross_time : ℝ)
  (opposite_directions : Bool) :
  speed_first_train = 120 / 3.6 →
  speed_second_train = 80 / 3.6 →
  cross_time = 9 →
  length_first_train = 260 →
  opposite_directions = true →
  ∃ (length_second_train : ℝ), length_second_train = 240 :=
by
  sorry

end length_of_second_train_l516_516577


namespace prob_no_rain_four_days_l516_516510

noncomputable def prob_rain_one_day : ℚ := 2 / 3

noncomputable def prob_no_rain_one_day : ℚ := 1 - prob_rain_one_day

def independent_events (events : List (Unit → Prop)) : Prop :=
  -- A statement about independence of events
  sorry

theorem prob_no_rain_four_days :
  let days := 4
  let prob_no_rain := prob_no_rain_one_day
  independent_events (List.replicate days (fun _ => prob_no_rain)) →
  (prob_no_rain^days) = (1/81) := 
by
  sorry

end prob_no_rain_four_days_l516_516510


namespace dice_same_number_probability_l516_516193

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l516_516193


namespace approx_number_place_l516_516614

theorem approx_number_place : ∀ x : ℝ, x = 0.80 → (num_decimal_places x = 2) ∧ (approximation_place x = "hundredths") :=
by
  sorry

-- Define what num_decimal_places and approximation_place means if not defined
def num_decimal_places (x : ℝ) : Nat := sorry  -- Assuming a function that calculates the number of decimal places
def approximation_place (x : ℝ) : String := 
  if num_decimal_places x = 2 then "hundredths" else "unknown"  -- Assuming a basic interpretation for simplicity

end approx_number_place_l516_516614
