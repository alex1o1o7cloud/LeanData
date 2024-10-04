import Mathlib

namespace optionC_has_min_4_l548_548006

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548006


namespace average_sale_correct_l548_548141

theorem average_sale_correct :
  let sales := [5700, 8550, 6855, 3850, 14045] in
  let total_sales := sales.sum in
  let number_of_months := 5 in
  total_sales / number_of_months = 7800 :=
by
  let sales := [5700, 8550, 6855, 3850, 14045]
  let total_sales := sales.sum
  let number_of_months := 5
  sorry

end average_sale_correct_l548_548141


namespace QH_perpendicular_to_HD_l548_548689

theorem QH_perpendicular_to_HD
  (A B C D P Q H : Type*)
  [square : is_square ABCD]
  (proof1 : ∃ x : ℝ, BP = x ∧ BQ = x)
  (proof2 : is_foot_perpendicular B PC H) :
  is_perpendicular QH HD :=
sorry

end QH_perpendicular_to_HD_l548_548689


namespace store_profit_l548_548153

variable {x y : ℝ} -- Purchase prices of the calculators.
variable {selling_price : ℝ := 64} -- Selling price of each calculator.
variable {profit_percent : ℝ := 0.60} -- Profit percentage for first calculator.
variable {loss_percent : ℝ := 0.20} -- Loss percentage for second calculator.

theorem store_profit :
  (selling_price * 2 - (x + y) = 8) ↔ 
  (64 * 2 - (40 + 80) = 8) := by
  have x_value : x = 40 := by sorry
  have y_value : y = 80 := by sorry
  sorry

end store_profit_l548_548153


namespace custom_operation_example_l548_548987

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l548_548987


namespace angle_bisectors_intersect_on_YZ_l548_548609

noncomputable theory
open_locale classical

variables {Γ₁ Γ₂ Γ : Type} {Y Z A B : Γ₁}

-- Define intersecting circles Γ₁ and Γ₂ at points Y and Z
axiom intersecting_circles (Γ₁ Γ₂ : Type) (Y Z : Γ₁) : Prop
-- Define circle Γ externally tangent to Γ₁ at A and to Γ₂ at B
axiom externally_tangent (Γ : Type) (Γ₁ : Type) (Γ₂ : Type) (A B : Γ₁) : Prop

-- Define the angle bisectors and their intersection lying on YZ
def intersection_on_YZ (Γ₁ Γ₂ Γ : Type) (Y Z A B : Γ₁)
  (h1 : intersecting_circles Γ₁ Γ₂ Y Z)
  (h2 : externally_tangent Γ Γ₁ Γ₂ A B) : Prop :=
  -- The intersection of the bisectors of ∠ZAY and ∠ZBY lies on YZ.
  sorry

-- The theorem to prove
theorem angle_bisectors_intersect_on_YZ (Γ₁ Γ₂ Γ : Type) (Y Z A B : Γ₁)
  (h1 : intersecting_circles Γ₁ Γ₂ Y Z)
  (h2 : externally_tangent Γ Γ₁ Γ₂ A B) :
  intersection_on_YZ Γ₁ Γ₂ Γ Y Z A B h1 h2 :=
begin
  sorry
end

end angle_bisectors_intersect_on_YZ_l548_548609


namespace sum_lcm_reciprocal_lt_two_l548_548375

noncomputable def lcm : ℕ → ℕ → ℕ := Nat.lcm

theorem sum_lcm_reciprocal_lt_two {a : ℕ → ℕ} (n : ℕ) (h1 : ∀ i j, i < j → a i < a j)
  (h2 : ∀ i, 0 < a i) :
  ∑ i in Finset.range n, (1 : ℚ) / a i = (1 : ℚ) / a 0 + (1 : ℚ) / lcm a 0 a 1 + (1 : ℚ) / (lcm (lcm a 0 a 1) a 2) + ... + (1 : ℚ) / lcm (lcm ... a (n - 1)) < 2 :=
sorry

end sum_lcm_reciprocal_lt_two_l548_548375


namespace rate_of_interest_per_annum_l548_548041

theorem rate_of_interest_per_annum :
  ∀ (P1 P2 T1 T2 SI1 SI2 TotalInterest R : ℝ),
  P1 = 5000 → T1 = 2 →
  P2 = 3000 → T2 = 4 →
  TotalInterest = 3300 →
  SI1 = P1 * T1 * R / 100 →
  SI2 = P2 * T2 * R / 100 →
  SI1 + SI2 = TotalInterest →
  R = 15 := by {
  intros P1 P2 T1 T2 SI1 SI2 TotalInterest R,
  intros hP1 hT1 hP2 hT2 hTotalInterest hSI1 hSI2 hSumInterest,
  
  sorry
}

end rate_of_interest_per_annum_l548_548041


namespace sum_of_a_b_vert_asymptotes_l548_548638

theorem sum_of_a_b_vert_asymptotes (a b : ℝ) 
  (h1 : ∀ x : ℝ, x = -1 → x^2 + a * x + b = 0) 
  (h2 : ∀ x : ℝ, x = 3 → x^2 + a * x + b = 0) : 
  a + b = -5 :=
sorry

end sum_of_a_b_vert_asymptotes_l548_548638


namespace relationship_xy_l548_548274

def M (x : ℤ) : Prop := ∃ m : ℤ, x = 3 * m + 1
def N (y : ℤ) : Prop := ∃ n : ℤ, y = 3 * n + 2

theorem relationship_xy (x y : ℤ) (hx : M x) (hy : N y) : N (x * y) ∧ ¬ M (x * y) :=
by
  sorry

end relationship_xy_l548_548274


namespace pyramid_volume_correct_l548_548487

noncomputable def pyramidVolume (a b h : ℝ) (ha : a = 7) (hb : b = 10) (hc : h = 15) : ℝ :=
  (1 / 3) * (a * b) * h

theorem pyramid_volume_correct :
  pyramidVolume 7 10 (√(187.75)) 7 10 15 = 1564.17 := 
by
  sorry

end pyramid_volume_correct_l548_548487


namespace geometric_seq_a6_value_l548_548306

theorem geometric_seq_a6_value 
    (a : ℕ → ℝ) 
    (q : ℝ) 
    (h_q_pos : q > 0)
    (h_a_pos : ∀ n, a n > 0)
    (h_a2 : a 2 = 1)
    (h_a8_eq : a 8 = a 6 + 2 * a 4) : 
    a 6 = 4 := 
by 
  sorry

end geometric_seq_a6_value_l548_548306


namespace cole_drive_time_to_work_is_72_minutes_l548_548522

variable (D : ℝ) -- Distance from home to work in km
variable (T_work T_home : ℝ) -- Time in hours

-- Definitions corresponding to given conditions
def avg_speed_home_to_work : ℝ := 80
def avg_speed_work_to_home : ℝ := 120
def total_time_round_trip : ℝ := 2

-- Calculation constraints based on conditions
def time_to_work (D : ℝ) : ℝ := D / avg_speed_home_to_work
def time_to_home (D : ℝ) : ℝ := D / avg_speed_work_to_home

-- The proof goal statement
theorem cole_drive_time_to_work_is_72_minutes :
  (∃ D : ℝ, (time_to_work D) + (time_to_home D) = total_time_round_trip) →
  (time_to_work D) * 60 = 72 :=
by
  sorry

end cole_drive_time_to_work_is_72_minutes_l548_548522


namespace f_n_formula_l548_548973

noncomputable def f (x : ℝ) := x / Real.exp x

def f_n (n : ℕ) : (ℝ → ℝ) :=
  match n with
  | 0     => f
  | (n+1) => fun x => (derivative (f_n n)) x

theorem f_n_formula (n : ℕ) (h : n > 0) :
  ∀ x, f_n n x = ((-1 : ℝ) ^ n) * (x - n) / Real.exp x :=
sorry

end f_n_formula_l548_548973


namespace CanadianTrainingProblem_l548_548353

variable {Point : Type}
variable [EuclideanGeometry Point]

open EuclideanGeometry

/--
Let H be the orthocenter of triangle ABC. 
Let D, E, F be the midpoints of BC, CA, and AB respectively.
A circle centered at H intersects line DE at P and Q, 
line EF at R and S, and line FD at T and U.
Prove that CP = CQ = AR = AS = BT = BU.
-/
theorem CanadianTrainingProblem
  (A B C H D E F P Q R S T U : Point)
  (h_orthocenter : orthocenter A B C H)
  (h_midpoints : midpoint D B C ∧ midpoint E C A ∧ midpoint F A B)
  (h_circle : circle H r ∧ intersects_circle_at_line H P Q DE ∧
                                         intersects_circle_at_line H R S EF ∧
                                         intersects_circle_at_line H T U FD) :
  distance C P = distance C Q ∧
  distance A R = distance A S ∧
  distance B T = distance B U :=
sorry

end CanadianTrainingProblem_l548_548353


namespace total_shaded_area_l548_548851

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  1 * S ^ 2 + 8 * (T ^ 2) = 13.5 := by
  sorry

end total_shaded_area_l548_548851


namespace sum_of_valid_four_digit_numbers_l548_548975

theorem sum_of_valid_four_digit_numbers : 
  let isPerfectSquare (n : ℕ) := ∃ m : ℕ, m * m = n in
  let validNumbers := { n : ℕ | 
    1000 ≤ n ∧ n < 10000 ∧ 
    let A := n / 1000,
        B := (n / 100 % 10),
        C := (n / 10 % 10),
        D := n % 10,
        AB := 10 * A + B,
        BC := 10 * B + C,
        CD := 10 * C + D 
    in isPerfectSquare AB ∧ isPerfectSquare BC ∧ isPerfectSquare CD } in
  ∑ n in validNumbers, n = 13462 :=
by
  sorry

end sum_of_valid_four_digit_numbers_l548_548975


namespace f_of_minus_5_eq_0_l548_548253

noncomputable def f : ℝ → ℝ := sorry

axiom domain : ∀ x : ℝ, f x ∈ ℝ
axiom even_f : ∀ x : ℝ, f (1 - 2 * x) = f (1 + 2 * x)
axiom odd_f : ∀ x : ℝ, f (x - 1) = -f (1 - x)

theorem f_of_minus_5_eq_0 : f (-5) = 0 := by
  sorry

end f_of_minus_5_eq_0_l548_548253


namespace journey_time_l548_548444

-- Define the distances as real numbers
variables (d1 d2 : ℝ)
-- Define the total time for the journey
variables (T : ℝ)

-- Conditions given in the problem
def condition_1 : Prop := 150 = 150 -- Total distance is 150 miles (always true, so tautological)
def condition_2 : Prop := d1 / 30 + (150 - d1) / 3 = T -- Harry and Jack's time
def condition_3 : Prop := d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T -- Tom's time
def condition_4 : Prop := (d1 - d2) / 4 + (150 - (d1 - d2)) / 30 = T -- Dick's time

-- Goal: prove the total journey time T is 32 hours given the conditions
theorem journey_time : 
  ∃ d1 d2 T, 
  (condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4) → T = 32 :=
by
  sorry

end journey_time_l548_548444


namespace range_of_omega_l548_548293

-- Let us define the key elements based on the problem and conditions:
def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

-- Given function f and the conditions on ω and x, we need to prove:
theorem range_of_omega (ω : ℝ) (h1 : ω > 0) :
  (∃! x : ℝ, x ∈ Set.Ioo (-Real.pi / 6) (Real.pi / 3) ∧ IsLocalExtremum (λ x, f ω x) x) →
  ω ∈ Set.Ioc (3 / 2) 3 :=
sorry

end range_of_omega_l548_548293


namespace machine_A_time_l548_548709

theorem machine_A_time :
  let A := 4 in
  ∀ (B C : ℝ), B = 12 → C = 8 →
  let T := 2.181818181818182 in
  (1 / A + 1 / B + 1 / C = 5.5 / 12) →
  A = 4 :=
by
  intros B C hB hC T h_combined_rate
  sorry -- Proof goes here

end machine_A_time_l548_548709


namespace probability_of_rerolling_one_die_l548_548680

theorem probability_of_rerolling_one_die (d1 d2 : ℕ) (h1 : 1 ≤ d1) (h2 : d1 ≤ 6) (h3 : 1 ≤ d2) (h4 : d2 ≤ 6) :
  (let sum := d1 + d2 in (sum ≠ 9) ∧ (3 <= sum) ∧ (sum <= 7)) →
  let prob_one_reroll := (1 : ℚ) / 18 + 1 / 12 + 1 / 9 + 5 / 36 + 1 / 6 in
  prob_one_reroll = 17 / 36 :=
sorry

end probability_of_rerolling_one_die_l548_548680


namespace drama_club_rows_l548_548497

theorem drama_club_rows (h : 108 = 2^2 * 3^3) : 
  ∃ (n : ℕ), (n = 4) ∧
  ∃ (rows : list ℕ), 
    (∀ x ∈ rows, x ∣ 108) ∧ 
    (∀ x ∈ rows, 6 ≤ x ∧ x ≤ 18) ∧
    rows.length = n :=
by
  sorry

end drama_club_rows_l548_548497


namespace smallest_solution_proof_l548_548209

noncomputable def smallest_solution (x : ℝ) : ℝ :=
  if x = (1 - Real.sqrt 65) / 4 then x else x

theorem smallest_solution_proof :
  ∃ x : ℝ, (2 * x / (x - 2) + (2 * x^2 - 24) / x = 11) ∧
           (∀ y : ℝ, 2 * y / (y - 2) + (2 * y^2 - 24) / y = 11 → y ≥ (1 - Real.sqrt 65) / 4) ∧
           x = (1 - Real.sqrt 65) /4 :=
sorry

end smallest_solution_proof_l548_548209


namespace discount_percentage_l548_548363

theorem discount_percentage (sale_price original_price : ℝ) (h1 : sale_price = 480) (h2 : original_price = 600) : 
  100 * (original_price - sale_price) / original_price = 20 := by 
  sorry

end discount_percentage_l548_548363


namespace find_digit_A_l548_548567

theorem find_digit_A (A : ℕ) (h : A ∈ {0, 1, 2, 3, 4}) :
    (∃ x : ℤ, 25 + 6 * A = x^2 ∧ ∃ y : ℤ, 36 + 7 * A = y^3) ↔ A = 4 :=
by
  sorry

end find_digit_A_l548_548567


namespace ratio_of_means_l548_548906

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end ratio_of_means_l548_548906


namespace find_x2_x1_x3_l548_548702

noncomputable def poly := (λ x : ℝ, (real.sqrt 120) * x^3 - 480 * x^2 + 8 * x + 1)

theorem find_x2_x1_x3 :
  ∃ (x1 x2 x3 : ℝ), x3 < x2 ∧ x2 < x1 ∧ (poly x1 = 0) ∧ (poly x2 = 0) ∧ (poly x3 = 0) ∧ (x2 * (x1 + x3) = -1 / 120) :=
sorry

end find_x2_x1_x3_l548_548702


namespace find_min_f_l548_548598

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem find_min_f (h_tangent : ∀ y : ℝ, f 1 = 0 ∧ (y = x - 1) → f' 1 = 1)
    (h_deriv : ∀ x : ℝ, f' x = Real.log x + 1) : 
    ∃ x_min : ℝ, x_min = 1 / Real.exp 1 ∧ f x_min = -1 / Real.exp 1 := 
sorry

end find_min_f_l548_548598


namespace plywood_cut_difference_l548_548074

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548074


namespace highest_score_not_necessarily_sixteen_l548_548831

variable (n : ℕ) (points : ℕ → ℕ)
variable (teams : Fin n)
variable (tournament : teams → teams → Sum ℕ (teams × teams))

theorem highest_score_not_necessarily_sixteen
  (h₁ : ∀ t₁ t₂, t₁ ≠ t₂ → Sum.inr (t₁, t₂) ∈ tournament t₁ t₂ ∨ Sum.inr (t₁, t₂) ∈ tournament t₂ t₁)
  (h₂ : ∀ t₁ t₂, Sum.inl 2 ∈ tournament t₁ t₂ ∨ Sum.inl 2 ∈ tournament t₂ t₁ → 2 ∣ points t₁ + points t₂)
  (h₃ : ∀ t₁, ∃ t₂, Sum.inr (t₁, t₂) ∈ tournament t₂ t₁)
  (h₄ : ∑ i in Finset.univ, points i = 240)
: ∃ t, points t ≤ 15 :=
begin
  -- The detailed proof steps will go here
  sorry
end

end highest_score_not_necessarily_sixteen_l548_548831


namespace water_percentage_in_fresh_mushrooms_l548_548216

theorem water_percentage_in_fresh_mushrooms
  (fresh_mushrooms_mass : ℝ)
  (dried_mushrooms_mass : ℝ)
  (dried_mushrooms_water_percentage : ℝ)
  (dried_mushrooms_non_water_mass : ℝ)
  (fresh_mushrooms_dry_percentage : ℝ)
  (fresh_mushrooms_water_percentage : ℝ)
  (h1 : fresh_mushrooms_mass = 22)
  (h2 : dried_mushrooms_mass = 2.5)
  (h3 : dried_mushrooms_water_percentage = 12 / 100)
  (h4 : dried_mushrooms_non_water_mass = dried_mushrooms_mass * (1 - dried_mushrooms_water_percentage))
  (h5 : fresh_mushrooms_dry_percentage = dried_mushrooms_non_water_mass / fresh_mushrooms_mass * 100)
  (h6 : fresh_mushrooms_water_percentage = 100 - fresh_mushrooms_dry_percentage) :
  fresh_mushrooms_water_percentage = 90 := 
by
  sorry

end water_percentage_in_fresh_mushrooms_l548_548216


namespace plywood_perimeter_difference_l548_548078

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548078


namespace parabola_find_c_l548_548835

theorem parabola_find_c (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 2)
  (h2 : (5 : ℝ)^2 + b * 5 + c = 2) : 
  c = 7 := by
  sorry

end parabola_find_c_l548_548835


namespace money_constraints_l548_548913

variable (a b : ℝ)

theorem money_constraints (h1 : 8 * a - b = 98) (h2 : 2 * a + b > 36) : a > 13.4 ∧ b > 9.2 :=
sorry

end money_constraints_l548_548913


namespace total_amount_in_euros_l548_548500

variable (x y z w : ℝ)
variable (share_y share_z : ℝ)
variable (total_usd : ℝ)
variable (conversion_rate : ℝ)
variable (total_eur : ℝ)

-- Conditions as given in problem
def condition1 : Prop := y = 0.8 * x
def condition2 : Prop := z = 0.7 * x
def condition3 : Prop := w = 0.6 * x
def condition4 : Prop := share_y = 42
def condition5 : Prop := share_z = 49
def condition6 : Prop := x + w = 120
def condition7 : Prop := conversion_rate = 0.85

-- Calculated total amount in euros
def target : Prop := total_usd = x + share_y + share_z + w
def conversion : Prop := total_eur = total_usd * conversion_rate

theorem total_amount_in_euros :
  condition1 → condition2 → condition3 → condition4 → condition5 → condition6 → condition7 →
  target → conversion →
  total_eur = 172.55 :=
by
  sorry

end total_amount_in_euros_l548_548500


namespace pump_C_time_l548_548873

/-- Pump A has a rate of fulfilling or pumping-out the tank in 2 hours --/
def rate_A := 1 / 2 -- 0.5 tanks per hour

/-- Pump B has a rate of fulfilling or pumping-out the tank in 3 hours --/
def rate_B := 1 / 3 -- approximately 0.333 tanks per hour

/-- The rate at which pumps A and B can together pump-out the water from half-full tank minus the rate at which pump C fills the tank, will empty the tank in 0.75 hours --/
def combined_outflow_minus_C (rate_C : ℝ) : Prop :=
  (rate_A + rate_B) - rate_C = 2 / 3  -- 0.667 tanks per hour

/-- To find how many hours it takes for pump C to fulfill or pump-out the full tank --/
def time_for_C (rate_C : ℝ) : ℝ :=
  1 / rate_C

/-- Main theorem: Given the conditions, prove the rate of pump C and then determine the time to fulfill or pump-out the full tank --/
theorem pump_C_time : 
  ∃ rate_C : ℝ, combined_outflow_minus_C rate_C ∧ time_for_C rate_C ≈ 6.024 :=
begin
  use 1 / 6.024, -- rate_C ≈ 0.166 tanks per hour
  split,
  { sorry }, -- proof of (rate_A + rate_B - rate_C = 2 / 3)
  { sorry } -- proof of (time_for_C rate_C ≈ 6.024)
end

end pump_C_time_l548_548873


namespace probability_two_face_cards_l548_548776

def cardDeck : ℕ := 52
def totalFaceCards : ℕ := 12

-- Probability of selecting one face card as the first card
def probabilityFirstFaceCard : ℚ := totalFaceCards / cardDeck

-- Probability of selecting another face card as the second card
def probabilitySecondFaceCard (cardsLeft : ℕ) : ℚ := (totalFaceCards - 1) / cardsLeft

-- Combined probability of selecting two face cards
theorem probability_two_face_cards :
  let combined_probability := probabilityFirstFaceCard * probabilitySecondFaceCard (cardDeck - 1)
  combined_probability = 22 / 442 := 
  by
    sorry

end probability_two_face_cards_l548_548776


namespace geometric_sequence_angle_count_l548_548189

theorem geometric_sequence_angle_count :
  (∃ θs : Finset ℝ, (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℕ, θ = k * (π / 2)) 
                    ∧ θs.card = 4
                    ∧ ∀ θ ∈ θs, ∃ a b c : ℝ, (a, b, c) = (Real.sin θ, Real.cos θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.sin θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.cos θ, Real.tan θ)
                                             ∧ b = a * c) :=
sorry

end geometric_sequence_angle_count_l548_548189


namespace radius_of_congruent_spheres_in_cone_l548_548678

noncomputable def radius_of_congruent_spheres (base_radius height : ℝ) : ℝ := 
  let slant_height := Real.sqrt (height^2 + base_radius^2)
  let r := (4 : ℝ) / (10 + 4) * slant_height
  r

theorem radius_of_congruent_spheres_in_cone :
  radius_of_congruent_spheres 4 10 = 4 * Real.sqrt 29 / 7 := by
  sorry

end radius_of_congruent_spheres_in_cone_l548_548678


namespace kamal_english_marks_l548_548334

theorem kamal_english_marks
  (marks_math : ℕ = 65)
  (marks_physics : ℕ = 77)
  (marks_chemistry : ℕ = 62)
  (marks_biology : ℕ = 75)
  (avg_marks : ℕ = 69)
  (num_subjects : ℕ = 5) :
  ∃ marks_english : ℕ, marks_english = 66 := 
by
  let total_marks := avg_marks * num_subjects
  let known_marks_sum := marks_math + marks_physics + marks_chemistry + marks_biology
  have marks_english := total_marks - known_marks_sum
  exact ⟨marks_english, sorry⟩

end kamal_english_marks_l548_548334


namespace common_movie_l548_548715

def movies : Type := {A, B, C, D, E}

def Zhao_set : Set movies := {A, C, D, E}
def Zhang_set : Set movies := {B, C, D, E}
def Li_set : Set movies := {A, B, D, E}
def Liu_set : Set movies := {A, B, C, D}

theorem common_movie :
  {m | m ∈ Zhao_set} ∩ {m | m ∈ Zhang_set} ∩ {m | m ∈ Li_set} ∩ {m | m ∈ Liu_set} = {D} :=
by
  sorry

end common_movie_l548_548715


namespace number_of_positive_integer_pairs_l548_548622

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l548_548622


namespace polynomial_r_properties_l548_548696

noncomputable def r (x : ℝ) : ℝ := sorry

theorem polynomial_r_properties :
  (∀ n : ℕ, n ≤ 7 → r (3^n) = 1 / 3^n) →
  degree (polynomial_of_fn r) = 7 →
  r 0 = 3 - 1 / 6561 :=
begin
  intros h1 h2,
  sorry
end

end polynomial_r_properties_l548_548696


namespace option_c_has_minimum_value_4_l548_548021

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548021


namespace domain_of_f_correct_l548_548743

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 2) + 1 / (4 - x^2)

-- Define the domain conditions for f(x) to be meaningful
def domain_condition_1 (x : ℝ) : Prop := x + 2 ≥ 0
def domain_condition_2 (x : ℝ) : Prop := 4 - x^2 ≠ 0

-- Define the domain of f(x)
def domain_of_f (x : ℝ) : Prop := x > -2 ∧ x ≠ 2

-- Proof statement to show what the domain of f(x) is, given the conditions
theorem domain_of_f_correct (x : ℝ) :
  (domain_condition_1 x ∧ domain_condition_2 x) ↔ domain_of_f x := by
  sorry

end domain_of_f_correct_l548_548743


namespace total_weight_of_new_people_l548_548738

theorem total_weight_of_new_people (W W_new : ℝ) :
  (∀ (old_weights : List ℝ), old_weights.length = 25 →
    ((old_weights.sum - (65 + 70 + 75)) + W_new = old_weights.sum + (4 * 25)) →
    W_new = 310) := by
  intros old_weights old_weights_length increase_condition
  -- Proof will be here
  sorry

end total_weight_of_new_people_l548_548738


namespace distribute_rabbits_l548_548379

/-- Definitions of the rabbits and conditions -/
section
variables (Store : Type) [Fintype Store] [DecidableEq Store]

/-- The rabbits: parents and children -/
inductive Rabbit : Type
| Peaches : Rabbit
| Pablo   : Rabbit
| Flash   : Rabbit
| Dash    : Rabbit
| Cash    : Rabbit
| Nash    : Rabbit

def Parents : Finset Rabbit := {Rabbit.Peaches, Rabbit.Pablo}
def Children : Finset Rabbit := {Rabbit.Flash, Rabbit.Dash, Rabbit.Cash, Rabbit.Nash}

/-- Define constraints and conditions. -/
def valid_distribution (dist : Rabbit → Store) : Prop :=
  ∀ s : Store,
    (∀ p ∈ Parents.to_finset, ∀ c ∈ Children.to_finset, dist p ≠ dist c) ∧
    (Parents.count_to_finset dist s ≤ 1) ∧ 
    (Children.count_to_finset dist s ≤ 2)

lemma count_rabbits 
  [finite : Fintype Store]
  (dist : Rabbit → Store) :
  valid_distribution dist →
  ∑ (rc : Rabbit) in {Rabbit.Peaches, Rabbit.Pablo, Rabbit.Flash, Rabbit.Dash, Rabbit.Cash, Rabbit.Nash}.to_finset, 
  ({dist rc}).card = 6 :=
by sorry

end

/-- Main proof statement. Z denotes the number of valid distributions. The goal is to show Z = 360. -/
theorem distribute_rabbits
  (Store : Type) [Fintype Store] [DecidableEq Store]
  (valid_distribution : Rabbit → Store → Prop) :
  ∃ Z : ℕ, Z = 360 :=
by sorry

end distribute_rabbits_l548_548379


namespace evaluate_expression_l548_548435

theorem evaluate_expression : 
  let S_n := (Finset.range 50).sum (λ k, (2 * k + 1)^2 - (2 * k + 3)^2)
  let D_n := (Finset.range 50).sum (λ k, (2 * k + 1) - (2 * k + 3))
  S_n / D_n = 100 := 
by
  sorry

end evaluate_expression_l548_548435


namespace number_of_committees_with_president_l548_548971

-- Define the conditions
variables (total_people : ℕ) (committee_size : ℕ) (remaining_people : ℕ)

/-- The main statement to prove: number of ways to form the committee with the president constraint -/
theorem number_of_committees_with_president : 
  (total_people = 12) → (committee_size = 5) → (remaining_people = 11) →
  nat.choose remaining_people (committee_size - 1) = 330 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end number_of_committees_with_president_l548_548971


namespace compute_paths_in_grid_l548_548526

def grid : List (List Char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'M', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'M', 'P', 'M', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'M', 'P', 'U', 'P', 'M', 'O', 'C', ' '],
  [' ', 'C', 'O', 'M', 'P', 'U', 'T', 'U', 'P', 'M', 'O', 'C'],
  ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'T', 'U', 'P', 'M', 'O', 'C']
]

def is_valid_path (path : List (Nat × Nat)) : Bool :=
  -- This function checks if a given path is valid according to the problem's grid and rules.
  sorry

def count_paths_from_C_to_E (grid: List (List Char)) : Nat :=
  -- This function would count the number of valid paths from a 'C' in the leftmost column to an 'E' in the rightmost column.
  sorry

theorem compute_paths_in_grid : count_paths_from_C_to_E grid = 64 :=
by
  sorry

end compute_paths_in_grid_l548_548526


namespace num_solutions_2_pow_2x_minus_3_pow_2y_eq_80_l548_548749

theorem num_solutions_2_pow_2x_minus_3_pow_2y_eq_80 :
  ∃! (solutions : Finset (ℤ × ℤ)), (solutions.card = 2) ∧
  (∀ (x y : ℤ), ((x, y) ∈ solutions ↔ 2^(2*x) - 3^(2*y) = 80)) :=
by
  sorry

end num_solutions_2_pow_2x_minus_3_pow_2y_eq_80_l548_548749


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548026

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548026


namespace carter_green_M_and_M_probability_l548_548178

theorem carter_green_M_and_M_probability :
  ∀ (initial_green initial_red eaten_green: ℕ) (sister_eats_fraction: ℚ) (added_yellow: ℕ),
  initial_green = 20 → 
  initial_red = 20 → 
  eaten_green = 12 → 
  sister_eats_fraction = 1/2 → 
  added_yellow = 14 → 
  ( (initial_green - eaten_green : ℕ) = 8 ) →
  ( (initial_red - (sister_eats_fraction * initial_red).to_nat) = 10 ) →
  ( percentage : ℚ = (8 / (8 + 10 + 14)) * 100) →
  percentage = 25 :=
by
  intros initial_green initial_red eaten_green sister_eats_fraction added_yellow h1 h2 h3 h4 h5 h6 h7_percentage
  have total_M_and_Ms : ℕ := 8 + 10 + 14
  have green_ratio : ℚ := 8 / total_M_and_Ms
  have percentage := green_ratio * 100
  exact sorry

end carter_green_M_and_M_probability_l548_548178


namespace hyperbola_equation_l548_548251

theorem hyperbola_equation (c a b : ℝ) (ecc : ℝ) (h_c : c = 3) (h_ecc : ecc = 3 / 2) (h_a : a = 2) (h_b : b^2 = c^2 - a^2) :
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 4 - y^2 / 5 = 1)) :=
by
  sorry

end hyperbola_equation_l548_548251


namespace smallest_even_integer_l548_548784

theorem smallest_even_integer :
  ∃ (x : ℤ), |3 * x - 4| ≤ 20 ∧ (∀ (y : ℤ), |3 * y - 4| ≤ 20 → (2 ∣ y) → x ≤ y) ∧ (2 ∣ x) :=
by
  use -4
  sorry

end smallest_even_integer_l548_548784


namespace speed_of_man_l548_548501

noncomputable def speed_of_man_kmh (train_length : ℝ) (cross_time : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let distance_covered := train_length
  let relative_speed_mps := distance_covered / cross_time
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man (train_length : ℝ) (cross_time : ℝ) (train_speed_kmph : ℝ) :
  train_length = 200 ∧ cross_time = 6 ∧ train_speed_kmph = 114.99 →
  speed_of_man_kmh train_length cross_time train_speed_kmph = 5.004 := 
by
  intros h
  cases h with a h1
  cases h1 with b c
  simp [a, b, c, speed_of_man_kmh]
  sorry

end speed_of_man_l548_548501


namespace ball_color_probability_l548_548912

def balls := fin 8               -- Define the balls as elements of a finite type of size 8.

-- Define a function to calculate the probability that each ball is different from more than half of the other balls.
def prob_diff_color_from_half (p : rat) : Prop :=
  p = 7/32

theorem ball_color_probability (h : ∀ b : balls, fin 2) :
  prob_diff_color_from_half (1 / 2 ^ 8 * choose 8 5 + 1 / 2 ^ 8 * choose 8 3) :=
sorry

end ball_color_probability_l548_548912


namespace boy_running_time_l548_548462

noncomputable def time_to_run_around_square (side : ℝ) (speed_kmh : ℝ) : ℝ :=
  let perimeter := 4 * side
  let speed_ms := speed_kmh * (1000 / 3600)
  perimeter / speed_ms

theorem boy_running_time (side : ℝ) (speed_kmh : ℝ) (h_side : side = 55) (h_speed : speed_kmh = 9) :
  time_to_run_around_square side speed_kmh = 88 :=
by
  rw [h_side, h_speed, time_to_run_around_square]
  norm_num
  sorry

end boy_running_time_l548_548462


namespace inequality_holds_l548_548387

theorem inequality_holds (n : ℕ) (x : Fin n → ℝ) 
  (h1 : ∀ i, 0 < x i) (h2 : (∏ i, x i) = 1) : 
  (∑ i, 1 / (n - 1 + x i)) ≤ 1 := 
by 
  sorry

end inequality_holds_l548_548387


namespace intersection_points_of_parametric_curve_l548_548408

def parametric_curve_intersection_points (t : ℝ) : Prop :=
  let x := t - 1
  let y := t + 2
  (x = -3 ∧ y = 0) ∨ (x = 0 ∧ y = 3)

theorem intersection_points_of_parametric_curve :
  ∃ t1 t2 : ℝ, parametric_curve_intersection_points t1 ∧ parametric_curve_intersection_points t2 := 
by
  sorry

end intersection_points_of_parametric_curve_l548_548408


namespace Christine_wandered_hours_l548_548888

theorem Christine_wandered_hours (distance speed : ℝ) (h_distance : distance = 20) (h_speed : speed = 4) : distance / speed = 5 := by
  sorry

end Christine_wandered_hours_l548_548888


namespace choosing_positions_from_group_l548_548310

theorem choosing_positions_from_group (n : ℕ) (h : n = 6) :
  (∏ i in finset.range 3, (n - i)) = 120 :=
by
  rw h
  norm_num
  sorry

end choosing_positions_from_group_l548_548310


namespace plywood_cut_difference_l548_548077

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548077


namespace number_of_leap_years_l548_548158

-- Define the given conditions for a leap year
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define the range of years
def year_range := {y // 2000 ≤ y ∧ y ≤ 4000}

-- Function to count leap years in the given range
def count_leap_years (s : Set year_range) : ℕ :=
  Set.count is_leap_year s

-- The number of leap years between 2000 and 4000 inclusive is 486
theorem number_of_leap_years : count_leap_years year_range = 486 :=
  sorry

end number_of_leap_years_l548_548158


namespace sqrt_7_approx_500_over_189_l548_548459

theorem sqrt_7_approx_500_over_189 : abs (sqrt 7 - (500 / 189 : ℚ)) < 0.001 :=
sorry

end sqrt_7_approx_500_over_189_l548_548459


namespace solve_eq_z6_neg_64_l548_548937

theorem solve_eq_z6_neg_64 (z : ℂ) (h : z ^ 6 = -64) :
z = (complex.sqrt[3] 2) * (1 + complex.I) ∨
z = (complex.sqrt[3] 2) * (-1 - complex.I) ∨
z = (complex.sqrt[3] 2) * (-1 + complex.I) ∨
z = (complex.sqrt[3] 2) * (1 - complex.I) :=
sorry

end solve_eq_z6_neg_64_l548_548937


namespace range_of_a_l548_548703

variable {ℝ : Type} [LinearOrderedField ℝ]

-- Given conditions as definitions in Lean 4
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def problem_conditions (f : ℝ → ℝ) (a : ℝ) : Prop :=
  is_odd_function f ∧ is_periodic f 3 ∧ f 1 > 1 ∧ f 2 = a

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : problem_conditions f a) : 
  a < -1 :=
by 
  -- Since we are proving an existence, we assume the conditions to be true and reach the conclusion
  cases h with oddf periodf rest,
  cases rest with f1gt1 f2_eq_a,
  have h1 : f (-1) = -f 1 := oddf 1,
  have h2 : f (-1) = f 2 := periodf (-1),
  rw [h1, f2_eq_a] at h2,
  have h3 : -f 1 = a := h2,
  linarith,
  sorry  -- handle missing proof steps

end range_of_a_l548_548703


namespace chord_length_eq_4_l548_548419

noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ :=
  (abs (a * px + b * py + c)) / sqrt (a * a + b * b)

noncomputable def chord_length (x1 y1 r a b c : ℝ) : ℝ :=
  let d := distance_point_to_line x1 y1 a b c in
  2 * sqrt (r * r - d * d)

theorem chord_length_eq_4 :
  chord_length 2 (-3) 3 1 (-2) (-3) = 4 :=
by
  sorry

end chord_length_eq_4_l548_548419


namespace altitude_proof_l548_548322

noncomputable def triangle_altitude_lt : Prop :=
  ∀ (A B C : Type) [ordered_field A] [linear_ordered_field B]
  (AB : A) (angle_CAB : B) (R : B),
  AB = 4 → angle_CAB = 60 → R = 2.2 →
  let CD := (2.2 * real.sqrt 3 / 2) * (real.sqrt 3 / 2) in
  CD < (11 * real.sqrt 3 / 5)

theorem altitude_proof : triangle_altitude_lt :=
sorry

end altitude_proof_l548_548322


namespace square_area_l548_548852

theorem square_area (p : ℝ → ℝ) (a b : ℝ) (h₁ : ∀ x, p x = x^2 + 3 * x + 2) (h₂ : p a = 5) (h₃ : p b = 5) (h₄ : a ≠ b) : (b - a)^2 = 21 :=
by
  sorry

end square_area_l548_548852


namespace num_digits_first_1500_odd_integers_l548_548455

theorem num_digits_first_1500_odd_integers : 
  (let num_one_digit_odd := (9 - 1) / 2 + 1 in
   let num_two_digit_odd := (99 - 11) / 2 + 1 in
   let num_three_digit_odd := (999 - 101) / 2 + 1 in
   let num_four_digit_odd := (2999 - 1001) / 2 + 1 in
   let total_digits := num_one_digit_odd * 1 + 
                       num_two_digit_odd * 2 + 
                       num_three_digit_odd * 3 + 
                       num_four_digit_odd * 4 in
   total_digits) = 5445 :=
by
  -- Proof omitted
  sorry

end num_digits_first_1500_odd_integers_l548_548455


namespace power_difference_of_squares_l548_548833

theorem power_difference_of_squares : (((7^2 - 3^2) : ℤ)^4) = 2560000 := by
  sorry

end power_difference_of_squares_l548_548833


namespace real_solution_unique_l548_548197

theorem real_solution_unique (x : ℝ) (h : x^4 + (2 - x)^4 + 2 * x = 34) : x = 0 :=
sorry

end real_solution_unique_l548_548197


namespace evaluate_expression_l548_548915

theorem evaluate_expression :
  3 + 2*Real.sqrt 3 + 1/(3 + 2*Real.sqrt 3) + 1/(2*Real.sqrt 3 - 3) = 3 + (16 * Real.sqrt 3) / 3 :=
by
  sorry

end evaluate_expression_l548_548915


namespace trailing_zeroes_mod_100_l548_548691

open Nat

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeroes (n : ℕ) : ℕ :=
  (if n = 0 then 0 else logFactMultiples 5 (n / 5)) where
    logFactMultiples (p m : ℕ) : ℕ := if m = 0 then 0 else m + logFactMultiples p (m / p)

theorem trailing_zeroes_mod_100 :
  let M := (List.range 1 (51)).map factorial |>.foldr (fun x acc => trailing_zeroes x + acc) 0
  in M % 100 = 14 :=
by { sorry }

end trailing_zeroes_mod_100_l548_548691


namespace plywood_perimeter_difference_l548_548081

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548081


namespace discount_equation_l548_548911

theorem discount_equation (x : ℝ) : 280 * (1 - x) ^ 2 = 177 := 
by 
  sorry

end discount_equation_l548_548911


namespace trig_identity_equiv_l548_548220

theorem trig_identity_equiv (α : ℝ) (h : Real.sin (Real.pi - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 :=
by
  sorry

end trig_identity_equiv_l548_548220


namespace polynomial_divisible_by_x_minus_6_l548_548058

theorem polynomial_divisible_by_x_minus_6 (m : ℤ) :
  let h : ℤ[X] := X^3 - 2*X^2 - (m^2 + 2*m)*X + 3*m^2 + 6*m + 3
  m = 7 ∨ m = -7 ↔ h.eval 6 = 0 :=
by
  sorry

end polynomial_divisible_by_x_minus_6_l548_548058


namespace sec_150_eq_neg_two_sqrt_three_div_three_l548_548919

theorem sec_150_eq_neg_two_sqrt_three_div_three : sec 150 = - (2 * sqrt 3 / 3) :=
by sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l548_548919


namespace right_angled_triangle_setC_l548_548909

-- Definitions of the given sets
def setA := (6, 15, 17)
def setB := (7, 12, 15)
def setC := (7, 24, 25)
def setD := (13, 15, 20)

-- Theorem stating that setC represents the sides of a right-angled triangle
theorem right_angled_triangle_setC : 
  let (a, b, c) := setC in a^2 + b^2 = c^2 :=
by
  let (a, b, c) := setC
  sorry

end right_angled_triangle_setC_l548_548909


namespace math_problem_l548_548519

noncomputable def a : ℝ := 2023 - Real.pi
noncomputable def b : ℝ := (1 / 2) ^ (-2)
noncomputable def c : ℝ := |1 - Real.sqrt 2|
noncomputable def d : ℝ := 2 * (Real.sqrt 2 / 2)

theorem math_problem :
  a^0 + b + c - d = 4 := by
  sorry

end math_problem_l548_548519


namespace urn_problem_l548_548460

noncomputable def probability_of_two_black_balls : ℚ := (10 / 15) * (9 / 14)

theorem urn_problem : probability_of_two_black_balls = 3 / 7 := 
by
  sorry

end urn_problem_l548_548460


namespace simplest_radical_expression_l548_548902

theorem simplest_radical_expression :
  let A := Real.sqrt 3
  let B := Real.sqrt 4
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 2)
  B = 2 :=
by
  sorry

end simplest_radical_expression_l548_548902


namespace tunnel_length_l548_548504

noncomputable def train_speed_mph : ℝ := 75
noncomputable def train_length_miles : ℝ := 1 / 4
noncomputable def passing_time_minutes : ℝ := 3

theorem tunnel_length :
  let speed_mpm := train_speed_mph / 60
  let total_distance_traveled := speed_mpm * passing_time_minutes
  let tunnel_length := total_distance_traveled - train_length_miles
  tunnel_length = 3.5 :=
by
  sorry

end tunnel_length_l548_548504


namespace s_1000_l548_548848

def s : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := if n % 2 = 0 then s (n / 2)
       else if n % 4 == 1 then s ((n - 1) / 2)
       else s ((n + 1) / 4) + (s ((n + 1) / 4))^2 / s (((n + 1) / 4 - 1) / 2)

theorem s_1000 : s 1000 = 720 :=
by
  sorry

end s_1000_l548_548848


namespace area_convex_quadrilateral_eq_l548_548658

noncomputable def area_of_convex_quadrilateral (a b : ℝ) : ℝ :=
  (b^2 - a^2) / 4

theorem area_convex_quadrilateral_eq (a b : ℝ) (convex_quadrilateral : convex_quadrilateral_condition a b) :
  convex_quadrilateral.area = area_of_convex_quadrilateral a b :=
by
  sorry

-- Predicate to describe the convex quadrilateral condition
def convex_quadrilateral_condition (a b : ℝ) : Prop :=
  ∃ (q : quadrilateral),
    q.opposite_sides_equal_and_perpendicular ∧
    q.other_sides_eq a b

end area_convex_quadrilateral_eq_l548_548658


namespace find_standard_eq_of_circle_l548_548295

-- Definitions of conditions
def radius (C : Type) [MetricSpace C] (r : ℝ) : Prop :=
∀ (p : C), dist p (0 : C) = r

def in_first_quadrant (p : ℝ × ℝ) : Prop :=
p.1 > 0 ∧ p.2 > 0

def tangent_to_line (C : Type) [MetricSpace C] (line : ℝ → ℝ) : Prop :=
∃ (p : C), (line p.1 = p.2) ∧ (dist p (0 : C) = 1)

def tangent_to_x_axis (p : ℝ × ℝ) : Prop :=
p.2 = 1

-- Statement to prove the standard equation of the circle
theorem find_standard_eq_of_circle :
  ∃ (a b : ℝ), radius (ℝ × ℝ) 1 ∧ in_first_quadrant (a, b) ∧ tangent_to_line (ℝ × ℝ) (λ x, (4 / 3) * x) ∧ tangent_to_x_axis (a, b) →
  ((a = 2) ∧ (b = 1) ∧ ∀ (x y : ℝ), ((x - 2)^2 + (y - 1)^2 = 1)) :=
sorry

end find_standard_eq_of_circle_l548_548295


namespace triangle_evaluation_l548_548345

def triangle (a b : ℤ) : ℤ := a^2 - 2 * b

theorem triangle_evaluation : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end triangle_evaluation_l548_548345


namespace find_f_of_7_over_3_l548_548347

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the odd function f

-- Hypothesis: f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Hypothesis: f(1 + x) = f(-x) for all x in ℝ
axiom functional_equation (x : ℝ) : f (1 + x) = f (-x)

-- Hypothesis: f(-1/3) = 1/3
axiom initial_condition : f (-1 / 3) = 1 / 3

-- The statement we need to prove
theorem find_f_of_7_over_3 : f (7 / 3) = - (1 / 3) :=
by
  sorry -- Proof to be provided

end find_f_of_7_over_3_l548_548347


namespace problem_solution_l548_548240

variable (U : Set Real) (a b : Real) (t : Real)
variable (A B : Set Real)

-- Conditions
def condition1 : U = Set.univ := sorry

def condition2 : ∀ x, a ≠ 0 → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a := sorry

def condition3 : a > b := sorry

def condition4 : t = (a^2 + b^2) / (a - b) := sorry

def condition5 : ∀ m, (∀ x, |x + 1| - |x - 3| ≤ m^2 - 3 * m) → m ∈ B := sorry

-- To Prove
theorem problem_solution : A ∩ (Set.univ \ B) = {m : Real | 2 * Real.sqrt 2 ≤ m ∧ m < 4} := sorry

end problem_solution_l548_548240


namespace number_of_six_digit_numbers_with_1_at_beginning_number_of_grouped_six_digit_numbers_number_of_non_adjacent_even_six_digit_numbers_l548_548534

/-- The number of distinct 6-digit numbers that can be formed 
  using any three even and three odd numbers from the digits 1 to 9, 
  where 1 is at the beginning, equals 2880. -/
theorem number_of_six_digit_numbers_with_1_at_beginning :
  let evens := {2, 4, 6, 8},
      odds := {3, 5, 7, 9},
      ways_to_choose_evens := (evens.choose 3).card,
      ways_to_choose_odds := (odds.choose 2).card,
      ways_to_permute := (Finset.univ : Finset (Fin 5)).card.factorial
  in ways_to_choose_evens * ways_to_choose_odds * ways_to_permute = 2880 := sorry

/-- The number of distinct 6-digit numbers that can be formed 
  using any three even and three odd numbers from the digits 1 to 9 
  with evens and odds grouped together, equals 2880. -/
theorem number_of_grouped_six_digit_numbers :
  let evens := {2, 4, 6, 8},
      odds := {1, 3, 5, 7, 9},
      ways_to_choose_and_permute_evens := (evens.choose 3).card * 3.factorial,
      ways_to_choose_and_permute_odds := (odds.choose 3).card * 3.factorial,
      ways_to_arrange_groups := (Finset.univ : Finset (Fin 2)).card.factorial
  in ways_to_arrange_groups * ways_to_choose_and_permute_evens * ways_to_choose_and_permute_odds = 2880 := sorry

/-- The number of distinct 6-digit numbers that can be formed 
  using any three even and three odd numbers from the digits 1 to 9
  where no two even digits are adjacent, equals 57600. -/
theorem number_of_non_adjacent_even_six_digit_numbers :
  let evens := {2, 4, 6, 8},
      odds := {1, 3, 5, 7, 9},
      ways_to_choose_and_arrange_odds := (Finset.choose_fin 5 3).card * (Finset.univ : Finset (Fin 3)).card.factorial,
      ways_to_choose_and_arrange_evens_in_4_slots := (Finset.choose_fin 4 3).card * 3.factorial
  in ways_to_choose_and_arrange_odds * ways_to_choose_and_arrange_evens_in_4_slots = 57600 := sorry

end number_of_six_digit_numbers_with_1_at_beginning_number_of_grouped_six_digit_numbers_number_of_non_adjacent_even_six_digit_numbers_l548_548534


namespace modulus_of_z_l548_548878

noncomputable def z : ℂ := (1 + complex.i) / (1 - complex.i)

theorem modulus_of_z : complex.abs z = 1 := sorry

end modulus_of_z_l548_548878


namespace larry_channels_l548_548683

theorem larry_channels : (initial_channels : ℕ) 
                         (channels_taken : ℕ) 
                         (channels_replaced : ℕ) 
                         (reduce_channels : ℕ) 
                         (sports_package : ℕ) 
                         (supreme_sports_package : ℕ) 
                         (final_channels : ℕ)
                         (h1 : initial_channels = 150)
                         (h2 : channels_taken = 20)
                         (h3 : channels_replaced = 12)
                         (h4 : reduce_channels = 10)
                         (h5 : sports_package = 8)
                         (h6 : supreme_sports_package = 7)
                         (h7 : final_channels = initial_channels - channels_taken + channels_replaced - reduce_channels + sports_package + supreme_sports_package)
                         : final_channels = 147 :=
by sorry

end larry_channels_l548_548683


namespace clean_per_hour_l548_548887

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l548_548887


namespace library_students_l548_548496

theorem library_students (total_books : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day3 = 6 →
  let books_used_day1 := students_day1 * books_per_student in
  let books_used_day2 := students_day2 * books_per_student in
  let books_used_day3 := students_day3 * books_per_student in
  let total_books_used := books_used_day1 + books_used_day2 + books_used_day3 in
  let remaining_books := total_books - total_books_used in
  remaining_books / books_per_student = 9 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end library_students_l548_548496


namespace largest_cos_x_l548_548351

-- Define the conditions
variables {x y z : ℝ}
hypothesis h1 : sin x = 1 / cos y
hypothesis h2 : sin y = 1 / cos z
hypothesis h3 : sin z = 1 / cos x

-- State what needs to be proven
theorem largest_cos_x : cos x = 0 := 
sorry

end largest_cos_x_l548_548351


namespace not_A_probability_l548_548910

variable (A : Prop)
variable (P : Prop → ℝ)

axiom PA : P A = 0.65

theorem not_A_probability : P (¬ A) = 0.35 := by
  have h : P (¬ A) = 1 - P A := sorry
  rw [PA] at h
  exact h

end not_A_probability_l548_548910


namespace exists_many_omopeiro_numbers_l548_548148

-- Definition of an omopeiro number
def is_omopeiro (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℤ), (∀ i, a i ≠ 0) ∧ (∑ i, (a i)^2 = 2021)

-- The theorem statement to prove the existence of at least 2019 omopeiro numbers
theorem exists_many_omopeiro_numbers : ∃ n, n > 1500 ∧ is_omopeiro n :=
by
  sorry

end exists_many_omopeiro_numbers_l548_548148


namespace option_c_has_minimum_value_4_l548_548009

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548009


namespace a_n_correct_l548_548431

noncomputable def S_n (n : ℕ) (_h : 0 < n) : ℝ := 2 * n - a_n n _h
noncomputable def a_n (n : ℕ) (_h : 0 < n) : ℝ := (2^n - 1) / (2^(n-1))

theorem a_n_correct (n : ℕ) (h : 0 < n) : a_n n h = (2^n - 1) / (2^(n-1)) := 
sorry

#check S_n
#check a_n
#check a_n_correct

end a_n_correct_l548_548431


namespace plywood_perimeter_difference_l548_548086

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548086


namespace part1_part2_l548_548233

variables {a : ℕ → ℕ}
variables {b : ℕ → ℝ}
variables {c : ℕ → ℝ}

-- Condition definitions
definition S : ℕ → ℝ := λ n, (n * (n + 1)) / 2
definition S_cond := ∀ n, S (n + 1) / S n = (n + 2) / n
def a_cond := a 1 = 1 ∧ ∀ n, a (n + 1) = n + 1
def b_cond := b 2 = 2 ∧ b 1 * b 2 * b 3 * b 4 * b 5 = 2 ^ 10
def c_def := ∀ n, c n = (2 + a n) / (a n * a (n + 1) * b (n + 1))

-- The first part: finding the general formula for {a_n}
theorem part1 (a_cond : a_cond) : ∀ n, a n = n := by
  sorry

-- The second part: proving c_1 + c_2 + ... + c_n < 1
theorem part2 (a_cond : a_cond) (b_cond : b_cond) (c_def : c_def) : ∀ n, (∑ i in finset.range (n + 1), c i) < 1 := by
  sorry

end part1_part2_l548_548233


namespace plywood_perimeter_difference_l548_548112

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548112


namespace focus_of_parabola_y_eq_4x2_l548_548407

theorem focus_of_parabola_y_eq_4x2 :
  let p := (1 / 4) * (1 / 4) in
  (0, p) = (0, 1 / 16) :=
by
  let p := (1 / 4) * (1 / 4)
  have : p = 1 / 16 := by sorry
  rw [this]
  rfl

end focus_of_parabola_y_eq_4x2_l548_548407


namespace fred_likes_12_pairs_of_digits_l548_548569

theorem fred_likes_12_pairs_of_digits :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs ↔ ∃ (n : ℕ), n < 100 ∧ n % 8 = 0 ∧ n = 10 * a + b) ∧
    pairs.card = 12) :=
by
  sorry

end fred_likes_12_pairs_of_digits_l548_548569


namespace real_number_a_l548_548636

theorem real_number_a (a : ℝ) (h : a ∈ {a^2 - a, 0}) : a = 2 :=
sorry

end real_number_a_l548_548636


namespace trapezium_other_side_length_l548_548201

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l548_548201


namespace find_distance_between_sides_of_trapezium_l548_548550

variable (side1 side2 h area : ℝ)
variable (h1 : side1 = 20)
variable (h2 : side2 = 18)
variable (h3 : area = 228)
variable (trapezium_area : area = (1 / 2) * (side1 + side2) * h)

theorem find_distance_between_sides_of_trapezium : h = 12 := by
  sorry

end find_distance_between_sides_of_trapezium_l548_548550


namespace blue_shoes_in_warehouse_l548_548437

theorem blue_shoes_in_warehouse (total blue purple green : ℕ) (h1 : total = 1250) (h2 : green = purple) (h3 : purple = 355) :
    blue = total - (green + purple) := by
  sorry

end blue_shoes_in_warehouse_l548_548437


namespace sequence_a1_sum_l548_548585

theorem sequence_a1_sum (k : ℚ) (a : ℕ → ℚ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = k * a n + 3 * k - 3) 
(h2 : k ≠ 0) (h3 : k ≠ 1) 
(h4 : ∀ i ∈ {2, 3, 4, 5}, a i ∈ {-678, -78, -3, 22, 222, 2222}) : 
  ∃ a1_values : Finset ℚ, (a 1) ∈ a1_values ∧ a1_values.sum id = 6023 / 3 :=
sorry

end sequence_a1_sum_l548_548585


namespace range_of_f_l548_548933

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * Real.sin x + 6

theorem range_of_f :
  ∀ (x : ℝ), Real.sin x ≠ 2 → 
  (1 ≤ f x ∧ f x ≤ 11) :=
by 
  sorry

end range_of_f_l548_548933


namespace roger_owes_correct_amount_l548_548723

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l548_548723


namespace A_left_after_3_days_l548_548817

def work_done_by_A_and_B_together (x : ℕ) : ℚ :=
  (1 / 21) * x + (1 / 28) * x

def work_done_by_B_alone (days : ℕ) : ℚ :=
  (1 / 28) * days

def total_work_done (x days_b_alone : ℕ) : ℚ :=
  work_done_by_A_and_B_together x + work_done_by_B_alone days_b_alone

theorem A_left_after_3_days :
  ∀ (x : ℕ), total_work_done x 21 = 1 ↔ x = 3 := by
  sorry

end A_left_after_3_days_l548_548817


namespace A_beats_B_by_54_83_meters_l548_548660

noncomputable def distance_A_beats_B (distance_A : ℝ) (time_A : ℝ) (time_difference : ℝ) : ℝ :=
  let speed_A := distance_A / time_A
  speed_A * time_difference

theorem A_beats_B_by_54_83_meters :
  distance_A_beats_B 1000 328.15384615384613 18 ≈ 54.83 :=
by
  sorry

end A_beats_B_by_54_83_meters_l548_548660


namespace line_tangent_to_circle_l548_548528

-- Define the parametric form of the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 1 - 2 * t)

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 + 2 * ρ * Real.cos θ - 2 * ρ * Real.sin θ = 0

-- Convert circle_C to Cartesian form
def circle_C_cartesian (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Define the distance between a point and a line
def distance_point_to_line (P : ℝ × ℝ) (A B : ℝ) (C : ℝ) : ℝ :=
  (abs (A * P.1 + B * P.2 + C)) / (sqrt (A^2 + B^2))

-- Check if line l is tangent to circle C
theorem line_tangent_to_circle :
  ∀ t : ℝ, 
    let P := (-1 : ℝ, 1 : ℝ) in
    let A := 1 in
    let B := 1 in
    let C := -2 in
    distance_point_to_line P A B C = sqrt 2 →
    circle_C_cartesian l_Cqrt :=
  sorry

end line_tangent_to_circle_l548_548528


namespace sqrt_expr_equals_sum_l548_548545

theorem sqrt_expr_equals_sum :
  ∃ x y z : ℤ,
    (x + y * Int.sqrt z = Real.sqrt (77 + 28 * Real.sqrt 3)) ∧
    (x^2 + y^2 * z = 77) ∧
    (2 * x * y = 28) ∧
    (x + y + z = 16) :=
by
  sorry

end sqrt_expr_equals_sum_l548_548545


namespace bus_departure_interval_l548_548447

theorem bus_departure_interval
  (v : ℝ) -- speed of B (per minute)
  (t_A : ℝ := 10) -- A is overtaken every 10 minutes
  (t_B : ℝ := 6) -- B is overtaken every 6 minutes
  (v_A : ℝ := 3 * v) -- speed of A
  (d_A : ℝ := v_A * t_A) -- distance covered by A in 10 minutes
  (d_B : ℝ := v * t_B) -- distance covered by B in 6 minutes
  (v_bus_minus_vA : ℝ := d_A / t_A) -- bus speed relative to A
  (v_bus_minus_vB : ℝ := d_B / t_B) -- bus speed relative to B) :
  (t : ℝ) -- time interval between bus departures
  : t = 5 := sorry

end bus_departure_interval_l548_548447


namespace sum_of_b_for_rational_roots_l548_548950

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l548_548950


namespace largest_shaded_area_of_figures_l548_548825

theorem largest_shaded_area_of_figures :
  let square_area (s : ℝ) := s * s
  let circle_area (r : ℝ) := real.pi * r * r
  
  let shaded_area_A := square_area 3 - circle_area (3 / 2)
  let shaded_area_B := (3 * 4) / 2 - (1 / 4) * real.pi * (2 * 2)
  let shaded_area_C := square_area 2 - circle_area 1
  shaded_area_B > shaded_area_A ∧ shaded_area_B > shaded_area_C := 
by
  -- Proof is not required, so we use sorry
  sorry

end largest_shaded_area_of_figures_l548_548825


namespace compound_proposition_l548_548360

theorem compound_proposition (Sn P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → Sn n = 2 * n^2 + 3 * n + 1) →
  (∀ n : ℕ, n > 0 → Sn n = 2 * P n + 1) →
  (¬(∀ n, n > 0 → ∃ d, (P (n + 1) - P n) = d)) ∧ (∀ n, n > 0 → P n = Q (n - 1)) :=
by
  sorry

end compound_proposition_l548_548360


namespace necessary_but_not_sufficient_condition_l548_548052

variable (a b : ℝ) (lna lnb : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : lna < lnb) (h2 : lna = Real.log a) (h3 : lnb = Real.log b) :
  (a > 0 ∧ b > 0 ∧ a < b ∧ a ^ 3 < b ^ 3) ∧ ¬(a ^ 3 < b ^ 3 → 0 < a ∧ a < b ∧ 0 < b) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l548_548052


namespace product_of_roots_l548_548893

-- Define the polynomial
def polynomial : Polynomial ℝ := Polynomial.monomial 3 4 - Polynomial.monomial 2 2 + Polynomial.monomial 1 (-30) + Polynomial.C 36

-- Define Vieta's formula product of roots for the given polynomial
theorem product_of_roots : (polynomial.coeff 3 ≠ 0) → 
  polynomial.coeff 0 / polynomial.coeff 3 = -9 := 
by 
  sorry

end product_of_roots_l548_548893


namespace average_of_possible_values_of_x_is_0_l548_548642

noncomputable def average_of_possible_values_of_x : ℝ :=
  let S := {x : ℝ | sqrt (3 * x^2 + 7) = sqrt 31} in
  if S = ∅ then 0 else (S.sum id) / S.card

theorem average_of_possible_values_of_x_is_0 
  (h : sqrt (3 * x^2 + 7) = sqrt 31) : 
  average_of_possible_values_of_x = 0 := 
by
  sorry

end average_of_possible_values_of_x_is_0_l548_548642


namespace problem_statement_l548_548965

noncomputable def zeta (x : ℝ) : ℝ := ∑' n : ℕ, 1 / n ^ x

lemma zeta_pos_real (x : ℝ) (hx : x > 1) : zeta x = ∑' n : ℕ, 1 / n ^ x :=
  by sorry

theorem problem_statement : (∀ x : ℝ, x > 1 → zeta x = ∑ n : ℕ, 1 / n ^ x) →
  ∑ k in (finset.range (∞)).filter (λ k, k ≥ 3), real.fract (zeta (3 * k - 2)) = 0 :=
  by sorry

end problem_statement_l548_548965


namespace min_value_of_f_l548_548607

open Real

def f (x : ℝ) : ℝ := 2 * cos x + sin (2 * x)

theorem min_value_of_f : ∃ x : ℝ, f x = -3 * sqrt 3 / 2 :=
by {
  -- Proof is omitted
  sorry
}

end min_value_of_f_l548_548607


namespace find_f_values_l548_548704

def func_property1 (f : ℕ → ℕ) : Prop := 
  ∀ a b : ℕ, a ≠ b → a * f a + b * f b > a * f b + b * f a

def func_property2 (f : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_values (f : ℕ → ℕ) (h1 : func_property1 f) (h2 : func_property2 f) : 
  f 1 + f 6 + f 28 = 66 :=
sorry

end find_f_values_l548_548704


namespace alicia_tax_deduction_l548_548163

theorem alicia_tax_deduction :
  let hourly_wage_dollars := 20
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 0.0145
  let tax_deduction_cents := tax_rate * hourly_wage_cents
  in tax_deduction_cents = 29 := by
sorry

end alicia_tax_deduction_l548_548163


namespace custom_operation_example_l548_548988

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l548_548988


namespace ratio_arithmetic_geometric_mean_l548_548905

/-- Let a and b be positive real numbers. Given the ratio of the arithmetic mean to
  the geometric mean of a and b is 25:24, prove that the ratio of a to b is 16:9 or 9:16. -/
theorem ratio_arithmetic_geometric_mean (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) / (2 * real.sqrt (a * b)) = 25 / 24) : 
  a / b = 16 / 9 ∨ a / b  = 9 / 16 :=
by
  sorry

end ratio_arithmetic_geometric_mean_l548_548905


namespace simplify_fraction_l548_548560

open Nat

noncomputable def a_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), 1 / (nat.choose n k)
noncomputable def b_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), k^2 / (nat.choose n k)

theorem simplify_fraction (n : ℕ) (hn : 0 < n) : 
  (a_n n) / (b_n n) = 1 / (n * (n - 1)) :=
sorry

end simplify_fraction_l548_548560


namespace women_in_luxury_compartment_l548_548726

def passengers : ℕ := 300
def percentage_women : ℝ := 0.70
def percentage_luxury_compartment : ℝ := 0.15

def number_of_women := passengers * percentage_women
def number_of_women_in_luxury_compartment := number_of_women * percentage_luxury_compartment

theorem women_in_luxury_compartment :
  (number_of_women_in_luxury_compartment).round = 32 := by
  sorry

end women_in_luxury_compartment_l548_548726


namespace best_calories_per_dollar_l548_548333

noncomputable def calories_per_dollar (calories : ℕ) (cost : ℕ) : ℚ :=
  calories / cost

-- Define the calorie and cost information
def burritos := (10 * 120, 6)
def burgers := (5 * 400, 8)
def pizza := (8 * 300, 10)
def donuts := (15 * 250, 12)

-- Calculate calories per dollar for each option
def cpd_burritos := calories_per_dollar 1200 6
def cpd_burgers := calories_per_dollar 2000 8
def cpd_pizza := calories_per_dollar 2400 10
def cpd_donuts := calories_per_dollar 3750 12

-- Theorem to prove which option is best and by how much it surpasses the second-best
theorem best_calories_per_dollar :
  cpd_donuts = 312.5 ∧ cpd_donuts - cpd_burgers = 62.5 :=
by
  sorry

end best_calories_per_dollar_l548_548333


namespace triangle_properties_proof_l548_548652

noncomputable def triangle_proporties : Prop :=
  ∀ (a b c : ℝ) (A B C : ℝ),
    b = sqrt 3 →
    c = 1 →
    B = 60 * real.pi / 180 →
    a = sqrt (b^2 + c^2) ∧
    A = 90 * real.pi / 180 ∧
    C = 30 * real.pi / 180

theorem triangle_properties_proof : triangle_proporties :=
  by {
    intros a b c A B C hb hc hB,
    have ha : a = sqrt (b^2 + c^2), by {
      dsimp only [hb, hc],
      norm_num,
    },
    have hA : A = 90 * real.pi / 180, by {
      obtain rfl : B = real.pi / 3 := by { exact hB, norm_num },
      obtain rfl : b = sqrt 3 := by { exact hb, norm_num },
      obtain rfl : c = 1 := by { exact hc, norm_num },
      calc
        A = π - (B + C)   : by sorry
        ... = π / 2       : by sorry,
    },
    have hC : C = 30 * real.pi / 180, by {
      obtain rfl : B = real.pi / 3 := by { exact hB, norm_num },
      obtain rfl : b = sqrt 3 := by { exact hb, norm_num },
      obtain rfl : c = 1 := by { exact hc, norm_num },
      calc
        C = real.asin (c * real.sin B / b) :
        ... = real.pi / 6 :
        sorry,
    },
    exact ⟨ha, hA, hC⟩,
  }

end triangle_properties_proof_l548_548652


namespace maximum_illuminated_surfaces_l548_548146

noncomputable def optimal_position (r R d : ℝ) (h : d > r + R) : ℝ :=
  d / (1 + Real.sqrt (R^3 / r^3))

theorem maximum_illuminated_surfaces (r R d : ℝ) (h : d > r + R) (h1 : r ≤ optimal_position r R d h) (h2 : optimal_position r R d h ≤ d - R) :
  (optimal_position r R d h = d / (1 + Real.sqrt (R^3 / r^3))) ∨ (optimal_position r R d h = r) :=
sorry

end maximum_illuminated_surfaces_l548_548146


namespace total_cost_of_fencing_l548_548202

noncomputable def π : ℝ := real.pi

noncomputable def calculate_total_cost (d : ℝ) (rate : ℝ) : ℝ :=
  let C := π * d
  let total_cost := rate * C
  total_cost

theorem total_cost_of_fencing :
  calculate_total_cost 18 2.50 ≈ 141.38 :=
by
  let C := π * 18
  let tc := 2.50 * C
  exact real.rat_cast tc ≈ 141.38

end total_cost_of_fencing_l548_548202


namespace length_of_train_l548_548860

-- Definitions used in the original problem:
variables (L : ℝ) (V : ℝ) (t_pole : ℝ := 18) (t_platform : ℝ := 39) (platform_length : ℝ := 350)

-- Conditions from the problem:
def speed_of_train := L / t_pole
def time_cross_platform := (L + platform_length) / V

-- Correct answer to be proved:
theorem length_of_train : L = 300 := by
  -- inferring speed from crossing time of the signal pole
  let V := L / t_pole
  -- equating total distance covered when crossing platform to time taken
  have h1 : L + platform_length = V * t_platform,
  -- substituting calculated speed (V)
  have h2 : L + platform_length = (L / t_pole) * t_platform,
  -- solving for L
  sorry

end length_of_train_l548_548860


namespace fertilizers_total_weight_l548_548823

theorem fertilizers_total_weight (field_area : ℝ) (app_rate_A_weight : ℝ) (app_rate_A_area : ℝ) (app_rate_B_weight : ℝ) (app_rate_B_area : ℝ) (application_area : ℝ)
  (h_field : field_area = 10800)
  (h_app_rate_A : app_rate_A_weight = 150 ∧ app_rate_A_area = 3000)
  (h_app_rate_B : app_rate_B_weight = 180 ∧ app_rate_B_area = 4000)
  (h_application_area : application_area = 3600) :
  let weight_A := app_rate_A_weight / app_rate_A_area * application_area
      weight_B := app_rate_B_weight / app_rate_B_area * application_area
  in weight_A + weight_B = 342 :=
by
  sorry

end fertilizers_total_weight_l548_548823


namespace least_number_of_entries_to_alter_l548_548999

/-- 
Proving that given the initial 4x4 array with equal row and column sums, 
the least number of entries that must be altered to make all row sums, 
column sums, and both main diagonal sums different from one another is 4.
-/
theorem least_number_of_entries_to_alter :
  ∀ (matrix : Array (Array Int)),
    (matrix = #[#[1, 3, 5, 7], #[4, 6, 8, 10], #[2, 4, 6, 8], #[5, 7, 9, 11]]) →
    (∀ i j, (i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3) → (j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3) →
      (Array.sum (matrix[i])) = (Array.sum (matrix[j]))) →
    (4 = 
      findMinCount 
        (λ m, 
          distinct (rowSums m) ∧ 
          distinct (colSums m) ∧ 
          distinct (diagSums m)))
  :=
begin
  intros matrix h_matrix h_sums,
  sorry
end


end least_number_of_entries_to_alter_l548_548999


namespace rose_work_days_l548_548331

variable (R : ℕ)

-- Defining the work rates as given conditions
def john_work_rate := 1 / 320
def rose_work_rate := 1 / (R : ℝ)
def combined_work_rate := 1 / 192

-- The Lean theorem for the problem
theorem rose_work_days :
  (john_work_rate + rose_work_rate) = combined_work_rate → R = 384 := by
  sorry

end rose_work_days_l548_548331


namespace a_eq_3x_or_neg2x_l548_548635

theorem a_eq_3x_or_neg2x (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 19 * x^3) (h3 : a - b = x) :
    a = 3 * x ∨ a = -2 * x :=
by
  -- The proof will go here
  sorry

end a_eq_3x_or_neg2x_l548_548635


namespace number_of_disconnected_regions_l548_548439

theorem number_of_disconnected_regions (n : ℕ) (h : 2 ≤ n) : 
  ∀ R : ℕ → ℕ, (R 1 = 2) → 
  (∀ k, R k = k^2 - k + 2 → R (k + 1) = (k + 1)^2 - (k + 1) + 2) → 
  R n = n^2 - n + 2 :=
sorry

end number_of_disconnected_regions_l548_548439


namespace remainder_97_pow_103_mul_7_mod_17_l548_548453

theorem remainder_97_pow_103_mul_7_mod_17 :
  (97 ^ 103 * 7) % 17 = 13 := by
  have h1 : 97 % 17 = -3 % 17 := by sorry
  have h2 : 9 % 17 = -8 % 17 := by sorry
  have h3 : 64 % 17 = 13 % 17 := by sorry
  have h4 : -21 % 17 = 13 % 17 := by sorry
  sorry

end remainder_97_pow_103_mul_7_mod_17_l548_548453


namespace find_c_and_d_l548_548342

theorem find_c_and_d (x y z : ℝ) (h1 : log 10 (2 * x + 3 * y) = z)
    (h2 : log 10 (x^2 + 2 * y^2) = z + 2) :
    ∃ c d : ℝ, (∀ a b c : ℝ, x^2 * y + x * y^2 = c * 10^(4 * z) + d * 10^(3 * z)) ∧ (c + d = 1 / 12) :=
by {
  sorry
}

end find_c_and_d_l548_548342


namespace car_average_speed_l548_548474

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l548_548474


namespace find_extrema_l548_548747

noncomputable def f (x : ℝ) : ℝ := (4 * x) / (x ^ 2 + 1)

theorem find_extrema :
  ∃ (a b : ℝ), (∀ x ∈ Set.Icc (-2:ℝ) 2, f x ≤ a) ∧ 
               (∃ x ∈ Set.Icc (-2:ℝ) 2, f x = a) ∧
               (∀ x ∈ Set.Icc (-2:ℝ) 2, b ≤ f x) ∧
               (∃ x ∈ Set.Icc (-2:ℝ) 2, f x = b) ∧
               a = 2 ∧ b = -(8 / 5) := 
by
  use [2, -(8 / 5)]
  sorry

end find_extrema_l548_548747


namespace tenth_number_in_sequence_is_803_l548_548754

-- Definitions required for conditions
def seq : ℕ → ℕ
| 1  := 11 
| 2  := 23
| 3  := 47 
| 4  := 83 
| 5  := 131 
| 6  := 191 
| 7  := 263 
| 8  := 347 
| 9  := 443 
| 10 := 551
| 11 := 671
| n  := seq (n - 1) + (seq (n - 1) - seq (n - 2) + 12)

-- Theorem statement to prove
theorem tenth_number_in_sequence_is_803 :
  seq 12 = 803 :=
by
  sorry

end tenth_number_in_sequence_is_803_l548_548754


namespace area_of_region_l548_548780

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 11 = 0) -> 
  ∃ (A : ℝ), A = 24 * Real.pi :=
by 
  sorry

end area_of_region_l548_548780


namespace polynomial_remainder_l548_548207

theorem polynomial_remainder :
  let p := (x^6 - 2*x^5 + x^4 - x^2 + 3*x - 1)
  let d := (x^2 - 1) * (x + 2)
  let r := (7 / 3 * x^2 + x - 7 / 3)
  ∃ q : ℚ[X], p = d * q + r := 
  sorry

end polynomial_remainder_l548_548207


namespace plywood_perimeter_difference_l548_548093

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548093


namespace part1_part2_l548_548603

-- Define the function f(x) = |x - 1| + |x - 2|
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

-- Prove the statement about f(x) and the inequality
theorem part1 : { x : ℝ | (2 / 3) ≤ x ∧ x ≤ 4 } ⊆ { x : ℝ | f x ≤ x + 1 } :=
sorry

-- State k = 1 as the minimum value of f(x)
def k : ℝ := 1

-- Prove the non-existence of positive a and b satisfying the given conditions
theorem part2 : ¬ ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = k ∧ (1 / a + 2 / b = 4) :=
sorry

end part1_part2_l548_548603


namespace kolya_advantageous_methods_l548_548466

-- Define the context and conditions
variables (n : ℕ) (h₀ : n ≥ 2)
variables (a b : ℕ) (h₁ : a + b = 2*n + 1) (h₂ : a ≥ 2) (h₃ : b ≥ 2)

-- Define outcomes of the methods
def method1_outcome (a b : ℕ) := max a b + min (a - 1) (b - 1)
def method2_outcome (a b : ℕ) := min a b + min (a - 1) (b - 1)
def method3_outcome (a b : ℕ) := max (method1_outcome a b - 1) (method2_outcome a b - 1)

-- Prove which methods are the most and least advantageous
theorem kolya_advantageous_methods :
  method1_outcome a b >= method2_outcome a b ∧ method1_outcome a b >= method3_outcome a b :=
sorry

end kolya_advantageous_methods_l548_548466


namespace distance_by_land_l548_548864

theorem distance_by_land (distance_by_sea total_distance distance_by_land : ℕ)
  (h1 : total_distance = 601)
  (h2 : distance_by_sea = 150)
  (h3 : total_distance = distance_by_land + distance_by_sea) : distance_by_land = 451 := by
  sorry

end distance_by_land_l548_548864


namespace percentage_subtraction_l548_548750

variable (a b x m : ℝ) (p : ℝ)

-- Conditions extracted from the problem.
def ratio_a_to_b : Prop := a / b = 4 / 5
def definition_of_x : Prop := x = 1.75 * a
def definition_of_m : Prop := m = b * (1 - p / 100)
def value_m_div_x : Prop := m / x = 0.14285714285714285

-- The proof problem in the form of a Lean statement.
theorem percentage_subtraction 
  (h1 : ratio_a_to_b a b)
  (h2 : definition_of_x a x)
  (h3 : definition_of_m b m p)
  (h4 : value_m_div_x x m) : p = 80 := 
sorry

end percentage_subtraction_l548_548750


namespace optionC_has_min_4_l548_548002

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548002


namespace probability_prime_sum_is_correct_l548_548040

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def cube_rolls_prob_prime_sum : ℚ :=
  let possible_outcomes := 36
  let prime_sums_count := 15
  prime_sums_count / possible_outcomes

theorem probability_prime_sum_is_correct :
  cube_rolls_prob_prime_sum = 5 / 12 :=
by
  -- The problem statement verifies that we have to show the calculation is correct
  sorry

end probability_prime_sum_is_correct_l548_548040


namespace exists_diff_nine_not_always_exists_diff_eleven_l548_548520

-- Problem 1: Proving existence of a difference of 9
theorem exists_diff_nine (S : Finset ℕ) (h1 : S.card = 55) (h2 : ∀ n ∈ S, n ≤ 100):
  ∃ (a b ∈ S), a ≠ b ∧ abs (a - b) = 9 := 
by {
  sorry
}

-- Problem 2: Proving non-existence of a difference of 11
theorem not_always_exists_diff_eleven (S : Finset ℕ) (h1 : S.card = 55) (h2 : ∀ n ∈ S, n ≤ 100):
  ¬(∀ a b ∈ S, abs (a - b) = 11) :=
by {
  sorry
}

end exists_diff_nine_not_always_exists_diff_eleven_l548_548520


namespace lcm_24_36_42_l548_548553

-- Definitions of the numbers involved
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 42

-- Statement for the lowest common multiple
theorem lcm_24_36_42 : Nat.lcm (Nat.lcm a b) c = 504 :=
by
  -- The proof will be filled in here
  sorry

end lcm_24_36_42_l548_548553


namespace general_formula_for_a_sum_of_b_l548_548586

-- Define the sequence Sn
def S (n : ℕ) : ℚ := (3 * n^2 + 5 * n) / 2

-- Problem 1: Prove general formula for an
theorem general_formula_for_a (n : ℕ) : 
  ∃ a : ℕ → ℚ, (∀ m, a m = (S m) - (S (m-1))) ∧ a n = 3 * n + 1 :=
by
  sorry

-- Define sequence a_n
def a (n : ℕ) : ℚ := 3 * n + 1

-- Define sequence b_n
def b (n : ℕ) : ℚ := 3 / (a n * a (n+1))

-- Sum of first n terms of sequence b
def sum_b (n : ℕ) : ℚ :=
  ∑ i in finset.range n, b (i + 1)

-- Problem 2: Sum of the first n terms of {bn} 
theorem sum_of_b (n : ℕ) : sum_b n = 1/4 - 1/(3*n+4) :=
by
  sorry

end general_formula_for_a_sum_of_b_l548_548586


namespace OlympicVolunteerArrangement_l548_548172

theorem OlympicVolunteerArrangement : 
  let volunteers := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
  ∑ (config : set (set char)), 
    config.card = 2 ∧ config ⊆ volunteers.to_finset ∧ 
    (∀ group ∈ config, group.card = 3) ∧ 
    (∀ group ∈ config, 'B' ∉ group → 'B' ∈ (config.filter (λ g, g.card = 3)).1) ∧
    (∀ group ∈ config, 'A' ∈ group ↔ 'B' ∉ group) = 290 := 
  sorry

end OlympicVolunteerArrangement_l548_548172


namespace find_other_parallel_side_l548_548198

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l548_548198


namespace area_of_entire_triangle_l548_548645

-- Given conditions
variable {T : Type} [MetricSpace T] [Triangle T] (A B C : T)
variable (M : T) (area_divided_triangle : ℝ)

-- Definition: M is the midpoint of the line segment BC
def is_median (A B C M : T) : Prop :=
  (dist B M = dist M C) ∧ (area_divided_triangle = 7)

-- Theorem statement: The area of the entire triangle
-- This follows from the fact that the median divides the triangle into two equal-area smaller triangles.
theorem area_of_entire_triangle (h : is_median A B C M) : 
  2 * area_divided_triangle = 14 := 
by 
  sorry

-- Assert the given condition about area
assert area_divided_triangle : ℝ := 7

end area_of_entire_triangle_l548_548645


namespace cube_tower_modulus_l548_548498

theorem cube_tower_modulus : 
  let cubes := {k | 1 ≤ k ∧ k ≤ 9}
  let condition (a b : Nat) : Prop := (a ≤ b + 3)
  let S := number of different towers using cubes under the condition
  (S % 1000 = 768) 
:=
sorry

end cube_tower_modulus_l548_548498


namespace option_c_has_minimum_value_4_l548_548008

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548008


namespace person_speed_l548_548167

namespace EscalatorProblem

/-- The speed of the person v_p walking on the moving escalator is 3 ft/sec given the conditions -/
theorem person_speed (v_p : ℝ) 
  (escalator_speed : ℝ := 12) 
  (escalator_length : ℝ := 150) 
  (time_taken : ℝ := 10) :
  escalator_length = (v_p + escalator_speed) * time_taken → v_p = 3 := 
by sorry

end EscalatorProblem

end person_speed_l548_548167


namespace range_of_m_l548_548412

noncomputable def quadraticExpr (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + 4 * m * x + m + 3

theorem range_of_m :
  (∀ x : ℝ, quadraticExpr m x ≥ 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l548_548412


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548033

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548033


namespace circle_condition_iff_l548_548756

-- Given a condition a < 2, we need to show it is a necessary and sufficient condition
-- for the equation x^2 + y^2 - 2x + 2y + a = 0 to represent a circle.

theorem circle_condition_iff (a : ℝ) :
  (∃ (x y : ℝ), (x - 1) ^ 2 + (y + 1) ^ 2 = 2 - a) ↔ (a < 2) :=
sorry

end circle_condition_iff_l548_548756


namespace factorization_correct_l548_548917

noncomputable def factor_polynomial (x : ℝ) : ℝ := 4 * x^3 - 4 * x^2 + x

theorem factorization_correct (x : ℝ) : 
  factor_polynomial x = x * (2 * x - 1)^2 :=
by
  sorry

end factorization_correct_l548_548917


namespace price_on_thursday_l548_548170

theorem price_on_thursday 
  (price_tuesday : ℝ)
  (increase_pct : ℝ)
  (discount_pct : ℝ)
  (price_wednesday : ℝ)
  (discount_amount : ℝ)
  (price_thursday : ℝ) :
  price_tuesday = 50 →
  increase_pct = 20 →
  discount_pct = 15 →
  price_wednesday = price_tuesday * (1 + increase_pct / 100) →
  discount_amount = price_wednesday * (discount_pct / 100) →
  price_thursday = price_wednesday - discount_amount →
  price_thursday = 51 :=
by simp; sorry

end price_on_thursday_l548_548170


namespace num_pairs_of_positive_integers_eq_77_l548_548618

theorem num_pairs_of_positive_integers_eq_77 : 
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.finite ∧
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.to_finset.card = 2 := 
by 
  sorry

end num_pairs_of_positive_integers_eq_77_l548_548618


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548027

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548027


namespace determine_n_l548_548786

noncomputable def binomial_theorem_expansion (p q : ℕ) (b k : ℕ) : ℕ :=
  binom p q * (2 * b) ^ (p - q) * k^q

theorem determine_n (n : ℕ) (b k : ℕ) (h1 : n ≥ 2) (h2 : b ≠ 0) (hk : k ≠ 0) (h3 : (binomial_theorem_expansion n 1 b k + binomial_theorem_expansion n 3 b k) = 0) : n = 3 := by
  sorry

end determine_n_l548_548786


namespace max_kings_12x12_l548_548666

def isNeighboring (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (abs (x1 - x2) ≤ 1) ∧ (abs (y1 - y2) ≤ 1)

noncomputable def maxKings (n : ℕ) : ℕ :=
  if h : n = 12 then 72 else 0

theorem max_kings_12x12 : maxKings 12 = 72 :=
by sorry

end max_kings_12x12_l548_548666


namespace hypotenuse_length_l548_548492

theorem hypotenuse_length (a b c : ℝ) (h₁ : a = 8) (h₂ : 1 / 2 * a * b = 48) :
  c = 4 * Real.sqrt 13 :=
by
  have h₃ : b = 12 := by
    calc
      b = 96 / 8 := by
        rw [← h₂, h₁]
        field_simp
      _ = 12 := by norm_num
    
  have h₄ : c^2 = 8^2 + 12^2 := by
    rw [h₁, h₃]
    ring
    
  rw Real.sqrt_sq_eq_abs at h₄
  norm_num at h₄

  have h₅ : c = Real.sqrt 208 := by
    rw [h₄, Real.sqrt_eq_rpow, Real.rpow_nat_cast]
    ring
    
  have h₆ : 208 = 16 * 13 := by norm_num

  rw [h₅, h₆, Real.sqrt_mul, Real.sqrt_nat_eq, Real.sqrt_nat_eq, Real.sqrt_nat]
  norm_num
  ring
  rw [Real.mul_pow]
  norm_num

  sorry

end hypotenuse_length_l548_548492


namespace convex_polygon_with_max_2n_sides_l548_548821

def convex_hull_area (X : Point) (polygon : Polygon) : ℝ := sorry

theorem convex_polygon_with_max_2n_sides (n : ℕ) (polygon : Polygon)
  (n_gt_2 : n > 2) 
  (area_polygon_lt_1 : area polygon < 1) :
  ∃ S : Set Point, (∀ X ∈ S, convex_hull_area X polygon = 1) ∧ is_convex_polygon S ∧ sides S ≤ 2 * n :=
begin
  sorry -- This is where the proof would go
end

end convex_polygon_with_max_2n_sides_l548_548821


namespace terrell_weight_lift_l548_548735

theorem terrell_weight_lift :
  ∀ (n : ℕ), 
  (let total_weight_25_lb := 3 * 25 * 10 in
   let total_weight_20_lb := 3 * 20 * n in
   n = 13 →
   total_weight_20_lb ≥ total_weight_25_lb) :=
by sorry

end terrell_weight_lift_l548_548735


namespace cloth_production_first_day_l548_548313

theorem cloth_production_first_day (a : ℕ → ℕ) (S₃₀ : ℕ) (a₃₀ : ℕ) 
  (h1 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h2 : S₃₀ = ∑ i in Finset.range 30, a (i + 1)) 
  (h3 : a₃₀ = a 30)
  (h4 : S₃₀ = 390)
  (h5 : a₃₀ = 21) :
  a 1 = 5 := 
sorry

end cloth_production_first_day_l548_548313


namespace quadratic_roots_prime_pairs_l548_548568

theorem quadratic_roots_prime_pairs (p q k : ℤ) (hp : p.prime) (hq : q.prime) :
  (25 * p^2 - 28 * q = k^2) → 
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) :=
sorry

end quadratic_roots_prime_pairs_l548_548568


namespace plywood_cut_difference_l548_548122

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548122


namespace roger_remaining_debt_is_correct_l548_548724

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l548_548724


namespace four_digit_number_count_l548_548615

-- Definitions for the problem statement
def is_even (n : ℕ) : Prop := n % 2 = 0
def valid_digits : set ℕ := {0, 2, 4, 6, 8}

-- Condition functions
def first_digit_set : set ℕ := {2, 4, 6, 8}
def third_digit_set : set ℕ := valid_digits
def fourth_digit_set : set ℕ := valid_digits

-- Function to determine if b is a valid second digit under the constraints
def valid_second_digit (a c b : ℕ) : Prop := b = (a + c) / 2 ∧ b ∈ valid_digits

-- Main proof problem statement
theorem four_digit_number_count :
  ∃ n : ℕ, n = 50 ∧
  (∀ a b c d : ℕ, a ∈ first_digit_set → c ∈ third_digit_set →
    b = (a + c) / 2 → b ∈ valid_digits → d ∈ fourth_digit_set →
    ⟨a, b, c, d⟩.1 ∈ {2, 4, 6, 8} ∧
    ⟨a, b, c, d⟩.3 ∈ {0, 2, 4, 6, 8} ∧
    ⟨a, b, c, d⟩.2 = (⟨a, b, c, d⟩.1 + ⟨a, b, c, d⟩.3) / 2 ∧
    ⟨a, b, c, d⟩.2 ∈ {0, 2, 4, 6, 8} →
    n = 50) :=
by sorry

end four_digit_number_count_l548_548615


namespace derivative_at_1_is_e_l548_548263

def f (x : ℝ) : ℝ := Real.exp x - 1 * x + (1 / 2) * x^2

theorem derivative_at_1_is_e : 
  let f (x : ℝ) := Real.exp x - 1 * x + (1 / 2) * x^2
  in  deriv f 1 = Real.exp 1 :=
by
  sorry

end derivative_at_1_is_e_l548_548263


namespace number_of_players_per_game_l548_548764

def total_players : ℕ := 50
def total_games : ℕ := 1225

-- If each player plays exactly one game with each of the other players,
-- there are C(total_players, 2) = total_games games.
theorem number_of_players_per_game : ∃ k : ℕ, k = 2 ∧ (total_players * (total_players - 1)) / 2 = total_games := 
  sorry

end number_of_players_per_game_l548_548764


namespace longest_and_shortest_sides_of_triangle_l548_548276

def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

def P1 : ℝ × ℝ := (1, 2)
def P2 : ℝ × ℝ := (4, 3)
def P3 : ℝ × ℝ := (3, -1)

def side_P1P2 := dist P1 P2
def side_P2P3 := dist P2 P3
def side_P1P3 := dist P1 P3

theorem longest_and_shortest_sides_of_triangle :
  side_P2P3 = Real.sqrt 17 ∧ side_P1P2 = Real.sqrt 10 :=
by
  sorry

end longest_and_shortest_sides_of_triangle_l548_548276


namespace upstream_distance_l548_548484

theorem upstream_distance
  (man_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (effective_downstream_speed: ℝ)
  (stream_speed : ℝ)
  (upstream_time : ℝ)
  (upstream_distance : ℝ):
  man_speed = 7 ∧ downstream_distance = 45 ∧ downstream_time = 5 ∧ effective_downstream_speed = man_speed + stream_speed 
  ∧ effective_downstream_speed * downstream_time = downstream_distance 
  ∧ upstream_time = 5 ∧ upstream_distance = (man_speed - stream_speed) * upstream_time 
  → upstream_distance = 25 :=
by
  sorry

end upstream_distance_l548_548484


namespace probability_two_same_color_l548_548301

/-- There are 4 white balls and 2 black balls in a box, and two balls are drawn at once.
    Prove that the probability of drawing two balls of the same color is 7/15. -/
theorem probability_two_same_color 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (drawn_balls : ℕ)
  (h_total : total_balls = 6) 
  (h_white : white_balls = 4) 
  (h_black : black_balls = 2) 
  (h_drawn : drawn_balls = 2) 
  : (number_of_favorable_outcomes total_balls white_balls black_balls drawn_balls) / (number_of_ways_to_draw total_balls drawn_balls) = 7 / 15 :=
sorry

/-- Helper function to calculate number of ways to draw k balls from n balls, i.e., combinations n choose k. -/
noncomputable def number_of_ways_to_draw : ℕ → ℕ → ℕ
| n, k := nat.choose n k

/-- Helper function to calculate number of favorable outcomes for drawing two balls of the same color. -/
noncomputable def number_of_favorable_outcomes : ℕ → ℕ → ℕ → ℕ → ℕ
| total, white, black, drawn := number_of_ways_to_draw white 2 + number_of_ways_to_draw black 2

end probability_two_same_color_l548_548301


namespace desired_depth_to_be_dug_l548_548812

noncomputable def total_man_hours_initial (d: ℝ) : ℝ :=
  72 * 8 * d

noncomputable def total_man_hours_desired (D: ℝ, d: ℝ) : ℝ :=
  160 * 6 * d

theorem desired_depth_to_be_dug :
  ∀ d : ℝ, 
    (total_man_hours_initial d) = (total_man_hours_desired 50 d) → 
    50 = 30 / (576 / 960) :=
by
  intros,
  sorry

end desired_depth_to_be_dug_l548_548812


namespace nth_term_is_4037_l548_548413

noncomputable def arithmetic_sequence_nth_term (n : ℕ) : ℤ :=
7 + (n - 1) * 6

theorem nth_term_is_4037 {n : ℕ} : arithmetic_sequence_nth_term 673 = 4037 :=
by
  sorry

end nth_term_is_4037_l548_548413


namespace logarithmic_product_solution_l548_548639

-- Define the given condition as z equals the product of logarithms
def product_of_logarithms := ∏ i in Finset.range 38, (Real.log (i + 4) / Real.log (i + 3))

-- Goal is to show that this product equals log base 3 of 41
theorem logarithmic_product_solution : product_of_logarithms = Real.log 41 / Real.log 3 := 
sorry

end logarithmic_product_solution_l548_548639


namespace greatest_possible_ratio_l548_548538

theorem greatest_possible_ratio : 
    ∃ A B C D : ℤ × ℤ,
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 16 ∧ B.1 ^ 2 + B.2 ^ 2 = 16 ∧ C.1 ^ 2 + C.2 ^ 2 = 16 ∧ D.1 ^ 2 + D.2 ^ 2 = 16) ∧
    (∃ (O : ℤ × ℤ) (hO : O = (0, 0)), ∠AOB = ∠COD) ∧
    (∃ rAB rCD: ℝ, irrational rAB ∧ irrational rCD ∧ euclidean_dist A B = rAB ∧ euclidean_dist C D = rCD) → 
    ∀ rAB rCD, euclidean_dist A B = rAB ∧ euclidean_dist C D = rCD → 
    (∀ x, x = rAB / rCD → x ≤ 1) :=
sorry

end greatest_possible_ratio_l548_548538


namespace sequence_general_formula_l548_548229

theorem sequence_general_formula :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n : ℕ, n > 0 → a n - a (n + 1) = 2 * a n * a (n + 1) / (n * (n + 1))) →
  ∀ n : ℕ, n > 0 → a n = n / (3 * n - 2) :=
by
  intros a h1 h_rec n hn
  sorry

end sequence_general_formula_l548_548229


namespace exists_acute_triangles_l548_548977

open Classical

variables {P : Type} [AffineSpace P ℝ] {L : Set P} (p q r : P) (n : ℕ)
variables (points : Fin n → P) (lines : ∀ i, points i ∈ L)

-- Representation of conditions
def line := L
def more_than_two_points (h : n > 2) := True

-- The statement to prove
theorem exists_acute_triangles (h : n > 2) :
  ∃ Q ∈ (comp (nonempty_complement L)),  -- ∃ Q not on L
    (∃ majority,  -- Majority count
        ∀ (i j), 
          (i ≠ j) →  
            acute_triangle (points i) (points j) Q) :=
sorry

end exists_acute_triangles_l548_548977


namespace number_added_is_59_l548_548287

theorem number_added_is_59 (x : ℤ) (h1 : -2 < 0) (h2 : -3 < 0) (h3 : -2 * -3 + x = 65) : x = 59 :=
by sorry

end number_added_is_59_l548_548287


namespace sum_of_integers_product_neg17_l548_548425

theorem sum_of_integers_product_neg17 (a b c : ℤ) (h : a * b * c = -17) : a + b + c = -15 ∨ a + b + c = 17 :=
sorry

end sum_of_integers_product_neg17_l548_548425


namespace sam_investment_time_l548_548385

theorem sam_investment_time (P r : ℝ) (n A t : ℕ) (hP : P = 8000) (hr : r = 0.10) (hn : n = 2) (hA : A = 8820) :
  A = P * (1 + r / n) ^ (n * t) → t = 1 :=
by
  sorry

end sam_investment_time_l548_548385


namespace calculateL_l548_548914

-- Defining the constants T, H, and C
def T : ℕ := 5
def H : ℕ := 10
def C : ℕ := 3

-- Definition of the formula for L
def crushingLoad (T H C : ℕ) : ℚ := (15 * T^3 : ℚ) / (H^2 + C)

-- The theorem to prove
theorem calculateL : crushingLoad T H C = 1875 / 103 := by
  -- Proof goes here
  sorry

end calculateL_l548_548914


namespace number_of_pairs_l548_548624

theorem number_of_pairs :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end number_of_pairs_l548_548624


namespace archery_competition_hits_l548_548665

theorem archery_competition_hits :
  ∃ a b c d : ℕ, 
    a + b + c + d = 10 ∧ 
    8 * a + 12 * b + 14 * c + 18 * d = 110 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 :=
by
  -- declaring the values of arrows hitting each ring
  let a := 5
  let b := 2
  let c := 2
  let d := 1
  use [a, b, c, d]
  -- proving total arrows shot
  have h1 : a + b + c + d = 10 :=
    by norm_num
  -- proving total score obtained
  have h2 : 8 * a + 12 * b + 14 * c + 18 * d = 110 :=
    by norm_num
  -- proving each ring is hit at least once
  have h3 : a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ d ≥ 1 :=
    by norm_num
  exact ⟨h1, h2, h3⟩

end archery_competition_hits_l548_548665


namespace katrina_tax_deduction_l548_548336

variable (hourlyWage : ℚ) (taxRate : ℚ)

def wageInCents (wage : ℚ) : ℚ := wage * 100
def taxInCents (wageInCents : ℚ) (rate : ℚ) : ℚ := wageInCents * rate / 100

theorem katrina_tax_deduction : 
  hourlyWage = 25 ∧ taxRate = 2.5 → taxInCents (wageInCents hourlyWage) taxRate = 62.5 := 
by 
  sorry

end katrina_tax_deduction_l548_548336


namespace sin_cos_sum_l548_548248

theorem sin_cos_sum (A : ℝ) (h : sin (2 * A) = 2 / 3) : sin A + cos A = sqrt(15) / 3 :=
sorry

end sin_cos_sum_l548_548248


namespace fraction_of_capacity_l548_548472

theorem fraction_of_capacity
    (bus_capacity : ℕ)
    (x : ℕ)
    (first_pickup : ℕ)
    (second_pickup : ℕ)
    (unable_to_board : ℕ)
    (bus_full : bus_capacity = x + (second_pickup - unable_to_board))
    (carry_fraction : x / bus_capacity = 3 / 5) : 
    true := 
sorry

end fraction_of_capacity_l548_548472


namespace water_flow_rate_l548_548129

-- Define the problem conditions.
def basin_capacity := 260 -- gallons
def leakage_rate := 4 -- gallons per second
def fill_time := 13 -- seconds

-- Define the unknown amount of water flow per second as W.
variable (W : ℕ)

-- Define the problem statement and prove it.
theorem water_flow_rate :
  13 * W - fill_time * leakage_rate = basin_capacity →
  W = 24 :=
by
  intros h
  have : 13 * W = basin_capacity + fill_time * leakage_rate := by linarith
  have : W = (basin_capacity + fill_time * leakage_rate) / 13 := by
    rw this
    exact Nat.div_eq_of_eq_mul_left (by norm_num) this
  rw this
  norm_num
  exact W

end water_flow_rate_l548_548129


namespace roots_difference_squared_l548_548349

theorem roots_difference_squared :
  ∀ p q : ℚ, (p - q) ^ 2 = 529 / 36 → is_root (6 * X^2 - 7 * X - 20) p → is_root (6 * X^2 - 7 * X - 20) q → 
  (p - q) ^ 2 = 529 / 36 := 
by
  intros p q h root_p root_q
  sorry

end roots_difference_squared_l548_548349


namespace tan_inequality_solution_l548_548432

noncomputable def solutionSet (k : ℤ) : Set ℝ :=
  {x | ∃ k : ℤ, -π/6 + k * π < x ∧ x < π/6 + k * π}

theorem tan_inequality_solution :
  ∀ x : ℝ, (3 * tan x + sqrt 3 > 0) ↔ (∃ k : ℤ, -π/6 + k * π < x ∧ x < π/6 + k * π) :=
by
  sorry

end tan_inequality_solution_l548_548432


namespace correct_calculation_l548_548791

theorem correct_calculation :
  ∃ (a : ℤ), (a^2 + a^2 = 2 * a^2) ∧ 
  (¬(3*a + 4*(a : ℤ) = 12*a*(a : ℤ))) ∧ 
  (¬((a*(a : ℤ)^2)^3 = a*(a : ℤ)^6)) ∧ 
  (¬((a + 3)^2 = a^2 + 9)) :=
by
  sorry

end correct_calculation_l548_548791


namespace perimeter_difference_l548_548069

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548069


namespace tank_filled_in_96_minutes_l548_548372

-- conditions
def pipeA_fill_time : ℝ := 6
def pipeB_empty_time : ℝ := 24
def time_with_both_pipes_open : ℝ := 96

-- rate computations and final proof
noncomputable def pipeA_fill_rate : ℝ := 1 / pipeA_fill_time
noncomputable def pipeB_empty_rate : ℝ := 1 / pipeB_empty_time
noncomputable def net_fill_rate : ℝ := pipeA_fill_rate - pipeB_empty_rate
noncomputable def tank_filled_in_time_with_both : ℝ := time_with_both_pipes_open * net_fill_rate

theorem tank_filled_in_96_minutes (HA : pipeA_fill_time = 6) (HB : pipeB_empty_time = 24)
  (HT : time_with_both_pipes_open = 96) : tank_filled_in_time_with_both = 1 :=
by
  sorry

end tank_filled_in_96_minutes_l548_548372


namespace prove_ab_values_l548_548575

/--
Given \( a, b \in \mathbb{R} \), prove that if \( a - 2i = (b + i)i \),
then \( a = -1 \) and \( b = -2 \).
-/
theorem prove_ab_values (a b : ℝ) (h : a - 2 * complex.I = (b + complex.I) * complex.I) : 
  a = -1 ∧ b = -2 :=
sorry

end prove_ab_values_l548_548575


namespace symmetric_line_about_point_l548_548314

def line_symmetric_to (m b : ℝ) (p : ℝ × ℝ) : ℝ → ℝ :=
  λ x, 2 * x - 3  -- This needs some appropriate function to compute it, hardcoding based on answer for now

theorem symmetric_line_about_point (m b : ℝ) (x₀ y₀ : ℝ)
  (h : ∀ x, (y ≠ (m * x + b) ↔ y ≠ line_symmetric_to m b (x₀, y₀) x)) :
  ∀ x, (y ≠ (2 * x + 1) ↔ y ≠ (2 * x - 3)) :=
sorry

end symmetric_line_about_point_l548_548314


namespace parallelogram_opposite_sides_eq_l548_548311

variables {A B C D O : Type} [AddCommGroup A] [Module ℝ A]

structure Parallelogram (A B C D : Type) :=
  (side_eq : (A = B) → (C = D))
  (opposite_sides_eq : ∀ (A B : Module ℝ A), A = B →  B = C) 

theorem parallelogram_opposite_sides_eq (h : Parallelogram A B C D) : A = C := 
by {
  sorry 
}

end parallelogram_opposite_sides_eq_l548_548311


namespace remainder_2011_2015_mod_23_l548_548783

theorem remainder_2011_2015_mod_23 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := 
by
  sorry

end remainder_2011_2015_mod_23_l548_548783


namespace tower_height_l548_548144

theorem tower_height (h CD AC DE AD : ℝ) (tan45 tan30 cos40 sin40 : ℝ)
  (H1 : tan45 = 1)
  (H2 : tan30 = 1/real.sqrt 3)
  (H3 : cos40 = real.cos (40 * real.pi / 180))
  (H4 : sin40 = real.sin (40 * real.pi / 180))
  (H5 : tan45 = h / AC)
  (H6 : AC = h)
  (H7 : DE = CD * sin40)
  (H8 : AD = CD * cos40)
  (H9 : CD = 10)
  (H10 : tan30 = h / (AD + DE)) :
  h = 10 * real.sqrt 3 * (cos40 + sin40) * real.sqrt 3 / (3 - real.sqrt 3) :=
begin
  sorry
end

end tower_height_l548_548144


namespace appears_in_31st_equation_l548_548771

theorem appears_in_31st_equation : 
  ∃ n : ℕ, 2016 ∈ {x | 2*x^2 ≤ 2016 ∧ 2016 < 2*(x+1)^2} ∧ n = 31 :=
by
  sorry

end appears_in_31st_equation_l548_548771


namespace range_of_f_l548_548452

def f (x : ℝ) : ℝ := 1 / (1 - 5 * x)^2

theorem range_of_f : set.Ioi (0 : ℝ) = set.range (f) :=
by
  sorry

end range_of_f_l548_548452


namespace parabola_vertex_l548_548761

theorem parabola_vertex : ∃ h k, ∀ x, -2 * x^2 + 3 = -2 * (x - h)^2 + k ∧ h = 0 ∧ k = 3 :=
by
  use 0
  use 3
  intros x
  split
  · ring
  split
  · refl
  · refl
  sorry

end parabola_vertex_l548_548761


namespace rectangle_area_constant_l548_548427

theorem rectangle_area_constant 
    (d : ℝ) 
    (length width : ℝ) 
    (h_ratio : length / width = 5 / 4) 
    (h_diag : d = real.sqrt (length^2 + width^2)) : 
    ∃ (k : ℝ), k = 20 / 41 ∧ ∀ A : ℝ, A = k * d^2 :=
by {
    use (20 / 41),
    split,
    {
        sorry,
    },
    {
        intro A,
        sorry,
    },
}

end rectangle_area_constant_l548_548427


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548029

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548029


namespace length_of_train_l548_548859

theorem length_of_train 
    (t_cross_platform : 39)
    (t_cross_pole : 18)
    (len_platform : 350)
    (speed := (L : ℝ) / t_cross_pole) :
    (L : ℝ) =
    (L * t_cross_platform / t_cross_pole = L + len_platform) :=
sorry

end length_of_train_l548_548859


namespace total_sample_mean_correct_l548_548134

theorem total_sample_mean_correct (n_male n_female : ℕ) (mean_male mean_female : ℕ) :
  n_male = 20 → mean_male = 12 → n_female = 30 → mean_female = 10 →
  (n_male * mean_male + n_female * mean_female) / (n_male + n_female) = 10.8 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
-- Sorry to skip the rest of the proof.
sory

end total_sample_mean_correct_l548_548134


namespace correct_statements_are_C_and_D_l548_548037

theorem correct_statements_are_C_and_D
  (a b c m : ℝ)
  (ha1 : -1 < a) (ha2 : a < 5)
  (hb1 : -2 < b) (hb2 : b < 3)
  (hab : a > b)
  (h_ac2bc2 : a * c^2 > b * c^2) (hc2_pos : c^2 > 0)
  (h_ab_pos : a > b) (h_b_pos : b > 0) (hm_pos : m > 0) :
  (¬(1 < a - b ∧ a - b < 2)) ∧ (¬(a^2 > b^2)) ∧ (a > b) ∧ ((b + m) / (a + m) > b / a) :=
by sorry

end correct_statements_are_C_and_D_l548_548037


namespace ellipse_properties_l548_548589

noncomputable def is_on_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties
  (a b c : ℝ)
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (heq : ∀ (x y : ℝ), is_on_ellipse x y a b → (x = 1) ∧ (y = Real.sqrt 3 / 2))
  (h4 : 2 * a = 4)
  (h5 : 1 / a^2 + 3 / (4 * b^2) = 1) : 
  ∃ (f1 f2 : ℝ × ℝ), 
    (c = Real.sqrt 3) ∧
    is_on_ellipse 1 (Real.sqrt 3 / 2) 2 1 ∧ 
    f1 = (-Real.sqrt 3, 0) ∧ 
    f2 = (Real.sqrt 3, 0) ∧ 
    (∃ m : ℝ, ∀ (x y1 y2 : ℝ), is_on_ellipse x y 2 1 ∧ y1 ≠ y2 ∧ x = 1 → 
        (let area := (2 * Real.sqrt (m^2 + 3)) / (m^2 + 4) 
          in m = 0 ∧ area = Real.sqrt 3 / 2)) := sorry

end ellipse_properties_l548_548589


namespace finish_job_together_in_l548_548039

-- Definitions
def work_rate_A := 1 / 18
def work_rate_B := 1 / 30
def combined_work_rate := work_rate_A + work_rate_B
def finish_time := 1 / combined_work_rate

-- Theorem to prove
theorem finish_job_together_in :
  finish_time = 11.25 :=
sorry

end finish_job_together_in_l548_548039


namespace steer_weight_in_pounds_l548_548766

theorem steer_weight_in_pounds :
  ∀ (weight_in_kg : ℝ), weight_in_kg = 200 → 
  let conversion_factor : ℝ := 0.4536 in
  round (weight_in_kg / conversion_factor) = 441 :=
by
  intro weight_in_kg h_weight_eq
  let conversion_factor : ℝ := 0.4536
  rw h_weight_eq
  sorry

end steer_weight_in_pounds_l548_548766


namespace monotonically_increasing_interval_l548_548991

noncomputable def f : ℝ → ℝ := λ x => x * Real.exp x

theorem monotonically_increasing_interval :
  ∀ x, x ≥ -1 → ∃ c, f' x = c ∧ c ≥ 0 :=
by
  sorry

end monotonically_increasing_interval_l548_548991


namespace problem_statement_l548_548518

theorem problem_statement : 100 * 29.98 * 2.998 * 1000 = (2998)^2 :=
by
  sorry

end problem_statement_l548_548518


namespace initial_number_of_persons_l548_548404

theorem initial_number_of_persons (n : ℕ) 
  (w_increase : ∀ (k : ℕ), k = 4) 
  (old_weight new_weight : ℕ) 
  (h_old : old_weight = 58) 
  (h_new : new_weight = 106) 
  (h_difference : new_weight - old_weight = 48) 
  : n = 12 := 
by
  sorry

end initial_number_of_persons_l548_548404


namespace sum_of_b_values_l548_548953

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l548_548953


namespace binomial_floor_divisible_by_prime_l548_548048

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem binomial_floor_divisible_by_prime (p n : ℕ) (hp : p.prime) (hn : n ≥ p) :
  (binomial_coefficient n p - (n / p)) % p = 0 :=
sorry

end binomial_floor_divisible_by_prime_l548_548048


namespace g_domain_size_l548_548531

def g : ℕ → ℕ
| 15 := 48
| n  := if n % 2 = 0 then n / 2 else 4 * n + 1

theorem g_domain_size : 
  ∃ s : set ℕ, 15 ∈ s ∧ 
    (∀ x ∈ s, g x ∈ s) ∧ 
    (s = {15, 48, 24, 12, 6, 3, 13, 53, 212} ∨
    sorry) ∧ -- Necessary to justify 's' covers all integers leading to the cycle
  s.card = 9 := 
sorry

end g_domain_size_l548_548531


namespace area_triangle_BDF_half_hexagon_l548_548319

variable {Point : Type}
variable [InscCircle : ∀ (A B C : Point), Prop]

/-- Define a hexagon inscribed in a circle. -/
structure Hexagon (A B C D E F : Point) (circ : InscCircle A B F) where
  AB_eq_BC : dist A B = dist B C
  CD_eq_DE : dist C D = dist D E
  EF_eq_FA : dist E F = dist F A

noncomputable def area (P Q R : Point) : ℝ := sorry
noncomputable def hex_area (A B C D E F : Point) : ℝ := sorry

theorem area_triangle_BDF_half_hexagon 
  {A B C D E F : Point} (h : Hexagon A B C D E F (by sorry)) :
  area B D F = 0.5 * hex_area A B C D E F :=
by sorry

end area_triangle_BDF_half_hexagon_l548_548319


namespace football_team_must_win_min_matches_l548_548302

theorem football_team_must_win_min_matches :
  ∀ (total_matches lose_matches win_points draw_points loss_points : ℕ),
  total_matches = 15 →
  lose_matches = 2 →
  win_points = 3 →
  draw_points = 1 →
  loss_points = 0 →
  ∀ points_needed, points_needed = 33 →
  ∃ (min_wins : ℕ), min_wins = 10 ∧
  ∀ (x : ℕ),
  (3 * x + draw_points * (total_matches - lose_matches - x) ≥ points_needed) ↔ (x ≥ min_wins) :=
by
  intros total_matches lose_matches win_points draw_points loss_points ht hl hw hd hlt points_needed hp
  use 10
  split
  · rfl
  · intro x
    split
    · intro hx
      calc
        3 * x + 1 * (15 - 2 - x) ≥ 33 : by assumption
        2 * x + 13 ≥ 33 : by rw [← add_assoc, mul_add, hl, sub_sub, sub_self, add_zero]
        2 * x ≥ 20 : by linarith
        x ≥ 10 : by linarith
      
    · intro hx
      calc
        3 * x + 1 * (15 - 2 - x)
            = 2 * x + 13 : by rw [mul_add, hl, sub_sub, sub_self, add_zero]
        ≥ 33       : by linarith
      sorry

end football_team_must_win_min_matches_l548_548302


namespace prob_more_twos_than_fours_l548_548641

theorem prob_more_twos_than_fours :
  let probability := (223 : ℚ) / 648 in
  probability = (223 : ℚ) / 648 := by
  sorry

end prob_more_twos_than_fours_l548_548641


namespace roger_remaining_debt_is_correct_l548_548725

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l548_548725


namespace B_starts_after_A_l548_548506

theorem B_starts_after_A :
  ∀ (A_walk_speed B_cycle_speed dist_from_start t : ℝ), 
    A_walk_speed = 10 →
    B_cycle_speed = 20 →
    dist_from_start = 80 →
    B_cycle_speed * (dist_from_start - A_walk_speed * t) / A_walk_speed = t →
    t = 4 :=
by 
  intros A_walk_speed B_cycle_speed dist_from_start t hA_speed hB_speed hdist heq;
  sorry

end B_starts_after_A_l548_548506


namespace correct_inequality_l548_548693

variable (a b c d : ℝ)
variable (h₁ : a > b)
variable (h₂ : b > 0)
variable (h₃ : 0 > c)
variable (h₄ : c > d)

theorem correct_inequality :
  (c / a) - (d / b) > 0 :=
by sorry

end correct_inequality_l548_548693


namespace rower_trip_time_to_Big_Rock_l548_548151

noncomputable def row_trip_time (rowing_speed_in_still_water : ℝ) (river_speed : ℝ) (distance_to_destination : ℝ) : ℝ :=
  let speed_upstream := rowing_speed_in_still_water - river_speed
  let speed_downstream := rowing_speed_in_still_water + river_speed
  let time_upstream := distance_to_destination / speed_upstream
  let time_downstream := distance_to_destination / speed_downstream
  time_upstream + time_downstream

theorem rower_trip_time_to_Big_Rock :
  row_trip_time 7 2 3.2142857142857144 = 1 :=
by
  sorry

end rower_trip_time_to_Big_Rock_l548_548151


namespace central_symmetry_implies_a_eq_neg_four_thirds_l548_548647

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (|x - a| + |x - 4|)

def is_centrally_symmetric (f : ℝ → ℝ) : Prop :=
  ∃ t, ∀ x, f x = f (2 * t - x)

theorem central_symmetry_implies_a_eq_neg_four_thirds (a : ℝ) :
  is_centrally_symmetric (f x a) → a = -4 / 3 :=
begin
  sorry
end

end central_symmetry_implies_a_eq_neg_four_thirds_l548_548647


namespace remaining_flight_time_l548_548543

-- Define the conditions
def flight_duration_hours : ℕ := 10
def tv_episodes : ℕ := 3
def tv_episode_length_minutes : ℕ := 25
def sleep_duration_hours : ℚ := 4.5
def movies : ℕ := 2
def movie_length_hours : ℚ := 1.75
def minutes_per_hour : ℕ := 60

-- Prove the remaining minutes before the flight ends
theorem remaining_flight_time : 
  (flight_duration_hours * minutes_per_hour) 
  - (tv_episodes * tv_episode_length_minutes 
     + (sleep_duration_hours * minutes_per_hour).to_nat 
     + (movies * (movie_length_hours * minutes_per_hour)).to_nat) 
  = 45 :=
by
  sorry

end remaining_flight_time_l548_548543


namespace perimeter_difference_l548_548062

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548062


namespace slope_intercept_of_line_l548_548583

theorem slope_intercept_of_line :
  ∃ (l : ℝ → ℝ), (∀ x, l x = (4 * x - 9) / 3) ∧ l 3 = 1 ∧ ∃ k, k / (1 + k^2) = 1 / 2 ∧ l x = (k^2 - 1) / (1 + k^2) := sorry

end slope_intercept_of_line_l548_548583


namespace fox_can_eat_80_fox_cannot_eat_65_l548_548824
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end fox_can_eat_80_fox_cannot_eat_65_l548_548824


namespace num_subsets_card_eq_l548_548421

theorem num_subsets_card_eq (S : Set α) (h : S.card = 4) : S.powerset.card = 16 :=
  sorry

end num_subsets_card_eq_l548_548421


namespace pages_573_digit_difference_l548_548541

def count_digit (d : ℕ) (n : ℕ) : ℕ :=
  (n.toString.filter (λ c, c = d.toString.head!)).length

def total_occurrences (digit : ℕ) (pages : fin 574) : ℕ :=
  (list.finRange 574).map (λ n, count_digit digit n.val).sum

theorem pages_573_digit_difference :
  total_occurrences 7 ⟨573, sorry⟩ - total_occurrences 3 ⟨573, sorry⟩ = -18 := 
by {
  sorry
}

end pages_573_digit_difference_l548_548541


namespace optionC_has_min_4_l548_548001

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548001


namespace triangle_side_length_x_l548_548677

theorem triangle_side_length_x
  (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ)
  (hy : y = 7)
  (hz : z = 3)
  (hcos : cos_Y_minus_Z = 7 / 8) :
  ∃ x : ℝ, x = Real.sqrt 18.625 :=
by
  sorry

end triangle_side_length_x_l548_548677


namespace sufficient_and_necessary_condition_l548_548359

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem sufficient_and_necessary_condition (a b : ℝ) : (a + b > 0) ↔ (f a + f b > 0) :=
by sorry

end sufficient_and_necessary_condition_l548_548359


namespace centroid_triangle_areas_equal_l548_548836

-- Definition of centroid and medians of a triangle
variable {A B C G : Type} [has_medians A B C G] [is_centroid A B C G]

-- Definition of triangles
def triangle1 := {A, B, G}
def triangle2 := {B, C, G}
def triangle3 := {C, A, G}
def triangle4 := {G, A, B}
def triangle5 := {G, B, C}
def triangle6 := {G, C, A}

-- Areas of opposite triangles
variable (area1 area4 : ℕ)
variable (area2 area5 : ℕ)
variable (area3 area6 : ℕ)

-- The Lean 4 theorem statement based on the given problem conditions and conclusions
theorem centroid_triangle_areas_equal :
  (area1 = area4) ∧ (area2 = area5) ∧ (area3 = area6) := sorry

end centroid_triangle_areas_equal_l548_548836


namespace sum_of_b_for_rational_roots_l548_548952

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l548_548952


namespace fish_guppies_consumption_l548_548329

/-- 
  Jason's daily fish consumption in terms of guppies:
  - a moray eel that eats 20 guppies a day.
  - 5 betta fish who each eat 7 guppies a day on odd-numbered days and 8 guppies on even-numbered days.
  - 3 angelfish who each eat 4 guppies a day but double their consumption every third day.
  - 2 lionfish who each eat 10 guppies a day on weekdays and 12 guppies on weekends.
  Prove that Jason needs to buy, on average, 95 guppies per day over the course of a week.
-/
theorem fish_guppies_consumption : 
  let moray_eel := 20 * 7,
      betta_fish := (5 * 7 * 4) + (5 * 8 * 3),
      angelfish := (3 * 4 * 4) + (3 * 8 * 3),
      lionfish := (2 * 10 * 5) + (2 * 12 * 2),
      total := moray_eel + betta_fish + angelfish + lionfish
  in total / 7 = 95 :=
by
  let moray_eel := 20 * 7
  let betta_fish := (5 * 7 * 4) + (5 * 8 * 3)
  let angelfish := (3 * 4 * 4) + (3 * 8 * 3)
  let lionfish := (2 * 10 * 5) + (2 * 12 * 2)
  let total := moray_eel + betta_fish + angelfish + lionfish
  have : total = 668 := by sorry
  have : total / 7 = 668 / 7 := by sorry
  exact this.symm.trans (show 668 / 7 = 95 from by norm_num)

end fish_guppies_consumption_l548_548329


namespace solve_congruence_l548_548393

theorem solve_congruence (n : ℤ) : 15 * n ≡ 9 [ZMOD 47] → n ≡ 18 [ZMOD 47] :=
by
  sorry

end solve_congruence_l548_548393


namespace captain_age_l548_548304

theorem captain_age
  (C W : ℕ)
  (avg_team_age : ℤ)
  (avg_remaining_players_age : ℤ)
  (total_team_age : ℤ)
  (total_remaining_players_age : ℤ)
  (remaining_players_count : ℕ)
  (total_team_count : ℕ)
  (total_team_age_eq : total_team_age = total_team_count * avg_team_age)
  (remaining_players_age_eq : total_remaining_players_age = remaining_players_count * avg_remaining_players_age)
  (total_team_eq : total_team_count = 11)
  (remaining_players_eq : remaining_players_count = 9)
  (avg_team_age_eq : avg_team_age = 23)
  (avg_remaining_players_age_eq : avg_remaining_players_age = avg_team_age - 1)
  (age_diff : W = C + 5)
  (players_age_sum : total_team_age = total_remaining_players_age + C + W) :
  C = 25 :=
by
  sorry

end captain_age_l548_548304


namespace distance_AF_eq_2_l548_548418

noncomputable def parabola := { p : ℝ × ℝ // p.1^2 = 4 * p.2 }

def focus : ℝ × ℝ := (0, 1)

def intersection_points (l : ℝ × ℝ → Prop) 
  (h : ∀ p ∈ parabola, l p → p ∈ parabola) : ℝ × ℝ × ℝ × ℝ :=
  sorry -- This is a placeholder for the actual intersection function

def tangent_line (A : ℝ × ℝ) : ℝ → ℝ := 
  λ x, (1 / 2) * A.1 * x - (1 / 2) * A.1 * A.1 + A.2

def area_of_triangle_MON (M N O : ℝ × ℝ) : ℝ :=
  (1 / 2) * (M.1 * (N.2 - O.2) + N.1 * (O.2 - M.2) + O.1 * (M.2 - N.2))

theorem distance_AF_eq_2 (l : ℝ × ℝ → Prop) 
  (h : ∀ p ∈ parabola, l p → p ∈ parabola) 
  (hMN : ∀ A ∈ parabola, let M := (A.1 / 2, 0), N := (0, -A.2) 
    in area_of_triangle_MON M N (0, 0) = 1 / 2) : 
  ∃ A ∈ parabola, |real.sqrt((A.1 - 0)^2 + (A.2 - 1)^2)| = 2 := 
sorry

end distance_AF_eq_2_l548_548418


namespace find_other_parallel_side_l548_548199

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l548_548199


namespace factorization_correct_l548_548916

theorem factorization_correct (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_correct_l548_548916


namespace interval_of_increase_of_f_l548_548746

noncomputable def f : ℝ → ℝ := λ x, real.logb 0.5 (8 + 2 * x - x ^ 2)

theorem interval_of_increase_of_f :
  set_of (λ x, -2 < x ∧ x < 4) ∧ ∀ x, (1 ≤ x ∧ x < 4 → deriv f x < 0) :=
sorry

end interval_of_increase_of_f_l548_548746


namespace sequence_sum_formula_l548_548980

/-- Definition of the arithmetic sequence a_n. -/
def arithmetic_seq (n : ℕ) : ℕ := 2 * n

/-- Definition of the geometric sequence b_n. -/
def geometric_seq (n : ℕ) : ℕ := 2 ^ (n - 1)

/-- Sum of the first n terms of the sequence {1/(a_n * a_(n+1))}. -/
def sum_sequence (n : ℕ) : ℝ := (∑ i in Finset.range n, 1 / (arithmetic_seq i * arithmetic_seq (i + 1)))

theorem sequence_sum_formula (n : ℕ) : sum_sequence n = n / (4 * (n + 1)) :=
by
  -- formal proof would go here
  sorry

end sequence_sum_formula_l548_548980


namespace count_five_digit_palindromes_with_two_odd_digits_l548_548468

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def five_digit_palindrome_with_two_odd_digits (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ is_palindrome n ∧ (nat.digits 10 n).countp (λ d, d % 2 = 1) = 2

theorem count_five_digit_palindromes_with_two_odd_digits :
  finset.card (finset.filter five_digit_palindrome_with_two_odd_digits (finset.range 100000).filter (λ n, n >= 10000 ∧ n < 100000)) = 225 :=
by sorry

end count_five_digit_palindromes_with_two_odd_digits_l548_548468


namespace Angle_Not_Equivalent_l548_548866

theorem Angle_Not_Equivalent (θ : ℤ) : (θ = -750) → (680 % 360 ≠ θ % 360) :=
by
  intro h
  have h1 : 680 % 360 = 320 := by norm_num
  have h2 : -750 % 360 = -30 % 360 := by norm_num
  have h3 : -30 % 360 = 330 := by norm_num
  rw [h, h2, h3]
  sorry

end Angle_Not_Equivalent_l548_548866


namespace perimeters_positive_difference_l548_548098

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548098


namespace min_value_expression_l548_548469

theorem min_value_expression : ∃ (x y : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 ≥ 0) ∧ (∀ (x y : ℝ), x = 4 ∧ y = -3 → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
sorry

end min_value_expression_l548_548469


namespace time_to_pass_jogger_l548_548830

noncomputable def jogger_speed_kmh := 9 -- in km/hr
noncomputable def train_speed_kmh := 45 -- in km/hr
noncomputable def jogger_headstart_m := 240 -- in meters
noncomputable def train_length_m := 100 -- in meters

noncomputable def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_mps := kmh_to_mps jogger_speed_kmh
noncomputable def train_speed_mps := kmh_to_mps train_speed_kmh
noncomputable def relative_speed := train_speed_mps - jogger_speed_mps
noncomputable def distance_to_be_covered := jogger_headstart_m + train_length_m

theorem time_to_pass_jogger : distance_to_be_covered / relative_speed = 34 := by
  sorry

end time_to_pass_jogger_l548_548830


namespace logically_equivalent_to_original_l548_548036

def original_statement (E W : Prop) : Prop := E → ¬ W
def statement_I (E W : Prop) : Prop := W → E
def statement_II (E W : Prop) : Prop := ¬ E → ¬ W
def statement_III (E W : Prop) : Prop := W → ¬ E
def statement_IV (E W : Prop) : Prop := ¬ E ∨ ¬ W

theorem logically_equivalent_to_original (E W : Prop) :
  (original_statement E W ↔ statement_III E W) ∧
  (original_statement E W ↔ statement_IV E W) :=
  sorry

end logically_equivalent_to_original_l548_548036


namespace correct_answer_l548_548261

def proposition1 : Prop :=
  (∀ x : ℝ, x^2 + 1 < 3 * x)

def proposition2 : Prop :=
  ∀ a : ℝ, (a > 2) → (a > 5)

def proposition3 : Prop :=
  ∀ x y : ℝ, (x * y = 0) → (x = 0 ∧ y = 0)

def proposition4 (p q : Prop) : Prop :=
  (¬(p ∨ q)) → (¬p ∧ ¬q)

def true_propositions_count : ℕ :=
  (if proposition1 then 1 else 0) +
  (if proposition2 then 1 else 0) +
  (if proposition3 then 1 else 0) +
  (if proposition4 false false then 1 else 0)

theorem correct_answer : true_propositions_count = 1 :=
sorry

end correct_answer_l548_548261


namespace circle_properties_l548_548269

noncomputable theory

open Real

theorem circle_properties (rho theta : ℝ) :
  rho^2 - 4 * sqrt 2 * rho * cos (theta - π / 4) + 6 = 0 →
  (∃ x y : ℝ, 
    (x - 2)^2 + (y - 2)^2 = 2 ∧ 
    ∀ θ : ℝ, x = 2 + sqrt 2 * cos θ ∧ y = 2 + sqrt 2 * sin θ ∧ 
    (∀ t : ℝ, t ∈ set.Icc (-sqrt 2) sqrt 2 → 
      (let xy := (t + sqrt 2)^2 + 1 in 
       (∃ xy_min xy_max : ℝ, xy_min = 1 ∧ xy_max = 9 ∧ xy_min ≤ xy ∧ xy ≤ xy_max)
      )
    )
  ) := sorry

end circle_properties_l548_548269


namespace perimeter_difference_l548_548064

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548064


namespace triangle_similar_l548_548727

variables {a b c m_a m_b m_c t : ℝ}

-- Define the triangle ABC and its properties
def triangle_ABC (a b c m_a m_b m_c t : ℝ) : Prop :=
  t = (1 / 2) * a * m_a ∧
  t = (1 / 2) * b * m_b ∧
  t = (1 / 2) * c * m_c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧
  t > 0

-- Define the similarity condition for the triangles
def similitude_from_reciprocals (a b c m_a m_b m_c t : ℝ) : Prop :=
  (1 / m_a) / (1 / m_b) = a / b ∧
  (1 / m_b) / (1 / m_c) = b / c ∧
  (1 / m_a) / (1 / m_c) = a / c

theorem triangle_similar (a b c m_a m_b m_c t : ℝ) :
  triangle_ABC a b c m_a m_b m_c t →
  similitude_from_reciprocals a b c m_a m_b m_c t :=
by
  intro h
  sorry

end triangle_similar_l548_548727


namespace integral_cos_squared_sin_l548_548926

theorem integral_cos_squared_sin :
  ∫ x in 0..1, cos x ^ 2 * sin x = -(cos 1) ^ 3 / 3 + (cos 0) ^ 3 / 3 :=
by
  -- placeholder for the actual proof
  sorry

end integral_cos_squared_sin_l548_548926


namespace cole_time_to_work_is_90_minutes_l548_548461

noncomputable def cole_drive_time_to_work (D : ℝ) : ℝ := D / 30

def cole_trip_proof : Prop :=
  ∃ (D : ℝ), (D / 30) + (D / 90) = 2 ∧ cole_drive_time_to_work D * 60 = 90

theorem cole_time_to_work_is_90_minutes : cole_trip_proof :=
  sorry

end cole_time_to_work_is_90_minutes_l548_548461


namespace ratio_of_means_l548_548907

theorem ratio_of_means (x y : ℝ) (h : (x + y) / (2 * Real.sqrt (x * y)) = 25 / 24) :
  (x / y = 16 / 9) ∨ (x / y = 9 / 16) :=
by
  sorry

end ratio_of_means_l548_548907


namespace complex_modulus_l548_548992

theorem complex_modulus (m n : ℝ) (hm_imag : ∀ (r : ℝ), log 2 (m^2 - 3 * m - 3) = r ∧ log 2 (m - 2) = 0) (line_eq : m + n = 2) : complex.abs (⟨m, n⟩ : ℂ) = 2 * sqrt 5 :=
by
  sorry

end complex_modulus_l548_548992


namespace sum_of_possible_values_of_N_l548_548562

theorem sum_of_possible_values_of_N :
  let N := {0, 1, 3, 4, 5, 6} in
  ∑ n in N, n = 19 :=
by
  let N := {0, 1, 3, 4, 5, 6}
  have hN : ∑ n in N, n = 19 := by sorry
  exact hN

end sum_of_possible_values_of_N_l548_548562


namespace dutyScheduleCount_l548_548773

noncomputable def countDutySchedules : ℕ :=
let people : Finset (Fin 3) := {0, 1, 2}
let days : Finset (Fin 6) := {0, 1, 2, 3, 4, 5}
let possibleSchedules := (days.powerset.filter (λ s => s.card = 2)).toFinset
let validSchedulesA := possibleSchedules.filter (λ s => 0 ∉ s)
let validSchedulesB := possibleSchedules.filter (λ s => 5 ∉ s)
let validSchedules := (validSchedulesA ∩ validSchedulesB).toFinset
validSchedules.card * 2 -- multiply by 2 as there are two schedules per person.

theorem dutyScheduleCount : countDutySchedules = 42 := by
  sorry

end dutyScheduleCount_l548_548773


namespace area_of_square_on_PS_l548_548657

theorem area_of_square_on_PS (PQ QR PR PS : ℝ) 
  (h₁ : PQ^2 = 25) (h₂ : QR^2 = 4) (h₃ : PR^2 = 49) 
  (h₄ : PR = Real.sqrt(49)) (h₅ : QR = Real.sqrt(4)) 
  (h₆ : PS = Real.sqrt(PR^2 + QR^2)) :
  PS^2 = 53 :=
by
  -- Here we would provide the proof steps, but it is omitted as the problem requires only the statement.
  sorry -- Placeholder to ensure lean code builds correctly.

end area_of_square_on_PS_l548_548657


namespace maximum_value_of_2x_plus_y_is_2sqrt10_div_5_l548_548697

noncomputable def max_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) : ℝ :=
  Sup { t : ℝ | ∃ x y : ℝ, t = 2*x + y ∧ 4*x^2 + y^2 + x*y = 1 }

theorem maximum_value_of_2x_plus_y_is_2sqrt10_div_5 (x y : ℝ) (h: 4 * x^2 + y^2 + x * y = 1) :
  max_value_of_2x_plus_y x y h = 2 * real.sqrt 10 / 5 := 
sorry

end maximum_value_of_2x_plus_y_is_2sqrt10_div_5_l548_548697


namespace polynomial_value_at_2_l548_548177

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define the transformation rules for each v_i according to Horner's Rule
def v0 : ℝ := 1
def v1 (x : ℝ) : ℝ := (v0 * x) - 12
def v2 (x : ℝ) : ℝ := (v1 x * x) + 60
def v3 (x : ℝ) : ℝ := (v2 x * x) - 160

-- State the theorem to be proven
theorem polynomial_value_at_2 : v3 2 = -80 := 
by 
  -- Since this is just a Lean 4 statement, we include sorry to defer proof
  sorry

end polynomial_value_at_2_l548_548177


namespace otimes_calculation_l548_548530

def otimes (x y : ℝ) : ℝ := x^2 - 2*y

theorem otimes_calculation (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k :=
by
  sorry

end otimes_calculation_l548_548530


namespace log_equation_exponentiation_equation_l548_548176

theorem log_equation :
  2 * log 25 / log 5 + 3 * log 64 / log 2 = 22 := sorry

theorem exponentiation_equation :
  125 ^ (2 / 3) + (1 / 2) ^ (-2) + 343 ^ (1 / 3) = 36 := sorry

end log_equation_exponentiation_equation_l548_548176


namespace cubes_sum_eq_cube_l548_548326

theorem cubes_sum_eq_cube (n : ℕ) (hn : n % 2 = 1) 
  (h : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^3 + y^3 + z^3 = (6^3)) :
  ∃ (a : Finset ℕ), a.card = n ∧ (∑ i in a, i^3) = k^3 ∧ ∀ x ∈ a, x ≠ y ∈ a ∧ x ≠ y :=
sorry

end cubes_sum_eq_cube_l548_548326


namespace probability_of_cosine_interval_l548_548143

def probability_condition (x : ℝ) : Prop :=
  x ∈ Icc 0 π ∧ -((real.sqrt 3) / 2) < real.cos x ∧ real.cos x < (real.sqrt 3) / 2

theorem probability_of_cosine_interval :
  let interval := Icc 0 π
  let favorableSet := { x : ℝ | probability_condition x }
  let totalLength := real.pi
  let favorableLength := (5 / 6) * real.pi - (1 / 6) * real.pi
  (favorableLength / totalLength) = 2 / 3 :=
by
  sorry

end probability_of_cosine_interval_l548_548143


namespace always_choose_disjoint_subsets_with_same_sum_l548_548849

theorem always_choose_disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), (∀ᵣ (n ∈ S), 10 ≤ n ∧ n < 100) ∧ S.card = 10 →
  ∃ (A B : Finset ℕ), A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end always_choose_disjoint_subsets_with_same_sum_l548_548849


namespace correct_conclusions_l548_548257

variable {d : ℝ}
variable {a1 a3 S8 S7 S12 : ℝ}

-- Given conditions:
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop := 
  ∀ n m : ℕ, a_n (n + 1) - a_n n = a_n (m + 1) - a_n m

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = ∑ i in finset.range n, a_n i

def condition (a1 a3 S8 : ℝ) : Prop :=
  a1 + 5 * a3 = S8

-- Prove the conclusions:
theorem correct_conclusions (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms a_n S_n)
  (h_cond : condition a1 a3 S8) :
  (a_n 10 = 0) ∧ (S_n 7 = S_n 12) :=
by 
  sorry

end correct_conclusions_l548_548257


namespace sum_of_valid_b_values_l548_548944

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l548_548944


namespace constant_term_in_expansion_l548_548924

theorem constant_term_in_expansion :
  let T (r : ℕ) : ℝ := (-3)^r * (Nat.choose 5 r) * x^((5:ℝ - 5*r)/4)
  in T 1 = -15 := by
  sorry

end constant_term_in_expansion_l548_548924


namespace monotonicity_interval_l548_548552

def t (x : ℝ) : ℝ := x^2 - 6 * x + 11

def y (x : ℝ) : ℝ := Real.log (t x) / Real.log (1 / 2)

theorem monotonicity_interval :
  ∃ a : ℝ, ∀ x : ℝ, x ∈ Iio a → MonotoneIncreasingOn y (Iio x) :=
sorry

end monotonicity_interval_l548_548552


namespace problem1_problem2_l548_548876

-- Problem 1
theorem problem1 (a b : ℝ)
  (hab : a >= 0) 
  (hbb : b > 0) :
  (a * 3 * b * sqrt a) / (b ^ (1/2)) = a ^ (3/2) * b ^ (1/2) :=
sorry

-- Problem 2
theorem problem2 :
  ( (9 / 4) ^ (1/2) - (-9.6) ^ 0 - (27 / 8) ^ (- 2 / 3) + 1.5 ^ (-2) ) = 1 / 2 :=
sorry

end problem1_problem2_l548_548876


namespace matches_needed_eq_l548_548491

def count_matches (n : ℕ) : ℕ :=
  let total_triangles := n * n
  let internal_matches := 3 * total_triangles
  let external_matches := 4 * n
  internal_matches - external_matches + external_matches

theorem matches_needed_eq (n : ℕ) : count_matches 10 = 320 :=
by
  sorry

end matches_needed_eq_l548_548491


namespace find_a_b_maximum_area_triangle_OAB_independent_pa2_pb2_l548_548668

noncomputable def ellipse_properties : Prop :=
  let a := 2
  let b := 1
  let c := sqrt 3
  let e := sqrt 3 / 2
  ∀ x y : ℝ, 
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (e = c / a) ∧
  ((abs (x - 0) ≤ a) → (abs (y - 0) ≤ b))

theorem find_a_b (hx : ellipse_properties) : ∃ a b : ℝ, a = 2 ∧ b = 1 :=
  begin
    sorry
  end

theorem maximum_area_triangle_OAB (k : ℝ) (hk : k = 1) : ∃ max_area : ℝ, max_area = 1 :=
  begin
    sorry
  end

theorem independent_pa2_pb2 (P : ℝ → ℝ) (h_ind : ∀ m : ℝ, (P m)^2 = (P (-m))^2) : ∃ k : ℝ, k = 1 / 2 ∨ k = -1 / 2 :=
  begin
    sorry
  end

end find_a_b_maximum_area_triangle_OAB_independent_pa2_pb2_l548_548668


namespace quadratic_inequality_solution_l548_548755

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 3 * x + 2 < 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l548_548755


namespace lattice_points_on_segment_l548_548280

theorem lattice_points_on_segment:
  let x1 := 5
  let y1 := 23
  let x2 := 53
  let y2 := 311
  let gcd_val := Nat.gcd (y2 - y1) (x2 - x1)
  let num_lattice_points := (x2 - x1) / gcd_val + 1 in
  num_lattice_points = 49 := 
by {
  -- Defining the coordinate differences
  let diff_x := x2 - x1,
  let diff_y := y2 - y1,
  
  -- Calculating the gcd of the differences
  let gcd_val := Nat.gcd diff_y diff_x,

  -- Defining the number of lattice points on the segment
  let num_lattice_points := diff_x / gcd_val + 1,
  
  -- The number of lattice points should be 49
  exact sorry,
}

end lattice_points_on_segment_l548_548280


namespace sum_of_b_values_l548_548954

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l548_548954


namespace area_triangle_PAF_l548_548239

-- Define the hyperbola and its properties
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

-- Define the points F, A, and P with their conditions
def F := (3, 0 : ℝ × ℝ)
def A := (0, 8 : ℝ × ℝ)
def P (y : ℝ) : ℝ × ℝ := (3, y)

-- Define the condition that P lies on the hyperbola
def P_on_hyperbola (y : ℝ) : Prop := hyperbola_eq 3 y

-- Define the perpendicular condition
def PF_perpendicular_x_axis (y : ℝ) : Prop := True

-- Define the coordinates of P constrained by the hyperbola
def P_coords : ℝ × ℝ := (3, 8)

-- Main theorem stating the area of triangle PAF
theorem area_triangle_PAF : 
  ∃ y : ℝ, P_on_hyperbola y ∧ PF_perpendicular_x_axis y → 
  let P := P_coords in
  let area := 1 / 2 * (A.fst - P.fst).abs * (P.snd - F.snd).abs in
  area = 12 :=
by
  sorry

end area_triangle_PAF_l548_548239


namespace plywood_cut_perimeter_difference_l548_548106

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548106


namespace nature_of_set_of_points_l548_548698

noncomputable def set_of_points (n : ℕ) (A : Fin n → ℝ × ℝ) (k : Fin n → ℝ) (C : ℝ) : set (ℝ × ℝ) :=
  { M : ℝ × ℝ | ∑ i in Finset.univ, k i * ((M.1 - (A i).1)^2 + (M.2 - (A i).2)^2) = C }

theorem nature_of_set_of_points {n : ℕ} (A : Fin n → ℝ × ℝ) (k : Fin n → ℝ) (C : ℝ) :
  (∑ i, k i ≠ 0 → (∃ r : ℝ, r > 0 ∧ ∃ center : ℝ × ℝ, set_of_points n A k C = {M : ℝ × ℝ | ∥M - center∥ = r} ∨ set_of_points n A k C = ∅ ∨ set_of_points n A k C = {M : ℝ × ℝ | M = center})) ∧
  (∑ i, k i = 0 → (∃ l : ℝ, ∃ a b : ℝ, set_of_points n A k C = {M : ℝ × ℝ | a * M.1 + b * M.2 + l = 0} ∨ set_of_points n A k C = set.univ)) :=
by
  sorry

end nature_of_set_of_points_l548_548698


namespace series_sum_l548_548559

theorem series_sum : (∑ k in range 1012, (if k % 2 = 0 then (2*k + 1) else -(2*k + 1))) + 2025 = 1013 :=
by
  sorry

end series_sum_l548_548559


namespace proper_sampling_method_l548_548440

-- Definitions for conditions
def large_bulbs : ℕ := 120
def medium_bulbs : ℕ := 60
def small_bulbs : ℕ := 20
def sample_size : ℕ := 25

-- Definition for the proper sampling method to use
def sampling_method : String := "Stratified sampling"

-- Theorem statement to prove the sampling method
theorem proper_sampling_method :
  ∃ method : String, 
  method = sampling_method ∧
  sampling_method = "Stratified sampling" := by
    sorry

end proper_sampling_method_l548_548440


namespace apple_consumption_l548_548867

-- Definitions for the portions of the apple above and below water
def portion_above_water := 1 / 5
def portion_below_water := 4 / 5

-- Rates of consumption by fish and bird
def fish_rate := 120  -- grams per minute
def bird_rate := 60  -- grams per minute

-- The question statements with the correct answers
theorem apple_consumption :
  (portion_below_water * (fish_rate / (fish_rate + bird_rate)) = 2 / 3) ∧ 
  (portion_above_water * (bird_rate / (fish_rate + bird_rate)) = 1 / 3) := 
sorry

end apple_consumption_l548_548867


namespace value_of_r_when_n_is_3_l548_548700

theorem value_of_r_when_n_is_3 :
  let r : ℕ := 4^s - 2*s,
      s : ℕ := 2^n + 2,
      n : ℕ := 3
  in r = 1048556 :=
by
  sorry

end value_of_r_when_n_is_3_l548_548700


namespace sum_of_b_values_l548_548955

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l548_548955


namespace rabbit_travel_time_l548_548841

theorem rabbit_travel_time (speed distance : ℝ) (h_speed : speed = 3) (h_distance : distance = 2) :
    let time_hours := distance / speed in
    let time_minutes := time_hours * 60 in
    time_minutes = 40 :=
by
  sorry

end rabbit_travel_time_l548_548841


namespace math_problem_l548_548880

theorem math_problem :
  -50 * 3 - (-2.5) / 0.1 = -125 := by
sorry

end math_problem_l548_548880


namespace quadratic_inequality_range_l548_548536

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end quadratic_inequality_range_l548_548536


namespace num_pairs_satisfying_equation_l548_548628

theorem num_pairs_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end num_pairs_satisfying_equation_l548_548628


namespace find_lambda_l548_548611

theorem find_lambda (a b : ℝ × ℝ) (λ : ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, 1)) 
  (h_perp : (a.1 + b.1, a.2 + b.2) ⊗ (a.1 - λ * b.1, a.2 - λ * b.2) = 0) :
  λ = 1 := sorry

end find_lambda_l548_548611


namespace F_at_3_l548_548265

noncomputable def f : ℝ → ℝ := sorry

def F (x : ℝ) : ℝ := f(x) + f(-x)

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom periodicity4 : ∀ x : ℝ, f(x) = f(x + 4)
axiom f_at_1 : f(1) = (Real.pi) / 3

theorem F_at_3 : F(3) = 2 * (Real.pi) / 3 :=
by
  sorry

end F_at_3_l548_548265


namespace find_reciprocal_l548_548701

open Real

theorem find_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^3 + y^3 + 1 / 27 = x * y) : 1 / x = 3 := 
sorry

end find_reciprocal_l548_548701


namespace larry_channels_l548_548685

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end larry_channels_l548_548685


namespace aquarium_water_depth_l548_548510

theorem aquarium_water_depth
  (width height base : ℝ)
  (tilted_cover_end : width = 10) 
  (tilted_cover_base : base = 7.5)
  (tilted_cover_height : height = 8) 
  (tilted_cover_volume : 300) 
  : ∃ h : ℝ, h = 3.75 :=
by
  sorry

end aquarium_water_depth_l548_548510


namespace savings_percentage_correct_l548_548828

variables (price_jacket : ℕ) (price_shirt : ℕ) (price_hat : ℕ)
          (discount_jacket : ℕ) (discount_shirt : ℕ) (discount_hat : ℕ)

def original_total_cost (price_jacket price_shirt price_hat : ℕ) : ℕ :=
  price_jacket + price_shirt + price_hat

def savings (price : ℕ) (discount : ℕ) : ℕ :=
  price * discount / 100

def total_savings (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  (savings price_jacket discount_jacket) + (savings price_shirt discount_shirt) + (savings price_hat discount_hat)

def total_savings_percentage (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  total_savings price_jacket price_shirt price_hat discount_jacket discount_shirt discount_hat * 100 /
  original_total_cost price_jacket price_shirt price_hat

theorem savings_percentage_correct :
  total_savings_percentage 100 50 30 30 60 50 = 4167 / 100 :=
sorry

end savings_percentage_correct_l548_548828


namespace range_of_a_l548_548600

-- Define the function f
def f (a x : ℝ) : ℝ := log a (a * x^2 - x + 1/2)

-- Main statement translating the problem into Lean
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ (Icc 1 2), f a x > 0) ↔ (a ∈ Ioo (1/2) (5/8) ∪ Ioi (3/2)) :=
by
  sorry

end range_of_a_l548_548600


namespace ellipse_C1_equation_max_triangle_area_l548_548669

open Real -- Explicitly opening the real number namespace

-- Definitions of ellipses and parabolas
def ellipse_eq (a b : ℝ) : Prop := 
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

def parabola_eq : ℝ → ℝ → Prop := 
  λ x y, (y = x^2)

def eccentricity (a b e : ℝ) : Prop := 
  e = sqrt 1 - (b^2 / a^2)

-- Conditions and correct answers
def ellipse_c1_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  eccentricity a b (sqrt 3 / 2) ∧ 
  2 * b = 2

def point_A (A : ℝ × ℝ) : Prop :=
  A = (0, 1/16)

def parabola_tangent_intersects_ellipse (a b t : ℝ) : Prop := 
  let N := (t, t^2) in
  ∃ B C : ℝ × ℝ, 
    -- N on parabola
    (parabola_eq N.1 N.2) ∧
    -- Linear equation of BC
    (let x := B.1 in
     let y := B.2 in
     let x' := C.1 in
     let y' := C.2 in
     ∃ m c : ℝ, (y = m * x + c) ∧ (y' = m * x' + c) ∧
     -- Ensure intersection with ellipse
     ellipse_eq a b x y ∧ ellipse_eq a b x' y')

noncomputable def max_area_of_triangle_ABC (a b : ℝ) (t : ℝ) : ℝ :=
  let N := (t, t^2) in
  let A := point_A A in
  let B_intersect := (λ x, ∃ y, ellipse_eq a b x y) in
  let BC_eq := parabola_tangent_intersects_ellipse a b t in
  let d := 1 + 16 * t^2 / 16 * sqrt (1 + 4 * t^2) in
  (1/2) * (1 + 4 * t^2) * sqrt (- (t^2 - 8)^2 + 65) / 16

theorem ellipse_C1_equation : 
  ∃ a b : ℝ, ellipse_c1_conditions a b →
  ellipse_eq a b :=
  begin
    intros a b h,
    sorry
  end

theorem max_triangle_area : 
  ∃ a b : ℝ, ∃ t : ℝ, 
  ellipse_c1_conditions a b →
  max_area_of_triangle_ABC a b t = sqrt(65) / 8 :=
  begin
    intros a b t h,
    sorry
  end

end ellipse_C1_equation_max_triangle_area_l548_548669


namespace interval_solution_l548_548548

open set

theorem interval_solution (x : ℝ) : 
  { x | (x - 1) / (x - 3) ≥ 3 } = Ioo 3 4 ∪ {4} :=
by
  sorry

end interval_solution_l548_548548


namespace solution_l548_548131

def seventh_grade_scores := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

theorem solution : 
  (mode seventh_grade_scores = 6) ∧ 
  (average eighth_grade_scores = 7.1) ∧ 
  (performance 7 seventh_grade_scores = "above average") ∧ 
  (performance 7 eighth_grade_scores = "below average") := 
  by
    sorry

end solution_l548_548131


namespace part_time_job_pay_per_month_l548_548875

def tuition_fee : ℝ := 90
def scholarship_percent : ℝ := 0.30
def scholarship_amount := scholarship_percent * tuition_fee
def amount_after_scholarship := tuition_fee - scholarship_amount
def remaining_amount : ℝ := 18
def months_to_pay : ℝ := 3
def amount_paid_so_far := amount_after_scholarship - remaining_amount

theorem part_time_job_pay_per_month : amount_paid_so_far / months_to_pay = 15 := by
  sorry

end part_time_job_pay_per_month_l548_548875


namespace additional_bluray_movies_purchased_l548_548142

-- Definitions of the problem conditions
variables (x y : ℕ)
variable (initial_total : ℕ)
variable (initial_ratio_dvd_bluray : ℕ → ℕ → Prop)
variable (final_ratio_dvd_bluray : ℕ → ℕ → Prop)

-- Helper definitions for ratios
def ratio (a b : ℕ) := ∃ k : ℕ, k * b = a

-- Problem conditions
def initial_state := initial_ratio_dvd_bluray 7 2
def final_state := final_ratio_dvd_bluray 13 4
def initial_sum := initial_total = 351
def total_dvd_bluray := 7 * x + 2 * x = initial_total

-- Lean problem statement
theorem additional_bluray_movies_purchased 
  (h_initial_ratio : initial_state)
  (h_final_ratio : final_state)
  (h_initial_sum : initial_sum)
  (h_total_dvd_bluray : total_dvd_bluray) :
  y = 6 :=
sorry

end additional_bluray_movies_purchased_l548_548142


namespace trigonometric_identity_l548_548246

theorem trigonometric_identity (x : ℝ) (h₁ : Real.sin x = 4 / 5) (h₂ : π / 2 ≤ x ∧ x ≤ π) :
  Real.cos x = -3 / 5 ∧ (Real.cos (-x) / (Real.sin (π / 2 - x) - Real.sin (2 * π - x)) = -3) := 
by
  sorry

end trigonometric_identity_l548_548246


namespace halfway_fraction_l548_548203

def one_fourth := (1 : ℝ) / 4
def one_half := (1 : ℝ) / 2
def three_eighths := (3 : ℝ) / 8

theorem halfway_fraction : (one_fourth + one_half) / 2 = three_eighths := 
by 
    sorry

end halfway_fraction_l548_548203


namespace exists_sameColorNeighbors_50x50_l548_548406

-- Definitions of the grid and coloring.
def GridColoring (n : ℕ) := fin (n * n) → fin 4

-- Predicate to check if a cell at position (i, j) has all four neighbors of the same color.
def sameColorNeighbors (n : ℕ) (grid : GridColoring n) (i j : fin n) : Prop :=
  ∀ di dj, di ∈ [{-1}, {1}] → dj ∈ [{-1}, {1}] → ∃ (i' j' : ℕ), 
  i' = n + i.val + di ∧ j' = n + j.val + dj →
  grid ⟨n * j' + i', sorry⟩ = grid ⟨n * j + i, sorry⟩

-- Main theorem to prove.
theorem exists_sameColorNeighbors_50x50 :
  ∃ i j : fin 50, ∃ (grid : GridColoring 50), sameColorNeighbors 50 grid i j :=
sorry

end exists_sameColorNeighbors_50x50_l548_548406


namespace plywood_cut_difference_l548_548118

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548118


namespace _l548_548796

noncomputable theorem time_to_finish_job :
  let P_rate := 1 / 3 
  let Q_rate := 1 / 15
  let combined_rate := P_rate + Q_rate
  let time_worked_together := 2
  let job_completed_together := combined_rate * time_worked_together
  let remaining_job := 1 - job_completed_together
  let P_time_to_finish := (remaining_job / P_rate)
  let P_time_to_finish_minutes := P_time_to_finish * 60
  P_time_to_finish_minutes = 36 := by
sorry

end _l548_548796


namespace CoCurcular_Equivalence_l548_548400

-- Definitions for points and the relevant properties
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Line :=
  (p1 : Point)
  (p2 : Point)

def concyclic (A B C D : Point) : Prop :=
  ∃ (O : Point) (r : ℝ), Circle.center = O ∧ Circle.radius = r ∧
  (sqrt ((A.x - O.x)^2 + (A.y - O.y)^2) = r) ∧ 
  (sqrt ((B.x - O.x)^2 + (B.y - O.y)^2) = r) ∧ 
  (sqrt ((C.x - O.x)^2 + (C.y - O.y)^2) = r) ∧
  (sqrt ((D.x - O.x)^2 + (D.y - O.y)^2) = r)

def equal_angles (A B C D : Point) : Prop :=
  ∀ α β, ∃ (line_ab : Line), α = ∠ (A B C) ∧ β = ∠ (A B D) ∧
  C ∈ same_side (line_ab, A, B) ∧ 
  D ∈ same_side (line_ab, A, B) ∧ 
  (α = β)

def supplementary_angles (A B C D : Point) : Prop :=
  ∀ α β, ∃ (line_ab : Line), α = ∠ (B A D) ∧ β = ∠ (D C B) ∧ 
  (α + β = π)

def product_of_segments_equal (A B C D : Point) : Prop :=
  ∃ P, ((A.x * P.x + C.x) = (B.x * P.x + D.x))

def combined (A B C D : Point) : Prop :=
(concyclic A B C D) ∧ 
(equal_angles A B C D) ∧ 
(supplementary_angles A B C D) ∧
(product_of_segments_equal A B C D)
    
theorem CoCurcular_Equivalence 
  (A B C D : Point) : 
  combined A B C D :=
sorry

end CoCurcular_Equivalence_l548_548400


namespace number_of_nurses_l548_548763

variables (D N : ℕ)

-- Condition: The total number of doctors and nurses is 250
def total_staff := D + N = 250

-- Condition: The ratio of doctors to nurses is 2 to 3
def ratio_doctors_to_nurses := D = (2 * N) / 3

-- Proof: The number of nurses is 150
theorem number_of_nurses (h1 : total_staff D N) (h2 : ratio_doctors_to_nurses D N) : N = 150 :=
sorry

end number_of_nurses_l548_548763


namespace power_mod_remainder_l548_548931

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l548_548931


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548022

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548022


namespace total_flags_l548_548192

theorem total_flags (x : ℕ) (hx1 : 4 * x + 20 > 8 * (x - 1)) (hx2 : 4 * x + 20 < 8 * x) : 4 * 6 + 20 = 44 :=
by sorry

end total_flags_l548_548192


namespace sum_of_possible_values_of_N_l548_548561

theorem sum_of_possible_values_of_N :
  let N := {0, 1, 3, 4, 5, 6} in
  ∑ n in N, n = 19 :=
by
  let N := {0, 1, 3, 4, 5, 6}
  have hN : ∑ n in N, n = 19 := by sorry
  exact hN

end sum_of_possible_values_of_N_l548_548561


namespace cube_roots_sum_l548_548525

theorem cube_roots_sum (r s t : ℝ) (α β γ : ℝ)
  (h_eqn : (x - α) * (x - β) * (x - γ) = (x - r) * (x - s) * (x - t) + (1/3))
  (h1 : α = real.cbrt 23)
  (h2 : β = real.cbrt 63)
  (h3 : γ = real.cbrt 113) :
  r^3 + s^3 + t^3 = 200 := 
sorry

end cube_roots_sum_l548_548525


namespace triangle_altitude_length_l548_548978

-- Define the problem
theorem triangle_altitude_length (l w h : ℝ) (hl : l = 2 * w) 
  (h_triangle_area : 0.5 * l * h = 0.5 * (l * w)) : h = w := 
by 
  -- Use the provided conditions and the equation setup to continue the proof
  sorry

end triangle_altitude_length_l548_548978


namespace H_paths_bound_l548_548976

-- Define the graph G and subgraph H
variables (G : Type) [graph G] (H : subgraph G) 

-- Define kappa_G(H) as the number of independent H-paths
def kappa_G (H : subgraph G) : ℕ := sorry

-- Define lambda_G(H) as the number of edge-disjoint H-paths
def lambda_G (H : subgraph G) : ℕ := sorry

theorem H_paths_bound :
  (∃ k : ℕ, k >= ⌊ (kappa_G H) / 2 ⌋ ) ∧ 
  (∃ l : ℕ, l >= ⌊ (lambda_G H) / 2 ⌋ ) :=
by {
  sorry,
}

end H_paths_bound_l548_548976


namespace magnitude_difference_l548_548277

open Real EuclideanGeometry

variables (a b : ℝ^3) (angle_ab : ℝ)
def magnitude (v : ℝ^3) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
noncomputable def dot_product (u v : ℝ^3) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Given conditions
axiom mag_a : magnitude a = 1
axiom mag_b : magnitude b = 2
axiom angle_ab_val : angle_ab = π / 3

theorem magnitude_difference : magnitude (a - b) = Real.sqrt 3 := sorry

end magnitude_difference_l548_548277


namespace common_non_integer_root_eq_l548_548377

theorem common_non_integer_root_eq (p1 p2 q1 q2 : ℤ) 
  (x : ℝ) (hx1 : x^2 + p1 * x + q1 = 0) (hx2 : x^2 + p2 * x + q2 = 0) 
  (hnint : ¬ ∃ (n : ℤ), x = n) : p1 = p2 ∧ q1 = q2 :=
sorry

end common_non_integer_root_eq_l548_548377


namespace abs_diff_p_q_eq_zero_l548_548350

theorem abs_diff_p_q_eq_zero :
  let p := Nat.find (λ n, 1000 ≤ n ∧ n % 13 = 3)
  let q := Nat.find (λ n, 1000 ≤ n ∧ n % 7 = 3)
  |p - q| = 0 :=
by
  sorry

end abs_diff_p_q_eq_zero_l548_548350


namespace min_product_roots_l548_548271

theorem min_product_roots 
  (m : ℝ) 
  (h_discriminant_nonnegative : 36 - 20 * m ≥ 0) :
  (∃ m : ℝ, m = 9 / 5 ∧ (∀ n : ℝ, 36 - 20*n ≥ 0 → n/5 ≥ (9/5) / 5)) :=
begin
  sorry
end

end min_product_roots_l548_548271


namespace gopi_servant_salary_l548_548612

theorem gopi_servant_salary (S : ℝ) (h1 : 9 / 12 * S + 110 = 150) : S = 200 :=
by
  sorry

end gopi_servant_salary_l548_548612


namespace real_solutions_to_system_l548_548922

theorem real_solutions_to_system (x y : ℝ) (h1 : x^3 + y^3 = 1) (h2 : x^4 + y^4 = 1) :
  (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end real_solutions_to_system_l548_548922


namespace range_of_k_for_ellipse_l548_548292

def represents_ellipse (x y k : ℝ) : Prop :=
  (k^2 - 3 > 0) ∧ 
  (k - 1 > 0) ∧ 
  (k - 1 ≠ k^2 - 3)

theorem range_of_k_for_ellipse (k : ℝ) : 
  represents_ellipse x y k → k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 :=
by
  sorry

end range_of_k_for_ellipse_l548_548292


namespace problem_l548_548993

theorem problem (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a^4 + b^4 = 228) :
  a * b = 8 :=
sorry

end problem_l548_548993


namespace regular_polygon_perimeter_l548_548150

theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) 
  (h1 : 360 / exterior_angle = n) (h2 : 20 = exterior_angle)
  (h3 : 10 = side_length) : 180 = n * side_length :=
by
  sorry

end regular_polygon_perimeter_l548_548150


namespace circumscribed_triangle_area_relation_l548_548478

theorem circumscribed_triangle_area_relation
  (a b c D E F : ℝ)
  (h₁ : a = 18) (h₂ : b = 24) (h₃ : c = 30)
  (triangle_right : a^2 + b^2 = c^2)
  (triangle_area : (1/2) * a * b = 216)
  (circle_area : π * (c / 2)^2 = 225 * π)
  (non_triangle_areas : D + E + 216 = F) :
  D + E + 216 = F :=
by
  sorry

end circumscribed_triangle_area_relation_l548_548478


namespace non_zero_const_c_l548_548415

theorem non_zero_const_c (a b c x1 x2 : ℝ) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) 
(h3 : (a - 1) * x1 ^ 2 + b * x1 + c = 0) 
(h4 : (a - 1) * x2 ^ 2 + b * x2 + c = 0)
(h5 : x1 * x2 = -1) 
(h6 : x1 ≠ x2) 
(h7 : x1 * x2 < 0): c ≠ 0 :=
sorry

end non_zero_const_c_l548_548415


namespace number_of_routes_A_to_B_l548_548871

theorem number_of_routes_A_to_B :
  (∃ f : ℕ × ℕ → ℕ,
  (∀ n m, f (n + 1, m) = f (n, m) + f (n + 1, m - 1)) ∧
  f (0, 0) = 1 ∧ 
  (∀ i, f (i, 0) = 1) ∧ 
  (∀ j, f (0, j) = 1) ∧ 
  f (3, 5) = 23) :=
sorry

end number_of_routes_A_to_B_l548_548871


namespace find_a_l548_548196

noncomputable def exists_nonconstant_function (a : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 ≠ f x2) ∧ 
  (∀ x : ℝ, f (a * x) = a^2 * f x) ∧
  (∀ x : ℝ, f (f x) = a * f x)

theorem find_a :
  ∀ (a : ℝ), exists_nonconstant_function a → (a = 0 ∨ a = 1) :=
by
  sorry

end find_a_l548_548196


namespace overlap_area_rhombus_l548_548897

noncomputable def area_of_overlap (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  1 / (Real.sin (α / 2))

theorem overlap_area_rhombus (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  area_of_overlap α hα = 1 / (Real.sin (α / 2)) :=
sorry

end overlap_area_rhombus_l548_548897


namespace at_least_one_negative_root_l548_548837

-- Define a quadratic polynomial with two distinct roots and the condition given
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

axiom distinct_roots {a b c : ℝ} (h : a ≠ 0) : quadratic_polynomial a b c has_two_distinct_roots

axiom quadratic_inequality_holds {a b c : ℝ} (h : a ≠ 0) :
  ∀ (x y : ℝ), quadratic_polynomial a b c (x^2 + y^2) ≥ quadratic_polynomial a b c (2 * x * y)

-- State the theorem: If the above conditions hold, then at least one root is negative
theorem at_least_one_negative_root {a b c : ℝ} (h_a : a ≠ 0) (h1 : distinct_roots h_a) (h2 : quadratic_inequality_holds h_a) :
  ∃ r : ℝ, is_root (quadratic_polynomial a b c) r ∧ r < 0 := sorry

end at_least_one_negative_root_l548_548837


namespace arithmetic_sequence_sum_l548_548588

variable (S : ℕ → ℝ)
variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_sum (h₁ : S 5 = 8) (h₂ : S 10 = 20) : S 15 = 36 := 
by
  sorry

end arithmetic_sequence_sum_l548_548588


namespace integer_triplets_satisfy_eq_l548_548187

theorem integer_triplets_satisfy_eq {x y z : ℤ} : 
  x^2 + y^2 + z^2 - x * y - y * z - z * x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) := 
by
  sorry

end integer_triplets_satisfy_eq_l548_548187


namespace max_value_sqrt_expression_l548_548352

theorem max_value_sqrt_expression 
  (x y z : ℝ)
  (h1 : x + y + z = 2)
  (h2 : x ≥ -1/2)
  (h3 : y ≥ -2)
  (h4 : z ≥ -3)
  (h5 : 2x + y = 1) : 
  sqrt (4 * x + 2) + sqrt (3 * y + 6) + sqrt (4 * z + 12) ≤ sqrt 68 := 
  sorry

end max_value_sqrt_expression_l548_548352


namespace exists_k_d_l548_548982

open Nat

theorem exists_k_d (a : ℕ → ℕ) 
  (h_cond: ∀ m n : ℕ, m > 0 → n > 0 → (∑ i in range (2 * m), a (i * n)) ≤ m) :
  ∃ k d : ℕ, k > 0 ∧ d > 0 ∧ (∑ i in range (2 * k), a (i * d)) = k - 2014 :=
by
  sorry

end exists_k_d_l548_548982


namespace find_h_l548_548344

noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...))))

theorem find_h (h : ℝ) : bowtie 5 h = 11 -> h = 30 :=
by {
  sorry
}

end find_h_l548_548344


namespace exterior_angle_BAC_l548_548499

theorem exterior_angle_BAC (square_octagon_coplanar : Prop) (common_side_AD : Prop) : 
    angle_BAC = 135 :=
by
  sorry

end exterior_angle_BAC_l548_548499


namespace max_sum_of_arithmetic_sequence_l548_548671

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (a1 : a 1 = 29) 
  (S10_eq_S20 : S 10 = S 20) :
  (∃ n, ∀ m, S n ≥ S m) ∧ ∃ n, (S n = S 15) :=
sorry

end max_sum_of_arithmetic_sequence_l548_548671


namespace arithmetic_sequence_problem_l548_548587

noncomputable def a1 := 3
noncomputable def S (n : ℕ) (a1 d : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_problem (d : ℕ) 
  (h1 : S 1 a1 d = 3) 
  (h2 : S 1 a1 d / 2 + S 4 a1 d / 4 = 18) : 
  S 5 a1 d = 75 :=
sorry

end arithmetic_sequence_problem_l548_548587


namespace no_distinct_positive_integers_2007_l548_548378

theorem no_distinct_positive_integers_2007 (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  ¬ (x^2007 + y! = y^2007 + x!) :=
by
  sorry

end no_distinct_positive_integers_2007_l548_548378


namespace triangle_AB_length_l548_548748

-- Define the necessary entities and conditions
variables {A B C M K P : Type} [Triangle A B C] [Median A M] [AngleBisector B K]
variable [IsPerpendicular (A M) (B K)]
variable {BC : ℝ} (hBC : BC = 12)

-- Define the proof problem statement
theorem triangle_AB_length : ℝ :=
  ∃ 6 (AB : ℝ) := AB = 6 -> BC = 12 -> IsPerpendicular (A M) (B K).

end triangle_AB_length_l548_548748


namespace total_fruits_purchased_l548_548475

-- Defining the costs of apples and bananas
def cost_per_apple : ℝ := 0.80
def cost_per_banana : ℝ := 0.70

-- Defining the total cost the customer spent
def total_cost : ℝ := 6.50

-- Defining the total number of fruits purchased as 9
theorem total_fruits_purchased (A B : ℕ) : 
  (cost_per_apple * A + cost_per_banana * B = total_cost) → 
  (A + B = 9) :=
by
  sorry

end total_fruits_purchased_l548_548475


namespace number_of_pairs_l548_548626

theorem number_of_pairs :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end number_of_pairs_l548_548626


namespace find_correct_value_l548_548813

theorem find_correct_value (k : ℕ) (h1 : 173 * 240 = 41520) (h2 : 41520 / 48 = 865) : k * 48 = 173 * 240 → k = 865 :=
by
  intros h
  sorry

end find_correct_value_l548_548813


namespace calculate_path_length_l548_548822

-- Define initial conditions for the cube and the dot
structure Cube where
  edge_length : ℝ

structure Dot where
  initial_position : ℝ × ℝ -- coordinates in cm

-- Define the conditions for the rolling cube
structure RollingCube where
  cube : Cube
  dot : Dot
  roll_sequence_complete : Bool -- the dot returns to its top position (True)

-- Define a path length function (stub for now)
noncomputable def path_length (rolling_cube : RollingCube) : ℝ :=
  if rolling_cube.roll_sequence_complete then
    sqrt 5 * Real.pi
  else 0

-- Formal statement to prove
theorem calculate_path_length :
  ∀ (cub : Cube) (dot : Dot),
    (cub.edge_length = 2) →
    (dot.initial_position = (1, 2)) →
    let rolling_cube : RollingCube := ⟨cub, dot, true⟩
    path_length rolling_cube = sqrt 5 * Real.pi :=
by
  intros
  sorry

end calculate_path_length_l548_548822


namespace tim_points_l548_548044

theorem tim_points (J T K : ℝ) (h1 : T = J + 20) (h2 : T = K / 2) (h3 : J + T + K = 100) : T = 30 := 
by 
  sorry

end tim_points_l548_548044


namespace smallest_lambda_inequality_l548_548208

theorem smallest_lambda_inequality (n : ℕ) (a : Fin n → ℕ) (h_odd : n % 2 = 1) (h_pos : ∀ i, 0 < a i) :
  ∃ λ, λ = 2 ∧
  (∀ i, (a i)^n + ∏ j, a j ≤ λ * (a i)^n) := sorry

end smallest_lambda_inequality_l548_548208


namespace smallest_positive_multiple_of_6_and_15_gt_40_l548_548556

-- Define the LCM function to compute the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Define the statement of the proof problem
theorem smallest_positive_multiple_of_6_and_15_gt_40 : 
  ∃ a : ℕ, (a % 6 = 0) ∧ (a % 15 = 0) ∧ (a > 40) ∧ (∀ b : ℕ, (b % 6 = 0) ∧ (b % 15 = 0) ∧ (b > 40) → a ≤ b) :=
sorry

end smallest_positive_multiple_of_6_and_15_gt_40_l548_548556


namespace intersection_points_of_circle_and_vertical_line_l548_548901

theorem intersection_points_of_circle_and_vertical_line :
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y1) ≠ (3, y2)) := 
by
  sorry

end intersection_points_of_circle_and_vertical_line_l548_548901


namespace symmetry_about_origin_l548_548249

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define the function v based on f and g
def v (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x * |g x|

-- The theorem statement
theorem symmetry_about_origin (f g : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even g) : 
  ∀ x : ℝ, v f g (-x) = -v f g x := 
by
  sorry

end symmetry_about_origin_l548_548249


namespace plywood_cut_difference_l548_548124

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548124


namespace round_4_85_l548_548383

-- Define the input value
def input_value : Float := 4.85

-- Define the rounding function according to the given problem
def round_to_nearest_even_tenth (x : Float) : Float :=
  let tenths := Float.floor (10 * x) / 10
  let next_digit := (10 * x) % 1
  if next_digit < 0.5 then tenths
  else if next_digit > 0.5 then tenths + 0.1
  else if (Float.floor (10 * tenths)) % 2 == 0 then tenths
  else tenths + 0.1

-- Assertion statement
theorem round_4_85 : round_to_nearest_even_tenth input_value = 5.0 := by
  sorry

end round_4_85_l548_548383


namespace prove_parallel_prove_perpendicular_l548_548245

variables (a b : ℝ)

-- Direction vector of line l
def m : ℝ × ℝ × ℝ := (1, a + b, a - b)

-- Normal vector of plane α
def n : ℝ × ℝ × ℝ := (1, 2, 3)

-- The line l is parallel to the plane α
def l_parallel_to_alpha : Prop := m a b.1 * n.1 + m a b.2 * n.2 + m a b.3 * n.3 = 0

-- The line l is perpendicular to the plane α
def l_perpendicular_to_alpha : Prop := m a b = n

theorem prove_parallel : l_parallel_to_alpha a b → (5 * a - b + 1 = 0) :=
by sorry

theorem prove_perpendicular : l_perpendicular_to_alpha a b → (a + b = 2) ∧ (a - b = 3) :=
by sorry

end prove_parallel_prove_perpendicular_l548_548245


namespace volume_of_space_inside_sphere_outside_cylinder_l548_548846

theorem volume_of_space_inside_sphere_outside_cylinder :
  (let r_sphere := 6 in
   let r_cylinder_base := 4 in
   let h_cylinder := 4 * Real.sqrt 5 in
   let V_sphere := (4 / 3) * Real.pi * r_sphere^3 in
   let V_cylinder := Real.pi * r_cylinder_base^2 * h_cylinder in
   V_sphere - V_cylinder = (288 - 64 * Real.sqrt 5) * Real.pi) :=
sorry

end volume_of_space_inside_sphere_outside_cylinder_l548_548846


namespace find_tangent_m_value_l548_548963

-- Define the line equation
def line (x m : ℝ) := 2*x + m

-- Define the curve equation
def curve (x : ℝ) := x * Real.log x

-- The tangent line condition
def isTangentAt (x0 m : ℝ) := deriv curve x0 = 2 ∧ curve x0 = line x0 m

-- Prove that the value of m for which the line y = 2x + m is tangent to the curve y = x ln x is -e
theorem find_tangent_m_value : ∃ m, isTangentAt Real.exp m ∧ m = -Real.exp :=
by 
  sorry

end find_tangent_m_value_l548_548963


namespace solve_z_6_eq_neg64_l548_548934

noncomputable def complex_solutions (k : ℕ) (hk : k < 6) :=
  2 * complex.exp (complex.I * (π / 6 + 2 * k * π / 6))

theorem solve_z_6_eq_neg64 : 
  {z : ℂ | z^6 = -64} = 
  {complex_solutions k (by norm_num : k < 6) | k : ℕ ∧ k < 6} :=
sorry

end solve_z_6_eq_neg64_l548_548934


namespace bus_driver_total_earnings_l548_548818

noncomputable def regular_rate : ℝ := 20
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours : ℝ := 45.714285714285715
noncomputable def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
noncomputable def overtime_hours : ℝ := total_hours - regular_hours
noncomputable def regular_pay : ℝ := regular_rate * regular_hours
noncomputable def overtime_pay : ℝ := overtime_rate * overtime_hours
noncomputable def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_total_earnings :
  total_compensation = 1000 :=
by
  sorry

end bus_driver_total_earnings_l548_548818


namespace find_pairs_l548_548355

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ q r : ℕ, a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  (a, b) = (50, 37) ∨ (a, b) = (37, 50) ∨ (a, b) = (50, 7) ∨ (a, b) = (7, 50) :=
by
  sorry

end find_pairs_l548_548355


namespace value_of_g_at_neg1_l548_548288

def g (x : ℝ) : ℝ := (2 * x - 3) / (5 * x + 2)

theorem value_of_g_at_neg1 : g (-1) = 5 / 3 :=
by
  sorry

end value_of_g_at_neg1_l548_548288


namespace fraction_of_innocent_cases_l548_548481

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end fraction_of_innocent_cases_l548_548481


namespace sum_a_values_eq_4_l548_548188

theorem sum_a_values_eq_4 (a_2 a_3 a_4 a_5 a_6 : ℤ) 
  (h1 : 0 ≤ a_2 ∧ a_2 < 3)
  (h2 : 0 ≤ a_3 ∧ a_3 < 4)
  (h3 : 0 ≤ a_4 ∧ a_4 < 5)
  (h4 : 0 ≤ a_5 ∧ a_5 < 6)
  (h5 : 0 ≤ a_6 ∧ a_6 < 7)
  (h_eqn : (3 / 5 : ℚ) = a_2 / 2! + a_3 / 3! + a_4 / 4! + a_5 / 5! + a_6 / 6!) :
  a_2 + a_3 + a_4 + a_5 + a_6 = 4 :=
by
  sorry

end sum_a_values_eq_4_l548_548188


namespace circle_radius_m_eq_circle_line_m_eq_l548_548599

theorem circle_radius_m_eq (m : ℝ) :
  (∃ r : ℝ, r = 2 ∧ ∃ x y : ℝ, (x^2 - y^2 - 2 * x - 4 * y + m = 0 ∧ (x-1)^2 + (y-2)^2 = 5 - m)) →
  (m = 1) :=
by
  sorry

theorem circle_line_m_eq (m : ℝ) :
  (∃ x y : ℝ, (x^2 - y^2 - 2x - 4y + m = 0 ∧ 
    (∃ u v : ℝ, ((u-1)^2 + (v-2)^2 ≤ 5 - m) ∧ (x + 2 * y - 4 = 0) ∧ (√((u - 1)^2 + (v - 2)^2) = (4 * √5) / 5)))) →
  (m = 4) :=
by
  sorry

end circle_radius_m_eq_circle_line_m_eq_l548_548599


namespace matt_profit_trade_l548_548711

theorem matt_profit_trade
  (total_cards : ℕ := 8)
  (value_per_card : ℕ := 6)
  (traded_cards_count : ℕ := 2)
  (trade_value_per_card : ℕ := 6)
  (received_cards_count_1 : ℕ := 3)
  (received_value_per_card_1 : ℕ := 2)
  (received_cards_count_2 : ℕ := 1)
  (received_value_per_card_2 : ℕ := 9)
  (profit : ℕ := 3) :
  profit = (received_cards_count_1 * received_value_per_card_1 
           + received_cards_count_2 * received_value_per_card_2) 
           - (traded_cards_count * trade_value_per_card) :=
  by
  sorry

end matt_profit_trade_l548_548711


namespace analytical_expression_of_f_range_of_f_A_in_acute_triangle_l548_548576

-- First part: Proving the analytical expression of f(x)
theorem analytical_expression_of_f (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ)
    (h1 : f = sin (ω • id + φ))
    (h2 : ω > 0)
    (h3 : abs φ < (π / 2))
    (h4 : ∀ x, f (x + (π / 2)) = -f x)
    (h5 : ∀ x, f (x - (π / 6)) = odd_fun (f x)) :
  f = sin (2 • id - (π / 3)) :=
sorry

-- Second part: Finding the range of values for f(A) in an acute triangle
theorem range_of_f_A_in_acute_triangle (a b c A B C : ℝ)
    (h1 : 0 < A ∧ A < π / 2)
    (h2 : 0 < B ∧ B < π / 2)
    (h3 : 0 < C ∧ C < π / 2)
    (h4 : a = 2 * c - a)
    (h5 : b = b)
    (h6 : cos B ≠ 0)
    (h7 : (2 * c - a) * cos B = b * cos A) :
  sin (2 * A - (π / 3)) ∈ (0, 1] :=
sorry

end analytical_expression_of_f_range_of_f_A_in_acute_triangle_l548_548576


namespace number_of_good_subsets_of_1_to_13_l548_548361

open Finset

theorem number_of_good_subsets_of_1_to_13 : 
  let n := 13
  let S := range (n+1) \ 0
  have cardinality_of_S : S.card = n := sorry
  have sum_S_is_odd : (S.sum id) % 2 = 1 := sorry
  ∃ (s : Finset ℕ), s ⊂ S ∧ (s.sum id) % 2 = 0 ∧ s ≠ ∅ → 
    (∃ (k : ℕ), k = 2 ^ (n - 1) - 1)
:=
begin
  sorry
end

end number_of_good_subsets_of_1_to_13_l548_548361


namespace volume_of_pyramid_l548_548320

structure RectangularParallelepiped :=
  (AB BC CG : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 2)
  (hCG : CG = 3)
  (N_is_midpoint : ∃ N : ℝ × ℝ × ℝ, N = ((4 + 0) / 2, (2 + 0) / 2, (3 + 0) / 2))

def volume_pyramid (base_area height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem volume_of_pyramid (p : RectangularParallelepiped) :
  ∃ (volume : ℝ),
    volume = volume_pyramid (4 * √20) (1.5) ∧
    volume = 2 * √5 :=
by
  use 2 * √5
  split
  { simp only [volume_pyramid, Real.mul_self_sqrt, mul_one_div],
    norm_num,
    rw Real.mul_one_div,
    congr' 1,
    norm_num,
    rw [← Real.sqrt_div, Real.of_real_mul, ← Real.sqrt_mul],
    exact sqrt_eq_rfl.mpr (pow_nonneg (by norm_num) 20),
    exact Real.sqrt_eq_rfl.mpr (by norm_num : (1 : ℝ) ≤ 20),
    sorry },
  { refl }

end volume_of_pyramid_l548_548320


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548024

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548024


namespace rectangular_solid_surface_area_l548_548539

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l548_548539


namespace length_of_segment_AC_l548_548180

theorem length_of_segment_AC:
  ∀ (K : Type) (circumference : ℝ) (A B C : K) (diameter : ℝ) (angle_AKC : ℝ),
  circumference = 18 * Real.pi →
  diameter = 2 * (18 * Real.pi / (2 * Real.pi)) →
  angle_AKC = 45 →
  ∃ (length_AC : ℝ), 
  length_AC = 9 * Real.sqrt 2 :=
by
  intros K circumference A B C diameter angle_AKC h1 h2 h3
  use 9 * Real.sqrt 2
  sorry

end length_of_segment_AC_l548_548180


namespace simple_interest_true_discount_l548_548465

theorem simple_interest_true_discount (P R T : ℝ) 
  (h1 : 85 = (P * R * T) / 100)
  (h2 : 80 = (85 * P) / (P + 85)) : P = 1360 :=
sorry

end simple_interest_true_discount_l548_548465


namespace arithmetic_sequence_a5_l548_548672

theorem arithmetic_sequence_a5 (a_n : ℕ → ℝ) 
  (h_arith : ∀ n, a_n (n+1) - a_n n = a_n (n+2) - a_n (n+1))
  (h_condition : a_n 1 + a_n 9 = 10) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_l548_548672


namespace simplify_expression_l548_548054

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l548_548054


namespace harmonic_mean_average_of_x_is_11_l548_548416

theorem harmonic_mean_average_of_x_is_11 :
  let h := (2 * 1008) / (2 + 1008)
  ∃ (x : ℕ), (h + x) / 2 = 11 → x = 18 := by
  sorry

end harmonic_mean_average_of_x_is_11_l548_548416


namespace sqrt_21_is_11th_term_l548_548272

theorem sqrt_21_is_11th_term :
  ∃ n : ℕ, (a_n = √21) ∧ (n = 11) :=
begin
  let a : ℕ → ℝ := λ n, real.sqrt (2 * n - 1),
  use 11,
  split,
  { dsimp [a],
    ring_nf, exact real.sqrt_eq_iff.mpr (or.inl ⟨0, rfl⟩) },
  { refl }
end

end sqrt_21_is_11th_term_l548_548272


namespace total_time_assignment_l548_548687

-- Define the time taken for each part
def time_first_part : ℕ := 25
def time_second_part : ℕ := 2 * time_first_part
def time_third_part : ℕ := 45

-- Define the total time taken for the assignment
def total_time : ℕ := time_first_part + time_second_part + time_third_part

-- The theorem stating that the total time is 120 minutes
theorem total_time_assignment : total_time = 120 := by
  sorry

end total_time_assignment_l548_548687


namespace laundry_per_hour_l548_548885

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l548_548885


namespace smallest_N_impossible_distribution_l548_548806

theorem smallest_N_impossible_distribution :
  ∃ (N : ℕ), N = 3100 ∧ 
  (N ≥ 3000) ∧
  (∀ student : ℕ, student < 31 → ∃ knows : fin N → Prop, 
    (∀ q, q < 3000 → knows q) ∧
    (∀ q, knows q → ∃ s, s < 31 ∧ knows q (fin.mk (3000 * s) (by linarith))) ∧
    (∀ q, q < N - 3000 → ¬ knows q) ∧
    (∀ q, ¬ knows q → (31 * (N - 3000)) ≥ N))
:=
by 
  sorry

end smallest_N_impossible_distribution_l548_548806


namespace fib_factors_l548_548401

def fib (n : ℕ) : ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_factors (A B : ℕ) (h k : ℕ) 
  (h_fibonacci : h < k ∧ ∃ n, h = fib n ∧ k = fib (n + 1))
  (div1 : A^19 ∣ B^93) 
  (div2 : B^19 ∣ A^93) : 
  (A * B)^h ∣ (A^4 + B^8)^k :=
by
  sorry

end fib_factors_l548_548401


namespace minimize_fence_perimeter_l548_548842

-- Define the area of the pen
def area (L W : ℝ) : ℝ := L * W

-- Define that only three sides of the fence need to be fenced
def perimeter (L W : ℝ) : ℝ := 2 * W + L

-- Given conditions
def A : ℝ := 54450  -- Area in square meters

-- The proof statement
theorem minimize_fence_perimeter :
  ∃ (L W : ℝ), 
  area L W = A ∧ 
  ∀ (L' W' : ℝ), area L' W' = A → perimeter L W ≤ perimeter L' W' ∧ L = 330 ∧ W = 165 :=
sorry

end minimize_fence_perimeter_l548_548842


namespace sum_of_valid_b_values_l548_548943

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l548_548943


namespace cos_minus_sin_of_angle_l548_548256

-- Lean 4 statement
theorem cos_minus_sin_of_angle :
  (∃ α : ℝ, (∀ x y : ℝ, (4*x - 3*y = 0) ∧ (x ≤ 0) → α = real.arctan (y / x)) → 
  real.cos α - real.sin α = 1 / 5) := 
sorry

end cos_minus_sin_of_angle_l548_548256


namespace calculate_f_neg_one_l548_548348

def g (x : ℝ) : ℝ := 2 - 3 * x^2

def f (y : ℝ) : ℝ := y / (2 * (2 / 3 - y))

theorem calculate_f_neg_one : f (-1) = -1/2 :=
  sorry

end calculate_f_neg_one_l548_548348


namespace plywood_cut_difference_l548_548076

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548076


namespace number_of_positive_integer_pairs_l548_548623

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l548_548623


namespace plywood_cut_perimeter_difference_l548_548103

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548103


namespace distance_from_right_focus_to_line_l548_548411

noncomputable def distance_from_focus_to_line : ℝ :=
  let focus : ℝ × ℝ := (-2, 0) in
  let line : ℝ × ℝ × ℝ := (1, -√3, 0) in
  -- Define the distance formula from a point (x0, y0) to a line ax + by + c = 0
  (λ (x0 y0 a b c : ℝ), abs (a * x0 + b * y0 + c) / √ (a^2 + b^2)) focus.1 focus.2 line.1 line.2 line.3

theorem distance_from_right_focus_to_line :
  distance_from_focus_to_line = 1 :=
sorry

end distance_from_right_focus_to_line_l548_548411


namespace option_c_has_minimum_value_4_l548_548018

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548018


namespace gen_formulas_sum_first_n_terms_l548_548986

-- Definitions based on conditions
def a_seq (n : ℕ) : ℕ := 3 * n
def b_seq (n : ℕ) : ℕ := 2^(n - 1)
def c_seq (n : ℕ) := a_seq n * b_seq n
def T_seq (n : ℕ) := ∑ i in finset.range n, c_seq (i + 1)

-- Conditions
axiom a1 : a_seq 1 = 3
axiom b1 : b_seq 1 = 1
axiom a2b2 : a_seq 2 * b_seq 2 = 12
axiom S3b2 : (finset.range 3).sum (λ i, a_seq (i + 1)) + b_seq 2 = 20

-- Statements to prove
theorem gen_formulas : (∀ n, a_seq n = 3 * n) ∧ (∀ n, b_seq n = 2^(n - 1)) := sorry

theorem sum_first_n_terms (n : ℕ) : T_seq n = (n-1) * 2^n + 1 := sorry

end gen_formulas_sum_first_n_terms_l548_548986


namespace basis_sets_l548_548243

variables (a b c : Type*) [Add a] [Add b] [Add c] [NonCoplanar a b c]

theorem basis_sets (a b c : Vector) (h_non_coplanar : non_coplanar a b c) :
  (linear_independent ℝ ![(a + b), (a - 2 * b), c]) ∧
  (linear_independent ℝ ![a, (2 * b), (b - c)]) :=
sorry

end basis_sets_l548_548243


namespace convert_45327_base8_to_base10_l548_548450

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 7 * 8^0

theorem convert_45327_base8_to_base10 : base8_to_base10 45327 = 19159 :=
  by
    unfold base8_to_base10
    calc
    4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 7 * 8^0
      = 4 * 4096 + 5 * 512 + 3 * 64 + 2 * 8 + 7 * 1 := by norm_num
    ... = 16384 + 2560 + 192 + 16 + 7 := by norm_num
    ... = 19159 := by norm_num

end convert_45327_base8_to_base10_l548_548450


namespace minimum_value_of_y_l548_548222

noncomputable def m := (Real.tan (Real.toRadians 22.5)) / (1 - (Real.tan (Real.toRadians 22.5))^2)

def y (x : ℝ) := 2 * m * x + (3 / (x - 1)) + 1

theorem minimum_value_of_y (x : ℝ) (h : x > 1) : y x = 2 + 2 * Real.sqrt 3 := sorry

end minimum_value_of_y_l548_548222


namespace solve_equation_l548_548394

theorem solve_equation (x : ℝ) (floor : ℝ → ℤ) 
  (h_floor : ∀ y, floor y ≤ y ∧ y < floor y + 1) :
  (floor (20 * x + 23) = 20 + 23 * x) ↔ 
  (∃ n : ℤ, 20 ≤ n ∧ n ≤ 43 ∧ x = (n - 23) / 20) := 
by
  sorry

end solve_equation_l548_548394


namespace circle_equation_when_m_equals_2_area_of_triangle_is_constant_distance_between_PQ_l548_548227

open Real

-- Defining the center A(m, 2/m) where m ∈ ℝ and m > 0
variables (m : ℝ) (hm : m > 0)

-- The circle A intersects the x-axis at O (origin) and B,
-- as well as the y-axis at O (origin) and C.
-- 1. When m = 2, find the standard equation of circle A.
theorem circle_equation_when_m_equals_2 : 
  (m = 2) → ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

-- 2. As m varies, the area of △OBC is a constant 4.
theorem area_of_triangle_is_constant :
  ∀ (m : ℝ), (m > 0) → ∃ k : ℝ, k = 4 :=
sorry

-- 3. When line l (2x + y - 4 = 0) intersects circle A
--    at points P and Q, and |OP| = |OQ|,
--    the value of |PQ| is (4 sqrt 30) / 5.
theorem distance_between_PQ :
  (m = 2) → 
  let l := λ x y : ℝ, 2 * x + y - 4 = 0 in
  ∀ (P Q : ℝ × ℝ), (|P.1| = |Q.1|) →
  (|P.2| = |Q.2|) →
  ∃ d : ℝ, d = 4 * sqrt 30 / 5 :=
sorry

end circle_equation_when_m_equals_2_area_of_triangle_is_constant_distance_between_PQ_l548_548227


namespace rocket_max_height_l548_548847

def height (t : ℝ) : ℝ := -20 * t ^ 2 + 80 * t + 50

theorem rocket_max_height : 
  ∃ t : ℝ, (∀ t' : ℝ, height t ≥ height t') ∧ height t = 130 :=
sorry

end rocket_max_height_l548_548847


namespace trigonometric_inequality_l548_548282

theorem trigonometric_inequality (a b x : ℝ) (hab : ab > 0) (hx : 0 < x ∧ x < π / 2) :
  (1 + a^2 / sin x) * (1 + b^2 / cos x) ≥ ((1 + sqrt 2 * a * b) ^ 2 * sin(2 * x)) / 2 :=
begin
  sorry
end

end trigonometric_inequality_l548_548282


namespace exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l548_548388

theorem exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles
    (a b c d α β γ δ: ℝ) (h_conv: a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)
    (h_angles: α < β + γ + δ ∧ β < α + γ + δ ∧ γ < α + β + δ ∧ δ < α + β + γ) :
    ∃ (a' b' c' d' α' β' γ' δ' : ℝ),
      (a' / b' = α / β) ∧ (b' / c' = β / γ) ∧ (c' / d' = γ / δ) ∧ (d' / a' = δ / α) ∧
      (a' < b' + c' + d') ∧ (b' < a' + c' + d') ∧ (c' < a' + b' + d') ∧ (d' < a' + b' + c') ∧
      (α' < β' + γ' + δ') ∧ (β' < α' + γ' + δ') ∧ (γ' < α' + β' + δ') ∧ (δ' < α' + β' + γ') :=
  sorry

end exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l548_548388


namespace unknown_cube_edge_length_l548_548745

theorem unknown_cube_edge_length (a b c x : ℕ) (h_a : a = 6) (h_b : b = 10) (h_c : c = 12) : a^3 + b^3 + x^3 = c^3 → x = 8 :=
by
  sorry

end unknown_cube_edge_length_l548_548745


namespace B_share_after_tax_l548_548855

noncomputable def B_share (x : ℝ) : ℝ := 3 * x
noncomputable def salary_proportion (A B C D : ℝ) (x : ℝ) :=
  A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 6 * x
noncomputable def D_more_than_C (D C : ℝ) : Prop :=
  D - C = 700
noncomputable def meets_minimum_wage (B : ℝ) : Prop :=
  B ≥ 1000
noncomputable def tax_deduction (B : ℝ) : ℝ :=
  if B > 1500 then B - 0.15 * (B - 1500) else B

theorem B_share_after_tax (A B C D : ℝ) (x : ℝ) (h1 : salary_proportion A B C D x)
  (h2 : D_more_than_C D C) (h3 : meets_minimum_wage B) :
  tax_deduction B = 1050 :=
by
  sorry

end B_share_after_tax_l548_548855


namespace canvas_cost_is_40_l548_548774

-- Definitions based on the conditions
def total_expense : ℝ := 90
def easel_cost : ℝ := 15
def paintbrushes_cost : ℝ := 15
def mixed_set_cost (canvas_cost : ℝ) : ℝ := canvas_cost / 2

-- Problem statement: Prove that the cost of the canvases is $40.00.
theorem canvas_cost_is_40 (C : ℝ) :
  C + mixed_set_cost C + easel_cost + paintbrushes_cost = total_expense ↔ C = 40 :=
by
  sorry

end canvas_cost_is_40_l548_548774


namespace alice_cookies_sum_l548_548788

open Nat

theorem alice_cookies_sum :
  let N_values := { N | N % 4 = 1 ∧ N % 6 = 5 ∧ N < 50 }
  (Σ N in N_values, N) = 208 :=
by
  sorry

end alice_cookies_sum_l548_548788


namespace rectangular_prism_diagonals_l548_548149

theorem rectangular_prism_diagonals (length width height : ℕ) (length_eq : length = 4) (width_eq : width = 3) (height_eq : height = 2) : 
  ∃ (total_diagonals : ℕ), total_diagonals = 16 :=
by
  let face_diagonals := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals
  use total_diagonals
  sorry

end rectangular_prism_diagonals_l548_548149


namespace sum_of_possible_b_values_l548_548942

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l548_548942


namespace river_flow_speed_l548_548856

-- Define the conditions given in the problem
def swimmer_speed_still_water (a : ℝ) : Prop := a > 0

def time_swim_upstream := 30 / 60 -- 30 minutes in hours

def distance_downstream := 1.2 -- kilometers

-- Define the speed of the river flow (which we need to find) and the swimmer's speed in still water
variables (x a : ℝ)

-- The swimmer's realized he lost his water bottle after 30 minutes
def time_to_catch_bottle (a x : ℝ) : ℝ := distance_downstream + time_swim_upstream * (a - x) / (a + x)

-- State the theorem to prove
theorem river_flow_speed (h_a : swimmer_speed_still_water a) (h_x : time_to_catch_bottle a x = distance_downstream / x - time_swim_upstream) : 
  x = 1.2 :=
sorry

end river_flow_speed_l548_548856


namespace slopes_arithmetic_sequence_min_area_triangle_PCD_l548_548341

-- Definitions and conditions
def ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

def on_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.fst A.snd ∧ ellipse B.fst B.snd

def not_point_P (A B : ℝ × ℝ) : Prop :=
  ¬(A = (0, 1) ∨ B = (0, 1))

def line_through_origin (A B : ℝ × ℝ) : Prop :=
  A.fst * B.snd = A.snd * B.fst

-- Problem statement
theorem slopes_arithmetic_sequence (A B : ℝ × ℝ) 
  (h1 : on_ellipse A B) 
  (h2 : not_point_P A B)
  (h3 : line_through_origin A B) :
  ∃ k_PA k_AB k_PB : ℝ, 
  k_PA = (A.snd - 1) / A.fst ∧
  k_AB = A.snd / A.fst ∧
  k_PB = (B.snd - 1) / B.fst ∧
  k_PA + k_PB = 2 * k_AB := sorry

theorem min_area_triangle_PCD (A B : ℝ × ℝ) 
  (h1 : on_ellipse A B) 
  (h2 : not_point_P A B)
  (h3 : line_through_origin A B) :
  ∃ mCD : ℝ, mCD ≤ sqrt 2 / 3 := sorry

end slopes_arithmetic_sequence_min_area_triangle_PCD_l548_548341


namespace correct_choice_is_B_l548_548606

theorem correct_choice_is_B (f : ℝ → ℝ) (h1 : ∀ x, f x = |cos x| * sin x) :
  (f (2014 * π / 3) = - sqrt 3 / 4) ∧
  ¬(∀ x, f x = f (x + π)) ∧
  (∀ x ∈ Icc (-π / 4) (π / 4), (∀ y ∈ Icc (-π / 4) x, f y < f x)) ∧
  ¬(∀ x, f ((π / 2) - x) = - f (x - (π / 2))) :=
sorry

end correct_choice_is_B_l548_548606


namespace roger_owes_correct_amount_l548_548722

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l548_548722


namespace intersection_M_N_l548_548273

-- Definitions of the sets M and N
def M : Set ℤ := {-3, -2, -1}
def N : Set ℤ := { x | -2 < x ∧ x < 3 }

-- The theorem stating that the intersection of M and N is {-1}
theorem intersection_M_N : M ∩ N = {-1} := by
  sorry

end intersection_M_N_l548_548273


namespace problem_l548_548637

noncomputable def c := Real.log 8
noncomputable def d := Real.log 27

theorem problem : 9^(c/d) + 2^(d/c) = 7 := by
  sorry

end problem_l548_548637


namespace point_P_in_first_quadrant_l548_548670

def lies_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : lies_in_first_quadrant 2 1 :=
by {
  sorry
}

end point_P_in_first_quadrant_l548_548670


namespace distance_between_points_lt_0_5_l548_548325

-- Define an equilateral triangle with side length 1
def equilateral_triangle (a : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (a = 1) ∧ (b = 1) ∧ (c = 1) ∧ (a + b + c = 3)

-- Define the property of placing points inside the equilateral triangle
def points_in_triangle (P : Finset (ℝ × ℝ)) (n : ℕ) (triangle : Finset (ℝ × ℝ)) : Prop :=
  P.card = n ∧ ∀ p ∈ P, p ∈ triangle

-- Define the main theorem statement
theorem distance_between_points_lt_0_5 (P : Finset (ℝ × ℝ)) (triangle : Finset (ℝ × ℝ)) :
  equilateral_triangle 1 1 1 →
  points_in_triangle P 5 triangle →
  ∃ p1 p2 ∈ P, p1 ≠ p2 ∧ (dist p1 p2 < 0.5) :=
sorry

end distance_between_points_lt_0_5_l548_548325


namespace plywood_perimeter_difference_l548_548111

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548111


namespace sum_of_b_for_rational_roots_l548_548951

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l548_548951


namespace option_c_has_minimum_value_4_l548_548012

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548012


namespace sqrt_arithmetic_l548_548210

theorem sqrt_arithmetic : 
  sqrt 1.21 / sqrt 0.81 + 
  sqrt 1.44 / sqrt 0.49 = 
  2 + 59 / 63 := by
  sorry

end sqrt_arithmetic_l548_548210


namespace sum_of_possible_b_values_l548_548939

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l548_548939


namespace office_expense_reduction_l548_548653

theorem office_expense_reduction (x : ℝ) (h : 0 ≤ x) (h' : x ≤ 1) : 
  2500 * (1 - x) ^ 2 = 1600 :=
sorry

end office_expense_reduction_l548_548653


namespace sum_of_possible_values_of_m_l548_548184

theorem sum_of_possible_values_of_m :
  let original_set := {1, 5, 8, 12}
  ∃ m ∈ Ioo 1 12, (m ≠ 1 ∧ m ≠ 5 ∧ m ≠ 8 ∧ m ≠ 12) ∧
                  let new_set := {1, 5, 8, 12, m}
                  let median := if m ≤ 5 then 5 else if m ≥ 8 then 8 else m
                  let mean := (1 + 5 + 8 + 12 + m) / 5
                  median = mean := by
  sorry

end sum_of_possible_values_of_m_l548_548184


namespace sum_b_up_to_3000_l548_548967

def b (p : ℕ) : ℕ :=
  let k := (Int.ceil (Real.sqrt p - 1/2)).toNat
  if k * k <= p ∧ p < (k + 1) * (k + 1) then k else k + 1

def sum_b (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), b i

theorem sum_b_up_to_3000 : sum_b 3000 = 22488 := by
  sorry

end sum_b_up_to_3000_l548_548967


namespace probability_writing_about_both_classes_l548_548713

theorem probability_writing_about_both_classes:
  (total_students French Spanish none: ℕ) (students_at_least_one_class both French_only Spanish_only: ℕ)
  (total_combinations undesired French_only_combinations Spanish_only_combinations: ℕ) 
  (probability undesired_probability gcd: ℚ):
  total_students = 30 →
  French = 22 →
  Spanish = 24 →
  none = 2 →
  students_at_least_one_class = total_students - none →
  both = French + Spanish - students_at_least_one_class →
  French_only = French - both →
  Spanish_only = Spanish - both →
  total_combinations = (total_students * (total_students - 1)) / 2 →
  French_only_combinations = (French_only * (French_only - 1)) / 2 →
  Spanish_only_combinations = (Spanish_only * (Spanish_only - 1)) / 2 →
  undesired = French_only_combinations + Spanish_only_combinations →
  undesired_probability = 1 - (undesired / total_combinations) →
  gcd = 29 →
  probability = (392 / 435) / gcd →
  probability = 14 / 15 := by
  intros total_students_eq French_eq Spanish_eq none_eq
         students_at_least_one_eq both_eq French_only_eq Spanish_only_eq
         total_combinations_eq French_only_comb_eq Spanish_only_comb_eq
         undesired_eq undesired_prob_eq gcd_eq
  sorry

end probability_writing_about_both_classes_l548_548713


namespace group_allocation_minimizes_time_total_duration_after_transfer_l548_548061

theorem group_allocation_minimizes_time :
  ∃ x y : ℕ,
  x + y = 52 ∧
  (x = 20 ∧ y = 32) ∧
  (min (60 / x) (100 / y) = 25 / 8) := sorry

theorem total_duration_after_transfer (x y x' y' : ℕ) (H : x = 20) (H1 : y = 32) (H2 : x' = x - 6) (H3 : y' = y + 6) :
  min ((100 * (2/5)) / x') ((152 * (2/3)) / y') = 27 / 7 := sorry

end group_allocation_minimizes_time_total_duration_after_transfer_l548_548061


namespace equation_of_ellipse_existence_of_fixed_point_l548_548235

noncomputable def ellipse_eq (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) (P : ℝ×ℝ)
  (h₂ : P = (1, real.sqrt 2 / 2)): Prop :=
(x y : ℝ), x^2/(a^2) + y^2/(b^2) = 1 

/- Given an ellipse through the point P, prove the equation of the ellipse -/
theorem equation_of_ellipse (a b : ℝ) (h₀ : a = real.sqrt 2 * b) (h₁ : b = 1)
    (P : ℝ×ℝ) (h₂ : P = (1, real.sqrt 2 / 2)):
    ellipse_eq real.sqrt 2 1 P := by
  sorry


/- Given a line intersecting the ellipse, prove the existence of fixed point T -/
theorem existence_of_fixed_point (k : ℝ) (h₀ : k ≠ 0) (h₁ : k ∈ ℝ) :
  ∃ T : ℝ × ℝ, T = (0,1) ∧ ∀ x y : ℝ, circle_with_diameter_AB (line_through (0, -1/3) k)
  (ellipse_eq real.sqrt 2 1 P) T := by
  sorry

end equation_of_ellipse_existence_of_fixed_point_l548_548235


namespace range_of_x_for_a_eq_2_range_of_a_for_p_necessary_l548_548974

-- First part: if a = 2 and p ∧ q is true, find the range of x
theorem range_of_x_for_a_eq_2 (a : ℝ) (x : ℝ) 
  (p : (x < a)) (q : (x^2 - 4 * x + 3 ≤ 0)) : 
  a = 2 → p ∧ q → 1 ≤ x ∧ x < 2 :=
by
  sorry

-- Second part: if p is a necessary but not sufficient condition for q, find the range of a
theorem range_of_a_for_p_necessary (a : ℝ) (p : ∀ x : ℝ, x < a) (q : ∀ x : ℝ, x^2 - 4 * x + 3 ≤ 0) :
  (∃ x : ℝ, q x → p x) ∧ (∃ x : ℝ, ¬(p x → q x)) → a > 3 :=
by
  sorry

end range_of_x_for_a_eq_2_range_of_a_for_p_necessary_l548_548974


namespace sec_150_eq_neg_two_sqrt_three_div_three_l548_548920

theorem sec_150_eq_neg_two_sqrt_three_div_three : sec 150 = - (2 * sqrt 3 / 3) :=
by sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l548_548920


namespace share_of_a_is_240_l548_548053

theorem share_of_a_is_240 (A B C : ℝ) 
  (h1 : A = (2/3) * (B + C)) 
  (h2 : B = (2/3) * (A + C)) 
  (h3 : A + B + C = 600) : 
  A = 240 := 
by sorry

end share_of_a_is_240_l548_548053


namespace general_term_formula_sum_of_reciprocals_l548_548753

section part1

-- Definitions and sequence a_n
def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3 else n * (n + 2)

-- The main theorem that we need to prove for part 1
theorem general_term_formula : ∀ n : ℕ, n > 1 → sequence n = n * (n + 2) := sorry

end part1

section part2

variables (c : ℝ) (a : ℕ → ℝ)

-- Assume the sequence is already defined as in part 1
def a_i (n : ℕ) : ℝ := n * (n + 2)

-- The main theorem that we need to prove for part 2
theorem sum_of_reciprocals (h : ∀ k : ℕ, k > 0 → ∑ i in finset.range k, 1 / a_i i < c) : 
  c ≥ 3 / 4 := sorry

end part2

end general_term_formula_sum_of_reciprocals_l548_548753


namespace find_number_l548_548732

def num_set := {16, 18, 19, 21}

def less_than_20 (n : ℕ) : Prop := n < 20

def not_multiple_of_2 (n : ℕ) : Prop := ¬ (n % 2 = 0)

theorem find_number : {n : ℕ | n ∈ num_set ∧ less_than_20 n ∧ not_multiple_of_2 n} = {19} :=
by
  -- Proof omitted
  sorry

end find_number_l548_548732


namespace option_c_has_minimum_value_4_l548_548015

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548015


namespace triangle_perimeter_l548_548423

theorem triangle_perimeter (side1 side2 side3 : ℕ) (h1 : side1 = 40) (h2 : side2 = 50) (h3 : side3 = 70) : 
  side1 + side2 + side3 = 160 :=
by 
  sorry

end triangle_perimeter_l548_548423


namespace train_A_length_l548_548777

theorem train_A_length
  (speed_A : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (len_A : ℕ)
  (h1 : speed_A = 54) 
  (h2 : speed_B = 36) 
  (h3 : time_to_cross = 15)
  (h4 : len_A = (speed_A + speed_B) * 1000 / 3600 * time_to_cross) :
  len_A = 375 :=
sorry

end train_A_length_l548_548777


namespace plywood_cut_perimeter_difference_l548_548107

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548107


namespace count_gcd_21_n_eq_3_l548_548968

theorem count_gcd_21_n_eq_3 :
  { n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 21 n = 3 }.to_finset.card = 57 :=
sorry

end count_gcd_21_n_eq_3_l548_548968


namespace product_bound_l548_548339

theorem product_bound (k : ℕ) (n : ℕ → ℕ) (a b : ℕ) 
  (hk : k ≥ 2) 
  (hn_pos : ∀ i, 1 ≤ i → i ≤ k → 1 < n i ) 
  (hn_inc : ∀ i j, 1 ≤ i → i < j → j ≤ k → n i < n j) 
  (hab : 0 < a ∧ 0 < b) 
  (prod_ineq : (∏ i in Finset.range k, (1 - 1 / (n (i+1) : ℤ))) ≤ (a : ℤ) / b ∧ (a : ℤ) / b < ∏ i in Finset.range (k - 1), (1 - 1 / (n (i+1) : ℤ))) :
  ∏ i in Finset.range k, (n (i+1) : ℤ) ≥ (4 * a) ^ (2^k - 1) :=
by
  sorry

end product_bound_l548_548339


namespace plywood_perimeter_difference_l548_548115

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548115


namespace minimum_area_triangle_OCD_l548_548985

noncomputable def point_on_ellipse (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos α, 2 * Real.sqrt 2 * Real.sin α)

def vertex_A : ℝ × ℝ := (0, 2 * Real.sqrt 2)
def vertex_B : ℝ × ℝ := (0, -2 * Real.sqrt 2)

def point_C (α : ℝ) : ℝ × ℝ :=
  (-2, (2 * Real.sqrt 2 * Real.cos α - 4 * Real.sin α + 4) / Real.cos α)

def point_D (α : ℝ) : ℝ × ℝ :=
  (-2, -(4 * Real.sin α + 2 * Real.sqrt 2 * Real.cos α + 4) / Real.cos α)

def area_OCD (α : ℝ) : ℝ :=
  abs (4 * Real.sqrt 2 + 8 / Real.cos α)

theorem minimum_area_triangle_OCD : 
  ∃ α : ℝ, area_OCD α = 8 - 4 * Real.sqrt 2 :=
sorry

end minimum_area_triangle_OCD_l548_548985


namespace problem_statement_l548_548354

noncomputable def Q (x : ℝ) : ℝ := d * x^4 + a * x^3 + b * x^2 + c * x + m

theorem problem_statement (d a b c m : ℝ) 
  (h1 : Q(0) = m) 
  (h2 : Q(1) = 3m) 
  (h3 : Q(-1) = 4m) : 
  Q(3) + Q(-3) = 144 * d + 47 * m :=
by 
  -- Proof required here
  sorry

end problem_statement_l548_548354


namespace product_of_roots_eq_30_l548_548535

-- Define the polynomials and their product
def poly1 : Polynomial ℝ := 3 * X^3 + 2 * X^2 - 5 * X + 15
def poly2 : Polynomial ℝ := 4 * X^3 - 20 * X^2 + 24

-- Define the product polynomial
def productPoly : Polynomial ℝ := poly1 * poly2

-- Formal statement: Proving the product of the roots of the polynomial is 30
theorem product_of_roots_eq_30 : 
  (∀ (p : Polynomial ℝ), p = productPoly → 
  (Polynomial.roots p).prod = 30) := 
sorry

end product_of_roots_eq_30_l548_548535


namespace bob_scoops_needed_l548_548515

def scoops_needed (total_flour : ℚ) (scoop_size : ℚ) : ℚ :=
  total_flour / scoop_size

theorem bob_scoops_needed : 
  let full_flour := (15/4 : ℚ)
  let combined_flour := (3/2 : ℚ)
  let total_flour := full_flour + combined_flour
  let scoop_size := (1/3 : ℚ)
  let scoops := scoops_needed total_flour scoop_size
  ⌈scoops⌉ = 16 :=
by {
  let full_flour := (15/4 : ℚ)
  let combined_flour := (3/2 : ℚ)
  let total_flour := full_flour + combined_flour
  let scoop_size := (1/3 : ℚ)
  let scoops := total_flour / scoop_size
  show ⌈scoops⌉ = 16,
  sorry
}

end bob_scoops_needed_l548_548515


namespace stratified_sampling_l548_548479

theorem stratified_sampling (total_employees : ℕ) (sample_size : ℕ) (department_employees : ℕ)
  (h_total : total_employees = 240)
  (h_sample : sample_size = 20)
  (h_department : department_employees = 60) :
  (department_employees * sample_size) / total_employees = 5 :=
by
  rw [h_total, h_sample, h_department]
  norm_num
  sorry

end stratified_sampling_l548_548479


namespace modulus_squared_of_complex_l548_548259

theorem modulus_squared_of_complex :
  (complex.abs (complex.div (3 - 2 * complex.I) (1 - complex.I)))^2 = 13 / 2 :=
by
  sorry

end modulus_squared_of_complex_l548_548259


namespace distinct_sum_values_count_l548_548290

theorem distinct_sum_values_count (a : Fin 10 → Int) (h : ∀ i, a i = 1 ∨ a i = -1) :
  (Finset.image (fun s => (Finset.sum Finset.univ a) s) (Finset.univ : Finset (Fin 10))).card = 11 :=
sorry

end distinct_sum_values_count_l548_548290


namespace log_condition_necessity_l548_548050

theorem log_condition_necessity (m a : ℝ) (h1 : (m - 1) * (a - 1) > 0)
  : ¬ ((log a m > 0) ↔ ((m - 1) * (a - 1) > 0)) := sorry

end log_condition_necessity_l548_548050


namespace num_pairs_satisfying_equation_l548_548629

theorem num_pairs_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end num_pairs_satisfying_equation_l548_548629


namespace option_c_has_minimum_value_4_l548_548013

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548013


namespace number_wall_m_value_l548_548308

theorem number_wall_m_value (m : ℕ) :
  let b1 := m
  let b2 := 6
  let b3 := 10
  let b4 := 9
  let b5 := b1 + b2
  let b6 := b2 + b3
  let b7 := b3 + b4
  let b8 := b5 + b6
  let b9 := b6 + b7
  let top := b8 + b9
  top = 64 → m = 7 :=
by {
  assume H : top = 64,
  sorry
}

end number_wall_m_value_l548_548308


namespace sum_of_solutions_l548_548454

theorem sum_of_solutions :
  ∑ x in Finset.filter (λ x : ℕ, (7 * (5 * x - 3)) % 12 = 14 % 12)
    (Finset.range 31), x = 39 := by
  sorry

end sum_of_solutions_l548_548454


namespace plywood_perimeter_difference_l548_548083

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548083


namespace perimeters_positive_difference_l548_548101

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548101


namespace plywood_cut_perimeter_difference_l548_548104

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548104


namespace order_of_three_numbers_l548_548284

theorem order_of_three_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 :=
by
  sorry

end order_of_three_numbers_l548_548284


namespace Harry_bought_five_packets_of_chili_pepper_l548_548140

noncomputable def price_pumpkin : ℚ := 2.50
noncomputable def price_tomato : ℚ := 1.50
noncomputable def price_chili_pepper : ℚ := 0.90
noncomputable def packets_pumpkin : ℕ := 3
noncomputable def packets_tomato : ℕ := 4
noncomputable def total_spent : ℚ := 18
noncomputable def packets_chili_pepper (p : ℕ) := price_pumpkin * packets_pumpkin + price_tomato * packets_tomato + price_chili_pepper * p = total_spent

theorem Harry_bought_five_packets_of_chili_pepper :
  ∃ p : ℕ, packets_chili_pepper p ∧ p = 5 :=
by 
  sorry

end Harry_bought_five_packets_of_chili_pepper_l548_548140


namespace probability_first_king_spades_second_spade_l548_548853

noncomputable def deck := fin 52

def is_king_of_spades (c: deck) : Prop := c = fin.mk 0 (by norm_num)
def is_spade (c: deck) : Prop := c.val / 13 = 0  -- first 13 cards are Spades

-- Define event that first card is King of Spades and second card is Spade
def event (first second : deck) : Prop :=
  is_king_of_spades first ∧ 
  is_spade second 

theorem probability_first_king_spades_second_spade :
  (∃ (first second : deck), event first second) →
  (Pr (event first second) = 1 / 221) :=
sorry

end probability_first_king_spades_second_spade_l548_548853


namespace cyclic_sum_inequality_l548_548574

theorem cyclic_sum_inequality (x y z a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ 3 / (a + b) :=
  sorry

end cyclic_sum_inequality_l548_548574


namespace g_at_neg1_l548_548591

-- Defining even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given functions f and g
variables (f g : ℝ → ℝ)

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^(1 - x)

-- Proof statement
theorem g_at_neg1 : g (-1) = -3 / 2 :=
by
  sorry

end g_at_neg1_l548_548591


namespace ellipse_cos_sin_identity_l548_548692

noncomputable def e : ℝ := sorry
noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry

theorem ellipse_cos_sin_identity
  (h1 : ∃ (Γ : Set (ℝ × ℝ)), ∃ (A B : (ℝ × ℝ)), ∀ C : (ℝ × ℝ), C ∈ Γ → (A and B are foci))
  (h2 : ∃ C : (ℝ × ℝ), C ∈ Γ)
  (h3 : e = (eccentricity of Γ)) :
  (1 + Real.cos A * Real.cos B) / (Real.sin A * Real.sin B) = (1 + e ^ 2) / (1 - e ^ 2) := 
sorry

end ellipse_cos_sin_identity_l548_548692


namespace candidate_1_fails_by_40_marks_l548_548128

-- Definitions based on the conditions
def total_marks (T : ℕ) := T
def passing_marks (pass : ℕ) := pass = 160
def candidate_1_failed_by (marks_failed_by : ℕ) := ∃ (T : ℕ), (0.4 : ℝ) * T = 0.4 * T ∧ (0.6 : ℝ) * T - 20 = 160

-- Theorem to prove the first candidate fails by 40 marks
theorem candidate_1_fails_by_40_marks (marks_failed_by : ℕ) : candidate_1_failed_by marks_failed_by → marks_failed_by = 40 :=
by
  sorry

end candidate_1_fails_by_40_marks_l548_548128


namespace rectangular_coordinates_from_polar_l548_548145

theorem rectangular_coordinates_from_polar (x y r θ : ℝ) (h1 : r * Real.cos θ = x) (h2 : r * Real.sin θ = y) :
    r = 10 ∧ θ = Real.arctan (6 / 8) ∧ (2 * r, 3 * θ) = (20, 3 * Real.arctan (6 / 8)) →
    (20 * Real.cos (3 * Real.arctan (6 / 8)), 20 * Real.sin (3 * Real.arctan (6 / 8))) = (-7.04, 18.72) :=
by
  intros
  -- We need to prove that the statement holds
  sorry

end rectangular_coordinates_from_polar_l548_548145


namespace sequence_properties_l548_548707

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a(n+1) = a(n) + d

noncomputable def geometric_property (a : ℕ → ℝ) : Prop :=
  (2 * a 2 + 2)^2 = a 1 * 5 * a 3

theorem sequence_properties :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n : ℕ, 2 * S n - n * a n = 10 * n) →
  (∃ a_1 a_2 a_3, a 1 = a_1 ∧ 2 * a 2 + 2 = a_2 ∧ 5 * a 3 = a_3) →
  (∃ d, d < 0) →
  is_arithmetic_sequence a ∧ geometric_property a ∧ 
  ∀ n : ℕ,
    let T := if n ≤ 11 then (1 / 2) * n * (21 - n)
             else (1 / 2) * n^2 - (21 / 2) * n + 110 in
  S n = T :=
by
  sorry

end sequence_properties_l548_548707


namespace optionC_has_min_4_l548_548005

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548005


namespace center_of_hyperbola_l548_548827

-- Definitions of the coordinates of the foci
def focus1 := (3 : ℝ, -2 : ℝ)
def focus2 := (7 : ℝ, 6 : ℝ)

-- Midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Statement to prove the midpoint is the center of the hyperbola
theorem center_of_hyperbola :
  midpoint focus1 focus2 = (5, 2) :=
by
  sorry

end center_of_hyperbola_l548_548827


namespace find_x_l548_548046

theorem find_x :
  let avg1 := (10 + 30 + 50) / 3 in
  let avg2 := (20 + 40 + x) / 3 in
  avg1 = avg2 + 8 → x = 6 :=
by
  sorry

end find_x_l548_548046


namespace horizontal_asymptote_l548_548532

theorem horizontal_asymptote :
  ∀ x, (6 * x^3 + 5 * x^2 - 9) / (4 * x^3 + 3 * x^2 + 5 * x + 2) → (y = 3/2) :=
begin
  sorry
end

end horizontal_asymptote_l548_548532


namespace sum_m_n_zero_l548_548578

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l548_548578


namespace product_zero_when_b_is_3_l548_548546

theorem product_zero_when_b_is_3 (b : ℤ) (h : b = 3) :
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) *
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  sorry
}

end product_zero_when_b_is_3_l548_548546


namespace wheels_motion_is_rotation_l548_548424

def motion_wheel_car := "rotation"
def question_wheels_motion := "What is the type of motion exhibited by the wheels of a moving car?"

theorem wheels_motion_is_rotation :
  (question_wheels_motion = "What is the type of motion exhibited by the wheels of a moving car?" ∧ 
   motion_wheel_car = "rotation") → motion_wheel_car = "rotation" :=
by
  sorry

end wheels_motion_is_rotation_l548_548424


namespace a1_value_a2_value_a3_value_a4_value_general_formula_is_geometric_l548_548708

noncomputable def sequence := ℕ+ → ℚ

variable (a : sequence)

-- Condition: the sequence has a sum of first n terms
def sum_seq (n : ℕ+) : ℚ := ∑ i in Finset.range n, a i

-- Condition: an = 2 - Sn
axiom seq_condition (n : ℕ+) : a n = 2 - sum_seq a n

-- Proving specific values: a1, a2, a3, a4
theorem a1_value : a 1 = 1 := 
  sorry

theorem a2_value : a 2 = 1 / 2 :=
  sorry

theorem a3_value : a 3 = 1 / 4 :=
  sorry

theorem a4_value : a 4 = 1 / 8 :=
  sorry

-- General formula proof
theorem general_formula (n : ℕ+) : a n = (1 / 2)^(n - 1) :=
  sorry

-- Proving the sequence is geometric with ratio 1/2
theorem is_geometric : ∃ p : ℚ, p ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) / a n = p :=
  sorry

end a1_value_a2_value_a3_value_a4_value_general_formula_is_geometric_l548_548708


namespace f_symmetric_about_center_l548_548264

noncomputable def f (x : ℝ) := x^2 / (x^2 - 10 * x + 50)

theorem f_symmetric_about_center :
  ∀ x : ℝ, f(10 - x) + f(x) = 2 :=
by
  sorry

end f_symmetric_about_center_l548_548264


namespace cannot_be_sum_of_consecutive_odds_l548_548793

theorem cannot_be_sum_of_consecutive_odds (S : ℕ) : 
  S ∈ {16, 40, 72, 100, 200} → (∃ n : ℤ, n % 2 = 1 ∧ S = 4 * n + 12) ↔ S ≠ 100 :=
by
  sorry

end cannot_be_sum_of_consecutive_odds_l548_548793


namespace min_norm_l548_548610

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b : ℝ × ℝ := (1, 2)

def vec_add (λ : ℝ) : ℝ × ℝ := (a.1 + λ * b.1, a.2 + λ * b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem min_norm : (λ : ℝ) -> (vec_norm (vec_add λ)) = minimal norm :=
  minimum = 3 * real.sqrt(5) / 5 :=
begin
  -- The proof will go here.
  sorry
end

end min_norm_l548_548610


namespace alicia_tax_deduction_l548_548162

theorem alicia_tax_deduction :
  let hourly_wage_dollars := 20
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 0.0145
  let tax_deduction_cents := tax_rate * hourly_wage_cents
  in tax_deduction_cents = 29 := by
sorry

end alicia_tax_deduction_l548_548162


namespace plywood_cut_perimeter_difference_l548_548109

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548109


namespace num_solutions_20_l548_548368

def num_solutions (n : ℕ) : ℕ :=
  4 * n

theorem num_solutions_20 : num_solutions 20 = 80 := by
  sorry

end num_solutions_20_l548_548368


namespace inequality_transitivity_l548_548283

theorem inequality_transitivity (a b c : ℝ) (h : a > b) : 
  a + c > b + c :=
sorry

end inequality_transitivity_l548_548283


namespace jericho_owes_annika_l548_548775

variable (J A M : ℝ)
variable (h1 : 2 * J = 60)
variable (h2 : M = A / 2)
variable (h3 : 30 - A - M = 9)

theorem jericho_owes_annika :
  A = 14 :=
by
  sorry

end jericho_owes_annika_l548_548775


namespace integral_transform_l548_548195

variable (C : ℝ)

theorem integral_transform : 
  ∫ x in set.univ, x^(1 / 4) * (1 + x^(1 / 2))^(-7 / 2) dx = (4 / (5 * (1 + x^(1 / 2))^(5 / 2))) + C :=
sorry

end integral_transform_l548_548195


namespace minimum_cells_l548_548448

universe u

-- Define the 8x8 board and positions
def Position : Type := Fin 8 × Fin 8

-- Define the hidden treasure's position and the distance function
variable (treasure : Position) 
def distance (p1 p2 : Position) : Nat := Nat.abs (p1.fst.val - p2.fst.val) + Nat.abs (p1.snd.val - p2.snd.val)

-- Define the notion of plaques
def plaque (p : Position) : Nat := distance p treasure

-- Define the minimum cells required to be dug up to find the treasure
def min_cells_to_find_treasure (board_size : Nat) : Nat := 3

-- The main theorem to be proven
theorem minimum_cells (treasure : Position) : min_cells_to_find_treasure 8 = 3 := by
  sorry -- Proof needs to be filled in, stating it requires exactly 3 cells

end minimum_cells_l548_548448


namespace three_is_square_root_of_nine_l548_548795

theorem three_is_square_root_of_nine :
  ∃ x : ℝ, x * x = 9 ∧ x = 3 :=
sorry

end three_is_square_root_of_nine_l548_548795


namespace bill_sunday_miles_l548_548801

def miles_run (B : ℕ) (J : ℕ) (total_miles : ℕ) : Prop :=
  total_miles = B + (B + 4) + J

theorem bill_sunday_miles (B : ℕ) (J : ℕ) (total_miles : ℕ) (h1 : J = 2 * (B + 4)) (h2 : total_miles = 20) :
  B = 2 → B + 4 = 6 :=
by
  intro hB
  have h3 : B + 4 = 6 := by linarith
  exact h3

#eval bill_sunday_miles 2 12 20 (by linarith) (by linarith)

end bill_sunday_miles_l548_548801


namespace circle_tangent_chord_length_l548_548521

/-
Problem:
Circles C1 and C2 are externally tangent, and they are both internally tangent to a larger circle C3.
The radii of C1 and C2 are 5 and 11, respectively, and the centers of all three circles are collinear.
Prove that a chord of C3, which is also a common external tangent of C1 and C2, has a length 
    of 2 * sqrt(105).
    Find m+n+p for the chord length expressed in the form m sqrt(n) / p, with m and p 
    relatively prime, and n not divisible by the square of any prime.
    The answer is 108.
-/
theorem circle_tangent_chord_length :
  ∃ m n p : ℕ,
    m + n + p = 108 ∧ 
    let r1 := 5 in
    let r2 := 11 in
    let r3 := r1 + r2 in
    let chord_length := 2 * Real.sqrt ((r3 ^ 2) - (11 ^ 2)) in
    chord_length = (m * Real.sqrt n) / p ∧
    Nat.Coprime m p ∧
    ∀ d : ℕ, d^2 ∣ n → d = 1 :=
by {
  sorry
}

end circle_tangent_chord_length_l548_548521


namespace tropical_algebra_distributive_l548_548374

theorem tropical_algebra_distributive (x y z : ℝ∞) :
  (min x y) + z = min (x + z) (y + z) := 
sorry

end tropical_algebra_distributive_l548_548374


namespace range_of_y_period_of_y_l548_548928

-- Define the function y
def y (x : ℝ) := cos (3 * x) ^ 2 + sin (6 * x)

-- Define the range of the function
def range_y := [(1 / 2 - (sqrt 5) / 2), (1 / 2 + (sqrt 5) / 2)]

-- Define the period of the function
def period_y := (π / 3)

-- Proof that the range of y is as specified
theorem range_of_y : 
  set.range y = set.Icc (1 / 2 - (sqrt 5) / 2) (1 / 2 + (sqrt 5) / 2) := 
sorry

-- Proof that the smallest period of y is as specified
theorem period_of_y :
  ∀ x, y (x + period_y) = y x :=
sorry

end range_of_y_period_of_y_l548_548928


namespace trucks_on_lot_saturday_morning_l548_548157

theorem trucks_on_lot_saturday_morning : 
  ∀ (total_trucks : ℕ) (rented_trucks : ℕ) (percentage_returned : ℕ),
    total_trucks = 20 ∧
    rented_trucks ≤ total_trucks ∧
    rented_trucks = 20 ∧
    percentage_returned = 50 →
    0.5 * rented_trucks + (total_trucks - rented_trucks) = 10 :=
begin
  intros _ _ _ h,
  have ht := h.1,
  have hr := h.2.2.1,
  have hp := h.2.2.2,
  have htotal := h.2.1,
  rw [ht, hr, hp],
  linarith,
end

end trucks_on_lot_saturday_morning_l548_548157


namespace find_alpha_l548_548592

theorem find_alpha 
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : - π / 2 < β ∧ β < 0)
  (h3 : cos (α - β) = 3 / 5)
  (h4 : sin β = -sqrt 2 / 10) :
  α = π / 4 :=
sorry

end find_alpha_l548_548592


namespace plywood_perimeter_difference_l548_548089

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548089


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548025

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548025


namespace partition_points_l548_548688

theorem partition_points (A : set (ℝ × ℝ)) (2n : ℕ) (hA : A.card = 2n) 
  (hno_collinear : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ A ∧ p2 ∈ A ∧ p3 ∈ A → affine_independent ℝ ![p1, p2, p3]) 
  (a b : (ℝ × ℝ)) (ha_in_A : a ∈ A) (hb_in_A : b ∈ A) (ha_neq_b : a ≠ b) :  
  ∃ (ℓ : set (ℝ × ℝ)), (∃ (L : affine_subspace ℝ (ℝ × ℝ)), L ∈ ℓ ∧ L.direction ∈ unit_vector ℝ (ℝ × ℝ) ∧ ∀ (p : (ℝ × ℝ)), p ≠ a ∧ p ≠ b → (p ∈ L ↔ p ≠ a ∧ p ≠ b)) ∧
  ∃ (S₁ S₂ : set (ℝ × ℝ)), S₁ ∪ S₂ = A ∧ S₁.card = n ∧ S₂.card = n ∧ a ∈ S₁ ∧ b ∈ S₂ :=
sorry

end partition_points_l548_548688


namespace plywood_perimeter_difference_l548_548117

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548117


namespace probability_between_zero_and_one_is_point_three_l548_548426

theorem probability_between_zero_and_one_is_point_three
  (X : Type)
  [Normal X 1 2]
  (h1 : ∀ X, P(X ≥ 2) = 0.2) :
  P(0 ≤ X ≤ 1) = 0.3 :=
sorry

end probability_between_zero_and_one_is_point_three_l548_548426


namespace curve_is_circle_l548_548742

theorem curve_is_circle :
  ∀ x y : ℝ, |x - 1| = sqrt (1 - (y + 1)^2) ↔ (x - 1)^2 + (y + 1)^2 = 1 :=
by
  intros
  sorry

end curve_is_circle_l548_548742


namespace calc_625_to_4_div_5_l548_548174

theorem calc_625_to_4_div_5 :
  (625 : ℝ)^(4/5) = 238 :=
sorry

end calc_625_to_4_div_5_l548_548174


namespace train_speed_in_kmh_l548_548503

-- Definitions of conditions
def time_to_cross_platform := 30  -- in seconds
def time_to_cross_man := 17  -- in seconds
def length_of_platform := 260  -- in meters

-- Conversion factor from m/s to km/h
def meters_per_second_to_kilometers_per_hour (v : ℕ) : ℕ :=
  v * 36 / 10

-- The theorem statement
theorem train_speed_in_kmh :
  (∃ (L V : ℕ),
    L = V * time_to_cross_man ∧
    L + length_of_platform = V * time_to_cross_platform ∧
    meters_per_second_to_kilometers_per_hour V = 72) :=
sorry

end train_speed_in_kmh_l548_548503


namespace fifth_inequality_l548_548714

theorem fifth_inequality (h1: 1 / Real.sqrt 2 < 1)
                         (h2: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
                         (h3: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
                         1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := 
sorry

end fifth_inequality_l548_548714


namespace max_value_Ahn_can_get_l548_548508

theorem max_value_Ahn_can_get :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (3 * (500 - n) = 1200) :=
by {
  use 100,
  split,
  { exact le_rfl },
  split,
  { exact nat.le_of_lt_succ (by norm_num) },
  { sorry }
}

end max_value_Ahn_can_get_l548_548508


namespace find_BP_l548_548808

variables (A B C D P : Type) [CircleType A B C D] -- Assuming CircleType represents the circular arrangement and intersect property

def BP_less_DP (BP DP : ℝ) : Prop := BP < DP

theorem find_BP (BP DP AP PC BD : ℝ)
  (h1 : AP = 5)
  (h2 : PC = 2)
  (h3 : BD = 9)
  (h4 : BP_less_DP BP DP)
  (h5 : BP = 1.3) :
  BP = 1.3 :=
sorry

end find_BP_l548_548808


namespace find_c_condition_l548_548391

-- Defining the problem
def rectangle_area : ℕ := 6

def divided_area : ℕ := rectangle_area / 2

-- Line equation through (c,0) and (4,2)
def line_eq (c : ℝ) (x : ℝ) : ℝ := (2 / (4 - c)) * (x - c)

-- Area of triangle formed by (c,0), (4,0), (4,2)
def triangle_area (c : ℝ) : ℝ := (1 / 2) * (4 - c) * 2

-- Condition to check if the area is equal to half the rectangle area
theorem find_c_condition (c : ℝ) : triangle_area c = divided_area ↔ c = 1 :=
by
sorry

end find_c_condition_l548_548391


namespace a_1998_value_l548_548779

-- Define the sequence using the given conditions
def a : ℕ → ℕ
| 1     := 1
| (n+2) := (n+3) * (a (n+1) + List.sum (List.ofFn a (n + 1))) / (n+1)

-- Mathematical equivalence proof problem
theorem a_1998_value : a 1998 = 1999 * 2^1996 :=
by
  sorry

end a_1998_value_l548_548779


namespace fred_grew_9_onions_l548_548384

theorem fred_grew_9_onions 
  (sally_onions : ℕ)
  (sally_fred_gave_sara : ℕ)
  (sally_fred_left : ℕ)
  (fred_onions : ℕ) :
  sally_onions = 5 →
  sally_fred_gave_sara = 4 →
  sally_fred_left = 10 →
  sally_onions + fred_onions = sally_fred_gave_sara + sally_fred_left →
  fred_onions = 9 :=
by
  intros h1 h2 h3 h4
  refine calc fred_onions = 9 : sorry

end fred_grew_9_onions_l548_548384


namespace sum_possible_b_quad_eq_rational_roots_l548_548958

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l548_548958


namespace anna_has_2_fewer_toys_than_amanda_l548_548868

-- Define the variables for the number of toys each person has
variables (A B : ℕ)

-- Define the conditions
def conditions (M : ℕ) : Prop :=
  M = 20 ∧ A = 3 * M ∧ A + M + B = 142

-- The theorem to prove
theorem anna_has_2_fewer_toys_than_amanda (M : ℕ) (h : conditions A B M) : B - A = 2 :=
sorry

end anna_has_2_fewer_toys_than_amanda_l548_548868


namespace simplest_radical_form_l548_548175

theorem simplest_radical_form (q : ℝ) (hq : 0 ≤ q) : 
  sqrt (15 * q) * sqrt (3 * q^2) * sqrt (8 * q^3) = 6 * q^3 * sqrt (10 * q) :=
by sorry

end simplest_radical_form_l548_548175


namespace calculate_expression_l548_548881

theorem calculate_expression : -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (Real.pi / 3) = 6 :=
by
  sorry

end calculate_expression_l548_548881


namespace percentage_subtraction_l548_548789

theorem percentage_subtraction (P : ℝ) : (700 - (P / 100 * 7000) = 700) → P = 0 :=
by
  sorry

end percentage_subtraction_l548_548789


namespace num_pairs_satisfying_equation_l548_548631

theorem num_pairs_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end num_pairs_satisfying_equation_l548_548631


namespace eccentricity_of_ellipse_l548_548744

-- Definitions from the conditions
def parametric_equations : ℝ → ℝ × ℝ :=
  λ φ, (3 * Real.cos φ, 5 * Real.sin φ)

-- Main theorem to prove
theorem eccentricity_of_ellipse :
  (∃ e : ℝ, e = 4 / 5 ∧
     (∀ φ : ℝ, let x := 3 * Real.cos φ,
                   y := 5 * Real.sin φ in
      ((x^2 / 9) + (y^2 / 25) = 1))) :=
by sorry

end eccentricity_of_ellipse_l548_548744


namespace tan_product_identity_l548_548281

noncomputable def tan_deg (θ : ℝ) := Real.tan (θ * Real.pi / 180)

theorem tan_product_identity:
  (∏ k in Finset.range 60, (1 + tan_deg k.succ)) = 2^30 :=
by
  sorry

end tan_product_identity_l548_548281


namespace solve_cubic_equation_l548_548921

theorem solve_cubic_equation :
  ∀ x : ℝ, (x^3 + 2 * (x + 1)^3 + (x + 2)^3 = (x + 4)^3) → x = 3 :=
by
  intro x
  sorry

end solve_cubic_equation_l548_548921


namespace sum_of_valid_b_values_l548_548947

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l548_548947


namespace Danica_additional_cars_l548_548898

theorem Danica_additional_cars (n : ℕ) (row_size : ℕ) (danica_cars : ℕ) (answer : ℕ) :
  row_size = 8 →
  danica_cars = 37 →
  answer = 3 →
  ∃ k : ℕ, (k + danica_cars) % row_size = 0 ∧ k = answer :=
by
  sorry

end Danica_additional_cars_l548_548898


namespace necessary_not_sufficient_condition_l548_548051

-- Definitions based on conditions
def line1 (a : ℝ) : set (ℝ × ℝ) := {p | p.1 - a * p.2 + 3 = 0}
def line2 (a : ℝ) : set (ℝ × ℝ) := {p | a * p.1 - 4 * p.2 + 5 = 0}
def lines_intersect (a : ℝ) : Prop := ∃ p : ℝ × ℝ, p ∈ line1 a ∧ p ∈ line2 a

-- Problem statement
theorem necessary_not_sufficient_condition (a : ℝ) (h : a ≠ 2) : ¬a = -2 → lines_intersect a := 
by sorry

end necessary_not_sufficient_condition_l548_548051


namespace total_days_in_month_eq_l548_548507

-- Definition of the conditions
def took_capsules_days : ℕ := 27
def forgot_capsules_days : ℕ := 4

-- The statement to be proved
theorem total_days_in_month_eq : took_capsules_days + forgot_capsules_days = 31 := by
  sorry

end total_days_in_month_eq_l548_548507


namespace smaller_of_x_y_l548_548734

theorem smaller_of_x_y (x y a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : x * y = c) (h6 : x^2 - b * x + a * y = 0) : min x y = c / a :=
by sorry

end smaller_of_x_y_l548_548734


namespace vector_computation_l548_548890

def v1 : ℤ × ℤ := (3, -5)
def v2 : ℤ × ℤ := (2, -10)
def s1 : ℤ := 4
def s2 : ℤ := 3

theorem vector_computation : s1 • v1 - s2 • v2 = (6, 10) :=
  sorry

end vector_computation_l548_548890


namespace plywood_perimeter_difference_l548_548116

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548116


namespace train_pass_time_l548_548502

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def train_speed_kmph : ℝ := 90 -- km/h
noncomputable def man_speed_kmph : ℝ := 10 -- km/h

-- Convert speeds to m/s
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def man_speed_mps : ℝ := man_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps 

-- Calculate time to pass
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_pass_time : time_to_pass ≈ 8.99 := 
by
  sorry

end train_pass_time_l548_548502


namespace alicia_taxes_l548_548165

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end alicia_taxes_l548_548165


namespace AF_length_l548_548268

-- Define the conditions
def parabola (P : ℝ × ℝ) : Prop := (P.snd)^2 = 4 * P.fst
def focus : ℝ × ℝ := (1, 0)
def directrix (P : ℝ × ℝ) : Prop := P.fst = -1
def A_on_parabola (A : ℝ × ℝ) : Prop := parabola A
def perpendicular_to_directrix (A A1 : ℝ × ℝ) : Prop := A1.fst = -1 ∧ A1.snd = A.snd
def intersects_y_axis (A F S : ℝ × ℝ) : Prop := 
  let k := (A.snd - F.snd) / (A.fst - F.fst) in S.fst = 0 ∧ S.snd = k * (0 - A.fst) + A.snd
def parallel (A F S T : ℝ × ℝ) : Prop := 
  let k_AF := (A.snd - F.snd) / (A.fst - F.fst) in
  let k_ST := (S.snd - T.snd) / (S.fst - T.fst) in
  k_AF = k_ST

-- Define the proof problem
theorem AF_length (A A1 F S T : ℝ × ℝ)
  (hA_on_parabola : A_on_parabola A)
  (hfocus : F = focus)
  (hdirectrix : directrix A1)
  (hperpendicular : perpendicular_to_directrix A A1)
  (hintersects_y_axis : intersects_y_axis A F S)
  (hparallel : parallel A F S T) :
  ∥ F - A ∥ = 4 :=
sorry

end AF_length_l548_548268


namespace plywood_cut_difference_l548_548071

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548071


namespace time_calculation_correct_l548_548327

theorem time_calculation_correct :
  let start_hour := 3
  let start_minute := 0
  let start_second := 0
  let hours_to_add := 158
  let minutes_to_add := 55
  let seconds_to_add := 32
  let total_seconds := seconds_to_add + minutes_to_add * 60 + hours_to_add * 3600
  let new_hour := (start_hour + (total_seconds / 3600) % 12) % 12
  let new_minute := (start_minute + (total_seconds / 60) % 60) % 60
  let new_second := (start_second + total_seconds % 60) % 60
  let A := new_hour
  let B := new_minute
  let C := new_second
  A + B + C = 92 :=
by
  sorry

end time_calculation_correct_l548_548327


namespace smallest_n_l548_548785

theorem smallest_n (n : ℕ) :
  (1 / 4 : ℚ) + (n / 8 : ℚ) > 1 ↔ n ≥ 7 := by
  sorry

end smallest_n_l548_548785


namespace quadrilateral_area_l548_548410

theorem quadrilateral_area (AC BD p q : ℝ) (h1 : AC = BD) (h2 : 2 * p = AC) (h3 : 2 * q = BD) : 
  let KLMN_area := (1 / 2) * p * q in
  KLMN_area = p * q := 
by sorry

end quadrilateral_area_l548_548410


namespace books_read_by_Megan_l548_548712

theorem books_read_by_Megan 
    (M : ℕ)
    (Kelcie : ℕ := M / 4)
    (Greg : ℕ := 2 * (M / 4) + 9)
    (total : M + Kelcie + Greg = 65) :
  M = 32 :=
by sorry

end books_read_by_Megan_l548_548712


namespace eddies_sister_pies_per_day_l548_548542

theorem eddies_sister_pies_per_day 
  (Eddie_daily : ℕ := 3) 
  (Mother_daily : ℕ := 8) 
  (total_days : ℕ := 7)
  (total_pies : ℕ := 119) :
  ∃ (S : ℕ), S = 6 ∧ (Eddie_daily * total_days + Mother_daily * total_days + S * total_days = total_pies) :=
by
  sorry

end eddies_sister_pies_per_day_l548_548542


namespace shortest_distance_to_cabin_l548_548137

variable (C : ℝ × ℝ) (B : ℝ × ℝ)

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem shortest_distance_to_cabin :
  ∀ C B : ℝ × ℝ,
  C = (0, -4) →
  B = (8, -11) →
  distance (0, 4) B = 17 :=
by
  intros C B hC hB
  sorry

end shortest_distance_to_cabin_l548_548137


namespace cake_cut_proof_l548_548047

theorem cake_cut_proof (O F I A B : ℝ) (r : ℝ)
  (angle_OFI : ∠OFI = 45)  -- The angle between OF and FI is 45 degrees
  (FA_eq_FB : F A = F B)    -- The segments FA and FB are equal
  (circle_radius_OA : O A = r)  -- OA is the radius of the circle
  (trig_identity : O F = (F I) * sqrt(2)) -- OF equals FI * sqrt(2)
  (sum_of_squares : F A ^ 2 + F B ^ 2 = 2 * r ^ 2)  -- Sum of squares of segments is double the square of the cake's radius
  : 
  sum_of_squares = 2 * r ^ 2 := 
by
  sorry

end cake_cut_proof_l548_548047


namespace total_amount_spent_l548_548395

def appetizer_cost : ℝ := 10
def entree_cost : ℝ := 20
def number_of_entrees : ℝ := 4
def discount_rate : ℝ := 0.1
def sales_tax_rate : ℝ := 0.08
def tip_rate : ℝ := 0.2
def number_of_people : ℝ := 5

def total_cost (appetizer_cost entree_cost number_of_entrees discount_rate sales_tax_rate tip_rate : ℝ)
               (number_of_people : ℝ) : ℝ :=
  let entrees_cost := entree_cost * number_of_entrees
  let discount := entrees_cost * discount_rate
  let discounted_entrees_cost := entrees_cost - discount
  let subtotal := discounted_entrees_cost + appetizer_cost
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  let rounded_tip := (Real.ceil (tip * 100) / 100)
  total_before_tip + rounded_tip

theorem total_amount_spent :
  total_cost appetizer_cost entree_cost number_of_entrees discount_rate sales_tax_rate tip_rate number_of_people = 106.27 :=
by
  unfold total_cost
  sorry

end total_amount_spent_l548_548395


namespace option_c_has_minimum_value_4_l548_548019

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548019


namespace rational_operation_example_l548_548989

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l548_548989


namespace center_of_symmetry_l548_548923

theorem center_of_symmetry (k : ℤ) : 
  (∃ x : ℝ, x = (k * Real.pi / 4) ∧ 3 * Real.tan(2 * x) = 0) :=
sorry

end center_of_symmetry_l548_548923


namespace problem1_problem2_l548_548523

theorem problem1 :
  ( (1/2) ^ (-2) - 0.01 ^ (-1) + (-(1 + 1/7)) ^ (0)) = -95 := by
  sorry

theorem problem2 (x : ℝ) :
  (x - 2) * (x + 1) - (x - 1) ^ 2 = x - 3 := by
  sorry

end problem1_problem2_l548_548523


namespace sum_of_sin_squares_l548_548894

theorem sum_of_sin_squares : (∑ k in Finset.range 59, Real.sin (3 * (k + 1) * Real.pi / 180) ^ 2) = 30 := 
by
  sorry

end sum_of_sin_squares_l548_548894


namespace find_x2_plus_y2_l548_548547

theorem find_x2_plus_y2 (x y : ℕ) (h1 : xy + x + y = 35) (h2 : x^2 * y + x * y^2 = 306) : x^2 + y^2 = 290 :=
sorry

end find_x2_plus_y2_l548_548547


namespace solve_pow_problem_l548_548879

theorem solve_pow_problem : (-2)^1999 + (-2)^2000 = 2^1999 := 
sorry

end solve_pow_problem_l548_548879


namespace evaluate_cubic_difference_l548_548593

theorem evaluate_cubic_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 :=
by sorry

end evaluate_cubic_difference_l548_548593


namespace investment_time_period_l548_548819

variable (P : ℝ) (r15 r12 : ℝ) (T : ℝ)
variable (hP : P = 15000)
variable (hr15 : r15 = 0.15)
variable (hr12 : r12 = 0.12)
variable (diff : 2250 * T - 1800 * T = 900)

theorem investment_time_period :
  T = 2 := by
  sorry

end investment_time_period_l548_548819


namespace sum_of_possible_values_of_intersection_points_l548_548563

/-- For a given set of four distinct lines in a plane, prove that the 
    sum of all possible values of the number of distinct intersection points 
    of pairs of lines is equal to 19. -/
theorem sum_of_possible_values_of_intersection_points :
  ∀ unique_pairs_of_lines (lines : Finset (Fin 4)
  (h_distinct : lines.card = 4),
  let possible_intersections := {0, 1, 3, 4, 5, 6} in
  (possible_intersections.sum id) = 19 
:= by
  sorry -- proof goes here

end sum_of_possible_values_of_intersection_points_l548_548563


namespace count_twelve_digit_numbers_with_three_consecutive_ones_l548_548278

def has_three_consecutive_ones (n : ℕ) : Prop :=
  ∃ i : Fin (n - 3 + 1), ∀ j : Fin 3, (n / 10^((n - i.val - 1))) % 10 = 1

def valid_digit (d : ℤ) : Prop := d = 1 ∨ d = 2

def twelve_digit_number_with_valid_digits (n : ℤ) : Prop :=
  n ≥ 10^11 ∧ n < 10^12 ∧ ∀ k : Fin 12, valid_digit ((n / 10^k.val) % 10)

theorem count_twelve_digit_numbers_with_three_consecutive_ones :
  ∃ count : ℕ, count = 3592 ∧
  count = (Finset.filter (λ n,
           twelve_digit_number_with_valid_digits n ∧ has_three_consecutive_ones n)
           (Finset.range (10^12))).card :=
by
  sorry

end count_twelve_digit_numbers_with_three_consecutive_ones_l548_548278


namespace plywood_cut_difference_l548_548073

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548073


namespace plywood_cut_difference_l548_548070

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548070


namespace max_handshakes_l548_548811

theorem max_handshakes (total_people : ℕ) (groupA groupB groupC : ℕ)
  (constraint1 : ∀ {p1 p2 : ℕ}, p1 ≠ p2 → p1 < total_people ∧ p2 < total_people → true)
  (constraint2 : ∀ p : ℕ, ¬ (p < total_people ∧ p = p))
  (constraint3 : groupA + groupB + groupC = total_people ∧ groupA = 30 ∧ groupB = 35 ∧ groupC = 35) :
   ∑ x in finset.range(total_people), ∑ y in finset.Ico(x+1, total_people), (1 : ℕ) - (
      if x < groupA ∧ y < groupA then 1
      else if x ≥ groupA ∧ x < groupA + groupB ∧ y ≥ groupA ∧ y < groupA + groupB then 1
      else if x ≥ groupA + groupB ∧ y ≥ groupA + groupB then 1
      else 0
    ) = 3325 :=
by
  sorry

end max_handshakes_l548_548811


namespace can_form_basis_l548_548242

variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b c : α)

-- Non-coplanar vectors condition
variables (h_non_coplanar : linear_independent ℝ ![a, b, c])

-- Definitions of the vector sets in the options A and C
def optionA_vectors := [a + b, a - 2 • b, c]
def optionC_vectors := [a, 2 • b, b - c]

-- Problem statement: proving which sets of vectors can form a basis for the space
theorem can_form_basis (hA : linear_independent ℝ optionA_vectors)
                       (hC : linear_independent ℝ optionC_vectors) :
    (linear_independent ℝ optionA_vectors) ∨ (linear_independent ℝ optionC_vectors) :=
begin
  -- We assume the given linear independence conditions to derive the final result equivalence
  exact or.inl hA ∨ or.inr hC,
  sorry
end

end can_form_basis_l548_548242


namespace part1_part2_l548_548605

noncomputable def f (a x : ℝ) : ℝ := log x + 2 / x + a * x - a - 2

theorem part1 (x : ℝ) : (∀ x > 0, f 1 x ≥ f 1 1) :=
begin
  intros x hx,
  sorry
end

theorem part2 (a : ℝ) : (∀ x ∈ set.Icc 1 3, f a x ≥ 0) ↔ 1 ≤ a :=
begin
  sorry
end

end part1_part2_l548_548605


namespace badminton_members_count_l548_548309

def total_members := 30
def neither_members := 2
def both_members := 6

def members_play_badminton_and_tennis (B T : ℕ) : Prop :=
  B + T - both_members = total_members - neither_members

theorem badminton_members_count (B T : ℕ) (hbt : B = T) :
  members_play_badminton_and_tennis B T → B = 17 :=
by
  intros h
  sorry

end badminton_members_count_l548_548309


namespace club_truncator_probability_l548_548889

theorem club_truncator_probability :
  let num_matches := 6
  let win_prob := (1 : ℚ) / 3
  let lose_prob := (1 : ℚ) / 3
  let tie_prob := (1 : ℚ) / 3
  let total_outcomes := 3 ^ num_matches
  let equal_wins_losses_ways :=
    nat.choose num_matches 3 * nat.choose 3 3 + -- 0 ties
    nat.choose num_matches 2 * nat.choose 2 2 * nat.choose 2 2 + -- 2 ties
    nat.choose num_matches 4 * nat.choose 1 1 + -- 4 ties
    1 -- 6 ties
  let prob_equal_wins_losses := equal_wins_losses_ways / total_outcomes
  let prob_more_wins_than_losses := (1 - prob_equal_wins_losses) / 2
  prob_more_wins_than_losses = 98 / 243 :=
sorry

end club_truncator_probability_l548_548889


namespace gcd_a_b_eq_one_l548_548891

def a : ℕ := 47^5 + 1
def b : ℕ := 47^5 + 47^3 + 1

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l548_548891


namespace option_c_has_minimum_value_4_l548_548010

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548010


namespace remainder_when_divided_by_l548_548456

theorem remainder_when_divided_by (x : ℝ) : 
  let f := λ x, x^9 + 3 in
  (f 2) = 515 :=
by
  let f := λ x, x^9 + 3
  show (f 2) = 515
  sorry

end remainder_when_divided_by_l548_548456


namespace mode_seventh_grade_average_eighth_grade_performance_analysis_l548_548132

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

-- Mode of the seventh-grade scores
theorem mode_seventh_grade : List.mode seventh_grade_scores = 6 := sorry

-- Average of the eighth-grade scores
noncomputable def average (l : List ℕ) : ℝ := (l.sum : ℝ) / (l.length : ℝ)
theorem average_eighth_grade : average eighth_grade_scores = 7.1 := sorry

-- Performance analysis of Xiao Li and Xiao Zhang based on their team's medians
def median (l : List ℕ) : ℝ :=
if l.length % 2 = 0 then
  let sorted := l.qsort (· ≤ ·)
  ((sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) : ℝ) / 2
else
  (l.qsort (· ≤ ·)).get! (l.length / 2)

theorem performance_analysis :
  median seventh_grade_scores = 6 ∧
  median eighth_grade_scores = 7.5 ∧
  7 > median seventh_grade_scores ∧
  7 < median eighth_grade_scores := 
sorry

end mode_seventh_grade_average_eighth_grade_performance_analysis_l548_548132


namespace average_increase_l548_548337

def scores : List ℕ := [92, 85, 90, 95]

def initial_average (s : List ℕ) : ℚ := (s.take 3).sum / 3

def new_average (s : List ℕ) : ℚ := s.sum / s.length

theorem average_increase :
  initial_average scores + 1.5 = new_average scores := 
by
  sorry

end average_increase_l548_548337


namespace solve_a_plus_b_l548_548596

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 * x + x^2 else -x^2 + 2 * x

theorem solve_a_plus_b (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : ∀ x ∈ set.Icc a b, f x ∈ set.Icc (1 / b) (1 / a))
  (h4 : is_odd_function f) :
  a + b = (3 + Real.sqrt 5) / 2 :=
sorry

end solve_a_plus_b_l548_548596


namespace quantity_costs_20_dollars_l548_548646

-- Given conditions
variable (Q : ℝ) (P : ℝ)
variable cost_Q : Q = 20
variable cost_3_5_Q : 3.5 * P = 28

-- Statement of the problem
theorem quantity_costs_20_dollars (Q P : ℝ) (cost_Q : Q = 20) (cost_3_5_Q : 3.5 * P = 28) :
  Q = 2.5 * P :=
sorry

end quantity_costs_20_dollars_l548_548646


namespace rent_to_expenses_ratio_l548_548335

theorem rent_to_expenses_ratio
  (salary rent remaining food_travel total_expenses : ℕ)
  (h_salary : salary = 5000)
  (h_rent : rent = 1200)
  (h_remaining : remaining = 2000)
  (h_total_expenses : total_expenses = salary - remaining)
  (h_food_travel : food_travel = total_expenses - rent) :
  rent / (nat.gcd rent food_travel) = 2 ∧ food_travel / (nat.gcd rent food_travel) = 3 :=
by
  sorry

end rent_to_expenses_ratio_l548_548335


namespace lines_tangent_form_parallelogram_l548_548896

-- Variables for points
variables (A B C D X I₁ I₂ V : Type)

-- Definitions of conditions:
variables [Trapezoid ABCD] (BC_parallel_AD : BC ∥ AD) (BC_lt_AD : BC < AD)
variables (ω₁ : Incircle (X, B, C)) (ω₂ : Excircle (X, A, D)) (ω₂_tangent_AD : Tangent_to_segment ω₂ AD)
variables (a : Tangent_line_to ω₁ A) (d : Tangent_line_to ω₁ D) (a_d_distinct_from_AB_CD : a ≠ AB ∧ a ≠ CD ∧ d ≠ AB ∧ d ≠ CD)
variables (b : Tangent_line_to ω₂ B) (c : Tangent_line_to ω₂ C) (b_c_distinct_from_AB_CD : b ≠ AB ∧ b ≠ CD ∧ c ≠ AB ∧ c ≠ CD)
variables (lines_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

-- Statement to prove that lines (a, b, c, d) form a parallelogram
theorem lines_tangent_form_parallelogram :
  (a ∥ c) ∧ (b ∥ d) :=
by sorry

end lines_tangent_form_parallelogram_l548_548896


namespace number_of_valid_subsets_l548_548524

-- We define the set
def T : set ℤ := {x | -10 ≤ x ∧ x ≤ 10}

-- Define the conditions for a valid subset S:
def valid_subset (S : set ℤ) : Prop :=
S ⊆ T ∧ S ≠ ∅ ∧ ∃ a ∈ S, |S| + a = 0 ∧ ∃ b ∈ S, b = 0

-- The theorem we want to prove:
theorem number_of_valid_subsets :
  { S : set ℤ | valid_subset S }.to_finset.card = 335 :=
sorry

end number_of_valid_subsets_l548_548524


namespace area_fraction_l548_548045

-- Definition of points A, B, C with coordinates
def A : (ℝ × ℝ) := (2, 0)
def B : (ℝ × ℝ) := (8, 12)
def C : (ℝ × ℝ) := (14, 0)

-- Definition of points X, Y, Z with coordinates
def X : (ℝ × ℝ) := (6, 0)
def Y : (ℝ × ℝ) := (8, 4)
def Z : (ℝ × ℝ) := (10, 0)

-- Calculate the base and height for the triangles
def base_ABC := (C.1 - A.1)
def height_ABC := B.2
def area_ABC := (1 / 2) * base_ABC * height_ABC

def base_XYZ := (Z.1 - X.1)
def height_XYZ := Y.2
def area_XYZ := (1 / 2) * base_XYZ * height_XYZ

-- Proof statement: the fraction of the area of triangle ABC that is the area of triangle XYZ
theorem area_fraction : area_XYZ / area_ABC = (1 / 9) := 
sorry

end area_fraction_l548_548045


namespace remaining_seat_number_l548_548303

theorem remaining_seat_number (total_students : ℕ) (sample_size : ℕ) (interval : ℕ)
  (seats_in_sample : set ℕ) :
  total_students = 52 →
  sample_size = 4 →
  interval = total_students / sample_size →
  (3 ∈ seats_in_sample ∧ 29 ∈ seats_in_sample ∧ 42 ∈ seats_in_sample) →
  ∃ seat_number : ℕ, seat_number ∈ seats_in_sample ∧ seat_number = 16 :=
begin
  intros,
  sorry
end

end remaining_seat_number_l548_548303


namespace min_value_x2_sub_xy_add_y2_l548_548900

/-- Given positive real numbers x and y such that x^2 + xy + 3y^2 = 10, 
prove that the minimum value of x^2 - xy + y^2 is 2. -/
theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + x * y + 3 * y^2 = 10) : 
  ∃ (value : ℝ), value = x^2 - x * y + y^2 ∧ value = 2 := 
by 
  sorry

end min_value_x2_sub_xy_add_y2_l548_548900


namespace rectangle_to_square_l548_548844

-- Definitions based on conditions
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 3
def area : ℕ := rectangle_width * rectangle_height
def parts : ℕ := 3
def part_area : ℕ := area / parts
def square_side : ℕ := Nat.sqrt area

-- Theorem to restate the problem
theorem rectangle_to_square : (area = 36) ∧ (part_area = 12) ∧ (square_side = 6) ∧
  (rectangle_width / parts = 4) ∧ (rectangle_height = 3) ∧ 
  ((rectangle_width / parts * parts) = rectangle_width) ∧ (parts * rectangle_height = square_side ^ 2) := by
  -- Placeholder for proof
  sorry

end rectangle_to_square_l548_548844


namespace find_point_C_l548_548984

def point (x y : ℝ) := (x, y)

def distance (p1 p2: ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def line_point (A B : ℝ × ℝ) (lambda : ℝ) : ℝ × ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  ((lambda * x2 + (1 - lambda) * x1) / (lambda + 1), (lambda * y2 + (1 - lambda) * y1) / (lambda + 1))

theorem find_point_C :
  let A := point 1 (-1)
  let B := point (-4) 5
  let AB := distance A B 
  let C1 := (16, -19)
  let C2 := (-14, 17)
  (distance A C1 = 3 * AB ∨ distance A C2 = 3 * AB) ∧
  ((distance A C1 = 3 * AB → C1 = (16, -19)) ∨ (distance A C2 = 3 * AB → C2 = (-14, 17))) :=
by
  sorry

end find_point_C_l548_548984


namespace number_of_valid_row_lengths_l548_548477

-- Definition of the problem conditions
def choir_members : ℕ := 90
def valid_row_lengths (n : ℕ) : Prop := 5 ≤ n ∧ n ≤ 15 ∧ n ∣ choir_members

-- The problem statement
theorem number_of_valid_row_lengths : (finset.filter valid_row_lengths (finset.divisors choir_members)).card = 5 := by sorry

end number_of_valid_row_lengths_l548_548477


namespace smallest_n_nonabelian_group_order_l548_548802

open Nat

theorem smallest_n_nonabelian_group_order (p : ℕ) (hp : Prime p) :
  ∃ G : Type, Group G ∧ ¬ IsAbelian G ∧ (∃ n, Fintype.card G = p^n) ∧ (∀ m, Fintype.card G = p^m → 3 ≤ m) :=
by
  sorry

end smallest_n_nonabelian_group_order_l548_548802


namespace cost_of_dozen_pens_is_2220_l548_548409

noncomputable def cost_of_one_dozen_pens
  (cost_3_pens_some_pencils : ℝ)
  (pen_pencil_ratio : ℝ) : ℝ :=
  let x := 37 in
  60 * 5 * x

theorem cost_of_dozen_pens_is_2220
  (h1 : cost_of_one_dozen_pens 200 (5)) :
  cost_of_one_dozen_pens 200 (5) = 2220 :=
by
  sorry

end cost_of_dozen_pens_is_2220_l548_548409


namespace initial_student_count_l548_548371

theorem initial_student_count (X : ℝ) 
    (h1 : 0 < X)
    (h2 : 1.1799 * X + 100 = 980) :
    X ≈ 746 :=
by
  sorry

end initial_student_count_l548_548371


namespace solution_l548_548130

def seventh_grade_scores := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

theorem solution : 
  (mode seventh_grade_scores = 6) ∧ 
  (average eighth_grade_scores = 7.1) ∧ 
  (performance 7 seventh_grade_scores = "above average") ∧ 
  (performance 7 eighth_grade_scores = "below average") := 
  by
    sorry

end solution_l548_548130


namespace error_committed_nearest_percent_l548_548486

theorem error_committed_nearest_percent (x : ℝ) (hx : 0 < x) : 
  (|8 * x - x / 8| / (8 * x) * 100) ≈ 98 :=
by
  sorry

end error_committed_nearest_percent_l548_548486


namespace minimum_knights_to_remove_l548_548193

-- Define the chessboard and conditions
def Chessboard := List (List (Option Nat)) -- Represents an 8x8 chessboard where each spot can have a knight (Nat) or be empty (None)
def Knight := Nat

-- Function to check the number of knights a given knight is attacking
def number_of_attacks (board : Chessboard) (x y : Nat) : Nat := 
  sorry -- Placeholder function to calculate the number of attacks

-- Define the problem statement: Minimum knights to be removed
theorem minimum_knights_to_remove (board : Chessboard) : 
  (∀ x y, number_of_attacks board x y ≠ 3) → (∃ k ≥ 8, ∀ x y, number_of_attacks (remove_knights board k) x y ≠ 3) :=
sorry

end minimum_knights_to_remove_l548_548193


namespace parabola_vertex_l548_548758

theorem parabola_vertex : 
  ∀ (x y : ℝ), y = -2 * x^2 + 3 → (x, y) = (0, 3) :=
begin
  intros x y h,
  have hx : x = 0, sorry,
  rw hx at h,
  have hy : y = 3, sorry,
  rw hy,
end

end parabola_vertex_l548_548758


namespace num_pairs_of_positive_integers_eq_77_l548_548616

theorem num_pairs_of_positive_integers_eq_77 : 
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.finite ∧
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.to_finset.card = 2 := 
by 
  sorry

end num_pairs_of_positive_integers_eq_77_l548_548616


namespace perimeters_positive_difference_l548_548097

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548097


namespace avg_weight_increase_l548_548403

theorem avg_weight_increase
  (A : ℝ) -- Initial average weight
  (n : ℕ) -- Initial number of people
  (w_old : ℝ) -- Weight of the person being replaced
  (w_new : ℝ) -- Weight of the new person
  (h_n : n = 8) -- Initial number of people is 8
  (h_w_old : w_old = 85) -- Weight of the replaced person is 85
  (h_w_new : w_new = 105) -- Weight of the new person is 105
  : ((8 * A + (w_new - w_old)) / 8) - A = 2.5 := 
sorry

end avg_weight_increase_l548_548403


namespace slope_of_intersection_line_l548_548969

-- Define the conditions of the problem
def line1 (u x y : ℝ) : Prop := 2 * x + 3 * y = 8 * u + 4
def line2 (u x y : ℝ) : Prop := 3 * x + 2 * y = 9 * u + 1

-- Prove the slope of the line on which all points of intersection (x, y) of the two lines lie
theorem slope_of_intersection_line (u : ℝ) : 
  ∀ (x y : ℝ), line1 u x y ∧ line2 u x y → y = (6 * x + 28) / 47 :=
begin
  sorry
end

end slope_of_intersection_line_l548_548969


namespace leveling_cost_correct_l548_548804

noncomputable def base : ℝ := 54
noncomputable def height : ℝ := 24
noncomputable def rate_per_10_sqm : ℝ := 50

def area (base height : ℝ) : ℝ := base * height
def cost_per_sqm (rate_per_10_sqm : ℝ) : ℝ := rate_per_10_sqm / 10
def total_cost (area cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

theorem leveling_cost_correct :
  total_cost (area base height) (cost_per_sqm rate_per_10_sqm) = 6480 :=
  by sorry

end leveling_cost_correct_l548_548804


namespace triangle_PD_PC_eq_DE_DC_l548_548321

theorem triangle_PD_PC_eq_DE_DC
  (A B C D E M N P : Type)
  [triangle ABC]
  (h1 : is_altitude C D AB)
  (h2 : intersects_semicircle C D E AB)
  (h3 : projections D AC BC M N)
  (h4 : intersects_segment MN CD P) :
  PD / PC = DE / DC := 
sorry

end triangle_PD_PC_eq_DE_DC_l548_548321


namespace find_f_neg_one_l548_548255

def f (x : ℝ) (m : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)

theorem find_f_neg_one (m : ℝ) (h_odd : ∀ x, f x m = -f (-x) m) (h_m : m = -1) : f (-1) m = -3 := by
  sorry

end find_f_neg_one_l548_548255


namespace part_one_part_two_l548_548231

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = n + 1

def sequence_S (S : ℕ → ℕ) : Prop :=
  ∀ n, S (n + 1) / S n = (n + 2) / n

def sequence_b (b : ℕ → ℕ) : Prop :=
  b 2 = 2 ∧ ∀ n, b 1 * b 2 * b 3 * b 4 * b 5 = 2^10 ∧
  ∀ k, b (k + 1) = 2^(k - 1)

def sequence_c (c : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  (a_n a_next_n b_next_n) := 
  (2 + a_n) / (a_n * a_next_n * b_next_n).

theorem part_one (a : ℕ → ℕ) (S : ℕ → ℕ) :
    sequence_S S ∧ a 1 = 1 ∧ (∀ n, a n = n) :=
  sorry

theorem part_two (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ → ℕ → ℕ → ℕ)
    [sequence_a a] [sequence_b b] :
    ∀ n, c a(n) (a n + 1) (b n + 1) (c 1 + c 2 + ⋯ + c n) < 1 :=
  sorry

end part_one_part_two_l548_548231


namespace infinitely_many_positive_integers_l548_548797

open Nat

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 1, (n+1)^3 * a (n+1) = 2 * n^2 * (2 * n + 1) * a n + 2 * (3 * n + 1)

theorem infinitely_many_positive_integers (a : ℕ → ℤ) (h : sequence a) : 
  ∃ᶠ n in at_top, a n > 0 :=
sorry

end infinitely_many_positive_integers_l548_548797


namespace unique_n_exists_pos_integers_l548_548533

theorem unique_n_exists_pos_integers (n : ℕ) (h₁ : ∀ (n : ℕ), n > 0) : n = 3 :=
  begin
    sorry
  end

end unique_n_exists_pos_integers_l548_548533


namespace distance_between_tangent_circles_l548_548181

noncomputable def distance_between_closest_points (center1 center2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  let distance_centers := Math.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - (r1 + r2)

theorem distance_between_tangent_circles :
  distance_between_closest_points (4, 5) (20, 8) 5 8 = Math.sqrt 265 - 13 :=
by
  sorry

end distance_between_tangent_circles_l548_548181


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548034

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548034


namespace geometric_sequence_n_l548_548673

-- Definition of the conditions

-- a_1 + a_n = 82
def condition1 (a₁ an : ℕ) : Prop := a₁ + an = 82
-- a_3 * a_{n-2} = 81
def condition2 (a₃ aₙm2 : ℕ) : Prop := a₃ * aₙm2 = 81
-- S_n = 121
def condition3 (Sₙ : ℕ) : Prop := Sₙ = 121

-- Prove n = 5 given the above conditions
theorem geometric_sequence_n (a₁ a₃ an aₙm2 Sₙ n : ℕ)
  (h1 : condition1 a₁ an)
  (h2 : condition2 a₃ aₙm2)
  (h3 : condition3 Sₙ) :
  n = 5 :=
sorry

end geometric_sequence_n_l548_548673


namespace mode_of_scores_is_102_l548_548428

def scores : List ℕ := [51, 51, 65, 72, 74, 80, 83, 86, 86, 91, 95, 95, 95, 98, 98, 98, 102, 102, 102, 102, 104, 110, 110, 110]

theorem mode_of_scores_is_102 : Multiset.mode scores = 102 := by
  sorry

end mode_of_scores_is_102_l548_548428


namespace minimum_choir_members_l548_548820

theorem minimum_choir_members:
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) → n ≤ m) → n = 990 :=
by
  sorry

end minimum_choir_members_l548_548820


namespace min_abs_sum_of_diffs_l548_548357

theorem min_abs_sum_of_diffs (x : ℝ) (α β : ℝ)
  (h₁ : α * α - 6 * α + 5 = 0)
  (h₂ : β * β - 6 * β + 5 = 0)
  (h_ne : α ≠ β) :
  ∃ m, ∀ x, m = min (|x - α| + |x - β|) :=
by
  use (4)
  sorry

end min_abs_sum_of_diffs_l548_548357


namespace product_evaluation_l548_548908

theorem product_evaluation :
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 :=
by sorry

end product_evaluation_l548_548908


namespace percentage_books_returned_l548_548485

theorem percentage_books_returned
    (initial_books : ℝ)
    (end_books : ℝ)
    (loaned_books : ℝ)
    (R : ℝ)
    (Percentage_Returned : ℝ) :
    initial_books = 75 →
    end_books = 65 →
    loaned_books = 50.000000000000014 →
    R = (75 - 65) →
    Percentage_Returned = (R / loaned_books) * 100 →
    Percentage_Returned = 20 :=
by
  intros
  sorry

end percentage_books_returned_l548_548485


namespace students_can_be_helped_on_fourth_day_l548_548494

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end students_can_be_helped_on_fourth_day_l548_548494


namespace possible_locations_area_l548_548228

-- Define the conditions as given in the problem
variables (A B C D P : ℝ × ℝ)
variable (rect : set (ℝ × ℝ)) 
hypothesis (H0 : A = (0, 0))
hypothesis (H1 : B = (48, 0))
hypothesis (H2 : C = (48, 96))
hypothesis (H3 : D = (0, 96))
hypothesis (H4 : ∀ (x : ℝ × ℝ), x ∈ rect ↔ x = A ∨ x = B ∨ x = C ∨ x = D)
hypothesis (H5 : ∀ (x : ℝ × ℝ), x ∈ rect → (fst x ≥ 0 ∧ fst x ≤ 48 ∧ snd x ≥ 0 ∧ snd x ≤ 96))

-- State that the area of all possible locations of P makes the creases non-intersecting is 1152π - 1152√3
theorem possible_locations_area : 
  ∃ k m n : ℕ, 
    ∃ P : set (ℝ × ℝ), 
      (P ⊆ rect) ∧ 
      let area := (1152 * Real.pi - 1152 * Real.sqrt 3) in
      P = {p | valid_location p} ∧ 
      k = 1152 ∧ m = 1152 ∧ n = 3 :=
      sorry

end possible_locations_area_l548_548228


namespace part_one_part_two_l548_548232

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = n + 1

def sequence_S (S : ℕ → ℕ) : Prop :=
  ∀ n, S (n + 1) / S n = (n + 2) / n

def sequence_b (b : ℕ → ℕ) : Prop :=
  b 2 = 2 ∧ ∀ n, b 1 * b 2 * b 3 * b 4 * b 5 = 2^10 ∧
  ∀ k, b (k + 1) = 2^(k - 1)

def sequence_c (c : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  (a_n a_next_n b_next_n) := 
  (2 + a_n) / (a_n * a_next_n * b_next_n).

theorem part_one (a : ℕ → ℕ) (S : ℕ → ℕ) :
    sequence_S S ∧ a 1 = 1 ∧ (∀ n, a n = n) :=
  sorry

theorem part_two (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ → ℕ → ℕ → ℕ)
    [sequence_a a] [sequence_b b] :
    ∀ n, c a(n) (a n + 1) (b n + 1) (c 1 + c 2 + ⋯ + c n) < 1 :=
  sorry

end part_one_part_two_l548_548232


namespace number_of_pairs_l548_548627

theorem number_of_pairs :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end number_of_pairs_l548_548627


namespace plywood_perimeter_difference_l548_548079

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548079


namespace graph_properties_l548_548392

theorem graph_properties :
  let p1 := (real.sqrt 3, 0)
  let p2 := (1, 1)
  let p3 := (0, real.sqrt 3)
  (p1.1^3 + p1.1 * p1.2 + p1.2^3 = 3) ∧
  (p2.1^3 + p2.1 * p2.2 + p2.2^3 = 3) ∧
  (p3.1^3 + p3.1 * p3.2 + p3.2^3 = 3) ∧
  ∀ x y : ℝ, (x^3 + x * y + y^3 = 3) → (y^3 + y * x + x^3 = 3) :=
by 
  sorry

end graph_properties_l548_548392


namespace central_angle_of_sector_l548_548594

variable (A : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions: A is the area of the sector, and r is the radius.
def is_sector (A : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  A = (1 / 2) * α * r^2

-- Proof that the central angle α given the conditions is 3π/4.
theorem central_angle_of_sector (h1 : is_sector (3 * Real.pi / 8) 1 α) : 
  α = 3 * Real.pi / 4 := 
  sorry

end central_angle_of_sector_l548_548594


namespace perpendicular_line_eq_l548_548551

-- Define the point and the original line
def point : (ℝ × ℝ) := (1, 1)
def line1 (x y : ℝ) := 3 * x + 4 * y + 2 = 0

-- Define the slope calculation and perpendicular condition
def perpendicular_slope (m : ℝ) := -(1 / m)

-- Define the equation of the line through the point and with the calculated slope
def line2 (x y : ℝ) := 4 * x - 3 * y - 1 = 0

-- The theorem to be proved
theorem perpendicular_line_eq :
  ∃ (line2 : ℝ → ℝ → Prop), 
    (∀ x y, line1 x y → False) →
    (∀ x y, line2 x y → (x = 1 ∧ y = 1)) →
    (∀ x y m, line1 x y → m = -3 / 4 → line2 x y) :=
by
  sorry

end perpendicular_line_eq_l548_548551


namespace least_possible_value_of_expression_l548_548781

noncomputable def min_expression_value (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023

theorem least_possible_value_of_expression :
  ∃ x : ℝ, min_expression_value x = 2022 :=
by
  sorry

end least_possible_value_of_expression_l548_548781


namespace clock_angle_at_2_30_l548_548279

theorem clock_angle_at_2_30 :
  let deg_per_hour := 360 / 12 in
  let minute_hand_angle := 180 in
  let hour_hand_angle := 2 * deg_per_hour + deg_per_hour / 2 in
  min ((minute_hand_angle - hour_hand_angle).nat_abs) (360 - (minute_hand_angle - hour_hand_angle).nat_abs) = 105 :=
by
  sorry

end clock_angle_at_2_30_l548_548279


namespace simplify_expression_l548_548286

theorem simplify_expression (y : ℝ) : (y - 2) ^ 2 + 2 * (y - 2) * (4 + y) + (4 + y) ^ 2 = 4 * (y + 1) ^ 2 := 
by 
  sorry

end simplify_expression_l548_548286


namespace no_good_polynomial_for_large_degree_l548_548147

noncomputable def is_good_polynomial (P : Polynomial ℤ) : Prop :=
  P.monic ∧ (∀ z ∈ P.roots, abs z ≤ 0.99) ∧ (∀ z ∈ P.roots, ∀ w ∈ P.roots, z ≠ w → w ≠ z)

theorem no_good_polynomial_for_large_degree :
  ∃ N : ℕ, ∀ n > N, ∀ P : Polynomial ℤ, P.degree = n → ¬ is_good_polynomial P :=
sorry

end no_good_polynomial_for_large_degree_l548_548147


namespace percentage_of_brand_z_l548_548166

noncomputable def percentage_brand_z_gasoline : ℕ := 43.75

def initial_fill_capacity (Z Y : ℚ) := Z = 1 ∧ Y = 0
def first_refill (Z Y : ℚ) := Z = 3/4 ∧ Y = 1/4
def second_refill (Z Y : ℚ) := Z = (3/8 + 1/2) ∧ Y = 1/8
def third_refill (Z Y : ℚ) := Z = 7/16 ∧ Y = (1/16 + 1/2)

theorem percentage_of_brand_z : 
  ∀ Z Y : ℚ,
  initial_fill_capacity Z Y →
  first_refill Z Y →
  second_refill Z Y →
  third_refill Z Y →
  Z / (Z + Y) * 100 = percentage_brand_z_gasoline := 
by
  sorry

end percentage_of_brand_z_l548_548166


namespace xiao_ying_performance_l548_548476

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50

def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

-- Define the function that calculates the weighted average
def semester_performance (rw mw fw rs ms fs : ℝ) : ℝ :=
  rw * rs + mw * ms + fw * fs

-- The theorem that the weighted average of the scores is 90
theorem xiao_ying_performance : semester_performance regular_weight midterm_weight final_weight regular_score midterm_score final_score = 90 := by
  sorry

end xiao_ying_performance_l548_548476


namespace calculate_coffee_cups_l548_548769

theorem calculate_coffee_cups (caffeine_per_cup goal caffeine_over_goal : ℕ)
(h1 : caffeine_per_cup = 80)
(h2 : goal = 200)
(h3 : caffeine_over_goal = 40)
: caffeine_per_cup * 3 = goal + caffeine_over_goal :=
by 
    rw [h1, h2, h3]
    trivial

end calculate_coffee_cups_l548_548769


namespace domain_of_tan_sub_pi_over_3_l548_548925

def is_domain_x (x : ℝ) : Prop :=
  ∀ (k : ℤ), x ≠ k * Real.pi + 5 * Real.pi / 6

theorem domain_of_tan_sub_pi_over_3 :
  ∀ x : ℝ, (is_domain_x x) ↔ ∀ (k : ℤ), x ≠ k * Real.pi + 5 * Real.pi / 6 := 
by {
  intro x,
  refine ⟨_, _⟩,
  { intro h, exact h },
  { intro h, exact h }
}

end domain_of_tan_sub_pi_over_3_l548_548925


namespace rectangle_perimeter_l548_548488

theorem rectangle_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : a * b = 2 * a + 2 * b + 12) : 
    2 * (a + b) = 32 :=
begin
  sorry
end

end rectangle_perimeter_l548_548488


namespace plywood_perimeter_difference_l548_548085

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548085


namespace sum_possible_b_quad_eq_rational_roots_l548_548962

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l548_548962


namespace cars_in_garage_l548_548762

theorem cars_in_garage (c : ℕ) 
  (bicycles : ℕ := 20) 
  (motorcycles : ℕ := 5) 
  (total_wheels : ℕ := 90) 
  (bicycle_wheels : ℕ := 2 * bicycles)
  (motorcycle_wheels : ℕ := 2 * motorcycles)
  (car_wheels : ℕ := 4 * c) 
  (eq : bicycle_wheels + car_wheels + motorcycle_wheels = total_wheels) : 
  c = 10 := 
by 
  sorry

end cars_in_garage_l548_548762


namespace divisibility_by_37_l548_548529

theorem divisibility_by_37 (a b c : ℕ) :
  (100 * a + 10 * b + c) % 37 = 0 → 
  (100 * b + 10 * c + a) % 37 = 0 ∧
  (100 * c + 10 * a + b) % 37 = 0 :=
by
  sorry

end divisibility_by_37_l548_548529


namespace power_mod_remainder_l548_548930

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l548_548930


namespace perimeters_positive_difference_l548_548096

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548096


namespace perimeters_positive_difference_l548_548100

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548100


namespace plywood_cut_perimeter_difference_l548_548108

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548108


namespace perimeter_difference_l548_548063

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548063


namespace difference_in_square_sides_l548_548135

-- Definition of conditions
variables {R : Type*} [OrderedField R]

def chord_distance (h : R) : Prop :=
  ∀ (r : R), r > 0 → ∃ (θ : R) (tanα : R), θ = real.atan tanα ∧ 2 * tanα = 8 * h / (5 * r)

-- Theorem statement
theorem difference_in_square_sides (h : R) (r : R) (h_pos : r > 0) : ∃ (difference : R), difference = 8 * h / 5 :=
by {
  sorry
}

end difference_in_square_sides_l548_548135


namespace probability_laurent_greater_chloe_l548_548179

open Probability Theory

noncomputable def chloe_distribution : ProbabilityMeasure ℝ := uniform 0 3000

noncomputable def laurent_distribution : ProbabilityMeasure ℝ := uniform 0 4000

theorem probability_laurent_greater_chloe :
  P(y > x) = 5 / 8 :=
by
  let x := random_variable chloe_distribution
  let y := random_variable laurent_distribution
  have h_uniform_x: x ~ U(0, 3000) := sorry
  have h_uniform_y: y ~ U(0, 4000) := sorry
  sorry

end probability_laurent_greater_chloe_l548_548179


namespace stratified_sampling_class2_l548_548182

theorem stratified_sampling_class2 (students_class1 : ℕ) (students_class2 : ℕ) (total_samples : ℕ) (h1 : students_class1 = 36) (h2 : students_class2 = 42) (h_tot : total_samples = 13) : 
  (students_class2 / (students_class1 + students_class2) * total_samples = 7) :=
by
  sorry

end stratified_sampling_class2_l548_548182


namespace trig_identity_l548_548810

theorem trig_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 :=
sorry

end trig_identity_l548_548810


namespace plywood_perimeter_difference_l548_548091

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548091


namespace sum_m_n_zero_l548_548579

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l548_548579


namespace probability_even_gx_l548_548649

noncomputable def f (a b x : ℝ) : ℝ := log (x^2 + a * x + 2 * b)
noncomputable def g (a b x : ℝ) : ℝ := (a^x - b^(-x)) / ((a + b) * x)

-- Mathematical conditions and statements
theorem probability_even_gx (a b : ℝ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) :
  (∀ x : ℝ, 0 < x^2 + a * x + 2 * b) →
  (∃ p : ℝ, p = 6 / 26 ∧ p = 3 / 13) :=
by
  sorry

end probability_even_gx_l548_548649


namespace sum_of_b_values_l548_548956

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l548_548956


namespace product_of_roots_l548_548206

theorem product_of_roots :
  (let a := 36
   let b := -24
   let c := -120
   a ≠ 0) →
  let roots_product := c / a
  roots_product = -10/3 :=
by
  sorry

end product_of_roots_l548_548206


namespace bus_students_after_fifth_stop_l548_548644

theorem bus_students_after_fifth_stop :
  let initial := 72
  let firstStop := (2 / 3 : ℚ) * initial
  let secondStop := (2 / 3 : ℚ) * firstStop
  let thirdStop := (2 / 3 : ℚ) * secondStop
  let fourthStop := (2 / 3 : ℚ) * thirdStop
  let fifthStop := fourthStop + 12
  fifthStop = 236 / 9 :=
by
  sorry

end bus_students_after_fifth_stop_l548_548644


namespace sum_possible_b_quad_eq_rational_roots_l548_548960

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l548_548960


namespace person2_speed_l548_548441

variables (v_1 : ℕ) (v_2 : ℕ)

def meet_time := 4
def catch_up_time := 16

def meet_equation : Prop := v_1 + v_2 = 22
def catch_up_equation : Prop := v_2 - v_1 = 4

theorem person2_speed :
  meet_equation v_1 v_2 → catch_up_equation v_1 v_2 →
  v_1 = 6 → v_2 = 10 :=
by
  intros h1 h2 h3
  sorry

end person2_speed_l548_548441


namespace sin_double_angle_half_l548_548634

theorem sin_double_angle_half (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_half_l548_548634


namespace pyramid_base_length_of_tangent_hemisphere_l548_548826

noncomputable def pyramid_base_side_length (radius height : ℝ) (tangent : ℝ → ℝ → Prop) : ℝ := sorry

theorem pyramid_base_length_of_tangent_hemisphere 
(r h : ℝ) (tangent : ℝ → ℝ → Prop) (tangent_property : ∀ x y, tangent x y → y = 0) 
(h_radius : r = 3) (h_height : h = 9) 
(tangent_conditions : tangent r h → tangent r h) : 
  pyramid_base_side_length r h tangent = 9 :=
sorry

end pyramid_base_length_of_tangent_hemisphere_l548_548826


namespace sequence_no_three_consecutive_primes_l548_548213

theorem sequence_no_three_consecutive_primes
  (a b : ℕ) (h : 1 < b < a) :
  ¬ ∃ (n : ℕ), prime (x_n a b n) ∧ prime (x_n a b (n + 1)) ∧ prime (x_n a b (n + 2))
  where x_n (a b : ℕ) (n : ℕ) := (a ^ n - 1) / (b ^ n - 1) := sorry

end sequence_no_three_consecutive_primes_l548_548213


namespace cubic_function_properties_l548_548358

def cubic_function (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

def cubic_derivative (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) (H1 : cubic_derivative a b (-1) = 0)
  (H2 : cubic_derivative a b 2 = 9) (H3 : cubic_function a b c 2 = 20)
  (H4 : ∀ x, ∃ ɛ > 0, cubic_function a b c x = 0 → -27 < c ∧ c < 5) :
  -27 < c ∧ c < 5 :=
by
  sorry

end cubic_function_properties_l548_548358


namespace length_of_train_l548_548861

-- Definitions used in the original problem:
variables (L : ℝ) (V : ℝ) (t_pole : ℝ := 18) (t_platform : ℝ := 39) (platform_length : ℝ := 350)

-- Conditions from the problem:
def speed_of_train := L / t_pole
def time_cross_platform := (L + platform_length) / V

-- Correct answer to be proved:
theorem length_of_train : L = 300 := by
  -- inferring speed from crossing time of the signal pole
  let V := L / t_pole
  -- equating total distance covered when crossing platform to time taken
  have h1 : L + platform_length = V * t_platform,
  -- substituting calculated speed (V)
  have h2 : L + platform_length = (L / t_pole) * t_platform,
  -- solving for L
  sorry

end length_of_train_l548_548861


namespace determine_angle_B_l548_548323

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)
  ∧ (a = 8)
  ∧ (b = Real.sqrt 3)

theorem determine_angle_B (A B C : ℝ) (a b c : ℝ)
  (h : problem_statement A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
by 
  sorry

end determine_angle_B_l548_548323


namespace no_tiling_with_t_blocks_l548_548883

-- Define the structure of the T-shaped piece
def TBlock (x y : ℕ) : set (ℕ × ℕ) :=
  {(x, y), (x+1, y), (x+2, y), (x+1, y+1)}

-- Define the chessboard as a set of coordinate pairs
def Chessboard : set (ℕ × ℕ) :=
  {p | p.1 < 10 ∧ p.2 < 10}

-- Define the proposition: the chessboard cannot be tiled with T blocks
theorem no_tiling_with_t_blocks :
  ¬ (∃ (tiles : fin 25 → set (ℕ × ℕ)), (∀ i, tiles i = TBlock (i.1) (i.2)) ∧
      Finset.univ.bUnion tiles = Chessboard) :=
sorry

end no_tiling_with_t_blocks_l548_548883


namespace percentage_increase_area_l548_548464

theorem percentage_increase_area (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A := L * W,
      L' := 1.20 * L,
      W' := 1.20 * W,
      A' := L' * W' in
  ((A' - A) / A) * 100 = 44 :=
by
  -- the proof follows here.
  let A := L * W,
      L' := 1.20 * L,
      W' := 1.20 * W,
      A' := L' * W'
  sorry

end percentage_increase_area_l548_548464


namespace percentage_of_fair_haired_employees_who_are_women_l548_548470

variable (E : ℝ) -- Total number of employees
variable (h1 : 0.1 * E = women_with_fair_hair_E) -- 10% of employees are women with fair hair
variable (h2 : 0.25 * E = fair_haired_employees_E) -- 25% of employees have fair hair

theorem percentage_of_fair_haired_employees_who_are_women :
  (women_with_fair_hair_E / fair_haired_employees_E) * 100 = 40 :=
by
  sorry

end percentage_of_fair_haired_employees_who_are_women_l548_548470


namespace painting_equation_l548_548865

def painting_problem := 
  ∃ (t : ℝ), 
    (Alice_rate : ℝ) (Bob_rate : ℝ) (combined_rate : ℝ),
     Alice_rate = 1 / 4 ∧ 
     Bob_rate = 1 / 6 ∧ 
     combined_rate = Alice_rate + Bob_rate ∧ 
     (combined_rate * (t - 1) = 1)

theorem painting_equation : painting_problem :=
  sorry

end painting_equation_l548_548865


namespace Ariane_victory_l548_548695

noncomputable def ensuresVictory (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ (s ∈ S), s ∈ (Finset.range 31).erase 0) ∧
  (S.card = n) ∧
  ∀ d : ℕ, d ≥ 2 → ¬∃ s1 s2 ∈ S, s1 ≠ s2 ∧ d ∣ s1 ∧ d ∣ s2

theorem Ariane_victory :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 30 → ensuresVictory n ↔ 1 ≤ n ∧ n ≤ 11 :=
by
  intro n
  intro hn
  sorry

end Ariane_victory_l548_548695


namespace modulus_remainder_l548_548190

namespace Proof

def a (n : ℕ) : ℕ := 88134 + n

theorem modulus_remainder :
  (2 * ((a 0)^2 + (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2)) % 11 = 3 := by
  sorry

end Proof

end modulus_remainder_l548_548190


namespace cone_volume_l548_548060

theorem cone_volume (n : ℕ) (R : ℝ) (h : n > 0) :
  let V := π * R^3 *
            (3 + real.sqrt (1 - 8 * (real.sin (π / n))^2))^3 *
            (1 + real.sqrt (1 - 8 * (real.sin (π / n))^2)) /
            (12 * (real.sin (π / n))^2 *
              (1 - 6 * (real.sin (π / n))^2 +
                real.sqrt (1 - 8 * (real.sin (π / n))^2)))
  in
  (n ≠ 9 → V = π * R^3 *
            (3 + real.sqrt (1 - 8 * (real.sin (π / n))^2))^3 *
            (1 + real.sqrt (1 - 8 * (real.sin (π / n))^2)) /
            (12 * (real.sin (π / n))^2 *
              (1 - 6 * (real.sin (π / n))^2 +
                real.sqrt (1 - 8 * (real.sin (π / n))^2)))) ∧
  (n = 9 → V = π * R^3 *
            (3 + real.sqrt (1 - 8 * (real.sin (π / 9))^2))^3 *
            (1 + real.sqrt (1 - 8 * (real.sin (π / 9))^2)) /
            (12 * (real.sin (π / 9))^2 *
              (1 - 6 * (real.sin (π / 9))^2 +
                real.sqrt (1 - 8 * (real.sin (π / 9))^2)))) ∧
         V = π * R^3 *
            (3 - real.sqrt (1 - 8 * (real.sin (π / 9))^2))^3 *
            (1 - real.sqrt (1 - 8 * (real.sin (π / 9))^2)) /
            (12 * (real.sin (π / 9))^2 *
              (1 - 6 * (real.sin (π / 9))^2 -
                real.sqrt (1 - 8 * (real.sin (π / 9))^2)))
 := sorry

end cone_volume_l548_548060


namespace min_sum_of_consecutive_terms_l548_548168

theorem min_sum_of_consecutive_terms (a : Fin 1999 → ℕ)
  (h : ∀ (n : ℕ) (i : Fin (2000 - n)), (∑ j in (Finset.range n).map (λ k, i + k), a j) ≠ 119) :
  (∑ i in Finset.range 1999, a i) ≥ 3903 :=
sorry

end min_sum_of_consecutive_terms_l548_548168


namespace equal_angles_AHE_AHF_l548_548315

open EuclideanGeometry

variables (A B C H D E F : Point)
variables (hAH : ¬ H = A ∧ H ∈ line_through A (foot of (altitude (triangle.mk A B C) A)))
variables (hD : D ∈ line_through A H)
variables (hE : E ∈ line_through B D ∧ E ∈ line_through A C)
variables (hF : F ∈ line_through C D ∧ F ∈ line_through A B)

theorem equal_angles_AHE_AHF (hABC : acute_triangle A B C) :
  ∠ A H E = ∠ A H F :=
sorry

end equal_angles_AHE_AHF_l548_548315


namespace simplify_expression_l548_548882

theorem simplify_expression (a : ℝ) : (2 * a - 3)^2 - (a + 5) * (a - 5) = 3 * a^2 - 12 * a + 34 :=
by
  sorry

end simplify_expression_l548_548882


namespace rectangular_solid_surface_area_l548_548540

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l548_548540


namespace proof_problem_l548_548398

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 3 else
  if x = 2 then 4 else
  if x = 3 then 6 else
  if x = 4 then 8 else
  if x = 5 then 9 else 0

noncomputable def finv (y : ℝ) : ℝ :=
  if y = 3 then 1 else
  if y = 4 then 2 else
  if y = 6 then 3 else
  if y = 8 then 4 else
  if y = 9 then 5 else 0

theorem proof_problem :
  f(f(2)) + f(finv(6)) + finv(finv(4)) = 15 :=
by
  -- Proof goes here
  sorry

end proof_problem_l548_548398


namespace percentage_saved_l548_548152

theorem percentage_saved (saved spent : ℝ) (h_saved : saved = 3) (h_spent : spent = 27) : 
  (saved / (saved + spent)) * 100 = 10 := by
  sorry

end percentage_saved_l548_548152


namespace find_a2019_l548_548258

-- Arithmetic sequence
def a (n : ℕ) : ℤ := sorry -- to be defined later

-- Given conditions
def sum_first_five_terms (a: ℕ → ℤ) : Prop := a 1 + a 2 + a 3 + a 4 + a 5 = 15
def term_six (a: ℕ → ℤ) : Prop := a 6 = 6

-- Question (statement to be proved)
def term_2019 (a: ℕ → ℤ) : Prop := a 2019 = 2019

-- Main theorem to be proved
theorem find_a2019 (a: ℕ → ℤ) 
  (h1 : sum_first_five_terms a)
  (h2 : term_six a) : 
  term_2019 a := 
by
  sorry

end find_a2019_l548_548258


namespace minimum_circles_to_cover_larger_circle_l548_548782

-- Definitions
def radius_larger_circle : ℝ := R
def radius_smaller_circles : ℝ := R / 2

-- Statement to prove
theorem minimum_circles_to_cover_larger_circle
  (R : ℝ) (R_pos : 0 < R) :
  ∃ (n : ℕ), n = 7 ∧ 
  covers (radius_smaller_circles) (radius_larger_circle) n :=
sorry

end minimum_circles_to_cover_larger_circle_l548_548782


namespace JakeThirdTestScore_l548_548737

noncomputable def JakeTestScores (first second third fourth : ℕ) : Prop :=
  let avg := (first + second + third + fourth) / 4 in
  avg = 75 ∧
  first = 80 ∧
  second = first + 10 ∧
  third = fourth ∧
  third = 65

theorem JakeThirdTestScore : ∃ third : ℕ, ∀ first second fourth : ℕ, JakeTestScores first second third fourth :=
begin
  sorry
end

end JakeThirdTestScore_l548_548737


namespace youngest_son_trips_l548_548717

theorem youngest_son_trips 
  (p : ℝ) (n_oldest : ℝ) (c : ℝ) (Y : ℝ)
  (h1 : p = 100)
  (h2 : n_oldest = 35)
  (h3 : c = 4)
  (h4 : p / c = Y) :
  Y = 25 := sorry

end youngest_son_trips_l548_548717


namespace angle_B_in_arithmetic_sequence_l548_548300

theorem angle_B_in_arithmetic_sequence (A B C : ℝ) (h_triangle_sum : A + B + C = 180) (h_arithmetic_sequence : 2 * B = A + C) : B = 60 := 
by 
  -- proof omitted
  sorry

end angle_B_in_arithmetic_sequence_l548_548300


namespace sum_of_b_for_rational_roots_l548_548949

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l548_548949


namespace maximum_visible_sum_of_stacked_cubes_l548_548970

-- Definition of the cube and its faces
def faces : finset ℕ := {1, 2, 3, 5, 10, 20}

-- Function to compute the maximum visible sum of faces for a single cube
noncomputable def max_visible_sum : ℕ :=
  -- The visible faces are the five highest values in the set
  finset.sum (finset.erase (faces : finset ℕ) 1)

-- Maximum possible sum of 16 visible numbers when four cubes are stacked
theorem maximum_visible_sum_of_stacked_cubes :
  let n := 4 * max_visible_sum in
  n = 160 := by
  sorry

end maximum_visible_sum_of_stacked_cubes_l548_548970


namespace correct_option_C_l548_548219

variables (α β : Plane) (l m n : Line)

-- Given conditions
axiom planes_different : α ≠ β
axiom lines_different : l ≠ m ∧ m ≠ n ∧ l ≠ n

-- Proposition to prove
theorem correct_option_C (h1 : α ⊥ β) (h2 : α ∩ β = l) (h3 : m ⊂ α) (h4 : m ⊥ l) : m ⊥ β :=
sorry

end correct_option_C_l548_548219


namespace min_box_height_l548_548661

-- Define the constants based on the given conditions
def ceiling_height : ℕ := 300
def light_bulb_offset : ℕ := 20
def alice_height : ℕ := 160
def alice_reach : ℕ := 50

-- The proof problem statement
theorem min_box_height : 
  let max_alice_reach := alice_height + alice_reach in
  let light_bulb_height := ceiling_height - light_bulb_offset in
  ∃ h : ℕ, max_alice_reach + h = light_bulb_height ∧ h = 70 :=
by 
  sorry -- Proof goes here

end min_box_height_l548_548661


namespace intersection_value_l548_548996

-- We are given two functions and their point of intersection
def y1 (x : ℝ) : ℝ := 3 / x
def y2 (x : ℝ) : ℝ := x - 1

-- a and b are the coordinates of the intersection point
variable (a b : ℝ)

-- Conditions for the point of intersection
axiom intersection_condition : y1 a = b ∧ y2 a = b

-- Statement of the problem
theorem intersection_value : intersection_condition a b → (1 / a - 1 / b = -1 / 3) :=
by 
  sorry

end intersection_value_l548_548996


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548030

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548030


namespace milk_production_per_cow_l548_548517

theorem milk_production_per_cow :
  ∀ (total_cows : ℕ) (milk_price_per_gallon butter_price_per_stick total_earnings : ℝ)
    (customers customer_milk_demand gallons_per_butter : ℕ),
  total_cows = 12 →
  milk_price_per_gallon = 3 →
  butter_price_per_stick = 1.5 →
  total_earnings = 144 →
  customers = 6 →
  customer_milk_demand = 6 →
  gallons_per_butter = 2 →
  (∀ (total_milk_sold_to_customers produced_milk used_for_butter : ℕ),
    total_milk_sold_to_customers = customers * customer_milk_demand →
    produced_milk = total_milk_sold_to_customers + used_for_butter →
    used_for_butter = (total_earnings - (total_milk_sold_to_customers * milk_price_per_gallon)) / butter_price_per_stick / gallons_per_butter →
    produced_milk / total_cows = 4)
:= by sorry

end milk_production_per_cow_l548_548517


namespace prob_A_two_qualified_l548_548299

noncomputable def prob_qualified (p : ℝ) : ℝ := p * p

def qualified_rate : ℝ := 0.8

theorem prob_A_two_qualified : prob_qualified qualified_rate = 0.64 :=
by
  sorry

end prob_A_two_qualified_l548_548299


namespace option_c_has_minimum_value_4_l548_548014

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548014


namespace cats_weight_difference_l548_548751

-- Define the weights of Anne's and Meg's cats
variables (A M : ℕ)

-- Given conditions:
-- 1. Ratio of weights Meg's cat to Anne's cat is 13:21
-- 2. Meg's cat's weight is 20 kg plus half the weight of Anne's cat

theorem cats_weight_difference (h1 : M = 20 + (A / 2)) (h2 : 13 * A = 21 * M) : A - M = 64 := 
by {
    sorry
}

end cats_weight_difference_l548_548751


namespace plywood_perimeter_difference_l548_548082

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548082


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548035

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548035


namespace linesIntersectOnGamma_l548_548467

noncomputable def intersectOnGamma (Γ : Circle) (A B C D E M N K L : Point) :=
  areOnSameCircle [A, B, C, D, E] (Γ) ∧
  intersectionOfLines (lineThrough A E) (lineThrough C D) = M ∧
  intersectionOfLines (lineThrough A B) (lineThrough D C) = N ∧
  parallelToThrough (lineThrough E C) M = lineThrough K ∧
  parallelToThrough (lineThrough B C) N = lineThrough K ∧
  parallelToThrough (lineThrough E D) M = lineThrough L ∧
  parallelToThrough (lineThrough B D) N = lineThrough L ∧
  ∃ T : Point, isOnLine (lineThrough L D) T ∧ isOnLine (lineThrough K C) T ∧ isOnCircle Γ T

theorem linesIntersectOnGamma 
  (Γ : Circle) 
  (A B C D E M N K L : Point) 
  (h1 : areOnSameCircle [A, B, C, D, E] (Γ))
  (h2 : intersectionOfLines (lineThrough A E) (lineThrough C D) = M)
  (h3 : intersectionOfLines (lineThrough A B) (lineThrough D C) = N)
  (h4 : parallelToThrough (lineThrough E C) M = lineThrough K)
  (h5 : parallelToThrough (lineThrough B C) N = lineThrough K)
  (h6 : parallelToThrough (lineThrough E D) M = lineThrough L)
  (h7 : parallelToThrough (lineThrough B D) N = lineThrough L) :
  ∃ T : Point, isOnLine (lineThrough L D) T ∧ isOnLine (lineThrough K C) T ∧ isOnCircle Γ T := by sorry

end linesIntersectOnGamma_l548_548467


namespace interval_monotonically_increasing_a_eq_zero_range_of_a_minimum_value_l548_548604

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem interval_monotonically_increasing_a_eq_zero :
∀ x : ℝ, (1 < x → (∃ δ > 0, ∀ ε < δ, x < x + ε ∧ f x 0 < f (x + ε) 0)) := sorry

theorem range_of_a_minimum_value :
∀ a : ℝ, ((0 ≤ a ∧ a ≤ 2) ↔ (∀ x : ℝ, f 0 a ≤ f x a)) := sorry

end interval_monotonically_increasing_a_eq_zero_range_of_a_minimum_value_l548_548604


namespace plywood_cut_difference_l548_548119

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548119


namespace fixed_point_exists_l548_548966

noncomputable def fixed_point : Prop := ∀ d : ℝ, ∃ (p q : ℝ), (p = -3) ∧ (q = 45) ∧ (q = 5 * p^2 + d * p + 3 * d)

theorem fixed_point_exists : fixed_point :=
by
  sorry

end fixed_point_exists_l548_548966


namespace Bob_wins_l548_548160

open Matrix

noncomputable def bob_has_winning_strategy : Prop :=
  ∃ strategy : (ℕ × ℕ) × ℝ → (ℕ × ℕ) × ℝ,
  ∀ (M : Matrix (Fin 2016) (Fin 2016) ℝ),
    (∀ i j, (i < 2016) → (j < 2016) → M ⟨i, _⟩ ⟨j, _⟩ = strategy ((i, j), M ⟨i, _⟩ ⟨j, _⟩))
    → ∃ M' : Matrix (Fin 2016) (Fin 2016) ℝ,
         (∀ i j, (i < 2016) → (j < 2016) → M' ⟨i, _⟩ ⟨j, _⟩ = M ⟨i, _⟩ ⟨j, _⟩)
         ∧ det M' = 0

theorem Bob_wins : bob_has_winning_strategy := 
sorry

end Bob_wins_l548_548160


namespace p_necessary_not_sufficient_l548_548223

theorem p_necessary_not_sufficient (k : ℤ) (x : ℝ) :
  (x = (Real.pi / 2) + k * Real.pi) → (sin x = 1) ↔ (∃ m : ℤ, x = (Real.pi / 2) + 2 * m * Real.pi) :=
by
  sorry

end p_necessary_not_sufficient_l548_548223


namespace geometric_sequence_a8_l548_548305

theorem geometric_sequence_a8 {a : ℕ → ℝ} (h1 : a 1 * a 3 = 4) (h9 : a 9 = 256) :
  a 8 = 128 ∨ a 8 = -128 :=
sorry

end geometric_sequence_a8_l548_548305


namespace power_mod_remainder_l548_548929

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l548_548929


namespace tangent_line_at_P_correct_l548_548260

noncomputable def curve : ℝ → ℝ := λ x, (1/3) * x^3

def point_P := (2, 8/3 : ℝ × ℝ)

def tangent_line_equation_correct (x y : ℝ) : Prop := 
  12 * x - 3 * y - 16 = 0

theorem tangent_line_at_P_correct :
  ∃ (m b : ℝ), (∀ x, tangent_line_equation_correct x (curve x) → m = 4 ∧ b = -8/3) :=
sorry

end tangent_line_at_P_correct_l548_548260


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548032

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548032


namespace cistern_wet_surface_area_l548_548798

def cistern (length : ℕ) (width : ℕ) (water_height : ℝ) : ℝ :=
  (length * width : ℝ) + 2 * (water_height * length) + 2 * (water_height * width)

theorem cistern_wet_surface_area :
  cistern 7 5 1.40 = 68.6 :=
by
  sorry

end cistern_wet_surface_area_l548_548798


namespace quadratic_no_real_roots_l548_548679

theorem quadratic_no_real_roots (a b c d : ℝ)  :
  a^2 - 4 * b < 0 → c^2 - 4 * d < 0 → ( (a + c) / 2 )^2 - 4 * ( (b + d) / 2 ) < 0 :=
by
  sorry

end quadratic_no_real_roots_l548_548679


namespace minimum_phi_for_symmetric_shift_l548_548414

theorem minimum_phi_for_symmetric_shift {phi : ℝ} (h1 : phi > 0) 
(h2 : ∀ x, (sin (2 * (x - phi))) = sin (2 * (2 * (x - φ))))
(h3 : ∀ x, (sin (2 * (x - (phi)))) = sin (2 * (x - (2 * (x - (π / 6)))) + C))): 
phi = (5 * π) / 12 := sorry

end minimum_phi_for_symmetric_shift_l548_548414


namespace chess_group_players_l548_548767

theorem chess_group_players (n : ℕ) (H : n * (n - 1) / 2 = 435) : n = 30 :=
by
  sorry

end chess_group_players_l548_548767


namespace annie_passes_bonnie_first_l548_548869

def bonnie_speed (v : ℝ) := v
def annie_speed (v : ℝ) := 1.3 * v
def track_length := 500

theorem annie_passes_bonnie_first (v t : ℝ) (ht : 0.3 * v * t = track_length) : 
  (annie_speed v * t) / track_length = 4 + 1 / 3 :=
by 
  sorry

end annie_passes_bonnie_first_l548_548869


namespace library_students_l548_548495

theorem library_students (total_books : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day3 = 6 →
  let books_used_day1 := students_day1 * books_per_student in
  let books_used_day2 := students_day2 * books_per_student in
  let books_used_day3 := students_day3 * books_per_student in
  let total_books_used := books_used_day1 + books_used_day2 + books_used_day3 in
  let remaining_books := total_books - total_books_used in
  remaining_books / books_per_student = 9 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end library_students_l548_548495


namespace coefficient_x7_in_expansion_l548_548739

noncomputable def a : ℝ := -1/2

theorem coefficient_x7_in_expansion : 
  (∀ (x : ℝ), (x - a) ^ 10) = x ^ 7 :=
begin
  sorry
end

end coefficient_x7_in_expansion_l548_548739


namespace can_form_basis_l548_548241

variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b c : α)

-- Non-coplanar vectors condition
variables (h_non_coplanar : linear_independent ℝ ![a, b, c])

-- Definitions of the vector sets in the options A and C
def optionA_vectors := [a + b, a - 2 • b, c]
def optionC_vectors := [a, 2 • b, b - c]

-- Problem statement: proving which sets of vectors can form a basis for the space
theorem can_form_basis (hA : linear_independent ℝ optionA_vectors)
                       (hC : linear_independent ℝ optionC_vectors) :
    (linear_independent ℝ optionA_vectors) ∨ (linear_independent ℝ optionC_vectors) :=
begin
  -- We assume the given linear independence conditions to derive the final result equivalence
  exact or.inl hA ∨ or.inr hC,
  sorry
end

end can_form_basis_l548_548241


namespace count_valid_c_values_l548_548212

theorem count_valid_c_values : 
  (finset.card (finset.filter (λ c, ∃ x, (5 * (x.floor : ℤ) + 3 * (x.ceil : ℤ) = c)) 
    (finset.range (1001)))) = 251 :=
sorry

end count_valid_c_values_l548_548212


namespace area_triangle_BEC_l548_548667

-- Definitions based on problem conditions
variables (x : ℝ)
variables (A B C D E : Type) [InHabited A] [InHabited B] [InHabited C] [InHabited D] [InHabited E]
variables (AD BE DC EC DE : ℝ)
variables (AB_perpendicular_AD DC_parallel_BE : Prop)

-- Given condition that x = 5
axiom x_value : x = 5

-- Given lengths according to the problem
axiom length_AD : AD = x
axiom length_AB : AB = x
axiom length_DC : DC = 2 * x
axiom midpoint_E : DE = x ∧ EC = x

-- Perpendicular and Parallel relationships
axiom perpendicular_AD_DC : AB_perpendicular_AD
axiom parallel_BE_AD : DC_parallel_BE

/-- 
Problem: Prove that the area of triangle BEC is 12.5 under the given conditions.
-/
theorem area_triangle_BEC : 
  let area_BEC := 1 / 2 * EC * BE 
  in EC = 5 ∧ BE = 5 → area_BEC = 12.5 :=
by
  sorry

end area_triangle_BEC_l548_548667


namespace plywood_perimeter_difference_l548_548084

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548084


namespace part1_part2_part3_l548_548602

def f (a b x : ℝ) := (-3^x + a) / (3^(x + 1) + b)

theorem part1: ∀ x, f 1 1 x ≥ 3^x → x ≤ -1 :=
by
  intros x h
  sorry

theorem part2: (∀ x, f 1 3 (-x) = -f 1 3 x) → f 1 3 = (λ x, (1 - 3^x) / (3 * (3^x + 1))) :=
by
  intros h
  sorry

theorem part3: (∀ x, f a b x) (∀ x1 x2, x1 < x2 → f a b x1 < f a b x2 ∨ f a b x1 = f a b x2 ∨ f a b x1 > f a b x2) → 
  if 3 * a + b > 0 then ( ∀ x1 x2, x1 < x2 →  f a b x1 > f a b x2) 
  else if 3 * a + b = 0 then ¬ (∀ x1 x2, x1 < x2 →  f a b x1 < f a b x2)  
  else ( ∀ x1 x2, x1 < x2 →  f a b x1 < f a b x2) :=
by
  intros hf hm
  sorry

end part1_part2_part3_l548_548602


namespace sum_of_possible_b_values_l548_548941

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l548_548941


namespace collinear_sufficient_not_necessary_l548_548597

-- Define unit vectors in a 2D plane
variables {a b c : EuclideanSpace ℝ (Fin 2)}

-- State conditions that a, b, and c are unit vectors
axiom unit_a : ∥a∥ = 1
axiom unit_b : ∥b∥ = 1
axiom unit_c : ∥c∥ = 1

-- State the proposition to be proved
theorem collinear_sufficient_not_necessary :
  ∥a + b - c∥ = 3 → collinear ℝ ({a, b}) ∧ ¬ (∥a + b - c∥ ≠ 3 → collinear ℝ ({a, b})) :=
sorry

end collinear_sufficient_not_necessary_l548_548597


namespace Alan_has_eight_pine_trees_l548_548159

noncomputable def number_of_pine_trees (total_pine_cones_per_tree : ℕ) (percentage_on_roof : ℚ) 
                                       (weight_per_pine_cone : ℚ) (total_weight_on_roof : ℚ) : ℚ :=
  total_weight_on_roof / (total_pine_cones_per_tree * percentage_on_roof * weight_per_pine_cone)

theorem Alan_has_eight_pine_trees :
  number_of_pine_trees 200 (30 / 100) 4 1920 = 8 :=
by
  sorry

end Alan_has_eight_pine_trees_l548_548159


namespace martin_distance_l548_548364

-- Define the given conditions
def speed : ℝ := 12.0
def time : ℝ := 6.0

-- State the theorem we want to prove
theorem martin_distance : speed * time = 72.0 := by
  sorry

end martin_distance_l548_548364


namespace value_of_expression_l548_548757

noncomputable def halfInverse : ℝ := (1/2)⁻¹
noncomputable def logThreeOne : ℝ := log 3 1

theorem value_of_expression : halfInverse + logThreeOne = 2 :=
by
  have h1 : halfInverse = 2 := by sorry
  have h2 : logThreeOne = 0 := by sorry
  rw [h1, h2]
  exact rfl

end value_of_expression_l548_548757


namespace sum_m_n_zero_l548_548580

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l548_548580


namespace optionC_has_min_4_l548_548000

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548000


namespace sum_of_valid_b_values_l548_548945

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l548_548945


namespace solution_set_l548_548262

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then -2 else if (x ≥ 1) then 1 else 0

def inequality (x : ℝ) : Prop :=
  real.log x / real.log 2 - (real.log (4 * x) / real.log (1/4) - 1) * f ((real.log x / real.log 3) + 1) ≤ 5

theorem solution_set : {x : ℝ | inequality x} = set.Ioo (1/3 : ℝ) 1 ∪ set.Icc 1 4 := sorry

end solution_set_l548_548262


namespace parabola_tangent_distance_l548_548236

-- Definitions for the problem:
def A := (-2 : ℝ, 3 : ℝ)
def C (p : ℝ) (y : ℝ) (x : ℝ) := y^2 = 2 * p * x
def directrix (p : ℝ) (x : ℝ) := x = - p / 2
def parabola (y : ℝ) (x : ℝ) := y^2 = 8 * x
def B := (8 : ℝ, 8 : ℝ)
def F := (2 : ℝ, 0 : ℝ)

-- Statement of the problem:
theorem parabola_tangent_distance :
  ∀ (p : ℝ), (p > 0) → directrix p (-2) → C p 3 (-2) →
  let A := (-2 : ℝ, 3 : ℝ),
      B := (8 : ℝ, 8 : ℝ),
      F := (2 : ℝ, 0 : ℝ) in
  dist B F = 10 := 
by 
  sorry

end parabola_tangent_distance_l548_548236


namespace domain_and_odd_condition_l548_548252

theorem domain_and_odd_condition (a : ℝ) (f : ℝ → ℝ) 
  (h_dom : ∀ x : ℝ, x ∈ (3-2*a, a+1) → f x (x ∈ (3-2*a, a+1)))
  (h_odd : ∀ x : ℝ, f(x+1) = -f(-(x+1))) : 
  a = 2 := 
sorry

end domain_and_odd_condition_l548_548252


namespace range_of_a_l548_548573

variable (x a : ℝ)

-- Definition of α: x > a
def α : Prop := x > a

-- Definition of β: (x - 1) / x > 0
def β : Prop := (x - 1) / x > 0

-- Theorem to prove the range of a
theorem range_of_a (h : α x a → β x) : 1 ≤ a :=
  sorry

end range_of_a_l548_548573


namespace ratio_arithmetic_geometric_mean_l548_548904

/-- Let a and b be positive real numbers. Given the ratio of the arithmetic mean to
  the geometric mean of a and b is 25:24, prove that the ratio of a to b is 16:9 or 9:16. -/
theorem ratio_arithmetic_geometric_mean (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) / (2 * real.sqrt (a * b)) = 25 / 24) : 
  a / b = 16 / 9 ∨ a / b  = 9 / 16 :=
by
  sorry

end ratio_arithmetic_geometric_mean_l548_548904


namespace min_value_expr_l548_548204

theorem min_value_expr (x : ℝ) (hx : x > 0) : 9 * x + x^(-6) ≥ 10 :=
sorry

end min_value_expr_l548_548204


namespace solve_exp_eq_l548_548433

theorem solve_exp_eq (x : ℝ) : 4^x - 4 * 2^x - 5 = 0 ↔ x = Real.log 5 / Real.log 2 :=
by
  sorry

end solve_exp_eq_l548_548433


namespace probability_of_rolling_less_than_four_at_least_six_times_l548_548139

def probability_number_less_than_four : ℚ := 3 / 6

theorem probability_of_rolling_less_than_four_at_least_six_times :
  (binomial 7 6 * (probability_number_less_than_four ^ 6) * ((1 - probability_number_less_than_four) ^ 1)) +
  (binomial 7 7 * (probability_number_less_than_four ^ 7)) = 15 / 128 :=
by sorry

end probability_of_rolling_less_than_four_at_least_six_times_l548_548139


namespace distance_from_origin_to_line_l548_548250

theorem distance_from_origin_to_line {A B : ℝ × ℝ}
  (F : ℝ × ℝ := (-1, 0))
  (C : set (ℝ × ℝ) := {p | p.1^2 / 2 + p.2^2 = 1})
  (line_passes_focus : ∃ l : ℝ × ℝ → Prop, ∀ p ∈ C, l p → p = F)
  (intersect_points : A ∈ C ∧ B ∈ C ∧ A ≠ B)
  (orthogonality : (A.1 * B.1) + (A.2 * B.2) = 0 ) :
  let d := (λ p : ℝ × ℝ, abs(A.1 * B.2 - A.2 * B.1) / sqrt((A.1 - B.1)^2 + (A.2 - B.2)^2)) in
  d (0, 0) = sqrt(6) / 3 :=
sorry

end distance_from_origin_to_line_l548_548250


namespace arithmetic_sequence_b_general_formula_a_l548_548752

noncomputable def a : ℕ → ℕ
| 0     := 0  -- technically a_0 should be defined, even though not in the original sequence
| 1     := 1
| 2     := 2
| (n+3) := 2 * (a (n+2)) - a (n+1) + 2

def b (n : ℕ) : ℕ := a (n + 1) - a n

theorem arithmetic_sequence_b : ∀ n, b (n + 1) = b n + 2 ∧ b 1 = 1 :=
by
  sorry

theorem general_formula_a : ∀ n, a n = n^2 - 2*n + 2 :=
by
  sorry

end arithmetic_sequence_b_general_formula_a_l548_548752


namespace number_of_positive_integer_pairs_l548_548621

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l548_548621


namespace initial_worth_of_wears_l548_548367

theorem initial_worth_of_wears (W : ℝ) 
  (h1 : W + 2/5 * W = 1.4 * W)
  (h2 : 0.85 * (W + 2/5 * W) = W + 95) : 
  W = 500 := 
by 
  sorry

end initial_worth_of_wears_l548_548367


namespace find_a_n_l548_548230

theorem find_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = 3^n + 2) :
  ∀ n, a n = if n = 1 then 5 else 2 * 3^(n - 1) := by
  sorry

end find_a_n_l548_548230


namespace bicycle_race_distance_l548_548038

theorem bicycle_race_distance (
    total_distance : ℕ,
    run_velocity : ℕ,
    total_time : ℕ,
    run_distance : ℕ,
    bicycle_velocity : ℕ
) : total_distance = 155 ∧ run_velocity = 10 ∧ total_time = 6 ∧ run_distance = 10 →
    (total_distance - run_distance = 145) :=
by
    intros
    sorry

end bicycle_race_distance_l548_548038


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548023

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548023


namespace larry_channels_l548_548686

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end larry_channels_l548_548686


namespace dog_catches_sheep_in_20_seconds_l548_548365

variable (v_sheep v_dog : ℕ) (d : ℕ)

def relative_speed (v_dog v_sheep : ℕ) := v_dog - v_sheep

def time_to_catch (d v_sheep v_dog : ℕ) : ℕ := d / (relative_speed v_dog v_sheep)

theorem dog_catches_sheep_in_20_seconds
  (h1 : v_sheep = 16)
  (h2 : v_dog = 28)
  (h3 : d = 240) :
  time_to_catch d v_sheep v_dog = 20 := by {
  sorry
}

end dog_catches_sheep_in_20_seconds_l548_548365


namespace num_pairs_of_positive_integers_eq_77_l548_548619

theorem num_pairs_of_positive_integers_eq_77 : 
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.finite ∧
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.to_finset.card = 2 := 
by 
  sorry

end num_pairs_of_positive_integers_eq_77_l548_548619


namespace next_ren_wu_year_geng_shen_years_20th_century_l548_548399

-- Definitions of Heavenly Stems and Earthly Branches with their cycles
def heavenly_stems := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthly_branches := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Cycle length
def cycle_length := 60

-- Year 2002 has a specific pair of Heavenly Stem and Earthly Branch
def year_2002_pair := ("Ren", "Wu")

-- Proof statement for each question
theorem next_ren_wu_year : (year_2002_pair = ("Ren", "Wu")) → 2002 + cycle_length = 2062 :=
by
  intros h,
  have h1 : cycle_length = 60 := rfl,
  rw h1,
  norm_num,
  sorry  -- Further steps to prove

noncomputable def jiawu_war_year : ℕ :=
by
  sorry -- Further steps to deduce the year 1894

noncomputable def wuxu_reform_year : ℕ :=
by
  sorry -- Further steps to deduce the year 1898

theorem geng_shen_years_20th_century : 
  (year_2002_pair = ("Ren", "Wu")) →
  (1980 ∈ [1920, 1980]) ∧ (1920 ∈ [1920, 1980]) :=
by
  intros h,
  sorry  -- Further steps to prove

end next_ren_wu_year_geng_shen_years_20th_century_l548_548399


namespace nocycleland_has_city_with_57_quasi_neighbors_l548_548809

noncomputable theory

open Finset
open_locale big_operators

def neighbors (G : SimpleGraph ℕ) (v : ℕ) : Finset ℕ :=
  G.Adj.support v

def quasi_neighbors (G : SimpleGraph ℕ) (v : ℕ) : Finset ℕ :=
  (neighbors G v).bUnion (neighbors G)

def has_at_least_57_quasi_neighbors (G : SimpleGraph ℕ) : Prop :=
  ∃ v, (quasi_neighbors G v).card ≥ 57

def nocycleland_graph (G : SimpleGraph ℕ) : Prop :=
  G.is_simple ∧
  (G.edge_finset.card = 2013) ∧
  (G.vertex_finset.card = 500) ∧
  (∀ v, G.Adj.deg v ≤ G.vertex_finset.card - 1) ∧
  ∀ v w x y, v ≠ w → w ≠ x → x ≠ y → y ≠ v → 
    G.Adj v w → G.Adj w x → G.Adj x y → G.Adj y v →
    false

theorem nocycleland_has_city_with_57_quasi_neighbors (G : SimpleGraph ℕ)
  (hG : nocycleland_graph G) :
  has_at_least_57_quasi_neighbors G :=
sorry

end nocycleland_has_city_with_57_quasi_neighbors_l548_548809


namespace triangle_area_l548_548267

theorem triangle_area (l1 l2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ 3 * x - y + 12 = 0)
  (h2 : ∀ x y, l2 x y ↔ 3 * x + 2 * y - 6 = 0) :
  ∃ A : ℝ, A = 9 :=
by
  sorry

end triangle_area_l548_548267


namespace angle_DAB_eq_30_degrees_l548_548057

variables {A B C D : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
 variables (AB AC AD BC : ℝ)
variables (AB_eq_BC : AB = BC) (AC_eq_2CD : ∀ CD : ℝ, AC = 2 * CD) (AD_eq_3CD : ∀ CD : ℝ, AD = 3 * CD)

theorem angle_DAB_eq_30_degrees (h1 : ∀ (B : ℝ), ∠BAD = 90) 
  (h2 : ∃ C ∈ line_segment A D, AC_eq_2CD CD) 
  (h3 : AB_eq_BC) 
  (h4 : AD_eq_3CD CD) : 
  ∠DAB = 30 := 
sorry

end angle_DAB_eq_30_degrees_l548_548057


namespace locus_A_thm_max_OP_OQ_thm_l548_548870

-- Part 1: Locus of point A
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

noncomputable def locus_A (x y : ℝ) (sqrt3 : ℝ) : Prop := 
  (x + sqrt3)^2 + y^2 = 4 ∧ x > 0

theorem locus_A_thm 
  (x y x0 y0 sqrt3 r : ℝ)
  (h0 : ellipse x0 y0)
  (h1 : 0 < r ∧ r < 1)
  (h2 : (x - x0)^2 + (y - y0)^2 = r^2)
  (dist_cond : |dist (x y) (sqrt3 0)| - |dist (other_conditions)| = 2 * r)
  : locus_A x y sqrt3 := 
  sorry

-- Part 2: Maximum value of |OP||OQ|
noncomputable def max_OP_OQ (k1 k2 x0 y0 r : ℝ) : Prop :=
  ∃ max_val : ℝ, max_val = |OP| * |OQ| → max_val = 5 / 2

theorem max_OP_OQ_thm
  (x0 y0 k1 k2 r : ℝ)
  (h0 : ellipse x0 y0)
  (h1 : 0 < r ∧ r < 1)
  (slope_cond : k1 * k2 = -1/4) 
  : max_OP_OQ k1 k2 x0 y0 r := 
  sorry

end locus_A_thm_max_OP_OQ_thm_l548_548870


namespace perimeters_positive_difference_l548_548094

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548094


namespace andy_demerits_l548_548511

theorem andy_demerits :
  let demerits_late := 6 * 2 in
  let demerits_joke := 15 in
  let demerits_received := demerits_late + demerits_joke in
  let demerits_remaining := 23 in
  demerits_received + demerits_remaining = 50 :=
by
  let demerits_late := 6 * 2
  let demerits_joke := 15
  let demerits_received := demerits_late + demerits_joke
  let demerits_remaining := 23
  exact (by sorry : demerits_received + demerits_remaining = 50)

end andy_demerits_l548_548511


namespace option_c_has_minimum_value_4_l548_548016

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548016


namespace people_taking_C_count_l548_548778

def systematic_sampling (total_people : ℕ) (selected_people : ℕ) : List ℕ :=
  List.range' 8 total_people 
  |> List.filter (fun x => (x - 8) % 20 = 0)
  |> List.take selected_people

def group_assignment (n : ℕ) : String :=
  if 1 ≤ n ∧ n ≤ 400 then "A"
  else if 401 ≤ n ∧ n ≤ 750 then "B"
  else "C"

theorem people_taking_C_count :
  let sampling_result := systematic_sampling 1000 50 
  List.countp (λ x => group_assignment x = "C") sampling_result = 12 := 
by
  sorry

end people_taking_C_count_l548_548778


namespace quadratic_negative_root_exists_l548_548839

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c

theorem quadratic_negative_root_exists 
  {a b c : ℝ} 
  (h_distinct_roots : ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_polynomial a b c x₁ = 0 ∧ quadratic_polynomial a b c x₂ = 0)
  (h_inequality : ∀ x y : ℝ, quadratic_polynomial a b c (x^2 + y^2) ≥ quadratic_polynomial a b c (2 * x * y)) :
  ∃ x, quadratic_polynomial a b c x = 0 ∧ x < 0 :=
sorry

end quadratic_negative_root_exists_l548_548839


namespace system_has_solution_l548_548422

noncomputable def system (a b x : ℝ) : Prop :=
  cos x = a * x + b ∧ sin x + a = 0

theorem system_has_solution (a b : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ cos x1 = a * x1 + b ∧ cos x2 = a * x2 + b) :
  ∃ x, system a b x :=
by
  sorry

end system_has_solution_l548_548422


namespace perimeter_difference_l548_548068

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548068


namespace focus_of_hyperbola_l548_548527

def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 49 - (y - 10)^2 / 9 = 1

def focus_with_larger_x : (ℝ × ℝ) :=
  (5 + Real.sqrt 58, 10)

theorem focus_of_hyperbola : ∀ x y, hyperbola x y → 
  focus_with_larger_x = (5 + Real.sqrt 58, 10) :=
by sorry

end focus_of_hyperbola_l548_548527


namespace plywood_cut_difference_l548_548121

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548121


namespace factorize_a_cubed_minus_25a_l548_548918

variable {a : ℝ}

theorem factorize_a_cubed_minus_25a (a : ℝ) : a^3 - 25 * a = a * (a + 5) * (a - 5) := 
by sorry

end factorize_a_cubed_minus_25a_l548_548918


namespace profit_maximization_l548_548136

theorem profit_maximization (a : ℝ) (h_a : 0 < a) : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ a → 
  (x = 2 → a ≥ 2 → y(x) = y(2)) ∧ 
  (x = a → a < 2 → y(x) = y(a))
  where 
  y(x) : ℝ := 19 - 24 / (x + 2) - 3 / 2 * x :=
begin
  sorry
end

end profit_maximization_l548_548136


namespace discount_percentage_l548_548845

-- Define the conditions
def retailer_pens := 140
def wholesaler_pens_price := 36
def market_price_per_pen := 1
def profit_percentage := 2.85

-- Define the proof goal
theorem discount_percentage (CP_total SP_total TMP: ℝ) 
  (h1 : CP_total = wholesaler_pens_price * market_price_per_pen)
  (h2: TMP = retailer_pens * market_price_per_pen)
  (h3: profit_percentage * CP_total = SP_total - CP_total) :
  (TMP - SP_total) / TMP * 100 = 1 := 
sorry

end discount_percentage_l548_548845


namespace interest_rate_of_A_to_B_l548_548482

theorem interest_rate_of_A_to_B :
  ∀ (principal gain interest_B_to_C : ℝ), 
  principal = 3500 →
  gain = 525 →
  interest_B_to_C = 0.15 →
  (principal * interest_B_to_C * 3 - gain) = principal * (10 / 100) * 3 :=
by
  intros principal gain interest_B_to_C h_principal h_gain h_interest_B_to_C
  sorry

end interest_rate_of_A_to_B_l548_548482


namespace trig_identity_proof_l548_548226

variable (θ a : ℝ)

-- Assume the given condition
axiom cos_pi6_sub_theta : cos (π / 6 - θ) = a

-- Prove the required statement
theorem trig_identity_proof
  (h : cos (π / 6 - θ) = a) : 
  cos (5 * π / 6 + θ) + sin (2 * π / 3 - θ) = 0 := 
by
  -- Proof will be inserted here
  sorry

end trig_identity_proof_l548_548226


namespace problem_sqrt_m_problem_sqrt_abc_l548_548997

theorem problem_sqrt_m (n : ℚ) (m : ℚ) (h1 : m ≥ 0) (h2 : sqrt m = 2*n + 1) (h3 : sqrt m = 4 - 3*n) :
  m = 121 ∨ m = 121 / 25 :=
by sorry

theorem problem_sqrt_abc (a b c n: ℚ) (h1 : |a - 1| + sqrt b + (c - n)^2 = 0) (h2 : n = 5) :
  sqrt (a + b + c) = sqrt 6 ∨ sqrt (a + b + c) = -sqrt 6 :=
by sorry

end problem_sqrt_m_problem_sqrt_abc_l548_548997


namespace finding_breadth_and_length_of_floor_l548_548463

noncomputable def length_of_floor (b : ℝ) := 3 * b
noncomputable def area_of_floor (b : ℝ) := (length_of_floor b) * b

theorem finding_breadth_and_length_of_floor
  (breadth : ℝ)
  (length : ℝ := length_of_floor breadth)
  (area : ℝ := area_of_floor breadth)
  (painting_cost : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : painting_cost = 100)
  (h2 : cost_per_sqm = 2)
  (h3 : area = painting_cost / cost_per_sqm) :
  length = Real.sqrt 150 :=
by
  sorry

end finding_breadth_and_length_of_floor_l548_548463


namespace number_142857_has_property_l548_548457

noncomputable def has_desired_property (n : ℕ) : Prop :=
∀ m ∈ [1, 2, 3, 4, 5, 6], ∀ d ∈ (Nat.digits 10 (n * m)), d ∈ (Nat.digits 10 n)

theorem number_142857_has_property : has_desired_property 142857 :=
sorry

end number_142857_has_property_l548_548457


namespace count_non_empty_subsets_S_with_sum_multiple_of_3_l548_548699

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 16}

-- Define condition for a subset A's sum to be a multiple of 3
def is_multiple_of_3 (A : Set ℕ) : Prop := (∑ x in A, x) % 3 = 0

-- Prove the total number of non-empty subsets A of S such that the sum of all elements in A is a multiple of 3 is 21855
theorem count_non_empty_subsets_S_with_sum_multiple_of_3 :
  {A : Set ℕ // A ⊆ S ∧ is_multiple_of_3 A}.card = 21855 :=
begin
  sorry
end

end count_non_empty_subsets_S_with_sum_multiple_of_3_l548_548699


namespace mode_seventh_grade_average_eighth_grade_performance_analysis_l548_548133

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

-- Mode of the seventh-grade scores
theorem mode_seventh_grade : List.mode seventh_grade_scores = 6 := sorry

-- Average of the eighth-grade scores
noncomputable def average (l : List ℕ) : ℝ := (l.sum : ℝ) / (l.length : ℝ)
theorem average_eighth_grade : average eighth_grade_scores = 7.1 := sorry

-- Performance analysis of Xiao Li and Xiao Zhang based on their team's medians
def median (l : List ℕ) : ℝ :=
if l.length % 2 = 0 then
  let sorted := l.qsort (· ≤ ·)
  ((sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) : ℝ) / 2
else
  (l.qsort (· ≤ ·)).get! (l.length / 2)

theorem performance_analysis :
  median seventh_grade_scores = 6 ∧
  median eighth_grade_scores = 7.5 ∧
  7 > median seventh_grade_scores ∧
  7 < median eighth_grade_scores := 
sorry

end mode_seventh_grade_average_eighth_grade_performance_analysis_l548_548133


namespace parabola_properties_l548_548254

noncomputable def parabola_condition (p : ℝ) (h : p > 0) : Prop :=
  let C := λ x y : ℝ, y^2 = 2 * p * x
  let F := (p / 2, 0)
  let M := (-3 / 2, 0)
  let l := λ m : ℝ, λ x y : ℝ, x = m * y + p / 2
  let isAB := λ A B : ℝ × ℝ, A.1 = B.1 ∧ C A.1 A.2 ∧ C B.1 B.2
  let P := λ x : ℝ, (x, 0)
  let Q := λ A B : ℝ × ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let perpendicular := λ A B P Q : ℝ × ℝ, A.2 - B.2 = - (P.1 - Q.1)
  ∃ A B : ℝ × ℝ, 
  isAB A B ∧ perpendicular A B M (Q A B) ∧
  (¬(C 3 x)) ∧
  (∀ x, |A.1 - B.1| + 3 * |F.1 - B.1| ≥ 27 / 2) ∧
  (|F.1 - B.1| * (|M.1 - A.1| + |M.1 - B.1|) = 2 * |M.1 - B.1| * |P F.1 - F.1|)

-- Now we state the theorem with the conditions and the equality we need to prove
theorem parabola_properties : ∀ p : ℝ, p > 0 → parabola_condition p sorry

end parabola_properties_l548_548254


namespace area_of_rectangle_l548_548843

noncomputable def rectangle_area : ℚ :=
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  length * width

theorem area_of_rectangle : rectangle_area = 392 / 9 :=
  by 
  let side1 : ℚ := 73 / 10
  let side2 : ℚ := 94 / 10
  let side3 : ℚ := 113 / 10
  let perimeter_triangle : ℚ := side1 + side2 + side3
  let width : ℚ := perimeter_triangle / 6
  let length : ℚ := 2 * width
  have : length * width = 392 / 9 := sorry
  exact this

end area_of_rectangle_l548_548843


namespace number_of_pairs_l548_548625

theorem number_of_pairs :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end number_of_pairs_l548_548625


namespace plywood_perimeter_difference_l548_548080

noncomputable theory

open classical

theorem plywood_perimeter_difference :
  ∃ (rect1 rect2 : ℕ) (a b : ℕ),
  (rect1 = 6 ∧ rect2 = 9 ∧ rect1 % 6 = 0 ∧ rect2 % 6 = 0) ∧ 
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≤ 20) ∧
  (∀ l w : ℕ, l * w = rect1 * rect2 / 6 → 2 * (l + w) ≥ 10) ∧ 
  (20 - 10 = 10) :=
by
  exists (6, 9, 6, 9)
  sorry

end plywood_perimeter_difference_l548_548080


namespace find_angle_A_l548_548862

-- Definitions
variables {A B C I M N X Y : Type} [Incenter I A B C]
variables {midpoint_M : M = midpoint A C}
variables {midpoint_N : N = midpoint A B}
variables {NI_meets_AC_at_X : ∃ (X : Type), line_through N I ∧ line_through A C}
variables {MI_meets_AB_at_Y : ∃ (Y : Type), line_through M I ∧ line_through A B}
variables {area_AXY_equals_area_ABC : area (triangle A X Y) = area (triangle A B C)}

-- Given all the above conditions, we need to prove that:
theorem find_angle_A (h : ∃ A B C I M N X Y, 
  (Incenter I A B C) ∧ 
  (M = midpoint A C) ∧
  (N = midpoint A B) ∧
  (∃ (X : Type), line_through N I ∧ line_through A C) ∧
  (∃ (Y : Type), line_through M I ∧ line_through A B) ∧
  (area (triangle A X Y) = area (triangle A B C))
  ) : ∠A = 60 := 
begin
  sorry,
end

end find_angle_A_l548_548862


namespace alicia_taxes_l548_548164

theorem alicia_taxes:
  let w := 20 -- Alicia earns 20 dollars per hour
  let r := 1.45 / 100 -- The local tax rate is 1.45%
  let wage_in_cents := w * 100 -- Convert dollars to cents
  let tax_deduction := wage_in_cents * r -- Calculate tax deduction in cents
  tax_deduction = 29 := 
by 
  sorry

end alicia_taxes_l548_548164


namespace simplify_cube_root_l548_548787

theorem simplify_cube_root {c d : ℕ}
  (h1 : c > 0)
  (h2 : d > 0)
  (h3 : ∀ k : ℕ, (∃ m, n, prime m ∧ prime n ∧ k = c * d * m ∧ d = n) → False)
  (h4 : (7200 : ℝ) = ↑(c * d)) : c + d = 452 :=
by
  sorry

end simplify_cube_root_l548_548787


namespace option_c_has_minimum_value_4_l548_548017

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548017


namespace pyramid_volume_l548_548417

theorem pyramid_volume (a : ℝ) :
  let S_AKC := (1 / 2) * (a / Real.sqrt 2) * (a / Real.sqrt 2) in
  let V_FABC := (1 / 3) * S_AKC * a in
  V_FABC = a^3 / 12 :=
by
  sorry

end pyramid_volume_l548_548417


namespace solve_z_6_eq_neg64_l548_548935

noncomputable def complex_solutions (k : ℕ) (hk : k < 6) :=
  2 * complex.exp (complex.I * (π / 6 + 2 * k * π / 6))

theorem solve_z_6_eq_neg64 : 
  {z : ℂ | z^6 = -64} = 
  {complex_solutions k (by norm_num : k < 6) | k : ℕ ∧ k < 6} :=
sorry

end solve_z_6_eq_neg64_l548_548935


namespace rachel_plant_arrangement_count_l548_548721

theorem rachel_plant_arrangement_count :
  let cactus_plants := 3,
      orchid_plants := 1,
      yellow_lamps := 3,
      blue_lamps := 2 in
  ∃ ways_to_arrange, ways_to_arrange = 13 := sorry

end rachel_plant_arrangement_count_l548_548721


namespace optionC_has_min_4_l548_548007

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548007


namespace find_k_in_quadratic_l548_548296

theorem find_k_in_quadratic (k : ℝ) : 
  let a := 5
  let b := 4
  let roots := λ k : ℝ, [(-b + (real.sqrt (b^2 - 4 * a * k)) / (2 * a)), (-b - (real.sqrt (b^2 - 4 * a * k)) / (2 * a))]
  (roots k) = [(-4 + i * real.sqrt 379) / 10, (-4 - i * real.sqrt 379) / 10] →
  k = 19.75 :=
by
  sorry

end find_k_in_quadratic_l548_548296


namespace no_such_line_exists_l548_548832

noncomputable theory
open_locale classical

def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def sum_distances_to_line_x_minus_2 (A B : ℝ × ℝ) : ℝ :=
  let (x1, _) := A
  let (x2, _) := B
  x1 + x2 + 4

theorem no_such_line_exists :
  ∀ A B : ℝ × ℝ, (parabola A.1 A.2) → (parabola B.1 B.2) →
  sum_distances_to_line_x_minus_2 A B = 5 →
  ∃ m : ℝ, (∀ y : ℝ, ((m * y + 1) = A.1 ∨ (m * y + 1) = B.1)) →
  False := 
by
  intro A B hA hB h_sum hm
  sorry

end no_such_line_exists_l548_548832


namespace quadratic_has_distinct_real_roots_l548_548381

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) :
  discriminant 1 (-(2*k - 1)) (k^2 - k) > 0 :=
by
  -- Definitions of coefficients
  let a : ℝ := 1
  let b : ℝ := -(2 * k - 1)
  let c : ℝ := k^2 - k

  -- Calculate discriminant
  have disc : discriminant a b c = 1 := sorry

  -- Show the discriminant is greater than 0
  show discriminant a b c > 0 from
    by rwa disc

end quadratic_has_distinct_real_roots_l548_548381


namespace problem_l548_548312

theorem problem (W X Y Z R S P Q : Type) 
  [is_rectangle WXYZ] 
  (hWZ : WZ = 8) 
  (hZX : ZX = 4)
  (hZP : ZP = ZX / 3) 
  (hPQ : PQ = ZX / 3)
  (hQX : QX = ZX / 3)
  (hP_on_ZX : P ∈ segment ZX)
  (hQ_on_ZX : Q ∈ segment ZX)
  (hintersection1 : intersect_line_at WP YZ R)
  (hintersection2 : intersect_line_at WQ YZ S)
  : ZR:RS:SY = 6:1:2 :=
begin 
  sorry 
end

end problem_l548_548312


namespace bees_seen_on_second_day_l548_548366

theorem bees_seen_on_second_day (b1 : ℕ) (h1 : b1 = 144) (h2 : ∀ b2 : ℕ, b2 = 3 * b1) : ∃ b2 : ℕ, b2 = 432 :=
by
  have b1_eq_144 : b1 = 144 := h1
  have b2_eq : ∀ b2 : ℕ, b2 = 3 * b1 := h2
  use 432
  rw [b2_eq, b1_eq_144]
  norm_num
  sorry

end bees_seen_on_second_day_l548_548366


namespace min_fence_posts_l548_548489

theorem min_fence_posts (length width wall_length interval : ℕ) (h_dim : length = 80) (w_dim : width = 50) (h_wall : wall_length = 150) (h_interval : interval = 10) : 
  length/interval + 1 + 2 * (width/interval - 1) = 17 :=
by
  sorry

end min_fence_posts_l548_548489


namespace fraction_eq_zero_has_solution_l548_548297

theorem fraction_eq_zero_has_solution :
  ∀ (x : ℝ), x^2 - x - 2 = 0 ∧ x + 1 ≠ 0 → x = 2 :=
by
  sorry

end fraction_eq_zero_has_solution_l548_548297


namespace number_of_girls_joined_l548_548442

-- Define the initial conditions
def initial_girls := 18
def initial_boys := 15
def boys_quit := 4
def total_children_after_changes := 36

-- Define the changes
def boys_after_quit := initial_boys - boys_quit
def girls_after_changes := total_children_after_changes - boys_after_quit
def girls_joined := girls_after_changes - initial_girls

-- State the theorem
theorem number_of_girls_joined :
  girls_joined = 7 :=
by
  sorry

end number_of_girls_joined_l548_548442


namespace find_slope_of_line_bisecting_circle_l548_548316

-- Definitions of conditions go here
def point (x y : ℝ) := (x, y)
def line_through_origin (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) (k : ℝ) : Prop := 
  ∀ (Q : ℝ × ℝ), l Q ↔ Q.2 = k * Q.1 + P.2

-- Circle definition: center at (2, 0) with radius 1, i.e., (x - 2)^2 + y^2 = 1
def circle (C : ℝ × ℝ → Prop) : Prop := 
  ∀ (p : ℝ × ℝ), C p ↔ (p.1 - 2)^2 + p.2^2 = 1

-- The line l passing through point P(0, 1)
def line_passing_through_P (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ → Prop := 
  λ Q, Q.2 = k * Q.1 + P.2

-- The line l bisects the area of the circle C
def line_bisects_area (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) : Prop := 
  ∀ (A B : ℝ × ℝ), C A ∧ l A ∧ C B ∧ ¬ l B → (∃ M N : ℝ × ℝ, l M ∧ ¬ C M ∧ ¬ l N ∧ C N)

-- Given conditions as Lean definitions
def P : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ → Prop := λ p, (p.1 - 2)^2 + p.2^2 = 1
def l (k : ℝ) : ℝ × ℝ → Prop := λ Q, Q.2 = k * Q.1 + P.2

-- Proof problem statement: Prove slope k = -1/2 given the conditions
theorem find_slope_of_line_bisecting_circle : ∃ k : ℝ, 
  (∀ Q : ℝ × ℝ, l k Q ↔ Q.2 = k * Q.1 + P.2) ∧
  (line_bisects_area (l k) C) → k = -1/2 :=
begin
  sorry
end

end find_slope_of_line_bisecting_circle_l548_548316


namespace John_spent_15_dollars_l548_548682

def minutes_in_hour : Nat := 60
def total_hours : Nat := 3
def total_minutes : Nat := total_hours * minutes_in_hour
def interval_minutes : Nat := 6
def cost_per_interval : Float := 0.50
def total_intervals : Nat := total_minutes / interval_minutes
def total_amount_spent : Float := total_intervals * cost_per_interval

theorem John_spent_15_dollars : total_amount_spent = 15 := by
  sorry

end John_spent_15_dollars_l548_548682


namespace num_pairs_of_positive_integers_eq_77_l548_548617

theorem num_pairs_of_positive_integers_eq_77 : 
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.finite ∧
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.to_finset.card = 2 := 
by 
  sorry

end num_pairs_of_positive_integers_eq_77_l548_548617


namespace trapezium_other_side_length_l548_548200

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l548_548200


namespace part1_part2_l548_548234

variables {a : ℕ → ℕ}
variables {b : ℕ → ℝ}
variables {c : ℕ → ℝ}

-- Condition definitions
definition S : ℕ → ℝ := λ n, (n * (n + 1)) / 2
definition S_cond := ∀ n, S (n + 1) / S n = (n + 2) / n
def a_cond := a 1 = 1 ∧ ∀ n, a (n + 1) = n + 1
def b_cond := b 2 = 2 ∧ b 1 * b 2 * b 3 * b 4 * b 5 = 2 ^ 10
def c_def := ∀ n, c n = (2 + a n) / (a n * a (n + 1) * b (n + 1))

-- The first part: finding the general formula for {a_n}
theorem part1 (a_cond : a_cond) : ∀ n, a n = n := by
  sorry

-- The second part: proving c_1 + c_2 + ... + c_n < 1
theorem part2 (a_cond : a_cond) (b_cond : b_cond) (c_def : c_def) : ∀ n, (∑ i in finset.range (n + 1), c i) < 1 := by
  sorry

end part1_part2_l548_548234


namespace sin_eq_cos_iff_cos2alpha_eq_zero_l548_548794

theorem sin_eq_cos_iff_cos2alpha_eq_zero (α : ℝ) :
  (sin α = cos α) → (cos (2 * α) = 0) ∧ ¬ ((cos (2 * α) = 0) → (sin α = cos α)) := 
by
  sorry

end sin_eq_cos_iff_cos2alpha_eq_zero_l548_548794


namespace problem_a_problem_b_l548_548807

-- Quadratic equation root calculations
noncomputable def quad_eq_roots (a b c : ℝ) : ℝ × ℝ :=
  let delta := b * b - 4 * a * c in
  ((-b + Real.sqrt delta) / (2 * a), (-b - Real.sqrt delta) / (2 * a))

-- Smallest root of the quadratic equation x^2 - 9x - 10 = 0
noncomputable def smallest_root_a : ℝ :=
  let (x1, x2) := quad_eq_roots 1 (-9) (-10) in min x1 x2

-- Smallest root of the quadratic equation x^2 - 9x + 10 = 0
noncomputable def smallest_root_b : ℝ :=
  let (x1, x2) := quad_eq_roots 1 (-9) 10 in min x1 x2

-- Theorem for Problem (a)
theorem problem_a : smallest_root_a ^ 4 - 909 * smallest_root_a = 910 :=
by sorry

-- Theorem for Problem (b)
theorem problem_b : smallest_root_b ^ 4 - 549 * smallest_root_b = -710 :=
by sorry

end problem_a_problem_b_l548_548807


namespace real_iff_m_eq_pm1_imag_iff_m_ne_pm1_pure_imag_iff_m_eq_neg2_l548_548214

section complex_numbers

variables {m : ℝ}

def z (m : ℝ) := complex.mk (m^2 + m - 2) (m^2 - 1)

-- 1. Prove \( z \) is a real number implies \( m = \pm 1 \)
theorem real_iff_m_eq_pm1 : (∃ (m : ℝ), (complex.im (z m) = 0) ∧ complex.re (z m) = z m) ↔ (m = 1 ∨ m = -1) := sorry

-- 2. Prove \( z \) is an imaginary number implies \( m \neq \pm 1 \)
theorem imag_iff_m_ne_pm1 : (∃ (m : ℝ), (complex.re (z m) = 0) ∧ (z m ≠ 0)) ↔ (m ≠ 1 ∧ m ≠ -1) := sorry

-- 3. Prove \( z \) is a pure imaginary number implies \( m = -2 \)
theorem pure_imag_iff_m_eq_neg2 : (∃ (m : ℝ), (complex.re (z m) = 0) ∧ (complex.im (z m) ≠ 0)) ↔ (m = -2) := sorry

end complex_numbers

end real_iff_m_eq_pm1_imag_iff_m_ne_pm1_pure_imag_iff_m_eq_neg2_l548_548214


namespace root_expression_l548_548343

noncomputable def alpha_beta_roots (p c : ℝ) : set ℝ :=
  {x | x^2 + p * x + c = 0}

noncomputable def gamma_delta_roots (q c : ℝ) : set ℝ :=
  {x | x^2 + q * x + c = 0}

theorem root_expression (α β γ δ p q c : ℝ)
  (h1 : α ∈ alpha_beta_roots p c)
  (h2 : β ∈ alpha_beta_roots p c)
  (h3 : γ ∈ gamma_delta_roots q c)
  (h4 : δ ∈ gamma_delta_roots q c) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = (p^2 - q^2) * c + c^2 - p * c - q * c :=
sorry

end root_expression_l548_548343


namespace flashes_green_in_671_seconds_l548_548483

def light_flashes_green (total_seconds : Nat) (green_interval : Nat) : Nat :=
  total_seconds / green_interval

def light_flashes_both_green_red (total_seconds : Nat) (green_interval : Nat) (red_interval : Nat) : Nat :=
  total_seconds / Nat.lcm green_interval red_interval

def light_flashes_both_green_blue (total_seconds : Nat) (green_interval : Nat) (blue_interval : Nat) : Nat :=
  total_seconds / Nat.lcm green_interval blue_interval

def light_flashes_all_colors (total_seconds : Nat) (green_interval : Nat) (red_interval : Nat) (blue_interval : Nat) : Nat :=
  total_seconds / Nat.lcm3 green_interval red_interval blue_interval

theorem flashes_green_in_671_seconds : light_flashes_green 671 3 
  - light_flashes_both_green_red 671 3 5 
  - light_flashes_both_green_blue 671 3 7 
  + light_flashes_all_colors 671 3 5 7 = 154 := by
  sorry

end flashes_green_in_671_seconds_l548_548483


namespace billy_piles_l548_548874

theorem billy_piles (Q D : ℕ) (h : 2 * Q + 3 * D = 20) :
  Q = 4 ∧ D = 4 :=
sorry

end billy_piles_l548_548874


namespace sum_possible_b_quad_eq_rational_roots_l548_548961

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l548_548961


namespace minimum_value_inequality_l548_548346

open Real

theorem minimum_value_inequality
  (a b c : ℝ)
  (ha : 2 ≤ a) 
  (hb : a ≤ b)
  (hc : b ≤ c)
  (hd : c ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 = 4 * (sqrt 5 ^ (1 / 4) - 1)^2 :=
sorry

end minimum_value_inequality_l548_548346


namespace limit_proof_l548_548595

variable (f : ℝ → ℝ)

theorem limit_proof (h_differentiable : DifferentiableAt ℝ f (-1))
(h_derivative : deriv f (-1) = -3) :
lim (λ (Δx : ℝ), (f (-1) - f (-1 + Δx))/(3 * Δx)) (𝓝 0) = 1 := 
sorry

end limit_proof_l548_548595


namespace track_length_l548_548173

theorem track_length (L : ℕ)
  (h1 : ∃ B S : ℕ, B = 120 ∧ (L - B) = S ∧ (S + 200) - B = (L + 80) - B)
  (h2 : L + 80 = 440 - L) : L = 180 := 
  by
    sorry

end track_length_l548_548173


namespace curve_is_parabola_l548_548816

noncomputable def parabola_focus_axis (C : Type) [curve C] (F : point) :=
  (∀ (light : ray) (bundle : light_bundle light),
    bundle ∶ parallel rays reflecting off curve C converging at point F) →
  ∃ (p : parabola), focus p = F ∧ (axis p).parallel (direction light)

def is_parabola_with_focus_and_axis (C : Type) [curve C] (F : point) (light_direction : direction) : Prop :=
  parabola_focus_axis C F

theorem curve_is_parabola (C : Type) [curve C] (F : point) (light_direction : direction) :
  (∀ (light : ray) (bundle : light_bundle light),
    parallel_rays light ∧ reflects_off_curve light C ∧ converges_at light bundle F) →
  is_parabola_with_focus_and_axis C F light_direction :=
sorry

end curve_is_parabola_l548_548816


namespace find_f_l548_548899

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def g (x : ℝ) : ℝ := if x < 0 then -x^2 + x else 0

theorem find_f (f : ℝ → ℝ)
  (h₁ : odd_function f)
  (h₂ : ∀ x, x < 0 → f x = -x^2 + x) :
  f 2 = 6 :=
by
  sorry

end find_f_l548_548899


namespace probability_two_odds_l548_548972

open Finset

-- Define the set A = {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the set O = {1, 3, 5} (the set of odd numbers in A)
def O : Finset ℕ := {1, 3, 5}

-- Define total number of outcomes when choosing 2 from A without replacement
def total_outcomes : ℕ := choose 5 2

-- Define the number of outcomes where both chosen numbers are odd
def odd_outcomes : ℕ := choose 3 2

-- Calculate the probability p of choosing two odd numbers from A without replacement
def probability_odd : ℚ := odd_outcomes / total_outcomes

theorem probability_two_odds :
  probability_odd = 3 / 10 :=
by
  sorry

end probability_two_odds_l548_548972


namespace points_not_on_C_do_not_satisfy_F_l548_548994

-- Given that all points satisfying F(x, y) = 0 are on curve C,
-- prove that points not on curve C do not satisfy F(x, y) = 0.

variable [Curve C] (F : ℝ × ℝ → ℝ)

axiom points_on_C_satisfy_F (x y : ℝ) : F (x, y) = 0 → (x, y) ∈ C

theorem points_not_on_C_do_not_satisfy_F (x y : ℝ) : (x, y) ∉ C → F (x, y) ≠ 0 :=
by
  intro h
  intro contra
  have : (x, y) ∈ C := points_on_C_satisfy_F x y contra
  contradiction

end points_not_on_C_do_not_satisfy_F_l548_548994


namespace domain_f_2x_minus_3_l548_548705

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1

def domain_f : set ℝ := {x | 1 ≤ f x ∧ f x ≤ 5}

theorem domain_f_2x_minus_3 :
  domain_f = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by sorry

end domain_f_2x_minus_3_l548_548705


namespace tree_leaves_not_shed_l548_548156

-- Definitions of conditions based on the problem.
variable (initial_leaves : ℕ) (shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves : ℕ)

-- Setting the conditions
def conditions :=
  initial_leaves = 5000 ∧
  shed_week1 = initial_leaves / 5 ∧
  shed_week2 = 30 * (initial_leaves - shed_week1) / 100 ∧
  shed_week3 = 60 * shed_week2 / 100 ∧
  shed_week4 = 50 * (initial_leaves - shed_week1 - shed_week2 - shed_week3) / 100 ∧
  shed_week5 = 2 * shed_week3 / 3 ∧
  remaining_leaves = initial_leaves - shed_week1 - shed_week2 - shed_week3 - shed_week4 - shed_week5

-- The proof statement
theorem tree_leaves_not_shed (h : conditions initial_leaves shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves) :
  remaining_leaves = 560 :=
sorry

end tree_leaves_not_shed_l548_548156


namespace laundry_per_hour_l548_548884

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l548_548884


namespace find_the_value_of_m_l548_548590

noncomputable def find_m (x : ℝ) (m : ℝ) : Prop :=
  log10 (tan x) + log10 (cot x) = 0 ∧ 
  log10 (tan x + cot x) = (1 / 2) * (log10 m - 1) → 
  m = 20

theorem find_the_value_of_m (x : ℝ) (m : ℝ) : find_m x m :=
  sorry

end find_the_value_of_m_l548_548590


namespace new_person_weight_l548_548659

theorem new_person_weight 
  (W : ℝ)
  (original_avg : W / 20)
  (new_avg : W / 20 + 4.5)
  (replacement_weight : 92 : ℝ)
  (number_of_people : 20 = 20) :
  let N := 20 * 4.5 + replacement_weight in
  N = 182 := 
begin
  sorry
end

end new_person_weight_l548_548659


namespace num_pairs_satisfying_equation_l548_548630

theorem num_pairs_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end num_pairs_satisfying_equation_l548_548630


namespace cosine_angle_BP_CD_l548_548675

-- Definition of points and trapezoid
variables {A B C D P : Type}

-- Conditions
variables 
  (BC_parallel_AD : ∀ (A B C D : Type), BC ∥ AD) -- BC parallel to AD
  (AB_eq_AD : ∀ (A B D : Type), AB = AD) -- AB = AD
  (angle_ABC : ∀ (A B C : Type), ∠ABC = 2 * Real.pi / 3) -- ∠ABC = 2π/3
  (angle_BCD : ∀ (B C D : Type), ∠BCD = Real.pi / 2) -- ∠BCD = π/2
  (cos_angle_AB_CD : ∀ (A B C D : Type), cos (angle_between AB CD) = Real.sqrt 3 / 6) -- cosine of the angle between AB and CD is √3/6

-- Question: Prove that the cosine of the angle between BP and CD is 1/2
theorem cosine_angle_BP_CD (A B C D P : Type)
  [BC_parallel_AD A B C D]
  [AB_eq_AD A B D]
  [angle_ABC A B C]
  [angle_BCD B C D]
  [cos_angle_AB_CD A B C D] :
  cos (angle_between BP CD) = 1/2 :=
sorry

end cosine_angle_BP_CD_l548_548675


namespace number_of_positive_integer_pairs_l548_548620

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l548_548620


namespace inequality_proof_l548_548719

theorem inequality_proof (m n p q : ℝ) (h_m : 0 < m) (h_n : 0 < n) (h_p : 0 < p) (h_q : 0 < q) : 
    let t := (m + n + p + q) / 2 in
    (m / (t + n + p + q) + n / (t + m + p + q) + p / (t + m + n + q) + q / (t + m + n + p)) ≥ 4 / 5 :=
by
    let t := (m + n + p + q) / 2
    sorry

end inequality_proof_l548_548719


namespace triangle_side_lengths_l548_548716

noncomputable def side_lengths : ℝ × ℝ := sorry

theorem triangle_side_lengths :
  ∃ (a b : ℝ),
    (∀ (α β : ℝ), sin α / sin β = 1 / 2) ∧
    (1 / 4 = 1 / 2 * a * b * sin (π / 2 - α - β)) ∧
    (1 = a^2 + b^2 - 2 * a * b * cos (π / 2 - α - β)) ∧
    (side_lengths = (0.5115, 1.0230)) :=
sorry

end triangle_side_lengths_l548_548716


namespace sequence_a_correct_l548_548979

def S (n : ℕ) : ℤ :=
  n^2 + 3 * n + 5

def a (n : ℕ) : ℤ :=
  if n = 1 then 9 else 2 * n + 2

theorem sequence_a_correct {n : ℕ} :
  a n = if n = 1 then 9 else 2 * n + 2 :=
by
  induction n with pn hpn
  case zero =>
    -- Prove for n = 1
    sorry
  case succ pn =>
    -- Prove for n > 1 using induction
    sorry

end sequence_a_correct_l548_548979


namespace waitress_tips_average_l548_548505

theorem waitress_tips_average :
  let tip1 := (2 : ℚ) / 4
  let tip2 := (3 : ℚ) / 8
  let tip3 := (5 : ℚ) / 16
  let tip4 := (1 : ℚ) / 4
  (tip1 + tip2 + tip3 + tip4) / 4 = 23 / 64 :=
by
  sorry

end waitress_tips_average_l548_548505


namespace range_of_a_is_eight_thirds_to_four_l548_548581

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
if h : x > 1 then a ^ x else (2 - a / 2) * x + 2

theorem range_of_a_is_eight_thirds_to_four (a : ℝ) :
  (∀ x y : ℝ, x < y → piecewise_f a x < piecewise_f a y) ↔ (8 / 3 ≤ a ∧ a < 4) :=
sorry

end range_of_a_is_eight_thirds_to_four_l548_548581


namespace parametric_line_eq_l548_548741

theorem parametric_line_eq (t : ℝ) : 
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x = 3 * t + 6 → y = 5 * t - 8 → y = m * x + b)) ∧ m = 5 / 3 ∧ b = -18 :=
sorry

end parametric_line_eq_l548_548741


namespace simplify_expression_l548_548055

variable (a : ℝ)

theorem simplify_expression (a : ℝ) : (3 * a) ^ 2 * a ^ 5 = 9 * a ^ 7 :=
by sorry

end simplify_expression_l548_548055


namespace unit_cube_probability_l548_548480

noncomputable def probability_of_painting : ℚ := 9 / 2583

theorem unit_cube_probability :
  let total_cubes := 125
  let painted_faces_cube := 1 
  let non_painted_faces_cube := 27
  let total_ways := Nat.choose total_cubes 2
  let successful_ways := painted_faces_cube * non_painted_faces_cube in
  (successful_ways / total_ways : ℚ) = probability_of_painting := by
{
  sorry
}

end unit_cube_probability_l548_548480


namespace abs_diff_C_D_base7_l548_548549

-- Conditions translated into Lean definitions
variable (C D : Nat)
variable (h1 : C ∈ Finset.range 7) -- C is a single digit in base 7
variable (h2 : D ∈ Finset.range 7) -- D is a single digit in base 7

-- The theorem to prove
theorem abs_diff_C_D_base7 : abs (C - D) = 3 := by
  sorry

end abs_diff_C_D_base7_l548_548549


namespace probability_of_seeing_red_light_l548_548154

def red_light_duration : ℝ := 30
def yellow_light_duration : ℝ := 5
def green_light_duration : ℝ := 40

def total_cycle_duration : ℝ := red_light_duration + yellow_light_duration + green_light_duration

theorem probability_of_seeing_red_light :
  (red_light_duration / total_cycle_duration) = 30 / 75 := by
  sorry

end probability_of_seeing_red_light_l548_548154


namespace rows_in_initial_patios_l548_548490

theorem rows_in_initial_patios (r c : ℕ) (h1 : r * c = 60) (h2 : (2 * c : ℚ) / r = 3 / 2) (h3 : (r + 5) * (c - 3) = 60) : r = 10 :=
sorry

end rows_in_initial_patios_l548_548490


namespace candies_share_equally_l548_548710

theorem candies_share_equally (mark_candies : ℕ) (peter_candies : ℕ) (john_candies : ℕ)
  (h_mark : mark_candies = 30) (h_peter : peter_candies = 25) (h_john : john_candies = 35) :
  (mark_candies + peter_candies + john_candies) / 3 = 30 :=
by
  sorry

end candies_share_equally_l548_548710


namespace triangle_area_l548_548449

open Real

def line1 (x y : ℝ) : Prop := y = 6
def line2 (x y : ℝ) : Prop := y = 2 + x
def line3 (x y : ℝ) : Prop := y = 2 - x

def is_vertex (x y : ℝ) (l1 l2 : ℝ → ℝ → Prop) : Prop := l1 x y ∧ l2 x y

def vertices (v1 v2 v3 : ℝ × ℝ) : Prop :=
  is_vertex v1.1 v1.2 line1 line2 ∧
  is_vertex v2.1 v2.2 line1 line3 ∧
  is_vertex v3.1 v3.2 line2 line3

def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2) -
             (v2.1 * v1.2 + v3.1 * v2.2 + v1.1 * v3.2))

theorem triangle_area : vertices (4, 6) (-4, 6) (0, 2) → area_triangle (4, 6) (-4, 6) (0, 2) = 8 :=
by
  sorry

end triangle_area_l548_548449


namespace zero_in_interval_l548_548436

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 2

theorem zero_in_interval : f 1 < 0 ∧ f 2 > 0 → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := 
by
  intros h
  sorry

end zero_in_interval_l548_548436


namespace polynomial_solution_l548_548927

theorem polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p (p x) = x * (p x) ^ 2 + x ^ 3) : 
  p = id :=
by {
    sorry
}

end polynomial_solution_l548_548927


namespace average_marks_five_subjects_l548_548854

theorem average_marks_five_subjects 
  (P total_marks : ℕ)
  (h1 : total_marks = P + 350) :
  (total_marks - P) / 5 = 70 :=
by
  sorry

end average_marks_five_subjects_l548_548854


namespace num_toothpicks_in_rectangle_l548_548643

theorem num_toothpicks_in_rectangle (length width : ℕ) (h_length : length = 20) (h_width : width = 10) :
    11 * length + 21 * width = 430 :=
by
  -- Length and width definitions
  have h_length_rows : 11 = 10 + 1 := by sorry
  have h_width_rows : 21 = 20 + 1 := by sorry

  -- Combine horizontally and vertically defined lengths
  rw [h_length, h_width]
  rw [h_length_rows, h_width_rows]
  simp

  -- Correct number of toothpicks
  exact rfl

end num_toothpicks_in_rectangle_l548_548643


namespace Winnie_keeps_two_balloons_l548_548458

statement : nat :=
  let blue := 24
  let red := 38
  let white := 44
  let green := 74
  let chartreuse := 92
  let total := blue + red + white + green + chartreuse
  total % 5

theorem Winnie_keeps_two_balloons :
  blue = 24 →
  red = 38 →
  white = 44 →
  green = 74 →
  chartreuse = 92 →
  (total % 5) = 2 := 
by
  intros
  cases blue
  cases red
  cases white
  cases green
  cases chartreuse
  sorry

end Winnie_keeps_two_balloons_l548_548458


namespace recycled_products_sale_l548_548662

/-- Given there are 3 student groups each producing 195 items, and 5 teachers each producing 70 items,
    with a maximum display limit of 500 items per unit and only 80% of items being satisfactory,
    prove the total number of recycled products they can sell at the fair is 748. -/
theorem recycled_products_sale :
  let student_groups := 3 in
  let teachers := 5 in
  let student_production := 195 in
  let teacher_production := 70 in
  let quality_factor := 0.8 in
  let max_display := 500 in
  let total_student_production := student_groups * student_production in
  let total_teacher_production := teachers * teacher_production in
  let total_production := total_student_production + total_teacher_production in
  let satisfactory_student := quality_factor * total_student_production in
  let satisfactory_teacher := quality_factor * total_teacher_production in
  let display_student := min satisfactory_student max_display in
  let display_teacher := min satisfactory_teacher max_display in
  total_production ≤ (3 * max_display + 5 * max_display) →
  (quality_factor * total_production ≤ max_display) →
  display_student + display_teacher = 748 :=
by
  intros student_groups teachers student_production teacher_production
         quality_factor max_display total_student_production
         total_teacher_production total_production satisfactory_student
         satisfactory_teacher display_student display_teacher h1 h2
  -- Proof skipped
  sorry

end recycled_products_sale_l548_548662


namespace count_valid_A_l548_548565

theorem count_valid_A : 
  let valid_A := {A | (42 % A = 0 ∧ (52348 + A * 10) % 4 = 0) ∧ (0 ≤ A ∧ A ≤ 9)} in
  fintype.elems valid_A = 2 :=
by
  sorry

end count_valid_A_l548_548565


namespace smallest_n_l548_548340

theorem smallest_n (k : ℕ) (hk : k ≥ 2) : ∃ n : ℕ, n = k + 4 ∧ (∀ S : set ℝ, S.finite → S.card = n → ∀ x ∈ S, ∃ T ⊆ S, T.finite ∧ T.card = k ∧ (T ∩ {x} = ∅) ∧ (x = T.sum)) :=
begin
  sorry
end

end smallest_n_l548_548340


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548031

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548031


namespace range_of_g_l548_548555

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arcsin (x / 3))^2 - 2 * Real.pi * (Real.arccos (x / 3)) +
  (Real.arccos (x / 3))^2 + (Real.pi^2 / 4) * (x^2 - 9 * x + 27)

theorem range_of_g :
  set.range (g : ℝ → ℝ) ∩ set.Icc (-3) 3 = set.Icc (-(3 * Real.pi^2) / 4) ((33 * Real.pi^2) / 4) :=
sorry

end range_of_g_l548_548555


namespace fraction_fewer_fish_l548_548516

variable (B C : Type) -- Type for Brian and Chris

-- Variables for number of fish
variable (brian_fish_per_trip chris_fish_per_trip total_fish : ℝ)
variable (brian_trips chris_trips : ℕ)

-- Conditions given in the problem
variable (h1 : brian_trips = 2 * chris_trips)
variable (h2 : brian_fish_per_trip = 400)
variable (h3 : brian_trips * brian_fish_per_trip + chris_trips * chris_fish_per_trip = 13600)
variable (h4 : chris_trips = 10)

-- The statement we need to prove
theorem fraction_fewer_fish :
  let f := 1 - (brian_fish_per_trip / chris_fish_per_trip) in
  f = 2 / 7 :=
by
  -- ... here goes the proof
  sorry

end fraction_fewer_fish_l548_548516


namespace room_dimension_l548_548307

theorem room_dimension {a : ℝ} (h1 : a > 0) 
  (h2 : 4 = 2^2) 
  (h3 : 14 = 2 * (7)) 
  (h4 : 2 * a = 14) :
  (a + 2 * a - 2 = 19) :=
sorry

end room_dimension_l548_548307


namespace sin_alpha_eq_l548_548998

-- Define the conditions
def point := (1/2 : ℝ, (sqrt 3) / 2)
def x := point.1
def y := point.2
def r := (√(x^2 + y^2) : ℝ)

-- Statement that needs to be proven
theorem sin_alpha_eq : sin (atan2 y x) = y / r :=
  sorry

end sin_alpha_eq_l548_548998


namespace power_mod_remainder_l548_548932

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l548_548932


namespace ellipse_standard_equation_and_slope_range_l548_548981

/-- Given an ellipse C with the equation (x^2/a^2) + (y^2/b^2) = 1, where a > b > 0, eccentricity 1/2,
    and a point (-1, 3/2) lying on the ellipse, prove that the standard equation of the ellipse is 
    (x^2/4) + (y^2 / 3) = 1 and the range of the slope of the line AR is [-1/8, 1/8]. --/
theorem ellipse_standard_equation_and_slope_range :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (1 / 2) = a * (1 / a^2 + 9 / (4 * b^2)) ∧
  (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) ∧
  (∀ m : ℝ, m ≠ 0 → ((1 / (4 * m + 4 / m)) ∈ Icc (- 1 / 8) (1 / 8))) :=
sorry

end ellipse_standard_equation_and_slope_range_l548_548981


namespace sum_of_possible_b_values_l548_548938

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l548_548938


namespace increase_area_between_circles_l548_548803

def area (r : ℝ) : ℝ := π * r^2

def percentage_increase (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem increase_area_between_circles :
  let r1 := 6
  let r2 := 4
  let r1_new := 1.5 * r1
  let r2_new := 0.75 * r2
  let area_between_original := area r1 - area r2
  let area_between_new := area r1_new - area r2_new
  percentage_increase area_between_original area_between_new = 260 :=
by {
  let r1 := 6
  let r2 := 4
  let r1_new := 1.5 * r1
  let r2_new := 0.75 * r2
  let area_between_original := area r1 - area r2
  let area_between_new := area r1_new - area r2_new
  have h1 : area r1 = π * r1^2 := rfl,
  have h2 : area r2 = π * r2^2 := rfl,
  have h3 : area r1_new = π * r1_new^2 := rfl,
  have h4 : area r2_new = π * r2_new^2 := rfl,
  have original_area_diff: area_between_original = 20 * π :=
    by simp [area_between_original, area, h1, h2]; ring,
  have new_area_diff: area_between_new = 72 * π :=
    by simp [area_between_new, area, h3, h4]; ring,
  have percentage_increase_proof: percentage_increase (20 * π) (72 * π) = 260 :=
    by simp [percentage_increase]; ring,
  exact percentage_increase_proof
}

end increase_area_between_circles_l548_548803


namespace problem1_problem2_l548_548601

-- Given the function f(x) = ln(x) + a * x - 1 / x + b
def f (x : ℝ) (a b : ℝ) : ℝ := log x + a * x - 1 / x + b

-- The function g(x) = f(x) + 2 / x
def g (x : ℝ) (a b : ℝ) : ℝ := f x a b + 2 / x

-- Prove a <= -1/4 given g(x) is decreasing
theorem problem1 (a b : ℝ) (h : ∀ x > 0, (g x a b)' ≤ 0) : a ≤ -1 / 4 :=
sorry

-- Prove a ≤ 1 - b given f(x) ≤ 0 always holds
theorem problem2 (a b : ℝ) (h : ∀ x > 0, f x a b ≤ 0) : a ≤ 1 - b :=
sorry

end problem1_problem2_l548_548601


namespace sum_of_products_less_than_one_l548_548434

theorem sum_of_products_less_than_one 
  (n : ℕ) (k : ℕ) 
  (a : Fin n → ℝ) 
  (h_sum : ∑ i in Finset.finRange n, a i = 1)
  (h_pos : ∀ i, 0 < a i)
  (h_nk : k < n):
  (∑ s in Finset.powersetLen k (Finset.finRange n), (s.val.toFinset : Finset (Fin n)).prod a) < 1 :=
sorry

end sum_of_products_less_than_one_l548_548434


namespace at_least_one_negative_root_l548_548838

-- Define a quadratic polynomial with two distinct roots and the condition given
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

axiom distinct_roots {a b c : ℝ} (h : a ≠ 0) : quadratic_polynomial a b c has_two_distinct_roots

axiom quadratic_inequality_holds {a b c : ℝ} (h : a ≠ 0) :
  ∀ (x y : ℝ), quadratic_polynomial a b c (x^2 + y^2) ≥ quadratic_polynomial a b c (2 * x * y)

-- State the theorem: If the above conditions hold, then at least one root is negative
theorem at_least_one_negative_root {a b c : ℝ} (h_a : a ≠ 0) (h1 : distinct_roots h_a) (h2 : quadratic_inequality_holds h_a) :
  ∃ r : ℝ, is_root (quadratic_polynomial a b c) r ∧ r < 0 := sorry

end at_least_one_negative_root_l548_548838


namespace total_roses_received_l548_548863

open Nat

def number_of_roses_from_parents : Nat := 2 * 12
def number_of_roses_from_friends (n_friends : Nat) : Nat := n_friends * 2

theorem total_roses_received (n_friends : Nat) (h : n_friends = 10) :
  number_of_roses_from_parents + number_of_roses_from_friends n_friends = 44 :=
by
  simp [number_of_roses_from_parents, number_of_roses_from_friends, h]
  sorry

end total_roses_received_l548_548863


namespace sqrt_chain_lt_3_l548_548376

theorem sqrt_chain_lt_3 (n : ℕ) (h : n > 0) : sqrt (2 * sqrt (3 * sqrt (4 * sqrt n))) < 3 := 
by sorry

end sqrt_chain_lt_3_l548_548376


namespace plywood_cut_difference_l548_548123

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548123


namespace alex_baked_cherry_pies_l548_548509

theorem alex_baked_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ)
  (h1 : total_pies = 30)
  (h2 : ratio_apple = 1)
  (h3 : ratio_blueberry = 5)
  (h4 : ratio_cherry = 4) :
  (total_pies * ratio_cherry / (ratio_apple + ratio_blueberry + ratio_cherry) = 12) :=
by {
  sorry
}

end alex_baked_cherry_pies_l548_548509


namespace solve_eq_z6_neg_64_l548_548936

theorem solve_eq_z6_neg_64 (z : ℂ) (h : z ^ 6 = -64) :
z = (complex.sqrt[3] 2) * (1 + complex.I) ∨
z = (complex.sqrt[3] 2) * (-1 - complex.I) ∨
z = (complex.sqrt[3] 2) * (-1 + complex.I) ∨
z = (complex.sqrt[3] 2) * (1 - complex.I) :=
sorry

end solve_eq_z6_neg_64_l548_548936


namespace max_draw_without_sum_multiple_of_five_l548_548570

theorem max_draw_without_sum_multiple_of_five : 
  ∃ (S : set ℕ), (∀ a b ∈ S, (a + b) % 5 ≠ 0) ∧ S ⊆ {n : ℕ | 1 ≤ n ∧ n ≤ 1000} ∧ S.card = 401 :=
sorry

end max_draw_without_sum_multiple_of_five_l548_548570


namespace find_p_q_l548_548995

noncomputable def quadratic_roots {p q : ℝ} : Prop :=
  ∃ (α β : ℝ), (x^2 + p * x + q = 0) ∧ α ≠ β ∧ α + β = -p ∧ α * β = q ∧
  ({α, β} ∩ {1, 2, 3, 4}) = {α, β} ∧ ({α, β} ∩ {2, 4, 5, 6}) = ∅

theorem find_p_q :
  quadratic_roots (-4) (3) :=
by 
  sorry

end find_p_q_l548_548995


namespace all_palaces_face_south_l548_548772

def days_to_face_south : ℕ := 525

theorem all_palaces_face_south :
  ∃ x : ℕ, x ≡ 15 [MOD 30] ∧ x ≡ 25 [MOD 50] ∧ x ≡ 35 [MOD 70] ∧ x = days_to_face_south := 
begin
  use days_to_face_south,
  split, { exact nat.modeq_intro rfl, },
  split, { exact nat.modeq_intro rfl, },
  split, { exact nat.modeq_intro rfl, },
  refl,
sorry

end all_palaces_face_south_l548_548772


namespace proof_problem_l548_548237

variable (p q : Prop)

-- Condition definitions based on the problem statement
def condition_p : Prop := ∃ α : ℝ, sin(π - α) = cos α
def condition_q : Prop := ∀ m : ℝ, m > 0 → (∃ e : ℝ, e = sqrt (1 + (m^2 / m^2)) ∧ e = sqrt 2)

theorem proof_problem (h₁ : condition_p) (h₂ : condition_q) : (¬ p ∨ q) :=
sorry

end proof_problem_l548_548237


namespace rearrange_digits_3003_l548_548614

theorem rearrange_digits_3003 : 
  let digits := [3, 3, 0, 0] in 
  let valid_permutations := (l : List ℕ) → l.length = 4 ∧ ¬l.head = some 0 in 
  (List.permutations digits).countP valid_permutations = 3 := 
by sorry

end rearrange_digits_3003_l548_548614


namespace value_of_expression_l548_548651

theorem value_of_expression (x : ℝ) (h : x^2 + x + 1 = 8) : 4 * x^2 + 4 * x + 9 = 37 :=
by
  sorry

end value_of_expression_l548_548651


namespace suzanna_total_distance_l548_548733

-- Conditions
def ride_speed := 1.5 -- miles per 7 minutes
def initial_ride_time := 21 -- minutes
def break_time := 5 -- minutes
def post_break_ride_time := 14 -- minutes
def interval := 7 -- minutes

-- Define the total time of riding not including break time
def total_ride_time := initial_ride_time + post_break_ride_time

-- Translate the total distance Suzanna rides to a Lean theorem
theorem suzanna_total_distance : 
    (initial_ride_time / interval) * ride_speed + (post_break_ride_time / interval) * ride_speed = 7.5 :=
by
  sorry

end suzanna_total_distance_l548_548733


namespace circles_intersect_and_common_chord_l548_548983

open Real

def circle1 (x y : ℝ) := x^2 + y^2 - 6 * x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6 = 0

theorem circles_intersect_and_common_chord :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) ∧ (∀ x y : ℝ, circle1 x y → circle2 x y → 3 * x - 2 * y = 0) :=
by
  sorry

end circles_intersect_and_common_chord_l548_548983


namespace length_FG_eq_12_point_8_l548_548169

/--
Given a square ABCD with side length 20, point E on AD such that AE = 15,
AF perpendicular to BE, and FG perpendicular to BC, 
the length of FG is 12.8.
-/
theorem length_FG_eq_12_point_8 
  (A B C D E F G : ℝ -> ℝ -> Prop) 
  (side_length : ℝ) (h_sq : side_length = 20) 
  (h_AE : E(0, 15)) 
  (h_AF_perp_BE : ∀ x y : ℝ, (F x y) -> (BE x y) -> ∃ a b : ℝ, x^2 + y^2 = a^2 + b^2)
  (h_FG_perp_BC : ∀ u v : ℝ, (G u v) -> (BC u v) -> u = 20 - v) : 
  ∃ fg_len : ℝ, fg_len = 12.8 := 
by 
  sorry

end length_FG_eq_12_point_8_l548_548169


namespace perimeters_positive_difference_l548_548099

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548099


namespace cos_A_plus_cos_C_arithmetic_sequence_tan_A_tan_C_geometric_sequence_l548_548298

-- Definitions for triangle angles and sides
variables {A B C : ℝ} {a b c : ℝ}

-- Definition of arithmetic sequence for angles
def is_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

-- Definition of geometric sequence for sides
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Problem 1
theorem cos_A_plus_cos_C_arithmetic_sequence (h1 : is_arithmetic_sequence A B C) (h2 : A + B + C = real.pi) :
  ∃ x ∈ Ioo (1/2 : ℝ) 1, cos A + cos C = x :=
sorry

-- Problem 2
theorem tan_A_tan_C_geometric_sequence (h1 : is_geometric_sequence a b c) (h2 : real.cos B = 4/5) :
  1 / real.tan A + 1 / real.tan C = 5 / 3 :=
sorry

end cos_A_plus_cos_C_arithmetic_sequence_tan_A_tan_C_geometric_sequence_l548_548298


namespace tank_capacity_proof_l548_548405

def capacity_of_each_tank (total_oil : ℕ) (num_tanks : ℕ) : ℕ :=
  (total_oil / num_tanks).to_nat

theorem tank_capacity_proof :
  ∀ (total_oil num_tanks : ℕ), total_oil = 728 → num_tanks = 23 → capacity_of_each_tank total_oil num_tanks = 32 :=
by
  intros total_oil num_tanks h_oil h_tanks
  rw [h_oil, h_tanks]
  simp [capacity_of_each_ttank, Nat.div]
  sorry -- Proof steps are omitted.

end tank_capacity_proof_l548_548405


namespace option_c_has_minimum_value_4_l548_548011

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548011


namespace plywood_cut_perimeter_difference_l548_548105

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548105


namespace car_average_speed_l548_548473

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l548_548473


namespace perimeter_difference_l548_548066

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548066


namespace average_speed_is_one_l548_548815

-- Definition of distance and time
def distance : ℕ := 1800
def time_in_minutes : ℕ := 30
def time_in_seconds : ℕ := time_in_minutes * 60

-- Definition of average speed as distance divided by time
def average_speed (distance : ℕ) (time : ℕ) : ℚ :=
  distance / time

-- Theorem: Given the distance and time, the average speed is 1 meter per second
theorem average_speed_is_one : average_speed distance time_in_seconds = 1 :=
  by
    sorry

end average_speed_is_one_l548_548815


namespace less_money_than_Bob_l548_548640

noncomputable def Jennas_money (P: ℝ) : ℝ := 2 * P
noncomputable def Phils_money (B: ℝ) : ℝ := B / 3
noncomputable def Bobs_money : ℝ := 60
noncomputable def Johns_money (P: ℝ) : ℝ := P + 0.35 * P
noncomputable def average (x y: ℝ) : ℝ := (x + y) / 2

theorem less_money_than_Bob :
  ∀ (P Q J B : ℝ),
    P = Phils_money B →
    J = Jennas_money P →
    Q = Johns_money P →
    B = Bobs_money →
    average J Q = B - 0.25 * B →
    B - J = 20
  :=
by
  intros P Q J B hP hJ hQ hB h_avg
  -- Proof goes here
  sorry

end less_money_than_Bob_l548_548640


namespace terminal_side_equivalence_l548_548790

theorem terminal_side_equivalence (a b c d : ℤ) : ((a ≡ 330 [ZMOD 360]) ∨ (b ≡ 330 [ZMOD 360]) ∨ (c ≡ 330 [ZMOD 360]) ∨ (d ≡ 330 [ZMOD 360])) :=
by
  have ha : 30 ≡ 30 [ZMOD 360] := by sorry
  have hb : -30 ≡ 330 [ZMOD 360] := by sorry
  have hc : 630 ≡ 270 [ZMOD 360] := by sorry
  have hd : -630 ≡ 90 [ZMOD 360] := by sorry
  exact Or.inr (Or.inl (Or.inl hb))

end terminal_side_equivalence_l548_548790


namespace real_roots_eventually_l548_548338

noncomputable theory

open Polynomial

-- Define the sequence of polynomials fn
def poly_seq (f : Polynomial ℝ) : ℕ → Polynomial ℝ
| 0       := f
| (n + 1) := poly_seq n + (poly_seq n).derivative

-- Statement of the problem
theorem real_roots_eventually
  (f : Polynomial ℝ) (hf : f ≠ 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∀ x : ℝ, (poly_seq f n).is_root x → x ∈ ℝ :=
sorry

end real_roots_eventually_l548_548338


namespace find_a_and_min_f_l548_548266

noncomputable def abs_diff (x : ℝ) (y : ℝ) : ℝ := abs (x - y)

theorem find_a_and_min_f (a : ℕ) (ha_pos : a > 0)
  (h3over2_in_A : abs_diff (3 / 2) 2 < a)
  (h1over2_not_in_A : abs_diff (1 / 2) 2 ≥ a) :
  a = 1 ∧ (∀ x, abs (x + 1) + abs (x - 2) ≥ 3) :=
by
  sorry

end find_a_and_min_f_l548_548266


namespace solve_system_of_equations_l548_548731

theorem solve_system_of_equations (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : 1 / (x * y) = x / z + 1)
  (h2 : 1 / (y * z) = y / x + 1)
  (h3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by
  sorry

end solve_system_of_equations_l548_548731


namespace regular_decagon_product_l548_548382

-- Define Q_k as the points of a regular decagon in the coordinate plane.
-- Q_1 is at (2,0) and Q_6 is at (4,0).

noncomputable def Q (n : ℕ) : ℂ := sorry -- Function to define the complex points Q_n

theorem regular_decagon_product :
  ∃ Q : ℕ → ℂ, 
  (Q 1 = 2) ∧
  (Q 6 = 4) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 10 → Q k = (x_1 + y_1 * (complex.I)) -- appropriate complex values for each Q_k
       ∨ Q k = (x_2 + y_2 * (complex.I))
       ∨ Q k = (x_3 + y_3 * (complex.I))
       ∨ Q k = (x_4 + y_4 * (complex.I))
       ∨ Q k = (x_5 + y_5 * (complex.I))
       ∨ Q k = (x_6 + y_6 * (complex.I))
       ∨ Q k = (x_7 + y_7 * (complex.I))
       ∨ Q k = (x_8 + y_8 * (complex.I))
       ∨ Q k = (x_9 + y_9 * (complex.I))
       ∨ Q k = (x_{10} + y_{10} * (complex.I))) ∧
  (finset.prod (finset.range 10) (λ i, Q (i + 1)) = 59048) :=
sorry -- Proof would go here

end regular_decagon_product_l548_548382


namespace eight_digit_divisible_by_eleven_l548_548537

theorem eight_digit_divisible_by_eleven (n : ℕ) (h : 0 ≤ n ∧ n < 10) : 9637n428 is divisible by 11 ↔ n = 9 :=
by
  -- Define the digits of the number 9637n428
  let digits := [9, 6, 3, 7, n, 4, 2, 8]
  
  -- Calculate the sum of the digits in odd positions
  let odd_sum := digits[0] + digits[2] + digits[4] + digits[6]
  
  -- Calculate the sum of the digits in even positions
  let even_sum := digits[1] + digits[3] + digits[5] + digits[7]
  
  -- Use the divisibility rule for 11
  have divisibility_condition : (odd_sum - even_sum) % 11 = 0 ↔ n = 9 := sorry

  exact divisibility_condition

end eight_digit_divisible_by_eleven_l548_548537


namespace Gage_received_one_third_of_Grady_blue_l548_548613

-- Definitions based on conditions
def Grady_red_cubes : ℕ := 20
def Grady_blue_cubes : ℕ := 15
def Gage_orig_red_cubes : ℕ := 10
def Gage_orig_blue_cubes : ℕ := 12
def Gage_total_cubes : ℕ := 35
def fraction_red_given_to_Gage : ℚ := 2 / 5

-- Converting conditions into corresponding Lean statements
def Grady_red_given_to_Gage := fraction_red_given_to_Gage * Grady_red_cubes   -- 2/5 of 20
def Gage_now_red_cubes := Gage_orig_red_cubes + Grady_red_given_to_Gage
def Gage_now_blue_cubes := Gage_total_cubes - Gage_now_red_cubes
def Gage_blue_received : ℕ := Gage_now_blue_cubes - Gage_orig_blue_cubes

-- Proof statement
theorem Gage_received_one_third_of_Grady_blue :
  Gage_blue_received / Grady_blue_cubes = 1 / 3 := by
  sorry

end Gage_received_one_third_of_Grady_blue_l548_548613


namespace distance_interval_l548_548161

noncomputable def distance_set (d : ℝ) : Prop :=
  (d < 8) ∧ (d > 7) ∧ (d > 6) ∧ (d ≠ 10)

theorem distance_interval (d : ℝ) (h : distance_set d) : d ∈ set.Ioo 7 8 :=
by
  sorry

end distance_interval_l548_548161


namespace total_payment_per_week_l548_548446

def X_pay (Y_pay : ℝ) := 1.2 * Y_pay
def total_pay (X_pay Y_pay : ℝ) := X_pay + Y_pay

theorem total_payment_per_week :
  ∀ (Y_pay : ℝ), Y_pay = 250 → total_pay (X_pay Y_pay) Y_pay = 550 := by
  intros Y_pay hY_pay
  rw [hY_pay]
  unfold X_pay total_pay
  norm_num
  sorry

end total_payment_per_week_l548_548446


namespace second_caterer_cheaper_l548_548370

theorem second_caterer_cheaper (x : ℕ) :
  (150 + 18 * x > 250 + 14 * x) → x ≥ 26 :=
by
  intro h
  sorry

end second_caterer_cheaper_l548_548370


namespace plywood_cut_difference_l548_548072

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548072


namespace ratio_of_scores_l548_548445

theorem ratio_of_scores 
  (u v : ℝ) 
  (h1 : u > v) 
  (h2 : u - v = (u + v) / 2) 
  : v / u = 1 / 3 :=
sorry

end ratio_of_scores_l548_548445


namespace diveScoreCalculation_l548_548655

noncomputable def calculateDiveScore (scores : List ℝ) (difficulty : ℝ) (multiplier : ℝ) : ℝ :=
  let sorted_scores := scores.erase (scores.maximum) ∘ scores.erase (scores.minimum)
  (sorted_scores.sum * difficulty) * multiplier

theorem diveScoreCalculation :
  calculateDiveScore [7.5, 8.0, 9.0, 6.0, 8.8, 9.5] 3.2 1.2 = 127.872 :=
sorry

end diveScoreCalculation_l548_548655


namespace bus_speed_l548_548127

def distance : ℝ := 350.028
def time : ℝ := 10
def speed_kmph : ℝ := 126.01

theorem bus_speed :
  (distance / time) * 3.6 = speed_kmph := 
sorry

end bus_speed_l548_548127


namespace max_sum_of_lengths_l548_548043

noncomputable def length_of_integer (k : ℕ) : ℕ :=
if h : k > 1 then (Multiset.card (Multiset.bind (Nat.factors k)))
else 0

theorem max_sum_of_lengths (x y : ℕ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
    length_of_integer x + length_of_integer y = 16 := by
    sorry

end max_sum_of_lengths_l548_548043


namespace sqrt_29_parts_a_plus_sqrt_5_b_value_l548_548380

theorem sqrt_29_parts :
  (⌊real.sqrt 29⌋ = 5) ∧ (real.sqrt 29 - 5 = real.sqrt 29 - ⌊real.sqrt 29⌋) := 
by sorry

theorem a_plus_sqrt_5_b_value :
  let a := 5 + real.sqrt 5 - ⌊5 + real.sqrt 5⌋
  let b := ⌊5 - real.sqrt 5⌋
  in a + real.sqrt 5 * b = 3 * real.sqrt 5 - 2 :=
by sorry

end sqrt_29_parts_a_plus_sqrt_5_b_value_l548_548380


namespace main_l548_548049

-- Definition for part (a)
def part_a : Prop :=
  ∀ (a b : ℕ), a = 300 ∧ b = 200 → 3^b > 2^a

-- Definition for part (b)
def part_b : Prop :=
  ∀ (c d : ℕ), c = 40 ∧ d = 28 → 3^d > 2^c

-- Definition for part (c)
def part_c : Prop :=
  ∀ (e f : ℕ), e = 44 ∧ f = 53 → 4^f > 5^e

-- Main conjecture proving all parts
theorem main : part_a ∧ part_b ∧ part_c :=
by
  sorry

end main_l548_548049


namespace perimeter_difference_l548_548067

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548067


namespace main_theorem_l548_548664

-- Define a point structure for convenience
structure Point (α : Type) :=
  (x : α)
  (y : α)

namespace Geometry

variables {α : Type} [Field α] -- Assume points are defined in a field for generality

-- Define the triangle and its relevant points
structure Triangle (α : Type) :=
  (A B C A1 B1 C1 A0 B0 C0 : Point α)

-- Define the condition of the acute triangle
def isAcute (T : Triangle α) : Prop := 
  -- Define what it means for a triangle to be acute here
  -- This is a placeholder for the actual condition.
  sorry

-- Define the conditions for the areas in question
def area (p1 p2 p3 : Point α) : α := 
  -- Area calculation here: placeholder
  sorry

def hexArea (A C1 B A1 C B1 : Point α) : α := 
  -- Hexagon area calculation here: placeholder
  sorry

def prob_1 : Prop :=
  ∀ T : Triangle α, isAcute T →
  area T.A0 T.B0 T.C0 = 2 * hexArea T.A T.C1 T.B T.A1 T.C T.B1

def prob_2 : Prop :=
  ∀ T : Triangle α, isAcute T →
  area T.A0 T.B0 T.C0 ≥ 4 * area T.A T.B T.C

-- Combining both problems
theorem main_theorem : prob_1 ∧ prob_2 :=
by
  -- You need to fill in this proof using Lean's tactics
  sorry

end Geometry

end main_theorem_l548_548664


namespace molecular_weight_3_moles_BaSO4_l548_548451

def atomic_weight_Ba := 137.327
def atomic_weight_S := 32.065
def atomic_weight_O := 15.999
def num_Oatoms := 4

def molecular_weight_BaSO4 : ℝ :=
  atomic_weight_Ba + atomic_weight_S + num_Oatoms * atomic_weight_O

def weight_3_moles_BaSO4 := 3 * molecular_weight_BaSO4

theorem molecular_weight_3_moles_BaSO4 : weight_3_moles_BaSO4 = 700.164 := by
  sorry

end molecular_weight_3_moles_BaSO4_l548_548451


namespace exists_finite_subgraph_set_l548_548211

-- Define the problem in Lean

variable {S : Type} [surface : CurvedSurface S]

-- The main existential statement
theorem exists_finite_subgraph_set (S : Type) [CurvedSurface S] :
  ∃ (H : ℕ → graph), (∀ (G : graph), embeddable G S ↔ (∀ i, ¬contains_subgraph G (H i))) :=
sorry

end exists_finite_subgraph_set_l548_548211


namespace bertha_family_daughters_and_granddaughters_no_daughters_l548_548514

theorem bertha_family_daughters_and_granddaughters_no_daughters :
  let daughters := 8 in
  let total_daughters_and_granddaughters := 36 in
  ∃ x : ℕ, 
    (x ≤ daughters) ∧ 
    (4 * x + daughters = total_daughters_and_granddaughters) ∧ 
    (daughters - x + 4 * x = total_daughters_and_granddaughters) ∧ 
    (daughters - x + 4 * x + (total_daughters_and_granddaughters - (daughters + 4 * x)) = 29) :=
begin
  sorry
end

end bertha_family_daughters_and_granddaughters_no_daughters_l548_548514


namespace goods_train_speed_approx_60_2_kmph_l548_548799

noncomputable def goods_train_speed 
  (man_train_speed_kmph : ℝ) 
  (time_to_pass_seconds : ℝ) 
  (goods_train_length_meters : ℝ) : ℝ := 
  let man_train_speed_mps := man_train_speed_kmph * 1000 / 3600
  let relative_speed_mps := goods_train_length_meters / time_to_pass_seconds
  let goods_train_speed_mps := relative_speed_mps - man_train_speed_mps
  goods_train_speed_mps * 3600 / 1000

theorem goods_train_speed_approx_60_2_kmph :
  goods_train_speed 55 10 320 ≈ 60.2 := 
by 
  sorry

end goods_train_speed_approx_60_2_kmph_l548_548799


namespace parabola_vertex_l548_548760

theorem parabola_vertex : ∃ h k, ∀ x, -2 * x^2 + 3 = -2 * (x - h)^2 + k ∧ h = 0 ∧ k = 3 :=
by
  use 0
  use 3
  intros x
  split
  · ring
  split
  · refl
  · refl
  sorry

end parabola_vertex_l548_548760


namespace complex_number_value_l548_548291

theorem complex_number_value (z : ℂ) (h : z * complex.I = 1 - complex.I) : z = -1 - complex.I := 
sorry

end complex_number_value_l548_548291


namespace distinct_z_values_count_l548_548512

noncomputable def num_distinct_z_values (x y z : ℤ) (a b c : ℕ) : ℕ :=
  let possible_values := {198, 396, 594, 792} in
  possible_values.card

theorem distinct_z_values_count : (num_distinct_z_values >= 0) := by
  sorry

end distinct_z_values_count_l548_548512


namespace negation_of_exists_l548_548270

theorem negation_of_exists (p : Prop) : 
  (∃ (x₀ : ℝ), x₀ > 0 ∧ |x₀| ≤ 2018) ↔ 
  ¬(∀ (x : ℝ), x > 0 → |x| > 2018) :=
by sorry

end negation_of_exists_l548_548270


namespace samantha_average_speed_l548_548386

-- Definitions for the conditions
def time_segment1 : ℝ := 5 + 45/60
def time_segment2 : ℝ := 7 + 15/60
def distance_segment1 : ℝ := 320
def distance_segment2 : ℝ := 400

-- Auxiliary definitions
def total_time : ℝ := time_segment1 + time_segment2
def total_distance : ℝ := distance_segment1 + distance_segment2
def average_speed : ℝ := total_distance / total_time

-- Main theorem to prove
theorem samantha_average_speed : average_speed ≈ 55.3846 :=
by
  -- Calculation auxiliary for readability and proof skipping
  let time1 := time_segment1
  let time2 := time_segment2
  let dist1 := distance_segment1
  let dist2 := distance_segment2
  let totalTime := time1 + time2
  let totalDistance := dist1 + dist2
  let speed := totalDistance / totalTime
  have h : speed ≈ (720 / 13) := sorry
  show speed ≈ 55.3846 from sorry

end samantha_average_speed_l548_548386


namespace plywood_perimeter_difference_l548_548092

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548092


namespace find_x2_plus_y2_l548_548633

noncomputable def xy : ℝ := 12
noncomputable def eq2 (x y : ℝ) : Prop := x^2 * y + x * y^2 + x + y = 120

theorem find_x2_plus_y2 (x y : ℝ) (h1 : xy = 12) (h2 : eq2 x y) : 
  x^2 + y^2 = 10344 / 169 :=
sorry

end find_x2_plus_y2_l548_548633


namespace plywood_cut_difference_l548_548125

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548125


namespace max_sum_grid_A_l548_548805

section MaximumSumOfGridA
  -- Define the grid and the sums
  def Grid := (Fin 10) → (Fin 10) → ℤ

  variable (A : Grid)

  def row_sum (i : Fin 10) : ℤ :=
    ∑ j in (Finset.univ : Finset (Fin 10)), A i j

  def col_sum (j : Fin 10) : ℤ :=
    ∑ i in (Finset.univ : Finset (Fin 10)), A i j

  -- Define grid B where B(i, j) = min(s_i, t_j)
  def gridB (A : Grid) (i j : Fin 10) : ℤ :=
    min (row_sum A i) (col_sum A j)

  -- The proof problem statement
  theorem max_sum_grid_A : ∃ (A : Grid), (∑ i in (Finset.univ : Finset (Fin 10)), row_sum A i) = 955 :=
  sorry

end MaximumSumOfGridA

end max_sum_grid_A_l548_548805


namespace monotonic_increasing_interval_l548_548420

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3) / Real.log (1 / 2)

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), (f x = Real.log (x^2 - 2 * x - 3) / Real.log (1 / 2))
    → (x < -1 ∨ x > 3)
    → (∀ y : ℝ, (0 < y) → f y ≤ f (y - ε))
    → (∀ t, f t ∈ (-∞, -1)) :=
sorry

end monotonic_increasing_interval_l548_548420


namespace min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548028

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def f_B (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def f_C (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def f_D (x : ℝ) : ℝ := log x + 4 / log x

theorem min_value_f_C : ∃ x : ℝ, f_C x = 4 :=
by sorry

theorem min_value_f_A : ∀ x : ℝ, f_A x ≠ 4 :=
by sorry

theorem min_value_f_B : ∀ x : ℝ, f_B x ≠ 4 :=
by sorry

theorem min_value_f_D : ∀ x : ℝ, f_D x ≠ 4 :=
by sorry

end min_value_f_C_min_value_f_A_min_value_f_B_min_value_f_D_l548_548028


namespace one_vertical_asymptote_l548_548215

def g (x c : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - 5*x + 6)

theorem one_vertical_asymptote (c : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ g x c = ∞)
  ↔ c = 2 := 
sorry

end one_vertical_asymptote_l548_548215


namespace optionC_has_min_4_l548_548004

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548004


namespace integral_cos8_0_2pi_l548_548877

noncomputable def definite_integral_cos8 (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.cos (x / 4)) ^ 8

theorem integral_cos8_0_2pi :
  definite_integral_cos8 0 (2 * Real.pi) = (35 * Real.pi) / 64 :=
by
  sorry

end integral_cos8_0_2pi_l548_548877


namespace train_length_l548_548155

theorem train_length :
  ∃ L : ℝ, 
    (∀ V : ℝ, V = L / 24 ∧ V = (L + 650) / 89) → 
    L = 240 :=
by
  sorry

end train_length_l548_548155


namespace complex_division_problem_l548_548317

def symmetric_imag_axis (z1 z2 : ℂ) : Prop :=
  z1 = complex.conj z2

noncomputable def z1 : ℂ := -1 + complex.I

theorem complex_division_problem :
  ∀ z2 : ℂ, symmetric_imag_axis z1 z2 → (z1 / z2) = complex.I :=
by
  intro z2
  intro h
  sorry

end complex_division_problem_l548_548317


namespace eccentricity_of_hyperbola_l548_548608

noncomputable def parabola := λ x y : ℝ, x^2 = 8 * y
noncomputable def hyperbola (a b : ℝ) := λ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
noncomputable def asymptote (a b : ℝ) := λ x y : ℝ, y = (b / a) * x
noncomputable def intersection_point (a b : ℝ) := (8 * b / a, 8 * b^2 / a^2)
noncomputable def distance_to_axis (a b : ℝ) := abs (8 * b^2 / a^2 - (-2))

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : distance_to_axis a b = 4) : 
  let e := (Real.sqrt (a^2 + b^2)) / a 
  in e = Real.sqrt 5 / 2 :=
by 
  sorry

end eccentricity_of_hyperbola_l548_548608


namespace inequality_proof_l548_548720

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by
  sorry

end inequality_proof_l548_548720


namespace ones_digit_of_largest_power_of_3_in_factorial_l548_548554

theorem ones_digit_of_largest_power_of_3_in_factorial (n : ℕ) (h : n = 27) : 
  ∃ k, ∀ m ≥ k, (3 ^ m) ∣ (nat.factorial n) ∧ (nat.mod (3 ^ m) 10) = 3 :=
by
  sorry

end ones_digit_of_largest_power_of_3_in_factorial_l548_548554


namespace sum_of_possible_b_values_l548_548940

theorem sum_of_possible_b_values : 
  (∑ b in { b | b ∈ {1, 2, 3, 4} ∧ ∃ k : ℕ, 49 - 12 * b = k * k }, b) = 6 :=
by 
  sorry

end sum_of_possible_b_values_l548_548940


namespace simplify_expression_l548_548729

theorem simplify_expression :
  (sqrt 450 / sqrt 200) - (sqrt 175 / sqrt 75) = (9 - 2 * sqrt 21) / 6 :=
by 
  sorry

end simplify_expression_l548_548729


namespace new_percentage_of_alcohol_correct_l548_548059

def volume_of_mixture : ℝ := 15
def percentage_alcohol_in_original : ℝ := 20
def added_water_volume : ℝ := 3
def new_percentage_alcohol : ℝ := 16.67

theorem new_percentage_of_alcohol_correct :
  let original_alcohol_volume := percentage_alcohol_in_original / 100 * volume_of_mixture in
  let new_total_volume := volume_of_mixture + added_water_volume in
  let calculated_percentage_alcohol := (original_alcohol_volume / new_total_volume) * 100 in
  calculated_percentage_alcohol = new_percentage_alcohol :=
by
  sorry

end new_percentage_of_alcohol_correct_l548_548059


namespace solve_trig_equation_l548_548728

-- Definition of the conditions
def sin_neq_0 (x : ℝ) : Prop := sin (2 * x) ≠ 0
def cos_neq_0 (x : ℝ) : Prop := cos (2 * x) ≠ 0

-- Main theorem statement
theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (frac (1 + sin x + cos x + sin (2 * x) + cos (2 * x)) (tan (2 * x)) = 0) →
  (sin_neq_0 x) →
  (cos_neq_0 x) →
  (∃ n : ℤ, x = (2 * π / 3) * (3 * n + 1) ∨ x = (2 * π / 3) * (3 * n - 1)) :=
by sorry

end solve_trig_equation_l548_548728


namespace larry_channels_l548_548684

theorem larry_channels : (initial_channels : ℕ) 
                         (channels_taken : ℕ) 
                         (channels_replaced : ℕ) 
                         (reduce_channels : ℕ) 
                         (sports_package : ℕ) 
                         (supreme_sports_package : ℕ) 
                         (final_channels : ℕ)
                         (h1 : initial_channels = 150)
                         (h2 : channels_taken = 20)
                         (h3 : channels_replaced = 12)
                         (h4 : reduce_channels = 10)
                         (h5 : sports_package = 8)
                         (h6 : supreme_sports_package = 7)
                         (h7 : final_channels = initial_channels - channels_taken + channels_replaced - reduce_channels + sports_package + supreme_sports_package)
                         : final_channels = 147 :=
by sorry

end larry_channels_l548_548684


namespace problem_solution_l548_548694

-- Define the function f
def f (x : ℝ) : ℝ := 4 / (8 ^ x + 2)

-- Statement of the theorem
theorem problem_solution :
  ∑ k in Finset.range 2001, f ((k + 1 : ℝ) / 2002) = 1000 := by
  sorry

end problem_solution_l548_548694


namespace length_of_train_l548_548858

theorem length_of_train 
    (t_cross_platform : 39)
    (t_cross_pole : 18)
    (len_platform : 350)
    (speed := (L : ℝ) / t_cross_pole) :
    (L : ℝ) =
    (L * t_cross_platform / t_cross_pole = L + len_platform) :=
sorry

end length_of_train_l548_548858


namespace mutual_arrangement_of_three_circles_l548_548443

noncomputable theory

section CircleArrangement

variables {A B C D : Point} {c1 c2 c3 : Circle}

-- Given condition
def three_circles_intersect_two_points (c1 c2 c3 : Circle) : Prop :=
  ∃ (P Q : Point), (P ∈ c1 ∧ P ∈ c2 ∧ P ∈ c3) ∧ (Q ∈ c1 ∧ Q ∈ c2 ∧ Q ∈ c3) ∧ (P ≠ Q)

-- Possible mutual arrangements of the circles
def possible_arrangements (A B C D : Point) : Prop :=
  (dist A B = dist A D) ∨ (dist A B = dist C D) ∨ (dist B C = dist A D) ∨ ((dist A B = dist C D) ∧ (dist B C = dist A D))

theorem mutual_arrangement_of_three_circles :
  three_circles_intersect_two_points c1 c2 c3 → possible_arrangements A B C D :=
sorry

end CircleArrangement

end mutual_arrangement_of_three_circles_l548_548443


namespace find_x_value_l548_548557

theorem find_x_value 
  (x : ℝ)
  (h1 : 2 * ((log x / log 5) * (log x / log 6) + (log x / log 5) * (log x / log 7) + (log x / log 6) * (log x / log 7)) =
        2 * (log x / log 5) * (log x / log 6) * (log x / log 7)) :
  x = 5 :=
begin
  sorry
end

end find_x_value_l548_548557


namespace find_x_l548_548289

theorem find_x (x : ℕ) : (x % 6 = 0) ∧ (x^2 > 200) ∧ (x < 30) → (x = 18 ∨ x = 24) :=
by
  intros
  sorry

end find_x_l548_548289


namespace simplify_sqrt_sum_l548_548390

noncomputable def sqrt_72 : ℝ := Real.sqrt 72
noncomputable def sqrt_32 : ℝ := Real.sqrt 32
noncomputable def sqrt_27 : ℝ := Real.sqrt 27
noncomputable def result : ℝ := 10 * Real.sqrt 2 + 3 * Real.sqrt 3

theorem simplify_sqrt_sum :
  sqrt_72 + sqrt_32 + sqrt_27 = result :=
by
  sorry

end simplify_sqrt_sum_l548_548390


namespace a1_a2_values_exists_t_sum_first_n_terms_l548_548429

-- Define the sequence {a_n}
def a : ℕ → ℝ
| 0     := arbitrary -- a_0 is not specified and not used in conditions
| 1     := arbitrary -- a_1 is to be proved
| 2     := arbitrary -- a_2 is to be proved
| (n+3) := 2 * a (n + 2) + 2^((n + 3):ℝ) + 1

-- The conditions given
axiom a_3_eq_27 : a 3 = 27
axiom seq_cond {n : ℕ} (h : n ≥ 2): a (n + 1) = 2 * a n + 2^((n + 1) : ℝ) + 1

-- Prove values for a1 and a2
theorem a1_a2_values : a 1 = 2 ∧ a 2 = 9 :=
sorry

-- Define the sequence {b_n} and prove the existence of t such that {b_n} is arithmetic
def b (t : ℝ) (n : ℕ) : ℝ := 1 / 2^n * (a n + t)

theorem exists_t (t : ℝ) : (∀ n, b t (n + 2) - b t (n + 1) = b t (n + 1) - b t n) ↔ t = 1 :=
sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), a k

theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 1) * 2^n - n + 1 :=
sorry

end a1_a2_values_exists_t_sum_first_n_terms_l548_548429


namespace eggs_broken_l548_548362

theorem eggs_broken (brown_eggs white_eggs total_pre total_post broken_eggs : ℕ) 
  (h1 : brown_eggs = 10)
  (h2 : white_eggs = 3 * brown_eggs)
  (h3 : total_pre = brown_eggs + white_eggs)
  (h4 : total_post = 20)
  (h5 : broken_eggs = total_pre - total_post) : broken_eggs = 20 :=
by
  sorry

end eggs_broken_l548_548362


namespace fraction_juniors_study_Japanese_l548_548513

-- Define the size of the junior and senior classes
variable (J S : ℕ)

-- Condition 1: The senior class is twice the size of the junior class
axiom senior_twice_junior : S = 2 * J

-- The fraction of the seniors studying Japanese
noncomputable def fraction_seniors_study_Japanese : ℚ := 3 / 8

-- The total fraction of students in both classes that study Japanese
noncomputable def fraction_total_study_Japanese : ℚ := 1 / 3

-- Define the unknown fraction of juniors studying Japanese
variable (x : ℚ)

-- The proof problem transformed from the questions and the correct answer
theorem fraction_juniors_study_Japanese :
  (fraction_seniors_study_Japanese * ↑S + x * ↑J = fraction_total_study_Japanese * (↑J + ↑S)) → (x = 1 / 4) :=
by
  -- We use the given conditions and solve for x
  sorry

end fraction_juniors_study_Japanese_l548_548513


namespace binomial_square_value_l548_548191

theorem binomial_square_value (c : ℝ) : (∃ d : ℝ, 16 * x^2 + 40 * x + c = (4 * x + d) ^ 2) → c = 25 :=
by
  sorry

end binomial_square_value_l548_548191


namespace sum_of_first_4_terms_l548_548318

theorem sum_of_first_4_terms (a_2 a_5 : ℕ) (h1 : a_2 = 9) (h2 : a_5 = 243) : 
  let r := 3 in 
  let a := 3 in
  a * (1 - r^4) / (1 - r) = 120 :=
by
  sorry

end sum_of_first_4_terms_l548_548318


namespace find_positive_integer_n_l548_548205

noncomputable def f (n : ℕ) : ℕ :=
  let non_square_list := List.filter (λ x, (∃ k, x ≠ k * k)) (List.range (n + 100))
  non_square_list.get n

theorem find_positive_integer_n :
  (∃ n : ℕ, (∀ k : ℕ, (apply_N_times f 2013 n = 2014^2 + 1)) ∧ n = 6077248) :=
begin
  sorry
end

end find_positive_integer_n_l548_548205


namespace gerald_total_pieces_eq_672_l548_548218

def pieces_per_table : Nat := 12
def pieces_per_chair : Nat := 8
def num_tables : Nat := 24
def num_chairs : Nat := 48

def total_pieces : Nat := pieces_per_table * num_tables + pieces_per_chair * num_chairs

theorem gerald_total_pieces_eq_672 : total_pieces = 672 :=
by
  sorry

end gerald_total_pieces_eq_672_l548_548218


namespace tournament_problem_proof_l548_548663

-- Tournament setup
variables (Participants : Finset ℕ) (Defeats : ℕ → Finset ℕ) (n : ℕ)

-- Conditions
-- Each participant faces everyone else and must have a result for each match (no ties)
axiom (each_faces_all : ∀ p ∈ Participants, (Defeats p) ⊆ Participants ∧ p ∉ Defeats p)

-- Each participant writes names of players they defeated and names of all players defeated by those they defeated
axiom (defeated_transitive : ∀ p ∈ Participants, ∀ q ∈ Defeats p, (Defeats q) ⊆ Defeats p)

noncomputable def exists_complete_list_except_writer : Prop :=
∃ (p ∈ Participants), Defeats p = Participants.erase p

theorem tournament_problem_proof : exists_complete_list_except_writer Participants Defeats n :=
  sorry

end tournament_problem_proof_l548_548663


namespace range_of_a_l548_548247

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : ∀ x, (-1/2 < x) ∧ (x < 0) → ∀ x₁ x₂, x₁ < x₂ → (logForBase a (x₁^3 - a * x₁))
    < (logForBase a (x₂^3 - a * x₂))):
    3/4 ≤ a ∧ a < 1 := sorry

-- Auxiliary definition for logarithm with different bases
noncomputable def logForBase (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x / Real.log a

end range_of_a_l548_548247


namespace obtuse_triangle_side_range_l548_548650

theorem obtuse_triangle_side_range {a : ℝ} (h1 : a > 3) (h2 : (a - 3)^2 < 36) : 3 < a ∧ a < 9 := 
by
  sorry

end obtuse_triangle_side_range_l548_548650


namespace time_to_plant_2500_trees_l548_548396

-- Definitions of the conditions
def rate_of_planting : ℝ := 10 / 3 -- trees per minute
def total_trees : ℝ := 2500 -- total number of trees
def minutes_per_hour : ℝ := 60 -- minutes in an hour

-- Statement to be proved
theorem time_to_plant_2500_trees : 
  (total_trees / (rate_of_planting * minutes_per_hour)) = 12.5 := 
begin
  let trees_per_minute := rate_of_planting,
  let trees_per_hour := trees_per_minute * minutes_per_hour,
  let time_taken := total_trees / trees_per_hour,
  have h1 : trees_per_hour = 200, 
  { exact calc 
    trees_per_hour = (10 / 3) * 60 : by rw [trees_per_minute]
    ... = 200 : by ring },
  have h2 : time_taken = 12.5,
  { exact calc 
    time_taken = 2500 / 200 : by rw [h1]
    ... = 12.5 : by norm_num },
  exact h2,
end

end time_to_plant_2500_trees_l548_548396


namespace correct_prices_l548_548138

noncomputable def cost_price : ℝ := 18
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)
noncomputable def store_a_commission : ℝ := 0.20
noncomputable def store_b_flat_fee : ℝ := 5
noncomputable def store_b_commission : ℝ := 0.10
noncomputable def store_c_commission : ℝ := 0.15

def observed_price_store_a : ℝ :=
  selling_price * (1 + store_a_commission)

def observed_price_store_b : ℝ :=
  selling_price + store_b_flat_fee + selling_price * store_b_commission

def observed_price_store_c : ℝ :=
  selling_price * (1 + store_c_commission)

theorem correct_prices :
  observed_price_store_a = 25.92 ∧ 
  observed_price_store_b = 28.76 ∧ 
  observed_price_store_c = 24.84 :=
by
  -- proof to be filled in
  sorry

end correct_prices_l548_548138


namespace sphere_division_proof_l548_548438

noncomputable def sphere_division (n : ℕ) (k : ℕ) : Prop :=
  ∃ regions : List (Set Point),
    (regions.length = n) ∧
    (∀ region ∈ regions, congruent region (regions.head)) ∧
    (∀ point ∈ marked_points, ∃! region ∈ regions, point ∈ region)

theorem sphere_division_proof (points : Fin (2006) → Point) : 
  ∃! regions : List (Set Point),
    (regions.length = 2006) ∧
    (∀ region ∈ regions, congruent region (regions.head)) ∧
    (∀ p ∈ points.to_list, ∃! region ∈ regions, p ∈ region) :=
begin
  -- We assume there are 2006 distinct marked points on the sphere.
  have H1 : ∀ i j, points i ≠ points j → i ≠ j,
  { sorry },  -- Condition that the points are distinct

  -- We assume the points are not at the poles and no two lie on the same parallel.
  have H2 : ∀ i, ¬(points i).is_pole ∧ ∀ j, i ≠ j → (points i).latitude ≠ (points j).latitude,
  { sorry },  -- Condition that points are not on the same parallel or at the poles

  -- Assert that for each point, parallels are drawn, divided into equal arcs, and regions are formed.
  have H3 : ∀ i, is_parallel_through (points i),
  { sorry },

  have H4 : ∀ i j, is_equal_arc (divide_parallel 2006 ((points i).parallel)) j,
  { sorry },

  -- We assert that the final configuration of regions meet the required conditions.
  exact sphere_division 2006 (list.length marked_points),
  
end

end sphere_division_proof_l548_548438


namespace solve_equation_l548_548397

theorem solve_equation (a b c : ℝ)
  (h1 : b + c = 15 - 4a)
  (h2 : a + c = -18 - 2b)
  (h3 : a + b = 9 - 5c):
  3 * a + 3 * b + 3 * c = 18 / 17 := 
  sorry

end solve_equation_l548_548397


namespace convex_hexagon_diagonal_l548_548800

theorem convex_hexagon_diagonal (hexagon : Type) [convex_hexagon hexagon]
  (side_lengths : ∀ (s : Side hexagon), length s > 1) :
  ¬ (∃ (d : Diagonal hexagon), length d > 2) :=
sorry

end convex_hexagon_diagonal_l548_548800


namespace skateboard_total_distance_is_3720_l548_548850

noncomputable def skateboard_distance : ℕ :=
  let a1 := 10
  let d := 9
  let n := 20
  let flat_time := 10
  let a_n := a1 + (n - 1) * d
  let ramp_distance := n * (a1 + a_n) / 2
  let flat_distance := a_n * flat_time
  ramp_distance + flat_distance

theorem skateboard_total_distance_is_3720 : skateboard_distance = 3720 := 
by
  sorry

end skateboard_total_distance_is_3720_l548_548850


namespace standard_equation_of_parabola_find_stable_points_l548_548275

-- Definitions based on the problem
def parabola (p : ℝ) : set (ℝ × ℝ) := {pt | ∃ y, pt = (y^2 / (2*p), y)}

def is_focus (A : ℝ × ℝ) (p : ℝ) : Prop := A = (p / 2, 0)

def on_parabola (M : ℝ × ℝ) (p : ℝ) : Prop := M ∈ parabola p

def perpendicular_to_axis (M A : ℝ × ℝ) : Prop := M = (0, A.2)

def triangle_area (O M N : ℝ × ℝ) : ℝ := 
  0.5 * abs (O.1 * (M.2 - N.2) + M.1 * (N.2 - O.2) + N.1 * (O.2 - M.2))

-- First part of the problem
theorem standard_equation_of_parabola (p : ℝ) (M A : ℝ × ℝ) (O : ℝ × ℝ)
  (hA_focus: is_focus A p) 
  (hMA_perp: perpendicular_to_axis M A) 
  (h1: triangle_area O M A = 18) :
  (p = 6) := 
sorry

-- Definitions and conditions for the second part of the problem
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def t_value (A : ℝ × ℝ) (M N : ℝ × ℝ) : ℝ :=
  1 / (dist A M) + 1 / (dist A N)

def stable_point (A : ℝ × ℝ) (p : ℝ) (M N : ℝ × ℝ) : Prop := 
  ∀ M, M ∈ parabola p → ∃ t, t_value A M N = t

-- Second part of the problem
theorem find_stable_points (p : ℝ) (A : ℝ × ℝ):
  (stable_point A p M N) ↔ (A = (3, 0)) :=
sorry

end standard_equation_of_parabola_find_stable_points_l548_548275


namespace plywood_cut_difference_l548_548120

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l548_548120


namespace plywood_perimeter_difference_l548_548087

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548087


namespace a_mod_p_l548_548690

theorem a_mod_p (p : ℕ) (h_prime : p = 101) : 
  (∑ k in Finset.range 11, (1 : ℚ) / Nat.choose p k) * Nat.factorial p ≡ 96 [MOD p] :=
by
  sorry

end a_mod_p_l548_548690


namespace sum_of_valid_b_values_l548_548946

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l548_548946


namespace range_of_x_l548_548238

-- Let p and q be propositions regarding the range of x:
def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Main theorem statement
theorem range_of_x 
  (h1 : ∀ x : ℝ, p x ∨ q x)
  (h2 : ∀ x : ℝ, ¬ q x) :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 4) := by
  sorry

end range_of_x_l548_548238


namespace harry_carries_buckets_rounds_l548_548571

noncomputable def george_rate := 2
noncomputable def total_buckets := 110
noncomputable def total_rounds := 22
noncomputable def harry_buckets_each_round := 3

theorem harry_carries_buckets_rounds :
  (george_rate * total_rounds + harry_buckets_each_round * total_rounds = total_buckets) :=
by sorry

end harry_carries_buckets_rounds_l548_548571


namespace area_of_similar_rectangle_l548_548183

theorem area_of_similar_rectangle:
  ∀ (R1 : ℝ → ℝ → Prop) (R2 : ℝ → ℝ → Prop),
  (∀ a b, R1 a b → a = 3 ∧ a * b = 18) →
  (∀ a b c d, R1 a b → R2 c d → c / d = a / b) →
  (∀ a b, R2 a b → a^2 + b^2 = 400) →
  ∃ areaR2, (∀ a b, R2 a b → a * b = areaR2) ∧ areaR2 = 160 :=
by
  intros R1 R2 hR1 h_similar h_diagonal
  use 160
  sorry

end area_of_similar_rectangle_l548_548183


namespace common_ratio_arith_geo_sequence_l548_548430

theorem common_ratio_arith_geo_sequence (a : ℕ → ℝ) (d : ℝ) (q : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1 + 2) * q = a 5 + 5) 
  (h_geo' : (a 5 + 5) * q = a 9 + 8) :
  q = 1 :=
by
  sorry

end common_ratio_arith_geo_sequence_l548_548430


namespace g_neg_2_eq_3_l548_548285

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem g_neg_2_eq_3 : g (-2) = 3 :=
by
  sorry

end g_neg_2_eq_3_l548_548285


namespace probability_even_card_l548_548765

theorem probability_even_card :
  let cards := [1921, 1994, 1935, 1949, 1978, 1980] in
  let even_card_count := cards.count (λ n => n % 2 = 0) in
  even_card_count.toReal / cards.length.toReal = 1 / 2 :=
by
  simp [cards, even_card_count]
  sorry

end probability_even_card_l548_548765


namespace students_can_be_helped_on_fourth_day_l548_548493

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end students_can_be_helped_on_fourth_day_l548_548493


namespace rational_operation_example_l548_548990

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end rational_operation_example_l548_548990


namespace winning_team_possible_scores_l548_548768

theorem winning_team_possible_scores :
  ∀ (K : ℕ), (15 ≤ K ∧ K ≤ 27) → (∃ S : finset ℕ, (∀ x ∈ S, 15 ≤ x ∧ x ≤ 27) ∧ S.card = 13) :=
by
  sorry

end winning_team_possible_scores_l548_548768


namespace plywood_perimeter_difference_l548_548113

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548113


namespace sum_of_b_for_rational_roots_l548_548948

theorem sum_of_b_for_rational_roots (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 4) (Δ : Nat) :
  (Δ = 49 - 12 * b ∧ (∃ k : Nat, Δ = k * k)) → b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 → 
  ∑ i in Finset.filter (λ b, (∃ (k : ℕ), 49 - 12 * b = k^2)) 
  (Finset.range' 1 5), b = 6 :=
by sorry

end sum_of_b_for_rational_roots_l548_548948


namespace plywood_perimeter_difference_l548_548114

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548114


namespace perimeters_positive_difference_l548_548095

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l548_548095


namespace laborer_savings_l548_548736

theorem laborer_savings
  (monthly_expenditure_first6 : ℕ := 70)
  (monthly_expenditure_next4 : ℕ := 60)
  (monthly_income : ℕ := 69)
  (expenditure_first6 := 6 * monthly_expenditure_first6)
  (income_first6 := 6 * monthly_income)
  (debt : ℕ := expenditure_first6 - income_first6)
  (expenditure_next4 := 4 * monthly_expenditure_next4)
  (income_next4 := 4 * monthly_income)
  (savings : ℕ := income_next4 - (expenditure_next4 + debt)) :
  savings = 30 := 
by
  sorry

end laborer_savings_l548_548736


namespace player2_wins_polynomial_game_l548_548770

theorem player2_wins_polynomial_game :
  ∀ (P : ℝ[X]) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ),
    P = X^10 + a₀ * X^9 + a₁ * X^8 + a₂ * X^7 + a₃ * X^6 + a₄ * X^5 + a₅ * X^4 + a₆ * X^3 + a₇ * X^2 + a₈ * X + 1 →
    (∀ b : ℝ, ∃ c : ℝ, ((∀ x : ℝ, eval x (X^10 + c * X^9 + a₁ * X^8 + a₂ * X^7 + a₃ * X^6 + a₄ * X^5 + a₅ * X^4 + a₆ * X^3 + a₇ * X^2 + a₈ * X + 1) = 0) → (eval x P = 0))) :=
  sorry

end player2_wins_polynomial_game_l548_548770


namespace find_v2008_l548_548356

def sequence_v (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else let k := (nat.sqrt (1 + 8 * (n - 1)) - 1) / 2 + 1 in
       let base := k * (3 * k - 1) / 2 in
       let group_start_value := 3 * k - 2 in
       group_start_value + 3 * (n - base - 1) 

theorem find_v2008 : sequence_v 2008 = 352 :=
by sorry

end find_v2008_l548_548356


namespace basis_sets_l548_548244

variables (a b c : Type*) [Add a] [Add b] [Add c] [NonCoplanar a b c]

theorem basis_sets (a b c : Vector) (h_non_coplanar : non_coplanar a b c) :
  (linear_independent ℝ ![(a + b), (a - 2 * b), c]) ∧
  (linear_independent ℝ ![a, (2 * b), (b - c)]) :=
sorry

end basis_sets_l548_548244


namespace find_c_value_l548_548558

theorem find_c_value (c : ℝ)
  (h : 4 * (3.6 * 0.48 * c / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) :
  c = 2.5 :=
by sorry

end find_c_value_l548_548558


namespace num_marbles_in_first_set_l548_548217

theorem num_marbles_in_first_set
  (x : ℕ)  -- number of marbles in the first set
  (h1 : 0.10 * x = 5)  -- 10% of the first set marbles are broken
  (h2 : 0.20 * 60 = 12)  -- 20% of the second set's marbles are broken
  (h3 : 17 - 12 = 5)  -- in total, 17 marbles are broken
  : x = 50 := 
by
  sorry

end num_marbles_in_first_set_l548_548217


namespace correct_calculation_l548_548792

-- Definitions of the conditions
def condition_A (a : ℝ) : Prop := a^2 + a^2 = a^4
def condition_B (a : ℝ) : Prop := 3 * a^2 + 2 * a^2 = 5 * a^2
def condition_C (a : ℝ) : Prop := a^4 - a^2 = a^2
def condition_D (a : ℝ) : Prop := 3 * a^2 - 2 * a^2 = 1

-- The theorem statement
theorem correct_calculation (a : ℝ) : condition_B a := by 
sorry

end correct_calculation_l548_548792


namespace inequality_transform_l548_548224

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end inequality_transform_l548_548224


namespace sum_of_possible_values_of_intersection_points_l548_548564

/-- For a given set of four distinct lines in a plane, prove that the 
    sum of all possible values of the number of distinct intersection points 
    of pairs of lines is equal to 19. -/
theorem sum_of_possible_values_of_intersection_points :
  ∀ unique_pairs_of_lines (lines : Finset (Fin 4)
  (h_distinct : lines.card = 4),
  let possible_intersections := {0, 1, 3, 4, 5, 6} in
  (possible_intersections.sum id) = 19 
:= by
  sorry -- proof goes here

end sum_of_possible_values_of_intersection_points_l548_548564


namespace clean_per_hour_l548_548886

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end clean_per_hour_l548_548886


namespace draw_two_green_marbles_probability_l548_548829

theorem draw_two_green_marbles_probability :
  let red := 5
  let green := 3
  let white := 7
  let total := red + green + white
  (green / total) * ((green - 1) / (total - 1)) = 1 / 35 :=
by
  sorry

end draw_two_green_marbles_probability_l548_548829


namespace transmission_time_l548_548544

def blocks : ℕ := 100
def chunks_per_block : ℕ := 600
def transmission_rate : ℕ := 150
def seconds_per_minute : ℕ := 60

theorem transmission_time :
  let total_chunks := blocks * chunks_per_block,
      total_seconds := total_chunks / transmission_rate,
      total_minutes := total_seconds / seconds_per_minute
  in total_minutes = 7 := by 
  -- Define all the variables and their values
  let total_chunks := blocks * chunks_per_block
  let total_seconds := total_chunks / transmission_rate
  let total_minutes := total_seconds / seconds_per_minute
  -- Insert the proof here
  sorry

end transmission_time_l548_548544


namespace perimeter_difference_l548_548065

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l548_548065


namespace sum_of_b_values_l548_548957

-- Definitions based on conditions from the problem statement
def quadratic_equation (b : ℕ) : Prop := ∃ x : ℚ, 3 * x^2 + 7 * x + b = 0

def has_rational_roots (b : ℕ) : Prop :=
  ∃ (m : ℤ), ∃ (k : ℤ), m^2 = 49 - 12 * b

def possible_b_values (b : ℕ) : Prop := b > 0 ∧ has_rational_roots b

-- The statement of the proof problem
theorem sum_of_b_values :
  (∑ b in { b | possible_b_values b }.to_finset, b) = 6 :=
sorry

end sum_of_b_values_l548_548957


namespace area_intersection_of_reflected_triangles_l548_548676

noncomputable def heron_formula (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in 
real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_intersection_of_reflected_triangles :
  ∀ (A B C : ℝ × ℝ) (H : ℝ × ℝ),
  dist A B = 8 → dist B C = 15 → dist A C = 17 →
  (H = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  let A' := (2 * H.1 - A.1, 2 * H.2 - A.2) in
  let B' := (2 * H.1 - B.1, 2 * H.2 - B.2) in
  let C' := (2 * H.1 - C.1, 2 * H.2 - C.2) in
  heron_formula 8 15 17 = 60 →
  ∃ area : ℝ, area = 60 := 
by
  intros A B C H hAB hBC hAC hH hArea
  sorry

end area_intersection_of_reflected_triangles_l548_548676


namespace smallest_n_satisfying_l548_548632

theorem smallest_n_satisfying (n : ℕ) (h₁ : (n + 1)! + (n + 3)! = n! * 482) : n = 7 :=
by
  sorry

end smallest_n_satisfying_l548_548632


namespace part_a_example_part_b_example_l548_548572

-- Define the conditions
variables {a b : ℝ} (ha : a < b)

-- (a) Function that oscillates between a and b
def f_oscillates_between_bounds (f : ℝ → ℝ) := 
∀ x, a ≤ f x ∧ f x ≤ b

-- (b) Function with unbounded increasing oscillations
def g_unbounded_increasing_oscillations (g : ℝ → ℝ) := 
∀ x, ∃ N > x, ∀ y > N, g y < y ∧ g y > -y

-- Part (a): Oscillates between finite bounds
theorem part_a_example : f_oscillates_between_bounds (λ x, a + (b - a) * (Real.sin x)^2) :=
sorry

-- Part (b): Unbounded increasing oscillations
theorem part_b_example : g_unbounded_increasing_oscillations (λ x, x * Real.sin x) :=
sorry

end part_a_example_part_b_example_l548_548572


namespace possible_p_values_l548_548895

noncomputable def polynomial (a r : ℤ) : Polynomial ℤ :=
  (Polynomial.X - Polynomial.C a) *
  (Polynomial.X - Polynomial.C (a / 3 + r)) *
  (Polynomial.X - Polynomial.C (a / 3 - r)) *
  (Polynomial.X - Polynomial.C (a / 3))

variable (a r : ℤ)

-- Given conditions
def conditions : Prop :=
  ∃ (a r : ℤ), 
    -- Integer zero, sum of other three zeros
    a = 1006 ∧
    -- Zeros are positive and distinct
    (3 ∣ a) ∧ 
    a > 0 ∧ (a / 3) > 0 ∧ (a / 3) + r > 0 ∧ (a / 3) - r > 0 ∧
    polynomial a r == Polynomial.eval 0 (polynomial a r)

-- Theorem stating the number of possible values for p
theorem possible_p_values : conditions a r → 
  (∃ p : ℤ, p = 112446) :=
by
  sorry

end possible_p_values_l548_548895


namespace dihedral_angle_measurement_l548_548582

def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem dihedral_angle_measurement
  (A B C D : ℝ × ℝ × ℝ)
  (h1 : (B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2 = 16)  -- AB = 4
  (h2 : (C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.3 - A.3)^2 = 36)  -- AC = 6
  (h3 : (D.1 - B.1)^2 + (D.2 - B.2)^2 + (D.3 - B.3)^2 = 64)  -- BD = 8
  (h4 : (D.1 - C.1)^2 + (D.2 - C.2)^2 + (D.3 - C.3)^2 = 68)  -- CD = 2√17
  (h5 : is_perpendicular (C.1 - A.1, C.2 - A.2, C.3 - A.3) (B.1 - A.1, B.2 - A.2, B.3 - A.3))
  (h6 : is_perpendicular (B.1 - A.1, B.2 - A.2, B.3 - A.3) (D.1 - B.1, D.2 - B.2, D.3 - B.3))
  : ∃ θ : ℝ, θ = pi / 3 :=     -- dihedral angle = 60°
sorry

end dihedral_angle_measurement_l548_548582


namespace problem1_problem2_l548_548389

theorem problem1 :
  (2 + 4 / 5)^0 + (2^(-2)) * (2 + 1 / 4)^(-1 / 2) - (8 / 27)^(1 / 3) = 1 / 2 := 
by
  sorry

theorem problem2 :
  2 * (log 2)^2 + log (sqrt 2) * log 5 + sqrt((log (sqrt 2))^2 - log 2 + 1) = 1 := 
by
  sorry

end problem1_problem2_l548_548389


namespace exists_k_for_inequality_l548_548584

noncomputable def C : ℕ := sorry -- C is a positive integer > 0
def a : ℕ → ℝ := sorry -- a sequence of positive real numbers

axiom C_pos : 0 < C
axiom a_pos : ∀ n : ℕ, 0 < a n
axiom recurrence_relation : ∀ n : ℕ, a (n + 1) = n / a n + C

theorem exists_k_for_inequality :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a (n + 2) > a n :=
  sorry

end exists_k_for_inequality_l548_548584


namespace jimmy_cards_left_l548_548330

-- Step-by-step conditions
def total_initial_cards := 23
def cards_given_to_bob := 4

def cards_jimmy_after_bob : ℕ := total_initial_cards - cards_given_to_bob

def cards_given_by_bob (total : ℕ) := total / 3  -- Approximating (integer division)
def cards_bob_after_sarah : ℕ := cards_given_to_bob - cards_given_by_bob(cards_given_to_bob)

def cards_given_to_mary := 2 * cards_given_to_bob
def cards_jimmy_after_mary : ℕ := cards_jimmy_after_bob - cards_given_to_mary

def cards_given_by_sarah := 0  -- 25% of her approximate whole cards is zero

-- Proof of the final cards left with Jimmy
theorem jimmy_cards_left : 
  cards_jimmy_after_mary = 11 :=
by 
  -- Using the given steps to achieve the proof.
  have h1 : cards_jimmy_after_bob = 19 := rfl
  have h2 : cards_given_to_mary = 8 := rfl
  have h3 : cards_jimmy_after_mary = 11 := rfl
  exact h3


end jimmy_cards_left_l548_548330


namespace plywood_cut_difference_l548_548075

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end plywood_cut_difference_l548_548075


namespace sin2alpha_eq_neg24_25_l548_548718

-- Define point A and point B on the unit circle
def PointA := (1 : ℝ, 0 : ℝ)
def PointB := (-3/5 : ℝ, 4/5 : ℝ)

-- Define the angle α such that AOB = α
variable (α : ℝ)

-- Define the trigonometric functions at α
def sin_alpha : ℝ := 4 / 5
def cos_alpha : ℝ := -3 / 5

-- Define the identity for sin(2α)
def sin_double_angle (sin_alpha cos_alpha : ℝ) : ℝ := 
  2 * sin_alpha * cos_alpha

-- The main theorem to prove
theorem sin2alpha_eq_neg24_25 : 
  sin_double_angle (4 / 5) (-3 / 5) = -24 / 25 := 
by 
  sorry

end sin2alpha_eq_neg24_25_l548_548718


namespace planar_section_area_solids_volume_ratio_l548_548740

section CubeIntersection

variable {Point : Type} [MetricSpace Point] [InnerProductSpace ℝ Point]

/-- Defining points A, B, E, P, and R in a unit cube -/
def A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 1 : ℝ, 0 : ℝ)
def E := (0 : ℝ, 0 : ℝ, 1 : ℝ)
def P := (0 : ℝ, 0 : ℝ, 0.5 : ℝ)
def R := (0.5 : ℝ, 0.5 : ℝ, 1 : ℝ)

/-- The area of the planar section intersecting the cube at points P, B, and R is 9/8 -/
theorem planar_section_area :
  let area := trapezoid_area (P, B, R) in
  area = 9 / 8 := sorry

/-- The volume ratio of the two solids formed by the plane cutting through the cube is 7:17 -/
theorem solids_volume_ratio :
  let ratio := volume_ratio (P, B, R) in
  ratio = 7 / 24 / (17 / 24) := sorry

end CubeIntersection

end planar_section_area_solids_volume_ratio_l548_548740


namespace plywood_cut_perimeter_difference_l548_548102

theorem plywood_cut_perimeter_difference :
  (∃ (l w : ℕ), (l * w = 54) ∧ (9 % w = 0) ∧ (6 % l = 0) ∧ (6 / l) * (9 / w) = 6) →
  10 =
  let p := λ l w, 2 * (l + w) in
  let perimeters := [
    p 1 9,
    p 1 6,
    p 2 3,
    p 3 2
  ]
  in (list.max precedence perimeters - list.min precedence perimeters) :=
begin
  sorry
end

end plywood_cut_perimeter_difference_l548_548102


namespace optionC_has_min_4_l548_548003

noncomputable def funcA (x : ℝ) : ℝ := x^2 + 2*x + 4
noncomputable def funcB (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)
noncomputable def funcC (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def funcD (x : ℝ) : ℝ := log x + 4 / log x

theorem optionC_has_min_4 (x : ℝ) : (∀ y, (y = funcA x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcB x) → y ≠ 4) ∧ 
                                   (∀ y, (y = funcD x) → y ≠ 4) ∧
                                   (∃ t, (t = 1) ∧ (funcC t = 4)) := 
by {
  sorry
}

end optionC_has_min_4_l548_548003


namespace water_collection_difference_l548_548857

theorem water_collection_difference
  (tank_capacity : ℕ)
  (initial_fraction : ℚ)
  (first_day_collection : ℕ)
  (third_day_extra_collection : ℕ)
  (initial_water : ℚ := initial_fraction * tank_capacity)
  (first_day_total : ℕ := initial_water + first_day_collection)
  (second_day_collection : ℕ := tank_capacity - first_day_total)
  (water_difference : ℕ := second_day_collection - first_day_collection)
  (h_capacity : tank_capacity = 100)
  (h_initial_fraction : initial_fraction = 2 / 5)
  (h_first_day_collection : first_day_collection = 15)
  (h_third_day_extra_collection : third_day_extra_collection = 25)
  : water_difference = 30 := by
  sorry

end water_collection_difference_l548_548857


namespace angle_ABD_not_acute_l548_548369

theorem angle_ABD_not_acute (A B C D : Type)
  [Triangle A B C] [LineSegment AC CD CB]
  (h1 : ∃ C_ext : Type, D ∈ AC ∧ CD = CB ∧ AC < AC_ext)
  (h2 : ∃ α β γ : ℝ, α + β + γ = 180 ∧ α < β ∧ β < γ):

  ∃ θ : ℝ, ∡ A B D = θ ∧ θ ≥ 90 := 
sorry

end angle_ABD_not_acute_l548_548369


namespace sqrt_product_l548_548892

open Real

theorem sqrt_product :
  sqrt 54 * sqrt 48 * sqrt 6 = 72 * sqrt 3 := by
  sorry

end sqrt_product_l548_548892


namespace problem1_problem2_l548_548221

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

-- Proof Problem 1
theorem problem1 (x : ℝ) (h : f x > 2) : x < -2 := sorry

-- Proof Problem 2
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ k * x + 1) : k ≤ -1 := sorry

end problem1_problem2_l548_548221


namespace Sn_proof_l548_548706

noncomputable def f (x : ℝ) : ℝ :=
  1 / 2 + Real.log2 (x / (1 - x))

noncomputable def S_n (n : ℕ) [h : fact (n ≥ 2)] : ℝ :=
  ∑ i in Finset.range (n - 1), f (i / n)

theorem Sn_proof (n : ℕ) [h : fact (n ≥ 2)] : S_n n = (n - 1) / 2 :=
by sorry

end Sn_proof_l548_548706


namespace candy_initial_amount_l548_548964

namespace CandyProblem

variable (initial_candy given_candy left_candy : ℕ)

theorem candy_initial_amount (h1 : given_candy = 10) (h2 : left_candy = 68) (h3 : left_candy = initial_candy - given_candy) : initial_candy = 78 := 
  sorry
end CandyProblem

end candy_initial_amount_l548_548964


namespace bounded_seq_inequality_l548_548814

-- Definitions and conditions
def bounded_seq (x : ℕ → ℝ) := ∃ M > 0, ∀ n, |x n| ≤ M

def sequence := ℕ → ℝ

-- Proof statement
theorem bounded_seq_inequality 
  (x : sequence) 
  (a : ℝ) 
  (h1 : bounded_seq x)
  (h2 : ∀ i j : ℕ, i ≠ j → |x i - x j| * |i - j| ^ a ≥ 1) :
  true 
:= sorry

end bounded_seq_inequality_l548_548814


namespace sum_m_n_zero_l548_548577

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l548_548577


namespace sugar_added_to_solution_l548_548126

theorem sugar_added_to_solution
  (original_volume : ℚ)
  (water_percentage : ℚ)
  (kola_percentage : ℚ)
  (added_water : ℚ)
  (added_kola : ℚ)
  (new_sugar_percentage : ℚ)
  (orig_water : ℚ := original_volume * water_percentage)
  (orig_kola : ℚ := original_volume * kola_percentage)
  (orig_sugar : ℚ := original_volume - orig_water - orig_kola)
  (new_total_volume : ℚ := original_volume + added_water + added_kola) :
    ∀ x : ℚ, new_sugar_percentage / 100 * (new_total_volume + x) = orig_sugar + x → 
       x ≈ 3.2 :=
by
  intro x
  sorry

end sugar_added_to_solution_l548_548126


namespace gain_percent_is_correct_l548_548042

noncomputable def gain_percent (C S : ℚ) : ℚ :=
let total_selling_price := 200 * S in
let gain := 50 * S in
let total_cost_price := 200 * C in
(gain / total_cost_price) * 100

theorem gain_percent_is_correct (C S : ℚ) (h : C = (3 / 4) * S) : gain_percent C S = 33.33 :=
by
  have total_selling_price : ℚ := 200 * S
  have gain : ℚ := 50 * S
  have total_cost_price : ℚ := 200 * C
  have h_gain : gain = total_selling_price - total_cost_price, sorry
  have h_cost_price : total_cost_price = 150 * S, sorry
  have h_gain_percent : gain_percent C S = (50 * S / (150 * S)) * 100, sorry
  have h_result : (50 * S / (150 * S)) * 100 = 33.33, sorry
  exact h_result

end gain_percent_is_correct_l548_548042


namespace solve_for_m_l548_548730

theorem solve_for_m (m : ℝ) (h : (m - 5)^3 = (1/27)^(-1)) : m = 8 :=
by
  sorry

end solve_for_m_l548_548730


namespace minimum_area_is_correct_l548_548872

noncomputable def minimum_rectangle_area (length width : ℝ) : ℝ :=
  if length_min < length and length_max > length and width_min < width and width_max > width then
    length_min * width_min
  else
    0

theorem minimum_area_is_correct :
  ∀ (length width : ℝ), 
  length = 2 → 
  width = 3 → 
  ∃ length_min width_min : ℝ, 
  length_min = 1.5 ∧ 
  width_min = 2.5 → 
  (minimum_rectangle_area length width = 3.75) :=
begin
  intros length width h_length h_width,
  use [1.5, 2.5],
  rw [h_length, h_width],
  sorry
end

end minimum_area_is_correct_l548_548872


namespace quadratic_negative_root_exists_l548_548840

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c

theorem quadratic_negative_root_exists 
  {a b c : ℝ} 
  (h_distinct_roots : ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_polynomial a b c x₁ = 0 ∧ quadratic_polynomial a b c x₂ = 0)
  (h_inequality : ∀ x y : ℝ, quadratic_polynomial a b c (x^2 + y^2) ≥ quadratic_polynomial a b c (2 * x * y)) :
  ∃ x, quadratic_polynomial a b c x = 0 ∧ x < 0 :=
sorry

end quadratic_negative_root_exists_l548_548840


namespace consecutive_probability_sum_divisible_by_3_probability_l548_548324

-- Definition of the boxes and the sets of numbers
def balls_in_box : set ℕ := {1, 2, 3, 4}
def outcomes : set (ℕ × ℕ) := set.product balls_in_box balls_in_box

-- Question (I): Probability of drawing two balls with consecutive numbers
def consecutive_pairs : set (ℕ × ℕ) := { (1,2), (2,1), (2,3), (3,2), (3,4), (4,3) }

theorem consecutive_probability :
  (consecutive_pairs.card : ℚ) / (outcomes.card : ℚ) = 3 / 8 := sorry

-- Question (II): Probability that the sum of numbers on two drawn balls is divisible by 3
def sum_divisible_by_3_pairs : set (ℕ × ℕ) := { (1,2), (2,1), (2,4), (3,3), (4,2) }

theorem sum_divisible_by_3_probability :
  (sum_divisible_by_3_pairs.card : ℚ) / (outcomes.card : ℚ) = 5 / 16 := sorry

end consecutive_probability_sum_divisible_by_3_probability_l548_548324


namespace roots_equal_of_quadratic_eq_zero_l548_548566

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end roots_equal_of_quadratic_eq_zero_l548_548566


namespace correct_pairings_l548_548681

-- Define the employees
inductive Employee
| Jia
| Yi
| Bing
deriving DecidableEq

-- Define the wives
inductive Wife
| A
| B
| C
deriving DecidableEq

-- Define the friendship and age relationships
def isGoodFriend (x y : Employee) : Prop :=
  -- A's husband is Yi's good friend.
  (x = Employee.Jia ∧ y = Employee.Yi) ∨
  (x = Employee.Yi ∧ y = Employee.Jia)

def isYoungest (x : Employee) : Prop :=
  -- Specify that Jia is the youngest
  x = Employee.Jia

def isOlder (x y : Employee) : Prop :=
  -- Bing is older than C's husband.
  x = Employee.Bing ∧ y ≠ Employee.Bing

-- The pairings of husbands and wives: Jia—A, Yi—C, Bing—B.
def pairings (x : Employee) : Wife :=
  match x with
  | Employee.Jia => Wife.A
  | Employee.Yi => Wife.C
  | Employee.Bing => Wife.B

-- Proving the given pairings fit the conditions.
theorem correct_pairings : 
  ∀ (x : Employee), 
  isGoodFriend (Employee.Jia) (Employee.Yi) ∧ 
  isYoungest Employee.Jia ∧ 
  (isOlder Employee.Bing Employee.Jia ∨ isOlder Employee.Bing Employee.Yi) → 
  pairings x = match x with
               | Employee.Jia => Wife.A
               | Employee.Yi => Wife.C
               | Employee.Bing => Wife.B :=
by
  sorry

end correct_pairings_l548_548681


namespace pie_split_l548_548171

theorem pie_split (initial_pie : ℚ) (number_of_people : ℕ) (amount_taken_by_each : ℚ) 
  (h1 : initial_pie = 5/6) (h2 : number_of_people = 4) : amount_taken_by_each = 5/24 :=
by
  sorry

end pie_split_l548_548171


namespace inequality_transform_l548_548225

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end inequality_transform_l548_548225


namespace seashells_found_l548_548186

theorem seashells_found (C B : ℤ) (h1 : 9 * B = 7 * C) (h2 : B = C - 12) : C = 54 :=
by
  sorry

end seashells_found_l548_548186


namespace parabola_vertex_l548_548759

theorem parabola_vertex : 
  ∀ (x y : ℝ), y = -2 * x^2 + 3 → (x, y) = (0, 3) :=
begin
  intros x y h,
  have hx : x = 0, sorry,
  rw hx at h,
  have hy : y = 3, sorry,
  rw hy,
end

end parabola_vertex_l548_548759


namespace sum_possible_b_quad_eq_rational_roots_l548_548959

theorem sum_possible_b_quad_eq_rational_roots :
  (∑ b in { b : ℕ | b > 0 ∧ (∃ k : ℕ, 7^2 - 4 * 3 * b = k^2) ∧ b ≤ 4 }, b) = 6 :=
by
  sorry

end sum_possible_b_quad_eq_rational_roots_l548_548959


namespace plywood_perimeter_difference_l548_548110

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end plywood_perimeter_difference_l548_548110


namespace option_c_has_minimum_value_4_l548_548020

theorem option_c_has_minimum_value_4 :
  (∀ x : ℝ, x^2 + 2 * x + 4 ≥ 3) ∧
  (∀ x : ℝ, |sin x| + 4 / |sin x| > 4) ∧
  (∀ x : ℝ, 2^x + 2^(2 - x) ≥ 4) ∧
  (∀ x : ℝ, ln x + 4 / ln x < 4) →
  (∀ x : ℝ, 2^x + 2^(2 - x) = 4 → x = 1) :=
by sorry

end option_c_has_minimum_value_4_l548_548020


namespace DM_plus_DN_l548_548056

-- Definitions for the objects and conditions in the problem
variables (A B C D M N : Point)
variable (angle_D_obtuse : obtuse (angle D))
variable (is_parallelogram : parallelogram A B C D)
variable (M_is_foot : foot M D (line A B))
variable (N_is_foot : foot N D (line B C))
variable (DB_eq_50 : dist D B = 50)
variable (DC_eq_50 : dist D C = 50)
variable (DA_eq_60 : dist D A = 60)

-- The theorem to be proved
theorem DM_plus_DN : dist D M + dist D N = 88 :=
sorry

end DM_plus_DN_l548_548056


namespace triangles_intersect_l548_548656

open Classical

noncomputable theory

-- Definitions used in the conditions:
def circle_radius : ℝ := 1

def triangle1_area (t1 : Set Point) : ℝ := sorry -- Define as needed to find area of t1
def triangle2_area (t2 : Set Point) : ℝ := sorry -- Define as needed to find area of t2

def inside_circle (t : Set Point) : Prop := sorry -- Define as needed for t_inside_circle

-- Proof Statement
theorem triangles_intersect (t1 t2 : Set Point) 
  (h1 : inside_circle t1) (h2 : inside_circle t2) 
  (h_area1 : triangle1_area t1 > 1) (h_area2 : triangle2_area t2 > 1) 
  (h_radius : circle_radius = 1) : 
  (t1 ∩ t2) ≠ ∅ :=
sorry

end triangles_intersect_l548_548656


namespace percent_yield_hydrogen_gas_l548_548903

theorem percent_yield_hydrogen_gas
  (moles_fe : ℝ) (moles_h2so4 : ℝ) (actual_yield_h2 : ℝ) (theoretical_yield_h2 : ℝ) :
  moles_fe = 3 →
  moles_h2so4 = 4 →
  actual_yield_h2 = 1 →
  theoretical_yield_h2 = moles_fe →
  (actual_yield_h2 / theoretical_yield_h2) * 100 = 33.33 :=
by
  intros h_moles_fe h_moles_h2so4 h_actual_yield_h2 h_theoretical_yield_h2
  sorry

end percent_yield_hydrogen_gas_l548_548903


namespace johns_groceries_cost_l548_548332

noncomputable def calculate_total_cost : ℝ := 
  let bananas_cost := 6 * 2
  let bread_cost := 2 * 3
  let butter_cost := 3 * 5
  let cereal_cost := 4 * (6 - 0.25 * 6)
  let subtotal := bananas_cost + bread_cost + butter_cost + cereal_cost
  if subtotal >= 50 then
    subtotal - 10
  else
    subtotal

-- The statement to prove
theorem johns_groceries_cost : calculate_total_cost = 41 := by
  sorry

end johns_groceries_cost_l548_548332


namespace youngest_child_age_possible_l548_548834

theorem youngest_child_age_possible 
  (total_bill : ℝ) (mother_charge : ℝ) 
  (yearly_charge_per_child : ℝ) (minimum_charge_per_child : ℝ) 
  (num_children : ℤ) (children_total_bill : ℝ)
  (total_years : ℤ)
  (youngest_possible_age : ℤ) :
  total_bill = 15.30 →
  mother_charge = 6 →
  yearly_charge_per_child = 0.60 →
  minimum_charge_per_child = 0.90 →
  num_children = 3 →
  children_total_bill = total_bill - mother_charge →
  children_total_bill - num_children * minimum_charge_per_child = total_years * yearly_charge_per_child →
  total_years = 11 →
  youngest_possible_age = 1 :=
sorry

end youngest_child_age_possible_l548_548834


namespace max_polyominoes_l548_548373

theorem max_polyominoes (grid_size : ℕ) (polyomino_cells : ℕ) : 
  grid_size = 8 → polyomino_cells = 4 → 
  ∃ max_polyominoes : ℕ, max_polyominoes = 7 :=
by
  intros grid_size_eq polyomino_cells_eq
  use 7
  sorry

end max_polyominoes_l548_548373


namespace point_on_inverse_proportion_l548_548648

theorem point_on_inverse_proportion (k : ℝ) (hk : k ≠ 0) :
  (2 * 3 = k) → (1 * 6 = k) :=
by
  intro h
  sorry

end point_on_inverse_proportion_l548_548648


namespace jack_hands_in_money_l548_548328

-- Define the quantities of each type of bill and coin Jack has.
def num_100_bills : Nat := 2
def num_50_bills : Nat := 1
def num_20_bills : Nat := 5
def num_10_bills : Nat := 3
def num_5_bills : Nat := 7
def num_1_bills : Nat := 27

def num_quarters : Nat := 42
def num_dimes : Nat := 19
def num_nickels : Nat := 36
def num_pennies : Nat := 47

-- Define the dollar values of each type of bill and coin.
def value_100_bill : Float := 100
def value_50_bill : Float := 50
def value_20_bill : Float := 20
def value_10_bill : Float := 10
def value_5_bill : Float := 5
def value_1_bill : Float := 1

def value_quarter : Float := 0.25
def value_dime : Float := 0.10
def value_nickel : Float := 0.05
def value_penny : Float := 0.01

-- Define the total amount of money in notes and coins.
def total_notes : Float :=
  num_100_bills * value_100_bill + 
  num_50_bills * value_50_bill + 
  num_20_bills * value_20_bill + 
  num_10_bills * value_10_bill + 
  num_5_bills * value_5_bill + 
  num_1_bills * value_1_bill

def total_coins : Float :=
  num_quarters * value_quarter + 
  num_dimes * value_dime + 
  num_nickels * value_nickel + 
  num_pennies * value_penny

-- Define the amount to leave in the till and the correct answer.
def amount_to_leave_in_till : Float := 300
def correct_answer : Float := 142

-- The theorem to prove: Jack will hand in $142 under the given conditions.
theorem jack_hands_in_money : total_notes - amount_to_leave_in_till = correct_answer := by
  sorry

end jack_hands_in_money_l548_548328


namespace odd_function_symmetric_l548_548294

theorem odd_function_symmetric {f : ℝ → ℝ} (h : ∀ x, f (-x) = -f x) (a : ℝ) : (a, f a) → (-a, -f a) := 
sorry

end odd_function_symmetric_l548_548294


namespace collinear_B_C_E_cyclic_A_B_D_E_l548_548471

variables {A B C D E : Type}
-- Conditions
variable (triangle_ABC : Type) 
variable [IsExcenter D triangle_ABC]
variable (A_reflection_over_DC_to_E : Reflection A D C E)

-- Definition and properties that need to be proven
theorem collinear_B_C_E : Collinear B C E := sorry
theorem cyclic_A_B_D_E : Cyclic A B D E := sorry

end collinear_B_C_E_cyclic_A_B_D_E_l548_548471


namespace sqrt_comparison_l548_548194

theorem sqrt_comparison : sqrt 10 - sqrt 3 > sqrt 14 - sqrt 7 := 
sorry

end sqrt_comparison_l548_548194


namespace plywood_perimeter_difference_l548_548088

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548088


namespace chinese_chess_paths_count_l548_548654

theorem chinese_chess_paths_count :
  let rows := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
  let cols := [1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let A := (4, 5) in
  let B_row := 10 in
  let a_i (i : ℕ) := if i = 4 ∨ i = 5 then 5 else (1 : ℕ → ℕ) in -- a_i is 5 for rows 4 and 5, arbitrary otherwise
  let valid_position (i : ℕ) (j : ℕ) := i ∈ rows ∧ j ∈ cols in -- valid positions on the board
  ∑ i in (range 10).filter (λ i, i ≥ 6), 9 = 6561 := -- Possibility calculation for rows from 6 to 9 (9^4 = 6561)
by sorry

end chinese_chess_paths_count_l548_548654


namespace find_length_of_AB_l548_548674

-- Definitions of the conditions
def areas_ratio (A B C D : Point) (areaABC areaADC : ℝ) :=
  (areaABC / areaADC) = (7 / 3)

def total_length (A B C D : Point) (AB CD : ℝ) :=
  AB + CD = 280

-- Statement of the proof problem
theorem find_length_of_AB
  (A B C D : Point)
  (AB CD : ℝ)
  (areaABC areaADC : ℝ)
  (h_height_not_zero : h ≠ 0) -- Assumption to ensure height is non-zero
  (h_areas_ratio : areas_ratio A B C D areaABC areaADC)
  (h_total_length : total_length A B C D AB CD) :
  AB = 196 := sorry

end find_length_of_AB_l548_548674


namespace plywood_perimeter_difference_l548_548090

/--
Given a 6-foot by 9-foot rectangular piece of plywood cut into 6 congruent rectangles 
with no wood left over and no wood lost due to the cuts,
prove that the positive difference between the greatest and the least perimeter of a single piece is 11 feet.
-/
theorem plywood_perimeter_difference :
  ∃ (rectangles : List (ℕ × ℕ)), 
  (∀ r ∈ rectangles, r.fst * r.snd = 9 * 6 / 6) ∧
  (Greatest (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) - 
  (Least (λ r : ℕ × ℕ, 2 * r.fst + 2 * r.snd) rectangles) = 11 :=
by
  sorry

end plywood_perimeter_difference_l548_548090


namespace polar_to_rectangular_l548_548185

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 5) (h₂ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5 / 2, -5 * Real.sqrt 3 / 2) :=
by sorry

end polar_to_rectangular_l548_548185


namespace average_weight_increase_l548_548402

theorem average_weight_increase
 (num_persons : ℕ) (weight_increase : ℝ) (replacement_weight : ℝ) (new_weight : ℝ) (weight_difference : ℝ) (avg_weight_increase : ℝ)
 (cond1 : num_persons = 10)
 (cond2 : replacement_weight = 65)
 (cond3 : new_weight = 90)
 (cond4 : weight_difference = new_weight - replacement_weight)
 (cond5 : weight_difference = weight_increase)
 (cond6 : avg_weight_increase = weight_increase / num_persons) :
avg_weight_increase = 2.5 :=
by
  sorry

end average_weight_increase_l548_548402
