import Mathlib

namespace average_of_consecutive_odds_is_24_l305_305606

theorem average_of_consecutive_odds_is_24 (a b c d : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : d = 27) 
  (h5 : b = d - 2) (h6 : c = d - 4) (h7 : a = d - 6) 
  (h8 : ∀ x : ℤ, x % 2 = 1) :
  ((a + b + c + d) / 4) = 24 :=
by {
  sorry
}

end average_of_consecutive_odds_is_24_l305_305606


namespace probability_four_dots_collinear_in_5x5_grid_l305_305118

theorem probability_four_dots_collinear_in_5x5_grid :
  let grid := fin 5 × fin 5 in
  (∑ r : fin 5, (nat.choose 5 4)) + 
  (∑ c : fin 5, (nat.choose 5 4)) + 
  (∑ d in ({-4, -3, -2, -1, 0, 1, 2, 3, 4} : set ℤ), if nat.abs d ≤ 3 then (nat.choose 5 4) else 0) = 
  (25 + 25 + 40) / (nat.choose 25 4) :=
by sorry

end probability_four_dots_collinear_in_5x5_grid_l305_305118


namespace prime_sum_count_is_six_l305_305190

noncomputable def nth_prime : ℕ → ℕ
| 0     => 2
| 1     => 3
| (n+2) => (Nat.succ (nth_prime (n+1))).find (λ m, m > nth_prime (n+1) ∧ Nat.prime m)

noncomputable def alt_sum (n : ℕ) : ℤ :=
  let rec aux (i : ℕ) (s : ℤ) (sign : ℤ) : ℤ :=
    if i >= n then s
    else aux (i + 1) (s + sign * nth_prime i) (-sign)
  aux 0 2 1

def is_prime (n : ℤ) : Bool :=
  n > 1 ∧ Nat.prime (Int.natAbs n)

def count_prime_sums : ℕ :=
   let rec aux (i : ℕ) (count : ℕ) : ℕ :=
     if i == 15 then count
     else if is_prime (alt_sum (i+1)) then aux (i + 1) (count + 1)
     else aux (i + 1) count
   aux 0 0

theorem prime_sum_count_is_six : count_prime_sums = 6 := 
  by
    sorry

end prime_sum_count_is_six_l305_305190


namespace rectangle_is_square_l305_305538

variables {A B C D M N Q : Type}
variables [IsRect A B C D] [LocatedOn M AD] [PerpendicularBisector MC BC N] [Intersection MN AB Q]
variables (AB_geq_BC : AB ≥ BC) (angle_cond : ∠ MQA = 2 * ∠ BCQ)

theorem rectangle_is_square : AB = BC :=
sorry

end rectangle_is_square_l305_305538


namespace single_digit_no_solution_l305_305264

theorem single_digit_no_solution : 
  ∀ (x a z b : ℕ), 
    (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) → 
    x = 1 / 4 * a → 
    z = 1 / 4 * b → 
    x^2 + a^2 = z^2 + b^2 → 
    (x + a)^3 > (z + b)^3 → 
    False := 
by {
  intros,
  sorry
}

end single_digit_no_solution_l305_305264


namespace num_ways_to_cut_figure_with_8_rect_1_sq_l305_305204

-- Assume the definition of the problem
def numWaysToCut (figure : List (List (Nat × Nat))) : Nat := sorry

-- The specific checkerboard configuration is abstracted as some figure f
constant f : List (List (Nat × Nat))

-- Condition: f is a specific figure consisting of 17 cells and can be analyzed as described
axiom a1 : count_cells f = 17
axiom a2 : checkerboard_pattern f  -- Assume this term asserts the checkerboard pattern condition

-- Statement: Prove the number of ways to cut f into 8 (1 × 2) rectangles and 1 (1 × 1) square is 10
theorem num_ways_to_cut_figure_with_8_rect_1_sq : numWaysToCut f = 10 :=
begin
  sorry
end

end num_ways_to_cut_figure_with_8_rect_1_sq_l305_305204


namespace series_sum_1503_l305_305782

noncomputable def sum_series (n : ℕ) : ℂ :=
  (∑ k in finset.range (n + 1), (k + 1) * i ^ (2 * (k + 1)))

theorem series_sum_1503 : sum_series 2004 = 1503 := 
sorry

end series_sum_1503_l305_305782


namespace problem1_problem2_problem3_l305_305396

-- Definitions and conditions
def seq (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) - 2 * a n = 2^n
def C (a : ℕ → ℝ) (n : ℕ) := (2 * a n - 2 * n) / n
def geometric_seq (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) / a n = r

-- Given the defined conditions
variable {a : ℕ → ℝ} (h1 : seq a) (h2 : a 1 = 1) (h3 : a 2 = 4)

-- Prove that the sequence {a_n / 2^n} is an arithmetic sequence
theorem problem1 : ∃ d : ℝ, ∀ n : ℕ, (a (n+1) / 2^(n+1)) - (a n / 2^n) = d := 
sorry

-- Find the sum of the first n terms of the sequence {a_n}, denoted as S_n
noncomputable def S (n : ℕ) := ∑ i in finset.range n, a (i+1)
theorem problem2 : ∀ n : ℕ, S n = (n-1) * 2^n + 1 := 
sorry

-- Prove the given inequality about C_n
theorem problem3 (n : ℕ) (hn : 2 ≤ n) :
  (1 / 2) - (1 / 2)^n < ∑ i in finset.range (n - 1) (λ i, 1 / (C a (i + 2))) ∧ 
  ∑ i in finset.range (n - 1) (λ i, 1 / (C a (i + 2))) ≤ 1 - (1 / 2)^(n - 1) := 
sorry

end problem1_problem2_problem3_l305_305396


namespace correct_equation_l305_305194

noncomputable def team_a_initial := 96
noncomputable def team_b_initial := 72
noncomputable def team_b_final (x : ℕ) := team_b_initial - x
noncomputable def team_a_final (x : ℕ) := team_a_initial + x

theorem correct_equation (x : ℕ) : 
  (1 / 3 : ℚ) * (team_a_final x) = (team_b_final x) := 
  sorry

end correct_equation_l305_305194


namespace hypotenuse_length_l305_305491

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305491


namespace number_of_kids_per_day_l305_305141

theorem number_of_kids_per_day (K : ℕ) 
    (kids_charge : ℕ := 3) 
    (adults_charge : ℕ := kids_charge * 2) 
    (daily_earnings_from_adults : ℕ := 10 * adults_charge) 
    (weekly_earnings : ℕ := 588) 
    (daily_earnings : ℕ := weekly_earnings / 7) :
    (daily_earnings - daily_earnings_from_adults) / kids_charge = 8 :=
by
  sorry

end number_of_kids_per_day_l305_305141


namespace band_total_earnings_l305_305274

variables (earnings_per_gig_per_member : ℕ)
variables (number_of_members : ℕ)
variables (number_of_gigs : ℕ)

theorem band_total_earnings :
  earnings_per_gig_per_member = 20 →
  number_of_members = 4 →
  number_of_gigs = 5 →
  earnings_per_gig_per_member * number_of_members * number_of_gigs = 400 :=
by
  intros
  sorry

end band_total_earnings_l305_305274


namespace necessary_but_not_sufficient_l305_305051

theorem necessary_but_not_sufficient (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (log 2 (exp (2 * x) - 1) < 2) ↔ (0 < x ∧ x < (1 / 2) * log 5) :=
sorry

end necessary_but_not_sufficient_l305_305051


namespace smallest_k_exists_l305_305540

noncomputable def min_k (n : ℕ) : ℕ := ⌈log 2 n⌉

theorem smallest_k_exists {S : Type*} (h : finite S) (n : ℕ) (hn : S.card = n) (k : ℕ) 
  (A : fin k → set S) : 
  (∀ (B : fin k → set S), (∀ i, B i = A i ∨ B i = S \ A i) → (⋃ i, B i) = S) →
  k ≥ min_k n :=
sorry

end smallest_k_exists_l305_305540


namespace difference_of_distinct_members_set_l305_305433

theorem difference_of_distinct_members_set :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≤ n → ∃ a b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), a ≠ b ∧ m = a - b ∨ m = b - a ∧ a > b) :=
by
  sorry

end difference_of_distinct_members_set_l305_305433


namespace problem_l305_305058

-- Define the points A, B, and C
structure Point (α : Type) :=
(x : α)
(y : α)

-- Define the coordinates in terms of real numbers
def A : Point ℝ := ⟨3, 0⟩
def B : Point ℝ := ⟨3, 0⟩
def C (α : ℝ) : Point ℝ := ⟨Real.cos α, Real.sin α⟩ 

-- Define the vectors AC and BC
def vector_AC (α : ℝ) : Point ℝ := ⟨(C α).x - A.x, (C α).y - A.y⟩
def vector_BC (α : ℝ) : Point ℝ := ⟨(C α).x - B.x, (C α).y - B.y⟩

-- Define the dot product of two vectors
def dot_product (u v : Point ℝ) : ℝ := u.x * v.x + u.y * v.y

-- Define the hypothesis and the targets
theorem problem (α : ℝ) (h : dot_product (vector_AC α) (vector_BC α) = -1/2) :
  sin α + cos α = 1/2 ∧ 
  (Real.sin (π - 4 * α) * Real.cos (2 * π - 2 * α)) / (1 + Real.sin (π/2 + 4 * α)) = -(3/4) := 
sorry

end problem_l305_305058


namespace track_length_is_correct_l305_305165

-- Definitions from the conditions
def Polly_laps_in_half_hour : ℕ := 12
def Gerald_speed_mph : ℝ := 3
def Gerald_relative_speed : ℝ := 1/2
def Polly_time_hours : ℝ := 0.5

-- Derived definitions using the conditions
def Polly_speed_mph := Gerald_speed_mph * (1 / Gerald_relative_speed)
def Polly_distance_miles := Polly_speed_mph * Polly_time_hours
def track_length_miles := Polly_distance_miles / Polly_laps_in_half_hour

-- Main theorem statement
theorem track_length_is_correct : track_length_miles = 0.25 :=
by
  sorry

end track_length_is_correct_l305_305165


namespace distinct_differences_count_l305_305437

-- Define the set of interest.
def mySet : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- The statement we want to prove.
theorem distinct_differences_count : 
  (finset.image (λ (x : ℕ × ℕ), (x.1 - x.2)) ((mySet.product mySet).filter (λ x, x.1 > x.2))).card = 7 :=
sorry

end distinct_differences_count_l305_305437


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305506

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305506


namespace find_x_l305_305157

def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem find_x 
  (x y : ℤ) 
  (h_star1 : star 5 4 2 2 = (7, 2)) 
  (h_eq : star x y 3 3 = (7, 2)) : 
  x = 4 := 
sorry

end find_x_l305_305157


namespace units_digit_two_pow_2010_l305_305256

-- Conditions from part a)
def two_power_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case will not occur due to modulo operation

-- Question translated to a proof problem
theorem units_digit_two_pow_2010 : (two_power_units_digit 2010) = 4 :=
by 
  -- Proof would go here
  sorry

end units_digit_two_pow_2010_l305_305256


namespace arithmetic_seq_a6_l305_305382

theorem arithmetic_seq_a6 (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (0 < q) →
  a 1 = 1 →
  S 3 = 7/4 →
  S n = (1 - q^n) / (1 - q) →
  (∀ n, a n = 1 * q^(n - 1)) →
  a 6 = 1 / 32 :=
by
  sorry

end arithmetic_seq_a6_l305_305382


namespace total_candies_l305_305081

variable (Maggie Harper Neil Liam : ℕ)

-- Conditions
def Maggie_candies := 50
def Harper_candies := Maggie_candies + Nat.floor (0.30 * Maggie_candies)
def Neil_candies := Harper_candies + Nat.floor (0.40 * Harper_candies)
def Liam_candies := Neil_candies + Nat.floor (0.20 * Neil_candies)

-- Theorem Statement
theorem total_candies : Maggie_candies + Harper_candies + Neil_candies + Liam_candies = 315 := by
  sorry

end total_candies_l305_305081


namespace hypotenuse_length_l305_305499

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305499


namespace sequence_sum_inverse_l305_305788

theorem sequence_sum_inverse (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : ∀ n, S n = 2^n) 
(h2 : a 1 = S 1)
(h3 : ∀ n, n ≥ 2 → a n = S n - S (n-1)) :
  ∀ n, (∑ i in finset.range (n+1), 1 / a (i + 1)) = (3/2) - (1 / 2^(n-1)) :=
by
  sorry

end sequence_sum_inverse_l305_305788


namespace additional_people_needed_l305_305344

-- Definition of the conditions
def person_hours (people: ℕ) (hours: ℕ) : ℕ := people * hours

-- Assertion that 8 people can paint the fence in 3 hours
def eight_people_three_hours : Prop := person_hours 8 3 = 24

-- Definition of the additional people required
def additional_people (initial_people required_people: ℕ) : ℕ := required_people - initial_people

-- Main theorem stating the problem
theorem additional_people_needed : eight_people_three_hours → additional_people 8 12 = 4 :=
by
  sorry

end additional_people_needed_l305_305344


namespace triangle_sides_l305_305743

variable {A B C T : ℝ} -- Define the given variables as real numbers

theorem triangle_sides (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hT : T ≠ 0) :
  let a := Real.sqrt(2 * T * Real.sin A / (Real.sin B * Real.sin C))
  let b := Real.sqrt(2 * T * Real.sin B / (Real.sin C * Real.sin A))
  let c := Real.sqrt(2 * T * Real.sin C / (Real.sin A * Real.sin B))
  in True :=
by
  sorry

end triangle_sides_l305_305743


namespace shaded_area_l305_305465

open Real

theorem shaded_area (AH HF GF : ℝ) (AH_eq : AH = 12) (HF_eq : HF = 16) (GF_eq : GF = 4) 
  (DG : ℝ) (DG_eq : DG = 3) (area_triangle_DGF : ℝ) (area_triangle_DGF_eq : area_triangle_DGF = 6) :
  let area_square : ℝ := 4 * 4
  let shaded_area : ℝ := area_square - area_triangle_DGF
  shaded_area = 10 := by
    sorry

end shaded_area_l305_305465


namespace high_heels_height_l305_305281

theorem high_heels_height (x : ℝ) :
  let height := 157
  let lower_limbs := 95
  let golden_ratio := 0.618
  (95 + x) / (157 + x) = 0.618 → x = 5.3 :=
sorry

end high_heels_height_l305_305281


namespace find_BD_l305_305912

noncomputable def triangle_ABC_area := 180
noncomputable def AC := 30

theorem find_BD 
  (ABC : Triangle)
  (h_right : ABC.angles = [90,α,β])
  (circle_diameter_BC_meets_AC_at_D : ∃ D, circle_diameter BC AC D)
  (h_area : ABC.area = triangle_ABC_area)
  (h_AC : ABC.side_length AC = AC) :
  exists BD : ℝ, BD = 12 := 
sorry

end find_BD_l305_305912


namespace diego_annual_savings_l305_305339

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end diego_annual_savings_l305_305339


namespace puppy_food_consumption_l305_305926

/-- Mathematically equivalent proof problem:
  Given the following conditions:
  1. days_per_week = 7
  2. initial_feeding_duration_weeks = 2
  3. initial_feeding_daily_portion = 1/4
  4. initial_feeding_frequency_per_day = 3
  5. subsequent_feeding_duration_weeks = 2
  6. subsequent_feeding_daily_portion = 1/2
  7. subsequent_feeding_frequency_per_day = 2
  8. today_feeding_portion = 1/2
  Prove that the total food consumption, including today, over the next 4 weeks is 25 cups.
-/
theorem puppy_food_consumption :
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  total = 25 := by
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  show total = 25 from sorry

end puppy_food_consumption_l305_305926


namespace jerome_money_left_l305_305837

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end jerome_money_left_l305_305837


namespace ellipse_focus_area_maximized_l305_305390

noncomputable def ellipse_focus_area_maximized_value (m n : ℝ) : Prop :=
  let F₁ := (-3, 0) in
  let F₂ := (3, 0) in
  let a := Real.sqrt (m / 12) in -- since a^2 = 12
  let b := Real.sqrt (n / 3) in -- since b^2 = 3
  m + n = 15

theorem ellipse_focus_area_maximized :
  ∃ (m n : ℝ), ellipse_focus_area_maximized_value m n := 
by
  use 12, 3
  simp [ellipse_focus_area_maximized_value]
  sorry

end ellipse_focus_area_maximized_l305_305390


namespace trisha_bought_amount_initially_l305_305643

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end trisha_bought_amount_initially_l305_305643


namespace initial_amount_liquid_A_l305_305282

-- Definitions and conditions
def initial_ratio (a : ℕ) (b : ℕ) := a = 4 * b
def replaced_mixture_ratio (a : ℕ) (b : ℕ) (r₀ r₁ : ℕ) := 4 * r₀ = 2 * (r₁ + 20)

-- Theorem to prove the initial amount of liquid A
theorem initial_amount_liquid_A (a b r₀ r₁ : ℕ) :
  initial_ratio a b → replaced_mixture_ratio a b r₀ r₁ → a = 16 := 
by
  sorry

end initial_amount_liquid_A_l305_305282


namespace number_of_elements_with_first_digit_3_l305_305911

-- Define the set S and relevant constants
def S := {k : ℕ | 0 ≤ k ∧ k ≤ 4500}

-- Given conditions
def three_power_4500_digits : ℕ := 2150
def first_digit_three_power_4500 : ℕ := 3

-- Proposition to prove
theorem number_of_elements_with_first_digit_3 :
  ∃ (count : ℕ), 
    (count = 2351) ∧
    ∀ k ∈ S, 
        let first_digit := (3^k).to_digits.to_list.head in
        first_digit = 3 
        → count = 2351 :=
sorry

end number_of_elements_with_first_digit_3_l305_305911


namespace value_of_x_squared_plus_y_squared_l305_305853

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l305_305853


namespace angles_of_cyclic_quadrilateral_l305_305701

theorem angles_of_cyclic_quadrilateral (A B C D : Point) (α β δ γ : ℝ)
(h1 : is_convex_quadrilateral ABCD)
(h2 : is_cyclic ABCD)
(h3 : is_angle_bisector BD ∠ABC)
(h4 : angle_between_diagonals = 65)
(h5 : angle_with_side_AD = 55) :
  (α = 60) ∧ (β = 30) ∧ (γ = 120) ∧ (δ = 150) :=
by
  sorry

end angles_of_cyclic_quadrilateral_l305_305701


namespace melanie_sale_revenue_correct_l305_305572

noncomputable def melanie_revenue : ℝ :=
let red_cost := 0.08
let green_cost := 0.10
let yellow_cost := 0.12
let red_gumballs := 15
let green_gumballs := 18
let yellow_gumballs := 22
let total_gumballs := red_gumballs + green_gumballs + yellow_gumballs
let total_cost := (red_cost * red_gumballs) + (green_cost * green_gumballs) + (yellow_cost * yellow_gumballs)
let discount := if total_gumballs >= 20 then 0.30 else if total_gumballs >= 10 then 0.20 else 0
let final_cost := total_cost * (1 - discount)
final_cost

theorem melanie_sale_revenue_correct : melanie_revenue = 3.95 :=
by
  -- All calculations and proofs omitted for brevity, as per instructions above
  sorry

end melanie_sale_revenue_correct_l305_305572


namespace find_U_value_l305_305552

def f : ℕ → ℝ
def U (T : ℕ) : ℝ :=
  (f T) / ((T - 1) * (f (T - 3)))

theorem find_U_value (h1 : ∀ n, n ≥ 6 → f n = (n-1) * f (n-1))
  (h2 : ∀ n, n ≥ 6 → f n ≠ 0) : U 11 = 72 :=
by
  sorry

end find_U_value_l305_305552


namespace oranges_left_uneaten_l305_305162

theorem oranges_left_uneaten :
  let total_oranges := 720 in
  let ripe_fraction := 3 / 7 in
  let partially_ripe_fraction := 2 / 5 in
  let spoiled_oranges := 15 in
  let unknown_oranges := 5 in
  let eaten_ripe_fraction := 2 / 3 in
  let eaten_partially_ripe_fraction := 4 / 7 in
  let eaten_unripe_fraction := 1 / 3 in
  let eaten_spoiled_fraction := 1 / 2 in
  let eaten_unknown_oranges := 1 in
  let ripe_oranges := (720 * 3 / 7).toInt in
  let partially_ripe_oranges := (720 * 2 / 5).toInt in
  let total_without_spoiled_and_unknown := 720 - 15 - 5 in
  let unripe_oranges := total_without_spoiled_and_unknown - ripe_oranges - partially_ripe_oranges in
  let eaten_ripe := (ripe_oranges * 2 / 3).toInt in
  let eaten_partially_ripe := (partially_ripe_oranges * 4 / 7).toInt in
  let eaten_unripe := (unripe_oranges * 1 / 3).toInt in
  let eaten_spoiled := (15 * 1 / 2).toInt in
  let total_eaten := eaten_ripe + eaten_partially_ripe + eaten_unripe + eaten_spoiled + 1 in
  720 - total_eaten = 309 :=
by
  -- Definitions and calculations would be filled here
  sorry

end oranges_left_uneaten_l305_305162


namespace largest_prime_factor_of_87_l305_305665

theorem largest_prime_factor_of_87 : 
  ∀ a ∈ ({65, 87, 143, 169, 187} : set ℕ), 
    (∀ p ∈ ({factorize 65, factorize 87, factorize 143, factorize 169, factorize 187} : set (list ℕ)), 
      max p = 29) → 
    max (factorize 87) = 29 := by
  sorry

end largest_prime_factor_of_87_l305_305665


namespace Jolyn_older_than_Therese_by_2_l305_305368

variables (Aivo Leon Therese Jolyn : ℕ)
variables (Therese_older_Aivo Leon_older_Aivo Jolyn_older_Leon : ℕ)

-- Conditions
def condition1 : Jolyn > Therese := sorry
def condition2 : Therese = Aivo + 5 := sorry
def condition3 : Leon = Aivo + 2 := sorry
def condition4 : Jolyn = Leon + 5 := sorry

-- Proving that Jolyn is 2 months older than Therese
theorem Jolyn_older_than_Therese_by_2 : Jolyn - Therese = 2 :=
by
  rw [condition4, condition3, condition2]
  exact sorry

end Jolyn_older_than_Therese_by_2_l305_305368


namespace negate_universal_statement_l305_305984

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, exp x - x - 1 < 0) :=
by sorry

end negate_universal_statement_l305_305984


namespace E_runs_is_20_l305_305464

-- Definitions of runs scored by each batsman as multiples of 4
def a := 28
def e := 20
def d := e + 12
def b := d + e
def c := 107 - b
def total_runs := a + b + c + d + e

-- Adding conditions
axiom A_max: a > b ∧ a > c ∧ a > d ∧ a > e
axiom runs_multiple_of_4: ∀ (x : ℕ), x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → x % 4 = 0
axiom average_runs: total_runs = 180
axiom d_condition: d = e + 12
axiom e_condition: e = a - 8
axiom b_condition: b = d + e
axiom bc_condition: b + c = 107

theorem E_runs_is_20 : e = 20 := by
  sorry

end E_runs_is_20_l305_305464


namespace complex_expression_value_l305_305361

theorem complex_expression_value :
  (i^3 * (1 + i)^2 = 2) :=
by
  sorry

end complex_expression_value_l305_305361


namespace percentage_not_sophomores_l305_305877

variable (Total : ℕ) (Juniors Senior : ℕ) (Freshmen Sophomores : ℕ)

-- Conditions
axiom total_students : Total = 800
axiom percent_juniors : (22 / 100) * Total = Juniors
axiom number_seniors : Senior = 160
axiom freshmen_sophomores_relation : Freshmen = Sophomores + 64
axiom total_composition : Freshmen + Sophomores + Juniors + Senior = Total

-- Proof Objective
theorem percentage_not_sophomores :
  (Total - Sophomores) / Total * 100 = 75 :=
by
  -- proof omitted
  sorry

end percentage_not_sophomores_l305_305877


namespace zeros_in_expansion_of_8000_pow_50_l305_305661

theorem zeros_in_expansion_of_8000_pow_50 :
  ∀ (n : ℕ), n = 50 → (8000 ^ n) = (8 ^ n) * (10 ^ (3 * n)) :=
by
  intros n hn
  rw [hn, Nat.pow_mul, Nat.pow_mul 8 10 150]
  sorry

end zeros_in_expansion_of_8000_pow_50_l305_305661


namespace craig_age_l305_305328

theorem craig_age (C M : ℕ) (h1 : C = M - 24) (h2 : C + M = 56) : C = 16 := 
by
  sorry

end craig_age_l305_305328


namespace guests_equal_cost_l305_305965

-- Rental costs and meal costs
def rental_caesars_palace : ℕ := 800
def deluxe_meal_cost : ℕ := 30
def premium_meal_cost : ℕ := 40
def rental_venus_hall : ℕ := 500
def venus_special_cost : ℕ := 35
def venus_platter_cost : ℕ := 45

-- Meal distribution percentages
def deluxe_meal_percentage : ℚ := 0.60
def premium_meal_percentage : ℚ := 0.40
def venus_special_percentage : ℚ := 0.60
def venus_platter_percentage : ℚ := 0.40

-- Total costs calculation
noncomputable def total_cost_caesars (G : ℕ) : ℚ :=
  rental_caesars_palace + deluxe_meal_cost * deluxe_meal_percentage * G + premium_meal_cost * premium_meal_percentage * G

noncomputable def total_cost_venus (G : ℕ) : ℚ :=
  rental_venus_hall + venus_special_cost * venus_special_percentage * G + venus_platter_cost * venus_platter_percentage * G

-- Statement to show the equivalence of guest count
theorem guests_equal_cost (G : ℕ) : total_cost_caesars G = total_cost_venus G → G = 60 :=
by
  sorry

end guests_equal_cost_l305_305965


namespace john_read_bible_in_weeks_l305_305901

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end john_read_bible_in_weeks_l305_305901


namespace magicians_trick_l305_305650

theorem magicians_trick :
  ∀ (circle : Type) (is_in_semicircle : circle → Prop)
    (points : Finset circle),
    points.card = 100 →
    ∃ (point_to_erase : circle),
      ∃ (remaining_points : Finset circle),
        remaining_points.card = 99 ∧
        points = insert point_to_erase remaining_points →
        ¬(∀ point : circle, is_in_semicircle point ↔ is_in_semicircle point_to_erase) → sorry :=
begin
  sorry -- Proof of the theorem
end

end magicians_trick_l305_305650


namespace clock_hands_angle_120_l305_305765

-- We are only defining the problem statement and conditions. No need for proof steps or calculations.

def angle_between_clock_hands (hour minute : ℚ) : ℚ :=
  abs ((30 * hour + minute / 2) - (6 * minute))

-- Given conditions
def time_in_range (hour : ℚ) (minute : ℚ) := 7 ≤ hour ∧ hour < 8

-- Problem statement to be proved
theorem clock_hands_angle_120 (hour minute : ℚ) :
  time_in_range hour minute → angle_between_clock_hands hour minute = 120 :=
sorry

end clock_hands_angle_120_l305_305765


namespace find_other_odd_integer_l305_305221

theorem find_other_odd_integer (n : ℤ) (h1 : odd n) (h2 : odd (19 - n)) (h3 : 19 + n ≥ 36) : n = 17 ∨ n = 21 :=
by
  sorry

end find_other_odd_integer_l305_305221


namespace balls_into_boxes_l305_305843

theorem balls_into_boxes :
  (finset.univ.sum (λ x : (finset (finset ℕ)), (if x.sum id = 5 then (finset.univ.prod (λ y : x.to_list (ℕ → ℕ), y.2)) else 0))) = 56 :=
begin
  sorry
end

end balls_into_boxes_l305_305843


namespace polynomial_grows_faster_than_logarithmic_l305_305255

noncomputable def grows_faster {f g : ℝ → ℝ} (I : Set ℝ) :=
  ∀ x ∈ I, f x > g x

theorem polynomial_grows_faster_than_logarithmic :
  grows_faster (λ x, 2 * x) (λ x, Real.log x + 1) (Set.Ioi 1) :=
by
  sorry

end polynomial_grows_faster_than_logarithmic_l305_305255


namespace car_travel_time_relation_l305_305687

variable (x : ℝ)

-- Conditions
def distance := 80
def small_car_speed := 3 * x
def large_car_travel_time := distance / x
def small_car_travel_time := distance / small_car_speed
def time_difference := 2
def arrival_difference := 2 / 3

-- Statement of the problem
theorem car_travel_time_relation
  (h_large_car : large_car_travel_time x = distance / x)
  (h_small_car : small_car_travel_time x = distance / small_car_speed x)
  (h_time_diff : time_difference = 2)
  (h_arrival_diff : arrival_difference = 2 / 3) :
  large_car_travel_time x - time_difference = small_car_travel_time x + arrival_difference := by
  sorry

end car_travel_time_relation_l305_305687


namespace min_abs_phi_l305_305864

theorem min_abs_phi {f : ℝ → ℝ} (h : ∀ x, f x = 3 * Real.sin (2 * x + φ) ∧ ∀ x, f (x) = f (2 * π / 3 - x)) :
  |φ| = π / 6 :=
by
  sorry

end min_abs_phi_l305_305864


namespace quadratic_sum_constants_l305_305974

theorem quadratic_sum_constants (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x = -3 ∨ x = 5)
  (h_min : ∀ x, a * x^2 + b * x + c ≥ 36) 
  (h_at : a * 1^2 + b * 1 + c = 36) :
  a + b + c = 36 :=
sorry

end quadratic_sum_constants_l305_305974


namespace total_employees_l305_305512

-- Define the total number of employees as a variable
variable (E : ℝ)

-- Define the conditions as hypotheses
def condition_1 : Prop := 0.25 * E
def condition_2 (m: ℝ) : Prop := 0.30 * m = 0.70 * m - 490
def condition_3 (m: ℝ) : Prop := 0.70 * (0.25 * E) ≈ 490

-- State the final theorem to prove the total number of employees
theorem total_employees (E : ℝ) 
  (h1 : condition_1 E) 
  (h2 : ∀ m, m = 0.25 * E → condition_2 m) 
  (h3 : ∀ m, m = 0.25 * E → condition_3 m) :
  E ≈ 2800 :=
by
  sorry

end total_employees_l305_305512


namespace sum_when_b_zero_l305_305381

variable (a_n : Nat → ℕ) -- Define sequence a_n
variable (S_n : Nat → ℕ) -- Define sum sequence S_n
variable (a1 : ℕ) -- First term a1
variable (b : ℕ) -- Multiplier b

-- Condition: sequence definition
axiom a_n_def (n : ℕ) : n > 0 → a_n(n) = b * a_n(n - 1)

-- Condition: sum sequence definition
axiom S_n_def (n : ℕ) : S_n(n) = list.sum (list.map a_n (list.range n))

-- Given condition when b = 0
variable (hb : b = 0)

-- Theorem to prove
theorem sum_when_b_zero : ∀ n : ℕ, S_n(n) = a1 :=
by sorry

end sum_when_b_zero_l305_305381


namespace part1_part2_l305_305027

variable (m : ℝ)

-- Condition definitions
def q : Prop := ∃ x₀ ∈ set.Icc 0 3, x₀^2 - 2*x₀ - m ≥ 0
def p : Prop := ∀ x, mx^2 - mx + 1 > 0

-- Required propositions
theorem part1 (hq : q m) : m ≤ 3 := sorry

theorem part2 (hp_or_q : p m ∨ q m) (not_hp_and_q : ¬(p m ∧ q m)) 
  : m < 0 ∨ (3 < m ∧ m < 4) := sorry

end part1_part2_l305_305027


namespace max_min_values_of_f_l305_305351

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≥ -18) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = -18)
:= by
  sorry  -- To be replaced with the actual proof

end max_min_values_of_f_l305_305351


namespace rice_cost_l305_305894

theorem rice_cost (ratio : ℝ) (cost_first_variety cost_mixture cost_second_variety : ℝ) :
  ratio = 0.625 →
  cost_first_variety = 5.5 →
  cost_mixture = 7.5 →
  cost_second_variety = 10.7 :=
by
  assume h1: ratio = 0.625,
  assume h2: cost_first_variety = 5.5,
  assume h3: cost_mixture = 7.5,
  -- Definition of the rule of alligation (setup)
  let equation := (cost_first_variety - cost_mixture) / (cost_mixture - cost_second_variety) = ratio,
  -- Simplification and direct proof can be realized now to show cost_second_variety = 10.7
  sorry

end rice_cost_l305_305894


namespace prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l305_305880

noncomputable def prob_zhang_nings_wins_2_1 :=
  2 * 0.4 * 0.6 * 0.6 = 0.288

theorem prob_zhang_nings_wins_2_1_correct : prob_zhang_nings_wins_2_1 := sorry

def prob_ξ_minus_2 := 0.4 * 0.4 = 0.16
def prob_ξ_minus_1 := 2 * 0.4 * 0.6 * 0.4 = 0.192
def prob_ξ_1 := 2 * 0.4 * 0.6 * 0.6 = 0.288
def prob_ξ_2 := 0.6 * 0.6 = 0.36

theorem prob_ξ_minus_2_correct : prob_ξ_minus_2 := sorry
theorem prob_ξ_minus_1_correct : prob_ξ_minus_1 := sorry
theorem prob_ξ_1_correct : prob_ξ_1 := sorry
theorem prob_ξ_2_correct : prob_ξ_2 := sorry

noncomputable def expected_value_ξ :=
  (-2 * 0.16) + (-1 * 0.192) + (1 * 0.288) + (2 * 0.36) = 0.496

theorem expected_value_ξ_correct : expected_value_ξ := sorry

end prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l305_305880


namespace compound_interest_calculation_l305_305866

theorem compound_interest_calculation : 
  ∀ (x y T SI: ℝ), 
  x = 5000 → T = 2 → SI = 500 → 
  (y = SI * 100 / (x * T)) → 
  (5000 * (1 + (y / 100))^T - 5000 = 512.5) :=
by 
  intros x y T SI hx hT hSI hy
  sorry

end compound_interest_calculation_l305_305866


namespace hexagon_triangle_sides_l305_305337

theorem hexagon_triangle_sides (O : EuclideanGeometry.Point ℝ) :
  ∀ (A₁ A₂ A₃ A₄ A₅ A₆ : EuclideanGeometry.Point ℝ),
  (EquilateralTriangle A₁ A₂ 1) ∧ (EquilateralTriangle A₂ A₃ 1) ∧ 
  (EquilateralTriangle A₃ A₄ 1) ∧ (EquilateralTriangle A₄ A₅ 1) ∧ 
  (EquilateralTriangle A₅ A₆ 1) ∧ (EquilateralTriangle A₆ A₁ 1) →
  ((O ≠ centroidOfHexagon A₁ A₂ A₃ A₄ A₅ A₆) ∨ (O = centroidOfHexagon A₁ A₂ A₃ A₄ A₅ A₆)) →
  ∃ (i j : Fin 6), (i ≠ j) ∧ (1 ≤ dist O Aᵢ) ∧ (1 ≤ dist O Aⱼ) ∧ 
  (∀ k, k ≠ i → k ≠ j → 1 ≤ dist (connectTri O Aᵢ Aⱼ) (connectTri O Aᵢ Aⱼ)). 

end hexagon_triangle_sides_l305_305337


namespace point_on_z_axis_l305_305128

open Real

theorem point_on_z_axis (P : ℝ × ℝ × ℝ) : 
  let A := (1, -2, 1)
  let B := (2, 2, 2)
  P.1 = 0 ∧ P.2 = 0 ∧ dist P A = dist P B → P = (0, 0, 3) :=
by
  -- Defining the distance function for Cartesian points in ℝ × ℝ × ℝ
  let dist (X Y : ℝ × ℝ × ℝ) := 
    sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (X.3 - Y.3)^2)
  sorry

end point_on_z_axis_l305_305128


namespace compare_a_b_c_l305_305023

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l305_305023


namespace probability_of_winning_l305_305211

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end probability_of_winning_l305_305211


namespace train_car_speed_ratio_l305_305636

theorem train_car_speed_ratio
  (distance_bus : ℕ) (time_bus : ℕ) (distance_car : ℕ) (time_car : ℕ)
  (speed_bus := distance_bus / time_bus)
  (speed_train := speed_bus / (3 / 4))
  (speed_car := distance_car / time_car)
  (ratio := (speed_train : ℚ) / (speed_car : ℚ))
  (h1 : distance_bus = 480)
  (h2 : time_bus = 8)
  (h3 : distance_car = 450)
  (h4 : time_car = 6) :
  ratio = 16 / 15 :=
by
  sorry

end train_car_speed_ratio_l305_305636


namespace compute_expression_l305_305752

theorem compute_expression :
  (-Real.sqrt 27 + Real.cos (Real.pi / 6) - (Real.pi - Real.sqrt 2)^0 + (-1/2)^(-1)) = -((5 * Real.sqrt 3 + 6) / 2) :=
by
  sorry

end compute_expression_l305_305752


namespace pentagon_rectangle_ratio_l305_305290

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end pentagon_rectangle_ratio_l305_305290


namespace area_of_smaller_circle_l305_305237

noncomputable def radius_of_smaller_circle (r : ℝ) : ℝ := r

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 3 * r

noncomputable def length_PA := 5
noncomputable def length_AB := 5

theorem area_of_smaller_circle (r : ℝ) (h1 : radius_of_smaller_circle r = r)
  (h2 : radius_of_larger_circle r = 3 * r)
  (h3 : length_PA = 5) (h4 : length_AB = 5) :
  π * r^2 = (25 / 3) * π :=
  sorry

end area_of_smaller_circle_l305_305237


namespace general_formula_l305_305616

def sequence : ℕ → ℕ 
| 1 := 1
| 2 := 3
| 3 := 6
| 4 := 10
| (n + 1) := sequence n + (n + 1)

theorem general_formula (n : ℕ) :
  sequence n = n * (n + 1) / 2 :=
sorry

end general_formula_l305_305616


namespace Kelly_needs_to_give_away_l305_305534

variable (n k : Nat)

theorem Kelly_needs_to_give_away (h_n : n = 20) (h_k : k = 12) : n - k = 8 := 
by
  sorry

end Kelly_needs_to_give_away_l305_305534


namespace return_journey_steps_l305_305875

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def calculate_net_steps : ℤ :=
  let prime_moves := (2 to 25).filter is_prime in
  let composite_moves := (2 to 25).filter (λ n, ¬ is_prime n) in
  2 * prime_moves.length - 3 * composite_moves.length

theorem return_journey_steps :
  calculate_net_steps = -27 :=
by
  sorry

end return_journey_steps_l305_305875


namespace tan_sum_cases_l305_305817

theorem tan_sum_cases 
  (α β : ℝ) 
  (p q : ℝ) 
  (H1 : tan α + tan β = p) 
  (H2 : cot α + cot β = q) :
  (p = 0 ∧ q = 0 → tan (α + β) = 0) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q → tan (α + β) = p * q / (q - p)) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p = q → ¬ ∃ y : ℝ, y = tan (α + β)) ∧
  ((p = 0 ∨ q = 0) ∧ p ≠ q → false) :=
by
  sorry

end tan_sum_cases_l305_305817


namespace cases_needed_to_raise_funds_l305_305937

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end cases_needed_to_raise_funds_l305_305937


namespace right_triangle_other_side_l305_305457

theorem right_triangle_other_side (c a : ℝ) (h_c : c = 10) (h_a : a = 6) : ∃ b : ℝ, b^2 = c^2 - a^2 ∧ b = 8 :=
by
  use 8
  rw [h_c, h_a]
  simp
  sorry

end right_triangle_other_side_l305_305457


namespace log_expression_l305_305980

-- Define the base 2 logarithm
def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the given conditions
def sixteen : ℝ := 2^4
def cube_root_four : ℝ := 2^(2/3)
def sixth_root_sixty_four : ℝ := 2^1

-- The main hypothesis to prove
theorem log_expression :
  log2 (sixteen * cube_root_four * sixth_root_sixty_four) = 17/3 :=
by
  sorry

end log_expression_l305_305980


namespace find_x_coordinate_l305_305885

theorem find_x_coordinate :
  ∃ x : ℝ, (∃ m b : ℝ, (∀ y x : ℝ, y = m * x + b) ∧ 
                     ((3 = m * 10 + b) ∧ 
                      (0 = m * 4 + b)
                     ) ∧ 
                     (-3 = m * x + b) ∧ 
                     (x = -2)) :=
sorry

end find_x_coordinate_l305_305885


namespace acute_angle_inequality_l305_305167

theorem acute_angle_inequality (a b : ℝ) (α β : ℝ) (γ : ℝ) (h : γ < π / 2) :
  (a^2 + b^2) * Real.cos (α - β) ≤ 2 * a * b :=
sorry

end acute_angle_inequality_l305_305167


namespace Jerome_money_left_l305_305839

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l305_305839


namespace area_of_triangle_ABC_is_30_l305_305871

-- Definitions for the sides and the right angle condition
def AB := 5
def BC := 12
def CD := 30
def DA := 34
def right_angle_CBA := true

-- Function to calculate the area of triangle ABC
def area_triangle_ABC (AB BC: ℝ) (right_angle: Prop) : ℝ :=
  if right_angle then (1/2) * AB * BC else 0

-- Proving the area of ABC is 30
theorem area_of_triangle_ABC_is_30 : area_triangle_ABC AB BC right_angle_CBA = 30 :=
by
  -- Calculation steps already given in the problem
  sorry

end area_of_triangle_ABC_is_30_l305_305871


namespace range_of_a_l305_305456

noncomputable def piecewise_function (x : ℝ) (a : ℝ) : ℝ :=
if h : x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, y ≥ 4 → ∃ x : ℝ, piecewise_function x a = y) → (1 < a ∧ a ≤ 2) :=
begin
  sorry
end

end range_of_a_l305_305456


namespace trisha_bought_amount_initially_l305_305644

-- Define the amounts spent on each item
def meat : ℕ := 17
def chicken : ℕ := 22
def veggies : ℕ := 43
def eggs : ℕ := 5
def dogs_food : ℕ := 45
def amount_left : ℕ := 35

-- Define the total amount spent
def total_spent : ℕ := meat + chicken + veggies + eggs + dogs_food

-- Define the amount brought at the beginning
def amount_brought_at_beginning : ℕ := total_spent + amount_left

-- Theorem stating the amount Trisha brought at the beginning is 167
theorem trisha_bought_amount_initially : amount_brought_at_beginning = 167 := by
  -- Formal proof would go here, we use sorry to skip the proof
  sorry

end trisha_bought_amount_initially_l305_305644


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305504

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305504


namespace probability_regular_2015_gon_l305_305726

theorem probability_regular_2015_gon (vertices : ℕ) (unit_circle : ℝ) 
  (distinct : ∀ i j, i ≠ j → ∥(OA i + OA j)∥ ≥ 1) : 
  vertices = 2015 → unit_circle = 1 → 
  (probability_conditions vertices unit_circle distinct) = (671 / 1007) :=
sorry

end probability_regular_2015_gon_l305_305726


namespace total_peanut_cost_l305_305311

def peanut_cost_per_pound : ℝ := 3
def minimum_pounds : ℝ := 15
def extra_pounds : ℝ := 20

theorem total_peanut_cost :
  (minimum_pounds + extra_pounds) * peanut_cost_per_pound = 105 :=
by
  sorry

end total_peanut_cost_l305_305311


namespace find_x_squared_plus_y_squared_l305_305855

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l305_305855


namespace total_students_l305_305463

-- Definitions for the number of students studying each language and combination of languages
variable (H S E F HS HE HF SE SF EF HSF HEF SEF HSEF : ℕ)
variable (N : ℕ)

-- Conditions
axiom h_h : H = 325
axiom h_s : S = 385
axiom h_e : E = 480
axiom h_f : F = 240

axiom h_h_phys : H∩P = 115
axiom h_s_chem : S∩C = 175
axiom h_e_hist : E∩H = 210
axiom h_f_math : F∩M = 95

axiom h_hs : HS = 140
axiom h_se : SE = 195
axiom h_ef : EF = 165
axiom h_hf : HF = 110
axiom h_hse : HSE = 35
axiom h_sef : SEF = 45
axiom h_hef : HEF = 30
axiom h_hsef : HSEF = 75

-- Question rewritten to Lean statement
theorem total_students : N = H + S + E + F - HS - SE - EF - HF - HSE - HSE - SEF - HEF + HSE + SEF + HEF + HSEF := by
  sorry

end total_students_l305_305463


namespace min_moves_is_22_l305_305180

def casket_coins : List ℕ := [9, 17, 12, 5, 18, 10, 20]

def target_coins (total_caskets : ℕ) (total_coins : ℕ) : ℕ :=
  total_coins / total_caskets

def total_caskets : ℕ := 7

def total_coins (coins : List ℕ) : ℕ :=
  coins.foldr (· + ·) 0

noncomputable def min_moves_to_equalize (coins : List ℕ) (target : ℕ) : ℕ := sorry

theorem min_moves_is_22 :
  min_moves_to_equalize casket_coins (target_coins total_caskets (total_coins casket_coins)) = 22 :=
sorry

end min_moves_is_22_l305_305180


namespace trisha_money_l305_305646

theorem trisha_money (money_meat money_chicken money_veggies money_eggs money_dogfood money_left : ℤ)
  (h_meat : money_meat = 17)
  (h_chicken : money_chicken = 22)
  (h_veggies : money_veggies = 43)
  (h_eggs : money_eggs = 5)
  (h_dogfood : money_dogfood = 45)
  (h_left : money_left = 35) :
  let total_spent := money_meat + money_chicken + money_veggies + money_eggs + money_dogfood
  in total_spent + money_left = 167 :=
by
  sorry

end trisha_money_l305_305646


namespace Randy_trip_distance_l305_305952

theorem Randy_trip_distance (x : ℝ) (h1 : x = 4 * (x / 4 + 30 + x / 6)) : x = 360 / 7 :=
by
  have h2 : x = ((3 * x + 36 * 30 + 2 * x) / 12) := sorry
  have h3 : x = (5 * x / 12 + 30) := sorry
  have h4 : 30 = x - (5 * x / 12) := sorry
  have h5 : 30 = 7 * x / 12 := sorry
  have h6 : x = (12 * 30) / 7 := sorry
  have h7 : x = 360 / 7 := sorry
  exact h7

end Randy_trip_distance_l305_305952


namespace original_square_perimeter_l305_305294

theorem original_square_perimeter (P : ℝ) (x : ℝ) (h1 : 4 * x * 2 + 4 * x = 56) : P = 32 :=
by
  sorry

end original_square_perimeter_l305_305294


namespace evaluate_f_l305_305067

noncomputable def f : Real → Real 
| x => if h : x ≥ 4 then x else f (x + 1)

theorem evaluate_f : f (2 + Real.log 3 / Real.log 2) = 4 + Real.log 3 / Real.log 2 := 
sorry

end evaluate_f_l305_305067


namespace smallest_positive_period_l305_305781

def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x + Real.pi / 4)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T', 0 < T' < T → ∃ x, f (x + T') ≠ f x :=
begin
  use (2 * Real.pi / 3),
  split,
  by norm_num,
  split,
  { intro x,
    simp [f],
    congr' 1,
    ring,
  },
  { intros T' hT',
    by_contra h,
    obtain ⟨k, hk⟩ := Real.exists_rat_btwn hT',
    have : f (x + k) ≠ f x,
    { simp [f],
      intro hf,
      apply irrational_part_neq,
      suffices : 3 * k * Real.pi = 2 * Real.pi * some_integer, from _,
      ring at this,
    },
    exact (h _ ⟨hk.1, hk.2⟩).this },
  sorry,
end

end smallest_positive_period_l305_305781


namespace angle_Z_proof_l305_305562

-- Definitions of the given conditions
variables {p q : Type} [Parallel p q]
variables {X Y Z : ℝ}
variables (mAngleX : X = 100)
variables (mAngleY : Y = 130)

-- Statement of the proof problem
theorem angle_Z_proof (hpq : Parallel p q) (hX : X = 100) (hY : Y = 130) : Z = 130 :=
sorry

end angle_Z_proof_l305_305562


namespace John_reads_Bible_in_4_weeks_l305_305904

def daily_reading_pages (hours_per_day reading_rate : ℕ) : ℕ :=
  hours_per_day * reading_rate

def weekly_reading_pages (daily_pages days_in_week : ℕ) : ℕ :=
  daily_pages * days_in_week

def weeks_to_finish (total_pages daily_pages : ℕ) : ℕ :=
  total_pages / daily_pages

theorem John_reads_Bible_in_4_weeks
  (hours_per_day : ℕ : 2)
  (reading_rate : ℕ := 50)
  (bible_pages : ℕ := 2800)
  (days_in_week : ℕ := 7) :
  weeks_to_finish bible_pages (weekly_reading_pages (daily_reading_pages hours_per_day reading_rate) days_in_week) = 4 :=
  sorry

end John_reads_Bible_in_4_weeks_l305_305904


namespace cube_volume_surface_area_l305_305248

theorem cube_volume_surface_area (V₁ : ℝ) (h₁ : V₁ = 64) :
  ∃ V₂ : ℝ, 
    (∃ s₁ s₂ : ℝ, s₁^3 = V₁ ∧ 6 * s₁^2 = 3 * 6 * s₂^2 ∧ s₂^3 = V₂) 
    ∧ V₂ = 192 * real.sqrt 3 :=
by
  sorry

end cube_volume_surface_area_l305_305248


namespace segments_form_triangle_with_60_angle_l305_305940

noncomputable def square_side : ℝ := 1

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Square :=
  (A B C D : Point)

def is_square (A B C D : Point) : Prop :=
  (|B.x - A.x| = square_side ∧ B.y = A.y) ∧
  (|C.x - B.x| = 0 ∧ |C.y - B.y| = square_side) ∧
  (D.x = A.x ∧ D.y = C.y)

def on_side_BC (P : Point) (B C : Point) : Prop :=
  P.x = B.x ∧ B.y ≤ P.y ∧ P.y ≤ C.y

def on_side_CD (P : Point) (C D : Point) : Prop :=
  P.y = C.y ∧ C.x ≤ P.x ∧ P.x <= D.x

def total_side_length_eq (M N B C : Point) : Prop :=
  (|M.y - B.y| + |N.x - C.x| = square_side)

def create_triangle_with_angle_60 (A B C D M N : Point) : Prop :=
  ∃ P Q R : Point,
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
  (Complex.angle ((Q.x - P.x) + (Q.y - P.y) * I) ((R.x - Q.x) + (R.y - Q.y) * I) = (π / 3) ∨
   Complex.angle ((R.x - Q.x) + (R.y - Q.y) * I) ((P.x - R.x) + (P.y - R.y) * I) = (π / 3) ∨
   Complex.angle ((P.x - R.x) + (P.y - R.y) * I) ((Q.x - P.x) + (Q.y - P.y) * I) = (π / 3))

theorem segments_form_triangle_with_60_angle :
  ∀ (A B C D M N : Point),
  is_square A B C D →
  on_side_BC M B C →
  on_side_CD N C D →
  total_side_length_eq M N B C →
  create_triangle_with_angle_60 A B C D M N :=
by sorry

end segments_form_triangle_with_60_angle_l305_305940


namespace probability_four_heads_l305_305253

-- Definitions for use in the conditions
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def biased_coin (h : ℚ) (n k : ℕ) : ℚ :=
  binomial_coefficient n k * (h ^ k) * ((1 - h) ^ (n - k))

-- Condition: probability of getting heads exactly twice is equal to getting heads exactly three times.
def condition (h : ℚ) : Prop :=
  biased_coin h 5 2 = biased_coin h 5 3

-- Theorem to be proven: probability of getting heads exactly four times out of five is 5/32.
theorem probability_four_heads (h : ℚ) (cond : condition h) : biased_coin h 5 4 = 5 / 32 :=
by
  sorry

end probability_four_heads_l305_305253


namespace diego_annual_savings_l305_305340

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end diego_annual_savings_l305_305340


namespace sum_floor_log2_l305_305013

theorem sum_floor_log2 : (∑ N in Finset.range 513, Nat.floor (Real.log N / Real.log 2)) = 3604 :=
by
  sorry

end sum_floor_log2_l305_305013


namespace dot_product_value_l305_305079

-- Define the vectors a and b within some vector space
variables {V : Type*} [inner_product_space ℝ V]
variable (a b : V)

-- Define the conditions
def condition1 : Prop := (∥a + b∥ = real.sqrt 10)
def condition2 : Prop := (∥a - b∥ = real.sqrt 6)

-- State the theorem to prove
theorem dot_product_value (h1 : condition1 a b) (h2 : condition2 a b) : inner_product_space.inner a b = 1 := 
sorry

end dot_product_value_l305_305079


namespace heesu_has_greatest_sum_l305_305597

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l305_305597


namespace monotonically_increasing_iff_l305_305825

def f (a : ℝ) : Part ℝ := 
{ x | if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6) }

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → (f a).get x₁ ≤ (f a).get x₂) ↔ a ∈ Set.Icc (9 / 4) 3 := sorry

end monotonically_increasing_iff_l305_305825


namespace perimeter_of_AF1B_l305_305062

noncomputable def ellipse_perimeter (a b x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (2 * a)

theorem perimeter_of_AF1B (h : (6:ℝ) = 6) :
  ellipse_perimeter 6 4 0 0 6 0 = 24 :=
by
  sorry

end perimeter_of_AF1B_l305_305062


namespace ratio_of_areas_l305_305613

theorem ratio_of_areas (side_length : ℝ) (num_corrals : ℕ)
  (corral_perimeter : ℝ) (total_fencing : ℝ)
  (large_corral_side_length : ℝ) (small_corral_area : ℝ) (total_small_corrals_area : ℝ) (large_corral_area : ℝ) :
  side_length = 10 →
  num_corrals = 6 →
  corral_perimeter = 3 * side_length →
  total_fencing = num_corrals * corral_perimeter →
  large_corral_side_length = total_fencing / 3 →
  small_corral_area = (sqrt 3 / 4) * side_length^2 →
  total_small_corrals_area = num_corrals * small_corral_area →
  large_corral_area = (sqrt 3 / 4) * large_corral_side_length^2 →
  total_small_corrals_area / large_corral_area = 1 / 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end ratio_of_areas_l305_305613


namespace hypotenuse_length_l305_305485

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305485


namespace tan_sum_of_angles_l305_305805

-- Angle θ has its terminal side passing through point P(1, 2) and
-- the initial side on the non-negative half of the x-axis.
variable (θ : ℝ)

noncomputable def tan_of_point (x y : ℝ) : ℝ := y / x

theorem tan_sum_of_angles :
  let x := 1
  let y := 2
  let θ := real.arctan (tan_of_point x y)
  tan (θ + π / 4) = -3 :=
by {
  let x := 1,
  let y := 2,
  let θ := real.arctan (tan_of_point x y),
  have h_tan_θ := tan_of_point x y,
  have h_tan_sum : tan (θ + π / 4) = (1 + h_tan_θ) / (1 - h_tan_θ),
  rw [h_tan_θ, h_tan_sum],
  sorry
}

end tan_sum_of_angles_l305_305805


namespace part1_part2_l305_305808

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem part1 (x : ℝ) : (f x)^2 - (g x)^2 = -4 :=
by sorry

theorem part2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : 
  g (x + y) / g (x - y) = 3 :=
by sorry

end part1_part2_l305_305808


namespace solve_for_x_l305_305962

theorem solve_for_x (x : ℝ) (h : 2 ^ (16 ^ x^2) = 16 ^ (2 ^ x^2)) : 
  x = sqrt (2 / 3) ∨ x = -sqrt (2 / 3) :=
by 
  sorry

end solve_for_x_l305_305962


namespace proof_AP_squared_l305_305112

variables (A B C H D M N P : Type)
variables [Points : Point A] [Point B] [Point C] [Point H] [Point D] [Point M] [Point N] [Point P]
variables [Plane ℝ A B C]

-- Conditions
variables (AB : ℝ) (BC : ℝ) (AH : ℝ) (BM : ℝ) (CM : ℝ) (N_mid : Midpoint H M N)
variables (AP : ℝ)
variables [Angle A B 45] [Angle C A 30]

-- Given AB = 10, angle A = 45 degrees, and angle C = 30 degrees
axiom ab_eq : AB = 10
axiom ang_a_eq : angle_iso A = 45
axiom ang_c_eq : angle_iso C = 30

-- Given AH ⊥ BC, BM = CM, N is midpoint of HM, and AP calculated via midpoint properties
axiom ah_perp_bc : Perpendicular AH BC
axiom bm_eq_cm : BM = CM
axiom n_midpoint_hm : midpoint H M N
axiom pn_perp_bc : Perpendicular PN BC

-- Question
theorem proof_AP_squared : AP^2 = 50 :=
begin
  sorry
end

end proof_AP_squared_l305_305112


namespace problem_solution_l305_305036

theorem problem_solution (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : {d ∈ (nat.prime_divisors x).to_finset | true}.card = 3) (h3 : 31 ∈ nat.prime_divisors x) : x = 32767 :=
sorry

end problem_solution_l305_305036


namespace find_number_l305_305283

theorem find_number (X a b : ℕ) (hX : X = 10 * a + b) 
  (h1 : a * b = 24) (h2 : 10 * b + a = X + 18) : X = 46 :=
by
  sorry

end find_number_l305_305283


namespace decimal_of_fraction_has_denominator_three_l305_305972

def S : ℚ := (1/3 : ℚ)

theorem decimal_of_fraction_has_denominator_three :
  ((0.\overline{3}) = S) → (S.denom = 3) :=
by
  sorry

end decimal_of_fraction_has_denominator_three_l305_305972


namespace line_through_center_l305_305587

theorem line_through_center 
  (A B C M N O : Type*)
  (circumscribed_circle : Type*)
  (ratio_condition : ∀ AM BM CM AN BN CN : ℝ, 
    (AM / BM = AM / CM) ∧ (AN / BN = AN / CN)) 
  (lineMN_passes_O : ∀ P : Type*, line_MN P = line_MN O) :
  ∃ M N O, line_through M N O := 
sorry

end line_through_center_l305_305587


namespace rachel_one_hour_earnings_l305_305949

theorem rachel_one_hour_earnings :
  let hourly_wage := 12.00
  let number_of_people_served := 20
  let tip_per_person := 1.25
  let total_tips := number_of_people_served * tip_per_person
  let total_earnings := hourly_wage + total_tips
  in total_earnings = 37.00 :=
by
  sorry

end rachel_one_hour_earnings_l305_305949


namespace range_of_theta_l305_305916

noncomputable def function := λx: ℝ, -sqrt x * (x + 1)

theorem range_of_theta (x : ℝ) (h : x ≠ 0):
  let θ := Real.arctan (-((3*x + 1) / (2*sqrt x)))
  (π / 2 < θ ∧ θ ≤ 2*π / 3) :=
by
  sorry

end range_of_theta_l305_305916


namespace distinct_triangles_l305_305439

open Set

-- Define the grid of points with distinct coordinates
def points_grid : Set (ℝ × ℝ) := 
  {(0, 0), (1, 0), (2, 0), 
   (0, 1), (1, 1), (2, 1),
   (0, 2), (1, 2), (2, 2)}

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop := 
  ∃ (a b c : ℝ), a * (p2.1 - p1.1) + b * (p3.1 - p1.1) = 0 ∧ 
                  a * (p2.2 - p1.2) + b * (p3.2 - p1.2) = 0

-- Define the main theorem
theorem distinct_triangles : ∃ (t : ℕ), t = 82 :=
by
  have le_elems : points_grid = {
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2)
  } := by rfl
  -- Count unique triangles from the grid points excluding collinear points
  let all_combinations := (points_grid.toFinset.powersetLen 3).filter (λ subset, 
    match subset.toList with
    | [p1, p2, p3] => ¬ collinear p1 p2 p3
    | _ => false
    end)
  
  -- The total number of valid triangles
  let total_distinct_triangles := all_combinations.card

  -- Show that the total number of valid triangles is 82
  existsi total_distinct_triangles
  sorry -- Proof steps to be filled or verified through Lean

end distinct_triangles_l305_305439


namespace product_of_roots_l305_305460

noncomputable def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_roots :
  ∀ (x1 x2 : ℝ), is_root 1 (-4) 3 x1 ∧ is_root 1 (-4) 3 x2 ∧ x1 ≠ x2 → x1 * x2 = 3 :=
by
  intros x1 x2 h
  sorry

end product_of_roots_l305_305460


namespace angle_A_measure_perimeter_triangle_l305_305133

theorem angle_A_measure (a b c A B C : ℝ) (h1: sqrt 3 * a * sin C = c + c * cos A) : 
  A = π / 3 := 
  sorry

theorem perimeter_triangle (a b c A B C : ℝ) (h1: sqrt 3 * a * sin C = c + c * cos A)
  (h2: a = 2 * sqrt 3) (h3: (1 / 2) * b * c * sin A = sqrt 3) :
  b + c + a = 2 * sqrt 3 + 2 * sqrt 6 :=
  sorry

open Classical

noncomputable def sqrt (x : ℝ) : ℝ :=
if h : 0 ≤ x then real.sqrt x else 0

#eval angle_A_measure 0 0 0 0 0 0 sorry
#eval perimeter_triangle 0 0 0 0 0 0 sorry sorry sorry

end angle_A_measure_perimeter_triangle_l305_305133


namespace puppy_total_food_l305_305923

def daily_food_first_two_weeks : ℝ := (1 / 4) * 3
def total_food_first_two_weeks : ℝ := daily_food_first_two_weeks * 14

def daily_food_second_two_weeks : ℝ := (1 / 2) * 2
def total_food_second_two_weeks : ℝ := daily_food_second_two_weeks * 14

def food_today : ℝ := 1 / 2

def total_food_in_4_weeks : ℝ := food_today + total_food_first_two_weeks + total_food_second_two_weeks

theorem puppy_total_food (W: ℝ:= 25) : total_food_in_4_weeks = W :=
by 
    sorry

end puppy_total_food_l305_305923


namespace friendly_triangle_75_60_45_friendly_triangle_BC_70_65_45_l305_305020

variable (A B C A1 B1 C1 : ℝ)

-- Definitions for friendly triangle
def is_friendly_triangle (A B C A1 B1 C1 : ℝ) : Prop :=
  (cos A / sin A1 = 1 ∧ cos B / sin B1 = 1 ∧ cos C / sin C1 = 1)

-- First part: prove the given angles form a friendly triangle
theorem friendly_triangle_75_60_45 : is_friendly_triangle 75 60 45 (sqrt(6) - sqrt(2)) (1 / 2) (sqrt(2) / 2) := sorry

-- Second part: if A = 70 and the triangle is friendly, then B, C are 65 and 45
def friendly_triangle_A_70 (A B C A1 B1 C1 : ℝ) : Prop :=
  A = 70 ∧ is_friendly_triangle A B C A1 B1 C1

theorem friendly_triangle_BC_70_65_45 (A B C : ℝ) : 
  friendly_triangle_A_70 A B C (sin 20) (sin (90 - B)) (sin (B - 20)) → 
  (B = 65 ∧ C = 45) := sorry

end friendly_triangle_75_60_45_friendly_triangle_BC_70_65_45_l305_305020


namespace acute_angle_iff_median_gt_half_base_l305_305548

theorem acute_angle_iff_median_gt_half_base 
  {A B C A1 : ℝ} (h_med : ∃ (A1 : ℝ), A1 = (B + C) / 2)
  (h_median : ∃ (A A1 : ℝ), A > 0 ∧ A1 > 0 ∧ A1 = (B + C) / 2):
  (∠ BAC < π / 2) ↔ (median AA1 > (1 / 2) * distance B C) := 
sorry

end acute_angle_iff_median_gt_half_base_l305_305548


namespace solve_for_T_l305_305233

theorem solve_for_T : ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (2 / 5) * (1 / 4) * 200 ∧ T = 80 :=
by
  use 80
  -- The proof part is omitted as instructed
  sorry

end solve_for_T_l305_305233


namespace projections_are_concyclic_l305_305285

noncomputable def are_projections_concyclic (A B C D K L M N : Point) (α : Plane)
  (cond1 : Intersect α (Edge A B) K)
  (cond2 : Intersect α (Edge B C) L)
  (cond3 : Intersect α (Edge C D) M)
  (cond4 : Intersect α (Edge D A) N)
  (cond5 : DihedralAngle (Face KLA) (Face KLM) = 
           DihedralAngle (Face LMB) (Face LMN)
           ∧ DihedralAngle (Face LMB) (Face LMN) = 
           DihedralAngle (Face MNC) (Face MNK)
           ∧ DihedralAngle (Face MNC) (Face MNK) = 
           DihedralAngle (Face NKD) (Face NKL)) : Prop := 
  ∃ (A' B' C' D' : Point), 
  (Projection A α = A' ∧ Projection B α = B' ∧
   Projection C α = C' ∧ Projection D α = D') ∧
  Concyclic A' B' C' D'

theorem projections_are_concyclic (A B C D K L M N : Point) (α : Plane)
  (cond1 : Intersect α (Edge A B) K)
  (cond2 : Intersect α (Edge B C) L)
  (cond3 : Intersect α (Edge C D) M)
  (cond4 : Intersect α (Edge D A) N)
  (cond5 : DihedralAngle (Face KLA) (Face KLM) = 
           DihedralAngle (Face LMB) (Face LMN)
           ∧ DihedralAngle (Face LMB) (Face LMN) = 
           DihedralAngle (Face MNC) (Face MNK)
           ∧ DihedralAngle (Face MNC) (Face MNK) = 
           DihedralAngle (Face NKD) (Face NKL)) :
  are_projections_concyclic A B C D K L M N α cond1 cond2 cond3 cond4 cond5 :=
sorry

end projections_are_concyclic_l305_305285


namespace original_earnings_l305_305900

variable (x : ℝ) -- John's original weekly earnings

theorem original_earnings:
  (1.20 * x = 72) → 
  (x = 60) :=
by
  intro h
  sorry

end original_earnings_l305_305900


namespace smallest_difference_factors_1764_l305_305094

theorem smallest_difference_factors_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a ≠ 0 ∧ b ≠ 0 ∧ (∀ (x y : ℕ), x * y = 1764 → x ≠ 0 → y ≠ 0 → (a - b).natAbs ≤ (x - y).natAbs) ∧ (a - b).natAbs = 0 :=
by
  sorry

end smallest_difference_factors_1764_l305_305094


namespace domain_of_sqrt_expression_l305_305777

theorem domain_of_sqrt_expression :
  { x : ℝ | -8 * x ^ 2 - 10 * x + 12 ≥ 0 } = set.Icc (-2 : ℝ) (3 / 4 : ℝ) :=
sorry

end domain_of_sqrt_expression_l305_305777


namespace next_podcast_length_l305_305175

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end next_podcast_length_l305_305175


namespace Diego_annual_savings_l305_305341

theorem Diego_annual_savings :
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year
  annual_savings = 4800 :=
by
  -- Definitions based on the conditions and required result
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year

  -- Assertion to check the correctness of annual savings
  have h : annual_savings = 4800 := by
    have h1 : monthly_savings = monthly_income - monthly_expenses := rfl
    have h2 : monthly_savings = 400 := by simp [monthly_income, monthly_expenses, h1]
    have h3 : annual_savings = monthly_savings * months_in_year := rfl
    simp [h2, months_in_year, h3]
  exact h

end Diego_annual_savings_l305_305341


namespace leoCurrentWeight_l305_305449

def currentWeightProblem (L K : Real) : Prop :=
  (L + 15 = 1.75 * K) ∧ (L + K = 250)

theorem leoCurrentWeight (L K : Real) (h : currentWeightProblem L K) : L = 154 :=
by
  sorry

end leoCurrentWeight_l305_305449


namespace John_more_marbles_than_Ben_l305_305742

theorem John_more_marbles_than_Ben :
  let ben_initial := 18
  let john_initial := 17
  let ben_gave := ben_initial / 2
  let ben_final := ben_initial - ben_gave
  let john_final := john_initial + ben_gave
  john_final - ben_final = 17 :=
by
  sorry

end John_more_marbles_than_Ben_l305_305742


namespace ruth_total_points_l305_305331

variables (total_points_dean : ℕ) (games_dean : ℕ) (games_difference : ℕ) (score_increase : ℚ)

-- Conditions
def dean_conditions (total_points_dean games_dean : ℕ) : Prop :=
  total_points_dean = 252 ∧ games_dean = 28

def ruth_conditions (games_dean games_difference score_increase : ℕ) : Prop :=
  games_difference = 10 ∧ score_increase = 0.5

-- Proof problem statement
theorem ruth_total_points (total_points_dean games_dean games_difference : ℕ) (score_increase : ℚ)
  (h1 : dean_conditions total_points_dean games_dean)
  (h2 : ruth_conditions games_dean games_difference score_increase) :
  let dean_avg := total_points_dean.to_rat / games_dean in
  let ruth_avg := dean_avg + score_increase in
  let games_ruth := games_dean - games_difference in
  let total_points_ruth := ruth_avg * games_ruth in
  total_points_ruth = 171 :=
by
  sorry

end ruth_total_points_l305_305331


namespace non_degenerate_ellipse_l305_305618

theorem non_degenerate_ellipse (k : ℝ) : (∃ (x y : ℝ), x^2 + 4*y^2 - 10*x + 56*y = k) ↔ k > -221 :=
sorry

end non_degenerate_ellipse_l305_305618


namespace relationship_abc_l305_305024

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l305_305024


namespace light_bulb_installation_l305_305634

-- Define the problem statement in Lean
theorem light_bulb_installation:
  ∃ (n : ℕ), 
    let colors := 4 in
    let vertices := 6 in
    let different_colors_assigned (top_points : ℕ) := 24 in
    let remaining_color_assigned := 3 in
    let remaining_points := 3 in
    n = different_colors_assigned(vertices - 3) * remaining_color_assigned * remaining_points 
    ∧ n = 216 := 
by 
  sorry

end light_bulb_installation_l305_305634


namespace bathtub_problem_l305_305691

theorem bathtub_problem (T : ℝ) (h1 : 1 / T - 1 / 12 = 1 / 60) : T = 10 := 
by {
  -- Sorry, skip the proof as requested
  sorry
}

end bathtub_problem_l305_305691


namespace alex_singles_percentage_l305_305769

theorem alex_singles_percentage (total_hits home_runs triples doubles: ℕ) 
  (h1 : total_hits = 50) 
  (h2 : home_runs = 2) 
  (h3 : triples = 3) 
  (h4 : doubles = 10) :
  ((total_hits - (home_runs + triples + doubles)) / total_hits : ℚ) * 100 = 70 := 
by
  sorry

end alex_singles_percentage_l305_305769


namespace correct_total_distance_of_race_l305_305870

noncomputable def total_distance_of_race : ℕ :=
  let D := 200
  let A_beats_B_by := 56
  let A_beats_B_by_time := 7
  let A_time := 18
  have A_speed: ℚ := D / A_time
  have B_speed_1: ℚ := (D - A_beats_B_by) / A_time
  have B_speed_2: ℚ := D / (A_time + A_beats_B_by_time)
  have proof_eq := B_speed_1 = B_speed_2
  have solve_D := (25 * (D - A_beats_B_by)) = 18 * D
  D

theorem correct_total_distance_of_race : total_distance_of_race = 200 := by
  sorry

end correct_total_distance_of_race_l305_305870


namespace arithmetic_seq_a4_value_l305_305968

theorem arithmetic_seq_a4_value
  (a : ℕ → ℤ)
  (h : 4 * a 3 + a 11 - 3 * a 5 = 10) :
  a 4 = 5 := 
sorry

end arithmetic_seq_a4_value_l305_305968


namespace Pablo_winning_strategy_l305_305712

theorem Pablo_winning_strategy :
  ∀ (a : fin 2020 → ℕ), (∀ i j, 0 ≤ i → i < j → j < 2020 → a i ≤ a j) →
  ∃ strategy_Pablo, ∀ strategy_Diego , 
    ∃ (final_move : ℕ), 
      (∀ i, a i = 0) ∧
      (∀ i j, 0 ≤ i → i < j → j < 2020 → a(i) ≤ a(j)) :=
sorry

end Pablo_winning_strategy_l305_305712


namespace find_length_of_CD_l305_305129

-- Given problem's conditions
variable {A B C D : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable (AB BC AC : ℝ)
variable (CDangleBisectorOfC : Prop) -- CD is the angle bisector of angle C
variable (triangleABC : Prop) -- triangle ABC with given sides and bisector
variable (AB_eq_8 : AB = 8)
variable (BC_eq_17 : BC = 17)
variable (AC_eq_15 : AC = 15)

-- Problem statement
theorem find_length_of_CD : 
  ∃ CD : ℝ, CD = 16.459 ∧ 
  triangleABC ∧ 
  AB_eq_8 ∧ 
  BC_eq_17 ∧ 
  AC_eq_15 ∧ 
  CDangleBisectorOfC :=
sorry

end find_length_of_CD_l305_305129


namespace polygon_sides_l305_305580

-- Define the conditions
def sum_interior_angles (x : ℕ) : ℝ := 180 * (x - 2)
def sum_given_angles (x : ℕ) : ℝ := 160 + 112 * (x - 1)

-- State the theorem
theorem polygon_sides (x : ℕ) (h : sum_interior_angles x = sum_given_angles x) : x = 6 := by
  sorry

end polygon_sides_l305_305580


namespace largest_possible_good_number_l305_305389

theorem largest_possible_good_number :
  ∃ m, (∀ (a : ℝ) (s : multiset ℝ), 
          s = multiset.repeat 1 1000 ∨ 
          ∃ b : ℝ, s = (s.erase b).insert_n 3 (b / 3) → 
          multiset.card (multiset.filter (λ x, x = a) s) ≥ m) ∧ 
        m = 667 :=
by
  sorry

end largest_possible_good_number_l305_305389


namespace max_pretty_subset_cardinality_l305_305153

def is_pretty_subset (S : Finset ℕ) : Prop :=
  ∀ (a b ∈ S), a ≠ b → ¬ Nat.prime (abs (a - b))

theorem max_pretty_subset_cardinality :
  ∀ (S : Finset ℕ), S ⊆ Finset.range 201 →
  is_pretty_subset S →
  S.card ≤ 50 := by
  sorry

end max_pretty_subset_cardinality_l305_305153


namespace point_coordinates_l305_305515

theorem point_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : abs y = 5) (h4 : abs x = 2) : x = -2 ∧ y = 5 :=
by
  sorry

end point_coordinates_l305_305515


namespace find_principal_l305_305259

-- Define the principal amount (P), rate (R), and the conditions
variable {P R : ℝ}

-- Given condition: original time period is 5 years
def t : ℝ := 5

-- Given condition: if the rate of interest had been 6% higher, the interest earned would be 90 more
def condition1 : Prop := 
  (P * (R + 6) * t) / 100 = (P * R * t) / 100 + 90

-- Prove that the principal amount P is 300
theorem find_principal : condition1 → P = 300 :=
by
  -- Omitted the proof
  sorry

end find_principal_l305_305259


namespace sets_choice_number_l305_305152

theorem sets_choice_number (n : ℕ) : ∃ S : Fin (n+1) × Fin (n+1) → Finset (Fin (2 * n)),
  (∀ i j : Fin (n+1), (S i j).card = i + j) ∧
  (∀ i j k l : Fin (n+1), i ≤ k → j ≤ l → S i j ⊆ S k l) →
  (card (Finset.univ (Fin (2 * n))).choose n * 2^(n^2) * (2 * n)! = 2^(n^2) * nat.factorial (2 * n)) := by
sorry

end sets_choice_number_l305_305152


namespace work_rate_a_l305_305260

theorem work_rate_a (W : ℝ) (R_a R_b R_c : ℝ) (h₁: R_b = W / 6) (h₂: R_c = W / 12) (h₃: R_a + R_b + R_c = W / 3.2) : R_a = W / 16 :=
begin
  sorry
end

end work_rate_a_l305_305260


namespace gas_temperature_proportion_gas_work_ratio_l305_305227

noncomputable def gasTemperatureChangeFactor (initialVolume finalVolume: ℝ) (initialTemperature finalTemperatureExpansion finalTemperatureReduction: ℝ) (n: ℝ) : Prop :=
  let V := initialVolume
  let T_1 := initialTemperature
  let T_2 := finalTemperatureExpansion
  let T_3 := finalTemperatureReduction
  (finalVolume = n * initialVolume) ∧ 
  (n = 2) ∧
  (finalVolume = initialVolume) ∧
  (P * 2 * V / (P / 3 * V) = 6)

noncomputable def workRatio (workIsobaric: ℝ) (workVolumeReduction: ℝ) (initialVolume finalVolume: ℝ) (P: ℝ) (n: ℝ) : Prop :=
  let V := initialVolume
  (workIsobaric = P * V) ∧
  (workVolumeReduction = (2 * P / 3) * V) ∧
  (n = 2) ∧
  (finalVolume = 2 * V) ∧
  (workIsobaric / workVolumeReduction = 3 / 2)

theorem gas_temperature_proportion (initialVolume finalVolume: ℝ) (initialTemperature finalTemperatureExpansion finalTemperatureReduction: ℝ) (n : ℝ):
  gasTemperatureChangeFactor initialVolume finalVolume initialTemperature finalTemperatureExpansion finalTemperatureReduction n := 
  sorry

theorem gas_work_ratio (workIsobaric workVolumeReduction initialVolume finalVolume: ℝ) (P: ℝ) (n: ℝ) :
  workRatio workIsobaric workVolumeReduction initialVolume finalVolume P n := 
  sorry

end gas_temperature_proportion_gas_work_ratio_l305_305227


namespace rational_sum_p_q_l305_305056

noncomputable def x := (Real.sqrt 5 - 1) / 2

theorem rational_sum_p_q :
  ∃ (p q : ℚ), x^3 + p * x + q = 0 ∧ p + q = -1 := by
  sorry

end rational_sum_p_q_l305_305056


namespace hypotenuse_length_l305_305483

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305483


namespace tan_half_sum_of_roots_of_quadratic_l305_305401

theorem tan_half_sum_of_roots_of_quadratic (a : ℝ) (h1 : 1 < a) (α β : ℝ) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) 
  (h3 : β ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)) 
  (h4 : (x^2 + 4 * a * x + 3 * a + 1).roots = [Real.tan α, Real.tan β]) : 
  Real.tan ((α + β) / 2) = -2 := 
by 
  sorry

end tan_half_sum_of_roots_of_quadratic_l305_305401


namespace fencing_required_l305_305286

theorem fencing_required (length width area : ℕ) (length_eq : length = 30) (area_eq : area = 810) 
  (field_area : length * width = area) : 2 * length + width = 87 := 
by
  sorry

end fencing_required_l305_305286


namespace minimum_value_a_eq_2_range_of_a_l305_305404

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  real.sqrt (x^2 - 2*x + 1) + abs (x + a)

theorem minimum_value_a_eq_2 : 
  ∃ x : ℝ, f x 2 = 3 :=
begin
  use 1,
  unfold f,
  simp,
  -- Proper proof is omitted.
  sorry
end

theorem range_of_a (x : ℝ) (a : ℝ) (h : x ∈ set.Icc (2/3) 1) : 
  (∀ x, f x a ≤ x) ↔ a ∈ set.Icc (-1 : ℝ) (-1/3 : ℝ) :=
begin
  -- Proper proof is omitted.
  sorry
end

end minimum_value_a_eq_2_range_of_a_l305_305404


namespace tournament_mono_color_path_l305_305144

-- Define a type for vertices
variable {V : Type}

-- Define edge colors
inductive Color
| red | blue

-- Define the tournament graph structure
structure Tournament :=
  (vertices : set V)
  (edge : V → V → option Color)
  (complete : ∀ u v, u ≠ v → edge u v ≠ none)
  (asymmetry : ∀ u v, u ≠ v → (edge u v = some c → edge v u = none) ∨ (edge v u = some c → edge u v = none))

-- The theorem we need to prove
theorem tournament_mono_color_path (G : Tournament) :
  ∃ v : V, ∀ u : V, u ≠ v → ∃ c : Color, (G.edge v u = some c ∨ ∃ w, G.edge v w = some c ∧ G.edge w u = some c) :=
sorry

end tournament_mono_color_path_l305_305144


namespace geometric_sequence_and_general_formula_l305_305074

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+1) = (2/3) * a n + 2) (ha1 : a 1 = 7) : 
  ∃ r : ℝ, ∀ n, a n = r ^ (n-1) + 6 :=
sorry

end geometric_sequence_and_general_formula_l305_305074


namespace MarysTotalCandies_l305_305571

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end MarysTotalCandies_l305_305571


namespace next_podcast_duration_l305_305177

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l305_305177


namespace difference_of_squares_l305_305109

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 10) : 
  x^2 - y^2 = 200 := 
sorry

end difference_of_squares_l305_305109


namespace heesu_has_greatest_sum_l305_305596

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l305_305596


namespace range_of_f_l305_305827

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - real.sqrt (x - 1)

theorem range_of_f : set.range f = set.Ici (15/8) := by
  sorry

end range_of_f_l305_305827


namespace probability_of_winning_l305_305212

theorem probability_of_winning (P_lose : ℚ) (h : P_lose = 3 / 7) : 
  let P_win := 1 - P_lose in P_win = 4 / 7 :=
by
  sorry

end probability_of_winning_l305_305212


namespace chessboard_problem_l305_305322

noncomputable def num_rectangles (n : ℕ) : ℕ := Nat.choose n 2 * Nat.choose n 2
noncomputable def num_squares (n : ℕ) : ℕ := (List.range n).sum (λ k => (k + 1)^2)

theorem chessboard_problem :
  let r := num_rectangles 10
  let s := num_squares 10
  let ratio := s / r
  ratio = (7 : ℚ) / 37 ∧ 7 + 37 = 44 :=
by
  let r := num_rectangles 10
  let s := num_squares 10
  let ratio := s / r
  sorry

end chessboard_problem_l305_305322


namespace Lincoln_High_School_max_principals_l305_305561

def max_principals (total_years : ℕ) (term_length : ℕ) (max_principals_count : ℕ) : Prop :=
  ∀ (period : ℕ), period = total_years → 
                  term_length = 4 → 
                  max_principals_count = 3

theorem Lincoln_High_School_max_principals 
  (total_years term_length max_principals_count : ℕ) :
  max_principals total_years term_length max_principals_count :=
by 
  intros period h1 h2
  have h3 : period = 10 := sorry
  have h4 : term_length = 4 := sorry
  have h5 : max_principals_count = 3 := sorry
  sorry

end Lincoln_High_School_max_principals_l305_305561


namespace coefficient_of_1_over_x_is_minus_29_l305_305196

theorem coefficient_of_1_over_x_is_minus_29 :
  ∑ k in finset.range 5, nat.choose 4 k * nat.choose 9 (4 - k) * (-1)^k = -29 :=
by sorry

end coefficient_of_1_over_x_is_minus_29_l305_305196


namespace difference_of_distinct_members_set_l305_305431

theorem difference_of_distinct_members_set :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≤ n → ∃ a b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), a ≠ b ∧ m = a - b ∨ m = b - a ∧ a > b) :=
by
  sorry

end difference_of_distinct_members_set_l305_305431


namespace square_area_l305_305576

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem square_area (p1 p2 : ℝ × ℝ) (h : p1 = (4,3) ∧ p2 = (5,7)) : 
  (distance p1 p2)^2 = 17 :=
by
  sorry

end square_area_l305_305576


namespace find_CE_l305_305132

theorem find_CE (BC BD DC DE : ℕ) (hBC : BC = 60) (hBD : BD = 42) (hDC : DC = 63) (hDE : DE = 21) :
  let CE := DC - DE in
  CE = 42 :=
by
  sorry

end find_CE_l305_305132


namespace possibleGeometricSolids_l305_305450

-- Define the geometric solids
inductive GeometricSolid
| cube
| cylinder
| cone
| triangularPrism

open GeometricSolid

-- Define condition that a geometric solid can have a circular cross-section
def hasCircularCrossSection : GeometricSolid → Prop
| cube            := false
| cylinder        := true
| cone            := true
| triangularPrism := false

-- Define the proof problem as a theorem statement
theorem possibleGeometricSolids : 
  {s : GeometricSolid | hasCircularCrossSection s} = {cylinder, cone} := by
  sorry

end possibleGeometricSolids_l305_305450


namespace Oshea_needs_50_small_planters_l305_305164

structure Planter :=
  (large : ℕ)     -- Number of large planters
  (medium : ℕ)    -- Number of medium planters
  (small : ℕ)     -- Number of small planters
  (capacity_large : ℕ := 20) -- Capacity of large planter
  (capacity_medium : ℕ := 10) -- Capacity of medium planter
  (capacity_small : ℕ := 4)  -- Capacity of small planter

structure Seeds :=
  (basil : ℕ)     -- Number of basil seeds
  (cilantro : ℕ)  -- Number of cilantro seeds
  (parsley : ℕ)   -- Number of parsley seeds

noncomputable def small_planters_needed (planters : Planter) (seeds : Seeds) : ℕ :=
  let basil_in_large := min seeds.basil (planters.large * planters.capacity_large)
  let basil_left := seeds.basil - basil_in_large
  let basil_in_medium := min basil_left (planters.medium * planters.capacity_medium)
  let basil_remaining := basil_left - basil_in_medium
  
  let cilantro_in_medium := min seeds.cilantro ((planters.medium * planters.capacity_medium) - basil_in_medium)
  let cilantro_remaining := seeds.cilantro - cilantro_in_medium
  
  let parsley_total := seeds.parsley + basil_remaining + cilantro_remaining
  parsley_total / planters.capacity_small

theorem Oshea_needs_50_small_planters :
  small_planters_needed 
    { large := 4, medium := 8, small := 0 }
    { basil := 200, cilantro := 160, parsley := 120 } = 50 := 
sorry

end Oshea_needs_50_small_planters_l305_305164


namespace sum_of_first_49_primes_is_10787_l305_305356

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                                167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]

theorem sum_of_first_49_primes_is_10787:
  first_49_primes.sum = 10787 :=
by
  -- Proof would go here.
  -- This is just a placeholder as per the requirements.
  sorry

end sum_of_first_49_primes_is_10787_l305_305356


namespace solution_set_l305_305393

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom monotone_increasing : ∀ x y, x < y → f x ≤ f y
axiom f_at_3 : f 3 = 2

-- Proof statement
theorem solution_set : {x : ℝ | -2 ≤ f (3 - x) ∧ f (3 - x) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

end solution_set_l305_305393


namespace negation_proposition_l305_305983

theorem negation_proposition (x y : ℝ) :
  (¬ ∃ (x y : ℝ), 2 * x + 3 * y + 3 < 0) ↔ (∀ (x y : ℝ), 2 * x + 3 * y + 3 ≥ 0) :=
by {
  sorry
}

end negation_proposition_l305_305983


namespace inequality_proof_l305_305919

variables {Ω : Type*} [ProbabilitySpace Ω]
variable (X : Ω → ℝ)

-- Given conditions
axiom exp_X_zero : E[X] = 0
axiom prob_X_le_one : ∀ ω, X ω ≤ 1

theorem inequality_proof :
  (E[|X|])^2 / 2 ≤ - E[λ ω, Real.log (1 - X ω)] :=
by sorry

end inequality_proof_l305_305919


namespace sum_trig_eq_52_l305_305744

theorem sum_trig_eq_52 : 
  (∑ x in finset.range 49, (2 * real.sin (x+2) * real.sin 2) * (1 + (real.sec (x) * real.sec (x+4)))) 
  =
  (∑ n in finset.range 2, (-1)^n * ((1 - real.cos (θ n)) / (1 - real.sec (θ n)))) 
  :=
sorry

end sum_trig_eq_52_l305_305744


namespace least_number_to_add_l305_305243

theorem least_number_to_add (x : ℕ) : (1056 + x) % 28 = 0 ↔ x = 4 :=
by sorry

end least_number_to_add_l305_305243


namespace solved_only_B_is_six_l305_305466

/-- Definitions to represent number of students who solved each subset of problems -/
def total_students := 25
def students_solved_A_only (a : ℕ) := true
def students_solved_B_only (b : ℕ) := true
def students_solved_C_only (c : ℕ) := true
def students_solved_A_and_B_not_C (d : ℕ) := true
def students_solved_A_and_C_not_B  (e : ℕ) := true
def students_solved_B_and_C_not_A  (f : ℕ) := true
def students_solved_ABC (g : ℕ) := true

/-- Condition 1: There are 25 students in total -/
def cond1 (a b c d e f g : ℕ) := a + b + c + d + e + f + g = total_students

/-- Condition 2: Each student solved at least one problem -/
-- By system definition less than one solving is infeasible hence implicit

/-- Condition 3: The number of students who solved problem B but not A is twice 
the number of students who solved problem C but not A -/
def cond3 (c d g : ℕ) := c + d = 2 * (d + g)

/-- Condition 4: The number of students who solved only problem A 
is one more than those who solved problem A among remaining set -/
def cond4 (a b e f : ℕ) := a = b + e + f + 1 

/-- Condition 5: Among students who solved one problem, half did not solve problem A -/
def cond5 (a b c : ℕ) := a + b + c = 2 * (b + c)

/-- Proof problem: Given conditions, prove required question result -/
theorem solved_only_B_is_six : ∃ a b c d e f g : ℕ,
  students_solved_A_only a ∧
  students_solved_B_only b ∧
  students_solved_C_only c ∧
  students_solved_A_and_B_not_C d ∧
  students_solved_A_and_C_not_B e ∧
  students_solved_B_and_C_not_A f ∧
  students_solved_ABC g ∧
  cond1 a b c d e f g ∧
  cond3 c d g ∧
  cond4 a b e f ∧
  cond5 a b c ∧
  b = 6 :=
by {
  existsi 1, existsi 6, existsi 4, existsi 2, existsi 4, existsi 7, existsi 1,
  split, trivial, split, trivial, split, trivial, split, trivial,
  split, trivial, split, trivial, split, trivial,
  repeat { sorry }
}

end solved_only_B_is_six_l305_305466


namespace triangle_side_length_l305_305893

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem triangle_side_length
  (h1 : a = Real.sqrt 3)
  (h2 : Real.sin A = Real.sqrt 3 / 2)
  (h3 : B = π / 6) :
  b = 1 :=
by
  sorry

end triangle_side_length_l305_305893


namespace complement_domain_l305_305159

open Set Real

noncomputable def U : Set ℝ := {x | x > 0}

noncomputable def f (x : ℝ) : ℝ := 1 / sqrt (1 - log x)

theorem complement_domain :
  let A := {x : ℝ | 0 < x ∧ x < exp 1} in
  U \ A = {x : ℝ | exp 1 ≤ x} :=
by
  sorry

end complement_domain_l305_305159


namespace pascal_triangle_even_count_l305_305086

/--
The number of even integers present in the top 15 rows of Pascal's Triangle is 213.
-/
theorem pascal_triangle_even_count : 
  (∑ n in finset.range 15, finset.filter (λ k, (binomial n k) % 2 = 0) (finset.range (n + 1)).card) = 213 :=
sorry

end pascal_triangle_even_count_l305_305086


namespace complex_modulus_proof_l305_305399

noncomputable def z : ℂ := (4 - 2 * complex.I) / (1 + complex.I)

theorem complex_modulus_proof : complex.abs z = real.sqrt 10 :=
by
  sorry

end complex_modulus_proof_l305_305399


namespace no_couples_next_to_each_other_l305_305513

def factorial (n: Nat): Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (m n p q: Nat): Nat :=
  factorial m - n * factorial (m - 1) + p * factorial (m - 2) - q * factorial (m - 3)

theorem no_couples_next_to_each_other :
  arrangements 7 8 24 32 + 16 * factorial 3 = 1488 :=
by
  -- Here we state that the calculation of special arrangements equals 1488.
  sorry

end no_couples_next_to_each_other_l305_305513


namespace compare_a_b_c_l305_305022

noncomputable
def a : ℝ := Real.exp 0.1 - 1

def b : ℝ := 0.1

noncomputable
def c : ℝ := Real.log 1.1

theorem compare_a_b_c : a > b ∧ b > c := by
  sorry

end compare_a_b_c_l305_305022


namespace binom_div_l305_305786

-- Define the generalized binomial coefficient for any real number a and positive integer k
noncomputable def binom (a : ℝ) (k : ℕ) : ℝ :=
  (List.prod (List.iota k).map (λ i, a - i)) / (Nat.factorial k)

-- Define the exact problem to solve in Lean 4 statement
theorem binom_div (a : ℝ) :
  (binom (-1/2) 100) / (binom (1/2) 100) = -199 :=
sorry

end binom_div_l305_305786


namespace area_abc_nine_x_l305_305527

-- Defining points and segments on triangle ABC
variables {A B C G D E R: Type}
variables [pt : Triangle A B C] [centroid G A B C] [midpoint D B C] [midpoint E A B]
variables [line DE : Line E D] [line CN : Line C N] [intersection R DE CN]
variables {x : ℝ}

-- The area of triangle GRD is x
variables [area_grd : Area (Triangle G R D) = x]

-- The ratio of GR to RD is 1:3
variables [ratio_gr_rd : Ratio Segment (Segment G R) (Segment R D) 1 3]

-- Theorem statement
theorem area_abc_nine_x (h1 : intersection G AM CN)
                       (h2 : midpoint D B C)
                       (h3 : midpoint E A B)
                       (h4 : intersection R DE CN)
                       (h5 : ratio_gr_rd = (1 : ℝ) / (3 : ℝ))
                       (h6 : Area (Triangle G R D) = x) :
  Area (Triangle A B C) = 9 * x :=
sorry


end area_abc_nine_x_l305_305527


namespace solve_inequality_l305_305964

theorem solve_inequality (a x : ℝ) :
  (a = 0 → x < 1) ∧
  (a ≠ 0 → ((a > 0 → (a-1)/a < x ∧ x < 1) ∧
            (a < 0 → (x < 1 ∨ x > (a-1)/a)))) :=
by
  sorry

end solve_inequality_l305_305964


namespace proof_sum_middle_m_terms_l305_305738

variables {m x d : ℕ} -- Declare variables for m, x, and d

-- Given conditions as hypotheses
def arithmetic_sequence_sum_first_2m := 2 * x + d = 100
def arithmetic_sequence_sum_last_2m := 2 * x + 3 * d = 200

-- Definition of sum for middle m terms
def sum_middle_m_terms := x + d

-- Main theorem stating the sum of the middle m terms is 75
theorem proof_sum_middle_m_terms (am_sum_first_2m : arithmetic_sequence_sum_first_2m) 
                                 (am_sum_last_2m : arithmetic_sequence_sum_last_2m) : 
  sum_middle_m_terms = 75 := 
sorry -- Proof to be completed

end proof_sum_middle_m_terms_l305_305738


namespace sum_distances_l305_305578

noncomputable def lengthAB : ℝ := 2
noncomputable def lengthA'B' : ℝ := 5
noncomputable def midpointAB : ℝ := lengthAB / 2
noncomputable def midpointA'B' : ℝ := lengthA'B' / 2
noncomputable def distancePtoD : ℝ := 0.5
noncomputable def proportionality_constant : ℝ := lengthA'B' / lengthAB

theorem sum_distances : distancePtoD + (proportionality_constant * distancePtoD) = 1.75 := by
  sorry

end sum_distances_l305_305578


namespace ratio_area_l305_305525

-- Define the lengths of the bases
def AB : ℝ := 10
def CD : ℝ := 20

-- Define a condition that establishes similarity based on the given lengths
def similar_triangles (AB CD : ℝ) : Prop := (CD / AB) = 2

-- Define the areas of the geometrical shapes involved
def area_EAB (AB CD : ℝ) : ℝ
def area_ABCD (AB CD : ℝ) : ℝ

-- Define the condition on the areas based on similarity
axiom area_condition (AB CD : ℝ) : area_EAB AB CD + area_ABCD AB CD = area_EAB AB CD * 4

-- Prove the required ratio of areas
theorem ratio_area (AB CD : ℝ) (h_similar: similar_triangles AB CD) (h_area: area_condition AB CD) : 
(area_EAB AB CD) / (area_ABCD AB CD) = 1 / 3 := 
by
  sorry

end ratio_area_l305_305525


namespace least_not_lucky_multiple_of_8_l305_305244

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

def is_multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_not_lucky_multiple_of_8 : ∃ (n : ℕ), is_multiple_of_8 n ∧ ¬is_lucky_integer n ∧ ∀ (m : ℕ), is_multiple_of_8 m ∧ ¬is_lucky_integer m → n ≤ m :=
  ∃ n, is_multiple_of_8 n ∧ ¬is_lucky_integer n ∧ (∀ m, is_multiple_of_8 m ∧ ¬is_lucky_integer m → n = 16) :=
    sorry

end least_not_lucky_multiple_of_8_l305_305244


namespace recurrence_solution_l305_305963

noncomputable def recurrence_a : ℕ → ℕ
| 0       := 5
| 1       := 10
| n + 2 := 5 * recurrence_a (n + 1) - 6 * recurrence_a n + 2 * (n + 2) - 3

theorem recurrence_solution (n : ℕ) : 
  recurrence_a n = 2 ^ (n + 1) + 3 ^ n + n + 2 :=
sorry

end recurrence_solution_l305_305963


namespace solution_correct_l305_305090

noncomputable def number_of_ways_to_partition : ℕ :=
  let S : ℕ → ℕ := λ n, (2^(n+1) - 1) in -- sum of series 2^0, 2^1, ..., 2^n
  let set_exponentials : Finset ℕ := Finset.range 2006 in
  let partitions (A B : Finset ℕ) := A \cap B = ∅ ∧ A ∪ B = set_exponentials ∧ A ≠ ∅ ∧ B ≠ ∅ in
  let satisfies_condition (A B : Finset ℕ) := ∃ x1 x2 : ℤ, x1 + x2 = S (A.card) ∧ x1 * x2 = S (B.card) in
  (Finset.univ.subset set_exponentials).card

theorem solution_correct : number_of_ways_to_partition = 1003 :=
sorry

end solution_correct_l305_305090


namespace centrifugal_force_and_weight_l305_305653

noncomputable def centrifugal_force_at_equator (r : ℝ) (T : ℝ) : ℝ :=
  4 * real.pi^2 * r / T^2

noncomputable def centrifugal_force_at_mountain (r h : ℝ) (T : ℝ) : ℝ :=
  4 * real.pi^2 * (r + h) / T^2

noncomputable def gravitational_acceleration_decrease (g0 r h : ℝ) : ℝ :=
  g0 * (1 - 2 * h / r)

noncomputable def weight_of_person (m g1 Cf : ℝ) : ℝ :=
  m * (g1 - Cf)

variables {r : ℝ} {h : ℝ} {T : ℝ} {g0 : ℝ} {m : ℝ}

theorem centrifugal_force_and_weight :
  r = 637000000 ∧ h = 800000 ∧ T = 86164 ∧ g0 = 980 ∧ m = 75000 →
  centrifugal_force_at_equator r T ≈ 3.387 ∧
  centrifugal_force_at_mountain r h T ≈ 3.392 ∧
  (gravitational_acceleration_decrease g0 r h ≈ g0 * (1 - 2 * h / r)) ∧
  (g0 - gravitational_acceleration_decrease g0 r h ≈ 0.25) ∧
  weight_of_person m (gravitational_acceleration_decrease g0 r h) (centrifugal_force_at_mountain r h T) / 981000 ≈ 74.62 :=
by { sorry }

end centrifugal_force_and_weight_l305_305653


namespace no_zero_sum_for_odd_n_at_least_six_zero_sum_labels_for_even_n_l305_305543

theorem no_zero_sum_for_odd_n (n : ℕ) (h_n : n ≥ 2) (h_odd : odd n) :
  ∀ (grid : fin n → fin n → ℤ), (∀ i j, grid i j = 1 ∨ grid i j = -1) →
    let row_sum := λ i, finset.sum (finset.univ : finset (fin n)) (λ j, grid i j) in
    let col_sum := λ j, finset.sum (finset.univ : finset (fin n)) (λ i, grid i j) in
    let S_n := finset.sum (finset.univ : finset (fin n)) row_sum + 
               finset.sum (finset.univ : finset (fin n)) col_sum in
    S_n ≠ 0 := 
sorry

theorem at_least_six_zero_sum_labels_for_even_n (n : ℕ) (h_n : n ≥ 2) (h_even : even n) :
  ∃ (labels : finset (fin n → fin n → ɛ)),
    labels.card ≥ 6 ∧ 
    (∀ grid ∈ labels, (∀ i j, grid i j = 1 ∨ grid i j = -1) ∧
    let row_sum := λ i, finset.sum (finset.univ : finset (fin n)) (λ j, grid i j) in
    let col_sum := λ j, finset.sum (finset.univ : finset (fin n)) (λ i, grid i j) in
    let S_n := finset.sum (finset.univ : finset (fin n)) row_sum + 
               finset.sum (finset.univ : finset (fin n)) col_sum in
    S_n = 0) := 
sorry

end no_zero_sum_for_odd_n_at_least_six_zero_sum_labels_for_even_n_l305_305543


namespace twin_prime_trio_unique_l305_305588

open Nat

def is_twin_prime (p q : ℕ) : Prop :=
  p + 2 = q ∧ Prime p ∧ Prime q

def is_prime_trio (p : ℕ) : Prop :=
  Prime p ∧ Prime (p + 2) ∧ Prime (p + 4)

theorem twin_prime_trio_unique :
  ∀ p, is_prime_trio p → p = 3 ∧ p+2 = 5 ∧ p+4 = 7 :=
by
  intro p
  intro h
  have hp : Prime p := h.1
  have hp2 : Prime (p + 2) := h.2.1
  have hp4 : Prime (p + 4) := h.2.2
  sorry

end twin_prime_trio_unique_l305_305588


namespace geometric_progression_example_l305_305767

theorem geometric_progression_example (x : ℝ) (r : ℝ) :
  let a := 10 + x
      b := 30 + x
      c := 90 + x in
  x = 0 → b^2 = a * c ∧ r = b / a ∧ r = c / b :=
begin
  intro h,
  rw h,
  dsimp [a, b, c],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end geometric_progression_example_l305_305767


namespace incorrect_propositions_count_l305_305735

theorem incorrect_propositions_count :
  let p1 := (∀ x, x^2 - 3 * x + 2 = 0 → x = 1) →
             (∀ x, x ≠ 1 → x^2 - 3 * x + 2 ≠ 0)
  let p2 := (∀ x, x^2 - 3 * x + 2 > 0 → x > 2)
  let p3 := (¬(p ∧ q) → ¬p ∧ ¬q)
  let p4 := (∃ x ∈ ℝ, x^2 + x + 1 < 0) → (∀ x ∈ ℝ, x^2 + x + 1 ≥ 0)
  2 = (ite (¬p1) 1 0) + (ite (¬p2) 1 0) + (ite (¬p3) 1 0) + (ite (¬p4) 1 0) :=
by
  sorry

end incorrect_propositions_count_l305_305735


namespace incorrect_statement_l305_305301

theorem incorrect_statement :
  ¬ (∀ (l1 l2 l3 : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), l3 x y → l1 x y) ∧ 
      (∀ (x y : ℝ), l3 x y → l2 x y) → 
      (∀ (x y : ℝ), l1 x y → l2 x y)) :=
by sorry

end incorrect_statement_l305_305301


namespace values_of_b_l305_305750

noncomputable def proof_problem (x y b : ℝ) : Prop :=
    (sqrt (x + y) = b^b) ∧ (log b (x^2 * y) + log b (y^2 * x) = 3 * b^3)

theorem values_of_b (x y b : ℝ) : proof_problem x y b → b > 0 :=
begin
    sorry
end

end values_of_b_l305_305750


namespace total_distance_traveled_is_960_l305_305693

-- Definitions of conditions
def first_day_distance : ℝ := 100
def second_day_distance : ℝ := 3 * first_day_distance
def third_day_distance : ℝ := second_day_distance + 110
def fourth_day_distance : ℝ := 150

-- The total distance traveled in four days
def total_distance : ℝ := first_day_distance + second_day_distance + third_day_distance + fourth_day_distance

-- Theorem statement
theorem total_distance_traveled_is_960 :
  total_distance = 960 :=
by
  sorry

end total_distance_traveled_is_960_l305_305693


namespace three_points_not_collinear_of_four_not_coplanar_l305_305048

/-- Definition of four points that are not coplanar -/
def points_not_coplanar (A B C D : ℝ × ℝ × ℝ) : Prop :=
  ¬∃ x y z : ℝ, ∀ (P : ℝ × ℝ × ℝ), P = A ∨ P = B ∨ P = C ∨ P = D → (x * P.1 + y * P.2 + z * P.3 = 1)

/-- Definition of three points that are not collinear -/
def points_not_collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ¬∃ x y z : ℝ, ∀ (P : ℝ × ℝ × ℝ), P = A ∨ P = B ∨ P = C → (x * P.1 + y * P.2 + z * P.3 = 1)

/-- Main theorem statement -/
theorem three_points_not_collinear_of_four_not_coplanar (A B C D : ℝ × ℝ × ℝ)
  (h : points_not_coplanar A B C D) :
  ∃ P Q R : ℝ × ℝ × ℝ, (P = A ∨ P = B ∨ P = C ∨ P = D) ∧
  (Q = A ∨ Q = B ∨ Q = C ∨ Q = D) ∧
  (R = A ∨ R = B ∨ R = C ∨ R = D) ∧
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
  points_not_collinear P Q R :=
sorry

end three_points_not_collinear_of_four_not_coplanar_l305_305048


namespace total_surface_area_of_pyramid_l305_305292

noncomputable def base_length_ab : ℝ := 8 -- Length of side AB
noncomputable def base_length_ad : ℝ := 6 -- Length of side AD
noncomputable def height_pf : ℝ := 15 -- Perpendicular height from peak P to the base's center F

noncomputable def base_area : ℝ := base_length_ab * base_length_ad
noncomputable def fm_distance : ℝ := Real.sqrt ((base_length_ab / 2)^2 + (base_length_ad / 2)^2)
noncomputable def slant_height_pm : ℝ := Real.sqrt (height_pf^2 + fm_distance^2)

noncomputable def lateral_area_ab : ℝ := 2 * (0.5 * base_length_ab * slant_height_pm)
noncomputable def lateral_area_ad : ℝ := 2 * (0.5 * base_length_ad * slant_height_pm)
noncomputable def total_surface_area : ℝ := base_area + lateral_area_ab + lateral_area_ad

theorem total_surface_area_of_pyramid :
  total_surface_area = 48 + 55 * Real.sqrt 10 := by
  sorry

end total_surface_area_of_pyramid_l305_305292


namespace vincent_total_packs_l305_305668

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l305_305668


namespace unique_positive_solution_l305_305335

noncomputable def equation (x : ℝ) : Prop :=
  Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x

theorem unique_positive_solution :
  ∃! x ∈ Set.Icc 0 1, equation x ∧ 0 < x := by
  sorry

end unique_positive_solution_l305_305335


namespace simplify_complex_subtraction_l305_305184

theorem simplify_complex_subtraction : (3 - 2 * Complex.i) - (5 - 2 * Complex.i) = -2 :=
by
  sorry

end simplify_complex_subtraction_l305_305184


namespace find_m_l305_305689

theorem find_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x < 0) → m = -1 :=
by sorry

end find_m_l305_305689


namespace arithmetic_sequence_a6_l305_305881

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h₀ : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
  (h₁ : ∃ x y : ℝ, x = a 4 ∧ y = a 8 ∧ (x^2 - 4 * x - 1 = 0) ∧ (y^2 - 4 * y - 1 = 0) ∧ (x + y = 4)) :
  a 6 = 2 := 
sorry

end arithmetic_sequence_a6_l305_305881


namespace sum_of_first_49_primes_l305_305359

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l305_305359


namespace find_b_l305_305111

-- Defining the conditions
variables {A B C : ℝ}
variable {a : ℝ}
variable {b : ℝ}

-- Given conditions
axiom angle_sum : A + B + C = 180
axiom b_given : B = 60
axiom c_given : C = 75
axiom a_given : a = 4

-- Required proof
theorem find_b : b = 2 * real.sqrt 6 :=
by
  -- Definitions from the conditions
  have A_val : A = 180 - B - C := by linarith [angle_sum, b_given, c_given]
  have law_of_sines : a / real.sin (A * real.pi / 180) = b / real.sin (B * real.pi / 180) := sorry
  have sin_45 : real.sin (A * real.pi / 180) = real.sqrt 2 / 2 := sorry
  have sin_60 : real.sin (B * real.pi / 180) = real.sqrt 3 / 2 := sorry
  rw a_given at law_of_sines
  rw [A_val, b_given, c_given] at law_of_sines
  simp only [sin_45, sin_60] at law_of_sines
  -- Algebraic manipulation to solve for b
  sorry  -- The actual algebraic steps to isolate b and show it equals 2sqrt(6)

end find_b_l305_305111


namespace relationship_among_a_b_c_l305_305832

noncomputable def a : ℝ := 0.3^2
noncomputable def b : ℝ := Real.logBase 2 0.3
noncomputable def c : ℝ := 2^0.3

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  -- Define a
  let a := 0.3^2
  -- Define b
  let b := Real.logBase 2 0.3
  -- Define c
  let c := 2^0.3
  -- Define relationships
  have h1: 0 < a := sorry
  have h2: a < 1 := sorry
  have h3: b < 0 := sorry
  have h4: c > 1 := sorry
  -- Combine all relationships to prove the theorem
  exact And.intro (lt_trans h3 h1) (lt_trans h1 h4)

end relationship_among_a_b_c_l305_305832


namespace geometric_mean_arithmetic_mean_log_l305_305791

theorem geometric_mean_arithmetic_mean_log (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  sqrt (x1 * x2) < (x1 - x2) / (Real.log x1 - Real.log x2) ∧ (x1 - x2) / (Real.log x1 - Real.log x2) < (x1 + x2) / 2 :=
by
  sorry

end geometric_mean_arithmetic_mean_log_l305_305791


namespace curve_C1_equation_curve_C2_equation_curve_C2_center_curve_C3_transformation_max_distance_to_C1_l305_305064

theorem curve_C1_equation (t : ℝ) : ∀ x y, (x = 2 * t + 1 ∧ y = 2 * t - 1) → (x - y - 2 = 0) :=
by sorry

theorem curve_C2_equation (a θ : ℝ) (h : a > 0) : 
  ∀ ρ, (ρ = 2 * a * cos θ) → (x^2 + y^2 - 4 * x = 0) :=
by sorry

theorem curve_C2_center (a : ℝ) (h : a > 0) : 
  ∀ x y, ((x - 2)^2 + y^2 = 4) :=
by sorry

theorem curve_C3_transformation : 
  ∀ x y, ((x - 2)^2 + y^2 = 4) → (x' = \frac{1}{2}x ∧ y' = \frac{\sqrt{3}}{2}y) → (x^2 + \frac{y^2}{3} = 1) :=
by sorry

theorem max_distance_to_C1 {θ:ℝ} : 
  ∀ P, (P = (cos θ, \sqrt{3} * sin θ)) → (max_d = \frac{|2sin(θ - \frac{π}{6}) + 2|}{\sqrt{2}}) → (θ = \frac{2π}{3}) → (max_d = 2\sqrt{2}) :=
by sorry

end curve_C1_equation_curve_C2_equation_curve_C2_center_curve_C3_transformation_max_distance_to_C1_l305_305064


namespace nested_fraction_eval_l305_305000

theorem nested_fraction_eval : (1 / (1 + (1 / (2 + (1 / (1 + (1 / 4))))))) = (14 / 19) :=
by
  sorry

end nested_fraction_eval_l305_305000


namespace range_of_a_l305_305863

theorem range_of_a (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (∀ x ∈ (set.Ioo 0 2), (log a (1 - 3 * a * x) > log a (1 - 3 * a * y)) → (x > y)) ↔ (0 < a ∧ a ≤ 1 / 6) :=
by
  sorry

end range_of_a_l305_305863


namespace find_n_l305_305223

theorem find_n (y n : ℤ) (h_n_gt_4 : n > 4)
  (h_sum_eq : ∑ i in finset.range (n + 1), (y + 3 * i)^3 = -6859) : 
  n = 4 :=
sorry

end find_n_l305_305223


namespace max_sum_of_xj4_minus_xj5_l305_305787

theorem max_sum_of_xj4_minus_xj5 (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ j => (x j)^4 - (x j)^5)) ≤ 1 / 12 :=
sorry

end max_sum_of_xj4_minus_xj5_l305_305787


namespace trapezoid_area_l305_305892

-- Definitions based on conditions
def CL_div_LD (CL LD : ℝ) : Prop := CL / LD = 1 / 4

-- The main statement we want to prove
theorem trapezoid_area (BC CD : ℝ) (h1 : BC = 9) (h2 : CD = 30) (CL LD : ℝ) (h3 : CL_div_LD CL LD) : 
  1/2 * (BC + AD) * 24 = 972 :=
sorry

end trapezoid_area_l305_305892


namespace ravi_prakash_finish_together_l305_305263

theorem ravi_prakash_finish_together (ravi_days prakash_days : ℕ) (h_ravi : ravi_days = 15) (h_prakash : prakash_days = 30) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 10 := 
by
  sorry

end ravi_prakash_finish_together_l305_305263


namespace a_sufficient_but_not_necessary_l305_305096

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → a^2 - 3 * a + 2 = 0) ∧ (a^2 - 3 * a + 2 = 0 → a = 1 ∨ a = 2) :=
by
  intro h1 h2
  split
  · intro ha1
    rw ha1
    norm_num
  · intro h_eq
    by_cases ha1 : a = 1
    · exact Or.inl ha1
    · use (h_eq.left symm)
      sorry

end a_sufficient_but_not_necessary_l305_305096


namespace sufficient_but_not_necessary_condition_l305_305997

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l305_305997


namespace problem1_problem2_l305_305069

section Problems

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Problem 1: Tangent line problem for a = 1
def tangent_line_eqn (x : ℝ) : Prop :=
  let a := 1
  let f := f a
  (∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b)

-- Problem 2: Minimum value problem
def min_value_condition (a : ℝ) : Prop :=
  f a (1 / 4) = (11 / 12)

theorem problem1 : tangent_line_eqn 0 :=
  sorry

theorem problem2 : min_value_condition (1 / 4) :=
  sorry

end Problems

end problem1_problem2_l305_305069


namespace find_age_l305_305225

variable (a b : ℕ)
variable (h₁ : 0 ≤ a ∧ a ≤ 9)
variable (h₂ : 0 ≤ b ∧ b ≤ 9)
variable (h₃ : 9 * (a - b) = 5 * (moves.age))

theorem find_age : age = 9 :=
by
  sorry

end find_age_l305_305225


namespace smallest_distance_CD_eq_zero_l305_305593

noncomputable def rational_woman_track (t : ℝ) : ℝ × ℝ :=
  (2 * Real.cos t, 2 * Real.sin t)

noncomputable def irrational_woman_track (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * Real.cos (t / 2), 4 + 3 * Real.sin (t / 2))

theorem smallest_distance_CD_eq_zero :
  let C := rational_woman_track
  let D := irrational_woman_track
  ∃ t1 t2 : ℝ, ∀ t, ∀ t', (C t - D t').dist = 0 :=
  sorry

end smallest_distance_CD_eq_zero_l305_305593


namespace tree_distance_l305_305345

theorem tree_distance 
  (num_trees : ℕ) (dist_first_to_fifth : ℕ) (length_of_road : ℤ) 
  (h1 : num_trees = 8) 
  (h2 : dist_first_to_fifth = 100) 
  (h3 : length_of_road = (dist_first_to_fifth * (num_trees - 1)) / 4 + 3 * dist_first_to_fifth) 
  :
  length_of_road = 175 := 
sorry

end tree_distance_l305_305345


namespace find_x_l305_305033

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : nat.factors x = [31, p1, p2]) : x = 32767 :=
by
  sorry

end find_x_l305_305033


namespace smallest_sum_of_squares_l305_305610

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 175) : 
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 + y^2 = 625 :=
sorry

end smallest_sum_of_squares_l305_305610


namespace pow_sub_self_divisible_by_prime_l305_305183

theorem pow_sub_self_divisible_by_prime (a : ℤ) (p : ℕ) [Fact (Nat.Prime p)] : p ∣ (a ^ p - a) :=
sorry

end pow_sub_self_divisible_by_prime_l305_305183


namespace count_powers_of_primes_below_70_l305_305088

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p k : ℕ), is_prime p ∧ k ≥ 1 ∧ n = p^k

def count_powers_of_primes (limit : ℕ) : ℕ :=
  (Finset.range (limit + 1)).filter is_power_of_prime |>.card

theorem count_powers_of_primes_below_70 : count_powers_of_primes 70 = 30 := by
  sorry

end count_powers_of_primes_below_70_l305_305088


namespace john_reads_3_books_in_15_hours_l305_305140

-- Definition of the problem parameters
def john_speed_ratio := 1.6         -- Brother reads books 1.6 times slower than John
def brother_time_per_book := 8      -- Brother takes 8 hours to read a book
def john_time_per_book := brother_time_per_book / john_speed_ratio  -- John's time to read one book
def john_books := 3

-- Problem: Prove that John's total time to read 3 books is 15 hours
theorem john_reads_3_books_in_15_hours :
  john_time_per_book * john_books = 15 :=
begin
  -- Prove that 5 hours per book multiplied by 3 books equals 15 hours
  sorry
end

end john_reads_3_books_in_15_hours_l305_305140


namespace find_y_l305_305442

theorem find_y 
  (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 18) 
  (h2 : x + 2 * y = 10) : 
  y = 1.5 := 
by 
  sorry

end find_y_l305_305442


namespace cos_angle_YXW_l305_305131

-- Define the sides of the triangle
def XY := 5
def XZ := 9
def YZ := 11

-- Define a point W such that XW bisects angle YXZ
def bisects_angle_YXZ := True  -- This is a placeholder for the actual bisect condition

-- The theorem to prove:
theorem cos_angle_YXW :
  ∃ (triangle_XYZ : Type) (XY XZ YZ : triangle_XYZ), XY = 5 ∧ XZ = 9 ∧ YZ = 11 ∧ bisects_angle_YXZ →
  cos (angle_X triangle_XYZ XY XZ YZ) = (sqrt 15) / 6 :=
by sorry

end cos_angle_YXW_l305_305131


namespace min_green_beads_l305_305718

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l305_305718


namespace actual_cost_before_decrease_l305_305733

theorem actual_cost_before_decrease (x : ℝ) (h : 0.76 * x = 1064) : x = 1400 :=
by
  sorry

end actual_cost_before_decrease_l305_305733


namespace find_x_given_conditions_l305_305191

variables {x y z : ℝ}

theorem find_x_given_conditions (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (576 : ℝ)^(1/7) := 
sorry

end find_x_given_conditions_l305_305191


namespace value_at_7_6_l305_305149

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) : f (x + 4) = f x := sorry

lemma f_on_interval (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) : f x = x := sorry

theorem value_at_7_6 : f 7.6 = -0.4 :=
by
  have p := periodic_f 7.6
  have q := periodic_f 3.6
  have r := f_on_interval (-0.4)
  sorry

end value_at_7_6_l305_305149


namespace simple_interest_rate_l305_305680

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (hSI : SI = 250) (hP : P = 1500) (hT : T = 5)
  (hSIFormula : SI = (P * R * T) / 100) :
  R = 3.33 := 
by 
  sorry

end simple_interest_rate_l305_305680


namespace visitors_not_enjoy_not_understood_l305_305873

theorem visitors_not_enjoy_not_understood
  (V : ℕ) (hV : V = 600)
  (hEN : 3 / 4 * V = V / 2 * 3 / 2) :
  ∃ N : ℕ, N = 75 :=
by
  let N := (1 / 8) * V
  have hN : N = 75 := by
    calc
      N = (1 / 8) * 600 : by rw hV
      ... = 75 : by norm_num
  exact ⟨N, hN⟩

end visitors_not_enjoy_not_understood_l305_305873


namespace inequality_must_hold_l305_305373

theorem inequality_must_hold (x y : ℝ) (h : x > y) : x - 3 > y - 3 :=
by
  exact h

end inequality_must_hold

end inequality_must_hold_l305_305373


namespace cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l305_305315

noncomputable def cos_negative_pi_over_3 : Real :=
  Real.cos (-Real.pi / 3)

theorem cos_neg_pi_over_3_eq_one_half :
  cos_negative_pi_over_3 = 1 / 2 :=
  by
    sorry

noncomputable def solutions_sin_eq_sqrt3_over_2 (x : Real) : Prop :=
  Real.sin x = Real.sqrt 3 / 2 ∧ 0 ≤ x ∧ x < 2 * Real.pi

theorem sin_eq_sqrt3_over_2_solutions :
  {x : Real | solutions_sin_eq_sqrt3_over_2 x} = {Real.pi / 3, 2 * Real.pi / 3} :=
  by
    sorry

end cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l305_305315


namespace integer_triples_condition_l305_305002

theorem integer_triples_condition (p q r : ℤ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) 
  (h4 : ((p - 1) * (q - 1) * (r - 1)) ∣ (p * q * r - 1)) : (p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15) :=
sorry

end integer_triples_condition_l305_305002


namespace convex_quadrilateral_distance_l305_305168

theorem convex_quadrilateral_distance
    {A B C D : Type} [ConvexQuadrilateral A B C D]
    (hBD : length (B, D) = BD)
    (hAC_le_BD : length (A, C) ≤ length (B, D)) :
    (∃ V, V ∈ {A, C} ∧ exists_perpendicular_distance V (B, D) (λ d, d ≤ BD / 2)) := 
sorry

end convex_quadrilateral_distance_l305_305168


namespace seating_arrangement_count_l305_305961

-- Define the cousins
inductive Cousin
| a1 | a2 | b1 | b2 | c1 | c2 | d

-- We use sets and lists to represent seating arrangements
open Cousin

def van_seating : Type := list (list Cousin)

-- Define conditions
def valid_arrangement (seating : list HD) : Prop :=
  -- cousins should not sit next to each other or directly in front of each other
  (∀ l ∈ seating, -- go through each row
    ∀ i j, i ≠ j → -- in each row, for each pair
      ... -- insert logic to ensure no cousins a1, a2 or b1, b2 sit next to each other
  ) ∧
  (∀ i, -- go through each cousin index
      ... -- insert logic to ensure no cousin sits directly in front of each other in different rows
  )

-- Total number of seating arrangements
noncomputable def num_seating_arrangements (seating : list HD) : nat :=
  ... -- insert combinatorial logic from the proof to calculate arrangements

-- The theorem statement
theorem seating_arrangement_count :
  ∃ seating, valid_arrangement seating ∧ (num_seating_arrangements seating = 240) :=
sorry

end seating_arrangement_count_l305_305961


namespace comparison_among_three_numbers_l305_305217

theorem comparison_among_three_numbers (a b c : ℝ) (h1 : a = 7 ^ 0.3) (h2 : b = 0.3 ^ 7) (h3 : c = Real.log 0.3) 
  (h4 : a > 1) (h5 : 0 < b ∧ b < 1) (h6 : c < 0) : a > b ∧ b > c :=
by
  sorry

end comparison_among_three_numbers_l305_305217


namespace length_of_notebook_is_24_l305_305251

-- Definitions
def span_of_hand : ℕ := 12
def length_of_notebook (span : ℕ) : ℕ := 2 * span

-- Theorem statement that proves the question == answer given conditions
theorem length_of_notebook_is_24 :
  length_of_notebook span_of_hand = 24 :=
sorry

end length_of_notebook_is_24_l305_305251


namespace distribution_ways_l305_305841

theorem distribution_ways :
  ∃ (n : ℕ), n = 56 ∧
  ∀ (balls boxes : ℕ), balls = 5 → boxes = 4 →
  (nat.choose (balls + boxes - 1) (boxes - 1)) = 56 :=
by
  sorry

end distribution_ways_l305_305841


namespace candidate_lost_by_1800_votes_l305_305692

noncomputable def total_votes : ℕ := 6000
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def candidate_votes : ℕ := (candidate_percentage * total_votes).to_nat
noncomputable def rival_votes : ℕ := total_votes - candidate_votes
noncomputable def vote_difference : ℕ := rival_votes - candidate_votes

theorem candidate_lost_by_1800_votes :
  vote_difference = 1800 := 
sorry

end candidate_lost_by_1800_votes_l305_305692


namespace original_saved_amount_l305_305722

theorem original_saved_amount (x : ℤ) (h : (3 * x - 42)^2 = 2241) : x = 30 := 
sorry

end original_saved_amount_l305_305722


namespace perpendicular_lines_l305_305544

variables (A B C O D E : Type) [euclidean_geometry A B C] 

-- Definitions based on given conditions
def circumcenter (A B C O : Type) : Prop := 
  center of the circumcircle of triangle ABC is O

def bisector_intersection (B A C D : Type) : Prop := 
  D is the intersection of the bisector of ∠B and the side AC

def point_on_side (B C E : Type) : Prop := 
  E is a point on BC such that AB = BE

-- Theorem to prove lines (BO) and (DE) are perpendicular
theorem perpendicular_lines 
  (triangle_ABC : Prop)
  (O_circumcenter : circumcenter A B C O)
  (D_bisector_intersection : bisector_intersection B A C D)
  (E_on_side : point_on_side B C E) :
  perpendicular (BO) (DE) :=
sorry

end perpendicular_lines_l305_305544


namespace sum_squares_reciprocal_l305_305222

variable (x y : ℝ)

theorem sum_squares_reciprocal (h₁ : x + y = 12) (h₂ : x * y = 32) :
  (1/x)^2 + (1/y)^2 = 5/64 := by
  sorry

end sum_squares_reciprocal_l305_305222


namespace cos_theta_minimum_value_l305_305662

def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sin x

theorem cos_theta_minimum_value (θ : ℝ) (h : ∀ x, f x ≥ f θ) : 
  Real.cos θ = -3 / Real.sqrt 10 :=
by 
  sorry

end cos_theta_minimum_value_l305_305662


namespace common_volume_of_tetrahedra_gt_half_l305_305123

/-- In space, there are two identical regular tetrahedra with side length √6. 
It is known that their centers coincide. Prove that the volume of the common 
part is greater than 1/2. -/
theorem common_volume_of_tetrahedra_gt_half (a : ℝ) (h1 : a = Real.sqrt 6) (h2 : ∀ (T1 T2 : EuclideanGeometry.Tetrahedron ℝ), 
  T1.side_length = a ∧ T2.side_length = a ∧ T1.center = T2.center) : 
  ∃ (V : ℝ), V > 1/2 := 
sorry

end common_volume_of_tetrahedra_gt_half_l305_305123


namespace at_most_three_spanning_trees_odd_number_of_spanning_trees_l305_305539

-- Definitions related to knots and their diagrams
def Knot : Type := sorry -- Knot in ℝ³
def Diagram : Type := sorry -- Diagram of the knot
def blackGraph (D : Diagram) : sorry := sorry -- Black graph of a diagram

-- The specific problem statements to be proved

-- Part (a): Determine all knots with a diagram such that the black graph has at most 3 spanning trees
theorem at_most_three_spanning_trees (K : Knot) (D : Diagram) (h : blackGraph(D).spanning_trees ≤ 3) : (K = trefoil ∨ K = unknot) := 
sorry

-- Part (b): Prove that for any knot and diagram, the black graph has an odd number of spanning trees
theorem odd_number_of_spanning_trees (K : Knot) (D : Diagram) : (blackGraph(D).spanning_trees % 2 = 1) :=
sorry

end at_most_three_spanning_trees_odd_number_of_spanning_trees_l305_305539


namespace width_of_smaller_cuboid_l305_305696

-- Conditions as definitions in Lean 4
def length_large : ℝ := 18
def width_large : ℝ := 15
def height_large : ℝ := 2

def length_small : ℝ := 6
def height_small : ℝ := 3
def num_smaller_cuboids : ℝ := 7.5

-- Problem statement and the proof goal
theorem width_of_smaller_cuboid : 
  (length_large * width_large * height_large = num_smaller_cuboids * (length_small * width_small * height_small)) →
  width_small = 4 :=
begin
  sorry
end

end width_of_smaller_cuboid_l305_305696


namespace distance_between_city_centers_l305_305199

def distance_on_map : ℝ := 45  -- Distance on the map in cm
def scale_factor : ℝ := 20     -- Scale factor (1 cm : 20 km)

theorem distance_between_city_centers : distance_on_map * scale_factor = 900 := by
  sorry

end distance_between_city_centers_l305_305199


namespace infinite_primes_solution_l305_305172

open Mathlib

theorem infinite_primes_solution (py : ℕ) : 
  ∀ p : ℕ, Prime p → ∃ x y : ℤ, x^2 + x + 1 = p * (y : ℤ) → (∃ count : ℕ, count > 0 ∧ ∀ n : ℕ, n < count → Prime (nth_prime n)) :=
by
  sorry

end infinite_primes_solution_l305_305172


namespace find_c_value_l305_305108

theorem find_c_value 
  (a b c : ℝ)
  (h_a : a = 5 / 2)
  (h_b : b = 17)
  (roots : ∀ x : ℝ, x = (-b + Real.sqrt 23) / 5 ∨ x = (-b - Real.sqrt 23) / 5)
  (discrim_eq : ∀ c : ℝ, b ^ 2 - 4 * a * c = 23) :
  c = 26.6 := by
  sorry

end find_c_value_l305_305108


namespace find_a_l305_305826

def f (x : ℝ) (a : ℝ) : ℝ := 
  if x < 1 then 3^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) : f (f 0 a) a = 6 → a = 1 := 
  by
  sorry

end find_a_l305_305826


namespace interest_rate_correct_l305_305739

-- Definitions based on the problem conditions
def P : ℝ := 7000 -- Principal investment amount
def A : ℝ := 8470 -- Future value of the investment
def n : ℕ := 1 -- Number of times interest is compounded per year
def t : ℕ := 2 -- Number of years

-- The interest rate r to be proven
def r : ℝ := 0.1 -- Annual interest rate

-- Statement of the problem that needs to be proven in Lean
theorem interest_rate_correct :
  A = P * (1 + r / n)^(n * t) :=
by
  sorry

end interest_rate_correct_l305_305739


namespace inscribed_square_area_percentage_l305_305249

theorem inscribed_square_area_percentage {r : ℝ} (h1 : r > 0) :
  let s := r * Real.sqrt 2
  let A_s := s^2
  let A_c := Real.pi * r^2
  64 = Int.round ((A_s / A_c) * 100) :=
sorry

end inscribed_square_area_percentage_l305_305249


namespace pirate_treasure_l305_305584

theorem pirate_treasure (x : ℕ) (h : 1 + 2 + ... + x = 3 * x) : 4 * x = 20 :=
by
  sorry

end pirate_treasure_l305_305584


namespace smaller_cubes_total_l305_305706

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l305_305706


namespace fixed_points_l305_305367

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end fixed_points_l305_305367


namespace sin_transform_l305_305402

def is_transformed (y : ℝ → ℝ) (T1 T2 : (ℝ → ℝ) → (ℝ → ℝ)) : Prop :=
  T1 (T2 y) = (fun x => real.sin (x / 2 + real.pi / 3))

def transformation1 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x / 2)

def transformation2 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (2 * x)

def transformation3 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x + real.pi / 3)

def transformation4 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x - real.pi / 3)

def transformation5 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x + 2 * real.pi / 3)

def transformation6 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x - 2 * real.pi / 3)

theorem sin_transform :
  ∃ T1 T2, (T1 = transformation3 ∧ T2 = transformation1 ∨ T1 = transformation2 ∧ T2 = transformation5) ∧
        is_transformed real.sin T1 T2 :=
by
  sorry

end sin_transform_l305_305402


namespace tan_theta_one_l305_305019

theorem tan_theta_one (θ : ℝ) (h₁ : θ ∈ Ioo (π/6) (π/3)) (h₂ : ∃ k : ℤ, 17 * θ = θ + 2 * k * π) :
  Real.tan θ = 1 :=
sorry

end tan_theta_one_l305_305019


namespace second_assistant_smoked_pipes_l305_305970

theorem second_assistant_smoked_pipes
    (x y z : ℚ)
    (H1 : (2 / 3) * x = (4 / 9) * y)
    (H2 : x + y = 1)
    (H3 : (x + z) / (y - z) = y / x) :
    z = 1 / 5 → x = 2 / 5 ∧ y = 3 / 5 →
    ∀ n : ℕ, n = 5 :=
by
  sorry

end second_assistant_smoked_pipes_l305_305970


namespace complex_expression_l305_305214

theorem complex_expression (i : ℂ) (h : i^2 = -1) : ( (1 + i) / (1 - i) )^2006 = -1 :=
by {
  sorry
}

end complex_expression_l305_305214


namespace exists_small_area_triangle_l305_305910

-- Define the square and its side length
def side_length := 20
def total_area := side_length * side_length

-- Define the set M: vertices of the square plus 1999 points inside the square
def num_vertices := 4
def extra_points := 1999
def num_points := num_vertices + extra_points

-- Define the number of triangles that can be formed by these points (using convex polygon triangulation)
def num_triangles := num_points - 2

-- Define the area threshold
def area_threshold := 1 / 10

theorem exists_small_area_triangle 
  (S : set (ℝ × ℝ)) -- the square
  (M : set (ℝ × ℝ)) -- the set of points
  (hS : S = { (x, y) | 0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length }) 
  (hM : finite M ∧ M.card = num_points ∧ (∀ x ∈ M, x ∈ S)) :
  ∃ (Δ : set (ℝ × ℝ)), is_triangle Δ ∧ (∀ (v ∈ Δ), v ∈ M) ∧ triangle_area Δ ≤ area_threshold :=
sorry

end exists_small_area_triangle_l305_305910


namespace positive_four_digit_integers_l305_305840

theorem positive_four_digit_integers : 
  let digits := [0, 0, 3, 9] in
  let permutations := Nat.factorial 4 / Nat.factorial 2 in
  let invalid := Nat.factorial 3 / Nat.factorial 1 in
  permutations - invalid = 6 := by
    sorry

end positive_four_digit_integers_l305_305840


namespace equilateral_triangle_functional_l305_305302

-- Defining the functional relationship problem

def is_functional_relationship (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f x = f y → x = y

def equilateral_triangle_area (a : ℝ) : ℝ :=
(√3 / 4) * a^2

-- Theorem statement
theorem equilateral_triangle_functional :
  is_functional_relationship equilateral_triangle_area :=
by
  sorry

end equilateral_triangle_functional_l305_305302


namespace side_length_of_square_base_l305_305967

theorem side_length_of_square_base (area : ℝ) (slant_height : ℝ) (s : ℝ) (h : slant_height = 40) (a : area = 160) : s = 8 :=
by sorry

end side_length_of_square_base_l305_305967


namespace binomial_expansion_coefficient_and_sum_l305_305127

theorem binomial_expansion_coefficient_and_sum :
  (fincoeff ((x : ℚ) + (1/x))^5 1 = fincoeff (x + 1/x)^5 1 = 10) ∧
  (sum_all_coeff ((x : ℚ) + (1/x))^5 = (2 : ℚ)^5 = 32) :=
by
  sorry

end binomial_expansion_coefficient_and_sum_l305_305127


namespace inradius_of_triangle_l305_305509

variable (A : ℝ) (p : ℝ) (r : ℝ) (s : ℝ)

theorem inradius_of_triangle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l305_305509


namespace Alice_min_speed_l305_305611

theorem Alice_min_speed (d : ℝ) (v_bob : ℝ) (delta_t : ℝ) (v_alice : ℝ) :
  d = 180 ∧ v_bob = 40 ∧ delta_t = 0.5 ∧ 0 < v_alice ∧ v_alice * (d / v_bob - delta_t) ≥ d →
  v_alice > 45 :=
by
  sorry

end Alice_min_speed_l305_305611


namespace collinear_X_K_O_l305_305685

/-- Points A, B, and C lie on the circumference of a circle with center O.
    Tangents are drawn to the circumcircles of triangles OAB and OAC at points P and Q 
    respectively, where P and Q are diametrically opposite O. The tangents intersect at K. 
    The line CA meets the circumcircle of triangle OAB at A and X. 
    Prove that X lies on the line KO. -/
theorem collinear_X_K_O 
    {A B C O P Q K X : Type}
    [inhabited A] [inhabited B] [inhabited C] [inhabited O] [inhabited P] [inhabited Q] 
    [inhabited K] [inhabited X]
    (h1 : is_on_circle A B C O)
    (h2 : diametrically_opposite P Q O)
    (h3 : tangent_at P (circumcircle_tri O A B))
    (h4 : tangent_at Q (circumcircle_tri O A C))
    (h5 : intersect K (tangents P Q))
    (h6 : meet_at X A (line_through C A) (circumcircle_tri O A B)) :
    is_on_line X (line_through K O) := 
sorry

end collinear_X_K_O_l305_305685


namespace correct_conclusions_l305_305801

noncomputable def f : ℝ → ℝ := sorry -- Assume f has already been defined elsewhere

-- The given conditions
variable (f : ℝ → ℝ)
variable (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
variable (H2 : f (1 / 2) = 0)
variable (H3 : f 0 ≠ 0)

-- Proof problem which includes the correct conclusions
theorem correct_conclusions (f : ℝ → ℝ)
  (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
  (H2 : f (1 / 2) = 0)
  (H3 : f 0 ≠ 0) :
  f 0 = 1 ∧ ∀ y : ℝ, f (1 / 2 + y) + f (1 / 2 - y) = 0 :=
by
  sorry

end correct_conclusions_l305_305801


namespace solve_congruences_l305_305637

theorem solve_congruences :
  ∃ x : ℤ, x ≡ 0 [MOD 5] ∧
           x ≡ 10 [MOD 715] ∧
           x ≡ 140 [MOD 247] ∧
           x ≡ 245 [MOD 391] ∧
           x ≡ 109 [MOD 187] ∧
           x % 5311735 = 10020 :=
by
  sorry

end solve_congruences_l305_305637


namespace problem_part1_problem_part2_l305_305050

noncomputable def log2 (x : ℝ) : ℝ := log x / log 2

theorem problem_part1 (a b : ℝ) (h : log2 a + log2 b = 1) : a * b = 2 := 
sorry

theorem problem_part2 (a b : ℝ) (h : a * b = 2) : (a + 1 / a) * (b + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end problem_part1_problem_part2_l305_305050


namespace sine_is_odd_minor_premise_incorrect_l305_305994

-- Define the sine function and a general function f.
def sine (x : ℝ) : ℝ := Real.sin x
def f (x : ℝ) : ℝ := Real.sin (x^2 + 1)

-- State that the sine function is odd.
theorem sine_is_odd : ∀ x : ℝ, sine (-x) = -sine (x) :=
begin
  intro x,
  sorry  -- The proof that sine is odd would go here
end

-- The proposition to prove: f(x) is not a sine function.
theorem minor_premise_incorrect : ¬(∃ g : ℝ → ℝ, ∀ x : ℝ, f x = sine (g x)) :=
begin
  sorry  -- The proof that f(x) is not a sine function would go here
end

end sine_is_odd_minor_premise_incorrect_l305_305994


namespace cube_removal_volume_l305_305323

theorem cube_removal_volume (s : ℝ) (r : ℝ) (h : ℝ) 
  (hs : s = 6) (hr : r = 3) (hh : h = 6) : 
  (s^3 - real.pi * r^2 * h = 216 - 54 * real.pi) :=
by
  have V_cube := s^3,
  have V_cylinder := real.pi * r^2 * h,
  sorry

end cube_removal_volume_l305_305323


namespace isosceles_triangle_perimeter_l305_305209

-- Define the conditions
def equilateral_triangle_side : ℕ := 15
def isosceles_triangle_side : ℕ := 15
def isosceles_triangle_base : ℕ := 10

-- Define the theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter : 
  (2 * isosceles_triangle_side + isosceles_triangle_base = 40) :=
by
  -- Placeholder for the actual proof
  sorry

end isosceles_triangle_perimeter_l305_305209


namespace derivative_value_at_one_l305_305065

-- Define the function f and its derivative f'
def f (x : ℝ) := 2 * x * (deriv f 1) + Real.log x

-- Define the proof problem
theorem derivative_value_at_one : 
  (deriv f 1 = -1) :=
by
  -- Placeholder for the proof steps
  sorry

end derivative_value_at_one_l305_305065


namespace sales_tax_difference_l305_305973

-- Definitions for the price and tax rates
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.065
def tax_rate2 : ℝ := 0.06
def tax_rate3 : ℝ := 0.07

-- Sales tax amounts derived from the given rates and item price
def tax_amount (rate : ℝ) (price : ℝ) : ℝ := rate * price

-- Calculate the individual tax amounts
def tax_amount1 : ℝ := tax_amount tax_rate1 item_price
def tax_amount2 : ℝ := tax_amount tax_rate2 item_price
def tax_amount3 : ℝ := tax_amount tax_rate3 item_price

-- Proposition stating the proof problem
theorem sales_tax_difference :
  max tax_amount1 (max tax_amount2 tax_amount3) - min tax_amount1 (min tax_amount2 tax_amount3) = 0.50 :=
by 
  sorry

end sales_tax_difference_l305_305973


namespace proposition_5_l305_305272

/-! 
  Proposition 5: If there are four points A, B, C, D in a plane, 
  then the vector addition relation: \overrightarrow{AC} + \overrightarrow{BD} = \overrightarrow{BC} + \overrightarrow{AD} must hold.
--/

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (AC BD BC AD : A)

-- Theorem Statement in Lean 4
theorem proposition_5 (AC BD BC AD : A)
  : AC + BD = BC + AD := by
  -- Proof by congruence and equality, will add actual steps here
  sorry

end proposition_5_l305_305272


namespace max_distance_point_l305_305697

-- Define the given circles
def circle1 (p : ℝ × ℝ) : Prop :=
  (p.1 + 1)^2 + p.2^2 = 1

def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 - 2 * p.1 - 8 = 0

-- Define the tangency conditions for the circles
def tangent_to_circle1 (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (center.1 + 1)^2 + center.2^2 = (1 + r)^2

def tangent_to_circle2 (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (center.1 - 1)^2 + center.2^2 = (3 - r)^2

-- Define the point p and its distance from the origin
noncomputable def max_distance_from_origin (p : ℝ × ℝ) : ℝ :=
  real.sqrt (p.1^2 + p.2^2)

-- Define the main theorem to prove
theorem max_distance_point :
  ∃ (p : ℝ × ℝ), (tangent_to_circle1 p 1) ∧ (tangent_to_circle2 p 3) ∧
  (max_distance_from_origin p = max_distance_from_origin (2, 0)) := 
sorry

end max_distance_point_l305_305697


namespace domain_of_h_h_is_odd_h_lt_zero_iff_range_of_a_l305_305807

-- Given definitions
def f (a x : ℝ) : ℝ := log a (1 + x)
def g (a x : ℝ) : ℝ := log a (1 - x)
def h (a x : ℝ) : ℝ := f a x - g a x

-- Proof problem statement
theorem domain_of_h (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  Set.Ioo (-1 : ℝ) 1 = { x : ℝ | h a x ≠ 0 } :=
sorry

theorem h_is_odd (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∀ x : ℝ, h a (-x) = -h a x :=
sorry

theorem h_lt_zero_iff (a : ℝ) (ha : 0 < a ∧ a ≠ 1) (h_f3 : f a 3 = 2) :
  ∀ x : ℝ, h a x < 0 ↔ x ∈ Ioo (-1 : ℝ) 0 :=
sorry

theorem range_of_a (a : ℝ) (ha : 0 < a ∧ a ≠ 1) (hrange : ∀ x ∈ Icc 0 (1/2 : ℝ), h a x ∈ Icc 0 1) :
  a = 3 :=
sorry

end domain_of_h_h_is_odd_h_lt_zero_iff_range_of_a_l305_305807


namespace sum_of_first_49_primes_l305_305358

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                                   61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 
                                   137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 
                                   199, 211, 223, 227]

theorem sum_of_first_49_primes : first_49_primes.sum = 10787 :=
by
  -- Proof to be filled in
  sorry

end sum_of_first_49_primes_l305_305358


namespace common_ratio_of_gp_l305_305007

variable (r : ℝ)(n : ℕ)

theorem common_ratio_of_gp (h1 : 9 * r ^ (n - 1) = 1/3) 
                           (h2 : 9 * (1 - r ^ n) / (1 - r) = 40 / 3) : 
                           r = 1/3 := 
sorry

end common_ratio_of_gp_l305_305007


namespace find_x_l305_305040

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l305_305040


namespace general_formula_a_sum_of_sequences_l305_305146

open_locale big_operators

-- Definition of the sequences
def a (n : ℕ) : ℕ := 2^n
def b (n : ℕ) : ℕ := 2*n - 1

-- Hypothesis
axiom H1 : a 1 = 2
axiom H2 : a 3 = a 2 + 4

-- Theorem statements
theorem general_formula_a : ∀ n : ℕ, a n = 2^n := 
by
  -- Skipping the actual proof
  sorry

theorem sum_of_sequences (n : ℕ) : ∑ i in finset.range n, (a i + b i) = 2^(n+1) + n^2 - 2 :=
by
  -- Skipping the actual proof
  sorry

end general_formula_a_sum_of_sequences_l305_305146


namespace inequality_exp_l305_305072

-- Definitions representing the conditions of the problem
def f (a x : ℝ) := (a * x - 1) * Real.exp x

-- Lean 4 statement for the second part of the problem:
theorem inequality_exp (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m > n) : m * Real.exp n + n < n * Real.exp m + m :=
sorry

end inequality_exp_l305_305072


namespace triangle_ABC_a_lambda_l305_305528

-- Definitions based on conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosA : ℝ)
variable (λ : ℝ)

axiom h1 : c = 5 / 2
axiom h2 : b = Real.sqrt 6
axiom h3 : 4 * a - 3 * Real.sqrt 6 * cosA = 0
axiom h4 : B = λ * A

-- Theorem statement
theorem triangle_ABC_a_lambda (A B C a b c cosA λ : ℝ) 
  (h1 : c = 5 / 2) 
  (h2 : b = Real.sqrt 6) 
  (h3 : 4 * a - 3 * Real.sqrt 6 * cosA = 0)
  (h4 : B = λ * A) :
  a = 3 / 2 ∧ λ = 2 :=
by
  sorry

end triangle_ABC_a_lambda_l305_305528


namespace f_constant_l305_305886

-- Define the regular tetrahedron and points E and F
open EuclideanGeometry

variable (λ : ℝ) (hλ : 0 < λ)

-- Define the regular tetrahedron and the points E and F on edges AB and CD
variables {A B C D E F : Point}

-- Condition stating ABCD is a regular tetrahedron.
axiom regular_tetrahedron (A B C D : Point) : is_regular_tetrahedron A B C D

-- Points E and F satisfy the given ratio conditions
axiom point_E (A B E : Point) : ∃ λ > 0, div_eq_ratio (segment A E) (segment E B) λ
axiom point_F (C D F : Point) : ∃ λ > 0, div_eq_ratio (segment C F) (segment F D) λ

-- Define the function f(λ)
noncomputable def f (λ : ℝ) : ℝ :=
  let α := angle (segment E F) (segment A C) in
  let β := angle (segment E F) (segment B D) in
  α + β

-- The main theorem to prove that f(λ) is constant and equals 90 degrees
theorem f_constant (λ : ℝ) (hλ : 0 < λ) : f λ = 90 :=
by
  sorry

end f_constant_l305_305886


namespace odd_function_f_1_l305_305097

theorem odd_function_f_1 (a : ℝ) (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_def : ∀ x, f x = a - 2 / (2^x + 1)) : f 1 = 1/3 :=
by
  let f0 := (a - 1)
  have ha : f0 = 0,
  sorry
  have hap : a = 1,
  sorry
  have f1 := (1 - 2 / (2 + 1)),
  have : f1 = 1/3,
  exact this
  sorry

end odd_function_f_1_l305_305097


namespace probability_zero_l305_305932

-- Define the conditions
def lydia_lap_time : ℝ := 120 -- seconds
def lucas_lap_time : ℝ := 100 -- seconds
def time_span : ℝ := 15 * 60 -- 15 minutes in seconds

def fraction_in_picture : ℝ := 1 / 3

-- Prove the probability
theorem probability_zero
  (lydia_laps : ℝ) (lucas_laps : ℝ) (picture_fraction : ℝ) (time_interval : ℝ) :
  lydia_laps = lydia_lap_time ∧ lucas_laps = lucas_lap_time ∧
  picture_fraction = fraction_in_picture ∧ time_interval = time_span →
  (lydia_laps > 0 ∧ lucas_laps > 0 ∧ picture_fraction > 0 ∧ time_interval > 0) →
  (15 * 60 <= time_interval ∧ time_interval <= 16 * 60) →
  0 = 0 :=
by
  intros conditions nonzero proper_span
  sorry

end probability_zero_l305_305932


namespace degree_and_number_of_terms_l305_305198

-- Define the given polynomial
def polynomial : Polynomial ℝ := -X^3 + 3

-- Define a theorem stating the degree and number of terms of the polynomial
theorem degree_and_number_of_terms : polynomial.degree = 3 ∧ polynomialSupport.card = 2 := by
  sorry

end degree_and_number_of_terms_l305_305198


namespace simplify_expression_l305_305960

def is_real (x : ℂ) : Prop := ∃ (y : ℝ), x = y

theorem simplify_expression 
  (x y c : ℝ) 
  (i : ℂ) 
  (hi : i^2 = -1) :
  (x + i*y + c)^2 = (x^2 + c^2 - y^2 + 2 * c * x + (2 * x * y + 2 * c * y) * i) :=
by
  sorry

end simplify_expression_l305_305960


namespace correct_proposition_l305_305026

-- Definitions according to conditions
variables (m n : Line) (α β : Plane)

-- Proposition C (Correct Answer)
theorem correct_proposition:
  (m ⟂ α) ∧ (n ∥ α) -> (m ⟂ n) :=
sorry

end correct_proposition_l305_305026


namespace triangle_is_isosceles_l305_305990

theorem triangle_is_isosceles
  (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (H : ∀ n : ℕ, n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n)) :
  p = q ∨ q = r ∨ r = p :=
by
  sorry

end triangle_is_isosceles_l305_305990


namespace coefficients_geq_4_over_9_l305_305374

theorem coefficients_geq_4_over_9 
  (A B : ℝ) (a b c d e f : ℝ)
  (hA : 0 ≤ A ∧ A ≤ 1)
  (hB : 0 ≤ B ∧ B ≤ 1)
  (h1 : ∀ x y : ℝ, ax^2 + bxy + cy^2 = (A*x + (1 - A)*y)^2)
  (h2 : ∀ x y : ℝ, (A*x + (1 - A)*y)*(B*x + (1 - B)*y) = dx^2 + exy + fy^2) :
  ∃ i ∈ {a, b, c}, i ≥ 4/9 ∧ ∃ j ∈ {d, e, f}, j ≥ 4/9 :=
by
  sorry

end coefficients_geq_4_over_9_l305_305374


namespace inequality_solution_1_inequality_solution_2_l305_305071

def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

theorem inequality_solution_1 :
  ∀ x : ℝ, f(x) ≥ 2 ↔ (-4 ≤ x ∧ x ≤ -8/3) :=
by
  sorry

theorem inequality_solution_2 (t : ℝ) :
  (∃ x : ℝ, f(x) - |3*t - 2| ≥ 0) ↔ (-1/3 ≤ t ∧ t ≤ 5/3) :=
by
  sorry

end inequality_solution_1_inequality_solution_2_l305_305071


namespace train_length_l305_305295

theorem train_length 
  (t_pole : ℝ) (t_tunnel : ℝ) (l_tunnel : ℝ) (V : ℝ) (L : ℝ)
  (h1 : L = V * t_pole)
  (h2 : L + l_tunnel = V * t_tunnel) : L = 500 :=
by
  have hV : V = L / t_pole := by { rw [← h1], ring }
  have heq : L + l_tunnel = (L / t_pole) * t_tunnel := by { rw [h2, hV], ring }
  have hsimp : L + l_tunnel = 2 * L := by { simp at heq, exact heq }
  have hL : 500 = L := by { linarith }
  exact hL

end train_length_l305_305295


namespace exists_sequence_continuous_pointwise_converge_to_f_l305_305914

noncomputable def f : ℝ × ℝ → ℝ := sorry  -- Assume the function f is provided.

-- The conditions required by the problem.
def is_continuous_fixed_x (x₀ : ℝ) : Continuous (λ x, f (x₀, x)) := sorry
def is_continuous_fixed_y (y₀ : ℝ) : Continuous (λ x, f (x, y₀)) := sorry

-- The goal statement in Lean 4 format
theorem exists_sequence_continuous_pointwise_converge_to_f :
  ∃ (h_n : ℕ → (ℝ × ℝ) → ℝ), (∀ n : ℕ, Continuous (h_n n)) ∧
  (∀ (x y : ℝ), tendsto (λ n, h_n n (x, y)) at_top (nhds (f (x, y)))) :=
sorry

end exists_sequence_continuous_pointwise_converge_to_f_l305_305914


namespace odd_function_f1_l305_305099

theorem odd_function_f1 (a : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = a - 2 / (2^x + 1))
  (h2 : ∀ x, f(-x) = -f(x)) :
  f(1) = 1 / 3 :=
by
  sorry

end odd_function_f1_l305_305099


namespace find_angle_C_l305_305629

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Type)

-- Given conditions
axiom ten_a_cos_B_eq_three_b_cos_A : 10 * a * Real.cos B = 3 * b * Real.cos A
axiom cos_A_value : Real.cos A = 5 * Real.sqrt 26 / 26

-- Required to prove
theorem find_angle_C : C = 3 * Real.pi / 4 := by
  sorry

end find_angle_C_l305_305629


namespace buy_to_get_ratio_l305_305899

theorem buy_to_get_ratio (normal_price : ℝ) (total_cans : ℤ) (amount_paid : ℝ)
  (h_price : normal_price = 0.60)
  (h_total_cans : total_cans = 30)
  (h_amount_paid : amount_paid = 9) : 
  ((amount_paid / normal_price).toInt = total_cans / 2) :=
by
  -- Proof omitted
  sorry

end buy_to_get_ratio_l305_305899


namespace cartesian_eq_of_curve_C2_min_distance_curve_C2_to_line_C1_l305_305124

-- Define the parametric equations of curve C1
def curve_C1_parametric (t : ℝ) : ℝ × ℝ := (2 * t - 1, -4 * t - 2)

-- Define the polar equation of curve C2
def curve_C2_polar (θ : ℝ) : ℝ := 2 / (1 - Real.cos θ)

-- Translate polar to Cartesian equation for curve C2
theorem cartesian_eq_of_curve_C2 :
  ∀ x y : ℝ, y^2 = 4 * (x - 1) ↔ ∃ θ : ℝ, x = 2 / (1 - Real.cos θ) ∧ y = 2 * Real.sin θ * (1 - Real.cos θ) := 
sorry

-- Define the line equation for curve C1
def line_C1 (x y : ℝ) : Prop := 2 * x + y + 4 = 0

-- Calculate the minimum distance from a point on curve C2 to the line C1
theorem min_distance_curve_C2_to_line_C1 :
  ∀ (M2x M2y : ℝ), (M2y^2 = 4 * (M2x - 1)) →
  ∃ d : ℝ, d = 2 * |(M2x + M2y + 1) / √5| ∧ d ≥ (3 * √5 / 10) :=
sorry

end cartesian_eq_of_curve_C2_min_distance_curve_C2_to_line_C1_l305_305124


namespace range_of_b_in_acute_triangle_l305_305511

theorem range_of_b_in_acute_triangle 
  (A B C : ℝ) (a b c : ℝ)
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_angles : A + B + C = π)
  (h_sides : a = 1)
  (h_B_eq_2A : B = 2 * A) :
  sqrt 2 < b ∧ b < sqrt 3 :=
by
  sorry

end range_of_b_in_acute_triangle_l305_305511


namespace avg_english_score_is_correct_l305_305604

noncomputable def avg_score_class (score_females : ℝ) (num_females : ℕ) (score_males : ℝ) (num_males : ℕ) : ℝ :=
  let total_students := num_females + num_males
  let total_score_females := score_females * (num_females : ℝ)
  let total_score_males := score_males * (num_males : ℝ)
  let total_score_all := total_score_females + total_score_males
  total_score_all / (total_students : ℝ)

theorem avg_english_score_is_correct :
  avg_score_class 83.1 10 84 8 = 83.5 :=
begin
  sorry
end

end avg_english_score_is_correct_l305_305604


namespace root_ratio_l305_305547

noncomputable theory

def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4
def largest_root (p : ℝ → ℝ) : ℝ := sorry -- Implementing the largest root functionality is non-trivial and outside the scope.

theorem root_ratio :
  let x1 := largest_root f;
      x2 := largest_root g in
  x2 = 2 * x1 → (x1 / x2 = 1 / 2) :=
by
  intros x1 x2 hx2
  sorry

end root_ratio_l305_305547


namespace solve_logarithmic_equation_l305_305187

theorem solve_logarithmic_equation (x : ℝ) (h : x > 0 ∧ x ≠ 1 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → log (k/(k+1) : ℝ) ≠ 0)) :
  (1 / (log x / log (1/2 : ℝ)) + 
   1 / (log x / log (2/3 : ℝ)) + 
   1 / (log x / log (3/4 : ℝ)) + 
   1 / (log x / log (4/5 : ℝ)) + 
   1 / (log x / log (5/6 : ℝ)) + 
   1 / (log x / log (6/7 : ℝ)) + 
   1 / (log x / log (7/8 : ℝ)) + 
   1 / (log x / log (8/9 : ℝ)) + 
   1 / (log x / log (9/10 : ℝ)) = 1) → 
  x = 1/10 := sorry

end solve_logarithmic_equation_l305_305187


namespace pentagon_rectangle_ratio_l305_305291

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end pentagon_rectangle_ratio_l305_305291


namespace complex_number_calculation_l305_305318

theorem complex_number_calculation :
  let z := (5 - 6 * complex.I) + (-2 - complex.I) - (3 + 4 * complex.I)
  in z = -11 * complex.I ∧ z.im < 0 ∧ z.re = 0 :=
sorry

end complex_number_calculation_l305_305318


namespace correct_conclusions_l305_305803

-- Given function f with the specified domain and properties
variable {f : ℝ → ℝ}

-- Given conditions
axiom functional_eq (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y
axiom f_one_half : f (1/2) = 0
axiom f_zero_not_zero : f 0 ≠ 0

-- Proving our conclusions
theorem correct_conclusions :
  f 0 = 1 ∧ (∀ y : ℝ, f (1/2 + y) = -f (1/2 - y))
:=
by
  sorry

end correct_conclusions_l305_305803


namespace Harold_speed_is_one_more_l305_305296

variable (Adrienne_speed Harold_speed : ℝ)
variable (distance_when_Harold_catches_Adr : ℝ)
variable (time_difference : ℝ)

axiom Adrienne_speed_def : Adrienne_speed = 3
axiom Harold_catches_distance : distance_when_Harold_catches_Adr = 12
axiom time_difference_def : time_difference = 1

theorem Harold_speed_is_one_more :
  Harold_speed - Adrienne_speed = 1 :=
by 
  have Adrienne_time := (distance_when_Harold_catches_Adr - Adrienne_speed * time_difference) / Adrienne_speed 
  have Harold_time := distance_when_Harold_catches_Adr / Harold_speed
  have := Adrienne_time = Harold_time - time_difference
  sorry

end Harold_speed_is_one_more_l305_305296


namespace kimberly_skittles_proof_l305_305535

variable (SkittlesInitial : ℕ) (SkittlesBought : ℕ) (OrangesBought : ℕ)

/-- Kimberly's initial number of Skittles --/
def kimberly_initial_skittles := SkittlesInitial

/-- Skittles Kimberly buys --/
def kimberly_skittles_bought := SkittlesBought

/-- Oranges Kimbery buys (irrelevant for Skittles count) --/
def kimberly_oranges_bought := OrangesBought

/-- Total Skittles Kimberly has --/
def kimberly_total_skittles (SkittlesInitial SkittlesBought : ℕ) : ℕ :=
  SkittlesInitial + SkittlesBought

/-- Proof statement --/
theorem kimberly_skittles_proof (h1 : SkittlesInitial = 5) (h2 : SkittlesBought = 7) : 
  kimberly_total_skittles SkittlesInitial SkittlesBought = 12 :=
by
  rw [h1, h2]
  exact rfl

end kimberly_skittles_proof_l305_305535


namespace bridge_length_correct_l305_305731

-- Define the constants
def train_length : ℝ := 120
def time_cross_bridge : ℝ := 26.997840172786177
def speed_kmph : ℝ := 36

-- Define the conversion factor from km/h to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_mps : ℝ := speed_kmph * kmph_to_mps

-- Define the total distance covered (train length + bridge length)
def total_distance_covered : ℝ := train_speed_mps * time_cross_bridge

-- Define the length of the bridge as given by the problem
def bridge_length : ℝ := total_distance_covered - train_length

-- Prove that the computed bridge length equals the expected value.
theorem bridge_length_correct : bridge_length = 149.97840172786177 := by
  sorry

end bridge_length_correct_l305_305731


namespace mass_percentage_of_carbon_in_ccl4_l305_305009

-- Define the atomic masses
def atomic_mass_c : Float := 12.01
def atomic_mass_cl : Float := 35.45

-- Define the molecular composition of Carbon Tetrachloride (CCl4)
def mol_mass_ccl4 : Float := (1 * atomic_mass_c) + (4 * atomic_mass_cl)

-- Theorem to prove the mass percentage of carbon in Carbon Tetrachloride is 7.81%
theorem mass_percentage_of_carbon_in_ccl4 : 
  (atomic_mass_c / mol_mass_ccl4) * 100 = 7.81 := by
  sorry

end mass_percentage_of_carbon_in_ccl4_l305_305009


namespace exists_infinite_irregular_set_l305_305583

-- Define what it means for a set to be irregular
def is_irregular (A : Set ℤ) : Prop :=
  ∀ x y ∈ A, x ≠ y → ∀ k : ℤ, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

-- The main theorem stating the existence of an infinite irregular set
theorem exists_infinite_irregular_set : ∃ (A : Set ℤ), Set.infinite A ∧ is_irregular A :=
begin
  -- Proof to be constructed here
  sorry
end

end exists_infinite_irregular_set_l305_305583


namespace cookie_cost_is_19_l305_305785

def days_in_march : ℕ := 31
def cookies_per_day : ℕ := 4
def total_amount_spent : ℕ := 2356

theorem cookie_cost_is_19 :
  let total_cookies := days_in_march * cookies_per_day in
  let cost_per_cookie := total_amount_spent / total_cookies in
  cost_per_cookie = 19 := by
    sorry

end cookie_cost_is_19_l305_305785


namespace sum_of_possible_values_l305_305628

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 8) = 4) : ∃ S : ℝ, S = 8 :=
sorry

end sum_of_possible_values_l305_305628


namespace abs_slope_of_line_equal_area_split_l305_305232

theorem abs_slope_of_line_equal_area_split (r : ℝ) (a b c d e f : ℝ) :
  r = 4 →
  a = 10 ∧ b = 80 ∧ c = 13 ∧ d = 64 ∧ e = 15 ∧ f = 72 →
  abs (m : ℝ) (m ≠ 0 ∧ (line_passing_through_point_and_divides_area (a + c) / 2 (b + d) / 2 (r, [(a, b), (c, d), (e, f)]) = m)) =
  8 / 5 :=
by
  intro hr hcenters
  have h1 := eq.refl (8 / 5)
  exact h1

end abs_slope_of_line_equal_area_split_l305_305232


namespace intersection_circumcircle_iff_de_eq_df_l305_305549

variables {A B C D E F M N : Type*}

-- Assume existence of functions and instances meeting the conditions stated
noncomputable def circumcircle (ABC : Triangle) : Circle := sorry
noncomputable def is_angle_bisector (P Q R : Point) (AD : Line) : Prop := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def is_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
variable (ABC : Triangle)
variable (AD BE CF : Line)
variable (D E F : Point)
variable (M N : Point)

hypothesis (h1 : is_angle_bisector A B C AD)
hypothesis (h2 : is_angle_bisector B C A BE)
hypothesis (h3 : is_angle_bisector C A B CF)
hypothesis (h4 : intersects AD BC D)
hypothesis (h5 : intersects BE CA E)
hypothesis (h6 : intersects CF AB F)
hypothesis (h7 : M = midpoint B C)
hypothesis (h8 : N = midpoint E F)
hypothesis (h9 : is_parallel (line_through M (parallel_line_through AD)) AD)

-- Statement to prove
theorem intersection_circumcircle_iff_de_eq_df :
  ((exists P : Point, intersects (line_through A N) (circumcircle ABC) P) ↔ (distance E D = distance F D)) :=
sorry

end intersection_circumcircle_iff_de_eq_df_l305_305549


namespace AC_eq_BD_implies_rectangle_l305_305953

variable {A B C D : Type} 

-- Define what it means for a quadrilateral to be a parallelogram
def isParallelogram (ABCD : Quad) : Prop := 
  -- Conditions for being a parallelogram

-- Define what it means for a quadrilateral to be a rectangle
def isRectangle (ABCD : Quad) : Prop := 
  -- Conditions for being a rectangle

-- Statement we want to prove
theorem AC_eq_BD_implies_rectangle (ABCD : Quad) : isParallelogram ABCD → diag1 ABCD = diag2 ABCD → isRectangle ABCD :=
by
  sorry

end AC_eq_BD_implies_rectangle_l305_305953


namespace solutions_of_quadratic_eq_l305_305219

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l305_305219


namespace prove_tangent_value_a_l305_305413

noncomputable def tangent_value_a : Prop :=
  ∃ (a : ℝ) (m n : ℝ), (m - n + 1 = 0) ∧ (n = real.log m - a) ∧ (1 / m = 1) ∧ (m = 1) ∧ (n = 2) ∧ (a = -2)

theorem prove_tangent_value_a : tangent_value_a := sorry

end prove_tangent_value_a_l305_305413


namespace sqrt_n_polynomial_of_acute_triangle_l305_305265

open Real

noncomputable def polynomial (P : ℝ → ℝ) (n : ℕ) :=
  ∀ x, P x ≥ 0 ∧ ∃ q : polynomial ℝ, degree q = n ∧ ∀ x, q.eval x = P x

theorem sqrt_n_polynomial_of_acute_triangle (P : ℝ → ℝ) (n : ℕ) (a b c : ℝ)
  (hP : polynomial P n)
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a^2 < b^2 + c^2 ∧ b^2 < a^2 + c^2 ∧ c^2 < a^2 + b^2) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x = real.sqrt (P a ^ 2 / n) ∧ 
  y = real.sqrt (P b ^ 2 / n) ∧ 
  z = real.sqrt (P c ^ 2 / n) ∧ 
  x^2 < y^2 + z^2 ∧ y^2 < x^2 + z^2 ∧ z^2 < x^2 + y^2 :=
sorry

end sqrt_n_polynomial_of_acute_triangle_l305_305265


namespace range_of_m_l305_305862

def quadratic_nonnegative (m : ℝ) : Prop :=
∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0

theorem range_of_m (m : ℝ) :
  quadratic_nonnegative m ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l305_305862


namespace cats_kittentotal_l305_305139

def kittens_given_away : ℕ := 2
def kittens_now : ℕ := 6
def kittens_original : ℕ := 8

theorem cats_kittentotal : kittens_now + kittens_given_away = kittens_original := 
by 
  sorry

end cats_kittentotal_l305_305139


namespace basketball_weight_l305_305015

theorem basketball_weight (b s : ℝ) (h1 : s = 20) (h2 : 5 * b = 4 * s) : b = 16 :=
by
  sorry

end basketball_weight_l305_305015


namespace almost_uniform_convergence_implies_convergence_in_measure_and_pointwise_l305_305957

variable {Ω : Type*} {𝓕 : MeasurableSpace Ω} {μ : MeasureTheory.Measure Ω}
variable {f : Ω → ℝ} {fns : ℕ → Ω → ℝ}

-- Definition: almost uniform convergence
def almost_uniform_convergence (fns : ℕ → Ω → ℝ) (f : Ω → ℝ) (μ : MeasureTheory.Measure Ω) : Prop :=
  ∀ ε > 0, ∃ A ∈ MeasureTheory.Measure.sets μ, 
    μ A < ε ∧ ∀ n, ∀ ω ∈ set.compl A, dist (fns n ω) (f ω) < ε

-- Definition: convergence in measure
def convergence_in_measure (fns : ℕ → Ω → ℝ) (f : Ω → ℝ) (μ : MeasureTheory.Measure Ω) : Prop :=
  ∀ ε > 0, ∀ δ > 0, ∃ N, ∀ n ≥ N, μ {ω | dist (fns n ω) (f ω) ≥ ε} < δ

-- Definition: convergence almost everywhere
def convergence_almost_everywhere (fns : ℕ → Ω → ℝ) (f : Ω → ℝ) (μ : MeasureTheory.Measure Ω) : Prop :=
  ∀ᵐ ω ∂μ, ∃ N, ∀ n ≥ N, fns n ω = f ω

theorem almost_uniform_convergence_implies_convergence_in_measure_and_pointwise
  (h : almost_uniform_convergence fns f μ) :
  convergence_in_measure fns f μ ∧ convergence_almost_everywhere fns f μ :=
by
  sorry  -- Proof goes here

end almost_uniform_convergence_implies_convergence_in_measure_and_pointwise_l305_305957


namespace cos_alpha_value_l305_305018

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = -4 / 5) (h2 : -π / 2 < α ∧ α < 0) :
  cos α = (3 - 4 * real.sqrt 3) / 10 :=
sorry

end cos_alpha_value_l305_305018


namespace U_value_l305_305554

def f : ℕ → ℝ := sorry
axiom f_rec (n : ℕ) (h : n ≥ 6) : f n = (n - 1) * f (n - 1)
axiom f_nonzero (n : ℕ) (h : n ≥ 6) : f n ≠ 0

def U (T : ℕ) : ℝ := f T / ((T - 1) * f (T - 3))

theorem U_value (T : ℕ) (hT : T ≥ 6) : U T = 72 := sorry

end U_value_l305_305554


namespace f_f_10_eq_2_l305_305372

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else log 10 x

theorem f_f_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end f_f_10_eq_2_l305_305372


namespace prove_condition_l305_305271

noncomputable def condition_necessary_but_not_sufficient : Prop :=
∀ x: ℝ, (2^x < 1 → x < 2) ∧ (¬ (x < 2) → ¬ (2^x < 1))

theorem prove_condition : condition_necessary_but_not_sufficient :=
by
  sorry

end prove_condition_l305_305271


namespace meaningful_expression_range_l305_305454

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = x / real.sqrt (x + 2)) → x > -2 := 
by
  intros h
  sorry

end meaningful_expression_range_l305_305454


namespace supermarket_display_cans_l305_305116

theorem supermarket_display_cans : 
  ∃ d : ℕ, 
    (∑ i in finset.range 10, (19 - (6 - i) * d)) < 150 ∧ d = 3 :=
begin
  sorry
end

end supermarket_display_cans_l305_305116


namespace bisect_EF_l305_305046

-- Given an acute triangle ABC with circumcircle Γ
variable (A B C : Point)
variable (Γ : Circle)

-- Altitudes AD, BE, CF
variable (D : Point) (AD : Line)
variable (E : Point) (BE : Line)
variable (F : Point) (CF : Line)

-- Line AD intersects Γ again at P
variable (P : Point)

-- PF and PE intersect Γ again at R and Q respectively
variable (R : Point) (PF : Line)
variable (Q : Point) (PE : Line)

-- O_1 and O_2 are the circumcenters of △BFR and △CEQ respectively
variable (O₁ : Point) (circumcenter_BFR : Circle)
variable (O₂ : Point) (circumcenter_CEQ : Circle)

-- Definitions necessary to connect the problem
def acute_triangle := (acute_angle ∠ BAC) ∧ (acute_angle ∠ ABC) ∧ (acute_angle ∠ ACB)
def triangle_circumcircle := circle_contains_points Γ {A, B, C}
def orthocenter := meets_altitudes {AD, BE, CF} (O₁, A, BC)

-- Required proof statement
theorem bisect_EF
  (hAcute : acute_triangle A B C)
  (hCircum : triangle_circumcircle Γ A B C)
  (hAD : altitude AD A Γ) (hBE : altitude BE B Γ) (hCF : altitude CF C Γ)
  (hAD_Intersects : AD ∩ Γ = P)
  (hPF_Repeat : PF ∩ Γ = R) (hPE_Repeat : PE ∩ Γ = Q)
  (hCircum_BFR : circumcenter_BFR = O₁)
  (hCircum_CEQ : circumcenter_CEQ = O₂):
  bisects (O₁O₂) (EF) := sorry

end bisect_EF_l305_305046


namespace stickers_total_l305_305671

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l305_305671


namespace unsolvable_algorithm_l305_305300

theorem unsolvable_algorithm (S : ℕ → ℝ) (A : S 0 = 1 + 2 + 3 + 4)
    (B : S 1 = (∑ k in finset.range 101, k^2))
    (C : S 2 = ∑ k in finset.range 10001, 1/(k+1))
    (D : S 3 = ∑' k, k) :
  ¬ ∃ n, alg_solution (S 3 n) :=
sorry

end unsolvable_algorithm_l305_305300


namespace log_proof_l305_305095

theorem log_proof (x : ℝ) (h : log 5 (log 4 (log 3 x)) = 0) : x ^ (-1 / 3) = 1 / 3 :=
  sorry

end log_proof_l305_305095


namespace candy_cost_l305_305699

theorem candy_cost
  (C : ℝ) -- cost per pound of the first candy
  (w1 : ℝ := 30) -- weight of the first candy
  (c2 : ℝ := 5) -- cost per pound of the second candy
  (w2 : ℝ := 60) -- weight of the second candy
  (w_mix : ℝ := 90) -- total weight of the mixture
  (c_mix : ℝ := 6) -- desired cost per pound of the mixture
  (h1 : w1 * C + w2 * c2 = w_mix * c_mix) -- cost equation for the mixture
  : C = 8 :=
by
  sorry

end candy_cost_l305_305699


namespace ivar_total_water_needed_l305_305135

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l305_305135


namespace calc_part_1_calc_part_2_l305_305320

theorem calc_part_1 : (∛8 + sqrt 9 - sqrt (1/4)) = 9/2 := sorry

theorem calc_part_2 : (abs (sqrt 3 - sqrt 5) + 2 * sqrt 3) = sqrt 5 + sqrt 3 := sorry

end calc_part_1_calc_part_2_l305_305320


namespace area_triangle_DBC_l305_305518

noncomputable def point := (ℝ × ℝ)

noncomputable def midpoint (P Q : point) : point :=
((P.1 + Q.1)/2, (P.2 + Q.2)/2)

noncomputable def area_triangle (A B C : point) : ℝ :=
0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_triangle_DBC :
  let A := (0,8) : point
  let B := (0,0) : point
  let C := (10,0) : point
  let D := midpoint A B
  -- E := midpoint B C -- E is not necessary to compute area of triangle DBC
  in area_triangle D B C = 20 :=
by
  sorry

end area_triangle_DBC_l305_305518


namespace triangle_XYZ_XY_l305_305529

theorem triangle_XYZ_XY :
  ∀ (X Y Z : Type) [linear_ordered_field X] [linear_ordered_field Y] [linear_ordered_field Z] 
  (YZ XY XZ : ℝ) (angle_X : ℝ) (tan_Z cos_Y : ℝ), 
  angle_X = 90 ∧ YZ = 25 ∧ tan_Z = 7 * cos_Y ∧ tan_Z = XY / XZ ∧ cos_Y = XY / YZ → 
  XY = 100 * sqrt 3 / 7 :=
begin
  sorry
end

end triangle_XYZ_XY_l305_305529


namespace probability_heads_tails_4_tosses_l305_305663

-- Define the probabilities of heads and tails
variables (p q : ℝ)

-- Define the conditions
def unfair_coin (p q : ℝ) : Prop :=
  p ≠ q ∧ p + q = 1 ∧ 2 * p * q = 1/2

-- Define the theorem to prove the probability of two heads and two tails
theorem probability_heads_tails_4_tosses 
  (h_unfair : unfair_coin p q) 
  : 6 * (p * q)^2 = 3 / 8 :=
by sorry

end probability_heads_tails_4_tosses_l305_305663


namespace max_gum_pieces_l305_305591

-- Define the given conditions.
def nickels := 5
def dimes := 6
def quarters := 4

def exchange_rate_nickel := 2
def exchange_rate_dime := 3
def exchange_rate_quarter := 5

def max_nickels := 3
def max_dimes := 4
def max_quarters := 2

def remaining_nickels := 2
def remaining_dimes := 1

def discount1 := 20
def discount2 := 0.10
def bonus3 := 5

-- Prove that the maximum number of gum pieces is 31.
theorem max_gum_pieces : 
  let gum_nickels := max_nickels * exchange_rate_nickel,
      gum_dimes := (dimes - remaining_dimes) * exchange_rate_dime,
      gum_quarters := max_quarters * exchange_rate_quarter,
      total_gum := gum_nickels + gum_dimes + gum_quarters
  in total_gum = 31 := 
sorry

end max_gum_pieces_l305_305591


namespace total_journey_time_l305_305679

-- Given conditions
variables (river_speed boat_speed : ℝ) (distance : ℝ)
variables (h_river_speed : river_speed = 2)
variables (h_boat_speed : boat_speed = 6)
variables (h_distance : distance = 56)

-- To be proved
theorem total_journey_time :
  let upstream_speed := boat_speed - river_speed in
  let downstream_speed := boat_speed + river_speed in
  let time_upstream := distance / upstream_speed in
  let time_downstream := distance / downstream_speed in
  time_upstream + time_downstream = 21 :=
by
  sorry

end total_journey_time_l305_305679


namespace find_lambda_for_collinear_vectors_l305_305835

theorem find_lambda_for_collinear_vectors 
  (a : ℝ × ℝ := (1,1)) 
  (b : ℝ × ℝ := (2,3)) 
  (c : ℝ × ℝ := (-7,-8))
  (h: ∃ λ : ℝ, (λ * a.1 - b.1, λ * a.2 - b.2) = λ • a - b ∧ is_collinear (λ * a.1 - b.1, λ * a.2 - b.2) c) :
  ∃ λ : ℝ, λ = -5 := 
sorry

end find_lambda_for_collinear_vectors_l305_305835


namespace distribution_ways_l305_305842

theorem distribution_ways :
  ∃ (n : ℕ), n = 56 ∧
  ∀ (balls boxes : ℕ), balls = 5 → boxes = 4 →
  (nat.choose (balls + boxes - 1) (boxes - 1)) = 56 :=
by
  sorry

end distribution_ways_l305_305842


namespace sqrt_eq_sum_iff_conditions_l305_305239

theorem sqrt_eq_sum_iff_conditions (a b c : ℝ) :
  sqrt (a^2 + b^2 + c^2) = a + b + c ↔ (ab + bc + ca = 0 ∧ a + b + c ≥ 0) :=
by
  sorry

end sqrt_eq_sum_iff_conditions_l305_305239


namespace scrap_cookie_radius_l305_305770

-- Definitions based on the problem conditions
def original_radius : ℝ := 4
def small_cookie_radius : ℝ := 1
def original_area : ℝ := π * original_radius^2
def num_small_cookies : ℕ := 8
def small_cookie_area : ℝ := π * small_cookie_radius^2
def total_small_cookie_area : ℝ := num_small_cookies * small_cookie_area
def leftover_area : ℝ := original_area - total_small_cookie_area
def scrap_radius := (leftover_area / π).sqrt

-- Theorem stating that the radius of the scrap cookie is \(2\sqrt{2}\) inches
theorem scrap_cookie_radius : scrap_radius = 2 * Real.sqrt 2 := by
  sorry

end scrap_cookie_radius_l305_305770


namespace find_y_l305_305847

theorem find_y (x y : ℤ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 :=
sorry

end find_y_l305_305847


namespace marker_cost_l305_305462

theorem marker_cost (s n c : ℕ) (h_majority : s > 20) (h_markers : n > 1) (h_cost : c > n) (h_total_cost : s * n * c = 3388) : c = 11 :=
by {
  sorry
}

end marker_cost_l305_305462


namespace light_bulb_arrangement_l305_305633

theorem light_bulb_arrangement :
  let B := 6
  let R := 7
  let W := 9
  let total_arrangements := Nat.choose (B + R) B * Nat.choose (B + R + 1) W
  total_arrangements = 3435432 :=
by
  sorry

end light_bulb_arrangement_l305_305633


namespace max_food_is_one_l305_305619

noncomputable def max_food_per_guest (total_food : ℝ) (number_of_guests : ℝ) : ℝ :=
  Real.floor (total_food / number_of_guests)

theorem max_food_is_one 
  (total_food : ℝ) (number_of_guests : ℝ) 
  (h1 : total_food = 323) 
  (h2 : number_of_guests = 162) : max_food_per_guest total_food number_of_guests = 1 :=
by 
  rw [max_food_per_guest, h1, h2]
  norm_num
  rfl

end max_food_is_one_l305_305619


namespace inequality_proof_l305_305182

theorem inequality_proof (x y : ℝ) :
  abs ((x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2))) ≤ 1 / 2 := 
sorry

end inequality_proof_l305_305182


namespace hypotenuse_length_l305_305497

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305497


namespace valid_arrangements_l305_305652

-- Helper definitions for squares and their relationships
structure Square := (id : ℕ)

variable (M N L A B C : Square)
variable (connected : Square → Square → Prop)
variable (higher : Square → Square → Prop)

-- Conditions given in the problem
axiom connected_squares : connected M N ∧ connected N L ∧ connected A B ∧ connected B C ∧ connected M A ∧ connected N B ∧ connected L C ∧ (∀ x y : Square, connected x y → higher x y)

-- Define the problem statement: prove there are exactly 12 valid assignments
theorem valid_arrangements : ∃ (assignment : Square → ℕ), 
  (∀ s : Square, assignment s ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  set.inj_on assignment {M, N, L, A, B, C} ∧
  ∀ s₁ s₂ : Square, higher s₁ s₂ → assignment s₁ > assignment s₂ ∧ ∑ t : Square, if (∃ x : ℕ, assignment t = x) then 1 else 0 = 12 :=
sorry

end valid_arrangements_l305_305652


namespace ABCD_perimeter_l305_305126

open Real -- Using the Real number domain

theorem ABCD_perimeter :
  ∀ (A B C D E : Type) (AE BE AB ED CD DA: ℝ),
    (BE = 20 * sqrt 2) ∧  -- given by right triangle BE
    (AB = 20 * sqrt 2) ∧  -- given by right triangle AB
    (ED = 10 * sqrt 2) ∧  -- given by 30-60-90 properties
    (CD = 10 * sqrt 6) ∧  -- given by 30-60-90 properties
    (DA = 40 + 10 * sqrt 2) → -- given by sum of DE + EA
    AE = 40 → -- given AE
    BE = AB →
    BE / 2 = ED →   -- property of 30-60-90 triangle
    ED * sqrt 3 = CD → -- property of 30-60-90 triangle
  (AB + BE + CD + DA = 40 + 50 * sqrt 2 + 10 * sqrt 6) := 
begin
  intros, 
  sorry  -- Proof skipped
end

end ABCD_perimeter_l305_305126


namespace alternatingSeqCount_l305_305280

definition isAlternatingSeq (s : List ℕ) : Prop :=
  (s.length = 5) ∧ (∀ i, i < 4 → ((i % 2 = 0 → s[i] > s[i + 1]) ∧ (i % 2 = 1 → s[i] < s[i + 1])) 
  ∨ (i % 2 = 0 → s[i] < s[i + 1]) ∧ (i % 2 = 1 → s[i] > s[i + 1]))

def numAlternatingSeqs : Nat :=
  32 * Nat.choose 20 5

theorem alternatingSeqCount :
  (List.filter (λ s => isAlternatingSeq s) (List.permutations [1,2,..,20]).length = numAlternatingSeqs := 
sorry

end alternatingSeqCount_l305_305280


namespace intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l305_305078

def l1 (x y : ℝ) : Prop := x + y = 2
def l2 (x y : ℝ) : Prop := x - 3 * y = -10
def l3 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

def M : (ℝ × ℝ) := (-1, 3)

-- Part (Ⅰ): Prove that M is the intersection point of l1 and l2
theorem intersection_l1_l2 : l1 M.1 M.2 ∧ l2 M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅱ): Prove the equation of the line passing through M and parallel to l3 is 3x - 4y + 15 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y + 15 = 0

theorem line_parallel_to_l3 : parallel_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅲ): Prove the equation of the line passing through M and perpendicular to l3 is 4x + 3y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

theorem line_perpendicular_to_l3 : perpendicular_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

end intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l305_305078


namespace find_natural_number_l305_305716

theorem find_natural_number (A d : ℕ) (h1 : ∀ k, k ∈ divisors A → k ≠ A → d = k) (h2 : ∀ k, k ∈ divisors (A + 2) → k ≠ (A + 2) → d + 2 = k) : A = 7 :=
sorry

end find_natural_number_l305_305716


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305503

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305503


namespace copy_pages_15_dollars_l305_305859

theorem copy_pages_15_dollars (cpp : ℕ) (budget : ℕ) (pages : ℕ) (h1 : cpp = 3) (h2 : budget = 1500) (h3 : pages = budget / cpp) : pages = 500 :=
by
  sorry

end copy_pages_15_dollars_l305_305859


namespace max_circle_sum_l305_305308

-- Define the seven numbered regions
def regions := {0, 1, 2, 3, 4, 5, 6}

-- Define the three circles, each circle consisting of four positions that add up to the same sum
def circle_a (a b c d : ℕ) := {a, b, c, d} -- Sum for circle A
def circle_b (e f g h : ℕ) := {e, f, g, h} -- Sum for circle B
def circle_c (i j k l : ℕ) := {i, j, k, l} -- Sum for circle C

-- Define the integers from 0 to 6
def integers_sum_to_21 := (0 + 1 + 2 + 3 + 4 + 5 + 6 = 21)

-- Define the conditions for the problem
theorem max_circle_sum (s : Finset ℕ) 
    (h₁ : s = {0, 1, 2, 3, 4, 5, 6})
    (a b c d e f g h i j k l : ℕ)
    (h₂ : 6 ∈ {a, b, c, d})
    (h₃ : 6 ∈ {e, f, g, h})
    (h₄ : 6 ∈ {i, j, k, l})
    (h₅ : a ≠ e ∧ b ≠ f ∧ c ≠ g ∧ d ≠ h)
    (h₆ : integers_sum_to_21)
    : 
    (a + b + c + d = e + f + g + h) ∧
    (e + f + g + h = i + j + k + l) ∧
    ((a + b + c + d) = 15) ∧
    ((a + b + c + d) = (e + f + g + h) ∧
    (e + f + g + h) = (i + j + k + l)) :=
  by sorry

end max_circle_sum_l305_305308


namespace probability_of_winning_l305_305210

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end probability_of_winning_l305_305210


namespace rectangle_same_color_l305_305772

theorem rectangle_same_color (colors : Finset ℕ) (h_col : colors.card = 4)
  (coloring : Fin 5 × Fin 41 → colors) :
  ∃ (p1 p2 p3 p4 : Fin 5 × Fin 41),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧
    (coloring p1 = coloring p2 ∧ coloring p2 = coloring p3 ∧ coloring p3 = coloring p4) ∧
    ((p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2) ∨
    (p1.1 = p3.1 ∧ p2.1 = p4.1 ∧ p1.2 = p2.2 ∧ p3.2 = p4.2)) :=
by
  sorry

end rectangle_same_color_l305_305772


namespace puppy_food_consumption_l305_305924

/-- Mathematically equivalent proof problem:
  Given the following conditions:
  1. days_per_week = 7
  2. initial_feeding_duration_weeks = 2
  3. initial_feeding_daily_portion = 1/4
  4. initial_feeding_frequency_per_day = 3
  5. subsequent_feeding_duration_weeks = 2
  6. subsequent_feeding_daily_portion = 1/2
  7. subsequent_feeding_frequency_per_day = 2
  8. today_feeding_portion = 1/2
  Prove that the total food consumption, including today, over the next 4 weeks is 25 cups.
-/
theorem puppy_food_consumption :
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  total = 25 := by
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  show total = 25 from sorry

end puppy_food_consumption_l305_305924


namespace tom_cannot_be_in_middle_seat_l305_305306

def people := ["Andy", "Jen", "Sally", "Mike", "Tom"]

def is_beside (a b : String) (arrangement : List String) : Prop :=
  ∃ i, i < arrangement.length - 1 ∧ 
       (arrangement.get i = a ∧ arrangement.get (i + 1) = b ∨ 
        arrangement.get i = b ∧ arrangement.get (i + 1) = a)

def is_not_beside (a b : String) (arrangement : List String) : Prop :=
  ¬ is_beside a b arrangement

def is_beside_condition (arrangement : List String) : Prop :=
  is_beside "Sally" "Mike" arrangement

def is_not_beside_condition (arrangement : List String) : Prop :=
  is_not_beside "Andy" "Jen" arrangement

def middle_seat (arrangement : List String) : String := arrangement.get 2

def tom_not_in_middle (arrangement : List String) : Prop :=
  middle_seat arrangement ≠ "Tom"

theorem tom_cannot_be_in_middle_seat :
  ∀ (arrangement : List String), 
  arrangement.length = 5 → 
  is_beside_condition arrangement → 
  is_not_beside_condition arrangement → 
  tom_not_in_middle arrangement :=
begin
  intros,
  sorry
end

end tom_cannot_be_in_middle_seat_l305_305306


namespace distinct_arrangements_of_seventeen_girls_in_circle_l305_305181

theorem distinct_arrangements_of_seventeen_girls_in_circle : fact (nat.facts 16! : nat) := sorry

end distinct_arrangements_of_seventeen_girls_in_circle_l305_305181


namespace sufficient_but_not_necessary_condition_l305_305999

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l305_305999


namespace water_added_l305_305468

-- Define parameters and conditions
def initial_volume : ℝ := 45
def initial_milk_ratio : ℝ := 4
def initial_water_ratio : ℝ := 1
def new_ratio : ℝ := 1.125

-- Finding the amount of water added to achieve the new ratio
theorem water_added :
  ∃ W : ℝ, (let milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume,
                 initial_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume,
                 new_water_volume := initial_water_volume + W
             in milk_volume / new_water_volume = new_ratio) ∧
            W = 23 :=
by
  -- Proof would go here
  sorry

end water_added_l305_305468


namespace division_result_l305_305771

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end division_result_l305_305771


namespace symmetric_axis_l305_305829

def f (a b x : ℝ) := a * Real.sin x + b * Real.cos x

theorem symmetric_axis (a b x₀ : ℝ) 
  (h₁ : ∃ x₀ : ℝ, ∀ x : ℝ, f a b (2*x₀ - x) = f a b x)
  (h₂ : Real.tan x₀ = 2) : 
  a = 2 * b := 
sorry

end symmetric_axis_l305_305829


namespace abc_divisible_by_7_l305_305621

theorem abc_divisible_by_7 (a b c : ℤ) (h : 7 ∣ (a^3 + b^3 + c^3)) : 7 ∣ (a * b * c) :=
sorry

end abc_divisible_by_7_l305_305621


namespace sample_size_is_24_l305_305277

noncomputable def total_elderly := 20
noncomputable def total_middle_aged := 120
noncomputable def total_young := 100
noncomputable def young_in_sample := 10

theorem sample_size_is_24 :
  ∃ n : ℕ, let ratio := 1 + 6 + 5 in
           let proportion_young := 5 / ratio in
           young_in_sample = 10 → 
           n = young_in_sample / proportion_young := 
begin
  use 24,
  intros,
  sorry,
end

end sample_size_is_24_l305_305277


namespace coefficient_of_x_neg_2_in_binomial_expansion_l305_305609

theorem coefficient_of_x_neg_2_in_binomial_expansion :
  let x := (x : ℚ)
  let term := (x^3 - (2 / x))^6
  (coeff_of_term : Int) ->
  (coeff_of_term = -192) :=
by
  -- Placeholder for the proof
  sorry

end coefficient_of_x_neg_2_in_binomial_expansion_l305_305609


namespace correct_proposition3_l305_305388

-- Define the proposition (1)
def proposition1 (α β : Type*) [plane α] [plane β] (line1 line2 line3 line4 : line) : Prop :=
  parallel line1 line3 ∧ parallel line2 line4 → angle_eq angle1 angle2

-- Define the proposition (2)
def proposition2 (α β : Type*) [plane α] [plane β] (a b : line) : Prop :=
  parallel a α ∧ parallel b α ∧ parallel a b → parallel a b

-- Define the proposition (3)
def proposition3 (α β : Type*) [plane α] [plane β] (m : line) : Prop :=
  perpendicular m α ∧ perpendicular m β → parallel α β

-- Define the proposition (4)
def proposition4 (α β : Type*) [plane α] [plane β] (m : line) : Prop :=
  parallel m α ∧ parallel m β → parallel α β

-- The final proposition statement.
theorem correct_proposition3 (α β : Type*) [plane α] [plane β] (m : line) : 
  perpendicular m α ∧ perpendicular m β → parallel α β :=
begin
  exact proposition3 α β m,
  sorry
end

end correct_proposition3_l305_305388


namespace intersection_of_complements_l305_305559

open Set

variable (U M N : Set ℕ)

theorem intersection_of_complements :
  U = {1, 2, 3, 4, 5} →
  M = {1, 2, 4} →
  N = {2, 4, 5} →
  (U \ M) ∩ (U \ N) = {3} :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  rw [diff_eq, diff_eq]
  rw [Inter_eq, Union_eq, compl_eq_mono]
  simp
  sorry

end intersection_of_complements_l305_305559


namespace crop_yield_solution_l305_305524

variable (x y : ℝ)

axiom h1 : 3 * x + 6 * y = 4.7
axiom h2 : 5 * x + 3 * y = 5.5

theorem crop_yield_solution :
  x = 0.9 ∧ y = 1/3 :=
by
  sorry

end crop_yield_solution_l305_305524


namespace polar_to_cartesian_l305_305200

-- Define the equation in polar coordinates 
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * sin θ - ρ * cos θ^2 - sin θ = 0

-- Define the condition on θ
def theta_condition (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < π

-- Define the Cartesian condition conversion
def cartesian_equation (x y : ℝ) : Prop :=
  x = 1 ∨ (x^2 + y^2 + y = 0 ∧ y ≠ 0)

-- Prove the equivalence
theorem polar_to_cartesian (ρ θ x y : ℝ) (h₁ : polar_equation ρ θ) (h₂ : theta_condition θ) :
  (x = ρ * cos θ ∧ y = ρ * sin θ) ↔ cartesian_equation x y :=
sorry

end polar_to_cartesian_l305_305200


namespace slope_of_line_l_passing_through_focus_l305_305385

theorem slope_of_line_l_passing_through_focus 
  (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : a^2 = 3 * c^2) 
  (line : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus : focus = (-c, 0)) 
  (intersect_A : ℝ × ℝ) 
  (intersect_B : ℝ × ℝ) 
  (h_A_on_ellipse : (intersect_A.1^2 / a^2 + intersect_A.2^2 / b^2 = 1))
  (h_B_on_ellipse : (intersect_B.1^2 / a^2 + intersect_B.2^2 / b^2 = 1))
  (h_line_through_focus : ∀ x : ℝ, line x = (intersect_A.1 + intersect_B.1) / 2 * x + intersect_A.2 - (intersect_A.1 + intersect_B.1) / 2 * intersect_A.1)
  (h_ratio_AF_FB : (intersect_A.1 + c) ^ 2 + intersect_A.2 ^ 2 = 9 * ((intersect_B.1 + c) ^ 2 + intersect_B.2 ^ 2)) :
  (line (focus.1 + c)) = (focus.2 + sqrt(3) / 3) ∨ line (focus.1 + c) = (focus.2 - sqrt(3) / 3) :=
by sorry

end slope_of_line_l_passing_through_focus_l305_305385


namespace triangle_is_isosceles_l305_305993

theorem triangle_is_isosceles
  (p q r : ℝ)
  (H : ∀ (n : ℕ), n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n))
  : p = q ∨ q = r ∨ r = p := 
begin
  sorry
end

end triangle_is_isosceles_l305_305993


namespace remainder_property_l305_305246

theorem remainder_property (a : ℤ) (h : ∃ k : ℤ, a = 45 * k + 36) :
  ∃ n : ℤ, a = 45 * n + 36 :=
by {
  sorry
}

end remainder_property_l305_305246


namespace mean_temperature_correct_l305_305623

-- Define the list of temperatures
def temperatures : List ℝ := [82, 83, 78, 86, 88, 90, 88]

-- Define the target mean temperature
def target_mean := 84.5714

theorem mean_temperature_correct :
  (List.sum temperatures / temperatures.length = target_mean) :=
by
  -- Summarize the steps required to solve the problem
  -- (these are given to prove that this statement can be completed)
  sorry

end mean_temperature_correct_l305_305623


namespace sequence_correct_sequence_initial_1_sequence_initial_2_sequence_general_formula_l305_305419

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2^n - n

theorem sequence_correct (n : ℕ) (hn : n ≥ 2) : 
  sequence (n+1) - 3 * sequence n + 2 * sequence (n-1) = 1 :=
by sorry

theorem sequence_initial_1 : sequence 1 = 1 := rfl

theorem sequence_initial_2 : sequence 2 = 2 := rfl

theorem sequence_general_formula (n : ℕ) : sequence n = 2^n - n :=
by sorry

end sequence_correct_sequence_initial_1_sequence_initial_2_sequence_general_formula_l305_305419


namespace broadway_show_total_amount_collected_l305_305690

theorem broadway_show_total_amount_collected (num_adults num_children : ℕ) 
  (adult_ticket_price child_ticket_ratio : ℕ) 
  (child_ticket_price : ℕ) 
  (h1 : num_adults = 400) 
  (h2 : num_children = 200) 
  (h3 : adult_ticket_price = 32) 
  (h4 : child_ticket_ratio = 2) 
  (h5 : adult_ticket_price = child_ticket_ratio * child_ticket_price) : 
  num_adults * adult_ticket_price + num_children * child_ticket_price = 16000 := 
  by 
    sorry

end broadway_show_total_amount_collected_l305_305690


namespace num_individuals_eliminated_l305_305641

theorem num_individuals_eliminated (pop_size : ℕ) (sample_size : ℕ) :
  (pop_size % sample_size) = 2 :=
by
  -- Given conditions
  let pop_size := 1252
  let sample_size := 50
  -- Proof skipped
  sorry

end num_individuals_eliminated_l305_305641


namespace alpha_plus_beta_l305_305053

theorem alpha_plus_beta (α β : ℝ)
  (h1 : tan α = -1 - tan β)
  (h2 : tan β = -2 / tan α)
  (h3 : -π / 2 < α ∧ α < π / 2)
  (h4 : -π / 2 < β ∧ β < π / 2) :
  α + β = π / 6 ∨ α + β = -5 * π / 6 := sorry

end alpha_plus_beta_l305_305053


namespace minimize_expr_l305_305352

-- Define the expression to be minimized
def my_expr (a b : ℝ) : ℝ :=
  (|a + 3 * b - b * (a + 9 * b)| + |3 * b - a + 3 * b * (a - b)|) / sqrt (a^2 + 9 * b^2)

-- Define the problem statement
theorem minimize_expr :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ my_expr a b = sqrt 10 / 5 :=
sorry

end minimize_expr_l305_305352


namespace music_store_cellos_l305_305276

/-- 
A certain music store stocks 600 violas. 
There are 100 cello-viola pairs, such that a cello and a viola were both made with wood from the same tree. 
The probability that the two instruments are made with wood from the same tree is 0.00020833333333333335. 
Prove that the store stocks 800 cellos.
-/
theorem music_store_cellos (V : ℕ) (P : ℕ) (Pr : ℚ) (C : ℕ) 
  (h1 : V = 600) 
  (h2 : P = 100) 
  (h3 : Pr = 0.00020833333333333335) 
  (h4 : Pr = P / (C * V)): C = 800 :=
by
  sorry

end music_store_cellos_l305_305276


namespace find_AC_l305_305471

theorem find_AC
  (A B C H X Y : Type)
  (AC AB AX AY : ℝ)
  (h1 : ∠BAC = 90)
  (h2 : Altitude AH from A to BC)
  (h3 : Circle passing through A and H intersects AB at X and intersects AC at Y)
  (h4 : AX = 5)
  (h5 : AY = 6)
  (h6 : AB = 9) :
  AC = 13.5 :=
sorry

end find_AC_l305_305471


namespace rectangle_other_side_l305_305603

theorem rectangle_other_side
  (a b : ℝ)
  (Area : ℝ := 12 * a ^ 2 - 6 * a * b)
  (side1 : ℝ := 3 * a)
  (side2 : ℝ := Area / side1) :
  side2 = 4 * a - 2 * b :=
by
  sorry

end rectangle_other_side_l305_305603


namespace minimize_fencing_l305_305931

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end minimize_fencing_l305_305931


namespace Jackson_missed_one_wednesday_l305_305896

theorem Jackson_missed_one_wednesday (weeks total_sandwiches missed_fridays sandwiches_eaten : ℕ) 
  (h1 : weeks = 36)
  (h2 : total_sandwiches = 2 * weeks)
  (h3 : missed_fridays = 2)
  (h4 : sandwiches_eaten = 69) :
  (total_sandwiches - missed_fridays - sandwiches_eaten) / 2 = 1 :=
by
  -- sorry to skip the proof.
  sorry

end Jackson_missed_one_wednesday_l305_305896


namespace hypotenuse_length_l305_305478

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305478


namespace solve_for_z_l305_305849

variable (x y z : ℝ)

theorem solve_for_z (h : 1 / x - 1 / y = 1 / z) : z = x * y / (y - x) := 
sorry

end solve_for_z_l305_305849


namespace minimum_radius_l305_305049

noncomputable theory
open_locale real

theorem minimum_radius (O M P : ℝ × ℝ) (r : ℝ) (P_on_circle : (P.1 - 5)^2 + (P.2 - 4)^2 = r^2) (r_pos : r > 0) (dist_PO_PM : (P.1^2 + P.2^2).sqrt = real.sqrt 2 * ((P.1 - 1)^2 + P.2^2).sqrt) : 
r = 5 - real.sqrt 2 :=
by {
  -- This is just a stub to represent the proof. Actual proof steps are omitted.
  sorry
}

end minimum_radius_l305_305049


namespace find_g_seven_l305_305101

variable {g : ℝ → ℝ}

theorem find_g_seven (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 :=
by
  sorry

end find_g_seven_l305_305101


namespace polygon_sides_arithmetic_sequence_l305_305104

theorem polygon_sides_arithmetic_sequence 
  (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : 2 * (180 * (n - 2)) = n * (100 + 140)) :
  n = 6 :=
  sorry

end polygon_sides_arithmetic_sequence_l305_305104


namespace little_john_height_l305_305566

theorem little_john_height :
  let m := 2 
  let cm_to_m := 8 * 0.01
  let mm_to_m := 3 * 0.001
  m + cm_to_m + mm_to_m = 2.083 := 
by
  sorry

end little_john_height_l305_305566


namespace compare_root_cubed_and_squared_l305_305748

-- Define the cube root and square root
def cube_root (x : ℝ) : ℝ := real.cbrt x
def square_root (x : ℝ) : ℝ := real.sqrt x

-- Define the constants
def root_cubed_3 := cube_root 3
def root_squared_2 := square_root 2

-- The statement to prove
theorem compare_root_cubed_and_squared : root_cubed_3 > root_squared_2 := by
  sorry

end compare_root_cubed_and_squared_l305_305748


namespace project_work_time_ratio_l305_305966

theorem project_work_time_ratio (A B C : ℕ) (h_ratio : A = x ∧ B = 2 * x ∧ C = 3 * x) (h_total : A + B + C = 120) : 
  (C - A = 40) :=
by
  sorry

end project_work_time_ratio_l305_305966


namespace divisor_count_fifth_power_mod_5_l305_305445

theorem divisor_count_fifth_power_mod_5 (
  n : ℕ) (h_pos : 0 < n) :
  let x := n ^ 5 in
  let d := (x.factors.foldr (λ p acc, (multiplicity p x : ℕ) + 1) 1) in
  d % 5 = 1 :=
by
  let x := n ^ 5
  let d := (x.factors.foldr (λ p acc, (multiplicity p x : ℕ) + 1) 1)
  have h_mod_5 : d % 5 = 1 := sorry
  exact h_mod_5

end divisor_count_fifth_power_mod_5_l305_305445


namespace vertex_to_diagonal_distance_le_half_diagonal_l305_305171

-- Assume A, B, C, D are points in a convex quadrilateral,
-- and AC, BD are the diagonals.
variables (A B C D A1 C1 : Point)
variable [convex_quadrilateral A B C D]
variable (AC BD : ℝ)

-- Let AA1 and CC1 be the perpendicular distances from A and C to BD
variable (AA1 CC1 : ℝ)

-- Given conditions
variables (h1 : AC ≤ BD) (h2 : AA1 + CC1 ≤ AC)

-- We need to prove that the distance from one of the vertices A or C
-- to the diagonal BD does not exceed half of that diagonal.
theorem vertex_to_diagonal_distance_le_half_diagonal : AA1 ≤ BD / 2 ∨ CC1 ≤ BD / 2 :=
by
  sorry

end vertex_to_diagonal_distance_le_half_diagonal_l305_305171


namespace possible_triangular_frames_B_l305_305230

-- Define the sides of the triangles and the similarity condition
def similar_triangles (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * b₃ = a₃ * b₁ ∧ a₂ * b₃ = a₃ * b₂

def sides_of_triangle_A := (50, 60, 80)

def is_a_possible_triangle (b₁ b₂ b₃ : ℕ) : Prop :=
  similar_triangles 50 60 80 b₁ b₂ b₃

-- Given conditions
def side_of_triangle_B := 20

-- Theorem to prove
theorem possible_triangular_frames_B :
  ∃ (b₂ b₃ : ℕ), (is_a_possible_triangle 20 b₂ b₃ ∨ is_a_possible_triangle b₂ 20 b₃ ∨ is_a_possible_triangle b₂ b₃ 20) :=
sorry

end possible_triangular_frames_B_l305_305230


namespace quadratic_square_binomial_l305_305766

theorem quadratic_square_binomial (d : ℝ) (h : ∃ b : ℝ, (x : ℝ) (x^2 + 80 * x + d = (x + b)^2)) : d = 1600 := by
  sorry

end quadratic_square_binomial_l305_305766


namespace candy_distribution_l305_305761

theorem candy_distribution (candies : ℕ) (family_members : ℕ) (required_candies : ℤ) :
  (candies = 45) ∧ (family_members = 5) →
  required_candies = 0 :=
by sorry

end candy_distribution_l305_305761


namespace balls_in_boxes_l305_305091

theorem balls_in_boxes : 
  ∃ n : ℕ, n = 6 ∧ (∃ m : ℕ, m = 4) ∧
  (∀ (balls boxes : ℕ), balls = 6 → boxes = 4 → 
    (number_of_ways balls boxes = 9)) :=
by {
  sorry
}

end balls_in_boxes_l305_305091


namespace range_of_m_triangle_function_l305_305192

noncomputable def is_triangle_function (f : ℝ → ℝ) (A : set ℝ) : Prop :=
  ∀ a b c ∈ A, let l1 := f a, l2 := f b, l3 := f c in
  l1 + l2 > l3 ∧ l1 + l3 > l2 ∧ l2 + l3 > l1

noncomputable def f_x (x m : ℝ) : ℝ := x * Real.log x + m

theorem range_of_m_triangle_function :
  ∃ (m : ℝ), (∀ a b c ∈ (set.Icc (1 / Real.exp 2) Real.exp), is_triangle_function (f_x m) (set.Icc (1 / Real.exp 2) Real.exp)) →
    m ∈ set.Ioi ((Real.exp ^ 2 + 2) / Real.exp) :=
sorry

end range_of_m_triangle_function_l305_305192


namespace probability_at_least_seven_stayed_l305_305451

theorem probability_at_least_seven_stayed (a b : ℕ) (p : ℚ) (h_a : a = 4) (h_b : b = 4) (h_p : p = 3/7) :
  (4.choose 3 * (p^3) * (1-p) + p^4) = 513/2401 :=
by
  sorry

end probability_at_least_seven_stayed_l305_305451


namespace angle_magnification_l305_305920

theorem angle_magnification (α : ℝ) (h : α = 20) : α = 20 := by
  sorry

end angle_magnification_l305_305920


namespace abs_A_minus_B_l305_305531

noncomputable def position (n : ℕ) : ℝ :=
if n % 2 = 0 then
  let a k := if k = 0 then 0 else (1/3 : ℝ) * position (2 * k - 1)
  a (n / 2)
else
  let b k := if k = 1 then (2/3) * 3 else (1/3 : ℝ) * position (2 * (k - 1)) + 2
  b ((n + 1) / 2)

def A : ℝ := lim (λ n, position (2 * n))

def B : ℝ := lim (λ n, position (2 * n - 1))

theorem abs_A_minus_B : |A - B| = 4 / 5 :=
sorry

end abs_A_minus_B_l305_305531


namespace sum_of_b_values_l305_305362

theorem sum_of_b_values (b1 b2 : ℝ) : 
  (∀ x : ℝ, (9 * x^2 + (b1 + 15) * x + 16 = 0 ∨ 9 * x^2 + (b2 + 15) * x + 16 = 0) ∧ 
           (b1 + 15)^2 - 4 * 9 * 16 = 0 ∧ 
           (b2 + 15)^2 - 4 * 9 * 16 = 0) → 
  (b1 + b2) = -30 := 
sorry

end sum_of_b_values_l305_305362


namespace correct_propositions_count_l305_305150

-- Definitions
variables (l m n : Line) (alpha beta : Plane)

def proposition1 : Prop :=
  (alpha ⟂ beta) ∧ (l ⟂ alpha) → (l ∥ beta)

def proposition2 : Prop :=
  (alpha ⟂ beta) ∧ (l ∈ alpha) → (l ⟂ beta)

def proposition3 : Prop :=
  (l ⟂ m) ∧ (m ⟂ n) → (l ∥ n)

def proposition4 : Prop :=
  (m ⟂ alpha) ∧ (n ∥ beta) ∧ (alpha ∥ beta) → (m ⟂ n)

-- The statement to prove
theorem correct_propositions_count :
  (¬ proposition1) ∧ (¬ proposition2) ∧ (¬ proposition3) ∧ proposition4 → (∃! p, p = proposition4) := 
by {
  sorry
}

end correct_propositions_count_l305_305150


namespace solve_ff2_l305_305407

noncomputable def f : ℝ → ℝ 
| x => if x ≤ 3 then 2^x else x - 1

theorem solve_ff2 : f(f(2)) = 3 := by 
  sorry

end solve_ff2_l305_305407


namespace h_has_only_one_zero_C2_below_C1_l305_305794

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 - 1/x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem h_has_only_one_zero (x : ℝ) (hx : x > 0) : 
  ∃! (x0 : ℝ), x0 > 0 ∧ h x0 = 0 := sorry

theorem C2_below_C1 (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) : 
  g x < f x := sorry

end h_has_only_one_zero_C2_below_C1_l305_305794


namespace no_valid_arrangement_exists_l305_305353

theorem no_valid_arrangement_exists :
  let nums := [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789] in
  (∃ (arrangement : List ℕ), arrangement.perm nums ∧ 
   ∀ (i : ℕ) (h : i < arrangement.length - 1), Nat.gcd (arrangement.get ⟨i, h⟩) (arrangement.get ⟨i + 1, sorry⟩) = 1) → 
  False := 
by
  sorry

end no_valid_arrangement_exists_l305_305353


namespace correct_propositions_l305_305122

/-- A quadrilateral with two pairs of equal opposite sides is a parallelogram. -/
def proposition1 (quad : Type) [quad_is_quadrilateral : is_quadrilateral quad] : Prop :=
  is_parallelogram quad

/-- A quadrilateral with four equal sides is a rhombus. -/
def proposition2 (quad : Type) [quad_is_quadrilateral : is_quadrilateral quad] : Prop :=
  is_rhombus quad

/-- Two lines parallel to the same line are parallel to each other. -/
def proposition3 (l₁ l₂ l₃ : Type) [line l₁] [line l₂] [line l₃]
  (h₁ : is_parallel l₁ l₃) (h₂ : is_parallel l₂ l₃) : Prop :=
  is_parallel l₁ l₂

/-- Two triangles are congruent if they have two sides and the included angle equal. -/
def proposition4 (Δ₁ Δ₂ : Type) [triangle Δ₁] [triangle Δ₂]
  (h₁ : side Δ₁ = side Δ₂) (h₂ : included_angle Δ₁ = included_angle Δ₂) : Prop :=
  are_congruent Δ₁ Δ₂

theorem correct_propositions :
  (proposition3 ∧ proposition4) :=
by
  sorry

end correct_propositions_l305_305122


namespace normal_price_l305_305657

variable (P : ℝ)

def final_price (P : ℝ) : ℝ :=
  (((P * 0.85) * 0.75) * 0.80) * 0.82

theorem normal_price (h : final_price P = 36) : P ≈ 81.30 :=
by
  -- Proof omitted
  sorry

end normal_price_l305_305657


namespace average_minutes_proof_l305_305312

-- Definition of conditions
def average_minutes_sixth := 8
def average_minutes_seventh := 12
def average_minutes_eighth := 16

def ratio_sixth_seventh := 1 / 2
def ratio_seventh_eighth := 1 / 2

-- Define the Lean theorem to prove the average number of minutes run per day
theorem average_minutes_proof (e : ℕ) : 
  let num_sixth := e / 4
    let num_seventh := e / 2
    let num_eighth := e
    let total_students := num_sixth + num_seventh + num_eighth
    let total_minutes := (average_minutes_sixth * num_sixth) + 
                        (average_minutes_seventh * num_seventh) + 
                        (average_minutes_eighth * num_eighth)
  in total_minutes / total_students = 96 / 7 :=
by 
  sorry

end average_minutes_proof_l305_305312


namespace inclination_angle_vertical_line_l305_305205

-- Definition of the line equation
def line_eq (x : ℝ) : Prop := 2 * x + 1 = 0

-- The proof statement
theorem inclination_angle_vertical_line :
  ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 ∧ ∃ x, line_eq x ∧ θ = real.arctan (1 / 0) :=
by
  sorry

end inclination_angle_vertical_line_l305_305205


namespace circle_eq_find_k_l305_305688

-- Define the conditions
variable (A : ℝ × ℝ) (B : ℝ × ℝ) (m : ℝ → ℝ → Prop)
variable (D : ℝ × ℝ) (k : ℝ)
variable (C_eqn : ℝ → ℝ → Prop)

-- Hypotheses based on problem conditions
hypothesis (hA : A = (1, 3))
hypothesis (hB : B = (2, 2))
hypothesis (h_m : ∀ x y, m x y ↔ 3 * x - 2 * y = 0)
hypothesis (hD : D = (0, 1))
hypothesis (hC : ∀ x y, C_eqn x y ↔ (x - 2)^2 + (y - 3)^2 = 1)

-- Statement to be proven:
-- Part 1: proving the equation of the circle
theorem circle_eq :
  ∃ center : ℝ × ℝ, ∃ radius : ℝ, 
  (∀ x y, C_eqn x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by
  sorry

-- Part 2: proving the value of k
theorem find_k (M N : ℝ × ℝ) (line_l : ℝ → ℝ → Prop) (hMN: (M ≠ N) ∧
  (line_l D.1 D.2) ∧
  (∀ x y, line_l x y ↔ y = k * x + 1) ∧
  (∀ x y, C_eqn x y ↔ (x - 2)^2 + (y - 3)^2 = 1) ∧
  ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 12)) :
  k = 1 :=
by
  sorry

end circle_eq_find_k_l305_305688


namespace min_value_arith_geom_seq_l305_305310

theorem min_value_arith_geom_seq (x y a1 a2 b1 b2 : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h_arith : x + y = 2 * a1) (h_geom : x * y = b1 * b2) :
  (x = y) → (a_1 + a_2) / (real.sqrt (b_1 * b_2)) = 2 :=
begin
  sorry
end

end min_value_arith_geom_seq_l305_305310


namespace hypotenuse_length_l305_305486

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305486


namespace remainder_of_power_series_l305_305245

theorem remainder_of_power_series (n : ℕ) (h : n = 4020) : (∑ i in finset.range n, 5 ^ (i + 1)) % 10 = 0 :=
by {
  sorry -- proof goes here
}

end remainder_of_power_series_l305_305245


namespace irene_investment_change_l305_305461

theorem irene_investment_change :
  (initial_investment first_year_loss_percent second_year_gain_percent : ℝ)
  (first_year_loss : first_year_loss_percent = 20)
  (second_year_gain : second_year_gain_percent = 10)
  (initial_investment_value : initial_investment = 150) :
  let remaining_after_first_year : ℝ := initial_investment * (1 - first_year_loss_percent / 100)
  let final_amount : ℝ := remaining_after_first_year * (1 + second_year_gain_percent / 100)
  let overall_change_percent : ℝ := (final_amount - initial_investment) / initial_investment * 100
  in overall_change_percent = -12 :=
by
  sorry

end irene_investment_change_l305_305461


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305507

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305507


namespace sum_gcf_lcm_36_56_84_l305_305660

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_gcf_lcm_36_56_84 :
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  gcf_36_56_84 + lcm_36_56_84 = 516 :=
by
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  show gcf_36_56_84 + lcm_36_56_84 = 516
  sorry

end sum_gcf_lcm_36_56_84_l305_305660


namespace value_of_x_squared_plus_y_squared_l305_305852

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l305_305852


namespace find_x_l305_305039

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l305_305039


namespace cosine_AB_AC_findValue_k_l305_305420

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point := ⟨-1, 2, 1⟩
def B : Point := ⟨0, 1, -2⟩
def C : Point := ⟨-3, 0, 2⟩

-- Vector subtraction
def vectorSub (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Dot product of vectors
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Magnitude (norm) of a vector
def magnitude (v : Point) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

-- Cosine of the angle between two vectors
def cosineAngle (v1 v2 : Point) : ℝ :=
  dotProduct v1 v2 / (magnitude v1 * magnitude v2)

-- Vectors AB and AC
def AB := vectorSub A B
def AC := vectorSub A C

-- Problem 1: Prove cosine of the angle between vectors AB and AC
theorem cosine_AB_AC : cosineAngle AB AC = - (Real.sqrt 11) / 11 := sorry

-- Problem 2: Prove the value of k given the perpendicularity condition
def perpendicular (v1 v2 : Point) : Prop := dotProduct v1 v2 = 0

def k : ℝ := 2
def testVec1 := vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub A B) A) B) A) B) A) B) A) B) C
def testVec2 := vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub (vectorSub A B) A) B) A) B) A) B) A) B) A) B) A) B) A) B) A) B) A) B) A) C

theorem findValue_k : k = 2 := sorry

end cosine_AB_AC_findValue_k_l305_305420


namespace juice_amount_P_l305_305273

variables (P_A P_Y V_A V_Y : ℝ)

-- Conditions
def condition_1 : Prop := P_A + P_Y = 24
def condition_2 : Prop := V_A + V_Y = 25
def condition_3 : Prop := V_A = P_A / 4
def condition_4 : Prop := P_Y = V_Y / 5

-- Proof statement
theorem juice_amount_P (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) : P_A = 20 :=
by
  sorry

end juice_amount_P_l305_305273


namespace problem_statement_l305_305943

theorem problem_statement (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (hprod : m * n = 5000) 
  (h_m_not_div_10 : ¬ ∃ k, m = 10 * k) (h_n_not_div_10 : ¬ ∃ k, n = 10 * k) :
  m + n = 633 :=
sorry

end problem_statement_l305_305943


namespace inequality_proof_l305_305155

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : (a / (1 + b + c) + b / (1 + c + a) + c / (1 + a + b)) ≥ (ab / (1 + a + b) + bc / (1 + b + c) + ca / (1 + c + a))) :
  (a^2 + b^2 + c^2) / (ab + bc + ca) + a + b + c + 2 ≥ 2 * (sqrt (a * b) + sqrt (b * c) + sqrt (c * a)) :=
by
  sorry

end inequality_proof_l305_305155


namespace find_x_squared_plus_y_squared_l305_305856

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l305_305856


namespace division_simplification_l305_305746

theorem division_simplification :
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18 / 7 :=
by
  sorry

end division_simplification_l305_305746


namespace monty_hall_solution_l305_305114

-- Define the probability of winning with the initial choice
def winning_initial_prob : ℚ := 1 / 3

-- Define the probability that the host opens door 3
def host_opens_door_3_prob : ℚ := 1 / 2

-- Define the probabilities for various events
def event_prob (event : ℕ → Prop) (prob : ℚ) : Prop := sorry

-- Define the Monty Hall problem conditions and relevant probabilities
def monty_hall_conditions : Prop :=
  let doors := {1, 2, 3}
  ∃ car_door initial_choice host_opens, 
  doors = {1, 2, 3} ∧ 
  car_door ∈ doors ∧ 
  initial_choice = 1 ∧ 
  host_opens ∈ {2, 3} ∧ 
  host_opens ≠ car_door ∧
  event_prob (λ cd => cd = car_door) (1/3) ∧
  event_prob (λ ho => ho = host_opens) (1/2) ∧
  ∀ cd, event_prob (λ st => st = cd) (if cd = 1 then 1/3 else 2/3)

-- The main theorem combining all proof statements
theorem monty_hall_solution : monty_hall_conditions ∧ 
  winning_initial_prob = 1 / 3 ∧
  host_opens_door_3_prob = 1 / 2 ∧
  ∀ cd, (event_prob (λ st => st = 2) 2/3) → (event_prob (λ st => st = 1) 1/3) → 
      (2 / 3 > 1 / 3) :=
sorry

end monty_hall_solution_l305_305114


namespace hypotenuse_length_l305_305500

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305500


namespace minimum_b1_b2_value_l305_305224

theorem minimum_b1_b2_value (b : ℕ → ℕ) :
  (∀ n ≥ 1, b(n + 2) = (b n + 2017) / (1 + b(n + 1))) → 
  (∀ n, b n > 0) →
  b 1 + b 2 = 2018 :=
by
  sorry

end minimum_b1_b2_value_l305_305224


namespace area_of_fold_points_l305_305377

theorem area_of_fold_points (P : Point) (A B C : Point) (abc_triangle : is_triangle A B C)
  (h1 : distance A B = 45)
  (h2 : distance A C = 90)
  (h3 : angle B = 90) :
  area_of_fold_points A B C P = 506.25 * π - 607.5 * Real.sqrt 3 :=
sorry

end area_of_fold_points_l305_305377


namespace min_green_beads_l305_305720

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l305_305720


namespace distance_center_to_line_l305_305066

-- Rewrite the conditions as definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + 4 = 0
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 4 = 0

-- Prove the target statement about the distance
theorem distance_center_to_line : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (p : ℝ × ℝ), (p = ⟨1, 2⟩ → 
  ∃ d : ℝ, d = 3 ∧ d = |3 * p.fst + 4 * p.snd + 4| / real.sqrt (3^2 + 4^2)) :=
begin
  sorry
end

end distance_center_to_line_l305_305066


namespace no_valid_pairs_l305_305087

theorem no_valid_pairs : ∀ (m n : ℕ), m ≥ n → m^2 - n^2 = 150 → false :=
by sorry

end no_valid_pairs_l305_305087


namespace max_electronic_thermometers_l305_305867

theorem max_electronic_thermometers :
  ∀ (x : ℕ), 10 * x + 3 * (53 - x) ≤ 300 → x ≤ 20 :=
by
  sorry

end max_electronic_thermometers_l305_305867


namespace soccer_ball_cost_l305_305728

theorem soccer_ball_cost (x : ℕ) (h : 5 * x + 4 * 65 = 980) : x = 144 :=
by
  sorry

end soccer_ball_cost_l305_305728


namespace sin_minus_cos_eq_sqrt_247_div_13_l305_305391

theorem sin_minus_cos_eq_sqrt_247_div_13 (θ : ℝ) (h1 : sin θ + cos θ = 7 / 13) (h2 : 0 < θ ∧ θ < real.pi) :
  sin θ - cos θ = real.sqrt 247 / 13 :=
by
  sorry

end sin_minus_cos_eq_sqrt_247_div_13_l305_305391


namespace hypotenuse_length_l305_305477

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305477


namespace find_alpha_l305_305522

theorem find_alpha (α : ℝ) :
    7 * α + 8 * α + 45 = 180 →
    α = 9 :=
by
  sorry

end find_alpha_l305_305522


namespace find_constants_l305_305775

theorem find_constants (P Q R : ℚ) (h : ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) = (P / (x - 1)) + (Q / (x - 4)) + (R / (x - 6))) : 
  (P, Q, R) = (-4/5, -1/2, 23/10) := 
  sorry

end find_constants_l305_305775


namespace cos_alpha_value_l305_305441

-- Definitions for conditions and theorem statement

def condition_1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def condition_2 (α : ℝ) : Prop :=
  Real.cos (Real.pi / 3 + α) = 1 / 3

theorem cos_alpha_value (α : ℝ) (h1 : condition_1 α) (h2 : condition_2 α) :
  Real.cos α = (1 + 2 * Real.sqrt 6) / 6 := sorry

end cos_alpha_value_l305_305441


namespace MarysTotalCandies_l305_305570

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end MarysTotalCandies_l305_305570


namespace integral_of_function_eq_two_thirds_l305_305346

theorem integral_of_function_eq_two_thirds :
  ∫ x in -1..1, (x + x^2 + sin x) = (2 / 3) :=
by
  sorry

end integral_of_function_eq_two_thirds_l305_305346


namespace common_difference_l305_305516

variable {α : Type*} [LinearOrderedField α]

variables (a : ℕ → α) (d : α)

def arithmetic_sequence (a n : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_first_nine (a : ℕ → α) : Prop :=
  (∑ i in finset.range 9, a i) = 81

def sum_nine_to_eighteen (a : ℕ → α) : Prop :=
  (∑ i in finset.range' 1 10, a i) = 171

theorem common_difference (a : ℕ → α) (d : α) 
  (h_seq : arithmetic_sequence a d)
  (h_sum9 : sum_first_nine a)
  (h_sum10 : sum_nine_to_eighteen a) :
  d = 10 :=
by
  -- proof required
  sorry

end common_difference_l305_305516


namespace problem_l305_305444

theorem problem (a b : ℝ) (m n : ℕ) (h1 : a ^ m = 2) (h2 : b ^ n = 4) : a ^ (2 * m) * b ^ (-2 * n) = 1 / 4 :=
by sorry

end problem_l305_305444


namespace abs_value_product_l305_305882

theorem abs_value_product (x : ℝ) (h : |x - 5| - 4 = 0) : ∃ y z, (y - 5 = 4 ∨ y - 5 = -4) ∧ (z - 5 = 4 ∨ z - 5 = -4) ∧ y * z = 9 :=
by 
  sorry

end abs_value_product_l305_305882


namespace geometry_problem_l305_305158

noncomputable def scalene_triangle_ABC (A B C : Point) : Prop := 
  ¬(A = B ∨ B = C ∨ C = A) ∧ ¬collinear A B C

noncomputable def altitudes (A B C D E F : Point) : Prop := 
  is_altitude A B C D ∧ is_altitude B A C E ∧ is_altitude C A B F

noncomputable def circumcenter_O (A B C O : Point) : Prop := 
  is_circumcenter A B C O

noncomputable def circumcircles_intersect (A B C D O P : Point) : Prop := 
  ons_circle (circumcircle A B C) P ∧ ons_circle (circumcircle A D O) P ∧ P ≠ A

noncomputable def circle_intersects_line (circ : Circle) (P E X : Point) : Prop := 
  ons_circle circ X ∧ online E P X ∧ X ≠ P

noncomputable def are_parallel (X Y B C : Point) : Prop := 
  parallel (line X Y) (line B C)

theorem geometry_problem
  (A B C D E F O P X Y : Point)
  (h_scalene : scalene_triangle_ABC A B C)
  (h_altitudes : altitudes A B C D E F)
  (h_circumcenter : circumcenter_O A B C O)
  (h_circumcircles : circumcircles_intersect A B C D O P)
  (h_intersect_PE : circle_intersects_line (circumcircle A B C) P E X)
  (h_intersect_PF : circle_intersects_line (circumcircle A B C) P F Y) :
  are_parallel X Y B C :=
  sorry

end geometry_problem_l305_305158


namespace root_interval_l305_305148

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval :
  (f 1 < 0) →
  (f 1.5 > 0) →
  (f 1.25 < 0) →
  ∃ ξ : ℝ, ξ ∈ (set.Ioo 1.25 1.5) ∧ f ξ = 0 :=
by {
  sorry
}

end root_interval_l305_305148


namespace fraction_of_odd_products_is_0_25_l305_305113

noncomputable def fraction_of_odd_products : ℝ :=
  let odd_products := 8 * 8
  let total_products := 16 * 16
  (odd_products / total_products : ℝ)

theorem fraction_of_odd_products_is_0_25 :
  fraction_of_odd_products = 0.25 :=
by sorry

end fraction_of_odd_products_is_0_25_l305_305113


namespace product_of_areas_eq_k3_times_square_of_volume_l305_305740

variables (a b c k : ℝ)

-- Defining the areas of bottom, side, and front of the box as provided
def area_bottom := k * a * b
def area_side := k * b * c
def area_front := k * c * a

-- Volume of the box
def volume := a * b * c

-- The lean statement to be proved
theorem product_of_areas_eq_k3_times_square_of_volume :
  (area_bottom a b k) * (area_side b c k) * (area_front c a k) = k^3 * (volume a b c)^2 :=
by
  sorry

end product_of_areas_eq_k3_times_square_of_volume_l305_305740


namespace option_D_correct_l305_305810

-- Define the vector a
def vector_a (x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin x, Real.cos x ^ 2)

-- Define the vector b
def vector_b (x : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos x, 2)

-- Define the dot product of vectors a and b
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  dot_product (vector_a x) (vector_b x) - 1

-- State the theorem about f(x)
theorem option_D_correct :
  ∃ g : ℝ → ℝ, (∀ x, g x = f (x - π/6)) ∧ (∀ x, g x = g (-x)) :=
by
  sorry

end option_D_correct_l305_305810


namespace cases_needed_to_raise_funds_l305_305936

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end cases_needed_to_raise_funds_l305_305936


namespace find_f_2016_l305_305821

noncomputable def f : ℝ → ℝ :=
  λ x, if (x % 2 <= 2 ∧ x % 2 > 0) then (Real.log x / Real.log 4) else sorry

theorem find_f_2016 : f 2016 = 1 / 2 :=
by
  /-
  Given:
  1. ∀ x, f(x + 2) = f(x)
  2. ∀ x ∈ (0, 2], f(x) = log₄(x)
  To prove: f(2016) = 1/2
  -/
  sorry

end find_f_2016_l305_305821


namespace mary_flour_indeterminate_l305_305933

theorem mary_flour_indeterminate 
  (sugar : ℕ) (flour : ℕ) (salt : ℕ) (needed_sugar_more : ℕ) 
  (h_sugar : sugar = 11) (h_flour : flour = 6)
  (h_salt : salt = 9) (h_condition : needed_sugar_more = 2) :
  ∃ (current_flour : ℕ), current_flour ≠ current_flour :=
by
  sorry

end mary_flour_indeterminate_l305_305933


namespace hypotenuse_length_l305_305489

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305489


namespace locus_is_circle_if_a_gt_K_l305_305378

variables (s a K : ℝ)

-- Condition on the sides of the right triangle
def is_right_triangle (A B C : (ℝ × ℝ)) : Prop :=
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xC, yC) := C in
  (xC = 0 ∧ yC = 0) ∧
  (xA = s ∧ yA = 0 ∨ xA = 0 ∧ yA = s) ∧
  (xB = 0 ∧ yB = s ∨ xB = s ∧ yB = 0) ∧
  (xA - xB)^2 + (yA - yB)^2 = (s * sqrt 2)^2

-- Definition of distances squared from P to vertices A, B, and C
def sum_of_squares_of_distances (P A B C : (ℝ × ℝ)) : ℝ :=
  let (xP, yP) := P in
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xC, yC) := C in
  (xP - xA)^2 + (yP - yA)^2 +
  (xP - xB)^2 + (yP - yB)^2 +
  (xP - xC)^2 + (yP - yC)^2

-- The problem statement: Prove the locus is a circle if a > K
theorem locus_is_circle_if_a_gt_K {A B C P : (ℝ × ℝ)} (h_tri : is_right_triangle A B C)
  (h_sum : sum_of_squares_of_distances P A B C = a) : a > K → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), (0 < radius) ∧ (∀ P', (P' ≠ P) → sum_of_squares_of_distances P' A B C = a :=
begin
  sorry
end

end locus_is_circle_if_a_gt_K_l305_305378


namespace perpendicular_lines_l305_305976

theorem perpendicular_lines (a : ℝ) : 
  ∀ x y : ℝ, 3 * y - x + 4 = 0 → 4 * y + a * x + 5 = 0 → a = 12 :=
by
  sorry

end perpendicular_lines_l305_305976


namespace quadratic_equation_general_form_l305_305760

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 2 * (x + 2)^2 + (x + 3) * (x - 2) = -11 ↔ 3 * x^2 + 9 * x + 13 = 0 :=
sorry

end quadratic_equation_general_form_l305_305760


namespace construct_triangle_l305_305326

theorem construct_triangle (a b c : ℝ) (γ : ℝ) (hγ1 : 0 < γ) (hγ2 : γ < 180)
  (hbc : b + c > a) : ∃ (A B C : Type) (AB AC BC : ℝ),
  BC = a ∧ (AB + AC) = b + c ∧ (angle A B C) = γ :=
sorry

end construct_triangle_l305_305326


namespace car_return_speed_l305_305694

theorem car_return_speed
    (distance : ℝ)
    (speed_to_B : ℝ)
    (average_speed_round_trip : ℝ)
    (h1 : distance = 150)
    (h2 : speed_to_B = 75)
    (h3 : average_speed_round_trip = 50)
    : ∃ r : ℝ, (300 / (2 + distance / r)) = average_speed_round_trip ∧ r = 37.5 :=
by
  use 37.5
  split
  sorry -- Proof needed here
  sorry -- Proof needed here

end car_return_speed_l305_305694


namespace smallest_positive_period_l305_305355

noncomputable def function_period (x : ℝ) : ℝ := 5 * tan ((2 / 5) * x + π / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, function_period (x + T) = function_period x ∧
  (∀ T' > 0, (∀ x : ℝ, function_period (x + T') = function_period x) → T ≤ T') :=
begin
  use (5 * π / 2),
  sorry
end

end smallest_positive_period_l305_305355


namespace shared_boundary_segment_l305_305017

theorem shared_boundary_segment 
    (grid_size : ℕ := 55) 
    (num_triangles : ℕ := 400) 
    (num_cells : ℕ := 500)
    (total_figures : ℕ := 900)
    : ∃ a b : ℕ, a ≠ b ∧ (shares_boundary_segment a b) := 
sorry

end shared_boundary_segment_l305_305017


namespace find_principal_amount_l305_305715

noncomputable def principal_amount_loan (SI R T : ℝ) : ℝ :=
  SI / (R * T)

theorem find_principal_amount (SI R T : ℝ) (h_SI : SI = 6480) (h_R : R = 0.12) (h_T : T = 3) :
  principal_amount_loan SI R T = 18000 :=
by
  rw [principal_amount_loan, h_SI, h_R, h_T]
  norm_num

#check find_principal_amount

end find_principal_amount_l305_305715


namespace frannie_jumps_l305_305369

theorem frannie_jumps (Meg_jumps : ℕ) (Frannie_jumps : ℕ) (h1 : Meg_jumps = 71) (h2 : Frannie_jumps = Meg_jumps - 18) : Frannie_jumps = 53 :=
by
  rw [h1, h2]
  norm_num

end frannie_jumps_l305_305369


namespace average_first_200_terms_l305_305325

def sequence_term (n : ℕ) : ℤ :=
  (-1)^(n + 1) * n

theorem average_first_200_terms : (∑ i in Finset.range 200, sequence_term (i + 1) : ℤ) / 200 = -1 / 2 :=
  sorry

end average_first_200_terms_l305_305325


namespace unique_n_sum_series_l305_305751

variable (n : ℕ)

def sum_series (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (i + 3) * 3^(i + 2)

theorem unique_n_sum_series :
  (sum_series n = 3^(n + 9)) → (n = 513) :=
by
  intro h
  sorry

end unique_n_sum_series_l305_305751


namespace real_root_sqrt_eq_l305_305780

theorem real_root_sqrt_eq (x : ℝ) (hx : sqrt x + sqrt (x + 2) = 10) : x = 2401 / 100 :=
sorry

end real_root_sqrt_eq_l305_305780


namespace obtuse_right_triangle_cannot_exist_l305_305667

-- Definitions of various types of triangles

def is_acute (θ : ℕ) : Prop := θ < 90
def is_right (θ : ℕ) : Prop := θ = 90
def is_obtuse (θ : ℕ) : Prop := θ > 90

def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c
def is_scalene (a b c : ℕ) : Prop := ¬ (a = b) ∧ ¬ (b = c) ∧ ¬ (a = c)
def is_triangle (a b c : ℕ) : Prop := a + b + c = 180

-- Propositions for the types of triangles given in the problem

def acute_isosceles_triangle (a b : ℕ) : Prop :=
  is_triangle a a (180 - 2 * a) ∧ is_acute a ∧ is_isosceles a a (180 - 2 * a)

def isosceles_right_triangle (a : ℕ) : Prop :=
  is_triangle a a 90 ∧ is_right 90 ∧ is_isosceles a a 90

def obtuse_right_triangle (a b : ℕ) : Prop :=
  is_triangle a 90 (180 - 90 - a) ∧ is_right 90 ∧ is_obtuse (180 - 90 - a)

def scalene_right_triangle (a b : ℕ) : Prop :=
  is_triangle a b 90 ∧ is_right 90 ∧ is_scalene a b 90

def scalene_obtuse_triangle (a b : ℕ) : Prop :=
  is_triangle a b (180 - a - b) ∧ is_obtuse (180 - a - b) ∧ is_scalene a b (180 - a - b)

-- The final theorem stating that obtuse right triangle cannot exist

theorem obtuse_right_triangle_cannot_exist (a b : ℕ) :
  ¬ exists (a b : ℕ), obtuse_right_triangle a b :=
by
  sorry

end obtuse_right_triangle_cannot_exist_l305_305667


namespace box_side_length_l305_305252

theorem box_side_length :
  ∃ (s : ℝ), 
    s ≈ 16.9 ∧
    (∃ (n : ℕ), 
      0.5 * n = 250 ∧ 
      2.4e6 / n = s ^ 3) := 
sorry

end box_side_length_l305_305252


namespace area_of_black_region_l305_305714

theorem area_of_black_region :
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  area_large - total_area_small = 94 :=
by
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  sorry

end area_of_black_region_l305_305714


namespace negate_universal_prop_l305_305215

theorem negate_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0 :=
sorry

end negate_universal_prop_l305_305215


namespace Matrix_inverse_zero_or_solve_l305_305762

open Matrix

-- Define the matrix A according to the given problem's conditions.
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 10], 
    ![-15, -30]]

-- Prove that if the determinant of matrix A is zero, it does not have an inverse, and thus the zero matrix is the solution.
theorem Matrix_inverse_zero_or_solve (A : Matrix (Fin 2) (Fin 2) ℤ) : 
  det A = 0 → ¬invertible A ∧ A = 0 := by
  sorry

end Matrix_inverse_zero_or_solve_l305_305762


namespace foci_distance_ellipse_l305_305647

-- Definitions for the points on the axes of the ellipse
def pt1 : ℝ × ℝ := (2, -4)
def pt2 : ℝ × ℝ := (-1, 5)
def pt3 : ℝ × ℝ := (7, 5)

-- Definition of the midpoint (center of the ellipse)
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Definitions for the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Definitions for semi-major axis and semi-minor axis
def semi_major_axis (p1 p2 : ℝ × ℝ) : ℝ :=
  distance p1 p2 / 2

def semi_minor_axis (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ :=
  distance p c

-- Definition for the distance between the foci
def foci_distance (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

-- Main theorem statement
theorem foci_distance_ellipse :
  foci_distance (semi_major_axis pt2 pt3) (semi_minor_axis pt1 (midpoint pt2 pt3)) = 2 * Real.sqrt (16 - Real.sqrt 82) :=
by
  sorry

end foci_distance_ellipse_l305_305647


namespace square_partition_l305_305329

theorem square_partition (a b c n : ℕ) (ha : a = 4) (hb : b = 3) (hc : c = 15) (hn : n = 9) : 
  c^2 = n * (a^2 + b^2) :=
by
  rw [ha, hb, hc, hn]
  simp
  sorry

end square_partition_l305_305329


namespace total_profit_is_1000_l305_305678

-- Define the initial investments
def investment_a := 800
def investment_b := 1000
def investment_c := 1200

-- Define C's share of the profit
def c_share := 400

-- Define the function to calculate the total profit
noncomputable def total_profit : ℕ :=
  let ratio_sum := 4 + 5 + 6 in
  let one_part_value := c_share / 6 in
  ratio_sum * one_part_value

-- The main proof statement asserting the total profit is Rs. 1000
theorem total_profit_is_1000 : total_profit = 1000 := by
  sorry

end total_profit_is_1000_l305_305678


namespace sum_of_powers_of_i_l305_305792

theorem sum_of_powers_of_i : 
  ∑ k in (Finset.range 2017).map (λ n => n + 1), i^k = i := 
sorry

end sum_of_powers_of_i_l305_305792


namespace series_sum_l305_305745

theorem series_sum :
  let T := ∑ n in Finset.range 50, (3 + 7 * (n + 1)) / 3 ^ (50 - n) 
in T = 171.5 :=
by
  sorry

end series_sum_l305_305745


namespace problem_statement_l305_305084

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fib(n+1) + fib(n)

def num_12digit_with_two_consecutive_1s : ℕ :=
  let total := 2 ^ 12
  let without_consecutive_1s := fib 13
  total - without_consecutive_1s

theorem problem_statement : num_12digit_with_two_consecutive_1s = 3719 := by
  sorry

end problem_statement_l305_305084


namespace Jerome_money_left_l305_305838

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l305_305838


namespace car_time_passed_l305_305275

variable (speed : ℝ) (distance : ℝ) (time_passed : ℝ)

theorem car_time_passed (h_speed : speed = 2) (h_distance : distance = 2) :
  time_passed = distance / speed := by
  rw [h_speed, h_distance]
  norm_num
  sorry

end car_time_passed_l305_305275


namespace trader_loss_percentage_l305_305730

def percentage_loss (CP MP SP : ℝ) : ℝ := ((CP - SP) / CP) * 100

theorem trader_loss_percentage (CP MP D SP : ℝ) (hCP : CP = 100) (hMP : MP = 1.4 * CP) (hD : D = 0.07857142857142857) (hSP : SP = MP * (1 - D)) :
  percentage_loss CP MP SP = 29 := by
  sorry

end trader_loss_percentage_l305_305730


namespace hyperbola_eccentricity_l305_305031

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : b / a = Real.sqrt 3 ∨ a / b = Real.sqrt 3) : 
  let c := if h₁.left then a * 2 else a * 2 / Real.sqrt 3 in
  let e := c / a in 
  e = 2 ∨ e = 2 * Real.sqrt 3 / 3 := 
sorry

end hyperbola_eccentricity_l305_305031


namespace joshua_more_than_mitch_l305_305934

variable (M : ℝ) -- miles' macarons count
variable (mitch_macarons : ℝ := 20)
variable (joshua_macarons : ℝ := M / 2)
variable (renz_macarons : ℝ := (3 / 4) * M - 1)
variable (total_kids : ℝ := 68)
variable (macarons_per_kid : ℝ := 2)
variable (total_macarons : ℝ := total_kids * macarons_per_kid)

theorem joshua_more_than_mitch :
  (mitch_macarons + joshua_macarons + renz_macarons = 136) →
  ((total_macarons = 136) →
  (joshua_macarons - mitch_macarons = 27)) :=
by {
  intro h1 h2,
  sorry
}

end joshua_more_than_mitch_l305_305934


namespace puppy_food_consumption_l305_305925

/-- Mathematically equivalent proof problem:
  Given the following conditions:
  1. days_per_week = 7
  2. initial_feeding_duration_weeks = 2
  3. initial_feeding_daily_portion = 1/4
  4. initial_feeding_frequency_per_day = 3
  5. subsequent_feeding_duration_weeks = 2
  6. subsequent_feeding_daily_portion = 1/2
  7. subsequent_feeding_frequency_per_day = 2
  8. today_feeding_portion = 1/2
  Prove that the total food consumption, including today, over the next 4 weeks is 25 cups.
-/
theorem puppy_food_consumption :
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  total = 25 := by
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  show total = 25 from sorry

end puppy_food_consumption_l305_305925


namespace find_x_l305_305848

theorem find_x (x : ℝ) (h : 2^8 = 32^x) : x = 8 / 5 :=
by
  sorry

end find_x_l305_305848


namespace vincent_total_packs_l305_305676

-- Definitions based on the conditions
def packs_yesterday : ℕ := 15
def extra_packs_today : ℕ := 10

-- Total packs calculation
def packs_today : ℕ := packs_yesterday + extra_packs_today
def total_packs : ℕ := packs_yesterday + packs_today

-- Proof statement
theorem vincent_total_packs : total_packs = 40 :=
by
  -- Calculate today’s packs
  have h1 : packs_today = 25 := by
    rw [packs_yesterday, extra_packs_today]
    norm_num
  
  -- Calculate the total packs
  have h2 : total_packs = 15 + 25 := by
    rw [packs_yesterday, h1]
  
  -- Conclude the total number of packs
  show total_packs = 40
  rw [h2]
  norm_num

end vincent_total_packs_l305_305676


namespace find_x_l305_305038

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l305_305038


namespace negation_of_p_l305_305415

variable (a b : ℤ)

/-- Proposition p: a and b are both even numbers -/
def p : Prop := (a % 2 = 0) ∧ (b % 2 = 0)

/-- Proposition ¬p: it is not true that a and b are both even numbers -/
def not_p : Prop := ¬p

theorem negation_of_p : not_p a b ↔ (¬ (a % 2 = 0 ∧ b % 2 = 0)) := by sorry

end negation_of_p_l305_305415


namespace hypotenuse_length_l305_305484

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305484


namespace company_p_percentage_increase_l305_305747

theorem company_p_percentage_increase :
  (460 - 400.00000000000006) / 400.00000000000006 * 100 = 15 := 
by
  sorry

end company_p_percentage_increase_l305_305747


namespace simplify_and_evaluate_expression_l305_305959

theorem simplify_and_evaluate_expression :
  (let x := 3 in ((x + 2) / (x - 2) + (x - x^2) / (x^2 - 4 * x + 4)) / ((x - 4) / (x - 2))) = 1 := 
by
  -- Declare the simplifications and substitutions made in the solution
  let x := 3
  have h1 : (x^2 - 4*x + 4) = (x - 2)^2 := by sorry
  have h2 : (x + 2) / (x - 2) + (x - x^2) / ((x - 2)^2) = (x - 4) / ((x - 2) ^ 2) := by sorry
  have h3 : ((x - 4) / ((x - 2) ^ 2)) / ((x - 4) / (x - 2)) = 1 / (x - 2) := by sorry
  have h4 : 1 / (x - 2) = 1 := by sorry
  -- Combine the simplifications and the given x value
  show (1 : ℝ) = 1 from sorry

end simplify_and_evaluate_expression_l305_305959


namespace randy_blocks_l305_305592

theorem randy_blocks (H : ℕ) (T : ℕ) (total : ℕ) :
  total = 95 →
  T = 50 →
  T = H + 30 →
  H = 20 :=
by
  intros h1 h2 h3
  have h : T = 50 := h2
  have e1 : T = H + 30 := h3
  rw [←e1] at h2
  have : 50 = H + 30 := h2
  linarith

end randy_blocks_l305_305592


namespace system_has_solution_l305_305208

theorem system_has_solution (a b : ℝ) (h : (∃ x1 x2 : ℝ, x1 < x2 ∧ (sin x1 + a = b * x1) ∧ (sin x2 + a = b * x2))) :
  ∃ x : ℝ, sin x + a = b * x ∧ cos x = b :=
by
  sorry

end system_has_solution_l305_305208


namespace problem_statement_l305_305764

theorem problem_statement :
  ∃ (k : ℕ) (b : Fin k → ℕ) (H_b : StrictMono b),
    (∑ i in Finset.range k, 2 ^ (b i)) = (2^201 + 1) / (2^12 + 1) ∧
    k = 192 :=
by
  sorry

end problem_statement_l305_305764


namespace tan_square_of_cos_double_angle_l305_305052

theorem tan_square_of_cos_double_angle (α : ℝ) (h : Real.cos (2 * α) = -1/9) : Real.tan (α)^2 = 5/4 :=
by
  sorry

end tan_square_of_cos_double_angle_l305_305052


namespace min_weeks_to_complete_graph_l305_305236

open Finset

-- Define the number of friends
def numFriends : ℕ := 12

-- Define the number of tables
def numTables : ℕ := 3

-- Define the number of people per table
def numPeoplePerTable : ℕ := 4

-- Define the total number of possible pairs in a complete graph K_{12}
def totalPairs {n : ℕ} (h : 2 ≤ n) : ℕ := (n * (n - 1)) / 2

-- State the problem
theorem min_weeks_to_complete_graph :
  let weeks : ℕ := 5 in
  ∀ (friends : Finset (Fin numFriends)), friends.card = numFriends → 
  ∀ (pairings : ℕ → Finset (Finset (Fin numFriends × Fin numFriends))) (week : ℕ),
  (∀ i < week, pairings i ⊆ (Finset.pairwise_on_table numTables numPeoplePerTable i)) →
  (∀ i < week, ∀ (p ∈ pairings i), p.1 ∈ friends ∧ p.2 ∈ friends) →
  -- We have accumulated enough pairs over 'weeks' to cover all possible pairs:
  (∃ week, totalPairs (nat.succ (numFriends - 1)) (nat.two_le_of_odd numFriends) ≤ 
  ∑ i in range week, (pairings i).card) :=
by sorry

end min_weeks_to_complete_graph_l305_305236


namespace odd_function_f1_l305_305100

theorem odd_function_f1 (a : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = a - 2 / (2^x + 1))
  (h2 : ∀ x, f(-x) = -f(x)) :
  f(1) = 1 / 3 :=
by
  sorry

end odd_function_f1_l305_305100


namespace number_of_ways_sum_2000_l305_305514

theorem number_of_ways_sum_2000 :
  (∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2000 ∧ a ≤ b ∧ b ≤ c) ↔ finset.card ((finset.range 2001).filter (λ n, ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2000 ∧ a ≤ b ∧ b ≤ c)) = 333963 := 
sorry

end number_of_ways_sum_2000_l305_305514


namespace tens_digit_4032_pow_4033_minus_4036_l305_305347

theorem tens_digit_4032_pow_4033_minus_4036 :
  let a := 4032
  let b := 4033
  let c := 4036
  (a^b - c) ≡ 96 [MOD 100] → (a^b - c) // 10 % 10 = 9 :=
by
  sorry

end tens_digit_4032_pow_4033_minus_4036_l305_305347


namespace find_x_l305_305032

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : nat.factors x = [31, p1, p2]) : x = 32767 :=
by
  sorry

end find_x_l305_305032


namespace find_x_perpendicular_l305_305421

section VectorPerpendicular

variable (x : ℝ)
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (x, -3)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_perpendicular (h : dot_product a b = 0) : x = 1 :=
by {
  -- sorry to skip the proof
  sorry
}

end VectorPerpendicular

end find_x_perpendicular_l305_305421


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305505

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305505


namespace min_points_square_l305_305469

-- Define the grid of points
def grid_points : set (ℕ × ℕ) :=
  { p | ∃ i j, 0 ≤ i ∧ i ≤ 3 ∧ 0 ≤ j ∧ j ≤ 3 ∧ p = (i, j) }

-- Define a function to check if four points form a square
def forms_square (p1 p2 p3 p4 : ℕ × ℕ) : Prop :=
  -- Check if these points form a square by their coordinate properties
  (dist p1 p2 = dist p3 p4 ∧ dist p1 p3 = dist p2 p4 ∧ dist p1 p4 = dist p2 p3) 

-- The main theorem to prove
theorem min_points_square (n : ℕ) (h : n ≥ 11)
  (points : finset (ℕ × ℕ)) (hp : points ⊆ grid_points) (hcount : finset.card points = n) :
  ∃ p1 p2 p3 p4 ∈ points, forms_square p1 p2 p3 p4 :=
sorry -- proof to be provided

end min_points_square_l305_305469


namespace exists_four_digit_triangular_difference_l305_305721

theorem exists_four_digit_triangular_difference (A B : ℕ) :
  (∃ n k : ℕ, n ∈ (1000/2 : ℕ) ≤ A ∧ k ∈ (1000/2 : ℕ) ≤ B ∧ 
            A = n * (n + 1)/2 ∧ B = k * (k + 1)/2 ∧ 2015 = A - B) :=
sorry

end exists_four_digit_triangular_difference_l305_305721


namespace equal_water_and_alcohol_l305_305695

variable (a m : ℝ)

-- Conditions:
-- Cup B initially contains m liters of water.
-- Transfers as specified in the problem.

theorem equal_water_and_alcohol (h : m > 0) :
  (a * (m / (m + a)) = a * (m / (m + a))) :=
by
  sorry

end equal_water_and_alcohol_l305_305695


namespace bounded_figure_has_at_most_one_center_of_symmetry_l305_305268

theorem bounded_figure_has_at_most_one_center_of_symmetry (F : Set Point) [Bounded F] :
  ∀ O1 O2 : Point, is_center_of_symmetry F O1 → is_center_of_symmetry F O2 → O1 = O2 := 
sorry

end bounded_figure_has_at_most_one_center_of_symmetry_l305_305268


namespace U_value_l305_305553

def f : ℕ → ℝ := sorry
axiom f_rec (n : ℕ) (h : n ≥ 6) : f n = (n - 1) * f (n - 1)
axiom f_nonzero (n : ℕ) (h : n ≥ 6) : f n ≠ 0

def U (T : ℕ) : ℝ := f T / ((T - 1) * f (T - 3))

theorem U_value (T : ℕ) (hT : T ≥ 6) : U T = 72 := sorry

end U_value_l305_305553


namespace pentagon_rectangle_ratio_l305_305289

theorem pentagon_rectangle_ratio :
  ∀ (p w l : ℝ), 
  5 * p = 20 → 
  2 * (w + l) = 20 →
  l = 2 * w →
  p / w = 6 / 5 :=
by
  intros p w l h₁ h₂ h₃
  have p_value : p = 4 := 
    by linarith
  have w_value : w = 10 / 3 := 
    by linarith
  rw [p_value, w_value]
  norm_num
  sorry

end pentagon_rectangle_ratio_l305_305289


namespace perpendicular_tangent_lines_bounded_ln_ineq_l305_305918

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (m n x : ℝ) := m * (x + n) / (x + 1)

theorem perpendicular_tangent_lines (h : 1 > (0 : ℝ)) : 
  ∃ n : ℝ, 
  (let m := (1 : ℝ) in (fderiv ℝ (fun y => f y) (1 : ℝ)) 1 * 
    (fderiv ℝ (fun y => g m n y) (1 : ℝ)) 1 = -1) → n = 5 := sorry


theorem bounded_ln_ineq :
  ∃ (n m : ℝ),
  (∀ x > 0, |f x| ≥ |g m n x|) → n = -1 ∧ m ≤ 2 := sorry

end perpendicular_tangent_lines_bounded_ln_ineq_l305_305918


namespace largest_k_l305_305784

def sigma (n : ℕ) : ℕ := sorry -- Sum of the positive divisors of n
def nu_p (p n : ℕ) : ℕ := sorry -- Largest power of p that divides n

def N := 6 ^ 1999

theorem largest_k : 
  let S := ∑ d in (finset.filter (λ x, x ∣ N) (finset.range (N+1))), nu_p 3 (d!) * (-1) ^ sigma d in
  ∃ k, 5 ^ k ∣ S ∧ ∀ l, 5 ^ (k + l) ∣ S → l = 0 :=
sorry

end largest_k_l305_305784


namespace find_symmetry_axis_l305_305201

noncomputable def symmetry_axis_function : ℝ → ℝ :=
  λ x, Real.sin (2 * x + 5 * Real.pi / 2)

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem find_symmetry_axis :
  is_symmetry_axis symmetry_axis_function (- Real.pi / 2) :=
by
  sorry

end find_symmetry_axis_l305_305201


namespace cole_drive_time_l305_305321

theorem cole_drive_time (D T1 T2 : ℝ) (h1 : T1 = D / 75) 
  (h2 : T2 = D / 105) (h3 : T1 + T2 = 6) : 
  (T1 * 60 = 210) :=
by sorry

end cole_drive_time_l305_305321


namespace correct_conclusions_l305_305802

-- Given function f with the specified domain and properties
variable {f : ℝ → ℝ}

-- Given conditions
axiom functional_eq (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y
axiom f_one_half : f (1/2) = 0
axiom f_zero_not_zero : f 0 ≠ 0

-- Proving our conclusions
theorem correct_conclusions :
  f 0 = 1 ∧ (∀ y : ℝ, f (1/2 + y) = -f (1/2 - y))
:=
by
  sorry

end correct_conclusions_l305_305802


namespace positive_difference_two_numbers_l305_305630

theorem positive_difference_two_numbers (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 2 * y - 3 * x = 5) : abs (y - x) = 8 := 
sorry

end positive_difference_two_numbers_l305_305630


namespace correct_propositions_l305_305944

variable {α : Type*} [field α] [add_comm_group α] [module α α]

-- Define lines and planes
def is_parallel {V : Type*} [module α V] (u v : V) : Prop := ∃ k : α, u = k • v
def is_perpendicular {V : Type*} [inner_product_space α V] (u v : V) : Prop := ⟪u, v⟫ = 0

-- Propositions
def proposition_1 (a b c : α) : Prop := is_parallel a b → is_parallel b c → is_parallel a c
def proposition_2 (a b c : α) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0):
  Prop := is_parallel a b → is_parallel b c → is_parallel a c
def proposition_3 (α β γ : α) : Prop := is_perpendicular α β → is_perpendicular β γ → is_parallel α γ
def proposition_4 (a b c : α) : Prop := is_perpendicular a b → is_perpendicular b c → is_parallel a c
def proposition_5 (a b : α) (β : α) : Prop := is_perpendicular a β → is_perpendicular b β → is_parallel a b

theorem correct_propositions :
  (proposition_1 (0 : α) 0 0) ∧
  (proposition_2 (0 : α) 0 0 zero_ne_one zero_ne_one zero_ne_one) ∧
  (proposition_5 (0 : α) 0 0) :=
by {
  sorry
}

end correct_propositions_l305_305944


namespace trig_identity_l305_305684

theorem trig_identity (α : ℝ) :
    (cos (2 * α - π / 2) + sin (3 * π - 4 * α) - cos (5 / 2 * π + 6 * α)) / (4 * sin (5 * π - 3 * α) * cos (α - 2 * π)) = cos (2 * α) := by
  sorry

end trig_identity_l305_305684


namespace hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305502

noncomputable def hypotenuse_length (a b c : ℝ) : ℝ :=
  let sum_sq := a^2 + b^2 + c^2
  in if sum_sq = 2500 ∧ c^2 = a^2 + b^2 then c else sorry

theorem hypotenuse_of_right_angled_triangle_is_25sqrt2
  {a b c : ℝ} (h1 : a^2 + b^2 + c^2 = 2500) (h2 : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_of_right_angled_triangle_is_25sqrt2_l305_305502


namespace vehicle_count_l305_305229

theorem vehicle_count (T B : ℕ) (h1 : T + B = 15) (h2 : 3 * T + 2 * B = 40) : T = 10 ∧ B = 5 :=
by
  sorry

end vehicle_count_l305_305229


namespace remainder_invariance_l305_305589

theorem remainder_invariance (S A K : ℤ) (h : ∃ B r : ℤ, S = A * B + r ∧ 0 ≤ r ∧ r < |A|) :
  (∃ B' r' : ℤ, S + A * K = A * B' + r' ∧ r' = r) ∧ (∃ B'' r'' : ℤ, S - A * K = A * B'' + r'' ∧ r'' = r) :=
by
  sorry

end remainder_invariance_l305_305589


namespace solve_system_of_equations_l305_305681

theorem solve_system_of_equations :
  ∃ x y : ℝ, (4 * x + y = 12) ∧ (y = 4) ∧ (x = 2) ∧ (3 * x - 2 * y = -2) :=
by
  use [2, 4]
  split
  · -- 4x + y = 12
    rw [mul_comm]
    exact eq.symm (sub_eq_zero.mpr (add_eq_of_eq_sub' (sub_eq_of_eq_add' rfl)))

  split
  · -- y = 4
    rfl

  split
  · -- x = 2
    rfl

  · -- 3x - 2y = -2
    exact (add_eq_of_eq_add'.mpr rfl)

end solve_system_of_equations_l305_305681


namespace no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l305_305895

theorem no_natural_n_such_that_6n2_plus_5n_is_power_of_2 :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 6 * n^2 + 5 * n = 2^k :=
by
  sorry

end no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l305_305895


namespace sum_of_areas_of_two_squares_l305_305231

theorem sum_of_areas_of_two_squares 
  (side_length1 side_length2 : ℕ) 
  (h1 : side_length1 = 8) 
  (h2 : side_length2 = 10) : 
  side_length1 * side_length1 + side_length2 * side_length2 = 164 :=
by
  rw [h1, h2]
  simp
  sorry

end sum_of_areas_of_two_squares_l305_305231


namespace combined_length_of_trains_l305_305649

theorem combined_length_of_trains
  (speed_A_kmph : ℕ) (speed_B_kmph : ℕ)
  (platform_length : ℕ) (time_A_sec : ℕ) (time_B_sec : ℕ)
  (h_speed_A : speed_A_kmph = 72) (h_speed_B : speed_B_kmph = 90)
  (h_platform_length : platform_length = 300)
  (h_time_A : time_A_sec = 30) (h_time_B : time_B_sec = 24) :
  let speed_A_ms := speed_A_kmph * 5 / 18
  let speed_B_ms := speed_B_kmph * 5 / 18
  let distance_A := speed_A_ms * time_A_sec
  let distance_B := speed_B_ms * time_B_sec
  let length_A := distance_A - platform_length
  let length_B := distance_B - platform_length
  length_A + length_B = 600 :=
by
  sorry

end combined_length_of_trains_l305_305649


namespace gcd_power_minus_one_l305_305154

theorem gcd_power_minus_one (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) : gcd (2^a - 1) (2^b - 1) = 2^(gcd a b) - 1 :=
by
  sorry

end gcd_power_minus_one_l305_305154


namespace smallest_K_2005_zero_l305_305363

def sequence (K : ℕ) : ℕ → ℕ
| 1       := K
| (n + 1) := if sequence n % 2 = 0 then sequence n - 1 else (sequence n - 1) / 2

theorem smallest_K_2005_zero : ∃ K, sequence K 2005 = 0 ∧ ∀ K', (sequence K' 2005 = 0 → K ≤ K') :=
  Exists.intro (2^1003 - 2) 
  (And.intro 
    sorry
    (λ K' hK', sorry))

end smallest_K_2005_zero_l305_305363


namespace area_enclosed_by_curves_l305_305521

noncomputable def area_of_figure (θ : ℝ) (ρ : ℝ) : ℝ :=
  if θ = 0 then 0
  else if θ = (π / 3) then √3 * ρ
  else if ρ * Real.cos θ + √3 * ρ * Real.sin θ = 1 then 1
  else 0

theorem area_enclosed_by_curves : 
  let θ₁ := 0
  let θ₂ := π / 3
  let line := λ (ρ : ℝ) (θ : ℝ), ρ * Real.cos θ + √3 * ρ * Real.sin θ = 1
  let A := (x + √3 * y = 1)
  let B := (0, θ=0)
  let C := (x, y= √3 * x)
  let D := (θ=π/3)
  ∃ ρ, ∀ θ, area_of_figure θ ρ = 1/2 * 1 * (√3/4) := 
  by sorry

end area_enclosed_by_curves_l305_305521


namespace product_of_16_and_21_point_3_l305_305102

theorem product_of_16_and_21_point_3 (h1 : 213 * 16 = 3408) : 16 * 21.3 = 340.8 :=
by sorry

end product_of_16_and_21_point_3_l305_305102


namespace number_of_distinct_positive_differences_is_seven_l305_305426

def set_of_integers := {1, 2, 3, 4, 5, 6, 7, 8}

theorem number_of_distinct_positive_differences_is_seven :
  (set_of_integers \ {0}).image (λ x, set_of_integers \ {x}) |>.nonempty :=
by
  sorry

end number_of_distinct_positive_differences_is_seven_l305_305426


namespace subsets_with_union_and_intersection_subset_selection_count_l305_305284

-- Definition of the set S
def S : Set ℕ := {a, b, c, d, e, f, g}

-- Defining the problem statement to be proven
theorem subsets_with_union_and_intersection :
  ∃ (A B : Set ℕ), (A ∪ B = S) ∧ (A ∩ B).card = 3 ∧ (count_indistinguishable_pairs A B S = 280) := by
  sorry

-- Placeholder for function to count indistinguishable pairs of subsets
noncomputable def count_indistinguishable_pairs (A B S : Set ℕ) : ℕ := 
  if (A ∪ B = S) ∧ ((A ∩ B).card = 3)
  then (Nat.choose 7 3 * 2^4 / 2)
  else 0

-- Defining subset selection (order irrelevant) condition
def subset_selection_correct (A B S : Set ℕ) : Prop :=
  (A ∪ B = S) ∧ ((A ∩ B).card = 3)

-- Theorem stating that there are 280 such subsets pairs
theorem subset_selection_count (S : Set ℕ) :
  count_indistinguishable_pairs forall A B S == 280 := by
  sorry

end subsets_with_union_and_intersection_subset_selection_count_l305_305284


namespace solution_set_f_l305_305061

noncomputable def f : ℝ → ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x = f x
axiom f_at_4 : f 4 = -3
axiom f_deriv_lt_3 : ∀ x : ℝ, deriv f x < 3

theorem solution_set_f (x : ℝ) : f x < 3 * x - 15 ↔ x > 4 :=
by 
  sorry

end solution_set_f_l305_305061


namespace black_area_remaining_after_changes_l305_305304

theorem black_area_remaining_after_changes :
  let initial_fraction_black := 1
  let change_factor := 8 / 9
  let num_changes := 4
  let final_fraction_black := (change_factor ^ num_changes)
  final_fraction_black = 4096 / 6561 :=
by
  sorry

end black_area_remaining_after_changes_l305_305304


namespace Heesu_has_greatest_sum_l305_305598

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l305_305598


namespace hypotenuse_length_l305_305479

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305479


namespace problem_1_part_1_problem_1_part_2_l305_305558

-- Define the function f
def f (x a : ℝ) := |x - a| + 3 * x

-- The first problem statement - Part (Ⅰ)
theorem problem_1_part_1 (x : ℝ) : { x | x ≥ 3 ∨ x ≤ -1 } = { x | f x 1 ≥ 3 * x + 2 } :=
by {
  sorry
}

-- The second problem statement - Part (Ⅱ)
theorem problem_1_part_2 : { x | x ≤ -1 } = { x | f x 2 ≤ 0 } :=
by {
  sorry
}

end problem_1_part_1_problem_1_part_2_l305_305558


namespace circumcenter_of_APQ_lies_on_angle_bisector_ABC_l305_305045

-- Define points A, B, C in a plane.
variables {A B C P Q O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace O]

-- Assume certain geometric relationships between points for our conditions
axiom point_on_extension_beyond_b (B C P : Point) (h1 : distance B P = distance B A) : lies_on_extension P B C
axiom point_on_extension_beyond_c (C B Q : Point) (h2 : distance C Q = distance C A) : lies_on_extension Q C B

-- The main theorem to prove
theorem circumcenter_of_APQ_lies_on_angle_bisector_ABC 
    (h1 : distance B P = distance B A) 
    (h2 : distance C Q = distance C A) 
    (circumcenter_O : Circumcenter O (triangle A P Q)) : 
    lies_on_angle_bisector O (angle_bisector (point A) (point B) (point C)) :=
sorry

end circumcenter_of_APQ_lies_on_angle_bisector_ABC_l305_305045


namespace simplified_expression_num_terms_l305_305186

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end simplified_expression_num_terms_l305_305186


namespace sum_first_50_odd_indexed_terms_l305_305988

noncomputable def a (n : ℕ) : ℝ := 0.5 * n -- The arithmetic sequence with a common difference of 0.5

def sum_first_n_terms (n : ℕ) : ℝ := n / 2 * (2 * a 1 + (n - 1) * 0.5) -- Sum of first n terms of the arithmetic sequence

theorem sum_first_50_odd_indexed_terms :
  (a 1 + a 3 + a 5 + ... + a 99) = 60 :=
by
  have h1 : sum_first_n_terms 100 = 145 := sorry, -- sum of first 100 terms
  have common_diff : 0.5, -- common difference
  have diff_between_sums : 50 * common_diff = 25, -- difference property of sums in odd/even positions
  have sum_all_even_odd : (a 2 + a 4 + ... + a 100) - (a 1 + a 3 + ... + a 99) = 25, -- another form of difference property
  have sum_first_n_terms : sum_all_even_odd + 2 * (a 1 + a 3 + ... + a 99) = 145, -- using provided sum of first 100 terms
  sorry -- solve the rest by algebraic manipulation

end sum_first_50_odd_indexed_terms_l305_305988


namespace width_domain_g_l305_305858

noncomputable def h : ℝ → ℝ := sorry
def domain_h := set.Icc (-12 : ℝ) (12 : ℝ)

def g (x : ℝ) := h (x / 3)
def domain_g := set.Icc (-36 : ℝ) (36 : ℝ)

theorem width_domain_g : ∀ x, g x ∈ domain_g → set.ord_connected.interval_width domain_g = 72 := by
  intro x hx
  sorry

end width_domain_g_l305_305858


namespace math_problem_l305_305595

variable (a a' b b' c c' : ℝ)

theorem math_problem 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b * b) 
  (h3 : a' * c' ≥ b' * b') : 
  (a + a') * (c + c') ≥ (b + b') * (b + b') := 
by
  sorry

end math_problem_l305_305595


namespace sufficient_but_not_necessary_l305_305546

variable {a b : ℝ}

theorem sufficient_but_not_necessary (h : b < a ∧ a < 0) : 1 / a < 1 / b :=
by
  sorry

end sufficient_but_not_necessary_l305_305546


namespace lemonade_water_cups_l305_305930

theorem lemonade_water_cups
  (W S L : ℕ)
  (h1 : W = 5 * S)
  (h2 : S = 3 * L)
  (h3 : L = 5) :
  W = 75 :=
by {
  sorry
}

end lemonade_water_cups_l305_305930


namespace inclination_angle_of_line_l305_305247

noncomputable def inclination_angle (x y : ℝ → ℝ) : ℝ :=
  Real.arctan ((y 0 - y 1) / (x 0 - x 1))

theorem inclination_angle_of_line :
  (∃ t : ℝ, (x t = 1 + t ∧ y t = 1 - t)) →
  inclination_angle (λ t, 1 + t) (λ t, 1 - t) = 3 * Real.pi / 4 :=
by
  sorry

end inclination_angle_of_line_l305_305247


namespace number_of_distinct_positive_differences_is_seven_l305_305425

def set_of_integers := {1, 2, 3, 4, 5, 6, 7, 8}

theorem number_of_distinct_positive_differences_is_seven :
  (set_of_integers \ {0}).image (λ x, set_of_integers \ {x}) |>.nonempty :=
by
  sorry

end number_of_distinct_positive_differences_is_seven_l305_305425


namespace unique_solution_l305_305333

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1))

theorem unique_solution : ∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) → system_of_equations x y z → (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z hx hy hz h
  sorry

end unique_solution_l305_305333


namespace lily_milk_amount_l305_305560

def initial_milk : ℚ := 5
def milk_given_to_james : ℚ := 18 / 4
def milk_received_from_neighbor : ℚ := 7 / 4

theorem lily_milk_amount : (initial_milk - milk_given_to_james + milk_received_from_neighbor) = 9 / 4 :=
by
  sorry

end lily_milk_amount_l305_305560


namespace proof_problem_l305_305030

-- Definitions
def f (x : ℝ) (a : ℝ) (b : ℝ) := (-2^x + b) / (2^(x + 1) + a)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def for_all_t (ineq : ℝ → Prop) := ∀ t : ℝ, ineq t

theorem proof_problem :
  ∃ a b : ℝ , 
    (∀ x, f(x) a b = -f(-x) a b) ∧ -- f is an odd function
    f 0 a b = 0 ∧  -- f(0) = 0 which implies b = 1
    f 1 a b = -f (-1) a b ∧ -- f(1) = -f(-1) which implies a = 2
    (∀ t : ℝ, f(t^2 - 2*t) a b + f(2*t^2 - k) a b < 0) →  -- the given inequality
    (a = 2 ∧ b = 1 ∧ ∀ t : ℝ, 3*t^2 - 2*t - k > 0 → k < -1/3) := 
begin
  sorry
end

end proof_problem_l305_305030


namespace no_integer_solutions_l305_305600

theorem no_integer_solutions (a b p x y : ℤ) (hp_prime : Nat.Prime p)
    (h1 : b % 4 = 1)
    (h2 : p % 4 = 3)
    (h3 : ∀ q, Nat.Prime q → q ∣ a → q % 4 = 3 → q ^ p ∣ a^2 ∧ ¬p ∣ (q - 1) ∧ (q = p → q ∣ b)) :
    ¬ ∃ x y : ℤ, x^2 + 4 * a^2 = y^p - b^p :=
by
  sorry

end no_integer_solutions_l305_305600


namespace ratio_A_B_l305_305724

theorem ratio_A_B (w : ℝ) (h : w > 0) : 
  let l := 2 * w in
  let A := w * l in
  let B := A / 2 in
  B / A = 1 / 2 :=
by
  sorry

end ratio_A_B_l305_305724


namespace rational_reachability_l305_305258

-- Define the operations f and g
def f (x : ℚ) : ℚ := (1 + x) / x
def g (x : ℚ) : ℚ := (1 - x) / x

-- The theorem statement: for any two nonzero rational numbers a and b,
-- there exists a finite sequence of applications of f and g that transforms a into b.
theorem rational_reachability (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (n : ℕ) (ops : vector ((ℚ → ℚ)) n), 
  (∀ (i: ℕ) (h: i < n), ops[i] = f ∨ ops[i] = g) ∧ 
  (ops.to_list.foldl (λ x op, op x) a = b) :=
sorry

end rational_reachability_l305_305258


namespace isosceles_triangle_angle_sum_l305_305530

theorem isosceles_triangle_angle_sum (A B C : Type) [IsoscelesTriangle A B C] (h1 : AB = AC) (h2 : ∠B = 55) : ∠A = 70 := 
by 
  sorry

end isosceles_triangle_angle_sum_l305_305530


namespace distinct_positive_differences_l305_305428

theorem distinct_positive_differences :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  ∑ i in s, ∑ j in s, if i ≠ j then (Nat.abs (i - j) : Finset ℕ) else 0 = {1, 2, 3, 4, 5, 6, 7} :=
sorry

end distinct_positive_differences_l305_305428


namespace path_length_correct_l305_305987

def segment_length (PQ : ℤ) : Prop :=
  PQ = 73

def path_length (PQ : ℤ) : ℤ :=
  3 * PQ

theorem path_length_correct (PQ : ℤ) (h : segment_length PQ) : path_length PQ = 219 := by
  rw [segment_length, path_length]
  rw h
  norm_num
  sorry

end path_length_correct_l305_305987


namespace triangle_side_angle_inequality_l305_305590

theorem triangle_side_angle_inequality {A B C G M2 : Type} [triangle A B C G] [midpoint M2 C A] :
  (AB > BC) ↔ (angle A G M2 > angle C G M2) :=
sorry

end triangle_side_angle_inequality_l305_305590


namespace wheel_radius_increase_proof_l305_305768

noncomputable def radius_increase (orig_distance odometer_distance : ℝ) (orig_radius : ℝ) : ℝ :=
  let orig_circumference := 2 * Real.pi * orig_radius
  let distance_per_rotation := orig_circumference / 63360
  let num_rotations_orig := orig_distance / distance_per_rotation
  let num_rotations_new := odometer_distance / distance_per_rotation
  let new_distance := orig_distance
  let new_radius := (new_distance / num_rotations_new) * 63360 / (2 * Real.pi)
  new_radius - orig_radius

theorem wheel_radius_increase_proof :
  radius_increase 600 580 16 = 0.42 :=
by 
  -- The proof is skipped.
  sorry

end wheel_radius_increase_proof_l305_305768


namespace log_difference_l305_305459

theorem log_difference (a : ℝ) (h : 1 + a^2 = 5) : 
  let b_max := log (by norm_num : (2 : ℝ))
      b_min := log (by norm_num : (1 / 4 : ℝ))
  in b_max - b_min = 3 := by
  sorry

end log_difference_l305_305459


namespace total_smaller_cubes_is_27_l305_305702

-- Given conditions
def painted_red (n : ℕ) : Prop := ∀ face, face ∈ cube.faces → face.color = red

def cut_into_smaller_cubes (n : ℕ) : Prop := ∃ k : ℕ, k = n + 1

def smaller_cubes_painted_on_2_faces (cubes_painted_on_2_faces : ℕ) (n : ℕ) : Prop :=
  cubes_painted_on_2_faces = 12 * (n - 1)

-- Question: Prove the total number of smaller cubes is equal to 27, given the conditions
theorem total_smaller_cubes_is_27 (n : ℕ) (h1 : painted_red n) (h2 : cut_into_smaller_cubes n) (h3 : smaller_cubes_painted_on_2_faces 12 n) :
  (n + 1)^3 = 27 := by
  sorry

end total_smaller_cubes_is_27_l305_305702


namespace acute_triangle_bound_l305_305228

open Finset

theorem acute_triangle_bound (P : Finset (ℝ × ℝ)) (h75 : P.card = 75) (h_non_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ P → p2 ∈ P → p3 ∈ P → (collinear (p1, p2, p3) → false)) :
  let total_triangles := 75.choose 3 in
  let max_acute_triangles := 0.7 * total_triangles in
  ∃ acute_triangles, acute_triangles ≤ max_acute_triangles :=
begin
  sorry,
end

end acute_triangle_bound_l305_305228


namespace triangle_is_isosceles_l305_305991

theorem triangle_is_isosceles
  (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (H : ∀ n : ℕ, n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n)) :
  p = q ∨ q = r ∨ r = p :=
by
  sorry

end triangle_is_isosceles_l305_305991


namespace count_12_digit_numbers_with_consecutive_1s_l305_305082

-- Define the recurrence relation for F_n
def F : ℕ → ℕ
| 0     := 1 -- base case
| 1     := 2
| 2     := 3
| (n+3) := F (n+2) + F (n+1)

-- Calculate the total number of 12-digit numbers
def total_12_digit_numbers := 2 ^ 12

-- Calculate F_12 for the number of numbers without two consecutive 1s
def F_12 := F 12

-- Formal statement
theorem count_12_digit_numbers_with_consecutive_1s : 
  total_12_digit_numbers - F_12 = 3719 := 
  by
    sorry

end count_12_digit_numbers_with_consecutive_1s_l305_305082


namespace smaller_tv_diagonal_l305_305995

/-- Define the function to calculate the area of a square television given the diagonal -/
def area (d : ℝ) : ℝ := (d / Real.sqrt 2) ^ 2

/-- Given the conditions of the problem, prove that the diagonal of the smaller television is 25 inches -/
theorem smaller_tv_diagonal :
  ∃ d : ℝ, (d / Real.sqrt 2) ^ 2 + 79.5 = (28 / Real.sqrt 2) ^ 2 ∧ d = 25 :=
by
  use 25
  sorry

end smaller_tv_diagonal_l305_305995


namespace rainfall_ratio_l305_305876

theorem rainfall_ratio (rain_15_days : ℕ) (total_rain : ℕ) (days_in_month : ℕ) (rain_per_day_first_15 : ℕ) :
  rain_per_day_first_15 * 15 = rain_15_days →
  rain_15_days + (days_in_month - 15) * (rain_per_day_first_15 * 2) = total_rain →
  days_in_month = 30 →
  total_rain = 180 →
  rain_per_day_first_15 = 4 →
  2 = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rainfall_ratio_l305_305876


namespace angle_EHG_65_l305_305173

/-- Quadrilateral $EFGH$ has $EF = FG = GH$, $\angle EFG = 80^\circ$, and $\angle FGH = 150^\circ$; and hence the degree measure of $\angle EHG$ is $65^\circ$. -/
theorem angle_EHG_65 {EF FG GH : ℝ} (h1 : EF = FG) (h2 : FG = GH) 
  (EFG : ℝ) (FGH : ℝ) (h3 : EFG = 80) (h4 : FGH = 150) : 
  ∃ EHG : ℝ, EHG = 65 :=
by
  sorry

end angle_EHG_65_l305_305173


namespace black_white_area_ratio_l305_305789

theorem black_white_area_ratio :
  let radius1 := 2
  let radius2 := 4
  let radius3 := 6
  let radius4 := 8
  let area (r : ℕ) := Real.pi * (r ^ 2)
  -- Areas of individual circles
  let area1 := area radius1
  let area2 := area radius2
  let area3 := area radius3
  let area4 := area radius4
  -- Areas of the rings
  let area_ring_white1 := area2 - area1
  let area_ring_black := area3 - area2
  let area_ring_white2 := area4 - area3
  -- Total areas for black and white regions
  let total_black_area := area1 + area_ring_black
  let total_white_area := area_ring_white1 + area_ring_white2
  -- Ratio of black area to white area
  ratio_black_to_white := total_black_area / total_white_area
  ratio_black_to_white = 3 / 5 :=
sorry

end black_white_area_ratio_l305_305789


namespace team_selection_l305_305713

open Nat

theorem team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_to_choose := 5
  let girls_to_choose := 3
  choose boys boys_to_choose * choose girls girls_to_choose = 55440 :=
by
  sorry

end team_selection_l305_305713


namespace Heesu_has_greatest_sum_l305_305599

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l305_305599


namespace sum_common_prime_divisors_390_9450_l305_305978

theorem sum_common_prime_divisors_390_9450 : 
  let common_prime_divisors (a b : ℕ) : List ℕ := 
    (List.filter Nat.Prime (Nat.factors a)).inter (List.filter Nat.Prime (Nat.factors b)) 
  in List.sum (common_prime_divisors 390 9450) = 10 :=
by
  have h1 : Nat.Prime 2 := by sorry
  have h2 : Nat.Prime 3 := by sorry
  have h3 : Nat.Prime 5 := by sorry
  have h4 : Nat.factors 390 = [2, 3, 5, 13] := by sorry
  have h5 : Nat.factors 9450 = [2, 3, 3, 3, 5, 5, 7] := by sorry
  have h6 : common_prime_divisors 390 9450 = [2, 3, 5] := by
    unfold common_prime_divisors
    simp [h4, h5, h1, h2, h3]
  simp [h6]
  norm_num

end sum_common_prime_divisors_390_9450_l305_305978


namespace gcd_of_128_144_480_is_16_l305_305654

-- Define the three numbers
def a := 128
def b := 144
def c := 480

-- Define the problem statement in Lean
theorem gcd_of_128_144_480_is_16 : Int.gcd (Int.gcd a b) c = 16 :=
by
  -- Definitions using given conditions
  -- use Int.gcd function to define the problem precisely.
  -- The proof will be left as "sorry" since we don't need to solve it
  sorry

end gcd_of_128_144_480_is_16_l305_305654


namespace y_pow_x_eq_x_pow_y_l305_305262

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) :
    let x := (1 + 1 / (n : ℝ)) ^ n
    let y := (1 + 1 / (n : ℝ)) ^ (n + 1)
    y ^ x = x ^ y := 
    sorry

end y_pow_x_eq_x_pow_y_l305_305262


namespace area_ratio_and_sum_mn_l305_305130

noncomputable def triangle_area_ratio (p q r : ℝ) (AB BC CA : ℝ) 
(pqs : p + q + r = 2/3) 
(p2q2r2 : p^2 + q^2 + r^2 = 2/5) : ℝ :=
  let pq_plus_qr_plus_rp := (2/3)^2 - 2/5 / 2 in
  pq_plus_qr_plus_rp - (p + q + r) + 1

theorem area_ratio_and_sum_mn (p q r : ℝ)
  (AB BC CA : ℝ) 
  (h_AB : AB = 13) (h_BC : BC = 15) (h_CA : CA = 17)
  (pqs : p + q + r = 2/3) 
  (p2q2r2 : p^2 + q^2 + r^2 = 2/5) :
  let ratio := triangle_area_ratio p q r AB BC CA pqs p2q2r2 in
  ratio = 16 / 45 ∧ (16 + 45 = 61) :=
by
  sorry

end area_ratio_and_sum_mn_l305_305130


namespace time_friday_equals_1_l305_305897

-- Definitions based on the conditions
def total_time_week : ℝ := 5
def time_monday : ℝ := 1.5
def time_wednesday : ℝ := 1.5
def time_tuesday := time_friday

-- Statement we need to prove
theorem time_friday_equals_1 :
  let time_friday := (total_time_week - (time_monday + time_wednesday)) / 2 in
  time_friday = 1 :=
by
  sorry

end time_friday_equals_1_l305_305897


namespace min_dihedral_sum_cube_l305_305375

def is_dihedral_angle (P A' B' C' : Point) (base: Plane) : ℝ → Prop := sorry
def is_edge_point (P: Point) (A C: Point) : Prop := sorry
def cube (A B C D A' B' C' D' : Point) : Prop := sorry
def edge_length_one (A B : Point) : Prop := dist A B = 1
def min_dihedral_sum (alpha beta : ℝ): ℝ := α + β

theorem min_dihedral_sum_cube (A B C D A' B' C' D' P : Point) (alpha beta : ℝ)
  (h1 : cube A B C D A' B' C' D')
  (h2: edge_length_one A B)
  (h3: is_edge_point P A C)
  (h4: is_dihedral_angle P A' B' (Plane A' B' C' D') alpha)
  (h5: is_dihedral_angle P B' C' (Plane A' B' C' D') beta) :
  min_dihedral_sum alpha beta = π - arctan (4 / 3) :=
sorry

end min_dihedral_sum_cube_l305_305375


namespace conditional_probability_sum_greater_than_2_first_odd_l305_305508

theorem conditional_probability_sum_greater_than_2_first_odd :
  let outcomes_dice := {1, 2, 3, 4, 5, 6}
  let odd_numbers := {1, 3, 5}
  let favorable_outcomes : ℕ := 
    (outcomes_dice.filter (λ x, x > 1)).card + 
    outcomes_dice.card + 
    outcomes_dice.card
  let total_possible_outcomes : ℕ := 3 * outcomes_dice.card
  (favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 17/18 :=
by
  sorry

end conditional_probability_sum_greater_than_2_first_odd_l305_305508


namespace total_smaller_cubes_is_27_l305_305703

-- Given conditions
def painted_red (n : ℕ) : Prop := ∀ face, face ∈ cube.faces → face.color = red

def cut_into_smaller_cubes (n : ℕ) : Prop := ∃ k : ℕ, k = n + 1

def smaller_cubes_painted_on_2_faces (cubes_painted_on_2_faces : ℕ) (n : ℕ) : Prop :=
  cubes_painted_on_2_faces = 12 * (n - 1)

-- Question: Prove the total number of smaller cubes is equal to 27, given the conditions
theorem total_smaller_cubes_is_27 (n : ℕ) (h1 : painted_red n) (h2 : cut_into_smaller_cubes n) (h3 : smaller_cubes_painted_on_2_faces 12 n) :
  (n + 1)^3 = 27 := by
  sorry

end total_smaller_cubes_is_27_l305_305703


namespace problem_statement_l305_305085

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fib(n+1) + fib(n)

def num_12digit_with_two_consecutive_1s : ℕ :=
  let total := 2 ^ 12
  let without_consecutive_1s := fib 13
  total - without_consecutive_1s

theorem problem_statement : num_12digit_with_two_consecutive_1s = 3719 := by
  sorry

end problem_statement_l305_305085


namespace problem_1_problem_2_problem_3_l305_305815

/-- Given conditions on vectors a and b in some inner product space --/
variables {V : Type*} [inner_product_space ℝ V] (a b : V)
hypothesis (h₁ : ∥a∥ = 10)
hypothesis (h₂ : ∥b∥ = 12)
hypothesis (h₃ : real.angle a b = real.pi / 3)

/-- To prove the following propositions --/
theorem problem_1 : inner a b = -60 :=
sorry

theorem problem_2 : inner (3 • a) ((1/5) • b) = -36 :=
sorry

theorem problem_3 : inner (3 • b - 2 • a) (4 • a + b) = -968 :=
sorry

end problem_1_problem_2_problem_3_l305_305815


namespace relationship_y1_y2_l305_305107

theorem relationship_y1_y2 (y1 y2 : ℝ) : 
  (y1 = -4 * (-2) - 3) ∧ (y2 = -4 * 5 - 3) → y2 < y1 :=
by {
  intro h,
  cases h with hy1 hy2,
  rw [neg_mul, hy1, hy2],
  exact sorry
}

end relationship_y1_y2_l305_305107


namespace parabola_slope_condition_l305_305073

theorem parabola_slope_condition :
  (∀ A B : ℝ × ℝ,
    (M : ℝ × ℝ) = (-2, 2) →
    ∃ k : ℝ,
    y^2 = 8 * x →
    (line_through_focus : ℝ → ℝ) = λ x, k * (x - 2) →
    (A = (x1, y1)) ∧ (B = (x2, y2)) →
    (vector_dot_product : ℝ × ℝ → ℝ × ℝ → ℝ)
      (x1 + 2, y1 - 2)
      (x2 + 2, y2 - 2) = 0 →
    k = 2.

end parabola_slope_condition_l305_305073


namespace z_in_third_quadrant_l305_305795

def z : ℂ := (3 * complex.I) / (1 - complex.I)

theorem z_in_third_quadrant : [z.re, z.im] ∈ {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0} :=
by
  sorry

end z_in_third_quadrant_l305_305795


namespace sum_modulus_l305_305366

def b (p : ℕ) : ℕ := 
  (λ k, (k : ℚ) ∈ Set.Ico (Real.sqrt p - (1/2 : ℚ)) (Real.sqrt p + (1/2 : ℚ))) ∘ Nat.pred
  sorry -/

def S : ℕ := ∑ p in Finset.range 2007, b (p + 1)

theorem sum_modulus :
  S % 1000 = 955 :=
by
  sorry

end sum_modulus_l305_305366


namespace pizza_area_comparison_l305_305602

def area (r : ℝ) : ℝ := Real.pi * r^2

theorem pizza_area_comparison : 
  let radius1 := 5
  let radius2 := 4
  let area1 := area radius1
  let area2 := area radius2
  let percent_increase := ((area1 - area2) / area2) * 100
  ⌊percent_increase⌉ = 56 :=
by
  sorry

end pizza_area_comparison_l305_305602


namespace Nigel_mothers_money_l305_305573

theorem Nigel_mothers_money :
  ∃ M : ℝ,
  (∀ (original_amount given_away twice_original final_amount : ℝ),
    original_amount = 45 →
    given_away = 25 →
    twice_original = 2 * original_amount →
    final_amount = twice_original + 10 →
    (original_amount - given_away + M = final_amount) →
    M = 80) :=
begin
  sorry
end

end Nigel_mothers_money_l305_305573


namespace probability_of_intersection_l305_305989

open Finset

noncomputable def a_seq (n : ℕ) := 6 * n - 4
noncomputable def b_seq (n : ℕ) := 2 ^ (n - 1)

def A := Finset.image a_seq (range 6).map (λ n, n + 1)
def B := Finset.image b_seq (range 6).map (λ n, n + 1)

def prob_A_inter_B : ℚ := (card (A ∩ B) : ℚ) / (card (A ∪ B) : ℚ)

theorem probability_of_intersection : prob_A_inter_B = 1 / 3 := sorry

end probability_of_intersection_l305_305989


namespace problem8x_eq_5_200timesreciprocal_l305_305448

theorem problem8x_eq_5_200timesreciprocal (x : ℚ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := 
by 
  sorry

end problem8x_eq_5_200timesreciprocal_l305_305448


namespace tensor_example_l305_305851
-- Import the necessary library

-- Define the binary operation ⊗
def tensor (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the main theorem
theorem tensor_example : tensor (tensor 8 6) 2 = 9 / 5 := by
  sorry

end tensor_example_l305_305851


namespace correct_statements_for_f_l305_305174

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.cos((21 / 4) * Real.pi - 2 * x)

theorem correct_statements_for_f :
  (f (5 * Real.pi / 8) = (1 / 2) * Real.cos(4 * Real.pi)) ∧
  (f (7 * Real.pi / 8) = 0) ∧
  (∀ x : ℝ, (f x ≠ (1 / 2) * Real.sin(2 * (x - 3 * Real.pi / 8)))) ∧
  (∀ x : ℝ, -Real.pi / 2 < x ∧ x ≤ 0 → f x ∈ Icc (-Real.sqrt(2) / 4) (1 / 2)) :=
by sorry

end correct_statements_for_f_l305_305174


namespace count_valid_numbers_l305_305089

def is_valid_number (n : ℕ) : Prop :=
  (n % 7 = 3) ∧ (n % 10 = 6) ∧ (n % 13 = 8) ∧ (100 ≤ n) ∧ (n < 1000)

theorem count_valid_numbers : finset.card (finset.filter is_valid_number (finset.range 1000)) = 5 := by
  sorry

end count_valid_numbers_l305_305089


namespace quadratic_binomial_form_l305_305202

theorem quadratic_binomial_form (y : ℝ) : ∃ (k : ℝ), y^2 + 14 * y + 40 = (y + 7)^2 + k :=
by
  use -9
  sorry

end quadratic_binomial_form_l305_305202


namespace distinct_differences_count_l305_305438

-- Define the set of interest.
def mySet : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- The statement we want to prove.
theorem distinct_differences_count : 
  (finset.image (λ (x : ℕ × ℕ), (x.1 - x.2)) ((mySet.product mySet).filter (λ x, x.1 > x.2))).card = 7 :=
sorry

end distinct_differences_count_l305_305438


namespace ellipse_equation_max_triangle_area_l305_305824

noncomputable def ellipse_c (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def tangent_intersection (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : Prop :=
  ellipse_c a b h₁ h₂ (-Math.sqrt(2) * b) 0

noncomputable def min_dot_product (a b x y : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : ellipse_c a b h₁ h₂ x y) : Prop :=
  let pp1 := (x + 1, y)
  let pp2 := (x - 1, y)
  (pp1.1 * pp2.1 + pp1.2 * pp2.2) = a / 2

theorem ellipse_equation (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∀ x y, ellipse_c a b h₁ h₂ x y) → 
  (a = Math.sqrt(2) * b) → (x^2 / 4 + y^2 / 2 = 1) :=
sorry

noncomputable def line_intersection (k m a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧ ellipse_c a b h₁ h₂ x₁ y₁ ∧ ellipse_c a b h₁ h₂ x₂ y₂

noncomputable def midpoint (O A B : ℝ) : ℝ := (A + B) / 2

noncomputable def area_triangle (A B O : ℝ) : ℝ := 
  0.5 * (O - A) * (B - O)

noncomputable def T_range (MP1 MP2 T : ℝ) : Prop :=
  T = (1 / MP1^2) - 2 * MP2 ∧ T ∈ [3 - 4 * Math.sqrt(2), 1)

theorem max_triangle_area (k m a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (x₁ x₂ y₁ y₂ MP1 MP2 : ℝ) :
  line_intersection k m a b h₁ h₂ x₁ x₂ y₁ y₂ → 
  ellipse_c a b h₁ h₂ x₁ y₁ → ellipse_c a b h₁ h₂ x₂ y₂ →
  T_range MP1 MP2 ((1 / MP1^2) - 2 * MP2) :=
sorry

end ellipse_equation_max_triangle_area_l305_305824


namespace segment_shadow_ratio_l305_305946

theorem segment_shadow_ratio (a b a' b' : ℝ) (h : a / b = a' / b') : a / a' = b / b' :=
sorry

end segment_shadow_ratio_l305_305946


namespace find_x_l305_305034

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : nat.factors x = [31, p1, p2]) : x = 32767 :=
by
  sorry

end find_x_l305_305034


namespace incorrect_option_D_l305_305467

-- The definition of a normal distribution and its properties
def normal_distribution (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∀ x, X x = (exp (-(x - μ)^2 / (2 * σ^2)) / (σ * sqrt (2 * pi)))

-- Given distribution parameters
def scores_distribution : Prop :=
  normal_distribution 110 σ X

-- Given properties of the normal distribution
def normal_properties (μ σ : ℝ) :=
  (∀ X, P (μ - σ ≤ X ≤ μ + σ) ≈ 0.6827) ∧
  (∀ X, P (μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0.9545) ∧
  (∀ X, P (μ - 3σ ≤ X ≤ μ + 3σ) ≈ 0.9973)

-- Main statement to prove
theorem incorrect_option_D (σ : ℝ) (X : ℝ → ℝ) : 
  scores_distribution →
  normal_properties 110 σ →
  (σ = 20 → ¬ P (X < 130) = 0.6827) :=
by 
  intros h1 h2 h3
  sorry

end incorrect_option_D_l305_305467


namespace f_31_eq_neg1_l305_305060

-- Definition of the function f based on the given conditions
def f : ℝ → ℝ :=
  λ x, if x ∈ Icc 0 1 then real.log (x + 1) / real.log 2 else 0 -- Placeholder for definition outside [0, 1]

-- Given conditions
axiom h1 : ∀ x, f (-x) = -f x
axiom h2 : ∀ x, f (1 + x) = f (1 - x)
axiom h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = real.log (x + 1) / real.log 2

-- Prove f(31) = -1
theorem f_31_eq_neg1 : f 31 = -1 := by
  sorry

end f_31_eq_neg1_l305_305060


namespace vincent_total_packs_l305_305669

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l305_305669


namespace smaller_cubes_count_l305_305710

theorem smaller_cubes_count (painted_faces: ℕ) (edge_cubes: ℕ) (total_cubes: ℕ) : 
  (painted_faces = 2 ∧ edge_cubes = 12) → total_cubes = 27 :=
by
  assume h : (painted_faces = 2 ∧ edge_cubes = 12)
  sorry

end smaller_cubes_count_l305_305710


namespace check_cofactors_l305_305349

def D : Matrix ℤ (Fin 3) (Fin 3) :=
  ![![ -1, 2, 3],
    ![  2, 0, -3],
    ![  3, 2,  5]]

def A_13 (m : Matrix ℤ (Fin 2) (Fin 2)) : ℤ :=
  (-1) ^ (1 + 3) * (m.det)

def A_21 (m : Matrix ℤ (Fin 2) (Fin 2)) : ℤ :=
  (-1) ^ (2 + 1) * (m.det)

def A_31 (m : Matrix ℤ (Fin 2) (Fin 2)) : ℤ :=
  (-1) ^ (3 + 1) * (m.det)

lemma minor_13 : A_13 ![![ 2, 0],
                         ![ 3, 2]] = 4 := by sorry

lemma minor_21 : A_21 ![![ 2, 3],
                         ![ 2, 5]] = -4 := by sorry

lemma minor_31 : A_31 ![![ 2, 3],
                         ![ 0, -3]] = -6 := by sorry

theorem check_cofactors : 
  A_13 ![![ 2, 0],
         ![ 3, 2]] = 4 ∧ 
  A_21 ![![ 2, 3],
         ![ 2, 5]] = -4 ∧ 
  A_31 ![![ 2, 3],
         ![ 0, -3]] = -6 := by
  exact ⟨by apply minor_13, by apply minor_21, by apply minor_31⟩

end check_cofactors_l305_305349


namespace book_purchase_equation_l305_305001

-- Definition of the conditions
def first_purchase (x : ℕ) : ℕ := x
def second_purchase (x : ℕ) : ℕ := x + 60
def cost_per_book_first (x : ℕ) : ℝ := 7000 / x
def cost_per_book_second (x : ℕ) : ℝ := 9000 / (x + 60)

-- Problem statement to prove
theorem book_purchase_equation (x : ℕ) (hx : x ≠ 0) (hx60 : x + 60 ≠ 0) :
  cost_per_book_first x = cost_per_book_second x := sorry

end book_purchase_equation_l305_305001


namespace cos_alpha_value_l305_305458

theorem cos_alpha_value (α β γ: ℝ) (h1: β = 2 * α) (h2: γ = 4 * α)
 (h3: 2 * (Real.sin β) = (Real.sin α + Real.sin γ)) : Real.cos α = -1/2 := 
by
  sorry

end cos_alpha_value_l305_305458


namespace num_paths_A_to_C_through_B_l305_305303

-- Points in the grid
inductive Point 
| A 
| B 
| C 
| D 
| E 
| F 
| G 
| H 
| I 

open Point

-- Conditions: travel only right or down
def isValidMove (p1 p2 : Point) : Prop :=
  match p1, p2 with
  | A, D => true
  | A, F => true
  | D, B => true
  | D, E => true
  | F, B => true
  | F, H => true
  | B, G => true
  | B, I => true
  | G, C => true
  | I, C => true
  | _, _ => false

-- Definition of Path
def Path : Type := List Point

-- Path validity definition
def isValidPath : Path → Prop 
| [] => true
| [p] => true
| p1 :: p2 :: ps => isValidMove p1 p2 ∧ isValidPath (p2 :: ps)

-- Main statement: there are exactly 4 valid paths from A to C passing through B
theorem num_paths_A_to_C_through_B : 
  ∃! (paths : List Path), (∀ p ∈ paths, isValidPath p ∧ A ∈ p.head' ∧ B ∈ p ∧ C ∈ p.last') ∧ paths.length = 4 :=
sorry

end num_paths_A_to_C_through_B_l305_305303


namespace number_of_partitions_l305_305909

-- Definition of the problem in Lean 4
theorem number_of_partitions (P : set ℕ) (hP : P.card = 7)
  (hP_prime : ∀ p ∈ P, Nat.Prime p) (C : set ℕ)
  (hC : C = { n | ∃ p1 p2 ∈ P, n = p1 * p2 } ∧ C.card = 28)
  (hPartition : ∀ S ∈ partitions C 7, 
                ∀ A ∈ S, A.card = 4 ∧ 
                ∀ a1 a2 ∈ A, ∃ p ∈ P, p ∣ a1 ∧ p ∣ a2 ∧ 
                (∃ a3 ∈ A, p ∣ a3) ∧ (∃ a4 ∈ A, p ∣ a4)) :
  ∃! S ∈ partitions C 7, ∀ A ∈ S, 
                       A.card = 4 ∧ 
                       ∀ a1 a2 ∈ A, ∃ p ∈ P, p ∣ a1 ∧ p ∣ a2 ∧ 
                       (∃ a3 ∈ A, p ∣ a3) ∧ (∃ a4 ∈ A, p ∣ a4) :=
begin
  sorry
end

end number_of_partitions_l305_305909


namespace domain_of_f_range_of_f_decreasing_on_minus_inf_minus_one_increasing_on_minus_one_inf_l305_305403

noncomputable def f (x : ℝ) : ℝ := Real.logr 2 |x + 1|

theorem domain_of_f : (Set.Ioo (-∞) (-1) ∪ Set.Ioo (-1) (∞) = set_of (λ x, f x ≠ 0)) := sorry

theorem range_of_f : (range f = Set.univ) := sorry

theorem decreasing_on_minus_inf_minus_one : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ ∈ Set.Ioo (-∞) (-1) ∧ x₂ ∈ Set.Ioo (-∞) (-1) → f x₁ > f x₂ := sorry

theorem increasing_on_minus_one_inf : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ ∈ Set.Ioo (-1) (∞) ∧ x₂ ∈ Set.Ioo (-1) (∞) → f x₁ < f x₂ := sorry

end domain_of_f_range_of_f_decreasing_on_minus_inf_minus_one_increasing_on_minus_one_inf_l305_305403


namespace months_for_three_times_collection_l305_305143

def Kymbrea_collection (n : ℕ) : ℕ := 40 + 3 * n
def LaShawn_collection (n : ℕ) : ℕ := 20 + 5 * n

theorem months_for_three_times_collection : ∃ n : ℕ, LaShawn_collection n = 3 * Kymbrea_collection n ∧ n = 25 := 
by
  sorry

end months_for_three_times_collection_l305_305143


namespace determine_last_card_back_l305_305575

theorem determine_last_card_back (n k l : ℕ) (cards : Fin (n + 1) → Fin (n + 2) × Fin (n + 2))
  (shown : Fin (n + 1) → Fin (n + 2)) :
  (∃ i j, 
    (∀ m, k ≤ m ≤ n → shown m = (cards i).1) ∨ 
    (∀ m, 0 ≤ m ≤ k → shown m = (cards i).2) ∨ 
    (k < l → (∀ m, k ≤ m < l → shown m = (cards i).1) ∧ (shown l = (cards j).2))
  ↔ 
  ∃ m, (shown (n - 1) = k ∨ shown (n - 1) = k - 1)
      ∨ (shown 0 = k + 1 ∨ shown 0 = k - 1)
      ∨ ((k < l) ∧ shown l = k - 1 ∧ (∀ m, k ≤ m < l → shown m = l))
) := 
sorry

end determine_last_card_back_l305_305575


namespace proof_problem_l305_305879

-- Definition of the parametric equations for circle C
def parametric_circle (α : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos α, 3 * Real.sin α)

-- Definition of the polar equation of line l
def polar_line : ℝ → Prop := λ θ, θ = Real.pi / 4

-- Definitions related to the proof problem
theorem proof_problem :
  (∀ (α : ℝ), ∃ (x y : ℝ), (x, y) = parametric_circle α) →
  (∃ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 9 ∧ (x, y) = (2, 0)) ∧
  (polar_line (Real.pi / 4)) →
  let d := Real.abs (2 - 0) / Real.sqrt 2,
      r := 3,
      chord_ab := 2 * Real.sqrt (r ^ 2 - d ^ 2),
      area_abc := 1 / 2 * chord_ab * d
  in area_abc = Real.sqrt 14 :=
by
  sorry

end proof_problem_l305_305879


namespace alice_preferred_number_l305_305734

def preferred_number (n : ℕ) : Prop :=
  n > 100 ∧ n < 200 ∧
  n % 11 = 0 ∧
  n % 2 ≠ 0 ∧
  (n.digits.sum % 3 = 0)

theorem alice_preferred_number : ∃ n, preferred_number n ∧ n = 165 :=
by {
  use 165,
  split,
  sorry,
}

end alice_preferred_number_l305_305734


namespace relationship_abc_l305_305025

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l305_305025


namespace find_a_of_even_function_l305_305455

-- Defining the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + (a-1)*x + a

-- Definition for the function f being even
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- The proof problem statement
theorem find_a_of_even_function : ∀ a : ℝ, is_even (λ x, f x a) → a = 1 :=
by
  -- The proof steps will go here, but we include sorry to indicate this is incomplete
  sorry

end find_a_of_even_function_l305_305455


namespace angle_sum_of_eq_z5_neg_32i_is_990_deg_l305_305006

theorem angle_sum_of_eq_z5_neg_32i_is_990_deg : 
  let z : ℂ → ℂ := λ x, x * x * x * x * x,
      θ (k : ℕ) : ℝ := (270 + 360 * k) / 5 in
  ∑ k in finset.range 5, θ k = 990 :=
by sorry

end angle_sum_of_eq_z5_neg_32i_is_990_deg_l305_305006


namespace hypotenuse_length_l305_305488

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305488


namespace Sharik_cannot_eat_all_meatballs_within_one_million_flies_l305_305120

theorem Sharik_cannot_eat_all_meatballs_within_one_million_flies:
  (∀ n: ℕ, ∃ i: ℕ, i > n ∧ ((∀ j < i, ∀ k: ℕ, ∃ m: ℕ, (m ≠ k) → (∃ f, f < 10^6) )) → f > 10^6 ) :=
sorry

end Sharik_cannot_eat_all_meatballs_within_one_million_flies_l305_305120


namespace rachel_earnings_one_hour_l305_305950

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end rachel_earnings_one_hour_l305_305950


namespace count_12_digit_numbers_with_consecutive_1s_l305_305083

-- Define the recurrence relation for F_n
def F : ℕ → ℕ
| 0     := 1 -- base case
| 1     := 2
| 2     := 3
| (n+3) := F (n+2) + F (n+1)

-- Calculate the total number of 12-digit numbers
def total_12_digit_numbers := 2 ^ 12

-- Calculate F_12 for the number of numbers without two consecutive 1s
def F_12 := F 12

-- Formal statement
theorem count_12_digit_numbers_with_consecutive_1s : 
  total_12_digit_numbers - F_12 = 3719 := 
  by
    sorry

end count_12_digit_numbers_with_consecutive_1s_l305_305083


namespace new_average_age_with_teacher_l305_305605

-- Define the given conditions
def students : ℕ := 30
def avg_students_age : ℕ := 15
def teacher_age : ℕ := 46

-- Mathematical statement to prove the new average age including the teacher
theorem new_average_age_with_teacher :
  let total_students_age := students * avg_students_age in
  let total_age_with_teacher := total_students_age + teacher_age in
  let new_total_individuals := students + 1 in
  total_age_with_teacher / new_total_individuals = 16 :=
by
  -- The proof will go here
  sorry

end new_average_age_with_teacher_l305_305605


namespace beatrice_tv_ratio_l305_305314

theorem beatrice_tv_ratio (T1 T2 T Ttotal : ℕ)
  (h1 : T1 = 8)
  (h2 : T2 = 10)
  (h_total : Ttotal = 42)
  (h_T : T = Ttotal - T1 - T2) :
  (T / gcd T T1, T1 / gcd T T1) = (3, 1) :=
by {
  sorry
}

end beatrice_tv_ratio_l305_305314


namespace next_podcast_duration_l305_305178

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l305_305178


namespace games_given_away_l305_305422

/-- Gwen had ninety-eight DS games. 
    After she gave some to her friends she had ninety-one left.
    Prove that she gave away 7 DS games. -/
theorem games_given_away (original_games : ℕ) (games_left : ℕ) (games_given : ℕ) 
  (h1 : original_games = 98) 
  (h2 : games_left = 91) 
  (h3 : games_given = original_games - games_left) : 
  games_given = 7 :=
sorry

end games_given_away_l305_305422


namespace length_of_segment_l305_305656

theorem length_of_segment : ∃ (a b : ℝ), (|a - (16 : ℝ)^(1/5)| = 3) ∧ (|b - (16 : ℝ)^(1/5)| = 3) ∧ abs (a - b) = 6 :=
by
  sorry

end length_of_segment_l305_305656


namespace farmer_apples_l305_305203

theorem farmer_apples (initial_apples : ℕ) (given_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 127) (h2 : given_apples = 88) 
  (h3 : final_apples = initial_apples - given_apples) : final_apples = 39 :=
by {
  -- proof steps would go here, but since only the statement is needed, we use 'sorry' to skip the proof
  sorry
}

end farmer_apples_l305_305203


namespace relationship_y1_y2_l305_305860

-- Define the conditions
variables {y1 y2 : ℝ}
def A := (1, y1)
def B := (-1, y2)
def line_equation : ℝ × ℝ → Prop := λ p, p.2 = -3 * p.1 + 2

-- Define the conditions using Lean definitions
def on_line_A : Prop := line_equation A
def on_line_B : Prop := line_equation B
def x1_greater_x2 : Prop := 1 > -1

-- Rewrite the problem statement in Lean
theorem relationship_y1_y2 (hA : on_line_A) (hB : on_line_B) (h_x : x1_greater_x2) : y1 < y2 :=
  sorry

end relationship_y1_y2_l305_305860


namespace distinct_differences_count_l305_305436

-- Define the set of interest.
def mySet : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- The statement we want to prove.
theorem distinct_differences_count : 
  (finset.image (λ (x : ℕ × ℕ), (x.1 - x.2)) ((mySet.product mySet).filter (λ x, x.1 > x.2))).card = 7 :=
sorry

end distinct_differences_count_l305_305436


namespace trig_identity_example_l305_305753

theorem trig_identity_example :
  sin (43 * real.pi / 180) * sin (17 * real.pi / 180) - cos (43 * real.pi / 180) * cos (17 * real.pi / 180) = -1 / 2 :=
by
  sorry

end trig_identity_example_l305_305753


namespace arrangements_count_l305_305986

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of positions
def num_positions : ℕ := 3

-- Define a type for the students
inductive Student
| A | B | C | D | E

-- Define the positions
inductive Position
| athletics | swimming | ball_games

-- Constraint: student A cannot be the swimming volunteer
def cannot_be_swimming_volunteer (s : Student) (p : Position) : Prop :=
  (s = Student.A → p ≠ Position.swimming)

-- Define the function to count the arrangements given the constraints
noncomputable def count_arrangements : ℕ :=
  (num_students.choose num_positions) - 1 -- Placeholder for the actual count based on given conditions

-- The theorem statement
theorem arrangements_count : count_arrangements = 16 :=
by
  sorry

end arrangements_count_l305_305986


namespace simplify_abs_expression_l305_305392

theorem simplify_abs_expression
  (a b : ℝ)
  (h1 : a < 0)
  (h2 : a * b < 0)
  : |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end simplify_abs_expression_l305_305392


namespace problem_proof_l305_305047

variable {α : Type*} [LinearOrder α] [Field α] [DecidableEq α]

-- Assume: Arithmetic sequence {a_n} and S_n is the sum of its first n terms
variable (a : ℕ → α) (S : ℕ → α)
variable (k : ℕ)

-- Conditions
axiom arithmetic_sequence (a : ℕ → α) : ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_of_arithmetic_sequence (S : ℕ → α) (a : ℕ → α) : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given
axiom given (S : ℕ → α) (k : ℕ) : S (2 * k + 1) > 0

-- To prove: a_{k+1} > 0
theorem problem_proof (a : ℕ → α) (S : ℕ → α) (k : ℕ)
  [arithmetic_sequence a]
  [sum_of_arithmetic_sequence S a]
  [given S k] :
  a (k + 1) > 0 :=
sorry

end problem_proof_l305_305047


namespace unique_toy_value_l305_305299

/-- Allie has 9 toys in total. The total worth of these toys is $52. 
One toy has a certain value "x" dollars and the remaining 8 toys each have a value of $5. 
Prove that the value of the unique toy is $12. -/
theorem unique_toy_value (x : ℕ) (h1 : 1 + 8 = 9) (h2 : x + 8 * 5 = 52) : x = 12 :=
by
  sorry

end unique_toy_value_l305_305299


namespace smaller_cubes_count_l305_305708

theorem smaller_cubes_count (painted_faces: ℕ) (edge_cubes: ℕ) (total_cubes: ℕ) : 
  (painted_faces = 2 ∧ edge_cubes = 12) → total_cubes = 27 :=
by
  assume h : (painted_faces = 2 ∧ edge_cubes = 12)
  sorry

end smaller_cubes_count_l305_305708


namespace hypotenuse_length_l305_305498

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305498


namespace distinct_positive_differences_l305_305430

theorem distinct_positive_differences :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  ∑ i in s, ∑ j in s, if i ≠ j then (Nat.abs (i - j) : Finset ℕ) else 0 = {1, 2, 3, 4, 5, 6, 7} :=
sorry

end distinct_positive_differences_l305_305430


namespace quadratic_inequality_problems_l305_305042

theorem quadratic_inequality_problems
  {a b c : ℝ}
  (h₁ : ∀ x : ℝ, ax^2 - bx + c < 0 ↔ x < -2 ∨ x > 3)
  (h₂ : b = a)
  (h₃ : c = -6 * a) :
  (a + 5 * b + c = 0) ∧
  (∀ x : ℝ, bx^2 - ax + c > 0 ↔ -2 < x ∧ x < 3) ∧
  (c > 0) ∧
  (¬ ∃ x : ℝ, cx^2 + ax - b < 0) :=
begin
  sorry
end

end quadratic_inequality_problems_l305_305042


namespace nora_must_sell_5_cases_l305_305938

-- Definitions based on given conditions
def packs_per_case : ℕ := 3
def muffins_per_pack : ℕ := 4
def price_per_muffin : ℕ := 2
def total_goal : ℕ := 120
def total_per_case := packs_per_case * muffins_per_pack * price_per_muffin  -- This calculates the earnings from one case

-- The problem statement as a Lean theorem
theorem nora_must_sell_5_cases : total_goal / total_per_case = 5 := by
  -- Providing the necessary preliminary calculations
  have packs_calc : packs_per_case = 3 := rfl
  have muffins_calc : muffins_per_pack = 4 := rfl
  have price_calc : price_per_muffin = 2 := rfl
  have goal_calc : total_goal = 120 := rfl
  have case_calc : total_per_case = 24 := by unfold total_per_case; simp [packs_calc, muffins_calc, price_calc, mul_assoc]
  calc
    total_goal / total_per_case = 120 / 24 : by congr; exact goal_calc; exact case_calc
    ... = 5 : by norm_num

end nora_must_sell_5_cases_l305_305938


namespace find_number_of_toonies_l305_305307

variable (L T : ℕ)

def condition1 : Prop := L + T = 10
def condition2 : Prop := L + 2 * T = 14

theorem find_number_of_toonies (h1 : condition1 L T) (h2 : condition2 L T) : T = 4 :=
by
  sorry

end find_number_of_toonies_l305_305307


namespace rectangular_to_cylindrical_l305_305759

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h1 : x = -3) (h2 : y = 4) (h3 : z = 5) (h4 : r = 5) (h5 : θ = Real.pi - Real.arctan (4 / 3)) :
  (r, θ, z) = (5, Real.pi - Real.arctan (4 / 3), 5) :=
by
  sorry

end rectangular_to_cylindrical_l305_305759


namespace smaller_cubes_total_l305_305705

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l305_305705


namespace largest_odd_same_cost_l305_305640

/-- Define the decimal cost transmission for a given integer. -/
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the binary cost transmission with the binary representation ending in at least two zeros. -/
def binary_cost (n : ℕ) : ℕ :=
  (n.shift_left 2).popcount

/-- The main proof problem as described. -/
theorem largest_odd_same_cost :
  ∃ n : ℕ, n < 2000 ∧ n % 2 = 1 ∧ decimal_cost n = binary_cost n ∧
  ∀ m : ℕ, m < 2000 ∧ m % 2 = 1 ∧ decimal_cost m = binary_cost m → m ≤ 1999 := by
  sorry

end largest_odd_same_cost_l305_305640


namespace correct_option_l305_305666

-- Definitions of the options as Lean statements
def optionA : Prop := (-1 : ℝ) / 6 > (-1 : ℝ) / 7
def optionB : Prop := (-4 : ℝ) / 3 < (-3 : ℝ) / 2
def optionC : Prop := (-2 : ℝ)^3 = -2^3
def optionD : Prop := -(-4.5 : ℝ) > abs (-4.6 : ℝ)

-- Theorem stating that optionC is the correct statement among the provided options
theorem correct_option : optionC :=
by
  unfold optionC
  rw [neg_pow, neg_pow, pow_succ, pow_succ]
  sorry  -- The proof is omitted as per instructions

end correct_option_l305_305666


namespace solve_inequalities_l305_305334

theorem solve_inequalities (x : ℝ) (h₁ : 5 * x - 8 > 12 - 2 * x) (h₂ : |x - 1| ≤ 3) : 
  (20 / 7) < x ∧ x ≤ 4 :=
by
  sorry

end solve_inequalities_l305_305334


namespace fraction_of_pizza_peter_ate_l305_305941

theorem fraction_of_pizza_peter_ate (total_slices : ℕ) (peter_slices : ℕ) (shared_slices : ℚ) 
  (pizza_fraction : ℚ) : 
  total_slices = 16 → 
  peter_slices = 2 → 
  shared_slices = 1/3 → 
  pizza_fraction = peter_slices / total_slices + (1 / 2) * shared_slices / total_slices → 
  pizza_fraction = 13 / 96 :=
by 
  intros h1 h2 h3 h4
  -- to be proved later
  sorry

end fraction_of_pizza_peter_ate_l305_305941


namespace intersection_nonempty_implies_m_eq_zero_l305_305076

theorem intersection_nonempty_implies_m_eq_zero (m : ℤ) (P Q : Set ℝ)
  (hP : P = { -1, ↑m } ) (hQ : Q = { x : ℝ | -1 < x ∧ x < 3/4 }) (h : (P ∩ Q).Nonempty) :
  m = 0 :=
by
  sorry

end intersection_nonempty_implies_m_eq_zero_l305_305076


namespace median_unchanged_when_removing_extremes_l305_305889

theorem median_unchanged_when_removing_extremes (scores : List ℝ) (h_len : scores.length = 7) :
  let scores_sorted := List.quicksort scores in
  let middle_original := scores_sorted.nth_le 3 sorry in
  let scores_filtered := scores_sorted.drop 1 |>.take 5 in
  let middle_filtered := scores_filtered.nth_le 2 sorry in
  middle_original = middle_filtered := sorry

end median_unchanged_when_removing_extremes_l305_305889


namespace range_of_x_l305_305405

noncomputable def f (x : ℝ) : ℝ := Real.exp(x - 1) + Real.exp(1 - x)

theorem range_of_x (x : ℝ) : 1 < x ∧ x < 3 ↔ f (x - 1) < Real.exp(1) + Real.exp(-1) :=
by 
  sorry

end range_of_x_l305_305405


namespace ordered_pairs_condition_l305_305394

theorem ordered_pairs_condition (m n : ℕ) (hmn : m ≥ n) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 3 * m * n = 8 * (m + n - 1)) :
    (m, n) = (16, 3) ∨ (m, n) = (6, 4) := by
  sorry

end ordered_pairs_condition_l305_305394


namespace angle_bisector_of_aed_l305_305163

variable {α : Type*}
variable [InnerProductSpace ℝ α]

-- Define quadrilateral and points
variables (A B C D O E : α)

-- Define the conditions of the problem
variable (h1 : ConvexQuadrilateral A B C D)
variable (h2 : O ∈ Segment ℝ A D)
variable (h3 : dist A O = dist B O)
variable (h4 : dist C O = dist D O)
variable (h5 : ∠. O A B = ∠. O C D)
variable (h6 : E = intersection (line A C) (line B D))

-- State the theorem
theorem angle_bisector_of_aed (h1 : ConvexQuadrilateral A B C D)
    (h2 : O ∈ Segment ℝ A D) (h3 : dist A O = dist B O) 
    (h4 : dist C O = dist D O) (h5 : ∠. O A B = ∠. O C D) 
    (h6 : E = intersection (line A C) (line B D)) : 
  is_angle_bisector E O (∠. A E D) :=
sorry

end angle_bisector_of_aed_l305_305163


namespace trisha_money_l305_305645

theorem trisha_money (money_meat money_chicken money_veggies money_eggs money_dogfood money_left : ℤ)
  (h_meat : money_meat = 17)
  (h_chicken : money_chicken = 22)
  (h_veggies : money_veggies = 43)
  (h_eggs : money_eggs = 5)
  (h_dogfood : money_dogfood = 45)
  (h_left : money_left = 35) :
  let total_spent := money_meat + money_chicken + money_veggies + money_eggs + money_dogfood
  in total_spent + money_left = 167 :=
by
  sorry

end trisha_money_l305_305645


namespace log_eq_solution_l305_305218

theorem log_eq_solution (x : ℝ) 
  (h1 : x^2 - 3 = 3x - 5) 
  (h2 : x^2 - 3 > 0) 
  (h3 : 3x - 5 > 0) : 
  x = 2 :=
sorry

end log_eq_solution_l305_305218


namespace one_third_of_nine_times_seven_l305_305776

theorem one_third_of_nine_times_seven : (1 / 3) * (9 * 7) = 21 := 
by
  sorry

end one_third_of_nine_times_seven_l305_305776


namespace no_integer_solution_system_l305_305945

theorem no_integer_solution_system (
  x y z : ℤ
) : x^6 + x^3 + x^3 * y + y ≠ 147 ^ 137 ∨ x^3 + x^3 * y + y^2 + y + z^9 ≠ 157 ^ 117 :=
by
  sorry

end no_integer_solution_system_l305_305945


namespace rachel_earnings_one_hour_l305_305951

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end rachel_earnings_one_hour_l305_305951


namespace puppy_total_food_l305_305921

def daily_food_first_two_weeks : ℝ := (1 / 4) * 3
def total_food_first_two_weeks : ℝ := daily_food_first_two_weeks * 14

def daily_food_second_two_weeks : ℝ := (1 / 2) * 2
def total_food_second_two_weeks : ℝ := daily_food_second_two_weeks * 14

def food_today : ℝ := 1 / 2

def total_food_in_4_weeks : ℝ := food_today + total_food_first_two_weeks + total_food_second_two_weeks

theorem puppy_total_food (W: ℝ:= 25) : total_food_in_4_weeks = W :=
by 
    sorry

end puppy_total_food_l305_305921


namespace hypotenuse_length_l305_305487

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l305_305487


namespace f_ge_g_l305_305818

variable (a b c : ℝ)
variable (α : ℝ)

def f (a b c α : ℝ) : ℝ :=
  a * b * c * (a^α + b^α + c^α)

def g (a b c α : ℝ) : ℝ :=
  a^(α+2) * (b + c - a) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c)

theorem f_ge_g (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : f a b c α ≥ g a b c α :=
by
  sorry

end f_ge_g_l305_305818


namespace alice_probability_p_q_sum_l305_305298

def alice_reaches_minus_eight : ℚ :=
  let total_moves := 10
  let positive_moves := 1 / 3
  let no_moves := 1 / 3
  let negative_moves := 1 / 3
  let total_outcomes := 3 ^ total_moves
  let ways_to_reach_neg_eight := (total_moves.fact / ((2).fact * (3).fact * (5).fact))
  (ways_to_reach_neg_eight / total_outcomes)

theorem alice_probability_p_q_sum : alice_reaches_minus_eight = 2520 / 59049 ∧ 2520 + 59049 = 61569 := 
by
  sorry

end alice_probability_p_q_sum_l305_305298


namespace nora_must_sell_5_cases_l305_305939

-- Definitions based on given conditions
def packs_per_case : ℕ := 3
def muffins_per_pack : ℕ := 4
def price_per_muffin : ℕ := 2
def total_goal : ℕ := 120
def total_per_case := packs_per_case * muffins_per_pack * price_per_muffin  -- This calculates the earnings from one case

-- The problem statement as a Lean theorem
theorem nora_must_sell_5_cases : total_goal / total_per_case = 5 := by
  -- Providing the necessary preliminary calculations
  have packs_calc : packs_per_case = 3 := rfl
  have muffins_calc : muffins_per_pack = 4 := rfl
  have price_calc : price_per_muffin = 2 := rfl
  have goal_calc : total_goal = 120 := rfl
  have case_calc : total_per_case = 24 := by unfold total_per_case; simp [packs_calc, muffins_calc, price_calc, mul_assoc]
  calc
    total_goal / total_per_case = 120 / 24 : by congr; exact goal_calc; exact case_calc
    ... = 5 : by norm_num

end nora_must_sell_5_cases_l305_305939


namespace cuboid_formation_and_end_points_l305_305756

namespace BeadCuboid

variables (p q r : ℕ)

def is_cube_formable (p q r : ℕ) : Prop := 
  ∃ config : list (ℕ × ℕ × ℕ), 
    list.foldr (λ (dim : ℕ × ℕ × ℕ) acc, (dim.1 * dim.2 * dim.3) + acc) 0 config = p * q * r

def end_points_meet (p q r : ℕ) : Prop :=
  (p % 2 = 0 ∧ q % 2 = 0) ∨ (p % 2 = 0 ∧ r % 2 = 0) ∨ (q % 2 = 0 ∧ r % 2 = 0)

theorem cuboid_formation_and_end_points (p q r : ℕ) : 
  (is_cube_formable p q r) ∧ (end_points_meet p q r ↔ (p % 2 = 0 ∧ q % 2 = 0 ∨ q % 2 = 0 ∧ r % 2 = 0 ∨ p % 2 = 0 ∧ r % 2 = 0)) :=
by sorry

end BeadCuboid

end cuboid_formation_and_end_points_l305_305756


namespace card_draw_probability_l305_305638

theorem card_draw_probability : 
  let P1 := (12 / 52 : ℚ) * (4 / 51 : ℚ) * (13 / 50 : ℚ)
  let P2 := (1 / 52 : ℚ) * (3 / 51 : ℚ) * (13 / 50 : ℚ)
  P1 + P2 = (63 / 107800 : ℚ) :=
by
  sorry

end card_draw_probability_l305_305638


namespace sequence_and_sum_result_l305_305044

noncomputable def an_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a (n + 1))^2 - a (n + 1) * a n - 2 * (a n)^2 = 0

noncomputable def arithmetic_mean_condition (a : ℕ → ℝ) : Prop :=
  a 3 + 2 = (a 2 + a 4) / 2

noncomputable def general_term (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2^n

noncomputable def bn_sequence (b a : ℕ → ℝ) : Prop :=
  ∀ n, b n = a n * Real.log 2 (a n)^(-1)

noncomputable def Sn_sum (S_n b : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = ∑ i in Finset.range n, b i

noncomputable def Sn_formula (S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (1 - n) * 2^(n + 1) + 2

theorem sequence_and_sum_result (a b S_n : ℕ → ℝ) :
  an_sequence a →
  arithmetic_mean_condition a →
  general_term a →
  bn_sequence b a →
  Sn_sum S_n b →
  Sn_formula S_n :=
by
  intro h1 h2 h3 h4 h5
  sorry

end sequence_and_sum_result_l305_305044


namespace vincent_total_packs_l305_305674

-- Definitions based on the conditions
def packs_yesterday : ℕ := 15
def extra_packs_today : ℕ := 10

-- Total packs calculation
def packs_today : ℕ := packs_yesterday + extra_packs_today
def total_packs : ℕ := packs_yesterday + packs_today

-- Proof statement
theorem vincent_total_packs : total_packs = 40 :=
by
  -- Calculate today’s packs
  have h1 : packs_today = 25 := by
    rw [packs_yesterday, extra_packs_today]
    norm_num
  
  -- Calculate the total packs
  have h2 : total_packs = 15 + 25 := by
    rw [packs_yesterday, h1]
  
  -- Conclude the total number of packs
  show total_packs = 40
  rw [h2]
  norm_num

end vincent_total_packs_l305_305674


namespace range_of_a_l305_305059

noncomputable def common_point_ellipse_parabola (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y

theorem range_of_a : ∀ a : ℝ, common_point_ellipse_parabola a → -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l305_305059


namespace unique_function_satisfying_conditions_l305_305541

open Nat

theorem unique_function_satisfying_conditions (k : ℕ) (h : 0 < k) : 
  ∀ (f : ℕ → ℕ),
  (∃ᶠ p in at_top, ∃ c : ℕ, f(c) = p ^ k) ∧ 
  (∀ m n : ℕ, f(m) + f(n) ∣ f(m + n)) ->
  (∀ n : ℕ, f(n) = n) :=
by
  sorry

end unique_function_satisfying_conditions_l305_305541


namespace find_ax5_by5_l305_305147

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 :=
sorry

end find_ax5_by5_l305_305147


namespace parallel_lines_condition_l305_305833

theorem parallel_lines_condition (k_1 k_2 : ℝ) :
  (k_1 = k_2) ↔ (∀ x y : ℝ, k_1 * x + y + 1 = 0 → k_2 * x + y - 1 = 0) :=
sorry

end parallel_lines_condition_l305_305833


namespace class_representation_l305_305443

theorem class_representation :
  (∀ (x y : ℕ), (x, y) = (7, 8) → (y = 8 ∧ x = 7)) →
  (∃ (x y : ℕ), (x, y) = (8, 6) ∧ (y = 6 ∧ x = 8)) :=
by
  intro h
  exists (8, 6)
  split
  sorry

end class_representation_l305_305443


namespace smaller_cubes_total_l305_305707

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l305_305707


namespace maximum_ab_minimum_frac_minimum_exp_l305_305021

variable {a b : ℝ}

theorem maximum_ab (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  ab <= 1/8 :=
sorry

theorem minimum_frac (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2/a + 1/b >= 8 :=
sorry

theorem minimum_exp (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2^a + 4^b >= 2 * Real.sqrt 2 :=
sorry

end maximum_ab_minimum_frac_minimum_exp_l305_305021


namespace find_number_l305_305774

theorem find_number (x : ℕ) (h : x - 18 = 3 * (86 - x)) : x = 69 :=
by
  sorry

end find_number_l305_305774


namespace range_of_x_l305_305828

noncomputable def f (x : ℝ) : ℝ := log ((|x| + 1) : ℝ) / log (1/2) + 1 / (x^2 + 1)

theorem range_of_x (x : ℝ) : f x > f (2*x - 1) ↔ x > 1 ∨ x < 1 / 3 := 
sorry

end range_of_x_l305_305828


namespace problem_statement_l305_305975

noncomputable def f (x φ : ℝ) : ℝ := -2 * sin^2 (x - φ / 2) + 3 / 2
noncomputable def g (x φ : ℝ) : ℝ := f (x + π / 6) φ

theorem problem_statement (φ : ℝ) (hφ : abs φ < π / 2) :
  (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (π / 4 + k * π / 2, 1 / 2) ∧ g c.1 φ = g (-c.1) φ)
  ∧ (∀ x : ℝ, 0 ≤ x → x ≤ π / 3 → 1 ≤ f x φ ∧ f x φ ≤ 3 / 2) := by
  sorry

end problem_statement_l305_305975


namespace sum_of_squares_of_roots_l305_305319

theorem sum_of_squares_of_roots (s1 s2 : ℝ) (h1 : s1 * s2 = 4) (h2 : s1 + s2 = 16) : s1^2 + s2^2 = 248 :=
by
  sorry

end sum_of_squares_of_roots_l305_305319


namespace increasing_intervals_of_piecewise_function_l305_305622

def piecewise_function (x : ℝ) : ℝ :=
if x ≤ 0 then 2 * x + 3
else if x ≤ 1 then x + 3
else -x + 5

theorem increasing_intervals_of_piecewise_function :
  ∀ x : ℝ, (x ≤ 1 → (if x ≤ 0 then 2 else if x ≤ 1 then 1 else -1) > 0 ) ↔ x ≤ 1 :=
begin
  sorry
end

end increasing_intervals_of_piecewise_function_l305_305622


namespace puppy_total_food_l305_305922

def daily_food_first_two_weeks : ℝ := (1 / 4) * 3
def total_food_first_two_weeks : ℝ := daily_food_first_two_weeks * 14

def daily_food_second_two_weeks : ℝ := (1 / 2) * 2
def total_food_second_two_weeks : ℝ := daily_food_second_two_weeks * 14

def food_today : ℝ := 1 / 2

def total_food_in_4_weeks : ℝ := food_today + total_food_first_two_weeks + total_food_second_two_weeks

theorem puppy_total_food (W: ℝ:= 25) : total_food_in_4_weeks = W :=
by 
    sorry

end puppy_total_food_l305_305922


namespace quadratic_intersection_points_l305_305625

theorem quadratic_intersection_points : 
  let f := fun x => x^2 - 2 * x + 1
  in (∃ x, f x = 0 ∧ ∃ y, ∀ x, f x = y → x = 0) →
     (let count_intersections := 1 + 1 in count_intersections = 2) :=
begin
  sorry
end

end quadratic_intersection_points_l305_305625


namespace intersection_M_N_l305_305077

noncomputable def M : set ℤ := {-1, 1}

noncomputable def N : set ℤ := {x | (1/2 : ℝ) < 2^(x + 1) ∧ 2^(x + 1) < 4}

theorem intersection_M_N :
  M ∩ N = {-1} :=
by {
  sorry
}

end intersection_M_N_l305_305077


namespace polygon_area_is_odd_l305_305798

theorem polygon_area_is_odd
  (vertices : Finₓ 100 → ℤ × ℤ)
  (parallel_axes : ∀ i j : Finₓ 100, (vertices i).1 = (vertices j).1 ∨ (vertices i).2 = (vertices j).2)
  (odd_side_lengths : ∀ i : Finₓ 99, ((vertices (i + 1)).1 - (vertices i).1) % 2 = 1 ∨ ((vertices (i + 1)).2 - (vertices i).2) % 2 = 1)
  (odd_side_length_last : ((vertices 0).1 - (vertices 99)).1 % 2 = 1 ∨ ((vertices 0).2 - (vertices 99)).2 % 2 = 1) :
  ∃ area : ℤ, area % 2 = 1 := 
sorry

end polygon_area_is_odd_l305_305798


namespace balls_into_boxes_l305_305844

theorem balls_into_boxes :
  (finset.univ.sum (λ x : (finset (finset ℕ)), (if x.sum id = 5 then (finset.univ.prod (λ y : x.to_list (ℕ → ℕ), y.2)) else 0))) = 56 :=
begin
  sorry
end

end balls_into_boxes_l305_305844


namespace simplify_and_evaluate_expr_l305_305185

noncomputable def original_expr (x : ℝ) : ℝ := 
  ((x / (x - 1)) - (x / (x^2 - 1))) / ((x^2 - x) / (x^2 - 2*x + 1))

noncomputable def x_val : ℝ := Real.sqrt 2 - 1

theorem simplify_and_evaluate_expr : original_expr x_val = 1 - (Real.sqrt 2) / 2 :=
  by
    sorry

end simplify_and_evaluate_expr_l305_305185


namespace interval_frequency_l305_305293

-- Defining the given conditions
def sample_capacity : ℕ := 20
def freq_1020 : ℕ := 2
def freq_2030 : ℕ := 3
def freq_3040 : ℕ := 4
def freq_4050 : ℕ := 5
def freq_5060 : ℕ := 4
def freq_6070 : ℕ := 2

-- Required proof statement
theorem interval_frequency : 
  (freq_1020 + freq_2030 + freq_3040 + freq_4050) / sample_capacity = 0.7 :=
by {
  sorry
}

end interval_frequency_l305_305293


namespace vector_computation_l305_305749

variables
  (a1 b1 : ℝ) (a2 b2 : ℝ) (a3 b3 : ℝ)
  (u1 u2 u3 : ℝ) (v1 v2 v3 : ℝ) (w1 w2 w3 : ℝ)

def vector4_3_2_5_3_2_neg_3_4 : Prop :=
  let vec1 := (4 • (3 : ℝ), 4 • (-2 : ℝ), 4 • (5 : ℝ)) in
  let vec2 := (3 • (2 : ℝ), 3 • (-3 : ℝ), 3 • (4 : ℝ)) in
  let result := (vec1.1 - vec2.1, vec1.2 - vec2.2, vec1.3 - vec2.3) in
  result = (6, 1, 8)

theorem vector_computation : vector4_3_2_5_3_2_neg_3_4 :=
sorry

end vector_computation_l305_305749


namespace symmetric_point_correct_l305_305008

-- Define the point P in a three-dimensional Cartesian coordinate system.
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the function to find the symmetric point with respect to the x-axis.
def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point P(1, -2, 3).
def P : Point3D := { x := 1, y := -2, z := 3 }

-- The expected symmetric point
def symmetricP : Point3D := { x := 1, y := 2, z := -3 }

-- The proposition we need to prove
theorem symmetric_point_correct :
  symmetricWithRespectToXAxis P = symmetricP :=
by
  sorry

end symmetric_point_correct_l305_305008


namespace hyperbola_eccentricity_l305_305206

theorem hyperbola_eccentricity 
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)
  (h4 : distance F1 F2 = 2 * c)
  (h5 : ∃ A B, bisects A F1 B ∧ angle_line_through F1 30 A B) :
  eccentricity = sqrt(3) :=
by
  sorry

end hyperbola_eccentricity_l305_305206


namespace sum_of_remainders_mod_13_l305_305254

theorem sum_of_remainders_mod_13 :
  ∀ (a b c d e : ℤ),
    a ≡ 3 [ZMOD 13] →
    b ≡ 5 [ZMOD 13] →
    c ≡ 7 [ZMOD 13] →
    d ≡ 9 [ZMOD 13] →
    e ≡ 11 [ZMOD 13] →
    (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_of_remainders_mod_13_l305_305254


namespace complex_proof_l305_305103

def complex (z : ℂ) : Prop :=
  ∀ (f : ℂ → ℂ), (∀ z : ℂ, f(1 - z) = 2 * z - complex.I) →
  (1 + complex.I) * f(1 - complex.I) = -1 + complex.I

theorem complex_proof : complex_z : ℂ, complex complex_z := by
  sorry

end complex_proof_l305_305103


namespace find_x_squared_plus_y_squared_l305_305857

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l305_305857


namespace jerome_money_left_l305_305836

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end jerome_money_left_l305_305836


namespace problem_part_1_problem_part_2_l305_305409

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := Real.log ((x + 2) / (x - 2))

theorem problem_part_1 :
  ∀ (x₁ x₂ : ℝ), 0 < x₂ ∧ x₂ < x₁ → Real.log x₁ + 2 * x₁ > Real.log x₂ + 2 * x₂ :=
sorry

theorem problem_part_2 :
  ∃ k : ℕ, ∀ (x₁ : ℝ), 0 < x₁ ∧ x₁ < 1 → (∃ (x₂ : ℝ), x₂ ∈ Set.Ioo (k : ℝ) (k + 1) ∧ Real.log x₁ + 2 * x₁ < Real.log ((x₂ + 2) / (x₂ - 2))) → k = 2 :=
sorry

end problem_part_1_problem_part_2_l305_305409


namespace parabola_line_chord_length_l305_305414

theorem parabola_line_chord_length:
  ∀ (x y: ℝ),
  let C := λ x y: ℝ, y^2 = 8 * x,
      F := (2, 0),
      l := λ x y: ℝ, y = sqrt(3) * (x - 2),
      chord_length := 
          let x1 := (4 + 2 * sqrt(2)) / 3,
              x2 := (4 - 2 * sqrt(2)) / 3 in (x1 + x2 + 2) in
  (∀ p ∈ C, ∃ m, ∃ b, l x y = m * x + b → p = F → 
      (sqrt(3) * x - y - 2 * sqrt(3) = 0 ∧ 
      chord_length = 32 / 3)) 
:= 
sorry

end parabola_line_chord_length_l305_305414


namespace find_x_l305_305014

theorem find_x (x : ℝ) (h : sqrt (x + 13) = 11) : x = 108 :=
by
  sorry

end find_x_l305_305014


namespace vincent_total_packs_l305_305675

-- Definitions based on the conditions
def packs_yesterday : ℕ := 15
def extra_packs_today : ℕ := 10

-- Total packs calculation
def packs_today : ℕ := packs_yesterday + extra_packs_today
def total_packs : ℕ := packs_yesterday + packs_today

-- Proof statement
theorem vincent_total_packs : total_packs = 40 :=
by
  -- Calculate today’s packs
  have h1 : packs_today = 25 := by
    rw [packs_yesterday, extra_packs_today]
    norm_num
  
  -- Calculate the total packs
  have h2 : total_packs = 15 + 25 := by
    rw [packs_yesterday, h1]
  
  -- Conclude the total number of packs
  show total_packs = 40
  rw [h2]
  norm_num

end vincent_total_packs_l305_305675


namespace chord_equation_and_area_l305_305732

open Real

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 18 * x

noncomputable def chord (x : ℝ) : ℝ := 3 * x - 9

theorem chord_equation_and_area :
  let x1 := 4
      y1 := 3
      x2 := 4 + sqrt 7
      y2 := 3 + 3 * sqrt 7
      x3 := 4 - sqrt 7
      y3 := 3 - 3 * sqrt 7
      AB := 2 * sqrt 70 in
  parabola x1 y1 ∧
  chord x1 = y1 ∧
  (parabola x2 y2 ∧ chord x2 = y2) ∧
  (parabola x3 y3 ∧ chord x3 = y3) ∧
  (sqrt ((x2 - x3)^2 + (y2 - y3)^2) = AB) ∧
  (14 * sqrt 7 = 14 * sqrt 7) :=
by
  intros
  rw [parabola, chord]
  sorry

end chord_equation_and_area_l305_305732


namespace distinct_positive_differences_l305_305427

theorem distinct_positive_differences :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  ∑ i in s, ∑ j in s, if i ≠ j then (Nat.abs (i - j) : Finset ℕ) else 0 = {1, 2, 3, 4, 5, 6, 7} :=
sorry

end distinct_positive_differences_l305_305427


namespace probability_at_least_four_girls_l305_305137

noncomputable def binomial_probability_at_least : ℕ → ℚ → ℚ
| n, p => ∑ k in finset.range (n + 1), if k ≥ 4 then (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) else 0

theorem probability_at_least_four_girls :
  binomial_probability_at_least 7 (3/5 : ℚ) = 0.452245 :=
sorry

end probability_at_least_four_girls_l305_305137


namespace angles_equal_of_rectangle_and_midpoints_l305_305557

-- Definitions based on the conditions
variables {A B C D E F G : Type*}
variable [rect : Rectangle A B C D]
variables [Midpoint E A D] [Midpoint F D C]
variable (G : Point) (h1 : Incident G (Line A F)) (h2 : Incident G (Line E C))

-- The statement to prove
theorem angles_equal_of_rectangle_and_midpoints :
  ∠C G F = ∠F B E := sorry

end angles_equal_of_rectangle_and_midpoints_l305_305557


namespace range_xf_ge_0_l305_305105

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 2 else - (-x) - 2

theorem range_xf_ge_0 :
  { x : ℝ | x * f x ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_xf_ge_0_l305_305105


namespace jasper_drinks_more_than_hot_dogs_l305_305138

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end jasper_drinks_more_than_hot_dogs_l305_305138


namespace area_triangle_ABM_range_l305_305806

noncomputable theory

def circle_eq (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1

def symmetric (M N : ℝ × ℝ) : Prop := M.1 = -N.1 ∧ M.2 = -N.2

def line_through (N : ℝ × ℝ) (k : ℝ) : ℝ × ℝ → Prop := λ P, P.2 = k * P.1 + N.2

def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def line_intersects_circle (N : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ A B, line_through N k A ∧ line_through N k B ∧ circle_eq A ∧ circle_eq B

def area_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * real.abs ((A.1 - M.1) * (B.2 - M.2) - (B.1 - M.1) * (A.2 - M.2))

def area_range (A B M : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ S, area_triangle A B M = S ∧ 0 < S ∧ S ≤ 2 * real.sqrt 2 / 3

theorem area_triangle_ABM_range : 
  ∀ (A B M : ℝ × ℝ) (k : ℝ),
  circle_eq A → circle_eq B → line_through (0, -real.sqrt 3 / 3) k A → line_through (0, -real.sqrt 3 / 3) k B →
  M = (0, real.sqrt 3 / 3) →
  line_intersects_circle (0, -real.sqrt 3 / 3) k →
  area_range A B M k := 
sorry

end area_triangle_ABM_range_l305_305806


namespace lcm_of_54_and_198_l305_305778

theorem lcm_of_54_and_198 : Nat.lcm 54 198 = 594 :=
by
  have fact1 : 54 = 2 ^ 1 * 3 ^ 3 := by norm_num
  have fact2 : 198 = 2 ^ 1 * 3 ^ 2 * 11 ^ 1 := by norm_num
  have lcm_prime : Nat.lcm 54 198 = 594 := by
    sorry -- Proof skipped
  exact lcm_prime

end lcm_of_54_and_198_l305_305778


namespace puppy_food_total_correct_l305_305928

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l305_305928


namespace frood_minimum_l305_305520

theorem frood_minimum (n : ℕ) (h : n = 10) : 
  (n * (n + 1) / 2) > (5 * n) := by
  rw h
  sorry

end frood_minimum_l305_305520


namespace infinitely_many_n_divide_2n_plus_1_l305_305956

theorem infinitely_many_n_divide_2n_plus_1 :
    ∃ (S : Set ℕ), (∀ n ∈ S, n > 0 ∧ n ∣ (2 * n + 1)) ∧ Set.Infinite S :=
by
  sorry

end infinitely_many_n_divide_2n_plus_1_l305_305956


namespace afternoon_emails_l305_305136

theorem afternoon_emails (A : ℕ) (five_morning_emails : ℕ) (two_more : five_morning_emails + 2 = A) : A = 7 :=
by
  sorry

end afternoon_emails_l305_305136


namespace consecutive_primes_quadratic_l305_305055

theorem consecutive_primes_quadratic:
  ∀ (p q : ℕ), prime p → prime q → q = nat.find next_prime p →
  ∃ a b : ℤ, a * b = p * q ∧ a + b = p + q ∧ (p + q).even ∧ (∀ r, r = a ∨ r = b → r ≥ p) ∧ (is_composite (p + q)) := by
sorry

end consecutive_primes_quadratic_l305_305055


namespace percent_calculation_l305_305316

theorem percent_calculation :
  let one_third_percent := 1 / 3 / 100
  let result := (one_third_percent * 200) + 50
  result = 50.6667 :=
by
  have h₁ : (1 : ℝ) / 3 / 100 = 1 / 3 / 100, from rfl
  have h₂ : (1 / 3 / 100 : ℝ) * 200 = 2 / 3, by sorry
  have h₃ : (2 / 3 : ℝ) + 50 = 50.6667, by sorry
  exact sorry

end percent_calculation_l305_305316


namespace technicians_count_l305_305510

/-- Given a workshop with 49 workers, where the average salary of all workers 
    is Rs. 8000, the average salary of the technicians is Rs. 20000, and the
    average salary of the rest is Rs. 6000, prove that the number of 
    technicians is 7. -/
theorem technicians_count (T R : ℕ) (h1 : T + R = 49) (h2 : 10 * T + 3 * R = 196) : T = 7 := 
by
  sorry

end technicians_count_l305_305510


namespace ellipse_focus_and_eccentricity_l305_305384

noncomputable theory

def focus_of_parabola (a : ℝ) : ℝ × ℝ := (a, 0)

def ellipse (x y m n : ℝ) : Prop := x^2 / m^2 + y^2 / n^2 = 1

theorem ellipse_focus_and_eccentricity 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (e : ℝ) (he : e = 1 / 2)
  (focus_ellipse : ℝ × ℝ)
  (focus_parabola : ℝ × ℝ := focus_of_parabola 2)
  (h_focus : focus_ellipse = focus_parabola) :
  ellipse 1 1 4 √3 :=
  sorry

end ellipse_focus_and_eccentricity_l305_305384


namespace arithmetic_mean_of_sequence_l305_305757

theorem arithmetic_mean_of_sequence : 
  let sequence : ℕ → ℕ := λ n, 5 + n
  let sum_sequence := (75 * (5 + 79)) / 2 
  let mean := sum_sequence / 75
  mean = 42 :=
by
  sorry

end arithmetic_mean_of_sequence_l305_305757


namespace square_area_l305_305125

noncomputable def possible_area (z : ℂ) : ℂ := complex.abs (z^2 - z + 1) ^ 2 

theorem square_area (z : ℂ) (h1 : (z^2 - z + 1) * conj (z^4 - z) = 0)
  (h2 : complex.abs (z^2 - z + 1) = complex.abs (z^4 - z))
  (h3 : z ≠ 0 ∧ z^2 + 1 ≠ 0 ∧ z^4 ≠ 0) : 
  possible_area z = 1 :=
sorry

end square_area_l305_305125


namespace dairy_protein_regression_l305_305971

open Real
open Finset

noncomputable def regression_slope (xs ys : List ℝ) (x̄ ȳ : ℝ) : ℝ :=
  let sum_prod := (List.zip xs ys).sumBy (λ p, (p.1 - x̄) * (p.2 - ȳ))
  let sum_sq := xs.sumBy (λ x, (x - x̄)^2)
  sum_prod / sum_sq

noncomputable def regression_intercept (x̄ ȳ b : ℝ) : ℝ :=
  ȳ - b * x̄

theorem dairy_protein_regression :
  ∃ (b a : ℝ),
    let x_vals := [0, 0.69, 1.39, 1.79, 2.40, 2.56, 2.94]
    let y_vals := [19, 32, 40, 44, 52, 53, 54]
    let x̄ := 1.68
    let ȳ := 42
    let b := regression_slope x_vals y_vals x̄ ȳ
    let a := regression_intercept x̄ ȳ b
    b = 11.99 ∧ a = 21.86 ∧ ∀ (y : ℝ), 60 ≤ y ∧ y ≤ 70 → ∃ (x : ℝ), 3.18 ≤ x ∧ x ≤ 4.02 ∧ y = b * x + a :=
by
  sorry

end dairy_protein_regression_l305_305971


namespace min_green_beads_l305_305719

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l305_305719


namespace order_of_a_b_c_l305_305545

noncomputable def a := (0.5 : ℝ) ^ (1 / 2)
noncomputable def b := (0.9 : ℝ) ^ (1 / 4)
noncomputable def c := Real.logBase 5 0.3

theorem order_of_a_b_c : b > a ∧ a > c :=
by 
  sorry

end order_of_a_b_c_l305_305545


namespace magnitude_a_plus_z_l305_305057

-- Given conditions
variables {z a : ℂ}
def pure_imaginary (z : ℂ) : Prop := z.re = 0
axiom condition1 : pure_imaginary z
axiom condition2 : (2 + (1:ℂ).i) * z = 1 + a * ( (1:ℂ).i ^ 3)

-- Theorem statement
theorem magnitude_a_plus_z : |a + z| = Real.sqrt 5 :=
by sorry

end magnitude_a_plus_z_l305_305057


namespace quadrilateral_is_parallelogram_l305_305234

open Function

variable {S A B C D P : Point}

/-- Prove that the quadrilateral ABCD is a parallelogram. -/
theorem quadrilateral_is_parallelogram 
  (through_vertices_parallel_opposite_lateral_edges : 
    ∃ P, (PA ∥ SC ∧ PC ∥ SA) ∧ (PB ∥ SD ∧ PD ∥ SB)) 
  : is_parallelogram A B C D :=
sorry

end quadrilateral_is_parallelogram_l305_305234


namespace perpendicular_condition_l305_305796

variables {α β : Type} [plane α] [plane β]
variables {m : line}

-- Conditions
axiom line_in_plane_m : m ∈ α
axiom distinct_planes : α ≠ β

-- Statement to prove
theorem perpendicular_condition :
  (∀ (hp : line m), (m ⊆ α) ∧ (perpendicular_to_plane m β)) →
  (perpendicular_to_plane α β) ∧ ¬ (∃ (hp : line m), (m ⊆ α) ∧ (perpendicular_to_plane α β) → (perpendicular_to_plane m β)) :=
sorry

end perpendicular_condition_l305_305796


namespace infinite_grid_fill_possible_l305_305727

theorem infinite_grid_fill_possible :
  ∃ f : ℕ × ℕ → ℕ, 
    (∀ i j, ∃ k, f(i, k) = j) ∧ 
    (∀ i j, ∃ k, f(k, i) = j) :=
sorry

end infinite_grid_fill_possible_l305_305727


namespace divisible_by_27000_l305_305305

theorem divisible_by_27000 (k : ℕ) (h₁ : k = 30) : ∃ n : ℕ, k^3 = 27000 * n :=
by {
  sorry
}

end divisible_by_27000_l305_305305


namespace problem_solution_l305_305035

theorem problem_solution (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : {d ∈ (nat.prime_divisors x).to_finset | true}.card = 3) (h3 : 31 ∈ nat.prime_divisors x) : x = 32767 :=
sorry

end problem_solution_l305_305035


namespace sum_of_digits_M_is_270_l305_305624

noncomputable def M : ℕ := (36^49 * 49^36)^(1/2 : ℝ)

theorem sum_of_digits_M_is_270 : nat.digits 10 M |>.sum = 270 :=
sorry

end sum_of_digits_M_is_270_l305_305624


namespace relay_for_life_distance_l305_305092

theorem relay_for_life_distance (pace time : ℕ) (h_pace : pace = 2) (h_time : time = 8) : pace * time = 16 := by
  simp [h_pace, h_time]
  sorry

end relay_for_life_distance_l305_305092


namespace probability_half_or_more_even_dice_l305_305790

-- Define the fair die probability and the event
def total_dice : ℕ := 4
def probability_of_even : ℚ := 3 / 6

-- Define the problem's goal to calculate the probability a of getting at least half even outcomes
def probability_at_least_half_even : ℚ := 11 / 16

theorem probability_half_or_more_even_dice (total_dice : ℕ)
    (probability_of_even : ℚ) :
    (total_dice = 4) → (probability_of_even = 3 / 6) →
    probability_at_least_half_even = 11 / 16 :=
by
  intros _ _
  -- The actual proof is omitted
  sorry

end probability_half_or_more_even_dice_l305_305790


namespace vertex_to_diagonal_distance_le_half_diagonal_l305_305170

-- Assume A, B, C, D are points in a convex quadrilateral,
-- and AC, BD are the diagonals.
variables (A B C D A1 C1 : Point)
variable [convex_quadrilateral A B C D]
variable (AC BD : ℝ)

-- Let AA1 and CC1 be the perpendicular distances from A and C to BD
variable (AA1 CC1 : ℝ)

-- Given conditions
variables (h1 : AC ≤ BD) (h2 : AA1 + CC1 ≤ AC)

-- We need to prove that the distance from one of the vertices A or C
-- to the diagonal BD does not exceed half of that diagonal.
theorem vertex_to_diagonal_distance_le_half_diagonal : AA1 ≤ BD / 2 ∨ CC1 ≤ BD / 2 :=
by
  sorry

end vertex_to_diagonal_distance_le_half_diagonal_l305_305170


namespace area_of_sector_l305_305861

theorem area_of_sector (r θ: ℝ) (h1: θ = (2 * Real.pi) / 3) (h2: r = 2) : 
  let A := (1 / 2) * θ * r^2 in A = (4 * Real.pi) / 3 :=
by
  sorry

end area_of_sector_l305_305861


namespace find_a4_b4_c4_l305_305846

theorem find_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 5) (h3 : a^3 + b^3 + c^3 = 15) : 
    a^4 + b^4 + c^4 = 35 := 
by 
  sorry

end find_a4_b4_c4_l305_305846


namespace monthly_salary_after_tax_is_9720_l305_305207

-- Definitions of the tax thresholds and rates
def threshold : ℤ := 5000
def rate1 : ℤ := 3
def rate2 : ℤ := 10
def rate3 : ℤ := 20
def rate4 : ℤ := 25

-- Tax brackets
def limit1 : ℤ := 3000
def limit2 : ℤ := 12000
def limit3 : ℤ := 25000
def limit4 : ℤ := 35000

-- Deduction for supporting elderly when the taxpayer has a sibling
def elderly_deduction : ℤ := 1000

-- Given tax payable in a month (May 2020)
def tax_payable : ℤ := 180

-- The monthly salary after tax
def after_tax_salary (salary : ℤ) : ℤ :=
  salary - tax_payable

-- Statement
theorem monthly_salary_after_tax_is_9720
  (salary : ℤ) -- Actual salary before tax
  (taxable_income : salary <= limit4)
  (deductions : salary - threshold - elderly_deduction > 0)
  (taxable_amount : (salary - threshold - elderly_deduction))
  (tax_calculation : 90 + (taxable_amount - 3000) * rate2 / 100 = tax_payable) :
  after_tax_salary salary = 9720 :=
sorry

end monthly_salary_after_tax_is_9720_l305_305207


namespace sum_of_products_le_one_half_l305_305758

-- Define the segment and the conditions for splitting
def initial_segment : Set ℝ := set.Icc 0 1

-- Define the function to calculate the product of lengths after a split
noncomputable def product_of_split (x : ℝ) (y : ℝ) : ℝ :=
 x * y

-- Define the main theorem
theorem sum_of_products_le_one_half :
  ∀ (n : ℕ) (splits : Fin n → ℝ × ℝ),
    (∀ i, 0 ≤ splits i.1 ∧ splits i.2 ≤ 1) →
    (∑ i, product_of_split (splits i).1 (splits i).2) ≤ 1 / 2 :=
by
  sorry

end sum_of_products_le_one_half_l305_305758


namespace marys_total_candy_l305_305568

/-
Given:
1. Mary has 3 times as much candy as Megan.
2. Megan has 5 pieces of candy.
3. Mary adds 10 more pieces of candy to her collection.

Show that the total number of pieces of candy Mary has is 25.
-/

theorem marys_total_candy :
  (∀ (mary_candy megan_candy : ℕ), mary_candy = 3 * megan_candy → megan_candy = 5 → mary_candy + 10 = 25) :=
begin
  intros mary_candy megan_candy h1 h2,
  rw h2 at h1,
  rw h1,
  norm_num,
end

end marys_total_candy_l305_305568


namespace median_unchanged_after_removal_l305_305887

noncomputable def median (l : List ℕ) : ℕ :=
if h : l ≠ [] then l.sort (· < ·) ![l.length / 2] else 0

def remove_extremes (l : List ℕ) : List ℕ :=
(l.erase (l.maximum' _)) .erase (l.minimum' _)

theorem median_unchanged_after_removal (scores : List ℕ) (h : scores.length = 7) :
  median (remove_extremes scores) = median scores :=
sorry

end median_unchanged_after_removal_l305_305887


namespace hypotenuse_length_l305_305472

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305472


namespace tan_double_angle_third_quadrant_l305_305816

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : α ∈ set.Ioo (π) (3 * π)) (h2 : Real.cos (α + π) = 4 / 5) :
  Real.tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_third_quadrant_l305_305816


namespace hypotenuse_length_l305_305476

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305476


namespace count_correct_statements_l305_305440

theorem count_correct_statements :
  ∀ (p1 p2 : Plane) (l1 l2 l3 : Line),
  (¬ (p1 || l1 ∧ p2 || l1) ∨ p1 = p2) ∧
  (p1 || p2 → p1 = p2) ∧
  (¬ (l1 ⊥ l2 ∧ l3 ⊥ l2) ∨ l1 = l3 ∨ l1 ∩ l3 = ∅) ∧
  (l1 ⊥ p1 ∧ l2 ⊥ p1 → l1 = l2) ∧
  (p1 ⊥ l1 ∧ p2 ⊥ l1 → p1 = p2)
  → (2 + 1 = 3) :=
by
  sorry

end count_correct_statements_l305_305440


namespace smallest_k_to_equalize_pressures_l305_305266

theorem smallest_k_to_equalize_pressures (n : ℕ) (cylinders : vector ℚ 40) :
  (∀ initial_pressures : vector ℚ 40,
    ∃ k : ℕ, k ≥ 1 ∧
    ∀ pressures_after_equalizing : vector ℚ 40,
    (∀ (subgroups : list (list ℚ)), 
      ∀ s ∈ subgroups, s.length ≤ k → 
      (∀ i, pressures_after_equalizing.nth i = 
        (∑ x in s, x) / s.length)) → 
      (∀ j, pressures_after_equalizing.nth 0 = pressures_after_equalizing.nth j)) → 
  k = 5 :=
by sorry

end smallest_k_to_equalize_pressures_l305_305266


namespace equation_of_ellipse_range_of_triangle_area_l305_305386

-- Definitions and conditions
variables (a b : ℝ)
def ellipse_equation := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom eccentricity : a^2 - b^2 = (a / 2)^2
axiom circumradius_triangle_B_A1_F1 : ∀ (R: ℝ), R = 2 * sqrt 21 / 3 → R = a * sqrt(a^2 + b^2) / b

-- Proofs to handle
theorem equation_of_ellipse : ellipse_equation 4 (2 * sqrt 3) :=
by
  sorry

theorem range_of_triangle_area : (0 : ℝ) ≤ (9 * sqrt 5 / 2 : ℝ) :=
by
  sorry

end equation_of_ellipse_range_of_triangle_area_l305_305386


namespace monotonic_decreasing_interval_l305_305982

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  {x : ℝ | x > 0} ∩ {x : ℝ | deriv f x < 0} = {x : ℝ | x > Real.exp 1} :=
by sorry

end monotonic_decreasing_interval_l305_305982


namespace number_of_functions_l305_305831

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the function f
def f : ℕ → ℕ

-- Define the functional iteration
def f_iter (n : ℕ) (x : ℕ) : ℕ :=
  if n = 0 then x
  else sorry  -- The exact mechanics of function iteration are not necessary for this statement.

-- The proof problem statement
theorem number_of_functions : 
  (∀ (x : ℕ), x ∈ A → (f_iter 0 x = x)) → 
  (∃ n : ℕ, n = 256) := sorry

end number_of_functions_l305_305831


namespace angle_A_of_triangle_eq_60_degrees_l305_305763

theorem angle_A_of_triangle_eq_60_degrees
  (ABC : Triangle)
  (A B C : Point)
  (bisector_A : Line)
  (orthocenter circumcenter : Point)
  (h1 : isTriangle ABC)
  (h2 : bisectsAngle bisector_A A B C)
  (h3 : isPerpendicular bisector_A (lineThrough orthocenter circumcenter)) :
  measureAngle A = 60 :=
sorry

end angle_A_of_triangle_eq_60_degrees_l305_305763


namespace curve_is_ellipse_with_major_axis_y_l305_305793

variable (k : ℝ)

def curve_equation (x y : ℝ) : Prop := (1 - k) * x^2 + y^2 = k^2 - 1

theorem curve_is_ellipse_with_major_axis_y (h : k < -1) :
  ∀ x y, curve_equation k x y → IsEllipseWithMajorAxisY (curve_equation k) :=
begin
  sorry
end

end curve_is_ellipse_with_major_axis_y_l305_305793


namespace ivar_total_water_needed_l305_305134

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l305_305134


namespace typeA_time_l305_305868

-- Conditions
def total_questions : ℕ := 200
def type_A_questions : ℕ := 100
def total_minutes : ℕ := 180
def typeA_time_twice_typeB : ∀ (t : ℝ), t_A = 2 * t

-- Definition of the total time spent on type A problems
def typeA_total_time (t : ℝ) : ℝ := type_A_questions * (2 * t)

theorem typeA_time : ∃ t : ℝ, typeA_total_time t = 120 :=
begin
  sorry
end

end typeA_time_l305_305868


namespace complex_number_hyperbola_l305_305700

theorem complex_number_hyperbola (z : ℂ) (h : |z + 2 * complex.I| - |z - 2 * complex.I| = 2) :
  ∃(h' : |z + 2 * complex.I| - |z - 2 * complex.I| = 2), true := 
sorry

end complex_number_hyperbola_l305_305700


namespace express_delivery_problems_l305_305935

def monthly_average_growth_rate (deliveries_march deliveries_may : ℕ) (growth_rate : ℝ) :=
  (deliveries_march:ℝ) * (1 + growth_rate)^2 = deliveries_may

def can_complete_june_task (current_staff : ℕ) (deliveries_may : ℕ)
  (growth_rate max_deliveries_per_person : ℝ) : Prop :=
  let june_deliveries := (deliveries_may:ℝ) * (1 + growth_rate)
  let total_capacity := (current_staff:ℝ) * max_deliveries_per_person * 1000
  total_capacity >= june_deliveries

def additional_staff_needed (current_staff : ℕ) (deliveries_may : ℕ)
  (growth_rate max_deliveries_per_person : ℝ) : ℕ :=
  let june_deliveries := (deliveries_may:ℝ) * (1 + growth_rate)
  let total_capacity := (current_staff:ℝ) * max_deliveries_per_person * 1000
  if total_capacity >= june_deliveries then 0
  else ⌈(june_deliveries - total_capacity) / (max_deliveries_per_person * 1000)⌉.toNat

theorem express_delivery_problems : 
  ∀ (deliveries_march deliveries_may : ℕ) (current_staff : ℕ) 
    (max_deliveries_per_person : ℝ),
    (deliveries_march = 100000) →
    (deliveries_may = 121000) →
    (current_staff = 21) →
    (max_deliveries_per_person = 0.6) →
    ∃ growth_rate : ℝ,
      monthly_average_growth_rate deliveries_march deliveries_may growth_rate ∧
      (growth_rate = 0.1) ∧
      ¬can_complete_june_task current_staff deliveries_may growth_rate max_deliveries_per_person ∧
      additional_staff_needed current_staff deliveries_may growth_rate max_deliveries_per_person = 2 := 
by 
  sorry

end express_delivery_problems_l305_305935


namespace probability_of_winning_l305_305213

theorem probability_of_winning (P_lose : ℚ) (h : P_lose = 3 / 7) : 
  let P_win := 1 - P_lose in P_win = 4 / 7 :=
by
  sorry

end probability_of_winning_l305_305213


namespace smallest_positive_angle_of_negative_angle_l305_305996

theorem smallest_positive_angle_of_negative_angle (n : ℤ) (h : n = -2002) : 
  ∃ k : ℤ, (360 * k + n) = 158 ∧ (360 * k + n) > 0 :=
begin
  -- Assume n is given as -2002.
  have hn : n = -2002 := h,
  -- We want to find an integer k such that 360 * k + n = 158 and it is positive.
  use 5,
  -- Perform the necessary calculations.
  split,
  {
    calc
      360 * 5 + n
          = 360 * 5 + -2002 : by rw hn
      ... = 1800 + -2002 : by norm_num
      ... = 158 : by norm_num,
  },
  {
    calc
      360 * 5 + n
          = 158 : by rw hn
      ... > 0 : by norm_num,
  },
  done
end

end smallest_positive_angle_of_negative_angle_l305_305996


namespace Jessie_l305_305898

theorem Jessie's_friends (total_muffins : ℕ) (muffins_per_person : ℕ) (num_people : ℕ) :
  total_muffins = 20 → muffins_per_person = 4 → num_people = total_muffins / muffins_per_person → num_people - 1 = 4 :=
by
  intros h1 h2 h3
  sorry

end Jessie_l305_305898


namespace max_odd_digits_on_board_l305_305639

open Nat

theorem max_odd_digits_on_board (a b : ℕ) (ha : 1000000000 ≤ a ∧ a < 10000000000) (hb : 1000000000 ≤ b ∧ b < 10000000000) :
  let digits := (toDigits 10 a ++ toDigits 10 b ++ toDigits 10 (a + b))
  in (digits.count (fun d => d % 2 = 1)) ≤ 30 :=
by
  sorry

end max_odd_digits_on_board_l305_305639


namespace square_area_ratio_l305_305309

theorem square_area_ratio (side_len : ℝ) :
  let AB := side_len,
      trisect_pt := AB / 3,
      area_ABCD := AB ^ 2,
      diagonals_MNPQ := (2 * (trisect_pt * real.sqrt 2)),
      area_MNPQ := diagonals_MNPQ ^ 2 in
  area_ABCD / area_MNPQ = 9 / 8 :=
by 
  sorry -- Proof is omitted

end square_area_ratio_l305_305309


namespace bounded_figure_single_center_of_symmetry_no_two_centers_of_symmetry_finite_set_max_three_almost_centers_l305_305269

-- Lean statement for problem (a)
theorem bounded_figure_single_center_of_symmetry
  (F : Set Point)
  (bounded_F : Bounded F)
  (O1 O2 : Point)
  (center_O1 : is_center_of_symmetry F O1)
  (center_O2 : is_center_of_symmetry F O2) :
  O1 = O2 := 
sorry

-- Lean statement for problem (b)
theorem no_two_centers_of_symmetry
  (F : Set Point)
  (O1 O2 : Point)
  (center_O1 : is_center_of_symmetry F O1)
  (center_O2 : is_center_of_symmetry F O2) :
  ∃ O3, O3 ≠ O1 ∧ O3 ≠ O2 ∧ is_center_of_symmetry F O3 :=
sorry

-- Lean statement for problem (c)
theorem finite_set_max_three_almost_centers
  (M : Finset Point)
  (M_finite : Set.Finite M) :
  (Card (almost_centers_of_symmetry M) ≤ 3) :=
sorry

end bounded_figure_single_center_of_symmetry_no_two_centers_of_symmetry_finite_set_max_three_almost_centers_l305_305269


namespace inequality_proof_l305_305537

variable {n : ℕ}
variable {a : Fin n → ℝ} (h_a : ∀ i j, i ≤ j → 0 ≤ a i ∧ a i ≤ a j)
variable (h_sum : ∑ i, a i = 1)
variable {x y : Fin n → ℝ} (hx : ∀ i, 0 ≤ x i)
variable (hy : ∀ i, 0 ≤ y i)

theorem inequality_proof :
  let sum_a_x := ∑ i, a i * x i
  let prod_a_x := ∏ i, x i ^ (a i)
  let sum_a_y := ∑ i, a i * y i
  let prod_a_y := ∏ i, y i ^ (a i)
  let sum_sqrt_x := ∑ i, Real.sqrt (x i)
  let sum_sqrt_y := ∑ i, Real.sqrt (y i)
  in (sum_a_x - prod_a_x) * (sum_a_y - prod_a_y) ≤
     (a (Fin.last n))^2 * 
     (n * Real.sqrt ((∑ i, x i) * (∑ i, y i)) - sum_sqrt_x * sum_sqrt_y)^2 := sorry

end inequality_proof_l305_305537


namespace smaller_cubes_count_l305_305709

theorem smaller_cubes_count (painted_faces: ℕ) (edge_cubes: ℕ) (total_cubes: ℕ) : 
  (painted_faces = 2 ∧ edge_cubes = 12) → total_cubes = 27 :=
by
  assume h : (painted_faces = 2 ∧ edge_cubes = 12)
  sorry

end smaller_cubes_count_l305_305709


namespace quadrants_I_and_II_l305_305779

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y > 3 * x
def condition2 (x y : ℝ) : Prop := y > 6 - x^2

-- Prove that any point satisfying the conditions lies in Quadrant I or II
theorem quadrants_I_and_II (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 6 - x^2) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- The proof steps are omitted
  sorry

end quadrants_I_and_II_l305_305779


namespace universal_quantifiers_and_propositions_l305_305651

-- Definitions based on conditions
def universal_quantifiers_phrases := ["for all", "for any"]
def universal_quantifier_symbol := "∀"
def universal_proposition := "Universal Proposition"
def universal_proposition_representation := "∀ x ∈ M, p(x)"

-- Main theorem
theorem universal_quantifiers_and_propositions :
  universal_quantifiers_phrases = ["for all", "for any"]
  ∧ universal_quantifier_symbol = "∀"
  ∧ universal_proposition = "Universal Proposition"
  ∧ universal_proposition_representation = "∀ x ∈ M, p(x)" :=
by
  sorry

end universal_quantifiers_and_propositions_l305_305651


namespace relationship_among_a_b_c_l305_305799

theorem relationship_among_a_b_c (m : ℝ) (f : ℝ → ℝ)
  (h_even : ∀ x, f(x) = f(-x))
  (h_def : ∀ x, f(x) = 2^(abs (x - m)) + 1) :
  let a := f (log (2) 2)
  let b := f (log (2) 4)
  let c := f (2 * m)
  in c < a ∧ a < b :=
begin
  sorry
end

end relationship_among_a_b_c_l305_305799


namespace min_green_beads_l305_305717

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l305_305717


namespace binomial_middle_term_coefficient_l305_305519

noncomputable def middle_term_coefficient (n : ℕ) : ℤ :=
  (nat.choose n (n / 2)) * ((-2) ^ (n / 2))

theorem binomial_middle_term_coefficient :
  ∀ n : ℕ, (∑ k in finset.range (n / 2 + 1), (nat.choose n (2 * k)) * (1 - 2 * 0)^ (n - 2 * k) * (-2) ^ (2 * k)) = 128 →
  n = 8 →
  middle_term_coefficient n = 1120 :=
by
  intros n h1 h2
  rw h2 at *
  simp [middle_term_coefficient, nat.choose]
  sorry

end binomial_middle_term_coefficient_l305_305519


namespace midpoints_form_square_l305_305327

section square_midpoints_of_centers

variables {A B C D E F G H F1 F2 F3 F4 : Type}

/-- Given a convex quadrilateral ABCD and squares constructed externally on each side.
    Let E, F, G, H be the centers of these squares on AB, BC, CD, and DA, respectively.
    Let F1 and F2 be the midpoints of the diagonals AC and BD of the quadrilateral.
    Let F3 and F4 be the midpoints of the line segments connecting the centers of opposite squares HF and EG, respectively.
    The quadrilateral formed by F1, F2, F3, and F4 is a square. -/
theorem midpoints_form_square
  (h_convex : convex_quadrilateral A B C D)
  (h_squares : is_square_on_sides A B C D)
  (h_centers : centers_of_squares E F G H)
  (h_midpoints : midpoints_of_segments F1 F2 F3 F4) :
  is_square F1 F2 F3 F4 :=
begin
  sorry -- The proof goes here.
end

end square_midpoints_of_centers

end midpoints_form_square_l305_305327


namespace problem_1_problem_2_l305_305063

variable (α β : ℝ)

-- Given conditions
axiom sin_alpha : sin α = (2 * sqrt 5) / 5
axiom quadrant_condition : 0 < α ∧ α < π
axiom rotate_condition : β = α + π / 2

-- Required to prove that cos(α + 2π/3) = (sqrt 5 + 2 sqrt 15) / 10
theorem problem_1 : cos (α + 2 * π / 3) = (sqrt 5 + 2 * sqrt 15) / 10 := 
sorry

-- Required to prove that (sin α - cos(α - π)) / (sin β + 2sin(π / 2 - β)) = 1 / 5
theorem problem_2 : (sin α - cos (α - π)) / (sin β + 2 * sin (π / 2 - β)) = 1 / 5 := 
sorry

end problem_1_problem_2_l305_305063


namespace maurice_rides_before_visit_l305_305161

-- Defining all conditions in Lean
variables
  (M : ℕ) -- Number of times Maurice had been horseback riding before visiting Matt
  (Matt_rides_with_M : ℕ := 8 * 2) -- Number of times Matt rode with Maurice (8 times, 2 horses each time)
  (Matt_rides_alone : ℕ := 16) -- Number of times Matt rode solo
  (total_Matt_rides : ℕ := Matt_rides_with_M + Matt_rides_alone) -- Total rides by Matt
  (three_times_M : ℕ := 3 * M) -- Three times the number of times Maurice rode before visiting
  (unique_horses_M : ℕ := 8) -- Total number of unique horses Maurice rode during his visit

-- Main theorem
theorem maurice_rides_before_visit  
  (h1: total_Matt_rides = three_times_M) 
  (h2: unique_horses_M = M) 
  : M = 10 := sorry

end maurice_rides_before_visit_l305_305161


namespace sum_two_digit_numbers_l305_305677

theorem sum_two_digit_numbers : ∑ (x : ℕ) in {22, 23, 27, 32, 33, 37, 72, 73, 77}, x = 396 :=
by
  sorry

end sum_two_digit_numbers_l305_305677


namespace eq_has_four_int_solutions_l305_305005

theorem eq_has_four_int_solutions (a : ℝ) :
  (∀ x : ℤ, a^3 + a^2 * |a + x| + |a^2 * x + 1| = 1) →
  a ∈ set.Iic (-3) ∪ set.Icc (-real.sqrt (3 : ℝ) / 3) (1 / 2) :=
by sorry

end eq_has_four_int_solutions_l305_305005


namespace locus_of_P_l305_305041

/-- Given conditions: 
- A(1,1) is a point on the parabola y = x^2
- The tangent at A intersects the x-axis at D and y-axis at B
- C is a point on the parabola
- E lies on segment AC such that AE / EC = λ₁
- F lies on segment BC such that BF / FC = λ₂
- λ₁ + λ₂ = 1
- CD intersects EF at P

Prove: The locus of point P follows the equation y = (1 / 3) * (3x - 1)^2, x ≠ 2 / 3.
-/
theorem locus_of_P (A B C D E F P : Point) (λ₁ λ₂ : Real) :
    A = (1, 1) ∧
    B = (0, -1) ∧
    D = (1/2, 0) ∧
    (∃ x0 : Real, C = (x0, x0^2)) ∧
    E ∈ segment A C ∧
    F ∈ segment B C ∧
    λ₁ * x + λ₂ * (1 - x) = 1 ∧
    CD.intersect EF = P →
    ∃ P, ((y = (1 / 3) * (3 * x - 1)^2)) ∧ (x ≠ 2 / 3) :=
sorry

end locus_of_P_l305_305041


namespace frequency_ratio_50_70_l305_305379

-- Defining the class intervals and their corresponding frequencies
def classIntervalFrequencies : List (Set ℝ × ℕ) :=
[(set.Ioc 10 20, 2), (set.Ioc 20 30, 3), (set.Ioc 30 40, 4), (set.Ioc 40 50, 5), (set.Ioc 50 60, 4), (set.Ioc 60 70, 2)]

-- Defining the sample volume
def sampleVolume : ℕ := 20

-- Defining the frequencies for (50, 60] and (60, 70]
def freq_50_60 : ℕ := 4
def freq_60_70 : ℕ := 2

-- Defining the total frequency for the interval (50, 70]
def freq_50_70 : ℕ := freq_50_60 + freq_60_70

-- Statement to prove the frequency ratio in the interval (50, 70]
theorem frequency_ratio_50_70 : (freq_50_70 : ℝ) / (sampleVolume : ℝ) = 0.3 := by
  sorry

end frequency_ratio_50_70_l305_305379


namespace convex_quadrilateral_distance_l305_305169

theorem convex_quadrilateral_distance
    {A B C D : Type} [ConvexQuadrilateral A B C D]
    (hBD : length (B, D) = BD)
    (hAC_le_BD : length (A, C) ≤ length (B, D)) :
    (∃ V, V ∈ {A, C} ∧ exists_perpendicular_distance V (B, D) (λ d, d ≤ BD / 2)) := 
sorry

end convex_quadrilateral_distance_l305_305169


namespace exists_person_knows_everyone_l305_305117

-- Definitions of the problem
def Person := Fin 11  -- Representing the 11 people as indices from 0 to 10
def Knows (p q : Person) : Prop

-- Conditions
axiom mutual_acquaintance 
  (p q : Person) (hpq : p ≠ q) :
  ∃! r : Person, r ≠ p ∧ r ≠ q ∧ Knows r p ∧ Knows r q

-- The proof statement (the conclusion to be shown)
theorem exists_person_knows_everyone :
  ∃ p : Person, ∀ q : Person, q ≠ p → Knows p q :=
sorry

end exists_person_knows_everyone_l305_305117


namespace pow_mod_eq_l305_305658

theorem pow_mod_eq (n : ℕ) : 
  (3^n % 5 = 3 % 5) → 
  (3^(n+1) % 5 = (3 * 3^n) % 5) → 
  (3^(n+2) % 5 = (3 * 3^(n+1)) % 5) → 
  (3^(n+3) % 5 = (3 * 3^(n+2)) % 5) → 
  (3^4 % 5 = 1 % 5) → 
  (2023 % 4 = 3) → 
  (3^2023 % 5 = 2 % 5) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pow_mod_eq_l305_305658


namespace problem_conditions_eval_l305_305737

open Set

theorem problem_conditions_eval : 
  (∀ x, (x ∈ ∅ → x ∈ ({a} : Set α)) ∧ 
        (a ⊂ ({a} : Set α) ↔ False) ∧ 
        (({a} : Set α) ⊆ {a}) ∧ 
        (({a} : Set α) ∈ ({a, b} : Set (Set α)) ↔ False) ∧ 
        (a ∈ ({a, b, c} : Set α)) ∧ 
        (∅ ∈ ({a, b} : Set α) ↔ False)) →
  ({1, 3, 5} = {x | 
     (x = 1 ∧ ∅ ⊆ ({a} : Set α)) ∨
     (x = 3 ∧ ({a} : Set α) ⊆ {a}) ∨
     (x = 5 ∧ (a ∈ ({a, b, c} : Set α)))})


end problem_conditions_eval_l305_305737


namespace hypotenuse_length_l305_305495

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305495


namespace inequality_proof_l305_305555

theorem inequality_proof (n : ℕ) (hn : 3 ≤ n) (x : ℕ → ℝ)
  (hx : ∀ i j : ℕ, i ≤ j → i < n → j < n → x i ≤ x j) :
  (∑ i in range (n-1), (x (i+1) * x 0) / x (i+2)) + (x n * x 0) / x 1 ≥ ∑ i in range n, x i :=
sorry

end inequality_proof_l305_305555


namespace units_digit_of_p_is_6_l305_305819

theorem units_digit_of_p_is_6 (p : ℤ) (h1 : p % 10 > 0) 
                             (h2 : ((p^3) % 10 - (p^2) % 10) = 0) 
                             (h3 : (p + 1) % 10 = 7) : 
                             p % 10 = 6 :=
by sorry

end units_digit_of_p_is_6_l305_305819


namespace total_smaller_cubes_is_27_l305_305704

-- Given conditions
def painted_red (n : ℕ) : Prop := ∀ face, face ∈ cube.faces → face.color = red

def cut_into_smaller_cubes (n : ℕ) : Prop := ∃ k : ℕ, k = n + 1

def smaller_cubes_painted_on_2_faces (cubes_painted_on_2_faces : ℕ) (n : ℕ) : Prop :=
  cubes_painted_on_2_faces = 12 * (n - 1)

-- Question: Prove the total number of smaller cubes is equal to 27, given the conditions
theorem total_smaller_cubes_is_27 (n : ℕ) (h1 : painted_red n) (h2 : cut_into_smaller_cubes n) (h3 : smaller_cubes_painted_on_2_faces 12 n) :
  (n + 1)^3 = 27 := by
  sorry

end total_smaller_cubes_is_27_l305_305704


namespace complement_of_65_degrees_l305_305453

def angle_complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_65_degrees : angle_complement 65 = 25 := by
  -- Proof would follow here, but it's omitted since 'sorry' is added.
  sorry

end complement_of_65_degrees_l305_305453


namespace parallel_vectors_sufficiency_l305_305834

noncomputable def parallel_vectors_sufficiency_problem (a b : ℝ × ℝ) (x : ℝ) : Prop :=
a = (1, x) ∧ b = (x, 4) →
(x = 2 → ∃ k : ℝ, k • a = b) ∧ (∃ k : ℝ, k • a = b → x = 2 ∨ x = -2)

theorem parallel_vectors_sufficiency (x : ℝ) :
  parallel_vectors_sufficiency_problem (1, x) (x, 4) x :=
sorry

end parallel_vectors_sufficiency_l305_305834


namespace max_value_2m_l305_305324

noncomputable theory
open_locale classical

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n k, a n ∣ (finset.range n).sum (λ i, a (k + i))

theorem max_value_2m (a : ℕ → ℕ) (m : ℕ) (h : satisfies_condition a) : 
  a (2 * m) ≤ 2^m - 1 :=
by sorry

end max_value_2m_l305_305324


namespace odd_function_f_1_l305_305098

theorem odd_function_f_1 (a : ℝ) (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_def : ∀ x, f x = a - 2 / (2^x + 1)) : f 1 = 1/3 :=
by
  let f0 := (a - 1)
  have ha : f0 = 0,
  sorry
  have hap : a = 1,
  sorry
  have f1 := (1 - 2 / (2 + 1)),
  have : f1 = 1/3,
  exact this
  sorry

end odd_function_f_1_l305_305098


namespace correct_conclusions_l305_305800

noncomputable def f : ℝ → ℝ := sorry -- Assume f has already been defined elsewhere

-- The given conditions
variable (f : ℝ → ℝ)
variable (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
variable (H2 : f (1 / 2) = 0)
variable (H3 : f 0 ≠ 0)

-- Proof problem which includes the correct conclusions
theorem correct_conclusions (f : ℝ → ℝ)
  (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
  (H2 : f (1 / 2) = 0)
  (H3 : f 0 ≠ 0) :
  f 0 = 1 ∧ ∀ y : ℝ, f (1 / 2 + y) + f (1 / 2 - y) = 0 :=
by
  sorry

end correct_conclusions_l305_305800


namespace ticket_combinations_adjacent_l305_305567

noncomputable def num_ticket_combinations_adjacent (winning_numbers : List ℕ) (total_numbers : ℕ) (k : ℕ) : ℕ :=
  let neighbors := winning_numbers.bind (λ n, [n-1, n+1])
  let remaining_numbers := total_numbers - winning_numbers.length - neighbors.length
  let num_combinations := (nat.choose neighbors.length k) + remaining_numbers * (nat.choose neighbors.length (k-1))
  num_combinations

theorem ticket_combinations_adjacent (winning_numbers : List ℕ) (total_numbers : ℕ) :
  winning_numbers = [7, 13, 28, 46, 75] →
  total_numbers = 90 →
  num_ticket_combinations_adjacent winning_numbers total_numbers 4 = 17052 := 
by
  intros h_winning h_total
  sorry

end ticket_combinations_adjacent_l305_305567


namespace lowest_score_l305_305594

-- Define the conditions
def test_scores (s1 s2 s3 : ℕ) := s1 = 86 ∧ s2 = 112 ∧ s3 = 91
def max_score := 120
def target_average := 95
def num_tests := 5
def total_points_needed := target_average * num_tests

-- Define the proof statement
theorem lowest_score 
  (s1 s2 s3 : ℕ)
  (condition1 : test_scores s1 s2 s3)
  (max_pts : ℕ := max_score) 
  (target_avg : ℕ := target_average) 
  (num_tests : ℕ := num_tests)
  (total_needed : ℕ := total_points_needed) :
  ∃ s4 s5 : ℕ, s4 ≤ max_pts ∧ s5 ≤ max_pts ∧ s4 + s5 + s1 + s2 + s3 = total_needed ∧ (s4 = 66 ∨ s5 = 66) :=
by
  sorry

end lowest_score_l305_305594


namespace tangent_line_equation_l305_305350

theorem tangent_line_equation (A : ℝ × ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3) (h2 : A.snd = (A.fst)^3)
  (h3 : ∀ x, deriv f x = 3 * x^2) (h4 : deriv f A.fst = 3) :
  (A = (1, 1) ∨ A = (-1, -1)) → 
  (∀ x, (A = (1, 1) → (∀ y, y = 3 * x - 2 → y = 3 * A.fst - 2))
   ∧ (A = (-1, -1) → (∀ y, y = 3 * x + 2 → y = 3 * A.fst + 2))) :=
begin
  sorry
end

end tangent_line_equation_l305_305350


namespace hypotenuse_length_l305_305492

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305492


namespace rachel_one_hour_earnings_l305_305948

theorem rachel_one_hour_earnings :
  let hourly_wage := 12.00
  let number_of_people_served := 20
  let tip_per_person := 1.25
  let total_tips := number_of_people_served * tip_per_person
  let total_earnings := hourly_wage + total_tips
  in total_earnings = 37.00 :=
by
  sorry

end rachel_one_hour_earnings_l305_305948


namespace square_root_meaningful_range_l305_305865

theorem square_root_meaningful_range (x : ℝ) : (∃ y : ℝ, y = sqrt(x - 5)) ↔ x ≥ 5 :=
by
  sorry

end square_root_meaningful_range_l305_305865


namespace generalized_tangent_identity_l305_305574

theorem generalized_tangent_identity
  (h1 : tan 10 * tan 20 + tan 20 * tan 60 + tan 60 * tan 10 = 1)
  (h2 : tan 5 * tan 10 + tan 10 * tan 75 + tan 75 * tan 5 = 1) :
  ∀ α β γ : ℝ, α + β + γ = π / 2 → tan α * tan β + tan β * tan γ + tan γ * tan α = 1 :=
by
  intros α β γ h_sum
  sorry

end generalized_tangent_identity_l305_305574


namespace emily_trip_duration_same_l305_305878

theorem emily_trip_duration_same (s : ℝ) (h_s_pos : 0 < s) : 
  let t1 := (90 : ℝ) / s
  let t2 := (360 : ℝ) / (4 * s)
  t2 = t1 := sorry

end emily_trip_duration_same_l305_305878


namespace decimal_addition_l305_305686

theorem decimal_addition : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end decimal_addition_l305_305686


namespace find_a_prove_inequality_l305_305411

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + 2 * x + a * Real.log x

theorem find_a (a : ℝ) (h : (2 * Real.exp 1 + 2 + a) * (-1 / 2) = -1) : a = -2 * Real.exp 1 :=
by
  sorry

theorem prove_inequality (a : ℝ) (h1 : a = -2 * Real.exp 1) :
    ∀ x : ℝ, x > 0 → f x a > x^2 + 2 :=
by
  sorry

end find_a_prove_inequality_l305_305411


namespace number_of_distinct_positive_differences_is_seven_l305_305424

def set_of_integers := {1, 2, 3, 4, 5, 6, 7, 8}

theorem number_of_distinct_positive_differences_is_seven :
  (set_of_integers \ {0}).image (λ x, set_of_integers \ {x}) |>.nonempty :=
by
  sorry

end number_of_distinct_positive_differences_is_seven_l305_305424


namespace range_of_m_l305_305406

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then (1 / 3)^(-x) - 2 
  else 2 * Real.log x / Real.log 3

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < -Real.sqrt 3} ∪ {m : ℝ | 1 < m} :=
by
  sorry

end range_of_m_l305_305406


namespace selling_price_with_increase_l305_305985

variable (a : ℝ)

theorem selling_price_with_increase (h : a > 0) : 1.1 * a = a + 0.1 * a := by
  -- Here you will add the proof, which we skip with sorry
  sorry

end selling_price_with_increase_l305_305985


namespace marked_vertices_coincide_l305_305119

theorem marked_vertices_coincide :
  ∀ (P Q : Fin 16 → Prop),
  (∃ A B C D E F G : Fin 16, P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G) →
  (∃ A' B' C' D' E' F' G' : Fin 16, Q A' ∧ Q B' ∧ Q C' ∧ Q D' ∧ Q E' ∧ Q F' ∧ Q G') →
  ∃ (r : Fin 16), ∃ (A B C D : Fin 16), 
  (Q ((A + r) % 16) ∧ Q ((B + r) % 16) ∧ Q ((C + r) % 16) ∧ Q ((D + r) % 16)) :=
by
  sorry

end marked_vertices_coincide_l305_305119


namespace hypotenuse_length_l305_305481

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305481


namespace yellow_balls_are_24_l305_305869

theorem yellow_balls_are_24 (x y z : ℕ) (h1 : x + y + z = 68) 
                             (h2 : y = 2 * x) (h3 : 3 * z = 4 * y) : y = 24 :=
by
  sorry

end yellow_balls_are_24_l305_305869


namespace sphere_radius_minimizing_volume_sum_l305_305804

theorem sphere_radius_minimizing_volume_sum (a : ℝ) :
  let r := a * (1 / 3) * Real.sqrt (2 / 3) in
  is_minimizing_radius r a :=
sorry

end sphere_radius_minimizing_volume_sum_l305_305804


namespace scenario1_is_linear_scenario2_is_quadratic_l305_305635

-- Condition for the first problem
def cond1 : ℝ → ℝ := λ x, 5 * (10 - x)

-- Condition for the second problem
def cond2 : ℝ → ℝ := λ x, (30 + x) * (20 + x)

-- Proof that cond1 is a linear function
theorem scenario1_is_linear : ∀ x : ℝ, ∃ m b : ℝ, cond1 x = m * x + b :=
by
  intros x
  -- Outline the values of m and b
  use -5, 50
  sorry

-- Proof that cond2 is a quadratic function
theorem scenario2_is_quadratic : ∀ x : ℝ, ∃ a b c : ℝ, cond2 x = a * x^2 + b * x + c :=
by
  intros x
  -- Outline the values of a, b, and c
  use 1, 50, 600
  sorry

end scenario1_is_linear_scenario2_is_quadratic_l305_305635


namespace problem1_problem2_l305_305813

-- Defining the standard constants and functions.
noncomputable def triangle_ABC 
  (a b c : ℝ) 
  (A B C : ℝ)
  (ha : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (habc: 0 < a ∧ 0 < b ∧ 0 < c)
  : Prop := 
  a^2 = b^2 + c^2 - 2 * b * c * real.cos A ∧
  b^2 = a^2 + c^2 - 2 * a * c * real.cos B ∧
  c^2 = a^2 + b^2 - 2 * a * b * real.cos C

theorem problem1 
(a b c A B C : ℝ)
(ha : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
(habc : 0 < a ∧ 0 < b ∧ 0 < c)
(hcond : (2 * b - c) / a = real.cos C / real.cos A):
  A = π / 3 :=
sorry

theorem problem2 
(a b c A B C : ℝ)
(ha : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
(habc : 0 < a ∧ 0 < b ∧ 0 < c)
(hcond : a = real.sqrt 3 ∧ (2 * b - c) / a = real.cos C / real.cos A):
  3 < b + c ∧ b + c ≤ 2 * real.sqrt 3 :=
sorry

end problem1_problem2_l305_305813


namespace triangle_formation_inequalities_l305_305586

variables {P Q R S : Type}
variables {a b c : ℝ} (h_collinear : collinear {P, Q, R, S})
variables (h_len1 : dist P Q = a) (h_len2 : dist P R = b) (h_len3 : dist P S = c)

theorem triangle_formation_inequalities (h : distinct [P, Q, R, S]) :
  (a < c / 3) ∧ (b < 2 * a + c) :=
by
  sorry

end triangle_formation_inequalities_l305_305586


namespace hypotenuse_length_l305_305496

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305496


namespace distinct_positive_differences_l305_305429

theorem distinct_positive_differences :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  ∑ i in s, ∑ j in s, if i ≠ j then (Nat.abs (i - j) : Finset ℕ) else 0 = {1, 2, 3, 4, 5, 6, 7} :=
sorry

end distinct_positive_differences_l305_305429


namespace rational_coefficient_exists_in_binomial_expansion_l305_305398

theorem rational_coefficient_exists_in_binomial_expansion :
  ∃! (n : ℕ), n > 0 ∧ (∀ r, (r % 3 = 0 → (n - r) % 2 = 0 → n = 7)) :=
by
  sorry

end rational_coefficient_exists_in_binomial_expansion_l305_305398


namespace largest_digit_product_sum_squares_65_l305_305755

theorem largest_digit_product_sum_squares_65 :
  ∃ (n : ℕ), 
    (∃ digs : List ℕ, 
      (∀ d ∈ digs, d > 0 ∧ d < 10) ∧
      (List.pairwise (· < ·) digs) ∧ 
      (digs.map (λ d => d ^ 2)).sum = 65 ∧ 
      n = digs.foldr (· * ·) 1
    ) ∧ n = 60 :=
sorry

end largest_digit_product_sum_squares_65_l305_305755


namespace all_lines_pass_through_single_point_l305_305470

-- Define the mathematical context for the problem
variables {P : Type} [Plane P]
variables (L : set (Line P))

-- Condition: finite number of mutually non-parallel lines
def mutually_non_parallel (lines : set (Line P)) : Prop :=
  ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ parallel l1 l2

-- Condition: through intersection point of any two lines passes another given line
def intersection_property (lines : set (Line P)) (l : Line P) : Prop :=
  ∀ l1 l2 ∈ lines, l1 ≠ l2 → l ∈ intersection_points l1 l2

-- Main theorem statement
theorem all_lines_pass_through_single_point
  (finite_lines : finite L)
  (non_parallel : mutually_non_parallel L)
  (intersect_line : ∃ l : Line P, intersection_property L l) :
  ∃ p : Point P, ∀ l ∈ L, p ∈ l :=
sorry

end all_lines_pass_through_single_point_l305_305470


namespace slope_of_line_is_constant_find_equation_of_ellipse_l305_305383

open Real

-- Define the terms used in the problem statement
noncomputable def ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def eccentricity (a b : ℝ) := (sqrt (a^2 - b^2)) / a

-- Points P, Q, M, N, and origin O
variable {P Q M N : ℝ × ℝ}
variable (O : ℝ × ℝ := (0, 0))

-- Areas of triangles
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := abs(1/2 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)))

-- Define the given condition as a theorem to prove
theorem slope_of_line_is_constant (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : eccentricity a b = sqrt 3 / 2)
 (h4 : (1, sqrt 3 / 2) ∈ ellipse a b)
 (h5 : P.1 > 0) (h6 : P.2 > 0) (h7 : Q.1 > 0) (h8 : Q.2 > 0)
 (h9 : M = (M.1, 0)) (h10 : N = (0, N.2))
 (h11 : (area_triangle P M O)^2 + (area_triangle Q M O)^2 = ((area_triangle P M O) * (area_triangle Q M O) *
     (area_triangle P N O)^2 + (area_triangle Q N O)^2) / ((area_triangle P N O) * (area_triangle Q N O))) :
  ∃ k : ℝ, k = -1/2 :=
sorry

-- The statement proving the ellipse's equation given properties
theorem find_equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
 (h3 : eccentricity a b = sqrt 3 / 2) (h4 : (1, sqrt 3 / 2) ∈ ellipse a b) :
  ellipse a b = { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 } :=
sorry

end slope_of_line_is_constant_find_equation_of_ellipse_l305_305383


namespace find_U_value_l305_305551

def f : ℕ → ℝ
def U (T : ℕ) : ℝ :=
  (f T) / ((T - 1) * (f (T - 3)))

theorem find_U_value (h1 : ∀ n, n ≥ 6 → f n = (n-1) * f (n-1))
  (h2 : ∀ n, n ≥ 6 → f n ≠ 0) : U 11 = 72 :=
by
  sorry

end find_U_value_l305_305551


namespace stickers_total_l305_305673

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l305_305673


namespace collinear_iff_linear_combination_l305_305166

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (k : ℝ)

theorem collinear_iff_linear_combination (O A B C : V) (k : ℝ) :
  (C = k • A + (1 - k) • B) ↔ ∃ (k' : ℝ), C - B = k' • (A - B) :=
sorry

end collinear_iff_linear_combination_l305_305166


namespace point_satisfies_equation_l305_305683

theorem point_satisfies_equation (x y : ℝ) :
  (-1 ≤ x ∧ x ≤ 3) ∧ (-5 ≤ y ∧ y ≤ 1) ∧
  ((3 * x + 2 * y = 5) ∨ (-3 * x + 2 * y = -1) ∨ (3 * x - 2 * y = 13) ∨ (-3 * x - 2 * y = 7))
  → 3 * |x - 1| + 2 * |y + 2| = 6 := 
by 
  sorry

end point_satisfies_equation_l305_305683


namespace bounded_figure_single_center_of_symmetry_no_two_centers_of_symmetry_finite_set_max_three_almost_centers_l305_305270

-- Lean statement for problem (a)
theorem bounded_figure_single_center_of_symmetry
  (F : Set Point)
  (bounded_F : Bounded F)
  (O1 O2 : Point)
  (center_O1 : is_center_of_symmetry F O1)
  (center_O2 : is_center_of_symmetry F O2) :
  O1 = O2 := 
sorry

-- Lean statement for problem (b)
theorem no_two_centers_of_symmetry
  (F : Set Point)
  (O1 O2 : Point)
  (center_O1 : is_center_of_symmetry F O1)
  (center_O2 : is_center_of_symmetry F O2) :
  ∃ O3, O3 ≠ O1 ∧ O3 ≠ O2 ∧ is_center_of_symmetry F O3 :=
sorry

-- Lean statement for problem (c)
theorem finite_set_max_three_almost_centers
  (M : Finset Point)
  (M_finite : Set.Finite M) :
  (Card (almost_centers_of_symmetry M) ≤ 3) :=
sorry

end bounded_figure_single_center_of_symmetry_no_two_centers_of_symmetry_finite_set_max_three_almost_centers_l305_305270


namespace sum_of_second_and_third_circles_l305_305240

-- Definitions for sum of digits
def S (n : ℕ) := n.digits.sum

-- Conditions and proof goal
theorem sum_of_second_and_third_circles (N : ℕ) (h1 : S N + S (N+1) = 200) (h2 : S (N+2) + S (N+3) = 105) :
  S (N+1) + S (N+2) = 103 :=
sorry

end sum_of_second_and_third_circles_l305_305240


namespace triangle_proportional_segments_l305_305942

theorem triangle_proportional_segments
  (A B C E F S M N K : Point)
  (hE : E ∈ Line (A B)) (hF : F ∈ Line (A C))
  (hS : Collinear S E F) (hMidM : Midpoint M B C) (hMidN : Midpoint N E F) 
  (hParallel : Parallel (Line (A K)) (Line (M N))) (hK : K ∈ Line (B C)) :
  (BKLengthCKLength (A B C E F S M N K hE hF hS hMidM hMidN hParallel hK)) = FSESRatio (A B C E F S M N K) :=
    sorry

end triangle_proportional_segments_l305_305942


namespace marys_total_candy_l305_305569

/-
Given:
1. Mary has 3 times as much candy as Megan.
2. Megan has 5 pieces of candy.
3. Mary adds 10 more pieces of candy to her collection.

Show that the total number of pieces of candy Mary has is 25.
-/

theorem marys_total_candy :
  (∀ (mary_candy megan_candy : ℕ), mary_candy = 3 * megan_candy → megan_candy = 5 → mary_candy + 10 = 25) :=
begin
  intros mary_candy megan_candy h1 h2,
  rw h2 at h1,
  rw h1,
  norm_num,
end

end marys_total_candy_l305_305569


namespace cheese_distribution_l305_305632

theorem cheese_distribution (w : Fin 25 → ℝ) (h_diff : ∀ i j, i ≠ j → w i ≠ w j) :
  ∃ i (a b : ℝ), a + b = w i ∧ 
  ∃ (S T : Finset (Fin 26)) (wS wT : ℝ), 
    S.card = 13 ∧ T.card = 13 ∧
    (∀ k ∈ S, ∃ j, (j ∈ (Finset.finRange 25) ∨ j = 25) ∧ (if j = 25 then k ∈ {i} else k = j)) ∧
    (∀ k ∈ T, ∃ j, (j ∈ (Finset.finRange 25) ∨ j = 25) ∧ (if j = 25 then k ∈ {i} else k = j)) ∧
    wS = S.sum (λ k, if k = 25 then a else w ⟨k, sorry⟩) ∧
    wT = T.sum (λ k, if k = 25 then b else w ⟨k, sorry⟩) ∧
    wS = wT :=
by 
  sorry

end cheese_distribution_l305_305632


namespace solve_for_p_l305_305004

def cubic_eq_has_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  5*(a:ℝ)^3 - 5*(p + 1)*(a:ℝ)^2 + (71*p - 1)*(a:ℝ) + 1 = 66*p ∧
  5*(b:ℝ)^3 - 5*(p + 1)*(b:ℝ)^2 + (71*p - 1)*(b:ℝ) + 1 = 66*p ∧
  5*(c:ℝ)^3 - 5*(p + 1)*(c:ℝ)^2 + (71*p - 1)*(c:ℝ) + 1 = 66*p

theorem solve_for_p : ∀ (p : ℝ), cubic_eq_has_natural_roots p → p = 76 :=
by
  sorry

end solve_for_p_l305_305004


namespace karen_class_size_l305_305906

namespace proof

theorem karen_class_size:
  let cookies_initial := 50 in
  let cookies_kept := 10 in
  let cookies_given_grandparents := 8 in
  let cookies_per_person := 2 in
  (cookies_initial - (cookies_kept + cookies_given_grandparents)) / cookies_per_person = 16 :=
by
  sorry

end proof

end karen_class_size_l305_305906


namespace find_a_plus_c_l305_305977

theorem find_a_plus_c {a b c d : ℝ} 
  (h1 : ∀ x, -|x - a| + b = |x - c| + d → x = 4 ∧ -|4 - a| + b = 7 ∨ x = 10 ∧ -|10 - a| + b = 3)
  (h2 : b + d = 12): a + c = 14 := by
  sorry

end find_a_plus_c_l305_305977


namespace find_e_l305_305360

theorem find_e (x y e : ℝ) (h1 : x / (2 * y) = 5 / e) (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : e = 2 := 
by
  sorry

end find_e_l305_305360


namespace scientific_notation_of_510000000_l305_305631

theorem scientific_notation_of_510000000 :
  (510000000 : ℝ) = 5.1 * 10^8 := 
sorry

end scientific_notation_of_510000000_l305_305631


namespace hypotenuse_length_l305_305493

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305493


namespace sum_even_integers_16_to_58_l305_305659

theorem sum_even_integers_16_to_58 : 
  let a := 16
  let d := 2
  let l := 58
  let n := (l - a) / d + 1
  2 * (l - a) / d + 1 = 22
  n * (a + l) / 2 = 814 :=
by
  let a := 16
  let d := 2
  let l := 58
  let n := (l - a) / d + 1
  have n_def : 2 * (l - a) / d + 1 = 22 := by
    have eq : 2 * (l - a) / d + 1 = 22 := by sorry
    exact eq
  have sum := n * (a + l) / 2 = 814 := by sorry
  exact sum

end sum_even_integers_16_to_58_l305_305659


namespace range_of_a_l305_305408

-- Define the piecewise function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x + Real.cos x else x * (a - x)

-- Define conditions for the inequality f(x) < π
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x < Real.pi}

-- The statement to prove:
theorem range_of_a (a : ℝ) : solution_set a = Set.Ioo Real.pi (-2 * Real.sqrt Real.pi) :=
  sorry

end range_of_a_l305_305408


namespace time_to_walk_to_school_l305_305845

variable (constant_speed : Prop)
variable (time_to_library : ℕ)
variable (distance_to_library : ℕ)
variable (distance_to_school : ℕ)

theorem time_to_walk_to_school
    (constant_speed : I walk at a constant speed)
    (time_to_library : It takes me 30 minutes to walk 4 kilometers)
    (distance_to_library : The library is 4 kilometers away from my house)
    (distance_to_school : My school is 2 kilometers away from my house)
    : time to walk from house to school = 15 := by
    sorry

end time_to_walk_to_school_l305_305845


namespace negative_terms_for_large_n_l305_305387

variables {a d : ℝ} {a_n : ℕ → ℝ}

def arithmetic_seq (a d : ℝ) : ℕ → ℝ := λ n, a + n * d

def sum_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a + (a + n * d)) / 2

axiom sum_6_lt_sum_7 : sum_first_n_terms a d 5 < sum_first_n_terms a d 6
axiom sum_7_gt_sum_8 : sum_first_n_terms a d 6 > sum_first_n_terms a d 7

theorem negative_terms_for_large_n (n : ℕ) (hn : n ≥ 8) : 
  (arithmetic_seq a d n) < 0 :=
by {
  sorry
}

end negative_terms_for_large_n_l305_305387


namespace rational_number_div_l305_305093

theorem rational_number_div (x : ℚ) (h : -2 / x = 8) : x = -1 / 4 := 
by
  sorry

end rational_number_div_l305_305093


namespace least_integer_divisible_by_17_with_digit_sum_17_l305_305151

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

theorem least_integer_divisible_by_17_with_digit_sum_17 :
  ∃ m : ℕ, (m > 0) ∧ (m % 17 = 0) ∧ (sum_of_digits m = 17) ∧ (∀ n : ℕ, (n > 0) ∧ (n % 17 = 0) ∧ (sum_of_digits n = 17) → n ≥ 476) :=
sorry

end least_integer_divisible_by_17_with_digit_sum_17_l305_305151


namespace calculate_result_l305_305016

def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

theorem calculate_result : ((star 2 3 (by norm_num)) ▸ (λ x : ℝ, star x 4 (by norm_num))) = 1 / 9 :=
by sorry

end calculate_result_l305_305016


namespace hypotenuse_length_l305_305501

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305501


namespace sum_mod_20_l305_305354

def sum_of_numbers : ℕ := (75 + 76 + 77 + 78 + 79 + 80 + 81 + 82 + 83 + 84 + 85 + 86 + 87 + 88)

theorem sum_mod_20 : sum_of_numbers % 20 = 1 :=
by {
  show (75 + 76 + 77 + 78 + 79 + 80 + 81 + 82 + 83 + 84 + 85 + 86 + 87 + 88) % 20 = 1,
  sorry
}

end sum_mod_20_l305_305354


namespace hypotenuse_length_l305_305473

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305473


namespace triangle_is_isosceles_l305_305992

theorem triangle_is_isosceles
  (p q r : ℝ)
  (H : ∀ (n : ℕ), n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n))
  : p = q ∨ q = r ∨ r = p := 
begin
  sorry
end

end triangle_is_isosceles_l305_305992


namespace hypotenuse_length_l305_305474

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305474


namespace hypotenuse_length_l305_305475

theorem hypotenuse_length (a b c : ℝ) (h_right : c^2 = a^2 + b^2) (h_sum_squares : a^2 + b^2 + c^2 = 2500) :
  c = 25 * Real.sqrt 2 := by
  sorry

end hypotenuse_length_l305_305475


namespace find_second_divisor_l305_305226

theorem find_second_divisor :
  ∃ y : ℝ, (320 / 2) / y = 53.33 ∧ y = 160 / 53.33 :=
by
  sorry

end find_second_divisor_l305_305226


namespace simplify_fraction_l305_305958

open Real

theorem simplify_fraction (x : ℝ) : (3 + 2 * sin x + 2 * cos x) / (3 + 2 * sin x - 2 * cos x) = 3 / 5 + (2 / 5) * cos x :=
by
  sorry

end simplify_fraction_l305_305958


namespace vincent_total_packs_l305_305670

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l305_305670


namespace num_pos_ints_satisfying_ineq_l305_305011

theorem num_pos_ints_satisfying_ineq :
  ∃ (n_set : Set ℕ), 
    (∀ n ∈ n_set, (List.ofFn (λ k : Fin 49 => n - 2 * (k : ℕ))).prod > 0) ∧ 
    n_set.card = 24 :=
sorry

end num_pos_ints_satisfying_ineq_l305_305011


namespace bounded_figure_has_at_most_one_center_of_symmetry_l305_305267

theorem bounded_figure_has_at_most_one_center_of_symmetry (F : Set Point) [Bounded F] :
  ∀ O1 O2 : Point, is_center_of_symmetry F O1 → is_center_of_symmetry F O2 → O1 = O2 := 
sorry

end bounded_figure_has_at_most_one_center_of_symmetry_l305_305267


namespace milk_fraction_after_operations_l305_305160

/-- Lucy places eight ounces of tea into a sixteen-ounce cup and eight ounces of milk into a second cup of the same size.
She then pours one third of the tea from the first cup to the second and, after stirring thoroughly, pours one quarter of the
liquid in the second cup back to the first. This theorem proves that the fraction of the liquid in the first cup that is now milk is 1/4. -/
theorem milk_fraction_after_operations :
  ∃ (t m t_final m_final : ℚ),
    t = 8 ∧ m = 8 ∧
    let t_transferred := (1 / 3) * t in
    let cup2_total := m + t_transferred in
    let liquid_back := (1 / 4) * cup2_total in
    let milk_ratio := m / cup2_total in
    let tea_ratio := t_transferred / cup2_total in
    let milk_back := liquid_back * milk_ratio in
    let tea_back := liquid_back * tea_ratio in
    let t_final := t - t_transferred + tea_back in
    let m_final := milk_back in
    m_final / (m_final + t_final) = 1 / 4 :=
begin
  sorry
end

end milk_fraction_after_operations_l305_305160


namespace rotate_A_180_about_B_l305_305585

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (-1, 1)

-- Define the 180 degrees rotation about B
def rotate_180_about (p q : ℝ × ℝ) : ℝ × ℝ :=
  let translated_p := (p.1 - q.1, p.2 - q.2) 
  let rotated_p := (-translated_p.1, -translated_p.2)
  (rotated_p.1 + q.1, rotated_p.2 + q.2)

-- Prove the image of point A after a 180 degrees rotation about point B
theorem rotate_A_180_about_B : rotate_180_about A B = (2, 7) :=
by
  sorry

end rotate_A_180_about_B_l305_305585


namespace quadratic_trinomials_disjoint_neg_intervals_l305_305617

theorem quadratic_trinomials_disjoint_neg_intervals
  (f g : ℝ → ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, x1 < x ∧ x < x2 → f(x) < 0)
  (h2 : ∀ x, x3 < x ∧ x < x4 → g(x) < 0)
  (h3 : x2 < x3) :
  ∃ α β : ℝ, 0 < α ∧ 0 < β ∧ ∀ x : ℝ, α * f(x) + β * g(x) > 0 := 
by {
  -- omitted proof
  sorry
}

end quadratic_trinomials_disjoint_neg_intervals_l305_305617


namespace sequence_divisible_1001_l305_305412

theorem sequence_divisible_1001 (a : Fin 10 → ℤ) : 
  ∃ x : Fin 10 → ℤ, (∀ i, x i ∈ {-1, 0, 1}) ∧ (∑ i, x i * a i) % 1001 = 0 ∧ 
  (∃ j, x j ≠ 0) := 
by 
  sorry

end sequence_divisible_1001_l305_305412


namespace sum_of_reciprocal_squares_l305_305612

theorem sum_of_reciprocal_squares
  (p q r : ℝ)
  (h1 : p + q + r = 9)
  (h2 : p * q + q * r + r * p = 8)
  (h3 : p * q * r = -2) :
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 25 := by
  sorry

end sum_of_reciprocal_squares_l305_305612


namespace translate_line_upwards_l305_305235

/-- Translate the line y = -2x + 1 vertically upwards by 2 units
and prove the new line equation is y = -2x + 3. -/
theorem translate_line_upwards :
  ∀ (x : ℝ), ∃ (y : ℝ), (y = -2 * x + 3) :=
begin
  intro x,
  use -2 * x + 3,
  sorry,
end

end translate_line_upwards_l305_305235


namespace water_depth_after_block_l305_305279

noncomputable def newWaterDepth (a : ℕ) : ℕ :=
  if a ≥ 28 then 30
  else if a = 8 then 10
  else if 8 < a ∧ a < 28 then a + 2
  else (5 * a) / 4

theorem water_depth_after_block (a : ℕ) (h : a ≤ 30) :
  ∃ d : ℕ, d = 30 ∨ d = 10 ∨ d = a + 2 ∨ d = (5 * a) / 4 :=
  ⟨newWaterDepth a, by
    unfold newWaterDepth
    split_ifs
    · left
      refl
    · right; left
      refl
    · right; right; left
      refl
    · right; right; right
      refl
  ⟩

end water_depth_after_block_l305_305279


namespace puppy_food_total_correct_l305_305927

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l305_305927


namespace find_angleC_l305_305809

def angleC (a b : ℝ) (A : ℝ) : ℝ :=
if h_angle : A < 180 ∧ A > 0 then sorry else sorry

theorem find_angleC (a b : ℝ) (A : ℝ) (h_a : a = 2) (h_b : b = sqrt 6) (h_A : A = 45) :
  angleC a b A = 15 ∨ angleC a b A = 75 :=
sorry

end find_angleC_l305_305809


namespace Tn_correct_inequality_condition_max_k_l305_305376

-- Define the sequence $(a_n)$ with the sum of the first $n$ terms $S_n = 2^n + r$
def Sn (n : ℕ) (r : ℝ) : ℝ := ∑ i in finset.range n, (2^i + r)

-- Define $b_n = 2(1 + \log_2 a_n)$
def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 * (1 + real.log (a n) / real.log 2)

-- Define $a_n$ as the difference of the sums $S_n - S_{n-1}$
def an (n : ℕ) (r : ℝ) : ℝ := (2^n + r) - (2^(n-1) + r)

-- Define $a_n b_n$ for $n \in \mathbf{N}^{*}$
def anb (a : ℕ → ℝ) (n : ℕ) : ℝ := an n (-1) * bn a n

-- Define $T_n$ as the sum of the first $n$ terms of $a_n b_n$
def Tn (n : ℕ) (a : ℕ → ℝ) : ℝ := ∑ i in finset.range (n+1), anb a i

-- Define the inequality condition
def inequality_condition (n : ℕ) (a : ℕ → ℝ) : Prop :=
  (∏ i in finset.range (n+1), (1 + bn a i) / bn a i) ≥ real.sqrt (n + 1) * (3 * real.sqrt 2 / 4)

theorem Tn_correct (n : ℕ) (a : ℕ → ℝ) :
  Tn n a = (n - 1) * 2^(n+1) + 2 := sorry

theorem inequality_condition_max_k (k : ℝ) (a : ℕ → ℝ) :
  (∀ n, inequality_condition n a) → k ≤ (3 * real.sqrt 2 / 4) := sorry

end Tn_correct_inequality_condition_max_k_l305_305376


namespace correct_propositions_for_curve_C_l305_305823

def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (4 - k) + y^2 / (k - 1) = 1)

theorem correct_propositions_for_curve_C (k : ℝ) :
  (∀ x y : ℝ, curve_C k) →
  ((∃ k, ((4 - k) * (k - 1) < 0) ↔ (k < 1 ∨ k > 4)) ∧
  ((1 < k ∧ k < (5 : ℝ) / 2) ↔
  (4 - k > k - 1 ∧ 4 - k > 0 ∧ k - 1 > 0))) :=
by {
  sorry
}

end correct_propositions_for_curve_C_l305_305823


namespace fourth_term_of_arithmetic_sequence_l305_305195

open Function

variable {α : Type*} [AddGroup α] [Module ℝ α]

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem fourth_term_of_arithmetic_sequence (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_cond : a 3 + a 5 = 10) :
  a 4 = 5 :=
by
  sorry

end fourth_term_of_arithmetic_sequence_l305_305195


namespace hallie_number_of_paintings_sold_l305_305080

/-- 
Hallie is an artist. She wins an art contest, and she receives a $150 prize. 
She sells some of her paintings for $50 each. 
She makes a total of $300 from her art. 
How many paintings did she sell?
-/
theorem hallie_number_of_paintings_sold 
    (prize : ℕ)
    (price_per_painting : ℕ)
    (total_earnings : ℕ)
    (prize_eq : prize = 150)
    (price_eq : price_per_painting = 50)
    (total_eq : total_earnings = 300) :
    (total_earnings - prize) / price_per_painting = 3 :=
by
  sorry

end hallie_number_of_paintings_sold_l305_305080


namespace difference_of_distinct_members_set_l305_305434

theorem difference_of_distinct_members_set :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≤ n → ∃ a b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), a ≠ b ∧ m = a - b ∨ m = b - a ∧ a > b) :=
by
  sorry

end difference_of_distinct_members_set_l305_305434


namespace johns_father_fraction_l305_305532

theorem johns_father_fraction (total_money : ℝ) (given_to_mother_fraction remaining_after_father : ℝ) :
  total_money = 200 →
  given_to_mother_fraction = 3 / 8 →
  remaining_after_father = 65 →
  ((total_money - given_to_mother_fraction * total_money) - remaining_after_father) / total_money
  = 3 / 10 :=
by
  intros h1 h2 h3
  sorry

end johns_father_fraction_l305_305532


namespace proof_problem_l305_305797

open Real

-- Define the problem statements as Lean hypotheses
def p : Prop := ∀ a : ℝ, exp a ≥ a + 1
def q : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

theorem proof_problem : p ∧ q :=
by
  sorry

end proof_problem_l305_305797


namespace number_of_distinct_positive_differences_is_seven_l305_305423

def set_of_integers := {1, 2, 3, 4, 5, 6, 7, 8}

theorem number_of_distinct_positive_differences_is_seven :
  (set_of_integers \ {0}).image (λ x, set_of_integers \ {x}) |>.nonempty :=
by
  sorry

end number_of_distinct_positive_differences_is_seven_l305_305423


namespace min_days_equal_shifts_l305_305608

theorem min_days_equal_shifts (k n : ℕ) (h : 9 * k + 10 * n = 66) : k + n = 7 :=
sorry

end min_days_equal_shifts_l305_305608


namespace area_diminished_by_64_percent_l305_305343

theorem area_diminished_by_64_percent (L W : ℝ) : 
  let A := L * W 
  let new_L := 0.60 * L 
  let new_W := 0.60 * W 
  let A' := new_L * new_W
  (A - A') / A * 100 = 64 :=
by 
  let A := L * W
  let new_L := 0.60 * L
  let new_W := 0.60 * W
  let A' := new_L * new_W
  have h1 : A = L * W := rfl
  have h2 : new_L = 0.60 * L := rfl
  have h3 : new_W = 0.60 * W := rfl
  have h4 : A' = (0.60 * L) * (0.60 * W) := rfl
  have h5 : (A - A') / A * 100 = (L * W - 0.36 * L * W) / (L * W) * 100 := by rw [h1, h4]
  have h6 : (L * W - 0.36 * L * W) / (L * W) * 100 = (0.64 * L * W) / (L * W) * 100 := by ring_nf
  have h7 : (0.64 * L * W) / (L * W) * 100 = 0.64 * 100 := by rw [mul_div_cancel_left (0.64 * L * W) (L * W)]
  have h8 : 0.64 * 100 = 64 := rfl
  rw [h5, h6, h7, h8]
  sorry -- proof steps are omitted

end area_diminished_by_64_percent_l305_305343


namespace M_equals_all_positive_integers_l305_305550

def M (n : ℕ) : Prop :=
  n ∈ M

-- Conditions as Lean definitions
def prop1 : Prop := M 2018
def prop2 : ∀ (m : ℕ), M m → ∀ d, d ∣ m → M d
def prop3 : ∀ (k m : ℕ), 1 < k → k < m → M k → M m → M (k * m + 1)

-- Theorem statement
theorem M_equals_all_positive_integers 
  (prop1 : prop1) 
  (prop2 : prop2) 
  (prop3 : prop3) : 
  ∀ n, 1 ≤ n → M n :=
sorry

end M_equals_all_positive_integers_l305_305550


namespace dragon_cake_votes_l305_305517

theorem dragon_cake_votes (W U D : ℕ) (x : ℕ) 
  (hW : W = 7) 
  (hU : U = 3 * W) 
  (hD : D = W + x) 
  (hTotal : W + U + D = 60) 
  (hx : x = D - W) : 
  x = 25 := 
by
  sorry

end dragon_cake_votes_l305_305517


namespace valid_b1_count_l305_305913

def sequence_rule (b : ℕ) (n : ℕ) : ℕ :=
  if b % 3 = 0 then b / 3
  else 2 * b + 2

def valid_b (b1 b2 b3 b4 : ℕ) : Prop :=
  b1 < b2 ∧ b1 < b3 ∧ b1 < b4

def valid_b1 (b1 : ℕ) : Prop :=
  valid_b b1 (sequence_rule b1 2) (sequence_rule (sequence_rule b1 2) 3) (sequence_rule (sequence_rule (sequence_rule b1 2) 3) 4)

def counting_valid_b1s (n : ℕ) : ℕ :=
  ∑ i in (Finset.range (n + 1)), if i % 3 ≠ 0 ∧ valid_b1 i then 1 else 0

theorem valid_b1_count (upper_bound : ℕ) :
  counting_valid_b1s upper_bound = 1000 :=
sorry

end valid_b1_count_l305_305913


namespace max_value_of_f_value_of_f_at_theta_plus_pi_over_3_l305_305068

noncomputable def f (x : ℝ) : ℝ := cos x * (sqrt 3 * sin x + cos x)

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ 3 / 2 :=
begin
  -- (Proof corresponding to max value is omitted)
  sorry
end

theorem value_of_f_at_theta_plus_pi_over_3 (θ : ℝ) (h : f (θ / 2) = 3 / 4) :
  f (θ + π / 3) = 7 / 8 :=
begin
  -- (Proof corresponding to this theorem is omitted)
  sorry
end

end max_value_of_f_value_of_f_at_theta_plus_pi_over_3_l305_305068


namespace stickers_total_l305_305672

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l305_305672


namespace range_of_a_l305_305397

theorem range_of_a (a : ℝ) (θ : ℝ)
  (h1 : ∃ θ, cos θ ≤ 0 ∧ sin θ > 0 ∧ θ ∈ {θ | (a-2, a+2) = (cos θ, sin θ)}) :
  -2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l305_305397


namespace probability_rain_at_most_3_days_l305_305627

noncomputable theory

open ProbabilityTheory

def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_rain_at_most_3_days :
  let p := (1 : ℝ) / 20
  let days := 62
  binomial_probability days 0 p +
  binomial_probability days 1 p +
  binomial_probability days 2 p +
  binomial_probability days 3 p ≈ 0.5383 :=
by
  sorry

end probability_rain_at_most_3_days_l305_305627


namespace total_students_l305_305447

theorem total_students (T : ℕ) (h1 : (35 / 100 : ℝ) * T ≠ 546) (h2 : (65 / 100 : ℝ) * T = 546) : T = 840 :=
by
  have h3 : T = 546 / (65 / 100) by
  sorry
  exact h3

end total_students_l305_305447


namespace no_first_or_fourth_quadrant_l305_305811

theorem no_first_or_fourth_quadrant (a b : ℝ) (h : a * b > 0) : 
  ¬ ((∃ x, a * x + b = 0 ∧ x > 0) ∧ (∃ x, b * x + a = 0 ∧ x > 0)) 
  ∧ ¬ ((∃ x, a * x + b = 0 ∧ x < 0) ∧ (∃ x, b * x + a = 0 ∧ x < 0)) := sorry

end no_first_or_fourth_quadrant_l305_305811


namespace problem_solution_l305_305037

theorem problem_solution (n : ℕ) (x : ℕ) (h1 : x = 8^n - 1) (h2 : {d ∈ (nat.prime_divisors x).to_finset | true}.card = 3) (h3 : 31 ∈ nat.prime_divisors x) : x = 32767 :=
sorry

end problem_solution_l305_305037


namespace part1_part2_l305_305416

-- Definitions
def p (m : ℝ) : Prop := ∀ x : ℝ, 4 * m * x^2 + x + m ≤ 0
def q (m : ℝ) : Prop := ∃ x : ℝ, (x ∈ set.Icc 2 8) ∧ (m * Real.log x / Real.log 2 + 1 ≥ 0)

-- The final Lean statements without proof
theorem part1 (m : ℝ) : p m → m ≤ -1 / 4 :=
by sorry

theorem part2 (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m < -1 ∨ m > -1 / 4) :=
by sorry

end part1_part2_l305_305416


namespace arithmetic_geometric_mean_l305_305216

theorem arithmetic_geometric_mean : 
  (A : ℝ) (G : ℝ) (hA : A = (1 + 2) / 2) (hG : G = Real.sqrt (1 * 2)) : A > G :=
sorry

end arithmetic_geometric_mean_l305_305216


namespace find_integer_solution_l305_305003

def sqrt_seq (x : ℕ) : ℕ → ℕ
| 0     := 0 -- This represents s_0 for initialization
| (n+1) := nat.sqrt (x + sqrt_seq n)

theorem find_integer_solution (x y : ℕ) 
  (h1998 : sqrt_seq x 1998 = y) 
  (h_ints : ∀ n, n ≤ 1998 → ∃ k : ℕ, sqrt_seq x n = k) 
  : x = 0 ∧ y = 0 :=
by sorry

end find_integer_solution_l305_305003


namespace solutions_of_quadratic_eq_l305_305220

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l305_305220


namespace final_projection_similar_original_l305_305884

-- Definitions for the problem setup
variable {Pyramid : Type}
variable [IsQuadrilateralPyramid Pyramid] -- Regular quadrilateral (square) pyramid

variable {Triangle : Type}
variable {lateralFace1 base lateralFace2 : Type}
variable [IsPlane lateralFace1]
variable [IsPlane base]
variable [IsPlane lateralFace2]
variable [IsLateralFaceOf lateralFace1 Pyramid]
variable [IsLateralFaceOf lateralFace2 Pyramid]
variable [IsBaseOf base Pyramid]
variable [IsAdjacent lateralFace1 lateralFace2]

variable {T : Triangle}
variable [IsInPlane T lateralFace1]
variable {T1 : Triangle}
variable [ProjectionOf T1 T base]
variable {T2 : Triangle}
variable [ProjectionOf T2 T1 lateralFace2]

-- Theorem to be proven
theorem final_projection_similar_original :
  SimilarTriangles T T2 :=
sorry

end final_projection_similar_original_l305_305884


namespace part_I_part_II_l305_305410

noncomputable def f (a x : ℝ) : ℝ := 
  exp (1 - x) * (-a + cos x)

def f_prime (a x : ℝ) : ℝ := 
  - exp (1 - x) * (sin x + cos x - a)

theorem part_I (a : ℝ) : 
  (∀ x ∈ set.Icc 0 real.pi, f_prime a x ≥ 0) ↔ a ≥ real.sqrt 2 :=
sorry

theorem part_II (a : ℝ) (h : f a (real.pi / 2) = 0) : 
  ∀ x ∈ set.Icc (-1 : ℝ) (1 / 2 : ℝ), 
  f a (-x - 1) + 2 * (f_prime a x) * cos (-x -1) > 0 :=
sorry

end part_I_part_II_l305_305410


namespace choose_students_l305_305278

/-- There are 50 students in the class, including one class president and one vice-president. 
    We want to select 5 students to participate in an activity such that at least one of 
    the class president or vice-president is included. We assert that there are exactly 2 
    distinct methods for making this selection. -/
theorem choose_students (students : Finset ℕ) (class_president vice_president : ℕ) (students_card : students.card = 50)
  (students_ex : class_president ∈ students ∧ vice_president ∈ students) : 
  ∃ valid_methods : Finset (Finset ℕ), valid_methods.card = 2 :=
by
  sorry

end choose_students_l305_305278


namespace difference_of_distinct_members_set_l305_305432

theorem difference_of_distinct_members_set :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≤ n → ∃ a b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), a ≠ b ∧ m = a - b ∨ m = b - a ∧ a > b) :=
by
  sorry

end difference_of_distinct_members_set_l305_305432


namespace distance_covered_by_jeep_l305_305711

def original_time := 3 -- original time in hours
def new_time := 1.5 -- new time in hours (3/2)
def new_speed := 293.3333333333333 -- new speed in km/h

-- Prove that the distance covered by the jeep is 440 km given the conditions
theorem distance_covered_by_jeep :
  ∃ (D : ℝ), D = new_speed * new_time ∧ D = 440 :=
by
  use 440
  split
  · have h : 440 = 293.3333333333333 * 1.5 := rfl
    exact h.symm
  · rfl

-- Example use
#eval distance_covered_by_jeep

end distance_covered_by_jeep_l305_305711


namespace max_area_trough_l305_305371

noncomputable def max_cross_sectional_area (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 4) * a^2

theorem max_area_trough (a : ℝ) :
  ∃ (α : ℝ), α = π / 6 ∧
  let s := a^2 * (1 + Real.sin α) * Real.cos α in
  s = max_cross_sectional_area a := by
  sorry

end max_area_trough_l305_305371


namespace intersection_points_l305_305110

theorem intersection_points (a : ℝ) (h : 2 < a) :
  (∃ n : ℕ, (n = 1 ∨ n = 2) ∧ (∃ x1 x2 : ℝ, y = (a-3)*x^2 - x - 1/4 ∧ x1 ≠ x2)) :=
sorry

end intersection_points_l305_305110


namespace smallest_seating_N_90_chairs_l305_305698

/-- Given a circular table with 90 chairs, prove that the smallest number N of people 
seated such that any new person must sit next to someone already seated is 23. -/
theorem smallest_seating_N_90_chairs : ∃ N, (∀ (new_person_position : ℕ), 
  (new_person_position < 90) → 
   (∃ already_seated_position : ℕ, 
     already_seated_position < 90 ∧
     abs (new_person_position - already_seated_position) <= 1)) → N = 23 :=
by
  sorry

end smallest_seating_N_90_chairs_l305_305698


namespace trig_identity_l305_305317

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end trig_identity_l305_305317


namespace hyperbolas_same_asymptotes_l305_305620

theorem hyperbolas_same_asymptotes (T : ℚ)
  (h1 : ∀ x y : ℚ, (y^2 / 49 - x^2 / 25 = 1) ↔ (y = 7/5 * x ∨ y = -7/5 * x))
  (h2 : ∀ x y : ℚ, (x^2 / T - y^2 / 18 = 1) ↔ (y = sqrt (18/T) * x ∨ y = -sqrt (18/T) * x)) :
  T = 450 / 49 :=
by
  sorry

end hyperbolas_same_asymptotes_l305_305620


namespace find_angleZ_l305_305564

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end find_angleZ_l305_305564


namespace volume_of_one_gram_l305_305981

theorem volume_of_one_gram (mass_per_cubic_meter : ℕ)
  (kilo_to_grams : ℕ)
  (cubic_meter_to_cubic_centimeters : ℕ)
  (substance_mass : mass_per_cubic_meter = 300)
  (kilo_conv : kilo_to_grams = 1000)
  (cubic_conv : cubic_meter_to_cubic_centimeters = 1000000)
  :
  ∃ v : ℝ, v = cubic_meter_to_cubic_centimeters / (mass_per_cubic_meter * kilo_to_grams) ∧ v = 10 / 3 := 
by 
  sorry

end volume_of_one_gram_l305_305981


namespace equal_area_division_through_JT_l305_305115

def Point := (ℕ × ℕ)

def grid_area := 36

def point_P : Point := (3, 3)

def potential_points : list Point :=
  [ (0, 4), (1, 5), (2, 6), (0, 6), (6, 0), (6, 2), (5, 3), (6, 4), (4, 6) ]

def line_through_P (p: Point) : Prop :=
  p.1 = 3 ∨ p.2 = 3 ∨ (p.1 + p.2 = 6) ∨ (p.1 + p.2 = 0)

def divides_when_lines_passing (p1 p2: Point) : Prop :=
  line_through_P p1 ∧ line_through_P p2 ∧
  (p1 = (0, 4) ∧ p2 = (6, 4)) ∨ (p1 = (6, 4) ∧ p2 = (0, 4))

theorem equal_area_division_through_JT : 
  divides_when_lines_passing (0, 4) (6, 4) :=
by
  sorry

end equal_area_division_through_JT_l305_305115


namespace distribute_slip_6_in_cup_J_l305_305297

noncomputable def slips := [1, 1, 1.5, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5, 5.5, 6]

inductive CupLabel 
| F | G | H | I | J | K
  deriving DecidableEq

structure Cup :=
(label : CupLabel)
(contents : List ℝ)

structure Distribution :=
(cups : List Cup)
(sum_integers : ∀ cup : Cup, ∃ n : ℤ, Real.ofInt n = cup.contents.sum)
(non_increasing : ∀ (c1 c2 : Cup), (c1.contents.sum ≤ c2.contents.sum))

theorem distribute_slip_6_in_cup_J (dist : Distribution) (slip4_allocated : Cup) : 
  (∃ (cup_with_6 : Cup), cup_with_6.label = CupLabel.J ∧ 6 ∈ cup_with_6.contents) :=
sorry

end distribute_slip_6_in_cup_J_l305_305297


namespace find_r_minus2_l305_305723

noncomputable def p : ℤ → ℤ := sorry
def r : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom p_minus1 : p (-1) = 2
axiom p_3 : p (3) = 5
axiom p_minus4 : p (-4) = -3

-- Definition of r(x) when p(x) is divided by (x + 1)(x - 3)(x + 4)
axiom r_def : ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * (sorry : ℤ → ℤ) + r x

-- Our goal to prove
theorem find_r_minus2 : r (-2) = 32 / 7 :=
sorry

end find_r_minus2_l305_305723


namespace cos_sum_le_sqrt5_l305_305365

theorem cos_sum_le_sqrt5 (α β γ : ℝ) (h : sin α + sin β + sin γ ≥ 2) :
  cos α + cos β + cos γ ≤ Real.sqrt 5 := 
sorry

end cos_sum_le_sqrt5_l305_305365


namespace median_unchanged_after_removal_l305_305888

noncomputable def median (l : List ℕ) : ℕ :=
if h : l ≠ [] then l.sort (· < ·) ![l.length / 2] else 0

def remove_extremes (l : List ℕ) : List ℕ :=
(l.erase (l.maximum' _)) .erase (l.minimum' _)

theorem median_unchanged_after_removal (scores : List ℕ) (h : scores.length = 7) :
  median (remove_extremes scores) = median scores :=
sorry

end median_unchanged_after_removal_l305_305888


namespace complex_pure_imaginary_l305_305917

/-- Let the complex number (1 + m * I) * (I + 2) be a pure imaginary number, then m = 2. -/
theorem complex_pure_imaginary (m : ℝ) : (∃ (a b : ℝ), (1 + m * Complex.i) * (Complex.i + 2) = Complex.i * b) → m = 2 :=
by 
  sorry

end complex_pure_imaginary_l305_305917


namespace correct_proposition_is_D_l305_305736

theorem correct_proposition_is_D (A B C D : Prop) :
  (∀ (H : Prop), (H = A ∨ H = B ∨ H = C) → ¬H) → D :=
by
  -- We assume that A, B, and C are false.
  intro h
  -- Now we need to prove that D is true.
  sorry

end correct_proposition_is_D_l305_305736


namespace Sn_2017_l305_305380

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
def a1 : Prop := a 1 = 1
def Sn (n : ℕ) : Prop := S n = ∑ i in Finset.range (n+1), a i
def cond (n : ℕ) (h : 2 ≤ n) : Prop := 2 * a n / (a n * S n - S n ^ 2) = 1

-- The goal
theorem Sn_2017 (h1 : a1) (h2 : ∀ n ≥ 2, cond n) : S 2017 = 1 / 1009 := by
  sorry

end Sn_2017_l305_305380


namespace pentagon_rectangle_ratio_l305_305288

theorem pentagon_rectangle_ratio :
  ∀ (p w l : ℝ), 
  5 * p = 20 → 
  2 * (w + l) = 20 →
  l = 2 * w →
  p / w = 6 / 5 :=
by
  intros p w l h₁ h₂ h₃
  have p_value : p = 4 := 
    by linarith
  have w_value : w = 10 / 3 := 
    by linarith
  rw [p_value, w_value]
  norm_num
  sorry

end pentagon_rectangle_ratio_l305_305288


namespace angle_Z_proof_l305_305563

-- Definitions of the given conditions
variables {p q : Type} [Parallel p q]
variables {X Y Z : ℝ}
variables (mAngleX : X = 100)
variables (mAngleY : Y = 130)

-- Statement of the proof problem
theorem angle_Z_proof (hpq : Parallel p q) (hX : X = 100) (hY : Y = 130) : Z = 130 :=
sorry

end angle_Z_proof_l305_305563


namespace g_at_5_l305_305615

noncomputable def g : ℝ → ℝ := sorry

axiom g_axiom1 : ∀ x y : ℝ, g(x * y) = g(x) * g(y)
axiom g_axiom2 : g(0) = 2

theorem g_at_5 : g(5) = 1 := 
by sorry

end g_at_5_l305_305615


namespace prism_height_l305_305891

def vec := (ℝ × ℝ × ℝ)

def dot_product (v₁ v₂ : vec) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

def magnitude (v : vec) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def height_of_prism (AB AC AA1 : vec) : ℝ :=
  let n := (AB.2 * AC.3 - AB.3 * AC.2, AB.3 * AC.1 - AB.1 * AC.3, AB.1 * AC.2 - AB.2 * AC.1)
  in real.abs (dot_product n AA1) / magnitude n

theorem prism_height :
  height_of_prism (0, 1, -1) (1, 4, 0) (1, -1, 4) = real.sqrt 2 / 6 := sorry

end prism_height_l305_305891


namespace sufficient_but_not_necessary_condition_l305_305998

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l305_305998


namespace mean_and_variance_of_y_l305_305417

noncomputable def sample_data (x : Fin 10 → ℝ) (mean variance : ℝ) :=
  (∑ i, x i / 10 = mean) ∧ (∑ i, (x i - mean) ^ 2 / 10 = variance)

variable {a : ℝ}
variable {x : Fin 10 → ℝ}
hypothesis hx_mean : ∑ i, x i / 10 = 2
hypothesis hx_variance : ∑ i, (x i - 2) ^ 2 / 10 = 5
variable {y : Fin 10 → ℝ} := (fun i => x i + a)

theorem mean_and_variance_of_y :
  (∑ i, y i / 10 = 2 + a) ∧ (∑ i, (y i - (2 + a)) ^ 2 / 10 = 5) :=
 by {
   sorry
 }

end mean_and_variance_of_y_l305_305417


namespace rabbit_total_distance_l305_305648

theorem rabbit_total_distance 
  (r₁ r₂ : ℝ) 
  (h1 : r₁ = 7) 
  (h2 : r₂ = 15) 
  (q : ∀ (x : ℕ), x = 4) 
  : (3.5 * π + 8 + 7.5 * π + 8 + 3.5 * π + 8) = 14.5 * π + 24 := 
by
  sorry

end rabbit_total_distance_l305_305648


namespace multiply_fractions_l305_305242

theorem multiply_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = (1 / 7) := by
  sorry

end multiply_fractions_l305_305242


namespace root_intervals_of_quadratic_l305_305850

theorem root_intervals_of_quadratic (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x1 x2 : ℝ, (x1 ∈ set.Ioo a b) ∧ (x2 ∈ set.Ioo b c) ∧
               (∀ x ∈ set.Ioo a c, (x = x1 ∨ x = x2) → f x = 0) :=
by
  let f := λ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)
  sorry

end root_intervals_of_quadratic_l305_305850


namespace find_a_for_line_and_hyperbola_intersection_l305_305395

theorem find_a_for_line_and_hyperbola_intersection (a : ℝ) :
  (∀ x y : ℝ, (y = a * x + 1) ∧ (3 * x^2 - y^2 = 1) → ∃ A B : ℝ × ℝ,
    line_segment A B ∧ on_circle_through_origin (circle_diameter_through_origin A B)) →
  (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_for_line_and_hyperbola_intersection_l305_305395


namespace johns_family_total_members_l305_305533

theorem johns_family_total_members (n_f : ℕ) (h_f : n_f = 10) (n_m : ℕ) (h_m : n_m = (13 * n_f) / 10) :
  n_f + n_m = 23 := by
  rw [h_f, h_m]
  norm_num
  sorry

end johns_family_total_members_l305_305533


namespace nathans_blanket_temperature_l305_305579

theorem nathans_blanket_temperature :
  let initial_temp := 50
  let type_A_count := 8
  let type_B_count := 6
  let type_C_count := 4
  let type_A_increase := 2
  let type_B_increase := 3
  let type_C_increase := 5
  let used_A_count := type_A_count / 2
  let total_A_increase := used_A_count * type_A_increase
  let total_B_increase := type_B_count * type_B_increase
  final_temp = initial_temp + total_A_increase + total_B_increase 
  in final_temp = 76
  :=
  sorry

end nathans_blanket_temperature_l305_305579


namespace tan_C_correct_l305_305526

noncomputable def tan_C (AD BC E F: Type) 
  (h1: AD ∥ BC) (h2: E ∈ AD) (h3: F ∈ BC)
  (h4: AE = 2 • ED) (h5: BF = 2 • FC)
  (h6: EF ⊥ BC) (h7: ∠B = 60°) : Real :=
2

theorem tan_C_correct (AD BC E F: Type) 
  (h1: AD ∥ BC) (h2: E ∈ AD) (h3: F ∈ BC)
  (h4: AE = 2 • ED) (h5: BF = 2 • FC)
  (h6: EF ⊥ BC) (h7: ∠B = 60°) :
  tan_C AD BC E F h1 h2 h3 h4 h5 h6 h7 = 2 :=
sorry

end tan_C_correct_l305_305526


namespace average_cars_given_per_year_l305_305582

/-- Definition of initial conditions and the proposition -/
def initial_cars : ℕ := 3500
def final_cars : ℕ := 500
def years : ℕ := 60

theorem average_cars_given_per_year : (initial_cars - final_cars) / years = 50 :=
by
  sorry

end average_cars_given_per_year_l305_305582


namespace puppy_food_total_correct_l305_305929

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l305_305929


namespace no_x_intersections_geometric_sequence_l305_305054

theorem no_x_intersections_geometric_sequence (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a * c > 0) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) = false :=
by
  sorry

end no_x_intersections_geometric_sequence_l305_305054


namespace John_reads_Bible_in_4_weeks_l305_305903

def daily_reading_pages (hours_per_day reading_rate : ℕ) : ℕ :=
  hours_per_day * reading_rate

def weekly_reading_pages (daily_pages days_in_week : ℕ) : ℕ :=
  daily_pages * days_in_week

def weeks_to_finish (total_pages daily_pages : ℕ) : ℕ :=
  total_pages / daily_pages

theorem John_reads_Bible_in_4_weeks
  (hours_per_day : ℕ : 2)
  (reading_rate : ℕ := 50)
  (bible_pages : ℕ := 2800)
  (days_in_week : ℕ := 7) :
  weeks_to_finish bible_pages (weekly_reading_pages (daily_reading_pages hours_per_day reading_rate) days_in_week) = 4 :=
  sorry

end John_reads_Bible_in_4_weeks_l305_305903


namespace hypotenuse_length_l305_305480

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305480


namespace students_ages_average_l305_305452

variables (a b c : ℕ)

theorem students_ages_average (h1 : (14 * a + 13 * b + 12 * c) = 13 * (a + b + c)) : a = c :=
by
  sorry

end students_ages_average_l305_305452


namespace gcd_of_squares_l305_305655

theorem gcd_of_squares :
  gcd (168^2 + 301^2 + 502^2) (169^2 + 300^2 + 501^2) = 1 := by
  let m := 168^2 + 301^2 + 502^2
  let n := 169^2 + 300^2 + 501^2
  have h : gcd m n = gcd (n - m) m := gcd_sub m n
  -- Proceeding with further simplifications would require more steps.
  -- As the proof is noncomputable or complex beyond simplifications, use sorry
  sorry

end gcd_of_squares_l305_305655


namespace next_podcast_length_l305_305176

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end next_podcast_length_l305_305176


namespace find_xyz_l305_305446

theorem find_xyz
  {x y z : ℕ}
  (h1 : x * y = 24 * (4 : ℝ) ^ (1 / 4))
  (h2 : x * z = 42 * (4 : ℝ) ^ (1 / 4))
  (h3 : y * z = 21 * (4 : ℝ) ^ (1 / 4))
  (h4 : x < y)
  (h5 : y < z) :
  x * y * z = 291.2 := 
sorry

end find_xyz_l305_305446


namespace surface_area_is_726_l305_305783

def edge_length : ℝ := 11

def surface_area_of_cube (e : ℝ) : ℝ := 6 * (e * e)

theorem surface_area_is_726 (h : edge_length = 11) : surface_area_of_cube edge_length = 726 := by
  sorry

end surface_area_is_726_l305_305783


namespace tara_additional_down_payment_l305_305601

noncomputable def laptop_price := 1000
noncomputable def down_payment_rate := 0.20
noncomputable def monthly_installment := 65
noncomputable def months_paid := 4
noncomputable def remaining_balance := 520

-- Definition of the total down payment as per 20% of the laptop price.
noncomputable def required_down_payment := laptop_price * down_payment_rate

-- Total amount paid through installments after 4 months.
noncomputable def total_installments_paid := monthly_installment * months_paid

-- Total amount Tara has paid including down payment and installments after 4 months.
noncomputable def total_paid := laptop_price - remaining_balance

-- Expected total without additional down payment (initial required down payment & 4 months of installments).
noncomputable def expected_total := required_down_payment + total_installments_paid

-- Additional amount Tara paid for the down payment.
noncomputable def additional_down_payment := total_paid - expected_total

-- Create the proof statement
theorem tara_additional_down_payment : additional_down_payment = 20 := 
by
  unfold additional_down_payment
  unfold total_paid
  unfold expected_total
  unfold required_down_payment
  unfold total_installments_paid
  unfold laptop_price
  unfold down_payment_rate
  unfold monthly_installment
  unfold months_paid
  unfold remaining_balance
  simp
  sorry

end tara_additional_down_payment_l305_305601


namespace average_age_of_4_students_l305_305969

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end average_age_of_4_students_l305_305969


namespace round_robin_tournament_l305_305874

theorem round_robin_tournament :
  let n := 9 in
  ∃ S : finset (finset (fin 9)),
    (∀ s ∈ S, s.card = 4) ∧
    (∀ s ∈ S, ∃ (A B C D : fin 9),
      s = {A, B, C, D} ∧
      (A, B) ∈ beating_pairs ∧
      (B, C) ∈ beating_pairs ∧
      (C, D) ∈ beating_pairs ∧
      (D, A) ∈ beating_pairs) ∧
    S.card = 756 :=
sorry

end round_robin_tournament_l305_305874


namespace ratio_of_x_to_y_l305_305336

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 3 / 5) : x / y = 16 / 15 :=
sorry

end ratio_of_x_to_y_l305_305336


namespace min_value_of_expression_l305_305915

theorem min_value_of_expression
  (a b c t : ℝ)
  (h1 : a + b + c = t)
  (h2 : a^2 + b^2 + c^2 = 1) :
  ∃ t, t = 2 ∧ t ≤ sqrt 3 := by
sorry

end min_value_of_expression_l305_305915


namespace solve_equation_l305_305188

theorem solve_equation : ∃ x : ℝ, 4 * x - 2 = 2 * (x + 2) ∧ x = 3 :=
by
  use 3
  split
  sorry

end solve_equation_l305_305188


namespace min_distance_curve_line_l305_305010

theorem min_distance_curve_line : 
  let M := (1, 0) in
  let line_eq := λ x : ℝ, 2 * x + 3 in
  let dist_point_line := 
    λ (x1 y1 : ℝ) (A B C : ℝ), |A * x1 + B * y1 + C| / Real.sqrt (A^2 + B^2) in
  dist_point_line 1 0 2 (-1) 3 = Real.sqrt 5 := 
by
  sorry

end min_distance_curve_line_l305_305010


namespace two_numbers_max_product_l305_305238

theorem two_numbers_max_product :
  ∃ x y : ℝ, x - y = 4 ∧ x + y = 35 ∧ ∀ z w : ℝ, z - w = 4 → z + w = 35 → z * w ≤ x * y :=
by
  sorry

end two_numbers_max_product_l305_305238


namespace at_least_1991_red_points_l305_305193

theorem at_least_1991_red_points (P : Fin 997 → ℝ × ℝ) :
  ∃ (R : Finset (ℝ × ℝ)), 1991 ≤ R.card ∧ (∀ (i j : Fin 997), i ≠ j → ((P i + P j) / 2) ∈ R) :=
sorry

end at_least_1991_red_points_l305_305193


namespace hypotenuse_length_l305_305482

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l305_305482


namespace mark_goal_remainder_l305_305313

-- Definitions aligning with conditions
def total_quizzes := 60
def goal_percentage := 0.85
def quizzes_taken := 40
def quizzes_A_in_first_40 := 30
def remaining_quizzes := total_quizzes - quizzes_taken
def total_A_quizzes_needed := goal_percentage * total_quizzes
def additional_A_needed := total_A_quizzes_needed - quizzes_A_in_first_40

-- Theorem statement
theorem mark_goal_remainder :
  additional_A_needed ≤ remaining_quizzes ↔ 0 :=
by
  unfold total_quizzes goal_percentage quizzes_taken quizzes_A_in_first_40 remaining_quizzes total_A_quizzes_needed additional_A_needed
  sorry

end mark_goal_remainder_l305_305313


namespace area_of_triangle_l305_305370

noncomputable def area_triangle (r α : ℝ) : ℝ :=
  r^2 * (Real.cos (α / 2))^2 * Real.cot (α / 2)

theorem area_of_triangle (A B C : Point) (r : ℝ) (a b c : ℝ) (α : ℝ) (hA_ext : A ∉ Circle r)
  (hTangent_AB : isTangent AB (Circle r) A B)
  (hTangent_AC : isTangent AC (Circle r) A C)
  (hAngle : ∠BAC = α) :
  area_triangle r α = r^2 * (Real.cos (α / 2))^2 * Real.cot (α / 2) :=
by
  sorry

end area_of_triangle_l305_305370


namespace cosine_of_angle_between_adjacent_faces_l305_305729

-- Define necessary elements and relationships in Lean 4.
variables (S A B C O L D : Type) 
variables (a : ℝ) -- Define the side length of the base

-- Define that the triangle is regular and the pyramid is regular
class RegularTriangularPyramid (S A B C : Type) :=
  (height_S0 : O)
  (midpoint_L : L)
  (OL_eq_a : OL = a)

def cosine_adjacent_faces [RegularTriangularPyramid S A B C] : ℝ := 
  (7 / 15)

-- The theorem to be proven
theorem cosine_of_angle_between_adjacent_faces [RegularTriangularPyramid S A B C] :
  cosine_adjacent_faces = (7 / 15) :=
by
  sorry


end cosine_of_angle_between_adjacent_faces_l305_305729


namespace recruit_people_l305_305954

variable (average_contribution : ℝ) (total_funds_needed : ℝ) (current_funds : ℝ)

theorem recruit_people (h₁ : average_contribution = 10) (h₂ : current_funds = 200) (h₃ : total_funds_needed = 1000) : 
    (total_funds_needed - current_funds) / average_contribution = 80 := by
  sorry

end recruit_people_l305_305954


namespace irrational_terms_probability_correct_l305_305773

-- Define the binomial expansion and the concept of rational and irrational terms.
def binomial_expansion (a b : ℝ) (n : ℕ) : List (ℝ × ℝ) :=
  List.range (n + 1).map (λ r, (Real.binom n r * a^(n - r) * b^r, r))

-- The given binomial to expand
def given_binomial_expansion : List (ℝ × ℝ) :=
  binomial_expansion 1 (2 / Real.sqrt 1) 6

-- Define a predicate to check if a term is rational.
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, (q : ℝ) = x

-- Define the irrational term count and their adjacency probability.
def irrational_terms_not_adjacent_probability : ℝ :=
  let irrational_terms := given_binomial_expansion.filter (λ term, ¬ is_rational term.1) in
  if irrational_terms.length = 3 then 2 / 7 else 0

-- The main theorem to be proven
theorem irrational_terms_probability_correct :
  irrational_terms_not_adjacent_probability = 2 / 7 := 
sorry

end irrational_terms_probability_correct_l305_305773


namespace gemstones_difference_is_2_l305_305330

-- Given conditions:
-- Binkie has 24 gemstones on his collar
def gemstones_Binkie : ℕ := 24

-- Binkie has four times as many gemstones as Frankie
def gemstones_Frankie (gems_Binkie : ℕ) : ℕ := gems_Binkie / 4

-- Spaatz has 1 gemstone on her collar
def gemstones_Spaatz : ℕ := 1

-- Question: prove the difference between the number of gemstones on Spaatz's collar and half the number on Frankie's collar is 2
theorem gemstones_difference_is_2 : (let half_gems_Frankie := gemstones_Frankie gemstones_Binkie / 2 in
  half_gems_Frankie - gemstones_Spaatz = 2) :=
by
  sorry

end gemstones_difference_is_2_l305_305330


namespace sequence_a_n_sequence_b_n_range_k_l305_305822

-- Define the geometric sequence {a_n} with initial conditions
def a (n : ℕ) : ℕ :=
  3 * 2^(n-1)

-- Define the sequence {b_n} with the given recurrence relation
def b : ℕ → ℕ
| 0 => 1
| (n+1) => 2 * (b n) + 1

theorem sequence_a_n (n : ℕ) : 
  (a n = 3 * 2^(n-1)) := sorry

theorem sequence_b_n (n : ℕ) :
  (b n = 2^n - 1) := sorry

-- Define the condition for k and the inequality
def condition_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (k * (↑(b n) + 5) / 2 - 3 * 2^(n-1) ≥ 8*n + 2*k - 24)

-- Prove the range for k
theorem range_k (k : ℝ) :
  (condition_k k ↔ k ≥ 4) := sorry

end sequence_a_n_sequence_b_n_range_k_l305_305822


namespace sequence_is_periodic_l305_305907
noncomputable theory

def periodic_sequence (seq : ℕ → ℤ) : Prop :=
  ∃ p > 0, ∀ n, seq (n + p) = seq n

theorem sequence_is_periodic
  (a : ℕ → ℤ)
  (h : ∀ n ≥ 2, 0 ≤ a (n-1) + ((1 - Real.sqrt 5) / 2) * a n + a (n+1) ∧ a (n-1) + ((1 - Real.sqrt 5) / 2) * a n + a (n+1) < 1) :
  periodic_sequence a :=
sorry

end sequence_is_periodic_l305_305907


namespace minimum_value_of_m_plus_n_l305_305028

noncomputable def m (a b : ℝ) : ℝ := b + (1 / a)
noncomputable def n (a b : ℝ) : ℝ := a + (1 / b)

theorem minimum_value_of_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 1) :
  m a b + n a b = 4 :=
sorry

end minimum_value_of_m_plus_n_l305_305028


namespace jonah_total_ingredients_in_cups_l305_305905

noncomputable def volume_of_ingredients_in_cups : ℝ :=
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  let almonds_in_ounces := 5.5
  let pumpkin_seeds_in_grams := 150
  let ounce_to_cup_conversion := 0.125
  let gram_to_cup_conversion := 0.00423
  let almonds := almonds_in_ounces * ounce_to_cup_conversion
  let pumpkin_seeds := pumpkin_seeds_in_grams * gram_to_cup_conversion
  yellow_raisins + black_raisins + almonds + pumpkin_seeds

theorem jonah_total_ingredients_in_cups : volume_of_ingredients_in_cups = 2.022 :=
by
  sorry

end jonah_total_ingredients_in_cups_l305_305905


namespace concurrency_of_circumcenters_l305_305947

open EuclideanGeometry

/-- Given a quadrilateral ABCD inscribed in a circle O, with diagonals AC and BD intersecting at P,
and the circumcenters of triangles ABP, BCP, CDP, and DAP being O1, O2, O3, and O4 respectively,
prove that lines OP, O1O3, and O2O4 are concurrent. -/
theorem concurrency_of_circumcenters 
  (A B C D P O O1 O2 O3 O4 : Point) 
  (h_cyclic : Cyclic ABCD)
  (h_AC_BD_intersect : Intersects AC BD at P)
  (h_circumcenter_ABP : Circumcenter ABP O1)
  (h_circumcenter_BCP : Circumcenter BCP O2)
  (h_circumcenter_CDP : Circumcenter CDP O3)
  (h_circumcenter_DAP : Circumcenter DAP O4) :
  Concurrent OP O1O3 O2O4 := sorry

end concurrency_of_circumcenters_l305_305947


namespace find_AX_l305_305883

-- Definitions for the conditions given in the problem
variables {A B C X : Type} [real_space A] [real_space B] [real_space C] [real_space X]
variable (AB AC BC AX BX : ℝ)

-- Conditions
axiom h1 : AB = 72
axiom h2 : AC = 40
axiom h3 : BC = 80
axiom h4 : AB = AX + BX
axiom h5 : BX = 2 * AX

-- Proof statement: Prove AX = 24
theorem find_AX : AX = 24 :=
by
  sorry

end find_AX_l305_305883


namespace Pam_current_balance_l305_305607

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end Pam_current_balance_l305_305607


namespace monotonic_decreasing_range_l305_305614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ -1 :=
  sorry

end monotonic_decreasing_range_l305_305614


namespace value_of_x_squared_plus_y_squared_l305_305854

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l305_305854


namespace largest_term_in_first_30_l305_305418

noncomputable def a (n : ℕ) : ℝ := (n - real.sqrt 98) / (n - real.sqrt 99)

theorem largest_term_in_first_30 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 → a 10 ≥ a n :=
by {
  sorry
}

end largest_term_in_first_30_l305_305418


namespace angle_between_lines_l305_305872

section QuadrilateralAngle
variable (A B C D : ℝ × ℝ)

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem angle_between_lines (hAB : side_length A B = 4)
                            (hCD : side_length C D = 6)
                            (hMidDist : side_length (midpoint B D) (midpoint A C) = 3)
                           : ∃ α : ℝ, α = real.arccos (1 / 3) :=
by
  sorry
end QuadrilateralAngle

end angle_between_lines_l305_305872


namespace total_cats_in_center_l305_305741

theorem total_cats_in_center (J F S : Finset ℤ) 
  (h_jump: J.card = 60)
  (h_jump_fetch: (J ∩ F).card = 25)
  (h_fetch: F.card = 40)
  (h_fetch_spin: (F ∩ S).card = 20)
  (h_spin: S.card = 50)
  (h_jump_spin: (J ∩ S).card = 30)
  (h_all_three: (J ∩ F ∩ S).card = 15)
  (h_none: (@Finset.univ ℤ _).card - (J ∪ F ∪ S).card = 5) :
  (@Finset.univ ℤ _).card = 95 :=
by
  sorry

end total_cats_in_center_l305_305741


namespace average_monthly_growth_rate_l305_305626

-- Define the conditions
variables (P : ℝ) (r : ℝ)
-- The condition that output in December is P times that of January
axiom growth_rate_condition : (1 + r)^11 = P

-- Define the goal to prove the average monthly growth rate
theorem average_monthly_growth_rate : r = (P^(1/11) - 1) :=
by
  sorry

end average_monthly_growth_rate_l305_305626


namespace sum_of_sequences_is_43_l305_305820

theorem sum_of_sequences_is_43
  (A B C D : ℕ)
  (hA_pos : 0 < A)
  (hB_pos : 0 < B)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D)
  (h_arith : A + (C - B) = B)
  (h_geom : C = (4 * B) / 3)
  (hD_def : D = (4 * C) / 3) :
  A + B + C + D = 43 :=
sorry

end sum_of_sequences_is_43_l305_305820


namespace hypotenuse_length_l305_305490

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305490


namespace approx_root_bisection_method_l305_305664

theorem approx_root_bisection_method (f : ℝ → ℝ) 
  (h1 : f 0.64 < 0) 
  (h2 : f 0.72 > 0) 
  (h3 : f 0.68 < 0) : 
  0.68 < 0.7 ∧ 0.7 < 0.72 :=
begin
  sorry
end

end approx_root_bisection_method_l305_305664


namespace sum_of_first_49_primes_is_10787_l305_305357

def first_49_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                                79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                                167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]

theorem sum_of_first_49_primes_is_10787:
  first_49_primes.sum = 10787 :=
by
  -- Proof would go here.
  -- This is just a placeholder as per the requirements.
  sorry

end sum_of_first_49_primes_is_10787_l305_305357


namespace probability_same_num_of_flips_l305_305642

theorem probability_same_num_of_flips (h: ℝ) (t: ℝ)
    (prob_head: h = 1 / 3) (prob_tail: t = 2 / 3) :
    (∑' n : ℕ, (t ^ (3 * n - 3)) * (h ^ 3)) = 1 / 19 :=
by
  have a := (1 : ℝ) / 27
  have r := (8 : ℝ) / 27
  have sum_geometric_series : (∑' n : ℕ, r ^ n) = 1 / (1 - r)
    by sorry
  calc
    (∑' n : ℕ, (t ^ (3 * n - 3)) * (h ^ 3))
        = a * (∑' n : ℕ, r ^ n) : by sorry
    ... = a * (27 / 19) : by sorry
    ... = 1 / 19 : by sorry
  sorry
 
end probability_same_num_of_flips_l305_305642


namespace correct_option_C_l305_305257

def is_linear_equation_in_two_variables (eq: String) : Prop :=
  -- This definition checks if an equation is linear in two variables format,
  -- the implementation detail is abstracted here for simplicity
  sorry

def system_of_linear_equations : String -> Prop
| "A" := is_linear_equation_in_two_variables "x + y = 5" ∧ is_linear_equation_in_two_variables "1/x + 1/y = 3"
| "B" := is_linear_equation_in_two_variables "x + y = 6" ∧ is_linear_equation_in_two_variables "y + z = 7"
| "C" := is_linear_equation_in_two_variables "x = 3" ∧ is_linear_equation_in_two_variables "2x - y = 7"
| "D" := is_linear_equation_in_two_variables "x - 1 = xy" ∧ is_linear_equation_in_two_variables "x - y = 0"

theorem correct_option_C : system_of_linear_equations "C" := 
  by {
    -- The proof should verify that Option C satisfies the criteria
    sorry
  }

end correct_option_C_l305_305257


namespace geom_sequence_common_ratio_l305_305197

-- We introduce the common ratio 'q' in the geometric sequence
theorem geom_sequence_common_ratio 
  (a : ℝ) 
  (log2_3 : ℝ) 
  (h_log4_3 : log2_3 / log2 4 = log2_3 / 2)
  (h_log8_3 : log2_3 / log2 8 = log2_3 / 3) : 
  let q := (a + log2_3 / 2) / (a + log2_3) in
  True := 
  sorry

end geom_sequence_common_ratio_l305_305197


namespace hypotenuse_length_l305_305494

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l305_305494


namespace min_cards_to_flip_l305_305581

-- Definitions of the cards based on their visible faces.
inductive Face
| yellow
| black
| happy_smiley
| sad_smiley

def cards : List (Face × Face) :=
[(Face.yellow, Face.sad_smiley), -- First card
 (Face.black, Face.happy_smiley), -- Second card
 (Face.happy_smiley, Face.yellow), -- Third card
 (Face.sad_smiley, Face.black)] -- Fourth card

-- The statement we need to check: "If there is a happy smiley on one side of a card, then the other side is painted yellow."
def statement (f1 f2 : Face) : Prop :=
match f1, f2 with
| Face.happy_smiley, Face.yellow => True
| Face.happy_smiley, _ => False
| _, _ => True

-- Checking the minimal number of cards to flip.
theorem min_cards_to_flip : ∀ c1 c2 c3 c4 : Face × Face,
  c1 = (Face.yellow, Face.sad_smiley) →
  c2 = (Face.black, Face.happy_smiley) →
  c3 = (Face.happy_smiley, Face.yellow) →
  c4 = (Face.sad_smiley, Face.black) →
  (statement (c2.2) (c2.1) ∨ statement (c3.1) (c3.2)) →
  (c2 = (Face.black, Face.happy_smiley) ∨ c3 = (Face.happy_smiley, Face.yellow)) →
  ∑ (b : Bool) (h : b = (statement (c3.1) (c3.2))) : ℕ = 2 :=
by
  intros c1 c2 c3 c4 h1 h2 h3 h4 h5 h6
  sorry

end min_cards_to_flip_l305_305581


namespace distinct_differences_count_l305_305435

-- Define the set of interest.
def mySet : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- The statement we want to prove.
theorem distinct_differences_count : 
  (finset.image (λ (x : ℕ × ℕ), (x.1 - x.2)) ((mySet.product mySet).filter (λ x, x.1 > x.2))).card = 7 :=
sorry

end distinct_differences_count_l305_305435


namespace not_possible_degrees_l305_305979

theorem not_possible_degrees (G : SimpleGraph (Fin 19)) (h : ∀ v, G.degree v = 1 ∨ G.degree v = 5 ∨ G.degree v = 9) :
  False :=
by
  sorry

end not_possible_degrees_l305_305979


namespace integer_solutions_system_l305_305523

theorem integer_solutions_system (x y z : ℤ) :
  z^x = y^{2 * x} ∧
  2^z = 2 * 4^x ∧
  x + y + z = 16 →
  (x = 4 ∧ y = 3 ∧ z = 9) :=
by
  intro h
  sorry

end integer_solutions_system_l305_305523


namespace DongfangElementary_total_students_l305_305338

theorem DongfangElementary_total_students (x y : ℕ) 
  (h1 : x = y + 2)
  (h2 : 10 * (y + 2) = 22 * 11 * (y - 22))
  (h3 : x - x / 11 = 2 * (y - 22)) :
  x + y = 86 :=
by
  sorry

end DongfangElementary_total_students_l305_305338


namespace exp_gt_one_add_x_l305_305250

theorem exp_gt_one_add_x {x : ℝ} (h : x ≠ 0) : exp x > 1 + x :=
begin
  sorry
end

end exp_gt_one_add_x_l305_305250


namespace maximizing_nine_pairs_l305_305241

def list_of_integers := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def sum_is_nine (a b : Int) : Prop := a + b = 9

def choose_two_distinct (lst : List Int) : List (Int × Int) :=
  lst.product lst.filter (λ (a, b), a < b)

def sum_pairs (lst : List Int) : List (Int × Int) :=
  (choose_two_distinct lst).filter (λ (a, b), sum_is_nine a b)

theorem maximizing_nine_pairs :
  (∀ x ∈ list_of_integers, length (sum_pairs (list_of_integers.erase x)) ≤ length (sum_pairs (list_of_integers.erase (-2)))) :=
sorry

end maximizing_nine_pairs_l305_305241


namespace find_angleZ_l305_305565

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end find_angleZ_l305_305565


namespace prime_factor_tau_divides_l305_305364

-- Definitions for number of divisors and sum of divisors
def tau (n : ℕ) : ℕ := n.divisors.count
def sigma (n : ℕ) : ℕ := n.divisors.sum id

-- The given conditions
variables (a b : ℕ)
condition : ∀ n : ℕ, sigma (a^n) ∣ sigma (b^n)

-- Prove that each prime factor of tau(a) divides tau(b)
theorem prime_factor_tau_divides (a b : ℕ) (H : ∀ n : ℕ, sigma (a^n) ∣ sigma (b^n)) :
  ∀ p : ℕ, p.prime → p ∣ tau(a) → p ∣ tau(b) :=
sorry

end prime_factor_tau_divides_l305_305364


namespace john_read_bible_in_weeks_l305_305902

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end john_read_bible_in_weeks_l305_305902


namespace increasing_sequence_range_l305_305075

theorem increasing_sequence_range (a : ℝ) (a_seq : ℕ → ℝ)
  (h₁ : ∀ (n : ℕ), n ≤ 5 → a_seq n = (5 - a) * n - 11)
  (h₂ : ∀ (n : ℕ), n > 5 → a_seq n = a ^ (n - 4))
  (h₃ : ∀ (n : ℕ), a_seq n < a_seq (n + 1)) :
  2 < a ∧ a < 5 := 
sorry

end increasing_sequence_range_l305_305075


namespace Diego_annual_savings_l305_305342

theorem Diego_annual_savings :
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year
  annual_savings = 4800 :=
by
  -- Definitions based on the conditions and required result
  let monthly_income := 5000
  let monthly_expenses := 4600
  let monthly_savings := monthly_income - monthly_expenses
  let months_in_year := 12
  let annual_savings := monthly_savings * months_in_year

  -- Assertion to check the correctness of annual savings
  have h : annual_savings = 4800 := by
    have h1 : monthly_savings = monthly_income - monthly_expenses := rfl
    have h2 : monthly_savings = 400 := by simp [monthly_income, monthly_expenses, h1]
    have h3 : annual_savings = monthly_savings * months_in_year := rfl
    simp [h2, months_in_year, h3]
  exact h

end Diego_annual_savings_l305_305342


namespace m_not_perfect_square_l305_305542

theorem m_not_perfect_square (m : ℕ) (h : ∀ d ∈ digits 10 m, d = 0 ∨ d = 6) : ¬ is_square m :=
sorry

end m_not_perfect_square_l305_305542


namespace polynomial_divisible_l305_305156

theorem polynomial_divisible 
  (f : ℂ[X]) 
  (h1 : f ≠ polynomial.X) 
  (n : ℕ) 
  (F_n : ℂ[X] := λ x, polynomial.eval x (f^(n+1) - polynomial.X)) :
  ∀ x : ℂ, f.eval x - x ∣ F_n x :=
sorry

end polynomial_divisible_l305_305156


namespace solve_system_l305_305348

theorem solve_system : ∃ x y : ℤ, 3 * x = -9 - 3 * y ∧ 2 * x = 3 * y - 22 ∧ x = -5 ∧ y = 2 :=
by
  use [-5, 2]
  split
  { -- Proving 3x = -9 - 3y
    calc
      3 * -5 = -15 : by norm_num
            ... = -9 - 3 * 2 : by norm_num
  }
  split
  { -- Proving 2x = 3y - 22
    calc
      2 * -5 = -10 : by norm_num
            ... = 3 * 2 - 22 : by norm_num
  }
  split
  { -- Proving x = -5
    rfl
  }
  { -- Proving y = 2
    rfl
  }


end solve_system_l305_305348


namespace locus_of_A_in_parallelogram_inside_K_l305_305029

theorem locus_of_A_in_parallelogram_inside_K
  (K : Type*)
  [metric_space K]
  (c : K) -- center of circle K
  (r : ℝ) -- radius of circle K
  (A : K) -- vertex A in parallelogram ABCD
  (B D : K) -- vertices B and D in parallelogram ABCD
  (BD_inside_K : dist B D ≤ 2 * r) -- condition BD inside circle K
  (AC_le_BD : dist A c ≤ dist B D) : 
  dist A c = sqrt 2 * r :=
sorry

end locus_of_A_in_parallelogram_inside_K_l305_305029


namespace wire_length_l305_305261

-- Definitions for the given conditions
def area_of_square := 69696
def side_length_of_square := Real.sqrt area_of_square
def perimeter_of_square := 4 * side_length_of_square
def total_length_of_wire := 15 * perimeter_of_square

-- Lean statement for the proof problem
theorem wire_length (h : area_of_square = 69696) : total_length_of_wire = 15840 := by
  sorry

end wire_length_l305_305261


namespace trigonometric_intersection_varphi_l305_305830

theorem trigonometric_intersection_varphi (phi : ℝ) (h_phi : 0 ≤ phi ∧ phi < π) :
  cos (π / 3) = sin (2 * (π / 3) + phi) → phi = π / 6 :=
by
  intros h
  sorry

end trigonometric_intersection_varphi_l305_305830


namespace tunnel_crossing_time_l305_305682

-- Define the walking times for each friend
def t1 := 1
def t2 := 2
def t5 := 5
def t10 := 10

-- Define a function that takes two friends' time and returns the slower one's time
def walk_time (a b : ℕ) : ℕ := max a b

theorem tunnel_crossing_time : ∃ (schedule : list (list ℕ)), 
  let total_time := 
    2 + 1 + 10 + 2 + 2 in
  total_time = 17 :=
begin
  use [[t1, t2], [t1], [t5, t10], [t2], [t1, t2]],
  simp [walk_time],
  norm_num,
end

end tunnel_crossing_time_l305_305682


namespace system_solution_equation_solution_l305_305189

-- Proof problem for the first system of equations
theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = 8) ∧ (3 * x - 5 * y = -7) → (x = 1 ∧ y = 2) :=
by sorry

-- Proof problem for the second equation
theorem equation_solution (x : ℝ) : 
  ((x - 2) / (x + 2) - 12 / (x^2 - 4) = 1) → (x = -1) :=
by sorry

end system_solution_equation_solution_l305_305189


namespace number_of_squares_in_10x10_checkerboard_l305_305754

theorem number_of_squares_in_10x10_checkerboard : 
  let total_squares := ∑ k in finset.range 11, k * k in
  total_squares = 385 :=
by
  sorry

end number_of_squares_in_10x10_checkerboard_l305_305754


namespace sam_bought_9_cans_l305_305179

-- Definitions based on conditions
def spent_amount_dollars := 20 - 5.50
def spent_amount_cents := 1450 -- to avoid floating point precision issues we equate to given value in cents
def coupon_discount_cents := 5 * 25
def total_cost_no_discount := spent_amount_cents + coupon_discount_cents
def cost_per_can := 175

-- Main statement to prove
theorem sam_bought_9_cans : total_cost_no_discount / cost_per_can = 9 :=
by
  sorry -- Proof goes here

end sam_bought_9_cans_l305_305179


namespace pie_cost_l305_305955

theorem pie_cost (initial_amount remaining_amount : ℕ) (h1 : initial_amount = 63) (h2 : remaining_amount = 57) : initial_amount - remaining_amount = 6 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add (by norm_num)

end pie_cost_l305_305955


namespace perpendiculars_ratio_intersection_l305_305577

noncomputable def points_on_line (l : Type) (A_1 B_1 C_1 A_2 B_2 C_2 : l) : Prop :=
  ∃ (l : Type), ∀ (X : l), X ∈ {A_1, B_1, C_1, A_2, B_2, C_2}

def perpendiculars_intersect_one_point (A_1 B_1 C_1 : Type) (ABC : Triangle) : Prop :=
  ∃ P : Type, is_intersection P (perpendicular A_1 (side BC)) (perpendicular B_1 (side CA)) (perpendicular C_1 (side AB))

def segment_ratios (A_1 B_1 C_1 A_2 B_2 C_2 : Point) : Prop :=
  (distance A_1 B_1 / distance B_1 C_1 = distance A_2 B_2 / distance B_2 C_2)

theorem perpendiculars_ratio_intersection {l : Type}
  {A_1 B_1 C_1 A_2 B_2 C_2 : l} (triangle_ABC : Triangle):
  points_on_line l A_1 B_1 C_1 A_2 B_2 C_2 →
  segment_ratios A_1 B_1 C_1 A_2 B_2 C_2 ↔
  perpendiculars_intersect_one_point A_1 B_1 C_1 triangle_ABC :=
sorry

end perpendiculars_ratio_intersection_l305_305577


namespace nth_number_in_pattern_l305_305106

theorem nth_number_in_pattern (n : ℕ) (h_n : n = 100) :
  let row_sum := λ k : ℕ, k * (k + 1) in
  let find_row := λ n m, ∃ k : ℕ, row_sum k < m ∧ m ≤ row_sum (k + 1) in
  let k := (nat.find (find_row n 100)) in
  2 * (k + 1) = 20 :=
by
  sorry

end nth_number_in_pattern_l305_305106


namespace median_unchanged_when_removing_extremes_l305_305890

theorem median_unchanged_when_removing_extremes (scores : List ℝ) (h_len : scores.length = 7) :
  let scores_sorted := List.quicksort scores in
  let middle_original := scores_sorted.nth_le 3 sorry in
  let scores_filtered := scores_sorted.drop 1 |>.take 5 in
  let middle_filtered := scores_filtered.nth_le 2 sorry in
  middle_original = middle_filtered := sorry

end median_unchanged_when_removing_extremes_l305_305890


namespace sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l305_305145

-- Define the real interval [0, π/2]
def interval_0_pi_over_2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Define the proposition to be proven
theorem sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq (a b : ℝ) 
  (ha : interval_0_pi_over_2 a) (hb : interval_0_pi_over_2 b) :
  (Real.sin a)^6 + 3 * (Real.sin a)^2 * (Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b :=
by
  sorry

end sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l305_305145


namespace range_arg_of_complex_l305_305400

theorem range_arg_of_complex (z : ℂ) (hz : ∥2 * z + 1 / z∥ = 1) :
  ∃ θ : ℝ,
  θ ∈ set.Icc (Real.arccos (Real.sqrt 2 / 4)) (π - Real.arccos (Real.sqrt 2 / 4)) ∪
  set.Icc (π + Real.arccos (Real.sqrt 2 / 4)) (2 * π - Real.arccos (Real.sqrt 2 / 4)) :=
sorry

end range_arg_of_complex_l305_305400


namespace f_seven_l305_305814

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ℝ) : f (-h) = -f (h)
axiom periodic_function (h : ℝ) : f (h + 4) = f (h)
axiom f_one : f 1 = 2

theorem f_seven : f (7) = -2 :=
by
  sorry

end f_seven_l305_305814


namespace part1_solution_set_part2_range_m_l305_305070

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Problem Part 1: Find the solution set of |f(x) - 3| ≤ 4
theorem part1_solution_set :
  {x : ℝ | abs (f x - 3) ≤ 4} = {x : ℝ | -6 ≤ x ∧ x ≤ 8} :=
sorry

-- Problem Part 2: Find the range of m such that f(x) + f(x + 3) ≥ m^2 - 2m always holds
theorem part2_range_m :
  {m : ℝ | ∀ x : ℝ, f x + f (x + 3) ≥ m^2 - 2m} = {m : ℝ | -1 ≤ m ∧ m ≤ 3} :=
sorry

end part1_solution_set_part2_range_m_l305_305070


namespace digimon_pack_price_l305_305142

-- Defining the given conditions as Lean variables
variables (total_spent baseball_cost : ℝ)
variables (packs_of_digimon : ℕ)

-- Setting given values from the problem
def keith_total_spent : total_spent = 23.86 := sorry
def baseball_deck_cost : baseball_cost = 6.06 := sorry
def number_of_digimon_packs : packs_of_digimon = 4 := sorry

-- Stating the main theorem/problem to prove
theorem digimon_pack_price 
  (h1 : total_spent = 23.86)
  (h2 : baseball_cost = 6.06)
  (h3 : packs_of_digimon = 4) : 
  ∃ (price_per_pack : ℝ), price_per_pack = 4.45 :=
sorry

end digimon_pack_price_l305_305142


namespace tan_double_angle_l305_305812

theorem tan_double_angle (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2)
  (h2 : sin (π + α) = -3 / 5) : tan (2 * α) = -24 / 7 :=
by
  sorry

end tan_double_angle_l305_305812


namespace square_vertices_of_inequality_l305_305908

variables {A B C : ℝ × ℝ} 

noncomputable def dist (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def area (A B C : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem square_vertices_of_inequality 
  (hA : ∃ x, (A.1 = x) ∧ ∃ y, (A.2 = y) ∧ int.cast x = x ∧ int.cast y = y) 
  (hB : ∃ x, (B.1 = x) ∧ ∃ y, (B.2 = y) ∧ int.cast x = x ∧ int.cast y = y) 
  (hC : ∃ x, (C.1 = x) ∧ ∃ y, (C.2 = y) ∧ int.cast x = x ∧ int.cast y = y) 
  (distinct_points : A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (inequality : (dist A B + dist B C) ^ 2 < 8 * area A B C + 1) : 
  (dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) :=
by sorry

end square_vertices_of_inequality_l305_305908


namespace skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l305_305121

-- Define the variables for the number of vehicles each type of worker can install
variables {x y : ℝ}

-- Define the conditions for system of equations
def skilled_and_new_workers_system1 (x y : ℝ) : Prop :=
  2 * x + y = 10

def skilled_and_new_workers_system2 (x y : ℝ) : Prop :=
  x + 3 * y = 10

-- Prove the number of vehicles each skilled worker and new worker can install
theorem skilled_new_worker_installation (x y : ℝ) (h1 : skilled_and_new_workers_system1 x y) (h2 : skilled_and_new_workers_system2 x y) : x = 4 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

-- Define the average cost equation for electric and gasoline vehicles
def avg_cost (m : ℝ) : Prop :=
  1 = 4 * (m / (m + 0.6))

-- Prove the average cost per kilometer of the electric vehicle
theorem avg_cost_electric_vehicle (m : ℝ) (h : avg_cost m) : m = 0.2 :=
by {
  -- Proof skipped
  sorry
}

-- Define annual cost equations and the comparison condition
variables {a : ℝ}
def annual_cost_electric_vehicle (a : ℝ) : ℝ :=
  0.2 * a + 6400

def annual_cost_gasoline_vehicle (a : ℝ) : ℝ :=
  0.8 * a + 4000

-- Prove that when the annual mileage is greater than 6667 kilometers, the annual cost of buying an electric vehicle is lower
theorem cost_comparison (a : ℝ) (h : a > 6667) : annual_cost_electric_vehicle a < annual_cost_gasoline_vehicle a :=
by {
  -- Proof skipped
  sorry
}

end skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l305_305121


namespace prob_X_less_than_4_l305_305043

variable {X : ℝ → ℝ}
variable {μ σ : ℝ}

-- Condition: X follows a normal distribution with mean μ and variance σ^2
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∀ x, X x = (1 / (σ * sqrt 2 / sqrt π)) * exp (-(x - μ)^2 / (2 * σ^2))

-- Condition: P(X < 2) = 0.2
def prob_X_less_than_2 (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∫ x in -∞..2, normal_dist X μ σ = 0.2

-- Condition: P(X < 3) = 0.5
def prob_X_less_than_3 (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∫ x in -∞..3, normal_dist X μ σ = 0.5

-- Statement to prove: P(X < 4) = 0.8
theorem prob_X_less_than_4 (X : ℝ → ℝ) (μ σ : ℝ) 
  (h1 : normal_dist X μ σ) 
  (h2 : prob_X_less_than_2 X μ σ) 
  (h3 : prob_X_less_than_3 X μ σ) : 
  ∫ x in -∞..4, normal_dist X μ σ = 0.8 :=
sorry

end prob_X_less_than_4_l305_305043


namespace sum_between_100_and_500_ending_in_3_l305_305012

-- Definition for the sum of all integers between 100 and 500 that end in 3
def sumOfIntegersBetween100And500EndingIn3 : ℕ :=
  let a := 103
  let d := 10
  let n := (493 - a) / d + 1
  (n * (a + 493)) / 2

-- Statement to prove that the sum is 11920
theorem sum_between_100_and_500_ending_in_3 : sumOfIntegersBetween100And500EndingIn3 = 11920 := by
  sorry

end sum_between_100_and_500_ending_in_3_l305_305012


namespace min_P_value_l305_305332

-- Definitions of [m/k] as the integer closest to m/k
def closest_int (m k : ℤ) : ℤ :=
  round (m.toReal / k.toReal)

-- Definitions of P(k) based on the condition given in the problem
noncomputable def P (k : ℤ) : ℚ :=
  let favorable_cases := finset.filter (fun n => closest_int n k + closest_int (200 - n) k = closest_int 200 k) (finset.range 200)
  favorable_cases.card / 199

-- Statement asserting the minimum possible value of P(k) for odd k in the interval 1 ≤ k ≤ 150
theorem min_P_value : 
  ∀ k ∈ (finset.range 150).filter (λ k, odd k ∧ 1 ≤ k ∧ k ≤ 150), 
  P k = (62 / 123) := 
  by sorry

end min_P_value_l305_305332


namespace max_diagonals_of_regular_ngon_l305_305556

theorem max_diagonals_of_regular_ngon (n : ℕ) (h : n ≥ 3) :
  if even n then max_non_intersecting_perpendicular_only_diagonals n = n - 2
  else max_non_intersecting_perpendicular_only_diagonals n = n - 3 := sorry

end max_diagonals_of_regular_ngon_l305_305556


namespace decagon_triangle_probability_l305_305287

theorem decagon_triangle_probability : 
  (∃ decagon_segments: finset (ℕ × ℕ), 
    ∀ (a b c : ℕ × ℕ), 
      a ∈ decagon_segments → b ∈ decagon_segments → c ∈ decagon_segments → 
      a ≠ b → b ≠ c → a ≠ c →
      ∃ (verts : fin 10 → ℝ × ℝ), 
        ∃ (sides: finset ℝ), 
        ∀ (k : ℕ), k > 0 → k ≤ 5 → 
        (2 * real.sin(k * real.pi / 10) ∈ sides) ∧ 
        (∃ (valid : ℕ), valid = 14190 - (number_of_violations a b c) ∧ 
          valid / 14190 = 153 / 190)) := 
sorry

end decagon_triangle_probability_l305_305287


namespace cost_of_each_pair_of_jeans_l305_305536

-- Conditions
def costWallet : ℕ := 50
def costSneakers : ℕ := 100
def pairsSneakers : ℕ := 2
def costBackpack : ℕ := 100
def totalSpent : ℕ := 450
def pairsJeans : ℕ := 2

-- Definitions
def totalSpentLeonard := costWallet + pairsSneakers * costSneakers
def totalSpentMichaelWithoutJeans := costBackpack

-- Goal: Prove the cost of each pair of jeans
theorem cost_of_each_pair_of_jeans :
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  costPerPairJeans = 50 :=
by
  intros
  let totalCostJeans := totalSpent - (totalSpentLeonard + totalSpentMichaelWithoutJeans)
  let costPerPairJeans := totalCostJeans / pairsJeans
  show costPerPairJeans = 50
  sorry

end cost_of_each_pair_of_jeans_l305_305536


namespace surface_area_of_sphere_l305_305725

-- Define the dimensions of the rectangular solid
def length := 4
def width := 3
def height := 2

-- Define the sphere and the condition that the space diagonal of the rectangular solid is the diameter of the sphere.
def space_diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)
def radius : ℝ := space_diagonal / 2
def surface_area : ℝ := 4 * Real.pi * radius^2

-- Theorem stating that the surface area of the sphere is 29π
theorem surface_area_of_sphere : surface_area = 29 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l305_305725
