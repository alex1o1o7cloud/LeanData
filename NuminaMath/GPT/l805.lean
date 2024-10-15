import Mathlib

namespace NUMINAMATH_GPT_total_water_capacity_of_coolers_l805_80546

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end NUMINAMATH_GPT_total_water_capacity_of_coolers_l805_80546


namespace NUMINAMATH_GPT_perimeter_of_resulting_figure_l805_80569

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_resulting_figure_l805_80569


namespace NUMINAMATH_GPT_number_of_ways_2020_l805_80510

-- We are defining b_i explicitly restricted by the conditions in the problem.
def b (i : ℕ) : ℕ :=
  sorry

-- Given conditions
axiom h_bounds : ∀ i, 0 ≤ b i ∧ b i ≤ 99
axiom h_indices : ∀ (i : ℕ), i < 4

-- Main theorem statement
theorem number_of_ways_2020 (M : ℕ) 
  (h : 2020 = b 3 * 1000 + b 2 * 100 + b 1 * 10 + b 0) 
  (htotal : M = 203) : 
  M = 203 :=
  by 
    sorry

end NUMINAMATH_GPT_number_of_ways_2020_l805_80510


namespace NUMINAMATH_GPT_tangent_line_eq_l805_80526

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

theorem tangent_line_eq {x y : ℝ} (hx : x = 1) (hy : y = 2) (H : circle_eq x y) :
  y = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l805_80526


namespace NUMINAMATH_GPT_vector_addition_correct_l805_80590

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A]

def vector_addition (da cd cb ba : A) : Prop :=
  da + cd - cb = ba

theorem vector_addition_correct (da cd cb ba : A) :
  vector_addition da cd cb ba :=
  sorry

end NUMINAMATH_GPT_vector_addition_correct_l805_80590


namespace NUMINAMATH_GPT_value_of_m_l805_80580

theorem value_of_m (z1 z2 m : ℝ) (h1 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z1 = 0)
  (h2 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z2 = 0)
  (h3 : |z1 - z2| = 3) : m = 4 ∨ m = 17 / 2 := sorry

end NUMINAMATH_GPT_value_of_m_l805_80580


namespace NUMINAMATH_GPT_prob_chair_theorem_l805_80534

def numAvailableChairs : ℕ := 10 - 1

def totalWaysToChooseTwoChairs : ℕ := Nat.choose numAvailableChairs 2

def adjacentPairs : ℕ :=
  let pairs := [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]
  pairs.length

def probNextToEachOther : ℚ := adjacentPairs / totalWaysToChooseTwoChairs

def probNotNextToEachOther : ℚ := 1 - probNextToEachOther

theorem prob_chair_theorem : probNotNextToEachOther = 5/6 :=
by
  sorry

end NUMINAMATH_GPT_prob_chair_theorem_l805_80534


namespace NUMINAMATH_GPT_lcm_of_numbers_l805_80561

/-- Define the numbers involved -/
def a := 456
def b := 783
def c := 935
def d := 1024
def e := 1297

/-- Prove the LCM of these numbers is 2308474368000 -/
theorem lcm_of_numbers :
  Int.lcm (Int.lcm (Int.lcm (Int.lcm a b) c) d) e = 2308474368000 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_l805_80561


namespace NUMINAMATH_GPT_calculate_expression_l805_80551

theorem calculate_expression : (0.25)^(-0.5) + (1/27)^(-1/3) - 625^(0.25) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l805_80551


namespace NUMINAMATH_GPT_verify_original_prices_l805_80535

noncomputable def original_price_of_sweater : ℝ := 43.11
noncomputable def original_price_of_shirt : ℝ := 35.68
noncomputable def original_price_of_pants : ℝ := 71.36

def price_of_shirt (sweater_price : ℝ) : ℝ := sweater_price - 7.43
def price_of_pants (shirt_price : ℝ) : ℝ := 2 * shirt_price
def discounted_sweater_price (sweater_price : ℝ) : ℝ := 0.85 * sweater_price
def total_cost (shirt_price pants_price discounted_sweater_price : ℝ) : ℝ := shirt_price + pants_price + discounted_sweater_price

theorem verify_original_prices 
  (total_cost_value : ℝ)
  (price_of_shirt_value : ℝ)
  (price_of_pants_value : ℝ)
  (discounted_sweater_price_value : ℝ) :
  total_cost_value = 143.67 ∧ 
  price_of_shirt_value = original_price_of_shirt ∧ 
  price_of_pants_value = original_price_of_pants ∧
  discounted_sweater_price_value = discounted_sweater_price original_price_of_sweater →
  total_cost (price_of_shirt original_price_of_sweater) 
             (price_of_pants (price_of_shirt original_price_of_sweater)) 
             (discounted_sweater_price original_price_of_sweater) = 143.67 :=
by
  intros
  sorry

end NUMINAMATH_GPT_verify_original_prices_l805_80535


namespace NUMINAMATH_GPT_fractional_cake_eaten_l805_80514

def total_cake_eaten : ℚ :=
  1 / 3 + 1 / 3 + 1 / 6 + 1 / 12 + 1 / 24 + 1 / 48

theorem fractional_cake_eaten :
  total_cake_eaten = 47 / 48 := by
  sorry

end NUMINAMATH_GPT_fractional_cake_eaten_l805_80514


namespace NUMINAMATH_GPT_average_class_score_l805_80579

theorem average_class_score : 
  ∀ (n total score_per_100 score_per_0 avg_rest : ℕ), 
  n = 20 → 
  total = 800 → 
  score_per_100 = 2 → 
  score_per_0 = 3 → 
  avg_rest = 40 → 
  ((score_per_100 * 100 + score_per_0 * 0 + (n - (score_per_100 + score_per_0)) * avg_rest) / n = 40)
:= by
  intros n total score_per_100 score_per_0 avg_rest h_n h_total h_100 h_0 h_rest
  sorry

end NUMINAMATH_GPT_average_class_score_l805_80579


namespace NUMINAMATH_GPT_china_GDP_in_2016_l805_80562

noncomputable def GDP_2016 (a r : ℝ) : ℝ := a * (1 + r / 100)^5

theorem china_GDP_in_2016 (a r : ℝ) :
  GDP_2016 a r = a * (1 + r / 100)^5 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_china_GDP_in_2016_l805_80562


namespace NUMINAMATH_GPT_no_natural_number_solution_for_divisibility_by_2020_l805_80560

theorem no_natural_number_solution_for_divisibility_by_2020 :
  ¬ ∃ k : ℕ, (k^3 - 3 * k^2 + 2 * k + 2) % 2020 = 0 :=
sorry

end NUMINAMATH_GPT_no_natural_number_solution_for_divisibility_by_2020_l805_80560


namespace NUMINAMATH_GPT_find_number_l805_80594

theorem find_number (S Q R N : ℕ) (hS : S = 555 + 445) (hQ : Q = 2 * (555 - 445)) (hR : R = 50) (h_eq : N = S * Q + R) :
  N = 220050 :=
by
  rw [hS, hQ, hR] at h_eq
  norm_num at h_eq
  exact h_eq

end NUMINAMATH_GPT_find_number_l805_80594


namespace NUMINAMATH_GPT_n_power_2020_plus_4_composite_l805_80582

theorem n_power_2020_plus_4_composite {n : ℕ} (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b := 
by
  sorry

end NUMINAMATH_GPT_n_power_2020_plus_4_composite_l805_80582


namespace NUMINAMATH_GPT_count_positive_integers_l805_80599

theorem count_positive_integers (n : ℕ) (x : ℝ) (h1 : n ≤ 1500) :
  (∃ x : ℝ, n = ⌊x⌋ + ⌊3*x⌋ + ⌊5*x⌋) ↔ n = 668 :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_l805_80599


namespace NUMINAMATH_GPT_middle_number_consecutive_odd_sum_l805_80509

theorem middle_number_consecutive_odd_sum (n : ℤ)
  (h1 : n % 2 = 1) -- n is an odd number
  (h2 : n + (n + 2) + (n + 4) = n + 20) : 
  n + 2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_consecutive_odd_sum_l805_80509


namespace NUMINAMATH_GPT_largest_y_value_l805_80567

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_y_value_l805_80567


namespace NUMINAMATH_GPT_symmetric_line_l805_80545

theorem symmetric_line (x y : ℝ) : (2 * x + y - 4 = 0) → (2 * x - y + 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_l805_80545


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l805_80576

-- Define the conditions and the proof problem
theorem number_of_men_in_first_group (M : ℕ) 
  (h1 : ∀ t : ℝ, 22 * t = M) 
  (h2 : ∀ t' : ℝ, 18 * 17.11111111111111 = t') :
  M = 14 := 
by
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l805_80576


namespace NUMINAMATH_GPT_inequality_proof_l805_80565

theorem inequality_proof (a b c d : ℕ) (h₀: a + c ≤ 1982) (h₁: (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)) (h₂: (a:ℚ)/b + (c:ℚ)/d < 1) :
  1 - (a:ℚ)/b - (c:ℚ)/d > 1 / (1983 ^ 3) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l805_80565


namespace NUMINAMATH_GPT_B_participated_Huangmei_Opera_l805_80539

-- Definitions using given conditions
def participated_A (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∨ 
  (c "Huangmei Flower Picking" ∧ ¬ c "Yue Family Boxing")

def participated_B (c : String → Prop) : Prop :=
  (c "Huangmei Opera" ∧ ¬ c "Huangmei Flower Picking") ∨
  (c "Yue Family Boxing" ∧ ¬ c "Huangmei Flower Picking")

def participated_C (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∧ c "Huangmei Flower Picking" ∧ c "Yue Family Boxing" ->
  (c "Huangmei Opera" ∨ c "Huangmei Flower Picking" ∨ c "Yue Family Boxing")

-- Proving the special class that B participated in
theorem B_participated_Huangmei_Opera :
  ∃ c : String → Prop, participated_A c ∧ participated_B c ∧ participated_C c → c "Huangmei Opera" :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_B_participated_Huangmei_Opera_l805_80539


namespace NUMINAMATH_GPT_focus_of_parabola_l805_80529

theorem focus_of_parabola (focus : ℝ × ℝ) : 
  (∃ p : ℝ, y = p * x^2 / 2 → focus = (0, 1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l805_80529


namespace NUMINAMATH_GPT_even_product_implies_even_factor_l805_80557

theorem even_product_implies_even_factor (a b : ℕ) (h : Even (a * b)) : Even a ∨ Even b :=
by
  sorry

end NUMINAMATH_GPT_even_product_implies_even_factor_l805_80557


namespace NUMINAMATH_GPT_calculate_f3_times_l805_80592

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4 * n + 2

theorem calculate_f3_times : f (f (f 3)) = 170 := by
  sorry

end NUMINAMATH_GPT_calculate_f3_times_l805_80592


namespace NUMINAMATH_GPT_problem_1_problem_2_l805_80597

theorem problem_1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : abc ≤ 3 * Real.sqrt 3 := 
sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : 
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) > (a + b + c) / 3 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l805_80597


namespace NUMINAMATH_GPT_trigonometric_inequality_C_trigonometric_inequality_D_l805_80574

theorem trigonometric_inequality_C (x : Real) : Real.cos (3*Real.pi/5) > Real.cos (-4*Real.pi/5) :=
by
  sorry

theorem trigonometric_inequality_D (y : Real) : Real.sin (Real.pi/10) < Real.cos (Real.pi/10) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_C_trigonometric_inequality_D_l805_80574


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l805_80544

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α)) / ((Real.cos α) ^ 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l805_80544


namespace NUMINAMATH_GPT_find_value_l805_80573

theorem find_value (a b c : ℝ) (h1 : a + b = 8) (h2 : a * b = c^2 + 16) : a + 2 * b + 3 * c = 12 := by
  sorry

end NUMINAMATH_GPT_find_value_l805_80573


namespace NUMINAMATH_GPT_sin_C_and_area_of_triangle_l805_80521

open Real

noncomputable section

theorem sin_C_and_area_of_triangle 
  (A B C : ℝ)
  (cos_A : Real := sqrt 3 / 3)
  (a b c : ℝ := (3 * sqrt 2)) 
  (cosA : cos A = sqrt 3 / 3)
  -- angles in radians, use radians for the angles when proving
  (side_c : c = sqrt 3)
  (side_a : a = 3 * sqrt 2) :
  (sin C = 1 / 3) ∧ (1 / 2 * a * b * sin C = 5 * sqrt 6 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sin_C_and_area_of_triangle_l805_80521


namespace NUMINAMATH_GPT_intersect_eq_l805_80585

variable (M N : Set Int)
def M_def : Set Int := { m | -3 < m ∧ m < 2 }
def N_def : Set Int := { n | -1 ≤ n ∧ n ≤ 3 }

theorem intersect_eq : M_def ∩ N_def = { -1, 0, 1 } := by
  sorry

end NUMINAMATH_GPT_intersect_eq_l805_80585


namespace NUMINAMATH_GPT_simplify_trig_expression_tan_alpha_value_l805_80584

-- Proof Problem (1)
theorem simplify_trig_expression :
  (∃ θ : ℝ, θ = (20:ℝ) ∧ 
    (∃ α : ℝ, α = (160:ℝ) ∧ 
      (∃ β : ℝ, β = 1 - 2 * (Real.sin θ) * (Real.cos θ) ∧ 
        (∃ γ : ℝ, γ = 1 - (Real.sin θ)^2 ∧ 
          (Real.sqrt β) / ((Real.sin α) - (Real.sqrt γ)) = -1)))) :=
sorry

-- Proof Problem (2)
theorem tan_alpha_value (α : ℝ) (h : Real.tan α = 1 / 3) :
  1 / (4 * (Real.cos α)^2 - 6 * (Real.sin α) * (Real.cos α)) = 5 / 9 :=
sorry

end NUMINAMATH_GPT_simplify_trig_expression_tan_alpha_value_l805_80584


namespace NUMINAMATH_GPT_probability_of_selecting_meiqi_l805_80522

def four_red_bases : List String := ["Meiqi", "Wangcunkou", "Zhulong", "Xiaoshun"]

theorem probability_of_selecting_meiqi :
  (1 / 4 : ℝ) = 1 / (four_red_bases.length : ℝ) :=
  by sorry

end NUMINAMATH_GPT_probability_of_selecting_meiqi_l805_80522


namespace NUMINAMATH_GPT_number_of_dress_designs_l805_80589

open Nat

theorem number_of_dress_designs : (3 * 4 = 12) :=
by
  rfl

end NUMINAMATH_GPT_number_of_dress_designs_l805_80589


namespace NUMINAMATH_GPT_remainder_31_31_plus_31_mod_32_l805_80570

theorem remainder_31_31_plus_31_mod_32 : (31 ^ 31 + 31) % 32 = 30 := 
by sorry

end NUMINAMATH_GPT_remainder_31_31_plus_31_mod_32_l805_80570


namespace NUMINAMATH_GPT_ada_originally_in_seat2_l805_80538

inductive Seat
| S1 | S2 | S3 | S4 | S5 deriving Inhabited, DecidableEq

def moveRight : Seat → Option Seat
| Seat.S1 => some Seat.S2
| Seat.S2 => some Seat.S3
| Seat.S3 => some Seat.S4
| Seat.S4 => some Seat.S5
| Seat.S5 => none

def moveLeft : Seat → Option Seat
| Seat.S1 => none
| Seat.S2 => some Seat.S1
| Seat.S3 => some Seat.S2
| Seat.S4 => some Seat.S3
| Seat.S5 => some Seat.S4

structure FriendState :=
  (bea ceci dee edie : Seat)
  (ada_left : Bool) -- Ada is away for snacks, identified by her not being in the seat row.

def initial_seating := FriendState.mk Seat.S2 Seat.S3 Seat.S4 Seat.S5 true

def final_seating (init : FriendState) : FriendState :=
  let bea' := match moveRight init.bea with
              | some pos => pos
              | none => init.bea
  let ceci' := init.ceci -- Ceci moves left then back, net zero movement
  let (dee', edie') := match moveRight init.dee, init.dee with
                      | some new_ee, ed => (new_ee, ed) -- Dee and Edie switch and Edie moves right
                      | _, _ => (init.dee, init.edie) -- If moves are invalid
  FriendState.mk bea' ceci' dee' edie' init.ada_left

theorem ada_originally_in_seat2 (init : FriendState) : init = initial_seating → final_seating init ≠ initial_seating → init.bea = Seat.S2 :=
by
  intro h_init h_finalne
  sorry -- Proof steps go here

end NUMINAMATH_GPT_ada_originally_in_seat2_l805_80538


namespace NUMINAMATH_GPT_jogger_distance_l805_80554

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  -- Definition and proof would go here
  --
  sorry

end NUMINAMATH_GPT_jogger_distance_l805_80554


namespace NUMINAMATH_GPT_negative_two_squared_l805_80519

theorem negative_two_squared :
  (-2 : ℤ)^2 = 4 := 
sorry

end NUMINAMATH_GPT_negative_two_squared_l805_80519


namespace NUMINAMATH_GPT_solve_system_l805_80520

theorem solve_system :
  ∃ x y : ℚ, 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_l805_80520


namespace NUMINAMATH_GPT_expected_waiting_time_l805_80583

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end NUMINAMATH_GPT_expected_waiting_time_l805_80583


namespace NUMINAMATH_GPT_gcd_equation_solution_l805_80577

theorem gcd_equation_solution (x y : ℕ) (h : Nat.gcd x y + x * y / Nat.gcd x y = x + y) : y ∣ x ∨ x ∣ y :=
 by
 sorry

end NUMINAMATH_GPT_gcd_equation_solution_l805_80577


namespace NUMINAMATH_GPT_stock_yield_calculation_l805_80572

theorem stock_yield_calculation (par_value market_value annual_dividend : ℝ)
  (h1 : par_value = 100)
  (h2 : market_value = 80)
  (h3 : annual_dividend = 0.04 * par_value) :
  (annual_dividend / market_value) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_stock_yield_calculation_l805_80572


namespace NUMINAMATH_GPT_part_one_part_two_l805_80542

noncomputable def f (x : ℝ) : ℝ := (3 * x) / (x + 1)

-- First part: Prove that f(x) is increasing on [2, 5]
theorem part_one (x₁ x₂ : ℝ) (hx₁ : 2 ≤ x₁) (hx₂ : x₂ ≤ 5) (h : x₁ < x₂) : f x₁ < f x₂ :=
by {
  -- Proof is to be filled in
  sorry
}

-- Second part: Find maximum and minimum of f(x) on [2, 5]
theorem part_two :
  f 2 = 2 ∧ f 5 = 5 / 2 :=
by {
  -- Proof is to be filled in
  sorry
}

end NUMINAMATH_GPT_part_one_part_two_l805_80542


namespace NUMINAMATH_GPT_intersection_of_sets_l805_80564

theorem intersection_of_sets {A B : Set Nat} (hA : A = {1, 3, 9}) (hB : B = {1, 5, 9}) :
  A ∩ B = {1, 9} :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l805_80564


namespace NUMINAMATH_GPT_find_A_and_B_l805_80508

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (∀ x : ℝ, x ≠ 10 ∧ x ≠ -3 → 5*x + 2 = A * (x + 3) + B * (x - 10)) ∧ 
    A = 4 ∧ B = 1 :=
  sorry

end NUMINAMATH_GPT_find_A_and_B_l805_80508


namespace NUMINAMATH_GPT_find_k_l805_80575

theorem find_k (k : ℝ) (h : 0.5 * |-2 * k| * |k| = 1) : k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_l805_80575


namespace NUMINAMATH_GPT_compute_expression_l805_80547

theorem compute_expression (x : ℝ) (h : x + (1 / x) = 7) :
  (x - 3)^2 + (49 / (x - 3)^2) = 23 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l805_80547


namespace NUMINAMATH_GPT_triangle_AB_C_min_perimeter_l805_80506

noncomputable def minimum_perimeter (a b c : ℕ) (A B C : ℝ) : ℝ := a + b + c

theorem triangle_AB_C_min_perimeter
  (a b c : ℕ)
  (A B C : ℝ)
  (h1 : A = 2 * B)
  (h2 : C > π / 2)
  (h3 : a^2 = b * (b + c))
  (h4 : ∀ x : ℕ, x > 0 → a ≠ 0)
  (h5 :  a + b > c ∧ a + c > b ∧ b + c > a) :
  minimum_perimeter a b c A B C = 77 := 
sorry

end NUMINAMATH_GPT_triangle_AB_C_min_perimeter_l805_80506


namespace NUMINAMATH_GPT_circle_center_distance_l805_80595

theorem circle_center_distance (R : ℝ) : 
  ∃ (d : ℝ), 
  (∀ (θ : ℝ), θ = 30 → 
  ∀ (r : ℝ), r = 2.5 →
  ∀ (center_on_other_side : ℝ), center_on_other_side = R + R →
  d = 5) :=
by 
  use 5
  intros θ θ_eq r r_eq center_on_other_side center_eq
  sorry

end NUMINAMATH_GPT_circle_center_distance_l805_80595


namespace NUMINAMATH_GPT_count_solutions_g_composition_eq_l805_80578

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the main theorem
theorem count_solutions_g_composition_eq :
  ∃ (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
by
  sorry

end NUMINAMATH_GPT_count_solutions_g_composition_eq_l805_80578


namespace NUMINAMATH_GPT_all_numbers_rational_l805_80540

-- Define the mathematical operations for the problem
def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem all_numbers_rational :
    (∃ x1 : ℚ, fourth_root 81 = x1) ∧
    (∃ x2 : ℚ, square_root 0.64 = x2) ∧
    (∃ x3 : ℚ, cube_root 0.001 = x3) ∧
    (∃ x4 : ℚ, (cube_root 8) * (square_root ((0.25)⁻¹)) = x4) :=
  sorry

end NUMINAMATH_GPT_all_numbers_rational_l805_80540


namespace NUMINAMATH_GPT_min_area_monochromatic_triangle_l805_80511

-- Definition of the integer lattice in the plane.
def lattice_points : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) }

-- The 3-coloring condition
def coloring (c : (ℤ × ℤ) → Fin 3) := ∀ p : (ℤ × ℤ), p ∈ lattice_points → (c p) < 3

-- Definition of the area of a triangle
def triangle_area (A B C : ℤ × ℤ) : ℝ :=
  0.5 * abs (((B.1 - A.1) * (C.2 - A.2)) - ((C.1 - A.1) * (B.2 - A.2)))

-- The statement we need to prove
theorem min_area_monochromatic_triangle :
  ∃ S : ℝ, S = 3 ∧ ∀ (c : (ℤ × ℤ) → Fin 3), coloring c → ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (c A = c B ∧ c B = c C) ∧ triangle_area A B C = S :=
sorry

end NUMINAMATH_GPT_min_area_monochromatic_triangle_l805_80511


namespace NUMINAMATH_GPT_discriminant_of_quadratic_is_321_l805_80503

-- Define the quadratic equation coefficients
def a : ℝ := 4
def b : ℝ := -9
def c : ℝ := -15

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The proof statement
theorem discriminant_of_quadratic_is_321 : discriminant a b c = 321 := by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_is_321_l805_80503


namespace NUMINAMATH_GPT_coin_flip_sequences_l805_80533

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_l805_80533


namespace NUMINAMATH_GPT_compute_exp_l805_80581

theorem compute_exp : 3 * 3^4 + 9^30 / 9^28 = 324 := 
by sorry

end NUMINAMATH_GPT_compute_exp_l805_80581


namespace NUMINAMATH_GPT_area_of_triangle_le_one_fourth_l805_80528

open Real

noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_le_one_fourth (t : ℝ) (x y : ℝ) (h_t : 0 < t ∧ t < 1) (h_x : 0 ≤ x ∧ x ≤ 1)
  (h_y : y = t * (2 * x - t)) :
  area_triangle t (t^2) 1 0 x y ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_le_one_fourth_l805_80528


namespace NUMINAMATH_GPT_abc_geq_expression_l805_80515

variable (a b c : ℝ) -- Define variables a, b, c as real numbers
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) -- Define conditions of a, b, c being positive

theorem abc_geq_expression : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
by 
  sorry -- Proof goes here

end NUMINAMATH_GPT_abc_geq_expression_l805_80515


namespace NUMINAMATH_GPT_complex_mul_l805_80552

theorem complex_mul (i : ℂ) (hi : i * i = -1) : (1 - i) * (3 + i) = 4 - 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_mul_l805_80552


namespace NUMINAMATH_GPT_percentage_students_passed_is_35_l805_80550

/-
The problem is to prove the percentage of students who passed the examination, given that 520 out of 800 students failed, is 35%.
-/

def total_students : ℕ := 800
def failed_students : ℕ := 520
def passed_students : ℕ := total_students - failed_students

def percentage_passed : ℕ := (passed_students * 100) / total_students

theorem percentage_students_passed_is_35 : percentage_passed = 35 :=
by
  -- Here the proof will go.
  sorry

end NUMINAMATH_GPT_percentage_students_passed_is_35_l805_80550


namespace NUMINAMATH_GPT_circle_diameter_length_l805_80523

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_circle_diameter_length_l805_80523


namespace NUMINAMATH_GPT_value_of_m_l805_80500

theorem value_of_m (m : ℤ) (h₁ : |m| = 2) (h₂ : m ≠ 2) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l805_80500


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l805_80517

theorem arithmetic_sequence_terms (a d n : ℤ) (last_term : ℤ)
  (h_a : a = 5)
  (h_d : d = 3)
  (h_last_term : last_term = 149)
  (h_n_eq : last_term = a + (n - 1) * d) :
  n = 49 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l805_80517


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l805_80587

theorem arithmetic_sequence_common_difference 
    (a : ℤ) (last_term : ℤ) (sum_terms : ℤ) (n : ℕ)
    (h1 : a = 3) 
    (h2 : last_term = 58) 
    (h3 : sum_terms = 488)
    (h4 : sum_terms = n * (a + last_term) / 2)
    (h5 : last_term = a + (n - 1) * d) :
    d = 11 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l805_80587


namespace NUMINAMATH_GPT_base13_addition_l805_80586

/--
Given two numbers in base 13: 528₁₃ and 274₁₃, prove that their sum is 7AC₁₃.
-/
theorem base13_addition :
  let u1 := 8
  let t1 := 2
  let h1 := 5
  let u2 := 4
  let t2 := 7
  let h2 := 2
  -- Add the units digits: 8 + 4 = 12; 12 is C in base 13
  let s1 := 12 -- 'C' in base 13
  let carry1 := 1
  -- Add the tens digits along with the carry: 2 + 7 + 1 = 10; 10 is A in base 13
  let s2 := 10 -- 'A' in base 13
  -- Add the hundreds digits: 5 + 2 = 7
  let s3 := 7 -- 7 in base 13
  s1 = 12 ∧ s2 = 10 ∧ s3 = 7 :=
by
  sorry

end NUMINAMATH_GPT_base13_addition_l805_80586


namespace NUMINAMATH_GPT_division_result_l805_80593

open Polynomial

noncomputable def dividend := (X ^ 6 - 5 * X ^ 4 + 3 * X ^ 3 - 7 * X ^ 2 + 2 * X - 8 : Polynomial ℤ)
noncomputable def divisor := (X - 3 : Polynomial ℤ)
noncomputable def expected_quotient := (X ^ 5 + 3 * X ^ 4 + 4 * X ^ 3 + 15 * X ^ 2 + 38 * X + 116 : Polynomial ℤ)
noncomputable def expected_remainder := (340 : ℤ)

theorem division_result : (dividend /ₘ divisor) = expected_quotient ∧ (dividend %ₘ divisor) = C expected_remainder := by
  sorry

end NUMINAMATH_GPT_division_result_l805_80593


namespace NUMINAMATH_GPT_inequality_x_add_inv_x_ge_two_l805_80527

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end NUMINAMATH_GPT_inequality_x_add_inv_x_ge_two_l805_80527


namespace NUMINAMATH_GPT_probability_at_least_6_heads_8_flips_l805_80524

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end NUMINAMATH_GPT_probability_at_least_6_heads_8_flips_l805_80524


namespace NUMINAMATH_GPT_fence_cost_l805_80516

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (side_length perimeter cost : ℝ) 
  (h1 : area = 289) 
  (h2 : price_per_foot = 55)
  (h3 : side_length = Real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : cost = perimeter * price_per_foot) :
  cost = 3740 := 
sorry

end NUMINAMATH_GPT_fence_cost_l805_80516


namespace NUMINAMATH_GPT_original_decimal_number_l805_80541

theorem original_decimal_number (x : ℝ) (h : 0.375 = (x / 1000) * 10) : x = 37.5 :=
sorry

end NUMINAMATH_GPT_original_decimal_number_l805_80541


namespace NUMINAMATH_GPT_find_vector_result_l805_80507

-- Define the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m: ℝ) : ℝ × ℝ := (-2, m)
def m := -4
def result := 2 • vector_a + 3 • vector_b m

-- State the theorem
theorem find_vector_result : result = (-4, -8) := 
by {
  -- skipping the proof
  sorry
}

end NUMINAMATH_GPT_find_vector_result_l805_80507


namespace NUMINAMATH_GPT_positive_integers_sequence_l805_80555

theorem positive_integers_sequence (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a ∣ (b + c + d)) (h5 : b ∣ (a + c + d)) 
  (h6 : c ∣ (a + b + d)) (h7 : d ∣ (a + b + c)) : 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 9) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 8 ∧ d = 12) ∨ 
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 10) ∨ 
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 21) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 10 ∧ d = 15) :=
sorry

end NUMINAMATH_GPT_positive_integers_sequence_l805_80555


namespace NUMINAMATH_GPT_dinner_guest_arrangement_l805_80568

noncomputable def number_of_ways (n k : ℕ) : ℕ :=
  if n < k then 0 else Nat.factorial n / Nat.factorial (n - k)

theorem dinner_guest_arrangement :
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements = 5040 :=
by
  -- Definitions and preliminary calculations
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements

  -- This is where the proof would go, but we insert sorry to skip it for now
  sorry

end NUMINAMATH_GPT_dinner_guest_arrangement_l805_80568


namespace NUMINAMATH_GPT_find_angle_D_l805_80525

theorem find_angle_D
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_C = 2 * angle_D)
  (h3 : angle_A = 100)
  (h4 : angle_B + angle_C + angle_D = 180) :
  angle_D = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_D_l805_80525


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l805_80502

-- Variables for the conditions
variables (x y : ℝ)

-- Conditions
def cond1 : Prop := x ≠ 1 ∨ y ≠ 4
def cond2 : Prop := x + y ≠ 5

-- Statement to prove the type of condition
theorem necessary_but_not_sufficient :
  cond2 x y → cond1 x y ∧ ¬(cond1 x y → cond2 x y) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l805_80502


namespace NUMINAMATH_GPT_brendan_remaining_money_l805_80556

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end NUMINAMATH_GPT_brendan_remaining_money_l805_80556


namespace NUMINAMATH_GPT_functional_relationship_remaining_oil_after_4_hours_l805_80504

-- Define the initial conditions and the functional form
def initial_oil : ℝ := 50
def consumption_rate : ℝ := 8
def remaining_oil (t : ℝ) : ℝ := initial_oil - consumption_rate * t

-- Prove the functional relationship and the remaining oil after 4 hours
theorem functional_relationship : ∀ (t : ℝ), remaining_oil t = 50 - 8 * t :=
by intros t
   exact rfl

theorem remaining_oil_after_4_hours : remaining_oil 4 = 18 :=
by simp [remaining_oil]
   norm_num
   sorry

end NUMINAMATH_GPT_functional_relationship_remaining_oil_after_4_hours_l805_80504


namespace NUMINAMATH_GPT_girls_insects_collected_l805_80501

theorem girls_insects_collected (boys_insects groups insects_per_group : ℕ) :
  boys_insects = 200 →
  groups = 4 →
  insects_per_group = 125 →
  (groups * insects_per_group) - boys_insects = 300 :=
by
  intros h1 h2 h3
  -- Prove the statement
  sorry

end NUMINAMATH_GPT_girls_insects_collected_l805_80501


namespace NUMINAMATH_GPT_candy_lollipops_l805_80531

theorem candy_lollipops (κ c l : ℤ) 
  (h1 : κ = l + c - 8)
  (h2 : c = l + κ - 14) :
  l = 11 :=
by
  sorry

end NUMINAMATH_GPT_candy_lollipops_l805_80531


namespace NUMINAMATH_GPT_white_balls_in_bag_l805_80591

theorem white_balls_in_bag : 
  ∀ x : ℕ, (3 + 2 + x ≠ 0) → (2 : ℚ) / (3 + 2 + x) = 1 / 4 → x = 3 :=
by
  intro x
  intro h1
  intro h2
  sorry

end NUMINAMATH_GPT_white_balls_in_bag_l805_80591


namespace NUMINAMATH_GPT_percent_difference_z_w_l805_80532

theorem percent_difference_z_w (w x y z : ℝ)
  (h1 : w = 0.60 * x)
  (h2 : x = 0.60 * y)
  (h3 : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
sorry

end NUMINAMATH_GPT_percent_difference_z_w_l805_80532


namespace NUMINAMATH_GPT_num_true_statements_is_two_l805_80571

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem num_true_statements_is_two :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_true_statements_is_two_l805_80571


namespace NUMINAMATH_GPT_compare_sine_values_1_compare_sine_values_2_l805_80537

theorem compare_sine_values_1 (h1 : 0 < Real.pi / 10) (h2 : Real.pi / 10 < Real.pi / 8) (h3 : Real.pi / 8 < Real.pi / 2) :
  Real.sin (- Real.pi / 10) > Real.sin (- Real.pi / 8) :=
by
  sorry

theorem compare_sine_values_2 (h1 : 0 < Real.pi / 8) (h2 : Real.pi / 8 < 3 * Real.pi / 8) (h3 : 3 * Real.pi / 8 < Real.pi / 2) :
  Real.sin (7 * Real.pi / 8) < Real.sin (5 * Real.pi / 8) :=
by
  sorry

end NUMINAMATH_GPT_compare_sine_values_1_compare_sine_values_2_l805_80537


namespace NUMINAMATH_GPT_line_through_points_a_minus_b_l805_80563

theorem line_through_points_a_minus_b :
  ∃ a b : ℝ, 
  (∀ x, (x = 3 → 7 = a * 3 + b) ∧ (x = 6 → 19 = a * 6 + b)) → 
  a - b = 9 :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_a_minus_b_l805_80563


namespace NUMINAMATH_GPT_walking_time_estimate_l805_80553

-- Define constants for distance, speed, and time conversion factor
def distance : ℝ := 1000
def speed : ℝ := 4000
def time_conversion : ℝ := 60

-- Define the expected time to walk from home to school in minutes
def expected_time : ℝ := 15

-- Prove the time calculation
theorem walking_time_estimate : (distance / speed) * time_conversion = expected_time :=
by
  sorry

end NUMINAMATH_GPT_walking_time_estimate_l805_80553


namespace NUMINAMATH_GPT_anita_gave_apples_l805_80543

theorem anita_gave_apples (initial_apples needed_for_pie apples_left_after_pie : ℝ)
  (h_initial : initial_apples = 10.0)
  (h_needed : needed_for_pie = 4.0)
  (h_left : apples_left_after_pie = 11.0) :
  ∃ (anita_apples : ℝ), anita_apples = 5 :=
by
  sorry

end NUMINAMATH_GPT_anita_gave_apples_l805_80543


namespace NUMINAMATH_GPT_min_n_probability_l805_80536

-- Define the number of members in teams
def num_members (n : ℕ) : ℕ := n

-- Define the total number of handshakes
def total_handshakes (n : ℕ) : ℕ := n * n

-- Define the number of ways to choose 2 handshakes from total handshakes
def choose_two_handshakes (n : ℕ) : ℕ := (total_handshakes n).choose 2

-- Define the number of ways to choose event A (involves exactly 3 different members)
def event_a_count (n : ℕ) : ℕ := 2 * n.choose 1 * (n - 1).choose 1

-- Define the probability of event A
def probability_event_a (n : ℕ) : ℚ := (event_a_count n : ℚ) / (choose_two_handshakes n : ℚ)

-- The minimum value of n such that the probability of event A is less than 1/10
theorem min_n_probability :
  ∃ n : ℕ, (probability_event_a n < (1 : ℚ) / 10) ∧ n ≥ 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_n_probability_l805_80536


namespace NUMINAMATH_GPT_find_multiple_l805_80505

-- Given conditions as definitions
def smaller_number := 21
def sum_of_numbers := 84

-- Definition of larger number being a multiple of the smaller number
def is_multiple (k : ℤ) (a b : ℤ) : Prop := b = k * a

-- Given that one number is a multiple of the other and their sum
def problem (L S : ℤ) (k : ℤ) : Prop := 
  is_multiple k S L ∧ S + L = sum_of_numbers

theorem find_multiple (L S : ℤ) (k : ℤ) (h1 : problem L S k) : k = 3 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_multiple_l805_80505


namespace NUMINAMATH_GPT_polygon_sides_l805_80513

theorem polygon_sides (h : ∀ (n : ℕ), (180 * (n - 2)) / n = 150) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l805_80513


namespace NUMINAMATH_GPT_time_to_cross_tree_l805_80518

def train_length : ℕ := 600
def platform_length : ℕ := 450
def time_to_pass_platform : ℕ := 105

-- Definition of the condition that leads to the speed of the train
def speed_of_train : ℚ := (train_length + platform_length) / time_to_pass_platform

-- Statement to prove the time to cross the tree
theorem time_to_cross_tree :
  (train_length : ℚ) / speed_of_train = 60 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_tree_l805_80518


namespace NUMINAMATH_GPT_angle_passing_through_point_l805_80558

-- Definition of the problem conditions
def is_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  let r := Real.sqrt (x^2 + y^2);
  (x = Real.cos α * r) ∧ (y = Real.sin α * r)

-- Lean 4 statement of the problem
theorem angle_passing_through_point (α : ℝ) :
  is_terminal_side_of_angle 1 (-1) α → α = - (Real.pi / 4) :=
by sorry

end NUMINAMATH_GPT_angle_passing_through_point_l805_80558


namespace NUMINAMATH_GPT_geometric_sequence_sum_l805_80559

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 7) 
  (h2 : a 2 + a 3 + a 4 = 14) 
  (geom_seq : ∃ q, ∀ n, a (n + 1) = q * a n ∧ q = 2) :
  a 4 + a 5 + a 6 = 56 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l805_80559


namespace NUMINAMATH_GPT_max_rank_awarded_l805_80548

theorem max_rank_awarded (num_participants rank_threshold total_possible_points : ℕ)
  (H1 : num_participants = 30)
  (H2 : rank_threshold = (30 * 29 / 2 : ℚ) * 0.60)
  (H3 : total_possible_points = (30 * 29 / 2)) :
  ∃ max_awarded : ℕ, max_awarded ≤ 23 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_max_rank_awarded_l805_80548


namespace NUMINAMATH_GPT_inequality_2_pow_n_gt_n_sq_for_n_5_l805_80512

theorem inequality_2_pow_n_gt_n_sq_for_n_5 : 2^5 > 5^2 := 
by {
    sorry -- Placeholder for the proof
}

end NUMINAMATH_GPT_inequality_2_pow_n_gt_n_sq_for_n_5_l805_80512


namespace NUMINAMATH_GPT_iodine_atomic_weight_l805_80549

noncomputable def atomic_weight_of_iodine : ℝ :=
  127.01

theorem iodine_atomic_weight
  (mw_AlI3 : ℝ := 408)
  (aw_Al : ℝ := 26.98)
  (formula_mw_AlI3 : mw_AlI3 = aw_Al + 3 * atomic_weight_of_iodine) :
  atomic_weight_of_iodine = 127.01 :=
by sorry

end NUMINAMATH_GPT_iodine_atomic_weight_l805_80549


namespace NUMINAMATH_GPT_f_of_3_l805_80566

def f (x : ℚ) : ℚ := (x + 3) / (x - 6)

theorem f_of_3 : f 3 = -2 := by
  sorry

end NUMINAMATH_GPT_f_of_3_l805_80566


namespace NUMINAMATH_GPT_cyclists_meet_fourth_time_l805_80598

theorem cyclists_meet_fourth_time 
  (speed1 speed2 speed3 speed4 : ℕ)
  (len : ℚ)
  (t_start : ℕ)
  (h_speed1 : speed1 = 6)
  (h_speed2 : speed2 = 9)
  (h_speed3 : speed3 = 12)
  (h_speed4 : speed4 = 15)
  (h_len : len = 1 / 3)
  (h_t_start : t_start = 12 * 60 * 60)
  : 
  (t_start + 4 * (20 * 60 + 40)) = 12 * 60 * 60 + 1600  :=
sorry

end NUMINAMATH_GPT_cyclists_meet_fourth_time_l805_80598


namespace NUMINAMATH_GPT_initial_fraction_spent_on_clothes_l805_80588

-- Define the conditions and the theorem to be proved
theorem initial_fraction_spent_on_clothes 
  (M : ℝ) (F : ℝ)
  (h1 : M = 249.99999999999994)
  (h2 : (3 / 4) * (4 / 5) * (1 - F) * M = 100) :
  F = 11 / 15 :=
sorry

end NUMINAMATH_GPT_initial_fraction_spent_on_clothes_l805_80588


namespace NUMINAMATH_GPT_remainder_when_x_minus_y_div_18_l805_80596

variable (k m : ℤ)
variable (x y : ℤ)
variable (h1 : x = 72 * k + 65)
variable (h2 : y = 54 * m + 22)

theorem remainder_when_x_minus_y_div_18 :
  (x - y) % 18 = 7 := by
sorry

end NUMINAMATH_GPT_remainder_when_x_minus_y_div_18_l805_80596


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l805_80530

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l805_80530
