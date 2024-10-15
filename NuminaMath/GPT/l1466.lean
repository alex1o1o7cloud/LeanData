import Mathlib

namespace NUMINAMATH_GPT_set_equality_proof_l1466_146654

theorem set_equality_proof :
  {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} :=
by
  sorry

end NUMINAMATH_GPT_set_equality_proof_l1466_146654


namespace NUMINAMATH_GPT_circle_common_chord_l1466_146681

theorem circle_common_chord (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧
  (x^2 + y^2 - 6 * x = 0) →
  (x + 3 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_circle_common_chord_l1466_146681


namespace NUMINAMATH_GPT_DeMorgansLaws_l1466_146683

variable (U : Type) (A B : Set U)

theorem DeMorgansLaws :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∧ (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ :=
by
  -- Statement of the theorems, proof is omitted
  sorry

end NUMINAMATH_GPT_DeMorgansLaws_l1466_146683


namespace NUMINAMATH_GPT_fraction_of_track_in_forest_l1466_146646

theorem fraction_of_track_in_forest (n : ℕ) (l : ℝ) (A B C : ℝ) :
  (∃ x, x = 2*l/3 ∨ x = l/3) → (∃ f, 0 < f ∧ f ≤ 1 ∧ (f = 2/3 ∨ f = 1/3)) :=
by
  -- sorry, the proof will go here
  sorry

end NUMINAMATH_GPT_fraction_of_track_in_forest_l1466_146646


namespace NUMINAMATH_GPT_S_13_eq_3510_l1466_146689

def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

theorem S_13_eq_3510 : S 13 = 3510 :=
by
  sorry

end NUMINAMATH_GPT_S_13_eq_3510_l1466_146689


namespace NUMINAMATH_GPT_jake_weight_l1466_146677

variable (J S : ℕ)

theorem jake_weight (h1 : J - 15 = 2 * S) (h2 : J + S = 132) : J = 93 := by
  sorry

end NUMINAMATH_GPT_jake_weight_l1466_146677


namespace NUMINAMATH_GPT_area_of_region_l1466_146695

noncomputable def T := 516

def region (x y : ℝ) : Prop :=
  |x| - |y| ≤ T - 500 ∧ |y| ≤ T - 500

theorem area_of_region :
  (4 * (T - 500)^2 = 1024) :=
  sorry

end NUMINAMATH_GPT_area_of_region_l1466_146695


namespace NUMINAMATH_GPT_minimization_problem_l1466_146686

theorem minimization_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) (h5 : x ≤ y) (h6 : y ≤ z) (h7 : z ≤ 3 * x) :
  x * y * z ≥ 1 / 18 := 
sorry

end NUMINAMATH_GPT_minimization_problem_l1466_146686


namespace NUMINAMATH_GPT_greatest_integer_gcd_four_l1466_146636

theorem greatest_integer_gcd_four {n : ℕ} (h1 : n < 150) (h2 : Nat.gcd n 12 = 4) : n <= 148 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_integer_gcd_four_l1466_146636


namespace NUMINAMATH_GPT_largest_value_l1466_146624

-- Define the five expressions as given in the conditions
def exprA : ℕ := 3 + 1 + 2 + 8
def exprB : ℕ := 3 * 1 + 2 + 8
def exprC : ℕ := 3 + 1 * 2 + 8
def exprD : ℕ := 3 + 1 + 2 * 8
def exprE : ℕ := 3 * 1 * 2 * 8

-- Define the theorem stating that exprE is the largest value
theorem largest_value : exprE = 48 ∧ exprE > exprA ∧ exprE > exprB ∧ exprE > exprC ∧ exprE > exprD := by
  sorry

end NUMINAMATH_GPT_largest_value_l1466_146624


namespace NUMINAMATH_GPT_trains_meet_time_l1466_146690

theorem trains_meet_time :
  (∀ (D : ℝ) (s1 s2 t1 t2 : ℝ),
    D = 155 ∧ 
    s1 = 20 ∧ 
    s2 = 25 ∧ 
    t1 = 7 ∧ 
    t2 = 8 →
    (∃ t : ℝ, 20 * t + 25 * t = D - 20)) →
  8 + 3 = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_trains_meet_time_l1466_146690


namespace NUMINAMATH_GPT_proof_problem_l1466_146685

-- Define the conditions for the problem

def is_factor (a b : ℕ) : Prop :=
  ∃ n : ℕ, b = a * n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Statement that needs to be proven
theorem proof_problem :
  is_factor 5 65 ∧ ¬(is_divisor 19 361 ∧ ¬is_divisor 19 190) ∧ ¬(¬is_divisor 36 144 ∨ ¬is_divisor 36 73) ∧ ¬(is_divisor 14 28 ∧ ¬is_divisor 14 56) ∧ is_factor 9 144 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1466_146685


namespace NUMINAMATH_GPT_total_fruits_is_174_l1466_146609

def basket1_apples : ℕ := 9
def basket1_oranges : ℕ := 15
def basket1_bananas : ℕ := 14
def basket1_grapes : ℕ := 12

def basket4_apples : ℕ := basket1_apples - 2
def basket4_oranges : ℕ := basket1_oranges - 2
def basket4_bananas : ℕ := basket1_bananas - 2
def basket4_grapes : ℕ := basket1_grapes - 2

def basket5_apples : ℕ := basket1_apples + 3
def basket5_oranges : ℕ := basket1_oranges - 5
def basket5_bananas : ℕ := basket1_bananas
def basket5_grapes : ℕ := basket1_grapes

def basket6_bananas : ℕ := basket1_bananas * 2
def basket6_grapes : ℕ := basket1_grapes / 2

def total_fruits_b1_3 : ℕ := basket1_apples + basket1_oranges + basket1_bananas + basket1_grapes
def total_fruits_b4 : ℕ := basket4_apples + basket4_oranges + basket4_bananas + basket4_grapes
def total_fruits_b5 : ℕ := basket5_apples + basket5_oranges + basket5_bananas + basket5_grapes
def total_fruits_b6 : ℕ := basket6_bananas + basket6_grapes

def total_fruits_all : ℕ := total_fruits_b1_3 + total_fruits_b4 + total_fruits_b5 + total_fruits_b6

theorem total_fruits_is_174 : total_fruits_all = 174 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_total_fruits_is_174_l1466_146609


namespace NUMINAMATH_GPT_fuchsia_to_mauve_l1466_146643

def fuchsia_to_mauve_amount (F : ℝ) : Prop :=
  let blue_in_fuchsia := (3 / 8) * F
  let red_in_fuchsia := (5 / 8) * F
  blue_in_fuchsia + 14 = 2 * red_in_fuchsia

theorem fuchsia_to_mauve (F : ℝ) (h : fuchsia_to_mauve_amount F) : F = 16 :=
by
  sorry

end NUMINAMATH_GPT_fuchsia_to_mauve_l1466_146643


namespace NUMINAMATH_GPT_min_ab_is_2sqrt6_l1466_146611

noncomputable def min_ab (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((2 / a) + (3 / b) = Real.sqrt (a * b)) then
      2 * Real.sqrt 6
  else
      0 -- or any other value, since this case should not occur in the context

theorem min_ab_is_2sqrt6 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 / a) + (3 / b) = Real.sqrt (a * b)) :
  min_ab a b = 2 * Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_min_ab_is_2sqrt6_l1466_146611


namespace NUMINAMATH_GPT_fraction_product_is_simplified_form_l1466_146673

noncomputable def fraction_product : ℚ := (2 / 3) * (5 / 11) * (3 / 8)

theorem fraction_product_is_simplified_form :
  fraction_product = 5 / 44 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_is_simplified_form_l1466_146673


namespace NUMINAMATH_GPT_band_and_chorus_but_not_orchestra_l1466_146614

theorem band_and_chorus_but_not_orchestra (B C O : Finset ℕ)
  (hB : B.card = 100) 
  (hC : C.card = 120) 
  (hO : O.card = 60)
  (hUnion : (B ∪ C ∪ O).card = 200)
  (hIntersection : (B ∩ C ∩ O).card = 10) : 
  ((B ∩ C).card - (B ∩ C ∩ O).card = 30) :=
by sorry

end NUMINAMATH_GPT_band_and_chorus_but_not_orchestra_l1466_146614


namespace NUMINAMATH_GPT_set_intersection_l1466_146649

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}
def B_complement : Set ℝ := {x | x ≥ 2}

theorem set_intersection :
  A ∩ B_complement = {x | 2 ≤ x ∧ x < 5} :=
by 
  sorry

end NUMINAMATH_GPT_set_intersection_l1466_146649


namespace NUMINAMATH_GPT_antipov_inequality_l1466_146668

theorem antipov_inequality (a b c : ℕ) 
  (h1 : ¬ (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) 
  (h2 : (ab + 1) ∣ (abc + 1)) : c ≥ b :=
sorry

end NUMINAMATH_GPT_antipov_inequality_l1466_146668


namespace NUMINAMATH_GPT_largest_fraction_among_fractions_l1466_146652

theorem largest_fraction_among_fractions :
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  (A < E) ∧ (B < E) ∧ (C < E) ∧ (D < E) :=
by
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  sorry

end NUMINAMATH_GPT_largest_fraction_among_fractions_l1466_146652


namespace NUMINAMATH_GPT_benjamin_trip_odd_number_conditions_l1466_146657

theorem benjamin_trip_odd_number_conditions (a b c : ℕ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a + b + c ≤ 9) 
  (h5 : ∃ x : ℕ, 60 * x = 99 * (c - a)) :
  a^2 + b^2 + c^2 = 35 := 
sorry

end NUMINAMATH_GPT_benjamin_trip_odd_number_conditions_l1466_146657


namespace NUMINAMATH_GPT_uniformColorGridPossible_l1466_146627

noncomputable def canPaintUniformColor (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) : Prop :=
  ∀ (row : Fin n), ∃ (c : Fin (n - 1)), ∀ (col : Fin n), G row col = c

theorem uniformColorGridPossible (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) :
  (∀ r : Fin n, ∃ c₁ c₂ : Fin n, c₁ ≠ c₂ ∧ G r c₁ = G r c₂) ∧
  (∀ c : Fin n, ∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ G r₁ c = G r₂ c) →
  ∃ c : Fin (n - 1), ∀ (row col : Fin n), G row col = c := by
  sorry

end NUMINAMATH_GPT_uniformColorGridPossible_l1466_146627


namespace NUMINAMATH_GPT_find_x_l1466_146612

theorem find_x (x : ℕ) : 8000 * 6000 = x * 10^5 → x = 480 := by
  sorry

end NUMINAMATH_GPT_find_x_l1466_146612


namespace NUMINAMATH_GPT_horner_v1_value_l1466_146650

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def horner (x : ℝ) (coeffs : List ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_v1_value :
  let x := 5
  let coeffs := [4, -12, 3.5, -2.6, 1.7, -0.8]
  let v0 := coeffs.head!
  let v1 := v0 * x + coeffs.getD 1 0
  v1 = 8 := by
  -- skip the actual proof steps
  sorry

end NUMINAMATH_GPT_horner_v1_value_l1466_146650


namespace NUMINAMATH_GPT_find_a4_l1466_146659

-- Define the arithmetic sequence and the sum of the first N terms
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first N terms in an arithmetic sequence
def sum_arithmetic_seq (a d N : ℕ) : ℕ := N * (2 * a + (N - 1) * d) / 2

-- Define the conditions
def condition1 (a d : ℕ) : Prop := a + (a + 2 * d) + (a + 4 * d) = 15
def condition2 (a d : ℕ) : Prop := sum_arithmetic_seq a d 4 = 16

-- Lean 4 statement to prove the value of a_4
theorem find_a4 (a d : ℕ) (h1 : condition1 a d) (h2 : condition2 a d) : arithmetic_seq a d 4 = 7 :=
sorry

end NUMINAMATH_GPT_find_a4_l1466_146659


namespace NUMINAMATH_GPT_difference_of_two_numbers_l1466_146682

def nat_sum := 22305
def a := ∃ a: ℕ, 5 ∣ a
def is_b (a b: ℕ) := b = a / 10 + 3

theorem difference_of_two_numbers (a b : ℕ) (h : a + b = nat_sum) (h1 : 5 ∣ a) (h2 : is_b a b) : a - b = 14872 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l1466_146682


namespace NUMINAMATH_GPT_age_ratio_l1466_146680

variable (R D : ℕ)

theorem age_ratio (h1 : D = 24) (h2 : R + 6 = 38) : R / D = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_age_ratio_l1466_146680


namespace NUMINAMATH_GPT_trivia_game_points_l1466_146613

theorem trivia_game_points (first_round_points second_round_points points_lost last_round_points : ℤ) 
    (h1 : first_round_points = 16)
    (h2 : second_round_points = 33)
    (h3 : points_lost = 48) : 
    first_round_points + second_round_points - points_lost = 1 :=
by
    rw [h1, h2, h3]
    rfl

end NUMINAMATH_GPT_trivia_game_points_l1466_146613


namespace NUMINAMATH_GPT_calc_value_l1466_146635

theorem calc_value (a b x : ℤ) (h₁ : a = 153) (h₂ : b = 147) (h₃ : x = 900) : x^2 / (a^2 - b^2) = 450 :=
by
  rw [h₁, h₂, h₃]
  -- Proof follows from the calculation in the provided steps
  sorry

end NUMINAMATH_GPT_calc_value_l1466_146635


namespace NUMINAMATH_GPT_line_through_circles_l1466_146665

theorem line_through_circles (D1 E1 D2 E2 : ℝ)
  (h1 : 2 * D1 - E1 + 2 = 0)
  (h2 : 2 * D2 - E2 + 2 = 0) :
  (2 * D1 - E1 + 2 = 0) ∧ (2 * D2 - E2 + 2 = 0) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_line_through_circles_l1466_146665


namespace NUMINAMATH_GPT_james_spent_6_dollars_l1466_146655

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end NUMINAMATH_GPT_james_spent_6_dollars_l1466_146655


namespace NUMINAMATH_GPT_ab_perpendicular_cd_l1466_146663

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assuming points are members of a metric space and distances are calculated using the distance function
variables (a b c d : A)

-- Given condition
def given_condition : Prop := 
  dist a c ^ 2 + dist b d ^ 2 = dist a d ^ 2 + dist b c ^ 2

-- Statement that needs to be proven
theorem ab_perpendicular_cd (h : given_condition a b c d) : dist a b * dist c d = 0 :=
sorry

end NUMINAMATH_GPT_ab_perpendicular_cd_l1466_146663


namespace NUMINAMATH_GPT_ratio_Mary_to_Seth_in_a_year_l1466_146632

-- Given conditions
def Seth_current_age : ℝ := 3.5
def age_difference : ℝ := 9

-- Definitions derived from conditions
def Mary_current_age : ℝ := Seth_current_age + age_difference
def Seth_age_in_a_year : ℝ := Seth_current_age + 1
def Mary_age_in_a_year : ℝ := Mary_current_age + 1

-- The statement to prove
theorem ratio_Mary_to_Seth_in_a_year : (Mary_age_in_a_year / Seth_age_in_a_year) = 3 := sorry

end NUMINAMATH_GPT_ratio_Mary_to_Seth_in_a_year_l1466_146632


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l1466_146648

noncomputable def evaluate_polynomial (x : ℂ) : ℂ :=
  x^100 + x^75 + x^50 + x^25 + 1

noncomputable def divisor_polynomial (x : ℂ) : ℂ :=
  x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_polynomial_division : 
  ∀ β : ℂ, divisor_polynomial β = 0 → evaluate_polynomial β = -1 :=
by
  intros β hβ
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l1466_146648


namespace NUMINAMATH_GPT_neg_eight_degrees_celsius_meaning_l1466_146628

-- Define the temperature in degrees Celsius
def temp_in_degrees_celsius (t : Int) : String :=
  if t >= 0 then toString t ++ "°C above zero"
  else toString (abs t) ++ "°C below zero"

-- Define the proof statement
theorem neg_eight_degrees_celsius_meaning :
  temp_in_degrees_celsius (-8) = "8°C below zero" :=
sorry

end NUMINAMATH_GPT_neg_eight_degrees_celsius_meaning_l1466_146628


namespace NUMINAMATH_GPT_solve_for_x_l1466_146660

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end NUMINAMATH_GPT_solve_for_x_l1466_146660


namespace NUMINAMATH_GPT_sam_original_seashells_count_l1466_146607

-- Definitions representing the conditions
def seashells_given_to_joan : ℕ := 18
def seashells_sam_has_now : ℕ := 17

-- The question and the answer translated to a proof problem
theorem sam_original_seashells_count :
  seashells_given_to_joan + seashells_sam_has_now = 35 :=
by
  sorry

end NUMINAMATH_GPT_sam_original_seashells_count_l1466_146607


namespace NUMINAMATH_GPT_new_boarders_joined_l1466_146647

theorem new_boarders_joined (boarders_initial day_students_initial boarders_final x : ℕ)
  (h1 : boarders_initial = 220)
  (h2 : (5:ℕ) * day_students_initial = (12:ℕ) * boarders_initial)
  (h3 : day_students_initial = 528)
  (h4 : (1:ℕ) * day_students_initial = (2:ℕ) * (boarders_initial + x)) :
  x = 44 := by
  sorry

end NUMINAMATH_GPT_new_boarders_joined_l1466_146647


namespace NUMINAMATH_GPT_calculation_result_l1466_146664

theorem calculation_result :
  (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 :=
by 
  sorry

end NUMINAMATH_GPT_calculation_result_l1466_146664


namespace NUMINAMATH_GPT_minimum_value_ineq_l1466_146674

theorem minimum_value_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l1466_146674


namespace NUMINAMATH_GPT_ducks_percentage_non_heron_birds_l1466_146651

theorem ducks_percentage_non_heron_birds
  (total_birds : ℕ)
  (geese_percent pelicans_percent herons_percent ducks_percent : ℝ)
  (H_geese : geese_percent = 20 / 100)
  (H_pelicans: pelicans_percent = 40 / 100)
  (H_herons : herons_percent = 15 / 100)
  (H_ducks : ducks_percent = 25 / 100)
  (hnz : total_birds ≠ 0) :
  (ducks_percent / (1 - herons_percent)) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_ducks_percentage_non_heron_birds_l1466_146651


namespace NUMINAMATH_GPT_running_hours_per_week_l1466_146670

theorem running_hours_per_week 
  (initial_days : ℕ) (additional_days : ℕ) (morning_run_time : ℕ) (evening_run_time : ℕ)
  (total_days : ℕ) (total_run_time_per_day : ℕ) (total_run_time_per_week : ℕ)
  (H1 : initial_days = 3)
  (H2 : additional_days = 2)
  (H3 : morning_run_time = 1)
  (H4 : evening_run_time = 1)
  (H5 : total_days = initial_days + additional_days)
  (H6 : total_run_time_per_day = morning_run_time + evening_run_time)
  (H7 : total_run_time_per_week = total_days * total_run_time_per_day) :
  total_run_time_per_week = 10 := 
sorry

end NUMINAMATH_GPT_running_hours_per_week_l1466_146670


namespace NUMINAMATH_GPT_infinite_fixpoints_l1466_146678

variable {f : ℕ+ → ℕ+}
variable (H : ∀ (m n : ℕ+), (∃ k : ℕ+ , k ≤ f n ∧ n ∣ f (m + k)) ∧ (∀ j : ℕ+ , j ≤ f n → j ≠ k → ¬ n ∣ f (m + j)))

theorem infinite_fixpoints : ∃ᶠ n in at_top, f n = n :=
sorry

end NUMINAMATH_GPT_infinite_fixpoints_l1466_146678


namespace NUMINAMATH_GPT_max_value_of_a_l1466_146697

noncomputable def maximum_a : ℝ := 1/3

theorem max_value_of_a :
  ∀ x : ℝ, 1 + maximum_a * Real.cos x ≥ (2/3) * Real.sin ((Real.pi / 2) + 2 * x) :=
by 
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1466_146697


namespace NUMINAMATH_GPT_tetrahedron_circumsphere_surface_area_eq_five_pi_l1466_146620

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def circumscribed_sphere_radius (a b : ℝ) : ℝ :=
  rectangle_diagonal a b / 2

noncomputable def circumscribed_sphere_surface_area (a b : ℝ) : ℝ :=
  4 * Real.pi * (circumscribed_sphere_radius a b)^2

theorem tetrahedron_circumsphere_surface_area_eq_five_pi :
  circumscribed_sphere_surface_area 2 1 = 5 * Real.pi := by
  sorry

end NUMINAMATH_GPT_tetrahedron_circumsphere_surface_area_eq_five_pi_l1466_146620


namespace NUMINAMATH_GPT_total_pizzas_eaten_l1466_146662

-- Definitions for the conditions
def pizzasA : ℕ := 8
def pizzasB : ℕ := 7

-- Theorem stating the total number of pizzas eaten by both classes
theorem total_pizzas_eaten : pizzasA + pizzasB = 15 := 
by
  -- Proof is not required for the task, so we use sorry
  sorry

end NUMINAMATH_GPT_total_pizzas_eaten_l1466_146662


namespace NUMINAMATH_GPT_arithmetic_progression_a_eq_1_l1466_146684

theorem arithmetic_progression_a_eq_1 
  (a : ℝ) 
  (h1 : 6 + 2 * a - 1 = 10 + 5 * a - (6 + 2 * a)) : 
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_a_eq_1_l1466_146684


namespace NUMINAMATH_GPT_sum_of_two_digit_divisors_l1466_146676

theorem sum_of_two_digit_divisors (d : ℕ) (h_pos : d > 0) (h_mod : 145 % d = 4) : d = 47 := 
by sorry

end NUMINAMATH_GPT_sum_of_two_digit_divisors_l1466_146676


namespace NUMINAMATH_GPT_base_n_representation_l1466_146605

theorem base_n_representation 
  (n : ℕ) 
  (hn : n > 0)
  (a b c : ℕ) 
  (ha : 0 ≤ a ∧ a < n)
  (hb : 0 ≤ b ∧ b < n) 
  (hc : 0 ≤ c ∧ c < n) 
  (h_digits_sum : a + b + c = 24)
  (h_base_repr : 1998 = a * n^2 + b * n + c) 
  : n = 15 ∨ n = 22 ∨ n = 43 :=
sorry

end NUMINAMATH_GPT_base_n_representation_l1466_146605


namespace NUMINAMATH_GPT_connie_marbles_l1466_146692

theorem connie_marbles (j c : ℕ) (h1 : j = 498) (h2 : j = c + 175) : c = 323 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_connie_marbles_l1466_146692


namespace NUMINAMATH_GPT_andy_demerits_l1466_146656

theorem andy_demerits (x : ℕ) :
  (∀ x, 6 * x + 15 = 27 → x = 2) :=
by
  intro
  sorry

end NUMINAMATH_GPT_andy_demerits_l1466_146656


namespace NUMINAMATH_GPT_cubic_inequality_l1466_146661

theorem cubic_inequality (x y z : ℝ) :
  x^3 + y^3 + z^3 + 3 * x * y * z ≥ x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) :=
sorry

end NUMINAMATH_GPT_cubic_inequality_l1466_146661


namespace NUMINAMATH_GPT_inequality_proof_l1466_146640

noncomputable def a := (1.01: ℝ) ^ (0.5: ℝ)
noncomputable def b := (1.01: ℝ) ^ (0.6: ℝ)
noncomputable def c := (0.6: ℝ) ^ (0.5: ℝ)

theorem inequality_proof : b > a ∧ a > c := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1466_146640


namespace NUMINAMATH_GPT_zero_points_of_f_l1466_146633

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f : (f (-1/2) = 0) ∧ (f (-1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_zero_points_of_f_l1466_146633


namespace NUMINAMATH_GPT_tickets_left_l1466_146617

theorem tickets_left (initial_tickets used_tickets tickets_left : ℕ) 
  (h1 : initial_tickets = 127) 
  (h2 : used_tickets = 84) : 
  tickets_left = initial_tickets - used_tickets := 
by
  sorry

end NUMINAMATH_GPT_tickets_left_l1466_146617


namespace NUMINAMATH_GPT_system_of_equations_xy_l1466_146630

theorem system_of_equations_xy (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = 5) :
  x - y = 2 := sorry

end NUMINAMATH_GPT_system_of_equations_xy_l1466_146630


namespace NUMINAMATH_GPT_total_houses_l1466_146629

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end NUMINAMATH_GPT_total_houses_l1466_146629


namespace NUMINAMATH_GPT_xyz_value_l1466_146616

-- Define the basic conditions
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (h1 : x * y = 40 * (4:ℝ)^(1/3))
variables (h2 : x * z = 56 * (4:ℝ)^(1/3))
variables (h3 : y * z = 32 * (4:ℝ)^(1/3))
variables (h4 : x + y = 18)

-- The target theorem
theorem xyz_value : x * y * z = 16 * (895:ℝ)^(1/2) :=
by
  -- Here goes the proof, but we add 'sorry' to end the theorem placeholder
  sorry

end NUMINAMATH_GPT_xyz_value_l1466_146616


namespace NUMINAMATH_GPT_total_order_cost_l1466_146699

theorem total_order_cost (n : ℕ) (cost_geo cost_eng : ℝ)
  (h1 : n = 35)
  (h2 : cost_geo = 10.50)
  (h3 : cost_eng = 7.50) :
  n * cost_geo + n * cost_eng = 630 := by
  -- proof steps should go here
  sorry

end NUMINAMATH_GPT_total_order_cost_l1466_146699


namespace NUMINAMATH_GPT_ranch_cows_variance_l1466_146669

variable (n : ℕ)
variable (p : ℝ)

-- Definition of the variance of a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem ranch_cows_variance : 
  binomial_variance 10 0.02 = 0.196 :=
by
  sorry

end NUMINAMATH_GPT_ranch_cows_variance_l1466_146669


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l1466_146671

theorem equation_of_perpendicular_line (a b c : ℝ) (p q : ℝ) (hx : a ≠ 0) (hy : b ≠ 0)
  (h_perpendicular : a * 2 + b * 1 = 0) (h_point : (-1) * a + 2 * b + c = 0)
  : a = 1 ∧ b = -2 ∧ c = -5 → (x:ℝ) * 1 + (y:ℝ) * (-2) + (-5) = 0 :=
by sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l1466_146671


namespace NUMINAMATH_GPT_triple_composition_l1466_146623

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end NUMINAMATH_GPT_triple_composition_l1466_146623


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l1466_146600

-- Define the number of wheels and axles conditions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def number_of_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the toll calculation formula
def toll (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

-- Lean theorem statement asserting that the toll for the given truck is 6 dollars
theorem toll_for_18_wheel_truck : toll number_of_axles = 6 := by
  -- Skipping the actual proof using sorry
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l1466_146600


namespace NUMINAMATH_GPT_smaller_angle_at_3_20_correct_l1466_146675

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_3_20_correct_l1466_146675


namespace NUMINAMATH_GPT_circle_radius_l1466_146601

theorem circle_radius 
  {XA XB XC r : ℝ}
  (h1 : XA = 3)
  (h2 : XB = 5)
  (h3 : XC = 1)
  (hx : XA * XB = XC * r)
  (hh : 2 * r = CD) :
  r = 8 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1466_146601


namespace NUMINAMATH_GPT_correct_number_of_statements_l1466_146625

-- Define the conditions as invalidity of the given statements
def statement_1_invalid : Prop := ¬ (true) -- INPUT a,b,c should use commas
def statement_2_invalid : Prop := ¬ (true) -- INPUT x=, 3 correct format
def statement_3_invalid : Prop := ¬ (true) -- 3=B , left side should be a variable name
def statement_4_invalid : Prop := ¬ (true) -- A=B=2, continuous assignment not allowed

-- Combine conditions
def all_statements_invalid : Prop := statement_1_invalid ∧ statement_2_invalid ∧ statement_3_invalid ∧ statement_4_invalid

-- State the theorem to prove
theorem correct_number_of_statements : all_statements_invalid → 0 = 0 := 
by sorry

end NUMINAMATH_GPT_correct_number_of_statements_l1466_146625


namespace NUMINAMATH_GPT_speed_of_first_train_l1466_146698

/-
Problem:
Two trains, with lengths 150 meters and 165 meters respectively, are running in opposite directions. One train is moving at 65 kmph, and they take 7.82006405004841 seconds to completely clear each other from the moment they meet. Prove that the speed of the first train is 79.99 kmph.
-/

theorem speed_of_first_train :
  ∀ (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) (speed1 : ℝ),
  length1 = 150 → length2 = 165 → speed2 = 65 → time = 7.82006405004841 →
  ( 3.6 * (length1 + length2) / time = speed1 + speed2 ) →
  speed1 = 79.99 :=
by
  intros length1 length2 speed2 time speed1 h_length1 h_length2 h_speed2 h_time h_formula
  rw [h_length1, h_length2, h_speed2, h_time] at h_formula
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l1466_146698


namespace NUMINAMATH_GPT_chord_length_l1466_146639

variable (x y : ℝ)

/--
The chord length cut by the line y = 2x - 2 on the circle (x-2)^2 + (y-2)^2 = 25 is 10.
-/
theorem chord_length (h₁ : y = 2 * x - 2) (h₂ : (x - 2)^2 + (y - 2)^2 = 25) : 
  ∃ length : ℝ, length = 10 :=
sorry

end NUMINAMATH_GPT_chord_length_l1466_146639


namespace NUMINAMATH_GPT_basketball_tournament_l1466_146606

theorem basketball_tournament (teams : Finset ℕ) (games_played : ℕ → ℕ → ℕ) (win_chance : ℕ → ℕ → Prop) 
(points : ℕ → ℕ) (X Y : ℕ) :
  teams.card = 6 → 
  (∀ t₁ t₂, t₁ ≠ t₂ → games_played t₁ t₂ = 1) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ ∨ win_chance t₂ t₁) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ → points t₁ = points t₁ + 1 ∧ points t₂ = points t₂) → 
  win_chance X Y →
  0.5 = 0.5 →
  0.5 * (1 - ((252 : ℚ) / 1024)) = (193 : ℚ) / 512 →
  ((63 : ℚ) / 256) + ((193 : ℚ) / 512) = (319 : ℚ) / 512 :=
by 
  sorry 

end NUMINAMATH_GPT_basketball_tournament_l1466_146606


namespace NUMINAMATH_GPT_Marla_colors_green_squares_l1466_146602

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end NUMINAMATH_GPT_Marla_colors_green_squares_l1466_146602


namespace NUMINAMATH_GPT_find_original_number_l1466_146653

theorem find_original_number (N : ℕ) (h : ∃ k : ℕ, N - 5 = 13 * k) : N = 18 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1466_146653


namespace NUMINAMATH_GPT_area_of_rectangular_field_l1466_146672

def length (L : ℝ) : Prop := L > 0
def breadth (L : ℝ) (B : ℝ) : Prop := B = 0.6 * L
def perimeter (L : ℝ) (B : ℝ) : Prop := 2 * L + 2 * B = 800
def area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

theorem area_of_rectangular_field (L B A : ℝ) 
  (h1 : breadth L B) 
  (h2 : perimeter L B) : 
  area L B 37500 :=
sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l1466_146672


namespace NUMINAMATH_GPT_geometric_sequence_tenth_term_l1466_146645

theorem geometric_sequence_tenth_term :
  let a := 5
  let r := 3 / 2
  let a_n (n : ℕ) := a * r ^ (n - 1)
  a_n 10 = 98415 / 512 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_tenth_term_l1466_146645


namespace NUMINAMATH_GPT_Bill_initial_money_l1466_146687

theorem Bill_initial_money (joint_money : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (final_bill_amount : ℕ) (initial_joint_money_eq : joint_money = 42) (pizza_cost_eq : pizza_cost = 11) (num_pizzas_eq : num_pizzas = 3) (final_bill_amount_eq : final_bill_amount = 39) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end NUMINAMATH_GPT_Bill_initial_money_l1466_146687


namespace NUMINAMATH_GPT_ratio_pen_pencil_l1466_146691

theorem ratio_pen_pencil (P : ℝ) (pencil_cost total_cost : ℝ) 
  (hc1 : pencil_cost = 8) 
  (hc2 : total_cost = 12)
  (hc3 : P + pencil_cost = total_cost) : 
  P / pencil_cost = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_pen_pencil_l1466_146691


namespace NUMINAMATH_GPT_apples_in_bowl_l1466_146638

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_bowl_l1466_146638


namespace NUMINAMATH_GPT_quad_eq_double_root_m_value_l1466_146626

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end NUMINAMATH_GPT_quad_eq_double_root_m_value_l1466_146626


namespace NUMINAMATH_GPT_paint_gallons_l1466_146644

theorem paint_gallons (W B : ℕ) (h1 : 5 * B = 8 * W) (h2 : W + B = 6689) : B = 4116 :=
by
  sorry

end NUMINAMATH_GPT_paint_gallons_l1466_146644


namespace NUMINAMATH_GPT_star_five_seven_l1466_146622

def star (a b : ℕ) : ℕ := (a + b + 3) ^ 2

theorem star_five_seven : star 5 7 = 225 := by
  sorry

end NUMINAMATH_GPT_star_five_seven_l1466_146622


namespace NUMINAMATH_GPT_tenth_term_is_19_over_4_l1466_146618

def nth_term_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

theorem tenth_term_is_19_over_4 :
  nth_term_arithmetic_sequence (1/4) (1/2) 10 = 19/4 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_is_19_over_4_l1466_146618


namespace NUMINAMATH_GPT_points_deducted_for_incorrect_answer_is_5_l1466_146696

-- Define the constants and variables used in the problem
def total_questions : ℕ := 30
def points_per_correct_answer : ℕ := 20
def correct_answers : ℕ := 19
def incorrect_answers : ℕ := total_questions - correct_answers
def final_score : ℕ := 325

-- Define a function that models the total score calculation
def calculate_final_score (points_deducted_per_incorrect : ℕ) : ℕ :=
  (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect)

-- The theorem that states the problem and expected solution
theorem points_deducted_for_incorrect_answer_is_5 :
  ∃ (x : ℕ), calculate_final_score x = final_score ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_points_deducted_for_incorrect_answer_is_5_l1466_146696


namespace NUMINAMATH_GPT_mixed_nuts_price_l1466_146693

theorem mixed_nuts_price (total_weight : ℝ) (peanut_price : ℝ) (cashew_price : ℝ) (cashew_weight : ℝ) 
  (H1 : total_weight = 100) 
  (H2 : peanut_price = 3.50) 
  (H3 : cashew_price = 4.00) 
  (H4 : cashew_weight = 60) : 
  (cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price) / total_weight = 3.80 :=
by 
  sorry

end NUMINAMATH_GPT_mixed_nuts_price_l1466_146693


namespace NUMINAMATH_GPT_f_2014_value_l1466_146603

def f : ℝ → ℝ :=
sorry

lemma f_periodic (x : ℝ) : f (x + 2) = f (x - 2) :=
sorry

lemma f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 4) : f x = x^2 :=
sorry

theorem f_2014_value : f 2014 = 4 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_f_2014_value_l1466_146603


namespace NUMINAMATH_GPT_largest_two_digit_number_divisible_by_6_and_ends_in_4_l1466_146666

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end NUMINAMATH_GPT_largest_two_digit_number_divisible_by_6_and_ends_in_4_l1466_146666


namespace NUMINAMATH_GPT_students_prefer_mac_l1466_146642

-- Define number of students in survey, and let M be the number who prefer Mac to Windows
variables (M E no_pref windows_pref : ℕ)
-- Total number of students surveyed
variable (total_students : ℕ)
-- Define that the total number of students is 210
axiom H_total : total_students = 210
-- Define that one third as many of the students who prefer Mac equally prefer both brands
axiom H_equal_preference : E = M / 3
-- Define that 90 students had no preference
axiom H_no_pref : no_pref = 90
-- Define that 40 students preferred Windows to Mac
axiom H_windows_pref : windows_pref = 40
-- Define that the total number of students is the sum of all groups
axiom H_students_sum : M + E + no_pref + windows_pref = total_students

-- The statement we need to prove
theorem students_prefer_mac :
  M = 60 :=
by sorry

end NUMINAMATH_GPT_students_prefer_mac_l1466_146642


namespace NUMINAMATH_GPT_transport_cost_l1466_146634

-- Define the conditions
def cost_per_kg : ℕ := 15000
def grams_per_kg : ℕ := 1000
def weight_in_grams : ℕ := 500

-- Define the main theorem stating the proof problem
theorem transport_cost
  (c : ℕ := cost_per_kg)
  (gpk : ℕ := grams_per_kg)
  (w : ℕ := weight_in_grams)
  : c * w / gpk = 7500 :=
by
  -- Since we are not required to provide the proof, adding sorry here
  sorry

end NUMINAMATH_GPT_transport_cost_l1466_146634


namespace NUMINAMATH_GPT_ricky_roses_l1466_146641

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end NUMINAMATH_GPT_ricky_roses_l1466_146641


namespace NUMINAMATH_GPT_quotient_base4_correct_l1466_146667

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 1302 => 1 * 4^3 + 3 * 4^2 + 0 * 4^1 + 2 * 4^0
  | 12 => 1 * 4^1 + 2 * 4^0
  | _ => 0

def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 19 => 1 * 4^2 + 0 * 4^1 + 3 * 4^0
  | _ => 0

theorem quotient_base4_correct : base10_to_base4 (114 / 6) = 103 := 
  by sorry

end NUMINAMATH_GPT_quotient_base4_correct_l1466_146667


namespace NUMINAMATH_GPT_range_of_k_l1466_146610

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- State the theorem
theorem range_of_k (k : ℝ) : (M ∩ N k).Nonempty ↔ k ∈ Set.Ici (-1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1466_146610


namespace NUMINAMATH_GPT_negation_of_exists_leq_zero_l1466_146604

theorem negation_of_exists_leq_zero (x : ℝ) : (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_leq_zero_l1466_146604


namespace NUMINAMATH_GPT_intersecting_lines_value_l1466_146621

theorem intersecting_lines_value (m b : ℚ)
  (h₁ : 10 = m * 7 + 5)
  (h₂ : 10 = 2 * 7 + b) :
  b + m = - (23 : ℚ) / 7 := 
sorry

end NUMINAMATH_GPT_intersecting_lines_value_l1466_146621


namespace NUMINAMATH_GPT_solution_set_f_ge_0_l1466_146679

noncomputable def f (x a : ℝ) : ℝ := 1 / Real.exp x - a / x

theorem solution_set_f_ge_0 (a m n : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ n ↔ 1 / Real.exp x - a / x ≥ 0) : 
  0 < a ∧ a < 1 / Real.exp 1 :=
  sorry

end NUMINAMATH_GPT_solution_set_f_ge_0_l1466_146679


namespace NUMINAMATH_GPT_simplify_exponent_expression_l1466_146619

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end NUMINAMATH_GPT_simplify_exponent_expression_l1466_146619


namespace NUMINAMATH_GPT_cab_speed_ratio_l1466_146688

variable (S_u S_c : ℝ)

theorem cab_speed_ratio (h1 : ∃ S_u S_c : ℝ, S_u * 25 = S_c * 30) : S_c / S_u = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_cab_speed_ratio_l1466_146688


namespace NUMINAMATH_GPT_regular_hours_l1466_146658

variable (R : ℕ)

theorem regular_hours (h1 : 5 * R + 6 * (44 - R) + 5 * R + 6 * (48 - R) = 472) : R = 40 :=
by
  sorry

end NUMINAMATH_GPT_regular_hours_l1466_146658


namespace NUMINAMATH_GPT_min_friend_pairs_l1466_146615

-- Define conditions
def n : ℕ := 2000
def invitations_per_person : ℕ := 1000
def total_invitations : ℕ := n * invitations_per_person

-- Mathematical problem statement
theorem min_friend_pairs : (total_invitations / 2) = 1000000 := 
by sorry

end NUMINAMATH_GPT_min_friend_pairs_l1466_146615


namespace NUMINAMATH_GPT_intersection_AB_l1466_146608

variable {x : ℝ}

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_AB : A ∩ B = {x | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_intersection_AB_l1466_146608


namespace NUMINAMATH_GPT_basketball_free_throws_l1466_146637

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = b) 
  (h3 : 2 * a + 3 * b + x = 73) : 
  x = 10 := 
by 
  sorry -- The actual proof is omitted as per the requirements.

end NUMINAMATH_GPT_basketball_free_throws_l1466_146637


namespace NUMINAMATH_GPT_conditions_for_k_b_l1466_146694

theorem conditions_for_k_b (k b : ℝ) :
  (∀ x : ℝ, (x - (kx + b) + 2) * (2) > 0) →
  (k = 1) ∧ (b < 2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_conditions_for_k_b_l1466_146694


namespace NUMINAMATH_GPT_books_sold_on_monday_75_l1466_146631

namespace Bookstore

variables (total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold : ℕ)
variable (percent_not_sold : ℝ)

def given_conditions : Prop :=
  total_books = 1200 ∧
  percent_not_sold = 0.665 ∧
  sold_Tuesday = 50 ∧
  sold_Wednesday = 64 ∧
  sold_Thursday = 78 ∧
  sold_Friday = 135 ∧
  books_not_sold = (percent_not_sold * total_books) ∧
  (total_books - books_not_sold) = (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Thursday + sold_Friday)

theorem books_sold_on_monday_75 (h : given_conditions total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold percent_not_sold) :
  sold_Monday = 75 :=
sorry

end Bookstore

end NUMINAMATH_GPT_books_sold_on_monday_75_l1466_146631
