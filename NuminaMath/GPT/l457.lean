import Mathlib

namespace NUMINAMATH_GPT_cars_to_hours_l457_45789

def car_interval := 20 -- minutes
def num_cars := 30
def minutes_per_hour := 60

theorem cars_to_hours :
  (car_interval * num_cars) / minutes_per_hour = 10 := by
  sorry

end NUMINAMATH_GPT_cars_to_hours_l457_45789


namespace NUMINAMATH_GPT_polygon_number_of_sides_l457_45743

theorem polygon_number_of_sides (h : ∀ (n : ℕ), (360 : ℝ) / (n : ℝ) = 1) : 
  360 = (1:ℝ) :=
  sorry

end NUMINAMATH_GPT_polygon_number_of_sides_l457_45743


namespace NUMINAMATH_GPT_solution_set_of_inequality_l457_45799

/-- Given an even function f that is monotonically increasing on [0, ∞) with f(3) = 0,
    show that the solution set for xf(2x - 1) < 0 is (-∞, -1) ∪ (0, 2). -/
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_value : f 3 = 0) :
  {x : ℝ | x * f (2*x - 1) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l457_45799


namespace NUMINAMATH_GPT_middle_number_divisible_by_4_l457_45736

noncomputable def three_consecutive_cubes_is_cube (x y : ℕ) : Prop :=
  (x-1)^3 + x^3 + (x+1)^3 = y^3

theorem middle_number_divisible_by_4 (x y : ℕ) (h : three_consecutive_cubes_is_cube x y) : 4 ∣ x :=
sorry

end NUMINAMATH_GPT_middle_number_divisible_by_4_l457_45736


namespace NUMINAMATH_GPT_crayons_left_l457_45792

-- Define the initial number of crayons
def initial_crayons : ℕ := 440

-- Define the crayons given away
def crayons_given : ℕ := 111

-- Define the crayons lost
def crayons_lost : ℕ := 106

-- Prove the final number of crayons left
theorem crayons_left : (initial_crayons - crayons_given - crayons_lost) = 223 :=
by
  sorry

end NUMINAMATH_GPT_crayons_left_l457_45792


namespace NUMINAMATH_GPT_lucy_snowballs_l457_45744

theorem lucy_snowballs : ∀ (c l : ℕ), c = l + 31 → c = 50 → l = 19 :=
by
  intros c l h1 h2
  sorry

end NUMINAMATH_GPT_lucy_snowballs_l457_45744


namespace NUMINAMATH_GPT_fraction_students_received_Bs_l457_45757

theorem fraction_students_received_Bs (fraction_As : ℝ) (fraction_As_or_Bs : ℝ) (h1 : fraction_As = 0.7) (h2 : fraction_As_or_Bs = 0.9) :
  fraction_As_or_Bs - fraction_As = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_students_received_Bs_l457_45757


namespace NUMINAMATH_GPT_problem1_problem2_l457_45707

-- Definitions based on the given conditions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := a^2 + a * b - 1

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 4 * a^2 + 5 * a * b - 2 * a - 3 :=
by sorry

-- Statement for problem (2)
theorem problem2 (a b : ℝ) (h : ∀ a, A a b - 2 * B a b = k) : 
  b = 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l457_45707


namespace NUMINAMATH_GPT_particular_number_l457_45782

theorem particular_number {x : ℕ} (h : x - 29 + 64 = 76) : x = 41 := by
  sorry

end NUMINAMATH_GPT_particular_number_l457_45782


namespace NUMINAMATH_GPT_six_pow_2n_plus1_plus_1_div_by_7_l457_45706

theorem six_pow_2n_plus1_plus_1_div_by_7 (n : ℕ) : (6^(2*n+1) + 1) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_six_pow_2n_plus1_plus_1_div_by_7_l457_45706


namespace NUMINAMATH_GPT_problem_statement_l457_45776

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variable (a x y t : ℝ) 

theorem problem_statement : 
  (log_base a x + 3 * log_base x a - log_base x y = 3) ∧ (a > 1) ∧ (x = a ^ t) ∧ (0 < t ∧ t ≤ 2) ∧ (y = 8) 
  → (a = 16) ∧ (x = 64) := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l457_45776


namespace NUMINAMATH_GPT_tiling_scenarios_unique_l457_45784

theorem tiling_scenarios_unique (m n : ℕ) 
  (h1 : 60 * m + 150 * n = 360) : m = 1 ∧ n = 2 :=
by {
  -- The proof will be provided here
  sorry
}

end NUMINAMATH_GPT_tiling_scenarios_unique_l457_45784


namespace NUMINAMATH_GPT_jessica_repay_l457_45752

theorem jessica_repay (P : ℝ) (r : ℝ) (n : ℝ) (x : ℕ)
  (hx : P = 20)
  (hr : r = 0.12)
  (hn : n = 3 * P) :
  x = 17 :=
sorry

end NUMINAMATH_GPT_jessica_repay_l457_45752


namespace NUMINAMATH_GPT_complex_division_l457_45756

open Complex

theorem complex_division :
  (1 + 2 * I) / (3 - 4 * I) = -1 / 5 + 2 / 5 * I :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l457_45756


namespace NUMINAMATH_GPT_equation_of_line_l_l457_45751

noncomputable def line_eq (a b c : ℚ) : ℚ → ℚ → Prop := λ x y => a * x + b * y + c = 0

theorem equation_of_line_l : 
  ∃ m : ℚ, 
  (∀ x y : ℚ, 
    (2 * x - 3 * y - 3 = 0 ∧ x + y + 2 = 0 → line_eq 3 1 m x y) ∧ 
    (3 * x + y - 1 = 0 → line_eq 3 1 0 x y)
  ) →
  line_eq 15 5 16 (-3/5) (-7/5) :=
by 
  sorry

end NUMINAMATH_GPT_equation_of_line_l_l457_45751


namespace NUMINAMATH_GPT_problem_l457_45715

noncomputable def discriminant (p q : ℝ) : ℝ := p^2 - 4 * q
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (p q : ℝ) (hq : q = -2 * p - 5) :
  (quadratic 1 p (q + 1) 2 = 0) →
  q = -2 * p - 5 ∧
  discriminant p q > 0 ∧
  (discriminant p (q + 1) = 0 → 
    (p = -4 ∧ q = 3 ∧ ∀ x : ℝ, quadratic 1 p q x = 0 ↔ (x = 1 ∨ x = 3))) :=
by
  intro hroot_eq
  sorry

end NUMINAMATH_GPT_problem_l457_45715


namespace NUMINAMATH_GPT_numeral_diff_local_face_value_l457_45714

theorem numeral_diff_local_face_value (P : ℕ) :
  7 * (10 ^ P - 1) = 693 → P = 2 ∧ (N = 700) :=
by
  intro h
  -- The actual proof is not required hence we insert sorry
  sorry

end NUMINAMATH_GPT_numeral_diff_local_face_value_l457_45714


namespace NUMINAMATH_GPT_smallest_positive_period_and_symmetry_l457_45794

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + (7 * Real.pi / 4)) + 
  Real.cos (x - (3 * Real.pi / 4))

theorem smallest_positive_period_and_symmetry :
  (∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ a, a = - (Real.pi / 4) ∧ ∀ x, f (2 * a - x) = f x) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_and_symmetry_l457_45794


namespace NUMINAMATH_GPT_minimum_waste_l457_45798

/-- Zenobia's cookout problem setup -/
def LCM_hot_dogs_buns : Nat := Nat.lcm 10 12

def hot_dog_packages : Nat := LCM_hot_dogs_buns / 10
def bun_packages : Nat := LCM_hot_dogs_buns / 12

def waste_hot_dog_packages : ℝ := hot_dog_packages * 0.4
def waste_bun_packages : ℝ := bun_packages * 0.3
def total_waste : ℝ := waste_hot_dog_packages + waste_bun_packages

theorem minimum_waste :
  hot_dog_packages = 6 ∧ bun_packages = 5 ∧ total_waste = 3.9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_waste_l457_45798


namespace NUMINAMATH_GPT_cost_of_500_pencils_in_dollars_l457_45780

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end NUMINAMATH_GPT_cost_of_500_pencils_in_dollars_l457_45780


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l457_45750

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arithmetic : ∀ n, a n = a1 + (n - 1) * d)
  (h_a4 : a 4 = 5) :
  2 * a 1 - a 5 + a 11 = 10 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l457_45750


namespace NUMINAMATH_GPT_find_circle_center_l457_45727

noncomputable def midpoint_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_center (x y : ℝ) : Prop :=
  6 * x - 5 * y = midpoint_line 40 (-20) ∧ 3 * x + 2 * y = 0

theorem find_circle_center : circle_center (20 / 27) (-10 / 9) :=
by
  -- Here would go the proof steps, but we skip it
  sorry

end NUMINAMATH_GPT_find_circle_center_l457_45727


namespace NUMINAMATH_GPT_mail_distribution_l457_45740

def total_mail : ℕ := 2758
def mail_for_first_block : ℕ := 365
def mail_for_second_block : ℕ := 421
def remaining_mail : ℕ := total_mail - (mail_for_first_block + mail_for_second_block)
def remaining_blocks : ℕ := 3
def mail_per_remaining_block : ℕ := remaining_mail / remaining_blocks

theorem mail_distribution :
  mail_per_remaining_block = 657 := by
  sorry

end NUMINAMATH_GPT_mail_distribution_l457_45740


namespace NUMINAMATH_GPT_negation_is_correct_l457_45710

-- Define the condition: we have two integers a and b
variables (a b : ℤ)

-- Original proposition: If the sum of two integers is even, then both integers are even.
def original_proposition := (a + b) % 2 = 0 → (a % 2 = 0) ∧ (b % 2 = 0)

-- Negation of the proposition: There exist two integers such that their sum is even and not both are even.
def negation_of_proposition := (a + b) % 2 = 0 ∧ ¬((a % 2 = 0) ∧ (b % 2 = 0))

theorem negation_is_correct :
  ¬ original_proposition a b = negation_of_proposition a b :=
by
  sorry

end NUMINAMATH_GPT_negation_is_correct_l457_45710


namespace NUMINAMATH_GPT_cost_of_1000_pieces_of_gum_l457_45705

theorem cost_of_1000_pieces_of_gum
  (cost_per_piece : ℕ)
  (num_pieces : ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (conversion_rate : ℕ)
  (h_cost : cost_per_piece = 2)
  (h_pieces : num_pieces = 1000)
  (h_threshold : discount_threshold = 500)
  (h_discount : discount_rate = 0.90)
  (h_conversion : conversion_rate = 100)
  (h_more_than_threshold : num_pieces > discount_threshold) :
  (num_pieces * cost_per_piece * discount_rate) / conversion_rate = 18 := 
sorry

end NUMINAMATH_GPT_cost_of_1000_pieces_of_gum_l457_45705


namespace NUMINAMATH_GPT_remaining_dresses_pockets_count_l457_45701

-- Definitions translating each condition in the problem.
def total_dresses : Nat := 24
def dresses_with_pockets : Nat := total_dresses / 2
def dresses_with_two_pockets : Nat := dresses_with_pockets / 3
def total_pockets : Nat := 32

-- Question translated into a proof problem using Lean's logic.
theorem remaining_dresses_pockets_count :
  (total_pockets - (dresses_with_two_pockets * 2)) / (dresses_with_pockets - dresses_with_two_pockets) = 3 := by
  sorry

end NUMINAMATH_GPT_remaining_dresses_pockets_count_l457_45701


namespace NUMINAMATH_GPT_volume_inequality_holds_l457_45763

def volume (x : ℕ) : ℤ :=
  (x^2 - 16) * (x^3 + 25)

theorem volume_inequality_holds :
  ∃ (n : ℕ), n = 1 ∧ ∃ x : ℕ, volume x < 1000 ∧ (x - 4) > 0 :=
by
  sorry

end NUMINAMATH_GPT_volume_inequality_holds_l457_45763


namespace NUMINAMATH_GPT_fraction_is_one_third_l457_45774

noncomputable def fraction_studying_japanese (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : ℚ :=
  J / (J + S)

theorem fraction_is_one_third (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : 
  fraction_studying_japanese J S h1 h2 = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_fraction_is_one_third_l457_45774


namespace NUMINAMATH_GPT_wrapping_paper_l457_45742

theorem wrapping_paper (total_used : ℚ) (decoration_used : ℚ) (presents : ℕ) (other_presents : ℕ) (individual_used : ℚ) 
  (h1 : total_used = 5 / 8) 
  (h2 : decoration_used = 1 / 24) 
  (h3 : presents = 4) 
  (h4 : other_presents = 3) 
  (h5 : individual_used = (5 / 8 - 1 / 24) / 3) : 
  individual_used = 7 / 36 := 
by
  -- The theorem will be proven here.
  sorry

end NUMINAMATH_GPT_wrapping_paper_l457_45742


namespace NUMINAMATH_GPT_machines_work_together_l457_45771

theorem machines_work_together (x : ℝ) (h₁ : 1/(x+4) + 1/(x+2) + 1/(x+3) = 1/x) : x = 1 :=
sorry

end NUMINAMATH_GPT_machines_work_together_l457_45771


namespace NUMINAMATH_GPT_equation_solution_l457_45791

theorem equation_solution (x : ℝ) (h : 8^(Real.log 5 / Real.log 8) = 10 * x + 3) : x = 1 / 5 :=
sorry

end NUMINAMATH_GPT_equation_solution_l457_45791


namespace NUMINAMATH_GPT_movie_theatre_total_seats_l457_45717

theorem movie_theatre_total_seats (A C : ℕ) 
  (hC : C = 188) 
  (hRevenue : 6 * A + 4 * C = 1124) 
  : A + C = 250 :=
by
  sorry

end NUMINAMATH_GPT_movie_theatre_total_seats_l457_45717


namespace NUMINAMATH_GPT_rowing_distance_l457_45760

theorem rowing_distance (v_b : ℝ) (v_s : ℝ) (t_total : ℝ) (D : ℝ) :
  v_b = 9 → v_s = 1.5 → t_total = 48 → D / (v_b + v_s) + D / (v_b - v_s) = t_total → D = 210 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rowing_distance_l457_45760


namespace NUMINAMATH_GPT_ball_box_distribution_l457_45723

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end NUMINAMATH_GPT_ball_box_distribution_l457_45723


namespace NUMINAMATH_GPT_youngest_child_age_l457_45739

variable (Y : ℕ) (O : ℕ) -- Y: the youngest child's present age
variable (P₀ P₁ P₂ P₃ : ℕ) -- P₀, P₁, P₂, P₃: the present ages of the 4 original family members

-- Conditions translated to Lean
variable (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
variable (h₂ : O = Y + 2)
variable (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24)

theorem youngest_child_age (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
                       (h₂ : O = Y + 2)
                       (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24) :
  Y = 3 := by 
  sorry

end NUMINAMATH_GPT_youngest_child_age_l457_45739


namespace NUMINAMATH_GPT_symmetric_points_parabola_l457_45764

theorem symmetric_points_parabola (x1 x2 y1 y2 m : ℝ) (h1 : y1 = 2 * x1^2) (h2 : y2 = 2 * x2^2)
    (h3 : x1 * x2 = -3 / 4) (h_sym: (y2 - y1) / (x2 - x1) = -1)
    (h_mid: (y2 + y1) / 2 = (x2 + x1) / 2 + m) :
    m = 2 := sorry

end NUMINAMATH_GPT_symmetric_points_parabola_l457_45764


namespace NUMINAMATH_GPT_tan_alpha_value_l457_45738

theorem tan_alpha_value
  (α : ℝ)
  (h_cos : Real.cos α = -4/5)
  (h_range : (Real.pi / 2) < α ∧ α < Real.pi) :
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_value_l457_45738


namespace NUMINAMATH_GPT_stickers_given_l457_45788

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end NUMINAMATH_GPT_stickers_given_l457_45788


namespace NUMINAMATH_GPT_Veronica_to_Half_Samir_Ratio_l457_45733

-- Mathematical conditions 
def Samir_stairs : ℕ := 318
def Total_stairs : ℕ := 495
def Half_Samir_stairs : ℚ := Samir_stairs / 2

-- Definition for Veronica's stairs as a multiple of half Samir's stairs
def Veronica_stairs (R: ℚ) : ℚ := R * Half_Samir_stairs

-- Lean statement to prove the ratio
theorem Veronica_to_Half_Samir_Ratio (R : ℚ) (H1 : Veronica_stairs R + Samir_stairs = Total_stairs) : R = 1.1132 := 
by
  sorry

end NUMINAMATH_GPT_Veronica_to_Half_Samir_Ratio_l457_45733


namespace NUMINAMATH_GPT_proof_problem_l457_45793

noncomputable def find_values (a b c x y z : ℝ) := 
  14 * x + b * y + c * z = 0 ∧ 
  a * x + 24 * y + c * z = 0 ∧ 
  a * x + b * y + 43 * z = 0 ∧ 
  a ≠ 14 ∧ b ≠ 24 ∧ c ≠ 43 ∧ x ≠ 0

theorem proof_problem (a b c x y z : ℝ) 
  (h : find_values a b c x y z):
  (a / (a - 14)) + (b / (b - 24)) + (c / (c - 43)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l457_45793


namespace NUMINAMATH_GPT_smallest_possible_value_of_d_l457_45712

noncomputable def smallest_value_of_d : ℝ :=
  2 + Real.sqrt 2

theorem smallest_possible_value_of_d (c d : ℝ) (h1 : 2 < c) (h2 : c < d)
    (triangle_condition1 : ¬ (2 + c > d ∧ 2 + d > c ∧ c + d > 2))
    (triangle_condition2 : ¬ ( (2 / d) + (2 / c) > 2)) : d = smallest_value_of_d :=
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_d_l457_45712


namespace NUMINAMATH_GPT_heat_of_neutralization_combination_l457_45728

-- Define instruments
inductive Instrument
| Balance
| MeasuringCylinder
| Beaker
| Burette
| Thermometer
| TestTube
| AlcoholLamp

def correct_combination : List Instrument :=
  [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer]

theorem heat_of_neutralization_combination :
  correct_combination = [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer] :=
sorry

end NUMINAMATH_GPT_heat_of_neutralization_combination_l457_45728


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l457_45773

theorem basic_astrophysics_degrees :
  let microphotonics_pct := 12
  let home_electronics_pct := 24
  let food_additives_pct := 15
  let gmo_pct := 29
  let industrial_lubricants_pct := 8
  let total_budget_percentage := 100
  let full_circle_degrees := 360
  let given_pct_sum := microphotonics_pct + home_electronics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct
  let astrophysics_pct := total_budget_percentage - given_pct_sum
  let astrophysics_degrees := (astrophysics_pct * full_circle_degrees) / total_budget_percentage
  astrophysics_degrees = 43.2 := by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l457_45773


namespace NUMINAMATH_GPT_abs_diff_eq_two_l457_45781

def equation (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

theorem abs_diff_eq_two (a b e : ℝ) (ha : equation e a) (hb : equation e b) (hab : a ≠ b) :
  |a - b| = 2 :=
sorry

end NUMINAMATH_GPT_abs_diff_eq_two_l457_45781


namespace NUMINAMATH_GPT_rose_paid_after_discount_l457_45711

noncomputable def discount_percentage : ℝ := 0.1
noncomputable def original_price : ℝ := 10
noncomputable def discount_amount := discount_percentage * original_price
noncomputable def final_price := original_price - discount_amount

theorem rose_paid_after_discount : final_price = 9 := by
  sorry

end NUMINAMATH_GPT_rose_paid_after_discount_l457_45711


namespace NUMINAMATH_GPT_geometric_sequence_a12_l457_45745

noncomputable def a_n (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r ^ (n - 1)

theorem geometric_sequence_a12 (a1 r : ℝ) 
  (h1 : a_n a1 r 7 * a_n a1 r 9 = 4)
  (h2 : a_n a1 r 4 = 1) :
  a_n a1 r 12 = 16 := sorry

end NUMINAMATH_GPT_geometric_sequence_a12_l457_45745


namespace NUMINAMATH_GPT_inequality_l457_45766

theorem inequality (A B : ℝ) (n : ℕ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hn : 1 ≤ n) : (A + B)^n ≤ 2^(n - 1) * (A^n + B^n) := 
  sorry

end NUMINAMATH_GPT_inequality_l457_45766


namespace NUMINAMATH_GPT_bottles_more_than_apples_l457_45746

def bottles_regular : ℕ := 72
def bottles_diet : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := bottles_regular + bottles_diet

theorem bottles_more_than_apples : (total_bottles - apples) = 26 := by
  sorry

end NUMINAMATH_GPT_bottles_more_than_apples_l457_45746


namespace NUMINAMATH_GPT_find_sum_of_digits_l457_45775

theorem find_sum_of_digits (a c : ℕ) (h1 : 200 + 10 * a + 3 + 427 = 600 + 10 * c + 9) (h2 : (600 + 10 * c + 9) % 3 = 0) : a + c = 4 :=
sorry

end NUMINAMATH_GPT_find_sum_of_digits_l457_45775


namespace NUMINAMATH_GPT_proportionality_problem_l457_45726

noncomputable def find_x (z w : ℝ) (k : ℝ) : ℝ :=
  k / (z^(3/2) * w^2)

theorem proportionality_problem :
  ∃ k : ℝ, 
    (find_x 16 2 k = 5) ∧
    (find_x 64 4 k = 5 / 32) :=
by
  sorry

end NUMINAMATH_GPT_proportionality_problem_l457_45726


namespace NUMINAMATH_GPT_walkway_area_l457_45724

theorem walkway_area (l w : ℕ) (walkway_width : ℕ) (total_length total_width pool_area walkway_area : ℕ)
  (hl : l = 20) 
  (hw : w = 8)
  (hww : walkway_width = 1)
  (htl : total_length = l + 2 * walkway_width)
  (htw : total_width = w + 2 * walkway_width)
  (hpa : pool_area = l * w)
  (hta : (total_length * total_width) = pool_area + walkway_area) :
  walkway_area = 60 := 
  sorry

end NUMINAMATH_GPT_walkway_area_l457_45724


namespace NUMINAMATH_GPT_unique_prime_sum_and_diff_l457_45720

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def is_sum_of_two_primes (p : ℕ) : Prop :=
  ∃ q1 q2 : ℕ, is_prime q1 ∧ is_prime q2 ∧ p = q1 + q2

noncomputable def is_diff_of_two_primes (p : ℕ) : Prop :=
  ∃ q3 q4 : ℕ, is_prime q3 ∧ is_prime q4 ∧ q3 > q4 ∧ p = q3 - q4

theorem unique_prime_sum_and_diff :
  ∀ p : ℕ, is_prime p ∧ is_sum_of_two_primes p ∧ is_diff_of_two_primes p ↔ p = 5 := 
by
  sorry

end NUMINAMATH_GPT_unique_prime_sum_and_diff_l457_45720


namespace NUMINAMATH_GPT_worth_of_used_car_l457_45786

theorem worth_of_used_car (earnings remaining : ℝ) (earnings_eq : earnings = 5000) (remaining_eq : remaining = 1000) : 
  ∃ worth : ℝ, worth = earnings - remaining ∧ worth = 4000 :=
by
  sorry

end NUMINAMATH_GPT_worth_of_used_car_l457_45786


namespace NUMINAMATH_GPT_club_members_remainder_l457_45708

theorem club_members_remainder (N : ℕ) (h1 : 50 < N) (h2 : N < 80)
  (h3 : N % 5 = 0) (h4 : N % 8 = 0 ∨ N % 7 = 0) :
  N % 9 = 6 ∨ N % 9 = 7 := by
  sorry

end NUMINAMATH_GPT_club_members_remainder_l457_45708


namespace NUMINAMATH_GPT_television_price_reduction_l457_45749

theorem television_price_reduction (P : ℝ) (h₁ : 0 ≤ P):
  ((P - (P * 0.7 * 0.8)) / P) * 100 = 44 :=
by
  sorry

end NUMINAMATH_GPT_television_price_reduction_l457_45749


namespace NUMINAMATH_GPT_line_through_point_parallel_l457_45772

theorem line_through_point_parallel 
    (x y : ℝ)
    (h0 : (x = -1) ∧ (y = 3))
    (h1 : ∃ c : ℝ, (∀ x y : ℝ, x - 2 * y + c = 0 ↔ x - 2 * y + 3 = 0)) :
     ∃ c : ℝ, ∀ x y : ℝ, (x = -1) ∧ (y = 3) → (∃ (a b : ℝ), a - 2 * b + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_l457_45772


namespace NUMINAMATH_GPT_range_of_slope_ellipse_chord_l457_45795

theorem range_of_slope_ellipse_chord :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
    (x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
    (x₁^2 + y₁^2 / 4 = 1 ∧ x₂^2 + y₂^2 / 4 = 1) →
    ((1 / 2) ≤ y₀ ∧ y₀ ≤ 1) →
    (-4 ≤ -2 / y₀ ∧ -2 / y₀ ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_slope_ellipse_chord_l457_45795


namespace NUMINAMATH_GPT_new_numbers_are_reciprocals_l457_45719

variable {x y : ℝ}

theorem new_numbers_are_reciprocals (h : (1 / x) + (1 / y) = 1) : 
  (x - 1 = 1 / (y - 1)) ∧ (y - 1 = 1 / (x - 1)) := 
by
  sorry

end NUMINAMATH_GPT_new_numbers_are_reciprocals_l457_45719


namespace NUMINAMATH_GPT_surface_area_comparison_l457_45748

theorem surface_area_comparison (a R : ℝ) (h_eq_volumes : (4 / 3) * Real.pi * R^3 = a^3) :
  6 * a^2 > 4 * Real.pi * R^2 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_comparison_l457_45748


namespace NUMINAMATH_GPT_min_tablets_to_get_two_each_l457_45783

def least_tablets_to_ensure_two_each (A B : ℕ) (A_eq : A = 10) (B_eq : B = 10) : ℕ :=
  if A ≥ 2 ∧ B ≥ 2 then 4 else 12

theorem min_tablets_to_get_two_each :
  least_tablets_to_ensure_two_each 10 10 rfl rfl = 12 :=
by
  sorry

end NUMINAMATH_GPT_min_tablets_to_get_two_each_l457_45783


namespace NUMINAMATH_GPT_transform_map_ABCD_to_A_l457_45718

structure Point :=
(x : ℤ)
(y : ℤ)

structure Rectangle :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)

def transform180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

def rect_transform180 (rect : Rectangle) : Rectangle :=
  { A := transform180 rect.A,
    B := transform180 rect.B,
    C := transform180 rect.C,
    D := transform180 rect.D }

def ABCD := Rectangle.mk ⟨-3, 2⟩ ⟨-1, 2⟩ ⟨-1, 5⟩ ⟨-3, 5⟩
def A'B'C'D' := Rectangle.mk ⟨3, -2⟩ ⟨1, -2⟩ ⟨1, -5⟩ ⟨3, -5⟩

theorem transform_map_ABCD_to_A'B'C'D' :
  rect_transform180 ABCD = A'B'C'D' :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_transform_map_ABCD_to_A_l457_45718


namespace NUMINAMATH_GPT_annual_interest_rate_is_correct_l457_45730

theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, r = 0.0583 ∧
  (200 * (1 + r)^2 = 224) :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_is_correct_l457_45730


namespace NUMINAMATH_GPT_jane_stick_length_l457_45731

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end NUMINAMATH_GPT_jane_stick_length_l457_45731


namespace NUMINAMATH_GPT_fraction_of_historical_fiction_new_releases_l457_45787

theorem fraction_of_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_percentage : ℝ := 0.4)
  (historical_fiction_new_releases_percentage : ℝ := 0.4)
  (other_genres_new_releases_percentage : ℝ := 0.7)
  (total_historical_fiction_books := total_books * historical_fiction_percentage)
  (total_other_books := total_books * (1 - historical_fiction_percentage))
  (historical_fiction_new_releases := total_historical_fiction_books * historical_fiction_new_releases_percentage)
  (other_genres_new_releases := total_other_books * other_genres_new_releases_percentage)
  (total_new_releases := historical_fiction_new_releases + other_genres_new_releases) :
  historical_fiction_new_releases / total_new_releases = 8 / 29 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_historical_fiction_new_releases_l457_45787


namespace NUMINAMATH_GPT_pool_fill_time_l457_45709

theorem pool_fill_time
  (faster_pipe_time : ℝ) (slower_pipe_factor : ℝ)
  (H1 : faster_pipe_time = 9) 
  (H2 : slower_pipe_factor = 1.25) : 
  (faster_pipe_time * (1 + slower_pipe_factor) / (faster_pipe_time + faster_pipe_time/slower_pipe_factor)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_pool_fill_time_l457_45709


namespace NUMINAMATH_GPT_find_a_l457_45779

variable (m : ℝ)

def root1 := 2 * m - 1
def root2 := m + 4

theorem find_a (h : root1 ^ 2 = root2 ^ 2) : ∃ a : ℝ, a = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l457_45779


namespace NUMINAMATH_GPT_gcd_256_162_450_l457_45762

theorem gcd_256_162_450 : Nat.gcd (Nat.gcd 256 162) 450 = 2 := sorry

end NUMINAMATH_GPT_gcd_256_162_450_l457_45762


namespace NUMINAMATH_GPT_luisa_trip_l457_45735

noncomputable def additional_miles (d1: ℝ) (s1: ℝ) (s2: ℝ) (desired_avg_speed: ℝ) : ℝ := 
  let t1 := d1 / s1
  let t := (d1 * (desired_avg_speed - s1)) / (s2 * (s1 - desired_avg_speed))
  s2 * t

theorem luisa_trip :
  additional_miles 18 36 60 45 = 18 :=
by
  sorry

end NUMINAMATH_GPT_luisa_trip_l457_45735


namespace NUMINAMATH_GPT_third_median_length_l457_45796

theorem third_median_length (m1 m2 area : ℝ) (h1 : m1 = 5) (h2 : m2 = 10) (h3 : area = 10 * Real.sqrt 10) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_third_median_length_l457_45796


namespace NUMINAMATH_GPT_total_selling_price_l457_45722

theorem total_selling_price 
  (n : ℕ) (p : ℕ) (c : ℕ) 
  (h_n : n = 85) (h_p : p = 15) (h_c : c = 85) : 
  (c + p) * n = 8500 :=
by
  sorry

end NUMINAMATH_GPT_total_selling_price_l457_45722


namespace NUMINAMATH_GPT_min_value_of_expr_l457_45729

theorem min_value_of_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = (1 / a) + (1 / b)) :
  ∃ x : ℝ, x = (1 / a) + (2 / b) ∧ x = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_expr_l457_45729


namespace NUMINAMATH_GPT_alex_buys_17p3_pounds_of_corn_l457_45777

noncomputable def pounds_of_corn (c b : ℝ) : Prop :=
    c + b = 30 ∧ 1.05 * c + 0.39 * b = 23.10

theorem alex_buys_17p3_pounds_of_corn :
    ∃ c b, pounds_of_corn c b ∧ c = 17.3 :=
by
    sorry

end NUMINAMATH_GPT_alex_buys_17p3_pounds_of_corn_l457_45777


namespace NUMINAMATH_GPT_acute_triangle_cannot_divide_into_two_obtuse_l457_45732

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem acute_triangle_cannot_divide_into_two_obtuse (A B C A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_divide : A + B + C = 180 ∧ A1 + B1 + C1 = 180 ∧ A2 + B2 + C2 = 180)
  (h_sum : A1 + A2 = A ∧ B1 + B2 = B ∧ C1 + C2 = C) :
  ¬ (is_obtuse_triangle A1 B1 C1 ∧ is_obtuse_triangle A2 B2 C2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_cannot_divide_into_two_obtuse_l457_45732


namespace NUMINAMATH_GPT_tory_sold_to_neighbor_l457_45759

def total_cookies : ℕ := 50
def sold_to_grandmother : ℕ := 12
def sold_to_uncle : ℕ := 7
def to_be_sold : ℕ := 26

def sold_to_neighbor : ℕ :=
  total_cookies - to_be_sold - (sold_to_grandmother + sold_to_uncle)

theorem tory_sold_to_neighbor :
  sold_to_neighbor = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tory_sold_to_neighbor_l457_45759


namespace NUMINAMATH_GPT_paul_money_duration_l457_45704

theorem paul_money_duration (mowing_income weed_eating_income weekly_spending money_last: ℕ) 
    (h1: mowing_income = 44) 
    (h2: weed_eating_income = 28) 
    (h3: weekly_spending = 9) 
    (h4: money_last = 8) 
    : (mowing_income + weed_eating_income) / weekly_spending = money_last := 
by
  sorry

end NUMINAMATH_GPT_paul_money_duration_l457_45704


namespace NUMINAMATH_GPT_binary_to_decimal_110101_l457_45700

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 53) :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_110101_l457_45700


namespace NUMINAMATH_GPT_jesse_bananas_total_l457_45737

theorem jesse_bananas_total (friends : ℝ) (bananas_per_friend : ℝ) (friends_eq : friends = 3) (bananas_per_friend_eq : bananas_per_friend = 21) : 
  friends * bananas_per_friend = 63 := by
  rw [friends_eq, bananas_per_friend_eq]
  norm_num

end NUMINAMATH_GPT_jesse_bananas_total_l457_45737


namespace NUMINAMATH_GPT_dot_product_to_linear_form_l457_45703

noncomputable def proof_problem (r a : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := a.1
  let B := a.2
  let C := -m
  (r.1 * a.1 + r.2 * a.2 = m) → (A * r.1 + B * r.2 + C = 0)

-- The theorem statement
theorem dot_product_to_linear_form (r a : ℝ × ℝ) (m : ℝ) :
  proof_problem r a m :=
sorry

end NUMINAMATH_GPT_dot_product_to_linear_form_l457_45703


namespace NUMINAMATH_GPT_avg_of_five_consecutive_from_b_l457_45790

-- Conditions
def avg_of_five_even_consecutive (a : ℕ) : ℕ := (2 * a + (2 * a + 2) + (2 * a + 4) + (2 * a + 6) + (2 * a + 8)) / 5

-- The main theorem
theorem avg_of_five_consecutive_from_b (a : ℕ) : 
  avg_of_five_even_consecutive a = 2 * a + 4 → 
  ((2 * a + 4 + (2 * a + 4 + 1) + (2 * a + 4 + 2) + (2 * a + 4 + 3) + (2 * a + 4 + 4)) / 5) = 2 * a + 6 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_five_consecutive_from_b_l457_45790


namespace NUMINAMATH_GPT_pascal_triangle_probability_l457_45721

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end NUMINAMATH_GPT_pascal_triangle_probability_l457_45721


namespace NUMINAMATH_GPT_total_ninja_stars_l457_45758

variable (e c j : ℕ)
variable (H1 : e = 4) -- Eric has 4 ninja throwing stars
variable (H2 : c = 2 * e) -- Chad has twice as many ninja throwing stars as Eric
variable (H3 : j = c - 2) -- Chad sells 2 ninja stars to Jeff
variable (H4 : j = 6) -- Jeff now has 6 ninja throwing stars

theorem total_ninja_stars :
  e + (c - 2) + 6 = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_ninja_stars_l457_45758


namespace NUMINAMATH_GPT_product_of_possible_values_l457_45770

theorem product_of_possible_values (x : ℚ) (h : abs ((18 : ℚ) / (2 * x) - 4) = 3) : (x = 9 ∨ x = 9/7) → (9 * (9/7) = 81/7) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_possible_values_l457_45770


namespace NUMINAMATH_GPT_find_a_from_circle_and_chord_l457_45702

theorem find_a_from_circle_and_chord 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0)
  (line_eq : ∀ x y : ℝ, x + y + 2 = 0)
  (chord_length : ∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 + 2*x1 - 2*y1 + a = 0 ∧ x2^2 + y2^2 + 2*x2 - 2*y2 + a = 0 ∧ x1 + y1 + 2 = 0 ∧ x2 + y2 + 2 = 0 → (x1 - x2)^2 + (y1 - y2)^2 = 16) :
  a = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_circle_and_chord_l457_45702


namespace NUMINAMATH_GPT_solve_system_of_equations_l457_45765

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y = z) ∧ (x * z = y) ∧ (y * z = x) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 0 ∧ y = 0 ∧ z = 0) := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l457_45765


namespace NUMINAMATH_GPT_root_of_polynomial_l457_45753

theorem root_of_polynomial :
  ∀ x : ℝ, (x^2 - 3 * x + 2) * x * (x - 4) = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 4) :=
by 
  sorry

end NUMINAMATH_GPT_root_of_polynomial_l457_45753


namespace NUMINAMATH_GPT_extreme_value_a_one_range_of_a_l457_45768

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 3

theorem extreme_value_a_one :
  ∀ x > 0, f x 1 ≤ f 1 1 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) → a ≥ Real.exp 2 :=
sorry

end NUMINAMATH_GPT_extreme_value_a_one_range_of_a_l457_45768


namespace NUMINAMATH_GPT_packets_of_candy_bought_l457_45713

theorem packets_of_candy_bought
    (candies_per_day_weekday : ℕ)
    (candies_per_day_weekend : ℕ)
    (days_weekday : ℕ)
    (days_weekend : ℕ)
    (weeks : ℕ)
    (candies_per_packet : ℕ)
    (total_candies : ℕ)
    (packets_bought : ℕ) :
    candies_per_day_weekday = 2 →
    candies_per_day_weekend = 1 →
    days_weekday = 5 →
    days_weekend = 2 →
    weeks = 3 →
    candies_per_packet = 18 →
    total_candies = (candies_per_day_weekday * days_weekday + candies_per_day_weekend * days_weekend) * weeks →
    packets_bought = total_candies / candies_per_packet →
    packets_bought = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_packets_of_candy_bought_l457_45713


namespace NUMINAMATH_GPT_example_problem_l457_45769

def Z (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem example_problem :
  Z 4 3 = -11 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_example_problem_l457_45769


namespace NUMINAMATH_GPT_carpet_breadth_l457_45716

theorem carpet_breadth
  (b : ℝ)
  (h1 : ∀ b, ∃ l, l = 1.44 * b)
  (h2 : 4082.4 = 45 * ((1.40 * l) * (1.25 * b)))
  : b = 6.08 :=
by
  sorry

end NUMINAMATH_GPT_carpet_breadth_l457_45716


namespace NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l457_45741

theorem sum_ratio_arithmetic_sequence (a₁ d : ℚ) (h : d ≠ 0) 
  (S : ℕ → ℚ)
  (h_sum : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l457_45741


namespace NUMINAMATH_GPT_perimeter_triangle_ABC_eq_18_l457_45755

theorem perimeter_triangle_ABC_eq_18 (h1 : ∀ (Δ : ℕ), Δ = 9) 
(h2 : ∀ (p : ℕ), p = 6) : 
∀ (perimeter_ABC : ℕ), perimeter_ABC = 18 := by
sorry

end NUMINAMATH_GPT_perimeter_triangle_ABC_eq_18_l457_45755


namespace NUMINAMATH_GPT_range_of_fx_over_x_l457_45734

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_fx_over_x (odd_f : is_odd f)
                           (increasing_f_pos : is_increasing_on f {x : ℝ | x > 0})
                           (hf1 : f (-1) = 0) :
  {x | f x / x < 0} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_range_of_fx_over_x_l457_45734


namespace NUMINAMATH_GPT_min_bottles_l457_45754

theorem min_bottles (a b : ℕ) (h1 : a > b) (h2 : b > 1) : 
  ∃ x : ℕ, x = Nat.ceil (a - a / b) := sorry

end NUMINAMATH_GPT_min_bottles_l457_45754


namespace NUMINAMATH_GPT_parabola_equation_l457_45797

theorem parabola_equation (p : ℝ) (h_pos : p > 0) (M : ℝ) (h_Mx : M = 3) (h_MF : abs (M + p/2) = 2 * p) :
  (forall x y, y^2 = 2 * p * x) -> (forall x y, y^2 = 4 * x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l457_45797


namespace NUMINAMATH_GPT_total_polled_votes_correct_l457_45767

variable (V : ℕ) -- Valid votes

-- Condition: One candidate got 30% of the valid votes
variable (C1_votes : ℕ) (C2_votes : ℕ)
variable (H1 : C1_votes = (3 * V) / 10)

-- Condition: The other candidate won by 5000 votes
variable (H2 : C2_votes = C1_votes + 5000)

-- Condition: One candidate got 70% of the valid votes
variable (H3 : C2_votes = (7 * V) / 10)

-- Condition: 100 votes were invalid
variable (invalid_votes : ℕ := 100)

-- Total polled votes (valid + invalid)
def total_polled_votes := V + invalid_votes

theorem total_polled_votes_correct 
  (V : ℕ) 
  (H1 : C1_votes = (3 * V) / 10) 
  (H2 : C2_votes = C1_votes + 5000) 
  (H3 : C2_votes = (7 * V) / 10) 
  (invalid_votes : ℕ := 100) : 
  total_polled_votes V = 12600 :=
by
  -- The steps of the proof are omitted
  sorry

end NUMINAMATH_GPT_total_polled_votes_correct_l457_45767


namespace NUMINAMATH_GPT_circumference_of_cone_l457_45747

theorem circumference_of_cone (V : ℝ) (h : ℝ) (C : ℝ) 
  (hV : V = 36 * Real.pi) (hh : h = 3) : 
  C = 12 * Real.pi :=
sorry

end NUMINAMATH_GPT_circumference_of_cone_l457_45747


namespace NUMINAMATH_GPT_number_of_girls_l457_45725

theorem number_of_girls (B G : ℕ) 
  (h1 : B = G + 124) 
  (h2 : B + G = 1250) : G = 563 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l457_45725


namespace NUMINAMATH_GPT_candy_problem_minimum_candies_l457_45778

theorem candy_problem_minimum_candies : ∃ (N : ℕ), N > 1 ∧ N % 2 = 1 ∧ N % 3 = 1 ∧ N % 5 = 1 ∧ N = 31 :=
by
  sorry

end NUMINAMATH_GPT_candy_problem_minimum_candies_l457_45778


namespace NUMINAMATH_GPT_Vince_ride_longer_l457_45761

def Vince_ride_length : ℝ := 0.625
def Zachary_ride_length : ℝ := 0.5

theorem Vince_ride_longer : Vince_ride_length - Zachary_ride_length = 0.125 := by
  sorry

end NUMINAMATH_GPT_Vince_ride_longer_l457_45761


namespace NUMINAMATH_GPT_least_k_divisible_by_240_l457_45785

theorem least_k_divisible_by_240 : ∃ (k : ℕ), k^2 % 240 = 0 ∧ k = 60 :=
by
  sorry

end NUMINAMATH_GPT_least_k_divisible_by_240_l457_45785
