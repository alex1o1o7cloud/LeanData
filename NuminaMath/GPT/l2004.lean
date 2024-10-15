import Mathlib

namespace NUMINAMATH_GPT_measure_of_angle_B_l2004_200441

noncomputable def angle_opposite_side (a b c : ℝ) (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : ℝ :=
  if h : (c^2)/(a+b) + (a^2)/(b+c) = b then 60 else 0

theorem measure_of_angle_B {a b c : ℝ} (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : 
  angle_opposite_side a b c h = 60 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_l2004_200441


namespace NUMINAMATH_GPT_financier_invariant_l2004_200429

theorem financier_invariant (D A : ℤ) (hD : D = 1 ∨ D = 10 * (A - 1) + D ∨ D = D - 1 + 10 * A)
  (hA : A = 0 ∨ A = A + 10 * (1 - D) ∨ A = A - 1):
  (D - A) % 11 = 1 := 
sorry

end NUMINAMATH_GPT_financier_invariant_l2004_200429


namespace NUMINAMATH_GPT_remainder_T2015_mod_12_eq_8_l2004_200456

-- Define sequences of length n consisting of the letters A and B,
-- with no more than two A's in a row and no more than two B's in a row
def T : ℕ → ℕ :=
  sorry  -- Definition for T(n) must follow the given rules

-- Theorem to prove that T(2015) modulo 12 equals 8
theorem remainder_T2015_mod_12_eq_8 :
  (T 2015) % 12 = 8 :=
  sorry

end NUMINAMATH_GPT_remainder_T2015_mod_12_eq_8_l2004_200456


namespace NUMINAMATH_GPT_thre_digit_num_condition_l2004_200486

theorem thre_digit_num_condition (n : ℕ) (h : n = 735) :
  (n % 35 = 0) ∧ (Nat.digits 10 n).sum = 15 := by
  sorry

end NUMINAMATH_GPT_thre_digit_num_condition_l2004_200486


namespace NUMINAMATH_GPT_smallest_discount_n_l2004_200407

noncomputable def effective_discount_1 (x : ℝ) : ℝ := 0.64 * x
noncomputable def effective_discount_2 (x : ℝ) : ℝ := 0.614125 * x
noncomputable def effective_discount_3 (x : ℝ) : ℝ := 0.63 * x 

theorem smallest_discount_n (x : ℝ) (n : ℕ) (hx : x > 0) :
  (1 - n / 100 : ℝ) * x < effective_discount_1 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_2 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_3 x ↔ n = 39 := 
sorry

end NUMINAMATH_GPT_smallest_discount_n_l2004_200407


namespace NUMINAMATH_GPT_leak_drains_in_34_hours_l2004_200406

-- Define the conditions
def pump_rate := 1 / 2 -- rate at which the pump fills the tank (tanks per hour)
def time_with_leak := 17 / 8 -- time to fill the tank with the pump and the leak (hours)

-- Define the combined rate of pump and leak
def combined_rate := 1 / time_with_leak -- tanks per hour

-- Define the leak rate
def leak_rate := pump_rate - combined_rate -- solve for leak rate

-- Define the proof statement
theorem leak_drains_in_34_hours : (1 / leak_rate) = 34 := by
    sorry

end NUMINAMATH_GPT_leak_drains_in_34_hours_l2004_200406


namespace NUMINAMATH_GPT_part_I_part_II_l2004_200475

-- Part I: Inequality solution
theorem part_I (x : ℝ) : 
  (abs (x - 1) ≥ 4 - abs (x - 3)) ↔ (x ≤ 0 ∨ x ≥ 4) := 
sorry

-- Part II: Minimum value of mn
theorem part_II (m n : ℕ) (h1 : (1:ℝ)/m + (1:ℝ)/(2*n) = 1) (hm : 0 < m) (hn : 0 < n) :
  (mn : ℕ) = 2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2004_200475


namespace NUMINAMATH_GPT_find_x_l2004_200427

theorem find_x : ∃ x : ℤ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 ∧ x = 28 := 
by sorry

end NUMINAMATH_GPT_find_x_l2004_200427


namespace NUMINAMATH_GPT_mrs_oaklyn_profit_is_correct_l2004_200413

def cost_of_buying_rugs (n : ℕ) (cost_per_rug : ℕ) : ℕ :=
  n * cost_per_rug

def transportation_fee (n : ℕ) (fee_per_rug : ℕ) : ℕ :=
  n * fee_per_rug

def selling_price_before_tax (n : ℕ) (price_per_rug : ℕ) : ℕ :=
  n * price_per_rug

def total_tax (price_before_tax : ℕ) (tax_rate : ℕ) : ℕ :=
  price_before_tax * tax_rate / 100

def total_selling_price_after_tax (price_before_tax : ℕ) (tax_amount : ℕ) : ℕ :=
  price_before_tax + tax_amount

def profit (selling_price_after_tax : ℕ) (cost_of_buying : ℕ) (transport_fee : ℕ) : ℕ :=
  selling_price_after_tax - (cost_of_buying + transport_fee)

def rugs := 20
def cost_per_rug := 40
def transport_fee_per_rug := 5
def price_per_rug := 60
def tax_rate := 10

theorem mrs_oaklyn_profit_is_correct : 
  profit 
    (total_selling_price_after_tax 
      (selling_price_before_tax rugs price_per_rug) 
      (total_tax (selling_price_before_tax rugs price_per_rug) tax_rate)
    )
    (cost_of_buying_rugs rugs cost_per_rug) 
    (transportation_fee rugs transport_fee_per_rug) 
  = 420 :=
by sorry

end NUMINAMATH_GPT_mrs_oaklyn_profit_is_correct_l2004_200413


namespace NUMINAMATH_GPT_min_S_n_condition_l2004_200420

noncomputable def a_n (n : ℕ) : ℤ := -28 + 4 * (n - 1)

noncomputable def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

theorem min_S_n_condition : S_n 7 = S_n 8 ∧ (∀ m < 7, S_n m > S_n 7) ∧ (∀ m < 8, S_n m > S_n 8) := 
by
  sorry

end NUMINAMATH_GPT_min_S_n_condition_l2004_200420


namespace NUMINAMATH_GPT_parallelogram_sides_l2004_200411

theorem parallelogram_sides (x y : ℕ) 
  (h₁ : 2 * x + 3 = 9) 
  (h₂ : 8 * y - 1 = 7) : 
  x + y = 4 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_sides_l2004_200411


namespace NUMINAMATH_GPT_solve_y_l2004_200417

theorem solve_y (y : ℝ) (h : (y ^ (7 / 8)) = 4) : y = 2 ^ (16 / 7) :=
sorry

end NUMINAMATH_GPT_solve_y_l2004_200417


namespace NUMINAMATH_GPT_find_digit_property_l2004_200465

theorem find_digit_property (a x : ℕ) (h : 10 * a + x = a + x + a * x) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_digit_property_l2004_200465


namespace NUMINAMATH_GPT_mary_needs_more_apples_l2004_200468

theorem mary_needs_more_apples (total_pies : ℕ) (apples_per_pie : ℕ) (harvested_apples : ℕ) (y : ℕ) :
  total_pies = 10 → apples_per_pie = 8 → harvested_apples = 50 → y = 30 :=
by
  intro h1 h2 h3
  have total_apples_needed := total_pies * apples_per_pie
  have apples_needed_to_buy := total_apples_needed - harvested_apples
  have proof_needed : apples_needed_to_buy = y := sorry
  have proof_given : y = 30 := sorry
  have apples_needed := total_pies * apples_per_pie - harvested_apples
  exact proof_given

end NUMINAMATH_GPT_mary_needs_more_apples_l2004_200468


namespace NUMINAMATH_GPT_AB_passes_fixed_point_locus_of_N_l2004_200460

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the point M which is the right-angle vertex
def M : ℝ × ℝ := (1, 2)

-- Statement for Part 1: Prove line AB passes through a fixed point
theorem AB_passes_fixed_point 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) :
    ∃ P : ℝ × ℝ, P = (5, -2) := sorry

-- Statement for Part 2: Find the locus of point N
theorem locus_of_N 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) 
    (N : ℝ × ℝ)
    (hN : ∃ t : ℝ, N = (t, -(t - 3))) :
    (N.1 - 3)^2 + N.2^2 = 8 ∧ N.1 ≠ 1 := sorry

end NUMINAMATH_GPT_AB_passes_fixed_point_locus_of_N_l2004_200460


namespace NUMINAMATH_GPT_sin_330_l2004_200430

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end NUMINAMATH_GPT_sin_330_l2004_200430


namespace NUMINAMATH_GPT_brenda_initial_peaches_l2004_200404

variable (P : ℕ)

def brenda_conditions (P : ℕ) : Prop :=
  let fresh_peaches := P - 15
  (P > 15) ∧ (fresh_peaches * 60 = 100 * 150)

theorem brenda_initial_peaches : ∃ (P : ℕ), brenda_conditions P ∧ P = 250 :=
by
  sorry

end NUMINAMATH_GPT_brenda_initial_peaches_l2004_200404


namespace NUMINAMATH_GPT_converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l2004_200450

variable (a b : ℝ)

theorem converse_of_proposition :
  (ab > 0 → a > 0 ∧ b > 0) = false := sorry

theorem inverse_of_proposition :
  (a ≤ 0 ∨ b ≤ 0 → ab ≤ 0) = false := sorry

theorem contrapositive_of_proposition :
  (ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) = true := sorry

end NUMINAMATH_GPT_converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l2004_200450


namespace NUMINAMATH_GPT_no_nat_fourfold_digit_move_l2004_200400

theorem no_nat_fourfold_digit_move :
  ¬ ∃ (N : ℕ), ∃ (a : ℕ), ∃ (n : ℕ), ∃ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (N = a * 10^n + x) ∧ 
    (4 * N = 10 * x + a) :=
by
  sorry

end NUMINAMATH_GPT_no_nat_fourfold_digit_move_l2004_200400


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_six_l2004_200423

theorem cos_alpha_plus_pi_six (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (α + Real.pi / 6) = - (4 / 5) := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_pi_six_l2004_200423


namespace NUMINAMATH_GPT_amusement_park_ticket_price_l2004_200426

-- Conditions as definitions in Lean
def weekday_adult_ticket_cost : ℕ := 22
def weekday_children_ticket_cost : ℕ := 7
def weekend_adult_ticket_cost : ℕ := 25
def weekend_children_ticket_cost : ℕ := 10
def adult_discount_rate : ℕ := 20
def sales_tax_rate : ℕ := 10
def num_of_adults : ℕ := 2
def num_of_children : ℕ := 2

-- Correct Answer to be proved equivalent:
def expected_total_price := 66

-- Statement translating the problem to Lean proof obligation
theorem amusement_park_ticket_price :
  let cost_before_discount := (num_of_adults * weekend_adult_ticket_cost) + (num_of_children * weekend_children_ticket_cost)
  let discount := (num_of_adults * weekend_adult_ticket_cost) * adult_discount_rate / 100
  let subtotal := cost_before_discount - discount
  let sales_tax := subtotal * sales_tax_rate / 100
  let total_cost := subtotal + sales_tax
  total_cost = expected_total_price :=
by
  sorry

end NUMINAMATH_GPT_amusement_park_ticket_price_l2004_200426


namespace NUMINAMATH_GPT_boys_playing_both_sports_l2004_200493

theorem boys_playing_both_sports : 
  ∀ (total boys basketball football neither both : ℕ), 
  total = 22 → boys = 22 → basketball = 13 → football = 15 → neither = 3 → 
  boys = basketball + football - both + neither → 
  both = 9 :=
by
  intros total boys basketball football neither both
  intros h_total h_boys h_basketball h_football h_neither h_formula
  sorry

end NUMINAMATH_GPT_boys_playing_both_sports_l2004_200493


namespace NUMINAMATH_GPT_fewest_colored_paper_l2004_200459
   
   /-- Jungkook, Hoseok, and Seokjin shared colored paper. 
       Jungkook took 10 cards, Hoseok took 7, and Seokjin took 2 less than Jungkook. 
       Prove that Hoseok took the fewest pieces of colored paper. -/
   theorem fewest_colored_paper 
       (Jungkook Hoseok Seokjin : ℕ)
       (hj : Jungkook = 10)
       (hh : Hoseok = 7)
       (hs : Seokjin = Jungkook - 2) :
       Hoseok < Jungkook ∧ Hoseok < Seokjin :=
   by
     sorry
   
end NUMINAMATH_GPT_fewest_colored_paper_l2004_200459


namespace NUMINAMATH_GPT_positive_number_square_roots_l2004_200453

theorem positive_number_square_roots (m : ℝ) 
  (h : (2 * m - 1) + (2 - m) = 0) :
  (2 - m)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_square_roots_l2004_200453


namespace NUMINAMATH_GPT_number_of_shoes_lost_l2004_200418

-- Definitions for the problem conditions
def original_pairs : ℕ := 20
def pairs_left : ℕ := 15
def shoes_per_pair : ℕ := 2

-- Translating the conditions to individual shoe counts
def original_shoes : ℕ := original_pairs * shoes_per_pair
def remaining_shoes : ℕ := pairs_left * shoes_per_pair

-- Statement of the proof problem
theorem number_of_shoes_lost : original_shoes - remaining_shoes = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_shoes_lost_l2004_200418


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2004_200494

theorem solve_equation_1 (x : ℝ) : x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6 := 
by sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2004_200494


namespace NUMINAMATH_GPT_simple_interest_sum_l2004_200416

variable {P R : ℝ}

theorem simple_interest_sum :
  P * (R + 6) = P * R + 3000 → P = 500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_simple_interest_sum_l2004_200416


namespace NUMINAMATH_GPT_total_flowers_l2004_200495

def number_of_flowers (F : ℝ) : Prop :=
  let vases := (F - 7.0) / 6.0
  vases = 6.666666667

theorem total_flowers : number_of_flowers 47.0 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_l2004_200495


namespace NUMINAMATH_GPT_contrapositive_false_1_negation_false_1_l2004_200485

theorem contrapositive_false_1 (m : ℝ) : ¬ (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

theorem negation_false_1 (m : ℝ) : ¬ ((m > 0) → ¬ (∃ x : ℝ, x^2 + x - m = 0)) :=
sorry

end NUMINAMATH_GPT_contrapositive_false_1_negation_false_1_l2004_200485


namespace NUMINAMATH_GPT_sign_of_b_l2004_200462

variable (a b : ℝ)

theorem sign_of_b (h1 : (a + b > 0 ∨ a - b > 0) ∧ (a + b < 0 ∨ a - b < 0)) 
                  (h2 : (ab > 0 ∨ a / b > 0) ∧ (ab < 0 ∨ a / b < 0))
                  (h3 : (ab > 0 → a > 0 ∧ b > 0) ∨ (ab < 0 → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0))) :
  b < 0 :=
sorry

end NUMINAMATH_GPT_sign_of_b_l2004_200462


namespace NUMINAMATH_GPT_circle_area_radius_8_l2004_200444

variable (r : ℝ) (π : ℝ)

theorem circle_area_radius_8 : r = 8 → (π * r^2) = 64 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_radius_8_l2004_200444


namespace NUMINAMATH_GPT_Trisha_walked_total_distance_l2004_200410

theorem Trisha_walked_total_distance 
  (d1 d2 d3 : ℝ) (h_d1 : d1 = 0.11) (h_d2 : d2 = 0.11) (h_d3 : d3 = 0.67) :
  d1 + d2 + d3 = 0.89 :=
by sorry

end NUMINAMATH_GPT_Trisha_walked_total_distance_l2004_200410


namespace NUMINAMATH_GPT_root_interval_l2004_200481

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  -- Proof by the Intermediate Value Theorem
  sorry

end NUMINAMATH_GPT_root_interval_l2004_200481


namespace NUMINAMATH_GPT_day_of_week_proof_l2004_200487

def day_of_week_17th_2003 := "Wednesday"
def day_of_week_305th_2003 := "Thursday"

theorem day_of_week_proof (d17 : day_of_week_17th_2003 = "Wednesday") : day_of_week_305th_2003 = "Thursday" := 
sorry

end NUMINAMATH_GPT_day_of_week_proof_l2004_200487


namespace NUMINAMATH_GPT_factorize_expression_l2004_200496

theorem factorize_expression (a x y : ℤ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2004_200496


namespace NUMINAMATH_GPT_gcd_32_48_l2004_200466

/--
The greatest common factor of 32 and 48 is 16.
-/
theorem gcd_32_48 : Int.gcd 32 48 = 16 :=
by
  sorry

end NUMINAMATH_GPT_gcd_32_48_l2004_200466


namespace NUMINAMATH_GPT_initial_bottles_l2004_200472

-- Define the conditions
def drank_bottles : ℕ := 144
def left_bottles : ℕ := 157

-- Define the total_bottles function
def total_bottles : ℕ := drank_bottles + left_bottles

-- State the theorem to be proven
theorem initial_bottles : total_bottles = 301 :=
by
  sorry

end NUMINAMATH_GPT_initial_bottles_l2004_200472


namespace NUMINAMATH_GPT_double_angle_second_quadrant_l2004_200449

theorem double_angle_second_quadrant (α : ℝ) (h : π/2 < α ∧ α < π) : 
  ¬((0 ≤ 2*α ∧ 2*α < π/2) ∨ (3*π/2 < 2*α ∧ 2*α < 2*π)) :=
sorry

end NUMINAMATH_GPT_double_angle_second_quadrant_l2004_200449


namespace NUMINAMATH_GPT_statement1_statement2_l2004_200421

def is_pow_of_two (a : ℕ) : Prop := ∃ n : ℕ, a = 2^(n + 1)
def in_A (a : ℕ) : Prop := is_pow_of_two a
def not_in_A (a : ℕ) : Prop := ¬ in_A a ∧ a ≠ 1

theorem statement1 : 
  ∀ (a : ℕ), in_A a → ∀ (b : ℕ), b < 2 * a - 1 → ¬ (2 * a ∣ b * (b + 1)) := 
by {
  sorry
}

theorem statement2 :
  ∀ (a : ℕ), not_in_A a → ∃ (b : ℕ), b < 2 * a - 1 ∧ (2 * a ∣ b * (b + 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_statement1_statement2_l2004_200421


namespace NUMINAMATH_GPT_intersection_point_l2004_200469

theorem intersection_point (x y : ℚ) 
  (h1 : 3 * y = -2 * x + 6) 
  (h2 : 2 * y = 7 * x - 4) :
  x = 24 / 25 ∧ y = 34 / 25 :=
sorry

end NUMINAMATH_GPT_intersection_point_l2004_200469


namespace NUMINAMATH_GPT_hotdogs_needed_l2004_200482

theorem hotdogs_needed 
  (ella_hotdogs : ℕ) (emma_hotdogs : ℕ)
  (luke_multiple : ℕ) (hunter_multiple : ℚ)
  (h_ella : ella_hotdogs = 2)
  (h_emma : emma_hotdogs = 2)
  (h_luke : luke_multiple = 2)
  (h_hunter : hunter_multiple = (3/2)) :
  ella_hotdogs + emma_hotdogs + luke_multiple * (ella_hotdogs + emma_hotdogs) + hunter_multiple * (ella_hotdogs + emma_hotdogs) = 18 := by
    sorry

end NUMINAMATH_GPT_hotdogs_needed_l2004_200482


namespace NUMINAMATH_GPT_future_tech_high_absentee_percentage_l2004_200492

theorem future_tech_high_absentee_percentage :
  let total_students := 180
  let boys := 100
  let girls := 80
  let absent_boys_fraction := 1 / 5
  let absent_girls_fraction := 1 / 4
  let absent_boys := absent_boys_fraction * boys
  let absent_girls := absent_girls_fraction * girls
  let total_absent_students := absent_boys + absent_girls
  let absent_percentage := (total_absent_students / total_students) * 100
  (absent_percentage = 22.22) := 
by
  sorry

end NUMINAMATH_GPT_future_tech_high_absentee_percentage_l2004_200492


namespace NUMINAMATH_GPT_inequality_solution_range_of_a_l2004_200458

noncomputable def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x| 

theorem inequality_solution (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := 
by sorry

theorem range_of_a (a x : ℝ) (h : a^2 + 2 * a + |1 + x| < f x) : -3 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_range_of_a_l2004_200458


namespace NUMINAMATH_GPT_real_roots_of_quadratic_l2004_200437

theorem real_roots_of_quadratic (k : ℝ) : (k ≤ 0 ∨ 1 ≤ k) →
  ∃ x : ℝ, x^2 + 2 * k * x + k = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_real_roots_of_quadratic_l2004_200437


namespace NUMINAMATH_GPT_single_dog_barks_per_minute_l2004_200455

theorem single_dog_barks_per_minute (x : ℕ) (h : 10 * 2 * x = 600) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_single_dog_barks_per_minute_l2004_200455


namespace NUMINAMATH_GPT_find_n_l2004_200448

noncomputable def parabola_focus : ℝ × ℝ :=
  (2, 0)

noncomputable def hyperbola_focus (n : ℝ) : ℝ × ℝ :=
  (Real.sqrt (3 + n), 0)

theorem find_n (n : ℝ) : hyperbola_focus n = parabola_focus → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2004_200448


namespace NUMINAMATH_GPT_percent_students_prefer_golf_l2004_200461

theorem percent_students_prefer_golf (students_north : ℕ) (students_south : ℕ)
  (percent_golf_north : ℚ) (percent_golf_south : ℚ) :
  students_north = 1800 →
  students_south = 2200 →
  percent_golf_north = 15 →
  percent_golf_south = 25 →
  (820 / 4000 : ℚ) = 20.5 :=
by
  intros h_north h_south h_percent_north h_percent_south
  sorry

end NUMINAMATH_GPT_percent_students_prefer_golf_l2004_200461


namespace NUMINAMATH_GPT_parabola_translation_correct_l2004_200463

variable (x : ℝ)

def original_parabola : ℝ := 5 * x^2

def translated_parabola : ℝ := 5 * (x - 2)^2 + 3

theorem parabola_translation_correct :
  translated_parabola x = 5 * (x - 2)^2 + 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_translation_correct_l2004_200463


namespace NUMINAMATH_GPT_find_m_l2004_200483

theorem find_m
  (m : ℝ)
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (m, 2, 3))
  (hB : B = (1, -1, 1))
  (h_dist : (Real.sqrt ((m - 1) ^ 2 + (2 - (-1)) ^ 2 + (3 - 1) ^ 2) = Real.sqrt 13)) :
  m = 1 := 
sorry

end NUMINAMATH_GPT_find_m_l2004_200483


namespace NUMINAMATH_GPT_seventh_term_geometric_sequence_l2004_200445

theorem seventh_term_geometric_sequence (a : ℝ) (a3 : ℝ) (r : ℝ) (n : ℕ) (term : ℕ → ℝ)
    (h_a : a = 3)
    (h_a3 : a3 = 3 / 64)
    (h_term : ∀ n, term n = a * r ^ (n - 1))
    (h_r : r = 1 / 8) :
    term 7 = 3 / 262144 :=
by
  sorry

end NUMINAMATH_GPT_seventh_term_geometric_sequence_l2004_200445


namespace NUMINAMATH_GPT_initial_interest_rate_l2004_200470

theorem initial_interest_rate
    (P R : ℝ) 
    (h1 : P * R = 10120) 
    (h2 : P * (R + 6) = 12144) : 
    R = 30 :=
sorry

end NUMINAMATH_GPT_initial_interest_rate_l2004_200470


namespace NUMINAMATH_GPT_atleast_one_genuine_l2004_200435

noncomputable def products : ℕ := 12
noncomputable def genuine : ℕ := 10
noncomputable def defective : ℕ := 2
noncomputable def selected : ℕ := 3

theorem atleast_one_genuine :
  (selected = 3) →
  (genuine + defective = 12) →
  (genuine ≥ 3) →
  (selected ≥ 1) →
  ∃ g d : ℕ, g + d = 3 ∧ g > 0 ∧ d ≤ 2 :=
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_atleast_one_genuine_l2004_200435


namespace NUMINAMATH_GPT_B_can_complete_work_in_6_days_l2004_200419

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_B_can_complete_work_in_6_days_l2004_200419


namespace NUMINAMATH_GPT_triangle_perimeter_inequality_l2004_200434

theorem triangle_perimeter_inequality (x : ℕ) (h₁ : 15 + 24 > x) (h₂ : 15 + x > 24) (h₃ : 24 + x > 15) 
    (h₄ : ∃ x : ℕ, x > 9 ∧ x < 39) : 15 + 24 + x = 49 :=
by { sorry }

end NUMINAMATH_GPT_triangle_perimeter_inequality_l2004_200434


namespace NUMINAMATH_GPT_first_player_winning_strategy_l2004_200409

-- Definitions based on conditions
def initial_position (m n : ℕ) : ℕ × ℕ := (m - 1, n - 1)

-- Main theorem statement
theorem first_player_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (initial_position m n).fst ≠ (initial_position m n).snd ↔ m ≠ n :=
by
  sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l2004_200409


namespace NUMINAMATH_GPT_tom_steps_l2004_200484

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end NUMINAMATH_GPT_tom_steps_l2004_200484


namespace NUMINAMATH_GPT_middle_term_is_35_l2004_200477

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d ∧ e - d = f - e

-- Given sequence values
def seq1 := 23
def seq6 := 47

-- Theorem stating that the middle term y in the sequence is 35
theorem middle_term_is_35 (x y z w : ℤ) :
  arithmetic_sequence seq1 x y z w seq6 → y = 35 :=
by
  sorry

end NUMINAMATH_GPT_middle_term_is_35_l2004_200477


namespace NUMINAMATH_GPT_ratio_Ford_to_Toyota_l2004_200447

-- Definitions based on the conditions
variables (Ford Dodge Toyota VW : ℕ)

axiom h1 : Ford = (1/3 : ℚ) * Dodge
axiom h2 : VW = (1/2 : ℚ) * Toyota
axiom h3 : VW = 5
axiom h4 : Dodge = 60

-- Theorem statement to be proven
theorem ratio_Ford_to_Toyota : Ford / Toyota = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_Ford_to_Toyota_l2004_200447


namespace NUMINAMATH_GPT_largest_value_among_expressions_l2004_200432

theorem largest_value_among_expressions 
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1) (h2 : a1 < a2) (h3 : a2 < 1)
  (h4 : 0 < b1) (h5 : b1 < b2) (h6 : b2 < 1)
  (ha : a1 + a2 = 1) (hb : b1 + b2 = 1) :
  a1 * b1 + a2 * b2 > a1 * a2 + b1 * b2 ∧ 
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end NUMINAMATH_GPT_largest_value_among_expressions_l2004_200432


namespace NUMINAMATH_GPT_rectangle_original_area_l2004_200438

theorem rectangle_original_area (L L' A : ℝ) 
  (h1: A = L * 10)
  (h2: L' * 10 = (4 / 3) * A)
  (h3: 2 * L' + 2 * 10 = 60) : A = 150 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_original_area_l2004_200438


namespace NUMINAMATH_GPT_total_chocolate_bars_in_large_box_l2004_200405

def large_box_contains_18_small_boxes : ℕ := 18
def small_box_contains_28_chocolate_bars : ℕ := 28

theorem total_chocolate_bars_in_large_box :
  (large_box_contains_18_small_boxes * small_box_contains_28_chocolate_bars) = 504 := 
by
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_in_large_box_l2004_200405


namespace NUMINAMATH_GPT_max_a_satisfies_no_lattice_points_l2004_200464

-- Define the conditions
def no_lattice_points (m : ℚ) (x_upper : ℕ) :=
  ∀ x : ℕ, 0 < x ∧ x ≤ x_upper → ¬∃ y : ℤ, y = m * x + 3

-- Final statement we need to prove
theorem max_a_satisfies_no_lattice_points :
  ∃ a : ℚ, a = 51 / 151 ∧ ∀ m : ℚ, 1 / 3 < m → m < a → no_lattice_points m 150 :=
sorry

end NUMINAMATH_GPT_max_a_satisfies_no_lattice_points_l2004_200464


namespace NUMINAMATH_GPT_adela_numbers_l2004_200498

theorem adela_numbers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = a^2 - b^2 - 4038) :
  (a = 2020 ∧ b = 1) ∨ (a = 2020 ∧ b = 2019) ∨ (a = 676 ∧ b = 3) ∨ (a = 676 ∧ b = 673) :=
sorry

end NUMINAMATH_GPT_adela_numbers_l2004_200498


namespace NUMINAMATH_GPT_frog_arrangement_l2004_200401

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_frog_arrangement_l2004_200401


namespace NUMINAMATH_GPT_sequence_product_l2004_200443

theorem sequence_product {n : ℕ} (h : 1 < n) (a : ℕ → ℕ) (h₀ : ∀ n, a n = 2^n) : 
  a (n-1) * a (n+1) = 4^n :=
by sorry

end NUMINAMATH_GPT_sequence_product_l2004_200443


namespace NUMINAMATH_GPT_argument_friends_count_l2004_200403

-- Define the conditions
def original_friends: ℕ := 20
def current_friends: ℕ := 19
def new_friend: ℕ := 1

-- Define the statement that needs to be proved
theorem argument_friends_count : 
  (original_friends + new_friend - current_friends = 1) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_argument_friends_count_l2004_200403


namespace NUMINAMATH_GPT_marbles_problem_l2004_200476

theorem marbles_problem (p : ℕ) (m n r : ℕ) 
(hp : Nat.Prime p) 
(h1 : p = 2017)
(h2 : N = p^m * n)
(h3 : ¬ p ∣ n)
(h4 : r = n % p) 
(h N : ∀ (N : ℕ), N = 3 * p * 632 - 1)
: p * m + r = 3913 := 
sorry

end NUMINAMATH_GPT_marbles_problem_l2004_200476


namespace NUMINAMATH_GPT_proposition_C_is_correct_l2004_200473

theorem proposition_C_is_correct :
  ∃ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4) :=
by
  sorry

end NUMINAMATH_GPT_proposition_C_is_correct_l2004_200473


namespace NUMINAMATH_GPT_equal_charges_at_x_l2004_200439

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end NUMINAMATH_GPT_equal_charges_at_x_l2004_200439


namespace NUMINAMATH_GPT_basil_plants_yielded_l2004_200425

def initial_investment (seed_cost soil_cost : ℕ) : ℕ :=
  seed_cost + soil_cost

def total_revenue (net_profit initial_investment : ℕ) : ℕ :=
  net_profit + initial_investment

def basil_plants (total_revenue price_per_plant : ℕ) : ℕ :=
  total_revenue / price_per_plant

theorem basil_plants_yielded
  (seed_cost soil_cost net_profit price_per_plant expected_plants : ℕ)
  (h_seed_cost : seed_cost = 2)
  (h_soil_cost : soil_cost = 8)
  (h_net_profit : net_profit = 90)
  (h_price_per_plant : price_per_plant = 5)
  (h_expected_plants : expected_plants = 20) :
  basil_plants (total_revenue net_profit (initial_investment seed_cost soil_cost)) price_per_plant = expected_plants :=
by
  -- Proof steps will be here
  sorry

end NUMINAMATH_GPT_basil_plants_yielded_l2004_200425


namespace NUMINAMATH_GPT_find_g_values_l2004_200414

theorem find_g_values
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x * y) = x * g y)
  (h2 : g 1 = 30) :
  g 50 = 1500 ∧ g 0.5 = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_g_values_l2004_200414


namespace NUMINAMATH_GPT_line_forms_equivalence_l2004_200428

noncomputable def points (P Q : ℝ × ℝ) : Prop := 
  ∃ m c, ∃ b d, P = (b, m * b + c) ∧ Q = (d, m * d + c)

theorem line_forms_equivalence :
  points (-2, 3) (4, -1) →
  (∀ x y : ℝ, (y + 1) / (3 + 1) = (x - 4) / (-2 - 4)) ∧
  (∀ x y : ℝ, y + 1 = - (2 / 3) * (x - 4)) ∧
  (∀ x y : ℝ, y = - (2 / 3) * x + 5 / 3) ∧
  (∀ x y : ℝ, x / (5 / 2) + y / (5 / 3) = 1) :=
  sorry

end NUMINAMATH_GPT_line_forms_equivalence_l2004_200428


namespace NUMINAMATH_GPT_projectile_height_l2004_200457

theorem projectile_height (t : ℝ) (h : (-16 * t^2 + 80 * t = 100)) : t = 2.5 :=
sorry

end NUMINAMATH_GPT_projectile_height_l2004_200457


namespace NUMINAMATH_GPT_chickens_bought_l2004_200480

theorem chickens_bought (total_spent : ℤ) (egg_count : ℤ) (egg_price : ℤ) (chicken_price : ℤ) (egg_cost : ℤ := egg_count * egg_price) (chicken_spent : ℤ := total_spent - egg_cost) : total_spent = 88 → egg_count = 20 → egg_price = 2 → chicken_price = 8 → chicken_spent / chicken_price = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chickens_bought_l2004_200480


namespace NUMINAMATH_GPT_max_sin_sin2x_l2004_200488

open Real

theorem max_sin_sin2x (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  ∃ x : ℝ, (0 < x ∧ x < π / 2) ∧ (sin x * sin (2 * x) = 4 * sqrt 3 / 9) := 
sorry

end NUMINAMATH_GPT_max_sin_sin2x_l2004_200488


namespace NUMINAMATH_GPT_last_two_digits_of_quotient_l2004_200478

noncomputable def greatest_integer_not_exceeding (x : ℝ) : ℤ := ⌊x⌋

theorem last_two_digits_of_quotient :
  let a : ℤ := 10 ^ 93
  let b : ℤ := 10 ^ 31 + 3
  let x : ℤ := greatest_integer_not_exceeding (a / b : ℝ)
  (x % 100) = 8 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_quotient_l2004_200478


namespace NUMINAMATH_GPT_residue_of_neg_2035_mod_47_l2004_200436

theorem residue_of_neg_2035_mod_47 : (-2035 : ℤ) % 47 = 33 := 
by
  sorry

end NUMINAMATH_GPT_residue_of_neg_2035_mod_47_l2004_200436


namespace NUMINAMATH_GPT_fraction_simplest_form_l2004_200471

def fracA (a b : ℤ) : ℤ × ℤ := (|2 * a|, 5 * a^2 * b)
def fracB (a : ℤ) : ℤ × ℤ := (a, a^2 - 2 * a)
def fracC (a b : ℤ) : ℤ × ℤ := (3 * a + b, a + b)
def fracD (a b : ℤ) : ℤ × ℤ := (a^2 - a * b, a^2 - b^2)

theorem fraction_simplest_form (a b : ℤ) : (fracC a b).1 / (fracC a b).2 = (3 * a + b) / (a + b) :=
by sorry

end NUMINAMATH_GPT_fraction_simplest_form_l2004_200471


namespace NUMINAMATH_GPT_exists_pos_int_such_sqrt_not_int_l2004_200474

theorem exists_pos_int_such_sqrt_not_int (a b c : ℤ) : ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, k * k = n^3 + a * n^2 + b * n + c :=
by
  sorry

end NUMINAMATH_GPT_exists_pos_int_such_sqrt_not_int_l2004_200474


namespace NUMINAMATH_GPT_max_value_of_f_l2004_200442

noncomputable def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

theorem max_value_of_f : ∃ x : ℝ, f x = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2004_200442


namespace NUMINAMATH_GPT_division_exponentiation_addition_l2004_200452

theorem division_exponentiation_addition :
  6 / -3 + 2^2 * (1 - 4) = -14 := by
sorry

end NUMINAMATH_GPT_division_exponentiation_addition_l2004_200452


namespace NUMINAMATH_GPT_find_m_l2004_200479

theorem find_m {x : ℝ} (m : ℝ) (h : ∀ x, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2 * x > m * x)) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l2004_200479


namespace NUMINAMATH_GPT_product_of_roots_of_t_squared_equals_49_l2004_200440

theorem product_of_roots_of_t_squared_equals_49 : 
  ∃ t : ℝ, (t^2 = 49) ∧ (t = 7 ∨ t = -7) ∧ (t * (7 + -7)) = -49 := 
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_t_squared_equals_49_l2004_200440


namespace NUMINAMATH_GPT_brick_length_is_50_l2004_200451

theorem brick_length_is_50
  (x : ℝ)
  (brick_volume_eq : x * 11.25 * 6 * 3200 = 800 * 600 * 22.5) :
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_brick_length_is_50_l2004_200451


namespace NUMINAMATH_GPT_initial_average_score_l2004_200491

theorem initial_average_score (A : ℝ) :
  (∃ (A : ℝ), (16 * A = 15 * 64 + 24)) → A = 61.5 := 
by 
  sorry 

end NUMINAMATH_GPT_initial_average_score_l2004_200491


namespace NUMINAMATH_GPT_shaded_region_area_correct_l2004_200467

noncomputable def area_shaded_region : ℝ := 
  let side_length := 2
  let radius := 1
  let area_square := side_length^2
  let area_circle := Real.pi * radius^2
  area_square - area_circle

theorem shaded_region_area_correct : area_shaded_region = 4 - Real.pi :=
  by
    sorry

end NUMINAMATH_GPT_shaded_region_area_correct_l2004_200467


namespace NUMINAMATH_GPT_percentage_waiting_for_parts_l2004_200454

def totalComputers : ℕ := 20
def unfixableComputers : ℕ := (20 * 20) / 100
def fixedRightAway : ℕ := 8
def waitingForParts : ℕ := totalComputers - (unfixableComputers + fixedRightAway)

theorem percentage_waiting_for_parts : (waitingForParts : ℝ) / totalComputers * 100 = 40 := 
by 
  have : waitingForParts = 8 := sorry
  have : (8 / 20 : ℝ) * 100 = 40 := sorry
  exact sorry

end NUMINAMATH_GPT_percentage_waiting_for_parts_l2004_200454


namespace NUMINAMATH_GPT_blood_flow_scientific_notation_l2004_200415

theorem blood_flow_scientific_notation (blood_flow : ℝ) (h : blood_flow = 4900) : 
  4900 = 4.9 * (10 ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_blood_flow_scientific_notation_l2004_200415


namespace NUMINAMATH_GPT_angles_congruence_mod_360_l2004_200431

theorem angles_congruence_mod_360 (a b c d : ℤ) : 
  (a = 30) → (b = -30) → (c = 630) → (d = -630) →
  (b % 360 = 330 % 360) ∧ 
  (a % 360 ≠ 330 % 360) ∧ (c % 360 ≠ 330 % 360) ∧ (d % 360 ≠ 330 % 360) :=
by
  intros
  sorry

end NUMINAMATH_GPT_angles_congruence_mod_360_l2004_200431


namespace NUMINAMATH_GPT_price_of_one_shirt_l2004_200433

variable (P : ℝ)

-- Conditions
def cost_two_shirts := 1.5 * P
def cost_three_shirts := 1.9 * P 
def full_price_three_shirts := 3 * P
def savings := full_price_three_shirts - cost_three_shirts

-- Correct answer
theorem price_of_one_shirt (hs : savings = 11) : P = 10 :=
by
  sorry

end NUMINAMATH_GPT_price_of_one_shirt_l2004_200433


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l2004_200499

theorem geometric_series_common_ratio (a₁ q : ℝ) (S₃ : ℝ)
  (h1 : S₃ = 7 * a₁)
  (h2 : S₃ = a₁ + a₁ * q + a₁ * q^2) :
  q = 2 ∨ q = -3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l2004_200499


namespace NUMINAMATH_GPT_problem_statement_l2004_200497

noncomputable def a : ℚ := 18 / 11
noncomputable def c : ℚ := -30 / 11

theorem problem_statement (a b c : ℚ) (h1 : b / a = 4)
    (h2 : b = 18 - 7 * a) (h3 : c = 2 * a - 6):
    a = 18 / 11 ∧ c = -30 / 11 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2004_200497


namespace NUMINAMATH_GPT_find_15th_term_l2004_200402

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end NUMINAMATH_GPT_find_15th_term_l2004_200402


namespace NUMINAMATH_GPT_triangle_XDE_area_l2004_200446

theorem triangle_XDE_area 
  (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 12) (hXZ : XZ = 14)
  (D E : ℝ → ℝ) (XD XE : ℝ) (hXD : XD = 3) (hXE : XE = 9) :
  ∃ (A : ℝ), A = 1/2 * XD * XE * (15 * Real.sqrt 17 / 56) ∧ A = 405 * Real.sqrt 17 / 112 :=
  sorry

end NUMINAMATH_GPT_triangle_XDE_area_l2004_200446


namespace NUMINAMATH_GPT_Danica_additional_cars_l2004_200412

theorem Danica_additional_cars (num_cars : ℕ) (cars_per_row : ℕ) (current_cars : ℕ) 
  (h_cars_per_row : cars_per_row = 8) (h_current_cars : current_cars = 35) :
  ∃ n, num_cars = 5 ∧ n = 40 ∧ n - current_cars = num_cars := 
by
  sorry

end NUMINAMATH_GPT_Danica_additional_cars_l2004_200412


namespace NUMINAMATH_GPT_cone_height_l2004_200424

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

end NUMINAMATH_GPT_cone_height_l2004_200424


namespace NUMINAMATH_GPT_find_z_l2004_200422

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_find_z_l2004_200422


namespace NUMINAMATH_GPT_g_at_3_l2004_200408

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 : (∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2 + 1) → 
  g 3 = 130 / 21 := 
by 
  sorry

end NUMINAMATH_GPT_g_at_3_l2004_200408


namespace NUMINAMATH_GPT_max_value_m_l2004_200490

/-- Proof that the inequality (a^2 + 4(b^2 + c^2))(b^2 + 4(a^2 + c^2))(c^2 + 4(a^2 + b^2)) 
    is greater than or equal to 729 for all a, b, c ∈ ℝ \ {0} with 
    |1/a| + |1/b| + |1/c| ≤ 3. -/
theorem max_value_m (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h_cond : |1 / a| + |1 / b| + |1 / c| ≤ 3) :
  (a^2 + 4 * (b^2 + c^2)) * (b^2 + 4 * (a^2 + c^2)) * (c^2 + 4 * (a^2 + b^2)) ≥ 729 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_m_l2004_200490


namespace NUMINAMATH_GPT_geometric_seq_arithmetic_condition_l2004_200489

open Real

noncomputable def common_ratio (q : ℝ) := (q > 0) ∧ (q^2 - q - 1 = 0)

def arithmetic_seq_condition (a1 a2 a3 : ℝ) := (a2 = (a1 + a3) / 2)

theorem geometric_seq_arithmetic_condition (a1 a2 a3 a4 a5 : ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : q^2 - q - 1 = 0)
  (h3 : a2 = q * a1)
  (h4 : a3 = q * a2)
  (h5 : a4 = q * a3)
  (h6 : a5 = q * a4)
  (h7 : arithmetic_seq_condition a1 a2 a3) :
  (a4 + a5) / (a3 + a4) = (1 + sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_geometric_seq_arithmetic_condition_l2004_200489
