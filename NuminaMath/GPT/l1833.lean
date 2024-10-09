import Mathlib

namespace inequality_problem_l1833_183389

-- Define the problem conditions and goal
theorem inequality_problem (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) : 
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
sorry

end inequality_problem_l1833_183389


namespace mod_calculation_l1833_183398

theorem mod_calculation :
  (3 * 43 + 6 * 37) % 60 = 51 :=
by
  sorry

end mod_calculation_l1833_183398


namespace greatest_multiple_of_5_and_6_less_than_1000_l1833_183306

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l1833_183306


namespace circle_line_distance_l1833_183301

theorem circle_line_distance (c : ℝ) : 
  (∃ (P₁ P₂ P₃ : ℝ × ℝ), 
     (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) ∧
     ((P₁.1 - 2)^2 + (P₁.2 - 2)^2 = 18) ∧
     ((P₂.1 - 2)^2 + (P₂.2 - 2)^2 = 18) ∧
     ((P₃.1 - 2)^2 + (P₃.2 - 2)^2 = 18) ∧
     (abs (P₁.1 - P₁.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₂.1 - P₂.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₃.1 - P₃.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2)) ↔ 
  -2 ≤ c ∧ c ≤ 2 :=
sorry

end circle_line_distance_l1833_183301


namespace no_two_items_share_color_l1833_183321

theorem no_two_items_share_color (shirts pants hats : Fin 5) :
  ∃ num_outfits : ℕ, num_outfits = 60 :=
by
  sorry

end no_two_items_share_color_l1833_183321


namespace solve_equation_in_natural_numbers_l1833_183385

theorem solve_equation_in_natural_numbers (x y : ℕ) :
  2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := 
sorry

end solve_equation_in_natural_numbers_l1833_183385


namespace students_neither_music_nor_art_l1833_183328

theorem students_neither_music_nor_art
  (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_both : ℕ)
  (h_total : total_students = 500)
  (h_music : students_music = 30)
  (h_art : students_art = 10)
  (h_both : students_both = 10)
  : total_students - (students_music + students_art - students_both) = 460 :=
by
  rw [h_total, h_music, h_art, h_both]
  norm_num
  sorry

end students_neither_music_nor_art_l1833_183328


namespace alice_bracelets_given_away_l1833_183344

theorem alice_bracelets_given_away
    (total_bracelets : ℕ)
    (cost_of_materials : ℝ)
    (price_per_bracelet : ℝ)
    (profit : ℝ)
    (bracelets_given_away : ℕ)
    (bracelets_sold : ℕ)
    (total_revenue : ℝ)
    (h1 : total_bracelets = 52)
    (h2 : cost_of_materials = 3)
    (h3 : price_per_bracelet = 0.25)
    (h4 : profit = 8)
    (h5 : total_revenue = profit + cost_of_materials)
    (h6 : total_revenue = price_per_bracelet * bracelets_sold)
    (h7 : total_bracelets = bracelets_sold + bracelets_given_away) :
    bracelets_given_away = 8 :=
by
  sorry

end alice_bracelets_given_away_l1833_183344


namespace randy_biscuits_l1833_183354

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l1833_183354


namespace linear_regression_equation_l1833_183304

theorem linear_regression_equation (x y : ℝ) (h : {(1, 2), (2, 3), (3, 4), (4, 5)} ⊆ {(x, y) | y = x + 1}) : 
  (∀ x y, (x = 1 → y = 2) ∧ (x = 2 → y = 3) ∧ (x = 3 → y = 4) ∧ (x = 4 → y = 5)) ↔ (y = x + 1) :=
by
  sorry

end linear_regression_equation_l1833_183304


namespace x_can_be_any_sign_l1833_183396

theorem x_can_be_any_sign
  (x y p q : ℝ)
  (h1 : abs (x / y) < abs (p) / q^2)
  (h2 : y ≠ 0) (h3 : q ≠ 0) :
  ∃ (x' : ℝ), True :=
by
  sorry

end x_can_be_any_sign_l1833_183396


namespace original_cost_l1833_183387

theorem original_cost (C : ℝ) (h : 550 = 1.35 * C) : C = 550 / 1.35 :=
by
  sorry

end original_cost_l1833_183387


namespace bucket_full_weight_l1833_183377

theorem bucket_full_weight (x y c d : ℝ)
  (h1 : x + 3 / 4 * y = c)
  (h2 : x + 1 / 3 * y = d) :
  x + y = (8 / 5) * c - (7 / 5) * d :=
by
  sorry

end bucket_full_weight_l1833_183377


namespace cos_neg_3pi_plus_alpha_l1833_183315

/-- Given conditions: 
  1. 𝚌𝚘𝚜(3π/2 + α) = -3/5,
  2. α is an angle in the fourth quadrant,
Prove: cos(-3π + α) = -4/5 -/
theorem cos_neg_3pi_plus_alpha (α : Real) (h1 : Real.cos (3 * Real.pi / 2 + α) = -3 / 5) (h2 : 0 ≤ α ∧ α < 2 * Real.pi ∧ Real.sin α < 0) :
  Real.cos (-3 * Real.pi + α) = -4 / 5 := 
sorry

end cos_neg_3pi_plus_alpha_l1833_183315


namespace complex_fraction_sum_real_parts_l1833_183337

theorem complex_fraction_sum_real_parts (a b : ℝ) (h : (⟨0, 1⟩ / ⟨1, 1⟩ : ℂ) = a + b * ⟨0, 1⟩) : a + b = 1 := by
  sorry

end complex_fraction_sum_real_parts_l1833_183337


namespace triangles_hyperbola_parallel_l1833_183394

variable (a b c a1 b1 c1 : ℝ)

-- Defining the property that all vertices lie on the hyperbola y = 1/x
def on_hyperbola (x : ℝ) (y : ℝ) : Prop := y = 1 / x

-- Defining the parallelism condition for line segments
def parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem triangles_hyperbola_parallel
  (H1A : on_hyperbola a (1 / a))
  (H1B : on_hyperbola b (1 / b))
  (H1C : on_hyperbola c (1 / c))
  (H2A : on_hyperbola a1 (1 / a1))
  (H2B : on_hyperbola b1 (1 / b1))
  (H2C : on_hyperbola c1 (1 / c1))
  (H_AB_parallel_A1B1 : parallel ((b - a) / (a * b * (a - b))) ((b1 - a1) / (a1 * b1 * (a1 - b1))))
  (H_BC_parallel_B1C1 : parallel ((c - b) / (b * c * (b - c))) ((c1 - b1) / (b1 * c1 * (b1 - c1)))) :
  parallel ((c1 - a) / (a * c1 * (a - c1))) ((c - a1) / (a1 * c * (a1 - c))) :=
sorry

end triangles_hyperbola_parallel_l1833_183394


namespace horse_revolutions_l1833_183345

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions_l1833_183345


namespace log4_21_correct_l1833_183359

noncomputable def log4_21 (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2)
                                     (h2 : Real.log 2 = b * Real.log 7) : ℝ :=
  (a * b + 1) / (2 * b)

theorem log4_21_correct (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2) 
                        (h2 : Real.log 2 = b * Real.log 7) : 
  log4_21 a b h1 h2 = (a * b + 1) / (2 * b) := 
sorry

end log4_21_correct_l1833_183359


namespace octal_to_base5_conversion_l1833_183338

-- Define the octal to decimal conversion
def octalToDecimal (n : ℕ) : ℕ :=
  2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 1 * 8^0

-- Define the base-5 number
def base5Representation : ℕ := 13113

-- Theorem statement
theorem octal_to_base5_conversion :
  octalToDecimal 2011 = base5Representation := 
sorry

end octal_to_base5_conversion_l1833_183338


namespace determine_m_l1833_183324

theorem determine_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) → m = 2 :=
sorry

end determine_m_l1833_183324


namespace fourth_sphere_radius_l1833_183339

theorem fourth_sphere_radius (R r : ℝ) (h1 : R > 0)
  (h2 : ∀ (a b c d : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    dist a b = 2*R ∧ dist b c = 2*R ∧ dist c d = 2*R ∧ dist d a = R + r ∧
    dist a c = R + r ∧ dist b d = R + r) :
  r = 4*R/3 :=
  sorry

end fourth_sphere_radius_l1833_183339


namespace total_kids_in_lawrence_county_l1833_183326

def kids_stayed_home : ℕ := 644997
def kids_went_to_camp : ℕ := 893835
def kids_from_outside : ℕ := 78

theorem total_kids_in_lawrence_county : kids_stayed_home + kids_went_to_camp = 1538832 := by
  sorry

end total_kids_in_lawrence_county_l1833_183326


namespace mika_stickers_l1833_183386

def s1 : ℝ := 20.5
def s2 : ℝ := 26.3
def s3 : ℝ := 19.75
def s4 : ℝ := 6.25
def s5 : ℝ := 57.65
def s6 : ℝ := 15.8

theorem mika_stickers 
  (M : ℝ)
  (hM : M = s1 + s2 + s3 + s4 + s5 + s6) 
  : M = 146.25 :=
sorry

end mika_stickers_l1833_183386


namespace triangle_ratio_l1833_183314

theorem triangle_ratio (a b c : ℝ) (P Q : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c)
  (h₄ : P > 0) (h₅ : Q > P) (h₆ : Q < c) (h₇ : P = 21) (h₈ : Q - P = 35) (h₉ : c - Q = 100)
  (h₁₀ : P + (Q - P) + (c - Q) = c)
  (angle_trisect : ∃ x y : ℝ, x ≠ y ∧ x = a / b ∧ y = 7 / 45) :
  ∃ p q r : ℕ, p + q + r = 92 ∧ p.gcd r = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ q := sorry

end triangle_ratio_l1833_183314


namespace eval_expression_l1833_183308

theorem eval_expression : (49^2 - 25^2 + 10^2) = 1876 := by
  sorry

end eval_expression_l1833_183308


namespace new_marketing_percentage_l1833_183364

theorem new_marketing_percentage 
  (total_students : ℕ)
  (initial_finance_percentage : ℕ)
  (initial_marketing_percentage : ℕ)
  (initial_operations_management_percentage : ℕ)
  (new_finance_percentage : ℕ)
  (operations_management_percentage : ℕ)
  (total_percentage : ℕ) :
  total_students = 5000 →
  initial_finance_percentage = 85 →
  initial_marketing_percentage = 80 →
  initial_operations_management_percentage = 10 →
  new_finance_percentage = 92 →
  operations_management_percentage = 10 →
  total_percentage = 175 →
  initial_marketing_percentage - (new_finance_percentage - initial_finance_percentage) = 73 :=
by
  sorry

end new_marketing_percentage_l1833_183364


namespace find_y_l1833_183378

theorem find_y (y : ℕ) (h1 : y % 6 = 5) (h2 : y % 7 = 6) (h3 : y % 8 = 7) : y = 167 := 
by
  sorry  -- Proof is omitted

end find_y_l1833_183378


namespace max_value_of_symmetric_function_l1833_183374

noncomputable def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) → ∃ x : ℝ, ∀ y : ℝ, f x a b ≥ f y a b ∧ f x a b = 16 :=
sorry

end max_value_of_symmetric_function_l1833_183374


namespace tom_age_l1833_183395

theorem tom_age (c : ℕ) (h1 : 2 * c - 1 = tom) (h2 : c + 3 = dave) (h3 : c + (2 * c - 1) + (c + 3) = 30) : tom = 13 :=
  sorry

end tom_age_l1833_183395


namespace unique_solution_real_l1833_183352

theorem unique_solution_real {x y : ℝ} (h1 : x * (x + y)^2 = 9) (h2 : x * (y^3 - x^3) = 7) :
  x = 1 ∧ y = 2 :=
sorry

end unique_solution_real_l1833_183352


namespace complement_union_eq_l1833_183399

open Set

variable (U A B : Set ℤ)

noncomputable def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3}

noncomputable def setA : Set ℤ := {-1, 0, 3}

noncomputable def setB : Set ℤ := {1, 3}

theorem complement_union_eq :
  A ∪ B = {-1, 0, 1, 3} →
  U = universal_set →
  A = setA →
  B = setB →
  (U \ (A ∪ B)) = {-2, 2} := by
  intros
  sorry

end complement_union_eq_l1833_183399


namespace fishing_problem_l1833_183347

theorem fishing_problem :
  ∃ F : ℕ, (F % 3 = 1 ∧
            ((F - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3) % 3 = 1) ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3 - 1) = 0) :=
sorry

end fishing_problem_l1833_183347


namespace tan_proof_l1833_183335

noncomputable def prove_tan_relation (α β : ℝ) : Prop :=
  2 * (Real.tan α) = 3 * (Real.tan β)

theorem tan_proof (α β : ℝ) (h : Real.tan (α - β) = (Real.sin (2*β)) / (5 - Real.cos (2*β))) : 
  prove_tan_relation α β :=
sorry

end tan_proof_l1833_183335


namespace initial_liquid_X_percentage_is_30_l1833_183367

variable (initial_liquid_X_percentage : ℝ)

theorem initial_liquid_X_percentage_is_30
  (solution_total_weight : ℝ := 8)
  (initial_water_percentage : ℝ := 70)
  (evaporated_water_weight : ℝ := 3)
  (added_solution_weight : ℝ := 3)
  (new_liquid_X_percentage : ℝ := 41.25)
  (total_new_solution_weight : ℝ := 8)
  :
  initial_liquid_X_percentage = 30 :=
sorry

end initial_liquid_X_percentage_is_30_l1833_183367


namespace solve_equation_l1833_183392

theorem solve_equation (x : ℝ) (h : x ≠ 4) :
  (x - 3) / (4 - x) - 1 = 1 / (x - 4) → x = 3 :=
by
  sorry

end solve_equation_l1833_183392


namespace part_one_part_two_l1833_183390

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end part_one_part_two_l1833_183390


namespace xn_plus_inv_xn_is_integer_l1833_183388

theorem xn_plus_inv_xn_is_integer (x : ℝ) (hx : x ≠ 0) (k : ℤ) (h : x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end xn_plus_inv_xn_is_integer_l1833_183388


namespace g_h_2_eq_583_l1833_183311

def g (x : ℝ) : ℝ := 3*x^2 - 5

def h (x : ℝ) : ℝ := -2*x^3 + 2

theorem g_h_2_eq_583 : g (h 2) = 583 :=
by
  sorry

end g_h_2_eq_583_l1833_183311


namespace cost_per_gallon_is_45_l1833_183316

variable (totalArea coverage cost_jason cost_jeremy dollars_per_gallon : ℕ)

-- Conditions
def total_area := 1600
def coverage_per_gallon := 400
def num_coats := 2
def contribution_jason := 180
def contribution_jeremy := 180

-- Gallons needed calculation
def gallons_per_coat := total_area / coverage_per_gallon
def total_gallons := gallons_per_coat * num_coats

-- Total cost calculation
def total_cost := contribution_jason + contribution_jeremy

-- Cost per gallon calculation
def cost_per_gallon := total_cost / total_gallons

-- Proof statement
theorem cost_per_gallon_is_45 : cost_per_gallon = 45 :=
by
  sorry

end cost_per_gallon_is_45_l1833_183316


namespace molecular_weight_ammonia_l1833_183362

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.008
def count_N : ℕ := 1
def count_H : ℕ := 3

theorem molecular_weight_ammonia :
  (count_N * atomic_weight_N) + (count_H * atomic_weight_H) = 17.034 :=
by
  sorry

end molecular_weight_ammonia_l1833_183362


namespace instantaneous_velocity_at_3_l1833_183383

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- State the main theorem that we need to prove
theorem instantaneous_velocity_at_3 : (deriv displacement 3 = 5) := by
  sorry

end instantaneous_velocity_at_3_l1833_183383


namespace tessa_initial_apples_l1833_183351

-- Define conditions as variables
variable (initial_apples anita_gave : ℕ)
variable (apples_needed_for_pie : ℕ := 10)
variable (apples_additional_now_needed : ℕ := 1)

-- Define the current amount of apples Tessa has
noncomputable def current_apples :=
  apples_needed_for_pie - apples_additional_now_needed

-- Define the initial apples Tessa had before Anita gave her 5 apples
noncomputable def initial_apples_calculated :=
  current_apples - anita_gave

-- Lean statement to prove the initial number of apples Tessa had
theorem tessa_initial_apples (h_initial_apples : anita_gave = 5) : initial_apples_calculated = 4 :=
by
  -- Here is where the proof would go; we use sorry to indicate it's not provided
  sorry

end tessa_initial_apples_l1833_183351


namespace find_x_eq_2_l1833_183357

theorem find_x_eq_2 (x : ℕ) (h : 7899665 - 36 * x = 7899593) : x = 2 := 
by
  sorry

end find_x_eq_2_l1833_183357


namespace no_combination_of_three_coins_sums_to_52_cents_l1833_183320

def is_valid_coin (c : ℕ) : Prop :=
  c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50 ∨ c = 100

theorem no_combination_of_three_coins_sums_to_52_cents :
  ¬ ∃ a b c : ℕ, is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ a + b + c = 52 :=
by 
  sorry

end no_combination_of_three_coins_sums_to_52_cents_l1833_183320


namespace compute_x_squared_y_plus_xy_squared_l1833_183391

theorem compute_x_squared_y_plus_xy_squared 
  (x y : ℝ)
  (h1 : (1 / x) + (1 / y) = 4)
  (h2 : x * y + x + y = 7) :
  x^2 * y + x * y^2 = 49 := 
  sorry

end compute_x_squared_y_plus_xy_squared_l1833_183391


namespace circle_radius_l1833_183332

theorem circle_radius :
  ∃ radius : ℝ, (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 16 → (x - 2)^2 + (y - 1)^2 = radius^2)
  ∧ radius = 4 :=
sorry

end circle_radius_l1833_183332


namespace goldfish_problem_l1833_183325

theorem goldfish_problem (x : ℕ) : 
  (18 + (x - 5) * 7 = 4) → (x = 3) :=
by
  intros
  sorry

end goldfish_problem_l1833_183325


namespace quadratic_root_condition_l1833_183358

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l1833_183358


namespace suitable_high_jump_athlete_l1833_183369

structure Athlete where
  average : ℕ
  variance : ℝ

def A : Athlete := ⟨169, 6.0⟩
def B : Athlete := ⟨168, 17.3⟩
def C : Athlete := ⟨169, 5.0⟩
def D : Athlete := ⟨168, 19.5⟩

def isSuitableCandidate (athlete: Athlete) (average_threshold: ℕ) : Prop :=
  athlete.average = average_threshold

theorem suitable_high_jump_athlete : isSuitableCandidate C 169 ∧
  (∀ a, isSuitableCandidate a 169 → a.variance ≥ C.variance) := by
  sorry

end suitable_high_jump_athlete_l1833_183369


namespace greatest_integer_difference_l1833_183302

theorem greatest_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∀ d : ℤ, (d = y - x) → d ≤ 6 := 
sorry

end greatest_integer_difference_l1833_183302


namespace percentage_k_equal_125_percent_j_l1833_183331

theorem percentage_k_equal_125_percent_j
  (j k l m : ℝ)
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := 
sorry

end percentage_k_equal_125_percent_j_l1833_183331


namespace red_light_max_probability_l1833_183330

theorem red_light_max_probability {m : ℕ} (h1 : m > 0) (h2 : m < 35) :
  m = 3 ∨ m = 15 ∨ m = 30 ∨ m = 40 → m = 30 :=
by
  sorry

end red_light_max_probability_l1833_183330


namespace sequence_formula_l1833_183382

theorem sequence_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 3 + 2 * a_n n) :
  ∀ n, a_n n = -3 * 2^(n - 1) :=
by
  sorry

end sequence_formula_l1833_183382


namespace intersection_M_complement_N_l1833_183381

open Set Real

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def N : Set ℝ := {x | (Real.log 2) ^ (1 - x) < 1}
def complement_N := {x : ℝ | x ≥ 1}

theorem intersection_M_complement_N :
  M ∩ complement_N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_complement_N_l1833_183381


namespace parameter_condition_l1833_183307

theorem parameter_condition (a : ℝ) :
  let D := 4 - 4 * a
  let diff_square := ((-2 / a) ^ 2 - 4 * (1 / a))
  D = 9 * diff_square -> a = -3 :=
by
  sorry -- Proof omitted

end parameter_condition_l1833_183307


namespace jerry_charge_per_hour_l1833_183300

-- Define the conditions from the problem
def time_painting : ℝ := 8
def time_fixing_counter : ℝ := 3 * time_painting
def time_mowing_lawn : ℝ := 6
def total_time_worked : ℝ := time_painting + time_fixing_counter + time_mowing_lawn
def total_payment : ℝ := 570

-- The proof statement
theorem jerry_charge_per_hour : 
  total_payment / total_time_worked = 15 :=
by
  sorry

end jerry_charge_per_hour_l1833_183300


namespace total_parents_in_auditorium_l1833_183397

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end total_parents_in_auditorium_l1833_183397


namespace range_of_alpha_minus_beta_l1833_183366

variable (α β : ℝ)

theorem range_of_alpha_minus_beta (h1 : -90 < α) (h2 : α < β) (h3 : β < 90) : -180 < α - β ∧ α - β < 0 := 
by
  sorry

end range_of_alpha_minus_beta_l1833_183366


namespace cuboid_volume_l1833_183327

theorem cuboid_volume (P h : ℝ) (P_eq : P = 32) (h_eq : h = 9) :
  ∃ (s : ℝ), 4 * s = P ∧ s * s * h = 576 :=
by
  sorry

end cuboid_volume_l1833_183327


namespace intersection_sums_l1833_183334

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y - 2)^2 - 6

theorem intersection_sums (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : y1 = parabola1 x1) (h2 : y2 = parabola1 x2)
  (h3 : y3 = parabola1 x3) (h4 : y4 = parabola1 x4)
  (k1 : x1 + 6 = y1^2 - 4*y1 + 4) (k2 : x2 + 6 = y2^2 - 4*y2 + 4)
  (k3 : x3 + 6 = y3^2 - 4*y3 + 4) (k4 : x4 + 6 = y4^2 - 4*y4 + 4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 16 := 
sorry

end intersection_sums_l1833_183334


namespace geom_series_common_ratio_l1833_183361

theorem geom_series_common_ratio (a r S : ℝ) (h1 : S = a / (1 - r)) 
  (h2 : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
sorry

end geom_series_common_ratio_l1833_183361


namespace no_real_solution_for_quadratic_eq_l1833_183336

theorem no_real_solution_for_quadratic_eq (y : ℝ) :
  (8 * y^2 + 155 * y + 3) / (4 * y + 45) = 4 * y + 3 →  (¬ ∃ y : ℝ, (8 * y^2 + 37 * y + 33/2 = 0)) :=
by
  sorry

end no_real_solution_for_quadratic_eq_l1833_183336


namespace worker_overtime_hours_l1833_183360

theorem worker_overtime_hours :
  ∃ (x y : ℕ), 60 * x + 90 * y = 3240 ∧ x + y = 50 ∧ y = 8 :=
by
  sorry

end worker_overtime_hours_l1833_183360


namespace mode_of_data_set_is_60_l1833_183310

theorem mode_of_data_set_is_60
  (data : List ℕ := [65, 60, 75, 60, 80])
  (mode : ℕ := 60) :
  mode = 60 ∧ (∀ x ∈ data, data.count x ≤ data.count 60) :=
by {
  sorry
}

end mode_of_data_set_is_60_l1833_183310


namespace average_of_remaining_numbers_l1833_183371

theorem average_of_remaining_numbers 
  (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50) = 20)
  (h_disc : 45 ∈ numbers ∧ 55 ∈ numbers) 
  (h_count_45_55 : numbers.count 45 = 1 ∧ numbers.count 55 = 1) :
  (numbers.sum - 45 - 55) / (50 - 2) = 18.75 :=
by
  sorry

end average_of_remaining_numbers_l1833_183371


namespace no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l1833_183329

theorem no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime :
  ¬∃ n : ℕ, 2 ≤ n ∧ Nat.Prime (n^4 + n^2 + 1) :=
sorry

end no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l1833_183329


namespace C_finishes_work_in_days_l1833_183355

theorem C_finishes_work_in_days :
  (∀ (unit : ℝ) (A B C combined: ℝ),
    combined = 1 / 4 ∧
    A = 1 / 12 ∧
    B = 1 / 24 ∧
    combined = A + B + 1 / C) → 
    C = 8 :=
  sorry

end C_finishes_work_in_days_l1833_183355


namespace triangle_side_relation_l1833_183353

theorem triangle_side_relation (a b c : ℝ) 
    (h_angles : 55 = 55 ∧ 15 = 15 ∧ 110 = 110) :
    c^2 - a^2 = a * b :=
  sorry

end triangle_side_relation_l1833_183353


namespace remainder_19_pow_19_plus_19_mod_20_l1833_183318

theorem remainder_19_pow_19_plus_19_mod_20 : (19 ^ 19 + 19) % 20 = 18 := 
by {
  sorry
}

end remainder_19_pow_19_plus_19_mod_20_l1833_183318


namespace miles_traveled_total_l1833_183341

-- Define the initial distance and the additional distance
def initial_distance : ℝ := 212.3
def additional_distance : ℝ := 372.0

-- Define the total distance as the sum of the initial and additional distances
def total_distance : ℝ := initial_distance + additional_distance

-- Prove that the total distance is 584.3 miles
theorem miles_traveled_total : total_distance = 584.3 := by
  sorry

end miles_traveled_total_l1833_183341


namespace general_eq_line_BC_std_eq_circumscribed_circle_ABC_l1833_183319

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove the general equation of line BC is x + 1 = 0
theorem general_eq_line_BC : ∀ x y : ℝ, (x = -1) → y = 2 ∧ (x = -4) → y = 1 → x + 1 = 0 :=
by
  sorry

-- Prove the standard equation of the circumscribed circle of triangle ABC is (x + 5/2)^2 + (y - 3/2)^2 = 5/2
theorem std_eq_circumscribed_circle_ABC :
  ∀ x y : ℝ,
  (x, y) = (A : ℝ × ℝ) ∨ (x, y) = (B : ℝ × ℝ) ∨ (x, y) = (C : ℝ × ℝ) →
  (x + 5/2)^2 + (y - 3/2)^2 = 5/2 :=
by
  sorry

end general_eq_line_BC_std_eq_circumscribed_circle_ABC_l1833_183319


namespace number_made_l1833_183322

theorem number_made (x y : ℕ) (h1 : x + y = 24) (h2 : x = 11) : 7 * x + 5 * y = 142 := by
  sorry

end number_made_l1833_183322


namespace power_comparison_l1833_183350

noncomputable
def compare_powers : Prop := 
  1.5^(1 / 3.1) < 2^(1 / 3.1) ∧ 2^(1 / 3.1) < 2^(3.1)

theorem power_comparison : compare_powers :=
by
  sorry

end power_comparison_l1833_183350


namespace integer_solutions_system_l1833_183379

theorem integer_solutions_system :
  {x : ℤ | (4 * (1 + x) / 3 - 1 ≤ (5 + x) / 2) ∧ (x - 5 ≤ (3 * (3 * x - 2)) / 2)} = {0, 1, 2} :=
by
  sorry

end integer_solutions_system_l1833_183379


namespace avg_rate_of_change_interval_1_2_l1833_183376

def f (x : ℝ) : ℝ := 2 * x + 1

theorem avg_rate_of_change_interval_1_2 : 
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end avg_rate_of_change_interval_1_2_l1833_183376


namespace river_joe_collected_money_l1833_183368

theorem river_joe_collected_money :
  let price_catfish : ℤ := 600 -- in cents to avoid floating point issues
  let price_shrimp : ℤ := 350 -- in cents to avoid floating point issues
  let total_orders : ℤ := 26
  let shrimp_orders : ℤ := 9
  let catfish_orders : ℤ := total_orders - shrimp_orders
  let total_catfish_sales : ℤ := catfish_orders * price_catfish
  let total_shrimp_sales : ℤ := shrimp_orders * price_shrimp
  let total_money_collected : ℤ := total_catfish_sales + total_shrimp_sales
  total_money_collected = 13350 := -- in cents, so $133.50 is 13350 cents
by
  sorry

end river_joe_collected_money_l1833_183368


namespace find_x_l1833_183317

theorem find_x (x y : ℕ) (h1 : y = 144) (h2 : x^3 * 6^2 / 432 = y) : x = 12 := 
by
  sorry

end find_x_l1833_183317


namespace area_of_R2_l1833_183348

theorem area_of_R2
  (a b : ℝ)
  (h1 : b = 3 * a)
  (h2 : a^2 + b^2 = 225) :
  a * b = 135 / 2 :=
by
  sorry

end area_of_R2_l1833_183348


namespace sum_of_a_and_b_l1833_183305

noncomputable def log_function (a b x : ℝ) : ℝ := Real.log (x + b) / Real.log a

theorem sum_of_a_and_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log_function a b 2 = 1)
                      (h4 : ∃ x : ℝ, log_function a b x = 8 ∧ log_function a b x = 2) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l1833_183305


namespace john_adds_and_subtracts_l1833_183313

theorem john_adds_and_subtracts :
  (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - 79) :=
by {
  sorry
}

end john_adds_and_subtracts_l1833_183313


namespace find_value_l1833_183370

theorem find_value (x : ℤ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 :=
by
  sorry

end find_value_l1833_183370


namespace cost_of_one_bag_of_onions_l1833_183309

theorem cost_of_one_bag_of_onions (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) (h_price : price_per_onion = 200) (h_onions : total_onions = 180) (h_bags : num_bags = 6) :
  (total_onions / num_bags) * price_per_onion = 6000 := 
  by
  sorry

end cost_of_one_bag_of_onions_l1833_183309


namespace intercepts_of_line_l1833_183340

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := x / 4 - y / 3 = 1

-- Define the intercepts
def intercepts (x_intercept y_intercept : ℝ) : Prop :=
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept)

-- The problem statement: proving the values of intercepts
theorem intercepts_of_line :
  intercepts 4 (-3) :=
by
  sorry

end intercepts_of_line_l1833_183340


namespace largest_prime_divisor_in_range_l1833_183342

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.floor (Real.sqrt n) ∧ 
  (∀ q, Prime q ∧ q ≤ Int.floor (Real.sqrt n) → q ≤ p) :=
sorry

end largest_prime_divisor_in_range_l1833_183342


namespace number_of_members_l1833_183375

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end number_of_members_l1833_183375


namespace determine_values_l1833_183363

-- Define the main problem conditions
def A := 1.2
def B := 12

-- The theorem statement capturing the problem conditions and the solution
theorem determine_values (A B : ℝ) (h1 : A + B = 13.2) (h2 : B = 10 * A) : A = 1.2 ∧ B = 12 :=
  sorry

end determine_values_l1833_183363


namespace find_a_b_l1833_183372

theorem find_a_b (a b : ℤ) (h : ({a, 0, -1} : Set ℤ) = {4, b, 0}) : a = 4 ∧ b = -1 := by
  sorry

end find_a_b_l1833_183372


namespace number_of_observations_l1833_183312

theorem number_of_observations (n : ℕ) (h1 : 200 - 6 = 194) (h2 : 200 * n - n * 6 = n * 194) :
  n > 0 :=
by
  sorry

end number_of_observations_l1833_183312


namespace compute_expression_l1833_183365

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end compute_expression_l1833_183365


namespace mnmn_not_cube_in_base_10_and_find_smallest_base_b_l1833_183373

theorem mnmn_not_cube_in_base_10_and_find_smallest_base_b 
    (m n : ℕ) (h1 : m * 10^3 + n * 10^2 + m * 10 + n < 10000) :
    ¬ (∃ k : ℕ, (m * 10^3 + n * 10^2 + m * 10 + n) = k^3) 
    ∧ ∃ b : ℕ, b > 1 ∧ (∃ k : ℕ, (m * b^3 + n * b^2 + m * b + n = k^3)) :=
by sorry

end mnmn_not_cube_in_base_10_and_find_smallest_base_b_l1833_183373


namespace task_completion_time_l1833_183393

theorem task_completion_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 15
  let rate_C := 1 / 15
  let combined_rate := rate_A + rate_B + rate_C
  let working_days_A := 2
  let working_days_B := 1
  let rest_day_A := 1
  let rest_days_B := 2
  let work_done_A := rate_A * working_days_A
  let work_done_B := rate_B * working_days_B
  let work_done_C := rate_C * (working_days_A + rest_day_A)
  let work_done := work_done_A + work_done_B + work_done_C
  let remaining_work := 1 - work_done
  let total_days := (work_done / combined_rate) + rest_day_A + rest_days_B
  total_days = 4 + 1 / 7 := by sorry

end task_completion_time_l1833_183393


namespace min_value_of_even_function_l1833_183349

-- Define f(x) = (x + a)(x + b)
def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

-- Given conditions
variables (a b : ℝ)
#check f  -- Ensuring the definition works

-- Prove that the minimum value of f(x) is -4 given that f(x) is an even function
theorem min_value_of_even_function (h_even : ∀ x : ℝ, f x a b = f (-x) a b)
  (h_domain : a + 4 > a) : ∃ c : ℝ, (f c a b = -4) :=
by
  -- We state that this function is even and consider the provided domain.
  sorry  -- Placeholder for the proof

end min_value_of_even_function_l1833_183349


namespace danny_distance_to_work_l1833_183346

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l1833_183346


namespace incorrect_equation_l1833_183323

theorem incorrect_equation (x : ℕ) (h : x + 2 * (12 - x) = 20) : 2 * (12 - x) - 20 ≠ x :=
by 
  sorry

end incorrect_equation_l1833_183323


namespace find_y_l1833_183343

-- Define the conditions (inversely proportional and sum condition)
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k
def sum_condition (x y : ℝ) : Prop := x + y = 50 ∧ x = 3 * y

-- Given these conditions, prove the value of y when x = -12
theorem find_y (k x y : ℝ)
  (h1 : inversely_proportional x y k)
  (h2 : sum_condition 37.5 12.5)
  (hx : x = -12) :
  y = -39.0625 :=
sorry

end find_y_l1833_183343


namespace boxes_per_case_l1833_183303

-- Define the conditions
def total_boxes : ℕ := 54
def total_cases : ℕ := 9

-- Define the result we want to prove
theorem boxes_per_case : total_boxes / total_cases = 6 := 
by sorry

end boxes_per_case_l1833_183303


namespace DanteSoldCoconuts_l1833_183333

variable (Paolo_coconuts : ℕ) (Dante_coconuts : ℕ) (coconuts_left : ℕ)

def PaoloHasCoconuts := Paolo_coconuts = 14

def DanteHasThriceCoconuts := Dante_coconuts = 3 * Paolo_coconuts

def DanteLeftCoconuts := coconuts_left = 32

theorem DanteSoldCoconuts 
  (h1 : PaoloHasCoconuts Paolo_coconuts) 
  (h2 : DanteHasThriceCoconuts Paolo_coconuts Dante_coconuts) 
  (h3 : DanteLeftCoconuts coconuts_left) : 
  Dante_coconuts - coconuts_left = 10 := 
by
  rw [PaoloHasCoconuts, DanteHasThriceCoconuts, DanteLeftCoconuts] at *
  sorry

end DanteSoldCoconuts_l1833_183333


namespace parent_payment_per_year_l1833_183356

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end parent_payment_per_year_l1833_183356


namespace greatest_four_digit_number_divisible_by_6_and_12_l1833_183380

theorem greatest_four_digit_number_divisible_by_6_and_12 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧ (n % 12 = 0) ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 6 = 0) ∧ (m % 12 = 0) → m ≤ n) ∧
  n = 9996 := 
by
  sorry

end greatest_four_digit_number_divisible_by_6_and_12_l1833_183380


namespace maddie_episodes_friday_l1833_183384

theorem maddie_episodes_friday :
  let total_episodes : ℕ := 8
  let episode_duration : ℕ := 44
  let monday_time : ℕ := 138
  let thursday_time : ℕ := 21
  let weekend_time : ℕ := 105
  let total_time : ℕ := total_episodes * episode_duration
  let non_friday_time : ℕ := monday_time + thursday_time + weekend_time
  let friday_time : ℕ := total_time - non_friday_time
  let friday_episodes : ℕ := friday_time / episode_duration
  friday_episodes = 2 :=
by
  sorry

end maddie_episodes_friday_l1833_183384
