import Mathlib

namespace remaining_pencils_l243_24340

/-
Given the initial number of pencils in the drawer and the number of pencils Sally took out,
prove that the number of pencils remaining in the drawer is 5.
-/
def pencils_in_drawer (initial_pencils : ℕ) (pencils_taken : ℕ) : ℕ :=
  initial_pencils - pencils_taken

theorem remaining_pencils : pencils_in_drawer 9 4 = 5 := by
  sorry

end remaining_pencils_l243_24340


namespace gasoline_tank_capacity_l243_24366

theorem gasoline_tank_capacity (x : ℕ) (h1 : 5 * x / 6 - 2 * x / 3 = 15) : x = 90 :=
sorry

end gasoline_tank_capacity_l243_24366


namespace max_a_plus_b_cubed_plus_c_fourth_l243_24373

theorem max_a_plus_b_cubed_plus_c_fourth (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 := sorry

end max_a_plus_b_cubed_plus_c_fourth_l243_24373


namespace selling_price_correct_l243_24375

/-- Define the total number of units to be sold -/
def total_units : ℕ := 5000

/-- Define the variable cost per unit -/
def variable_cost_per_unit : ℕ := 800

/-- Define the total fixed costs -/
def fixed_costs : ℕ := 1000000

/-- Define the desired profit -/
def desired_profit : ℕ := 1500000

/-- The selling price p must be calculated such that revenues exceed expenses by the desired profit -/
theorem selling_price_correct : 
  ∃ p : ℤ, p = 1300 ∧ (total_units * p) - (fixed_costs + (total_units * variable_cost_per_unit)) = desired_profit :=
by
  sorry

end selling_price_correct_l243_24375


namespace product_divisible_by_15_l243_24363

theorem product_divisible_by_15 (n : ℕ) (hn1 : n % 2 = 1) (hn2 : n > 0) :
  15 ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end product_divisible_by_15_l243_24363


namespace units_digit_5_pow_17_mul_4_l243_24301

theorem units_digit_5_pow_17_mul_4 : ((5 ^ 17) * 4) % 10 = 0 :=
by
  sorry

end units_digit_5_pow_17_mul_4_l243_24301


namespace remainder_of_sum_division_l243_24384

def a1 : ℕ := 2101
def a2 : ℕ := 2103
def a3 : ℕ := 2105
def a4 : ℕ := 2107
def a5 : ℕ := 2109
def n : ℕ := 12

theorem remainder_of_sum_division : ((a1 + a2 + a3 + a4 + a5) % n) = 1 :=
by {
  sorry
}

end remainder_of_sum_division_l243_24384


namespace last_digit_of_x95_l243_24350

theorem last_digit_of_x95 (x : ℕ) : 
  (x^95 % 10) - (3^58 % 10) = 4 % 10 → (x^95 % 10 = 3) := by
  sorry

end last_digit_of_x95_l243_24350


namespace distribution_ways_l243_24330

theorem distribution_ways (n_problems n_friends : ℕ) (h_problems : n_problems = 6) (h_friends : n_friends = 8) : (n_friends ^ n_problems) = 262144 :=
by
  rw [h_problems, h_friends]
  norm_num

end distribution_ways_l243_24330


namespace problem_1_problem_2_l243_24331

theorem problem_1 
  (h1 : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : 
  Int.floor (5 - Real.sqrt 2) = 3 :=
sorry

theorem problem_2 
  (h2 : Real.sqrt 3 > 1) : 
  abs (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1 :=
sorry

end problem_1_problem_2_l243_24331


namespace tyler_cd_purchase_l243_24346

theorem tyler_cd_purchase :
  ∀ (initial_cds : ℕ) (given_away_fraction : ℝ) (final_cds : ℕ) (bought_cds : ℕ),
    initial_cds = 21 →
    given_away_fraction = 1 / 3 →
    final_cds = 22 →
    bought_cds = 8 →
    final_cds = initial_cds - initial_cds * given_away_fraction + bought_cds :=
by
  intros
  sorry

end tyler_cd_purchase_l243_24346


namespace complex_pure_imaginary_l243_24343

theorem complex_pure_imaginary (a : ℝ) : (↑a + Complex.I) / (1 - Complex.I) = 0 + b * Complex.I → a = 1 :=
by
  intro h
  -- Proof content here
  sorry

end complex_pure_imaginary_l243_24343


namespace zinc_copper_mixture_weight_l243_24319

theorem zinc_copper_mixture_weight (Z C : ℝ) (h1 : Z / C = 9 / 11) (h2 : Z = 31.5) : Z + C = 70 := by
  sorry

end zinc_copper_mixture_weight_l243_24319


namespace amy_race_time_l243_24333

theorem amy_race_time (patrick_time : ℕ) (manu_time : ℕ) (amy_time : ℕ)
  (h1 : patrick_time = 60)
  (h2 : manu_time = patrick_time + 12)
  (h3 : amy_time = manu_time / 2) : 
  amy_time = 36 := 
sorry

end amy_race_time_l243_24333


namespace probability_three_blue_jellybeans_l243_24338

theorem probability_three_blue_jellybeans:
  let total_jellybeans := 20
  let blue_jellybeans := 10
  let red_jellybeans := 10
  let draws := 3
  let q := (1 / 2) * (9 / 19) * (4 / 9)
  q = 2 / 19 :=
sorry

end probability_three_blue_jellybeans_l243_24338


namespace page_mistakenly_added_twice_l243_24352

theorem page_mistakenly_added_twice (n k: ℕ) (h₁: n = 77) (h₂: (n * (n + 1)) / 2 + k = 3050) : k = 47 :=
by
  -- sorry here to indicate the proof is not needed
  sorry

end page_mistakenly_added_twice_l243_24352


namespace inverse_h_l243_24312

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h (x : ℝ) : h⁻¹ (x) = (x - 7) / 12 :=
sorry

end inverse_h_l243_24312


namespace rectangle_length_eq_15_l243_24374

theorem rectangle_length_eq_15 (w l s p_rect p_square : ℝ)
    (h_w : w = 9)
    (h_s : s = 12)
    (h_p_square : p_square = 4 * s)
    (h_p_rect : p_rect = 2 * w + 2 * l)
    (h_eq_perimeters : p_square = p_rect) : l = 15 := by
  sorry

end rectangle_length_eq_15_l243_24374


namespace wine_barrels_l243_24329

theorem wine_barrels :
  ∃ x y : ℝ, (6 * x + 4 * y = 48) ∧ (5 * x + 3 * y = 38) :=
by
  -- Proof is left out
  sorry

end wine_barrels_l243_24329


namespace cesaro_lupu_real_analysis_l243_24362

noncomputable def proof_problem (a b c x y z : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) ∧
  (0 < x) ∧ (0 < y) ∧ (0 < z) ∧
  (a^x = b * c) ∧ (b^y = c * a) ∧ (c^z = a * b) →
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z) ≤ 3 / 4)

theorem cesaro_lupu_real_analysis (a b c x y z : ℝ) :
  proof_problem a b c x y z :=
by sorry

end cesaro_lupu_real_analysis_l243_24362


namespace negation_exists_positive_real_square_plus_one_l243_24325

def exists_positive_real_square_plus_one : Prop :=
  ∃ (x : ℝ), x^2 + 1 > 0

def forall_non_positive_real_square_plus_one : Prop :=
  ∀ (x : ℝ), x^2 + 1 ≤ 0

theorem negation_exists_positive_real_square_plus_one :
  ¬ exists_positive_real_square_plus_one ↔ forall_non_positive_real_square_plus_one :=
by
  sorry

end negation_exists_positive_real_square_plus_one_l243_24325


namespace solve_for_c_l243_24361

variables (m c b a : ℚ) -- Declaring variables as rationals for added precision

theorem solve_for_c (h : m = (c * b * a) / (a - c)) : 
  c = (m * a) / (m + b * a) := 
by 
  sorry -- Proof not required as per the instructions

end solve_for_c_l243_24361


namespace remainder_of_8673_div_7_l243_24354

theorem remainder_of_8673_div_7 : 8673 % 7 = 3 :=
by
  -- outline structure, proof to be inserted
  sorry

end remainder_of_8673_div_7_l243_24354


namespace water_fraction_after_replacements_l243_24328

-- Initially given conditions
def radiator_capacity : ℚ := 20
def initial_water_fraction : ℚ := 1
def antifreeze_quarts : ℚ := 5
def replacements : ℕ := 5

-- Derived condition
def water_remain_fraction : ℚ := 3 / 4

-- Statement of the problem
theorem water_fraction_after_replacements :
  (water_remain_fraction ^ replacements) = 243 / 1024 :=
by
  -- Proof goes here
  sorry

end water_fraction_after_replacements_l243_24328


namespace contradiction_with_angles_l243_24388

-- Definitions of conditions
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

-- The proposition we want to prove by contradiction
def at_least_one_angle_not_greater_than_60 (α β γ : ℝ) : Prop := α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The assumption for contradiction
def all_angles_greater_than_60 (α β γ : ℝ) : Prop := α > 60 ∧ β > 60 ∧ γ > 60

-- The proof problem
theorem contradiction_with_angles (α β γ : ℝ) (h : triangle α β γ) :
  ¬ all_angles_greater_than_60 α β γ → at_least_one_angle_not_greater_than_60 α β γ :=
sorry

end contradiction_with_angles_l243_24388


namespace find_sin_angle_BAD_l243_24396

def isosceles_right_triangle (A B C : ℝ → ℝ → Prop) (AB BC AC : ℝ) : Prop :=
  AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2

def right_triangle_on_hypotenuse (A C D : ℝ → ℝ → Prop) (AC CD DA : ℝ) (DAC : ℝ) : Prop :=
  AC = 2 * Real.sqrt 2 ∧ CD = DA / 2 ∧ DAC = Real.pi / 6

def equal_perimeters (AC CD DA : ℝ) : Prop := 
  AC + CD + DA = 4 + 2 * Real.sqrt 2

theorem find_sin_angle_BAD :
  ∀ (A B C D : ℝ → ℝ → Prop) (AB BC AC CD DA : ℝ),
  isosceles_right_triangle A B C AB BC AC →
  right_triangle_on_hypotenuse A C D AC CD DA (Real.pi / 6) →
  equal_perimeters AC CD DA →
  Real.sin (2 * (Real.pi / 4 + Real.pi / 6)) = 1 / 2 :=
by
  intros
  sorry

end find_sin_angle_BAD_l243_24396


namespace real_roots_exist_for_nonzero_K_l243_24317

theorem real_roots_exist_for_nonzero_K (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by
  sorry

end real_roots_exist_for_nonzero_K_l243_24317


namespace midpoint_sum_l243_24391

theorem midpoint_sum (x1 y1 x2 y2 : ℕ) (h₁ : x1 = 4) (h₂ : y1 = 7) (h₃ : x2 = 12) (h₄ : y2 = 19) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 21 :=
by
  sorry

end midpoint_sum_l243_24391


namespace minimum_routes_A_C_l243_24392

namespace SettlementRoutes

-- Define three settlements A, B, and C
variable (A B C : Type)

-- Assume there are more than one roads connecting each settlement pair directly
variable (k m n : ℕ) -- k: roads between A and B, m: roads between B and C, n: roads between A and C

-- Conditions: Total paths including intermediate nodes
axiom h1 : k + m * n = 34
axiom h2 : m + k * n = 29

-- Theorem: Minimum number of routes connecting A and C is 26
theorem minimum_routes_A_C : ∃ n k m : ℕ, k + m * n = 34 ∧ m + k * n = 29 ∧ n + k * m = 26 := sorry

end SettlementRoutes

end minimum_routes_A_C_l243_24392


namespace range_of_a_l243_24305

-- Define propositions p and q
def p := { x : ℝ | (4 * x - 3) ^ 2 ≤ 1 }
def q (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- Define sets A and B
def A := { x : ℝ | 1 / 2 ≤ x ∧ x ≤ 1 }
def B (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

-- negation of p (p' is a necessary but not sufficient condition for q')
def p_neg := { x : ℝ | ¬ ((4 * x - 3) ^ 2 ≤ 1) }
def q_neg (a : ℝ) := { x : ℝ | ¬ (a ≤ x ∧ x ≤ a + 1) }

-- range of real number a
theorem range_of_a (a : ℝ) : (A ⊆ B a ∧ A ≠ B a) → 0 ≤ a ∧ a ≤ 1 / 2 := by
  sorry

end range_of_a_l243_24305


namespace find_f_28_l243_24344

theorem find_f_28 (f : ℕ → ℚ) (h1 : ∀ n : ℕ, f (n + 1) = (3 * f n + n) / 3) (h2 : f 1 = 1) :
  f 28 = 127 := by
sorry

end find_f_28_l243_24344


namespace medicine_dose_per_part_l243_24394

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l243_24394


namespace geom_seq_mult_l243_24326

variable {α : Type*} [LinearOrderedField α]

def is_geom_seq (a : ℕ → α) :=
  ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geom_seq_mult (a : ℕ → α) (h : is_geom_seq a) (hpos : ∀ n, 0 < a n) (h4_8 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 := 
sorry

end geom_seq_mult_l243_24326


namespace lattice_point_distance_l243_24377

theorem lattice_point_distance (d : ℝ) : 
  (∃ (r : ℝ), r = 2020 ∧ (∀ (A B C D : ℝ), 
  A = 0 ∧ B = 4040 ∧ C = 2020 ∧ D = 4040) 
  ∧ (∃ (P Q : ℝ), P = 0.25 ∧ Q = 1)) → 
  d = 0.3 := 
by
  sorry

end lattice_point_distance_l243_24377


namespace not_possible_perimeter_72_l243_24339

variable (a b : ℕ)
variable (P : ℕ)

def valid_perimeter_range (a b : ℕ) : Set ℕ := 
  { P | ∃ x, 15 < x ∧ x < 35 ∧ P = a + b + x }

theorem not_possible_perimeter_72 :
  (a = 10) → (b = 25) → ¬ (72 ∈ valid_perimeter_range 10 25) := 
by
  sorry

end not_possible_perimeter_72_l243_24339


namespace series_converges_to_l243_24342

noncomputable def series_sum := ∑' n : Nat, (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

theorem series_converges_to : series_sum = 1 / 200 := 
by 
  sorry

end series_converges_to_l243_24342


namespace central_angle_of_sector_l243_24383

theorem central_angle_of_sector {r l : ℝ} 
  (h1 : 2 * r + l = 4) 
  (h2 : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by 
  sorry

end central_angle_of_sector_l243_24383


namespace mike_unbroken_seashells_l243_24300

-- Define the conditions from the problem
def totalSeashells : ℕ := 6
def brokenSeashells : ℕ := 4
def unbrokenSeashells : ℕ := totalSeashells - brokenSeashells

-- Statement to prove
theorem mike_unbroken_seashells : unbrokenSeashells = 2 := by
  sorry

end mike_unbroken_seashells_l243_24300


namespace find_greatest_natural_number_l243_24353

-- Definitions for terms used in the conditions

def sum_of_squares (m : ℕ) : ℕ :=
  (m * (m + 1) * (2 * m + 1)) / 6

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, b * b = a

-- Conditions defined in Lean terms
def condition1 (n : ℕ) : Prop := n ≤ 2010

def condition2 (n : ℕ) : Prop := 
  let sum1 := sum_of_squares n
  let sum2 := sum_of_squares (2 * n) - sum_of_squares n
  is_perfect_square (sum1 * sum2)

-- Main theorem statement
theorem find_greatest_natural_number : ∃ n, n ≤ 2010 ∧ condition2 n ∧ ∀ m, m ≤ 2010 ∧ condition2 m → m ≤ n := 
by 
  sorry

end find_greatest_natural_number_l243_24353


namespace sum_bases_exponents_max_product_l243_24323

theorem sum_bases_exponents_max_product (A : ℕ) (hA : A = 3 ^ 670 * 2 ^ 2) : 
    (3 + 2 + 670 + 2 = 677) := by
  sorry

end sum_bases_exponents_max_product_l243_24323


namespace proof_equation_of_line_l243_24359
   
   -- Define the point P
   structure Point where
     x : ℝ
     y : ℝ
     
   -- Define conditions
   def passesThroughP (line : ℝ → ℝ → Prop) : Prop :=
     line 2 (-1)
     
   def interceptRelation (line : ℝ → ℝ → Prop) : Prop :=
     ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, line x y ↔ (x / a + y / (2 * a) = 1))
   
   -- Define the line equation
   def line_equation (line : ℝ → ℝ → Prop) : Prop :=
     passesThroughP line ∧ interceptRelation line
     
   -- The final statement
   theorem proof_equation_of_line (line : ℝ → ℝ → Prop) :
     line_equation line →
     (∀ x y, line x y ↔ (2 * x + y = 3)) ∨ (∀ x y, line x y ↔ (x + 2 * y = 0)) :=
   by
     sorry
   
end proof_equation_of_line_l243_24359


namespace comparison_arctan_l243_24310

theorem comparison_arctan (a b c : ℝ) (h : Real.arctan a + Real.arctan b + Real.arctan c + Real.pi / 2 = 0) :
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) :=
by
  sorry

end comparison_arctan_l243_24310


namespace gcd_372_684_is_12_l243_24334

theorem gcd_372_684_is_12 :
  Nat.gcd 372 684 = 12 :=
sorry

end gcd_372_684_is_12_l243_24334


namespace vanessa_earnings_l243_24347

def cost : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost

theorem vanessa_earnings : money_made = 16 := by
  sorry

end vanessa_earnings_l243_24347


namespace product_odd_primes_mod_32_l243_24393

open Nat

theorem product_odd_primes_mod_32 : 
  let primes := [3, 5, 7, 11, 13] 
  let product := primes.foldl (· * ·) 1 
  product % 32 = 7 := 
by
  sorry

end product_odd_primes_mod_32_l243_24393


namespace integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l243_24345

-- Proof problem 1
theorem integers_abs_no_greater_than_2 :
    {n : ℤ | |n| ≤ 2} = {-2, -1, 0, 1, 2} :=
by {
  sorry
}

-- Proof problem 2
theorem pos_div_by_3_less_than_10 :
    {n : ℕ | n > 0 ∧ n % 3 = 0 ∧ n < 10} = {3, 6, 9} :=
by {
  sorry
}

-- Proof problem 3
theorem non_neg_int_less_than_5 :
    {n : ℤ | n = |n| ∧ n < 5} = {0, 1, 2, 3, 4} :=
by {
  sorry
}

-- Proof problem 4
theorem sum_eq_6_in_nat :
    {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0} = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} :=
by {
  sorry
}

-- Proof problem 5
theorem expressing_sequence:
    {-3, -1, 1, 3, 5} = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3} :=
by {
  sorry
}

end integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l243_24345


namespace minimum_xy_l243_24332

theorem minimum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 1/y = 1/2) : xy ≥ 16 :=
sorry

end minimum_xy_l243_24332


namespace swim_club_members_l243_24327

theorem swim_club_members (X : ℝ) 
  (h1 : 0.30 * X = 0.30 * X)
  (h2 : 0.70 * X = 42) : X = 60 :=
sorry

end swim_club_members_l243_24327


namespace distance_walked_on_third_day_l243_24335

theorem distance_walked_on_third_day:
  ∃ x : ℝ, 
    4 * x + 2 * x + x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 378 ∧
    x = 48 := 
by
  sorry

end distance_walked_on_third_day_l243_24335


namespace inf_many_non_prime_additions_l243_24376

theorem inf_many_non_prime_additions :
  ∃ᶠ (a : ℕ) in at_top, ∀ n : ℕ, n > 0 → ¬ Prime (n^4 + a) :=
by {
  sorry -- proof to be provided
}

end inf_many_non_prime_additions_l243_24376


namespace find_angle_A_min_perimeter_l243_24397

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h₄ : a > 0 ∧ b > 0 ∧ c > 0) (h5 : b + c * Real.cos A = c + a * Real.cos C) 
  (hTriangle : A + B + C = Real.pi)
  (hSineLaw : Real.sin B = Real.sin C * Real.cos A + Real.sin A * Real.cos C) :
  A = Real.pi / 3 := 
by 
  sorry

theorem min_perimeter (a b c : ℝ) (A : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ A = Real.pi / 3)
  (h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3)
  (h_cosine : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  a + b + c = 6 :=
by 
  sorry

end find_angle_A_min_perimeter_l243_24397


namespace find_c_l243_24309

theorem find_c (c : ℝ) 
  (h1 : ∃ x : ℝ, 3 * x^2 + 23 * x - 75 = 0 ∧ x = ⌊c⌋) 
  (h2 : ∃ y : ℝ, 4 * y^2 - 19 * y + 3 = 0 ∧ y = c - ⌊c⌋) : 
  c = -11.84 :=
by
  sorry

end find_c_l243_24309


namespace exists_face_with_fewer_than_six_sides_l243_24321

theorem exists_face_with_fewer_than_six_sides
  (N K M : ℕ) 
  (h_euler : N - K + M = 2)
  (h_vertices : M ≤ 2 * K / 3) : 
  ∃ n_i : ℕ, n_i < 6 :=
by
  sorry

end exists_face_with_fewer_than_six_sides_l243_24321


namespace ratio_of_men_to_women_l243_24324
open Nat

theorem ratio_of_men_to_women 
  (total_players : ℕ) 
  (players_per_group : ℕ) 
  (extra_women_per_group : ℕ) 
  (H_total_players : total_players = 20) 
  (H_players_per_group : players_per_group = 3) 
  (H_extra_women_per_group : extra_women_per_group = 1) 
  : (7 / 13 : ℝ) = 7 / 13 :=
by
  -- Conditions
  have H1 : total_players = 20 := H_total_players
  have H2 : players_per_group = 3 := H_players_per_group
  have H3 : extra_women_per_group = 1 := H_extra_women_per_group
  -- The correct answer
  sorry

end ratio_of_men_to_women_l243_24324


namespace gcd_euclidean_algorithm_l243_24357

theorem gcd_euclidean_algorithm (a b : ℕ) : 
  ∃ d : ℕ, d = gcd a b ∧ ∀ m : ℕ, (m ∣ a ∧ m ∣ b) → m ∣ d :=
by
  sorry

end gcd_euclidean_algorithm_l243_24357


namespace coin_flips_probability_equal_heads_l243_24399

def fair_coin (p : ℚ) := p = 1 / 2
def second_coin (p : ℚ) := p = 3 / 5
def third_coin (p : ℚ) := p = 2 / 3

theorem coin_flips_probability_equal_heads :
  ∀ p1 p2 p3, fair_coin p1 → second_coin p2 → third_coin p3 →
  ∃ m n, m + n = 119 ∧ m / n = 29 / 90 :=
by
  sorry

end coin_flips_probability_equal_heads_l243_24399


namespace num_people_end_race_l243_24315

-- Define the conditions
def num_cars : ℕ := 20
def initial_passengers_per_car : ℕ := 2
def drivers_per_car : ℕ := 1
def additional_passengers_per_car : ℕ := 1

-- Define the total number of people in a car at the start
def total_people_per_car_initial := initial_passengers_per_car + drivers_per_car

-- Define the total number of people in a car after halfway point
def total_people_per_car_end := total_people_per_car_initial + additional_passengers_per_car

-- Define the total number of people in all cars at the end
def total_people_end := num_cars * total_people_per_car_end

-- Theorem statement
theorem num_people_end_race : total_people_end = 80 := by
  sorry

end num_people_end_race_l243_24315


namespace min_value_of_expression_l243_24367

theorem min_value_of_expression :
  ∃ (a b : ℝ), (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ x^2 + a * x + b - 3 = 0) ∧ a^2 + (b - 4)^2 = 2 :=
sorry

end min_value_of_expression_l243_24367


namespace lucas_change_l243_24348

-- Define the initial amount of money Lucas has
def initial_amount : ℕ := 20

-- Define the cost of one avocado
def cost_per_avocado : ℕ := 2

-- Define the number of avocados Lucas buys
def number_of_avocados : ℕ := 3

-- Calculate the total cost of avocados
def total_cost : ℕ := number_of_avocados * cost_per_avocado

-- Calculate the remaining amount of money (change)
def remaining_amount : ℕ := initial_amount - total_cost

-- The proposition to prove: Lucas brings home $14
theorem lucas_change : remaining_amount = 14 := by
  sorry

end lucas_change_l243_24348


namespace nursery_school_students_l243_24386

theorem nursery_school_students (S : ℕ)
  (h1 : ∃ x, x = S / 10)
  (h2 : 20 + (S / 10) = 25) : S = 50 :=
by
  sorry

end nursery_school_students_l243_24386


namespace intersection_of_A_and_B_l243_24336

open Set

def A : Set Int := {x | x + 2 = 0}

def B : Set Int := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by
  sorry

end intersection_of_A_and_B_l243_24336


namespace solve_cyclic_quadrilateral_area_l243_24389

noncomputable def cyclic_quadrilateral_area (AB BC AD CD : ℝ) (cyclic : Bool) : ℝ :=
  if cyclic ∧ AB = 2 ∧ BC = 6 ∧ AD = 4 ∧ CD = 4 then 8 * Real.sqrt 3 else 0

theorem solve_cyclic_quadrilateral_area :
  cyclic_quadrilateral_area 2 6 4 4 true = 8 * Real.sqrt 3 :=
by
  sorry

end solve_cyclic_quadrilateral_area_l243_24389


namespace sufficient_not_necessary_not_necessary_l243_24303

theorem sufficient_not_necessary (x : ℝ) (h1: x > 2) : x^2 - 3 * x + 2 > 0 :=
sorry

theorem not_necessary (x : ℝ) (h2: x^2 - 3 * x + 2 > 0) : (x > 2 ∨ x < 1) :=
sorry

end sufficient_not_necessary_not_necessary_l243_24303


namespace monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l243_24380

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Monotonicity of f(x)
theorem monotonic_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

-- 2. f(x) is odd implies a = 1
theorem odd_function_implies_a_eq_1 (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

-- 3. Find max m such that f(x) ≥ m / 2^x for all x ∈ [2, 3]
theorem find_max_m (h : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f 1 x ≥ m / 2^x) : m ≤ 12/5 := sorry

end monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l243_24380


namespace weight_of_b_l243_24307

theorem weight_of_b (a b c d : ℝ)
  (h1 : a + b + c + d = 160)
  (h2 : a + b = 50)
  (h3 : b + c = 56)
  (h4 : c + d = 64) :
  b = 46 :=
by sorry

end weight_of_b_l243_24307


namespace koala_fiber_absorption_l243_24379

theorem koala_fiber_absorption (x : ℝ) (h1 : 0 < x) (h2 : x * 0.30 = 15) : x = 50 :=
sorry

end koala_fiber_absorption_l243_24379


namespace initial_scooter_value_l243_24358

theorem initial_scooter_value (V : ℝ) (h : V * (3/4)^2 = 22500) : V = 40000 :=
by
  sorry

end initial_scooter_value_l243_24358


namespace evaluate_expression_at_one_l243_24349

theorem evaluate_expression_at_one : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end evaluate_expression_at_one_l243_24349


namespace calculate_fraction_l243_24304

theorem calculate_fraction :
  let a := 7
  let b := 5
  let c := -2
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 + c^2) = 460 / 43 :=
by
  sorry

end calculate_fraction_l243_24304


namespace initial_girls_l243_24313

theorem initial_girls (G : ℕ) 
  (h1 : G + 7 + (15 - 4) = 36) : G = 18 :=
by
  sorry

end initial_girls_l243_24313


namespace geometric_sequence_sum_l243_24320

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ 
  (a 2 * a 4 = 1) ∧ 
  (a 1 * (q^0 + q^1 + q^2) = 7) ∧ 
  (a 1 / (1 - q) * (1 - q^5) = 31 / 4) := by
  sorry

end geometric_sequence_sum_l243_24320


namespace num_children_eq_3_l243_24390

-- Definitions from the conditions
def regular_ticket_cost : ℕ := 9
def child_ticket_discount : ℕ := 2
def given_amount : ℕ := 20 * 2
def received_change : ℕ := 1
def num_adults : ℕ := 2

-- Derived data
def total_ticket_cost : ℕ := given_amount - received_change
def adult_ticket_cost : ℕ := num_adults * regular_ticket_cost
def children_ticket_cost : ℕ := total_ticket_cost - adult_ticket_cost
def child_ticket_cost : ℕ := regular_ticket_cost - child_ticket_discount

-- Statement to prove
theorem num_children_eq_3 : (children_ticket_cost / child_ticket_cost) = 3 := by
  sorry

end num_children_eq_3_l243_24390


namespace tan_sum_simplification_l243_24318
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l243_24318


namespace unique_factor_and_multiple_of_13_l243_24372

theorem unique_factor_and_multiple_of_13 (n : ℕ) (h1 : n ∣ 13) (h2 : 13 ∣ n) : n = 13 :=
sorry

end unique_factor_and_multiple_of_13_l243_24372


namespace sophie_marble_exchange_l243_24314

theorem sophie_marble_exchange (sophie_initial_marbles joe_initial_marbles : ℕ) 
  (final_ratio : ℕ) (sophie_gives_joe : ℕ) : 
  sophie_initial_marbles = 120 → joe_initial_marbles = 19 → final_ratio = 3 → 
  (120 - sophie_gives_joe = 3 * (19 + sophie_gives_joe)) → sophie_gives_joe = 16 := 
by
  intros h1 h2 h3 h4
  sorry

end sophie_marble_exchange_l243_24314


namespace exact_consecutive_hits_l243_24365

/-
Prove the number of ways to arrange 8 shots with exactly 3 hits such that exactly 2 out of the 3 hits are consecutive is 30.
-/

def count_distinct_sequences (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ hits = 3 ∧ consecutive_hits = 2 then 30 else 0

theorem exact_consecutive_hits :
  count_distinct_sequences 8 3 2 = 30 :=
by
  -- The proof is omitted.
  sorry

end exact_consecutive_hits_l243_24365


namespace fare_calculation_l243_24302

-- Definitions for given conditions
def initial_mile_fare : ℝ := 3.00
def additional_rate : ℝ := 0.30
def initial_miles : ℝ := 0.5
def available_fare : ℝ := 15 - 3  -- Total minus tip

-- Proof statement
theorem fare_calculation (miles : ℝ) : initial_mile_fare + additional_rate * (miles - initial_miles) / 0.10 = available_fare ↔ miles = 3.5 :=
by
  sorry

end fare_calculation_l243_24302


namespace consecutive_vertices_product_l243_24364

theorem consecutive_vertices_product (n : ℕ) (hn : n = 90) :
  ∃ (i : ℕ), 1 ≤ i ∧ i ≤ n ∧ ((i * (i % n + 1)) ≥ 2014) := 
sorry

end consecutive_vertices_product_l243_24364


namespace sum_of_a_and_b_l243_24351

noncomputable def a : ℕ :=
sorry

noncomputable def b : ℕ :=
sorry

theorem sum_of_a_and_b :
  (100 ≤ a ∧ a ≤ 999) ∧ (1000 ≤ b ∧ b ≤ 9999) ∧ (10000 * a + b = 7 * a * b) ->
  a + b = 1458 :=
by
  sorry

end sum_of_a_and_b_l243_24351


namespace eliza_is_shorter_by_2_inch_l243_24311

theorem eliza_is_shorter_by_2_inch
  (total_height : ℕ)
  (height_sibling1 height_sibling2 height_sibling3 height_eliza : ℕ) :
  total_height = 330 →
  height_sibling1 = 66 →
  height_sibling2 = 66 →
  height_sibling3 = 60 →
  height_eliza = 68 →
  total_height - (height_sibling1 + height_sibling2 + height_sibling3 + height_eliza) - height_eliza = 2 :=
by
  sorry

end eliza_is_shorter_by_2_inch_l243_24311


namespace two_A_plus_B_l243_24385

theorem two_A_plus_B (A B : ℕ) (h1 : A = Nat.gcd (Nat.gcd 12 18) 30) (h2 : B = Nat.lcm (Nat.lcm 12 18) 30) : 2 * A + B = 192 :=
by
  sorry

end two_A_plus_B_l243_24385


namespace picture_frame_length_l243_24395

theorem picture_frame_length (h : ℕ) (l : ℕ) (P : ℕ) (h_eq : h = 12) (P_eq : P = 44) (perimeter_eq : P = 2 * (l + h)) : l = 10 :=
by
  -- proof would go here
  sorry

end picture_frame_length_l243_24395


namespace distance_A_focus_l243_24356

-- Definitions from the problem conditions
def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def point_A (x : ℝ) : Prop := parabola_eq x 4
def focus_y_coord : ℝ := 1 -- Derived from the standard form of the parabola x^2 = 4py where p=1

-- State the theorem in Lean 4
theorem distance_A_focus (x : ℝ) (hA : point_A x) : |4 - focus_y_coord| = 3 :=
by
  -- Proof would go here
  sorry

end distance_A_focus_l243_24356


namespace smaller_of_x_and_y_is_15_l243_24381

variable {x y : ℕ}

/-- Given two positive numbers x and y are in the ratio 3:5, 
and the sum of x and y plus 10 equals 50,
prove that the smaller of x and y is 15. -/
theorem smaller_of_x_and_y_is_15 (h1 : x * 5 = y * 3) (h2 : x + y + 10 = 50) (h3 : 0 < x) (h4 : 0 < y) : x = 15 :=
by
  sorry

end smaller_of_x_and_y_is_15_l243_24381


namespace similar_triangles_perimeters_and_area_ratios_l243_24398

theorem similar_triangles_perimeters_and_area_ratios
  (m1 m2 : ℝ) (p_sum : ℝ) (ratio_p : ℝ) (ratio_a : ℝ) :
  m1 = 10 →
  m2 = 4 →
  p_sum = 140 →
  ratio_p = 5 / 2 →
  ratio_a = 25 / 4 →
  (∃ (p1 p2 : ℝ), p1 + p2 = p_sum ∧ p1 = (5 / 7) * p_sum ∧ p2 = (2 / 7) * p_sum ∧ ratio_a = (ratio_p)^2) :=
by
  sorry

end similar_triangles_perimeters_and_area_ratios_l243_24398


namespace right_triangle_condition_l243_24322

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B = 90) → (A + B + C = 180) → (C = 90) := 
by
  sorry

end right_triangle_condition_l243_24322


namespace imaginary_part_of_z_l243_24316

-- Step 1: Define the imaginary unit.
def i : ℂ := Complex.I  -- ℂ represents complex numbers in Lean and Complex.I is the imaginary unit.

-- Step 2: Define the complex number z.
noncomputable def z : ℂ := (4 - 3 * i) / i

-- Step 3: State the theorem.
theorem imaginary_part_of_z : Complex.im z = -4 :=
by 
  sorry

end imaginary_part_of_z_l243_24316


namespace range_of_m_l243_24369

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem range_of_m (m : ℝ) (x : ℝ) (h1 : x ∈ Set.Icc (-1 : ℝ) 2) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ 2 < m := 
by 
  sorry

end range_of_m_l243_24369


namespace oil_bill_additional_amount_l243_24308

variables (F JanuaryBill : ℝ) (x : ℝ)

-- Given conditions
def condition1 : Prop := F / JanuaryBill = 5 / 4
def condition2 : Prop := (F + x) / JanuaryBill = 3 / 2
def JanuaryBillVal : Prop := JanuaryBill = 180

-- The theorem to prove
theorem oil_bill_additional_amount
  (h1 : condition1 F JanuaryBill)
  (h2 : condition2 F JanuaryBill x)
  (h3 : JanuaryBillVal JanuaryBill) :
  x = 45 := 
  sorry

end oil_bill_additional_amount_l243_24308


namespace owner_overtakes_thief_l243_24355

theorem owner_overtakes_thief :
  let thief_speed_initial := 45 -- kmph
  let discovery_time := 0.5 -- hours
  let owner_speed := 50 -- kmph
  let mud_road_speed := 35 -- kmph
  let mud_road_distance := 30 -- km
  let speed_bumps_speed := 40 -- kmph
  let speed_bumps_distance := 5 -- km
  let traffic_speed := 30 -- kmph
  let head_start_distance := thief_speed_initial * discovery_time
  let mud_road_time := mud_road_distance / mud_road_speed
  let speed_bumps_time := speed_bumps_distance / speed_bumps_speed
  let total_distance_before_traffic := mud_road_distance + speed_bumps_distance
  let total_time_before_traffic := mud_road_time + speed_bumps_time
  let distance_owner_travelled := owner_speed * total_time_before_traffic
  head_start_distance + total_distance_before_traffic < distance_owner_travelled →
  discovery_time + total_time_before_traffic = 1.482 :=
by sorry


end owner_overtakes_thief_l243_24355


namespace log_expression_eq_zero_l243_24306

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_eq_zero : 2 * log_base 5 10 + log_base 5 0.25 = 0 :=
by
  sorry

end log_expression_eq_zero_l243_24306


namespace max_sum_at_11_l243_24341

noncomputable def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem max_sum_at_11 (a : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_seq a) (h_a1_gt_0 : a 0 > 0)
 (h_sum_eq : sum_seq a 13 = sum_seq a 7) : 
  ∃ n : ℕ, sum_seq a n = sum_seq a 10 + (a 10 + a 11) := sorry


end max_sum_at_11_l243_24341


namespace gross_profit_value_l243_24378

theorem gross_profit_value
  (sales_price : ℝ)
  (gross_profit_percentage : ℝ)
  (sales_price_eq : sales_price = 91)
  (gross_profit_percentage_eq : gross_profit_percentage = 1.6)
  (C : ℝ)
  (cost_eqn : sales_price = C + gross_profit_percentage * C) :
  gross_profit_percentage * C = 56 :=
by
  sorry

end gross_profit_value_l243_24378


namespace zora_is_shorter_by_eight_l243_24382

noncomputable def zora_height (z : ℕ) (b : ℕ) (i : ℕ) (zara : ℕ) (average_height : ℕ) : Prop :=
  i = z + 4 ∧
  zara = b ∧
  average_height = 61 ∧
  (z + i + zara + b) / 4 = average_height

theorem zora_is_shorter_by_eight (Z B : ℕ)
  (h1 : zora_height Z B (Z + 4) 64 61) : (B - Z) = 8 :=
by
  sorry

end zora_is_shorter_by_eight_l243_24382


namespace students_remaining_after_four_stops_l243_24371

theorem students_remaining_after_four_stops :
  let initial_students := 60 
  let fraction_remaining := (2 / 3 : ℚ)
  let stop1_students := initial_students * fraction_remaining
  let stop2_students := stop1_students * fraction_remaining
  let stop3_students := stop2_students * fraction_remaining
  let stop4_students := stop3_students * fraction_remaining
  stop4_students = (320 / 27 : ℚ) :=
by
  sorry

end students_remaining_after_four_stops_l243_24371


namespace number_of_students_l243_24337

theorem number_of_students (T : ℕ) (n : ℕ) (h1 : (T + 20) / n = T / n + 1 / 2) : n = 40 :=
  sorry

end number_of_students_l243_24337


namespace wood_allocation_l243_24387

theorem wood_allocation (x y : ℝ) (h1 : 50 * x * 4 = 300 * y) (h2 : x + y = 5) : x = 3 :=
by
  sorry

end wood_allocation_l243_24387


namespace sum_of_squares_of_roots_l243_24360

theorem sum_of_squares_of_roots : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 10) → (r1 * r2 = 9) → (r1 > 5 ∨ r2 > 5) → (r1^2 + r2^2 = 82) :=
by
  intros r1 r2 h1 h2 h3
  sorry

end sum_of_squares_of_roots_l243_24360


namespace average_score_l243_24370

theorem average_score (avg1 avg2 : ℕ) (matches1 matches2 : ℕ) (h_avg1 : avg1 = 60) (h_matches1 : matches1 = 10) (h_avg2 : avg2 = 70) (h_matches2 : matches2 = 15) : 
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 :=
by
  sorry

end average_score_l243_24370


namespace shortest_wire_length_l243_24368

theorem shortest_wire_length (d1 d2 : ℝ) (r1 r2 : ℝ) (t : ℝ) :
  d1 = 8 ∧ d2 = 20 ∧ r1 = 4 ∧ r2 = 10 ∧ t = 8 * Real.sqrt 10 + 17.4 * Real.pi → 
  ∃ l : ℝ, l = t :=
by 
  sorry

end shortest_wire_length_l243_24368
