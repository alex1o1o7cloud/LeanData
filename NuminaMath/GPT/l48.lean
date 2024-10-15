import Mathlib

namespace NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_of_4095_l48_4834

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def greatest_prime_divisor_of_4095 : ℕ := 13

theorem sum_of_digits_of_greatest_prime_divisor_of_4095 :
  sum_of_digits greatest_prime_divisor_of_4095 = 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_of_4095_l48_4834


namespace NUMINAMATH_GPT_quadrilateral_side_squares_inequality_l48_4867

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end NUMINAMATH_GPT_quadrilateral_side_squares_inequality_l48_4867


namespace NUMINAMATH_GPT_vertex_of_parabola_l48_4849

theorem vertex_of_parabola (c d : ℝ) (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  ∃ v : ℝ × ℝ, v = (1, 25) :=
sorry

end NUMINAMATH_GPT_vertex_of_parabola_l48_4849


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l48_4852

theorem inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ (-2 < k ∧ k < 6) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l48_4852


namespace NUMINAMATH_GPT_remainder_abc_mod9_l48_4860

open Nat

-- Define the conditions for the problem
variables (a b c : ℕ)

-- Assume conditions: a, b, c are non-negative and less than 9, and the given congruences
theorem remainder_abc_mod9 (h1 : a < 9) (h2 : b < 9) (h3 : c < 9)
  (h4 : (a + 3 * b + 2 * c) % 9 = 3)
  (h5 : (2 * a + 2 * b + 3 * c) % 9 = 6)
  (h6 : (3 * a + b + 2 * c) % 9 = 1) :
  (a * b * c) % 9 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_abc_mod9_l48_4860


namespace NUMINAMATH_GPT_ratio_of_chris_to_amy_l48_4843

-- Definitions based on the conditions in the problem
def combined_age (Amy_age Jeremy_age Chris_age : ℕ) : Prop :=
  Amy_age + Jeremy_age + Chris_age = 132

def amy_is_one_third_jeremy (Amy_age Jeremy_age : ℕ) : Prop :=
  Amy_age = Jeremy_age / 3

def jeremy_age : ℕ := 66

-- The main theorem we need to prove
theorem ratio_of_chris_to_amy (Amy_age Chris_age : ℕ) (h1 : combined_age Amy_age jeremy_age Chris_age)
  (h2 : amy_is_one_third_jeremy Amy_age jeremy_age) : Chris_age / Amy_age = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_chris_to_amy_l48_4843


namespace NUMINAMATH_GPT_total_delegates_l48_4878

theorem total_delegates 
  (D: ℕ) 
  (h1: 16 ≤ D)
  (h2: (D - 16) % 2 = 0)
  (h3: 10 ≤ D - 16) : D = 36 := 
sorry

end NUMINAMATH_GPT_total_delegates_l48_4878


namespace NUMINAMATH_GPT_circle_center_and_radius_l48_4893

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  (exists (h k r : ℝ), (h, k) = (2, 0) ∧ r = 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l48_4893


namespace NUMINAMATH_GPT_find_x_cube_plus_reciprocal_cube_l48_4825

variable {x : ℝ}

theorem find_x_cube_plus_reciprocal_cube (hx : x + 1/x = 10) : x^3 + 1/x^3 = 970 :=
sorry

end NUMINAMATH_GPT_find_x_cube_plus_reciprocal_cube_l48_4825


namespace NUMINAMATH_GPT_salt_concentration_l48_4871

theorem salt_concentration (volume_water volume_solution concentration_solution : ℝ)
  (h1 : volume_water = 1)
  (h2 : volume_solution = 0.5)
  (h3 : concentration_solution = 0.45) :
  (volume_solution * concentration_solution) / (volume_water + volume_solution) = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_salt_concentration_l48_4871


namespace NUMINAMATH_GPT_four_consecutive_integers_product_plus_one_is_square_l48_4845

theorem four_consecutive_integers_product_plus_one_is_square (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n^2 + n - 1)^2 := by
  sorry

end NUMINAMATH_GPT_four_consecutive_integers_product_plus_one_is_square_l48_4845


namespace NUMINAMATH_GPT_cheryl_material_usage_l48_4883

theorem cheryl_material_usage:
  let bought := (3 / 8) + (1 / 3)
  let left := (15 / 40)
  let used := bought - left
  used = (1 / 3) := 
by
  sorry

end NUMINAMATH_GPT_cheryl_material_usage_l48_4883


namespace NUMINAMATH_GPT_coin_flip_sequences_l48_4862

theorem coin_flip_sequences : 
  let flips := 10
  let choices := 2
  let total_sequences := choices ^ flips
  total_sequences = 1024 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_l48_4862


namespace NUMINAMATH_GPT_zach_babysitting_hours_l48_4812

theorem zach_babysitting_hours :
  ∀ (bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed : ℕ),
    bike_cost = 100 →
    weekly_allowance = 5 →
    mowing_pay = 10 →
    babysitting_rate = 7 →
    saved_amount = 65 →
    needed_additional_amount = 6 →
    saved_amount + weekly_allowance + mowing_pay + hours_needed * babysitting_rate = bike_cost - needed_additional_amount →
    hours_needed = 2 :=
by
  intros bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_zach_babysitting_hours_l48_4812


namespace NUMINAMATH_GPT_polynomial_has_root_of_multiplicity_2_l48_4890

theorem polynomial_has_root_of_multiplicity_2 (r s k : ℝ)
  (h1 : x^3 + k * x - 128 = (x - r)^2 * (x - s)) -- polynomial has a root of multiplicity 2
  (h2 : -2 * r - s = 0)                         -- relationship from coefficient of x²
  (h3 : r^2 + 2 * r * s = k)                    -- relationship from coefficient of x
  (h4 : r^2 * s = 128)                          -- relationship from constant term
  : k = -48 := 
sorry

end NUMINAMATH_GPT_polynomial_has_root_of_multiplicity_2_l48_4890


namespace NUMINAMATH_GPT_total_tickets_sold_l48_4814

theorem total_tickets_sold
  (advanced_ticket_cost : ℕ)
  (door_ticket_cost : ℕ)
  (total_collected : ℕ)
  (advanced_tickets_sold : ℕ)
  (door_tickets_sold : ℕ) :
  advanced_ticket_cost = 8 →
  door_ticket_cost = 14 →
  total_collected = 1720 →
  advanced_tickets_sold = 100 →
  total_collected = (advanced_tickets_sold * advanced_ticket_cost) + (door_tickets_sold * door_ticket_cost) →
  100 + door_tickets_sold = 165 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l48_4814


namespace NUMINAMATH_GPT_polynomial_solution_l48_4889

theorem polynomial_solution (P : Polynomial ℝ) (h1 : P.eval 0 = 0) (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  ∀ x : ℝ, P.eval x = x :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l48_4889


namespace NUMINAMATH_GPT_percentage_of_loss_is_15_percent_l48_4837

/-- 
Given:
  SP₁ = 168 -- Selling price when gaining 20%
  Gain = 20% 
  SP₂ = 119 -- Selling price when calculating loss

Prove:
  The percentage of loss when the article is sold for Rs. 119 is 15%
--/

noncomputable def percentage_loss (CP SP₂: ℝ) : ℝ :=
  ((CP - SP₂) / CP) * 100

theorem percentage_of_loss_is_15_percent (CP SP₂ SP₁: ℝ) (Gain: ℝ):
  CP = 140 ∧ SP₁ = 168 ∧ SP₂ = 119 ∧ Gain = 20 → percentage_loss CP SP₂ = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_of_loss_is_15_percent_l48_4837


namespace NUMINAMATH_GPT_product_of_five_consecutive_not_square_l48_4804

theorem product_of_five_consecutive_not_square (n : ℤ) :
  ¬ ∃ k : ℤ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2) :=
by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_not_square_l48_4804


namespace NUMINAMATH_GPT_moles_of_Cu_CN_2_is_1_l48_4820

def moles_of_HCN : Nat := 2
def moles_of_CuSO4 : Nat := 1
def moles_of_Cu_CN_2_formed (hcn : Nat) (cuso4 : Nat) : Nat :=
  if hcn = 2 ∧ cuso4 = 1 then 1 else 0

theorem moles_of_Cu_CN_2_is_1 : moles_of_Cu_CN_2_formed moles_of_HCN moles_of_CuSO4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_Cu_CN_2_is_1_l48_4820


namespace NUMINAMATH_GPT_proof_theorem_l48_4810

noncomputable def proof_problem (y1 y2 y3 y4 y5 : ℝ) :=
  y1 + 8*y2 + 27*y3 + 64*y4 + 125*y5 = 7 ∧
  8*y1 + 27*y2 + 64*y3 + 125*y4 + 216*y5 = 100 ∧
  27*y1 + 64*y2 + 125*y3 + 216*y4 + 343*y5 = 1000 →
  64*y1 + 125*y2 + 216*y3 + 343*y4 + 512*y5 = -5999

theorem proof_theorem : ∀ (y1 y2 y3 y4 y5 : ℝ), proof_problem y1 y2 y3 y4 y5 :=
  by intros y1 y2 y3 y4 y5
     unfold proof_problem
     intro h
     sorry

end NUMINAMATH_GPT_proof_theorem_l48_4810


namespace NUMINAMATH_GPT_min_possible_value_of_coefficient_x_l48_4827

theorem min_possible_value_of_coefficient_x 
  (c d : ℤ) 
  (h1 : c * d = 15) 
  (h2 : ∃ (C : ℤ), C = c + d) 
  (h3 : c ≠ d ∧ c ≠ 34 ∧ d ≠ 34) :
  (∃ (C : ℤ), C = c + d ∧ C = 34) :=
sorry

end NUMINAMATH_GPT_min_possible_value_of_coefficient_x_l48_4827


namespace NUMINAMATH_GPT_gcd_13924_32451_eq_one_l48_4803

-- Define the two given integers.
def x : ℕ := 13924
def y : ℕ := 32451

-- State and prove that the greatest common divisor of x and y is 1.
theorem gcd_13924_32451_eq_one : Nat.gcd x y = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_13924_32451_eq_one_l48_4803


namespace NUMINAMATH_GPT_petya_friends_l48_4826

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_petya_friends_l48_4826


namespace NUMINAMATH_GPT_boat_stream_speed_l48_4824

theorem boat_stream_speed :
  ∀ (v : ℝ), (∀ (downstream_speed boat_speed : ℝ), boat_speed = 22 ∧ downstream_speed = 54/2 ∧ downstream_speed = boat_speed + v) -> v = 5 :=
by
  sorry

end NUMINAMATH_GPT_boat_stream_speed_l48_4824


namespace NUMINAMATH_GPT_area_of_rotated_squares_l48_4882

noncomputable def side_length : ℝ := 8
noncomputable def rotation_middle : ℝ := 45
noncomputable def rotation_top : ℝ := 75

-- Theorem: The area of the resulting 24-sided polygon.
theorem area_of_rotated_squares :
  (∃ (polygon_area : ℝ), polygon_area = 96) :=
sorry

end NUMINAMATH_GPT_area_of_rotated_squares_l48_4882


namespace NUMINAMATH_GPT_algebraic_expression_domain_l48_4897

theorem algebraic_expression_domain (x : ℝ) : 
  (x + 2 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 3) := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_domain_l48_4897


namespace NUMINAMATH_GPT_greatest_integer_jo_thinking_of_l48_4831

theorem greatest_integer_jo_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (∃ k : ℕ, n = 9 * k - 1) ∧ (∃ m : ℕ, n = 5 * m - 2) ∧ n = 143 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_jo_thinking_of_l48_4831


namespace NUMINAMATH_GPT_range_of_m_l48_4830

theorem range_of_m (x m : ℝ) (h₁ : x^2 - 3 * x + 2 > 0) (h₂ : ¬(x^2 - 3 * x + 2 > 0) → x < m) : 2 < m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l48_4830


namespace NUMINAMATH_GPT_sum_of_interior_angles_n_plus_2_l48_4800

-- Define the sum of the interior angles formula for a convex polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the degree measure of the sum of the interior angles of a convex polygon with n sides being 1800
def sum_of_n_sides_is_1800 (n : ℕ) : Prop := sum_of_interior_angles n = 1800

-- Translate the proof problem as a theorem statement in Lean
theorem sum_of_interior_angles_n_plus_2 (n : ℕ) (h: sum_of_n_sides_is_1800 n) : 
  sum_of_interior_angles (n + 2) = 2160 :=
sorry

end NUMINAMATH_GPT_sum_of_interior_angles_n_plus_2_l48_4800


namespace NUMINAMATH_GPT_cos_sum_to_9_l48_4842

open Real

theorem cos_sum_to_9 {x y z : ℝ} (h1 : cos x + cos y + cos z = 3) (h2 : sin x + sin y + sin z = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 9 := 
sorry

end NUMINAMATH_GPT_cos_sum_to_9_l48_4842


namespace NUMINAMATH_GPT_cricket_team_members_l48_4861

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℚ) (wk_keeper_age : ℚ) 
  (avg_whole_team : ℚ) (avg_remaining_players : ℚ)
  (h1 : captain_age = 25)
  (h2 : wk_keeper_age = 28)
  (h3 : avg_whole_team = 22)
  (h4 : avg_remaining_players = 21)
  (h5 : 22 * n = 25 + 28 + 21 * (n - 2)) :
  n = 11 :=
by sorry

end NUMINAMATH_GPT_cricket_team_members_l48_4861


namespace NUMINAMATH_GPT_calculate_m_squared_l48_4891

-- Define the conditions
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def num_slices := 4

-- Define the question
def longest_segment_length_in_piece := 2 * pizza_radius
def m := longest_segment_length_in_piece -- Length of the longest line segment in one piece

-- Rewrite the math proof problem
theorem calculate_m_squared :
  m^2 = 256 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calculate_m_squared_l48_4891


namespace NUMINAMATH_GPT_complement_union_eq_l48_4876

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_eq :
  U \ (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l48_4876


namespace NUMINAMATH_GPT_find_January_salary_l48_4855

-- Definitions and conditions
variables (J F M A May : ℝ)
def avg_Jan_to_Apr : Prop := (J + F + M + A) / 4 = 8000
def avg_Feb_to_May : Prop := (F + M + A + May) / 4 = 8300
def May_salary : Prop := May = 6500

-- Theorem statement
theorem find_January_salary (h1 : avg_Jan_to_Apr J F M A) 
                            (h2 : avg_Feb_to_May F M A May) 
                            (h3 : May_salary May) : 
                            J = 5300 :=
sorry

end NUMINAMATH_GPT_find_January_salary_l48_4855


namespace NUMINAMATH_GPT_girls_left_class_l48_4868

variable (G B G₂ B₁ : Nat)

theorem girls_left_class (h₁ : 5 * B = 6 * G) 
                         (h₂ : B = 120)
                         (h₃ : 2 * B₁ = 3 * G₂)
                         (h₄ : B₁ = B) : 
                         G - G₂ = 20 :=
by
  sorry

end NUMINAMATH_GPT_girls_left_class_l48_4868


namespace NUMINAMATH_GPT_count_primes_with_squares_in_range_l48_4879

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end NUMINAMATH_GPT_count_primes_with_squares_in_range_l48_4879


namespace NUMINAMATH_GPT_initial_spiders_correct_l48_4866

-- Define the initial number of each type of animal
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5

-- Conditions about the changes in the number of animals
def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

-- Number of animals left in the store
def total_animals_left : Nat := 25

-- Define the remaining animals after sales and adoptions
def remaining_birds : Nat := initial_birds - birds_sold
def remaining_puppies : Nat := initial_puppies - puppies_adopted
def remaining_cats : Nat := initial_cats

-- Define the remaining animals excluding spiders
def animals_without_spiders : Nat := remaining_birds + remaining_puppies + remaining_cats

-- Define the number of remaining spiders
def remaining_spiders : Nat := total_animals_left - animals_without_spiders

-- Prove the initial number of spiders
def initial_spiders : Nat := remaining_spiders + spiders_loose

theorem initial_spiders_correct :
  initial_spiders = 15 := by 
  sorry

end NUMINAMATH_GPT_initial_spiders_correct_l48_4866


namespace NUMINAMATH_GPT_t_is_perfect_square_l48_4822

variable (n : ℕ) (hpos : 0 < n)
variable (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2))

theorem t_is_perfect_square (n : ℕ) (hpos : 0 < n) (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2)) : 
  ∃ k : ℕ, t = k * k := 
sorry

end NUMINAMATH_GPT_t_is_perfect_square_l48_4822


namespace NUMINAMATH_GPT_find_two_numbers_l48_4821

theorem find_two_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 5) (harmonic_mean : 2 * a * b / (a + b) = 5 / 3) :
  (a = (15 + Real.sqrt 145) / 4 ∧ b = (15 - Real.sqrt 145) / 4) ∨
  (a = (15 - Real.sqrt 145) / 4 ∧ b = (15 + Real.sqrt 145) / 4) :=
by
  sorry

end NUMINAMATH_GPT_find_two_numbers_l48_4821


namespace NUMINAMATH_GPT_jelly_bean_problem_l48_4872

variable (b c : ℕ)

theorem jelly_bean_problem (h1 : b = 3 * c) (h2 : b - 15 = 4 * (c - 15)) : b = 135 :=
sorry

end NUMINAMATH_GPT_jelly_bean_problem_l48_4872


namespace NUMINAMATH_GPT_real_roots_approx_correct_to_4_decimal_places_l48_4851

noncomputable def f (x : ℝ) : ℝ := x^4 - (2 * 10^10 + 1) * x^2 - x + 10^20 + 10^10 - 1

theorem real_roots_approx_correct_to_4_decimal_places :
  ∃ x1 x2 : ℝ, 
  abs (x1 - 99999.9997) ≤ 0.0001 ∧ 
  abs (x2 - 100000.0003) ≤ 0.0001 ∧ 
  f x1 = 0 ∧ 
  f x2 = 0 :=
sorry

end NUMINAMATH_GPT_real_roots_approx_correct_to_4_decimal_places_l48_4851


namespace NUMINAMATH_GPT_find_AB_value_l48_4828

theorem find_AB_value :
  ∃ A B : ℕ, (A + B = 5 ∧ (A - B) % 11 = 5 % 11) ∧
           990 * 991 * 992 * 993 = 966428 * 100000 + A * 9100 + B * 40 :=
sorry

end NUMINAMATH_GPT_find_AB_value_l48_4828


namespace NUMINAMATH_GPT_pairs_satisfying_int_l48_4809

theorem pairs_satisfying_int (a b : ℕ) :
  ∃ n : ℕ, a = 2 * n^2 + 1 ∧ b = n ↔ (2 * a * b^2 + 1) ∣ (a^3 + 1) := by
  sorry

end NUMINAMATH_GPT_pairs_satisfying_int_l48_4809


namespace NUMINAMATH_GPT_no_three_digit_number_exists_l48_4829

theorem no_three_digit_number_exists (a b c : ℕ) (h₁ : 0 ≤ a ∧ a < 10) (h₂ : 0 ≤ b ∧ b < 10) (h₃ : 0 ≤ c ∧ c < 10) (h₄ : a ≠ 0) :
  ¬ ∃ k : ℕ, k^2 = 99 * (a - c) :=
by
  sorry

end NUMINAMATH_GPT_no_three_digit_number_exists_l48_4829


namespace NUMINAMATH_GPT_meaningful_domain_of_function_l48_4813

theorem meaningful_domain_of_function : ∀ x : ℝ, (∃ y : ℝ, y = 3 / Real.sqrt (x - 2)) → x > 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_meaningful_domain_of_function_l48_4813


namespace NUMINAMATH_GPT_part_a_impossibility_l48_4815

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end NUMINAMATH_GPT_part_a_impossibility_l48_4815


namespace NUMINAMATH_GPT_ions_electron_shell_structure_l48_4853

theorem ions_electron_shell_structure
  (a b n m : ℤ) 
  (same_electron_shell_structure : a + n = b - m) :
  a + m = b - n :=
by
  sorry

end NUMINAMATH_GPT_ions_electron_shell_structure_l48_4853


namespace NUMINAMATH_GPT_find_rate_of_new_machine_l48_4898

noncomputable def rate_of_new_machine (R : ℝ) : Prop :=
  let old_rate := 100
  let total_bolts := 350
  let time_in_hours := 84 / 60
  let bolts_by_old_machine := old_rate * time_in_hours
  let bolts_by_new_machine := total_bolts - bolts_by_old_machine
  R = bolts_by_new_machine / time_in_hours

theorem find_rate_of_new_machine : rate_of_new_machine 150 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_of_new_machine_l48_4898


namespace NUMINAMATH_GPT_backyard_area_l48_4895

-- Definitions from conditions
def length : ℕ := 1000 / 25
def perimeter : ℕ := 1000 / 10
def width : ℕ := (perimeter - 2 * length) / 2

-- Theorem statement: Given the conditions, the area of the backyard is 400 square meters
theorem backyard_area : length * width = 400 :=
by 
  -- Sorry to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_backyard_area_l48_4895


namespace NUMINAMATH_GPT_inequality_triangle_areas_l48_4807

theorem inequality_triangle_areas (a b c α β γ : ℝ) (hα : α = 2 * Real.sqrt (b * c)) (hβ : β = 2 * Real.sqrt (a * c)) (hγ : γ = 2 * Real.sqrt (a * b)) : 
  a / α + b / β + c / γ ≥ 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_triangle_areas_l48_4807


namespace NUMINAMATH_GPT_remaining_statue_weight_l48_4874

theorem remaining_statue_weight (w_initial w1 w2 w_discarded w_remaining : ℕ) 
    (h_initial : w_initial = 80)
    (h_w1 : w1 = 10)
    (h_w2 : w2 = 18)
    (h_discarded : w_discarded = 22) :
    2 * w_remaining = w_initial - w_discarded - w1 - w2 :=
by
  sorry

end NUMINAMATH_GPT_remaining_statue_weight_l48_4874


namespace NUMINAMATH_GPT_smallest_number_of_students_l48_4844

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end NUMINAMATH_GPT_smallest_number_of_students_l48_4844


namespace NUMINAMATH_GPT_division_problem_l48_4840

theorem division_problem (x y n : ℕ) 
  (h1 : x = n * y + 4) 
  (h2 : 2 * x = 14 * y + 1) 
  (h3 : 5 * y - x = 3) : n = 4 := 
sorry

end NUMINAMATH_GPT_division_problem_l48_4840


namespace NUMINAMATH_GPT_tan_15_degree_l48_4877

theorem tan_15_degree : 
  let a := 45 * (Real.pi / 180)
  let b := 30 * (Real.pi / 180)
  Real.tan (a - b) = 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_15_degree_l48_4877


namespace NUMINAMATH_GPT_largest_integer_satisfying_l48_4887

theorem largest_integer_satisfying (x : ℤ) : 
  (∃ x, (2/7 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < 3/4) → x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_l48_4887


namespace NUMINAMATH_GPT_interval_of_x_l48_4841

theorem interval_of_x (x : ℝ) (h : x = ((-x)^2 / x) + 3) : 3 < x ∧ x ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_interval_of_x_l48_4841


namespace NUMINAMATH_GPT_inequality_sol_set_a_eq_2_inequality_sol_set_general_l48_4801

theorem inequality_sol_set_a_eq_2 :
  ∀ x : ℝ, (x^2 - x + 2 - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem inequality_sol_set_general (a : ℝ) :
  (∀ x : ℝ, (x^2 - x + a - a^2 ≤ 0) ↔
    (if a < 1/2 then a ≤ x ∧ x ≤ 1 - a
    else if a > 1/2 then 1 - a ≤ x ∧ x ≤ a
    else x = 1/2)) :=
by sorry

end NUMINAMATH_GPT_inequality_sol_set_a_eq_2_inequality_sol_set_general_l48_4801


namespace NUMINAMATH_GPT_inequality_am_gm_l48_4875

variable (a b x y : ℝ)

theorem inequality_am_gm (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  (a^2 / x) + (b^2 / y) ≥ (a + b)^2 / (x + y) :=
by {
  -- proof will be filled here
  sorry
}

end NUMINAMATH_GPT_inequality_am_gm_l48_4875


namespace NUMINAMATH_GPT_value_of_expression_l48_4863

theorem value_of_expression (x : ℕ) (h : x = 2) : x + x * x^x = 10 := by
  rw [h] -- Substituting x = 2
  sorry

end NUMINAMATH_GPT_value_of_expression_l48_4863


namespace NUMINAMATH_GPT_distance_A_to_focus_l48_4885

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  ((b^2 - 4*a*c) / (4*a), 0)

theorem distance_A_to_focus 
  (P : ℝ × ℝ) (parabola : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hP : P = (-2, 0))
  (hPar : ∀ x y, parabola x y ↔ y^2 = 4 * x)
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y^2 = 4 * x → (x, y) = A ∨ (x, y) = B)
  (hDist : dist P A = (1 / 2) * dist A B)
  (hFocus : focus_of_parabola 1 0 (-1) = (1, 0)) :
  dist A (1, 0) = 5 / 3 :=
sorry

end NUMINAMATH_GPT_distance_A_to_focus_l48_4885


namespace NUMINAMATH_GPT_number_of_girls_in_colins_class_l48_4850

variables (g b : ℕ)

theorem number_of_girls_in_colins_class
  (h1 : g / b = 3 / 4)
  (h2 : g + b = 35)
  (h3 : b > 15) :
  g = 15 :=
sorry

end NUMINAMATH_GPT_number_of_girls_in_colins_class_l48_4850


namespace NUMINAMATH_GPT_pamphlet_cost_l48_4805

theorem pamphlet_cost (p : ℝ) 
  (h1 : 9 * p < 10)
  (h2 : 10 * p > 11) : p = 1.11 :=
sorry

end NUMINAMATH_GPT_pamphlet_cost_l48_4805


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l48_4818

theorem eccentricity_of_ellipse {a b c e : ℝ} 
  (h1 : b^2 = 3) 
  (h2 : c = 1 / 4)
  (h3 : a^2 = b^2 + c^2)
  (h4 : a = 7 / 4) 
  : e = c / a → e = 1 / 7 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l48_4818


namespace NUMINAMATH_GPT_no_play_students_count_l48_4888

theorem no_play_students_count :
  let total_students := 420
  let football_players := 325
  let cricket_players := 175
  let both_players := 130
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end NUMINAMATH_GPT_no_play_students_count_l48_4888


namespace NUMINAMATH_GPT_possible_values_of_k_l48_4899

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, k = 2 ^ t ∧ 2 ^ t ≥ n :=
sorry

end NUMINAMATH_GPT_possible_values_of_k_l48_4899


namespace NUMINAMATH_GPT_apple_cost_l48_4819

theorem apple_cost (l q : ℕ)
  (h1 : 30 * l + 6 * q = 366)
  (h2 : 15 * l = 150)
  (h3 : 30 * l + (333 - 30 * l) / q * q = 333) :
  30 + (333 - 30 * l) / q = 33 := 
sorry

end NUMINAMATH_GPT_apple_cost_l48_4819


namespace NUMINAMATH_GPT_probability_same_color_is_one_third_l48_4838

-- Define a type for colors
inductive Color 
| red 
| white 
| blue 

open Color

-- Define the function to calculate the probability of the same color selection
def sameColorProbability : ℚ :=
  let total_outcomes := 3 * 3
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

-- Theorem stating that the probability is 1/3
theorem probability_same_color_is_one_third : sameColorProbability = 1 / 3 :=
by
  -- Steps of proof will be provided here
  sorry

end NUMINAMATH_GPT_probability_same_color_is_one_third_l48_4838


namespace NUMINAMATH_GPT_first_six_divisors_l48_4847

theorem first_six_divisors (a b : ℤ) (h : 5 * b = 14 - 3 * a) : 
  ∃ n, n = 5 ∧ ∀ k ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), (3 * b + 18) % k = 0 ↔ k ∈ ({1, 2, 3, 5, 6} : Finset ℕ) :=
by
  sorry

end NUMINAMATH_GPT_first_six_divisors_l48_4847


namespace NUMINAMATH_GPT_solve_base7_addition_problem_l48_4880

noncomputable def base7_addition_problem : Prop :=
  ∃ (X Y: ℕ), 
    (5 * 7^2 + X * 7 + Y) + (3 * 7^1 + 2) = 6 * 7^2 + 2 * 7 + X ∧
    X + Y = 10 

theorem solve_base7_addition_problem : base7_addition_problem :=
by sorry

end NUMINAMATH_GPT_solve_base7_addition_problem_l48_4880


namespace NUMINAMATH_GPT_problem_graph_empty_l48_4881

open Real

theorem problem_graph_empty : ∀ x y : ℝ, ¬ (x^2 + 3 * y^2 - 4 * x - 12 * y + 28 = 0) :=
by
  intro x y
  -- Apply the contradiction argument based on the conditions given
  sorry


end NUMINAMATH_GPT_problem_graph_empty_l48_4881


namespace NUMINAMATH_GPT_shadow_building_length_l48_4870

-- Define the basic parameters
def height_flagpole : ℕ := 18
def shadow_flagpole : ℕ := 45
def height_building : ℕ := 20

-- Define the condition on similar conditions
def similar_conditions (h₁ s₁ h₂ s₂ : ℕ) : Prop :=
  h₁ * s₂ = h₂ * s₁

-- Theorem statement
theorem shadow_building_length :
  similar_conditions height_flagpole shadow_flagpole height_building 50 := 
sorry

end NUMINAMATH_GPT_shadow_building_length_l48_4870


namespace NUMINAMATH_GPT_terminal_zeros_75_480_l48_4808

theorem terminal_zeros_75_480 :
  let x := 75
  let y := 480
  let fact_x := 5^2 * 3
  let fact_y := 2^5 * 3 * 5
  let product := fact_x * fact_y
  let num_zeros := min (3) (5)
  num_zeros = 3 :=
by
  sorry

end NUMINAMATH_GPT_terminal_zeros_75_480_l48_4808


namespace NUMINAMATH_GPT_balloons_remaining_each_friend_l48_4811

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end NUMINAMATH_GPT_balloons_remaining_each_friend_l48_4811


namespace NUMINAMATH_GPT_quadratic_integers_pairs_l48_4884

theorem quadratic_integers_pairs (m n : ℕ) :
  (0 < m ∧ m < 9) ∧ (0 < n ∧ n < 9) ∧ (m^2 > 9 * n) ↔ ((m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2)) :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_quadratic_integers_pairs_l48_4884


namespace NUMINAMATH_GPT_diameter_of_larger_sphere_l48_4832

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end NUMINAMATH_GPT_diameter_of_larger_sphere_l48_4832


namespace NUMINAMATH_GPT_distinct_terms_count_l48_4896

/-!
  Proving the number of distinct terms in the expansion of (x + 2y)^12
-/

theorem distinct_terms_count (x y : ℕ) : 
  (x + 2 * y) ^ 12 = 13 :=
by sorry

end NUMINAMATH_GPT_distinct_terms_count_l48_4896


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l48_4806

theorem solve_quadratic1 (x : ℝ) :
  x^2 + 10 * x + 16 = 0 ↔ (x = -2 ∨ x = -8) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  x * (x + 4) = 8 * x + 12 ↔ (x = -2 ∨ x = 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l48_4806


namespace NUMINAMATH_GPT_solution_set_of_inequality_l48_4859

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l48_4859


namespace NUMINAMATH_GPT_henri_drove_more_miles_l48_4873

-- Defining the conditions
def Gervais_average_miles_per_day := 315
def Gervais_days_driven := 3
def Henri_total_miles := 1250

-- Total miles driven by Gervais
def Gervais_total_miles := Gervais_average_miles_per_day * Gervais_days_driven

-- The proof problem statement
theorem henri_drove_more_miles : Henri_total_miles - Gervais_total_miles = 305 := 
by 
  sorry

end NUMINAMATH_GPT_henri_drove_more_miles_l48_4873


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l48_4854

theorem sufficient_not_necessary_condition
  (x : ℝ) : 
  x^2 - 4*x - 5 > 0 → (x > 5 ∨ x < -1) ∧ (x > 5 → x^2 - 4*x - 5 > 0) ∧ ¬(x^2 - 4*x - 5 > 0 → x > 5) := 
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l48_4854


namespace NUMINAMATH_GPT_daniel_initial_noodles_l48_4817

variable (give : ℕ)
variable (left : ℕ)
variable (initial : ℕ)

theorem daniel_initial_noodles (h1 : give = 12) (h2 : left = 54) (h3 : initial = left + give) : initial = 66 := by
  sorry

end NUMINAMATH_GPT_daniel_initial_noodles_l48_4817


namespace NUMINAMATH_GPT_marbles_jack_gave_l48_4858

-- Definitions based on conditions
def initial_marbles : ℕ := 22
def final_marbles : ℕ := 42

-- Theorem stating that the difference between final and initial marbles Josh collected is the marbles Jack gave
theorem marbles_jack_gave :
  final_marbles - initial_marbles = 20 :=
  sorry

end NUMINAMATH_GPT_marbles_jack_gave_l48_4858


namespace NUMINAMATH_GPT_perfect_square_expression_l48_4839

theorem perfect_square_expression (n : ℕ) (h : 7 ≤ n) : ∃ k : ℤ, (n + 2) ^ 2 = k ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l48_4839


namespace NUMINAMATH_GPT_sally_picked_3_plums_l48_4869

theorem sally_picked_3_plums (melanie_picked : ℕ) (dan_picked : ℕ) (total_picked : ℕ) 
    (h1 : melanie_picked = 4) (h2 : dan_picked = 9) (h3 : total_picked = 16) : 
    total_picked - (melanie_picked + dan_picked) = 3 := 
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_sally_picked_3_plums_l48_4869


namespace NUMINAMATH_GPT_molecular_weight_2N_5O_l48_4848

def molecular_weight (num_N num_O : ℕ) (atomic_weight_N atomic_weight_O : ℝ) : ℝ :=
  (num_N * atomic_weight_N) + (num_O * atomic_weight_O)

theorem molecular_weight_2N_5O :
  molecular_weight 2 5 14.01 16.00 = 108.02 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_molecular_weight_2N_5O_l48_4848


namespace NUMINAMATH_GPT_pow_calculation_l48_4846

-- We assume a is a non-zero real number or just a variable
variable (a : ℝ)

theorem pow_calculation : (2 * a^2)^3 = 8 * a^6 := 
by
  sorry

end NUMINAMATH_GPT_pow_calculation_l48_4846


namespace NUMINAMATH_GPT_ratio_sum_eq_l48_4816

variable {x y z : ℝ}

-- Conditions: 3x, 4y, 5z form a geometric sequence
def geom_sequence (x y z : ℝ) : Prop :=
  (∃ r : ℝ, 4 * y = 3 * x * r ∧ 5 * z = 4 * y * r)

-- Conditions: 1/x, 1/y, 1/z form an arithmetic sequence
def arith_sequence (x y z : ℝ) : Prop :=
  2 * x * z = y * z + x * y

-- Conclude: x/z + z/x = 34/15
theorem ratio_sum_eq (h1 : geom_sequence x y z) (h2 : arith_sequence x y z) : 
  (x / z + z / x) = (34 / 15) :=
sorry

end NUMINAMATH_GPT_ratio_sum_eq_l48_4816


namespace NUMINAMATH_GPT_eggs_used_afternoon_l48_4802

theorem eggs_used_afternoon (eggs_pumpkin eggs_apple eggs_cherry eggs_total : ℕ)
  (h_pumpkin : eggs_pumpkin = 816)
  (h_apple : eggs_apple = 384)
  (h_cherry : eggs_cherry = 120)
  (h_total : eggs_total = 1820) :
  eggs_total - (eggs_pumpkin + eggs_apple + eggs_cherry) = 500 :=
by
  sorry

end NUMINAMATH_GPT_eggs_used_afternoon_l48_4802


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l48_4886

theorem sum_of_squares_of_roots (x1 x2 : ℝ) (h1 : 2 * x1^2 + 5 * x1 - 12 = 0) (h2 : 2 * x2^2 + 5 * x2 - 12 = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 = 73 / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l48_4886


namespace NUMINAMATH_GPT_num_classes_received_basketballs_l48_4857

theorem num_classes_received_basketballs (total_basketballs left_basketballs : ℕ) 
  (h : total_basketballs = 54) (h_left : left_basketballs = 5) : 
  (total_basketballs - left_basketballs) / 7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_classes_received_basketballs_l48_4857


namespace NUMINAMATH_GPT_exists_divisible_by_2021_l48_4835

def concatenated_number (n m : ℕ) : ℕ := 
  -- This function should concatenate the digits from n to m inclusively
  sorry

theorem exists_divisible_by_2021 : ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concatenated_number n m :=
by
  sorry

end NUMINAMATH_GPT_exists_divisible_by_2021_l48_4835


namespace NUMINAMATH_GPT_calculate_integral_cos8_l48_4864

noncomputable def integral_cos8 : ℝ :=
  ∫ x in (Real.pi / 2)..(2 * Real.pi), 2^8 * (Real.cos x)^8

theorem calculate_integral_cos8 :
  integral_cos8 = 219 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_calculate_integral_cos8_l48_4864


namespace NUMINAMATH_GPT_find_principal_l48_4894

noncomputable def compoundPrincipal (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem find_principal :
  let A := 3969
  let r := 0.05
  let n := 1
  let t := 2
  compoundPrincipal A r n t = 3600 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l48_4894


namespace NUMINAMATH_GPT_days_passed_before_cows_ran_away_l48_4892

def initial_cows := 1000
def initial_days := 50
def cows_left := 800
def cows_run_away := initial_cows - cows_left
def total_food := initial_cows * initial_days
def remaining_food (x : ℕ) := total_food - initial_cows * x
def food_needed := cows_left * initial_days

theorem days_passed_before_cows_ran_away (x : ℕ) :
  (remaining_food x = food_needed) → (x = 10) :=
by
  sorry

end NUMINAMATH_GPT_days_passed_before_cows_ran_away_l48_4892


namespace NUMINAMATH_GPT_ava_legs_count_l48_4856

-- Conditions:
-- There are a total of 9 animals in the farm.
-- There are only chickens and buffalos in the farm.
-- There are 5 chickens in the farm.

def total_animals : Nat := 9
def num_chickens : Nat := 5
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

-- Proof statement: Ava counted 26 legs.
theorem ava_legs_count (num_buffalos : Nat) 
  (H1 : total_animals = num_chickens + num_buffalos) : 
  num_chickens * legs_per_chicken + num_buffalos * legs_per_buffalo = 26 :=
by 
  have H2 : num_buffalos = total_animals - num_chickens := by sorry
  sorry

end NUMINAMATH_GPT_ava_legs_count_l48_4856


namespace NUMINAMATH_GPT_kiera_total_envelopes_l48_4836

-- Define the number of blue envelopes
def blue_envelopes : ℕ := 14

-- Define the number of yellow envelopes as 6 fewer than the number of blue envelopes
def yellow_envelopes : ℕ := blue_envelopes - 6

-- Define the number of green envelopes as 3 times the number of yellow envelopes
def green_envelopes : ℕ := 3 * yellow_envelopes

-- The total number of envelopes is the sum of blue, yellow, and green envelopes
def total_envelopes : ℕ := blue_envelopes + yellow_envelopes + green_envelopes

-- Prove that the total number of envelopes is 46
theorem kiera_total_envelopes : total_envelopes = 46 := by
  sorry

end NUMINAMATH_GPT_kiera_total_envelopes_l48_4836


namespace NUMINAMATH_GPT_count_divisible_by_8_l48_4823

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end NUMINAMATH_GPT_count_divisible_by_8_l48_4823


namespace NUMINAMATH_GPT_largest_b_l48_4865

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end NUMINAMATH_GPT_largest_b_l48_4865


namespace NUMINAMATH_GPT_total_population_milburg_l48_4833

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end NUMINAMATH_GPT_total_population_milburg_l48_4833
