import Mathlib

namespace NUMINAMATH_GPT_convex_quadrilateral_inequality_l2334_233433

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

theorem convex_quadrilateral_inequality
    (AB CD BC AD AC BD : ℝ)
    (h : AB * CD + BC * AD >= AC * BD)
    (convex_quadrilateral : Prop) :
  AB * CD + BC * AD >= AC * BD :=
by
  sorry

end NUMINAMATH_GPT_convex_quadrilateral_inequality_l2334_233433


namespace NUMINAMATH_GPT_increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l2334_233402

-- Define the function z = x * y
def z (x y : ℝ) : ℝ := x * y

-- Initial point M0
def M0 : ℝ × ℝ := (1, 2)

-- Points to which we move
def M1 : ℝ × ℝ := (1.1, 2)
def M2 : ℝ × ℝ := (1, 1.9)
def M3 : ℝ × ℝ := (1.1, 2.2)

-- Proofs for the increments
theorem increment_M0_to_M1 : z M1.1 M1.2 - z M0.1 M0.2 = 0.2 :=
by sorry

theorem increment_M0_to_M2 : z M2.1 M2.2 - z M0.1 M0.2 = -0.1 :=
by sorry

theorem increment_M0_to_M3 : z M3.1 M3.2 - z M0.1 M0.2 = 0.42 :=
by sorry

end NUMINAMATH_GPT_increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l2334_233402


namespace NUMINAMATH_GPT_baseball_team_grouping_l2334_233423

theorem baseball_team_grouping (new_players returning_players : ℕ) (group_size : ℕ) 
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group : group_size = 5) : 
  (new_players + returning_players) / group_size = 2 := 
  by 
  sorry

end NUMINAMATH_GPT_baseball_team_grouping_l2334_233423


namespace NUMINAMATH_GPT_difference_largest_smallest_l2334_233442

def num1 : ℕ := 10
def num2 : ℕ := 11
def num3 : ℕ := 12

theorem difference_largest_smallest :
  (max num1 (max num2 num3)) - (min num1 (min num2 num3)) = 2 :=
by
  -- Proof can be filled here
  sorry

end NUMINAMATH_GPT_difference_largest_smallest_l2334_233442


namespace NUMINAMATH_GPT_modulus_of_z_l2334_233445

section complex_modulus
open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 + I) = 10 - 5 * I) : Complex.abs z = 5 :=
by
  sorry
end complex_modulus

end NUMINAMATH_GPT_modulus_of_z_l2334_233445


namespace NUMINAMATH_GPT_number_division_reduction_l2334_233499

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 24) : x = 36 := sorry

end NUMINAMATH_GPT_number_division_reduction_l2334_233499


namespace NUMINAMATH_GPT_largest_house_number_l2334_233461

theorem largest_house_number (phone_number_digits : List ℕ) (house_number_digits : List ℕ) :
  phone_number_digits = [5, 0, 4, 9, 3, 2, 6] →
  phone_number_digits.sum = 29 →
  (∀ (d1 d2 : ℕ), d1 ∈ house_number_digits → d2 ∈ house_number_digits → d1 ≠ d2) →
  house_number_digits.sum = 29 →
  house_number_digits = [9, 8, 7, 5] :=
by
  intros
  sorry

end NUMINAMATH_GPT_largest_house_number_l2334_233461


namespace NUMINAMATH_GPT_subset_implies_all_elements_l2334_233465

variable {U : Type}

theorem subset_implies_all_elements (P Q : Set U) (hPQ : P ⊆ Q) (hP_nonempty : P ≠ ∅) (hQ_nonempty : Q ≠ ∅) :
  ∀ x ∈ P, x ∈ Q :=
by 
  sorry

end NUMINAMATH_GPT_subset_implies_all_elements_l2334_233465


namespace NUMINAMATH_GPT_keun_bae_jumps_fourth_day_l2334_233472

def jumps (n : ℕ) : ℕ :=
  match n with
  | 0 => 15
  | n + 1 => 2 * jumps n

theorem keun_bae_jumps_fourth_day : jumps 3 = 120 :=
by
  sorry

end NUMINAMATH_GPT_keun_bae_jumps_fourth_day_l2334_233472


namespace NUMINAMATH_GPT_sufficiency_and_necessity_of_p_and_q_l2334_233435

noncomputable def p : Prop := ∀ k, k = Real.sqrt 3
noncomputable def q : Prop := ∀ k, ∃ y x, y = k * x + 2 ∧ x^2 + y^2 = 1

theorem sufficiency_and_necessity_of_p_and_q : (p → q) ∧ (¬ (q → p)) := by
  sorry

end NUMINAMATH_GPT_sufficiency_and_necessity_of_p_and_q_l2334_233435


namespace NUMINAMATH_GPT_sides_of_nth_hexagon_l2334_233474

-- Definition of the arithmetic sequence condition.
def first_term : ℕ := 6
def common_difference : ℕ := 5

-- The function representing the n-th term of the sequence.
def num_sides (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

-- Now, we state the theorem that the n-th term equals 5n + 1.
theorem sides_of_nth_hexagon (n : ℕ) : num_sides n = 5 * n + 1 := by
  sorry

end NUMINAMATH_GPT_sides_of_nth_hexagon_l2334_233474


namespace NUMINAMATH_GPT_smallest_common_multiple_l2334_233459

theorem smallest_common_multiple (n : ℕ) : 
  (2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧ 1000 ≤ n ∧ n < 10000) → n = 1008 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_common_multiple_l2334_233459


namespace NUMINAMATH_GPT_num_rectangular_arrays_with_48_chairs_l2334_233454

theorem num_rectangular_arrays_with_48_chairs : 
  ∃ n, (∀ (r c : ℕ), 2 ≤ r ∧ 2 ≤ c ∧ r * c = 48 → (n = 8 ∨ n = 0)) ∧ (n = 8) :=
by 
  sorry

end NUMINAMATH_GPT_num_rectangular_arrays_with_48_chairs_l2334_233454


namespace NUMINAMATH_GPT_bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l2334_233413

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end NUMINAMATH_GPT_bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l2334_233413


namespace NUMINAMATH_GPT_scientific_notation_of_number_l2334_233490

def number := 460000000
def scientific_notation (n : ℕ) (s : ℝ) := s * 10 ^ n

theorem scientific_notation_of_number :
  scientific_notation 8 4.6 = number :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l2334_233490


namespace NUMINAMATH_GPT_pears_to_peaches_l2334_233457

-- Define the weights of pears and peaches
variables (pear peach : ℝ) 

-- Given conditions: 9 pears weigh the same as 6 peaches
axiom weight_ratio : 9 * pear = 6 * peach

-- Theorem to prove: 36 pears weigh the same as 24 peaches
theorem pears_to_peaches (h : 9 * pear = 6 * peach) : 36 * pear = 24 * peach :=
by
  sorry

end NUMINAMATH_GPT_pears_to_peaches_l2334_233457


namespace NUMINAMATH_GPT_max_product_h_k_l2334_233432

theorem max_product_h_k {h k : ℝ → ℝ} (h_bound : ∀ x, -3 ≤ h x ∧ h x ≤ 5) (k_bound : ∀ x, -1 ≤ k x ∧ k x ≤ 4) :
  ∃ x y, h x * k y = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_product_h_k_l2334_233432


namespace NUMINAMATH_GPT_simplify_expression_l2334_233439

variable (x : ℝ)

theorem simplify_expression :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) = 2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2334_233439


namespace NUMINAMATH_GPT_gcd_poly_l2334_233456

theorem gcd_poly (k : ℕ) : Nat.gcd ((4500 * k)^2 + 11 * (4500 * k) + 40) (4500 * k + 8) = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_poly_l2334_233456


namespace NUMINAMATH_GPT_necessary_not_sufficient_l2334_233421

theorem necessary_not_sufficient (m : ℝ) (x : ℝ) (h₁ : m > 0) (h₂ : 0 < x ∧ x < m) (h₃ : x / (x - 1) < 0) 
: m = 1 / 2 := 
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l2334_233421


namespace NUMINAMATH_GPT_units_digit_17_pow_27_l2334_233476

-- Define the problem: the units digit of 17^27
theorem units_digit_17_pow_27 : (17 ^ 27) % 10 = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_17_pow_27_l2334_233476


namespace NUMINAMATH_GPT_percent_both_correct_l2334_233430

-- Definitions of the given percentages
def A : ℝ := 75
def B : ℝ := 25
def N : ℝ := 20

-- The proof problem statement
theorem percent_both_correct (A B N : ℝ) (hA : A = 75) (hB : B = 25) (hN : N = 20) : A + B - N - 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_both_correct_l2334_233430


namespace NUMINAMATH_GPT_equivalent_single_percentage_change_l2334_233489

theorem equivalent_single_percentage_change :
  let original_price : ℝ := 250
  let num_items : ℕ := 400
  let first_increase : ℝ := 0.15
  let second_increase : ℝ := 0.20
  let discount : ℝ := -0.10
  let third_increase : ℝ := 0.25

  -- Calculations
  let price_after_first_increase := original_price * (1 + first_increase)
  let price_after_second_increase := price_after_first_increase * (1 + second_increase)
  let price_after_discount := price_after_second_increase * (1 + discount)
  let final_price := price_after_discount * (1 + third_increase)

  -- Calculate percentage change
  let percentage_change := ((final_price - original_price) / original_price) * 100

  percentage_change = 55.25 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_percentage_change_l2334_233489


namespace NUMINAMATH_GPT_parabola_x_intercept_unique_l2334_233417

theorem parabola_x_intercept_unique : ∃! (x : ℝ), ∀ (y : ℝ), x = -y^2 + 2*y + 3 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercept_unique_l2334_233417


namespace NUMINAMATH_GPT_initial_books_l2334_233422

-- Define the variables and conditions
def B : ℕ := 75
def loaned_books : ℕ := 60
def returned_books : ℕ := (70 * loaned_books) / 100
def not_returned_books : ℕ := loaned_books - returned_books
def end_of_month_books : ℕ := 57

-- State the theorem
theorem initial_books (h1 : returned_books = 42)
                      (h2 : end_of_month_books = 57)
                      (h3 : loaned_books = 60) :
  B = end_of_month_books + not_returned_books :=
by sorry

end NUMINAMATH_GPT_initial_books_l2334_233422


namespace NUMINAMATH_GPT_first_rectangle_dimensions_second_rectangle_dimensions_l2334_233412

theorem first_rectangle_dimensions (x y : ℕ) (h : x * y = 2 * (x + y) + 1) : (x = 7 ∧ y = 3) ∨ (x = 3 ∧ y = 7) :=
sorry

theorem second_rectangle_dimensions (a b : ℕ) (h : a * b = 2 * (a + b) - 1) : (a = 5 ∧ b = 3) ∨ (a = 3 ∧ b = 5) :=
sorry

end NUMINAMATH_GPT_first_rectangle_dimensions_second_rectangle_dimensions_l2334_233412


namespace NUMINAMATH_GPT_geometric_progression_fourth_term_l2334_233415

theorem geometric_progression_fourth_term :
  ∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ = 2^(1/2) ∧ a₂ = 2^(1/4) ∧ a₃ = 2^(1/6) ∧ (a₂ / a₁ = r) ∧ (a₃ = a₂ * r⁻¹) ∧ (a₄ = a₃ * r) → a₄ = 2^(1/8) := by
intro a₁ a₂ a₃ a₄
intro h
sorry

end NUMINAMATH_GPT_geometric_progression_fourth_term_l2334_233415


namespace NUMINAMATH_GPT_josie_gift_money_l2334_233487

-- Define the cost of each cassette tape
def tape_cost : ℕ := 9

-- Define the number of cassette tapes Josie plans to buy
def num_tapes : ℕ := 2

-- Define the cost of the headphone set
def headphone_cost : ℕ := 25

-- Define the amount of money Josie will have left after the purchases
def money_left : ℕ := 7

-- Define the total cost of tapes
def total_tape_cost := num_tapes * tape_cost

-- Define the total cost of both tapes and headphone set
def total_cost := total_tape_cost + headphone_cost

-- The total money Josie will have would be total_cost + money_left
theorem josie_gift_money : total_cost + money_left = 50 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_josie_gift_money_l2334_233487


namespace NUMINAMATH_GPT_lateral_surface_area_of_parallelepiped_is_correct_l2334_233451

noncomputable def lateral_surface_area (diagonal : ℝ) (angle : ℝ) (base_area : ℝ) : ℝ :=
  let h := diagonal * Real.sin angle
  let s := diagonal * Real.cos angle
  let side1_sq := s ^ 2  -- represents DC^2 + AD^2
  let base_diag_sq := 25  -- already given as 25 from BD^2
  let added := side1_sq + 2 * base_area
  2 * h * Real.sqrt added

theorem lateral_surface_area_of_parallelepiped_is_correct :
  lateral_surface_area 10 (Real.pi / 3) 12 = 70 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_parallelepiped_is_correct_l2334_233451


namespace NUMINAMATH_GPT_simplify_expression_l2334_233470

theorem simplify_expression (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2334_233470


namespace NUMINAMATH_GPT_radian_measure_sector_l2334_233416

theorem radian_measure_sector (r l : ℝ) (h1 : 2 * r + l = 12) (h2 : (1 / 2) * l * r = 8) :
  l / r = 1 ∨ l / r = 4 := by
  sorry

end NUMINAMATH_GPT_radian_measure_sector_l2334_233416


namespace NUMINAMATH_GPT_usual_time_to_school_l2334_233447

-- Define the conditions
variables (R T : ℝ) (h1 : 0 < T) (h2 : 0 < R)
noncomputable def boy_reaches_school_early : Prop :=
  (7/6 * R) * (T - 5) = R * T

-- The theorem stating the usual time to reach the school
theorem usual_time_to_school (h : boy_reaches_school_early R T) : T = 35 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_school_l2334_233447


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l2334_233424

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l2334_233424


namespace NUMINAMATH_GPT_fruit_weights_determined_l2334_233469

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end NUMINAMATH_GPT_fruit_weights_determined_l2334_233469


namespace NUMINAMATH_GPT_planted_fraction_l2334_233458

theorem planted_fraction (a b : ℕ) (hypotenuse : ℚ) (distance_to_hypotenuse : ℚ) (x : ℚ)
  (h_triangle : a = 5 ∧ b = 12 ∧ hypotenuse = 13)
  (h_distance : distance_to_hypotenuse = 3)
  (h_x : x = 39 / 17)
  (h_square_area : x^2 = 1521 / 289)
  (total_area : ℚ) (planted_area : ℚ)
  (h_total_area : total_area = 30)
  (h_planted_area : planted_area = 7179 / 289) :
  planted_area / total_area = 2393 / 2890 :=
by
  sorry

end NUMINAMATH_GPT_planted_fraction_l2334_233458


namespace NUMINAMATH_GPT_prove_mouse_cost_l2334_233468

variable (M K : ℕ)

theorem prove_mouse_cost (h1 : K = 3 * M) (h2 : M + K = 64) : M = 16 :=
by
  sorry

end NUMINAMATH_GPT_prove_mouse_cost_l2334_233468


namespace NUMINAMATH_GPT_system_of_equations_solution_l2334_233480

theorem system_of_equations_solution (x y z : ℝ) :
  x^2 - y * z = -23 ∧ y^2 - z * x = -4 ∧ z^2 - x * y = 34 →
  (x = 5 ∧ y = 6 ∧ z = 8) ∨ (x = -5 ∧ y = -6 ∧ z = -8) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2334_233480


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l2334_233409

theorem remainder_when_divided_by_7 
  {k : ℕ} 
  (h1 : k % 5 = 2) 
  (h2 : k % 6 = 5) 
  (h3 : k < 41) : 
  k % 7 = 3 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l2334_233409


namespace NUMINAMATH_GPT_find_value_of_m_l2334_233408

variables (x y m : ℝ)

theorem find_value_of_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m) (hz_max : ∀ z, (z = x - 3 * y) → z ≤ 8) :
  m = -4 :=
sorry

end NUMINAMATH_GPT_find_value_of_m_l2334_233408


namespace NUMINAMATH_GPT_sqrt_37_range_l2334_233491

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_37_range_l2334_233491


namespace NUMINAMATH_GPT_race_lead_distance_l2334_233418

theorem race_lead_distance :
  ∀ (d12 d13 : ℝ) (s1 s2 s3 t : ℝ), 
  d12 = 2 →
  d13 = 4 →
  t > 0 →
  s1 = (d12 / t + s2) →
  s1 = (d13 / t + s3) →
  s2 * t - s3 * t = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_race_lead_distance_l2334_233418


namespace NUMINAMATH_GPT_tan_double_angle_sub_l2334_233493

theorem tan_double_angle_sub (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β) = 1 / 5) : Real.tan (2 * α - β) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_sub_l2334_233493


namespace NUMINAMATH_GPT_triangle_side_AC_value_l2334_233484

theorem triangle_side_AC_value
  (AB BC : ℝ) (AC : ℕ)
  (hAB : AB = 1)
  (hBC : BC = 2007)
  (hAC_int : ∃ (n : ℕ), AC = n) :
  AC = 2007 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_AC_value_l2334_233484


namespace NUMINAMATH_GPT_average_brown_mms_l2334_233405

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

def average (lst : List Nat) : Float :=
  (lst.foldl (· + ·) 0).toFloat / lst.length.toFloat
  
theorem average_brown_mms :
  average brown_smiley_counts = 8 ∧
  average brown_star_counts = 9.6 :=
by 
  sorry

end NUMINAMATH_GPT_average_brown_mms_l2334_233405


namespace NUMINAMATH_GPT_find_a20_l2334_233441

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem find_a20 (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1 : a 1 = -1)
  (h_S10 : S 10 = 35) :
  a 20 = 18 :=
sorry

end NUMINAMATH_GPT_find_a20_l2334_233441


namespace NUMINAMATH_GPT_reciprocal_eq_self_l2334_233462

open Classical

theorem reciprocal_eq_self (a : ℝ) (h : a = 1 / a) : a = 1 ∨ a = -1 := 
sorry

end NUMINAMATH_GPT_reciprocal_eq_self_l2334_233462


namespace NUMINAMATH_GPT_probability_difference_l2334_233437

noncomputable def Ps (red black : ℕ) : ℚ :=
  let total := red + black
  (red * (red - 1) + black * (black - 1)) / (total * (total - 1))

noncomputable def Pd (red black : ℕ) : ℚ :=
  let total := red + black
  (red * black * 2) / (total * (total - 1))

noncomputable def abs_diff (Ps Pd : ℚ) : ℚ :=
  |Ps - Pd|

theorem probability_difference :
  let red := 1200
  let black := 800
  let total := red + black
  abs_diff (Ps red black) (Pd red black) = 789 / 19990 := by
  sorry

end NUMINAMATH_GPT_probability_difference_l2334_233437


namespace NUMINAMATH_GPT_probability_of_pairing_with_friends_l2334_233411

theorem probability_of_pairing_with_friends (n : ℕ) (f : ℕ) (h1 : n = 32) (h2 : f = 2):
  (f / (n - 1) : ℚ) = 2 / 31 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_probability_of_pairing_with_friends_l2334_233411


namespace NUMINAMATH_GPT_ab_bc_ca_leq_zero_l2334_233410

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_leq_zero_l2334_233410


namespace NUMINAMATH_GPT_sector_angle_solution_l2334_233460

theorem sector_angle_solution (R α : ℝ) (h1 : 2 * R + α * R = 6) (h2 : (1/2) * R^2 * α = 2) : α = 1 ∨ α = 4 := 
sorry

end NUMINAMATH_GPT_sector_angle_solution_l2334_233460


namespace NUMINAMATH_GPT_cos_5alpha_eq_sin_5alpha_eq_l2334_233426

noncomputable def cos_five_alpha (α : ℝ) : ℝ := 16 * (Real.cos α) ^ 5 - 20 * (Real.cos α) ^ 3 + 5 * (Real.cos α)
noncomputable def sin_five_alpha (α : ℝ) : ℝ := 16 * (Real.sin α) ^ 5 - 20 * (Real.sin α) ^ 3 + 5 * (Real.sin α)

theorem cos_5alpha_eq (α : ℝ) : Real.cos (5 * α) = cos_five_alpha α :=
by sorry

theorem sin_5alpha_eq (α : ℝ) : Real.sin (5 * α) = sin_five_alpha α :=
by sorry

end NUMINAMATH_GPT_cos_5alpha_eq_sin_5alpha_eq_l2334_233426


namespace NUMINAMATH_GPT_carly_trimmed_nails_correct_l2334_233438

-- Definitions based on the conditions
def total_dogs : Nat := 11
def three_legged_dogs : Nat := 3
def paws_per_four_legged_dog : Nat := 4
def paws_per_three_legged_dog : Nat := 3
def nails_per_paw : Nat := 4

-- Mathematically equivalent proof problem in Lean 4 statement
theorem carly_trimmed_nails_correct :
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := paws_per_four_legged_dog * nails_per_paw
  let nails_per_three_legged_dog := paws_per_three_legged_dog * nails_per_paw
  let total_nails_trimmed :=
    (four_legged_dogs * nails_per_four_legged_dog) +
    (three_legged_dogs * nails_per_three_legged_dog)
  total_nails_trimmed = 164 := by
  sorry

end NUMINAMATH_GPT_carly_trimmed_nails_correct_l2334_233438


namespace NUMINAMATH_GPT_promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l2334_233479

-- Definitions of conditions
def attracts_new_clients (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (new_clients : Prop), new_clients

def promotes_partnerships (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (partnerships : Prop), partnerships

def enhances_competitiveness (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (competitiveness : Prop), competitiveness

def liquidity_advantage (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (liquidity : Prop), liquidity

def no_expiry_concerns (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (no_expiry : Prop), no_expiry

def no_partner_limitations (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (partner_limitations : Prop), ¬partner_limitations

-- Lean statements for the proof problems
theorem promoting_cashback_beneficial_for_bank (bank_promotes_cashback : Prop) :
  attracts_new_clients bank_promotes_cashback ∧
  promotes_partnerships bank_promotes_cashback ∧ 
  enhances_competitiveness bank_promotes_cashback →
  bank_promotes_cashback := 
sorry

theorem cashback_in_rubles_preferable (cashback_rubles : Prop) :
  liquidity_advantage cashback_rubles ∧
  no_expiry_concerns cashback_rubles ∧
  no_partner_limitations cashback_rubles →
  cashback_rubles :=
sorry

end NUMINAMATH_GPT_promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l2334_233479


namespace NUMINAMATH_GPT_combined_resistance_parallel_l2334_233488

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5)
  (h_combined : 1 / r = 1 / x + 1 / y) : r = 20 / 9 := by
  sorry

end NUMINAMATH_GPT_combined_resistance_parallel_l2334_233488


namespace NUMINAMATH_GPT_function_neither_odd_nor_even_l2334_233494

def f (x : ℝ) : ℝ := x^2 + 6 * x

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_function_neither_odd_nor_even_l2334_233494


namespace NUMINAMATH_GPT_factorable_b_even_l2334_233400

-- Defining the conditions
def is_factorable (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    m * p = 15 ∧ n * q = 15 ∧ b = m * q + n * p

-- The theorem to be stated
theorem factorable_b_even (b : ℤ) : is_factorable b ↔ ∃ k : ℤ, b = 2 * k :=
sorry

end NUMINAMATH_GPT_factorable_b_even_l2334_233400


namespace NUMINAMATH_GPT_two_digit_number_representation_l2334_233486

theorem two_digit_number_representation (x : ℕ) (h : x < 10) : 10 * x + 5 < 100 :=
by sorry

end NUMINAMATH_GPT_two_digit_number_representation_l2334_233486


namespace NUMINAMATH_GPT_lucy_cardinals_vs_blue_jays_l2334_233481

noncomputable def day1_cardinals : ℕ := 3
noncomputable def day1_blue_jays : ℕ := 2
noncomputable def day2_cardinals : ℕ := 3
noncomputable def day2_blue_jays : ℕ := 3
noncomputable def day3_cardinals : ℕ := 4
noncomputable def day3_blue_jays : ℕ := 2

theorem lucy_cardinals_vs_blue_jays :
  (day1_cardinals + day2_cardinals + day3_cardinals) - (day1_blue_jays + day2_blue_jays + day3_blue_jays) = 3 :=
  by sorry

end NUMINAMATH_GPT_lucy_cardinals_vs_blue_jays_l2334_233481


namespace NUMINAMATH_GPT_cross_shape_rectangle_count_l2334_233407

def original_side_length := 30
def smallest_square_side_length := 1
def cut_corner_length := 10
def N : ℕ := sorry  -- total number of rectangles in the resultant graph paper
def result : ℕ := 14413

theorem cross_shape_rectangle_count :
  (1/10 : ℚ) * N = result := 
sorry

end NUMINAMATH_GPT_cross_shape_rectangle_count_l2334_233407


namespace NUMINAMATH_GPT_negation_universal_proposition_l2334_233473

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l2334_233473


namespace NUMINAMATH_GPT_speed_upstream_calculation_l2334_233406

def speed_boat_still_water : ℝ := 60
def speed_current : ℝ := 17

theorem speed_upstream_calculation : speed_boat_still_water - speed_current = 43 := by
  sorry

end NUMINAMATH_GPT_speed_upstream_calculation_l2334_233406


namespace NUMINAMATH_GPT_wilted_flowers_correct_l2334_233466

-- Definitions based on the given conditions
def total_flowers := 45
def flowers_per_bouquet := 5
def bouquets_made := 2

-- Calculating the number of flowers used for bouquets
def used_flowers : ℕ := bouquets_made * flowers_per_bouquet

-- Question: How many flowers wilted before the wedding?
-- Statement: Prove the number of wilted flowers is 35.
theorem wilted_flowers_correct : total_flowers - used_flowers = 35 := by
  sorry

end NUMINAMATH_GPT_wilted_flowers_correct_l2334_233466


namespace NUMINAMATH_GPT_intersection_x_value_l2334_233444

theorem intersection_x_value : ∃ x y : ℝ, y = 3 * x + 7 ∧ 3 * x - 2 * y = -4 ∧ x = -10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_value_l2334_233444


namespace NUMINAMATH_GPT_count_zero_vectors_l2334_233427

variable {V : Type} [AddCommGroup V]

variables (A B C D M O : V)

def vector_expressions_1 := (A - B) + (B - C) + (C - A) = 0
def vector_expressions_2 := (A - B) + (M - B) + (B - O) + (O - M) ≠ 0
def vector_expressions_3 := (A - B) - (A - C) + (B - D) - (C - D) = 0
def vector_expressions_4 := (O - A) + (O - C) + (B - O) + (C - O) ≠ 0

theorem count_zero_vectors :
  (vector_expressions_1 A B C) ∧
  (vector_expressions_2 A B M O) ∧
  (vector_expressions_3 A B C D) ∧
  (vector_expressions_4 O A C B) →
  (2 = 2) :=
sorry

end NUMINAMATH_GPT_count_zero_vectors_l2334_233427


namespace NUMINAMATH_GPT_fly_total_distance_l2334_233453

-- Definitions and conditions
def cyclist_speed : ℝ := 10 -- speed of each cyclist in miles per hour
def initial_distance : ℝ := 50 -- initial distance between the cyclists in miles
def fly_speed : ℝ := 15 -- speed of the fly in miles per hour

-- Statement to prove
theorem fly_total_distance : 
  (cyclist_speed * 2 * initial_distance / (cyclist_speed + cyclist_speed) / fly_speed * fly_speed) = 37.5 :=
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_fly_total_distance_l2334_233453


namespace NUMINAMATH_GPT_cost_of_case_of_rolls_l2334_233475

noncomputable def cost_of_multiple_rolls (n : ℕ) (individual_cost : ℝ) : ℝ :=
  n * individual_cost

theorem cost_of_case_of_rolls :
  ∀ (n : ℕ) (C : ℝ) (individual_cost savings_perc : ℝ),
    n = 12 →
    individual_cost = 1 →
    savings_perc = 0.25 →
    C = cost_of_multiple_rolls n (individual_cost * (1 - savings_perc)) →
    C = 9 :=
by
  intros n C individual_cost savings_perc h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_of_case_of_rolls_l2334_233475


namespace NUMINAMATH_GPT_problem_a2014_l2334_233492

-- Given conditions
def seq (a : ℕ → ℕ) := a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 1

-- Prove the required statement
theorem problem_a2014 (a : ℕ → ℕ) (h : seq a) : a 2014 = 2014 :=
by sorry

end NUMINAMATH_GPT_problem_a2014_l2334_233492


namespace NUMINAMATH_GPT_tan_17pi_over_4_l2334_233425

theorem tan_17pi_over_4 : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_17pi_over_4_l2334_233425


namespace NUMINAMATH_GPT_peter_pairs_of_pants_l2334_233483

-- Define the conditions
def shirt_cost_condition (S : ℕ) : Prop := 2 * S = 20
def pants_cost (P : ℕ) : Prop := P = 6
def purchase_condition (P S : ℕ) (number_of_pants : ℕ) : Prop :=
  P * number_of_pants + 5 * S = 62

-- State the proof problem:
theorem peter_pairs_of_pants (S P number_of_pants : ℕ) 
  (h1 : shirt_cost_condition S)
  (h2 : pants_cost P) 
  (h3 : purchase_condition P S number_of_pants) :
  number_of_pants = 2 := by
  sorry

end NUMINAMATH_GPT_peter_pairs_of_pants_l2334_233483


namespace NUMINAMATH_GPT_rabbit_roaming_area_l2334_233464

noncomputable def rabbit_area_midpoint_long_side (r: ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

noncomputable def rabbit_area_3_ft_from_corner (R r: ℝ) : ℝ :=
  (3/4) * Real.pi * R^2 - (1/4) * Real.pi * r^2

theorem rabbit_roaming_area (r R : ℝ) (h_r_pos: 0 < r) (h_R_pos: r < R) :
  rabbit_area_3_ft_from_corner R r - rabbit_area_midpoint_long_side R = 22.75 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_rabbit_roaming_area_l2334_233464


namespace NUMINAMATH_GPT_tangent_line_at_x_is_2_l2334_233436

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - 3 * Real.log x

theorem tangent_line_at_x_is_2 :
  ∃ x₀ : ℝ, (x₀ > 0) ∧ ((1/2) * x₀ - (3 / x₀) = -1/2) ∧ x₀ = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_x_is_2_l2334_233436


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2334_233496

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 601 ≤ 9} = {x : ℝ | 19.25545 ≤ x ∧ x ≤ 30.74455} :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2334_233496


namespace NUMINAMATH_GPT_average_price_of_returned_cans_l2334_233455

theorem average_price_of_returned_cans (total_cans : ℕ) (returned_cans : ℕ) (remaining_cans : ℕ)
  (avg_price_total : ℚ) (avg_price_remaining : ℚ) :
  total_cans = 6 →
  returned_cans = 2 →
  remaining_cans = 4 →
  avg_price_total = 36.5 →
  avg_price_remaining = 30 →
  (avg_price_total * total_cans - avg_price_remaining * remaining_cans) / returned_cans = 49.5 :=
by
  intros h_total_cans h_returned_cans h_remaining_cans h_avg_price_total h_avg_price_remaining
  rw [h_total_cans, h_returned_cans, h_remaining_cans, h_avg_price_total, h_avg_price_remaining]
  sorry

end NUMINAMATH_GPT_average_price_of_returned_cans_l2334_233455


namespace NUMINAMATH_GPT_ladder_base_distance_l2334_233467

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end NUMINAMATH_GPT_ladder_base_distance_l2334_233467


namespace NUMINAMATH_GPT_ziggy_song_requests_l2334_233446

theorem ziggy_song_requests :
  ∃ T : ℕ, 
    (T = (1/2) * T + (1/6) * T + 5 + 2 + 1 + 2) →
    T = 30 :=
by 
  sorry

end NUMINAMATH_GPT_ziggy_song_requests_l2334_233446


namespace NUMINAMATH_GPT_number_of_balls_is_fifty_l2334_233449

variable (x : ℝ)
variable (h : x - 40 = 60 - x)

theorem number_of_balls_is_fifty : x = 50 :=
by
  have : 2 * x = 100 := by
    linarith
  linarith

end NUMINAMATH_GPT_number_of_balls_is_fifty_l2334_233449


namespace NUMINAMATH_GPT_binom_n_2_l2334_233431

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end NUMINAMATH_GPT_binom_n_2_l2334_233431


namespace NUMINAMATH_GPT_math_city_police_officers_needed_l2334_233403

def number_of_streets : Nat := 10
def initial_intersections : Nat := Nat.choose number_of_streets 2
def non_intersections : Nat := 2
def effective_intersections : Nat := initial_intersections - non_intersections

theorem math_city_police_officers_needed :
  effective_intersections = 43 := by
  sorry

end NUMINAMATH_GPT_math_city_police_officers_needed_l2334_233403


namespace NUMINAMATH_GPT_interest_rate_for_lending_l2334_233495

def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ :=
  (P * R * T) / 100

theorem interest_rate_for_lending :
  ∀ (P T R_b G R_l : ℕ),
  P = 20000 →
  T = 6 →
  R_b = 8 →
  G = 200 →
  simple_interest P R_b T + G * T = simple_interest P R_l T →
  R_l = 9 :=
by
  intros P T R_b G R_l
  sorry

end NUMINAMATH_GPT_interest_rate_for_lending_l2334_233495


namespace NUMINAMATH_GPT_simplify_expression_l2334_233443

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^3 + 2 * b^2) - 2 * b^2 + 5 = 9 * b^4 + 6 * b^3 - 2 * b^2 + 5 := sorry

end NUMINAMATH_GPT_simplify_expression_l2334_233443


namespace NUMINAMATH_GPT_domain_of_f_l2334_233440

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x) + Real.sqrt (x * (x + 1))

theorem domain_of_f :
  {x : ℝ | -x ≥ 0 ∧ x * (x + 1) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x = 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2334_233440


namespace NUMINAMATH_GPT_factor_expression_l2334_233434

theorem factor_expression (x : ℝ) :
  84 * x ^ 5 - 210 * x ^ 9 = -42 * x ^ 5 * (5 * x ^ 4 - 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2334_233434


namespace NUMINAMATH_GPT_n_not_both_perfect_squares_l2334_233429

open Int

theorem n_not_both_perfect_squares (n x y : ℤ) (h1 : n > 0) :
  ¬ ((n + 1 = x^2) ∧ (4 * n + 1 = y^2)) :=
by {
  -- Problem restated in Lean, proof not required
  sorry
}

end NUMINAMATH_GPT_n_not_both_perfect_squares_l2334_233429


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2334_233448

variable {a b : ℝ}

theorem problem_a (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : ab ≤ 1 / 8 := sorry

theorem problem_b (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (1 / a) + (8 / b) ≥ 25 := sorry

theorem problem_c (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 := sorry

theorem problem_d (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 - b^2 > -1 / 4 := sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2334_233448


namespace NUMINAMATH_GPT_sky_falls_distance_l2334_233428

def distance_from_city (x : ℕ) (y : ℕ) : Prop := 50 * x = y

theorem sky_falls_distance :
    ∃ D_s : ℕ, distance_from_city D_s 400 ∧ D_s = 8 :=
by
  sorry

end NUMINAMATH_GPT_sky_falls_distance_l2334_233428


namespace NUMINAMATH_GPT_min_value_frac_ineq_l2334_233477

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∃ x, x = (1/a) + (2/b) ∧ x ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_frac_ineq_l2334_233477


namespace NUMINAMATH_GPT_max_value_of_g_is_34_l2334_233485
noncomputable def g : ℕ → ℕ
| n => if n < 15 then n + 20 else g (n - 7)

theorem max_value_of_g_is_34 : ∃ n, g n = 34 ∧ ∀ m, g m ≤ 34 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_g_is_34_l2334_233485


namespace NUMINAMATH_GPT_smallest_x_plus_y_l2334_233498

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l2334_233498


namespace NUMINAMATH_GPT_profit_percentage_is_ten_l2334_233452

-- Definitions based on conditions
def cost_price := 500
def selling_price := 550

-- Defining the profit percentage
def profit := selling_price - cost_price
def profit_percentage := (profit / cost_price) * 100

-- The proof that the profit percentage is 10
theorem profit_percentage_is_ten : profit_percentage = 10 :=
by
  -- Using the definitions given
  sorry

end NUMINAMATH_GPT_profit_percentage_is_ten_l2334_233452


namespace NUMINAMATH_GPT_extra_time_A_to_reach_destination_l2334_233420

theorem extra_time_A_to_reach_destination (speed_ratio : ℕ -> ℕ -> Prop) (t_A t_B : ℝ)
  (h_ratio : speed_ratio 3 4)
  (time_A : t_A = 2)
  (distance_constant : ∀ a b : ℝ, a / b = (3 / 4)) :
  (t_A - t_B) * 60 = 30 :=
by
  sorry

end NUMINAMATH_GPT_extra_time_A_to_reach_destination_l2334_233420


namespace NUMINAMATH_GPT_lemon_bag_mass_l2334_233419

variable (m : ℝ)  -- mass of one bag of lemons in kg

-- Conditions
def max_load := 900  -- maximum load in kg
def num_bags := 100  -- number of bags
def extra_load := 100  -- additional load in kg

-- Proof statement (target)
theorem lemon_bag_mass : num_bags * m + extra_load = max_load → m = 8 :=
by
  sorry

end NUMINAMATH_GPT_lemon_bag_mass_l2334_233419


namespace NUMINAMATH_GPT_cookie_baking_l2334_233497

/-- It takes 7 minutes to bake 1 pan of cookies. In 28 minutes, you can bake 4 pans of cookies. -/
theorem cookie_baking (bake_time_per_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) 
  (h1 : bake_time_per_pan = 7)
  (h2 : total_time = 28) : 
  num_pans = 4 := 
by
  sorry

end NUMINAMATH_GPT_cookie_baking_l2334_233497


namespace NUMINAMATH_GPT_pyramid_volume_formula_l2334_233404

noncomputable def pyramid_volume (a α β : ℝ) : ℝ :=
  (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β)

theorem pyramid_volume_formula (a α β : ℝ) :
  (base_is_isosceles_triangle : Prop) → (lateral_edges_inclined : Prop) → 
  pyramid_volume a α β = (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β) :=
by
  intros c1 c2
  exact sorry

end NUMINAMATH_GPT_pyramid_volume_formula_l2334_233404


namespace NUMINAMATH_GPT_minimum_value_l2334_233478

theorem minimum_value (x : ℝ) (hx : x > 0) : 4 * x^2 + 1 / x^3 ≥ 5 ∧ (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_l2334_233478


namespace NUMINAMATH_GPT_teal_sales_l2334_233401

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end NUMINAMATH_GPT_teal_sales_l2334_233401


namespace NUMINAMATH_GPT_bill_experience_l2334_233463

theorem bill_experience (B J : ℕ) (h1 : J - 5 = 3 * (B - 5)) (h2 : J = 2 * B) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_bill_experience_l2334_233463


namespace NUMINAMATH_GPT_sum_r_p_values_l2334_233450

def p (x : ℝ) : ℝ := |x| - 2
def r (x : ℝ) : ℝ := -|p x - 1|
def r_p (x : ℝ) : ℝ := r (p x)

theorem sum_r_p_values :
  (r_p (-4) + r_p (-3) + r_p (-2) + r_p (-1) + r_p 0 + r_p 1 + r_p 2 + r_p 3 + r_p 4) = -11 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_r_p_values_l2334_233450


namespace NUMINAMATH_GPT_ninth_square_more_than_eighth_l2334_233471

noncomputable def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

noncomputable def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_square_more_than_eighth : (tile_count 9 - tile_count 8) = 72 :=
by sorry

end NUMINAMATH_GPT_ninth_square_more_than_eighth_l2334_233471


namespace NUMINAMATH_GPT_total_additions_and_multiplications_l2334_233482

def f(x : ℝ) : ℝ := 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem total_additions_and_multiplications {x : ℝ} (h : x = 0.6) :
  let horner_f := ((((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x + 7)
  (horner_f = f x) ∧ (6 + 6 = 12) :=
by
  sorry

end NUMINAMATH_GPT_total_additions_and_multiplications_l2334_233482


namespace NUMINAMATH_GPT_estimate_time_pm_l2334_233414

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end NUMINAMATH_GPT_estimate_time_pm_l2334_233414
