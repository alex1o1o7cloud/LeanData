import Mathlib

namespace NUMINAMATH_GPT_intersection_S_T_l759_75958

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_S_T : S ∩ T = {x | -2 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l759_75958


namespace NUMINAMATH_GPT_rectangle_area_difference_l759_75908

theorem rectangle_area_difference :
  let area (l w : ℝ) := l * w
  let combined_area (l w : ℝ) := 2 * area l w
  combined_area 11 19 - combined_area 9.5 11 = 209 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_difference_l759_75908


namespace NUMINAMATH_GPT_copy_pages_15_dollars_l759_75974

theorem copy_pages_15_dollars (cpp : ℕ) (budget : ℕ) (pages : ℕ) (h1 : cpp = 3) (h2 : budget = 1500) (h3 : pages = budget / cpp) : pages = 500 :=
by
  sorry

end NUMINAMATH_GPT_copy_pages_15_dollars_l759_75974


namespace NUMINAMATH_GPT_increasing_function_range_l759_75949

theorem increasing_function_range (k : ℝ) :
  (∀ x y : ℝ, x < y → (k + 2) * x + 1 < (k + 2) * y + 1) ↔ k > -2 :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_range_l759_75949


namespace NUMINAMATH_GPT_right_triangle_conditions_l759_75928

theorem right_triangle_conditions (x y z h α β : ℝ) : 
  x - y = α → 
  z - h = β → 
  x^2 + y^2 = z^2 → 
  x * y = h * z → 
  β > α :=
by 
sorry

end NUMINAMATH_GPT_right_triangle_conditions_l759_75928


namespace NUMINAMATH_GPT_polygon_sides_l759_75930

theorem polygon_sides (n : ℕ) (h : n ≥ 3) (sum_angles : (n - 2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l759_75930


namespace NUMINAMATH_GPT_sum_of_acute_angles_l759_75952

theorem sum_of_acute_angles (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : β > 0 ∧ β < π / 2) (h3: γ > 0 ∧ γ < π / 2) (h4 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) :
  (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
by
  sorry

end NUMINAMATH_GPT_sum_of_acute_angles_l759_75952


namespace NUMINAMATH_GPT_find_value_of_a_squared_b_plus_ab_squared_l759_75981

theorem find_value_of_a_squared_b_plus_ab_squared 
  (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : ab = 2) : 
  a^2 * b + a * b^2 = -6 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_a_squared_b_plus_ab_squared_l759_75981


namespace NUMINAMATH_GPT_candle_height_l759_75929

variable (h d a b x : ℝ)

theorem candle_height (h d a b : ℝ) : x = h * (1 + d / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_candle_height_l759_75929


namespace NUMINAMATH_GPT_total_area_of_figure_l759_75925

theorem total_area_of_figure :
  let h := 7
  let w1 := 6
  let h1 := 2
  let h2 := 3
  let h3 := 1
  let w2 := 5
  let a1 := h * w1
  let a2 := (h - h1) * (11 - 7)
  let a3 := (h - h1 - h2) * (11 - 7)
  let a4 := (15 - 11) * h3
  a1 + a2 + a3 + a4 = 74 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_figure_l759_75925


namespace NUMINAMATH_GPT_average_salary_l759_75946

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 14000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

theorem average_salary : (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8200 := 
  by 
    sorry

end NUMINAMATH_GPT_average_salary_l759_75946


namespace NUMINAMATH_GPT_find_larger_number_l759_75947

theorem find_larger_number (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l759_75947


namespace NUMINAMATH_GPT_fraction_sum_is_five_l759_75982

noncomputable def solve_fraction_sum (x y z : ℝ) : Prop :=
  (x + 1/y = 5) ∧ (y + 1/z = 2) ∧ (z + 1/x = 3) ∧ 0 < x ∧ 0 < y ∧ 0 < z → 
  (x / y + y / z + z / x = 5)
    
theorem fraction_sum_is_five (x y z : ℝ) : solve_fraction_sum x y z :=
  sorry

end NUMINAMATH_GPT_fraction_sum_is_five_l759_75982


namespace NUMINAMATH_GPT_initial_amount_l759_75951

theorem initial_amount (bread_price : ℝ) (bread_qty : ℝ) (pb_price : ℝ) (leftover : ℝ) :
  bread_price = 2.25 → bread_qty = 3 → pb_price = 2 → leftover = 5.25 →
  bread_qty * bread_price + pb_price + leftover = 14 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num


end NUMINAMATH_GPT_initial_amount_l759_75951


namespace NUMINAMATH_GPT_f_divisible_by_27_l759_75944

theorem f_divisible_by_27 (n : ℕ) : 27 ∣ (2^(2*n - 1) - 9 * n^2 + 21 * n - 14) :=
sorry

end NUMINAMATH_GPT_f_divisible_by_27_l759_75944


namespace NUMINAMATH_GPT_smallest_prime_factor_2379_l759_75942

-- Define the given number
def n : ℕ := 2379

-- Define the condition that 3 is a prime number.
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define the smallest prime factor
def smallest_prime_factor (n p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q, is_prime q → q ∣ n → p ≤ q)

-- The statement that 3 is the smallest prime factor of 2379
theorem smallest_prime_factor_2379 : smallest_prime_factor n 3 :=
sorry

end NUMINAMATH_GPT_smallest_prime_factor_2379_l759_75942


namespace NUMINAMATH_GPT_parrots_false_statements_l759_75938

theorem parrots_false_statements (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 140 ∧ 
    (∀ statements : ℕ → Prop, 
      (statements 0 = false) ∧ 
      (∀ i : ℕ, 1 ≤ i → i < n → 
          (statements i = true → 
            (∃ fp : ℕ, fp < i ∧ 7 * (fp + 1) > 10 * i)))) := 
by
  sorry

end NUMINAMATH_GPT_parrots_false_statements_l759_75938


namespace NUMINAMATH_GPT_jars_left_when_boxes_full_l759_75994

-- Conditions
def jars_in_first_set_of_boxes : Nat := 12 * 10
def jars_in_second_set_of_boxes : Nat := 10 * 30
def total_jars : Nat := 500

-- Question (equivalent proof problem)
theorem jars_left_when_boxes_full : total_jars - (jars_in_first_set_of_boxes + jars_in_second_set_of_boxes) = 80 := 
by
  sorry

end NUMINAMATH_GPT_jars_left_when_boxes_full_l759_75994


namespace NUMINAMATH_GPT_total_money_given_to_children_l759_75945

theorem total_money_given_to_children (B : ℕ) (x : ℕ) (total : ℕ) 
  (h1 : B = 300) 
  (h2 : x = B / 3) 
  (h3 : total = (2 * x) + (3 * x) + (4 * x)) : 
  total = 900 := 
by 
  sorry

end NUMINAMATH_GPT_total_money_given_to_children_l759_75945


namespace NUMINAMATH_GPT_product_of_midpoint_coordinates_l759_75903

def x1 := 10
def y1 := -3
def x2 := 4
def y2 := 7

def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

theorem product_of_midpoint_coordinates : 
  midpoint_x * midpoint_y = 14 :=
by
  sorry

end NUMINAMATH_GPT_product_of_midpoint_coordinates_l759_75903


namespace NUMINAMATH_GPT_matrix_pow_three_l759_75965

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end NUMINAMATH_GPT_matrix_pow_three_l759_75965


namespace NUMINAMATH_GPT_smallest_lcm_value_l759_75918

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end NUMINAMATH_GPT_smallest_lcm_value_l759_75918


namespace NUMINAMATH_GPT_determine_k_values_l759_75954

theorem determine_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a ^ 2 + 6 * a + k = 0 ∧ 3 * b ^ 2 + 6 * b + k = 0 ∧ |a - b| = 1 / 2 * (a ^ 2 + b ^ 2)) → (k = 0 ∨ k = 12) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_values_l759_75954


namespace NUMINAMATH_GPT_annika_current_age_l759_75900

-- Define the conditions
def hans_age_current : ℕ := 8
def hans_age_in_4_years : ℕ := hans_age_current + 4
def annika_age_in_4_years : ℕ := 3 * hans_age_in_4_years

-- lean statement to prove Annika's current age
theorem annika_current_age (A : ℕ) (hyp : A + 4 = annika_age_in_4_years) : A = 32 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_annika_current_age_l759_75900


namespace NUMINAMATH_GPT_find_circle_center_l759_75910

def circle_center (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 16 = 0

theorem find_circle_center (x y : ℝ) :
  circle_center x y ↔ (x, y) = (3, 4) :=
by
  sorry

end NUMINAMATH_GPT_find_circle_center_l759_75910


namespace NUMINAMATH_GPT_negative_y_implies_negative_y_is_positive_l759_75905

theorem negative_y_implies_negative_y_is_positive (y : ℝ) (h : y < 0) : -y > 0 :=
sorry

end NUMINAMATH_GPT_negative_y_implies_negative_y_is_positive_l759_75905


namespace NUMINAMATH_GPT_choose_5_from_12_l759_75907

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end NUMINAMATH_GPT_choose_5_from_12_l759_75907


namespace NUMINAMATH_GPT_age_problem_l759_75943

theorem age_problem
  (D M : ℕ)
  (h1 : M = D + 45)
  (h2 : M - 5 = 6 * (D - 5)) :
  D = 14 ∧ M = 59 := by
  sorry

end NUMINAMATH_GPT_age_problem_l759_75943


namespace NUMINAMATH_GPT_find_m_l759_75963

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, -1)
noncomputable def a_minus_b : ℝ × ℝ := (2, 3)
noncomputable def m_a_plus_b (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m - 1)

theorem find_m (m : ℝ) : (a_minus_b.1 * (m_a_plus_b m).1 + a_minus_b.2 * (m_a_plus_b m).2) = 0 → m = 5 / 8 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l759_75963


namespace NUMINAMATH_GPT_weight_of_D_l759_75961

open Int

def weights (A B C D : Int) : Prop :=
  A < B ∧ B < C ∧ C < D ∧ 
  A + B = 45 ∧ A + C = 49 ∧ A + D = 55 ∧ 
  B + C = 54 ∧ B + D = 60 ∧ C + D = 64

theorem weight_of_D {A B C D : Int} (h : weights A B C D) : D = 35 := 
  by
    sorry

end NUMINAMATH_GPT_weight_of_D_l759_75961


namespace NUMINAMATH_GPT_percent_increase_twice_eq_44_percent_l759_75936

variable (P : ℝ) (x : ℝ)

theorem percent_increase_twice_eq_44_percent (h : P * (1 + x)^2 = P * 1.44) : x = 0.2 :=
by sorry

end NUMINAMATH_GPT_percent_increase_twice_eq_44_percent_l759_75936


namespace NUMINAMATH_GPT_polynomial_coeff_divisible_by_5_l759_75909

theorem polynomial_coeff_divisible_by_5
  (a b c : ℤ)
  (h : ∀ k : ℤ, (a * k^2 + b * k + c) % 5 = 0) :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_divisible_by_5_l759_75909


namespace NUMINAMATH_GPT_group_membership_l759_75924

theorem group_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 11 = 6) (h3 : 100 ≤ n ∧ n ≤ 200) :
  n = 116 ∨ n = 193 :=
sorry

end NUMINAMATH_GPT_group_membership_l759_75924


namespace NUMINAMATH_GPT_original_number_divisible_by_3_l759_75953

theorem original_number_divisible_by_3:
  ∃ (a b c d e f g h : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h) ∧
  (c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h) ∧
  (d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h) ∧
  (e ≠ f ∧ e ≠ g ∧ e ≠ h) ∧
  (f ≠ g ∧ f ≠ h) ∧
  (g ≠ h) ∧ 
  (a + b + c + b + d + e + f + e + g + d + h) % 3 = 0 :=
sorry

end NUMINAMATH_GPT_original_number_divisible_by_3_l759_75953


namespace NUMINAMATH_GPT_bike_helmet_cost_increase_l759_75992

open Real

theorem bike_helmet_cost_increase :
  let old_bike_cost := 150
  let old_helmet_cost := 50
  let new_bike_cost := old_bike_cost + 0.10 * old_bike_cost
  let new_helmet_cost := old_helmet_cost + 0.20 * old_helmet_cost
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let total_increase := new_total_cost - old_total_cost
  let percent_increase := (total_increase / old_total_cost) * 100
  percent_increase = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_bike_helmet_cost_increase_l759_75992


namespace NUMINAMATH_GPT_find_intersection_find_range_of_a_l759_75921

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | x < -2 ∨ (3 < x ∧ x < 4) }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 5 }

-- Proof Problem 1: Prove the intersection A ∩ B
theorem find_intersection : (A ∩ B) = { x : ℝ | 3 < x ∧ x ≤ 5 } := by
  sorry

-- Define the set C and the condition B ∩ C = B
def C (a : ℝ) : Set ℝ := { x : ℝ | x ≥ a }
def condition (a : ℝ) : Prop := B ∩ C a = B

-- Proof Problem 2: Find the range of a
theorem find_range_of_a : ∀ a : ℝ, condition a → a ≤ -3 := by
  sorry

end NUMINAMATH_GPT_find_intersection_find_range_of_a_l759_75921


namespace NUMINAMATH_GPT_tan_add_pi_over_six_l759_75948

theorem tan_add_pi_over_six (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + Real.pi / 6) = 5 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_add_pi_over_six_l759_75948


namespace NUMINAMATH_GPT_total_preparation_and_cooking_time_l759_75977

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end NUMINAMATH_GPT_total_preparation_and_cooking_time_l759_75977


namespace NUMINAMATH_GPT_susan_correct_guess_probability_l759_75926

theorem susan_correct_guess_probability :
  (1 - (5/6)^6) = 31031/46656 := 
sorry

end NUMINAMATH_GPT_susan_correct_guess_probability_l759_75926


namespace NUMINAMATH_GPT_max_k_for_3_pow_11_as_sum_of_consec_integers_l759_75956

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end NUMINAMATH_GPT_max_k_for_3_pow_11_as_sum_of_consec_integers_l759_75956


namespace NUMINAMATH_GPT_volume_Q4_l759_75904

noncomputable def tetrahedron_sequence (n : ℕ) : ℝ :=
  -- Define the sequence recursively
  match n with
  | 0       => 1
  | (n + 1) => tetrahedron_sequence n + (4^n * (1 / 27)^(n + 1))

theorem volume_Q4 : tetrahedron_sequence 4 = 1.173832 :=
by
  sorry

end NUMINAMATH_GPT_volume_Q4_l759_75904


namespace NUMINAMATH_GPT_trapezoid_area_equal_l759_75989

namespace Geometry

-- Define the areas of the outer and inner equilateral triangles.
def outer_triangle_area : ℝ := 25
def inner_triangle_area : ℝ := 4

-- The number of congruent trapezoids formed between the triangles.
def number_of_trapezoids : ℕ := 4

-- Prove that the area of one trapezoid is 5.25 square units.
theorem trapezoid_area_equal :
  (outer_triangle_area - inner_triangle_area) / number_of_trapezoids = 5.25 := by
  sorry

end Geometry

end NUMINAMATH_GPT_trapezoid_area_equal_l759_75989


namespace NUMINAMATH_GPT_p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l759_75932

variable (x : ℝ)

def p := |x| = x
def q := x^2 + x ≥ 0

theorem p_implies_q : p x → q x :=
by sorry

theorem q_not_implies_p : q x → ¬p x :=
by sorry

theorem p_sufficient_but_not_necessary : (p x → q x) ∧ ¬(q x → p x) :=
by sorry

end NUMINAMATH_GPT_p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l759_75932


namespace NUMINAMATH_GPT_variance_of_white_balls_l759_75920

section
variable (n : ℕ := 7) 
variable (p : ℚ := 3/7)

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls : binomial_variance n p = 12/7 :=
by
  sorry
end

end NUMINAMATH_GPT_variance_of_white_balls_l759_75920


namespace NUMINAMATH_GPT_percentage_difference_l759_75906

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.60 * x) (h2 : z = 0.60 * y) :
  abs ((z - x) / z * 100) = 4.17 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l759_75906


namespace NUMINAMATH_GPT_combine_expr_l759_75986

variable (a b : ℝ)

theorem combine_expr : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end NUMINAMATH_GPT_combine_expr_l759_75986


namespace NUMINAMATH_GPT_initial_number_of_nurses_l759_75927

theorem initial_number_of_nurses (N : ℕ) (initial_doctors : ℕ) (remaining_staff : ℕ) 
  (h1 : initial_doctors = 11) 
  (h2 : remaining_staff = 22) 
  (h3 : initial_doctors - 5 + N - 2 = remaining_staff) : N = 18 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_initial_number_of_nurses_l759_75927


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l759_75937

theorem polynomial_coeff_sum :
  let p := ((Polynomial.C 1 + Polynomial.X)^3 * (Polynomial.C 2 + Polynomial.X)^2)
  let a0 := p.coeff 0
  let a2 := p.coeff 2
  let a4 := p.coeff 4
  a4 + a2 + a0 = 36 := by 
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l759_75937


namespace NUMINAMATH_GPT_parallelogram_is_central_not_axis_symmetric_l759_75966

-- Definitions for the shapes discussed in the problem
def is_central_symmetric (shape : Type) : Prop := sorry
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Specific shapes being used in the problem
def rhombus : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry

-- Example additional assumptions about shapes can be added here if needed

-- The problem assertion
theorem parallelogram_is_central_not_axis_symmetric :
  is_central_symmetric parallelogram ∧ ¬ is_axis_symmetric parallelogram :=
sorry

end NUMINAMATH_GPT_parallelogram_is_central_not_axis_symmetric_l759_75966


namespace NUMINAMATH_GPT_change_calculation_l759_75976

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end NUMINAMATH_GPT_change_calculation_l759_75976


namespace NUMINAMATH_GPT_total_money_shared_l759_75919

theorem total_money_shared (rA rB rC : ℕ) (pA : ℕ) (total : ℕ) 
  (h_ratio : rA = 1 ∧ rB = 2 ∧ rC = 7) 
  (h_A_money : pA = 20) 
  (h_total : total = pA * rA + pA * rB + pA * rC) : 
  total = 200 := by 
  sorry

end NUMINAMATH_GPT_total_money_shared_l759_75919


namespace NUMINAMATH_GPT_expr_value_l759_75991

theorem expr_value : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 :=
by
  sorry

end NUMINAMATH_GPT_expr_value_l759_75991


namespace NUMINAMATH_GPT_find_y_l759_75931

noncomputable def angle_ABC := 75
noncomputable def angle_BAC := 70
noncomputable def angle_CDE := 90
noncomputable def angle_BCA : ℝ := 180 - (angle_ABC + angle_BAC)
noncomputable def y : ℝ := 90 - angle_BCA

theorem find_y : y = 55 :=
by
  have h1: angle_BCA = 180 - (75 + 70) := rfl
  have h2: y = 90 - angle_BCA := rfl
  rw [h1] at h2
  exact h2.trans (by norm_num)

end NUMINAMATH_GPT_find_y_l759_75931


namespace NUMINAMATH_GPT_minimize_distance_sum_l759_75997

open Real

noncomputable def distance_squared (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem minimize_distance_sum : 
  ∀ P : ℝ × ℝ, (P.1 = P.2) → 
    let A : ℝ × ℝ := (1, -1)
    let B : ℝ × ℝ := (2, 2)
    (distance_squared P A + distance_squared P B) ≥ 
    (distance_squared (1, 1) A + distance_squared (1, 1) B) := by
  intro P hP
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (2, 2)
  sorry

end NUMINAMATH_GPT_minimize_distance_sum_l759_75997


namespace NUMINAMATH_GPT_find_sum_of_numbers_l759_75914

-- Define the problem using the given conditions
def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

-- The main theorem we want to prove
theorem find_sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) :
  sum_of_three_numbers a b c = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_numbers_l759_75914


namespace NUMINAMATH_GPT_win_lottery_amount_l759_75955

theorem win_lottery_amount (W : ℝ) (cond1 : W * 0.20 + 5 = 35) : W = 50 := by
  sorry

end NUMINAMATH_GPT_win_lottery_amount_l759_75955


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l759_75902

noncomputable def distinct_integers (p q r s t : ℤ) : Prop :=
  (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ 
  (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ 
  (r ≠ s) ∧ (r ≠ t) ∧ 
  (s ≠ t)

theorem sum_of_distinct_integers
  (p q r s t : ℤ)
  (h_distinct : distinct_integers p q r s t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120) :
  p + q + r + s + t = 22 :=
  sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l759_75902


namespace NUMINAMATH_GPT_power_sum_l759_75984

theorem power_sum
: (-2)^(2005) + (-2)^(2006) = 2^(2005) := by
  sorry

end NUMINAMATH_GPT_power_sum_l759_75984


namespace NUMINAMATH_GPT_correct_fill_l759_75950

/- Define the conditions and the statement in Lean 4 -/
def sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

/- The condition is that the phrase without the number has 21 characters -/
def initial_length : ℕ := 21

/- Define the term "тридцать две" as the correct number to fill the blank -/
def correct_number := "тридцать две"

/- The target phrase with the correct number filled in -/
def target_sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ " ++ correct_number ++ " БУКВЫ"

/- Prove that the correct number fills the blank correctly -/
theorem correct_fill :
  (String.length target_sentence = 38) :=
by
  /- Convert everything to string length and verify -/
  sorry

end NUMINAMATH_GPT_correct_fill_l759_75950


namespace NUMINAMATH_GPT_expand_and_simplify_l759_75911

theorem expand_and_simplify (x : ℝ) : 
  -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 :=
sorry

end NUMINAMATH_GPT_expand_and_simplify_l759_75911


namespace NUMINAMATH_GPT_find_angle_A_l759_75964

theorem find_angle_A 
  (a b : ℝ) (A B : ℝ) 
  (h1 : b = 2 * a)
  (h2 : B = A + 60) : 
  A = 30 :=
  sorry

end NUMINAMATH_GPT_find_angle_A_l759_75964


namespace NUMINAMATH_GPT_negation_equiv_l759_75941

noncomputable def negate_existential : Prop :=
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0

noncomputable def universal_negation : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0

theorem negation_equiv : negate_existential = universal_negation :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_negation_equiv_l759_75941


namespace NUMINAMATH_GPT_problem_l759_75988

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x + 2
noncomputable def f' (a x : ℝ) : ℝ := a * (Real.log x + 1) + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x - x^2 - (a + 2) * x + a

theorem problem (a x : ℝ) (h : 1 ≤ x) (ha : 0 < a) : f' a x < x^2 + (a + 2) * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l759_75988


namespace NUMINAMATH_GPT_upstream_speed_l759_75973

variable (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)

def speed_of_man_in_still_water := V_m = 35
def speed_of_man_downstream := V_downstream = 45
def speed_of_man_upstream := V_upstream = 25

theorem upstream_speed
  (h1: speed_of_man_in_still_water V_m)
  (h2: speed_of_man_downstream V_downstream)
  : speed_of_man_upstream V_upstream :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_upstream_speed_l759_75973


namespace NUMINAMATH_GPT_cost_price_computer_table_l759_75901

variable (CP SP : ℝ)

theorem cost_price_computer_table (h1 : SP = 2 * CP) (h2 : SP = 1000) : CP = 500 := by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l759_75901


namespace NUMINAMATH_GPT_compare_fractions_l759_75971

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end NUMINAMATH_GPT_compare_fractions_l759_75971


namespace NUMINAMATH_GPT_Kenny_played_basketball_for_10_hours_l759_75923

theorem Kenny_played_basketball_for_10_hours
  (played_basketball ran practiced_trumpet : ℕ)
  (H1 : practiced_trumpet = 40)
  (H2 : ran = 2 * played_basketball)
  (H3 : practiced_trumpet = 2 * ran) :
  played_basketball = 10 :=
by
  sorry

end NUMINAMATH_GPT_Kenny_played_basketball_for_10_hours_l759_75923


namespace NUMINAMATH_GPT_smallest_circle_radius_eq_l759_75912

open Real

-- Declaring the problem's conditions
def largestCircleRadius : ℝ := 10
def smallestCirclesCount : ℕ := 6
def congruentSmallerCirclesFitWithinLargerCircle (r : ℝ) : Prop :=
  3 * (2 * r) = 2 * largestCircleRadius

-- Stating the theorem to prove
theorem smallest_circle_radius_eq :
  ∃ r : ℝ, congruentSmallerCirclesFitWithinLargerCircle r ∧ r = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_circle_radius_eq_l759_75912


namespace NUMINAMATH_GPT_sum_of_series_l759_75972

theorem sum_of_series : 
  (6 + 16 + 26 + 36 + 46) + (14 + 24 + 34 + 44 + 54) = 300 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_l759_75972


namespace NUMINAMATH_GPT_green_apples_more_than_red_apples_l759_75998

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end NUMINAMATH_GPT_green_apples_more_than_red_apples_l759_75998


namespace NUMINAMATH_GPT_solve_for_y_l759_75968

theorem solve_for_y : (12^3 * 6^2) / 432 = 144 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l759_75968


namespace NUMINAMATH_GPT_speed_of_B_is_three_l759_75969

noncomputable def speed_of_B (rounds_per_hour : ℕ) : Prop :=
  let A_speed : ℕ := 2
  let crossings : ℕ := 5
  let time_hours : ℕ := 1
  rounds_per_hour = (crossings - A_speed)

theorem speed_of_B_is_three : speed_of_B 3 :=
  sorry

end NUMINAMATH_GPT_speed_of_B_is_three_l759_75969


namespace NUMINAMATH_GPT_solve_for_x_l759_75957

theorem solve_for_x :
  (16^x * 16^x * 16^x * 4^(3 * x) = 64^(4 * x)) → x = 0 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l759_75957


namespace NUMINAMATH_GPT_max_value_f_l759_75933

def f (a x y : ℝ) : ℝ := a * x + y

theorem max_value_f (a : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : |x| + |y| ≤ 1) :
    f a x y ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_l759_75933


namespace NUMINAMATH_GPT_fraction_spent_on_museum_ticket_l759_75995

theorem fraction_spent_on_museum_ticket (initial_money : ℝ) (sandwich_fraction : ℝ) (book_fraction : ℝ) (remaining_money : ℝ) (h1 : initial_money = 90) (h2 : sandwich_fraction = 1/5) (h3 : book_fraction = 1/2) (h4 : remaining_money = 12) : (initial_money - remaining_money) / initial_money - (sandwich_fraction * initial_money + book_fraction * initial_money) / initial_money = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_museum_ticket_l759_75995


namespace NUMINAMATH_GPT_cost_of_one_pie_l759_75987

theorem cost_of_one_pie (x c2 c5 : ℕ) 
  (h1: 4 * x = c2 + 60)
  (h2: 5 * x = c5 + 60) 
  (h3: 6 * x = c2 + c5 + 60) : 
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_pie_l759_75987


namespace NUMINAMATH_GPT_distance_is_20_sqrt_6_l759_75983

-- Definitions for problem setup
def distance_between_parallel_lines (r d : ℝ) : Prop :=
  ∃ O C D E F P Q : ℝ, 
  40^2 * 40 + (d / 2)^2 * 40 = 40 * r^2 ∧ 
  15^2 * 30 + (d / 2)^2 * 30 = 30 * r^2

-- The main statement to be proved
theorem distance_is_20_sqrt_6 :
  ∀ r d : ℝ,
  distance_between_parallel_lines r d →
  d = 20 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_distance_is_20_sqrt_6_l759_75983


namespace NUMINAMATH_GPT_find_x_l759_75960

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l759_75960


namespace NUMINAMATH_GPT_delta_delta_delta_l759_75913

-- Define the function Δ
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Mathematical statement to be proved
theorem delta_delta_delta (x : ℝ) : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end NUMINAMATH_GPT_delta_delta_delta_l759_75913


namespace NUMINAMATH_GPT_reduced_price_is_25_l759_75915

def original_price (P : ℝ) (X : ℝ) (R : ℝ) : Prop :=
  R = 0.85 * P ∧ 
  500 = X * P ∧ 
  500 = (X + 3) * R

theorem reduced_price_is_25 (P X R : ℝ) (h : original_price P X R) :
  R = 25 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_is_25_l759_75915


namespace NUMINAMATH_GPT_exists_infinitely_many_n_with_increasing_ω_l759_75967

open Nat

/--
  Let ω(n) represent the number of distinct prime factors of a natural number n (where n > 1).
  Prove that there exist infinitely many n such that ω(n) < ω(n + 1) < ω(n + 2).
-/
theorem exists_infinitely_many_n_with_increasing_ω (ω : ℕ → ℕ) (hω : ∀ (n : ℕ), n > 1 → ∃ k, ω k < ω (k + 1) ∧ ω (k + 1) < ω (k + 2)) :
  ∃ (infinitely_many : ℕ → Prop), ∀ N : ℕ, ∃ n : ℕ, N < n ∧ infinitely_many n :=
by
  sorry

end NUMINAMATH_GPT_exists_infinitely_many_n_with_increasing_ω_l759_75967


namespace NUMINAMATH_GPT_smallest_number_of_three_l759_75996

theorem smallest_number_of_three (a b c : ℕ) (h1 : a + b + c = 78) (h2 : b = 27) (h3 : c = b + 5) :
  a = 19 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_three_l759_75996


namespace NUMINAMATH_GPT_student_A_final_score_l759_75934

theorem student_A_final_score (total_questions : ℕ) (correct_responses : ℕ) 
  (h1 : total_questions = 100) (h2 : correct_responses = 93) : 
  correct_responses - 2 * (total_questions - correct_responses) = 79 :=
by
  rw [h1, h2]
  -- sorry

end NUMINAMATH_GPT_student_A_final_score_l759_75934


namespace NUMINAMATH_GPT_supremum_of_function_l759_75959

theorem supremum_of_function : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
  (∃ M : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ M) ∧
    (∀ K : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ K) → M ≤ K) → M = -9 / 2) := 
sorry

end NUMINAMATH_GPT_supremum_of_function_l759_75959


namespace NUMINAMATH_GPT_base_seven_sum_of_digits_of_product_l759_75917

theorem base_seven_sum_of_digits_of_product :
  let a := 24
  let b := 30
  let product := a * b
  let base_seven_product := 105 -- The product in base seven notation
  let sum_of_digits (n : ℕ) : ℕ := n.digits 7 |> List.sum
  sum_of_digits base_seven_product = 6 :=
by
  sorry

end NUMINAMATH_GPT_base_seven_sum_of_digits_of_product_l759_75917


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l759_75970

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_l759_75970


namespace NUMINAMATH_GPT_bicycle_owners_no_car_l759_75980

-- Definitions based on the conditions in (a)
def total_adults : ℕ := 500
def bicycle_owners : ℕ := 450
def car_owners : ℕ := 120
def both_owners : ℕ := bicycle_owners + car_owners - total_adults

-- Proof problem statement
theorem bicycle_owners_no_car : (bicycle_owners - both_owners = 380) :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_bicycle_owners_no_car_l759_75980


namespace NUMINAMATH_GPT_tan_identity_l759_75940

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end NUMINAMATH_GPT_tan_identity_l759_75940


namespace NUMINAMATH_GPT_find_f_l759_75990

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sqrt x + 4) = x + 8 * Real.sqrt x) :
  ∀ (x : ℝ), x ≥ 4 → f x = x^2 - 16 :=
by
  sorry

end NUMINAMATH_GPT_find_f_l759_75990


namespace NUMINAMATH_GPT_f_2016_eq_neg1_l759_75922

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

theorem f_2016_eq_neg1 : f 2016 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_f_2016_eq_neg1_l759_75922


namespace NUMINAMATH_GPT_expand_and_simplify_product_l759_75985

variable (x : ℝ)

theorem expand_and_simplify_product :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_product_l759_75985


namespace NUMINAMATH_GPT_dawn_annual_salary_l759_75939

variable (M : ℝ)

theorem dawn_annual_salary (h1 : 0.10 * M = 400) : M * 12 = 48000 := by
  sorry

end NUMINAMATH_GPT_dawn_annual_salary_l759_75939


namespace NUMINAMATH_GPT_find_seating_capacity_l759_75999

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_seating_capacity_l759_75999


namespace NUMINAMATH_GPT_quadratic_equation_solutions_l759_75978

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 + 7 * x = 0 ↔ (x = 0 ∨ x = -7) := 
by 
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_equation_solutions_l759_75978


namespace NUMINAMATH_GPT_star_result_l759_75935

-- Define the operation star
def star (m n p q : ℚ) := (m * p) * (n / q)

-- Given values
def a := (5 : ℚ) / 9
def b := (10 : ℚ) / 6

-- Condition to check
theorem star_result : star 5 9 10 6 = 75 := by
  sorry

end NUMINAMATH_GPT_star_result_l759_75935


namespace NUMINAMATH_GPT_delta_value_l759_75975

theorem delta_value (Δ : ℝ) (h : 4 * 3 = Δ - 6) : Δ = 18 :=
sorry

end NUMINAMATH_GPT_delta_value_l759_75975


namespace NUMINAMATH_GPT_circle_radius_l759_75979

theorem circle_radius (r x y : ℝ) (hx : x = π * r^2) (hy : y = 2 * π * r) (h : x + y = 90 * π) : r = 9 := by
  sorry

end NUMINAMATH_GPT_circle_radius_l759_75979


namespace NUMINAMATH_GPT_solve_y_determinant_l759_75993

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_determinant_l759_75993


namespace NUMINAMATH_GPT_min_distance_between_M_and_N_l759_75916

noncomputable def f (x : ℝ) := Real.sin x + (1 / 6) * x^3
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_between_M_and_N :
  ∃ (x1 x2 : ℝ), x1 ≥ 0 ∧ x2 ≥ 0 ∧ f x1 = g x2 ∧ (x2 - x1 = 1) :=
sorry

end NUMINAMATH_GPT_min_distance_between_M_and_N_l759_75916


namespace NUMINAMATH_GPT_function_increasing_iff_m_eq_1_l759_75962

theorem function_increasing_iff_m_eq_1 (m : ℝ) : 
  (m^2 - 4 * m + 4 = 1) ∧ (m^2 - 6 * m + 8 > 0) ↔ m = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_function_increasing_iff_m_eq_1_l759_75962
