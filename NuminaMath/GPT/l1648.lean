import Mathlib

namespace range_of_m_for_nonnegative_quadratic_l1648_164807

-- The statement of the proof problem in Lean
theorem range_of_m_for_nonnegative_quadratic {x m : ℝ} : 
  (∀ x, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_nonnegative_quadratic_l1648_164807


namespace quadratic_vertex_coords_l1648_164853

theorem quadratic_vertex_coords :
  ∀ x : ℝ, (y = (x-2)^2 - 1) → (2, -1) = (2, -1) :=
by
  sorry

end quadratic_vertex_coords_l1648_164853


namespace smallest_common_multiple_l1648_164836

theorem smallest_common_multiple (n : ℕ) : 
  (2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧ 1000 ≤ n ∧ n < 10000) → n = 1008 :=
by {
    sorry
}

end smallest_common_multiple_l1648_164836


namespace sides_of_nth_hexagon_l1648_164819

-- Definition of the arithmetic sequence condition.
def first_term : ℕ := 6
def common_difference : ℕ := 5

-- The function representing the n-th term of the sequence.
def num_sides (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

-- Now, we state the theorem that the n-th term equals 5n + 1.
theorem sides_of_nth_hexagon (n : ℕ) : num_sides n = 5 * n + 1 := by
  sorry

end sides_of_nth_hexagon_l1648_164819


namespace combined_resistance_parallel_l1648_164840

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5)
  (h_combined : 1 / r = 1 / x + 1 / y) : r = 20 / 9 := by
  sorry

end combined_resistance_parallel_l1648_164840


namespace gcd_228_1995_l1648_164885

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l1648_164885


namespace alice_probability_at_least_one_multiple_of_4_l1648_164852

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l1648_164852


namespace total_cost_of_shoes_before_discount_l1648_164863

theorem total_cost_of_shoes_before_discount (S J H : ℝ) (D : ℝ) (shoes jerseys hats : ℝ) :
  jerseys = 1/4 * shoes ∧
  hats = 2 * jerseys ∧
  D = 0.9 * (6 * shoes + 4 * jerseys + 3 * hats) ∧
  D = 620 →
  6 * shoes = 486.30 := by
  sorry

end total_cost_of_shoes_before_discount_l1648_164863


namespace sum_series_l1648_164808

noncomputable def f (n : ℕ) : ℝ :=
  (6 * (n : ℝ)^3 - 3 * (n : ℝ)^2 + 2 * (n : ℝ) - 1) / 
  ((n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ)^2 + (n : ℝ) + 1) * ((n : ℝ)^2 - (n : ℝ) + 1))

theorem sum_series:
  (∑' n, if h : 2 ≤ n then f n else 0) = 1 := 
by
  sorry

end sum_series_l1648_164808


namespace subset_implies_all_elements_l1648_164823

variable {U : Type}

theorem subset_implies_all_elements (P Q : Set U) (hPQ : P ⊆ Q) (hP_nonempty : P ≠ ∅) (hQ_nonempty : Q ≠ ∅) :
  ∀ x ∈ P, x ∈ Q :=
by 
  sorry

end subset_implies_all_elements_l1648_164823


namespace subset_div_chain_l1648_164868

theorem subset_div_chain (m n : ℕ) (h_m : m > 0) (h_n : n > 0) (S : Finset ℕ) (hS : S.card = (2^m - 1) * n + 1) (hS_subset : S ⊆ Finset.range (2^(m) * n + 1)) :
  ∃ (a : Fin (m+1) → ℕ), (∀ i, a i ∈ S) ∧ (∀ k : ℕ, k < m → a k ∣ a (k + 1)) :=
sorry

end subset_div_chain_l1648_164868


namespace rectangle_symmetry_l1648_164894

-- Definitions of symmetry properties
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Specific shapes
def EquilateralTriangle : Type := sorry
def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def RegularPentagon : Type := sorry

-- The theorem we want to prove
theorem rectangle_symmetry : 
  isAxisymmetric Rectangle ∧ isCentrallySymmetric Rectangle := sorry

end rectangle_symmetry_l1648_164894


namespace number_division_reduction_l1648_164800

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 24) : x = 36 := sorry

end number_division_reduction_l1648_164800


namespace probability_of_exactly_one_success_probability_of_at_least_one_success_l1648_164867

variable (PA : ℚ := 1/2)
variable (PB : ℚ := 2/5)
variable (P_A_bar : ℚ := 1 - PA)
variable (P_B_bar : ℚ := 1 - PB)

theorem probability_of_exactly_one_success :
  PA * P_B_bar + PB * P_A_bar = 1/2 :=
sorry

theorem probability_of_at_least_one_success :
  1 - (P_A_bar * P_A_bar * P_B_bar * P_B_bar) = 91/100 :=
sorry

end probability_of_exactly_one_success_probability_of_at_least_one_success_l1648_164867


namespace julie_read_yesterday_l1648_164882

variable (x : ℕ)
variable (y : ℕ := 2 * x)
variable (remaining_pages_after_two_days : ℕ := 120 - (x + y))

theorem julie_read_yesterday :
  (remaining_pages_after_two_days / 2 = 42) -> (x = 12) :=
by
  sorry

end julie_read_yesterday_l1648_164882


namespace wilted_flowers_correct_l1648_164813

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

end wilted_flowers_correct_l1648_164813


namespace pears_to_peaches_l1648_164815

-- Define the weights of pears and peaches
variables (pear peach : ℝ) 

-- Given conditions: 9 pears weigh the same as 6 peaches
axiom weight_ratio : 9 * pear = 6 * peach

-- Theorem to prove: 36 pears weigh the same as 24 peaches
theorem pears_to_peaches (h : 9 * pear = 6 * peach) : 36 * pear = 24 * peach :=
by
  sorry

end pears_to_peaches_l1648_164815


namespace point_outside_circle_l1648_164861

theorem point_outside_circle (a : ℝ) :
  (a > 1) → (a, a) ∉ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * a * p.1 + a^2 - a = 0} :=
by sorry

end point_outside_circle_l1648_164861


namespace musketeer_statements_triplets_count_l1648_164896

-- Definitions based on the conditions
def musketeers : Type := { x : ℕ // x < 3 }

def is_guilty (m : musketeers) : Prop := sorry  -- Placeholder for the property of being guilty

def statement (m1 m2 : musketeers) : Prop := sorry  -- Placeholder for the statement made by one musketeer about another

-- Condition that each musketeer makes one statement
def made_statement (m : musketeers) : Prop := sorry

-- Condition that exactly one musketeer lied
def exactly_one_lied : Prop := sorry

-- The final proof problem statement:
theorem musketeer_statements_triplets_count : ∃ n : ℕ, n = 99 :=
  sorry

end musketeer_statements_triplets_count_l1648_164896


namespace sector_angle_solution_l1648_164837

theorem sector_angle_solution (R α : ℝ) (h1 : 2 * R + α * R = 6) (h2 : (1/2) * R^2 * α = 2) : α = 1 ∨ α = 4 := 
sorry

end sector_angle_solution_l1648_164837


namespace sqrt_37_range_l1648_164817

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 :=
by
  sorry

end sqrt_37_range_l1648_164817


namespace triangle_side_AC_value_l1648_164845

theorem triangle_side_AC_value
  (AB BC : ℝ) (AC : ℕ)
  (hAB : AB = 1)
  (hBC : BC = 2007)
  (hAC_int : ∃ (n : ℕ), AC = n) :
  AC = 2007 :=
by
  sorry

end triangle_side_AC_value_l1648_164845


namespace sq_97_l1648_164877

theorem sq_97 : 97^2 = 9409 :=
by
  sorry

end sq_97_l1648_164877


namespace basic_computer_price_l1648_164871

theorem basic_computer_price (C P : ℝ) 
(h1 : C + P = 2500) 
(h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end basic_computer_price_l1648_164871


namespace min_value_frac_expr_l1648_164898

theorem min_value_frac_expr (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a < 1) (h₃ : 0 ≤ b) (h₄ : b < 1) (h₅ : 0 ≤ c) (h₆ : c < 1) :
  (1 / ((2 - a) * (2 - b) * (2 - c)) + 1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1 / 8 :=
sorry

end min_value_frac_expr_l1648_164898


namespace min_value_frac_ineq_l1648_164831

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∃ x, x = (1/a) + (2/b) ∧ x ≥ 9 :=
sorry

end min_value_frac_ineq_l1648_164831


namespace reciprocal_eq_self_l1648_164838

open Classical

theorem reciprocal_eq_self (a : ℝ) (h : a = 1 / a) : a = 1 ∨ a = -1 := 
sorry

end reciprocal_eq_self_l1648_164838


namespace fruit_weights_determined_l1648_164806

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

end fruit_weights_determined_l1648_164806


namespace sum_r_p_values_l1648_164803

def p (x : ℝ) : ℝ := |x| - 2
def r (x : ℝ) : ℝ := -|p x - 1|
def r_p (x : ℝ) : ℝ := r (p x)

theorem sum_r_p_values :
  (r_p (-4) + r_p (-3) + r_p (-2) + r_p (-1) + r_p 0 + r_p 1 + r_p 2 + r_p 3 + r_p 4) = -11 :=
by 
  -- Proof omitted
  sorry

end sum_r_p_values_l1648_164803


namespace profit_percentage_is_ten_l1648_164833

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

end profit_percentage_is_ten_l1648_164833


namespace scientific_notation_of_number_l1648_164842

def number := 460000000
def scientific_notation (n : ℕ) (s : ℝ) := s * 10 ^ n

theorem scientific_notation_of_number :
  scientific_notation 8 4.6 = number :=
sorry

end scientific_notation_of_number_l1648_164842


namespace celery_cost_l1648_164890

noncomputable def supermarket_problem
  (total_money : ℕ)
  (price_cereal discount_cereal price_bread : ℕ)
  (price_milk discount_milk price_potato num_potatoes : ℕ)
  (leftover_money : ℕ) 
  (total_cost : ℕ) 
  (cost_of_celery : ℕ) :=
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) + 
   leftover_money = total_money ∧
  total_cost = total_money - leftover_money ∧
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) = total_cost - cost_of_celery

theorem celery_cost (total_money : ℕ := 60) 
  (price_cereal : ℕ := 12) 
  (discount_cereal : ℕ := 50) 
  (price_bread : ℕ := 8) 
  (price_milk : ℕ := 10) 
  (discount_milk : ℕ := 90) 
  (price_potato : ℕ := 1) 
  (num_potatoes : ℕ := 6) 
  (leftover_money : ℕ := 26) 
  (total_cost : ℕ := 34) :
  supermarket_problem total_money price_cereal discount_cereal price_bread price_milk discount_milk price_potato num_potatoes leftover_money total_cost 5 :=
by
  sorry

end celery_cost_l1648_164890


namespace radius_of_circle_eq_zero_l1648_164883

theorem radius_of_circle_eq_zero :
  ∀ x y: ℝ, (x^2 + 8 * x + y^2 - 10 * y + 41 = 0) → (0 : ℝ) = 0 :=
by
  intros x y h
  sorry

end radius_of_circle_eq_zero_l1648_164883


namespace minimize_quadratic_l1648_164865

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end minimize_quadratic_l1648_164865


namespace intersection_point_ordinate_interval_l1648_164879

theorem intersection_point_ordinate_interval:
  ∃ m : ℤ, ∀ x : ℝ, e ^ x = 5 - x → 3 < x ∧ x < 4 :=
by sorry

end intersection_point_ordinate_interval_l1648_164879


namespace find_m_minus_n_l1648_164857

theorem find_m_minus_n (x y m n : ℤ) (h1 : x = -2) (h2 : y = 1) 
  (h3 : 3 * x + 2 * y = m) (h4 : n * x - y = 1) : m - n = -3 :=
by sorry

end find_m_minus_n_l1648_164857


namespace swimming_time_per_style_l1648_164854

theorem swimming_time_per_style (d v1 v2 v3 v4 t: ℝ) 
    (h1: d = 600) 
    (h2: v1 = 45) 
    (h3: v2 = 35) 
    (h4: v3 = 40) 
    (h5: v4 = 30)
    (h6: t = 15) 
    (h7: d / 4 = 150) 
    : (t / 4 = 3.75) :=
by
  sorry

end swimming_time_per_style_l1648_164854


namespace geometric_series_sum_l1648_164873

def first_term : ℤ := 3
def common_ratio : ℤ := -2
def last_term : ℤ := -1536
def num_terms : ℕ := 10
def sum_of_series (a r : ℤ) (n : ℕ) : ℤ := a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum :
  sum_of_series first_term common_ratio num_terms = -1023 := by
  sorry

end geometric_series_sum_l1648_164873


namespace cost_of_case_of_rolls_l1648_164820

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

end cost_of_case_of_rolls_l1648_164820


namespace limit_example_l1648_164899

theorem limit_example (ε : ℝ) (hε : 0 < ε) :
  ∃ δ : ℝ, 0 < δ ∧ 
  (∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε) :=
sorry -- The proof is not provided

end limit_example_l1648_164899


namespace gcd_poly_l1648_164844

theorem gcd_poly (k : ℕ) : Nat.gcd ((4500 * k)^2 + 11 * (4500 * k) + 40) (4500 * k + 8) = 3 := by
  sorry

end gcd_poly_l1648_164844


namespace largest_house_number_l1648_164843

theorem largest_house_number (phone_number_digits : List ℕ) (house_number_digits : List ℕ) :
  phone_number_digits = [5, 0, 4, 9, 3, 2, 6] →
  phone_number_digits.sum = 29 →
  (∀ (d1 d2 : ℕ), d1 ∈ house_number_digits → d2 ∈ house_number_digits → d1 ≠ d2) →
  house_number_digits.sum = 29 →
  house_number_digits = [9, 8, 7, 5] :=
by
  intros
  sorry

end largest_house_number_l1648_164843


namespace number_of_possible_tower_heights_l1648_164893

-- Axiom for the possible increment values when switching brick orientations
def possible_increments : Set ℕ := {4, 7}

-- Base height when all bricks contribute the smallest dimension
def base_height (num_bricks : ℕ) (smallest_side : ℕ) : ℕ :=
  num_bricks * smallest_side

-- Check if a given height can be achieved by changing orientations of the bricks
def can_achieve_height (h : ℕ) (n : ℕ) (increments : Set ℕ) : Prop :=
  ∃ m k : ℕ, h = base_height n 2 + m * 4 + k * 7

-- Final proof statement
theorem number_of_possible_tower_heights :
  (50 : ℕ) = 50 →
  (∀ k : ℕ, (100 + k * 4 <= 450) → can_achieve_height (100 + k * 4) 50 possible_increments) →
  ∃ (num_possible_heights : ℕ), num_possible_heights = 90 :=
by
  sorry

end number_of_possible_tower_heights_l1648_164893


namespace function_neither_odd_nor_even_l1648_164826

def f (x : ℝ) : ℝ := x^2 + 6 * x

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end function_neither_odd_nor_even_l1648_164826


namespace solve_equation_l1648_164880

theorem solve_equation (x : ℝ) : (x + 3)^4 + (x + 1)^4 = 82 → x = 0 ∨ x = -4 :=
by
  sorry

end solve_equation_l1648_164880


namespace rabbit_roaming_area_l1648_164827

noncomputable def rabbit_area_midpoint_long_side (r: ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

noncomputable def rabbit_area_3_ft_from_corner (R r: ℝ) : ℝ :=
  (3/4) * Real.pi * R^2 - (1/4) * Real.pi * r^2

theorem rabbit_roaming_area (r R : ℝ) (h_r_pos: 0 < r) (h_R_pos: r < R) :
  rabbit_area_3_ft_from_corner R r - rabbit_area_midpoint_long_side R = 22.75 * Real.pi :=
by
  sorry

end rabbit_roaming_area_l1648_164827


namespace l_shaped_tile_rectangle_multiple_of_8_l1648_164881

theorem l_shaped_tile_rectangle_multiple_of_8 (m n : ℕ) 
  (h : ∃ k : ℕ, 4 * k = m * n) : ∃ s : ℕ, m * n = 8 * s :=
by
  sorry

end l_shaped_tile_rectangle_multiple_of_8_l1648_164881


namespace equivalent_single_percentage_change_l1648_164841

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

end equivalent_single_percentage_change_l1648_164841


namespace vertex_of_parabola_l1648_164860

theorem vertex_of_parabola 
  (a b c : ℝ) 
  (h1 : a * 2^2 + b * 2 + c = 5)
  (h2 : -b / (2 * a) = 2) : 
  (2, 4 * a + 2 * b + c) = (2, 5) :=
by
  sorry

end vertex_of_parabola_l1648_164860


namespace final_middle_pile_cards_l1648_164855

-- Definitions based on conditions
def initial_cards_per_pile (n : ℕ) (h : n ≥ 2) := n

def left_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n - 2
def middle_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n + 2
def right_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n

def right_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n - 1
def middle_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n + 3

def left_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := n
def middle_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := (n + 3) - n

-- The proof problem to solve
theorem final_middle_pile_cards (n : ℕ) (h : n ≥ 2) : middle_pile_after_step_4 n h = 5 :=
sorry

end final_middle_pile_cards_l1648_164855


namespace medical_team_combinations_l1648_164889

-- Number of combinations function
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem medical_team_combinations :
  let maleDoctors := 6
  let femaleDoctors := 5
  let requiredMale := 2
  let requiredFemale := 1
  choose maleDoctors requiredMale * choose femaleDoctors requiredFemale = 75 :=
by
  sorry

end medical_team_combinations_l1648_164889


namespace jake_eats_papayas_in_one_week_l1648_164878

variable (J : ℕ)
variable (brother_eats : ℕ := 5)
variable (father_eats : ℕ := 4)
variable (total_papayas_in_4_weeks : ℕ := 48)

theorem jake_eats_papayas_in_one_week (h : 4 * (J + brother_eats + father_eats) = total_papayas_in_4_weeks) : J = 3 :=
by
  sorry

end jake_eats_papayas_in_one_week_l1648_164878


namespace gf_neg3_eq_1262_l1648_164856

def f (x : ℤ) : ℤ := x^3 + 6
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 2

theorem gf_neg3_eq_1262 : g (f (-3)) = 1262 := by
  sorry

end gf_neg3_eq_1262_l1648_164856


namespace josie_gift_money_l1648_164818

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

end josie_gift_money_l1648_164818


namespace minimum_value_l1648_164809

theorem minimum_value (x : ℝ) (hx : x > 0) : 4 * x^2 + 1 / x^3 ≥ 5 ∧ (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) :=
by {
  sorry
}

end minimum_value_l1648_164809


namespace problem_a2014_l1648_164824

-- Given conditions
def seq (a : ℕ → ℕ) := a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 1

-- Prove the required statement
theorem problem_a2014 (a : ℕ → ℕ) (h : seq a) : a 2014 = 2014 :=
by sorry

end problem_a2014_l1648_164824


namespace g_7_eq_98_l1648_164869

noncomputable def g : ℕ → ℝ := sorry

axiom g_0 : g 0 = 0
axiom g_1 : g 1 = 2
axiom functional_equation (m n : ℕ) (h : m ≥ n) : g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

theorem g_7_eq_98 : g 7 = 98 :=
sorry

end g_7_eq_98_l1648_164869


namespace ladder_base_distance_l1648_164805

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l1648_164805


namespace lucy_cardinals_vs_blue_jays_l1648_164834

noncomputable def day1_cardinals : ℕ := 3
noncomputable def day1_blue_jays : ℕ := 2
noncomputable def day2_cardinals : ℕ := 3
noncomputable def day2_blue_jays : ℕ := 3
noncomputable def day3_cardinals : ℕ := 4
noncomputable def day3_blue_jays : ℕ := 2

theorem lucy_cardinals_vs_blue_jays :
  (day1_cardinals + day2_cardinals + day3_cardinals) - (day1_blue_jays + day2_blue_jays + day3_blue_jays) = 3 :=
  by sorry

end lucy_cardinals_vs_blue_jays_l1648_164834


namespace smallest_x_plus_y_l1648_164850

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l1648_164850


namespace simplify_expression_l1648_164821

theorem simplify_expression (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b :=
by sorry

end simplify_expression_l1648_164821


namespace xiao_ming_arrival_time_l1648_164862

def left_home (departure_time : String) : Prop :=
  departure_time = "6:55"

def time_spent (duration : Nat) : Prop :=
  duration = 30

def arrival_time (arrival : String) : Prop :=
  arrival = "7:25"

theorem xiao_ming_arrival_time :
  left_home "6:55" → time_spent 30 → arrival_time "7:25" :=
by sorry

end xiao_ming_arrival_time_l1648_164862


namespace probability_of_2_red_1_black_l1648_164892

theorem probability_of_2_red_1_black :
  let P_red := 4 / 7
  let P_black := 3 / 7 
  let prob_RRB := P_red * P_red * P_black 
  let prob_RBR := P_red * P_black * P_red 
  let prob_BRR := P_black * P_red * P_red 
  let total_prob := 3 * prob_RRB
  total_prob = 144 / 343 :=
by
  sorry

end probability_of_2_red_1_black_l1648_164892


namespace quadratic_inequality_solution_l1648_164814

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 601 ≤ 9} = {x : ℝ | 19.25545 ≤ x ∧ x ≤ 30.74455} :=
by 
  sorry

end quadratic_inequality_solution_l1648_164814


namespace no_12_term_geometric_seq_in_1_to_100_l1648_164858

theorem no_12_term_geometric_seq_in_1_to_100 :
  ¬ ∃ (s : Fin 12 → Set ℕ),
    (∀ i, ∃ (a q : ℕ), (s i = {a * q^n | n : ℕ}) ∧ (∀ x ∈ s i, 1 ≤ x ∧ x ≤ 100)) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ i, n ∈ s i) := 
sorry

end no_12_term_geometric_seq_in_1_to_100_l1648_164858


namespace average_price_of_returned_cans_l1648_164835

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

end average_price_of_returned_cans_l1648_164835


namespace ninth_square_more_than_eighth_l1648_164851

noncomputable def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

noncomputable def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_square_more_than_eighth : (tile_count 9 - tile_count 8) = 72 :=
by sorry

end ninth_square_more_than_eighth_l1648_164851


namespace promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l1648_164810

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

end promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l1648_164810


namespace problem_a_problem_b_problem_c_problem_d_l1648_164811

variable {a b : ℝ}

theorem problem_a (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : ab ≤ 1 / 8 := sorry

theorem problem_b (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (1 / a) + (8 / b) ≥ 25 := sorry

theorem problem_c (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 := sorry

theorem problem_d (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 - b^2 > -1 / 4 := sorry

end problem_a_problem_b_problem_c_problem_d_l1648_164811


namespace total_additions_and_multiplications_l1648_164822

def f(x : ℝ) : ℝ := 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem total_additions_and_multiplications {x : ℝ} (h : x = 0.6) :
  let horner_f := ((((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x + 7)
  (horner_f = f x) ∧ (6 + 6 = 12) :=
by
  sorry

end total_additions_and_multiplications_l1648_164822


namespace no_integer_solutions_exist_l1648_164870

theorem no_integer_solutions_exist (n m : ℤ) : 
  (n ^ 2 - m ^ 2 = 250) → false := 
sorry 

end no_integer_solutions_exist_l1648_164870


namespace peter_pairs_of_pants_l1648_164839

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

end peter_pairs_of_pants_l1648_164839


namespace keun_bae_jumps_fourth_day_l1648_164828

def jumps (n : ℕ) : ℕ :=
  match n with
  | 0 => 15
  | n + 1 => 2 * jumps n

theorem keun_bae_jumps_fourth_day : jumps 3 = 120 :=
by
  sorry

end keun_bae_jumps_fourth_day_l1648_164828


namespace sample_older_employees_count_l1648_164876

-- Definitions of known quantities
def N := 400
def N_older := 160
def N_no_older := 240
def n := 50

-- The proof statement showing that the number of employees older than 45 in the sample equals 20
theorem sample_older_employees_count : 
  let proportion_older := (N_older:ℝ) / (N:ℝ)
  let n_older := proportion_older * (n:ℝ)
  n_older = 20 := by
  sorry

end sample_older_employees_count_l1648_164876


namespace correct_calculation_l1648_164886

theorem correct_calculation (a b : ℝ) : 
  ¬(3 * a + b = 3 * a * b) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬((a - b)^2 = a^2 - b^2) ∧ 
  ((-3 * a)^2 = 9 * a^2) :=
by
  sorry

end correct_calculation_l1648_164886


namespace system_of_equations_solution_l1648_164847

theorem system_of_equations_solution (x y z : ℝ) :
  x^2 - y * z = -23 ∧ y^2 - z * x = -4 ∧ z^2 - x * y = 34 →
  (x = 5 ∧ y = 6 ∧ z = 8) ∨ (x = -5 ∧ y = -6 ∧ z = -8) :=
by
  sorry

end system_of_equations_solution_l1648_164847


namespace number_of_balls_is_fifty_l1648_164812

variable (x : ℝ)
variable (h : x - 40 = 60 - x)

theorem number_of_balls_is_fifty : x = 50 :=
by
  have : 2 * x = 100 := by
    linarith
  linarith

end number_of_balls_is_fifty_l1648_164812


namespace interest_rate_for_lending_l1648_164849

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

end interest_rate_for_lending_l1648_164849


namespace smallest_number_of_cookies_proof_l1648_164888

def satisfies_conditions (a : ℕ) : Prop :=
  (a % 6 = 5) ∧ (a % 8 = 6) ∧ (a % 10 = 9) ∧ (∃ n : ℕ, a = n * n)

def smallest_number_of_cookies : ℕ :=
  2549

theorem smallest_number_of_cookies_proof :
  satisfies_conditions smallest_number_of_cookies :=
by
  sorry

end smallest_number_of_cookies_proof_l1648_164888


namespace units_digit_17_pow_27_l1648_164830

-- Define the problem: the units digit of 17^27
theorem units_digit_17_pow_27 : (17 ^ 27) % 10 = 3 :=
sorry

end units_digit_17_pow_27_l1648_164830


namespace max_x_lcm_15_21_105_l1648_164895

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l1648_164895


namespace cookie_baking_l1648_164801

/-- It takes 7 minutes to bake 1 pan of cookies. In 28 minutes, you can bake 4 pans of cookies. -/
theorem cookie_baking (bake_time_per_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) 
  (h1 : bake_time_per_pan = 7)
  (h2 : total_time = 28) : 
  num_pans = 4 := 
by
  sorry

end cookie_baking_l1648_164801


namespace find_whole_number_M_l1648_164874

theorem find_whole_number_M (M : ℕ) (h : 8 < M / 4 ∧ M / 4 < 9) : M = 33 :=
sorry

end find_whole_number_M_l1648_164874


namespace fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l1648_164887

-- Define the sequences as functions
def row1 (n : ℕ) : ℤ := (-2)^n
def row2 (n : ℕ) : ℤ := row1 n + 2
def row3 (n : ℕ) : ℤ := (-1) * (-2)^n

-- Theorems to be proven

-- (1) Prove the fourth term in row ① is 16 
theorem fourth_term_row1_is_16 : row1 4 = 16 := sorry

-- (1) Prove the nth term in row ① is (-2)^n
theorem nth_term_row1 (n : ℕ) : row1 n = (-2)^n := sorry

-- (2) Let the nth number in row ① be a, prove the nth number in row ② is a + 2
theorem nth_term_row2 (n : ℕ) : row2 n = row1 n + 2 := sorry

-- (3) If the sum of three consecutive numbers in row ③ is -192, find these numbers
theorem sum_three_consecutive_row3 : ∃ n : ℕ, row3 n + row3 (n + 1) + row3 (n + 2) = -192 ∧ 
  row3 n  = -64 ∧ row3 (n + 1) = 128 ∧ row3 (n + 2) = -256 := sorry

end fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l1648_164887


namespace max_value_of_g_is_34_l1648_164846
noncomputable def g : ℕ → ℕ
| n => if n < 15 then n + 20 else g (n - 7)

theorem max_value_of_g_is_34 : ∃ n, g n = 34 ∧ ∀ m, g m ≤ 34 :=
by
  sorry

end max_value_of_g_is_34_l1648_164846


namespace tan_double_angle_sub_l1648_164825

theorem tan_double_angle_sub (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β) = 1 / 5) : Real.tan (2 * α - β) = 7 / 9 :=
by
  sorry

end tan_double_angle_sub_l1648_164825


namespace negation_P_l1648_164884

variable (P : Prop) (P_def : ∀ x : ℝ, Real.sin x ≤ 1)

theorem negation_P : ¬P ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_P_l1648_164884


namespace prove_mouse_cost_l1648_164802

variable (M K : ℕ)

theorem prove_mouse_cost (h1 : K = 3 * M) (h2 : M + K = 64) : M = 16 :=
by
  sorry

end prove_mouse_cost_l1648_164802


namespace find_smallest_c_l1648_164897

theorem find_smallest_c (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (graph_eq : ∀ x, (a * Real.sin (b * x + c) + d) = 5 → x = (π / 6))
    (amplitude_eq : a = 3) : c = π / 2 :=
sorry

end find_smallest_c_l1648_164897


namespace sum_first_five_even_numbers_l1648_164866

theorem sum_first_five_even_numbers : (2 + 4 + 6 + 8 + 10) = 30 :=
by
  sorry

end sum_first_five_even_numbers_l1648_164866


namespace two_digit_number_representation_l1648_164848

theorem two_digit_number_representation (x : ℕ) (h : x < 10) : 10 * x + 5 < 100 :=
by sorry

end two_digit_number_representation_l1648_164848


namespace negation_universal_proposition_l1648_164829

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by
  sorry

end negation_universal_proposition_l1648_164829


namespace arccos_sqrt3_div_2_eq_pi_div_6_l1648_164872

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l1648_164872


namespace lateral_surface_area_of_parallelepiped_is_correct_l1648_164804

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

end lateral_surface_area_of_parallelepiped_is_correct_l1648_164804


namespace yangyang_helps_mom_for_5_days_l1648_164875

-- Defining the conditions
def quantity_of_rice_in_warehouses_are_same : Prop := sorry
def dad_transports_all_rice_in : ℕ := 10
def mom_transports_all_rice_in : ℕ := 12
def yangyang_transports_all_rice_in : ℕ := 15
def dad_and_mom_start_at_same_time : Prop := sorry
def yangyang_helps_mom_then_dad : Prop := sorry
def finish_transporting_at_same_time : Prop := sorry

-- The theorem to prove
theorem yangyang_helps_mom_for_5_days (h1 : quantity_of_rice_in_warehouses_are_same) 
    (h2 : dad_and_mom_start_at_same_time) 
    (h3 : yangyang_helps_mom_then_dad) 
    (h4 : finish_transporting_at_same_time) : 
    yangyang_helps_mom_then_dad :=
sorry

end yangyang_helps_mom_for_5_days_l1648_164875


namespace total_marbles_l1648_164864

variable (r : ℝ) -- number of red marbles
variable (b g y : ℝ) -- number of blue, green, and yellow marbles

-- Conditions
axiom h1 : r = 1.3 * b
axiom h2 : g = 1.5 * r
axiom h3 : y = 0.8 * g

/-- Theorem: The total number of marbles in the collection is 4.47 times the number of red marbles -/
theorem total_marbles (r b g y : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.5 * r) (h3 : y = 0.8 * g) :
  b + r + g + y = 4.47 * r :=
sorry

end total_marbles_l1648_164864


namespace calculate_sheep_l1648_164859

-- Conditions as definitions
def cows : Nat := 24
def goats : Nat := 113
def total_animals_to_transport (groups size_per_group : Nat) : Nat := groups * size_per_group
def cows_and_goats (cows goats : Nat) : Nat := cows + goats

-- The problem statement: Calculate the number of sheep such that the total number of animals matches the target.
theorem calculate_sheep
  (groups : Nat) (size_per_group : Nat) (cows goats : Nat) (transportation_total animals_present : Nat) 
  (h1 : groups = 3) (h2 : size_per_group = 48) (h3 : cows = 24) (h4 : goats = 113) 
  (h5 : animals_present = cows + goats) (h6 : transportation_total = groups * size_per_group) :
  transportation_total - animals_present = 7 :=
by 
  -- To be proven 
  sorry

end calculate_sheep_l1648_164859


namespace bill_experience_l1648_164832

theorem bill_experience (B J : ℕ) (h1 : J - 5 = 3 * (B - 5)) (h2 : J = 2 * B) : B = 10 :=
by
  sorry

end bill_experience_l1648_164832


namespace planted_fraction_l1648_164816

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

end planted_fraction_l1648_164816


namespace sharp_sharp_sharp_20_l1648_164891

def sharp (N : ℝ) : ℝ := (0.5 * N)^2 + 1

theorem sharp_sharp_sharp_20 : sharp (sharp (sharp 20)) = 1627102.64 :=
by
  sorry

end sharp_sharp_sharp_20_l1648_164891
