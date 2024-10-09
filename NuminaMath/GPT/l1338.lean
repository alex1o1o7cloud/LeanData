import Mathlib

namespace problem_l1338_133876

noncomputable def cubeRoot (x : ℝ) : ℝ :=
  x ^ (1 / 3)

theorem problem (t : ℝ) (h : t = 1 / (1 - cubeRoot 2)) :
  t = (1 + cubeRoot 2) * (1 + cubeRoot 4) :=
by
  sorry

end problem_l1338_133876


namespace evaluate_expression_l1338_133873

theorem evaluate_expression : (-1 : ℤ)^(3^3) + (1 : ℤ)^(3^3) = 0 := 
by
  sorry

end evaluate_expression_l1338_133873


namespace honey_last_nights_l1338_133872

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end honey_last_nights_l1338_133872


namespace cheolsu_initial_number_l1338_133816

theorem cheolsu_initial_number (x : ℚ) (h : x + (-5/12) - (-5/2) = 1/3) : x = -7/4 :=
by 
  sorry

end cheolsu_initial_number_l1338_133816


namespace probability_of_selecting_one_of_each_color_l1338_133886

noncomputable def number_of_ways_to_select_4_marbles_from_10 := Nat.choose 10 4
noncomputable def ways_to_select_1_red := Nat.choose 3 1
noncomputable def ways_to_select_1_blue := Nat.choose 3 1
noncomputable def ways_to_select_1_green := Nat.choose 2 1
noncomputable def ways_to_select_1_yellow := Nat.choose 2 1

theorem probability_of_selecting_one_of_each_color :
  (ways_to_select_1_red * ways_to_select_1_blue * ways_to_select_1_green * ways_to_select_1_yellow) / number_of_ways_to_select_4_marbles_from_10 = 6 / 35 :=
by
  sorry

end probability_of_selecting_one_of_each_color_l1338_133886


namespace find_value_of_a_l1338_133888

-- Given conditions
def equation1 (x y : ℝ) : Prop := 4 * y + x + 5 = 0
def equation2 (x y : ℝ) (a : ℝ) : Prop := 3 * y + a * x + 4 = 0

-- The proof problem statement
theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y a → a = -12) :=
sorry

end find_value_of_a_l1338_133888


namespace initial_length_proof_l1338_133869

variables (L : ℕ)

-- Conditions from the problem statement
def condition1 (L : ℕ) : Prop := L - 25 > 118
def condition2 : Prop := 125 - 7 = 118
def initial_length : Prop := L = 143

-- Proof statement
theorem initial_length_proof (L : ℕ) (h1 : condition1 L) (h2 : condition2) : initial_length L :=
sorry

end initial_length_proof_l1338_133869


namespace units_digit_of_27_mul_36_l1338_133855

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l1338_133855


namespace ceiling_lights_l1338_133883

variable (S M L : ℕ)

theorem ceiling_lights (hM : M = 12) (hL : L = 2 * M)
    (hBulbs : S + 2 * M + 3 * L = 118) : S - M = 10 :=
by
  sorry

end ceiling_lights_l1338_133883


namespace max_hours_wednesday_l1338_133821

theorem max_hours_wednesday (x : ℕ) 
    (h1 : ∀ (d w : ℕ), w = x → d = x → d + w + (x + 3) = 3 * 3) 
    (h2 : ∀ (a b c : ℕ), a = b → b = c → (a + b + (c + 3))/3 = 3) :
  x = 2 := 
by
  sorry

end max_hours_wednesday_l1338_133821


namespace find_value_l1338_133864

theorem find_value 
    (x y : ℝ) 
    (hx : x = 1 / (Real.sqrt 2 + 1)) 
    (hy : y = 1 / (Real.sqrt 2 - 1)) : 
    x^2 - 3 * x * y + y^2 = 3 := 
by 
    sorry

end find_value_l1338_133864


namespace f_96_value_l1338_133827

noncomputable def f : ℕ → ℕ :=
sorry

axiom condition_1 (a b : ℕ) : 
  f (a * b) = f a + f b

axiom condition_2 (n : ℕ) (hp : Nat.Prime n) (hlt : 10 < n) : 
  f n = 0

axiom condition_3 : 
  f 1 < f 243 ∧ f 243 < f 2 ∧ f 2 < 11

axiom condition_4 : 
  f 2106 < 11

theorem f_96_value :
  f 96 = 31 :=
sorry

end f_96_value_l1338_133827


namespace bus_departure_l1338_133832

theorem bus_departure (current_people : ℕ) (min_people : ℕ) (required_people : ℕ) 
  (h1 : current_people = 9) (h2 : min_people = 16) : required_people = 7 :=
by 
  sorry

end bus_departure_l1338_133832


namespace sufficient_budget_for_kvass_l1338_133836

variables (x y : ℝ)

theorem sufficient_budget_for_kvass (h1 : x + y = 1) (h2 : 0.6 * x + 1.2 * y = 1) : 
  3 * y ≥ 1.44 * y :=
by
  sorry

end sufficient_budget_for_kvass_l1338_133836


namespace opposite_of_fraction_l1338_133820

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end opposite_of_fraction_l1338_133820


namespace people_at_first_concert_l1338_133837

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l1338_133837


namespace element_of_sequence_l1338_133809

/-
Proving that 63 is an element of the sequence defined by aₙ = n² + 2n.
-/
theorem element_of_sequence (n : ℕ) (h : 63 = n^2 + 2 * n) : ∃ n : ℕ, 63 = n^2 + 2 * n :=
by
  sorry

end element_of_sequence_l1338_133809


namespace convert_speed_l1338_133858

theorem convert_speed (v_m_s : ℚ) (conversion_factor : ℚ) :
  v_m_s = 12 / 43 → conversion_factor = 3.6 → v_m_s * conversion_factor = 1.0046511624 := by
  intros h1 h2
  have h3 : v_m_s = 12 / 43 := h1
  have h4 : conversion_factor = 3.6 := h2
  rw [h3, h4]
  norm_num
  sorry

end convert_speed_l1338_133858


namespace koi_fish_after_three_weeks_l1338_133887

theorem koi_fish_after_three_weeks
  (f_0 : ℕ := 280) -- initial total number of fish
  (days : ℕ := 21) -- days in 3 weeks
  (koi_added_per_day : ℕ := 2)
  (goldfish_added_per_day : ℕ := 5)
  (goldfish_after_3_weeks : ℕ := 200) :
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  koi_after_3_weeks = 227 :=
by
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  sorry

end koi_fish_after_three_weeks_l1338_133887


namespace linear_function_through_origin_l1338_133840

theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 1) * x + m ^ 2 - 1) → (x = 0 ∧ y = 0) → m = -1) :=
sorry

end linear_function_through_origin_l1338_133840


namespace paving_stone_length_l1338_133815

theorem paving_stone_length 
  (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (num_stones : ℕ) (stone_width : ℝ) 
  (courtyard_area : ℝ) 
  (total_stones_area : ℝ) 
  (L : ℝ) :
  courtyard_length = 50 →
  courtyard_width = 16.5 →
  num_stones = 165 →
  stone_width = 2 →
  courtyard_area = courtyard_length * courtyard_width →
  total_stones_area = num_stones * stone_width * L →
  courtyard_area = total_stones_area →
  L = 2.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end paving_stone_length_l1338_133815


namespace common_difference_l1338_133823

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

theorem common_difference (h : a 2015 = a 2013 + 6) : 
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 := 
by
  use 3
  sorry

end common_difference_l1338_133823


namespace solution_of_equation_l1338_133826

theorem solution_of_equation (a b c : ℕ) :
    a^(b + 20) * (c - 1) = c^(b + 21) - 1 ↔ 
    (∃ b' : ℕ, b = b' ∧ a = 1 ∧ c = 0) ∨ 
    (∃ a' b' : ℕ, a = a' ∧ b = b' ∧ c = 1) :=
by sorry

end solution_of_equation_l1338_133826


namespace find_inverse_of_512_l1338_133812

-- Define the function f with the given properties
def f : ℕ → ℕ := sorry

axiom f_initial : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

-- State the problem as a theorem
theorem find_inverse_of_512 : ∃ x, f x = 512 ∧ x = 1280 :=
by 
  -- Sorry to skip the proof
  sorry

end find_inverse_of_512_l1338_133812


namespace trajectory_of_center_l1338_133889

-- Define the given conditions
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

def tangent_y_axis (x : ℝ) : Prop := x = 0

-- Define the theorem with the given conditions and the desired conclusion
theorem trajectory_of_center (x y : ℝ) (h1 : tangent_circle x y) (h2 : tangent_y_axis x) :
  (y^2 = 8 * x) ∨ (y = 0 ∧ x ≤ 0) :=
sorry

end trajectory_of_center_l1338_133889


namespace interval_of_a_l1338_133894

theorem interval_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_monotone : ∀ x y, x < y → f y ≤ f x)
  (h_condition : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) : 
  a ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 5 :=
by
  sorry

end interval_of_a_l1338_133894


namespace heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l1338_133853

namespace PolygonColoring

/-- Define a regular n-gon and its coloring -/
def regular_ngon (n : ℕ) : Type := sorry

def isosceles_triangle {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

def same_color {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

/-- Part (a) statement -/
theorem heptagon_isosceles_triangle_same_color : 
  ∀ (p : regular_ngon 7), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (b) statement -/
theorem octagon_no_isosceles_triangle_same_color :
  ∃ (p : regular_ngon 8), ¬∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (c) statement -/
theorem general_ngon_isosceles_triangle_same_color :
  ∀ (n : ℕ), (n = 5 ∨ n = 7 ∨ n ≥ 9) → 
  ∀ (p : regular_ngon n), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

end PolygonColoring

end heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l1338_133853


namespace part1_part2_l1338_133877

section

variable (a x : ℝ)

def A : Set ℝ := { x | x ≤ -1 } ∪ { x | x ≥ 5 }
def B (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 2 }

-- Part 1
theorem part1 (h : a = -1) :
  B a = { x | -2 ≤ x ∧ x ≤ 1 } ∧
  (A ∩ B a) = { x | -2 ≤ x ∧ x ≤ -1 } ∧
  (A ∪ B a) = { x | x ≤ 1 ∨ x ≥ 5 } := 
sorry

-- Part 2
theorem part2 (h : A ∩ B a = B a) :
  a ≤ -3 ∨ a > 2 := 
sorry

end

end part1_part2_l1338_133877


namespace circle_equation_l1338_133801

theorem circle_equation (x y : ℝ) :
  let C := (4, -6)
  let r := 4
  (x - C.1)^2 + (y - C.2)^2 = r^2 →
  (x - 4)^2 + (y + 6)^2 = 16 :=
by
  intros
  sorry

end circle_equation_l1338_133801


namespace no_maximal_radius_of_inscribed_cylinder_l1338_133885

theorem no_maximal_radius_of_inscribed_cylinder
  (base_radius_cone : ℝ) (height_cone : ℝ)
  (h_base_radius : base_radius_cone = 5) (h_height : height_cone = 10) :
  ¬ ∃ r : ℝ, 0 < r ∧ r < 5 ∧
    ∀ t : ℝ, 0 < t ∧ t < 5 → 2 * Real.pi * (10 * r - r ^ 2) ≥ 2 * Real.pi * (10 * t - t ^ 2) :=
by
  sorry

end no_maximal_radius_of_inscribed_cylinder_l1338_133885


namespace find_initial_quarters_l1338_133824

variables {Q : ℕ} -- Initial number of quarters

def quarters_to_dollars (q : ℕ) : ℝ := q * 0.25

noncomputable def initial_cash : ℝ := 40
noncomputable def cash_given_to_sister : ℝ := 5
noncomputable def quarters_given_to_sister : ℕ := 120
noncomputable def remaining_total : ℝ := 55

theorem find_initial_quarters (Q : ℕ) (h1 : quarters_to_dollars Q + 40 = 90) : Q = 200 :=
by { sorry }

end find_initial_quarters_l1338_133824


namespace adam_completes_work_in_10_days_l1338_133861

theorem adam_completes_work_in_10_days (W : ℝ) (A : ℝ)
  (h1 : (W / 25) + A = W / 20) :
  W / 10 = (W / 100) * 10 :=
by
  sorry

end adam_completes_work_in_10_days_l1338_133861


namespace milk_cartons_total_l1338_133810

theorem milk_cartons_total (regular_milk soy_milk : ℝ) (h1 : regular_milk = 0.5) (h2 : soy_milk = 0.1) :
  regular_milk + soy_milk = 0.6 :=
by
  rw [h1, h2]
  norm_num

end milk_cartons_total_l1338_133810


namespace remainder_of_poly_div_l1338_133849

theorem remainder_of_poly_div (n : ℕ) (h : n > 2) : (n^3 + 3) % (n + 1) = 2 :=
by 
  sorry

end remainder_of_poly_div_l1338_133849


namespace arith_seq_a12_value_l1338_133899

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ (a₄ : ℝ), a 4 = 1 ∧ a 7 = a 4 + 3 * d ∧ a 9 = a 4 + 5 * d

theorem arith_seq_a12_value
  (h₁ : arithmetic_sequence a (13 / 8))
  (h₂ : a 7 + a 9 = 15)
  (h₃ : a 4 = 1) :
  a 12 = 14 :=
sorry

end arith_seq_a12_value_l1338_133899


namespace quadratic_coefficient_nonzero_l1338_133856

theorem quadratic_coefficient_nonzero (a : ℝ) (x : ℝ) :
  (a - 3) * x^2 - 3 * x - 4 = 0 → a ≠ 3 :=
sorry

end quadratic_coefficient_nonzero_l1338_133856


namespace n_must_be_even_l1338_133895

open Nat

-- Define the system of equations:
def equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → (-x (i-1) + 2 * x i - x (i+1) = 1)) ∧
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)

-- Define the last equation separately due to its unique form:
def last_equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (n ≥ 1 → -x (n-1) + 2 * x n = 1)

-- The theorem to prove that n must be even:
theorem n_must_be_even (n : ℕ) (x : ℕ → ℤ) : 
  equation n x → last_equation n x → Even n :=
by
  intros h₁ h₂
  sorry

end n_must_be_even_l1338_133895


namespace opposite_of_negative_five_l1338_133807

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end opposite_of_negative_five_l1338_133807


namespace tom_gas_spending_l1338_133878

-- Defining the conditions given in the problem
def miles_per_gallon := 50
def miles_per_day := 75
def gas_price := 3
def number_of_days := 10

-- Defining the main theorem to be proven
theorem tom_gas_spending : 
  (miles_per_day * number_of_days) / miles_per_gallon * gas_price = 45 := 
by 
  sorry

end tom_gas_spending_l1338_133878


namespace kids_stay_home_correct_l1338_133857

def total_number_of_kids : ℕ := 1363293
def kids_who_go_to_camp : ℕ := 455682
def kids_staying_home : ℕ := total_number_of_kids - kids_who_go_to_camp

theorem kids_stay_home_correct :
  kids_staying_home = 907611 := by 
  sorry

end kids_stay_home_correct_l1338_133857


namespace monotonic_f_iff_l1338_133882

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a * x + 5 else 1 + 1 / x

theorem monotonic_f_iff {a : ℝ} :  
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end monotonic_f_iff_l1338_133882


namespace assembly_time_constants_l1338_133859

theorem assembly_time_constants (a b : ℕ) (f : ℕ → ℝ)
  (h1 : ∀ x, f x = if x < b then a / (Real.sqrt x) else a / (Real.sqrt b))
  (h2 : f 4 = 15)
  (h3 : f b = 10) :
  a = 30 ∧ b = 9 :=
by
  sorry

end assembly_time_constants_l1338_133859


namespace Ronald_eggs_initially_l1338_133814

def total_eggs_shared (friends eggs_per_friend : Nat) : Nat :=
  friends * eggs_per_friend

theorem Ronald_eggs_initially (eggs : Nat) (candies : Nat) (friends : Nat) (eggs_per_friend : Nat)
  (h1 : friends = 8) (h2 : eggs_per_friend = 2) (h_share : total_eggs_shared friends eggs_per_friend = 16) :
  eggs = 16 := by
  sorry

end Ronald_eggs_initially_l1338_133814


namespace martha_pins_l1338_133817

theorem martha_pins (k : ℕ) :
  (2 + 9 * k > 45) ∧ (2 + 14 * k < 90) ↔ (k = 5 ∨ k = 6) :=
by
  sorry

end martha_pins_l1338_133817


namespace total_arrangements_l1338_133866

-- Question: 
-- Given 6 teachers and 4 schools with specific constraints, 
-- prove that the number of different ways to arrange the teachers is 240.

def teachers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def schools : List Nat := [1, 2, 3, 4]

def B_and_D_in_same_school (assignment: Char → Nat) : Prop :=
  assignment 'B' = assignment 'D'

def each_school_has_at_least_one_teacher (assignment: Char → Nat) : Prop :=
  ∀ s ∈ schools, ∃ t ∈ teachers, assignment t = s

noncomputable def num_arrangements : Nat := sorry -- This would actually involve complex combinatorial calculations

theorem total_arrangements : num_arrangements = 240 :=
  sorry

end total_arrangements_l1338_133866


namespace general_formula_seq_arithmetic_l1338_133822

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Conditions from the problem
axiom sum_condition (n : ℕ) : (1 - q) * S n + q^n = 1
axiom nonzero_q : q * (q - 1) ≠ 0
axiom arithmetic_S : S 3 + S 9 = 2 * S 6

-- Stating the proof goals
theorem general_formula (n : ℕ) : a n = q^(n-1) :=
sorry

theorem seq_arithmetic : a 2 + a 8 = 2 * a 5 :=
sorry

end general_formula_seq_arithmetic_l1338_133822


namespace find_k_l1338_133898

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end find_k_l1338_133898


namespace largest_lcm_l1338_133806

theorem largest_lcm :
  max (max (max (max (max (Nat.lcm 12 2) (Nat.lcm 12 4)) 
                    (Nat.lcm 12 6)) 
                 (Nat.lcm 12 8)) 
            (Nat.lcm 12 10)) 
      (Nat.lcm 12 12) = 60 :=
by sorry

end largest_lcm_l1338_133806


namespace area_of_rectangle_given_conditions_l1338_133874

-- Defining the conditions given in the problem
variables (s d r a : ℝ)

-- Given conditions for the problem
def is_square_inscribed_in_circle (s d : ℝ) := 
  d = s * Real.sqrt 2 ∧ 
  d = 4

def is_circle_inscribed_in_rectangle (r : ℝ) :=
  r = 2

def rectangle_dimensions (length width : ℝ) :=
  length = 2 * width ∧ 
  width = 2

-- The theorem we want to prove
theorem area_of_rectangle_given_conditions :
  ∀ (s d r length width : ℝ),
  is_square_inscribed_in_circle s d →
  is_circle_inscribed_in_rectangle r →
  rectangle_dimensions length width →
  a = length * width →
  a = 8 :=
by
  intros s d r length width h1 h2 h3 h4
  sorry

end area_of_rectangle_given_conditions_l1338_133874


namespace sequence_length_arithmetic_sequence_l1338_133843

theorem sequence_length_arithmetic_sequence :
  ∀ (a d l n : ℕ), a = 5 → d = 3 → l = 119 → l = a + (n - 1) * d → n = 39 :=
by
  intros a d l n ha hd hl hln
  sorry

end sequence_length_arithmetic_sequence_l1338_133843


namespace product_of_5_consecutive_integers_divisible_by_60_l1338_133884

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l1338_133884


namespace hyperbola_asymptote_perpendicular_to_line_l1338_133854

variable {a : ℝ}

theorem hyperbola_asymptote_perpendicular_to_line (h : a > 0)
  (C : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1)
  (l : ∀ x y : ℝ, 2 * x - y + 1 = 0) :
  a = 2 :=
by
  sorry

end hyperbola_asymptote_perpendicular_to_line_l1338_133854


namespace family_vacation_days_l1338_133870

theorem family_vacation_days
  (rained_days : ℕ)
  (total_days : ℕ)
  (clear_mornings : ℕ)
  (H1 : rained_days = 13)
  (H2 : total_days = 18)
  (H3 : clear_mornings = 11) :
  total_days = 18 :=
by
  -- proof to be filled in here
  sorry

end family_vacation_days_l1338_133870


namespace cubing_identity_l1338_133808

theorem cubing_identity (x : ℂ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
  sorry

end cubing_identity_l1338_133808


namespace largest_integer_solution_of_inequality_l1338_133800

theorem largest_integer_solution_of_inequality :
  ∃ x : ℤ, x < 2 ∧ (∀ y : ℤ, y < 2 → y ≤ x) ∧ -x + 3 > 1 :=
sorry

end largest_integer_solution_of_inequality_l1338_133800


namespace vector_parallel_l1338_133871

theorem vector_parallel (x y : ℝ) (a b : ℝ × ℝ × ℝ) (h_parallel : a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ k : ℝ, a = k • b) : x + y = 6 :=
by sorry

end vector_parallel_l1338_133871


namespace stratified_sampling_model_A_l1338_133825

theorem stratified_sampling_model_A (r_A r_B r_C n x : ℕ) 
  (r_A_eq : r_A = 2) (r_B_eq : r_B = 3) (r_C_eq : r_C = 5) 
  (n_eq : n = 80) : 
  (r_A * n / (r_A + r_B + r_C) = x) -> x = 16 := 
by 
  intros h
  rw [r_A_eq, r_B_eq, r_C_eq, n_eq] at h
  norm_num at h
  exact h.symm

end stratified_sampling_model_A_l1338_133825


namespace sequence_general_term_l1338_133845

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ k ≥ 1, a (k + 1) = 2 * a k) : a n = 2 ^ (n - 1) :=
sorry

end sequence_general_term_l1338_133845


namespace numerical_puzzle_l1338_133838

noncomputable def THETA (T : ℕ) (A : ℕ) : ℕ := 1000 * T + 100 * T + 10 * T + A
noncomputable def BETA (B : ℕ) (T : ℕ) (A : ℕ) : ℕ := 1000 * B + 100 * T + 10 * T + A
noncomputable def GAMMA (Γ : ℕ) (E : ℕ) (M : ℕ) (A : ℕ) : ℕ := 10000 * Γ + 1000 * E + 100 * M + 10 * M + A

theorem numerical_puzzle
  (T : ℕ) (B : ℕ) (E : ℕ) (M : ℕ) (Γ : ℕ) (A : ℕ)
  (h1 : A = 0)
  (h2 : Γ = 1)
  (h3 : T + T = M)
  (h4 : 2 * E = M)
  (h5 : T ≠ B)
  (h6 : B ≠ E)
  (h7 : E ≠ M)
  (h8 : M ≠ Γ)
  (h9 : Γ ≠ T)
  (h10 : Γ ≠ B)
  (h11 : THETA T A + BETA B T A = GAMMA Γ E M A) :
  THETA 4 0 + BETA 5 4 0 = GAMMA 1 9 8 0 :=
by {
  sorry
}

end numerical_puzzle_l1338_133838


namespace angle_sum_at_point_l1338_133839

theorem angle_sum_at_point (x : ℝ) (h : 170 + 3 * x = 360) : x = 190 / 3 :=
by
  sorry

end angle_sum_at_point_l1338_133839


namespace outer_term_in_proportion_l1338_133867

theorem outer_term_in_proportion (a b x : ℝ) (h_ab : a * b = 1) (h_x : x = 0.2) : b = 5 :=
by
  sorry

end outer_term_in_proportion_l1338_133867


namespace num_new_students_l1338_133841

theorem num_new_students 
  (original_avg_age : ℕ) 
  (original_num_students : ℕ) 
  (new_avg_age : ℕ) 
  (age_decrease : ℕ) 
  (total_age_orginal : ℕ := original_num_students * original_avg_age) 
  (total_new_students : ℕ := (original_avg_age - age_decrease) * (original_num_students + 12))
  (x : ℕ := total_new_students - total_age_orginal) :
  original_avg_age = 40 → 
  original_num_students = 12 →
  new_avg_age = 32 →
  age_decrease = 4 →
  x = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end num_new_students_l1338_133841


namespace remainder_when_divided_by_s_minus_2_l1338_133879

noncomputable def f (s : ℤ) : ℤ := s^15 + s^2 + 3

theorem remainder_when_divided_by_s_minus_2 : f 2 = 32775 := 
by
  sorry

end remainder_when_divided_by_s_minus_2_l1338_133879


namespace sandra_total_beignets_l1338_133835

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l1338_133835


namespace students_not_making_the_cut_l1338_133819

-- Define the total number of girls, boys, and the number of students called back
def number_of_girls : ℕ := 39
def number_of_boys : ℕ := 4
def students_called_back : ℕ := 26

-- Define the total number of students trying out
def total_students : ℕ := number_of_girls + number_of_boys

-- Formulate the problem statement as a theorem
theorem students_not_making_the_cut : total_students - students_called_back = 17 := 
by 
  -- Omitted proof, just the statement
  sorry

end students_not_making_the_cut_l1338_133819


namespace range_of_m_condition_l1338_133818

theorem range_of_m_condition (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ * x₁ - 2 * m * x₁ + m - 3 = 0) 
  (h₂ : x₂ * x₂ - 2 * m * x₂ + m - 3 = 0)
  (hx₁ : x₁ > -1 ∧ x₁ < 0)
  (hx₂ : x₂ > 3) :
  m > 6 / 5 ∧ m < 3 :=
sorry

end range_of_m_condition_l1338_133818


namespace part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l1338_133844

-- Part 1: Proof that the solutions of 2x + y - 6 = 0 under positive integer constraints are (2, 2) and (1, 4)
theorem part1_positive_integer_solutions : 
  (∃ x y : ℤ, 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0) → 
  ({(x, y) | 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0} = {(2, 2), (1, 4)})
:= sorry

-- Part 2: Proof that if x = y, the value of m that satisfies the system of equations is -4
theorem part2_value_of_m (x y m : ℤ) : 
  x = y → (∃ m, (2 * x + y - 6 = 0 ∧ 2 * x - 2 * y + m * y + 8 = 0)) → m = -4
:= sorry

-- Part 3: Proof that regardless of m, there is a fixed solution (x, y) = (-4, 0) for the equation 2x - 2y + my + 8 = 0
theorem part3_fixed_solution (m : ℤ) : 
  2 * x - 2 * y + m * y + 8 = 0 → (x, y) = (-4, 0)
:= sorry

end part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l1338_133844


namespace equation_one_solution_equation_two_solution_l1338_133891

theorem equation_one_solution (x : ℕ) : 8 * (x + 1)^3 = 64 ↔ x = 1 := by 
  sorry

theorem equation_two_solution (x : ℤ) : (x + 1)^2 = 100 ↔ x = 9 ∨ x = -11 := by 
  sorry

end equation_one_solution_equation_two_solution_l1338_133891


namespace zoo_visitors_sunday_l1338_133892

-- Definitions based on conditions
def friday_visitors : ℕ := 1250
def saturday_multiplier : ℚ := 3
def sunday_decrease_percent : ℚ := 0.15

-- Assert the equivalence
theorem zoo_visitors_sunday : 
  let saturday_visitors := friday_visitors * saturday_multiplier
  let sunday_visitors := saturday_visitors * (1 - sunday_decrease_percent)
  round (sunday_visitors : ℚ) = 3188 :=
by
  sorry

end zoo_visitors_sunday_l1338_133892


namespace average_output_l1338_133846

theorem average_output (time1 time2 rate1 rate2 cogs1 cogs2 total_cogs total_time: ℝ) :
  rate1 = 20 → cogs1 = 60 → time1 = cogs1 / rate1 →
  rate2 = 60 → cogs2 = 60 → time2 = cogs2 / rate2 →
  total_cogs = cogs1 + cogs2 → total_time = time1 + time2 →
  (total_cogs / total_time = 30) :=
by
  intros hrate1 hcogs1 htime1 hrate2 hcogs2 htime2 htotalcogs htotaltime
  sorry

end average_output_l1338_133846


namespace certain_number_eq_0_08_l1338_133868

theorem certain_number_eq_0_08 (x : ℝ) (h : 1 / x = 12.5) : x = 0.08 :=
by
  sorry

end certain_number_eq_0_08_l1338_133868


namespace number_symmetry_equation_l1338_133803

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) :=
by
  sorry

end number_symmetry_equation_l1338_133803


namespace correct_propositions_identification_l1338_133852

theorem correct_propositions_identification (x y : ℝ) (h1 : x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)
    (h2 : ¬(x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0))
    (h3 : ¬(¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)))
    (h4 : (¬(x * y ≥ 0) → ¬(x ≥ 0) ∨ ¬(y ≥ 0))) :
  true :=
by
  -- Proof skipped
  sorry

end correct_propositions_identification_l1338_133852


namespace find_ratio_of_square_to_circle_radius_l1338_133863

def sector_circle_ratio (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) : Prop :=
  (R = (5 * a * sqrt2) / 2) →
  (r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) →
  (a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2))

theorem find_ratio_of_square_to_circle_radius
  (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) (h1 : R = (5 * a * sqrt2) / 2)
  (h2 : r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) :
  a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2) :=
  sorry

end find_ratio_of_square_to_circle_radius_l1338_133863


namespace player_A_wins_4_points_game_game_ends_after_5_points_l1338_133848

def prob_A_winning_when_serving : ℚ := 2 / 3
def prob_A_winning_when_B_serving : ℚ := 1 / 4
def prob_A_winning_in_4_points : ℚ := 1 / 12
def prob_game_ending_after_5_points : ℚ := 19 / 216

theorem player_A_wins_4_points_game :
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) = prob_A_winning_in_4_points := 
  sorry

theorem game_ends_after_5_points : 
  ((1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (prob_A_winning_when_serving)) + 
  ((prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (1 - prob_A_winning_when_serving)) = 
  prob_game_ending_after_5_points :=
  sorry

end player_A_wins_4_points_game_game_ends_after_5_points_l1338_133848


namespace inequality_solution_l1338_133865

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x < -2} ∪
  {x : ℝ | -2 < x ∧ x ≤ -1} ∪
  {x : ℝ | 1 ≤ x}

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x + 2)^2 ≥ 0} = solution_set := by
  sorry

end inequality_solution_l1338_133865


namespace cos_double_angle_l1338_133828

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l1338_133828


namespace original_sales_tax_percentage_l1338_133893

theorem original_sales_tax_percentage
  (current_sales_tax : ℝ := 10 / 3) -- 3 1/3% in decimal
  (difference : ℝ := 10.999999999999991) -- Rs. 10.999999999999991
  (market_price : ℝ := 6600) -- Rs. 6600
  (original_sales_tax : ℝ := 3.5) -- Expected original tax
  :  ((original_sales_tax / 100) * market_price = (current_sales_tax / 100) * market_price + difference) 
  := sorry

end original_sales_tax_percentage_l1338_133893


namespace discount_is_28_l1338_133890

-- Definitions
def price_notebook : ℕ := 15
def price_planner : ℕ := 10
def num_notebooks : ℕ := 4
def num_planners : ℕ := 8
def total_cost_with_discount : ℕ := 112

-- The original cost without discount
def original_cost : ℕ := num_notebooks * price_notebook + num_planners * price_planner

-- The discount amount
def discount_amount : ℕ := original_cost - total_cost_with_discount

-- Proof statement
theorem discount_is_28 : discount_amount = 28 := by
  sorry

end discount_is_28_l1338_133890


namespace fish_tagging_problem_l1338_133802

theorem fish_tagging_problem
  (N : ℕ) (T : ℕ)
  (h1 : N = 1250)
  (h2 : T = N / 25) :
  T = 50 :=
sorry

end fish_tagging_problem_l1338_133802


namespace melted_mixture_weight_l1338_133847

variable (zinc copper total_weight : ℝ)
variable (ratio_zinc ratio_copper : ℝ := 9 / 11)
variable (weight_zinc : ℝ := 31.5)

theorem melted_mixture_weight :
  (zinc / copper = ratio_zinc / ratio_copper) ∧ (zinc = weight_zinc) →
  (total_weight = zinc + copper) →
  total_weight = 70 := 
sorry

end melted_mixture_weight_l1338_133847


namespace cos_alpha_add_pi_over_4_l1338_133834

theorem cos_alpha_add_pi_over_4 (x y r : ℝ) (α : ℝ) (h1 : P = (3, -4)) (h2 : r = Real.sqrt (x^2 + y^2)) (h3 : x / r = Real.cos α) (h4 : y / r = Real.sin α) :
  Real.cos (α + Real.pi / 4) = (7 * Real.sqrt 2) / 10 := by
  sorry

end cos_alpha_add_pi_over_4_l1338_133834


namespace rationalize_denominator_l1338_133862

theorem rationalize_denominator :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by
  sorry

end rationalize_denominator_l1338_133862


namespace solve_math_problem_l1338_133850

theorem solve_math_problem (x : ℕ) (h1 : x > 0) (h2 : x % 3 = 0) (h3 : x % x = 9) : x = 30 := by
  sorry

end solve_math_problem_l1338_133850


namespace simplify_expression_l1338_133880
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are available

-- Define the main problem as a theorem
theorem simplify_expression (t : ℝ) : 
  (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end simplify_expression_l1338_133880


namespace ratio_of_numbers_l1338_133829

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l1338_133829


namespace find_length_of_shop_l1338_133813

noncomputable def length_of_shop (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / annual_rent_per_sqft / width

theorem find_length_of_shop
  (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ)
  (h_monthly_rent : monthly_rent = 3600)
  (h_width : width = 20)
  (h_annual_rent_per_sqft : annual_rent_per_sqft = 120) 
  : length_of_shop monthly_rent width annual_rent_per_sqft = 18 := 
sorry

end find_length_of_shop_l1338_133813


namespace simplify_expression_eq_l1338_133811

theorem simplify_expression_eq (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by 
  sorry

end simplify_expression_eq_l1338_133811


namespace greatest_prime_factor_391_l1338_133851

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end greatest_prime_factor_391_l1338_133851


namespace totalCupsOfLiquid_l1338_133830

def amountOfOil : ℝ := 0.17
def amountOfWater : ℝ := 1.17

theorem totalCupsOfLiquid : amountOfOil + amountOfWater = 1.34 := by
  sorry

end totalCupsOfLiquid_l1338_133830


namespace erik_orange_juice_count_l1338_133881

theorem erik_orange_juice_count (initial_money bread_loaves bread_cost orange_juice_cost remaining_money : ℤ)
  (h₁ : initial_money = 86)
  (h₂ : bread_loaves = 3)
  (h₃ : bread_cost = 3)
  (h₄ : orange_juice_cost = 6)
  (h₅ : remaining_money = 59) :
  (initial_money - remaining_money - (bread_loaves * bread_cost)) / orange_juice_cost = 3 :=
by
  sorry

end erik_orange_juice_count_l1338_133881


namespace julia_gold_watch_percentage_l1338_133875

def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def total_watches_before_gold : ℕ := silver_watches + bronze_watches
def total_watches_after_gold : ℕ := 88
def gold_watches : ℕ := total_watches_after_gold - total_watches_before_gold
def percentage_gold_watches : ℚ := (gold_watches : ℚ) / (total_watches_after_gold : ℚ) * 100

theorem julia_gold_watch_percentage :
  percentage_gold_watches = 9.09 := by
  sorry

end julia_gold_watch_percentage_l1338_133875


namespace circle_center_l1338_133897

theorem circle_center (x y : ℝ) : (x^2 - 6 * x + y^2 + 2 * y = 20) → (x,y) = (3,-1) :=
by {
  sorry
}

end circle_center_l1338_133897


namespace value_of_a_l1338_133804

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := (2 : ℝ) * x - y - 1 = 0

def line2 (x y a : ℝ) : Prop := (2 : ℝ) * x + (a + 1) * y + 2 = 0

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 x y) → (line2 x y a)

-- The theorem to be proved
theorem value_of_a (a : ℝ) : parallel_lines a → a = -2 :=
sorry

end value_of_a_l1338_133804


namespace average_is_five_plus_D_over_two_l1338_133896

variable (A B C D : ℝ)

def condition1 := 1001 * C - 2004 * A = 4008
def condition2 := 1001 * B + 3005 * A - 1001 * D = 6010

theorem average_is_five_plus_D_over_two (h1 : condition1 A C) (h2 : condition2 A B D) : 
  (A + B + C + D) / 4 = (5 + D) / 2 := 
by
  sorry

end average_is_five_plus_D_over_two_l1338_133896


namespace point_not_on_graph_l1338_133831

theorem point_not_on_graph : ¬ ∃ (x y : ℝ), (y = (x - 1) / (x + 2)) ∧ (x = -2) ∧ (y = 3) :=
by
  sorry

end point_not_on_graph_l1338_133831


namespace quadratic_real_roots_l1338_133805

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end quadratic_real_roots_l1338_133805


namespace find_a7_l1338_133833

-- Definitions based on given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * (a 1 - a 0)

-- Given condition in Lean statement
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 11 = 22

-- Proof problem
theorem find_a7 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : a 7 = 11 := 
  sorry

end find_a7_l1338_133833


namespace Jackson_to_Williams_Ratio_l1338_133860

-- Define the amounts of money Jackson and Williams have, given the conditions.
def JacksonMoney : ℤ := 125
def TotalMoney : ℤ := 150
-- Define Williams' money based on the given conditions.
def WilliamsMoney : ℤ := TotalMoney - JacksonMoney

-- State the theorem that the ratio of Jackson's money to Williams' money is 5:1
theorem Jackson_to_Williams_Ratio : JacksonMoney / WilliamsMoney = 5 := 
by
  -- Proof steps are omitted as per the instruction.
  sorry

end Jackson_to_Williams_Ratio_l1338_133860


namespace red_blue_beads_ratio_l1338_133842

-- Definitions based on the conditions
def has_red_beads (betty : Type) := betty → ℕ
def has_blue_beads (betty : Type) := betty → ℕ

def betty : Type := Unit

-- Given conditions
def num_red_beads : has_red_beads betty := λ _ => 30
def num_blue_beads : has_blue_beads betty := λ _ => 20
def red_to_blue_ratio := 3 / 2

-- Theorem to prove the ratio
theorem red_blue_beads_ratio (R B: ℕ) (h_red : R = 30) (h_blue : B = 20) :
  (R / gcd R B) / (B / gcd R B ) = red_to_blue_ratio :=
by sorry

end red_blue_beads_ratio_l1338_133842
