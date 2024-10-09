import Mathlib

namespace range_of_m_for_roots_greater_than_1_l1498_149804

theorem range_of_m_for_roots_greater_than_1:
  ∀ m : ℝ, 
  (∀ x : ℝ, 8 * x^2 - (m - 1) * x + (m - 7) = 0 → 1 < x) ↔ 25 ≤ m :=
by
  sorry

end range_of_m_for_roots_greater_than_1_l1498_149804


namespace evaluate_expression_l1498_149899

theorem evaluate_expression :
  (3 ^ (1 ^ (0 ^ 8)) + ( (3 ^ 1) ^ 0 ) ^ 8) = 4 :=
by
  sorry

end evaluate_expression_l1498_149899


namespace age_relation_l1498_149822

/--
Given that a woman is 42 years old and her daughter is 8 years old,
prove that in 9 years, the mother will be three times as old as her daughter.
-/
theorem age_relation (x : ℕ) (mother_age daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) 
  (h3 : 42 + x = 3 * (8 + x)) : 
  x = 9 :=
by
  sorry

end age_relation_l1498_149822


namespace roots_polynomial_sum_squares_l1498_149833

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end roots_polynomial_sum_squares_l1498_149833


namespace power_function_value_l1498_149805

-- Given conditions
def f : ℝ → ℝ := fun x => x^(1 / 3)

theorem power_function_value :
  f (Real.log 5 / (Real.log 2 * 8) + Real.log 160 / (Real.log (1 / 2))) = -2 := by
  sorry

end power_function_value_l1498_149805


namespace journey_distance_l1498_149810

theorem journey_distance 
  (T : ℝ) 
  (s1 s2 s3 : ℝ) 
  (hT : T = 36) 
  (hs1 : s1 = 21)
  (hs2 : s2 = 45)
  (hs3 : s3 = 24) : ∃ (D : ℝ), D = 972 :=
  sorry

end journey_distance_l1498_149810


namespace gcd_16_12_eq_4_l1498_149845

theorem gcd_16_12_eq_4 : Int.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_eq_4_l1498_149845


namespace sector_area_l1498_149883

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) : 
    (1/2 * r^2 * θ) = Real.pi :=
by
  sorry

end sector_area_l1498_149883


namespace min_value_f_when_a_is_zero_inequality_holds_for_f_l1498_149880

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Problem (1): Prove the minimum value of f(x) when a = 0 is 2 - 2 * ln 2.
theorem min_value_f_when_a_is_zero : 
  (∃ x : ℝ, f x 0 = 2 - 2 * Real.log 2) :=
sorry

-- Problem (2): Prove that for a < (exp(1) / 2) - 1, f(x) > (exp(1) / 2) - 1 for all x in (0, +∞).
theorem inequality_holds_for_f :
  ∀ a : ℝ, a < (Real.exp 1) / 2 - 1 → 
  ∀ x : ℝ, 0 < x → f x a > (Real.exp 1) / 2 - 1 :=
sorry

end min_value_f_when_a_is_zero_inequality_holds_for_f_l1498_149880


namespace arccos_neg_half_eq_two_pi_over_three_l1498_149820

theorem arccos_neg_half_eq_two_pi_over_three :
  Real.arccos (-1/2) = 2 * Real.pi / 3 := sorry

end arccos_neg_half_eq_two_pi_over_three_l1498_149820


namespace smallest_integer_satisfies_inequality_l1498_149882

theorem smallest_integer_satisfies_inequality :
  ∃ (x : ℤ), (x^2 < 2 * x + 3) ∧ ∀ (y : ℤ), (y^2 < 2 * y + 3) → x ≤ y ∧ x = 0 :=
sorry

end smallest_integer_satisfies_inequality_l1498_149882


namespace acute_angle_tan_eq_one_l1498_149827

theorem acute_angle_tan_eq_one (A : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 :=
by
  sorry

end acute_angle_tan_eq_one_l1498_149827


namespace union_M_N_intersection_M_complement_N_l1498_149802

open Set

variable (U : Set ℝ) (M N : Set ℝ)

-- Define the universal set
def is_universal_set (U : Set ℝ) : Prop :=
  U = univ

-- Define the set M
def is_set_M (M : Set ℝ) : Prop :=
  M = {x | ∃ y, y = (x - 2).sqrt}  -- or equivalently x ≥ 2

-- Define the set N
def is_set_N (N : Set ℝ) : Prop :=
  N = {x | x < 1 ∨ x > 3}

-- Define the complement of N in U
def complement_set_N (U N : Set ℝ) : Set ℝ :=
  U \ N

-- Prove M ∪ N = {x | x < 1 ∨ x ≥ 2}
theorem union_M_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∪ N = {x | x < 1 ∨ x ≥ 2} :=
  sorry

-- Prove M ∩ (complement of N in U) = {x | 2 ≤ x ≤ 3}
theorem intersection_M_complement_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∩ (complement_set_N U N) = {x | 2 ≤ x ∧ x ≤ 3} :=
  sorry

end union_M_N_intersection_M_complement_N_l1498_149802


namespace length_of_third_wall_l1498_149821

-- Define the dimensions of the first two walls
def wall1_length : ℕ := 30
def wall1_height : ℕ := 12
def wall1_area : ℕ := wall1_length * wall1_height

def wall2_length : ℕ := 30
def wall2_height : ℕ := 12
def wall2_area : ℕ := wall2_length * wall2_height

-- Total area needed
def total_area_needed : ℕ := 960

-- Calculate the area for the third wall
def two_walls_area : ℕ := wall1_area + wall2_area
def third_wall_area : ℕ := total_area_needed - two_walls_area

-- Height of the third wall
def third_wall_height : ℕ := 12

-- Calculate the length of the third wall
def third_wall_length : ℕ := third_wall_area / third_wall_height

-- Final claim: Length of the third wall is 20 feet
theorem length_of_third_wall : third_wall_length = 20 := by
  sorry

end length_of_third_wall_l1498_149821


namespace solution_to_system_l1498_149858

theorem solution_to_system :
  ∀ (x y z : ℝ), 
  x * (3 * y^2 + 1) = y * (y^2 + 3) →
  y * (3 * z^2 + 1) = z * (z^2 + 3) →
  z * (3 * x^2 + 1) = x * (x^2 + 3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end solution_to_system_l1498_149858


namespace population_proof_l1498_149898

def population (tosses : ℕ) (values : ℕ) : Prop :=
  (tosses = 7768) ∧ (values = 6)

theorem population_proof : 
  population 7768 6 :=
by
  unfold population
  exact And.intro rfl rfl

end population_proof_l1498_149898


namespace find_y_l1498_149873

theorem find_y (t : ℝ) (x : ℝ := 3 - 2 * t) (y : ℝ := 5 * t + 6) (h : x = 1) : y = 11 :=
by
  sorry

end find_y_l1498_149873


namespace total_area_of_squares_l1498_149895

theorem total_area_of_squares (x : ℝ) (hx : 4 * x^2 = 240) : 
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  2 * small_square_area + large_square_area = 360 :=
by
  let small_square_area := x^2
  let large_square_area := (2 * x)^2
  sorry

end total_area_of_squares_l1498_149895


namespace polynomial_expansion_l1498_149851

theorem polynomial_expansion (x : ℝ) :
  (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 :=
by sorry

end polynomial_expansion_l1498_149851


namespace no_prime_p_for_base_eqn_l1498_149846

theorem no_prime_p_for_base_eqn (p : ℕ) (hp: p.Prime) :
  let f (p : ℕ) := 1009 * p^3 + 307 * p^2 + 115 * p + 126 + 7
  let g (p : ℕ) := 143 * p^2 + 274 * p + 361
  f p = g p → false :=
sorry

end no_prime_p_for_base_eqn_l1498_149846


namespace product_of_coefficients_l1498_149890

theorem product_of_coefficients (b c : ℤ)
  (H1 : ∀ r, r^2 - 2 * r - 1 = 0 → r^5 - b * r - c = 0):
  b * c = 348 :=
by
  -- Solution steps would go here
  sorry

end product_of_coefficients_l1498_149890


namespace solve_for_x_l1498_149887

theorem solve_for_x (x y : ℚ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 :=
by
  intro h
  sorry

end solve_for_x_l1498_149887


namespace joseph_power_cost_ratio_l1498_149839

theorem joseph_power_cost_ratio
  (electric_oven_cost : ℝ)
  (total_cost : ℝ)
  (water_heater_cost : ℝ)
  (refrigerator_cost : ℝ)
  (H1 : electric_oven_cost = 500)
  (H2 : 2 * water_heater_cost = electric_oven_cost)
  (H3 : refrigerator_cost + water_heater_cost + electric_oven_cost = total_cost)
  (H4 : total_cost = 1500):
  (refrigerator_cost / water_heater_cost) = 3 := sorry

end joseph_power_cost_ratio_l1498_149839


namespace quiz_sum_correct_l1498_149855

theorem quiz_sum_correct (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (h_sub : x - y = 4) (h_mul : x * y = 104) :
  x + y = 20 := by
  sorry

end quiz_sum_correct_l1498_149855


namespace EM_parallel_AC_l1498_149832

-- Define the points A, B, C, D, E, and M
variables (A B C D E M : Type) 

-- Define the conditions described in the problem
variables {x y : Real}

-- Given that ABCD is an isosceles trapezoid with AB parallel to CD and AB > CD
variable (isosceles_trapezoid : Prop)

-- E is the foot of the perpendicular from D to AB
variable (foot_perpendicular : Prop)

-- M is the midpoint of BD
variable (midpoint : Prop)

-- We need to prove that EM is parallel to AC
theorem EM_parallel_AC (h1 : isosceles_trapezoid) (h2 : foot_perpendicular) (h3 : midpoint) : Prop := sorry

end EM_parallel_AC_l1498_149832


namespace number_of_ways_to_choose_books_l1498_149860

def num_books := 15
def books_to_choose := 3

theorem number_of_ways_to_choose_books : Nat.choose num_books books_to_choose = 455 := by
  sorry

end number_of_ways_to_choose_books_l1498_149860


namespace sum_of_x_and_y_l1498_149838

theorem sum_of_x_and_y (x y : ℝ) (h1 : x + abs x + y = 5) (h2 : x + abs y - y = 6) : x + y = 9 / 5 :=
by
  sorry

end sum_of_x_and_y_l1498_149838


namespace determine_a_b_l1498_149878

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the first derivative of the function f
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Define the conditions given in the problem
def conditions (a b : ℝ) : Prop :=
  (f' 1 a b = 0) ∧ (f 1 a b = 10)

-- Provide the main theorem stating the required proof
theorem determine_a_b (a b : ℝ) (h : conditions a b) : a = 4 ∧ b = -11 :=
by {
  sorry
}

end determine_a_b_l1498_149878


namespace system_of_equations_solution_l1498_149825

theorem system_of_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 10) ∧ (12 * x - 8 * y = 8) ∧ (x = 14 / 9) ∧ (y = 4 / 3) :=
by
  sorry

end system_of_equations_solution_l1498_149825


namespace total_money_divided_l1498_149817

noncomputable def children_share_total (A B E : ℕ) :=
  (12 * A = 8 * B ∧ 8 * B = 6 * E ∧ A = 84) → 
  A + B + E = 378

theorem total_money_divided (A B E : ℕ) : children_share_total A B E :=
by
  intros h
  sorry

end total_money_divided_l1498_149817


namespace puppies_per_cage_l1498_149872

theorem puppies_per_cage (initial_puppies sold_puppies cages remaining_puppies puppies_per_cage : ℕ)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : cages = 3)
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 5 := by
  sorry

end puppies_per_cage_l1498_149872


namespace choose_agency_l1498_149844

variables (a : ℝ) (x : ℕ)

def cost_agency_A (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * x

def cost_agency_B (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * (x + 1) * a

theorem choose_agency (a : ℝ) (x : ℕ) : if (x = 1) then 
                                            (cost_agency_B a x ≤ cost_agency_A a x)
                                         else if (x ≥ 2) then 
                                            (cost_agency_A a x ≤ cost_agency_B a x)
                                         else
                                            true :=
by
  sorry

end choose_agency_l1498_149844


namespace largest_value_of_y_l1498_149807

theorem largest_value_of_y :
  (∃ x y : ℝ, x^2 + 3 * x * y - y^2 = 27 ∧ 3 * x^2 - x * y + y^2 = 27 ∧ y ≤ 3) → (∃ y : ℝ, y = 3) :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  -- proof steps go here
  sorry

end largest_value_of_y_l1498_149807


namespace annual_population_increase_l1498_149831

theorem annual_population_increase (x : ℝ) (initial_pop : ℝ) :
    (initial_pop * (1 + (x - 1) / 100)^3 = initial_pop * 1.124864) → x = 5.04 :=
by
  -- Provided conditions
  intros h
  -- The hypothesis conditionally establishes that this will derive to show x = 5.04
  sorry

end annual_population_increase_l1498_149831


namespace inequality_system_solution_l1498_149863

theorem inequality_system_solution {x : ℝ} (h1 : 2 * x - 1 < x + 5) (h2 : (x + 1)/3 < x - 1) : 2 < x ∧ x < 6 :=
by
  sorry

end inequality_system_solution_l1498_149863


namespace solve_congruence_l1498_149829

theorem solve_congruence (n : ℕ) (hn : n < 47) 
  (congr_13n : 13 * n ≡ 9 [MOD 47]) : n ≡ 20 [MOD 47] :=
sorry

end solve_congruence_l1498_149829


namespace ensure_A_win_product_l1498_149893

theorem ensure_A_win_product {s : Finset ℕ} (h1 : s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : 8 ∈ s) (h3 : 5 ∈ s) :
  (4 ∈ s ∧ 6 ∈ s ∧ 7 ∈ s) →
  4 * 6 * 7 = 168 := 
by 
  intro _ 
  exact Nat.mul_assoc 4 6 7

end ensure_A_win_product_l1498_149893


namespace total_amount_received_l1498_149870

-- Definitions based on conditions
def days_A : Nat := 6
def days_B : Nat := 8
def days_ABC : Nat := 3

def share_A : Nat := 300
def share_B : Nat := 225
def share_C : Nat := 75

-- The theorem stating the total amount received for the work
theorem total_amount_received (dA dB dABC : Nat) (sA sB sC : Nat)
  (h1 : dA = days_A) (h2 : dB = days_B) (h3 : dABC = days_ABC)
  (h4 : sA = share_A) (h5 : sB = share_B) (h6 : sC = share_C) : 
  sA + sB + sC = 600 := by
  sorry

end total_amount_received_l1498_149870


namespace moles_of_NaOH_combined_l1498_149888

-- Given conditions
def moles_AgNO3 := 3
def moles_AgOH := 3
def balanced_ratio_AgNO3_NaOH := 1 -- 1:1 ratio as per the equation

-- Problem statement
theorem moles_of_NaOH_combined : 
  moles_AgOH = moles_AgNO3 → balanced_ratio_AgNO3_NaOH = 1 → 
  (∃ moles_NaOH, moles_NaOH = 3) := by
  sorry

end moles_of_NaOH_combined_l1498_149888


namespace range_of_m_l1498_149857

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * Real.sqrt 3) / 3 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l1498_149857


namespace time_saved_l1498_149837

theorem time_saved (speed_with_tide distance1 time1 distance2 time2: ℝ) 
  (h1: speed_with_tide = 5) 
  (h2: distance1 = 5) 
  (h3: time1 = 1) 
  (h4: distance2 = 40) 
  (h5: time2 = 10) : 
  time2 - (distance2 / speed_with_tide) = 2 := 
sorry

end time_saved_l1498_149837


namespace added_number_after_doubling_l1498_149869

theorem added_number_after_doubling (x y : ℤ) (h1 : x = 4) (h2 : 3 * (2 * x + y) = 51) : y = 9 :=
by
  -- proof goes here
  sorry

end added_number_after_doubling_l1498_149869


namespace number_of_pupils_not_in_programX_is_639_l1498_149864

-- Definitions for the conditions
def total_girls_elementary : ℕ := 192
def total_boys_elementary : ℕ := 135
def total_girls_middle : ℕ := 233
def total_boys_middle : ℕ := 163
def total_girls_high : ℕ := 117
def total_boys_high : ℕ := 89

def programX_girls_elementary : ℕ := 48
def programX_boys_elementary : ℕ := 28
def programX_girls_middle : ℕ := 98
def programX_boys_middle : ℕ := 51
def programX_girls_high : ℕ := 40
def programX_boys_high : ℕ := 25

-- Question formulation
theorem number_of_pupils_not_in_programX_is_639 :
  (total_girls_elementary - programX_girls_elementary) +
  (total_boys_elementary - programX_boys_elementary) +
  (total_girls_middle - programX_girls_middle) +
  (total_boys_middle - programX_boys_middle) +
  (total_girls_high - programX_girls_high) +
  (total_boys_high - programX_boys_high) = 639 := 
  by
  sorry

end number_of_pupils_not_in_programX_is_639_l1498_149864


namespace eccentricity_of_ellipse_l1498_149865

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : c = Real.sqrt (a^2 - b^2))
  (h4 : e = c / a) :
  e = 4 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l1498_149865


namespace sum_of_numbers_Carolyn_removes_l1498_149875

noncomputable def game_carolyn_paul_sum : ℕ :=
  let initial_list := [1, 2, 3, 4, 5]
  let removed_by_paul := [3, 4]
  let removed_by_carolyn := [1, 2, 5]
  removed_by_carolyn.sum

theorem sum_of_numbers_Carolyn_removes :
  game_carolyn_paul_sum = 8 :=
by
  sorry

end sum_of_numbers_Carolyn_removes_l1498_149875


namespace new_recipe_water_l1498_149824

theorem new_recipe_water (flour water sugar : ℕ)
  (h_orig : flour = 10 ∧ water = 6 ∧ sugar = 3)
  (h_new : ∀ (new_flour new_water new_sugar : ℕ), 
            new_flour = 10 ∧ new_water = 3 ∧ new_sugar = 3)
  (h_sugar : sugar = 4) :
  new_water = 4 := 
  sorry

end new_recipe_water_l1498_149824


namespace spherical_to_rectangular_l1498_149830

theorem spherical_to_rectangular :
  let ρ := 6
  let θ := 7 * Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) :=
by
  sorry

end spherical_to_rectangular_l1498_149830


namespace inequalities_indeterminate_l1498_149897

variable (s x y z : ℝ)

theorem inequalities_indeterminate (h_s : s > 0) (h_ineq : s * x > z * y) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (¬ (x > z)) ∨ (¬ (-x > -z)) ∨ (¬ (s > z / x)) ∨ (¬ (s < y / x)) :=
by sorry

end inequalities_indeterminate_l1498_149897


namespace symmetric_about_x_axis_l1498_149826

noncomputable def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

theorem symmetric_about_x_axis (a : ℝ) :
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f a x = - g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end symmetric_about_x_axis_l1498_149826


namespace tetrahedron_vertex_angle_sum_l1498_149861

theorem tetrahedron_vertex_angle_sum (A B C D : Type) (angles_at : Type → Type → Type → ℝ) :
  (∃ A, (∀ X Y Z W, X = A ∨ Y = A ∨ Z = A ∨ W = A → angles_at X Y A + angles_at Z W A > 180)) →
  ¬ (∃ A B, A ≠ B ∧ 
    (∀ X Y, X = A ∨ Y = A → angles_at X Y A + angles_at Y X A > 180) ∧ 
    (∀ X Y, X = B ∨ Y = B → angles_at X Y B + angles_at Y X B > 180)) := 
sorry

end tetrahedron_vertex_angle_sum_l1498_149861


namespace youseff_blocks_from_office_l1498_149886

def blocks_to_office (x : ℕ) : Prop :=
  let walk_time := x  -- it takes x minutes to walk
  let bike_time := (20 * x) / 60  -- it takes (20 / 60) * x = (1 / 3) * x minutes to ride a bike
  walk_time = bike_time + 4  -- walking takes 4 more minutes than biking

theorem youseff_blocks_from_office (x : ℕ) (h : blocks_to_office x) : x = 6 :=
  sorry

end youseff_blocks_from_office_l1498_149886


namespace jack_buttons_total_l1498_149814

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l1498_149814


namespace find_perfect_matching_l1498_149871

-- Define the boys and girls
inductive Boy | B1 | B2 | B3
inductive Girl | G1 | G2 | G3

-- Define the knowledge relationship
def knows : Boy → Girl → Prop
| Boy.B1, Girl.G1 => true
| Boy.B1, Girl.G2 => true
| Boy.B2, Girl.G1 => true
| Boy.B2, Girl.G3 => true
| Boy.B3, Girl.G2 => true
| Boy.B3, Girl.G3 => true
| _, _ => false

-- Proposition to prove
theorem find_perfect_matching :
  ∃ (pairing : Boy → Girl), 
    (∀ b : Boy, knows b (pairing b)) ∧ 
    (∀ g : Girl, ∃ b : Boy, pairing b = g) :=
by
  sorry

end find_perfect_matching_l1498_149871


namespace stream_speed_l1498_149868

theorem stream_speed (v : ℝ) (h1 : 36 > 0) (h2 : 80 > 0) (h3 : 40 > 0) (t_down : 80 / (36 + v) = 40 / (36 - v)) : v = 12 := 
by
  sorry

end stream_speed_l1498_149868


namespace smallest_integer_with_eight_factors_l1498_149867

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l1498_149867


namespace check_random_event_l1498_149834

def random_event (A B C D : Prop) : Prop := ∃ E, D = E

def event_A : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_B : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_C : Prop :=
  ∀ (probability : ℝ), probability = 1

def event_D : Prop :=
  ∀ (probability : ℝ), 0 < probability ∧ probability < 1

theorem check_random_event :
  random_event event_A event_B event_C event_D :=
sorry

end check_random_event_l1498_149834


namespace minimize_distance_on_ellipse_l1498_149813

theorem minimize_distance_on_ellipse (a m n : ℝ) (hQ : 0 < a ∧ a ≠ Real.sqrt 3)
  (hP : m^2 / 3 + n^2 / 2 = 1) :
  |minimize_distance| = Real.sqrt 3 ∨ |minimize_distance| = 3 * a := sorry

end minimize_distance_on_ellipse_l1498_149813


namespace sam_final_investment_l1498_149806

-- Definitions based on conditions
def initial_investment : ℝ := 10000
def first_interest_rate : ℝ := 0.20
def years_first_period : ℕ := 3
def triple_amount : ℕ := 3
def second_interest_rate : ℝ := 0.15
def years_second_period : ℕ := 1

-- Lean function to accumulate investment with compound interest
def compound_interest (P r: ℝ) (n: ℕ) : ℝ := P * (1 + r) ^ n

-- Sam's investment calculations
def amount_after_3_years : ℝ := compound_interest initial_investment first_interest_rate years_first_period
def new_investment : ℝ := triple_amount * amount_after_3_years
def final_amount : ℝ := compound_interest new_investment second_interest_rate years_second_period

-- Proof goal (statement with the proof skipped)
theorem sam_final_investment : final_amount = 59616 := by
  sorry

end sam_final_investment_l1498_149806


namespace permutation_formula_l1498_149843

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_formula (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : permutation n k = Nat.factorial n / Nat.factorial (n - k) :=
by
  unfold permutation
  sorry

end permutation_formula_l1498_149843


namespace solve_for_x2_plus_9y2_l1498_149850

variable (x y : ℝ)

def condition1 : Prop := x + 3 * y = 3
def condition2 : Prop := x * y = -6

theorem solve_for_x2_plus_9y2 (h1 : condition1 x y) (h2 : condition2 x y) :
  x^2 + 9 * y^2 = 45 :=
by
  sorry

end solve_for_x2_plus_9y2_l1498_149850


namespace trip_distance_first_part_l1498_149853

theorem trip_distance_first_part (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 70) (h3 : 32 = 70 / ((x / 48) + ((70 - x) / 24))) : x = 35 :=
by
  sorry

end trip_distance_first_part_l1498_149853


namespace unique_solution_a_exists_l1498_149894

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end unique_solution_a_exists_l1498_149894


namespace reciprocal_sum_l1498_149854

theorem reciprocal_sum :
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  1 / (a + b) = 20 / 9 :=
by
  let a := (1 / 4 : ℚ)
  let b := (1 / 5 : ℚ)
  have h : a + b = 9 / 20 := by sorry
  have h_rec : 1 / (a + b) = 20 / 9 := by sorry
  exact h_rec

end reciprocal_sum_l1498_149854


namespace second_most_eater_l1498_149812

variable (C M K B T : ℕ)  -- Assuming the quantities of food each child ate are positive integers

theorem second_most_eater
  (h1 : C > M)
  (h2 : B < K)
  (h3 : T < K)
  (h4 : K < M) :
  ∃ x, x = M ∧ (∀ y, y ≠ C → x ≥ y) ∧ (∃ z, z ≠ C ∧ z > M) :=
by {
  sorry
}

end second_most_eater_l1498_149812


namespace sin_alpha_value_l1498_149801

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end sin_alpha_value_l1498_149801


namespace real_coefficient_polynomials_with_special_roots_l1498_149891

noncomputable def P1 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) * (Polynomial.X ^ 2 - Polynomial.X + 1)
noncomputable def P2 : Polynomial ℝ := (Polynomial.X + 1) ^ 3 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2)
noncomputable def P3 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 3 * (Polynomial.X - 2)
noncomputable def P4 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 3
noncomputable def P5 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2)
noncomputable def P6 : Polynomial ℝ := (Polynomial.X + 1) * (Polynomial.X - 1 / 2) ^ 2 * (Polynomial.X - 2) ^ 2
noncomputable def P7 : Polynomial ℝ := (Polynomial.X + 1) ^ 2 * (Polynomial.X - 1 / 2) * (Polynomial.X - 2) ^ 2

theorem real_coefficient_polynomials_with_special_roots (P : Polynomial ℝ) :
  (∀ α, Polynomial.IsRoot P α → Polynomial.IsRoot P (1 - α) ∧ Polynomial.IsRoot P (1 / α)) →
  P = P1 ∨ P = P2 ∨ P = P3 ∨ P = P4 ∨ P = P5 ∨ P = P6 ∨ P = P7 :=
  sorry

end real_coefficient_polynomials_with_special_roots_l1498_149891


namespace prove_incorrect_conclusion_l1498_149818

-- Define the parabola as y = ax^2 + bx + c
def parabola_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points
def point1 (a b c : ℝ) : Prop := parabola_eq a b c (-2) = 0
def point2 (a b c : ℝ) : Prop := parabola_eq a b c (-1) = 4
def point3 (a b c : ℝ) : Prop := parabola_eq a b c 0 = 6
def point4 (a b c : ℝ) : Prop := parabola_eq a b c 1 = 6

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  point1 a b c ∧ point2 a b c ∧ point3 a b c ∧ point4 a b c

-- Define the incorrect conclusion
def incorrect_conclusion (a b c : ℝ) : Prop :=
  ¬ (parabola_eq a b c 2 = 0)

-- The statement to be proven
theorem prove_incorrect_conclusion (a b c : ℝ) (h : conditions a b c) : incorrect_conclusion a b c :=
sorry

end prove_incorrect_conclusion_l1498_149818


namespace least_possible_value_l1498_149836

noncomputable def least_value_expression (x : ℝ) : ℝ :=
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024

theorem least_possible_value : ∃ x : ℝ, least_value_expression x = 2023 :=
  sorry

end least_possible_value_l1498_149836


namespace mrs_petersons_change_l1498_149841

-- Define the conditions
def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def discount_rate : ℚ := 0.10
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

-- Formulate the proof statement
theorem mrs_petersons_change :
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * value_per_bill
  let change_received := total_amount_paid - total_cost_after_discount
  change_received = 95 := by sorry

end mrs_petersons_change_l1498_149841


namespace max_candies_l1498_149866

/-- There are 28 ones written on the board. Every minute, Karlsson erases two arbitrary numbers
and writes their sum on the board, and then eats an amount of candy equal to the product of 
the two erased numbers. Prove that the maximum number of candies he could eat in 28 minutes is 378. -/
theorem max_candies (karlsson_eats_max_candies : ℕ → ℕ → ℕ) (n : ℕ) (initial_count : n = 28) :
  (∀ a b, karlsson_eats_max_candies a b = a * b) →
  (∃ max_candies, max_candies = 378) :=
sorry

end max_candies_l1498_149866


namespace bailey_credit_cards_l1498_149811

theorem bailey_credit_cards (dog_treats : ℕ) (chew_toys : ℕ) (rawhide_bones : ℕ) (items_per_charge : ℕ) (total_items : ℕ) (credit_cards : ℕ)
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : rawhide_bones = 10)
  (h4 : items_per_charge = 5)
  (h5 : total_items = dog_treats + chew_toys + rawhide_bones)
  (h6 : credit_cards = total_items / items_per_charge) :
  credit_cards = 4 :=
by
  sorry

end bailey_credit_cards_l1498_149811


namespace ratio_Rose_to_Mother_l1498_149849

variable (Rose_age : ℕ) (Mother_age : ℕ)

-- Define the conditions
axiom sum_of_ages : Rose_age + Mother_age = 100
axiom Rose_is_25 : Rose_age = 25
axiom Mother_is_75 : Mother_age = 75

-- Define the main theorem to prove the ratio
theorem ratio_Rose_to_Mother : (Rose_age : ℚ) / (Mother_age : ℚ) = 1 / 3 := by
  sorry

end ratio_Rose_to_Mother_l1498_149849


namespace difference_in_gems_l1498_149840

theorem difference_in_gems (r d : ℕ) (h : d = 3 * r) : d - r = 2 * r := 
by 
  sorry

end difference_in_gems_l1498_149840


namespace baker_final_stock_l1498_149815

-- Given conditions as Lean definitions
def initial_cakes : Nat := 173
def additional_cakes : Nat := 103
def damaged_percentage : Nat := 25
def sold_first_day : Nat := 86
def sold_next_day_percentage : Nat := 10

-- Calculate new cakes Baker adds to the stock after accounting for damaged cakes
def new_undamaged_cakes : Nat := (additional_cakes * (100 - damaged_percentage)) / 100

-- Calculate stock after adding new cakes
def stock_after_new_cakes : Nat := initial_cakes + new_undamaged_cakes

-- Calculate stock after first day's sales
def stock_after_first_sale : Nat := stock_after_new_cakes - sold_first_day

-- Calculate cakes sold on the second day
def sold_next_day : Nat := (stock_after_first_sale * sold_next_day_percentage) / 100

-- Final stock calculations
def final_stock : Nat := stock_after_first_sale - sold_next_day

-- Prove that Baker has 148 cakes left
theorem baker_final_stock : final_stock = 148 := by
  sorry

end baker_final_stock_l1498_149815


namespace lost_card_number_l1498_149879

theorem lost_card_number (p : ℕ) (c : ℕ) (h : 0 ≤ c ∧ c ≤ 9)
  (sum_remaining_cards : 10 * p + 45 - (p + c) = 2012) : p + c = 223 := by
  sorry

end lost_card_number_l1498_149879


namespace find_q_sum_l1498_149848

variable (q : ℕ → ℕ)

def conditions :=
  q 3 = 2 ∧ 
  q 8 = 20 ∧ 
  q 16 = 12 ∧ 
  q 21 = 30

theorem find_q_sum (h : conditions q) : 
  (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + 
   q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20 + q 21 + q 22) = 352 := 
  sorry

end find_q_sum_l1498_149848


namespace real_roots_iff_integer_roots_iff_l1498_149842

noncomputable def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * k * (k - 1)

theorem real_roots_iff (k : ℝ) : 
  (discriminant k ≥ 0) ↔ (∃ (a b : ℝ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) := sorry

theorem integer_roots_iff (k : ℝ) : 
  (∃ (a b : ℤ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) ↔ 
  (k = 0 ∨ k = 1 ∨ k = -1/7) := sorry

-- These theorems need to be proven within Lean 4 itself

end real_roots_iff_integer_roots_iff_l1498_149842


namespace ratio_XZ_ZY_equals_one_l1498_149852

theorem ratio_XZ_ZY_equals_one (A : ℕ) (B : ℕ) (C : ℕ) (total_area : ℕ) (area_bisected : ℕ)
  (decagon_area : total_area = 12) (halves_area : area_bisected = 6)
  (above_LZ : A + B = area_bisected) (below_LZ : C + D = area_bisected)
  (symmetry : XZ = ZY) :
  (XZ / ZY = 1) := 
by
  sorry

end ratio_XZ_ZY_equals_one_l1498_149852


namespace ball_arrangements_l1498_149881

-- Define the structure of the boxes and balls
structure BallDistributions where
  white_balls_box1 : ℕ
  black_balls_box1 : ℕ
  white_balls_box2 : ℕ
  black_balls_box2 : ℕ
  white_balls_box3 : ℕ
  black_balls_box3 : ℕ

-- Problem conditions
def valid_distribution (d : BallDistributions) : Prop :=
  d.white_balls_box1 + d.black_balls_box1 ≥ 2 ∧
  d.white_balls_box2 + d.black_balls_box2 ≥ 2 ∧
  d.white_balls_box3 + d.black_balls_box3 ≥ 2 ∧
  d.white_balls_box1 ≥ 1 ∧
  d.black_balls_box1 ≥ 1 ∧
  d.white_balls_box2 ≥ 1 ∧
  d.black_balls_box2 ≥ 1 ∧
  d.white_balls_box3 ≥ 1 ∧
  d.black_balls_box3 ≥ 1

def total_white_balls (d : BallDistributions) : ℕ :=
  d.white_balls_box1 + d.white_balls_box2 + d.white_balls_box3

def total_black_balls (d : BallDistributions) : ℕ :=
  d.black_balls_box1 + d.black_balls_box2 + d.black_balls_box3

def correct_distribution (d : BallDistributions) : Prop :=
  total_white_balls d = 4 ∧ total_black_balls d = 5

-- Main theorem to prove
theorem ball_arrangements : ∃ (d : BallDistributions), valid_distribution d ∧ correct_distribution d ∧ (number_of_distributions = 18) :=
  sorry

end ball_arrangements_l1498_149881


namespace shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l1498_149884

noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1^2 + p.2^2 - 4*p.1 - 2*p.2) = 0 }

def point_M_in_circle : Prop :=
  (1, 0) ∈ circle_C

theorem shortest_chord_through_M_is_x_plus_y_minus_1_eq_0 :
  point_M_in_circle →
  ∃ (a b c : ℝ), a * 1 + b * 0 + c = 0 ∧
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l1498_149884


namespace tan_alpha_over_tan_beta_l1498_149803

theorem tan_alpha_over_tan_beta (α β : ℝ) (h1 : Real.sin (α + β) = 2 / 3) (h2 : Real.sin (α - β) = 1 / 3) :
  (Real.tan α / Real.tan β = 3) :=
sorry

end tan_alpha_over_tan_beta_l1498_149803


namespace cubes_difference_l1498_149808

theorem cubes_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) (h3 : a + b = 6) : a^3 - b^3 = 432.25 :=
by
  sorry

end cubes_difference_l1498_149808


namespace minimum_value_l1498_149847

theorem minimum_value (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ y_min, y_min = -2 / 7 ∧ x = 6 / 7 ∧ (x^2 + (y - 1)^2 + z^2) = 18 / 7 :=
by
  sorry

end minimum_value_l1498_149847


namespace geo_series_sum_eight_terms_l1498_149835

theorem geo_series_sum_eight_terms :
  let a_0 := 1 / 3
  let r := 1 / 3 
  let S_8 := a_0 * (1 - r^8) / (1 - r)
  S_8 = 3280 / 6561 :=
by
  /- :: Proof Steps Omitted. -/
  sorry

end geo_series_sum_eight_terms_l1498_149835


namespace playerB_hit_rate_playerA_probability_l1498_149809

theorem playerB_hit_rate (p : ℝ) (h : (1 - p)^2 = 1/16) : p = 3/4 :=
sorry

theorem playerA_probability (hit_rate : ℝ) (h : hit_rate = 1/2) : 
  (1 - (1 - hit_rate)^2) = 3/4 :=
sorry

end playerB_hit_rate_playerA_probability_l1498_149809


namespace inequality_proof_l1498_149816

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ ((a ^ 2 + b ^ 2 + c ^ 2) * (a * b + b * c + c * a)) / (a * b * c * (a + b + c)) + 3 := 
by
  -- Adding 'sorry' to indicate the proof is omitted
  sorry

end inequality_proof_l1498_149816


namespace train_length_l1498_149896

theorem train_length
  (time : ℝ) (man_speed train_speed : ℝ) (same_direction : Prop)
  (h_time : time = 62.99496040316775)
  (h_man_speed : man_speed = 6)
  (h_train_speed : train_speed = 30)
  (h_same_direction : same_direction) :
  (train_speed - man_speed) * (1000 / 3600) * time = 1259.899208063355 := 
sorry

end train_length_l1498_149896


namespace files_per_folder_l1498_149877

-- Define the conditions
def initial_files : ℕ := 43
def deleted_files : ℕ := 31
def num_folders : ℕ := 2

-- Define the final problem statement
theorem files_per_folder :
  (initial_files - deleted_files) / num_folders = 6 :=
by
  -- proof would go here
  sorry

end files_per_folder_l1498_149877


namespace parts_of_milk_in_drink_A_l1498_149876

theorem parts_of_milk_in_drink_A (x : ℝ) (h : 63 * (4 * x) / (7 * (x + 3)) = 63 * 3 / (x + 3) + 21) : x = 16.8 :=
by
  sorry

end parts_of_milk_in_drink_A_l1498_149876


namespace calc_1_calc_2_calc_3_calc_4_l1498_149823

-- Problem 1
theorem calc_1 : 26 - 7 + (-6) + 17 = 30 := 
by
  sorry

-- Problem 2
theorem calc_2 : -81 / (9 / 4) * (-4 / 9) / (-16) = -1 := 
by
  sorry

-- Problem 3
theorem calc_3 : ((2 / 3) - (3 / 4) + (1 / 6)) * (-36) = -3 := 
by
  sorry

-- Problem 4
theorem calc_4 : -1^4 + 12 / (-2)^2 + (1 / 4) * (-8) = 0 := 
by
  sorry


end calc_1_calc_2_calc_3_calc_4_l1498_149823


namespace mittens_per_box_l1498_149828

theorem mittens_per_box (total_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) 
  (h_total_boxes : total_boxes = 4) 
  (h_scarves_per_box : scarves_per_box = 2) 
  (h_total_clothing : total_clothing = 32) : 
  (total_clothing - total_boxes * scarves_per_box) / total_boxes = 6 := 
by
  -- Sorry, proof is omitted
  sorry

end mittens_per_box_l1498_149828


namespace system_solution_unique_l1498_149862

theorem system_solution_unique
  (a b m n : ℝ)
  (h1 : a * 1 + b * 2 = 10)
  (h2 : m * 1 - n * 2 = 8) :
  (a / 2 * (4 + -2) + b / 3 * (4 - -2) = 10) ∧
  (m / 2 * (4 + -2) - n / 3 * (4 - -2) = 8) := 
  by
    sorry

end system_solution_unique_l1498_149862


namespace rachel_age_when_emily_half_age_l1498_149874

theorem rachel_age_when_emily_half_age 
  (E_0 : ℕ) (R_0 : ℕ) (h1 : E_0 = 20) (h2 : R_0 = 24) 
  (age_diff : R_0 - E_0 = 4) : 
  ∃ R : ℕ, ∃ E : ℕ, E = R / 2 ∧ R = E + 4 ∧ R = 8 :=
by
  sorry

end rachel_age_when_emily_half_age_l1498_149874


namespace percent_of_y_l1498_149800

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 * y) / 20 + (3 * y) / 10) = (35/100) * y :=
by
  sorry

end percent_of_y_l1498_149800


namespace intersection_of_M_and_N_l1498_149859

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l1498_149859


namespace expected_scurried_home_mn_sum_l1498_149885

theorem expected_scurried_home_mn_sum : 
  let expected_fraction : ℚ := (1/2 + 2/3 + 3/4 + 4/5 + 5/6 + 6/7 + 7/8)
  let m : ℕ := 37
  let n : ℕ := 7
  m + n = 44 := by
  sorry

end expected_scurried_home_mn_sum_l1498_149885


namespace julie_savings_multiple_l1498_149856

theorem julie_savings_multiple (S : ℝ) (hS : 0 < S) :
  (12 * 0.25 * S) / (0.75 * S) = 4 :=
by
  sorry

end julie_savings_multiple_l1498_149856


namespace product_of_real_roots_l1498_149819

theorem product_of_real_roots (x : ℝ) (hx : x ^ (Real.log x / Real.log 5) = 5) :
  (∃ a b : ℝ, a ^ (Real.log a / Real.log 5) = 5 ∧ b ^ (Real.log b / Real.log 5) = 5 ∧ a * b = 1) :=
sorry

end product_of_real_roots_l1498_149819


namespace isosceles_right_triangle_area_l1498_149892

theorem isosceles_right_triangle_area
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : c = a * Real.sqrt 2) 
  (area : ℝ) 
  (h_area : area = 50)
  (h3 : (1/2) * a * b = area) :
  (a + b + c) / area = 0.4 + 0.2 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_area_l1498_149892


namespace mutually_exclusive_event_3_l1498_149889

-- Definitions based on the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Events based on problem conditions
def event_1 (a b : ℕ) : Prop := is_even a ∧ is_odd b ∨ is_odd a ∧ is_even b
def event_2 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_odd a ∧ is_odd b
def event_3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event_4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

-- Problem: Proving that event_3 is mutually exclusive with other events.
theorem mutually_exclusive_event_3 :
  ∀ (a b : ℕ), (event_3 a b) → ¬ (event_1 a b ∨ event_2 a b ∨ event_4 a b) :=
by
  sorry

end mutually_exclusive_event_3_l1498_149889
