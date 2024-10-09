import Mathlib

namespace find_a_b_l1050_105002

theorem find_a_b (a b : ℝ) (h : (⟨1, 2⟩ * (a : ℂ) + b = ⟨0, 2⟩)) : a = 1 ∧ b = -1 := 
by
  sorry

end find_a_b_l1050_105002


namespace geometric_sequence_and_general_formula_l1050_105034

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+1) = (2/3) * a n + 2) (ha1 : a 1 = 7) : 
  ∃ r : ℝ, ∀ n, a n = r ^ (n-1) + 6 :=
sorry

end geometric_sequence_and_general_formula_l1050_105034


namespace sin_cos_identity_l1050_105068

theorem sin_cos_identity {x : Real} 
    (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
    Real.sin x ^ 12 + Real.cos x ^ 12 = 5 / 18 :=
sorry

end sin_cos_identity_l1050_105068


namespace nested_fraction_eval_l1050_105065

theorem nested_fraction_eval : (1 / (1 + (1 / (2 + (1 / (1 + (1 / 4))))))) = (14 / 19) :=
by
  sorry

end nested_fraction_eval_l1050_105065


namespace baker_cakes_left_l1050_105042

theorem baker_cakes_left (cakes_made cakes_bought : ℕ) (h1 : cakes_made = 155) (h2 : cakes_bought = 140) : cakes_made - cakes_bought = 15 := by
  sorry

end baker_cakes_left_l1050_105042


namespace scientific_notation_600_million_l1050_105066

theorem scientific_notation_600_million : (600000000 : ℝ) = 6 * 10^8 := 
by 
  -- Insert the proof here
  sorry

end scientific_notation_600_million_l1050_105066


namespace equal_real_roots_of_quadratic_eq_l1050_105094

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l1050_105094


namespace triangle_with_positive_area_l1050_105046

noncomputable def num_triangles_with_A (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) : ℕ :=
  let points_excluding_A := total_points.erase A
  let total_pairs := points_excluding_A.card.choose 2
  let collinear_pairs := 20  -- Derived from the problem; in practice this would be calculated
  total_pairs - collinear_pairs

theorem triangle_with_positive_area (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) (h : total_points.card = 25):
  num_triangles_with_A total_points A = 256 :=
by
  sorry

end triangle_with_positive_area_l1050_105046


namespace area_increase_of_square_garden_l1050_105063

theorem area_increase_of_square_garden
  (length : ℝ) (width : ℝ)
  (h_length : length = 60)
  (h_width : width = 20) :
  let perimeter := 2 * (length + width)
  let side_length := perimeter / 4
  let initial_area := length * width
  let square_area := side_length ^ 2
  square_area - initial_area = 400 :=
by
  sorry

end area_increase_of_square_garden_l1050_105063


namespace ratio_of_distances_l1050_105044

-- Define the given conditions
variables (w x y : ℕ)
variables (h1 : w > 0) -- walking speed must be positive
variables (h2 : x > 0) -- distance from home must be positive
variables (h3 : y > 0) -- distance to stadium must be positive

-- Define the two times:
-- Time taken to walk directly to the stadium
def time_walk (w y : ℕ) := y / w

-- Time taken to walk home, then bike to the stadium
def time_walk_bike (w x y : ℕ) := x / w + (x + y) / (5 * w)

-- Given that both times are equal
def times_equal (w x y : ℕ) := time_walk w y = time_walk_bike w x y

-- We want to prove that the ratio of x to y is 2/3
theorem ratio_of_distances (w x y : ℕ) (h_time_eq : times_equal w x y) : x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l1050_105044


namespace arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l1050_105059

theorem arrangement_two_rows :
  ∃ (ways : ℕ), ways = 5040 := by
  sorry

theorem arrangement_no_head_tail (A : ℕ):
  ∃ (ways : ℕ), ways = 3600 := by
  sorry

theorem arrangement_girls_together :
  ∃ (ways : ℕ), ways = 576 := by
  sorry

theorem arrangement_no_boys_next :
  ∃ (ways : ℕ), ways = 1440 := by
  sorry

end arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l1050_105059


namespace total_amount_to_be_divided_l1050_105088

theorem total_amount_to_be_divided
  (k m x : ℕ)
  (h1 : 18 * k = x)
  (h2 : 20 * m = x)
  (h3 : 13 * m = 11 * k + 1400) :
  x = 36000 := 
sorry

end total_amount_to_be_divided_l1050_105088


namespace molecular_weight_of_3_moles_HBrO3_l1050_105055

-- Definitions from the conditions
def mol_weight_H : ℝ := 1.01  -- atomic weight of H
def mol_weight_Br : ℝ := 79.90  -- atomic weight of Br
def mol_weight_O : ℝ := 16.00  -- atomic weight of O

-- Definition of molecular weight of HBrO3
def mol_weight_HBrO3 : ℝ := mol_weight_H + mol_weight_Br + 3 * mol_weight_O

-- The goal: The molecular weight of 3 moles of HBrO3 is 386.73 grams
theorem molecular_weight_of_3_moles_HBrO3 : 3 * mol_weight_HBrO3 = 386.73 :=
by
  -- We will insert the proof here later
  sorry

end molecular_weight_of_3_moles_HBrO3_l1050_105055


namespace focus_of_parabola_l1050_105005

-- Define the given parabola equation
def given_parabola (x : ℝ) : ℝ := 4 * x^2

-- Define what it means to be the focus of this parabola
def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (0, 1 / 16)

-- The theorem to prove
theorem focus_of_parabola : ∃ focus : ℝ × ℝ, is_focus focus :=
  by 
    use (0, 1 / 16)
    exact sorry

end focus_of_parabola_l1050_105005


namespace power_addition_rule_l1050_105020

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l1050_105020


namespace focus_coordinates_of_hyperbola_l1050_105040

theorem focus_coordinates_of_hyperbola (x y : ℝ) :
  (∃ c : ℝ, (c = 5 ∧ y = 10) ∧ (c = 5 + Real.sqrt 97)) ↔ 
  (x, y) = (5 + Real.sqrt 97, 10) :=
by
  sorry

end focus_coordinates_of_hyperbola_l1050_105040


namespace binary_representation_of_23_l1050_105051

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end binary_representation_of_23_l1050_105051


namespace gather_all_candies_l1050_105008

theorem gather_all_candies (n : ℕ) (h₁ : n ≥ 4) (candies : ℕ) (h₂ : candies ≥ 4)
    (plates : Fin n → ℕ) :
    ∃ plate : Fin n, ∀ i : Fin n, i ≠ plate → plates i = 0 :=
sorry

end gather_all_candies_l1050_105008


namespace principal_amount_borrowed_l1050_105074

theorem principal_amount_borrowed
  (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) 
  (hR : R = 12) 
  (hT : T = 20) 
  (hSI : SI = 2100) 
  (hFormula : SI = (P * R * T) / 100) : 
  P = 875 := 
by 
  -- Assuming the initial steps 
  sorry

end principal_amount_borrowed_l1050_105074


namespace elmer_saves_14_3_percent_l1050_105064

-- Define the problem statement conditions and goal
theorem elmer_saves_14_3_percent (old_efficiency new_efficiency : ℝ) (old_cost new_cost : ℝ) :
  new_efficiency = 1.75 * old_efficiency →
  new_cost = 1.5 * old_cost →
  (500 / old_efficiency * old_cost - 500 / new_efficiency * new_cost) / (500 / old_efficiency * old_cost) * 100 = 14.3 := by
  -- sorry to skip the actual proof
  sorry

end elmer_saves_14_3_percent_l1050_105064


namespace domain_of_sqrt_fun_l1050_105075

theorem domain_of_sqrt_fun : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 7 → 7 + 6 * x - x^2 ≥ 0) :=
sorry

end domain_of_sqrt_fun_l1050_105075


namespace maximum_reflections_l1050_105060

theorem maximum_reflections (θ : ℕ) (h : θ = 10) (max_angle : ℕ) (h_max : max_angle = 180) : 
∃ n : ℕ, n ≤ max_angle / θ ∧ n = 18 := by
  sorry

end maximum_reflections_l1050_105060


namespace jason_tattoos_on_each_leg_l1050_105093

-- Define the basic setup
variable (x : ℕ)

-- Define the number of tattoos Jason has on each leg
def tattoos_on_each_leg := x

-- Define the total number of tattoos Jason has
def total_tattoos_jason := 2 + 2 + 2 * x

-- Define the total number of tattoos Adam has
def total_tattoos_adam := 23

-- Define the relation between Adam's and Jason's tattoos
def relation := 2 * total_tattoos_jason + 3 = total_tattoos_adam

-- The proof statement we need to show
theorem jason_tattoos_on_each_leg : tattoos_on_each_leg = 3  :=
by
  sorry

end jason_tattoos_on_each_leg_l1050_105093


namespace pet_store_cages_l1050_105018

theorem pet_store_cages 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (puppies_per_cage : ℕ) 
  (h_initial_puppies : initial_puppies = 45) 
  (h_sold_puppies : sold_puppies = 11) 
  (h_puppies_per_cage : puppies_per_cage = 7) 
  : (initial_puppies - sold_puppies + puppies_per_cage - 1) / puppies_per_cage = 5 :=
by sorry

end pet_store_cages_l1050_105018


namespace largest_value_among_given_numbers_l1050_105079

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem largest_value_among_given_numbers :
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20 
  b > a ∧ b > c ∧ b > d :=
by
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20
  -- Add the necessary steps to show that b is the largest value
  sorry

end largest_value_among_given_numbers_l1050_105079


namespace tangent_of_7pi_over_4_l1050_105012

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end tangent_of_7pi_over_4_l1050_105012


namespace total_jumps_is_400_l1050_105036

-- Define the variables according to the conditions 
def Ronald_jumps := 157
def Rupert_jumps := Ronald_jumps + 86

-- Prove the total jumps
theorem total_jumps_is_400 : Ronald_jumps + Rupert_jumps = 400 := by
  sorry

end total_jumps_is_400_l1050_105036


namespace derivative_at_neg_one_l1050_105052

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

theorem derivative_at_neg_one : deriv f (-1) = -1 :=
by
  -- definition of the function
  -- proof of the statement
  sorry

end derivative_at_neg_one_l1050_105052


namespace change_received_proof_l1050_105053

-- Define the costs and amounts
def regular_ticket_cost : ℕ := 9
def children_ticket_discount : ℕ := 2
def amount_given : ℕ := 2 * 20

-- Define the number of people
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 3

-- Define the costs calculations
def child_ticket_cost := regular_ticket_cost - children_ticket_discount
def total_adults_cost := number_of_adults * regular_ticket_cost
def total_children_cost := number_of_children * child_ticket_cost
def total_cost := total_adults_cost + total_children_cost
def change_received := amount_given - total_cost

-- Lean statement to prove the change received
theorem change_received_proof : change_received = 1 := by
  sorry

end change_received_proof_l1050_105053


namespace mrs_wilsborough_vip_tickets_l1050_105023

theorem mrs_wilsborough_vip_tickets:
  let S := 500 -- Initial savings
  let PVIP := 100 -- Price per VIP ticket
  let preg := 50 -- Price per regular ticket
  let nreg := 3 -- Number of regular tickets
  let R := 150 -- Remaining savings after purchase
  
  -- The total amount spent on tickets is S - R
  S - R = PVIP * 2 + preg * nreg := 
by sorry

end mrs_wilsborough_vip_tickets_l1050_105023


namespace arithmetic_sequence_sum_l1050_105076

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℕ} {S : ℕ → ℕ},
  (∀ n, a (n + 1) - a n = a 1 - a 0) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 1 + a 9 = 18 →
  a 4 = 7 →
  S 8 = 64 :=
by
  intros a S h_arith_seq h_sum_formula h_a1_a9 h_a4
  sorry

end arithmetic_sequence_sum_l1050_105076


namespace complement_intersection_l1050_105072

noncomputable def M : Set ℝ := {x | |x| > 2}
noncomputable def N : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | 1 < x ∧ x ≤ 2} :=
sorry

end complement_intersection_l1050_105072


namespace suff_not_necess_cond_perpendicular_l1050_105027

theorem suff_not_necess_cond_perpendicular (m : ℝ) :
  (m = 1 → ∀ x y : ℝ, x - y = 0 ∧ x + y = 0) ∧
  (m ≠ 1 → ∃ (x y : ℝ), ¬ (x - y = 0 ∧ x + y = 0)) :=
sorry

end suff_not_necess_cond_perpendicular_l1050_105027


namespace expression_for_A_l1050_105077

theorem expression_for_A (A k : ℝ)
  (h : ∀ k : ℝ, Ax^2 + 6 * k * x + 2 = 0 → k = 0.4444444444444444 → (6 * k)^2 - 4 * A * 2 = 0) :
  A = 9 * k^2 / 2 := 
sorry

end expression_for_A_l1050_105077


namespace prism_aligns_l1050_105003

theorem prism_aligns (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ prism_dimensions = (a * 5, b * 10, c * 20) :=
by
  sorry

end prism_aligns_l1050_105003


namespace isosceles_triangle_perimeter_l1050_105097

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l1050_105097


namespace student_b_speed_l1050_105078

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l1050_105078


namespace negation_exists_equiv_forall_l1050_105007

theorem negation_exists_equiv_forall :
  (¬ (∃ x : ℤ, x^2 + 2*x - 1 < 0)) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) :=
by
  sorry

end negation_exists_equiv_forall_l1050_105007


namespace math_proof_problem_l1050_105081

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

def parallel (x : Line) (y : Plane) : Prop := sorry
def contained_in (x : Line) (y : Plane) : Prop := sorry
def perpendicular (x : Plane) (y : Plane) : Prop := sorry
def perpendicular_line_plane (x : Line) (y : Plane) : Prop := sorry

theorem math_proof_problem :
  (perpendicular α β) ∧ (perpendicular_line_plane m β) ∧ ¬(contained_in m α) → parallel m α :=
by
  sorry

end math_proof_problem_l1050_105081


namespace devin_teaching_years_l1050_105049

section DevinTeaching
variable (Calculus Algebra Statistics Geometry DiscreteMathematics : ℕ)

theorem devin_teaching_years :
  Calculus = 4 ∧
  Algebra = 2 * Calculus ∧
  Statistics = 5 * Algebra ∧
  Geometry = 3 * Statistics ∧
  DiscreteMathematics = Geometry / 2 ∧
  (Calculus + Algebra + Statistics + Geometry + DiscreteMathematics) = 232 :=
by
  sorry
end DevinTeaching

end devin_teaching_years_l1050_105049


namespace sum_dihedral_angles_gt_360_l1050_105067

-- Define the structure Tetrahedron
structure Tetrahedron (α : Type*) :=
  (A B C D : α)

-- Define the dihedral angles function
noncomputable def sum_dihedral_angles {α : Type*} (T : Tetrahedron α) : ℝ := 
  -- Placeholder for the actual sum of dihedral angles of T
  sorry

-- Statement of the problem
theorem sum_dihedral_angles_gt_360 {α : Type*} (T : Tetrahedron α) :
  sum_dihedral_angles T > 360 := 
sorry

end sum_dihedral_angles_gt_360_l1050_105067


namespace triangle_side_b_l1050_105022

-- Define the conditions and state the problem
theorem triangle_side_b (A C : ℕ) (a b c : ℝ)
  (h1 : C = 4 * A)
  (h2 : a = 36)
  (h3 : c = 60) :
  b = 45 := by
  sorry

end triangle_side_b_l1050_105022


namespace sum_of_squares_of_two_numbers_l1050_105089

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) :
  x^2 + y^2 = 289 := 
  sorry

end sum_of_squares_of_two_numbers_l1050_105089


namespace greatest_savings_option2_l1050_105086

-- Define the initial price
def initial_price : ℝ := 15000

-- Define the discounts for each option
def discounts_option1 : List ℝ := [0.75, 0.85, 0.95]
def discounts_option2 : List ℝ := [0.65, 0.90, 0.95]
def discounts_option3 : List ℝ := [0.70, 0.90, 0.90]

-- Define a function to compute the final price after successive discounts
def final_price (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * d) initial

-- Define the savings for each option
def savings_option1 : ℝ := initial_price - (final_price initial_price discounts_option1)
def savings_option2 : ℝ := initial_price - (final_price initial_price discounts_option2)
def savings_option3 : ℝ := initial_price - (final_price initial_price discounts_option3)

-- Formulate the proof
theorem greatest_savings_option2 :
  max (max savings_option1 savings_option2) savings_option3 = savings_option2 :=
by
  sorry

end greatest_savings_option2_l1050_105086


namespace isosceles_triangle_height_l1050_105041

theorem isosceles_triangle_height (s h : ℝ) (eq_areas : (2 * s * s) = (1/2 * s * h)) : h = 4 * s :=
by
  sorry

end isosceles_triangle_height_l1050_105041


namespace equal_circles_common_point_l1050_105010

theorem equal_circles_common_point (n : ℕ) (r : ℝ) 
  (centers : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ∃ (p : ℝ × ℝ),
      dist p (centers i) = r ∧
      dist p (centers j) = r ∧
      dist p (centers k) = r) :
  ∃ O : ℝ × ℝ, ∀ i : Fin n, dist O (centers i) = r := sorry

end equal_circles_common_point_l1050_105010


namespace smallest_positive_period_and_axis_of_symmetry_l1050_105092

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem smallest_positive_period_and_axis_of_symmetry :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ k : ℤ, ∀ x, 2 * x - Real.pi / 4 = k * Real.pi + Real.pi / 2 → x = k * Real.pi / 2 - Real.pi / 8) :=
  sorry

end smallest_positive_period_and_axis_of_symmetry_l1050_105092


namespace relationship_between_n_and_m_l1050_105004

variable {n m : ℕ}
variable {x y : ℝ}
variable {a : ℝ}
variable {z : ℝ}

def mean_sample_combined (n m : ℕ) (x y z a : ℝ) : Prop :=
  z = a * x + (1 - a) * y ∧ a > 1 / 2

theorem relationship_between_n_and_m 
  (hx : ∀ (i : ℕ), i < n → x = x)
  (hy : ∀ (j : ℕ), j < m → y = y)
  (hz : mean_sample_combined n m x y z a)
  (hne : x ≠ y) : n < m :=
sorry

end relationship_between_n_and_m_l1050_105004


namespace insurance_covers_90_percent_l1050_105048

-- We firstly define the variables according to the conditions.
def adoption_fee : ℕ := 150
def training_cost_per_week : ℕ := 250
def training_weeks : ℕ := 12
def certification_cost : ℕ := 3000
def total_out_of_pocket_cost : ℕ := 3450

-- We now compute intermediate results based on the conditions provided.
def total_training_cost : ℕ := training_cost_per_week * training_weeks
def out_of_pocket_cert_cost : ℕ := total_out_of_pocket_cost - adoption_fee - total_training_cost
def insurance_coverage_amount : ℕ := certification_cost - out_of_pocket_cert_cost
def insurance_coverage_percentage : ℕ := (insurance_coverage_amount * 100) / certification_cost

-- Now, we state the theorem that needs to be proven.
theorem insurance_covers_90_percent : insurance_coverage_percentage = 90 := by
  sorry

end insurance_covers_90_percent_l1050_105048


namespace sin_inequality_l1050_105025

open Real

theorem sin_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (haq : a < π/4) (hb : 0 < b) (hbq : b < π/4) (hn : 0 < n) :
  (sin a)^n + (sin b)^n / (sin a + sin b)^n ≥ (sin (2 * a))^n + (sin (2 * b))^n / (sin (2 * a) + sin (2* b))^n :=
sorry

end sin_inequality_l1050_105025


namespace sam_new_crime_books_l1050_105032

theorem sam_new_crime_books (used_adventure_books : ℝ) (used_mystery_books : ℝ) (total_books : ℝ) :
  used_adventure_books = 13.0 →
  used_mystery_books = 17.0 →
  total_books = 45.0 →
  total_books - (used_adventure_books + used_mystery_books) = 15.0 :=
by
  intros ha hm ht
  rw [ha, hm, ht]
  norm_num
  -- sorry

end sam_new_crime_books_l1050_105032


namespace average_salary_increase_l1050_105019

theorem average_salary_increase 
  (average_salary : ℕ) (manager_salary : ℕ)
  (n : ℕ) (initial_count : ℕ) (new_count : ℕ) (initial_average : ℕ)
  (total_salary : ℕ) (new_total_salary : ℕ) (new_average : ℕ)
  (salary_increase : ℕ) :
  initial_average = 1500 →
  manager_salary = 3600 →
  initial_count = 20 →
  new_count = initial_count + 1 →
  total_salary = initial_count * initial_average →
  new_total_salary = total_salary + manager_salary →
  new_average = new_total_salary / new_count →
  salary_increase = new_average - initial_average →
  salary_increase = 100 := by
  sorry

end average_salary_increase_l1050_105019


namespace ball_bounce_height_l1050_105021

noncomputable def min_bounces (h₀ h_min : ℝ) (bounce_factor : ℝ) := 
  Nat.ceil (Real.log (h_min / h₀) / Real.log bounce_factor)

theorem ball_bounce_height :
  min_bounces 512 40 (3/4) = 8 :=
by
  sorry

end ball_bounce_height_l1050_105021


namespace step_of_induction_l1050_105013

theorem step_of_induction (k : ℕ) (h : ∃ m : ℕ, 5^k - 2^k = 3 * m) :
  5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k := 
by
  sorry

end step_of_induction_l1050_105013


namespace well_depth_is_2000_l1050_105014

-- Given conditions
def total_time : ℝ := 10
def stone_law (t₁ : ℝ) : ℝ := 20 * t₁^2
def sound_velocity : ℝ := 1120

-- Statement to be proven
theorem well_depth_is_2000 :
  ∃ (d t₁ t₂ : ℝ), 
    d = stone_law t₁ ∧ t₂ = d / sound_velocity ∧ t₁ + t₂ = total_time :=
sorry

end well_depth_is_2000_l1050_105014


namespace arithmetic_sequence_sum_l1050_105033

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : a 1 = -2012)
  (h₂ : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1)))
  (h₃ : (S 12) / 12 - (S 10) / 10 = 2) :
  S 2012 = -2012 := by
  sorry

end arithmetic_sequence_sum_l1050_105033


namespace log_max_reciprocal_min_l1050_105038

open Real

-- Definitions for the conditions
variables (x y : ℝ)
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

-- Theorem statement for the first question
theorem log_max (x y : ℝ) (h : conditions x y) : log x + log y ≤ 1 :=
sorry

-- Theorem statement for the second question
theorem reciprocal_min (x y : ℝ) (h : conditions x y) : (1 / x) + (1 / y) ≥ (7 + 2 * sqrt 10) / 20 :=
sorry

end log_max_reciprocal_min_l1050_105038


namespace distance_between_A_and_B_l1050_105001

theorem distance_between_A_and_B 
  (v t t1 : ℝ)
  (h1 : 5 * v * t + 4 * v * t = 9 * v * t)
  (h2 : t1 = 10 / (4.8 * v))
  (h3 : 10 / 4.8 = 25 / 12):
  (9 * v * t + 4 * v * t1) = 450 :=
by 
  -- Proof to be completed
  sorry

end distance_between_A_and_B_l1050_105001


namespace total_cows_on_farm_l1050_105035

-- Defining the conditions
variables (X H : ℕ) -- X is the number of cows per herd, H is the total number of herds
axiom half_cows_counted : 2800 = X * H / 2

-- The theorem stating the total number of cows on the entire farm
theorem total_cows_on_farm (X H : ℕ) (h1 : 2800 = X * H / 2) : 5600 = X * H := 
by 
  sorry

end total_cows_on_farm_l1050_105035


namespace carla_total_students_l1050_105080

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end carla_total_students_l1050_105080


namespace Edmund_can_wrap_15_boxes_every_3_days_l1050_105024

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l1050_105024


namespace min_frac_sum_l1050_105011

open Real

noncomputable def minValue (m n : ℝ) : ℝ := 1 / m + 2 / n

theorem min_frac_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  minValue m n = 3 + 2 * sqrt 2 := by
  sorry

end min_frac_sum_l1050_105011


namespace roll_contains_25_coins_l1050_105056

variable (coins_per_roll : ℕ)

def rolls_per_teller := 10
def number_of_tellers := 4
def total_coins := 1000

theorem roll_contains_25_coins : 
  (number_of_tellers * rolls_per_teller * coins_per_roll = total_coins) → 
  (coins_per_roll = 25) :=
by
  sorry

end roll_contains_25_coins_l1050_105056


namespace x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l1050_105061

variable {x y : ℝ}

theorem x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 - y^3 = 176 * Real.sqrt 13 := 
by
  sorry

end x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l1050_105061


namespace equation_in_terms_of_y_l1050_105083

theorem equation_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x :=
sorry

end equation_in_terms_of_y_l1050_105083


namespace find_reciprocal_l1050_105057

open Real

theorem find_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^3 + y^3 + 1 / 27 = x * y) : 1 / x = 3 := 
sorry

end find_reciprocal_l1050_105057


namespace age_problem_contradiction_l1050_105058

theorem age_problem_contradiction (C1 C2 : ℕ) (k : ℕ)
  (h1 : 15 = k * (C1 + C2))
  (h2 : 20 = 2 * (C1 + 5 + C2 + 5)) : false :=
by
  sorry

end age_problem_contradiction_l1050_105058


namespace minimum_value_of_function_l1050_105017

noncomputable def y (x : ℝ) : ℝ := 4 * x + 25 / x

theorem minimum_value_of_function : ∃ x > 0, y x = 20 :=
by
  sorry

end minimum_value_of_function_l1050_105017


namespace josephine_milk_containers_l1050_105096

theorem josephine_milk_containers :
  3 * 2 + 2 * 0.75 + 5 * x = 10 → x = 0.5 :=
by
  intro h
  sorry

end josephine_milk_containers_l1050_105096


namespace original_number_of_people_l1050_105090

-- Defining the conditions
variable (n : ℕ) -- number of people originally
variable (total_cost : ℕ := 375)
variable (equal_cost_split : n > 0 ∧ total_cost = 375) -- total cost is $375 and n > 0
variable (cost_condition : 375 / n + 50 = 375 / 5)

-- The proof statement
theorem original_number_of_people (h1 : total_cost = 375) (h2 : 375 / n + 50 = 375 / 5) : n = 15 :=
by
  sorry

end original_number_of_people_l1050_105090


namespace range_of_a_l1050_105098

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * x + 3 ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end range_of_a_l1050_105098


namespace triangle_inequality_a2_a3_a4_l1050_105054

variables {a1 a2 a3 a4 d : ℝ}

def is_arithmetic_sequence (a1 a2 a3 a4 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℝ) : Prop :=
  0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4

theorem triangle_inequality_a2_a3_a4 (h1: positive_terms a1 a2 a3 a4)
  (h2: is_arithmetic_sequence a1 a2 a3 a4 d) (h3: d > 0) :
  (a2 + a3 > a4) ∧ (a2 + a4 > a3) ∧ (a3 + a4 > a2) :=
sorry

end triangle_inequality_a2_a3_a4_l1050_105054


namespace max_isosceles_triangles_l1050_105070

theorem max_isosceles_triangles 
  {A B C D P : ℝ} 
  (h_collinear: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D)
  (h_non_collinear: P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
  : (∀ a b c : ℝ, (a = P ∨ a = A ∨ a = B ∨ a = C ∨ a = D) ∧ (b = P ∨ b = A ∨ b = B ∨ b = C ∨ b = D) ∧ (c = P ∨ c = A ∨ c = B ∨ c = C ∨ c = D) 
    ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((a - b)^2 + (b - c)^2 = (a - c)^2 ∨ (a - c)^2 + (b - c)^2 = (a - b)^2 ∨ (a - b)^2 + (a - c)^2 = (b - c)^2)) → 
    isosceles_triangle_count = 6 :=
sorry

end max_isosceles_triangles_l1050_105070


namespace problem_statement_l1050_105047

def f (x : ℝ) : ℝ := x^6 + x^2 + 7 * x

theorem problem_statement : f 3 - f (-3) = 42 := by
  sorry

end problem_statement_l1050_105047


namespace Abby_has_17_quarters_l1050_105000

theorem Abby_has_17_quarters (q n : ℕ) (h1 : q + n = 23) (h2 : 25 * q + 5 * n = 455) : q = 17 :=
sorry

end Abby_has_17_quarters_l1050_105000


namespace beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1050_105016

def beautiful_association_number (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 8 :=
by sorry

theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 4 ↔ x = 6 ∨ x = 0 :=
by sorry

theorem beautiful_association_number_part3 (x0 x1 x2 x3 x4 : ℚ) :
  beautiful_association_number x0 x1 1 1 ∧ 
  beautiful_association_number x1 x2 2 1 ∧ 
  beautiful_association_number x2 x3 3 1 ∧ 
  beautiful_association_number x3 x4 4 1 →
  x1 + x2 + x3 + x4 = 10 :=
by sorry

end beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1050_105016


namespace domain_of_f_l1050_105028

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | 0 < x + 1} ∩ {x : ℝ | x ≠ 0} ∩ {x : ℝ | 9 - x^2 ≥ 0} = (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioc 0 (3 : ℝ)) :=
by
  sorry

end domain_of_f_l1050_105028


namespace intersecting_lines_solution_l1050_105039

theorem intersecting_lines_solution (a b : ℝ) :
  (∃ (a b : ℝ), 
    ((a^2 + 1) * 2 - 2 * b * (-3) = 4) ∧ 
    ((1 - a) * 2 + b * (-3) = 9)) →
  (a, b) = (4, -5) ∨ (a, b) = (-2, -1) :=
by
  sorry

end intersecting_lines_solution_l1050_105039


namespace find_constant_k_l1050_105037

theorem find_constant_k (k : ℝ) :
  (-x^2 - (k + 9) * x - 8 = - (x - 2) * (x - 4)) → k = -15 :=
by 
  sorry

end find_constant_k_l1050_105037


namespace find_b_l1050_105006

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 35 * b) : b = 63 := 
by 
  sorry

end find_b_l1050_105006


namespace find_A_l1050_105031

-- Define the four-digit number being a multiple of 9 and the sum of its digits condition
def digit_sum_multiple_of_9 (A : ℤ) : Prop :=
  (3 + A + A + 1) % 9 = 0

-- The Lean statement for the proof problem
theorem find_A (A : ℤ) (h : digit_sum_multiple_of_9 A) : A = 7 :=
sorry

end find_A_l1050_105031


namespace b_n_geometric_a_n_formula_T_n_sum_less_than_2_l1050_105073

section problem

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {C_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Given conditions
axiom seq_a (n : ℕ) : a_n 1 = 1
axiom recurrence (n : ℕ) : 2 * a_n (n + 1) - a_n n = (n - 2) / (n * (n + 1) * (n + 2))
axiom seq_b (n : ℕ) : b_n n = a_n n - 1 / (n * (n + 1))

-- Required proofs
theorem b_n_geometric : ∀ n : ℕ, b_n n = (1 / 2) ^ n := sorry
theorem a_n_formula : ∀ n : ℕ, a_n n = (1 / 2) ^ n + 1 / (n * (n + 1)) := sorry
theorem T_n_sum_less_than_2 : ∀ n : ℕ, T_n n < 2 := sorry

end problem

end b_n_geometric_a_n_formula_T_n_sum_less_than_2_l1050_105073


namespace observation_count_l1050_105029

theorem observation_count (n : ℤ) (mean_initial : ℝ) (erroneous_value correct_value : ℝ) (mean_corrected : ℝ) :
  mean_initial = 36 →
  erroneous_value = 20 →
  correct_value = 34 →
  mean_corrected = 36.45 →
  n ≥ 0 →
  ∃ n : ℤ, (n * mean_initial + (correct_value - erroneous_value) = n * mean_corrected) ∧ (n = 31) :=
by
  intros h1 h2 h3 h4 h5
  use 31
  sorry

end observation_count_l1050_105029


namespace find_e_l1050_105082

theorem find_e (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + b = 32)
  (h6 : a + c = 36)
  (h7 : b + c = 37)
  (h8 : c + e = 48)
  (h9 : d + e = 51) : e = 55 / 2 :=
  sorry

end find_e_l1050_105082


namespace correct_calculation_result_l1050_105095

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end correct_calculation_result_l1050_105095


namespace all_metals_conduct_electricity_l1050_105030

def Gold_conducts : Prop := sorry
def Silver_conducts : Prop := sorry
def Copper_conducts : Prop := sorry
def Iron_conducts : Prop := sorry
def inductive_reasoning : Prop := sorry

theorem all_metals_conduct_electricity (g: Gold_conducts) (s: Silver_conducts) (c: Copper_conducts) (i: Iron_conducts) : inductive_reasoning := 
sorry

end all_metals_conduct_electricity_l1050_105030


namespace find_a_if_odd_l1050_105043

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem find_a_if_odd (a : ℝ) : (∀ x : ℝ, f (-x) a = -f x a) → a = 0 := by
  intro h
  have h0 : f 0 a = 0 := by
    simp [f]
    specialize h 0
    simp [f] at h
    exact h
  sorry

end find_a_if_odd_l1050_105043


namespace distance_between_first_and_last_tree_l1050_105087

theorem distance_between_first_and_last_tree
  (n : ℕ)
  (trees : ℕ)
  (dist_between_first_and_fourth : ℕ)
  (eq_dist : ℕ):
  trees = 6 ∧ dist_between_first_and_fourth = 60 ∧ eq_dist = dist_between_first_and_fourth / 3 ∧ n = (trees - 1) * eq_dist → n = 100 :=
by
  intro h
  sorry

end distance_between_first_and_last_tree_l1050_105087


namespace probability_A_score_not_less_than_135_l1050_105026

/-- A certain school organized a competition with the following conditions:
  - The test has 25 multiple-choice questions, each with 4 options.
  - Each correct answer scores 6 points, each unanswered question scores 2 points, and each wrong answer scores 0 points.
  - Both candidates answered the first 20 questions correctly.
  - Candidate A will attempt only the last 3 questions, and for each, A can eliminate 1 wrong option,
    hence the probability of answering any one question correctly is 1/3.
  - A gives up the last 2 questions.
  - Prove that the probability that A's total score is not less than 135 points is equal to 7/27.
-/
theorem probability_A_score_not_less_than_135 :
  let prob_success := 1 / 3
  let prob_2_successes := (3 * (prob_success^2) * (2/3))
  let prob_3_successes := (prob_success^3)
  prob_2_successes + prob_3_successes = 7 / 27 := 
by
  sorry

end probability_A_score_not_less_than_135_l1050_105026


namespace closest_point_on_line_l1050_105085

theorem closest_point_on_line (x y: ℚ) (h1: y = -4 * x + 3) (h2: ∀ p q: ℚ, y = -4 * p + 3 ∧ y = q * (-4 * p) - q * (-4 * 1 + 0)): (x, y) = (-1 / 17, 55 / 17) :=
sorry

end closest_point_on_line_l1050_105085


namespace area_of_rectangle_l1050_105069

variables {group_interval rate : ℝ}

theorem area_of_rectangle (length_of_small_rectangle : ℝ) (height_of_small_rectangle : ℝ) :
  (length_of_small_rectangle = group_interval) → (height_of_small_rectangle = rate / group_interval) →
  length_of_small_rectangle * height_of_small_rectangle = rate :=
by
  intros h_length h_height
  rw [h_length, h_height]
  exact mul_div_cancel' rate (by sorry)

end area_of_rectangle_l1050_105069


namespace fraction_product_l1050_105045

theorem fraction_product :
  ((1: ℚ) / 2) * (3 / 5) * (7 / 11) = 21 / 110 :=
by {
  sorry
}

end fraction_product_l1050_105045


namespace find_longer_diagonal_l1050_105015

-- Define the necessary conditions
variables (d1 d2 : ℝ)
variable (A : ℝ)
axiom ratio_condition : d1 / d2 = 2 / 3
axiom area_condition : A = 12

-- Define the problem of finding the length of the longer diagonal
theorem find_longer_diagonal : ∃ (d : ℝ), d = d2 → d = 6 :=
by 
  sorry

end find_longer_diagonal_l1050_105015


namespace range_of_p_l1050_105099

-- Definitions of A and B
def A (p : ℝ) := {x : ℝ | x^2 + (p + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}

-- Condition of the problem: A ∩ B = ∅
def condition (p : ℝ) := ∀ x ∈ A p, x ∉ B

-- The statement to prove: p > -4
theorem range_of_p (p : ℝ) : condition p → p > -4 :=
by
  intro h
  sorry

end range_of_p_l1050_105099


namespace geometric_sequence_problem_l1050_105084

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given condition for the geometric sequence
variables {a : ℕ → ℝ} (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27)

-- Theorem to be proven
theorem geometric_sequence_problem (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27) : a 1 * a 9 = 9 :=
sorry

end geometric_sequence_problem_l1050_105084


namespace max_XYZ_plus_terms_l1050_105009

theorem max_XYZ_plus_terms {X Y Z : ℕ} (h : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
sorry

end max_XYZ_plus_terms_l1050_105009


namespace max_value_expr_l1050_105062

theorem max_value_expr (x : ℝ) : 
  ( x ^ 6 / (x ^ 12 + 3 * x ^ 8 - 6 * x ^ 6 + 12 * x ^ 4 + 36) <= 1/18 ) :=
by
  sorry

end max_value_expr_l1050_105062


namespace probability_even_sum_of_spins_l1050_105050

theorem probability_even_sum_of_spins :
  let prob_even_first := 3 / 6
  let prob_odd_first := 3 / 6
  let prob_even_second := 2 / 5
  let prob_odd_second := 3 / 5
  let prob_both_even := prob_even_first * prob_even_second
  let prob_both_odd := prob_odd_first * prob_odd_second
  prob_both_even + prob_both_odd = 1 / 2 := 
by 
  sorry

end probability_even_sum_of_spins_l1050_105050


namespace tank_capacity_l1050_105091

theorem tank_capacity :
  ∀ (T : ℚ), (3 / 4) * T + 4 = (7 / 8) * T → T = 32 :=
by
  intros T h
  sorry

end tank_capacity_l1050_105091


namespace max_edges_in_8_points_graph_no_square_l1050_105071

open Finset

-- Define what a graph is and the properties needed for the problem
structure Graph (V : Type*) :=
  (edges : Finset (V × V))
  (sym : ∀ {x y : V}, (x, y) ∈ edges ↔ (y, x) ∈ edges)
  (irrefl : ∀ {x : V}, ¬ (x, x) ∈ edges)

-- Define the conditions of the problem
def no_square {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c d : V), 
    (a, b) ∈ G.edges → (b, c) ∈ G.edges → (c, d) ∈ G.edges → (d, a) ∈ G.edges →
    (a, c) ∈ G.edges → (b, d) ∈ G.edges → False

-- Define 8 vertices
inductive Vertices
| A | B | C | D | E | F | G | H

-- Define the number of edges
noncomputable def max_edges_no_square : ℕ :=
  11

-- Define the final theorem
theorem max_edges_in_8_points_graph_no_square :
  ∃ (G : Graph Vertices), 
    no_square G ∧ (G.edges.card = max_edges_no_square) :=
sorry

end max_edges_in_8_points_graph_no_square_l1050_105071
