import Mathlib

namespace tangent_line_eq_l1378_137856

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 - x + 1) (h_point : (x, y) = (0, 1)) : x + y - 1 = 0 := 
sorry

end tangent_line_eq_l1378_137856


namespace semi_circle_radius_l1378_137887

theorem semi_circle_radius (P : ℝ) (π : ℝ) (r : ℝ) (hP : P = 10.797344572538567) (hπ : π = 3.14159) :
  (π + 2) * r = P → r = 2.1 :=
by
  intro h
  sorry

end semi_circle_radius_l1378_137887


namespace average_hours_l1378_137847

def hours_studied (week1 week2 week3 week4 week5 week6 week7 : ℕ) : ℕ :=
  week1 + week2 + week3 + week4 + week5 + week6 + week7

theorem average_hours (x : ℕ)
  (h1 : hours_studied 8 10 9 11 10 7 x / 7 = 9) :
  x = 8 :=
by
  sorry

end average_hours_l1378_137847


namespace find_x_for_equation_l1378_137872

theorem find_x_for_equation 
  (x : ℝ)
  (h : (32 : ℝ)^(x-2) / (8 : ℝ)^(x-2) = (512 : ℝ)^(3 * x)) : 
  x = -4/25 :=
by
  sorry

end find_x_for_equation_l1378_137872


namespace percentage_of_150_l1378_137891

theorem percentage_of_150 : (1 / 5 * (1 / 100) * 150 : ℝ) = 0.3 := by
  sorry

end percentage_of_150_l1378_137891


namespace simplest_quadratic_radical_l1378_137838

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l1378_137838


namespace cos_pi_div_4_add_alpha_l1378_137808

variable (α : ℝ)

theorem cos_pi_div_4_add_alpha (h : Real.sin (Real.pi / 4 - α) = Real.sqrt 2 / 2) :
  Real.cos (Real.pi / 4 + α) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_pi_div_4_add_alpha_l1378_137808


namespace minimum_value_of_z_l1378_137889

theorem minimum_value_of_z 
  (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : 2 * x - y - 2 ≤ 0) 
  (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x + y ∧ z = -6 :=
sorry

end minimum_value_of_z_l1378_137889


namespace find_number_l1378_137843

theorem find_number (x : ℝ) (h : 5020 - 502 / x = 5015) : x = 100.4 :=
by
  sorry

end find_number_l1378_137843


namespace combined_weight_of_two_new_students_l1378_137845

theorem combined_weight_of_two_new_students (W : ℕ) (X : ℕ) 
  (cond1 : (W - 150 + X) / 8 = (W / 8) - 2) :
  X = 134 := 
sorry

end combined_weight_of_two_new_students_l1378_137845


namespace range_of_a_l1378_137850

-- Define the assumptions and target proof
theorem range_of_a {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0)
  : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0 → a < 3 :=
by
  intro a h
  sorry

end range_of_a_l1378_137850


namespace constraint_condition_2000_yuan_wage_l1378_137813

-- Definitions based on the given conditions
def wage_carpenter : ℕ := 50
def wage_bricklayer : ℕ := 40
def total_wage : ℕ := 2000

-- Let x be the number of carpenters and y be the number of bricklayers
variable (x y : ℕ)

-- The proof problem statement
theorem constraint_condition_2000_yuan_wage (x y : ℕ) : 
  wage_carpenter * x + wage_bricklayer * y = total_wage → 5 * x + 4 * y = 200 :=
by
  intro h
  -- Simplification step will be placed here
  sorry

end constraint_condition_2000_yuan_wage_l1378_137813


namespace smallest_prime_factor_in_setB_l1378_137895

def setB : Set ℕ := {55, 57, 58, 59, 61}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 2 then 2 else (Nat.minFac (Nat.pred n)).succ

theorem smallest_prime_factor_in_setB :
  ∃ n ∈ setB, smallest_prime_factor n = 2 := by
  sorry

end smallest_prime_factor_in_setB_l1378_137895


namespace smallest_n_watches_l1378_137893

variable {n d : ℕ}

theorem smallest_n_watches (h1 : d > 0)
  (h2 : 10 * n - 30 = 100) : n = 13 :=
by
  sorry

end smallest_n_watches_l1378_137893


namespace length_of_each_piece_is_correct_l1378_137804

noncomputable def rod_length : ℝ := 38.25
noncomputable def num_pieces : ℕ := 45
noncomputable def length_each_piece_cm : ℝ := 85

theorem length_of_each_piece_is_correct : (rod_length / num_pieces) * 100 = length_each_piece_cm :=
by
  sorry

end length_of_each_piece_is_correct_l1378_137804


namespace arithmetic_sequence_sum_l1378_137807

theorem arithmetic_sequence_sum 
    (a : ℕ → ℤ)
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Arithmetic sequence condition
    (h2 : a 5 = 3)
    (h3 : a 6 = -2) :
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end arithmetic_sequence_sum_l1378_137807


namespace crease_length_l1378_137863

noncomputable def length_of_crease (theta : ℝ) : ℝ :=
  8 * Real.sin theta

theorem crease_length (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ π / 2) : 
  length_of_crease theta = 8 * Real.sin theta :=
by sorry

end crease_length_l1378_137863


namespace value_of_expression_l1378_137871

theorem value_of_expression (x y : ℤ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 :=
by
  -- Substitute the given values into the expression and calculate
  sorry

end value_of_expression_l1378_137871


namespace intercept_condition_l1378_137886

theorem intercept_condition (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ x = -c / a ∧ y = -c / b ∧ x = y) → (c = 0 ∨ a = b) :=
by
  sorry

end intercept_condition_l1378_137886


namespace smallest_positive_multiple_l1378_137849

/-- Prove that the smallest positive multiple of 15 that is 7 more than a multiple of 65 is 255. -/
theorem smallest_positive_multiple : 
  ∃ n : ℕ, n > 0 ∧ n % 15 = 0 ∧ n % 65 = 7 ∧ n = 255 :=
sorry

end smallest_positive_multiple_l1378_137849


namespace no_positive_integer_solutions_l1378_137875

theorem no_positive_integer_solutions :
  ∀ (A : ℕ), 1 ≤ A ∧ A ≤ 9 → ¬∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = A * 10 + A ∧ x + y = 10 * A + 1 := by
  sorry

end no_positive_integer_solutions_l1378_137875


namespace find_principal_l1378_137829

-- Defining the conditions
def A : ℝ := 5292
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The theorem statement
theorem find_principal :
  ∃ (P : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ P = 4800 :=
by
  sorry

end find_principal_l1378_137829


namespace average_of_three_numbers_is_165_l1378_137827

variable (x y z : ℕ)
variable (hy : y = 90)
variable (h1 : z = 4 * y)
variable (h2 : y = 2 * x)

theorem average_of_three_numbers_is_165 : (x + y + z) / 3 = 165 := by
  sorry

end average_of_three_numbers_is_165_l1378_137827


namespace benzene_carbon_mass_percentage_l1378_137865

noncomputable def carbon_mass_percentage_in_benzene 
  (carbon_atomic_mass : ℝ) (hydrogen_atomic_mass : ℝ) 
  (benzene_formula_ratio : (ℕ × ℕ)) : ℝ := 
    let (num_carbon_atoms, num_hydrogen_atoms) := benzene_formula_ratio
    let total_carbon_mass := num_carbon_atoms * carbon_atomic_mass
    let total_hydrogen_mass := num_hydrogen_atoms * hydrogen_atomic_mass
    let total_mass := total_carbon_mass + total_hydrogen_mass
    100 * (total_carbon_mass / total_mass)

theorem benzene_carbon_mass_percentage 
  (carbon_atomic_mass : ℝ := 12.01) 
  (hydrogen_atomic_mass : ℝ := 1.008) 
  (benzene_formula_ratio : (ℕ × ℕ) := (6, 6)) : 
    carbon_mass_percentage_in_benzene carbon_atomic_mass hydrogen_atomic_mass benzene_formula_ratio = 92.23 :=
by 
  unfold carbon_mass_percentage_in_benzene
  sorry

end benzene_carbon_mass_percentage_l1378_137865


namespace Max_students_count_l1378_137846

variables (M J : ℕ)

theorem Max_students_count :
  (M = 2 * J + 100) → 
  (M + J = 5400) → 
  M = 3632 := 
by 
  intros h1 h2
  sorry

end Max_students_count_l1378_137846


namespace sid_spent_on_snacks_l1378_137828

theorem sid_spent_on_snacks :
  let original_money := 48
  let money_spent_on_computer_accessories := 12
  let money_left_after_computer_accessories := original_money - money_spent_on_computer_accessories
  let remaining_money_after_purchases := 4 + original_money / 2
  ∃ snacks_cost, money_left_after_computer_accessories - snacks_cost = remaining_money_after_purchases ∧ snacks_cost = 8 :=
by
  sorry

end sid_spent_on_snacks_l1378_137828


namespace walking_distance_l1378_137814

-- Define the pace in miles per hour.
def pace : ℝ := 2

-- Define the duration in hours.
def duration : ℝ := 8

-- Define the total distance walked.
def total_distance (pace : ℝ) (duration : ℝ) : ℝ := pace * duration

-- Define the theorem we need to prove.
theorem walking_distance :
  total_distance pace duration = 16 := by
  sorry

end walking_distance_l1378_137814


namespace find_time_same_height_l1378_137823

noncomputable def height_ball (t : ℝ) : ℝ := 60 - 9 * t - 8 * t^2
noncomputable def height_bird (t : ℝ) : ℝ := 3 * t^2 + 4 * t

theorem find_time_same_height : ∃ t : ℝ, t = 20 / 11 ∧ height_ball t = height_bird t := 
by
  use 20 / 11
  sorry

end find_time_same_height_l1378_137823


namespace ratio_of_girls_to_boys_l1378_137819

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) (h_ratio : girls = 4 * (girls + boys) / 7) (h_total : total_students = 70) : 
  girls = 40 ∧ boys = 30 :=
by
  sorry

end ratio_of_girls_to_boys_l1378_137819


namespace find_x_solutions_l1378_137833

theorem find_x_solutions (x : ℝ) :
  let f (x : ℝ) := x^2 - 4*x + 1
  let f2 (x : ℝ) := (f x)^2
  f (f x) = f2 x ↔ x = 2 + (Real.sqrt 13) / 2 ∨ x = 2 - (Real.sqrt 13) / 2 := by
  sorry

end find_x_solutions_l1378_137833


namespace Wilsons_number_l1378_137858

theorem Wilsons_number (N : ℝ) (h : N - N / 3 = 16 / 3) : N = 8 := sorry

end Wilsons_number_l1378_137858


namespace beam_equation_correctness_l1378_137867

-- Define the conditions
def total_selling_price : ℕ := 6210
def freight_per_beam : ℕ := 3

-- Define the unknown quantity
variable (x : ℕ)

-- State the theorem
theorem beam_equation_correctness
  (h1 : total_selling_price = 6210)
  (h2 : freight_per_beam = 3) :
  freight_per_beam * (x - 1) = total_selling_price / x := 
sorry

end beam_equation_correctness_l1378_137867


namespace possible_to_position_guards_l1378_137870

-- Define the conditions
def guard_sees (d : ℝ) : Prop := d = 100

-- Prove that it is possible to arrange guards around a point object so that neither the object nor the guards can be approached unnoticed
theorem possible_to_position_guards (num_guards : ℕ) (d : ℝ) (h : guard_sees d) : 
  (0 < num_guards) → 
  (∀ θ : ℕ, θ < num_guards → (θ * (360 / num_guards)) < 360) → 
  True :=
by 
  -- Details of the proof would go here
  sorry

end possible_to_position_guards_l1378_137870


namespace one_cow_one_bag_in_46_days_l1378_137848

-- Defining the conditions
def cows_eat_husk (n_cows n_bags n_days : ℕ) := n_cows = n_bags ∧ n_cows = n_days ∧ n_bags = n_days

-- The main theorem to be proved
theorem one_cow_one_bag_in_46_days (h : cows_eat_husk 46 46 46) : 46 = 46 := by
  sorry

end one_cow_one_bag_in_46_days_l1378_137848


namespace range_of_a_l1378_137882

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * a * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l1378_137882


namespace interest_difference_l1378_137834

noncomputable def difference_between_interest (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) : ℝ :=
  let SI := P * R * T / 100
  let CI := P * (1 + (R / (n*100)))^(n * T) - P
  CI - SI

theorem interest_difference (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  difference_between_interest P R T n = -59.25 := by
  sorry

end interest_difference_l1378_137834


namespace sum_of_other_endpoint_coordinates_l1378_137825

theorem sum_of_other_endpoint_coordinates :
  ∃ (x y: ℤ), (8 + x) / 2 = 6 ∧ y / 2 = -10 ∧ x + y = -16 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l1378_137825


namespace minimize_acme_cost_l1378_137805

theorem minimize_acme_cost (x : ℕ) : 75 + 12 * x < 16 * x → x = 19 :=
by
  intro h
  sorry

end minimize_acme_cost_l1378_137805


namespace tangent_product_constant_l1378_137851

variable (a x₁ x₂ y₁ y₂ : ℝ)

def point_on_parabola (x y : ℝ) := x^2 = 4 * y
def point_P := (a, -2)
def point_A := (x₁, y₁)
def point_B := (x₂, y₂)

theorem tangent_product_constant
  (h₁ : point_on_parabola x₁ y₁)
  (h₂ : point_on_parabola x₂ y₂)
  (h₃ : ∃ k₁ k₂ : ℝ, 
        (y₁ + 2 = k₁ * (x₁ - a) ∧ y₂ + 2 = k₂ * (x₂ - a)) 
        ∧ (k₁ * k₂ = -2)) :
  x₁ * x₂ + y₁ * y₂ = -4 :=
sorry

end tangent_product_constant_l1378_137851


namespace sam_seashells_l1378_137824

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end sam_seashells_l1378_137824


namespace greatest_possible_individual_award_l1378_137801

variable (prize : ℕ)
variable (total_winners : ℕ)
variable (min_award : ℕ)
variable (fraction_prize : ℚ)
variable (fraction_winners : ℚ)

theorem greatest_possible_individual_award 
  (h1 : prize = 2500)
  (h2 : total_winners = 25)
  (h3 : min_award = 50)
  (h4 : fraction_prize = 3/5)
  (h5 : fraction_winners = 2/5) :
  ∃ award, award = 1300 := by
  sorry

end greatest_possible_individual_award_l1378_137801


namespace area_percentage_of_smaller_square_l1378_137820

theorem area_percentage_of_smaller_square 
  (radius : ℝ)
  (a A O B: ℝ)
  (side_length_larger_square side_length_smaller_square : ℝ) 
  (hyp1 : side_length_larger_square = 4)
  (hyp2 : radius = 2 * Real.sqrt 2)
  (hyp3 : a = 4) 
  (hyp4 : A = 2 + side_length_smaller_square / 4)
  (hyp5 : O = 2 * Real.sqrt 2)
  (hyp6 : side_length_smaller_square = 0.8) :
  (side_length_smaller_square^2 / side_length_larger_square^2) = 0.04 :=
by
  sorry

end area_percentage_of_smaller_square_l1378_137820


namespace unicorn_rope_length_l1378_137873

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 1500
noncomputable def c : ℕ := 3

theorem unicorn_rope_length : a + b + c = 1593 :=
by
  -- The steps to prove the theorem should go here, but as stated, we skip this with "sorry".
  sorry

end unicorn_rope_length_l1378_137873


namespace log_sum_equality_l1378_137859

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end log_sum_equality_l1378_137859


namespace sum_of_fractions_l1378_137800

theorem sum_of_fractions :
  (7:ℚ) / 12 + (11:ℚ) / 15 = 79 / 60 :=
by
  sorry

end sum_of_fractions_l1378_137800


namespace evaluate_expression_l1378_137880

theorem evaluate_expression : 
  let a := 45
  let b := 15
  (a + b)^2 - (a^2 + b^2 + 2 * a * 5) = 900 :=
by
  let a := 45
  let b := 15
  sorry

end evaluate_expression_l1378_137880


namespace distinct_roots_iff_l1378_137866

def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 2 * Real.sqrt (3 + 2*a*x1 - 4*a)) ∧
                       (x2 + |x2| = 2 * Real.sqrt (3 + 2*a*x2 - 4*a))

theorem distinct_roots_iff (a : ℝ) :
  has_two_distinct_roots a ↔ (a ∈ Set.Ioo 0 (3 / 4 : ℝ) ∨ 3 < a) :=
sorry

end distinct_roots_iff_l1378_137866


namespace factorize_expr_l1378_137899

def my_expr (a b : ℤ) : ℤ := 4 * a^2 * b - b

theorem factorize_expr (a b : ℤ) : my_expr a b = b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end factorize_expr_l1378_137899


namespace minimum_value_of_fraction_l1378_137821

theorem minimum_value_of_fraction {x : ℝ} (hx : x ≥ 3/2) :
  ∃ y : ℝ, y = (2 * (x - 1) + (1 / (x - 1)) + 2) ∧ y = 2 * Real.sqrt 2 + 2 :=
sorry

end minimum_value_of_fraction_l1378_137821


namespace face_value_of_each_ticket_without_tax_l1378_137810

theorem face_value_of_each_ticket_without_tax (total_people : ℕ) (total_cost : ℝ) (sales_tax : ℝ) (face_value : ℝ)
  (h1 : total_people = 25)
  (h2 : total_cost = 945)
  (h3 : sales_tax = 0.05)
  (h4 : total_cost = (1 + sales_tax) * face_value * total_people) :
  face_value = 36 := by
  sorry

end face_value_of_each_ticket_without_tax_l1378_137810


namespace function_solution_l1378_137855

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))

theorem function_solution (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x + 2 * f ((x - 1) / x) = 3 * x) →
  (∃ c : ℝ, ∀ x : ℝ, f x = if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))) :=
by
  intro h
  use (f 0)
  intro x
  split_ifs with h0 h1
  rotate_left -- to handle the cases x ≠ 0, 1 at first.
  sorry -- Additional proof steps required here.
  sorry -- Use the given conditions and functional equation to conclude f(0) = c.
  sorry -- Use the given conditions and functional equation to conclude f(1) = 3 - 2c.

end function_solution_l1378_137855


namespace evaluate_cubic_difference_l1378_137885

theorem evaluate_cubic_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 :=
by sorry

end evaluate_cubic_difference_l1378_137885


namespace butter_needed_for_original_recipe_l1378_137830

-- Define the conditions
def butter_to_flour_ratio : ℚ := 12 / 56

def flour_for_original_recipe : ℚ := 14

def butter_for_original_recipe (ratio : ℚ) (flour : ℚ) : ℚ :=
  ratio * flour

-- State the theorem
theorem butter_needed_for_original_recipe :
  butter_for_original_recipe butter_to_flour_ratio flour_for_original_recipe = 3 := 
sorry

end butter_needed_for_original_recipe_l1378_137830


namespace difference_students_rabbits_l1378_137861

-- Define the number of students per classroom
def students_per_classroom := 22

-- Define the number of rabbits per classroom
def rabbits_per_classroom := 4

-- Define the number of classrooms
def classrooms := 6

-- Calculate the total number of students
def total_students := students_per_classroom * classrooms

-- Calculate the total number of rabbits
def total_rabbits := rabbits_per_classroom * classrooms

-- Prove the difference between the number of students and rabbits is 108
theorem difference_students_rabbits : total_students - total_rabbits = 108 := by
  sorry

end difference_students_rabbits_l1378_137861


namespace sampling_methods_suitability_l1378_137816

-- Define sample sizes and population sizes
def n1 := 2  -- Number of students to be selected in sample ①
def N1 := 10  -- Population size for sample ①
def n2 := 50  -- Number of students to be selected in sample ②
def N2 := 1000  -- Population size for sample ②

-- Define what it means for a sampling method to be suitable
def is_simple_random_sampling_suitable (n N : Nat) : Prop :=
  N <= 50 ∧ n < N

def is_systematic_sampling_suitable (n N : Nat) : Prop :=
  N > 50 ∧ n < N ∧ n ≥ 50 / 1000 * N  -- Ensuring suitable systematic sampling size

-- The proof statement
theorem sampling_methods_suitability :
  is_simple_random_sampling_suitable n1 N1 ∧ is_systematic_sampling_suitable n2 N2 :=
by
  -- Sorry blocks are used to skip the proofs
  sorry

end sampling_methods_suitability_l1378_137816


namespace prism_surface_area_l1378_137868

theorem prism_surface_area (a : ℝ) : 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  surface_area_cubes - surface_area_shared_faces = 14 * a^2 := 
by 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  have : surface_area_cubes - surface_area_shared_faces = 14 * a^2 := sorry
  exact this

end prism_surface_area_l1378_137868


namespace smallest_N_conditions_l1378_137892

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l1378_137892


namespace multiple_of_8_and_12_l1378_137837

theorem multiple_of_8_and_12 (x y : ℤ) (hx : ∃ k : ℤ, x = 8 * k) (hy : ∃ k : ℤ, y = 12 * k) :
  (∃ k : ℤ, y = 4 * k) ∧ (∃ k : ℤ, x - y = 4 * k) :=
by
  /- Proof goes here, based on the given conditions -/
  sorry

end multiple_of_8_and_12_l1378_137837


namespace three_digit_integers_count_l1378_137831

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end three_digit_integers_count_l1378_137831


namespace simplify_and_evaluate_expression_l1378_137897

theorem simplify_and_evaluate_expression : 
  ∀ a : ℚ, a = -1/2 → (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := 
by
  intro a ha
  simp only [ha]
  sorry

end simplify_and_evaluate_expression_l1378_137897


namespace necessary_but_not_sufficient_condition_l1378_137852

noncomputable def condition (m : ℝ) : Prop := 1 < m ∧ m < 3

def represents_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  (x ^ 2) / (m - 1) + (y ^ 2) / (3 - m) = 1

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∃ x y, represents_ellipse m x y) → condition m :=
sorry

end necessary_but_not_sufficient_condition_l1378_137852


namespace trajectory_of_intersection_l1378_137869

-- Define the conditions and question in Lean
structure Point where
  x : ℝ
  y : ℝ

def on_circle (C : Point) : Prop :=
  C.x^2 + C.y^2 = 1

def perp_to_x_axis (C D : Point) : Prop :=
  C.x = D.x ∧ C.y = -D.y

theorem trajectory_of_intersection (A B C D M : Point)
  (hA : A = {x := -1, y := 0})
  (hB : B = {x := 1, y := 0})
  (hC : on_circle C)
  (hD : on_circle D)
  (hCD : perp_to_x_axis C D)
  (hM : ∃ m n : ℝ, C = {x := m, y := n} ∧ M = {x := 1 / m, y := n / m}) :
  M.x^2 - M.y^2 = 1 ∧ M.y ≠ 0 :=
by
  sorry

end trajectory_of_intersection_l1378_137869


namespace correct_result_after_mistakes_l1378_137860

theorem correct_result_after_mistakes (n : ℕ) (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ)
    (h1 : f n 4 * 4 + 18 = g 12 18) : 
    g (f n 4 * 4) 18 = 498 :=
by
  sorry

end correct_result_after_mistakes_l1378_137860


namespace greatest_3_digit_base8_num_div_by_7_eq_511_l1378_137874

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end greatest_3_digit_base8_num_div_by_7_eq_511_l1378_137874


namespace scientific_notation_of_star_diameter_l1378_137884

theorem scientific_notation_of_star_diameter:
    (∃ (c : ℝ) (n : ℕ), 1 ≤ c ∧ c < 10 ∧ 16600000000 = c * 10^n) → 
    16600000000 = 1.66 * 10^10 :=
by
  sorry

end scientific_notation_of_star_diameter_l1378_137884


namespace tina_sales_ratio_l1378_137862

theorem tina_sales_ratio (katya_sales ricky_sales t_sold_more : ℕ) 
  (h_katya : katya_sales = 8) 
  (h_ricky : ricky_sales = 9) 
  (h_tina_sold : t_sold_more = katya_sales + 26) 
  (h_tina_multiple : ∃ m : ℕ, t_sold_more = m * (katya_sales + ricky_sales)) :
  t_sold_more / (katya_sales + ricky_sales) = 2 := 
by 
  sorry

end tina_sales_ratio_l1378_137862


namespace quadratic_real_roots_iff_find_m_given_condition_l1378_137832

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let disc := quadratic_discriminant a b c
  if disc < 0 then (0, 0)
  else ((-b + disc.sqrt) / (2 * a), (-b - disc.sqrt) / (2 * a))

theorem quadratic_real_roots_iff (m : ℝ) :
  (quadratic_discriminant 1 (-2 * (m + 1)) (m ^ 2 + 5) ≥ 0) ↔ (m ≥ 2) :=
by sorry

theorem find_m_given_condition (x1 x2 m : ℝ) (h1 : x1 + x2 = 2 * (m + 1)) (h2 : x1 * x2 = m ^ 2 + 5) (h3 : (x1 - 1) * (x2 - 1) = 28) :
  m = 6 :=
by sorry

end quadratic_real_roots_iff_find_m_given_condition_l1378_137832


namespace largest_of_three_numbers_l1378_137854

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l1378_137854


namespace passes_through_fixed_point_l1378_137896

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x-2) - 3

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -2 :=
by
  sorry

end passes_through_fixed_point_l1378_137896


namespace filling_tank_with_pipes_l1378_137840

theorem filling_tank_with_pipes :
  let Ra := 1 / 70
  let Rb := 2 * Ra
  let Rc := 2 * Rb
  let Rtotal := Ra + Rb + Rc
  Rtotal = 1 / 10 →  -- Given the combined rate fills the tank in 10 hours
  3 = 3 :=  -- Number of pipes used to fill the tank
by
  intros Ra Rb Rc Rtotal h
  simp [Ra, Rb, Rc] at h
  sorry

end filling_tank_with_pipes_l1378_137840


namespace press_x_squared_three_times_to_exceed_10000_l1378_137809

theorem press_x_squared_three_times_to_exceed_10000 :
  ∃ (n : ℕ), n = 3 ∧ (5^(2^n) > 10000) :=
by
  sorry

end press_x_squared_three_times_to_exceed_10000_l1378_137809


namespace clearance_sale_total_earnings_l1378_137803

-- Define the variables used in the problem
def total_jackets := 214
def price_before_noon := 31.95
def price_after_noon := 18.95
def jackets_sold_after_noon := 133

-- Calculate the total earnings
def total_earnings_from_clearance_sale : Prop :=
  (133 * 18.95 + (214 - 133) * 31.95) = 5107.30

-- State the theorem to be proven
theorem clearance_sale_total_earnings : total_earnings_from_clearance_sale :=
  by sorry

end clearance_sale_total_earnings_l1378_137803


namespace arithmetic_sequence_length_correct_l1378_137883

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l1378_137883


namespace apples_given_to_Larry_l1378_137864

-- Define the initial conditions
def initial_apples : ℕ := 75
def remaining_apples : ℕ := 23

-- The statement that we need to prove
theorem apples_given_to_Larry : initial_apples - remaining_apples = 52 :=
by
  -- skip the proof
  sorry

end apples_given_to_Larry_l1378_137864


namespace intersection_M_N_l1378_137881

open Set

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by
  sorry

end intersection_M_N_l1378_137881


namespace find_n_l1378_137802

theorem find_n (n : ℕ) (h : 2 * 2^2 * 2^n = 2^10) : n = 7 :=
sorry

end find_n_l1378_137802


namespace journey_duration_is_9_hours_l1378_137877

noncomputable def journey_time : ℝ :=
  let d1 := 90 -- Distance traveled by Tom and Dick by car before Tom got off
  let d2 := 60 -- Distance Dick backtracked to pick up Harry
  let T := (d1 / 30) + ((120 - d1) / 5) -- Time taken for Tom's journey
  T

theorem journey_duration_is_9_hours : journey_time = 9 := 
by 
  sorry

end journey_duration_is_9_hours_l1378_137877


namespace yeast_population_correct_l1378_137890

noncomputable def yeast_population_estimation 
    (count_per_small_square : ℕ)
    (dimension_large_square : ℝ)
    (dilution_factor : ℝ)
    (thickness : ℝ)
    (total_volume : ℝ) 
    : ℝ :=
    (count_per_small_square:ℝ) / ((dimension_large_square * dimension_large_square * thickness) / 400) * dilution_factor * total_volume

theorem yeast_population_correct:
    yeast_population_estimation 5 1 10 0.1 10 = 2 * 10^9 :=
by
    sorry

end yeast_population_correct_l1378_137890


namespace find_a_plus_b_l1378_137853

noncomputable def f (a b x : ℝ) := a * x + b
noncomputable def g (x : ℝ) := 3 * x - 4

theorem find_a_plus_b (a b : ℝ) (h : ∀ (x : ℝ), g (f a b x) = 4 * x + 5) : a + b = 13 / 3 := 
  sorry

end find_a_plus_b_l1378_137853


namespace composite_of_n_gt_one_l1378_137822

theorem composite_of_n_gt_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by
  sorry

end composite_of_n_gt_one_l1378_137822


namespace smallest_identical_digit_divisible_by_18_l1378_137888

theorem smallest_identical_digit_divisible_by_18 :
  ∃ n : Nat, (∀ d : Nat, d < n → ∃ a : Nat, (n = a * (10 ^ d - 1) / 9 + 1 ∧ (∃ k : Nat, n = 18 * k))) ∧ n = 666 :=
by
  sorry

end smallest_identical_digit_divisible_by_18_l1378_137888


namespace square_triangle_same_area_l1378_137841

theorem square_triangle_same_area (perimeter_square height_triangle : ℤ) (same_area : ℚ) 
  (h_perimeter_square : perimeter_square = 64) 
  (h_height_triangle : height_triangle = 64)
  (h_same_area : same_area = 256) :
  ∃ x : ℚ, x = 8 :=
by
  sorry

end square_triangle_same_area_l1378_137841


namespace max_apartment_size_l1378_137842

theorem max_apartment_size (rental_price_per_sqft : ℝ) (budget : ℝ) (h1 : rental_price_per_sqft = 1.20) (h2 : budget = 720) : 
  budget / rental_price_per_sqft = 600 :=
by 
  sorry

end max_apartment_size_l1378_137842


namespace inverse_function_correct_inequality_solution_l1378_137812

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log (1 - y)

theorem inverse_function_correct (x : ℝ) (hx : -1 < x ∧ x < 1) :
  f_inv (f x) = x :=
sorry

theorem inequality_solution :
  ∀ x, (1 / 2 < x ∧ x < 1) ↔ (f_inv x > Real.log (1 + x) + 1) :=
sorry

end inverse_function_correct_inequality_solution_l1378_137812


namespace value_of_expression_l1378_137826

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end value_of_expression_l1378_137826


namespace infinitely_many_pairs_l1378_137835

theorem infinitely_many_pairs : ∀ b : ℕ, ∃ a : ℕ, 2019 < 2^a / 3^b ∧ 2^a / 3^b < 2020 := 
by
  sorry

end infinitely_many_pairs_l1378_137835


namespace smallest_nonprime_in_range_l1378_137844

def smallest_nonprime_with_no_prime_factors_less_than_20 (m : ℕ) : Prop :=
  ¬(Nat.Prime m) ∧ m > 10 ∧ ∀ p : ℕ, Nat.Prime p → p < 20 → ¬(p ∣ m)

theorem smallest_nonprime_in_range :
  smallest_nonprime_with_no_prime_factors_less_than_20 529 ∧ 520 < 529 ∧ 529 ≤ 540 := 
by 
  sorry

end smallest_nonprime_in_range_l1378_137844


namespace vertex_of_f_C_l1378_137839

def f_A (x : ℝ) : ℝ := (x + 4) ^ 2 - 3
def f_B (x : ℝ) : ℝ := (x + 4) ^ 2 + 3
def f_C (x : ℝ) : ℝ := (x - 4) ^ 2 - 3
def f_D (x : ℝ) : ℝ := (x - 4) ^ 2 + 3

theorem vertex_of_f_C : ∃ (h k : ℝ), h = 4 ∧ k = -3 ∧ ∀ x, f_C x = (x - h) ^ 2 + k :=
by
  sorry

end vertex_of_f_C_l1378_137839


namespace sin_double_angle_l1378_137878

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 4) :
  Real.sin (2 * α) = -3 / 4 :=
sorry

end sin_double_angle_l1378_137878


namespace everton_college_calculators_l1378_137815

theorem everton_college_calculators (total_cost : ℤ) (num_scientific_calculators : ℤ) 
  (cost_per_scientific : ℤ) (cost_per_graphing : ℤ) (total_scientific_cost : ℤ) 
  (num_graphing_calculators : ℤ) (total_graphing_cost : ℤ) (total_calculators : ℤ) :
  total_cost = 1625 ∧
  num_scientific_calculators = 20 ∧
  cost_per_scientific = 10 ∧
  cost_per_graphing = 57 ∧
  total_scientific_cost = num_scientific_calculators * cost_per_scientific ∧
  total_graphing_cost = num_graphing_calculators * cost_per_graphing ∧
  total_cost = total_scientific_cost + total_graphing_cost ∧
  total_calculators = num_scientific_calculators + num_graphing_calculators → 
  total_calculators = 45 :=
by
  intros
  sorry

end everton_college_calculators_l1378_137815


namespace find_s_l_l1378_137894

theorem find_s_l :
  ∃ s l : ℝ, ∀ t : ℝ, 
  (-8 + l * t, s + -6 * t) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p.snd = 3 / 4 * x + 2 ∧ p.fst = x} ∧ 
  (s = -4 ∧ l = -8) :=
by
  sorry

end find_s_l_l1378_137894


namespace min_value_sin_cos_expr_l1378_137836

open Real

theorem min_value_sin_cos_expr :
  (∀ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 ≥ 3 / 5) ∧ 
  (∃ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 = 3 / 5) :=
by
  sorry

end min_value_sin_cos_expr_l1378_137836


namespace shaded_area_is_correct_l1378_137817

-- Conditions definition
def shaded_numbers : ℕ := 2015
def boundary_properties (segment : ℕ) : Prop := 
  segment = 1 ∨ segment = 2

theorem shaded_area_is_correct : ∀ n : ℕ, n = shaded_numbers → boundary_properties n → 
  (∃ area : ℚ, area = 47.5) :=
by
  sorry

end shaded_area_is_correct_l1378_137817


namespace count_yellow_highlighters_l1378_137857

-- Definitions of the conditions
def pink_highlighters : ℕ := 9
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 22

-- Definition based on the question
def yellow_highlighters : ℕ := total_highlighters - (pink_highlighters + blue_highlighters)

-- The theorem to prove the number of yellow highlighters
theorem count_yellow_highlighters : yellow_highlighters = 8 :=
by
  -- Proof omitted as instructed
  sorry

end count_yellow_highlighters_l1378_137857


namespace exists_solution_negation_correct_l1378_137818

theorem exists_solution_negation_correct :
  (∃ x : ℝ, x^2 - x = 0) ↔ (∃ x : ℝ, True) ∧ (∀ x : ℝ, ¬ (x^2 - x = 0)) :=
by
  sorry

end exists_solution_negation_correct_l1378_137818


namespace smallest_unit_of_money_correct_l1378_137898

noncomputable def smallest_unit_of_money (friends : ℕ) (total_bill paid_amount : ℚ) : ℚ :=
  if (total_bill % friends : ℚ) = 0 then
    total_bill / friends
  else
    1 % 100

theorem smallest_unit_of_money_correct :
  smallest_unit_of_money 9 124.15 124.11 = 1 % 100 := 
by
  sorry

end smallest_unit_of_money_correct_l1378_137898


namespace min_value_of_x2_y2_sub_xy_l1378_137879

theorem min_value_of_x2_y2_sub_xy (x y : ℝ) (h : x^2 + y^2 + x * y = 315) : 
  ∃ m : ℝ, (∀ (u v : ℝ), u^2 + v^2 + u * v = 315 → u^2 + v^2 - u * v ≥ m) ∧ m = 105 :=
sorry

end min_value_of_x2_y2_sub_xy_l1378_137879


namespace factor_polynomial_l1378_137806

theorem factor_polynomial (y : ℝ) :
  y^8 - 4 * y^6 + 6 * y^4 - 4 * y^2 + 1 = ((y - 1) * (y + 1))^4 :=
sorry

end factor_polynomial_l1378_137806


namespace translation_correct_l1378_137876

def parabola1 (x : ℝ) : ℝ := -2 * (x + 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -2 * (x - 1)^2 - 1

theorem translation_correct :
  ∀ x : ℝ, parabola2 (x - 3) = parabola1 x - 4 :=
by
  sorry

end translation_correct_l1378_137876


namespace total_original_cost_l1378_137811

theorem total_original_cost (discounted_price1 discounted_price2 discounted_price3 : ℕ) 
  (discount_rate1 discount_rate2 discount_rate3 : ℚ)
  (h1 : discounted_price1 = 4400)
  (h2 : discount_rate1 = 0.56)
  (h3 : discounted_price2 = 3900)
  (h4 : discount_rate2 = 0.35)
  (h5 : discounted_price3 = 2400)
  (h6 : discount_rate3 = 0.20) :
  (discounted_price1 / (1 - discount_rate1) + discounted_price2 / (1 - discount_rate2) 
    + discounted_price3 / (1 - discount_rate3) = 19000) :=
by
  sorry

end total_original_cost_l1378_137811
