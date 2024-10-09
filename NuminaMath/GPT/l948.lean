import Mathlib

namespace parallel_line_plane_no_common_points_l948_94896

noncomputable def line := Type
noncomputable def plane := Type

variable {l : line}
variable {α : plane}

-- Definitions for parallel lines and planes, and relations between lines and planes
def parallel_to_plane (l : line) (α : plane) : Prop := sorry -- Definition of line parallel to plane
def within_plane (m : line) (α : plane) : Prop := sorry -- Definition of line within plane
def no_common_points (l m : line) : Prop := sorry -- Definition of no common points between lines

theorem parallel_line_plane_no_common_points
  (h₁ : parallel_to_plane l α)
  (l2 : line)
  (h₂ : within_plane l2 α) :
  no_common_points l l2 :=
sorry

end parallel_line_plane_no_common_points_l948_94896


namespace positive_integer_divisibility_by_3_l948_94872

theorem positive_integer_divisibility_by_3 (n : ℕ) (h : 0 < n) :
  (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end positive_integer_divisibility_by_3_l948_94872


namespace find_f_of_odd_function_periodic_l948_94864

noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem find_f_of_odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_periodic : ∀ x k : ℤ, f x = f (x + 3 * k))
    (α : ℝ) (h_tan : Real.tan α = 3) :
  f (2015 * Real.sin (2 * (arctan 3))) = 0 :=
sorry

end find_f_of_odd_function_periodic_l948_94864


namespace price_restoration_l948_94846

theorem price_restoration {P : ℝ} (hP : P > 0) :
  (P - 0.85 * P) / (0.85 * P) * 100 = 17.65 :=
by
  sorry

end price_restoration_l948_94846


namespace necessary_but_not_sufficient_l948_94838

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by {
  sorry
}

end necessary_but_not_sufficient_l948_94838


namespace evaluate_expression_l948_94841

theorem evaluate_expression : (Real.sqrt (Real.sqrt 5 ^ 4))^3 = 125 := by
  sorry

end evaluate_expression_l948_94841


namespace min_value_of_M_l948_94853

noncomputable def min_val (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :=
  max (1/(a*c) + b) (max (1/a + b*c) (a/b + c))

theorem min_value_of_M (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (min_val a b c h1 h2 h3) >= 2 :=
sorry

end min_value_of_M_l948_94853


namespace fraction_to_decimal_l948_94881

theorem fraction_to_decimal : (47 : ℝ) / 160 = 0.29375 :=
by
  sorry

end fraction_to_decimal_l948_94881


namespace min_fencing_dims_l948_94875

theorem min_fencing_dims (x : ℕ) (h₁ : x * (x + 5) ≥ 600) (h₂ : x = 23) : 
  2 * (x + (x + 5)) = 102 := 
by
  -- Placeholder for the proof
  sorry

end min_fencing_dims_l948_94875


namespace expression_value_l948_94882

-- Define the variables and the main statement
variable (w x y z : ℕ)

theorem expression_value :
  2^w * 3^x * 5^y * 11^z = 825 → w + 2 * x + 3 * y + 4 * z = 12 :=
by
  sorry -- Proof omitted

end expression_value_l948_94882


namespace midpoint_sum_coordinates_l948_94899

theorem midpoint_sum_coordinates :
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 9 :=
by
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint.1 + midpoint.2 = 9
  sorry

end midpoint_sum_coordinates_l948_94899


namespace expand_expression_l948_94863

theorem expand_expression (x y : ℕ) : 
  (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 :=
by 
  sorry

end expand_expression_l948_94863


namespace car_speed_ratio_l948_94859

theorem car_speed_ratio (v_A v_B : ℕ) (h1 : v_B = 50) (h2 : 6 * v_A + 2 * v_B = 1000) :
  v_A / v_B = 3 :=
sorry

end car_speed_ratio_l948_94859


namespace determine_a_value_l948_94842

theorem determine_a_value (a : ℤ) (h : ∀ x : ℝ, x^2 + 2 * (a:ℝ) * x + 1 > 0) : a = 0 := 
sorry

end determine_a_value_l948_94842


namespace symmetric_angle_set_l948_94818

theorem symmetric_angle_set (α β : ℝ) (k : ℤ) 
  (h1 : β = 2 * (k : ℝ) * Real.pi + Real.pi / 12)
  (h2 : α = -Real.pi / 3)
  (symmetric : α + β = -Real.pi / 4) :
  ∃ k : ℤ, β = 2 * (k : ℝ) * Real.pi + Real.pi / 12 :=
sorry

end symmetric_angle_set_l948_94818


namespace max_servings_possible_l948_94874

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l948_94874


namespace discriminant_of_given_quadratic_l948_94854

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l948_94854


namespace binding_cost_is_correct_l948_94821

-- Definitions for the conditions used in the problem
def total_cost : ℝ := 250      -- Total cost to copy and bind 10 manuscripts
def copy_cost_per_page : ℝ := 0.05   -- Cost per page to copy
def pages_per_manuscript : ℕ := 400  -- Number of pages in each manuscript
def num_manuscripts : ℕ := 10      -- Number of manuscripts

-- The target value we want to prove
def binding_cost_per_manuscript : ℝ := 5 

-- The theorem statement proving the binding cost per manuscript
theorem binding_cost_is_correct :
  let copy_cost_per_manuscript := pages_per_manuscript * copy_cost_per_page
  let total_copy_cost := num_manuscripts * copy_cost_per_manuscript
  let total_binding_cost := total_cost - total_copy_cost
  (total_binding_cost / num_manuscripts) = binding_cost_per_manuscript :=
by
  sorry

end binding_cost_is_correct_l948_94821


namespace ants_crushed_l948_94848

theorem ants_crushed {original_ants left_ants crushed_ants : ℕ} 
  (h1 : original_ants = 102) 
  (h2 : left_ants = 42) 
  (h3 : crushed_ants = original_ants - left_ants) : 
  crushed_ants = 60 := 
by
  sorry

end ants_crushed_l948_94848


namespace find_s_t_l948_94893

noncomputable def problem_constants (a b c : ℝ) : Prop :=
  (a^3 + 3 * a^2 + 4 * a - 11 = 0) ∧
  (b^3 + 3 * b^2 + 4 * b - 11 = 0) ∧
  (c^3 + 3 * c^2 + 4 * c - 11 = 0)

theorem find_s_t (a b c s t : ℝ) (h1 : problem_constants a b c) (h2 : (a + b) * (b + c) * (c + a) = -t)
  (h3 : (a + b) * (b + c) + (b + c) * (c + a) + (c + a) * (a + b) = s) :
s = 8 ∧ t = 23 :=
sorry

end find_s_t_l948_94893


namespace repeating_decimals_fraction_l948_94852

theorem repeating_decimals_fraction :
  (0.81:ℚ) / (0.36:ℚ) = 9 / 4 :=
by
  have h₁ : (0.81:ℚ) = 81 / 99 := sorry
  have h₂ : (0.36:ℚ) = 36 / 99 := sorry
  sorry

end repeating_decimals_fraction_l948_94852


namespace imperative_sentence_structure_l948_94892

theorem imperative_sentence_structure (word : String) (is_base_form : word = "Surround") :
  (word = "Surround" ∨ word = "Surrounding" ∨ word = "Surrounded" ∨ word = "Have surrounded") →
  (∃ sentence : String, sentence = word ++ " yourself with positive people, and you will keep focused on what you can do instead of what you can’t.") →
  word = "Surround" :=
by
  intros H_choice H_sentence
  cases H_choice
  case inl H1 => assumption
  case inr H2_1 =>
    cases H2_1
    case inl H2_1_1 => sorry
    case inr H2_1_2 =>
      cases H2_1_2
      case inl H2_1_2_1 => sorry
      case inr H2_1_2_2 => sorry

end imperative_sentence_structure_l948_94892


namespace smallest_n_for_gcd_lcm_l948_94804

theorem smallest_n_for_gcd_lcm (n a b : ℕ) (h_gcd : Nat.gcd a b = 999) (h_lcm : Nat.lcm a b = Nat.factorial n) :
  n = 37 := sorry

end smallest_n_for_gcd_lcm_l948_94804


namespace Sophie_donuts_l948_94810

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l948_94810


namespace find_number_l948_94819

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 100) : N = 192 :=
sorry

end find_number_l948_94819


namespace find_positive_solutions_l948_94827

noncomputable def satisfies_eq1 (x y : ℝ) : Prop :=
  2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0

noncomputable def satisfies_eq2 (x y : ℝ) : Prop :=
  2 * x^2 + x^2 * y^4 = 18 * y^2

theorem find_positive_solutions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  satisfies_eq1 x y ∧ satisfies_eq2 x y ↔ 
  (x = 2 ∧ y = 2) ∨ 
  (x = Real.sqrt 286^(1/4) / 4 ∧ y = Real.sqrt 286^(1/4)) :=
sorry

end find_positive_solutions_l948_94827


namespace incorrect_conversion_l948_94851

/--
Incorrect conversion of -150° to radians.
-/
theorem incorrect_conversion : (¬(((-150 : ℝ) * (Real.pi / 180)) = (-7 * Real.pi / 6))) :=
by
  sorry

end incorrect_conversion_l948_94851


namespace inequality_proof_l948_94837

theorem inequality_proof (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
(h7 : a * y + b * x = c) (h8 : c * x + a * z = b) 
(h9 : b * z + c * y = a) :
x / (1 - y * z) + y / (1 - z * x) + z / (1 - x * y) ≤ 2 :=
sorry

end inequality_proof_l948_94837


namespace original_useful_item_is_pencil_l948_94812

def code_language (x : String) : String :=
  if x = "item" then "pencil"
  else if x = "pencil" then "mirror"
  else if x = "mirror" then "board"
  else x

theorem original_useful_item_is_pencil : 
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") ∧
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") 
  → "mirror" = "pencil" :=
by sorry

end original_useful_item_is_pencil_l948_94812


namespace hillary_climbing_rate_l948_94816

theorem hillary_climbing_rate :
  ∀ (H : ℕ) (Eddy_rate : ℕ) (Hillary_climb : ℕ) (Hillary_descend_rate : ℕ) (pass_time : ℕ) (start_to_summit : ℕ),
    Eddy_rate = 500 →
    Hillary_climb = 4000 →
    Hillary_descend_rate = 1000 →
    pass_time = 6 →
    start_to_summit = 5000 →
    (Hillary_climb + Eddy_rate * pass_time = Hillary_climb + (pass_time - Hillary_climb / H) * Hillary_descend_rate) →
    H = 800 :=
by
  intros H Eddy_rate Hillary_climb Hillary_descend_rate pass_time start_to_summit
  intro h1 h2 h3 h4 h5 h6
  sorry

end hillary_climbing_rate_l948_94816


namespace cost_for_sugar_substitutes_l948_94839

def packets_per_cup : ℕ := 1
def cups_per_day : ℕ := 2
def days : ℕ := 90
def packets_per_box : ℕ := 30
def price_per_box : ℕ := 4

theorem cost_for_sugar_substitutes : 
  (packets_per_cup * cups_per_day * days / packets_per_box) * price_per_box = 24 := by
  sorry

end cost_for_sugar_substitutes_l948_94839


namespace fish_population_l948_94855

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l948_94855


namespace successful_experimental_operation_l948_94828

/-- Problem statement:
Given the following biological experimental operations:
1. spreading diluted E. coli culture on solid medium,
2. introducing sterile air into freshly inoculated grape juice with yeast,
3. inoculating soil leachate on beef extract peptone medium,
4. using slightly opened rose flowers as experimental material for anther culture.

Prove that spreading diluted E. coli culture on solid medium can successfully achieve the experimental objective of obtaining single colonies.
-/
theorem successful_experimental_operation :
  ∃ objective_result,
    (objective_result = "single_colonies" →
     let operation_A := "spreading diluted E. coli culture on solid medium"
     let operation_B := "introducing sterile air into freshly inoculated grape juice with yeast"
     let operation_C := "inoculating soil leachate on beef extract peptone medium"
     let operation_D := "slightly opened rose flowers as experimental material for anther culture"
     ∃ successful_operation,
       successful_operation = operation_A
       ∧ (successful_operation = operation_A → objective_result = "single_colonies")
       ∧ (successful_operation = operation_B → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_C → objective_result ≠ "single_colonies")
       ∧ (successful_operation = operation_D → objective_result ≠ "single_colonies")) :=
sorry

end successful_experimental_operation_l948_94828


namespace find_int_solutions_l948_94860

theorem find_int_solutions (x : ℤ) :
  (∃ p : ℤ, Prime p ∧ 2*x^2 - x - 36 = p^2) ↔ (x = 5 ∨ x = 13) := 
sorry

end find_int_solutions_l948_94860


namespace quadratic_roots_l948_94849

theorem quadratic_roots (k : ℝ) :
  (∃ x : ℝ, x = 2 ∧ 4 * x ^ 2 - k * x + 6 = 0) →
  k = 11 ∧ (∃ x : ℝ, x ≠ 2 ∧ 4 * x ^ 2 - 11 * x + 6 = 0 ∧ x = 3 / 4) := 
by
  sorry

end quadratic_roots_l948_94849


namespace intersection_of_sets_l948_94844

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets : (setA ∩ { x | 1 - x^2 ∈ setB }) = Set.Icc (-1) 1 :=
by
  sorry

end intersection_of_sets_l948_94844


namespace range_of_a_l948_94845

theorem range_of_a (x a : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) : -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l948_94845


namespace correct_number_l948_94895

theorem correct_number : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  -- proof starts here
  sorry

end correct_number_l948_94895


namespace arrange_3x3_grid_l948_94890

-- Define the problem conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := ¬ is_odd n

-- Define the function to count the number of such arrangements
noncomputable def count_arrangements : ℕ :=
  6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9

-- State the main theorem
theorem arrange_3x3_grid (nums : ℕ → Prop) (table : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 7) :
  (∀ i, is_odd (table i 0 + table i 1 + table i 2)) ∧ (∀ j, is_odd (table 0 j + table 1 j + table 2 j)) →
  count_arrangements = 6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9 :=
by sorry

end arrange_3x3_grid_l948_94890


namespace female_democrats_ratio_l948_94865

theorem female_democrats_ratio 
  (M F : ℕ) 
  (H1 : M + F = 660)
  (H2 : (1 / 3 : ℝ) * 660 = 220)
  (H3 : ∃ dem_males : ℕ, dem_males = (1 / 4 : ℝ) * M)
  (H4 : ∃ dem_females : ℕ, dem_females = 110) :
  110 / F = 1 / 2 :=
by
  sorry

end female_democrats_ratio_l948_94865


namespace son_l948_94873

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l948_94873


namespace extra_fruits_l948_94884

theorem extra_fruits (r g s : Nat) (hr : r = 42) (hg : g = 7) (hs : s = 9) : r + g - s = 40 :=
by
  sorry

end extra_fruits_l948_94884


namespace arrangement_ways_l948_94898

def num_ways_arrange_boys_girls : Nat :=
  let boys := 2
  let girls := 3
  let ways_girls := Nat.factorial girls
  let ways_boys := Nat.factorial boys
  ways_girls * ways_boys

theorem arrangement_ways : num_ways_arrange_boys_girls = 12 :=
  by
    sorry

end arrangement_ways_l948_94898


namespace find_x_eq_e_l948_94805

noncomputable def f (x : ℝ) : ℝ := x + x * (Real.log x) ^ 2

noncomputable def f' (x : ℝ) : ℝ :=
  1 + (Real.log x) ^ 2 + 2 * Real.log x

theorem find_x_eq_e : ∃ (x : ℝ), (x * f' x = 2 * f x) ∧ (x = Real.exp 1) :=
by
  sorry

end find_x_eq_e_l948_94805


namespace staff_price_l948_94833

theorem staff_price (d : ℝ) : (d - 0.55 * d) / 2 = 0.225 * d := by
  sorry

end staff_price_l948_94833


namespace sum_of_distinct_digits_base6_l948_94808

theorem sum_of_distinct_digits_base6 (A B C : ℕ) (hA : A < 6) (hB : B < 6) (hC : C < 6) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_first_col : C + C % 6 = 4)
  (h_second_col : B + B % 6 = C)
  (h_third_col : A + B % 6 = A) :
  A + B + C = 6 := by
  sorry

end sum_of_distinct_digits_base6_l948_94808


namespace John_writing_years_l948_94809

def books_written (total_earnings per_book_earning : ℕ) : ℕ :=
  total_earnings / per_book_earning

def books_per_year (months_in_year months_per_book : ℕ) : ℕ :=
  months_in_year / months_per_book

def years_writing (total_books books_per_year : ℕ) : ℕ :=
  total_books / books_per_year

theorem John_writing_years :
  let total_earnings := 3600000
  let per_book_earning := 30000
  let months_in_year := 12
  let months_per_book := 2
  let total_books := books_written total_earnings per_book_earning
  let books_per_year := books_per_year months_in_year months_per_book
  years_writing total_books books_per_year = 20 := by
sorry

end John_writing_years_l948_94809


namespace manuscript_copy_cost_l948_94866

theorem manuscript_copy_cost (total_cost : ℝ) (binding_cost : ℝ) (num_manuscripts : ℕ) (pages_per_manuscript : ℕ) (x : ℝ) :
  total_cost = 250 ∧ binding_cost = 5 ∧ num_manuscripts = 10 ∧ pages_per_manuscript = 400 →
  x = (total_cost - binding_cost * num_manuscripts) / (num_manuscripts * pages_per_manuscript) →
  x = 0.05 :=
by
  sorry

end manuscript_copy_cost_l948_94866


namespace remainder_of_E_div_88_l948_94826

-- Define the given expression E and the binomial coefficient 
noncomputable def E : ℤ :=
  1 - 90 * Nat.choose 10 1 + 90 ^ 2 * Nat.choose 10 2 - 90 ^ 3 * Nat.choose 10 3 + 
  90 ^ 4 * Nat.choose 10 4 - 90 ^ 5 * Nat.choose 10 5 + 90 ^ 6 * Nat.choose 10 6 - 
  90 ^ 7 * Nat.choose 10 7 + 90 ^ 8 * Nat.choose 10 8 - 90 ^ 9 * Nat.choose 10 9 + 
  90 ^ 10 * Nat.choose 10 10

-- The theorem that we need to prove
theorem remainder_of_E_div_88 : E % 88 = 1 := by
  sorry

end remainder_of_E_div_88_l948_94826


namespace circle_center_line_distance_l948_94883

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l948_94883


namespace shortest_side_of_similar_triangle_l948_94887

theorem shortest_side_of_similar_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2)
  (scale_factor : ℝ) (shortest_side_first : ℝ) (hypo_second : ℝ)
  (h4 : scale_factor = 100 / 25) 
  (h5 : hypo_second = 100) 
  (h6 : b = 7) 
  : (shortest_side_first * scale_factor = 28) :=
by
  sorry

end shortest_side_of_similar_triangle_l948_94887


namespace distance_walked_is_4_point_6_l948_94840

-- Define the number of blocks Sarah walked in each direction
def blocks_west : ℕ := 8
def blocks_south : ℕ := 15

-- Define the length of each block in miles
def block_length : ℚ := 1 / 5

-- Calculate the total number of blocks
def total_blocks : ℕ := blocks_west + blocks_south

-- Calculate the total distance walked in miles
def total_distance_walked : ℚ := total_blocks * block_length

-- Statement to prove the total distance walked is 4.6 miles
theorem distance_walked_is_4_point_6 : total_distance_walked = 4.6 := sorry

end distance_walked_is_4_point_6_l948_94840


namespace chocolates_initial_count_l948_94831

theorem chocolates_initial_count (remaining_chocolates: ℕ) 
    (daily_percentage: ℝ) (days: ℕ) 
    (final_chocolates: ℝ) 
    (remaining_fraction_proof: remaining_fraction = 0.7) 
    (days_proof: days = 3) 
    (final_chocolates_proof: final_chocolates = 28): 
    (remaining_fraction^days * (initial_chocolates:ℝ) = final_chocolates) → 
    (initial_chocolates = 82) := 
by 
  sorry

end chocolates_initial_count_l948_94831


namespace circle_possible_m_values_l948_94800

theorem circle_possible_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + m * x - m * y + 2 = 0) ↔ m > 2 ∨ m < -2 :=
by
  sorry

end circle_possible_m_values_l948_94800


namespace find_g_neg3_l948_94807

def f (x : ℚ) : ℚ := 4 * x - 6
def g (u : ℚ) : ℚ := 3 * (f u)^2 + 4 * (f u) - 2

theorem find_g_neg3 : g (-3) = 43 / 16 := by
  sorry

end find_g_neg3_l948_94807


namespace best_regression_effect_l948_94880

theorem best_regression_effect (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.36)
  (h2 : R2_2 = 0.95)
  (h3 : R2_3 = 0.74)
  (h4 : R2_4 = 0.81):
  max (max (max R2_1 R2_2) R2_3) R2_4 = 0.95 := by
  sorry

end best_regression_effect_l948_94880


namespace sequence_term_formula_l948_94889

open Real

def sequence_sum_condition (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S n + a n = 4 - 1 / (2 ^ (n - 2))

theorem sequence_term_formula 
  (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : sequence_sum_condition S a) :
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) :=
sorry

end sequence_term_formula_l948_94889


namespace probability_bons_wins_even_rolls_l948_94832
noncomputable def probability_of_Bons_winning (p6 : ℚ) (p_not6 : ℚ) : ℚ := 
  let r := p_not6^2
  let a := p_not6 * p6
  a / (1 - r)

theorem probability_bons_wins_even_rolls : 
  let p6 := (1 : ℚ) / 6
  let p_not6 := (5 : ℚ) / 6
  probability_of_Bons_winning p6 p_not6 = (5 : ℚ) / 11 := 
  sorry

end probability_bons_wins_even_rolls_l948_94832


namespace trajectory_midpoint_of_chord_l948_94850

theorem trajectory_midpoint_of_chord :
  ∀ (M: ℝ × ℝ), (∃ (C D : ℝ × ℝ), (C.1^2 + C.2^2 = 25 ∧ D.1^2 + D.2^2 = 25 ∧ dist C D = 8) ∧ M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  → M.1^2 + M.2^2 = 9 :=
sorry

end trajectory_midpoint_of_chord_l948_94850


namespace divisibility_by_3_l948_94856

theorem divisibility_by_3 (x y z : ℤ) (h : x^3 + y^3 = z^3) : 3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := 
sorry

end divisibility_by_3_l948_94856


namespace total_enemies_l948_94830

theorem total_enemies (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) (enemies_defeated : ℕ) :  
  (3 = points_per_enemy) → 
  (12 = points_earned) → 
  (2 = enemies_left) → 
  (points_earned / points_per_enemy = enemies_defeated) → 
  (enemies_defeated + enemies_left = 6) := 
by
  intros
  sorry

end total_enemies_l948_94830


namespace three_digit_divisible_by_7_iff_last_two_digits_equal_l948_94876

-- Define the conditions as given in the problem
variable (a b c : ℕ)

-- Ensure the sum of the digits is 7, as given by the problem conditions
def sum_of_digits_eq_7 := a + b + c = 7

-- Ensure that it is a three-digit number
def valid_three_digit_number := a ≠ 0

-- Define what it means to be divisible by 7
def divisible_by_7 (n : ℕ) := n % 7 = 0

-- Define the problem statement in Lean
theorem three_digit_divisible_by_7_iff_last_two_digits_equal (h1 : sum_of_digits_eq_7 a b c) (h2 : valid_three_digit_number a) :
  divisible_by_7 (100 * a + 10 * b + c) ↔ b = c :=
by sorry

end three_digit_divisible_by_7_iff_last_two_digits_equal_l948_94876


namespace algebraic_expression_value_l948_94871

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 :=
by
  sorry

end algebraic_expression_value_l948_94871


namespace solution_set_of_inequality_l948_94894

theorem solution_set_of_inequality (x : ℝ) : x * (x + 3) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 0 :=
by
  sorry

end solution_set_of_inequality_l948_94894


namespace sum_of_integers_is_23_l948_94817

theorem sum_of_integers_is_23
  (x y : ℕ) (x_pos : 0 < x) (y_pos : 0 < y) (h : x * y + x + y = 155) 
  (rel_prime : Nat.gcd x y = 1) (x_lt_30 : x < 30) (y_lt_30 : y < 30) :
  x + y = 23 :=
by
  sorry

end sum_of_integers_is_23_l948_94817


namespace g_15_33_eq_165_l948_94879

noncomputable def g : ℕ → ℕ → ℕ := sorry

axiom g_self (x : ℕ) : g x x = x
axiom g_comm (x y : ℕ) : g x y = g y x
axiom g_equation (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_33_eq_165 : g 15 33 = 165 := by sorry

end g_15_33_eq_165_l948_94879


namespace sally_popped_3_balloons_l948_94869

-- Defining the conditions
def joans_initial_balloons : ℕ := 9
def jessicas_balloons : ℕ := 2
def total_balloons_now : ℕ := 6

-- Definition for the number of balloons Sally popped
def sally_balloons_popped : ℕ := joans_initial_balloons - (total_balloons_now - jessicas_balloons)

-- The theorem statement
theorem sally_popped_3_balloons : sally_balloons_popped = 3 := 
by
  -- Proof omitted; use sorry
  sorry

end sally_popped_3_balloons_l948_94869


namespace flat_fee_l948_94815

theorem flat_fee (f n : ℝ) (h1 : f + 3 * n = 215) (h2 : f + 6 * n = 385) : f = 45 :=
  sorry

end flat_fee_l948_94815


namespace evaluate_expression_l948_94877

theorem evaluate_expression :
  8^6 * 27^6 * 8^15 * 27^15 = 216^21 :=
by
  sorry

end evaluate_expression_l948_94877


namespace value_of_x_yplusz_l948_94868

theorem value_of_x_yplusz (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 :=
by
  sorry

end value_of_x_yplusz_l948_94868


namespace min_value_range_l948_94843

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) → (0 < a ∧ a ≤ 1) :=
by 
  sorry

end min_value_range_l948_94843


namespace range_of_a_l948_94888

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2 * t - a < 0) ↔ a ≤ -1 :=
by sorry

end range_of_a_l948_94888


namespace factorize_expression_l948_94835

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l948_94835


namespace sum_of_coefficients_polynomial_expansion_l948_94847

theorem sum_of_coefficients_polynomial_expansion :
  let polynomial := (2 * (1 : ℤ) + 3)^5
  ∃ b_5 b_4 b_3 b_2 b_1 b_0 : ℤ,
  polynomial = b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0 ∧
  (b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 3125 :=
by
  sorry

end sum_of_coefficients_polynomial_expansion_l948_94847


namespace water_flow_into_sea_per_minute_l948_94820

noncomputable def river_flow_rate_kmph : ℝ := 4
noncomputable def river_depth_m : ℝ := 5
noncomputable def river_width_m : ℝ := 19
noncomputable def hours_to_minutes : ℝ := 60
noncomputable def km_to_m : ℝ := 1000

noncomputable def flow_rate_m_per_min : ℝ := (river_flow_rate_kmph * km_to_m) / hours_to_minutes
noncomputable def cross_sectional_area_m2 : ℝ := river_depth_m * river_width_m
noncomputable def volume_per_minute_m3 : ℝ := cross_sectional_area_m2 * flow_rate_m_per_min

theorem water_flow_into_sea_per_minute :
  volume_per_minute_m3 = 6333.65 := by 
  -- Proof would go here
  sorry

end water_flow_into_sea_per_minute_l948_94820


namespace evaluate_g_at_3_l948_94897

def g (x : ℤ) : ℤ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem evaluate_g_at_3 : g 3 = 113 := by
  -- Proof of g(3) = 113 skipped
  sorry

end evaluate_g_at_3_l948_94897


namespace total_action_figures_l948_94861

def jerry_original_count : Nat := 4
def jerry_added_count : Nat := 6

theorem total_action_figures : jerry_original_count + jerry_added_count = 10 :=
by
  sorry

end total_action_figures_l948_94861


namespace average_of_remaining_two_l948_94867

theorem average_of_remaining_two (a1 a2 a3 a4 a5 a6 : ℝ)
    (h_avg6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
    (h_avg2_1 : (a1 + a2) / 2 = 3.4)
    (h_avg2_2 : (a3 + a4) / 2 = 3.85) :
    (a5 + a6) / 2 = 4.6 := 
sorry

end average_of_remaining_two_l948_94867


namespace mateen_backyard_area_l948_94870

theorem mateen_backyard_area :
  (∀ (L : ℝ), 30 * L = 1200) →
  (∀ (P : ℝ), 12 * P = 1200) →
  (∃ (L W : ℝ), 2 * L + 2 * W = 100 ∧ L * W = 400) := by
  intros hL hP
  use 40
  use 10
  apply And.intro
  sorry
  sorry

end mateen_backyard_area_l948_94870


namespace fraction_of_earth_surface_humans_can_inhabit_l948_94814

theorem fraction_of_earth_surface_humans_can_inhabit :
  (1 / 3) * (2 / 3) = (2 / 9) :=
by
  sorry

end fraction_of_earth_surface_humans_can_inhabit_l948_94814


namespace Jerry_remaining_pages_l948_94857

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l948_94857


namespace ratio_of_doctors_to_nurses_l948_94806

def total_staff : ℕ := 250
def nurses : ℕ := 150
def doctors : ℕ := total_staff - nurses

theorem ratio_of_doctors_to_nurses : 
  (doctors : ℚ) / (nurses : ℚ) = 2 / 3 := by
  sorry

end ratio_of_doctors_to_nurses_l948_94806


namespace maximum_value_of_f_on_interval_l948_94822

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3

theorem maximum_value_of_f_on_interval :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 3) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 57 :=
by
  sorry

end maximum_value_of_f_on_interval_l948_94822


namespace find_special_three_digit_numbers_l948_94825

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l948_94825


namespace banquet_food_consumption_l948_94836

theorem banquet_food_consumption (n : ℕ) (food_per_guest : ℕ) (total_food : ℕ) 
  (h1 : ∀ g : ℕ, g ≤ n -> g * food_per_guest ≤ total_food)
  (h2 : n = 169) 
  (h3 : food_per_guest = 2) :
  total_food = 338 := 
sorry

end banquet_food_consumption_l948_94836


namespace cheapest_third_company_l948_94858

theorem cheapest_third_company (x : ℕ) :
  (120 + 18 * x ≥ 150 + 15 * x) ∧ (220 + 13 * x ≥ 150 + 15 * x) → 36 ≤ x :=
by
  intro h
  cases h with
  | intro h1 h2 =>
    sorry

end cheapest_third_company_l948_94858


namespace ice_cream_depth_l948_94801

noncomputable def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3
noncomputable def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

theorem ice_cream_depth
  (radius_sphere : ℝ)
  (radius_cylinder : ℝ)
  (density_constancy : volume_sphere radius_sphere = volume_cylinder radius_cylinder (h : ℝ)) :
  h = 9 / 25 := by
  sorry

end ice_cream_depth_l948_94801


namespace torn_out_sheets_count_l948_94824

theorem torn_out_sheets_count :
  ∃ (sheets : ℕ), (first_page = 185 ∧
                   last_page = 518 ∧
                   pages_torn_out = last_page - first_page + 1 ∧ 
                   sheets = pages_torn_out / 2 ∧
                   sheets = 167) :=
by
  sorry

end torn_out_sheets_count_l948_94824


namespace sequences_correct_l948_94813

def arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def geometric_sequence (b a₁ b₁ : ℕ) : Prop :=
  a₁ * a₁ = b * b₁

noncomputable def sequence_a (n : ℕ) :=
  (n * (n + 1)) / 2

noncomputable def sequence_b (n : ℕ) :=
  ((n + 1) * (n + 1)) / 2

theorem sequences_correct :
  (∀ n : ℕ,
    n ≥ 1 →
    arithmetic_sequence (sequence_a n) (sequence_b n) (sequence_a (n + 1)) ∧
    geometric_sequence (sequence_b n) (sequence_a (n + 1)) (sequence_b (n + 1))) ∧
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (sequence_a 2 = 3) :=
by
  sorry

end sequences_correct_l948_94813


namespace daily_shoppers_correct_l948_94803

noncomputable def daily_shoppers (P : ℝ) : Prop :=
  let weekly_taxes : ℝ := 6580
  let daily_taxes := weekly_taxes / 7
  let percent_taxes := 0.94
  percent_taxes * P = daily_taxes

theorem daily_shoppers_correct : ∃ P : ℝ, daily_shoppers P ∧ P = 1000 :=
by
  sorry

end daily_shoppers_correct_l948_94803


namespace pizza_volume_piece_l948_94834

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l948_94834


namespace multiplier_for_deans_height_l948_94811

theorem multiplier_for_deans_height (h_R : ℕ) (h_R_eq : h_R = 13) (d : ℕ) (d_eq : d = 255) (h_D : ℕ) (h_D_eq : h_D = h_R + 4) : 
  d / h_D = 15 := by
  sorry

end multiplier_for_deans_height_l948_94811


namespace inequality_condition_l948_94885

theorem inequality_condition 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c > Real.sqrt (a^2 + b^2)) := 
sorry

end inequality_condition_l948_94885


namespace arithmetic_seq_sum_l948_94862

theorem arithmetic_seq_sum (a : ℕ → ℤ) (a1 a7 a3 a5 : ℤ) (S7 : ℤ)
  (h1 : a1 = a 1) (h7 : a7 = a 7) (h3 : a3 = a 3) (h5 : a5 = a 5)
  (h_arith : ∀ n m, a (n + m) = a n + a m - a 0)
  (h_S7 : (7 * (a1 + a7)) / 2 = 14) :
  a3 + a5 = 4 :=
sorry

end arithmetic_seq_sum_l948_94862


namespace face_value_of_ticket_l948_94886
noncomputable def face_value_each_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) : ℝ :=
  total_price / (group_size * (1 + tax_rate))

theorem face_value_of_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) :
  total_price = 945 →
  group_size = 25 →
  tax_rate = 0.05 →
  face_value_each_ticket total_price group_size tax_rate = 36 := 
by
  intros h_total_price h_group_size h_tax_rate
  rw [h_total_price, h_group_size, h_tax_rate]
  simp [face_value_each_ticket]
  sorry

end face_value_of_ticket_l948_94886


namespace remainder_pow_700_eq_one_l948_94891

theorem remainder_pow_700_eq_one (number : ℤ) (h : number ^ 700 % 100 = 1) : number ^ 700 % 100 = 1 :=
  by
  exact h

end remainder_pow_700_eq_one_l948_94891


namespace total_accepted_cartons_l948_94823

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l948_94823


namespace investment_C_l948_94878

theorem investment_C (A_invest B_invest profit_A total_profit C_invest : ℕ)
  (hA_invest : A_invest = 6300) 
  (hB_invest : B_invest = 4200) 
  (h_profit_A : profit_A = 3900) 
  (h_total_profit : total_profit = 13000) 
  (h_proportional : profit_A / total_profit = A_invest / (A_invest + B_invest + C_invest)) :
  C_invest = 10500 := by
  sorry

end investment_C_l948_94878


namespace six_by_six_board_partition_l948_94802

theorem six_by_six_board_partition (P : Prop) (Q : Prop) 
(board : ℕ × ℕ) (domino : ℕ × ℕ) 
(h1 : board = (6, 6)) 
(h2 : domino = (2, 1)) 
(h3 : P → Q ∧ Q → P) :
  ∃ R₁ R₂ : ℕ × ℕ, (R₁ = (p, q) ∧ R₂ = (r, s) ∧ ((R₁.1 * R₁.2 + R₂.1 * R₂.2) = 36)) :=
sorry

end six_by_six_board_partition_l948_94802


namespace triangle_is_isosceles_l948_94829

open Real

variables (α β γ : ℝ) (a b : ℝ)

theorem triangle_is_isosceles
(h1 : a + b = tan (γ / 2) * (a * tan α + b * tan β)) :
α = β :=
by
  sorry

end triangle_is_isosceles_l948_94829
