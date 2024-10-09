import Mathlib

namespace minimum_k_exists_l1848_184819

theorem minimum_k_exists :
  ∀ (s : Finset ℝ), s.card = 3 → (∀ (a b : ℝ), a ∈ s → b ∈ s → (|a - b| ≤ (1.5 : ℝ) ∨ |(1 / a) - (1 / b)| ≤ 1.5)) :=
by
  sorry

end minimum_k_exists_l1848_184819


namespace min_value_of_f_l1848_184848

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem min_value_of_f :
  (∀ x : ℝ, x > 1 → f x ≥ (5 + 4 * Real.sqrt 3)) ∧
  (f (1 + 2 * Real.sqrt 3 / 3) = 5 + 4 * Real.sqrt 3) :=
by
  sorry

end min_value_of_f_l1848_184848


namespace matrix_det_l1848_184809

def matrix := ![
  ![2, -4, 2],
  ![0, 6, -1],
  ![5, -3, 1]
]

theorem matrix_det : Matrix.det matrix = -34 := by
  sorry

end matrix_det_l1848_184809


namespace factor_expression_l1848_184806

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2 * x + 1) * (2 * x - 1) :=
sorry

end factor_expression_l1848_184806


namespace eight_b_value_l1848_184875

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 16 :=
by
  sorry

end eight_b_value_l1848_184875


namespace Lisa_flight_time_l1848_184854

theorem Lisa_flight_time :
  let distance := 500
  let speed := 45
  (distance : ℝ) / (speed : ℝ) = 500 / 45 := by
  sorry

end Lisa_flight_time_l1848_184854


namespace range_of_m_l1848_184890

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l1848_184890


namespace blisters_on_rest_of_body_l1848_184807

theorem blisters_on_rest_of_body (blisters_per_arm total_blisters : ℕ) (h1 : blisters_per_arm = 60) (h2 : total_blisters = 200) : 
  total_blisters - 2 * blisters_per_arm = 80 :=
by {
  -- The proof can be written here
  sorry
}

end blisters_on_rest_of_body_l1848_184807


namespace part1_part2_part3_l1848_184874

variable (a b c : ℝ) (f : ℝ → ℝ)
-- Defining the polynomial function f
def polynomial (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem part1 (h0 : polynomial a b 6 0 = 6) : c = 6 :=
by sorry

theorem part2 (h1 : polynomial a b (-2) 0 = -2) (h2 : polynomial a b (-2) 1 = 5) : polynomial a b (-2) (-1) = -9 :=
by sorry

theorem part3 (h3 : polynomial a b 3 5 + polynomial a b 3 (-5) = 6) (h4 : polynomial a b 3 2 = 8) : polynomial a b 3 (-2) = -2 :=
by sorry

end part1_part2_part3_l1848_184874


namespace multiplication_problem_division_problem_l1848_184856

theorem multiplication_problem :
  125 * 76 * 4 * 8 * 25 = 7600000 :=
sorry

theorem division_problem :
  (6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741 :=
sorry

end multiplication_problem_division_problem_l1848_184856


namespace radius_of_O2016_l1848_184865

-- Define the centers and radii of circles
variable (a : ℝ) (n : ℕ) (r : ℕ → ℝ)

-- Conditions
-- Radius of the first circle
def initial_radius := r 1 = 1 / (2 * a)
-- Sequence of the radius difference based on solution step
def radius_recursive := ∀ n > 1, r (n + 1) - r n = 1 / a

-- The final statement to be proven
theorem radius_of_O2016 (h1 : initial_radius a r) (h2 : radius_recursive a r) :
  r 2016 = 4031 / (2 * a) := 
by sorry

end radius_of_O2016_l1848_184865


namespace multiple_of_pumpkins_l1848_184863

theorem multiple_of_pumpkins (M S : ℕ) (hM : M = 14) (hS : S = 54) (h : S = x * M + 12) : x = 3 := sorry

end multiple_of_pumpkins_l1848_184863


namespace quotient_change_l1848_184841

variables {a b : ℝ} (h : a / b = 0.78)

theorem quotient_change (a b : ℝ) (h : a / b = 0.78) : (10 * a) / (b / 10) = 78 :=
by
  sorry

end quotient_change_l1848_184841


namespace inversely_proportional_percentage_change_l1848_184828

variable {x y k : ℝ}
variable (a b : ℝ)

/-- Given that x and y are positive numbers and inversely proportional,
if x increases by a% and y decreases by b%, then b = 100a / (100 + a) -/
theorem inversely_proportional_percentage_change
  (hx : 0 < x) (hy : 0 < y) (hinv : y = k / x)
  (ha : 0 < a) (hb : 0 < b)
  (hchange : ((1 + a / 100) * x) * ((1 - b / 100) * y) = k) :
  b = 100 * a / (100 + a) :=
sorry

end inversely_proportional_percentage_change_l1848_184828


namespace stationery_shop_costs_l1848_184853

theorem stationery_shop_costs (p n : ℝ) 
  (h1 : 9 * p + 6 * n = 3.21)
  (h2 : 8 * p + 5 * n = 2.84) :
  12 * p + 9 * n = 4.32 :=
sorry

end stationery_shop_costs_l1848_184853


namespace selling_price_correct_l1848_184861

-- Define the parameters
def stamp_duty_rate : ℝ := 0.002
def commission_rate : ℝ := 0.0035
def bought_shares : ℝ := 3000
def buying_price_per_share : ℝ := 12
def profit : ℝ := 5967

-- Define the selling price per share
noncomputable def selling_price_per_share (x : ℝ) : ℝ :=
  bought_shares * x - bought_shares * buying_price_per_share -
  bought_shares * x * (stamp_duty_rate + commission_rate) - 
  bought_shares * buying_price_per_share * (stamp_duty_rate + commission_rate)

-- The target selling price per share
def target_selling_price_per_share : ℝ := 14.14

-- Statement of the problem
theorem selling_price_correct (x : ℝ) : selling_price_per_share x = profit → x = target_selling_price_per_share := by
  sorry

end selling_price_correct_l1848_184861


namespace last_colored_cell_is_51_50_l1848_184813

def last_spiral_cell (width height : ℕ) : ℕ × ℕ :=
  -- Assuming an external or pre-defined process to calculate the last cell for a spiral pattern
  sorry 

theorem last_colored_cell_is_51_50 :
  last_spiral_cell 200 100 = (51, 50) :=
sorry

end last_colored_cell_is_51_50_l1848_184813


namespace relation_between_y_l1848_184827

/-- Definition of the points on the parabola y = -(x-3)^2 - 4 --/
def pointA (y₁ : ℝ) : Prop := y₁ = -(1/4 - 3)^2 - 4
def pointB (y₂ : ℝ) : Prop := y₂ = -(1 - 3)^2 - 4
def pointC (y₃ : ℝ) : Prop := y₃ = -(4 - 3)^2 - 4 

/-- Relationship between y₁, y₂, y₃ for given points on the quadratic function --/
theorem relation_between_y (y₁ y₂ y₃ : ℝ) 
  (hA : pointA y₁)
  (hB : pointB y₂)
  (hC : pointC y₃) : 
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end relation_between_y_l1848_184827


namespace positive_integer_satisfies_condition_l1848_184884

def num_satisfying_pos_integers : ℕ :=
  1

theorem positive_integer_satisfies_condition :
  ∃ (n : ℕ), 16 - 4 * n > 10 ∧ n = num_satisfying_pos_integers := by
  sorry

end positive_integer_satisfies_condition_l1848_184884


namespace problem_solution_l1848_184871

theorem problem_solution
  (N1 N2 : ℤ)
  (h : ∀ x : ℝ, 50 * x - 42 ≠ 0 → x ≠ 2 → x ≠ 3 → 
    (50 * x - 42) / (x ^ 2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) : 
  N1 * N2 = -6264 :=
sorry

end problem_solution_l1848_184871


namespace yellow_tickets_needed_l1848_184800

def yellow_from_red (r : ℕ) : ℕ := r / 10
def red_from_blue (b : ℕ) : ℕ := b / 10
def blue_needed (current_blue : ℕ) (additional_blue : ℕ) : ℕ := current_blue + additional_blue
def total_blue_from_tickets (y : ℕ) (r : ℕ) (b : ℕ) : ℕ := (y * 10 * 10) + (r * 10) + b

theorem yellow_tickets_needed (y r b additional_blue : ℕ) (h : total_blue_from_tickets y r b + additional_blue = 1000) :
  yellow_from_red (red_from_blue (total_blue_from_tickets y r b + additional_blue)) = 10 := 
by
  sorry

end yellow_tickets_needed_l1848_184800


namespace lab_tech_items_l1848_184820

theorem lab_tech_items (num_uniforms : ℕ) (num_coats : ℕ) (num_techs : ℕ) (total_items : ℕ)
  (h_uniforms : num_uniforms = 12)
  (h_coats : num_coats = 6 * num_uniforms)
  (h_techs : num_techs = num_uniforms / 2)
  (h_total : total_items = num_coats + num_uniforms) :
  total_items / num_techs = 14 :=
by
  -- Placeholder for proof, ensuring theorem builds correctly.
  sorry

end lab_tech_items_l1848_184820


namespace contrapositive_of_square_sum_zero_l1848_184826

theorem contrapositive_of_square_sum_zero (a b : ℝ) :
  (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  sorry

end contrapositive_of_square_sum_zero_l1848_184826


namespace total_income_l1848_184840

def ron_ticket_price : ℝ := 2.00
def kathy_ticket_price : ℝ := 4.50
def total_tickets : ℕ := 20
def ron_tickets_sold : ℕ := 12

theorem total_income : ron_tickets_sold * ron_ticket_price + (total_tickets - ron_tickets_sold) * kathy_ticket_price = 60.00 := by
  sorry

end total_income_l1848_184840


namespace geometric_series_sum_l1848_184851

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  sum_geometric_series (1/4) (1/4) 7 = 4/3 :=
by
  -- Proof is omitted
  sorry

end geometric_series_sum_l1848_184851


namespace problem1_l1848_184873

theorem problem1 : 2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15 / 2 := 
by
  sorry

end problem1_l1848_184873


namespace expand_expression_l1848_184877

variable {R : Type*} [CommRing R]
variable (x y : R)

theorem expand_expression : 
  ((10 * x - 6 * y + 9) * 3 * y) = (30 * x * y - 18 * y * y + 27 * y) :=
by
  sorry

end expand_expression_l1848_184877


namespace barry_should_pay_l1848_184899

def original_price : ℝ := 80
def discount_rate : ℝ := 0.15

theorem barry_should_pay:
  original_price * (1 - discount_rate) = 68 := 
by 
  -- Original price: 80
  -- Discount rate: 0.15
  -- Question: Final price after discount
  sorry

end barry_should_pay_l1848_184899


namespace quadratic_root_ratio_l1848_184838

theorem quadratic_root_ratio {m p q : ℝ} (h₁ : m ≠ 0) (h₂ : p ≠ 0) (h₃ : q ≠ 0)
  (h₄ : ∀ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) →
    (∃ t₁ t₂ : ℝ, t₁ = 3 * s₁ ∧ t₂ = 3 * s₂ ∧ (t₁ + t₂ = -m ∧ t₁ * t₂ = p))) :
  p / q = 27 :=
by
  sorry

end quadratic_root_ratio_l1848_184838


namespace problems_per_page_is_eight_l1848_184833

noncomputable def totalProblems := 60
noncomputable def finishedProblems := 20
noncomputable def totalPages := 5
noncomputable def problemsLeft := totalProblems - finishedProblems
noncomputable def problemsPerPage := problemsLeft / totalPages

theorem problems_per_page_is_eight :
  problemsPerPage = 8 :=
by
  sorry

end problems_per_page_is_eight_l1848_184833


namespace intersection_of_sets_l1848_184887

def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {1, 3, 5}

theorem intersection_of_sets : A ∩ B = {1, 5} :=
by
  sorry

end intersection_of_sets_l1848_184887


namespace sum_of_squares_l1848_184814

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : ab + bc + ca = 5) : a^2 + b^2 + c^2 = 390 :=
by sorry

end sum_of_squares_l1848_184814


namespace find_a_l1848_184847

theorem find_a (a : ℝ) (h1 : ∀ (x y : ℝ), ax + 2*y - 2 = 0 → (x + y) = 0)
  (h2 : ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 6 → (∃ A B : ℝ × ℝ, A ≠ B ∧ (A = (x, y) ∧ B = (-x, -y))))
  : a = -2 := 
sorry

end find_a_l1848_184847


namespace peytons_children_l1848_184835

theorem peytons_children (C : ℕ) (juice_per_week : ℕ) (weeks_in_school_year : ℕ) (total_juice_boxes : ℕ) 
  (h1 : juice_per_week = 5) 
  (h2 : weeks_in_school_year = 25) 
  (h3 : total_juice_boxes = 375)
  (h4 : C * (juice_per_week * weeks_in_school_year) = total_juice_boxes) 
  : C = 3 :=
sorry

end peytons_children_l1848_184835


namespace martha_found_blocks_l1848_184879

variable (initial_blocks final_blocks found_blocks : ℕ)

theorem martha_found_blocks 
    (h_initial : initial_blocks = 4) 
    (h_final : final_blocks = 84) 
    (h_found : found_blocks = final_blocks - initial_blocks) : 
    found_blocks = 80 := by
  sorry

end martha_found_blocks_l1848_184879


namespace negation_of_p_l1848_184876

variable (f : ℝ → ℝ)

theorem negation_of_p :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔ (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
by
  sorry

end negation_of_p_l1848_184876


namespace trigonometric_identity_l1848_184817

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 2 + α) = - 1 / 3 := 
by
  sorry

end trigonometric_identity_l1848_184817


namespace white_pairs_coincide_l1848_184829

def num_red : Nat := 4
def num_blue : Nat := 4
def num_green : Nat := 2
def num_white : Nat := 6
def red_pairs : Nat := 3
def blue_pairs : Nat := 2
def green_pairs : Nat := 1 
def red_white_pairs : Nat := 2
def green_blue_pairs : Nat := 1

theorem white_pairs_coincide :
  (num_red = 4) ∧ 
  (num_blue = 4) ∧ 
  (num_green = 2) ∧ 
  (num_white = 6) ∧ 
  (red_pairs = 3) ∧ 
  (blue_pairs = 2) ∧ 
  (green_pairs = 1) ∧ 
  (red_white_pairs = 2) ∧ 
  (green_blue_pairs = 1) → 
  4 = 4 :=
by
  sorry

end white_pairs_coincide_l1848_184829


namespace larger_number_is_30_l1848_184843

-- Formalizing the conditions
variables (x y : ℝ)

-- Define the conditions given in the problem
def sum_condition : Prop := x + y = 40
def ratio_condition : Prop := x / y = 3

-- Formalize the problem statement
theorem larger_number_is_30 (h1 : sum_condition x y) (h2 : ratio_condition x y) : x = 30 :=
sorry

end larger_number_is_30_l1848_184843


namespace bobby_books_count_l1848_184858

variable (KristiBooks BobbyBooks : ℕ)

theorem bobby_books_count (h1 : KristiBooks = 78) (h2 : BobbyBooks = KristiBooks + 64) : BobbyBooks = 142 :=
by
  sorry

end bobby_books_count_l1848_184858


namespace range_of_real_number_l1848_184894

theorem range_of_real_number (a : ℝ) : (a > 0) ∧ (a - 1 > 0) → a > 1 :=
by
  sorry

end range_of_real_number_l1848_184894


namespace petya_vasya_cubic_roots_diff_2014_l1848_184815

theorem petya_vasya_cubic_roots_diff_2014 :
  ∀ (p q r : ℚ), ∃ (x1 x2 x3 : ℚ), x1 ≠ 0 ∧ (x1 - x2 = 2014 ∨ x1 - x3 = 2014 ∨ x2 - x3 = 2014) :=
sorry

end petya_vasya_cubic_roots_diff_2014_l1848_184815


namespace isosceles_right_triangle_inscribed_circle_l1848_184867

theorem isosceles_right_triangle_inscribed_circle
  (h r x : ℝ)
  (h_def : h = 2 * r)
  (r_def : r = Real.sqrt 2 / 4)
  (x_def : x = h - r) :
  x = Real.sqrt 2 / 4 :=
by
  sorry

end isosceles_right_triangle_inscribed_circle_l1848_184867


namespace complement_union_eq_l1848_184872

universe u

-- Definitions based on conditions in a)
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

-- The goal to prove based on c)
theorem complement_union_eq :
  (U \ (M ∪ N)) = {5, 6} := 
by sorry

end complement_union_eq_l1848_184872


namespace greatest_unexpressible_sum_l1848_184845

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem greatest_unexpressible_sum : 
  ∀ (n : ℕ), (∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n) → n ≤ 11 :=
by
  sorry

end greatest_unexpressible_sum_l1848_184845


namespace triangle_solution_l1848_184803

noncomputable def solve_triangle (a : ℝ) (α : ℝ) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let s := 75
  let b := 41
  let c := 58
  let β := 43 + 36 / 60 + 10 / 3600
  let γ := 77 + 19 / 60 + 11 / 3600
  ((b, c), (β, γ))

theorem triangle_solution :
  solve_triangle 51 (59 + 4 / 60 + 39 / 3600) 1020 = ((41, 58), (43 + 36 / 60 + 10 / 3600, 77 + 19 / 60 + 11 / 3600)) :=
sorry  

end triangle_solution_l1848_184803


namespace min_value_frac_l1848_184830

theorem min_value_frac (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ (∀ y, 0 < y ∧ y < 1 → (a * a / y + b * b / (1 - y)) ≥ (a + b) * (a + b)) ∧ 
       a * a / x + b * b / (1 - x) = (a + b) * (a + b) := 
by {
  sorry
}

end min_value_frac_l1848_184830


namespace no_real_x_condition_l1848_184850

theorem no_real_x_condition (x : ℝ) : 
(∃ a b : ℕ, 4 * x^5 - 7 = a^2 ∧ 4 * x^13 - 7 = b^2) → false := 
by {
  sorry
}

end no_real_x_condition_l1848_184850


namespace proof_problem_l1848_184886

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem_l1848_184886


namespace trig_identity_l1848_184842

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : ℝ) := Real.tan (x * Real.pi / 180)

theorem trig_identity :
  (2 * sin_deg 50 + sin_deg 10 * (1 + Real.sqrt 3 * tan_deg 10) * Real.sqrt 2 * (sin_deg 80)^2) = Real.sqrt 6 :=
by
  sorry

end trig_identity_l1848_184842


namespace train_length_is_correct_l1848_184870

noncomputable def length_of_train (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_m_s := train_speed * (1000 / 3600)
  let total_distance := speed_m_s * time_to_cross
  total_distance - bridge_length

theorem train_length_is_correct :
  length_of_train 36 24.198064154867613 132 = 109.98064154867613 :=
by
  sorry

end train_length_is_correct_l1848_184870


namespace solve_for_x_l1848_184839

-- Define the custom operation for real numbers
def custom_op (a b c d : ℝ) : ℝ := a * c - b * d

-- The theorem to prove
theorem solve_for_x (x : ℝ) (h : custom_op (-x) 3 (x - 2) (-6) = 10) :
  x = 4 ∨ x = -2 :=
sorry

end solve_for_x_l1848_184839


namespace sum_of_transformed_numbers_l1848_184883

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 :=
by
  sorry

end sum_of_transformed_numbers_l1848_184883


namespace compound_interest_semiannual_l1848_184866

theorem compound_interest_semiannual
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (initial_amount : P = 900)
  (interest_rate : r = 0.10)
  (compounding_periods : n = 2)
  (time_period : t = 1) :
  P * (1 + r / n) ^ (n * t) = 992.25 :=
by
  sorry

end compound_interest_semiannual_l1848_184866


namespace sugar_water_inequality_triangle_inequality_l1848_184859

-- Condition for question (1)
variable (x y m : ℝ)
variable (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0)

-- Proof problem for question (1)
theorem sugar_water_inequality : y / x < (y + m) / (x + m) :=
sorry

-- Condition for question (2)
variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (hab : b + c > a) (hac : a + c > b) (hbc : a + b > c)

-- Proof problem for question (2)
theorem triangle_inequality : 
  a / (b + c) + b / (a + c) + c / (a + b) < 2 :=
sorry

end sugar_water_inequality_triangle_inequality_l1848_184859


namespace tangent_parabola_line_l1848_184818

theorem tangent_parabola_line (a : ℝ) :
  (∃ x : ℝ, ax^2 + 1 = x ∧ ∀ y : ℝ, (y = ax^2 + 1 → y = x)) ↔ a = 1/4 :=
by
  sorry

end tangent_parabola_line_l1848_184818


namespace radius_increase_50_percent_l1848_184802

theorem radius_increase_50_percent 
  (r : ℝ)
  (h1 : 1.5 * r = r + r * 0.5) : 
  (3 * Real.pi * r = 2 * Real.pi * r + (2 * Real.pi * r * 0.5)) ∧
  (2.25 * Real.pi * r^2 = Real.pi * r^2 + (Real.pi * r^2 * 1.25)) := 
sorry

end radius_increase_50_percent_l1848_184802


namespace number_of_zeros_of_f_l1848_184878

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin (2 * x)

theorem number_of_zeros_of_f : (∃ l : List ℝ, (∀ x ∈ l, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧ l.length = 4) := 
by
  sorry

end number_of_zeros_of_f_l1848_184878


namespace unique_function_l1848_184895

noncomputable def find_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a + b > 2019 → a + f b ∣ a^2 + b * f a

theorem unique_function (r : ℕ) (f : ℕ → ℕ) :
  find_function f → (∀ x : ℕ, f x = r * x) :=
sorry

end unique_function_l1848_184895


namespace list_price_is_40_l1848_184891

theorem list_price_is_40 (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to indicate we're skipping the proof.
  sorry

end list_price_is_40_l1848_184891


namespace arithmetic_mean_odd_primes_lt_30_l1848_184868

theorem arithmetic_mean_odd_primes_lt_30 : 
  (3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29) / 9 = 14 :=
by
  sorry

end arithmetic_mean_odd_primes_lt_30_l1848_184868


namespace isosceles_right_triangle_area_l1848_184812

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l1848_184812


namespace alcohol_percentage_solution_x_l1848_184805

theorem alcohol_percentage_solution_x :
  ∃ (P : ℝ), 
  (∀ (vol_x vol_y : ℝ), vol_x = 50 → vol_y = 150 →
    ∀ (percent_y percent_new : ℝ), percent_y = 30 → percent_new = 25 →
      ((P / 100) * vol_x + (percent_y / 100) * vol_y) / (vol_x + vol_y) = percent_new) → P = 10 :=
by
  -- Given conditions
  let vol_x := 50
  let vol_y := 150
  let percent_y := 30
  let percent_new := 25

  -- The proof body should be here
  sorry

end alcohol_percentage_solution_x_l1848_184805


namespace joe_saves_6000_l1848_184801

-- Definitions based on the conditions
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

-- Total expenses
def total_expenses : ℕ := flight_cost + hotel_cost + food_cost

-- Total savings
def total_savings : ℕ := total_expenses + money_left

-- The proof statement
theorem joe_saves_6000 : total_savings = 6000 := by
  -- Proof goes here
  sorry

end joe_saves_6000_l1848_184801


namespace distribution_of_K_l1848_184896

theorem distribution_of_K (x y z : ℕ) 
  (h_total : x + y + z = 370)
  (h_diff : y + z - x = 50)
  (h_prop : x * z = y^2) :
  x = 160 ∧ y = 120 ∧ z = 90 := by
  sorry

end distribution_of_K_l1848_184896


namespace find_prime_pairs_l1848_184880

def is_solution_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1)

theorem find_prime_pairs :
  {pq : ℕ × ℕ | is_solution_pair pq.1 pq.2} =
  { (2, 13), (13, 2), (3, 7), (7, 3) } :=
by
  sorry

end find_prime_pairs_l1848_184880


namespace population_net_increase_l1848_184844

theorem population_net_increase
  (birth_rate : ℕ) (death_rate : ℕ) (T : ℕ)
  (h1 : birth_rate = 7) (h2 : death_rate = 3) (h3 : T = 86400) :
  (birth_rate - death_rate) * (T / 2) = 172800 :=
by
  sorry

end population_net_increase_l1848_184844


namespace roots_squared_sum_l1848_184837

theorem roots_squared_sum {x y : ℝ} (hx : 3 * x^2 - 7 * x + 5 = 0) (hy : 3 * y^2 - 7 * y + 5 = 0) (hxy : x ≠ y) :
  x^2 + y^2 = 19 / 9 :=
sorry

end roots_squared_sum_l1848_184837


namespace Alex_sandwich_count_l1848_184889

theorem Alex_sandwich_count :
  let meats := 10
  let cheeses := 9
  let sandwiches := meats * (cheeses.choose 2)
  sandwiches = 360 :=
by
  -- Here start your proof
  sorry

end Alex_sandwich_count_l1848_184889


namespace terry_age_proof_l1848_184893

theorem terry_age_proof
  (nora_age : ℕ)
  (h1 : nora_age = 10)
  (terry_age_in_10_years : ℕ)
  (h2 : terry_age_in_10_years = 4 * nora_age)
  (nora_age_in_5_years : ℕ)
  (h3 : nora_age_in_5_years = nora_age + 5)
  (sam_age_in_5_years : ℕ)
  (h4 : sam_age_in_5_years = 2 * nora_age_in_5_years)
  (sam_current_age : ℕ)
  (h5 : sam_current_age = sam_age_in_5_years - 5)
  (terry_current_age : ℕ)
  (h6 : sam_current_age = terry_current_age + 6) :
  terry_current_age = 19 :=
by
  sorry

end terry_age_proof_l1848_184893


namespace relationship_among_abc_l1848_184846

noncomputable def a : ℝ := 36^(1/5)
noncomputable def b : ℝ := 3^(4/3)
noncomputable def c : ℝ := 9^(2/5)

theorem relationship_among_abc (a_def : a = 36^(1/5)) 
                              (b_def : b = 3^(4/3)) 
                              (c_def : c = 9^(2/5)) : a < c ∧ c < b :=
by
  rw [a_def, b_def, c_def]
  sorry

end relationship_among_abc_l1848_184846


namespace value_of_m_making_365m_divisible_by_12_l1848_184804

theorem value_of_m_making_365m_divisible_by_12
  (m : ℕ)
  (h1 : (3650 + m) % 3 = 0)
  (h2 : (50 + m) % 4 = 0) :
  m = 0 :=
sorry

end value_of_m_making_365m_divisible_by_12_l1848_184804


namespace min_value_of_f_l1848_184897

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 1425

theorem min_value_of_f : ∃ (x : ℝ), f x = 1397 :=
by
  sorry

end min_value_of_f_l1848_184897


namespace c_alone_finishes_in_60_days_l1848_184860

-- Definitions for rates of work
variables (A B C : ℝ)

-- The conditions given in the problem
-- A and B together can finish the job in 15 days
def condition1 : Prop := A + B = 1 / 15
-- A, B, and C together can finish the job in 12 days
def condition2 : Prop := A + B + C = 1 / 12

-- The statement to prove: C alone can finish the job in 60 days
theorem c_alone_finishes_in_60_days 
  (h1 : condition1 A B) 
  (h2 : condition2 A B C) : 
  (1 / C) = 60 :=
by
  sorry

end c_alone_finishes_in_60_days_l1848_184860


namespace find_minimum_value_2a_plus_b_l1848_184849

theorem find_minimum_value_2a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_re_z : (3 * a * b + 2) = 4) : 2 * a + b = (4 * Real.sqrt 3) / 3 :=
sorry

end find_minimum_value_2a_plus_b_l1848_184849


namespace train_passes_platform_in_39_2_seconds_l1848_184864

def length_of_train : ℝ := 360
def speed_in_kmh : ℝ := 45
def length_of_platform : ℝ := 130

noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_in_mps

theorem train_passes_platform_in_39_2_seconds :
  time_to_pass_platform = 39.2 := by
  sorry

end train_passes_platform_in_39_2_seconds_l1848_184864


namespace product_consecutive_two_digits_l1848_184862

theorem product_consecutive_two_digits (a b c : ℕ) : 
  ¬(∃ n : ℕ, (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2)) :=
by
  sorry

end product_consecutive_two_digits_l1848_184862


namespace probability_of_point_within_two_units_l1848_184825

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let area_of_circle := 4 * Real.pi
  let area_of_square := 36
  area_of_circle / area_of_square

theorem probability_of_point_within_two_units :
  probability_within_two_units_of_origin = Real.pi / 9 := 
by
  -- The proof steps are omitted as per the requirements
  sorry

end probability_of_point_within_two_units_l1848_184825


namespace number_of_prime_factors_30_factorial_l1848_184831

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l1848_184831


namespace quadratic_roots_identity_l1848_184822

theorem quadratic_roots_identity (α β : ℝ) (hαβ : α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0) : 
  α^2 + α*β - 3*α = 0 := 
by 
  sorry

end quadratic_roots_identity_l1848_184822


namespace length_of_train_l1848_184869

-- Define the conditions
def bridge_length : ℕ := 200
def train_crossing_time : ℕ := 60
def train_speed : ℕ := 5

-- Define the total distance traveled by the train while crossing the bridge
def total_distance : ℕ := train_speed * train_crossing_time

-- The problem is to show the length of the train
theorem length_of_train :
  total_distance - bridge_length = 100 :=
by sorry

end length_of_train_l1848_184869


namespace hyperbola_asymptotes_l1848_184834

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the equations for the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- State the theorem
theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola_eq x y → (asymptote_pos x y ∨ asymptote_neg x y) := 
by
  sorry

end hyperbola_asymptotes_l1848_184834


namespace decagon_adjacent_probability_l1848_184832

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l1848_184832


namespace ship_speeds_l1848_184823

theorem ship_speeds (x : ℝ) 
  (h1 : (2 * x) ^ 2 + (2 * (x + 3)) ^ 2 = 174 ^ 2) :
  x = 60 ∧ x + 3 = 63 :=
by
  sorry

end ship_speeds_l1848_184823


namespace total_notes_l1848_184898

theorem total_notes (total_money : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) (fivehundred_value : ℕ) (fivehundred_notes : ℕ) :
  total_money = 10350 →
  fifty_notes = 57 →
  fifty_value = 50 →
  fivehundred_value = 500 →
  57 * 50 + fivehundred_notes * 500 = 10350 →
  fifty_notes + fivehundred_notes = 72 :=
by
  intros h_total_money h_fifty_notes h_fifty_value h_fivehundred_value h_equation
  sorry

end total_notes_l1848_184898


namespace mnpq_product_l1848_184811

noncomputable def prove_mnpq_product (a b x y : ℝ) : Prop :=
  ∃ (m n p q : ℤ), (a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4 ∧
                    m * n * p * q = 4

theorem mnpq_product (a b x y : ℝ) (h : a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) :
  prove_mnpq_product a b x y :=
sorry

end mnpq_product_l1848_184811


namespace trip_to_museum_l1848_184810

theorem trip_to_museum (x y z w : ℕ) 
  (h2 : y = 2 * x) 
  (h3 : z = 2 * x - 6) 
  (h4 : w = x + 9) 
  (htotal : x + y + z + w = 75) : 
  x = 12 := 
by 
  sorry

end trip_to_museum_l1848_184810


namespace find_other_parallel_side_l1848_184824

variable (a b h : ℝ) (Area : ℝ)

-- Conditions
axiom h_pos : h = 13
axiom a_val : a = 18
axiom area_val : Area = 247
axiom area_formula : Area = (1 / 2) * (a + b) * h

-- Theorem (to be proved by someone else)
theorem find_other_parallel_side (a b h : ℝ) 
  (h_pos : h = 13) 
  (a_val : a = 18) 
  (area_val : Area = 247) 
  (area_formula : Area = (1 / 2) * (a + b) * h) : 
  b = 20 :=
by
  sorry

end find_other_parallel_side_l1848_184824


namespace trigonometric_expression_simplification_l1848_184857

theorem trigonometric_expression_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := 
sorry

end trigonometric_expression_simplification_l1848_184857


namespace equation_solution_l1848_184882

theorem equation_solution (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end equation_solution_l1848_184882


namespace find_A_l1848_184816

-- Define the polynomial and the partial fraction decomposition equation
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_A (A B C : ℝ) (h : ∀ x : ℝ, 1 / polynomial x = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) : 
  A = 1 / 16 :=
sorry

end find_A_l1848_184816


namespace largest_consecutive_sum_to_35_l1848_184888

theorem largest_consecutive_sum_to_35 (n : ℕ) (h : ∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 35) : n ≤ 7 :=
by
  sorry

end largest_consecutive_sum_to_35_l1848_184888


namespace max_value_expr_l1848_184855

theorem max_value_expr (a b c d : ℝ) (ha : -12.5 ≤ a ∧ a ≤ 12.5) (hb : -12.5 ≤ b ∧ b ≤ 12.5) (hc : -12.5 ≤ c ∧ c ≤ 12.5) (hd : -12.5 ≤ d ∧ d ≤ 12.5) :
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 650 :=
sorry

end max_value_expr_l1848_184855


namespace pow_neg_cubed_squared_l1848_184881

variable (a : ℝ)

theorem pow_neg_cubed_squared : 
  (-a^3)^2 = a^6 := 
by 
  sorry

end pow_neg_cubed_squared_l1848_184881


namespace find_f_prime_one_l1848_184852

noncomputable def f (f'_1 : ℝ) (x : ℝ) := f'_1 * x^3 - 2 * x^2 + 3

theorem find_f_prime_one (f'_1 : ℝ) 
  (h_derivative : ∀ x : ℝ, deriv (f f'_1) x = 3 * f'_1 * x^2 - 4 * x)
  (h_value_at_1 : deriv (f f'_1) 1 = f'_1) :
  f'_1 = 2 :=
by 
  sorry

end find_f_prime_one_l1848_184852


namespace average_speed_of_train_l1848_184892

theorem average_speed_of_train (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 125) (h2 : d2 = 270) (h3 : t1 = 2.5) (h4 : t2 = 3) :
  (d1 + d2) / (t1 + t2) = 71.82 :=
by
  sorry

end average_speed_of_train_l1848_184892


namespace sheep_ratio_l1848_184808

theorem sheep_ratio (s : ℕ) (h1 : s = 400) (h2 : s / 4 + 150 = s - s / 4) : (s / 4 * 3 - 150) / 150 = 1 :=
by {
  sorry
}

end sheep_ratio_l1848_184808


namespace cost_of_3000_pencils_l1848_184821

theorem cost_of_3000_pencils (pencils_per_box : ℕ) (cost_per_box : ℝ) (pencils_needed : ℕ) (unit_cost : ℝ): 
  pencils_per_box = 120 → cost_per_box = 36 → pencils_needed = 3000 → unit_cost = 0.30 →
  (pencils_needed * unit_cost = (3000 : ℝ) * 0.30) :=
by
  intros _ _ _ _
  sorry

end cost_of_3000_pencils_l1848_184821


namespace partA_l1848_184885

theorem partA (a b : ℝ) : (a - b) ^ 2 ≥ 0 → (a^2 + b^2) / 2 ≥ a * b := 
by
  intro h
  sorry

end partA_l1848_184885


namespace timesToFillBottlePerWeek_l1848_184836

noncomputable def waterConsumptionPerDay : ℕ := 4 * 5
noncomputable def waterConsumptionPerWeek : ℕ := 7 * waterConsumptionPerDay
noncomputable def bottleCapacity : ℕ := 35

theorem timesToFillBottlePerWeek : 
  waterConsumptionPerWeek / bottleCapacity = 4 := 
by
  sorry

end timesToFillBottlePerWeek_l1848_184836
