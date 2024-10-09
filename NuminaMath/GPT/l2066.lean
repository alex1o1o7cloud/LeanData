import Mathlib

namespace cone_base_radius_and_slant_height_l2066_206634

noncomputable def sector_angle := 300
noncomputable def sector_radius := 10
noncomputable def arc_length := (sector_angle / 360) * 2 * Real.pi * sector_radius

theorem cone_base_radius_and_slant_height :
  ∃ (r l : ℝ), arc_length = 2 * Real.pi * r ∧ l = sector_radius ∧ r = 8 ∧ l = 10 :=
by 
  sorry

end cone_base_radius_and_slant_height_l2066_206634


namespace find_pq_l2066_206691

noncomputable def area_of_triangle (p q : ℝ) : ℝ := 1/2 * (12 / p) * (12 / q)

theorem find_pq (p q : ℝ) (hp : p > 0) (hq : q > 0) (harea : area_of_triangle p q = 12) : p * q = 6 := 
by
  sorry

end find_pq_l2066_206691


namespace alice_burger_spending_l2066_206630

theorem alice_burger_spending :
  let daily_burgers := 4
  let burger_cost := 13
  let days_in_june := 30
  let mondays_wednesdays := 8
  let fridays := 4
  let fifth_purchase_coupons := 6
  let discount_10_percent := 0.9
  let discount_50_percent := 0.5
  let full_price := days_in_june * daily_burgers * burger_cost
  let discount_10 := mondays_wednesdays * daily_burgers * burger_cost * discount_10_percent
  let fridays_cost := (daily_burgers - 1) * fridays * burger_cost
  let discount_50 := fifth_purchase_coupons * burger_cost * discount_50_percent
  full_price - discount_10 - fridays_cost - discount_50 + fridays_cost = 1146.6 := by sorry

end alice_burger_spending_l2066_206630


namespace cost_per_revision_l2066_206624

theorem cost_per_revision
  (x : ℝ)
  (initial_cost : ℝ)
  (revised_once : ℝ)
  (revised_twice : ℝ)
  (total_pages : ℝ)
  (total_cost : ℝ)
  (cost_per_page_first_time : ℝ) :
  initial_cost = cost_per_page_first_time * total_pages →
  revised_once * x + revised_twice * (2 * x) + initial_cost = total_cost →
  revised_once + revised_twice + (total_pages - (revised_once + revised_twice)) = total_pages →
  total_pages = 200 →
  initial_cost = 1000 →
  cost_per_page_first_time = 5 →
  revised_once = 80 →
  revised_twice = 20 →
  total_cost = 1360 →
  x = 3 :=
by
  intros h_initial h_total_cost h_tot_pages h_tot_pages_200 h_initial_1000 h_cost_5 h_revised_once h_revised_twice h_given_cost
  -- Proof steps to be filled
  sorry

end cost_per_revision_l2066_206624


namespace points_lie_on_line_l2066_206675

noncomputable def x (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 + 2 * t + 2) / t
noncomputable def y (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 - 2 * t + 2) / t

theorem points_lie_on_line : ∀ (t : ℝ) (ht : t ≠ 0), y t ht = x t ht - 4 :=
by 
  intros t ht
  simp [x, y]
  sorry

end points_lie_on_line_l2066_206675


namespace find_c_l2066_206603

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end find_c_l2066_206603


namespace smallest_circle_radius_l2066_206629

-- Define the problem as a proposition
theorem smallest_circle_radius (r : ℝ) (R1 R2 : ℝ) (hR1 : R1 = 6) (hR2 : R2 = 4) (h_right_triangle : (r + R2)^2 + (r + R1)^2 = (R2 + R1)^2) : r = 2 := 
sorry

end smallest_circle_radius_l2066_206629


namespace find_value_l2066_206687

noncomputable def S2013 (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) : ℂ :=
  (x / (x + y))^2013 + (y / (x + y))^2013

theorem find_value (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) :
  S2013 x y h h_eq = -2 :=
sorry

end find_value_l2066_206687


namespace euclidean_remainder_2022_l2066_206619

theorem euclidean_remainder_2022 : 
  (2022 ^ (2022 ^ 2022)) % 11 = 5 := 
by sorry

end euclidean_remainder_2022_l2066_206619


namespace number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l2066_206661

theorem number_of_whole_numbers_between_sqrt_18_and_sqrt_120 : 
  ∀ (n : ℕ), 
  (5 ≤ n ∧ n ≤ 10) ↔ (6 = 6) :=
sorry

end number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l2066_206661


namespace central_cell_value_l2066_206672

theorem central_cell_value (a1 a2 a3 a4 a5 a6 a7 a8 C : ℕ) 
  (h1 : a1 + a3 + C = 13) (h2 : a2 + a4 + C = 13)
  (h3 : a5 + a7 + C = 13) (h4 : a6 + a8 + C = 13)
  (h5 : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 40) : 
  C = 3 := 
sorry

end central_cell_value_l2066_206672


namespace compute_f5_l2066_206662

-- Definitions of the logical operations used in the conditions
axiom x1 : Prop
axiom x2 : Prop
axiom x3 : Prop
axiom x4 : Prop
axiom x5 : Prop

noncomputable def x6 : Prop := x1 ∨ x3
noncomputable def x7 : Prop := x2 ∧ x6
noncomputable def x8 : Prop := x3 ∨ x5
noncomputable def x9 : Prop := x4 ∧ x8
noncomputable def f5 : Prop := x7 ∨ x9

-- Proof statement to be proven
theorem compute_f5 : f5 = (x7 ∨ x9) :=
by sorry

end compute_f5_l2066_206662


namespace solution_to_equation_l2066_206644

theorem solution_to_equation :
  ∃ x : ℝ, x = (11 - 3 * Real.sqrt 5) / 2 ∧ x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 31 :=
by
  sorry

end solution_to_equation_l2066_206644


namespace fred_sheets_left_l2066_206692

def sheets_fred_had_initially : ℕ := 212
def sheets_jane_given : ℕ := 307
def planned_percentage_more : ℕ := 50
def given_percentage : ℕ := 25

-- Prove that after all transactions, Fred has 389 sheets left
theorem fred_sheets_left :
  let planned_sheets := (sheets_jane_given * 100) / (planned_percentage_more + 100)
  let sheets_jane_actual := planned_sheets + (planned_sheets * planned_percentage_more) / 100
  let total_sheets := sheets_fred_had_initially + sheets_jane_actual
  let charles_given := (total_sheets * given_percentage) / 100
  let fred_sheets_final := total_sheets - charles_given
  fred_sheets_final = 389 := 
by
  sorry

end fred_sheets_left_l2066_206692


namespace largest_integer_value_l2066_206607

theorem largest_integer_value (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 2/3 ∧ (x : ℚ) < 10 → x = 3 := 
by
  sorry

end largest_integer_value_l2066_206607


namespace cylinder_height_relationship_l2066_206625

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_equal : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relationship : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l2066_206625


namespace peaches_picked_l2066_206673

variable (o t : ℕ)
variable (p : ℕ)

theorem peaches_picked : (o = 34) → (t = 86) → (t = o + p) → p = 52 :=
by
  intros ho ht htot
  rw [ho, ht] at htot
  sorry

end peaches_picked_l2066_206673


namespace value_of_star_15_25_l2066_206664

noncomputable def star (x y : ℝ) : ℝ := Real.log x / Real.log y

axiom condition1 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star (star (x^2) y) y = star x y
axiom condition2 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star x (star y y) = star (star x y) (star x 1)
axiom condition3 (h : 1 > 0) : star 1 1 = 0

theorem value_of_star_15_25 : star 15 25 = (Real.log 3 / (2 * Real.log 5)) + 1 / 2 := 
by 
  sorry

end value_of_star_15_25_l2066_206664


namespace largest_sum_distinct_factors_l2066_206697

theorem largest_sum_distinct_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (h4 : A * B * C = 2023) : A + B + C = 297 :=
sorry

end largest_sum_distinct_factors_l2066_206697


namespace demand_decrease_l2066_206613

theorem demand_decrease (original_price_increase effective_price_increase demand_decrease : ℝ)
  (h1 : original_price_increase = 0.2)
  (h2 : effective_price_increase = original_price_increase / 2)
  (h3 : new_price = original_price * (1 + effective_price_increase))
  (h4 : 1 / new_price = original_demand)
  : demand_decrease = 0.0909 := sorry

end demand_decrease_l2066_206613


namespace cannot_assemble_highlighted_shape_l2066_206602

-- Define the rhombus shape with its properties
structure Rhombus :=
  (white_triangle gray_triangle : Prop)

-- Define the assembly condition
def can_rotate (shape : Rhombus) : Prop := sorry

-- Define the specific shape highlighted that Petya cannot form
def highlighted_shape : Prop := sorry

-- The statement we need to prove
theorem cannot_assemble_highlighted_shape (shape : Rhombus) 
  (h_rotate : can_rotate shape)
  (h_highlight : highlighted_shape) : false :=
by sorry

end cannot_assemble_highlighted_shape_l2066_206602


namespace money_distribution_l2066_206689

-- Conditions
variable (A B x y : ℝ)
variable (h1 : x + 1/2 * y = 50)
variable (h2 : 2/3 * x + y = 50)

-- Problem statement
theorem money_distribution : x = A → y = B → (x + 1/2 * y = 50 ∧ 2/3 * x + y = 50) :=
by
  intro hx hy
  rw [hx, hy]
  exfalso -- using exfalso to skip proof body
  sorry

end money_distribution_l2066_206689


namespace triangle_no_two_obtuse_angles_l2066_206637

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l2066_206637


namespace negation_of_positive_l2066_206658

def is_positive (x : ℝ) : Prop := x > 0
def is_non_positive (x : ℝ) : Prop := x ≤ 0

theorem negation_of_positive (a b c : ℝ) :
  (¬ (is_positive a ∨ is_positive b ∨ is_positive c)) ↔ (is_non_positive a ∧ is_non_positive b ∧ is_non_positive c) :=
by
  sorry

end negation_of_positive_l2066_206658


namespace largest_x_quadratic_inequality_l2066_206668

theorem largest_x_quadratic_inequality : 
  ∃ (x : ℝ), (x^2 - 10 * x + 24 ≤ 0) ∧ (∀ y, (y^2 - 10 * y + 24 ≤ 0) → y ≤ x) :=
sorry

end largest_x_quadratic_inequality_l2066_206668


namespace find_a_and_b_l2066_206663

theorem find_a_and_b (a b : ℕ) :
  42 = a * 6 ∧ 72 = 6 * b ∧ 504 = 42 * 12 → (a, b) = (7, 12) :=
by
  sorry

end find_a_and_b_l2066_206663


namespace complex_fraction_identity_l2066_206671

theorem complex_fraction_identity
  (a b : ℂ) (ζ : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ζ ^ 3 = 1) (h4 : ζ ≠ 1) 
  (h5 : a ^ 2 + a * b + b ^ 2 = 0) :
  (a ^ 9 + b ^ 9) / ((a - b) ^ 9) = (2 : ℂ) / (81 * (ζ - 1)) :=
sorry

end complex_fraction_identity_l2066_206671


namespace china_junior_1990_problem_l2066_206681

theorem china_junior_1990_problem 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1)
  (h1 : a * x = y * z / (y + z))
  (h2 : b * y = x * z / (x + z))
  (h3 : c * z = x * y / (x + y)) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1) :=
sorry

end china_junior_1990_problem_l2066_206681


namespace abs_sum_neq_3_nor_1_l2066_206633

theorem abs_sum_neq_3_nor_1 (a b : ℤ) (h₁ : |a| = 3) (h₂ : |b| = 1) : (|a + b| ≠ 3) ∧ (|a + b| ≠ 1) := sorry

end abs_sum_neq_3_nor_1_l2066_206633


namespace solutions_to_shifted_parabola_l2066_206600

noncomputable def solution_equation := ∀ (a b : ℝ) (m : ℝ) (x : ℝ),
  (a ≠ 0) →
  ((a * (x + m) ^ 2 + b = 0) → (x = 2 ∨ x = -1)) →
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0))

-- We'll leave the proof for this theorem as 'sorry'
theorem solutions_to_shifted_parabola (a b m : ℝ) (h : a ≠ 0)
  (h1 : ∀ (x : ℝ), a * (x + m) ^ 2 + b = 0 → (x = 2 ∨ x = -1)) 
  (x : ℝ) : 
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0)) := sorry

end solutions_to_shifted_parabola_l2066_206600


namespace gcd_factorial_8_10_l2066_206601

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l2066_206601


namespace exist_nat_nums_l2066_206651

theorem exist_nat_nums :
  ∃ (a b c d : ℕ), (a / (b : ℚ) + c / (d : ℚ) = 1) ∧ (a / (d : ℚ) + c / (b : ℚ) = 2008) :=
sorry

end exist_nat_nums_l2066_206651


namespace calc_neg_half_times_neg_two_pow_l2066_206645

theorem calc_neg_half_times_neg_two_pow :
  - (0.5 ^ 20) * ((-2) ^ 26) = -64 := by
  sorry

end calc_neg_half_times_neg_two_pow_l2066_206645


namespace problem_y_equals_x_squared_plus_x_minus_6_l2066_206628

theorem problem_y_equals_x_squared_plus_x_minus_6 (x y : ℝ) :
  (y = x^2 + x - 6 ∧ x = 0 → y = -6) ∧ 
  (y = 0 → x = -3 ∨ x = 2) :=
by
  sorry

end problem_y_equals_x_squared_plus_x_minus_6_l2066_206628


namespace simplify_expression_l2066_206640

variable (a b : Real)

theorem simplify_expression (a b : Real) : 
    3 * b * (3 * b ^ 2 + 2 * b) - b ^ 2 + 2 * a * (2 * a ^ 2 - 3 * a) - 4 * a * b = 
    9 * b ^ 3 + 5 * b ^ 2 + 4 * a ^ 3 - 6 * a ^ 2 - 4 * a * b := by
  sorry

end simplify_expression_l2066_206640


namespace find_p_from_binomial_distribution_l2066_206683

theorem find_p_from_binomial_distribution (p : ℝ) (h₁ : 0 ≤ p ∧ p ≤ 1) 
    (h₂ : ∀ n k : ℕ, k ≤ n → 0 ≤ p^(k:ℝ) * (1-p)^((n-k):ℝ)) 
    (h₃ : (1 - (1 - p)^2 = 5 / 9)) : p = 1 / 3 :=
by sorry

end find_p_from_binomial_distribution_l2066_206683


namespace combined_weight_l2066_206641

noncomputable def Jake_weight : ℕ := 196
noncomputable def Kendra_weight : ℕ := 94

-- Condition: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom lose_8_pounds (j k : ℕ) : (j - 8 = 2 * k) → j = Jake_weight → k = Kendra_weight

-- To Prove: The combined weight of Jake and Kendra is 290 pounds
theorem combined_weight (j k : ℕ) (h₁ : j = Jake_weight) (h₂ : k = Kendra_weight) : j + k = 290 := 
by  sorry

end combined_weight_l2066_206641


namespace interior_edges_sum_l2066_206674

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l2066_206674


namespace max_price_of_product_l2066_206615

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end max_price_of_product_l2066_206615


namespace smallest_possible_value_of_M_l2066_206636

theorem smallest_possible_value_of_M :
  ∃ (N M : ℕ), N > 0 ∧ M > 0 ∧ 
               ∃ (r_6 r_36 r_216 r_M : ℕ), 
               r_6 < 6 ∧ 
               r_6 < r_36 ∧ r_36 < 36 ∧ 
               r_36 < r_216 ∧ r_216 < 216 ∧ 
               r_216 < r_M ∧ 
               r_36 = (r_6 * r) ∧ 
               r_216 = (r_6 * r^2) ∧ 
               r_M = (r_6 * r^3) ∧ 
               Nat.mod N 6 = r_6 ∧ 
               Nat.mod N 36 = r_36 ∧ 
               Nat.mod N 216 = r_216 ∧ 
               Nat.mod N M = r_M ∧ 
               M = 2001 :=
sorry

end smallest_possible_value_of_M_l2066_206636


namespace max_a_plus_ab_plus_abc_l2066_206684

noncomputable def f (a b c: ℝ) := a + a * b + a * b * c

theorem max_a_plus_ab_plus_abc (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h2 : a + b + c = 1) :
  ∃ x, (f a b c ≤ x) ∧ (∀ y, f a b c ≤ y → y = 1) :=
sorry

end max_a_plus_ab_plus_abc_l2066_206684


namespace solution_set_of_inequality_l2066_206693

theorem solution_set_of_inequality (x : ℝ) :  (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
by
  sorry

end solution_set_of_inequality_l2066_206693


namespace range_of_vector_magnitude_l2066_206620

variable {V : Type} [NormedAddCommGroup V]

theorem range_of_vector_magnitude
  (A B C : V)
  (h_AB : ‖A - B‖ = 8)
  (h_AC : ‖A - C‖ = 5) :
  3 ≤ ‖B - C‖ ∧ ‖B - C‖ ≤ 13 :=
sorry

end range_of_vector_magnitude_l2066_206620


namespace trigonometric_expression_identity_l2066_206638

theorem trigonometric_expression_identity :
  (2 * Real.sin (100 * Real.pi / 180) - Real.cos (70 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180)
  = 2 * Real.sqrt 3 - 1 :=
sorry

end trigonometric_expression_identity_l2066_206638


namespace sum_of_endpoints_l2066_206665

noncomputable def triangle_side_length (PQ QR PR QS PS : ℝ) (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = (PS / QS)) : ℝ :=
  if 4 < PR ∧ PR < 18 then 4 + 18 else 0

theorem sum_of_endpoints {PQ PR QS PS : ℝ} (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = ( PS / QS)) :
  triangle_side_length PQ 0 PR QS PS h1 h2 h3 = 22 := by
  sorry

end sum_of_endpoints_l2066_206665


namespace M_intersection_N_eq_M_l2066_206685

def is_element_of_M (y : ℝ) : Prop := ∃ x : ℝ, y = 2^x
def is_element_of_N (y : ℝ) : Prop := ∃ x : ℝ, y = x^2

theorem M_intersection_N_eq_M : {y | is_element_of_M y} ∩ {y | is_element_of_N y} = {y | is_element_of_M y} :=
by
  sorry

end M_intersection_N_eq_M_l2066_206685


namespace solve_equation_l2066_206677

theorem solve_equation : ∀ x : ℝ, (3 * (x - 2) + 1 = x - (2 * x - 1)) → x = 3 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l2066_206677


namespace total_distance_traveled_l2066_206631

theorem total_distance_traveled :
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  show total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2)
  sorry

end total_distance_traveled_l2066_206631


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l2066_206667

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l2066_206667


namespace impossible_to_load_two_coins_l2066_206666

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l2066_206666


namespace binomials_product_evaluation_l2066_206669

-- Define the binomials and the resulting polynomial
def binomial_one (x : ℝ) := 4 * x + 3
def binomial_two (x : ℝ) := 2 * x - 6
def resulting_polynomial (x : ℝ) := 8 * x^2 - 18 * x - 18

-- Define the proof problem
theorem binomials_product_evaluation :
  ∀ (x : ℝ), (binomial_one x) * (binomial_two x) = resulting_polynomial x ∧ 
  resulting_polynomial (-1) = 8 := 
by 
  intro x
  have h1 : (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 := sorry
  have h2 : resulting_polynomial (-1) = 8 := sorry
  exact ⟨h1, h2⟩

end binomials_product_evaluation_l2066_206669


namespace mass_percentage_Al_aluminum_carbonate_l2066_206676

theorem mass_percentage_Al_aluminum_carbonate :
  let m_Al := 26.98  -- molar mass of Al in g/mol
  let m_C := 12.01  -- molar mass of C in g/mol
  let m_O := 16.00  -- molar mass of O in g/mol
  let molar_mass_CO3 := m_C + 3 * m_O  -- molar mass of CO3 in g/mol
  let molar_mass_Al2CO33 := 2 * m_Al + 3 * molar_mass_CO3  -- molar mass of Al2(CO3)3 in g/mol
  let mass_Al_in_Al2CO33 := 2 * m_Al  -- mass of Al in Al2(CO3)3 in g/mol
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  -- Proof goes here
  sorry

end mass_percentage_Al_aluminum_carbonate_l2066_206676


namespace bianca_initial_cupcakes_l2066_206626

theorem bianca_initial_cupcakes (X : ℕ) (h : X - 6 + 17 = 25) : X = 14 := by
  sorry

end bianca_initial_cupcakes_l2066_206626


namespace gumball_water_wednesday_l2066_206606

variable (water_Mon_Thu_Sat : ℕ)
variable (water_Tue_Fri_Sun : ℕ)
variable (water_total : ℕ)
variable (water_Wed : ℕ)

theorem gumball_water_wednesday 
  (h1 : water_Mon_Thu_Sat = 9) 
  (h2 : water_Tue_Fri_Sun = 8) 
  (h3 : water_total = 60) 
  (h4 : 3 * water_Mon_Thu_Sat + 3 * water_Tue_Fri_Sun + water_Wed = water_total) : 
  water_Wed = 9 := 
by 
  sorry

end gumball_water_wednesday_l2066_206606


namespace multiples_of_9_ending_in_5_l2066_206610

theorem multiples_of_9_ending_in_5 (n : ℕ) :
  (∃ k : ℕ, n = 9 * k ∧ 0 < n ∧ n < 600 ∧ n % 10 = 5) → 
  ∃ l, l = 7 := 
by
sorry

end multiples_of_9_ending_in_5_l2066_206610


namespace maximum_ratio_x_over_y_l2066_206649

theorem maximum_ratio_x_over_y {x y : ℕ} (hx : x > 9 ∧ x < 100) (hy : y > 9 ∧ y < 100)
  (hmean : x + y = 110) (hsquare : ∃ z : ℕ, z^2 = x * y) : x = 99 ∧ y = 11 := 
by
  -- mathematical proof
  sorry

end maximum_ratio_x_over_y_l2066_206649


namespace highest_value_meter_l2066_206648

theorem highest_value_meter (A B C : ℝ) 
  (h_avg : (A + B + C) / 3 = 6)
  (h_A_min : A = 2)
  (h_B_min : B = 2) : C = 14 :=
by {
  sorry
}

end highest_value_meter_l2066_206648


namespace select_best_athlete_l2066_206604

theorem select_best_athlete :
  let avg_A := 185
  let var_A := 3.6
  let avg_B := 180
  let var_B := 3.6
  let avg_C := 185
  let var_C := 7.4
  let avg_D := 180
  let var_D := 8.1
  avg_A = 185 ∧ var_A = 3.6 ∧
  avg_B = 180 ∧ var_B = 3.6 ∧
  avg_C = 185 ∧ var_C = 7.4 ∧
  avg_D = 180 ∧ var_D = 8.1 →
  (∃ x, (x = avg_A ∧ avg_A = 185 ∧ var_A = 3.6) ∧
        (∀ (y : ℕ), (y = avg_A) 
        → avg_A = 185 
        ∧ var_A <= var_C ∧ 
        var_A <= var_D 
        ∧ var_A <= var_B)) :=
by {
  sorry
}

end select_best_athlete_l2066_206604


namespace complement_of_A_l2066_206682

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

theorem complement_of_A :
  U \ A = {x | -3 < x ∧ x < 0} :=
sorry

end complement_of_A_l2066_206682


namespace min_value_pm_pn_l2066_206635

theorem min_value_pm_pn (x y : ℝ)
  (h : x ^ 2 - y ^ 2 / 3 = 1) 
  (hx : 1 ≤ x) : (8 * x - 3) = 5 :=
sorry

end min_value_pm_pn_l2066_206635


namespace gumballs_difference_l2066_206659

variable (x y : ℕ)

def total_gumballs := 16 + 12 + 20 + x + y
def avg_gumballs (T : ℕ) := T / 5

theorem gumballs_difference (h1 : 18 <= avg_gumballs (total_gumballs x y)) 
                            (h2 : avg_gumballs (total_gumballs x y) <= 27) : (87 - 42) = 45 := by
  sorry

end gumballs_difference_l2066_206659


namespace problem_1_problem_2_l2066_206688

-- Proof Problem 1
theorem problem_1 (x : ℝ) : (x^2 + 2 > |x - 4| - |x - 1|) ↔ (x > 1 ∨ x ≤ -1) :=
sorry

-- Proof Problem 2
theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂, x₁^2 + 2 ≥ |x₂ - a| - |x₂ - 1|) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l2066_206688


namespace work_completion_time_l2066_206679

theorem work_completion_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : B + C = 1 / 15)
  (h3 : C + A = 1 / 20) :
  1 / (A + B + C) = 10 :=
by
  sorry

end work_completion_time_l2066_206679


namespace cafe_purchase_max_items_l2066_206632

theorem cafe_purchase_max_items (total_money sandwich_cost soft_drink_cost : ℝ) (total_money_pos sandwich_cost_pos soft_drink_cost_pos : total_money > 0 ∧ sandwich_cost > 0 ∧ soft_drink_cost > 0) :
    total_money = 40 ∧ sandwich_cost = 5 ∧ soft_drink_cost = 1.50 →
    ∃ s d : ℕ, s + d = 10 ∧ total_money = sandwich_cost * s + soft_drink_cost * d :=
by
  sorry

end cafe_purchase_max_items_l2066_206632


namespace volume_of_rectangular_solid_l2066_206642

theorem volume_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : c * a = 6) :
  a * b * c = 30 := 
by
  -- sorry placeholder for the proof
  sorry

end volume_of_rectangular_solid_l2066_206642


namespace ordered_pair_a_c_l2066_206690

theorem ordered_pair_a_c (a c : ℝ) (h_quad: ∀ x : ℝ, a * x^2 + 16 * x + c = 0)
    (h_sum: a + c = 25) (h_ineq: a < c) : (a = 3 ∧ c = 22) :=
by
  -- The proof is omitted
  sorry

end ordered_pair_a_c_l2066_206690


namespace max_sqrt_expression_l2066_206699

open Real

theorem max_sqrt_expression (x y z : ℝ) (h_sum : x + y + z = 3)
  (hx : x ≥ -1) (hy : y ≥ -(2/3)) (hz : z ≥ -2) :
  sqrt (3 * x + 3) + sqrt (3 * y + 2) + sqrt (3 * z + 6) ≤ 2 * sqrt 15 := by
  sorry

end max_sqrt_expression_l2066_206699


namespace probability_Xavier_Yvonne_not_Zelda_l2066_206623

-- Define the probabilities of success for Xavier, Yvonne, and Zelda
def pXavier := 1 / 5
def pYvonne := 1 / 2
def pZelda := 5 / 8

-- Define the probability that Zelda does not solve the problem
def pNotZelda := 1 - pZelda

-- The desired probability that we want to prove equals 3/80
def desiredProbability := (pXavier * pYvonne * pNotZelda) = (3 / 80)

-- The statement of the problem in Lean
theorem probability_Xavier_Yvonne_not_Zelda :
  desiredProbability := by
  sorry

end probability_Xavier_Yvonne_not_Zelda_l2066_206623


namespace sum_odd_is_13_over_27_l2066_206622

-- Define the probability for rolling an odd and an even number
def prob_odd := 1 / 3
def prob_even := 2 / 3

-- Define the probability that the sum of three die rolls is odd
def prob_sum_odd : ℚ :=
  3 * prob_odd * prob_even^2 + prob_odd^3

-- Statement asserting the goal to be proved
theorem sum_odd_is_13_over_27 :
  prob_sum_odd = 13 / 27 :=
by
  sorry

end sum_odd_is_13_over_27_l2066_206622


namespace find_a_l2066_206616

theorem find_a (x y : ℝ) (a : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : a * x + 2 * y = 1) : a = -1 := by
  sorry

end find_a_l2066_206616


namespace area_of_rectangle_l2066_206656

-- Given conditions
def shadedSquareArea : ℝ := 4
def nonShadedSquareArea : ℝ := shadedSquareArea
def largerSquareArea : ℝ := 4 * 4  -- Since the side length is twice the previous squares

-- Problem statement
theorem area_of_rectangle (shadedSquareArea nonShadedSquareArea largerSquareArea : ℝ) :
  shadedSquareArea + nonShadedSquareArea + largerSquareArea = 24 :=
sorry

end area_of_rectangle_l2066_206656


namespace count_factors_of_product_l2066_206609

theorem count_factors_of_product :
  let n := 8^4 * 7^3 * 9^1 * 5^5
  ∃ (count : ℕ), count = 936 ∧ 
    ∀ f : ℕ, f ∣ n → ∃ a b c d : ℕ,
      a ≤ 12 ∧ b ≤ 2 ∧ c ≤ 5 ∧ d ≤ 3 ∧ 
      f = 2^a * 3^b * 5^c * 7^d :=
by sorry

end count_factors_of_product_l2066_206609


namespace masha_number_l2066_206639

theorem masha_number (x : ℝ) (n : ℤ) (ε : ℝ) (h1 : 0 ≤ ε) (h2 : ε < 1) (h3 : x = n + ε) (h4 : (n : ℝ) = 0.57 * x) : x = 100 / 57 :=
by
  sorry

end masha_number_l2066_206639


namespace car_speed_l2066_206657

-- Definitions based on the conditions
def distance : ℕ := 375
def time : ℕ := 5

-- Mathematically equivalent proof statement
theorem car_speed : distance / time = 75 := 
  by
  -- The actual proof will be placed here, but we'll skip it for now.
  sorry

end car_speed_l2066_206657


namespace sum_of_two_coprimes_l2066_206652

theorem sum_of_two_coprimes (n : ℤ) (h : n ≥ 7) : 
  ∃ a b : ℤ, a + b = n ∧ Int.gcd a b = 1 ∧ a > 1 ∧ b > 1 :=
by
  sorry

end sum_of_two_coprimes_l2066_206652


namespace find_number_of_observations_l2066_206647

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end find_number_of_observations_l2066_206647


namespace team_a_daily_work_rate_l2066_206654

theorem team_a_daily_work_rate
  (L : ℕ) (D1 : ℕ) (D2 : ℕ) (w : ℕ → ℕ)
  (hL : L = 8250)
  (hD1 : D1 = 4)
  (hD2 : D2 = 7)
  (hwB : ∀ (x : ℕ), w x = x + 150)
  (hwork : ∀ (x : ℕ), D1 * x + D2 * (x + (w x)) = L) :
  ∃ x : ℕ, x = 400 :=
by
  sorry

end team_a_daily_work_rate_l2066_206654


namespace initial_amount_l2066_206605

theorem initial_amount (x : ℝ) (h : 0.015 * x = 750) : x = 50000 :=
by
  sorry

end initial_amount_l2066_206605


namespace count_valid_triangles_l2066_206621

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem count_valid_triangles : 
  (∃ n : ℕ, n = 14 ∧ 
  ∃ (a b c : ℕ), valid_triangle a b c ∧ 
  ((b = 5 ∧ c > 5) ∨ (c = 5 ∧ b > 5)) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0)) :=
by { sorry }

end count_valid_triangles_l2066_206621


namespace evaluate_expression_l2066_206653

-- Define x as given in the condition
def x : ℤ := 5

-- State the theorem we need to prove
theorem evaluate_expression : x^3 - 3 * x = 110 :=
by
  -- Proof will be provided here
  sorry

end evaluate_expression_l2066_206653


namespace abc_ineq_l2066_206611

theorem abc_ineq (a b c : ℝ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : c > 0) (h₄ : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27 / 8 :=
sorry

end abc_ineq_l2066_206611


namespace multiple_of_one_third_l2066_206698

theorem multiple_of_one_third (x : ℚ) (h : x * (1 / 3) = 2 / 9) : x = 2 / 3 :=
sorry

end multiple_of_one_third_l2066_206698


namespace only_n_equal_one_l2066_206627

theorem only_n_equal_one (n : ℕ) (hn : 0 < n) : 
  (5 ^ (n - 1) + 3 ^ (n - 1)) ∣ (5 ^ n + 3 ^ n) → n = 1 := by
  intro h_div
  sorry

end only_n_equal_one_l2066_206627


namespace problem_3034_1002_20_04_div_sub_l2066_206694

theorem problem_3034_1002_20_04_div_sub:
  3034 - (1002 / 20.04) = 2984 :=
by
  sorry

end problem_3034_1002_20_04_div_sub_l2066_206694


namespace num_terminating_decimals_l2066_206655

-- Define the problem conditions and statement
def is_terminating_decimal (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_terminating_decimals : 
  ∃ (k : ℕ), k = 220 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 660 → is_terminating_decimal n ↔ n % 3 = 0) := 
by
  sorry

end num_terminating_decimals_l2066_206655


namespace vertices_integer_assignment_zero_l2066_206686

theorem vertices_integer_assignment_zero (f : ℕ → ℤ) (h100 : ∀ i, i < 100 → (i + 3) % 100 < 100) 
  (h : ∀ i, (i < 97 → f i + f (i + 2) = f (i + 1)) 
            ∨ (i < 97 → f (i + 1) + f (i + 3) = f (i + 2)) 
            ∨ (i < 97 → f i + f (i + 1) = f (i + 2))): 
  ∀ i, i < 100 → f i = 0 :=
by
  sorry

end vertices_integer_assignment_zero_l2066_206686


namespace increase_in_cost_l2066_206646

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def initial_total_cost : ℝ := initial_lumber_cost + initial_nails_cost + initial_fabric_cost

def new_lumber_cost : ℝ := initial_lumber_cost * (1 + lumber_inflation_rate)
def new_nails_cost : ℝ := initial_nails_cost * (1 + nails_inflation_rate)
def new_fabric_cost : ℝ := initial_fabric_cost * (1 + fabric_inflation_rate)

def new_total_cost : ℝ := new_lumber_cost + new_nails_cost + new_fabric_cost

theorem increase_in_cost :
  new_total_cost - initial_total_cost = 97 := 
sorry

end increase_in_cost_l2066_206646


namespace hands_per_student_l2066_206643

theorem hands_per_student (hands_without_peter : ℕ) (total_students : ℕ) (hands_peter : ℕ) 
  (h1 : hands_without_peter = 20) 
  (h2 : total_students = 11) 
  (h3 : hands_peter = 2) : 
  (hands_without_peter + hands_peter) / total_students = 2 :=
by
  sorry

end hands_per_student_l2066_206643


namespace parametric_to_cartesian_l2066_206608

theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : x = 5 + 3 * t) (h2 : y = 10 - 4 * t) : 4 * x + 3 * y = 50 :=
by sorry

end parametric_to_cartesian_l2066_206608


namespace sale_prices_correct_l2066_206670

-- Define the cost prices and profit percentages
def cost_price_A : ℕ := 320
def profit_percentage_A : ℕ := 50

def cost_price_B : ℕ := 480
def profit_percentage_B : ℕ := 70

def cost_price_C : ℕ := 600
def profit_percentage_C : ℕ := 40

-- Define the expected sale prices
def sale_price_A : ℕ := 480
def sale_price_B : ℕ := 816
def sale_price_C : ℕ := 840

-- Define a function to compute sale price
def compute_sale_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (profit_percentage * cost_price) / 100

-- The proof statement
theorem sale_prices_correct :
  compute_sale_price cost_price_A profit_percentage_A = sale_price_A ∧
  compute_sale_price cost_price_B profit_percentage_B = sale_price_B ∧
  compute_sale_price cost_price_C profit_percentage_C = sale_price_C :=
by {
  sorry
}

end sale_prices_correct_l2066_206670


namespace find_f_of_3pi_by_4_l2066_206650

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem find_f_of_3pi_by_4 : f (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end find_f_of_3pi_by_4_l2066_206650


namespace part_a_part_b_l2066_206695

theorem part_a (A B : ℕ) (hA : 1 ≤ A) (hB : 1 ≤ B) : 
  (A + B = 70) → 
  (A * (4 : ℚ) / 35 + B * (4 : ℚ) / 35 = 8) :=
  by
    sorry

theorem part_b (C D : ℕ) (r : ℚ) (hC : C > 1) (hD : D > 1) (hr : r > 1) :
  (C + D = 8 / r) → 
  (C * r + D * r = 8) → 
  (∃ ki : ℕ, (C + D = (70 : ℕ) / ki ∧ 1 < ki ∧ ki ∣ 70)) :=
  by
    sorry

end part_a_part_b_l2066_206695


namespace min_shift_symmetric_y_axis_l2066_206696

theorem min_shift_symmetric_y_axis :
  ∃ (m : ℝ), m = 7 * Real.pi / 6 ∧ 
             (∀ x : ℝ, 2 * Real.cos (x + Real.pi / 3) = 2 * Real.cos (x + Real.pi / 3 + m)) ∧ 
             m > 0 :=
by
  sorry

end min_shift_symmetric_y_axis_l2066_206696


namespace cat_toy_cost_correct_l2066_206680

-- Define the initial amount of money Jessica had.
def initial_amount : ℝ := 11.73

-- Define the amount left after spending.
def amount_left : ℝ := 1.51

-- Define the cost of the cat toy.
def toy_cost : ℝ := initial_amount - amount_left

-- Theorem and statement to prove the cost of the cat toy.
theorem cat_toy_cost_correct : toy_cost = 10.22 := sorry

end cat_toy_cost_correct_l2066_206680


namespace hose_removal_rate_l2066_206618

def pool_volume (length width depth : ℕ) : ℕ :=
  length * width * depth

def draining_rate (volume time : ℕ) : ℕ :=
  volume / time

theorem hose_removal_rate :
  let length := 150
  let width := 80
  let depth := 10
  let total_volume := pool_volume length width depth
  total_volume = 1200000 ∧
  let time := 2000
  draining_rate total_volume time = 600 :=
by
  sorry

end hose_removal_rate_l2066_206618


namespace division_quotient_difference_l2066_206678

theorem division_quotient_difference :
  (32.5 / 1.3) - (60.8 / 7.6) = 17 :=
by
  sorry

end division_quotient_difference_l2066_206678


namespace john_bought_six_bagels_l2066_206660

theorem john_bought_six_bagels (b m : ℕ) (expenditure_in_dollars_whole : (90 * b + 60 * m) % 100 = 0) (total_items : b + m = 7) : 
b = 6 :=
by
  -- The proof goes here. For now, we skip it with sorry.
  sorry

end john_bought_six_bagels_l2066_206660


namespace travel_time_third_to_first_l2066_206614

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end travel_time_third_to_first_l2066_206614


namespace shaded_area_is_14_percent_l2066_206612

def side_length : ℕ := 20
def rectangle_width : ℕ := 35
def rectangle_height : ℕ := side_length
def rectangle_area : ℕ := rectangle_width * rectangle_height
def overlap_length : ℕ := 2 * side_length - rectangle_width
def shaded_area : ℕ := overlap_length * side_length
def shaded_percentage : ℚ := (shaded_area : ℚ) / rectangle_area * 100

theorem shaded_area_is_14_percent : shaded_percentage = 14 := by
  sorry

end shaded_area_is_14_percent_l2066_206612


namespace solve_r_l2066_206617

-- Define E(a, b, c) as given
def E (a b c : ℕ) : ℕ := a * b^c

-- Lean 4 statement for the proof
theorem solve_r (r : ℕ) (r_pos : 0 < r) : E r r 3 = 625 → r = 5 :=
by
  intro h
  sorry

end solve_r_l2066_206617
