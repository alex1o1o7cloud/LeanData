import Mathlib

namespace largest_B_div_by_4_l1584_158487

-- Given conditions
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- The seven-digit integer is 4B6792X
def number (B X : ℕ) : ℕ := 4000000 + B * 100000 + 60000 + 7000 + 900 + 20 + X

-- Problem statement: Prove that the largest digit B so that the seven-digit integer 4B6792X is divisible by 4
theorem largest_B_div_by_4 
(B X : ℕ) 
(hX : is_digit X)
(div_4 : divisible_by_4 (number B X)) : 
B = 9 := sorry

end largest_B_div_by_4_l1584_158487


namespace sum_of_ages_l1584_158402

theorem sum_of_ages {l t : ℕ} (h1 : t > l) (h2 : t * t * l = 72) : t + t + l = 14 :=
sorry

end sum_of_ages_l1584_158402


namespace probability_no_adjacent_same_roll_l1584_158459

theorem probability_no_adjacent_same_roll :
  let A := 1 -- rolls a six-sided die
  let B := 2 -- rolls a six-sided die
  let C := 3 -- rolls a six-sided die
  let D := 4 -- rolls a six-sided die
  let E := 5 -- rolls a six-sided die
  let people := [A, B, C, D, E]
  -- A and C are required to roll different numbers
  let prob_A_C_diff := 5 / 6
  -- B must roll different from A and C
  let prob_B_diff := 4 / 6
  -- D must roll different from C and A
  let prob_D_diff := 4 / 6
  -- E must roll different from D and A
  let prob_E_diff := 3 / 6
  (prob_A_C_diff * prob_B_diff * prob_D_diff * prob_E_diff) = 10 / 27 :=
by
  sorry

end probability_no_adjacent_same_roll_l1584_158459


namespace find_side_b_l1584_158461

theorem find_side_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * Real.sin B = Real.sin A + Real.sin C)
  (h2 : Real.cos B = 3 / 5)
  (h3 : (1 / 2) * a * c * Real.sin B = 4) :
  b = 4 * Real.sqrt 6 / 3 := 
sorry

end find_side_b_l1584_158461


namespace sum_1026_is_2008_l1584_158469

def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let groups_sum : ℕ := (n * n)
    let extra_2s := (2008 - groups_sum) / 2
    (n * (n + 1)) / 2 + extra_2s

theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 :=
  sorry

end sum_1026_is_2008_l1584_158469


namespace car_bus_washing_inconsistency_l1584_158419

theorem car_bus_washing_inconsistency :
  ∀ (C B : ℕ), 
    C % 2 = 0 →
    B % 2 = 1 →
    7 * C + 18 * B = 309 →
    3 + 8 + 5 + C + B = 15 →
    false :=
by
  sorry

end car_bus_washing_inconsistency_l1584_158419


namespace num_valid_m_divisors_of_1750_l1584_158434

theorem num_valid_m_divisors_of_1750 : 
  ∃! (m : ℕ) (h1 : m > 0), ∃ (k : ℕ), k > 0 ∧ 1750 = k * (m^2 - 4) :=
sorry

end num_valid_m_divisors_of_1750_l1584_158434


namespace ones_digit_of_73_pow_351_l1584_158472

theorem ones_digit_of_73_pow_351 : 
  (73 ^ 351) % 10 = 7 := 
by 
  sorry

end ones_digit_of_73_pow_351_l1584_158472


namespace inequality_solution_set_l1584_158421

theorem inequality_solution_set : 
  { x : ℝ | (x + 1) / (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry 

end inequality_solution_set_l1584_158421


namespace polygonal_number_8_8_l1584_158464

-- Definitions based on conditions
def triangular_number (n : ℕ) : ℕ := (n^2 + n) / 2
def square_number (n : ℕ) : ℕ := n^2
def pentagonal_number (n : ℕ) : ℕ := (3 * n^2 - n) / 2
def hexagonal_number (n : ℕ) : ℕ := (4 * n^2 - 2 * n) / 2

-- General formula for k-sided polygonal number
def polygonal_number (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

-- The proposition to be proved
theorem polygonal_number_8_8 : polygonal_number 8 8 = 176 := by
  sorry

end polygonal_number_8_8_l1584_158464


namespace calculate_Y_payment_l1584_158478

-- Define the known constants
def total_payment : ℝ := 590
def x_to_y_ratio : ℝ := 1.2

-- Main theorem statement, asserting the value of Y's payment
theorem calculate_Y_payment (Y : ℝ) (X : ℝ) 
  (h1 : X = x_to_y_ratio * Y) 
  (h2 : X + Y = total_payment) : 
  Y = 268.18 :=
by
  sorry

end calculate_Y_payment_l1584_158478


namespace evaluate_expression_l1584_158441

theorem evaluate_expression (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := 
sorry

end evaluate_expression_l1584_158441


namespace rancher_total_animals_l1584_158448

theorem rancher_total_animals
  (H C : ℕ) (h1 : C = 5 * H) (h2 : C = 140) :
  C + H = 168 := 
sorry

end rancher_total_animals_l1584_158448


namespace stream_speed_l1584_158453

theorem stream_speed (v : ℝ) (t : ℝ) (h1 : t > 0)
  (h2 : ∃ k : ℝ, k = 2 * t)
  (h3 : (9 + v) * t = (9 - v) * (2 * t)) :
  v = 3 := 
sorry

end stream_speed_l1584_158453


namespace max_value_of_3x_plus_4y_l1584_158473

theorem max_value_of_3x_plus_4y (x y : ℝ) 
(h : x^2 + y^2 = 14 * x + 6 * y + 6) : 
3 * x + 4 * y ≤ 73 := 
sorry

end max_value_of_3x_plus_4y_l1584_158473


namespace perpendicular_sum_value_of_m_l1584_158426

-- Let a and b be defined as vectors in R^2
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product for vectors in R^2
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors using dot product
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the sum of two vectors
def vector_sum (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- State our proof problem
theorem perpendicular_sum_value_of_m :
  is_perpendicular (vector_sum vector_a (vector_b (-7 / 2))) vector_a :=
by
  -- Proof omitted
  sorry

end perpendicular_sum_value_of_m_l1584_158426


namespace find_B_l1584_158449

structure Point where
  x : Int
  y : Int

def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 2⟩
def BA : Point := ⟨3, 3⟩
def B : Point := ⟨-4, -1⟩

theorem find_B :
  vector_sub A BA = B :=
by
  sorry

end find_B_l1584_158449


namespace pq_square_identity_l1584_158407

theorem pq_square_identity (p q : ℝ) (h1 : p - q = 4) (h2 : p * q = -2) : p^2 + q^2 = 12 :=
by
  sorry

end pq_square_identity_l1584_158407


namespace tickets_needed_l1584_158475

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l1584_158475


namespace license_plate_palindrome_probability_l1584_158447

theorem license_plate_palindrome_probability : 
  let p := 775 
  let q := 67600  
  p + q = 776 :=
by
  let p := 775
  let q := 67600
  show p + q = 776
  sorry

end license_plate_palindrome_probability_l1584_158447


namespace ratio_Jane_to_John_l1584_158435

-- Define the conditions as given in the problem.
variable (J N : ℕ) -- total products inspected by John and Jane
variable (rJ rN rT : ℚ) -- rejection rates for John, Jane, and total

-- Setting up the provided conditions
axiom h1 : rJ = 0.005 -- John rejected 0.5% of the products he inspected
axiom h2 : rN = 0.007 -- Jane rejected 0.7% of the products she inspected
axiom h3 : rT = 0.0075 -- 0.75% of the total products were rejected

-- Prove the ratio of products inspected by Jane to products inspected by John is 5
theorem ratio_Jane_to_John : (rJ * J + rN * N) = rT * (J + N) → N = 5 * J :=
by 
  sorry

end ratio_Jane_to_John_l1584_158435


namespace range_of_a_l1584_158466

theorem range_of_a (a : ℝ) : 1 ∉ {x : ℝ | x^2 - 2 * x + a > 0} → a ≤ 1 :=
by
  sorry

end range_of_a_l1584_158466


namespace total_area_of_removed_triangles_l1584_158450

theorem total_area_of_removed_triangles (side_length : ℝ) (half_leg_length : ℝ) :
  side_length = 16 →
  half_leg_length = side_length / 4 →
  4 * (1 / 2) * half_leg_length^2 = 32 :=
by
  intro h_side_length h_half_leg_length
  simp [h_side_length, h_half_leg_length]
  sorry

end total_area_of_removed_triangles_l1584_158450


namespace danny_initial_wrappers_l1584_158467

def initial_wrappers (total_wrappers: ℕ) (found_wrappers: ℕ): ℕ :=
  total_wrappers - found_wrappers

theorem danny_initial_wrappers : initial_wrappers 57 30 = 27 :=
by
  exact rfl

end danny_initial_wrappers_l1584_158467


namespace comb_n_plus_1_2_l1584_158412

theorem comb_n_plus_1_2 (n : ℕ) (h : 0 < n) : 
  (n + 1).choose 2 = (n + 1) * n / 2 :=
by sorry

end comb_n_plus_1_2_l1584_158412


namespace cistern_emptying_time_l1584_158406

theorem cistern_emptying_time (R L : ℝ) (h1 : R * 8 = 1) (h2 : (R - L) * 10 = 1) : 1 / L = 40 :=
by
  -- proof omitted
  sorry

end cistern_emptying_time_l1584_158406


namespace xiaoming_statement_incorrect_l1584_158445

theorem xiaoming_statement_incorrect (s : ℕ) : 
    let x_h := 3
    let x_m := 6
    let steps_xh := (x_h - 1) * s
    let steps_xm := (x_m - 1) * s
    (steps_xm ≠ 2 * steps_xh) :=
by
  let x_h := 3
  let x_m := 6
  let steps_xh := (x_h - 1) * s
  let steps_xm := (x_m - 1) * s
  sorry

end xiaoming_statement_incorrect_l1584_158445


namespace fraction_problem_l1584_158436

def fractions : List (ℚ) := [4/3, 7/5, 12/10, 23/20, 45/40, 89/80]
def subtracted_value : ℚ := -8

theorem fraction_problem :
  (fractions.sum - subtracted_value) = -163 / 240 := by
  sorry

end fraction_problem_l1584_158436


namespace intersection_with_y_axis_l1584_158432

theorem intersection_with_y_axis :
  ∃ (y : ℝ), (y = -x^2 + 3*x - 4) ∧ (x = 0) ∧ (y = -4) := 
by
  sorry

end intersection_with_y_axis_l1584_158432


namespace Heechul_has_most_books_l1584_158489

namespace BookCollection

variables (Heejin Heechul Dongkyun : ℕ)

theorem Heechul_has_most_books (h_h : ℕ) (h_j : ℕ) (d : ℕ) 
  (h_h_eq : h_h = h_j + 2) (d_lt_h_j : d < h_j) : 
  h_h > h_j ∧ h_h > d := 
by
  sorry

end BookCollection

end Heechul_has_most_books_l1584_158489


namespace different_pronunciation_in_group_C_l1584_158488

theorem different_pronunciation_in_group_C :
  let groupC := [("戏谑", "xuè"), ("虐待", "nüè"), ("瘠薄", "jí"), ("脊梁", "jǐ"), ("赝品", "yàn"), ("义愤填膺", "yīng")]
  ∀ {a : String} {b : String}, (a, b) ∈ groupC → a ≠ b :=
by
  intro groupC h
  sorry

end different_pronunciation_in_group_C_l1584_158488


namespace trapezoid_midsegment_inscribed_circle_l1584_158492

theorem trapezoid_midsegment_inscribed_circle (P : ℝ) (hP : P = 40) 
    (inscribed : Π (a b c d : ℝ), a + b = c + d) : 
    (∃ (c d : ℝ), (c + d) / 2 = 10) :=
by
  sorry

end trapezoid_midsegment_inscribed_circle_l1584_158492


namespace profit_function_and_optimal_price_l1584_158494

variable (cost selling base_units additional_units: ℝ)
variable (x: ℝ) (y: ℝ)

def profit (x: ℝ): ℝ := -20 * x^2 + 100 * x + 6000

theorem profit_function_and_optimal_price:
  (cost = 40) →
  (selling = 60) →
  (base_units = 300) →
  (additional_units = 20) →
  (0 ≤ x) →
  (x < 20) →
  (y = profit x) →
  exists x_max y_max: ℝ, (x_max = 2.5) ∧ (y_max = 6125) :=
by 
  sorry

end profit_function_and_optimal_price_l1584_158494


namespace calculate_expression_l1584_158499

theorem calculate_expression :
  |1 - Real.sqrt 2| + (1/2)^(-2 : ℤ) - (Real.pi - 2023)^0 = Real.sqrt 2 + 2 := 
by
  sorry

end calculate_expression_l1584_158499


namespace number_of_students_in_first_group_l1584_158497

def total_students : ℕ := 24
def second_group : ℕ := 8
def third_group : ℕ := 7
def fourth_group : ℕ := 4
def summed_other_groups : ℕ := second_group + third_group + fourth_group
def students_first_group : ℕ := total_students - summed_other_groups

theorem number_of_students_in_first_group :
  students_first_group = 5 :=
by
  -- proof required here
  sorry

end number_of_students_in_first_group_l1584_158497


namespace ratio_of_speeds_l1584_158422

theorem ratio_of_speeds (v_A v_B : ℝ) (t : ℝ) (hA : v_A = 120 / t) (hB : v_B = 60 / t) : v_A / v_B = 2 :=
by {
  sorry
}

end ratio_of_speeds_l1584_158422


namespace customers_per_table_l1584_158420

theorem customers_per_table (total_tables : ℝ) (left_tables : ℝ) (total_customers : ℕ)
  (h1 : total_tables = 44.0)
  (h2 : left_tables = 12.0)
  (h3 : total_customers = 256) :
  total_customers / (total_tables - left_tables) = 8 :=
by {
  sorry
}

end customers_per_table_l1584_158420


namespace range_of_phi_l1584_158477

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + 2 * φ)

theorem range_of_phi :
  ∀ φ : ℝ,
  (0 < φ) ∧ (φ < π / 2) →
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → g x φ ≤ g (x + π/6) φ) →
  (∃ x : ℝ, -π/6 < x ∧ x < 0 ∧ g x φ = 0) →
  φ ∈ Set.Ioc (π / 4) (π / 3) := 
by
  intros φ h1 h2 h3
  sorry

end range_of_phi_l1584_158477


namespace pump_fills_tank_without_leak_l1584_158463

theorem pump_fills_tank_without_leak (T : ℝ) (h1 : 1 / 12 = 1 / T - 1 / 12) : T = 6 :=
sorry

end pump_fills_tank_without_leak_l1584_158463


namespace multiples_of_3_or_4_probability_l1584_158414

theorem multiples_of_3_or_4_probability :
  let total_cards := 36
  let multiples_of_3 := 12
  let multiples_of_4 := 9
  let multiples_of_both := 3
  let favorable_outcomes := multiples_of_3 + multiples_of_4 - multiples_of_both
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 2 :=
by
  sorry

end multiples_of_3_or_4_probability_l1584_158414


namespace not_divisible_l1584_158431

theorem not_divisible (x y : ℕ) (hx : x % 61 ≠ 0) (hy : y % 61 ≠ 0) (h : (7 * x + 34 * y) % 61 = 0) : (5 * x + 16 * y) % 61 ≠ 0 := 
sorry

end not_divisible_l1584_158431


namespace simplify_expression_l1584_158465

theorem simplify_expression (x : ℝ) :
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) = x^4 - 1 :=
  by 
    sorry

end simplify_expression_l1584_158465


namespace scientific_notation_of_4212000_l1584_158405

theorem scientific_notation_of_4212000 :
  4212000 = 4.212 * 10^6 :=
by
  sorry

end scientific_notation_of_4212000_l1584_158405


namespace total_savings_eighteen_l1584_158479

theorem total_savings_eighteen :
  let fox_price := 15
  let pony_price := 18
  let discount_rate_sum := 50
  let fox_quantity := 3
  let pony_quantity := 2
  let pony_discount_rate := 50
  let total_price_without_discount := (fox_quantity * fox_price) + (pony_quantity * pony_price)
  let discounted_pony_price := (pony_price * (1 - (pony_discount_rate / 100)))
  let total_price_with_discount := (fox_quantity * fox_price) + (pony_quantity * discounted_pony_price)
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 18 :=
by sorry

end total_savings_eighteen_l1584_158479


namespace john_total_payment_l1584_158440

theorem john_total_payment :
  let cost_per_appointment := 400
  let total_appointments := 3
  let pet_insurance_cost := 100
  let insurance_coverage := 0.80
  let first_appointment_cost := cost_per_appointment
  let subsequent_appointments := total_appointments - 1
  let subsequent_appointments_cost := subsequent_appointments * cost_per_appointment
  let covered_cost := subsequent_appointments_cost * insurance_coverage
  let uncovered_cost := subsequent_appointments_cost - covered_cost
  let total_cost := first_appointment_cost + pet_insurance_cost + uncovered_cost
  total_cost = 660 :=
by
  sorry

end john_total_payment_l1584_158440


namespace stacked_cubes_surface_area_is_945_l1584_158428

def volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

def side_length (v : ℕ) : ℕ := v^(1/3)

def num_visible_faces (i : ℕ) : ℕ :=
  if i == 0 then 5 else 3 -- Bottom cube has 5 faces visible, others have 3 due to rotation

def surface_area (s : ℕ) (faces : ℕ) : ℕ :=
  faces * s^2

def total_surface_area (volumes : List ℕ) : ℕ :=
  (volumes.zipWith surface_area (volumes.enum.map (λ (i, v) => num_visible_faces i))).sum

theorem stacked_cubes_surface_area_is_945 :
  total_surface_area volumes = 945 := 
by 
  sorry

end stacked_cubes_surface_area_is_945_l1584_158428


namespace three_digit_multiples_of_seven_l1584_158484

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l1584_158484


namespace slices_per_pizza_l1584_158495

def num_pizzas : ℕ := 2
def total_slices : ℕ := 16

theorem slices_per_pizza : total_slices / num_pizzas = 8 := by
  sorry

end slices_per_pizza_l1584_158495


namespace part_a_part_b_part_c_l1584_158411

-- Definitions for the convex polyhedron, volume, and surface area
structure ConvexPolyhedron :=
  (volume : ℝ)
  (surface_area : ℝ)

variable {P : ConvexPolyhedron}

-- Statement for Part (a)
theorem part_a (r : ℝ) (h_r : r ≤ P.surface_area) :
  P.volume / P.surface_area ≥ r / 3 := sorry

-- Statement for Part (b)
theorem part_b :
  Exists (fun r : ℝ => r = P.volume / P.surface_area) := sorry

-- Definitions and conditions for the outer and inner polyhedron
structure ConvexPolyhedronPair :=
  (outer_polyhedron : ConvexPolyhedron)
  (inner_polyhedron : ConvexPolyhedron)

variable {CP : ConvexPolyhedronPair}

-- Statement for Part (c)
theorem part_c :
  3 * CP.outer_polyhedron.volume / CP.outer_polyhedron.surface_area ≥
  CP.inner_polyhedron.volume / CP.inner_polyhedron.surface_area := sorry

end part_a_part_b_part_c_l1584_158411


namespace sum_of_three_consecutive_integers_divisible_by_3_l1584_158460

theorem sum_of_three_consecutive_integers_divisible_by_3 (a : ℤ) :
  ∃ k : ℤ, k = 3 ∧ (a - 1 + a + (a + 1)) % k = 0 :=
by
  use 3
  sorry

end sum_of_three_consecutive_integers_divisible_by_3_l1584_158460


namespace paul_account_balance_after_transactions_l1584_158482

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l1584_158482


namespace sum_of_integers_l1584_158496

theorem sum_of_integers {n : ℤ} (h : n + 2 = 9) : n + (n + 1) + (n + 2) = 24 := by
  sorry

end sum_of_integers_l1584_158496


namespace find_a_and_b_l1584_158457

theorem find_a_and_b (a b : ℝ) :
  (∀ x, y = a + b / x) →
  (y = 3 → x = 2) →
  (y = -1 → x = -4) →
  a + b = 4 :=
by sorry

end find_a_and_b_l1584_158457


namespace smartphone_price_l1584_158423

theorem smartphone_price (S : ℝ) (pc_price : ℝ) (tablet_price : ℝ) 
  (total_cost : ℝ) (h1 : pc_price = S + 500) 
  (h2 : tablet_price = 2 * S + 500) 
  (h3 : S + pc_price + tablet_price = 2200) : 
  S = 300 :=
by
  sorry

end smartphone_price_l1584_158423


namespace decagon_perimeter_l1584_158454

-- Define the number of sides in a decagon
def num_sides : ℕ := 10

-- Define the length of each side in the decagon
def side_length : ℕ := 3

-- Define the perimeter of a decagon given the number of sides and the side length
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s

-- State the theorem we want to prove: the perimeter of our given regular decagon
theorem decagon_perimeter : perimeter num_sides side_length = 30 := 
by sorry

end decagon_perimeter_l1584_158454


namespace cylinder_volume_options_l1584_158409

theorem cylinder_volume_options (length width : ℝ) (h₀ : length = 4) (h₁ : width = 2) :
  ∃ V, (V = (4 / π) ∨ V = (8 / π)) :=
by
  sorry

end cylinder_volume_options_l1584_158409


namespace f_23_plus_f_neg14_l1584_158452

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x, f (x + 5) = f x
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

theorem f_23_plus_f_neg14 : f 23 + f (-14) = -1 := by
  sorry

end f_23_plus_f_neg14_l1584_158452


namespace boat_speed_still_water_l1584_158429

theorem boat_speed_still_water (v c : ℝ) (h1 : v + c = 13) (h2 : v - c = 4) : v = 8.5 :=
by sorry

end boat_speed_still_water_l1584_158429


namespace find_13_points_within_radius_one_l1584_158438

theorem find_13_points_within_radius_one (points : Fin 25 → ℝ × ℝ)
  (h : ∀ i j k : Fin 25, min (dist (points i) (points j)) (min (dist (points i) (points k)) (dist (points j) (points k))) < 1) :
  ∃ (subset : Finset (Fin 25)), subset.card = 13 ∧ ∃ (center : ℝ × ℝ), ∀ i ∈ subset, dist (points i) center < 1 :=
  sorry

end find_13_points_within_radius_one_l1584_158438


namespace weekly_car_mileage_l1584_158490

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end weekly_car_mileage_l1584_158490


namespace max_b_n_occurs_at_n_l1584_158443

def a_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  a1 + (n-1) * d

def S_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  n * a1 + (n * (n-1) / 2) * d

def b_n (n : ℕ) (an : ℚ) : ℚ :=
  (1 + an) / an

theorem max_b_n_occurs_at_n :
  ∀ (n : ℕ) (a1 d : ℚ),
  (a1 = -5/2) →
  (S_n 4 a1 d = 2 * S_n 2 a1 d + 4) →
  n = 4 := sorry

end max_b_n_occurs_at_n_l1584_158443


namespace rowing_speed_downstream_l1584_158483

/--
A man can row upstream at 25 kmph and downstream at a certain speed. 
The speed of the man in still water is 30 kmph. 
Prove that the speed of the man rowing downstream is 35 kmph.
-/
theorem rowing_speed_downstream (V_u V_sw V_s V_d : ℝ)
  (h1 : V_u = 25) 
  (h2 : V_sw = 30) 
  (h3 : V_u = V_sw - V_s) 
  (h4 : V_d = V_sw + V_s) :
  V_d = 35 :=
by
  sorry

end rowing_speed_downstream_l1584_158483


namespace num_ordered_triples_l1584_158430

theorem num_ordered_triples 
  (a b c : ℕ)
  (h_cond1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ c)
  (h_cond2 : a * b * c = 4 * (a * b + b * c + c * a)) : 
  ∃ (n : ℕ), n = 5 :=
sorry

end num_ordered_triples_l1584_158430


namespace shaded_area_percentage_l1584_158470

theorem shaded_area_percentage (total_area shaded_area : ℕ) (h_total : total_area = 49) (h_shaded : shaded_area = 33) : 
  (shaded_area : ℚ) / total_area = 33 / 49 := 
by
  sorry

end shaded_area_percentage_l1584_158470


namespace eight_percent_of_fifty_is_four_l1584_158425

theorem eight_percent_of_fifty_is_four : 0.08 * 50 = 4 := by
  sorry

end eight_percent_of_fifty_is_four_l1584_158425


namespace neg_sqrt_17_estimate_l1584_158413

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l1584_158413


namespace sum_of_remainders_mod_13_l1584_158462

theorem sum_of_remainders_mod_13 
  (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_mod_13_l1584_158462


namespace b_alone_completion_days_l1584_158416

theorem b_alone_completion_days (Rab : ℝ) (w_12_days : (1 / (Rab + 4 * Rab)) = 12⁻¹) : 
    (1 / Rab = 60) :=
sorry

end b_alone_completion_days_l1584_158416


namespace ones_mult_palindrome_l1584_158458

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 
  digits = digits.reverse

def ones (k : ℕ) : ℕ := (10 ^ k - 1) / 9

theorem ones_mult_palindrome (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_palindrome (ones m * ones n) ↔ (m = n ∧ m ≤ 9 ∧ n ≤ 9) := 
sorry

end ones_mult_palindrome_l1584_158458


namespace rational_linear_function_l1584_158493

theorem rational_linear_function (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
sorry

end rational_linear_function_l1584_158493


namespace cyclist_distance_l1584_158474

theorem cyclist_distance
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1) * (3 * t / 4))
  (h3 : d = (x - 1) * (t + 3)) :
  d = 18 :=
by {
  sorry
}

end cyclist_distance_l1584_158474


namespace range_of_alpha_l1584_158485

theorem range_of_alpha :
  ∀ P : ℝ, 
  (∃ y : ℝ, y = 4 / (Real.exp P + 1)) →
  (∃ α : ℝ, α = Real.arctan (4 / (Real.exp P + 2 + 1 / Real.exp P)) ∧ (Real.tan α) ∈ Set.Ico (-1) 0) → 
  Set.Ico (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_alpha_l1584_158485


namespace domain_f_x_plus_2_l1584_158456

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Define the given condition: the domain of y = f(2x - 3) is [-2, 3]
def domain_f_2x_minus_3 : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 3}

-- Express this condition formally
axiom domain_f_2x_minus_3_axiom :
  ∀ (x : ℝ), (x ∈ domain_f_2x_minus_3) → (2 * x - 3 ∈ Set.Icc (-7 : ℝ) 3)

-- Prove the desired result: the domain of y = f(x + 2) is [-9, 1]
theorem domain_f_x_plus_2 :
  ∀ (x : ℝ), (x ∈ Set.Icc (-9 : ℝ) 1) ↔ ((x + 2) ∈ Set.Icc (-7 : ℝ) 3) :=
sorry

end domain_f_x_plus_2_l1584_158456


namespace mental_math_quiz_l1584_158427

theorem mental_math_quiz : ∃ (q_i q_c : ℕ), q_c + q_i = 100 ∧ 10 * q_c - 5 * q_i = 850 ∧ q_i = 10 :=
by
  sorry

end mental_math_quiz_l1584_158427


namespace simplify_and_evaluate_l1584_158480

theorem simplify_and_evaluate : 
  ∀ (x y : ℚ), x = 1 / 2 → y = 2 / 3 →
  ((x - 2 * y)^2 + (x - 2 * y) * (x + 2 * y) - 3 * x * (2 * x - y)) / (2 * x) = -4 / 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l1584_158480


namespace motorboat_time_to_C_l1584_158471

variables (r s p t_B : ℝ)

-- Condition declarations
def kayak_speed := r + s
def motorboat_speed := p
def meeting_time := 12

-- Problem statement: to prove the time it took for the motorboat to reach dock C before turning back
theorem motorboat_time_to_C :
  (2 * r + s) * t_B = r * 12 + s * 6 → t_B = (r * 12 + s * 6) / (2 * r + s) := 
by
  intros h
  sorry

end motorboat_time_to_C_l1584_158471


namespace bridge_length_is_correct_l1584_158424

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_covered := speed_mps * time
  distance_covered - train_length

theorem bridge_length_is_correct :
  length_of_bridge 100 16.665333439991468 54 = 149.97999909987152 :=
by sorry

end bridge_length_is_correct_l1584_158424


namespace smallest_positive_integer_congruence_l1584_158410

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 18 [MOD 31] ∧ 0 < x ∧ x < 31 ∧ x = 16 := 
by sorry

end smallest_positive_integer_congruence_l1584_158410


namespace smallest_base10_integer_l1584_158408

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l1584_158408


namespace fraction_meaningful_range_l1584_158444

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l1584_158444


namespace lcm_of_36_and_105_l1584_158491

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l1584_158491


namespace number_of_companion_relation_subsets_l1584_158401

def isCompanionRelationSet (A : Set ℚ) : Prop :=
  ∀ x ∈ A, (x ≠ 0 → (1 / x) ∈ A)

def M : Set ℚ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem number_of_companion_relation_subsets :
  ∃ n, n = 15 ∧
  (∀ A ⊆ M, isCompanionRelationSet A) :=
sorry

end number_of_companion_relation_subsets_l1584_158401


namespace intersection_l1584_158415

noncomputable def M : Set ℝ := { x : ℝ | Real.sqrt (x + 1) ≥ 0 }
noncomputable def N : Set ℝ := { x : ℝ | x^2 + x - 2 < 0 }

theorem intersection (x : ℝ) : x ∈ (M ∩ N) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end intersection_l1584_158415


namespace gcd_176_88_l1584_158468

theorem gcd_176_88 : Nat.gcd 176 88 = 88 :=
by
  sorry

end gcd_176_88_l1584_158468


namespace average_length_of_strings_l1584_158446

theorem average_length_of_strings {l1 l2 l3 : ℝ} (h1 : l1 = 2) (h2 : l2 = 6) (h3 : l3 = 9) : 
  (l1 + l2 + l3) / 3 = 17 / 3 :=
by
  sorry

end average_length_of_strings_l1584_158446


namespace ratio_of_a_to_b_l1584_158404

theorem ratio_of_a_to_b (a y b : ℝ) (h1 : a = 0) (h2 : b = 2 * y) : a / b = 0 :=
by
  sorry

end ratio_of_a_to_b_l1584_158404


namespace choose_4_from_15_l1584_158486

theorem choose_4_from_15 : (Nat.choose 15 4) = 1365 :=
by
  sorry

end choose_4_from_15_l1584_158486


namespace last_four_digits_of_7_pow_5000_l1584_158418

theorem last_four_digits_of_7_pow_5000 (h : 7 ^ 250 ≡ 1 [MOD 1250]) : 7 ^ 5000 ≡ 1 [MOD 1250] :=
by
  -- Proof (will be omitted)
  sorry

end last_four_digits_of_7_pow_5000_l1584_158418


namespace loss_percentage_is_17_l1584_158439

noncomputable def loss_percentage (CP SP : ℝ) := ((CP - SP) / CP) * 100

theorem loss_percentage_is_17 :
  let CP : ℝ := 1500
  let SP : ℝ := 1245
  loss_percentage CP SP = 17 :=
by
  sorry

end loss_percentage_is_17_l1584_158439


namespace hens_count_l1584_158403

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 :=
by
  sorry

end hens_count_l1584_158403


namespace students_6_to_8_hours_study_l1584_158481

-- Condition: 100 students were surveyed
def total_students : ℕ := 100

-- Hypothetical function representing the number of students studying for a specific range of hours based on the histogram
def histogram_students (lower_bound upper_bound : ℕ) : ℕ :=
  sorry  -- this would be defined based on actual histogram data

-- Question: Prove the number of students who studied for 6 to 8 hours
theorem students_6_to_8_hours_study : histogram_students 6 8 = 30 :=
  sorry -- the expected answer based on the histogram data

end students_6_to_8_hours_study_l1584_158481


namespace ratio_of_larger_to_smaller_l1584_158498

theorem ratio_of_larger_to_smaller 
    (x y : ℝ) 
    (hx : x > 0) 
    (hy : y > 0) 
    (h : x + y = 7 * (x - y)) : 
    x / y = 4 / 3 := 
by 
    sorry

end ratio_of_larger_to_smaller_l1584_158498


namespace tan_theta_minus_pi_over_4_l1584_158442

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over_4_l1584_158442


namespace garden_ratio_l1584_158455

theorem garden_ratio (L W : ℝ) (h1 : 2 * L + 2 * W = 180) (h2 : L = 60) : L / W = 2 :=
by
  -- this is where you would put the proof
  sorry

end garden_ratio_l1584_158455


namespace blueberry_pies_correct_l1584_158400

def total_pies := 36
def apple_pie_ratio := 3
def blueberry_pie_ratio := 4
def cherry_pie_ratio := 5

-- Total parts in the ratio
def total_ratio_parts := apple_pie_ratio + blueberry_pie_ratio + cherry_pie_ratio

-- Number of pies per part
noncomputable def pies_per_part := total_pies / total_ratio_parts

-- Number of blueberry pies
noncomputable def blueberry_pies := blueberry_pie_ratio * pies_per_part

theorem blueberry_pies_correct : blueberry_pies = 12 := 
by
  sorry

end blueberry_pies_correct_l1584_158400


namespace v_not_closed_under_operations_l1584_158451

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def v : Set ℕ := {n | ∃ m : ℕ, n = m * m}

def addition_followed_by_multiplication (a b : ℕ) : ℕ :=
  (a + b) * a

def multiplication_followed_by_addition (a b : ℕ) : ℕ :=
  (a * b) + a

def division_followed_by_subtraction (a b : ℕ) : ℕ :=
  if b ≠ 0 then (a / b) - b else 0

def extraction_root_followed_by_multiplication (a b : ℕ) : ℕ :=
  (Nat.sqrt a) * (Nat.sqrt b)

theorem v_not_closed_under_operations : 
  ¬ (∀ a ∈ v, ∀ b ∈ v, addition_followed_by_multiplication a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, multiplication_followed_by_addition a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, division_followed_by_subtraction a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, extraction_root_followed_by_multiplication a b ∈ v) :=
sorry

end v_not_closed_under_operations_l1584_158451


namespace test_group_type_A_probability_atleast_one_type_A_group_probability_l1584_158417

noncomputable def probability_type_A_group : ℝ :=
  let pA := 2 / 3
  let pB := 1 / 2
  let P_A1 := 2 * (1 - pA) * pA
  let P_A2 := pA * pA
  let P_B0 := (1 - pB) * (1 - pB)
  let P_B1 := 2 * (1 - pB) * pB
  P_B0 * P_A1 + P_B0 * P_A2 + P_B1 * P_A2

theorem test_group_type_A_probability :
  probability_type_A_group = 4 / 9 :=
by
  sorry

noncomputable def at_least_one_type_A_in_3_groups : ℝ :=
  let P_type_A_group := 4 / 9
  1 - (1 - P_type_A_group) ^ 3

theorem atleast_one_type_A_group_probability :
  at_least_one_type_A_in_3_groups = 604 / 729 :=
by
  sorry

end test_group_type_A_probability_atleast_one_type_A_group_probability_l1584_158417


namespace permutation_probability_l1584_158437

theorem permutation_probability (total_digits: ℕ) (zeros: ℕ) (ones: ℕ) 
  (total_permutations: ℕ) (favorable_permutations: ℕ) (probability: ℚ)
  (h1: total_digits = 6) 
  (h2: zeros = 2) 
  (h3: ones = 4) 
  (h4: total_permutations = 2 ^ total_digits) 
  (h5: favorable_permutations = Nat.choose total_digits zeros) 
  (h6: probability = favorable_permutations / total_permutations) : 
  probability = 15 / 64 := 
sorry

end permutation_probability_l1584_158437


namespace at_most_one_solution_l1584_158433

theorem at_most_one_solution (a b c : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (hcpos : 0 < c) :
  ∃! x : ℝ, a * x + b * ⌊x⌋ - c = 0 :=
sorry

end at_most_one_solution_l1584_158433


namespace factorize_xy_squared_minus_x_l1584_158476

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l1584_158476
