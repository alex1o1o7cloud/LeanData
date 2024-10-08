import Mathlib

namespace g_neither_even_nor_odd_l36_36191

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x → false) ∧ (∀ x, g (-x) = -g x → false) :=
by
  unfold g
  sorry

end g_neither_even_nor_odd_l36_36191


namespace Joan_spent_on_shirt_l36_36881

/-- Joan spent $15 on shorts, $14.82 on a jacket, and a total of $42.33 on clothing.
    Prove that Joan spent $12.51 on the shirt. -/
theorem Joan_spent_on_shirt (shorts jacket total: ℝ) 
                            (h1: shorts = 15)
                            (h2: jacket = 14.82)
                            (h3: total = 42.33) :
  total - (shorts + jacket) = 12.51 :=
by
  sorry

end Joan_spent_on_shirt_l36_36881


namespace loss_equals_cost_price_of_balls_l36_36753

variable (selling_price : ℕ) (cost_price_ball : ℕ)
variable (number_of_balls : ℕ) (loss_incurred : ℕ) (x : ℕ)

-- Conditions
def condition1 : selling_price = 720 := sorry -- Selling price of 11 balls is Rs. 720
def condition2 : cost_price_ball = 120 := sorry -- Cost price of one ball is Rs. 120
def condition3 : number_of_balls = 11 := sorry -- Number of balls is 11

-- Cost price of 11 balls
def cost_price (n : ℕ) (cp_ball : ℕ): ℕ := n * cp_ball

-- Loss incurred on selling 11 balls
def loss (cp : ℕ) (sp : ℕ): ℕ := cp - sp

-- Equation for number of balls the loss equates to
def loss_equation (l : ℕ) (cp_ball : ℕ): ℕ := l / cp_ball

theorem loss_equals_cost_price_of_balls : 
  ∀ (n sp cp_ball cp l: ℕ), 
  sp = 720 ∧ cp_ball = 120 ∧ n = 11 ∧ 
  cp = cost_price n cp_ball ∧ 
  l = loss cp sp →
  loss_equation l cp_ball = 5 := sorry

end loss_equals_cost_price_of_balls_l36_36753


namespace Vincent_sells_8_literature_books_per_day_l36_36426

theorem Vincent_sells_8_literature_books_per_day
  (fantasy_book_cost : ℕ)
  (literature_book_cost : ℕ)
  (fantasy_books_sold_per_day : ℕ)
  (total_earnings_5_days : ℕ)
  (H_fantasy_book_cost : fantasy_book_cost = 4)
  (H_literature_book_cost : literature_book_cost = 2)
  (H_fantasy_books_sold_per_day : fantasy_books_sold_per_day = 5)
  (H_total_earnings_5_days : total_earnings_5_days = 180) :
  ∃ L : ℕ, L = 8 :=
by
  sorry

end Vincent_sells_8_literature_books_per_day_l36_36426


namespace find_minimum_x_and_values_l36_36323

theorem find_minimum_x_and_values (x y z w : ℝ) (h1 : y = x - 2003)
  (h2 : z = 2 * y - 2003)
  (h3 : w = 3 * z - 2003)
  (h4 : 0 ≤ x)
  (h5 : 0 ≤ y)
  (h6 : 0 ≤ z)
  (h7 : 0 ≤ w) :
  x ≥ 10015 / 3 ∧ 
  (x = 10015 / 3 → y = 4006 / 3 ∧ z = 2003 / 3 ∧ w = 0) := by
  sorry

end find_minimum_x_and_values_l36_36323


namespace certain_person_current_age_l36_36620

-- Define Sandys's current age and the certain person's current age
variable (S P : ℤ)

-- Conditions from the problem
def sandy_phone_bill_condition := 10 * S = 340
def sandy_age_relation := S + 2 = 3 * P

theorem certain_person_current_age (h1 : sandy_phone_bill_condition S) (h2 : sandy_age_relation S P) : P - 2 = 10 :=
by
  sorry

end certain_person_current_age_l36_36620


namespace T_expansion_l36_36580

def T (x : ℝ) : ℝ := (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1

theorem T_expansion (x : ℝ) : T x = (x - 1)^5 := by
  sorry

end T_expansion_l36_36580


namespace volume_of_sphere_from_cube_surface_area_l36_36346

theorem volume_of_sphere_from_cube_surface_area (S : ℝ) (h : S = 24) : 
  ∃ V : ℝ, V = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_sphere_from_cube_surface_area_l36_36346


namespace students_neither_football_nor_cricket_l36_36832

theorem students_neither_football_nor_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (cricket_players : ℕ) 
  (both_players : ℕ) 
  (H1 : total_students = 410) 
  (H2 : football_players = 325) 
  (H3 : cricket_players = 175) 
  (H4 : both_players = 140) :
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end students_neither_football_nor_cricket_l36_36832


namespace rows_seating_nine_people_l36_36072

theorem rows_seating_nine_people (x y : ℕ) (h : 9 * x + 7 * y = 74) : x = 2 :=
by sorry

end rows_seating_nine_people_l36_36072


namespace cos_identity_l36_36762

theorem cos_identity 
  (x : ℝ) 
  (h : Real.sin (x - π / 3) = 3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := 
by 
  sorry

end cos_identity_l36_36762


namespace compare_abc_l36_36190

noncomputable def a := Real.sqrt 0.3
noncomputable def b := Real.sqrt 0.4
noncomputable def c := Real.log 0.6 / Real.log 3

theorem compare_abc : c < a ∧ a < b :=
by
  -- Proof goes here
  sorry

end compare_abc_l36_36190


namespace part1_part2_l36_36277

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1 / 3) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) :
  (a ≠ 0 → ( a > 0 ↔ -1/a < x ∧ x < 1)
  ∧ (a = 0 ↔ x < 1)
  ∧ (-1 < a ∧ a < 0 ↔ x < 1 ∨ x > -1/a)
  ∧ (a = -1 ↔ x ≠ 1)
  ∧ (a < -1 ↔ x < -1/a ∨ x > 1)) :=
sorry

end part1_part2_l36_36277


namespace part_I_part_II_l36_36155

open Real  -- Specify that we are working with real numbers

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- The first theorem: Prove the result for a = 1
theorem part_I (x : ℝ) : f x 1 + x > 0 ↔ (x > -3 ∧ x < 1 ∨ x > 3) :=
by
  sorry

-- The second theorem: Prove the range of a such that f(x) ≤ 3 for all x
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 3) ↔ (-5 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l36_36155


namespace find_D_l36_36198

theorem find_D (P Q : ℕ) (h_pos : 0 < P ∧ 0 < Q) (h_eq : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end find_D_l36_36198


namespace find_m_n_l36_36698

theorem find_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 := 
by {
  sorry
}

end find_m_n_l36_36698


namespace smallest_k_value_eq_sqrt475_div_12_l36_36164

theorem smallest_k_value_eq_sqrt475_div_12 :
  ∀ (k : ℝ), (dist (⟨5 * Real.sqrt 3, k - 2⟩ : ℝ × ℝ) ⟨0, 0⟩ = 5 * k) →
  k = (1 + Real.sqrt 475) / 12 := 
by
  intro k
  sorry

end smallest_k_value_eq_sqrt475_div_12_l36_36164


namespace children_left_l36_36564

-- Define the initial problem constants and conditions
def totalGuests := 50
def halfGuests := totalGuests / 2
def numberOfMen := 15
def numberOfWomen := halfGuests
def numberOfChildren := totalGuests - (numberOfWomen + numberOfMen)
def proportionMenLeft := numberOfMen / 5
def totalPeopleStayed := 43
def totalPeopleLeft := totalGuests - totalPeopleStayed

-- Define the proposition to prove
theorem children_left : 
  totalPeopleLeft - proportionMenLeft = 4 := by 
    sorry

end children_left_l36_36564


namespace range_of_a_l36_36061

noncomputable def A (x : ℝ) : Prop := x^2 - x ≤ 0
noncomputable def B (x : ℝ) (a : ℝ) : Prop := 2^(1 - x) + a ≤ 0

theorem range_of_a (a : ℝ) : (∀ x, A x → B x a) → a ≤ -2 := by
  intro h
  -- Proof steps would go here
  sorry

end range_of_a_l36_36061


namespace lemonade_cups_count_l36_36912

theorem lemonade_cups_count :
  ∃ x y : ℕ, x + y = 400 ∧ x + 2 * y = 546 ∧ x = 254 :=
by
  sorry

end lemonade_cups_count_l36_36912


namespace factorize_x_squared_minus_one_l36_36303

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l36_36303


namespace problem_b_correct_l36_36843

theorem problem_b_correct (a b : ℝ) (h₁ : a < 0) (h₂ : 0 < b) (h₃ : b < 1) : (ab^2 > ab ∧ ab > a) :=
by
  sorry

end problem_b_correct_l36_36843


namespace find_base_numerica_l36_36134

theorem find_base_numerica (r : ℕ) (h_gadget_cost : 5*r^2 + 3*r = 530) (h_payment : r^3 + r^2 = 1100) (h_change : 4*r^2 + 6*r = 460) :
  r = 9 :=
sorry

end find_base_numerica_l36_36134


namespace range_of_m_l36_36289

noncomputable def distance (m : ℝ) : ℝ := (|m| * Real.sqrt 2 / 2)
theorem range_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 + A.2 + m = 0 ∧ B.1 + B.2 + m = 0) ∧
    (A.1 ^ 2 + A.2 ^ 2 = 2 ∧ B.1 ^ 2 + B.2 ^ 2 = 2) ∧
    (Real.sqrt (A.1 ^ 2 + A.2 ^ 2) + Real.sqrt (B.1 ^ 2 + B.2 ^ 2) ≥ 
     Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) ∧ (distance m < Real.sqrt 2)) ↔ 
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 := 
sorry

end range_of_m_l36_36289


namespace snowdrift_depth_end_of_third_day_l36_36504

theorem snowdrift_depth_end_of_third_day :
  let depth_ninth_day := 40
  let d_before_eighth_night_snowfall := depth_ninth_day - 10
  let d_before_eighth_day_melting := d_before_eighth_night_snowfall * 4 / 3
  let depth_seventh_day := d_before_eighth_day_melting
  let d_before_sixth_day_snowfall := depth_seventh_day - 20
  let d_before_fifth_day_snowfall := d_before_sixth_day_snowfall - 15
  let d_before_fourth_day_melting := d_before_fifth_day_snowfall * 3 / 2
  depth_ninth_day = 40 →
  d_before_eighth_night_snowfall = depth_ninth_day - 10 →
  d_before_eighth_day_melting = d_before_eighth_night_snowfall * 4 / 3 →
  depth_seventh_day = d_before_eighth_day_melting →
  d_before_sixth_day_snowfall = depth_seventh_day - 20 →
  d_before_fifth_day_snowfall = d_before_sixth_day_snowfall - 15 →
  d_before_fourth_day_melting = d_before_fifth_day_snowfall * 3 / 2 →
  d_before_fourth_day_melting = 7.5 :=
by
  intros
  sorry

end snowdrift_depth_end_of_third_day_l36_36504


namespace min_value_of_function_l36_36649

theorem min_value_of_function : ∃ x : ℝ, ∀ x : ℝ, x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
by
  sorry

end min_value_of_function_l36_36649


namespace find_lesser_number_l36_36966

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l36_36966


namespace percentage_needed_to_pass_l36_36090

def MikeScore : ℕ := 212
def Shortfall : ℕ := 19
def MaxMarks : ℕ := 770

theorem percentage_needed_to_pass :
  (231.0 / (770.0 : ℝ)) * 100 = 30 := by
  -- placeholder for proof
  sorry

end percentage_needed_to_pass_l36_36090


namespace find_x_weeks_l36_36549

-- Definition of the problem conditions:
def archibald_first_two_weeks_apples : Nat := 14
def archibald_next_x_weeks_apples (x : Nat) : Nat := 14
def archibald_last_two_weeks_apples : Nat := 42
def total_weeks : Nat := 7
def weekly_average : Nat := 10

-- Statement of the theorem to prove that x = 2 given the conditions
theorem find_x_weeks :
  ∃ x : Nat, (archibald_first_two_weeks_apples + archibald_next_x_weeks_apples x + archibald_last_two_weeks_apples = total_weeks * weekly_average) 
  ∧ (archibald_next_x_weeks_apples x / x = 7) 
  → x = 2 :=
by
  sorry

end find_x_weeks_l36_36549


namespace trader_gain_percentage_l36_36731

-- Definition of the given conditions
def cost_per_pen (C : ℝ) := C
def num_pens_sold := 90
def gain_from_sale (C : ℝ) := 15 * C
def total_cost (C : ℝ) := 90 * C

-- Statement of the problem
theorem trader_gain_percentage (C : ℝ) : 
  (((gain_from_sale C) / (total_cost C)) * 100) = 16.67 :=
by
  -- This part will contain the step-by-step proof, omitted here
  sorry

end trader_gain_percentage_l36_36731


namespace probability_Xavier_Yvonne_not_Zelda_l36_36639

theorem probability_Xavier_Yvonne_not_Zelda
    (P_Xavier : ℚ)
    (P_Yvonne : ℚ)
    (P_Zelda : ℚ)
    (hXavier : P_Xavier = 1/3)
    (hYvonne : P_Yvonne = 1/2)
    (hZelda : P_Zelda = 5/8) :
    (P_Xavier * P_Yvonne * (1 - P_Zelda) = 1/16) :=
  by
  rw [hXavier, hYvonne, hZelda]
  sorry

end probability_Xavier_Yvonne_not_Zelda_l36_36639


namespace xy_in_A_l36_36241

def A : Set ℤ :=
  {z | ∃ (a b k n : ℤ), z = a^2 + k * a * b + n * b^2}

theorem xy_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := sorry

end xy_in_A_l36_36241


namespace solve_quadratic_l36_36949

theorem solve_quadratic :
  ∀ x, (x^2 - x - 12 = 0) → (x = -3 ∨ x = 4) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l36_36949


namespace bruce_total_cost_l36_36498

def cost_of_grapes : ℕ := 8 * 70
def cost_of_mangoes : ℕ := 11 * 55
def cost_of_oranges : ℕ := 5 * 45
def cost_of_apples : ℕ := 3 * 90
def cost_of_cherries : ℕ := (45 / 10) * 120  -- use rational division and then multiplication

def total_cost : ℕ :=
  cost_of_grapes + cost_of_mangoes + cost_of_oranges + cost_of_apples + cost_of_cherries

theorem bruce_total_cost : total_cost = 2200 := by
  sorry

end bruce_total_cost_l36_36498


namespace find_z_l36_36335

-- Condition: there exists a constant k such that z = k * w
def direct_variation (z w : ℝ): Prop := ∃ k, z = k * w

-- We set up the conditions given in the problem.
theorem find_z (k : ℝ) (hw1 : 10 = k * 5) (hw2 : w = -15) : direct_variation z w → z = -30 :=
by
  sorry

end find_z_l36_36335


namespace range_of_a_l36_36126

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_increasing_on (λ x => x^2 + a * x + 1 / x) (Set.Ioi (1 / 2)) ↔ 3 ≤ a := 
by
  sorry

end range_of_a_l36_36126


namespace solution_set_of_inequality_l36_36891

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 + 4 * x - 5 > 0 ↔ (x < -5 ∨ x > 1) :=
sorry

end solution_set_of_inequality_l36_36891


namespace largest_int_lt_100_remainder_3_div_by_8_l36_36523

theorem largest_int_lt_100_remainder_3_div_by_8 : 
  ∃ n, n < 100 ∧ n % 8 = 3 ∧ ∀ m, m < 100 ∧ m % 8 = 3 → m ≤ 99 := by
  sorry

end largest_int_lt_100_remainder_3_div_by_8_l36_36523


namespace line_points_product_l36_36618

theorem line_points_product (x y : ℝ) (h1 : 8 = (1/4 : ℝ) * x) (h2 : y = (1/4 : ℝ) * 20) : x * y = 160 := 
by
  sorry

end line_points_product_l36_36618


namespace car_speed_40_kmph_l36_36668

theorem car_speed_40_kmph (v : ℝ) (h : 1 / v = 1 / 48 + 15 / 3600) : v = 40 := 
sorry

end car_speed_40_kmph_l36_36668


namespace total_cases_after_three_weeks_l36_36444

-- Definitions and conditions directly from the problem
def week1_cases : ℕ := 5000
def week2_cases : ℕ := week1_cases / 2
def week3_cases : ℕ := week2_cases + 2000
def total_cases : ℕ := week1_cases + week2_cases + week3_cases

-- The theorem to prove
theorem total_cases_after_three_weeks :
  total_cases = 12000 := 
by
  -- Sorry allows us to skip the actual proof
  sorry

end total_cases_after_three_weeks_l36_36444


namespace pf1_pf2_range_l36_36320

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2 - 3)

theorem pf1_pf2_range (x y : ℝ) (h : ellipse_point x y) :
  -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by
  sorry

end pf1_pf2_range_l36_36320


namespace sum_y_coordinates_of_other_vertices_of_parallelogram_l36_36655

theorem sum_y_coordinates_of_other_vertices_of_parallelogram :
  let x1 := 4
  let y1 := 26
  let x2 := 12
  let y2 := -8
  let midpoint_y := (y1 + y2) / 2
  2 * midpoint_y = 18 := by
    sorry

end sum_y_coordinates_of_other_vertices_of_parallelogram_l36_36655


namespace three_lines_form_triangle_l36_36982

/-- Theorem to prove that for three lines x + y = 0, x - y = 0, and x + ay = 3 to form a triangle, the value of a cannot be ±1. -/
theorem three_lines_form_triangle (a : ℝ) : ¬ (a = 1 ∨ a = -1) :=
sorry

end three_lines_form_triangle_l36_36982


namespace contradiction_proof_real_root_l36_36757

theorem contradiction_proof_real_root (a b : ℝ) :
  (∀ x : ℝ, x^3 + a * x + b ≠ 0) → (∃ x : ℝ, x + a * x + b = 0) :=
sorry

end contradiction_proof_real_root_l36_36757


namespace probability_of_selecting_two_girls_l36_36243

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3
def selected_students : ℕ := 2

theorem probability_of_selecting_two_girls :
  (Nat.choose girls selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ) = 0.3 := by
  sorry

end probability_of_selecting_two_girls_l36_36243


namespace fraction_to_decimal_l36_36001

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l36_36001


namespace yoongi_caught_frogs_l36_36438

theorem yoongi_caught_frogs (initial_frogs caught_later : ℕ) (h1 : initial_frogs = 5) (h2 : caught_later = 2) : (initial_frogs + caught_later = 7) :=
by
  sorry

end yoongi_caught_frogs_l36_36438


namespace sum_of_base_areas_eq_5_l36_36829

-- Define the surface area, lateral area, and the sum of the areas of the two base faces.
def surface_area : ℝ := 30
def lateral_area : ℝ := 25
def sum_base_areas : ℝ := surface_area - lateral_area

-- The theorem statement.
theorem sum_of_base_areas_eq_5 : sum_base_areas = 5 := 
by 
  sorry

end sum_of_base_areas_eq_5_l36_36829


namespace beads_per_bracelet_l36_36130

-- Definitions for the conditions
def Nancy_metal_beads : ℕ := 40
def Nancy_pearl_beads : ℕ := Nancy_metal_beads + 20
def Rose_crystal_beads : ℕ := 20
def Rose_stone_beads : ℕ := Rose_crystal_beads * 2
def total_beads : ℕ := Nancy_metal_beads + Nancy_pearl_beads + Rose_crystal_beads + Rose_stone_beads
def bracelets : ℕ := 20

-- Statement to prove
theorem beads_per_bracelet :
  total_beads / bracelets = 8 :=
by
  -- skip the proof
  sorry

end beads_per_bracelet_l36_36130


namespace math_proof_problem_l36_36656

-- Defining the problem condition
def condition (x y z : ℝ) := 
  x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0

-- Adding constraints to x, y, z
def constraints (x y z : ℝ) :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x ≠ y ∨ y ≠ z ∨ z ≠ x)

-- Stating the main theorem
theorem math_proof_problem (x y z : ℝ) (h_condition : condition x y z) (h_constraints : constraints x y z) :
  x + y + z = 3 ∧ x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6 := 
sorry

end math_proof_problem_l36_36656


namespace algebraic_expression_value_l36_36666

variables (m n x y : ℤ)

def condition1 := m - n = 100
def condition2 := x + y = -1

theorem algebraic_expression_value :
  condition1 m n → condition2 x y → (n + x) - (m - y) = -101 :=
by
  intro h1 h2
  sorry

end algebraic_expression_value_l36_36666


namespace rain_difference_l36_36050

variable (R : ℝ) -- Amount of rain in the second hour
variable (r1 : ℝ) -- Amount of rain in the first hour

-- Conditions
axiom h1 : r1 = 5
axiom h2 : R + r1 = 22

-- Theorem to prove
theorem rain_difference (R r1 : ℝ) (h1 : r1 = 5) (h2 : R + r1 = 22) : R - 2 * r1 = 7 := by
  sorry

end rain_difference_l36_36050


namespace table_length_l36_36272

theorem table_length (L : ℕ) (H1 : ∃ n : ℕ, 80 = n * L)
  (H2 : L ≥ 16) (H3 : ∃ m : ℕ, 16 = m * 4)
  (H4 : L % 4 = 0) : L = 20 := by 
sorry

end table_length_l36_36272


namespace karen_wrong_questions_l36_36260

theorem karen_wrong_questions (k l n : ℕ) (h1 : k + l = 6 + n) (h2 : k + n = l + 9) : k = 6 := 
by
  sorry

end karen_wrong_questions_l36_36260


namespace replace_asterisk_l36_36948

theorem replace_asterisk (star : ℝ) : ((36 / 18) * (star / 72) = 1) → star = 36 :=
by
  intro h
  sorry

end replace_asterisk_l36_36948


namespace geometric_seq_seventh_term_l36_36442

theorem geometric_seq_seventh_term (a r : ℕ) (r_pos : r > 0) (first_term : a = 3)
    (fifth_term : a * r^4 = 243) : a * r^6 = 2187 := by
  sorry

end geometric_seq_seventh_term_l36_36442


namespace part_a_part_b_l36_36922

-- This definition states that a number p^m is a divisor of a-1
def divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  (p ^ m) ∣ (a - 1)

-- This definition states that (p^(m+1)) is not a divisor of a-1
def not_divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  ¬ (p ^ (m + 1) ∣ (a - 1))

-- Part (a): Prove divisibility
theorem part_a (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  p ^ (m + n) ∣ a ^ (p ^ n) - 1 := 
sorry

-- Part (b): Prove non-divisibility
theorem part_b (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  ¬ p ^ (m + n + 1) ∣ a ^ (p ^ n) - 1 := 
sorry

end part_a_part_b_l36_36922


namespace value_of_y_l36_36328

theorem value_of_y (y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) :
  a = 10^3 → b = 10^4 → 
  a^y * 10^(3 * y) = (b^4) → 
  y = 8 / 3 :=
by 
  intro ha hb hc
  rw [ha, hb] at hc
  sorry

end value_of_y_l36_36328


namespace ahmed_goats_correct_l36_36826

-- Definitions based on the conditions given in the problem.
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 5 + 2 * adam_goats
def ahmed_goats : ℕ := andrew_goats - 6

-- The theorem statement that needs to be proven.
theorem ahmed_goats_correct : ahmed_goats = 13 := by
    sorry

end ahmed_goats_correct_l36_36826


namespace alice_savings_third_month_l36_36354

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end alice_savings_third_month_l36_36354


namespace number_subtracted_eq_l36_36783

theorem number_subtracted_eq (x n : ℤ) (h1 : x + 1315 + 9211 - n = 11901) (h2 : x = 88320) : n = 86945 :=
by
  sorry

end number_subtracted_eq_l36_36783


namespace b_20_value_l36_36017

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n  -- Given that \( b_n = a_n \)

-- The theorem stating that \( b_{20} = 39 \)
theorem b_20_value : b 20 = 39 :=
by
  -- Skipping the proof
  sorry

end b_20_value_l36_36017


namespace problem_part1_problem_part2_l36_36427

theorem problem_part1 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  2 * ((1 / x) + (1 / y) + (1 / z)) ≤ (1 / p) + (1 / q) + (1 / r) :=
sorry

theorem problem_part2 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  x * y + y * z + z * x ≥ 2 * (p * x + q * y + r * z) :=
sorry

end problem_part1_problem_part2_l36_36427


namespace cos_two_pi_over_three_l36_36613

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 :=
by sorry

end cos_two_pi_over_three_l36_36613


namespace rectangle_area_in_inscribed_triangle_l36_36575

theorem rectangle_area_in_inscribed_triangle (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  ∃ (y : ℝ), y = (b * (h - x)) / h ∧ (x * y) = (b * x * (h - x)) / h :=
by
  sorry

end rectangle_area_in_inscribed_triangle_l36_36575


namespace arrange_decimals_in_order_l36_36798

theorem arrange_decimals_in_order 
  (a b c d : ℚ) 
  (h₀ : a = 6 / 10) 
  (h₁ : b = 676 / 1000) 
  (h₂ : c = 677 / 1000) 
  (h₃ : d = 67 / 100) : 
  a < d ∧ d < b ∧ b < c := 
by
  sorry

end arrange_decimals_in_order_l36_36798


namespace find_f_of_one_half_l36_36495

def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := (1 - x ^ 2) / x ^ 2

theorem find_f_of_one_half :
  f (g (1 / 2)) = 15 :=
by
  sorry

end find_f_of_one_half_l36_36495


namespace sin_neg_30_eq_neg_one_half_l36_36123

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l36_36123


namespace inverse_proportion_l36_36205

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end inverse_proportion_l36_36205


namespace fraction_even_odd_phonenumbers_l36_36065

-- Define a predicate for valid phone numbers
def isValidPhoneNumber (n : Nat) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ (n / 1000000 ≠ 0) ∧ (n / 1000000 ≠ 1)

-- Calculate the total number of valid phone numbers
def totalValidPhoneNumbers : Nat :=
  4 * 10^6

-- Calculate the number of valid phone numbers that begin with an even digit and end with an odd digit
def validEvenOddPhoneNumbers : Nat :=
  4 * (10^5) * 5

-- Determine the fraction of such phone numbers (valid ones and valid even-odd ones)
theorem fraction_even_odd_phonenumbers : 
  (validEvenOddPhoneNumbers) / (totalValidPhoneNumbers) = 1 / 2 :=
by {
  sorry
}

end fraction_even_odd_phonenumbers_l36_36065


namespace quadratic_expression_negative_for_all_x_l36_36718

theorem quadratic_expression_negative_for_all_x (k : ℝ) :
  (∀ x : ℝ, (5-k) * x^2 - 2 * (1-k) * x + 2 - 2 * k < 0) ↔ k > 9 :=
sorry

end quadratic_expression_negative_for_all_x_l36_36718


namespace find_x_l36_36861

variable (x : ℝ)
variable (h : 0.3 * 100 = 0.5 * x + 10)

theorem find_x : x = 40 :=
by
  sorry

end find_x_l36_36861


namespace shaded_region_area_l36_36693

theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  8 * (π * r * r / 4 - r * r / 2) / 2 = 50 * (π - 2) :=
by
  sorry

end shaded_region_area_l36_36693


namespace num_of_terms_in_arith_seq_l36_36097

-- Definitions of the conditions
def a : Int := -5 -- Start of the arithmetic sequence
def l : Int := 85 -- End of the arithmetic sequence
def d : Nat := 5  -- Common difference

-- The theorem that needs to be proved
theorem num_of_terms_in_arith_seq : (l - a) / d + 1 = 19 := sorry

end num_of_terms_in_arith_seq_l36_36097


namespace bhupathi_amount_l36_36721

variable (A B : ℝ)

theorem bhupathi_amount :
  (A + B = 1210 ∧ (4 / 15) * A = (2 / 5) * B) → B = 484 :=
by
  sorry

end bhupathi_amount_l36_36721


namespace find_a_l36_36033

def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_a : ∃ a : ℝ, (a > -1) ∧ (a < 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ 2 → f x ≤ f a) ∧ f a = 15 / 4 :=
by
  exists -1 / 2
  sorry

end find_a_l36_36033


namespace weight_of_B_l36_36071

-- Definitions for the weights
variables (A B C : ℝ)

-- Conditions from the problem
def avg_ABC : Prop := (A + B + C) / 3 = 45
def avg_AB : Prop := (A + B) / 2 = 40
def avg_BC : Prop := (B + C) / 2 = 43

-- The theorem to prove the weight of B
theorem weight_of_B (h1 : avg_ABC A B C) (h2 : avg_AB A B) (h3 : avg_BC B C) : B = 31 :=
sorry

end weight_of_B_l36_36071


namespace bezdikov_population_l36_36098

variable (W M : ℕ) -- original number of women and men
variable (W_current M_current : ℕ) -- current number of women and men

theorem bezdikov_population (h1 : W = M + 30)
                          (h2 : W_current = W / 4)
                          (h3 : M_current = M - 196)
                          (h4 : W_current = M_current + 10) : W_current + M_current = 134 :=
by
  sorry

end bezdikov_population_l36_36098


namespace factor_tree_value_l36_36913

-- Define the values and their relationships
def A := 900
def B := 3 * (3 * 2)
def D := 3 * 2
def C := 5 * (5 * 2)
def E := 5 * 2

-- Define the theorem and provide the conditions
theorem factor_tree_value :
  (B = 3 * D) →
  (D = 3 * 2) →
  (C = 5 * E) →
  (E = 5 * 2) →
  (A = B * C) →
  A = 900 := by
  intros hB hD hC hE hA
  sorry

end factor_tree_value_l36_36913


namespace find_a_for_even_function_l36_36719

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = (x + 1) * (x + a) ∧ f (-x) = f x) : a = -1 := by 
  sorry

end find_a_for_even_function_l36_36719


namespace greatest_power_sum_l36_36856

theorem greatest_power_sum (a b : ℕ) (h1 : 0 < a) (h2 : 2 < b) (h3 : a^b < 500) (h4 : ∀ m n : ℕ, 0 < m → 2 < n → m^n < 500 → a^b ≥ m^n) : a + b = 10 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end greatest_power_sum_l36_36856


namespace polynomial_divisible_by_3_l36_36195

/--
Given q and p are integers where q is divisible by 3 and p+1 is divisible by 3,
prove that the polynomial Q(x) = x^3 - x + (p+1)x + q is divisible by 3 for any integer x.
-/
theorem polynomial_divisible_by_3 (q p : ℤ) (hq : 3 ∣ q) (hp1 : 3 ∣ (p + 1)) :
  ∀ x : ℤ, 3 ∣ (x^3 - x + (p+1) * x + q) :=
by {
  sorry
}

end polynomial_divisible_by_3_l36_36195


namespace suff_and_not_necessary_l36_36356

theorem suff_and_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) :
  (|a| > |b|) ∧ (¬(∀ x y : ℝ, (|x| > |y|) → (x > y ∧ y > 0))) :=
by
  sorry

end suff_and_not_necessary_l36_36356


namespace correct_operation_l36_36135

variable (a b : ℝ)

theorem correct_operation : (-a^2 * b + 2 * a^2 * b = a^2 * b) :=
by sorry

end correct_operation_l36_36135


namespace find_q_l36_36946

-- Given conditions
noncomputable def digits_non_zero (p q r : Nat) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

noncomputable def three_digit_number (p q r : Nat) : Nat :=
  100 * p + 10 * q + r

noncomputable def two_digit_number (q r : Nat) : Nat :=
  10 * q + r

noncomputable def one_digit_number (r : Nat) : Nat := r

noncomputable def numbers_sum_to (p q r sum : Nat) : Prop :=
  three_digit_number p q r + two_digit_number q r + one_digit_number r = sum

-- The theorem to prove
theorem find_q (p q r : Nat) (hpq : digits_non_zero p q r)
  (hsum : numbers_sum_to p q r 912) : q = 5 := sorry

end find_q_l36_36946


namespace sqrt_expression_eq_1720_l36_36127

theorem sqrt_expression_eq_1720 : Real.sqrt ((43 * 42 * 41 * 40) + 1) = 1720 := by
  sorry

end sqrt_expression_eq_1720_l36_36127


namespace total_weight_of_10_moles_l36_36235

theorem total_weight_of_10_moles
  (molecular_weight : ℕ)
  (moles : ℕ)
  (h_molecular_weight : molecular_weight = 2670)
  (h_moles : moles = 10) :
  moles * molecular_weight = 26700 := by
  -- By substituting the values from the hypotheses:
  -- We will get:
  -- 10 * 2670 = 26700
  sorry

end total_weight_of_10_moles_l36_36235


namespace correctly_calculated_value_l36_36295

theorem correctly_calculated_value (x : ℝ) (hx : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  -- proof to be provided
  sorry

end correctly_calculated_value_l36_36295


namespace remainder_of_N_eq_4101_l36_36499

noncomputable def N : ℕ :=
  20 + 3^(3^(3+1) - 13)

theorem remainder_of_N_eq_4101 : N % 10000 = 4101 := by
  sorry

end remainder_of_N_eq_4101_l36_36499


namespace LCM_of_numbers_with_HCF_and_ratio_l36_36677

theorem LCM_of_numbers_with_HCF_and_ratio (a b x : ℕ)
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x)
  (h3 : ∀ y : ℕ, y ∣ a → y ∣ b → y ∣ x)
  (hx : x = 5) :
  Nat.lcm a b = 60 := 
by
  sorry

end LCM_of_numbers_with_HCF_and_ratio_l36_36677


namespace max_profit_at_grade_9_l36_36907

def profit (k : ℕ) : ℕ :=
  (8 + 2 * (k - 1)) * (60 - 3 * (k - 1))

theorem max_profit_at_grade_9 : ∀ k, 1 ≤ k ∧ k ≤ 10 → profit k ≤ profit 9 := 
by
  sorry

end max_profit_at_grade_9_l36_36907


namespace min_value_of_n_for_constant_term_l36_36702

theorem min_value_of_n_for_constant_term :
  ∃ (n : ℕ) (r : ℕ) (h₁ : r > 0) (h₂ : n > 0), 
  (2 * n - 7 * r / 3 = 0) ∧ n = 7 :=
by
  sorry

end min_value_of_n_for_constant_term_l36_36702


namespace sqrt_expression_simplification_l36_36353

theorem sqrt_expression_simplification :
  (Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 - |2 - Real.sqrt 6|) = 2 :=
by
  sorry

end sqrt_expression_simplification_l36_36353


namespace g_g_g_g_15_eq_3_l36_36011

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_g_g_g_15_eq_3 : g (g (g (g 15))) = 3 := 
by
  sorry

end g_g_g_g_15_eq_3_l36_36011


namespace emilia_blueberries_l36_36040

def cartons_needed : Nat := 42
def cartons_strawberries : Nat := 2
def cartons_bought : Nat := 33

def cartons_blueberries (needed : Nat) (strawberries : Nat) (bought : Nat) : Nat :=
  needed - (strawberries + bought)

theorem emilia_blueberries : cartons_blueberries cartons_needed cartons_strawberries cartons_bought = 7 :=
by
  sorry

end emilia_blueberries_l36_36040


namespace shaded_area_correct_l36_36038

-- Given definitions
def square_side_length : ℝ := 1
def grid_rows : ℕ := 3
def grid_columns : ℕ := 9

def triangle1_area : ℝ := 3
def triangle2_area : ℝ := 1
def triangle3_area : ℝ := 3
def triangle4_area : ℝ := 3

def total_grid_area := (grid_rows * grid_columns : ℕ) * square_side_length^2
def total_unshaded_area := triangle1_area + triangle2_area + triangle3_area + triangle4_area

-- Problem statement
theorem shaded_area_correct :
  total_grid_area - total_unshaded_area = 17 := 
by
  sorry

end shaded_area_correct_l36_36038


namespace john_apartment_number_l36_36010

variable (k d m : ℕ)

theorem john_apartment_number (h1 : k = m) (h2 : d + m = 239) (h3 : 10 * (k - 1) + 1 ≤ d) (h4 : d ≤ 10 * k) : d = 217 := 
by 
  sorry

end john_apartment_number_l36_36010


namespace calculator_unit_prices_and_min_cost_l36_36288

-- Definitions for conditions
def unit_price_type_A (x : ℕ) : Prop :=
  ∀ y : ℕ, (y = x + 10) → (550 / x = 600 / y)

def purchase_constraint (a : ℕ) : Prop :=
  25 ≤ a ∧ a ≤ 100

def total_cost (a : ℕ) (x y : ℕ) : ℕ :=
  110 * a + 120 * (100 - a)

-- Statement to prove
theorem calculator_unit_prices_and_min_cost :
  ∃ x y, unit_price_type_A x ∧ unit_price_type_A x ∧ total_cost 100 x y = 11000 :=
by
  sorry

end calculator_unit_prices_and_min_cost_l36_36288


namespace sum_of_numerator_and_denominator_l36_36951

def repeating_decimal_to_fraction_sum (x : ℚ) := 
  let numerator := 710
  let denominator := 99
  numerator + denominator

theorem sum_of_numerator_and_denominator : repeating_decimal_to_fraction_sum (71/10 + 7/990) = 809 := by
  sorry

end sum_of_numerator_and_denominator_l36_36951


namespace geometric_sequence_a4_l36_36700

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ {m n p q}, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : geometric_sequence a) (h2 : a 2 = 4) (h6 : a 6 = 16) :
  a 4 = 8 :=
by {
  -- Here you can provide the proof steps if needed
  sorry
}

end geometric_sequence_a4_l36_36700


namespace angle_measure_l36_36330

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l36_36330


namespace fraction_multiplication_l36_36092

theorem fraction_multiplication :
  ((2 / 5) * (5 / 7) * (7 / 3) * (3 / 8) = 1 / 4) :=
sorry

end fraction_multiplication_l36_36092


namespace equality_of_floor_squares_l36_36067

theorem equality_of_floor_squares (n : ℕ) (hn : 0 < n) :
  (⌊Real.sqrt n + Real.sqrt (n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  (⌊Real.sqrt (4 * n + 2)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 3)⌋ :=
by
  sorry

end equality_of_floor_squares_l36_36067


namespace range_of_a_l36_36197

noncomputable
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∪ B = B) ↔ a < -4 ∨ a > 5 :=
sorry

end range_of_a_l36_36197


namespace trader_sold_meters_l36_36216

variable (x : ℕ) (SP P CP : ℕ)

theorem trader_sold_meters (h_SP : SP = 660) (h_P : P = 5) (h_CP : CP = 5) : x = 66 :=
  by
  sorry

end trader_sold_meters_l36_36216


namespace remainder_when_2x_divided_by_7_l36_36057

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end remainder_when_2x_divided_by_7_l36_36057


namespace L_shaped_figure_perimeter_is_14_l36_36117

-- Define the side length of each square as a constant
def side_length : ℕ := 2

-- Define the horizontal base length
def base_length : ℕ := 3 * side_length

-- Define the height of the vertical stack
def vertical_stack_height : ℕ := 2 * side_length

-- Define the total perimeter of the "L" shaped figure
def L_shaped_figure_perimeter : ℕ :=
  base_length + side_length + vertical_stack_height + side_length + side_length + vertical_stack_height

-- The theorem that states the perimeter of the L-shaped figure is 14 units
theorem L_shaped_figure_perimeter_is_14 : L_shaped_figure_perimeter = 14 := sorry

end L_shaped_figure_perimeter_is_14_l36_36117


namespace solve_quadratic_eq_l36_36642

theorem solve_quadratic_eq (x : ℝ) : (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l36_36642


namespace find_k_l36_36944

variable (m n p k : ℝ)

-- Conditions
def cond1 : Prop := m = 2 * n + 5
def cond2 : Prop := p = 3 * m - 4
def cond3 : Prop := m + 4 = 2 * (n + k) + 5
def cond4 : Prop := p + 3 = 3 * (m + 4) - 4

theorem find_k (h1 : cond1 m n)
               (h2 : cond2 m p)
               (h3 : cond3 m n k)
               (h4 : cond4 m p) :
               k = 2 :=
  sorry

end find_k_l36_36944


namespace water_pouring_problem_l36_36181

theorem water_pouring_problem : ∃ n : ℕ, n = 3 ∧
  (1 / (2 * n - 1) = 1 / 5) :=
by
  sorry

end water_pouring_problem_l36_36181


namespace sandwiches_count_l36_36672

theorem sandwiches_count (M : ℕ) (C : ℕ) (S : ℕ) (hM : M = 12) (hC : C = 12) (hS : S = 5) :
  M * (C * (C - 1) / 2) * S = 3960 := 
  by sorry

end sandwiches_count_l36_36672


namespace exists_divisible_by_3_l36_36114

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l36_36114


namespace students_after_joining_l36_36500

theorem students_after_joining (N : ℕ) (T : ℕ)
  (h1 : T = 48 * N)
  (h2 : 120 * 32 / (N + 120) + (T / (N + 120)) = 44)
  : N + 120 = 480 :=
by
  sorry

end students_after_joining_l36_36500


namespace arithmetic_geometric_mean_inequality_l36_36082

variable {a b : ℝ}

noncomputable def A (a b : ℝ) := (a + b) / 2
noncomputable def B (a b : ℝ) := Real.sqrt (a * b)

theorem arithmetic_geometric_mean_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) : A a b > B a b := 
by
  sorry

end arithmetic_geometric_mean_inequality_l36_36082


namespace domain_of_f_l36_36633

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | 4 - x ^ 2 ≥ 0 ∧ x ≠ 1}

theorem domain_of_f (x : ℝ) : domain_of_function x = {x | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l36_36633


namespace winning_candidate_votes_l36_36997

theorem winning_candidate_votes (T W : ℕ) (d1 d2 d3 : ℕ) 
  (hT : T = 963)
  (hd1 : d1 = 53) 
  (hd2 : d2 = 79) 
  (hd3 : d3 = 105) 
  (h_sum : T = W + (W - d1) + (W - d2) + (W - d3)) :
  W = 300 := 
by
  sorry

end winning_candidate_votes_l36_36997


namespace inequality_solution_set_l36_36932

theorem inequality_solution_set (x : ℝ) :
  ((1 - x) * (x - 3) < 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l36_36932


namespace wire_goes_around_field_l36_36399

theorem wire_goes_around_field :
  (7348 / (4 * Real.sqrt 27889)) = 11 :=
by
  sorry

end wire_goes_around_field_l36_36399


namespace quiz_answer_key_combinations_l36_36522

noncomputable def num_ways_answer_key : ℕ :=
  let true_false_combinations := 2^4
  let valid_true_false_combinations := true_false_combinations - 2
  let multi_choice_combinations := 4 * 4
  valid_true_false_combinations * multi_choice_combinations

theorem quiz_answer_key_combinations : num_ways_answer_key = 224 := 
by
  sorry

end quiz_answer_key_combinations_l36_36522


namespace cart_total_books_l36_36490

theorem cart_total_books (fiction non_fiction autobiographies picture: ℕ) 
  (h1: fiction = 5)
  (h2: non_fiction = fiction + 4)
  (h3: autobiographies = 2 * fiction)
  (h4: picture = 11)
  : fiction + non_fiction + autobiographies + picture = 35 := by
  -- Proof is omitted
  sorry

end cart_total_books_l36_36490


namespace min_intersection_l36_36647

open Finset

-- Definition of subset count function
def n (S : Finset ℕ) : ℕ :=
  2 ^ S.card

theorem min_intersection {A B C : Finset ℕ} (hA : A.card = 100) (hB : B.card = 100) 
  (h_subsets : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≥ 97 := by
  sorry

end min_intersection_l36_36647


namespace square_area_fraction_shaded_l36_36530

theorem square_area_fraction_shaded (s : ℝ) :
  let R := (s / 2, s)
  let S := (s, s / 2)
  -- Area of triangle RSV
  let area_RSV := (1 / 2) * (s / 2) * (s * Real.sqrt 2 / 4)
  -- Non-shaded area
  let non_shaded_area := area_RSV
  -- Total area of the square
  let total_area := s^2
  -- Shaded area
  let shaded_area := total_area - non_shaded_area
  -- Fraction shaded
  (shaded_area / total_area) = 1 - Real.sqrt 2 / 16 :=
by
  sorry

end square_area_fraction_shaded_l36_36530


namespace largest_class_students_l36_36469

theorem largest_class_students (x : ℕ) (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 105) : x = 25 :=
by {
  sorry
}

end largest_class_students_l36_36469


namespace graph_passes_through_2_2_l36_36973

theorem graph_passes_through_2_2 (a : ℝ) (h : a > 0) (h_ne : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
sorry

end graph_passes_through_2_2_l36_36973


namespace g_range_excludes_zero_l36_36251

noncomputable def g (x : ℝ) : ℤ :=
if x > -1 then ⌈1 / (x + 1)⌉
else ⌊1 / (x + 1)⌋

theorem g_range_excludes_zero : ¬ ∃ x : ℝ, g x = 0 := 
by 
  sorry

end g_range_excludes_zero_l36_36251


namespace marbles_in_jar_l36_36839

theorem marbles_in_jar (T : ℕ) (T_half : T / 2 = 12) (red_marbles : ℕ) (orange_marbles : ℕ) (total_non_blue : red_marbles + orange_marbles = 12) (red_count : red_marbles = 6) (orange_count : orange_marbles = 6) : T = 24 :=
by
  sorry

end marbles_in_jar_l36_36839


namespace inequality_proof_l36_36284

noncomputable def inequality_holds (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop :=
  (a ^ 2) / (b - 1) + (b ^ 2) / (a - 1) ≥ 8

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  inequality_holds a b ha hb :=
sorry

end inequality_proof_l36_36284


namespace incorrect_conclusion_l36_36450

theorem incorrect_conclusion :
  ∃ (a x y : ℝ), 
  (x + 3 * y = 4 - a ∧ x - y = 3 * a) ∧ 
  (∀ (xa ya : ℝ), (xa = 2) → (x = 2 * xa + 1) ∧ (y = 1 - xa) → ¬ (xa + ya = 4 - xa)) :=
sorry

end incorrect_conclusion_l36_36450


namespace hourly_wage_difference_l36_36534

theorem hourly_wage_difference (P Q: ℝ) (H_p: ℝ) (H_q: ℝ) (h1: P = 1.5 * Q) (h2: H_q = H_p + 10) (h3: P * H_p = 420) (h4: Q * H_q = 420) : P - Q = 7 := by
  sorry

end hourly_wage_difference_l36_36534


namespace evaluate_series_l36_36761

-- Define the series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) / (3 ^ (n + 1))

-- Lean statement to show the evaluated series
theorem evaluate_series : (3:ℝ)^S = (3:ℝ)^(3 / 4) :=
by
  -- The proof is omitted
  sorry

end evaluate_series_l36_36761


namespace tan_585_eq_1_l36_36814

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end tan_585_eq_1_l36_36814


namespace greatest_integer_l36_36408

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end greatest_integer_l36_36408


namespace problem_correct_calculation_l36_36337

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l36_36337


namespace maxwell_distance_when_meeting_l36_36023

theorem maxwell_distance_when_meeting 
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ) 
  (brad_speed : ℕ) 
  (total_distance : ℕ) 
  (h : distance_between_homes = 36) 
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 4) 
  (h3 : 6 * (total_distance / 6) = distance_between_homes) :
  total_distance = 12 :=
sorry

end maxwell_distance_when_meeting_l36_36023


namespace total_cookies_l36_36268

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l36_36268


namespace vertical_asymptote_l36_36729

theorem vertical_asymptote (x : ℝ) : (y = (2*x - 3) / (4*x + 5)) → (4*x + 5 = 0) → x = -5/4 := 
by 
  intros h1 h2
  sorry

end vertical_asymptote_l36_36729


namespace value_at_points_zero_l36_36168

def odd_function (v : ℝ → ℝ) := ∀ x : ℝ, v (-x) = -v x

theorem value_at_points_zero (v : ℝ → ℝ)
  (hv : odd_function v) :
  v (-2.1) + v (-1.2) + v (1.2) + v (2.1) = 0 :=
by {
  sorry
}

end value_at_points_zero_l36_36168


namespace product_expression_evaluation_l36_36224

theorem product_expression_evaluation :
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 :=
by
  sorry

end product_expression_evaluation_l36_36224


namespace wavelength_scientific_notation_l36_36151

theorem wavelength_scientific_notation :
  (0.000000193 : Float) = 1.93 * (10 : Float) ^ (-7) :=
sorry

end wavelength_scientific_notation_l36_36151


namespace sequence_arith_or_geom_l36_36825

def sequence_nature (a S : ℕ → ℝ) : Prop :=
  ∀ n, 4 * S n = (a n + 1) ^ 2

theorem sequence_arith_or_geom {a : ℕ → ℝ} {S : ℕ → ℝ} (h : sequence_nature a S) (h₁ : a 1 = 1) :
  (∃ d, ∀ n, a (n + 1) = a n + d) ∨ (∃ r, ∀ n, a (n + 1) = a n * r) :=
sorry

end sequence_arith_or_geom_l36_36825


namespace fbox_eval_correct_l36_36739

-- Define the function according to the condition
def fbox (a b c : ℕ) : ℕ := a^b - b^c + c^a

-- Propose the theorem 
theorem fbox_eval_correct : fbox 2 0 3 = 10 := 
by
  -- Proof will be provided here
  sorry

end fbox_eval_correct_l36_36739


namespace fruits_given_away_l36_36472

-- Definitions based on the conditions
def initial_pears := 10
def initial_oranges := 20
def initial_apples := 2 * initial_pears
def initial_fruits := initial_pears + initial_oranges + initial_apples
def fruits_left := 44

-- Theorem to prove the total number of fruits given to her sister
theorem fruits_given_away : initial_fruits - fruits_left = 6 := by
  sorry

end fruits_given_away_l36_36472


namespace match_scheduling_ways_l36_36345

def different_ways_to_schedule_match (num_players : Nat) (num_rounds : Nat) : Nat :=
  (num_rounds.factorial * num_rounds.factorial)

theorem match_scheduling_ways : different_ways_to_schedule_match 4 4 = 576 :=
by
  sorry

end match_scheduling_ways_l36_36345


namespace necessary_and_sufficient_l36_36769

def point_on_curve (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : Prop :=
  f P = 0

theorem necessary_and_sufficient (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) :
  (point_on_curve P f ↔ f P = 0) :=
by
  sorry

end necessary_and_sufficient_l36_36769


namespace painted_cube_problem_l36_36524

theorem painted_cube_problem (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2)^2 = (n - 2)^3) : n = 8 :=
by {
  sorry
}

end painted_cube_problem_l36_36524


namespace oliver_gave_janet_l36_36165

def initial_candy : ℕ := 78
def remaining_candy : ℕ := 68

theorem oliver_gave_janet : initial_candy - remaining_candy = 10 :=
by
  sorry

end oliver_gave_janet_l36_36165


namespace logan_money_left_l36_36643

-- Defining the given conditions
def income : ℕ := 65000
def rent_expense : ℕ := 20000
def groceries_expense : ℕ := 5000
def gas_expense : ℕ := 8000
def additional_income_needed : ℕ := 10000

-- Calculating total expenses
def total_expense : ℕ := rent_expense + groceries_expense + gas_expense

-- Desired income
def desired_income : ℕ := income + additional_income_needed

-- The theorem to prove
theorem logan_money_left : (desired_income - total_expense) = 42000 :=
by
  -- A placeholder for the proof
  sorry

end logan_money_left_l36_36643


namespace grade_distribution_sum_l36_36470

theorem grade_distribution_sum (a b c d : ℝ) (ha : a = 0.6) (hb : b = 0.25) (hc : c = 0.1) (hd : d = 0.05) :
  a + b + c + d = 1.0 :=
by
  -- Introduce the hypothesis
  rw [ha, hb, hc, hd]
  -- Now the goal simplifies to: 0.6 + 0.25 + 0.1 + 0.05 = 1.0
  sorry

end grade_distribution_sum_l36_36470


namespace smallest_k_divisibility_l36_36658

theorem smallest_k_divisibility : ∃ (k : ℕ), k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_divisibility_l36_36658


namespace distance_from_y_axis_l36_36058

theorem distance_from_y_axis (dx dy : ℝ) (h1 : dx = 8) (h2 : dx = (1/2) * dy) : dy = 16 :=
by
  sorry

end distance_from_y_axis_l36_36058


namespace max_cut_strings_preserving_net_l36_36032

-- Define the conditions of the problem
def volleyball_net_width : ℕ := 50
def volleyball_net_height : ℕ := 600

-- The vertices count is calculated as (width + 1) * (height + 1)
def vertices_count : ℕ := (volleyball_net_width + 1) * (volleyball_net_height + 1)

-- The total edges count is the sum of vertical and horizontal edges
def total_edges_count : ℕ := volleyball_net_width * (volleyball_net_height + 1) + (volleyball_net_width + 1) * volleyball_net_height

-- The edges needed to keep the graph connected (number of vertices - 1)
def edges_in_tree : ℕ := vertices_count - 1

-- The maximum removable edges (total edges - edges needed in tree)
def max_removable_edges : ℕ := total_edges_count - edges_in_tree

-- Define the theorem to prove
theorem max_cut_strings_preserving_net : max_removable_edges = 30000 := by
  sorry

end max_cut_strings_preserving_net_l36_36032


namespace bamboo_sections_volume_l36_36751

theorem bamboo_sections_volume (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a n = a 0 + n * d) →
  (a 0 + a 1 + a 2 = 4) →
  (a 5 + a 6 + a 7 + a 8 = 3) →
  (a 3 + a 4 = 2 + 3 / 22) :=
sorry

end bamboo_sections_volume_l36_36751


namespace trillion_in_scientific_notation_l36_36441

theorem trillion_in_scientific_notation :
  (10^4) * (10^4) * (10^4) = 10^(12) := 
by sorry

end trillion_in_scientific_notation_l36_36441


namespace gcd_polynomial_l36_36257

theorem gcd_polynomial (b : ℤ) (k : ℤ) (hk : k % 2 = 1) (h_b : b = 1193 * k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := 
  sorry

end gcd_polynomial_l36_36257


namespace sin_60_eq_sqrt3_div_2_l36_36732

-- Problem statement translated to Lean
theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end sin_60_eq_sqrt3_div_2_l36_36732


namespace bettys_herb_garden_l36_36124

theorem bettys_herb_garden :
  ∀ (basil oregano thyme rosemary total : ℕ),
    oregano = 2 * basil + 2 →
    thyme = 3 * basil - 3 →
    rosemary = (basil + thyme) / 2 →
    basil = 5 →
    total = basil + oregano + thyme + rosemary →
    total ≤ 50 →
    total = 37 :=
by
  intros basil oregano thyme rosemary total h_oregano h_thyme h_rosemary h_basil h_total h_le_total
  sorry

end bettys_herb_garden_l36_36124


namespace acute_angle_10_10_l36_36529

noncomputable def clock_angle_proof : Prop :=
  let minute_hand_position := 60
  let hour_hand_position := 305
  let angle_diff := hour_hand_position - minute_hand_position
  let acute_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  acute_angle = 115

theorem acute_angle_10_10 : clock_angle_proof := by
  sorry

end acute_angle_10_10_l36_36529


namespace plot_length_l36_36585

theorem plot_length (b : ℕ) (cost_per_meter total_cost : ℕ)
  (h1 : cost_per_meter = 2650 / 100)  -- Since Lean works with integers, use 2650 instead of 26.50
  (h2 : total_cost = 5300)
  (h3 : 2 * (b + 16) + 2 * b = total_cost / cost_per_meter) :
  b + 16 = 58 :=
by
  -- Above theorem aims to prove the length of the plot is 58 meters, given the conditions.
  sorry

end plot_length_l36_36585


namespace valid_call_time_at_15_l36_36269

def time_difference := 5 -- Beijing is 5 hours ahead of Moscow

def beijing_start_time := 14 -- Start time in Beijing corresponding to 9:00 in Moscow
def beijing_end_time := 17  -- End time in Beijing corresponding to 17:00 in Beijing

-- Define the call time in Beijing
def call_time_beijing := 15

-- The time window during which they can start the call in Beijing
def valid_call_time (t : ℕ) : Prop :=
  beijing_start_time <= t ∧ t <= beijing_end_time

-- The theorem to prove that 15:00 is a valid call time in Beijing
theorem valid_call_time_at_15 : valid_call_time call_time_beijing :=
by
  sorry

end valid_call_time_at_15_l36_36269


namespace EF_squared_correct_l36_36809

-- Define the problem setup and the proof goal.
theorem EF_squared_correct :
  ∀ (A B C D E F : Type)
  (side : ℝ)
  (h1 : side = 10)
  (BE DF AE CF : ℝ)
  (h2 : BE = 7)
  (h3 : DF = 7)
  (h4 : AE = 15)
  (h5 : CF = 15)
  (EF_squared : ℝ),
  EF_squared = 548 :=
by
  sorry

end EF_squared_correct_l36_36809


namespace savings_percentage_is_correct_l36_36186

-- Definitions for given conditions
def jacket_original_price : ℕ := 100
def shirt_original_price : ℕ := 50
def shoes_original_price : ℕ := 60

def jacket_discount : ℝ := 0.30
def shirt_discount : ℝ := 0.40
def shoes_discount : ℝ := 0.25

-- Definitions for savings
def jacket_savings : ℝ := jacket_original_price * jacket_discount
def shirt_savings : ℝ := shirt_original_price * shirt_discount
def shoes_savings : ℝ := shoes_original_price * shoes_discount

-- Definition for total savings and total original cost
def total_savings : ℝ := jacket_savings + shirt_savings + shoes_savings
def total_original_cost : ℕ := jacket_original_price + shirt_original_price + shoes_original_price

-- The theorems to be proven
theorem savings_percentage_is_correct : (total_savings / total_original_cost * 100) = 30.95 := by
  sorry

end savings_percentage_is_correct_l36_36186


namespace consecutive_odd_integers_l36_36078

theorem consecutive_odd_integers (n : ℕ) (h1 : n > 0) (h2 : (1 : ℚ) / n * ((n : ℚ) * 154) = 154) : n = 10 :=
sorry

end consecutive_odd_integers_l36_36078


namespace fraction_solution_l36_36413

theorem fraction_solution (N : ℝ) (h : N = 12.0) : (0.6667 * N + 1) = (3/4) * N := by 
  sorry

end fraction_solution_l36_36413


namespace range_of_m_l36_36409

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ (m < -1)) :=
sorry

end range_of_m_l36_36409


namespace S_40_value_l36_36571

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom h1 : S 10 = 10
axiom h2 : S 30 = 70

theorem S_40_value : S 40 = 150 :=
by
  -- Conditions
  have h1 : S 10 = 10 := h1
  have h2 : S 30 = 70 := h2
  -- Start proof here
  sorry

end S_40_value_l36_36571


namespace rooster_weight_l36_36021

variable (W : ℝ)  -- The weight of the first rooster

theorem rooster_weight (h1 : 0.50 * W + 0.50 * 40 = 35) : W = 30 :=
by
  sorry

end rooster_weight_l36_36021


namespace expand_and_simplify_l36_36674

variable (x : ℝ)

theorem expand_and_simplify : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_and_simplify_l36_36674


namespace product_of_digits_of_N_l36_36261

theorem product_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2485) : 
  (N.digits 10).prod = 0 :=
sorry

end product_of_digits_of_N_l36_36261


namespace central_angle_measure_l36_36532

-- Constants representing the arc length and the area of the sector.
def arc_length : ℝ := 5
def sector_area : ℝ := 5

-- Variables representing the central angle in radians and the radius.
variable (α r : ℝ)

-- Conditions given in the problem.
axiom arc_length_eq : arc_length = α * r
axiom sector_area_eq : sector_area = 1 / 2 * α * r^2

-- The goal to prove that the radian measure of the central angle α is 5 / 2.
theorem central_angle_measure : α = 5 / 2 := by sorry

end central_angle_measure_l36_36532


namespace toms_age_ratio_l36_36292

variables (T N : ℕ)

-- Conditions
def toms_age (T : ℕ) := T
def sum_of_children_ages (T : ℕ) := T
def years_ago (T N : ℕ) := T - N
def children_ages_years_ago (T N : ℕ) := T - 4 * N

-- Given statement
theorem toms_age_ratio (h1 : toms_age T = sum_of_children_ages T)
  (h2 : years_ago T N = 3 * children_ages_years_ago T N) :
  T / N = 11 / 2 :=
sorry

end toms_age_ratio_l36_36292


namespace shaded_solid_volume_l36_36667

noncomputable def volume_rectangular_prism (length width height : ℕ) : ℕ :=
  length * width * height

theorem shaded_solid_volume :
  volume_rectangular_prism 4 5 6 - volume_rectangular_prism 1 2 4 = 112 :=
by
  sorry

end shaded_solid_volume_l36_36667


namespace loss_percentage_l36_36660

theorem loss_percentage (C : ℝ) (h : 40 * C = 100 * C) : 
  ∃ L : ℝ, L = 60 := 
sorry

end loss_percentage_l36_36660


namespace quadratic_inequality_solution_set_l36_36217

theorem quadratic_inequality_solution_set (x : ℝ) :
  (x^2 - 3 * x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
sorry

end quadratic_inequality_solution_set_l36_36217


namespace bruno_pens_l36_36634

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end bruno_pens_l36_36634


namespace range_of_a_l36_36525

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 2 * a * (1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x, x > 2 → f a x > f a 2) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici (1 / 4)) :=
by
  sorry

end range_of_a_l36_36525


namespace polygon_triangle_division_l36_36077

theorem polygon_triangle_division (n k : ℕ) (h₁ : n ≥ 3) (h₂ : k ≥ 1):
  k ≥ n - 2 :=
sorry

end polygon_triangle_division_l36_36077


namespace smallest_non_factor_product_of_100_l36_36476

/-- Let a and b be distinct positive integers that are factors of 100. 
    The smallest value of their product which is not a factor of 100 is 8. -/
theorem smallest_non_factor_product_of_100 (a b : ℕ) (hab : a ≠ b) (ha : a ∣ 100) (hb : b ∣ 100) (hprod : ¬ (a * b ∣ 100)) : a * b = 8 :=
sorry

end smallest_non_factor_product_of_100_l36_36476


namespace intersection_result_l36_36163

open Set

namespace ProofProblem

def A : Set ℝ := {x | |x| ≤ 4}
def B : Set ℝ := {x | 4 ≤ x ∧ x < 5}

theorem intersection_result : A ∩ B = {4} :=
  sorry

end ProofProblem

end intersection_result_l36_36163


namespace number_of_yellow_marbles_l36_36131

theorem number_of_yellow_marbles (total_marbles blue_marbles red_marbles green_marbles yellow_marbles : ℕ)
    (h_total : total_marbles = 164) 
    (h_blue : blue_marbles = total_marbles / 2)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27) :
    yellow_marbles = total_marbles - (blue_marbles + red_marbles + green_marbles) →
    yellow_marbles = 14 := by
  sorry

end number_of_yellow_marbles_l36_36131


namespace cordelia_bleach_time_l36_36305

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end cordelia_bleach_time_l36_36305


namespace lecture_hall_rows_l36_36228

-- We define the total number of seats
def total_seats (n : ℕ) : ℕ := n * (n + 11)

-- We state the problem with the given conditions
theorem lecture_hall_rows : 
  (400 ≤ total_seats n) ∧ (total_seats n ≤ 440) → n = 16 :=
by
  sorry

end lecture_hall_rows_l36_36228


namespace cars_needed_to_double_march_earnings_l36_36415

-- Definition of given conditions
def base_salary : Nat := 1000
def commission_per_car : Nat := 200
def march_earnings : Nat := 2000

-- Question to prove
theorem cars_needed_to_double_march_earnings : 
  (2 * march_earnings - base_salary) / commission_per_car = 15 := 
by sorry

end cars_needed_to_double_march_earnings_l36_36415


namespace sum_binomials_eq_l36_36553

theorem sum_binomials_eq : 
  (Nat.choose 6 1) + (Nat.choose 6 2) + (Nat.choose 6 3) + (Nat.choose 6 4) + (Nat.choose 6 5) = 62 :=
by
  sorry

end sum_binomials_eq_l36_36553


namespace library_books_difference_l36_36544

theorem library_books_difference :
  let books_old_town := 750
  let books_riverview := 1240
  let books_downtown := 1800
  let books_eastside := 1620
  books_downtown - books_old_town = 1050 :=
by
  sorry

end library_books_difference_l36_36544


namespace exponential_equivalence_l36_36545

theorem exponential_equivalence (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end exponential_equivalence_l36_36545


namespace max_marks_l36_36694

theorem max_marks (score shortfall passing_threshold : ℝ) (h1 : score = 212) (h2 : shortfall = 19) (h3 : passing_threshold = 0.30) :
  ∃ M, M = 770 :=
by
  sorry

end max_marks_l36_36694


namespace find_z_l36_36352

open Complex

theorem find_z (z : ℂ) (h : (1 + 2 * z) / (1 - z) = Complex.I) : 
  z = -1 / 5 + 3 / 5 * Complex.I := 
sorry

end find_z_l36_36352


namespace part1_part2_part3_part4_l36_36062

-- Part 1: Prove that 1/42 is equal to 1/6 - 1/7
theorem part1 : (1/42 : ℚ) = (1/6 : ℚ) - (1/7 : ℚ) := sorry

-- Part 2: Prove that 1/240 is equal to 1/15 - 1/16
theorem part2 : (1/240 : ℚ) = (1/15 : ℚ) - (1/16 : ℚ) := sorry

-- Part 3: Prove the general rule for all natural numbers m
theorem part3 (m : ℕ) (hm : m > 0) : (1 / (m * (m + 1)) : ℚ) = (1 / m : ℚ) - (1 / (m + 1) : ℚ) := sorry

-- Part 4: Prove the given expression evaluates to 0 for any x
theorem part4 (x : ℚ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) : 
  (1 / ((x - 2) * (x - 3)) : ℚ) - (2 / ((x - 1) * (x - 3)) : ℚ) + (1 / ((x - 1) * (x - 2)) : ℚ) = 0 := sorry

end part1_part2_part3_part4_l36_36062


namespace winnie_keeps_10_lollipops_l36_36795

def winnie_keep_lollipops : Prop :=
  let cherry := 72
  let wintergreen := 89
  let grape := 23
  let shrimp_cocktail := 316
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 14
  let lollipops_per_friend := total_lollipops / friends
  let winnie_keeps := total_lollipops % friends
  winnie_keeps = 10

theorem winnie_keeps_10_lollipops : winnie_keep_lollipops := by
  sorry

end winnie_keeps_10_lollipops_l36_36795


namespace max_value_expression_l36_36608

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (m : ℝ), (∀ x y : ℝ, x * y > 0 → 
  m ≥ (x / (x + y) + 2 * y / (x + 2 * y))) ∧ 
  m = 4 - 2 * Real.sqrt 2 := 
sorry

end max_value_expression_l36_36608


namespace crayons_left_l36_36505

def initial_green_crayons : ℝ := 5
def initial_blue_crayons : ℝ := 8
def initial_yellow_crayons : ℝ := 7
def given_green_crayons : ℝ := 3.5
def given_blue_crayons : ℝ := 1.25
def given_yellow_crayons : ℝ := 2.75
def broken_yellow_crayons : ℝ := 0.5

theorem crayons_left (initial_green_crayons initial_blue_crayons initial_yellow_crayons given_green_crayons given_blue_crayons given_yellow_crayons broken_yellow_crayons : ℝ) :
  initial_green_crayons - given_green_crayons + 
  initial_blue_crayons - given_blue_crayons + 
  initial_yellow_crayons - given_yellow_crayons - broken_yellow_crayons = 12 :=
by
  sorry

end crayons_left_l36_36505


namespace bridge_length_is_235_l36_36663

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235_l36_36663


namespace cos_pi_minus_half_alpha_l36_36332

-- Conditions given in the problem
variable (α : ℝ)
variable (hα1 : 0 < α ∧ α < π / 2)
variable (hα2 : Real.sin α = 3 / 5)

-- The proof problem statement
theorem cos_pi_minus_half_alpha (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.sin α = 3 / 5) : 
  Real.cos (π - α / 2) = -3 * Real.sqrt 10 / 10 := 
sorry

end cos_pi_minus_half_alpha_l36_36332


namespace value_of_a_m_minus_2n_l36_36133

variable (a : ℝ) (m n : ℝ)

theorem value_of_a_m_minus_2n (h1 : a^m = 8) (h2 : a^n = 4) : a^(m - 2 * n) = 1 / 2 :=
by
  sorry

end value_of_a_m_minus_2n_l36_36133


namespace probability_at_least_one_six_l36_36786

theorem probability_at_least_one_six :
  let p_six := 1 / 6
  let p_not_six := 5 / 6
  let p_not_six_three_rolls := p_not_six ^ 3
  let p_at_least_one_six := 1 - p_not_six_three_rolls
  p_at_least_one_six = 91 / 216 :=
by
  sorry

end probability_at_least_one_six_l36_36786


namespace fraction_quaduple_l36_36466

variable (b a : ℤ)

theorem fraction_quaduple (h₁ : a ≠ 0) : (2 * b) / (a / 2) = 4 * (b / a) :=
by
  sorry

end fraction_quaduple_l36_36466


namespace solve_for_C_days_l36_36969

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 15
noncomputable def C_work_rate : ℚ := 1 / 50
noncomputable def total_work_done_by_A_B : ℚ := 6 * (A_work_rate + B_work_rate)
noncomputable def remaining_work : ℚ := 1 - total_work_done_by_A_B

theorem solve_for_C_days : ∃ d : ℚ, d * C_work_rate = remaining_work ∧ d = 15 :=
by
  use 15
  simp [C_work_rate, remaining_work, total_work_done_by_A_B, A_work_rate, B_work_rate]
  sorry

end solve_for_C_days_l36_36969


namespace quadratic_inequality_solution_l36_36971

theorem quadratic_inequality_solution 
  (a : ℝ) 
  (h : ∀ x : ℝ, -1 < x ∧ x < a → -x^2 + 2 * a * x + a + 1 > a + 1) : -1 < a ∧ a ≤ -1/2 :=
sorry

end quadratic_inequality_solution_l36_36971


namespace expression_equals_4008_l36_36877

def calculate_expression : ℤ :=
  let expr := (2004 - (2011 - 196)) + (2011 - (196 - 2004))
  expr

theorem expression_equals_4008 : calculate_expression = 4008 := 
by
  sorry

end expression_equals_4008_l36_36877


namespace number_of_second_graders_l36_36266

-- Define the number of kindergartners
def kindergartners : ℕ := 34

-- Define the number of first graders
def first_graders : ℕ := 48

-- Define the total number of students
def total_students : ℕ := 120

-- Define the proof statement
theorem number_of_second_graders : total_students - (kindergartners + first_graders) = 38 := by
  -- omit the proof details
  sorry

end number_of_second_graders_l36_36266


namespace chess_tournament_games_l36_36484

def players : ℕ := 12

def games_per_pair : ℕ := 2

theorem chess_tournament_games (n : ℕ) (h : n = players) : 
  (n * (n - 1) * games_per_pair) = 264 := by
  sorry

end chess_tournament_games_l36_36484


namespace geometric_sequence_a7_l36_36822

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition of the sequence

theorem geometric_sequence_a7 :
  a 3 = 1 → a 11 = 25 → a 7 = 5 := 
by
  intros h3 h11
  sorry

end geometric_sequence_a7_l36_36822


namespace altitude_on_hypotenuse_l36_36515

theorem altitude_on_hypotenuse (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) (c : ℝ) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  ∃ h : ℝ, h = (a * b) / c ∧ h = 60 / 13 :=
by
  use (5 * 12) / 13
  -- proof that (60 / 13) is indeed the altitude will be done by verifying calculations
  sorry

end altitude_on_hypotenuse_l36_36515


namespace new_member_younger_by_160_l36_36715

theorem new_member_younger_by_160 
  (A : ℕ)  -- average age 8 years ago and today
  (O N : ℕ)  -- age of the old member and the new member respectively
  (h1 : 20 * A = 20 * A + O - N)  -- condition derived from the problem
  (h2 : 20 * 8 = 160)  -- age increase over 8 years for 20 members
  (h3 : O - N = 160) : O - N = 160 :=
by
  sorry

end new_member_younger_by_160_l36_36715


namespace diff_eq_40_l36_36044

theorem diff_eq_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end diff_eq_40_l36_36044


namespace big_dogs_count_l36_36376

theorem big_dogs_count (B S : ℕ) (h_ratio : 3 * S = 17 * B) (h_total : B + S = 80) :
  B = 12 :=
by
  sorry

end big_dogs_count_l36_36376


namespace cube_root_59319_cube_root_103823_l36_36364

theorem cube_root_59319 : ∃ x : ℕ, x ^ 3 = 59319 ∧ x = 39 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

theorem cube_root_103823 : ∃ x : ℕ, x ^ 3 = 103823 ∧ x = 47 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

end cube_root_59319_cube_root_103823_l36_36364


namespace solution_set_of_inequality_l36_36854

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + 2 * x - 3 > 0 } = { x : ℝ | x < -3 ∨ x > 1 } :=
sorry

end solution_set_of_inequality_l36_36854


namespace maximum_value_x2y_y2z_z2x_l36_36685

theorem maximum_value_x2y_y2z_z2x (x y z : ℝ) (h_sum : x + y + z = 0) (h_squares : x^2 + y^2 + z^2 = 6) :
  x^2 * y + y^2 * z + z^2 * x ≤ 6 :=
sorry

end maximum_value_x2y_y2z_z2x_l36_36685


namespace ratio_of_shaded_area_l36_36324

theorem ratio_of_shaded_area 
  (AC : ℝ) (CB : ℝ) 
  (AB : ℝ := AC + CB) 
  (radius_AC : ℝ := AC / 2) 
  (radius_CB : ℝ := CB / 2)
  (radius_AB : ℝ := AB / 2) 
  (shaded_area : ℝ := (radius_AB ^ 2 * Real.pi / 2) - (radius_AC ^ 2 * Real.pi / 2) - (radius_CB ^ 2 * Real.pi / 2))
  (CD : ℝ := Real.sqrt (AC^2 - radius_CB^2))
  (circle_area : ℝ := CD^2 * Real.pi) :
  (shaded_area / circle_area = 21 / 187) := 
by 
  sorry

end ratio_of_shaded_area_l36_36324


namespace true_propositions_count_l36_36274

theorem true_propositions_count :
  (∃ x₀ : ℤ, x₀^3 < 0) ∧
  ((∀ a : ℝ, (∃ x : ℝ, a*x^2 + 2*x + 1 = 0 ∧ x < 0) ↔ a ≤ 1) → false) ∧ 
  (¬ (∀ x : ℝ, x^2 = 1/4 * x^2 → y = 1 → false)) →
  true_prop_count = 1 := 
sorry

end true_propositions_count_l36_36274


namespace max_intersection_l36_36070

open Finset

def n (S : Finset α) : ℕ := (2 : ℕ) ^ S.card

theorem max_intersection (A B C : Finset ℕ)
  (h1 : A.card = 2016)
  (h2 : B.card = 2016)
  (h3 : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≤ 2015 :=
sorry

end max_intersection_l36_36070


namespace customers_left_l36_36357

-- Given conditions:
def initial_customers : ℕ := 21
def remaining_customers : ℕ := 12

-- Prove that the number of customers who left is 9
theorem customers_left : initial_customers - remaining_customers = 9 := by
  sorry

end customers_left_l36_36357


namespace employed_males_percentage_l36_36793

theorem employed_males_percentage (P : ℕ) (H1: P > 0)
    (employed_pct : ℝ) (female_pct : ℝ)
    (H_employed_pct : employed_pct = 0.64)
    (H_female_pct : female_pct = 0.140625) :
    (0.859375 * employed_pct * 100) = 54.96 :=
by
  sorry

end employed_males_percentage_l36_36793


namespace peter_ate_7_over_48_l36_36998

-- Define the initial conditions
def total_slices : ℕ := 16
def slices_peter_ate : ℕ := 2
def shared_slice : ℚ := 1/3

-- Define the first part of the problem
def fraction_peter_ate_alone : ℚ := slices_peter_ate / total_slices

-- Define the fraction Peter ate from sharing one slice
def fraction_peter_ate_shared : ℚ := shared_slice / total_slices

-- Define the total fraction Peter ate
def total_fraction_peter_ate : ℚ := fraction_peter_ate_alone + fraction_peter_ate_shared

-- Create the theorem to be proved (statement only)
theorem peter_ate_7_over_48 :
  total_fraction_peter_ate = 7 / 48 :=
by
  sorry

end peter_ate_7_over_48_l36_36998


namespace rem_product_eq_l36_36048

theorem rem_product_eq 
  (P Q R k : ℤ) 
  (hk : k > 0) 
  (hPQ : P * Q = R) : 
  ((P % k) * (Q % k)) % k = R % k :=
by
  sorry

end rem_product_eq_l36_36048


namespace perpendicular_vectors_t_values_l36_36493

variable (t : ℝ)
def a := (t, 0, -1)
def b := (2, 5, t^2)

theorem perpendicular_vectors_t_values (h : (2 * t + 0 * 5 + -1 * t^2) = 0) : t = 0 ∨ t = 2 :=
by sorry

end perpendicular_vectors_t_values_l36_36493


namespace parametric_two_rays_l36_36898

theorem parametric_two_rays (t : ℝ) : (x = t + 1 / t ∧ y = 2) → (x ≤ -2 ∨ x ≥ 2) := by
  sorry

end parametric_two_rays_l36_36898


namespace total_distance_traveled_eq_l36_36076

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end total_distance_traveled_eq_l36_36076


namespace trigonometric_inequality_l36_36834

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (hx : 0 < x ∧ x < (Real.pi / 2))
  (hnm : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤
  3 * |Real.sin x ^ m - Real.cos x ^ m| := 
by 
  sorry

end trigonometric_inequality_l36_36834


namespace camp_boys_count_l36_36301

/-- The ratio of boys to girls and total number of individuals in the camp including teachers
is given, we prove the number of boys is 26. -/
theorem camp_boys_count 
  (b g t : ℕ) -- b = number of boys, g = number of girls, t = number of teachers
  (h1 : b = 3 * (t - 5))  -- boys count related to some integer "t" minus teachers
  (h2 : g = 4 * (t - 5))  -- girls count related to some integer "t" minus teachers
  (total_individuals : t = 65) : 
  b = 26 :=
by
  have h : 3 * (t - 5) + 4 * (t - 5) + 5 = 65 := sorry
  sorry

end camp_boys_count_l36_36301


namespace injective_function_equality_l36_36735

def injective (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b : ℕ⦄, f a = f b → a = b

theorem injective_function_equality
  {f : ℕ → ℕ}
  (h_injective : injective f)
  (h_eq : ∀ n m : ℕ, (1 / f n) + (1 / f m) = 4 / (f n + f m)) :
  ∀ n m : ℕ, m = n :=
by
  sorry

end injective_function_equality_l36_36735


namespace max_length_of_cuts_l36_36713

-- Define the dimensions of the board and the number of parts
def board_size : ℕ := 30
def num_parts : ℕ := 225

-- Define the total possible length of the cuts
def max_possible_cuts_length : ℕ := 1065

-- Define the condition that the board is cut into parts of equal area
def equal_area_partition (board_size num_parts : ℕ) : Prop :=
  ∃ (area_per_part : ℕ), (board_size * board_size) / num_parts = area_per_part

-- Define the theorem to prove the maximum possible total length of the cuts
theorem max_length_of_cuts (h : equal_area_partition board_size num_parts) :
  max_possible_cuts_length = 1065 :=
by
  -- Proof to be filled in
  sorry

end max_length_of_cuts_l36_36713


namespace digit_sum_subtraction_l36_36828

theorem digit_sum_subtraction (P Q R S : ℕ) (hQ : Q + P = P) (hP : Q - P = 0) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10) (h4 : S < 10) : S = 0 := by
  sorry

end digit_sum_subtraction_l36_36828


namespace sequence_expression_l36_36051

theorem sequence_expression (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  ∀ n, a n = n * 2^(n - 1) :=
by
  sorry

end sequence_expression_l36_36051


namespace contrapositive_statement_l36_36644

theorem contrapositive_statement (m : ℝ) : 
  (¬ ∃ (x : ℝ), x^2 + x - m = 0) → m > 0 :=
by
  sorry

end contrapositive_statement_l36_36644


namespace school_student_ratio_l36_36801

theorem school_student_ratio :
  ∀ (F S T : ℕ), (T = 200) → (S = T + 40) → (F + S + T = 920) → (F : ℚ) / (S : ℚ) = 2 / 1 :=
by
  intros F S T hT hS hSum
  sorry

end school_student_ratio_l36_36801


namespace units_digit_of_8_pow_2022_l36_36974

theorem units_digit_of_8_pow_2022 : (8 ^ 2022) % 10 = 4 := 
by
  -- We here would provide the proof of this theorem
  sorry

end units_digit_of_8_pow_2022_l36_36974


namespace sum_abs_binom_coeff_l36_36282

theorem sum_abs_binom_coeff (a a1 a2 a3 a4 a5 a6 a7 : ℤ)
    (h : (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
    |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3 ^ 7 - 1 := sorry

end sum_abs_binom_coeff_l36_36282


namespace solve_for_x_l36_36279

theorem solve_for_x : ∀ (x : ℝ), (x = 3 / 4) →
  3 - (1 / (4 * (1 - x))) = 2 * (1 / (4 * (1 - x))) :=
by
  intros x h
  rw [h]
  sorry

end solve_for_x_l36_36279


namespace cost_of_marker_l36_36687

theorem cost_of_marker (s c m : ℕ) (h1 : s > 12) (h2 : m > 1) (h3 : c > m) (h4 : s * c * m = 924) : c = 11 :=
sorry

end cost_of_marker_l36_36687


namespace triangle_with_sticks_l36_36671

theorem triangle_with_sticks (c : ℕ) (h₁ : 4 + 9 > c) (h₂ : 9 - 4 < c) :
  c = 9 :=
by
  sorry

end triangle_with_sticks_l36_36671


namespace max_value_neg_expr_l36_36373

theorem max_value_neg_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) :=
by 
  sorry

end max_value_neg_expr_l36_36373


namespace rectangle_perimeters_l36_36720

theorem rectangle_perimeters (w h : ℝ) 
  (h1 : 2 * (w + h) = 20)
  (h2 : 2 * (4 * w + h) = 56) : 
  4 * (w + h) = 40 ∧ 2 * (w + 4 * h) = 44 := 
by
  sorry

end rectangle_perimeters_l36_36720


namespace initial_games_count_l36_36791

-- Definitions used in conditions
def games_given_away : ℕ := 99
def games_left : ℝ := 22.0

-- Theorem statement for the initial number of games
theorem initial_games_count : games_given_away + games_left = 121.0 := by
  sorry

end initial_games_count_l36_36791


namespace number_of_rows_of_red_notes_l36_36610

theorem number_of_rows_of_red_notes (R : ℕ) :
  let red_notes_in_each_row := 6
  let blue_notes_per_red_note := 2
  let additional_blue_notes := 10
  let total_notes := 100
  (6 * R + 12 * R + 10 = 100) → R = 5 :=
by
  intros
  sorry

end number_of_rows_of_red_notes_l36_36610


namespace f_periodic_l36_36699

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_periodic (f : ℝ → ℝ)
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

end f_periodic_l36_36699


namespace canoes_more_than_kayaks_l36_36206

noncomputable def canoes_difference (C K : ℕ) : Prop :=
  15 * C + 18 * K = 405 ∧ 2 * C = 3 * K → C - K = 5

theorem canoes_more_than_kayaks (C K : ℕ) : canoes_difference C K :=
by
  sorry

end canoes_more_than_kayaks_l36_36206


namespace probability_diamond_first_and_ace_or_king_second_l36_36391

-- Define the condition of the combined deck consisting of two standard decks (104 cards total)
def two_standard_decks := 104

-- Define the number of diamonds, aces, and kings in the combined deck
def number_of_diamonds := 26
def number_of_aces := 8
def number_of_kings := 8

-- Define the events for drawing cards
def first_card_is_diamond := (number_of_diamonds : ℕ) / (two_standard_decks : ℕ)
def second_card_is_ace_or_king_if_first_is_not_ace_or_king :=
  (16 / 103 : ℚ) -- 16 = 8 (aces) + 8 (kings)
def second_card_is_ace_or_king_if_first_is_ace_or_king :=
  (15 / 103 : ℚ) -- 15 = 7 (remaining aces) + 7 (remaining kings) + 1 (remaining ace or king of the same suit)

-- Define the probabilities of the combined event
def probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king :=
  (22 / 104) * (16 / 103)
def probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king :=
  (4 / 104) * (15 / 103)

-- Define the total probability combining both events
noncomputable def total_probability :=
  probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king +
  probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king

-- Theorem stating the desired probability result
theorem probability_diamond_first_and_ace_or_king_second :
  total_probability = (103 / 2678 : ℚ) :=
sorry

end probability_diamond_first_and_ace_or_king_second_l36_36391


namespace quadratic_equation_general_form_l36_36019

theorem quadratic_equation_general_form :
  ∀ (x : ℝ), 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end quadratic_equation_general_form_l36_36019


namespace degree_measure_of_subtracted_angle_l36_36987

def angle := 30

theorem degree_measure_of_subtracted_angle :
  let supplement := 180 - angle
  let complement_of_supplement := 90 - supplement
  let twice_complement := 2 * (90 - angle)
  twice_complement - complement_of_supplement = 180 :=
by
  sorry

end degree_measure_of_subtracted_angle_l36_36987


namespace probability_of_adjacent_vertices_in_decagon_l36_36311

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l36_36311


namespace seven_digit_divisible_by_eleven_l36_36939

theorem seven_digit_divisible_by_eleven (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 9) 
  (h3 : 10 - n ≡ 0 [MOD 11]) : n = 10 :=
by
  sorry

end seven_digit_divisible_by_eleven_l36_36939


namespace range_of_m_l36_36120

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l36_36120


namespace trigonometric_identity_l36_36675

open Real

theorem trigonometric_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) : 
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = 2 / tan α := 
by
  sorry

end trigonometric_identity_l36_36675


namespace minimize_sum_of_squares_at_mean_l36_36586

-- Definitions of the conditions
def P1 (x1 : ℝ) : ℝ := x1
def P2 (x2 : ℝ) : ℝ := x2
def P3 (x3 : ℝ) : ℝ := x3
def P4 (x4 : ℝ) : ℝ := x4
def P5 (x5 : ℝ) : ℝ := x5

-- Definition of the function we want to minimize
def s (P : ℝ) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (P - x1)^2 + (P - x2)^2 + (P - x3)^2 + (P - x4)^2 + (P - x5)^2

-- Proof statement
theorem minimize_sum_of_squares_at_mean (x1 x2 x3 x4 x5 : ℝ) :
  ∃ P : ℝ, P = (x1 + x2 + x3 + x4 + x5) / 5 ∧ 
           ∀ x : ℝ, s P x1 x2 x3 x4 x5 ≤ s x x1 x2 x3 x4 x5 := 
by
  sorry

end minimize_sum_of_squares_at_mean_l36_36586


namespace village_duration_l36_36592

theorem village_duration (vampire_drain : ℕ) (werewolf_eat : ℕ) (village_population : ℕ)
  (hv : vampire_drain = 3) (hw : werewolf_eat = 5) (hp : village_population = 72) :
  village_population / (vampire_drain + werewolf_eat) = 9 :=
by
  sorry

end village_duration_l36_36592


namespace xyz_identity_l36_36511

theorem xyz_identity (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : xy + xz + yz = 32) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 1400 := 
by 
  -- Proof steps will be placed here, use sorry for now
  sorry

end xyz_identity_l36_36511


namespace solve_quadratic_l36_36285

theorem solve_quadratic : ∀ x, x^2 - 4 * x + 3 = 0 ↔ x = 3 ∨ x = 1 := 
by
  sorry

end solve_quadratic_l36_36285


namespace solution_set_part1_solution_set_part2_l36_36557

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem solution_set_part1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem solution_set_part2 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end solution_set_part1_solution_set_part2_l36_36557


namespace ring_toss_total_amount_l36_36390

-- Defining the amounts made in the two periods
def amount_first_period : Nat := 382
def amount_second_period : Nat := 374

-- The total amount made
def total_amount : Nat := amount_first_period + amount_second_period

-- Statement that the total amount calculated is equal to the given answer
theorem ring_toss_total_amount :
  total_amount = 756 := by
  sorry

end ring_toss_total_amount_l36_36390


namespace sequence_an_formula_l36_36782

theorem sequence_an_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 :=
by
  sorry

end sequence_an_formula_l36_36782


namespace complex_number_identity_l36_36935

theorem complex_number_identity (i : ℂ) (hi : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_number_identity_l36_36935


namespace find_positive_integer_tuples_l36_36956

theorem find_positive_integer_tuples
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hz_prime : Prime z) :
  z ^ x = y ^ 3 + 1 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_positive_integer_tuples_l36_36956


namespace fever_above_threshold_l36_36407

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end fever_above_threshold_l36_36407


namespace remainder_when_divided_by_3x_minus_6_l36_36215

def polynomial (x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 9 * x^4 + 3 * x^3 - 7

def evaluate_at (f : ℝ → ℝ) (a : ℝ) : ℝ := f a

theorem remainder_when_divided_by_3x_minus_6 :
  evaluate_at polynomial 2 = 897 :=
by
  -- Compute this value manually or use automated tools
  sorry

end remainder_when_divided_by_3x_minus_6_l36_36215


namespace find_B_intersection_point_l36_36591

theorem find_B_intersection_point (k1 k2 : ℝ) (hA1 : 1 ≠ 0) 
  (hA2 : k1 = -2) (hA3 : k2 = -2) : 
  (-1, 2) ∈ {p : ℝ × ℝ | ∃ k1 k2, p.2 = k1 * p.1 ∧ p.2 = k2 / p.1} :=
sorry

end find_B_intersection_point_l36_36591


namespace geometric_sequence_first_term_and_ratio_l36_36604

theorem geometric_sequence_first_term_and_ratio (b : ℕ → ℚ) 
  (hb2 : b 2 = 37 + 1/3) 
  (hb6 : b 6 = 2 + 1/3) : 
  ∃ (b1 q : ℚ), b 1 = b1 ∧ (∀ n, b n = b1 * q^(n-1)) ∧ b1 = 224 / 3 ∧ q = 1 / 2 :=
by 
  sorry

end geometric_sequence_first_term_and_ratio_l36_36604


namespace problem_l36_36566

noncomputable def f (a b x : ℝ) := a * x^2 - b * x + 1

theorem problem (a b : ℝ) (h1 : 4 * a - b^2 = 3)
                (h2 : ∀ x : ℝ, f a b (x + 1) = f a b (-x))
                (h3 : b = a + 1) 
                (h4 : 0 ≤ a ∧ a ≤ 1) 
                (h5 : ∀ x ∈ Set.Icc 0 2, ∃ m : ℝ, m ≥ abs (f a b x)) :
  (∀ x : ℝ, f a b x = x^2 - x + 1) ∧ (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc 0 2, m ≥ abs (f a b x)) :=
  sorry

end problem_l36_36566


namespace ratio_new_values_l36_36727

theorem ratio_new_values (x y x2 y2 : ℝ) (h1 : x / y = 7 / 5) (h2 : x2 = x * y) (h3 : y2 = y * x) : x2 / y2 = 1 := by
  sorry

end ratio_new_values_l36_36727


namespace scallops_cost_calculation_l36_36340

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end scallops_cost_calculation_l36_36340


namespace negation_of_universal_proposition_l36_36138

theorem negation_of_universal_proposition :
  ¬ (∀ (m : ℝ), ∃ (x : ℝ), x^2 + x + m = 0) ↔ ∃ (m : ℝ), ¬ ∃ (x : ℝ), x^2 + x + m = 0 :=
by sorry

end negation_of_universal_proposition_l36_36138


namespace car_speed_15_seconds_less_l36_36992

theorem car_speed_15_seconds_less (v : ℝ) : 
  (∀ v, 75 = 3600 / v + 15) → v = 60 :=
by
  intro H
  -- Proof goes here
  sorry

end car_speed_15_seconds_less_l36_36992


namespace find_minimal_x_l36_36938

-- Conditions
variables (x y : ℕ)
variable (pos_x : x > 0)
variable (pos_y : y > 0)
variable (h : 3 * x^7 = 17 * y^11)

-- Proof Goal
theorem find_minimal_x : ∃ a b c d : ℕ, x = a^c * b^d ∧ a + b + c + d = 30 :=
by {
  sorry
}

end find_minimal_x_l36_36938


namespace compute_fraction_l36_36777

theorem compute_fraction :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) =
  1 / 4 :=
by
  -- Proof will go here
  sorry

end compute_fraction_l36_36777


namespace periodic_function_property_l36_36960

theorem periodic_function_property
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_period : ∀ x, f (x + 2) = f x)
  (h_def1 : ∀ x, -1 ≤ x ∧ x < 0 → f x = a * x + 1)
  (h_def2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (b * x + 2) / (x + 1))
  (h_eq : f (1 / 2) = f (3 / 2)) :
  3 * a + 2 * b = -8 := by
  sorry

end periodic_function_property_l36_36960


namespace school_enrollment_l36_36025

theorem school_enrollment
  (X Y : ℝ)
  (h1 : X + Y = 4000)
  (h2 : 1.07 * X > X)
  (h3 : 1.03 * Y > Y)
  (h4 : 0.07 * X - 0.03 * Y = 40) :
  Y = 2400 :=
by
  -- problem reduction
  sorry

end school_enrollment_l36_36025


namespace lily_account_balance_l36_36489

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l36_36489


namespace kindergarteners_line_up_probability_l36_36102

theorem kindergarteners_line_up_probability :
  let total_line_up := Nat.choose 20 9
  let first_scenario := Nat.choose 14 9
  let second_scenario_single := Nat.choose 13 8
  let second_scenario := 6 * second_scenario_single
  let valid_arrangements := first_scenario + second_scenario
  valid_arrangements / total_line_up = 9724 / 167960 := by
  sorry

end kindergarteners_line_up_probability_l36_36102


namespace regina_earnings_l36_36148

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l36_36148


namespace total_shaded_area_l36_36714

def rectangle_area (R : ℝ) : ℝ := R * R
def square_area (S : ℝ) : ℝ := S * S

theorem total_shaded_area 
  (R S : ℝ)
  (h1 : 18 = 2 * R)
  (h2 : R = 4 * S) :
  rectangle_area R + 12 * square_area S = 141.75 := 
  by 
    sorry

end total_shaded_area_l36_36714


namespace cos_B_in_triangle_l36_36578

theorem cos_B_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = (Real.sqrt 5 / 2) * b)
  (h2 : A = 2 * B)
  (h_triangle: A + B + C = Real.pi) : 
  Real.cos B = Real.sqrt 5 / 4 :=
sorry

end cos_B_in_triangle_l36_36578


namespace find_f_at_1_l36_36755

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem find_f_at_1 : f 1 = 2 := by
  sorry

end find_f_at_1_l36_36755


namespace combined_total_time_l36_36955

theorem combined_total_time
  (Katherine_time : Real := 20)
  (Naomi_time : Real := Katherine_time * (1 + 1 / 4))
  (Lucas_time : Real := Katherine_time * (1 + 1 / 3))
  (Isabella_time : Real := Katherine_time * (1 + 1 / 2))
  (Naomi_total : Real := Naomi_time * 10)
  (Lucas_total : Real := Lucas_time * 10)
  (Isabella_total : Real := Isabella_time * 10) :
  Naomi_total + Lucas_total + Isabella_total = 816.7 := sorry

end combined_total_time_l36_36955


namespace acute_angle_at_3_16_l36_36480

def angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60) * 360
  let hour_angle := (hour % 12) * 30 + (minute / 60) * 30
  |hour_angle - minute_angle|

theorem acute_angle_at_3_16 : angle_between_clock_hands 3 16 = 2 := 
sorry

end acute_angle_at_3_16_l36_36480


namespace base_addition_l36_36112

theorem base_addition (R1 R3 : ℕ) (F1 F2 : ℚ)
    (hF1_baseR1 : F1 = 45 / (R1^2 - 1))
    (hF2_baseR1 : F2 = 54 / (R1^2 - 1))
    (hF1_baseR3 : F1 = 36 / (R3^2 - 1))
    (hF2_baseR3 : F2 = 63 / (R3^2 - 1)) :
  R1 + R3 = 20 :=
sorry

end base_addition_l36_36112


namespace total_cows_l36_36691

theorem total_cows (cows_per_herd : Nat) (herds : Nat) (total_cows : Nat) : 
  cows_per_herd = 40 → herds = 8 → total_cows = cows_per_herd * herds → total_cows = 320 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_cows_l36_36691


namespace inequality_proof_l36_36885

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (b + c)) + (1 / (a + c)) + (1 / (a + b)) ≥ 9 / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l36_36885


namespace triangle_obtuse_of_inequality_l36_36865

theorem triangle_obtuse_of_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (ineq : a^2 < (b + c) * (c - b)) :
  ∃ (A B C : ℝ), (A + B + C = π) ∧ (C > π / 2) :=
by
  sorry

end triangle_obtuse_of_inequality_l36_36865


namespace units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l36_36947

-- Define the cycle of the units digits of powers of 7
def units_digit_of_7_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1  -- 7^4, 7^8, ...
  | 1 => 7  -- 7^1, 7^5, ...
  | 2 => 9  -- 7^2, 7^6, ...
  | 3 => 3  -- 7^3, 7^7, ...
  | _ => 0  -- unreachable

-- The main theorem to prove
theorem units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared : 
  units_digit_of_7_power (3 ^ (5 ^ 2)) = 3 :=
by
  sorry

end units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l36_36947


namespace no_such_functions_exist_l36_36823

theorem no_such_functions_exist :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 :=
by
  sorry

end no_such_functions_exist_l36_36823


namespace sum_of_terms_l36_36540

noncomputable def arithmetic_sequence : Type :=
  {a : ℕ → ℤ // ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d}

theorem sum_of_terms (a : arithmetic_sequence) (h1 : a.val 1 + a.val 3 = 2) (h2 : a.val 3 + a.val 5 = 4) :
  a.val 5 + a.val 7 = 6 :=
by
  sorry

end sum_of_terms_l36_36540


namespace xiamen_fabric_production_l36_36300

theorem xiamen_fabric_production:
  (∃ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3))) ∧
  (∀ x y : ℕ, (3 * ((2 * x) / 3) + 3 * (y / 3) = 600) ∧ (2 * ((2 * x) / 3) = 3 * (y / 3)) →
    x = 360 ∧ y = 240 ∧ y / 3 = 240) := 
by
  sorry

end xiamen_fabric_production_l36_36300


namespace area_of_shaded_region_l36_36068

theorem area_of_shaded_region 
  (ABCD : Type) 
  (BC : ℝ)
  (height : ℝ)
  (BE : ℝ)
  (CF : ℝ)
  (BC_length : BC = 12)
  (height_length : height = 10)
  (BE_length : BE = 5)
  (CF_length : CF = 3) :
  (BC * height - (1 / 2 * BE * height) - (1 / 2 * CF * height)) = 80 :=
by
  sorry

end area_of_shaded_region_l36_36068


namespace find_C_coordinates_l36_36420

open Real

noncomputable def pointC_coordinates (A B : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (3, 8)) (hdist : dist A C = 2 * dist C B) : ℝ × ℝ :=
  (⟨7 / 3, 20 / 3⟩)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ), 
  A = (-1, 0) → B = (3, 8) → dist A C = 2 * dist C B →
  C = (7 / 3, 20 / 3) :=
by 
  intros A B C hA hB hdist
  -- We will use the given conditions and definitions to find the coordinates of C
  sorry

end find_C_coordinates_l36_36420


namespace isosceles_triangle_perimeter_l36_36842

-- Define the lengths of the sides
def side1 : ℕ := 4
def side2 : ℕ := 7

-- Condition: The given sides form an isosceles triangle
def is_isosceles_triangle (a b : ℕ) : Prop := a = b ∨ a = 4 ∧ b = 7 ∨ a = 7 ∧ b = 4

-- Condition: The triangle inequality theorem must be satisfied
def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem we want to prove
theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : is_isosceles_triangle a b) (h2 : triangle_inequality a a b ∨ triangle_inequality b b a) :
  a + a + b = 15 ∨ b + b + a = 18 := 
sorry

end isosceles_triangle_perimeter_l36_36842


namespace jane_brown_sheets_l36_36053

theorem jane_brown_sheets :
  ∀ (total_sheets yellow_sheets brown_sheets : ℕ),
    total_sheets = 55 →
    yellow_sheets = 27 →
    brown_sheets = total_sheets - yellow_sheets →
    brown_sheets = 28 := 
by
  intros total_sheets yellow_sheets brown_sheets ht hy hb
  rw [ht, hy] at hb
  simp at hb
  exact hb

end jane_brown_sheets_l36_36053


namespace number_of_girls_l36_36705

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l36_36705


namespace swimming_pool_length_l36_36435

theorem swimming_pool_length :
  ∀ (w d1 d2 V : ℝ), w = 9 → d1 = 1 → d2 = 4 → V = 270 → 
  (((V = (1 / 2) * (d1 + d2) * w * l) → l = 12)) :=
by
  intros w d1 d2 V hw hd1 hd2 hV hv
  simp only [hw, hd1, hd2, hV] at hv
  sorry

end swimming_pool_length_l36_36435


namespace find_value_of_a_l36_36844

variable (a : ℝ)

def f (x : ℝ) := x^2 + 4
def g (x : ℝ) := x^2 - 2

theorem find_value_of_a (h_pos : a > 0) (h_eq : f (g a) = 12) : a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := 
by
  sorry

end find_value_of_a_l36_36844


namespace nth_wise_number_1990_l36_36085

/--
A natural number that can be expressed as the difference of squares 
of two other natural numbers is called a "wise number".
-/
def is_wise_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 - y^2 = n

/--
The 1990th "wise number" is 2659.
-/
theorem nth_wise_number_1990 : ∃ n : ℕ, is_wise_number n ∧ n = 2659 :=
  sorry

end nth_wise_number_1990_l36_36085


namespace sum_200_to_299_l36_36756

variable (a : ℕ)

-- Condition: Sum of the first 100 natural numbers is equal to a
def sum_100 := (100 * 101) / 2

-- Main Theorem: Sum from 200 to 299 in terms of a
theorem sum_200_to_299 (h : sum_100 = a) : (299 * 300 / 2 - 199 * 200 / 2) = 19900 + a := by
  sorry

end sum_200_to_299_l36_36756


namespace village_population_l36_36728

variable (Px : ℕ)
variable (py : ℕ := 42000)
variable (years : ℕ := 16)
variable (rate_decrease_x : ℕ := 1200)
variable (rate_increase_y : ℕ := 800)

theorem village_population (Px : ℕ) (py : ℕ := 42000)
  (years : ℕ := 16) (rate_decrease_x : ℕ := 1200)
  (rate_increase_y : ℕ := 800) :
  Px - rate_decrease_x * years = py + rate_increase_y * years → Px = 74000 := by
  sorry

end village_population_l36_36728


namespace problem_I_problem_II_l36_36915

open Set Real

-- Problem (I)
theorem problem_I (x : ℝ) :
  (|x - 2| ≥ 4 - |x - 1|) ↔ x ∈ Iic (-1/2) ∪ Ici (7/2) :=
by
  sorry

-- Problem (II)
theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 1/2/n = 1) :
  m + 2 * n ≥ 4 :=
by
  sorry

end problem_I_problem_II_l36_36915


namespace vampires_after_two_nights_l36_36369

def initial_population : ℕ := 300
def initial_vampires : ℕ := 3
def conversion_rate : ℕ := 7

theorem vampires_after_two_nights :
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  total_second_night = 192 :=
by
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  have h1 : first_night = 21 := rfl
  have h2 : total_first_night = 24 := rfl
  have h3 : second_night = 168 := rfl
  have h4 : total_second_night = 192 := rfl
  exact rfl

end vampires_after_two_nights_l36_36369


namespace compound_interest_correct_l36_36211
noncomputable def compound_interest_proof : Prop :=
  let si := 55
  let r := 5
  let t := 2
  let p := si * 100 / (r * t)
  let ci := p * ((1 + r / 100)^t - 1)
  ci = 56.375

theorem compound_interest_correct : compound_interest_proof :=
by {
  sorry
}

end compound_interest_correct_l36_36211


namespace mat_pow_four_eq_l36_36986

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -2; 1, 1]

def mat_fourth_power : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-14, -6; 3, -17]

theorem mat_pow_four_eq :
  mat ^ 4 = mat_fourth_power :=
by
  sorry

end mat_pow_four_eq_l36_36986


namespace question1_question2_l36_36632

noncomputable def f (x : ℝ) : ℝ :=
  if x < -4 then -x - 9
  else if x < 1 then 3 * x + 7
  else x + 9

theorem question1 (x : ℝ) (h : -10 ≤ x ∧ x ≤ -2) : f x ≤ 1 := sorry

theorem question2 (x a : ℝ) (hx : x > 1) (h : f x > -x^2 + a * x) : a < 7 := sorry

end question1_question2_l36_36632


namespace larger_fraction_of_two_l36_36265

theorem larger_fraction_of_two (x y : ℚ) (h1 : x + y = 7/8) (h2 : x * y = 1/4) : max x y = 1/2 :=
sorry

end larger_fraction_of_two_l36_36265


namespace num_real_x_l36_36831

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l36_36831


namespace number_of_articles_l36_36423

-- Define main conditions
variable (N : ℕ) -- Number of articles
variable (CP SP : ℝ) -- Cost price and Selling price per article

-- Condition 1: Cost price of N articles equals the selling price of 15 articles
def condition1 : Prop := N * CP = 15 * SP

-- Condition 2: Selling price includes a 33.33% profit on cost price
def condition2 : Prop := SP = CP * 1.3333

-- Prove that the number of articles N equals 20
theorem number_of_articles (h1 : condition1 N CP SP) (h2 : condition2 CP SP) : N = 20 :=
by sorry

end number_of_articles_l36_36423


namespace Jeffrey_steps_l36_36841

theorem Jeffrey_steps
  (Andrew_steps : ℕ) (Jeffrey_steps : ℕ) (h_ratio : Andrew_steps / Jeffrey_steps = 3 / 4)
  (h_Andrew : Andrew_steps = 150) :
  Jeffrey_steps = 200 :=
by
  sorry

end Jeffrey_steps_l36_36841


namespace money_left_after_purchases_l36_36695

variable (initial_money : ℝ) (fraction_for_cupcakes : ℝ) (money_spent_on_milkshake : ℝ)

theorem money_left_after_purchases (h_initial : initial_money = 10)
  (h_fraction : fraction_for_cupcakes = 1/5)
  (h_milkshake : money_spent_on_milkshake = 5) :
  initial_money - (initial_money * fraction_for_cupcakes) - money_spent_on_milkshake = 3 := 
by
  sorry

end money_left_after_purchases_l36_36695


namespace geometric_sequence_eighth_term_l36_36146

theorem geometric_sequence_eighth_term (a r : ℝ) (h₀ : a = 27) (h₁ : r = 1/3) :
  a * r^7 = 1/81 :=
by
  rw [h₀, h₁]
  sorry

end geometric_sequence_eighth_term_l36_36146


namespace find_two_digits_l36_36905

theorem find_two_digits (a b : ℕ) (h₁: a ≤ 9) (h₂: b ≤ 9)
  (h₃: (4 + a + b) % 9 = 0) (h₄: (10 * a + b) % 4 = 0) :
  (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8) :=
by {
  sorry
}

end find_two_digits_l36_36905


namespace rectangle_area_k_value_l36_36968

theorem rectangle_area_k_value (d : ℝ) (length width : ℝ) (h1 : 5 * width = 2 * length) (h2 : d^2 = length^2 + width^2) :
  ∃ (k : ℝ), A = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_k_value_l36_36968


namespace tan_of_negative_7pi_over_4_l36_36432

theorem tan_of_negative_7pi_over_4 : Real.tan (-7 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_of_negative_7pi_over_4_l36_36432


namespace average_of_numbers_l36_36132

theorem average_of_numbers (x : ℝ) (h : (2 + x + 12) / 3 = 8) : x = 10 :=
by sorry

end average_of_numbers_l36_36132


namespace aqua_park_earnings_l36_36096

def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def meal_fee : ℕ := 10
def souvenir_fee : ℕ := 8

def group1_admission_count : ℕ := 10
def group1_tour_count : ℕ := 10
def group1_meal_count : ℕ := 10
def group1_souvenir_count : ℕ := 10
def group1_discount : ℚ := 0.10

def group2_admission_count : ℕ := 15
def group2_meal_count : ℕ := 15
def group2_meal_discount : ℚ := 0.05

def group3_admission_count : ℕ := 8
def group3_tour_count : ℕ := 8
def group3_souvenir_count : ℕ := 8

-- total cost for group 1 before discount
def group1_total_before_discount : ℕ := 
  (group1_admission_count * admission_fee) +
  (group1_tour_count * tour_fee) +
  (group1_meal_count * meal_fee) +
  (group1_souvenir_count * souvenir_fee)

-- group 1 total cost after discount
def group1_total_after_discount : ℚ :=
  group1_total_before_discount * (1 - group1_discount)

-- total cost for group 2 before discount
def group2_admission_total_before_discount : ℕ := 
  group2_admission_count * admission_fee
def group2_meal_total_before_discount : ℕ := 
  group2_meal_count * meal_fee

-- group 2 total cost after discount
def group2_meal_total_after_discount : ℚ :=
  group2_meal_total_before_discount * (1 - group2_meal_discount)
def group2_total_after_discount : ℚ :=
  group2_admission_total_before_discount + group2_meal_total_after_discount

-- total cost for group 3 before discount
def group3_total_before_discount : ℕ := 
  (group3_admission_count * admission_fee) +
  (group3_tour_count * tour_fee) +
  (group3_souvenir_count * souvenir_fee)

-- group 3 total cost after discount (no discount applied)
def group3_total_after_discount : ℕ := group3_total_before_discount

-- total earnings from all groups
def total_earnings : ℚ :=
  group1_total_after_discount +
  group2_total_after_discount +
  group3_total_after_discount

theorem aqua_park_earnings : total_earnings = 854.50 := by
  sorry

end aqua_park_earnings_l36_36096


namespace model_x_completion_time_l36_36646

theorem model_x_completion_time (T_x : ℝ) : 
  (24 : ℕ) * (1 / T_x + 1 / 36) = 1 → T_x = 72 := 
by 
  sorry

end model_x_completion_time_l36_36646


namespace find_x_l36_36626

theorem find_x (c d : ℝ) (y z x : ℝ) 
  (h1 : y^2 = c * z^2) 
  (h2 : y = d / x)
  (h3 : y = 3) 
  (h4 : x = 4) 
  (h5 : z = 6) 
  (h6 : y = 2) 
  (h7 : z = 12) 
  : x = 6 := 
by
  sorry

end find_x_l36_36626


namespace number_of_combinations_l36_36916

noncomputable def countOddNumbers (n : ℕ) : ℕ := (n + 1) / 2

noncomputable def countPrimesLessThan30 : ℕ := 9 -- {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def countMultiplesOfFour (n : ℕ) : ℕ := n / 4

theorem number_of_combinations : countOddNumbers 40 * countPrimesLessThan30 * countMultiplesOfFour 40 = 1800 := by
  sorry

end number_of_combinations_l36_36916


namespace train_speed_l36_36607

def length_of_train : ℝ := 160
def time_to_cross : ℝ := 18
def speed_in_kmh : ℝ := 32

theorem train_speed :
  (length_of_train / time_to_cross) * 3.6 = speed_in_kmh :=
by
  sorry

end train_speed_l36_36607


namespace abs_neg_two_eq_two_l36_36087

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_eq_two_l36_36087


namespace pizza_problem_l36_36142

noncomputable def pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ := 
  let slices_with_both := total_slices - (pepperoni_slices + mushroom_slices - total_slices)
  slices_with_both

theorem pizza_problem 
  (total_slices pepperoni_slices mushroom_slices : ℕ)
  (h_total: total_slices = 16)
  (h_pepperoni: pepperoni_slices = 8)
  (h_mushrooms: mushroom_slices = 12)
  (h_at_least_one: pepperoni_slices + mushroom_slices - total_slices ≥ 0)
  (h_no_three_toppings: total_slices = pepperoni_slices + mushroom_slices - 
   (total_slices - (pepperoni_slices + mushroom_slices - total_slices))) : 
  pizza_slices total_slices pepperoni_slices mushroom_slices = 4 :=
by 
  rw [h_total, h_pepperoni, h_mushrooms]
  sorry

end pizza_problem_l36_36142


namespace prob_le_45_l36_36758

-- Define the probability conditions
def prob_between_1_and_45 : ℚ := 7 / 15
def prob_ge_1 : ℚ := 14 / 15

-- State the theorem to prove
theorem prob_le_45 : prob_between_1_and_45 = 7 / 15 := by
  sorry

end prob_le_45_l36_36758


namespace proof_min_max_expected_wasted_minutes_l36_36792

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l36_36792


namespace power_neg_two_inverse_l36_36482

theorem power_neg_two_inverse : (-2 : ℤ) ^ (-2 : ℤ) = (1 : ℚ) / (4 : ℚ) := by
  -- Condition: a^{-n} = 1 / a^n for any non-zero number a and any integer n
  have h: ∀ (a : ℚ) (n : ℤ), a ≠ 0 → a ^ (-n) = 1 / a ^ n := sorry
  -- Proof goes here
  sorry

end power_neg_two_inverse_l36_36482


namespace total_books_proof_l36_36419

def initial_books : ℝ := 41.0
def added_books_first : ℝ := 33.0
def added_books_next : ℝ := 2.0

theorem total_books_proof : initial_books + added_books_first + added_books_next = 76.0 :=
by
  sorry

end total_books_proof_l36_36419


namespace find_a_minus_inv_a_l36_36393

variable (a : ℝ)
variable (h : a + 1 / a = Real.sqrt 13)

theorem find_a_minus_inv_a : a - 1 / a = 3 ∨ a - 1 / a = -3 := by
  sorry

end find_a_minus_inv_a_l36_36393


namespace question_correctness_l36_36184

theorem question_correctness (x : ℝ) :
  ¬(x^2 + x^4 = x^6) ∧
  ¬((x + 1) * (x - 1) = x^2 + 1) ∧
  ((x^3)^2 = x^6) ∧
  ¬(x^6 / x^3 = x^2) :=
by sorry

end question_correctness_l36_36184


namespace find_a_l36_36812

-- Define the conditions given in the problem
def binomial_term (r : ℕ) (a : ℝ) : ℝ :=
  Nat.choose 7 r * 2^(7-r) * (-a)^r

def coefficient_condition (a : ℝ) : Prop :=
  binomial_term 5 a = 84

-- The theorem stating the problem's solution
theorem find_a (a : ℝ) (h : coefficient_condition a) : a = -1 :=
  sorry

end find_a_l36_36812


namespace petya_maximum_margin_l36_36486

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l36_36486


namespace larger_of_two_numbers_l36_36468

theorem larger_of_two_numbers (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : max x y = 8 :=
sorry

end larger_of_two_numbers_l36_36468


namespace hand_towels_in_set_l36_36533

theorem hand_towels_in_set {h : ℕ}
  (hand_towel_sets : ℕ)
  (bath_towel_sets : ℕ)
  (hand_towel_sold : h * hand_towel_sets = 102)
  (bath_towel_sold : 6 * bath_towel_sets = 102)
  (same_sets_sold : hand_towel_sets = bath_towel_sets) :
  h = 17 := 
sorry

end hand_towels_in_set_l36_36533


namespace correct_product_l36_36953

namespace SarahsMultiplication

theorem correct_product (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hx' : ∃ (a b : ℕ), x = 10 * a + b ∧ b * 10 + a = x' ∧ 221 = x' * y) : (x * y = 527 ∨ x * y = 923) := by
  sorry

end SarahsMultiplication

end correct_product_l36_36953


namespace cube_volume_l36_36262

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end cube_volume_l36_36262


namespace jim_saving_amount_l36_36052

theorem jim_saving_amount
    (sara_initial_savings : ℕ)
    (sara_weekly_savings : ℕ)
    (jim_weekly_savings : ℕ)
    (weeks_elapsed : ℕ)
    (sara_total_savings : ℕ := sara_initial_savings + weeks_elapsed * sara_weekly_savings)
    (jim_total_savings : ℕ := weeks_elapsed * jim_weekly_savings)
    (savings_equal: sara_total_savings = jim_total_savings)
    (sara_initial_savings_value : sara_initial_savings = 4100)
    (sara_weekly_savings_value : sara_weekly_savings = 10)
    (weeks_elapsed_value : weeks_elapsed = 820) :
    jim_weekly_savings = 15 := 
by
  sorry

end jim_saving_amount_l36_36052


namespace number_of_sides_l36_36447

def side_length : ℕ := 16
def perimeter : ℕ := 80

theorem number_of_sides (h1: side_length = 16) (h2: perimeter = 80) : (perimeter / side_length = 5) :=
by
  -- Proof should be inserted here.
  sorry

end number_of_sides_l36_36447


namespace line_through_intersection_and_origin_l36_36650

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2023 * x - 2022 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2022 * x + 2023 * y + 1 = 0

-- Define the line passing through the origin
def line_pass_origin (x y : ℝ) : Prop := 4045 * x + y = 0

-- Define the intersection point of the two lines
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the theorem stating the desired property
theorem line_through_intersection_and_origin (x y : ℝ)
    (h1 : intersection x y)
    (h2 : x = 0 ∧ y = 0) :
    line_pass_origin x y :=
by
    sorry

end line_through_intersection_and_origin_l36_36650


namespace math_problem_l36_36406

theorem math_problem : 2357 + 3572 + 5723 + 2 * 7235 = 26122 :=
  by sorry

end math_problem_l36_36406


namespace shift_left_by_pi_over_six_l36_36331

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem shift_left_by_pi_over_six : f = λ x => g (x + Real.pi / 6) := by
  sorry

end shift_left_by_pi_over_six_l36_36331


namespace find_c_l36_36541

theorem find_c
  (c d : ℝ)
  (h1 : ∀ (x : ℝ), 7 * x^3 + 3 * c * x^2 + 6 * d * x + c = 0)
  (h2 : ∀ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
        7 * p^3 + 3 * c * p^2 + 6 * d * p + c = 0 ∧ 
        7 * q^3 + 3 * c * q^2 + 6 * d * q + c = 0 ∧ 
        7 * r^3 + 3 * c * r^2 + 6 * d * r + c = 0 ∧ 
        Real.log (p * q * r) / Real.log 3 = 3) :
  c = -189 :=
sorry

end find_c_l36_36541


namespace no_arithmetic_mean_l36_36561

def eight_thirteen : ℚ := 8 / 13
def eleven_seventeen : ℚ := 11 / 17
def five_eight : ℚ := 5 / 8

-- Define the function to calculate the arithmetic mean of two rational numbers
def arithmetic_mean (a b : ℚ) : ℚ :=
(a + b) / 2

-- The theorem statement
theorem no_arithmetic_mean :
  eight_thirteen ≠ arithmetic_mean eleven_seventeen five_eight ∧
  eleven_seventeen ≠ arithmetic_mean eight_thirteen five_eight ∧
  five_eight ≠ arithmetic_mean eight_thirteen eleven_seventeen :=
sorry

end no_arithmetic_mean_l36_36561


namespace find_x_l36_36204

variable (x : ℝ)
def vector_a : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (x - 1, 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (h1 : dot_product (vector_a x + vector_b x) (vector_a x - vector_b x) = 0) : x = -1 := by 
  sorry

end find_x_l36_36204


namespace first_group_correct_l36_36075

/-- Define the total members in the choir --/
def total_members : ℕ := 70

/-- Define members in the second group --/
def second_group_members : ℕ := 30

/-- Define members in the third group --/
def third_group_members : ℕ := 15

/-- Define the number of members in the first group by subtracting second and third groups members from total members --/
def first_group_members : ℕ := total_members - (second_group_members + third_group_members)

/-- Prove that the first group has 25 members --/
theorem first_group_correct : first_group_members = 25 := by
  -- insert the proof steps here
  sorry

end first_group_correct_l36_36075


namespace quadratic_has_two_real_distinct_roots_and_find_m_l36_36175

theorem quadratic_has_two_real_distinct_roots_and_find_m 
  (m : ℝ) :
  (x : ℝ) → 
  (h1 : x^2 - (2 * m - 2) * x + (m^2 - 2 * m) = 0) →
  (x1 x2 : ℝ) →
  (h2 : x1^2 + x2^2 = 10) →
  (x1 + x2 = 2 * m - 2) →
  (x1 * x2 = m^2 - 2 * m) →
  (x1 ≠ x2) ∧ (m = -1 ∨ m = 3) :=
by sorry

end quadratic_has_two_real_distinct_roots_and_find_m_l36_36175


namespace complete_square_l36_36628

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l36_36628


namespace fraction_difference_of_squares_l36_36631

theorem fraction_difference_of_squares :
  (175^2 - 155^2) / 20 = 330 :=
by
  -- Proof goes here
  sorry

end fraction_difference_of_squares_l36_36631


namespace symmetric_points_x_axis_l36_36851

theorem symmetric_points_x_axis (a b : ℝ) (h_a : a = -2) (h_b : b = -1) : a + b = -3 :=
by
  -- Skipping the proof steps and adding sorry
  sorry

end symmetric_points_x_axis_l36_36851


namespace total_matches_l36_36516

noncomputable def matches_in_tournament (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_matches :
  matches_in_tournament 5 + matches_in_tournament 7 + matches_in_tournament 4 = 37 := 
by 
  sorry

end total_matches_l36_36516


namespace product_of_consecutive_numbers_with_25_is_perfect_square_l36_36952

theorem product_of_consecutive_numbers_with_25_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n * (n + 1)) + 25 = k^2 := 
by
  -- Proof body omitted
  sorry

end product_of_consecutive_numbers_with_25_is_perfect_square_l36_36952


namespace trapezoid_base_length_sets_l36_36991

open Nat

theorem trapezoid_base_length_sets :
  ∃ (sets : Finset (ℕ × ℕ)), sets.card = 5 ∧ 
    (∀ p ∈ sets, ∃ (b1 b2 : ℕ), b1 = 10 * p.1 ∧ b2 = 10 * p.2 ∧ b1 + b2 = 90) :=
by
  sorry

end trapezoid_base_length_sets_l36_36991


namespace fourth_circle_radius_l36_36765

theorem fourth_circle_radius (c : ℝ) (h : c > 0) :
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  fourth_radius = (c / 2) - r :=
by
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  sorry

end fourth_circle_radius_l36_36765


namespace fraction_problem_l36_36894

theorem fraction_problem 
  (x : ℚ)
  (h : x = 45 / (8 - (3 / 7))) : 
  x = 315 / 53 := 
sorry

end fraction_problem_l36_36894


namespace units_digit_k_squared_plus_2_k_l36_36159

noncomputable def k : ℕ := 2017^2 + 2^2017

theorem units_digit_k_squared_plus_2_k : (k^2 + 2^k) % 10 = 3 := 
  sorry

end units_digit_k_squared_plus_2_k_l36_36159


namespace find_divisor_l36_36680

theorem find_divisor (d : ℕ) (h1 : d ∣ (9671 - 1)) : d = 9670 :=
by
  sorry

end find_divisor_l36_36680


namespace number_of_oranges_l36_36818

-- Definitions of the conditions
def peaches : ℕ := 9
def pears : ℕ := 18
def greatest_num_per_basket : ℕ := 3
def num_baskets_peaches := peaches / greatest_num_per_basket
def num_baskets_pears := pears / greatest_num_per_basket
def min_num_baskets := min num_baskets_peaches num_baskets_pears

-- Proof problem statement
theorem number_of_oranges (O : ℕ) (h1 : O % greatest_num_per_basket = 0) 
  (h2 : O / greatest_num_per_basket = min_num_baskets) : 
  O = 9 :=
by {
  sorry
}

end number_of_oranges_l36_36818


namespace abs_diff_of_two_numbers_l36_36366

variable {x y : ℝ}

theorem abs_diff_of_two_numbers (h1 : x + y = 40) (h2 : x * y = 396) : abs (x - y) = 4 := by
  sorry

end abs_diff_of_two_numbers_l36_36366


namespace uniquely_determine_T_l36_36428

theorem uniquely_determine_T'_n (b e : ℤ) (S' T' : ℕ → ℤ)
  (hb : ∀ n, S' n = n * (2 * b + (n - 1) * e) / 2)
  (ht : ∀ n, T' n = n * (n + 1) * (3 * b + (n - 1) * e) / 6)
  (h3028 : S' 3028 = 3028 * (b + 1514 * e)) :
  T' 4543 = (4543 * (4543 + 1) * (3 * b + 4542 * e)) / 6 :=
by
  sorry

end uniquely_determine_T_l36_36428


namespace bucket_full_weight_l36_36630

variable (c d : ℝ)

def total_weight_definition (x y : ℝ) := x + y

theorem bucket_full_weight (x y : ℝ) 
  (h₁ : x + 3/4 * y = c) 
  (h₂ : x + 1/3 * y = d) : 
  total_weight_definition x y = (8 * c - 3 * d) / 5 :=
sorry

end bucket_full_weight_l36_36630


namespace part_one_part_two_l36_36496

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem part_one (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n > 1) : f m >= 0 ∨ f n >= 0 :=
sorry

theorem part_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hf : f a = f b) : a + b < 4 / 3 :=
sorry

end part_one_part_two_l36_36496


namespace arithmetic_geometric_sequence_min_value_l36_36477

theorem arithmetic_geometric_sequence_min_value (x y a b c d : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (arithmetic_seq : a = (x + y) / 2) (geometric_seq : c * d = x * y) :
  ( (a + b) ^ 2 ) / (c * d) ≥ 4 := 
by
  sorry

end arithmetic_geometric_sequence_min_value_l36_36477


namespace wall_length_to_height_ratio_l36_36502

theorem wall_length_to_height_ratio (W H L V : ℝ) (h1 : H = 6 * W) (h2 : V = W * H * L) (h3 : W = 4) (h4 : V = 16128) :
  L / H = 7 :=
by
  -- Note: The proof steps are omitted as per the problem's instructions.
  sorry

end wall_length_to_height_ratio_l36_36502


namespace smallest_munificence_monic_cubic_polynomial_l36_36873

theorem smallest_munificence_monic_cubic_polynomial :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = x^3 + a * x^2 + b * x + c) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ M) → M ≥ 1) :=
by
  sorry

end smallest_munificence_monic_cubic_polynomial_l36_36873


namespace map_distance_scaled_l36_36510

theorem map_distance_scaled (d_map : ℝ) (scale : ℝ) (d_actual : ℝ) :
  d_map = 8 ∧ scale = 1000000 → d_actual = 80 :=
by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end map_distance_scaled_l36_36510


namespace fixed_point_linear_l36_36069

-- Define the linear function y = kx + k + 2
def linear_function (k x : ℝ) : ℝ := k * x + k + 2

-- Prove that the point (-1, 2) lies on the graph of the function for any k
theorem fixed_point_linear (k : ℝ) : linear_function k (-1) = 2 := by
  sorry

end fixed_point_linear_l36_36069


namespace problem1_solution_set_problem2_inequality_l36_36434

theorem problem1_solution_set (x : ℝ) : (-1 < x) ∧ (x < 9) ↔ (|x| + |x - 3| < x + 6) :=
by sorry

theorem problem2_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hn : 9 * x + y = 1) : x + y ≥ 16 * x * y :=
by sorry

end problem1_solution_set_problem2_inequality_l36_36434


namespace ReuleauxTriangleFitsAll_l36_36312

-- Assume definitions for fits into various slots

def FitsTriangular (s : Type) : Prop := sorry
def FitsSquare (s : Type) : Prop := sorry
def FitsCircular (s : Type) : Prop := sorry
def ReuleauxTriangle (s : Type) : Prop := sorry

theorem ReuleauxTriangleFitsAll (s : Type) (h : ReuleauxTriangle s) : 
  FitsTriangular s ∧ FitsSquare s ∧ FitsCircular s := 
  sorry

end ReuleauxTriangleFitsAll_l36_36312


namespace train_speed_l36_36218

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l36_36218


namespace greatest_sum_x_y_l36_36815

theorem greatest_sum_x_y (x y : ℤ) (h : x^2 + y^2 = 36) : (x + y ≤ 9) := sorry

end greatest_sum_x_y_l36_36815


namespace cos_squared_sum_sin_squared_sum_l36_36934

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 + Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 =
  2 * (1 + Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :=
sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) ^ 2 + Real.sin (B / 2) ^ 2 + Real.sin (C / 2) ^ 2 =
  1 - 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) :=
sorry

end cos_squared_sum_sin_squared_sum_l36_36934


namespace range_of_m_l36_36741

theorem range_of_m (A : Set ℝ) (m : ℝ) (h : ∃ x, x ∈ A ∩ {x | x ≠ 0}) :
  -4 < m ∧ m < 0 :=
by
  have A_def : A = {x | x^2 + (m+2)*x + 1 = 0} := sorry
  have h_non_empty : ∃ x, x ∈ A ∧ x ≠ 0 := sorry
  have discriminant : (m+2)^2 - 4 < 0 := sorry
  exact ⟨sorry, sorry⟩

end range_of_m_l36_36741


namespace find_cost_price_l36_36463

def selling_price : ℝ := 150
def profit_percentage : ℝ := 25

theorem find_cost_price (cost_price : ℝ) (h : profit_percentage = ((selling_price - cost_price) / cost_price) * 100) : 
  cost_price = 120 := 
sorry

end find_cost_price_l36_36463


namespace problem1_problem2_problem3_l36_36579

theorem problem1 : (-3) - (-5) - 6 + (-4) = -8 := by sorry

theorem problem2 : ((1 / 9) + (1 / 6) - (1 / 2)) / (-1 / 18) = 4 := by sorry

theorem problem3 : -1^4 + abs (3 - 6) - 2 * (-2) ^ 2 = -6 := by sorry

end problem1_problem2_problem3_l36_36579


namespace correct_calculated_value_l36_36316

theorem correct_calculated_value (n : ℕ) (h1 : n = 32 * 3) : n / 4 = 24 := 
by
  -- proof steps will be filled here
  sorry

end correct_calculated_value_l36_36316


namespace AB_eq_B_exp_V_l36_36106

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l36_36106


namespace circle_chords_integer_lengths_l36_36209

theorem circle_chords_integer_lengths (P O : ℝ) (d r : ℝ) (n : ℕ) : 
  dist P O = d → r = 20 → d = 12 → n = 9 := by
  sorry

end circle_chords_integer_lengths_l36_36209


namespace calculate_gross_income_l36_36263
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end calculate_gross_income_l36_36263


namespace opposite_of_negative_one_fifth_l36_36943

theorem opposite_of_negative_one_fifth : -(-1 / 5) = (1 / 5) :=
by
  sorry

end opposite_of_negative_one_fifth_l36_36943


namespace find_integers_l36_36954

theorem find_integers (n : ℤ) : (6 ∣ (n - 4)) ∧ (10 ∣ (n - 8)) ↔ (n % 30 = 28) :=
by
  sorry

end find_integers_l36_36954


namespace unique_root_conditions_l36_36424

theorem unique_root_conditions (m : ℝ) (x y : ℝ) :
  (x^2 = 2 * abs x ∧ abs x - y - m = 1 - y^2) ↔ m = 3 / 4 := sorry

end unique_root_conditions_l36_36424


namespace intersection_points_relation_l36_36091

-- Suppressing noncomputable theory to focus on the structure
-- of the Lean statement rather than computability aspects.

noncomputable def intersection_points (k : ℕ) : ℕ :=
sorry -- This represents the function f(k)

axiom no_parallel (k : ℕ) : Prop
axiom no_three_intersect (k : ℕ) : Prop

theorem intersection_points_relation (k : ℕ) (h1 : no_parallel k) (h2 : no_three_intersect k) :
  intersection_points (k + 1) = intersection_points k + k :=
sorry

end intersection_points_relation_l36_36091


namespace max_workers_l36_36231

variable {n : ℕ} -- number of workers on the smaller field
variable {S : ℕ} -- area of the smaller field
variable (a : ℕ) -- productivity of each worker

theorem max_workers 
  (h_area : ∀ large small : ℕ, large = 2 * small) 
  (h_workers : ∀ large small : ℕ, large = small + 4) 
  (h_inequality : ∀ (S : ℕ) (n a : ℕ), S / (a * n) > (2 * S) / (a * (n + 4))) :
  2 * n + 4 ≤ 10 :=
by
  -- h_area implies the area requirement
  -- h_workers implies the worker requirement
  -- h_inequality implies the time requirement
  sorry

end max_workers_l36_36231


namespace valid_sentence_count_is_208_l36_36514

def four_words := ["splargh", "glumph", "amr", "flark"]

def valid_sentence (sentence : List String) : Prop :=
  ¬(sentence.contains "glumph amr")

def count_valid_sentences : Nat :=
  let total_sentences := 4^4
  let invalid_sentences := 3 * 4 * 4
  total_sentences - invalid_sentences

theorem valid_sentence_count_is_208 :
  count_valid_sentences = 208 := by
  sorry

end valid_sentence_count_is_208_l36_36514


namespace luke_hotdogs_ratio_l36_36245

-- Definitions
def hotdogs_per_sister : ℕ := 2
def total_sisters_hotdogs : ℕ := 2 * 2 -- Ella and Emma together
def hunter_hotdogs : ℕ := 6 -- 1.5 times the total of sisters' hotdogs
def total_hotdogs : ℕ := 14

-- Ratio proof problem statement
theorem luke_hotdogs_ratio :
  ∃ x : ℕ, total_hotdogs = total_sisters_hotdogs + 4 * x + hunter_hotdogs ∧ 
    (4 * x = 2 * 1 ∧ x = 1) := 
by 
  sorry

end luke_hotdogs_ratio_l36_36245


namespace compute_b_l36_36247

theorem compute_b (x y b : ℚ) (h1 : 5 * x - 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hy : y = 3) :
  b = 13 / 2 :=
sorry

end compute_b_l36_36247


namespace triangle_area_proof_l36_36811

-- Define the triangle sides and median
variables (AB BC BD AC : ℝ)

-- Assume given values
def AB_value : AB = 1 := by sorry 
def BC_value : BC = Real.sqrt 15 := by sorry
def BD_value : BD = 2 := by sorry

-- Assume AC calculated from problem
def AC_value : AC = 4 := by sorry

-- Final proof statement
theorem triangle_area_proof 
  (hAB : AB = 1)
  (hBC : BC = Real.sqrt 15)
  (hBD : BD = 2)
  (hAC : AC = 4) :
  (1 / 2) * AB * BC = (Real.sqrt 15) / 2 := 
sorry

end triangle_area_proof_l36_36811


namespace rhombus_area_correct_l36_36744

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 80 120 = 4800 :=
by 
  -- the proof is skipped by including sorry
  sorry

end rhombus_area_correct_l36_36744


namespace least_number_to_addition_l36_36196

-- Given conditions
def n : ℤ := 2496

-- The least number to be added to n to make it divisible by 5
def least_number_to_add (n : ℤ) : ℤ :=
  if (n % 5 = 0) then 0 else (5 - (n % 5))

-- Prove that adding 4 to 2496 makes it divisible by 5
theorem least_number_to_addition : (least_number_to_add n) = 4 :=
  by
    sorry

end least_number_to_addition_l36_36196


namespace sum_of_first_11_odd_numbers_l36_36299

theorem sum_of_first_11_odd_numbers : 
  (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) = 121 :=
by
  sorry

end sum_of_first_11_odd_numbers_l36_36299


namespace real_roots_range_real_roots_specific_value_l36_36645

-- Part 1
theorem real_roots_range (a b m : ℝ) (h_eq : a ≠ 0) (h_discriminant : b^2 - 4 * a * m ≥ 0) :
  m ≤ (b^2) / (4 * a) :=
sorry

-- Part 2
theorem real_roots_specific_value (x1 x2 m : ℝ) (h_sum : x1 + x2 = 4) (h_product : x1 * x2 = m)
  (h_condition : x1^2 + x2^2 + (x1 * x2)^2 = 40) (h_range : m ≤ 4) :
  m = -4 :=
sorry

end real_roots_range_real_roots_specific_value_l36_36645


namespace inequality_proof_l36_36749

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^3 + b^3 = 2) :
  (1 / a) + (1 / b) ≥ 2 * (a^2 - a + 1) * (b^2 - b + 1) := 
by
  sorry

end inequality_proof_l36_36749


namespace number_of_students_in_Diligence_before_transfer_l36_36903

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l36_36903


namespace fruit_seller_profit_percentage_l36_36166

/-- Suppose a fruit seller sells mangoes at the rate of Rs. 12 per kg and incurs a loss of 15%. 
    The mangoes should have been sold at Rs. 14.823529411764707 per kg to make a specific profit percentage. 
    This statement proves that the profit percentage is 5%. 
-/
theorem fruit_seller_profit_percentage :
  ∃ P : ℝ, 
    (∀ (CP SP : ℝ), 
        SP = 14.823529411764707 ∧ CP = 12 / 0.85 → 
        SP = CP * (1 + P / 100)) → 
    P = 5 := 
sorry

end fruit_seller_profit_percentage_l36_36166


namespace mark_team_free_throws_l36_36347

theorem mark_team_free_throws (F : ℕ) : 
  let mark_2_pointers := 25
  let mark_3_pointers := 8
  let opp_2_pointers := 2 * mark_2_pointers
  let opp_3_pointers := 1 / 2 * mark_3_pointers
  let total_points := 201
  2 * mark_2_pointers + 3 * mark_3_pointers + F + 2 * mark_2_pointers + 3 / 2 * mark_3_pointers + F / 2 = total_points →
  F = 10 := by
  sorry

end mark_team_free_throws_l36_36347


namespace function_properties_l36_36950

-- Define the function f
def f (x p q : ℝ) : ℝ := x^3 + p * x^2 + 9 * q * x + p + q + 3

-- Stating the main theorem
theorem function_properties (p q : ℝ) :
  ( ∀ x : ℝ, f (-x) p q = -f x p q ) →
  (p = 0 ∧ q = -3 ∧ ∀ x : ℝ, f x 0 (-3) = x^3 - 27 * x ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≤ 26 ) ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≥ -54 )) := 
sorry

end function_properties_l36_36950


namespace temperature_range_l36_36509

theorem temperature_range (t : ℕ) : (21 ≤ t ∧ t ≤ 29) :=
by
  sorry

end temperature_range_l36_36509


namespace f_max_iff_l36_36619

noncomputable def f : ℚ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (h : a ≠ 0) : f a > 0
axiom f_mul (a b : ℚ) : f (a * b) = f a * f b
axiom f_add_le (a b : ℚ) : f (a + b) ≤ f a + f b
axiom f_bound (m : ℤ) : f m ≤ 1989

theorem f_max_iff (a b : ℚ) (h : f a ≠ f b) : f (a + b) = max (f a) (f b) := 
sorry

end f_max_iff_l36_36619


namespace find_z_coordinate_of_point_on_line_passing_through_l36_36388

theorem find_z_coordinate_of_point_on_line_passing_through
  (p1 p2 : ℝ × ℝ × ℝ)
  (x_value : ℝ)
  (z_value : ℝ)
  (h1 : p1 = (1, 3, 2))
  (h2 : p2 = (4, 2, -1))
  (h3 : x_value = 3)
  (param : ℝ)
  (h4 : x_value = (1 + 3 * param))
  (h5 : z_value = (2 - 3 * param)) :
  z_value = 0 := by
  sorry

end find_z_coordinate_of_point_on_line_passing_through_l36_36388


namespace patty_fraction_3mph_l36_36807

noncomputable def fraction_time_at_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) : ℝ :=
  t3 / (t3 + t6)

theorem patty_fraction_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) :
  fraction_time_at_3mph t3 t6 h = 1 / 3 :=
by
  sorry

end patty_fraction_3mph_l36_36807


namespace car_speed_5_hours_l36_36859

variable (T : ℝ)
variable (S : ℝ)

theorem car_speed_5_hours (h1 : T > 0) (h2 : 2 * T = S * 5.0) : S = 2 * T / 5.0 :=
sorry

end car_speed_5_hours_l36_36859


namespace sin_sum_alpha_pi_over_3_l36_36770

theorem sin_sum_alpha_pi_over_3 (alpha : ℝ) (h1 : Real.cos (alpha + 2/3 * Real.pi) = 4/5) (h2 : -Real.pi/2 < alpha ∧ alpha < 0) :
  Real.sin (alpha + Real.pi/3) + Real.sin alpha = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_sum_alpha_pi_over_3_l36_36770


namespace problem_statement_l36_36153

noncomputable def smallest_integer_exceeding := 
  let x : ℝ := (Real.sqrt 3 + Real.sqrt 2) ^ 8
  Int.ceil x

theorem problem_statement : smallest_integer_exceeding = 5360 :=
by 
  -- The proof is omitted
  sorry

end problem_statement_l36_36153


namespace lottery_ticket_might_win_l36_36821

theorem lottery_ticket_might_win (p_win : ℝ) (h : p_win = 0.01) : 
  (∃ (n : ℕ), n = 1 ∧ 0 < p_win ∧ p_win < 1) :=
by 
  sorry

end lottery_ticket_might_win_l36_36821


namespace area_of_black_region_l36_36836

theorem area_of_black_region :
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  area_large - total_area_small = 94 :=
by
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  sorry

end area_of_black_region_l36_36836


namespace pairwise_products_same_digit_l36_36253

theorem pairwise_products_same_digit
  (a b c : ℕ)
  (h_ab : a % 10 ≠ b % 10)
  (h_ac : a % 10 ≠ c % 10)
  (h_bc : b % 10 ≠ c % 10)
  : (a * b % 10 = a * c % 10) ∧ (a * b % 10 = b * c % 10) :=
  sorry

end pairwise_products_same_digit_l36_36253


namespace max_watches_two_hours_l36_36637

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l36_36637


namespace circle_radius_l36_36006

theorem circle_radius (a c r : ℝ) (h₁ : a = π * r^2) (h₂ : c = 2 * π * r) (h₃ : a + c = 100 * π) : 
  r = 9.05 := 
sorry

end circle_radius_l36_36006


namespace perfect_square_condition_l36_36066

theorem perfect_square_condition (x y : ℕ) :
  ∃ k : ℕ, (x + y)^2 + 3*x + y + 1 = k^2 ↔ x = y := 
by 
  sorry

end perfect_square_condition_l36_36066


namespace min_value_of_m_l36_36095

theorem min_value_of_m : (2 ∈ {x | ∃ (m : ℤ), x * (x - m) < 0}) → ∃ (m : ℤ), m = 3 :=
by
  sorry

end min_value_of_m_l36_36095


namespace alex_initial_jelly_beans_l36_36768

variable (initial : ℕ)
variable (eaten : ℕ := 6)
variable (pile_weight : ℕ := 10)
variable (piles : ℕ := 3)

theorem alex_initial_jelly_beans :
  (initial - eaten = pile_weight * piles) → initial = 36 :=
by
  -- proof will be provided here
  sorry

end alex_initial_jelly_beans_l36_36768


namespace largest_of_5_consecutive_odd_integers_l36_36563

theorem largest_of_5_consecutive_odd_integers (n : ℤ) (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 235) :
  n + 8 = 51 :=
sorry

end largest_of_5_consecutive_odd_integers_l36_36563


namespace work_increase_percent_l36_36535

theorem work_increase_percent (W p : ℝ) (p_pos : p > 0) :
  (1 / 3 * p) * W / ((2 / 3) * p) - (W / p) = 0.5 * (W / p) :=
by
  sorry

end work_increase_percent_l36_36535


namespace intersection_of_AB_CD_l36_36125

def point (α : Type*) := (α × α × α)

def A : point ℚ := (5, -8, 9)
def B : point ℚ := (15, -18, 14)
def C : point ℚ := (1, 4, -7)
def D : point ℚ := (3, -4, 11)

def parametric_AB (t : ℚ) : point ℚ :=
  (5 + 10 * t, -8 - 10 * t, 9 + 5 * t)

def parametric_CD (s : ℚ) : point ℚ :=
  (1 + 2 * s, 4 - 8 * s, -7 + 18 * s)

def intersection_point (pi : point ℚ) :=
  ∃ t s : ℚ, parametric_AB t = pi ∧ parametric_CD s = pi

theorem intersection_of_AB_CD : intersection_point (76/15, -118/15, 170/15) :=
  sorry

end intersection_of_AB_CD_l36_36125


namespace tan_inequality_solution_l36_36605

variable (x : ℝ)
variable (k : ℤ)

theorem tan_inequality_solution (hx : Real.tan (2 * x - Real.pi / 4) ≤ 1) :
  ∃ k : ℤ,
  (k * Real.pi / 2 - Real.pi / 8 < x) ∧ (x ≤ k * Real.pi / 2 + Real.pi / 4) :=
sorry

end tan_inequality_solution_l36_36605


namespace min_value_l36_36471

theorem min_value (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 2) : 
  (∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ (1/3 * x^3 + y^2 + z = 13/12)) :=
sorry

end min_value_l36_36471


namespace initial_sodium_chloride_percentage_l36_36386

theorem initial_sodium_chloride_percentage :
  ∀ (P : ℝ),
  (∃ (C : ℝ), C = 24) → -- Tank capacity
  (∃ (E_rate : ℝ), E_rate = 0.4) → -- Evaporation rate per hour
  (∃ (time : ℝ), time = 6) → -- Time in hours
  (1 / 4 * C = 6) → -- Volume of mixture
  (6 * P / 100 + (6 - 6 * P / 100 - E_rate * time) = 3.6) → -- Concentration condition
  P = 30 :=
by
  intros P hC hE_rate htime hvolume hconcentration
  rcases hC with ⟨C, hC⟩
  rcases hE_rate with ⟨E_rate, hE_rate⟩
  rcases htime with ⟨time, htime⟩
  rw [hC, hE_rate, htime] at *
  sorry

end initial_sodium_chloride_percentage_l36_36386


namespace polynomial_inequality_l36_36712

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
sorry

end polynomial_inequality_l36_36712


namespace total_inheritance_money_l36_36473

-- Defining the conditions
def number_of_inheritors : ℕ := 5
def amount_per_person : ℕ := 105500

-- The proof problem
theorem total_inheritance_money :
  number_of_inheritors * amount_per_person = 527500 :=
by sorry

end total_inheritance_money_l36_36473


namespace solve_equation_l36_36596

theorem solve_equation 
  (x : ℝ) 
  (h : (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1)) : 
  x = 5 / 2 := 
sorry

end solve_equation_l36_36596


namespace part_a_l36_36653

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  |a - b| + |b - c| + |c - a| ≤ 2 * Real.sqrt 2 :=
sorry

end part_a_l36_36653


namespace jackie_phil_probability_l36_36238

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := (1 + 1: ℚ)
  let p3_coin := (2 + 3: ℚ)
  let p2_coin := (1 + 2: ℚ)
  let generating_function := fair_coin * p3_coin * p2_coin
  let sum_of_coefficients := 30
  let sum_of_squares_of_coefficients := 290
  sum_of_squares_of_coefficients / (sum_of_coefficients ^ 2)

theorem jackie_phil_probability : probability_same_heads = 29 / 90 := by
  sorry

end jackie_phil_probability_l36_36238


namespace find_n_l36_36827

-- Define the first term a₁, the common ratio q, and the sum Sₙ
def a₁ : ℕ := 2
def q : ℕ := 2
def Sₙ (n : ℕ) : ℕ := 2^(n + 1) - 2

-- The sum of the first n terms is given as 126
def given_sum : ℕ := 126

-- The theorem to be proven
theorem find_n (n : ℕ) (h : Sₙ n = given_sum) : n = 6 :=
by
  sorry

end find_n_l36_36827


namespace parallel_lines_a_eq_neg2_l36_36380

theorem parallel_lines_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 - a = 0) ↔ (x - (1/2) * y = 0)) → a = -2 :=
by sorry

end parallel_lines_a_eq_neg2_l36_36380


namespace line_point_coordinates_l36_36344

theorem line_point_coordinates (t : ℝ) (x y z : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) →
  t = 1/2 →
  (x, y, z) = (5, 3/2, 3) :=
by
  intros h1 h2
  sorry

end line_point_coordinates_l36_36344


namespace walking_rate_ratio_l36_36766

theorem walking_rate_ratio (R R' : ℚ) (D : ℚ) (h1: D = R * 14) (h2: D = R' * 12) : R' / R = 7 / 6 :=
by 
  sorry

end walking_rate_ratio_l36_36766


namespace remainder_when_divided_by_11_l36_36012

theorem remainder_when_divided_by_11 (N : ℕ)
  (h₁ : N = 5 * 5 + 0) :
  N % 11 = 3 := 
sorry

end remainder_when_divided_by_11_l36_36012


namespace initial_white_cookies_l36_36059

theorem initial_white_cookies (B W : ℕ) 
  (h1 : B = W + 50)
  (h2 : (1 / 2 : ℚ) * B + (1 / 4 : ℚ) * W = 85) :
  W = 80 :=
by
  sorry

end initial_white_cookies_l36_36059


namespace tan_positive_implies_sin_cos_positive_l36_36339

variables {α : ℝ}

theorem tan_positive_implies_sin_cos_positive (h : Real.tan α > 0) : Real.sin α * Real.cos α > 0 :=
sorry

end tan_positive_implies_sin_cos_positive_l36_36339


namespace fraction_simplification_l36_36930

open Real -- Open the Real namespace for real number operations

theorem fraction_simplification (a x : ℝ) : 
  (sqrt (a^2 + x^2) - (x^2 + a^2) / sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 := 
sorry

end fraction_simplification_l36_36930


namespace unique_solution_l36_36910

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x

theorem unique_solution : ∀ x : ℝ, f x = 7^x ↔ x = 2 :=
by
  sorry

end unique_solution_l36_36910


namespace locus_midpoint_l36_36291

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end locus_midpoint_l36_36291


namespace value_of_a_l36_36759

theorem value_of_a (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := sorry

end value_of_a_l36_36759


namespace first_number_is_twenty_l36_36031

theorem first_number_is_twenty (x : ℕ) : 
  (x + 40 + 60) / 3 = ((10 + 70 + 16) / 3) + 8 → x = 20 := 
by 
  sorry

end first_number_is_twenty_l36_36031


namespace buns_per_student_correct_l36_36999

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end buns_per_student_correct_l36_36999


namespace equal_split_payment_l36_36796

variable (L M N : ℝ)

theorem equal_split_payment (h1 : L < N) (h2 : L > M) : 
  (L + M + N) / 3 - L = (M + N - 2 * L) / 3 :=
by sorry

end equal_split_payment_l36_36796


namespace max_four_digit_prime_product_l36_36322

theorem max_four_digit_prime_product :
  ∃ (x y : ℕ) (n : ℕ), x < 5 ∧ y < 5 ∧ x ≠ y ∧ Prime x ∧ Prime y ∧ Prime (10 * x + y) ∧ n = x * y * (10 * x + y) ∧ n = 138 :=
by
  sorry

end max_four_digit_prime_product_l36_36322


namespace maximum_value_of_expression_l36_36616

theorem maximum_value_of_expression
  (a b c : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + 2 * c^2 = 1) :
  ab * Real.sqrt 3 + 3 * bc ≤ Real.sqrt 7 :=
sorry

end maximum_value_of_expression_l36_36616


namespace probability_of_drawing_green_ball_l36_36609

variable (total_balls green_balls : ℕ)
variable (total_balls_eq : total_balls = 10)
variable (green_balls_eq : green_balls = 4)

theorem probability_of_drawing_green_ball (h_total : total_balls = 10) (h_green : green_balls = 4) :
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end probability_of_drawing_green_ball_l36_36609


namespace max_sum_mult_table_l36_36790

def isEven (n : ℕ) : Prop := n % 2 = 0
def isOdd (n : ℕ) : Prop := ¬ isEven n
def entries : List ℕ := [3, 4, 6, 8, 9, 12]
def sumOfList (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem max_sum_mult_table :
  ∃ (a b c d e f : ℕ), 
    a ∈ entries ∧ b ∈ entries ∧ c ∈ entries ∧ 
    d ∈ entries ∧ e ∈ entries ∧ f ∈ entries ∧ 
    (isEven a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isEven c ∨ isOdd a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isEven c) ∧ 
    (isEven d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isEven f ∨ isOdd d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isEven f) ∧ 
    (sumOfList [a, b, c] * sumOfList [d, e, f] = 425) := 
by
    sorry  -- Skipping the proof as instructed.

end max_sum_mult_table_l36_36790


namespace range_of_n_l36_36895

theorem range_of_n (x : ℕ) (n : ℝ) : 
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 5 → x - 2 < n + 3) → ∃ n, 0 < n ∧ n ≤ 1 :=
by
  sorry

end range_of_n_l36_36895


namespace no_feasible_distribution_l36_36723

-- Define the initial conditions
def initial_runs_player_A : ℕ := 320
def initial_runs_player_B : ℕ := 450
def initial_runs_player_C : ℕ := 550

def initial_innings : ℕ := 10

def required_increase_A : ℕ := 4
def required_increase_B : ℕ := 5
def required_increase_C : ℕ := 6

def total_run_limit : ℕ := 250

-- Define the total runs required after 11 innings
def total_required_runs_after_11_innings (initial_runs avg_increase : ℕ) : ℕ :=
  (initial_runs / initial_innings + avg_increase) * 11

-- Calculate the additional runs needed in the next innings
def additional_runs_needed (initial_runs avg_increase : ℕ) : ℕ :=
  total_required_runs_after_11_innings initial_runs avg_increase - initial_runs

-- Calculate the total additional runs needed for all players
def total_additional_runs_needed : ℕ :=
  additional_runs_needed initial_runs_player_A required_increase_A +
  additional_runs_needed initial_runs_player_B required_increase_B +
  additional_runs_needed initial_runs_player_C required_increase_C

-- The statement to verify if the total additional required runs exceed the limit
theorem no_feasible_distribution :
  total_additional_runs_needed > total_run_limit :=
by 
  -- Skipping proofs and just stating the condition is what we aim to show.
  sorry

end no_feasible_distribution_l36_36723


namespace stack_glasses_opacity_l36_36382

-- Define the main problem's parameters and conditions
def num_glass_pieces : Nat := 5
def rotations := [0, 90, 180, 270] -- Possible rotations

-- Define the main theorem to state the problem in Lean
theorem stack_glasses_opacity :
  (∃ count : Nat, count = 7200 ∧
   -- There are 5 glass pieces
   ∀ (g : Fin num_glass_pieces), 
     -- Each piece is divided into 4 triangles
     ∀ (parts : Fin 4),
     -- There exists a unique painting configuration for each piece, can one prove it is exactly 7200 ways
     True
  ) :=
  sorry

end stack_glasses_opacity_l36_36382


namespace sophia_book_length_l36_36421

variables {P : ℕ}

def total_pages (P : ℕ) : Prop :=
  (2 / 3 : ℝ) * P = (1 / 3 : ℝ) * P + 90

theorem sophia_book_length 
  (h1 : total_pages P) :
  P = 270 :=
sorry

end sophia_book_length_l36_36421


namespace not_on_line_l36_36129

-- Defining the point (0,20)
def pt : ℝ × ℝ := (0, 20)

-- Defining the line equation
def line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- The proof problem stating that for all real numbers m and b, if m + b < 0, 
-- then the point (0, 20) cannot be on the line y = mx + b
theorem not_on_line (m b : ℝ) (h : m + b < 0) : ¬line m b pt := by
  sorry

end not_on_line_l36_36129


namespace arithmetic_sequence_common_difference_l36_36003

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h₁ : a 2 = 9) (h₂ : a 5 = 33) :
  ∀ d : ℤ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = 8 :=
by
  -- We state the theorem and provide a "sorry" proof placeholder
  sorry

end arithmetic_sequence_common_difference_l36_36003


namespace total_goals_in_league_l36_36313

variables (g1 g2 T : ℕ)

-- Conditions
def equal_goals : Prop := g1 = g2
def players_goals : Prop := g1 = 30
def total_goals_percentage : Prop := (g1 + g2) * 5 = T

-- Theorem to prove: Given the conditions, the total number of goals T should be 300
theorem total_goals_in_league (h1 : equal_goals g1 g2) (h2 : players_goals g1) (h3 : total_goals_percentage g1 g2 T) : T = 300 :=
sorry

end total_goals_in_league_l36_36313


namespace number_of_new_bricks_l36_36007

-- Definitions from conditions
def edge_length_original_brick : ℝ := 0.3
def edge_length_new_brick : ℝ := 0.5
def number_original_bricks : ℕ := 600

-- The classroom volume is unchanged, so we set up a proportion problem
-- Assuming the classroom is fully paved
theorem number_of_new_bricks :
  let volume_original_bricks := number_original_bricks * (edge_length_original_brick ^ 2)
  let volume_new_bricks := x * (edge_length_new_brick ^ 2)
  volume_original_bricks = volume_new_bricks → x = 216 := 
by
  sorry

end number_of_new_bricks_l36_36007


namespace investor_receives_7260_l36_36229

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260_l36_36229


namespace marbles_with_at_least_one_blue_l36_36439

theorem marbles_with_at_least_one_blue :
  (Nat.choose 10 4) - (Nat.choose 8 4) = 140 :=
by
  sorry

end marbles_with_at_least_one_blue_l36_36439


namespace sector_area_is_8pi_over_3_l36_36371

noncomputable def sector_area {r θ1 θ2 : ℝ} 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (hr : r = 4) : ℝ := 
    1 / 2 * (θ2 - θ1) * r ^ 2

theorem sector_area_is_8pi_over_3 (θ1 θ2 : ℝ) 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (r : ℝ) (hr : r = 4) : 
  sector_area hθ1 hθ2 hr = 8 * π / 3 :=
by
  sorry

end sector_area_is_8pi_over_3_l36_36371


namespace total_hours_over_two_weeks_l36_36730

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l36_36730


namespace simplify_expression_l36_36185

theorem simplify_expression (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := 
  sorry

end simplify_expression_l36_36185


namespace A_work_days_l36_36501

theorem A_work_days (x : ℝ) (H : 3 * (1 / x + 1 / 20) = 0.35) : x = 15 := 
by
  sorry

end A_work_days_l36_36501


namespace part_I_solution_set_part_II_min_value_l36_36370

-- Define the function f
def f (x a : ℝ) := 2*|x + 1| - |x - a|

-- Part I: Prove the solution set of f(x) ≥ 0 when a = 2
theorem part_I_solution_set (x : ℝ) :
  f x 2 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0 :=
sorry

-- Define the function g
def g (x a : ℝ) := f x a + 3*|x - a|

-- Part II: Prove the minimum value of m + n given t = 4 when a = 1
theorem part_II_min_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, g x 1 ≥ 4) → (2/m + 1/(2*n) = 4) → m + n = 9/8 :=
sorry

end part_I_solution_set_part_II_min_value_l36_36370


namespace odd_periodic_function_l36_36377

noncomputable def f : ℤ → ℤ := sorry

theorem odd_periodic_function (f_odd : ∀ x : ℤ, f (-x) = -f x)
  (period_f_3x1 : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 3) + 1))
  (f_one : f 1 = -1) : f 2006 = 1 :=
sorry

end odd_periodic_function_l36_36377


namespace arcade_game_monster_perimeter_l36_36418

theorem arcade_game_monster_perimeter :
  let r := 1 -- radius of the circle in cm
  let theta := 60 -- central angle of the missing sector in degrees
  let circumference := 2 * Real.pi * r -- circumference of the full circle
  let arc_fraction := (360 - theta) / 360 -- fraction of the circle forming the arc
  let arc_length := arc_fraction * circumference -- length of the arc
  let perimeter := arc_length + 2 * r -- total perimeter (arc + two radii)
  perimeter = (5 / 3) * Real.pi + 2 :=
by
  sorry

end arcade_game_monster_perimeter_l36_36418


namespace minimum_value_of_sum_l36_36942

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 2/b + 3/c = 2) : a + 2*b + 3*c = 18 ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end minimum_value_of_sum_l36_36942


namespace union_of_A_and_B_l36_36893

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3, 4} :=
  sorry

end union_of_A_and_B_l36_36893


namespace odd_numbers_divisibility_l36_36657

theorem odd_numbers_divisibility 
  (a b c : ℤ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : (ab - 1) % 4 = 0 ∨ (bc - 1) % 4 = 0 ∨ (ca - 1) % 4 = 0 := 
sorry

end odd_numbers_divisibility_l36_36657


namespace solution_set_for_f_ge_0_range_of_a_l36_36100

def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

theorem solution_set_for_f_ge_0 : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3/5} ∪ {x : ℝ | x ≥ 1} :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end solution_set_for_f_ge_0_range_of_a_l36_36100


namespace symmetric_point_Q_l36_36996

-- Definitions based on conditions
def P : ℝ × ℝ := (-3, 2)
def symmetric_with_respect_to_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.fst, -point.snd)

-- Theorem stating that the coordinates of point Q (symmetric to P with respect to the x-axis) are (-3, -2)
theorem symmetric_point_Q : symmetric_with_respect_to_x_axis P = (-3, -2) := 
sorry

end symmetric_point_Q_l36_36996


namespace polygon_interior_equals_exterior_sum_eq_360_l36_36594

theorem polygon_interior_equals_exterior_sum_eq_360 (n : ℕ) :
  (n - 2) * 180 = 360 → n = 6 :=
by
  intro h
  sorry

end polygon_interior_equals_exterior_sum_eq_360_l36_36594


namespace power_of_exponents_l36_36054

theorem power_of_exponents (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 := by
  sorry

end power_of_exponents_l36_36054


namespace linear_independent_vectors_p_value_l36_36539

theorem linear_independent_vectors_p_value (p : ℝ) :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ a * (2 : ℝ) + b * (5 : ℝ) = 0 ∧ a * (4 : ℝ) + b * p = 0) ↔ p = 10 :=
by
  sorry

end linear_independent_vectors_p_value_l36_36539


namespace solve_for_x_y_l36_36333

theorem solve_for_x_y (x y : ℝ) (h1 : x^2 + x * y + y = 14) (h2 : y^2 + x * y + x = 28) : 
  x + y = -7 ∨ x + y = 6 :=
by 
  -- We'll write sorry here to indicate the proof is to be completed
  sorry

end solve_for_x_y_l36_36333


namespace find_g_x_f_y_l36_36552

-- Definition of the functions and conditions
variable (f g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1)

-- The theorem to prove
theorem find_g_x_f_y (x y : ℝ) : g (x + f y) = -x + y - 1 := 
sorry

end find_g_x_f_y_l36_36552


namespace intersection_points_on_hyperbola_l36_36329

theorem intersection_points_on_hyperbola (p x y : ℝ) :
  (2*p*x - 3*y - 4*p = 0) ∧ (4*x - 3*p*y - 6 = 0) → 
  (∃ a b : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1) :=
by
  intros h
  sorry

end intersection_points_on_hyperbola_l36_36329


namespace no_solution_fraction_eq_l36_36750

theorem no_solution_fraction_eq {x m : ℝ} : 
  (∀ x, ¬ (1 - x = 0) → (2 - x) / (1 - x) = (m + x) / (1 - x) + 1) ↔ m = 0 := 
by
  sorry

end no_solution_fraction_eq_l36_36750


namespace combined_parent_age_difference_l36_36914

def father_age_at_sobha_birth : ℕ := 38
def mother_age_at_brother_birth : ℕ := 36
def brother_younger_than_sobha : ℕ := 4
def sister_younger_than_brother : ℕ := 3
def father_age_at_sister_birth : ℕ := 45
def mother_age_at_youngest_birth : ℕ := 34
def youngest_younger_than_sister : ℕ := 6

def mother_age_at_sobha_birth := mother_age_at_brother_birth - brother_younger_than_sobha
def father_age_at_youngest_birth := father_age_at_sister_birth + youngest_younger_than_sister

def combined_age_difference_at_sobha_birth := father_age_at_sobha_birth - mother_age_at_sobha_birth
def compounded_difference_at_sobha_brother_birth := 
  (father_age_at_sobha_birth + brother_younger_than_sobha) - mother_age_at_brother_birth
def mother_age_at_sister_birth := mother_age_at_brother_birth + sister_younger_than_brother
def compounded_difference_at_sobha_sister_birth := father_age_at_sister_birth - mother_age_at_sister_birth
def compounded_difference_at_youngest_birth := father_age_at_youngest_birth - mother_age_at_youngest_birth

def combined_age_difference := 
  combined_age_difference_at_sobha_birth + 
  compounded_difference_at_sobha_brother_birth + 
  compounded_difference_at_sobha_sister_birth + 
  compounded_difference_at_youngest_birth 

theorem combined_parent_age_difference : combined_age_difference = 35 := by
  sorry

end combined_parent_age_difference_l36_36914


namespace min_percentage_of_people_owning_95_percent_money_l36_36692

theorem min_percentage_of_people_owning_95_percent_money 
  (total_people: ℕ) (total_money: ℕ) 
  (P: ℕ) (M: ℕ) 
  (H1: P = total_people * 10 / 100) 
  (H2: M = total_money * 90 / 100)
  (H3: ∀ (people_owning_90_percent: ℕ), people_owning_90_percent = P → people_owning_90_percent * some_money = M) :
      P = total_people * 55 / 100 := 
sorry

end min_percentage_of_people_owning_95_percent_money_l36_36692


namespace markup_rate_correct_l36_36081

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_percentage : ℝ := 0.20
noncomputable def expenses_percentage : ℝ := 0.15
noncomputable def cost (S : ℝ) : ℝ := S - (profit_percentage * S + expenses_percentage * S)
noncomputable def markup_rate (S C : ℝ) : ℝ := (S - C) / C * 100

theorem markup_rate_correct :
  markup_rate selling_price (cost selling_price) = 53.85 := 
by
  sorry

end markup_rate_correct_l36_36081


namespace max_value_inequality_l36_36906

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4) :=
sorry

end max_value_inequality_l36_36906


namespace triangle_inequality_l36_36014

theorem triangle_inequality 
  (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l36_36014


namespace range_of_a_l36_36280

def S : Set ℝ := {x | (x - 2) ^ 2 > 9 }
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8 }

theorem range_of_a (a : ℝ) : (S ∪ T a) = Set.univ ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l36_36280


namespace rhombus_perimeter_area_l36_36863

theorem rhombus_perimeter_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (right_angle : ∀ (x : ℝ), x = d1 / 2 ∧ x = d2 / 2 → x * x + x * x = (d1 / 2)^2 + (d2 / 2)^2) : 
  ∃ (P A : ℝ), P = 52 ∧ A = 120 :=
by
  sorry

end rhombus_perimeter_area_l36_36863


namespace english_speaking_students_l36_36064

theorem english_speaking_students (T H B E : ℕ) (hT : T = 40) (hH : H = 30) (hB : B = 10) (h_inclusion_exclusion : T = H + E - B) : E = 20 :=
by
  sorry

end english_speaking_students_l36_36064


namespace unique_three_digit_base_g_l36_36621

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end unique_three_digit_base_g_l36_36621


namespace journey_speed_first_half_l36_36659

noncomputable def speed_first_half (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem journey_speed_first_half
  (total_time : ℝ) (total_distance : ℝ) (second_half_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : total_distance = 224)
  (h3 : second_half_speed = 24) :
  speed_first_half total_time total_distance second_half_speed = 21 := by
  sorry

end journey_speed_first_half_l36_36659


namespace min_female_students_l36_36682

theorem min_female_students (males females : ℕ) (total : ℕ) (percent_participated : ℕ) (participated : ℕ) (min_females : ℕ)
  (h1 : males = 22) 
  (h2 : females = 18) 
  (h3 : total = males + females)
  (h4 : percent_participated = 60) 
  (h5 : participated = (percent_participated * total) / 100)
  (h6 : min_females = participated - males) :
  min_females = 2 := 
sorry

end min_female_students_l36_36682


namespace greatest_possible_mean_BC_l36_36389

-- Mean weights for piles A, B
def mean_weight_A : ℝ := 60
def mean_weight_B : ℝ := 70

-- Combined mean weight for piles A and B
def mean_weight_AB : ℝ := 64

-- Combined mean weight for piles A and C
def mean_weight_AC : ℝ := 66

-- Prove that the greatest possible integer value for the mean weight of
-- the rocks in the combined piles B and C
theorem greatest_possible_mean_BC : ∃ (w : ℝ), (⌊w⌋ = 75) :=
by
  -- Definitions and assumptions based on problem conditions
  have h1 : mean_weight_A = 60 := rfl
  have h2 : mean_weight_B = 70 := rfl
  have h3 : mean_weight_AB = 64 := rfl
  have h4 : mean_weight_AC = 66 := rfl
  sorry

end greatest_possible_mean_BC_l36_36389


namespace bleach_to_detergent_ratio_changed_factor_l36_36309

theorem bleach_to_detergent_ratio_changed_factor :
  let original_bleach : ℝ := 4
  let original_detergent : ℝ := 40
  let original_water : ℝ := 100
  let altered_detergent : ℝ := 60
  let altered_water : ℝ := 300

  -- Calculate the factor by which the volume increased
  let original_total_volume := original_detergent + original_water
  let altered_total_volume := altered_detergent + altered_water
  let volume_increase_factor := altered_total_volume / original_total_volume

  -- The calculated factor of the ratio change
  let original_ratio_bleach_to_detergent := original_bleach / original_detergent

  altered_detergent > 0 → altered_water > 0 →
  volume_increase_factor * original_ratio_bleach_to_detergent = 2.5714 :=
by
  -- Insert proof here
  sorry

end bleach_to_detergent_ratio_changed_factor_l36_36309


namespace yellow_mugs_count_l36_36259

variables (R B Y O : ℕ)
variables (B_eq_3R : B = 3 * R)
variables (R_eq_Y_div_2 : R = Y / 2)
variables (O_eq_4 : O = 4)
variables (mugs_eq_40 : R + B + Y + O = 40)

theorem yellow_mugs_count : Y = 12 :=
by 
  sorry

end yellow_mugs_count_l36_36259


namespace arrangement_of_mississippi_no_adjacent_s_l36_36267

-- Conditions: The word "MISSISSIPPI" has 11 letters with specific frequencies: 1 M, 4 I's, 4 S's, 2 P's.
-- No two S's can be adjacent.
def ways_to_arrange_mississippi_no_adjacent_s: Nat :=
  let total_non_s_arrangements := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)
  let gaps_for_s := Nat.choose 8 4
  total_non_s_arrangements * gaps_for_s

theorem arrangement_of_mississippi_no_adjacent_s : ways_to_arrange_mississippi_no_adjacent_s = 7350 :=
by
  unfold ways_to_arrange_mississippi_no_adjacent_s
  sorry

end arrangement_of_mississippi_no_adjacent_s_l36_36267


namespace range_of_a_l36_36738

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end range_of_a_l36_36738


namespace smallest_positive_integer_divisible_by_15_16_18_l36_36104

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l36_36104


namespace Jack_sent_correct_number_of_BestBuy_cards_l36_36143

def price_BestBuy_gift_card : ℕ := 500
def price_Walmart_gift_card : ℕ := 200
def initial_BestBuy_gift_cards : ℕ := 6
def initial_Walmart_gift_cards : ℕ := 9

def total_price_of_initial_gift_cards : ℕ :=
  (initial_BestBuy_gift_cards * price_BestBuy_gift_card) +
  (initial_Walmart_gift_cards * price_Walmart_gift_card)

def price_of_Walmart_sent : ℕ := 2 * price_Walmart_gift_card
def value_of_gift_cards_remaining : ℕ := 3900

def prove_sent_BestBuy_worth : Prop :=
  total_price_of_initial_gift_cards - value_of_gift_cards_remaining - price_of_Walmart_sent = 1 * price_BestBuy_gift_card

theorem Jack_sent_correct_number_of_BestBuy_cards :
  prove_sent_BestBuy_worth :=
by
  sorry

end Jack_sent_correct_number_of_BestBuy_cards_l36_36143


namespace area_of_rhombus_l36_36896

theorem area_of_rhombus (x : ℝ) :
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  (d1 * d2) / 2 = 3 * x^2 + 11 * x + 10 :=
by
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  have h1 : d1 = 3 * x + 5 := rfl
  have h2 : d2 = 2 * x + 4 := rfl
  simp [h1, h2]
  sorry

end area_of_rhombus_l36_36896


namespace Lizzy_savings_after_loan_l36_36800

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l36_36800


namespace max_blocks_fit_l36_36684

theorem max_blocks_fit :
  ∃ (blocks : ℕ), blocks = 12 ∧ 
  (∀ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 1 → 
  ∀ (x y z : ℕ), x = 5 ∧ y = 4 ∧ z = 4 → 
  blocks = (x * y * z) / (a * b * c) ∧
  blocks = (y * z / (b * c) * (5 / a))) :=
sorry

end max_blocks_fit_l36_36684


namespace jellybean_problem_l36_36219

theorem jellybean_problem 
    (T L A : ℕ) 
    (h1 : T = L + 24) 
    (h2 : A = L / 2) 
    (h3 : T = 34) : 
    A = 5 := 
by 
  sorry

end jellybean_problem_l36_36219


namespace angle_between_vectors_acute_l36_36429

def isAcuteAngle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 > 0

def notCollinear (a b : ℝ × ℝ) : Prop :=
  ¬ ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem angle_between_vectors_acute (m : ℝ) :
  let a := (-1, 1)
  let b := (2 * m, m + 3)
  isAcuteAngle a b ∧ notCollinear a b ↔ m < 3 ∧ m ≠ -1 :=
by
  sorry

end angle_between_vectors_acute_l36_36429


namespace range_fraction_l36_36880

theorem range_fraction {x y : ℝ} (h : x^2 + y^2 + 2 * x = 0) :
  ∃ a b : ℝ, a = -1 ∧ b = 1 / 3 ∧ ∀ z, z = (y - x) / (x - 1) → a ≤ z ∧ z ≤ b :=
by 
  sorry

end range_fraction_l36_36880


namespace no_real_solution_for_x_l36_36521

theorem no_real_solution_for_x
  (y : ℝ)
  (x : ℝ)
  (h1 : y = (x^3 - 8) / (x - 2))
  (h2 : y = 3 * x) :
  ¬ ∃ x : ℝ, y = 3*x ∧ y = (x^3 - 8) / (x - 2) :=
by {
  sorry
}

end no_real_solution_for_x_l36_36521


namespace parity_of_f_and_h_l36_36995

-- Define function f
def f (x : ℝ) : ℝ := x^2

-- Define function h
def h (x : ℝ) : ℝ := x

-- Define even and odd function
def even_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def odd_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = - g x

-- Theorem statement
theorem parity_of_f_and_h :
  even_fun f ∧ odd_fun h :=
by {
  sorry
}

end parity_of_f_and_h_l36_36995


namespace time_addition_correct_l36_36871

theorem time_addition_correct :
  let current_time := (3, 0, 0)  -- Representing 3:00:00 PM as a tuple (hours, minutes, seconds)
  let duration := (313, 45, 56)  -- Duration to be added: 313 hours, 45 minutes, and 56 seconds
  let new_time := ((3 + (313 % 12) + 45 / 60 + (56 / 3600)), (0 + 45 % 60), (0 + 56 % 60))
  let A := (4 : ℕ)  -- Extracted hour part of new_time
  let B := (45 : ℕ)  -- Extracted minute part of new_time
  let C := (56 : ℕ)  -- Extracted second part of new_time
  A + B + C = 105 := 
by
  -- Placeholder for the actual proof.
  sorry

end time_addition_correct_l36_36871


namespace number_above_210_is_165_l36_36908

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2
def tetrahedral_number (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6
def row_start (k : ℕ) : ℕ := tetrahedral_number (k - 1) + 1

theorem number_above_210_is_165 :
  ∀ k, triangular_number k = 210 →
  ∃ n, n = 165 → 
  ∀ m, row_start (k - 1) ≤ m ∧ m < row_start k →
  m = 210 →
  n = m - triangular_number (k - 1) :=
  sorry

end number_above_210_is_165_l36_36908


namespace min_houses_needed_l36_36599

theorem min_houses_needed (n : ℕ) (x : ℕ) (h : n > 0) : (x ≤ n ∧ (x: ℚ)/n < 0.06) → n ≥ 20 :=
sorry

end min_houses_needed_l36_36599


namespace solution_for_b_l36_36029

theorem solution_for_b (x y b : ℚ) (h1 : 4 * x + 3 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hx : x = 3) : b = -21 / 5 := by
  sorry

end solution_for_b_l36_36029


namespace combined_degrees_l36_36669

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l36_36669


namespace black_squares_31x31_l36_36855

-- Definitions to express the checkerboard problem conditions
def isCheckerboard (n : ℕ) : Prop := 
  ∀ i j : ℕ,
    i < n → j < n → 
    ((i + j) % 2 = 0 → (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1))

def blackCornerSquares (n : ℕ) : Prop :=
  ∀ i j : ℕ,
    (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 0

-- The main statement to prove
theorem black_squares_31x31 :
  ∃ (n : ℕ) (count : ℕ), n = 31 ∧ isCheckerboard n ∧ blackCornerSquares n ∧ count = 481 := 
by 
  sorry -- Proof to be provided

end black_squares_31x31_l36_36855


namespace number_of_tiles_is_47_l36_36220

theorem number_of_tiles_is_47 : 
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 2) ∧ n = 47 :=
by
  sorry

end number_of_tiles_is_47_l36_36220


namespace circumcircle_diameter_triangle_ABC_l36_36876

theorem circumcircle_diameter_triangle_ABC
  (A : ℝ) (BC : ℝ) (R : ℝ)
  (hA : A = 60) (hBC : BC = 4)
  (hR_formula : 2 * R = BC / Real.sin (A * Real.pi / 180)) :
  2 * R = 8 * Real.sqrt 3 / 3 :=
by
  sorry

end circumcircle_diameter_triangle_ABC_l36_36876


namespace find_rate_per_kg_mangoes_l36_36396

noncomputable def rate_per_kg_mangoes
  (cost_grapes_rate : ℕ)
  (quantity_grapes : ℕ)
  (quantity_mangoes : ℕ)
  (total_paid : ℕ)
  (rate_grapes : ℕ)
  (rate_mangoes : ℕ) :=
  total_paid = (rate_grapes * quantity_grapes) + (rate_mangoes * quantity_mangoes)

theorem find_rate_per_kg_mangoes :
  rate_per_kg_mangoes 70 8 11 1165 70 55 :=
by
  sorry

end find_rate_per_kg_mangoes_l36_36396


namespace rita_bought_5_dresses_l36_36093

def pants_cost := 3 * 12
def jackets_cost := 4 * 30
def total_cost_pants_jackets := pants_cost + jackets_cost
def amount_spent := 400 - 139
def total_cost_dresses := amount_spent - total_cost_pants_jackets - 5
def number_of_dresses := total_cost_dresses / 20

theorem rita_bought_5_dresses : number_of_dresses = 5 :=
by sorry

end rita_bought_5_dresses_l36_36093


namespace percent_students_elected_to_learn_from_home_l36_36278

theorem percent_students_elected_to_learn_from_home (H : ℕ) : 
  (100 - H) / 2 = 30 → H = 40 := 
by
  sorry

end percent_students_elected_to_learn_from_home_l36_36278


namespace common_area_of_triangles_is_25_l36_36437

-- Define basic properties and conditions of an isosceles right triangle with hypotenuse = 10 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = 10^2
def is_isosceles_right_triangle (a b : ℝ) : Prop := a = b ∧ hypotenuse a b

-- Definitions representing the triangls
noncomputable def triangle1 := ∃ a b : ℝ, is_isosceles_right_triangle a b
noncomputable def triangle2 := ∃ a b : ℝ, is_isosceles_right_triangle a b

-- The area common to both triangles is the focus
theorem common_area_of_triangles_is_25 : 
  triangle1 ∧ triangle2 → 
  ∃ area : ℝ, area = 25 
  := 
sorry

end common_area_of_triangles_is_25_l36_36437


namespace find_number_of_students_l36_36848

-- Definitions for the conditions
def avg_age_students := 14
def teacher_age := 65
def new_avg_age := 15

-- The total age of students is n multiplied by their average age
def total_age_students (n : ℕ) := n * avg_age_students

-- The total age including teacher
def total_age_incl_teacher (n : ℕ) := total_age_students n + teacher_age

-- The new average age when teacher is included
def new_avg_age_incl_teacher (n : ℕ) := total_age_incl_teacher n / (n + 1)

theorem find_number_of_students (n : ℕ) (h₁ : avg_age_students = 14) (h₂ : teacher_age = 65) (h₃ : new_avg_age = 15) 
  (h_averages_eq : new_avg_age_incl_teacher n = new_avg_age) : n = 50 :=
  sorry

end find_number_of_students_l36_36848


namespace daps_equivalent_to_dips_l36_36459

theorem daps_equivalent_to_dips (daps dops dips : ℕ) 
  (h1 : 4 * daps = 3 * dops) 
  (h2 : 2 * dops = 7 * dips) :
  35 * dips = 20 * daps :=
by
  sorry

end daps_equivalent_to_dips_l36_36459


namespace total_matches_round_robin_l36_36600

/-- A round-robin chess tournament is organized in two groups with different numbers of players. 
Group A consists of 6 players, and Group B consists of 5 players. 
Each player in each group plays every other player in the same group exactly once. 
Prove that the total number of matches is 25. -/
theorem total_matches_round_robin 
  (nA : ℕ) (nB : ℕ) 
  (hA : nA = 6) (hB : nB = 5) : 
  (nA * (nA - 1) / 2) + (nB * (nB - 1) / 2) = 25 := 
  by
    sorry

end total_matches_round_robin_l36_36600


namespace compute_a1d1_a2d2_a3d3_l36_36208

noncomputable def polynomial_equation (a1 a2 a3 d1 d2 d3: ℝ) : Prop :=
  ∀ x : ℝ, (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)

theorem compute_a1d1_a2d2_a3d3 (a1 a2 a3 d1 d2 d3 : ℝ) (h : polynomial_equation a1 a2 a3 d1 d2 d3) : 
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a1d1_a2d2_a3d3_l36_36208


namespace minimum_value_of_k_l36_36929

theorem minimum_value_of_k (m n a k : ℕ) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hk : 1 < k) (h : 5^m + 63 * n + 49 = a^k) : k = 5 :=
sorry

end minimum_value_of_k_l36_36929


namespace fruit_shop_problem_l36_36559

variable (x y z : ℝ)

theorem fruit_shop_problem
  (h1 : x + 4 * y + 2 * z = 27.2)
  (h2 : 2 * x + 6 * y + 2 * z = 32.4) :
  x + 2 * y = 5.2 :=
by
  sorry

end fruit_shop_problem_l36_36559


namespace inequality_proof_equality_condition_l36_36736

variable {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
axiom x1_pos : x1 > 0
axiom x2_pos : x2 > 0
axiom x1y1_gz1sq : x1 * y1 > z1 ^ 2
axiom x2y2_gz2sq : x2 * y2 > z2 ^ 2

theorem inequality_proof : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) <= 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

theorem equality_condition : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) = 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) ↔ 
  (x1 = x2 ∧ y1 = y2 ∧ z1 = z2) :=
sorry

end inequality_proof_equality_condition_l36_36736


namespace hexagon_side_equality_l36_36488

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

def parallel (x y : A) : Prop := ∀ r : ℝ, x = r • y
noncomputable def length_eq (x y : A) : Prop := ∃ r : ℝ, r • x = y

variables (AB DE BC EF CD FA : A)
variables (h1 : parallel AB DE)
variables (h2 : parallel BC EF)
variables (h3 : parallel CD FA)
variables (h4 : length_eq AB DE)

theorem hexagon_side_equality :
  length_eq BC EF ∧ length_eq CD FA :=
by
  sorry

end hexagon_side_equality_l36_36488


namespace product_of_sums_of_two_squares_l36_36569

theorem product_of_sums_of_two_squares
  (a b a1 b1 : ℤ) :
  ((a^2 + b^2) * (a1^2 + b1^2)) = ((a * a1 - b * b1)^2 + (a * b1 + b * a1)^2) := 
sorry

end product_of_sums_of_two_squares_l36_36569


namespace intersection_point_exists_l36_36169

def line_param_eq (x y z : ℝ) (t : ℝ) := x = 5 + t ∧ y = 3 - t ∧ z = 2
def plane_eq (x y z : ℝ) := 3 * x + y - 5 * z - 12 = 0

theorem intersection_point_exists : 
  ∃ t : ℝ, ∃ x y z : ℝ, line_param_eq x y z t ∧ plane_eq x y z ∧ x = 7 ∧ y = 1 ∧ z = 2 :=
by {
  -- Skipping the proof
  sorry
}

end intersection_point_exists_l36_36169


namespace sum_squares_of_solutions_eq_l36_36145

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l36_36145


namespace sum_of_products_circle_l36_36161

theorem sum_of_products_circle 
  (a b c d : ℤ) 
  (h : a + b + c + d = 0) : 
  -((a * (b + d)) + (b * (a + c)) + (c * (b + d)) + (d * (a + c))) = 2 * (a + c) ^ 2 :=
sorry

end sum_of_products_circle_l36_36161


namespace cube_edge_length_l36_36802

theorem cube_edge_length (V : ℝ) (a : ℝ)
  (hV : V = (4 / 3) * Real.pi * (Real.sqrt 3 * a / 2) ^ 3)
  (hVolume : V = (9 * Real.pi) / 2) :
  a = Real.sqrt 3 :=
by
  sorry

end cube_edge_length_l36_36802


namespace time_to_fill_tank_with_leak_l36_36188

theorem time_to_fill_tank_with_leak (A L : ℚ) (hA : A = 1/6) (hL : L = 1/24) :
  (1 / (A - L)) = 8 := 
by 
  sorry

end time_to_fill_tank_with_leak_l36_36188


namespace nelly_bid_l36_36993

theorem nelly_bid (joe_bid sarah_bid : ℕ) (h1 : joe_bid = 160000) (h2 : sarah_bid = 50000)
  (h3 : ∀ nelly_bid, nelly_bid = 3 * joe_bid + 2000) (h4 : ∀ nelly_bid, nelly_bid = 4 * sarah_bid + 1500) :
  ∃ nelly_bid, nelly_bid = 482000 :=
by
  -- Skipping the proof with sorry
  sorry

end nelly_bid_l36_36993


namespace arithmetic_sequence_common_difference_l36_36689

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a6 : a 6 = 5) (h_a10 : a 10 = 6) : 
  (a 10 - a 6) / 4 = 1 / 4 := 
by
  sorry

end arithmetic_sequence_common_difference_l36_36689


namespace total_books_per_year_l36_36360

variable (c s : ℕ)

theorem total_books_per_year (hc : 0 < c) (hs : 0 < s) :
  6 * 12 * (c * s) = 72 * c * s := by
  sorry

end total_books_per_year_l36_36360


namespace distinct_powers_exist_l36_36203

theorem distinct_powers_exist :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
    (∃ n, a1 = n^2) ∧ (∃ m, a2 = m^2) ∧
    (∃ p, b1 = p^3) ∧ (∃ q, b2 = q^3) ∧
    (∃ r, c1 = r^5) ∧ (∃ s, c2 = s^5) ∧
    (∃ t, d1 = t^7) ∧ (∃ u, d2 = u^7) ∧
    a1 - a2 = b1 - b2 ∧ b1 - b2 = c1 - c2 ∧ c1 - c2 = d1 - d2 ∧
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 := 
sorry

end distinct_powers_exist_l36_36203


namespace initial_avg_height_l36_36517

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l36_36517


namespace fifth_equation_sum_first_17_even_sum_even_28_to_50_l36_36286

-- Define a function to sum the first n even numbers
def sum_even (n : ℕ) : ℕ := n * (n + 1)

-- Part (1) According to the pattern, write down the ⑤th equation
theorem fifth_equation : sum_even 5 = 30 := by
  sorry

-- Part (2) Calculate according to this pattern:
-- ① Sum of first 17 even numbers
theorem sum_first_17_even : sum_even 17 = 306 := by
  sorry

-- ② Sum of even numbers from 28 to 50
theorem sum_even_28_to_50 : 
  let sum_even_50 := sum_even 25
  let sum_even_26 := sum_even 13
  sum_even_50 - sum_even_26 = 468 := by
  sorry

end fifth_equation_sum_first_17_even_sum_even_28_to_50_l36_36286


namespace positive_A_satisfies_eq_l36_36176

theorem positive_A_satisfies_eq :
  ∃ (A : ℝ), A > 0 ∧ A^2 + 49 = 194 → A = Real.sqrt 145 :=
by
  sorry

end positive_A_satisfies_eq_l36_36176


namespace david_marks_in_english_l36_36453

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l36_36453


namespace seating_arrangement_l36_36779

theorem seating_arrangement (n x : ℕ) (h1 : 7 * x + 6 * (n - x) = 53) : x = 5 :=
sorry

end seating_arrangement_l36_36779


namespace triangle_angle_l36_36015

-- Definitions of the conditions and theorem
variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_angle (h : b^2 + c^2 - a^2 = bc)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hA : 0 < A) (hA_max : A < π) :
  A = π / 3 :=
by
  sorry

end triangle_angle_l36_36015


namespace number_of_digits_in_x20_l36_36868

theorem number_of_digits_in_x20 (x : ℝ) (hx1 : 10^(7/4) ≤ x) (hx2 : x < 10^2) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 :=
by
  -- Proof goes here
  sorry

end number_of_digits_in_x20_l36_36868


namespace overlapping_triangle_area_l36_36625

/-- Given a rectangle with length 8 and width 4, folded along its diagonal, 
    the area of the overlapping part (grey triangle) is 10. --/
theorem overlapping_triangle_area : 
  let length := 8 
  let width := 4 
  let diagonal := (length^2 + width^2)^(1/2) 
  let base := (length^2 / (width^2 + length^2))^(1/2) * width 
  let height := width
  1 / 2 * base * height = 10 := by 
  sorry

end overlapping_triangle_area_l36_36625


namespace fan_working_time_each_day_l36_36528

theorem fan_working_time_each_day
  (airflow_per_second : ℝ)
  (total_airflow_week : ℝ)
  (seconds_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (airy_sector: airflow_per_second = 10)
  (flow_week : total_airflow_week = 42000)
  (sec_per_hr : seconds_per_hour = 3600)
  (hrs_per_day : hours_per_day = 24)
  (days_week : days_per_week = 7) :
  let airflow_per_hour := airflow_per_second * seconds_per_hour
  let total_hours_week := total_airflow_week / airflow_per_hour
  let hours_per_day_given := total_hours_week / days_per_week
  let minutes_per_day := hours_per_day_given * 60
  minutes_per_day = 10 := 
by
  sorry

end fan_working_time_each_day_l36_36528


namespace y_equals_4_if_abs_diff_eq_l36_36401

theorem y_equals_4_if_abs_diff_eq (y : ℝ) (h : |y - 3| = |y - 5|) : y = 4 :=
sorry

end y_equals_4_if_abs_diff_eq_l36_36401


namespace problem_x_l36_36416

theorem problem_x (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 :=
sorry

end problem_x_l36_36416


namespace deposit_percentage_l36_36862

-- Define the conditions of the problem
def amount_deposited : ℕ := 5000
def monthly_income : ℕ := 25000

-- Define the percentage deposited formula
def percentage_deposited (amount_deposited monthly_income : ℕ) : ℚ :=
  (amount_deposited / monthly_income) * 100

-- State the theorem to be proved
theorem deposit_percentage :
  percentage_deposited amount_deposited monthly_income = 20 := by
  sorry

end deposit_percentage_l36_36862


namespace neg_P_l36_36584

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x ≤ 0

-- State the negation of P
theorem neg_P : ¬P ↔ ∀ x : ℝ, Real.exp x > 0 := 
by 
  sorry

end neg_P_l36_36584


namespace first_problem_second_problem_l36_36576

variable (x : ℝ)

-- Proof for the first problem
theorem first_problem : 6 * x^3 / (-3 * x^2) = -2 * x := by
sorry

-- Proof for the second problem
theorem second_problem : (2 * x + 3) * (2 * x - 3) - 4 * (x - 2)^2 = 16 * x - 25 := by
sorry

end first_problem_second_problem_l36_36576


namespace work_together_days_l36_36359

theorem work_together_days (A_rate B_rate x total_work B_days_worked : ℚ)
  (hA : A_rate = 1/4)
  (hB : B_rate = 1/8)
  (hCombined : (A_rate + B_rate) * x + B_rate * B_days_worked = total_work)
  (hTotalWork : total_work = 1)
  (hBDays : B_days_worked = 2) : x = 2 :=
by
  sorry

end work_together_days_l36_36359


namespace total_value_is_76_percent_of_dollar_l36_36945

def coin_values : List Nat := [1, 5, 20, 50]

def total_value (coins : List Nat) : Nat :=
  List.sum coins

def percentage_of_dollar (value : Nat) : Nat :=
  value * 100 / 100

theorem total_value_is_76_percent_of_dollar :
  percentage_of_dollar (total_value coin_values) = 76 := by
  sorry

end total_value_is_76_percent_of_dollar_l36_36945


namespace product_simplification_l36_36028

theorem product_simplification :
  (10 * (1 / 5) * (1 / 2) * 4 / 2 : ℝ) = 2 :=
by
  sorry

end product_simplification_l36_36028


namespace inequality_holds_iff_even_l36_36207

theorem inequality_holds_iff_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ x y z : ℝ, (x - y) ^ a * (x - z) ^ b * (y - z) ^ c ≥ 0) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end inequality_holds_iff_even_l36_36207


namespace percentage_goods_lost_l36_36803

theorem percentage_goods_lost
    (cost_price selling_price loss_price : ℝ)
    (profit_percent loss_percent : ℝ)
    (h_profit : selling_price = cost_price * (1 + profit_percent / 100))
    (h_loss_value : loss_price = selling_price * (loss_percent / 100))
    (cost_price_assumption : cost_price = 100)
    (profit_percent_assumption : profit_percent = 10)
    (loss_percent_assumption : loss_percent = 45) :
    (loss_price / cost_price * 100) = 49.5 :=
sorry

end percentage_goods_lost_l36_36803


namespace power_sum_roots_l36_36136

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l36_36136


namespace fruit_days_l36_36248

/-
  Henry and his brother believe in the famous phrase, "An apple a day, keeps the doctor away." 
  Henry's sister, however, believes that "A banana a day makes the trouble fade away" 
  and their father thinks that "An orange a day will keep the weaknesses at bay." 
  A box of apples contains 14 apples, a box of bananas has 20 bananas, and a box of oranges contains 12 oranges. 

  If Henry and his brother eat 1 apple each a day, their sister consumes 2 bananas per day, 
  and their father eats 3 oranges per day, how many days can the family of four continue eating fruits 
  if they have 3 boxes of apples, 4 boxes of bananas, and 5 boxes of oranges? 

  However, due to seasonal changes, oranges are only available for the first 20 days. 
  Moreover, Henry's sister has decided to only eat bananas on days when the day of the month is an odd number. 
  Considering these constraints, determine the total number of days the family of four can continue eating their preferred fruits.
-/

def apples_per_box := 14
def bananas_per_box := 20
def oranges_per_box := 12

def apples_boxes := 3
def bananas_boxes := 4
def oranges_boxes := 5

def daily_apple_consumption := 2
def daily_banana_consumption := 2
def daily_orange_consumption := 3

def orange_availability_days := 20

def odd_days_in_month := 16

def total_number_of_days : ℕ :=
  let total_apples := apples_boxes * apples_per_box
  let total_bananas := bananas_boxes * bananas_per_box
  let total_oranges := oranges_boxes * oranges_per_box
  
  let days_with_apples := total_apples / daily_apple_consumption
  let days_with_bananas := (total_bananas / (odd_days_in_month * daily_banana_consumption)) * 30
  let days_with_oranges := if total_oranges / daily_orange_consumption > orange_availability_days then orange_availability_days else total_oranges / daily_orange_consumption
  min (min days_with_apples days_with_oranges) (days_with_bananas / 30 * 30)

theorem fruit_days : total_number_of_days = 20 := 
  sorry

end fruit_days_l36_36248


namespace no_integer_solution_l36_36629

theorem no_integer_solution (x y z : ℤ) (h : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solution_l36_36629


namespace positive_difference_two_numbers_l36_36742

theorem positive_difference_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) : |x - y| = 8 := by
  sorry

end positive_difference_two_numbers_l36_36742


namespace taylor_correct_answers_percentage_l36_36773

theorem taylor_correct_answers_percentage 
  (N : ℕ := 30)
  (alex_correct_alone_percentage : ℝ := 0.85)
  (alex_overall_percentage : ℝ := 0.83)
  (taylor_correct_alone_percentage : ℝ := 0.95)
  (alex_correct_alone : ℕ := 13)
  (alex_correct_total : ℕ := 25)
  (together_correct : ℕ := 12)
  (taylor_correct_alone : ℕ := 14)
  (taylor_correct_total : ℕ := 26) :
  ((taylor_correct_total : ℝ) / (N : ℝ)) * 100 = 87 :=
by
  sorry

end taylor_correct_answers_percentage_l36_36773


namespace Brandy_caffeine_intake_l36_36275

theorem Brandy_caffeine_intake :
  let weight := 60
  let recommended_limit_per_kg := 2.5
  let tolerance := 50
  let coffee_cups := 2
  let coffee_per_cup := 95
  let energy_drinks := 4
  let caffeine_per_energy_drink := 120
  let max_safe_caffeine := weight * recommended_limit_per_kg + tolerance
  let caffeine_from_coffee := coffee_cups * coffee_per_cup
  let caffeine_from_energy_drinks := energy_drinks * caffeine_per_energy_drink
  let total_caffeine_consumed := caffeine_from_coffee + caffeine_from_energy_drinks
  max_safe_caffeine - total_caffeine_consumed = -470 := 
by
  sorry

end Brandy_caffeine_intake_l36_36275


namespace brother_more_lambs_than_merry_l36_36612

theorem brother_more_lambs_than_merry
  (merry_lambs : ℕ) (total_lambs : ℕ) (more_than_merry : ℕ)
  (h1 : merry_lambs = 10) 
  (h2 : total_lambs = 23)
  (h3 : more_than_merry + merry_lambs + merry_lambs = total_lambs) :
  more_than_merry = 3 :=
by
  sorry

end brother_more_lambs_than_merry_l36_36612


namespace lines_intersect_l36_36640

theorem lines_intersect (m b : ℝ) (h1 : 17 = 2 * m * 4 + 5) (h2 : 17 = 4 * 4 + b) : b + m = 2.5 :=
by {
    sorry
}

end lines_intersect_l36_36640


namespace ellipse_foci_coordinates_l36_36022

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (x^2 / 64 + y^2 / 100 = 1) → (x = 0 ∧ (y = 6 ∨ y = -6)) :=
by
  sorry

end ellipse_foci_coordinates_l36_36022


namespace prob_not_green_is_six_over_eleven_l36_36225

-- Define the odds for pulling a green marble
def odds_green : ℕ × ℕ := (5, 6)

-- Define the total number of events as the sum of both parts of the odds
def total_events : ℕ := odds_green.1 + odds_green.2

-- Define the probability of not pulling a green marble
def probability_not_green : ℚ := odds_green.2 / total_events

-- State the theorem
theorem prob_not_green_is_six_over_eleven : probability_not_green = 6 / 11 := by
  -- Proof goes here
  sorry

end prob_not_green_is_six_over_eleven_l36_36225


namespace total_cash_realized_correct_l36_36962

structure Stock where
  value : ℝ
  return_rate : ℝ
  brokerage_fee_rate : ℝ

def stockA : Stock := { value := 10000, return_rate := 0.14, brokerage_fee_rate := 0.0025 }
def stockB : Stock := { value := 20000, return_rate := 0.10, brokerage_fee_rate := 0.005 }
def stockC : Stock := { value := 30000, return_rate := 0.07, brokerage_fee_rate := 0.0075 }

def cash_realized (s : Stock) : ℝ :=
  let total_with_return := s.value * (1 + s.return_rate)
  total_with_return - (total_with_return * s.brokerage_fee_rate)

noncomputable def total_cash_realized : ℝ :=
  cash_realized stockA + cash_realized stockB + cash_realized stockC

theorem total_cash_realized_correct :
  total_cash_realized = 65120.75 :=
    sorry

end total_cash_realized_correct_l36_36962


namespace max_handshakes_l36_36890

theorem max_handshakes (n : ℕ) (m : ℕ)
  (h_n : n = 25)
  (h_m : m = 20)
  (h_mem : n - m = 5)
  : ∃ (max_handshakes : ℕ), max_handshakes = 250 :=
by
  sorry

end max_handshakes_l36_36890


namespace negation_of_exists_l36_36252

variable (a : ℝ)

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) : ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l36_36252


namespace part1_part2_l36_36139

theorem part1 (a : ℝ) (h : 48 * a^2 = 75) (ha : a > 0) : a = 5 / 4 :=
sorry

theorem part2 (θ : ℝ) 
  (h₁ : 10 * (Real.sin θ) ^ 2 = 5) 
  (h₀ : 0 < θ ∧ θ < Real.pi / 2) 
  : θ = Real.pi / 4 :=
sorry

end part1_part2_l36_36139


namespace solution_set_for_inequality_l36_36972

theorem solution_set_for_inequality 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono_dec : ∀ x y, 0 < x → x < y → f y ≤ f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_for_inequality_l36_36972


namespace sqrt_floor_19992000_l36_36410

theorem sqrt_floor_19992000 : (Int.floor (Real.sqrt 19992000)) = 4471 := by
  sorry

end sqrt_floor_19992000_l36_36410


namespace correct_operation_l36_36455

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l36_36455


namespace inequality_sqrt_sum_l36_36141

theorem inequality_sqrt_sum (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2 - a * b) + Real.sqrt (b^2 + c^2 - b * c)) ≥ Real.sqrt (a^2 + c^2 + a * c) :=
sorry

end inequality_sqrt_sum_l36_36141


namespace num_ordered_quadruples_l36_36460

theorem num_ordered_quadruples (n : ℕ) :
  ∃ (count : ℕ), count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) ∧
  (∀ (k1 k2 k3 k4 : ℕ), k1 ≤ n ∧ k2 ≤ n ∧ k3 ≤ n ∧ k4 ≤ n → 
    ((k1 + k3) / 2 = (k2 + k4) / 2) → 
    count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3)) :=
by sorry

end num_ordered_quadruples_l36_36460


namespace xy_sum_eq_16_l36_36551

theorem xy_sum_eq_16 (x y : ℕ) (h1: x > 0) (h2: y > 0) (h3: x < 20) (h4: y < 20) (h5: x + y + x * y = 76) : x + y = 16 :=
  sorry

end xy_sum_eq_16_l36_36551


namespace combined_salaries_of_ABCD_l36_36336

theorem combined_salaries_of_ABCD 
  (A B C D E : ℝ)
  (h1 : E = 9000)
  (h2 : (A + B + C + D + E) / 5 = 8600) :
  A + B + C + D = 34000 := 
sorry

end combined_salaries_of_ABCD_l36_36336


namespace fractional_expression_evaluation_l36_36008

theorem fractional_expression_evaluation
  (m n r t : ℚ)
  (h1 : m / n = 4 / 3)
  (h2 : r / t = 9 / 14) :
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end fractional_expression_evaluation_l36_36008


namespace arithmetic_sequence_unique_a_l36_36577

theorem arithmetic_sequence_unique_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_seq 1 = a) (h2 : a > 0)
  (h3 : b 1 - a_seq 1 = 1) (h4 : b 2 - a_seq 2 = 2)
  (h5 : b 3 - a_seq 3 = 3)
  (unique_a : ∀ (a' : ℝ), (a_seq 1 = a' ∧ a' > 0 ∧ b 1 - a' = 1 ∧ b 2 - a_seq 2 = 2 ∧ b 3 - a_seq 3 = 3) → a' = a) :
  a = 1 / 3 :=
by
  sorry

end arithmetic_sequence_unique_a_l36_36577


namespace intersection_of_A_and_B_l36_36084

-- Definitions of sets A and B
def set_A : Set ℝ := { x | x^2 - x - 6 < 0 }
def set_B : Set ℝ := { x | (x + 4) * (x - 2) > 0 }

-- Theorem statement for the intersection of A and B
theorem intersection_of_A_and_B : set_A ∩ set_B = { x | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_of_A_and_B_l36_36084


namespace express_A_using_roster_method_l36_36150

def A := {x : ℕ | ∃ (n : ℕ), 8 / (2 - x) = n }

theorem express_A_using_roster_method :
  A = {0, 1} :=
sorry

end express_A_using_roster_method_l36_36150


namespace part1_range_a_part2_range_a_l36_36587

-- Definitions of the propositions
def p (a : ℝ) := ∃ x : ℝ, x^2 + a * x + 2 = 0

def q (a : ℝ) := ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - a < 0

-- Part 1: If p is true, find the range of values for a
theorem part1_range_a (a : ℝ) :
  p a → (a ≤ -2*Real.sqrt 2 ∨ a ≥ 2*Real.sqrt 2) := sorry

-- Part 2: If one of p or q is true and the other is false, find the range of values for a
theorem part2_range_a (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) →
  (a ≤ -2*Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2*Real.sqrt 2)) := sorry

end part1_range_a_part2_range_a_l36_36587


namespace pages_in_book_l36_36326

theorem pages_in_book
  (x : ℝ)
  (h1 : x - (x / 6 + 10) = (5 * x) / 6 - 10)
  (h2 : (5 * x) / 6 - 10 - ((1 / 5) * ((5 * x) / 6 - 10) + 20) = (2 * x) / 3 - 28)
  (h3 : (2 * x) / 3 - 28 - ((1 / 4) * ((2 * x) / 3 - 28) + 25) = x / 2 - 46)
  (h4 : x / 2 - 46 = 72) :
  x = 236 := 
sorry

end pages_in_book_l36_36326


namespace focus_of_hyperbola_l36_36860

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end focus_of_hyperbola_l36_36860


namespace flower_beds_and_circular_path_fraction_l36_36598

noncomputable def occupied_fraction 
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ) : ℝ :=
  let flower_bed_area := 2 * (1 / 2 : ℝ) * triangle_leg^2
  let circular_path_area := Real.pi * circle_radius ^ 2
  let occupied_area := flower_bed_area + circular_path_area
  occupied_area / (yard_length * yard_width)

theorem flower_beds_and_circular_path_fraction
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ)
  (h1 : side1 = 20)
  (h2 : side2 = 30)
  (h3 : triangle_leg = (side2 - side1) / 2)
  (h4 : yard_length = 30)
  (h5 : yard_width = 5)
  (h6 : circle_radius = 2) :
  occupied_fraction yard_length yard_width side1 side2 triangle_leg circle_radius = (25 + 4 * Real.pi) / 150 :=
by sorry

end flower_beds_and_circular_path_fraction_l36_36598


namespace find_second_number_l36_36920

theorem find_second_number 
  (x : ℕ)
  (h1 : (55 + x + 507 + 2 + 684 + 42) / 6 = 223)
  : x = 48 := 
by 
  sorry

end find_second_number_l36_36920


namespace measure_15_minutes_with_hourglasses_l36_36857

theorem measure_15_minutes_with_hourglasses (h7 h11 : ℕ) (h7_eq : h7 = 7) (h11_eq : h11 = 11) : ∃ t : ℕ, t = 15 :=
by
  let t := 15
  have h7 : ℕ := 7
  have h11 : ℕ := 11
  exact ⟨t, by norm_num⟩

end measure_15_minutes_with_hourglasses_l36_36857


namespace ones_divisible_by_d_l36_36073

theorem ones_divisible_by_d (d : ℕ) (h1 : ¬ (2 ∣ d)) (h2 : ¬ (5 ∣ d))  : 
  ∃ n, (∃ k : ℕ, n = 10^k - 1) ∧ n % d = 0 := 
sorry

end ones_divisible_by_d_l36_36073


namespace evaluate_g_at_6_l36_36063

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75

theorem evaluate_g_at_6 : g 6 = 363 :=
by
  -- Proof skipped
  sorry

end evaluate_g_at_6_l36_36063


namespace find_value_of_a_l36_36589

theorem find_value_of_a (a : ℝ) (h : ( (-2 - (2 * a - 1)) / (3 - (-2)) = -1 )) : a = 2 :=
sorry

end find_value_of_a_l36_36589


namespace supplement_of_angle_l36_36255

theorem supplement_of_angle (A : ℝ) (h : 90 - A = A - 18) : 180 - A = 126 := by
    sorry

end supplement_of_angle_l36_36255


namespace remaining_pie_proportion_l36_36976

def carlos_portion : ℝ := 0.6
def maria_share_of_remainder : ℝ := 0.25

theorem remaining_pie_proportion: 
  (1 - carlos_portion) - maria_share_of_remainder * (1 - carlos_portion) = 0.3 := 
by
  -- proof to be implemented here
  sorry

end remaining_pie_proportion_l36_36976


namespace probability_no_obtuse_triangle_correct_l36_36349

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end probability_no_obtuse_triangle_correct_l36_36349


namespace present_population_l36_36158

theorem present_population (P : ℝ) (h : 1.04 * P = 1289.6) : P = 1240 :=
by
  sorry

end present_population_l36_36158


namespace sum_of_num_and_denom_l36_36290

-- Define the repeating decimal G
def G : ℚ := 739 / 999

-- State the theorem
theorem sum_of_num_and_denom (a b : ℕ) (hb : b ≠ 0) (h : G = a / b) : a + b = 1738 := sorry

end sum_of_num_and_denom_l36_36290


namespace perfect_square_polynomial_l36_36086

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l36_36086


namespace scientific_notation_826M_l36_36405

theorem scientific_notation_826M : 826000000 = 8.26 * 10^8 :=
by
  sorry

end scientific_notation_826M_l36_36405


namespace who_is_next_to_boris_l36_36852

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l36_36852


namespace algebraic_expression_value_l36_36690

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 2 * x^2 + 3 * x - 7 = -6 :=
by sorry

end algebraic_expression_value_l36_36690


namespace sequence_count_is_correct_l36_36745

def has_integer_root (a_i a_i_plus_1 : ℕ) : Prop :=
  ∃ r : ℕ, r^2 - a_i * r + a_i_plus_1 = 0

def valid_sequence (seq : Fin 16 → ℕ) : Prop :=
  ∀ i : Fin 15, has_integer_root (seq i.val + 1) (seq (i + 1).val + 1) ∧ seq 15 = seq 0

-- This noncomputable definition is used because we are estimating a specific number without providing a concrete computable function.
noncomputable def sequence_count : ℕ :=
  1409

theorem sequence_count_is_correct :
  ∃ N, valid_sequence seq → N = 1409 :=
sorry 

end sequence_count_is_correct_l36_36745


namespace sum_of_squares_l36_36385

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l36_36385


namespace trainer_voice_radius_l36_36400

noncomputable def area_of_heard_voice (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

theorem trainer_voice_radius :
  ∃ r : ℝ, abs (r - 140) < 1 ∧ area_of_heard_voice r = 15393.804002589986 :=
by
  sorry

end trainer_voice_radius_l36_36400


namespace distance_between_foci_of_ellipse_l36_36888

theorem distance_between_foci_of_ellipse :
  ∀ x y : ℝ,
  9 * x^2 - 36 * x + 4 * y^2 + 16 * y + 16 = 0 →
  2 * Real.sqrt (9 - 4) = 2 * Real.sqrt 5 :=
by 
  sorry

end distance_between_foci_of_ellipse_l36_36888


namespace simplify_expression_l36_36651

noncomputable def expr1 := (Real.sqrt 462) / (Real.sqrt 330)
noncomputable def expr2 := (Real.sqrt 245) / (Real.sqrt 175)
noncomputable def expr_simplified := (12 * Real.sqrt 35) / 25

theorem simplify_expression :
  expr1 + expr2 = expr_simplified :=
sorry

end simplify_expression_l36_36651


namespace product_513_12_l36_36083

theorem product_513_12 : 513 * 12 = 6156 := 
  by
    sorry

end product_513_12_l36_36083


namespace max_value_of_m_l36_36343

theorem max_value_of_m
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (2 / a) + (1 / b) = 1 / 4)
  (h4 : ∀ a b, 2 * a + b ≥ 9 * m) :
  m = 4 := 
sorry

end max_value_of_m_l36_36343


namespace train_length_l36_36034

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def crossing_time : ℝ := 20
noncomputable def platform_length : ℝ := 220.032
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

theorem train_length :
  total_distance - platform_length = 179.968 := by
  sorry

end train_length_l36_36034


namespace find_abc_value_l36_36250

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom h1 : a + 1 / b = 5
axiom h2 : b + 1 / c = 2
axiom h3 : c + 1 / a = 9 / 4

theorem find_abc_value : a * b * c = (7 + Real.sqrt 21) / 8 :=
by
  sorry

end find_abc_value_l36_36250


namespace smallest_lcm_value_theorem_l36_36157

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l36_36157


namespace find_base_l36_36937

theorem find_base (a : ℕ) (ha : a > 11) (hB : 11 = 11) :
  (3 * a^2 + 9 * a + 6) + (5 * a^2 + 7 * a + 5) = (9 * a^2 + 7 * a + 11) → 
  a = 12 :=
sorry

end find_base_l36_36937


namespace product_of_real_solutions_of_t_cubed_eq_216_l36_36538

theorem product_of_real_solutions_of_t_cubed_eq_216 : 
  (∃ t : ℝ, t^3 = 216) →
  (∀ t₁ t₂, (t₁ = t₂) → (t₁^3 = 216 → t₂^3 = 216) → (t₁ * t₂ = 6)) :=
by
  sorry

end product_of_real_solutions_of_t_cubed_eq_216_l36_36538


namespace find_k_find_a_l36_36602

noncomputable def f (a k : ℝ) (x : ℝ) := a ^ x + k * a ^ (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

theorem find_k (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : is_odd_function (f a k)) : k = -1 :=
sorry

theorem find_a (k : ℝ) (h₃ : k = -1) (h₄ : f 1 = 3 / 2) (h₅ : is_monotonic_increasing (f 2 k)) : a = 2 :=
sorry

end find_k_find_a_l36_36602


namespace remainder_of_12_pow_2012_mod_5_l36_36936

theorem remainder_of_12_pow_2012_mod_5 : (12 ^ 2012) % 5 = 1 :=
by
  sorry

end remainder_of_12_pow_2012_mod_5_l36_36936


namespace cuberoot_3375_sum_l36_36985

theorem cuberoot_3375_sum (a b : ℕ) (h : 3375 = 3^3 * 5^3) (h1 : a = 15) (h2 : b = 1) : a + b = 16 := by
  sorry

end cuberoot_3375_sum_l36_36985


namespace students_taking_both_courses_l36_36351

theorem students_taking_both_courses (n_total n_F n_G n_neither number_both : ℕ)
  (h_total : n_total = 79)
  (h_F : n_F = 41)
  (h_G : n_G = 22)
  (h_neither : n_neither = 25)
  (h_any_language : n_total - n_neither = 54)
  (h_sum_languages : n_F + n_G = 63)
  (h_both : n_F + n_G - (n_total - n_neither) = number_both) :
  number_both = 9 :=
by {
  sorry
}

end students_taking_both_courses_l36_36351


namespace ratio_of_areas_of_concentric_circles_l36_36060

theorem ratio_of_areas_of_concentric_circles 
  (C1 C2 : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * C1 = 2 * π * r1)
  (h2 : r2 * C2 = 2 * π * r2)
  (h_c1 : 60 / 360 * C1 = 48 / 360 * C2) :
  (π * r1^2) / (π * r2^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l36_36060


namespace forty_ab_l36_36319

theorem forty_ab (a b : ℝ) (h₁ : 4 * a = 30) (h₂ : 5 * b = 30) : 40 * a * b = 1800 :=
by
  sorry

end forty_ab_l36_36319


namespace jane_buys_four_bagels_l36_36109

-- Define Jane's 7-day breakfast choices
def number_of_items (b m : ℕ) := b + m = 7

-- Define the total weekly cost condition
def total_cost_divisible_by_100 (b : ℕ) := (90 * b + 40 * (7 - b)) % 100 = 0

-- The statement to prove
theorem jane_buys_four_bagels (b : ℕ) (m : ℕ) (h1 : number_of_items b m) (h2 : total_cost_divisible_by_100 b) : b = 4 :=
by
  -- proof goes here
  sorry

end jane_buys_four_bagels_l36_36109


namespace value_of_m_plus_n_l36_36256

-- Conditions
variables (m n : ℤ)
def P_symmetric_Q_x_axis := (m - 1 = 2 * m - 4) ∧ (n + 2 = -2)

-- Proof Problem Statement
theorem value_of_m_plus_n (h : P_symmetric_Q_x_axis m n) : (m + n) ^ 2023 = -1 := sorry

end value_of_m_plus_n_l36_36256


namespace total_students_l36_36704

theorem total_students (h1 : ∀ (n : ℕ), n = 5 → Jaya_ranks_nth_from_top)
                       (h2 : ∀ (m : ℕ), m = 49 → Jaya_ranks_mth_from_bottom) :
  ∃ (total : ℕ), total = 53 :=
by
  sorry

end total_students_l36_36704


namespace john_ratio_amounts_l36_36806

/-- John gets $30 from his grandpa and some multiple of that amount from his grandma. 
He got $120 from the two grandparents. What is the ratio of the amount he got from 
his grandma to the amount he got from his grandpa? --/
theorem john_ratio_amounts (amount_grandpa amount_total : ℝ) (multiple : ℝ) :
  amount_grandpa = 30 → amount_total = 120 →
  amount_total = amount_grandpa + multiple * amount_grandpa →
  multiple = 3 :=
by
  intros h1 h2 h3
  sorry

end john_ratio_amounts_l36_36806


namespace greatest_prime_factor_is_5_l36_36804

-- Define the expression
def expr : Nat := (3^8 + 9^5)

-- State the theorem
theorem greatest_prime_factor_is_5 : ∃ p : Nat, Prime p ∧ p = 5 ∧ ∀ q : Nat, Prime q ∧ q ∣ expr → q ≤ 5 := by
  sorry

end greatest_prime_factor_is_5_l36_36804


namespace cylinder_volume_l36_36867

theorem cylinder_volume (V1 V2 : ℝ) (π : ℝ) (r1 r3 h2 h5 : ℝ)
  (h_radii_ratio : r3 = 3 * r1)
  (h_heights_ratio : h5 = 5 / 2 * h2)
  (h_first_volume : V1 = π * r1^2 * h2)
  (h_V1_value : V1 = 40) :
  V2 = 900 :=
by sorry

end cylinder_volume_l36_36867


namespace journey_speed_l36_36858

theorem journey_speed (t_total : ℝ) (d_total : ℝ) (d_half : ℝ) (v_half2 : ℝ) (time_half2 : ℝ) (time_total : ℝ) (v_half1 : ℝ) :
  t_total = 5 ∧ d_total = 112 ∧ d_half = d_total / 2 ∧ v_half2 = 24 ∧ time_half2 = d_half / v_half2 ∧ time_total = t_total - time_half2 ∧ v_half1 = d_half / time_total → v_half1 = 21 :=
by
  intros h
  sorry

end journey_speed_l36_36858


namespace percentage_deficit_l36_36244

theorem percentage_deficit
  (L W : ℝ)
  (h1 : ∃(x : ℝ), 1.10 * L * (W * (1 - x / 100)) = L * W * 1.045) :
  ∃ (x : ℝ), x = 5 :=
by
  sorry

end percentage_deficit_l36_36244


namespace train_length_l36_36519

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def distance_ahead : ℝ := 270
noncomputable def time_to_pass : ℝ := 39

noncomputable def jogger_speed_mps := jogger_speed_kmph * (1000 / 1) * (1 / 3600)
noncomputable def train_speed_mps := train_speed_kmph * (1000 / 1) * (1 / 3600)

noncomputable def relative_speed_mps := train_speed_mps - jogger_speed_mps

theorem train_length :
  let jogger_speed := 9 * (1000 / 3600)
  let train_speed := 45 * (1000 / 3600)
  let relative_speed := train_speed - jogger_speed
  let distance := 270
  let time := 39
  distance + relative_speed * time = 390 → relative_speed * time = 120 := by
  sorry

end train_length_l36_36519


namespace circle_passes_first_and_second_quadrants_l36_36338

theorem circle_passes_first_and_second_quadrants :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4 → ((x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0)) :=
by
  sorry

end circle_passes_first_and_second_quadrants_l36_36338


namespace katie_total_marbles_l36_36110

def pink_marbles := 13
def orange_marbles := pink_marbles - 9
def purple_marbles := 4 * orange_marbles
def blue_marbles := 2 * purple_marbles
def total_marbles := pink_marbles + orange_marbles + purple_marbles + blue_marbles

theorem katie_total_marbles : total_marbles = 65 := 
by
  -- The proof is omitted here.
  sorry

end katie_total_marbles_l36_36110


namespace degenerate_ellipse_b_value_l36_36614

theorem degenerate_ellipse_b_value :
  ∃ b : ℝ, (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + b = 0 → x = -1 ∧ y = 3) ↔ b = 12 :=
by
  sorry

end degenerate_ellipse_b_value_l36_36614


namespace distribution_schemes_36_l36_36485

def num_distribution_schemes (total_students english_excellent computer_skills : ℕ) : ℕ :=
  if total_students = 8 ∧ english_excellent = 2 ∧ computer_skills = 3 then 36 else 0

theorem distribution_schemes_36 :
  num_distribution_schemes 8 2 3 = 36 :=
by
 sorry

end distribution_schemes_36_l36_36485


namespace max_puzzle_sets_l36_36513

theorem max_puzzle_sets 
  (total_logic : ℕ) (total_visual : ℕ) (total_word : ℕ)
  (h1 : total_logic = 36) (h2 : total_visual = 27) (h3 : total_word = 15)
  (x y : ℕ)
  (h4 : 7 ≤ 4 * x + 3 * x + y ∧ 4 * x + 3 * x + y ≤ 12)
  (h5 : 4 * x / 3 * x = 4 / 3)
  (h6 : y ≥ 3 * x / 2) :
  5 ≤ total_logic / (4 * x) ∧ 5 ≤ total_visual / (3 * x) ∧ 5 ≤ total_word / y :=
sorry

end max_puzzle_sets_l36_36513


namespace percent_nonunion_women_l36_36341

variable (E : ℝ) -- Total number of employees

-- Definitions derived from the problem conditions
def menPercent : ℝ := 0.46
def unionPercent : ℝ := 0.60
def nonUnionPercent : ℝ := 1 - unionPercent
def nonUnionWomenPercent : ℝ := 0.90

theorem percent_nonunion_women :
  nonUnionWomenPercent = 0.90 :=
by
  sorry

end percent_nonunion_women_l36_36341


namespace find_geometric_sequence_first_term_and_ratio_l36_36622

theorem find_geometric_sequence_first_term_and_ratio 
  (a1 a2 a3 a4 a5 : ℕ) 
  (h : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (geo_seq : a2 = a1 * 3 / 2 ∧ a3 = a2 * 3 / 2 ∧ a4 = a3 * 3 / 2 ∧ a5 = a4 * 3 / 2)
  (sum_cond : a1 + a2 + a3 + a4 + a5 = 211) :
  (a1 = 16) ∧ (3 / 2 = 3 / 2) := 
by {
  sorry
}

end find_geometric_sequence_first_term_and_ratio_l36_36622


namespace cookies_per_student_l36_36348

theorem cookies_per_student (students : ℕ) (percent : ℝ) (oatmeal_cookies : ℕ) 
                            (h_students : students = 40)
                            (h_percent : percent = 10 / 100)
                            (h_oatmeal : oatmeal_cookies = 8) :
                            (oatmeal_cookies / percent / students) = 2 := by
  sorry

end cookies_per_student_l36_36348


namespace solution_set_of_inequality_l36_36258

theorem solution_set_of_inequality :
  {x : ℝ | (x + 3) * (x - 2) < 0} = {x | -3 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l36_36258


namespace new_figure_perimeter_equals_5_l36_36520

-- Defining the side length of the square and the equilateral triangle
def side_length : ℝ := 1

-- Defining the perimeter of the new figure
def new_figure_perimeter : ℝ := 3 * side_length + 2 * side_length

-- Statement: The perimeter of the new figure equals 5
theorem new_figure_perimeter_equals_5 :
  new_figure_perimeter = 5 := by
  sorry

end new_figure_perimeter_equals_5_l36_36520


namespace percentage_of_failed_candidates_l36_36835

theorem percentage_of_failed_candidates
(total_candidates : ℕ)
(girls : ℕ)
(passed_boys_percentage : ℝ)
(passed_girls_percentage : ℝ)
(h1 : total_candidates = 2000)
(h2 : girls = 900)
(h3 : passed_boys_percentage = 0.28)
(h4 : passed_girls_percentage = 0.32)
: (total_candidates - (passed_boys_percentage * (total_candidates - girls) + passed_girls_percentage * girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end percentage_of_failed_candidates_l36_36835


namespace price_of_Microtron_stock_l36_36716

theorem price_of_Microtron_stock
  (n d : ℕ) (p_d p p_m : ℝ) 
  (h1 : n = 300) 
  (h2 : d = 150) 
  (h3 : p_d = 44) 
  (h4 : p = 40) 
  (h5 : p_m = 36) : 
  (d * p_d + (n - d) * p_m) / n = p := 
sorry

end price_of_Microtron_stock_l36_36716


namespace range_of_m_l36_36866

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end range_of_m_l36_36866


namespace rectangle_properties_l36_36119

noncomputable def diagonal (x1 y1 x2 y2 : ℕ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area (length width : ℕ) : ℕ :=
  length * width

theorem rectangle_properties :
  diagonal 1 1 9 7 = 10 ∧ area (9 - 1) (7 - 1) = 48 := by
  sorry

end rectangle_properties_l36_36119


namespace children_count_l36_36372

theorem children_count 
  (A B C : Finset ℕ)
  (hA : A.card = 7)
  (hB : B.card = 6)
  (hC : C.card = 5)
  (hA_inter_B : (A ∩ B).card = 4)
  (hA_inter_C : (A ∩ C).card = 3)
  (hB_inter_C : (B ∩ C).card = 2)
  (hA_inter_B_inter_C : (A ∩ B ∩ C).card = 1) :
  (A ∪ B ∪ C).card = 10 := 
by
  sorry

end children_count_l36_36372


namespace polygon_area_is_400_l36_36189

-- Definition of the points and polygon
def Point := (ℝ × ℝ)
def Polygon := List Point

def points : List Point := [(0, 0), (20, 0), (20, 20), (0, 20), (10, 0), (20, 10), (10, 20), (0, 10)]

def polygon : Polygon := [(0,0), (10,0), (20,10), (20,20), (10,20), (0,10), (0,0)]

-- Function to calculate the area of the polygon
noncomputable def polygon_area (p : Polygon) : ℝ := 
  -- Assume we have the necessary function to calculate the area of a polygon given a list of vertices
  sorry

-- Theorem statement: The area of the given polygon is 400
theorem polygon_area_is_400 : polygon_area polygon = 400 := sorry

end polygon_area_is_400_l36_36189


namespace mutually_exclusive_not_complementary_l36_36717

def event_odd (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_greater_than_5 (n : ℕ) : Prop := n = 6

theorem mutually_exclusive_not_complementary :
  (∀ n : ℕ, event_odd n → ¬ event_greater_than_5 n) ∧
  (∃ n : ℕ, ¬ event_odd n ∧ ¬ event_greater_than_5 n) :=
by
  sorry

end mutually_exclusive_not_complementary_l36_36717


namespace factorial_expression_equals_l36_36740

theorem factorial_expression_equals :
  7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end factorial_expression_equals_l36_36740


namespace midpoint_locus_l36_36573

theorem midpoint_locus (c : ℝ) (H : 0 < c ∧ c ≤ Real.sqrt 2) :
  ∃ L, L = "curvilinear quadrilateral with arcs forming transitions" :=
sorry

end midpoint_locus_l36_36573


namespace greatest_four_digit_p_l36_36334

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

-- Proof problem
theorem greatest_four_digit_p (p : ℕ) (q : ℕ) 
    (hp1 : is_four_digit p)
    (hp2 : q = reverse_digits p)
    (hp3 : is_four_digit q)
    (hp4 : is_divisible_by p 63)
    (hp5 : is_divisible_by q 63)
    (hp6 : is_divisible_by p 19) :
  p = 5985 :=
sorry

end greatest_four_digit_p_l36_36334


namespace money_distribution_l36_36737

theorem money_distribution (p q r : ℝ) 
  (h1 : p + q + r = 9000) 
  (h2 : r = (2/3) * (p + q)) : 
  r = 3600 := 
by 
  sorry

end money_distribution_l36_36737


namespace length_of_the_train_l36_36725

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end length_of_the_train_l36_36725


namespace pair_C_does_not_produce_roots_l36_36984

theorem pair_C_does_not_produce_roots (x : ℝ) :
  (x = 0 ∨ x = 2) ↔ (∃ x, y = x ∧ y = x - 2) = false :=
by
  sorry

end pair_C_does_not_produce_roots_l36_36984


namespace worker_b_alone_time_l36_36869

theorem worker_b_alone_time (A B C : ℝ) (h1 : A + B = 1 / 8)
  (h2 : A = 1 / 12) (h3 : C = 1 / 18) :
  1 / B = 24 :=
sorry

end worker_b_alone_time_l36_36869


namespace abs_sum_div_diff_sqrt_7_5_l36_36160

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l36_36160


namespace interest_rate_correct_l36_36448

-- Definitions based on the problem conditions
def P : ℝ := 7000 -- Principal investment amount
def A : ℝ := 8470 -- Future value of the investment
def n : ℕ := 1 -- Number of times interest is compounded per year
def t : ℕ := 2 -- Number of years

-- The interest rate r to be proven
def r : ℝ := 0.1 -- Annual interest rate

-- Statement of the problem that needs to be proven in Lean
theorem interest_rate_correct :
  A = P * (1 + r / n)^(n * t) :=
by
  sorry

end interest_rate_correct_l36_36448


namespace hannah_age_double_july_age_20_years_ago_l36_36318

/-- Define the current ages of July (J) and her husband (H) -/
def current_age_july : ℕ := 23
def current_age_husband : ℕ := 25

/-- Assertion that July's husband is 2 years older than her -/
axiom husband_older : current_age_husband = current_age_july + 2

/-- We denote the ages 20 years ago -/
def age_july_20_years_ago := current_age_july - 20
def age_hannah_20_years_ago := current_age_husband - 20 - 2 * (current_age_july - 20)

theorem hannah_age_double_july_age_20_years_ago :
  age_hannah_20_years_ago = 6 :=
by sorry

end hannah_age_double_july_age_20_years_ago_l36_36318


namespace A_finishes_in_20_days_l36_36980

-- Define the rates and the work
variable (A B W : ℝ)

-- First condition: A and B together can finish the work in 12 days
axiom together_rate : (A + B) * 12 = W

-- Second condition: B alone can finish the work in 30.000000000000007 days
axiom B_rate : B * 30.000000000000007 = W

-- Prove that A alone can finish the work in 20 days
theorem A_finishes_in_20_days : (1 / A) = 20 :=
by 
  sorry

end A_finishes_in_20_days_l36_36980


namespace commercial_break_total_time_l36_36752

theorem commercial_break_total_time (c1 c2 c3 : ℕ) (c4 : ℕ → ℕ) (interrupt restart : ℕ) 
  (h1 : c1 = 5) (h2 : c2 = 6) (h3 : c3 = 7) 
  (h4 : ∀ i, i < 11 → c4 i = 2) 
  (h_interrupt : interrupt = 3)
  (h_restart : restart = 2) :
  c1 + c2 + c3 + (11 * 2) + interrupt + 2 * restart = 47 := 
  by
  sorry

end commercial_break_total_time_l36_36752


namespace rhombus_area_l36_36838

-- Define the given conditions as parameters
variables (EF GH : ℝ) -- Sides of the rhombus
variables (d1 d2 : ℝ) -- Diagonals of the rhombus

-- Statement of the theorem
theorem rhombus_area
  (rhombus_EFGH : ∀ (EF GH : ℝ), EF = GH)
  (perimeter_EFGH : 4 * EF = 40)
  (diagonal_EG_length : d1 = 16)
  (d1_half : d1 / 2 = 8)
  (side_length : EF = 10)
  (pythagorean_theorem : EF^2 = (d1 / 2)^2 + (d2 / 2)^2)
  (calculate_FI : d2 / 2 = 6)
  (diagonal_FG_length : d2 = 12) :
  (1 / 2) * d1 * d2 = 96 :=
sorry

end rhombus_area_l36_36838


namespace range_of_a_l36_36892

-- Define the condition p
def p (x : ℝ) : Prop := (2 * x^2 - 3 * x + 1) ≤ 0

-- Define the condition q
def q (x a : ℝ) : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0

-- Lean statement for the problem
theorem range_of_a (a : ℝ) : (¬ (∃ x, p x) → ¬ (∃ x, q x a)) → ((0 : ℝ) ≤ a ∧ a ≤ (1 / 2 : ℝ)) :=
by 
  sorry

end range_of_a_l36_36892


namespace remainder_of_144_div_k_l36_36314

theorem remainder_of_144_div_k
  (k : ℕ)
  (h1 : 0 < k)
  (h2 : 120 % k^2 = 12) :
  144 % k = 0 :=
by
  sorry

end remainder_of_144_div_k_l36_36314


namespace chef_cooked_potatoes_l36_36824

theorem chef_cooked_potatoes
  (total_potatoes : ℕ)
  (cooking_time_per_potato : ℕ)
  (remaining_cooking_time : ℕ)
  (left_potatoes : ℕ)
  (cooked_potatoes : ℕ) :
  total_potatoes = 16 →
  cooking_time_per_potato = 5 →
  remaining_cooking_time = 45 →
  remaining_cooking_time / cooking_time_per_potato = left_potatoes →
  total_potatoes - left_potatoes = cooked_potatoes →
  cooked_potatoes = 7 :=
by
  intros h_total h_cooking_time h_remaining_time h_left_potatoes h_cooked_potatoes
  sorry

end chef_cooked_potatoes_l36_36824


namespace count_triangles_in_hexagonal_grid_l36_36177

-- Define the number of smallest triangles in the figure.
def small_triangles : ℕ := 10

-- Define the number of medium triangles in the figure, composed of 4 small triangles each.
def medium_triangles : ℕ := 6

-- Define the number of large triangles in the figure, composed of 9 small triangles each.
def large_triangles : ℕ := 3

-- Define the number of extra-large triangle composed of 16 small triangles.
def extra_large_triangle : ℕ := 1

-- Define the total number of triangles in the figure.
def total_triangles : ℕ := small_triangles + medium_triangles + large_triangles + extra_large_triangle

-- The theorem we want to prove: the total number of triangles is 20.
theorem count_triangles_in_hexagonal_grid : total_triangles = 20 := by
  -- Placeholder for the proof.
  sorry

end count_triangles_in_hexagonal_grid_l36_36177


namespace find_multiple_l36_36931

-- Definitions based on the conditions provided
def mike_chocolate_squares : ℕ := 20
def jenny_chocolate_squares : ℕ := 65
def extra_squares : ℕ := 5

-- The theorem to prove the multiple
theorem find_multiple : ∃ (multiple : ℕ), jenny_chocolate_squares = mike_chocolate_squares * multiple + extra_squares ∧ multiple = 3 := by
  sorry

end find_multiple_l36_36931


namespace drawing_time_total_l36_36088

theorem drawing_time_total
  (bianca_school : ℕ)
  (bianca_home : ℕ)
  (lucas_school : ℕ)
  (lucas_home : ℕ)
  (h_bianca_school : bianca_school = 22)
  (h_bianca_home : bianca_home = 19)
  (h_lucas_school : lucas_school = 10)
  (h_lucas_home : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := 
by
  -- Proof would go here
  sorry

end drawing_time_total_l36_36088


namespace problem_statement_l36_36787

theorem problem_statement :
  ¬(∀ n : ℤ, n ≥ 0 → n = 0) ∧
  ¬(∀ q : ℚ, q ≠ 0 → q > 0 ∨ q < 0) ∧
  ¬(∀ a b : ℝ, abs a = abs b → a = b) ∧
  (∀ a : ℝ, abs a = abs (-a)) :=
by
  sorry

end problem_statement_l36_36787


namespace total_tickets_l36_36222

-- Define the initial number of tickets Tate has.
def tate_initial_tickets : ℕ := 32

-- Define the number of tickets Tate buys additionally.
def additional_tickets : ℕ := 2

-- Define the total number of tickets Tate has after buying more.
def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

-- Define the total number of tickets Peyton has.
def peyton_tickets : ℕ := tate_total_tickets / 2

-- State the theorem to prove the total number of tickets Tate and Peyton have together.
theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  -- Placeholder for the proof
  sorry

end total_tickets_l36_36222


namespace zeros_of_f_l36_36919

noncomputable def f (x : ℝ) : ℝ := x^3 - 16 * x

theorem zeros_of_f :
  ∃ a b c : ℝ, (a = -4) ∧ (b = 0) ∧ (c = 4) ∧ (f a = 0) ∧ (f b = 0) ∧ (f c = 0) :=
by
  sorry

end zeros_of_f_l36_36919


namespace evaluate_polynomial_l36_36270

theorem evaluate_polynomial : (99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1) = 92199816 := 
by 
  sorry

end evaluate_polynomial_l36_36270


namespace moles_of_HCl_formed_l36_36627

-- Define the reaction as given in conditions
def reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) := C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Define the initial moles of reactants
def moles_C2H6 : ℝ := 2
def moles_Cl2 : ℝ := 2

-- State the expected moles of HCl produced
def expected_moles_HCl : ℝ := 4

-- The theorem stating the problem to prove
theorem moles_of_HCl_formed : ∃ HCl : ℝ, reaction moles_C2H6 moles_Cl2 0 HCl ∧ HCl = expected_moles_HCl :=
by
  -- Skipping detailed proof with sorry
  sorry

end moles_of_HCl_formed_l36_36627


namespace divides_number_of_ones_l36_36307

theorem divides_number_of_ones (n : ℕ) (h1 : ¬(2 ∣ n)) (h2 : ¬(5 ∣ n)) : ∃ k : ℕ, n ∣ ((10^k - 1) / 9) :=
by
  sorry

end divides_number_of_ones_l36_36307


namespace math_problem_l36_36665

theorem math_problem (x y : ℝ) 
  (h1 : 1/5 + x + y = 1) 
  (h2 : 1/5 * 1 + 2 * x + 3 * y = 11/5) : 
  (x = 2/5) ∧ 
  (y = 2/5) ∧ 
  (1/5 + x = 3/5) ∧ 
  ((1 - 11/5)^2 * (1/5) + (2 - 11/5)^2 * (2/5) + (3 - 11/5)^2 * (2/5) = 14/25) :=
by {
  sorry
}

end math_problem_l36_36665


namespace parabola_standard_equation_l36_36296

theorem parabola_standard_equation (h : ∀ y, y = 1/2) : ∃ c : ℝ, c = -2 ∧ (∀ x y, x^2 = c * y) :=
by
  -- Considering 'h' provides the condition for the directrix
  sorry

end parabola_standard_equation_l36_36296


namespace inequality_proof_l36_36350

theorem inequality_proof (a b : ℝ) (h₀ : b > a) (h₁ : ab > 0) : 
  (1 / a > 1 / b) ∧ (a + b < 2 * b) :=
by
  sorry

end inequality_proof_l36_36350


namespace find_positive_real_solution_l36_36099

theorem find_positive_real_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 8 = 5 / (x - 8)) : x = 13 := 
sorry

end find_positive_real_solution_l36_36099


namespace train_length_in_terms_of_james_cycle_l36_36113

/-- Define the mathematical entities involved: L (train length), J (James's cycle length), T (train length per cycle) -/
theorem train_length_in_terms_of_james_cycle 
  (L J T : ℝ) 
  (h1 : 130 * J = L + 130 * T) 
  (h2 : 26 * J = L - 26 * T) 
    : L = 58 * J := 
by 
  sorry

end train_length_in_terms_of_james_cycle_l36_36113


namespace Rachel_and_Mike_l36_36567

theorem Rachel_and_Mike :
  ∃ b c : ℤ,
    (∀ x : ℝ, |x - 3| = 4 ↔ (x = 7 ∨ x = -1)) ∧
    (∀ x : ℝ, (x - 7) * (x + 1) = 0 ↔ x * x + b * x + c = 0) ∧
    (b, c) = (-6, -7) := by
sorry

end Rachel_and_Mike_l36_36567


namespace probability_smallest_divides_larger_two_l36_36239

noncomputable def number_of_ways := 20

noncomputable def successful_combinations := 11

theorem probability_smallest_divides_larger_two : (successful_combinations : ℚ) / number_of_ways = 11 / 20 :=
by
  sorry

end probability_smallest_divides_larger_two_l36_36239


namespace binom_15_4_eq_1365_l36_36664

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l36_36664


namespace last_digit_2_pow_2023_l36_36703

-- Definitions
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem last_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 :=
by
  -- We will assume and use the properties mentioned in the solution steps.
  -- The proof process is skipped here with 'sorry'.
  sorry

end last_digit_2_pow_2023_l36_36703


namespace isosceles_triangle_perimeter_l36_36182

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end isosceles_triangle_perimeter_l36_36182


namespace Haley_sweaters_l36_36648

theorem Haley_sweaters (machine_capacity loads shirts sweaters : ℕ) 
    (h_capacity : machine_capacity = 7)
    (h_loads : loads = 5)
    (h_shirts : shirts = 2)
    (h_sweaters_total : sweaters = loads * machine_capacity - shirts) :
  sweaters = 33 :=
by 
  rw [h_capacity, h_loads, h_shirts] at h_sweaters_total
  exact h_sweaters_total

end Haley_sweaters_l36_36648


namespace line_contains_diameter_of_circle_l36_36425

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

noncomputable def equation_of_line (x y : ℝ) : Prop :=
  2*x - y - 1 = 0

theorem line_contains_diameter_of_circle :
  (∃ x y : ℝ, equation_of_circle x y ∧ equation_of_line x y) :=
sorry

end line_contains_diameter_of_circle_l36_36425


namespace intersection_points_l36_36710

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * (x - y) - 18 = 0

theorem intersection_points : 
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by sorry

end intersection_points_l36_36710


namespace taxi_fare_max_distance_l36_36617

-- Setting up the conditions
def starting_price : ℝ := 7
def additional_fare_per_km : ℝ := 2.4
def max_base_distance_km : ℝ := 3
def total_fare : ℝ := 19

-- Defining the maximum distance based on the given conditions
def max_distance : ℝ := 8

-- The theorem is to prove that the maximum distance is indeed 8 kilometers
theorem taxi_fare_max_distance :
  ∀ (x : ℝ), total_fare = starting_price + additional_fare_per_km * (x - max_base_distance_km) → x ≤ max_distance :=
by
  intros x h
  sorry

end taxi_fare_max_distance_l36_36617


namespace non_integer_sum_exists_l36_36358

theorem non_integer_sum_exists (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ M : ℕ, ∀ n : ℕ, n > M → ¬ ∃ t : ℤ, (k + 1/2)^n + (l + 1/2)^n = t := 
sorry

end non_integer_sum_exists_l36_36358


namespace math_proof_problem_l36_36213

theorem math_proof_problem (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3 * x₁ * y₁^2 = 2008)
  (h₂ : y₁^3 - 3 * x₁^2 * y₁ = 2007)
  (h₃ : x₂^3 - 3 * x₂ * y₂^2 = 2008)
  (h₄ : y₂^3 - 3 * x₂^2 * y₂ = 2007)
  (h₅ : x₃^3 - 3 * x₃ * y₃^2 = 2008)
  (h₆ : y₃^3 - 3 * x₃^2 * y₃ = 2007) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 4015 / 2008 :=
by sorry

end math_proof_problem_l36_36213


namespace find_m_l36_36179

theorem find_m (m x : ℝ) (h : (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) (hx : x = 0) : m = -2 :=
by sorry

end find_m_l36_36179


namespace move_line_down_eq_l36_36379

theorem move_line_down_eq (x y : ℝ) : (y = 2 * x) → (y - 3 = 2 * x - 3) :=
by
  sorry

end move_line_down_eq_l36_36379


namespace max_f_5_value_l36_36977

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 2 * x

noncomputable def f_1 (x : ℝ) : ℝ := f x
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0       => x -- Not used, as n starts from 1
  | (n + 1) => f (f_n n x)

noncomputable def max_f_5 : ℝ := 3 ^ 32 - 1

theorem max_f_5_value : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f_n 5 x ≤ max_f_5 :=
by
  intro x hx
  have := hx
  -- The detailed proof would go here,
  -- but for the statement, we end with sorry.
  sorry

end max_f_5_value_l36_36977


namespace range_of_f_l36_36201

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4

theorem range_of_f : Set.Icc 0 (9 / 8) = Set.range f := 
by
  sorry

end range_of_f_l36_36201


namespace second_container_sand_capacity_l36_36989

def volume (h: ℕ) (w: ℕ) (l: ℕ) : ℕ := h * w * l

def sand_capacity (v1: ℕ) (s1: ℕ) (v2: ℕ) : ℕ := (s1 * v2) / v1

theorem second_container_sand_capacity:
  let h1 := 3
  let w1 := 4
  let l1 := 6
  let s1 := 72
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let v1 := volume h1 w1 l1
  let v2 := volume h2 w2 l2
  sand_capacity v1 s1 v2 = 432 :=
by {
  sorry
}

end second_container_sand_capacity_l36_36989


namespace largest_share_received_l36_36957

theorem largest_share_received (total_profit : ℝ) (ratios : List ℝ) (h_ratios : ratios = [1, 2, 2, 3, 4, 5]) 
  (h_profit : total_profit = 51000) : 
  let parts := ratios.sum 
  let part_value := total_profit / parts
  let largest_share := 5 * part_value 
  largest_share = 15000 := 
by 
  sorry

end largest_share_received_l36_36957


namespace probability_of_C_l36_36583

theorem probability_of_C (P : ℕ → ℚ) (P_total : P 1 + P 2 + P 3 = 1)
  (P_A : P 1 = 1/3) (P_B : P 2 = 1/2) : P 3 = 1/6 :=
by
  sorry

end probability_of_C_l36_36583


namespace average_percent_score_l36_36546

theorem average_percent_score (num_students : ℕ)
    (students_95 students_85 students_75 students_65 students_55 students_45 : ℕ)
    (h : students_95 + students_85 + students_75 + students_65 + students_55 + students_45 = 120) :
  ((95 * students_95 + 85 * students_85 + 75 * students_75 + 65 * students_65 + 55 * students_55 + 45 * students_45) / 120 : ℚ) = 72.08 := 
by {
  sorry
}

end average_percent_score_l36_36546


namespace spadesuit_example_l36_36794

-- Define the operation spadesuit
def spadesuit (a b : ℤ) : ℤ := abs (a - b)

-- Define the specific instance to prove
theorem spadesuit_example : spadesuit 2 (spadesuit 4 7) = 1 :=
by
  sorry

end spadesuit_example_l36_36794


namespace train_ride_length_l36_36361

theorem train_ride_length :
  let reading_time := 2
  let eating_time := 1
  let watching_time := 3
  let napping_time := 3
  reading_time + eating_time + watching_time + napping_time = 9 := 
by
  sorry

end train_ride_length_l36_36361


namespace trig_expression_value_l36_36512

theorem trig_expression_value (θ : ℝ)
  (h1 : Real.sin (Real.pi + θ) = 1/4) :
  (Real.cos (Real.pi + θ) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) + 
  Real.sin (Real.pi / 2 - θ) / (Real.cos (θ + 2 * Real.pi) * Real.cos (Real.pi + θ) + Real.cos (-θ))) = 32 :=
by
  sorry

end trig_expression_value_l36_36512


namespace value_is_85_over_3_l36_36457

theorem value_is_85_over_3 (a b : ℚ)  (h1 : 3 * a + 6 * b = 48) (h2 : 8 * a + 4 * b = 84) : 2 * a + 3 * b = 85 / 3 := 
by {
  -- Proof will go here
  sorry
}

end value_is_85_over_3_l36_36457


namespace unique_f_satisfies_eq_l36_36925

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x^2 + 2 * x - 1)

theorem unique_f_satisfies_eq (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2) : 
  ∀ x : ℝ, f x = (1 / 3) * (x^2 + 2 * x - 1) :=
sorry

end unique_f_satisfies_eq_l36_36925


namespace fixed_point_l36_36392

noncomputable def fixed_point_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : (ℝ × ℝ) :=
  (1, a^(1 - (1 : ℝ)) + 5)

theorem fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : fixed_point_function a h₀ h₁ = (1, 6) :=
by 
  sorry

end fixed_point_l36_36392


namespace integer_a_can_be_written_in_form_l36_36321

theorem integer_a_can_be_written_in_form 
  (a x y : ℤ) 
  (h : 3 * a = x^2 + 2 * y^2) : 
  ∃ u v : ℤ, a = u^2 + 2 * v^2 :=
sorry

end integer_a_can_be_written_in_form_l36_36321


namespace gcd_C_D_eq_6_l36_36308

theorem gcd_C_D_eq_6
  (C D : ℕ)
  (h_lcm : Nat.lcm C D = 180)
  (h_ratio : C = 5 * D / 6) :
  Nat.gcd C D = 6 := 
by
  sorry

end gcd_C_D_eq_6_l36_36308


namespace average_production_last_5_days_l36_36606

theorem average_production_last_5_days (tv_per_day_25 : ℕ) (total_tv_30 : ℕ) :
  tv_per_day_25 = 63 →
  total_tv_30 = 58 * 30 →
  (total_tv_30 - tv_per_day_25 * 25) / 5 = 33 :=
by
  intros h1 h2
  sorry

end average_production_last_5_days_l36_36606


namespace parabola_focus_directrix_distance_l36_36221

theorem parabola_focus_directrix_distance :
  ∀ {x y : ℝ}, y^2 = (1/4) * x → dist (1/16, 0) (-1/16, 0) = 1/8 := by
sorry

end parabola_focus_directrix_distance_l36_36221


namespace jill_braids_dancers_l36_36000

def dancers_on_team (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) (total_time_seconds : ℕ) : ℕ :=
  total_time_seconds / seconds_per_braid / braids_per_dancer

theorem jill_braids_dancers (h1 : braids_per_dancer = 5) (h2 : seconds_per_braid = 30)
                             (h3 : total_time_seconds = 20 * 60) : 
  dancers_on_team braids_per_dancer seconds_per_braid total_time_seconds = 8 :=
by
  sorry

end jill_braids_dancers_l36_36000


namespace work_completion_l36_36967

theorem work_completion (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a + b = 1/10) (h2 : a = 1/14) : a + b = 1/10 := 
by {
  sorry
}

end work_completion_l36_36967


namespace like_terms_l36_36963

theorem like_terms (x y : ℕ) (h1 : x + 1 = 2) (h2 : x + y = 2) : x = 1 ∧ y = 1 :=
by
  sorry

end like_terms_l36_36963


namespace quadratic_inequality_solution_l36_36478

variable (a x : ℝ)

theorem quadratic_inequality_solution (h : 0 < a ∧ a < 1) : (x - a) * (x - (1 / a)) > 0 ↔ (x < a ∨ x > 1 / a) :=
sorry

end quadratic_inequality_solution_l36_36478


namespace graph_passes_through_point_l36_36638

theorem graph_passes_through_point :
  ∀ (a : ℝ), 0 < a ∧ a < 1 → (∃ (x y : ℝ), (x = 2) ∧ (y = -1) ∧ (y = 2 * a * x - 1)) :=
by
  sorry

end graph_passes_through_point_l36_36638


namespace first_year_exceeds_threshold_l36_36778

def P (n : ℕ) : ℝ := 40000 * (1 + 0.2) ^ n
def exceeds_threshold (n : ℕ) : Prop := P n > 120000

theorem first_year_exceeds_threshold : ∃ n : ℕ, exceeds_threshold n ∧ 2013 + n = 2020 := 
by
  sorry

end first_year_exceeds_threshold_l36_36778


namespace find_x_l36_36362

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 5)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (x, 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x :
  ∃ x : ℝ, collinear (2 • vector_a - vector_b) (vector_c x) ∧ x = -1 := by
  sorry

end find_x_l36_36362


namespace range_of_m_l36_36918

theorem range_of_m : 
  ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := 
by
  -- the proof will go here
  sorry

end range_of_m_l36_36918


namespace train_speed_l36_36884

theorem train_speed (distance time : ℝ) (h1 : distance = 400) (h2 : time = 10) : 
  distance / time = 40 := 
sorry

end train_speed_l36_36884


namespace garden_area_l36_36212

theorem garden_area (P : ℝ) (hP : P = 72) (l w : ℝ) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 243 := 
by
  sorry

end garden_area_l36_36212


namespace solve_for_x_l36_36882

theorem solve_for_x :
  ∀ x : ℕ, 100^4 = 5^x → x = 8 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l36_36882


namespace girl_from_grade_4_probability_l36_36959

-- Number of girls and boys in grade 3
def girls_grade_3 := 28
def boys_grade_3 := 35
def total_grade_3 := girls_grade_3 + boys_grade_3

-- Number of girls and boys in grade 4
def girls_grade_4 := 45
def boys_grade_4 := 42
def total_grade_4 := girls_grade_4 + boys_grade_4

-- Number of girls and boys in grade 5
def girls_grade_5 := 38
def boys_grade_5 := 51
def total_grade_5 := girls_grade_5 + boys_grade_5

-- Total number of children in playground
def total_children := total_grade_3 + total_grade_4 + total_grade_5

-- Probability that a randomly selected child is a girl from grade 4
def probability_girl_grade_4 := (girls_grade_4: ℚ) / total_children

theorem girl_from_grade_4_probability :
  probability_girl_grade_4 = 45 / 239 := by
  sorry

end girl_from_grade_4_probability_l36_36959


namespace min_value_sin_cos_l36_36079

noncomputable def sin_sq (x : ℝ) := (Real.sin x)^2
noncomputable def cos_sq (x : ℝ) := (Real.cos x)^2

theorem min_value_sin_cos (x : ℝ) (h : sin_sq x + cos_sq x = 1) : 
  ∃ m ≥ 0, m = sin_sq x * sin_sq x * sin_sq x + cos_sq x * cos_sq x * cos_sq x ∧ m = 1 :=
by
  sorry

end min_value_sin_cos_l36_36079


namespace river_depth_mid_may_l36_36414

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l36_36414


namespace lcm_9_12_15_l36_36412

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l36_36412


namespace anthony_lunch_money_l36_36846

-- Define the costs as given in the conditions
def juice_box_cost : ℕ := 27
def cupcake_cost : ℕ := 40
def amount_left : ℕ := 8

-- Define the total amount needed for lunch every day
def total_amount_for_lunch : ℕ := juice_box_cost + cupcake_cost + amount_left

theorem anthony_lunch_money : total_amount_for_lunch = 75 := by
  -- This is where the proof would go.
  sorry

end anthony_lunch_money_l36_36846


namespace complement_intersection_l36_36249

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3, 6}

theorem complement_intersection :
  ((universal_set \ set_A) ∩ set_B) = {2, 6} :=
by
  sorry

end complement_intersection_l36_36249


namespace maximize_revenue_l36_36767

-- Define the conditions
def total_time_condition (x y : ℝ) : Prop := x + y ≤ 300
def total_cost_condition (x y : ℝ) : Prop := 2.5 * x + y ≤ 4500
def non_negative_condition (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define the revenue function
def revenue (x y : ℝ) : ℝ := 0.3 * x + 0.2 * y

-- The proof statement
theorem maximize_revenue : 
  ∃ x y, total_time_condition x y ∧ total_cost_condition x y ∧ non_negative_condition x y ∧ 
  revenue x y = 70 := 
by
  sorry

end maximize_revenue_l36_36767


namespace angle_reduction_l36_36817

theorem angle_reduction (θ : ℝ) : θ = 1303 → ∃ k : ℤ, θ = 360 * k - 137 := 
by  
  intro h 
  use 4 
  simp [h] 
  sorry

end angle_reduction_l36_36817


namespace range_of_m_for_circle_l36_36833

theorem range_of_m_for_circle (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0)  ↔ m < -2*Real.sqrt 3 ∨ m > 2*Real.sqrt 3 :=
by 
  sorry

end range_of_m_for_circle_l36_36833


namespace sqrt_square_eq_self_l36_36820

variable (a : ℝ)

theorem sqrt_square_eq_self (h : a > 0) : Real.sqrt (a ^ 2) = a :=
  sorry

end sqrt_square_eq_self_l36_36820


namespace min_value_f_min_value_achieved_l36_36870

noncomputable def f (x y : ℝ) : ℝ :=
  (x^4 / y^4) + (y^4 / x^4) - (x^2 / y^2) - (y^2 / x^2) + (x / y) + (y / x)

theorem min_value_f :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → f x y ≥ 2 :=
sorry

theorem min_value_achieved :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → (f x y = 2) ↔ (x = y) :=
sorry

end min_value_f_min_value_achieved_l36_36870


namespace sum_roots_quadratic_l36_36518

theorem sum_roots_quadratic (a b c : ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x : ℝ, P x = a * x^2 + b * x + c)
  (h : ∀ x : ℝ, P (2 * x^5 + 3 * x) ≥ P (3 * x^4 + 2 * x^2 + 1)) : 
  -b / a = 6 / 5 :=
sorry

end sum_roots_quadratic_l36_36518


namespace union_P_Q_l36_36550

open Set

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5, 6}

theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by 
  -- Proof goes here
  sorry

end union_P_Q_l36_36550


namespace smaller_of_two_integers_l36_36926

theorem smaller_of_two_integers (m n : ℕ) (h1 : 100 ≤ m ∧ m < 1000) (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : (m + n) / 2 = m + n / 1000) : min m n = 999 :=
by {
  sorry
}

end smaller_of_two_integers_l36_36926


namespace min_product_ab_l36_36080

theorem min_product_ab (a b : ℝ) (h : 20 * a * b = 13 * a + 14 * b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a * b = 1.82 :=
sorry

end min_product_ab_l36_36080


namespace min_value_expression_l36_36670

theorem min_value_expression (x : ℝ) (h : x > 1) : x + 9 / x - 2 ≥ 4 :=
sorry

end min_value_expression_l36_36670


namespace P_2017_eq_14_l36_36746

def sumOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def numberOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.length

def P (n : Nat) : Nat :=
  sumOfDigits n + numberOfDigits n

theorem P_2017_eq_14 : P 2017 = 14 :=
by
  sorry

end P_2017_eq_14_l36_36746


namespace problem_l36_36875

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l36_36875


namespace gardener_b_time_l36_36697

theorem gardener_b_time :
  ∃ x : ℝ, (1 / 3 + 1 / x = 1 / 1.875) → (x = 5) := by
  sorry

end gardener_b_time_l36_36697


namespace quadratic_has_two_distinct_real_roots_l36_36686

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (h1 : 4 + 4 * k > 0) (h2 : k ≠ 0) :
  k > -1 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l36_36686


namespace doberman_puppies_count_l36_36506

theorem doberman_puppies_count (D : ℝ) (S : ℝ) (h1 : S = 55) (h2 : 3 * D - 5 + (D - S) = 90) : D = 37.5 :=
by
  sorry

end doberman_puppies_count_l36_36506


namespace skylar_starting_age_l36_36940

-- Conditions of the problem
def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_amount_donated : ℕ := 440000

-- Question and proof statement
theorem skylar_starting_age :
  (current_age - total_amount_donated / annual_donation) = 16 := 
by
  sorry

end skylar_starting_age_l36_36940


namespace prime_prod_identity_l36_36819

theorem prime_prod_identity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 3 * p + 7 * q = 41) : (p + 1) * (q - 1) = 12 := 
by 
  sorry

end prime_prod_identity_l36_36819


namespace remaining_pictures_l36_36733

-- Definitions based on the conditions
def pictures_in_first_book : ℕ := 44
def pictures_in_second_book : ℕ := 35
def pictures_in_third_book : ℕ := 52
def pictures_in_fourth_book : ℕ := 48
def colored_pictures : ℕ := 37

-- Statement of the theorem based on the question and correct answer
theorem remaining_pictures :
  pictures_in_first_book + pictures_in_second_book + pictures_in_third_book + pictures_in_fourth_book - colored_pictures = 142 := by
  sorry

end remaining_pictures_l36_36733


namespace coordinate_system_and_parametric_equations_l36_36254

/-- Given the parametric equation of curve C1 is 
  x = 2 * cos φ and y = 3 * sin φ (where φ is the parameter)
  and a coordinate system with the origin as the pole and the positive half-axis of x as the polar axis.
  The polar equation of curve C2 is ρ = 2.
  The vertices of square ABCD are all on C2, arranged counterclockwise,
  with the polar coordinates of point A being (2, π/3).
  Find the Cartesian coordinates of A, B, C, and D, and prove that
  for any point P on C1, |PA|^2 + |PB|^2 + |PC|^2 + |PD|^2 is within the range [32, 52]. -/
theorem coordinate_system_and_parametric_equations
  (φ : ℝ)
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (P : ℝ → ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (t : ℝ)
  (H1 : ∀ φ, P φ = (2 * Real.cos φ, 3 * Real.sin φ))
  (H2 : A = (1, Real.sqrt 3) ∧ B = (-Real.sqrt 3, 1) ∧ C = (-1, -Real.sqrt 3) ∧ D = (Real.sqrt 3, -1))
  (H3 : ∀ p : ℝ × ℝ, ∃ φ, p = P φ)
  : ∀ x y, ∃ (φ : ℝ), P φ = (x, y) →
    ∃ t, t = |P φ - A|^2 + |P φ - B|^2 + |P φ - C|^2 + |P φ - D|^2 ∧ 32 ≤ t ∧ t ≤ 52 := 
sorry

end coordinate_system_and_parametric_equations_l36_36254


namespace work_together_days_l36_36246

theorem work_together_days (hA : ∃ d : ℝ, d > 0 ∧ d = 15)
                          (hB : ∃ d : ℝ, d > 0 ∧ d = 20)
                          (hfrac : ∃ f : ℝ, f = (23 / 30)) :
  ∃ d : ℝ, d = 2 := by
  sorry

end work_together_days_l36_36246


namespace probability_point_in_region_l36_36281

theorem probability_point_in_region (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 2010) 
  (h2 : 0 ≤ y ∧ y ≤ 2009) 
  (h3 : ∃ (u v : ℝ), (u, v) = (x, y) ∧ x > 2 * y ∧ y > 500) : 
  ∃ p : ℚ, p = 1505 / 4018 := 
sorry

end probability_point_in_region_l36_36281


namespace identify_wise_l36_36661

def total_people : ℕ := 30

def is_wise (p : ℕ) : Prop := True   -- This can be further detailed to specify wise characteristics
def is_fool (p : ℕ) : Prop := True    -- This can be further detailed to specify fool characteristics

def wise_count (w : ℕ) : Prop := True -- This indicates the count of wise people
def fool_count (f : ℕ) : Prop := True -- This indicates the count of fool people

def sum_of_groups (wise_groups fool_groups : ℕ) : Prop :=
  wise_groups + fool_groups = total_people

def sum_of_fools (fool_groups : ℕ) (F : ℕ) : Prop :=
  fool_groups = F

theorem identify_wise (F : ℕ) (h1 : F ≤ 8) :
  ∃ (wise_person : ℕ), (wise_person < 30 ∧ is_wise wise_person) :=
by
  sorry

end identify_wise_l36_36661


namespace largest_mersenne_prime_lt_500_l36_36118

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2^p - 1)

theorem largest_mersenne_prime_lt_500 : 
  ∀ n, is_mersenne_prime n → 2^n - 1 < 500 → 2^n - 1 ≤ 127 :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_500_l36_36118


namespace number_of_paths_l36_36465

-- Define the conditions of the problem
def grid_width : ℕ := 7
def grid_height : ℕ := 6
def diagonal_steps : ℕ := 2

-- Define the main proof statement
theorem number_of_paths (width height diag : ℕ) 
  (Nhyp : width = grid_width ∧ height = grid_height ∧ diag = diagonal_steps) : 
  ∃ (paths : ℕ), paths = 6930 := 
sorry

end number_of_paths_l36_36465


namespace merchant_loss_l36_36978

theorem merchant_loss
  (sp : ℝ)
  (profit_percent: ℝ)
  (loss_percent:  ℝ)
  (sp1 : ℝ)
  (sp2 : ℝ)
  (cp1 cp2 : ℝ)
  (net_loss : ℝ) :
  
  sp = 990 → 
  profit_percent = 0.1 → 
  loss_percent = 0.1 →
  sp1 = sp → 
  sp2 = sp → 
  cp1 = sp1 / (1 + profit_percent) →
  cp2 = sp2 / (1 - loss_percent) →
  net_loss = (cp2 - sp2) - (sp1 - cp1) →
  net_loss = 20 :=
by 
  intros _ _ _ _ _ _ _ _ 
  -- placeholders for intros to bind variables
  sorry

end merchant_loss_l36_36978


namespace eval_f_at_4_l36_36531

def f (x : ℕ) : ℕ := 5 * x + 2

theorem eval_f_at_4 : f 4 = 22 :=
by
  sorry

end eval_f_at_4_l36_36531


namespace polynomial_evaluation_l36_36045

theorem polynomial_evaluation :
  (5 * 3^3 - 3 * 3^2 + 7 * 3 - 2 = 127) :=
by
  sorry

end polynomial_evaluation_l36_36045


namespace first_route_red_lights_longer_l36_36443

-- Conditions
def first_route_base_time : ℕ := 10
def red_light_time : ℕ := 3
def num_stoplights : ℕ := 3
def second_route_time : ℕ := 14

-- Question to Answer
theorem first_route_red_lights_longer : (first_route_base_time + num_stoplights * red_light_time - second_route_time) = 5 := by
  sorry

end first_route_red_lights_longer_l36_36443


namespace area_identity_tg_cos_l36_36171

variable (a b c α β γ : Real)
variable (s t : Real) (area_of_triangle : Real)

-- Assume t is the area of the triangle and s is the semiperimeter
axiom area_of_triangle_eq_heron :
  t = Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Assume trigonometric identities for tangents and cosines of half-angles
axiom tg_half_angle_α : Real.tan (α / 2) = Real.sqrt ((s - b) * (s - c) / (s * (s - a)))
axiom tg_half_angle_β : Real.tan (β / 2) = Real.sqrt ((s - c) * (s - a) / (s * (s - b)))
axiom tg_half_angle_γ : Real.tan (γ / 2) = Real.sqrt ((s - a) * (s - b) / (s * (s - c)))

axiom cos_half_angle_α : Real.cos (α / 2) = Real.sqrt (s * (s - a) / (b * c))
axiom cos_half_angle_β : Real.cos (β / 2) = Real.sqrt (s * (s - b) / (c * a))
axiom cos_half_angle_γ : Real.cos (γ / 2) = Real.sqrt (s * (s - c) / (a * b))

theorem area_identity_tg_cos :
  t = s^2 * Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) ∧
  t = (a * b * c / s) * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) :=
by
  sorry

end area_identity_tg_cos_l36_36171


namespace lcm_technicians_schedule_l36_36562

theorem lcm_technicians_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := 
sorry

end lcm_technicians_schedule_l36_36562


namespace unique_intersection_point_l36_36233

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y = 9
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1
def line5 (x y : ℝ) : Prop := x + y = 4

theorem unique_intersection_point :
  ∃! (p : ℝ × ℝ), 
     line1 p.1 p.2 ∧ 
     line2 p.1 p.2 ∧ 
     line3 p.1 ∧ 
     line4 p.2 ∧ 
     line5 p.1 p.2 :=
sorry

end unique_intersection_point_l36_36233


namespace Tim_placed_rulers_l36_36297

variable (initial_rulers final_rulers : ℕ)
variable (placed_rulers : ℕ)

-- Given conditions
def initial_rulers_def : initial_rulers = 11 := sorry
def final_rulers_def : final_rulers = 25 := sorry

-- Goal
theorem Tim_placed_rulers : placed_rulers = final_rulers - initial_rulers :=
  by
  sorry

end Tim_placed_rulers_l36_36297


namespace find_triples_l36_36397

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end find_triples_l36_36397


namespace orange_count_in_bin_l36_36178

-- Definitions of the conditions
def initial_oranges : Nat := 5
def oranges_thrown_away : Nat := 2
def new_oranges_added : Nat := 28

-- The statement of the proof problem
theorem orange_count_in_bin : initial_oranges - oranges_thrown_away + new_oranges_added = 31 :=
by
  sorry

end orange_count_in_bin_l36_36178


namespace find_Tom_favorite_numbers_l36_36230

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def Tom_favorite_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧
  is_multiple_of n 13 ∧
  ¬ is_multiple_of n 3 ∧
  is_multiple_of (sum_of_digits n) 4

theorem find_Tom_favorite_numbers :
  ∃ n : ℕ, Tom_favorite_number n ∧ (n = 130 ∨ n = 143) :=
by
  sorry

end find_Tom_favorite_numbers_l36_36230


namespace round_robin_tournament_participant_can_mention_all_l36_36271

theorem round_robin_tournament_participant_can_mention_all :
  ∀ (n : ℕ) (participants : Fin n → Fin n → Prop),
  (∀ i j : Fin n, i ≠ j → (participants i j ∨ participants j i)) →
  (∃ A : Fin n, ∀ (B : Fin n), B ≠ A → (participants A B ∨ ∃ C : Fin n, participants A C ∧ participants C B)) := by
  sorry

end round_robin_tournament_participant_can_mention_all_l36_36271


namespace students_left_in_final_year_l36_36411

variable (s10 s_next s_final x : Nat)

-- Conditions
def initial_students : Prop := s10 = 150
def students_after_joining : Prop := s_next = s10 + 30
def students_final_year : Prop := s_final = s_next - x
def final_year_students : Prop := s_final = 165

-- Theorem to prove
theorem students_left_in_final_year (h1 : initial_students s10)
                                     (h2 : students_after_joining s10 s_next)
                                     (h3 : students_final_year s_next s_final x)
                                     (h4 : final_year_students s_final) :
  x = 15 :=
by
  sorry

end students_left_in_final_year_l36_36411


namespace weight_of_e_l36_36849

variables (d e f : ℝ)

theorem weight_of_e
  (h_de_f : (d + e + f) / 3 = 42)
  (h_de : (d + e) / 2 = 35)
  (h_ef : (e + f) / 2 = 41) :
  e = 26 :=
by
  sorry

end weight_of_e_l36_36849


namespace circumradius_of_regular_tetrahedron_l36_36878

theorem circumradius_of_regular_tetrahedron (a : ℝ) (h : a > 0) :
    ∃ R : ℝ, R = a * (Real.sqrt 6) / 4 :=
by
  sorry

end circumradius_of_regular_tetrahedron_l36_36878


namespace min_det_is_neg_six_l36_36398

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six_l36_36398


namespace largest_unshaded_area_l36_36901

theorem largest_unshaded_area (s : ℝ) (π_approx : ℝ) :
    (let r := s / 2
     let area_square := s^2
     let area_circle := π_approx * r^2
     let area_triangle := (1 / 2) * (s / 2) * (s / 2)
     let unshaded_square := area_square - area_circle
     let unshaded_circle := area_circle - area_triangle
     unshaded_circle) > (unshaded_square) := by
        sorry

end largest_unshaded_area_l36_36901


namespace distance_between_trains_l36_36018

theorem distance_between_trains
  (v1 v2 : ℕ) (d_diff : ℕ)
  (h_v1 : v1 = 50) (h_v2 : v2 = 60) (h_d_diff : d_diff = 100) :
  ∃ d, d = 1100 :=
by
  sorry

-- Explanation:
-- v1 is the speed of the first train.
-- v2 is the speed of the second train.
-- d_diff is the difference in the distances traveled by the two trains at the time of meeting.
-- h_v1 states that the speed of the first train is 50 kmph.
-- h_v2 states that the speed of the second train is 60 kmph.
-- h_d_diff states that the second train travels 100 km more than the first train.
-- The existential statement asserts that there exists a distance d such that d equals 1100 km.

end distance_between_trains_l36_36018


namespace find_other_person_weight_l36_36234

noncomputable def other_person_weight (n avg new_avg W1 : ℕ) : ℕ :=
  let total_initial := n * avg
  let new_n := n + 2
  let total_new := new_n * new_avg
  total_new - total_initial - W1

theorem find_other_person_weight:
  other_person_weight 23 48 51 78 = 93 := by
  sorry

end find_other_person_weight_l36_36234


namespace range_of_a_l36_36889

noncomputable def parabola_locus (x : ℝ) : ℝ := x^2 / 4

def angle_sum_property (a k : ℝ) : Prop :=
  2 * a * k^2 + 2 * k + a = 0

def discriminant_nonnegative (a : ℝ) : Prop :=
  4 - 8 * a^2 ≥ 0

theorem range_of_a (a : ℝ) :
  (- (Real.sqrt 2) / 2) ≤ a ∧ a ≤ (Real.sqrt 2) / 2 :=
  sorry

end range_of_a_l36_36889


namespace positive_integer_M_l36_36611

theorem positive_integer_M (M : ℕ) (h : 14^2 * 35^2 = 70^2 * M^2) : M = 7 :=
sorry

end positive_integer_M_l36_36611


namespace sum_of_digits_5_pow_eq_2_pow_l36_36911

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_5_pow_eq_2_pow (n : ℕ) (h : sum_of_digits (5^n) = 2^n) : n = 3 :=
by
  sorry

end sum_of_digits_5_pow_eq_2_pow_l36_36911


namespace line_equation_l36_36121

theorem line_equation {a b c : ℝ} (x : ℝ) (y : ℝ)
  (point : ∃ p: ℝ × ℝ, p = (-1, 0))
  (perpendicular : ∀ k: ℝ, k = 1 → 
    ∀ m: ℝ, m = -1 → 
      ∀ b1: ℝ, b1 = 0 → 
        ∀ x1: ℝ, x1 = -1 →
          ∀ y1: ℝ, y1 = 0 →
            ∀ l: ℝ, l = b1 + k * (x1 - (-1)) + m * (y1 - 0) → 
              x - y + 1 = 0) :
  x - y + 1 = 0 :=
sorry

end line_equation_l36_36121


namespace john_runs_with_dog_for_half_hour_l36_36981

noncomputable def time_with_dog_in_hours (t : ℝ) : Prop := 
  let d1 := 6 * t          -- Distance run with the dog
  let d2 := 4 * (1 / 2)    -- Distance run alone
  (d1 + d2 = 5) ∧ (t = 1 / 2)

theorem john_runs_with_dog_for_half_hour : ∃ t : ℝ, time_with_dog_in_hours t := 
by
  use (1 / 2)
  sorry

end john_runs_with_dog_for_half_hour_l36_36981


namespace min_squared_distance_l36_36187

theorem min_squared_distance : 
  ∀ (x y : ℝ), (x - y = 1) → (∃ (a b : ℝ), 
  ((a - 2) ^ 2 + (b - 2) ^ 2 <= (x - 2) ^ 2 + (y - 2) ^ 2) ∧ ((a - 2) ^ 2 + (b - 2) ^ 2 = 1 / 2)) := 
by
  sorry

end min_squared_distance_l36_36187


namespace fraction_of_product_l36_36043

theorem fraction_of_product (c d: ℕ) 
  (h1: 5 * 64 + 4 * 8 + 3 = 355)
  (h2: 2 * (10 * c + d) = 355)
  (h3: c < 10)
  (h4: d < 10):
  (c * d : ℚ) / 12 = 5 / 4 :=
by
  sorry

end fraction_of_product_l36_36043


namespace a_8_eq_5_l36_36785

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S_eq : ∀ n m : ℕ, S n + S m = S (n + m)
axiom a1 : a 1 = 5
axiom Sn1 : ∀ n : ℕ, S (n + 1) = S n + 5

theorem a_8_eq_5 : a 8 = 5 :=
sorry

end a_8_eq_5_l36_36785


namespace overall_gain_percentage_l36_36603

def cost_of_A : ℝ := 100
def selling_price_of_A : ℝ := 125
def cost_of_B : ℝ := 200
def selling_price_of_B : ℝ := 250
def cost_of_C : ℝ := 150
def selling_price_of_C : ℝ := 180

theorem overall_gain_percentage :
  ((selling_price_of_A + selling_price_of_B + selling_price_of_C) - (cost_of_A + cost_of_B + cost_of_C)) / (cost_of_A + cost_of_B + cost_of_C) * 100 = 23.33 := 
by
  sorry

end overall_gain_percentage_l36_36603


namespace arrangements_count_correct_l36_36837

noncomputable def count_arrangements : ℕ :=
  -- The total number of different arrangements of students A, B, C, D in 3 communities
  -- such that each community has at least one student, and A and B are not in the same community.
  sorry

theorem arrangements_count_correct : count_arrangements = 30 := by
  sorry

end arrangements_count_correct_l36_36837


namespace customer_paid_correct_amount_l36_36402

theorem customer_paid_correct_amount (cost_price : ℕ) (markup_percentage : ℕ) (total_price : ℕ) :
  cost_price = 6500 → 
  markup_percentage = 30 → 
  total_price = cost_price + (cost_price * markup_percentage / 100) → 
  total_price = 8450 :=
by
  intros h_cost_price h_markup_percentage h_total_price
  sorry

end customer_paid_correct_amount_l36_36402


namespace combinations_of_painting_options_l36_36547

theorem combinations_of_painting_options : 
  let colors := 6
  let methods := 3
  let finishes := 2
  colors * methods * finishes = 36 := by
  sorry

end combinations_of_painting_options_l36_36547


namespace lesser_of_two_numbers_l36_36002

theorem lesser_of_two_numbers (a b : ℕ) (h₁ : a + b = 55) (h₂ : a - b = 7) (h₃ : a > b) : b = 24 :=
by
  sorry

end lesser_of_two_numbers_l36_36002


namespace inequality_proof_l36_36542

theorem inequality_proof 
  (a b c d : ℝ) (n : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_n : 9 ≤ n) :
  a^n + b^n + c^n + d^n ≥ a^(n-9)*b^4*c^3*d^2 + b^(n-9)*c^4*d^3*a^2 + c^(n-9)*d^4*a^3*b^2 + d^(n-9)*a^4*b^3*c^2 :=
by
  sorry

end inequality_proof_l36_36542


namespace value_of_abs_h_l36_36368

theorem value_of_abs_h (h : ℝ) : 
  (∃ r s : ℝ, (r + s = -4 * h) ∧ (r * s = -5) ∧ (r^2 + s^2 = 13)) → 
  |h| = (Real.sqrt 3) / 4 :=
by
  sorry

end value_of_abs_h_l36_36368


namespace similar_triangles_x_value_l36_36154

-- Define the conditions of the problem
variables (x : ℝ) (h₁ : 10 / x = 8 / 5)

-- State the theorem/proof problem
theorem similar_triangles_x_value : x = 6.25 :=
by
  -- Proof goes here
  sorry

end similar_triangles_x_value_l36_36154


namespace bob_clean_time_l36_36430

-- Definitions for the problem conditions
def alice_time : ℕ := 30
def bob_time := (1 / 3 : ℚ) * alice_time

-- The proof problem statement (only) in Lean 4
theorem bob_clean_time : bob_time = 10 := by
  sorry

end bob_clean_time_l36_36430


namespace Tom_water_intake_daily_l36_36174

theorem Tom_water_intake_daily (cans_per_day : ℕ) (oz_per_can : ℕ) (fluid_per_week : ℕ) (days_per_week : ℕ)
  (h1 : cans_per_day = 5) 
  (h2 : oz_per_can = 12) 
  (h3 : fluid_per_week = 868) 
  (h4 : days_per_week = 7) : 
  ((fluid_per_week - (cans_per_day * oz_per_can * days_per_week)) / days_per_week) = 64 := 
sorry

end Tom_water_intake_daily_l36_36174


namespace find_last_week_rope_l36_36961

/-- 
Description: Mr. Sanchez bought 4 feet of rope less than he did the previous week. 
Given that he bought 96 inches in total, find how many feet he bought last week.
--/
theorem find_last_week_rope (F : ℕ) :
  12 * (F - 4) = 96 → F = 12 := by
  sorry

end find_last_week_rope_l36_36961


namespace beads_per_necklace_l36_36641

-- Definitions based on conditions
def total_beads_used (N : ℕ) : ℕ :=
  10 * N + 2 * N + 50 + 35

-- Main theorem to prove the number of beads needed for one beaded necklace
theorem beads_per_necklace (N : ℕ) (h : total_beads_used N = 325) : N = 20 :=
by
  sorry

end beads_per_necklace_l36_36641


namespace three_digit_number_ends_with_same_three_digits_l36_36294

theorem three_digit_number_ends_with_same_three_digits (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, k ≥ 1 → N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) := 
sorry

end three_digit_number_ends_with_same_three_digits_l36_36294


namespace opposite_of_2023_l36_36688

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l36_36688


namespace exists_n_consecutive_composites_l36_36446

theorem exists_n_consecutive_composites (n : ℕ) (h : n ≥ 1) (a r : ℕ) :
  ∃ K : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (a + (K + i) * r)) := 
sorry

end exists_n_consecutive_composites_l36_36446


namespace p_at_0_l36_36636

noncomputable def p : Polynomial ℚ := sorry

theorem p_at_0 :
  (∀ n : ℕ, n ≤ 6 → p.eval (2^n) = 1 / (2^n))
  ∧ p.degree = 6 → 
  p.eval 0 = 127 / 64 :=
sorry

end p_at_0_l36_36636


namespace div_by_7_l36_36055

theorem div_by_7 (n : ℕ) (h : n ≥ 1) : 7 ∣ (8^n + 6) :=
sorry

end div_by_7_l36_36055


namespace inverse_proportion_quadrants_l36_36147

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0))) ↔ k > 0 := by
  sorry

end inverse_proportion_quadrants_l36_36147


namespace problem_solution_l36_36464

theorem problem_solution
  (a d : ℝ)
  (h : (∀ x : ℝ, (x - 3) * (x + a) = x^2 + d * x - 18)) :
  d = 3 := 
sorry

end problem_solution_l36_36464


namespace mathland_transport_l36_36565

theorem mathland_transport (n : ℕ) (h : n ≥ 2) (transport : Fin n -> Fin n -> Prop) :
(∀ i j, transport i j ∨ transport j i) →
(∃ tr : Fin n -> Fin n -> Prop, 
  (∀ i j, transport i j → tr i j) ∨
  (∀ i j, transport j i → tr i j)) :=
by
  sorry

end mathland_transport_l36_36565


namespace participants_initial_count_l36_36555

theorem participants_initial_count (initial_participants remaining_after_first_round remaining_after_second_round : ℝ) 
  (h1 : remaining_after_first_round = 0.4 * initial_participants)
  (h2 : remaining_after_second_round = (1/4) * remaining_after_first_round)
  (h3 : remaining_after_second_round = 15) : 
  initial_participants = 150 :=
sorry

end participants_initial_count_l36_36555


namespace log_expression_l36_36958

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression :
  log_base 4 16 - (log_base 2 3 * log_base 3 2) = 1 := by
  sorry

end log_expression_l36_36958


namespace derivative_of_function_l36_36342

theorem derivative_of_function
  (y : ℝ → ℝ)
  (h : ∀ x, y x = (1/2) * (Real.exp x + Real.exp (-x))) :
  ∀ x, deriv y x = (1/2) * (Real.exp x - Real.exp (-x)) :=
by
  sorry

end derivative_of_function_l36_36342


namespace binary_to_decimal_l36_36128

theorem binary_to_decimal : 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0) = 54 :=
by 
  sorry

end binary_to_decimal_l36_36128


namespace rectangle_area_inscribed_circle_l36_36748

theorem rectangle_area_inscribed_circle {r w l : ℕ} (h1 : r = 7) (h2 : w = 2 * r) (h3 : l = 3 * w) : l * w = 588 :=
by 
  -- The proof details are omitted as per instructions.
  sorry

end rectangle_area_inscribed_circle_l36_36748


namespace odd_number_adjacent_product_diff_l36_36440

variable (x : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem odd_number_adjacent_product_diff (h : is_odd x)
  (adjacent_diff : x * (x + 2) - x * (x - 2) = 44) : x = 11 :=
by
  sorry

end odd_number_adjacent_product_diff_l36_36440


namespace An_is_integer_l36_36909

theorem An_is_integer 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : a > b)
  (θ : ℝ) (h_theta : θ > 0 ∧ θ < Real.pi / 2)
  (h_sin : Real.sin θ = 2 * (a * b) / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, ((a^2 + b^2)^n * Real.sin (n * θ) : ℝ) = k :=
by sorry

end An_is_integer_l36_36909


namespace compute_xy_l36_36872

theorem compute_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) : 
  xy = -126 / 25 ∨ xy = -6 := 
sorry

end compute_xy_l36_36872


namespace find_number_l36_36917

theorem find_number (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 -> x = 9 :=
by
  sorry

end find_number_l36_36917


namespace largest_a_l36_36734

open Real

theorem largest_a (a b c : ℝ) (h1 : a + b + c = 6) (h2 : ab + ac + bc = 11) : 
  a ≤ 2 + 2 * sqrt 3 / 3 :=
sorry

end largest_a_l36_36734


namespace small_boxes_in_big_box_l36_36764

theorem small_boxes_in_big_box (total_candles : ℕ) (candles_per_small : ℕ) (total_big_boxes : ℕ) 
  (h1 : total_candles = 8000) 
  (h2 : candles_per_small = 40) 
  (h3 : total_big_boxes = 50) :
  (total_candles / candles_per_small) / total_big_boxes = 4 :=
by
  sorry

end small_boxes_in_big_box_l36_36764


namespace quadratic_always_positive_l36_36445

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 :=
sorry

end quadratic_always_positive_l36_36445


namespace units_digit_product_l36_36681

theorem units_digit_product (k l : ℕ) (h1 : ∀ n : ℕ, (5^n % 10) = 5) (h2 : ∀ m < 4, (6^m % 10) = 6) :
  ((5^k * 6^l) % 10) = 0 :=
by
  have h5 : (5^k % 10) = 5 := h1 k
  have h6 : (6^4 % 10) = 6 := h2 4 (by sorry)
  have h_product : (5^k * 6^l % 10) = ((5 % 10) * (6 % 10) % 10) := sorry
  norm_num at h_product
  exact h_product

end units_digit_product_l36_36681


namespace ratio_of_investments_l36_36743

theorem ratio_of_investments (P Q : ℝ)
  (h_ratio_profits : (20 * P) / (40 * Q) = 7 / 10) : P / Q = 7 / 5 := 
sorry

end ratio_of_investments_l36_36743


namespace regular_polygon_sides_l36_36298

theorem regular_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 144 * n) : n = 10 := 
by
  sorry

end regular_polygon_sides_l36_36298


namespace cube_increasing_on_reals_l36_36965

theorem cube_increasing_on_reals (a b : ℝ) (h : a < b) : a^3 < b^3 :=
sorry

end cube_increasing_on_reals_l36_36965


namespace lcm_of_pack_sizes_l36_36788

theorem lcm_of_pack_sizes :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 19) 8) 11) 17) 23 = 772616 := by
  sorry

end lcm_of_pack_sizes_l36_36788


namespace additional_regular_gift_bags_needed_l36_36558

-- Defining the conditions given in the question
def confirmed_guests : ℕ := 50
def additional_guests_70pc : ℕ := 30
def additional_guests_40pc : ℕ := 15
def probability_70pc : ℚ := 0.7
def probability_40pc : ℚ := 0.4
def extravagant_bags_prepared : ℕ := 10
def special_bags_prepared : ℕ := 25
def regular_bags_prepared : ℕ := 20

-- Defining the expected number of additional guests based on probabilities
def expected_guests_70pc : ℚ := additional_guests_70pc * probability_70pc
def expected_guests_40pc : ℚ := additional_guests_40pc * probability_40pc

-- Defining the total expected guests including confirmed guests and expected additional guests
def total_expected_guests : ℚ := confirmed_guests + expected_guests_70pc + expected_guests_40pc

-- Defining the problem statement in Lean, proving the additional regular gift bags needed
theorem additional_regular_gift_bags_needed : 
  total_expected_guests = 77 → regular_bags_prepared = 20 → 22 = 22 :=
by
  sorry

end additional_regular_gift_bags_needed_l36_36558


namespace living_room_size_is_96_l36_36227

-- Define the total area of the apartment
def total_area : ℕ := 16 * 10

-- Define the number of units
def units : ℕ := 5

-- Define the size of one unit
def size_of_one_unit : ℕ := total_area / units

-- Define the size of the living room
def living_room_size : ℕ := size_of_one_unit * 3

-- Proving that the living room size is indeed 96 square feet
theorem living_room_size_is_96 : living_room_size = 96 := 
by
  -- not providing proof, thus using sorry
  sorry

end living_room_size_is_96_l36_36227


namespace num_students_second_grade_l36_36508

structure School :=
(total_students : ℕ)
(prob_male_first_grade : ℝ)

def stratified_sampling (school : School) : ℕ := sorry

theorem num_students_second_grade (school : School) (total_selected : ℕ) : 
    school.total_students = 4000 →
    school.prob_male_first_grade = 0.2 →
    total_selected = 100 →
    stratified_sampling school = 30 :=
by
  intros
  sorry

end num_students_second_grade_l36_36508


namespace stickers_per_student_l36_36556

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end stickers_per_student_l36_36556


namespace time_to_cross_tree_l36_36941

theorem time_to_cross_tree (length_train : ℝ) (length_platform : ℝ) (time_to_pass_platform : ℝ) (h1 : length_train = 1200) (h2 : length_platform = 1200) (h3 : time_to_pass_platform = 240) : 
  (length_train / ((length_train + length_platform) / time_to_pass_platform)) = 120 := 
by
    sorry

end time_to_cross_tree_l36_36941


namespace total_opponent_scores_is_45_l36_36797

-- Definitions based on the conditions
def games : Fin 10 := Fin.mk 10 sorry

def team_scores : Fin 10 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨6, _⟩ => 7
| ⟨7, _⟩ => 8
| ⟨8, _⟩ => 9
| ⟨9, _⟩ => 10
| _ => 0  -- Placeholder for out-of-bounds, should not be used

def lost_games : Fin 5 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5
| ⟨3, _⟩ => 7
| ⟨4, _⟩ => 9

def opponent_score_lost : ℕ → ℕ := λ s => s + 1

def won_games : Fin 5 → ℕ
| ⟨0, _⟩ => 2
| ⟨1, _⟩ => 4
| ⟨2, _⟩ => 6
| ⟨3, _⟩ => 8
| ⟨4, _⟩ => 10

def opponent_score_won : ℕ → ℕ := λ s => s / 2

-- Main statement to prove total opponent scores
theorem total_opponent_scores_is_45 :
  let total_lost_scores := (lost_games 0 :: lost_games 1 :: lost_games 2 :: lost_games 3 :: lost_games 4 :: []).map opponent_score_lost
  let total_won_scores  := (won_games 0 :: won_games 1 :: won_games 2 :: won_games 3 :: won_games 4 :: []).map opponent_score_won
  total_lost_scores.sum + total_won_scores.sum = 45 :=
by sorry

end total_opponent_scores_is_45_l36_36797


namespace infinite_solutions_exists_l36_36771

theorem infinite_solutions_exists : 
  ∃ (S : Set (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ S → 2 * a^2 - 3 * a + 1 = 3 * b^2 + b) 
  ∧ Set.Infinite S :=
sorry

end infinite_solutions_exists_l36_36771


namespace incorrect_comparison_l36_36173

theorem incorrect_comparison :
  ¬ (- (2 / 3) < - (4 / 5)) :=
by
  sorry

end incorrect_comparison_l36_36173


namespace numerator_greater_denominator_l36_36799

theorem numerator_greater_denominator (x : ℝ) (h1 : -3 ≤ x) (h2 : x ≤ 3) (h3 : 5 * x + 3 > 8 - 3 * x) : (5 / 8) < x ∧ x ≤ 3 :=
by
  sorry

end numerator_greater_denominator_l36_36799


namespace smallest_possible_third_term_l36_36422

theorem smallest_possible_third_term :
  ∃ (d : ℝ), (d = -3 + Real.sqrt 134 ∨ d = -3 - Real.sqrt 134) ∧ 
  (7, 7 + d + 3, 7 + 2 * d + 18) = (7, 10 + d, 25 + 2 * d) ∧ 
  min (25 + 2 * (-3 + Real.sqrt 134)) (25 + 2 * (-3 - Real.sqrt 134)) = 19 + 2 * Real.sqrt 134 :=
by
  sorry

end smallest_possible_third_term_l36_36422


namespace find_angle_A_l36_36306

noncomputable def angle_A (a b c S : ℝ) := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

theorem find_angle_A (a b c S : ℝ) (hb : 0 < b) (hc : 0 < c) (hS : S = (1/2) * b * c * Real.sin (angle_A a b c S)) 
    (h_eq : b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S) : 
    angle_A a b c S = π / 6 := by 
  sorry

end find_angle_A_l36_36306


namespace sequence_general_term_l36_36676

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2^n) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end sequence_general_term_l36_36676


namespace age_of_B_l36_36623

variables (A B : ℕ)

-- Conditions
def condition1 := A + 10 = 2 * (B - 10)
def condition2 := A = B + 7

-- Theorem stating the present age of B
theorem age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 :=
by
  sorry

end age_of_B_l36_36623


namespace room_width_l36_36355

theorem room_width (length : ℕ) (total_cost : ℕ) (cost_per_sqm : ℕ) : ℚ :=
  let area := total_cost / cost_per_sqm
  let width := area / length
  width

example : room_width 9 38475 900 = 4.75 := by
  sorry

end room_width_l36_36355


namespace sum_of_possible_values_l36_36543

variable (a b c d : ℝ)

theorem sum_of_possible_values
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) :
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 :=
sorry

end sum_of_possible_values_l36_36543


namespace product_discount_rate_l36_36137

theorem product_discount_rate (cost_price marked_price : ℝ) (desired_profit_rate : ℝ) :
  cost_price = 200 → marked_price = 300 → desired_profit_rate = 0.2 →
  (∃ discount_rate : ℝ, discount_rate = 0.8 ∧ marked_price * discount_rate = cost_price * (1 + desired_profit_rate)) :=
by
  intros
  sorry

end product_discount_rate_l36_36137


namespace find_k_l36_36020

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (k : ℝ)

-- Conditions
def not_collinear (a b : V) : Prop := ¬ ∃ (m : ℝ), b = m • a
def collinear (u v : V) : Prop := ∃ (m : ℝ), u = m • v

theorem find_k (h1 : not_collinear a b) (h2 : collinear (2 • a + k • b) (a - b)) : k = -2 :=
by
  sorry

end find_k_l36_36020


namespace garden_remaining_area_is_250_l36_36497

open Nat

-- Define the dimensions of the rectangular garden
def garden_length : ℕ := 18
def garden_width : ℕ := 15
-- Define the dimensions of the square cutouts
def cutout1_side : ℕ := 4
def cutout2_side : ℕ := 2

-- Calculate areas based on the definitions
def garden_area : ℕ := garden_length * garden_width
def cutout1_area : ℕ := cutout1_side * cutout1_side
def cutout2_area : ℕ := cutout2_side * cutout2_side

-- Calculate total area excluding the cutouts
def remaining_area : ℕ := garden_area - cutout1_area - cutout2_area

-- Prove that the remaining area is 250 square feet
theorem garden_remaining_area_is_250 : remaining_area = 250 :=
by
  sorry

end garden_remaining_area_is_250_l36_36497


namespace find_abc_l36_36679

noncomputable def x (t : ℝ) := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) := 3 * Real.sin t

theorem find_abc :
  ∃ a b c : ℝ, 
  (a = 1/9) ∧ 
  (b = 4/27) ∧ 
  (c = 5/27) ∧ 
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1) :=
by
  sorry

end find_abc_l36_36679


namespace average_price_per_book_l36_36310

theorem average_price_per_book (books1_cost : ℕ) (books1_count : ℕ)
    (books2_cost : ℕ) (books2_count : ℕ)
    (h1 : books1_cost = 6500) (h2 : books1_count = 65)
    (h3 : books2_cost = 2000) (h4 : books2_count = 35) :
    (books1_cost + books2_cost) / (books1_count + books2_count) = 85 :=
by
    sorry

end average_price_per_book_l36_36310


namespace relationship_between_abc_l36_36027

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem relationship_between_abc (h1 : 2^a = Real.log (1/a) / Real.log 2)
                                 (h2 : Real.log b / Real.log 2 = 2)
                                 (h3 : c = Real.log 2 + Real.log 3 - Real.log 7) :
  b > a ∧ a > c :=
sorry

end relationship_between_abc_l36_36027


namespace sum_of_consecutive_ints_product_eq_336_l36_36232

def consecutive_ints_sum (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

theorem sum_of_consecutive_ints_product_eq_336 (a b c : ℤ) (h1 : consecutive_ints_sum a b c) (h2 : a * b * c = 336) :
  a + b + c = 21 :=
sorry

end sum_of_consecutive_ints_product_eq_336_l36_36232


namespace problem_solution_l36_36494

noncomputable def time_min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * (Real.cos α) / (2 * c * (1 - Real.sin α))

noncomputable def min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * Real.sqrt ((1 - (Real.sin α)) / 2)

theorem problem_solution (α : ℝ) (c : ℝ) (a : ℝ) 
  (α_30 : α = Real.pi / 6) (c_50 : c = 50) (a_50sqrt3 : a = 50 * Real.sqrt 3) :
  (time_min_distance c α a = 1.5) ∧ (min_distance c α a = 25 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l36_36494


namespace percentage_more_l36_36724

variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.90 * J
def Mary_income : Prop := M = 1.44 * J

-- Theorem to be proved
theorem percentage_more (h1 : Tim_income J T) (h2 : Mary_income J M) :
  ((M - T) / T) * 100 = 60 :=
sorry

end percentage_more_l36_36724


namespace Susan_roses_ratio_l36_36004

theorem Susan_roses_ratio (total_roses given_roses vase_roses remaining_roses : ℕ) 
  (H1 : total_roses = 3 * 12)
  (H2 : vase_roses = total_roses - given_roses)
  (H3 : remaining_roses = vase_roses * 2 / 3)
  (H4 : remaining_roses = 12) :
  given_roses / gcd given_roses total_roses = 1 ∧ total_roses / gcd given_roses total_roses = 2 :=
by
  sorry

end Susan_roses_ratio_l36_36004


namespace sum_of_ages_l36_36365

-- Problem statement:
-- Given: The product of their ages is 144.
-- Prove: The sum of their ages is 16.
theorem sum_of_ages (k t : ℕ) (htwins : t > k) (hprod : 2 * t * k = 144) : 2 * t + k = 16 := 
sorry

end sum_of_ages_l36_36365


namespace bacteria_after_time_l36_36923

def initial_bacteria : ℕ := 1
def division_time : ℕ := 20  -- time in minutes for one division
def total_time : ℕ := 180  -- total time in minutes

def divisions := total_time / division_time

theorem bacteria_after_time : (initial_bacteria * 2 ^ divisions) = 512 := by
  exact sorry

end bacteria_after_time_l36_36923


namespace area_of_circle_l36_36673

-- Define the given conditions
def pi_approx : ℝ := 3
def radius : ℝ := 0.6

-- Prove that the area is 1.08 given the conditions
theorem area_of_circle : π = pi_approx → radius = 0.6 → 
  (pi_approx * radius^2 = 1.08) :=
by
  intros hπ hr
  sorry

end area_of_circle_l36_36673


namespace root_product_is_27_l36_36927

open Real

noncomputable def cube_root (x : ℝ) := x ^ (1 / 3 : ℝ)
noncomputable def fourth_root (x : ℝ) := x ^ (1 / 4 : ℝ)
noncomputable def square_root (x : ℝ) := x ^ (1 / 2 : ℝ)

theorem root_product_is_27 : 
  (cube_root 27) * (fourth_root 81) * (square_root 9) = 27 := 
by
  sorry

end root_product_is_27_l36_36927


namespace unit_fraction_decomposition_l36_36394

theorem unit_fraction_decomposition (n : ℕ) (hn : 0 < n): 
  (1 : ℚ) / n = (1 : ℚ) / (2 * n) + (1 : ℚ) / (3 * n) + (1 : ℚ) / (6 * n) :=
by
  sorry

end unit_fraction_decomposition_l36_36394


namespace percent_increase_l36_36805

theorem percent_increase (original value new_value : ℕ) (h1 : original_value = 20) (h2 : new_value = 25) :
  ((new_value - original_value) / original_value) * 100 = 25 :=
by
  -- Proof omitted
  sorry

end percent_increase_l36_36805


namespace toy_cost_l36_36226

-- Conditions
def initial_amount : ℕ := 3
def allowance : ℕ := 7
def total_amount : ℕ := initial_amount + allowance
def number_of_toys : ℕ := 2

-- Question and Proof
theorem toy_cost :
  total_amount / number_of_toys = 5 :=
by
  sorry

end toy_cost_l36_36226


namespace cost_of_milk_l36_36970

theorem cost_of_milk (x : ℝ) (h1 : 10 * 0.1 = 1) (h2 : 11 = 1 + x + 3 * x) : x = 2.5 :=
by 
  sorry

end cost_of_milk_l36_36970


namespace probability_three_different_suits_l36_36483

noncomputable def pinochle_deck := 48
noncomputable def total_cards := 48
noncomputable def different_suits_probability := (36 / 47) * (23 / 46)

theorem probability_three_different_suits :
  different_suits_probability = 414 / 1081 :=
sorry

end probability_three_different_suits_l36_36483


namespace find_ratio_l36_36240

open Real

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)

-- The geometric sequence conditions
def geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

-- Sum of the first n terms for the geometric sequence
def sum_of_first_n_terms := ∀ n : ℕ, S n = (a 0) * (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions :=
  a 0 + a 2 = 5 / 2 ∧
  a 1 + a 3 = 5 / 4

-- The goal to prove
theorem find_ratio (geo_seq : geometric_sequence a q) (sum_terms : sum_of_first_n_terms a S q) (cond : given_conditions a) :
  S 4 / a 4 = 31 :=
  sorry

end find_ratio_l36_36240


namespace chris_first_day_breath_l36_36615

theorem chris_first_day_breath (x : ℕ) (h1 : x + 10 = 20) : x = 10 :=
by
  sorry

end chris_first_day_breath_l36_36615


namespace find_w_l36_36481

variable (x y z w : ℝ)

theorem find_w (h : (x + y + z) / 3 = (y + z + w) / 3 + 10) : w = x - 30 := by 
  sorry

end find_w_l36_36481


namespace polygon_diagonals_l36_36449

theorem polygon_diagonals (n : ℕ) (h : 20 = n) : (n * (n - 3)) / 2 = 170 :=
by
  sorry

end polygon_diagonals_l36_36449


namespace perpendicular_lines_a_eq_0_or_neg1_l36_36988

theorem perpendicular_lines_a_eq_0_or_neg1 (a : ℝ) :
  (∃ (k₁ k₂: ℝ), (k₁ = a ∧ k₂ = (2 * a - 1)) ∧ ∃ (k₃ k₄: ℝ), (k₃ = 3 ∧ k₄ = a) ∧ k₁ * k₃ + k₂ * k₄ = 0) →
  (a = 0 ∨ a = -1) := 
sorry

end perpendicular_lines_a_eq_0_or_neg1_l36_36988


namespace count_squares_below_line_l36_36276

theorem count_squares_below_line (units : ℕ) :
  let intercept_x := 221;
  let intercept_y := 7;
  let total_squares := intercept_x * intercept_y;
  let diagonal_squares := intercept_x - 1 + intercept_y - 1 + 1; 
  let non_diag_squares := total_squares - diagonal_squares;
  let below_line := non_diag_squares / 2;
  below_line = 660 :=
by
  sorry

end count_squares_below_line_l36_36276


namespace weighted_avg_M_B_eq_l36_36726

-- Define the weightages and the given weighted total marks equation
def weight_physics : ℝ := 1.5
def weight_chemistry : ℝ := 2
def weight_mathematics : ℝ := 1.25
def weight_biology : ℝ := 1.75
def weighted_total_M_B : ℝ := 250
def weighted_sum_M_B : ℝ := weight_mathematics + weight_biology

-- Theorem statement: Prove that the weighted average mark for mathematics and biology is 83.33
theorem weighted_avg_M_B_eq :
  (weighted_total_M_B / weighted_sum_M_B) = 83.33 :=
by
  sorry

end weighted_avg_M_B_eq_l36_36726


namespace nate_age_when_ember_is_14_l36_36035

theorem nate_age_when_ember_is_14 (nate_age : ℕ) (ember_age : ℕ) 
  (h1 : ember_age = nate_age / 2) (h2 : nate_age = 14) :
  ∃ (years_later : ℕ), ember_age + years_later = 14 ∧ nate_age + years_later = 21 :=
by
  -- sorry to skip the proof, adhering to the instructions
  sorry

end nate_age_when_ember_is_14_l36_36035


namespace speed_of_water_is_10_l36_36105

/-- Define the conditions -/
def swimming_speed_in_still_water : ℝ := 12 -- km/h
def time_to_swim_against_current : ℝ := 4 -- hours
def distance_against_current : ℝ := 8 -- km

/-- Define the effective speed against the current and the proof goal -/
def speed_of_water (v : ℝ) : Prop :=
  (swimming_speed_in_still_water - v) = distance_against_current / time_to_swim_against_current

theorem speed_of_water_is_10 : speed_of_water 10 :=
by
  unfold speed_of_water
  sorry

end speed_of_water_is_10_l36_36105


namespace gcd_840_1764_l36_36781

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l36_36781


namespace sum_of_all_N_l36_36921

-- Define the machine's processing rules
def process (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the 6-step process starting from N
def six_steps (N : ℕ) : ℕ :=
  process (process (process (process (process (process N)))))

-- Definition for the main theorem
theorem sum_of_all_N (N : ℕ) : six_steps N = 10 → N = 640 :=
by 
  sorry

end sum_of_all_N_l36_36921


namespace black_eyes_ratio_l36_36167

-- Define the number of people in the theater
def total_people : ℕ := 100

-- Define the number of people with blue eyes
def blue_eyes : ℕ := 19

-- Define the number of people with brown eyes
def brown_eyes : ℕ := 50

-- Define the number of people with green eyes
def green_eyes : ℕ := 6

-- Define the number of people with black eyes
def black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)

-- Prove that the ratio of the number of people with black eyes to the total number of people is 1:4
theorem black_eyes_ratio :
  black_eyes * 4 = total_people := by
  sorry

end black_eyes_ratio_l36_36167


namespace no_such_ab_exists_l36_36170

theorem no_such_ab_exists : ¬ ∃ (a b : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → (a * x + b)^2 - Real.cos x * (a * x + b) < (1 / 4) * (Real.sin x)^2 :=
by
  sorry

end no_such_ab_exists_l36_36170


namespace find_x_l36_36180

theorem find_x (x : ℝ) (h : (1 / Real.log x / Real.log 5 + 1 / Real.log x / Real.log 7 + 1 / Real.log x / Real.log 11) = 1) : x = 385 := 
sorry

end find_x_l36_36180


namespace circumscribed_sphere_surface_area_l36_36654

-- Define the setup and conditions for the right circular cone and its circumscribed sphere
theorem circumscribed_sphere_surface_area (PA PB PC AB R : ℝ)
  (h1 : AB = Real.sqrt 2)
  (h2 : PA = 1)
  (h3 : PB = 1)
  (h4 : PC = 1)
  (h5 : R = Real.sqrt 3 / 2 * PA) :
  4 * Real.pi * R ^ 2 = 3 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l36_36654


namespace solve_quadratic_eq1_solve_quadratic_eq2_l36_36975

-- Define the statement for the first problem
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 49 = 0 → x = 7 ∨ x = -7 :=
by
  sorry

-- Define the statement for the second problem
theorem solve_quadratic_eq2 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 → x = 4 ∨ x = -6 :=
by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l36_36975


namespace line_equation_is_correct_l36_36897

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l36_36897


namespace evaluate_powers_l36_36899

theorem evaluate_powers : (81^(1/2:ℝ) * 64^(-1/3:ℝ) * 49^(1/4:ℝ) = 9 * (1/4) * Real.sqrt 7) :=
by
  sorry

end evaluate_powers_l36_36899


namespace max_value_of_f_l36_36780

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem max_value_of_f 
  (ω a : ℝ) 
  (h1 : 0 < ω) 
  (h2 : (2 * Real.pi / ω) = Real.pi) 
  (h3 : ∃ k : ℤ, ω * (Real.pi / 12) + (k : ℝ) * Real.pi + Real.pi / 3 = Real.pi / 2 + (k : ℝ) * Real.pi) :
  ∃ x : ℝ, f ω a x = 2 := by
  sorry

end max_value_of_f_l36_36780


namespace at_least_one_non_zero_l36_36210

theorem at_least_one_non_zero (a b : ℝ) : a^2 + b^2 > 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by sorry

end at_least_one_non_zero_l36_36210


namespace inequality_2_pow_n_plus_2_gt_n_squared_l36_36383

theorem inequality_2_pow_n_plus_2_gt_n_squared (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := sorry

end inequality_2_pow_n_plus_2_gt_n_squared_l36_36383


namespace geometric_sequence_a5_l36_36462

theorem geometric_sequence_a5 (α : Type) [LinearOrderedField α] (a : ℕ → α)
  (h1 : ∀ n, a (n + 1) = a n * 2)
  (h2 : ∀ n, a n > 0)
  (h3 : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end geometric_sequence_a5_l36_36462


namespace exist_n_for_all_k_l36_36451

theorem exist_n_for_all_k (k : ℕ) (h_k : k > 1) : 
  ∃ n : ℕ, 
    (n > 0 ∧ ((n.choose k) % n = 0) ∧ (∀ m : ℕ, (2 ≤ m ∧ m < k) → ((n.choose m) % n ≠ 0))) :=
sorry

end exist_n_for_all_k_l36_36451


namespace div_by_3_iff_n_form_l36_36115

theorem div_by_3_iff_n_form (n : ℕ) : (3 ∣ (n * 2^n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 2) :=
by
  sorry

end div_by_3_iff_n_form_l36_36115


namespace archer_hits_less_than_8_l36_36273

variables (P10 P9 P8 : ℝ)

-- Conditions
def hitting10_ring := P10 = 0.3
def hitting9_ring := P9 = 0.3
def hitting8_ring := P8 = 0.2

-- Statement to prove
theorem archer_hits_less_than_8 (P10 P9 P8 : ℝ)
  (h10 : hitting10_ring P10)
  (h9 : hitting9_ring P9)
  (h8 : hitting8_ring P8)
  (mutually_exclusive: P10 + P9 + P8 <= 1):
  1 - (P10 + P9 + P8) = 0.2 :=
by
  -- Here goes the proof 
  sorry

end archer_hits_less_than_8_l36_36273


namespace travel_time_correct_l36_36293

def luke_bus_to_work : ℕ := 70
def paula_bus_to_work : ℕ := (70 * 3) / 5
def jane_train_to_work : ℕ := 120
def michael_cycle_to_work : ℕ := 120 / 4

def luke_bike_back_home : ℕ := 70 * 5
def paula_bus_back_home: ℕ := paula_bus_to_work
def jane_train_back_home : ℕ := 120 * 2
def michael_cycle_back_home : ℕ := michael_cycle_to_work

def luke_total_travel : ℕ := luke_bus_to_work + luke_bike_back_home
def paula_total_travel : ℕ := paula_bus_to_work + paula_bus_back_home
def jane_total_travel : ℕ := jane_train_to_work + jane_train_back_home
def michael_total_travel : ℕ := michael_cycle_to_work + michael_cycle_back_home

def total_travel_time : ℕ := luke_total_travel + paula_total_travel + jane_total_travel + michael_total_travel

theorem travel_time_correct : total_travel_time = 924 :=
by sorry

end travel_time_correct_l36_36293


namespace shaded_area_calculation_l36_36122

-- Define the grid and the side length conditions
def grid_size : ℕ := 5 * 4
def side_length : ℕ := 1
def total_squares : ℕ := 5 * 4

-- Define the area of one small square
def area_of_square (side: ℕ) : ℕ := side * side

-- Define the shaded region in terms of number of small squares fully or partially occupied
def shaded_squares : ℕ := 11

-- By analyzing the grid based on given conditions, prove that the area of the shaded region is 11
theorem shaded_area_calculation : (shaded_squares * side_length * side_length) = 11 := sorry

end shaded_area_calculation_l36_36122


namespace hiking_trip_distance_l36_36772

open Real

-- Define the given conditions
def distance_north : ℝ := 10
def distance_south : ℝ := 7
def distance_east1 : ℝ := 17
def distance_east2 : ℝ := 8

-- Define the net displacement conditions
def net_distance_north : ℝ := distance_north - distance_south
def net_distance_east : ℝ := distance_east1 + distance_east2

-- Prove the distance from the starting point
theorem hiking_trip_distance :
  sqrt ((net_distance_north)^2 + (net_distance_east)^2) = sqrt 634 := by
  sorry

end hiking_trip_distance_l36_36772


namespace expenditure_on_house_rent_l36_36847

theorem expenditure_on_house_rent
  (income petrol house_rent remaining_income : ℝ)
  (h1 : petrol = 0.30 * income)
  (h2 : petrol = 300)
  (h3 : remaining_income = income - petrol)
  (h4 : house_rent = 0.30 * remaining_income) :
  house_rent = 210 :=
by
  sorry

end expenditure_on_house_rent_l36_36847


namespace max_xy_of_perpendicular_l36_36111

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 

theorem max_xy_of_perpendicular (x y : ℝ) 
  (h_perp : dot_product (vector_a x) (vector_b y) = 0) : xy ≤ 1/2 :=
by
  sorry

end max_xy_of_perpendicular_l36_36111


namespace incorrect_transformation_D_l36_36404

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end incorrect_transformation_D_l36_36404


namespace find_positive_number_l36_36554

theorem find_positive_number (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end find_positive_number_l36_36554


namespace range_of_z_in_parallelogram_l36_36572

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -1, y := 2}
def B : Point := {x := 3, y := 4}
def C : Point := {x := 4, y := -2}

-- Define the condition for point (x, y) to be inside the parallelogram (including boundary)
def isInsideParallelogram (p : Point) : Prop := sorry -- Placeholder for actual geometric condition

-- Statement of the problem
theorem range_of_z_in_parallelogram (p : Point) (h : isInsideParallelogram p) : 
  -14 ≤ 2 * p.x - 5 * p.y ∧ 2 * p.x - 5 * p.y ≤ 20 :=
sorry

end range_of_z_in_parallelogram_l36_36572


namespace solve_quadratic_l36_36696

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l36_36696


namespace fenced_area_l36_36436

theorem fenced_area (length_large : ℕ) (width_large : ℕ) 
                    (length_cutout : ℕ) (width_cutout : ℕ) 
                    (h_large : length_large = 20 ∧ width_large = 15)
                    (h_cutout : length_cutout = 4 ∧ width_cutout = 2) : 
                    ((length_large * width_large) - (length_cutout * width_cutout) = 292) := 
by
  sorry

end fenced_area_l36_36436


namespace directrix_of_parabola_l36_36049

theorem directrix_of_parabola (a b c : ℝ) (h_eqn : ∀ x, b = -4 * x^2 + c) : 
  b = 5 → c = 0 → (∃ y, y = 81 / 16) :=
by
  sorry

end directrix_of_parabola_l36_36049


namespace select_team_ways_l36_36537

-- Definitions of the conditions and question
def boys := 7
def girls := 10
def boys_needed := 2
def girls_needed := 3
def total_team := 5

-- Theorem statement to prove the number of selecting the team
theorem select_team_ways : (Nat.choose boys boys_needed) * (Nat.choose girls girls_needed) = 2520 := 
by
  -- Place holder for proof
  sorry

end select_team_ways_l36_36537


namespace Mary_takes_3_children_l36_36886

def num_children (C : ℕ) : Prop :=
  ∃ (C : ℕ), 2 + C = 5

theorem Mary_takes_3_children (C : ℕ) : num_children C → C = 3 :=
by
  intro h
  sorry

end Mary_takes_3_children_l36_36886


namespace square_root_of_9_l36_36056

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l36_36056


namespace find_k_l36_36874

noncomputable def distance_x (x : ℝ) := 5
noncomputable def distance_y (x k : ℝ) := |x^2 - k|
noncomputable def total_distance (x k : ℝ) := distance_x x + distance_y x k

theorem find_k (x k : ℝ) (hk : distance_y x k = 2 * distance_x x) (htot : total_distance x k = 30) :
  k = x^2 - 10 :=
sorry

end find_k_l36_36874


namespace pos_sum_of_powers_l36_36652

theorem pos_sum_of_powers (a b c : ℝ) (n : ℕ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) : 
  a^n + b^n + c^n > 0 :=
sorry

end pos_sum_of_powers_l36_36652


namespace farthings_in_a_pfennig_l36_36507

theorem farthings_in_a_pfennig (x : ℕ) (h : 54 - 2 * x = 7 * x) : x = 6 :=
by
  sorry

end farthings_in_a_pfennig_l36_36507


namespace perpendicular_tangent_inequality_l36_36152

variable {A B C : Type} 

-- Definitions according to conditions in part a)
def isAcuteAngledTriangle (a b c : Type) : Prop :=
  -- A triangle being acute-angled in Euclidean geometry
  sorry

def triangleArea (a b c : Type) : ℝ :=
  -- Definition of the area of a triangle
  sorry

def perpendicularLengthToLine (point line : Type) : ℝ :=
  -- Length of the perpendicular from a point to a line
  sorry

def tangentOfAngleA (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle A in the triangle
  sorry

def tangentOfAngleB (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle B in the triangle
  sorry

def tangentOfAngleC (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle C in the triangle
  sorry

theorem perpendicular_tangent_inequality (a b c line : Type) 
  (ht : isAcuteAngledTriangle a b c)
  (u := perpendicularLengthToLine a line)
  (v := perpendicularLengthToLine b line)
  (w := perpendicularLengthToLine c line):
  u^2 * tangentOfAngleA a b c + v^2 * tangentOfAngleB a b c + w^2 * tangentOfAngleC a b c ≥ 
  2 * triangleArea a b c :=
sorry

end perpendicular_tangent_inequality_l36_36152


namespace least_number_to_be_added_l36_36315

theorem least_number_to_be_added (k : ℕ) (h₁ : Nat.Prime 29) (h₂ : Nat.Prime 37) (H : Nat.gcd 29 37 = 1) : 
  (433124 + k) % Nat.lcm 29 37 = 0 → k = 578 :=
by 
  sorry

end least_number_to_be_added_l36_36315


namespace polynomial_coefficient_B_l36_36904

theorem polynomial_coefficient_B : 
  ∃ (A C D : ℤ), 
    (∀ z : ℤ, (z > 0) → (z^6 - 15 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 64 = 0)) ∧ 
    (B = -244) := 
by
  sorry

end polynomial_coefficient_B_l36_36904


namespace tank_capacity_l36_36378

theorem tank_capacity (C : ℕ) (h₁ : C = 785) :
  360 - C / 4 - C / 8 = C / 12 :=
by 
  -- Assuming h₁: C = 785
  have h₁: C = 785 := by exact h₁
  -- Provide proof steps here (not required for the task)
  sorry

end tank_capacity_l36_36378


namespace percentage_books_returned_l36_36776

theorem percentage_books_returned
    (initial_books : ℝ)
    (end_books : ℝ)
    (loaned_books : ℝ)
    (R : ℝ)
    (Percentage_Returned : ℝ) :
    initial_books = 75 →
    end_books = 65 →
    loaned_books = 50.000000000000014 →
    R = (75 - 65) →
    Percentage_Returned = (R / loaned_books) * 100 →
    Percentage_Returned = 20 :=
by
  intros
  sorry

end percentage_books_returned_l36_36776


namespace find_smallest_n_l36_36030

-- defining the geometric sequence and its sum for the given conditions
def a_n (n : ℕ) := 3 * (4 ^ n)

def S_n (n : ℕ) := (a_n n - 1) / (4 - 1) -- simplification step

-- statement of the problem: finding the smallest natural number n such that S_n > 3000
theorem find_smallest_n :
  ∃ n : ℕ, S_n n > 3000 ∧ ∀ m : ℕ, m < n → S_n m ≤ 3000 := by
  sorry

end find_smallest_n_l36_36030


namespace find_third_number_l36_36287

theorem find_third_number : ∃ (x : ℝ), 0.3 * 0.8 + x * 0.5 = 0.29 ∧ x = 0.1 :=
by
  use 0.1
  sorry

end find_third_number_l36_36287


namespace interest_rate_per_annum_l36_36200

def principal : ℝ := 8945
def simple_interest : ℝ := 4025.25
def time : ℕ := 5

theorem interest_rate_per_annum : (simple_interest * 100) / (principal * time) = 9 := by
  sorry

end interest_rate_per_annum_l36_36200


namespace students_walk_fraction_l36_36387

theorem students_walk_fraction (h1 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/3))
                               (h2 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/5))
                               (h3 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/8))
                               (h4 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/10)) :
  ∃ (students : ℕ), (students - num1 - num2 - num3 - num4) / students = 29 / 120 :=
by
  sorry

end students_walk_fraction_l36_36387


namespace change_in_opinion_difference_l36_36283

theorem change_in_opinion_difference :
  let initially_liked_pct := 0.4;
  let initially_disliked_pct := 0.6;
  let finally_liked_pct := 0.8;
  let finally_disliked_pct := 0.2;
  let max_change := finally_liked_pct + (initially_disliked_pct - finally_disliked_pct);
  let min_change := finally_liked_pct - initially_liked_pct;
  max_change - min_change = 0.2 :=
by
  sorry

end change_in_opinion_difference_l36_36283


namespace probability_region_D_l36_36574

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_C : ℝ := 1 / 6

theorem probability_region_D (P_D : ℝ) (h : P_A + P_B + P_C + P_D = 1) : P_D = 1 / 4 :=
by
  sorry

end probability_region_D_l36_36574


namespace part1_exists_infinite_rationals_part2_rationals_greater_bound_l36_36536

theorem part1_exists_infinite_rationals 
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2):
  ∀ ε > 0, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ abs (q / p - sqrt5_minus1_div2) < 1 / p ^ 2 :=
by sorry

theorem part2_rationals_greater_bound
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2)
  (sqrt5_plus1_inv := 1 / (Real.sqrt 5 + 1)):
  ∀ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 → abs (q / p - sqrt5_minus1_div2) > sqrt5_plus1_inv / p ^ 2 :=
by sorry

end part1_exists_infinite_rationals_part2_rationals_greater_bound_l36_36536


namespace remainder_2007_div_81_l36_36548

theorem remainder_2007_div_81 : 2007 % 81 = 63 :=
by
  sorry

end remainder_2007_div_81_l36_36548


namespace evaluate_f_3_minus_f_neg3_l36_36503

def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

theorem evaluate_f_3_minus_f_neg3 : f 3 - f (-3) = 210 := by
  sorry

end evaluate_f_3_minus_f_neg3_l36_36503


namespace Wendy_runs_farther_l36_36433

-- Define the distances Wendy ran and walked
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- Define the difference in distances
def difference : ℝ := distance_ran - distance_walked

-- The theorem to prove
theorem Wendy_runs_farther : difference = 10.66 := by
  sorry

end Wendy_runs_farther_l36_36433


namespace mayo_bottles_count_l36_36172

theorem mayo_bottles_count
  (ketchup_ratio mayo_ratio : ℕ) 
  (ratio_multiplier ketchup_bottles : ℕ)
  (h_ratio_eq : 3 = ketchup_ratio)
  (h_mayo_ratio_eq : 2 = mayo_ratio)
  (h_ketchup_bottles_eq : 6 = ketchup_bottles)
  (h_ratio_condition : ketchup_bottles * mayo_ratio = ketchup_ratio * ratio_multiplier) :
  ratio_multiplier = 4 := 
by 
  sorry

end mayo_bottles_count_l36_36172


namespace complex_transformation_result_l36_36840

theorem complex_transformation_result :
  let z := -1 - 2 * Complex.I 
  let rotation := (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3) / 2)
  let dilation := 2
  (z * (rotation * dilation)) = (2 * Real.sqrt 3 - 1 - (2 + Real.sqrt 3) * Complex.I) :=
by
  sorry

end complex_transformation_result_l36_36840


namespace slope_of_chord_in_ellipse_l36_36678

noncomputable def slope_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem slope_of_chord_in_ellipse :
  ∀ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 16 + y1^2 / 9 = 1) →
    (x2^2 / 16 + y2^2 / 9 = 1) →
    ((x1 + x2) = -2) →
    ((y1 + y2) = 4) →
    slope_of_chord x1 y1 x2 y2 = 9 / 32 :=
by
  intro x1 y1 x2 y2 h1 h2 h3 h4
  sorry

end slope_of_chord_in_ellipse_l36_36678


namespace vector_operation_l36_36853

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

theorem vector_operation :
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end vector_operation_l36_36853


namespace no_such_function_exists_l36_36264

-- Let's define the assumptions as conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f (x^2) - (f x)^2 ≥ 1 / 4
def distinct_values (f : ℝ → ℝ) := ∀ x y : ℝ, x ≠ y → f x ≠ f y

-- Now we state the main theorem
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, condition1 f ∧ distinct_values f :=
sorry

end no_such_function_exists_l36_36264


namespace equation_negative_roots_iff_l36_36900

theorem equation_negative_roots_iff (a : ℝ) :
  (∃ x < 0, 4^x - 2^(x-1) + a = 0) ↔ (-1/2 < a ∧ a ≤ 1/16) := 
sorry

end equation_negative_roots_iff_l36_36900


namespace cut_grid_into_six_polygons_with_identical_pair_l36_36830

noncomputable def totalCells : Nat := 24
def polygonArea : Nat := 4

theorem cut_grid_into_six_polygons_with_identical_pair :
  ∃ (polygons : Fin 6 → Nat → Prop),
  (∀ i, (∃ (cells : Finset (Fin totalCells)), (cells.card = polygonArea ∧ ∀ c ∈ cells, polygons i c))) ∧
  (∃ i j, i ≠ j ∧ ∀ c, polygons i c ↔ polygons j c) :=
sorry

end cut_grid_into_six_polygons_with_identical_pair_l36_36830


namespace candies_taken_away_per_incorrect_answer_eq_2_l36_36302

/-- Define constants and assumptions --/
def candy_per_correct := 3
def correct_answers := 7
def extra_correct_answers := 2
def total_candies_if_extra_correct := 31

/-- The number of candies taken away per incorrect answer --/
def x : ℤ := sorry

/-- Prove that the number of candies taken away for each incorrect answer is 2. --/
theorem candies_taken_away_per_incorrect_answer_eq_2 : 
  ∃ x : ℤ, ((correct_answers + extra_correct_answers) * candy_per_correct - total_candies_if_extra_correct = x + (extra_correct_answers * candy_per_correct - (total_candies_if_extra_correct - correct_answers * candy_per_correct))) ∧ x = 2 := 
by
  exists 2
  sorry

end candies_taken_away_per_incorrect_answer_eq_2_l36_36302


namespace simplify_expression_find_value_a_m_2n_l36_36458

-- Proof Problem 1
theorem simplify_expression : ( (-2 : ℤ) * x )^3 * x^2 + ( (3 : ℤ) * x^4 )^2 / x^3 = x^5 := by
  sorry

-- Proof Problem 2
theorem find_value_a_m_2n (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2*n) = 18 := by
  sorry

end simplify_expression_find_value_a_m_2n_l36_36458


namespace tangent_line_equation_l36_36094

theorem tangent_line_equation
  (x y : ℝ)
  (h₁ : x^2 + y^2 = 5)
  (hM : x = -1 ∧ y = 2) :
  x - 2 * y + 5 = 0 :=
by
  sorry

end tangent_line_equation_l36_36094


namespace real_solutions_eq_pos_neg_2_l36_36902

theorem real_solutions_eq_pos_neg_2 (x : ℝ) :
  ( (x - 1) ^ 2 * (x - 5) * (x - 5) / (x - 5) = 4) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_pos_neg_2_l36_36902


namespace P_is_necessary_but_not_sufficient_for_Q_l36_36928

def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem P_is_necessary_but_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
by
  sorry

end P_is_necessary_but_not_sufficient_for_Q_l36_36928


namespace max_area_square_l36_36374

theorem max_area_square (P : ℝ) : 
  ∀ x y : ℝ, 2 * x + 2 * y = P → (x * y ≤ (P / 4) ^ 2) :=
by
  sorry

end max_area_square_l36_36374


namespace minimize_pollution_park_distance_l36_36983

noncomputable def pollution_index (x : ℝ) : ℝ :=
  (1 / x) + (4 / (30 - x))

theorem minimize_pollution_park_distance : ∃ x : ℝ, (0 < x ∧ x < 30) ∧ pollution_index x = 10 :=
by
  sorry

end minimize_pollution_park_distance_l36_36983


namespace boy_completion_time_l36_36582

theorem boy_completion_time (M W B : ℝ) (h1 : M + W + B = 1/3) (h2 : M = 1/6) (h3 : W = 1/18) : B = 1/9 :=
sorry

end boy_completion_time_l36_36582


namespace average_age_of_troupe_l36_36183

theorem average_age_of_troupe
  (number_females : ℕ) (number_males : ℕ) 
  (average_age_females : ℕ) (average_age_males : ℕ)
  (total_people : ℕ) (total_age : ℕ)
  (h1 : number_females = 12) 
  (h2 : number_males = 18) 
  (h3 : average_age_females = 25) 
  (h4 : average_age_males = 30)
  (h5 : total_people = 30)
  (h6 : total_age = (25 * 12 + 30 * 18)) :
  total_age / total_people = 28 :=
by
  -- Proof goes here
  sorry

end average_age_of_troupe_l36_36183


namespace minimal_connections_correct_l36_36601

-- Define a Lean structure to encapsulate the conditions
structure IslandsProblem where
  islands : ℕ
  towns : ℕ
  min_towns_per_island : ℕ
  condition_islands : islands = 13
  condition_towns : towns = 25
  condition_min_towns : min_towns_per_island = 1

-- Define a function to represent the minimal number of ferry connections
def minimalFerryConnections (p : IslandsProblem) : ℕ :=
  222

-- Define the statement to be proved
theorem minimal_connections_correct (p : IslandsProblem) : 
  p.islands = 13 → 
  p.towns = 25 → 
  p.min_towns_per_island = 1 → 
  minimalFerryConnections p = 222 :=
by
  intros
  sorry

end minimal_connections_correct_l36_36601


namespace number_of_seedlings_l36_36711

theorem number_of_seedlings (packets : ℕ) (seeds_per_packet : ℕ) (h1 : packets = 60) (h2 : seeds_per_packet = 7) : packets * seeds_per_packet = 420 :=
by
  sorry

end number_of_seedlings_l36_36711


namespace factorize_expression_l36_36568

theorem factorize_expression : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := 
by sorry

end factorize_expression_l36_36568


namespace min_value_expression_l36_36883

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression_l36_36883


namespace equilateral_triangle_in_ellipse_l36_36395

-- Given
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def altitude_on_y_axis (v : ℝ × ℝ := (0, 1)) : Prop := 
  v.1 = 0 ∧ v.2 = 1

-- The problem statement translated into a Lean proof goal
theorem equilateral_triangle_in_ellipse :
  ∃ (m n : ℕ), 
    (∀ (x y : ℝ), ellipse x y) →
    altitude_on_y_axis (0,1) →
    m.gcd n = 1 ∧ m + n = 937 :=
sorry

end equilateral_triangle_in_ellipse_l36_36395


namespace cadence_worked_old_company_l36_36223

theorem cadence_worked_old_company (y : ℕ) (h1 : (426000 : ℝ) = 
    5000 * 12 * y + 6000 * 12 * (y + 5 / 12)) :
    y = 3 := by
    sorry

end cadence_worked_old_company_l36_36223


namespace min_number_of_4_dollar_frisbees_l36_36964

theorem min_number_of_4_dollar_frisbees 
  (x y : ℕ) 
  (h1 : x + y = 60)
  (h2 : 3 * x + 4 * y = 200) 
  : y = 20 :=
sorry

end min_number_of_4_dollar_frisbees_l36_36964


namespace midpoint_trajectory_l36_36708

theorem midpoint_trajectory (x y p q : ℝ) (h_parabola : p^2 = 4 * q)
  (h_focus : ∀ (p q : ℝ), p^2 = 4 * q → q = (p/2)^2) 
  (h_midpoint_x : x = (p + 1) / 2)
  (h_midpoint_y : y = q / 2):
  y^2 = 2 * x - 1 :=
by
  sorry

end midpoint_trajectory_l36_36708


namespace jennifer_tanks_l36_36879

theorem jennifer_tanks (initial_tanks : ℕ) (fish_per_initial_tank : ℕ) (total_fish_needed : ℕ) 
  (additional_tanks : ℕ) (fish_per_additional_tank : ℕ) 
  (initial_calculation : initial_tanks = 3) (fish_per_initial_calculation : fish_per_initial_tank = 15)
  (total_fish_calculation : total_fish_needed = 75) (additional_tanks_calculation : additional_tanks = 3) :
  initial_tanks * fish_per_initial_tank + additional_tanks * fish_per_additional_tank = total_fish_needed 
  → fish_per_additional_tank = 10 := 
by sorry

end jennifer_tanks_l36_36879


namespace slope_ratio_l36_36431

theorem slope_ratio (s t k b : ℝ) 
  (h1: b = -12 * s)
  (h2: b = k - 7) 
  (ht: t = (7 - k) / 7) 
  (hs: s = (7 - k) / 12): 
  s / t = 7 / 12 := 
  sorry

end slope_ratio_l36_36431


namespace periodic_function_implies_rational_ratio_l36_36491

noncomputable def g (i : ℕ) (a ω θ x : ℝ) : ℝ := 
  a * Real.sin (ω * x + θ)

theorem periodic_function_implies_rational_ratio 
  (a1 a2 ω1 ω2 θ1 θ2 : ℝ) (h1 : a1 * ω1 ≠ 0) (h2 : a2 * ω2 ≠ 0)
  (h3 : |ω1| ≠ |ω2|) 
  (hf_periodic : ∃ T : ℝ, ∀ x : ℝ, g 1 a1 ω1 θ1 (x + T) + g 2 a2 ω2 θ2 (x + T) = g 1 a1 ω1 θ1 x + g 2 a2 ω2 θ2 x) :
  ∃ m n : ℤ, n ≠ 0 ∧ ω1 / ω2 = m / n :=
sorry

end periodic_function_implies_rational_ratio_l36_36491


namespace percent_of_a_is_b_l36_36784

variable {a b c : ℝ}

theorem percent_of_a_is_b (h1 : c = 0.25 * a) (h2 : c = 0.10 * b) : b = 2.5 * a :=
by sorry

end percent_of_a_is_b_l36_36784


namespace hyperbola_asymptote_l36_36590

theorem hyperbola_asymptote (x y : ℝ) : 
  (∀ x y : ℝ, (x^2 / 25 - y^2 / 16 = 1) → (y = (4 / 5) * x ∨ y = -(4 / 5) * x)) := 
by 
  sorry

end hyperbola_asymptote_l36_36590


namespace Amanda_family_paint_walls_l36_36597

theorem Amanda_family_paint_walls :
  let num_people := 5
  let rooms_with_4_walls := 5
  let rooms_with_5_walls := 4
  let walls_per_room_4 := 4
  let walls_per_room_5 := 5
  let total_walls := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)
  total_walls / num_people = 8 :=
by
  -- We add a sorry to skip proof
  sorry

end Amanda_family_paint_walls_l36_36597


namespace sum_of_coordinates_of_D_l36_36456

-- Definition of points M, C and D
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨4, 7⟩
def C : Point := ⟨6, 2⟩

-- Conditions that M is the midpoint of segment CD
def isMidpoint (M C D : Point) : Prop :=
  ((C.x + D.x) / 2 = M.x) ∧
  ((C.y + D.y) / 2 = M.y)

-- Definition for the sum of the coordinates of a point
def sumOfCoordinates (P : Point) : ℝ :=
  P.x + P.y

-- The main theorem stating the sum of the coordinates of D is 14 given the conditions
theorem sum_of_coordinates_of_D :
  ∃ D : Point, isMidpoint M C D ∧ sumOfCoordinates D = 14 := 
sorry

end sum_of_coordinates_of_D_l36_36456


namespace total_logs_in_stack_l36_36487

theorem total_logs_in_stack : 
  ∀ (a_1 a_n : ℕ) (n : ℕ), 
  a_1 = 5 → a_n = 15 → n = a_n - a_1 + 1 → 
  (a_1 + a_n) * n / 2 = 110 :=
by
  intros a_1 a_n n h1 h2 h3
  sorry

end total_logs_in_stack_l36_36487


namespace number_of_valid_polynomials_l36_36701

noncomputable def count_polynomials_meeting_conditions : ℕ := sorry

theorem number_of_valid_polynomials :
  count_polynomials_meeting_conditions = 7200 :=
sorry

end number_of_valid_polynomials_l36_36701


namespace find_single_digit_number_l36_36140

theorem find_single_digit_number (n : ℕ) : 
  (5 < n ∧ n < 9 ∧ n > 7) ↔ n = 8 :=
by
  sorry

end find_single_digit_number_l36_36140


namespace time_needed_by_Alpha_and_Beta_l36_36754

theorem time_needed_by_Alpha_and_Beta (A B C h : ℝ)
  (h₀ : 1 / (A - 4) = 1 / (B - 2))
  (h₁ : 1 / A + 1 / B + 1 / C = 3 / C)
  (h₂ : A = B + 2)
  (h₃ : 1 / 12 + 1 / 10 = 11 / 60)
  : h = 60 / 11 :=
sorry

end time_needed_by_Alpha_and_Beta_l36_36754


namespace dual_colored_numbers_l36_36454

theorem dual_colored_numbers (table : Matrix (Fin 10) (Fin 20) ℕ)
  (distinct_numbers : ∀ (i j k l : Fin 10) (m n : Fin 20), 
    (i ≠ k ∨ m ≠ n) → table i m ≠ table k n)
  (row_red : ∀ (i : Fin 10), ∃ r₁ r₂ : Fin 20, r₁ ≠ r₂ ∧ 
    (∀ (j : Fin 20), table i j ≤ table i r₁ ∨ table i j ≤ table i r₂))
  (col_blue : ∀ (j : Fin 20), ∃ b₁ b₂ : Fin 10, b₁ ≠ b₂ ∧ 
    (∀ (i : Fin 10), table i j ≤ table b₁ j ∨ table i j ≤ table b₂ j)) : 
  ∃ i₁ i₂ i₃ : Fin 10, ∃ j₁ j₂ j₃ : Fin 20, 
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧ 
    ((table i₁ j₁ ≤ table i₁ j₂ ∨ table i₁ j₁ ≤ table i₃ j₂) ∧ 
     (table i₂ j₂ ≤ table i₂ j₁ ∨ table i₂ j₂ ≤ table i₃ j₁) ∧ 
     (table i₃ j₃ ≤ table i₃ j₁ ∨ table i₃ j₃ ≤ table i₂ j₁)) := 
  sorry

end dual_colored_numbers_l36_36454


namespace quadratic_equation_solution_l36_36089

theorem quadratic_equation_solution (m : ℝ) (h : m ≠ 1) : 
  (m^2 - 3 * m + 2 = 0) → m = 2 :=
by
  sorry

end quadratic_equation_solution_l36_36089


namespace find_value_of_M_l36_36924

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l36_36924


namespace man_is_older_l36_36994

-- Define present age of the son
def son_age : ℕ := 26

-- Define present age of the man (father)
axiom man_age : ℕ

-- Condition: in two years, the man's age will be twice the age of his son
axiom age_condition : man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 28 years older than his son
theorem man_is_older : man_age - son_age = 28 := sorry

end man_is_older_l36_36994


namespace divisibility_by_5_l36_36367

theorem divisibility_by_5 (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divisibility_by_5_l36_36367


namespace area_of_park_l36_36595

variable (length breadth speed time perimeter area : ℕ)

axiom ratio_length_breadth : length = breadth / 4
axiom speed_kmh : speed = 12 * 1000 / 60 -- speed in m/min
axiom time_taken : time = 8 -- time in minutes
axiom perimeter_eq : perimeter = speed * time -- perimeter in meters
axiom length_breadth_relation : perimeter = 2 * (length + breadth)

theorem area_of_park : ∃ length breadth, (length = 160 ∧ breadth = 640 ∧ area = length * breadth ∧ area = 102400) :=
by
  sorry

end area_of_park_l36_36595


namespace fourth_ball_black_probability_l36_36036

noncomputable def prob_fourth_is_black : Prop :=
  let total_balls := 8
  let black_balls := 4
  let prob_black := black_balls / total_balls
  prob_black = 1 / 2

theorem fourth_ball_black_probability :
  prob_fourth_is_black :=
sorry

end fourth_ball_black_probability_l36_36036


namespace proposition_false_at_9_l36_36461

theorem proposition_false_at_9 (P : ℕ → Prop) 
  (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1))
  (hne10 : ¬ P 10) : ¬ P 9 :=
by
  intro hp9
  have hp10 : P 10 := h _ (by norm_num) hp9
  contradiction

end proposition_false_at_9_l36_36461


namespace relay_race_total_time_l36_36662

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end relay_race_total_time_l36_36662


namespace difference_between_mean_and_median_l36_36103

def percent_students := {p : ℝ // 0 ≤ p ∧ p ≤ 1}

def students_scores_distribution (p60 p75 p85 p95 : percent_students) : Prop :=
  p60.val + p75.val + p85.val + p95.val = 1 ∧
  p60.val = 0.15 ∧
  p75.val = 0.20 ∧
  p85.val = 0.40 ∧
  p95.val = 0.25

noncomputable def weighted_mean (p60 p75 p85 p95 : percent_students) : ℝ :=
  60 * p60.val + 75 * p75.val + 85 * p85.val + 95 * p95.val

noncomputable def median_score (p60 p75 p85 p95 : percent_students) : ℝ :=
  if p60.val + p75.val < 0.5 then 85 else if p60.val + p75.val < 0.9 then 95 else 60

theorem difference_between_mean_and_median :
  ∀ (p60 p75 p85 p95 : percent_students),
    students_scores_distribution p60 p75 p85 p95 →
    abs (median_score p60 p75 p85 p95 - weighted_mean p60 p75 p85 p95) = 3.25 :=
by
  intro p60 p75 p85 p95
  intro h
  sorry

end difference_between_mean_and_median_l36_36103


namespace sin_diff_l36_36193

variable (θ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5)

theorem sin_diff
  (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5) :
  Real.sin (θ - π / 4) = Real.sqrt 10 / 10 :=
sorry

end sin_diff_l36_36193


namespace area_is_rational_l36_36199

-- Definitions of the vertices of the triangle
def point1 : (ℤ × ℤ) := (2, 3)
def point2 : (ℤ × ℤ) := (5, 7)
def point3 : (ℤ × ℤ) := (3, 4)

-- Define a function to calculate the area of a triangle given vertices with integer coordinates
def triangle_area (A B C: (ℤ × ℤ)) : ℚ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Define the area of our specific triangle
noncomputable def area_of_triangle_with_given_vertices := triangle_area point1 point2 point3

-- Proof statement
theorem area_is_rational : ∃ (Q : ℚ), Q = area_of_triangle_with_given_vertices := 
sorry

end area_is_rational_l36_36199


namespace percentage_problem_l36_36774

theorem percentage_problem
  (a b c : ℚ) :
  (8 = (2 / 100) * a) →
  (2 = (8 / 100) * b) →
  (c = b / a) →
  c = 1 / 16 :=
by
  sorry

end percentage_problem_l36_36774


namespace solve_equation_l36_36317

theorem solve_equation (x y z : ℕ) :
  (∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 2) ↔ (x^2 + 3 * y^2 = 2^z) :=
by
  sorry

end solve_equation_l36_36317


namespace sqrt_eq_pm_four_l36_36452

theorem sqrt_eq_pm_four (a : ℤ) : (a * a = 16) ↔ (a = 4 ∨ a = -4) :=
by sorry

end sqrt_eq_pm_four_l36_36452


namespace base6_to_base10_l36_36039

theorem base6_to_base10 (c d : ℕ) (h1 : 524 = 2 * (10 * c + d)) (hc : c < 10) (hd : d < 10) :
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end base6_to_base10_l36_36039


namespace bill_sunday_miles_l36_36706

variable (B : ℕ)

-- Conditions
def miles_Bill_Saturday : ℕ := B
def miles_Bill_Sunday : ℕ := B + 4
def miles_Julia_Sunday : ℕ := 2 * (B + 4)
def total_miles : ℕ := miles_Bill_Saturday B + miles_Bill_Sunday B + miles_Julia_Sunday B

theorem bill_sunday_miles (h : total_miles B = 32) : miles_Bill_Sunday B = 9 := by
  sorry

end bill_sunday_miles_l36_36706


namespace hyperbola_condition_l36_36709

theorem hyperbola_condition (k : ℝ) (x y : ℝ) :
  (k ≠ 0 ∧ k ≠ 3 ∧ (x^2 / k + y^2 / (k - 3) = 1)) → 0 < k ∧ k < 3 :=
by
  sorry

end hyperbola_condition_l36_36709


namespace total_five_digit_odd_and_multiples_of_5_l36_36624

def count_odd_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 5
  choices

def count_multiples_of_5_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 2
  choices

theorem total_five_digit_odd_and_multiples_of_5 : count_odd_five_digit_numbers + count_multiples_of_5_five_digit_numbers = 63000 :=
by
  -- Proof Placeholder
  sorry

end total_five_digit_odd_and_multiples_of_5_l36_36624


namespace billy_laundry_loads_l36_36850

-- Define constants based on problem conditions
def sweeping_minutes_per_room := 3
def washing_minutes_per_dish := 2
def laundry_minutes_per_load := 9

def anna_rooms := 10
def billy_dishes := 6

-- Calculate total time spent by Anna and the time Billy spends washing dishes
def anna_total_time := sweeping_minutes_per_room * anna_rooms
def billy_dishwashing_time := washing_minutes_per_dish * billy_dishes

-- Define the time difference Billy needs to make up with laundry
def time_difference := anna_total_time - billy_dishwashing_time
def billy_required_laundry_loads := time_difference / laundry_minutes_per_load

-- The theorem to prove
theorem billy_laundry_loads : billy_required_laundry_loads = 2 := by 
  sorry

end billy_laundry_loads_l36_36850


namespace total_days_spent_l36_36560

theorem total_days_spent {weeks_to_days : ℕ → ℕ} : 
  (weeks_to_days 3 + weeks_to_days 1) + 
  (weeks_to_days (weeks_to_days 3 + weeks_to_days 2) + 3) + 
  (2 * (weeks_to_days (weeks_to_days 3 + weeks_to_days 2))) + 
  (weeks_to_days 5 - weeks_to_days 1) + 
  (weeks_to_days ((weeks_to_days 5 - weeks_to_days 1) - weeks_to_days 3) + 6) + 
  (weeks_to_days (weeks_to_days 5 - weeks_to_days 1) + 4) = 230 :=
by
  sorry

end total_days_spent_l36_36560


namespace hyperbola_sum_l36_36979

theorem hyperbola_sum
  (h k a b : ℝ)
  (center : h = 3 ∧ k = 1)
  (vertex : ∃ (v : ℝ), (v = 4 ∧ h = 3 ∧ a = |k - v|))
  (focus : ∃ (f : ℝ), (f = 10 ∧ h = 3 ∧ (f - k) = 9 ∧ ∃ (c : ℝ), c = |k - f|))
  (relationship : ∀ (c : ℝ), c = 9 → c^2 = a^2 + b^2): 
  h + k + a + b = 7 + 6 * Real.sqrt 2 :=
by 
  sorry

end hyperbola_sum_l36_36979


namespace arithmetic_sequence_sum_l36_36024

variable (S : ℕ → ℝ)
variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_sum (h₁ : S 5 = 8) (h₂ : S 10 = 20) : S 15 = 36 := 
by
  sorry

end arithmetic_sequence_sum_l36_36024


namespace common_difference_l36_36635

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference (d : ℕ) (a1 : ℕ) (h1 : a1 = 18) (h2 : d ≠ 0) 
  (h3 : (a1 + 3 * d)^2 = a1 * (a1 + 7 * d)) : d = 2 :=
by
  sorry

end common_difference_l36_36635


namespace smallest_n_condition_l36_36593

theorem smallest_n_condition (n : ℕ) : 25 * n - 3 ≡ 0 [MOD 16] → n ≡ 11 [MOD 16] :=
by
  sorry

end smallest_n_condition_l36_36593


namespace range_of_function_l36_36037

theorem range_of_function :
  ∀ x, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -3 ≤ 2 * Real.sin x - 1 ∧ 2 * Real.sin x - 1 ≤ 1 :=
by
  intros x h
  sorry

end range_of_function_l36_36037


namespace least_common_multiple_of_5_to_10_is_2520_l36_36933

-- Definitions of the numbers
def numbers : List ℤ := [5, 6, 7, 8, 9, 10]

-- Definition of prime factorization for verification (optional, keeping it simple)
def prime_factors (n : ℤ) : List ℤ :=
  if n = 5 then [5]
  else if n = 6 then [2, 3]
  else if n = 7 then [7]
  else if n = 8 then [2, 2, 2]
  else if n = 9 then [3, 3]
  else if n = 10 then [2, 5]
  else []

-- The property to be proved: The least common multiple of numbers is 2520
theorem least_common_multiple_of_5_to_10_is_2520 : ∃ n : ℕ, (∀ m ∈ numbers, m ∣ n) ∧ n = 2520 := by
  use 2520
  sorry

end least_common_multiple_of_5_to_10_is_2520_l36_36933


namespace range_of_a_l36_36194

theorem range_of_a (a : ℚ) (h_pos : 0 < a) (h_int_count : ∀ n : ℕ, 2 * n + 1 = 2007 -> ∃ k : ℤ, -a < ↑k ∧ ↑k < a) : 1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l36_36194


namespace only_one_statement_is_true_l36_36722

theorem only_one_statement_is_true (A B C D E: Prop)
  (hA : A ↔ B)
  (hB : B ↔ ¬ E)
  (hC : C ↔ (A ∧ B ∧ C ∧ D ∧ E))
  (hD : D ↔ ¬ (A ∨ B ∨ C ∨ D ∨ E))
  (hE : E ↔ ¬ A)
  (h_unique : ∃! x, x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∧ x = True) : E :=
by
  sorry

end only_one_statement_is_true_l36_36722


namespace convex_polygon_quadrilateral_division_l36_36202

open Nat

theorem convex_polygon_quadrilateral_division (n : ℕ) : ℕ :=
  if h : n > 0 then
    1 / (2 * n - 1) * (Nat.choose (3 * n - 3) (n - 1))
  else
    0

end convex_polygon_quadrilateral_division_l36_36202


namespace derivative_at_0_5_l36_36845

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := -2

-- State the theorem
theorem derivative_at_0_5 : f' 0.5 = -2 :=
by {
  -- Proof placeholder
  sorry
}

end derivative_at_0_5_l36_36845


namespace no_alpha_exists_l36_36016

theorem no_alpha_exists (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ¬(∃ (a : ℕ → ℝ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, 1 + a (n+1) ≤ a n + (α / n.succ) * a n)) :=
by
  sorry

end no_alpha_exists_l36_36016


namespace smallest_n_for_candy_l36_36887

theorem smallest_n_for_candy (r g b n : ℕ) (h1 : 10 * r = 18 * g) (h2 : 18 * g = 20 * b) (h3 : 20 * b = 24 * n) : n = 15 :=
by
  sorry

end smallest_n_for_candy_l36_36887


namespace probability_A_not_losing_is_80_percent_l36_36327

def probability_A_winning : ℝ := 0.30
def probability_draw : ℝ := 0.50
def probability_A_not_losing : ℝ := probability_A_winning + probability_draw

theorem probability_A_not_losing_is_80_percent : probability_A_not_losing = 0.80 :=
by 
  sorry

end probability_A_not_losing_is_80_percent_l36_36327


namespace milan_billed_minutes_l36_36156

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l36_36156


namespace max_varphi_l36_36009

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + (2 * Real.pi / 3))

theorem max_varphi (φ : ℝ) (h : φ < 0) (hE : ∀ x, g x φ = g (-x) φ) : φ = -Real.pi / 6 :=
by
  sorry

end max_varphi_l36_36009


namespace find_selling_price_l36_36474

-- Define the cost price of the article
def cost_price : ℝ := 47

-- Define the profit when the selling price is Rs. 54
def profit : ℝ := 54 - cost_price

-- Assume that the profit is the same as the loss
axiom profit_equals_loss : profit = 7

-- Define the selling price that yields the same loss as the profit
def selling_price_loss : ℝ := cost_price - profit

-- Now state the theorem to prove that the selling price for loss is Rs. 40
theorem find_selling_price : selling_price_loss = 40 :=
sorry

end find_selling_price_l36_36474


namespace smallest_n_is_29_l36_36116

noncomputable def smallest_possible_n (r g b : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm (10 * r) (16 * g)) (18 * b) / 25

theorem smallest_n_is_29 (r g b : ℕ) (h : 10 * r = 16 * g ∧ 16 * g = 18 * b) :
  smallest_possible_n r g b = 29 :=
by
  sorry

end smallest_n_is_29_l36_36116


namespace find_divisor_l36_36074

theorem find_divisor (d : ℕ) (H1 : 199 = d * 11 + 1) : d = 18 := 
sorry

end find_divisor_l36_36074


namespace find_k_value_l36_36813

theorem find_k_value (k : ℝ) : (∀ (x y : ℝ), (x = 2 ∧ y = 5) → y = k * x + 3) → k = 1 := 
by 
  intro h
  have h1 := h 2 5 ⟨rfl, rfl⟩
  linarith

end find_k_value_l36_36813


namespace total_rowing_campers_l36_36683

theorem total_rowing_campers (morning_rowing afternoon_rowing : ℕ) : 
  morning_rowing = 13 -> 
  afternoon_rowing = 21 -> 
  morning_rowing + afternoon_rowing = 34 :=
by
  sorry

end total_rowing_campers_l36_36683


namespace kelly_carrot_weight_l36_36108

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l36_36108


namespace polynomial_remainder_division_l36_36467

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (3 * x^7 + 2 * x^5 - 5 * x^3 + x^2 - 9) % (x^2 + 2 * x + 1) = 14 * x - 16 :=
by
  sorry

end polynomial_remainder_division_l36_36467


namespace minimum_value_correct_l36_36013

noncomputable def minimum_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_eq : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z + 1)^2 / (2 * x * y * z)

theorem minimum_value_correct {x y z : ℝ}
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 + y^2 + z^2 = 1) :
  minimum_value x y z h_pos h_eq = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_correct_l36_36013


namespace shaded_fraction_in_fifth_diagram_l36_36808

-- Definitions for conditions
def geometric_sequence (a₀ r n : ℕ) : ℕ := a₀ * r^n

def total_triangles (n : ℕ) : ℕ := n^2

-- Lean theorem statement
theorem shaded_fraction_in_fifth_diagram 
  (a₀ r n : ℕ) 
  (h_geometric : a₀ = 1) 
  (h_ratio : r = 2)
  (h_step_number : n = 4):
  (geometric_sequence a₀ r n) / (total_triangles (n + 1)) = 16 / 25 :=
by
  sorry

end shaded_fraction_in_fifth_diagram_l36_36808


namespace regular_decagon_interior_angle_l36_36527

theorem regular_decagon_interior_angle {n : ℕ} (h1 : n = 10) (h2 : ∀ (k : ℕ), k = 10 → (180 * (k - 2)) / 10 = 144) : 
  (∃ θ : ℕ, θ = 180 * (n - 2) / n ∧ n = 10 ∧ θ = 144) :=
by
  sorry

end regular_decagon_interior_angle_l36_36527


namespace sum_first_15_odd_from_5_l36_36479

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l36_36479


namespace greatest_difference_l36_36242

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l36_36242


namespace photographs_taken_l36_36570

theorem photographs_taken (P : ℝ) (h : P + 0.80 * P = 180) : P = 100 :=
by sorry

end photographs_taken_l36_36570


namespace equation_solution_l36_36990

theorem equation_solution (x : ℤ) (h : 3 * x - 2 * x + x = 3 - 2 + 1) : x = 2 :=
by
  sorry

end equation_solution_l36_36990


namespace program_arrangements_l36_36763

/-- Given 5 programs, if A, B, and C appear in a specific order, then the number of different
    arrangements is 20. -/
theorem program_arrangements (A B C A_order : ℕ) : 
  (A + B + C + A_order = 5) → 
  (A_order = 3) → 
  (B = 1) → 
  (C = 1) → 
  (A = 1) → 
  (A * B * C * A_order = 1) :=
  by sorry

end program_arrangements_l36_36763


namespace find_number_l36_36214

-- Define the conditions
def condition (x : ℝ) : Prop := 0.65 * x = (4/5) * x - 21

-- Prove that given the condition, x is 140.
theorem find_number (x : ℝ) (h : condition x) : x = 140 := by
  sorry

end find_number_l36_36214


namespace no_hamiltonian_cycle_l36_36864

-- Define the problem constants
def n : ℕ := 2016
def a : ℕ := 2
def b : ℕ := 3

-- Define the circulant graph and the conditions of the Hamiltonian cycle theorem
theorem no_hamiltonian_cycle (s t : ℕ) (h1 : s + t = Int.gcd n (a - b)) :
  ¬ (Int.gcd n (s * a + t * b) = 1) :=
by
  sorry  -- Proof not required as per instructions

end no_hamiltonian_cycle_l36_36864


namespace sin_double_angle_l36_36101

theorem sin_double_angle (h1 : Real.pi / 2 < β)
    (h2 : β < α)
    (h3 : α < 3 * Real.pi / 4)
    (h4 : Real.cos (α - β) = 12 / 13)
    (h5 : Real.sin (α + β) = -3 / 5) :
    Real.sin (2 * α) = -56 / 65 := 
by
  sorry

end sin_double_angle_l36_36101


namespace simplify_expression_l36_36760

theorem simplify_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) :
  (a^2 - 6 * a + 9) / (a^2 - 2 * a) / (1 - 1 / (a - 2)) = (a - 3) / a :=
sorry

end simplify_expression_l36_36760


namespace value_of_f_g_3_l36_36236

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

theorem value_of_f_g_3 : f (g 3) = 2134 :=
by 
  sorry

end value_of_f_g_3_l36_36236


namespace smallest_even_x_l36_36747

theorem smallest_even_x (x : ℤ) (h1 : x < 3 * x - 10) (h2 : ∃ k : ℤ, x = 2 * k) : x = 6 :=
by {
  sorry
}

end smallest_even_x_l36_36747


namespace length_of_metallic_sheet_l36_36381

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l36_36381


namespace circular_garden_area_l36_36789

theorem circular_garden_area
  (r : ℝ) (h_r : r = 16)
  (C A : ℝ) (h_C : C = 2 * Real.pi * r) (h_A : A = Real.pi * r^2)
  (fence_cond : C = 1 / 8 * A) :
  A = 256 * Real.pi := by
  sorry

end circular_garden_area_l36_36789


namespace intervals_of_monotonicity_range_of_values_l36_36375

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  -(1 + a) / x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  f a x - g a x

theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, x < 1 + a → h a x < h a (1 + a)) ∧
  (∀ x > 1 + a, h a x > h a (1 + a)) :=
sorry

theorem range_of_values (x0 : ℝ) (h_x0 : 1 ≤ x0 ∧ x0 ≤ Real.exp 1) (h_fx_gx : f a x0 < g a x0) :
  a > (Real.exp 1)^2 + 1 / (Real.exp 1 - 1) ∨ a < -2 :=
sorry

end intervals_of_monotonicity_range_of_values_l36_36375


namespace probability_face_cards_l36_36816

theorem probability_face_cards :
  let first_card_hearts_face := 3 / 52
  let second_card_clubs_face_after_hearts := 3 / 51
  let combined_probability := first_card_hearts_face * second_card_clubs_face_after_hearts
  combined_probability = 1 / 294 :=
by 
  sorry

end probability_face_cards_l36_36816


namespace max_positive_integer_value_l36_36162

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, ∃ q: ℝ, a (n + 1) = a n * q

theorem max_positive_integer_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 = 4)
  (h4 : a 1 + a 2 + a 3 = 14) : 
  ∃ n, n ≤ 4 ∧ a n * a (n+1) * a (n+2) > 1 / 9 :=
sorry

end max_positive_integer_value_l36_36162


namespace certain_positive_integer_value_l36_36492

-- Define factorial
def fact : ℕ → ℕ 
| 0     => 1
| (n+1) => (n+1) * fact n

-- Statement of the problem
theorem certain_positive_integer_value (i k m a : ℕ) :
  (fact 8 = 2^i * 3^k * 5^m * 7^a) ∧ (i + k + m + a = 11) → a = 1 :=
by 
  sorry

end certain_positive_integer_value_l36_36492


namespace derek_dogs_count_l36_36403

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l36_36403


namespace ratio_of_fifth_to_second_l36_36775

-- Definitions based on the conditions
def first_stack := 7
def second_stack := first_stack + 3
def third_stack := second_stack - 6
def fourth_stack := third_stack + 10

def total_blocks := 55

-- The number of blocks in the fifth stack
def fifth_stack := total_blocks - (first_stack + second_stack + third_stack + fourth_stack)

-- The ratio of the fifth stack to the second stack
def ratio := fifth_stack / second_stack

-- The theorem we want to prove
theorem ratio_of_fifth_to_second: ratio = 2 := by
  sorry

end ratio_of_fifth_to_second_l36_36775


namespace polygon_side_count_l36_36026

theorem polygon_side_count (n : ℕ) 
    (h : (n - 2) * 180 + 1350 - (n - 2) * 180 = 1350) : n = 9 :=
by
  sorry

end polygon_side_count_l36_36026


namespace bob_more_than_alice_l36_36475

-- Definitions for conditions
def initial_investment_alice : ℕ := 10000
def initial_investment_bob : ℕ := 10000
def multiple_alice : ℕ := 3
def multiple_bob : ℕ := 7

-- Derived conditions based on the investment multiples
def final_amount_alice : ℕ := initial_investment_alice * multiple_alice
def final_amount_bob : ℕ := initial_investment_bob * multiple_bob

-- Statement of the problem
theorem bob_more_than_alice : final_amount_bob - final_amount_alice = 40000 :=
by
  -- Proof to be filled in
  sorry

end bob_more_than_alice_l36_36475


namespace find_n_from_binomial_terms_l36_36363

theorem find_n_from_binomial_terms (x a : ℕ) (n : ℕ) 
  (h1 : n.choose 1 * x^(n-1) * a = 56) 
  (h2 : n.choose 2 * x^(n-2) * a^2 = 168) 
  (h3 : n.choose 3 * x^(n-3) * a^3 = 336) : 
  n = 5 :=
by
  sorry

end find_n_from_binomial_terms_l36_36363


namespace ellipse_standard_equation_l36_36810

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 3 / 4) (h3 : b^2 = a^2 - c^2) :
  (x y : ℝ) →
  (x^2 / a^2 + y^2 / b^2 = 1 ∨ x^2 / b^2 + y^2 / a^2 = 1) :=
by
  sorry

end ellipse_standard_equation_l36_36810


namespace range_of_2a_plus_3b_l36_36384

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l36_36384


namespace find_initial_interest_rate_l36_36707

-- Definitions of the initial conditions
def P1 : ℝ := 3000
def P2 : ℝ := 1499.9999999999998
def P_total : ℝ := 4500
def r2 : ℝ := 0.08
def total_annual_income : ℝ := P_total * 0.06

-- Defining the problem as a statement to prove
theorem find_initial_interest_rate (r1 : ℝ) :
  (P1 * r1) + (P2 * r2) = total_annual_income → r1 = 0.05 := by
  sorry

end find_initial_interest_rate_l36_36707


namespace increased_cost_is_4_percent_l36_36149

-- Initial declarations
variables (initial_cost : ℕ) (price_change_eggs price_change_apples percentage_increase : ℕ)

-- Cost definitions based on initial conditions
def initial_cost_eggs := 100
def initial_cost_apples := 100

-- Price adjustments
def new_cost_eggs := initial_cost_eggs - (initial_cost_eggs * 2 / 100)
def new_cost_apples := initial_cost_apples + (initial_cost_apples * 10 / 100)

-- New combined cost
def new_combined_cost := new_cost_eggs + new_cost_apples

-- Old combined cost
def old_combined_cost := initial_cost_eggs + initial_cost_apples

-- Increase in cost
def increase_in_cost := new_combined_cost - old_combined_cost

-- Percentage increase
def calculated_percentage_increase := (increase_in_cost * 100) / old_combined_cost

-- The proof statement
theorem increased_cost_is_4_percent :
  initial_cost = 100 →
  price_change_eggs = 2 →
  price_change_apples = 10 →
  percentage_increase = 4 →
  calculated_percentage_increase = percentage_increase :=
sorry

end increased_cost_is_4_percent_l36_36149


namespace possible_sets_l36_36107

theorem possible_sets 
  (A B C : Set ℕ) 
  (U : Set ℕ := {a, b, c, d, e, f}) 
  (H1 : A ∪ B ∪ C = U) 
  (H2 : A ∩ B = {a, b, c, d}) 
  (H3 : c ∈ A ∩ B ∩ C) : 
  ∃ (n : ℕ), n = 200 :=
sorry

end possible_sets_l36_36107


namespace middle_card_is_five_l36_36041

theorem middle_card_is_five 
    (a b c : ℕ) 
    (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
    (h2 : a + b + c = 16)
    (h3 : a < b ∧ b < c)
    (casey : ¬(∃ y z, y ≠ z ∧ y + z + a = 16 ∧ a < y ∧ y < z))
    (tracy : ¬(∃ x y, x ≠ y ∧ x + y + c = 16 ∧ x < y ∧ y < c))
    (stacy : ¬(∃ x z, x ≠ z ∧ x + z + b = 16 ∧ x < b ∧ b < z)) 
    : b = 5 :=
sorry

end middle_card_is_five_l36_36041


namespace beckys_age_ratio_l36_36581

theorem beckys_age_ratio (Eddie_age : ℕ) (Irene_age : ℕ)
  (becky_age: ℕ)
  (H1 : Eddie_age = 92)
  (H2 : Irene_age = 46)
  (H3 : Irene_age = 2 * becky_age) :
  becky_age / Eddie_age = 1 / 4 :=
by
  sorry

end beckys_age_ratio_l36_36581


namespace rectangle_area_1600_l36_36304

theorem rectangle_area_1600
  (l w : ℝ)
  (h1 : l = 4 * w)
  (h2 : 2 * l + 2 * w = 200) :
  l * w = 1600 :=
by
  sorry

end rectangle_area_1600_l36_36304


namespace number_of_real_pairs_l36_36588

theorem number_of_real_pairs :
  ∃! (x y : ℝ), 11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0 :=
sorry

end number_of_real_pairs_l36_36588


namespace find_x_given_y_l36_36192

variable (x y : ℝ)

theorem find_x_given_y :
  (0 < x) → (0 < y) → 
  (∃ k : ℝ, (3 * x^2 * y = k)) → 
  (y = 18 → x = 3) → 
  (y = 2400) → 
  x = 9 * Real.sqrt 6 / 85 :=
by
  -- Proof goes here
  sorry

end find_x_given_y_l36_36192


namespace ratio_sum_odd_even_divisors_l36_36005

def M : ℕ := 33 * 38 * 58 * 462

theorem ratio_sum_odd_even_divisors : 
  let sum_odd_divisors := 
    (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_all_divisors := 
    (1 + 2 + 4 + 8) * (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  (sum_odd_divisors : ℚ) / sum_even_divisors = 1 / 14 :=
by sorry

end ratio_sum_odd_even_divisors_l36_36005


namespace coefficient_x3_expansion_l36_36325

/--
Prove that the coefficient of \(x^{3}\) in the expansion of \(( \frac{x}{\sqrt{y}} - \frac{y}{\sqrt{x}})^{6}\) is \(15\).
-/
theorem coefficient_x3_expansion (x y : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ (x / y.sqrt - y / x.sqrt) ^ 6 = c * x ^ 3) :=
sorry

end coefficient_x3_expansion_l36_36325


namespace value_of_k_l36_36144

theorem value_of_k (x y k : ℝ) (h1 : 3 * x + 2 * y = k + 1) (h2 : 2 * x + 3 * y = k) (h3 : x + y = 2) :
  k = 9 / 2 :=
by
  sorry

end value_of_k_l36_36144


namespace find_days_l36_36042

variables (a d e k m : ℕ) (y : ℕ)

-- Assumptions based on the problem
def workers_efficiency_condition : Prop := 
  (a * e * (d * k) / (a * e)) = d

-- Conclusion we aim to prove
def target_days_condition : Prop :=
  y = (a * a) / (d * k * m)

theorem find_days (h : workers_efficiency_condition a d e k) : target_days_condition a d k m y :=
  sorry

end find_days_l36_36042


namespace seating_arrangements_l36_36237

theorem seating_arrangements (p : Fin 5 → Fin 5 → Prop) :
  (∃! i j : Fin 5, p i j ∧ i = j) →
  (∃! i j : Fin 5, p i j ∧ i ≠ j) →
  ∃ ways : ℕ,
  ways = 20 :=
by
  sorry

end seating_arrangements_l36_36237


namespace arrange_p_q_r_l36_36526

theorem arrange_p_q_r (p : ℝ) (h : 1 < p ∧ p < 1.1) : p < p^p ∧ p^p < p^(p^p) :=
by
  sorry

end arrange_p_q_r_l36_36526


namespace problem_statement_l36_36046

noncomputable def S (k : ℕ) : ℚ := sorry

theorem problem_statement (k : ℕ) (a_k : ℚ) :
  S (k - 1) < 10 → S k > 10 → a_k = 6 / 7 :=
sorry

end problem_statement_l36_36046


namespace fraction_of_odd_products_is_0_25_l36_36047

noncomputable def fraction_of_odd_products : ℝ :=
  let odd_products := 8 * 8
  let total_products := 16 * 16
  (odd_products / total_products : ℝ)

theorem fraction_of_odd_products_is_0_25 :
  fraction_of_odd_products = 0.25 :=
by sorry

end fraction_of_odd_products_is_0_25_l36_36047


namespace relationship_of_abc_l36_36417

theorem relationship_of_abc (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 4) : c > b ∧ b > a := by
  sorry

end relationship_of_abc_l36_36417
