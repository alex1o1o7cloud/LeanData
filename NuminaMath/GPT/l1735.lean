import Mathlib

namespace delta_value_l1735_173581

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end delta_value_l1735_173581


namespace drawings_in_five_pages_l1735_173552

theorem drawings_in_five_pages :
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  a₁ + a₂ + a₃ + a₄ + a₅ = 155 :=
by
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  sorry

end drawings_in_five_pages_l1735_173552


namespace Jamie_correct_percentage_l1735_173536

theorem Jamie_correct_percentage (y : ℕ) : ((8 * y - 2 * y : ℕ) / (8 * y : ℕ) : ℚ) * 100 = 75 := by
  sorry

end Jamie_correct_percentage_l1735_173536


namespace students_taking_geometry_or_science_but_not_both_l1735_173533

def students_taking_both : ℕ := 15
def students_taking_geometry : ℕ := 30
def students_taking_science_only : ℕ := 18

theorem students_taking_geometry_or_science_but_not_both : students_taking_geometry - students_taking_both + students_taking_science_only = 33 := by
  sorry

end students_taking_geometry_or_science_but_not_both_l1735_173533


namespace unique_zero_point_condition1_unique_zero_point_condition2_l1735_173514

noncomputable def func (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem unique_zero_point_condition1 {a b : ℝ} (h1 : 1 / 2 < a) (h2 : a ≤ Real.exp 2 / 2) (h3 : b > 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

theorem unique_zero_point_condition2 {a b : ℝ} (h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

end unique_zero_point_condition1_unique_zero_point_condition2_l1735_173514


namespace number_of_people_l1735_173513

theorem number_of_people (total_cookies : ℕ) (cookies_per_person : ℝ) (h1 : total_cookies = 144) (h2 : cookies_per_person = 24.0) : total_cookies / cookies_per_person = 6 := 
by 
  -- Placeholder for actual proof.
  sorry

end number_of_people_l1735_173513


namespace grant_received_money_l1735_173548

theorem grant_received_money :
  let total_teeth := 20
  let lost_teeth := 2
  let first_tooth_amount := 20
  let other_tooth_amount_per_tooth := 2
  let remaining_teeth := total_teeth - lost_teeth - 1
  let total_amount_received := first_tooth_amount + remaining_teeth * other_tooth_amount_per_tooth
  total_amount_received = 54 :=
by  -- Start the proof mode
  sorry  -- This is where the actual proof would go

end grant_received_money_l1735_173548


namespace statement1_statement2_statement3_statement4_statement5_statement6_l1735_173542

/-
Correct syntax statements in pseudo code
-/

def correct_assignment1 (A B : ℤ) : Prop :=
  B = A ∧ A = 50

def correct_assignment2 (x y z : ℕ) : Prop :=
  x = 1 ∧ y = 2 ∧ z = 3

def correct_input1 (s : String) (x : ℕ) : Prop :=
  s = "How old are you?" ∧ x ≥ 0

def correct_input2 (x : ℕ) : Prop :=
  x ≥ 0

def correct_print1 (s1 : String) (C : ℤ) : Prop :=
  s1 = "A+B=" ∧ C < 100  -- additional arbitrary condition for C

def correct_print2 (s2 : String) : Prop :=
  s2 = "Good-bye!"

theorem statement1 (A : ℤ) : ∃ B, correct_assignment1 A B :=
sorry

theorem statement2 : ∃ (x y z : ℕ), correct_assignment2 x y z :=
sorry

theorem statement3 (x : ℕ) : ∃ s, correct_input1 s x :=
sorry

theorem statement4 (x : ℕ) : correct_input2 x :=
sorry

theorem statement5 (C : ℤ) : ∃ s1, correct_print1 s1 C :=
sorry

theorem statement6 : ∃ s2, correct_print2 s2 :=
sorry

end statement1_statement2_statement3_statement4_statement5_statement6_l1735_173542


namespace max_even_a_exists_max_even_a_l1735_173509

theorem max_even_a (a : ℤ): (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k) → a ≤ 8 := sorry

theorem exists_max_even_a : ∃ a : ℤ, (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k ∧ a = 8) := sorry

end max_even_a_exists_max_even_a_l1735_173509


namespace quadratic_roots_expression_l1735_173538

theorem quadratic_roots_expression :
  ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 3) →
  (x₁ * x₂ = -1) →
  (x₁^2 * x₂ + x₁ * x₂^2 = -3) :=
by
  intros x₁ x₂ h1 h2
  sorry

end quadratic_roots_expression_l1735_173538


namespace propositions_correctness_l1735_173530

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def P : Prop := ∃ x : ℝ, x^2 - x - 1 > 0
def negP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

theorem propositions_correctness :
    (∀ a, a ∈ M → a ∈ N) = false ∧
    (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
    (∀ p q, ¬(p ∧ q) → ¬p ∧ ¬q) = false ∧ 
    (¬P ↔ negP) :=
by
  sorry

end propositions_correctness_l1735_173530


namespace barbeck_steve_guitar_ratio_l1735_173507

theorem barbeck_steve_guitar_ratio (b s d : ℕ) 
  (h1 : b = s) 
  (h2 : d = 3 * b) 
  (h3 : b + s + d = 27) 
  (h4 : d = 18) : 
  b / s = 2 / 1 := 
by 
  sorry

end barbeck_steve_guitar_ratio_l1735_173507


namespace hyperbola_center_l1735_173539

theorem hyperbola_center :
  ∃ (h : ℝ × ℝ), h = (9 / 2, 2) ∧
  (∃ (x y : ℝ), 9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0) :=
  sorry

end hyperbola_center_l1735_173539


namespace natural_numbers_condition_l1735_173518

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_numbers_condition (n : ℕ) (p1 p2 : ℕ)
  (hp1_prime : is_prime p1) (hp2_prime : is_prime p2)
  (hn : n = p1 ^ 2) (hn72 : n + 72 = p2 ^ 2) :
  n = 49 ∨ n = 289 :=
  sorry

end natural_numbers_condition_l1735_173518


namespace june_spent_on_music_books_l1735_173571

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end june_spent_on_music_books_l1735_173571


namespace part1_part2_part3_l1735_173588

-- Conditions
def A : Set ℝ := { x : ℝ | 2 < x ∧ x < 6 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m }

-- Proof statements
theorem part1 : A ∪ B 2 = { x : ℝ | 2 < x ∧ x < 6 } := by
  sorry

theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) → m ≤ 3 := by
  sorry

theorem part3 (m : ℝ) : (∃ x, x ∈ B m) ∧ (∀ x, x ∉ A ∩ B m) → m ≥ 5 := by
  sorry

end part1_part2_part3_l1735_173588


namespace Theresa_helper_hours_l1735_173592

theorem Theresa_helper_hours :
  ∃ x : ℕ, (7 + 10 + 8 + 11 + 9 + 7 + x) / 7 = 9 ∧ x ≥ 10 := by
  sorry

end Theresa_helper_hours_l1735_173592


namespace quadrilateral_divided_similarity_iff_trapezoid_l1735_173532

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry
noncomputable def is_trapezoid (A B C D : Type) : Prop := sorry
noncomputable def similar_quadrilaterals (E F A B C D : Type) : Prop := sorry

theorem quadrilateral_divided_similarity_iff_trapezoid {A B C D E F : Type}
  (h1 : convex_quadrilateral A B C D)
  (h2 : similar_quadrilaterals E F A B C D): 
  is_trapezoid A B C D ↔ similar_quadrilaterals E F A B C D :=
sorry

end quadrilateral_divided_similarity_iff_trapezoid_l1735_173532


namespace dot_product_vec1_vec2_l1735_173531

-- Define the vectors
def vec1 := (⟨-4, -1⟩ : ℤ × ℤ)
def vec2 := (⟨6, 8⟩ : ℤ × ℤ)

-- Define the dot product function
def dot_product (v1 v2 : ℤ × ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of vec1 and vec2 is -32
theorem dot_product_vec1_vec2 : dot_product vec1 vec2 = -32 :=
by
  sorry

end dot_product_vec1_vec2_l1735_173531


namespace usual_time_eq_three_l1735_173594

variable (S T : ℝ)
variable (usual_speed : S > 0)
variable (usual_time : T > 0)
variable (reduced_speed : S' = 6/7 * S)
variable (reduced_time : T' = T + 0.5)

theorem usual_time_eq_three (h : 7/6 = T' / T) : T = 3 :=
by
  -- proof to be filled in
  sorry

end usual_time_eq_three_l1735_173594


namespace oak_trees_in_park_l1735_173560

theorem oak_trees_in_park (planting_today : ℕ) (total_trees : ℕ) 
  (h1 : planting_today = 4) (h2 : total_trees = 9) : 
  total_trees - planting_today = 5 :=
by
  -- proof goes here
  sorry

end oak_trees_in_park_l1735_173560


namespace abs_not_eq_three_implies_x_not_eq_three_l1735_173549

theorem abs_not_eq_three_implies_x_not_eq_three (x : ℝ) (h : |x| ≠ 3) : x ≠ 3 :=
sorry

end abs_not_eq_three_implies_x_not_eq_three_l1735_173549


namespace vehicle_count_l1735_173526

theorem vehicle_count (T B : ℕ) (h1 : T + B = 15) (h2 : 3 * T + 2 * B = 40) : T = 10 ∧ B = 5 :=
by
  sorry

end vehicle_count_l1735_173526


namespace xy_condition_l1735_173579

theorem xy_condition (x y : ℝ) (h : x * y + x / y + y / x = -3) : (x - 2) * (y - 2) = 3 :=
sorry

end xy_condition_l1735_173579


namespace johns_photo_world_sitting_fee_l1735_173598

variable (J : ℝ)

theorem johns_photo_world_sitting_fee
  (h1 : ∀ n : ℝ, n = 12 → 2.75 * n + J = 1.50 * n + 140) : J = 125 :=
by
  -- We will skip the proof since it is not required by the problem statement.
  sorry

end johns_photo_world_sitting_fee_l1735_173598


namespace common_chord_length_l1735_173593

theorem common_chord_length (r d : ℝ) (hr : r = 12) (hd : d = 16) : 
  ∃ l : ℝ, l = 8 * Real.sqrt 5 := 
by
  sorry

end common_chord_length_l1735_173593


namespace percentage_less_than_l1735_173551

theorem percentage_less_than (x y : ℝ) (h : y = 1.80 * x) : (x / y) * 100 = 100 - 44.44 :=
by
  sorry

end percentage_less_than_l1735_173551


namespace largest_x_value_l1735_173556

-- Definition of the equation
def equation (x : ℚ) : Prop := 3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)

-- The problem to prove is that the largest value of x satisfying the equation is -1/2
theorem largest_x_value : ∃ x : ℚ, equation x ∧ ∀ y : ℚ, equation y → y ≤ -1/2 := by
  sorry

end largest_x_value_l1735_173556


namespace unique_solution_l1735_173583

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (f x)) + f (f y) = f y + x

-- Define the main theorem
theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y, functional_eq f x y) → (∀ x, f x = x) :=
by
  intros h x
  -- Proof steps would go here
  sorry

end unique_solution_l1735_173583


namespace wrapping_paper_amount_l1735_173567

theorem wrapping_paper_amount (x : ℝ) (h : x + (3/4) * x + (x + (3/4) * x) = 7) : x = 2 :=
by
  sorry

end wrapping_paper_amount_l1735_173567


namespace min_sum_of_factors_l1735_173553

theorem min_sum_of_factors (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 1806) :
  x + y + z ≥ 72 := 
sorry

end min_sum_of_factors_l1735_173553


namespace proof_problem_l1735_173500

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x)

theorem proof_problem
  (a : ℝ)
  (h1 : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2))
  (h2 : f (-1/4) = a) :
  f (9/4) = -a :=
by sorry

end proof_problem_l1735_173500


namespace range_of_m_l1735_173512

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (m * x^2 + (m - 3) * x + 1 = 0)) →
  m ∈ Set.Iic 1 := by
  sorry

end range_of_m_l1735_173512


namespace non_deg_ellipse_b_l1735_173528

theorem non_deg_ellipse_b (b : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = b ∧ (∀ x y : ℝ, (x - 3)^2 + 9*(y + 3/2)^2 = b + 145/4)) → b > -145/4 :=
sorry

end non_deg_ellipse_b_l1735_173528


namespace max_stamps_without_discount_theorem_l1735_173544

def total_money := 5000
def price_per_stamp := 50
def max_stamps_without_discount := 100

theorem max_stamps_without_discount_theorem :
  price_per_stamp * max_stamps_without_discount ≤ total_money ∧
  ∀ n, n > max_stamps_without_discount → price_per_stamp * n > total_money := by
  sorry

end max_stamps_without_discount_theorem_l1735_173544


namespace collinear_points_on_curve_sum_zero_l1735_173587

theorem collinear_points_on_curve_sum_zero
  {x1 y1 x2 y2 x3 y3 : ℝ}
  (h_curve1 : y1^2 = x1^3)
  (h_curve2 : y2^2 = x2^3)
  (h_curve3 : y3^2 = x3^3)
  (h_collinear : ∃ (a b c k : ℝ), k ≠ 0 ∧ 
    a * x1 + b * y1 + c = 0 ∧
    a * x2 + b * y2 + c = 0 ∧
    a * x3 + b * y3 + c = 0) :
  x1 / y1 + x2 / y2 + x3 / y3 = 0 :=
sorry

end collinear_points_on_curve_sum_zero_l1735_173587


namespace greatest_power_of_2_factor_of_expr_l1735_173576

theorem greatest_power_of_2_factor_of_expr :
  (∃ k, 2 ^ k ∣ 12 ^ 600 - 8 ^ 400 ∧ ∀ m, 2 ^ m ∣ 12 ^ 600 - 8 ^ 400 → m ≤ 1204) :=
sorry

end greatest_power_of_2_factor_of_expr_l1735_173576


namespace sin_pow_cos_pow_eq_l1735_173566

theorem sin_pow_cos_pow_eq (x : ℝ) (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41 / 216 := by
  sorry

end sin_pow_cos_pow_eq_l1735_173566


namespace crackers_per_friend_l1735_173569

theorem crackers_per_friend (Total_crackers Left_crackers Friends : ℕ) (h1 : Total_crackers = 23) (h2 : Left_crackers = 11) (h3 : Friends = 2):
  (Total_crackers - Left_crackers) / Friends = 6 :=
by
  sorry

end crackers_per_friend_l1735_173569


namespace sum_of_three_consecutive_odds_l1735_173534

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l1735_173534


namespace ypsilon_calendar_l1735_173558

theorem ypsilon_calendar (x y z : ℕ) 
  (h1 : 28 * x + 30 * y + 31 * z = 365) : x + y + z = 12 :=
sorry

end ypsilon_calendar_l1735_173558


namespace circle_equation_l1735_173520

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * y = 0

theorem circle_equation
  (x y : ℝ)
  (center_on_y_axis : ∃ r : ℝ, r > 0 ∧ x^2 + (y - r)^2 = r^2)
  (tangent_to_x_axis : ∃ r : ℝ, r > 0 ∧ y = r)
  (passes_through_point : x = 3 ∧ y = 1) :
  equation_of_circle x y :=
by
  sorry

end circle_equation_l1735_173520


namespace total_cost_cardshop_l1735_173537

theorem total_cost_cardshop : 
  let price_A := 1.25
  let price_B := 1.50
  let price_C := 2.25
  let price_D := 2.50
  let discount_10_percent := 0.10
  let discount_15_percent := 0.15
  let sales_tax_rate := 0.06
  let qty_A := 6
  let qty_B := 4
  let qty_C := 10
  let qty_D := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  total_cost = 60.82
:= 
by
  have price_A : ℝ := 1.25
  have price_B : ℝ := 1.50
  have price_C : ℝ := 2.25
  have price_D : ℝ := 2.50
  have discount_10_percent : ℝ := 0.10
  have discount_15_percent : ℝ := 0.15
  have sales_tax_rate : ℝ := 0.06
  have qty_A : ℕ := 6
  have qty_B : ℕ := 4
  have qty_C : ℕ := 10
  have qty_D : ℕ := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  sorry

end total_cost_cardshop_l1735_173537


namespace quadratic_inequalities_solution_l1735_173506

noncomputable def a : Type := sorry
noncomputable def b : Type := sorry
noncomputable def c : Type := sorry

theorem quadratic_inequalities_solution (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ y, cx^2 + bx + a < 0 ↔ -3 < y ∧ y < 1/2 :=
sorry

end quadratic_inequalities_solution_l1735_173506


namespace opposite_of_neg2_l1735_173565

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l1735_173565


namespace perpendicular_line_and_plane_implication_l1735_173545

variable (l m : Line)
variable (α β : Plane)

-- Given conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line l is perpendicular to plane α

def line_in_plane (m : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line m is included in plane α

def line_perpendicular_to_line (l m : Line) : Prop :=
sorry -- Assume this checks if line l is perpendicular to line m

-- Lean statement for the proof problem
theorem perpendicular_line_and_plane_implication
  (h1 : line_perpendicular_to_plane l α)
  (h2 : line_in_plane m α) :
  line_perpendicular_to_line l m :=
sorry

end perpendicular_line_and_plane_implication_l1735_173545


namespace square_table_seats_4_pupils_l1735_173521

-- Define the conditions given in the problem
def num_rectangular_tables := 7
def seats_per_rectangular_table := 10
def total_pupils := 90
def num_square_tables := 5

-- Define what we want to prove
theorem square_table_seats_4_pupils (x : ℕ) :
  total_pupils = num_rectangular_tables * seats_per_rectangular_table + num_square_tables * x →
  x = 4 :=
by
  sorry

end square_table_seats_4_pupils_l1735_173521


namespace eval_expression_l1735_173516

theorem eval_expression (x y : ℕ) (h_x : x = 2001) (h_y : y = 2002) :
  (x^3 - 3*x^2*y + 5*x*y^2 - y^3 - 2) / (x * y) = 1999 :=
  sorry

end eval_expression_l1735_173516


namespace inequality_bounds_l1735_173595

noncomputable def f (a b A B : ℝ) (θ : ℝ) : ℝ :=
  1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)

theorem inequality_bounds (a b A B : ℝ) (h : ∀ θ : ℝ, f a b A B θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_bounds_l1735_173595


namespace num_combinations_l1735_173515

-- The conditions given in the problem.
def num_pencil_types : ℕ := 2
def num_eraser_types : ℕ := 3

-- The theorem to prove.
theorem num_combinations (pencils : ℕ) (erasers : ℕ) (h1 : pencils = num_pencil_types) (h2 : erasers = num_eraser_types) : pencils * erasers = 6 :=
by 
  have hp : pencils = 2 := h1
  have he : erasers = 3 := h2
  cases hp
  cases he
  rfl

end num_combinations_l1735_173515


namespace total_fish_l1735_173535

theorem total_fish (goldfish bluefish : ℕ) (h1 : goldfish = 15) (h2 : bluefish = 7) : goldfish + bluefish = 22 := 
by
  sorry

end total_fish_l1735_173535


namespace firefighter_remaining_money_correct_l1735_173508

noncomputable def firefighter_weekly_earnings : ℕ := 30 * 48
noncomputable def firefighter_monthly_earnings : ℕ := firefighter_weekly_earnings * 4
noncomputable def firefighter_rent_expense : ℕ := firefighter_monthly_earnings / 3
noncomputable def firefighter_food_expense : ℕ := 500
noncomputable def firefighter_tax_expense : ℕ := 1000
noncomputable def firefighter_total_expenses : ℕ := firefighter_rent_expense + firefighter_food_expense + firefighter_tax_expense
noncomputable def firefighter_remaining_money : ℕ := firefighter_monthly_earnings - firefighter_total_expenses

theorem firefighter_remaining_money_correct :
  firefighter_remaining_money = 2340 :=
by 
  rfl

end firefighter_remaining_money_correct_l1735_173508


namespace gamma_suff_not_nec_for_alpha_l1735_173570

variable {α β γ : Prop}

theorem gamma_suff_not_nec_for_alpha
  (h1 : β → α)
  (h2 : γ ↔ β) :
  (γ → α) ∧ (¬(α → γ)) :=
by {
  sorry
}

end gamma_suff_not_nec_for_alpha_l1735_173570


namespace average_words_per_puzzle_l1735_173501

-- Define the conditions
def uses_up_pencil_every_two_weeks : Prop := ∀ (days_used : ℕ), days_used = 14
def words_to_use_up_pencil : ℕ := 1050
def puzzles_completed_per_day : ℕ := 1

-- Problem statement: Prove the average number of words in each crossword puzzle
theorem average_words_per_puzzle :
  (words_to_use_up_pencil / 14 = 75) :=
by
  -- Definitions used directly from the conditions
  sorry

end average_words_per_puzzle_l1735_173501


namespace max_profit_l1735_173582

noncomputable def profit (x : ℝ) : ℝ :=
  20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem max_profit :
  ∃ x : ℝ, 4 ≤ x ∧ x ≤ 12 ∧ 
  (∀ y : ℝ, 4 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧ profit x = 96 * Real.log 6 - 78 :=
by
  sorry

end max_profit_l1735_173582


namespace improper_fraction_decomposition_l1735_173505

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := 
sorry

end improper_fraction_decomposition_l1735_173505


namespace point_coordinates_in_second_quadrant_l1735_173529

theorem point_coordinates_in_second_quadrant (P : ℝ × ℝ)
  (hx : P.1 ≤ 0)
  (hy : P.2 ≥ 0)
  (dist_x_axis : abs P.2 = 3)
  (dist_y_axis : abs P.1 = 10) :
  P = (-10, 3) :=
by
  sorry

end point_coordinates_in_second_quadrant_l1735_173529


namespace part1_part2_l1735_173527

-- Definitions corresponding to the conditions
def angle_A := 35
def angle_B1 := 40
def three_times_angle_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 3 * B ∨ B = 3 * A ∨ C = 3 * A ∨ A = 3 * C ∨ B = 3 * C ∨ C = 3 * B)

-- Part 1: Checking if triangle ABC is a "three times angle triangle".
theorem part1 : three_times_angle_triangle angle_A angle_B1 (180 - angle_A - angle_B1) :=
  sorry

-- Definitions corresponding to the new conditions
def angle_B2 := 60

-- Part 2: Finding the smallest interior angle in triangle ABC.
theorem part2 (angle_A angle_C : ℕ) :
  three_times_angle_triangle angle_A angle_B2 angle_C → (angle_A = 20 ∨ angle_A = 30 ∨ angle_C = 20 ∨ angle_C = 30) :=
  sorry

end part1_part2_l1735_173527


namespace sticks_picked_up_l1735_173555

variable (original_sticks left_sticks picked_sticks : ℕ)

theorem sticks_picked_up :
  original_sticks = 99 → left_sticks = 61 → picked_sticks = original_sticks - left_sticks → picked_sticks = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sticks_picked_up_l1735_173555


namespace cost_of_painting_l1735_173541

def area_of_house : ℕ := 484
def price_per_sqft : ℕ := 20

theorem cost_of_painting : area_of_house * price_per_sqft = 9680 := by
  sorry

end cost_of_painting_l1735_173541


namespace reflect_across_y_axis_l1735_173568

theorem reflect_across_y_axis (x y : ℝ) :
  (x, y) = (1, 2) → (-x, y) = (-1, 2) :=
by
  intro h
  cases h
  sorry

end reflect_across_y_axis_l1735_173568


namespace jane_donuts_l1735_173547

def croissant_cost := 60
def donut_cost := 90
def days := 6

theorem jane_donuts (c d k : ℤ) 
  (h1 : c + d = days)
  (h2 : donut_cost * d + croissant_cost * c = 100 * k + 50) :
  d = 3 :=
sorry

end jane_donuts_l1735_173547


namespace distinct_primes_p_q_r_l1735_173564

theorem distinct_primes_p_q_r (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (eqn : r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q) : p * q * r = 2014 :=
by
  sorry

end distinct_primes_p_q_r_l1735_173564


namespace circle_radius_doubling_l1735_173561

theorem circle_radius_doubling (r : ℝ) : 
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  (new_circumference = 2 * original_circumference) ∧ (new_area = 4 * original_area) :=
by
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  have hc : new_circumference = 2 * original_circumference := by
    sorry
  have ha : new_area = 4 * original_area := by
    sorry
  exact ⟨hc, ha⟩

end circle_radius_doubling_l1735_173561


namespace dive_has_five_judges_l1735_173540

noncomputable def number_of_judges 
  (scores : List ℝ)
  (difficulty : ℝ)
  (point_value : ℝ) : ℕ := sorry

theorem dive_has_five_judges :
  number_of_judges [7.5, 8.0, 9.0, 6.0, 8.8] 3.2 77.76 = 5 :=
by
  sorry

end dive_has_five_judges_l1735_173540


namespace solve_linear_system_l1735_173503

theorem solve_linear_system :
  ∃ x y : ℤ, x + 9773 = 13200 ∧ 2 * x - 3 * y = 1544 ∧ x = 3427 ∧ y = 1770 := by
  sorry

end solve_linear_system_l1735_173503


namespace googoo_total_buttons_l1735_173580

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end googoo_total_buttons_l1735_173580


namespace find_face_value_l1735_173559

-- Define the conditions as variables in Lean
variable (BD TD FV : ℝ)
variable (hBD : BD = 36)
variable (hTD : TD = 30)
variable (hRel : BD = TD + (TD * BD / FV))

-- State the theorem we want to prove
theorem find_face_value (BD TD : ℝ) (FV : ℝ) 
  (hBD : BD = 36) (hTD : TD = 30) (hRel : BD = TD + (TD * BD / FV)) : 
  FV = 180 := 
  sorry

end find_face_value_l1735_173559


namespace intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l1735_173502

theorem intersect_at_point_m_eq_1_3_n_eq_neg_73_9 
  (m : ℚ) (n : ℚ) : 
  (m^2 + 8 + n = 0) ∧ (3*m - 1 = 0) → 
  (m = 1/3 ∧ n = -73/9) := 
by 
  sorry

theorem lines_parallel_pass_through 
  (m : ℚ) (n : ℚ) :
  (m ≠ 0) → (m^2 = 16) ∧ (3*m - 8 + n = 0) → 
  (m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20) :=
by 
  sorry

theorem lines_perpendicular_y_intercept 
  (m : ℚ) (n : ℚ) :
  (m = 0 ∧ 8*(-1) + n = 0) → 
  (m = 0 ∧ n = 8) :=
by 
  sorry

end intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l1735_173502


namespace center_circle_sum_l1735_173557

theorem center_circle_sum (h k : ℝ) :
  (∃ h k : ℝ, h + k = 6 ∧ ∃ R, (x - h)^2 + (y - k)^2 = R^2) ↔ ∃ h k : ℝ, h = 3 ∧ k = 3 ∧ h + k = 6 := 
by
  sorry

end center_circle_sum_l1735_173557


namespace overtime_hours_l1735_173596

theorem overtime_hours (x y : ℕ) 
  (h1 : 60 * x + 90 * y = 3240) 
  (h2 : x + y = 50) : 
  y = 8 :=
by
  sorry

end overtime_hours_l1735_173596


namespace find_b_l1735_173510

theorem find_b
  (b : ℝ)
  (h1 : ∃ r : ℝ, 2 * r^2 + b * r - 65 = 0 ∧ r = 5)
  (h2 : 2 * 5^2 + b * 5 - 65 = 0) :
  b = 3 := by
  sorry

end find_b_l1735_173510


namespace burgers_ordered_l1735_173554

theorem burgers_ordered (H : ℕ) (Ht : H + 2 * H = 45) : 2 * H = 30 := by
  sorry

end burgers_ordered_l1735_173554


namespace kitty_cleaning_weeks_l1735_173590

def time_spent_per_week (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust_furniture: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust_furniture

def total_weeks (total_time: ℕ) (time_per_week: ℕ) : ℕ :=
  total_time / time_per_week

theorem kitty_cleaning_weeks
  (pick_up_time : ℕ := 5)
  (vacuum_time : ℕ := 20)
  (clean_windows_time : ℕ := 15)
  (dust_furniture_time : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  : total_weeks total_cleaning_time (time_spent_per_week pick_up_time vacuum_time clean_windows_time dust_furniture_time) = 4 :=
by
  sorry

end kitty_cleaning_weeks_l1735_173590


namespace perpendicular_lines_k_value_l1735_173519

theorem perpendicular_lines_k_value (k : ℝ) : 
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0 ↔ k = -3 ∨ k = 1 :=
by
  sorry

end perpendicular_lines_k_value_l1735_173519


namespace new_ratio_boarders_to_day_students_l1735_173573

-- Given conditions
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def initial_boarders : ℕ := 120
def new_boarders : ℕ := 30

-- Derived definitions
def initial_day_students : ℕ :=
  (initial_boarders * (5 : ℕ)) / 2

def total_boarders : ℕ := initial_boarders + new_boarders
def total_day_students : ℕ := initial_day_students

-- Theorem to prove the new ratio
theorem new_ratio_boarders_to_day_students : total_boarders / total_day_students = 1 / 2 :=
  sorry

end new_ratio_boarders_to_day_students_l1735_173573


namespace blue_to_red_face_area_ratio_l1735_173572

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l1735_173572


namespace time_after_2345_minutes_l1735_173546

-- Define the constants
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24
def startTime : Nat := 0 -- midnight on January 1, 2022, treated as 0 minutes.

-- Prove the equivalent time after 2345 minutes
theorem time_after_2345_minutes :
    let totalMinutes := 2345
    let totalHours := totalMinutes / minutesInHour
    let remainingMinutes := totalMinutes % minutesInHour
    let totalDays := totalHours / hoursInDay
    let remainingHours := totalHours % hoursInDay
    startTime + totalDays * hoursInDay * minutesInHour + remainingHours * minutesInHour + remainingMinutes = startTime + 1 * hoursInDay * minutesInHour + 15 * minutesInHour + 5 :=
    by
    sorry

end time_after_2345_minutes_l1735_173546


namespace largest_possible_expression_value_l1735_173504

-- Definition of the conditions.
def distinct_digits (X Y Z : ℕ) : Prop := X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- The main theorem statement.
theorem largest_possible_expression_value : ∀ (X Y Z : ℕ), distinct_digits X Y Z → 
  (100 * X + 10 * Y + Z - 10 * Z - Y - X) ≤ 900 :=
by
  sorry

end largest_possible_expression_value_l1735_173504


namespace rabbit_speed_l1735_173524

theorem rabbit_speed (x : ℕ) :
  2 * (2 * x + 4) = 188 → x = 45 := by
  sorry

end rabbit_speed_l1735_173524


namespace least_positive_number_of_linear_combination_of_24_20_l1735_173563

-- Define the conditions as integers
def problem_statement (x y : ℤ) : Prop := 24 * x + 20 * y = 4

theorem least_positive_number_of_linear_combination_of_24_20 :
  ∃ (x y : ℤ), (24 * x + 20 * y = 4) := 
by
  sorry

end least_positive_number_of_linear_combination_of_24_20_l1735_173563


namespace count_of_sequence_l1735_173589

theorem count_of_sequence : 
  let a := 156
  let d := -6
  let final_term := 36
  (∃ n, a + (n - 1) * d = final_term) -> n = 21 := 
by
  sorry

end count_of_sequence_l1735_173589


namespace total_weight_of_fish_is_correct_l1735_173523

noncomputable def totalWeightInFirstTank := 15 * 0.08 + 12 * 0.05

noncomputable def totalWeightInSecondTank := 2 * 15 * 0.08 + 3 * 12 * 0.05

noncomputable def totalWeightInThirdTank := 3 * 15 * 0.08 + 2 * 12 * 0.05 + 5 * 0.14

noncomputable def totalWeightAllTanks := totalWeightInFirstTank + totalWeightInSecondTank + totalWeightInThirdTank

theorem total_weight_of_fish_is_correct : 
  totalWeightAllTanks = 11.5 :=
by         
  sorry

end total_weight_of_fish_is_correct_l1735_173523


namespace ten_differences_le_100_exists_l1735_173562

theorem ten_differences_le_100_exists (s : Finset ℤ) (h_card : s.card = 101) (h_range : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1000) :
∃ S : Finset ℕ, S.card = 10 ∧ (∀ y ∈ S, y ≤ 100) :=
by {
  sorry
}

end ten_differences_le_100_exists_l1735_173562


namespace problem_statement_l1735_173517

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem problem_statement :
  (M ∩ N) = N :=
by
  sorry

end problem_statement_l1735_173517


namespace evaluate_expression_l1735_173586

variable (a b c d e : ℝ)

-- The equivalent proof problem statement
theorem evaluate_expression 
  (h : (a / b * c - d + e = a / (b * c - d - e))) : 
  a / b * c - d + e = a / (b * c - d - e) :=
by 
  exact h

-- Placeholder for the proof
#check evaluate_expression

end evaluate_expression_l1735_173586


namespace mrs_franklin_gave_38_packs_l1735_173578

-- Define the initial number of Valentines
def initial_valentines : Int := 450

-- Define the remaining Valentines after giving some away
def remaining_valentines : Int := 70

-- Define the size of each pack
def pack_size : Int := 10

-- Define the number of packs given away
def packs_given (initial remaining pack_size : Int) : Int :=
  (initial - remaining) / pack_size

theorem mrs_franklin_gave_38_packs :
  packs_given 450 70 10 = 38 := sorry

end mrs_franklin_gave_38_packs_l1735_173578


namespace cost_of_bench_eq_150_l1735_173522

theorem cost_of_bench_eq_150 (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
sorry

end cost_of_bench_eq_150_l1735_173522


namespace salary_of_E_l1735_173525

theorem salary_of_E (A B C D E : ℕ) (avg_salary : ℕ) 
  (hA : A = 8000) 
  (hB : B = 5000) 
  (hC : C = 11000) 
  (hD : D = 7000) 
  (h_avg : avg_salary = 8000) 
  (h_total_avg : avg_salary * 5 = A + B + C + D + E) : 
  E = 9000 :=
by {
  sorry
}

end salary_of_E_l1735_173525


namespace snow_probability_at_least_once_l1735_173575

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l1735_173575


namespace min_sum_weights_l1735_173511

theorem min_sum_weights (S : ℕ) (h1 : S > 280) (h2 : S % 70 = 30) : S = 310 :=
sorry

end min_sum_weights_l1735_173511


namespace max_value_of_expr_l1735_173577

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end max_value_of_expr_l1735_173577


namespace chocolates_difference_l1735_173574

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
  (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 :=
by
  sorry

end chocolates_difference_l1735_173574


namespace third_row_number_l1735_173591

-- Define the conditions to fill the grid
def grid (n : Nat) := Fin 4 → Fin 4 → Fin n

-- Ensure each number 1-4 in each cell such that numbers do not repeat
def unique_in_row (g : grid 4) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → g i j1 ≠ g i j2

def unique_in_col (g : grid 4) : Prop :=
  ∀ j i1 i2, i1 ≠ i2 → g i1 j ≠ g i1 j

-- Define the external hints condition, encapsulating the provided hints.
def hints_condition (g : grid 4) : Prop :=
  -- Example placeholders for hint conditions that would be expanded accordingly.
  g 0 0 = 3 ∨ g 0 1 = 2 -- First row hints interpreted constraints
  -- Additional hint conditions to be added accordingly

-- Prove the correct number formed by the numbers in the third row is 4213
theorem third_row_number (g : grid 4) :
  unique_in_row g ∧ unique_in_col g ∧ hints_condition g →
  (g 2 0 = 4 ∧ g 2 1 = 2 ∧ g 2 2 = 1 ∧ g 2 3 = 3) :=
by
  sorry

end third_row_number_l1735_173591


namespace drink_total_amount_l1735_173550

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end drink_total_amount_l1735_173550


namespace find_number_250_l1735_173597

theorem find_number_250 (N : ℤ)
  (h1 : 5 * N = 8 * 156 + 2): N = 250 :=
sorry

end find_number_250_l1735_173597


namespace ellipse_foci_cond_l1735_173585

theorem ellipse_foci_cond (m n : ℝ) (h_cond : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → (m > n ∧ n > 0)) ∧ ((m > n ∧ n > 0) → ∀ x y : ℝ, mx^2 + ny^2 = 1) :=
sorry

end ellipse_foci_cond_l1735_173585


namespace find_theta_l1735_173543

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l1735_173543


namespace number_of_folds_l1735_173599

theorem number_of_folds (n : ℕ) :
  (3 * (8 * 8)) / n = 48 → n = 4 :=
by
  sorry

end number_of_folds_l1735_173599


namespace number_of_birds_flew_up_correct_l1735_173584

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l1735_173584
