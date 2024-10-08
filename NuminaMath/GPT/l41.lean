import Mathlib

namespace standard_polar_representation_l41_41365

theorem standard_polar_representation {r θ : ℝ} (hr : r < 0) (hθ : θ = 5 * Real.pi / 6) :
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ (r', θ') = (5, 11 * Real.pi / 6) := 
by {
  sorry
}

end standard_polar_representation_l41_41365


namespace percentage_difference_l41_41324

theorem percentage_difference (n z x y y_decreased : ℝ)
  (h1 : x = 8 * y)
  (h2 : y = 2 * |z - n|)
  (h3 : z = 1.1 * n)
  (h4 : y_decreased = 0.75 * y) :
  (x - y_decreased) / x * 100 = 90.625 := by
sorry

end percentage_difference_l41_41324


namespace compare_abc_l41_41261

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l41_41261


namespace polynomial_roots_l41_41645

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end polynomial_roots_l41_41645


namespace parallel_perpendicular_implies_perpendicular_l41_41683

-- Definitions of the geometric relationships
variables {Line Plane : Type}
variables (a b : Line) (alpha beta : Plane)

-- Conditions as per the problem statement
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Lean statement of the proof problem
theorem parallel_perpendicular_implies_perpendicular
  (h1 : parallel_line_plane a alpha)
  (h2 : perpendicular_line_plane b alpha) :
  perpendicular_lines a b :=  
sorry

end parallel_perpendicular_implies_perpendicular_l41_41683


namespace range_of_a_l41_41060

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, (2 * a < x ∧ x < a + 5) → (x < 6)) ↔ (1 < a ∧ a < 5) :=
by
  sorry

end range_of_a_l41_41060


namespace soccer_team_wins_l41_41613

theorem soccer_team_wins 
  (total_matches : ℕ)
  (total_points : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ)
  (points_per_loss : ℕ)
  (losses : ℕ)
  (H1 : total_matches = 10)
  (H2 : total_points = 17)
  (H3 : points_per_win = 3)
  (H4 : points_per_draw = 1)
  (H5 : points_per_loss = 0)
  (H6 : losses = 3) : 
  ∃ (wins : ℕ), wins = 5 := 
by
  sorry

end soccer_team_wins_l41_41613


namespace quadratic_discriminant_constraint_l41_41460

theorem quadratic_discriminant_constraint (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4*x1 + c = 0 ∧ x2^2 - 4*x2 + c = 0) ↔ c < 4 := 
by
  sorry

end quadratic_discriminant_constraint_l41_41460


namespace sum_of_x_y_l41_41398

theorem sum_of_x_y (x y : ℝ) (h1 : 3 * x + 2 * y = 10) (h2 : 2 * x + 3 * y = 5) : x + y = 3 := 
by
  sorry

end sum_of_x_y_l41_41398


namespace reciprocals_expression_eq_zero_l41_41550

theorem reciprocals_expression_eq_zero {m n : ℝ} (h : m * n = 1) : (2 * m - 2 / n) * (1 / m + n) = 0 :=
by
  sorry

end reciprocals_expression_eq_zero_l41_41550


namespace total_surface_area_of_new_solid_l41_41253

-- Define the heights of the pieces using the given conditions
def height_A := 1 / 4
def height_B := 1 / 5
def height_C := 1 / 6
def height_D := 1 / 7
def height_E := 1 / 8
def height_F := 1 - (height_A + height_B + height_C + height_D + height_E)

-- Assembling the pieces back in reverse order (F to A), encapsulate the total surface area calculation
theorem total_surface_area_of_new_solid : 
  (2 * (1 : ℝ)) + (2 * (1 * 1 : ℝ)) + (2 * (1 * 1 : ℝ)) = 6 :=
by
  sorry

end total_surface_area_of_new_solid_l41_41253


namespace point_inside_circle_l41_41351

theorem point_inside_circle :
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_squared < radius^2 :=
by
  -- Definitions
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2

  -- Goal
  show distance_squared < radius ^ 2
  
  -- Skip Proof
  sorry

end point_inside_circle_l41_41351


namespace triangle_ABC_area_l41_41846

-- Define the vertices of the triangle
def A := (-4, 0)
def B := (24, 0)
def C := (0, 2)

-- Function to calculate the determinant, used for the area calculation
def det (x1 y1 x2 y2 x3 y3 : ℝ) :=
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

-- Area calculation for triangle given vertices using determinant method
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * |det x1 y1 x2 y2 x3 y3|

-- The goal is to prove that the area of triangle ABC is 14
theorem triangle_ABC_area :
  triangle_area (-4) 0 24 0 0 2 = 14 := sorry

end triangle_ABC_area_l41_41846


namespace find_missing_values_l41_41309

theorem find_missing_values :
  (∃ x y : ℕ, 4 / 5 = 20 / x ∧ 4 / 5 = y / 20 ∧ 4 / 5 = 80 / 100) →
  (x = 25 ∧ y = 16 ∧ 4 / 5 = 80 / 100) :=
by
  sorry

end find_missing_values_l41_41309


namespace multiply_inequalities_positive_multiply_inequalities_negative_l41_41570

variable {a b c d : ℝ}

theorem multiply_inequalities_positive (h₁ : a > b) (h₂ : c > d) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) (h₆ : 0 < d) :
  a * c > b * d :=
sorry

theorem multiply_inequalities_negative (h₁ : a < b) (h₂ : c < d) (h₃ : a < 0) (h₄ : b < 0) (h₅ : c < 0) (h₆ : d < 0) :
  a * c > b * d :=
sorry

end multiply_inequalities_positive_multiply_inequalities_negative_l41_41570


namespace percentage_vehicles_updated_2003_l41_41940

theorem percentage_vehicles_updated_2003 (a : ℝ) (h1 : 1.1^4 = 1.46) (h2 : 1.1^5 = 1.61) :
  (a * 1 / (a * 1.61) * 100 = 16.4) :=
  by sorry

end percentage_vehicles_updated_2003_l41_41940


namespace valid_passwords_count_l41_41292

def total_passwords : Nat := 10 ^ 5
def restricted_passwords : Nat := 10

theorem valid_passwords_count : total_passwords - restricted_passwords = 99990 := by
  sorry

end valid_passwords_count_l41_41292


namespace anna_current_age_l41_41264

theorem anna_current_age (A : ℕ) (Clara_now : ℕ) (years_ago : ℕ) (Clara_age_ago : ℕ) 
    (H1 : Clara_now = 80) 
    (H2 : years_ago = 41) 
    (H3 : Clara_age_ago = Clara_now - years_ago) 
    (H4 : Clara_age_ago = 3 * (A - years_ago)) : 
    A = 54 :=
by
  sorry

end anna_current_age_l41_41264


namespace intersection_distance_l41_41868

theorem intersection_distance (p q : ℕ) (h1 : p = 65) (h2 : q = 2) :
  p - q = 63 := 
by
  sorry

end intersection_distance_l41_41868


namespace selling_price_l41_41064

theorem selling_price (cost_price profit_percentage : ℝ) (h_cost : cost_price = 250) (h_profit : profit_percentage = 0.60) :
  cost_price + profit_percentage * cost_price = 400 := sorry

end selling_price_l41_41064


namespace arithmetic_sequence_common_difference_l41_41880

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 1)
  (h3 : a 3 = 11)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = d) : d = 5 :=
sorry

end arithmetic_sequence_common_difference_l41_41880


namespace roots_equation_l41_41038
-- We bring in the necessary Lean libraries

-- Define the conditions as Lean definitions
variable (x1 x2 : ℝ)
variable (h1 : x1^2 + x1 - 3 = 0)
variable (h2 : x2^2 + x2 - 3 = 0)

-- Lean 4 statement we need to prove
theorem roots_equation (x1 x2 : ℝ) (h1 : x1^2 + x1 - 3 = 0) (h2 : x2^2 + x2 - 3 = 0) : 
  x1^3 - 4 * x2^2 + 19 = 0 := 
sorry

end roots_equation_l41_41038


namespace quadratic_unique_real_root_l41_41564

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l41_41564


namespace find_quotient_l41_41618

-- Define the given conditions
def dividend : ℤ := 144
def divisor : ℤ := 11
def remainder : ℤ := 1

-- Define the quotient logically derived from the given conditions
def quotient : ℤ := dividend / divisor

-- The theorem we need to prove
theorem find_quotient : quotient = 13 := by
  sorry

end find_quotient_l41_41618


namespace nails_needed_l41_41117

-- Define the number of nails needed for each plank
def nails_per_plank : ℕ := 2

-- Define the number of planks used by John
def planks_used : ℕ := 16

-- The total number of nails needed.
theorem nails_needed : (nails_per_plank * planks_used) = 32 :=
by
  -- Our goal is to prove that nails_per_plank * planks_used = 32
  sorry

end nails_needed_l41_41117


namespace rectangle_area_l41_41005

theorem rectangle_area (l : ℝ) (w : ℝ) (h_l : l = 15) (h_ratio : (2 * l + 2 * w) / w = 5) : (l * w) = 150 :=
by
  sorry

end rectangle_area_l41_41005


namespace fraction_product_l41_41045

theorem fraction_product :
  (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 :=
by sorry

end fraction_product_l41_41045


namespace smallest_base_for_101_l41_41344

theorem smallest_base_for_101 : ∃ b : ℕ, b = 10 ∧ b ≤ 101 ∧ 101 < b^2 :=
by
  -- We state the simplest form of the theorem,
  -- then use the answer from the solution step.
  use 10
  sorry

end smallest_base_for_101_l41_41344


namespace pencils_more_than_pens_l41_41208

theorem pencils_more_than_pens (pencils pens : ℕ) (h_ratio : 5 * pencils = 6 * pens) (h_pencils : pencils = 48) : 
  pencils - pens = 8 :=
by
  sorry

end pencils_more_than_pens_l41_41208


namespace count_with_consecutive_ones_l41_41023

noncomputable def countValidIntegers : ℕ := 512
noncomputable def invalidCount : ℕ := 89

theorem count_with_consecutive_ones :
  countValidIntegers - invalidCount = 423 :=
by
  sorry

end count_with_consecutive_ones_l41_41023


namespace value_of_m_l41_41423

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end value_of_m_l41_41423


namespace center_of_circle_l41_41341

theorem center_of_circle : ∃ c : ℝ × ℝ, 
  (∃ r : ℝ, ∀ x y : ℝ, (x - c.1) * (x - c.1) + (y - c.2) * (y - c.2) = r ↔ x^2 + y^2 - 6*x - 2*y - 15 = 0) → c = (3, 1) :=
by 
  sorry

end center_of_circle_l41_41341


namespace circle_equation_l41_41028

/-- Given a circle passing through points P(4, -2) and Q(-1, 3), and with the length of the segment 
intercepted by the circle on the y-axis as 4, prove that the standard equation of the circle
is (x-1)^2 + y^2 = 13 or (x-5)^2 + (y-4)^2 = 37 -/
theorem circle_equation {P Q : ℝ × ℝ} {a b k : ℝ} :
  P = (4, -2) ∧ Q = (-1, 3) ∧ k = 4 →
  (∃ (r : ℝ), (∀ y : ℝ, (b - y)^2 = r^2) ∧
    ((a - 1)^2 + b^2 = 13 ∨ (a - 5)^2 + (b - 4)^2 = 37)
  ) :=
by
  sorry

end circle_equation_l41_41028


namespace train_length_150_m_l41_41164

def speed_in_m_s (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_train (speed_in_m_s : ℕ) (time_s : ℕ) : ℕ :=
  speed_in_m_s * time_s

theorem train_length_150_m (speed_kmh : ℕ) (time_s : ℕ) (speed_m_s : speed_in_m_s speed_kmh = 15) (time_pass_pole : time_s = 10) : length_of_train (speed_in_m_s speed_kmh) time_s = 150 := by
  sorry

end train_length_150_m_l41_41164


namespace num_invalid_d_l41_41870

noncomputable def square_and_triangle_problem (d : ℕ) : Prop :=
  ∃ a b : ℕ, 3 * a - 4 * b = 1989 ∧ a - b = d ∧ b > 0

theorem num_invalid_d : ∀ (d : ℕ), (d ≤ 663) → ¬ square_and_triangle_problem d :=
by {
  sorry
}

end num_invalid_d_l41_41870


namespace price_of_child_ticket_l41_41175

theorem price_of_child_ticket (total_seats : ℕ) (adult_ticket_price : ℕ) (total_revenue : ℕ)
  (child_tickets_sold : ℕ) (child_ticket_price : ℕ) :
  total_seats = 80 →
  adult_ticket_price = 12 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (17 * adult_ticket_price) + (child_tickets_sold * child_ticket_price) = total_revenue →
  child_ticket_price = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_child_ticket_l41_41175


namespace walnut_trees_total_l41_41354

theorem walnut_trees_total : 33 + 44 = 77 :=
by
  sorry

end walnut_trees_total_l41_41354


namespace first_term_geometric_series_l41_41812

variable (a : ℝ)
variable (r : ℝ := 1/4)
variable (S : ℝ := 80)

theorem first_term_geometric_series 
  (h1 : r = 1/4) 
  (h2 : S = 80)
  : a = 60 :=
by 
  sorry

end first_term_geometric_series_l41_41812


namespace amoeba_after_ten_days_l41_41974

def amoeba_count (n : ℕ) : ℕ := 
  3^n

theorem amoeba_after_ten_days : amoeba_count 10 = 59049 := 
by
  -- proof omitted
  sorry

end amoeba_after_ten_days_l41_41974


namespace inequality_solution_ab_l41_41470

theorem inequality_solution_ab (a b : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 4 ↔ |x + a| < b) : a * b = -3 := 
by
  sorry

end inequality_solution_ab_l41_41470


namespace compute_c_plus_d_l41_41228

theorem compute_c_plus_d (c d : ℝ) 
  (h1 : c^3 - 18 * c^2 + 25 * c - 75 = 0) 
  (h2 : 9 * d^3 - 72 * d^2 - 345 * d + 3060 = 0) : 
  c + d = 10 := 
sorry

end compute_c_plus_d_l41_41228


namespace fixed_point_of_line_l41_41717

theorem fixed_point_of_line (m : ℝ) : 
  (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  sorry

end fixed_point_of_line_l41_41717


namespace ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l41_41571

theorem ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : 0 < a * b * c)
  : a * b + b * c + c * a < (Real.sqrt (a * b * c)) / 2 + 1 / 4 := 
sorry

end ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l41_41571


namespace initial_goats_l41_41720

theorem initial_goats (G : ℕ) (h1 : 2 + 3 + G + 3 + 5 + 2 = 21) : G = 4 :=
by
  sorry

end initial_goats_l41_41720


namespace smallest_whole_number_larger_than_sum_l41_41977

noncomputable def mixed_number1 : ℚ := 3 + 2/3
noncomputable def mixed_number2 : ℚ := 4 + 1/4
noncomputable def mixed_number3 : ℚ := 5 + 1/5
noncomputable def mixed_number4 : ℚ := 6 + 1/6
noncomputable def mixed_number5 : ℚ := 7 + 1/7

noncomputable def sum_of_mixed_numbers : ℚ :=
  mixed_number1 + mixed_number2 + mixed_number3 + mixed_number4 + mixed_number5

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℤ, (n : ℚ) > sum_of_mixed_numbers ∧ n = 27 :=
by
  sorry

end smallest_whole_number_larger_than_sum_l41_41977


namespace multiplication_mistake_l41_41823

theorem multiplication_mistake (x : ℕ) (H : 43 * x - 34 * x = 1215) : x = 135 :=
sorry

end multiplication_mistake_l41_41823


namespace isosceles_triangle_side_length_condition_l41_41427

theorem isosceles_triangle_side_length_condition (x y : ℕ) :
    y = x + 1 ∧ 2 * x + y = 16 → (y = 6 → x = 5) :=
by sorry

end isosceles_triangle_side_length_condition_l41_41427


namespace product_is_two_l41_41638

theorem product_is_two : 
  ((10 : ℚ) * (1/5) * 4 * (1/16) * (1/2) * 8 = 2) :=
sorry

end product_is_two_l41_41638


namespace non_participating_members_l41_41410

noncomputable def members := 35
noncomputable def badminton_players := 15
noncomputable def tennis_players := 18
noncomputable def both_players := 3

theorem non_participating_members : 
  members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end non_participating_members_l41_41410


namespace polynomial_subtraction_simplify_l41_41827

open Polynomial

noncomputable def p : Polynomial ℚ := 3 * X^2 + 9 * X - 5
noncomputable def q : Polynomial ℚ := 2 * X^2 + 3 * X - 10
noncomputable def result : Polynomial ℚ := X^2 + 6 * X + 5

theorem polynomial_subtraction_simplify : 
  p - q = result :=
by
  sorry

end polynomial_subtraction_simplify_l41_41827


namespace who_made_statements_and_fate_l41_41955

namespace IvanTsarevichProblem

-- Define the characters and their behaviors
inductive Animal
| Bear : Animal
| Fox : Animal
| Wolf : Animal

def always_true (s : Prop) : Prop := s
def always_false (s : Prop) : Prop := ¬s
def alternates (s1 s2 : Prop) : Prop := s1 ∧ ¬s2

-- Statements made by the animals
def statement1 (save_die : Bool) : Prop := save_die = true
def statement2 (safe_sound_save : Bool) : Prop := safe_sound_save = true
def statement3 (safe_lose : Bool) : Prop := safe_lose = true

-- Analyze truth based on behaviors
noncomputable def belongs_to (a : Animal) (s : Prop) : Prop :=
  match a with
  | Animal.Bear => always_true s
  | Animal.Fox => always_false s
  | Animal.Wolf =>
    match s with
    | ss => alternates (ss = true) (ss = false)

-- Given conditions
axiom h1 : statement1 false -- Fox lies, so "You will save the horse. But you will die." is false
axiom h2 : statement2 false -- Wolf alternates, so "You will stay safe and sound. And you will save the horse." is a mix
axiom h3 : statement3 true  -- Bear tells the truth, so "You will survive. But you will lose the horse." is true

-- Conclusion: Animal who made each statement
theorem who_made_statements_and_fate : 
  belongs_to Animal.Fox (statement1 false) ∧ 
  belongs_to Animal.Wolf (statement2 false) ∧ 
  belongs_to Animal.Bear (statement3 true) ∧ 
  (¬safe_lose) := sorry

end IvanTsarevichProblem

end who_made_statements_and_fate_l41_41955


namespace insurance_not_covered_percentage_l41_41754

noncomputable def insurance_monthly_cost : ℝ := 20
noncomputable def insurance_months : ℝ := 24
noncomputable def procedure_cost : ℝ := 5000
noncomputable def amount_saved : ℝ := 3520

theorem insurance_not_covered_percentage :
  ((procedure_cost - amount_saved - (insurance_monthly_cost * insurance_months)) / procedure_cost) * 100 = 20 :=
by
  sorry

end insurance_not_covered_percentage_l41_41754


namespace total_rainfall_in_january_l41_41798

theorem total_rainfall_in_january 
  (r1 r2 : ℝ)
  (h1 : r2 = 1.5 * r1)
  (h2 : r2 = 18) : 
  r1 + r2 = 30 := by
  sorry

end total_rainfall_in_january_l41_41798


namespace number_of_ordered_pairs_l41_41366

theorem number_of_ordered_pairs : ∃ (s : Finset (ℂ × ℂ)), 
    (∀ (a b : ℂ), (a, b) ∈ s → a^5 * b^3 = 1 ∧ a^9 * b^2 = 1) ∧ 
    s.card = 17 := 
by
  sorry

end number_of_ordered_pairs_l41_41366


namespace walter_zoo_time_l41_41419

theorem walter_zoo_time (S: ℕ) (H1: S + 8 * S + 13 = 130) : S = 13 :=
by sorry

end walter_zoo_time_l41_41419


namespace intersection_of_M_and_N_l41_41104

noncomputable def M : Set ℝ := {x | x - 2 > 0}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {x | x > 2} :=
sorry

end intersection_of_M_and_N_l41_41104


namespace correct_calculation_l41_41989

-- Define the statements for each option
def option_A (a : ℕ) : Prop := (a^2)^3 = a^5
def option_B (a : ℕ) : Prop := a^3 + a^2 = a^6
def option_C (a : ℕ) : Prop := a^6 / a^3 = a^3
def option_D (a : ℕ) : Prop := a^3 * a^2 = a^6

-- Define the theorem stating that option C is the only correct one
theorem correct_calculation (a : ℕ) : ¬option_A a ∧ ¬option_B a ∧ option_C a ∧ ¬option_D a := by
  sorry

end correct_calculation_l41_41989


namespace abc_plus_2_gt_a_plus_b_plus_c_l41_41547

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : |c| < 1) : abc + 2 > a + b + c :=
by
  sorry

end abc_plus_2_gt_a_plus_b_plus_c_l41_41547


namespace people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l41_41194

def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * 3 ^ ((n - 8) / 12) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0 -- default case for unsupported values

def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0 -- default case for unsupported values

theorem people_entering_2pm_to_3pm :
  f 21 + f 22 + f 23 + f 24 = 17460 := sorry

theorem people_leaving_2pm_to_3pm :
  g 21 + g 22 + g 23 + g 24 = 9000 := sorry

theorem peak_visitors_time :
  ∀ n, 1 ≤ n ∧ n ≤ 45 → 
    (n = 28 ↔ ∀ m, 1 ≤ m ∧ m ≤ 45 → f m - g m ≤ f 28 - g 28) := sorry

end people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l41_41194


namespace option_d_correct_l41_41969

variable (a b m n : ℝ)

theorem option_d_correct :
  6 * a + a ≠ 6 * a ^ 2 ∧
  -2 * a + 5 * b ≠ 3 * a * b ∧
  4 * m ^ 2 * n - 2 * m * n ^ 2 ≠ 2 * m * n ∧
  3 * a * b ^ 2 - 5 * b ^ 2 * a = -2 * a * b ^ 2 := by
  sorry

end option_d_correct_l41_41969


namespace find_initial_shells_l41_41844

theorem find_initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end find_initial_shells_l41_41844


namespace minimum_possible_n_l41_41259

theorem minimum_possible_n (n p : ℕ) (h1: p > 0) (h2: 15 * n - 45 = 105) : n = 10 :=
sorry

end minimum_possible_n_l41_41259


namespace daisy_dog_toys_l41_41409

theorem daisy_dog_toys (X : ℕ) (lost_toys : ℕ) (total_toys_after_found : ℕ) : 
    (X - lost_toys + (3 + 3) - lost_toys + 5 = total_toys_after_found) → total_toys_after_found = 13 → X = 5 :=
by
  intros h1 h2
  sorry

end daisy_dog_toys_l41_41409


namespace gnomes_red_hats_small_noses_l41_41719

theorem gnomes_red_hats_small_noses :
  ∀ (total_gnomes red_hats blue_hats big_noses_blue_hats : ℕ),
  total_gnomes = 28 →
  red_hats = (3 * total_gnomes) / 4 →
  blue_hats = total_gnomes - red_hats →
  big_noses_blue_hats = 6 →
  (total_gnomes / 2) - big_noses_blue_hats = 8 →
  red_hats - 8 = 13 :=
by
  intros total_gnomes red_hats blue_hats big_noses_blue_hats
  intros h1 h2 h3 h4 h5
  sorry

end gnomes_red_hats_small_noses_l41_41719


namespace expression_simplification_l41_41981

noncomputable def given_expression : ℝ :=
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1)))

noncomputable def expected_expression : ℝ :=
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3)

theorem expression_simplification :
  given_expression = expected_expression :=
by
  sorry

end expression_simplification_l41_41981


namespace all_real_possible_values_l41_41869

theorem all_real_possible_values 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 1) : 
  ∃ r : ℝ, r = (a^4 + b^4 + c^4) / (ab + bc + ca) :=
sorry

end all_real_possible_values_l41_41869


namespace least_total_bananas_is_1128_l41_41909

noncomputable def least_total_bananas : ℕ :=
  let b₁ := 252
  let b₂ := 252
  let b₃ := 336
  let b₄ := 288
  b₁ + b₂ + b₃ + b₄

theorem least_total_bananas_is_1128 :
  least_total_bananas = 1128 :=
by
  sorry

end least_total_bananas_is_1128_l41_41909


namespace quadrant_angle_l41_41862

theorem quadrant_angle (θ : ℝ) (k : ℤ) (h_theta : 0 < θ ∧ θ < 90) : 
  ((180 * k + θ) % 360 < 90) ∨ (180 * k + θ) % 360 ≥ 180 ∧ (180 * k + θ) % 360 < 270 :=
sorry

end quadrant_angle_l41_41862


namespace range_of_a_l41_41588

theorem range_of_a (A M : ℝ × ℝ) (a : ℝ) (C : ℝ × ℝ → ℝ) (hA : A = (-3, 0)) 
(hM : C M = 1) (hMA : dist M A = 2 * dist M (0, 0)) :
  a ∈ (Set.Icc (1/2 : ℝ) (3/2) ∪ Set.Icc (-3/2) (-1/2)) :=
sorry

end range_of_a_l41_41588


namespace arccos_range_l41_41540

theorem arccos_range (a : ℝ) (x : ℝ) (h₀ : x = Real.sin a) 
  (h₁ : -Real.pi / 4 ≤ a ∧ a ≤ 3 * Real.pi / 4) :
  ∀ y, y = Real.arccos x → 0 ≤ y ∧ y ≤ 3 * Real.pi / 4 := 
sorry

end arccos_range_l41_41540


namespace balls_in_boxes_l41_41020

theorem balls_in_boxes:
  ∃ (x y z : ℕ), 
  x + y + z = 320 ∧ 
  6 * x + 11 * y + 15 * z = 1001 ∧
  x > 0 ∧ y > 0 ∧ z > 0 :=
by
  sorry

end balls_in_boxes_l41_41020


namespace passes_through_point_P_l41_41811

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 7 + a^(x - 1)

theorem passes_through_point_P
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 1 = 8 :=
by
  -- Proof omitted
  sorry

end passes_through_point_P_l41_41811


namespace Eva_needs_weeks_l41_41971

theorem Eva_needs_weeks (apples : ℕ) (days_in_week : ℕ) (weeks : ℕ) 
  (h1 : apples = 14)
  (h2 : days_in_week = 7) 
  (h3 : apples = weeks * days_in_week) : 
  weeks = 2 := 
by 
  sorry

end Eva_needs_weeks_l41_41971


namespace peter_change_left_l41_41171

theorem peter_change_left
  (cost_small : ℕ := 3)
  (cost_large : ℕ := 5)
  (total_money : ℕ := 50)
  (num_small : ℕ := 8)
  (num_large : ℕ := 5) :
  total_money - (num_small * cost_small + num_large * cost_large) = 1 :=
by
  sorry

end peter_change_left_l41_41171


namespace money_left_after_distributions_and_donations_l41_41825

theorem money_left_after_distributions_and_donations 
  (total_income : ℕ)
  (percent_to_children : ℕ)
  (percent_to_each_child : ℕ)
  (number_of_children : ℕ)
  (percent_to_wife : ℕ)
  (percent_to_orphan_house : ℕ)
  (remaining_income_percentage : ℕ)
  (children_distribution : ℕ → ℕ → ℕ)
  (wife_distribution : ℕ → ℕ)
  (calculate_remaining : ℕ → ℕ → ℕ)
  (calculate_donation : ℕ → ℕ → ℕ)
  (calculate_money_left : ℕ → ℕ → ℕ)
  (income : ℕ := 400000)
  (result : ℕ := 57000) :
  children_distribution percent_to_each_child number_of_children = 60 →
  percent_to_wife = 25 →
  remaining_income_percentage = 15 →
  percent_to_orphan_house = 5 →
  wife_distribution percent_to_wife = 100000 →
  calculate_remaining 100 85 = 15 →
  calculate_donation percent_to_orphan_house (calculate_remaining 100 85 * total_income) = 3000 →
  calculate_money_left (calculate_remaining 100 85 * total_income) 3000 = result →
  total_income = income →
  income - (60 * income / 100 + 25 * income / 100 + 5 * (15 * income / 100) / 100) = result
  :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end money_left_after_distributions_and_donations_l41_41825


namespace travel_time_l41_41695

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l41_41695


namespace non_arithmetic_sequence_l41_41818

theorem non_arithmetic_sequence (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) :
    (∀ n, S_n n = n^2 + 2 * n - 1) →
    (∀ n, a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1)) →
    ¬(∀ d, ∀ n, a_n (n+1) = a_n n + d) :=
by
  intros hS ha
  sorry

end non_arithmetic_sequence_l41_41818


namespace total_cost_is_58_l41_41858

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l41_41858


namespace earnings_percentage_difference_l41_41742

-- Defining the conditions
def MikeEarnings : ℕ := 12
def PhilEarnings : ℕ := 6

-- Proving the percentage difference
theorem earnings_percentage_difference :
  ((MikeEarnings - PhilEarnings: ℕ) * 100 / MikeEarnings = 50) :=
by 
  sorry

end earnings_percentage_difference_l41_41742


namespace find_ordered_pairs_l41_41212

theorem find_ordered_pairs (x y : ℝ) :
  x^2 * y = 3 ∧ x + x * y = 4 → (x, y) = (1, 3) ∨ (x, y) = (3, 1 / 3) :=
sorry

end find_ordered_pairs_l41_41212


namespace rug_inner_rectangle_length_l41_41604

theorem rug_inner_rectangle_length
  (width : ℕ)
  (shaded1_width : ℕ)
  (shaded2_width : ℕ)
  (areas_in_ap : ℕ → ℕ → ℕ → Prop)
  (h1 : width = 2)
  (h2 : shaded1_width = 2)
  (h3 : shaded2_width = 2)
  (h4 : ∀ y a1 a2 a3, 
        a1 = 2 * y →
        a2 = 6 * (y + 4) →
        a3 = 10 * (y + 8) →
        areas_in_ap a1 (a2 - a1) (a3 - a2) →
        (a2 - a1 = a3 - a2)) :
  ∃ y, y = 4 :=
by
  sorry

end rug_inner_rectangle_length_l41_41604


namespace income_of_m_l41_41291

theorem income_of_m (M N O : ℝ)
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (M + O) / 2 = 5200) :
  M = 4000 :=
by
  -- sorry is used to skip the actual proof.
  sorry

end income_of_m_l41_41291


namespace cargo_transport_possible_l41_41048

theorem cargo_transport_possible 
  (total_cargo_weight : ℝ) 
  (weight_limit_per_box : ℝ) 
  (number_of_trucks : ℕ) 
  (max_load_per_truck : ℝ)
  (h1 : total_cargo_weight = 13.5)
  (h2 : weight_limit_per_box = 0.35)
  (h3 : number_of_trucks = 11)
  (h4 : max_load_per_truck = 1.5) :
  ∃ (n : ℕ), n ≤ number_of_trucks ∧ (total_cargo_weight / max_load_per_truck) ≤ n :=
by
  sorry

end cargo_transport_possible_l41_41048


namespace find_a_l41_41006

theorem find_a (x y z a : ℝ) (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h2 : a > 0) (h3 : ∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + 6 * z^2 = a → (x + y + z) ≤ 1) :
  a = 1 := 
sorry

end find_a_l41_41006


namespace sequence_monotonic_decreasing_l41_41417

theorem sequence_monotonic_decreasing (t : ℝ) :
  (∀ n : ℕ, n > 0 → (- (n + 1) ^ 2 + t * (n + 1)) - (- n ^ 2 + t * n) < 0) ↔ (t < 3) :=
by 
  sorry

end sequence_monotonic_decreasing_l41_41417


namespace ratio_of_milk_water_in_larger_vessel_l41_41781

-- Definitions of conditions
def volume1 (V : ℝ) : ℝ := 3 * V
def volume2 (V : ℝ) : ℝ := 5 * V

def ratio_milk_water_1 : ℝ × ℝ := (1, 2)
def ratio_milk_water_2 : ℝ × ℝ := (3, 2)

-- Define the problem statement
theorem ratio_of_milk_water_in_larger_vessel (V : ℝ) (hV : V > 0) :
  (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = V ∧ 
  2 * (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = 2 * V ∧ 
  3 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 3 * V ∧ 
  2 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 2 * V →
  (4 * V) / (4 * V) = 1 :=
sorry

end ratio_of_milk_water_in_larger_vessel_l41_41781


namespace different_values_count_l41_41345

theorem different_values_count (i : ℕ) (h : 1 ≤ i ∧ i ≤ 2015) : 
  ∃ l : Finset ℕ, (∀ j ∈ l, ∃ i : ℕ, (1 ≤ i ∧ i ≤ 2015) ∧ j = (i^2 / 2015)) ∧
  l.card = 2016 := 
sorry

end different_values_count_l41_41345


namespace unique_solution_abs_eq_l41_41660

theorem unique_solution_abs_eq : 
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 :=
by
  use -5
  sorry

end unique_solution_abs_eq_l41_41660


namespace mean_of_squares_of_first_four_odd_numbers_l41_41436

theorem mean_of_squares_of_first_four_odd_numbers :
  (1^2 + 3^2 + 5^2 + 7^2) / 4 = 21 := 
by
  sorry

end mean_of_squares_of_first_four_odd_numbers_l41_41436


namespace problem_1_problem_2_problem_3_l41_41603

-- Definitions based on problem conditions
def total_people := 12
def choices := 5
def special_people_count := 3

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Proof problem 1: A, B, and C must be chosen, so select 2 more from the remaining 9 people
theorem problem_1 : choose 9 2 = 36 :=
by sorry

-- Proof problem 2: Only one among A, B, and C is chosen, so select 4 more from the remaining 9 people
theorem problem_2 : choose 3 1 * choose 9 4 = 378 :=
by sorry

-- Proof problem 3: At most two among A, B, and C are chosen
theorem problem_3 : choose 12 5 - choose 9 2 = 756 :=
by sorry

end problem_1_problem_2_problem_3_l41_41603


namespace square_side_length_l41_41121

theorem square_side_length (x : ℝ) (h : x^2 = (1/2) * x * 2) : x = 1 := by
  sorry

end square_side_length_l41_41121


namespace marian_balance_proof_l41_41326

noncomputable def marian_new_balance : ℝ :=
  let initial_balance := 126.00
  let uk_purchase := 50.0
  let uk_discount := 0.10
  let uk_rate := 1.39
  let france_purchase := 70.0
  let france_discount := 0.15
  let france_rate := 1.18
  let japan_purchase := 10000.0
  let japan_discount := 0.05
  let japan_rate := 0.0091
  let towel_return := 45.0
  let interest_rate := 0.015
  let uk_usd := (uk_purchase * (1 - uk_discount)) * uk_rate
  let france_usd := (france_purchase * (1 - france_discount)) * france_rate
  let japan_usd := (japan_purchase * (1 - japan_discount)) * japan_rate
  let gas_usd := (uk_purchase / 2) * uk_rate
  let balance_before_interest := initial_balance + uk_usd + france_usd + japan_usd + gas_usd - towel_return
  let interest := balance_before_interest * interest_rate
  balance_before_interest + interest

theorem marian_balance_proof :
  abs (marian_new_balance - 340.00) < 1 :=
by
  sorry

end marian_balance_proof_l41_41326


namespace smallest_n_l41_41794

theorem smallest_n (n : ℕ) : (n > 0) ∧ (2^n % 30 = 1) → n = 4 :=
by
  intro h
  sorry

end smallest_n_l41_41794


namespace false_proposition_of_quadratic_l41_41316

theorem false_proposition_of_quadratic
  (a : ℝ) (h0 : a ≠ 0)
  (h1 : ¬(5 = a * (1/2)^2 + (-a^2 - 1) * (1/2) + a))
  (h2 : (a^2 + 1) / (2 * a) > 0)
  (h3 : (0, a) = (0, x) ∧ x > 0)
  (h4 : ∀ x : ℝ, a * x^2 + (-a^2 - 1) * x + a ≤ 0) :
  false :=
sorry

end false_proposition_of_quadratic_l41_41316


namespace isosceles_trapezoid_AC_length_l41_41217

noncomputable def length_of_AC (AB AD BC CD AC : ℝ) :=
  AB = 30 ∧ AD = 15 ∧ BC = 15 ∧ CD = 12 → AC = 23.32

theorem isosceles_trapezoid_AC_length :
  length_of_AC 30 15 15 12 23.32 := by
  sorry

end isosceles_trapezoid_AC_length_l41_41217


namespace largest_is_three_l41_41393

variable (p q r : ℝ)

def cond1 : Prop := p + q + r = 3
def cond2 : Prop := p * q + p * r + q * r = 1
def cond3 : Prop := p * q * r = -6

theorem largest_is_three
  (h1 : cond1 p q r)
  (h2 : cond2 p q r)
  (h3 : cond3 p q r) :
  p = 3 ∨ q = 3 ∨ r = 3 := sorry

end largest_is_three_l41_41393


namespace domain_of_g_l41_41548

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

theorem domain_of_g :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)} = -- Expected domain of g(x)
  { x : ℝ |
    (0 ≤ x ∧ x ≤ 6) ∧ -- Domain of f is 0 ≤ x ≤ 6
    2 * x ≤ 6 ∧ -- For g(x) to be in the domain of f(2x)
    0 ≤ 2 * x ∧ -- Ensures 2x fits within the domain 0 < 2x < 6
    x ≠ 2 } -- x cannot be 2
:= sorry

end domain_of_g_l41_41548


namespace solution_exists_l41_41925

theorem solution_exists (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := 
sorry

end solution_exists_l41_41925


namespace find_k_l41_41804

noncomputable def parabola_k : ℝ := 4

theorem find_k (k : ℝ) (h1 : ∀ x, y = k^2 - x^2) (h2 : k > 0)
    (h3 : ∀ A D : (ℝ × ℝ), A = (-k, 0) ∧ D = (k, 0))
    (h4 : ∀ V : (ℝ × ℝ), V = (0, k^2))
    (h5 : 2 * (2 * k + k^2) = 48) : k = 4 :=
  sorry

end find_k_l41_41804


namespace mary_average_speed_l41_41376

noncomputable def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / ((t1 + t2) / 60)

theorem mary_average_speed :
  average_speed 1.5 1.5 45 15 = 3 := by
  sorry

end mary_average_speed_l41_41376


namespace sum_f_inv_l41_41639

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 1 else x ^ 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 9 then (y + 1) / 2 else Real.sqrt y

theorem sum_f_inv :
  (f_inv (-3) + f_inv (-2) + 
   f_inv (-1) + f_inv 0 + 
   f_inv 1 + f_inv 2 + 
   f_inv 3 + f_inv 4 + 
   f_inv 9) = 9 :=
by
  sorry

end sum_f_inv_l41_41639


namespace fewest_coach_handshakes_l41_41736

theorem fewest_coach_handshakes (n_A n_B k_A k_B : ℕ) (h1 : n_A = n_B + 2)
    (h2 : ((n_A * (n_A - 1)) / 2) + ((n_B * (n_B - 1)) / 2) + (n_A * n_B) + k_A + k_B = 620) :
  k_A + k_B = 189 := 
sorry

end fewest_coach_handshakes_l41_41736


namespace find_list_price_l41_41304

noncomputable def list_price (x : ℝ) (alice_price_diff bob_price_diff : ℝ) (alice_comm_fraction bob_comm_fraction : ℝ) : Prop :=
  alice_comm_fraction * (x - alice_price_diff) = bob_comm_fraction * (x - bob_price_diff)

theorem find_list_price : list_price 40 15 25 0.15 0.25 :=
by
  sorry

end find_list_price_l41_41304


namespace Adam_total_shopping_cost_l41_41514

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l41_41514


namespace find_prism_height_l41_41583

variables (base_side_length : ℝ) (density : ℝ) (weight : ℝ) (height : ℝ)

-- Assume the base_side_length is 2 meters, density is 2700 kg/m³, and weight is 86400 kg
def given_conditions := (base_side_length = 2) ∧ (density = 2700) ∧ (weight = 86400)

-- Define the volume based on weight and density
noncomputable def volume (density weight : ℝ) : ℝ := weight / density

-- Define the area of the base
def base_area (side_length : ℝ) : ℝ := side_length * side_length

-- Define the height of the prism
noncomputable def prism_height (volume base_area : ℝ) : ℝ := volume / base_area

-- The proof statement
theorem find_prism_height (h : ℝ) : given_conditions base_side_length density weight → prism_height (volume density weight) (base_area base_side_length) = h :=
by
  intros h_cond
  sorry

end find_prism_height_l41_41583


namespace factorize_cubic_l41_41793

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l41_41793


namespace num_rectangular_tables_l41_41992

theorem num_rectangular_tables (R : ℕ) 
  (rectangular_tables_seat : R * 10 = 70) :
  R = 7 := by
  sorry

end num_rectangular_tables_l41_41992


namespace evaluate_expression_l41_41566

theorem evaluate_expression : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) :=
by
  sorry

end evaluate_expression_l41_41566


namespace estimate_total_height_l41_41926

theorem estimate_total_height :
  let middle_height := 100
  let left_height := 0.80 * middle_height
  let right_height := (left_height + middle_height) - 20
  left_height + middle_height + right_height = 340 := 
by
  sorry

end estimate_total_height_l41_41926


namespace unique_integer_m_l41_41282

theorem unique_integer_m :
  ∃! (m : ℤ), m - ⌊m / (2005 : ℝ)⌋ = 2005 :=
by
  --- Here belongs the proof part, but we leave it with a sorry
  sorry

end unique_integer_m_l41_41282


namespace points_for_victory_l41_41232

theorem points_for_victory (V : ℕ) :
  (∃ (played total_games : ℕ) (points_after_games : ℕ) (remaining_games : ℕ) (needed_points : ℕ) 
     (draw_points defeat_points : ℕ) (minimum_wins : ℕ), 
     played = 5 ∧
     total_games = 20 ∧ 
     points_after_games = 12 ∧
     remaining_games = total_games - played ∧
     needed_points = 40 - points_after_games ∧
     draw_points = 1 ∧
     defeat_points = 0 ∧
     minimum_wins = 7 ∧
     7 * V ≥ needed_points ∧
     remaining_games = total_games - played ∧
     needed_points = 28) → V = 4 :=
sorry

end points_for_victory_l41_41232


namespace rectangular_prism_volume_l41_41945

theorem rectangular_prism_volume (w : ℝ) (w_pos : 0 < w) 
    (h_edges_sum : 4 * w + 8 * (2 * w) + 4 * (w / 2) = 88) :
    (2 * w) * w * (w / 2) = 85184 / 343 :=
by
  sorry

end rectangular_prism_volume_l41_41945


namespace original_price_l41_41675

-- Definitions of conditions
def SalePrice : Float := 70
def DecreasePercentage : Float := 30

-- Statement to prove
theorem original_price (P : Float) (h : 0.70 * P = SalePrice) : P = 100 := by
  sorry

end original_price_l41_41675


namespace mariel_dogs_count_l41_41922

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l41_41922


namespace cats_awake_l41_41353

theorem cats_awake (total_cats asleep_cats cats_awake : ℕ) (h1 : total_cats = 98) (h2 : asleep_cats = 92) (h3 : cats_awake = total_cats - asleep_cats) : cats_awake = 6 :=
by
  -- Definitions and conditions
  subst h1
  subst h2
  subst h3
  -- The statement we need to prove
  sorry

end cats_awake_l41_41353


namespace ratio_of_areas_l41_41019

-- Definitions based on the conditions given
def square_side_length : ℕ := 48
def rectangle_width : ℕ := 56
def rectangle_height : ℕ := 63

-- Areas derived from the definitions
def square_area := square_side_length * square_side_length
def rectangle_area := rectangle_width * rectangle_height

-- Lean statement to prove the ratio of areas
theorem ratio_of_areas :
  (square_area : ℚ) / rectangle_area = 2 / 3 := 
sorry

end ratio_of_areas_l41_41019


namespace solve_system_eq_l41_41787

theorem solve_system_eq (x y : ℝ) :
  x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0 ↔
  x = 4 ∧ y = 1 :=
sorry

end solve_system_eq_l41_41787


namespace arithmetic_seq_value_zero_l41_41917

theorem arithmetic_seq_value_zero (a b c : ℝ) (a_seq : ℕ → ℝ)
    (l m n : ℕ) (h_arith : ∀ k, a_seq (k + 1) - a_seq k = a_seq 1 - a_seq 0)
    (h_l : a_seq l = 1 / a)
    (h_m : a_seq m = 1 / b)
    (h_n : a_seq n = 1 / c) :
    (l - m) * a * b + (m - n) * b * c + (n - l) * c * a = 0 := 
sorry

end arithmetic_seq_value_zero_l41_41917


namespace line_inclination_angle_l41_41281

theorem line_inclination_angle (θ : ℝ) : 
  (∃ θ : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → θ = 3 * π / 4) := sorry

end line_inclination_angle_l41_41281


namespace max_value_S_n_S_m_l41_41928

noncomputable def a (n : ℕ) : ℤ := -(n : ℤ)^2 + 12 * n - 32

noncomputable def S : ℕ → ℤ
| 0       => 0
| (n + 1) => S n + a (n + 1)

theorem max_value_S_n_S_m : ∀ m n : ℕ, m < n → m > 0 → S n - S m ≤ 10 :=
by
  sorry

end max_value_S_n_S_m_l41_41928


namespace fraction_simplification_l41_41552

theorem fraction_simplification :
  (20 / 21) * (35 / 54) * (63 / 50) = (7 / 9) :=
by
  sorry

end fraction_simplification_l41_41552


namespace min_value_seq_div_n_l41_41667

-- Definitions of the conditions
def a_seq (n : ℕ) : ℕ := 
  if n = 0 then 0 else if n = 1 then 98 else 102 + (n - 2) * (2 * n + 2)

-- The property we need to prove
theorem min_value_seq_div_n :
  (∀ n : ℕ, (n ≥ 1) → (a_seq n / n) ≥ 26) ∧ (∃ n : ℕ, (n ≥ 1) ∧ (a_seq n / n) = 26) :=
sorry

end min_value_seq_div_n_l41_41667


namespace problem_statement_l41_41658

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def x : ℝ := alpha ^ 1000
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l41_41658


namespace swimming_class_attendance_l41_41338

theorem swimming_class_attendance (total_students : ℕ) (chess_percentage : ℝ) (swimming_percentage : ℝ) 
  (H1 : total_students = 1000) 
  (H2 : chess_percentage = 0.20) 
  (H3 : swimming_percentage = 0.10) : 
  200 * 0.10 = 20 := 
by sorry

end swimming_class_attendance_l41_41338


namespace find_a10_l41_41240

variable {G : Type*} [LinearOrderedField G]
variable (a : ℕ → G)

-- Conditions
def geometric_sequence (a : ℕ → G) (r : G) := ∀ n, a (n + 1) = r * a n
def positive_terms (a : ℕ → G) := ∀ n, 0 < a n
def specific_condition (a : ℕ → G) := a 3 * a 11 = 16

theorem find_a10
  (h_geom : geometric_sequence a 2)
  (h_pos : positive_terms a)
  (h_cond : specific_condition a) :
  a 10 = 32 := by
  sorry

end find_a10_l41_41240


namespace correct_operation_l41_41662

theorem correct_operation : ¬ (-2 * x + 5 * x = -7 * x) 
                          ∧ (y * x - 3 * x * y = -2 * x * y) 
                          ∧ ¬ (-x^2 - x^2 = 0) 
                          ∧ ¬ (x^2 - x = x) := 
by {
    sorry
}

end correct_operation_l41_41662


namespace find_j_l41_41342

theorem find_j (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / j = 142.07) : j = 400 :=
by
  sorry

end find_j_l41_41342


namespace disjoint_subsets_same_sum_l41_41968

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l41_41968


namespace find_added_number_l41_41649

theorem find_added_number (R X : ℕ) (hR : R = 45) (h : 2 * (2 * R + X) = 188) : X = 4 :=
by 
  -- We would normally provide the proof here
  sorry  -- We skip the proof as per the instructions

end find_added_number_l41_41649


namespace lucy_lovely_age_ratio_l41_41205

theorem lucy_lovely_age_ratio (L l : ℕ) (x : ℕ) (h1 : L = 50) (h2 : 45 = x * (l - 5)) (h3 : 60 = 2 * (l + 10)) :
  (45 / (l - 5)) = 3 :=
by
  sorry

end lucy_lovely_age_ratio_l41_41205


namespace steps_per_flight_l41_41245

-- Define the problem conditions
def jack_flights_up := 3
def jack_flights_down := 6
def steps_height_inches := 8
def jack_height_change_feet := 24

-- Convert the height change to inches
def jack_height_change_inches := jack_height_change_feet * 12

-- Calculate the net flights down
def net_flights_down := jack_flights_down - jack_flights_up

-- Calculate total height change in inches for net flights
def total_height_change_inches := net_flights_down * jack_height_change_inches

-- Calculate the number of steps in each flight
def number_of_steps_per_flight :=
  total_height_change_inches / (steps_height_inches * net_flights_down)

theorem steps_per_flight :
  number_of_steps_per_flight = 108 :=
sorry

end steps_per_flight_l41_41245


namespace sum_of_digits_base8_product_l41_41244

theorem sum_of_digits_base8_product
  (a b : ℕ)
  (a_base8 : a = 3 * 8^1 + 4 * 8^0)
  (b_base8 : b = 2 * 8^1 + 2 * 8^0)
  (product : ℕ := a * b)
  (product_base8 : ℕ := (product / 64) * 8^2 + ((product / 8) % 8) * 8^1 + (product % 8)) :
  ((product_base8 / 8^2) + ((product_base8 / 8) % 8) + (product_base8 % 8)) = 1 * 8^1 + 6 * 8^0 :=
sorry

end sum_of_digits_base8_product_l41_41244


namespace mass_percentage_O_in_mixture_l41_41467

/-- Mass percentage of oxygen in a mixture of Acetone and Methanol -/
theorem mass_percentage_O_in_mixture 
  (mass_acetone: ℝ)
  (mass_methanol: ℝ)
  (mass_O_acetone: ℝ)
  (mass_O_methanol: ℝ) 
  (total_mass: ℝ) : 
  mass_acetone = 30 → 
  mass_methanol = 20 → 
  mass_O_acetone = (16 / 58.08) * 30 →
  mass_O_methanol = (16 / 32.04) * 20 →
  total_mass = mass_acetone + mass_methanol →
  ((mass_O_acetone + mass_O_methanol) / total_mass) * 100 = 36.52 :=
by
  sorry

end mass_percentage_O_in_mixture_l41_41467


namespace incorrect_statement_D_l41_41809

noncomputable def f : ℝ → ℝ := sorry

axiom A1 : ∃ x : ℝ, f x ≠ 0
axiom A2 : ∀ x : ℝ, f (x + 1) = -f (2 - x)
axiom A3 : ∀ x : ℝ, f (x + 3) = f (x - 3)

theorem incorrect_statement_D :
  ¬ (∀ x : ℝ, f (3 + x) + f (3 - x) = 0) :=
sorry

end incorrect_statement_D_l41_41809


namespace digits_in_base_5_l41_41186

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end digits_in_base_5_l41_41186


namespace find_cost_price_l41_41111

theorem find_cost_price (C : ℝ) (h1 : 0.88 * C + 1500 = 1.12 * C) : C = 6250 := 
by
  sorry

end find_cost_price_l41_41111


namespace rectangular_garden_length_l41_41757

theorem rectangular_garden_length (P B L : ℕ) (h1 : P = 1800) (h2 : B = 400) (h3 : P = 2 * (L + B)) : L = 500 :=
sorry

end rectangular_garden_length_l41_41757


namespace students_sampled_from_schoolB_l41_41632

-- Definitions from the conditions in a)
def schoolA_students := 800
def schoolB_students := 500
def total_students := schoolA_students + schoolB_students
def schoolA_sampled_students := 48

-- Mathematically equivalent proof problem
theorem students_sampled_from_schoolB : 
  let proportionA := (schoolA_students : ℝ) / total_students
  let proportionB := (schoolB_students : ℝ) / total_students
  let total_sampled_students := schoolA_sampled_students / proportionA
  let b_sampled_students := proportionB * total_sampled_students
  b_sampled_students = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end students_sampled_from_schoolB_l41_41632


namespace intersecting_lines_product_l41_41626

theorem intersecting_lines_product 
  (a b : ℝ)
  (T : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ))
  (hT : T = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ a * x + y - 3 = 0})
  (hS : S = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x - y - b = 0})
  (h_intersect : (2, 1) ∈ T) (h_intersect_S : (2, 1) ∈ S) :
  a * b = 1 := 
by
  sorry

end intersecting_lines_product_l41_41626


namespace total_balls_in_bag_l41_41961

theorem total_balls_in_bag (x : ℕ) (H : 3/(4 + x) = x/(4 + x)) : 3 + 1 + x = 7 :=
by
  -- We would provide the proof here, but it's not required as per the instructions.
  sorry

end total_balls_in_bag_l41_41961


namespace harry_annual_pet_feeding_cost_l41_41599

def monthly_cost_snake := 10
def monthly_cost_iguana := 5
def monthly_cost_gecko := 15
def num_snakes := 4
def num_iguanas := 2
def num_geckos := 3
def months_in_year := 12

theorem harry_annual_pet_feeding_cost :
  (num_snakes * monthly_cost_snake + 
   num_iguanas * monthly_cost_iguana + 
   num_geckos * monthly_cost_gecko) * 
   months_in_year = 1140 := 
sorry

end harry_annual_pet_feeding_cost_l41_41599


namespace group_elements_eq_one_l41_41838
-- Import the entire math library

-- Define the main theorem
theorem group_elements_eq_one 
  {G : Type*} [Group G] 
  (a b : G) 
  (h1 : a * b^2 = b^3 * a) 
  (h2 : b * a^2 = a^3 * b) : 
  a = 1 ∧ b = 1 := 
  by 
  sorry

end group_elements_eq_one_l41_41838


namespace leif_has_more_oranges_than_apples_l41_41144

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l41_41144


namespace solution_set_for_f_l41_41458

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -x^2 + x

theorem solution_set_for_f (x : ℝ) :
  f (x^2 - x + 1) < 12 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_set_for_f_l41_41458


namespace union_of_sets_l41_41891

theorem union_of_sets (A B : Set ℤ) (hA : A = {-1, 3}) (hB : B = {2, 3}) : A ∪ B = {-1, 2, 3} := 
by
  sorry

end union_of_sets_l41_41891


namespace number_of_people_l41_41091

def avg_weight_increase : ℝ := 2.5
def old_person_weight : ℝ := 45
def new_person_weight : ℝ := 65

theorem number_of_people (n : ℕ) 
  (h1 : avg_weight_increase = 2.5) 
  (h2 : old_person_weight = 45) 
  (h3 : new_person_weight = 65) :
  n = 8 :=
  sorry

end number_of_people_l41_41091


namespace smallest_integer_m_l41_41670

theorem smallest_integer_m (m : ℕ) : m > 1 ∧ m % 13 = 2 ∧ m % 5 = 2 ∧ m % 3 = 2 → m = 197 := 
by 
  sorry

end smallest_integer_m_l41_41670


namespace orthogonal_projection_l41_41506

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_squared := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_squared * u.1, dot_uv / norm_u_squared * u.2)

theorem orthogonal_projection
  (a b : ℝ × ℝ)
  (h_orth : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj a (4, -4) = (-4/5, -8/5)) :
  proj b (4, -4) = (24/5, -12/5) :=
sorry

end orthogonal_projection_l41_41506


namespace lines_in_plane_l41_41035

  -- Define the necessary objects in Lean
  structure Line (α : Type) := (equation : α → α → Prop)

  def same_plane (l1 l2 : Line ℝ) : Prop := 
  -- Here you can define what it means for l1 and l2 to be in the same plane.
  sorry

  def intersect (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to intersect.
  sorry

  def parallel (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to be parallel.
  sorry

  theorem lines_in_plane (l1 l2 : Line ℝ) (h : same_plane l1 l2) : 
    (intersect l1 l2) ∨ (parallel l1 l2) := 
  by 
      sorry
  
end lines_in_plane_l41_41035


namespace wine_distribution_l41_41616

theorem wine_distribution (m n k s : ℕ) (h : Nat.gcd m (Nat.gcd n k) = 1) (h_s : s < m + n + k) :
  ∃ g : ℕ, g = s := by
  sorry

end wine_distribution_l41_41616


namespace negation_of_prop_l41_41180

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end negation_of_prop_l41_41180


namespace maggie_fraction_caught_l41_41475

theorem maggie_fraction_caught :
  let total_goldfish := 100
  let allowed_to_take_home := total_goldfish / 2
  let remaining_goldfish_to_catch := 20
  let goldfish_caught := allowed_to_take_home - remaining_goldfish_to_catch
  (goldfish_caught / allowed_to_take_home : ℚ) = 3 / 5 :=
by
  sorry

end maggie_fraction_caught_l41_41475


namespace triangle_median_perpendicular_l41_41457

theorem triangle_median_perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : (x1 - (x2 + x3) / 2) * (x2 - (x1 + x3) / 2) + (y1 - (y2 + y3) / 2) * (y2 - (y1 + y3) / 2) = 0)
  (h2 : (x2 - x3) ^ 2 + (y2 - y3) ^ 2 = 64)
  (h3 : (x1 - x3) ^ 2 + (y1 - y3) ^ 2 = 25) : 
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = 22.25 := sorry

end triangle_median_perpendicular_l41_41457


namespace sujis_age_l41_41760

theorem sujis_age (x : ℕ) (Abi Suji : ℕ)
  (h1 : Abi = 5 * x)
  (h2 : Suji = 4 * x)
  (h3 : (Abi + 3) / (Suji + 3) = 11 / 9) : 
  Suji = 24 := 
by 
  sorry

end sujis_age_l41_41760


namespace jim_taxi_distance_l41_41855

theorem jim_taxi_distance (initial_fee charge_per_segment total_charge : ℝ) (segment_len_miles : ℝ)
(init_fee_eq : initial_fee = 2.5)
(charge_per_seg_eq : charge_per_segment = 0.35)
(total_charge_eq : total_charge = 5.65)
(segment_length_eq : segment_len_miles = 2/5):
  let charge_for_distance := total_charge - initial_fee
  let num_segments := charge_for_distance / charge_per_segment
  let total_miles := num_segments * segment_len_miles
  total_miles = 3.6 :=
by
  intros
  sorry

end jim_taxi_distance_l41_41855


namespace find_rate_of_grapes_l41_41083

def rate_per_kg_of_grapes (G : ℝ) : Prop :=
  let cost_of_grapes := 8 * G
  let cost_of_mangoes := 10 * 55
  let total_paid := 1110
  cost_of_grapes + cost_of_mangoes = total_paid

theorem find_rate_of_grapes : rate_per_kg_of_grapes 70 :=
by
  unfold rate_per_kg_of_grapes
  sorry

end find_rate_of_grapes_l41_41083


namespace quadratic_roots_inverse_sum_l41_41367

theorem quadratic_roots_inverse_sum (t q α β : ℝ) (h1 : α + β = t) (h2 : α * β = q) 
  (h3 : ∀ n : ℕ, n ≥ 1 → α^n + β^n = t) : (1 / α^2011 + 1 / β^2011) = 2 := 
by 
  sorry

end quadratic_roots_inverse_sum_l41_41367


namespace kaylin_is_younger_by_five_l41_41347

def Freyja_age := 10
def Kaylin_age := 33
def Eli_age := Freyja_age + 9
def Sarah_age := 2 * Eli_age
def age_difference := Sarah_age - Kaylin_age

theorem kaylin_is_younger_by_five : age_difference = 5 := 
by
  show 5 = Sarah_age - Kaylin_age
  sorry

end kaylin_is_younger_by_five_l41_41347


namespace moles_of_H2O_combined_l41_41574

theorem moles_of_H2O_combined (mole_NH4Cl mole_NH4OH : ℕ) (reaction : mole_NH4Cl = 1 ∧ mole_NH4OH = 1) : 
  ∃ mole_H2O : ℕ, mole_H2O = 1 :=
by
  sorry

end moles_of_H2O_combined_l41_41574


namespace integer_solutions_count_eq_11_l41_41105

theorem integer_solutions_count_eq_11 :
  ∃ (count : ℕ), (∀ n : ℤ, (n + 2) * (n - 5) + n ≤ 10 ↔ (n ≥ -4 ∧ n ≤ 6)) ∧ count = 11 :=
by
  sorry

end integer_solutions_count_eq_11_l41_41105


namespace range_of_y_l41_41642

theorem range_of_y (x y : ℝ) (h1 : |y - 2 * x| = x^2) (h2 : -1 < x) (h3 : x < 0) : -3 < y ∧ y < 0 :=
by
  sorry

end range_of_y_l41_41642


namespace sum_nonnegative_reals_l41_41985

variable {x y z : ℝ}

theorem sum_nonnegative_reals (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 := 
by sorry

end sum_nonnegative_reals_l41_41985


namespace max_value_k_l41_41942

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 4
  | (n+1) => 3 * seq n - 2

theorem max_value_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → k * (seq n) ≤ 9^n) → k ≤ 9 / 4 :=
sorry

end max_value_k_l41_41942


namespace calculate_difference_of_squares_l41_41933

theorem calculate_difference_of_squares : (153^2 - 147^2) = 1800 := by
  sorry

end calculate_difference_of_squares_l41_41933


namespace remaining_distance_l41_41997

theorem remaining_distance (speed time distance_covered total_distance remaining_distance : ℕ) 
  (h1 : speed = 60) 
  (h2 : time = 2) 
  (h3 : total_distance = 300)
  (h4 : distance_covered = speed * time) 
  (h5 : remaining_distance = total_distance - distance_covered) : 
  remaining_distance = 180 := 
by
  sorry

end remaining_distance_l41_41997


namespace eval_expression_l41_41197

theorem eval_expression : 4 * (8 - 3) - 6 = 14 :=
by
  sorry

end eval_expression_l41_41197


namespace round_robin_games_l41_41321

theorem round_robin_games (x : ℕ) (h : ∃ (n : ℕ), n = 15) : (x * (x - 1)) / 2 = 15 :=
sorry

end round_robin_games_l41_41321


namespace octal_742_to_decimal_l41_41937

theorem octal_742_to_decimal : (7 * 8^2 + 4 * 8^1 + 2 * 8^0 = 482) :=
by
  sorry

end octal_742_to_decimal_l41_41937


namespace mandy_chocolate_pieces_l41_41082

def chocolate_pieces_total : ℕ := 60
def half (n : ℕ) : ℕ := n / 2

def michael_taken : ℕ := half chocolate_pieces_total
def paige_taken : ℕ := half (chocolate_pieces_total - michael_taken)
def ben_taken : ℕ := half (chocolate_pieces_total - michael_taken - paige_taken)
def mandy_left : ℕ := chocolate_pieces_total - michael_taken - paige_taken - ben_taken

theorem mandy_chocolate_pieces : mandy_left = 8 :=
  by
  -- proof to be provided here
  sorry

end mandy_chocolate_pieces_l41_41082


namespace power_function_even_l41_41405

-- Define the function and its properties
def f (x : ℝ) (α : ℤ) : ℝ := x ^ (Int.toNat α)

-- State the theorem with given conditions
theorem power_function_even (α : ℤ) 
    (h : f 1 α ^ 2 + f (-1) α ^ 2 = 2 * (f 1 α + f (-1) α - 1)) : 
    ∀ x : ℝ, f x α = f (-x) α :=
by
  sorry

end power_function_even_l41_41405


namespace value_of_a_100_l41_41895

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (succ k) => sequence k + 4

theorem value_of_a_100 : sequence 99 = 397 := by
  sorry

end value_of_a_100_l41_41895


namespace range_of_m_l41_41820

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + |x - 1| > m) → m < 1 :=
by
  sorry

end range_of_m_l41_41820


namespace general_equation_l41_41015

theorem general_equation (n : ℤ) : 
    ∀ (a b : ℤ), 
    (a = 2 ∧ b = 6) ∨ (a = 5 ∧ b = 3) ∨ (a = 7 ∧ b = 1) ∨ (a = 10 ∧ b = -2) → 
    (a / (a - 4) + b / (b - 4) = 2) →
    (n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2) :=
by
  intros a b h_cond h_eq
  sorry

end general_equation_l41_41015


namespace armistice_day_is_wednesday_l41_41350

-- Define the starting date
def start_day : Nat := 5 -- 5 represents Friday if we consider 0 = Sunday

-- Define the number of days after which armistice was signed
def days_after : Nat := 2253

-- Define the target day (Wednesday = 3)
def expected_day : Nat := 3

-- Define the function to calculate the day of the week after a number of days
def day_after_n_days (start_day : Nat) (n : Nat) : Nat :=
  (start_day + n) % 7

-- Define the theorem to prove the equivalent mathematical problem
theorem armistice_day_is_wednesday : day_after_n_days start_day days_after = expected_day := by
  sorry

end armistice_day_is_wednesday_l41_41350


namespace avg_first_12_even_is_13_l41_41403

-- Definition of the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- The sum of the first 12 even numbers
def sum_first_12_even_numbers : ℕ := first_12_even_numbers.sum

-- Number of first 12 even numbers
def count_12_even_numbers : ℕ := first_12_even_numbers.length

-- The average of the first 12 even numbers
def average_12_even_numbers : ℕ := sum_first_12_even_numbers / count_12_even_numbers

-- Proof statement that the average of the first 12 even numbers is 13
theorem avg_first_12_even_is_13 : average_12_even_numbers = 13 := by
  sorry

end avg_first_12_even_is_13_l41_41403


namespace solution_set_l41_41723

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end solution_set_l41_41723


namespace go_total_pieces_l41_41103

theorem go_total_pieces (T : ℕ) (h : T > 0) (prob_black : T = (3 : ℕ) * 4) : T = 12 := by
  sorry

end go_total_pieces_l41_41103


namespace sand_loss_l41_41635

variable (initial_sand : ℝ) (final_sand : ℝ)

theorem sand_loss (h1 : initial_sand = 4.1) (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
  -- With the given conditions we'll prove this theorem
  sorry

end sand_loss_l41_41635


namespace alice_lost_second_game_l41_41807

/-- Alice, Belle, and Cathy had an arm-wrestling contest. In each game, two girls wrestled, while the third rested.
After each game, the winner played the next game against the girl who had rested.
Given that Alice played 10 times, Belle played 15 times, and Cathy played 17 times; prove Alice lost the second game. --/

theorem alice_lost_second_game (alice_plays : ℕ) (belle_plays : ℕ) (cathy_plays : ℕ) :
  alice_plays = 10 → belle_plays = 15 → cathy_plays = 17 → 
  ∃ (lost_second_game : String), lost_second_game = "Alice" := by
  intros hA hB hC
  sorry

end alice_lost_second_game_l41_41807


namespace ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l41_41201

-- Problem 1
theorem ab_eq_6_pos_or_neg (a b : ℚ) (h : a * b = 6) : a + b > 0 ∨ a + b < 0 := sorry

-- Problem 2
theorem max_ab_when_sum_neg5 (a b : ℤ) (h : a + b = -5) : a * b ≤ 6 := sorry

-- Problem 3
theorem ab_lt_0_sign_of_sum (a b : ℚ) (h : a * b < 0) : (a + b > 0 ∨ a + b = 0 ∨ a + b < 0) := sorry

end ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l41_41201


namespace minimum_possible_length_of_third_side_l41_41714

theorem minimum_possible_length_of_third_side (a b : ℝ) (h : a = 8 ∧ b = 15 ∨ a = 15 ∧ b = 8) : 
  ∃ c : ℝ, (c * c = a * a + b * b ∨ c * c = a * a - b * b ∨ c * c = b * b - a * a) ∧ c = Real.sqrt 161 :=
by
  sorry

end minimum_possible_length_of_third_side_l41_41714


namespace derivative_f_at_1_l41_41894

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1_l41_41894


namespace number_of_members_is_44_l41_41443

-- Define necessary parameters and conditions
def paise_per_rupee : Nat := 100

def total_collection_in_paise : Nat := 1936

def number_of_members_in_group (n : Nat) : Prop :=
  n * n = total_collection_in_paise

-- Proposition to prove
theorem number_of_members_is_44 : number_of_members_in_group 44 :=
by
  sorry

end number_of_members_is_44_l41_41443


namespace find_first_number_l41_41518

theorem find_first_number 
  (first_number second_number hcf lcm : ℕ) 
  (hCF_condition : hcf = 12) 
  (lCM_condition : lcm = 396) 
  (one_number_condition : first_number = 99) 
  (relation_condition : first_number * second_number = hcf * lcm) : 
  second_number = 48 :=
by
  sorry

end find_first_number_l41_41518


namespace problem1_solution_problem2_solution_l41_41142

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : 2 * x + 3 * y = 8) (h2 : x = y - 1) : x = 1 ∧ y = 2 := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 2 * x - y = -1) (h2 : x + 3 * y = 17) : x = 2 ∧ y = 5 := by
  sorry

end problem1_solution_problem2_solution_l41_41142


namespace sequence_nonzero_l41_41492

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n ≥ 3, 
    if (a (n - 1) * a (n - 2)) % 2 = 0 then 
      a n = 5 * (a (n - 1)) - 3 * (a (n - 2)) 
    else 
      a n = (a (n - 1)) - (a (n - 2))

theorem sequence_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, a n ≠ 0 := 
by sorry

end sequence_nonzero_l41_41492


namespace tg_half_product_l41_41769

open Real

variable (α β : ℝ)

theorem tg_half_product (h1 : sin α + sin β = 2 * sin (α + β))
                        (h2 : ∀ n : ℤ, α + β ≠ 2 * π * n) :
  tan (α / 2) * tan (β / 2) = 1 / 3 := by
  sorry

end tg_half_product_l41_41769


namespace min_value_of_quadratic_l41_41468

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 2 ∧ (∀ y : ℝ, quadratic y ≥ 2) :=
by
  use 4
  sorry

end min_value_of_quadratic_l41_41468


namespace first_player_winning_strategy_l41_41586

noncomputable def optimal_first_move : ℕ := 45

-- Prove that with 300 matches initially and following the game rules,
-- taking 45 matches on the first turn leaves the opponent in a losing position.

theorem first_player_winning_strategy (n : ℕ) (h₀ : n = 300) :
    ∃ m : ℕ, (m ≤ n / 2 ∧ n - m = 255) :=
by
  exists optimal_first_move
  sorry

end first_player_winning_strategy_l41_41586


namespace no_natural_numbers_satisfy_equation_l41_41084

theorem no_natural_numbers_satisfy_equation :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y + x + y = 2019 :=
by
  sorry

end no_natural_numbers_satisfy_equation_l41_41084


namespace simplify_frac_op_l41_41831

-- Definition of the operation *
def frac_op (a b c d : ℚ) : ℚ := (a * c) * (d / (b + 1))

-- Proof problem stating the specific operation result
theorem simplify_frac_op :
  frac_op 5 11 9 4 = 15 :=
by
  sorry

end simplify_frac_op_l41_41831


namespace certain_number_l41_41687

theorem certain_number (x y a : ℤ) (h1 : 4 * x + y = a) (h2 : 2 * x - y = 20) 
  (h3 : y ^ 2 = 4) : a = 46 :=
sorry

end certain_number_l41_41687


namespace solve_tetrahedron_side_length_l41_41024

noncomputable def side_length_of_circumscribing_tetrahedron (r : ℝ) (tangent_spheres : ℕ) (radius_spheres_equal : ℝ) : ℝ := 
  if h : r = 1 ∧ tangent_spheres = 4 then
    2 + 2 * Real.sqrt 6
  else
    0

theorem solve_tetrahedron_side_length :
  side_length_of_circumscribing_tetrahedron 1 4 1 = 2 + 2 * Real.sqrt 6 :=
by
  sorry

end solve_tetrahedron_side_length_l41_41024


namespace soccer_team_lineups_l41_41333

noncomputable def num_starting_lineups (n k t g : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) k)

theorem soccer_team_lineups :
  num_starting_lineups 18 9 1 1 = 3501120 := by
    sorry

end soccer_team_lineups_l41_41333


namespace roger_initial_money_l41_41800

theorem roger_initial_money (x : ℤ) 
    (h1 : x + 28 - 25 = 19) : 
    x = 16 := 
by 
    sorry

end roger_initial_money_l41_41800


namespace substract_repeating_decimal_l41_41530

noncomputable def repeating_decimal : ℝ := 1 / 3

theorem substract_repeating_decimal (x : ℝ) (h : x = repeating_decimal) : 
  1 - x = 2 / 3 :=
by
  sorry

end substract_repeating_decimal_l41_41530


namespace no_fib_right_triangle_l41_41863

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem no_fib_right_triangle (n : ℕ) : 
  ¬ (fibonacci n)^2 + (fibonacci (n+1))^2 = (fibonacci (n+2))^2 := 
by 
  sorry

end no_fib_right_triangle_l41_41863


namespace lunch_people_count_l41_41522

theorem lunch_people_count
  (C : ℝ)   -- total lunch cost including gratuity
  (G : ℝ)   -- gratuity rate
  (P : ℝ)   -- average price per person excluding gratuity
  (n : ℕ)   -- number of people
  (h1 : C = 207.0)  -- condition: total cost with gratuity
  (h2 : G = 0.15)   -- condition: gratuity rate of 15%
  (h3 : P = 12.0)   -- condition: average price per person
  (h4 : C = (1 + G) * n * P) -- condition: total cost with gratuity is (1 + gratuity rate) * number of people * average price per person
  : n = 15 :=       -- conclusion: number of people
sorry

end lunch_people_count_l41_41522


namespace arithmetic_sequence_ninth_term_l41_41014

theorem arithmetic_sequence_ninth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 5 * d = 11) :
  a + 8 * d = 17 := by
  sorry

end arithmetic_sequence_ninth_term_l41_41014


namespace average_distinct_k_values_l41_41337

theorem average_distinct_k_values (k : ℕ) (h : ∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ r1 > 0 ∧ r2 > 0) : k = 15 :=
sorry

end average_distinct_k_values_l41_41337


namespace find_smallest_n_l41_41931

theorem find_smallest_n 
  (n : ℕ) 
  (hn : 23 * n ≡ 789 [MOD 8]) : 
  ∃ n : ℕ, n > 0 ∧ n ≡ 3 [MOD 8] :=
sorry

end find_smallest_n_l41_41931


namespace find_initial_order_l41_41641

variables (x : ℕ)

def initial_order (x : ℕ) :=
  x + 60 = 72 * (x / 90 + 1)

theorem find_initial_order (h1 : initial_order x) : x = 60 :=
  sorry

end find_initial_order_l41_41641


namespace lcm_of_numbers_is_91_l41_41904

def ratio (a b : ℕ) (p q : ℕ) : Prop := p * b = q * a

theorem lcm_of_numbers_is_91 (a b : ℕ) (h_ratio : ratio a b 7 13) (h_gcd : Nat.gcd a b = 15) :
  Nat.lcm a b = 91 := 
by sorry

end lcm_of_numbers_is_91_l41_41904


namespace positive_difference_16_l41_41503

def avg_is_37 (y : ℤ) : Prop := (45 + y) / 2 = 37

def positive_difference (a b : ℤ) : ℤ := if a > b then a - b else b - a

theorem positive_difference_16 (y : ℤ) (h : avg_is_37 y) : positive_difference 45 y = 16 :=
by
  sorry

end positive_difference_16_l41_41503


namespace find_math_marks_l41_41958

theorem find_math_marks :
  ∀ (english marks physics chemistry biology : ℕ) (average : ℕ),
  average = 78 →
  english = 91 →
  physics = 82 →
  chemistry = 67 →
  biology = 85 →
  (english + marks + physics + chemistry + biology) / 5 = average →
  marks = 65 :=
by
  intros english marks physics chemistry biology average h_average h_english h_physics h_chemistry h_biology h_avg_eq
  sorry

end find_math_marks_l41_41958


namespace problem_statement_l41_41612

theorem problem_statement (a b : ℝ) (h1 : 1 + b = 0) (h2 : a - 3 = 0) : 
  3 * (a^2 - 2 * a * b + b^2) - (4 * a^2 - 2 * (1 / 2 * a^2 + a * b - 3 / 2 * b^2)) = 12 :=
by
  sorry

end problem_statement_l41_41612


namespace decreasing_on_transformed_interval_l41_41587

theorem decreasing_on_transformed_interval
  (f : ℝ → ℝ)
  (h : ∀ ⦃x₁ x₂ : ℝ⦄, 1 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 2 → f x₁ ≤ f x₂) :
  ∀ ⦃x₁ x₂ : ℝ⦄, -1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 0 → f (1 - x₂) ≤ f (1 - x₁) :=
sorry

end decreasing_on_transformed_interval_l41_41587


namespace cos_A_value_find_c_l41_41995

theorem cos_A_value (a b c A B C : ℝ) (h : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) : 
  Real.cos A = 1 / 2 := 
  sorry

theorem find_c (B C : ℝ) (A : B + C = Real.pi - A) (h1 : 1 = 1) 
  (h2 : Real.cos (B / 2) * Real.cos (B / 2) + Real.cos (C / 2) * Real.cos (C / 2) = 1 + Real.sqrt (3) / 4) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt (3) / 3 ∨ c = Real.sqrt (3) / 3 := 
  sorry

end cos_A_value_find_c_l41_41995


namespace area_hexagon_DEFD_EFE_l41_41876

variable (D E F D' E' F' : Type)
variable (perimeter_DEF : ℝ) (radius_circumcircle : ℝ)
variable (area_hexagon : ℝ)

theorem area_hexagon_DEFD_EFE' (h1 : perimeter_DEF = 42)
    (h2 : radius_circumcircle = 7)
    (h_def : area_hexagon = 147) :
  area_hexagon = 147 := 
sorry

end area_hexagon_DEFD_EFE_l41_41876


namespace fifth_inequality_l41_41488

theorem fifth_inequality :
  1 + (1 / (2^2 : ℝ)) + (1 / (3^2 : ℝ)) + (1 / (4^2 : ℝ)) + (1 / (5^2 : ℝ)) + (1 / (6^2 : ℝ)) < (11 / 6 : ℝ) :=
by
  sorry

end fifth_inequality_l41_41488


namespace same_color_probability_l41_41780

def sides := 12
def violet_sides := 3
def orange_sides := 4
def lime_sides := 5

def prob_violet := violet_sides / sides
def prob_orange := orange_sides / sides
def prob_lime := lime_sides / sides

theorem same_color_probability :
  (prob_violet * prob_violet) + (prob_orange * prob_orange) + (prob_lime * prob_lime) = 25 / 72 :=
by
  sorry

end same_color_probability_l41_41780


namespace circle_center_eq_l41_41797

theorem circle_center_eq (x y : ℝ) :
    (x^2 + y^2 - 2*x + y + 1/4 = 0) → (x = 1 ∧ y = -1/2) :=
by
  sorry

end circle_center_eq_l41_41797


namespace no_infinite_prime_sequence_l41_41395

theorem no_infinite_prime_sequence (p : ℕ → ℕ)
  (h : ∀ k : ℕ, Nat.Prime (p k) ∧ p (k + 1) = 5 * p k + 4) :
  ¬ ∀ n : ℕ, Nat.Prime (p n) :=
by
  sorry

end no_infinite_prime_sequence_l41_41395


namespace find_f_10_l41_41596

variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f x = 2 * x^2 + y
def condition2 : Prop := f 2 = 30

-- Theorem to prove
theorem find_f_10 (h1 : condition1 f y) (h2 : condition2 f) : f 10 = 222 := 
sorry

end find_f_10_l41_41596


namespace parity_of_expression_l41_41428

theorem parity_of_expression (a b c : ℕ) (ha : a % 2 = 1) (hb : b % 2 = 0) :
  (3 ^ a + (b - 1) ^ 2 * (c + 1)) % 2 = if c % 2 = 0 then 1 else 0 :=
by
  sorry

end parity_of_expression_l41_41428


namespace new_average_l41_41216

variable (avg9 : ℝ) (score10 : ℝ) (n : ℕ)
variable (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9)

theorem new_average (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9) :
  ((n * avg9 + score10) / (n + 1)) = 82 :=
by
  rw [h, h10, n9]
  sorry

end new_average_l41_41216


namespace puppies_count_l41_41472

theorem puppies_count 
  (dogs : ℕ := 3)
  (dog_meal_weight : ℕ := 4)
  (dog_meals_per_day : ℕ := 3)
  (total_food : ℕ := 108)
  (puppy_meal_multiplier : ℕ := 2)
  (puppy_meal_frequency_multiplier : ℕ := 3) :
  ∃ (puppies : ℕ), puppies = 4 :=
by
  let dog_daily_food := dog_meal_weight * dog_meals_per_day
  let puppy_meal_weight := dog_meal_weight / puppy_meal_multiplier
  let puppy_daily_food := puppy_meal_weight * puppy_meal_frequency_multiplier * dog_meals_per_day
  let total_dog_food := dogs * dog_daily_food
  let total_puppy_food := total_food - total_dog_food
  let puppies := total_puppy_food / puppy_daily_food
  use puppies
  have h_puppies_correct : puppies = 4 := sorry
  exact h_puppies_correct

end puppies_count_l41_41472


namespace cannot_factorize_using_difference_of_squares_l41_41967

theorem cannot_factorize_using_difference_of_squares (x y : ℝ) :
  ¬ ∃ a b : ℝ, -x^2 - y^2 = a^2 - b^2 :=
sorry

end cannot_factorize_using_difference_of_squares_l41_41967


namespace region_in_quadrants_l41_41865

theorem region_in_quadrants (x y : ℝ) :
  (y > 3 * x) → (y > 5 - 2 * x) → (x > 0 ∧ y > 0) :=
by
  intros h₁ h₂
  sorry

end region_in_quadrants_l41_41865


namespace peaches_left_l41_41923

/-- Brenda picks 3600 peaches, 37.5% are fresh, and 250 are disposed of. Prove that Brenda has 1100 peaches left. -/
theorem peaches_left (total_peaches : ℕ) (percent_fresh : ℚ) (peaches_disposed : ℕ) (h1 : total_peaches = 3600) (h2 : percent_fresh = 3 / 8) (h3 : peaches_disposed = 250) : 
  total_peaches * percent_fresh - peaches_disposed = 1100 := 
by
  sorry

end peaches_left_l41_41923


namespace expanded_polynomial_correct_l41_41247

noncomputable def polynomial_product (x : ℚ) : ℚ :=
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3)

theorem expanded_polynomial_correct (x : ℚ) : 
  polynomial_product x = 2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := 
by
  sorry

end expanded_polynomial_correct_l41_41247


namespace find_ratio_l41_41036

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given conditions
axiom sum_arithmetic_a (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom sum_arithmetic_b (n : ℕ) : T n = n / 2 * (b 1 + b n)
axiom sum_ratios (n : ℕ) : S n / T n = (2 * n + 1) / (3 * n + 2)

-- The proof problem
theorem find_ratio : (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
sorry

end find_ratio_l41_41036


namespace find_q_l41_41480

variable {a d q : ℝ}
variables (M N : Set ℝ)

theorem find_q (hM : M = {a, a + d, a + 2 * d}) 
              (hN : N = {a, a * q, a * q^2})
              (ha : a ≠ 0)
              (heq : M = N) :
  q = -1 / 2 :=
sorry

end find_q_l41_41480


namespace approx_average_sqft_per_person_l41_41320

noncomputable def average_sqft_per_person 
  (population : ℕ) 
  (land_area_sqmi : ℕ) 
  (sqft_per_sqmi : ℕ) : ℕ :=
(sqft_per_sqmi * land_area_sqmi) / population

theorem approx_average_sqft_per_person :
  average_sqft_per_person 331000000 3796742 (5280 ^ 2) = 319697 := 
sorry

end approx_average_sqft_per_person_l41_41320


namespace weight_of_new_person_l41_41231

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → old_weight = 60 → 
  new_weight = old_weight + num_persons * avg_increase → new_weight = 80 :=
  by
    intros
    sorry

end weight_of_new_person_l41_41231


namespace find_FC_l41_41543

-- Define all given values and relationships
variables (DC CB AD AB ED FC : ℝ)
variables (h1 : DC = 9) (h2 : CB = 6)
variables (h3 : AB = (1/3) * AD)
variables (h4 : ED = (2/3) * AD)

-- Define the goal
theorem find_FC :
  FC = 9 :=
sorry

end find_FC_l41_41543


namespace tickets_per_box_l41_41525

-- Definitions
def boxes (G: Type) : ℕ := 9
def total_tickets (G: Type) : ℕ := 45

-- Theorem statement
theorem tickets_per_box (G: Type) : total_tickets G / boxes G = 5 :=
by
  sorry

end tickets_per_box_l41_41525


namespace players_count_l41_41703

def total_socks : ℕ := 22
def socks_per_player : ℕ := 2

theorem players_count : total_socks / socks_per_player = 11 :=
by
  sorry

end players_count_l41_41703


namespace length_of_parallel_at_60N_l41_41963

noncomputable def parallel_length (R : ℝ) (lat_deg : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos (Real.pi * lat_deg / 180)

theorem length_of_parallel_at_60N :
  parallel_length 20 60 = 20 * Real.pi :=
by
  sorry

end length_of_parallel_at_60N_l41_41963


namespace find_common_ratio_l41_41507

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_common_ratio (a1 a4 q : ℝ) (hq : q ^ 3 = 8) (ha1 : a1 = 8) (ha4 : a4 = 64)
  (a_def : is_geometric_sequence (fun n => a1 * q ^ n) q) :
  q = 2 :=
by
  sorry

end find_common_ratio_l41_41507


namespace mean_of_two_equals_mean_of_three_l41_41173

theorem mean_of_two_equals_mean_of_three (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → 
  z = 25 / 3 := 
by 
  sorry

end mean_of_two_equals_mean_of_three_l41_41173


namespace cosine_double_angle_l41_41252

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l41_41252


namespace num_2_edge_paths_l41_41031

-- Let T be a tetrahedron with vertices connected such that each vertex has exactly 3 edges.
-- Prove that the number of distinct 2-edge paths from a starting vertex P to an ending vertex Q is 3.

def tetrahedron : Type := ℕ -- This is a simplified representation of vertices

noncomputable def edges (a b : tetrahedron) : Prop := true -- Each pair of distinct vertices is an edge in a tetrahedron

theorem num_2_edge_paths (P Q : tetrahedron) (hP : P ≠ Q) : 
  -- There are 3 distinct 2-edge paths from P to Q  
  ∃ (paths : Finset (tetrahedron × tetrahedron)), 
    paths.card = 3 ∧ 
    ∀ (p : tetrahedron × tetrahedron), p ∈ paths → 
      edges P p.1 ∧ edges p.1 p.2 ∧ p.2 = Q :=
by 
  sorry

end num_2_edge_paths_l41_41031


namespace inequality_negative_solution_l41_41542

theorem inequality_negative_solution (a : ℝ) (h : a ≥ -17/4 ∧ a < 4) : 
  ∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a| :=
by
  sorry

end inequality_negative_solution_l41_41542


namespace polynomial_divisibility_by_5_l41_41298

theorem polynomial_divisibility_by_5
  (a b c d : ℤ)
  (divisible : ∀ x : ℤ, 5 ∣ (a * x ^ 3 + b * x ^ 2 + c * x + d)) :
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d :=
sorry

end polynomial_divisibility_by_5_l41_41298


namespace boys_number_is_60_l41_41962

-- Definitions based on the conditions
variables (x y : ℕ)

def sum_boys_girls (x y : ℕ) : Prop := 
  x + y = 150

def girls_percentage (x y : ℕ) : Prop := 
  y = (x * 150) / 100

-- Prove that the number of boys equals 60
theorem boys_number_is_60 (x y : ℕ) 
  (h1 : sum_boys_girls x y) 
  (h2 : girls_percentage x y) : 
  x = 60 := by
  sorry

end boys_number_is_60_l41_41962


namespace tyson_age_l41_41125

noncomputable def age_proof : Prop :=
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  t = 20           -- Statement that needs to be proved

theorem tyson_age : age_proof :=
by
  let k := 25      -- Kyle's age
  let j := k - 5   -- Julian's age
  let f := j + 20  -- Frederick's age
  let t := f / 2   -- Tyson's age
  show t = 20
  sorry

end tyson_age_l41_41125


namespace area_of_plot_l41_41806

def cm_to_miles (a : ℕ) : ℕ := a * 9

def miles_to_acres (b : ℕ) : ℕ := b * 640

theorem area_of_plot :
  let bottom := 12
  let top := 18
  let height := 10
  let area_cm2 := ((bottom + top) * height) / 2
  let area_miles2 := cm_to_miles area_cm2
  let area_acres := miles_to_acres area_miles2
  area_acres = 864000 :=
by
  sorry

end area_of_plot_l41_41806


namespace problem1_problem2_l41_41948

noncomputable def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

noncomputable def g' (x : ℝ) : ℝ := 15 * x^4 - 16 * x^3 + 6 * x^2 - 56 * x + 15

theorem problem1 : g 6 = 17568 := 
by {
  sorry
}

theorem problem2 : g' 6 = 15879 := 
by {
  sorry
}

end problem1_problem2_l41_41948


namespace sum_of_S_values_l41_41388

noncomputable def a : ℕ := 32
noncomputable def b1 : ℕ := 16 -- When M = 73
noncomputable def c : ℕ := 25
noncomputable def b2 : ℕ := 89 -- When M = 146
noncomputable def x1 : ℕ := 14 -- When M = 73
noncomputable def x2 : ℕ := 7 -- When M = 146
noncomputable def y1 : ℕ := 3 -- When M = 73
noncomputable def y2 : ℕ := 54 -- When M = 146
noncomputable def z1 : ℕ := 8 -- When M = 73
noncomputable def z2 : ℕ := 4 -- When M = 146

theorem sum_of_S_values :
  let M1 := a + b1 + c
  let M2 := a + b2 + c
  let S1 := M1 + x1 + y1 + z1
  let S2 := M2 + x2 + y2 + z2
  (S1 = 98) ∧ (S2 = 211) ∧ (S1 + S2 = 309) := by
  sorry

end sum_of_S_values_l41_41388


namespace Trevor_future_age_when_brother_is_three_times_now_l41_41764

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end Trevor_future_age_when_brother_is_three_times_now_l41_41764


namespace unique_n_for_given_divisors_l41_41563

theorem unique_n_for_given_divisors :
  ∃! (n : ℕ), 
    ∀ (k : ℕ) (d : ℕ → ℕ), 
      k ≥ 22 ∧ 
      d 1 = 1 ∧ d k = n ∧ 
      (∀ i j, i < j → d i < d j) ∧ 
      (d 7) ^ 2 + (d 10) ^ 2 = (n / d 22) ^ 2 →
      n = 2^3 * 3 * 5 * 17 :=
sorry

end unique_n_for_given_divisors_l41_41563


namespace updated_mean_l41_41202

theorem updated_mean
  (n : ℕ) (obs_mean : ℝ) (decrement : ℝ)
  (h1 : n = 50) (h2 : obs_mean = 200) (h3 : decrement = 47) :
  (obs_mean - decrement) = 153 := by
  sorry

end updated_mean_l41_41202


namespace projective_transformation_is_cross_ratio_preserving_l41_41062

theorem projective_transformation_is_cross_ratio_preserving (P : ℝ → ℝ) :
  (∃ a b c d : ℝ, (ad - bc ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))) ↔
  (∀ x1 x2 x3 x4 : ℝ, (x1 - x3) * (x2 - x4) / ((x1 - x4) * (x2 - x3)) =
       (P x1 - P x3) * (P x2 - P x4) / ((P x1 - P x4) * (P x2 - P x3))) :=
sorry

end projective_transformation_is_cross_ratio_preserving_l41_41062


namespace minimum_bench_sections_l41_41311

theorem minimum_bench_sections (N : ℕ) (hN : 8 * N = 12 * N) : N = 3 :=
sorry

end minimum_bench_sections_l41_41311


namespace landscape_breadth_l41_41229

theorem landscape_breadth (L B : ℕ)
  (h1 : B = 6 * L)
  (h2 : 4200 = (1 / 7 : ℚ) * 6 * L^2) :
  B = 420 := 
  sorry

end landscape_breadth_l41_41229


namespace exists_midpoint_with_integer_coordinates_l41_41218

theorem exists_midpoint_with_integer_coordinates (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ ((points i).1 + (points j).1) % 2 = 0 ∧ ((points i).2 + (points j).2) % 2 = 0 :=
by
  sorry

end exists_midpoint_with_integer_coordinates_l41_41218


namespace solve_quadratic1_solve_quadratic2_l41_41237

theorem solve_quadratic1 :
  (∀ x, x^2 + x - 4 = 0 → x = ( -1 + Real.sqrt 17 ) / 2 ∨ x = ( -1 - Real.sqrt 17 ) / 2) := sorry

theorem solve_quadratic2 :
  (∀ x, (2*x + 1)^2 + 15 = 8*(2*x + 1) → x = 1 ∨ x = 2) := sorry

end solve_quadratic1_solve_quadratic2_l41_41237


namespace sum_largest_smallest_ABC_l41_41161

def hundreds (n : ℕ) : ℕ := n / 100
def units (n : ℕ) : ℕ := n % 10
def tens (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_largest_smallest_ABC : 
  (hundreds 297 = 2) ∧ (units 297 = 7) ∧ (hundreds 207 = 2) ∧ (units 207 = 7) →
  (297 + 207 = 504) :=
by
  sorry

end sum_largest_smallest_ABC_l41_41161


namespace factor_expression_1_factor_expression_2_l41_41268

theorem factor_expression_1 (a b c : ℝ) : a^2 + 2 * a * b + b^2 + a * c + b * c = (a + b) * (a + b + c) :=
  sorry

theorem factor_expression_2 (a x y : ℝ) : 4 * a^2 - x^2 + 4 * x * y - 4 * y^2 = (2 * a + x - 2 * y) * (2 * a - x + 2 * y) :=
  sorry

end factor_expression_1_factor_expression_2_l41_41268


namespace cory_needs_22_weeks_l41_41159

open Nat

def cory_birthday_money : ℕ := 100 + 45 + 20
def bike_cost : ℕ := 600
def weekly_earning : ℕ := 20

theorem cory_needs_22_weeks : ∃ x : ℕ, cory_birthday_money + x * weekly_earning ≥ bike_cost ∧ x = 22 := by
  sorry

end cory_needs_22_weeks_l41_41159


namespace john_total_payment_in_month_l41_41211

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l41_41211


namespace area_per_cabbage_is_one_l41_41153

noncomputable def area_per_cabbage (x y : ℕ) : ℕ :=
  let num_cabbages_this_year : ℕ := 10000
  let increase_in_cabbages : ℕ := 199
  let area_this_year : ℕ := y^2
  let area_last_year : ℕ := x^2
  let area_per_cabbage : ℕ := area_this_year / num_cabbages_this_year
  area_per_cabbage

theorem area_per_cabbage_is_one (x y : ℕ) (hx : y^2 = 10000) (hy : y^2 = x^2 + 199) : area_per_cabbage x y = 1 :=
by 
  sorry

end area_per_cabbage_is_one_l41_41153


namespace hair_cut_length_l41_41471

-- Definitions corresponding to the conditions in the problem
def initial_length : ℕ := 18
def current_length : ℕ := 9

-- Statement to prove
theorem hair_cut_length : initial_length - current_length = 9 :=
by
  sorry

end hair_cut_length_l41_41471


namespace bananas_count_l41_41149

theorem bananas_count
    (total_fruit : ℕ)
    (apples_ratio : ℕ)
    (persimmons_ratio : ℕ)
    (apples_and_persimmons : apples_ratio * bananas + persimmons_ratio * bananas = total_fruit)
    (apples_ratio_val : apples_ratio = 4)
    (persimmons_ratio_val : persimmons_ratio = 3)
    (total_fruit_value : total_fruit = 210) :
    bananas = 30 :=
by
  sorry

end bananas_count_l41_41149


namespace infinite_arith_prog_contains_infinite_nth_powers_l41_41843

theorem infinite_arith_prog_contains_infinite_nth_powers
  (a d : ℕ) (n : ℕ) 
  (h_pos: 0 < d) 
  (h_power: ∃ k : ℕ, ∃ m : ℕ, a + k * d = m^n) :
  ∃ infinitely_many k : ℕ, ∃ m : ℕ, a + k * d = m^n :=
sorry

end infinite_arith_prog_contains_infinite_nth_powers_l41_41843


namespace meal_combinations_l41_41569

theorem meal_combinations :
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  meats * vegetable_combinations * desserts = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  show meats * vegetable_combinations * desserts = 150
  sorry

end meal_combinations_l41_41569


namespace jade_transactions_l41_41859

-- Definitions for each condition
def transactions_mabel : ℕ := 90
def transactions_anthony : ℕ := transactions_mabel + (transactions_mabel / 10)
def transactions_cal : ℕ := 2 * transactions_anthony / 3
def transactions_jade : ℕ := transactions_cal + 17

-- The theorem stating that Jade handled 83 transactions
theorem jade_transactions : transactions_jade = 83 := by
  sorry

end jade_transactions_l41_41859


namespace project_hours_l41_41636

variable (K : ℕ)

theorem project_hours 
    (h_total : K + 2 * K + 3 * K + K / 2 = 180)
    (h_k_nearest : K = 28) :
    3 * K - K = 56 := 
by
  -- Proof goes here
  sorry

end project_hours_l41_41636


namespace expected_winnings_correct_l41_41092

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 1 then 0
  else if roll % 4 = 0 then 2 * roll
  else roll

def expected_winnings : ℚ :=
  (winnings 1) / 8 + (winnings 2) / 8 +
  (winnings 3) / 8 + (winnings 4) / 8 +
  (winnings 5) / 8 + (winnings 6) / 8 +
  (winnings 7) / 8 + (winnings 8) / 8

theorem expected_winnings_correct : expected_winnings = 3.75 := by 
  sorry

end expected_winnings_correct_l41_41092


namespace range_of_a_l41_41656

-- Definitions based on conditions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 4

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → f a x ≤ f a 4) → a ≤ -3 :=
by
  sorry

end range_of_a_l41_41656


namespace polynomial_transformation_l41_41860

theorem polynomial_transformation :
  ∀ (a h k : ℝ), (8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → a + h + k = -23.5 :=
by
  intros a h k h_eq
  sorry

end polynomial_transformation_l41_41860


namespace quadratic_function_n_neg_l41_41248

theorem quadratic_function_n_neg (n : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + n = 0 → x > 0) → n < 0 :=
by
  sorry

end quadratic_function_n_neg_l41_41248


namespace ratio_future_age_l41_41124

variables (S : ℕ) (M : ℕ) (S_future : ℕ) (M_future : ℕ)

def son_age := 44
def man_age := son_age + 46
def son_age_future := son_age + 2
def man_age_future := man_age + 2

theorem ratio_future_age : man_age_future / son_age_future = 2 := by
  -- You can add the proof here if you want
  sorry

end ratio_future_age_l41_41124


namespace triangle_is_right_triangle_l41_41930

theorem triangle_is_right_triangle 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : b - a * Real.cos B = a * Real.cos C - c) 
  (h₂ : ∀ (angle : ℝ), 0 < angle ∧ angle < π) : A = π / 2 := 
sorry

end triangle_is_right_triangle_l41_41930


namespace discarded_marble_weight_l41_41302

-- Define the initial weight of the marble block and the weights of the statues
def initial_weight : ℕ := 80
def weight_statue_1 : ℕ := 10
def weight_statue_2 : ℕ := 18
def weight_statue_3 : ℕ := 15
def weight_statue_4 : ℕ := 15

-- The proof statement: the discarded weight of marble is 22 pounds.
theorem discarded_marble_weight :
  initial_weight - (weight_statue_1 + weight_statue_2 + weight_statue_3 + weight_statue_4) = 22 :=
by
  sorry

end discarded_marble_weight_l41_41302


namespace range_of_a_l41_41832

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 = 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + a > 0

theorem range_of_a (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ≥ 2 :=
sorry

end range_of_a_l41_41832


namespace value_of_f_at_13_over_2_l41_41877

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_at_13_over_2
  (h1 : ∀ x : ℝ , f (-x) = -f (x))
  (h2 : ∀ x : ℝ , f (x - 2) = f (x + 2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = -x^2) :
  f (13 / 2) = 9 / 4 :=
sorry

end value_of_f_at_13_over_2_l41_41877


namespace new_perimeter_after_adding_tiles_l41_41235

-- Define the original condition as per the problem statement
def original_T_shape (n : ℕ) : Prop :=
  n = 6

def original_perimeter (p : ℕ) : Prop :=
  p = 12

-- Define hypothesis required to add three more tiles while sharing a side with existing tiles
def add_three_tiles_with_shared_side (original_tiles : ℕ) (new_tiles_added : ℕ) : Prop :=
  original_tiles + new_tiles_added = 9

-- Prove the new perimeter after adding three tiles to the original T-shaped figure
theorem new_perimeter_after_adding_tiles
  (n : ℕ) (p : ℕ) (new_tiles : ℕ) (new_p : ℕ)
  (h1 : original_T_shape n)
  (h2 : original_perimeter p)
  (h3 : add_three_tiles_with_shared_side n new_tiles)
  : new_p = 16 :=
sorry

end new_perimeter_after_adding_tiles_l41_41235


namespace largest_integer_square_two_digits_l41_41998

theorem largest_integer_square_two_digits : 
  ∃ M : ℤ, (M * M ≥ 10 ∧ M * M < 100) ∧ (∀ x : ℤ, (x * x ≥ 10 ∧ x * x < 100) → x ≤ M) ∧ M = 9 := 
by
  sorry

end largest_integer_square_two_digits_l41_41998


namespace evaluate_expression_l41_41878

theorem evaluate_expression : 
  (2 ^ 2015 + 2 ^ 2013 + 2 ^ 2011) / (2 ^ 2015 - 2 ^ 2013 + 2 ^ 2011) = 21 / 13 := 
by 
 sorry

end evaluate_expression_l41_41878


namespace young_or_old_woman_lawyer_probability_l41_41508

/-- 
40 percent of the members of a study group are women.
Among these women, 30 percent are young lawyers.
10 percent are old lawyers.
Prove the probability that a member randomly selected is a young or old woman lawyer is 0.16.
-/
theorem young_or_old_woman_lawyer_probability :
  let total_members := 100
  let women_percentage := 40
  let young_lawyers_percentage := 30
  let old_lawyers_percentage := 10
  let total_women := (women_percentage * total_members) / 100
  let young_women_lawyers := (young_lawyers_percentage * total_women) / 100
  let old_women_lawyers := (old_lawyers_percentage * total_women) / 100
  let women_lawyers := young_women_lawyers + old_women_lawyers
  let probability := women_lawyers / total_members
  probability = 0.16 := 
by {
  sorry
}

end young_or_old_woman_lawyer_probability_l41_41508


namespace number_of_australians_l41_41303

-- Conditions are given here as definitions
def total_people : ℕ := 49
def number_americans : ℕ := 16
def number_chinese : ℕ := 22

-- Goal is to prove the number of Australians is 11
theorem number_of_australians : total_people - (number_americans + number_chinese) = 11 := by
  sorry

end number_of_australians_l41_41303


namespace factorize_quartic_l41_41250

-- Specify that p and q are real numbers (ℝ)
variables {p q : ℝ}

-- Statement: For any real numbers p and q, the polynomial x^4 + p x^2 + q can always be factored into two quadratic polynomials.
theorem factorize_quartic (p q : ℝ) : 
  ∃ a b c d e f : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + p * x^2 + q :=
sorry

end factorize_quartic_l41_41250


namespace least_multiple_of_25_gt_475_l41_41421

theorem least_multiple_of_25_gt_475 : ∃ n : ℕ, n > 475 ∧ n % 25 = 0 ∧ ∀ m : ℕ, (m > 475 ∧ m % 25 = 0) → n ≤ m := 
  sorry

end least_multiple_of_25_gt_475_l41_41421


namespace rectangle_area_l41_41594

theorem rectangle_area (a b k : ℕ)
  (h1 : k = 6 * (a + b) + 36)
  (h2 : k = 114)
  (h3 : a / b = 8 / 5) :
  a * b = 40 :=
by {
  sorry
}

end rectangle_area_l41_41594


namespace sum_of_possible_values_of_x_l41_41775

theorem sum_of_possible_values_of_x :
  let sq_side := (x - 4)
  let rect_length := (x - 5)
  let rect_width := (x + 6)
  let sq_area := (sq_side)^2
  let rect_area := rect_length * rect_width
  (3 * (sq_area) = rect_area) → ∃ (x1 x2 : ℝ), (3 * (x1 - 4) ^ 2 = (x1 - 5) * (x1 + 6)) ∧ (3 * (x2 - 4) ^ 2 = (x2 - 5) * (x2 + 6)) ∧ (x1 + x2 = 12.5) := 
by
  sorry

end sum_of_possible_values_of_x_l41_41775


namespace expand_product_l41_41242

-- Define the problem
theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 :=
by
  sorry

end expand_product_l41_41242


namespace triangle_perimeter_correct_l41_41523

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
    a + b + c

theorem triangle_perimeter_correct (a b c : ℕ) (h1 : a = b - 1) (h2 : b = c - 1) (h3 : c = 2 * a) : triangle_perimeter a b c = 15 :=
    sorry

end triangle_perimeter_correct_l41_41523


namespace second_investment_amount_l41_41204

/-
A $500 investment and another investment have a combined yearly return of 8.5 percent of the total of the two investments.
The $500 investment has a yearly return of 7 percent.
The other investment has a yearly return of 9 percent.
Prove that the amount of the second investment is $1500.
-/

theorem second_investment_amount :
  ∃ x : ℝ, 35 + 0.09 * x = 0.085 * (500 + x) → x = 1500 :=
by
  sorry

end second_investment_amount_l41_41204


namespace quadratic_inequality_solution_l41_41920

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 :=
by {
  sorry
}

end quadratic_inequality_solution_l41_41920


namespace measure_of_angle_x_l41_41016

-- Given conditions
def angle_ABC : ℝ := 120
def angle_BAD : ℝ := 31
def angle_BDA (x : ℝ) : Prop := x + 60 + 31 = 180 

-- Statement to prove
theorem measure_of_angle_x : 
  ∃ x : ℝ, angle_BDA x → x = 89 :=
by
  sorry

end measure_of_angle_x_l41_41016


namespace shortest_hypotenuse_max_inscribed_circle_radius_l41_41139

variable {a b c r : ℝ}

-- Condition 1: The perimeter of the right-angled triangle is 1 meter.
def perimeter_condition (a b : ℝ) : Prop :=
  a + b + Real.sqrt (a^2 + b^2) = 1

-- Problem 1: Prove the shortest length of the hypotenuse is √2 - 1.
theorem shortest_hypotenuse (a b : ℝ) (h : perimeter_condition a b) :
  Real.sqrt (a^2 + b^2) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the maximum value of the inscribed circle radius is 3/2 - √2.
theorem max_inscribed_circle_radius (a b r : ℝ) (h : perimeter_condition a b) :
  (a * b = r) → r = 3/2 - Real.sqrt 2 :=
sorry

end shortest_hypotenuse_max_inscribed_circle_radius_l41_41139


namespace ratio_of_potatoes_l41_41615

-- Definitions as per conditions
def initial_potatoes : ℕ := 300
def given_to_gina : ℕ := 69
def remaining_potatoes : ℕ := 47
def k : ℕ := 2  -- Identify k is 2 based on the ratio

-- Calculate given_to_tom and total given away
def given_to_tom : ℕ := k * given_to_gina
def given_to_anne : ℕ := given_to_tom / 3

-- Arithmetical conditions derived from the problem
def total_given_away : ℕ := given_to_gina + given_to_tom + given_to_anne + remaining_potatoes

-- Proof statement to show the ratio between given_to_tom and given_to_gina is 2
theorem ratio_of_potatoes :
  k = 2 → total_given_away = initial_potatoes → given_to_tom / given_to_gina = 2 := by
  intros h1 h2
  sorry

end ratio_of_potatoes_l41_41615


namespace abs_expression_value_l41_41748

theorem abs_expression_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs (abs x - 2 * x) - abs x) - x) = 6069 :=
by sorry

end abs_expression_value_l41_41748


namespace rahim_average_price_per_book_l41_41433

noncomputable section

open BigOperators

def store_A_price_per_book : ℝ := 
  let original_total := 1600
  let discount := original_total * 0.15
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.05
  let final_total := discounted_total + sales_tax
  final_total / 25

def store_B_price_per_book : ℝ := 
  let original_total := 3200
  let effective_books_paid := 35 - (35 / 4)
  original_total / effective_books_paid

def store_C_price_per_book : ℝ := 
  let original_total := 3800
  let discount := 0.10 * (4 * (original_total / 40))
  let discounted_total := original_total - discount
  let service_charge := discounted_total * 0.07
  let final_total := discounted_total + service_charge
  final_total / 40

def store_D_price_per_book : ℝ := 
  let original_total := 2400
  let discount := 0.50 * (original_total / 30)
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.06
  let final_total := discounted_total + sales_tax
  final_total / 30

def store_E_price_per_book : ℝ := 
  let original_total := 1800
  let discount := original_total * 0.08
  let discounted_total := original_total - discount
  let additional_fee := discounted_total * 0.04
  let final_total := discounted_total + additional_fee
  final_total / 20

def total_books : ℝ := 25 + 35 + 40 + 30 + 20

def total_amount : ℝ := 
  store_A_price_per_book * 25 + 
  store_B_price_per_book * 35 + 
  store_C_price_per_book * 40 + 
  store_D_price_per_book * 30 + 
  store_E_price_per_book * 20

def average_price_per_book : ℝ := total_amount / total_books

theorem rahim_average_price_per_book : average_price_per_book = 85.85 :=
sorry

end rahim_average_price_per_book_l41_41433


namespace seventh_observation_l41_41578

-- Declare the conditions with their definitions
def average_of_six (sum6 : ℕ) : Prop := sum6 = 6 * 14
def new_average_decreased (sum6 sum7 : ℕ) : Prop := sum7 = sum6 + 7 ∧ 13 = (sum6 + 7) / 7

-- The main statement to prove that the seventh observation is 7
theorem seventh_observation (sum6 sum7 : ℕ) (h_avg6 : average_of_six sum6) (h_new_avg : new_average_decreased sum6 sum7) :
  sum7 - sum6 = 7 := 
  sorry

end seventh_observation_l41_41578


namespace real_solutions_count_l41_41663

noncomputable def number_of_real_solutions : ℕ := 2

theorem real_solutions_count (x : ℝ) :
  (x^2 - 5)^2 = 36 → number_of_real_solutions = 2 := by
  sorry

end real_solutions_count_l41_41663


namespace unit_vector_perpendicular_l41_41784

theorem unit_vector_perpendicular (x y : ℝ)
  (h1 : 4 * x + 2 * y = 0) 
  (h2 : x^2 + y^2 = 1) :
  (x = (Real.sqrt 5) / 5 ∧ y = -(2 * (Real.sqrt 5) / 5)) ∨ 
  (x = -(Real.sqrt 5) / 5 ∧ y = 2 * (Real.sqrt 5) / 5) :=
sorry

end unit_vector_perpendicular_l41_41784


namespace digit_start_l41_41039

theorem digit_start (a n p q : ℕ) (hp : a * 10^p < 2^n) (hq : 2^n < (a + 1) * 10^p)
  (hr : a * 10^q < 5^n) (hs : 5^n < (a + 1) * 10^q) :
  a = 3 :=
by
  -- The proof goes here.
  sorry

end digit_start_l41_41039


namespace gcd_6051_10085_l41_41230

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end gcd_6051_10085_l41_41230


namespace expected_value_of_win_is_162_l41_41106

noncomputable def expected_value_of_win : ℝ :=
  (1/8) * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3)

theorem expected_value_of_win_is_162 : expected_value_of_win = 162 := 
by 
  sorry

end expected_value_of_win_is_162_l41_41106


namespace minimum_value_of_f_l41_41501

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2) + 2 * x

theorem minimum_value_of_f (h : ∀ x > 0, f x ≥ 3) : ∃ x, x > 0 ∧ f x = 3 :=
by
  sorry

end minimum_value_of_f_l41_41501


namespace minimum_grade_Ahmed_l41_41185

theorem minimum_grade_Ahmed (assignments : ℕ) (Ahmed_grade : ℕ) (Emily_grade : ℕ) (final_assignment_grade_Emily : ℕ) 
  (sum_grades_Emily : ℕ) (sum_grades_Ahmed : ℕ) (total_points_Ahmed : ℕ) (total_points_Emily : ℕ) :
  assignments = 9 →
  Ahmed_grade = 91 →
  Emily_grade = 92 →
  final_assignment_grade_Emily = 90 →
  sum_grades_Emily = 828 →
  sum_grades_Ahmed = 819 →
  total_points_Ahmed = sum_grades_Ahmed + 100 →
  total_points_Emily = sum_grades_Emily + final_assignment_grade_Emily →
  total_points_Ahmed > total_points_Emily :=
by
  sorry

end minimum_grade_Ahmed_l41_41185


namespace max_elements_in_set_l41_41155

theorem max_elements_in_set (S : Finset ℕ) (hS : ∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → 
  ∃ (k : ℕ) (c d : ℕ), c < d ∧ c ∈ S ∧ d ∈ S ∧ a + b = c^k * d) :
  S.card ≤ 48 :=
sorry

end max_elements_in_set_l41_41155


namespace part1_part2_l41_41906

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l41_41906


namespace regular_polygon_sides_l41_41491

theorem regular_polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : 
  n = 6 :=
sorry

end regular_polygon_sides_l41_41491


namespace pastrami_sandwich_cost_l41_41986

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l41_41986


namespace employed_females_percentage_l41_41678

theorem employed_females_percentage (total_employed_percentage employed_males_percentage employed_females_percentage : ℝ) 
    (h1 : total_employed_percentage = 64) 
    (h2 : employed_males_percentage = 48) 
    (h3 : employed_females_percentage = total_employed_percentage - employed_males_percentage) :
    (employed_females_percentage / total_employed_percentage * 100) = 25 :=
by
  sorry

end employed_females_percentage_l41_41678


namespace max_value_HMMT_l41_41582

theorem max_value_HMMT :
  ∀ (H M T : ℤ), H * M ^ 2 * T = H + 2 * M + T → H * M ^ 2 * T ≤ 8 :=
by
  sorry

end max_value_HMMT_l41_41582


namespace probability_different_colors_l41_41190

theorem probability_different_colors
  (red_chips green_chips : ℕ)
  (total_chips : red_chips + green_chips = 10)
  (prob_red : ℚ := red_chips / 10)
  (prob_green : ℚ := green_chips / 10) :
  ((prob_red * prob_green) + (prob_green * prob_red) = 12 / 25) := by
sorry

end probability_different_colors_l41_41190


namespace option_C_is_correct_l41_41795

theorem option_C_is_correct (a b c : ℝ) (h : a > b) : c - a < c - b := 
by
  linarith

end option_C_is_correct_l41_41795


namespace large_envelopes_count_l41_41874

theorem large_envelopes_count
  (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ)
  (H1 : total_letters = 80)
  (H2 : small_envelope_letters = 20)
  (H3 : letters_per_large_envelope = 2) :
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 :=
sorry

end large_envelopes_count_l41_41874


namespace transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l41_41499

-- Define the polynomial transformation
def transform_polynomial (x : ℝ) : ℝ := x^2 + 8 * x - 1

-- Transformation problem
theorem transform_polynomial_eq (x m n : ℝ) :
  (x + 4)^2 - 17 = transform_polynomial x := 
sorry

-- Define the polynomial for correction
def factor_polynomial (x : ℝ) : ℝ := x^2 - 3 * x - 40

-- Factoring correction problem
theorem correct_factorization (x : ℝ) :
  factor_polynomial x = (x + 5) * (x - 8) := 
sorry

-- Define the polynomial for the positivity proof
def positive_polynomial (x y : ℝ) : ℝ := x^2 + y^2 - 2 * x - 4 * y + 16

-- Positive polynomial proof
theorem positive_polynomial_gt_zero (x y : ℝ) :
  positive_polynomial x y > 0 := 
sorry

end transform_polynomial_eq_correct_factorization_positive_polynomial_gt_zero_l41_41499


namespace peter_contains_five_l41_41413

theorem peter_contains_five (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → ∃ i : ℕ, 5 ≤ 10^i * (N * 5^m / 10^i) % 10 :=
sorry

end peter_contains_five_l41_41413


namespace solve_cubic_eq_l41_41178

theorem solve_cubic_eq (x : ℝ) (h1 : (x + 1)^3 = x^3) (h2 : 0 ≤ x) (h3 : x < 1) : x = 0 :=
by
  sorry

end solve_cubic_eq_l41_41178


namespace plates_used_l41_41224

def plates_per_course : ℕ := 2
def courses_breakfast : ℕ := 2
def courses_lunch : ℕ := 2
def courses_dinner : ℕ := 3
def courses_late_snack : ℕ := 3
def courses_per_day : ℕ := courses_breakfast + courses_lunch + courses_dinner + courses_late_snack
def plates_per_day : ℕ := courses_per_day * plates_per_course

def parents_and_siblings_stay : ℕ := 6
def grandparents_stay : ℕ := 4
def cousins_stay : ℕ := 3

def parents_and_siblings_count : ℕ := 5
def grandparents_count : ℕ := 2
def cousins_count : ℕ := 4

def plates_parents_and_siblings : ℕ := parents_and_siblings_count * plates_per_day * parents_and_siblings_stay
def plates_grandparents : ℕ := grandparents_count * plates_per_day * grandparents_stay
def plates_cousins : ℕ := cousins_count * plates_per_day * cousins_stay

def total_plates_used : ℕ := plates_parents_and_siblings + plates_grandparents + plates_cousins

theorem plates_used (expected : ℕ) : total_plates_used = expected :=
by
  sorry

end plates_used_l41_41224


namespace simplify_and_evaluate_sqrt_log_product_property_l41_41325

-- Problem I
theorem simplify_and_evaluate_sqrt (a : ℝ) (h : 0 < a) : 
  Real.sqrt (a^(1/4) * Real.sqrt (a * Real.sqrt a)) = Real.sqrt a := 
by
  sorry

-- Problem II
theorem log_product_property : 
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := 
by
  sorry

end simplify_and_evaluate_sqrt_log_product_property_l41_41325


namespace alice_bob_meet_l41_41136

theorem alice_bob_meet (n : ℕ) (h_n : n = 18) (alice_move : ℕ) (bob_move : ℕ)
  (h_alice : alice_move = 7) (h_bob : bob_move = 13) :
  ∃ k : ℕ, alice_move * k % n = (n - bob_move) * k % n :=
by
  sorry

end alice_bob_meet_l41_41136


namespace larger_segment_length_l41_41099

theorem larger_segment_length 
  (x y : ℝ)
  (h1 : 40^2 = x^2 + y^2)
  (h2 : 90^2 = (110 - x)^2 + y^2) :
  110 - x = 84.55 :=
by
  sorry

end larger_segment_length_l41_41099


namespace maddox_more_profit_than_theo_l41_41921

-- Definitions (conditions)
def cost_per_camera : ℕ := 20
def num_cameras : ℕ := 3
def total_cost : ℕ := num_cameras * cost_per_camera

def maddox_selling_price_per_camera : ℕ := 28
def theo_selling_price_per_camera : ℕ := 23

-- Total selling price
def maddox_total_selling_price : ℕ := num_cameras * maddox_selling_price_per_camera
def theo_total_selling_price : ℕ := num_cameras * theo_selling_price_per_camera

-- Profits
def maddox_profit : ℕ := maddox_total_selling_price - total_cost
def theo_profit : ℕ := theo_total_selling_price - total_cost

-- Proof Statement
theorem maddox_more_profit_than_theo : maddox_profit - theo_profit = 15 := by
  sorry

end maddox_more_profit_than_theo_l41_41921


namespace h_of_neg2_eq_11_l41_41306

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg2_eq_11 : h (-2) = 11 := by
  sorry

end h_of_neg2_eq_11_l41_41306


namespace total_books_in_bookcase_l41_41746

def num_bookshelves := 8
def num_layers_per_bookshelf := 5
def books_per_layer := 85

theorem total_books_in_bookcase : 
  (num_bookshelves * num_layers_per_bookshelf * books_per_layer) = 3400 := by
  sorry

end total_books_in_bookcase_l41_41746


namespace james_huskies_count_l41_41120

theorem james_huskies_count 
  (H : ℕ) 
  (pitbulls : ℕ := 2) 
  (golden_retrievers : ℕ := 4) 
  (husky_pups_per_husky : ℕ := 3) 
  (pitbull_pups_per_pitbull : ℕ := 3) 
  (extra_pups_per_golden_retriever : ℕ := 2) 
  (pup_difference : ℕ := 30) :
  H + pitbulls + golden_retrievers + pup_difference = 3 * H + pitbulls * pitbull_pups_per_pitbull + golden_retrievers * (husky_pups_per_husky + extra_pups_per_golden_retriever) :=
sorry

end james_huskies_count_l41_41120


namespace wire_length_before_cutting_l41_41486

theorem wire_length_before_cutting (L S : ℝ) (h1 : S = 40) (h2 : S = 2 / 5 * L) : L + S = 140 :=
by
  sorry

end wire_length_before_cutting_l41_41486


namespace single_elimination_games_l41_41375

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l41_41375


namespace is_linear_equation_with_one_var_l41_41146

-- Definitions
def eqA := ∀ (x : ℝ), x^2 + 1 = 5
def eqB := ∀ (x y : ℝ), x + 2 = y - 3
def eqC := ∀ (x : ℝ), 1 / (2 * x) = 10
def eqD := ∀ (x : ℝ), x = 4

-- Theorem stating which equation represents a linear equation in one variable
theorem is_linear_equation_with_one_var : eqD :=
by
  -- Proof skipped
  sorry

end is_linear_equation_with_one_var_l41_41146


namespace glycerin_solution_l41_41445

theorem glycerin_solution (x : ℝ) :
    let total_volume := 100
    let final_glycerin_percentage := 0.75
    let volume_first_solution := 75
    let volume_second_solution := 75
    let second_solution_percentage := 0.90
    let final_glycerin_volume := final_glycerin_percentage * total_volume
    let glycerin_second_solution := second_solution_percentage * volume_second_solution
    let glycerin_first_solution := x * volume_first_solution / 100
    glycerin_first_solution + glycerin_second_solution = final_glycerin_volume →
    x = 10 :=
by
    sorry

end glycerin_solution_l41_41445


namespace grant_earnings_l41_41355

theorem grant_earnings 
  (baseball_cards_sale : ℕ) 
  (baseball_bat_sale : ℕ) 
  (baseball_glove_price : ℕ) 
  (baseball_glove_discount : ℕ) 
  (baseball_cleats_sale : ℕ) : 
  baseball_cards_sale + baseball_bat_sale + (baseball_glove_price - baseball_glove_discount) + 2 * baseball_cleats_sale = 79 :=
by
  let baseball_cards_sale := 25
  let baseball_bat_sale := 10
  let baseball_glove_price := 30
  let baseball_glove_discount := (30 * 20) / 100
  let baseball_cleats_sale := 10
  sorry

end grant_earnings_l41_41355


namespace maximize_daily_profit_l41_41740

noncomputable def daily_profit : ℝ → ℝ → ℝ
| x, c => if h : 0 < x ∧ x ≤ c then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x)) else 0

theorem maximize_daily_profit (c : ℝ) (x : ℝ) (h1 : 0 < c) (h2 : c < 6) :
  (y = daily_profit x c) ∧
  (if 0 < c ∧ c < 3 then x = c else if 3 ≤ c ∧ c < 6 then x = 3 else False) :=
by
  sorry

end maximize_daily_profit_l41_41740


namespace percentage_error_in_square_area_l41_41750

-- Given an error of 1% in excess while measuring the side of a square,
-- prove that the percentage of error in the calculated area of the square is 2.01%.

theorem percentage_error_in_square_area (s : ℝ) (h : s ≠ 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let calculated_area := (1.01 * s) ^ 2
  let error_in_area := calculated_area - actual_area
  let percentage_error := (error_in_area / actual_area) * 100
  percentage_error = 2.01 :=
by {
  let measured_side := 1.01 * s;
  let actual_area := s ^ 2;
  let calculated_area := (1.01 * s) ^ 2;
  let error_in_area := calculated_area - actual_area;
  let percentage_error := (error_in_area / actual_area) * 100;
  sorry
}

end percentage_error_in_square_area_l41_41750


namespace max_rubles_earned_l41_41451

theorem max_rubles_earned :
  ∀ (cards_with_1 cards_with_2 : ℕ), 
  cards_with_1 = 2013 ∧ cards_with_2 = 2013 →
  ∃ (max_moves : ℕ), max_moves = 5 :=
by
  intros cards_with_1 cards_with_2 h
  sorry

end max_rubles_earned_l41_41451


namespace fourth_powers_count_l41_41349

theorem fourth_powers_count (n m : ℕ) (h₁ : n^4 ≥ 100) (h₂ : m^4 ≤ 10000) :
  ∃ k, k = m - n + 1 ∧ k = 7 :=
by
  sorry

end fourth_powers_count_l41_41349


namespace coeff_a_zero_l41_41089

-- Define the problem in Lean 4

theorem coeff_a_zero (a b c : ℝ) (h : ∀ p : ℝ, 0 < p → ∀ x, a * x^2 + b * x + c + p = 0 → 0 < x) :
  a = 0 :=
sorry

end coeff_a_zero_l41_41089


namespace arithmetic_geometric_sequence_l41_41813

theorem arithmetic_geometric_sequence (d : ℤ) (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = a (n - 1) * a (n + 2)) :
  a 2017 = 1 :=
sorry

end arithmetic_geometric_sequence_l41_41813


namespace number_of_second_graders_l41_41459

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end number_of_second_graders_l41_41459


namespace ratio_city_XY_l41_41627

variable (popZ popY popX : ℕ)

-- Definition of the conditions
def condition1 := popY = 2 * popZ
def condition2 := popX = 16 * popZ

-- The goal to prove
theorem ratio_city_XY 
  (h1 : condition1 popY popZ)
  (h2 : condition2 popX popZ) :
  popX / popY = 8 := 
  by sorry

end ratio_city_XY_l41_41627


namespace sphere_radius_is_16_25_l41_41374

def sphere_in_cylinder_radius (r : ℝ) : Prop := 
  ∃ (x : ℝ), (x ^ 2 + 15 ^ 2 = r ^ 2) ∧ ((x + 10) ^ 2 = r ^ 2) ∧ (r = 16.25)

theorem sphere_radius_is_16_25 : 
  sphere_in_cylinder_radius 16.25 :=
sorry

end sphere_radius_is_16_25_l41_41374


namespace min_steps_for_humpty_l41_41589

theorem min_steps_for_humpty (x y : ℕ) (H : 47 * x - 37 * y = 1) : x + y = 59 :=
  sorry

end min_steps_for_humpty_l41_41589


namespace prime_factorization_min_x_l41_41255

-- Define the conditions
variable (x y : ℕ) (a b e f : ℕ)

-- Given conditions: x and y are positive integers, and 5x^7 = 13y^11
axiom condition1 : 0 < x ∧ 0 < y
axiom condition2 : 5 * x^7 = 13 * y^11

-- Prove the mathematical equivalence
theorem prime_factorization_min_x (a b e f : ℕ) 
    (hx : 5 * x^7 = 13 * y^11)
    (h_prime : a = 13 ∧ b = 5 ∧ e = 6 ∧ f = 1) :
    a + b + e + f = 25 :=
sorry

end prime_factorization_min_x_l41_41255


namespace rectangular_prism_volume_l41_41134

theorem rectangular_prism_volume
  (L W h : ℝ)
  (h1 : L - W = 23)
  (h2 : 2 * L + 2 * W = 166) :
  L * W * h = 1590 * h :=
by
  sorry

end rectangular_prism_volume_l41_41134


namespace determine_k_completed_square_l41_41008

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l41_41008


namespace find_percentage_l41_41900

theorem find_percentage (P : ℝ) (h1 : (3 / 5) * 150 = 90) (h2 : (P / 100) * 90 = 36) : P = 40 :=
by
  sorry

end find_percentage_l41_41900


namespace intersect_points_count_l41_41949

open Classical
open Real

noncomputable def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

axiom f_invertible : ∀ x y : ℝ, f x = f y ↔ x = y

theorem intersect_points_count : ∃ (count : ℕ), count = 3 ∧ ∀ x : ℝ, (f (x ^ 3) = f (x ^ 5)) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end intersect_points_count_l41_41949


namespace smaller_than_neg3_l41_41856

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l41_41856


namespace suitable_comprehensive_survey_l41_41107

theorem suitable_comprehensive_survey :
  ¬(A = "comprehensive") ∧ ¬(B = "comprehensive") ∧ (C = "comprehensive") ∧ ¬(D = "comprehensive") → 
  suitable_survey = "C" :=
by
  sorry

end suitable_comprehensive_survey_l41_41107


namespace water_filter_capacity_l41_41098

theorem water_filter_capacity (x : ℝ) (h : 0.30 * x = 36) : x = 120 :=
sorry

end water_filter_capacity_l41_41098


namespace alan_needs_more_wings_l41_41595

theorem alan_needs_more_wings 
  (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) (target_wings : ℕ) : 
  kevin_wings = 64 → kevin_time = 8 → alan_rate = 5 → target_wings = 3 → 
  (kevin_wings / kevin_time < alan_rate + target_wings) :=
by
  intros kevin_eq time_eq rate_eq target_eq
  sorry

end alan_needs_more_wings_l41_41595


namespace line_circle_intersection_common_points_l41_41086

noncomputable def radius (d : ℝ) := d / 2

theorem line_circle_intersection_common_points 
  (diameter : ℝ) (distance_from_center_to_line : ℝ) 
  (h_dlt_r : distance_from_center_to_line < radius diameter) :
  ∃ common_points : ℕ, common_points = 2 :=
by
  sorry

end line_circle_intersection_common_points_l41_41086


namespace factor_expression_l41_41310

theorem factor_expression (x : ℕ) : 63 * x + 54 = 9 * (7 * x + 6) :=
by
  sorry

end factor_expression_l41_41310


namespace max_min_product_l41_41372

theorem max_min_product (A B : ℕ) (h : A + B = 100) : 
  (∃ (maxProd : ℕ), maxProd = 2500 ∧ (∀ (A B : ℕ), A + B = 100 → A * B ≤ maxProd)) ∧
  (∃ (minProd : ℕ), minProd = 0 ∧ (∀ (A B : ℕ), A + B = 100 → minProd ≤ A * B)) :=
by 
  -- Proof omitted
  sorry

end max_min_product_l41_41372


namespace inv_sum_eq_six_l41_41116

theorem inv_sum_eq_six (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := 
by 
  sorry

end inv_sum_eq_six_l41_41116


namespace Jorge_is_24_years_younger_l41_41699

-- Define the conditions
def Jorge_age_2005 := 16
def Simon_age_2010 := 45

-- Prove that Jorge is 24 years younger than Simon
theorem Jorge_is_24_years_younger :
  (Simon_age_2010 - (Jorge_age_2005 + 5) = 24) :=
by
  sorry

end Jorge_is_24_years_younger_l41_41699


namespace find_pyramid_volume_l41_41842

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β

theorem find_pyramid_volume (α β R : ℝ) 
  (base_isosceles : ∀ {a b c : ℝ}, a = b) -- Represents the isosceles triangle condition
  (dihedral_angles_equal : ∀ {angle : ℝ}, angle = β) -- Dihedral angle at the base
  (circumcircle_radius : {radius : ℝ // radius = R}) -- Radius of the circumcircle
  (height_through_point : true) -- Condition: height passes through a point inside the triangle
  :
  volume_of_pyramid α β R = (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β :=
by {
  sorry
}

end find_pyramid_volume_l41_41842


namespace three_irrational_numbers_l41_41741

theorem three_irrational_numbers (a b c d e : ℝ) 
  (ha : ¬ ∃ q1 q2 : ℚ, a = q1 + q2) 
  (hb : ¬ ∃ q1 q2 : ℚ, b = q1 + q2) 
  (hc : ¬ ∃ q1 q2 : ℚ, c = q1 + q2) 
  (hd : ¬ ∃ q1 q2 : ℚ, d = q1 + q2) 
  (he : ¬ ∃ q1 q2 : ℚ, e = q1 + q2) : 
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) 
  ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) 
  ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)
  ∧ (¬ ∃ q1 q2 : ℚ, x + y = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, y + z = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, z + x = q1 + q2) :=
sorry

end three_irrational_numbers_l41_41741


namespace constant_seq_is_arith_not_always_geom_l41_41319

theorem constant_seq_is_arith_not_always_geom (c : ℝ) (seq : ℕ → ℝ) (h : ∀ n, seq n = c) :
  (∀ n, seq (n + 1) - seq n = 0) ∧ (c = 0 ∨ (∀ n, seq (n + 1) / seq n = 1)) :=
by
  sorry

end constant_seq_is_arith_not_always_geom_l41_41319


namespace sum_of_x_values_satisfying_eq_l41_41339

noncomputable def rational_eq_sum (x : ℝ) : Prop :=
3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

theorem sum_of_x_values_satisfying_eq :
  (∃ (x : ℝ), rational_eq_sum x) ∧ (x ≠ -3 → (x_1 + x_2) = 6) :=
sorry

end sum_of_x_values_satisfying_eq_l41_41339


namespace average_speed_is_69_l41_41896

-- Definitions for the conditions
def distance_hr1 : ℕ := 90
def distance_hr2 : ℕ := 30
def distance_hr3 : ℕ := 60
def distance_hr4 : ℕ := 120
def distance_hr5 : ℕ := 45
def total_distance : ℕ := distance_hr1 + distance_hr2 + distance_hr3 + distance_hr4 + distance_hr5
def total_time : ℕ := 5

-- The theorem to be proven
theorem average_speed_is_69 :
  (total_distance / total_time) = 69 :=
by
  sorry

end average_speed_is_69_l41_41896


namespace value_of_r_minus_q_l41_41916

variable (q r : ℝ)
variable (slope : ℝ)
variable (h_parallel : slope = 3 / 2)
variable (h_points : (r - q) / (-2) = slope)

theorem value_of_r_minus_q (h_parallel : slope = 3 / 2) (h_points : (r - q) / (-2) = slope) : 
  r - q = -3 := by
  sorry

end value_of_r_minus_q_l41_41916


namespace remaining_distance_l41_41322

-- Definitions of conditions
def distance_to_grandmother : ℕ := 300
def speed_per_hour : ℕ := 60
def time_elapsed : ℕ := 2

-- Statement of the proof problem
theorem remaining_distance : distance_to_grandmother - (speed_per_hour * time_elapsed) = 180 :=
by 
  sorry

end remaining_distance_l41_41322


namespace point_not_similar_inflection_point_ln_l41_41561

noncomputable def similar_inflection_point (C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ (m : ℝ → ℝ), (∀ x, m x = (deriv C P.1) * (x - P.1) + P.2) ∧
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → (C (P.1 + h) > m (P.1 + h) ∧ C (P.1 - h) < m (P.1 - h)) ∨ 
                     (C (P.1 + h) < m (P.1 + h) ∧ C (P.1 - h) > m (P.1 - h))

theorem point_not_similar_inflection_point_ln :
  ¬ similar_inflection_point (fun x => Real.log x) (1, 0) :=
sorry

end point_not_similar_inflection_point_ln_l41_41561


namespace line_intersects_circle_l41_41400

noncomputable def diameter : ℝ := 8
noncomputable def radius : ℝ := diameter / 2
noncomputable def center_to_line_distance : ℝ := 3

theorem line_intersects_circle :
  center_to_line_distance < radius → True :=
by {
  /- The proof would go here, but for now, we use sorry. -/
  sorry
}

end line_intersects_circle_l41_41400


namespace brothers_complete_task_in_3_days_l41_41666

theorem brothers_complete_task_in_3_days :
  (1 / 4 + 1 / 12) * 3 = 1 :=
by
  sorry

end brothers_complete_task_in_3_days_l41_41666


namespace no_four_consecutive_powers_l41_41943

/-- 
  There do not exist four consecutive natural numbers 
  such that each of them is a power (greater than 1) of another natural number.
-/
theorem no_four_consecutive_powers : 
  ¬ ∃ (n : ℕ), (∀ (i : ℕ), i < 4 → ∃ (a k : ℕ), k > 1 ∧ n + i = a^k) := sorry

end no_four_consecutive_powers_l41_41943


namespace inequality_sqrt_a_b_c_l41_41705

noncomputable def sqrt (x : ℝ) := x ^ (1 / 2)

theorem inequality_sqrt_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a ^ (1 - a) * b ^ (1 - b) * c ^ (1 - c)) ≤ 1 / 3 := 
sorry

end inequality_sqrt_a_b_c_l41_41705


namespace height_of_middle_brother_l41_41555

theorem height_of_middle_brother (h₁ h₂ h₃ : ℝ) (h₁_le_h₂ : h₁ ≤ h₂) (h₂_le_h₃ : h₂ ≤ h₃)
  (avg_height : (h₁ + h₂ + h₃) / 3 = 1.74) (avg_height_tallest_shortest : (h₁ + h₃) / 2 = 1.75) :
  h₂ = 1.72 :=
by
  -- Proof to be filled here
  sorry

end height_of_middle_brother_l41_41555


namespace converse_xy_implies_x_is_true_l41_41744

/-- Prove that the converse of the proposition "If \(xy = 0\), then \(x = 0\)" is true. -/
theorem converse_xy_implies_x_is_true {x y : ℝ} (h : x = 0) : x * y = 0 :=
by sorry

end converse_xy_implies_x_is_true_l41_41744


namespace students_per_class_l41_41567

theorem students_per_class
  (cards_per_student : Nat)
  (periods_per_day : Nat)
  (cost_per_pack : Nat)
  (total_spent : Nat)
  (cards_per_pack : Nat)
  (students_per_class : Nat)
  (H1 : cards_per_student = 10)
  (H2 : periods_per_day = 6)
  (H3 : cost_per_pack = 3)
  (H4 : total_spent = 108)
  (H5 : cards_per_pack = 50)
  (H6 : students_per_class = 30)
  :
  students_per_class = (total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day) :=
sorry

end students_per_class_l41_41567


namespace number_when_added_by_5_is_30_l41_41315

theorem number_when_added_by_5_is_30 (x: ℕ) (h: x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end number_when_added_by_5_is_30_l41_41315


namespace cistern_fill_time_l41_41810

theorem cistern_fill_time
  (T : ℝ)
  (H1 : 0 < T)
  (rate_first_tap : ℝ := 1 / T)
  (rate_second_tap : ℝ := 1 / 6)
  (net_rate : ℝ := 1 / 12)
  (H2 : rate_first_tap - rate_second_tap = net_rate) :
  T = 4 :=
sorry

end cistern_fill_time_l41_41810


namespace number_of_people_joining_group_l41_41384

theorem number_of_people_joining_group (x : ℕ) (h1 : 180 / 18 = 10) 
  (h2 : 180 / (18 + x) = 9) : x = 2 :=
by
  sorry

end number_of_people_joining_group_l41_41384


namespace intersection_eq_l41_41524

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem intersection_eq : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_eq_l41_41524


namespace probability_one_head_one_tail_l41_41725

def toss_outcomes : List (String × String) := [("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")]

def favorable_outcomes (outcomes : List (String × String)) : List (String × String) :=
  outcomes.filter (fun x => (x = ("head", "tail")) ∨ (x = ("tail", "head")))

theorem probability_one_head_one_tail :
  (favorable_outcomes toss_outcomes).length / toss_outcomes.length = 1 / 2 :=
by
  -- Proof will be filled in here
  sorry

end probability_one_head_one_tail_l41_41725


namespace least_positive_integer_special_property_l41_41978

/-- 
  Prove that 9990 is the least positive integer whose digits sum to a multiple of 27 
  and the number itself is not a multiple of 27.
-/
theorem least_positive_integer_special_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (Nat.digits 10 n).sum % 27 = 0 ∧ 
  n % 27 ≠ 0 ∧ 
  ∀ m : ℕ, (m > 0 ∧ (Nat.digits 10 m).sum % 27 = 0 ∧ m % 27 ≠ 0 → n ≤ m) := 
by
  sorry

end least_positive_integer_special_property_l41_41978


namespace abc_inequality_l41_41331

theorem abc_inequality (x y z : ℝ) (a b c : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a = (x * (y - z) ^ 2) ^ 2) (h2 : b = (y * (z - x) ^ 2) ^ 2) (h3 : c = (z * (x - y) ^ 2) ^ 2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) :=
by {
  sorry
}

end abc_inequality_l41_41331


namespace problem_solution_l41_41655

theorem problem_solution (a b d : ℤ) (ha : a = 2500) (hb : b = 2409) (hd : d = 81) :
  (a - b) ^ 2 / d = 102 := by
  sorry

end problem_solution_l41_41655


namespace coefficient_of_8th_term_l41_41711

-- Define the general term of the binomial expansion
def binomial_expansion_term (n r : ℕ) (a b : ℕ) : ℕ := 
  Nat.choose n r * a^(n - r) * b^r

-- Define the specific scenario given in the problem
def specific_binomial_expansion_term : ℕ := 
  binomial_expansion_term 8 7 2 1  -- a = 2, b = x (consider b as 1 for coefficient calculation)

-- Problem statement to prove the coefficient of the 8th term is 16
theorem coefficient_of_8th_term : specific_binomial_expansion_term = 16 := by
  sorry

end coefficient_of_8th_term_l41_41711


namespace sum_moments_equal_l41_41119

theorem sum_moments_equal
  (x1 x2 x3 y1 y2 : ℝ)
  (m1 m2 m3 n1 n2 : ℝ) :
  n1 * y1 + n2 * y2 = m1 * x1 + m2 * x2 + m3 * x3 :=
sorry

end sum_moments_equal_l41_41119


namespace simplify_exponent_l41_41385

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l41_41385


namespace triangle_inequality_part_a_triangle_inequality_part_b_l41_41779

variable {a b c S : ℝ}

/-- Part (a): Prove that for any triangle ABC, the inequality a^2 + b^2 + c^2 ≥ 4 √3 S holds
    where equality holds if and only if ABC is an equilateral triangle. -/
theorem triangle_inequality_part_a (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

/-- Part (b): Prove that for any triangle ABC,
    the inequality a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 √3 S
    holds where equality also holds if and only if a = b = c. -/
theorem triangle_inequality_part_b (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_part_a_triangle_inequality_part_b_l41_41779


namespace total_coffee_consumed_l41_41032

def Ivory_hourly_coffee := 2
def Kimberly_hourly_coffee := Ivory_hourly_coffee
def Brayan_hourly_coffee := 4
def Raul_hourly_coffee := Brayan_hourly_coffee / 2
def duration_hours := 10

theorem total_coffee_consumed :
  (Brayan_hourly_coffee * duration_hours) + 
  (Ivory_hourly_coffee * duration_hours) + 
  (Kimberly_hourly_coffee * duration_hours) + 
  (Raul_hourly_coffee * duration_hours) = 100 :=
by sorry

end total_coffee_consumed_l41_41032


namespace intersection_point_exists_l41_41112

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + 6 * t)
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3 * u)

theorem intersection_point_exists :
  ∃ t u : ℝ, line1 t = (1, 2) ∧ line2 u = (1, 2) := 
by
  sorry

end intersection_point_exists_l41_41112


namespace abc_sum_16_l41_41222

theorem abc_sum_16 (a b c : ℕ) (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4) (h4 : a ≠ b ∨ b ≠ c ∨ a ≠ c)
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by
  sorry

end abc_sum_16_l41_41222


namespace no_solution_fraction_equation_l41_41549

theorem no_solution_fraction_equation (x : ℝ) (h : x ≠ 2) : 
  (1 - x) / (x - 2) + 2 = 1 / (2 - x) → false :=
by 
  intro h_eq
  sorry

end no_solution_fraction_equation_l41_41549


namespace sum_series_div_3_powers_l41_41839

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l41_41839


namespace mia_weight_l41_41830

theorem mia_weight (a m : ℝ) (h1 : a + m = 220) (h2 : m - a = 2 * a) : m = 165 :=
sorry

end mia_weight_l41_41830


namespace number_of_pipes_l41_41682

theorem number_of_pipes (h_same_height : forall (height : ℝ), height > 0)
  (diam_large : ℝ) (hl : diam_large = 6)
  (diam_small : ℝ) (hs : diam_small = 1) :
  (π * (diam_large / 2)^2) / (π * (diam_small / 2)^2) = 36 :=
by
  sorry

end number_of_pipes_l41_41682


namespace polynomial_divisible_l41_41951

theorem polynomial_divisible (A B : ℝ) (h : ∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by
  sorry

end polynomial_divisible_l41_41951


namespace magic_triangle_max_sum_l41_41408

theorem magic_triangle_max_sum :
  ∃ (a b c d e f : ℕ), ((a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8 ∨ a = 9 ∨ a = 10) ∧
                        (b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8 ∨ b = 9 ∨ b = 10) ∧
                        (c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10) ∧
                        (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 ∨ d = 10) ∧
                        (e = 5 ∨ e = 6 ∨ e = 7 ∨ e = 8 ∨ e = 9 ∨ e = 10) ∧
                        (f = 5 ∨ f = 6 ∨ f = 7 ∨ f = 8 ∨ f = 9 ∨ f = 10) ∧
                        (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
                        (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
                        (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
                        (d ≠ e) ∧ (d ≠ f) ∧
                        (e ≠ f) ∧
                        (a + b + c = 24) ∧ (c + d + e = 24) ∧ (e + f + a = 24)) :=
sorry

end magic_triangle_max_sum_l41_41408


namespace point_P_quadrant_IV_l41_41919

theorem point_P_quadrant_IV (x y : ℝ) (h1 : x > 0) (h2 : y < 0) : x > 0 ∧ y < 0 :=
by
  sorry

end point_P_quadrant_IV_l41_41919


namespace right_triangle_construction_condition_l41_41241

theorem right_triangle_construction_condition (A B C : Point) (b d : ℝ) :
  AC = b → AC + BC - AB = d → b > d :=
by
  intro h1 h2
  sorry

end right_triangle_construction_condition_l41_41241


namespace twenty_percent_greater_l41_41138

theorem twenty_percent_greater (x : ℝ) (h : x = 52 + 0.2 * 52) : x = 62.4 :=
by {
  sorry
}

end twenty_percent_greater_l41_41138


namespace largest_area_polygons_l41_41505

-- Define the area of each polygon
def area_P := 4
def area_Q := 6
def area_R := 3 + 3 * (1 / 2)
def area_S := 6 * (1 / 2)
def area_T := 5 + 2 * (1 / 2)

-- Proof of the polygons with the largest area
theorem largest_area_polygons : (area_Q = 6 ∧ area_T = 6) ∧ area_Q ≥ area_P ∧ area_Q ≥ area_R ∧ area_Q ≥ area_S :=
by
  sorry

end largest_area_polygons_l41_41505


namespace man_savings_percentage_l41_41628

theorem man_savings_percentage
  (salary expenses : ℝ)
  (increase_percentage : ℝ)
  (current_savings : ℝ)
  (P : ℝ)
  (h1 : salary = 7272.727272727273)
  (h2 : increase_percentage = 0.05)
  (h3 : current_savings = 400)
  (h4 : current_savings + (increase_percentage * salary) = (P / 100) * salary) :
  P = 10.5 := 
sorry

end man_savings_percentage_l41_41628


namespace sum_of_coordinates_of_other_endpoint_l41_41854

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (midpoint_cond : (x + 1) / 2 = 3)
  (midpoint_cond2 : (y - 3) / 2 = 5) :
  x + y = 18 :=
sorry

end sum_of_coordinates_of_other_endpoint_l41_41854


namespace bella_items_l41_41056

theorem bella_items (M F D : ℕ) 
  (h1 : M = 60)
  (h2 : M = 2 * F)
  (h3 : F = D + 20) :
  (7 * M + 7 * F + 7 * D) / 5 = 140 := 
by
  sorry

end bella_items_l41_41056


namespace overall_average_score_l41_41030

theorem overall_average_score (first_6_avg last_4_avg : ℝ) (n_first n_last n_total : ℕ) 
    (h_matches : n_first + n_last = n_total)
    (h_first_avg : first_6_avg = 41)
    (h_last_avg : last_4_avg = 35.75)
    (h_n_first : n_first = 6)
    (h_n_last : n_last = 4)
    (h_n_total : n_total = 10) :
    ((first_6_avg * n_first + last_4_avg * n_last) / n_total) = 38.9 := by
  sorry

end overall_average_score_l41_41030


namespace amount_received_is_500_l41_41479

-- Define the conditions
def books_per_month : ℕ := 3
def months_per_year : ℕ := 12
def price_per_book : ℕ := 20
def loss : ℕ := 220

-- Calculate number of books bought in a year
def books_per_year : ℕ := books_per_month * months_per_year

-- Calculate total amount spent on books in a year
def total_spent : ℕ := books_per_year * price_per_book

-- Calculate the amount Jack got from selling the books based on the given loss
def amount_received : ℕ := total_spent - loss

-- Proving the amount received is $500
theorem amount_received_is_500 : amount_received = 500 := by
  sorry

end amount_received_is_500_l41_41479


namespace sums_of_integers_have_same_remainder_l41_41068

theorem sums_of_integers_have_same_remainder (n : ℕ) (n_pos : 0 < n) : 
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2 * n) ∧ (1 ≤ j ∧ j ≤ 2 * n) ∧ i ≠ j ∧ ((i + i) % (2 * n) = (j + j) % (2 * n)) :=
by
  sorry

end sums_of_integers_have_same_remainder_l41_41068


namespace sum_of_angles_l41_41591

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (sin_α : Real.sin α = 2 * Real.sqrt 5 / 5) (sin_beta : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_angles_l41_41591


namespace polynomial_divisibility_l41_41140

theorem polynomial_divisibility (m : ℕ) (h_pos : 0 < m) : 
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1)^(2 * m) - x^(2 * m) - 2 * x - 1 :=
sorry

end polynomial_divisibility_l41_41140


namespace ratio_correct_l41_41127

def cost_of_flasks := 150
def remaining_budget := 25
def total_budget := 325
def spent_budget := total_budget - remaining_budget
def cost_of_test_tubes := 100
def cost_of_safety_gear := cost_of_test_tubes / 2
def ratio_test_tubes_flasks := cost_of_test_tubes / cost_of_flasks

theorem ratio_correct :
  spent_budget = cost_of_flasks + cost_of_test_tubes + cost_of_safety_gear → 
  ratio_test_tubes_flasks = 2 / 3 :=
by
  sorry

end ratio_correct_l41_41127


namespace domain_of_f_l41_41379

def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∀ x, f x ∈ D

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  domain f {y | y ≠ -2} :=
by sorry

end domain_of_f_l41_41379


namespace panic_percentage_l41_41679

theorem panic_percentage (original_population disappeared_after first_population second_population : ℝ) 
  (h₁ : original_population = 7200)
  (h₂ : disappeared_after = original_population * 0.10)
  (h₃ : first_population = original_population - disappeared_after)
  (h₄ : second_population = 4860)
  (h₅ : second_population = first_population - (first_population * 0.25)) : 
  second_population = first_population * (1 - 0.25) :=
by
  sorry

end panic_percentage_l41_41679


namespace wholesale_cost_l41_41122

theorem wholesale_cost (W R : ℝ) (h1 : R = 1.20 * W) (h2 : 0.70 * R = 168) : W = 200 :=
by
  sorry

end wholesale_cost_l41_41122


namespace problem_l41_41074

variable (p q : Prop)

theorem problem (h₁ : ¬ p) (h₂ : ¬ (p ∧ q)) : ¬ (p ∨ q) := sorry

end problem_l41_41074


namespace fraction_evaporated_l41_41829

theorem fraction_evaporated (x : ℝ) (h : (1 - x) * (1/4) = 1/6) : x = 1/3 :=
by
  sorry

end fraction_evaporated_l41_41829


namespace initial_solution_amount_l41_41556

theorem initial_solution_amount (x : ℝ) (h1 : x - 200 + 1000 = 2000) : x = 1200 := by
  sorry

end initial_solution_amount_l41_41556


namespace prob_is_correct_l41_41454

def total_balls : ℕ := 500
def white_balls : ℕ := 200
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50
def red_balls : ℕ := 30
def purple_balls : ℕ := 20
def orange_balls : ℕ := 30

noncomputable def probability_green_yellow_blue : ℚ :=
  (green_balls + yellow_balls + blue_balls) / total_balls

theorem prob_is_correct :
  probability_green_yellow_blue = 0.44 := 
  by
  sorry

end prob_is_correct_l41_41454


namespace number_of_faces_of_prism_proof_l41_41835

noncomputable def number_of_faces_of_prism (n : ℕ) : ℕ := 2 + n

theorem number_of_faces_of_prism_proof (n : ℕ) (E_p E_py : ℕ) (h1 : E_p + E_py = 30) (h2 : E_p = 3 * n) (h3 : E_py = 2 * n) :
  number_of_faces_of_prism n = 8 :=
by
  sorry

end number_of_faces_of_prism_proof_l41_41835


namespace candy_problem_l41_41219

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l41_41219


namespace ingrid_tax_rate_l41_41610

def john_income : ℝ := 57000
def ingrid_income : ℝ := 72000
def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_tax_rate :
  let john_tax := john_tax_rate * john_income
  let combined_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * combined_income
  let ingrid_tax := total_tax - john_tax
  let ingrid_tax_rate := ingrid_tax / ingrid_income
  ingrid_tax_rate = 0.40 :=
by
  sorry

end ingrid_tax_rate_l41_41610


namespace sum_geometric_sequence_l41_41052

variable {α : Type*} [LinearOrderedField α]

theorem sum_geometric_sequence {S : ℕ → α} {n : ℕ} (h1 : S n = 3) (h2 : S (3 * n) = 21) :
    S (2 * n) = 9 := 
sorry

end sum_geometric_sequence_l41_41052


namespace fans_per_set_l41_41694

theorem fans_per_set (total_fans : ℕ) (sets_of_bleachers : ℕ) (fans_per_set : ℕ)
  (h1 : total_fans = 2436) (h2 : sets_of_bleachers = 3) : fans_per_set = 812 :=
by
  sorry

end fans_per_set_l41_41694


namespace gcd_gx_x_l41_41269

noncomputable def g (x : ℕ) := (5 * x + 3) * (11 * x + 2) * (6 * x + 7) * (3 * x + 8)

theorem gcd_gx_x {x : ℕ} (hx : 36000 ∣ x) : Nat.gcd (g x) x = 144 := by
  sorry

end gcd_gx_x_l41_41269


namespace restaurant_vegetarian_dishes_l41_41050

theorem restaurant_vegetarian_dishes (n : ℕ) : 
    5 ≥ 2 → 200 < Nat.choose 5 2 * Nat.choose n 2 → n ≥ 7 :=
by
  intros h_combinations h_least
  sorry

end restaurant_vegetarian_dishes_l41_41050


namespace range_of_a_l41_41271

theorem range_of_a
  (a : ℝ)
  (h : ∀ (x : ℝ), 1 < x ∧ x < 4 → x^2 - 3 * x - 2 - a > 0) :
  a < 2 :=
sorry

end range_of_a_l41_41271


namespace entrants_total_l41_41673

theorem entrants_total (N : ℝ) (h1 : N > 800)
  (h2 : 0.35 * N = NumFemales)
  (h3 : 0.65 * N = NumMales)
  (h4 : NumMales - NumFemales = 252) :
  N = 840 := 
sorry

end entrants_total_l41_41673


namespace nearest_integer_to_expr_l41_41808

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l41_41808


namespace average_speed_last_segment_l41_41674

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed1 speed2 speed3 : ℕ)
  (last_segment_time : ℕ)
  (average_speed_total : ℕ) :
  total_distance = 180 →
  total_time = 180 →
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 60 →
  average_speed_total = 60 →
  last_segment_time = 45 →
  ∃ (speed4 : ℕ), speed4 = 90 :=
by sorry

end average_speed_last_segment_l41_41674


namespace triangle_identity_l41_41346

theorem triangle_identity
  (A B C : ℝ) (a b c: ℝ)
  (h1: A + B + C = Real.pi)
  (h2: a = 2 * R * Real.sin A)
  (h3: b = 2 * R * Real.sin B)
  (h4: c = 2 * R * Real.sin C)
  (h5: Real.sin A = Real.sin B * Real.cos C + Real.cos B * Real.sin C) :
  (b * Real.cos C + c * Real.cos B) / a = 1 := 
  by 
  sorry

end triangle_identity_l41_41346


namespace tree_height_at_end_of_2_years_l41_41383

-- Conditions:
-- 1. The tree tripled its height every year.
-- 2. The tree reached a height of 243 feet at the end of 5 years.
theorem tree_height_at_end_of_2_years (h5 : ℕ) (H5 : h5 = 243) : 
  ∃ h2, h2 = 9 := 
by sorry

end tree_height_at_end_of_2_years_l41_41383


namespace sandy_books_from_second_shop_l41_41061

noncomputable def books_from_second_shop (books_first: ℕ) (cost_first: ℕ) (cost_second: ℕ) (avg_price: ℕ): ℕ :=
  let total_cost := cost_first + cost_second
  let total_books := books_first + (total_cost / avg_price) - books_first
  total_cost / avg_price - books_first

theorem sandy_books_from_second_shop :
  books_from_second_shop 65 1380 900 19 = 55 :=
by
  sorry

end sandy_books_from_second_shop_l41_41061


namespace equation_of_line_passing_through_point_with_slope_l41_41378

theorem equation_of_line_passing_through_point_with_slope :
  ∃ (l : ℝ → ℝ), l 0 = -1 ∧ ∀ (x y : ℝ), y = l x ↔ y + 1 = 2 * x :=
sorry

end equation_of_line_passing_through_point_with_slope_l41_41378


namespace mandy_cinnamon_amount_correct_l41_41849

def mandy_cinnamon_amount (nutmeg : ℝ) (cinnamon : ℝ) : Prop :=
  cinnamon = nutmeg + 0.17

theorem mandy_cinnamon_amount_correct :
  mandy_cinnamon_amount 0.5 0.67 :=
by
  sorry

end mandy_cinnamon_amount_correct_l41_41849


namespace amount_added_to_doubled_number_l41_41976

theorem amount_added_to_doubled_number (N A : ℝ) (h1 : N = 6.0) (h2 : 2 * N + A = 17) : A = 5.0 :=
by
  sorry

end amount_added_to_doubled_number_l41_41976


namespace total_number_of_pieces_paper_l41_41166

-- Define the number of pieces of paper each person picked up
def olivia_pieces : ℝ := 127.5
def edward_pieces : ℝ := 345.25
def sam_pieces : ℝ := 518.75

-- Define the total number of pieces of paper picked up
def total_pieces : ℝ := olivia_pieces + edward_pieces + sam_pieces

-- The theorem to be proven
theorem total_number_of_pieces_paper :
  total_pieces = 991.5 :=
by
  -- Sorry is used as we are not required to provide a proof here
  sorry

end total_number_of_pieces_paper_l41_41166


namespace initial_blue_balls_l41_41778

-- Define the initial conditions
variables (B : ℕ) (total_balls : ℕ := 15) (removed_blue_balls : ℕ := 3)
variable (prob_after_removal : ℚ := 1 / 3)
variable (remaining_balls : ℕ := total_balls - removed_blue_balls)
variable (remaining_blue_balls : ℕ := B - removed_blue_balls)

-- State the theorem
theorem initial_blue_balls : 
  remaining_balls = 12 → remaining_blue_balls = remaining_balls * prob_after_removal → B = 7 :=
by
  intros h1 h2
  sorry

end initial_blue_balls_l41_41778


namespace packs_needed_l41_41828

def pouches_per_pack : ℕ := 6
def team_members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := team_members + coaches + helpers

theorem packs_needed (people : ℕ) (pouches_per_pack : ℕ) : ℕ :=
  (people + pouches_per_pack - 1) / pouches_per_pack

example : packs_needed total_people pouches_per_pack = 3 :=
by
  have h1 : total_people = 18 := rfl
  have h2 : pouches_per_pack = 6 := rfl
  rw [h1, h2]
  norm_num
  sorry

end packs_needed_l41_41828


namespace black_white_tile_ratio_l41_41633

theorem black_white_tile_ratio :
  let original_black_tiles := 10
  let original_white_tiles := 15
  let total_tiles_in_original_square := original_black_tiles + original_white_tiles
  let side_length_of_original_square := Int.sqrt total_tiles_in_original_square -- this should be 5
  let side_length_of_extended_square := side_length_of_original_square + 2
  let total_black_tiles_in_border := 4 * (side_length_of_extended_square - 1) / 2 -- Each border side starts and ends with black
  let total_white_tiles_in_border := (side_length_of_extended_square * 4 - 4) - total_black_tiles_in_border 
  let new_total_black_tiles := original_black_tiles + total_black_tiles_in_border
  let new_total_white_tiles := original_white_tiles + total_white_tiles_in_border
  (new_total_black_tiles / gcd new_total_black_tiles new_total_white_tiles) / 
  (new_total_white_tiles / gcd new_total_black_tiles new_total_white_tiles) = 26 / 23 :=
by
  sorry

end black_white_tile_ratio_l41_41633


namespace compute_cd_l41_41370

-- Define the variables c and d as real numbers
variables (c d : ℝ)

-- Define the conditions
def condition1 : Prop := c + d = 10
def condition2 : Prop := c^3 + d^3 = 370

-- State the theorem we need to prove
theorem compute_cd (h1 : condition1 c d) (h2 : condition2 c d) : c * d = 21 :=
by
  sorry

end compute_cd_l41_41370


namespace bulb_standard_probability_l41_41148

noncomputable def prob_A 
  (P_H1 : ℝ) (P_H2 : ℝ) (P_A_given_H1 : ℝ) (P_A_given_H2 : ℝ) :=
  P_A_given_H1 * P_H1 + P_A_given_H2 * P_H2

theorem bulb_standard_probability 
  (P_H1 : ℝ := 0.6) (P_H2 : ℝ := 0.4) 
  (P_A_given_H1 : ℝ := 0.95) (P_A_given_H2 : ℝ := 0.85) :
  prob_A P_H1 P_H2 P_A_given_H1 P_A_given_H2 = 0.91 :=
by
  sorry

end bulb_standard_probability_l41_41148


namespace permutations_without_HMMT_l41_41399

noncomputable def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem permutations_without_HMMT :
  let total_permutations := multinomial 8 2 2 4
  let block_permutations := multinomial 5 1 1 2
  (total_permutations - block_permutations + 1) = 361 :=
by
  sorry

end permutations_without_HMMT_l41_41399


namespace no_periodic_sequence_first_non_zero_digit_l41_41584

/-- 
Definition of the first non-zero digit from the unit's place in the decimal representation of n! 
-/
def first_non_zero_digit (n : ℕ) : ℕ :=
  -- This function should compute the first non-zero digit from the unit's place in n!
  -- Implementation details are skipped here.
  sorry

/-- 
Prove that no natural number \( N \) exists such that the sequence \( a_{N+1}, a_{N+2}, a_{N+3}, \ldots \) 
forms a periodic sequence, where \( a_n \) is the first non-zero digit from the unit's place in the decimal 
representation of \( n! \). 
-/
theorem no_periodic_sequence_first_non_zero_digit :
  ¬ ∃ (N : ℕ), ∃ (T : ℕ), ∀ (k : ℕ), first_non_zero_digit (N + k * T) = first_non_zero_digit (N + ((k + 1) * T)) :=
by
  sorry

end no_periodic_sequence_first_non_zero_digit_l41_41584


namespace average_of_rstu_l41_41295

theorem average_of_rstu (r s t u : ℝ) (h : (5 / 4) * (r + s + t + u) = 15) : (r + s + t + u) / 4 = 3 :=
by
  sorry

end average_of_rstu_l41_41295


namespace sector_area_l41_41759

noncomputable def area_of_sector (r : ℝ) (theta : ℝ) : ℝ :=
  1 / 2 * r * r * theta

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = Real.pi) (h_theta : theta = 2 * Real.pi / 3) :
  area_of_sector r theta = Real.pi^3 / 6 :=
by
  sorry

end sector_area_l41_41759


namespace simplify_expression_eq_l41_41420

theorem simplify_expression_eq (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a - 1/a) / ((a^2 - 2 * a + 1) / a) = (a + 1) / (a - 1) :=
by
  sorry

end simplify_expression_eq_l41_41420


namespace subset_range_a_l41_41606

def setA : Set ℝ := { x | (x^2 - 4 * x + 3) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (2^(1 - x) + a) ≤ 0 ∧ (x^2 - 2*(a + 7)*x + 5) ≤ 0 }

theorem subset_range_a (a : ℝ) : setA ⊆ setB a ↔ -4 ≤ a ∧ a ≤ -1 := 
  sorry

end subset_range_a_l41_41606


namespace div_trans_l41_41500

variable {a b c : ℝ}

theorem div_trans :
  a / b = 3 → b / c = 5 / 2 → c / a = 2 / 15 :=
  by
  intro h1 h2
  sorry

end div_trans_l41_41500


namespace line_y_intercept_l41_41063

theorem line_y_intercept (t : ℝ) (h : ∃ (t : ℝ), ∀ (x y : ℝ), x - 2 * y + t = 0 → (x = 2 ∧ y = -1)) :
  ∃ y : ℝ, (0 - 2 * y + t = 0) ∧ y = -2 :=
by
  sorry

end line_y_intercept_l41_41063


namespace jogging_time_after_two_weeks_l41_41510

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end jogging_time_after_two_weeks_l41_41510


namespace perpendicular_lines_l41_41285

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y - a = 0) → (a * x - (2 * a - 3) * y - 1 = 0) → 
    (∀ x y : ℝ, ( -1 / a ) * ( -a / (2 * a - 3)) = 1 )) → a = 3 := 
by
  sorry

end perpendicular_lines_l41_41285


namespace cos_squared_identity_l41_41927

variable (θ : ℝ)

-- Given condition
def tan_theta : Prop := Real.tan θ = 2

-- Question: Find the value of cos²(θ + π/4)
theorem cos_squared_identity (h : tan_theta θ) : Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 10 := 
  sorry

end cos_squared_identity_l41_41927


namespace slope_of_perpendicular_line_l41_41042

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line_l41_41042


namespace sum_of_money_proof_l41_41002

noncomputable def total_sum (A B C : ℝ) : ℝ := A + B + C

theorem sum_of_money_proof (A B C : ℝ) (h1 : B = 0.65 * A) (h2 : C = 0.40 * A) (h3 : C = 64) : total_sum A B C = 328 :=
by 
  sorry

end sum_of_money_proof_l41_41002


namespace cars_to_sell_l41_41763

theorem cars_to_sell (n : ℕ) 
  (h1 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → ∃ m, m = 3)
  (h2 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → c ∈ {c' : ℕ | c' < 3})
  (h3 : 15 * 3 = 45)
  (h4 : ∀ n, n * 3 = 45 → n = 15):
  n = 15 := 
  by
    have n_eq: n * 3 = 45 := sorry
    exact h4 n n_eq

end cars_to_sell_l41_41763


namespace tan_diff_identity_l41_41209

theorem tan_diff_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β + π / 4) = 1 / 4) :
  Real.tan (α - π / 4) = 3 / 22 :=
sorry

end tan_diff_identity_l41_41209


namespace sum_of_solutions_l41_41677

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : ∃ (a b : ℝ), (a = 5 ∨ a = -5) ∧ (b = 5 ∨ b = -5) ∧ a + b = 0 :=
sorry

end sum_of_solutions_l41_41677


namespace area_percentage_increase_l41_41913

theorem area_percentage_increase (r₁ r₂ : ℝ) (π : ℝ) :
  r₁ = 6 ∧ r₂ = 4 ∧ π > 0 →
  (π * r₁^2 - π * r₂^2) / (π * r₂^2) * 100 = 125 := 
by {
  sorry
}

end area_percentage_increase_l41_41913


namespace second_number_value_l41_41497

theorem second_number_value 
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a / b = 3 / 4)
  (h3 : b / c = 2 / 5) :
  b = 480 / 17 :=
by
  sorry

end second_number_value_l41_41497


namespace find_m_l41_41386

theorem find_m (m : ℝ) :
  (∀ x y : ℝ, (3 * x + (m + 1) * y - (m - 7) = 0) → 
              (m * x + 2 * y + 3 * m = 0)) →
  (m + 1 ≠ 0) →
  m = -3 :=
by
  sorry

end find_m_l41_41386


namespace integer_cube_less_than_triple_unique_l41_41802

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l41_41802


namespace georg_can_identify_fake_coins_l41_41783

theorem georg_can_identify_fake_coins :
  ∀ (coins : ℕ) (baron : ℕ → ℕ → ℕ) (queries : ℕ),
    coins = 100 →
    ∃ (fake_count : ℕ → ℕ) (exaggeration : ℕ),
      (∀ group_size : ℕ, 10 ≤ group_size ∧ group_size ≤ 20) →
      (∀ (show_coins : ℕ), show_coins ≤ group_size → fake_count show_coins = baron show_coins exaggeration) →
      queries < 120 :=
by
  sorry

end georg_can_identify_fake_coins_l41_41783


namespace simplify_fraction_mul_l41_41431

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : a = 210) (h2 : b = 7350) (h3 : c = 1) (h4 : d = 35) (h5 : 210 / gcd 210 7350 = 1) (h6: 7350 / gcd 210 7350 = 35) :
  (a / b) * 14 = 2 / 5 :=
by
  sorry

end simplify_fraction_mul_l41_41431


namespace yield_percentage_of_stock_is_8_percent_l41_41664

theorem yield_percentage_of_stock_is_8_percent :
  let face_value := 100
  let dividend_rate := 0.20
  let market_price := 250
  annual_dividend = dividend_rate * face_value →
  yield_percentage = (annual_dividend / market_price) * 100 →
  yield_percentage = 8 := 
by
  sorry

end yield_percentage_of_stock_is_8_percent_l41_41664


namespace sum_of_coefficients_l41_41551

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) :
  (1 - 2 * x)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                  a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -1 :=
sorry

end sum_of_coefficients_l41_41551


namespace gcd_187_119_base5_l41_41226

theorem gcd_187_119_base5 :
  ∃ b : Nat, Nat.gcd 187 119 = 17 ∧ 17 = 3 * 5 + 2 ∧ 3 = 0 * 5 + 3 ∧ b = 3 * 10 + 2 := by
  sorry

end gcd_187_119_base5_l41_41226


namespace x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l41_41438

theorem x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq (x y : ℝ) :
  ¬((x > y) → (x^2 > y^2)) ∧ ¬((x^2 > y^2) → (x > y)) :=
by
  sorry

end x_gt_y_neither_sufficient_nor_necessary_for_x_sq_gt_y_sq_l41_41438


namespace ROI_diff_after_2_years_is_10_l41_41126

variables (investment_Emma : ℝ) (investment_Briana : ℝ)
variables (yield_Emma : ℝ) (yield_Briana : ℝ)
variables (years : ℝ)

def annual_ROI_Emma (investment_Emma yield_Emma : ℝ) : ℝ :=
  yield_Emma * investment_Emma

def annual_ROI_Briana (investment_Briana yield_Briana : ℝ) : ℝ :=
  yield_Briana * investment_Briana

def total_ROI_Emma (investment_Emma yield_Emma years : ℝ) : ℝ :=
  annual_ROI_Emma investment_Emma yield_Emma * years

def total_ROI_Briana (investment_Briana yield_Briana years : ℝ) : ℝ :=
  annual_ROI_Briana investment_Briana yield_Briana * years

def ROI_difference (investment_Emma investment_Briana yield_Emma yield_Briana years : ℝ) : ℝ :=
  total_ROI_Briana investment_Briana yield_Briana years - total_ROI_Emma investment_Emma yield_Emma years

theorem ROI_diff_after_2_years_is_10 :
  ROI_difference 300 500 0.15 0.10 2 = 10 :=
by
  sorry

end ROI_diff_after_2_years_is_10_l41_41126


namespace parabola_distance_l41_41029

theorem parabola_distance (p : ℝ) : 
  (∃ p: ℝ, y^2 = 10*x ∧ 2*p = 10) → p = 5 :=
by
  sorry

end parabola_distance_l41_41029


namespace arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l41_41069

theorem arcsin_one_half_eq_pi_over_six : Real.arcsin (1/2) = Real.pi/6 :=
by 
  sorry

theorem arccos_one_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi/3 :=
by 
  sorry

end arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l41_41069


namespace inverse_of_5_mod_35_l41_41293

theorem inverse_of_5_mod_35 : (5 * 28) % 35 = 1 :=
by
  sorry

end inverse_of_5_mod_35_l41_41293


namespace units_digit_of_sum_64_8_75_8_is_1_l41_41288

def units_digit_in_base_8_sum (a b : ℕ) : ℕ :=
  (a + b) % 8

theorem units_digit_of_sum_64_8_75_8_is_1 :
  units_digit_in_base_8_sum 0o64 0o75 = 1 :=
sorry

end units_digit_of_sum_64_8_75_8_is_1_l41_41288


namespace coordinates_with_respect_to_origin_l41_41771

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (2, -6)) : (x, y) = (2, -6) :=
by
  sorry

end coordinates_with_respect_to_origin_l41_41771


namespace coin_diameter_l41_41362

theorem coin_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  rw [h]
  norm_num

end coin_diameter_l41_41362


namespace speed_of_car_first_hour_98_l41_41502

def car_speed_in_first_hour_is_98 (x : ℕ) : Prop :=
  (70 + x) / 2 = 84 → x = 98

theorem speed_of_car_first_hour_98 (x : ℕ) (h : car_speed_in_first_hour_is_98 x) : x = 98 :=
  by
  sorry

end speed_of_car_first_hour_98_l41_41502


namespace find_product_l41_41782

theorem find_product
  (a b c d : ℝ) :
  3 * a + 2 * b + 4 * c + 6 * d = 60 →
  4 * (d + c) = b^2 →
  4 * b + 2 * c = a →
  c - 2 = d →
  a * b * c * d = 0 :=
by
  sorry

end find_product_l41_41782


namespace discount_difference_l41_41461

open Real

noncomputable def single_discount (B : ℝ) (d1 : ℝ) : ℝ :=
  B * (1 - d1)

noncomputable def successive_discounts (B : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (B * (1 - d2)) * (1 - d3)

theorem discount_difference (B : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  B = 12000 →
  d1 = 0.30 →
  d2 = 0.25 →
  d3 = 0.05 →
  abs (single_discount B d1 - successive_discounts B d2 d3) = 150 := by
  intros h_B h_d1 h_d2 h_d3
  rw [h_B, h_d1, h_d2, h_d3]
  rw [single_discount, successive_discounts]
  sorry

end discount_difference_l41_41461


namespace gcd_n_cube_plus_25_n_plus_3_l41_41619

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end gcd_n_cube_plus_25_n_plus_3_l41_41619


namespace contrapositive_of_x_squared_lt_one_is_true_l41_41123

variable {x : ℝ}

theorem contrapositive_of_x_squared_lt_one_is_true
  (h : ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) :
  ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1 :=
by
  sorry

end contrapositive_of_x_squared_lt_one_is_true_l41_41123


namespace find_p_and_q_solution_set_l41_41394

theorem find_p_and_q (p q : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) : 
  p = 5 ∧ q = -6 :=
sorry

theorem solution_set (p q : ℝ) (h_p : p = 5) (h_q : q = -6) : 
  ∀ x : ℝ, q * x^2 - p * x - 1 > 0 ↔ - (1 / 2) < x ∧ x < - (1 / 3) :=
sorry

end find_p_and_q_solution_set_l41_41394


namespace problem_statement_l41_41558

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {4, 5}
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Statement
theorem problem_statement : A ∩ (C_U B) = {2} :=
  sorry

end problem_statement_l41_41558


namespace min_value_of_m_n_l41_41356

variable {a b : ℝ}
variable (ab_eq_4 : a * b = 4)
variable (m : ℝ := b + 1 / a)
variable (n : ℝ := a + 1 / b)

theorem min_value_of_m_n (h1 : 0 < a) (h2 : 0 < b) : m + n = 5 :=
sorry

end min_value_of_m_n_l41_41356


namespace scoops_of_natural_seedless_raisins_l41_41785

theorem scoops_of_natural_seedless_raisins 
  (cost_natural : ℝ := 3.45) 
  (cost_golden : ℝ := 2.55) 
  (num_golden : ℝ := 20) 
  (cost_mixture : ℝ := 3) : 
  ∃ x : ℝ, (3.45 * x + 20 * 2.55 = 3 * (x + 20)) ∧ x = 20 :=
sorry

end scoops_of_natural_seedless_raisins_l41_41785


namespace songs_today_is_14_l41_41270

-- Define the number of songs Jeremy listened to yesterday
def songs_yesterday (x : ℕ) : ℕ := x

-- Define the number of songs Jeremy listened to today
def songs_today (x : ℕ) : ℕ := x + 5

-- Given conditions
def total_songs (x : ℕ) : Prop := songs_yesterday x + songs_today x = 23

-- Prove the number of songs Jeremy listened to today
theorem songs_today_is_14 : ∃ x: ℕ, total_songs x ∧ songs_today x = 14 :=
by {
  sorry
}

end songs_today_is_14_l41_41270


namespace necessary_and_sufficient_condition_for_parallel_lines_l41_41387

theorem necessary_and_sufficient_condition_for_parallel_lines (a l : ℝ) :
  (a = -1) ↔ (∀ x y : ℝ, ax + 3 * y + 3 = 0 → x + (a - 2) * y + l = 0) := 
sorry

end necessary_and_sufficient_condition_for_parallel_lines_l41_41387


namespace x_div_11p_is_integer_l41_41109

theorem x_div_11p_is_integer (x p : ℕ) (h1 : x > 0) (h2 : Prime p) (h3 : x = 66) : ∃ k : ℤ, x / (11 * p) = k := by
  sorry

end x_div_11p_is_integer_l41_41109


namespace range_of_a_decreasing_l41_41046

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≥ f y

theorem range_of_a_decreasing (a : ℝ) :
  (∃ a : ℝ, (1/6) ≤ a ∧ a < (1/3)) ↔ is_decreasing (f a) :=
sorry

end range_of_a_decreasing_l41_41046


namespace fractions_of_group_money_l41_41684

def moneyDistribution (m l n o : ℕ) (moeGave : ℕ) (lokiGave : ℕ) (nickGave : ℕ) : Prop :=
  moeGave = 1 / 5 * m ∧
  lokiGave = 1 / 4 * l ∧
  nickGave = 1 / 3 * n ∧
  moeGave = lokiGave ∧
  lokiGave = nickGave ∧
  o = moeGave + lokiGave + nickGave

theorem fractions_of_group_money (m l n o total : ℕ) :
  moneyDistribution m l n o 1 1 1 →
  total = m + l + n →
  (o : ℚ) / total = 1 / 4 :=
by sorry

end fractions_of_group_money_l41_41684


namespace problem_2002_multiples_l41_41708

theorem problem_2002_multiples :
  ∃ (n : ℕ), 
    n = 1800 ∧
    (∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 149 →
      2002 ∣ (10^j - 10^i) ↔ j - i ≡ 0 [MOD 6]) :=
sorry

end problem_2002_multiples_l41_41708


namespace coloring_impossible_l41_41990

-- Define vertices for the outer pentagon and inner star
inductive Vertex
| A | B | C | D | E | A' | B' | C' | D' | E'

open Vertex

-- Define segments in the figure
def Segments : List (Vertex × Vertex) :=
  [(A, B), (B, C), (C, D), (D, E), (E, A),
   (A, A'), (B, B'), (C, C'), (D, D'), (E, E'),
   (A', C), (C, E'), (E, B'), (B, D'), (D, A')]

-- Color type
inductive Color
| Red | Green | Blue

open Color

-- Condition for coloring: no two segments of the same color share a common endpoint
def distinct_color (c : Vertex → Color) : Prop :=
  ∀ (v1 v2 v3 : Vertex) (h1 : (v1, v2) ∈ Segments) (h2 : (v2, v3) ∈ Segments),
  c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v1 ≠ c v3

-- Statement of the proof problem
theorem coloring_impossible : ¬ ∃ (c : Vertex → Color), distinct_color c := 
by 
  sorry

end coloring_impossible_l41_41990


namespace train_speed_kmph_l41_41455

/-- Define the lengths of the train and bridge, as well as the time taken to cross the bridge. --/
def train_length : ℝ := 150
def bridge_length : ℝ := 150
def crossing_time_seconds : ℝ := 29.997600191984642

/-- Calculate the speed of the train in km/h. --/
theorem train_speed_kmph : 
  let total_distance := train_length + bridge_length
  let time_in_hours := crossing_time_seconds / 3600
  let speed_mph := total_distance / time_in_hours
  let speed_kmph := speed_mph / 1000
  speed_kmph = 36 := by
  /- Proof omitted -/
  sorry

end train_speed_kmph_l41_41455


namespace digitalEarthFunctions_l41_41817

axiom OptionA (F : Type) : Prop
axiom OptionB (F : Type) : Prop
axiom OptionC (F : Type) : Prop
axiom OptionD (F : Type) : Prop

axiom isRemoteSensing (F : Type) : OptionA F
axiom isGIS (F : Type) : OptionB F
axiom isGPS (F : Type) : OptionD F

theorem digitalEarthFunctions {F : Type} : OptionC F :=
sorry

end digitalEarthFunctions_l41_41817


namespace factorial_less_power_l41_41819

open Nat

noncomputable def factorial_200 : ℕ := 200!

noncomputable def power_100_200 : ℕ := 100 ^ 200

theorem factorial_less_power : factorial_200 < power_100_200 :=
by
  -- Proof goes here
  sorry

end factorial_less_power_l41_41819


namespace probability_not_exceed_60W_l41_41210

noncomputable def total_bulbs : ℕ := 250
noncomputable def bulbs_100W : ℕ := 100
noncomputable def bulbs_60W : ℕ := 50
noncomputable def bulbs_25W : ℕ := 50
noncomputable def bulbs_15W : ℕ := 50

noncomputable def probability_of_event (event : ℕ) (total : ℕ) : ℝ := 
  event / total

noncomputable def P_A : ℝ := probability_of_event bulbs_60W total_bulbs
noncomputable def P_B : ℝ := probability_of_event bulbs_25W total_bulbs
noncomputable def P_C : ℝ := probability_of_event bulbs_15W total_bulbs
noncomputable def P_D : ℝ := probability_of_event bulbs_100W total_bulbs

theorem probability_not_exceed_60W : 
  P_A + P_B + P_C = 3 / 5 :=
by
  sorry

end probability_not_exceed_60W_l41_41210


namespace at_least_one_pass_l41_41716

variable (n : ℕ) (p : ℝ)

theorem at_least_one_pass (h_p_range : 0 < p ∧ p < 1) :
  (1 - (1 - p) ^ n) = 1 - (1 - p) ^ n :=
sorry

end at_least_one_pass_l41_41716


namespace mary_shirts_left_l41_41545

theorem mary_shirts_left :
  let blue_shirts := 35
  let brown_shirts := 48
  let red_shirts := 27
  let yellow_shirts := 36
  let green_shirts := 18
  let blue_given_away := 4 / 5 * blue_shirts
  let brown_given_away := 5 / 6 * brown_shirts
  let red_given_away := 2 / 3 * red_shirts
  let yellow_given_away := 3 / 4 * yellow_shirts
  let green_given_away := 1 / 3 * green_shirts
  let blue_left := blue_shirts - blue_given_away
  let brown_left := brown_shirts - brown_given_away
  let red_left := red_shirts - red_given_away
  let yellow_left := yellow_shirts - yellow_given_away
  let green_left := green_shirts - green_given_away
  blue_left + brown_left + red_left + yellow_left + green_left = 45 := by
  sorry

end mary_shirts_left_l41_41545


namespace complement_set_l41_41875

def U := {x : ℝ | x > 0}
def A := {x : ℝ | x > 2}
def complement_U_A := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem complement_set :
  {x : ℝ | x ∈ U ∧ x ∉ A} = complement_U_A :=
sorry

end complement_set_l41_41875


namespace trains_at_initial_stations_l41_41630

-- Define the durations of round trips for each line.
def red_round_trip : ℕ := 14
def blue_round_trip : ℕ := 16
def green_round_trip : ℕ := 18

-- Define the total time we are analyzing.
def total_time : ℕ := 2016

-- Define the statement that needs to be proved.
theorem trains_at_initial_stations : 
  (total_time % red_round_trip = 0) ∧ 
  (total_time % blue_round_trip = 0) ∧ 
  (total_time % green_round_trip = 0) := 
by
  -- The proof can be added here.
  sorry

end trains_at_initial_stations_l41_41630


namespace max_m_value_l41_41864

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem max_m_value 
  (t : ℝ) 
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) <= x) : m ≤ 4 :=
sorry

end max_m_value_l41_41864


namespace solution_to_equation_l41_41623

theorem solution_to_equation (x y : ℤ) (h : x^6 - y^2 = 648) : 
  (x = 3 ∧ y = 9) ∨ 
  (x = -3 ∧ y = 9) ∨ 
  (x = 3 ∧ y = -9) ∨ 
  (x = -3 ∧ y = -9) :=
sorry

end solution_to_equation_l41_41623


namespace quadratic_to_completed_square_l41_41465

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 2

-- Define the completed square form of the function.
def completed_square_form (x : ℝ) : ℝ := (x + 1)^2 - 3

-- The theorem statement that needs to be proven.
theorem quadratic_to_completed_square :
  ∀ x : ℝ, quadratic_function x = completed_square_form x :=
by sorry

end quadratic_to_completed_square_l41_41465


namespace solve_quadratic_eq_solve_cubic_eq_l41_41898

-- Problem 1: Solve (x-1)^2 = 9
theorem solve_quadratic_eq (x : ℝ) (h : (x - 1) ^ 2 = 9) : x = 4 ∨ x = -2 := 
by 
  sorry

-- Problem 2: Solve (x+3)^3 / 3 - 9 = 0
theorem solve_cubic_eq (x : ℝ) (h : (x + 3) ^ 3 / 3 - 9 = 0) : x = 0 := 
by 
  sorry

end solve_quadratic_eq_solve_cubic_eq_l41_41898


namespace max_tries_needed_to_open_lock_l41_41975

-- Definitions and conditions
def num_buttons : ℕ := 9
def sequence_length : ℕ := 4
def opposite_trigrams : ℕ := 2  -- assumption based on the problem's example
def total_combinations : ℕ := 3024

theorem max_tries_needed_to_open_lock :
  (total_combinations - (8 * 1 * 7 * 6 + 8 * 6 * 1 * 6 + 8 * 6 * 4 * 1)) = 2208 :=
by
  sorry

end max_tries_needed_to_open_lock_l41_41975


namespace alpha_beta_property_l41_41266

theorem alpha_beta_property
  (α β : ℝ)
  (hαβ_roots : ∀ x : ℝ, (x = α ∨ x = β) → x^2 + x - 2023 = 0) :
  α^2 + 2 * α + β = 2022 :=
by
  sorry

end alpha_beta_property_l41_41266


namespace no_divisor_form_24k_20_l41_41078

theorem no_divisor_form_24k_20 (n : ℕ) : ¬ ∃ k : ℕ, 24 * k + 20 ∣ 3^n + 1 :=
sorry

end no_divisor_form_24k_20_l41_41078


namespace range_of_a_l41_41733

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end range_of_a_l41_41733


namespace gcd_solutions_l41_41225

theorem gcd_solutions (x m n p: ℤ) (h_eq: x * (4 * x - 5) = 7) (h_gcd: Int.gcd m (Int.gcd n p) = 1)
  (h_form: ∃ x1 x2: ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p) : m + n + p = 150 :=
by
  have disc_eq : 25 + 112 = 137 :=
    by norm_num
  sorry

end gcd_solutions_l41_41225


namespace unique_solution_sin_tan_eq_l41_41440

noncomputable def S (x : ℝ) : ℝ := Real.tan (Real.sin x) - Real.sin x

theorem unique_solution_sin_tan_eq (h : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.arcsin (1/2) → S x < S y) :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) := by
sorry

end unique_solution_sin_tan_eq_l41_41440


namespace solve_percentage_chromium_first_alloy_l41_41650

noncomputable def percentage_chromium_first_alloy (x : ℝ) : Prop :=
  let w1 := 15 -- weight of the first alloy
  let c2 := 10 -- percentage of chromium in the second alloy
  let w2 := 35 -- weight of the second alloy
  let w_total := 50 -- total weight of the new alloy formed by mixing
  let c_new := 10.6 -- percentage of chromium in the new alloy
  -- chromium percentage equation
  ((x / 100) * w1 + (c2 / 100) * w2) = (c_new / 100) * w_total

theorem solve_percentage_chromium_first_alloy : percentage_chromium_first_alloy 12 :=
  sorry -- proof goes here

end solve_percentage_chromium_first_alloy_l41_41650


namespace distance_traveled_by_car_l41_41416

theorem distance_traveled_by_car :
  let total_distance := 90
  let distance_by_foot := (1 / 5 : ℝ) * total_distance
  let distance_by_bus := (2 / 3 : ℝ) * total_distance
  let distance_by_car := total_distance - (distance_by_foot + distance_by_bus)
  distance_by_car = 12 :=
by
  sorry

end distance_traveled_by_car_l41_41416


namespace exists_positive_M_l41_41007

open Set

noncomputable def f (x : ℝ) : ℝ := sorry

theorem exists_positive_M 
  (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, f x > 0)
  (h₂ : ∀ x ∈ Ioo (0 : ℝ) 1, f (2 * x / (1 + x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Ioo (0 : ℝ) 1, f x ≤ M :=
sorry

end exists_positive_M_l41_41007


namespace price_decrease_for_original_price_l41_41600

theorem price_decrease_for_original_price (P : ℝ) (h : P > 0) :
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  decrease = 20 :=
by
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  sorry

end price_decrease_for_original_price_l41_41600


namespace find_a_plus_b_l41_41647

def star (a b : ℕ) : ℕ := a^b - a*b + 5

theorem find_a_plus_b (a b : ℕ) (ha : 2 ≤ a) (hb : 3 ≤ b) (h : star a b = 13) : a + b = 6 :=
  sorry

end find_a_plus_b_l41_41647


namespace minimum_value_of_2a5_a4_l41_41249

variable {a : ℕ → ℝ} {q : ℝ}

-- Defining that the given sequence is geometric, i.e., a_{n+1} = a_n * q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

-- The condition given in the problem is
def condition (a : ℕ → ℝ) : Prop :=
2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

-- The sequence is positive
def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem minimum_value_of_2a5_a4 (h_geom : is_geometric_sequence a q) (h_cond : condition a) (h_pos : positive_sequence a) (h_q : q > 0) :
  2 * a 5 + a 4 = 12 * Real.sqrt 3 :=
sorry

end minimum_value_of_2a5_a4_l41_41249


namespace polar_distance_l41_41193

theorem polar_distance {r1 θ1 r2 θ2 : ℝ} (A : r1 = 1 ∧ θ1 = π/6) (B : r2 = 3 ∧ θ2 = 5*π/6) : 
  (r1^2 + r2^2 - 2*r1*r2 * Real.cos (θ2 - θ1)) = 13 :=
  sorry

end polar_distance_l41_41193


namespace point_M_coordinates_l41_41620

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 4 * x

-- Define the condition given in the problem: instantaneous rate of change
def rate_of_change (a : ℝ) : Prop := f' a = -4

-- Define the point on the curve
def point_M (a b : ℝ) : Prop := f a = b

-- Proof statement
theorem point_M_coordinates : 
  ∃ (a b : ℝ), rate_of_change a ∧ point_M a b ∧ a = -1 ∧ b = 3 :=  
by
  sorry

end point_M_coordinates_l41_41620


namespace sum_greater_than_four_l41_41957

theorem sum_greater_than_four (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hprod : x * y > x + y) : x + y > 4 :=
by
  sorry

end sum_greater_than_four_l41_41957


namespace base_seven_to_ten_l41_41791

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l41_41791


namespace last_three_digits_of_5_power_odd_l41_41821

theorem last_three_digits_of_5_power_odd (n : ℕ) (h : n % 2 = 1) : (5 ^ n) % 1000 = 125 :=
sorry

end last_three_digits_of_5_power_odd_l41_41821


namespace tan_2theta_sin_cos_fraction_l41_41021

variable {θ : ℝ} (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1)

-- Part (I)
theorem tan_2theta (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : Real.tan (2 * θ) = 4 / 3 :=
by sorry

-- Part (II)
theorem sin_cos_fraction (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 :=
by sorry

end tan_2theta_sin_cos_fraction_l41_41021


namespace compare_neg_fractions_l41_41000

theorem compare_neg_fractions :
  - (10 / 11 : ℤ) > - (11 / 12 : ℤ) :=
sorry

end compare_neg_fractions_l41_41000


namespace pure_ghee_percentage_l41_41118

theorem pure_ghee_percentage (Q : ℝ) (vanaspati_percentage : ℝ:= 0.40) (additional_pure_ghee : ℝ := 10) (new_vanaspati_percentage : ℝ := 0.20) (original_quantity : ℝ := 10) :
  (Q = original_quantity) ∧ (vanaspati_percentage = 0.40) ∧ (additional_pure_ghee = 10) ∧ (new_vanaspati_percentage = 0.20) →
  (100 - (vanaspati_percentage * 100)) = 60 :=
by
  sorry

end pure_ghee_percentage_l41_41118


namespace train_speed_kmh_l41_41260

def man_speed_kmh : ℝ := 3 -- The man's speed in km/h
def train_length_m : ℝ := 110 -- The train's length in meters
def passing_time_s : ℝ := 12 -- Time taken to pass the man in seconds

noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600 -- Convert man's speed to m/s

theorem train_speed_kmh :
  (110 / 12) - (5 / 6) * (3600 / 1000) = 30 := by
  -- Omitted steps will go here
  sorry

end train_speed_kmh_l41_41260


namespace boys_meet_once_excluding_start_finish_l41_41565

theorem boys_meet_once_excluding_start_finish 
    (d : ℕ) 
    (h1 : 0 < d) 
    (boy1_speed : ℕ) (boy2_speed : ℕ) 
    (h2 : boy1_speed = 6) (h3 : boy2_speed = 10)
    (relative_speed : ℕ) (h4 : relative_speed = boy1_speed + boy2_speed) 
    (time_to_meet_A_again : ℕ) (h5 : time_to_meet_A_again = d / relative_speed) 
    (boy1_laps_per_sec boy2_laps_per_sec : ℕ) 
    (h6 : boy1_laps_per_sec = boy1_speed / d) 
    (h7 : boy2_laps_per_sec = boy2_speed / d)
    (lcm_laps : ℕ) (h8 : lcm_laps = Nat.lcm 6 10)
    (meetings_per_lap : ℕ) (h9 : meetings_per_lap = lcm_laps / d)
    (total_meetings : ℕ) (h10 : total_meetings = meetings_per_lap * time_to_meet_A_again)
  : total_meetings = 1 := by
  sorry

end boys_meet_once_excluding_start_finish_l41_41565


namespace minimum_a_l41_41534

theorem minimum_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → 
  a ≥ -5/2 :=
sorry

end minimum_a_l41_41534


namespace prob_task1_and_not_task2_l41_41770

def prob_task1_completed : ℚ := 5 / 8
def prob_task2_completed : ℚ := 3 / 5

theorem prob_task1_and_not_task2 : 
  ((prob_task1_completed) * (1 - prob_task2_completed)) = 1 / 4 := 
by 
  sorry

end prob_task1_and_not_task2_l41_41770


namespace positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l41_41279

theorem positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5 : 
  ∃ (x : ℕ), (x = 594) ∧ (18 ∣ x) ∧ (24 ≤ Real.sqrt (x) ∧ Real.sqrt (x) ≤ 24.5) := 
by 
  sorry

end positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l41_41279


namespace quotient_correct_l41_41546

def dividend : ℤ := 474232
def divisor : ℤ := 800
def remainder : ℤ := -968

theorem quotient_correct : (dividend + abs remainder) / divisor = 594 := by
  sorry

end quotient_correct_l41_41546


namespace find_divisor_l41_41532

theorem find_divisor (d : ℕ) (h1 : 109 % d = 1) (h2 : 109 / d = 9) : d = 12 := by
  sorry

end find_divisor_l41_41532


namespace max_value_of_ratio_l41_41357

theorem max_value_of_ratio (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : 
  ∃ z, z = (x / y) ∧ z ≤ 1 := sorry

end max_value_of_ratio_l41_41357


namespace jane_evening_pages_l41_41970

theorem jane_evening_pages :
  ∀ (P : ℕ), (7 * (5 + P) = 105) → P = 10 :=
by
  intros P h
  sorry

end jane_evening_pages_l41_41970


namespace cost_price_is_92_percent_l41_41439

noncomputable def cost_price_percentage_of_selling_price (profit_percentage : ℝ) : ℝ :=
  let CP := (1 / ((profit_percentage / 100) + 1))
  CP * 100

theorem cost_price_is_92_percent (profit_percentage : ℝ) (h : profit_percentage = 8.695652173913043) :
  cost_price_percentage_of_selling_price profit_percentage = 92 :=
by
  rw [h]
  -- now we need to show that cost_price_percentage_of_selling_price 8.695652173913043 = 92
  -- by definition, cost_price_percentage_of_selling_price 8.695652173913043 is:
  -- let CP := 1 / (8.695652173913043 / 100 + 1)
  -- CP * 100 = (1 / (8.695652173913043 / 100 + 1)) * 100
  sorry

end cost_price_is_92_percent_l41_41439


namespace alec_correct_problems_l41_41953

-- Definitions of conditions and proof problem
theorem alec_correct_problems (c w : ℕ) (s : ℕ) (H1 : s = 30 + 4 * c - w) (H2 : s > 90)
  (H3 : ∀ s', 90 < s' ∧ s' < s → ¬(∃ c', ∃ w', s' = 30 + 4 * c' - w')) :
  c = 16 :=
by
  sorry

end alec_correct_problems_l41_41953


namespace complement_union_l41_41223

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l41_41223


namespace chewing_gums_count_l41_41988

-- Given conditions
def num_chocolate_bars : ℕ := 55
def num_candies : ℕ := 40
def total_treats : ℕ := 155

-- Definition to be proven
def num_chewing_gums : ℕ := total_treats - (num_chocolate_bars + num_candies)

-- Theorem statement
theorem chewing_gums_count : num_chewing_gums = 60 :=
by 
  -- here would be the proof steps, but it's omitted as per the instruction
  sorry

end chewing_gums_count_l41_41988


namespace intersection_eq_l41_41057

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by sorry

end intersection_eq_l41_41057


namespace books_problem_l41_41071

theorem books_problem
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) :
  M = 10 :=
by
  sorry

end books_problem_l41_41071


namespace perpendicular_unit_vector_exists_l41_41950

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end perpendicular_unit_vector_exists_l41_41950


namespace total_flowers_sold_l41_41799

def flowers_sold_on_monday : ℕ := 4
def flowers_sold_on_tuesday : ℕ := 8
def flowers_sold_on_friday : ℕ := 2 * flowers_sold_on_monday

theorem total_flowers_sold : flowers_sold_on_monday + flowers_sold_on_tuesday + flowers_sold_on_friday = 20 := by
  sorry

end total_flowers_sold_l41_41799


namespace sum_of_tangents_slopes_at_vertices_l41_41597

noncomputable def curve (x : ℝ) := (x + 3) * (x ^ 2 + 3)

theorem sum_of_tangents_slopes_at_vertices {x_A x_B x_C : ℝ}
  (h1 : curve x_A = x_A * (x_A ^ 2 + 6 * x_A + 9) + 3)
  (h2 : curve x_B = x_B * (x_B ^ 2 + 6 * x_B + 9) + 3)
  (h3 : curve x_C = x_C * (x_C ^ 2 + 6 * x_C + 9) + 3)
  : (3 * x_A ^ 2 + 6 * x_A + 3) + (3 * x_B ^ 2 + 6 * x_B + 3) + (3 * x_C ^ 2 + 6 * x_C + 3) = 237 :=
sorry

end sum_of_tangents_slopes_at_vertices_l41_41597


namespace frequency_rate_identity_l41_41195

theorem frequency_rate_identity (n : ℕ) : 
  (36 : ℕ) / (n : ℕ) = (0.25 : ℝ) → 
  n = 144 := by
  sorry

end frequency_rate_identity_l41_41195


namespace min_value_ineq_solve_ineq_l41_41447

theorem min_value_ineq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a^3 + 1 / b^3 + 1 / c^3 + 3 * a * b * c) ≥ 6 :=
sorry

theorem solve_ineq (x : ℝ) (h : |x + 1| - 2 * x < 6) : x > -7/3 :=
sorry

end min_value_ineq_solve_ineq_l41_41447


namespace alice_age_l41_41343

theorem alice_age (x : ℕ) (h1 : ∃ n : ℕ, x - 4 = n^2) (h2 : ∃ m : ℕ, x + 2 = m^3) : x = 58 :=
sorry

end alice_age_l41_41343


namespace solve_3x_plus_7y_eq_23_l41_41096

theorem solve_3x_plus_7y_eq_23 :
  ∃ (x y : ℕ), 3 * x + 7 * y = 23 ∧ x = 3 ∧ y = 2 := by
sorry

end solve_3x_plus_7y_eq_23_l41_41096


namespace quadratic_expression_value_l41_41553

variable (x y : ℝ)

theorem quadratic_expression_value (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 100 := 
by 
  sorry

end quadratic_expression_value_l41_41553


namespace below_sea_level_representation_l41_41867

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end below_sea_level_representation_l41_41867


namespace original_price_of_house_l41_41328

theorem original_price_of_house (P: ℝ) (sold_price: ℝ) (profit: ℝ) (commission: ℝ):
  sold_price = 100000 ∧ profit = 0.20 ∧ commission = 0.05 → P = 86956.52 :=
by
  sorry -- Proof not provided

end original_price_of_house_l41_41328


namespace union_M_N_l41_41463

noncomputable def M : Set ℝ := { x | x^2 - 3 * x = 0 }
noncomputable def N : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }

theorem union_M_N : M ∪ N = {0, 2, 3} :=
by {
  sorry
}

end union_M_N_l41_41463


namespace remainder_2023_div_73_l41_41637

theorem remainder_2023_div_73 : 2023 % 73 = 52 := 
by
  -- Proof goes here
  sorry

end remainder_2023_div_73_l41_41637


namespace third_side_range_l41_41489

theorem third_side_range (a : ℝ) (h₃ : 0 < a ∧ a ≠ 0) (h₅ : 0 < a ∧ a ≠ 0): 
  (2 < a ∧ a < 8) ↔ (3 - 5 < a ∧ a < 3 + 5) :=
by
  sorry

end third_side_range_l41_41489


namespace max_sequence_sum_l41_41738

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a1 d : α) (n : ℕ) : α :=
  a1 + d * n

noncomputable def sequenceSum (a1 d : α) (n : ℕ) : α :=
  n * (a1 + (a1 + d * (n - 1))) / 2

theorem max_sequence_sum (a1 d : α) (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 10)
    (h1 : d < 0) (h2 : sequenceSum a1 d 5 = sequenceSum a1 d 10) :
    n = 7 ∨ n = 8 :=
  sorry

end max_sequence_sum_l41_41738


namespace hyperbola_foci_product_l41_41464

theorem hyperbola_foci_product
  (F1 F2 P : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 5, 0))
  (hF2 : F2 = (Real.sqrt 5, 0))
  (hP : P.1 ^ 2 / 4 - P.2 ^ 2 = 1)
  (hDot : (P.1 + Real.sqrt 5) * (P.1 - Real.sqrt 5) + P.2 ^ 2 = 0) :
  (Real.sqrt ((P.1 + Real.sqrt 5) ^ 2 + P.2 ^ 2)) * (Real.sqrt ((P.1 - Real.sqrt 5) ^ 2 + P.2 ^ 2)) = 2 :=
sorry

end hyperbola_foci_product_l41_41464


namespace student_correct_answers_l41_41277

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 73) : C = 91 :=
sorry

end student_correct_answers_l41_41277


namespace problem_statement_l41_41617

noncomputable def r (C: ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def A (r: ℝ) : ℝ := Real.pi * r^2

noncomputable def combined_area_difference (C1 C2 C3: ℝ) : ℝ :=
  let r1 := r C1
  let r2 := r C2
  let r3 := r C3
  let A1 := A r1
  let A2 := A r2
  let A3 := A r3
  (A3 - A1) - A2

theorem problem_statement : combined_area_difference 528 704 880 = -9.76 :=
by
  sorry

end problem_statement_l41_41617


namespace possible_analytical_expression_for_f_l41_41172

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.cos (2 * x))

theorem possible_analytical_expression_for_f :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ x : ℝ, f (x - π/4) = f (-x)) ∧
  (∀ x : ℝ, π/8 < x ∧ x < π/2 → f x < f (x - 1)) :=
by
  sorry

end possible_analytical_expression_for_f_l41_41172


namespace geom_series_sum_l41_41852

noncomputable def geom_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1) 

theorem geom_series_sum (S : ℕ) (a r n : ℕ) (eq1 : a = 1) (eq2 : r = 3)
  (eq3 : 19683 = a * r^(n-1)) (S_eq : S = geom_sum a r n) : 
  S = 29524 :=
by
  sorry

end geom_series_sum_l41_41852


namespace decrease_hours_by_13_percent_l41_41539

theorem decrease_hours_by_13_percent (W H : ℝ) (hW_pos : W > 0) (hH_pos : H > 0) :
  let W_new := 1.15 * W
  let H_new := H / 1.15
  let income_decrease_percentage := (1 - H_new / H) * 100
  abs (income_decrease_percentage - 13.04) < 0.01 := 
by
  sorry

end decrease_hours_by_13_percent_l41_41539


namespace britney_has_more_chickens_l41_41018

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end britney_has_more_chickens_l41_41018


namespace min_value_of_function_l41_41790

theorem min_value_of_function (x : ℝ) (h : x > 2) : (x + 1 / (x - 2)) ≥ 4 :=
  sorry

end min_value_of_function_l41_41790


namespace polynomial_approx_eq_l41_41044

theorem polynomial_approx_eq (x : ℝ) (h : x^4 - 4*x^3 + 4*x^2 + 4 = 4.999999999999999) : x = 1 :=
sorry

end polynomial_approx_eq_l41_41044


namespace find_k_of_division_property_l41_41509

theorem find_k_of_division_property (k : ℝ) :
  (3 * (1 / 3)^3 - k * (1 / 3)^2 + 4) % (3 * (1 / 3) - 1) = 5 → k = -8 :=
by sorry

end find_k_of_division_property_l41_41509


namespace print_rolls_sold_l41_41893

-- Defining the variables and conditions
def num_sold := 480
def total_amount := 2340
def solid_price := 4
def print_price := 6

-- Proposed theorem statement
theorem print_rolls_sold (S P : ℕ) (h1 : S + P = num_sold) (h2 : solid_price * S + print_price * P = total_amount) : P = 210 := sorry

end print_rolls_sold_l41_41893


namespace slowerPainterDuration_l41_41528

def slowerPainterStartTime : ℝ := 14 -- 2:00 PM in 24-hour format
def fasterPainterStartTime : ℝ := slowerPainterStartTime + 3 -- 3 hours later
def finishTime : ℝ := 24.6 -- 0.6 hours past midnight

theorem slowerPainterDuration :
  finishTime - slowerPainterStartTime = 10.6 :=
by
  sorry

end slowerPainterDuration_l41_41528


namespace total_cost_l41_41227

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l41_41227


namespace Willy_Lucy_more_crayons_l41_41883

def Willy_initial : ℕ := 1400
def Lucy_initial : ℕ := 290
def Max_crayons : ℕ := 650
def Willy_giveaway_percent : ℚ := 25 / 100
def Lucy_giveaway_percent : ℚ := 10 / 100

theorem Willy_Lucy_more_crayons :
  let Willy_remaining := Willy_initial - Willy_initial * Willy_giveaway_percent
  let Lucy_remaining := Lucy_initial - Lucy_initial * Lucy_giveaway_percent
  Willy_remaining + Lucy_remaining - Max_crayons = 661 := by
  sorry

end Willy_Lucy_more_crayons_l41_41883


namespace cube_decomposition_smallest_number_91_l41_41093

theorem cube_decomposition_smallest_number_91 (m : ℕ) (h1 : 0 < m) (h2 : (91 - 1) / 2 + 2 = m * m - m + 1) : m = 10 := by {
  sorry
}

end cube_decomposition_smallest_number_91_l41_41093


namespace greatest_large_chips_l41_41609

theorem greatest_large_chips (s l p : ℕ) (h1 : s + l = 80) (h2 : s = l + p) (hp : Nat.Prime p) : l ≤ 39 :=
by
  sorry

end greatest_large_chips_l41_41609


namespace granger_bought_4_loaves_of_bread_l41_41724

-- Define the prices of items
def price_of_spam : Nat := 3
def price_of_pb : Nat := 5
def price_of_bread : Nat := 2

-- Define the quantities bought by Granger
def qty_spam : Nat := 12
def qty_pb : Nat := 3
def total_amount_paid : Nat := 59

-- The problem statement in Lean: Prove the number of loaves of bread bought
theorem granger_bought_4_loaves_of_bread :
  (qty_spam * price_of_spam) + (qty_pb * price_of_pb) + (4 * price_of_bread) = total_amount_paid :=
sorry

end granger_bought_4_loaves_of_bread_l41_41724


namespace board_train_immediately_probability_l41_41102

-- Define conditions
def total_time : ℝ := 10
def favorable_time : ℝ := 1

-- Define the probability P(A) as favorable_time / total_time
noncomputable def probability_A : ℝ := favorable_time / total_time

-- State the proposition to prove that the probability is 1/10
theorem board_train_immediately_probability : probability_A = 1 / 10 :=
by sorry

end board_train_immediately_probability_l41_41102


namespace roses_in_vase_l41_41983

theorem roses_in_vase (initial_roses added_roses : ℕ) (h₀ : initial_roses = 10) (h₁ : added_roses = 8) : initial_roses + added_roses = 18 :=
by
  sorry

end roses_in_vase_l41_41983


namespace norma_initial_cards_l41_41979

def initial_card_count (lost: ℕ) (left: ℕ) : ℕ :=
  lost + left

theorem norma_initial_cards : initial_card_count 70 18 = 88 :=
  by
    -- skipping proof
    sorry

end norma_initial_cards_l41_41979


namespace defective_probability_l41_41601

variable (total_products defective_products qualified_products : ℕ)
variable (first_draw_defective second_draw_defective : Prop)

-- Definitions of the problem
def total_prods := 10
def def_prods := 4
def qual_prods := 6
def p_A := def_prods / total_prods
def p_AB := (def_prods / total_prods) * ((def_prods - 1) / (total_prods - 1))
def p_B_given_A := p_AB / p_A

-- Theorem: The probability of drawing a defective product on the second draw given the first was defective is 1/3.
theorem defective_probability 
  (hp1 : total_products = total_prods)
  (hp2 : defective_products = def_prods)
  (hp3 : qualified_products = qual_prods)
  (pA_eq : p_A = 2 / 5)
  (pAB_eq : p_AB = 2 / 15) : 
  p_B_given_A = 1 / 3 := sorry

end defective_probability_l41_41601


namespace min_distance_racetracks_l41_41495

theorem min_distance_racetracks : 
  ∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (((B.1 - 1) ^ 2) / 16 + (B.2 ^ 2) / 4 = 1) → 
  dist A B ≥ (Real.sqrt 33 - 3) / 3 := by
  sorry

end min_distance_racetracks_l41_41495


namespace freshmen_minus_sophomores_eq_24_l41_41220

def total_students := 800
def percent_juniors := 27 / 100
def percent_not_sophomores := 75 / 100
def number_seniors := 160

def number_juniors := percent_juniors * total_students
def number_not_sophomores := percent_not_sophomores * total_students
def number_sophomores := total_students - number_not_sophomores
def number_freshmen := total_students - (number_juniors + number_sophomores + number_seniors)

theorem freshmen_minus_sophomores_eq_24 :
  number_freshmen - number_sophomores = 24 :=
sorry

end freshmen_minus_sophomores_eq_24_l41_41220


namespace min_distance_eq_5_l41_41239

-- Define the conditions
def condition1 (a b : ℝ) : Prop := b = 4 * Real.log a - a^2
def condition2 (c d : ℝ) : Prop := d = 2 * c + 2

-- Define the function to prove the minimum value
def minValue (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- The main theorem statement
theorem min_distance_eq_5 (a b c d : ℝ) (ha : a > 0) (h1: condition1 a b) (h2: condition2 c d) : 
  ∃ a c b d, minValue a b c d = 5 := 
sorry

end min_distance_eq_5_l41_41239


namespace solve_eq_l41_41429

theorem solve_eq {x : ℝ} (h : x * (x - 1) = x) : x = 0 ∨ x = 2 := 
by {
    sorry
}

end solve_eq_l41_41429


namespace shortest_distance_dasha_vasya_l41_41097

variables (dasha galia asya borya vasya : Type)
variables (dist : ∀ (a b : Type), ℕ)
variables (dist_dasha_galia : dist dasha galia = 15)
variables (dist_vasya_galia : dist vasya galia = 17)
variables (dist_asya_galia : dist asya galia = 12)
variables (dist_galia_borya : dist galia borya = 10)
variables (dist_asya_borya : dist asya borya = 8)

theorem shortest_distance_dasha_vasya : dist dasha vasya = 18 :=
by sorry

end shortest_distance_dasha_vasya_l41_41097


namespace gimbap_total_cost_l41_41697

theorem gimbap_total_cost :
  let basic_gimbap_cost := 2000
  let tuna_gimbap_cost := 3500
  let red_pepper_gimbap_cost := 3000
  let beef_gimbap_cost := 4000
  let nude_gimbap_cost := 3500
  let cost_of_two gimbaps := (tuna_gimbap_cost * 2) + (beef_gimbap_cost * 2) + (nude_gimbap_cost * 2)
  cost_of_two gimbaps = 22000 := 
by 
  sorry

end gimbap_total_cost_l41_41697


namespace average_age_of_team_l41_41529

theorem average_age_of_team (A : ℝ) : 
    (11 * A =
         9 * (A - 1) + 53) → 
    A = 31 := 
by 
  sorry

end average_age_of_team_l41_41529


namespace light_coloured_blocks_in_tower_l41_41100

theorem light_coloured_blocks_in_tower :
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  total_light_coloured_blocks = 20 :=
by
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  show total_light_coloured_blocks = 20
  sorry

end light_coloured_blocks_in_tower_l41_41100


namespace hyperbola_standard_equation_l41_41562

theorem hyperbola_standard_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h_real_axis : 2 * a = 4 * Real.sqrt 2) (h_eccentricity : a / Real.sqrt (a^2 + b^2) = Real.sqrt 6 / 2) :
    (a = 2 * Real.sqrt 2) ∧ (b = 2) → ∀ x y : ℝ, (x^2)/8 - (y^2)/4 = 1 :=
sorry

end hyperbola_standard_equation_l41_41562


namespace problem_solution_l41_41377

theorem problem_solution (x y : ℚ) (h1 : |x| + x + y - 2 = 14) (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := 
by
  -- It remains to prove
  sorry

end problem_solution_l41_41377


namespace pencils_multiple_of_28_l41_41680

theorem pencils_multiple_of_28 (students pens pencils : ℕ) 
  (h1 : students = 28) 
  (h2 : pens = 1204) 
  (h3 : ∃ k, pens = students * k) 
  (h4 : ∃ n, pencils = students * n) : 
  ∃ m, pencils = 28 * m :=
by
  sorry

end pencils_multiple_of_28_l41_41680


namespace ben_fewer_pints_than_kathryn_l41_41424

-- Define the conditions
def annie_picked := 8
def kathryn_picked := annie_picked + 2
def total_picked := 25

-- Add noncomputable because constants are involved
noncomputable def ben_picked : ℕ := total_picked - (annie_picked + kathryn_picked)

theorem ben_fewer_pints_than_kathryn : ben_picked = kathryn_picked - 3 := 
by 
  -- The problem statement does not require proof body
  sorry

end ben_fewer_pints_than_kathryn_l41_41424


namespace bushes_for_zucchinis_l41_41559

def bushes_yield := 10 -- containers per bush
def container_to_zucchini := 3 -- containers per zucchini
def zucchinis_required := 60 -- total zucchinis needed

theorem bushes_for_zucchinis (hyld : bushes_yield = 10) (ctz : container_to_zucchini = 3) (zreq : zucchinis_required = 60) :
  ∃ bushes : ℕ, bushes = 60 * container_to_zucchini / bushes_yield :=
sorry

end bushes_for_zucchinis_l41_41559


namespace largest_consecutive_integer_product_2520_l41_41912

theorem largest_consecutive_integer_product_2520 :
  ∃ (n : ℕ), n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧ (n + 3) = 8 :=
by {
  sorry
}

end largest_consecutive_integer_product_2520_l41_41912


namespace factorize_polynomial_l41_41568

theorem factorize_polynomial (a b c : ℚ) : 
  b^2 - c^2 + a * (a + 2 * b) = (a + b + c) * (a + b - c) :=
by
  sorry

end factorize_polynomial_l41_41568


namespace value_of_b_l41_41890

theorem value_of_b : (15^2 * 9^2 * 356 = 6489300) :=
by 
  sorry

end value_of_b_l41_41890


namespace expand_and_simplify_l41_41215

variable (y : ℝ)

theorem expand_and_simplify :
  -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 :=
  sorry

end expand_and_simplify_l41_41215


namespace probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l41_41590

open BigOperators

/-- Suppose 30 balls are tossed independently and at random into one 
of the 6 bins. Let p be the probability that one bin ends up with 3 
balls, another with 6 balls, another with 5, another with 4, another 
with 2, and the last one with 10 balls. Let q be the probability 
that each bin ends up with 5 balls. Calculate p / q. 
-/
theorem probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5 :
  (Nat.factorial 5 ^ 6 : ℚ) / ((Nat.factorial 3:ℚ) * Nat.factorial 6 * Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 10) = 0.125 := 
sorry

end probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l41_41590


namespace inequality_proof_l41_41747

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 := 
by 
  sorry

end inequality_proof_l41_41747


namespace unique_y_for_star_l41_41251

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

theorem unique_y_for_star : (∀ y : ℝ, star 4 y = 17 → y = 0) ∧ (∃! y : ℝ, star 4 y = 17) := by
  sorry

end unique_y_for_star_l41_41251


namespace number_of_envelopes_l41_41095

theorem number_of_envelopes (total_weight_grams : ℕ) (weight_per_envelope_grams : ℕ) (n : ℕ) :
  total_weight_grams = 7480 ∧ weight_per_envelope_grams = 8500 ∧ n = 880 → total_weight_grams = n * weight_per_envelope_grams := 
sorry

end number_of_envelopes_l41_41095


namespace initial_pieces_l41_41158

-- Define the conditions
def pieces_used : ℕ := 156
def pieces_left : ℕ := 744

-- Define the total number of pieces of paper Isabel bought initially
def total_pieces : ℕ := pieces_used + pieces_left

-- State the theorem that we need to prove
theorem initial_pieces (h1 : pieces_used = 156) (h2 : pieces_left = 744) : total_pieces = 900 :=
by
  sorry

end initial_pieces_l41_41158


namespace Lacy_correct_percent_l41_41446

theorem Lacy_correct_percent (x : ℝ) (h1 : 7 * x > 0) : ((5 * 100) / 7) = 71.43 :=
by
  sorry

end Lacy_correct_percent_l41_41446


namespace min_value_reciprocal_sum_l41_41861

theorem min_value_reciprocal_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 2) : 
  (∃ c : ℝ, c = (1/x) + (1/y) + (1/z) ∧ c ≥ 9/2) :=
by
  -- proof would go here
  sorry

end min_value_reciprocal_sum_l41_41861


namespace gcd_136_1275_l41_41773

theorem gcd_136_1275 : Nat.gcd 136 1275 = 17 := by
sorry

end gcd_136_1275_l41_41773


namespace find_formula_l41_41557

variable (x : ℕ) (y : ℕ)

theorem find_formula (h1: (x = 2 ∧ y = 10) ∨ (x = 3 ∧ y = 21) ∨ (x = 4 ∧ y = 38) ∨ (x = 5 ∧ y = 61) ∨ (x = 6 ∧ y = 90)) :
  y = 3 * x^2 - 2 * x + 2 :=
  sorry

end find_formula_l41_41557


namespace find_d_from_factor_condition_l41_41070

theorem find_d_from_factor_condition (d : ℚ) : (∀ x, x = 5 → d * x^4 + 13 * x^3 - 2 * d * x^2 - 58 * x + 65 = 0) → d = -28 / 23 :=
by
  intro h
  sorry

end find_d_from_factor_condition_l41_41070


namespace inscribed_squares_ratio_l41_41910

theorem inscribed_squares_ratio (x y : ℝ) 
  (h₁ : 5^2 + 12^2 = 13^2)
  (h₂ : x = 144 / 17)
  (h₃ : y = 5) :
  x / y = 144 / 85 :=
by
  sorry

end inscribed_squares_ratio_l41_41910


namespace sum_mod_18_l41_41734

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end sum_mod_18_l41_41734


namespace circle_radius_l41_41009
open Real

theorem circle_radius (d : ℝ) (h_diam : d = 24) : d / 2 = 12 :=
by
  -- The proof will be here
  sorry

end circle_radius_l41_41009


namespace total_students_in_high_school_l41_41577

theorem total_students_in_high_school (sample_size first_year third_year second_year : ℕ) (total_students : ℕ) 
  (h1 : sample_size = 45) 
  (h2 : first_year = 20) 
  (h3 : third_year = 10) 
  (h4 : second_year = 300)
  (h5 : sample_size = first_year + third_year + (sample_size - first_year - third_year)) :
  total_students = 900 :=
by
  sorry

end total_students_in_high_school_l41_41577


namespace problem1_problem2_l41_41425

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- Problem 1: Find x such that f(x) < -2 when a = 1
theorem problem1 : 
  {x : ℝ | f x 1 < -2} = {x | x > 3 / 2} :=
sorry

-- Problem 2: Find the range of values for 'a' when -2 + f(y) ≤ f(x) ≤ 2 + f(y) for all x, y ∈ ℝ
theorem problem2 : 
  (∀ x y : ℝ, -2 + f y a ≤ f x a ∧ f x a ≤ 2 + f y a) ↔ (-3 ≤ a ∧ a ≤ -1) :=
sorry

end problem1_problem2_l41_41425


namespace max_value_ineq_l41_41504

variables {R : Type} [LinearOrderedField R]

theorem max_value_ineq (a b c x y z : R) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : 0 ≤ x) (h5 : 0 ≤ y) (h6 : 0 ≤ z)
  (h7 : a + b + c = 1) (h8 : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1 / 16 :=
sorry

end max_value_ineq_l41_41504


namespace sum_tens_ones_digits_3_plus_4_power_17_l41_41872

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end sum_tens_ones_digits_3_plus_4_power_17_l41_41872


namespace sum_of_28_terms_l41_41624

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_of_28_terms
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h2 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∃ d ≠ 0, ∃ a₁, ∀ n, a (n + 1) = a₁ + n * d)
  (h4 : f (a 6) = f (a 23)) :
  sum_arithmetic_sequence 28 (a 1) ((a 2) - (a 1)) = 28 :=
by sorry

end sum_of_28_terms_l41_41624


namespace roots_ratio_sum_l41_41516

theorem roots_ratio_sum (a b m : ℝ) 
  (m1 m2 : ℝ)
  (h_roots : a ≠ b ∧ b ≠ 0 ∧ m ≠ 0 ∧ a ≠ 0 ∧ 
    ∀ x : ℝ, m * (x^2 - 3 * x) + 2 * x + 7 = 0 → (x = a ∨ x = b)) 
  (h_ratio : (a / b) + (b / a) = 7 / 3)
  (h_m1_m2_eq : ((3 * m - 2) ^ 2) / (7 * m) - 2 = 7 / 3)
  (h_m_vieta : (3 * m - 2) ^ 2 - 27 * m * (91 / 3) = 0) :
  (m1 + m2 = 127 / 27) ∧ (m1 * m2 = 4 / 9) →
  ((m1 / m2) + (m2 / m1) = 47.78) :=
sorry

end roots_ratio_sum_l41_41516


namespace expression_value_l41_41286

def α : ℝ := 60
def β : ℝ := 20
def AB : ℝ := 1

noncomputable def γ : ℝ := 180 - (α + β)

noncomputable def AC : ℝ := AB * (Real.sin γ / Real.sin β)
noncomputable def BC : ℝ := (Real.sin α / Real.sin γ) * AB

theorem expression_value : (1 / AC - BC) = 2 := by
  sorry

end expression_value_l41_41286


namespace Marcus_fit_pies_l41_41168

theorem Marcus_fit_pies (x : ℕ) 
(h1 : ∀ b, (7 * b - 8) = 27) : x = 5 := by
  sorry

end Marcus_fit_pies_l41_41168


namespace first_number_is_nine_l41_41373

theorem first_number_is_nine (x : ℤ) (h : 11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) : x = 9 :=
by {
  sorry
}

end first_number_is_nine_l41_41373


namespace product_of_coordinates_of_D_l41_41340

theorem product_of_coordinates_of_D (D : ℝ × ℝ) (N : ℝ × ℝ) (C : ℝ × ℝ) 
  (hN : N = (4, 3)) (hC : C = (5, -1)) (midpoint : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 * D.2 = 21 :=
by
  sorry

end product_of_coordinates_of_D_l41_41340


namespace points_three_units_away_from_neg3_l41_41768

theorem points_three_units_away_from_neg3 (x : ℝ) : (abs (x + 3) = 3) ↔ (x = 0 ∨ x = -6) :=
by
  sorry

end points_three_units_away_from_neg3_l41_41768


namespace Vanya_correct_answers_l41_41188

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l41_41188


namespace determine_p_and_q_l41_41701

theorem determine_p_and_q (x p q : ℝ) : 
  (x + 4) * (x - 1) = x^2 + p * x + q → (p = 3 ∧ q = -4) := 
by 
  sorry

end determine_p_and_q_l41_41701


namespace equality_of_costs_l41_41756

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end equality_of_costs_l41_41756


namespace jill_arrives_before_jack_l41_41607

theorem jill_arrives_before_jack {distance speed_jill speed_jack : ℝ} (h1 : distance = 1) 
  (h2 : speed_jill = 10) (h3 : speed_jack = 4) :
  (60 * (distance / speed_jack) - 60 * (distance / speed_jill)) = 9 :=
by
  sorry

end jill_arrives_before_jack_l41_41607


namespace vinny_fifth_month_loss_l41_41692

theorem vinny_fifth_month_loss (start_weight : ℝ) (end_weight : ℝ) (first_month_loss : ℝ) (second_month_loss : ℝ) (third_month_loss : ℝ) (fourth_month_loss : ℝ) (total_loss : ℝ):
  start_weight = 300 ∧
  first_month_loss = 20 ∧
  second_month_loss = first_month_loss / 2 ∧
  third_month_loss = second_month_loss / 2 ∧
  fourth_month_loss = third_month_loss / 2 ∧
  (start_weight - end_weight) = total_loss ∧
  end_weight = 250.5 →
  (total_loss - (first_month_loss + second_month_loss + third_month_loss + fourth_month_loss)) = 12 :=
by
  sorry

end vinny_fifth_month_loss_l41_41692


namespace rectangle_perimeter_l41_41272

theorem rectangle_perimeter : 
  ∃ (x y a b : ℝ), 
  (x * y = 2016) ∧ 
  (a * b = 2016) ∧ 
  (x^2 + y^2 = 4 * (a^2 - b^2)) → 
  2 * (x + y) = 8 * Real.sqrt 1008 :=
sorry

end rectangle_perimeter_l41_41272


namespace probability_of_picking_dumpling_with_egg_l41_41402

-- Definitions based on the conditions
def total_dumplings : ℕ := 10
def dumplings_with_eggs : ℕ := 3

-- The proof statement
theorem probability_of_picking_dumpling_with_egg :
  (dumplings_with_eggs : ℚ) / total_dumplings = 3 / 10 :=
by
  sorry

end probability_of_picking_dumpling_with_egg_l41_41402


namespace complex_square_l41_41456

theorem complex_square (a b : ℤ) (i : ℂ) (h1: a = 5) (h2: b = 3) (h3: i^2 = -1) :
  ((↑a) + (↑b) * i)^2 = 16 + 30 * i := by
  sorry

end complex_square_l41_41456


namespace height_table_l41_41634

variable (l w h : ℝ)

theorem height_table (h_eq1 : l + h - w = 32) (h_eq2 : w + h - l = 28) : h = 30 := by
  sorry

end height_table_l41_41634


namespace find_b_l41_41181

-- Define functions p and q
def p (x : ℝ) : ℝ := 3 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

-- Set the target value for p(q(3))
def target_val : ℝ := 9

-- Prove that b = 22/3
theorem find_b (b : ℝ) : p (q 3 b) = target_val → b = 22 / 3 := by
  intro h
  sorry

end find_b_l41_41181


namespace square_of_complex_l41_41494

def z : Complex := 5 - 2 * Complex.I

theorem square_of_complex : z^2 = 21 - 20 * Complex.I := by
  sorry

end square_of_complex_l41_41494


namespace tail_length_l41_41206

theorem tail_length {length body tail : ℝ} (h1 : length = 30) (h2 : tail = body / 2) (h3 : length = body) : tail = 15 := by
  sorry

end tail_length_l41_41206


namespace dice_sum_probability_15_l41_41300
open Nat

theorem dice_sum_probability_15 (n : ℕ) (h : n = 3432) : 
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 : ℕ,
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
  (1 ≤ d3 ∧ d3 ≤ 6) ∧ (1 ≤ d4 ∧ d4 ≤ 6) ∧ 
  (1 ≤ d5 ∧ d5 ≤ 6) ∧ (1 ≤ d6 ∧ d6 ≤ 6) ∧ 
  (1 ≤ d7 ∧ d7 ≤ 6) ∧ (1 ≤ d8 ∧ d8 ≤ 6) ∧ 
  (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 = 15) :=
by
  sorry

end dice_sum_probability_15_l41_41300


namespace minibuses_not_enough_l41_41477

def num_students : ℕ := 300
def minibus_capacity : ℕ := 23
def num_minibuses : ℕ := 13

theorem minibuses_not_enough :
  num_minibuses * minibus_capacity < num_students :=
by
  sorry

end minibuses_not_enough_l41_41477


namespace solve_for_D_d_Q_R_l41_41010

theorem solve_for_D_d_Q_R (D d Q R : ℕ) 
    (h1 : D = d * Q + R) 
    (h2 : d * Q = 135) 
    (h3 : R = 2 * d) : 
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by
  sorry

end solve_for_D_d_Q_R_l41_41010


namespace corridor_length_correct_l41_41766

/-- Scale representation in the blueprint: 1 cm represents 10 meters. --/
def scale_cm_to_m (cm: ℝ): ℝ := cm * 10

/-- Length of the corridor in the blueprint. --/
def blueprint_length_cm: ℝ := 9.5

/-- Real-life length of the corridor. --/
def real_life_length: ℝ := 95

/-- Proof that the real-life length of the corridor is correctly calculated. --/
theorem corridor_length_correct :
  scale_cm_to_m blueprint_length_cm = real_life_length :=
by
  sorry

end corridor_length_correct_l41_41766


namespace joe_initial_tests_l41_41605

theorem joe_initial_tests (S n : ℕ) (h1 : S = 60 * n) (h2 : (S - 45) = 65 * (n - 1)) : n = 4 :=
by {
  sorry
}

end joe_initial_tests_l41_41605


namespace next_term_in_geometric_sequence_l41_41246

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end next_term_in_geometric_sequence_l41_41246


namespace first_negative_term_at_14_l41_41629

-- Define the n-th term of the arithmetic sequence
def a_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Given values
def a₁ := 51
def d := -4

-- Proof statement
theorem first_negative_term_at_14 : ∃ n : ℕ, a_n a₁ d n < 0 ∧ ∀ m < n, a_n a₁ d m ≥ 0 :=
  by sorry

end first_negative_term_at_14_l41_41629


namespace matt_and_peter_worked_together_days_l41_41743

variables (W : ℝ) -- Represents total work
noncomputable def work_rate_peter := W / 35
noncomputable def work_rate_together := W / 20

theorem matt_and_peter_worked_together_days (x : ℝ) :
  (x / 20) + (14 / 35) = 1 → x = 12 :=
by {
  sorry
}

end matt_and_peter_worked_together_days_l41_41743


namespace geometric_prod_eight_l41_41653

theorem geometric_prod_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : ∀ n, a n ≠ 0)
  (h_eq : a 4 + 3 * a 8 = 2 * (a 7)^2)
  (h_geom : ∀ {m n : ℕ}, b m * b (m + n) = b (2 * m + n))
  (h_b_eq_a : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 :=
sorry

end geometric_prod_eight_l41_41653


namespace prove_expression_l41_41646

theorem prove_expression (a b : ℕ) 
  (h1 : 180 % 2^a = 0 ∧ 180 % 2^(a+1) ≠ 0)
  (h2 : 180 % 3^b = 0 ∧ 180 % 3^(b+1) ≠ 0) :
  (1 / 4 : ℚ)^(b - a) = 1 := 
sorry

end prove_expression_l41_41646


namespace correct_regression_eq_l41_41934

-- Definitions related to the conditions
def negative_correlation (y x : ℝ) : Prop :=
  -- y is negatively correlated with x implies a negative slope in regression
  ∃ a b : ℝ, a < 0 ∧ ∀ x, y = a * x + b

-- The potential regression equations
def regression_eq1 (x : ℝ) : ℝ := -10 * x + 200
def regression_eq2 (x : ℝ) : ℝ := 10 * x + 200
def regression_eq3 (x : ℝ) : ℝ := -10 * x - 200
def regression_eq4 (x : ℝ) : ℝ := 10 * x - 200

-- Prove that the correct regression equation is selected given the conditions
theorem correct_regression_eq (y x : ℝ) (h : negative_correlation y x) : 
  (∀ x : ℝ, y = regression_eq1 x) ∨ (∀ x : ℝ, y = regression_eq2 x) ∨ 
  (∀ x : ℝ, y = regression_eq3 x) ∨ (∀ x : ℝ, y = regression_eq4 x) →
  ∀ x : ℝ, y = regression_eq1 x := by
  -- This theorem states that given negative correlation and the possible options, 
  -- the correct regression equation consistent with all conditions must be regression_eq1.
  sorry

end correct_regression_eq_l41_41934


namespace vehicle_capacity_rental_plans_l41_41709

variables (a b x y : ℕ)

/-- Conditions -/
axiom cond1 : 2*x + y = 11
axiom cond2 : x + 2*y = 13

/-- Resulting capacities for each vehicle type -/
theorem vehicle_capacity : 
  x = 3 ∧ y = 5 :=
by
  sorry

/-- Rental plans for transporting 33 tons of drugs -/
theorem rental_plans :
  3*a + 5*b = 33 ∧ ((a = 6 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
by
  sorry

end vehicle_capacity_rental_plans_l41_41709


namespace digit_difference_l41_41885

theorem digit_difference (X Y : ℕ) (h1 : 0 ≤ X ∧ X ≤ 9) (h2 : 0 ≤ Y ∧ Y ≤ 9) (h3 : (10 * X + Y) - (10 * Y + X) = 54) : X - Y = 6 :=
sorry

end digit_difference_l41_41885


namespace min_value_problem1_l41_41932

theorem min_value_problem1 (x : ℝ) (hx : x > -1) : 
  ∃ m, m = 2 * Real.sqrt 2 + 1 ∧ (∀ y, y = (x^2 + 3 * x + 4) / (x + 1) ∧ x > -1 → y ≥ m) :=
sorry

end min_value_problem1_l41_41932


namespace jim_age_l41_41466

variable (J F S : ℕ)

theorem jim_age (h1 : J = 2 * F) (h2 : F = S + 9) (h3 : J - 6 = 5 * (S - 6)) : J = 46 := 
by
  sorry

end jim_age_l41_41466


namespace abs_less_than_2_sufficient_but_not_necessary_l41_41715

theorem abs_less_than_2_sufficient_but_not_necessary (x : ℝ) : 
  (|x| < 2 → (x^2 - x - 6 < 0)) ∧ ¬(x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_less_than_2_sufficient_but_not_necessary_l41_41715


namespace brick_width_l41_41058

/-- Let dimensions of the wall be 700 cm (length), 600 cm (height), and 22.5 cm (thickness).
    Let dimensions of each brick be 25 cm (length), W cm (width), and 6 cm (height).
    Given that 5600 bricks are required to build the wall, prove that the width of each brick is 11.25 cm. -/
theorem brick_width (W : ℝ)
  (h_wall_dimensions : 700 = 700) (h_wall_height : 600 = 600) (h_wall_thickness : 22.5 = 22.5)
  (h_brick_length : 25 = 25) (h_brick_height : 6 = 6) (h_num_bricks : 5600 = 5600)
  (h_wall_volume : 700 * 600 * 22.5 = 9450000)
  (h_brick_volume : 25 * W * 6 = 9450000 / 5600) :
  W = 11.25 :=
sorry

end brick_width_l41_41058


namespace sin_neg_p_l41_41176

theorem sin_neg_p (a : ℝ) : (¬ ∃ x : ℝ, Real.sin x > a) → (a ≥ 1) := 
by
  sorry

end sin_neg_p_l41_41176


namespace solve_system_of_equations_l41_41851

def sys_eq1 (x y : ℝ) : Prop := 6 * (1 - x) ^ 2 = 1 / y
def sys_eq2 (x y : ℝ) : Prop := 6 * (1 - y) ^ 2 = 1 / x

theorem solve_system_of_equations (x y : ℝ) :
  sys_eq1 x y ∧ sys_eq2 x y ↔
  ((x = 3 / 2 ∧ y = 2 / 3) ∨
   (x = 2 / 3 ∧ y = 3 / 2) ∨
   (x = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)) ∧ y = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)))) :=
sorry

end solve_system_of_equations_l41_41851


namespace minimum_value_l41_41327

theorem minimum_value (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (2 / (x + 3 * y) + 1 / (x - y)) = (3 + 2 * Real.sqrt 2) / 2 := sorry

end minimum_value_l41_41327


namespace correct_calculation_l41_41115

theorem correct_calculation :
  (-2 * a * b^2)^3 = -8 * a^3 * b^6 :=
by sorry

end correct_calculation_l41_41115


namespace product_of_even_and_odd_is_odd_l41_41531

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def odd_product (f g : ℝ → ℝ) : Prop := ∀ x, (f x) * (g x) = - (f x) * (g x)
 
theorem product_of_even_and_odd_is_odd 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : odd_function g) : odd_product f g :=
by
  sorry

end product_of_even_and_odd_is_odd_l41_41531


namespace good_goods_not_cheap_is_sufficient_condition_l41_41156

theorem good_goods_not_cheap_is_sufficient_condition
  (goods_good : Prop)
  (goods_not_cheap : Prop)
  (h : goods_good → goods_not_cheap) :
  (goods_good → goods_not_cheap) :=
by
  exact h

end good_goods_not_cheap_is_sufficient_condition_l41_41156


namespace age_difference_l41_41727

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end age_difference_l41_41727


namespace even_num_Z_tetrominoes_l41_41254

-- Definitions based on the conditions of the problem
def is_tiled_with_S_tetrominoes (P : Type) : Prop := sorry
def tiling_uses_S_Z_tetrominoes (P : Type) : Prop := sorry
def num_Z_tetrominoes (P : Type) : ℕ := sorry

-- The theorem statement
theorem even_num_Z_tetrominoes (P : Type) 
  (hTiledWithS : is_tiled_with_S_tetrominoes P) 
  (hTilingWithSZ : tiling_uses_S_Z_tetrominoes P) : num_Z_tetrominoes P % 2 = 0 :=
sorry

end even_num_Z_tetrominoes_l41_41254


namespace D_double_prime_coordinates_l41_41055

-- The coordinates of points A, B, C, D as given in the problem
def A : (ℝ × ℝ) := (3, 6)
def B : (ℝ × ℝ) := (5, 10)
def C : (ℝ × ℝ) := (7, 6)
def D : (ℝ × ℝ) := (5, 2)

-- Reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def D' : ℝ × ℝ := reflect_x D

-- Translate the point (x, y) by (dx, dy)
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

-- Reflect across the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combined translation and reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := translate p 0 (-2)
  let p_reflected := reflect_y_eq_x p_translated
  translate p_reflected 0 2

def D'' : ℝ × ℝ := reflect_y_eq_x_plus_2 D'

theorem D_double_prime_coordinates : D'' = (-4, 7) := by
  sorry

end D_double_prime_coordinates_l41_41055


namespace probability_below_8_l41_41671

def prob_hit_10 := 0.20
def prob_hit_9 := 0.30
def prob_hit_8 := 0.10

theorem probability_below_8 : (1 - (prob_hit_10 + prob_hit_9 + prob_hit_8) = 0.40) :=
by
  sorry

end probability_below_8_l41_41671


namespace relationship_between_sets_l41_41729

def M (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k - 2
def P (x : ℤ) : Prop := ∃ n : ℤ, x = 5 * n + 3
def S (x : ℤ) : Prop := ∃ m : ℤ, x = 10 * m + 3

theorem relationship_between_sets :
  (∀ x, S x → P x) ∧ (∀ x, P x → M x) ∧ (∀ x, M x → P x) :=
by
  sorry

end relationship_between_sets_l41_41729


namespace total_number_of_trees_l41_41924

theorem total_number_of_trees (D P : ℕ) (cost_D cost_P total_cost : ℕ)
  (hD : D = 350)
  (h_cost_D : cost_D = 300)
  (h_cost_P : cost_P = 225)
  (h_total_cost : total_cost = 217500)
  (h_cost_equation : cost_D * D + cost_P * P = total_cost) :
  D + P = 850 :=
by
  rw [hD, h_cost_D, h_cost_P, h_total_cost] at h_cost_equation
  sorry

end total_number_of_trees_l41_41924


namespace fraction_of_total_money_l41_41449

variable (Max Leevi Nolan Ollie : ℚ)

-- Condition: Each of Max, Leevi, and Nolan gave Ollie the same amount of money
variable (x : ℚ) (h1 : Max / 6 = x) (h2 : Leevi / 3 = x) (h3 : Nolan / 2 = x)

-- Proving that the fraction of the group's (Max, Leevi, Nolan, Ollie) total money possessed by Ollie is 3/11.
theorem fraction_of_total_money (h4 : Max + Leevi + Nolan + Ollie = Max + Leevi + Nolan + 3 * x) : 
  x / (Max + Leevi + Nolan + x) = 3 / 11 := 
by
  sorry

end fraction_of_total_money_l41_41449


namespace express_in_scientific_notation_l41_41422

theorem express_in_scientific_notation (n : ℝ) (h : n = 456.87 * 10^6) : n = 4.5687 * 10^8 :=
by 
  -- sorry to skip the proof
  sorry

end express_in_scientific_notation_l41_41422


namespace vecMA_dotProduct_vecBA_range_l41_41075

-- Define the conditions
def pointM : ℝ × ℝ := (1, 0)

def onEllipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

def vecMA (A : ℝ × ℝ) := (A.1 - pointM.1, A.2 - pointM.2)
def vecMB (B : ℝ × ℝ) := (B.1 - pointM.1, B.2 - pointM.2)
def vecBA (A B : ℝ × ℝ) := (A.1 - B.1, A.2 - B.2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the statement
theorem vecMA_dotProduct_vecBA_range (A B : ℝ × ℝ) (α : ℝ) :
  onEllipse A → onEllipse B → dotProduct (vecMA A) (vecMB B) = 0 → 
  A = (2 * Real.cos α, Real.sin α) → 
  (2/3 ≤ dotProduct (vecMA A) (vecBA A B) ∧ dotProduct (vecMA A) (vecBA A B) ≤ 9) :=
sorry

end vecMA_dotProduct_vecBA_range_l41_41075


namespace correct_product_l41_41221

-- Definitions for conditions
def reversed_product (a b : ℕ) : Prop :=
  let reversed_a := (a % 10) * 10 + (a / 10)
  reversed_a * b = 204

theorem correct_product (a b : ℕ) (h : reversed_product a b) : a * b = 357 := 
by
  sorry

end correct_product_l41_41221


namespace range_of_a_l41_41886

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ (Set.Iio (-3) ∪ Set.Ioi 1) :=
by sorry

end range_of_a_l41_41886


namespace original_strength_of_class_l41_41236

-- Definitions from the problem conditions
def average_age_original (x : ℕ) : ℕ := 40 * x
def total_students (x : ℕ) : ℕ := x + 17
def total_age_new_students : ℕ := 17 * 32
def new_average_age : ℕ := 36

-- Lean statement to prove that the original strength of the class is 17.
theorem original_strength_of_class :
  ∃ x : ℕ, average_age_original x + total_age_new_students = total_students x * new_average_age ∧ x = 17 :=
by
  sorry

end original_strength_of_class_l41_41236


namespace sum_of_three_numbers_l41_41993

theorem sum_of_three_numbers (x : ℝ) (a b c : ℝ) (h1 : a = 5 * x) (h2 : b = x) (h3 : c = 4 * x) (h4 : c = 400) :
  a + b + c = 1000 := by
  sorry

end sum_of_three_numbers_l41_41993


namespace probability_in_smaller_spheres_l41_41076

theorem probability_in_smaller_spheres 
    (R r : ℝ)
    (h_eq : ∀ (R r : ℝ), R + r = 4 * r)
    (vol_eq : ∀ (R r : ℝ), (4/3) * π * r^3 * 5 = (4/3) * π * R^3 * (5/27)) :
    P = 0.2 := by
  sorry

end probability_in_smaller_spheres_l41_41076


namespace average_cookies_per_package_l41_41622

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies_per_package :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 125 / 9 :=
by
  sorry

end average_cookies_per_package_l41_41622


namespace visitors_correct_l41_41234

def visitors_that_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def total_visitors_before_that_day : ℕ := 522
def visitors_two_days_before : ℕ := total_visitors_before_that_day - visitors_previous_day - visitors_that_day

theorem visitors_correct : visitors_two_days_before = 11 := by
  -- Sorry, proof to be filled in
  sorry

end visitors_correct_l41_41234


namespace other_number_is_29_l41_41258

theorem other_number_is_29
    (k : ℕ)
    (some_number : ℕ)
    (h1 : k = 2)
    (h2 : (5 + k) * (5 - k) = some_number - 2^3) :
    some_number = 29 :=
by
  sorry

end other_number_is_29_l41_41258


namespace coordinates_of_point_P_l41_41059

theorem coordinates_of_point_P {x y : ℝ} (hx : |x| = 2) (hy : y = 1 ∨ y = -1) (hxy : x < 0 ∧ y > 0) : 
  (x, y) = (-2, 1) := 
by 
  sorry

end coordinates_of_point_P_l41_41059


namespace find_point_of_intersection_l41_41001
noncomputable def point_of_intersection_curve_line : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^y = y^x ∧ y = x ∧ x = Real.exp 1 ∧ y = Real.exp 1

theorem find_point_of_intersection : point_of_intersection_curve_line :=
sorry

end find_point_of_intersection_l41_41001


namespace johns_weekly_allowance_l41_41404

theorem johns_weekly_allowance (A : ℝ) (h1: A - (3/5) * A = (2/5) * A)
  (h2: (2/5) * A - (1/3) * (2/5) * A = (4/15) * A)
  (h3: (4/15) * A = 0.92) : A = 3.45 :=
by {
  sorry
}

end johns_weekly_allowance_l41_41404


namespace lindsay_dolls_problem_l41_41476

theorem lindsay_dolls_problem :
  let blonde_dolls := 6
  let brown_dolls := 3 * blonde_dolls
  let black_dolls := brown_dolls / 2
  let red_dolls := 2 * black_dolls
  let combined_dolls := black_dolls + brown_dolls + red_dolls
  combined_dolls - blonde_dolls = 39 :=
by
  sorry

end lindsay_dolls_problem_l41_41476


namespace convert_to_base8_l41_41853

theorem convert_to_base8 (n : ℕ) (h : n = 1024) : 
  (∃ (d3 d2 d1 d0 : ℕ), n = d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0 ∧ d3 = 2 ∧ d2 = 0 ∧ d1 = 0 ∧ d0 = 0) :=
by
  sorry

end convert_to_base8_l41_41853


namespace intersection_point_x_coordinate_l41_41973

noncomputable def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) = 1

noncomputable def c := 1 + Real.sqrt 3

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_point_x_coordinate
  (x y b : ℝ)
  (h_hyperbola : hyperbola x y b)
  (h_distance_foci : distance (2 * c, 0) (0, 0) = 2 * c)
  (h_circle_center : distance (x, y) (0, 0) = c)
  (h_p_distance : distance (x, y) (2 * c, 0) = c + 2) :
  x = (Real.sqrt 3 + 1) / 2 :=
sorry

end intersection_point_x_coordinate_l41_41973


namespace k_at_1_value_l41_41483

def h (x p : ℝ) := x^3 + p * x^2 + 2 * x + 20
def k (x p q r : ℝ) := x^4 + 2 * x^3 + q * x^2 + 50 * x + r

theorem k_at_1_value (p q r : ℝ) (h_distinct_roots : ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ → h x₁ p = 0 → h x₂ p = 0 → h x₃ p = 0 → k x₁ p q r = 0 ∧ k x₂ p q r = 0 ∧ k x₃ p q r = 0):
  k 1 (-28) (2 - -28 * -30) (-20 * -30) = -155 :=
by
  sorry

end k_at_1_value_l41_41483


namespace ratio_of_socks_l41_41481

theorem ratio_of_socks (y p : ℝ) (h1 : 5 * p + y * 2 * p = 5 * p + 4 * y * p / 3) :
  (5 : ℝ) / y = 11 / 2 :=
by
  sorry

end ratio_of_socks_l41_41481


namespace max_profit_at_nine_l41_41902

noncomputable def profit (x : ℝ) : ℝ := - (1 / 3) * x^3 + 81 * x - 23

theorem max_profit_at_nine :
  ∃ x : ℝ, x = 9 ∧ ∀ (ε : ℝ), ε > 0 → 
  (profit (9 - ε) < profit 9 ∧ profit (9 + ε) < profit 9) := 
by
  sorry

end max_profit_at_nine_l41_41902


namespace max_t_eq_one_l41_41560

theorem max_t_eq_one {x y : ℝ} (hx : x > 0) (hy : y > 0) : 
  max (min x (y / (x^2 + y^2))) 1 = 1 :=
sorry

end max_t_eq_one_l41_41560


namespace negation_of_existential_l41_41652

theorem negation_of_existential:
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 = 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existential_l41_41652


namespace sameTypeTerm_l41_41786

variable (a b : ℝ) -- Assume a and b are real numbers 

-- Definitions for each term in the conditions
def term1 : ℝ := 2 * a * b^2
def term2 : ℝ := -a^2 * b
def term3 : ℝ := -2 * a * b
def term4 : ℝ := 5 * a^2

-- The term we are comparing against
def compareTerm : ℝ := 3 * a^2 * b

-- The condition we want to prove
theorem sameTypeTerm : term2 = compareTerm :=
  sorry


end sameTypeTerm_l41_41786


namespace cost_formula_l41_41845

-- Given Conditions
def flat_fee := 5  -- flat service fee in cents
def first_kg_cost := 12  -- cost for the first kilogram in cents
def additional_kg_cost := 5  -- cost for each additional kilogram in cents

-- Integer weight in kilograms
variable (P : ℕ)

-- Total cost calculation proof problem
theorem cost_formula : ∃ C, C = flat_fee + first_kg_cost + additional_kg_cost * (P - 1) → C = 5 * P + 12 :=
by
  sorry

end cost_formula_l41_41845


namespace final_pressure_of_helium_l41_41054

theorem final_pressure_of_helium
  (p v v' : ℝ) (k : ℝ)
  (h1 : p = 4)
  (h2 : v = 3)
  (h3 : v' = 6)
  (h4 : p * v = k)
  (h5 : ∀ p' : ℝ, p' * v' = k → p' = 2) :
  p' = 2 := by
  sorry

end final_pressure_of_helium_l41_41054


namespace max_sum_x_y_l41_41297

theorem max_sum_x_y (x y : ℝ) (h : (2015 + x^2) * (2015 + y^2) = 2 ^ 22) : 
  x + y ≤ 2 * Real.sqrt 33 :=
sorry

end max_sum_x_y_l41_41297


namespace modified_pyramid_volume_l41_41314

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) 
  (hV : V = 1/3 * s^2 * h) (hV_eq : V = 72) :
  (1/3) * (3 * s)^2 * (2 * h) = 1296 := by
  sorry

end modified_pyramid_volume_l41_41314


namespace fraction_of_sides_area_of_triangle_l41_41174

-- Part (1)
theorem fraction_of_sides (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) : (a + b) / c = 2 :=
sorry

-- Part (2)
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) (h_C : C = π / 3) : (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
sorry

end fraction_of_sides_area_of_triangle_l41_41174


namespace least_x_y_z_value_l41_41081

theorem least_x_y_z_value :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (3 * x = 4 * y) ∧ (4 * y = 7 * z) ∧ (3 * x = 7 * z) ∧ (x - y + z = 19) :=
by
  sorry

end least_x_y_z_value_l41_41081


namespace sufficient_condition_perpendicular_l41_41774

variables {Plane Line : Type}
variables (l : Line) (α β : Plane)

-- Definitions for perpendicularity and parallelism
def perp (l : Line) (α : Plane) : Prop := sorry
def parallel (α β : Plane) : Prop := sorry

theorem sufficient_condition_perpendicular
  (h1 : perp l α) 
  (h2 : parallel α β) : 
  perp l β :=
sorry

end sufficient_condition_perpendicular_l41_41774


namespace no_distinct_roots_exist_l41_41469

theorem no_distinct_roots_exist :
  ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a^2 - 2 * b * a + c^2 = 0) ∧
  (b^2 - 2 * c * b + a^2 = 0) ∧ 
  (c^2 - 2 * a * c + b^2 = 0) := 
sorry

end no_distinct_roots_exist_l41_41469


namespace joey_more_fish_than_peter_l41_41301

-- Define the conditions
variables (A P J : ℕ)

-- Condition that Ali's fish weight is twice that of Peter's
def ali_double_peter (A P : ℕ) : Prop := A = 2 * P

-- Condition that Ali caught 12 kg of fish
def ali_caught_12 (A : ℕ) : Prop := A = 12

-- Condition that the total weight of the fish is 25 kg
def total_weight (A P J : ℕ) : Prop := A + P + J = 25

-- Prove that Joey caught 1 kg more fish than Peter
theorem joey_more_fish_than_peter (A P J : ℕ) :
  ali_double_peter A P → ali_caught_12 A → total_weight A P J → J = 1 :=
by 
  intro h1 h2 h3
  sorry

end joey_more_fish_than_peter_l41_41301


namespace trapezoid_EC_length_l41_41730

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end trapezoid_EC_length_l41_41730


namespace actual_distance_traveled_l41_41334

theorem actual_distance_traveled (T : ℝ) :
  ∀ D : ℝ, (D = 4 * T) → (D + 6 = 5 * T) → D = 24 :=
by
  intro D h1 h2
  sorry

end actual_distance_traveled_l41_41334


namespace pow_mod_remainder_l41_41929

theorem pow_mod_remainder (n : ℕ) (h : 9 ≡ 2 [MOD 7]) (h2 : 9^2 ≡ 4 [MOD 7]) (h3 : 9^3 ≡ 1 [MOD 7]) : 9^123 % 7 = 1 := by
  sorry

end pow_mod_remainder_l41_41929


namespace roots_equation_l41_41602

theorem roots_equation (m n : ℝ) (h1 : ∀ x, (x - m) * (x - n) = x^2 + 2 * x - 2025) : m^2 + 3 * m + n = 2023 :=
by
  sorry

end roots_equation_l41_41602


namespace apples_distribution_l41_41407

theorem apples_distribution (total_apples : ℕ) (rotten_apples : ℕ) (boxes : ℕ) (remaining_apples : ℕ) (apples_per_box : ℕ) :
  total_apples = 40 →
  rotten_apples = 4 →
  boxes = 4 →
  remaining_apples = total_apples - rotten_apples →
  apples_per_box = remaining_apples / boxes →
  apples_per_box = 9 :=
by
  intros
  sorry

end apples_distribution_l41_41407


namespace endpoint_correctness_l41_41696

-- Define two points in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define start point (2, 2)
def startPoint : Point := ⟨2, 2⟩

-- Define the endpoint's conditions
def endPoint (x y : ℝ) : Prop :=
  y = 2 * x + 1 ∧ (x > 0) ∧ (Real.sqrt ((x - startPoint.x) ^ 2 + (y - startPoint.y) ^ 2) = 6)

-- The solution to the problem proving (3.4213, 7.8426) satisfies the conditions
theorem endpoint_correctness : ∃ (x y : ℝ), endPoint x y ∧ x = 3.4213 ∧ y = 7.8426 := by
  use 3.4213
  use 7.8426
  sorry

end endpoint_correctness_l41_41696


namespace range_of_a_l41_41307

section
  variable {x a : ℝ}

  -- Define set A
  def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }

  -- Define set B
  def setB (a : ℝ) : Set ℝ := 
    { x | (2*x + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

  -- The proof problem statement
  theorem range_of_a (a : ℝ) : 
    (setA ⊆ setB a) ↔ (-4 ≤ a ∧ a ≤ -2) :=
  sorry
end

end range_of_a_l41_41307


namespace g_analytical_expression_g_minimum_value_l41_41994

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def M (a : ℝ) : ℝ := if (a ≥ 1/3 ∧ a ≤ 1/2) then f a 1 else f a 3
noncomputable def N (a : ℝ) : ℝ := f a (1/a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/3 ∧ a ≤ 1/2 then M a - N a 
  else if a > 1/2 ∧ a ≤ 1 then M a - N a
  else 0 -- outside the given interval, by definition may be kept as 0

theorem g_analytical_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) : 
  g a = if (1/3 ≤ a ∧ a ≤ 1/2) then a + 1/a - 2 else 9 * a + 1/a - 6 := 
sorry

theorem g_minimum_value (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ∃ (a' : ℝ), 1/3 ≤ a' ∧ a' ≤ 1 ∧ (∀ a, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ g a') ∧ g a' = 1/2 := 
sorry

end g_analytical_expression_g_minimum_value_l41_41994


namespace math_problem_mod_1001_l41_41964

theorem math_problem_mod_1001 :
  (2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12) % 1001 = 400 := by
  sorry

end math_problem_mod_1001_l41_41964


namespace triangle_has_three_altitudes_l41_41737

-- Assuming a triangle in ℝ² space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Definition of an altitude in the context of Lean
def altitude (T : Triangle) (p : ℝ × ℝ) := 
  ∃ (a : ℝ) (b : ℝ), T.A.1 * p.1 + T.A.2 * p.2 = a * p.1 + b -- Placeholder, real definition of altitude may vary

-- Prove that a triangle has exactly 3 altitudes
theorem triangle_has_three_altitudes (T : Triangle) : ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
  altitude T p₁ ∧ altitude T p₂ ∧ altitude T p₃ :=
sorry

end triangle_has_three_altitudes_l41_41737


namespace tiffany_total_bags_l41_41803

def initial_bags : ℕ := 10
def found_on_tuesday : ℕ := 3
def found_on_wednesday : ℕ := 7
def total_bags : ℕ := 20

theorem tiffany_total_bags (initial_bags : ℕ) (found_on_tuesday : ℕ) (found_on_wednesday : ℕ) (total_bags : ℕ) :
    initial_bags + found_on_tuesday + found_on_wednesday = total_bags :=
by
  sorry

end tiffany_total_bags_l41_41803


namespace proof_of_a_neg_two_l41_41751

theorem proof_of_a_neg_two (a : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_real : (1 + i)^2 - a / i = (a + 2) * i → ∃ r : ℝ, (1 + i)^2 - a / i = r) : a = -2 :=
sorry

end proof_of_a_neg_two_l41_41751


namespace blocks_added_l41_41169

theorem blocks_added (a b : Nat) (h₁ : a = 86) (h₂ : b = 95) : b - a = 9 :=
by
  sorry

end blocks_added_l41_41169


namespace find_constants_PQR_l41_41484

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end find_constants_PQR_l41_41484


namespace spadesuit_calculation_l41_41713

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 2 (spadesuit 6 1) = -1221 := by
  sorry

end spadesuit_calculation_l41_41713


namespace percentage_B_to_C_l41_41762

variables (total_students : ℕ)
variables (pct_A pct_B pct_C pct_A_to_C pct_B_to_C : ℝ)

-- Given conditions
axiom total_students_eq_100 : total_students = 100
axiom pct_A_eq_60 : pct_A = 60
axiom pct_B_eq_40 : pct_B = 40
axiom pct_A_to_C_eq_30 : pct_A_to_C = 30
axiom pct_C_eq_34 : pct_C = 34

-- Proof goal
theorem percentage_B_to_C :
  pct_B_to_C = 40 :=
sorry

end percentage_B_to_C_l41_41762


namespace solution_set_of_quadratic_inequality_l41_41881

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 4 * x - 5 > 0} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l41_41881


namespace ramu_repairs_cost_l41_41796

theorem ramu_repairs_cost :
  ∃ R : ℝ, 64900 - (42000 + R) = (29.8 / 100) * (42000 + R) :=
by
  use 8006.16
  sorry

end ramu_repairs_cost_l41_41796


namespace find_m_l41_41512

variables (m : ℝ)
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

-- Define the property of vector parallelism in ℝ.
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Statement to be proven
theorem find_m :
    parallel (1, m - 1) c →
    m = -1 :=
by
  sorry

end find_m_l41_41512


namespace probability_intersection_inside_nonagon_correct_l41_41329

def nonagon_vertices : ℕ := 9

def total_pairs_of_points := Nat.choose nonagon_vertices 2

def sides_of_nonagon : ℕ := nonagon_vertices

def diagonals_of_nonagon := total_pairs_of_points - sides_of_nonagon

def pairs_of_diagonals := Nat.choose diagonals_of_nonagon 2

def sets_of_intersecting_diagonals := Nat.choose nonagon_vertices 4

noncomputable def probability_intersection_inside_nonagon : ℚ :=
  sets_of_intersecting_diagonals / pairs_of_diagonals

theorem probability_intersection_inside_nonagon_correct :
  probability_intersection_inside_nonagon = 14 / 39 := 
  sorry

end probability_intersection_inside_nonagon_correct_l41_41329


namespace sum_of_series_l41_41330

noncomputable def sum_term (k : ℕ) : ℝ :=
  (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

theorem sum_of_series : (∑' k : ℕ, sum_term (k + 1)) = 7 / 4 := by
  sorry

end sum_of_series_l41_41330


namespace crayon_boxes_needed_l41_41290

theorem crayon_boxes_needed (total_crayons : ℕ) (crayons_per_box : ℕ) (h1 : total_crayons = 80) (h2 : crayons_per_box = 8) : (total_crayons / crayons_per_box) = 10 :=
by
  sorry

end crayon_boxes_needed_l41_41290


namespace perimeter_increase_ratio_of_sides_l41_41361

def width_increase (a : ℝ) : ℝ := 1.1 * a
def length_increase (b : ℝ) : ℝ := 1.2 * b
def original_perimeter (a b : ℝ) : ℝ := 2 * (a + b)
def new_perimeter (a b : ℝ) : ℝ := 2 * (1.1 * a + 1.2 * b)

theorem perimeter_increase : ∀ a b : ℝ, 
  (a > 0) → (b > 0) → 
  (new_perimeter a b - original_perimeter a b) / (original_perimeter a b) * 100 < 20 := 
by
  sorry

theorem ratio_of_sides (a b : ℝ) (h : new_perimeter a b = 1.18 * original_perimeter a b) : a / b = 1 / 4 := 
by
  sorry

end perimeter_increase_ratio_of_sides_l41_41361


namespace quadratic_has_two_distinct_real_roots_l41_41911

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ((k - 1) * x^2 + 2 * x - 2 = 0) → (1 / 2 < k ∧ k ≠ 1) :=
sorry

end quadratic_has_two_distinct_real_roots_l41_41911


namespace math_problem_l41_41371

theorem math_problem (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end math_problem_l41_41371


namespace mia_study_time_l41_41179

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end mia_study_time_l41_41179


namespace arith_seq_a15_l41_41772

variable {α : Type} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arith_seq_a15 (a : ℕ → α) (k l m : ℕ) (x y : α) 
  (h_seq : is_arith_seq a)
  (h_k : a k = x)
  (h_l : a l = y) :
  a (l + (l - k)) = 2 * y - x := 
  sorry

end arith_seq_a15_l41_41772


namespace students_on_field_trip_l41_41752

theorem students_on_field_trip 
    (vans : ℕ)
    (van_capacity : ℕ)
    (adults : ℕ)
    (students : ℕ)
    (H1 : vans = 3)
    (H2 : van_capacity = 8)
    (H3 : adults = 2)
    (H4 : students = vans * van_capacity - adults) :
    students = 22 := 
by 
  sorry

end students_on_field_trip_l41_41752


namespace last_digit_of_one_over_729_l41_41154

def last_digit_of_decimal_expansion (n : ℕ) : ℕ := (n % 10)

theorem last_digit_of_one_over_729 : last_digit_of_decimal_expansion (1 / 729) = 9 :=
sorry

end last_digit_of_one_over_729_l41_41154


namespace find_base_of_exponent_l41_41276

theorem find_base_of_exponent
  (x : ℝ)
  (h1 : 4 ^ (2 * x + 2) = (some_number : ℝ) ^ (3 * x - 1))
  (x_eq : x = 1) :
  some_number = 16 := 
by
  -- proof steps would go here
  sorry

end find_base_of_exponent_l41_41276


namespace find_y_minus_x_l41_41980

theorem find_y_minus_x (x y : ℝ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : y - x = 7.5 :=
by
  sorry

end find_y_minus_x_l41_41980


namespace repetend_of_frac_4_div_17_is_235294_l41_41305

noncomputable def decimalRepetend_of_4_div_17 : String :=
  let frac := 4 / 17
  let repetend := "235294"
  repetend

theorem repetend_of_frac_4_div_17_is_235294 :
  (∃ n m : ℕ, (4 / 17 : ℚ) = n + (m / 10^6) ∧ m % 10^6 = 235294) :=
sorry

end repetend_of_frac_4_div_17_is_235294_l41_41305


namespace expected_value_of_X_is_5_over_3_l41_41625

-- Define the probabilities of getting an interview with company A, B, and C
def P_A : ℚ := 2 / 3
def P_BC (p : ℚ) : ℚ := p

-- Define the random variable X representing the number of interview invitations
def X (P_A P_BC : ℚ) : ℚ := sorry

-- Define the probability of receiving no interview invitations
def P_X_0 (P_A P_BC : ℚ) : ℚ := (1 - P_A) * (1 - P_BC)^2

-- Given condition that P(X=0) is 1/12
def condition_P_X_0 (P_A P_BC : ℚ) : Prop := P_X_0 P_A P_BC = 1 / 12

-- Given p = 1/2 as per the problem solution
def p : ℚ := 1 / 2

-- Expected value of X
def E_X (P_A P_BC : ℚ) : ℚ := (1 * (2 * P_BC * (1 - P_BC) + 2 * P_BC^2 * (1 - P_BC) + (1 - P_A) * P_BC^2)) +
                               (2 * (P_A * P_BC * (1 - P_BC) + P_A * (1 - P_BC)^2 + P_BC * P_BC * (1 - P_A))) +
                               (3 * (P_A * P_BC^2))

-- Theorem proving the expected value of X given the above conditions
theorem expected_value_of_X_is_5_over_3 : E_X P_A (P_BC p) = 5 / 3 :=
by
  -- here you will write the proof later
  sorry

end expected_value_of_X_is_5_over_3_l41_41625


namespace sequence_value_x_l41_41490

theorem sequence_value_x (x : ℕ) (h1 : 1 + 3 = 4) (h2 : 4 + 3 = 7) (h3 : 7 + 3 = 10) (h4 : 10 + 3 = x) (h5 : x + 3 = 16) : x = 13 := by
  sorry

end sequence_value_x_l41_41490


namespace total_air_removed_after_5_strokes_l41_41640

theorem total_air_removed_after_5_strokes:
  let initial_air := 1
  let remaining_air_after_first_stroke := initial_air * (2 / 3)
  let remaining_air_after_second_stroke := remaining_air_after_first_stroke * (3 / 4)
  let remaining_air_after_third_stroke := remaining_air_after_second_stroke * (4 / 5)
  let remaining_air_after_fourth_stroke := remaining_air_after_third_stroke * (5 / 6)
  let remaining_air_after_fifth_stroke := remaining_air_after_fourth_stroke * (6 / 7)
  initial_air - remaining_air_after_fifth_stroke = 5 / 7 := by
  sorry

end total_air_removed_after_5_strokes_l41_41640


namespace true_discount_correct_l41_41214

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  (BD * FV) / (BD + FV)

theorem true_discount_correct :
  true_discount 270 54 = 45 :=
by
  sorry

end true_discount_correct_l41_41214


namespace square_roots_N_l41_41914

theorem square_roots_N (m N : ℤ) (h1 : (3 * m - 4) ^ 2 = N) (h2 : (7 - 4 * m) ^ 2 = N) : N = 25 := 
by
  sorry

end square_roots_N_l41_41914


namespace john_spent_at_candy_store_l41_41726

-- Conditions
def weekly_allowance : ℚ := 2.25
def spent_at_arcade : ℚ := (3 / 5) * weekly_allowance
def remaining_after_arcade : ℚ := weekly_allowance - spent_at_arcade
def spent_at_toy_store : ℚ := (1 / 3) * remaining_after_arcade
def remaining_after_toy_store : ℚ := remaining_after_arcade - spent_at_toy_store

-- Problem: Prove that John spent $0.60 at the candy store
theorem john_spent_at_candy_store : remaining_after_toy_store = 0.60 :=
by
  sorry

end john_spent_at_candy_store_l41_41726


namespace percentage_exceeds_self_l41_41576

theorem percentage_exceeds_self (N : ℕ) (P : ℝ) (h1 : N = 150) (h2 : N = (P / 100) * N + 126) : P = 16 := by
  sorry

end percentage_exceeds_self_l41_41576


namespace inequality_proof_l41_41088

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by sorry

end inequality_proof_l41_41088


namespace sequence_bounds_l41_41332

theorem sequence_bounds (θ : ℝ) (n : ℕ) (a : ℕ → ℝ) (hθ : 0 < θ ∧ θ < π / 2) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1 - 2 * (Real.sin θ * Real.cos θ)^2) 
  (h_recurrence : ∀ n, a (n + 2) - a (n + 1) + a n * (Real.sin θ * Real.cos θ)^2 = 0) :
  1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - (Real.sin (2 * θ))^n * (1 - 1 / 2 ^ (n - 1)) := 
sorry

end sequence_bounds_l41_41332


namespace find_f_l41_41712

theorem find_f (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > 0)
  (h2 : f 1 = 1)
  (h3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2) : ∀ x : ℝ, f x = x := by
  sorry

end find_f_l41_41712


namespace abcd_sum_l41_41335

theorem abcd_sum : 
  ∃ (a b c d : ℕ), 
    (∃ x y : ℝ, x + y = 5 ∧ 2 * x * y = 6 ∧ 
      (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)) →
    a + b + c + d = 21 :=
by
  sorry

end abcd_sum_l41_41335


namespace josette_paid_correct_amount_l41_41527

-- Define the number of small and large bottles
def num_small_bottles : ℕ := 3
def num_large_bottles : ℕ := 2

-- Define the cost of each type of bottle
def cost_per_small_bottle : ℝ := 1.50
def cost_per_large_bottle : ℝ := 2.40

-- Define the total number of bottles purchased
def total_bottles : ℕ := num_small_bottles + num_large_bottles

-- Define the discount rate applicable when purchasing 5 or more bottles
def discount_rate : ℝ := 0.10

-- Calculate the initial total cost before any discount
def total_cost_before_discount : ℝ :=
  (num_small_bottles * cost_per_small_bottle) + 
  (num_large_bottles * cost_per_large_bottle)

-- Calculate the discount amount if applicable
def discount_amount : ℝ :=
  if total_bottles >= 5 then
    discount_rate * total_cost_before_discount
  else
    0

-- Calculate the final amount Josette paid after applying any discount
def final_amount_paid : ℝ :=
  total_cost_before_discount - discount_amount

-- Prove that the final amount paid is €8.37
theorem josette_paid_correct_amount :
  final_amount_paid = 8.37 :=
by
  sorry

end josette_paid_correct_amount_l41_41527


namespace smallest_unfound_digit_in_odd_units_l41_41414

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l41_41414


namespace find_triangle_C_coordinates_find_triangle_area_l41_41213

noncomputable def triangle_C_coordinates (A B : (ℝ × ℝ)) (median_eq altitude_eq : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ C : ℝ × ℝ, C = (3, 1) ∧
    let A := (1,2)
    let B := (3, 4)
    let median_eq := (2, 1, -7)
    let altitude_eq := (2, -1, -2)
    true

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : Prop :=
  ∃ S : ℝ, S = 3 ∧
    let A := (1,2)
    let B := (3, 4)
    let C := (3, 1)
    true

theorem find_triangle_C_coordinates : triangle_C_coordinates (1,2) (3,4) (2, 1, -7) (2, -1, -2) :=
by { sorry }

theorem find_triangle_area : triangle_area (1,2) (3,4) (3,1) :=
by { sorry }

end find_triangle_C_coordinates_find_triangle_area_l41_41213


namespace perpendicular_pair_is_14_l41_41101

variable (x y : ℝ)

def equation1 := 4 * y - 3 * x = 16
def equation2 := -3 * x - 4 * y = 15
def equation3 := 4 * y + 3 * x = 16
def equation4 := 3 * y + 4 * x = 15

theorem perpendicular_pair_is_14 : (∃ y1 y2 x1 x2 : ℝ,
  4 * y1 - 3 * x1 = 16 ∧ 3 * y2 + 4 * x2 = 15 ∧ (3 / 4) * (-4 / 3) = -1) :=
sorry

end perpendicular_pair_is_14_l41_41101


namespace Joey_swimming_days_l41_41145

-- Define the conditions and required proof statement
theorem Joey_swimming_days (E : ℕ) (h1 : 3 * E / 4 = 9) : E / 2 = 6 :=
by
  sorry

end Joey_swimming_days_l41_41145


namespace find_smallest_result_l41_41363

namespace small_result

def num_set : Set Int := { -10, -4, 0, 2, 7 }

def all_results : Set Int := 
  { z | ∃ x ∈ num_set, ∃ y ∈ num_set, z = x * y ∨ z = x + y }

def smallest_result := -70

theorem find_smallest_result : ∃ z ∈ all_results, z = smallest_result :=
by
  sorry

end small_result

end find_smallest_result_l41_41363


namespace cos_double_angle_example_l41_41572

theorem cos_double_angle_example (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 := by
  sorry

end cos_double_angle_example_l41_41572


namespace words_per_page_l41_41263

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l41_41263


namespace geometric_sequence_formula_l41_41487

noncomputable def a_n (q : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n - 1)

theorem geometric_sequence_formula (q : ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0) :
  a_n q n = 2^(n - 1) :=
sorry

end geometric_sequence_formula_l41_41487


namespace opposite_of_two_l41_41686

def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_two : opposite 2 = -2 :=
by
  -- proof skipped
  sorry

end opposite_of_two_l41_41686


namespace cube_difference_l41_41538

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l41_41538


namespace resulting_ratio_correct_l41_41654

-- Define initial conditions
def initial_coffee : ℕ := 20
def joe_drank : ℕ := 3
def joe_added_cream : ℕ := 4
def joAnn_added_cream : ℕ := 3
def joAnn_drank : ℕ := 4

-- Define the resulting amounts of cream
def joe_cream : ℕ := joe_added_cream
def joAnn_initial_cream_frac : ℚ := joAnn_added_cream / (initial_coffee + joAnn_added_cream)
def joAnn_cream_drank : ℚ := (joAnn_drank : ℚ) * joAnn_initial_cream_frac
def joAnn_cream_left : ℚ := joAnn_added_cream - joAnn_cream_drank

-- Define the resulting ratio of cream in Joe's coffee to JoAnn's coffee
def resulting_ratio : ℚ := joe_cream / joAnn_cream_left

-- Theorem stating the resulting ratio is 92/45
theorem resulting_ratio_correct : resulting_ratio = 92 / 45 :=
by
  unfold resulting_ratio joe_cream joAnn_cream_left joAnn_cream_drank joAnn_initial_cream_frac
  norm_num
  sorry

end resulting_ratio_correct_l41_41654


namespace trees_planted_l41_41659

theorem trees_planted (current_short_trees planted_short_trees total_short_trees : ℕ)
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees = 217) :
  planted_short_trees = 105 :=
by
  sorry

end trees_planted_l41_41659


namespace minimum_value_l41_41401

open Real

theorem minimum_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ z : ℝ, (z = (3 / 2) * x^2 + y^2) ∧ z = 15 :=
by
  sorry

end minimum_value_l41_41401


namespace percentage_increase_in_area_l41_41317

variable (L W : Real)

theorem percentage_increase_in_area (hL : L > 0) (hW : W > 0) :
  ((1 + 0.25) * L * (1 + 0.25) * W - L * W) / (L * W) * 100 = 56.25 := by
  sorry

end percentage_increase_in_area_l41_41317


namespace volume_of_circumscribed_sphere_of_cube_l41_41452

theorem volume_of_circumscribed_sphere_of_cube (a : ℝ) (h : a = 1) : 
  (4 / 3) * Real.pi * ((Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 / 2) * Real.pi :=
by sorry

end volume_of_circumscribed_sphere_of_cube_l41_41452


namespace angle_B_measure_l41_41273

open Real EuclideanGeometry Classical

noncomputable def measure_angle_B (A C : ℝ) : ℝ := 180 - (180 - A - C)

theorem angle_B_measure
  (l m : ℝ → ℝ → Prop) -- parallel lines l and m (can be interpreted as propositions for simplicity)
  (h_parallel : ∀ x y, l x y → m x y → x = y) -- Lines l and m are parallel
  (A C : ℝ)
  (hA : A = 120)
  (hC : C = 70) :
  measure_angle_B A C = 130 := 
by
  sorry

end angle_B_measure_l41_41273


namespace base7_addition_XY_l41_41521

theorem base7_addition_XY (X Y : ℕ) (h1 : (Y + 2) % 7 = X % 7) (h2 : (X + 5) % 7 = 9 % 7) : X + Y = 6 :=
by sorry

end base7_addition_XY_l41_41521


namespace shortest_player_height_l41_41137

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l41_41137


namespace cubic_identity_l41_41151

theorem cubic_identity (x : ℝ) (h : x + 1/x = -6) : x^3 + 1/x^3 = -198 := 
by
  sorry

end cubic_identity_l41_41151


namespace exists_a_div_by_3_l41_41110

theorem exists_a_div_by_3 (a : ℝ) (h : ∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 4) = k) :
  ∃ k : ℤ, a = k / 3 :=
by
  sorry

end exists_a_div_by_3_l41_41110


namespace sequences_recurrence_relation_l41_41025

theorem sequences_recurrence_relation 
    (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : a 1 = 1) (h2 : b 1 = 3) (h3 : c 1 = 2)
    (ha : ∀ i : ℕ, a (i + 1) = a i + c i - b i + 2)
    (hb : ∀ i : ℕ, b (i + 1) = (3 * c i - a i + 5) / 2)
    (hc : ∀ i : ℕ, c (i + 1) = 2 * a i + 2 * b i - 3) : 
    (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2^n + 1) ∧ (∀ n, c n = 3 * 2^(n-1) - 1) := 
sorry

end sequences_recurrence_relation_l41_41025


namespace four_times_sum_of_squares_gt_sum_squared_l41_41478

open Real

theorem four_times_sum_of_squares_gt_sum_squared
  {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 :=
sorry

end four_times_sum_of_squares_gt_sum_squared_l41_41478


namespace values_of_n_l41_41160

/-
  Given a natural number n and a target sum 100,
  we need to find if there exists a combination of adding and subtracting 1 through n
  such that the sum equals 100.

- A value k is representable as a sum or difference of 1 through n if the sum of the series
  can be manipulated to produce k.
- The sum of the first n natural numbers S_n = n * (n + 1) / 2 must be even and sufficiently large.
- The specific values that satisfy the conditions are of the form n = 15 + 4 * k or n = 16 + 4 * k.
-/

def exists_sum_to_100 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k

theorem values_of_n (n : ℕ) : exists_sum_to_100 n ↔ (∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k) :=
by { sorry }

end values_of_n_l41_41160


namespace income_expenditure_ratio_l41_41681

theorem income_expenditure_ratio (I E S : ℕ) (hI : I = 19000) (hS : S = 11400) (hRel : S = I - E) :
  I / E = 95 / 38 :=
by
  sorry

end income_expenditure_ratio_l41_41681


namespace find_n_tan_eq_348_l41_41485

theorem find_n_tan_eq_348 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) : 
  (Real.tan (n * Real.pi / 180) = Real.tan (348 * Real.pi / 180)) ↔ (n = -12) := by
  sorry

end find_n_tan_eq_348_l41_41485


namespace m_and_n_must_have_same_parity_l41_41482

-- Define the problem conditions
def square_has_four_colored_edges (square : Type) : Prop :=
  ∃ (colors : Fin 4 → square), true

def m_and_n_same_parity (m n : ℕ) : Prop :=
  (m % 2 = n % 2)

-- Formalize the proof statement based on the conditions
theorem m_and_n_must_have_same_parity (m n : ℕ) (square : Type)
  (H : square_has_four_colored_edges square) : 
  m_and_n_same_parity m n :=
by 
  sorry

end m_and_n_must_have_same_parity_l41_41482


namespace polynomial_remainder_l41_41462

theorem polynomial_remainder (y : ℝ) : 
  let a := 3 ^ 50 - 2 ^ 50
  let b := 2 ^ 50 - 2 * 3 ^ 50 + 2 ^ 51
  (y ^ 50) % (y ^ 2 - 5 * y + 6) = a * y + b :=
by
  sorry

end polynomial_remainder_l41_41462


namespace xy_square_sum_l41_41581

theorem xy_square_sum (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 :=
by
  sorry

end xy_square_sum_l41_41581


namespace ranking_Fiona_Giselle_Ella_l41_41999

-- Definitions of scores 
variable (score : String → ℕ)

-- Conditions based on the problem statement
def ella_not_highest : Prop := ¬ (score "Ella" = max (score "Ella") (max (score "Fiona") (score "Giselle")))
def giselle_not_lowest : Prop := ¬ (score "Giselle" = min (score "Ella") (score "Giselle"))

-- The goal is to rank the scores from highest to lowest
def score_ranking : Prop := (score "Fiona" > score "Giselle") ∧ (score "Giselle" > score "Ella")

theorem ranking_Fiona_Giselle_Ella :
  ella_not_highest score →
  giselle_not_lowest score →
  score_ranking score :=
by
  sorry

end ranking_Fiona_Giselle_Ella_l41_41999


namespace polygon_area_144_l41_41152

-- Given definitions
def polygon (n : ℕ) : Prop := -- definition to capture n squares arrangement
  n = 36

def is_perpendicular (sides : ℕ) : Prop := -- every pair of adjacent sides is perpendicular
  sides = 4

def all_sides_congruent (length : ℕ) : Prop := -- all sides have the same length
  true

def total_perimeter (perimeter : ℕ) : Prop := -- total perimeter of the polygon
  perimeter = 72

-- The side length s leading to polygon's perimeter
def side_length (s perimeter : ℕ) : Prop :=
  perimeter = 36 * s / 2 

-- Prove the area of polygon is 144
theorem polygon_area_144 (n sides length perimeter s: ℕ) 
    (h1 : polygon n) 
    (h2 : is_perpendicular sides) 
    (h3 : all_sides_congruent length) 
    (h4 : total_perimeter perimeter) 
    (h5 : side_length s perimeter) : 
    n * s * s = 144 := 
sorry

end polygon_area_144_l41_41152


namespace max_k_l41_41776

-- Definitions and conditions
def original_number (A B : ℕ) : ℕ := 10 * A + B
def new_number (A C B : ℕ) : ℕ := 100 * A + 10 * C + B

theorem max_k (A C B k : ℕ) (hA : A ≠ 0) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3: 0 ≤ C ∧ C ≤ 9) :
  ((original_number A B) * k = (new_number A C B)) → 
  (∀ (A: ℕ), 1 ≤ k) → 
  k ≤ 19 :=
by
  sorry

end max_k_l41_41776


namespace sum_of_max_min_a_l41_41939

theorem sum_of_max_min_a (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 20 * a^2 < 0) →
  (∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 ∧ x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) →
  (∀ max_min_sum : ℝ, max_min_sum = 1 + (-1) → max_min_sum = 0) := 
sorry

end sum_of_max_min_a_l41_41939


namespace find_k_inv_h_8_l41_41882

variable (h k : ℝ → ℝ)

-- Conditions
axiom h_inv_k_x (x : ℝ) : h⁻¹ (k x) = 3 * x - 4
axiom h_3x_minus_4 (x : ℝ) : k x = h (3 * x - 4)

-- The statement we want to prove
theorem find_k_inv_h_8 : k⁻¹ (h 8) = 8 := 
  sorry

end find_k_inv_h_8_l41_41882


namespace new_average_of_remaining_numbers_l41_41593

theorem new_average_of_remaining_numbers (sum_12 avg_12 n1 n2 : ℝ) 
  (h1 : avg_12 = 90)
  (h2 : sum_12 = 1080)
  (h3 : n1 = 80)
  (h4 : n2 = 85)
  : (sum_12 - n1 - n2) / 10 = 91.5 := 
by
  sorry

end new_average_of_remaining_numbers_l41_41593


namespace find_x_l41_41043

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l41_41043


namespace find_constants_l41_41824

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![0, 4]
]

def I : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 0],
  ![0, 1]
]

theorem find_constants (c d : ℚ) : 
  (N⁻¹ = c • N + d • I) ↔ (c = -1/12 ∧ d = 7/12) :=
by
  sorry

end find_constants_l41_41824


namespace slices_with_only_mushrooms_l41_41199

theorem slices_with_only_mushrooms :
  ∀ (T P M n : ℕ),
    T = 16 →
    P = 9 →
    M = 12 →
    (9 - n) + (12 - n) + n = 16 →
    M - n = 7 :=
by
  intros T P M n hT hP hM h_eq
  sorry

end slices_with_only_mushrooms_l41_41199


namespace gcd_102_238_l41_41415

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l41_41415


namespace increase_in_area_l41_41816

theorem increase_in_area (a : ℝ) : 
  let original_radius := 3
  let new_radius := original_radius + a
  let original_area := π * original_radius ^ 2
  let new_area := π * new_radius ^ 2
  new_area - original_area = π * (3 + a) ^ 2 - 9 * π := 
by
  sorry

end increase_in_area_l41_41816


namespace inequality_proof_l41_41535

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = x^2 + 3 * x + 2) →
  a > 0 →
  b > 0 →
  b ≤ a / 7 →
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) :=
by
  sorry

end inequality_proof_l41_41535


namespace second_player_win_strategy_l41_41262

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end second_player_win_strategy_l41_41262


namespace minimum_rotation_angle_of_square_l41_41631

theorem minimum_rotation_angle_of_square : 
  ∀ (angle : ℝ), (∃ n : ℕ, angle = 360 / n) ∧ (n ≥ 1) ∧ (n ≤ 4) → angle = 90 :=
by
  sorry

end minimum_rotation_angle_of_square_l41_41631


namespace equation_of_line_l41_41129

noncomputable def vector := (Real × Real)
noncomputable def point := (Real × Real)

def line_equation (x y : Real) : Prop := 
  let v1 : vector := (-1, 2)
  let p : point := (3, -4)
  let lhs := (v1.1 * (x - p.1) + v1.2 * (y - p.2)) = 0
  lhs

theorem equation_of_line (x y : Real) :
  line_equation x y ↔ y = (1/2) * x - (11/2) := 
  sorry

end equation_of_line_l41_41129


namespace rectangle_perimeter_l41_41608

theorem rectangle_perimeter (A W : ℝ) (hA : A = 300) (hW : W = 15) : 
  (2 * ((A / W) + W)) = 70 := 
  sorry

end rectangle_perimeter_l41_41608


namespace notebook_problem_l41_41848

/-- Conditions:
1. If each notebook costs 3 yuan, 6 more notebooks can be bought.
2. If each notebook costs 5 yuan, there is a 30-yuan shortfall.

We need to show:
1. The total number of notebooks \( x \).
2. The number of 3-yuan notebooks \( n_3 \). -/
theorem notebook_problem (x y n3 : ℕ) (h1 : y = 3 * x + 18) (h2 : y = 5 * x - 30) (h3 : 3 * n3 + 5 * (x - n3) = y) :
  x = 24 ∧ n3 = 15 :=
by
  -- proof to be provided
  sorry

end notebook_problem_l41_41848


namespace find_m_l41_41952

theorem find_m (x y m : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : m * x - y = 3) : m = 6 := 
by
  sorry

end find_m_l41_41952


namespace find_people_and_carriages_l41_41442

theorem find_people_and_carriages (x y : ℝ) :
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) ↔
  (x / 3 = y + 2) ∧ ((x - 9) / 2 = y) :=
by
  sorry

end find_people_and_carriages_l41_41442


namespace train_crossing_time_l41_41034

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end train_crossing_time_l41_41034


namespace tina_total_leftover_l41_41067

def monthly_income : ℝ := 1000

def june_savings : ℝ := 0.25 * monthly_income
def june_expenses : ℝ := 200 + 0.05 * monthly_income
def june_leftover : ℝ := monthly_income - june_savings - june_expenses

def july_savings : ℝ := 0.20 * monthly_income
def july_expenses : ℝ := 250 + 0.15 * monthly_income
def july_leftover : ℝ := monthly_income - july_savings - july_expenses

def august_savings : ℝ := 0.30 * monthly_income
def august_expenses : ℝ := 250 + 50 + 0.10 * monthly_income
def august_gift : ℝ := 50
def august_leftover : ℝ := (monthly_income - august_savings - august_expenses) + august_gift

def total_leftover : ℝ :=
  june_leftover + july_leftover + august_leftover

theorem tina_total_leftover (I : ℝ) (hI : I = 1000) :
  total_leftover = 1250 := by
  rw [←hI] at *
  show total_leftover = 1250
  sorry

end tina_total_leftover_l41_41067


namespace complement_eq_target_l41_41051

namespace ComplementProof

-- Define the universal set U
def U : Set ℕ := {2, 4, 6, 8, 10}

-- Define the set A
def A : Set ℕ := {2, 6, 8}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the target set
def target_set : Set ℕ := {4, 10}

-- Prove that the complement of A with respect to U is equal to {4, 10}
theorem complement_eq_target :
  complement_U_A = target_set := by sorry

end ComplementProof

end complement_eq_target_l41_41051


namespace roots_of_quadratic_eval_l41_41094

theorem roots_of_quadratic_eval :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 4 * x₁ + 2 = 0) ∧ (x₂^2 + 4 * x₂ + 2 = 0) ∧ (x₁ + x₂ = -4) ∧ (x₁ * x₂ = 2) →
    x₁^3 + 14 * x₂ + 55 = 7 :=
by
  sorry

end roots_of_quadratic_eval_l41_41094


namespace original_denominator_is_21_l41_41698

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21_l41_41698


namespace max_combined_weight_l41_41676

theorem max_combined_weight (E A : ℕ) (h1 : A = 2 * E) (h2 : A + E = 90) (w_A : ℕ := 5) (w_E : ℕ := 2 * w_A) :
  E * w_E + A * w_A = 600 :=
by
  sorry

end max_combined_weight_l41_41676


namespace books_got_rid_of_l41_41434

-- Define the number of books they originally had
def original_books : ℕ := 87

-- Define the number of shelves used
def shelves_used : ℕ := 9

-- Define the number of books per shelf
def books_per_shelf : ℕ := 6

-- Define the number of books left after placing them on shelves
def remaining_books : ℕ := shelves_used * books_per_shelf

-- The statement to prove
theorem books_got_rid_of : original_books - remaining_books = 33 := 
by 
-- here is proof body you need to fill in 
  sorry

end books_got_rid_of_l41_41434


namespace trigonometric_values_l41_41841

theorem trigonometric_values (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1 / 3) 
  (h2 : Real.cos x - Real.cos y = 1 / 5) : 
  Real.cos (x + y) = 208 / 225 ∧ Real.sin (x - y) = -15 / 17 := 
by 
  sorry

end trigonometric_values_l41_41841


namespace condition_A_sufficient_not_necessary_condition_B_l41_41135

theorem condition_A_sufficient_not_necessary_condition_B {a b : ℝ} (hA : a > 1 ∧ b > 1) : 
  (a + b > 2 ∧ ab > 1) ∧ ¬∀ a b, (a + b > 2 ∧ ab > 1) → (a > 1 ∧ b > 1) :=
by
  sorry

end condition_A_sufficient_not_necessary_condition_B_l41_41135


namespace number_of_integers_in_double_inequality_l41_41526

noncomputable def pi_approx : ℝ := 3.14
noncomputable def sqrt_pi_approx : ℝ := Real.sqrt pi_approx
noncomputable def lower_bound : ℝ := -12 * sqrt_pi_approx
noncomputable def upper_bound : ℝ := 15 * pi_approx

theorem number_of_integers_in_double_inequality : 
  ∃ n : ℕ, n = 13 ∧ ∀ k : ℤ, lower_bound ≤ (k^2 : ℝ) ∧ (k^2 : ℝ) ≤ upper_bound → (-6 ≤ k ∧ k ≤ 6) :=
by
  sorry

end number_of_integers_in_double_inequality_l41_41526


namespace min_value_AF_BF_l41_41585

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 1

theorem min_value_AF_BF :
  ∀ (x1 x2 y1 y2 k : ℝ),
  parabola_eq x1 y1 →
  parabola_eq x2 y2 →
  line_eq k x1 = y1 →
  line_eq k x2 = y2 →
  (x1 ≠ x2) →
  parabola_focus = (0, 1) →
  (|y1 + 2| + 1) * (|y2 + 1|) = 2 * Real.sqrt 2 + 3 := 
by
  intros
  sorry

end min_value_AF_BF_l41_41585


namespace chocolates_initial_count_l41_41643

theorem chocolates_initial_count : 
  ∀ (chocolates_first_day chocolates_second_day chocolates_third_day chocolates_fourth_day chocolates_fifth_day initial_chocolates : ℕ),
  chocolates_first_day = 4 →
  chocolates_second_day = 2 * chocolates_first_day - 3 →
  chocolates_third_day = chocolates_first_day - 2 →
  chocolates_fourth_day = chocolates_third_day - 1 →
  chocolates_fifth_day = 12 →
  initial_chocolates = chocolates_first_day + chocolates_second_day + chocolates_third_day + chocolates_fourth_day + chocolates_fifth_day →
  initial_chocolates = 24 :=
by {
  -- the proof will go here,
  sorry
}

end chocolates_initial_count_l41_41643


namespace tens_of_80_tens_of_190_l41_41837

def tens_place (n : Nat) : Nat :=
  (n / 10) % 10

theorem tens_of_80 : tens_place 80 = 8 := 
  by
  sorry

theorem tens_of_190 : tens_place 190 = 9 := 
  by
  sorry

end tens_of_80_tens_of_190_l41_41837


namespace sum_seven_consecutive_integers_l41_41537

theorem sum_seven_consecutive_integers (n : ℕ) : 
  ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) = 7 * k := 
by 
  -- Use sum of integers and factor to demonstrate that the sum is multiple of 7
  sorry

end sum_seven_consecutive_integers_l41_41537


namespace relationship_between_m_and_n_l41_41689

theorem relationship_between_m_and_n
  (m n : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0)
  (line_eq : ∀ (x y : ℝ), m * x + 2 * n * y - 4 = 0) :
  m - n - 2 = 0 := 
  sorry

end relationship_between_m_and_n_l41_41689


namespace roots_greater_than_two_range_l41_41984

theorem roots_greater_than_two_range (m : ℝ) :
  ∀ x1 x2 : ℝ, (x1^2 + (m - 4) * x1 + 6 - m = 0) ∧ (x2^2 + (m - 4) * x2 + 6 - m = 0) ∧ (x1 > 2) ∧ (x2 > 2) →
  -2 < m ∧ m ≤ 2 - 2 * Real.sqrt 3 :=
by
  sorry

end roots_greater_than_two_range_l41_41984


namespace find_c_if_quadratic_lt_zero_l41_41498

theorem find_c_if_quadratic_lt_zero (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c * x - 12 < 0 ↔ (x < 2 ∨ x > 7)) → c = 9 := 
by
  sorry

end find_c_if_quadratic_lt_zero_l41_41498


namespace z_in_fourth_quadrant_l41_41888

-- Given complex numbers z1 and z2
def z1 : ℂ := 3 - 2 * Complex.I
def z2 : ℂ := 1 + Complex.I

-- Define the multiplication of z1 and z2
def z : ℂ := z1 * z2

-- Prove that z is located in the fourth quadrant
theorem z_in_fourth_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- Construction and calculations skipped for the math proof,
  -- the result should satisfy the conditions for being in the fourth quadrant
  sorry

end z_in_fourth_quadrant_l41_41888


namespace find_counterfeit_two_weighings_l41_41826

-- defining the variables and conditions
variable (coins : Fin 7 → ℝ)
variable (real_weight : ℝ)
variable (fake_weight : ℝ)
variable (is_counterfeit : Fin 7 → Prop)

-- conditions
axiom counterfeit_weight_diff : ∀ i, is_counterfeit i ↔ (coins i = fake_weight)
axiom consecutive_counterfeits : ∃ (start : Fin 7), ∀ i, (start ≤ i ∧ i < start + 4) → is_counterfeit (i % 7)
axiom weight_diff : fake_weight < real_weight

-- Theorem statement
theorem find_counterfeit_two_weighings : 
  (coins (1 : Fin 7) + coins (2 : Fin 7) = coins (4 : Fin 7) + coins (5 : Fin 7)) →
  is_counterfeit (6 : Fin 7) ∧ is_counterfeit (7 : Fin 7) := 
sorry

end find_counterfeit_two_weighings_l41_41826


namespace find_solutions_l41_41080

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end find_solutions_l41_41080


namespace nell_has_cards_left_l41_41114

def initial_cards : ℕ := 242
def cards_given_away : ℕ := 136

theorem nell_has_cards_left :
  initial_cards - cards_given_away = 106 :=
by
  sorry

end nell_has_cards_left_l41_41114


namespace volume_of_rectangular_prism_l41_41889

-- Defining the conditions as assumptions
variables (l w h : ℝ) 
variable (lw_eq : l * w = 10)
variable (wh_eq : w * h = 14)
variable (lh_eq : l * h = 35)

-- Stating the theorem to prove
theorem volume_of_rectangular_prism : l * w * h = 70 :=
by
  have lw := lw_eq
  have wh := wh_eq
  have lh := lh_eq
  sorry

end volume_of_rectangular_prism_l41_41889


namespace negation_if_proposition_l41_41026

variable (a b : Prop)

theorem negation_if_proposition (a b : Prop) : ¬ (a → b) = a ∧ ¬b := 
sorry

end negation_if_proposition_l41_41026


namespace average_weight_of_all_boys_l41_41956

theorem average_weight_of_all_boys 
  (n₁ n₂ : ℕ) (w₁ w₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : w₁ = 50.25) 
  (h₃ : n₂ = 8) (h₄ : w₂ = 45.15) :
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂) = 48.79 := 
by
  sorry

end average_weight_of_all_boys_l41_41956


namespace inequality_solution_l41_41493

theorem inequality_solution (a : ℝ) (h : a^2 > 2 * a - 1) : a ≠ 1 := 
sorry

end inequality_solution_l41_41493


namespace remainder_when_112222333_divided_by_37_l41_41391

theorem remainder_when_112222333_divided_by_37 : 112222333 % 37 = 0 :=
by
  sorry

end remainder_when_112222333_divided_by_37_l41_41391


namespace pencils_multiple_of_30_l41_41187

-- Defines the conditions of the problem
def num_pens : ℕ := 2010
def max_students : ℕ := 30
def equal_pens_per_student := num_pens % max_students = 0

-- Proves that the number of pencils must be a multiple of 30
theorem pencils_multiple_of_30 (P : ℕ) (h1 : equal_pens_per_student) (h2 : ∀ n, n ≤ max_students → ∃ m, n * m = num_pens) : ∃ k : ℕ, P = max_students * k :=
sorry

end pencils_multiple_of_30_l41_41187


namespace parallelogram_s_value_l41_41908

noncomputable def parallelogram_area (s : ℝ) : ℝ :=
  s * 2 * (s / Real.sqrt 2)

theorem parallelogram_s_value (s : ℝ) (h₀ : parallelogram_area s = 8 * Real.sqrt 2) : 
  s = 2 * Real.sqrt 2 :=
by
  sorry

end parallelogram_s_value_l41_41908


namespace length_of_plot_l41_41233

theorem length_of_plot (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) 
  (h1 : cost_per_meter = 26.50) 
  (h2 : total_cost = 5300)
  (h3 : breadth + 20 = 60) :
  2 * ((breadth + 20) + breadth) = total_cost / cost_per_meter := 
by
  sorry

end length_of_plot_l41_41233


namespace region_area_l41_41162

theorem region_area (x y : ℝ) : 
  (|2 * x - 16| + |3 * y + 9| ≤ 6) → ∃ A, A = 72 :=
sorry

end region_area_l41_41162


namespace expected_value_is_minus_one_fifth_l41_41017

-- Define the parameters given in the problem
def p_heads := 2 / 5
def p_tails := 3 / 5
def win_heads := 4
def loss_tails := -3

-- Calculate the expected value for heads and tails
def expected_heads := p_heads * win_heads
def expected_tails := p_tails * loss_tails

-- The theorem stating that the expected value is -1/5
theorem expected_value_is_minus_one_fifth :
  expected_heads + expected_tails = -1 / 5 :=
by
  -- The proof can be filled in here
  sorry

end expected_value_is_minus_one_fifth_l41_41017


namespace sum_of_eight_numbers_l41_41511

theorem sum_of_eight_numbers (average : ℝ) (h : average = 5) :
  (8 * average) = 40 :=
by
  sorry

end sum_of_eight_numbers_l41_41511


namespace simplify_and_evaluate_expr_l41_41611

theorem simplify_and_evaluate_expr (x : Real) (h : x = Real.sqrt 3 - 1) :
  1 - (x / (x + 1)) / (x / (x ^ 2 - 1)) = 3 - Real.sqrt 3 :=
sorry

end simplify_and_evaluate_expr_l41_41611


namespace bird_families_to_Asia_l41_41702

theorem bird_families_to_Asia (total_families initial_families left_families went_to_Africa went_to_Asia: ℕ) 
  (h1 : total_families = 85) 
  (h2 : went_to_Africa = 23) 
  (h3 : left_families = 25) 
  (h4 : went_to_Asia = total_families - left_families - went_to_Africa) 
  : went_to_Asia = 37 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  exact h4

end bird_families_to_Asia_l41_41702


namespace polynomial_divisibility_by_120_l41_41732

theorem polynomial_divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end polynomial_divisibility_by_120_l41_41732


namespace block_wall_min_blocks_l41_41150

theorem block_wall_min_blocks :
  ∃ n,
    n = 648 ∧
    ∀ (row_height wall_height block1_length block2_length wall_length: ℕ),
    row_height = 1 ∧
    wall_height = 8 ∧
    block1_length = 1 ∧
    block2_length = 3/2 ∧
    wall_length = 120 ∧
    (∀ i : ℕ, i < wall_height → ∃ k m : ℕ, k * block1_length + m * block2_length = wall_length) →
    n = (wall_height * (1 + 2 * 79))
:= by sorry

end block_wall_min_blocks_l41_41150


namespace tax_rate_computation_l41_41833

-- Define the inputs
def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 134.4

-- Define the derived taxable amount
def taxable_amount : ℝ := total_value - non_taxable_amount

-- Define the expected tax rate
def expected_tax_rate : ℝ := 0.12

-- State the theorem
theorem tax_rate_computation : 
  (tax_paid / taxable_amount * 100) = expected_tax_rate * 100 := 
by
  sorry

end tax_rate_computation_l41_41833


namespace rth_term_of_arithmetic_progression_l41_41177

noncomputable def Sn (n : ℕ) : ℕ := 2 * n + 3 * n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) : 
  (Sn r - Sn (r - 1)) = 3 * r^2 + 5 * r - 2 :=
by sorry

end rth_term_of_arithmetic_progression_l41_41177


namespace find_y_l41_41814

theorem find_y (y : ℝ) (h : (3 * y) / 7 = 12) : y = 28 :=
by
  -- The proof would go here
  sorry

end find_y_l41_41814


namespace ball_hit_ground_in_time_l41_41668

theorem ball_hit_ground_in_time :
  ∃ t : ℝ, t ≥ 0 ∧ -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 :=
by sorry

end ball_hit_ground_in_time_l41_41668


namespace percentage_third_day_l41_41892

def initial_pieces : ℕ := 1000
def percentage_first_day : ℝ := 0.10
def percentage_second_day : ℝ := 0.20
def pieces_left_after_third_day : ℕ := 504

theorem percentage_third_day :
  let pieces_first_day := initial_pieces * percentage_first_day
  let remaining_after_first_day := initial_pieces - pieces_first_day
  let pieces_second_day := remaining_after_first_day * percentage_second_day
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day
  let pieces_third_day := remaining_after_second_day - pieces_left_after_third_day
  (pieces_third_day / remaining_after_second_day * 100 = 30) :=
by
  sorry

end percentage_third_day_l41_41892


namespace Gina_gave_fraction_to_mom_l41_41336

variable (M : ℝ)

theorem Gina_gave_fraction_to_mom :
  (∃ M, M + (1/8 : ℝ) * 400 + (1/5 : ℝ) * 400 + 170 = 400) →
  M / 400 = 1/4 :=
by
  intro h
  sorry

end Gina_gave_fraction_to_mom_l41_41336


namespace min_value_a_l41_41573

theorem min_value_a (a b : ℕ) (h1: a = b - 2005) 
  (h2: ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p + q = a ∧ p * q = b) : a ≥ 95 := sorry

end min_value_a_l41_41573


namespace contradiction_even_odd_l41_41996

theorem contradiction_even_odd (a b c : ℕ) :
  (∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (¬((x % 2 = 0 ∧ y % 2 ≠ 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 = 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 ≠ 0 ∧ z % 2 = 0)))) → false :=
by
  sorry

end contradiction_even_odd_l41_41996


namespace geometric_sequence_sum_l41_41444

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (∀ n : ℕ, a (n + 1) = a n * r)

theorem geometric_sequence_sum
  (a_seq_geometric : is_geometric_sequence a)
  (a_pos : ∀ n : ℕ, a n > 0)
  (eqn : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 :=
by
  sorry

end geometric_sequence_sum_l41_41444


namespace polygon_with_interior_sum_1260_eq_nonagon_l41_41614

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l41_41614


namespace Xingyou_age_is_3_l41_41359

theorem Xingyou_age_is_3 (x : ℕ) (h1 : x = x) (h2 : x + 3 = 2 * x) : x = 3 :=
by
  sorry

end Xingyou_age_is_3_l41_41359


namespace find_f_three_l41_41003

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b
axiom f_two : f 2 = 3

theorem find_f_three : f 3 = 9 / 2 :=
by
  sorry

end find_f_three_l41_41003


namespace proof_problem_l41_41935

noncomputable def expr (a b : ℚ) : ℚ :=
  ((a / b + b / a + 2) * ((a + b) / (2 * a) - (b / (a + b)))) /
  ((a + 2 * b + b^2 / a) * (a / (a + b) + b / (a - b)))

theorem proof_problem : expr (3/4 : ℚ) (4/3 : ℚ) = -7/24 :=
by
  sorry

end proof_problem_l41_41935


namespace PropA_impl_PropB_not_PropB_impl_PropA_l41_41027

variable {x : ℝ}

def PropA (x : ℝ) : Prop := abs (x - 1) < 5
def PropB (x : ℝ) : Prop := abs (abs x - 1) < 5

theorem PropA_impl_PropB : PropA x → PropB x :=
by sorry

theorem not_PropB_impl_PropA : ¬(PropB x → PropA x) :=
by sorry

end PropA_impl_PropB_not_PropB_impl_PropA_l41_41027


namespace same_leading_digit_l41_41198

theorem same_leading_digit (n : ℕ) (hn : 0 < n) : 
  (∀ a k l : ℕ, (a * 10^k < 2^n ∧ 2^n < (a+1) * 10^k) ∧ (a * 10^l < 5^n ∧ 5^n < (a+1) * 10^l) → a = 3) := 
sorry

end same_leading_digit_l41_41198


namespace initial_marbles_l41_41690

-- Define the conditions as constants
def marbles_given_to_Juan : ℕ := 73
def marbles_left_with_Connie : ℕ := 70

-- Prove that Connie initially had 143 marbles
theorem initial_marbles (initial_marbles : ℕ) :
  initial_marbles = marbles_given_to_Juan + marbles_left_with_Connie → 
  initial_marbles = 143 :=
by
  intro h
  rw [h]
  rfl

end initial_marbles_l41_41690


namespace greatest_integer_less_than_M_over_100_l41_41903

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l41_41903


namespace shadow_length_correct_l41_41289

theorem shadow_length_correct :
  let light_source := (0, 16)
  let disc_center := (6, 10)
  let radius := 2
  let m := 4
  let n := 17
  let length_form := m * Real.sqrt n
  length_form = 4 * Real.sqrt 17 :=
by
  sorry

end shadow_length_correct_l41_41289


namespace find_root_interval_l41_41183

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem find_root_interval : ∃ k : ℕ, (f 1 < 0 ∧ f 2 > 0) → k = 1 :=
by
  sorry

end find_root_interval_l41_41183


namespace exist_n_div_k_l41_41753

open Function

theorem exist_n_div_k (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.gcd k 6 = 1) :
  ∃ n : ℕ, n ≥ 0 ∧ k ∣ (2^n + 3^n + 6^n - 1) := 
sorry

end exist_n_div_k_l41_41753


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l41_41072

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l41_41072


namespace fraction_evaluation_l41_41073

theorem fraction_evaluation :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 :=
by
  sorry

end fraction_evaluation_l41_41073


namespace rectangle_perimeter_l41_41200

theorem rectangle_perimeter (a b : ℝ) (h1 : (a + 3) * (b + 3) = a * b + 48) : 
  2 * (a + 3 + b + 3) = 38 :=
by
  sorry

end rectangle_perimeter_l41_41200


namespace h_plus_k_l41_41381

theorem h_plus_k :
  ∀ h k : ℝ, (∀ x : ℝ, x^2 + 4 * x + 4 = (x + h) ^ 2 - k) → h + k = 2 :=
by
  intro h k H
  -- using sorry to indicate the proof is omitted
  sorry

end h_plus_k_l41_41381


namespace daily_wage_of_man_l41_41857

-- Define the wages for men and women
variables (M W : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 24 * M + 16 * W = 11600
def condition2 : Prop := 12 * M + 37 * W = 11600

-- The theorem we want to prove
theorem daily_wage_of_man (h1 : condition1 M W) (h2 : condition2 M W) : M = 350 :=
by
  sorry

end daily_wage_of_man_l41_41857


namespace range_of_x_l41_41580

-- Define the problem conditions and the conclusion to be proved
theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y)
  (h_ineq : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, 1 ≤ x ∧ x < 3 / 2 :=
by
  sorry

end range_of_x_l41_41580


namespace arithmetic_seq_a7_l41_41167

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) (h1 : ∀ (n m : ℕ), a (n + m) = a n + m * d)
  (h2 : a 4 + a 9 = 24) (h3 : a 6 = 11) :
  a 7 = 13 :=
sorry

end arithmetic_seq_a7_l41_41167


namespace sarah_monthly_payment_l41_41022

noncomputable def monthly_payment (loan_amount : ℝ) (down_payment : ℝ) (years : ℝ) : ℝ :=
  let financed_amount := loan_amount - down_payment
  let months := years * 12
  financed_amount / months

theorem sarah_monthly_payment : monthly_payment 46000 10000 5 = 600 := by
  sorry

end sarah_monthly_payment_l41_41022


namespace find_number_l41_41077

theorem find_number:
  ∃ x : ℝ, (3/4 * x + 9 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 :=
by
  sorry

end find_number_l41_41077


namespace number_of_boys_in_class_l41_41396

theorem number_of_boys_in_class
  (g_ratio : ℕ) (b_ratio : ℕ) (total_students : ℕ)
  (h_ratio : g_ratio / b_ratio = 4 / 3)
  (h_total_students : g_ratio + b_ratio = 7 * (total_students / 56)) :
  total_students = 56 → 3 * (total_students / (4 + 3)) = 24 :=
by
  intros total_students_56
  sorry

end number_of_boys_in_class_l41_41396


namespace determine_chris_age_l41_41544

theorem determine_chris_age (a b c : ℚ)
  (h1 : (a + b + c) / 3 = 10)
  (h2 : c - 5 = 2 * a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 283 / 15 :=
by
  sorry

end determine_chris_age_l41_41544


namespace greatest_divisor_l41_41184

theorem greatest_divisor (d : ℕ) (h₀ : 1657 % d = 6) (h₁ : 2037 % d = 5) : d = 127 :=
by
  -- Proof skipped
  sorry

end greatest_divisor_l41_41184


namespace length_squared_t_graph_interval_l41_41450

noncomputable def p (x : ℝ) : ℝ := -x + 2
noncomputable def q (x : ℝ) : ℝ := x + 2
noncomputable def r (x : ℝ) : ℝ := 2
noncomputable def t (x : ℝ) : ℝ :=
  if x ≤ -2 then p x
  else if x ≤ 2 then r x
  else q x

theorem length_squared_t_graph_interval :
  let segment_length (f : ℝ → ℝ) (a b : ℝ) : ℝ := Real.sqrt ((f b - f a)^2 + (b - a)^2)
  segment_length t (-4) (-2) + segment_length t (-2) 2 + segment_length t 2 4 = 4 + 2 * Real.sqrt 32 →
  (4 + 2 * Real.sqrt 32)^2 = 80 :=
sorry

end length_squared_t_graph_interval_l41_41450


namespace geese_problem_l41_41040

theorem geese_problem 
  (G : ℕ)  -- Total number of geese in the original V formation
  (T : ℕ)  -- Number of geese that flew up from the trees to join the new V formation
  (h1 : G / 2 + T = 12)  -- Final number of geese flying in the V formation was 12 
  (h2 : T = G / 2)  -- Number of geese that flew out from the trees is the same as the number of geese that landed initially
: T = 6 := 
sorry

end geese_problem_l41_41040


namespace range_of_a_l41_41369

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = h x) →
  1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1 :=
sorry

end range_of_a_l41_41369


namespace new_person_weight_l41_41430

/-- Conditions: The average weight of 8 persons increases by 6 kg when a new person replaces one of them weighing 45 kg -/
theorem new_person_weight (W : ℝ) (new_person_wt : ℝ) (avg_increase : ℝ) (replaced_person_wt : ℝ) 
  (h1 : avg_increase = 6) (h2 : replaced_person_wt = 45) (weight_increase : 8 * avg_increase = new_person_wt - replaced_person_wt) :
  new_person_wt = 93 :=
by
  sorry

end new_person_weight_l41_41430


namespace carol_points_loss_l41_41520

theorem carol_points_loss 
  (first_round_points : ℕ) (second_round_points : ℕ) (end_game_points : ℕ) 
  (h1 : first_round_points = 17) 
  (h2 : second_round_points = 6) 
  (h3 : end_game_points = 7) : 
  (first_round_points + second_round_points - end_game_points = 16) :=
by 
  sorry

end carol_points_loss_l41_41520


namespace converse_of_implication_l41_41382

-- Given propositions p and q
variables (p q : Prop)

-- Proving the converse of "if p then q" is "if q then p"

theorem converse_of_implication (h : p → q) : q → p :=
sorry

end converse_of_implication_l41_41382


namespace pentagon_edges_and_vertices_sum_l41_41707

theorem pentagon_edges_and_vertices_sum :
  let edges := 5
  let vertices := 5
  edges + vertices = 10 := by
  sorry

end pentagon_edges_and_vertices_sum_l41_41707


namespace inscribed_sphere_to_cube_volume_ratio_l41_41406

theorem inscribed_sphere_to_cube_volume_ratio :
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  (V_sphere / V_cube) = Real.pi / 6 :=
by
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  sorry

end inscribed_sphere_to_cube_volume_ratio_l41_41406


namespace friend_gain_is_20_percent_l41_41965

noncomputable def original_cost : ℝ := 52325.58
noncomputable def loss_percentage : ℝ := 0.14
noncomputable def friend_selling_price : ℝ := 54000
noncomputable def friend_percentage_gain : ℝ :=
  ((friend_selling_price - (original_cost * (1 - loss_percentage))) / (original_cost * (1 - loss_percentage))) * 100

theorem friend_gain_is_20_percent :
  friend_percentage_gain = 20 := by
  sorry

end friend_gain_is_20_percent_l41_41965


namespace least_number_of_apples_l41_41087

theorem least_number_of_apples (b : ℕ) : (b % 3 = 2) → (b % 4 = 3) → (b % 5 = 1) → b = 11 :=
by
  intros h1 h2 h3
  sorry

end least_number_of_apples_l41_41087


namespace B_time_to_finish_race_l41_41192

theorem B_time_to_finish_race (t : ℝ) 
  (race_distance : ℝ := 130)
  (A_time : ℝ := 36)
  (A_beats_B_by : ℝ := 26)
  (A_speed : ℝ := race_distance / A_time) 
  (B_distance_when_A_finishes : ℝ := race_distance - A_beats_B_by) 
  (B_speed := B_distance_when_A_finishes / t) :
  B_speed * (t - A_time) = A_beats_B_by → t = 48 := 
by
  intros h
  sorry

end B_time_to_finish_race_l41_41192


namespace find_function_l41_41648

theorem find_function (f : ℚ → ℚ) (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_function_l41_41648


namespace complex_problem_solution_l41_41735

noncomputable def complex_problem (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) : ℂ :=
  (c^12 + d^12) / (c + d)^12

theorem complex_problem_solution (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) :
  complex_problem c d h1 h2 h3 = 2 / 81 := 
sorry

end complex_problem_solution_l41_41735


namespace find_special_four_digit_number_l41_41147

theorem find_special_four_digit_number :
  ∃ (N : ℕ), 
  (N % 131 = 112) ∧ 
  (N % 132 = 98) ∧ 
  (1000 ≤ N) ∧ 
  (N < 10000) ∧ 
  (N = 1946) :=
sorry

end find_special_four_digit_number_l41_41147


namespace ellipse_area_l41_41688

theorem ellipse_area (P : ℝ) (b : ℝ) (a : ℝ) (A : ℝ) (h1 : P = 18)
  (h2 : a = b + 4)
  (h3 : A = π * a * b) :
  A = 5 * π :=
by
  sorry

end ellipse_area_l41_41688


namespace no_solution_iff_discriminant_l41_41731

theorem no_solution_iff_discriminant (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) ↔ -2 ≤ k ∧ k ≤ 2 := by
  sorry

end no_solution_iff_discriminant_l41_41731


namespace probability_same_color_l41_41598

theorem probability_same_color :
  let bagA_white := 8
  let bagA_red := 4
  let bagB_white := 6
  let bagB_red := 6
  let totalA := bagA_white + bagA_red
  let totalB := bagB_white + bagB_red
  let prob_white_white := (bagA_white / totalA) * (bagB_white / totalB)
  let prob_red_red := (bagA_red / totalA) * (bagB_red / totalB)
  let total_prob := prob_white_white + prob_red_red
  total_prob = 1 / 2 := 
by 
  sorry

end probability_same_color_l41_41598


namespace subtract_decimal_l41_41267

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l41_41267


namespace age_ratio_in_9_years_l41_41621

-- Initial age definitions for Mike and Sam
def ages (m s : ℕ) : Prop :=
  (m - 5 = 2 * (s - 5)) ∧ (m - 12 = 3 * (s - 12))

-- Proof that in 9 years the ratio of their ages will be 3:2
theorem age_ratio_in_9_years (m s x : ℕ) (h_ages : ages m s) :
  (m + x) * 2 = 3 * (s + x) ↔ x = 9 :=
by {
  sorry
}

end age_ratio_in_9_years_l41_41621


namespace radish_patch_area_l41_41700

-- Definitions from the conditions
variables (R P : ℕ) -- R: area of radish patch, P: area of pea patch
variable (h1 : P = 2 * R) -- The pea patch is twice as large as the radish patch
variable (h2 : P / 6 = 5) -- One-sixth of the pea patch is 5 square feet

-- Goal statement
theorem radish_patch_area : R = 15 :=
by
  sorry

end radish_patch_area_l41_41700


namespace overtaking_time_l41_41788

variable (a_speed b_speed k_speed : ℕ)
variable (b_delay : ℕ) 
variable (t : ℕ)
variable (t_k : ℕ)

theorem overtaking_time (h1 : a_speed = 30)
                        (h2 : b_speed = 40)
                        (h3 : k_speed = 60)
                        (h4 : b_delay = 5)
                        (h5 : 30 * t = 40 * (t - 5))
                        (h6 : 30 * t = 60 * t_k)
                         : k_speed / 3 = 10 :=
by sorry

end overtaking_time_l41_41788


namespace allen_blocks_l41_41013

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l41_41013


namespace bird_families_flew_away_for_winter_l41_41352

def bird_families_africa : ℕ := 38
def bird_families_asia : ℕ := 80
def total_bird_families_flew_away : ℕ := bird_families_africa + bird_families_asia

theorem bird_families_flew_away_for_winter : total_bird_families_flew_away = 118 := by
  -- proof goes here (not required)
  sorry

end bird_families_flew_away_for_winter_l41_41352


namespace exists_real_solution_real_solution_specific_values_l41_41066

theorem exists_real_solution (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

theorem real_solution_specific_values  (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

end exists_real_solution_real_solution_specific_values_l41_41066


namespace find_parallel_line_through_P_l41_41426

noncomputable def line_parallel_passing_through (p : (ℝ × ℝ)) (line : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, _) := line
  let (x, y) := p
  (a, b, - (a * x + b * y))

theorem find_parallel_line_through_P :
  line_parallel_passing_through (4, -1) (3, -4, 6) = (3, -4, -16) :=
by 
  sorry

end find_parallel_line_through_P_l41_41426


namespace sam_bought_new_books_l41_41360

   def books_question (a m u : ℕ) : ℕ := (a + m) - u

   theorem sam_bought_new_books (a m u : ℕ) (h1 : a = 13) (h2 : m = 17) (h3 : u = 15) :
     books_question a m u = 15 :=
   by sorry
   
end sam_bought_new_books_l41_41360


namespace matrix_power_B_l41_41113

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem matrix_power_B :
  B ^ 150 = 1 :=
by sorry

end matrix_power_B_l41_41113


namespace remaining_credit_l41_41592

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l41_41592


namespace product_mod_9_l41_41312

theorem product_mod_9 (a b c : ℕ) (h1 : a % 6 = 2) (h2 : b % 7 = 3) (h3 : c % 8 = 4) : (a * b * c) % 9 = 6 :=
by
  sorry

end product_mod_9_l41_41312


namespace domain_of_composite_function_l41_41801

theorem domain_of_composite_function :
  ∀ (f : ℝ → ℝ), (∀ x, -1 ≤ x ∧ x ≤ 3 → ∃ y, f x = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f (2*x - 1) = y) :=
by
  intros f domain_f x hx
  sorry

end domain_of_composite_function_l41_41801


namespace largest_value_of_b_l41_41966

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b_l41_41966


namespace arithmetic_sequence_sum_a3_a4_a5_l41_41901

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_a3_a4_a5
  (ha : is_arithmetic_sequence a d)
  (h : a 2 + a 3 + a 4 = 12) : 
  (7 * (a 0 + a 6)) / 2 = 28 := 
sorry

end arithmetic_sequence_sum_a3_a4_a5_l41_41901


namespace tangent_circles_distance_l41_41296

-- Define the radii of the circles.
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 2

-- Define the condition that the circles are tangent.
def tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2 ∨ d = r1 - r2

-- State the theorem.
theorem tangent_circles_distance (d : ℝ) :
  tangent radius_O1 radius_O2 d → (d = 1 ∨ d = 5) :=
by
  sorry

end tangent_circles_distance_l41_41296


namespace solve_linear_system_l41_41847

theorem solve_linear_system (x y a : ℝ) (h1 : 4 * x + 3 * y = 1) (h2 : a * x + (a - 1) * y = 3) (hxy : x = y) : a = 11 :=
by
  sorry

end solve_linear_system_l41_41847


namespace intersection_M_N_l41_41541

def M : Set ℝ := { x | x^2 ≤ 4 }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≥ 1 }

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l41_41541


namespace problem_1_problem_2_l41_41348

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }

def B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 3 * a + 1 }

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = 1 / 4) : A ∩ B a = { x | 1 < x ∧ x < 7 / 4 } :=
by
  rw [h]
  sorry

-- Problem 2
theorem problem_2 : (∀ x, A x → B a x) → ∀ a, 1 / 3 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_problem_2_l41_41348


namespace sine_of_pi_minus_alpha_l41_41182

theorem sine_of_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 3) : Real.sin (π - α) = 1 / 3 :=
by
  sorry

end sine_of_pi_minus_alpha_l41_41182


namespace meal_combinations_count_l41_41982

/-- Define the number of menu items -/
def num_menu_items : ℕ := 15

/-- Define the number of distinct combinations of meals Maryam and Jorge can order,
    considering they may choose the same dish and distinguishing who orders what -/
theorem meal_combinations_count (maryam_dishes jorge_dishes : ℕ) : 
  maryam_dishes = num_menu_items ∧ jorge_dishes = num_menu_items → 
  maryam_dishes * jorge_dishes = 225 :=
by
  intros h
  simp only [num_menu_items] at h -- Utilize the definition of num_menu_items
  sorry

end meal_combinations_count_l41_41982


namespace dot_product_not_sufficient_nor_necessary_for_parallel_l41_41283

open Real

-- Definitions for plane vectors \overrightarrow{a} and \overrightarrow{b}
variables (a b : ℝ × ℝ)

-- Dot product definition for two plane vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Parallelism condition for plane vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k • v2) ∨ v2 = (k • v1)

-- Statement to be proved
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ¬ (∀ a b : ℝ × ℝ, (dot_product a b > 0) ↔ (parallel a b)) :=
sorry

end dot_product_not_sufficient_nor_necessary_for_parallel_l41_41283


namespace towers_remainder_l41_41287

noncomputable def count_towers (k : ℕ) : ℕ := sorry

theorem towers_remainder : (count_towers 9) % 1000 = 768 := sorry

end towers_remainder_l41_41287


namespace find_subtracted_value_l41_41065

theorem find_subtracted_value (N : ℤ) (V : ℤ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 :=
by
  sorry

end find_subtracted_value_l41_41065


namespace solve_equation_simplify_expression_l41_41728

-- Problem (1)
theorem solve_equation : ∀ x : ℝ, x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by
  sorry

-- Problem (2)
theorem simplify_expression : ∀ a b : ℝ, a ≠ b → (a ≠ 0 ∧ b ≠ 0) →
  (3 * a ^ 2 - 3 * b ^ 2) / (a ^ 2 * b + a * b ^ 2) /
  (1 - (a ^ 2 + b ^ 2) / (2 * a * b)) = -6 / (a - b) := by
  sorry

end solve_equation_simplify_expression_l41_41728


namespace find_a_and_other_root_l41_41243

-- Define the quadratic equation with a
def quadratic_eq (a x : ℝ) : ℝ := (a + 1) * x^2 + x - 1

-- Define the conditions where -1 is a root
def condition (a : ℝ) : Prop := quadratic_eq a (-1) = 0

theorem find_a_and_other_root (a : ℝ) :
  condition a → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ quadratic_eq 1 x = 0 ∧ x = 1 / 2) :=
by
  intro h
  sorry

end find_a_and_other_root_l41_41243


namespace find_f_l41_41665

theorem find_f (f : ℤ → ℤ) (h : ∀ n : ℤ, n^2 + 4 * (f n) = (f (f n))^2) :
  (∀ x : ℤ, f x = 1 + x) ∨
  (∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)) ∨
  (f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)) :=
sorry

end find_f_l41_41665


namespace blue_balls_count_l41_41437

theorem blue_balls_count (Y B : ℕ) (h_ratio : 4 * B = 3 * Y) (h_total : Y + B = 35) : B = 15 :=
sorry

end blue_balls_count_l41_41437


namespace value_of_each_other_toy_l41_41536

-- Definitions for the conditions
def total_toys : ℕ := 9
def total_worth : ℕ := 52
def single_toy_value : ℕ := 12

-- Definition to represent the value of each of the other toys
def other_toys_value (same_value : ℕ) : Prop :=
  (total_worth - single_toy_value) / (total_toys - 1) = same_value

-- The theorem to be proven
theorem value_of_each_other_toy : other_toys_value 5 :=
  sorry

end value_of_each_other_toy_l41_41536


namespace solve_for_x_l41_41959

theorem solve_for_x (x : ℝ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 := 
  sorry

end solve_for_x_l41_41959


namespace derek_age_l41_41871

theorem derek_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end derek_age_l41_41871


namespace quadratic_transformation_l41_41265

theorem quadratic_transformation :
  ∀ (x : ℝ), (x^2 + 6*x - 2 = 0) → ((x + 3)^2 = 11) :=
by
  intros x h
  sorry

end quadratic_transformation_l41_41265


namespace sin_135_degree_l41_41722

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l41_41722


namespace investment_period_more_than_tripling_l41_41049

theorem investment_period_more_than_tripling (r : ℝ) (multiple : ℝ) (n : ℕ) 
  (h_r: r = 0.341) (h_multiple: multiple > 3) :
  (1 + r)^n ≥ multiple → n = 4 :=
by
  sorry

end investment_period_more_than_tripling_l41_41049


namespace circle_area_sum_l41_41004

theorem circle_area_sum (x y z : ℕ) (A₁ A₂ A₃ total_area : ℕ) (h₁ : A₁ = 6) (h₂ : A₂ = 15) 
  (h₃ : A₃ = 83) (h₄ : total_area = 220) (hx : x = 4) (hy : y = 2) (hz : z = 2) :
  A₁ * x + A₂ * y + A₃ * z = total_area := by
  sorry

end circle_area_sum_l41_41004


namespace complement_of_A_in_U_l41_41899

open Set

-- Define the sets U and A with their respective elements in the real numbers
def U : Set ℝ := Icc 0 1
def A : Set ℝ := Ico 0 1

-- State the theorem
theorem complement_of_A_in_U : (U \ A) = {1} := by
  sorry

end complement_of_A_in_U_l41_41899


namespace factor_cubic_expression_l41_41936

theorem factor_cubic_expression :
  ∃ a b c : ℕ, 
  a > b ∧ b > c ∧ 
  x^3 - 16 * x^2 + 65 * x - 80 = (x - a) * (x - b) * (x - c) ∧ 
  3 * b - c = 12 := 
sorry

end factor_cubic_expression_l41_41936


namespace geom_seq_a4_a5_a6_value_l41_41257

theorem geom_seq_a4_a5_a6_value (a : ℕ → ℝ) (h_geom : ∃ r, 0 < r ∧ ∀ n, a (n + 1) = r * a n)
  (h_roots : ∃ x y, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 9 = y) :
  a 4 * a 5 * a 6 = 64 :=
by
  sorry

end geom_seq_a4_a5_a6_value_l41_41257


namespace correct_transformation_l41_41834

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l41_41834


namespace range_of_a_l41_41657

theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 - x ≤ 0 → 2^(1 - x) + a ≤ 0)) ↔ (a ≤ -2) := by
  sorry

end range_of_a_l41_41657


namespace prime_sum_20_to_30_l41_41163

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l41_41163


namespace leak_empties_tank_in_18_hours_l41_41090

theorem leak_empties_tank_in_18_hours :
  let A : ℚ := 1 / 6
  let L : ℚ := 1 / 6 - 1 / 9
  (1 / L) = 18 := by
    sorry

end leak_empties_tank_in_18_hours_l41_41090


namespace probability_two_girls_from_twelve_l41_41165

theorem probability_two_girls_from_twelve : 
  let total_members := 12
  let boys := 4
  let girls := 8
  let choose_two_total := Nat.choose total_members 2
  let choose_two_girls := Nat.choose girls 2
  let probability := (choose_two_girls : ℚ) / (choose_two_total : ℚ)
  probability = (14 / 33) := by
  -- Proof goes here
  sorry

end probability_two_girls_from_twelve_l41_41165


namespace fraction_of_grid_covered_by_triangle_l41_41299

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))|

noncomputable def area_of_grid : ℝ := 7 * 6

noncomputable def fraction_covered : ℝ :=
  area_of_triangle (-1, 2) (3, 5) (2, 2) / area_of_grid

theorem fraction_of_grid_covered_by_triangle : fraction_covered = (3 / 28) :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l41_41299


namespace problem1_correct_problem2_correct_l41_41012

noncomputable def problem1 : Real :=
  2 * Real.sqrt (2 / 3) - 3 * Real.sqrt (3 / 2) + Real.sqrt 24

theorem problem1_correct : problem1 = (7 * Real.sqrt 6) / 6 := by
  sorry

noncomputable def problem2 : Real :=
  Real.sqrt (25 / 2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2

theorem problem2_correct : problem2 = (11 * Real.sqrt 2) / 2 - 3 := by
  sorry

end problem1_correct_problem2_correct_l41_41012


namespace remainder_division_l41_41767

-- Definition of the number in terms of its components
def num : ℤ := 98 * 10^6 + 76 * 10^4 + 54 * 10^2 + 32

-- The modulus
def m : ℤ := 25

-- The given problem restated as a hypothesis and goal
theorem remainder_division : num % m = 7 :=
by
  sorry

end remainder_division_l41_41767


namespace determine_a_l41_41448

theorem determine_a (a b c : ℤ)
  (vertex_condition : ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = -3 → y = a * (x - 2) ^ 2 - 3)
  (point_condition : ∀ x : ℝ, x = 1 → ∀ y : ℝ, y = -2 → y = a * (x - 2) ^ 2 - 3) :
  a = 1 :=
by
  sorry

end determine_a_l41_41448


namespace quotient_remainder_threefold_l41_41944

theorem quotient_remainder_threefold (a b c d : ℤ)
  (h : a = b * c + d) :
  3 * a = 3 * b * c + 3 * d :=
by sorry

end quotient_remainder_threefold_l41_41944


namespace proposition_B_correct_l41_41866

theorem proposition_B_correct : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x :=
by
  sorry

end proposition_B_correct_l41_41866


namespace quadratic_roots_r_l41_41879

theorem quadratic_roots_r (a b m p r : ℚ) :
  (∀ x : ℚ, x^2 - m * x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x : ℚ, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a + 1)) →
  r = 19 / 3 :=
by
  sorry

end quadratic_roots_r_l41_41879


namespace distance_squared_l41_41079

noncomputable def circumcircle_radius (R : ℝ) : Prop := sorry
noncomputable def excircle_radius (p : ℝ) : Prop := sorry
noncomputable def distance_between_centers (d : ℝ) (R : ℝ) (p : ℝ) : Prop := sorry

theorem distance_squared (R p d : ℝ) (h1 : circumcircle_radius R) (h2 : excircle_radius p) (h3 : distance_between_centers d R p) :
  d^2 = R^2 + 2 * R * p := sorry

end distance_squared_l41_41079


namespace remainder_of_sum_modulo_9_l41_41517

theorem remainder_of_sum_modulo_9 : 
  (8230 + 8231 + 8232 + 8233 + 8234 + 8235) % 9 = 0 := by
  sorry

end remainder_of_sum_modulo_9_l41_41517


namespace equivalent_operation_l41_41972

theorem equivalent_operation (x : ℚ) :
  (x * (2/3)) / (5/6) = x * (4/5) :=
by
  -- Normal proof steps might follow here
  sorry

end equivalent_operation_l41_41972


namespace minimum_value_of_f_l41_41390

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x + 1 / x) + 1 / (x^2 + 1 / x^2)

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 3) ∧ (f 1 = 3) :=
by
  sorry

end minimum_value_of_f_l41_41390


namespace solve_for_x_l41_41033

theorem solve_for_x (x : ℝ) (h : x + 3 * x = 500 - (4 * x + 5 * x)) : x = 500 / 13 := 
by 
  sorry

end solve_for_x_l41_41033


namespace total_area_correct_l41_41389

-- Define the conditions from the problem
def side_length_small : ℕ := 2
def side_length_medium : ℕ := 4
def side_length_large : ℕ := 8

-- Define the areas of individual squares
def area_small : ℕ := side_length_small * side_length_small
def area_medium : ℕ := side_length_medium * side_length_medium
def area_large : ℕ := side_length_large * side_length_large

-- Define the additional areas as suggested by vague steps in the solution
def area_term1 : ℕ := 4 * 4 / 2 * 2
def area_term2 : ℕ := 2 * 2 / 2
def area_term3 : ℕ := (8 + 2) * 2 / 2 * 2

-- Define the total area as the sum of all calculated parts
def total_area : ℕ := area_large + (area_medium * 3) + area_small + area_term1 + area_term2 + area_term3

-- The theorem to prove total area is 150 square centimeters
theorem total_area_correct : total_area = 150 :=
by
  -- Proof goes here (steps from the solution)...
  sorry

end total_area_correct_l41_41389


namespace games_new_friends_l41_41991

-- Definitions based on the conditions
def total_games_all_friends : ℕ := 141
def games_old_friends : ℕ := 53

-- Statement of the problem
theorem games_new_friends {games_new_friends : ℕ} :
  games_new_friends = total_games_all_friends - games_old_friends :=
sorry

end games_new_friends_l41_41991


namespace cupcake_price_l41_41938

theorem cupcake_price
  (x : ℝ)
  (h1 : 5 * x + 6 * 1 + 4 * 2 + 15 * 0.6 = 33) : x = 2 :=
by
  sorry

end cupcake_price_l41_41938


namespace power_of_fraction_l41_41085

theorem power_of_fraction :
  ( (2 / 5: ℝ) ^ 7 = 128 / 78125) :=
by
  sorry

end power_of_fraction_l41_41085


namespace simple_interest_principal_l41_41170

theorem simple_interest_principal (A r t : ℝ) (ht_pos : t > 0) (hr_pos : r > 0) (hA_pos : A > 0) :
  (A = 1120) → (r = 0.08) → (t = 2.4) → ∃ (P : ℝ), abs (P - 939.60) < 0.01 :=
by
  intros hA hr ht
  -- Proof would go here
  sorry

end simple_interest_principal_l41_41170


namespace min_sum_first_n_terms_l41_41191

variable {a₁ d c : ℝ} (n : ℕ)

noncomputable def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_first_n_terms (h₁ : ∀ x, 1/3 ≤ x ∧ x ≤ 4/5 → a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0)
                              (h₂ : a₁ = -15/4 * d)
                              (h₃ : d > 0) :
                              ∃ n : ℕ, n > 0 ∧ sum_first_n_terms a₁ d n ≤ sum_first_n_terms a₁ d 4 :=
by
  use 4
  sorry

end min_sum_first_n_terms_l41_41191


namespace solve_inequality_system_l41_41418

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l41_41418


namespace complement_U_M_correct_l41_41189

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}
def complement_U_M : Set ℕ := {1, 2, 3}

theorem complement_U_M_correct : U \ M = complement_U_M :=
  by sorry

end complement_U_M_correct_l41_41189


namespace sampling_prob_equal_l41_41274

theorem sampling_prob_equal (N n : ℕ) (P_1 P_2 P_3 : ℝ)
  (H_random : ∀ i, 1 ≤ i ∧ i ≤ N → P_1 = 1 / N)
  (H_systematic : ∀ i, 1 ≤ i ∧ i ≤ N → P_2 = 1 / N)
  (H_stratified : ∀ i, 1 ≤ i ∧ i ≤ N → P_3 = 1 / N) :
  P_1 = P_2 ∧ P_2 = P_3 :=
by
  sorry

end sampling_prob_equal_l41_41274


namespace find_n_from_exponent_equation_l41_41761

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l41_41761


namespace numbers_less_than_reciprocal_l41_41280

theorem numbers_less_than_reciprocal :
  (1 / 3 < 3) ∧ (1 / 2 < 2) ∧ ¬(1 < 1) ∧ ¬(2 < 1 / 2) ∧ ¬(3 < 1 / 3) :=
by
  sorry

end numbers_less_than_reciprocal_l41_41280


namespace unique_solutions_of_system_l41_41412

theorem unique_solutions_of_system (a : ℝ) :
  (∃! (x y : ℝ), a^2 - 2 * a * x - 6 * y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  (a ∈ Set.union (Set.Ioo (-12) (-6)) (Set.union {0} (Set.Ioo 6 12))) :=
by
  sorry

end unique_solutions_of_system_l41_41412


namespace x_is_integer_l41_41873

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ a : ℤ, a = x^1960 - x^1919)
  (h2 : ∃ b : ℤ, b = x^2001 - x^1960) :
  ∃ k : ℤ, x = k :=
sorry

end x_is_integer_l41_41873


namespace plane_overtake_time_is_80_minutes_l41_41130

noncomputable def plane_overtake_time 
  (speed_a speed_b : ℝ)
  (head_start : ℝ) 
  (t : ℝ) : Prop :=
  speed_a * (t + head_start) = speed_b * t

theorem plane_overtake_time_is_80_minutes :
  plane_overtake_time 200 300 (2/3) (80 / 60)
:=
  sorry

end plane_overtake_time_is_80_minutes_l41_41130


namespace geometric_series_ratio_l41_41323

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (h2 : 0 < r ∧ r < 1) :
    (a / (1 - r)) = 81 * (ar^4 / (1 - r)) → r = 1 / 3 :=
by
  sorry

end geometric_series_ratio_l41_41323


namespace recurring_decimal_fraction_l41_41836

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l41_41836


namespace final_coordinates_of_A_l41_41946

-- Define the initial points
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (5, -5)
def C : ℝ × ℝ := (2, -4)

-- Define the translation operation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Define the rotation operation (180 degrees around a point (h, k))
def rotate180 (p : ℝ × ℝ) (h k : ℝ) : ℝ × ℝ :=
  (2 * h - p.1, 2 * k - p.2)

-- Translate point A
def A' := translate A 4 3

-- Rotate the translated point A' 180 degrees around the point (4, 0)
def A'' := rotate180 A' 4 0

-- The final coordinates of point A after transformations should be (1, -1)
theorem final_coordinates_of_A : A'' = (1, -1) :=
  sorry

end final_coordinates_of_A_l41_41946


namespace fraction_expression_of_repeating_decimal_l41_41141

theorem fraction_expression_of_repeating_decimal :
  ∃ (x : ℕ), x = 79061333 ∧ (∀ y : ℚ, y = 0.71 + 264 * (1/999900) → x / 999900 = y) :=
by
  sorry

end fraction_expression_of_repeating_decimal_l41_41141


namespace even_function_a_eq_neg1_l41_41805

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * (Real.exp x + a * Real.exp (-x))

/-- Given that the function f(x) = x(e^x + a e^{-x}) is an even function, prove that a = -1. -/
theorem even_function_a_eq_neg1 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
sorry

end even_function_a_eq_neg1_l41_41805


namespace chord_square_length_l41_41397

theorem chord_square_length
    (r1 r2 r3 L1 L2 L3 : ℝ)
    (h1 : r1 = 4) 
    (h2 : r2 = 8) 
    (h3 : r3 = 12) 
    (tangent1 : ∀ x, (L1 - x)^2 + (L2 - x)^2 = (r1 + r2)^2)
    (tangent2 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r2)^2) 
    (tangent3 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r1)^2) : L1^2 = 3584 / 9 :=
by
  sorry

end chord_square_length_l41_41397


namespace area_OPA_l41_41313

variable (x : ℝ)

def y (x : ℝ) : ℝ := -x + 6

def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, y x)

def area_triangle (O A P : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.fst * P.snd + P.fst * O.snd + O.fst * A.snd - A.snd * P.fst - P.snd * O.fst - O.snd * A.fst)

theorem area_OPA : 0 < x ∧ x < 6 → area_triangle O A (P x) = 12 - 2 * x := by
  -- proof to be provided here
  sorry


end area_OPA_l41_41313


namespace right_angle_vertex_trajectory_l41_41380

theorem right_angle_vertex_trajectory (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  let P := (x, y)
  (∃ (x y : ℝ), (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16) →
  x ≠ 2 ∧ x ≠ -2 →
  x^2 + y^2 = 4 :=
by
  intro h₁ h₂
  sorry

end right_angle_vertex_trajectory_l41_41380


namespace annual_income_of_A_l41_41496

variable (Cm : ℝ)
variable (Bm : ℝ)
variable (Am : ℝ)
variable (Aa : ℝ)

-- Given conditions
axiom h1 : Cm = 12000
axiom h2 : Bm = Cm + 0.12 * Cm
axiom h3 : (Am / Bm) = 5 / 2

-- Statement to prove
theorem annual_income_of_A : Aa = 403200 := by
  sorry

end annual_income_of_A_l41_41496


namespace cylinder_volume_l41_41651

noncomputable def volume_cylinder (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) : ℝ :=
  let ratio_r := r_cylinder / r_cone
  let ratio_h := h_cylinder / h_cone
  (3 : ℝ) * ratio_r^2 * ratio_h * V_cone

theorem cylinder_volume (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) :
    r_cylinder / r_cone = 2 / 3 →
    h_cylinder / h_cone = 4 / 3 →
    V_cone = 5.4 →
    volume_cylinder V_cone r_cylinder r_cone h_cylinder h_cone = 3.2 :=
by
  intros h1 h2 h3
  rw [volume_cylinder, h1, h2, h3]
  sorry

end cylinder_volume_l41_41651


namespace magazines_sold_l41_41704

theorem magazines_sold (total_sold : Float) (newspapers_sold : Float) (magazines_sold : Float)
  (h1 : total_sold = 425.0)
  (h2 : newspapers_sold = 275.0) :
  magazines_sold = total_sold - newspapers_sold :=
by
  sorry

#check magazines_sold

end magazines_sold_l41_41704


namespace min_value_M_proof_l41_41392

noncomputable def min_value_M (a b c d e f g M : ℝ) : Prop :=
  (∀ (a b c d e f g : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0 ∧ 
    a + b + c + d + e + f + g = 1 ∧ 
    M = max (max (max (max (a + b + c) (b + c + d)) (c + d + e)) (d + e + f)) (e + f + g)
  → M ≥ (1 / 3))

theorem min_value_M_proof : min_value_M a b c d e f g M :=
by
  sorry

end min_value_M_proof_l41_41392


namespace sum_of_cubes_ratio_l41_41441

theorem sum_of_cubes_ratio (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) (h2 : d + e + f = 0) :
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = (a * b * c) / (d * e * f) := 
by 
  sorry

end sum_of_cubes_ratio_l41_41441


namespace slope_intercept_form_correct_l41_41207

theorem slope_intercept_form_correct:
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) → (∃ m b, y = m * x + b ∧ m = 2 ∧ b = -10) :=
by
  intro x y h
  use 2, -10
  sorry

end slope_intercept_form_correct_l41_41207


namespace stratified_sampling_total_students_sampled_l41_41905

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l41_41905


namespace dad_strawberries_final_weight_l41_41887

variable {M D : ℕ}

theorem dad_strawberries_final_weight :
  M + D = 22 →
  36 - M + 30 + D = D' →
  D' = 46 :=
by
  intros h h1
  sorry

end dad_strawberries_final_weight_l41_41887


namespace arrow_reading_l41_41792

-- Define the interval and values within it
def in_range (x : ℝ) : Prop := 9.75 ≤ x ∧ x ≤ 10.00
def closer_to_990 (x : ℝ) : Prop := |x - 9.90| < |x - 9.875|

-- The main theorem statement expressing the problem
theorem arrow_reading (x : ℝ) (hx1 : in_range x) (hx2 : closer_to_990 x) : x = 9.90 :=
by sorry

end arrow_reading_l41_41792


namespace prob_diff_colors_correct_l41_41840

noncomputable def total_outcomes : ℕ :=
  let balls_pocket1 := 2 + 3 + 5
  let balls_pocket2 := 2 + 4 + 4
  balls_pocket1 * balls_pocket2

noncomputable def favorable_outcomes_same_color : ℕ :=
  let white_balls := 2 * 2
  let red_balls := 3 * 4
  let yellow_balls := 5 * 4
  white_balls + red_balls + yellow_balls

noncomputable def prob_same_color : ℚ :=
  favorable_outcomes_same_color / total_outcomes

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_diff_colors_correct :
  prob_different_color = 16 / 25 :=
by sorry

end prob_diff_colors_correct_l41_41840


namespace find_k_l41_41758

-- Definitions
variable (m n k : ℝ)

-- Given conditions
def on_line_1 : Prop := m = 2 * n + 5
def on_line_2 : Prop := (m + 5) = 2 * (n + k) + 5

-- Desired conclusion
theorem find_k (h1 : on_line_1 m n) (h2 : on_line_2 m n k) : k = 2.5 :=
sorry

end find_k_l41_41758


namespace keychain_arrangement_l41_41693

open Function

theorem keychain_arrangement (keys : Finset ℕ) (h : keys.card = 7)
  (house_key car_key office_key : ℕ) (hmem : house_key ∈ keys)
  (cmem : car_key ∈ keys) (omem : office_key ∈ keys) : 
  ∃ n : ℕ, n = 72 :=
by
  sorry

end keychain_arrangement_l41_41693


namespace susan_spending_ratio_l41_41308

theorem susan_spending_ratio (initial_amount clothes_spent books_left books_spent left_after_clothes gcd_ratio : ℤ)
  (h1 : initial_amount = 600)
  (h2 : clothes_spent = initial_amount / 2)
  (h3 : left_after_clothes = initial_amount - clothes_spent)
  (h4 : books_left = 150)
  (h5 : books_spent = left_after_clothes - books_left)
  (h6 : gcd books_spent left_after_clothes = 150)
  (h7 : books_spent / gcd_ratio = 1)
  (h8 : left_after_clothes / gcd_ratio = 2) :
  books_spent / gcd books_spent left_after_clothes = 1 ∧ left_after_clothes / gcd books_spent left_after_clothes = 2 :=
sorry

end susan_spending_ratio_l41_41308


namespace find_c_value_l41_41519

theorem find_c_value (c : ℝ)
  (h : 4 * (3.6 * 0.48 * c / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) :
  c = 2.5 :=
by sorry

end find_c_value_l41_41519


namespace b100_mod_50_l41_41579

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b100_mod_50 : b 100 % 50 = 2 := by
  sorry

end b100_mod_50_l41_41579


namespace age_of_student_who_left_l41_41275

variables
  (avg_age_students : ℝ)
  (num_students_before : ℕ)
  (num_students_after : ℕ)
  (age_teacher : ℝ)
  (new_avg_age_class : ℝ)

theorem age_of_student_who_left
  (h1 : avg_age_students = 14)
  (h2 : num_students_before = 45)
  (h3 : num_students_after = 44)
  (h4 : age_teacher = 45)
  (h5 : new_avg_age_class = 14.66)
: ∃ (age_student_left : ℝ), abs (age_student_left - 15.3) < 0.1 :=
sorry

end age_of_student_who_left_l41_41275


namespace distance_between_trees_l41_41196

theorem distance_between_trees (num_trees: ℕ) (total_length: ℕ) (trees_at_end: ℕ) 
(h1: num_trees = 26) (h2: total_length = 300) (h3: trees_at_end = 2) :
  total_length / (num_trees - 1) = 12 :=
by sorry

end distance_between_trees_l41_41196


namespace manager_salary_l41_41669

theorem manager_salary (n : ℕ) (avg_salary : ℕ) (increment : ℕ) (new_avg_salary : ℕ) (new_total_salary : ℕ) (old_total_salary : ℕ) :
  n = 20 →
  avg_salary = 1500 →
  increment = 1000 →
  new_avg_salary = avg_salary + increment →
  old_total_salary = n * avg_salary →
  new_total_salary = (n + 1) * new_avg_salary →
  (new_total_salary - old_total_salary) = 22500 :=
by
  intros h_n h_avg_salary h_increment h_new_avg_salary h_old_total_salary h_new_total_salary
  sorry

end manager_salary_l41_41669


namespace find_initial_jellybeans_l41_41294

-- Definitions of the initial conditions
def jellybeans_initial (x : ℝ) (days : ℕ) (remaining : ℝ) := 
  days = 4 ∧ remaining = 48 ∧ (0.7 ^ days) * x = remaining

-- The theorem to prove
theorem find_initial_jellybeans (x : ℝ) : 
  jellybeans_initial x 4 48 → x = 200 :=
sorry

end find_initial_jellybeans_l41_41294


namespace walters_exceptional_days_l41_41739

variable (b w : ℕ)
variable (days_total dollars_total : ℕ)
variable (normal_earn exceptional_earn : ℕ)
variable (at_least_exceptional_days : ℕ)

-- Conditions
def conditions : Prop :=
  days_total = 15 ∧
  dollars_total = 70 ∧
  normal_earn = 4 ∧
  exceptional_earn = 6 ∧
  at_least_exceptional_days = 5 ∧
  b + w = days_total ∧
  normal_earn * b + exceptional_earn * w = dollars_total ∧
  w ≥ at_least_exceptional_days

-- Theorem to prove the number of exceptional days is 5
theorem walters_exceptional_days (h : conditions b w days_total dollars_total normal_earn exceptional_earn at_least_exceptional_days) : w = 5 :=
sorry

end walters_exceptional_days_l41_41739


namespace lines_perpendicular_slope_l41_41749

theorem lines_perpendicular_slope (k : ℝ) :
  (∀ (x : ℝ), k * 2 = -1) → k = (-1:ℝ)/2 :=
by
  sorry

end lines_perpendicular_slope_l41_41749


namespace hexagon_area_is_32_l41_41685

noncomputable def area_of_hexagon : ℝ := 
  let p0 : ℝ × ℝ := (0, 0)
  let p1 : ℝ × ℝ := (2, 4)
  let p2 : ℝ × ℝ := (5, 4)
  let p3 : ℝ × ℝ := (7, 0)
  let p4 : ℝ × ℝ := (5, -4)
  let p5 : ℝ × ℝ := (2, -4)
  -- Triangle 1: p0, p1, p2
  let area_tri1 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 2: p2, p3, p4
  let area_tri2 := 1 / 2 * (8 : ℝ) * (2 : ℝ)
  -- Triangle 3: p4, p5, p0
  let area_tri3 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 4: p1, p2, p5
  let area_tri4 := 1 / 2 * (8 : ℝ) * (3 : ℝ)
  area_tri1 + area_tri2 + area_tri3 + area_tri4

theorem hexagon_area_is_32 : area_of_hexagon = 32 := 
by
  sorry

end hexagon_area_is_32_l41_41685


namespace intersection_is_open_interval_l41_41143

open Set
open Real

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_is_open_interval :
  M ∩ N = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_is_open_interval_l41_41143


namespace breadth_is_13_l41_41533

variable (b l : ℕ) (breadth : ℕ)

/-
We have the following conditions:
1. The area of the rectangular plot is 23 times its breadth.
2. The difference between the length and the breadth is 10 metres.
We need to prove that the breadth of the plot is 13 metres.
-/

theorem breadth_is_13
  (h1 : l * b = 23 * b)
  (h2 : l - b = 10) :
  b = 13 := 
sorry

end breadth_is_13_l41_41533


namespace susan_cars_fewer_than_carol_l41_41047

theorem susan_cars_fewer_than_carol 
  (Lindsey_cars Carol_cars Susan_cars Cathy_cars : ℕ)
  (h1 : Lindsey_cars = Cathy_cars + 4)
  (h2 : Susan_cars < Carol_cars)
  (h3 : Carol_cars = 2 * Cathy_cars)
  (h4 : Cathy_cars = 5)
  (h5 : Cathy_cars + Carol_cars + Lindsey_cars + Susan_cars = 32) :
  Carol_cars - Susan_cars = 2 :=
sorry

end susan_cars_fewer_than_carol_l41_41047


namespace neg_of_proposition_l41_41954

variable (a : ℝ)

def proposition := ∀ x : ℝ, 0 < a^x

theorem neg_of_proposition (h₀ : 0 < a) (h₁ : a ≠ 1) : ¬proposition a ↔ ∃ x : ℝ, a^x ≤ 0 :=
by
  sorry

end neg_of_proposition_l41_41954


namespace max_value_on_ellipse_l41_41238

theorem max_value_on_ellipse (b : ℝ) (hb : b > 0) :
  ∃ (M : ℝ), 
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / b^2 = 1) → x^2 + 2 * y ≤ M) ∧
    ((b ≤ 4 → M = b^2 / 4 + 4) ∧ (b > 4 → M = 2 * b)) :=
  sorry

end max_value_on_ellipse_l41_41238


namespace multiple_is_eight_l41_41364

theorem multiple_is_eight (m : ℝ) (h : 17 = m * 2.625 - 4) : m = 8 :=
by
  sorry

end multiple_is_eight_l41_41364


namespace evaluate_expr_l41_41453

-- Define the imaginary unit i
def i := Complex.I

-- Define the expressions for the proof
def expr1 := (1 + 2 * i) * i ^ 3
def expr2 := 2 * i ^ 2

-- The main statement we need to prove
theorem evaluate_expr : expr1 + expr2 = -i :=
by 
  sorry

end evaluate_expr_l41_41453


namespace lcm_36_65_l41_41915

-- Definitions based on conditions
def number1 : ℕ := 36
def number2 : ℕ := 65

-- The prime factorization conditions can be implied through deriving LCM hence added as comments to clarify the conditions.
-- 36 = 2^2 * 3^2
-- 65 = 5 * 13

-- Theorem statement that the LCM of number1 and number2 is 2340
theorem lcm_36_65 : Nat.lcm number1 number2 = 2340 := 
by 
  sorry

end lcm_36_65_l41_41915


namespace minimum_value_expression_l41_41721

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end minimum_value_expression_l41_41721


namespace area_circle_l41_41672

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end area_circle_l41_41672


namespace probability_of_at_most_one_white_ball_l41_41432

open Nat

-- Definitions based on conditions in a)
def black_balls : ℕ := 10
def red_balls : ℕ := 12
def white_balls : ℕ := 3
def total_balls : ℕ := black_balls + red_balls + white_balls
def select_balls : ℕ := 3

-- The combinatorial function C(n, k) as defined in combinatorics
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Defining the expression and correct answer
def expr : ℚ := (C white_balls 1 * C (black_balls + red_balls) 2 + C (black_balls + red_balls) 3 : ℚ) / (C total_balls 3 : ℚ)
def correct_answer : ℚ := (C white_balls 0 * C (black_balls + red_balls) 3 + C white_balls 1 * C (black_balls + red_balls) 2 : ℚ) / (C total_balls 3 : ℚ)

-- Lean 4 theorem statement
theorem probability_of_at_most_one_white_ball :
  expr = correct_answer := sorry

end probability_of_at_most_one_white_ball_l41_41432


namespace probability_three_same_color_is_one_seventeenth_l41_41777

def standard_deck := {cards : Finset ℕ // cards.card = 52 ∧ ∃ reds blacks, reds.card = 26 ∧ blacks.card = 26 ∧ (reds ∪ blacks = cards)}

def num_ways_to_pick_3_same_color : ℕ :=
  (26 * 25 * 24) + (26 * 25 * 24)

def total_ways_to_pick_3 : ℕ :=
  52 * 51 * 50

def probability_top_three_same_color := (num_ways_to_pick_3_same_color / total_ways_to_pick_3 : ℚ)

theorem probability_three_same_color_is_one_seventeenth :
  probability_top_three_same_color = (1 / 17 : ℚ) := by sorry

end probability_three_same_color_is_one_seventeenth_l41_41777


namespace min_value_of_x_plus_y_l41_41789

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l41_41789


namespace volunteers_correct_l41_41435

-- Definitions of given conditions and the required result
def sheets_per_member : ℕ := 10
def cookies_per_sheet : ℕ := 16
def total_cookies : ℕ := 16000

-- Number of members who volunteered
def members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

-- Proof statement
theorem volunteers_correct :
  members = 100 :=
sorry

end volunteers_correct_l41_41435


namespace derivative_at_0_l41_41907

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else x^2 * Real.exp (|x|) * Real.sin (1 / x^2)

theorem derivative_at_0 : deriv f 0 = 0 := by
  sorry

end derivative_at_0_l41_41907


namespace no_equal_partition_product_l41_41884

theorem no_equal_partition_product (n : ℕ) (h : n > 1) : 
  ¬ ∃ A B : Finset ℕ, 
    (A ∪ B = (Finset.range n).erase 0 ∧ A ∩ B = ∅ ∧ (A ≠ ∅) ∧ (B ≠ ∅) 
    ∧ A.prod id = B.prod id) := 
sorry

end no_equal_partition_product_l41_41884


namespace arithmetic_sequence_fifth_term_l41_41131

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (a6 : a 6 = -3) 
  (S6 : S 6 = 12)
  (h_sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 1 - a 0)) / 2)
  : a 5 = -1 :=
sorry

end arithmetic_sequence_fifth_term_l41_41131


namespace intersection_M_N_eq_segment_l41_41278

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq_segment : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_segment_l41_41278


namespace relationship_between_abc_l41_41256

noncomputable def a : Real := Real.sqrt 1.2
noncomputable def b : Real := Real.exp 0.1
noncomputable def c : Real := 1 + Real.log 1.1

theorem relationship_between_abc : b > a ∧ a > c :=
by {
  -- a = sqrt(1.2)
  -- b = exp(0.1)
  -- c = 1 + log(1.1)
  -- We need to prove: b > a > c
  sorry
}

end relationship_between_abc_l41_41256


namespace conditional_probability_l41_41987

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end conditional_probability_l41_41987


namespace verify_a_l41_41765

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l41_41765


namespace divisible_by_3_l41_41691

theorem divisible_by_3 :
  ∃ n : ℕ, (5 + 2 + n + 4 + 8) % 3 = 0 ∧ n = 2 := 
by
  sorry

end divisible_by_3_l41_41691


namespace scientific_notation_of_area_l41_41644

theorem scientific_notation_of_area : 2720000 = 2.72 * 10^6 :=
by
  sorry

end scientific_notation_of_area_l41_41644


namespace sheila_hourly_wage_l41_41284

-- Sheila works 8 hours per day on Monday, Wednesday, and Friday
-- Sheila works 6 hours per day on Tuesday and Thursday
-- Sheila does not work on Saturday and Sunday
-- Sheila earns $288 per week

def hours_worked (monday_wednesday_friday_hours : Nat) (tuesday_thursday_hours : Nat) : Nat :=
  (monday_wednesday_friday_hours * 3) + (tuesday_thursday_hours * 2)

def weekly_earnings : Nat := 288
def total_hours_worked : Nat := hours_worked 8 6
def hourly_wage : Nat := weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage = 8 := by
  -- Proof (omitted)
  sorry

end sheila_hourly_wage_l41_41284


namespace magnitude_of_power_l41_41473

-- Given conditions
def z : ℂ := 3 + 2 * Complex.I
def n : ℕ := 6

-- Mathematical statement to prove
theorem magnitude_of_power :
  Complex.abs (z ^ n) = 2197 :=
by
  sorry

end magnitude_of_power_l41_41473


namespace maximum_profit_l41_41133

/-- 
Given:
- The fixed cost is 3000 (in thousand yuan).
- The revenue per hundred vehicles is 500 (in thousand yuan).
- The additional cost y is defined as follows:
  - y = 10*x^2 + 100*x for 0 < x < 40
  - y = 501*x + 10000/x - 4500 for x ≥ 40
  
Prove:
1. The profit S(x) (in thousand yuan) in 2020 is:
   - S(x) = -10*x^2 + 400*x - 3000 for 0 < x < 40
   - S(x) = 1500 - x - 10000/x for x ≥ 40
2. The production volume x (in hundreds of vehicles) to achieve the maximum profit is 100,
   and the maximum profit is 1300 (in thousand yuan).
-/
noncomputable def profit_function (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 40) then
    -10 * x^2 + 400 * x - 3000
  else if (x ≥ 40) then
    1500 - x - 10000 / x
  else
    0 -- Undefined for other values, though our x will always be positive in our case

theorem maximum_profit : ∃ x : ℝ, 0 < x ∧ 
  (profit_function x = 1300 ∧ x = 100) ∧
  ∀ y, 0 < y → profit_function y ≤ 1300 :=
sorry

end maximum_profit_l41_41133


namespace cubes_sum_l41_41037

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 8) (h2 : a * b + a * c + b * c = 9) (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 :=
by
  sorry

end cubes_sum_l41_41037


namespace shop_makes_off_each_jersey_l41_41318

theorem shop_makes_off_each_jersey :
  ∀ (T : ℝ) (jersey_earnings : ℝ),
  (T = 25) →
  (jersey_earnings = T + 90) →
  jersey_earnings = 115 := by
  intros T jersey_earnings ht hj
  sorry

end shop_makes_off_each_jersey_l41_41318


namespace rotated_and_shifted_line_eq_l41_41755

theorem rotated_and_shifted_line_eq :
  let rotate_line_90 (x y : ℝ) := ( -y, x )
  let shift_right (x y : ℝ) := (x + 1, y)
  ∃ (new_a new_b new_c : ℝ), 
  (∀ (x y : ℝ), (y = 3 * x → x * new_a + y * new_b + new_c = 0)) ∧ 
  (new_a = 1) ∧ (new_b = 3) ∧ (new_c = -1) := by
  sorry

end rotated_and_shifted_line_eq_l41_41755


namespace total_soccer_balls_purchased_l41_41918

theorem total_soccer_balls_purchased : 
  (∃ (x : ℝ), 
    800 / x * 2 = 1560 / (x - 2)) → 
  (800 / x + 1560 / (x - 2) = 30) :=
by
  sorry

end total_soccer_balls_purchased_l41_41918


namespace percent_increase_l41_41710

theorem percent_increase (P x : ℝ) (h1 : P + x/100 * P - 0.2 * (P + x/100 * P) = P) : x = 25 :=
by
  sorry

end percent_increase_l41_41710


namespace find_a3_in_arith_geo_seq_l41_41515

theorem find_a3_in_arith_geo_seq
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : S 6 / S 3 = -19 / 8)
  (h2 : a 4 - a 2 = -15 / 8) :
  a 3 = 9 / 4 :=
sorry

end find_a3_in_arith_geo_seq_l41_41515


namespace IncorrectStatement_l41_41513

-- Definitions of the events
def EventA (planeShot : ℕ → Prop) : Prop := planeShot 1 ∧ planeShot 2
def EventB (planeShot : ℕ → Prop) : Prop := ¬planeShot 1 ∧ ¬planeShot 2
def EventC (planeShot : ℕ → Prop) : Prop := (planeShot 1 ∧ ¬planeShot 2) ∨ (¬planeShot 1 ∧ planeShot 2)
def EventD (planeShot : ℕ → Prop) : Prop := planeShot 1 ∨ planeShot 2

-- Theorem statement to be proved (negation of the incorrect statement)
theorem IncorrectStatement (planeShot : ℕ → Prop) :
  ¬((EventA planeShot ∨ EventC planeShot) = (EventB planeShot ∨ EventD planeShot)) :=
by
  sorry

end IncorrectStatement_l41_41513


namespace min_isosceles_triangle_area_l41_41661

theorem min_isosceles_triangle_area 
  (x y n : ℕ)
  (h1 : 2 * x * y = 7 * n^2)
  (h2 : ∃ m k, m = n / 2 ∧ k = 2 * m) 
  (h3 : n % 3 = 0) : 
  x = 4 * n / 3 ∧ y = n / 3 ∧ 
  ∃ A, A = 21 / 4 := 
sorry

end min_isosceles_triangle_area_l41_41661


namespace algebra_expression_value_l41_41108

theorem algebra_expression_value (a : ℤ) (h : (2023 - a) ^ 2 + (a - 2022) ^ 2 = 7) :
  (2023 - a) * (a - 2022) = -3 := 
sorry

end algebra_expression_value_l41_41108


namespace juan_speed_l41_41822

theorem juan_speed (J : ℝ) :
  (∀ (time : ℝ) (distance : ℝ) (peter_speed : ℝ),
    time = 1.5 →
    distance = 19.5 →
    peter_speed = 5 →
    distance = J * time + peter_speed * time) →
  J = 8 :=
by
  intro h
  sorry

end juan_speed_l41_41822


namespace find_a4_b4_l41_41411

theorem find_a4_b4
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end find_a4_b4_l41_41411


namespace randi_has_6_more_nickels_than_peter_l41_41203

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end randi_has_6_more_nickels_than_peter_l41_41203


namespace find_pairs_l41_41157

theorem find_pairs (n k : ℕ) (h1 : (10^(k-1) ≤ n^n) ∧ (n^n < 10^k)) (h2 : (10^(n-1) ≤ k^k) ∧ (k^k < 10^n)) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) := by
  sorry

end find_pairs_l41_41157


namespace average_percentage_of_15_students_l41_41745

open Real

theorem average_percentage_of_15_students :
  ∀ (x : ℝ),
  (15 + 10 = 25) →
  (10 * 90 = 900) →
  (25 * 84 = 2100) →
  (15 * x + 900 = 2100) →
  x = 80 :=
by
  intro x h_sum h_10_avg h_25_avg h_total
  sorry

end average_percentage_of_15_students_l41_41745


namespace solve_for_x_l41_41041

theorem solve_for_x : ∀ (x : ℕ), (y = 2 / (4 * x + 2)) → (y = 1 / 2) → (x = 1/2) :=
by
  sorry

end solve_for_x_l41_41041


namespace kids_go_to_camp_l41_41897

variable (total_kids staying_home going_to_camp : ℕ)

theorem kids_go_to_camp (h1 : total_kids = 313473) (h2 : staying_home = 274865) (h3 : going_to_camp = total_kids - staying_home) :
  going_to_camp = 38608 :=
by
  sorry

end kids_go_to_camp_l41_41897


namespace find_x_value_l41_41011

theorem find_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := 
by 
  sorry

end find_x_value_l41_41011


namespace cube_edge_length_l41_41368

theorem cube_edge_length (total_edge_length : ℕ) (num_edges : ℕ) (h1 : total_edge_length = 108) (h2 : num_edges = 12) : total_edge_length / num_edges = 9 := by 
  -- additional formal mathematical steps can follow here
  sorry

end cube_edge_length_l41_41368


namespace tangent_line_at_1_2_l41_41358

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

def tangent_eq (x y : ℝ) : Prop := y = 2 * x

theorem tangent_line_at_1_2 : tangent_eq 1 2 :=
by
  have f_1 := 1
  have f'_1 := 2
  sorry

end tangent_line_at_1_2_l41_41358


namespace quadruplets_sets_l41_41706

theorem quadruplets_sets (a b c babies: ℕ) (h1: 2 * a + 3 * b + 4 * c = 1200) (h2: b = 5 * c) (h3: a = 2 * b) :
  4 * c = 123 :=
by
  sorry

end quadruplets_sets_l41_41706


namespace johns_number_l41_41575

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number_l41_41575


namespace paperclips_volume_75_l41_41718

noncomputable def paperclips (v : ℝ) : ℝ := 60 / Real.sqrt 27 * Real.sqrt v

theorem paperclips_volume_75 :
  paperclips 75 = 100 :=
by
  sorry

end paperclips_volume_75_l41_41718


namespace total_preparation_time_l41_41554

theorem total_preparation_time
    (minutes_per_game : ℕ)
    (number_of_games : ℕ)
    (h1 : minutes_per_game = 10)
    (h2 : number_of_games = 15) :
    minutes_per_game * number_of_games = 150 :=
by
  -- Lean 4 proof goes here
  sorry

end total_preparation_time_l41_41554


namespace vivians_mail_in_august_l41_41132

-- Definitions based on the conditions provided
def mail_july : ℕ := 40
def business_days_august : ℕ := 22
def weekend_days_august : ℕ := 9

-- Lean 4 statement to prove the equivalent proof problem
theorem vivians_mail_in_august :
  let mail_business_days := 2 * mail_july
  let total_mail_business_days := business_days_august * mail_business_days
  let mail_weekend_days := mail_july / 2
  let total_mail_weekend_days := weekend_days_august * mail_weekend_days
  total_mail_business_days + total_mail_weekend_days = 1940 := by
  sorry

end vivians_mail_in_august_l41_41132


namespace kyungsoo_came_second_l41_41815

theorem kyungsoo_came_second
  (kyungsoo_jump : ℝ) (younghee_jump : ℝ) (jinju_jump : ℝ) (chanho_jump : ℝ)
  (h_kyungsoo : kyungsoo_jump = 2.3)
  (h_younghee : younghee_jump = 0.9)
  (h_jinju : jinju_jump = 1.8)
  (h_chanho : chanho_jump = 2.5) :
  kyungsoo_jump = 2.3 := 
by
  sorry

end kyungsoo_came_second_l41_41815


namespace math_problem_l41_41947

variable {R : Type} [LinearOrderedField R]

theorem math_problem
  (a b : R) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
by
  sorry

end math_problem_l41_41947


namespace smallest_integer_to_make_multiple_of_five_l41_41053

theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k: ℕ, 0 < k ∧ (726 + k) % 5 = 0 ∧ k = 4 := 
by
  use 4
  sorry

end smallest_integer_to_make_multiple_of_five_l41_41053


namespace second_person_more_heads_probability_l41_41474

noncomputable def coin_flip_probability (n m : ℕ) : ℚ :=
  if n < m then 1 / 2 else 0

theorem second_person_more_heads_probability :
  coin_flip_probability 10 11 = 1 / 2 :=
by
  sorry

end second_person_more_heads_probability_l41_41474


namespace rectangle_area_invariant_l41_41850

theorem rectangle_area_invariant (l w : ℝ) (A : ℝ) 
  (h0 : A = l * w)
  (h1 : A = (l + 3) * (w - 1))
  (h2 : A = (l - 1.5) * (w + 2)) :
  A = 13.5 :=
by
  sorry

end rectangle_area_invariant_l41_41850


namespace travel_distance_l41_41941

variables (speed time : ℕ) (distance : ℕ)

theorem travel_distance (hspeed : speed = 75) (htime : time = 4) : distance = speed * time → distance = 300 :=
by
  intros hdist
  rw [hspeed, htime] at hdist
  simp at hdist
  assumption

end travel_distance_l41_41941


namespace probability_of_both_selected_l41_41960

noncomputable def ramSelectionProbability : ℚ := 1 / 7
noncomputable def raviSelectionProbability : ℚ := 1 / 5

theorem probability_of_both_selected : 
  ramSelectionProbability * raviSelectionProbability = 1 / 35 :=
by sorry

end probability_of_both_selected_l41_41960


namespace reflection_point_sum_l41_41128

theorem reflection_point_sum (m b : ℝ) (H : ∀ x y : ℝ, (1, 2) = (x, y) ∨ (7, 6) = (x, y) → 
    y = m * x + b) : m + b = 8.5 := by
  sorry

end reflection_point_sum_l41_41128
