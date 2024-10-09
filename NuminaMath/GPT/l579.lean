import Mathlib

namespace quadratic_fraction_equality_l579_57970

theorem quadratic_fraction_equality (r : ℝ) (h1 : r ≠ 4) (h2 : r ≠ 6) (h3 : r ≠ 5) 
(h4 : r ≠ -4) (h5 : r ≠ -3): 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) →
  r = -7/4 :=
by {
  sorry
}

end quadratic_fraction_equality_l579_57970


namespace binary_div_mul_l579_57984

-- Define the binary numbers
def a : ℕ := 0b101110
def b : ℕ := 0b110100
def c : ℕ := 0b110

-- Statement to prove the given problem
theorem binary_div_mul : (a * b) / c = 0b101011100 := by
  -- Skipping the proof
  sorry

end binary_div_mul_l579_57984


namespace benny_final_comic_books_l579_57982

-- Define the initial number of comic books
def initial_comic_books : ℕ := 22

-- Define the comic books sold (half of the initial)
def comic_books_sold : ℕ := initial_comic_books / 2

-- Define the comic books left after selling half
def comic_books_left_after_sale : ℕ := initial_comic_books - comic_books_sold

-- Define the number of comic books bought
def comic_books_bought : ℕ := 6

-- Define the final number of comic books
def final_comic_books : ℕ := comic_books_left_after_sale + comic_books_bought

-- Statement to prove that Benny has 17 comic books at the end
theorem benny_final_comic_books : final_comic_books = 17 := by
  sorry

end benny_final_comic_books_l579_57982


namespace cornbread_pieces_count_l579_57912

def cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) : ℕ := 
  (pan_length * pan_width) / (piece_length * piece_width)

theorem cornbread_pieces_count :
  cornbread_pieces 24 20 3 3 = 53 :=
by
  -- The definitions and the equivalence transformation tell us that this is true
  sorry

end cornbread_pieces_count_l579_57912


namespace gasoline_price_increase_l579_57964

theorem gasoline_price_increase
  (P Q : ℝ) -- Prices and quantities
  (x : ℝ) -- The percentage increase in price
  (h1 : (P * (1 + x / 100)) * (Q * 0.95) = P * Q * 1.14) -- Given condition
  : x = 20 := 
sorry

end gasoline_price_increase_l579_57964


namespace sara_quarters_l579_57922

theorem sara_quarters (initial_quarters : ℕ) (additional_quarters : ℕ) (total_quarters : ℕ) 
    (h1 : initial_quarters = 21) 
    (h2 : additional_quarters = 49) 
    (h3 : total_quarters = initial_quarters + additional_quarters) : 
    total_quarters = 70 :=
sorry

end sara_quarters_l579_57922


namespace coeff_x3_product_l579_57998

open Polynomial

noncomputable def poly1 := (C 3 * X ^ 3) + (C 2 * X ^ 2) + (C 4 * X) + (C 5)
noncomputable def poly2 := (C 4 * X ^ 3) + (C 6 * X ^ 2) + (C 5 * X) + (C 2)

theorem coeff_x3_product : coeff (poly1 * poly2) 3 = 10 := by
  sorry

end coeff_x3_product_l579_57998


namespace max_mondays_in_51_days_l579_57997

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l579_57997


namespace savings_calculation_l579_57915

theorem savings_calculation (income expenditure : ℝ) (h_ratio : income = 5 / 4 * expenditure) (h_income : income = 19000) :
  income - expenditure = 3800 := 
by
  -- The solution will be filled in here,
  -- showing the calculus automatically.
  sorry

end savings_calculation_l579_57915


namespace positive_number_percentage_of_itself_is_9_l579_57900

theorem positive_number_percentage_of_itself_is_9 (x : ℝ) (hx_pos : 0 < x) (h_condition : 0.01 * x^2 = 9) : x = 30 :=
by
  sorry

end positive_number_percentage_of_itself_is_9_l579_57900


namespace color_swap_rectangle_l579_57946

theorem color_swap_rectangle 
  (n : ℕ) 
  (square_size : ℕ := 2*n - 1) 
  (colors : Finset ℕ := Finset.range n) 
  (vertex_colors : Fin (square_size + 1) × Fin (square_size + 1) → ℕ) 
  (h_vertex_colors : ∀ v, vertex_colors v ∈ colors) :
  ∃ row, ∃ (v₁ v₂ : Fin (square_size + 1) × Fin (square_size + 1)),
    (v₁.1 = row ∧ v₂.1 = row ∧ v₁ ≠ v₂ ∧
    (∃ r₀ r₁ r₂, r₀ ≠ r₁ ∧ r₁ ≠ r₂ ∧ r₂ ≠ r₀ ∧
    vertex_colors v₁ = vertex_colors (r₀, v₁.2) ∧
    vertex_colors v₂ = vertex_colors (r₀, v₂.2) ∧
    vertex_colors (r₁, v₁.2) = vertex_colors (r₂, v₂.2))) := 
sorry

end color_swap_rectangle_l579_57946


namespace daffodil_bulb_cost_l579_57995

theorem daffodil_bulb_cost :
  let total_bulbs := 55
  let crocus_cost := 0.35
  let total_budget := 29.15
  let num_crocus_bulbs := 22
  let total_crocus_cost := num_crocus_bulbs * crocus_cost
  let remaining_budget := total_budget - total_crocus_cost
  let num_daffodil_bulbs := total_bulbs - num_crocus_bulbs
  remaining_budget / num_daffodil_bulbs = 0.65 := 
by
  -- proof to be filled in
  sorry

end daffodil_bulb_cost_l579_57995


namespace correct_propositions_l579_57920

-- Definitions of parallel and perpendicular
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Main theorem
theorem correct_propositions (m n α β γ : Type) :
  ( (parallel m α ∧ parallel n β ∧ parallel α β → parallel m n) ∧
    (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ∧
    (perpendicular α γ ∧ perpendicular β γ → parallel α β) ) →
  ( (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ) :=
  sorry

end correct_propositions_l579_57920


namespace other_root_of_quadratic_l579_57932

theorem other_root_of_quadratic (a b : ℝ) (h : (1:ℝ) = 1) (h_root : (1:ℝ) ^ 2 + a * (1:ℝ) + 2 = 0): b = 2 :=
by
  sorry

end other_root_of_quadratic_l579_57932


namespace simplify_expression_l579_57987

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-2 / 3 : ℝ) = 49 :=
by 
  sorry

end simplify_expression_l579_57987


namespace schedule_arrangements_l579_57986

-- Define the initial setup of the problem
def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def periods_morning : List String := ["P1", "P2", "P3", "P4"]
def periods_afternoon : List String := ["P5", "P6", "P7"]

-- Define the constraints
def are_consecutive (subj1 subj2 : String) : Bool := 
  (subj1 = "Chinese" ∧ subj2 = "Mathematics") ∨ 
  (subj1 = "Mathematics" ∧ subj2 = "Chinese")

def can_schedule_max_one_period (subject : String) : Bool :=
  subject = "English" ∨ subject = "Physics" ∨ subject = "Chemistry" ∨ subject = "Biology"

-- Define the math problem as a proof in Lean
theorem schedule_arrangements : 
  ∃ n : Nat, n = 336 :=
by
  -- The detailed proof steps would go here
  sorry

end schedule_arrangements_l579_57986


namespace angle_A_range_l579_57996

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem angle_A_range (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_strict_inc : strictly_increasing f {x | 0 < x})
  (h_f_half : f (1 / 2) = 0)
  (A : ℝ)
  (h_cos_A : f (Real.cos A) < 0) :
  (π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π) :=
by
  sorry

end angle_A_range_l579_57996


namespace computation_l579_57961

theorem computation : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end computation_l579_57961


namespace river_flow_volume_l579_57975

/-- Given a river depth of 7 meters, width of 75 meters, 
and flow rate of 4 kilometers per hour,
the volume of water running into the sea per minute 
is 35,001.75 cubic meters. -/
theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (rate_kmph : ℝ)
  (depth_val : depth = 7)
  (width_val : width = 75)
  (rate_val : rate_kmph = 4) :
  ( width * depth * (rate_kmph * 1000 / 60) ) = 35001.75 :=
by
  rw [depth_val, width_val, rate_val]
  sorry

end river_flow_volume_l579_57975


namespace area_below_line_l579_57983

noncomputable def circle_eqn (x y : ℝ) := 
  x^2 + 2 * x + (y^2 - 6 * y) + 50 = 0

noncomputable def line_eqn (x y : ℝ) := 
  y = x + 1

theorem area_below_line : 
  (∃ (x y : ℝ), circle_eqn x y ∧ y < x + 1) →
  ∃ (a : ℝ), a = 20 * π :=
by
  sorry

end area_below_line_l579_57983


namespace count_valid_triangles_l579_57931

def triangle_area (a b c : ℕ) : ℕ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c < 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

theorem count_valid_triangles : { n : ℕ // n = 24 } :=
  sorry

end count_valid_triangles_l579_57931


namespace completing_square_solution_l579_57937

theorem completing_square_solution (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
sorry

end completing_square_solution_l579_57937


namespace profit_difference_l579_57904

variable (P : ℕ) -- P is the total profit
variable (r1 r2 : ℚ) -- r1 and r2 are the parts of the ratio for X and Y, respectively

noncomputable def X_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r1 / (r1 + r2)) * P

noncomputable def Y_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r2 / (r1 + r2)) * P

theorem profit_difference (P : ℕ) (r1 r2 : ℚ) (hP : P = 800) (hr1 : r1 = 1/2) (hr2 : r2 = 1/3) :
  X_share P r1 r2 - Y_share P r1 r2 = 160 := by
  sorry

end profit_difference_l579_57904


namespace find_second_divisor_l579_57939

theorem find_second_divisor:
  ∃ x: ℝ, (8900 / 6) / x = 370.8333333333333 ∧ x = 4 :=
sorry

end find_second_divisor_l579_57939


namespace rotate_cd_to_cd_l579_57974

def rotate180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

theorem rotate_cd_to_cd' :
  let C := (-1, 2)
  let C' := (1, -2)
  let D := (3, 2)
  let D' := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' :=
by
  sorry

end rotate_cd_to_cd_l579_57974


namespace find_x_l579_57902

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |> List.sum

def k := (10^45 - 999999999999999999999999999999999999999999994 : ℕ)

theorem find_x :
  digit_sum k = 397 := 
sorry

end find_x_l579_57902


namespace hillary_minutes_read_on_saturday_l579_57956

theorem hillary_minutes_read_on_saturday :
  let total_minutes := 60
  let friday_minutes := 16
  let sunday_minutes := 16
  total_minutes - (friday_minutes + sunday_minutes) = 28 := by
sorry

end hillary_minutes_read_on_saturday_l579_57956


namespace intersection_of_sets_l579_57947

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x | x < -1 ∨ x > 1}

noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

noncomputable def complement_U_M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

noncomputable def intersection_N_complement_U_M : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets :
  N ∩ complement_U_M = intersection_N_complement_U_M := 
sorry

end intersection_of_sets_l579_57947


namespace findB_coords_l579_57985

namespace ProofProblem

-- Define point A with its coordinates.
def A : ℝ × ℝ := (-3, 2)

-- Define a property that checks if a line segment AB is parallel to the x-axis.
def isParallelToXAxis (A B : (ℝ × ℝ)) : Prop :=
  A.2 = B.2

-- Define a property that checks if the length of line segment AB is 4.
def hasLengthFour (A B : (ℝ × ℝ)) : Prop :=
  abs (A.1 - B.1) = 4

-- The proof problem statement.
theorem findB_coords :
  ∃ B : ℝ × ℝ, isParallelToXAxis A B ∧ hasLengthFour A B ∧ (B = (-7, 2) ∨ B = (1, 2)) :=
  sorry

end ProofProblem

end findB_coords_l579_57985


namespace triangle_inequality_l579_57966

theorem triangle_inequality (x : ℕ) (hx : x > 0) :
  (x ≥ 34) ↔ (x + (10 + x) > 24) ∧ (x + 24 > 10 + x) ∧ ((10 + x) + 24 > x) := by
  sorry

end triangle_inequality_l579_57966


namespace person_age_is_30_l579_57943

-- Definitions based on the conditions
def age (x : ℕ) := x
def age_5_years_hence (x : ℕ) := x + 5
def age_5_years_ago (x : ℕ) := x - 5

-- The main theorem to prove
theorem person_age_is_30 (x : ℕ) (h : 3 * age_5_years_hence x - 3 * age_5_years_ago x = age x) : x = 30 :=
by
  sorry

end person_age_is_30_l579_57943


namespace price_of_books_sold_at_lower_price_l579_57991

-- Define the conditions
variable (n m p q t : ℕ) (earnings price_high price_low : ℝ)

-- The given conditions
def total_books : ℕ := 10
def books_high_price : ℕ := 2 * total_books / 5 -- 2/5 of total books
def books_low_price : ℕ := total_books - books_high_price
def high_price : ℝ := 2.50
def total_earnings : ℝ := 22

-- The proposition to prove
theorem price_of_books_sold_at_lower_price
  (h_books_high_price : books_high_price = 4)
  (h_books_low_price : books_low_price = 6)
  (h_total_earnings : total_earnings = 22)
  (h_high_price : high_price = 2.50) :
  (price_low = 2) := 
-- Proof goes here 
sorry

end price_of_books_sold_at_lower_price_l579_57991


namespace marco_strawberries_weight_l579_57936

theorem marco_strawberries_weight 
  (m : ℕ) 
  (total_weight : ℕ := 40) 
  (dad_weight : ℕ := 32) 
  (h : total_weight = m + dad_weight) : 
  m = 8 := 
sorry

end marco_strawberries_weight_l579_57936


namespace derivative_at_pi_div_2_l579_57913

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_at_pi_div_2 : deriv f (Real.pi / 2) = -Real.pi := by
  sorry

end derivative_at_pi_div_2_l579_57913


namespace average_monthly_income_l579_57914

theorem average_monthly_income (P Q R : ℝ) (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250) (h3 : P = 4000) : (P + R) / 2 = 5200 := by
  sorry

end average_monthly_income_l579_57914


namespace selling_price_percentage_l579_57930

  variable (L : ℝ)  -- List price
  variable (C : ℝ)  -- Cost price after discount
  variable (M : ℝ)  -- Marked price
  variable (S : ℝ)  -- Selling price after discount

  -- Conditions
  def cost_price_condition (L : ℝ) : ℝ := 0.7 * L
  def profit_condition (C S : ℝ) : Prop := 0.75 * S = C
  def marked_price_condition (S M : ℝ) : Prop := 0.85 * M = S

  theorem selling_price_percentage (L : ℝ) (h1 : C = cost_price_condition L)
    (h2 : profit_condition C S) (h3 : marked_price_condition S M) :
    S = 0.9333 * L :=
  by
    -- This is where the proof would go
    sorry
  
end selling_price_percentage_l579_57930


namespace number_of_lines_through_point_intersect_hyperbola_once_l579_57911

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_P : ℝ × ℝ :=
  (-4, 1)

noncomputable def line_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

noncomputable def one_point_intersection (l : ℝ × ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, l p ∧ H p.1 p.2

theorem number_of_lines_through_point_intersect_hyperbola_once :
  (∃ (l₁ l₂ : ℝ × ℝ → Prop),
    line_through point_P l₁ ∧
    line_through point_P l₂ ∧
    one_point_intersection l₁ hyperbola ∧
    one_point_intersection l₂ hyperbola ∧
    l₁ ≠ l₂) ∧ ¬ (∃ (l₃ : ℝ × ℝ → Prop),
    line_through point_P l₃ ∧
    one_point_intersection l₃ hyperbola ∧
    ∃! (other_line : ℝ × ℝ → Prop),
    line_through point_P other_line ∧
    one_point_intersection other_line hyperbola ∧
    l₃ ≠ other_line) :=
sorry

end number_of_lines_through_point_intersect_hyperbola_once_l579_57911


namespace calories_350_grams_mint_lemonade_l579_57938

-- Definitions for the weights of ingredients in grams
def lemon_juice_weight := 150
def sugar_weight := 200
def water_weight := 300
def mint_weight := 50
def total_weight := lemon_juice_weight + sugar_weight + water_weight + mint_weight

-- Definitions for the caloric content per specified weight
def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def mint_calories_per_10g := 7
def water_calories := 0

-- Calculate total calories from each ingredient
def lemon_juice_calories := (lemon_juice_calories_per_100g * lemon_juice_weight) / 100
def sugar_calories := (sugar_calories_per_100g * sugar_weight) / 100
def mint_calories := (mint_calories_per_10g * mint_weight) / 10

-- Calculate total calories in the lemonade
def total_calories := lemon_juice_calories + sugar_calories + mint_calories + water_calories

noncomputable def calories_in_350_grams : ℕ := (total_calories * 350) / total_weight

-- Theorem stating the number of calories in 350 grams of Marco’s lemonade
theorem calories_350_grams_mint_lemonade : calories_in_350_grams = 440 := 
by
  sorry

end calories_350_grams_mint_lemonade_l579_57938


namespace proof_problem_l579_57925

noncomputable def question (a b c d m : ℚ) : ℚ :=
  2 * a + 2 * b + (a + b - 3 * (c * d)) - m

def condition1 (m : ℚ) : Prop :=
  abs (m + 1) = 4

def condition2 (a b : ℚ) : Prop :=
  a = -b

def condition3 (c d : ℚ) : Prop :=
  c * d = 1

theorem proof_problem (a b c d m : ℚ) :
  condition1 m → condition2 a b → condition3 c d →
  (question a b c d m = 2 ∨ question a b c d m = -6) :=
by
  sorry

end proof_problem_l579_57925


namespace measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l579_57928

noncomputable def measure_of_angle_a (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : Prop :=
  A = Real.pi / 3

theorem measure_of_angle_A_proof (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : measure_of_angle_a a b c S h_c h_S :=
sorry

noncomputable def range_of_values_of_b_plus_c_over_a (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : Set ℝ :=
  {x : ℝ | 1 < x ∧ x ≤ 2}

theorem range_of_values_of_b_plus_c_over_a_proof (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : 
  ∃ x, x ∈ range_of_values_of_b_plus_c_over_a a b c A h_A h_c :=
sorry

end measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l579_57928


namespace find_m_if_extraneous_root_l579_57924

theorem find_m_if_extraneous_root :
  (∃ x : ℝ, x = 2 ∧ (∀ z : ℝ, z ≠ 2 → (m / (z-2) - 2*z / (2-z) = 1)) ∧ m = -4) :=
sorry

end find_m_if_extraneous_root_l579_57924


namespace initial_salt_percentage_l579_57990

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage_l579_57990


namespace ratio_fifteenth_term_l579_57921

-- Definitions of S_n and T_n based on the given conditions
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def T_n (b e n : ℕ) : ℕ := n * (2 * b + (n - 1) * e) / 2

-- Statement of the problem
theorem ratio_fifteenth_term 
  (a b d e : ℕ) 
  (h : ∀ n, (S_n a d n : ℚ) / (T_n b e n : ℚ) = (9 * n + 5) / (6 * n + 31)) : 
  (a + 14 * d : ℚ) / (b + 14 * e : ℚ) = (92 : ℚ) / 71 :=
by sorry

end ratio_fifteenth_term_l579_57921


namespace initial_balance_l579_57978

-- Define the conditions given in the problem
def transferred_percent_of_balance (X : ℝ) : ℝ := 0.15 * X
def balance_after_transfer (X : ℝ) : ℝ := 0.85 * X
def final_balance_after_refund (X : ℝ) (refund : ℝ) : ℝ := 0.85 * X + refund

-- Define the given values
def refund : ℝ := 450
def final_balance : ℝ := 30000

-- The theorem statement to prove the initial balance
theorem initial_balance (X : ℝ) (h : final_balance_after_refund X refund = final_balance) : 
  X = 34564.71 :=
by
  sorry

end initial_balance_l579_57978


namespace total_tea_cups_l579_57989

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups_l579_57989


namespace geometric_seq_ratio_l579_57905

theorem geometric_seq_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a (n+1) = q * a n)
  (h2 : 0 < q)                    -- ensuring positivity
  (h3 : 3 * a 0 + 2 * q * a 0 = q^2 * a 0)  -- condition from problem
  : ∀ n, (a (n+3) + a (n+2)) / (a (n+1) + a n) = 9 :=
by
  sorry

end geometric_seq_ratio_l579_57905


namespace some_base_value_l579_57955

noncomputable def some_base (x y : ℝ) (h1 : x * y = 1) (h2 : (some_base : ℝ) → (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : ℝ :=
  7

theorem some_base_value (x y : ℝ) (h1 : x * y = 1) (h2 : ∀ some_base : ℝ, (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : some_base x y h1 h2 = 7 :=
by
  sorry

end some_base_value_l579_57955


namespace total_carrots_l579_57963

-- Define the number of carrots grown by Sally and Fred
def sally_carrots := 6
def fred_carrots := 4

-- Theorem: The total number of carrots grown by Sally and Fred
theorem total_carrots : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l579_57963


namespace inequality_chain_l579_57934

open Real

theorem inequality_chain (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_chain_l579_57934


namespace roots_polynomial_l579_57973

theorem roots_polynomial (n r s : ℚ) (c d : ℚ)
  (h1 : c * c - n * c + 3 = 0)
  (h2 : d * d - n * d + 3 = 0)
  (h3 : (c + 1/d) * (d + 1/c) = s)
  (h4 : c * d = 3) :
  s = 16/3 :=
by
  sorry

end roots_polynomial_l579_57973


namespace triangle_fraction_squared_l579_57927

theorem triangle_fraction_squared (a b c : ℝ) (h1 : b > a) 
  (h2 : a / b = (1 / 2) * (b / c)) (h3 : a + b + c = 12) 
  (h4 : c = Real.sqrt (a^2 + b^2)) : 
  (a / b)^2 = 1 / 2 := 
by 
  sorry

end triangle_fraction_squared_l579_57927


namespace no_solution_exists_only_solution_is_1963_l579_57968

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n % 10 + sum_of_digits (n / 10)

-- Proof problem for part (a)
theorem no_solution_exists :
  ¬ ∃ x : ℕ, x + sum_of_digits x + sum_of_digits (sum_of_digits x) = 1993 :=
sorry

-- Proof problem for part (b)
theorem only_solution_is_1963 :
  ∃ x : ℕ, (x + sum_of_digits x + sum_of_digits (sum_of_digits x) + sum_of_digits (sum_of_digits (sum_of_digits x)) = 1993) ∧ (x = 1963) :=
sorry

end no_solution_exists_only_solution_is_1963_l579_57968


namespace sum_s_h_e_base_three_l579_57945

def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧ S < 3 ∧ H < 3 ∧ E < 3 ∧ S ≠ H ∧ H ≠ E ∧ S ≠ E

def base_three_addition (S H E : ℕ) :=
  (S + H * 3 + E * 9) + (H + E * 3) == (H * 3 + S * 9 + S*27)

theorem sum_s_h_e_base_three (S H E : ℕ) (h1 : distinct_non_zero_digits S H E) (h2 : base_three_addition S H E) :
  (S + H + E = 5) := by sorry

end sum_s_h_e_base_three_l579_57945


namespace custom_operation_correct_l579_57908

noncomputable def custom_operation (a b c : ℕ) : ℝ :=
  (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem custom_operation_correct : custom_operation 6 15 5 = 2 := by
  sorry

end custom_operation_correct_l579_57908


namespace recurring_decimal_product_l579_57940

theorem recurring_decimal_product : (0.3333333333 : ℝ) * (0.4545454545 : ℝ) = (5 / 33 : ℝ) :=
sorry

end recurring_decimal_product_l579_57940


namespace problem_l579_57992

theorem problem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 7 := 
sorry

end problem_l579_57992


namespace number_of_men_l579_57906

theorem number_of_men (M W C : ℕ) 
  (h1 : M + W + C = 10000)
  (h2 : C = 2500)
  (h3 : C = 5 * W) : 
  M = 7000 := 
by
  sorry

end number_of_men_l579_57906


namespace deborah_international_letters_l579_57965

theorem deborah_international_letters (standard_postage : ℝ) 
                                      (additional_charge : ℝ) 
                                      (total_letters : ℕ) 
                                      (total_cost : ℝ) 
                                      (h_standard_postage: standard_postage = 1.08)
                                      (h_additional_charge: additional_charge = 0.14)
                                      (h_total_letters: total_letters = 4)
                                      (h_total_cost: total_cost = 4.60) :
                                      ∃ (x : ℕ), x = 2 :=
by
  sorry

end deborah_international_letters_l579_57965


namespace articles_correct_l579_57942

-- Define the problem conditions
def refersToSpecific (word : String) : Prop :=
  word = "keyboard"

def refersToGeneral (word : String) : Prop :=
  word = "computer"

-- Define the articles
def the_article : String := "the"
def a_article : String := "a"

-- State the theorem for the corresponding solution
theorem articles_correct :
  refersToSpecific "keyboard" → refersToGeneral "computer" →  
  (the_article, a_article) = ("the", "a") :=
by
  intro h1 h2
  sorry

end articles_correct_l579_57942


namespace remainder_of_sum_division_l579_57957

theorem remainder_of_sum_division (x y : ℕ) (k m : ℕ) 
  (hx : x = 90 * k + 75) (hy : y = 120 * m + 115) :
  (x + y) % 30 = 10 :=
by sorry

end remainder_of_sum_division_l579_57957


namespace fraction_simplification_l579_57967

theorem fraction_simplification :
  (1 / 330) + (19 / 30) = 7 / 11 :=
by
  sorry

end fraction_simplification_l579_57967


namespace D_is_painting_l579_57910

def A_activity (act : String) : Prop := 
  act ≠ "walking" ∧ act ≠ "playing basketball"

def B_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "running"

def C_activity_implies_A_activity (C_act A_act : String) : Prop :=
  C_act = "walking" → A_act = "dancing"

def D_activity (act : String) : Prop :=
  act ≠ "playing basketball" ∧ act ≠ "running"

def C_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "playing basketball"

theorem D_is_painting :
  (∃ a b c d : String,
    A_activity a ∧
    B_activity b ∧
    C_activity_implies_A_activity c a ∧
    D_activity d ∧
    C_activity c) →
  ∃ d : String, d = "painting" :=
by
  intros h
  sorry

end D_is_painting_l579_57910


namespace find_sam_current_age_l579_57960

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l579_57960


namespace geometric_sequence_a5_l579_57959

variable (a : ℕ → ℝ) (q : ℝ)

axiom pos_terms : ∀ n, a n > 0

axiom a1a3_eq : a 1 * a 3 = 16
axiom a3a4_eq : a 3 + a 4 = 24

theorem geometric_sequence_a5 :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) → a 5 = 32 :=
by
  sorry

end geometric_sequence_a5_l579_57959


namespace a_10_eq_18_l579_57954

variable {a : ℕ → ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a2 : a 2 = 2
axiom a3 : a 3 = 4
axiom arithmetic_seq : is_arithmetic_sequence a

-- problem: prove a_{10} = 18
theorem a_10_eq_18 : a 10 = 18 :=
sorry

end a_10_eq_18_l579_57954


namespace total_strength_of_college_l579_57977

-- Declare the variables for number of students playing each sport
variables (C B Both : ℕ)

-- Given conditions in the problem
def cricket_players : ℕ := 500
def basketball_players : ℕ := 600
def both_players : ℕ := 220

-- Theorem stating the total strength of the college
theorem total_strength_of_college (h_C : C = cricket_players) 
                                  (h_B : B = basketball_players) 
                                  (h_Both : Both = both_players) : 
                                  C + B - Both = 880 :=
by
  sorry

end total_strength_of_college_l579_57977


namespace mean_equality_l579_57907

-- Define average calculation function
def average (a b c : ℕ) : ℕ :=
  (a + b + c) / 3

def average_two (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem mean_equality (x : ℕ) 
  (h : average 8 16 24 = average_two 10 x) : 
  x = 22 :=
by {
  -- The actual proof is here
  sorry
}

end mean_equality_l579_57907


namespace shadow_length_when_eight_meters_away_l579_57909

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l579_57909


namespace cost_of_each_book_is_six_l579_57923

-- Define variables for the number of books bought
def books_about_animals := 8
def books_about_outer_space := 6
def books_about_trains := 3

-- Define the total number of books
def total_books := books_about_animals + books_about_outer_space + books_about_trains

-- Define the total amount spent
def total_amount_spent := 102

-- Define the cost per book
def cost_per_book := total_amount_spent / total_books

-- Prove that the cost per book is $6
theorem cost_of_each_book_is_six : cost_per_book = 6 := by
  sorry

end cost_of_each_book_is_six_l579_57923


namespace danny_watermelon_slices_l579_57929

theorem danny_watermelon_slices : 
  ∀ (x : ℕ), 3 * x + 15 = 45 -> x = 10 := by
  intros x h
  sorry

end danny_watermelon_slices_l579_57929


namespace min_trucks_needed_l579_57949

theorem min_trucks_needed (n : ℕ) (w : ℕ) (t : ℕ) (total_weight : ℕ) (max_box_weight : ℕ) : 
    (total_weight = 10) → 
    (max_box_weight = 1) → 
    (t = 3) →
    (n * max_box_weight = total_weight) →
    (n ≥ 10) →
    ∀ min_trucks : ℕ, (min_trucks * t ≥ total_weight) → 
    min_trucks = 5 :=
by
  intro total_weight_eq max_box_weight_eq truck_capacity box_total_weight_eq n_lower_bound min_trucks min_trucks_condition
  sorry

end min_trucks_needed_l579_57949


namespace volume_ratio_proof_l579_57951

-- Definitions:
def height_ratio := 2 / 3
def volume_ratio (r : ℚ) := r^3
def small_pyramid_volume_ratio := volume_ratio height_ratio
def frustum_volume_ratio := 1 - small_pyramid_volume_ratio
def volume_ratio_small_to_frustum (v_small v_frustum : ℚ) := v_small / v_frustum

-- Lean 4 Statement:
theorem volume_ratio_proof
  (height_ratio : ℚ := 2 / 3)
  (small_pyramid_volume_ratio : ℚ := volume_ratio height_ratio)
  (frustum_volume_ratio : ℚ := 1 - small_pyramid_volume_ratio)
  (v_orig : ℚ) :
  volume_ratio_small_to_frustum (small_pyramid_volume_ratio * v_orig) (frustum_volume_ratio * v_orig) = 8 / 19 :=
by
  sorry

end volume_ratio_proof_l579_57951


namespace problem1_problem2_l579_57933

-- Define the function f(x) = |x + 2| + |x - 1|
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- 1. Prove the solution set of f(x) > 5 is {x | x < -3 or x > 2}
theorem problem1 : {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} :=
by
  sorry

-- 2. Prove that if f(x) ≥ a^2 - 2a always holds, then -1 ≤ a ≤ 3
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f x ≥ a^2 - 2 * a) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end problem1_problem2_l579_57933


namespace solve_equation_l579_57976

theorem solve_equation (x : ℝ) : (x - 2) ^ 2 = 9 ↔ x = 5 ∨ x = -1 :=
by
  sorry -- Proof is skipped

end solve_equation_l579_57976


namespace a_7_is_127_l579_57918

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 0  -- Define a_0 which is not used but useful for indexing
| 1       => 1
| (n + 2) => 2 * (a (n + 1)) + 1

-- Prove that a_7 = 127
theorem a_7_is_127 : a 7 = 127 := 
sorry

end a_7_is_127_l579_57918


namespace find_m_l579_57979

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1 + m, -1)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) : dot_product vector_a (vector_sum m) = 0 → m = -4 :=
by
  sorry

end find_m_l579_57979


namespace sin_sum_angles_36_108_l579_57993

theorem sin_sum_angles_36_108 (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_angle : A = 36 ∨ A = 108 ∨ B = 36 ∨ B = 108 ∨ C = 36 ∨ C = 108) :
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 :=
by
  sorry

end sin_sum_angles_36_108_l579_57993


namespace intersection_of_sets_l579_57941

theorem intersection_of_sets :
  let M := { x : ℝ | -3 < x ∧ x ≤ 5 }
  let N := { x : ℝ | -5 < x ∧ x < 5 }
  M ∩ N = { x : ℝ | -3 < x ∧ x < 5 } := 
by
  sorry

end intersection_of_sets_l579_57941


namespace quadratic_has_two_roots_l579_57953

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l579_57953


namespace correctness_of_option_C_l579_57917

noncomputable def vec_a : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1/2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem correctness_of_option_C :
  is_orthogonal (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) :=
by
  sorry

end correctness_of_option_C_l579_57917


namespace hyperbola_asymptote_l579_57901

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, (x^2 - y^2 / a^2) = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 :=
by
  intro h_asymptote
  sorry

end hyperbola_asymptote_l579_57901


namespace common_factor_l579_57971

-- Define the polynomials
def P1 (x : ℝ) : ℝ := x^3 + x^2
def P2 (x : ℝ) : ℝ := x^2 + 2*x + 1
def P3 (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem common_factor (x : ℝ) : ∃ (f : ℝ → ℝ), (f x = x + 1) ∧ (∃ g1 g2 g3 : ℝ → ℝ, P1 x = f x * g1 x ∧ P2 x = f x * g2 x ∧ P3 x = f x * g3 x) :=
sorry

end common_factor_l579_57971


namespace factorial_sum_simplify_l579_57981

theorem factorial_sum_simplify :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 5) + 3 * (Nat.factorial 3) + (Nat.factorial 3) = 35904 :=
by
  sorry

end factorial_sum_simplify_l579_57981


namespace number_of_functions_l579_57980

-- Define the set of conditions
variables (x y : ℝ)

def relation1 := x - y = 0
def relation2 := y^2 = x
def relation3 := |y| = 2 * x
def relation4 := y^2 = x^2
def relation5 := y = 3 - x
def relation6 := y = 2 * x^2 - 1
def relation7 := y = 3 / x

-- Prove that there are 4 unambiguous functions of y with respect to x
theorem number_of_functions : 4 = 4 := sorry

end number_of_functions_l579_57980


namespace find_largest_m_l579_57999

variables (a b c t : ℝ)
def f (x : ℝ) := a * x^2 + b * x + c

theorem find_largest_m (a_ne_zero : a ≠ 0)
  (cond1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x) ∧ f a b c x ≥ x)
  (cond2 : ∀ x : ℝ, 0 < x ∧ x < 2 → f a b c x ≤ ((x + 1) / 2)^2)
  (cond3 : ∃ x : ℝ, ∀ y : ℝ, f a b c y ≥ f a b c x ∧ f a b c x = 0) :
  ∃ m : ℝ, 1 < m ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x) ∧ m = 9 := sorry

end find_largest_m_l579_57999


namespace ellipse_shortest_major_axis_l579_57972

theorem ellipse_shortest_major_axis (P : ℝ × ℝ) (a b : ℝ) 
  (ha : a > b) (hb : b > 0) (hP_on_line : P.2 = P.1 + 2)
  (h_foci_hyperbola : ∃ c : ℝ, c = 1 ∧ a^2 - b^2 = c^2) :
  (∃ a b : ℝ, a^2 = 5 ∧ b^2 = 4 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :=
sorry

end ellipse_shortest_major_axis_l579_57972


namespace sufficient_condition_l579_57962

theorem sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 2, (1/2 : ℝ) * x^2 - a ≥ 0) → a ≤ 0 :=
by
  sorry

end sufficient_condition_l579_57962


namespace range_of_m_l579_57948

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) * (2 * x - 1) - m * x + m

def exists_unique_int_n (m : ℝ) : Prop :=
∃! n : ℤ, f n m < 0

theorem range_of_m {m : ℝ} (h : m < 1) (h2 : exists_unique_int_n m) : 
  (Real.exp 1) * (1 / 2) ≤ m ∧ m < 1 :=
sorry

end range_of_m_l579_57948


namespace pizzas_needed_l579_57958

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end pizzas_needed_l579_57958


namespace find_k_l579_57916

theorem find_k (k : ℝ) (h1 : k > 1) 
(h2 : ∑' n : ℕ, (7 * (n + 1) - 3) / k^(n + 1) = 2) : 
  k = 2 + 3 * Real.sqrt 2 / 2 := 
sorry

end find_k_l579_57916


namespace function_equiv_proof_l579_57952

noncomputable def function_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

theorem function_equiv_proof : ∀ f : ℝ → ℝ,
  function_solution f ↔ (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = -x) := 
sorry

end function_equiv_proof_l579_57952


namespace Victor_Total_Money_l579_57903

-- Definitions for the conditions
def originalAmount : Nat := 10
def allowance : Nat := 8

-- The proof problem statement
theorem Victor_Total_Money : originalAmount + allowance = 18 := by
  sorry

end Victor_Total_Money_l579_57903


namespace determine_120_percent_of_y_l579_57926

def x := 0.80 * 350
def y := 0.60 * x
def result := 1.20 * y

theorem determine_120_percent_of_y : result = 201.6 := by
  sorry

end determine_120_percent_of_y_l579_57926


namespace double_variable_for_1600_percent_cost_l579_57950

theorem double_variable_for_1600_percent_cost (t b0 b1 : ℝ) (h : t ≠ 0) :
    (t * b1^4 = 16 * t * b0^4) → b1 = 2 * b0 :=
by
sorry

end double_variable_for_1600_percent_cost_l579_57950


namespace mixed_operation_with_rationals_l579_57919

theorem mixed_operation_with_rationals :
  (- (2 / 21)) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := 
by 
  sorry

end mixed_operation_with_rationals_l579_57919


namespace factorize_expression_l579_57969

theorem factorize_expression (a : ℝ) : 
  (a + 1) * (a + 2) + 1 / 4 = (a + 3 / 2)^2 := 
by 
  sorry

end factorize_expression_l579_57969


namespace calculate_total_notebooks_given_to_tom_l579_57988

noncomputable def total_notebooks_given_to_tom : ℝ :=
  let initial_red := 15
  let initial_blue := 17
  let initial_white := 19
  let red_given_day1 := 4.5
  let blue_given_day1 := initial_blue / 3
  let remaining_red_day1 := initial_red - red_given_day1
  let remaining_blue_day1 := initial_blue - blue_given_day1
  let white_given_day2 := initial_white / 2
  let blue_given_day2 := remaining_blue_day1 * 0.25
  let remaining_white_day2 := initial_white - white_given_day2
  let remaining_blue_day2 := remaining_blue_day1 - blue_given_day2
  let red_given_day3 := 3.5
  let blue_given_day3 := (remaining_blue_day2 * 2) / 5
  let remaining_red_day3 := remaining_red_day1 - red_given_day3
  let remaining_blue_day3 := remaining_blue_day2 - blue_given_day3
  let white_kept_day3 := remaining_white_day2 / 4
  let remaining_white_day3 := initial_white - white_kept_day3
  let remaining_notebooks_day3 := remaining_red_day3 + remaining_blue_day3 + remaining_white_day3
  let notebooks_total_day3 := initial_red + initial_blue + initial_white - red_given_day1 - blue_given_day1 - white_given_day2 - blue_given_day2 - red_given_day3 - blue_given_day3 - white_kept_day3
  let tom_notebooks := red_given_day1 + blue_given_day1
  notebooks_total_day3

theorem calculate_total_notebooks_given_to_tom : total_notebooks_given_to_tom = 10.17 :=
  sorry

end calculate_total_notebooks_given_to_tom_l579_57988


namespace decimal_fraction_to_percentage_l579_57994

theorem decimal_fraction_to_percentage (d : ℝ) (h : d = 0.03) : d * 100 = 3 := by
  sorry

end decimal_fraction_to_percentage_l579_57994


namespace min_sine_range_l579_57944

noncomputable def min_sine_ratio (α β γ : ℝ) := min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β)

theorem min_sine_range (α β γ : ℝ) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : α + β + γ = Real.pi) :
  1 ≤ min_sine_ratio α β γ ∧ min_sine_ratio α β γ < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end min_sine_range_l579_57944


namespace circle_inscribed_isosceles_trapezoid_l579_57935

theorem circle_inscribed_isosceles_trapezoid (r a c : ℝ) : 
  (∃ base1 base2 : ℝ,  2 * a = base1 ∧ 2 * c = base2) →
  (∃ O : ℝ, O = r) →
  r^2 = a * c :=
by
  sorry

end circle_inscribed_isosceles_trapezoid_l579_57935
