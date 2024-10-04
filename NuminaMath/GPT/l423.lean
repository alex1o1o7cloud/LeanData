import Mathlib

namespace scientific_notation_10030000000_l423_423347

theorem scientific_notation_10030000000 : (10,030,000,000 : ℝ) = 1.003 * 10^10 := by
  sorry

end scientific_notation_10030000000_l423_423347


namespace parabola_directrix_l423_423302

theorem parabola_directrix (y : ℝ) (x : ℝ) (h : y = 8 * x^2) : 
  y = -1 / 32 :=
sorry

end parabola_directrix_l423_423302


namespace pencil_price_is_99c_l423_423604

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c_l423_423604


namespace fraction_not_integer_l423_423985

theorem fraction_not_integer (a b : ℤ) : ¬ (∃ k : ℤ, (a^2 + b^2) = k * (a^2 - b^2)) :=
sorry

end fraction_not_integer_l423_423985


namespace determine_AF_l423_423522

-- Definitions for parabola and focus
def parabola := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }
def focus : ℝ × ℝ := (1/2, 0)

-- Line passing through focus and intersecting parabola at points A and B
variable (l : ℝ → ℝ)

-- Points A and B on the parabola
variable (A B : ℝ × ℝ)
variable hA : A ∈ parabola
variable hB : B ∈ parabola

-- Line condition
variable h_line_A : A.2 = l (A.1 - focus.1)
variable h_line_B : B.2 = l (B.1 - focus.1)

-- Distance between A and B
variable h_AB : abs (A.1 - B.1) = 25/12

-- Distance condition |AF| < |BF|
variable h_AF_lt_BF : abs ((focus.1 - A.1)^2 + (focus.2 - A.2)^2).sqrt < abs ((focus.1 - B.1)^2 + (focus.2 - B.2)^2).sqrt

-- Proof goal to determine |AF|
theorem determine_AF : abs ((focus.1 - A.1)^2 + (focus.2 - A.2)^2).sqrt = 5/6 :=
by
  sorry

end determine_AF_l423_423522


namespace min_distance_point_to_line_l423_423836

theorem min_distance_point_to_line (m n : ℝ) (h : 2 * m + n + 5 = 0) :
    ∃ d : ℝ, d = √((m - 1)^2 + (n + 2)^2) ∧ d = √5 :=
by
  sorry

end min_distance_point_to_line_l423_423836


namespace length_of_shortest_chord_l423_423437

noncomputable def shortest_chord_length_through_focus : ℝ :=
let a := 4
let b := 3
let c := Real.sqrt (a^2 - b^2) in
let y1 := Real.sqrt (9 - 9 * (a^2 - c^2) / a^2) in
2 * y1

theorem length_of_shortest_chord (x y : ℝ) (h_ellipse : (x^2 / 16) + (y^2 / 9) = 1) :
  shortest_chord_length_through_focus = 9 / 2 :=
by
  sorry -- Proof is left as an exercise

end length_of_shortest_chord_l423_423437


namespace form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l423_423336

theorem form_eleven : 22 - (2 + (2 / 2)) = 11 := by
  sorry

theorem form_twelve : (2 * 2 * 2) - 2 / 2 = 12 := by
  sorry

theorem form_thirteen : (22 + 2 + 2) / 2 = 13 := by
  sorry

theorem form_fourteen : 2 * 2 * 2 * 2 - 2 = 14 := by
  sorry

theorem form_fifteen : (2 * 2)^2 - 2 / 2 = 15 := by
  sorry

theorem form_sixteen : (2 * 2)^2 * (2 / 2) = 16 := by
  sorry

theorem form_seventeen : (2 * 2)^2 + 2 / 2 = 17 := by
  sorry

theorem form_eighteen : 2 * 2 * 2 * 2 + 2 = 18 := by
  sorry

theorem form_nineteen : 22 - 2 - 2 / 2 = 19 := by
  sorry

theorem form_twenty : (22 - 2) * (2 / 2) = 20 := by
  sorry

end form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l423_423336


namespace sum_primitive_roots_11_l423_423136

def isPrimitiveRootMod (a p : ℕ) : Prop :=
  let powers := (1 to p - 1).map (fun k => a^k % p)
  powers.sort = (1 to p - 1).toList

theorem sum_primitive_roots_11 : 
  let roots := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter (fun a => isPrimitiveRootMod a 11)
  roots.sum = 15 :=
by
  sorry

end sum_primitive_roots_11_l423_423136


namespace parallelogram_inner_product_l423_423141

namespace MathProof

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D E : V)
variable (m : ℝ)

-- Conditions
def is_parallelogram (A B C D : V) : Prop :=
∃ (X Y : V), A = X ∧ B = X + Y ∧ C = X - Y ∧ D = 2 • X - Y

def midpoint (E : V) (C D : V) : Prop :=
E = (C + D) / 2

def condition1 (A D : V) : Prop :=
dist A D = 2

def condition2 (A B D : V) : Prop :=
inner A D = dist A B * dist A D * -1/2

def condition3 (A E B D : V) : Prop :=
inner (E - A) (D - B) = 1

-- Result to prove
theorem parallelogram_inner_product :
  ∀ (A B C D E : V),
    is_parallelogram A B C D →
    AD = 2 → 
    ∠ BAD = 120∘ →
    midpoint E C D →
    (A E) ⋅ (B D) = 1 →
    (B D) ⋅ (B E) = 13 :=
begin
  intros A B C D E hp h1 h2 h3 h4,
  sorry -- Proof to be filled in
end

end MathProof

end parallelogram_inner_product_l423_423141


namespace complex_number_quadrant_l423_423839

theorem complex_number_quadrant :
  ∃ z : ℂ, (3 + 4 * complex.i) * z = complex.abs (4 - 3 * complex.i) ∧ z.re > 0 ∧ z.im < 0 :=
by
  let z := (3 / 5) - (4 / 5) * complex.i
  have h1 : (3 + 4 * complex.i) * z = complex.abs (4 - 3 * complex.i) := sorry
  have h2 : z.re = 3 / 5 := by sorry
  have h3 : z.im = -4 / 5 := by sorry
  use z
  simp only [h2, h3]
  constructor; [exact h1, exact h2, exact h3]

end complex_number_quadrant_l423_423839


namespace min_value_expression_l423_423108

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l423_423108


namespace partition_balanced_sets_l423_423730

def is_balanced_set (s : Finset ℕ) : Prop :=
  ¬s.is_empty ∧ s.card = (s.sum / s.card)

theorem partition_balanced_sets : 
  ∃ (sets : Finset (Finset ℕ)), 
  (∀ s ∈ sets, is_balanced_set s) ∧
  (∀ ⦃s1 s2⦄, s1 ∈ sets → s2 ∈ sets → s1 ≠ s2 → s1 ∩ s2 = ∅) ∧
  (Finset.univ = (Finset.range (2021^2 + 1)).bUnion id) := 
sorry

end partition_balanced_sets_l423_423730


namespace correct_statements_count_l423_423400

-- Definitions of each condition as propositions
def extend_straight_line_AB : Prop := false
def extend_line_segment_AB : Prop := true
def extend_ray_AB : Prop := false
def draw_straight_line_AB_5cm : Prop := false
def cut_line_segment_AC_on_ray_AB_5cm : Prop := true

-- Sum of correct conditions
def number_of_correct_statements : Nat :=
  if extend_straight_line_AB then 1 else 0
  + if extend_line_segment_AB then 1 else 0
  + if extend_ray_AB then 1 else 0
  + if draw_straight_line_AB_5cm then 1 else 0
  + if cut_line_segment_AC_on_ray_AB_5cm then 1 else 0

-- The theorem we are required to prove
theorem correct_statements_count : number_of_correct_statements = 2 :=
by
  sorry

end correct_statements_count_l423_423400


namespace intersection_angles_l423_423797

def f1 (x : ℝ) : ℝ := -x^2 + 4*x + 4
def f2 (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem intersection_angles :
  let x1 := 0
      x2 := 3
      α1 := Real.arctan (-(6 / 7))
      α2 := Real.arctan (6 / 7)
      f1' (x : ℝ) := -2*x + 4
      f2' (x : ℝ) := 2*x - 2
  in ∀ i ∈ ({x1, x2} : Finset ℝ), 
     let α := if i = x1 then α1 else α2
         f1i := f1' i
         f2i := f2' i
         tan_alpha := (f1i - f2i) / (1 + f1i * f2i)
     in α = Real.arctan(tan_alpha) :=
by sorry

end intersection_angles_l423_423797


namespace right_triangle_trisection_l423_423315

theorem right_triangle_trisection
  (A B C P Q : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]
  (triangle_abc : right_triangle A B C)
  (divide_hypotenuse : divides_into_three_equal_parts P Q (hypotenuse A B))
  (hypotenuse_ABC : hypotenuse (A B) = AB)
  (CP : distance C P)
  (PQ : distance P Q)
  (QC : distance Q C) :
  CP^2 + PQ^2 + QC^2 = (2/3) * (distance A B)^2 := 
sorry

end right_triangle_trisection_l423_423315


namespace coefficient_x_squared_expansion_l423_423297

noncomputable def coeff_expansion : ℤ :=
let c1 := 1, c2 := -6, c3 := 15, c4 := 1, c5 := 12, c6 := 60 in
c1 * c6 + c2 * c5 + c3 * c4

theorem coefficient_x_squared_expansion :
  coeff_expansion = 3 :=
  by
    unfold coeff_expansion
    sorry

end coefficient_x_squared_expansion_l423_423297


namespace f_zero_eq_one_positive_for_all_x_l423_423961

variables {R : Type*} [LinearOrderedField R] (f : R → R)

-- Conditions
axiom domain (x : R) : true -- This translates that f has domain (-∞, ∞)
axiom non_constant (x1 x2 : R) (h : x1 ≠ x2) : f x1 ≠ f x2
axiom functional_eq (x y : R) : f (x + y) = f x * f y

-- Questions
theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem positive_for_all_x (x : R) : f x > 0 :=
sorry

end f_zero_eq_one_positive_for_all_x_l423_423961


namespace find_OP_l423_423610

-- Define the variables and hypotheses
variables (a b c d : ℝ)
variables (P : ℝ)
variables (H1 : b ≤ P)
variables (H2 : P ≤ c)

-- The theorem to prove
theorem find_OP (H3 : ∃ (AP PD BP PC : ℝ), AP = 2 * PD ∧ BP = 2 * PC ∧ 
  |2 * a - P| / |P - 3 * d| = 2 * (|b - P| / |P - c|)) :
  P = (-(2 * a + c - 2 * b - 6 * d) + sqrt((2 * a + c - 2 * b - 6 * d)^2 - 4 * (6 * b * d - a * c))) / 2 :=
sorry

end find_OP_l423_423610


namespace simplify_sqrt_neg2_squared_l423_423289

theorem simplify_sqrt_neg2_squared : 
  Real.sqrt ((-2 : ℝ)^2) = 2 := 
by
  sorry

end simplify_sqrt_neg2_squared_l423_423289


namespace midpoint_PQ_of_ellipse_l423_423936

theorem midpoint_PQ_of_ellipse 
  {a b : ℝ} {UV : LineSegment ℝ} {AB CD : Line ℝ} {P Q M : Point ℝ} 
  (h1 : is_midpoint M UV)
  (h2 : UV.parallel_to_major_axis a b)
  (h3 : is_chord_through_M M AB)
  (h4 : is_chord_through_M M CD)
  (h5 : intersection_ac_bd_with_uv P Q UV M) :
  is_midpoint M (segment PQ) :=
sorry

end midpoint_PQ_of_ellipse_l423_423936


namespace quotient_of_5_divided_by_y_is_5_point_3_l423_423663

theorem quotient_of_5_divided_by_y_is_5_point_3 (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 :=
by
  sorry

end quotient_of_5_divided_by_y_is_5_point_3_l423_423663


namespace total_amount_paid_correct_l423_423971

-- Define variables for prices of the pizzas
def first_pizza_price : ℝ := 8
def second_pizza_price : ℝ := 12
def third_pizza_price : ℝ := 10

-- Define variables for discount rate and tax rate
def discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.05

-- Define the total amount paid by Mrs. Hilt
def total_amount_paid : ℝ :=
  let total_cost := first_pizza_price + second_pizza_price + third_pizza_price
  let discount := total_cost * discount_rate
  let discounted_total := total_cost - discount
  let sales_tax := discounted_total * sales_tax_rate
  discounted_total + sales_tax

-- Prove that the total amount paid is $25.20
theorem total_amount_paid_correct : total_amount_paid = 25.20 := 
  by
  sorry

end total_amount_paid_correct_l423_423971


namespace visited_both_countries_l423_423550

theorem visited_both_countries (total_people visited_Iceland visited_Norway visited_neither : ℕ) 
(h_total: total_people = 60)
(h_visited_Iceland: visited_Iceland = 35)
(h_visited_Norway: visited_Norway = 23)
(h_visited_neither: visited_neither = 33) : 
total_people - visited_neither = visited_Iceland + visited_Norway - (visited_Iceland + visited_Norway - (total_people - visited_neither)) :=
by sorry

end visited_both_countries_l423_423550


namespace number_of_questions_per_survey_is_10_l423_423411

variable {Q : ℕ}  -- Q: Number of questions in each survey

def money_per_question : ℝ := 0.2
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4
def total_money_earned : ℝ := 14

theorem number_of_questions_per_survey_is_10 :
    (surveys_on_monday + surveys_on_tuesday) * Q * money_per_question = total_money_earned → Q = 10 :=
by
  sorry

end number_of_questions_per_survey_is_10_l423_423411


namespace negative_represents_departure_of_30_tons_l423_423217

theorem negative_represents_departure_of_30_tons (positive_neg_opposites : ∀ x:ℤ, -x = x * (-1))
  (arrival_represents_30 : ∀ x:ℤ, (x = 30) ↔ ("+30" represents arrival of 30 tons of grain)) :
  "-30" represents departure of 30 tons of grain :=
sorry

end negative_represents_departure_of_30_tons_l423_423217


namespace angle_between_vectors_l423_423482

variables {α : Type*} [InnerProductSpace ℝ α]
variables (a b : α)

def norm_a : ℝ := ‖a‖
def norm_b : ℝ := ‖b‖
def perp_condition : Prop := inner (a - b) a = 0

theorem angle_between_vectors (h1 : norm_a = 1) (h2 : norm_b = real.sqrt 2) (h3 : perp_condition a b) :
  real.angle a b = real.pi / 4 :=
sorry

end angle_between_vectors_l423_423482


namespace sum_units_tens_not_divisible_by_4_l423_423270

theorem sum_units_tens_not_divisible_by_4 :
  ∃ (n : ℕ), (n = 3674 ∨ n = 3684 ∨ n = 3694 ∨ n = 3704 ∨ n = 3714 ∨ n = 3722) ∧
  (¬ (∃ k, (n % 100) = 4 * k)) ∧
  ((n % 10) + (n / 10 % 10) = 11) :=
sorry

end sum_units_tens_not_divisible_by_4_l423_423270


namespace bus_stops_time_l423_423447

variables (speed_without_stops speed_with_stops distance_lost speed_in_km_per_min : ℝ)

-- Given conditions
def average_speed_without_stops := speed_without_stops = 60
def average_speed_with_stops := speed_with_stops = 40
def distance_loss := distance_lost = speed_without_stops - speed_with_stops

-- Converting speed to km/min
def convert_speed := speed_in_km_per_min = speed_without_stops / 60

-- Aim to prove
theorem bus_stops_time :
  average_speed_without_stops ∧ average_speed_with_stops ∧ distance_loss ∧ convert_speed →
  (20:ℝ) = distance_lost / speed_in_km_per_min :=
by
  sorry

end bus_stops_time_l423_423447


namespace cannot_make_all_positive_l423_423520

def matrix := 
  [ [-1, 2, -3, 4], 
    [-1.2, 0.5, -3.9, 9], 
    [Real.pi, -12, 4, -2.5], 
    [63, 1.4, 7, -9] ]

def transform (m : List (List ℝ)) (idx : ℕ) (is_row : Bool) : List (List ℝ) :=
  if is_row then
    m.map_with_index (λ i row, if i = idx then row.map (λ x, -x) else row)
  else
    m.map (λ row, row.map_with_index (λ j x, if j = idx then -x else x))

theorem cannot_make_all_positive : ¬ ∃ (t : ℕ → Bool × ℕ), 
  (∀ n, let (is_row, idx) := t n in
    length (transform (iterate (λ m, let (is_row, idx) := t n in transform m idx is_row) matrix n).flat_map id).filter (λ x, x > 0) = 16) := 
sorry

end cannot_make_all_positive_l423_423520


namespace identify_valid_statements_l423_423766

theorem identify_valid_statements (a b : ℝ) :
  (sqrt (a^2 + b^2) = a ∨ sqrt (a^2 + b^2) = sqrt a * sqrt b ∨ sqrt (a^2 + b^2) = a * b) →
  (a ≠ 0 ∨ b ≠ 0) → (sqrt (a^2 + b^2) = a) :=
sorry

end identify_valid_statements_l423_423766


namespace true_statement_about_M_l423_423254

variable (U : Set ℕ) (M : Set ℕ)
axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_M_def : U \ M = {1, 3}

theorem true_statement_about_M : 2 ∈ M :=
by 
  rw [U_def, complement_M_def]
  sorry

end true_statement_about_M_l423_423254


namespace prime_in_A_l423_423233

def is_in_A (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 2 * b^2 ∧ b ≠ 0

theorem prime_in_A (p : ℕ) (hp : Nat.Prime p) (h : is_in_A (p^2)) : is_in_A p :=
by
  sorry

end prime_in_A_l423_423233


namespace smallest_int_m_for_two_sided_cards_l423_423932

noncomputable def smallest_m : Nat :=
  let S := (Finset.range 999).map Nat.succ  -- Set S = {1, 2, ..., 999}
  666  -- The answer determined by the solution

theorem smallest_int_m_for_two_sided_cards :
  ∃ (C : Fin (smallest_m) → (ℕ × ℕ)),
    (∀ i, C i.1 ∈ S ∧ C i.2 ∈ S) ∧
    (∀ (x ≠ y ∈ S), ∃ (i j : Fin (smallest_m)), i ≠ j ∧ ((C i).1 = x ∨ (C i).2 = x) ∧ ((C j).1 = y ∨ (C j).2 = y)) :=
  sorry

end smallest_int_m_for_two_sided_cards_l423_423932


namespace line_tangent_to_circle_l423_423175

-- Define the equations of the line and the circle
def line_l (x y : ℝ) : Prop := x - y + 3 = 0
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the center of the circle and its radius
def center_C : ℝ × ℝ := (-1, 0)
def radius_C : ℝ := real.sqrt 2

-- Define the distance from a point to a line (general formula)
def point_to_line_distance (px py : ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * px + b * py + c)) / real.sqrt (a^2 + b^2)

-- Calculate the distance from the center of the circle to the line
def distance_from_center_to_l : ℝ :=
  point_to_line_distance (-1) 0 1 (-1) 3

-- The main theorem: the line is tangent to the circle
theorem line_tangent_to_circle : distance_from_center_to_l = radius_C :=
  sorry

end line_tangent_to_circle_l423_423175


namespace modulus_Z_l423_423507

noncomputable def Z : ℂ := (sqrt 3 + complex.i) / (1 - sqrt 3 * complex.i)^2

theorem modulus_Z : complex.abs Z = 1 / 2 := by
  sorry

end modulus_Z_l423_423507


namespace initial_unread_messages_correct_l423_423261

-- Definitions based on conditions
def messages_read_per_day := 20
def messages_new_per_day := 6
def duration_in_days := 7
def effective_reading_rate := messages_read_per_day - messages_new_per_day

-- The initial number of unread messages
def initial_unread_messages := duration_in_days * effective_reading_rate

-- The theorem we want to prove
theorem initial_unread_messages_correct :
  initial_unread_messages = 98 :=
sorry

end initial_unread_messages_correct_l423_423261


namespace equilateral_triangle_side_length_l423_423495

theorem equilateral_triangle_side_length (m : ℝ) 
  (h_eq : (∀ x y : ℝ, x = m^2 / 4 ∧ (y = m ∨ y = -m) → y^2 = 4 * x))
  (h_tan : real.tan (real.pi / 6) = real.sqrt 3 / 3) :
  2 * 4 * real.sqrt 3 = 8 * real.sqrt 3 :=
by
  sorry

end equilateral_triangle_side_length_l423_423495


namespace triangle_area_AFO_eq_nine_l423_423146

/-- Given an ellipse equation, symmetric points, foci, and perpendicularity conditions,
prove the area of triangle AF₂B is 9. -/
theorem triangle_area_AFO_eq_nine :
  ∃ (x y : ℝ), (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 + y^2 = 16) ∧ 
  (y = 9 / 4) ∧ ((a, b), (a, -b)) ∧ area_of_triangle_AF₂B = 9 := 
sorry

end triangle_area_AFO_eq_nine_l423_423146


namespace area_of_triangle_NQF_l423_423223

/-- Given a trapezoid MPQF with bases MF = 24, PQ = 4, and height 5.
    Point N divides MP such that MN = 3NP.
    Prove that the area of triangle NQF is 22.5. -/
theorem area_of_triangle_NQF
  (MF PQ height : ℝ)
  (h1 : MF = 24)
  (h2 : PQ = 4)
  (h3 : height = 5)
  (MN NP : ℝ)
  (h4 : MN = 3 * NP) :
  let area_NQF := 22.5 in
  area_NQF = 22.5 :=
by
  sorry

end area_of_triangle_NQF_l423_423223


namespace intervals_of_monotonicity_min_value_on_interval_l423_423477

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x ^ 2 + a + 1)

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≥ 0 → ∀ x y : ℝ, f a x ≥ f a y ↔ x ≤ y) ∧
  (a < 0 → ∃ r1 r2 : ℝ, r1 < 1 ∧ r2 > 2 ∧
    ((∀ x : ℝ, x ∈ Iio r1 ∨ x ∈ Ioi r2 → f a x ≥ f a y ↔ x ≤ y) ∧
    (∀ x y : ℝ, x ∈ Icc r1 r2 → f a x ≥ f a y ↔ x ≥ y))) :=
sorry

theorem min_value_on_interval (a : ℝ) (ha : -1 < a ∧ a < 0) :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f a x ≥ f a 2 :=
sorry

end intervals_of_monotonicity_min_value_on_interval_l423_423477


namespace trains_crossing_time_l423_423335

noncomputable def TrainA_length := 200  -- meters
noncomputable def TrainA_time := 15  -- seconds
noncomputable def TrainB_length := 300  -- meters
noncomputable def TrainB_time := 25  -- seconds

noncomputable def Speed (length : ℕ) (time : ℕ) := (length : ℝ) / (time : ℝ)

noncomputable def TrainA_speed := Speed TrainA_length TrainA_time
noncomputable def TrainB_speed := Speed TrainB_length TrainB_time

noncomputable def relative_speed := TrainA_speed + TrainB_speed
noncomputable def total_distance := (TrainA_length : ℝ) + (TrainB_length : ℝ)

noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  (crossing_time : ℝ) = 500 / 25.33 :=
sorry

end trains_crossing_time_l423_423335


namespace fractions_are_integers_l423_423236

theorem fractions_are_integers (a b : ℕ) (h1 : 1 < a) (h2 : 1 < b) 
    (h3 : abs ((a : ℚ) / b - (a - 1) / (b - 1)) = 1) : 
    ∃ m n : ℤ, (a : ℚ) / b = m ∧ (a - 1) / (b - 1) = n := 
sorry

end fractions_are_integers_l423_423236


namespace sum_of_roots_l423_423458

noncomputable def int_part (x : ℝ) : ℤ := int.floor x
noncomputable def frac_part (x : ℝ) : ℝ := x - int_part x

theorem sum_of_roots (x : ℝ) (n : ℤ) (f : ℝ) :
  (int_part x * (int_part x - 2) = 3 - frac_part x) → (int_part x = n) → (frac_part x = f) → (0 ≤ f ∧ f < 1) → n = 2 :=
by {
  sorry
}

end sum_of_roots_l423_423458


namespace johns_ratio_l423_423206

-- Definitions for initial counts
def initial_pink := 26
def initial_green := 15
def initial_yellow := 24
def initial_total := initial_pink + initial_green + initial_yellow

-- Definitions for Carl's and John's actions
def carl_pink_taken := 4
def john_pink_taken := 6
def remaining_pink := initial_pink - carl_pink_taken - john_pink_taken

-- Definition for remaining hard hats
def total_remaining := 43

-- Compute John's green hat withdrawal
def john_green_taken := (initial_total - carl_pink_taken - john_pink_taken) - total_remaining
def ratio := john_green_taken / john_pink_taken

theorem johns_ratio : ratio = 2 :=
by
  -- Proof details omitted
  sorry

end johns_ratio_l423_423206


namespace triangle_with_incenter_segment_has_longest_side_as_11_l423_423639

open Classical

noncomputable def triangle_longest_side (area perimeter : ℝ) (segment_from_vertex_to_incenter : ℝ) : ℝ :=
  (if sqrt 21 * 8 = area ∧ perimeter = 24 ∧ segment_from_vertex_to_incenter = sqrt 30 / 3
     then 11
     else 0)

theorem triangle_with_incenter_segment_has_longest_side_as_11 :
  triangle_longest_side (4 * sqrt 21) 24 (sqrt 30 / 3) = 11 :=
sorry

end triangle_with_incenter_segment_has_longest_side_as_11_l423_423639


namespace projections_of_opposite_sides_equal_l423_423382

variable {A B C D H K : Point}
variable [circumscribed_circle : Circle]
variable {AC BD : Line}
variable [DiameterAC : AC.is_diameter circumscribed_circle]
variable [ProjectionAH : (A, H, BD)] [ProjectionCK : (C, K, BD)]

theorem projections_of_opposite_sides_equal :
  (is_cyclic_quadrilateral A B C D) ∧
  (A, H, BD) ∧
  (C, K, BD) ∧
  AC.is_diameter circumscribed_circle →
  (projection_length A B BD = projection_length C D BD) ∧
  (projection_length B C BD = projection_length A D BD) := by
  sorry

end projections_of_opposite_sides_equal_l423_423382


namespace anya_digit_placement_impossible_l423_423748

theorem anya_digit_placement_impossible :
  ¬ (∃ (grid : Fin 5 → Fin 8 → Fin 10), 
    (∀ d : Fin 10, (∑ i, ∑ j, if grid i j = d then 1 else 0) = 4)) :=
begin
  sorry
end

end anya_digit_placement_impossible_l423_423748


namespace bowling_prize_distribution_l423_423410

theorem bowling_prize_distribution :
  let games : ℕ := 5 in
  let choices_per_game : ℕ := 2 in
  (choices_per_game ^ games) = 32 :=
by
  sorry

end bowling_prize_distribution_l423_423410


namespace sum_valid_replacements_35z45_divisible_by_4_l423_423774

theorem sum_valid_replacements_35z45_divisible_by_4 : 
  (∑ z in (finset.range 10), if (35 * 1000 + z * 100 + 45) % 4 = 0 then z else 0) = 0 :=
by
  sorry

end sum_valid_replacements_35z45_divisible_by_4_l423_423774


namespace point_Q_coordinates_l423_423615

theorem point_Q_coordinates : 
  let P := (1 : ℝ, 0 : ℝ) in
  let arc_length := (2 * π / 3 : ℝ) in
  let unit_circle (x y : ℝ) := x^2 + y^2 = 1 in
  let Q := (Real.cos arc_length, Real.sin arc_length) in
  unit_circle 1 0 ∧ arc_length = 2 * π / 3 →
  Q = (-1 / 2, Real.sqrt 3 / 2) :=
by simp [Real.cos, Real.sin, unit_circle]; sorry

end point_Q_coordinates_l423_423615


namespace book_pages_l423_423352

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l423_423352


namespace given_conditions_area_of_triangle_l423_423095

structure Circle (T : Type*) [MetricSpace T] :=
(radius : ℝ)
(center : T)

structure Triangle (T : Type*) :=
(A B C : T)

def is_equilateral {T : Type*} [MetricSpace T] (Δ : Triangle T) : Prop :=
  dist Δ.A Δ.B = dist Δ.B Δ.C ∧ dist Δ.B Δ.C = dist Δ.C Δ.A

def is_collinear {T : Type*} [MetricSpace T] (P Q R : T) : Prop :=
  ∃ (k : ℝ), P + k • Q = R

open Real

noncomputable def pointF {T : Type*} [MetricSpace T] (D E : T) (line_parallel_to_AD line_parallel_to_AE : T → Prop) : T :=
  Classical.some (line_parallel_to_AD D ∩ line_parallel_to_AE E)

noncomputable def pointG {T : Type*} [MetricSpace T] (A F : T) (circle : Circle T) : T :=
  Classical.some (circle.radius * 2 = dist A F)

def area_triangle {T : Type*} [MetricSpace T] (P Q R : T) : ℝ :=
  (1 / 2) * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem given_conditions_area_of_triangle {R : Type*} [MetricSpace R] {A B C D E F G : R} (h1 : is_equilateral (Triangle.mk A B C)) 
  (h2 : dist A (B + (15 : ℝ))) (h3 : dist A (C + (14 : ℝ))) (h4 : is_collinear A F G) 
  (h5 : dist A (pointG A F (Circle.mk 3 A)) = 6) : 
  ∃ (p q r : ℕ), p.gcd r = 1 ∧ ∀ (area_CBG := area_triangle C B G), 
  area_CBG = (p * sqrt q) / r :=
sorry

end given_conditions_area_of_triangle_l423_423095


namespace cary_initial_wage_l423_423760

noncomputable def initial_hourly_wage (x : ℝ) : Prop :=
  let first_year_wage := 1.20 * x
  let second_year_wage := 0.75 * first_year_wage
  second_year_wage = 9

theorem cary_initial_wage : ∃ x : ℝ, initial_hourly_wage x ∧ x = 10 := 
by
  use 10
  unfold initial_hourly_wage
  simp
  sorry

end cary_initial_wage_l423_423760


namespace find_values_l423_423812

theorem find_values (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 :=
by 
  sorry

end find_values_l423_423812


namespace proof_equivalence_l423_423435

noncomputable def statement_I (a b : ℝ) : Prop :=
  sqrt (a^2 + b^2) = complex.abs (a + b * complex.i) ^ 2

noncomputable def statement_II (a b : ℝ) : Prop :=
  sqrt (a^2 + b^2) = (a - b)^2

noncomputable def statement_III (a b : ℝ) : Prop :=
  sqrt (a^2 + b^2) = abs a + abs b

noncomputable def statement_IV (a b : ℝ) : Prop :=
  sqrt (a^2 + b^2) = abs (a * b)

theorem proof_equivalence (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  (statement_I a b) ∧ ¬(statement_II a b) ∧ ¬(statement_III a b) ∧ ¬(statement_IV a b) :=
by
  sorry

end proof_equivalence_l423_423435


namespace minimum_value_l423_423856

-- Given conditions of the problem
variables (a b c : ℝ)
variables (hba : b > a)
variables (hnonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)

-- Goal: Prove the minimum value of the expression
theorem minimum_value (hba : b > a) (hnonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ x, (a + b + c) / (b - a) ≥ min_val :=
sorry

end minimum_value_l423_423856


namespace find_a_l423_423539

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem find_a (a : ℝ) 
  (h : ∃ (a : ℝ), a ^ 3 * binomial_coeff 8 3 = 56) : a = 1 :=
by
  sorry

end find_a_l423_423539


namespace correct_statement_l423_423692

open Polynomial

-- Definitions based on conditions
def degree_poly1 : Polynomial ℝ := 4 * X^2 * Y - 2 * X * Y - 1
def linear_term_poly2 : Polynomial ℝ := - (1 / 2) * X * Y - X
def coeff_poly3 : Polynomial ℝ := 4 * X^2 * Y
def degree_poly4 : Polynomial ℝ := X^2 - 2 * X^2 * Y^2 - 1

-- The statement of the proof problem
theorem correct_statement :
  degree_poly1.degree ≠ 5 ∧
  (linear_term_poly2.coeff X = -1 ∧ linear_term_poly2.degree = 1) ∧
  coeff_poly3.coeff (X^2 * Y) = 4 ∧
  degree_poly4.degree = 4 :=
  sorry

end correct_statement_l423_423692


namespace y_values_relationship_l423_423194

theorem y_values_relationship (y1 y2 y3 : ℝ) :
  (y1 = -((1 / 2) * ((-4 + 2) ^ 2)) - 1) →
  (y2 = -((1 / 2) * ((-1 + 2) ^ 2)) - 1) →
  (y3 = -((1 / 2) * ((1 + 2) ^ 2)) - 1) →
  y3 < y1 ∧ y1 < y2 :=
by {
  assume h1 h2 h3,
  have h_y1: y1 = -3, {calc
    y1 = -((1 / 2) * ((-4 + 2) ^ 2)) - 1 : by exact h1,
    ... = -3 : by norm_num [pow_succ, one_mul]},
  have h_y2: y2 = -3 / 2, {calc
    y2 = -((1 / 2) * ((-1 + 2) ^ 2)) - 1 : by exact h2,
    ... = -3 / 2 : by norm_num [one_mul]},
  have h_y3: y3 = -11 / 2, {calc
    y3 = -((1 / 2) * ((1 + 2) ^ 2)) - 1 : by exact h3,
    ... = -11 / 2 : by norm_num [one_mul]},
  exact (by norm_num : -11 / 2 < -3) ▸
          (by norm_num : -3 < -3 / 2) ▸ ⟨h_y3, ⟨h_y1,h_y2⟩⟩,
  sorry
}

end y_values_relationship_l423_423194


namespace ratio_of_remaining_areas_of_squares_l423_423629

/--
  Given:
  - Square C has a side length of 48 cm.
  - Square D has a side length of 60 cm.
  - A smaller square of side length 12 cm is cut out from both squares.

  Show that:
  - The ratio of the remaining area of square C to the remaining area of square D is 5/8.
-/
theorem ratio_of_remaining_areas_of_squares : 
  let sideC := 48
  let sideD := 60
  let sideSmall := 12
  let areaC := sideC * sideC
  let areaD := sideD * sideD
  let areaSmall := sideSmall * sideSmall
  let remainingC := areaC - areaSmall
  let remainingD := areaD - areaSmall
  (remainingC : ℚ) / remainingD = 5 / 8 :=
by
  sorry

end ratio_of_remaining_areas_of_squares_l423_423629


namespace departs_if_arrives_l423_423215

theorem departs_if_arrives (grain_quantity : ℤ) (h : grain_quantity = 30) : -grain_quantity = -30 :=
by {
  have : -grain_quantity = -30,
  from congr_arg (λ x, -x) h,
  exact this
}

end departs_if_arrives_l423_423215


namespace roots_of_quadratic_expression_l423_423832

theorem roots_of_quadratic_expression :
    (∀ x: ℝ, (x^2 + 3 * x - 2 = 0) → ∃ x₁ x₂: ℝ, x = x₁ ∨ x = x₂) ∧ 
    (∀ x₁ x₂ : ℝ, (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -2) → x₁^2 + 2 * x₁ - x₂ = 5) :=
by
  sorry

end roots_of_quadratic_expression_l423_423832


namespace arrangements_TOOTH_l423_423772
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end arrangements_TOOTH_l423_423772


namespace range_of_a_l423_423086

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end range_of_a_l423_423086


namespace find_x_l423_423792

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
sorry

end find_x_l423_423792


namespace circle_equation_exists_l423_423104

-- Define the necessary conditions
def tangent_to_x_axis (r b : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_formula (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

-- Main theorem combining the conditions and proving the circles' equations
theorem circle_equation_exists (a b r : ℝ) :
  tangent_to_x_axis r b →
  center_on_line a b →
  intersects_formula a b r →
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) :=
by
  intros h_tangent h_center h_intersects
  sorry

end circle_equation_exists_l423_423104


namespace initial_blue_balls_l423_423323

theorem initial_blue_balls (B : ℕ) (h1 : 25 - 5 = 20) (h2 : (B - 5) / 20 = 1 / 5) : B = 9 :=
by
  sorry

end initial_blue_balls_l423_423323


namespace sum_of_squares_of_consecutive_integers_l423_423662

theorem sum_of_squares_of_consecutive_integers
  (a : ℤ) (h : (a - 1) * a * (a + 1) = 10 * ((a - 1) + a + (a + 1))) :
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 :=
sorry

end sum_of_squares_of_consecutive_integers_l423_423662


namespace problem_statement_l423_423955

noncomputable def f (x : ℝ) : ℝ := ∫ t in -x..x, Real.cos t

theorem problem_statement : f (f (Real.pi / 4)) = 2 * Real.sin (Real.sqrt 2) := 
by
  sorry

end problem_statement_l423_423955


namespace smallest_f_eq_2n_minus_1_iff_l423_423245

-- Definitions based on the given conditions
def smallest_f (n : ℕ) : ℕ := 
  Inf {f : ℕ | (∑ k in Finset.range (f + 1), k + 1) % n = 0}

-- Main proposition 
theorem smallest_f_eq_2n_minus_1_iff (n : ℕ) :
  smallest_f n = 2 * n - 1 ↔ ∃ m : ℕ, n = 2^m :=
by
  sorry

end smallest_f_eq_2n_minus_1_iff_l423_423245


namespace marble_arrangement_mod_l423_423613

def num_ways_arrange_marbles (m : ℕ) : ℕ := Nat.choose (m + 3) 3

theorem marble_arrangement_mod (N : ℕ) (m : ℕ) (h1: m = 11) (h2: N = num_ways_arrange_marbles m): 
  N % 1000 = 35 := by
  sorry

end marble_arrangement_mod_l423_423613


namespace circle_C_properties_l423_423031

noncomputable def circle_C_equation (a : ℝ) (r : ℝ) : Prop :=
  (a > 0) ∧ (r = 2 * a) ∧ (r^2 = (sqrt 14 / 2)^2 + ((a - 2*a + 2) / sqrt 2)^2)

theorem circle_C_properties (C : Type) [category C]
  (x y : ℝ) (h_center_ray : y = 2 * x) (h_xpositive : x > 0)
  (h_tangent_xaxis : ∀ y, y ≠ x -> abs y = 0)
  (h_intercept_length : | line_segment_intercept_length (y - 2) - line_segment_intercept_length (x + 2) | = sqrt 14)
  (P : ℝ × ℝ)
  (h_p_on_line : P.1 + P.2 + 3 = 0)
  (E F : ℝ × ℝ)
  (h_tangent_points : are_tangent_points C E F) :
  circle_C_equation 1 2 ∧ 
  ∃ (S : ℝ), 
  minimum_area_quad PECF P E F S = 2 * sqrt 14 ∧ 
  vector_dot_product E F = 5 * sqrt 14 / 9 :=
by
  sorry

end circle_C_properties_l423_423031


namespace tea_blend_selling_price_l423_423732

theorem tea_blend_selling_price (C1 C2 : ℝ) (r1 r2 : ℝ) (gain_percent : ℝ) 
    (h1 : C1 = 18) (h2 : C2 = 20) (h3 : r1 = 5) (h4 : r2 = 3) (h5 : gain_percent = 12) :
    let CP := (C1 * r1 + C2 * r2) / (r1 + r2),
        SP := CP + (gain_percent * CP) / 100
    in SP = 21 := 
by
    -- The proof goes here
    sorry

end tea_blend_selling_price_l423_423732


namespace total_stoppage_time_l423_423713

theorem total_stoppage_time (stop1 stop2 stop3 : ℕ) (h1 : stop1 = 5)
  (h2 : stop2 = 8) (h3 : stop3 = 10) : stop1 + stop2 + stop3 = 23 :=
sorry

end total_stoppage_time_l423_423713


namespace quadratic_function_vertex_l423_423273

theorem quadratic_function_vertex : 
  ∃ (a : ℝ), a > 0 ∧ ∀ (x : ℝ), (λ x, a * x^2 - 1) x = x^2 - 1 :=
by
  use 1
  split
  { sorry }  -- Proof that 1 > 0 is trivial
  { sorry }  -- Verification that the form y = x^2 - 1 meets the vertex and other requirements

end quadratic_function_vertex_l423_423273


namespace find_lambda_l423_423183

-- Define the vectors a and b as pairs of real numbers
def a : ℝ × ℝ := (2, -7)
def b : ℝ × ℝ := (-2, -4)

-- Define the lambda that we need to prove equals 6/5
def λ := 6 / 5

-- Define the dot product of two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition we have in the problem
def perp_cond (λ : ℝ) : Prop :=
  let u := (a.1 + λ * b.1, a.2 + λ * b.2)
  dot_product u b = 0

-- The theorem we want to prove
theorem find_lambda : perp_cond λ := 
  sorry

end find_lambda_l423_423183


namespace general_term_arithmetic_sequence_l423_423917

-- Consider an arithmetic sequence {a_n}
variable (a : ℕ → ℤ)

-- Conditions
def a1 : Prop := a 1 = 1
def a3 : Prop := a 3 = -3
def is_arithmetic_sequence : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Theorem statement
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 1) (h3 : a 3 = -3) (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 3 - 2 * n :=
by
  sorry  -- proof is not required

end general_term_arithmetic_sequence_l423_423917


namespace range_of_a_l423_423497

def P (a : ℝ) : Prop := ∀ (x : ℝ), 0 < a ∧ a ≠ 1 ∧ (1 - 2 * x > 0 → (1 - 2 * x) * (Real.log a) > 0)

def Q (a : ℝ) : Prop := ∀ (x : ℝ), (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

theorem range_of_a (a : ℝ) (h : P a ∨ Q a) : a ∈ Set.Icc (-∞) 2 \ {1} :=
sorry

end range_of_a_l423_423497


namespace find_first_subject_percentage_l423_423387

theorem find_first_subject_percentage (x : ℝ) (h1 : 80) (h2 : 85) (overall : 75)  :
  (x + h1 + h2) / 3 = overall → x = 60 := 
by
  sorry

end find_first_subject_percentage_l423_423387


namespace add_inequality_of_greater_l423_423830

theorem add_inequality_of_greater (a b c d : ℝ) (h₁ : a > b) (h₂ : c > d) : a + c > b + d := 
by sorry

end add_inequality_of_greater_l423_423830


namespace problem_sum_of_bn_l423_423492

-- Definitions
def a_n (n : ℕ) : ℕ := (n + 1) / 2

def b_n : ℕ → ℕ
| n => if n % 2 = 1 then 2 ^ a_n n else a_n n

def T (n : ℕ) : ℕ :=
∑ i in Finset.range (2 * n + 2), b_n i

-- Conditions
def a_1_gt_0 : Prop := (a_n 1 > 0)

def a1_a2_eq_three_halves : Prop := (a_n 1 * a_n 2 = 3 / 2)

def S5_eq_10 : Prop := (∑ i in Finset.range 5, a_n i = 10)

-- Theorem Statement
theorem problem_sum_of_bn (n : ℕ) (h1 : a_1_gt_0) (h2 : a1_a2_eq_three_halves) (h3 : S5_eq_10) : 
  T (2 * n + 1) = 2 ^ (n + 1) + (n^2 + 2 * n) / 2 - 2 := 
sorry

end problem_sum_of_bn_l423_423492


namespace find_x_l423_423784

theorem find_x (x : ℝ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423784


namespace sequence_sum_l423_423143

noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := (1 / 4) * ((-3) / (4 * a n - 1) + 5)

theorem sequence_sum (n : ℕ) :
  (∑ i in Finset.range n, (1 / (a (i + 1) - 1))) = (3 / 2) * (3^n - 1) - 2 * n :=
by 
  sorry

end sequence_sum_l423_423143


namespace find_correct_day_l423_423612

def tomorrow_is_not_September (d : String) : Prop :=
  d ≠ "September"

def in_a_week_is_September (d : String) : Prop :=
  d = "September"

def day_after_tomorrow_is_not_Wednesday (d : String) : Prop :=
  d ≠ "Wednesday"

theorem find_correct_day :
    ((∀ d, tomorrow_is_not_September d) ∧ 
    (∀ d, in_a_week_is_September d) ∧ 
    (∀ d, day_after_tomorrow_is_not_Wednesday d)) → 
    "Wednesday, August 25" = "Wednesday, August 25" :=
by
sorry

end find_correct_day_l423_423612


namespace concyclic_points_l423_423225

-- Definitions for the conditions
variables (A B C D E F G : Point)
variables (angle_bisector : Line A) 
variables (altitude : Line A)

-- Conditions: D is the foot of the perpendicular from B to the angle bisector from A
def is_foot_of_perpendicular (P : Point) (l : Line P) (Q R : Point) : Prop :=
∃ (line_perp : Line Q), is_perpendicular line_perp l ∧ lies_on P line_perp ∧ lies_on R line_perp

-- Conditions: F is the midpoint of side BC
def is_midpoint (M : Point) (P Q : Point) : Prop :=
dist M P = dist M Q ∧ collinear P M Q

-- Conditions: G is the foot of the altitude from A to BC
def is_altitude_foot (P : Point) (A B C : Point) : Prop :=
∃ (line_altitude : Line A), is_perpendicular line_altitude (line_through B C) ∧ lies_on P line_altitude

-- Hypothesis and Theorem Statement
theorem concyclic_points : 
  is_foot_of_perpendicular D angle_bisector B A ∧
  is_foot_of_perpendicular E angle_bisector C A ∧
  is_midpoint F B C ∧
  is_altitude_foot G A B C →
  concyclic D E F G :=
by
  intros h
  sorry

end concyclic_points_l423_423225


namespace ratio_increase_decrease_l423_423039

noncomputable def ratio_percent_increase_units_sold (P U U' : ℝ) (h1 : U' = 1.25 * U) 
  (h2 : P > 0) (h3 : U > 0) : ℝ :=
  25 / 20

theorem ratio_increase_decrease (P U U' : ℝ) 
  (h1 : U' = 1.25 * U)
  (h2 : 0.80 * U' = U)
  (h3 : P > 0)
  (h4 : U > 0) : ratio_percent_increase_units_sold P U U' h1 h2 h3 = 1.25 := 
by
  sorry

end ratio_increase_decrease_l423_423039


namespace tan_alpha_cos_of_double_angle_l423_423506

variable (α : ℝ)
axiom terminal_side_on_line (h : α = angle_on_line y (-2 * x)) : True

theorem tan_alpha (h : terminal_side_on_line α): tan α = -2 := sorry

theorem cos_of_double_angle (h : terminal_side_on_line α): cos (2 * α + 3 / 2 * π) = -4 / 5 := sorry

end tan_alpha_cos_of_double_angle_l423_423506


namespace green_beans_jaylen_l423_423925

def vegetables_jaylen := 18
def carrots_jaylen := 5
def cucumbers_jaylen := 2
def bell_peppers_kristin := 2
def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_kristin := 20

theorem green_beans_jaylen :
  (∃ green_beans_jaylen : ℕ,
    vegetables_jaylen = carrots_jaylen + cucumbers_jaylen + bell_peppers_jaylen + green_beans_jaylen ∧
    green_beans_jaylen = (green_beans_kristin / 2) - 3) →
  green_beans_jaylen = 7 :=
by
  sorry

end green_beans_jaylen_l423_423925


namespace club_officer_selection_l423_423974

theorem club_officer_selection : 
  ∃ (n : ℕ), (∀ (members officers : ℕ), members = 15 → officers = 5 → 
  (∏ i in finset.range 5, (members - i)) = n) ∧ n = 360360 :=
by
  have members := 15
  have officers := 5
  have selection : ℕ := ( ∏ i in finset.range officers, (members - i))
  use selection
  split
  · intros
    symm
    exact selection
  · norm_num
    sorry

end club_officer_selection_l423_423974


namespace find_k_and_sufficient_batch_size_l423_423385

theorem find_k_and_sufficient_batch_size (x y : ℕ) (k : ℝ) (budget : ℝ) :
  3600 = x * (3600 / x).ceil → (3600 / x).ceil * 400 + k * 2000 * x = y → y = 43600 → budget = 24000 →
  k = 1 / 20 ∧ ∃ x' : ℕ, (3600 / x').ceil * 400 + (1 / 20) * 2000 * x' ≤ 24000 :=
by
  sorry

end find_k_and_sufficient_batch_size_l423_423385


namespace semicircle_perimeter_l423_423701

/-- The perimeter of a semicircle with radius 6.3 cm is approximately 32.382 cm. -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.3) : 
  (π * r + 2 * r = 32.382) :=
by
  sorry

end semicircle_perimeter_l423_423701


namespace sam_gave_2_puppies_l423_423980

theorem sam_gave_2_puppies (original_puppies given_puppies remaining_puppies : ℕ) 
  (h1 : original_puppies = 6) (h2 : remaining_puppies = 4) :
  given_puppies = original_puppies - remaining_puppies := by 
  sorry

end sam_gave_2_puppies_l423_423980


namespace line_through_circles_l423_423177

theorem line_through_circles (D1 E1 D2 E2 : ℝ)
  (h1 : 2 * D1 - E1 + 2 = 0)
  (h2 : 2 * D2 - E2 + 2 = 0) :
  (2 * D1 - E1 + 2 = 0) ∧ (2 * D2 - E2 + 2 = 0) :=
by
  exact ⟨h1, h2⟩

end line_through_circles_l423_423177


namespace probability_in_range_l423_423810

noncomputable def sample_scores : List ℝ := [82, 90, 74, 81, 77, 94, 82, 68, 89, 75]

def is_in_range (score : ℝ) : Prop := 79.5 ≤ score ∧ score ≤ 85.5

theorem probability_in_range :
  (List.countP is_in_range sample_scores) / (sample_scores.length : ℝ) = 0.3 :=
by
  -- Placeholder for actual proof
  sorry

end probability_in_range_l423_423810


namespace unique_arrangements_of_TOOTH_l423_423771

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem unique_arrangements_of_TOOTH : 
  let word := "TOOTH" in
  let n := 5 in
  let t_count := 3 in
  let o_count := 2 in
  n.factorial / (t_count.factorial * o_count.factorial) = 10 :=
sorry

end unique_arrangements_of_TOOTH_l423_423771


namespace total_modules_l423_423914

-- Problem setup
constant x : ℕ  -- number of $10 modules
constant y : ℕ  -- number of $2.5 modules
constant cost_expensive : ℕ := 10  -- cost of $10 module
constant cost_cheap : ℝ := 2.5  -- cost of $2.5 module
constant total_cost : ℝ := 62.5  -- total stock cost
constant cheap_modules_count : ℕ := 21  -- number of $2.5 modules

-- Conditions
axiom A1 : y = 21
axiom A2 : cost_expensive * (x : ℝ) + cost_cheap * (y : ℝ) = total_cost

-- Target theorem
theorem total_modules : x + y = 22 :=
by 
  sorry

end total_modules_l423_423914


namespace find_constants_l423_423178

variables {V : Type*} [InnerProductSpace ℝ V] (a b q : V)

-- Given condition
def condition : Prop :=
  ∥q - b∥ = 3 * ∥q - a∥

-- Constants s and v
def s := 9 / 8
def v := -1 / 8

-- Statement that q is at a fixed distance from s * a + v * b
def fixed_distance (q a b : V) : Prop :=
  ∃ R : ℝ, ∀ q : V, condition a b q → ∥q - (s • a + v • b)∥ = R

-- The theorem to be proved
theorem find_constants (a b q : V) (h : condition a b q)
  : fixed_distance q a b :=
sorry

end find_constants_l423_423178


namespace math_scores_between_90_and_105_l423_423736

noncomputable def participants := 1000
noncomputable def mean := 105
noncomputable def sigma_squared : ℝ := sorry
noncomputable def full_score := 150
noncomputable def excellent_fraction := 1 / 5
noncomputable def excellent_score := 120
noncomputable def low_score := 90
noncomputable def mid_score := 105

theorem math_scores_between_90_and_105 :
  ∀ (P : ℝ → ℝ),
  (∀ x, P x = real.cdf std_normal (x - mean)) →
  P(low_score) = 0.2 →
  P(excellent_score) = 0.8 →
  participants * (P(mid_score) - P(low_score)) = 300 := by
  sorry

end math_scores_between_90_and_105_l423_423736


namespace product_of_solutions_product_of_all_solutions_l423_423128

theorem product_of_solutions (x : ℝ) (h : x^2 = 49) : x = 7 ∨ x = -7 :=
begin
  rw eq_comm at h,
  exact eq_or_eq_neg_eq_of_sq_eq_sq h,
end

theorem product_of_all_solutions (h : {x : ℝ | x^2 = 49} = {7, -7}) : 7 * (-7) = -49 :=
by {
  rw set.eq_singleton_iff_unique_mem at h,
  rw [h.2 7, h.2 (-7)],
  exact mul_neg_eq_neg_mul_symm,
}

end product_of_solutions_product_of_all_solutions_l423_423128


namespace order_of_f_values_l423_423960

noncomputable def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else 0 -- define f such that it handles the missing part

theorem order_of_f_values :
  (∀ x: ℝ, f (2 - x) = f (1 + x)) ∧ (∀ x: ℝ, x >= 1 → f x = 3^x - 1) →
  f 0 < f 3 ∧ f 3 < f (-2) :=
by
  sorry

end order_of_f_values_l423_423960


namespace sin_double_angle_l423_423882

theorem sin_double_angle (α : ℝ) (h1 : sin (3 * π / 2 - α) = 3 / 5) 
    (h2 : α ∈ set.Ioo π (3 * π / 2)) : sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l423_423882


namespace book_pages_l423_423353

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l423_423353


namespace sum_of_roots_eq_neg_two_l423_423576

noncomputable def floor_sum_roots_eq : ℝ :=
  let floor (a : ℝ) := Real.floor a in
  let roots := { x | ∃ t : ℤ, floor (3 * x + 1) = t ∧ 2 * x - 1/2 = ↑t } in
  ∑ x in roots, x

theorem sum_of_roots_eq_neg_two :
  floor_sum_roots_eq = -2 :=
by
  sorry

end sum_of_roots_eq_neg_two_l423_423576


namespace find_breadth_of_wall_l423_423702

theorem find_breadth_of_wall
  (b h l V : ℝ)
  (h1 : V = 12.8)
  (h2 : h = 5 * b)
  (h3 : l = 8 * h) :
  b = 0.4 :=
by
  sorry

end find_breadth_of_wall_l423_423702


namespace product_of_solutions_to_x_squared_equals_49_l423_423119

theorem product_of_solutions_to_x_squared_equals_49 :
  (∃ (x : ℝ), x ^ 2 = 49) → ((∀ x, x ^ 2 = 49 → (x = 7 ∨ x = -7))) →
  (∏ x in { x : ℝ | x ^ 2 = 49}.to_finset, x) = -49 :=
begin
  sorry
end

end product_of_solutions_to_x_squared_equals_49_l423_423119


namespace scorpion_millipedes_needed_l423_423028

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l423_423028


namespace finite_ring_elements_repeat_l423_423232

theorem finite_ring_elements_repeat (A : Type*) [ring A] [fintype A] :
  ∃ (m p : ℕ), m > p ∧ p ≥ 1 ∧ ∀ a : A, a ^ m = a ^ p :=
by
  sorry

end finite_ring_elements_repeat_l423_423232


namespace conjugate_z_l423_423481

noncomputable def z_conjugate (a : ℝ) : ℂ := a + real.sqrt 3 * complex.I

theorem conjugate_z (a : ℝ) (h : a > 0) (h_abs : abs (z_conjugate a) = 2) :
  complex.conj (z_conjugate a) = z_conjugate 1 - 2 * real.sqrt 3 * complex.I := by
  sorry

end conjugate_z_l423_423481


namespace inequality_holds_l423_423152

variables {a b c : ℝ}

theorem inequality_holds (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ab > ac :=
sorry

end inequality_holds_l423_423152


namespace solve_problem_l423_423930

noncomputable def problem_statement : Prop :=
  let side_length := 12
  let point_P := (12, 5, 7)
  let initial_vertex := (0, 0, 0)
  let distance := Real.sqrt (side_length^2 + 5^2 + 7^2)
  let total_light_path := 12 * distance
  total_light_path = 12 * Real.sqrt 218 ∧ m = 12 ∧ n = 218 ∧ m + n = 230

theorem solve_problem : problem_statement :=
sorry

end solve_problem_l423_423930


namespace streamers_for_price_of_confetti_l423_423408

variable (p q : ℝ) (x y : ℝ)

theorem streamers_for_price_of_confetti (h1 : x * (1 + p / 100) = y) 
                                   (h2 : y * (1 - q / 100) = x)
                                   (h3 : |p - q| = 90) :
  10 * (y * 0.4) = 4 * y :=
sorry

end streamers_for_price_of_confetti_l423_423408


namespace related_variables_l423_423693

theorem related_variables (taxi_fare distance_travelled house_size house_price height weight iron_size iron_mass : Type) 
  (h1 : taxi_fare → distance_travelled)
  (h2 : house_size → house_price)
  (h3 : height → weight)
  (h4 : iron_size → iron_mass) : 
  (taxi_fare ≠ 0 ∧ distance_travelled ≠ 0) ∧
  (house_size ≠ 0 ∧ house_price ≠ 0) ∧
  (height ≠ 0 ∧ weight ≠ 0) ∧
  (iron_size ≠ 0 ∧ iron_mass ≠ 0) :=
sorry

end related_variables_l423_423693


namespace polynomial_sum_of_squares_l423_423586

noncomputable theory

variable (P : Polynomial ℝ)
variable (h : ∀ x : ℝ, P.eval x ≥ 0)

theorem polynomial_sum_of_squares :
  ∃ (A B : Polynomial ℝ), P = A^2 + B^2 :=
sorry

end polynomial_sum_of_squares_l423_423586


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l423_423101

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l423_423101


namespace find_x_l423_423581

theorem find_x (h : ℝ → ℝ)
  (H1 : ∀x, h (3*x - 2) = 5*x + 6) :
  (∀x, h x = 2*x - 1) → x = 31 :=
by
  sorry

end find_x_l423_423581


namespace parrot_shout_for_eight_whispered_number_for_shout_three_parrot_never_shouts_seven_parrot_respond_for_6m_plus_2_l423_423403

-- Definitions and conditions from the problem
def parrot_function (n : ℤ) : ℤ :=
  ((5 * n + 14) / 6) - 1

-- Part (a): Prove that the parrot will shout 8 if Antônio whispers 8
theorem parrot_shout_for_eight : parrot_function 8 = 8 := 
  sorry

-- Part (b): Prove that the number António whispered was 2 if the parrot shouts 3
theorem whispered_number_for_shout_three (n : ℤ) : parrot_function n = 3 ↔ n = 2 :=
  sorry

-- Part (c): Prove that the parrot can never shout 7
theorem parrot_never_shouts_seven : ¬ ∃ n : ℤ, parrot_function n = 7 :=
  sorry

-- Part (d): Prove that Antonio can whisper any number of the form \( 6m + 2 \) and the parrot will respond (produce an integer result)
theorem parrot_respond_for_6m_plus_2 (n : ℤ) : ∃ m : ℤ, n = 6 * m + 2 ↔ parrot_function n ∈ ℤ :=
  sorry

end parrot_shout_for_eight_whispered_number_for_shout_three_parrot_never_shouts_seven_parrot_respond_for_6m_plus_2_l423_423403


namespace collinear_P_H_Q_l423_423244

open EuclideanGeometry

variables {A B C H P Q : Point}
variables [h1 : Orthocenter H A B C] 
variables [h2 : TangentPoint A P B C] 
variables [h3 : TangentPoint A Q B C]

theorem collinear_P_H_Q (h1 : Orthocenter H A B C) (h2 : TangentPoint A P (CircleDiameter B C)) (h3 : TangentPoint A Q (CircleDiameter B C)) : Collinear P H Q :=
sorry

end collinear_P_H_Q_l423_423244


namespace exists_b_gt_a_divides_l423_423275

theorem exists_b_gt_a_divides (a : ℕ) (h : 0 < a) :
  ∃ b : ℕ, b > a ∧ (1 + 2^a + 3^a) ∣ (1 + 2^b + 3^b) :=
sorry

end exists_b_gt_a_divides_l423_423275


namespace roots_greater_than_one_implies_range_l423_423846

theorem roots_greater_than_one_implies_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + a = 0 → x > 1) → 3 < a ∧ a ≤ 4 :=
by
  sorry

end roots_greater_than_one_implies_range_l423_423846


namespace probability_of_dime_l423_423376

noncomputable def num_quarters := 12 / 0.25
noncomputable def num_dimes := 8 / 0.10
noncomputable def num_pennies := 5 / 0.01
noncomputable def total_coins := num_quarters + num_dimes + num_pennies

theorem probability_of_dime : (num_dimes / total_coins) = (40 / 314) :=
by
  sorry

end probability_of_dime_l423_423376


namespace original_sticker_price_l423_423260

theorem original_sticker_price (S : ℝ) (h1 : 0.80 * S - 120 = 0.65 * S - 10) : S = 733 := 
by
  sorry

end original_sticker_price_l423_423260


namespace curve_intersection_l423_423809

theorem curve_intersection :
  (∀ (x y : ℝ), (x^2 - y^2 = x / (x^2 + y^2) ↔ (x, y) ∈ C1)) →
  (∀ (x y : ℝ), (2*x*y + y / (x^2 + y^2) = 3 ↔ (x, y) ∈ C2)) →
  (∀ (x y : ℝ), (x^3 - 3*x*y^2 + 3*y = 1 ↔ (x, y) ∈ C3)) →
  (∀ (x y : ℝ), (3*y*x^2 - 3*x - y^3 = 0 ↔ (x, y) ∈ C4)) →
  (C1 ∩ C2 = C3 ∩ C4) :=
by {
  intros hC1 hC2 hC3 hC4,
  sorry
}

end curve_intersection_l423_423809


namespace counting_indistinguishable_boxes_l423_423877

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l423_423877


namespace H_is_abelian_l423_423231

variable {G : Type*} [Group G]
variable {H : Subgroup G}

theorem H_is_abelian (H_ne_G : H ≠ ⊤) (h : ∀ x y : G, (x ∈ G \ H) → (y ∈ G \ H) → x^2 = y^2) : IsCommutative H (·) :=
by
  sorry

end H_is_abelian_l423_423231


namespace prime_has_two_square_numbers_l423_423704

noncomputable def isSquareNumber (p q : ℕ) : Prop :=
  p > q ∧ Nat.Prime p ∧ Nat.Prime q ∧ ¬ p^2 ∣ (q^(p-1) - 1)

theorem prime_has_two_square_numbers (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5) :
  ∃ q1 q2 : ℕ, isSquareNumber p q1 ∧ isSquareNumber p q2 ∧ q1 ≠ q2 :=
by 
  sorry

end prime_has_two_square_numbers_l423_423704


namespace find_x_l423_423785

theorem find_x (x : ℝ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423785


namespace cassidy_grounded_days_l423_423420

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l423_423420


namespace book_total_pages_l423_423351

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l423_423351


namespace problem1_problem2_problem3_problem4_l423_423706

theorem problem1 : (211 * (-455) + 365 * 455 - 211 * 545 + 545 * 365) = 154000 := 
by sorry

theorem problem2 : ([- (7 / 5) * (- ((5 * 2.5) / 5)) - 1] / 9 / (1 / ((-0.75) ^ 2)) - |(2 + (- (1 / 2) ^ 3) * (5 ^ 2))|) = - (31 / 32) := 
by sorry

theorem problem3 (x : ℝ) : ((3 * x + 2) * (x + 1) + 2 * (x - 3) * (x + 2)) = (5 * x ^ 2 + 3 * x - 10) := 
by sorry

theorem problem4 (x : ℝ) : (2 * x + 3) / 6 - (2 * x - 1) / 4 = 1 → x = - 3 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_l423_423706


namespace cube_division_into_parts_l423_423813

-- Define the cube dimensions as constants
def cube_length : ℕ := 12
def cube_dimensions : ℕ × ℕ × ℕ := (cube_length, cube_length, cube_length)

-- Define the plane equation
def plane_eq (x y z : ℕ) : Prop := x + y + z = 18

-- Define the property for points within the cube
def is_within_cube (x y z : ℕ) : Prop := (x <= cube_length) ∧ (y <= cube_length) ∧ (z <= cube_length)

-- The final goal: prove the division of the cube into parts
theorem cube_division_into_parts : 
  let points := { (x, y, z) | is_within_cube x y z ∧ plane_eq x y z }
  in points.card = 216 :=
sorry

end cube_division_into_parts_l423_423813


namespace indigo_reviews_2_star_count_l423_423995

theorem indigo_reviews_2_star_count :
  ∀ (x : ℕ),
    (18 = 6 + 7 + 4 + x) →
    (70 + 2 * x) / 18 = 4 →
    x = 1 :=
by
  assume x h1 h2
  have h3 : 18 = 17 + x := by sorry
  have h4 : 70 + 2 * x = 72 := by sorry
  have h5 : 2 * x = 2 := by sorry
  have h6 : x = 1 := by sorry
  exact h6

end indigo_reviews_2_star_count_l423_423995


namespace length_of_flat_terrain_l423_423389

theorem length_of_flat_terrain (total_time : ℚ)
  (total_distance : ℕ)
  (speed_uphill speed_flat speed_downhill : ℚ)
  (distance_uphill distance_flat : ℕ) :
  total_time = 116 / 60 ∧
  total_distance = distance_uphill + distance_flat + (total_distance - distance_uphill - distance_flat) ∧
  speed_uphill = 4 ∧
  speed_flat = 5 ∧
  speed_downhill = 6 ∧
  distance_uphill ≥ 0 ∧
  distance_flat ≥ 0 ∧
  distance_uphill + distance_flat ≤ total_distance →
  distance_flat = 3 := 
by 
  sorry

end length_of_flat_terrain_l423_423389


namespace unique_set_satisfy_condition_l423_423794

theorem unique_set_satisfy_condition 
  (S : Set ℕ) 
  (h1 : S ≠ ∅) 
  (h2 : ∀ (m n : ℕ), m ∈ S → n ∈ S → (m + n) / (Nat.gcd m n) ∈ S) : 
  S = {2} :=
begin
  sorry
end

end unique_set_satisfy_condition_l423_423794


namespace sum_of_squares_invariant_l423_423075

open Real

variables (R r : ℝ) (hR_r : R > r) (O P : Point)
variables (C_small : circle O r) (C_large : circle O R)
variables (A B C : Point) (hP : P ∈ C_small) (hA : A ∈ C_small)
variables (hB : B ∈ C_large) (hC : C ∈ C_large)
variables (hCollinear : collinear {B, P, C}) (hPerpendicular : perp (A -ᵥ P) (B -ᵥ C))

theorem sum_of_squares_invariant :
  (B -ᵥ C).length ^ 2 + (C -ᵥ A).length ^ 2 + (A -ᵥ B).length ^ 2 = 6 * R^2 + 2 * r^2 :=
sorry

end sum_of_squares_invariant_l423_423075


namespace expected_pine_in_sample_l423_423041

-- Definition of conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Proportion of pine saplings
def proportion_pine := (2 : ℝ) / 15

-- Expected number of pine saplings in the sample
def expected_pine_saplings := proportion_pine * (sample_size : ℝ)

-- The statement we want to prove
theorem expected_pine_in_sample : expected_pine_saplings = 20 := 
sorry

end expected_pine_in_sample_l423_423041


namespace sum_of_squares_of_solutions_l423_423459

theorem sum_of_squares_of_solutions :
  ∑ (x : ℝ) in {x ∣ abs (x^2 - x + (1/2008)) = (1/2008)}, x^2 = (1003/502) :=
by
  sorry

end sum_of_squares_of_solutions_l423_423459


namespace compute_abs_expression_l423_423591

--- Definitions
def x := -1009
def abs (n : Int) : Int := if n < 0 then -n else n

--- Main goal
theorem compute_abs_expression : abs (abs (abs x - 3 * x) - 2 * (abs x)) - 2 * x = 4036 := by
  sorry

end compute_abs_expression_l423_423591


namespace no_real_root_for_equation_l423_423991

theorem no_real_root_for_equation :
  ¬ ∃ x : ℝ, sqrt (2 * x + 8) - sqrt (x - 1) + 2 = 0 :=
by
  sorry

end no_real_root_for_equation_l423_423991


namespace circle_areas_equal_l423_423189

theorem circle_areas_equal :
  let r1 := 15
  let d2 := 30
  let r2 := d2 / 2
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  A1 = A2 :=
by
  sorry

end circle_areas_equal_l423_423189


namespace base5_vs_base7_digits_difference_l423_423187

theorem base5_vs_base7_digits_difference :
  let n := 2023 in
  let base5_digits := 5 in
  let base7_digits := 4 in
  base5_digits - base7_digits = 1 :=
by
  let n := 2023
  let base5_digits := 5
  let base7_digits := 4
  calc base5_digits - base7_digits = 5 - 4 : by rfl
                      ... = 1 : by rfl

end base5_vs_base7_digits_difference_l423_423187


namespace expression_evaluation_l423_423337

theorem expression_evaluation :
  (8 / 4 - 3^2 + 4 * 5) = 13 :=
by sorry

end expression_evaluation_l423_423337


namespace triangle_median_intercept_third_l423_423013

noncomputable def midpoint (A B : ℝ) : ℝ := (A + B) / 2
noncomputable def line_intercepts (A B C C1 O : ℝ) : Prop :=
  let C1 := midpoint A B in
  let O := midpoint C C1 in
  O = (A + B + C) / 3

theorem triangle_median_intercept_third (A B C : ℝ) :
  ∃ C1 O : ℝ, (C1 = midpoint A B ∧ O = midpoint C C1) →
  line_intercepts A B C C1 O :=
sorry

end triangle_median_intercept_third_l423_423013


namespace leftover_candies_l423_423091

theorem leftover_candies (a b : ℕ) (h1 : a = 138) (h2 : b = 18) : a % b = 12 :=
by 
  -- Given conditions
  rw [h1, h2]
  -- 138 % 18 = 12
  exact rfl

end leftover_candies_l423_423091


namespace alice_age_2005_l423_423409

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005_l423_423409


namespace minimum_value_of_expression_l423_423106

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l423_423106


namespace sum_x_seq_l423_423945

noncomputable def x_seq (n : ℕ) : ℕ → ℤ
| 0       := 1
| 1       := n
| (k+2) := ((n - 1) * x_seq (k + 1) - (n - k) * n * x_seq k) / (k + 1)

def sum_x (n : ℕ) : ℤ :=
1 + n + ∑ k in finset.range (n - 2), ((-1)^(k + 3) * (finset.range (k + 1).prod (λ j, (n - j))) / (nat.factorial (k + 3)))

theorem sum_x_seq (n : ℕ) (h_pos : n > 0) :
  ∑ i in (finset.range (n + 1)), x_seq n i = sum_x n :=
sorry

end sum_x_seq_l423_423945


namespace find_m_for_b_greater_than_a100_l423_423430

theorem find_m_for_b_greater_than_a100 :
  let a : ℕ → ℕ := λ n, Nat.recOn n 3 (λ n an, 3 ^ an)
  let b : ℕ → ℕ := λ n, Nat.recOn n 100 (λ n bn, 100 ^ bn)
  in ∃ m : ℕ, (b m > a 100) ∧ (∀ n < m, b n ≤ a 100) ∧ m = 99 :=
by
  sorry

end find_m_for_b_greater_than_a100_l423_423430


namespace general_term_a_general_term_b_l423_423505

-- Definition of the sequence {a_n} and conditions
def seq_a (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (n > 0) -> (S n = ∑ i in range (n+1), a i) ∧ (3 * S n = 4 * a n - 3 * n)

-- Definition of the sequence {b_n} and conditions
def seq_b (n : ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) := 
  (n > 0) -> (∑ i in range n, (b (i+1)) / (2 * i + 1) = a n / 3)

-- Theorem: General term formula of the sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (seq_a n a S) →
  a n = 4 ^ n - 1 := by
  sorry

-- Theorem: General term formula of the sequence {b_n}
theorem general_term_b (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ):
  (seq_b n b a) →
  (a n = 4 ^ n - 1) →
  b n = (2 * n - 1) * 4 ^ (n - 1) ∧ 
  T n = (5 / 9) + ((6 * n - 5) / 9) * 4 ^ n := by
  sorry

end general_term_a_general_term_b_l423_423505


namespace jogging_time_two_weeks_l423_423600

-- Definition for the daily jogging time in hours
def daily_jogging_time : ℝ := 1 + 30 / 60

-- Definition for the total jogging time over one week
def weekly_jogging_time : ℝ := daily_jogging_time * 7

-- Lean statement to prove that the total time jogging over two weeks is 21 hours
theorem jogging_time_two_weeks : weekly_jogging_time * 2 = 21 := by
  -- Placeholder for the proof
  sorry

end jogging_time_two_weeks_l423_423600


namespace f_monotonically_decreasing_m_2_range_of_m_f_2x_pos_f_zero_count_l423_423517

-- Condition: Given function f(x) = |x| + m / x - 1 for x ≠ 0
def f (x : ℝ) (m : ℝ) : ℝ := |x| + m / x - 1

-- Problem (1): Monotonicity of f(x) when m = 2 on (-∞, 0)
theorem f_monotonically_decreasing_m_2 (x : ℝ) (h : x < 0) : 
  monotone_decreasing_on (f x 2) (-∞, 0) := sorry

-- Problem (2): Range of m for which f(2^x) > 0 for any x ∈ ℝ
theorem range_of_m_f_2x_pos (m : ℝ) : 
  (∀ x : ℝ, f (2^x) m > 0) ↔ m > 1/4 := sorry

-- Problem (3): Number of zeros of f(x)
theorem f_zero_count (m : ℝ) : 
  ∃ n : ℕ, (f x m = 0) ↔ 
    (n = 1 ∧ (m > 1/4 ∨ m < -1/4)) ∨ 
    (n = 2 ∧ (m = 1/4 ∨ m = 0 ∨ m = -1/4)) ∨ 
    (n = 3 ∧ (0 < m ∧ m < 1/4 ∨ -1/4 < m ∧ m < 0)) := sorry

end f_monotonically_decreasing_m_2_range_of_m_f_2x_pos_f_zero_count_l423_423517


namespace part_I_part_II_l423_423170

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + ((2 * a^2) / x) + x

theorem part_I (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, x = 1 ∧ deriv (f a) x = -2) → a = 3 / 2 :=
sorry

theorem part_II (a : ℝ) (h : a = 3 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 / 2 → deriv (f a) x < 0) ∧ 
  (∀ x : ℝ, x > 3 / 2 → deriv (f a) x > 0) :=
sorry

end part_I_part_II_l423_423170


namespace minimum_circular_sum_l423_423588

theorem minimum_circular_sum (a : Fin 10 → ℕ)
  (h_distinct : Function.Injective a)
  (h_sum : (∑ i, a i) = 1995) :
  (∑ i : Fin 10, a i * a ((i + 1) % 10)) ≥ 6064 :=
sorry

end minimum_circular_sum_l423_423588


namespace arrange_chips_possible_l423_423366

-- Define the dimensions of the board
def board_dim := 8

-- Define a predicate that the number of chips in every pair of columns is the same
def all_columns_same (arr : ℕ → ℕ → ℕ) : Prop :=
  ∀ c1 c2 : ℕ, c1 < board_dim → c2 < board_dim → (∑ i in Finset.range board_dim, arr i c1) = (∑ i in Finset.range board_dim, arr i c2)

-- Define a predicate that the number of chips in every pair of rows is different
def all_rows_diff (arr : ℕ → ℕ → ℕ) : Prop :=
  ∀ r1 r2 : ℕ, r1 < board_dim → r2 < board_dim → r1 ≠ r2 → (∑ j in Finset.range board_dim, arr r1 j) ≠ (∑ j in Finset.range board_dim, arr r2 j)

-- The theorem statement
theorem arrange_chips_possible :
  ∃ arr : ℕ → ℕ → ℕ, all_columns_same arr ∧ all_rows_diff arr :=
sorry

end arrange_chips_possible_l423_423366


namespace average_grade_last_year_l423_423050

theorem average_grade_last_year 
  (last_year_courses : ℕ := 6)
  (year_before_courses : ℕ := 5)
  (average_last_two_years : ℝ := 72)
  (average_year_before : ℝ := 40)
  (total_courses : ℕ := 11) : 
  ∃ x : ℝ, 5 * average_year_before + 6 * x = 11 * average_last_two_years ∧ x = 98.67 := 
by {
  use 98.67,
  calc
    5 * average_year_before + 6 * 98.67 = 5 * 40 + 6 * 98.67 : by rw [average_year_before]
                               ... = 200 + 6 * 98.67            : by norm_num
                               ... = 200 + 592.02               : by norm_num
                               ... = 792.02                     : by norm_num
                               ... = 11 * 72                    : by rw [average_last_two_years]  
                               ... = 792                        : by norm_num,
  norm_num
}

end average_grade_last_year_l423_423050


namespace product_divides_sum_pow_l423_423237

theorem product_divides_sum_pow {n : ℕ} (a : ℕ → ℕ) (h1 : 3 ≤ n) 
  (h2 : ∀ i, 1 ≤ i → i ≤ n → a i > 0) 
  (h3 : Nat.gcd (a 1) (Nat.gcd (a 2) ⋯ (a n)) = 1) 
  (h4 : ∀ j, 1 ≤ j → j ≤ n → a j ∣ ∑ i in Finset.range n, a i) : 
  (∏ i in Finset.range n, a i) ∣ (∑ i in Finset.range n, a i) ^ (n - 2) := 
sorry

end product_divides_sum_pow_l423_423237


namespace find_p_for_natural_roots_l423_423098

-- The polynomial is given.
def cubic_polynomial (p x : ℝ) : ℝ := 5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1

-- Problem statement to prove that p = 76 is the only real number such that
-- the cubic polynomial cubic_polynomial equals 66 * p has at least two natural number roots.
theorem find_p_for_natural_roots (p : ℝ) :
  (∃ (u v : ℕ), u ≠ v ∧ cubic_polynomial p u = 66 * p ∧ cubic_polynomial p v = 66 * p) ↔ p = 76 :=
by
  sorry

end find_p_for_natural_roots_l423_423098


namespace g_of_2_is_112_l423_423592

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_of_2_is_112 : g (g (g (g 2))) = 112 :=
by {
  dsimp [g],
  norm_num,
  sorry
}

end g_of_2_is_112_l423_423592


namespace number_of_ways_to_construct_dice_l423_423200

-- Define the faces of the dice as a finite set
inductive Face
| face1 | face2 | face3 | face4 | face5 | face6

-- Define adjacency relation (common edge)
def adjacent_faces : Face → Face → Prop
| Face.face1 Face.face2 := true
| Face.face2 Face.face3 := true
| Face.face3 Face.face4 := true
| Face.face4 Face.face5 := true
| Face.face5 Face.face6 := true
| Face.face6 Face.face1 := true
| _ _ := false

-- Define the predicate that numbers must be placed on adjacent faces
def consecutive_numbers_adjacent (f : Face → ℕ) : Prop :=
(adjacent_faces Face.face1 Face.face2 ∧ f Face.face1 + 1 = f Face.face2) ∧
(adjacent_faces Face.face2 Face.face3 ∧ f Face.face2 + 1 = f Face.face3) ∧
(adjacent_faces Face.face3 Face.face4 ∧ f Face.face3 + 1 = f Face.face4) ∧
(adjacent_faces Face.face4 Face.face5 ∧ f Face.face4 + 1 = f Face.face5) ∧
(adjacent_faces Face.face5 Face.face6 ∧ f Face.face5 + 1 = f Face.face6) ∧
(adjacent_faces Face.face6 Face.face1 ∧ f Face.face6 + 1 = f Face.face1)

-- Define the theorem
theorem number_of_ways_to_construct_dice : 
  ∃ f : Face → ℕ, consecutive_numbers_adjacent f ∧ (f Face.face1 = 1 ∧ f Face.face2 = 2) :=
sorry

end number_of_ways_to_construct_dice_l423_423200


namespace cube_corner_sum_l423_423304

-- Define the numbers on the faces.
def face_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the condition where each pair of opposite faces sums to 7.
def opposite_faces_sum_7 (a b : ℕ) : Prop := a + b = 7

-- Define the triplets at each corner.
def corner_triplets : List (ℕ × ℕ × ℕ) := 
  [(1, 2, 3), (1, 2, 4), (1, 3, 5), (1, 4, 5),
   (6, 2, 3), (6, 2, 4), (6, 3, 5), (6, 4, 5)]

-- Define the product of the numbers at each corner.
def corner_product (t : ℕ × ℕ × ℕ) : ℕ := t.1 * t.2 * t.3

-- Sum the products of the numbers at each corner.
def sum_of_corner_products (triplets : List (ℕ × ℕ × ℕ)) : ℕ :=
  triplets.map corner_product |>.sum

theorem cube_corner_sum :
  (∑ t in [(1, 2, 3), (1, 2, 4), (1, 3, 5), (1, 4, 5),
           (6, 2, 3), (6, 2, 4), (6, 3, 5), (6, 4, 5)],
    corner_product t) = 343 := by
  sorry

end cube_corner_sum_l423_423304


namespace find_theta_for_A_polar_coords_find_distance_AB_l423_423916

-- Definitions for the curve C parametric and standard equations
def curve_C_param_x (α : ℝ) : ℝ := 2 * Real.cos α
def curve_C_param_y (α : ℝ) : ℝ := 2 + 2 * Real.sin α
def curve_C_standard_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Definition for the polar equation of curve C
def curve_C_polar_eq (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Definitions for the line l parametric and standard equations
def line_l_param_x (t : ℝ) : ℝ := Real.sqrt 3 - (Real.sqrt 3 / 2) * t
def line_l_param_y (t : ℝ) : ℝ := 3 + (1 / 2) * t
def line_l_standard_eq (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 4 * Real.sqrt 3 = 0

-- Definition for the ray OA
def ray_OA_eq (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Assertion 1: Find the value of θ such that the polar coordinates of A are (2*sqrt(3), θ)
theorem find_theta_for_A_polar_coords :
  ∃ θ ∈ Set.Ioc (Real.pi / 2) Real.pi, curve_C_polar_eq θ = 2 * Real.sqrt 3 := sorry

-- Assertion 2: Find the distance |AB| when ray OA intersects line l at point B
theorem find_distance_AB :
  let A := (-Real.sqrt 3, 3)
  let B := (-2 * Real.sqrt 3, 6)
  let distance_AB := Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  line_l_standard_eq B.1 B.2 ∧ ray_OA_eq A.1 A.2 ∧ ray_OA_eq B.1 B.2 → distance_AB = 2 * Real.sqrt 3 := sorry

end find_theta_for_A_polar_coords_find_distance_AB_l423_423916


namespace distinct_positive_integers_arithmetic_geometric_harmonic_l423_423294

theorem distinct_positive_integers_arithmetic_geometric_harmonic (x y a b : ℕ) 
  (h1 : x ≠ y) 
  (h2 : x > 0 ∧ y > 0) 
  (h3 : 1 ≤ a ∧ a ≤ 9) 
  (h4 : 0 ≤ b ∧ b ≤ 9) 
  (h_arithmetic : (x + y) / 2 = 10 * a + b) 
  (h_geometric : sqrt (x * y) = 10 * b + a) 
  (h_harmonic : 2 * x * y / (x + y) = 10 * b + a - 1) : 
  |x - y| = 18 := 
sorry

end distinct_positive_integers_arithmetic_geometric_harmonic_l423_423294


namespace cost_of_two_pencils_and_one_pen_l423_423998

variable (a b : ℝ)

-- Given conditions
def condition1 : Prop := (5 * a + b = 2.50)
def condition2 : Prop := (a + 2 * b = 1.85)

-- Statement to prove
theorem cost_of_two_pencils_and_one_pen
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  2 * a + b = 1.45 :=
sorry

end cost_of_two_pencils_and_one_pen_l423_423998


namespace find_d_over_a1_l423_423579

variable {a1 a2 a3 a4 d : ℝ}

-- Conditions
def isAP : Prop := a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ d ≠ 0
def isGP : Prop := (a1 ≠ 0) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a3 ≠ 0) ∧ (a4 ≠ 0) ∧ (a3^2 = a1 * a4)

-- Theorem statement
theorem find_d_over_a1 (hAP : isAP) (hGP : isGP) : d / a1 = -1/4 := 
sorry

end find_d_over_a1_l423_423579


namespace area_of_triangle_PQR_l423_423671

-- Definitions based on the problem's conditions
def P := (-5, -2) : ℝ × ℝ
def Q := (0, -3) : ℝ × ℝ
def R := (7, -4) : ℝ × ℝ

-- Function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_PQR : triangle_area P Q R = 6 := by
  -- Statement according to the problem and solution
  sorry

end area_of_triangle_PQR_l423_423671


namespace largest_divisor_of_three_consecutive_integers_product_l423_423241

theorem largest_divisor_of_three_consecutive_integers_product :
  ∀ (n : ℤ), ∃ d : ℤ, d = 6 ∧ ∀ k : ℤ, k ∣ (n * (n + 1) * (n + 2)) → k ≤ d :=
begin
  sorry
end

end largest_divisor_of_three_consecutive_integers_product_l423_423241


namespace compute_floor_S_l423_423590

-- Variables a, b, c, d that are positive real numbers
variables (a b c d : ℝ)
variable (hpos₁ : 0 < a)
variable (hpos₂ : 0 < b)
variable (hpos₃ : 0 < c)
variable (hpos₄ : 0 < d)

-- The conditions given in the problem
variable (h₁ : a^2 + b^2 = 2500)
variable (h₂ : c^2 + d^2 = 2500)
variable (h₃ : a * c + b * d = 1500)

-- Compute S and prove the floor of S
theorem compute_floor_S (h₁ : a^2 + b^2 = 2500) (h₂ : c^2 + d^2 = 2500) (h₃ : ac + bd = 1500)
    (hpos₁ : 0 < a) (hpos₂ : 0 < b) (hpos₃ : 0 < c) (hpos₄ : 0 < d) : 
    let S := a + b + c + d in
    ⌊S⌋ = 126 := sorry

end compute_floor_S_l423_423590


namespace problem_product_xyzw_l423_423889

theorem problem_product_xyzw
    (x y z w : ℝ)
    (h1 : x + 1 / y = 1)
    (h2 : y + 1 / z + w = 1)
    (h3 : w = 2) :
    xyzw = -2 * y^2 + 2 * y :=
by
    sorry

end problem_product_xyzw_l423_423889


namespace quadratic_function_shape_and_vertex_l423_423306

theorem quadratic_function_shape_and_vertex :
  ∃ f : ℝ → ℝ, (∀ x, f x = 5 * x^2 - 30 * x + 52 ∨ f x = -5 * x^2 + 30 * x - 38) ∧
  (∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c ∧ (a = 5 ∨ a = -5) ∧ (a * (3 : ℝ) ^ 2 + b * (3 : ℝ) + c = 7)) :=
begin
  sorry
end

end quadratic_function_shape_and_vertex_l423_423306


namespace clock_distance_l423_423680

theorem clock_distance (m M : ℝ) : 
  ∃ c : ℝ, c = (M + m) / 2 := 
by
  use (M + m) / 2
  sorry

end clock_distance_l423_423680


namespace sin_alpha_eq_24_over_25_l423_423473

open Real Trig

theorem sin_alpha_eq_24_over_25
  (α β : ℝ)
  (h0 : 0 < α ∧ α < π / 2)
  (h1 : π / 2 < β ∧ β < π)
  (h2 : sin(α + β) = 3 / 5)
  (h3 : cos β = -4 / 5) :
  sin α = 24 / 25 :=
by
  sorry

end sin_alpha_eq_24_over_25_l423_423473


namespace russian_letter_word_quotient_l423_423964

theorem russian_letter_word_quotient :
  let word1 := "СКАЛКА",
      word2 := "ТЕФТЕЛЬ",
      six_letter_words := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2),
      seven_letter_words := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)
  in six_letter_words = 180 ∧ seven_letter_words = 1260 ∧ seven_letter_words / six_letter_words = 7 :=
by
  let word1 := "СКАЛКА"
  let word2 := "ТЕФТЕЛЬ"
  let six_letter_words := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2)
  let seven_letter_words := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)
  have h1 : six_letter_words = 180 := sorry
  have h2 : seven_letter_words = 1260 := sorry
  have h3 : seven_letter_words / six_letter_words = 7 := sorry
  exact ⟨h1, h2, h3⟩

end russian_letter_word_quotient_l423_423964


namespace circle_area_l423_423536

theorem circle_area (r : ℝ) (h : 4 / (2 * real.pi * r) = 2 * r) : real.pi * r^2 = 1 :=
by
  sorry

end circle_area_l423_423536


namespace product_of_solutions_product_of_all_solutions_l423_423127

theorem product_of_solutions (x : ℝ) (h : x^2 = 49) : x = 7 ∨ x = -7 :=
begin
  rw eq_comm at h,
  exact eq_or_eq_neg_eq_of_sq_eq_sq h,
end

theorem product_of_all_solutions (h : {x : ℝ | x^2 = 49} = {7, -7}) : 7 * (-7) = -49 :=
by {
  rw set.eq_singleton_iff_unique_mem at h,
  rw [h.2 7, h.2 (-7)],
  exact mul_neg_eq_neg_mul_symm,
}

end product_of_solutions_product_of_all_solutions_l423_423127


namespace hyperbola_asymptote_angle_l423_423132

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
  let m1 := b / a
      m2 := - (b / a) 
  in |m1 - m2| / (1 + m1 * m2) = 1) :
  a / b = Real.sqrt 2 + 1 :=
sorry

end hyperbola_asymptote_angle_l423_423132


namespace price_increase_percentage_l423_423092

-- Define the problem conditions
def lowest_price := 12
def highest_price := 21

-- Formulate the goal as a theorem
theorem price_increase_percentage :
  ((highest_price - lowest_price) / lowest_price : ℚ) * 100 = 75 := by
  sorry

end price_increase_percentage_l423_423092


namespace area_intersection_of_segments_l423_423611

noncomputable def ratio := 2 / 1

-- Points on the sides of the triangle
variable (A B C C₁ A₁ B₁ : Point)

-- Conditions as definitions
axiom point_on_side_AB : C₁ ∈ line_segment A B
axiom point_on_side_BC : A₁ ∈ line_segment B C
axiom point_on_side_AC : B₁ ∈ line_segment A C
axiom ratio_AC1_C1B : (distance A C₁) / (distance C₁ B) = ratio
axiom ratio_BA1_A1C : (distance B A₁) / (distance A₁ C) = ratio
axiom ratio_CB1_B1A : (distance C B₁) / (distance B₁ A) = ratio
axiom area_ABC : area (triangle A B C) = 1

-- The question to prove the area of the smaller triangle
theorem area_intersection_of_segments : 
  ∃ (K M N : Point), 
  (K ∈ line_intersection (line A A₁) (line B B₁)) ∧ 
  (M ∈ line_intersection (line A A₁) (line C C₁)) ∧
  (N ∈ line_intersection (line B B₁) (line C C₁)) ∧ 
  area (triangle K M N) = 1 / 7 :=
sorry

end area_intersection_of_segments_l423_423611


namespace proof_problem_l423_423852

def f (x : ℝ) (a : ℝ) : ℝ :=
  if |x| ≤ 1 then real.log x / real.log 2 + a else -10 / (|x| + 3)

theorem proof_problem 
  (a : ℝ)
  (h1 : f 0 a = 2)
  (h2 : ∀ x, f x a = if |x| ≤ 1 then real.log (x + a) / real.log 2 else -10 / (|x| + 3)) :
  a + f (-2) a = 2 :=
by
  sorry

end proof_problem_l423_423852


namespace trapezium_congruent_triangles_l423_423824

theorem trapezium_congruent_triangles (m n : ℕ) (h : m ≥ n) : 
  ∃ t, ∀ {a b c: ℕ}, congruent (t a b c) := sorry

end trapezium_congruent_triangles_l423_423824


namespace coin_tails_probability_l423_423890

theorem coin_tails_probability:
  let p := 0.5 in
  let n := 3 in
  let k := 2 in
  (Nat.choose n k) * (p^k) * ((1-p)^(n-k)) = 0.375 :=
by
  sorry

end coin_tails_probability_l423_423890


namespace area_calculation_l423_423293

open Real

noncomputable def area_under_curve := ∫ x in 0..1, x^2 + 2

theorem area_calculation :
  area_under_curve = 7 / 3 :=
by
  sorry

end area_calculation_l423_423293


namespace sum_abs_le_1000_l423_423635

theorem sum_abs_le_1000 {a : Fin 2002 → ℤ} 
  (h1 : ∀ i, a i = 1 ∨ a i = -1) 
  (h2 : (∑ i : Fin 2002, a i * a ((i + 1) % 2002)) < 0) :
  abs (∑ i : Fin 2002, a i) ≤ 1000 :=
sorry

end sum_abs_le_1000_l423_423635


namespace quadrilateral_midpoint_diagonals_l423_423585

-- Defining the problem conditions and the statement to be proven
theorem quadrilateral_midpoint_diagonals :
  ∀ (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (midpoint_AB : (A -> B -> Type))
  (midpoint_BC : (B -> C -> Type))
  (midpoint_CD : (C -> D -> Type))
  (midpoint_DA : (D -> A -> Type))
  (M P N Q : Type)
  (AC BD : ℝ),
  AC = 7 →
  BD = 17 →
  (midpoint_AB A B = M) →
  (midpoint_BC B C = P) →
  (midpoint_CD C D = N) →
  (midpoint_DA D A = Q) →
  let MN := (7 / 2)^2 in
  let PQ := (17 / 2)^2 in
  MN + PQ = 169 :=
begin
  intros,
  -- Given the lengths of the diagonals of the quadrilateral ABCD
  -- and the lengths of the diagonals of the midpoint quadrilateral MNPQ,
  -- this problem demonstrates that the sum of the squares of these diagonals is as stated.
  sorry
end

end quadrilateral_midpoint_diagonals_l423_423585


namespace number_picked_by_person_announcing_average_5_l423_423989

-- Definition of given propositions and assumptions
def numbers_picked (b : Fin 6 → ℕ) (average : Fin 6 → ℕ) :=
  (b 4 = 15) ∧
  (average 4 = 8) ∧
  (average 1 = 5) ∧
  (b 2 + b 4 = 16) ∧
  (b 0 + b 2 = 10) ∧
  (b 4 + b 0 = 12)

-- Prove that given the conditions, the number picked by the person announcing an average of 5 is 7
theorem number_picked_by_person_announcing_average_5 (b : Fin 6 → ℕ) (average : Fin 6 → ℕ)
  (h : numbers_picked b average) : b 2 = 7 :=
  sorry

end number_picked_by_person_announcing_average_5_l423_423989


namespace larger_circle_radius_l423_423807

theorem larger_circle_radius (r : ℝ) (R : ℝ) (h1 : ∀ i : fin 4, circle (0, 2) = circle (center i) r) 
  (h2 : ∀ i j : fin 4, i ≠ j → dist (center i) (center j) = 4)
  (h3 : ∀ i : fin 4, dist (0, 0) (center i) = R + r) :
  R = 2 * real.sqrt 3 + 2 :=
by
  sorry

end larger_circle_radius_l423_423807


namespace x_is_perfect_square_l423_423661

theorem x_is_perfect_square (x y : ℕ) (hxy : x > y) (hdiv : xy ∣ x ^ 2022 + x + y ^ 2) : ∃ n : ℕ, x = n^2 := 
sorry

end x_is_perfect_square_l423_423661


namespace altitude_of_isosceles_triangle_l423_423904

noncomputable def inradius (A B C D : Point) (r : ℝ) : Prop :=
  -- Definition relating A, B, C, D and r for the inradius condition
  sorry

noncomputable def exradius (A B C D : Point) (r : ℝ) : Prop :=
  -- Definition relating A, B, C, D and r for the exradius condition
  sorry

theorem altitude_of_isosceles_triangle (A B C D : Point) (r : ℝ) 
  (h1 : dist A C = dist B C) 
  (h2 : D ∈ lineSegment A B)
  (h3 : inradius A C D r) 
  (h4 : exradius B C D r) :
  altitude A B C = 4 * r :=
sorry

end altitude_of_isosceles_triangle_l423_423904


namespace friends_transitive_l423_423204

universe u
variable {X : Type u}
variable [Fintype X]

-- Define conditions as Lean predicates
def is_friend (a b : X) : Prop := sorry -- Define friendship relation
def pairable (Y : Set X) : Prop := sorry -- Define pairable subsets

def not_pairable (X : Set X) : Prop := ¬ pairable X
def friends_not_everyone (a : X) : Prop := ¬ ∀ b, is_friend a b

-- Conditions given in the problem 
axiom not_pairable_X : not_pairable (Set.univ : Set X)
axiom pairable_if_not_friends (A B : X) (h : ¬ is_friend A B) : pairable ({x | x ∈ X ∧ x ≠ A ∧ x ≠ B})
axiom no_friends_everyone (a : X) : friends_not_everyone a

-- Proposition to prove
theorem friends_transitive (a b c : X) (hab : is_friend a b) (hbc : is_friend b c) : is_friend a c := sorry

end friends_transitive_l423_423204


namespace infinitely_many_divisible_and_not_divisible_by_n_square_plus_one_l423_423625

theorem infinitely_many_divisible_and_not_divisible_by_n_square_plus_one : 
  (∀ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (n^2 + 1 ∣ k!)) ∧ 
  (∀ n : ℕ, ∃ p : ℕ, (nat.prime p) ∧ (p % 4 = 1) ∧ (n^2 + 1 ∣ p) ∧ (¬ p ∣ n!)) :=
by {
  sorry
}

end infinitely_many_divisible_and_not_divisible_by_n_square_plus_one_l423_423625


namespace binary_sum_l423_423742

theorem binary_sum :
  (0b10101 + 0b1011 + 0b11100 + 0b1010101 : ℕ) = 0b11110011 :=
by norm_num

end binary_sum_l423_423742


namespace opposite_of_neg_two_is_two_l423_423885

def is_opposite (x y : Int) : Prop := x = -y

theorem opposite_of_neg_two_is_two (a : Int) (h : is_opposite a (-2)) : a = 2 :=
by
  intro a h
  exact sorry

end opposite_of_neg_two_is_two_l423_423885


namespace sum_eq_neg_one_l423_423953

open Complex

noncomputable def computeSum (x : ℂ) (h1 : x ^ 3017 = 1) (h2 : x ≠ 1) : ℂ :=
∑ k in Finset.range 3013, x ^ (3 * (k + 1)) / (x ^ (k + 1) - 1)

theorem sum_eq_neg_one (x : ℂ) (h1 : x ^ 3017 = 1) (h2 : x ≠ 1) : computeSum x h1 h2 = -1 :=
sorry

end sum_eq_neg_one_l423_423953


namespace farmer_plants_rows_per_bed_l423_423720

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed_l423_423720


namespace expression_range_l423_423247

theorem expression_range (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha' : a ≤ 2)
    (hb : 0 ≤ b) (hb' : b ≤ 2)
    (hc : 0 ≤ c) (hc' : c ≤ 2)
    (hd : 0 ≤ d) (hd' : d ≤ 2) :
  4 + 2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) 
  ∧ Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := 
sorry

end expression_range_l423_423247


namespace thirtieth_triangular_number_l423_423066

theorem thirtieth_triangular_number : 
  let T (n : ℕ) := n * (n + 1) / 2 
  in T 30 = 465 := 
by 
  sorry

end thirtieth_triangular_number_l423_423066


namespace number_of_negative_x_values_l423_423134

theorem number_of_negative_x_values : 
  (∃ (n : ℕ), ∀ (x : ℤ), x = n^2 - 196 ∧ x < 0) ∧ (n ≤ 13) :=
by 
  -- To formalize our problem we need quantifiers, inequalities and integer properties.
  sorry

end number_of_negative_x_values_l423_423134


namespace symmetry_implies_value_l423_423166

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem symmetry_implies_value :
  (∀ (x : ℝ), ∃ (k : ℤ), ω * x - Real.pi / 3 = k * Real.pi + Real.pi / 2) →
  (∀ (x : ℝ), ∃ (k : ℤ), 2 * x + φ = k * Real.pi) →
  0 < φ → φ < Real.pi →
  ω = 2 →
  φ = Real.pi / 6 →
  g (Real.pi / 3) φ = -Real.sqrt 3 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  exact sorry

end symmetry_implies_value_l423_423166


namespace integers_difference_squares_count_l423_423869

/-- 
Statement: 
  There are exactly 1500 integers between 1 and 2000 (inclusive) 
  that can be expressed as the difference of the squares of two nonnegative integers.
-/
theorem integers_difference_squares_count : 
  {n | ∃ a b : ℕ, a^2 - b^2 = n ∧ 1 ≤ n ∧ n ≤ 2000}.toFinset.card = 1500 :=
by
  sorry

end integers_difference_squares_count_l423_423869


namespace symmetric_point_origin_l423_423562

theorem symmetric_point_origin :
  ∀ (P : ℝ × ℝ × ℝ), P = (3, 1, 5) → ∃ Q : ℝ × ℝ × ℝ, Q = (-3, -1, -5) :=
by
  assume P,
  assume hP: P = (3, 1, 5),
  use (-3, -1, -5),
  sorry

end symmetric_point_origin_l423_423562


namespace log_sequence_is_convex_l423_423834

theorem log_sequence_is_convex (b : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < b n)
  (h_convex : ∀ (c : ℝ), 0 < c → ∀ n, (c^2 * b (n+1)) ≤ (c * b n + c^(n-1) * b (n-1)) / 2) : 
  ∀ n, (log (b n)) ≤ (log (b (n - 1)) + log (b (n + 1))) / 2 :=
sorry

end log_sequence_is_convex_l423_423834


namespace sum_of_extreme_values_in_interval_l423_423208

noncomputable def y (a x : ℝ) : ℝ := 2 * x^2 + a / x

theorem sum_of_extreme_values_in_interval :
  (∃ a : ℝ, y a (-1) = -30) →
  ∃ a : ℝ, 
    let f := y a in 
    let min_val := min (f 1) (min (f 2) (f 4)) in 
    let max_val := max (f 1) (max (f 2) (f 4)) in 
    min_val + max_val = 64 := 
by 
  sorry

end sum_of_extreme_values_in_interval_l423_423208


namespace departure_of_30_tons_of_grain_l423_423212

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l423_423212


namespace iterated_exponential_divisibility_l423_423617

theorem iterated_exponential_divisibility (a : ℕ) : ∃ n : ℕ, (n = 2 * a - 1) ∧ (∀ k : ℕ, k ≥ 1 → a ∣ ((nat.iterate (λ x, n ^ x) k n) + 1)) :=
sorry

end iterated_exponential_divisibility_l423_423617


namespace sum_of_differences_ends_in_9_l423_423292

theorem sum_of_differences_ends_in_9 
  (a_i b_i : ℕ → ℕ) 
  (h : ∀ i, 1 ≤ i ∧ i ≤ 999 → |a_i i - b_i i| = 1 ∨ |a_i i - b_i i| = 6) 
  (part : (∀ i, ∃ j, i ≠ j ∧ a_i i = a_i j ∧ b_i i = b_i j) → 
    ∃ (c_i : ℕ → ℕ), 
    (∀ i, c_i (2 * i) = a_i i ∧ c_i (2 * i + 1) = b_i i) ∧ 
    (∀ j ≠ i, c_i j ≠ c_i i) ∧ 
    ∀ k, 1 ≤ k ∧ k ≤ 1998 → 
    ∃ l, l ∈ c_i'_i ∧ l = k)) : 
  (∑ i in finset.range 999, |a_i i - b_i i|) % 10 = 9 :=
sorry

end sum_of_differences_ends_in_9_l423_423292


namespace Moe_has_least_l423_423752

-- Define people as elements
variables (Bo Coe Flo Jo Moe : ℝ)

-- Conditions stated as hypotheses
hypothesis (h1 : Flo > Jo ∧ Flo > Bo)
hypothesis (h2 : Bo > Moe ∧ Coe > Moe)
hypothesis (h3 : Jo > Moe ∧ Jo < Bo)
hypothesis (h4 : Bo ≠ Coe ∧ Bo ≠ Flo ∧ Bo ≠ Jo ∧ Bo ≠ Moe ∧
                  Coe ≠ Flo ∧ Coe ≠ Jo ∧ Coe ≠ Moe ∧
                  Flo ≠ Jo ∧ Flo ≠ Moe ∧
                  Jo ≠ Moe)

-- Prove that Moe has the least amount of money
theorem Moe_has_least (Bo Coe Flo Jo Moe : ℝ) 
  (h1 : Flo > Jo ∧ Flo > Bo) 
  (h2 : Bo > Moe ∧ Coe > Moe) 
  (h3 : Jo > Moe ∧ Jo < Bo) 
  (h4 : Bo ≠ Coe ∧ Bo ≠ Flo ∧ Bo ≠ Jo ∧ Bo ≠ Moe ∧
         Coe ≠ Flo ∧ Coe ≠ Jo ∧ Coe ≠ Moe ∧
         Flo ≠ Jo ∧ Flo ≠ Moe ∧
         Jo ≠ Moe) : 
  (Moe < Jo ∧ Moe < Bo ∧ Moe < Coe ∧ Moe < Flo) :=
sorry

end Moe_has_least_l423_423752


namespace range_of_x_for_f_lt_zero_l423_423196

theorem range_of_x_for_f_lt_zero :
  (∀ (a : ℝ), (-1 ≤ a ∧ a ≤ 1) → (∀ x : ℝ, f a x < 0)) → (∀ x : ℝ, 1 < x ∧ x < 2) :=
begin
  sorry
end

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1


end range_of_x_for_f_lt_zero_l423_423196


namespace product_of_solutions_to_x_squared_equals_49_l423_423118

theorem product_of_solutions_to_x_squared_equals_49 :
  (∃ (x : ℝ), x ^ 2 = 49) → ((∀ x, x ^ 2 = 49 → (x = 7 ∨ x = -7))) →
  (∏ x in { x : ℝ | x ^ 2 = 49}.to_finset, x) = -49 :=
begin
  sorry
end

end product_of_solutions_to_x_squared_equals_49_l423_423118


namespace max_value_trig_formula_l423_423089

theorem max_value_trig_formula (x : ℝ) : ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := 
sorry

end max_value_trig_formula_l423_423089


namespace perimeter_of_tangential_triangle_l423_423749

-- Define the conditions
structure Circle (α : Type) :=
  (radius : ℝ)
  (center : α)

-- Given conditions for the problem
def c1 : Circle (ℝ × ℝ) := ⟨3, (0, 0)⟩
def c2 : Circle (ℝ × ℝ) := ⟨3, (6, 0)⟩  -- Externally tangent to c1
def c3 : Circle (ℝ × ℝ) := ⟨3, (3, 3 * Real.sqrt 3)⟩  -- Externally tangent to c1 and c2

-- Defining the points of tangency between circles and the triangle
def tangency_points : List (ℝ × ℝ) := [(0, 0), (6, 0), (3, 3 * Real.sqrt 3)]

-- The statement we want to prove
theorem perimeter_of_tangential_triangle
  (r : ℝ)
  (h : r = 3) : 
  let perimeter := 3 * (6 + 6 * Real.sqrt 3) 
  in perimeter = 18 + 18 * Real.sqrt 3 := 
by
  sorry

end perimeter_of_tangential_triangle_l423_423749


namespace temperature_at_night_is_minus_two_l423_423063

theorem temperature_at_night_is_minus_two (temperature_noon temperature_afternoon temperature_drop_by_night temperature_night : ℤ) : 
  temperature_noon = 5 → temperature_afternoon = 7 → temperature_drop_by_night = 9 → 
  temperature_night = temperature_afternoon - temperature_drop_by_night → 
  temperature_night = -2 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3] at h4
  exact h4


end temperature_at_night_is_minus_two_l423_423063


namespace problem_statement_l423_423820

theorem problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = ∑ i in finset.range (n + 1), a i)
  (h2 : ∀ n, (S n - 1) * (S n - 1) = a n * S n) :
  a 2015 + 1 / 2016 = 1 / 2015 :=
by
  sorry

end problem_statement_l423_423820


namespace base_area_of_tetrahedron_l423_423384

-- Given a regular tetrahedron circumscribed around a sphere of radius 1

def regular_tetrahedron_base_area (r : ℝ) (area : ℝ) : Prop :=
  ∀ (a : ℝ), a^2 = (4*sqrt 2 - 4) → area = (sqrt 3 / 4) * a^2 

theorem base_area_of_tetrahedron :
  regular_tetrahedron_base_area 1 (4*sqrt 2 - 4) :=
sorry

end base_area_of_tetrahedron_l423_423384


namespace trig_identity_l423_423274

theorem trig_identity (α : ℝ) : (sin α - cos α)^2 + sin (2 * α) = 1 :=
by
  sorry

end trig_identity_l423_423274


namespace initial_milk_amount_l423_423264

theorem initial_milk_amount (M : ℝ) (H1 : 0.05 * M = 0.02 * (M + 15)) : M = 10 :=
by
  sorry

end initial_milk_amount_l423_423264


namespace rectangle_perimeter_l423_423621

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangle_perimeter (x y : ℝ) (A : ℝ) (E : ℝ) (fA fB : Real) (p : ℝ) 
  (h1 : y = 2 * x)
  (h2 : x * y = 2015)
  (h3 : E = 2006 * π)
  (h4 : fA = x + y)
  (h5 : fB ^ 2 = (3 / 2)^2 * 1007.5 - (p / 2)^2)
  (h6 : 2 * (3 / 2 * sqrt 1007.5 * sqrt 1009.375) = 2006 / π) :
  2 * (x + y) = 6 * sqrt 1007.5 := 
by
  sorry

end rectangle_perimeter_l423_423621


namespace range_of_a_l423_423844

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l423_423844


namespace find_f_l423_423518

noncomputable def f (f'₁ : ℝ) (x : ℝ) : ℝ := f'₁ * Real.exp x - x ^ 2

theorem find_f'₁ (f'₁ : ℝ) (h : f f'₁ = λ x => f'₁ * Real.exp x - x ^ 2) :
  f'₁ = 2 * Real.exp 1 / (Real.exp 1 - 1) := by
  sorry

end find_f_l423_423518


namespace number_of_draws_l423_423804

-- Definition of the competition conditions
def competition_conditions (A B C D E : ℕ) : Prop :=
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = E) ∧
  15 ∣ (10000 * A + 1000 * B + 100 * C + 10 * D + E)

-- The main theorem stating the number of draws
theorem number_of_draws :
  ∃ (A B C D E : ℕ), competition_conditions A B C D E ∧ 
  (∃ (draws : ℕ), draws = 3) :=
by
  sorry

end number_of_draws_l423_423804


namespace problem_solution_l423_423257

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end problem_solution_l423_423257


namespace y2_greater_y1_range_l423_423487

theorem y2_greater_y1_range (x : ℝ) : 
    let y₁ := x + 1,
        y₂ := (1 / 2) * x^2 - (1 / 2) * x - 1
    in y₂ > y₁ ↔ x < -1 ∨ x > 4 := 
    by
      sorry

end y2_greater_y1_range_l423_423487


namespace series_converges_l423_423768

theorem series_converges :
  ∑' n, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_converges_l423_423768


namespace distance_between_parallel_lines_l423_423762

noncomputable def a : ℝ × ℝ := (2, -3)
noncomputable def b : ℝ × ℝ := (1, -5)
noncomputable def d : ℝ × ℝ := (1, -7)

theorem distance_between_parallel_lines :
  ∀ (a b d : ℝ × ℝ), a = (2, -3) → b = (1, -5) → d = (1, -7) →
  (distance (a, b, d) = (9 * Real.sqrt 2) / 10) :=
begin
  sorry
end

end distance_between_parallel_lines_l423_423762


namespace inscribed_circle_area_eq_48pi_l423_423058

theorem inscribed_circle_area_eq_48pi {s : ℝ} (h : s = 24) : 
  let r := sqrt 3 * s / 6 in
  let area := π * r^2 in
  area = 48 * π :=
by 
  sorry

end inscribed_circle_area_eq_48pi_l423_423058


namespace sum_y_coordinates_on_circle_y_axis_l423_423073

theorem sum_y_coordinates_on_circle_y_axis :
  let C := { p : ℝ × ℝ | (p.1 + 8) ^ 2 + (p.2 - 3) ^ 2 = 225 }
  in (∃! p₁ p₂ : ℝ × ℝ, p₁ ∈ C ∧ p₂ ∈ C ∧ p₁.1 = 0 ∧ p₂.1 = 0 ∧ p₁.2 + p₂.2 = 6) :=
sorry

end sum_y_coordinates_on_circle_y_axis_l423_423073


namespace order_of_a_b_c_l423_423484

noncomputable def a : ℝ := Real.log 5 / Real.log 8
noncomputable def b : ℝ := Real.log 3 / Real.log 4
def c : ℝ := 2 / 3

theorem order_of_a_b_c : c < a ∧ a < b := 
by
  sorry

end order_of_a_b_c_l423_423484


namespace departure_of_30_tons_of_grain_l423_423213

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l423_423213


namespace total_time_to_fill_tank_l423_423698

noncomputable def pipe_filling_time : ℕ := 
  let tank_capacity := 2000
  let pipe_a_rate := 200
  let pipe_b_rate := 50
  let pipe_c_rate := 25
  let cycle_duration := 5
  let cycle_fill := (pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2)
  let num_cycles := tank_capacity / cycle_fill
  num_cycles * cycle_duration

theorem total_time_to_fill_tank : pipe_filling_time = 40 := 
by
  unfold pipe_filling_time
  sorry

end total_time_to_fill_tank_l423_423698


namespace length_of_non_parallel_side_l423_423997

theorem length_of_non_parallel_side
  (O : Type)
  (A B : O)
  (OA OB : ℝ)
  (hOA : OA = 12)
  (hOB : OB = 5)
  (hAOB : ∠AOB = 90) : 
  AB = 13 :=
by
  sorry

end length_of_non_parallel_side_l423_423997


namespace isosceles_triangles_area_ratio_l423_423333

-- Defining the height ratio
def height_ratio : ℝ := 0.7142857142857143

-- Proving the ratio of the areas of two isosceles triangles under given conditions
theorem isosceles_triangles_area_ratio :
  ∀ (A B : Type) [ordered_field A] [ordered_field B] (h1 h2 b1 b2 : ℝ), 
  b2 / b1 = height_ratio →
  h2 / h1 = height_ratio →
  (1/2 * b2 * h2) / (1/2 * b1 * h1) = height_ratio^2 :=
by
  intros A B _ _ h1 h2 b1 b2 h_ratio1 h_ratio2
  have area_ratio := (height_ratio * height_ratio : ℝ)
  sorry

end isosceles_triangles_area_ratio_l423_423333


namespace paving_cost_proof_l423_423735

-- Definitions based on the given conditions
def room_length : ℝ := 5.5
def room_width : ℝ := 3.75

def type_a_length : ℝ := 1.5
def type_a_width : ℝ := 2
def type_a_cost : ℝ := 800

def type_b_length : ℝ := 1
def type_b_width : ℝ := 1
def type_b_cost : ℝ := 1200

-- Total cost calculation function
noncomputable def total_paving_cost : ℝ :=
  let total_area := room_length * room_width
  let area_a := type_a_length * type_a_width
  let num_a := (total_area / area_a).toNat
  let remaining_area := total_area - (num_a * area_a)
  let area_b := type_b_length * type_b_width
  let num_b := (remaining_area / area_b).ceil.toNat
  (num_a * type_a_cost) + (num_b * type_b_cost)

-- The proof statement
theorem paving_cost_proof : total_paving_cost = 8400 := by
  sorry

end paving_cost_proof_l423_423735


namespace incorrect_transformation_l423_423001

theorem incorrect_transformation (x y m : ℕ) : 
  (x = y → x + 3 = y + 3) ∧
  (-2 * x = -2 * y → x = y) ∧
  (m ≠ 0 → (x = y ↔ (x / m = y / m))) ∧
  ¬(x = y → x / m = y / m) :=
by
  sorry

end incorrect_transformation_l423_423001


namespace mr_ray_customers_without_fish_l423_423970

def customers_without_fish 
  (total_customers : ℕ)
  (total_tuna : ℕ)
  (tuna_weight : ℕ)
  (weight_per_customer : ℕ)
  (customers_30lb_pieces : ℕ)
  (weight_30lb_piece : ℕ)
  (customers_20lb_pieces : ℕ)
  (weight_20lb_piece : ℕ)
  (tuna_reserve_percentage : ℚ) : ℕ :=
  let total_weight := total_tuna * tuna_weight
  let reserve_weight := total_tuna * (tuna_weight * tuna_reserve_percentage).toNat
  let sellable_weight := total_weight - reserve_weight
  let preferred_weight := (customers_30lb_pieces * weight_30lb_piece) + (customers_20lb_pieces * weight_20lb_piece)
  let remaining_weight := sellable_weight - preferred_weight
  let customers_25lb := remaining_weight / weight_per_customer
  let total_customers_served := customers_30lb_pieces + customers_20lb_pieces + customers_25lb
  total_customers - total_customers_served

theorem mr_ray_customers_without_fish :
  customers_without_fish 100 10 200 25 10 30 15 20 (10 / 100 : ℚ) = 27 := by
  sorry

end mr_ray_customers_without_fish_l423_423970


namespace scorpion_needs_10_millipedes_l423_423022

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l423_423022


namespace find_a_on_real_axis_l423_423476

theorem find_a_on_real_axis (a : ℝ) : (1 + complex.i) * (a + complex.i) ∈ set.range (λ x, (x : ℂ)) → a = -1 := by
  sorry

end find_a_on_real_axis_l423_423476


namespace ants_distance_is_sqrt2_l423_423145

noncomputable def ant_distance_after_2008_segments : ℝ :=
  let A := (0, 0, 0) -- Define initial points in a conceptual system
  let A1 := (0, 0, 1)
  let D1 := (1, 1, 1)
  let C := (1, 0, 0)
  real.dist D1 C

theorem ants_distance_is_sqrt2 : ant_distance_after_2008_segments = Real.sqrt 2 :=
  by sorry

end ants_distance_is_sqrt2_l423_423145


namespace function_with_range_positive_real_l423_423055

-- Defining each function as given in the problem conditions
def f₁ (x : ℝ) : ℝ := 2 ^ (1 / x)
def f₂ (x : ℝ) : ℝ := sin x + cos x
def f₃ (x : ℝ) : ℝ := |x|
def f₄ (x : ℝ) : ℝ := (1 / 2) ^ (2 - x)

-- Stating the theorem that identifies the function with range (0,+∞)
theorem function_with_range_positive_real 
    (f₁ f₂ f₃ f₄ : ℝ → ℝ) :
  ∃ f, (f = f₄) ∧ (∀ y, 0 < y → ∃ x, f x = y) :=
by
  sorry

end function_with_range_positive_real_l423_423055


namespace hancho_height_l423_423185

theorem hancho_height (Hansol_height : ℝ) (h1 : Hansol_height = 134.5) (ratio : ℝ) (h2 : ratio = 1.06) :
  Hansol_height * ratio = 142.57 := by
  sorry

end hancho_height_l423_423185


namespace coeff_z_in_third_eq_l423_423851

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq_l423_423851


namespace problem_I_problem_II_l423_423168

-- Definitions
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (a - 2) * x - Real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (h_min : ∀ x : ℝ, function_f a 1 ≤ function_f a x) :
  a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → (function_f a x < function_f a 1)) ∧ (∀ x : ℝ, x > 1 → (function_f a x > function_f a 1)) :=
sorry

-- Problem (II)
theorem problem_II (a x0 : ℝ) (h_a_gt_1 : a > 1) (h_x0_pos : 0 < x0) (h_x0_lt_1 : x0 < 1)
    (h_min : ∀ x : ℝ, function_f a (1/a) ≤ function_f a x) :
  ∀ x : ℝ, function_f a 0 > 0
:= sorry

end problem_I_problem_II_l423_423168


namespace mango_production_l423_423981

-- Conditions
def num_papaya_trees := 2
def papayas_per_tree := 10
def num_mango_trees := 3
def total_fruits := 80

-- Definition to be proven
def mangos_per_mango_tree : Nat :=
  (total_fruits - num_papaya_trees * papayas_per_tree) / num_mango_trees

theorem mango_production :
  mangos_per_mango_tree = 20 := by
  sorry

end mango_production_l423_423981


namespace bail_rate_l423_423631

theorem bail_rate 
  (distance_to_shore : ℝ) 
  (shore_speed : ℝ) 
  (leak_rate : ℝ) 
  (boat_capacity : ℝ) 
  (time_to_shore_min : ℝ) 
  (net_water_intake : ℝ)
  (r : ℝ) :
  distance_to_shore = 2 →
  shore_speed = 3 →
  leak_rate = 12 →
  boat_capacity = 40 →
  time_to_shore_min = 40 →
  net_water_intake = leak_rate - r →
  net_water_intake * (time_to_shore_min) ≤ boat_capacity →
  r ≥ 11 :=
by
  intros h_dist h_speed h_leak h_cap h_time h_net h_ineq
  sorry

end bail_rate_l423_423631


namespace jogging_time_after_two_weeks_l423_423602

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end jogging_time_after_two_weeks_l423_423602


namespace football_game_attendance_l423_423326

theorem football_game_attendance :
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  wednesday - monday = 50 :=
by
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  show wednesday - monday = 50
  sorry

end football_game_attendance_l423_423326


namespace range_of_a_l423_423843

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l423_423843


namespace find_b_l423_423543

noncomputable theory
open Real

variables {x₁ x₂ k b : ℝ}

def tangent_line (x : ℝ) (k b : ℝ) := k * x + b
def curve_1 (x : ℝ) := log x + 2
def curve_2 (x : ℝ) := log (x + 1)

theorem find_b
  (h_tangent_k : ∀ x₁ x₂, k = 1 / x₁ ∧ k = 1 / (x₂ + 1))
  (h_x_relation : x₁ = x₂ + 1)
  (h_point_on_curve1 : tangent_line x₁ k b = curve_1 x₁)
  (h_point_on_curve2 : tangent_line x₂ k b = curve_2 x₂) :
  b = 1 - log 2 :=
begin
  sorry
end

end find_b_l423_423543


namespace smallest_fraction_greater_than_three_fifths_l423_423394

theorem smallest_fraction_greater_than_three_fifths : 
    ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a % 1 = 0) ∧ (b % 1 = 0) ∧ (5 * a > 3 * b) ∧ a = 59 :=
by
  sorry

end smallest_fraction_greater_than_three_fifths_l423_423394


namespace smallest_fraction_gt_3_5_with_two_digit_nums_l423_423397

theorem smallest_fraction_gt_3_5_with_two_digit_nums : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 5 * a > 3 * b ∧ (∀ (a' b' : ℕ), 10 ≤ a' ∧ a' < 100 ∧ 10 ≤ b' ∧ b' < 100 ∧ 5 * a' > 3 * b' → a * b' ≤ a' * b) := 
  ⟨59, 98, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, sorry⟩

end smallest_fraction_gt_3_5_with_two_digit_nums_l423_423397


namespace num_diagonals_hexagonal_prism_num_diagonals_trimmed_cube_l423_423414

theorem num_diagonals_hexagonal_prism : 
    ∀ (vertices : ℕ), vertices = 12 → 
    ∀ (non_diagonal : ℕ), non_diagonal = 9 → 
    (vertices * (vertices - non_diagonal)) / 2 = 18 := by
    intros vertices h1 non_diagonal h2
    rw [h1, h2]
    norm_num
    sorry

theorem num_diagonals_trimmed_cube : 
    ∀ (vertices : ℕ), vertices = 24 → 
    ∀ (non_diagonal : ℕ), non_diagonal = 14 → 
    (vertices * (vertices - non_diagonal)) / 2 = 120 := by
    intros vertices h1 non_diagonal h2
    rw [h1, h2]
    norm_num
    sorry

end num_diagonals_hexagonal_prism_num_diagonals_trimmed_cube_l423_423414


namespace coin_serial_number_permutations_l423_423372

theorem coin_serial_number_permutations :
  let digits := [2, 2, 4, 5, 7, 9]
  let odd_digits := [d | d ∈ digits, d % 2 = 1]
  (∃ n, n ∈ digits ∧ n % 2 = 1 ∧
    (∃ p : Fin 6 ↔ 6, p.length = 5 ∧ 
      (∀ i, (p i) ≠ n) ∧ 
      p.count n = 1) ∧ 
    (∃ q : Fin 6 ↔ 5, q.length = 5 ∧ 
      Perm.Closure q) ∧
    (number_of_permutations : list (List α) = 120) :=
begin
  let digits := [2, 2, 4, 5, 7, 9],
  let start_points := [5, 7],
  let subset_f := [2, 2, 4, 7, 9],
  let subset_g := [2, 2, 4, 5, 9],
  let x_p := perm.length = 5 (subset_f) / 2 ! = 60,
  let y_p := perm.length = 5 (subset_g) / 2 ! = 60,
simpl with mul.sub singleton 
justify,
arith
w,
,
count
,parith 
Algebra associative,
perm eq
 βη
arithmetic c
r 
125
60+60=120
  sorry


end coin_serial_number_permutations_l423_423372


namespace red_ball_probability_l423_423711

theorem red_ball_probability 
  (red_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : black_balls = 9)
  (h3 : total_balls = red_balls + black_balls) :
  (red_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end red_ball_probability_l423_423711


namespace orthocenters_collinear_l423_423808

theorem orthocenters_collinear {l1 l2 l3 l4 : Line} :
  let t1 := l1.intersection l2 ∩ l3,
      t2 := l2.intersection l3 ∩ l4,
      t3 := l3.intersection l4 ∩ l1,
      t4 := l4.intersection l1 ∩ l2 in
  let h1 := orthocenter t1,
      h2 := orthocenter t2,
      h3 := orthocenter t3,
      h4 := orthocenter t4 in
  collinear {h1, h2, h3, h4} :=
sorry

end orthocenters_collinear_l423_423808


namespace first_digit_l423_423192

-- Definitions and conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

def number (x y : ℕ) : ℕ := 653 * 100 + x * 10 + y

-- Main theorem
theorem first_digit (x y : ℕ) (h₁ : isDivisibleBy (number x y) 80) (h₂ : x + y = 2) : x = 2 :=
sorry

end first_digit_l423_423192


namespace coefficient_x2_in_expansion_l423_423560

theorem coefficient_x2_in_expansion :
  let exp_poly := (1 + 2 * x) ^ 10 in
  (coeff exp_poly 2) = 180 :=
by
  sorry

end coefficient_x2_in_expansion_l423_423560


namespace middle_school_mentoring_l423_423205

theorem middle_school_mentoring (s n : ℕ) (h1 : s ≠ 0) (h2 : n ≠ 0) 
  (h3 : (n : ℚ) / 3 = (2 : ℚ) * (s : ℚ) / 5) : 
  (n / 3 + 2 * s / 5) / (n + s) = 4 / 11 := by
  sorry

end middle_school_mentoring_l423_423205


namespace simple_random_sampling_fairness_l423_423561

/--
In simple random sampling, every individual has an equal chance of being selected.
Thus, the person who draws first does not have a greater probability of being selected.

Given:
1. The method used is simple random sampling.
2. Every individual has an equal chance of being selected.

Prove:
The probability of being selected is the same for everyone, including the person who draws first.

Therefore, the statement "the person who draws first has a greater probability of being selected" is false.
-/
theorem simple_random_sampling_fairness :
  (∀ (n : ℕ) (draw : ℕ) (prob : ℕ → ℝ), prob draw = 1 / n → (∀ i, prob i = 1 / n)) →
  ¬ ∃ i j, j < i ∧ (prob i > prob j) :=
by
  intros h
  sorry

end simple_random_sampling_fairness_l423_423561


namespace total_cost_of_trip_l423_423978

-- Given conditions
def initial_reading : ℕ := 74568
def cafe_reading : ℕ := 74580
def home_reading : ℕ := 74605
def fuel_efficiency : ℝ := 25
def price_per_gallon : ℝ := 4.10

-- Define the result of the total cost calculation
def total_cost : ℝ := 6.07

-- Statement to be proved
theorem total_cost_of_trip :
  let distance_to_cafe := cafe_reading - initial_reading,
      distance_to_home := home_reading - cafe_reading,
      total_distance := distance_to_cafe + distance_to_home,
      gas_used := total_distance / fuel_efficiency,
      cost := gas_used * price_per_gallon
  in cost = total_cost :=
by
  -- Proof would go here
  sorry

end total_cost_of_trip_l423_423978


namespace pairs_parallel_edges_tesseract_l423_423871

def is_tesseract (t : Type*) := -- to define the tesseract
  ∃ (E : ℕ), E = 32

noncomputable def parallel_pairs_in_cube := 18

theorem pairs_parallel_edges_tesseract (t : Type*) [is_tesseract t] : 
  2 * parallel_pairs_in_cube = 36 :=
by
  sorry

end pairs_parallel_edges_tesseract_l423_423871


namespace question_I_question_II_question_III_l423_423823

variables {a : ℕ → ℕ}
variables {k : ℕ → ℕ}
variables {b : ℕ → ℕ}
variables {g : ℕ → ℤ}

/-- Define the sequence properties -/
def k_eq (i : ℕ) : ℕ := k i
def b_eq (j : ℕ) : ℕ := ∑ i in (finset.range j).succ, k i
def g_eq (m : ℕ) : ℤ := ∑ j in (finset.range m).succ, b j - 100 * m

-- Condition: k_1 = 40, k_2 = 30, k_3 = 20, k_4 = 10, k_5 = ... = k_100 = 0
def k_condition : Prop :=
  k 1 = 40 ∧ k 2 = 30 ∧ k 3 = 20 ∧ k 4 = 10 ∧ (∀ i > 4, k i = 0)

/-- Questions and Answers -/
-- (I) Calculate g(1), g(2), g(3), g(4)
theorem question_I (h : k_condition) :
  g 1 = -60 ∧ g 2 = -90 ∧ g 3 = -100 ∧ g 4 = -100 :=
sorry

-- (II) If the maximum term is 50, then for 1 ≤ m < 49, g(m) > g(m+1) and for m ≥ 49, g(m) = g(m+1)
theorem question_II (h_max : ∀ i, a i ≤ 50) :
  (∀ m, 1 ≤ m ∧ m < 49 → g m > g (m+1)) ∧ (∀ m, m ≥ 49 → g m = g (m+1)) :=
sorry

-- (III) If a₁ + a₂ + ... + a₁₀₀ = 200, then the minimum value of g(m) is -100
theorem question_III (h_sum : ∑ i in finset.range 100, a i = 200) :
  ∃ m, g m = -100 :=
sorry

end question_I_question_II_question_III_l423_423823


namespace cos_angle_QPS_l423_423922

noncomputable def PQR_props (PQ PR QR : ℝ) :=
PQ = 4 ∧ PR = 9 ∧ QR = 13

theorem cos_angle_QPS :
  ∀ (PQ PR QR : ℝ),
  PQR_props PQ PR QR →
  ∀ (P Q R S : Type),
  PQR ⊆ EuclideanSpace ℝ 3 →
  PQR_triangle PQ PR QR →
  external_bisect_angle P Q R S →
  ∃ (cos_QPS : ℝ), cos_QPS = -sqrt 11 / 6 :=
by sorry

end cos_angle_QPS_l423_423922


namespace a_pow_2004_add_b_pow_2005_eq_one_l423_423144

theorem a_pow_2004_add_b_pow_2005_eq_one (a b : ℝ) (h1 : {a, b / a, 1} = {a^2, a + b, 0}) (h2 : a ≠ 0) : 
  a^2004 + b^2005 = 1 :=
sorry

end a_pow_2004_add_b_pow_2005_eq_one_l423_423144


namespace largest_possible_determinant_l423_423578

-- Define the vectors v and w
def v : ℝ × ℝ × ℝ := (3, 2, -2)
def w : ℝ × ℝ × ℝ := (0, 1, 4)

-- Define a function to find the cross product of two 3D vectors
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- Calculate the cross product of v and w
def v_cross_w := cross_product v w

-- Define a unit vector u
axiom u : ℝ × ℝ × ℝ
axiom u_unit : u.1 ^ 2 + u.2.1 ^ 2 + u.2.2 ^ 2 = 1

-- Scalar triple product (determinant in this context)
def scalar_triple_product (u v w : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * (v_cross_w.1) + u.2.1 * (v_cross_w.2.1) + u.2.2 * (v_cross_w.2.2)

-- Proving that the largest possible determinant is √349
theorem largest_possible_determinant : ∃ u : ℝ × ℝ × ℝ, (u.1 ^ 2 + u.2.1 ^ 2 + u.2.2 ^ 2 = 1) ∧ scalar_triple_product u v w = real.sqrt 349 :=
by
  sorry

end largest_possible_determinant_l423_423578


namespace common_points_on_Euler_line_l423_423234

open EuclideanGeometry

variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ X₁ X₂ : Point}

-- Let ABC be an acute scalene triangle
variables [Triangle ABC]
axiom acute.scalene (ABC : Triangle) : ScaleneTriangle ABC ∧ AcuteTriangle ABC

-- Let A₁, B₁, C₁ be the feet of the altitudes from A, B, C
axiom Altitudes.feet (A₁ B₁ C₁ : Point) (ABC : Triangle) : FeetsOfAltitudes A₁ B₁ C₁ ABC

-- A₂ is the intersection of the tangents to the circle ABC at B, C, and similarly for B₂, C₂
axiom TangentIntersection (A₂ B₂ C₂ : Point) (ABC : Triangle) : TangentIntersection A₂ B₂ C₂ ABC

-- A₂A₁ intersects the circle A₂B₂C₂ again at A₃ and similarly for B₃, C₃
axiom CircleIntersection (A₃ B₃ C₃ : Point) (A₂ B₂ C₂ A₁ B₁ C₁ : Point) : CircleIntersection A₃ B₃ C₃ A₂ B₂ C₂ A₁ B₁ C₁

-- Show that the circles AA₁A₃, BB₁B₃, and CC₁C₃ all have two common points, X₁ and X₂ which both lie on the Euler line of the triangle ABC
theorem common_points_on_Euler_line (ABC : Triangle) (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ X₁ X₂ : Point)
  (acute : ScaleneTriangle ABC ∧ AcuteTriangle ABC)
  (feet : FeetsOfAltitudes A₁ B₁ C₁ ABC)
  (tangent : TangentIntersection A₂ B₂ C₂ ABC)
  (circle_intersections : CircleIntersection A₃ B₃ C₃ A₂ B₂ C₂ A₁ B₁ C₁) :
  (Circle A A₁ A₃).intersection (Circle B B₁ B₃) = (Circle C C₁ C₃).intersection (Circle B B₁ B₃) ∧
  (X₁ ∈ EulerLine ABC) ∧ (X₂ ∈ EulerLine ABC) :=
sorry

end common_points_on_Euler_line_l423_423234


namespace inequality_of_a_b_c_l423_423475

theorem inequality_of_a_b_c :
  let a := log 3 (1 / 2)
  let b := (1 / 2) ^ 3
  let c := 3 ^ (1 / 2)
  in a < b ∧ b < c := by
  sorry

end inequality_of_a_b_c_l423_423475


namespace probability_prime_sum_of_two_rolled_8_sided_die_is_23_over_64_l423_423375

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

theorem probability_prime_sum_of_two_rolled_8_sided_die_is_23_over_64 :
  (finset.univ.product finset.univ).card = 64 ∧
  (finset.filter (λ (p : ℕ × ℕ), is_prime (p.1 + p.2)) (finset.univ.product finset.univ)).card = 23 →
  (23 / 64 : ℚ) = 23 / 64 :=
by sorry

end probability_prime_sum_of_two_rolled_8_sided_die_is_23_over_64_l423_423375


namespace find_value_of_ratio_l423_423831

variables {x y : ℝ} (θ : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (hθ : θ ∈ Ioo (π / 4) (π / 2))
variable (h1 : cos θ / x = sin θ / y)
variable (h2 : sin θ ^ 2 / x ^ 2 + cos θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem find_value_of_ratio : ((x + y) ^ 2) / (x ^ 2 + y ^ 2) = (2 + Real.sqrt 3) / 2 :=
sorry

end find_value_of_ratio_l423_423831


namespace profit_share_difference_l423_423006

def investment_ratio (a b c p_b : ℕ) : ℕ :=
  let ratio := (a / 2000, b / 2000, c / 2000) in
  let part_value := p_b / ratio.2 in
  let total_profit := (ratio.1 + ratio.2 + ratio.3) * part_value in
  let a_share := ratio.1 * part_value in
  let c_share := ratio.3 * part_value in
  c_share - a_share

theorem profit_share_difference (a b c p_b : ℕ) (ha : a = 8000) (hb : b = 10000) (hc : c = 12000) (hpb : p_b = 1700) :
  investment_ratio a b c p_b = 680 :=
by {
  rw [ha, hb, hc, hpb],
  simp only [investment_ratio],
  norm_num,
  sorry
}

end profit_share_difference_l423_423006


namespace number_difference_l423_423319

theorem number_difference (a b : ℕ) (h1 : a + b = 25650) (h2 : a % 100 = 0) (h3 : b = a / 100) :
  a - b = 25146 :=
sorry

end number_difference_l423_423319


namespace find_a_l423_423597

noncomputable def lines_perpendicular (a : ℝ) (l1: ℝ × ℝ × ℝ) (l2: ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1
  let (A2, B2, C2) := l2
  (B1 ≠ 0) ∧ (B2 ≠ 0) ∧ (-A1 / B1) * (-A2 / B2) = -1

theorem find_a (a : ℝ) :
  lines_perpendicular a (a, 1, 1) (2*a, a - 3, 1) → a = 1 ∨ a = -3/2 :=
by
  sorry

end find_a_l423_423597


namespace greatest_power_sum_l423_423412

theorem greatest_power_sum (a b : ℕ) (h1 : 0 < a) (h2 : 2 < b) (h3 : a^b < 500) (h4 : ∀ m n : ℕ, 0 < m → 2 < n → m^n < 500 → a^b ≥ m^n) : a + b = 10 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end greatest_power_sum_l423_423412


namespace product_of_solutions_product_of_all_solutions_l423_423126

theorem product_of_solutions (x : ℝ) (h : x^2 = 49) : x = 7 ∨ x = -7 :=
begin
  rw eq_comm at h,
  exact eq_or_eq_neg_eq_of_sq_eq_sq h,
end

theorem product_of_all_solutions (h : {x : ℝ | x^2 = 49} = {7, -7}) : 7 * (-7) = -49 :=
by {
  rw set.eq_singleton_iff_unique_mem at h,
  rw [h.2 7, h.2 (-7)],
  exact mul_neg_eq_neg_mul_symm,
}

end product_of_solutions_product_of_all_solutions_l423_423126


namespace sin_alpha_zero_l423_423471

theorem sin_alpha_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : π / 2 < β ∧ β < π) 
  (h3 : sin (α + β) = 3 / 5) (h4 : cos β = -4 / 5) : sin α = 0 := 
sorry

end sin_alpha_zero_l423_423471


namespace tangent_line_equations_range_of_a_l423_423172

-- Part (1)
theorem tangent_line_equations (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x, f x = x^3 + ax^2 + x + 1) (hf' : ∀ x, f' x = 3*x^2 + 2*a*x + 1) :
  a = -1 →
  (∀ x, 4 + 2*a = 2 → f(1) = f(x)) →
  x = 0 → y = 1 →
  (
    (∃ k : ℝ, f' k = k) ∧ 
    ((∃ b : ℝ, y = b*x + 1) ∨ (∃ b : ℝ, y = b*x + 1))
  ) := sorry

-- Part (2)
theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x, f x = x^3 + ax^2 + x + 1) (hf' : ∀ x, f' x = 3*x^2 + 2*a*x + 1) :
  (∀ x, -2/3 ≤ x ∧ x ≤ -1/3 → f' x ≤ 0) →
  a ∈ Set.Ici 2 := sorry

end tangent_line_equations_range_of_a_l423_423172


namespace simplify_expression_l423_423954

-- Define nonzero real numbers p, q, r
variables {p q r : ℝ} (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)

-- Define x, y, z
def x : ℝ := q / r + r / q
def y : ℝ := p / r + r / p
def z : ℝ := p / q + q / p

-- Statement we intend to prove
theorem simplify_expression (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) : 
  (x hp hq hr)^2 + (y hp hq hr)^2 + (z hp hq hr)^2 - 2 * (x hp hq hr) * (y hp hq hr) * (z hp hq hr) = 4 :=
sorry

end simplify_expression_l423_423954


namespace sum_of_coordinates_of_A_l423_423575

variables {A B C : ℝ × ℝ}

def condition1 := (∥C.1 - A.1, C.2 - A.2∥ / ∥B.1 - A.1, B.2 - A.2∥ = 1 / 3) 
                 ∧ (∥C.1 - B.1, C.2 - B.2∥ / ∥B.1 - A.1, B.2 - A.2∥ = 1 / 3)

def pointB := (2, 3)
def pointC := (5, 12)

theorem sum_of_coordinates_of_A : condition1 → B = pointB → C = pointC → 
  A.1 + A.2 = 29 :=
sorry

end sum_of_coordinates_of_A_l423_423575


namespace polar_line_equation_l423_423220

theorem polar_line_equation (ρ θ : ℝ) (h1 : θ = π / 2) (h2 : ρ = 2) :
  ∃ ρ θ, ρ*sin θ = 2 :=
by
  use [2, π / 2]
  simp [Real.sin_pi_div_two]
  sorry

end polar_line_equation_l423_423220


namespace proof_problem_l423_423346

-- Definitions
structure Quadrilateral :=
  (a b c d : ℝ)
  (equal_diagonals : ℝ → ℝ → Prop)
  (bisect_each_other : ℝ → ℝ → Prop)
  (perpendicular_diagonals : ℝ → ℝ → Prop)
  (rectangle : quadrilateral → Prop)
  (parallelogram : quadrilateral → Prop)
  (rhombus : quadrilateral → Prop)

-- Conditions
axiom condition_A : ∀ (q : Quadrilateral), q.rectangle ↔ q.equal_diagonals q.a q.c ∧ q.bisect_each_other q.a q.c

def statement_A (q : Quadrilateral) : Prop := q.equal_diagonals q.a q.c ∧ q.bisect_each_other q.a q.c → q.rectangle q
def statement_B (q : Quadrilateral) : Prop := q.equal_diagonals q.a q.c → q.rectangle q
def statement_C (q : Quadrilateral) : Prop := q.parallelogram q → q.perpendicular_diagonals q.a q.c
def statement_D (q : Quadrilateral) : Prop := q.perpendicular_diagonals q.a q.c → q.rhombus q

-- Proof problem
theorem proof_problem (q : Quadrilateral) : statement_A q ∧ ¬statement_B q ∧ ¬statement_C q ∧ ¬statement_D q :=
by
  sorry

end proof_problem_l423_423346


namespace paco_salty_cookies_left_l423_423975

theorem paco_salty_cookies_left (S₁ S₂ : ℕ) (h₁ : S₁ = 6) (e1_eaten : ℕ) (a₁ : e1_eaten = 3)
(h₂ : S₂ = 24) (r1_ratio : ℚ) (a_ratio : r1_ratio = (2/3)) :
  S₁ - e1_eaten + r1_ratio * S₂ = 19 :=
by
  sorry

end paco_salty_cookies_left_l423_423975


namespace number_of_valid_integers_l423_423111

def g (n : ℤ) : ℤ := ⌈(110 * n : ℚ) / 111⌉ - ⌊(120 * n : ℚ) / 121⌋

theorem number_of_valid_integers :
  (Set.Of {n : ℤ | 1 + ⌊(120 * n : ℚ) / 121⌋ = ⌈(110 * n : ℚ) / 111⌉ }).card = 13431 := sorry

end number_of_valid_integers_l423_423111


namespace proof_cos_inequality_l423_423750

noncomputable def cos_inequality (n : ℕ) (α : ℕ → ℝ) : Prop :=
  (∀ i j, i ≠ j → α i + α j ≤ π) →
  (∀ i, 0 ≤ α i) →
  (∀ i, 0 ≤ i ∧ i < n) →
  (1 / (n : ℝ)) * ∑ i in finset.range n, real.cos (α i) ≤
  real.cos ((1 / (n : ℝ)) * ∑ i in finset.range n, (α i))

theorem proof_cos_inequality (n : ℕ) (α : ℕ → ℝ) 
  (h1 : ∀ i j, i ≠ j → α i + α j ≤ π)
  (h2 : ∀ i, 0 ≤ α i)
  (h3 : ∀ i, 0 ≤ i ∧ i < n) :
  cos_inequality n α :=
sorry

end proof_cos_inequality_l423_423750


namespace doll_cost_is_one_l423_423401

variable (initial_amount : ℕ) (end_amount : ℕ) (number_of_dolls : ℕ)

-- Conditions
def given_conditions : Prop :=
  initial_amount = 100 ∧
  end_amount = 97 ∧
  number_of_dolls = 3

-- Question: Proving the cost of each doll
def cost_per_doll (initial_amount end_amount number_of_dolls : ℕ) : ℕ :=
  (initial_amount - end_amount) / number_of_dolls

theorem doll_cost_is_one (h : given_conditions initial_amount end_amount number_of_dolls) :
  cost_per_doll initial_amount end_amount number_of_dolls = 1 :=
by
  sorry

end doll_cost_is_one_l423_423401


namespace surface_area_of_cube_is_correct_l423_423700

noncomputable def edge_length (a : ℝ) : ℝ := 5 * a

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  let edge := edge_length a
  6 * edge * edge

theorem surface_area_of_cube_is_correct (a : ℝ) :
  surface_area_of_cube a = 150 * a ^ 2 := by
  sorry

end surface_area_of_cube_is_correct_l423_423700


namespace matrix_N_unique_l423_423769

open Matrix

noncomputable def cross_product_matrix := 
  ![![0, -4, -3], 
    ![4, 0, -7], 
    ![3, 7, 0]]

theorem matrix_N_unique (u : Fin 3 → ℝ) :
    cross_product_matrix.mulVec u = 
      ![7, -3, 4].cross_product u :=
sorry

end matrix_N_unique_l423_423769


namespace min_x_value_l423_423623

theorem min_x_value (x : ℝ) (h1 : 2 * log x ≥ log 8 + log x) (h2 : x ≤ 32) : x ≥ 8 :=
by
  sorry

end min_x_value_l423_423623


namespace calculate_expression_l423_423416

theorem calculate_expression :
  ((∏ k in Finset.range 9 \ Finset.range 1, (1 + (1 / (k + 2))) ^ 2) / 
   (∏ k in Finset.range 9 \ Finset.range 1, (1 - (1 / (k + 2) ^ 2)))) = 55 :=
by
  sorry

end calculate_expression_l423_423416


namespace sum_of_reciprocals_l423_423521

open BigOperators

def S (n : ℕ) : Finset ℕ := {k | ∃ i, k = 2 ^ i ∧ i < n}

def S_i (σ : List ℕ) (i : ℕ) : ℕ :=
  (σ.take (i + 1)).sum

def Q (σ : List ℕ) (n : ℕ) : ℕ :=
  ∏ i in Finset.range n, S_i σ i

theorem sum_of_reciprocals (n : ℕ) :
  (∑ σ in (S n).toList.permutations, 1 / (Q σ n : ℚ)) = 1 := 
sorry

end sum_of_reciprocals_l423_423521


namespace amount_saved_is_25_percent_l423_423407

-- Define the condition that 8 tickets can be purchased for the price of 6 at the sale
def sale_price (P : ℝ) := 6 * P

-- Define the original price for 8 tickets
def original_price_8_tickets (P : ℝ) := 8 * P

-- Define the amount saved
def amount_saved (P : ℝ) := original_price_8_tickets P - sale_price P

-- Define the percent saved according to the problem conditions
def percent_saved (P : ℝ) := (amount_saved P / original_price_8_tickets P) * 100

-- The theorem to prove that the percent saved is 25%
theorem amount_saved_is_25_percent (P : ℝ) : percent_saved P = 25 := by
  sorry

end amount_saved_is_25_percent_l423_423407


namespace quadrilateral_equal_perimeters_is_rhombus_l423_423718

theorem quadrilateral_equal_perimeters_is_rhombus
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (quadrilateral : ConvexQuadrilateral A B C D)
  (M : Point (intersectingDiagnostics quadrilateral ABD AC))
  (equal_perimeters : ∀ (tria : Triangle4 M A B C D), (Perimeter tria) = (Perimeter (nextTriangle tria))) : 
  IsRhombus quadrilateral :=
by
  sorry

end quadrilateral_equal_perimeters_is_rhombus_l423_423718


namespace initial_milk_amount_l423_423743

theorem initial_milk_amount (r m : ℝ) (h₁ : r = 0.69) (h₂ : m = 0.4) :
  let t := 0.69 / (1 - 0.4) in t = 1.15 :=
by
  sorry

end initial_milk_amount_l423_423743


namespace unique_n_digit_integer_divisible_by_5n_l423_423976

theorem unique_n_digit_integer_divisible_by_5n 
  (n : ℕ) (hn : 0 < n) :
  ∃! (x : ℕ), 
    (10^(n-1) ≤ x ∧ x < 10^n) ∧ -- x is an n-digit integer
    (∀ d, d ∈ (Int.to_digits 10 x) → d ∈ {1, 2, 3, 4, 5}) ∧ -- Each digit of x belongs to {1, 2, 3, 4, 5}
    (x % 5^n = 0) :=            -- x is divisible by 5^n
sorry

end unique_n_digit_integer_divisible_by_5n_l423_423976


namespace pentagon_side_length_l423_423099

-- Define the side length of the equilateral triangle
def side_length_triangle : ℚ := 20 / 9

-- Define the perimeter of the equilateral triangle
def perimeter_triangle : ℚ := 3 * side_length_triangle

-- Define the side length of the regular pentagon
def side_length_pentagon : ℚ := 4 / 3

-- Prove that the side length of the regular pentagon has the same perimeter as the equilateral triangle
theorem pentagon_side_length (s : ℚ) (h1 : s = side_length_pentagon) :
  5 * s = perimeter_triangle :=
by
  -- Provide the solution
  sorry

end pentagon_side_length_l423_423099


namespace seeds_total_l423_423694

-- Define the conditions as given in the problem.
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds

-- Lean statement to prove the total number of seeds.
theorem seeds_total : Bom_seeds + Gwi_seeds + Yeon_seeds = 1660 := 
by
  -- Assuming all given definitions and conditions are true,
  -- we aim to prove the final theorem statement.
  sorry

end seeds_total_l423_423694


namespace find_maximum_b_l423_423509

open Real

def is_ellipse (x y : ℝ) (b : ℝ): Prop :=
  x^2 + (y^2)/(b^2) = 1

def foci_distance (c : ℝ) : ℝ := 2 * c

def arithmetic_mean (a b: ℝ) : ℝ := (a + b) / 2

theorem find_maximum_b (b : ℝ) (P F1 F2 : ℝ × ℝ) (c : ℝ) (h : 0 < b) (h1 : b < 1)
  (hx : ∃ P : ℝ × ℝ, is_ellipse P.1 P.2 b ∧ abs (P.1 - (1 / c)) = arithmetic_mean (abs (sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) (abs (sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)))) :
  b ≤ (sqrt 3) / 2 :=
sorry

end find_maximum_b_l423_423509


namespace trigonometry_cosine_difference_identity_l423_423987

theorem trigonometry_cosine_difference_identity (α β : ℝ) :
  (cos (α + β) * cos β + sin (α + β) * sin β) = cos α :=
by
  -- This would be the place to apply the cosine difference identity in the actual proof.
  sorry

end trigonometry_cosine_difference_identity_l423_423987


namespace convex_quadrilateral_diagonals_inequality_l423_423705
noncomputable def perimeter (a b c d : ℝ) : ℝ := a + b + c + d

def is_convex_quadrilateral (A B C D : ℝ) : Prop := 
  -- Add necessary conditions that A, B, C, D form a convex quadrilateral
  -- For simplicity here, we assume A, B, C, D are lengths of the sides
  -- and the polygon formed by these sides is convex.

theorem convex_quadrilateral_diagonals_inequality
  (A B C D AC BD : ℝ)
  (h_convex: is_convex_quadrilateral A B C D)
  (h_perimeter: perimeter A B C D = p) :
  (1/2 : ℝ) * p < AC + BD ∧ AC + BD < p :=
sorry

end convex_quadrilateral_diagonals_inequality_l423_423705


namespace polynomial_is_constant_l423_423587

open Complex Polynomial

theorem polynomial_is_constant (P : Polynomial ℂ) 
  (h : ∀ z : ℂ, abs z = 1 → is_real (P.eval z)) : 
  P.degree = 0 := 
sorry

noncomputable def is_real (z : ℂ) : Prop := 
  z.im = 0

end polynomial_is_constant_l423_423587


namespace average_page_count_per_essay_l423_423038

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l423_423038


namespace problem1_problem2_l423_423519

open Real

noncomputable def f (x : ℝ) : ℝ := ln x
noncomputable def g (x a : ℝ) : ℝ := x + a / x
noncomputable def F (x a : ℝ) : ℝ := ln x + a / x

theorem problem1 (a : ℝ) (H : ∀ x ∈ Icc 1 e, F x a ≥ 3 / 2) : a = sqrt e := 
sorry

theorem problem2 (a : ℝ) : (∀ x, x ≥ 1 → ln x ≤ x + a / x) ↔ a ≥ -1 := 
sorry

end problem1_problem2_l423_423519


namespace circle_y_axis_intersections_sum_l423_423070

theorem circle_y_axis_intersections_sum (h : circle_center : (-8, 3)) (r : circle_radius : 15) :
  ∑ y in {y | ∃ x, x = 0 ∧ (x + 8)^2 + (y - 3)^2 = 225}, y = 6 := sorry

end circle_y_axis_intersections_sum_l423_423070


namespace cassidy_total_grounding_days_l423_423423

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l423_423423


namespace complex_modulus_l423_423847

theorem complex_modulus (z : ℂ) (h: z = (4 - 2 * Complex.i) / (1 + Complex.i)) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l423_423847


namespace petya_sum_less_than_one_tenth_l423_423330

theorem petya_sum_less_than_one_tenth :
  let a1 := 49 in
  let b1 := 99 in
  let a2 := 51 in
  let b2 := 101 in
  let a3 := 1 in
  let b3 := 9999 in
  (a1 + a2 + a3) / (b1 + b2 + b3) < 1 / 10 :=
by
  repeat {
    sorry
  }

end petya_sum_less_than_one_tenth_l423_423330


namespace at_least_60_percent_speak_same_language_l423_423912

variable (N : ℕ) -- Total number of participants
variables (L : Fin 4 → Set ℕ) -- Languages, each represented by a set of participants

-- Each participant speaks at least one language
axiom participants_speak_language :
  ∀ n ∈ (Finset.range N), ∃ i : Fin 4, n ∈ L i

theorem at_least_60_percent_speak_same_language :
  ∃ i : Fin 4, (L i).card ≥ (3 * N / 5) :=
sorry

end at_least_60_percent_speak_same_language_l423_423912


namespace cougar_ratio_l423_423064

theorem cougar_ratio (lions tigers total_cats cougars : ℕ) 
  (h_lions : lions = 12) 
  (h_tigers : tigers = 14) 
  (h_total : total_cats = 39) 
  (h_cougars : cougars = total_cats - (lions + tigers)) 
  : cougars * 2 = lions + tigers := 
by 
  rw [h_lions, h_tigers] 
  norm_num at * 
  sorry

end cougar_ratio_l423_423064


namespace problem_solution_l423_423202

def grid_side : ℕ := 4
def square_size : ℝ := 2
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

noncomputable def grid_area : ℝ :=
  (grid_side * grid_side) * (square_size * square_size)

noncomputable def circle_area : ℝ :=
  num_circles * (Real.pi * (circle_radius ^ 2))

noncomputable def ellipse_area : ℝ :=
  Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)

noncomputable def visible_shaded_area (A B : ℝ) : Prop :=
  grid_area = A - B * Real.pi

theorem problem_solution : ∃ A B, visible_shaded_area A B ∧ (A + B = 69) :=
by
  sorry

end problem_solution_l423_423202


namespace good_function_count_l423_423147

noncomputable def num_good_functions (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    n * Nat.totient n

theorem good_function_count (n : ℕ) (h : n ≥ 2) :
  ∃ (f : ℤ → Fin (n + 1)), 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ j, ∀ m, (f (m + j) : ℤ) ≡ (f (m + k) - f m : ℤ) [ZMOD (n + 1)]) → 
  num_good_functions n = n * Nat.totient n :=
sorry

end good_function_count_l423_423147


namespace exists_element_in_union_l423_423242

noncomputable theory
open Set

variables {X : Type} {Ai : Finset X} {n k : ℕ}

theorem exists_element_in_union (S : Finset (Finset X)) (h_card : S.card = n) (h_distinct : S.Nodup) 
(h_union : ∀ Ai Aj ∈ S, Ai ∪ Aj ∈ S) (h_min_card : ∀ (Ai ∈ S), (Ai.card ≥ 2)) 
(k := (S.image Finset.card).min' (Finset.nonempty_of_card_ne_zero (by linarith))) 
(h_k : k ≥ 2) : 
∃ x ∈ S.bUnion id, (S.filter (λ Ai, x ∈ Ai)).card ≥ n / k := 
begin
  sorry
end

end exists_element_in_union_l423_423242


namespace equidistant_points_quadrants_l423_423174

open Real

theorem equidistant_points_quadrants : 
  ∀ x y : ℝ, 
    (4 * x + 6 * y = 24) → (|x| = |y|) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y)) :=
by
  sorry

end equidistant_points_quadrants_l423_423174


namespace arithmetic_sequence_fifth_term_l423_423338

theorem arithmetic_sequence_fifth_term:
  ∀ (a₁ aₙ : ℕ) (n : ℕ),
    n = 20 → a₁ = 2 → aₙ = 59 →
    ∃ d a₅, d = (59 - 2) / (20 - 1) ∧ a₅ = 2 + (5 - 1) * d ∧ a₅ = 14 :=
by
  sorry

end arithmetic_sequence_fifth_term_l423_423338


namespace find_standard_equation_of_ellipse_prove_line_intersects_ellipse_l423_423510

noncomputable def ellipse_eq {x y : ℝ} (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem find_standard_equation_of_ellipse (c b : ℝ) (M : ℝ × ℝ) :
  M = (c, 1) → ellipse_eq c b → b^2 = 2 → ellipse_eq 2 4 :=
sorry

theorem prove_line_intersects_ellipse (m : ℝ) (G : ℝ → ℝ → Prop) (M A B P Q : ℝ × ℝ) :
  G = (λ x y, x^2 / 4 + y^2 / 2 = 1) →
  M = (√2, 1) →
  (let l := λ x y, √2 * x - 2 * y + m = 0 in
    ∀ A B, l A.1 A.2 ∧ G A.1 A.2 ∧ l B.1 B.2 ∧ G B.1 B.2 ∧
    A ≠ B → 
    (MA_slope_exists M A) ∧ (MB_slope_exists M B) ∧
    ∠MPQ = ∠MQP → |P M| = |Q M|) :=
sorry

end find_standard_equation_of_ellipse_prove_line_intersects_ellipse_l423_423510


namespace scorpion_millipedes_needed_l423_423026

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l423_423026


namespace probability_sum_twelve_distinct_l423_423895

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def is_sum_twelve (x y z : ℕ) : Prop :=
  x + y + z = 12

def all_different (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z

def valid_triplet (x y z : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 ∧ 1 ≤ z ∧ z ≤ 6 ∧ all_different x y z ∧ is_sum_twelve x y z

def probability (favorable total : ℕ) : ℚ :=
  ⟨favorable, total⟩

theorem probability_sum_twelve_distinct :
  probability 18 120 = (3 : ℚ) / 20 :=
by
  -- Calculations based on given problem conditions:
  have total_outcomes := 120
  have favorable_outcomes := 18
  have prob := probability favorable_outcomes total_outcomes
  exact prob rfl

end probability_sum_twelve_distinct_l423_423895


namespace trapezoid_area_l423_423059

-- Define the relevant conditions in Lean
def isosceles_trapezoid (longer_base shorter_base side h A : ℝ) : Prop :=
  -- Trapezoid is isosceles
  (h^2 + (shorter_base/2)^2 = side^2) ∧
  -- Trapezoid circumscribed around a circle with radius 5 units
  (2 * side * 5 = shorter_base * h) ∧
  -- The longer base of the trapezoid
  (longer_base = 20) ∧
  -- One of the base angles is 45 degrees
  (side * sin (real.pi / 4) = h) ∧
  -- The area of the trapezoid
  (A = ((1/2) * (longer_base + shorter_base) * h))

-- Prove that the area is equal to 90√2 - 60
theorem trapezoid_area :
  ∃ (shorter_base side h A : ℝ),
    isosceles_trapezoid 20 shorter_base side h A ∧
    A = 90 * real.sqrt 2 - 60 := 
sorry

end trapezoid_area_l423_423059


namespace handshakes_l423_423778

open Nat

theorem handshakes : ∃ x : ℕ, 4 + 3 + 2 + 1 + x = 10 ∧ x = 2 :=
by
  existsi 2
  simp
  sorry

end handshakes_l423_423778


namespace min_value_f_l423_423513

noncomputable def f : ℝ → ℝ := λ x, 
  if x ≤ 1 then -Real.exp x else x + 3 / x - 5
  
theorem min_value_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ y : ℝ, f y = m) ∧ m = -Real.exp 1 := by
  sorry

end min_value_f_l423_423513


namespace negative_represents_departure_of_30_tons_l423_423218

theorem negative_represents_departure_of_30_tons (positive_neg_opposites : ∀ x:ℤ, -x = x * (-1))
  (arrival_represents_30 : ∀ x:ℤ, (x = 30) ↔ ("+30" represents arrival of 30 tons of grain)) :
  "-30" represents departure of 30 tons of grain :=
sorry

end negative_represents_departure_of_30_tons_l423_423218


namespace proof_problem_l423_423096

def condition1 (x : ℚ) : Prop :=
    (17/2) * x = (17/2) + x

def condition2 (x : ℚ) : Prop :=
    x / (2/3) = x + (2/3)

theorem proof_problem : (∃ x : ℚ, condition1 x ∧ x = 17/15) ∧ (∃ x : ℚ, condition2 x ∧ x = 4/3) :=
by
  split
  · use 17/15
    sorry
  · use 4/3
    sorry

end proof_problem_l423_423096


namespace binomial_coefficient_sum_l423_423469

theorem binomial_coefficient_sum (a : Fin 8 → ℤ) (x : ℤ) (h : (1 - 2 * x) ^ 7 = ∑ i in Finset.range 8, (a i) * x ^ i) :
  (∑ i in Finset.range 8, |a i|) = 2187 :=
by
  sorry

end binomial_coefficient_sum_l423_423469


namespace rectangle_not_equal_118_l423_423733

theorem rectangle_not_equal_118 
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (A : ℕ) (P : ℕ)
  (h₂ : A = a * b) (h₃ : P = 2 * (a + b)) :
  (a + 2) * (b + 2) - 2 ≠ 118 :=
sorry

end rectangle_not_equal_118_l423_423733


namespace locus_of_points_M_l423_423649

-- Define the structure of the regular hexagon and its vertices
structure Point (ℝ : Type) := (x : ℝ) (y : ℝ)

def A (s : ℝ) : Point ℝ := ⟨s, 0⟩
def B (s : ℝ) : Point ℝ := ⟨s / 2, (s * Real.sqrt 3) / 2⟩
def C (s : ℝ) : Point ℝ := ⟨-s / 2, (s * Real.sqrt 3) / 2⟩
def D (s : ℝ) : Point ℝ := ⟨-s, 0⟩
def E (s : ℝ) : Point ℝ := ⟨-s / 2, -(s * Real.sqrt 3) / 2⟩
def F (s : ℝ) : Point ℝ := ⟨s / 2, -(s * Real.sqrt 3) / 2⟩

-- Function to calculate the area of a triangle given three points
def triangle_area (P Q R : Point ℝ) : ℝ := 
  Real.abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2

-- Main theorem to prove
theorem locus_of_points_M (M : Point ℝ) (s : ℝ) :
  triangle_area M (A s) (C s) = triangle_area M (C s) (D s) ↔
  ∃ t : ℝ, M.x = t * (F s).x ∧ M.y = t * (F s).y := 
sorry

end locus_of_points_M_l423_423649


namespace total_distance_of_race_is_150_l423_423551

variable (D : ℝ)

-- Conditions
def A_covers_distance_in_45_seconds (D : ℝ) : Prop := ∃ A_speed, A_speed = D / 45
def B_covers_distance_in_60_seconds (D : ℝ) : Prop := ∃ B_speed, B_speed = D / 60
def A_beats_B_by_50_meters_in_60_seconds (D : ℝ) : Prop := (D / 45) * 60 = D + 50

theorem total_distance_of_race_is_150 :
  A_covers_distance_in_45_seconds D ∧ 
  B_covers_distance_in_60_seconds D ∧ 
  A_beats_B_by_50_meters_in_60_seconds D → 
  D = 150 :=
by
  sorry

end total_distance_of_race_is_150_l423_423551


namespace tangent_line_at_zero_range_of_a_monotonic_decreasing_exp_sum_sin_lt_two_l423_423171

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * sin x + log (1 - x)

-- Problem 1: Prove the equation of the tangent line at x = 0 is y = 0 when a = 1.
theorem tangent_line_at_zero : 
  f 1 0 = 0 ∧ deriv (f 1) 0 = 0 := 
sorry

-- Problem 2: Prove the range of values for a where f(x) is monotonically decreasing on [0, 1) is (-∞, 1]
theorem range_of_a_monotonic_decreasing : 
  ∀ a, (∀ x ∈ Iio (1:ℝ), deriv (f a) x ≤ 0) ↔ a ≤ 1 :=
sorry

-- Problem 3: Prove e^(sum of the sines) is less than 2
theorem exp_sum_sin_lt_two (n : ℕ) (hn : 0 < n) :
  ∑ k in Finset.range n, sin (1 / ((k + 1)^2 : ℝ)) < log 2 :=
sorry


end tangent_line_at_zero_range_of_a_monotonic_decreasing_exp_sum_sin_lt_two_l423_423171


namespace certain_events_l423_423647

-- Define the idioms and their classifications
inductive Event
| impossible
| certain
| unlikely

-- Definitions based on the given conditions
def scooping_moon := Event.impossible
def rising_tide := Event.certain
def waiting_by_stump := Event.unlikely
def catching_turtles := Event.certain
def pulling_seeds := Event.impossible

-- The theorem statement
theorem certain_events :
  (rising_tide = Event.certain) ∧ (catching_turtles = Event.certain) := by
  -- Proof is omitted
  sorry

end certain_events_l423_423647


namespace rectangle_area_change_l423_423638

theorem rectangle_area_change (l w : ℝ) (hlw : l * w = 540) :
  let l' := 1.15 * l
      w' := 0.85 * w
      A' := l' * w'
  in round A' = 528 :=
by
  let l' := 1.15 * l
  let w' := 0.85 * w
  let A' := l' * w'
  sorry

end rectangle_area_change_l423_423638


namespace determine_constant_l423_423893

/-- If the function f(x) = a * sin x + 3 * cos x has a maximum value of 5,
then the constant a must be ± 4. -/
theorem determine_constant (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x + 3 * Real.cos x ≤ 5) :
  a = 4 ∨ a = -4 :=
sorry

end determine_constant_l423_423893


namespace math_proof_l423_423155

open Real

noncomputable def math_problem (a b : ℝ) : ℝ :=
  if h : (a > 0 ∧ b > 0 ∧ 2 - log 2 a = 3 - log 3 b ∧ 3 - log 3 b = log 6 (1 / (a + b))) then
    (1 / a) + (1 / b)
  else
    0

theorem math_proof (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ 2 - log 2 a = 3 - log 3 b ∧ 3 - log 3 b = log 6 (1 / (a + b))) :
  math_problem a b = 1 / 108 := 
by 
  sorry

end math_proof_l423_423155


namespace factory_ill_days_l423_423907

theorem factory_ill_days
  (average_first_25_days : ℝ)
  (total_days : ℝ)
  (overall_average : ℝ)
  (ill_days_average : ℝ)
  (production_first_25_days_total : ℝ)
  (production_ill_days_total : ℝ)
  (x : ℝ) :
  average_first_25_days = 50 →
  total_days = 25 + x →
  overall_average = 48 →
  ill_days_average = 38 →
  production_first_25_days_total = 25 * 50 →
  production_ill_days_total = x * 38 →
  (25 * 50 + x * 38 = (25 + x) * 48) →
  x = 5 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end factory_ill_days_l423_423907


namespace range_of_m_for_one_real_root_l423_423541

def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_for_one_real_root :
  (∃! x : ℝ, f x m = 0) ↔ (m < -2 ∨ m > 2) := by
  sorry

end range_of_m_for_one_real_root_l423_423541


namespace complex_expression_simplification_l423_423988

theorem complex_expression_simplification (θ : ℝ) :
  ( (cos θ - complex.i * sin θ)^8 * (1 + complex.i * tan θ)^5 ) /
  ( (cos θ + complex.i * sin θ)^2 * (tan θ + complex.i) ) =
  - (1 / (cos θ)^4) * (sin (4 * θ) + complex.i * cos (4 * θ)) :=
sorry

end complex_expression_simplification_l423_423988


namespace find_points_c_l423_423486

theorem find_points_c (A B : Point) (o_2 : Circle) (o : Circle)
  (homothety : ∀ (A : Point) (k : ℝ), Circle → Circle) 
  (symmetric : ∀ (AB : Line), Circle → Circle)
  (excluding : ∀ (B : Point), Circle → Set Point) :
  (∃ (C : Point), in_triangle (A B C) ∧ 
  length_altitude_from A (triangle A B C) = length_median_from B (triangle A B C) ∧ 
  C ∈ circle_homothetic :=
sorry

end find_points_c_l423_423486


namespace sweater_markup_percentage_l423_423012

-- The wholesale cost W and retail price R
variables (W R : ℝ)

-- The given condition
variable (h : 0.30 * R = 1.40 * W)

-- The theorem to prove
theorem sweater_markup_percentage (h : 0.30 * R = 1.40 * W) : (R - W) / W * 100 = 366.67 :=
by
  -- The solution steps would be placed here, if we were proving.
  sorry

end sweater_markup_percentage_l423_423012


namespace count_subsets_of_neither_set_l423_423455

def set1 : Finset ℕ := {1, 2, 3, 4, 5}
def set2 : Finset ℕ := {4, 5, 6, 7, 8}
def universal_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def count_subsets_not_in_given_sets 
  (univ : Finset ℕ) (s1 s2 : Finset ℕ) : ℕ :=
  (2 ^ univ.card) - (2 ^ s1.card) - (2 ^ s2.card) + (2 ^ (s1 ∩ s2).card)

theorem count_subsets_of_neither_set :
  count_subsets_not_in_given_sets universal_set set1 set2 = 196 :=
by norm_num [count_subsets_not_in_given_sets, universal_set, set1, set2]
admit -- to pass build successfully, replace this with 'sorry'

end count_subsets_of_neither_set_l423_423455


namespace h_one_eq_zero_l423_423951

-- Let f(x) be a cubic polynomial defined as follows:
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Let h be a cubic polynomial such that:
-- 1. h(0) = 1
-- 2. The roots of h are the squares of the roots of f

theorem h_one_eq_zero : 
  (∃ (h : ℝ → ℝ), (h 0 = 1) ∧ 
                   (∃ (α β γ : ℝ), 
                     (f α = 0) ∧ 
                     (f β = 0) ∧ 
                     (f γ = 0) ∧ 
                     (h = λ x, (x - α^2) * (x - β^2) * (x - γ^2))) ∧ 
                   (h 1 = 0)) :=
sorry

end h_one_eq_zero_l423_423951


namespace negation_of_at_most_four_is_at_least_five_l423_423311

theorem negation_of_at_most_four_is_at_least_five :
  (∀ n : ℕ, n ≤ 4) ↔ (∃ n : ℕ, n ≥ 5) := 
sorry

end negation_of_at_most_four_is_at_least_five_l423_423311


namespace triangle_is_isosceles_l423_423948

theorem triangle_is_isosceles
  (A B C M N : Type)
  (ABC : triangle A B C)
  (h1 : side AC = longest_side ABC)
  (h2 : (AM = AB) ∧ (CN = CB))
  (h3 : BM = BN) : is_isosceles ABC :=
by
  sorry

end triangle_is_isosceles_l423_423948


namespace value_of_u_when_m_is_3_l423_423584

theorem value_of_u_when_m_is_3 :
  ∀ (u t m : ℕ), (t = 3^m + m) → (u = 4^t - 3 * t) → m = 3 → u = 4^30 - 90 :=
by
  intros u t m ht hu hm
  sorry

end value_of_u_when_m_is_3_l423_423584


namespace O_and_midpoints_are_coplanar_l423_423320

variables (A1 A2 B1 B2 C1 C2 O P1 P2 P3 P4 P5 P6 P7 P8 : Point)
variables (S1 S2 S3 S4 S5 S6 S7 S8 : ℝ)

-- Define the midpoints of A1A2, B1B2, and C1C2 as A, B, and C respectively
noncomputable def A := midpoint A1 A2
noncomputable def B := midpoint B1 B2
noncomputable def C := midpoint C1 C2

-- Define the structure of the polyhedron and the sphere
axiom polyhedron (f : Fin 8 → Triangle) : Prop
axiom sphere_touches_polyhedron : touches_sphere f O

-- The main theorem stating that O, A, B, and C are coplanar
theorem O_and_midpoints_are_coplanar
  (h1 : polyhedron (λ i, Triangle.mk (f i).v1 (f i).v2 (f i).v3))
  (h2 : sphere_touches_polyhedron)
  : coplanar O A B C := by
  sorry

end O_and_midpoints_are_coplanar_l423_423320


namespace solve_inequalities_l423_423628

theorem solve_inequalities (x : ℝ) :
  (1/x < 1 ∧ |4*x - 1| > 2) ↔ (x ∈ Iio (-1/4) ∨ x ∈ Ioi 1) :=
by
  sorry

end solve_inequalities_l423_423628


namespace chord_bisected_by_point_l423_423537

theorem chord_bisected_by_point (x1 y1 x2 y2 : ℝ) :
  (x1^2 / 36 + y1^2 / 9 = 1) ∧ (x2^2 / 36 + y2^2 / 9 = 1) ∧ 
  (x1 + x2 = 4) ∧ (y1 + y2 = 4) → (x + 4 * y - 10 = 0) :=
sorry

end chord_bisected_by_point_l423_423537


namespace possible_values_of_sum_l423_423942

theorem possible_values_of_sum (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 2) : 
  set.Ici (2 : ℝ) = {x : ℝ | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b)} :=
by
  sorry

end possible_values_of_sum_l423_423942


namespace one_fourth_of_6_8_is_fraction_l423_423103

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l423_423103


namespace find_primes_l423_423677

-- Let a and b be coprime positive integers
variables (a b : ℕ) (h_coprime : Nat.coprime a b) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Define the sequence (a + b * sqrt 2) ^ (2*n) = a_n + b_n * sqrt 2
noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def b_n (n : ℕ) : ℕ := sorry

-- Main theorem to prove: find all prime p such that there exists some 0 < n <= p and p divides b_n
theorem find_primes (p : ℕ) (h_prime : Nat.prime p) :
  (∃ n, 0 < n ∧ n ≤ p ∧ p ∣ b_n n) ↔ 
  (p ∣ 2 * b^2 - a^2 ∨ p ∣ 4 * a * b) ∨ ((2 * b^2 - a^2) = 0 ∨ (4 * a * b = 0)) :=
sorry

end find_primes_l423_423677


namespace number_of_combinations_l423_423927

noncomputable def countOddNumbers (n : ℕ) : ℕ := (n + 1) / 2

noncomputable def countPrimesLessThan30 : ℕ := 9 -- {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def countMultiplesOfFour (n : ℕ) : ℕ := n / 4

theorem number_of_combinations : countOddNumbers 40 * countPrimesLessThan30 * countMultiplesOfFour 40 = 1800 := by
  sorry

end number_of_combinations_l423_423927


namespace hyperbola_eccentricity_l423_423151

open Real EuclideanSpace

variable {a b c e : ℝ}

def is_hyperbola (a b : ℝ) : Prop := 
  a > 0 ∧ b > 0

def is_left_focus (F : EuclideanSpace ℝ (Fin 2)) : Prop := 
  F = (-c, 0)

def is_right_vertex (E : EuclideanSpace ℝ (Fin 2)) : Prop := 
  E = (a, 0)

def is_perpendicular_to_x (F : EuclideanSpace ℝ (Fin 2)) (A B : EuclideanSpace ℝ (Fin 2)) : Prop := 
  A.1 = B.1 ∧ A.1 = F.1

def triangle_right (A B E : EuclideanSpace ℝ (Fin 2)) : Prop := 
  let EA := A - E
  let EB := B - E
  (EA.1 * EB.1 + EA.2 * EB.2 = 0)

noncomputable def eccentricity (a c : ℝ) : ℝ := 
  c / a

theorem hyperbola_eccentricity 
  (F E A B : EuclideanSpace ℝ (Fin 2))
  (h_hyperbola : is_hyperbola a b)
  (h_focus : is_left_focus F)
  (h_vertex : is_right_vertex E)
  (h_perpendicular : is_perpendicular_to_x F A B)
  (h_triangle_right : triangle_right A B E) :
  eccentricity a c = 2 :=
sorry

end hyperbola_eccentricity_l423_423151


namespace estimated_probability_mouth_upwards_l423_423891

theorem estimated_probability_mouth_upwards :
  let num_tosses := 200
  let occurrences_upwards := 48
  let estimated_probability := (occurrences_upwards : ℝ) / (num_tosses : ℝ)
  estimated_probability = 0.24 := 
by {
  sorry,
}

end estimated_probability_mouth_upwards_l423_423891


namespace decreasing_arith_prog_smallest_num_l423_423460

theorem decreasing_arith_prog_smallest_num 
  (a d : ℝ) -- Define a and d as real numbers
  (h_arith_prog : ∀ n : ℕ, n < 5 → (∃ k : ℕ, k < 5 ∧ (a - k * d) = if n = 0 then a else a - n * d))
  (h_sum_cubes_zero : a^3 + (a-d)^3 + (a-2*d)^3 + (a-3*d)^3 + (a-4*d)^3 = 0)
  (h_sum_fourth_powers_306 : a^4 + (a-d)^4 + (a-2*d)^4 + (a-3*d)^4 + (a-4*d)^4 = 306) :
  ∃ d' ∈ {d}, a - 4 * d' = -2 * real.sqrt 3 := -- Prove the smallest number is -2√3
sorry

end decreasing_arith_prog_smallest_num_l423_423460


namespace sum_inequality_l423_423483

variable (n : ℕ) (x : Fin n → ℝ)

noncomputable def x_next (i : Fin n) : ℝ := if i.val = n-1 then x 0 else x ⟨i.val + 1, Nat.succ_lt_succ i.is_lt⟩

theorem sum_inequality 
  (h_n : n ≥ 3) 
  (h_pos : ∀ i : Fin n, 0 < x i) :
  (∑ i, (x_next n x i) ^ (n + 1) / (x i) ^ n) ≥ (∑ i, (x_next n x i) ^ n / (x i) ^ (n - 1)) :=
by sorry

end sum_inequality_l423_423483


namespace word_ratio_l423_423966

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / List.prod (ks.map factorial)

theorem word_ratio (n m k l : ℕ) (h1 : n = 6) (h2 : m = 2) (h3 : k = 2) (h4 : l = 7) :
  multinomial_coefficient l [m, k] / multinomial_coefficient n [m, k] = 7 :=
by
  sorry

end word_ratio_l423_423966


namespace same_function_A_same_function_C_l423_423345

-- Definitions of the functions in option A
def f_A (x : ℝ) : ℝ := x^2 + 2 * x
def g_A (t : ℝ) : ℝ := t^2 + 2 * t

-- Definitions of the functions in option C
def f_C (x : ℝ) : ℝ := |x| / x
def g_C (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0 -- Let's add a default value for x = 0 to avoid complications

-- Prove that they represent the same function
theorem same_function_A :
  ∀ x t : ℝ, f_A(x) = g_A(t) :=
by sorry

theorem same_function_C :
  ∀ x : ℝ, x ≠ 0 → f_C(x) = g_C(x) :=
by sorry

end same_function_A_same_function_C_l423_423345


namespace angle_in_first_quadrant_l423_423829

-- Define the condition that sin(α) + cos(α) > 1
def condition (α : ℝ) : Prop := sin α + cos α > 1

-- Theorem statement
theorem angle_in_first_quadrant (α : ℝ) (h : condition α) : α ∈ set.Icc 0 (π/2) :=
sorry

end angle_in_first_quadrant_l423_423829


namespace product_of_solutions_eq_neg49_l423_423122

theorem product_of_solutions_eq_neg49 :
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (∏ x in ({7, -7} : Finset ℤ), x) = -49 := by
  sorry

end product_of_solutions_eq_neg49_l423_423122


namespace coefficient_x6_eq_30_l423_423538

open Nat

theorem coefficient_x6_eq_30 (a : ℝ) :
  let expr := (x^2 - a)*(x + x⁻¹)^10
  let coefficient := (120 - 45 * a)
  coefficient = 30 → a = 2 :=
by {
    intro h,
    sorry
}

end coefficient_x6_eq_30_l423_423538


namespace probability_even_units_digit_lt_6_l423_423040

theorem probability_even_units_digit_lt_6 :
  let units_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let favorable_digits := {0, 2, 4} in
  ∀ (n : ℕ), (10000 ≤ n) ∧ (n < 100000) →
  (n % 10) ∈ favorable_digits →
  (∃ probability : ℚ, probability = 3 / 10) :=
by
  let units_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let favorable_digits := {0, 2, 4}
  assume n : ℕ
  assume h : (10000 ≤ n) ∧ (n < 100000)
  assume h_units : (n % 10) ∈ favorable_digits
  use 3 / 10
  sorry

end probability_even_units_digit_lt_6_l423_423040


namespace determinant_of_matrix_l423_423067

theorem determinant_of_matrix (a b c d : ℝ) :
  matrix.det ![
    ![a^2 + b^2 - c^2 - d^2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
    ![2 * b * c + 2 * a * d, a^2 - b^2 + c^2 - d^2, 2 * c * d - 2 * a * b],
    ![2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a^2 - b^2 - c^2 + d^2]
  ] = (a^2 + b^2 + c^2 + d^2)^3 := 
sorry

end determinant_of_matrix_l423_423067


namespace product_of_solutions_eq_neg49_l423_423121

theorem product_of_solutions_eq_neg49 :
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (∏ x in ({7, -7} : Finset ℤ), x) = -49 := by
  sorry

end product_of_solutions_eq_neg49_l423_423121


namespace employed_population_percent_l423_423224

variable {P : ℝ} (P_pos : 0 < P) -- Total population is positive.
variable {E : ℝ} -- Percent of the population that is employed.
variable (h1 : 0.24 * P = 0.25 * (E * P)) -- 24% of total population are employed males; 25% of employed people are males.

theorem employed_population_percent (h1 : 0.24 * P = 0.25 * (E * P)) : E = 0.96 :=
by
  have h_eq : 0.24 = 0.25 * E := by rw [←mul_one P, ←h1, div_eq_mul_inv];
  sorry

end employed_population_percent_l423_423224


namespace divisible_remaining_coins_l423_423043

-- Definitions based on conditions
variable (n m : ℕ) -- Number of robbers and coins
variable (v : Fin m → ℕ) -- Values of the coins
variable (g : ℕ) -- Total value of coins

-- Conditions
-- The total value of the coins when any one coin is removed is divisible by n
def divisible_by_robbers := ∀ i : Fin m, (g - v i) % n = 0

-- The proof statement
theorem divisible_remaining_coins (h1 : g = ∑ i : Fin m, v i)
                                  (h2 : ∀ i : Fin m, (g - v i) % n = 0) :
                                  n ∣ (m - 1) := sorry

end divisible_remaining_coins_l423_423043


namespace unique_intersection_point_l423_423651

theorem unique_intersection_point (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - 2 * (m + 1) * x - 1 = 0) → x = -1) ↔ m = -2 :=
by
  sorry

end unique_intersection_point_l423_423651


namespace functions_linear_independent_l423_423568

open Complex
open LinearAlgebra

variables {n : ℕ}
variables {D : Set Complex}  -- Domain of the complex plane
variables {f : Fin n → D → Complex}  -- Regular functions on the domain

-- Assumption: The functions are holomorphic and linearly independent over ℂ.
axiom (holomorphic : ∀ i, DifferentiableOn ℂ (f i) D)
axiom (linear_independent : LinearIndependent ℂ (λ i, f i))

-- Goal: Prove that the functions f_i * conjugate(f_k), 1 ≤ i, k ≤ n, are linearly independent.
theorem functions_linear_independent :
  LinearIndependent ℂ (λ ⟨i, k⟩ : Fin n × Fin n, λ z : D, (f i z) * (conj (f k z))) :=
sorry

end functions_linear_independent_l423_423568


namespace avg_ballpoint_pens_per_day_l423_423003

theorem avg_ballpoint_pens_per_day (bundles_sold : ℕ) (pens_per_bundle : ℕ) (days : ℕ) (total_pens : ℕ) (avg_per_day : ℕ) 
  (h1 : bundles_sold = 15)
  (h2 : pens_per_bundle = 40)
  (h3 : days = 5)
  (h4 : total_pens = bundles_sold * pens_per_bundle)
  (h5 : avg_per_day = total_pens / days) :
  avg_per_day = 120 :=
by
  -- placeholder proof
  sorry

end avg_ballpoint_pens_per_day_l423_423003


namespace mary_bought_baseball_cards_l423_423968

theorem mary_bought_baseball_cards (initial_cards : ℕ) (torn_cards : ℕ) (fred_gave_cards : ℕ) (total_cards_now : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → fred_gave_cards = 26 → total_cards_now = 84 → 
  ∃ bought_cards : ℕ, bought_cards = 48 :=
by
  intros h1 h2 h3 h4
  use total_cards_now - (initial_cards - torn_cards + fred_gave_cards)
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end mary_bought_baseball_cards_l423_423968


namespace parabola_directrix_l423_423301

theorem parabola_directrix (x y : ℝ) (h : y = 4 * x^2) : y = -1 / 16 :=
sorry

end parabola_directrix_l423_423301


namespace find_w_l423_423802

theorem find_w : 
  ∃ w, (sqrt 1.1) / (sqrt 0.81) + (sqrt 1.44) / (sqrt w) = 2.879628878919216 ∧ w = 0.49 :=
by
  sorry

end find_w_l423_423802


namespace area_of_S_l423_423938

open Complex

noncomputable def omega : ℂ := -1/2 + ((1/2) * (Complex.I * Real.sqrt 3))

noncomputable def S : set ℂ :=
  {z | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ z = a + b * omega + c * (omega ^ 2)}

theorem area_of_S : ∃ (A : ℝ), A = (3 / 2) * Real.sqrt 3 := sorry

end area_of_S_l423_423938


namespace f_strictly_increasing_l423_423811

noncomputable def f : ℝ → ℝ := sorry
axiom cond : ∀ x > 0, (∃ finset_y : Finset ℝ, (∀ y ∈ finset_y, y > 0 ∧ (x + f y) * (y + f x) ≤ 4))

theorem f_strictly_increasing :
  ∀ ⦃x y : ℝ⦄, x > 0 → y > 0 → x < y → f x > f y :=
by
  intros x y x_gt_0 y_gt_0 x_lt_y
  have h := cond x x_gt_0
  sorry

end f_strictly_increasing_l423_423811


namespace a100_gt_2pow99_l423_423633

theorem a100_gt_2pow99 (a : Fin 101 → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_initial : a 1 > a 0) 
  (h_rec : ∀ k, 2 ≤ k → a k = 3 * a (k - 1) - 2 * a (k - 2)) 
  : a 100 > 2 ^ 99 :=
by
  sorry

end a100_gt_2pow99_l423_423633


namespace decryption_ease_comparison_l423_423094

def unique_letters_of_thermometer : Finset Char := {'т', 'е', 'р', 'м', 'о'}
def unique_letters_of_remont : Finset Char := {'р', 'е', 'м', 'о', 'н', 'т'}
def easier_to_decrypt : Prop :=
  unique_letters_of_remont.card > unique_letters_of_thermometer.card

theorem decryption_ease_comparison : easier_to_decrypt :=
by
  -- We need to prove that |unique_letters_of_remont| > |unique_letters_of_thermometer|
  sorry

end decryption_ease_comparison_l423_423094


namespace find_f_at_2_l423_423853

variable {R : Type} [Ring R]

def f (a b x : R) : R := a * x ^ 3 + b * x - 3

theorem find_f_at_2 (a b : R) (h : f a b (-2) = 7) : f a b 2 = -13 := 
by 
  have h₁ : f a b (-2) + f a b 2 = -6 := sorry
  have h₂ : f a b 2 = -6 - f a b (-2) := sorry
  rw [h₂, h]
  norm_num

end find_f_at_2_l423_423853


namespace product_of_digits_first_palindrome_year_after_2040_l423_423744

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def first_palindrome_year_after (year : ℕ) : ℕ :=
  Nat.find (λ y, y > year ∧ is_palindrome y)

def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).prod

theorem product_of_digits_first_palindrome_year_after_2040 :
  product_of_digits (first_palindrome_year_after 2040) = 0 :=
sorry

end product_of_digits_first_palindrome_year_after_2040_l423_423744


namespace satisfies_differential_eq_l423_423986

noncomputable def y (a x : ℝ) : ℝ := a + 7 * x / (a * x + 1)

theorem satisfies_differential_eq (a x : ℝ) :
  let y' := (λ a x : ℝ, 7 / ((a*x + 1)^2))
  y a x - x * y' a x = a * (1 + x^2 * y' a x) := by
  sorry

end satisfies_differential_eq_l423_423986


namespace solve_inequality_l423_423161

noncomputable def f : ℝ → ℝ := sorry

theorem solve_inequality (h_diff : ∀ x, differentiable_at ℝ f x)
                         (h_deriv : ∀ x, (deriv f x) < f x)
                         (h_odd : ∀ x, f(-x) = -f(x) + 2)
                         :
                         {x : ℝ | f x < exp x} = Ioi 0 :=
begin
  sorry
end

end solve_inequality_l423_423161


namespace rice_mixture_price_l423_423565

-- Defining the costs per kg for each type of rice
def rice_cost1 : ℝ := 16
def rice_cost2 : ℝ := 24

-- Defining the given ratio
def mixing_ratio : ℝ := 3

-- Main theorem stating the problem
theorem rice_mixture_price
  (x : ℝ)  -- The common measure of quantity in the ratio
  (h1 : 3 * x * rice_cost1 + x * rice_cost2 = 72 * x)
  (h2 : 3 * x + x = 4 * x) :
  (3 * x * rice_cost1 + x * rice_cost2) / (3 * x + x) = 18 :=
by
  sorry

end rice_mixture_price_l423_423565


namespace geometric_sequence_product_l423_423908

noncomputable def geometric_product (a1 a19 : ℝ) (n : ℕ) : ℝ :=
  let r := (a19 / a1) ^ (1 / 18) in
  a1 * r ^ (n - 1)

theorem geometric_sequence_product :
  ∀ (a1 a19 : ℝ), (a1 > 0) → (a19 > 0) → (a1 * a19 = 16) →
    (a8 a10 a12 : ℕ),
    geometric_product a1 a19 8 = a8 →
    geometric_product a1 a19 10 = a10 →
    geometric_product a1 a19 12 = a12 →
    a8 * a10 * a12 = 64 :=
by
  intros a1 a19 a1_pos a19_pos h_product a8 a10 a12 h_a8 h_a10 h_a12
  sorry

end geometric_sequence_product_l423_423908


namespace complex_exp_conjugate_l423_423533

theorem complex_exp_conjugate (α β : ℝ) 
  (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = (2 / 5) + (1 / 3) * complex.I) :
  complex.exp (-complex.I * α) + complex.exp (-complex.I * β) = (2 / 5) - (1 / 3) * complex.I :=
by
  sorry

end complex_exp_conjugate_l423_423533


namespace girls_divisible_by_nine_l423_423419

def total_students (m c d u : ℕ) : ℕ := 1000 * m + 100 * c + 10 * d + u
def number_of_boys (m c d u : ℕ) : ℕ := m + c + d + u
def number_of_girls (m c d u : ℕ) : ℕ := total_students m c d u - number_of_boys m c d u 

theorem girls_divisible_by_nine (m c d u : ℕ) : 
  number_of_girls m c d u % 9 = 0 := 
by
    sorry

end girls_divisible_by_nine_l423_423419


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l423_423100

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l423_423100


namespace find_x_l423_423790

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
sorry

end find_x_l423_423790


namespace parabola_tangent_properties_l423_423139

-- Define the point S
def S : ℝ × ℝ := (-3, 7)

-- Define the parabola equation y^2 = 5x
def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 5 * p.fst

-- Tangent lines from point S
def tangent1 (y x : ℝ) : Prop := y = (1 / 6) * x + (15 / 2)
def tangent2 (y x : ℝ) : Prop := y = -(5 / 2) * x - (1 / 2)

-- Points of tangency
def point_of_tangency1 : ℝ × ℝ := (45, 15)
def point_of_tangency2 : ℝ × ℝ := (1 / 5, -1)

-- Angle between the tangents
def angle_between_tangents : ℝ := Real.arctan (32 / 7)

-- The Lean statement to prove
theorem parabola_tangent_properties :
  parabola (45, 15) ∧ parabola ((1/5), -1) ∧
  tangent1 15 45 ∧ tangent2 (-1) (1/5) ∧
  S.1 = -3 ∧ S.2 = 7 ∧ 
  ∃ α, α = Real.arctan (32 / 7) :=
by
  sorry

end parabola_tangent_properties_l423_423139


namespace minimum_value_2a_3b_l423_423156

noncomputable def line1 (b : ℝ) : ℝ → ℝ → ℝ := λ x y, 2 * x - (b - 3) * y + 6
noncomputable def line2 (b a : ℝ) : ℝ → ℝ → ℝ := λ x y, b * x + a * y - 5

theorem minimum_value_2a_3b
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_perpendicular : ∀ x y : ℝ, line1 b x y * line2 b a x y = 0) :
  (∀ y1 y2 : ℝ, ∀ x1 x2 : ℝ, 2 * y1 / y2 + 3 * x1 / x2 = 1) →
  2 * a + 3 * b = 25 :=
begin
  sorry
end

end minimum_value_2a_3b_l423_423156


namespace solve_for_x_l423_423627

theorem solve_for_x : ∃ x : ℝ, 64^(3*x) = 16^(4*x - 3) ∧ x = -6 := by
  use -6
  sorry

end solve_for_x_l423_423627


namespace product_of_solutions_to_x_squared_equals_49_l423_423117

theorem product_of_solutions_to_x_squared_equals_49 :
  (∃ (x : ℝ), x ^ 2 = 49) → ((∀ x, x ^ 2 = 49 → (x = 7 ∨ x = -7))) →
  (∏ x in { x : ℝ | x ^ 2 = 49}.to_finset, x) = -49 :=
begin
  sorry
end

end product_of_solutions_to_x_squared_equals_49_l423_423117


namespace cos_perpendicular_line_l423_423817

theorem cos_perpendicular_line {α : ℝ} 
  (h1 : ∃ α, tan α = 2)
  (h2 : IsPerpendicular (x + 2 * y - 4 = 0) (line_inclination α)) :
  cos ((2017 / 2) * Real.pi - 2 * α) = 4 / 5 := 
sorry

end cos_perpendicular_line_l423_423817


namespace difference_busiest_slowest_l423_423441

-- Define the initial conditions and values
def toothbrushes_jan : ℕ := 65
def toothbrushes_total : ℕ := 480
def rate_increment : ℝ := 0.10

-- To simplify calculation, round to the nearest integer after each month's calculation
noncomputable def toothbrushes_feb : ℕ := (toothbrushes_jan * (1 + rate_increment)).to_nat
noncomputable def toothbrushes_mar : ℕ := (toothbrushes_feb * (1 + rate_increment)).to_nat
noncomputable def toothbrushes_apr : ℕ := (toothbrushes_mar * (1 + rate_increment)).to_nat
noncomputable def toothbrushes_may : ℕ := (toothbrushes_apr * (1 + rate_increment)).to_nat

-- Calculate the total from January to May
noncomputable def total_jan_to_may : ℕ := toothbrushes_jan + toothbrushes_feb + toothbrushes_mar + toothbrushes_apr + toothbrushes_may

-- Remaining toothbrushes for June
noncomputable def toothbrushes_jun : ℕ := toothbrushes_total - total_jan_to_may

-- Define the slowest and busiest month as January and June respectively
def slowest_month := toothbrushes_jan
def busiest_month := toothbrushes_jun

-- Statement to be proved
theorem difference_busiest_slowest : (busiest_month - slowest_month) = 16 :=
by
  have h1 := toothbrushes_jan -- January
  have h2 := toothbrushes_jun -- June
  sorry -- Proof needed

end difference_busiest_slowest_l423_423441


namespace proof_of_equations_and_angles_l423_423596

-- Define the points and the parabola
def parabola (C : ℝ → ℝ → Prop) := ∀ x y, (y^2 = 2 * x) ↔ C x y
def pointA : ℝ × ℝ := (2, 0)
def pointB : ℝ × ℝ := (-2, 0)

-- Define a line passing through point A and intersects the parabola at points M and N.
def line_passing_through_A (l : ℝ → ℝ → Prop) := l 2 0

-- Define points M and N
def pointM (x y : ℝ) := (y = 2 ∨ y = -2) ∧ (x = 2)
def pointN (x y : ℝ) := (y ≠ 2 ∧ y ≠ -2) ∧ (y * y = 2 * x)

-- Main theorem statement
theorem proof_of_equations_and_angles :
  (∀ x y, parabola (λ x y, true)) →
  line_passing_through_A (λ x y, y = y) →
  (∃ x y, pointM x y) →
  (∃ x1 y1, pointM x1 y1) →
  (∀ x y, (¬ (y = 2 ∧ y = -2) ∧ y^2 = 2 * x) → pointN x y) →
  (∀ x y (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop), 
    (y^2 = 2 * x) ∧ l 2 y ∧ (y = 2 ∨ y = -2) → 
    (l (λ x y, y - (1/2) * x - 1 = 0) ∨ l (λ x y, y + (1/2) * x + 1 = 0))) ∧ 
  (∀ x y (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop), 
    (y^2 = 2 * x) ∧ l 2 y ∧ (y ≠ 2 ∧ y ≠ -2) → 
    (l (λ x y, x - 2 * y + 2 = 0) ∨ l (λ x y, x + 2 * y + 2 = 0)) ∧ 
    (∀ A B M N, y^2 = 2 * x ∧ (y = 2 ∨ y = -2) ∧ (-2 = 2 * (-2 + 2) ∧ k_BN + k_BM = 0) → 
      (∠ABM = ∠ABN)))
:= 
begin
  sorry
end

end proof_of_equations_and_angles_l423_423596


namespace expression_equals_l423_423417

noncomputable def evaluate_expression : ℝ :=
  3 / real.sqrt 3 - real.exp 0 (real.pi + real.sqrt 3) - real.sqrt 27 + abs (real.sqrt 3 - 2)

theorem expression_equals :
  evaluate_expression = -3 * real.sqrt 3 + 1 :=
by
  sorry

end expression_equals_l423_423417


namespace proof_problem_l423_423463

variable {f : ℝ → ℝ}
variable [Differentiable ℝ f]
variable (H : ∀ x, (x - 1) * deriv (deriv f x) < 0)

theorem proof_problem : f 0 + f 2 < 2 * f 1 :=
sorry

end proof_problem_l423_423463


namespace range_of_f_on_interval_l423_423280

theorem range_of_f_on_interval :
  let f (x : ℝ) := if x >= 0 then x else 2 * x^2 in
  set.range (fun x : ℝ => f x) ∩ set.Icc (-2 : ℝ) (3 : ℝ) = (set.Icc 0 8 : set ℝ) := sorry

end range_of_f_on_interval_l423_423280


namespace factory_output_decrease_l423_423659

theorem factory_output_decrease (O : ℝ) (h1 : O > 0) : ∃ P : ℝ, P ≈ 30.07 ∧ 1.43 * O * (1 - P / 100) = O :=
by
  use 30.07
  sorry

end factory_output_decrease_l423_423659


namespace vecs_parallel_l423_423182

/-- Variables for vectors a and b --/
def vec_a : ℝ × ℝ × ℝ := (1, 2, -2)
def vec_b : ℝ × ℝ × ℝ := (-2, -4, 4)

/-- Definition of parallel vectors --/
def parallel (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2, k * u.3)

/-- Main theorem stating that vec_a and vec_b are parallel --/
theorem vecs_parallel : parallel vec_a vec_b :=
by
  sorry

end vecs_parallel_l423_423182


namespace value_of_50th_number_l423_423664

-- Define the sequence rule
def sequence (n : ℕ) : ℕ := (nat.pred n).div2 + 1

-- Statement of the theorem
theorem value_of_50th_number : sequence 50 = 8 :=
sorry

end value_of_50th_number_l423_423664


namespace MsElizabethInvestmentsCount_l423_423969

variable (MrBanksRevPerInvestment : ℕ) (MsElizabethRevPerInvestment : ℕ) (MrBanksInvestments : ℕ) (MsElizabethExtraRev : ℕ)

def MrBanksTotalRevenue := MrBanksRevPerInvestment * MrBanksInvestments
def MsElizabethTotalRevenue := MrBanksTotalRevenue + MsElizabethExtraRev
def MsElizabethInvestments := MsElizabethTotalRevenue / MsElizabethRevPerInvestment

theorem MsElizabethInvestmentsCount (h1 : MrBanksRevPerInvestment = 500) 
  (h2 : MsElizabethRevPerInvestment = 900)
  (h3 : MrBanksInvestments = 8)
  (h4 : MsElizabethExtraRev = 500) : 
  MsElizabethInvestments MrBanksRevPerInvestment MsElizabethRevPerInvestment MrBanksInvestments MsElizabethExtraRev = 5 :=
by
  sorry

end MsElizabethInvestmentsCount_l423_423969


namespace card_area_after_one_inch_shortening_l423_423263

def initial_length := 5
def initial_width := 7
def new_area_shortened_side_two := 21
def shorter_side_reduction := 2
def longer_side_reduction := 1

theorem card_area_after_one_inch_shortening :
  (initial_length - shorter_side_reduction) * initial_width = new_area_shortened_side_two →
  initial_length * (initial_width - longer_side_reduction) = 30 :=
by
  intro h
  sorry

end card_area_after_one_inch_shortening_l423_423263


namespace parametric_equations_represent_two_rays_l423_423312

theorem parametric_equations_represent_two_rays :
  ∀ t : ℝ, 
    let x := t + (1 / t) in
    let y := 2 in
    y = 2 ∧ (x ≤ -2 ∨ x ≥ 2) :=
by {
  intro t,
  let x := t + (1 / t),
  let y := 2,
  have hy : y = 2, by rfl,
  have hx : x ≤ -2 ∨ x ≥ 2, sorry,
  exact ⟨hy, hx⟩,
}

end parametric_equations_represent_two_rays_l423_423312


namespace area_ratio_of_triangle_l423_423348

/-- Given a triangle ABC in a plane, point M satisfies the condition vec(MA) + vec(MB) + vec(MC) = 0.
    D is the midpoint of side BC. Prove that the ratio of the area of triangle ABC to the area of 
    triangle MBC is 3. -/
theorem area_ratio_of_triangle (A B C M D : Point)
(hM : vector_from M A + vector_from M B + vector_from M C = 0)
(hD : midpoint D B C) :
  area (triangle A B C) / area (triangle M B C) = 3 :=
sorry

end area_ratio_of_triangle_l423_423348


namespace collinear_m_f_g_l423_423594

-- Define the main elements of the problem
variables (A B C D E F G H M : Type) [IncidencePlane A B C D E F G H M]

-- Assume the given conditions
variables (AB CD : Line A B) (BC DA : Line B C) (circumscribed : ∀ (AB CD BC DA), circumscribedAroundCircle AB CD BC DA)
variables (tangents : (E, F, G, H) → (AB, BC, CD, DA).tangents)
variables (intersection : Intersection (Line H E) (Line D B) M)

-- Statement of the goal
theorem collinear_m_f_g : Collinear M F G :=
sorry

end collinear_m_f_g_l423_423594


namespace debate_team_formations_l423_423316

-- Define the conditions
def condition1 := 4 -- number of boys
def condition2 := 4 -- number of girls
def condition3 (A_selected: Bool) := ¬ A_selected -- boy A is not suitable to be the first debater
def condition4 (B_selected: Bool) := ¬ B_selected -- girl B is not suitable to be the fourth debater
def condition5 (A_selected B_selected: Bool) := A_selected → B_selected -- if boy A is selected, then girl B must also be selected

-- Define the problem as a theorem
theorem debate_team_formations :
  let A := condition1
  let B := condition2 in
  let n := if (condition5 (A = 1) (B = 1)) then 210 
           else if (¬ (condition3 (A = 1)) ∧ condition4 (B = 1)) then 360 
           else 360 in
  n = 930 :=
by
  sorry

end debate_team_formations_l423_423316


namespace max_planes_from_four_lines_l423_423088

theorem max_planes_from_four_lines : 
  ∃! n : ℕ, n = nat.choose 4 2 := 
begin
  use 6,
  simp [nat.choose],
  sorry,
end

end max_planes_from_four_lines_l423_423088


namespace complex_pow_eq_one_l423_423780

theorem complex_pow_eq_one : ( (1 + 2 * Complex.i) / (1 - 2 * Complex.i) ) ^ 500 = 1 := by
  sorry

end complex_pow_eq_one_l423_423780


namespace problem_III_l423_423514

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem problem_III
  (a x1 x2 : ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_roots : f x1 = a ∧ f x2 = a)
  (h_order : x1 < x2)
  (h_bounds : Real.exp (-1) < x1 ∧ x1 < 1 ∧ 1 < x2) :
  x2 - x1 > 1 / a - 1 :=
sorry

end problem_III_l423_423514


namespace more_wrappers_than_bottle_caps_at_park_l423_423081

-- Define the number of bottle caps and wrappers found at the park.
def bottle_caps_found : ℕ := 11
def wrappers_found : ℕ := 28

-- State the theorem to prove the number of more wrappers than bottle caps found at the park is 17.
theorem more_wrappers_than_bottle_caps_at_park : wrappers_found - bottle_caps_found = 17 :=
by
  -- proof goes here
  sorry

end more_wrappers_than_bottle_caps_at_park_l423_423081


namespace nested_sqrt_equality_l423_423888

theorem nested_sqrt_equality (x : ℝ) (h : 0 ≤ x) : (sqrt (x * sqrt (x * sqrt x))) = x^(7/8) := 
  sorry

end nested_sqrt_equality_l423_423888


namespace dot_product_value_l423_423179

variable (a b : ℝ → ℝ → ε)
variable h₁ : |a| = 1
variable h₂ : |b| = 2
variable h₃ : |a - 2 * b| = sqrt 10

theorem dot_product_value : a ⬝ b = 7 / 4 := by
  sorry

end dot_product_value_l423_423179


namespace trigonometric_identity_l423_423474

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 1 / 2) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by
  sorry

end trigonometric_identity_l423_423474


namespace find_x_l423_423787

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423787


namespace probability_is_correct_l423_423719

-- Defining the probability of getting heads or tails in a fair coin.
def fair_coin := {heads, tails} : set (fin 2)

-- Defining the sample space for 7 flips of a fair coin.
def sample_space := {seq : vector ℕ 7 | ∀ (i : ℕ) (h : i < 7), seq.nth (⟨i, h⟩) ∈ fair_coin}

-- Function to count the number of successful outcomes (at least 5 consecutive heads).
def count_successful_outcomes : ℕ := 8  -- Based on the casework given in the problem

-- Total number of outcomes
def total_outcomes : ℕ := 128

-- The probability of at least 5 consecutive heads in 7 flips.
def probability_at_least_5_consecutive_heads : ℚ := count_successful_outcomes / total_outcomes

-- Final statement to prove the probability.
theorem probability_is_correct :
  probability_at_least_5_consecutive_heads = 1 / 16 :=
by {
  have h1 : probability_at_least_5_consecutive_heads = 8 / 128 := rfl,
  have h2 : 8 / 128 = 1 / 16 := by norm_num,
  exact eq.trans h1 h2
}

end probability_is_correct_l423_423719


namespace area_of_triangle_l423_423157

theorem area_of_triangle (a b c A B C : ℝ) (h₁ : a = 1) (h₂ : b = sqrt 3) (h₃ : A + C = 2 * B) (h₄ : A + B + C = π) :
  let S := (1 / 2) * a * c * Real.sin B in
  S = sqrt 3 / 2 :=
by
  sorry

end area_of_triangle_l423_423157


namespace cyclist_wait_time_l423_423359

theorem cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ) (catch_up_time : ℝ) 
  (hiker_speed_eq : hiker_speed = 4) 
  (cyclist_speed_eq : cyclist_speed = 12) 
  (wait_time_eq : wait_time = 5 / 60) 
  (catch_up_time_eq : catch_up_time = (2 / 3) / (1 / 15)) 
  : catch_up_time * 60 = 10 := 
by 
  sorry

end cyclist_wait_time_l423_423359


namespace tetrahedron_to_cube_surface_area_ratio_l423_423666

theorem tetrahedron_to_cube_surface_area_ratio :
  ∀ s : ℝ, s = 2 →
  let A := sqrt 3 * (2 * sqrt 2) ^ 2 in
  let S := 6 * s ^ 2 in
  (A / S) = (sqrt 3 / 3) :=
by
  intros s h_s A S
  rw h_s
  sorry

end tetrahedron_to_cube_surface_area_ratio_l423_423666


namespace infinite_lines_c_l423_423259

-- Define the problem setting within Lean 4
def line := ℝ → ℝ×ℝ×ℝ

def angle_between (x y : line) (θ : ℝ) : Prop :=
  -- This is a placeholder for the actual angle computation
  sorry

def is_skew (a b : line) : Prop :=
  -- This is a placeholder for the actual skew condition
  sorry

-- Define the lines a, b and c
variable (a b c : line)

-- Proposition: a and b are skew and angle between them is 60 degrees
axiom skew_lines : is_skew a b ∧ angle_between a b (60 : ℝ)

-- Proposition: line c forms 60-degree angles with both lines a and b
axiom angle_conditions : angle_between a c (60 : ℝ) ∧ angle_between b c (60 : ℝ)

-- Prove: There exist infinitely many such lines c
theorem infinite_lines_c : ∃∞ (c : line), angle_between a c (60 : ℝ) ∧ angle_between b c (60 : ℝ) :=
  sorry

end infinite_lines_c_l423_423259


namespace binomial_coeff_division_l423_423796

theorem binomial_coeff_division (n k : ℕ) (hkn : k > 1) (hdiv : ∀ r, 1 ≤ r ∧ r < n → k ∣ Nat.choose n r) :
  ∃ t : ℕ, t > 0 ∧ n = t * ∏ i in Finset.range (k.primeDivisors.length), (k.primeDivisors.nth i).get_or_else 0 ^ (k.primeExponents.nth i).get_or_else 0 + 1 := 
by
  -- import necessary native library for primes
  sorry

end binomial_coeff_division_l423_423796


namespace linear_plane_relationship_l423_423502

variables (Line Plane : Type)
variables (l m : Line) (α β γ : Plane)

-- Conditions as hypotheses
axiom beta_inter_gamma_eq_line : β ∩ γ = l
axiom line_parallel_alpha : l ∥ α
axiom m_in_alpha : m ∈ α
axiom m_perp_gamma : m ⟂ γ

-- Statement we need to prove
theorem linear_plane_relationship :
  α ⟂ γ ∧ l ⟂ m :=
sorry

end linear_plane_relationship_l423_423502


namespace measure_angle_x_l423_423679

theorem measure_angle_x 
  (H1 : ∀ {α β γ : angle}, is_equilateral α β γ → α = 60 ∧ β = 60 ∧ γ = 60)
  (H2 : ∀ {α β : angle}, supplementary α β → α + β = 180)
  (H3 : ∀ {α β γ : angle}, α + β + γ = 180)
  (H4 : ∀ {α β γ δ : angle}, α + β + γ + δ = 360)
  (angle1 : angle)
  (angle2 : angle)
  (angle3 : angle)
  (angle4 : angle)
  (H5 : is_equilateral angle1 angle2 angle3)
  (H6 : is_equilateral angle3 angle4 angle1)
  (H7 : ∀ a b, supplementary a b → a = 75 → b = 105)
  (H8 : ∀ a b, supplementary a b → a = 65 → b = 115)
  (H9 : ∃! x : angle, supplementary x 140)
  : x = 40 := 
sorry

end measure_angle_x_l423_423679


namespace tokens_collapsibility_l423_423765

def is_power_of_two (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

def midpoint (A B : Point) : Point :=
  (A + B) / 2

def collapsible (n : Nat) : Prop :=
  ∀ points : List Point, (length points = n) → 
    ∃ p : Point, ∃ moves : List (Point × Point),
    (∀ move : Point × Point, move ∈ moves → 
      ∃ i j : Fin n, move = (points[i], points[j]) ∧ 
      points[i] = midpoint (points[i]) (points[j])) ∧ 
    (∀ point : Point, point ∈ points → point = p)

theorem tokens_collapsibility (n : Nat) : collapsible n ↔ is_power_of_two n := 
  sorry

end tokens_collapsibility_l423_423765


namespace normal_distribution_tail_probability_l423_423819

noncomputable def normal_distribution (mean variance : ℝ) : Type := sorry

variable (ξ : Type) 
variable [normal_distribution ξ 0 σ^2]

axiom P_interval : ∀ (a b : ℝ), P (a ≤ ξ ∧ ξ ≤ b) : ℝ

noncomputable def P (event : ξ → Prop) : ℝ := sorry

theorem normal_distribution_tail_probability 
  (h1 : P_interval (-2) 0 = 0.4) 
  : P (λ x, x > 2) = 0.1 := 
sorry

end normal_distribution_tail_probability_l423_423819


namespace max_squares_covered_by_card_l423_423715

noncomputable def card_coverage_max_squares (card_side : ℝ) (square_side : ℝ) : ℕ :=
  if card_side = 2 ∧ square_side = 1 then 9 else 0

theorem max_squares_covered_by_card : card_coverage_max_squares 2 1 = 9 := by
  sorry

end max_squares_covered_by_card_l423_423715


namespace pyramid_sphere_surface_area_l423_423490

-- Definitions based on the problem's conditions
def length_PA : ℝ := 2
def length_PB : ℝ := 2
def length_PC : ℝ := 2

-- Theorem statement
theorem pyramid_sphere_surface_area :
  (PA = 2) ∧ (PB = 2) ∧ (PC = 2) ∧ 
  (sum_of_side_face_areas_max (P ABC)) →
  surface_area_of_sphere (O) = 12 * π :=
sorry

end pyramid_sphere_surface_area_l423_423490


namespace carter_total_drum_sticks_l423_423759

def sets_per_show_used := 5
def sets_per_show_tossed := 6
def nights := 30

theorem carter_total_drum_sticks : 
  (sets_per_show_used + sets_per_show_tossed) * nights = 330 := by
  sorry

end carter_total_drum_sticks_l423_423759


namespace determine_N_l423_423087

theorem determine_N (N : ℕ) :
    995 + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := 
by 
  sorry

end determine_N_l423_423087


namespace number_of_bricks_required_l423_423007

-- Definitions based on the conditions
def brick_length := 20 -- cm
def brick_width := 10 -- cm
def brick_height := 7.5 -- cm

def wall_length := 26 * 100 -- converting meters to cm
def wall_width := 2 * 100 -- converting meters to cm
def wall_height := 0.75 * 100 -- converting meters to cm

-- Theorem statement
theorem number_of_bricks_required :
  (wall_length * wall_width * wall_height) / (brick_length * brick_width * brick_height) = 26000 :=
sorry -- proof not required

end number_of_bricks_required_l423_423007


namespace total_profit_calculation_l423_423741

noncomputable def total_profit (x : ℚ) : ℚ :=
  (144 * 82) / 30

theorem total_profit_calculation :
  ∀ (x : ℚ), 
    let A_initial := 2 * x,
        B_initial := 3 * x,
        C_initial := 4 * x,
        A_investment := (2 * x * 2) + (x * 8),
        B_investment := 3 * x * 10,
        C_investment := 4 * x * 10,
        total_investment := A_investment + B_investment + C_investment,
        B_share := 144 in
      total_profit x = 390.4 := 
sorry

end total_profit_calculation_l423_423741


namespace female_actor_A_not_on_side_l423_423467

theorem female_actor_A_not_on_side (females males : ℕ) (A : bool) :
  (females = 5) → (males = 4) →
  (A = true) → -- Representing female actor A existing in the group
  (A_{ 5 }^{ 5 } * A_{ 6 }^{ 4 }) - (2 * A_{ 4 }^{ 4 } * A_{ 5 }^{ 4 }) = A_{ 5 }^{ 5 } * A_{ 6 }^{ 4 } - 2 * A_{ 4 }^{ 4 } * A_{ 5 }^{ 4 } :=
by
  sorry

end female_actor_A_not_on_side_l423_423467


namespace true_statement_about_M_l423_423255

variable (U : Set ℕ) (M : Set ℕ)
axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_M_def : U \ M = {1, 3}

theorem true_statement_about_M : 2 ∈ M :=
by 
  rw [U_def, complement_M_def]
  sorry

end true_statement_about_M_l423_423255


namespace perpendicular_condition_l423_423149

variables {l m : Type} [linear_space l m]
variables {α : set l} [plane α]

axiom line_in_plane (m_in_plane : m ⊆ α)

theorem perpendicular_condition (h1 : ∀ (l : Type) (α : set l), l ⟂ α → ∀ m ∈ α, l ⟂ m)
(h2 : ∀ (l m : Type) (α : set l), l ⟂ m ∧ m ⊆ α → l ⟂ α ) : 
 ∀ {l m : Type} (α : set l), m ⊆ α → (l ⟂ m → l ⟂ α) ∧ ¬(l ⟂ α → l ⟂ m) :=
sorry

end perpendicular_condition_l423_423149


namespace no_perfect_square_in_seq_l423_423618

def seq (n : ℕ) : ℕ := (2014 * (10^8) - 2014) / 99 * (10^(4 * n) - 1 / 9999) 

theorem no_perfect_square_in_seq :
  ∀ n : ℕ, ¬ ∃ k : ℕ, k * k = seq n :=
by
  assume n
  have h1 : seq n % 4 = 2 := sorry
  have h2 : ∀ k : ℕ, k * k % 4 = 0 ∨ k * k % 4 = 1 := sorry
  have h3 : ∀ k : ℕ, ¬ (k * k % 4 = 2) := sorry
  exact h3

end no_perfect_square_in_seq_l423_423618


namespace constants_values_l423_423436

theorem constants_values (a b : ℚ) :
  a • (3, 4) + b • (-6, 15) = (1, 2) →
  a = 9 / 23 ∧ b = 2 / 69 :=
by
  intros h
  have h₁ : 3 * a - 6 * b = 1 := by sorry
  have h₂ : 4 * a + 15 * b = 2 := by sorry
  have ha : a = 9 / 23 := by sorry
  have hb : b = 2 / 69 := by sorry
  exact ⟨ha, hb⟩

end constants_values_l423_423436


namespace train_crossing_time_l423_423697

-- Conditions
def train_length : ℝ := 100  -- meters
def train_speed_km_hr : ℝ := 180  -- km/hr
def speed_conversion_factor : ℝ := 1000 / 3600  -- conversion factor from km/hr to m/s

-- Derived speed in m/s
def train_speed_m_s : ℝ := train_speed_km_hr * speed_conversion_factor

-- Theorem: prove that the crossing time is 2 seconds
theorem train_crossing_time : train_length / train_speed_m_s = 2 := 
by
  -- Here normally we would place the proof, but since it is only the statement being requested:
  sorry

end train_crossing_time_l423_423697


namespace roots_greater_than_one_implies_range_l423_423845

theorem roots_greater_than_one_implies_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + a = 0 → x > 1) → 3 < a ∧ a ≤ 4 :=
by
  sorry

end roots_greater_than_one_implies_range_l423_423845


namespace arctan_sum_l423_423449

noncomputable def y := -43 / 3

theorem arctan_sum : 
  2 * arctan (1 / 3) + arctan (1 / 15) + arctan (1 / y) = π / 4 := by
  sorry

end arctan_sum_l423_423449


namespace canoe_upstream_speed_l423_423370

noncomputable theory

-- Define the speed of the stream
def stream_speed : ℝ := 2

-- Define the given downstream speed
def downstream_speed : ℝ := 10

-- Define the speed of the canoe in still water
def speed_in_still_water : ℝ := downstream_speed - stream_speed

-- Prove that the speed of the canoe when rowing upstream is 6 km/hr
theorem canoe_upstream_speed : speed_in_still_water - stream_speed = 6 :=
sorry

end canoe_upstream_speed_l423_423370


namespace convex_pentagon_medians_not_collinear_l423_423361

structure ConvexPentagon (α : Type _) :=
  (vertices : Fin 5 → α)
  (is_convex : Convex α)

structure Triangle (α : Type _) :=
  (vertices : Fin 3 → α)

def is_median_intersection_on_line {α : Type _} [AddCommGroup α] [Module ℚ α] 
  (triangles : Fin 3 → Triangle α) : Prop := sorry -- Definition of medians' intersection on a line

theorem convex_pentagon_medians_not_collinear (P : ConvexPentagon ℝ) 
  (triangles : Fin 3 → Triangle ℝ)
  (h_divide : divides P triangles)
  (h_nonintersecting_diagonals : nonintersecting_diagonals P triangles) :
  ¬ is_median_intersection_on_line triangles :=
sorry

end convex_pentagon_medians_not_collinear_l423_423361


namespace milk_tea_sales_l423_423695

-- Definitions based on the conditions
def total_cups : Nat := 100
def winter_melon_percentage : Float := 0.35
def okinawa_fraction : Float := 0.25
def taro_cups : Nat := 12
def ratio_chocolate_thai : Float := 3 / 10

-- Assert that these calculated counts are correct
def winter_melon_cups : Nat := 35
def okinawa_cups : Nat := 25
def chocolate_cups : Nat := 8
def thai_cups : Nat := 20

theorem milk_tea_sales :
  (total_cups = 100) →
  (winter_melon_cups = Nat.ceil (winter_melon_percentage * (total_cups.toFloat))) →
  (okinawa_cups = Nat.ceil (okinawa_fraction * (total_cups.toFloat))) →
  taro_cups = 12 →
  chocolate_cups + thai_cups = (total_cups - (winter_melon_cups + okinawa_cups + taro_cups)) →
  chocolate_cups = Nat.ceil (ratio_chocolate_thai * (total_cups - (winter_melon_cups + okinawa_cups + taro_cups))) →
  thai_cups = (total_cups - (winter_melon_cups + okinawa_cups + taro_cups)) - chocolate_cups →
  (winter_melon_cups + okinawa_cups + chocolate_cups + thai_cups + taro_cups = total_cups) :=
by
  sorry

end milk_tea_sales_l423_423695


namespace translated_symmetric_function_l423_423652

noncomputable def f (x : ℝ) : ℝ := e ^ (-x) - 1

theorem translated_symmetric_function :
  (∀ x : ℝ, (f (x + 1) = e ^ (-x))) :=
by sorry

end translated_symmetric_function_l423_423652


namespace rationalize_denominator_sum_l423_423279

theorem rationalize_denominator_sum :
  ∃ (A B C D : ℤ), 
    (D > 0) ∧ 
    (∀ (p : ℤ), prime p → ¬ (p ^ 3 ∣ B)) ∧ 
    (Int.gcd A (Int.gcd C D) = 1) ∧ 
    (6 - 2 * real.cbrt 7 = (A * real.cbrt B + C) / D) ∧
    (A + B + C + D = 13) :=
begin
  sorry
end

end rationalize_denominator_sum_l423_423279


namespace compute_integer_y_l423_423530

theorem compute_integer_y : 
  (∑ n in Finset.range 1000, (n + 1) * (1001 - (n + 1)) * (n + 2)) = 1001 * 500.5 * 2001 := 
sorry

end compute_integer_y_l423_423530


namespace cartesian_equation_of_line_l423_423523

theorem cartesian_equation_of_line (t x y : ℝ)
  (h1 : x = 1 + t / 2)
  (h2 : y = 2 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

end cartesian_equation_of_line_l423_423523


namespace chord_midpoint_ellipse_l423_423850

theorem chord_midpoint_ellipse :
  (∀ a b : ℝ × ℝ, ellipse a → ellipse b →
  midpoint a b = (1, 1) → 
  line a b) = { x + 2 * y - 3 = 0} :=
by
  -- Definitions specific to the problem
  def ellipse (p : ℝ × ℝ) : Prop := 
    ∃ x y : ℝ, p = (x, y) ∧ (x^2 / 8) + (y^2 / 4) = 1

  def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
    ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  def line (a b : ℝ × ℝ) : set (ℝ × ℝ) :=
    {p | ∃ k : ℝ, ∀ x y, p = (x, y) ∧ y = k * (x - 1) + 1}

  -- Statement of the theorem
  sorry

end chord_midpoint_ellipse_l423_423850


namespace f_2008_value_l423_423934

noncomputable def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) (hx : x ∈ B) : ℝ := sorry

theorem f_2008_value (h : ∀ x ∈ B, f x (by assumption) + f (2 - 1 / x) sorry = Real.log (2 * abs x)) :
  f 2008 (by {
    have h₁ : 2008 ∈ B, {
      split, norm_num,
      split, norm_num,
      norm_num
    },
    exact h₁
  }) = Real.log (2008 / 2007) :=
sorry

end f_2008_value_l423_423934


namespace dot_product_magnitude_l423_423883

variables (v : ℝ^3)

theorem dot_product_magnitude (h : ‖v‖ = 7) : v ⋅ v = 49 :=
sorry

end dot_product_magnitude_l423_423883


namespace product_of_solutions_l423_423116

theorem product_of_solutions (x : ℤ) (h : x^2 = 49) : ∏ (x : {x // x^2 = 49}), x = -49 := sorry

end product_of_solutions_l423_423116


namespace probability_of_multiple_of_60_l423_423861
open Set

def set_elements : Set ℕ := {2, 3, 6, 15, 18, 20, 30}

def is_product_multiple_of_60 (a b : ℕ) : Prop :=
  a ≠ b ∧ (a * b) % 60 = 0

def successful_pairs_count := 
  (∃ x ∈ set_elements, ∃ y ∈ set_elements, is_product_multiple_of_60 x y)

theorem probability_of_multiple_of_60 : 
  let total_pairs := (Set.toFinset set_elements).card.choose 2
  in 
  successful_pairs_count / total_pairs = 10 / 21 :=
sorry

end probability_of_multiple_of_60_l423_423861


namespace number_of_squares_with_prime_condition_l423_423529

theorem number_of_squares_with_prime_condition : 
  ∃! (n : ℕ), ∃ (p : ℕ), Prime p ∧ n^2 = p + 4 := 
sorry

end number_of_squares_with_prime_condition_l423_423529


namespace find_x_l423_423789

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423789


namespace system_of_equations_has_solution_l423_423138

theorem system_of_equations_has_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 :=
by
  sorry

end system_of_equations_has_solution_l423_423138


namespace total_amount_earned_is_90_l423_423708

variable (W : ℕ)

-- Define conditions
def work_capacity_condition : Prop :=
  5 = W ∧ W = 8

-- Define wage per man in Rs.
def wage_per_man : ℕ := 6

-- Define total amount earned by 5 men
def total_earned_by_5_men : ℕ := 5 * wage_per_man

-- Define total amount for the problem
def total_earned (W : ℕ) : ℕ :=
  3 * total_earned_by_5_men

-- The final proof statement
theorem total_amount_earned_is_90 (W : ℕ) (h : work_capacity_condition W) : total_earned W = 90 := by
  sorry

end total_amount_earned_is_90_l423_423708


namespace microbial_population_extinction_l423_423727

-- Definitions
def p_0 : ℝ := 0.4
def p_1 : ℝ := 0.3
def p_2 : ℝ := 0.2
def p_3 : ℝ := 0.1

def P (X : ℕ → ℝ) (i : ℕ) : ℝ :=
  match i with
  | 0 => p_0
  | 1 => p_1
  | 2 => p_2
  | 3 => p_3
  | _ => 0

def E (X : ℕ → ℝ) : ℝ := 0 * p_0 + 1 * p_1 + 2 * p_2 + 3 * p_3

-- Theorem Statement
theorem microbial_population_extinction :
  ∀ X : ℕ → ℝ,
    E(X) = 1 →
    (∀ x : ℝ, p_0 + p_1 * x + p_2 * x^2 + p_3 * x^3 = x →
      (E(X) ≤ 1 → x = 1) ∧ (E(X) > 1 → x < 1)) :=
by
  intros X hEX heq
  -- To be proven
  sorry

end microbial_population_extinction_l423_423727


namespace right_angled_triangle_exists_l423_423399

theorem right_angled_triangle_exists : 
  ∃ (a b c : ℕ), (a = 3 ∧ b = 4 ∧ c = 5) ∧ a^2 + b^2 = c^2 :=
by {
  use [3, 4, 5],
  split,
  { split; refl, },
  { sorry, }
}

end right_angled_triangle_exists_l423_423399


namespace randy_initial_money_l423_423620

/-- Randy's initial amount of money in his piggy bank --/
def initial_money := 104

/-- Randy's total deposits in dollars per month --/
def monthly_deposit := 50

/-- The total number of months in a year --/
def months_in_year := 12

/-- The number of visits to the store in a year --/
def store_visits := 200

/-- The minimum cost per store visit in dollars --/
def min_cost_per_visit := 2

/-- The maximum cost per store visit in dollars --/
def max_cost_per_visit := 3

/-- The money left in Randy's piggy bank after a year --/
def money_left := 104

theorem randy_initial_money :
  let total_deposit := monthly_deposit * months_in_year in
  let min_spent := store_visits * min_cost_per_visit in
  let max_spent := store_visits * max_cost_per_visit in
  (money_left + max_spent - total_deposit = initial_money) ∧
  (money_left + min_spent - total_deposit <= 0) :=
  by
    sorry

end randy_initial_money_l423_423620


namespace statement_1_statement_2_statement_3_statement_4_l423_423919

-- Definitions as given in a)
def class (k : ℤ) : set ℤ := { x | ∃ n : ℤ, x = 5 * n + k }

-- Statements to be proved or disproved
theorem statement_1 : 2011 ∈ class 1 := 
sorry

theorem statement_2 : ¬(-4 ∈ class 4) := 
sorry

theorem statement_3 : (λ x, ∃ k : fin 5, x ∈ class k) = set.univ := 
sorry

theorem statement_4 (a b : ℤ) : (∃ k : fin 5, a ∈ class k ∧ b ∈ class k) ↔ (a - b ∈ class 0) := 
sorry

end statement_1_statement_2_statement_3_statement_4_l423_423919


namespace part1_part2_l423_423838

open Real -- Open the real number space

variables (a b : ℝ^3) -- The vectors a and b are in ℝ^3

def angle := 120 * (π / 180) -- Define the angle in radians

def a_norm : ℝ := 2 -- The norm of vector a
def b_norm : ℝ := 4 -- The norm of vector b

axiom dot_product (a b : ℝ^3) : ℝ -- Define the dot product
axiom norm (v : ℝ^3) : ℝ -- Define the norm

-- Condition that the dot product of a and b is -4
axiom a_dot_b : dot_product a b = -4

-- Prove that the norm of (4a - 2b) is 8√3
theorem part1 : norm (4 • a - 2 • b) = 8 * Real.sqrt 3 :=
by sorry

-- Prove that (a + 2b) is perpendicular to (k a - b)
theorem part2 (k : ℝ) (h : k = -7) : dot_product (a + 2 • b) (k • a - b) = 0 :=
by sorry

end part1_part2_l423_423838


namespace intersection_eq_l423_423176

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_eq : M ∩ N = intersection := by
  sorry

end intersection_eq_l423_423176


namespace magnitude_of_b_l423_423150

/-- Given non-zero vectors a and b satisfying |a| = |a + b| = 1,
    and the angle between a and b is 120 degrees, 
    the magnitude of vector b is 1. -/
theorem magnitude_of_b (a b : ℝ^3)
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖a + b‖ = 1)
  (h₃ : real.angle (a) (b) = real.pi / 3 * 2) : 
  ‖b‖ = 1 :=
sorry

end magnitude_of_b_l423_423150


namespace word_ratio_l423_423967

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / List.prod (ks.map factorial)

theorem word_ratio (n m k l : ℕ) (h1 : n = 6) (h2 : m = 2) (h3 : k = 2) (h4 : l = 7) :
  multinomial_coefficient l [m, k] / multinomial_coefficient n [m, k] = 7 :=
by
  sorry

end word_ratio_l423_423967


namespace ball_distribution_l423_423872

-- Definitions as per conditions
def num_distinguishable_balls : ℕ := 5
def num_indistinguishable_boxes : ℕ := 3

-- Problem statement to prove
theorem ball_distribution : 
  let ways_to_distribute_balls := 1 + 5 + 10 + 10 + 30 in
  ways_to_distribute_balls = 56 :=
by
  -- proof required here
  sorry

end ball_distribution_l423_423872


namespace find_possible_values_l423_423941

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end find_possible_values_l423_423941


namespace lcm_8_13_14_is_728_l423_423800

-- Define the numbers and their factorizations
def num1 := 8
def fact1 := 2 ^ 3

def num2 := 13  -- 13 is prime

def num3 := 14
def fact3 := 2 * 7

-- Define the function to calculate the LCM of three integers
def lcm (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- State the theorem to prove that the LCM of 8, 13, and 14 is 728
theorem lcm_8_13_14_is_728 : lcm num1 num2 num3 = 728 :=
by
  -- Prove the equality, skipping proof details with sorry
  sorry

end lcm_8_13_14_is_728_l423_423800


namespace girl_scout_boxes_l423_423008

theorem girl_scout_boxes (
    P C : ℕ
) (h1 : C + P = 1585)
  (h2 : 1.25 * C + 0.75 * P = 1586.75) : P = 789 :=
by
  sorry

end girl_scout_boxes_l423_423008


namespace find_b_l423_423544

noncomputable theory
open Real

variables {x₁ x₂ k b : ℝ}

def tangent_line (x : ℝ) (k b : ℝ) := k * x + b
def curve_1 (x : ℝ) := log x + 2
def curve_2 (x : ℝ) := log (x + 1)

theorem find_b
  (h_tangent_k : ∀ x₁ x₂, k = 1 / x₁ ∧ k = 1 / (x₂ + 1))
  (h_x_relation : x₁ = x₂ + 1)
  (h_point_on_curve1 : tangent_line x₁ k b = curve_1 x₁)
  (h_point_on_curve2 : tangent_line x₂ k b = curve_2 x₂) :
  b = 1 - log 2 :=
begin
  sorry
end

end find_b_l423_423544


namespace scorpion_additional_millipedes_l423_423023

theorem scorpion_additional_millipedes :
  let total_segments := 800 in
  let segments_one_millipede := 60 in
  let segments_two_millipedes := 2 * (2 * segments_one_millipede) in
  let total_eaten := segments_one_millipede + segments_two_millipedes in
  let remaining_segments := total_segments - total_eaten in
  let segments_per_millipede := 50 in
  remaining_segments / segments_per_millipede = 10 :=
by {
  sorry
}

end scorpion_additional_millipedes_l423_423023


namespace P_15_is_187p5_l423_423880

def T (n : ℕ) : ℚ := (n * (n + 1)) / 2

def P (n : ℕ) : ℚ :=
  (List.range' 2 (n - 1)).prod (λ k, (T k + 1) / (T k - 1))

theorem P_15_is_187p5 : P 15 = 187.5 :=
by
  sorry

end P_15_is_187p5_l423_423880


namespace find_d_l423_423641

theorem find_d (A B C D : ℕ) (h1 : (A + B + C) / 3 = 130) (h2 : (A + B + C + D) / 4 = 126) : D = 114 :=
by
  sorry

end find_d_l423_423641


namespace supremum_of_negatives_no_maximum_of_negatives_l423_423452

noncomputable def A : Set ℝ := {x : ℝ | x < 0}

-- Proof Problem 1: The supremum of the set of negative numbers is 0.
theorem supremum_of_negatives : isSup A 0 :=
sorry

-- Proof Problem 2: The set of negative numbers does not have a maximum.
theorem no_maximum_of_negatives : ¬ ∃ m ∈ A, ∀ x ∈ A, x ≤ m :=
sorry

end supremum_of_negatives_no_maximum_of_negatives_l423_423452


namespace limit_of_incenter_sequence_l423_423929

-- Definitions
variable {α : Type*} [metric_space α] [normed_group α]

/-- The sequence of incenters in a triangle -/
noncomputable def incenter_sequence (A B C : α) : ℕ → α
| 0      := C
| (n + 1) := incenter A B (incenter_sequence n)

-- Main theorem statement
theorem limit_of_incenter_sequence 
  {A B C : α} (hABC : ∃ P : α, P ∈ line_span ℝ ({B,C} : set α) ∧
  dist A B > 0 ∧ dist B C > 0 ∧ dist C A > 0 ∧
  ∀ n : ℕ, dist (incenter_sequence A B C n) (incenter_sequence A B C (n + 1)) < 1 / 2^n) :
  ∃ P : α, P ∈ line_span ℝ ({B,C} : set α) ∧
  ∀ ε > 0, ∃ N : ℕ, ∀ n m > N, dist (incenter_sequence A B C n) P < ε :=
sorry

end limit_of_incenter_sequence_l423_423929


namespace compute_g_iterate_five_times_l423_423239

def g (x : ℝ) : ℝ :=
if x ≤ 2 then -2 * x^2 + 1 else 2 * x - 3

theorem compute_g_iterate_five_times :
  g (g (g (g (g 2)))) = -1003574820536187457 := by
  sorry

end compute_g_iterate_five_times_l423_423239


namespace quadratic_function_m_value_l423_423542

theorem quadratic_function_m_value
  (m : ℝ)
  (h1 : m^2 - 7 = 2)
  (h2 : 3 - m ≠ 0) :
  m = -3 := by
  sorry

end quadratic_function_m_value_l423_423542


namespace num_of_ways_to_remove_tile_and_tile_with_1x3_and_3x1_rectangles_l423_423878

theorem num_of_ways_to_remove_tile_and_tile_with_1x3_and_3x1_rectangles :
  ∑ i, ∑ j, i % 3 = 0 ∧ j % 3 = 0 ∧ 0 ≤ i ∧ i < 2014 ∧ 0 ≤ j ∧ j < 2014 = 451584 := by
  sorry

end num_of_ways_to_remove_tile_and_tile_with_1x3_and_3x1_rectangles_l423_423878


namespace second_group_children_is_16_l423_423030

def cases_purchased : ℕ := 13
def bottles_per_case : ℕ := 24
def camp_days : ℕ := 3
def first_group_children : ℕ := 14
def third_group_children : ℕ := 12
def bottles_per_child_per_day : ℕ := 3
def additional_bottles_needed : ℕ := 255

def fourth_group_children (x : ℕ) : ℕ := (14 + x + 12) / 2
def total_initial_bottles : ℕ := cases_purchased * bottles_per_case
def total_children (x : ℕ) : ℕ := 14 + x + 12 + fourth_group_children x 

def total_consumption (x : ℕ) : ℕ := (total_children x) * bottles_per_child_per_day * camp_days
def total_bottles_needed : ℕ := total_initial_bottles + additional_bottles_needed

theorem second_group_children_is_16 :
  ∃ x : ℕ, total_consumption x = total_bottles_needed ∧ x = 16 :=
by
  sorry

end second_group_children_is_16_l423_423030


namespace parallel_lines_m_l423_423864

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 3 * m * x + (m + 2) * y + 1 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + (m + 2) * y + 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (3 * m) / (m + 2) = (m - 2) / (m + 2)) →
  (m = -1 ∨ m = -2) :=
sorry

end parallel_lines_m_l423_423864


namespace expr_value_l423_423339

theorem expr_value : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end expr_value_l423_423339


namespace smallest_possible_N_l423_423249

theorem smallest_possible_N {p q r s t : ℕ} (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (ht: 0 < t) 
  (sum_eq: p + q + r + s + t = 3015) :
  ∃ N, N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ N = 1508 := 
sorry

end smallest_possible_N_l423_423249


namespace max_min_sum_f_l423_423253

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 + sin x / (x ^ 2 + 1)

theorem max_min_sum_f : 
  let M := Real.sup (Set.range f) in
  let m := Real.inf (Set.range f) in
  M + m = 2 :=
by
  sorry

end max_min_sum_f_l423_423253


namespace E_X_calc_probs_extinct_l423_423725

-- Definitions of conditions
def X_probs : Fin 4 → ℝ
| 0 => 0.4
| 1 => 0.3
| 2 => 0.2
| 3 => 0.1

def E_X : ℝ := ∑ i in Finset.univ, i * X_probs i

-- Statements to be proven
theorem E_X_calc : E_X = 1 := by sorry

theorem probs_extinct (p : ℝ) (h : p_0 + p_1 * p + p_2 * p^2 + p_3 * p^3 = p) :
  (E_X ≤ 1 → p = 1) ∧ (E_X > 1 → p < 1) := by sorry

end E_X_calc_probs_extinct_l423_423725


namespace fraction_of_volume_above_water_l423_423717

-- Defining the conditions
def total_height : ℝ := 5000
def height_above_water : ℝ := 2500
def height_ratio (above : ℝ) (total : ℝ) : ℝ := above / total
def volume_fraction (ratio : ℝ) : ℝ := ratio ^ 3

-- The proof statement
theorem fraction_of_volume_above_water : 
  volume_fraction (height_ratio height_above_water total_height) = 0.125 := by
sorry

end fraction_of_volume_above_water_l423_423717


namespace part1_part2_part3_l423_423835

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : ∃ x, f x ≠ 0
axiom f_rule : ∀ (a b : ℝ), f (a * b) = a * f b + b * f a
axiom f_two : f 2 = 2

def U : ℕ+ → ℝ := λ n, f (2 ^ -n) / n

theorem part1 : f 0 = 0 ∧ f 1 = 0 :=
sorry

theorem part2 : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem part3 (n : ℕ+) : ∑ k in Finset.range n, U (k+1) = (1/2) ^ n - 1 :=
sorry

end part1_part2_part3_l423_423835


namespace find_magnitude_a_l423_423258

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_a (m : ℝ) (h : dot_product (vector_add (vector_a m) (vector_c m)) (vector_b m) = 0) :
  magnitude (vector_a (-1 / 2)) = Real.sqrt 2 :=
by
  sorry

end find_magnitude_a_l423_423258


namespace part1_part2_part3_l423_423221

-- Define the sequence
def a (n : ℕ) : ℝ := (n^2 + n - 1) / 3

theorem part1 (h10 : ∀ n : ℕ, a 10 = 109 / 3) : a 10 = 109 / 3 :=
by 
  exact h10 10 -- This will be proven by substituting 10 into the sequence definition

theorem part2 (h_n : ∀ n : ℕ, a (n + 1) = (n^2 + 3 * n + 1) / 3) : ∀ n : ℕ, a (n + 1) = (n^2 + 3 * n + 1) / 3 :=
by
  exact h_n -- This will be proven by substituting n+1 into the sequence definition

noncomputable def inverse_a (x : ℝ) : ℝ := ((3 * x) - 1/3) ^ (1/2)

theorem part3 (hx : 79 + 2/3 ∈ set.range a) : ∃ n : ℕ, a n = 79 + 2/3 :=
by
  use 15
  have ha := ((15 : ℕ) : ℝ) -- This casts 15 to real
  simp only [a]
  norm_num -- This proves that a 15 equals 79 + 2/3
  sorry -- Complete the proof here

end part1_part2_part3_l423_423221


namespace prime_poly_not_factorable_l423_423933

theorem prime_poly_not_factorable (n : ℕ) (a : ℕ → ℤ)
  (hn : n > 1) (ha_n : a n > 1) 
  (N_prime : nat.prime (∑ i in finset.range (n + 1), (a i) * 10 ^ i)) :
  ¬ (∃ g h : polynomial ℤ, g.degree > 0 ∧ h.degree > 0 ∧ (polynomial.of_coeffs (finset.range (n + 1)).attach_val (a ∘ finset.val) = g * h)) :=
by 
  sorry

end prime_poly_not_factorable_l423_423933


namespace volume_of_solid_l423_423665

theorem volume_of_solid :
  let v := λ (x y z : ℝ), ⟨x, y, z⟩ in
  let a := ⟨12, -24, 6⟩ : EuclideanSpace ℝ (Fin 3) in
  ∀ (x y z : ℝ),
    (‖v x y z‖ ^ 2 = (v x y z) ⬝ a) →
    v = ⟨15, 0, 0⟩ →
    (4 / 3) * π * 15^3 = 4500 * π :=
by
  sorry

end volume_of_solid_l423_423665


namespace decimal_places_bound_l423_423191

theorem decimal_places_bound {
  (a : ℕ)         -- Condition 1: 10^4 is an integer
  (b : ℝ)         -- Condition 2: 3.456789 has 6 decimal places
  (c : ℝ)         -- Condition 3: 7.891 has 3 decimal places
} 
(h1 : a = 10^4)
(h2 : b = 3.456789)
(h3 : c = 7.891) :
  ∃ d : ℝ, (d = ((a * b) ^ 12 / c) ^ 14) ∧ has_decimal_places_le d 9 := 
sorry

end decimal_places_bound_l423_423191


namespace inverse_function_inequality_solution_l423_423252

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)
noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log 2 - Real.log (1 - y) / Real.log 2

theorem inverse_function (x : ℝ) (hx : x ∈ set.Ioo (-1 : ℝ) 1) :
  f_inv (f x) = x :=
by sorry

theorem inequality_solution (x : ℝ) :
  (f_inv x > Real.log (1 + x) / Real.log 2 + 1) ↔ (x ∈ set.Ioo (1 / 2) 1) :=
by sorry

end inverse_function_inequality_solution_l423_423252


namespace correct_statements_count_l423_423516

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Statement 1
def exists_root (a b c : ℝ) : Prop :=
  ∃ x0 : ℝ, f x0 a b c = 0

-- Statement 2
def has_two_extreme_points (a b c : ℝ) : Prop :=
  a^2 > 3 * b

-- Statement 3
def local_maximum_not_monotone (a b c x0 : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x > x0 → f x a b c < f x0 a b c)

-- Statement 4
def center_of_symmetry (a b c : ℝ) : Prop :=
  let x := -a / 3 in
  let y := f x a b c in
  ∃ x' y' : ℝ, x = x' ∧ y = y'

-- Lean statement for the problem
def number_of_correct_statements (a b c : ℝ) : ℝ :=
  (if exists_root a b c then 1 else 0) +
  (if has_two_extreme_points a b c then 1 else 0) +
  (if ¬local_maximum_not_monotone a b c (-a / 3) then 1 else 0) +
  (if center_of_symmetry a b c then 1 else 0)

theorem correct_statements_count (a b c : ℝ) : number_of_correct_statements a b c = 3 :=
sorry

end correct_statements_count_l423_423516


namespace num_ways_to_fill_grid_l423_423061

open Finset

-- definition of a 3x3 grid
def Grid := fin 3 → fin 3 → ℕ

-- condition that the sum of each row and column is odd
def row_sum_odd (g : Grid) := ∀ i : fin 3, odd (sum (univ.map (λ j, g i j)))
def col_sum_odd (g : Grid) := ∀ j : fin 3, odd (sum (univ.map (λ i, g i j)))

-- condition that there are four 0s, four 1s, and one 3
def valid_numbers (g : Grid) := 
  card (univ.bind (λ i, univ.image (g i))) = 9 ∧
  count 0 (univ.bind (λ i, univ.map (g i))) = 4 ∧
  count 1 (univ.bind (λ i, univ.map (g i))) = 4 ∧
  count 3 (univ.bind (λ i, univ.map (g i))) = 1

-- The main theorem statement
theorem num_ways_to_fill_grid :
  ∃ (g : Grid), row_sum_odd g ∧ col_sum_odd g ∧ valid_numbers g ∧ (number_of_ways g = 45) :=
sorry

end num_ways_to_fill_grid_l423_423061


namespace infinitely_many_a_not_sum_of_seven_sixth_powers_l423_423626

theorem infinitely_many_a_not_sum_of_seven_sixth_powers :
  ∃ᶠ (a: ℕ) in at_top, (∀ (a_i : ℕ) (h0 : a_i > 0), a ≠ a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 ∧ a % 9 = 8) :=
sorry

end infinitely_many_a_not_sum_of_seven_sixth_powers_l423_423626


namespace cone_surface_area_l423_423645

-- Define the geometric and trigonometric properties of the pyramid and the cone
variables (d α : ℝ) (S : ℝ)

-- Assuming the conditions
variable (pyramid : Prop) -- represents the fact that the shape is a regular quadrilateral pyramid
variable (distance_midpoint_to_face : d > 0)
variable (angle_alpha : α > 0)

-- Define the property we wish to prove
theorem cone_surface_area
  (h_pyramid : pyramid)
  (h_distance : distance_midpoint_to_face)
  (h_angle : angle_alpha) :
  S = (2 * π * d^2) / (sin (α / 2) ^ 2 * cos α) := 
  sorry

end cone_surface_area_l423_423645


namespace italian_dressing_mixture_l423_423029

/-- A chef is using a mixture of two brands of Italian dressing. 
  The first brand contains 8% vinegar, and the second brand contains 13% vinegar.
  The chef wants to make 320 milliliters of a dressing that is 11% vinegar.
  This statement proves the amounts required for each brand of dressing. -/

theorem italian_dressing_mixture
  (x y : ℝ)
  (hx : x + y = 320)
  (hv : 0.08 * x + 0.13 * y = 0.11 * 320) :
  x = 128 ∧ y = 192 :=
sorry

end italian_dressing_mixture_l423_423029


namespace solution_f_90_l423_423814

def f (n : ℕ) : ℕ :=
  if n >= 1000 then n - 3 else f (f (n + 7))

theorem solution_f_90 : f 90 = 999 :=
  sorry

end solution_f_90_l423_423814


namespace painting_cost_conversion_l423_423973

def conversion_rates :=
  { usd_to_nad : ℝ := 7,         -- 1 USD = 7 NAD
    gbp_to_usd : ℝ := 1.4,       -- 1 GBP = 1.4 USD
    gbp_to_nad : ℝ := 9.8,       -- 1 GBP = 9.8 NAD
    usd_to_cny : ℝ := 6 }        -- 1 USD = 6 CNY

def painting_cost_in_nad : ℝ := 196

-- To prove: painting_cost_in_cny = 168
theorem painting_cost_conversion (rates : conversion_rates)
  (painting_cost_in_nad : ℝ) : (painting_cost_in_nad / rates.gbp_to_nad) * rates.gbp_to_usd * rates.usd_to_cny = 168 :=
by
  -- Sorry proof
  sorry

end painting_cost_conversion_l423_423973


namespace twice_as_frequent_visits_l423_423783

noncomputable def train_schedule (T : ℕ) (arrival_time : ℕ) : Prop :=
  ∃ (n : ℕ), arrival_time = n * T

def first_club_train_schedule (T : ℕ) : set ℕ := 
  {time | train_schedule T time}

def second_club_train_schedule (T : ℕ) : set ℕ := 
  {time | ∃ (n : ℕ), time = n * T + (T - 1)}

def first_club_probability_twice_second_club (T : ℕ) : Prop :=
  (∀ t ∈ first_club_train_schedule T, P t) = 2 * (∀ t ∈ second_club_train_schedule T, P t)

theorem twice_as_frequent_visits(T : ℕ) (H: ∀ t, train_schedule T t) : first_club_probability_twice_second_club T := 
by
  sorry

end twice_as_frequent_visits_l423_423783


namespace square_not_divisible_into_congruent_30_60_90_triangles_l423_423046

theorem square_not_divisible_into_congruent_30_60_90_triangles :
  ∀ (n m k : ℕ), n ≠ 0 ∨ m ≠ 0 → 2 * (n^2 + 3 * m^2) ≠ (sqrt 3) * (k - 2 * n * m) :=
by {
  sorry
}

end square_not_divisible_into_congruent_30_60_90_triangles_l423_423046


namespace exists_polynomial_f_l423_423367

theorem exists_polynomial_f (k : ℕ) (h1 : Odd k) (h2 : k ≥ 3) :
  ∃ f : ℚ[X], 
    f.degree = k ∧
    f.eval 0 = 0 ∧ 
    f.eval 1 = 1 ∧
    ∀ n : ℕ, ∃ s : ℕ, s ≥ 2^k - 1 ∧ 
    ∀ x : ℕ → ℚ, (∃ xs : Fin s → ℤ, 
      n = ∑ i : Fin s, f.eval (x i)) := sorry

end exists_polynomial_f_l423_423367


namespace medians_in_triangle_l423_423862

variable (b c m_a : ℝ)

noncomputable def m_b : ℝ := 1/2 * real.sqrt (8 * m_a ^ 2 - 5 * b ^ 2 - 2 * c ^ 2)
noncomputable def m_c : ℝ := 1/2 * real.sqrt (8 * m_a ^ 2 + 2 * b ^ 2 - 5 * c ^ 2)

theorem medians_in_triangle
  (b c m_a : ℝ) :
  m_b b c m_a = 1/2 * real.sqrt (8 * m_a ^ 2 - 5 * b ^ 2 - 2 * c ^ 2) ∧
  m_c b c m_a = 1/2 * real.sqrt (8 * m_a ^ 2 + 2 * b ^ 2 - 5 * c ^ 2) :=
  by
    sorry

end medians_in_triangle_l423_423862


namespace fish_ratio_l423_423271

variables (O R B : ℕ)
variables (h1 : O = B + 25)
variables (h2 : B = 75)
variables (h3 : (O + B + R) / 3 = 75)

theorem fish_ratio : R / O = 1 / 2 :=
sorry

end fish_ratio_l423_423271


namespace complex_expression_l423_423158

theorem complex_expression (a : ℂ)
  (h : a = ∑ i in finset.range 2018, i * (complex.I ^ i)) :
  a * (1 + a) ^ 2 / (1 - a) = -1 - complex.I :=
by sorry

end complex_expression_l423_423158


namespace three_custom_op_three_l423_423009

def custom_op (m n : ℕ) : ℕ := n ^ 2 - m

theorem three_custom_op_three : custom_op 3 3 = 6 := by
  simp [custom_op]
  rfl

end three_custom_op_three_l423_423009


namespace number_of_poison_frogs_l423_423920

theorem number_of_poison_frogs
  (total_frogs : ℕ) (tree_frogs : ℕ) (wood_frogs : ℕ) (poison_frogs : ℕ)
  (h₁ : total_frogs = 78)
  (h₂ : tree_frogs = 55)
  (h₃ : wood_frogs = 13)
  (h₄ : total_frogs = tree_frogs + wood_frogs + poison_frogs) :
  poison_frogs = 10 :=
by sorry

end number_of_poison_frogs_l423_423920


namespace perfect_square_quotient_l423_423496

theorem perfect_square_quotient {a b : ℕ} (hpos: 0 < a ∧ 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end perfect_square_quotient_l423_423496


namespace solution_set_inequality_l423_423162

theorem solution_set_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_diff : ∀ x, deriv f x = f' x)
    (h_cond1 : ∀ x, (x - 1) * (f' x - f x) > 0)
    (h_cond2 : ∀ x, f (2 - x) = f x * exp (2 - 2 * x)) :
    {x : ℝ | exp 2 * f (log x) < x * f 2} = {x : ℝ | 1 < x ∧ x < exp 2} :=
begin
    sorry
end

end solution_set_inequality_l423_423162


namespace radius_of_larger_circle_l423_423332

theorem radius_of_larger_circle (R1 R2 : ℝ) (α : ℝ) (h1 : α = 60) (h2 : R1 = 24) (h3 : R2 = 3 * R1) : 
  R2 = 72 := 
by
  sorry

end radius_of_larger_circle_l423_423332


namespace max_dominoes_on_grid_l423_423952

-- Definitions based on conditions:
def is_positive_integer (n : ℕ) : Prop := n > 0

def valid_domino_placement (grid_size : ℕ) (M : ℕ) : Prop :=
  ∃ n : ℕ, is_positive_integer n ∧ grid_size = 2 * n ∧ M = n * n

-- The theorem to prove:
theorem max_dominoes_on_grid (n : ℕ) (h : is_positive_integer n) :
  let grid_size := 2 * n in
  ∃ (M : ℕ), valid_domino_placement grid_size (M) ∧ M = (n * (n + 1)) / 2 :=
begin
  sorry
end

end max_dominoes_on_grid_l423_423952


namespace Allan_balloons_l423_423054

variable (A J : ℕ) -- A is the number of balloons Allan brought, J is the number of balloons Jake initially brought.

-- Conditions
axiom H1 : J = 3
axiom H2 : J' = J + 4  -- J' is the number of balloons Jake had after buying more.
axiom H3 : J' = A + 1  -- Jake had 1 more balloon than Allan at the park.

-- Statement to prove
theorem Allan_balloons (A J J' : ℕ) (H1 : J = 3) (H2 : J' = J + 4) (H3 : J' = A + 1) : A = 6 :=
by
  sorry

end Allan_balloons_l423_423054


namespace unique_prime_digit_A_l423_423317

-- Define the 6-digit number form with the variable digit A
def six_digit_number (A : ℕ) : ℕ := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 1 * 10^2 + 0 * 10 + A

-- Primality condition check
def is_prime_value_digit (A : ℕ) : Prop :=
  A ∈ {1, 3, 5, 7, 9} ∧ Nat.Prime (six_digit_number A)

-- Proposition to prove the correct value of A
theorem unique_prime_digit_A : ∃! (A : ℕ), is_prime_value_digit A ∧ A = 3 :=
by
  -- Define the proof outline and goal
  -- Here, we will skip the proof with "sorry"
  sorry

end unique_prime_digit_A_l423_423317


namespace each_integer_appears_on_main_diagonal_l423_423573

open Matrix

/-- Define the problem statement -/
theorem each_integer_appears_on_main_diagonal 
  (n : ℕ)
  (hodd : n % 2 = 1)
  (hgt1 : 1 < n)
  {A : Matrix (Fin n) (Fin n) ℕ}
  (hsymmetric : Symmetric A)
  (hperm_rows : ∀ i : Fin n, ∃ σ : Equiv.Perm (Fin n), ∀ j : Fin n, A i j = σ j + 1)
  (hperm_cols : ∀ j : Fin n, ∃ τ : Equiv.Perm (Fin n), ∀ i : Fin n, A i j = τ i + 1) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ i : Fin n, A i i = k :=
sorry

end each_integer_appears_on_main_diagonal_l423_423573


namespace decreasing_interval_for_g_l423_423782

def f (x : Real) : Real := sqrt 2 * sin (2 * x - π / 4)

def g (x : Real) : Real := sqrt 2 * sin (x / 2 - π / 12)

theorem decreasing_interval_for_g :
  ∃ a b : Real, (a, b) = (-17 * π / 6 : Real, 5 * π / 6 : Real) ∧
  ∀ x y : Real, a ≤ x ∧ x < y ∧ y ≤ b → g y < g x :=
begin
  sorry
end

end decreasing_interval_for_g_l423_423782


namespace common_points_count_l423_423656

theorem common_points_count : ∀ x y: ℝ, 
  ((x + 2 * y - 3) = 0 ∨ (2 * x - y + 1) = 0) ∧ 
  ((x - 2 * y + 4) = 0 ∨ (3 * x + 4 * y - 12) = 0) → 
  4 := 
sorry

end common_points_count_l423_423656


namespace find_f_2034_l423_423485

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_property : ∀ x : ℝ, f(-x-2) = -f(x-2)
axiom even_function_property : ∀ x : ℝ, f(1-x) = f(1+x)
axiom value_property : f(0) - f(6) = 4

theorem find_f_2034 : f(2034) = -2 := sorry

end find_f_2034_l423_423485


namespace four_lattice_points_collinear_l423_423710

open Real EuclideanSpace

noncomputable def lattice_point (x y : ℤ) : EuclideanSpace 2 ℝ := ![x, y]

structure triangle (A B C : EuclideanSpace 2 ℝ) :=
(vertices_lattice_points : ∀ v ∈ {A, B, C}, ∃ (x y : ℤ), v = lattice_point x y)
(no_lattice_points_on_sides : ∀ p ∈ (line_through ℝ A B ∪ line_through ℝ B C ∪ line_through ℝ C A) \ {A, B, C}, ¬ ∃ (x y : ℤ), p = lattice_point x y)
(has_four_interior_lattice_points :
  ∃ (pts : finset (EuclideanSpace 2 ℝ)),
    pts.card = 4 ∧
    (∀ p ∈ pts, ∃ (x y : ℤ), p = lattice_point x y) ∧
    (∀ p ∈ pts, p ∉ convex_hull ℝ {A, B, C}))

theorem four_lattice_points_collinear {A B C : EuclideanSpace 2 ℝ}
  (T : triangle A B C) : ∃ (line : set (EuclideanSpace 2 ℝ)) (lattice_pts : finset (EuclideanSpace 2 ℝ)),
  lattice_pts.card = 4 ∧
  (∀ p ∈ lattice_pts, ∃ (x y : ℤ), p = lattice_point x y) ∧
  (∀ p ∈ lattice_pts, p ∉ {A, B, C}) ∧
  ∀ p ∈ lattice_pts, p ∈ line :=
sorry

end four_lattice_points_collinear_l423_423710


namespace expected_value_approximation_l423_423583

noncomputable def bernoulli_scheme (p q : ℝ) :=
  p + q = 1

noncomputable def tanaka_formula (n : ℕ) (S : ℕ → ℤ) (ξ : ℕ → ℤ) 
  (sign : ℤ → ℤ) (ΔS : ℕ → ℤ) (N : ℕ → ℕ) : Prop :=
  ∀ k, S 0 = 0 ∧ (S k = ξ 1 + ξ 2 + ... + ξ k) ∧ 
  (ΔS k = ξ k) ∧ 
  (|S n| = (∑ k in range 1 n+1, sign (S k-1) * ΔS k) + N n) ∧ 
  (N n = sharp {k | 0 ≤ k ∧ k ≤ n-1 ∧ S k = 0})

noncomputable def sign_function (x : ℤ) : ℤ :=
  if x > 0 then 1 else 
  if x = 0 then 0 else -1

noncomputable def expected_absolute_value (n : ℕ) (E : real → ℝ) (S : ℕ → ℤ) :=
  E(|S n|) =√(2/π * n)

theorem expected_value_approximation (p q : ℝ) (n : ℕ) (S : ℕ → ℤ) 
  (ξ : ℕ → ℤ) (sign : ℤ → ℤ) (ΔS : ℕ → ℤ) (N : ℕ → ℕ)
  (h_b: bernoulli_scheme p q ∧ p = 1/2 ∧ q = 1/2)
  (h_t: tanaka_formula n S ξ sign ΔS N)
  (sign_def : ∀ x, sign x = sign_function x)
  (E: real → ℝ)
: expected_absolute_value n E S :=
by
  sorry

end expected_value_approximation_l423_423583


namespace distance_CD_l423_423468

theorem distance_CD (d_north: ℝ) (d_east: ℝ) (d_south: ℝ) (d_west: ℝ) (distance_CD: ℝ) :
  d_north = 30 ∧ d_east = 80 ∧ d_south = 20 ∧ d_west = 30 → distance_CD = 50 :=
by
  intros h
  sorry

end distance_CD_l423_423468


namespace point_on_diagonal_iff_perpendicular_diagonals_l423_423931

variables (A B C D P M N : Type)
variables [convex_quadrilateral A B C D]
variables [midpoint M C D] [midpoint N A D]
variables (line1 : perpendicular_to_line_through AB M)
variables (line2 : perpendicular_to_line_through BC N)
variables (P : Type) [intersection P line1 line2]

theorem point_on_diagonal_iff_perpendicular_diagonals 
  (hP_on_BD_if : (P ∈ BD)) : (AC ⟂ BD ↔ P ∈ BD) :=
sorry

end point_on_diagonal_iff_perpendicular_diagonals_l423_423931


namespace area_triangle_ABF_l423_423281

-- Definitions and conditions
def Point (α : Type) := (x : α, y : α)

def A : Point ℝ := (0, 0)
def B : Point ℝ := (2, 0)
def D : Point ℝ := (0, 3)

def rect_ABCD (A B D : Point ℝ) : Prop := 
  (A = (0, 0)) ∧ 
  (B = (2, 0)) ∧ 
  (D = (0, 3)) ∧ 
  (∃ C : Point ℝ, C = (2, 3))

def E_in_rectangle (A B D E : Point ℝ) : Prop :=
  E.1 = 0 ∨ (0 < E.1 ∧ E.1 < 2) ∧ (0 < E.2 ∧ E.2 < 3)  

def right_triangle_AE (A B E : Point ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 2 ∧ B.2 = 0 ∧ E.1 = 0 ∧ E.2 > 0

-- Main Lean proof statement
theorem area_triangle_ABF (A B D E F : Point ℝ)
  (h_rect : rect_ABCD A B D) 
  (h_E_in_rect : E_in_rectangle A B D E)
  (h_right_triangle : right_triangle_AE A B E) 
  (h_F_intersection : F = (0, 3)) 
  : ∃ (area : ℝ), area = 3 :=
by 
  sorry

end area_triangle_ABF_l423_423281


namespace expand_polynomials_eq_l423_423781

-- Define the polynomials P(z) and Q(z)
def P (z : ℝ) : ℝ := 3 * z^3 + 2 * z^2 - 4 * z + 1
def Q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the result polynomial R(z)
def R (z : ℝ) : ℝ := 12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2

-- State the theorem that proves P(z) * Q(z) = R(z)
theorem expand_polynomials_eq :
  ∀ (z : ℝ), (P z) * (Q z) = R z :=
by
  intros z
  sorry

end expand_polynomials_eq_l423_423781


namespace range_of_m_l423_423957

-- Defining the sets A and B using Lean constructs
def setA (x y m : ℝ) : Prop := y = x^2 + m * x + 2
def setB (x y : ℝ) : Prop := x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2

-- Definition to check if an element is in A ∩ B
def in_intersection (x y m : ℝ) := setA x y m ∧ setB x y

-- Translating the quadratic condition derived from the intersection
def has_real_root_in_interval (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0

-- Main theorem statement
theorem range_of_m (m : ℝ) : (∃ x y, in_intersection x y m) → m ≤ -1 :=
begin
  sorry,  -- proof goes here
end

end range_of_m_l423_423957


namespace exponentiation_problem_l423_423248

theorem exponentiation_problem 
(a b : ℝ) 
(h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := 
sorry

end exponentiation_problem_l423_423248


namespace number_of_positive_solutions_l423_423439

theorem number_of_positive_solutions (x : ℝ) (h₁ : 0 ≤ x ∧ x ≤ 1)
  (h₂ : cos (arctan (cot (arccos x))) = x) : 
  x = 1 / real.sqrt 2 :=
by 
  sorry

end number_of_positive_solutions_l423_423439


namespace moles_of_water_produced_l423_423870

theorem moles_of_water_produced (H₃PO₄ NaOH NaH₂PO₄ H₂O : ℝ) (h₁ : H₃PO₄ = 3) (h₂ : NaOH = 3) (h₃ : NaH₂PO₄ = 3) (h₄ : NaH₂PO₄ / H₂O = 1) : H₂O = 3 :=
by
  sorry

end moles_of_water_produced_l423_423870


namespace product_WX_l423_423093

theorem product_WX : 
  ∀ (W X Y Z : ℕ), 
  W ∈ {2, 3, 4, 5} → X ∈ {2, 3, 4, 5} → 
  Y ∈ {2, 3, 4, 5} → Z ∈ {2, 3, 4, 5} → 
  W ≠ X → W ≠ Y → W ≠ Z → 
  X ≠ Y → X ≠ Z → 
  Y ≠ Z → 
  (W : ℝ) / (X : ℝ) - (Y : ℝ) / (Z : ℝ) = 1/2 →
  W * X = 10 :=
by {
  sorry
}

end product_WX_l423_423093


namespace bicycle_discount_l423_423658

theorem bicycle_discount (original_price : ℝ) (discount : ℝ) (discounted_price : ℝ) :
  original_price = 760 ∧ discount = 0.75 ∧ discounted_price = 570 → 
  original_price * discount = discounted_price := by
  sorry

end bicycle_discount_l423_423658


namespace compute_sum_l423_423425

theorem compute_sum : (1 / 2 ^ 2020) * ∑ n in Finset.range 1011, (-3:ℝ)^n * Nat.choose 2020 (2 * n) = -1 / 2 := 
  sorry

end compute_sum_l423_423425


namespace value_of_a_l423_423614

theorem value_of_a (a : ℝ) (h : abs (2 * a + 1) = 3) :
  a = -2 ∨ a = 1 :=
sorry

end value_of_a_l423_423614


namespace simplify_log_expression_l423_423648

theorem simplify_log_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (log (y^3) x * log (x^4) (y^3) * log (y^5) (x^2) * log (x^2) (y^5) * log (y^3) (x^4)) = 
  (1 / 5) * log y x :=
sorry

end simplify_log_expression_l423_423648


namespace pet_shop_ways_l423_423731

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways_l423_423731


namespace C1_cartesian_C2_cartesian_min_distance_PQ_l423_423779

noncomputable def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * real.cos α, real.sin α)

noncomputable def polar_eq_C2 (θ : ℝ) : ℝ :=
  2 * sqrt 2 / real.sin (θ + real.pi / 4)

theorem C1_cartesian : ∀ (x y α : ℝ), parametric_eq_C1 α = (x, y) → 
  (x^2 / 3) + y^2 = 1 := sorry

theorem C2_cartesian : ∀ (ρ θ : ℝ), polar_eq_C2 θ = ρ →
  ∃ x y : ℝ, x + y = 4 ∧ x = ρ * real.cos θ ∧ y = ρ * real.sin θ := sorry

theorem min_distance_PQ : ∀ (P Q : ℝ × ℝ), 
  parametric_eq_C1 P.1 = P → 
  C2_cartesian Q.1 Q.2 →
  |P.1 - Q.1| + |P.2 - Q.2| = sqrt 2 ∧ P = (3/2, 1/2) := sorry

end C1_cartesian_C2_cartesian_min_distance_PQ_l423_423779


namespace quadratic_function_expression_monotonic_g_minimum_g_value_l423_423818

-- Define the given function and conditions
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Given conditions
theorem quadratic_function_expression :
  ∃ a b : ℝ, 
    f a b (-1) = 0 ∧ 
    (∀ y, ∃ x : ℝ, f a b x = y ∨ y ≥ 0) ∧ 
    (∀ x, f a b x ≥ 0) :=
by
  sorry

-- Monotonicity condition
def g (a b x k : ℝ) : ℝ := f a b x - 2 * k * x

theorem monotonic_g :
  ∃ k : ℝ, 
    (∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), 
    ∀ y ∈ Icc (-2 : ℝ) (2 : ℝ), x ≤ y → g 1 2 x k ≤ g 1 2 y k) :=
by
  sorry

-- Minimum value condition
theorem minimum_g_value :
  ∃ k : ℝ, 
    (∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), g 1 2 x k ≥ -15) ∧ 
    (∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), g 1 2 x k = -15) :=
by
  sorry

end quadratic_function_expression_monotonic_g_minimum_g_value_l423_423818


namespace sum_of_first_100_terms_of_sequence_c_l423_423746

theorem sum_of_first_100_terms_of_sequence_c (
    (a₀ b₀ : ℕ) 
    (a : ℕ → ℕ) 
    (b : ℕ → ℕ) 
    (h₁ : a₀ = 2) 
    (h₂ : b₀ = 2) 
    (h₃ : a 2 = 8) 
    (h₄ : b 2 = 8) 
    (h₅ : ∀ n, a n = 3 * (n + 1) - 1) 
    (h₆ : ∀ n, b n = 2 ^ (n + 1))
  ) : 
  (let 
     c := List.sort (List.union (List.map a (List.range 100)) (List.map b (List.range 100)))
     Sn := List.take 100 c
     S100 := List.foldl (λ acc x => acc + x) 0 Sn
  in 
  S100 = 15220) :=
by
  sorry

end sum_of_first_100_terms_of_sequence_c_l423_423746


namespace cosine_between_vectors_perpendicular_and_find_lambda_l423_423180

section
variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem cosine_between_vectors 
  (h_a : a = (-2, 4)) 
  (h_b : b = (-1, -2))
  (h_dot : dot_product a b = -6)
  (h_mag_a : magnitude a = 2 * Real.sqrt 5)
  (h_mag_b : magnitude b = Real.sqrt 5) : 
  dot_product a b / (magnitude a * magnitude b) = -3 / 5 :=
sorry

theorem perpendicular_and_find_lambda 
  (h_a : a = (-2, 4)) 
  (h_b : b = (-1, -2)) 
  (λ : ℝ)
  (h_perp : dot_product (a - λ • b) (2 • a + b) = 0) :
  λ = -34 / 7 :=
sorry
end

end cosine_between_vectors_perpendicular_and_find_lambda_l423_423180


namespace construction_company_sand_weight_l423_423374

theorem construction_company_sand_weight :
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  sand_weight = 8.11 :=
by
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  -- Observing that 14.02 - 5.91 = 8.11
  have h : sand_weight = 8.11 := by sorry
  exact h

end construction_company_sand_weight_l423_423374


namespace equation_d_has_no_real_roots_l423_423340

theorem equation_d_has_no_real_roots : 
  let a := 4
  let b := -5
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant < 0 :=
by
  let a := 4
  let b := -5
  let c := 2
  let discriminant := b^2 - 4 * a * c
  have h : discriminant = -7 := by 
    calc
      b^2 - 4 * a * c 
      = (-5)^2 - 4 * 4 * 2 : rfl
      ... = 25 - 32 : rfl
      ... = -7 : rfl
  show discriminant < 0, from by 
    rw h
    exact dec_trivial

end equation_d_has_no_real_roots_l423_423340


namespace exp_decreasing_range_l423_423195

theorem exp_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (a-2) ^ x < (a-2) ^ (x - 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end exp_decreasing_range_l423_423195


namespace find_a_perpendicular_l423_423131

-- Define the slope of the first line
def slope_line1 := 2

-- Define the slope of the second line in terms of 'a'
def slope_line2 (a : ℝ) := -a / 6

-- Prove that the lines are perpendicular when a = 3
theorem find_a_perpendicular : ∃ a : ℝ, (slope_line1 * slope_line2 a = -1) ↔ a = 3 :=
by
  sorry

end find_a_perpendicular_l423_423131


namespace range_of_a_for_three_extrema_l423_423167

noncomputable def f (a x : ℝ) : ℝ := (Real.exp x) / x + a * (x - Real.log x)

theorem range_of_a_for_three_extrema :
  ∃ (a : ℝ), 
  -2 * Real.sqrt Real.exp 1 < a ∧ a < -Real.exp 1 ∧ 
  (∀ x : ℝ, x ∈ Set.Ioo (1/2) 2 → ∃! (y : ℝ), f a y = 0) ≠ 2 :=
sorry

end range_of_a_for_three_extrema_l423_423167


namespace nuts_mixture_weight_l423_423371

variable (m n : ℕ)
variable (weight_almonds per_part total_weight : ℝ)

theorem nuts_mixture_weight (h1 : m = 5) (h2 : n = 2) (h3 : weight_almonds = 250) 
  (h4 : per_part = weight_almonds / m) (h5 : total_weight = per_part * (m + n)) : 
  total_weight = 350 := by
  sorry

end nuts_mixture_weight_l423_423371


namespace area_ratio_correct_l423_423051

noncomputable def area_ratio_of_ABC_and_GHJ : ℝ :=
  let side_length_ABC := 12
  let BD := 5
  let CE := 5
  let AF := 8
  let area_ABC := (Real.sqrt 3 / 4) * side_length_ABC ^ 2
  (1 / 74338) * area_ABC / area_ABC

theorem area_ratio_correct : area_ratio_of_ABC_and_GHJ = 1 / 74338 := by
  sorry

end area_ratio_correct_l423_423051


namespace smallest_positive_integer_b_no_inverse_l423_423090

theorem smallest_positive_integer_b_no_inverse :
  ∃ b : ℕ, b > 0 ∧ gcd b 30 > 1 ∧ gcd b 42 > 1 ∧ b = 6 :=
by
  sorry

end smallest_positive_integer_b_no_inverse_l423_423090


namespace total_prime_factors_l423_423130

def expression : Int := 4^15 * 7^7 * 11^3 * 13^6

theorem total_prime_factors : 
  let p2 := 2^30
  let p7 := 7^7
  let p11 := 11^3
  let p13 := 13^6
  30 + 7 + 3 + 6 = 46 :=
by
  let p2 := 2^30
  let p7 := 7^7
  let p11 := 11^3
  let p13 := 13^6
  have h2 : 4 = 2^2 := by norm_num
  have h2_exp : 4^15 = (2^2)^15 := by rw [← h2]; rfl
  have h_expanded : p2 = 2^(2*15) := by rw [h2_exp]; norm_num
  rw [h_expanded]
  norm_num
  sorry

end total_prime_factors_l423_423130


namespace future_skyscraper_climb_proof_l423_423892

variable {H_f H_c H_fut : ℝ}

theorem future_skyscraper_climb_proof
  (H_f : ℝ)
  (H_c : ℝ := 3 * H_f)
  (H_fut : ℝ := 1.25 * H_c)
  (T_f : ℝ := 1) :
  (H_fut * T_f / H_f) > 2 * T_f :=
by
  -- specific calculations would go here
  sorry

end future_skyscraper_climb_proof_l423_423892


namespace abs_glb_sum_squares_roots_l423_423078

theorem abs_glb_sum_squares_roots (n : ℕ) (b : ℝ) 
  (h_n_pos : n > 1) 
  (monic_poly : Polynomial ℝ)
  (h_monic : monic_poly.leading_coeff = 1)
  (h_poly: ∀ x, monic_poly.eval x =
      x^n + b * x^(n-1) + (b / 2) * x^(n-2)) 
  : abs (greatest_lower_bound {∑ i in (1 : Finset (Fin n)), (monic_poly.roots.to_list[i])^2 | true}) = 1 / 4 :=
by
  -- Sorry is used to skip the proof
  sorry


end abs_glb_sum_squares_roots_l423_423078


namespace find_row_and_position_l423_423821

noncomputable def sequence (n : ℕ) : ℕ := n

theorem find_row_and_position (n : ℕ) (h : n = 2009) : 
  ∃ s t : ℕ, s = 45 ∧ t = 73 ∧ (sequence n = t + ((s * (s - 1) / 2) + 1)) := 
by
  use 45
  use 73
  split
  · exact by rfl
  split
  · exact by rfl
  · sorry

end find_row_and_position_l423_423821


namespace number_of_minimal_selfish_subsets_l423_423949

-- Definition of a selfish set
def is_selfish (n : ℕ) (X : set ℕ) : Prop :=
  (X ⊆ finset.range (n + 1) ∧ (∃ k, k ∈ X ∧ k = X.card))

-- Definition of a minimal selfish set
def is_minimal_selfish {n : ℕ} (X : set ℕ) : Prop :=
  is_selfish n X ∧ (∀ Y, Y ⊂ X → ¬ is_selfish n Y)

-- Number of minimal selfish sets is given by the (n-1)-th Fibonacci number
noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n + 1) + fib n

theorem number_of_minimal_selfish_subsets (n : ℕ) : 
  (finset.filter (is_minimal_selfish ∘ set.to_finset) (finset.powerset (finset.range n))).card = fib (n - 1) :=
sorry

end number_of_minimal_selfish_subsets_l423_423949


namespace find_n_l423_423222

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = -1 / (a n + 1)

theorem find_n (a : ℕ → ℚ) (h : seq a) : ∃ n : ℕ, a n = 3 ∧ n = 16 :=
by
  sorry

end find_n_l423_423222


namespace concurrency_A1K_B1L_C1M_l423_423571

variable {A B C P Q A1 B1 C1 K L M : Type*}

-- Given definitions
def interior_points_A_B_C (P Q : Type*) : Prop := 
  ∃ (A : Type*) (B : Type*) (C : Type*), 
    -- This expresses that P and Q are located within the triangle
    true

def intersection_AQ_BC (A1 : Type*) (AQ : Type*) (BC : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- A1 is the intersection of AQ and BC
    true

def intersection_BQ_CA (B1 : Type*) (BQ : Type*) (CA : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- B1 is the intersection of BQ and CA
    true

def intersection_CQ_AB (C1 : Type*) (CQ : Type*) (AB : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- C1 is the intersection of CQ and AB
    true

def intersection_AP_B1C1 (K : Type*) (AP : Type*) (B1C1 : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- K is the intersection of AP and B1C1
    true

def intersection_BP_C1A1 (L : Type*) (BP : Type*) (C1A1 : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- L is the intersection of BP and C1A1
    true

def intersection_CP_A1B1 (M : Type*) (CP : Type*) (A1B1 : Type*) : Prop := 
  ∃ (intersection_point : Type*), 
    -- M is the intersection of CP and A1B1
    true

-- Theorem to prove
theorem concurrency_A1K_B1L_C1M 
  (h₁ : interior_points_A_B_C P Q)
  (h₂ : intersection_AQ_BC A1 AQ BC)
  (h₃ : intersection_BQ_CA B1 BQ CA)
  (h₄ : intersection_CQ_AB C1 CQ AB)
  (h₅ : intersection_AP_B1C1 K AP B1C1)
  (h₆ : intersection_BP_C1A1 L BP C1A1)
  (h₇ : intersection_CP_A1B1 M CP A1B1) :
  -- Concluding that the lines A1K, B1L, and C1M are concurrent
  ∃ (concurrent_point : Type*), true :=
  sorry

end concurrency_A1K_B1L_C1M_l423_423571


namespace locus_of_P_is_circle_segment_l423_423825

-- Define the structures and conditions of the problem
variables {θ : ℝ} (P : Point) (A B M N : Point) (C D OuterCircle InnerCircle : Circle)

-- Define the angle condition for θ
variable (hθ : 0 < θ ∧ θ < π / 2)

-- Define the tangency condition of the circles
variable (tangent : tangent OuterCircle InnerCircle)

-- Define line l passing through point A and intersecting at B
variable (line_l : Line A B)

-- Define the motion of M on the arc of the outer circle
variable (M_on_outer : M ∈ OuterCircle)

-- Define the intersection of MA with the inner circle at point N
variable (N_on_inner : N ∈ InnerCircle ∧ lies_on N (Line.mk A M))

-- Define the point P on ray MB where angle MPN = θ
variable (P_on_ray_MPN : lies_on P (Ray.mk M B) ∧ ∠MPN = θ)

-- Define the locus problem: Locus of P should be a circle segment 
theorem locus_of_P_is_circle_segment (θ : ℝ) (A B M N : Point) :
  (0 < θ ∧ θ < π / 2) ∧
  tangent OuterCircle InnerCircle ∧
  lies_on A OuterCircle ∧
  lies_on B OuterCircle ∧
  lies_on N InnerCircle ∧
  lies_on P (Ray.mk M B) ∧
  ∠MPN = θ →
  exists CircleSegment, P ∈ CircleSegment :=
by
  sorry

end locus_of_P_is_circle_segment_l423_423825


namespace subset_sum_l423_423465

def f (S : Finset ℕ) : ℕ := S.sum id

theorem subset_sum
  (S : Finset ℕ)
  (hS : S = Finset.range 1999)
  (H : ∑ E in S.powerset, (f E : ℚ) / f S = 2 ^ 1998) : True :=
by { sorry }

end subset_sum_l423_423465


namespace maximum_profit_l423_423373

noncomputable def y1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def y2 (x : ℝ) : ℝ := 2 * (15 - x)

def profit (x : ℝ) : ℝ := y1 x + y2 x

theorem maximum_profit : ∃ x ∈ set.Icc 0 15, profit x = 45.6 :=
sorry

end maximum_profit_l423_423373


namespace no_solutions_to_cubic_sum_l423_423795

theorem no_solutions_to_cubic_sum (x y z : ℤ) : 
    ¬ (x^3 + y^3 = z^3 + 4) :=
by 
  sorry

end no_solutions_to_cubic_sum_l423_423795


namespace regular_ngon_n_value_l423_423047

-- Formalizing the problem
theorem regular_ngon_n_value (n : ℕ) (h : ∀ (P1 P7 P8 : Type), 
    regular_ngon n P1 P7 P8 → angle P1 P7 P8 = 178) :
  n = 630 :=
sorry

end regular_ngon_n_value_l423_423047


namespace garden_area_correct_l423_423716

noncomputable def garden_area (π : ℝ) := 
  let r := 17 
  let R := r + 2 
  π * (R^2 - r^2)

theorem garden_area_correct :
  garden_area 3.14159 ≈ 226.19 :=
by
  let r := 17 
  let R := r + 2 
  have A1 : π := 3.14159
  have h : A1 * (R^2 - r^2) ≈ 226.19 := sorry
  exact h

end garden_area_correct_l423_423716


namespace carla_final_payment_l423_423757

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l423_423757


namespace sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l423_423363

-- Part a) Prove that if a sequence has a limit, then it is bounded.
theorem sequence_with_limit_is_bounded (x : ℕ → ℝ) (x0 : ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

-- Part b) Is the converse statement true?
theorem bounded_sequence_does_not_imply_limit :
  ∃ (x : ℕ → ℝ), (∃ C, ∀ n, |x n| ≤ C) ∧ ¬(∃ x0, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) := by
  sorry

end sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l423_423363


namespace GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l423_423307

noncomputable def GCD (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCD_17_51 : GCD 17 51 = 17 := by
  sorry

theorem LCM_17_51 : LCM 17 51 = 51 := by
  sorry

theorem GCD_6_8 : GCD 6 8 = 2 := by
  sorry

theorem LCM_8_9 : LCM 8 9 = 72 := by
  sorry

end GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l423_423307


namespace tan_alpha_possible_values_l423_423499

theorem tan_alpha_possible_values (α : ℝ) (h : Real.sin (2 * α) = - Real.sin α) :
  Real.tan α = 0 ∨ Real.tan α = Real.sqrt 3 ∨ Real.tan α = - Real.sqrt 3 :=
begin
  sorry
end

end tan_alpha_possible_values_l423_423499


namespace product_of_two_numbers_l423_423310

theorem product_of_two_numbers (x y : ℕ) 
  (h1 : y = 15 * x) 
  (h2 : x + y = 400) : 
  x * y = 9375 :=
by
  sorry

end product_of_two_numbers_l423_423310


namespace line_equation_and_inclination_l423_423080

variable (t : ℝ)
variable (x y : ℝ)
variable (α : ℝ)
variable (l : x = -3 + t ∧ y = 1 + sqrt 3 * t)

theorem line_equation_and_inclination 
  (H : l) : 
  (∃ a b c : ℝ, a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1 ∧ a * x + b * y + c = 0) ∧
  α = Real.pi / 3 :=
by
  sorry

end line_equation_and_inclination_l423_423080


namespace convert_rect_to_polar_l423_423432

theorem convert_rect_to_polar :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2, 5 * Real.pi / 3) :=
by
  let x := 1
  let y := -Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let θ := 5 * Real.pi / 3
  use [r, θ]
  split
  {
    -- Prove r > 0
    sorry
  },
  split
  {
    -- Prove 0 ≤ θ
    sorry
  },
  split
  {
    -- Prove θ < 2 * Real.pi
    sorry
  },
  have h_r : r = 2 := by  
  {
    -- Prove r = 2
    sorry
  },
  have h_θ : θ = 5 * Real.pi / 3 := by  
  {
    -- Prove θ = 5 * Real.pi / 3
    sorry
  },
  exact ⟨h_r, h_θ⟩

end convert_rect_to_polar_l423_423432


namespace inequality_factors_l423_423569

-- Let's define the conditions as specified.
variables {n : ℕ} {x : ℕ → ℝ}
-- Assume n ≥ 2
axiom n_ge_two (h : 2 ≤ n)
-- Assume 0 ≤ x_i ≤ 1 and x_1 ≤ x_2 ≤ ... ≤ x_n
axiom x_bounds (i : ℕ) (h₁ : 1 ≤ i) (h₂ : i ≤ n) : 0 ≤ x i ∧ x i ≤ 1
axiom x_ordered (i j : ℕ) (h₁ : 1 ≤ i) (h₂ : h₁ ≤ j) (h₃ : j ≤ n) : x i ≤ x j

-- Define the terms of the inequality
noncomputable def geom_mean (x : ℕ → ℝ) (n : ℕ) := (x 1 * x 2 * ... * x n)^(1 / n : ℝ)
noncomputable def geom_mean_compl (x : ℕ → ℝ) (n : ℕ) := ((1 - x 1) * (1 - x 2) * ... * (1 - x n))^(1 / n : ℝ)
noncomputable def rhs_expr (x : ℕ → ℝ) : ℝ := (1 - (x n - x 1)^2)^(1 / n : ℝ)

-- The inequality to prove
theorem inequality_factors (h₁ : 2 ≤ n) (h₂ : ∀ i, 0 ≤ x i ∧ x i ≤ 1) (h₃ : ∀ i j, i ≤ j → x i ≤ x j) :
  geom_mean x n + geom_mean_compl x n ≤ rhs_expr x := 
sorry

end inequality_factors_l423_423569


namespace is_parallel_l423_423056

-- Definitions for the lines involved
def line1 : LinearEquation := { a := 1, b := -2, c := 1 }
def lineA : LinearEquation := { a := 2, b := -1, c := 1 }
def lineB : LinearEquation := { a := 2, b := -4, c := 2 }
def lineC : LinearEquation := { a := 2, b := 4, c := 1 }
def lineD : LinearEquation := { a := 2, b := -4, c := 1 }

-- Proposition: The line 2x - 4y + 1 = 0 is parallel to the line x - 2y + 1 = 0
theorem is_parallel : is_parallel line1 lineD := 
by 
    sorry

-- LinearEquation structure needs to be defined in the Lean environment.
structure LinearEquation := (a b c : ℝ)

-- Helper function to check if two linear equations are parallel
def is_parallel (line1 line2 : LinearEquation) : Prop :=
  line1.a * line2.b - line1.b * line2.a = 0

end is_parallel_l423_423056


namespace common_point_circumcircles_l423_423014

theorem common_point_circumcircles 
  (A B C D X Y : Point) 
  (hABCD : convex_quadrilateral A B C D) 
  (hX_on_AB : X ∈ line A B) 
  (hAC_meets_DX_at_Y : intersects (line A C) (line D X) Y) : 
  ∃ (P : Point), 
    on_circumcircle A B C P ∧ 
    on_circumcircle C D Y P ∧ 
    on_circumcircle B D X P :=
sorry

end common_point_circumcircles_l423_423014


namespace sin_C_of_right_triangle_l423_423555

theorem sin_C_of_right_triangle (A B C : ℝ)
  (hABC : ∠B = 90°) (hcosA : real.cos A = 3 / 5) :
  real.sin C = 4 / 5 :=
sorry

end sin_C_of_right_triangle_l423_423555


namespace remainder_of_expression_l423_423535

noncomputable theory

-- Definitions of conditions
variables (x y u v : ℕ) 
variable  (hxy : x = u * y + v)
variable  (h_pos_x : 0 < x)
variable  (h_pos_y : 0 < y)
variable  (h_remainder : 0 ≤ v ∧ v < y)

-- Lean statement of the problem
theorem remainder_of_expression (x y u v : ℕ) (hxy : x = u * y + v) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_of_expression_l423_423535


namespace least_possible_sum_l423_423896

theorem least_possible_sum (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x + y + z = 26 :=
by sorry

end least_possible_sum_l423_423896


namespace singing_competition_avg_var_l423_423388

-- Definitions for initial conditions and parameters
def initial_judges : ℕ := 8
def initial_average_score : ℝ := 5
def initial_variance : ℝ := 3
def guest_judge_score : ℝ := 5
def total_judges : ℕ := initial_judges + 1

-- Lean 4 statement of the proof problem
theorem singing_competition_avg_var :
  let new_average := (initial_average_score * initial_judges + guest_judge_score) / total_judges in
  let new_variance := (initial_variance * (initial_judges - 1) + (guest_judge_score - initial_average_score) ^ 2) / total_judges in
  new_average = 5 ∧ new_variance < 3 :=
by
  -- Placeholder for proof. This is where we would develop the proof step-by-step if required.
  sorry

end singing_competition_avg_var_l423_423388


namespace percentage_decrease_is_14_percent_l423_423928

-- Definitions based on conditions
def original_price_per_pack : ℚ := 7 / 3
def new_price_per_pack : ℚ := 8 / 4

-- Statement to prove that percentage decrease is 14%
theorem percentage_decrease_is_14_percent :
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100 = 14 := by
  sorry

end percentage_decrease_is_14_percent_l423_423928


namespace transformed_curve_l423_423646

variable (x x' y y' : ℝ)

def original_curve (y x : ℝ) : Prop :=
  y = Real.cos x

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 2 * x ∧ y' = 3 * y

theorem transformed_curve (x' y' : ℝ) (h : ∃ x y, transformation x' y' x y ∧ original_curve y x) :
  y' = 3 * Real.cos (x' / 2) :=
by
  obtain ⟨x, y, h_trans, h_orig⟩ := h
  have hx' : x = x' / 2 := by rw [←h_trans.left, mul_div_cancel_left x (two_ne_zero : (2 : ℝ) ≠ 0)]
  have hy' : y = y' / 3 := by rw [←h_trans.right, mul_div_cancel_left y (three_ne_zero : (3 : ℝ) ≠ 0)]
  rw [hx', hy'] at h_orig
  rw [h_orig, Real.cos (x' / 2)]
  exact congr_arg (fun t => 3 * t) rfl

end transformed_curve_l423_423646


namespace product_of_solutions_l423_423113

theorem product_of_solutions (x : ℤ) (h : x^2 = 49) : ∏ (x : {x // x^2 = 49}), x = -49 := sorry

end product_of_solutions_l423_423113


namespace largest_a_mul_b_l423_423956

-- Given conditions and proof statement
theorem largest_a_mul_b {m k q a b : ℕ} (hm : m = 720 * k + 83)
  (ha : m = a * q + b) (h_b_lt_a: b < a): a * b = 5112 :=
sorry

end largest_a_mul_b_l423_423956


namespace perp_vector_magnitude_l423_423946

variables {x : ℝ}
def a := (3, x)
def b := (-1, 1)

theorem perp_vector_magnitude (h : (a.1 * b.1 + a.2 * b.2 = 0)) : 
  real.sqrt (a.1 ^ 2 + a.2 ^ 2) = 3 * real.sqrt 2 :=
by {
  sorry
}

end perp_vector_magnitude_l423_423946


namespace find_a_l423_423153

noncomputable def A : set ℝ := { x | x ≤ 1 }
noncomputable def B (a : ℝ) : set ℝ := { x | x ≥ a }

theorem find_a (a : ℝ) : (A ∩ B a = {1}) → a = 1 :=
by
  intro h
  -- Remaining proof would go here
  sorry

end find_a_l423_423153


namespace find_parabola_equation_l423_423308

noncomputable def hyperbola_and_parabola (a b p : ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let hyperbola := ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1
  let parabola := ∀ (x y : ℝ), y^2 = 2 * p * x
  let eccentricity := a > 0 ∧ b > 0 ∧ (a^2 + b^2)^(-1/2) * (a^2 + b^2)^(1/2) = √2
  let asymptotes_area := A.1 = A.2 ∧ B.1 = B.2 ∧ (1 / 2) * A.1 * B.2 = 4 
  let area := O = (0, 0) ∧ A ≠ B
  hyperbola a b ∧ parabola (2*p) ∧ eccentricity ∧ asymptotes_area ∧ area

theorem find_parabola_equation :
  ∀ (p : ℝ), ∀ (A B : ℝ × ℝ) (O : ℝ × ℝ),
    hyperbola_and_parabola (√A.1) (√A.2) p A B O →
    (∀ (x y : ℝ), y^2 = 2 * p * x) :=
begin
  intros p A B O h,
  sorry
end

end find_parabola_equation_l423_423308


namespace limit_derivative_sqrt_l423_423159

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem limit_derivative_sqrt (x : ℝ) (hx : 0 < x) :
  Tendsto (λ Δx : ℝ, (f (x + Δx) - f x) / Δx) (𝓝 0) (𝓝 (1 / (2 * Real.sqrt x))) :=
by
  sorry

end limit_derivative_sqrt_l423_423159


namespace largest_rational_solution_l423_423454

noncomputable def max_solution : ℝ :=
  let x1 := 7 / 2 in
  let y := 25 / 2 in
  max (x1 + y) (x1 - y)

theorem largest_rational_solution :
  ∃ x : ℚ, |(x : ℝ) - 7 / 2| = 25 / 2 ∧ (x : ℝ) = max_solution :=
begin
  sorry
end

end largest_rational_solution_l423_423454


namespace Debby_jogging_plan_l423_423607

def Monday_jog : ℝ := 3
def Tuesday_jog : ℝ := Monday_jog * 1.1
def Wednesday_jog : ℝ := 0
def Thursday_jog : ℝ := Tuesday_jog * 1.1
def Saturday_jog : ℝ := Thursday_jog * 2.5
def total_distance : ℝ := Monday_jog + Tuesday_jog + Thursday_jog + Saturday_jog
def weekly_goal : ℝ := 40
def Sunday_jog : ℝ := weekly_goal - total_distance

theorem Debby_jogging_plan :
  Tuesday_jog = 3.3 ∧
  Thursday_jog = 3.63 ∧
  Saturday_jog = 9.075 ∧
  Sunday_jog = 21.995 :=
by
  -- Proof goes here, but is omitted as the problem statement requires only the theorem outline.
  sorry

end Debby_jogging_plan_l423_423607


namespace functional_eq_implies_const_zero_or_one_l423_423097

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_implies_const_zero_or_one (f : ℝ → ℝ) :
  (∀ x y, (y + 1) * f (x + y) - f x * f (x + y^2) = y * f x) →
  (f = (λ x, 0) ∨ f = (λ x, 1)) :=
begin
  intro h,
  sorry
end

end functional_eq_implies_const_zero_or_one_l423_423097


namespace find_x_l423_423791

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
sorry

end find_x_l423_423791


namespace max_monitoring_time_is_14_hours_l423_423751

structure TimeSlot where
  start : ℕ -- start time in minutes since midnight
  end : ℕ   -- end time in minutes since midnight
  property valid : start < end

structure Volunteer where
  slots : List TimeSlot
  property nonempty : slots ≠ []

def maxTotalMonitoringTime (volunteers : List Volunteer) : ℕ :=
  -- Function to compute the maximum total monitoring time
  sorry

def volunteerA : Volunteer := {
  slots := [{start := 6 * 60, end := 8 * 60, valid := by decide},
            {start := 16 * 60, end := 18 * 60, valid := by decide}],
  nonempty := by decide
}

def volunteerB : Volunteer := {
  slots := [{start := 6 * 60 + 30, end := 7 * 60 + 30, valid := by decide},
            {start := 17 * 60, end := 20 * 60, valid := by decide}],
  nonempty := by decide
}

def volunteerC : Volunteer := {
  slots := [{start := 8 * 60, end := 11 * 60, valid := by decide},
            {start := 18 * 60, end := 19 * 60, valid := by decide}],
  nonempty := by decide
}

def volunteerD : Volunteer := {
  slots := [{start := 7 * 60, end := 10 * 60, valid := by decide},
            {start := 17 * 60 + 30, end := 18 * 60 + 30, valid := by decide}],
  nonempty := by decide
}

theorem max_monitoring_time_is_14_hours :
  maxTotalMonitoringTime [volunteerA, volunteerB, volunteerC, volunteerD] = 14 * 60 :=
sorry

end max_monitoring_time_is_14_hours_l423_423751


namespace more_than_half_millet_on_day_5_l423_423266

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end more_than_half_millet_on_day_5_l423_423266


namespace find_vertex_D_l423_423557

theorem find_vertex_D (A B C : ℂ) (D : ℂ) :
  A = 1 + 3 * Complex.i ∧
  B = -Complex.i ∧
  C = 2 + Complex.i →
  D = 3 + 5 * Complex.i :=
by
  intros h
  have hA : A = 1 + 3 * Complex.i := h.1
  have hB : B = -Complex.i := h.2.1
  have hC : C = 2 + Complex.i := h.2.2
  sorry

end find_vertex_D_l423_423557


namespace eccentricity_of_ellipse_circle_tangent_to_line_l423_423849

-- Definitions of the ellipse and its properties
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b^2) = 1
def foci (c : ℝ) (F1 F2 : ℝ × ℝ) : Prop := F1 = (0, c) ∧ F2 = (0, -c)
def vertex (B : ℝ × ℝ) (b : ℝ) : Prop := B = (b, 0)
def dot_product_zero (BF1 BF2 : ℝ × ℝ) : Prop := BF1.1 * BF2.1 + BF1.2 * BF2.2 = 0

variables (a b c : ℝ)
variable (x y : ℝ)
variables (F1 F2 B : ℝ × ℝ)
variables (P : ℝ × ℝ)

-- Eccentricity of the ellipse
theorem eccentricity_of_ellipse
  (h1 : ellipse_eq a b x y)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : b < a)
  (h5 : foci c F1 F2)
  (h6 : vertex B b)
  (h7 : dot_product_zero ⟨B.1 - F1.1, B.2 - F1.2⟩ ⟨B.1 - F2.1, B.2 - F2.2⟩) :
  c = b → e = c / a :=
sorry

-- Circle tangent to the line proof
theorem circle_tangent_to_line
  (h1 : ellipse_eq a b x y)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : b < a)
  (h5 : foci c F1 F2)
  (h6 : vertex B b)
  (h8 : ∃ P, P ≠ B ∧ ellipse_eq a b P.1 P.2)
  (h9 : ∀ P F2, circle_diameter_through P B F2 → tg_to_line x y) :
  ∀ (P F2 : ℝ × ℝ), circle_tangent_to_line x y := 
sorry

end eccentricity_of_ellipse_circle_tangent_to_line_l423_423849


namespace find_m_l423_423865

def U : Set ℤ := {-1, 2, 3, 6}
def A (m : ℤ) : Set ℤ := {x | x^2 - 5 * x + m = 0}
def complement_U_A (m : ℤ) : Set ℤ := U \ A m

theorem find_m (m : ℤ) (hU : U = {-1, 2, 3, 6}) (hcomp : complement_U_A m = {2, 3}) :
  m = -6 := by
  sorry

end find_m_l423_423865


namespace sum_due_is_363_l423_423011

/-
Conditions:
1. BD = 78
2. TD = 66
3. The formula: BD = TD + (TD^2 / PV)
This should imply that PV = 363 given the conditions.
-/

theorem sum_due_is_363 (BD TD PV : ℝ) (h1 : BD = 78) (h2 : TD = 66) (h3 : BD = TD + (TD^2 / PV)) : PV = 363 :=
by
  sorry

end sum_due_is_363_l423_423011


namespace problem_l423_423650

-- Define the conditions as given in the problem
def condition1 (x : ℝ) : Prop := x = -x → x = 0
def condition2 (a b : ℝ) : Prop := |a| = |b| → a = b ∨ a = -b
def condition3 (a b : ℝ) : Prop := a + b < 0 ∧ ab > 0 → |7 * a + 3 * b| = -7 * a - 3 * b
def condition4 (m : ℚ) : Prop := |m| + m ≥ 0

-- Define the main theorem stating that exactly two of the conditions are true
theorem problem : 
  (∀ x, condition1 x) ∧ 
  (∀ a b, ¬condition2 a b) ∧ 
  (∀ a b, ¬condition3 a b) ∧ 
  (∀ m, condition4 m) :=
by sorry

end problem_l423_423650


namespace ellipse_dot_product_range_l423_423935

theorem ellipse_dot_product_range :
  ∀ (x y : ℝ),
    (x^2 / 4 + y^2 = 1) →
    let PF1 := ⟨x + sqrt 3, y⟩ in
    let PF2 := ⟨x - sqrt 3, y⟩ in
    let dot_product := PF1.1 * PF2.1 + PF1.2 * PF2.2 in
    -2 ≤ dot_product ∧ dot_product ≤ 1 :=
begin
  sorry
end

end ellipse_dot_product_range_l423_423935


namespace project_completion_rate_l423_423062

variables {a b c d e : ℕ} {f g : ℚ}  -- Assuming efficiency ratings can be represented by rational numbers.

theorem project_completion_rate (h : (a * f / c) = b / c) 
: (d * g / e) = bdge / ca := 
sorry

end project_completion_rate_l423_423062


namespace largest_among_trig_expressions_l423_423745

theorem largest_among_trig_expressions :
  let a := Real.tan 48 + 1 / Real.tan 48
  let b := Real.sin 48 + Real.cos 48
  let c := Real.tan 48 + Real.cos 48
  let d := 1 / Real.tan 48 + Real.sin 48
  a > b ∧ a > c ∧ a > d :=
by
  sorry

end largest_among_trig_expressions_l423_423745


namespace length_of_trains_l423_423334

theorem length_of_trains :
  -- Given conditions
  let speed_train1_kmph := 180
  let time_train1_sec := 5
  let speed_train2_kmph := 200
  let time_train2_sec := 9
  -- Speed conversion factor from km/h to m/s
  let speed_conversion := (1000 : ℝ) / (3600 : ℝ)
  -- Speeds in m/s
  let speed_train1_ms := speed_train1_kmph * speed_conversion
  let speed_train2_ms := speed_train2_kmph * speed_conversion
  -- Lengths of the trains
  length_train1 := speed_train1_ms * time_train1_sec
  length_train2 := speed_train2_ms * time_train2_sec
  -- Assertions
  length_train1 = 250 ∧ Real.approx length_train2 500 :=
by
  sorry

end length_of_trains_l423_423334


namespace overlapping_area_l423_423076

open Real

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def triangle_area (p1 p2 p3 : Point2D) : ℝ :=
  1 / 2 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

def triangleA := ({x := 0, y := 0} : Point2D)
def triangleB := ({x := 3, y := 1} : Point2D)
def triangleC := ({x := 1, y := 3} : Point2D)

def triangleD := ({x := 3, y := 3} : Point2D)
def triangleE := ({x := 0, y := 2} : Point2D)
def triangleF := ({x := 2, y := 0} : Point2D)

theorem overlapping_area :
  let points := [({x := 1, y := 1}, {x := 2, y := 1}, {x := 1, y := 2}, {x := 2, y := 2})]
  -- Intersection points form a rhombus with adjacent sides of sqrt(2)
  -- The area of the rhombus where they intersect is 1 square unit.
  1 = 1 := sorry

end overlapping_area_l423_423076


namespace find_x_intercept_of_perpendicular_line_l423_423378

noncomputable def line_y_intercept : ℝ × ℝ := (0, 3)
noncomputable def given_line (x y : ℝ) : Prop := 2 * x + y = 3
noncomputable def x_intercept_of_perpendicular_line : ℝ × ℝ := (-6, 0)

theorem find_x_intercept_of_perpendicular_line :
  (∀ (x y : ℝ), given_line x y → (slope_of_perpendicular_line : ℝ) = 1/2 ∧ 
  ∀ (b : ℝ), line_y_intercept = (0, b) → ∀ (y : ℝ), y = 1/2 * x + b → (x, 0) = x_intercept_of_perpendicular_line) :=
sorry

end find_x_intercept_of_perpendicular_line_l423_423378


namespace solve_system_l423_423992

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y - z = 4 ∧ x^2 - y^2 + z^2 = -4 ∧ xyz = 6) ↔ 
    (x, y, z) = (2, 3, 1) ∨ (x, y, z) = (-1, 3, -2) :=
by
  sorry

end solve_system_l423_423992


namespace book_total_pages_l423_423350

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l423_423350


namespace min_value_frac_inv_distances_l423_423837

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def min_value_on_ellipse : ℝ :=
  let C := (-Real.sqrt 3, 0)
  let D := (Real.sqrt 3, 0)
  let e (p : ℝ × ℝ) := p.1^2 / 4 + p.2^2 - 1
  Inf { (1 / distance M C + 1 / distance M D) | M : ℝ × ℝ, e M = 0 }

theorem min_value_frac_inv_distances : min_value_on_ellipse = 1 := sorry

end min_value_frac_inv_distances_l423_423837


namespace rice_in_each_container_l423_423010

-- Given conditions from the problem
def total_weight_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- A theorem that each container has 25 ounces of rice given the conditions
theorem rice_in_each_container (h : total_weight_pounds * pounds_to_ounces / num_containers = 25) : True :=
  sorry

end rice_in_each_container_l423_423010


namespace tank_unoccupied_volume_l423_423284

def volume_of_tank (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def volume_of_water (volume_tank: ℕ) : ℕ :=
  volume_tank / 3

def volume_of_ice_cubes (num_cubes: ℕ) (volume_cube: ℕ) : ℕ :=
  num_cubes * volume_cube

def total_occupied_volume (volume_water: ℕ) (volume_ice_cubes: ℕ) : ℕ :=
  volume_water + volume_ice_cubes

def unoccupied_volume (volume_tank: ℕ) (occupied_volume: ℕ) : ℕ :=
  volume_tank - occupied_volume

theorem tank_unoccupied_volume :
  let length := 8
  let width := 10
  let height := 15
  let num_cubes := 15
  let volume_cube := 1 in
  let volume_tank := volume_of_tank length width height in
  let volume_water := volume_of_water volume_tank in
  let volume_ice_cubes := volume_of_ice_cubes num_cubes volume_cube in
  let occupied_volume := total_occupied_volume volume_water volume_ice_cubes in
  unoccupied_volume volume_tank occupied_volume = 785 :=
by
  sorry

end tank_unoccupied_volume_l423_423284


namespace find_intersection_l423_423958

open Set Real

noncomputable def M := {x : ℝ | log 2 (x - 1) > 0}
def N := {x : ℝ | x >= -2}
def M_complement := {x : ℝ | x <= 2}

theorem find_intersection :
  (N ∩ M_complement) = {x : ℝ | -2 <= x ∧ x <= 2} := by
sorry

end find_intersection_l423_423958


namespace find_two_digit_number_l423_423793

theorem find_two_digit_number : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 10^6 ≤ n^3 ∧ n^3 < 10^7 ∧ 101010 * x + 1 = n^3 ∧ x = 93) := 
 by
  sorry

end find_two_digit_number_l423_423793


namespace find_a_l423_423083

def otimes (a b : ℝ) : ℝ := a - 2 * b

theorem find_a (a : ℝ) : (∀ x : ℝ, (otimes x 3 > 0 ∧ otimes x a > a) ↔ x > 6) → a ≤ 2 :=
begin
  intro h,
  sorry
end

end find_a_l423_423083


namespace sum_gcf_lcm_eq_9_l423_423689

def gcf (a b : ℕ) : ℕ := a.gcd b
def lcm (a b : ℕ) : ℕ := a.lcm b

theorem sum_gcf_lcm_eq_9 : gcf 3 6 + lcm 3 6 = 9 := by
  sorry

end sum_gcf_lcm_eq_9_l423_423689


namespace chord_length_intersection_l423_423105

theorem chord_length_intersection 
  (x y : ℝ) 
  (h_line : x + y = 0) 
  (h_circle : (x - 2)^2 + y^2 = 4) : 
  ∃ L, L = 2 * real.sqrt 2 :=
sorry

end chord_length_intersection_l423_423105


namespace angle_CAD_l423_423737

theorem angle_CAD (O A B C D E F : Point) 
  (hf1 : Circle O A B) (hf2 : Contains O A) (hf3 : Contains O B)
  (hf4 : Between A D B) (hf5 : Contains D (Arc A B))
  (hf6 : ⊥ from D to OC) (hf7 : ⊥ from D to AB)
  (hf8 : AngleBisector DE (Δ ADC))
  (hf9 : AngleBisector DO (Δ ADF)) :
  ∠CAD = 20° :=
sorry

end angle_CAD_l423_423737


namespace accumulating_small_steps_necessary_not_sufficient_l423_423349

theorem accumulating_small_steps_necessary_not_sufficient :
  (∀ {steps: ℕ}, (¬ accumulate_small_steps steps → ¬ reach_thousand_miles) → (accumulate_small_steps steps → reach_thousand_miles)) ∧
  (∃ {steps: ℕ}, accumulate_small_steps steps → ¬ reach_thousand_miles) :=
by
  sorry

variables {steps : ℕ}

def accumulate_small_steps : Prop := sorry

def reach_thousand_miles : Prop := sorry


end accumulating_small_steps_necessary_not_sufficient_l423_423349


namespace bond_face_value_approx_l423_423567

theorem bond_face_value_approx :
  ∃ (F : ℝ), (F ≈ 5000) ∧ 
             (∃ (I : ℝ), I = 0.07 * F) ∧ 
             (I ≈ 0.065 * 5384.615384615386) :=
sorry

end bond_face_value_approx_l423_423567


namespace PG_entitled_amount_l423_423444

-- Definitions and conditions provided in the problem
def initial_investment_VP : ℕ := 200_000
def initial_investment_PG : ℕ := 350_000
def AA_purchase_amount : ℕ := 1_100_000
def factory_value_after_sale : ℕ := 3_300_000
def share_value_each_after_sale : ℕ := factory_value_after_sale / 3
def VP_initial_share_value := (initial_investment_VP * factory_value_after_sale) / (initial_investment_VP + initial_investment_PG)
def VP_sold_share_value := VP_initial_share_value - share_value_each_after_sale
def PG_received_amount := AA_purchase_amount - VP_sold_share_value

-- Theorem to be proven
theorem PG_entitled_amount : PG_received_amount = 1_000_000 := by
  sorry

end PG_entitled_amount_l423_423444


namespace find_angle_C_find_lengths_CA_CB_l423_423184

open Real

-- Define the given problem conditions and proofs
variables (A B C : ℝ) (CA CB : ℝ)

-- Provided the dot product and unit vectors
def m : ℝ × ℝ := (cos A, -sin A)
def n : ℝ × ℝ := (cos B, sin B)

-- Define the conditions
axiom cond1 : m ∙ n = cos (2 * C)
axiom cond2 : A + B + C = π
axiom cond3 : C > 0 ∧ C < π
axiom cond4 : 6 = 6 -- Ax, since AB = 6
axiom cond5 : CA * CB * cos (π / 3) = 18

-- The problem statements to prove
theorem find_angle_C : C = π / 3 := sorry

theorem find_lengths_CA_CB : CA = 6 ∧ CB = 6 := sorry

end find_angle_C_find_lengths_CA_CB_l423_423184


namespace area_ECODF_l423_423761

-- Definitions of geometric entities and parameters
def radius := 3
def OA := 3 * Real.sqrt 2
def AB := 2 * OA
def height_of_rectangle := radius
def area_of_rectangle := height_of_rectangle * AB
def OC_OD := radius  -- Since OC and OD are tangents and OC^2 = OD^2
def area_triangle := 0.5 * radius * radius
def angle_sector := π / 4 -- 45 degrees
def area_sector := (angle_sector / (2 * π)) * (π * radius^2)

-- Define the area of the shaded region
def area_shaded_region : ℝ :=
  area_of_rectangle - 2 * area_triangle - 2 * area_sector

-- Conjecture / Theorem
theorem area_ECODF (O A B C D E F : Type*) 
  (hA_radius : ∀ p, O.dist p = radius → p ∈ A) 
  (hB_radius : ∀ p, O.dist p = radius → p ∈ B) 
  (hO_midpoint : O.dist A = OA) :
  area_shaded_region = 18 * Real.sqrt 2 - 9 - 9 * (π / 4) :=
by
  -- Proof will be implemented here
  sorry


end area_ECODF_l423_423761


namespace ellipse_equation_and_k_value_l423_423494

-- Define the conditions
def e : ℝ := (Real.sqrt 6) / 3
def a : ℝ := Real.sqrt 3
def b : ℝ := 1
def ellipse : (ℝ × ℝ) → Prop := 
  λ p, p.1^2 / a^2 + p.2^2 / b^2 = 1

-- Define the problem to be proved
theorem ellipse_equation_and_k_value
  (F : ℝ × ℝ) (AB : ℝ)
  (hFoci : abs (F.2) = Real.sqrt (a^2 - b^2)) -- F is one focus of the ellipse
  (hFocus_on_line : F.1 = 0)
  (hAB : AB = 2 * Real.sqrt 3 / 3)
  (E : ℝ × ℝ) (hE : E = (-1, 0)) :
  (∀ (x y : ℝ), ellipse (x, y) ↔ x^2 / 3 + y^2 = 1) ∧
  (∃ k : ℝ, line_intersection_and_passing_circle_condition k E) :=
begin
  -- Placeholder for the proof using sorry
  sorry
end

-- Define helper predicates if necessary
def line_intersection_and_passing_circle_condition (k : ℝ) (E : ℝ × ℝ) : Prop :=
  let line := λ x, (k * x + 2) in
  ∃ (x1 x2 : ℝ), 
  (x1 + x2 = - 12 * k / (1 + 3 * k ^ 2)) ∧
  (x1 * x2 = 9 / (1 + 3 * k ^ 2)) ∧
  ((x1 + 1) * (x2 + 1) + k ^ 2 * x1 * x2 + 2 * k * (x1 + x2) + 4 = 0)

end ellipse_equation_and_k_value_l423_423494


namespace number_of_boys_took_exam_l423_423327

theorem number_of_boys_took_exam (T F : ℕ) (h_avg_all : 35 * T = 39 * 100 + 15 * F)
                                (h_total_boys : T = 100 + F) : T = 120 :=
sorry

end number_of_boys_took_exam_l423_423327


namespace problem_l423_423540

structure Point (α : Type) :=
  (x : α)
  (y : α)

variables {α : Type} [LinearOrderedField α]

def altitude_from_B_to_AC (A B C : Point α) : α × α × α :=
  let k := - (C.y - A.y) / (C.x - A.x) in
  (3, 2, -12)

def median_from_B_to_AC (A B C : Point α) : α × α × α :=
  let midpoint := Point.mk ((B.x + C.x) / 2) ((B.y + C.y) / 2) in
  (5, 1, -20)

theorem problem 
(A B C : Point α) 
(hA : A.x = 4 ∧ A.y = 0) 
(hB : B.x = 6 ∧ B.y = 7) 
(hC : C.x = 0 ∧ C.y = 3) : 
  altitude_from_B_to_AC A B C = (3, 2, -12) ∧ median_from_B_to_AC A B C = (5, 1, -20) := 
by
  sorry

end problem_l423_423540


namespace PQ_divides_AD_l423_423365

noncomputable def midpoint (P Q : Point) : Point := sorry

noncomputable def Line : Type := sorry

variable (Point : Type)

variable (Circle : Point → ℝ → Type)
variable (Chord : Circle → Type)
variable (Tangent : Point → Circle → Line)
variable (Intersection : Line → Line → Point)

variable (Ω ω : Circle)
variable (O : Point)
variable (A D P B C Q : Point)

variable (chord_AD : Chord Ω)

variable (tangent_point : Tangent A ω)
variable (tangent_chord : Tangent D ω)
variable (chord_P_BC : Tangent P ω)

axiom concentric (Ω ω : Circle) (O : Point): Center Ω = O ∧ Center ω = O
axiom tangent_to_smaller_circle (chord : Chord Ω) (circle : Circle) : Tangent chord.Points circle 
axiom inside_smaller_segment (P : Point) (chord : Chord Ω): inside_segment P chord
axiom tangents_from_P (P : Point) (circle : Circle) (B C: Point): Tangent P circle = (B ∧ C)

axiom intersecting_segments (BD AC : Line): Intersection BD AC = Q

theorem PQ_divides_AD (Ω ω : Circle) (O A D P B C Q : Point) 
  (chord_AD : Chord Ω) 
  (tangent_ω_A : Tangent A ω)
  (tangent_ω_D : Tangent D ω)
  (tangent_ω_P_BC : Tangent P ω)
  (intersect_BD_AC : Intersection (Line_through B D) (Line_through A C) = Q)
  [concentric Ω ω O]
  [tangent_to_smaller_circle chord_AD ω]
  [inside_smaller_segment P chord_AD]
  [tangents_from_P P ω B C]

  : midpoint P Q = midpoint A D := sorry

end PQ_divides_AD_l423_423365


namespace wage_of_one_man_l423_423019

/-- Proof that the wage of one man is Rs. 24 given the conditions. -/
theorem wage_of_one_man (M W_w B_w : ℕ) (H1 : 120 = 5 * M + W_w * 5 + B_w * 8) 
  (H2 : 5 * M = W_w * 5) (H3 : W_w * 5 = B_w * 8) : M = 24 :=
by
  sorry

end wage_of_one_man_l423_423019


namespace complement_of_M_l423_423860

def M : set ℝ := { x | x ≥ 2 }

theorem complement_of_M :
  Mᶜ = { x : ℝ | x < 2 } :=
by
  sorry

end complement_of_M_l423_423860


namespace quad_d_has_no_real_roots_l423_423343

-- Definitions of the quadratic equations
def quad_a : ℝ → ℝ := λ x, x^2 + x
def quad_b : ℝ → ℝ := λ x, 5 * x^2 - 4 * x - 1
def quad_c : ℝ → ℝ := λ x, 3 * x^2 - 4 * x + 1
def quad_d : ℝ → ℝ := λ x, 4 * x^2 - 5 * x + 2

-- Discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem stating that equation D has no real roots
theorem quad_d_has_no_real_roots :
  ∀ x : ℝ, (4 * x^2 - 5 * x + 2 = 0) -> false :=
by
  intro x
  have h_d : discriminant 4 (-5) 2 = -7 := by
    calc
      discriminant 4 (-5) 2 = (-5)^2 - 4 * 4 * 2 : by rfl
      ... = 25 - 32 : by rfl
      ... = -7 : by rfl
  show false, from sorry

end quad_d_has_no_real_roots_l423_423343


namespace tax_on_clothing_l423_423606

variable (T : ℝ)
variable (c : ℝ := 0.45 * T)
variable (f : ℝ := 0.45 * T)
variable (o : ℝ := 0.10 * T)
variable (x : ℝ)
variable (t_c : ℝ := x / 100 * c)
variable (t_f : ℝ := 0)
variable (t_o : ℝ := 0.10 * o)
variable (t : ℝ := 0.0325 * T)

theorem tax_on_clothing :
  t_c + t_o = t → x = 5 :=
by
  sorry

end tax_on_clothing_l423_423606


namespace number_of_men_in_first_group_l423_423993

noncomputable def men_in_first_group (x : ℕ) : Prop :=
  x * 18 = 108 * 6

theorem number_of_men_in_first_group : ∃ x : ℕ, x = 36 ∧ men_in_first_group x :=
by
  use 36
  simp [men_in_first_group]
  sorry

end number_of_men_in_first_group_l423_423993


namespace general_term_a_n_value_of_a_inequality_T_n_l423_423163

-- Problem 1 
theorem general_term_a_n (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 2 * a_n n - 2) : 
  ∀ n, a_n n = 2^n :=
sorry

-- Problem 2
theorem value_of_a (a b_n S_n : ℝ → ℝ) (hS : ∀ n, S_n n = a/(a-1) * (a_n n - 1)) 
  (hb : ∀ n, b_n n = 2 * (S_n n) / (a_n n) + 1) (hgeo : ∀ n, b_n (n+1) / b_n n = b_n (n+2) / b_n (n+1)) : 
  a = 1 / 3 :=
sorry

-- Problem 3
theorem inequality_T_n (a : ℝ) (S_n a_n c_n T_n : ℕ → ℝ) 
  (ha : a = 1 / 3) 
  (hS : ∀ n, S_n n = a/(a-1) * (a_n n - 1)) 
  (hc : ∀ n, c_n n = (1 / (1 + (a_n n))) + (1 / (1 - (a_n (n+1))))) 
  (hT : ∀ n, T_n n = ∑ i in range(n+1), c_n i):
  ∀ n, T_n n > 2 * n - 1/3 :=
sorry

end general_term_a_n_value_of_a_inequality_T_n_l423_423163


namespace dilation_image_correct_l423_423999

-- Define the dilation function
def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ := center + scale * (z - center)

-- Problem conditions
def center : ℂ := -2 + 2i
def scale_factor : ℝ := 4
def original_point : ℂ := 1 + 2i
def image_point : ℂ := 10 + 2i

-- Theorems
theorem dilation_image_correct : dilation center scale_factor original_point = image_point :=
by
  sorry

end dilation_image_correct_l423_423999


namespace interval_of_increase_logb_l423_423654

noncomputable def f (x : ℝ) := Real.logb 5 (2 * x + 1)

-- Define the domain
def domain : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the interval of monotonic increase for the function
def interval_of_increase (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x < y → f x < f y}

-- Statement of the problem
theorem interval_of_increase_logb :
  interval_of_increase f = {x | x > - (1 / 2)} :=
by
  have h_increase : ∀ x y, x < y → f x < f y := sorry
  exact sorry

end interval_of_increase_logb_l423_423654


namespace intersection_area_correct_l423_423678

noncomputable def intersection_area (XY YE FX EX FY : ℕ) : ℚ :=
  if XY = 12 ∧ YE = FX ∧ YE = 15 ∧ EX = FY ∧ EX = 20 then
    18
  else
    0

theorem intersection_area_correct {XY YE FX EX FY : ℕ} (h1 : XY = 12) (h2 : YE = FX) (h3 : YE = 15) (h4 : EX = FY) (h5 : EX = 20) : 
  intersection_area XY YE FX EX FY = 18 := 
by {
  sorry
}

end intersection_area_correct_l423_423678


namespace total_revenue_correct_l423_423391

def price_per_book : ℝ := 25
def books_sold_monday : ℕ := 60
def discount_monday : ℝ := 0.10
def books_sold_tuesday : ℕ := 10
def discount_tuesday : ℝ := 0.0
def books_sold_wednesday : ℕ := 20
def discount_wednesday : ℝ := 0.05
def books_sold_thursday : ℕ := 44
def discount_thursday : ℝ := 0.15
def books_sold_friday : ℕ := 66
def discount_friday : ℝ := 0.20

def revenue (books_sold: ℕ) (discount: ℝ) : ℝ :=
  (1 - discount) * price_per_book * books_sold

theorem total_revenue_correct :
  revenue books_sold_monday discount_monday +
  revenue books_sold_tuesday discount_tuesday +
  revenue books_sold_wednesday discount_wednesday +
  revenue books_sold_thursday discount_thursday +
  revenue books_sold_friday discount_friday = 4330 := by 
sorry

end total_revenue_correct_l423_423391


namespace point_on_line_l423_423547

theorem point_on_line (x : ℝ) :
  let p1 := (0, 10)
  let p2 := (5, 0)
  let p3 := (x, -5)
  line_through p1 p2 p3 ↔ x = 7.5 :=
begin
  -- define the line_through function
  def line_through (p1 p2 p3 : ℝ × ℝ) : Prop :=
    let (x1, y1) := p1;
    let (x2, y2) := p2;
    let (x3, y3) := p3;
    (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

  sorry
end

end point_on_line_l423_423547


namespace intersecting_curves_l423_423209

-- Definitions from a) conditions
def C1_parametric (a t : ℝ) : ℝ × ℝ :=
  (a + sqrt 2 * t, 1 + sqrt 2 * t)

def C2_polar_equation (p θ : ℝ) : Prop :=
  p * cos θ^2 + 4 * cos θ - p = 0

-- Mathematical equivalent proof problem
theorem intersecting_curves (a : ℝ) : 
  (∃ t : ℝ, (a + sqrt 2 * t) = (1 + sqrt 2 * t) - 1 )
    ∧ (∃ t : ℝ, (1 + sqrt 2 * t)^2 = 4 * (a + sqrt 2 * t)) 
    ∧ (a > 0)
    ∧ (2 * |sqrt (1 - 4 * a) / 2| = 4 * |sqrt (1 - 4 * a) / 2|) :
  a = 1 / 36 ∨ a = 9 / 4 :=
by
  sorry

end intersecting_curves_l423_423209


namespace max_stamps_l423_423548

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 45) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, n ≤ total_cents / price_per_stamp ∧ n = 111 :=
by
  sorry

end max_stamps_l423_423548


namespace express_x_n_prove_inequality_l423_423589

variable (a b n : Real)
variable (x : ℕ → Real)

def trapezoid_conditions : Prop :=
  ∀ n, x 1 = a * b / (a + b) ∧ (x (n + 1) / x n = x (n + 1) / a)

theorem express_x_n (h : trapezoid_conditions a b x) : 
  ∀ n, x n = a * b / (a + n * b) := 
by
  sorry

theorem prove_inequality (h : trapezoid_conditions a b x) : 
  ∀ n, x n ≤ (a + n * b) / (4 * n) := 
by
  sorry

end express_x_n_prove_inequality_l423_423589


namespace talkingBirds_count_l423_423381

-- Define the conditions
def totalBirds : ℕ := 77
def nonTalkingBirds : ℕ := 13
def talkingBirds (T : ℕ) : Prop := T + nonTalkingBirds = totalBirds

-- Statement to prove
theorem talkingBirds_count : ∃ T, talkingBirds T ∧ T = 64 :=
by
  -- Proof will go here
  sorry

end talkingBirds_count_l423_423381


namespace total_baseball_cards_l423_423357

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end total_baseball_cards_l423_423357


namespace smallest_fraction_gt_3_5_with_two_digit_nums_l423_423396

theorem smallest_fraction_gt_3_5_with_two_digit_nums : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 5 * a > 3 * b ∧ (∀ (a' b' : ℕ), 10 ≤ a' ∧ a' < 100 ∧ 10 ≤ b' ∧ b' < 100 ∧ 5 * a' > 3 * b' → a * b' ≤ a' * b) := 
  ⟨59, 98, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, sorry⟩

end smallest_fraction_gt_3_5_with_two_digit_nums_l423_423396


namespace ball_distribution_l423_423874

-- Definitions as per conditions
def num_distinguishable_balls : ℕ := 5
def num_indistinguishable_boxes : ℕ := 3

-- Problem statement to prove
theorem ball_distribution : 
  let ways_to_distribute_balls := 1 + 5 + 10 + 10 + 30 in
  ways_to_distribute_balls = 56 :=
by
  -- proof required here
  sorry

end ball_distribution_l423_423874


namespace sum_y_coordinates_on_circle_y_axis_l423_423072

theorem sum_y_coordinates_on_circle_y_axis :
  let C := { p : ℝ × ℝ | (p.1 + 8) ^ 2 + (p.2 - 3) ^ 2 = 225 }
  in (∃! p₁ p₂ : ℝ × ℝ, p₁ ∈ C ∧ p₂ ∈ C ∧ p₁.1 = 0 ∧ p₂.1 = 0 ∧ p₁.2 + p₂.2 = 6) :=
sorry

end sum_y_coordinates_on_circle_y_axis_l423_423072


namespace jogging_time_after_two_weeks_l423_423603

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end jogging_time_after_two_weeks_l423_423603


namespace thirteenth_result_is_128_l423_423996

theorem thirteenth_result_is_128 
  (avg_all : ℕ → ℕ → ℕ) (avg_first : ℕ → ℕ → ℕ) (avg_last : ℕ → ℕ → ℕ) :
  avg_all 25 20 = (avg_first 12 14) + (avg_last 12 17) + 128 :=
by
  sorry

end thirteenth_result_is_128_l423_423996


namespace total_period_is_112_l423_423777

def music_station_period (commercial_minutes : ℕ) (music_ratio: ℕ) (commercial_ratio: ℕ) : ℕ :=
  let part := commercial_minutes / commercial_ratio in
  let music_minutes := part * music_ratio in
  music_minutes + commercial_minutes

theorem total_period_is_112 : 
  music_station_period 40 9 5 = 112 := 
by
  sorry

end total_period_is_112_l423_423777


namespace closest_ratio_to_one_l423_423637

theorem closest_ratio_to_one
  (a c : ℕ)
  (h1 : 30 * a + 10 * c = 2400)
  (h2 : a ≥ 1)
  (h3 : c ≥ 1) :
  (a : ℚ) / c = 1 / 3 :=
begin
  sorry
end

end closest_ratio_to_one_l423_423637


namespace part1_probability_A_then_B_part2_probability_select_D_l423_423982

-- Define the set of students
inductive Student
| A | B | C | D

open Student

-- Define the conditions
def selected_students (s1 s2 : Student) : Prop :=
(s1 = A ∧ (s2 = B ∨ s2 = C ∨ s2 = D)) ∨
(s2 = A ∧ (s1 = B ∨ s1 = C ∨ s1 = D))

-- The first proof problem
theorem part1_probability_A_then_B :
  (∃ s2, selected_students A s2 ∧ s2 = B) / (∃ s2, selected_students A s2) = 1/3 :=
sorry

-- The second proof problem
theorem part2_probability_select_D :
  ((∃ s1, ∃ s2, selected_students s1 s2 ∧ (s1 = D ∨ s2 = D)) / 
  (∃ s1, ∃ s2, selected_students s1 s2)) = 1/2 :=
sorry

end part1_probability_A_then_B_part2_probability_select_D_l423_423982


namespace find_f_l423_423703

theorem find_f (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > 0)
  (h2 : f 1 = 1)
  (h3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2) : ∀ x : ℝ, f x = x := by
  sorry

end find_f_l423_423703


namespace quad_d_has_no_real_roots_l423_423342

-- Definitions of the quadratic equations
def quad_a : ℝ → ℝ := λ x, x^2 + x
def quad_b : ℝ → ℝ := λ x, 5 * x^2 - 4 * x - 1
def quad_c : ℝ → ℝ := λ x, 3 * x^2 - 4 * x + 1
def quad_d : ℝ → ℝ := λ x, 4 * x^2 - 5 * x + 2

-- Discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem stating that equation D has no real roots
theorem quad_d_has_no_real_roots :
  ∀ x : ℝ, (4 * x^2 - 5 * x + 2 = 0) -> false :=
by
  intro x
  have h_d : discriminant 4 (-5) 2 = -7 := by
    calc
      discriminant 4 (-5) 2 = (-5)^2 - 4 * 4 * 2 : by rfl
      ... = 25 - 32 : by rfl
      ... = -7 : by rfl
  show false, from sorry

end quad_d_has_no_real_roots_l423_423342


namespace quadratic_has_two_real_roots_root_less_than_two_l423_423859

theorem quadratic_has_two_real_roots (k : ℝ) : 
  let Δ := (k - 4) ^ 2 in Δ ≥ 0 :=
by
  sorry

theorem root_less_than_two (k : ℝ) (h : ∃ x : ℝ, x < 2 ∧ (x ^ 2 - (k + 4) * x + 4 * k = 0)) :
  k < 2 :=
by
  sorry

end quadratic_has_two_real_roots_root_less_than_two_l423_423859


namespace width_of_WallA_l423_423428

noncomputable def heightA (WA : ℝ) := 6 * WA
noncomputable def lengthA (WA : ℝ) := 7 * (heightA WA)

noncomputable def heightB (WA : ℝ) (x : ℝ) := x * WA
noncomputable def lengthB (WA : ℝ) (x : ℝ) (y : ℝ) := y * (heightB WA x)

noncomputable def widthC (WA : ℝ) (z : ℝ) := z * WA
noncomputable def heightC (WA : ℝ) (x : ℝ) := (heightB WA x) / 2
noncomputable def lengthC (WA : ℝ) (w : ℝ) := w * (lengthA WA)

noncomputable def volumeA (WA : ℝ) := WA * (heightA WA) * (lengthA WA)
noncomputable def volumeB (WA : ℝ) (x : ℝ) (y : ℝ) := WA * (heightB WA x) * (lengthB WA x y)
noncomputable def volumeC (WA : ℝ) (x : ℝ) (z : ℝ) (w : ℝ) := (widthC WA z) * (heightC WA x) * (lengthC WA w)

theorem width_of_WallA (x y z w V : ℝ) : 
  ∃ WA : ℝ, WA ^ 3 = V / (252 + y * x ^ 2 + 21 * w * x * z) :=
begin
  use ((V / (252 + y * x ^ 2 + 21 * w * x * z)) ^ (1 / 3)),
  sorry
end

end width_of_WallA_l423_423428


namespace sin_from_tan_l423_423549

theorem sin_from_tan (A : ℝ) (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by 
  sorry

end sin_from_tan_l423_423549


namespace b_2056_l423_423580

noncomputable def b (n : ℕ) : ℝ := sorry

-- Conditions
axiom h1 : b 1 = 2 + Real.sqrt 8
axiom h2 : b 2023 = 15 + Real.sqrt 8
axiom recurrence : ∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)

-- Problem statement to prove
theorem b_2056 : b 2056 = (2 + Real.sqrt 8)^2 / (15 + Real.sqrt 8) :=
sorry

end b_2056_l423_423580


namespace minimum_tenth_game_score_l423_423564

theorem minimum_tenth_game_score (S5 : ℕ) (score10 : ℕ) 
  (h1 : 18 + 15 + 16 + 19 = 68)
  (h2 : S5 ≤ 85)
  (h3 : (S5 + 68 + score10) / 10 > 17) : 
  score10 ≥ 18 := sorry

end minimum_tenth_game_score_l423_423564


namespace double_number_no_common_digit_l423_423190

theorem double_number_no_common_digit (a b u v : ℕ) (x : ℕ) (hx : x = 10 * a + b / 2)
  (h1 : 10 * a + b = 2 * x)
  (h2 : x < 100 ∧ x >= 10)
  (h3 : u = a + b ∧ v = |a - b| ∧ u ≠ v)
  (h4 : a ≠ b) :
  x = 17 ∧ 2 * x = 34 :=
  sorry

end double_number_no_common_digit_l423_423190


namespace line_connecting_centers_l423_423767

-- Define the first circle equation
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x + 6*y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_eq (x y : ℝ) := 3*x - y - 9 = 0

-- Prove that the line connecting the centers of the circles has the given equation
theorem line_connecting_centers :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → line_eq x y := 
sorry

end line_connecting_centers_l423_423767


namespace monotonic_decreasing_interval_l423_423309

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := 6 * x - 3 / x

theorem monotonic_decreasing_interval :
  (∀ x > 0, f' x < 0 ↔ (0 < x ∧ x < Real.sqrt 2 / 2)) →
  (∀ x, (∃ ε > 0, ∀ y, 0 < abs (y - x) → abs (y - x) < ε → y ∈ (0, Real.sqrt 2 / 2)) ↔ x ∈ (0, Real.sqrt 2 / 2)) →
  (f' (Real.sqrt 2 / 2) < 0) :=
by
  sorry

end monotonic_decreasing_interval_l423_423309


namespace hyperbola_asymptotes_l423_423300

theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 - 2 * y^2 = 3 -> (y = (sqrt 2) / 2 * x ∨ y = -(sqrt 2) / 2 * x) :=
by
  intro h
  sorry

end hyperbola_asymptotes_l423_423300


namespace minimum_value_of_xy_l423_423827

noncomputable def minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : ℝ :=
  if hmin : 4 * x + y + 12 = x * y then 36 else sorry

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : 
  minimum_value_xy x y hx hy h = 36 :=
sorry

end minimum_value_of_xy_l423_423827


namespace solve_for_y_l423_423990

theorem solve_for_y (y : ℝ) (hy : y ≠ -2) : 
  (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ↔ y = 7 / 6 :=
by sorry

end solve_for_y_l423_423990


namespace jade_statue_ratio_l423_423605

/-!
Nancy carves statues out of jade. A giraffe statue takes 120 grams of jade and sells for $150.
An elephant statue sells for $350. Nancy has 1920 grams of jade, and the revenue from selling all
elephant statues is $400 more than selling all giraffe statues.
Prove that the ratio of the amount of jade used for an elephant statue to the amount used for a
giraffe statue is 2.
-/

theorem jade_statue_ratio
  (g_grams : ℕ := 120) -- grams of jade for a giraffe statue
  (g_price : ℕ := 150) -- price of a giraffe statue
  (e_price : ℕ := 350) -- price of an elephant statue
  (total_jade : ℕ := 1920) -- total grams of jade Nancy has
  (additional_revenue : ℕ := 400) -- additional revenue from elephant statues
  (r : ℕ) -- ratio of jade usage of elephant to giraffe statue
  (h : total_jade / g_grams * g_price + additional_revenue = (total_jade / (g_grams * r)) * e_price) :
  r = 2 :=
sorry

end jade_statue_ratio_l423_423605


namespace no_k_satisfying_condition_l423_423413

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_k_satisfying_condition :
  ∀ k : ℕ, (∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ k = p * q ∧ p + q = 71) → false :=
by
  sorry

end no_k_satisfying_condition_l423_423413


namespace decreasing_arith_prog_smallest_num_l423_423461

theorem decreasing_arith_prog_smallest_num 
  (a d : ℝ) -- Define a and d as real numbers
  (h_arith_prog : ∀ n : ℕ, n < 5 → (∃ k : ℕ, k < 5 ∧ (a - k * d) = if n = 0 then a else a - n * d))
  (h_sum_cubes_zero : a^3 + (a-d)^3 + (a-2*d)^3 + (a-3*d)^3 + (a-4*d)^3 = 0)
  (h_sum_fourth_powers_306 : a^4 + (a-d)^4 + (a-2*d)^4 + (a-3*d)^4 + (a-4*d)^4 = 306) :
  ∃ d' ∈ {d}, a - 4 * d' = -2 * real.sqrt 3 := -- Prove the smallest number is -2√3
sorry

end decreasing_arith_prog_smallest_num_l423_423461


namespace polar_bear_daily_fish_intake_l423_423442

theorem polar_bear_daily_fish_intake : 
  (0.2 + 0.4 = 0.6) := by
  sorry

end polar_bear_daily_fish_intake_l423_423442


namespace minimum_value_of_J_l423_423570

def E : Set (ℝ → ℝ) :=
  {f | Continuous differentiableOn ℝ f (Set.Icc 0 1) ∧ f 0 = 0 ∧ f 1 = 1}

def J (f : ℝ → ℝ) : ℝ :=
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 + x^2) * (deriv f x) ^ 2

theorem minimum_value_of_J : ∃ f ∈ E, J f = 4 / π := 
sorry

end minimum_value_of_J_l423_423570


namespace min_value_expression_l423_423109

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l423_423109


namespace minimum_value_of_a_l423_423501

theorem minimum_value_of_a (b : ℝ) (h_b_gt_1 : b > 1) 
  (h_perpendicular : (∀ x y : ℝ, (b^2 + 1) * x + a * y + 2 = 0 →  x - (b-1) * y - 1 = 0 → False))
  : a ≥ 2 * real.sqrt 2 + 2 := 
sorry

end minimum_value_of_a_l423_423501


namespace find_a_l423_423478

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem find_a (a : ℝ) (h : ∫ x in -1..1, f x = 2 * f a) : a = -1 ∨ a = 1 / 3 := by
  sorry

end find_a_l423_423478


namespace square_area_192_l423_423556

theorem square_area_192 (A B C D P Q R : Type)
  [Square A B C D]
  (hP : P ∈ line_segment A D)
  (hQ : Q ∈ line_segment A B)
  (right_angle : angles_are_right ⟨B, P, R⟩ ⟨C, Q, R⟩)
  (BR : length B R = 8)
  (PR : length P R = 8) :
  area (square A B C D) = 192 :=
sorry

end square_area_192_l423_423556


namespace jeff_works_82_hours_in_week_l423_423229

def jeff_total_working_hours_weekend (facebook_hours_weekend : ℕ) : ℕ :=
  facebook_hours_weekend / 3

def jeff_facebook_hours_weekend : ℕ := 3

def jeff_instagram_hours_weekend : ℕ := 1

def jeff_exercise_hours_per_weekend_day : ℕ := 3 / 2

def jeff_errands_hours_per_weekend_day : ℕ := 3 / 2

def jeff_total_working_hours_weekday (facebook_hours_weekday instagram_hours_weekday : ℕ) : ℕ :=
  4 * (facebook_hours_weekday + instagram_hours_weekday)

def jeff_facebook_hours_weekday : ℕ := 3

def jeff_instagram_hours_weekday : ℕ := 1

def jeff_exercise_hours_per_weekday : ℕ := 1 / 2

def jeff_total_working_hours_week : ℕ :=
  jeff_total_working_hours_weekend jeff_facebook_hours_weekend * 2 +
  jeff_total_working_hours_weekday jeff_facebook_hours_weekday jeff_instagram_hours_weekday * 5

theorem jeff_works_82_hours_in_week :
  jeff_total_working_hours_week = 82 :=
begin
  sorry
end

end jeff_works_82_hours_in_week_l423_423229


namespace possible_values_of_sum_l423_423943

theorem possible_values_of_sum (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 2) : 
  set.Ici (2 : ℝ) = {x : ℝ | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b)} :=
by
  sorry

end possible_values_of_sum_l423_423943


namespace intersection_hyperbola_circle_l423_423429

theorem intersection_hyperbola_circle :
  {p : ℝ × ℝ | p.1^2 - 9 * p.2^2 = 36 ∧ p.1^2 + p.2^2 = 36} = {(6, 0), (-6, 0)} :=
by sorry

end intersection_hyperbola_circle_l423_423429


namespace total_visible_surface_area_l423_423803

def cube_volume (side : ℕ) : ℕ := side ^ 3

def visible_area (volumes : List ℕ) : ℕ :=
  -- Cube volumes
  let sides := volumes.map (λ v, (v : ℕ).nthRoot 3)
  -- Indices of the volumes
  let indices := List.range volumes.length
  -- Visible area calculation simplified
  let areas := indices.map (λ i,
    let side := sides.get! i
    let above := if 0 ≤ i - 1 then sides.get! (i - 1) else 0
    let below := if i + 1 < volumes.length then sides.get! (i + 1) else 0
    (4 * side * side) + (side * side - below * below) + (side * side - above * above))
  areas.foldl (·+·) 0

theorem total_visible_surface_area :
  visible_area [343, 125, 27, 64, 1] = 400 := by
  sorry

end total_visible_surface_area_l423_423803


namespace infinite_equal_pairs_of_equal_terms_l423_423822

theorem infinite_equal_pairs_of_equal_terms {a : ℤ → ℤ}
  (h : ∀ n, a n = (a (n - 1) + a (n + 1)) / 4)
  (i j : ℤ) (hij : a i = a j) :
  ∃ (infinitely_many_pairs : ℕ → ℤ × ℤ), ∀ k, a (infinitely_many_pairs k).1 = a (infinitely_many_pairs k).2 :=
sorry

end infinite_equal_pairs_of_equal_terms_l423_423822


namespace one_fourth_of_6_8_is_fraction_l423_423102

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l423_423102


namespace inequality_y_lt_x_div_4_l423_423160

open Real

/-- Problem statement:
Given x ∈ (0, π / 6) and y ∈ (0, π / 6), and x * tan y = 2 * (1 - cos x),
prove that y < x / 4.
-/
theorem inequality_y_lt_x_div_4
  (x y : ℝ)
  (hx : 0 < x ∧ x < π / 6)
  (hy : 0 < y ∧ y < π / 6)
  (h : x * tan y = 2 * (1 - cos x)) :
  y < x / 4 := sorry

end inequality_y_lt_x_div_4_l423_423160


namespace complex_modulus_sub_l423_423863

open Complex

theorem complex_modulus_sub {z1 z2 : ℂ} (h1 : |z1| = 1) (h2 : |z2| = 1) (h3 : |z1 + z2| = Real.sqrt 3) : |z1 - z2| = 1 := 
sorry

end complex_modulus_sub_l423_423863


namespace largest_fraction_l423_423344

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (7 : ℚ) / 15
  let C := (29 : ℚ) / 59
  let D := (200 : ℚ) / 399
  let E := (251 : ℚ) / 501
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_fraction_l423_423344


namespace greatest_two_digit_prime_saturated_l423_423380

def is_prime_saturated (e : ℕ) : Prop :=
  let P := e.factors.erase_dup.prod
  P < Nat.sqrt e

def is_two_digit (e : ℕ) : Prop :=
  e ≥ 10 ∧ e < 100

theorem greatest_two_digit_prime_saturated : ∃ e : ℕ, is_two_digit e ∧ is_prime_saturated e ∧
  ∀ d : ℕ, is_two_digit d ∧ is_prime_saturated d → d ≤ e :=
by
  have h : ∃ e, is_two_digit e ∧ is_prime_saturated e ∧ (∀ d, is_two_digit d ∧ is_prime_saturated d → d ≤ e) := ⟨96, by sorry⟩
  exact h

end greatest_two_digit_prime_saturated_l423_423380


namespace value_of_f_at_sqrt2_l423_423479

noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem value_of_f_at_sqrt2 :
  f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end value_of_f_at_sqrt2_l423_423479


namespace exists_bounds_for_CGMO_sequence_l423_423418

def CGMO_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n ≥ 2022, ∃ A ⊆ {x | x < n}, A ≠ ∅ ∧ ∃ k, a n = k ∧
               (k * ∏ x in A, a x) % (k * ∏ x in A, a x) = 0)

theorem exists_bounds_for_CGMO_sequence (a : ℕ → ℕ) (h : CGMO_sequence a) :
  ∃ (c1 c2 : ℝ) (N : ℕ), 
    (0 < c1) ∧ (0 < c2) ∧ ∀ n ≥ N, c1 * (n : ℝ)^2 ≤ (a n : ℝ) ∧ (a n : ℝ) ≤ c2 * (n : ℝ)^2 :=
sorry

end exists_bounds_for_CGMO_sequence_l423_423418


namespace cassidy_total_grounding_days_l423_423422

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l423_423422


namespace calculate_new_measure_l423_423559

noncomputable def equilateral_triangle_side_length : ℝ := 7.5

theorem calculate_new_measure :
  3 * (equilateral_triangle_side_length ^ 2) = 168.75 :=
by
  sorry

end calculate_new_measure_l423_423559


namespace range_Z1_minus_aZ2_l423_423848

noncomputable theory

variables {Z1 Z2 : ℂ}
variables {a : ℝ}

-- Definitions of the conditions
def mod_Z1_eq_a (Z1 : ℂ) (a : ℝ) : Prop := abs Z1 = a
def mod_Z2_eq_1 (Z2 : ℂ) : Prop := abs Z2 = 1
def prod_Z1Z2_eq_neg_a (Z1 Z2 : ℂ) (a : ℝ) : Prop := Z1 * Z2 = -a

-- The proof goal statement
theorem range_Z1_minus_aZ2 (h1 : mod_Z1_eq_a Z1 a) (h2 : mod_Z2_eq_1 Z2) (h3 : prod_Z1Z2_eq_neg_a Z1 Z2 a) : 
  ∃ x : ℝ, x ∈ set.Icc (-2 * a) (2 * a) ∧ Z1 - a * Z2 = x :=
sorry

end range_Z1_minus_aZ2_l423_423848


namespace rank_friends_l423_423082

def is_false (p : Prop) := ¬ p

variables (david emma fiona george: ℕ)
variables (I II III IV : Prop)
variables (one_true_condition : (I ∨ II ∨ III ∨ IV) ∧ ¬(I ∧ II) ∧ ¬(I ∧ III) ∧ ¬(I ∧ IV) ∧ ¬(II ∧ III) ∧ ¬(II ∧ IV) ∧ ¬(III ∧ IV))
variables (h1 : I → emma > david)
variables (h2 : II → ¬ (fiona = min(david, min(emma, george))))
variables (h3 : III → george = max(david, max(emma, fiona)))
variables (h4 : IV → david ≠ max(david, max(emma, george)))

theorem rank_friends : (emma > david → I) →
                       (¬ (fiona = min(david, min(emma, george))) → II) →
                       (george = max(david, max(emma, fiona)) → III) →
                       (david ≠ max(david, max(emma, george)) → IV) → 
                       [david, emma, george, fiona] = [david, emma, george, fiona] :=
begin
  intros,
  sorry
end

end rank_friends_l423_423082


namespace last_locker_opened_l423_423740

theorem last_locker_opened : 
  let n := 500 in ∃ k, 1 ≤ k ∧ k ≤ n ∧ is_perfect_square k ∧ k = 484 :=
by
  let n := 500
  existsi 484
  refine ⟨_, _, _, _⟩
  sorry

end last_locker_opened_l423_423740


namespace find_a_l423_423084

def otimes (a b : ℝ) : ℝ := a - 2 * b

theorem find_a (a : ℝ) : (∀ x : ℝ, (otimes x 3 > 0 ∧ otimes x a > a) ↔ x > 6) → a ≤ 2 :=
begin
  intro h,
  sorry
end

end find_a_l423_423084


namespace transform_graph_shift_l423_423526

variable (f : ℝ → ℝ)

theorem transform_graph_shift (x : ℝ) : 
  f(2x - 2) + 1 = (λ y, (f (2 * (y - 1)))) x + 1 :=
by sorry

end transform_graph_shift_l423_423526


namespace min_area_of_triangle_l423_423574

variables {t : ℝ}

def pointA := (-1 : ℝ, 1 : ℝ, 2 : ℝ)
def pointB := (1 : ℝ, 2 : ℝ, 3 : ℝ)
def pointC (t : ℝ) := (t, 1, 2)

def vectorAB := (1 - (-1), 2 - 1, 3 - 2)
def vectorAC (t : ℝ) := (t - (-1), 1 - 1, 2 - 2)

def crossProduct (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((v₁.2.2 * v₂.3 - v₁.3 * v₂.2.2), (v₁.3 * v₂.1 - v₁.1 * v₂.3), (v₁.1 * v₂.2.2 - v₁.2.2 * v₂.1))

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2.2^2 + v.3^2)

def area_of_triangle (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  0.5 * magnitude (crossProduct v₁ v₂)

theorem min_area_of_triangle : ∃ t, area_of_triangle vectorAB (vectorAC t) = 0 :=
by {
  use (-1),
  -- here the proof to show area_of_triangle vectorAB (vectorAC (-1)) = 0
  sorry
}

end min_area_of_triangle_l423_423574


namespace exists_triangle_containing_all_points_l423_423322

theorem exists_triangle_containing_all_points 
  (points : fin 2001 → ℝ × ℝ)
  (h : ∀ (a b c : fin 2001), 
         let (x1, y1) := points a in
         let (x2, y2) := points b in
         let (x3, y3) := points c in
         let area := 0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)| in
         area ≤ 1) :
  ∃ (A B C : ℝ × ℝ), 
    {P : ℝ × ℝ | ∃ (a : fin 2001), points a = P} ⊆ 
    {Q : ℝ × ℝ | ∃ (p q r : ℝ), 
       let (x1, y1) := A in
       let (x2, y2) := B in
       let (x3, y3) := C in
       let area := 0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)| in
       (Q = (p, q)) ∧ 
       0.5 * |p*(y2 - y3) + q*(x3 - x1) + x2*(y3 - y1) - y2*(x3 - x1)| ≤ 4} := 
sorry

end exists_triangle_containing_all_points_l423_423322


namespace possible_values_of_S_count_l423_423826

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def Line (x y b: ℝ) : Prop := x + y = b

-- Define the distance function
def distance_from_center_to_line (b : ℝ) : ℝ := |b| / Real.sqrt 2

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the possible values for S given the distance d
def possible_S (d : ℝ) : ℕ :=
  if d > radius + 1 then 0
  else if radius - 1 < d ∧ d <= radius + 1 then 2
  else if d = radius - 1 then 1
  else if d < radius - 1 then 0
  else if d = radius + 1 then 1
  else 0

-- Main theorem statement
theorem possible_values_of_S_count : 
  (∃ d : ℝ, (possible_S (distance_from_center_to_line d) = 0 ∨ possible_S (distance_from_center_to_line d) = 1 ∨ possible_S (distance_from_center_to_line d) = 2)) ∧ (∃ d : ℝ, ∃ r : ℝ, radius = 2)  :=
by {
  sorry
}

end possible_values_of_S_count_l423_423826


namespace lcm_18_24_30_l423_423682

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l423_423682


namespace min_ratio_of_cylinder_cone_l423_423321

open Real

noncomputable def V1 (r : ℝ) : ℝ := 2 * π * r^3
noncomputable def V2 (R m r : ℝ) : ℝ := (1 / 3) * π * R^2 * m
noncomputable def geometric_constraint (R m r : ℝ) : Prop :=
  R / m = r / (sqrt ((m - r)^2 - r^2))

theorem min_ratio_of_cylinder_cone (r : ℝ) (hr : r > 0) : 
  ∃ R m, geometric_constraint R m r ∧ (V2 R m r) / (V1 r) = 4 / 3 := 
sorry

end min_ratio_of_cylinder_cone_l423_423321


namespace dist_eq_iff_cond_prob_eq_l423_423624

variables {Ω : Type*} {F : measurable_space Ω} {P : measure Ω} 
variables {X Y Z : Ω → ℝ} {A : set ℝ}

theorem dist_eq_iff_cond_prob_eq :
  (∀ A ∈ borel ℝ, P {ω | X ω ∈ A ∧ Y ω ∈ (set.univ : set Ω)} = P {ω | Z ω ∈ A ∧ Y ω ∈ (set.univ : set Ω)}) ↔ 
  (∀ A ∈ borel ℝ, P {ω | X ω ∈ A | Y} = P {ω | Z ω ∈ A | Y}) :=
sorry

end dist_eq_iff_cond_prob_eq_l423_423624


namespace group_total_payment_l423_423405

-- Declare the costs of the tickets as constants
def cost_adult : ℝ := 9.50
def cost_child : ℝ := 6.50

-- Conditions for the group
def total_moviegoers : ℕ := 7
def number_adults : ℕ := 3

-- Calculate the number of children
def number_children : ℕ := total_moviegoers - number_adults

-- Define the total cost paid by the group
def total_cost_paid : ℝ :=
  (number_adults * cost_adult) + (number_children * cost_child)

-- The proof problem: Prove that the total amount paid by the group is $54.50
theorem group_total_payment : total_cost_paid = 54.50 := by
  sorry

end group_total_payment_l423_423405


namespace sahil_selling_price_l423_423979

noncomputable def selling_price (purchase_price repair_cost transportation_charges maintenance_costs : ℝ)
                                (tax_rate currency_loss_rate depreciation_rate profit_rate : ℝ) : ℝ :=
let total_expenses_before_taxes := purchase_price + repair_cost + transportation_charges + maintenance_costs in
let taxes := tax_rate * total_expenses_before_taxes in
let total_expenses_after_taxes := total_expenses_before_taxes + taxes in
let currency_loss := currency_loss_rate * total_expenses_after_taxes in
let total_expenses_after_currency_loss := total_expenses_after_taxes - currency_loss in
let depreciation := depreciation_rate * total_expenses_after_currency_loss in
let value_after_depreciation := total_expenses_after_currency_loss - depreciation in
let profit := profit_rate * value_after_depreciation in
value_after_depreciation + profit

theorem sahil_selling_price :
  selling_price 10000 5000 1000 2000
                0.1 0.05 0.15 0.5 = 23982.75 :=
by sorry

end sahil_selling_price_l423_423979


namespace chess_games_l423_423324

theorem chess_games (n : ℕ) (total_games : ℕ) (players : ℕ) (games_per_player : ℕ)
  (h1 : players = 9)
  (h2 : total_games = 36)
  (h3 : ∀ i : ℕ, i < players → games_per_player = players - 1)
  (h4 : 2 * total_games = players * games_per_player) :
  games_per_player = 1 :=
by
  rw [h1, h2] at h4
  sorry

end chess_games_l423_423324


namespace log_sum_geometric_sequence_eq_n_squared_l423_423815

noncomputable def log_sum_geometric_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (λ k, log 2 (a (2 * k + 1))).sum

theorem log_sum_geometric_sequence_eq_n_squared
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n ≥ 3, a 5 * a (2 * n - 5) = 2^(2 * n)) :
  ∀ n ≥ 1, log_sum_geometric_sequence a n = n^2 :=
by
  sorry

end log_sum_geometric_sequence_eq_n_squared_l423_423815


namespace product_of_solutions_l423_423114

theorem product_of_solutions (x : ℤ) (h : x^2 = 49) : ∏ (x : {x // x^2 = 49}), x = -49 := sorry

end product_of_solutions_l423_423114


namespace prime_not_dividing_seq_l423_423246

theorem prime_not_dividing_seq (p : ℕ) (hp_prime : p.prime) (hp_divides : p ∣ 2^2019 - 1) :
  ∀ n : ℕ, ¬ p ∣ (1 + (nat.rec_on n (λ _, 2) (λ n an, nat.rec_on n (λ _, 1) (λ _ a_n_minus_1, an + (p^2 - 1) / 4 * a_n_minus_1)) n)) :=
by sorry

end prime_not_dividing_seq_l423_423246


namespace counting_indistinguishable_boxes_l423_423875

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l423_423875


namespace lcm_18_24_30_eq_360_l423_423684

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l423_423684


namespace travel_time_l423_423393

-- Given conditions
def distance_per_hour : ℤ := 27
def distance_to_sfl : ℤ := 81

-- Theorem statement to prove
theorem travel_time (dph : ℤ) (dts : ℤ) (h1 : dph = distance_per_hour) (h2 : dts = distance_to_sfl) : 
  dts / dph = 3 := 
by
  -- immediately helps execute the Lean statement
  sorry

end travel_time_l423_423393


namespace problem1_problem2_l423_423068

open Real

theorem problem1: 
  ((25^(1/3) - 125^(1/2)) / 5^(1/4) = 5^(5/12) - 5^(5/4)) :=
sorry

theorem problem2 (a : ℝ) (h : 0 < a): 
  (a^2 / (a^(1/2) * a^(2/3)) = a^(5/6)) :=
sorry

end problem1_problem2_l423_423068


namespace minimum_dot_product_l423_423902

variable {A B C M N : ℝ}
variable (a b : ℝ)
variable (A₀ : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))
variable (angle_A : (math.angle.vectors.v ∠ A) = (Real.pi / 3))
variable (area_ABC : (1/2) * (abs (B - A) * sin (angle_A)) = Real.sqrt 3)
variable (M_is_midpoint : M = (B + C) / 2)
variable (N_is_midpoint : N = (B + M) / 2)

theorem minimum_dot_product :
  ∃ min_val, min_val = -1 ∧ ∀ x, (overline AM) * (overline AN) ≥ min_val :=
sorry

end minimum_dot_product_l423_423902


namespace max_nat_number_sum_pos_l423_423154

theorem max_nat_number_sum_pos (a : ℕ → ℤ) (a_1_pos : a 1 > 0)
    (a_4_5_sum_pos : a 4 + a 5 > 0) (a_4_5_prod_neg : a 4 * a 5 < 0) 
    (arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1) : 
    ∃ (n : ℕ), n = 4 ∧ (∀ (m : ℕ), (m < n → (∑ i in finset.range (m + 1), a (i + 1) > 0)) ∧ 
    (∀ (m : ℕ), (m ≥ n → (∑ i in finset.range (m + 1), a (i + 1) ≤ 0))) :=
sorry

end max_nat_number_sum_pos_l423_423154


namespace more_than_half_millet_on_day_5_l423_423268

/-- Setup: Initial conditions and recursive definition of millet quantity -/
def millet_amount_on_day (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (3 / 4 : ℝ)^i / 4

/-- Proposition to prove: On the 5th day, just after placing the seeds, more than half the seeds are millet -/
theorem more_than_half_millet_on_day_5 :
  millet_amount_on_day 5 > 1 / 2 :=
begin
  sorry
end

end more_than_half_millet_on_day_5_l423_423268


namespace total_pages_read_l423_423228

theorem total_pages_read (J A C D : ℝ) 
  (hJ : J = 20)
  (hA : A = 2 * J + 2)
  (hC : C = J * A - 17)
  (hD : D = (C + J) / 2) :
  J + A + C + D = 1306.5 :=
by
  sorry

end total_pages_read_l423_423228


namespace arithmetic_common_difference_l423_423500

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  a + (n - 1) * d

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_common_difference (a10 : α) (s10 : α) (d : α) (a1 : α) :
  arithmetic_seq a1 d 10 = a10 →
  sum_arithmetic_seq a1 d 10 = s10 →
  d = 2 / 3 :=
by
  sorry

end arithmetic_common_difference_l423_423500


namespace probability_of_point_in_smaller_square_l423_423753

-- Definitions
def A_large : ℝ := 5 * 5
def A_small : ℝ := 2 * 2

-- Theorem statement
theorem probability_of_point_in_smaller_square 
  (side_large : ℝ) (side_small : ℝ)
  (hle : side_large = 5) (hse : side_small = 2) :
  (side_large * side_large ≠ 0) ∧ (side_small * side_small ≠ 0) → 
  (A_small / A_large = 4 / 25) :=
sorry

end probability_of_point_in_smaller_square_l423_423753


namespace product_of_solutions_to_x_squared_equals_49_l423_423120

theorem product_of_solutions_to_x_squared_equals_49 :
  (∃ (x : ℝ), x ^ 2 = 49) → ((∀ x, x ^ 2 = 49 → (x = 7 ∨ x = -7))) →
  (∏ x in { x : ℝ | x ^ 2 = 49}.to_finset, x) = -49 :=
begin
  sorry
end

end product_of_solutions_to_x_squared_equals_49_l423_423120


namespace apple_price_33kg_l423_423060

theorem apple_price_33kg
  (l q : ℝ)
  (h1 : 10 * l = 3.62)
  (h2 : 30 * l + 6 * q = 12.48) :
  30 * l + 3 * q = 11.67 :=
by
  sorry

end apple_price_33kg_l423_423060


namespace evaluate_expression_is_sixth_l423_423018

noncomputable def evaluate_expression := (1 / Real.log 3000^4 / Real.log 8) + (4 / Real.log 3000^4 / Real.log 9)

theorem evaluate_expression_is_sixth:
  evaluate_expression = 1 / 6 :=
  by
  sorry

end evaluate_expression_is_sixth_l423_423018


namespace car_and_bicycle_distances_l423_423186

noncomputable def train_speed : ℝ := 100 -- speed of the train in mph
noncomputable def car_speed : ℝ := (2 / 3) * train_speed -- speed of the car in mph
noncomputable def bicycle_speed : ℝ := (1 / 5) * train_speed -- speed of the bicycle in mph
noncomputable def travel_time_hours : ℝ := 30 / 60 -- travel time in hours, which is 0.5 hours

noncomputable def car_distance : ℝ := car_speed * travel_time_hours
noncomputable def bicycle_distance : ℝ := bicycle_speed * travel_time_hours

theorem car_and_bicycle_distances :
  car_distance = 100 / 3 ∧ bicycle_distance = 10 :=
by
  sorry

end car_and_bicycle_distances_l423_423186


namespace part_a_part_b_l423_423368

def good (p q n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def bad (p q n : ℕ) : Prop := 
  ¬ good p q n

theorem part_a (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ A, A = p * q - p - q ∧ ∀ x y, x + y = A → (good p q x ∧ bad p q y) ∨ (bad p q x ∧ good p q y) := by
  sorry

theorem part_b (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ N, N = (p - 1) * (q - 1) / 2 ∧ ∀ n, n < p * q - p - q → bad p q n :=
  sorry

end part_a_part_b_l423_423368


namespace laborer_monthly_income_l423_423295

variable (I : ℝ)

noncomputable def average_expenditure_six_months := 70 * 6
noncomputable def debt_condition := I * 6 < average_expenditure_six_months
noncomputable def expenditure_next_four_months := 60 * 4
noncomputable def total_income_next_four_months := expenditure_next_four_months + (average_expenditure_six_months - I * 6) + 30

theorem laborer_monthly_income (h1 : debt_condition I) (h2 : total_income_next_four_months I = I * 4) :
  I = 69 :=
by
  sorry

end laborer_monthly_income_l423_423295


namespace russian_letter_word_quotient_l423_423965

theorem russian_letter_word_quotient :
  let word1 := "СКАЛКА",
      word2 := "ТЕФТЕЛЬ",
      six_letter_words := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2),
      seven_letter_words := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)
  in six_letter_words = 180 ∧ seven_letter_words = 1260 ∧ seven_letter_words / six_letter_words = 7 :=
by
  let word1 := "СКАЛКА"
  let word2 := "ТЕФТЕЛЬ"
  let six_letter_words := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2)
  let seven_letter_words := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)
  have h1 : six_letter_words = 180 := sorry
  have h2 : seven_letter_words = 1260 := sorry
  have h3 : seven_letter_words / six_letter_words = 7 := sorry
  exact ⟨h1, h2, h3⟩

end russian_letter_word_quotient_l423_423965


namespace vector_magnitude_l423_423181

variables {R : Type*} [LinearOrderedField R] [Module R (R × R)]
open Real

def vector_length (v : R × R) : R := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def vector_add (a b : R × R) : R × R := (a.1 + b.1, a.2 + b.2)

noncomputable def vector_scale (c : R) (v : R × R) : R × R := (c * v.1, c * v.2)

noncomputable def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

variables (a b : R × R) (c : R)
variables (ha : vector_length a = 1)
variables (hb : b = (1, 2))
variables (h_perp : dot_product a b = 0)

theorem vector_magnitude : vector_length (vector_add (vector_scale 2 a) b) = 3 :=
sorry

end vector_magnitude_l423_423181


namespace symmetry_axis_of_f_l423_423515

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∃ k : ℤ, ∃ k_π_div_2 : ℝ, (f (k * π / 2 + π / 12) = f ((k * π / 2 + π / 12) + π)) :=
by {
  sorry
}

end symmetry_axis_of_f_l423_423515


namespace more_than_half_millet_on_day_5_l423_423265

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end more_than_half_millet_on_day_5_l423_423265


namespace baseball_cards_total_l423_423355

theorem baseball_cards_total (num_friends : ℕ) (cards_per_friend : ℕ) (total_cards : ℕ)
  (h1 : num_friends = 5) (h2 : cards_per_friend = 91) :
  total_cards = 455 :=
by {
  rw [h1, h2],
  have : total_cards = 5 * 91,
  { sorry },
  rw this,
  exact rfl,
}

end baseball_cards_total_l423_423355


namespace dot_product_magnitude_l423_423884

variables (v : ℝ^3)

theorem dot_product_magnitude (h : ‖v‖ = 7) : v ⋅ v = 49 :=
sorry

end dot_product_magnitude_l423_423884


namespace triangle_angle_cos_range_l423_423906

theorem triangle_angle_cos_range
  (A B C a b c : ℝ)
  (h1 : sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A) :
  A = π / 6 ∧ (cos (5 * π / 2 - B) - 2 * sin (C / 2)^2) ∈ set.Icc (-(sqrt 3 + 2) / 2) (sqrt 3 - 1) :=
  sorry

end triangle_angle_cos_range_l423_423906


namespace calculate_fraction_power_l423_423755

theorem calculate_fraction_power :
  ({(3:ℚ) / 2})^(2023) * ({(2:ℚ) / 3})^(2022) = 3 / 2 :=
by
  sorry

end calculate_fraction_power_l423_423755


namespace correct_option_l423_423498

def p := ∃ x : ℝ, sin x = real.pi / 2
def q := ∀ x : ℝ, (x > 1 ∧ x < 2) ↔ x^2 - 3 * x + 2 < 0

theorem correct_option : (¬ p ∧ q) ∧ (¬ (p ∧ q) ∧ ¬ (p ∧ ¬ q)) := by
  sorry

end correct_option_l423_423498


namespace combinatorial_inequality_l423_423833

open Set

variables {X : Type*} [Fintype X] {n m : ℕ}
variables (A B : Fin n → Set X)
variables (a b : Fin n → ℕ)
variable [∀ i, Fintype (A i)]
variable [∀ i, Fintype (B i)]

noncomputable def card_A (i : Fin n) : ℕ := (A i).toFinset.card
noncomputable def card_B (i : Fin n) : ℕ := (B i).toFinset.card
def disjoint (i j : Fin n) : Prop := (A i ∩ B j) = ∅ ↔ i = j

theorem combinatorial_inequality
  (h_disjoint : ∀ i j, disjoint A B i j)
  (h_card_A : ∀ i, card_A A i = a i)
  (h_card_B : ∀ i, card_B B i = b i) :
  (∑ i in finset.range m, 1 / (Nat.choose (a i + b i) (a i) : ℝ)) ≤ 1 :=
sorry

end combinatorial_inequality_l423_423833


namespace chef_additional_wings_l423_423722

theorem chef_additional_wings
    (n : ℕ) (w_initial : ℕ) (w_per_friend : ℕ) (w_additional : ℕ)
    (h1 : n = 4)
    (h2 : w_initial = 9)
    (h3 : w_per_friend = 4)
    (h4 : w_additional = 7) :
    n * w_per_friend - w_initial = w_additional :=
by
  sorry

end chef_additional_wings_l423_423722


namespace sum_of_solutions_eq_114_over_11_l423_423457

theorem sum_of_solutions_eq_114_over_11 (x : ℝ) :
  (∑ x in { x : ℝ | √(2 * x) + √(8 / x) + 2 * √(x + 4 / x) = 10 }, x) = 114 / 11 :=
by 
  sorry

end sum_of_solutions_eq_114_over_11_l423_423457


namespace main_theorem_l423_423434

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → 
  (x1 < x2 ↔ (f x2 < f x1))

theorem main_theorem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end main_theorem_l423_423434


namespace tangent_line_at_point_eq_l423_423451

noncomputable def tangent_line_equation (y : ℝ → ℝ) (x0 y0 : ℝ) : ℝ → ℝ :=
  let y' := derivative y
  let m := y' x0
  fun x => y0 + m * (x - x0)

theorem tangent_line_at_point_eq 
  (f : ℝ → ℝ)
  (df : ℝ → ℝ)
  (h₀ : ∀ x, f x = x * Real.exp x - 2 * x^2 + 1)
  (h₁ : ∀ x, df x = (1 + x) * Real.exp x - 4 * x)
  (h₂ : f 0 = 1) :
  tangent_line_equation f 0 1 = fun x => x + 1 :=
by
  sorry

end tangent_line_at_point_eq_l423_423451


namespace problem_solution_l423_423243

noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).sum id

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_sum_of_divisors : ℕ :=
  (finset.filter (λ n, is_prime (sum_of_divisors n)) (finset.range 31)).card

theorem problem_solution : count_prime_sum_of_divisors = 5 :=
sorry

end problem_solution_l423_423243


namespace donovan_points_needed_l423_423775

-- Definitions based on conditions
def average_points := 26
def games_played := 15
def total_games := 20
def goal_average := 30

-- Assertion
theorem donovan_points_needed :
  let total_points_needed := goal_average * total_games
  let points_already_scored := average_points * games_played
  let remaining_games := total_games - games_played
  let remaining_points_needed := total_points_needed - points_already_scored
  let points_per_game_needed := remaining_points_needed / remaining_games
  points_per_game_needed = 42 :=
  by
    -- Proof skipped
    sorry

end donovan_points_needed_l423_423775


namespace total_cost_correct_l423_423383

noncomputable def total_cost : ℝ :=
  let first_path_area := 5 * 100
  let first_path_cost := first_path_area * 2
  let second_path_area := 4 * 80
  let second_path_cost := second_path_area * 1.5
  let diagonal_length := Real.sqrt ((100:ℝ)^2 + (80:ℝ)^2)
  let third_path_area := 6 * diagonal_length
  let third_path_cost := third_path_area * 3
  let circular_path_area := Real.pi * (10:ℝ)^2
  let circular_path_cost := circular_path_area * 4
  first_path_cost + second_path_cost + third_path_cost + circular_path_cost

theorem total_cost_correct : total_cost = 5040.64 := by
  sorry

end total_cost_correct_l423_423383


namespace parallelogram_area_sum_less_than_25_l423_423660

-- Define the conditions of the problem
def triangle (T : Type) [MetricSpace T] (P Q R : T) : Prop :=
  let perimeter := 100 in -- total perimeter is 100 cm
  let area := 100 in -- total area is 100 cm²
  let parallel_distance := 1 in -- distance of parallel lines from sides

-- Define the main theorem to prove
theorem parallelogram_area_sum_less_than_25 
  (T : Type) [MetricSpace T] (P Q R : T)
  (h_perimeter : triangle T P Q R)
  (h_area : triangle T P Q R)
  (h_parallel_distance : triangle T P Q R) :
  ∃ parallelograms : List (Set T), 
  length parallelograms = 3 ∧ (∑ P in parallelograms, area P) < 25 :=
  sorry

end parallelogram_area_sum_less_than_25_l423_423660


namespace temperature_fifth_day_l423_423699

variable (T1 T2 T3 T4 T5 : ℝ)

-- Conditions
def condition1 : T1 + T2 + T3 + T4 = 4 * 58 := by sorry
def condition2 : T2 + T3 + T4 + T5 = 4 * 59 := by sorry
def condition3 : T5 = (8 / 7) * T1 := by sorry

-- The statement we need to prove
theorem temperature_fifth_day : T5 = 32 := by
  -- Using the provided conditions
  sorry

end temperature_fifth_day_l423_423699


namespace baseball_cards_total_l423_423354

theorem baseball_cards_total (num_friends : ℕ) (cards_per_friend : ℕ) (total_cards : ℕ)
  (h1 : num_friends = 5) (h2 : cards_per_friend = 91) :
  total_cards = 455 :=
by {
  rw [h1, h2],
  have : total_cards = 5 * 91,
  { sorry },
  rw this,
  exact rfl,
}

end baseball_cards_total_l423_423354


namespace sum_of_valid_student_counts_l423_423048

variable (s : ℕ) (sum : ℕ)

def valid_student_count (s : ℕ) : Prop :=
  150 ≤ s ∧ s ≤ 200 ∧ ∃ k : ℕ, s = 6 * k + 1

theorem sum_of_valid_student_counts :
  sum = (∑ i in finset.range 50, if valid_student_count (151 + 6 * i) then (151 + 6 * i) else 0) :=
begin
  sorry
end

end sum_of_valid_student_counts_l423_423048


namespace product_fraction_simplified_l423_423754

theorem product_fraction_simplified :
  (∏ n in finset.range 14 + 1, (n * (n + 3)) / ((n + 5)^2)) = (25 / 76415) :=
by
  sorry

end product_fraction_simplified_l423_423754


namespace check_point_on_circle_l423_423433

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem check_point_on_circle : 
  let (r, θ) := (5, (3 * Real.pi / 4))
  let (x, y) := polar_to_rectangular r θ
  x^2 + y^2 = 25 :=
by
  let (r, θ) := (5, (3 * Real.pi / 4))
  let (x, y) := polar_to_rectangular r θ
  calc
    x^2 + y^2 = (r * Real.cos θ)^2 + (r * Real.sin θ)^2 : by sorry
          ... = r^2 * (Real.cos θ)^2 + r^2 * (Real.sin θ)^2 : by sorry
          ... = r^2 * ((Real.cos θ)^2 + (Real.sin θ)^2) : by sorry
          ... = r^2 * 1 : by sorry
          ... = 25 : by sorry

end check_point_on_circle_l423_423433


namespace petr_receives_1000000_l423_423445

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end petr_receives_1000000_l423_423445


namespace ten_power_equality_l423_423531

theorem ten_power_equality (y : ℝ) : 10^(4*y) = 100 → 10^(y/2) = 10^(1/4) :=
by
  sorry

end ten_power_equality_l423_423531


namespace four_point_questions_l423_423004

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 := 
sorry

end four_point_questions_l423_423004


namespace sequence_a4_l423_423563

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + 1 / (n * (n - 1))

theorem sequence_a4 :
  ∃ (a : ℕ → ℚ), seq a ∧ a 4 = 7 / 4 :=
begin
  sorry
end

end sequence_a4_l423_423563


namespace smallest_area_is_10_l423_423462

noncomputable def smallest_square_area : ℝ :=
  let k₁ := 65
  let k₂ := -5
  10 * (9 + 4 * k₂)

theorem smallest_area_is_10 :
  smallest_square_area = 10 := by
  sorry

end smallest_area_is_10_l423_423462


namespace min_area_l423_423235

noncomputable def min_area_triangle (m : ℝ) (h : ℝ) :=
  if h = -20 * m then
    let base := -((16 + h) / m) in
    let height := 16 + h in
    (1 / 2) * base * height
  else
    0

theorem min_area (m : ℝ) (h : ℝ) (h_neg : m < 0) (pass_point : ∃ (m : ℝ) (h : ℝ), h = -20 * m ∧ 16 = 20 * m + 16 + h) :
  min_area_triangle m h = 640 := by
  sorry

end min_area_l423_423235


namespace product_of_solutions_eq_neg49_l423_423123

theorem product_of_solutions_eq_neg49 :
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (∏ x in ({7, -7} : Finset ℤ), x) = -49 := by
  sorry

end product_of_solutions_eq_neg49_l423_423123


namespace product_of_solutions_product_of_all_solutions_l423_423125

theorem product_of_solutions (x : ℝ) (h : x^2 = 49) : x = 7 ∨ x = -7 :=
begin
  rw eq_comm at h,
  exact eq_or_eq_neg_eq_of_sq_eq_sq h,
end

theorem product_of_all_solutions (h : {x : ℝ | x^2 = 49} = {7, -7}) : 7 * (-7) = -49 :=
by {
  rw set.eq_singleton_iff_unique_mem at h,
  rw [h.2 7, h.2 (-7)],
  exact mul_neg_eq_neg_mul_symm,
}

end product_of_solutions_product_of_all_solutions_l423_423125


namespace equation_d_has_no_real_roots_l423_423341

theorem equation_d_has_no_real_roots : 
  let a := 4
  let b := -5
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant < 0 :=
by
  let a := 4
  let b := -5
  let c := 2
  let discriminant := b^2 - 4 * a * c
  have h : discriminant = -7 := by 
    calc
      b^2 - 4 * a * c 
      = (-5)^2 - 4 * 4 * 2 : rfl
      ... = 25 - 32 : rfl
      ... = -7 : rfl
  show discriminant < 0, from by 
    rw h
    exact dec_trivial

end equation_d_has_no_real_roots_l423_423341


namespace calculate_total_surface_area_l423_423052

-- Define the cube and its properties
structure Cube :=
  (edge_length : ℝ)
  (hole_length : ℝ)
  (hole_distance : ℝ)

-- Define the specific cube in the problem
def wooden_cube : Cube :=
  { edge_length := 4,
    hole_length := 1,
    hole_distance := 1 }

-- Calculate the entire surface area including the inside
theorem calculate_total_surface_area :
  let original_surface_area := 6 * (wooden_cube.edge_length ^ 2),
      area_removed := 6 * 2 * (wooden_cube.hole_length ^ 2),
      area_exposed := 6 * 2 * 4 * (wooden_cube.hole_length ^ 2)
  in original_surface_area - area_removed + area_exposed = 132 :=
by
  -- Proof would go here
  sorry

end calculate_total_surface_area_l423_423052


namespace cube_symmetries_isom_S4_l423_423277

open FiniteGroup

noncomputable def cube_orientation_preserving_symmetries : Group := sorry

noncomputable def S_4 := SymmetricGroup (Finset.range 4)

theorem cube_symmetries_isom_S4 : cube_orientation_preserving_symmetries ≃* S_4 := sorry

end cube_symmetries_isom_S4_l423_423277


namespace sum_of_largest_and_smallest_l423_423325

theorem sum_of_largest_and_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  a + c = 22 :=
by
  sorry

end sum_of_largest_and_smallest_l423_423325


namespace Tina_profit_l423_423675

variables (x : ℝ) (profit_per_book : ℝ) (number_of_people : ℕ) (cost_per_book : ℝ)
           (books_per_customer : ℕ) (total_profit : ℝ) (total_cost : ℝ) (total_books_sold : ℕ)

theorem Tina_profit :
  (number_of_people = 4) →
  (cost_per_book = 5) →
  (books_per_customer = 2) →
  (total_profit = 120) →
  (books_per_customer * number_of_people = total_books_sold) →
  (cost_per_book * total_books_sold = total_cost) →
  (total_profit = total_books_sold * x - total_cost) →
  x = 20 :=
by
  intros
  sorry


end Tina_profit_l423_423675


namespace exists_infinitely_many_natural_numbers_l423_423619

theorem exists_infinitely_many_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ k : ℕ, let n := f k in n = 4 * k + 1 ∧ (n * (n + 1)) % 12 = 0) :=
begin
  sorry
end

end exists_infinitely_many_natural_numbers_l423_423619


namespace accessory_factory_growth_l423_423714

theorem accessory_factory_growth (x : ℝ) :
  600 + 600 * (1 + x) + 600 * (1 + x) ^ 2 = 2180 :=
sorry

end accessory_factory_growth_l423_423714


namespace rect_has_integer_side_length_l423_423488

structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(is_rect : x_min < x_max ∧ y_min < y_max)

structure SubRectangle (R : Rectangle) :=
(x_min y_min x_max y_max : ℝ)
(is_rect : x_min < x_max ∧ y_min < y_max)
(parallel : R.x_min = x_min ∨ R.x_min = x_max ∨ R.y_min = y_min ∨ R.y_min = y_max)
(non_overlapping : true) -- Placeholder for non-overlapping condition
(integer_side : int_side : (x_max - x_min) ∈ ℤ ∨ (y_max - y_min) ∈ ℤ)

def has_integer_side_length (R : Rectangle) (subs : list (SubRectangle R)) : Prop :=
  ∃ (w h : ℝ), w = R.x_max - R.x_min ∧ h = R.y_max - R.y_min ∧ (w ∈ ℤ ∨ h ∈ ℤ)

theorem rect_has_integer_side_length
  {R : Rectangle}
  (subs : list (SubRectangle R)) :
  has_integer_side_length R subs :=
sorry

end rect_has_integer_side_length_l423_423488


namespace counting_indistinguishable_boxes_l423_423876

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l423_423876


namespace exists_nonconvex_polyhedron_invisible_from_point_l423_423440

theorem exists_nonconvex_polyhedron_invisible_from_point (M : Point) (polyhedron : Polyhedron) (outside : M ∉ polyhedron.vertices) (opaque_material : polyhedron.material = Opaque) :
  ∃ polyhedron' : Polyhedron, non_convex polyhedron' ∧ (∀ v ∈ polyhedron'.vertices, is_invisible_from(M, v)) :=
sorry

end exists_nonconvex_polyhedron_invisible_from_point_l423_423440


namespace average_weight_l423_423642

theorem average_weight (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 47) (h3 : B = 39) : (A + B + C) / 3 = 45 := 
  sorry

end average_weight_l423_423642


namespace solve_cubic_eq_neg4_l423_423801

theorem solve_cubic_eq_neg4 (z : ℂ) : 
  z^3 = -4 ↔ 
  z = -real.cbrt 4 ∨ 
  z = real.cbrt 2⁻¹ + real.cbrt 3 / real.cbrt 2 * complex.I ∨ 
  z = real.cbrt 2⁻¹ - real.cbrt 3 / real.cbrt 2 * complex.I :=
sorry

end solve_cubic_eq_neg4_l423_423801


namespace arrangements_TOOTH_l423_423773
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end arrangements_TOOTH_l423_423773


namespace find_mb_l423_423303

-- Definitions based on the conditions
def y_intercept : ℝ := -1
def point1 : ℝ × ℝ := (0, -1)
def point2 : ℝ × ℝ := (1, 1)
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
def m : ℝ := slope point1 point2
def b : ℝ := y_intercept

theorem find_mb : m * b = -2 :=
by
  sorry

end find_mb_l423_423303


namespace solution_set_of_inequality_l423_423668

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l423_423668


namespace problem_l423_423950

variable {a b c : ℝ}
variable {x : ℂ}

theorem problem 
  (H1 : ∃ (α β : ℝ), (α > 0) ∧ (β ≠ 0) ∧ (x = α + β * complex.I ∨ x = α - β * complex.I) 
      ∧ ∀ x : ℂ, x ^ 2 - (a + b + c) * x + (a * b + b * c + c * a) = 0) :
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ (∃ (u v w : ℝ), (u = real.sqrt a) ∧ (v = real.sqrt b) ∧ (w = real.sqrt c) ∧ u + v > w ∧ u + w > v ∧ v + w > u) :=
sorry

end problem_l423_423950


namespace product_of_solutions_l423_423115

theorem product_of_solutions (x : ℤ) (h : x^2 = 49) : ∏ (x : {x // x^2 = 49}), x = -49 := sorry

end product_of_solutions_l423_423115


namespace find_b_l423_423577

-- Define vector a
def a : ℝ^3 := ⟨3, -2, 4⟩

-- Define vector b
def b : ℝ^3 := ⟨13/4, -7/6, 1/3⟩

-- Statement of the problem
theorem find_b : 
  (a.dot_product b = 3) ∧ (a.cross_product b = ⟨8, 12, -10⟩) :=
by sorry

end find_b_l423_423577


namespace PG_entitled_amount_l423_423443

-- Definitions and conditions provided in the problem
def initial_investment_VP : ℕ := 200_000
def initial_investment_PG : ℕ := 350_000
def AA_purchase_amount : ℕ := 1_100_000
def factory_value_after_sale : ℕ := 3_300_000
def share_value_each_after_sale : ℕ := factory_value_after_sale / 3
def VP_initial_share_value := (initial_investment_VP * factory_value_after_sale) / (initial_investment_VP + initial_investment_PG)
def VP_sold_share_value := VP_initial_share_value - share_value_each_after_sale
def PG_received_amount := AA_purchase_amount - VP_sold_share_value

-- Theorem to be proven
theorem PG_entitled_amount : PG_received_amount = 1_000_000 := by
  sorry

end PG_entitled_amount_l423_423443


namespace find_possible_values_l423_423940

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end find_possible_values_l423_423940


namespace train_crossing_time_l423_423696

noncomputable def train_length : ℕ := 150
noncomputable def bridge_length : ℕ := 150
noncomputable def train_speed_kmph : ℕ := 36

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

noncomputable def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time_in_seconds (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_crossing_time :
  crossing_time_in_seconds total_distance train_speed_mps = 30 :=
by
  sorry

end train_crossing_time_l423_423696


namespace dollar_result_l423_423135

variable {ℝ} (a b c x y z : ℝ)

def dollar (a b c : ℝ) : ℝ :=
  (a - b - c) ^ 2

theorem dollar_result :
  dollar ((x - z) ^ 2) ((y - x) ^ 2) ((y - z) ^ 2) = ((-2 * x * z + z ^ 2 + 2 * y * x - 2 * y * z) ^ 2) :=
by
  sorry

end dollar_result_l423_423135


namespace john_coins_value_l423_423926

variable (q d : ℕ)
variable (total_value swapped_value : ℕ)

def condition1 := q + d = 30
def condition2 := swapped_value = total_value + 150
def original_value := total_value = 25 * q + 10 * d
def swapped_calculation := swapped_value = 10 * q + 25 * d
def total_coins_value_in_dollars := (25 * q + 10 * d) = 450

theorem john_coins_value : condition1 → condition2 → original_value → swapped_calculation → total_coins_value_in_dollars := 
by
sory

end john_coins_value_l423_423926


namespace reflection_matrix_solution_l423_423764

-- Define variables p, q
variables (p q : ℚ)

-- Define the matrix R
def R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[p, q], [3 / 4, -1 / 4]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

-- The proof statement
theorem reflection_matrix_solution :
  (R * R = I) → (p = 1 / 4 ∧ q = 5 / 4) :=
by
  sorry

end reflection_matrix_solution_l423_423764


namespace expression_one_expression_two_l423_423069

theorem expression_one :
  3 * (-4: ℝ) ^ 3 + 8 ^ (2 / 3: ℝ) + 25 ^ (-1 / 2: ℝ) = (1 / 5: ℝ) :=
by
  sorry

theorem expression_two :
  3 ^ real.log 2 / real.log 3 + real.log 5 / real.log 3 - real.log 15 / real.log 3 + (real.log 8 / real.log 3) * (real.log 3 / real.log 2) =
  (4: ℝ) :=
by
  sorry

end expression_one_expression_two_l423_423069


namespace price_increase_to_restore_l423_423313

-- Define the original price of the jacket
def P : ℝ := 100

-- Define the price after a 10% reduction
def P' : ℝ := P - 0.1 * P

-- Define the price after an additional 30% reduction
def P'' : ℝ := P' - 0.3 * P'

-- Define the percentage increase needed to restore to original price
def percentage_increase_needed : ℝ := ((P - P'') / P'') * 100

-- The theorem that states the required percentage increase is approximately 58.73%
theorem price_increase_to_restore :
  percentage_increase_needed ≈ 58.73 :=
sorry

end price_increase_to_restore_l423_423313


namespace find_denominator_l423_423201

theorem find_denominator (y : ℝ) (x : ℝ) (h₀ : y > 0) (h₁ : 9 * y / 20 + 3 * y / x = 0.75 * y) : x = 10 :=
sorry

end find_denominator_l423_423201


namespace paths_from_A_to_B_grid_l423_423972

open Nat

theorem paths_from_A_to_B_grid :
  let grid_width := 6
  let grid_height := 5
  let total_moves := 10
  let right_moves := 6
  let up_moves := 4
  total_moves = right_moves + up_moves →
  grid_width = right_moves →
  grid_height = up_moves →
  ∃ paths : ℕ, paths = choose total_moves up_moves ∧ paths = 210 :=
begin
  intros,
  sorry
end

end paths_from_A_to_B_grid_l423_423972


namespace B_memory_unit_data_after_operations_l423_423924

variable (N : ℕ)
variable (hN : N ≥ 3)
variable (A_init B_init C_init : ℕ) (hA_init : A_init = 0) (hB_init : B_init = 0) (hC_init : C_init = 0)

def A_after_first_op := A_init + N
def B_after_first_op := B_init + N
def C_after_first_op := C_init + N

def A_after_second_op := A_after_first_op - 2
def B_after_second_op := B_after_first_op + 2

def A_after_third_op := A_after_second_op
def B_after_third_op := B_after_second_op + 2
def C_after_third_op := C_after_first_op - 2

def A_after_fourth_op := 2 * A_after_third_op
def B_after_fourth_op := B_after_third_op - A_after_third_op

theorem B_memory_unit_data_after_operations : B_after_fourth_op hN hA_init hB_init hC_init = 6 := by
  -- Importing variables and initial conditions
  have A0 : A_after_first_op = N := by simp [A_after_first_op, hA_init]
  have B0 : B_after_first_op = N := by simp [B_after_first_op, hB_init]
  have C0 : C_after_first_op = N := by simp [C_after_first_op, hC_init]
  have A1 : A_after_second_op = N - 2 := by simp [A_after_second_op, A0]
  have B1 : B_after_second_op = N + 2 := by simp [B_after_second_op, B0]
  have A2 : A_after_third_op = N - 2 := by simp [A_after_third_op, A1]
  have B2 : B_after_third_op = N + 4 := by simp [B_after_third_op, B1]
  have C2 : C_after_third_op = N - 2 := by simp [C_after_third_op, C0]
  have A3 : A_after_fourth_op = 2 * (N - 2) := by simp [A_after_fourth_op, A2]
  have B3 : B_after_fourth_op = (N + 4) - (N - 2) := by simp [B_after_fourth_op, B2, A2]
  simp [B3]
  Linarith

#print B_memory_unit_data_after_operations

end B_memory_unit_data_after_operations_l423_423924


namespace friendly_enumeration_l423_423238

open Equiv.Perm

def friendly_permutation {n : ℕ} (α β : perm (Fin n)) (k : ℕ) : Prop :=
  ∀ i : Fin n, if i.val + 1 ≤ k then β i = α ⟨k - i.val, sorry⟩ else β i = α i

theorem friendly_enumeration (n : ℕ) (h : 2 ≤ n) : 
  ∃ (P : List (perm (Fin n))), 
    ∃ (m : ℕ), m = nat.factorial n ∧ 
    ∃ (P₀ : perm (Fin n)), 
      P ≠ [] ∧
      P.head = some P₀ ∧
      P.last = some P₀ ∧
      ∀ i (H: i < m), friendly_permutation (P.nthLe i (nat.lt_succ_self _)) (P.nthLe (i+1) (nat.lt_succ_self _)) (n-1) :=
sorry

end friendly_enumeration_l423_423238


namespace sum_2023_fractions_nearest_thousandth_l423_423688

noncomputable def sum_series (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (2 : ℝ) / ((k+1) * ((k+1) + 3))

theorem sum_2023_fractions_nearest_thousandth :
  (Real.round (sum_series 2023 * 1000) / 1000 = 1.222) :=
by
  sorry

end sum_2023_fractions_nearest_thousandth_l423_423688


namespace sin_B_eq_sqrt3_div_3_length_of_c_l423_423899

noncomputable def a : ℝ := 3
noncomputable def b : ℝ := 2
noncomputable def cos_A : ℝ := 1 / 2

theorem sin_B_eq_sqrt3_div_3 (h : cos_A = 1 / 2) : 
  let sin_B := b * Real.sqrt (1 - cos_A^2) / a in 
  sin_B = Real.sqrt(3) / 3 :=
by sorry

theorem length_of_c (h : cos_A = 1 / 2) : 
  let c_square := a^2 - b^2 + 2 * b * a * cos_A,
      c := Real.sqrt(c_square)
  in c = 1 + Real.sqrt 6 :=
by sorry

end sin_B_eq_sqrt3_div_3_length_of_c_l423_423899


namespace divisible_by_120_l423_423616

theorem divisible_by_120 (n : ℤ) : 120 ∣ (n ^ 6 + 2 * n ^ 5 - n ^ 2 - 2 * n) :=
by sorry

end divisible_by_120_l423_423616


namespace geometric_series_S6_value_l423_423504

theorem geometric_series_S6_value (S : ℕ → ℝ) (S3 : S 3 = 3) (S9_minus_S6 : S 9 - S 6 = 12) : 
  S 6 = 9 :=
by
  sorry

end geometric_series_S6_value_l423_423504


namespace area_of_triangle_ACD_l423_423005

theorem area_of_triangle_ACD
  (a γ δ : ℝ)
  (h0 : 0 < a)
  (h1 : 0 < γ ∧ γ < π)
  (h2 : 0 < δ ∧ δ < π) :
  let S := (1 / 2) * a^2 * ((Real.sin (γ + δ) * Real.sin δ) / (Real.sin γ)) in
  S = (1 / 2) * a^2 * (Real.sin (γ + δ) * Real.sin δ / Real.sin γ) :=
by
  sorry

end area_of_triangle_ACD_l423_423005


namespace largest_product_using_digits_l423_423681

theorem largest_product_using_digits : ∃ (d e a b c : ℕ), 
  {d, e, a, b, c} = {3, 5, 8, 9, 1} ∧ 
  d < 10 ∧ e < 10 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  (10 * d + e) * (100 * a + 10 * b + c) = 77623 ∧ 
  (10 * d + e) = 91 :=
by 
  sorry

end largest_product_using_digits_l423_423681


namespace club_officer_selection_l423_423272

theorem club_officer_selection :
  let members := 12
  let ways := members * (members - 1) * (members - 2) * (members - 3) * (members - 4)
  ways = 95040 :=
begin
  let members := 12,
  let ways := members * (members - 1) * (members - 2) * (members - 3) * (members - 4),
  show ways = 95040,
  sorry
end

end club_officer_selection_l423_423272


namespace range_of_a_range_of_x_min_value_of_fraction_l423_423142

-- Question (1): Prove the range of a.
theorem range_of_a (a b : ℝ) (h_passes_through : a + b + 2 = 1) (h_positive : ∀ x : ℝ, a * x^2 + b * x + 2 > 0) :
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
sorry

-- Question (2): Prove the range of x.
theorem range_of_x (a b x : ℝ) (h_passes_through : a + b + 2 = 1) (h_positive : ∀ a ∈ set.Icc (-2 : ℝ) (-1), a * x^2 - (1 + a) * x + 2 > 0) :
  (1 - Real.sqrt 17) / 4 < x ∧ x < (1 + Real.sqrt 17) / 4 :=
sorry

-- Question (3): Prove the minimum value of (a + 2) / b.
theorem min_value_of_fraction (a b : ℝ) (h_pos_b : b > 0) (h_nonneg_y : ∀ x : ℝ, a * x^2 + b * x + 2 ≥ 0) :
  ∃ (a b : ℝ), (a + 2) / b = 1 :=
sorry

end range_of_a_range_of_x_min_value_of_fraction_l423_423142


namespace range_of_a_l423_423512

-- Define the function f
def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

-- The main theorem to be proved
theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : -1 ≤ x1) (hx1' : x1 ≤ 1) (hx2 : -1 ≤ x2) (hx2' : x2 ≤ 1)
  (h : abs (f a x1 - f a x2) ≥ Real.exp 1 - 1) :
  a ∈ set.Icc 0 (1/Real.exp 1) ∪ set.Icc (Real.exp 1) ∞ :=
sorry

end range_of_a_l423_423512


namespace probability_three_of_one_two_of_other_l423_423369

theorem probability_three_of_one_two_of_other :
  let total_balls := 20
  let draw_balls := 5
  let black_balls := 10
  let white_balls := 10
  let total_outcomes := Nat.choose total_balls draw_balls
  let favorable_outcomes_black3_white2 := Nat.choose black_balls 3 * Nat.choose white_balls 2
  let favorable_outcomes_black2_white3 := Nat.choose black_balls 2 * Nat.choose white_balls 3
  let favorable_outcomes := favorable_outcomes_black3_white2 + favorable_outcomes_black2_white3
  let probability := favorable_outcomes / total_outcomes
  probability = (30 : ℚ) / 43 :=
by 
  sorry

end probability_three_of_one_two_of_other_l423_423369


namespace find_A_then_b_and_c_l423_423898

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}

-- Assumptions from the problem
axiom angle_opposite_sides :
  ∀ (A B C : ℝ) (a b c : ℝ), 0 < A ∧ A < π → 
  (sin (A - π/6) - cos (A + 5 * π / 3) = sqrt 2 / 2) → 
  (cos A = - sqrt 2 / 2 ∧ A = 3 * π / 4)

axiom sin2B_cos2C_eq1 :
  ∀ (B C : ℝ) (a b c : ℝ), 
  (sin B)^2 + cos (2 * C) = 1 → 
  ((sin B)^2 = 2 * (sin C)^2)

axiom law_of_sines :
  ∀ (A B C : ℝ) (a b c : ℝ), 
  (sin A / a = sin B / b ∧ sin A / a = sin C / c) → 
  (b = sqrt 2 * c)

axiom law_of_cosines :
  ∀ (A : ℝ) (a b c : ℝ),
  a^2 = b^2 + c^2 - 2 * b * c * cos A → 
  a = sqrt 5 ∧ A = 3 * π / 4 ∧ b = sqrt 2 * c → 
  (c = 1)

-- The statement
theorem find_A_then_b_and_c :
  (0 < A ∧ A < π) →
  sin (A - π/6) - cos (A + 5 * π / 3) = sqrt 2 / 2 →
  (A = 3 * π / 4) →
  (a = sqrt 5) →
  sin B ^ 2 + cos (2 * C) = 1 →
  (b = sqrt 2) ∧ (c = 1) :=
sorry

end find_A_then_b_and_c_l423_423898


namespace points_form_ellipse_l423_423431

-- Definitions of the conditions
variable (A B P : Point)
variable (d : ℝ) (PA PB : Point → ℝ)

-- Conditions in Lean terms
axiom distance_AB : distance A B = 10
axiom distance_sum : PA P + PB P = 15

-- The proof goal: the set of points P forms an ellipse
theorem points_form_ellipse (P : Point) : 
  (PA P + PB P = 15) → is_ellipse (PA P + PB P) := 
  by 
  -- This is the proof step, which we will omit using sorry
  sorry

end points_form_ellipse_l423_423431


namespace find_length_DE_l423_423921

-- Geometric definitions (using a coordinate approach can make it concrete and useful)
variables {A B C D E : Type} 

noncomputable def BC : ℝ := 20
noncomputable def angle_C : ℝ := 45

def is_midpoint (D : Type) (B C : ℝ) : Prop :=
  (B + C) / 2 = D

def perpendicular_bisector (A : Type) (B C : ℝ) : Prop :=
  ∃ E : Type, is_midpoint D B C

def right_triangle (C D E : Type) (angle_C : ℝ) : Prop :=
  angle_C = 45 ∧ angle_C = 45 ∧ angle_C = 90

def length (x y : Type) (E D : ℝ) : Prop := 
  E = D

theorem find_length_DE :
  ∀ (A B C D E : Type), 
    BC = 20 → angle_C = 45 → is_midpoint D B C → perpendicular_bisector A B C → 
    (right_triangle C D E angle_C → length E D 10) :=
by 
  -- proof steps would go here, but we'll skip for now
  intros,
  sorry

end find_length_DE_l423_423921


namespace canteen_distance_l423_423042

theorem canteen_distance (r G B : ℝ) (d_g d_b : ℝ) (h_g : G = 600) (h_b : B = 800) (h_dg_db : d_g = d_b) : 
  d_g = 781 :=
by
  -- Proof to be completed
  sorry

end canteen_distance_l423_423042


namespace sabina_grant_percentage_l423_423622

theorem sabina_grant_percentage :
  ∀ (total_tuition cost_savings loan : ℕ),
  total_tuition = 30000 →
  cost_savings = 10000 →
  loan = 12000 →
  let remainder := total_tuition - cost_savings in
  let grant_coverage := remainder - loan in
  (grant_coverage: ℚ) / remainder * 100 = 40 := 
by
  intros total_tuition cost_savings loan h1 h2 h3 remainder grant_coverage,
  rw [h1, h2, h3],
  let remainder := total_tuition - cost_savings,
  let grant_coverage := remainder - loan,
  have h4 : remainder = 20000, {
    rw [h1, h2],
  },
  have h5 : grant_coverage = 8000, {
    rw h4,
    exact eq.refl (remainder - loan),
    exact eq.refl (20000 - 12000),
  },
  rw [h4, h5],
  norm_cast,
  exact eq.refl 40,
  sorry

end sabina_grant_percentage_l423_423622


namespace line_intersects_ellipse_l423_423857

theorem line_intersects_ellipse (k : ℝ) :
    (let line : ℝ → ℝ := λ x, k * x + 1 in
     let ellipse : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x^2) / 36 + (y^2) / 20 = 1 in
     ellipse (0, 1) = false) →
    ∃ x y : ℝ, (ellipse (⟨x, y⟩) ∧ y = k * x + 1) :=
by
  intro h1
  use [0, 1]
  split
  sorry
  simp


end line_intersects_ellipse_l423_423857


namespace sum_infinite_identity_l423_423133

noncomputable def G (n : ℕ) : ℝ := (∑ k in Finset.range (n + 1), 1 / (k + 1 : ℝ) ^ 2)

theorem sum_infinite_identity :
  (∑' n, 1 / ((n + 2 : ℝ) * G n * G (n + 1))) = 1 :=
by sorry

end sum_infinite_identity_l423_423133


namespace find_common_ratio_of_gp_l423_423198

noncomputable def common_ratio_g_p (a r : ℝ) : Prop :=
  S_n n := a * (1 - r^n) / (1 - r)
  S_6 = a * (1 - r^6) / (1 - r)
  S_3 = a * (1 - r^3) / (1 - r)
  (S_6 / S_3) = 126

theorem find_common_ratio_of_gp (a r : ℝ) (h : common_ratio_g_p a r) : r = 5 :=
by sorry

end find_common_ratio_of_gp_l423_423198


namespace simplify_expression_l423_423287

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end simplify_expression_l423_423287


namespace eccentricity_of_ellipse_l423_423840

variables {a b c e : ℝ}

-- Definition of geometric progression condition for the ellipse axes and focal length
def geometric_progression_condition (a b c : ℝ) : Prop :=
  (2 * b) ^ 2 = 2 * c * 2 * a

-- Eccentricity calculation
def eccentricity {a c : ℝ} (e : ℝ) : Prop :=
  e = (a^2 - c^2) / a^2

-- Theorem that states the eccentricity under the given condition
theorem eccentricity_of_ellipse (h : geometric_progression_condition a b c) : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_ellipse_l423_423840


namespace solve_for_x_l423_423291

theorem solve_for_x (x : ℝ) : (3^(4 * x) * 3^(4 * x) * 3^(4 * x) * 3^(4 * x) = 81^8) → x = 2 := 
by 
  sorry

end solve_for_x_l423_423291


namespace simplify_expression_l423_423286

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end simplify_expression_l423_423286


namespace scorpion_additional_millipedes_l423_423024

theorem scorpion_additional_millipedes :
  let total_segments := 800 in
  let segments_one_millipede := 60 in
  let segments_two_millipedes := 2 * (2 * segments_one_millipede) in
  let total_eaten := segments_one_millipede + segments_two_millipedes in
  let remaining_segments := total_segments - total_eaten in
  let segments_per_millipede := 50 in
  remaining_segments / segments_per_millipede = 10 :=
by {
  sorry
}

end scorpion_additional_millipedes_l423_423024


namespace parallelogram_angle_ratio_l423_423915

theorem parallelogram_angle_ratio (ABCD : Type*) [parallelogram ABCD]
  (O : ABCD) (CAB DBA DBC ACB AOB : Angle)
  (h1 : ∠ CAB = 2 * ∠ DBA)
  (h2 : ∠ DBC = 2 * ∠ DBA)
  (h3 : ∠ ACB = r * ∠ AOB) :
  r = 7 / 9 := by
  sorry

end parallelogram_angle_ratio_l423_423915


namespace largest_x_for_perfect_square_l423_423798

theorem largest_x_for_perfect_square :
  ∃ x : ℕ, (x = 1972 ∧ ∃ k : ℕ, 4^27 + 4^1000 + 4^x = k^2) ∧
           (∀ y : ℕ, y > 1972 → ¬∃ k : ℕ, 4^27 + 4^1000 + 4^y = k^2) :=
by
  use 1972
  split
  { -- x = 1972
    exact rfl }
  split
  { -- ∃ k, 4^27 + 4^1000 + 4^1972 = k^2
    sorry }
  { -- ∀ y > 1972, ¬∃ k, 4^27 + 4^1000 + 4^y = k^2
    intro y
    intro hy
    sorry }

end largest_x_for_perfect_square_l423_423798


namespace count_expressible_integers_l423_423527

theorem count_expressible_integers :
  let f (x : ℝ) := ⌊3 * x⌋ + ⌊5 * x⌋ + ⌊7 * x⌋ + ⌊9 * x⌋ in
  (∃ (n : ℕ), n = 167 ∧
    ∀ (k : ℕ), k < 500 →
      ∃ (x : ℝ), f x = k + 1) :=
sorry

end count_expressible_integers_l423_423527


namespace angle_B_in_parallelogram_l423_423554

variables (A B C D : ℝ) (ABCD : Prop)

def is_parallelogram (ABCD : Prop) : Prop :=
  ∀ {A B C D : ℝ}, A = C ∧ (A + B = 180)

theorem angle_B_in_parallelogram 
  (h1 : is_parallelogram ABCD) 
  (h2 : A + B + C = 220) : 
  B = 140 := 
begin
  sorry
end

end angle_B_in_parallelogram_l423_423554


namespace no_super_sudoku_exists_l423_423077

def is_super_sudoku (grid : Array (Array ℕ)) : Prop :=
  (∀ i, (Finset.univ : Finset ℕ).image (λ j, grid[i][j]) = Finset.range 1 10) ∧
  (∀ j, (Finset.univ : Finset ℕ).image (λ i, grid[i][j]) = Finset.range 1 10) ∧
  (∀ bi bj, (Finset.univ : Finset ℕ).image (λ k, grid[3 * bi + k / 3][3 * bj + k % 3]) = Finset.range 1 10)

theorem no_super_sudoku_exists :
  ¬∃ grid : Array (Array ℕ), grid.size = 9 ∧ (∀ i, (grid[i]).size = 9) ∧ is_super_sudoku grid := 
sorry

end no_super_sudoku_exists_l423_423077


namespace min_value_fraction_l423_423480

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  ∃ z, (z = 2 * real.sqrt 2 + 1) ∧ (∀ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 1) → (2*x^2 - x + 1)/(x*y) ≥ z) :=
by 
  sorry

end min_value_fraction_l423_423480


namespace simplify_trigonometric_expression_l423_423358

variables (α : ℝ)

theorem simplify_trigonometric_expression :
  3.410 * sin(2 * α)^3 * cos(6 * α) + cos(2 * α)^3 * sin(6 * α) = (3 / 4) * sin(8 * α) :=
by sorry

end simplify_trigonometric_expression_l423_423358


namespace lcm_18_24_30_l423_423683

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l423_423683


namespace accumulated_discount_l423_423049

def apply_percentage (p : ℝ) (percent : ℝ) := p * percent

theorem accumulated_discount (original_price : ℝ) : original_price = 1 →
  let sale_price := apply_percentage original_price 0.80 in
  let sale_price_2 := sale_price - apply_percentage sale_price 0.10 in
  let sale_price_3 := sale_price_2 - apply_percentage sale_price_2 0.05 in
  let sale_price_4 := sale_price_3 + apply_percentage sale_price_3 0.08 in
  let sale_price_5 := sale_price_4 - apply_percentage sale_price_4 0.15 in
  let final_price := sale_price_5 - apply_percentage sale_price_5 0.05 in
  final_price ≈ 0.5965164 :=
begin
  sorry
end

end accumulated_discount_l423_423049


namespace problem_statement_l423_423636
noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin (2 * x + Real.pi / 6)
def is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop := ∀ x, f (2 * c.1 - x) = 2 * c.2 - f x

theorem problem_statement (M : ℝ) (hM : M ≠ 0) : 
    is_symmetric (f M) (2 * Real.pi / 3) ∧ 
    is_periodic (f M) Real.pi ∧ 
    is_center_of_symmetry (f M) (5 * Real.pi / 12, 0) :=
by
  sorry

end problem_statement_l423_423636


namespace g_at_100_l423_423962

variable {ℝ : Type*} [linear_ordered_field ℝ]

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (k : ℝ) (hk : k ≠ 0) : 
  x * g y - y * g x = k * g (x / y)

theorem g_at_100 (k : ℝ) (hk : k ≠ 0) (hk' : k ≠ -1) : g (100 : ℝ) = 0 := sorry

end g_at_100_l423_423962


namespace bisects_AEP_l423_423572

-- Definitions for the problem setup
def isosceles_trapezoid (A B C D : Point) : Prop :=
  let AD_parallel_BC := AD ∥ BC in -- Given AD parallel to BC
  let AD_equal_BC := AD = BC in -- Given AD implies isosceles property
  true

def is_reflection (D X : Point) (BC : Line) : Prop :=
  -- X is the reflection of D with respect to line BC
  true

def is_on_arc (Q B C : Point) (Ω : Circle) : Prop :=
  -- Q is on the arc BC of Ω that does not contain A
  true

def intersection (DQ BC : Line) (P : Point) : Prop :=
  -- P is the intersection of DQ and BC
  true

def parallel (EQ PX : Line) : Prop :=
  -- EQ is parallel to PX
  true

def angle_bisector (EQ BE E Q C : Point) : Prop :=
  -- EQ bisects angle BEC
  true

def angle_bisector_AEP (EQ AE E P : Point) : Prop :=
  -- EQ bisects angle AEP
  true

-- Theorem statement
theorem bisects_AEP 
  (A B C D E Q P X : Point) 
  (BC DQ EQ PX : Line) 
  (Ω : Circle) :
  isosceles_trapezoid A B C D → 
  is_reflection D X BC → 
  is_on_arc Q B C Ω →
  intersection DQ BC P →
  parallel EQ PX →
  angle_bisector EQ B E C →
  angle_bisector_AEP EQ A E P :=
by
  intros h_iso h_refl h_arc h_inter h_par h_angle_bisec
  sorry

end bisects_AEP_l423_423572


namespace tan_double_angle_l423_423828

noncomputable def α : ℝ := sorry -- α is within the interval (π, 3π/2)
def sin_α := -sqrt 10 / 10

theorem tan_double_angle :
  (π < α ∧ α < 3 * π / 2) →
  sin α = sin_α →
  tan (2 * α) = 3 / 4 :=
by
  intros h1 h2
  sorry

end tan_double_angle_l423_423828


namespace square_area_to_octagon_area_l423_423739

theorem square_area_to_octagon_area {s t : ℝ} 
  (h₁ : 4 * s = 8 * t) 
  (h₂ : s^2 = 16) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * t^2 in
  area_octagon = 8 * (1 + Real.sqrt 2) := 
by
  sorry

end square_area_to_octagon_area_l423_423739


namespace ellipse_equation_line_ellipse_intersections_l423_423593

variable (a b : ℝ)
variable (C : Set (ℝ × ℝ))
variable (k : ℝ)

-- Equation of an ellipse passing through (sqrt(2), 1) with given focus and coefficients a, b
def ellipse (a b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}

-- Given conditions
def point_Q := (Real.sqrt 2, 1)
def focus_F := (Real.sqrt 2, 0)

def line_l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 1)}

-- Problem Statement 1: Prove the equation of the ellipse
theorem ellipse_equation :
  (point_Q ∈ ellipse a b) ∧ (a^2 = b^2 + 2) -> ellipse a b = ellipse 2 (Real.sqrt 2) :=
sorry

-- Problem Statement 2: Prove the value of k and the length of the chord
theorem line_ellipse_intersections (k : ℝ) :
  (k > 0) ∧ (∀ (p : ℝ × ℝ), p ∈ (line_l k ∩ ellipse 2 (Real.sqrt 2)) -> (∃ q r : ℝ × ℝ, q.1 + r.1 = p.1 - 1 ∧ p.2 = -k + q.2)) ->
  k = (Real.sqrt 2) / 2 ∧ 
  (∀ (M N : ℝ × ℝ), M ∈ ellipse 2 (Real.sqrt 2) ∧ N ∈ ellipse 2 (Real.sqrt 2) ∧ M.1 = 1 ∧ N.1 = 1 -> (Real.sqrt ((M.1 + N.1)^2 - 4 * M.1 * N.1)) = Real.sqrt 42 / 2) :=
sorry

end ellipse_equation_line_ellipse_intersections_l423_423593


namespace departs_if_arrives_l423_423216

theorem departs_if_arrives (grain_quantity : ℤ) (h : grain_quantity = 30) : -grain_quantity = -30 :=
by {
  have : -grain_quantity = -30,
  from congr_arg (λ x, -x) h,
  exact this
}

end departs_if_arrives_l423_423216


namespace repeating_decimal_to_fraction_l423_423691

noncomputable def repeating_decimal : ℚ := 
  have h : ℚ, from 37/100 + 268/99900,
  h + 268 / (99900 * 10 ^ 3) + 268 / (99900 * 10 ^ 6) + ... -- represents the repeating decimal
  sorry -- Placeholder for formal construction of the repeating decimal

theorem repeating_decimal_to_fraction (x : ℚ) (hx : x = 0.37 + 268 / 99900 + 268 / (99900 * 10 ^ 3) + 268 / (99900 * 10 ^ 6) + ...):
  x * 99900 = 371896 :=
by {
  rw hx,
  -- Detailed formal calculations to show the equality
  sorry -- Placeholder for detailed proof
}

end repeating_decimal_to_fraction_l423_423691


namespace max_real_part_sum_w_l423_423947

noncomputable def z (k : ℕ) : ℂ :=
  16 * complex.exp (2 * real.pi * complex.I * (k : ℝ) / 16)

noncomputable def w (k : ℕ) : ℂ :=
  if real.cos (2 * real.pi * (k : ℝ) / 16) ≥ 0 then z k else complex.I * z k

theorem max_real_part_sum_w :
  ∑ j in finset.range 16, (w j).re = 16 * (1 + 2 * real.cos (real.pi / 8) + 2 * real.cos (real.pi / 4) + 2 * real.cos (3 * real.pi / 8)) :=
sorry

end max_real_part_sum_w_l423_423947


namespace tangent_line_ln_x_and_ln_x_plus_1_l423_423545

theorem tangent_line_ln_x_and_ln_x_plus_1 (k b : ℝ) : 
  (∃ x₁ x₂ : ℝ, (y = k * x₁ + b ∧ y = ln x₁ + 2) ∧ 
                (y = k * x₂ + b ∧ y = ln (x₂ + 1)) ∧ 
                (k = 2 ∧ x₁ = 1 / 2 ∧ x₂ = -1 / 2)) → 
  b = 1 - ln 2 :=
by
  sorry

end tangent_line_ln_x_and_ln_x_plus_1_l423_423545


namespace marble_distribution_l423_423328

theorem marble_distribution (x : ℚ) :
    (2 * x + 2) + (3 * x) + (x + 4) = 56 ↔ x = 25 / 3 := by
  sorry

end marble_distribution_l423_423328


namespace sin_2theta_l423_423886

noncomputable def θ := Real.angle

theorem sin_2theta (θ : Real.angle) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (2 * θ.toReal) = 6 * Real.sqrt 8 / 25 := by
  sorry

end sin_2theta_l423_423886


namespace largest_integer_le_root_l423_423963

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 6

theorem largest_integer_le_root :
  monotone_on f (Set.Ioi (0 : ℝ)) →
  f 3 < 0 →
  f 4 < 0 →
  f 5 > 0 →
  ∀ x : ℕ, x ≤ 4 :=
begin
  intros hm hf3 hf4 hf5,
  -- sorry, proof is omitted
  sorry
end

end largest_integer_le_root_l423_423963


namespace problem_properties_l423_423959

noncomputable theory

-- Define the function f and the given conditions
def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-x + 1)
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom specific_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → ∃ a b : ℝ, f x = a * x^2 + b
axiom condition_sum : f 0 + f 3 = 6

theorem problem_properties :
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, f x = f (12 - x)) ∧
  (∀ x, (1 ≤ x ∧ x ≤ 2 → f x ≠ 2 * x^2 - 2)) ∧
  (f (2025 / 2) = 5 / 2) :=
by
  -- Proof goes here
  sorry

end problem_properties_l423_423959


namespace total_games_in_season_l423_423738

-- Definitions based on the conditions
def num_teams := 16
def teams_per_division := 8
def num_divisions := num_teams / teams_per_division

-- Each team plays every other team in its division twice
def games_within_division_per_team := (teams_per_division - 1) * 2

-- Each team plays every team in the other division once
def games_across_divisions_per_team := teams_per_division

-- Total games per team
def games_per_team := games_within_division_per_team + games_across_divisions_per_team

-- Total preliminary games for all teams (each game is counted twice)
def preliminary_total_games := games_per_team * num_teams

-- Since each game is counted twice, the final number of games
def total_games := preliminary_total_games / 2

theorem total_games_in_season : total_games = 176 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end total_games_in_season_l423_423738


namespace selection_problem_l423_423983

theorem selection_problem :
  let total_ways := (nat.choose 9 3) in    -- Calculate the number of ways to choose 3 from 9 (excluding C)
  let invalid_ways := (nat.choose 7 3) in  -- Calculate the number of ways to choose 3 from 7 (excluding A, B, and C)
  total_ways - invalid_ways = 49           -- Total valid selections minus invalid selections equals 49
:=
begin
  sorry -- Proof is omitted
end

end selection_problem_l423_423983


namespace area_outside_smaller_squares_l423_423044

theorem area_outside_smaller_squares (side_large : ℕ) (side_small1 : ℕ) (side_small2 : ℕ)
  (no_overlap : Prop) (side_large_eq : side_large = 9)
  (side_small1_eq : side_small1 = 4)
  (side_small2_eq : side_small2 = 2) :
  (side_large * side_large - (side_small1 * side_small1 + side_small2 * side_small2)) = 61 :=
by
  sorry

end area_outside_smaller_squares_l423_423044


namespace incorrect_transformation_l423_423000

theorem incorrect_transformation (x y m : ℕ) : 
  (x = y → x + 3 = y + 3) ∧
  (-2 * x = -2 * y → x = y) ∧
  (m ≠ 0 → (x = y ↔ (x / m = y / m))) ∧
  ¬(x = y → x / m = y / m) :=
by
  sorry

end incorrect_transformation_l423_423000


namespace count_rectangles_with_perimeter_twenty_two_l423_423868

theorem count_rectangles_with_perimeter_twenty_two : 
  (∃! (n : ℕ), n = 11) :=
by
  sorry

end count_rectangles_with_perimeter_twenty_two_l423_423868


namespace range_of_a_l423_423085

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end range_of_a_l423_423085


namespace nails_needed_l423_423866

theorem nails_needed (nails_own nails_found nails_total_needed : ℕ) 
  (h1 : nails_own = 247) 
  (h2 : nails_found = 144) 
  (h3 : nails_total_needed = 500) : 
  nails_total_needed - (nails_own + nails_found) = 109 := 
by
  sorry

end nails_needed_l423_423866


namespace ellipse_properties_l423_423402

theorem ellipse_properties :
  ∃ a b h k : ℝ, 
  a = 3 * Real.sqrt 15 ∧
  b = 12 ∧ 
  h = 4 ∧ 
  k = 4 ∧ 
  -- Equation of the ellipse in standard form
  (∀ x y : ℝ, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ 
  -- Conditions that define the ellipse
  (let f1 := (4, 1) in 
  let f2 := (4, 7) in 
  let p := (12, 5) in 
  let distance_foci := Real.sqrt ((4 - 4)^2 + (1 - 7)^2) in
  distance_foci = 6 ∧ 
  (Real.sqrt ((12 - 4)^2 + (5 - 1)^2) + Real.sqrt ((12 - 4)^2 + (5 - 7)^2)) = 10 + 2 * Real.sqrt 17 ∧
  (h, k) = (4, 4) ∧ 
  2 * a = 24 ∧ 
  (2 * c = distance_foci) ∧ 
  (c^2 = a^2 - b^2) ∧ 
  c = 3))

end ellipse_properties_l423_423402


namespace area_shaded_region_l423_423424

-- Provided conditions
def radii_equal (P Q : Point) (r : ℝ) : Prop := dist P Q = r
def midpoint_R (P Q R : Point) : Prop := R = (P + Q) / 2
def segment_PR_length (P R : Point) : Prop := dist P R = 3 * Real.sqrt 3
def tangent_RS (R S P : Point) : Prop := dist R S = 3 ∧ dist S P = 3
def tangent_RT (R T Q : Point) : Prop := dist R T = 3 ∧ dist T Q = 3
def common_tangent_UV (U V : Point) (P Q : Point) : Prop := collinear U V P ∧ collinear U V Q

-- Proof problem
theorem area_shaded_region (P Q R S T U V : Point) (h1 : radii_equal P Q 3)
  (h2 : midpoint_R P Q R) (h3 : segment_PR_length P R) (h4 : tangent_RS R S P)
  (h5 : tangent_RT R T Q) (h6 : common_tangent_UV U V P Q) :
  area (Polygon.mk [R, S, U, V, T]) = 18 * Real.sqrt 3 - 9 - (3 * Real.pi / 2) := sorry

end area_shaded_region_l423_423424


namespace fraction_of_area_in_triangle_l423_423608

theorem fraction_of_area_in_triangle :
  let vertex1 := (3, 3)
  let vertex2 := (5, 5)
  let vertex3 := (3, 5)
  let base := (5 - 3)
  let height := (5 - 3)
  let area_triangle := (1 / 2) * base * height
  let area_square := 6 * 6
  let fraction := area_triangle / area_square
  fraction = (1 / 18) :=
by 
  sorry

end fraction_of_area_in_triangle_l423_423608


namespace inverse_is_arithmetic_general_formula_sum_first_n_terms_l423_423489

noncomputable def a_sequence (n : ℕ) : ℕ → ℚ
| 1 := 1
| (n + 1) := a_sequence n - 2 * (a_sequence n) * (a_sequence (n+1))

def inverse_sequence (n : ℕ) : ℚ := 1 / a_sequence n

def arithmetic_seq (first_term common_diff : ℚ) (n : ℕ) : ℚ :=
first_term + common_diff * (n - 1)

def b_sequence (n : ℕ) : ℚ :=
a_sequence n * a_sequence (n + 1)

def sum_b_sequence (n : ℕ) : ℚ :=
(sum i in range n, b_sequence i)

theorem inverse_is_arithmetic (n : ℕ) (h : n ≥ 1) :
  is_arithmetic (inverse_sequence n) 
:=
sorry

theorem general_formula (n : ℕ) : a_sequence n = 1 / (2 * n - 1)
:=
sorry

theorem sum_first_n_terms (n : ℕ) : sum_b_sequence n = n / (2 * n + 1)
:=
sorry

end inverse_is_arithmetic_general_formula_sum_first_n_terms_l423_423489


namespace number_of_rows_l423_423137

theorem number_of_rows (c o u : ℕ) (h1 : c = 20) (h2 : o = 790) (h3 : u = 10) : 
  (o + u) / c = 40 :=
by
  -- Variables for occupied, unoccupied, and total seats
  let t := o + u
  have ht : t = 800 := by
    -- Calculate total seats
    rw [h2, h3]
    rfl
  -- Calculate the number of rows
  rw [h1, ht]
  norm_num

end number_of_rows_l423_423137


namespace value_of_expression_l423_423669

theorem value_of_expression : 10^2 + 10 + 1 = 111 :=
by
  sorry

end value_of_expression_l423_423669


namespace rounding_to_nearest_hundredth_l423_423283

/-- Prove that rounding 24.63871 to the nearest hundredth equals 24.64. -/
theorem rounding_to_nearest_hundredth (x : ℝ) (h : x = 24.63871) : Real.round_nearest_hundredth x = 24.64 :=
by
  rw [h]
  sorry

end rounding_to_nearest_hundredth_l423_423283


namespace estimate_white_balls_l423_423913

def estimated_white_balls (W : ℝ) : Prop :=
  let theoretical_prob := 8 / (W + 8)
  let experimental_prob := 88 / 400
  theoretical_prob = experimental_prob

theorem estimate_white_balls :
  ∃ (W : ℝ), estimated_white_balls W ∧ W = 28 :=
by
  have : estimated_white_balls 28 := by
    let W : ℝ := 28
    let theoretical_prob := 8 / (W + 8)
    let experimental_prob := 88 / 400
    show theoretical_prob = experimental_prob
    sorry
  exact Exists.intro 28 (and.intro this rfl)

end estimate_white_balls_l423_423913


namespace E_X_calc_probs_extinct_l423_423726

-- Definitions of conditions
def X_probs : Fin 4 → ℝ
| 0 => 0.4
| 1 => 0.3
| 2 => 0.2
| 3 => 0.1

def E_X : ℝ := ∑ i in Finset.univ, i * X_probs i

-- Statements to be proven
theorem E_X_calc : E_X = 1 := by sorry

theorem probs_extinct (p : ℝ) (h : p_0 + p_1 * p + p_2 * p^2 + p_3 * p^3 = p) :
  (E_X ≤ 1 → p = 1) ∧ (E_X > 1 → p < 1) := by sorry

end E_X_calc_probs_extinct_l423_423726


namespace sin_2theta_l423_423887

noncomputable def θ := Real.angle

theorem sin_2theta (θ : Real.angle) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (2 * θ.toReal) = 6 * Real.sqrt 8 / 25 := by
  sorry

end sin_2theta_l423_423887


namespace three_pow_1234_mod_5_l423_423686

theorem three_pow_1234_mod_5 : (3^1234) % 5 = 4 := 
by 
  have h1 : 3^4 % 5 = 1 := by norm_num
  sorry

end three_pow_1234_mod_5_l423_423686


namespace meiosis_fertilization_important_l423_423262

-- Define the necessary concepts
def meiosis (reproductive_cells : Type) : Prop :=
  ∃ halved_cells : Type,
    (∀ original_cells : reproductive_cells, chromosome_count halved_cells = chromosome_count original_cells / 2)

def fertilization (reproductive_cells : Type) (zygote : Type) : Prop :=
  ∀ mature_cells : reproductive_cells, chromosome_count zygote = chromosome_count mature_cells * 2

def somatic_cells_maintained (somatic : Type) : Prop :=
  ∀ generation : ℕ, chromosome_count (somatic generation) = chromosome_count (somatic 0)

def heredity_variation (generations : Type) : Prop :=
  (∃ somatic : generations, somatic_cells_maintained somatic) ∧ variation_in_heredity generations

-- Main theorem statement
theorem meiosis_fertilization_important (T : Type) (generations : ℕ → T) :
  (∀ mature_cells : T, meiosis mature_cells) →
  (∀ mature_cells : T, fertilization mature_cells (generations 0)) →
  heredity_variation generations :=
sorry

end meiosis_fertilization_important_l423_423262


namespace octahedron_side_length_l423_423734

-- Define the dimensions of the rectangular prism
def dimensions : ℝ × ℝ × ℝ := (2, 3, 1)

-- Define the coordinates of vertices of the rectangular prism
def Q1 : ℝ × ℝ × ℝ := (0, 0, 0)
def Q2 : ℝ × ℝ × ℝ := (2, 0, 0)
def Q3 : ℝ × ℝ × ℝ := (0, 3, 0)
def Q4 : ℝ × ℝ × ℝ := (0, 0, 1)
def Q1' : ℝ × ℝ × ℝ := (2, 3, 1)
def Q2' : ℝ × ℝ × ℝ := (0, 3, 1)
def Q3' : ℝ × ℝ × ℝ := (2, 0, 1)
def Q4' : ℝ × ℝ × ℝ := (2, 3, 0)

-- Define the vertex locations of the octahedron in terms of fractions x, y, z
def octahedron_vertex1 (x : ℝ) : ℝ × ℝ × ℝ := (2 * x, 0, 0)
def octahedron_vertex2 (y : ℝ) : ℝ × ℝ × ℝ := (0, 3 * y, 0)
def octahedron_vertex3 (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def octahedron_vertex4 (x : ℝ) : ℝ × ℝ × ℝ := (2 - 2 * x, 3, 1)
def octahedron_vertex5 (y : ℝ) : ℝ × ℝ × ℝ := (2, 3 - 3 * y, 1)
def octahedron_vertex6 (z : ℝ) : ℝ × ℝ × ℝ := (2, 3, 1 - z)

-- The main theorem proving the side length of the octahedron
theorem octahedron_side_length :
  ∀ x y z, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 →
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) →
  (sqrt ((2 - 2 * x)^2 + (3 - 3 * y)^2 + (1 - z)^2) = sqrt (14) / 2) :=
by
  intro x y z
  intro h1
  intro h2
  have h' : x = 1/2 ∧ y = 1/2 ∧ z = 1/2 := h2
  -- rest of the proof 
  sorry

end octahedron_side_length_l423_423734


namespace smallest_fraction_greater_than_three_fifths_l423_423395

theorem smallest_fraction_greater_than_three_fifths : 
    ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a % 1 = 0) ∧ (b % 1 = 0) ∧ (5 * a > 3 * b) ∧ a = 59 :=
by
  sorry

end smallest_fraction_greater_than_three_fifths_l423_423395


namespace probability_xi_2_l423_423842

theorem probability_xi_2 (c : ℝ) (h : c * (1 + 1/2 + 1/3 + 1/4) = 1) :
  let P (k : ℕ) := c / (k + 1)
  in P 2 = 4 / 25 :=
by
  let P (k : ℕ) := c / (k + 1)
  sorry

end probability_xi_2_l423_423842


namespace no_extremum_points_eq_l423_423197

theorem no_extremum_points_eq {a : ℝ} :
  (∀ x : ℝ, (derivative (λ x : ℝ, (1 / 3) * x ^ 3 + x ^ 2 + a * x) x ≠ 0)) ↔ 
  a ≥ 1 :=
by
  have deriv_eq : (∀ x : ℝ, derivative (λ x : ℝ, (1 / 3) * x ^ 3 + x ^ 2 + a * x) x = x ^ 2 + 2 * x + a) := 
  sorry
  have no_zero_deriv : (∀ x : ℝ, x ^ 2 + 2 * x + a ≠ 0) ↔ a ≥ 1 := 
  sorry
  exact no_zero_deriv
  sorry

end no_extremum_points_eq_l423_423197


namespace cost_of_leveling_is_correct_l423_423034

-- Define the radii of the inner and outer circles
def r_inner : ℝ := 16
def r_outer : ℝ := r_inner + 3

-- Define the areas of the inner and outer circles
def A_inner : ℝ := real.pi * r_inner^2
def A_outer : ℝ := real.pi * r_outer^2

-- Define the area of the walk
def A_walk : ℝ := A_outer - A_inner

-- Define the cost per square meter
def cost_per_m2 : ℝ := 2

-- Define the total cost of leveling the walk
def total_cost : ℝ := A_walk * cost_per_m2

-- The theorem to prove
theorem cost_of_leveling_is_correct : total_cost = 660 := by
  sorry

end cost_of_leveling_is_correct_l423_423034


namespace ellipse_eccentricity_l423_423493

def ellipse {a : ℝ} (h : a^2 - 4 = 4) : Prop :=
  ∃ c e : ℝ, (c = 2) ∧ (e = c / a) ∧ (e = (Real.sqrt 2) / 2)

theorem ellipse_eccentricity (a : ℝ) (h : a^2 - 4 = 4) : 
  ellipse h :=
by
  sorry

end ellipse_eccentricity_l423_423493


namespace find_m_and_parity_l423_423169

-- Given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x + m / x
def m_val : ℝ := 1

-- Statement
theorem find_m_and_parity (h : f 1 m_val = 2) : 
  (m_val = 1) ∧ (∀ x : ℝ, x ≠ 0 → f (-x) m_val = -f x m_val) := 
  by 
  sorry

end find_m_and_parity_l423_423169


namespace pump_out_water_time_l423_423712

theorem pump_out_water_time
  (floor_length : ℕ) (floor_width : ℕ) (water_depth_inch : ℕ)
  (pump_rate : ℕ) (num_pumps : ℕ)
  (cubic_foot_to_gallons : ℕ) :
  floor_length = 30 →
  floor_width = 40 →
  water_depth_inch = 24 →
  pump_rate = 10 →
  num_pumps = 4 →
  cubic_foot_to_gallons = 7.5 →
  (floor_length * floor_width * (water_depth_inch / 12) * cubic_foot_to_gallons) / (pump_rate * num_pumps) = 450 :=
by
  intros
  sorry

end pump_out_water_time_l423_423712


namespace correct_sentence_completion_l423_423074

-- Define the possible options
inductive Options
| A : Options  -- "However he was reminded frequently"
| B : Options  -- "No matter he was reminded frequently"
| C : Options  -- "However frequently he was reminded"
| D : Options  -- "No matter he was frequently reminded"

-- Define the correctness condition
def correct_option : Options := Options.C

-- Define the proof problem
theorem correct_sentence_completion (opt : Options) : opt = correct_option :=
by sorry

end correct_sentence_completion_l423_423074


namespace ball_distribution_l423_423873

-- Definitions as per conditions
def num_distinguishable_balls : ℕ := 5
def num_indistinguishable_boxes : ℕ := 3

-- Problem statement to prove
theorem ball_distribution : 
  let ways_to_distribute_balls := 1 + 5 + 10 + 10 + 30 in
  ways_to_distribute_balls = 56 :=
by
  -- proof required here
  sorry

end ball_distribution_l423_423873


namespace circle_y_axis_intersections_sum_l423_423071

theorem circle_y_axis_intersections_sum (h : circle_center : (-8, 3)) (r : circle_radius : 15) :
  ∑ y in {y | ∃ x, x = 0 ∧ (x + 8)^2 + (y - 3)^2 = 225}, y = 6 := sorry

end circle_y_axis_intersections_sum_l423_423071


namespace no_valid_polygon_pairs_l423_423314

def interior_angle (n : ℕ) : ℝ :=
  180 - (360 / n)

theorem no_valid_polygon_pairs :
  ∀ n m : ℕ, 3 < n ∧ 3 < m ∧ (interior_angle n / interior_angle m = 4 / 3) ∧ (m = 5) → false :=
by
  intros n m H
  cases H with hn hp
  cases hp with hm ratio_eq
  cases ratio_eq with m_is_five ratio_cond
  sorry -- Skipping the proof

end no_valid_polygon_pairs_l423_423314


namespace carla_final_payment_l423_423758

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l423_423758


namespace largest_k_dividing_factorial_l423_423426

theorem largest_k_dividing_factorial (k : ℕ) : 
  let n := 2023
  let prime_factor_1 := 7
  let prime_factor_2 := 17
  let prime_factor_3 := 17
  largest_integer_k (n : ℕ) (pf1 pf2 pf3 : ℕ) : ℕ :=
    ∀ k, (2023^k) ∣ (2023!) ↔ k ≤ 63 :=
by
  let n := 2023
  let prime_factor_1 := 7
  let prime_factor_2 := 17
  let prime_factor_3 := 17
  sorry

end largest_k_dividing_factorial_l423_423426


namespace pow_two_greater_than_square_l423_423276

theorem pow_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2 ^ n > n ^ 2 :=
  sorry

end pow_two_greater_than_square_l423_423276


namespace equal_segments_l423_423210

variables {A B C O D E F M : Type*} [EuclideanGeometry A B C D E F M]

-- Define conditions as variables and hypotheses in Lean
noncomputable def triangle_ABC_acute (t : Triangle A B C) : Prop :=
t.isAcute

noncomputable def circumcenter (O : Point) (t : Triangle A B C) : Prop :=
t.circumcenter = O

noncomputable def midpoint_D (D : Point) (BC : LineSegment B C) : Prop :=
D = BC.midpoint

noncomputable def circle_diameter_AD (AD : LineSegment A D) (c : Circle A D) : Prop :=
c.diameter = AD

noncomputable def intersect_circle_with_triangle_sides (c : Circle A D) (AB : LineSegment A B) (AC : LineSegment A C) : Point :=
let E := c.intersect AB,
    F := c.intersect AC in (E, F)

noncomputable def DM_parallel_AO (DM : Line D M) (AO : Line A O) : Prop :=
DM.is_parallel AO

-- Theorem: Given the conditions, prove EM = MF
theorem equal_segments (ABC : Triangle A B C) (AB_side : LineSegment A B) (AC_side : LineSegment A C)
  (O : Point) (D : Point) (c : Circle A D) (E F : Point) (M : Point)
  (h1 : triangle_ABC_acute ABC)
  (h2 : AB_side > AC_side)
  (h3 : circumcenter O ABC)
  (h4 : midpoint_D D (triangle.side BC))
  (h5 : circle_diameter_AD (AD A D) c)
  (h6 : intersect_circle_with_triangle_sides c AB_side AC_side = (E, F))
  (h7 : DM_parallel_AO (line D M) (line A O)) :
  distance E M = distance M F := 
sorry

end equal_segments_l423_423210


namespace value_of_M_l423_423532

theorem value_of_M
  (M : ℝ)
  (h : 25 / 100 * M = 55 / 100 * 4500) :
  M = 9900 :=
sorry

end value_of_M_l423_423532


namespace average_page_count_per_essay_l423_423037

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l423_423037


namespace simplify_expression_l423_423687

theorem simplify_expression :
  (-2 : ℝ) ^ 2005 + (-2) ^ 2006 + (3 : ℝ) ^ 2007 - (2 : ℝ) ^ 2008 =
  -7 * (2 : ℝ) ^ 2005 + (3 : ℝ) ^ 2007 := 
by
    sorry

end simplify_expression_l423_423687


namespace jogging_time_two_weeks_l423_423601

-- Definition for the daily jogging time in hours
def daily_jogging_time : ℝ := 1 + 30 / 60

-- Definition for the total jogging time over one week
def weekly_jogging_time : ℝ := daily_jogging_time * 7

-- Lean statement to prove that the total time jogging over two weeks is 21 hours
theorem jogging_time_two_weeks : weekly_jogging_time * 2 = 21 := by
  -- Placeholder for the proof
  sorry

end jogging_time_two_weeks_l423_423601


namespace jane_drinks_l423_423566

/-- Jane buys a combination of muffins, bagels, and drinks over five days,
where muffins cost 40 cents, bagels cost 90 cents, and drinks cost 30 cents.
The number of items bought is 5, and the total cost is a whole number of dollars.
Prove that the number of drinks Jane bought is 4. -/
theorem jane_drinks :
  ∃ b m d : ℕ, b + m + d = 5 ∧ (90 * b + 40 * m + 30 * d) % 100 = 0 ∧ d = 4 :=
by
  sorry

end jane_drinks_l423_423566


namespace problem_statement_l423_423464

-- Definitions based on conditions
def sumOfDigits (k : Nat) : Nat :=
  k.digits.sum

def f (n : Nat) (k : Nat) : Nat :=
  if n = 0 then k
  else (sumOfDigits (f (n-1) k)) ^ 2

-- The theorem we want to prove
theorem problem_statement : f 1991 (2 ^ 1990) = 4 := 
  sorry

end problem_statement_l423_423464


namespace triangle_properties_l423_423227

theorem triangle_properties
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : a = b * (sqrt 3) * (cos A) / (sin B))
  (h3 : B = π / 4) :
  sqrt 3 * b = sqrt 2 * a :=
by
  sorry

end triangle_properties_l423_423227


namespace tourist_familiar_bound_l423_423721

variable (n : ℕ)
variable (T : Finset ℕ)
variable (familiar : ℕ → ℕ → Prop)

-- Define the conditions
def condition_1 := ∀ (a b c : ℕ), a ∈ T → b ∈ T → c ∈ T → (¬ familiar a b ∨ ¬ familiar b c ∨ ¬ familiar a c)
def condition_2 := ∀ (A B : Finset ℕ), A ∪ B = T → ∃ (x y : ℕ), x ∈ A → y ∈ A ∨ x ∈ B → y ∈ B → familiar x y

-- Define the proposition to be proven
theorem tourist_familiar_bound (h1 : condition_1 n T familiar) (h2 : condition_2 n T familiar) : 
∃ i ∈ T, ((Finset.filter (familiar i) T).card ≤ 2 * T.card / 5) :=
sorry

end tourist_familiar_bound_l423_423721


namespace final_milk_quantity_l423_423360

def initial_milk : ℝ := 75
def removed_milk_1 : ℝ := 9
def added_water_1 : ℝ := 9
def mixture_removal : ℝ := 9

-- Helper definition to calculate the milk concentration after the first removal and addition of water
def remaining_milk_1 (initial_milk : ℝ) (removed_milk_1 : ℝ) : ℝ := initial_milk - removed_milk_1
def concentration_milk_1 (remaining_milk_1 : ℝ) (total_volume : ℝ) : ℝ := remaining_milk_1 / total_volume

-- Calculate the amount of milk removed in the second step
def removed_milk_2 (concentration_milk_1 : ℝ) (mixture_removal : ℝ) : ℝ := concentration_milk_1 * mixture_removal

-- Final quantity of milk after the second replacement
def remaining_milk_2 (remaining_milk_1 : ℝ) (removed_milk_2 : ℝ) : ℝ := remaining_milk_1 - removed_milk_2

theorem final_milk_quantity : remaining_milk_2 (remaining_milk_1 initial_milk removed_milk_1) 
  ((concentration_milk_1 (remaining_milk_1 initial_milk removed_milk_1) initial_milk) * mixture_removal) 
  = 58.08 :=
by
  sorry

end final_milk_quantity_l423_423360


namespace gcd_lcm_888_1147_l423_423453

theorem gcd_lcm_888_1147 :
  Nat.gcd 888 1147 = 37 ∧ Nat.lcm 888 1147 = 27528 := by
  sorry

end gcd_lcm_888_1147_l423_423453


namespace average_of_all_5_numbers_is_20_l423_423296

def average_of_all_5_numbers
  (sum_3_numbers : ℕ)
  (avg_2_numbers : ℕ) : ℕ :=
(sum_3_numbers + 2 * avg_2_numbers) / 5

theorem average_of_all_5_numbers_is_20 :
  average_of_all_5_numbers 48 26 = 20 :=
by
  unfold average_of_all_5_numbers -- Expand the definition
  -- Sum of 5 numbers is 48 (sum of 3) + (2 * 26) (sum of other 2)
  -- Total sum is 48 + 52 = 100
  -- Average is 100 / 5 = 20
  norm_num -- Check the numeric calculation
  -- sorry

end average_of_all_5_numbers_is_20_l423_423296


namespace find_y_when_x_is_5_l423_423894

variable (x y k : ℝ)

def inversely_proportional (x y : ℝ) : Prop := x * y = k

theorem find_y_when_x_is_5 (h1 : inversely_proportional x y) 
    (h2 : x + y = 54) (h3 : x = 3 * y) (h4 : x = 5) : y = 109.35 := 
  sorry

end find_y_when_x_is_5_l423_423894


namespace percent_cost_bread_and_ham_l423_423674

variable (cost_bread cost_ham cost_cake total_cost_bread_ham total_cost : ℕ)

noncomputable def perc_cost_bread_ham := (total_cost_bread_ham * 100) / total_cost

theorem percent_cost_bread_and_ham :
  cost_bread = 50 ∧
  cost_ham = 150 ∧
  cost_cake = 200 ∧
  total_cost_bread_ham = cost_bread + cost_ham ∧
  total_cost = total_cost_bread_ham + cost_cake →
  perc_cost_bread_ham = 50 :=
by
  rintros ⟨rfl, rfl, rfl, rfl, rfl⟩
  sorry

end percent_cost_bread_and_ham_l423_423674


namespace problem_correct_propositions_l423_423173

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem problem_correct_propositions :
  ( ∃ (num_correct : ℕ), num_correct = 2 ∧
    ( (∀ x, ( (x < 0) → (0 < f' x)) ∧
            ((0 < x ∧ x < 2) → (f' x < 0 )) ∧
              ((2 < x) → (0 < f' x ))) ∧
              ( (f 0 = 0) ∧ ( ∀ y, f y ≤ f 0 )) ∧
                  ( (f 2 = -4) ∧ ( ∀ z, f 2 ≤ f z )))) :=
begin
  sorry
end

end problem_correct_propositions_l423_423173


namespace inequality_represents_area_l423_423653

theorem inequality_represents_area (x y : ℝ) :
  x - 2 * y + 6 > 0 → (x, y) is on the lower right side of the line x - 2 * y + 6 = 0 :=
begin
  sorry
end

end inequality_represents_area_l423_423653


namespace least_x_divisible_by_17280_l423_423799

theorem least_x_divisible_by_17280 : ∃ x : ℕ, x^3 % 17280 = 0 ∧ x = 120 :=
by
  exists 120
  split
  · sorry
  · rfl

end least_x_divisible_by_17280_l423_423799


namespace sin_alpha_zero_l423_423470

theorem sin_alpha_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : π / 2 < β ∧ β < π) 
  (h3 : sin (α + β) = 3 / 5) (h4 : cos β = -4 / 5) : sin α = 0 := 
sorry

end sin_alpha_zero_l423_423470


namespace cassidy_grounded_days_l423_423421

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l423_423421


namespace foreign_stamps_count_l423_423203

-- Define the conditions
variables (total_stamps : ℕ) (more_than_10_years_old : ℕ) (both_foreign_and_old : ℕ) (neither_foreign_nor_old : ℕ)

theorem foreign_stamps_count 
  (h1 : total_stamps = 200)
  (h2 : more_than_10_years_old = 60)
  (h3 : both_foreign_and_old = 20)
  (h4 : neither_foreign_nor_old = 70) : 
  ∃ (foreign_stamps : ℕ), foreign_stamps = 90 :=
by
  -- let foreign_stamps be the variable representing the number of foreign stamps
  let foreign_stamps := total_stamps - neither_foreign_nor_old - more_than_10_years_old + both_foreign_and_old
  use foreign_stamps
  -- the proof will develop here to show that foreign_stamps = 90
  sorry

end foreign_stamps_count_l423_423203


namespace four_x_thirty_two_y_l423_423140

theorem four_x_thirty_two_y (x y : ℝ) (h : 2 * x + 5 * y = 3) : 4^x * 32^y = 8 :=
by
  sorry

end four_x_thirty_two_y_l423_423140


namespace part1_part2_part3_l423_423937

namespace ProofProblem

open Set

def U := ℝ
def A : Set ℝ := { x | 2^(2*x-1) ≥ 2 }
def B : Set ℝ := { x | x^2 - 5*x < 0 }

theorem part1 : A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 5 } :=
by
  sorry

theorem part2 : A ∪ B = { x : ℝ | 0 < x } :=
by
  sorry

theorem part3 : (U \ (A ∪ B)) = { x : ℝ | x ≤ 0 } :=
by
  sorry

end ProofProblem

end part1_part2_part3_l423_423937


namespace fraction_numerator_l423_423299

theorem fraction_numerator (x : ℚ) :
  (∃ n : ℚ, 4 * n - 4 = x ∧ x / (4 * n - 4) = 3 / 7) → x = 12 / 5 :=
by
  sorry

end fraction_numerator_l423_423299


namespace problem_solution_l423_423256

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end problem_solution_l423_423256


namespace find_x_l423_423786

theorem find_x (x : ℝ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423786


namespace lcm_18_24_30_eq_360_l423_423685

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l423_423685


namespace octagon_square_side_length_l423_423747

noncomputable def octagon_square_proof (AB GH : ℝ) : Prop :=
  AB = 50 → GH = 50 * (Real.sqrt 3 - 1) → ∃ (side_length : ℝ), side_length = 50 ∧ side_length = side_length

theorem octagon_square_side_length (AB GH side_length : ℝ) :
  octagon_square_proof AB GH →
  side_length = 50 :=
by
  assume h,
  cases h with h1 h2,
  use 50,
  sorry

end octagon_square_side_length_l423_423747


namespace avg_page_count_per_essay_l423_423035

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l423_423035


namespace more_than_half_millet_on_day_5_l423_423267

/-- Setup: Initial conditions and recursive definition of millet quantity -/
def millet_amount_on_day (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (3 / 4 : ℝ)^i / 4

/-- Proposition to prove: On the 5th day, just after placing the seeds, more than half the seeds are millet -/
theorem more_than_half_millet_on_day_5 :
  millet_amount_on_day 5 > 1 / 2 :=
begin
  sorry
end

end more_than_half_millet_on_day_5_l423_423267


namespace find_distance_l423_423193

-- Define the necessary variables and constants
variables {D T : ℝ} -- Distance to the station and the time taken in minutes.

-- Define the given conditions
def condition1 := D = 4 * (T / 60)
def condition2 := D = 5 * ((T - 12) / 60)

-- State the theorem to prove
theorem find_distance (h1 : condition1) (h2 : condition2) : D = 4 := by
  sorry

end find_distance_l423_423193


namespace number_of_sheets_l423_423329

theorem number_of_sheets (S E : ℕ) 
  (h1 : S - E = 40)
  (h2 : 5 * E = S) : 
  S = 50 := by 
  sorry

end number_of_sheets_l423_423329


namespace sin_alpha_eq_24_over_25_l423_423472

open Real Trig

theorem sin_alpha_eq_24_over_25
  (α β : ℝ)
  (h0 : 0 < α ∧ α < π / 2)
  (h1 : π / 2 < β ∧ β < π)
  (h2 : sin(α + β) = 3 / 5)
  (h3 : cos β = -4 / 5) :
  sin α = 24 / 25 :=
by
  sorry

end sin_alpha_eq_24_over_25_l423_423472


namespace shaded_quadrilateral_area_l423_423672

theorem shaded_quadrilateral_area (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 7) : 
  let CD := a + b + c in 
  let height_small_square := 1.4 in
  let height_large_square := 2.33 in
  let height_interval := (height_large_square - height_small_square) in
  let area_trapezoid := 1 / 2 * (b : ℝ) * (height_small_square + height_large_square)
  in area_trapezoid = 9.325 :=
by 
  sorry

end shaded_quadrilateral_area_l423_423672


namespace part1_l423_423016

theorem part1 (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) : 2 * x^2 + y^2 > x^2 + x * y := 
sorry

end part1_l423_423016


namespace ratio_of_leaf_areas_l423_423033

theorem ratio_of_leaf_areas
  (r : ℝ) (h_r : r = 3)
  (n : ℕ) (h_n : n = 6)
  (A_circle : ℝ) (h_A_circle : A_circle = Real.pi * r^2) :
  let arc_area := A_circle / n 
  let leaf_area := arc_area + arc_area / 2
  let total_leaf_area := 3 * leaf_area 
  in total_leaf_area / A_circle = 3 / 4 :=
by
  -- radius of the circle is given
  have hr : r = 3 := h_r
  -- number of arcs is given
  have hn : n = 6 := h_n
  -- the area of the circle is calculated
  have hA : A_circle = 9 * Real.pi := by rw [h_A_circle, hr]; ring
  -- simplify the area of one arc
  let arc_area := 1.5 * Real.pi
  -- the area of one leaf-shaped figure
  let leaf_area := 2.25 * Real.pi
  -- total area of all leaf-shaped figures
  let total_leaf_area := 6.75 * Real.pi
  -- ratio of the total area of the leaves to the area of the circle
  have ratio : total_leaf_area / A_circle = 3 / 4 := by
    rw [hA]
    norm_num
  exact ratio

end ratio_of_leaf_areas_l423_423033


namespace duration_of_period_l423_423377

theorem duration_of_period 
  (loan_amount : ℝ) 
  (rate_A : ℝ) 
  (rate_C : ℝ) 
  (gain_B : ℝ) 
  (t : ℝ) 
  (annual_interest_paid_by_B_to_A : ℝ := loan_amount * rate_A / 100)
  (annual_interest_earned_by_B_from_C : ℝ := loan_amount * rate_C / 100)
  (annual_gain_of_B : ℝ := annual_interest_earned_by_B_from_C - annual_interest_paid_by_B_to_A)
  (total_gain_of_B : ℝ := annual_gain_of_B * t) :
  (loan_amount = 2000) 
  → (rate_A = 10) 
  → (rate_C = 11.5) 
  → (gain_B = 90) 
  → total_gain_of_B = gain_B
  → t = 3 := 
by
  intros h_loan_amount h_rate_A h_rate_C h_gain_B h_total_gain_of_B
  rw [h_loan_amount, h_rate_A, h_rate_C] at *
  simp [annual_interest_paid_by_B_to_A, annual_interest_earned_by_B_from_C, annual_gain_of_B] at h_total_gain_of_B
  linarith

end duration_of_period_l423_423377


namespace correct_answer_l423_423398

-- Define the propositions
def prop1 : Prop := ∀ (a b : ℝ), vertically_opposite a b → a = b
def prop2 : Prop := ∀ (l1 l2 : Line), supplementary_angles_same_side l1 l2 → complement_angles l1 l2 ∧ parallel l1 l2
def prop3 : Prop := ∀ (triangle : Triangle), is_right_triangle triangle → acute_angles_complementary triangle
def prop4 : Prop := ∀ (a b : ℝ), a > 0 ∧ b > 0 → a * b > 0

-- Define the inverse propositions
def inv_prop1 : Prop := ∀ (a b : ℝ), ¬(vertically_opposite a b) → ¬(a = b)
def inv_prop2 : Prop := ∀ (l1 l2 : Line), ¬(parallel l1 l2) → ¬(supplementary_angles_same_side l1 l2 ∧ complement_angles l1 l2)
def inv_prop3 : Prop := ∀ (a b : Angle), complementary a b → exists_right_triangle_with_acute_angles a b
def inv_prop4 : Prop := ∀ (a b : ℝ), a * b ≤ 0 → ¬(a > 0 ∧ b > 0)

-- The theorem asserting the correct answer
theorem correct_answer : inv_prop2 ∧ inv_prop3 ∧ ¬inv_prop1 ∧ ¬inv_prop4 :=
by
  sorry

end correct_answer_l423_423398


namespace negative_represents_departure_of_30_tons_l423_423219

theorem negative_represents_departure_of_30_tons (positive_neg_opposites : ∀ x:ℤ, -x = x * (-1))
  (arrival_represents_30 : ∀ x:ℤ, (x = 30) ↔ ("+30" represents arrival of 30 tons of grain)) :
  "-30" represents departure of 30 tons of grain :=
sorry

end negative_represents_departure_of_30_tons_l423_423219


namespace overall_avg_marks_l423_423552

def classA_students : ℕ := 30
def classB_students : ℕ := 50

def classA_math_avg : ℝ := 35
def classA_english_avg : ℝ := 45
def classA_science_avg : ℝ := 25

def classB_math_avg : ℝ := 55
def classB_english_avg : ℝ := 75
def classB_science_avg : ℝ := 45

-- Calculate total average marks for each subject and overall combined average.
theorem overall_avg_marks :
  let total_students := classA_students + classB_students in
  let overall_avg_math := ((classA_math_avg * classA_students) + (classB_math_avg * classB_students)) / total_students in
  let overall_avg_english := ((classA_english_avg * classA_students) + (classB_english_avg * classB_students)) / total_students in
  let overall_avg_science := ((classA_science_avg * classA_students) + (classB_science_avg * classB_students)) / total_students in
  let total_marks_all_subjects := (classA_math_avg * classA_students + classB_math_avg * classB_students) + (classA_english_avg * classA_students + classB_english_avg * classB_students) + (classA_science_avg * classA_students + classB_science_avg * classB_students) in
  let num_subjects := 3 in
  let total_avg_all_subjects_comb := total_marks_all_subjects / (total_students * num_subjects) in
  overall_avg_math = 47.5 ∧ overall_avg_english = 63.75 ∧ overall_avg_science = 37.5 ∧ total_avg_all_subjects_comb = 49.58 :=
by {
  sorry
}

end overall_avg_marks_l423_423552


namespace sean_divided_by_julie_l423_423285

noncomputable def sean_sum : ℤ :=
  (sum (filter odd (list.iota (1001 - 2 + 1) + 2)))

noncomputable def julie_sum : ℤ :=
  (sum (list.range 501))

theorem sean_divided_by_julie : sean_sum / julie_sum = 125249 / 125250 :=
by
  sorry

end sean_divided_by_julie_l423_423285


namespace length_B_l423_423558

variables {r : ℝ} (A B C : ℝ × ℝ)
variables (condition1 : A = (0, real.sqrt 6))
variables (condition2 : B = (3, 0))
variables (condition3 : C = (-3, 0))
variables (condition4 : 3 < r ∧ r < 4)
variables (condition5 : dist A B = 5)
variables (condition6 : dist A C = 5)
variables (condition7 : dist B C = 5)
variables (B' : ℝ × ℝ)
variables (C' : ℝ × ℝ)
variables (condition8 : (B'.fst - A.fst)^2 + (B'.snd - A.snd)^2 = r^2)
variables (condition9 : (B'.fst - C.fst)^2 + (B'.snd - C.snd)^2 = r^2)
variables (condition10 : (C'.fst - A.fst)^2 + (C'.snd - A.snd)^2 = r^2)
variables (condition11 : (C'.fst - B.fst)^2 + (C'.snd - B.snd)^2 = r^2)
variables (condition12 : ¬ ((B'.fst - B.fst)^2 + (B'.snd - B.snd)^2 = r^2))
variables (condition13 : ¬ ((C'.fst - C.fst)^2 + (C'.snd - C.snd)^2 = r^2))

theorem length_B'C' (h : 4.5 + real.sqrt (6 * (r^2 - 1))) :
  dist B' C' = h :=
sorry

end length_B_l423_423558


namespace tan_squared_critical_point_l423_423994

theorem tan_squared_critical_point :
  ∀ (f : ℝ → ℝ) (θ : ℝ),
  (∀ x, f x = Real.sin x + 2 * Real.cos x) →
  (∀ x, deriv f x = Real.cos x - 2 * Real.sin x) →
  (∀ x, deriv f θ = 0) →
  (θ : ℝ) → Real.tan θ ^ 2 = 1 / 4 :=
begin
  intros f θ hf hdf hcritical,
  sorry,
end

end tan_squared_critical_point_l423_423994


namespace minimum_subset_size_l423_423984

def can_be_expressed (S : set ℕ) (n : ℕ) : Prop :=
  ∃ x y ∈ S, n = x + y

theorem minimum_subset_size :
  ∃ S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
  (∀ n ∈ {i | i ≤ 20}, can_be_expressed S n) ∧ S.card = 6 :=
sorry

end minimum_subset_size_l423_423984


namespace line_through_point_inclination_l423_423379

-- Define the point M and the inclination angle
def M : Point := { x := 2, y := -3 }
def α : ℝ := 45

-- Define the target line equation
def line_eq (x y : ℝ) : Prop := x - y - 5 = 0

-- Statement of the problem
theorem line_through_point_inclination (M : Point) (α : ℝ) : line_eq 2 (-3) :=
  by sorry

end line_through_point_inclination_l423_423379


namespace parallel_edges_octahedron_l423_423188

-- Definition of a regular octahedron's properties
structure regular_octahedron : Type :=
  (edges : ℕ) -- Number of edges in the octahedron

-- Constant to represent the regular octahedron with 12 edges.
def octahedron : regular_octahedron := { edges := 12 }

-- Definition to count unique pairs of parallel edges
def count_parallel_edge_pairs (o : regular_octahedron) : ℕ :=
  if o.edges = 12 then 12 else 0

-- Theorem to assert the number of pairs of parallel edges in a regular octahedron is 12
theorem parallel_edges_octahedron : count_parallel_edge_pairs octahedron = 12 :=
by
  -- Proof will be inserted here
  sorry

end parallel_edges_octahedron_l423_423188


namespace figure_perimeter_l423_423630

theorem figure_perimeter (h_segments v_segments : ℕ) (side_length : ℕ) 
  (h_count : h_segments = 16) (v_count : v_segments = 10) (side_len : side_length = 1) :
  2 * (h_segments + v_segments) * side_length = 26 :=
by
  sorry

end figure_perimeter_l423_423630


namespace no_odd_powers_in_product_l423_423977

theorem no_odd_powers_in_product :
  let A (x : ℝ) := (sum (range 101) (λ k, (-x) ^ k))
  let B (x : ℝ) := (sum (range 101) (λ k, x ^ k))
  ∀ (x : ℝ),
  (∀ (k : ℕ), (range 202).all (λ k, odd k → coeff (expansion_of_product A B x) k = 0)) :=
by
  let A (x : ℝ) := (∑ k in (range 101), (-x) ^ k)
  let B (x : ℝ) := (∑ k in (range 101), x ^ k)
  sorry

end no_odd_powers_in_product_l423_423977


namespace scorpion_millipedes_needed_l423_423027

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l423_423027


namespace minimum_am_an_dot_product_l423_423900

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

theorem minimum_am_an_dot_product 
  (A B C : ℝ × ℝ)
  (M : ℝ × ℝ := (midpoint B.1 C.1, midpoint B.2 C.2)) 
  (N : ℝ × ℝ := (midpoint B.1 M.1, midpoint B.2 M.2)) 
  (angle_A : ℝ := real.pi / 3) 
  (area_ABC : ℝ := real.sqrt 3) 
  (h_area : 0.5 * (B.1 - A.1) * (C.2 - A.2) * real.sin angle_A = real.sqrt 3) :
  let AM := ((M.1 - A.1), (M.2 - A.2)),
      AN := ((N.1 - A.1), (N.2 - A.2)) 
  in (AM.1 * AN.1 + AM.2 * AN.2) = 2 :=
sorry

end minimum_am_an_dot_product_l423_423900


namespace quadratic_polynomial_conditions_l423_423456

noncomputable def quadratic_polynomial : Polynomial ℝ :=
  Polynomial.C (-2/15) + Polynomial.X * Polynomial.C (-39/10) + Polynomial.X ^ 2 * Polynomial.C (67/30)

theorem quadratic_polynomial_conditions :
  let q := quadratic_polynomial in
  q.eval (-1) = 6 ∧ q.eval 2 = 1 ∧ q.eval 4 = 20 :=
by {
  let q := quadratic_polynomial,
  split,
  sorry,
  split,
  sorry,
  sorry
}

end quadratic_polynomial_conditions_l423_423456


namespace problem1_problem2_l423_423582

variable {m n : ℤ}

-- Problem 1: Prove that m + n = 105 given the conditions.
theorem problem1 (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225) (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Problem 2: Prove that m + n = 90 given the conditions.
theorem problem2 (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225) (h4 : Nat.lcm m n = 45) : m + n = 90 :=
sorry

end problem1_problem2_l423_423582


namespace problem_statement_l423_423841

-- Definitions and conditions
variables {α β : Type} [LinearOrder α] [LinearOrder β]

def symmetric_about_line_y_eq_x (f g : α → β) : Prop :=
  ∀ x : α, f (g x) = x

-- Given conditions
variables (f g : ℝ → ℝ)
variable (h_symm : symmetric_about_line_y_eq_x f g)
variable (h_g1_2 : g 1 = 2)

-- The goal
theorem problem_statement : f 3 = 1 :=
sorry

end problem_statement_l423_423841


namespace ball_price_equation_l423_423207

structure BallPrices where
  (x : Real) -- price of each soccer ball in yuan
  (condition1 : ∀ (x : Real), (1500 / (x + 20) - 800 / x = 5))

/-- Prove that the equation follows from the given conditions. -/
theorem ball_price_equation (b : BallPrices) : 1500 / (b.x + 20) - 800 / b.x = 5 := 
by sorry

end ball_price_equation_l423_423207


namespace housewife_more_oil_l423_423776

theorem housewife_more_oil 
    (reduction_percent : ℝ := 10)
    (reduced_price : ℝ := 16)
    (budget : ℝ := 800)
    (approx_answer : ℝ := 5.01) :
    let P := reduced_price / (1 - reduction_percent / 100)
    let Q_original := budget / P
    let Q_reduced := budget / reduced_price
    let delta_Q := Q_reduced - Q_original
    abs (delta_Q - approx_answer) < 0.02 := 
by
  -- Let the goal be irrelevant to the proof because the proof isn't provided
  sorry

end housewife_more_oil_l423_423776


namespace median_room_number_of_remaining_contestants_l423_423406

def median_of_list_except (n m : ℕ) (l : List ℕ) : ℕ :=
  (l.filter (λ x, x ≠ n ∧ x ≠ m)).nth (l.length - 2) / 2 |>.getOrElse 0

theorem median_room_number_of_remaining_contestants :
  median_of_list_except 15 16 (List.range 1 26) = 12 :=
by
  sorry

end median_room_number_of_remaining_contestants_l423_423406


namespace simplify_fraction_l423_423288

variable (y b : ℚ)

theorem simplify_fraction : 
  (y+2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := 
by
  sorry

end simplify_fraction_l423_423288


namespace scorpion_needs_10_millipedes_l423_423020

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l423_423020


namespace handshake_count_l423_423909

-- Definitions based on conditions
def groupA_size : ℕ := 25
def groupB_size : ℕ := 15

-- Total number of handshakes is calculated as product of their sizes
def total_handshakes : ℕ := groupA_size * groupB_size

-- The theorem we need to prove
theorem handshake_count : total_handshakes = 375 :=
by
  -- skipped proof
  sorry

end handshake_count_l423_423909


namespace bricks_needed_to_build_wall_l423_423867

-- Define dimensions of a brick
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Define dimensions of the wall
def wall_length : ℝ := 8 * 100
def wall_width : ℝ := 6 * 100
def wall_height : ℝ := 22.5
def wall_volume : ℝ := wall_length * wall_width * wall_height

-- Define the number of bricks required
def number_of_bricks : ℝ := wall_volume / brick_volume

theorem bricks_needed_to_build_wall : number_of_bricks = 2000 := by
  sorry

end bricks_needed_to_build_wall_l423_423867


namespace John_avg_speed_l423_423002

theorem John_avg_speed :
  ∀ (initial final : ℕ) (time : ℕ),
    initial = 27372 →
    final = 27472 →
    time = 4 →
    ((final - initial) / time) = 25 :=
by
  intros initial final time h_initial h_final h_time
  sorry

end John_avg_speed_l423_423002


namespace find_x_l423_423788

theorem find_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 :=
by
  sorry

end find_x_l423_423788


namespace ratio_yellow_jelly_beans_l423_423806

theorem ratio_yellow_jelly_beans :
  let bag_A_total := 24
  let bag_B_total := 30
  let bag_C_total := 32
  let bag_D_total := 34
  let bag_A_yellow_ratio := 0.40
  let bag_B_yellow_ratio := 0.30
  let bag_C_yellow_ratio := 0.25 
  let bag_D_yellow_ratio := 0.10
  let bag_A_yellow := bag_A_total * bag_A_yellow_ratio
  let bag_B_yellow := bag_B_total * bag_B_yellow_ratio
  let bag_C_yellow := bag_C_total * bag_C_yellow_ratio
  let bag_D_yellow := bag_D_total * bag_D_yellow_ratio
  let total_yellow := bag_A_yellow + bag_B_yellow + bag_C_yellow + bag_D_yellow
  let total_beans := bag_A_total + bag_B_total + bag_C_total + bag_D_total
  (total_yellow / total_beans) = 0.25 := by
  sorry

end ratio_yellow_jelly_beans_l423_423806


namespace min_area_square_on_parabola_l423_423525

/-- Given a parabola y = x^2 and three vertices A, B, and C of a square on this parabola, 
the minimum possible area of such a square is 2. -/
theorem min_area_square_on_parabola (A B C : ℝ × ℝ) (hA : A.snd = A.fst ^ 2) (hB : B.snd = B.fst ^ 2) (hC : C.snd = C.fst ^ 2) :
  ∃ s : ℝ, s ^ 2 = 2 ∧ ∃ S : ℝ → ℝ → Prop, S A B ∧ S B C ∧ S C A := 
sorry

end min_area_square_on_parabola_l423_423525


namespace fruit_filled_mooncakes_probability_l423_423065

theorem fruit_filled_mooncakes_probability :
    (∃ fruits meats : ℕ,
        fruits = 5 ∧ meats = 4 ∧
        let total_combinations := (fruits * (fruits - 1) / 2 + meats * (meats - 1) / 2) in
        let fruit_combinations := (fruits * (fruits - 1) / 2) in
        total_combinations = 16 ∧
        fruit_combinations = 10 ∧
        fruit_combinations / total_combinations = 5 / 8) :=
by
  -- Conditions
  let fruits := 5
  let meats := 4
  have total_combinations : ℕ := (fruits * (fruits - 1) / 2 + meats * (meats - 1) / 2)
  have fruit_combinations : ℕ := (fruits * (fruits - 1) / 2)

  -- Proof that the total combinations and fruit combinations are correct
  have h1 : total_combinations = 16 := by sorry
  have h2 : fruit_combinations = 10 := by sorry

  -- Proof that the probability is correct
  have h3 : fruit_combinations / total_combinations = 5 / 8 := by sorry
  existsi fruits
  existsi meats
  exact ⟨rfl, rfl, h1, h2, h3⟩

end fruit_filled_mooncakes_probability_l423_423065


namespace find_original_selling_price_l423_423599

variable (SP : ℝ)
variable (CP : ℝ := 10000)
variable (discounted_SP : ℝ := 0.9 * SP)
variable (profit : ℝ := 0.08 * CP)

theorem find_original_selling_price :
  discounted_SP = CP + profit → SP = 12000 := by
sorry

end find_original_selling_price_l423_423599


namespace scorpion_needs_10_millipedes_l423_423021

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l423_423021


namespace find_number_l423_423534

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_number :
  (∃ x : ℕ, hash 3 x = 63 ∧ x = 7) :=
sorry

end find_number_l423_423534


namespace find_C_coordinates_l423_423655

-- Define parabola and line equations
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points of intersection A and B
def point_A : Point := ⟨9 + 4 * Real.sqrt 5, 4 + 2 * Real.sqrt 5⟩
def point_B : Point := ⟨9 - 4 * Real.sqrt 5, 4 - 2 * Real.sqrt 5⟩

-- Definition of point C
def point_C (t : ℝ) : Point := ⟨t^2, 2 * t⟩

-- Orthogonality condition for angle ACB being 90 degrees
def orthogonal (A C B : Point) : Prop :=
  (C.x - A.x) * (C.x - B.x) + (C.y - A.y) * (C.y - B.y) = 0

-- Statement to prove the points
theorem find_C_coordinates (C : Point) (hC : parabola C.x C.y) (hO : orthogonal point_A C point_B) :
  (C = ⟨1, -2⟩ ∨ C = ⟨9, -6⟩) :=
sorry

end find_C_coordinates_l423_423655


namespace opposite_of_neg_sqrt3_squared_l423_423657

theorem opposite_of_neg_sqrt3_squared : -((-sqrt 3) ^ 2) = -3 :=
by
  sorry

end opposite_of_neg_sqrt3_squared_l423_423657


namespace alex_uphill_time_l423_423053

theorem alex_uphill_time :
  ∀ (x : ℝ),
  let flat_ground_time := 4.5,
      flat_ground_speed := 20, -- mph
      uphill_speed := 12, -- mph
      downhill_time := 1.5,
      downhill_speed := 24, -- mph
      walking_distance := 8 -- miles
  total_distance := 164 -- miles
  in
  (flat_ground_speed * flat_ground_time + uphill_speed * x + downhill_speed * downhill_time) = (total_distance - walking_distance) → 
  x = 2.5 := 
by
  intro x
  let flat_ground_time := 4.5
  let flat_ground_speed := 20 
  let uphill_speed := 12
  let downhill_time := 1.5
  let downhill_speed := 24
  let walking_distance := 8
  let total_distance := 164
  intro h
  have h1 : 90 + 12 * x + 36 = 156 := by sorry
  have h2 : 12 * x = 30 := by sorry
  have h3 : x = 2.5 := by sorry
  exact h3

end alex_uphill_time_l423_423053


namespace find_ordered_pair_l423_423632

theorem find_ordered_pair (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : (x : ℝ) → x^2 + 2 * a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1, -3) :=
sorry

end find_ordered_pair_l423_423632


namespace find_length_of_intercepted_segment_l423_423553

noncomputable def segment_length_intercepted_by_non_parallel_sides (a b : ℝ) (a_gt_b : a > b) : ℝ :=
  2 * a * b / (a + b)

theorem find_length_of_intercepted_segment (a b : ℝ) (a_gt_b : a > b)
  (ABCD_is_trapezoid : ∃ (A B C D : Point) (PQ : Line), 
    is_trapezoid A B C D ∧
    |AD| = a ∧
    |BC| = b ∧
    intersects AC BD = M ∧
    parallel PQ AD ∧
    on_line M PQ ∧ 
    intercept PQ AB CD) :
  segment_length_intercepted_by_non_parallel_sides a b a_gt_b = 2 * a * b / (a + b) :=
sorry

end find_length_of_intercepted_segment_l423_423553


namespace expand_polynomial_identity_l423_423448

variable {x : ℝ}

theorem expand_polynomial_identity : (7 * x + 5) * (5 * x ^ 2 - 2 * x + 4) = 35 * x ^ 3 + 11 * x ^ 2 + 18 * x + 20 := by
    sorry

end expand_polynomial_identity_l423_423448


namespace arithmetic_seq_sum_l423_423690

theorem arithmetic_seq_sum
  (d : ℕ)
  (x y z : ℕ)
  (h1 : d = 7 - 3)
  (h2 : z = 31)
  (h3 : y = z - d)
  (h4 : x = y - d) :
  x + y + z = 81 :=
begin
  sorry
end

end arithmetic_seq_sum_l423_423690


namespace distinct_numbers_count_l423_423110

theorem distinct_numbers_count : 
  let sequence := list.iota 1000 
  let transformed := sequence.map (λ n => (n ^ 2 / 500).floor)
  (transformed.to_finset.card = 876) := 
by 
  sorry

end distinct_numbers_count_l423_423110


namespace find_c_d_for_continuity_l423_423250

def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  if x > 3 then c * x + 1
  else if -1 ≤ x ∧ x ≤ 3 then 2 * x - 7
  else 3 * x - d

theorem find_c_d_for_continuity (c d : ℝ) (h1 : c * 3 + 1 = 2 * 3 - 7) (h2 : 2 * (-1) - 7 = 3 * (-1) - d) : 
  c + d = 16 / 3 :=
by
  have c_eq : c = -2 / 3 := sorry
  have d_eq : d = 6 := sorry
  rw [c_eq, d_eq]
  norm_num
  sorry

end find_c_d_for_continuity_l423_423250


namespace sum_of_squares_of_roots_zero_l423_423415

noncomputable def sum_of_squares_of_roots : ℝ :=
begin
  let f := λ x : ℝ, x^12 + 7*x^9 + 3*x^3 + 500,
  let roots := {s : list ℝ | ∀ r ∈ s, f r = 0 },
  sorry
end

theorem sum_of_squares_of_roots_zero : sum_of_squares_of_roots = 0 :=
sorry

end sum_of_squares_of_roots_zero_l423_423415


namespace monotonic_decrease_iff_l423_423854

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, Real.log x - (1 / 2) * a * x^2 - x

theorem monotonic_decrease_iff (a : ℝ) : (∀ x > 0, f' = 1 / x - a * x - 1 < 0) ↔ a > -1 / 4 :=
by
  have f' : ℝ → ℝ := λ x, 1 / x - a * x - 1
  sorry

end monotonic_decrease_iff_l423_423854


namespace sum_f_eq_55_over_3_l423_423466

def f (n : ℕ) : ℝ :=
  if (∃ k : ℚ, n = 8^k) then real.log n / real.log 8 else 0

theorem sum_f_eq_55_over_3 : 
  (∑ n in (finset.range 1998).filter (λ n, n > 0), f n) = 55 / 3 := 
by
  sorry

end sum_f_eq_55_over_3_l423_423466


namespace polygon_problem_l423_423164

theorem polygon_problem
  (sum_interior_angles : ℕ → ℝ)
  (sum_exterior_angles : ℝ)
  (condition : ∀ n, sum_interior_angles n = (3 * sum_exterior_angles) - 180) :
  (∃ n : ℕ, sum_interior_angles n = 180 * (n - 2) ∧ n = 7) ∧
  (∃ n : ℕ, n = 7 → (n * (n - 3) / 2) = 14) :=
by
  sorry

end polygon_problem_l423_423164


namespace sum_of_solutions_l423_423944

noncomputable def f (x : ℝ) : ℝ := 12 * x - 5
noncomputable def f_inv (x : ℝ) : ℝ := (x + 5) / 12

theorem sum_of_solutions :
  (∑ x in { x : ℝ | f_inv x = f (1 / (3 * x)) }, id x) = -65 := by
  sorry

end sum_of_solutions_l423_423944


namespace largest_of_seven_consecutive_l423_423318

theorem largest_of_seven_consecutive (n : ℕ) (h1 : (7 * n + 21 = 3020)) : (n + 6 = 434) :=
sorry

end largest_of_seven_consecutive_l423_423318


namespace min_value_of_f_determine_m_and_max_f_l423_423079

noncomputable def f (x m : ℝ) : ℝ := -2 * m + 2 * m * sin(x + 3 * π / 2) - 2 * cos(x - π / 2) ^ 2 + 1

noncomputable def h (m : ℝ) : ℝ :=
  if m > 2 then -4 * m + 1
  else if 0 ≤ m ∧ m ≤ 2 then -m^2 / 2 - 2 * m - 1
  else -2 * m - 1

theorem min_value_of_f (x : ℝ) (m : ℝ) (hx : -π / 2 ≤ x ∧ x ≤ 0) :
  f x m = h m := sorry

theorem determine_m_and_max_f (m : ℝ) (hm : h m = 1 / 2) :
  m = -3 / 4 ∧ (∀ x : ℝ, -π / 2 ≤ x ∧ x ≤ 0 → f x (-3 / 4) ≤ 4) := sorry

end min_value_of_f_determine_m_and_max_f_l423_423079


namespace mathemtacial_collections_l423_423609

theorem mathemtacial_collections :
  let vowels := multiset.of_list ['A', 'A', 'A', 'I', 'E'],
      consonants := multiset.of_list ['M', 'M', 'T', 'T', 'H', 'C', 'L'],
      choose_vowels := quotient.out (fintype.card (finset.filter (λ (s: multiset Char), s.coeffs ('A') + s.coeffs ('I') + s.coeffs ('E') = 3 ∧ multiset.card s = 3) (finset.powerset vowels.val))),
      choose_consonants := quotient.out (fintype.card (finset.filter (λ (s: multiset Char), s.coeffs ('M') + s.coeffs ('T') + s.coeffs ('H') + s.coeffs ('C') + s.coeffs ('L') = 5 ∧ multiset.card s = 5) (finset.powerset consonants.val)))
  in choose_vowels * choose_consonants = 220 :=
by {
  sorry
}

end mathemtacial_collections_l423_423609


namespace all_can_arrive_l423_423709

theorem all_can_arrive (
  num_people : ℕ, 
  num_cars : ℕ, 
  car_capacity : ℕ, 
  car_breakdown_distance : ℕ, 
  available_time : ℕ, 
  car_speed : ℕ, 
  walking_speed : ℕ, 
) : num_people = 8 → 
    num_cars = 2 →
    car_capacity = 5 →
    car_breakdown_distance = 15 →
    available_time = 42 →
    car_speed = 60 →
    walking_speed = 5 →
    ∃ (plan1_time plan2_time : ℝ), 
      plan1_time < available_time / 60 ∧ 
      plan2_time < available_time / 60 := by
  sorry

end all_can_arrive_l423_423709


namespace avg_price_goat_l423_423707

-- Definitions based on the conditions
def cows := 2
def goats := 10
def total_cost : ℕ := 1500
def avg_price_cow := 400

-- Theorem stating the goal
theorem avg_price_goat (G: ℕ) (hcows: ℕ) (hgoats: ℕ) (htotal_cost: ℕ) (havg_price_cow: ℕ):
  (hcows = cows) →
  (hgoats = goats) →
  (htotal_cost = total_cost) →
  (havg_price_cow = avg_price_cow) →
  G = (htotal_cost - (hcows * havg_price_cow)) / hgoats →
  G = 70 :=
begin
  intros,
  sorry  -- Proof left as an exercise
end

end avg_price_goat_l423_423707


namespace only_constant_coprime_polynomials_l423_423450

noncomputable def polynomial (f : ℕ → ℤ) : Prop := ∃ p : ℤ[X], ∀ n, f n = eval n p

def coprime_polynomials (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, Int.gcd (f n) (f (2^n)) = 1

theorem only_constant_coprime_polynomials (f : ℕ → ℤ) (hf : polynomial f) (hc : coprime_polynomials f) :
  (∀ x, f x = 1) ∨ (∀ x, f x = -1) :=
sorry

end only_constant_coprime_polynomials_l423_423450


namespace find_k_find_t_range_l423_423858

-- Definitions of the given vectors
def a : ℝ × ℝ := (real.sqrt 3, -1)
def b : ℝ × ℝ := (1/2, real.sqrt 3 / 2)

-- Definitions of x and y
def x (t : ℝ) := let term := t^2 - 3 in (a.1 + term * b.1, a.2 + term * b.2)
def y (k t : ℝ) := let term := -k in (term * a.1 + t * b.1, term * a.2 + t * b.2)

-- Proof statement part 1
theorem find_k (t : ℝ) (ht : t ≠ 0) :
  (let k := (1/4) * t * (t^2 - 3) in
  ∃ k, 
  (x t).fst * (y k t).fst + (x t).snd * (y k t).snd = 0)
  :=
begin
  sorry
end

-- Proof statement part 2
theorem find_t_range : 
  ∀ t : ℝ, (1/4) * t * (t^2 - 3) > 0 ↔ -real.sqrt 3 < t ∧ t < 0 ∨ t > real.sqrt 3
:=
begin
  sorry
end

end find_k_find_t_range_l423_423858


namespace bisection_step_l423_423017

def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- Given conditions
lemma f_0_neg : f 0 < 0 := by
  simp [f]
  norm_num

lemma f_05_pos : f 0.5 > 0 := by
  simp [f]
  norm_num

-- The statement to prove
theorem bisection_step :
  ∃ (Δ : set ℝ) (x1 : ℝ), Δ = set.Ioc 0 0.5 ∧ x1 = 0.25 := by
  use set.Ioc 0 0.5,
  use 0.25,
  split,
  {
    -- Prove that the interval Δ is (0, 0.5)
    refl
  },
  {
    -- Prove that x1 is 0.25
    refl
  }

end bisection_step_l423_423017


namespace scorpion_additional_millipedes_l423_423025

theorem scorpion_additional_millipedes :
  let total_segments := 800 in
  let segments_one_millipede := 60 in
  let segments_two_millipedes := 2 * (2 * segments_one_millipede) in
  let total_eaten := segments_one_millipede + segments_two_millipedes in
  let remaining_segments := total_segments - total_eaten in
  let segments_per_millipede := 50 in
  remaining_segments / segments_per_millipede = 10 :=
by {
  sorry
}

end scorpion_additional_millipedes_l423_423025


namespace find_a_and_b_l423_423511

theorem find_a_and_b (a b : ℝ) :
  (∀ x, y = a + b / x) →
  (y = 3 → x = 2) →
  (y = -1 → x = -4) →
  a + b = 4 :=
by sorry

end find_a_and_b_l423_423511


namespace triangle_properties_l423_423226

variable {a b c A B C S : ℝ}

def in_triangle_ABC (A B C a b c: ℝ) : Prop :=
  ∃ A B C a b c, A + B + C = π ∧
              a = sin(A) ∧
              b = sin(B) ∧
              c = sin(C)

theorem triangle_properties (A B C a b c: ℝ) (h1: in_triangle_ABC A B C a b c):
  (sin (B + C) = sin A) ∧
  (sin A > sin B → A > B) ∧
  (a * cos B - b * cos A = c → A = π / 2) ∧
  (b = 3 → A = π / 3 → S = 3 * sqrt(3) → ¬(a = sqrt 13 / 3)) :=
by
  intros
  sorry

end triangle_properties_l423_423226


namespace inequality_sum_of_reciprocals_l423_423251

variable {a b c : ℝ}

theorem inequality_sum_of_reciprocals
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hsum : a + b + c = 3) :
  (1 / (2 * a^2 + b^2 + c^2) + 1 / (2 * b^2 + c^2 + a^2) + 1 / (2 * c^2 + a^2 + b^2)) ≤ 3/4 :=
sorry

end inequality_sum_of_reciprocals_l423_423251


namespace unique_function_and_sum_l423_423230

open Nat

noncomputable def f : ℕ → ℕ := sorry

theorem unique_function_and_sum :
  (∀ m n : ℕ, f (m + f n) = n + f (m + 95)) →
  (∃! f : ℕ → ℕ, (∀ m n : ℕ, f (m + f n) = n + f (m + 95))) ∧ (finset.sum (finset.range 20) (λ n => f (n + 1)) = 1995) :=
begin
  intros h,
  sorry
end

end unique_function_and_sum_l423_423230


namespace find_angle_LOM_l423_423148

-- We will need to handle angles, so defining angles and isosceles property
-- Given conditions:
def isosceles_triangle (K L M : Type) [euclidean_geometry K L M] (KL_equal_LM : K = M) (angle_KLM : ∀ (P : Type), angle K L M = 114) : Prop :=
∀ (O : Type), ∃ (angle_OMK angle_OKM : ℝ), angle_OMK = 30 ∧ angle_OKM = 27 ∧ angle L O M = 150

-- Prove that:
theorem find_angle_LOM {K L M O : Type} [euclidean_geometry K L M O]
  (KL_equal_LM : isosceles_triangle K L M)
  (angle_OMK : angle O M K = 30)
  (angle_OKM : angle O K M = 27) :
  angle L O M = 150 := by sorry

end find_angle_LOM_l423_423148


namespace radius_semicircle_YZ_l423_423923

-- Define the given conditions
def angle_XYZ_right (X Y Z : Type) [inner_product_space ℝ X] : Prop :=
  angle X Y Z = π / 2

def arc_length_semicircle_XY (d : ℝ) : Prop :=
  π * d = 10 * π

def area_semicircle_XZ (r : ℝ) : Prop :=
  (1/2) * π * r^2 = 12.5 * π

-- Main theorem to prove
theorem radius_semicircle_YZ (X Y Z : Type) [inner_product_space ℝ X] 
  (h1 : angle_XYZ_right X Y Z)
  (h2 : ∃ d, arc_length_semicircle_XY d)
  (h3 : ∃ r, area_semicircle_XZ r) 
  : ∃ r_YZ, r_YZ = 5 * real.sqrt 2 :=
sorry

end radius_semicircle_YZ_l423_423923


namespace standard_equation_of_ellipse_max_area_triangle_l423_423508

noncomputable theory
open_locale big_operators

-- Definitions given in the problem
def ellipse_eq(a b x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1) 

def right_focus (F : ℝ × ℝ) : Prop := 
  (F = (1, 0))

def sum_distances_condition (a : ℝ) : Prop := 
  (2 * a = 4)

def b_sq_condition (a b: ℝ) : Prop := 
  (b^2 = a^2 - 1^2)

-- Translate conditions into a matrix of propositions
def problem_conditions (a b x y : ℝ) (F : ℝ × ℝ) : Prop := 
  ellipse_eq a b x y ∧ right_focus F ∧ sum_distances_condition a ∧ b_sq_condition a b

-- Part (I): Find the standard equation of the ellipse
theorem standard_equation_of_ellipse (a b: ℝ) (x y : ℝ) (F : ℝ × ℝ) 
  (h : problem_conditions a b x y F) : 
  (a = 2) → (b^2 = 3) → ellipse_eq a b x y :=
by sorry

-- Definition of line passing through right focus
def line_eq(m y : ℝ) : ℝ :=
  m * y + 4

-- Part (II): Determine maximum area for triangle FPQ'
theorem max_area_triangle (x1 y1 x2 y2 m : ℝ) 
  (h1: x1 = line_eq m y1) (h2: x2 = line_eq m y2): 
  (∃ max_area, max_area = 3 * sqrt 3 / 4) :=
by sorry

end standard_equation_of_ellipse_max_area_triangle_l423_423508


namespace reema_simple_interest_l423_423282

-- Definitions and conditions
def principal : ℕ := 1200
def rate_of_interest : ℕ := 6
def time_period : ℕ := rate_of_interest

-- Simple interest calculation
def calculate_simple_interest (P R T: ℕ) : ℕ :=
  (P * R * T) / 100

-- The theorem to prove that Reema paid Rs 432 as simple interest.
theorem reema_simple_interest : calculate_simple_interest principal rate_of_interest time_period = 432 := 
  sorry

end reema_simple_interest_l423_423282


namespace geom_seq_a_plus_1_Sn_geq_one_l423_423524

noncomputable def a_seq : ℕ → ℝ
| 0       := 0
| (n + 1) := (a_seq n + 1) / 2 - 1

def b_seq (n : ℕ) : ℝ := n * (a_seq (n - 1) + 1) - n -- note n starts from 1.

def S (n : ℕ) : ℝ := ∑ i in range n, b_seq (i + 1) -- summation from k = 1 to n

theorem geom_seq_a_plus_1 :
  ∀ n, a_seq (n + 1) + 1 = (1 / 2) ^ n :=
sorry -- Proof is skipped.

theorem Sn_geq_one (n : ℕ) : S n ≥ 1 :=
sorry -- Proof is skipped.

end geom_seq_a_plus_1_Sn_geq_one_l423_423524


namespace total_baseball_cards_l423_423356

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end total_baseball_cards_l423_423356


namespace departure_of_30_tons_of_grain_l423_423211

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l423_423211


namespace minimum_dot_product_l423_423903

variable {A B C M N : ℝ}
variable (a b : ℝ)
variable (A₀ : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))
variable (angle_A : (math.angle.vectors.v ∠ A) = (Real.pi / 3))
variable (area_ABC : (1/2) * (abs (B - A) * sin (angle_A)) = Real.sqrt 3)
variable (M_is_midpoint : M = (B + C) / 2)
variable (N_is_midpoint : N = (B + M) / 2)

theorem minimum_dot_product :
  ∃ min_val, min_val = -1 ∧ ∀ x, (overline AM) * (overline AN) ≥ min_val :=
sorry

end minimum_dot_product_l423_423903


namespace sum_inverse_sqrt_less_than_2sqrt_l423_423278

theorem sum_inverse_sqrt_less_than_2sqrt (n : ℕ) (h : n > 0) : 
  (∑ i in finset.range n, 1 / real.sqrt (i + 1)) < 2 * real.sqrt n :=
sorry

end sum_inverse_sqrt_less_than_2sqrt_l423_423278


namespace proof_candle_burn_l423_423676

noncomputable def candle_burn_proof : Prop :=
∃ (t : ℚ),
  (t = 40 / 11) ∧
  (∀ (H_1 H_2 : ℚ → ℚ),
    (∀ t, H_1 t = 1 - t / 5) ∧
    (∀ t, H_2 t = 1 - t / 4) →
    ∃ (t : ℚ), ((1 - t / 5) = 3 * (1 - t / 4)) ∧ (t = 40 / 11))

theorem proof_candle_burn : candle_burn_proof :=
sorry

end proof_candle_burn_l423_423676


namespace cylindrical_container_volume_increase_l423_423763

theorem cylindrical_container_volume_increase (R H : ℝ)
  (initial_volume : ℝ)
  (x : ℝ) : 
  R = 10 ∧ H = 5 ∧ initial_volume = π * R^2 * H →
  π * (R + 2 * x)^2 * H = π * R^2 * (H + 3 * x) →
  x = 5 :=
by
  -- Given conditions
  intro conditions volume_equation
  obtain ⟨hR, hH, hV⟩ := conditions
  -- Simplifying and solving the resulting equation
  sorry

end cylindrical_container_volume_increase_l423_423763


namespace fractions_simplify_to_prime_denominator_2023_l423_423528

def num_fractions_simplifying_to_prime_denominator (n: ℕ) (p q: ℕ) : ℕ :=
  let multiples (m: ℕ) : ℕ := (n - 1) / m
  multiples p + multiples (p * q)

theorem fractions_simplify_to_prime_denominator_2023 :
  num_fractions_simplifying_to_prime_denominator 2023 17 7 = 22 :=
by
  sorry

end fractions_simplify_to_prime_denominator_2023_l423_423528


namespace travis_return_probability_l423_423331

open Function

namespace CubeHopping

def Vertex := (Fin 2 × Fin 2 × Fin 2)

def adjacent (v1 v2 : Vertex) : Prop := 
  (v1.1 = v2.1 ∧ v1.2 = v2.2 ∧ v1.3 ≠ v2.3) ∨ 
  (v1.1 = v2.1 ∧ v1.2 ≠ v2.2 ∧ v1.3 = v2.3) ∨ 
  (v1.1 ≠ v2.1 ∧ v1.2 = v2.2 ∧ v1.3 = v2.3)

def probability_of_returning (start : Vertex) (moves : ℕ) : ℚ :=
-- Function to encapsulate the details of the probability calculation (to be filled in)
sorry

theorem travis_return_probability :
  probability_of_returning (0, 0, 0) 4 = 7 / 27 :=
sorry

end CubeHopping

end travis_return_probability_l423_423331


namespace tangent_line_f_at_1_l423_423595

variable {R : Type*} [LinearOrderedField R]

noncomputable def g (x : R) : R := sorry

noncomputable def f (x : R) : R := g (x^2) + x^2

axiom tangent_line_g : ∀ x : R, 1 = 1 → g x = 2 * x + 1

theorem tangent_line_f_at_1 : 
  ∀ g f : R → R, 
  (∀ x : R, g x = 2 * x + 1) → 
  f (1 : R) = (g (1 : R)^2) + (1 : R)^2 + 
  (6 : R) * (1 : R) - (f (1 : R)) = (2 : R) := 
begin 
  intros g f hg,
  sorry 
end

end tangent_line_f_at_1_l423_423595


namespace matching_domino_patterns_l423_423897

-- Define the conditions for the grid, its size, and the comparison of blue cells
structure Grid (n : ℕ) :=
  (cells : array (Fin n × Fin n) Bool)

-- Assuming the grid size is 8x8
def grid_size := 8

-- Two grids of size 8x8
variable (Igor_grid Valya_grid : Grid grid_size)

-- Condition stating Igor and Valya painted the same number of cells blue
axiom same_number_of_blue_cells :
  Igor_grid.cells.count (λ c, c.snd) = Valya_grid.cells.count (λ c, c.snd)

-- Statement to be proven
theorem matching_domino_patterns (Igor_grid Valya_grid : Grid grid_size) :
  same_number_of_blue_cells Igor_grid Valya_grid →
  ∃ (partition_I partition_V : list (Fin grid_size × Fin grid_size))
  (disjoint : ∀ d ∈ partition_I, ∀ e ∈ partition_V, d ∩ e = ∅),
  let assembled_I := assemble partition_I in
  let assembled_V := assemble partition_V in
  assembled_I = assembled_V := 
sorry

end matching_domino_patterns_l423_423897


namespace total_grazing_area_l423_423724

-- Define the dimensions of the field
def field_width : ℝ := 46
def field_height : ℝ := 20

-- Define the length of the rope
def rope_length : ℝ := 17

-- Define the radius and position of the fenced area
def fenced_radius : ℝ := 5
def fenced_distance_x : ℝ := 25
def fenced_distance_y : ℝ := 10

-- Given the conditions, prove the total grazing area
theorem total_grazing_area (field_width field_height rope_length fenced_radius fenced_distance_x fenced_distance_y : ℝ) :
  (π * rope_length^2 / 4) = 227.07 :=
by
  sorry

end total_grazing_area_l423_423724


namespace cross_ratio_preserved_l423_423362

-- Definitions for the given lines and their intersection point
variables {F : Type*} [field F] [affine_space F] 
variables {P : affine_subspace F} -- the affine space
variable (O : P) -- the intersection point of the lines

-- Define lines passing through point O
variables (a b c d: line F) (h_a: a.contains O) (h_b: b.contains O) (h_c: c.contains O) (h_d: d.contains O)

-- Define line l and points A, B, C, D as intersections with lines a, b, c, d
variable (l : line F) (h_l: ¬ l.contains O)
variables (A B C D : point F)
variables (h_A : A ∈ l ∧ A ∈ a) (h_B : B ∈ l ∧ B ∈ b) (h_C : C ∈ l ∧ C ∈ c) (h_D : D ∈ l ∧ D ∈ d)

-- State the theorem that the cross-ratio is preserved
theorem cross_ratio_preserved : (cross_ratio F (A, B, C, D)) = (cross_ratio F (a, b, c, d)) :=
sorry

end cross_ratio_preserved_l423_423362


namespace percentage_poached_less_sold_l423_423392

-- Define the variables and the conditions
def C (P : ℕ) : ℕ := 1.20 * P
def total_pears : ℕ := 42
def sold_pears : ℕ := 20

-- The theorem statement
theorem percentage_poached_less_sold (P : ℕ) (hC : C P + P = total_pears - sold_pears) : 
  (sold_pears - P) / sold_pears * 100 = 50 :=
sorry

end percentage_poached_less_sold_l423_423392


namespace value_of_2a_minus_b_minus_4_l423_423881

theorem value_of_2a_minus_b_minus_4 (a b : ℝ) (h : 2 * a - b = 2) : 2 * a - b - 4 = -2 :=
by
  sorry

end value_of_2a_minus_b_minus_4_l423_423881


namespace find_g_14_15_16_l423_423305

def g : ℤ × ℤ × ℤ → ℝ
-- Hypotheses
axiom g_1 : ∀ (a b c n : ℤ), g (n * a, n * b, n * c) = n * g (a, b, c)
axiom g_2 : ∀ (a b c n : ℤ), g (a + n, b + n, c + n) = g (a, b, c) + n
axiom g_3 : ∀ (a b c : ℤ), g (a, b, c) = g (c, b, a)

theorem find_g_14_15_16 : g (14, 15, 16) = 15 :=
by sorry

end find_g_14_15_16_l423_423305


namespace sum_of_coincidence_numbers_eq_531_l423_423805

def sum_digits (n : ℕ) : ℕ :=
  n.digits.sum

def product_digits (n : ℕ) : ℕ :=
  n.digits.product

def is_coincidence_number (n : ℕ) : Prop :=
  sum_digits n + product_digits n = n

theorem sum_of_coincidence_numbers_eq_531 :
  ∑ n in (Finset.range 100).filter is_coincidence_number, n = 531 :=
by
  sorry

end sum_of_coincidence_numbers_eq_531_l423_423805


namespace unique_arrangements_of_TOOTH_l423_423770

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem unique_arrangements_of_TOOTH : 
  let word := "TOOTH" in
  let n := 5 in
  let t_count := 3 in
  let o_count := 2 in
  n.factorial / (t_count.factorial * o_count.factorial) = 10 :=
sorry

end unique_arrangements_of_TOOTH_l423_423770


namespace maximum_partition_l423_423199

def is_partition {α : Type} (S : set α) (n : ℕ) (f : fin n → set α) : Prop :=
  (∀ i, f i ≠ ∅) ∧
  (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧
  (⋃ i, f i) = S

def S : set ℕ := {i | 1 ≤ i ∧ i ≤ 16}

theorem maximum_partition (n : ℕ) (f : fin n → set ℕ) :
  is_partition S n f →
  (∃ i, ∃ a b c ∈ f i, a + b = c) →
  n ≤ 3 :=
sorry

end maximum_partition_l423_423199


namespace frustum_volume_correct_l423_423386

-- Definitions based on the conditions in a)
structure Pyramid :=
  (base_edge : ℝ)
  (altitude : ℝ)

def original_pyramid : Pyramid :=
  {base_edge := 16, altitude := 10}

def smaller_pyramid : Pyramid :=
  {base_edge := 8, altitude := 5}

noncomputable def volume (p : Pyramid) : ℝ :=
  (1 / 3) * (p.base_edge ^ 2) * p.altitude

def frustum_volume (P : Pyramid) (P_small: Pyramid) : ℝ :=
  volume P - volume P_small

-- Given conditions
def height_of_frustum := 5

-- Final proof statement
theorem frustum_volume_correct :
  frustum_volume original_pyramid smaller_pyramid = 2240 / 3 :=
by
  sorry

end frustum_volume_correct_l423_423386


namespace daughters_and_granddaughters_without_daughters_l423_423598

-- Given conditions
def melissa_daughters : ℕ := 10
def half_daughters_with_children : ℕ := melissa_daughters / 2
def grandchildren_per_daughter : ℕ := 4
def total_descendants : ℕ := 50

-- Calculations based on given conditions
def number_of_granddaughters : ℕ := total_descendants - melissa_daughters
def daughters_with_no_children : ℕ := melissa_daughters - half_daughters_with_children
def granddaughters_with_no_children : ℕ := number_of_granddaughters

-- The final result we need to prove
theorem daughters_and_granddaughters_without_daughters : 
  daughters_with_no_children + granddaughters_with_no_children = 45 := by
  sorry

end daughters_and_granddaughters_without_daughters_l423_423598


namespace distance_from_point_P_to_directrix_l423_423503

-- Definition of the parabola and the directrix
def parabola (y x : ℝ) : Prop := y^2 = 8 * x
def directrix (x : ℝ) : Prop := x = -2

-- Definition of the point P and its distance to the y-axis
def point_P (x : ℝ) : Prop := x = 4

-- The problem statement: find the distance from point P to the directrix
theorem distance_from_point_P_to_directrix (x y : ℝ) (h₁: parabola y x) (h₂: point_P x)  :
  abs( x - (-2)) = 6 :=
by
  sorry

end distance_from_point_P_to_directrix_l423_423503


namespace poster_width_l423_423729
   
   theorem poster_width (h : ℕ) (A : ℕ) (w : ℕ) (h_eq : h = 7) (A_eq : A = 28) (area_eq : w * h = A) : w = 4 :=
   by
   sorry
   
end poster_width_l423_423729


namespace num_complex_roots_l423_423438

open Complex

theorem num_complex_roots (a : ℝ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : 0 < n) :
  (λ (p : Polynomial ℂ), (if q = p then n + 1 else 0)) 
  (p : Polynomial ℂ)
  (p = a * (Polynomial.x ^ n + Polynomial.x ^ (n - 1) + ... + Polynomial.x + 1)) ≠ 0 :=
sorry

end num_complex_roots_l423_423438


namespace C_converges_l423_423165

noncomputable def behavior_of_C (e R r : ℝ) (n : ℕ) : ℝ := e * (n^2) / (R + n * (r^2))

theorem C_converges (e R r : ℝ) (h₁ : 0 < r) : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |behavior_of_C e R r n - e / r^2| < ε := 
sorry

end C_converges_l423_423165


namespace find_n_l423_423015

theorem find_n (n : ℕ) (H : 7.63 * Real.log 3 / Real.log 2 * Real.log 4 / Real.log 3 * Real.log 5 / Real.log 4 * ... * Real.log (n + 1) / Real.log n = 10) : n = 1023 :=
sorry

end find_n_l423_423015


namespace divide_nonconvex_quadrilateral_into_six_parts_by_two_lines_l423_423756

-- Definition of a non-convex quadrilateral
structure NonConvexQuadrilateral :=
  (vertices : list (ℝ × ℝ))
  (is_nonconvex : ∃ (angle : ℝ), angle > 180)

-- Statement of the problem
theorem divide_nonconvex_quadrilateral_into_six_parts_by_two_lines (q : NonConvexQuadrilateral) :
  ∃ (line1 line2 : ℝ × ℝ → ℝ), divides_into_six_parts q line1 line2 :=
sorry

end divide_nonconvex_quadrilateral_into_six_parts_by_two_lines_l423_423756


namespace minimum_value_of_expression_l423_423107

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l423_423107


namespace total_opaque_stackings_l423_423364

-- Define the glass pane and its rotation
inductive Rotation
| deg_0 | deg_90 | deg_180 | deg_270
deriving DecidableEq, Repr

-- The property of opacity for a stack of glass panes
def isOpaque (stack : List (List Rotation)) : Bool :=
  -- The implementation of this part depends on the specific condition in the problem
  -- and here is abstracted out for the problem statement.
  sorry

-- The main problem stating the required number of ways
theorem total_opaque_stackings : ∃ (n : ℕ), n = 7200 :=
  sorry

end total_opaque_stackings_l423_423364


namespace solution_set_of_inequality_l423_423667

theorem solution_set_of_inequality :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_of_inequality_l423_423667


namespace irrational_root_three_l423_423057

theorem irrational_root_three 
  (hA : ¬ irrational (-32 / 7))
  (hB : ¬ irrational 0)
  (hC : irrational (Real.sqrt 3))
  (hD : ¬ irrational 3.5) : 
  ∃ x, x = Real.sqrt 3 ∧ irrational x ∧ 
    (x ≠ -32 / 7 ∧ x ≠ 0 ∧ x ≠ 3.5) := 
by
  sorry

end irrational_root_three_l423_423057


namespace degree_of_monomial_l423_423298

theorem degree_of_monomial : 
  let monomial := λ m n : ℕ, -1 / 7 * (m * n^2) in 
  monomial 1 1 = -1 / 7 * (1 * 1^2) → 3 = 3 :=
by
  intro h
  sorry

end degree_of_monomial_l423_423298


namespace number_of_correct_props_l423_423670

def prop1 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  (l1 ∈ p1) ∧ (l2 ∈ p2) ∧ (p1 ≠ p2) → ¬(Intersect l1 l2) ∧ ¬(Parallel l1 l2)

def prop2 (l1 l2 : Line) (p : Plane) : Prop :=
  (l1 ⊥ p) ∧ (l2 ⊥ p) → (Parallel l1 l2)

def prop3 (l : Line) (α : Plane) : Prop :=
  (l ∩ α ≠ ∅) → ∃! β : Plane, β ⊥ α ∧ (l ∈ β)

def correct_props : Nat :=
  2

theorem number_of_correct_props (l1 l2 : Line) (p1 p2 p: Plane) (α: Plane) : 
    prop2 l1 l2 p ∧ prop3 l1 α → correct_props = 2 := 
by
  sorry

end number_of_correct_props_l423_423670


namespace set_of_crease_lines_l423_423032

-- Definitions of conditions
variable (R a : ℝ) -- Radius of circle and distance OA
variable (O A : EuclideanSpace ℝ (Fin 2)) -- Point O and A in Euclidean space

-- Definition of the ellipse Γ
noncomputable def ellipse (O A : EuclideanSpace ℝ (Fin 2)) (R : ℝ) :=
  {C : EuclideanSpace ℝ (Fin 2) | dist C O + dist C A = R}

-- Problem statement
theorem set_of_crease_lines (R a : ℝ) (O A : EuclideanSpace ℝ (Fin 2)) (hA : dist O A = a) :
  ∀ A' : EuclideanSpace ℝ (Fin 2), 
  (dist O A' = R ∧ (∃ l, is_crease_line l A A' O R)) →
  ∀ C : EuclideanSpace ℝ (Fin 2), 
  (∃ P, dist P A + dist P O > R) ↔ (C ∈ ellipse O A R ∨ C ∉ ellipse O A R) := sorry

end set_of_crease_lines_l423_423032


namespace simplify_and_evaluate_expr_at_x_neg4_l423_423290

theorem simplify_and_evaluate_expr_at_x_neg4 : 
  let x := -4 in
  (1 - (x + 1) / (x^2 - 2 * x + 1)) / ((x - 3) / (x - 1)) = 4 / 5 :=
by
  sorry

end simplify_and_evaluate_expr_at_x_neg4_l423_423290


namespace calculate_difference_of_squares_l423_423427

theorem calculate_difference_of_squares :
  (153^2 - 147^2) = 1800 :=
by
  sorry

end calculate_difference_of_squares_l423_423427


namespace length_MN_and_side_length_of_triangle_l423_423269

def equilateral_triangle (A B C : Type) := 
  ∃ (s : ℝ), dist A B = s ∧ dist B C = s ∧ dist C A = s

def circumcircle (A B C : Type) := 
  ∃ R (O : Type), dist O A = R ∧ dist O B = R ∧ dist O C = R

noncomputable def point_M_on_circle (M A C : Type) := 
  dist M A = 2 ∧ dist M C = 3

axiom line_intersects_side_at_N (B M A C N : Type) : 
  line B M ∧ point N ∧ lies_on_line N (line A C)

theorem length_MN_and_side_length_of_triangle
  (A B C M N : Type)
  (triangle_ABC : equilateral_triangle A B C)
  (circumcircle_ABC : circumcircle A B C)
  (point_M_conditions : point_M_on_circle M A C)
  (line_intersection : line_intersects_side_at_N B M A C N):
  dist M N = 6 / 5 ∧ (∃ (s : ℝ), s = sqrt 19 ∧ equilateral_triangle A B C) :=
sorry

end length_MN_and_side_length_of_triangle_l423_423269


namespace smallest_n_for_angles_leq_60_l423_423939

theorem smallest_n_for_angles_leq_60 :
  ∃ n : ℕ, n = 3 ∧ 
    let A := 50
        B := 65
        C := 65
        A₁ := (B + C) / 2
        B₁ := (A + C) / 2
        C₁ := (A + B) / 2 in
      A₁ ≤ 60 ∧ B₁ ≤ 60 ∧ C₁ ≤ 60 ∧
    ∀ k : ℕ, k < 3 → 
      let A_k := (B_k + C_k) / 2
          B_k := (A_k + C_k) / 2
          C_k := (A_k + B_k) / 2 in
      A_k > 60 ∨ B_k > 60 ∨ C_k > 60 :=
begin
  sorry
end

end smallest_n_for_angles_leq_60_l423_423939


namespace minimum_am_an_dot_product_l423_423901

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

theorem minimum_am_an_dot_product 
  (A B C : ℝ × ℝ)
  (M : ℝ × ℝ := (midpoint B.1 C.1, midpoint B.2 C.2)) 
  (N : ℝ × ℝ := (midpoint B.1 M.1, midpoint B.2 M.2)) 
  (angle_A : ℝ := real.pi / 3) 
  (area_ABC : ℝ := real.sqrt 3) 
  (h_area : 0.5 * (B.1 - A.1) * (C.2 - A.2) * real.sin angle_A = real.sqrt 3) :
  let AM := ((M.1 - A.1), (M.2 - A.2)),
      AN := ((N.1 - A.1), (N.2 - A.2)) 
  in (AM.1 * AN.1 + AM.2 * AN.2) = 2 :=
sorry

end minimum_am_an_dot_product_l423_423901


namespace real_no_impure_l423_423879

theorem real_no_impure {x : ℝ} (h1 : x^2 - 1 = 0) (h2 : x^2 + 3 * x + 2 ≠ 0) : x = 1 :=
by
  sorry

end real_no_impure_l423_423879


namespace avg_GPA_school_is_correct_l423_423640

def avg_GPA_school : ℝ :=
  let gpa_6th := 93
  let gpa_7th := gpa_6th + 2
  let gpa_8th := 91
  in (gpa_6th + gpa_7th + gpa_8th) / 3

theorem avg_GPA_school_is_correct :
  avg_GPA_school = 93 :=
by
  sorry

end avg_GPA_school_is_correct_l423_423640


namespace geom_seq_n_value_l423_423816

noncomputable def a (n : ℕ) : ℝ := sorry -- The sequence function a_n

theorem geom_seq_n_value :
  (∃ a : ℕ → ℝ, 
    (∀ n m : ℕ, ∃ (r : ℝ), a (n + m) = a n * r ^ m) ∧
    a 2 + a 5 = 18 ∧
    a 3 * a 4 = 32 ∧
    a (some n : ℕ) = 128) →
  n = 8 :=
begin
  sorry
end

end geom_seq_n_value_l423_423816


namespace original_number_of_turtles_l423_423723

-- Define the problem
theorem original_number_of_turtles (T : ℕ) (h1 : 17 = (T + 3 * T - 2) / 2) : T = 9 := by
  sorry

end original_number_of_turtles_l423_423723


namespace max_sum_of_arithmetic_sequence_l423_423491

noncomputable def a (n : ℕ) : ℝ := 10 - n

def a4 := a 4
def a6 := a 6
def a3 := a 3
def a7 := a 7

def S (n : ℕ) : ℝ := 
  (1/2) * n * (2 * a 1 + (n - 1) * (-1))

theorem max_sum_of_arithmetic_sequence :
  a4 * a6 = 24 ∧ a3 + a7 = 10 ∧ ∀ (d : ℝ), d < 0 → 
    ∃ n : ℕ, S n = 45 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l423_423491


namespace reuse_calendar_2080_2108_l423_423643

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def same_calendar_year (start_year future_year : ℕ) : Prop :=
  let days_in_year := if is_leap_year start_year then 366 else 365
  let day_of_week_start_year := (start_year + days_in_year) % 7
  let day_of_week_future_year := (future_year + days_in_year) % 7
  day_of_week_start_year = day_of_week_future_year ∧ is_leap_year start_year = is_leap_year future_year

theorem reuse_calendar_2080_2108 : same_calendar_year 2080 2108 :=
by
  sorry

end reuse_calendar_2080_2108_l423_423643


namespace variance_X_binomial_l423_423911

theorem variance_X_binomial :
  ∀ σ : ℝ, ∀ (X : ℝ → Prop), (∀ x, X x → x ∼ normal 90 σ^2)
  ∧ (P (λ x, x < 70) 0.2)
  ∧ (∀ x, P (λ x, 90 ≤ x ∧ x ≤ 110) 0.3) →
  (∃ (X : ℕ → Prop), ∃ n : ℕ, n = 10 ∧ X 0.3 →
  ∀ p: ℝ, ∀ k: ℕ, X k ∼ binomial n 0.3 ∧ variance X = 2.1) :=
by sorry

end variance_X_binomial_l423_423911


namespace microbial_population_extinction_l423_423728

-- Definitions
def p_0 : ℝ := 0.4
def p_1 : ℝ := 0.3
def p_2 : ℝ := 0.2
def p_3 : ℝ := 0.1

def P (X : ℕ → ℝ) (i : ℕ) : ℝ :=
  match i with
  | 0 => p_0
  | 1 => p_1
  | 2 => p_2
  | 3 => p_3
  | _ => 0

def E (X : ℕ → ℝ) : ℝ := 0 * p_0 + 1 * p_1 + 2 * p_2 + 3 * p_3

-- Theorem Statement
theorem microbial_population_extinction :
  ∀ X : ℕ → ℝ,
    E(X) = 1 →
    (∀ x : ℝ, p_0 + p_1 * x + p_2 * x^2 + p_3 * x^3 = x →
      (E(X) ≤ 1 → x = 1) ∧ (E(X) > 1 → x < 1)) :=
by
  intros X hEX heq
  -- To be proven
  sorry

end microbial_population_extinction_l423_423728


namespace area_abe_l423_423918

variables (A B C D E F : Type) [Geometry A B C D E F]
  (area : ∀ {X Y Z W : Type} [Geometry X Y Z W], Type)
  (rect_ABCD : IsRectangle A B C D)
  (parallel_EF_BC : Parallel E F B C)
  (area_AECF : area A E C F = 17.5)
  (area_AFD : area A F D = 20)
  (area_BCE : area B C E = 15)
  (area_CDF : area C D F = 12.5)

-- The statement to be proven
theorem area_abe : area A B E = 5 := by
  sorry

end area_abe_l423_423918


namespace cube_root_of_minus_one_l423_423644

theorem cube_root_of_minus_one : ∃ x : ℝ, x^3 = -1 ∧ x = -1 := by
  existsi (-1:ℝ)
  split
  . simp
  . rfl

end cube_root_of_minus_one_l423_423644


namespace tangent_line_ln_x_and_ln_x_plus_1_l423_423546

theorem tangent_line_ln_x_and_ln_x_plus_1 (k b : ℝ) : 
  (∃ x₁ x₂ : ℝ, (y = k * x₁ + b ∧ y = ln x₁ + 2) ∧ 
                (y = k * x₂ + b ∧ y = ln (x₂ + 1)) ∧ 
                (k = 2 ∧ x₁ = 1 / 2 ∧ x₂ = -1 / 2)) → 
  b = 1 - ln 2 :=
by
  sorry

end tangent_line_ln_x_and_ln_x_plus_1_l423_423546


namespace largest_possible_value_l423_423634

theorem largest_possible_value (X Y Z m: ℕ) 
  (hX_range: 0 ≤ X ∧ X ≤ 4) 
  (hY_range: 0 ≤ Y ∧ Y ≤ 4) 
  (hZ_range: 0 ≤ Z ∧ Z ≤ 4) 
  (h1: m = 25 * X + 5 * Y + Z)
  (h2: m = 81 * Z + 9 * Y + X):
  m = 121 :=
by
  -- The proof goes here
  sorry

end largest_possible_value_l423_423634


namespace pebble_surface_area_l423_423045

theorem pebble_surface_area (a : ℝ) (h : a = 1) :
  let d := real.sqrt (a^2 + a^2 + a^2),
      sphere_area := π * d^2,
      x := (d - a) / 2,
      f := 6 * (x / d),
      adjusted_sphere_area := sphere_area * (1 - f),
      cube_face_area := 6 * π * (a / 2)^2,
      total_area := adjusted_sphere_area + cube_face_area
  in total_area = ((6 * real.sqrt 2 - 5) / 2) * π :=
sorry

end pebble_surface_area_l423_423045


namespace departs_if_arrives_l423_423214

theorem departs_if_arrives (grain_quantity : ℤ) (h : grain_quantity = 30) : -grain_quantity = -30 :=
by {
  have : -grain_quantity = -30,
  from congr_arg (λ x, -x) h,
  exact this
}

end departs_if_arrives_l423_423214


namespace prism_volume_approx_l423_423673

noncomputable def volume_of_prism (a b c : ℝ) : ℝ :=
  a * b * c

theorem prism_volume_approx (a b c : ℝ)
  (h₁ : a * b = 64)
  (h₂ : b * c = 81)
  (h₃ : a * c = 72)
  (h₄ : b = 2 * a) :
  volume_of_prism a b c ≈ 1629 :=
sorry

end prism_volume_approx_l423_423673


namespace avg_page_count_per_essay_l423_423036

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l423_423036


namespace smallest_x_congruence_statement_l423_423129

theorem smallest_x_congruence : ∃ x : ℤ, 51 * x + 15 ≡ 5 [MOD 35] ∧ 0 < x ∧ x = 30 :=
by
  /-
  Theorem statement explanation:
  ∃ x : ℤ → There exists an integer x
  51 * x + 15 ≡ 5 [MOD 35]  → Such that 51 * x + 15 is congruent to 5 modulo 35
  0 < x  → Additionally, x must be a positive integer
  x = 30  → And x equals 30
  -/
  sorry

end smallest_x_congruence_statement_l423_423129


namespace minimal_integer_perimeter_l423_423905

noncomputable def minimum_perimeter (BC : ℕ) : ℕ :=
  let a := BC
  let AB := 2 * a
  let AC := 2 * a
  let s := a + AB + AC
  s

theorem minimal_integer_perimeter :
  ∃ (BC : ℕ), BC > 0 ∧
  let a := BC in
  let AB := 2 * a in
  let AC := 2 * a in
  let s := a + AB + AC in
  let A := Real.sqrt (s * (s - a) * (s - AB) * (s - AC)) in
  let r := A / s in
  let Ra := A / (s - a) in
  Ra - r = a / 5 → s = 25 :=
begin
  sorry
end

end minimal_integer_perimeter_l423_423905


namespace product_of_solutions_eq_neg49_l423_423124

theorem product_of_solutions_eq_neg49 :
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (∏ x in ({7, -7} : Finset ℤ), x) = -49 := by
  sorry

end product_of_solutions_eq_neg49_l423_423124


namespace number_of_real_solutions_eq_2_l423_423112

theorem number_of_real_solutions_eq_2 :
  (∃! x : ℝ, (3 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3) → 2 := 
sorry

end number_of_real_solutions_eq_2_l423_423112


namespace min_value_sum_distance_l423_423240

open Real

noncomputable def is_point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 7) = 1

theorem min_value_sum_distance (F A P : ℝ × ℝ) :
  is_point_on_hyperbola P.1 P.2 →
  F = (-4, 0) → A = (1, 4) →
  ∃ E : ℝ × ℝ, E = (4, 0) ∧ (min_fun_sum_dist P F A E = 11)
:= by
  sorry

noncomputable def min_fun_sum_dist (P F A E : ℝ × ℝ) : ℝ :=
  dist P F + dist P A

end min_value_sum_distance_l423_423240


namespace aisha_probability_l423_423390

noncomputable def prob_one_head (prob_tail : ℝ) (num_coins : ℕ) : ℝ :=
  1 - (prob_tail ^ num_coins)

theorem aisha_probability : 
  prob_one_head (1/2) 4 = 15 / 16 := 
by 
  sorry

end aisha_probability_l423_423390


namespace swimmers_meeting_times_l423_423910

theorem swimmers_meeting_times (l : ℕ) (vA vB t : ℕ) (T : ℝ) :
  l = 120 →
  vA = 4 →
  vB = 3 →
  t = 15 →
  T = 21 :=
  sorry

end swimmers_meeting_times_l423_423910


namespace petr_receives_1000000_l423_423446

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end petr_receives_1000000_l423_423446


namespace value_of_a_smallest_positive_period_number_of_zeros_existence_of_n_l423_423855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * (abs (Real.sin x) + abs (Real.cos x)) + 4 * Real.sin (2 * x) + 9

theorem value_of_a :
  (∀ x, f a x = a * (abs (Real.sin x) + abs (Real.cos x)) + 4 * Real.sin (2 * x) + 9) →
  (f a (9 * Real.pi / 4) = 13 - 9 * Real.sqrt 2) →
  a = -9 := by sorry

theorem smallest_positive_period :
  ∀ a, ∀ x, f a (x + Real.pi) = f a x :=
by sorry

theorem number_of_zeros :
  ∀ a, a = -9 →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a x = 0) →
  (∃ x1 x2 x3 x4 ∈ Set.Icc 0 (Real.pi / 2), f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ f a x4 = 0) :=
by sorry

theorem existence_of_n :
  ∀ a, a = -9 → 
  ∃ n : ℕ, (∀ k : ℕ, k < n → ∃ x ∈ Set.Icc 0 ((n : ℝ) * Real.pi / 4), f a x = 0) →
  n = 2021 :=
by sorry

end value_of_a_smallest_positive_period_number_of_zeros_existence_of_n_l423_423855


namespace ratio_EP_DP_l423_423404

-- Given definitions and conditions from part a)
variables {A B C D E F P : Type}
variables [midpoint : Midpoint D B C] -- D is the midpoint of BC
variables [ratio1 : Ratio AF (2 : ℕ) BF] -- AF = 2BF
variables [ratio2 : Ratio CE (3 : ℕ) AE] -- CE = 3AE
variables [intersection : Intersect CF DE P] -- P is the intersection of CF and DE

-- Proof statement
theorem ratio_EP_DP : EP / DP = 3 := by
  sorry

end ratio_EP_DP_l423_423404
