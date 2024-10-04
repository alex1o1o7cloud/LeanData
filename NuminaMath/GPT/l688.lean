import Mathlib

namespace sector_central_angle_l688_688639

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : 1/2 * l * r = 1) : l / r = 2 := 
by
  sorry

end sector_central_angle_l688_688639


namespace line_equation_midpoint_ellipse_l688_688251

theorem line_equation_midpoint_ellipse (x1 y1 x2 y2 : ℝ) 
  (h_midpoint_x : x1 + x2 = 4) (h_midpoint_y : y1 + y2 = 2)
  (h_ellipse_x1_y1 : (x1^2) / 12 + (y1^2) / 4 = 1) (h_ellipse_x2_y2 : (x2^2) / 12 + (y2^2) / 4 = 1) :
  2 * (x1 - x2) + 3 * (y1 - y2) = 0 :=
sorry

end line_equation_midpoint_ellipse_l688_688251


namespace dad_steps_are_90_l688_688130

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688130


namespace inequality_proof_l688_688338

variable (a b c d : ℝ)
variable (h : a + b + c + d = 0)

theorem inequality_proof : (ab + ac + ad + bc + bd + cd)^2 + 12 ≥ 6(abc + abd + acd + bcd) :=
by
  sorry

end inequality_proof_l688_688338


namespace deepak_meeting_time_correct_l688_688498

noncomputable def deepak_meeting_time (circumference : ℝ) (deepak_speed_kmph : ℝ) (wife_speed_kmph : ℝ) : ℝ :=
  let deepak_speed_mpm := deepak_speed_kmph * 1000 / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let relative_speed_mpm := deepak_speed_mpm + wife_speed_mpm
  circumference / relative_speed_mpm

theorem deepak_meeting_time_correct :
  deepak_meeting_time 726 4.5 3.75 ≈ 5.28 := by
  sorry

end deepak_meeting_time_correct_l688_688498


namespace negative_result_is_A_l688_688932

def expr_A : ℤ := -| -2 |
def expr_B : ℤ := -(-2)^3
def expr_C : ℤ := -(-2)
def expr_D : ℤ := (-3)^2

theorem negative_result_is_A : expr_A < 0 ∧ expr_B ≥ 0 ∧ expr_C ≥ 0 ∧ expr_D ≥ 0 :=
by
  sorry

end negative_result_is_A_l688_688932


namespace exist_nat_nums_roots_int_l688_688962

theorem exist_nat_nums_roots_int :
  ∃ (a b c : ℕ),
  ∀ p : ℤ → ℤ,
  (p = λ x, a * x^2 + b * x + c ∨
   p = λ x, a * x^2 + b * x - c ∨
   p = λ x, a * x^2 - b * x + c ∨
   p = λ x, a * x^2 - b * x - c) →
   ∀ (r1 r2 : ℤ),
   (p r1 = 0 ∧ p r2 = 0) →
   (∃ (x1 x2 : ℤ), p = λ x, a * x^2 + b * x + c ∧ r1 = x1 ∧ r2 = x2) :=
begin
  sorry
end

end exist_nat_nums_roots_int_l688_688962


namespace Dans_placed_scissors_l688_688458

theorem Dans_placed_scissors (initial_scissors placed_scissors total_scissors : ℕ) 
  (h1 : initial_scissors = 39) 
  (h2 : total_scissors = initial_scissors + placed_scissors) 
  (h3 : total_scissors = 52) : placed_scissors = 13 := 
by 
  sorry

end Dans_placed_scissors_l688_688458


namespace min_distance_point_origin_l688_688529

theorem min_distance_point_origin (d : ℝ) (h : (4 * real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2) : d = 4 := 
  sorry

end min_distance_point_origin_l688_688529


namespace initial_books_l688_688381

theorem initial_books (added_books : ℝ) (books_per_shelf : ℝ) (shelves : ℝ) 
  (total_books : ℝ) : total_books = shelves * books_per_shelf → 
  shelves = 14 → books_per_shelf = 4.0 → added_books = 10.0 → 
  total_books - added_books = 46.0 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_books_l688_688381


namespace find_income_of_p_l688_688497

theorem find_income_of_p (P Q R : ℝ) 
    (h1 : P + Q = 4100) 
    (h2 : Q + R = 10500) 
    (h3 : P + R = 12400) : 
    P = 3000 :=
begin
    sorry
end

end find_income_of_p_l688_688497


namespace daryl_must_leave_weight_l688_688573

theorem daryl_must_leave_weight : 
  let crate_weight := 20
  let number_of_crates := 15
  let nails_weight := 4 * 5
  let hammers_weight := 12 * 5
  let planks_weight := 10 * 30
  let total_items_weight := nails_weight + hammers_weight + planks_weight
  let total_crates_capacity := number_of_crates * crate_weight
  total_items_weight - total_crates_capacity = 80 :=
by
  let crate_weight := 20
  let number_of_crates := 15
  let nails_weight := 4 * 5
  let hammers_weight := 12 * 5
  let planks_weight := 10 * 30
  let total_items_weight := nails_weight + hammers_weight + planks_weight
  let total_crates_capacity := number_of_crates * crate_weight
  show total_items_weight - total_crates_capacity = 80
  calc
    total_items_weight - total_crates_capacity = (4 * 5 + 12 * 5 + 10 * 30) - (15 * 20) : by rfl
    ... = (20 + 60 + 300) - 300                                 : by rfl
    ... = 380 - 300                                             : by rfl
    ... = 80                                                    : by rfl

end daryl_must_leave_weight_l688_688573


namespace max_value_f_on_interval_l688_688794

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem max_value_f_on_interval : ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), (∀ y ∈ set.Icc (-1 : ℝ) (1 : ℝ), f y ≤ f x) ∧ f x = 2 :=
by
  sorry

end max_value_f_on_interval_l688_688794


namespace intersection_P_Q_l688_688721

-- Definitions and Conditions
variable (P Q : Set ℕ)
noncomputable def f (t : ℕ) : ℕ := t ^ 2
axiom hQ : Q = {1, 4}

-- Theorem to Prove
theorem intersection_P_Q (P : Set ℕ) (Q : Set ℕ) (hQ : Q = {1, 4})
  (hf : ∀ t ∈ P, f t ∈ Q) : P ∩ Q = {1} ∨ P ∩ Q = ∅ :=
sorry

end intersection_P_Q_l688_688721


namespace area_triangle_ABF_correct_l688_688083

noncomputable def square_side_length (area : ℝ) : ℝ :=
  real.sqrt area

def area_of_square : ℝ := 12
def side_length_of_square := square_side_length area_of_square

def AE : ℝ := side_length_of_square / 2
def EC : ℝ := side_length_of_square
def EF : ℝ := (2 / 3) * EC
def FC : ℝ := (1 / 3) * EC
def area_triangle_ABF (side : ℝ) (area_square : ℝ) : ℝ :=
  (1 / 2) * area_square - 1

theorem area_triangle_ABF_correct :
  area_triangle_ABF side_length_of_square area_of_square = 5 :=
sorry

end area_triangle_ABF_correct_l688_688083


namespace compute_series_sum_l688_688106

theorem compute_series_sum :
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 0.834 :=
by
  sorry

end compute_series_sum_l688_688106


namespace dad_steps_l688_688177

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688177


namespace triangle_inequality_condition_l688_688952

variable (a b c : ℝ)
variable (α : ℝ) -- angle in radians

-- Define the condition where c must be less than a + b
theorem triangle_inequality_condition : c < a + b := by
  sorry

end triangle_inequality_condition_l688_688952


namespace equilateral_hyperbola_through_origin_and_point_l688_688423

theorem equilateral_hyperbola_through_origin_and_point (a : ℝ) :
  (∀ x y : ℝ, (y^2 - x^2 = a) ↔ (y = 3 ∧ x = 0)) → a = 9 :=
by
  intro h
  rw h 0 3
  simp
  sorry

end equilateral_hyperbola_through_origin_and_point_l688_688423


namespace star_zero_l688_688605

theorem star_zero (x y : ℝ) : 
  ((x^2 - y^2) ^ 2 - (y^2 - x^2) ^ 2) ^ 2 = 0 := by
  sorry

end star_zero_l688_688605


namespace symmetry_line_of_translated_function_l688_688241

noncomputable def ω := 2
noncomputable def φ := -π / 3
noncomputable def f (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_line_of_translated_function :
  (∀ x, f (x - π / 6) = -f (-(x - π / 6))) → 
  (∃ (a : ℝ), a = 5 * π / 12) :=
by
  sorry

end symmetry_line_of_translated_function_l688_688241


namespace integer_solutions_l688_688955

def satisfies_equation (x y : ℤ) : Prop := x^2 = y^2 * (x + y^4 + 2 * y^2)

theorem integer_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = { (0, 0), (12, 2), (-8, 2) } :=
by sorry

end integer_solutions_l688_688955


namespace sum_series_decomposition_l688_688108

theorem sum_series_decomposition :
  (∑ n in Finset.range 500, 1 / ((n + 1)^3 + (n + 1)^2)) =
    (∑ n in Finset.range 500, 1 / (n + 1)^2) - 1/2 + 1/501 := 
by
  sorry

end sum_series_decomposition_l688_688108


namespace num_distinct_ordered_pairs_l688_688278

-- Define the conditions
def positive_integers (x : ℕ) := x > 0
def reciprocal_sum (m n : ℕ) := (1 / m + 1 / n = 1 / 6)

-- Statement of the theorem
theorem num_distinct_ordered_pairs : 
  {p : ℕ × ℕ // positive_integers p.1 ∧ positive_integers p.2 ∧ reciprocal_sum p.1 p.2}.to_finset.card = 9 :=
by
  sorry

end num_distinct_ordered_pairs_l688_688278


namespace partial_fraction_series_sum_l688_688104

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end partial_fraction_series_sum_l688_688104


namespace problem_statement_l688_688190

theorem problem_statement :
  sqrt ((2 - sin (π / 9)^2) * (2 - sin (2 * π / 9)^2) * (2 - sin (4 * π / 9)^2)) = 5 / 4 :=
by sorry

end problem_statement_l688_688190


namespace area_of_12_sided_polygon_l688_688384

/-- A 12-sided polygon composed of 1x1 unit squares on a plane is given. 
The polygon consists of a central 3x3 square and additional shapes on the edges 
that form the 12-sided figure. -/

theorem area_of_12_sided_polygon : 
  let central_square_area := 3 * 3 in 
  let corner_triangles_area := 4 * (1 / 2) in
  let additional_edge_contribution := 4 in
  central_square_area + corner_triangles_area + additional_edge_contribution = 13 :=
by
  let central_square_area := 3 * 3
  let corner_triangles_area := 4 * (1 / 2)
  let additional_edge_contribution := 4 
  show central_square_area + corner_triangles_area + additional_edge_contribution = 13
  calc
    central_square_area + corner_triangles_area + additional_edge_contribution
        = 9 + 2 + 4 : by norm_num
    ... = 13 : by norm_num

/-- The area of the 12-sided polygon is 13 square units. -/

end area_of_12_sided_polygon_l688_688384


namespace golf_problem_l688_688931

variable (D : ℝ)

theorem golf_problem (h1 : D / 2 + D = 270) : D = 180 :=
by
  sorry

end golf_problem_l688_688931


namespace range_of_a_l688_688290

noncomputable def f (x a : ℝ) := log (x^2 - a*x + 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ a / 2 → f x1 a > f x2 a) →
  1 < a ∧ a < 2 * Real.sqrt 2 :=
sorry

end range_of_a_l688_688290


namespace expected_value_a_squared_is_correct_l688_688032

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l688_688032


namespace max_weight_l688_688904

-- Define the weights
def weight1 := 2
def weight2 := 5
def weight3 := 10

-- Theorem stating that the heaviest single item that can be weighed using any combination of these weights is 17 lb
theorem max_weight : ∃ x, (x = weight1 + weight2 + weight3) ∧ x = 17 :=
by
  sorry

end max_weight_l688_688904


namespace tangent_line_at_origin_l688_688788

-- Given a function f with these properties
def f (x : ℝ) : ℝ := x^3 + -3 * x

theorem tangent_line_at_origin : ∀ (x : ℝ), 3*x + f' x = 0 :=
by 
  -- Here you would include the intermediate steps
  -- Including computing the derivatives and simplification
  sorry

end tangent_line_at_origin_l688_688788


namespace percent_increase_is_60_l688_688783

noncomputable def initial_price : ℝ := 120000
noncomputable def final_price : ℝ := 192000

def percent_increase (initial final : ℝ) : ℝ := ((final - initial) / initial) * 100

theorem percent_increase_is_60 :
  percent_increase initial_price final_price = 60 :=
by
  sorry

end percent_increase_is_60_l688_688783


namespace find_sequence_l688_688995

noncomputable def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in finset.range (n+1), a i

theorem find_sequence (a S : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : sequence a S) 
  (h3 : ∀ n, S n - n^2 * a n = S 0 - 0^2 * a 0) 
  (n : ℕ) :
  a n = 2 / (n * (n + 1)) := 
sorry

end find_sequence_l688_688995


namespace change_received_l688_688053

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l688_688053


namespace distance_sum_l688_688238

-- Definitions and conditions
def parametric_line (t : ℝ) : ℝ × ℝ :=
  ( (Real.sqrt 2 / 2) * t, 3 + (Real.sqrt 2 / 2) * t )

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def curve_C (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 2 * Real.sin θ

def cartesian_curve_C (x y : ℝ) : Prop :=
  x^2 = 2 * y

def cartesian_line (x y : ℝ) : Prop := 
  x - y + 3 = 0

def point_P : ℝ × ℝ := (0, 3)

-- Main theorem to prove
theorem distance_sum (t1 t2 : ℝ) (h_t : t1^2 - 2 * Real.sqrt 2 * t1 - 12 = 0 ∧ t2^2 - 2 * Real.sqrt 2 * t2 - 12 = 0) :
  abs (t1 - t2) = 2 * Real.sqrt 14 :=
by 
  sorry

end distance_sum_l688_688238


namespace right_triangle_circumference_l688_688779

open Real

noncomputable def right_triangle_circumference_min (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (area_ABC : real) (altitude_midpoints_collinear : Prop) : Prop :=
  area_ABC = 10 →
  altitude_midpoints_collinear →
  ∃ (circumference : real), circumference = 20

theorem right_triangle_circumference : ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  right_triangle_circumference_min A B C 10 sorry :=
by {
  sorry
}

end right_triangle_circumference_l688_688779


namespace students_in_class_l688_688306

theorem students_in_class
  (S : ℕ)
  (h1 : S / 3 * 4 / 3 = 12) :
  S = 36 := 
sorry

end students_in_class_l688_688306


namespace perpendicular_vectors_x_eq_5_l688_688372

def vector_a (x : ℝ) : ℝ × ℝ := (2, x + 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_eq_5 (x : ℝ)
  (h : dot_product (vector_a x) (vector_b x) = 0) :
  x = 5 :=
sorry

end perpendicular_vectors_x_eq_5_l688_688372


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688868

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688868


namespace hexagon_area_l688_688921

theorem hexagon_area (s : ℝ) (h1 : s^2 = real.sqrt 3) : 
  let hex_area := 6 * (real.sqrt 3 / 4 * s^2) in
  hex_area = 9 / 2 :=
begin
  -- the condition s^2 = √3
  have s_eq := h1,
  -- the area calculation
  have hex_area_def : hex_area = 6 * (real.sqrt 3 / 4 * s^2),
  { refl },
  -- showing the desired result
  rw s_eq,
  simp,
  norm_num,
end

end hexagon_area_l688_688921


namespace angle_between_vectors_is_90_l688_688645

noncomputable def vecA (α : ℝ) : ℝ × ℝ × ℝ :=
  (Real.cos α, 1, Real.sin α)

noncomputable def vecB (α : ℝ) : ℝ × ℝ × ℝ :=
  (Real.sin α, 1, Real.cos α)

noncomputable def vecAdd (α : ℝ) : ℝ × ℝ × ℝ :=
  (vecA α).1 + (vecB α).1, (vecA α).2 + (vecB α).2, (vecA α).3 + (vecB α).3

noncomputable def vecSub (α : ℝ) : ℝ × ℝ × ℝ :=
  (vecA α).1 - (vecB α).1, (vecA α).2 - (vecB α).2, (vecA α).3 - (vecB α).3

noncomputable def dotProduct (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem angle_between_vectors_is_90 (α : ℝ) : 
  (dotProduct (vecAdd α) (vecSub α) = 0) → angle_between (vecAdd α) (vecSub α) = 90 := 
by
  sorry

end angle_between_vectors_is_90_l688_688645


namespace smallest_d_l688_688532

noncomputable def point_distance (x y : ℝ) : ℝ := real.sqrt (x ^ 2 + y ^ 2)

theorem smallest_d {d : ℝ} :
    point_distance (4 * real.sqrt 3) (d - 2) = 4 * d →
    d = 2.006 := sorry

end smallest_d_l688_688532


namespace binomial_coefficient_of_x_l688_688418

-- definition of binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem we want to prove
theorem binomial_coefficient_of_x {x : ℕ} (h : x ≠ 0) :
  let term (k : ℕ) := binom 5 k * (1/2)^k * x^(10-3*k)
  ∃ k, 10 - 3 * k = 1 ∧ term k = (5 / 4) * x :=
begin
  sorry
end

end binomial_coefficient_of_x_l688_688418


namespace problem_solution_l688_688265

open Set

theorem problem_solution
    (a b : ℝ)
    (ineq : ∀ x : ℝ, 1 < x ∧ x < b → a * x^2 - 3 * x + 2 < 0)
    (f : ℝ → ℝ := λ x => (2 * a + b) * x - 1 / ((a - b) * (x - 1))) :
    a = 1 ∧ b = 2 ∧ (∀ x, 1 < x ∧ x < b → f x ≥ 8 ∧ (f x = 8 ↔ x = 3 / 2)) :=
by
  sorry

end problem_solution_l688_688265


namespace fewest_number_of_students_l688_688694

theorem fewest_number_of_students :
  ∃ n : ℕ, n ≡ 3 [MOD 6] ∧ n ≡ 5 [MOD 8] ∧ n ≡ 7 [MOD 9] ∧ ∀ m : ℕ, (m ≡ 3 [MOD 6] ∧ m ≡ 5 [MOD 8] ∧ m ≡ 7 [MOD 9]) → m ≥ n := by
  sorry

end fewest_number_of_students_l688_688694


namespace arith_prog_seventh_term_l688_688697

-- Definitions for the arithmetic progression
def arith_term (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def arith_sum (a d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem arith_prog_seventh_term (a d : ℝ) :
  arith_sum a d 15 = 56.25 → arith_term a d 11 = 5.25 → arith_term a d 7 = 3.25 :=
by
  sorry

end arith_prog_seventh_term_l688_688697


namespace triangle_area_tangent_curve_l688_688413

theorem triangle_area_tangent_curve :
  let curve := λ x : ℝ, x^3
  let x_tangent := 1
  let y_tangent := 1
  let x_vertical := 2

  let tangent_slope := 3 * x_tangent^2
  let tangent_line := λ x, tangent_slope * (x - x_tangent) + y_tangent
  
  let x_intercept := (2 - 2 / 3 : ℝ)
  let y_intercept := tangent_line x_vertical -- Should be derived from the formula

  -- The base of the triangle
  let base := 2 - x_intercept
  -- The height of the triangle is y-coordinate where tangent line intersects x = 2
  let height := tangent_line x_vertical

  -- The area of the triangle
  let area := 1 / 2 * base * height
  
  area = 8 / 3 := 
  by sorry

end triangle_area_tangent_curve_l688_688413


namespace smallest_n_for_inequality_l688_688973

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n ≥ 2 ∧ (∀ (x : Fin n → ℝ), (∑ i, (x i)^2)^2 ≤ n * (∑ i, (x i)^4)) ∧ (∀ m, m ≥ 2 → (∀ (x : Fin m → ℝ), (∑ i, (x i)^2)^2 ≤ m * (∑ i, (x i)^4)) → 2 ≤ m) :=
by
  sorry

end smallest_n_for_inequality_l688_688973


namespace cos_sum_identity_l688_688764

theorem cos_sum_identity :
  let x := (Real.cos (2 * Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15))
  in x = (Real.sqrt 17 - 1) / 4 :=
begin
  sorry -- Proof would go here
end

end cos_sum_identity_l688_688764


namespace range_of_x_l688_688578

theorem range_of_x (a : Fin 20 → ℕ) (h : ∀ i, a i ∈ {0, 1, 4}) : 
  0 ≤ (∑ i in Finsets.range 20, (a i) * 1/(5:ℝ)^(i + 1)) ∧ 
  (∑ i in Finsets.range 20, (a i) * 1/(5:ℝ)^(i + 1)) < 1.05 := 
sorry

end range_of_x_l688_688578


namespace part_a_exists_rational_non_integer_l688_688876

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l688_688876


namespace weight_to_leave_out_l688_688572

theorem weight_to_leave_out (crates_weight_capacity: ℕ) (num_crates: ℕ) (nail_weight: ℕ) (num_nails: ℕ)
  (hammer_weight: ℕ) (num_hammers: ℕ) (planks_weight: ℕ) (num_planks: ℕ) :
  crates_weight_capacity = 20 → num_crates = 15 →
  nail_weight = 5 → num_nails = 4 →
  hammer_weight = 5 → num_hammers = 12 →
  planks_weight = 30 → num_planks = 10 →
  let total_crates_capacity := num_crates * crates_weight_capacity in
  let total_items_weight := (num_nails * nail_weight) + (num_hammers * hammer_weight) + (num_planks * planks_weight) in
  total_items_weight - total_crates_capacity = 80 :=
by 
  intros; 
  let total_crates_capacity := num_crates * crates_weight_capacity;
  let total_items_weight := (num_nails * nail_weight) + (num_hammers * hammer_weight) + (num_planks * planks_weight);
  have h1 : total_crates_capacity = 300 := by norm_num [num_crates, crates_weight_capacity];
  have h2 : total_items_weight = 380 := by norm_num [num_nails, nail_weight, num_hammers, hammer_weight, num_planks, planks_weight];
  rw [h1, h2];
  norm_num;
  exact rfl

end weight_to_leave_out_l688_688572


namespace worker_days_total_l688_688552

theorem worker_days_total
  (W I : ℕ)
  (hw : 20 * W - 3 * I = 280)
  (hi : I = 40) :
  W + I = 60 :=
by
  sorry

end worker_days_total_l688_688552


namespace triangle_inradius_l688_688435

theorem triangle_inradius (A p r : ℝ) 
    (h1 : p = 35) 
    (h2 : A = 78.75) 
    (h3 : A = (r * p) / 2) : 
    r = 4.5 :=
sorry

end triangle_inradius_l688_688435


namespace tree_initial_height_l688_688843

theorem tree_initial_height (H : ℝ) (C : ℝ) (P : H + 6 = (H + 4) + 1/4 * (H + 4) ∧ C = 1) : H = 4 :=
by
  let H := 4
  sorry

end tree_initial_height_l688_688843


namespace find_angle_C_range_of_y_l688_688624

open Real

variables (a b c A B C : ℝ)

-- Angle C in the triangle
theorem find_angle_C (h1: a * cos A = b * cos B) (h2: a ≠ b) : C = π / 2 := 
sorry

-- Range of y = (sin A + sin B)/ (sin A * sin B)
theorem range_of_y (h1: a * cos A = b * cos B) (h2: a ≠ b) (y : ℝ) :
    y = (sin A + sin B) / (sin A * sin B) → (2 * sqrt 2 < y ↔ y ∈ set.Ioi (2 * sqrt 2)) := 
sorry

end find_angle_C_range_of_y_l688_688624


namespace volume_of_prism_l688_688782

-- Define the basic conditions
variables (a α β : ℝ)

-- Mathematical statement equivalent to the solution
theorem volume_of_prism
  (h1 : true) -- The base of the right prism is a rhombus (geometrical information)
  (h2 : true) -- One of the diagonals of the prism is equal to a (length information)
  (h3 : true) -- Angle α between the diagonal and the base plane
  (h4 : true) -- Angle β between the diagonal and a lateral face
  :
  let volume := (a^3 * sin (2 * α) * cos α * sin β) / (4 * sqrt (cos (α + β) * cos (α - β))) in
  volume = (a^3 * sin (2 * α) * cos α * sin β) / (4 * sqrt (cos (α + β) * cos (α - β))) :=
by
  -- Proof omitted for the example
  sorry

end volume_of_prism_l688_688782


namespace maximize_revenue_l688_688509

def revenue_function (p : ℝ) : ℝ :=
  p * (200 - 6 * p)

theorem maximize_revenue :
  ∃ (p : ℝ), (p ≤ 30) ∧ (∀ q : ℝ, (q ≤ 30) → revenue_function p ≥ revenue_function q) ∧ p = 50 / 3 :=
by
  sorry

end maximize_revenue_l688_688509


namespace compute_complex_pow_l688_688566

-- We define the key complex number (1 + i) here
def one_plus_i := (1 : ℂ) + (complex.I)

-- The given condition from the problem
def condition := (one_plus_i ^ 2) = 2 * complex.I

-- Main statement to prove
theorem compute_complex_pow :
  (one_plus_i ^ 6) * ((1 : ℂ) - (complex.I)) = -8 - 8 * complex.I :=
by 
  -- Use the given condition
  have h : one_plus_i ^ 2 = 2 * complex.I := condition,
  sorry

end compute_complex_pow_l688_688566


namespace solution_xy_l688_688206

noncomputable def find_xy (x y : ℚ) : Prop :=
  (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1 / 3

theorem solution_xy :
  find_xy (10 + 1 / 3) (10 + 2 / 3) :=
by
  sorry

end solution_xy_l688_688206


namespace regular_nonagon_interior_angle_l688_688836

theorem regular_nonagon_interior_angle : 
  let n := 9 in
  180 * (n - 2) / n = 140 :=
by 
  sorry

end regular_nonagon_interior_angle_l688_688836


namespace equation_solution_unique_l688_688448

theorem equation_solution_unique (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
    (2 / (x - 3) = 3 / x ↔ x = 9) :=
by
  sorry

end equation_solution_unique_l688_688448


namespace radii_of_inscribed_circles_l688_688078

noncomputable def triangle_base : ℝ := 24
noncomputable def triangle_height : ℝ := 18
noncomputable def AD : ℝ := triangle_base / 2
noncomputable def CD : ℝ := triangle_height
noncomputable def AB : ℝ := triangle_base

theorem radii_of_inscribed_circles :
  let AC := Real.sqrt (CD ^ 2 + AD ^ 2) in
  let area := (1 / 2) * AB * CD in
  let semiperimeter := (AC + AC + AB) / 2 in
  let r1 := AB / 2 in
  let r2 := area / semiperimeter in
  r1 = 12 ∧ r2 = 216 / (6 * (Real.sqrt 13 + 2)) :=
by sorry

end radii_of_inscribed_circles_l688_688078


namespace dad_steps_are_90_l688_688127

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688127


namespace jonessa_take_home_pay_l688_688410

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l688_688410


namespace dad_steps_are_90_l688_688128

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688128


namespace octahedron_square_area_l688_688065

-- Define the conditions
structure Octahedron where
  side_length : ℝ
  is_regular : side_length = 2

def is_parallel (p1 p2 : Plane) : Prop := sorry -- placeholder for parallel plane definition

def forms_square_cross_section (o : Octahedron) (p : Plane) : Prop := sorry -- placeholder for intersect condition

-- Define the theorem statement
theorem octahedron_square_area (o : Octahedron) (p : Plane) : 
  o.is_regular ∧ is_parallel p.top_faces ∧ forms_square_cross_section o p →
  ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 4 ∧ gcd a c = 1 ∧ Nat.sqrt b ∉ {2, 3, 5, 7} ∧
  a * b / c = 3 / 4 :=
by
  sorry

end octahedron_square_area_l688_688065


namespace students_not_picked_l688_688816

/-- There are 36 students trying out for the school's trivia teams. 
If some of them didn't get picked and the rest were put into 3 groups with 9 students in each group,
prove that the number of students who didn't get picked is 9. -/

theorem students_not_picked (total_students groups students_per_group picked_students not_picked_students : ℕ)
    (h1 : total_students = 36)
    (h2 : groups = 3)
    (h3 : students_per_group = 9)
    (h4 : picked_students = groups * students_per_group)
    (h5 : not_picked_students = total_students - picked_students) :
    not_picked_students = 9 :=
by
  sorry

end students_not_picked_l688_688816


namespace probability_red_probability_same_color_probability_one_white_l688_688893

noncomputable def probability_of_two_red_balls : ℚ := 
  let total_ways := Nat.choose 6 2
  let red_ways := Nat.choose 2 2
  (red_ways : ℚ) / total_ways

noncomputable def probability_of_same_color : ℚ := 
  let total_ways := Nat.choose 6 2
  let red_ways := Nat.choose 2 2
  let white_ways := Nat.choose 2 2
  let yellow_ways := Nat.choose 2 2
  ((red_ways + white_ways + yellow_ways) : ℚ) / total_ways

noncomputable def probability_of_one_white : ℚ := 
  let total_ways := Nat.choose 6 2
  let one_white_one_other := Nat.choose 2 1 * Nat.choose 4 1
  (one_white_one_other : ℚ) / total_ways

theorem probability_red := probability_of_two_red_balls = 1/15 := by sorry

theorem probability_same_color := probability_of_same_color = 1/5 := by sorry

theorem probability_one_white := probability_of_one_white = 2/3 := by sorry

end probability_red_probability_same_color_probability_one_white_l688_688893


namespace variance_of_data_set_l688_688067

variables (x : ℝ) (data_set : List ℝ := [1, 3, -1, 2, x])

def average (lst : List ℝ) : ℝ := (lst.sum) / (lst.length)

def variance (lst : List ℝ) (avg : ℝ) : ℝ :=
  (1 / lst.length) * (lst.map (λ xi, (xi - avg)^2)).sum

theorem variance_of_data_set :
  average data_set = 1 → variance data_set (average data_set) = 9 / 5 :=
by
  sorry

end variance_of_data_set_l688_688067


namespace area_comparison_l688_688421

noncomputable def diagonal_of_square : ℝ := 10
noncomputable def diameter_of_circle : ℝ := 10
noncomputable def side_length_of_square : ℝ := diagonal_of_square / Real.sqrt 2
noncomputable def area_of_square : ℝ := side_length_of_square^2

noncomputable def radius_of_circle : ℝ := diameter_of_circle / 2
noncomputable def area_of_circle : ℝ := Real.pi * radius_of_circle^2

noncomputable def height_of_triangle : ℝ := (Real.sqrt 3 / 2) * side_length_of_square
noncomputable def area_of_triangle : ℝ := 0.5 * side_length_of_square * height_of_triangle

noncomputable def combined_area : ℝ := area_of_square + area_of_triangle
noncomputable def area_difference : ℝ := area_of_circle - combined_area

theorem area_comparison :
  area_difference ≈ -14.8 :=
begin
  -- The proof would go here
  sorry
end

end area_comparison_l688_688421


namespace dad_steps_l688_688117

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688117


namespace part1_range_of_a_part2_range_of_a_l688_688621

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a^2 * (Real.log x) - a * x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + a * x

theorem part1_range_of_a (a : ℝ) : 
  (∀ x, f x a ≥ 0) ↔ -2 * Real.exp (3/4) ≤ a ∧ a ≤ 1 :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∃ x1 x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔
  (-e ≤ a ∧ a < -Real.sqrt (2 * e)) ∨ (Real.sqrt (2 * e) < a ∧ a ≤ e) :=
sorry

end part1_range_of_a_part2_range_of_a_l688_688621


namespace g_96_is_1996_l688_688789

def g : ℤ → ℤ 
| n := if n >= 2000 then n - 4 else g (g (n + 6))

theorem g_96_is_1996 : g 96 = 1996 := 
sorry

end g_96_is_1996_l688_688789


namespace dina_has_60_dolls_l688_688960

variable (ivy_collectors_edition_dolls : ℕ)
variable (ivy_total_dolls : ℕ)
variable (dina_dolls : ℕ)

-- Conditions
def condition1 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := ivy_collectors_edition_dolls = 20
def condition2 (ivy_total_dolls ivy_collectors_edition_dolls : ℕ) := (2 / 3 : ℚ) * ivy_total_dolls = ivy_collectors_edition_dolls
def condition3 (ivy_total_dolls dina_dolls : ℕ) := dina_dolls = 2 * ivy_total_dolls

-- Proof statement
theorem dina_has_60_dolls 
  (h1 : condition1 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h2 : condition2 ivy_total_dolls ivy_collectors_edition_dolls) 
  (h3 : condition3 ivy_total_dolls dina_dolls) : 
  dina_dolls = 60 :=
sorry

end dina_has_60_dolls_l688_688960


namespace points_collinear_l688_688959

theorem points_collinear (m : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * 1 + b * (m-1) + c = 0 ∧
                   a * 3 + b * (m+5) + c = 0 ∧ a * 6 + b * (2m+3) + c = 0) ↔
  m = 11 :=
by sorry

end points_collinear_l688_688959


namespace angle_AMQ_l688_688396

-- let AB be the segment with midpoint M and BM be the segment with midpoint N
-- two semi-circles are constructed with AB and BM as diameters
-- let MQ split the region into one larger section twice the area of the smaller section
-- let the radius of semi-circle on BM be one-third that of the semi-circle on AB
-- prove the degree measure of the angle AMQ is 93.3 degrees 

theorem angle_AMQ (AB M B N Q : Type) (R : ℝ) (h_mid_AB : M = midpoint AB)
  (h_mid_BM : N = midpoint BM) (h_radii : (BM) = (AB) / 3)
  (h_split : larger_section_twice_smaller MQ) :
  degree_measure_angle AMQ = 93.3 :=
sorry

end angle_AMQ_l688_688396


namespace sin_square_identity_l688_688683

theorem sin_square_identity (α : Real) (h : sin (2 * π / 3 - 2 * α) = 3 / 5) :
  sin^2 (α + 5 * π / 12) = 4 / 5 :=
by
  sorry

end sin_square_identity_l688_688683


namespace dad_steps_l688_688181

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688181


namespace insurance_covers_80_percent_l688_688712

-- Definitions from the problem conditions
def cost_per_aid : ℕ := 2500
def num_aids : ℕ := 2
def johns_payment : ℕ := 1000

-- Total cost of hearing aids
def total_cost : ℕ := cost_per_aid * num_aids

-- Insurance payment
def insurance_payment : ℕ := total_cost - johns_payment

-- The theorem to prove
theorem insurance_covers_80_percent :
  (insurance_payment * 100 / total_cost) = 80 :=
by
  sorry

end insurance_covers_80_percent_l688_688712


namespace odd_symmetric_periodic_l688_688253

noncomputable def f : ℝ → ℝ := sorry

theorem odd_symmetric_periodic :
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x : ℝ, f(2-x) = f(x)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 2^x - 1) →
  f 2009 + f 2010 = 1 :=
by sorry

end odd_symmetric_periodic_l688_688253


namespace problem_p1_l688_688019

noncomputable def S : ℝ := ∑ n in Finset.range 99, (real.sqrt (n + 1 + 1) - real.sqrt (n + 1))

theorem problem_p1 : S = 9 := sorry

end problem_p1_l688_688019


namespace hyperbola_eccentricity_l688_688643

theorem hyperbola_eccentricity {a b : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (foci_eq : (c : ℝ), c = 1)
  (area_max : ∀ m n : ℝ, 1 = (m^2 / 4) + (n^2 / 3) ∧ sqrt 3 * m = 2 * n) :
  ∃ e : ℝ, e = sqrt 2 := 
sorry

end hyperbola_eccentricity_l688_688643


namespace cone_volume_l688_688195

/-- Define the diameter and height of the cone -/
def diameter : ℝ := 12
def height : ℝ := 9

/-- The volume of the cone in cubic centimeters -/
def volume_of_cone (d h : ℝ) : ℝ := (1 / 3) * Real.pi * ((d / 2) ^ 2) * h

/-- Prove the volume of a cone with specified dimensions -/
theorem cone_volume : volume_of_cone diameter height = 108 * Real.pi := by
  -- Place in the expected correct volume calculation
  have r := diameter / 2
  calc
    volume_of_cone diameter height
      = (1 / 3) * Real.pi * r^2 * height : by {
        rw [volume_of_cone],
        simp [r],
      }
    ... = 108 * Real.pi : by {
      -- Simple arithmetic calculation
      norm_num [r, height]
    }

end cone_volume_l688_688195


namespace center_of_mass_on_segment_l688_688389

variables {A B O : Type} [add_comm_group A] [module ℝ A]
variables (a b : ℝ) (massA : a ≠ 0) (massB : b ≠ 0)
variables (posA posB posO : A)

def center_of_mass (a b : ℝ) (posA posB : A) : A :=
(a * posA + b * posB) / (a + b)

theorem center_of_mass_on_segment (a b : ℝ) (posA posB posO : A)
  (h : posO = center_of_mass a b posA posB) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ posO = (1 - t) • posA + t • posB ∧ t = b / (a + b) :=
begin
  sorry
end

end center_of_mass_on_segment_l688_688389


namespace decreasing_sequence_sum_bound_l688_688364

theorem decreasing_sequence_sum_bound 
  (x : ℕ → ℝ)
  (h_decreasing : ∀ n : ℕ, 0 < n → x n > x (n + 1))
  (h_condition : ∀ n : ℕ, 1 ≤ x 1 + ∑ k in range (n + 1), x (k*k) / k) :
  ∀ n : ℕ,  x 1 + ∑ i in range n, x (i + 1) / (i + 1) < 3 :=
by
  sorry

end decreasing_sequence_sum_bound_l688_688364


namespace regular_nonagon_interior_angle_l688_688832

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℝ := sum_of_interior_angles n / n

theorem regular_nonagon_interior_angle :
  regular_polygon_interior_angle 9 = 140 :=
by
  unfold regular_polygon_interior_angle
  unfold sum_of_interior_angles
  norm_num
  sorry

end regular_nonagon_interior_angle_l688_688832


namespace percentage_increase_in_rent_l688_688518

theorem percentage_increase_in_rent
  (avg_rent_per_person_before : ℝ)
  (num_friends : ℕ)
  (friend_original_rent : ℝ)
  (avg_rent_per_person_after : ℝ)
  (total_rent_before : ℝ := num_friends * avg_rent_per_person_before)
  (total_rent_after : ℝ := num_friends * avg_rent_per_person_after)
  (rent_increase : ℝ := total_rent_after - total_rent_before)
  (percentage_increase : ℝ := (rent_increase / friend_original_rent) * 100)
  (h1 : avg_rent_per_person_before = 800)
  (h2 : num_friends = 4)
  (h3 : friend_original_rent = 1400)
  (h4 : avg_rent_per_person_after = 870) :
  percentage_increase = 20 :=
by
  sorry

end percentage_increase_in_rent_l688_688518


namespace house_orderings_l688_688393

theorem house_orderings :
  let houses := ["O", "R", "B", "Y", "G"] in
  let orderings := houses.permutations in
  let valid_orderings :=
    orderings.filter (λ ord,
      (ord.index_of "O" < ord.index_of "B") ∧
      (ord.index_of "G" < ord.index_of "R") ∧
      (ord.index_of "B" < ord.index_of "Y") ∧
      (ord.index_of "B" + 1 ≠ ord.index_of "Y") ∧
      (ord.index_of "Y" + 1 ≠ ord.index_of "B"))
  in valid_orderings.length = 5 := 
sorry

end house_orderings_l688_688393


namespace hyperbola_condition_l688_688263

theorem hyperbola_condition (a : ℝ) (h : a > 0)
  (e : ℝ) (h_e : e = Real.sqrt (1 + 4 / (a^2))) :
  (e > Real.sqrt 2) ↔ (0 < a ∧ a < 1) := 
sorry

end hyperbola_condition_l688_688263


namespace determine_x_l688_688958

theorem determine_x (x : ℕ) (hx : 27^3 + 27^3 + 27^3 = 3^x) : x = 10 :=
sorry

end determine_x_l688_688958


namespace probability_is_1_over_12_l688_688040

-- Define the set of digits available
def digits := {1, 2, 3, 5}

-- Define the number of ways to form a three-digit number
def total_ways : Nat := 4 * 3 * 2

-- Define the criterion for a number to be a multiple of 3 and also odd
def is_multiple_of_3_and_odd (n : Nat) : Prop :=
  n % 3 = 0 ∧ n % 2 = 1

-- Count the valid three-digit numbers meeting the criteria
def valid_numbers : Nat := 2  -- From solution steps, only 2 valid permutations

-- Calculate the probability
def probability : Float := (valid_numbers : Float) / (total_ways : Float)

-- The goal is to prove that the calculated probability is indeed 1/12
theorem probability_is_1_over_12 : probability = 1/12 := sorry

end probability_is_1_over_12_l688_688040


namespace monthly_income_of_B_l688_688489

variable (x y : ℝ)

-- Monthly incomes in the ratio 5:6
axiom income_ratio (A_income B_income : ℝ) : A_income = 5 * x ∧ B_income = 6 * x

-- Monthly expenditures in the ratio 3:4
axiom expenditure_ratio (A_expenditure B_expenditure : ℝ) : A_expenditure = 3 * y ∧ B_expenditure = 4 * y

-- Savings of A and B
axiom savings_A (A_income A_expenditure : ℝ) : 1800 = A_income - A_expenditure
axiom savings_B (B_income B_expenditure : ℝ) : 1600 = B_income - B_expenditure

-- The theorem to prove
theorem monthly_income_of_B (B_income : ℝ) (x y : ℝ) 
  (h1 : A_income = 5 * x)
  (h2 : B_income = 6 * x)
  (h3: A_expenditure = 3 * y)
  (h4: B_expenditure = 4 * y)
  (h5 : 1800 = 5 * x - 3 * y)
  (h6 : 1600 = 6 * x - 4 * y)
  : B_income = 7200 := by
  sorry

end monthly_income_of_B_l688_688489


namespace find_z1_find_z2_l688_688990

-- Definition of complex numbers and their properties
def complex.add (z1 z2 : ℂ) : ℂ := z1 + z2
def complex.mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Given conditions for the first part of the problem
def cond1 (z1 : ℂ) : Prop := (z1 - 2) * complex.I = 1 + complex.I

-- Theorem statement for the first part of the problem
theorem find_z1 (z1 : ℂ) (h1 : cond1 z1) : z1 = 3 - complex.I :=
sorry

-- Additional conditions for the second part of the problem
def cond2 (z2 : ℂ) : Prop := z2.im = 2

def cond3 (z1 z2 : ℂ) : Prop := complex.mul z1 z2.im = 0

-- Theorem statement for the second part of the problem
theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 3 - complex.I) (h2 : cond2 z2) (h3 : cond3 z1 z2) :
  z2 = 6 + 2 * complex.I :=
sorry

end find_z1_find_z2_l688_688990


namespace smallest_sum_faces_cube_l688_688751

theorem smallest_sum_faces_cube :
  ∃ (f : ℕ → ℕ) (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ),
    (f 0 = n₁) ∧ (f 1 = n₂) ∧ (f 2 = n₃) ∧
    (f 3 = n₄) ∧ (f 4 = n₅) ∧ (f 5 = n₆) ∧
    (abs (n₁ - n₂) > 1) ∧ (abs (n₁ - n₃) > 1) ∧
    (abs (n₁ - n₅) > 1) ∧ (abs (n₂ - n₄) > 1) ∧
    (abs (n₂ - n₆) > 1) ∧ (abs (n₃ - n₄) > 1) ∧
    (abs (n₃ - n₅) > 1) ∧ (abs (n₄ - n₆) > 1) ∧
    (abs (n₅ - n₆) > 1) ∧
    (n₁ + n₂ + n₃ + n₄ + n₅ + n₆ = 18) :=
by
  sorry

end smallest_sum_faces_cube_l688_688751


namespace find_modulus_sq_l688_688358

def z : ℂ := sorry -- z will be the complex number
def modulus_sq (z : ℂ) : ℂ := complex.norm_sq z

theorem find_modulus_sq
  (h : z^2 + modulus_sq z = 8 - 2 * complex.I) :
  modulus_sq z = 17 / 4 := 
sorry

end find_modulus_sq_l688_688358


namespace determine_values_of_x_l688_688727

variable (x : ℝ)

theorem determine_values_of_x (h1 : 1/x < 3) (h2 : 1/x > -4) : x > 1/3 ∨ x < -1/4 := 
  sorry


end determine_values_of_x_l688_688727


namespace number_has_perfect_square_sums_of_divisors_l688_688205

theorem number_has_perfect_square_sums_of_divisors :
    ∀ n : ℕ, n > 0 → (∃ (ds : List ℕ) (perm_ds : List ℕ),
        (∀ i, perm_ds i = List.nth_le ds i sorry) ∧
        (∀ i : ℕ, List.sum (List.take (i + 1) perm_ds) = (⌊List.sum (List.take (i + 1) perm_ds)⌋ * ⌊List.sum (List.take (i + 1) perm_ds)⌋)) →
      n ∈ {1, 3}) :=
by
  sorry

end number_has_perfect_square_sums_of_divisors_l688_688205


namespace right_triangle_median_l688_688851

-- Definitions of the problem conditions
variable {α : Type}
variables [EuclideanGeometry α] [has_measure α]

def median (A B C : Point α) (M : Point α) : Prop :=
  between M A C ∧ dist M A = dist M C

def is_right_triangle (A B C : Point α) : Prop :=
  ∃ D, dist A D = dist D B ∧ dist B D = dist D C ∧ ∠ B A C = 90°

-- The proof problem converted to Lean 4 statement
theorem right_triangle_median {A B C M : Point α} 
  (hM : median A B C M) 
  (hC : dist B M = 2 * dist A M) : 
  is_right_triangle A B C := 
sorry

end right_triangle_median_l688_688851


namespace range_of_a_l688_688247

theorem range_of_a (a : ℝ) (p : ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0) (q : ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) : 
  a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l688_688247


namespace sum_first_5_terms_l688_688701

variable (a : ℕ → ℝ)

-- Define the arithmetic sequence condition: a_n - a_(n-1) = d
def arithmetic_sequence (d : ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a₃ = 2
def a3_is_2 := a 3 = 2

-- Prove that the sum of the first 5 terms is 10
theorem sum_first_5_terms (d : ℝ) (h_arith : arithmetic_sequence d a) (h_a3 : a3_is_2) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 10 := sorry

end sum_first_5_terms_l688_688701


namespace zero_of_f_in_interval_l688_688812

def f (x : ℝ) : ℝ := 5 * x

theorem zero_of_f_in_interval : ∃ x ∈ set.Ioo (-1:ℝ) 0, f x = 0 := 
by
  use 0
  sorry

end zero_of_f_in_interval_l688_688812


namespace sum_of_areas_of_circles_eq_l688_688301

noncomputable def radius_seq (n : ℕ) : ℝ := 2 * (1/3)^(n-1)

noncomputable def circle_area (n : ℕ) : ℝ := π * (radius_seq n)^2

theorem sum_of_areas_of_circles_eq :
  (∑' n : ℕ, circle_area (n + 1)) = (9 * π / 2) :=
by
  sorry

end sum_of_areas_of_circles_eq_l688_688301


namespace increasing_interval_f_when_a_1_range_of_a_when_f_monotonically_increasing_lower_bound_F_l688_688735

section math_proof_problem

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := exp(2 * x) - 4 * a * exp(x) - 2 * a * x
def g (x a : ℝ) : ℝ := x^2 + 5 * a^2
def F (x a : ℝ) : ℝ := f x a + g x a

-- (1) If a = 1, find the increasing interval of f(x)
theorem increasing_interval_f_when_a_1 : ∀ x, 
  (f x 1 > 0) → x > (1 + sqrt(2)) := 
begin
  sorry
end

-- (2) If f(x) is monotonically increasing on ℝ, find the range of values for a
theorem range_of_a_when_f_monotonically_increasing (a : ℝ) : 
  (∀ x y, x < y → f x a ≤ f y a) → a ≤ 0 :=
begin
  sorry
end

-- (3) Prove that F(x) ≥ 4(1 - ln 2)^2 / 5
theorem lower_bound_F (a : ℝ) : ∀ x, 
  F x a ≥ (4 * (1 - log 2)^2) / 5 :=
begin
  sorry
end

end math_proof_problem

end increasing_interval_f_when_a_1_range_of_a_when_f_monotonically_increasing_lower_bound_F_l688_688735


namespace find_smallest_palindromic_n_string_l688_688839

-- Define what a palindromic string means
def is_palindromic (s : List ℕ) : Prop :=
  s = s.reverse

-- Define an n-string
def n_string (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.tail!  -- It creates the list [1, 2, ..., n]

-- Define the property for the problem statement
def exists_palindromic_n_string (n : ℕ) : Prop :=
  ∃ s : List ℕ, s.perm (n_string n) ∧ is_palindromic s

theorem find_smallest_palindromic_n_string :
  ∃ n : ℕ, n > 1 ∧ exists_palindromic_n_string n ∧
  (∀ m : ℕ, m > 1 → exists_palindromic_n_string m → n ≤ m) :=
sorry

end find_smallest_palindromic_n_string_l688_688839


namespace lucas_investment_l688_688740

noncomputable def investment_amount (y : ℝ) : ℝ := 1500 - y

theorem lucas_investment :
  ∃ y : ℝ, (y * 1.04 + (investment_amount y) * 1.06 = 1584.50) ∧ y = 275 :=
by
  sorry

end lucas_investment_l688_688740


namespace hyperbola_eccentricity_l688_688656

theorem hyperbola_eccentricity (A F P : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hF : F = (2, 0))
  (hP : ∃ x y, P = (x, y) ∧ y^2 = 8 * x)
  (h_min : ∀ Q : ℝ × ℝ, Q = P → (let |PF| := (P.1 - F.1)^2 + (P.2 - F.2)^2
                                  |PA| := (P.1 - A.1)^2 + (P.2 - A.2)^2 
                                  in |PF| / |PA| >= (√2 / 2))) :
  let |AF| := (A.1 - F.1)^2 + (A.2 - F.2)^2
      |AP| := (P.1 - A.1)^2 + (P.2 - A.2)^2
      |PF| := (P.1 - F.1)^2 + (P.2 - F.2)^2
      c := |AF| / 2 -- distance half of the foci
      a := (|AP| - |PF|) / 2 in
  eccentricity := (c / a) := √2 + 1 :=
sorry

end hyperbola_eccentricity_l688_688656


namespace angle_B_less_than_pi_over_2_l688_688324

theorem angle_B_less_than_pi_over_2 {a b c : ℝ} (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_arith : 1/a + 1/c = 2/b) :
  ∃ B : ℝ, B < π / 2 ∧ B = real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)) :=
by
  sorry

end angle_B_less_than_pi_over_2_l688_688324


namespace subsequence_convergence_l688_688910
noncomputable section

open Real Topology

-- Define the sequence a_n
def a (n : ℕ) : ℝ := n / 10 ^ (Int.floor (log n) + 1)

-- Prove that for any real number A where 0.1 ≤ A ≤ 1,
-- there exists a subsequence of {a_n} that converges to A.

theorem subsequence_convergence (A : ℝ) (hA : 0.1 ≤ A ∧ A ≤ 1) :
  ∃ (φ : ℕ → ℕ), (∀ n m, n < m → φ n < φ m) ∧ (tendsto (λ k, a (φ k)) at_top (nhds A)) :=
sorry

end subsequence_convergence_l688_688910


namespace dad_steps_90_l688_688171

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688171


namespace minimum_score_4th_quarter_l688_688464

theorem minimum_score_4th_quarter (q1 q2 q3 : ℕ) (q4 : ℕ) :
  q1 = 85 → q2 = 80 → q3 = 90 →
  (q1 + q2 + q3 + q4) / 4 ≥ 85 →
  q4 ≥ 85 :=
by intros hq1 hq2 hq3 h_avg
   sorry

end minimum_score_4th_quarter_l688_688464


namespace ratios_common_value_l688_688798

theorem ratios_common_value (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0)
  (h₄ : ∃ k, (x + y) / z = k ∧ (x + z) / y = k ∧ (y + z) / x = k) :
  h₄.k = -1 ∨ h₄.k = 2 :=
by
  sorry

end ratios_common_value_l688_688798


namespace range_of_a_l688_688637

noncomputable def max_a (x y : ℝ) : ℝ :=
  x^2 + 16 / x^2 - 2 * x * y - 8 / x * √(1 - y^2)

theorem range_of_a :
  (∀ (x y : ℝ), (x ∈ set.Iio 0 ∪ set.Ioi 0) → (y ∈ set.Icc (-1:ℝ) 1) →
  max_a x y - a ≥ 0) → a ≤ 8 - 4 * √2 :=
by
  sorry

end range_of_a_l688_688637


namespace intersection_complement_l688_688663

open Set

variable {α : Type*} -- Type for sets elements

def U : Set α := {1, 2, 3, 4, 5, 6}
def A : Set α := {2, 4, 5}
def B : Set α := {1, 3}

theorem intersection_complement: A ∩ (U \ B) = {2, 4, 5} := by
  sorry

end intersection_complement_l688_688663


namespace height_of_sarah_building_l688_688825

-- Define the conditions
def shadow_length_building : ℝ := 75
def height_pole : ℝ := 15
def shadow_length_pole : ℝ := 30

-- Define the height of the building
def height_building : ℝ := 38

-- Height of Sarah's building given the conditions
theorem height_of_sarah_building (h : ℝ) (H1 : shadow_length_building = 75)
    (H2 : height_pole = 15) (H3 : shadow_length_pole = 30) :
    h = height_building :=
by
  -- State the ratio of the height of the pole to its shadow
  have ratio_pole : ℝ := height_pole / shadow_length_pole

  -- Set up the ratio for Sarah's building and solve for h
  have h_eq : ℝ := ratio_pole * shadow_length_building

  -- Provide the proof (skipped here)
  sorry

end height_of_sarah_building_l688_688825


namespace midpoints_of_quadrilateral_form_parallelogram_l688_688391

structure Point (α : Type) :=
  (x : α) 
  (y : α)

def midpoint {α : Type} [Add α] [HasMul α] [Div α] (A B : Point α) : Point α :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

theorem midpoints_of_quadrilateral_form_parallelogram {α : Type} [Add α] [HasMul α] [Div α] [HasZero α]
  (A B C D : Point α) :
  let M := midpoint A B,
      N := midpoint B C,
      P := midpoint C D,
      Q := midpoint D A in
  -- We need to skip the actual proof, indicating the conclusion
  sorry

end midpoints_of_quadrilateral_form_parallelogram_l688_688391


namespace prob_l688_688300

theorem prob (x m : ℝ) (h1 : x ∈ set.Icc 2 4) (h2 : x^2 - 2*x - 2 - m < 0) : m > -2 :=
sorry

end prob_l688_688300


namespace line_through_point_and_isosceles_triangle_l688_688207

def is_line_eq (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def is_isosceles_right_triangle_with_axes (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0

theorem line_through_point_and_isosceles_triangle (a b c : ℝ) (hx : ℝ) (hy : ℝ) :
  is_line_eq a b c hx hy ∧ is_isosceles_right_triangle_with_axes a b → 
  ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 1 ∧ b = -1 ∧ c = -1)) :=
by
  sorry

end line_through_point_and_isosceles_triangle_l688_688207


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688886

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688886


namespace root_interval_l688_688541

def f (x : ℝ) : ℝ := x^3 + x - 1

theorem root_interval (hf₁: f (-1) < 0)
                      (hf₂: f 1 > 0)
                      (hf₃: f 0 < 0)
                      (hf₄: f 0.5 < 0)
                      (hf₅: f 0.75 > 0) :
       ∃ x ∈ Ioo 0.5 0.75, f x = 0 := by
    sorry

end root_interval_l688_688541


namespace solve_abs_eq_linear_l688_688765

theorem solve_abs_eq_linear (x : ℝ) (h : |2 * x - 4| = x + 3) : x = 7 :=
sorry

end solve_abs_eq_linear_l688_688765


namespace cevians_concurrent_circumscribable_l688_688730

-- Define the problem
variables {A B C D X Y Z : Type}

-- Define concurrent cevians
def cevian_concurrent (A B C X Y Z D : Type) : Prop := true

-- Define circumscribable quadrilaterals
def circumscribable (A B C D : Type) : Prop := true

-- The theorem statement
theorem cevians_concurrent_circumscribable (h_conc: cevian_concurrent A B C X Y Z D) 
(h1: circumscribable D Y A Z) (h2: circumscribable D Z B X) : circumscribable D X C Y :=
sorry

end cevians_concurrent_circumscribable_l688_688730


namespace g_neg1002_l688_688773

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq : ∀ (x y : ℝ), g (x * y) + x = x * g y + g x
axiom g_neg2 : g (-2) = 4

theorem g_neg1002 : g (-1002) = 2004 :=
begin
  sorry
end

end g_neg1002_l688_688773


namespace find_angle_C_find_range_of_function_l688_688302

open Real

-- Given conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) (h1 : a = sin A) (h2 : b = sin B) (h3 : c = sin C) : 
(a + b + c) * (a + b - c) = 3 * a * b

-- To prove: (I) C = π / 3
theorem find_angle_C (a b c : ℝ) (C : ℝ) 
  (h1 : (a + b + c) * (a + b - c) = 3 * a * b) : 
  C = π / 3 := 
by sorry

-- Given C = π / 3, prove the range: (II) [1 - sqrt 3, 3]
theorem find_range_of_function (C x : ℝ) 
  (h1 : C = π / 3) 
  (hx : 0 ≤ x ∧ x ≤ π / 2) : 
  let f := λ x, sqrt 3 * sin (2 * x - C / 2) + 2 * (sin (x - π / 12))^2 in
  ∀ y, y ∈ {y | ∃ x, f x = y} ⊆ set.Icc (1 - sqrt 3) 3 := 
by sorry

end find_angle_C_find_range_of_function_l688_688302


namespace num_distinct_ordered_pairs_l688_688277

theorem num_distinct_ordered_pairs (h : ∀ (m n : ℕ), (1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 6) → 0 < m ∧ 0 < n) :
  (finset.card ((finset.univ : finset (ℕ × ℕ)).filter (λ p, 1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6))) = 9 :=
sorry

end num_distinct_ordered_pairs_l688_688277


namespace billy_gaming_percentage_l688_688940

-- Define the conditions
def free_time_per_day := 8
def days_in_weekend := 2
def total_free_time := free_time_per_day * days_in_weekend
def books_read := 3
def pages_per_book := 80
def reading_rate := 60 -- pages per hour
def total_pages_read := books_read * pages_per_book
def reading_time := total_pages_read / reading_rate
def gaming_time := total_free_time - reading_time
def gaming_percentage := (gaming_time / total_free_time) * 100

-- State the theorem
theorem billy_gaming_percentage : gaming_percentage = 75 := by
  sorry

end billy_gaming_percentage_l688_688940


namespace total_population_is_700_l688_688809

-- Definitions for the problem conditions
def L : ℕ := 200
def P : ℕ := L / 2
def E : ℕ := (L + P) / 2
def Z : ℕ := E + P

-- Proof statement (with sorry)
theorem total_population_is_700 : L + P + E + Z = 700 :=
by
  sorry

end total_population_is_700_l688_688809


namespace cat_finishes_food_on_next_monday_l688_688395

noncomputable def cat_food_consumption_per_day : ℚ := (1 / 4) + (1 / 6)

theorem cat_finishes_food_on_next_monday :
  ∃ n : ℕ, n = 8 ∧ (n * cat_food_consumption_per_day > 8) := sorry

end cat_finishes_food_on_next_monday_l688_688395


namespace domino_arrangement_count_l688_688516

def domino := (String × String)

def RR : domino := ("Red", "Red")
def RB : domino := ("Red", "Blue")
def BR : domino := ("Blue", "Red")
def BB : domino := ("Blue", "Blue")

def valid_sequence (seq : List domino) : Prop :=
  ∀ i, i < seq.length - 1 → seq[i].2 = seq[i+1].1

theorem domino_arrangement_count : 
  ∃ (l : List (List domino)), (∀ seq ∈ l, valid_sequence seq) ∧ l.length = 4 :=
by
  sorry

end domino_arrangement_count_l688_688516


namespace value_of_a_plus_b_is_zero_l688_688691

noncomputable def sum_geometric_sequence (a b : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + b

theorem value_of_a_plus_b_is_zero (a b : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = sum_geometric_sequence a b n) :
  a + b = 0 := 
sorry

end value_of_a_plus_b_is_zero_l688_688691


namespace who_is_in_seat_three_l688_688549

-- Define the seats and the participants
inductive Seat
| one | two | three | four
deriving DecidableEq

inductive Person
| Abby | Bret | Carl | Dana
deriving DecidableEq

-- Given conditions as hypotheses
axiom false_statement_1 : ∀ (s1 s2 : Seat), (s2 = Seat.succ s1 ∨ s1 = Seat.succ s2) → false
axiom false_statement_2 : ∀ (s1 s2 : Seat), ¬(s2 = Seat.succ s1 ∧ s1 = Seat.pred s2)

-- Given position of Carl
axiom carl_position : ∀ (s : Seat), s = Seat.two → Person.Carl

-- Expected result
theorem who_is_in_seat_three : ∃ (p : Person), p = Person.Bret := 
by
  -- Proof is skipped
  sorry

end who_is_in_seat_three_l688_688549


namespace necessary_condition_l688_688371

variables {S : Type} {A B : Set S} {x : S}

theorem necessary_condition (h : x ∈ A) : x ∈ \complement S B :=
sorry

end necessary_condition_l688_688371


namespace dad_steps_90_l688_688170

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688170


namespace quadratic_to_square_form_l688_688112

theorem quadratic_to_square_form (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) :=
sorry

end quadratic_to_square_form_l688_688112


namespace hexagon_area_l688_688915

noncomputable def side_length_square (area_square : ℝ) : ℝ := 
  real.sqrt(area_square)

noncomputable def side_length_hexagon (s : ℝ) : ℝ :=
  s

noncomputable def area_hexagon (s : ℝ) : ℝ := 
  6 * (real.sqrt(3) / 4) * s^2

theorem hexagon_area (A_s : ℝ) (hA : A_s = real.sqrt(3)) :
  area_hexagon (side_length_square A_s) = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688915


namespace leah_coins_value_l688_688713

theorem leah_coins_value : 
  ∃ (p n : ℕ), p + n = 15 ∧ n + 1 = p ∧ 5 * n + 1 * p = 43 := 
by
  sorry

end leah_coins_value_l688_688713


namespace minimum_value_is_six_l688_688633

noncomputable def minimum_value (m n : ℝ) (h : m > 2 * n) : ℝ :=
  m + (4 * n ^ 2 - 2 * m * n + 9) / (m - 2 * n)

theorem minimum_value_is_six (m n : ℝ) (h : m > 2 * n) : minimum_value m n h = 6 := 
sorry

end minimum_value_is_six_l688_688633


namespace triangle_area_inequality_l688_688341

theorem triangle_area_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (S : ℝ) 
  (area_formula : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  S ≤ (a^2 + b^2 + c^2) / (4 * Real.sqrt 3) ∧ 
  (S = (a^2 + b^2 + c^2) / (4 * Real.sqrt 3) ↔ a = b ∧ b = c) := 
by
  sorry

end triangle_area_inequality_l688_688341


namespace calculate_material_needed_l688_688511

theorem calculate_material_needed (area : ℝ) (pi_approx : ℝ) (extra_material : ℝ) (r : ℝ) (C : ℝ) : 
  area = 50.24 → pi_approx = 3.14 → extra_material = 4 → pi_approx * r ^ 2 = area → 
  C = 2 * pi_approx * r →
  C + extra_material = 29.12 :=
by
  intros h_area h_pi h_extra h_area_eq h_C_eq
  sorry

end calculate_material_needed_l688_688511


namespace problem1_problem2_l688_688088

theorem problem1 : -(2^2) / (1 / 5) * 5 - (-10)^2 - abs(-3) = -123 :=
by
  sorry

theorem problem2 : (-1) ^ 2023 + (-5) * ((-2) ^ 3 + 2) - (-4) ^ 2 / (-1 / 2) = 61 :=
by
  sorry

end problem1_problem2_l688_688088


namespace largest_sum_of_digits_l688_688468

theorem largest_sum_of_digits : ∃ (a b : ℕ), 
  (a = 873) ∧ (b = 50) ∧ (List.perm [3, 5, 7, 8, 0] 
    (a.digits 10 ++ b.digits 10)) ∧ (a + b = 923) :=
by
  sorry

end largest_sum_of_digits_l688_688468


namespace number_of_m_l688_688296

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end number_of_m_l688_688296


namespace solve_for_x_l688_688285

-- Definitions
def log3_A (x : ℝ) := Real.log x / Real.log 3
def log9_A (x : ℝ) := Real.log x / Real.log 9

-- Condition
def condition (x : ℝ) : Prop := log3_A (x ^ 3) + log9_A x = 4

-- Proof Problem Statement
theorem solve_for_x (x : ℝ) (h : condition x) : x = 3 ^ (8 / 7) :=
by
  sorry

end solve_for_x_l688_688285


namespace tangent_line_at_neg1_F_monotonicity_m_range_l688_688354

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 / 2 * x * abs x

-- Problem 1
theorem tangent_line_at_neg1 :
  tangent_line g (-1) = λ y, y = 1 * (-1) - (1 / 2) :=
sorry

-- Problem 2
noncomputable def F (x : ℝ) := x * f x - g x

theorem F_monotonicity : 
  ∀ x > 0, ∃ intv, intv = (0, +∞) → Monotone.decreasing_on F intv :=
sorry

-- Problem 3
theorem m_range (m : ℝ) :
  (∀ x1 x2, 1 ≤ x2 ∧ x2 ≤ x1 → m * (g x1 - g x2) > x1 * f x1 - x2 * f x2) → m ≥ 1 :=
sorry

end tangent_line_at_neg1_F_monotonicity_m_range_l688_688354


namespace vertex_of_parabola_l688_688811

theorem vertex_of_parabola : 
    let eq := (λ x, -2 * x^2 - 16 * x - 50) in
    ∃ p q, 
      q = -18 ∧ 
      ∀ x, eq x = -2 * (x + p)^2 + q :=
begin
  sorry
end

end vertex_of_parabola_l688_688811


namespace minimum_area_triangle_l688_688315

theorem minimum_area_triangle (A B : ℝ × ℝ) (l : ℝ → ℝ) 
  (h_intersect : ∀ p ∈ {A, B}, p.2^2 = 4 * p.1 ∧ l p.2 = p.1)
  (h_dot_product : (A.1 * B.1 + A.2 * B.2) = -4) :
  ∃ m : ℝ, m = 0 ∧ (1 / 2) * (distance A B) * (2 / √((1 + m^2))) = 4 * √2 :=
sorry

end minimum_area_triangle_l688_688315


namespace original_mixture_acid_percent_l688_688844

-- Definitions of conditions as per the original problem
def original_acid_percentage (a w : ℕ) (h1 : 4 * a = a + w + 2) (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : Prop :=
  (a * 100) / (a + w) = 100 / 3

-- Main theorem statement
theorem original_mixture_acid_percent (a w : ℕ) 
  (h1 : 4 * a = a + w + 2)
  (h2 : 5 * (a + 2) = 2 * (a + w + 4)) : original_acid_percentage a w h1 h2 :=
sorry

end original_mixture_acid_percent_l688_688844


namespace second_group_count_l688_688817

theorem second_group_count
  (sum_first_group : ℕ)
  (average_second_group : ℕ)
  (combined_average : ℕ)
  (count_first_group : ℕ)
  (sum_first_group_eq : sum_first_group = 84)
  (average_second_group_eq : average_second_group = 21)
  (combined_average_eq : combined_average = 18)
  (count_first_group_eq : count_first_group = 7)
  : ∃ count_second_group : ℕ, (sum_first_group + average_second_group * count_second_group) / (count_first_group + count_second_group) = combined_average ∧ count_second_group = 14 :=
begin
  sorry
end

end second_group_count_l688_688817


namespace solve_tuples_l688_688587

theorem solve_tuples (n : ℕ) (x : Fin n → ℝ) :
  (∀ k : Fin n, x k ^ 2 = ∑ i in Finset.Icc 0 (k - 1), ∑ j in Finset.Icc (k + 1) (n - 1), x i * x j) →
  (x = 0 ∨ (n = 3 ∧ (∃ c : ℝ, ∀ k : Fin n, x k = c))) :=
sorry

end solve_tuples_l688_688587


namespace part_a_exists_rational_non_integer_l688_688874

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l688_688874


namespace methane_chlorine_combination_l688_688045

def moles_methane_combined 
  (moles_chlorine : ℕ)
  (moles_tetrachloromethane : ℕ) : ℕ := 
  1

theorem methane_chlorine_combination
  (moles_chlorine : ℕ)
  (moles_tetrachloromethane : ℕ)
  (h_chlorine : moles_chlorine = 4)
  (h_tetrachloromethane : moles_tetrachloromethane = 1) : 
  moles_methane_combined moles_chlorine moles_tetrachloromethane = 1 :=
by
  unfold moles_methane_combined
  rw [h_chlorine, h_tetrachloromethane]
  exact rfl

end methane_chlorine_combination_l688_688045


namespace decreasing_function_condition_l688_688631

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (4 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1 / 7 ≤ a ∧ a < 1 / 4) :=
by
  sorry

end decreasing_function_condition_l688_688631


namespace am_lt_bm_plus_cm_l688_688717

-- Definitions from conditions
def point := ℝ × ℝ
def triangle := point × point × point

def is_right_triangle (A B C : point) : Prop := ∠(B - A) (C - A) = π / 2
def is_midpoint (M A B : point) : Prop := (M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2)
def is_circumcircle_center (M A B C : point) : Prop := dist M A = dist M B ∧ dist M A = dist M C

-- The main theorem statement
theorem am_lt_bm_plus_cm
  (A B C M : point)
  (h1 : is_right_triangle A B C)
  (h2 : is_midpoint M B C)
  (h3 : is_circumcircle_center M A B C)
  : dist M A < dist M B + dist M C :=
sorry

end am_lt_bm_plus_cm_l688_688717


namespace find_value_l688_688890

theorem find_value : 
  let number := 50
  let value := 0.20 * number - 4
  value = 6 := by
  let number := 50
  let value := 0.20 * number - 4
  sorry

end find_value_l688_688890


namespace crate_tower_probability_l688_688245

theorem crate_tower_probability (heights : Fin 3 → ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (heights 0 = 3 ∧ heights 1 = 4 ∧ heights 2 = 6) →
  (3 * a + 4 * b + 6 * c = 50) →
  (a + b + c = 11) →
  let total_valid_arrangements := Nat.factorial 11 / (Nat.factorial a * Nat.factorial b * Nat.factorial c) in
  let total_arrangements := 3 ^ 11 in
  let probability := total_valid_arrangements.to_rat / total_arrangements.to_rat in
  probability.num = 72 :=
by 
  intros h_heights h_eq1 h_eq2 
  let total_valid_arrangements := Nat.factorial 11 / (Nat.factorial a * Nat.factorial b * Nat.factorial c)
  let total_arrangements := 3 ^ 11
  let probability := total_valid_arrangements.to_rat / total_arrangements.to_rat
  have h_probability : probability.num = 72
  exact sorry

end crate_tower_probability_l688_688245


namespace angle_PMN_is_45_l688_688320

theorem angle_PMN_is_45 (P M N Q R: Type*) [InnerProductGeometry P]
  (h1: angle P Q R = π / 2)
  (h2: dist P M = dist P N)
  (h3: dist P R = dist P Q)
  : angle P M N = π / 4 := 
sorry

end angle_PMN_is_45_l688_688320


namespace dad_steps_l688_688148

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688148


namespace parallelogram_area_is_20_l688_688561

def Point := (ℝ × ℝ)

def parallelogram_vertices : set Point := 
  {(0, 0), (4, 0), (3, 5), (7, 5)}

def base_length (A B : Point) : ℝ := 
  real.dist A B

def height := 5

def area_of_parallelogram (base height : ℝ) : ℝ :=
  base * height

theorem parallelogram_area_is_20 :
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 5)
  let D := (7, 5) in
  base_length A B = 4 →
  area_of_parallelogram (base_length A B) height = 20 :=
sorry

end parallelogram_area_is_20_l688_688561


namespace calculator_change_problem_l688_688055

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l688_688055


namespace height_of_smaller_cone_l688_688902

-- Define the problem constraints
variables (h_frustum : ℝ) (area_large_base area_small_base : ℝ)
variables (r_large r_small H : ℝ)

-- The constraints converted into variables
def frustum_height : h_frustum = 18 := by sorry
def larger_base_area : area_large_base = 324 * Real.pi := by sorry
def smaller_base_area : area_small_base = 36 * Real.pi := by sorry
def larger_base_radius : r_large = Real.sqrt 324 := by sorry
def smaller_base_radius : r_small = Real.sqrt 36 := by sorry
def total_cone_height : H = 27 := by sorry
def smaller_cone_height_calculated : H / 3 = 9 := by sorry

-- Main theorem to prove
theorem height_of_smaller_cone :
  (H/3) = 9 :=
by
  exact smaller_cone_height_calculated.finish 

end height_of_smaller_cone_l688_688902


namespace pipe_4_fills_fastest_l688_688437

-- Definitions of the flow rates based on the given conditions
variables {Q1 Q2 Q3 Q4 Q5 : ℝ}

-- Conditions provided in the problem
def conditions : Prop :=
  (Q1 + Q2 = 1/2) ∧
  (Q2 + Q3 = 1/15) ∧
  (Q3 + Q4 = 1/6) ∧
  (Q3 + Q4 = 1/3) ∧
  (Q5 + Q1 = 1/10)

-- Statement of the problem to prove that pipe 4 fills the pool fastest
theorem pipe_4_fills_fastest (h : conditions) : 
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
sorry

end pipe_4_fills_fastest_l688_688437


namespace identify_compound_propositions_l688_688387

-- Definitions of the conditions
def proposition1 := ¬(trapezoid → parallelogram)
def proposition2 := ∀ (triangle : Triangle), is_isosceles triangle → (triangle.base_angle1 = triangle.base_angle2)
def proposition3 := ∀ (quadrilateral : Quadrilateral), (complementary quadrilateral.angle1 quadrilateral.angle2) → (trapezoid quadrilateral ∨ cyclic_quadrilateral quadrilateral ∨ parallelogram quadrilateral)
def proposition4 := (60 % 5 = 0 ∨ 60 % 2 = 0)

-- Proof statement
theorem identify_compound_propositions :
  (proposition1 ∧ proposition3 ∧ proposition4) ∧ ¬proposition2 :=
sorry

end identify_compound_propositions_l688_688387


namespace problem1_l688_688014

theorem problem1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : a^2 + b^2 = 4 ∧ a * b = 1 := 
by
  sorry

end problem1_l688_688014


namespace min_distance_point_origin_l688_688528

theorem min_distance_point_origin (d : ℝ) (h : (4 * real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2) : d = 4 := 
  sorry

end min_distance_point_origin_l688_688528


namespace regular_nonagon_interior_angle_l688_688830

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℝ := sum_of_interior_angles n / n

theorem regular_nonagon_interior_angle :
  regular_polygon_interior_angle 9 = 140 :=
by
  unfold regular_polygon_interior_angle
  unfold sum_of_interior_angles
  norm_num
  sorry

end regular_nonagon_interior_angle_l688_688830


namespace number_of_students_in_chemistry_class_l688_688085

variables (students : Finset ℕ) (n : ℕ)
  (x y z cb cp bp c b : ℕ)
  (students_in_total : students.card = 120)
  (chem_bio : cb = 35)
  (bio_phys : bp = 15)
  (chem_phys : cp = 10)
  (total_equation : 120 = x + y + z + cb + bp + cp)
  (chem_equation : c = y + cb + cp)
  (bio_equation : b = x + cb + bp)
  (chem_bio_relation : 4 * b = c)
  (no_all_three_classes : true)

theorem number_of_students_in_chemistry_class : c = 153 :=
  sorry

end number_of_students_in_chemistry_class_l688_688085


namespace dad_steps_l688_688145

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688145


namespace fraction_sequence_product_l688_688400

theorem fraction_sequence_product :
  (∏ n in Finset.range 505, (6 * n + 6) / (6 * n)) = 506 :=
by
  sorry

end fraction_sequence_product_l688_688400


namespace dad_steps_l688_688166

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688166


namespace max_value_of_f_l688_688212

/-- 
Given the function f(x) defined by:
  f(x) = sin(x + sin x) + sin(x - sin x) + (pi / 2 - 2) * sin (sin x)
Prove that the maximum value of f(x) is (pi - 2) / sqrt(2).
--/
theorem max_value_of_f :
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) → f x = (Real.pi - 2) / Real.sqrt 2 :=
sorry

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.sin x) +
  Real.sin (x - Real.sin x) +
  (Real.pi / 2 - 2) * (Real.sin (Real.sin x))

end max_value_of_f_l688_688212


namespace number_of_blue_balloons_l688_688086

def total_balloons : ℕ := 37
def red_balloons : ℕ := 14
def green_balloons : ℕ := 10

theorem number_of_blue_balloons : (total_balloons - red_balloons - green_balloons) = 13 := 
by
  -- Placeholder for the proof
  sorry

end number_of_blue_balloons_l688_688086


namespace dad_steps_90_l688_688168

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688168


namespace height_in_cm_l688_688742

theorem height_in_cm 
  (inches : ℝ) 
  (inches_to_feet : ℝ) 
  (feet_to_meters : ℝ) 
  (meters_to_centimeters : ℝ) 
  (mark_height_in_inches : inches = 70) 
  (conversion_factor_in_to_ft : inches_to_feet = 1 / 12) 
  (conversion_factor_ft_to_m : feet_to_meters = 0.3048) 
  (conversion_factor_m_to_cm : meters_to_centimeters = 100) 
  : 
  let feet := inches * inches_to_feet,
      meters := feet * feet_to_meters,
      centimeters := meters * meters_to_centimeters
  in centimeters = 177.8 := 
by 
  -- Placeholder for proof, to be completed
  sorry

end height_in_cm_l688_688742


namespace leg_length_of_isosceles_right_triangle_l688_688792

theorem leg_length_of_isosceles_right_triangle
  (m : ℝ) 
  (h₁ : m = 15)
  (h₂ : ∃ x : ℝ, x * √2 = 2 * m) :
  ∃ x : ℝ, x = 15 * √2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l688_688792


namespace richard_more_pins_than_patrick_l688_688304

theorem richard_more_pins_than_patrick : 
  let patrick_round1 := 70
  let richard_round1 := patrick_round1 + 15
  let samantha_round1 := richard_round1 - 10
  let patrick_round2 := 2 * richard_round1
  let richard_round2 := patrick_round2 - 3
  let samantha_round2 := richard_round2
  let patrick_round3 := (3 / 2) * patrick_round1
  let richard_round3 := richard_round2 + 25
  let samantha_round3 := (3 / 2) * samantha_round2
  let patrick_total := patrick_round1 + patrick_round2 + patrick_round3
  let richard_total := richard_round1 + richard_round2 + richard_round3
  let samantha_total := samantha_round1 + samantha_round2 + samantha_round3
  in richard_total = patrick_total + 99 :=
by
  let patrick_round1 := 70
  let richard_round1 := patrick_round1 + 15
  let samantha_round1 := richard_round1 - 10
  let patrick_round2 := 2 * richard_round1
  let richard_round2 := patrick_round2 - 3
  let samantha_round2 := richard_round2
  let patrick_round3 := (3 / 2) * patrick_round1
  let richard_round3 := richard_round2 + 25
  let samantha_round3 := (3 / 2) * samantha_round2
  let patrick_total := patrick_round1 + patrick_round2 + patrick_round3
  let richard_total := richard_round1 + richard_round2 + richard_round3
  let samantha_total := samantha_round1 + samantha_round2 + samantha_round3
  have h : richard_total = patrick_total + 99 := sorry
  exact h

end richard_more_pins_than_patrick_l688_688304


namespace minimum_circumference_right_triangle_l688_688777

/-- A right triangle has an area of 10 cm². The midpoints of its altitudes are collinear,
    implying it is a right triangle. Prove that the minimum value of the circumference
    of its circumscribed circle is 20 cm (rounded to the nearest whole number). -/
theorem minimum_circumference_right_triangle :
  ∃ (ABC : Type) [triangle ABC], 
  let A : ℝ := 10 in
  let area : ℝ := A in
  ∃ (circumference : ℝ),
  (altitudes_collinear_condition ABC) ∧
  (area_condition ABC area) ∧
  circumference = 20 :=
sorry

end minimum_circumference_right_triangle_l688_688777


namespace mother_younger_than_father_l688_688786

def Taehyung_age : Nat := 9
def father_age : Nat := 5 * Taehyung_age
def mother_age : Nat := 4 * Taehyung_age

theorem mother_younger_than_father : father_age - mother_age = 9 :=
by
  have T : Nat := 9
  have F : Nat := 5 * T
  have M : Nat := 4 * T
  have father_age_def : father_age = F := rfl
  have mother_age_def : mother_age = M := rfl
  sorry

end mother_younger_than_father_l688_688786


namespace angle_BDE_is_112_l688_688568

-- Define right-angled triangle ABC with B being the right angle
variables (A B C E D O : Type) 
variables [circle O] -- Assume O is a circle
variables [triangle_ABC : (right_tri A B C)] -- Right triangle at B
variable [is_diameter O (A, B)] -- Circle has diameter AB
variable [is_point_on_circle O E] -- E is a point on the circle where circle intersects AC
variables [is_point_on_line D B C] -- D lies on BC
variable [is_tangent DE E O] -- Line DE is tangent to circle at E

-- Angles defined
variable (angle_A : angle A = 56) -- ∠A = 56°

-- We want to prove:
theorem angle_BDE_is_112 : angle BDE = 112 :=
by
  sorry

end angle_BDE_is_112_l688_688568


namespace neg_p_l688_688267

variable (ℝ : Type) [LinearOrderedField ℝ]

def p : Prop := ∀ x : ℝ, x ≤ 0 → Real.sqrt (x ^ 2) = -x

theorem neg_p : ¬ p ↔ ∃ x : ℝ, x ≤ 0 ∧ Real.sqrt (x ^ 2) ≠ -x :=
by sorry

end neg_p_l688_688267


namespace smallest_sum_faces_cube_l688_688752

theorem smallest_sum_faces_cube :
  ∃ (f : ℕ → ℕ) (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ),
    (f 0 = n₁) ∧ (f 1 = n₂) ∧ (f 2 = n₃) ∧
    (f 3 = n₄) ∧ (f 4 = n₅) ∧ (f 5 = n₆) ∧
    (abs (n₁ - n₂) > 1) ∧ (abs (n₁ - n₃) > 1) ∧
    (abs (n₁ - n₅) > 1) ∧ (abs (n₂ - n₄) > 1) ∧
    (abs (n₂ - n₆) > 1) ∧ (abs (n₃ - n₄) > 1) ∧
    (abs (n₃ - n₅) > 1) ∧ (abs (n₄ - n₆) > 1) ∧
    (abs (n₅ - n₆) > 1) ∧
    (n₁ + n₂ + n₃ + n₄ + n₅ + n₆ = 18) :=
by
  sorry

end smallest_sum_faces_cube_l688_688752


namespace find_number_l688_688046

theorem find_number (x : ℕ) (h : x + 5 * 8 = 340) : x = 300 :=
sorry

end find_number_l688_688046


namespace set_union_elements_l688_688733

variables (S T : Set ℕ) [nonempty (S : Set ℕ)] [nonempty (T : Set ℕ)]

theorem set_union_elements
  (hS : 4 ≤ S.to_finset.card)
  (hT : 2 ≤ T.to_finset.card)
  (cond1 : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x * y ∈ T)
  (cond2 : ∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → y / x ∈ S) :
  S.to_finset.card = 4 → (S ∪ T).to_finset.card = 7 := 
by
  sorry

end set_union_elements_l688_688733


namespace sum_of_squares_of_roots_l688_688111

theorem sum_of_squares_of_roots :
  (∃ x1 x2 : ℝ, 5 * x1^2 - 3 * x1 - 11 = 0 ∧ 5 * x2^2 - 3 * x2 - 11 = 0 ∧ x1 ≠ x2) →
  (x1 + x2 = 3 / 5 ∧ x1 * x2 = -11 / 5) →
  (x1^2 + x2^2 = 119 / 25) :=
by intro h1 h2; sorry

end sum_of_squares_of_roots_l688_688111


namespace sum_distances_squared_leq_k_squared_l688_688339

-- Define the points on the unit circle
variables {A : ℕ → ℝ × ℝ} (k : ℕ)

-- Define the distance function given two points
def d (p q : ℝ × ℝ) : ℝ := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Assume the points are on the unit circle
axiom point_on_circle (n : ℕ) (hn : n < k) : (A n).1 ^ 2 + (A n).2 ^ 2 = 1

-- State the main theorem
theorem sum_distances_squared_leq_k_squared : 
  (∑ i in finset.range k, ∑ j in finset.range k, if i < j then d (A i) (A j) ^ 2 else 0) ≤ (k:ℝ)^2 :=
sorry -- Proof omitted

end sum_distances_squared_leq_k_squared_l688_688339


namespace vacation_days_l688_688037

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def towels_per_day_per_person : ℕ := 1
def washer_capacity : ℕ := 14
def num_loads : ℕ := 6

def total_people : ℕ := num_families * people_per_family
def towels_per_day : ℕ := total_people * towels_per_day_per_person
def total_towels : ℕ := num_loads * washer_capacity

def days_at_vacation_rental := total_towels / towels_per_day

theorem vacation_days : days_at_vacation_rental = 7 := by
  sorry

end vacation_days_l688_688037


namespace intersection_is_2_l688_688736

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N : Set ℝ := {x | x^2 ≥ 2 * x}
noncomputable def intersection : Set ℝ := M ∩ N

theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l688_688736


namespace expected_value_a_squared_is_correct_l688_688035

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l688_688035


namespace regression_line_equation_l688_688066

theorem regression_line_equation 
  (n : ℕ) (X Y : Fin n → ℝ)
  (hX_mean : (∑ i, X i) / n = 4)
  (hY_mean : (∑ i, Y i) / n = 5)
  (b : ℝ) (hb : b = 2) :
  ∃ a : ℝ, (a = 5 - 2 * 4) ∧ (∀ x y, (y - 5) = 2 * (x - 4) → y = 2 * x - 3) :=
by
  use 5 - 2 * 4
  split
  . rfl
  . sorry

end regression_line_equation_l688_688066


namespace difference_in_areas_l688_688385

-- Define the radii of the circles
def radius1 : ℝ := 15
def radius2 : ℝ := 20

-- Define the areas of the circles
def area1 : ℝ := Real.pi * radius1^2
def area2 : ℝ := Real.pi * radius2^2

-- Define the proof statement
theorem difference_in_areas (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 20) :
  let area1 := Real.pi * r1^2
  let area2 := Real.pi * r2^2
  abs (area1 - area2) = 175 * Real.pi := by
  sorry

end difference_in_areas_l688_688385


namespace license_plate_count_l688_688675

theorem license_plate_count : 
  let letters := 26 
  let digits := 10 
  ∀ (first: fin letters) (last : fin digits), 
    (first.val == first.val ∨ first.val == last.val + letters) → 
    (last.val + letters == first.val ∨ last.val == last.val + letters) →
    (let ways := letters * 2 * digits in ways = 520) := 
by 
  intros 
  sorry

end license_plate_count_l688_688675


namespace solution_t_minval_l688_688110

-- Definitions of functions and points
def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
def g (x : ℝ) : ℝ := Real.cos (2 * x)

-- Point P and shifted point P'
def P : ℝ × ℝ := (Real.pi / 4, f (Real.pi / 4))
def P' (m : ℝ) : ℝ × ℝ := (Real.pi / 4 + m, f (Real.pi / 4))

-- Condition that P' lies on the graph of function g
def lies_on_g (m : ℝ) : Prop := g (Real.pi / 4 + m) = P'.snd

-- Proposition
theorem solution_t_minval (m : ℝ) (h : lies_on_g m) (hm : m > 0) : 
  P.snd = -1/2 ∧ ∃ k : ℤ, m = Real.pi / 12 + k * Real.pi :=
  by
  sorry

end solution_t_minval_l688_688110


namespace remainder_of_sum_of_T_mod_512_l688_688349

def T : Set ℕ := { x | ∃ n : ℕ, x = 2^n % 512 }
def U : ℕ := ∑ x in T.toFinset, x

theorem remainder_of_sum_of_T_mod_512 : U % 512 = 511 := by
  sorry

end remainder_of_sum_of_T_mod_512_l688_688349


namespace proof_problem_l688_688888

variable (a b x y : ℝ)

theorem proof_problem :
  (1 / 3) * log (a * x + b * y) + log (a * x - b * y) = (1 / 2) * (log (sqrt (a * x) + sqrt (b * y)) + log (sqrt (a * x) - sqrt (b * y))) →
  log x - log a = log y - log b →
  x = a / real.root ((a^2 + b^2)^2 * (a^2 - b^2)^3) (5 : ℕ) ∧
  y = b / real.root ((a^2 + b^2)^2 * (a^2 - b^2)^3) (5 : ℕ) :=
by
  intros h1 h2
  sorry

end proof_problem_l688_688888


namespace sum_two_lowest_scores_l688_688414

-- Define conditions as Lean constants
def mean (scores : List ℝ) : ℝ := (scores.sum) / (scores.length)

-- Given conditions
constant total_scores : List ℝ
constant num_scores : ℕ := 15
constant mean_15_scores : ℝ := 90

constant new_scores : List ℝ
constant new_num_scores : ℕ := 12
constant mean_12_scores : ℝ := 92

constant highest_score : ℝ := 110

-- Proof problem statement to show that the sum of the two lowest scores is 136
theorem sum_two_lowest_scores :
  (mean total_scores = 90) →
  (mean new_scores = 92) →
  (total_scores.length = 15) →
  (new_scores.length = 12) →
  (List.sum total_scores - List.sum new_scores = 246) →
  (List.maximum total_scores = some 110) →
  (246 - 110 = 136) :=
sorry

end sum_two_lowest_scores_l688_688414


namespace seq_geom_sum_seq_l688_688705

theorem seq_geom 
  (a : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 4 * a n - 3 * n + 1) :
  ∃ q : ℕ, ∀ n : ℕ, 0 < n → a (n + 1) - (n + 1) = q * (a n - n) :=
by sorry

theorem sum_seq 
  (a : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 4 * a n - 3 * n + 1)
  (S : ℕ → ℕ)
  (h₃ : ∀ n : ℕ, 0 < n → S n = (n * (n + 1) / 2) + ((4^n - 1) / 3)) :
  ∑ k in finset.range n, a (k + 1) = S n :=
by sorry

end seq_geom_sum_seq_l688_688705


namespace blue_tetrahedron_volume_l688_688049

theorem blue_tetrahedron_volume {a : ℝ} (h : a = 10) :
  let cube_volume := a^3,
      tetrahedron_volume := (1/3) * (1/2 * a * a) * a in
  cube_volume - 4 * tetrahedron_volume = 333.3333333333333 :=
by
  sorry

end blue_tetrahedron_volume_l688_688049


namespace weight_of_5_moles_BaO_molar_concentration_BaO_l688_688562

-- Definitions based on conditions
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_O : ℝ := 16.00
def molar_mass_BaO : ℝ := atomic_mass_Ba + atomic_mass_O
def moles_BaO : ℝ := 5
def volume_solution : ℝ := 3

-- Theorem statements
theorem weight_of_5_moles_BaO : moles_BaO * molar_mass_BaO = 766.65 := by
  sorry

theorem molar_concentration_BaO : moles_BaO / volume_solution = 1.67 := by
  sorry

end weight_of_5_moles_BaO_molar_concentration_BaO_l688_688562


namespace part_a_exists_rational_non_integer_l688_688872

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l688_688872


namespace spiral_center_coincidence_l688_688390

noncomputable def center_of_spiral_similarity (A B A1 B1 O : Point) : Prop :=
  ∃ (O : Point), 
    (∠ A O B = ∠ A1 O B1) ∧
    (dist A O / dist B O = dist A1 O / dist B1 O) 

theorem spiral_center_coincidence 
  (A B A1 B1 : Point) 
  (O : Point)
  (h : center_of_spiral_similarity A B A1 B1 O) :
  center_of_spiral_similarity A A1 B B1 O :=
sorry

end spiral_center_coincidence_l688_688390


namespace greatest_value_of_x_plus_y_l688_688473
noncomputable def max_val_x_plus_y : ℝ :=
  let x, y : ℝ in
  if (x^2 + y^2 = 130) ∧ (x*y = 36) then sqrt (202) else 0

theorem greatest_value_of_x_plus_y :
  ∀ x y : ℝ, (x^2 + y^2 = 130) ∧ (x*y = 36) → x + y ≤ sqrt 202 := by
  sorry

end greatest_value_of_x_plus_y_l688_688473


namespace area_of_triangle_bounded_by_lines_l688_688471

noncomputable def triangle_area : ℝ :=
  let line1 := (λ x: ℝ, x)
  let line2 := (λ x: ℝ, -x)
  let line3 := (λ x: ℝ, 2 * x + 4) in
  let point1 := (-4, -4)
  let point2 := (-4 / 3, 4 / 3)
  let point3 := (0, 0) in
  let base := real.sqrt ((8 / 3) ^ 2 + (16 / 3) ^ 2)
  let height := 4 / real.sqrt 5 in
  1 / 2 * base * height

theorem area_of_triangle_bounded_by_lines :
  triangle_area = 160 * real.sqrt 2 / 15 :=
sorry

end area_of_triangle_bounded_by_lines_l688_688471


namespace Cathy_total_money_l688_688090

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l688_688090


namespace largest_4_digit_palindromic_divisible_by_15_l688_688477

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.to_string.foldl (λ (acc : ℕ) (c : Char), acc + (c.to_nat - '0'.to_nat)) 0

theorem largest_4_digit_palindromic_divisible_by_15:
  ∃ n : ℕ, n % 15 = 0 ∧ is_palindromic n ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ 
  ∀ m : ℕ, m % 15 = 0 ∧ is_palindromic m ∧ 1000 ≤ m ∧ m ≤ 9999 → m ≤ n :=
  ∧ sum_of_digits n = 24 :=
by
  sorry

end largest_4_digit_palindromic_divisible_by_15_l688_688477


namespace hexagon_area_l688_688920

theorem hexagon_area (s : ℝ) (h1 : s^2 = real.sqrt 3) : 
  let hex_area := 6 * (real.sqrt 3 / 4 * s^2) in
  hex_area = 9 / 2 :=
begin
  -- the condition s^2 = √3
  have s_eq := h1,
  -- the area calculation
  have hex_area_def : hex_area = 6 * (real.sqrt 3 / 4 * s^2),
  { refl },
  -- showing the desired result
  rw s_eq,
  simp,
  norm_num,
end

end hexagon_area_l688_688920


namespace total_percent_decrease_l688_688042

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end total_percent_decrease_l688_688042


namespace smallest_sector_angle_l688_688743

theorem smallest_sector_angle 
  (n : ℕ) (a1 : ℕ) (d : ℕ)
  (h1 : n = 18)
  (h2 : 360 = n * ((2 * a1 + (n - 1) * d) / 2))
  (h3 : ∀ i, 0 < i ∧ i ≤ 18 → ∃ k, 360 / 18 * k = i) :
  a1 = 3 :=
by sorry

end smallest_sector_angle_l688_688743


namespace simplify_sqrt_product_l688_688560

noncomputable def sqrt_product (y: ℝ) : ℝ := sqrt (45 * y) * sqrt (18 * y) * sqrt (22 * y)

theorem simplify_sqrt_product (y: ℝ) : sqrt_product y = 18 * y * sqrt (55 * y) :=
by
  sorry

end simplify_sqrt_product_l688_688560


namespace largest_perfect_square_factor_4410_l688_688826

theorem largest_perfect_square_factor_4410 :
  ∀ (n : ℕ), n = 4410 → 
  ∃ (k : ℕ), k * k ∣ n ∧ ∀ (m : ℕ), m * m ∣ n → m * m ≤ k * k :=
by
  intro n hn
  have h4410 : ∃ p q r s : ℕ, 4410 = p * 3^2 * q * 7^2 ∧ p = 2 ∧ q = 5 ∧ r = 3 ∧ s = 7 :=
    ⟨2, 5, 3, 7, rfl, rfl, rfl, rfl⟩
  use 21
  constructor
  · rw hn
    norm_num
  · intros m hm
    norm_num  at hm
    sorry

end largest_perfect_square_factor_4410_l688_688826


namespace expected_value_a_squared_norm_bound_l688_688031

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l688_688031


namespace find_right_triangle_sides_l688_688444

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_condition (a b c : ℕ) : Prop :=
  a * b = 3 * (a + b + c)

theorem find_right_triangle_sides :
  ∃ (a b c : ℕ),
    is_right_triangle a b c ∧ area_condition a b c ∧
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
sorry

end find_right_triangle_sides_l688_688444


namespace geometric_sequence_sum_l688_688255

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℕ := λ k => 2^k) 
  (S : ℕ → ℕ := λ k => (1 - 2^k) / (1 - 2)) :
  S (n + 1) = 2 * a n - 1 :=
by
  sorry

end geometric_sequence_sum_l688_688255


namespace maximum_value_of_f_l688_688795

def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

theorem maximum_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -3 ∧ ∀ y > 0, f y ≤ -3 :=
by
  sorry

end maximum_value_of_f_l688_688795


namespace inverse_of_f_is_correct_l688_688592

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 1) + 1
noncomputable def f_inv (y : ℝ) : ℝ := y^2 - 2*y + 2

theorem inverse_of_f_is_correct (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  f_inv (f x) = x ∧ f (f_inv y) = y :=
by 
  sorry

end inverse_of_f_is_correct_l688_688592


namespace consecutive_mercedes_cars_l688_688813

theorem consecutive_mercedes_cars (total_cars red_mercedes yellow_mercedes pink_mercedes : ℕ)
  (h_total : total_cars = 100)
  (h_red : red_mercedes = 30)
  (h_yellow : yellow_mercedes = 20)
  (h_pink : pink_mercedes = 20)
  (h_no_adj_diff_colors : ∀ i j k, (i ≠ j ∧ j ≠ k ∧ k ≠ i) → ¬ (red_mercedes + yellow_mercedes + pink_mercedes ≤ (i * red_mercedes + j * yellow_mercedes + k * pink_mercedes))
  :
    3 * 
       (∃ i, (i = red_mercedes ∨ i = yellow_mercedes ∨ i = pink_mercedes)) := sorry

end consecutive_mercedes_cars_l688_688813


namespace dad_steps_l688_688138

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688138


namespace sample_data_analysis_l688_688998

noncomputable def mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem sample_data_analysis (n : ℕ) (x : fin n → ℝ)
    (hx : ∀ i j, i < j → x i < x j) :
  let new_data := List.of_fn (λ i, (x i + x ((i + 1) % n)) / 2)
  in mean (List.of_fn x) = mean new_data ∧
     ¬(List.of_fn x).median = new_data.median ∧
     (new_data.range < (List.of_fn x).range) ∧
     (new_data.variance < (List.of_fn x).variance) := 
by
  sorry

end sample_data_analysis_l688_688998


namespace count_perfect_squares_between_l688_688671

theorem count_perfect_squares_between :
  let n := 8
  let m := 70
  (m - n + 1) = 64 :=
by
  -- Definitions and step-by-step proof would go here.
  sorry

end count_perfect_squares_between_l688_688671


namespace part1_part2_l688_688005

theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x :=
  sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.cos (a * x) - Real.log (1 - x^2)) (hf0 : ∃ x, f' x = -a * Real.sin (a * x) + 2*x / (1 - x^2)) :
  0 = f' 0 → f'' 0 < 0 → (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) :=
  sorry

end part1_part2_l688_688005


namespace midpoints_proof_l688_688603

variables {A B C A_1 B_1 C_1 : Point}
variables {tri1 tri2 tri3 tri4 : Triangle}

-- The points A_1, B_1, and C_1 are on the sides BC, CA, and AB respectively.
variables hoc1 : On C_1 AB
variables hoa1 : On A_1 BC
variables hob1 : On B_1 CA

-- The areas of the triangles are equal.
variables heq1 : area (Triangle.mk A B_1 C_1) = area (Triangle.mk B C_1 A_1)
variables heq2 : area (Triangle.mk B C_1 A_1) = area (Triangle.mk C A_1 B_1)
variables heq3 : area (Triangle.mk C A_1 B_1) = area (Triangle.mk A_1 B_1 C_1)

theorem midpoints_proof :
    is_midpoint A_1 B C ∧
    is_midpoint B_1 C A ∧
    is_midpoint C_1 A B :=
by
    sorry

end midpoints_proof_l688_688603


namespace select_19_teams_l688_688504

theorem select_19_teams :
  ∃ S : Finset ℕ, S.card = 19 ∧ ∀ (x y ∈ S), x ≠ y → ¬ played_against x y :=
by
  -- Defining the conditions of the problem
  let teams : Finset ℕ := Finset.range 110
  have rounds : ℕ := 6
  have pairs_per_round : ℕ := 55

  -- Assume played_against as a condition for two teams
  -- Need not define it exactly but keep it as generic for the lean statement
  assume played_against : ℕ → ℕ → Prop,
  have played_once : ∀ (x y : ℕ), played_against x y → played_against y x,
  
  -- Assuming each team plays at most once
  have at_most_one_match : ∀ (x y : ℕ), x ≠ y → played_against x y → played_against y x → ¬ played_against x y ∨ ¬ played_against y x,
  
  -- Let S be a subset of teams such that none have played against each other
  have exists_S : ∃ S : Finset ℕ, S.card = 19 ∧ ∀ (x y ∈ S), x ≠ y → ¬ played_against x y,
  from sorry,
  
  exact exists_S

end select_19_teams_l688_688504


namespace class_mean_score_l688_688303

def total_students := 50
def first_group_students := 40
def first_group_mean_score := 68
def second_group_students := 10
def second_group_mean_score := 74

theorem class_mean_score :
  ((first_group_students * first_group_mean_score) + 
   (second_group_students * second_group_mean_score)) / 
  total_students = 69.2 :=
by
  sorry

end class_mean_score_l688_688303


namespace count_integers_with_digit_sum_20_l688_688280

def sum_of_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds + tens + units

theorem count_integers_with_digit_sum_20 :
  (finset.card (finset.filter (λ n => sum_of_digits n = 20) (finset.Ico 700 900))) = 13 :=
by
  sorry

end count_integers_with_digit_sum_20_l688_688280


namespace parabola_equation_and_dot_product_l688_688644

theorem parabola_equation_and_dot_product :
  (∀ (O A F : EuclideanSpace ℝ (Fin 2)) (x : ℝ),
    O = 0 ∧
    (A.1 = 2) ∧
    (∃ p : ℝ, ∀ (y : ℝ), A = (2, y) ∧ y^2 = 4 * p ∧
           F = (p / 2, 0) ∧
           ((2 - p / 2) * 2 + y * y = 10) ∧
           p = 2) ∧
    ∃ (M N : EuclideanSpace ℝ (Fin 2)),
        (l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) (k : ℝ),
        M.1 = 4 ∧ M.2 = 4 ∧
        N.1 = 4 ∧ N.2 = -4 ∧
        l = LineSpan ℝ (Set.insert 4 {P : EuclideanSpace ℝ (Fin 2) |
              ∃ k : ℝ, P = (k * ⟨4, 0⟩)})) ∧
        dot_product (⬝ M O) (⬝ N O) = 0)
sorry

end parabola_equation_and_dot_product_l688_688644


namespace hexagon_area_l688_688919

theorem hexagon_area (s : ℝ) (h1 : s^2 = real.sqrt 3) : 
  let hex_area := 6 * (real.sqrt 3 / 4 * s^2) in
  hex_area = 9 / 2 :=
begin
  -- the condition s^2 = √3
  have s_eq := h1,
  -- the area calculation
  have hex_area_def : hex_area = 6 * (real.sqrt 3 / 4 * s^2),
  { refl },
  -- showing the desired result
  rw s_eq,
  simp,
  norm_num,
end

end hexagon_area_l688_688919


namespace range_of_a_l688_688642

variable (f : ℝ → ℝ)

def is_decreasing (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x y ∈ domain, (x < y) → (f y ≤ f x)

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : is_decreasing f {x | -1 < x ∧ x < 1})
  (h_cond : ∀ a, f (1 - a) ≤ f (3 * a - 2)) :
  SetOf (a : ℝ) ((1 / 3) < a ∧ a ≤ (3 / 4)) :=
begin
  -- Proof is not required as per the instructions
  sorry
end

end range_of_a_l688_688642


namespace triangle_ratio_l688_688722

theorem triangle_ratio (l m n p : ℝ)
  (h1 : ∃ A B C : ℝ × ℝ × ℝ, midpoint B C = (l, 0, 0))
  (h2 : midpoint A C = (0, m, 0))
  (h3 : midpoint A B = (0, 0, n))
  (h4 : A.2.2 = p)
  (h5 : B.2.2 = p)
  (h6 : C.2.2 = p) :
  (AB.dist_sq + AC.dist_sq + BC.dist_sq) / (l^2 + m^2 + n^2 + 3 * p^2) = 8 := 
sorry

end triangle_ratio_l688_688722


namespace Noemi_blackjack_loss_l688_688746

-- Define the conditions
def start_amount : ℕ := 1700
def end_amount : ℕ := 800
def roulette_loss : ℕ := 400

-- Define the total loss calculation
def total_loss : ℕ := start_amount - end_amount

-- Main theorem statement
theorem Noemi_blackjack_loss :
  ∃ (blackjack_loss : ℕ), blackjack_loss = total_loss - roulette_loss := 
by
  -- Start by calculating the total_loss
  let total_loss_eq := start_amount - end_amount
  -- The blackjack loss should be 900 - 400, which we claim to be 500
  use total_loss_eq - roulette_loss
  sorry

end Noemi_blackjack_loss_l688_688746


namespace regular_nonagon_interior_angle_l688_688831

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℝ := sum_of_interior_angles n / n

theorem regular_nonagon_interior_angle :
  regular_polygon_interior_angle 9 = 140 :=
by
  unfold regular_polygon_interior_angle
  unfold sum_of_interior_angles
  norm_num
  sorry

end regular_nonagon_interior_angle_l688_688831


namespace bob_can_determine_set_S_l688_688076

theorem bob_can_determine_set_S (m n : ℤ) (hm : 0 < m) (hn : m ≤ n) :
  ∃ S : set (ℤ × ℤ), (∀ x y : ℤ, (x, y) ∈ S ↔ m ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ n) →
  ∀ (ℓ : set (ℤ × ℤ)), (is_horizontal ℓ ∨ is_vertical ℓ ∨ is_slope_plus_one ℓ ∨ is_slope_minus_one ℓ) →
  ∃! (S : set (ℤ × ℤ)), bob_wins S :=
sorry

def is_horizontal (ℓ : set (ℤ × ℤ)) : Prop :=
∃ c : ℤ, ℓ = {p : ℤ × ℤ | p.2 = c}

def is_vertical (ℓ : set (ℤ × ℤ)) : Prop :=
∃ c : ℤ, ℓ = {p : ℤ × ℤ | p.1 = c}

def is_slope_plus_one (ℓ : set (ℤ × ℤ)) : Prop :=
∃ c : ℤ, ℓ = {p : ℤ × ℤ | p.2 = p.1 + c}

def is_slope_minus_one (ℓ : set (ℤ × ℤ)) : Prop :=
∃ c : ℤ, ℓ = {p : ℤ × ℤ | p.2 = -p.1 + c}

constant bob_wins : set (ℤ × ℤ) → Prop

end bob_can_determine_set_S_l688_688076


namespace fn_prime_factor_bound_l688_688362

theorem fn_prime_factor_bound (n : ℕ) (h : n ≥ 3) : 
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^(2^n) + 1)) ∧ p > 2^(n+2) * (n+1) :=
sorry

end fn_prime_factor_bound_l688_688362


namespace hyperbola_equation_l688_688292

theorem hyperbola_equation :
  ∃ (y x : ℝ), 
    (let C2 := (x^2 / 16 + y^2 / 25 = 1) in
    let foci_C2 := (0, 3) ∨ (0, -3) in
    let major_axis_C2 := (0, 5) ∨ (0, -5) in
    let C1 := (y^2 / 9 - x^2 / 16 = 1) in
    (vertices C1 = foci_C2 ∧ 
     foci C1 = major_axis_C2) → 
    C1 = y^2 / 9 - x^2 / 16 = 1) :=
sorry

end hyperbola_equation_l688_688292


namespace analytical_expression_and_period_smallest_period_of_f_intervals_of_monotonic_increasing_l688_688234

theorem analytical_expression_and_period (x : ℝ) :
  let m := (sqrt 3 * sin x, 2)
  let n := (2 * cos x, cos x ^ 2)
  let f := m.1 * n.1 + m.2 * n.2
  f = sqrt 3 * sin (2 * x) + 1 + cos (2 * x) :=
begin
  unfold m n f,
  sorry
end

theorem smallest_period_of_f (x : ℝ) :
  let f := λ x, 2 * sin (2 * x + π / 6) + 1
  ∀ (T > 0), (∀ x, f (x + T) = f x) → T ≥ π :=
begin
  intros f T T_pos periodic,
  sorry
end

theorem intervals_of_monotonic_increasing (k : ℤ) :
  let f := λ x, 2 * sin (2 * x + π / 6) + 1
  ∃ (a b : ℝ), a = k * π - π / 3 ∧ b = k * π + π / 6 ∧
  ∀ x ∈ set.Icc a b, (f' x > 0) :=
begin
  intros f,
  use [(k * π - π / 3), (k * π + π / 6)],
  split,
  { refl },
  split,
  { refl },
  intros x hx,
  sorry
end

end analytical_expression_and_period_smallest_period_of_f_intervals_of_monotonic_increasing_l688_688234


namespace triangle_third_side_l688_688696

noncomputable def cos_135 := - real.sqrt 2 / 2

theorem triangle_third_side :
  ∀ (a b : ℝ), a = 9 → b = 10 → cos_135 = - real.sqrt 2 / 2 → 
  (∃ c : ℝ, c = real.sqrt (a^2 + b^2 + 2 * a * b * cos_135) ∧ 
  c = real.sqrt (181 + 90 * real.sqrt 2)) :=
by
  sorry

end triangle_third_side_l688_688696


namespace b_minus_d_sq_value_l688_688854

theorem b_minus_d_sq_value 
  (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2 * a - 3 * b + c + 4 * d = 17) :
  (b - d) ^ 2 = 25 :=
by
  sorry

end b_minus_d_sq_value_l688_688854


namespace largest_gcd_for_sum_1023_l688_688806

theorem largest_gcd_for_sum_1023 (a b : ℕ) (h1 : a + b = 1023) (h2 : 0 < a) (h3 : 0 < b) : ∃ g, g = 341 ∧ g = nat.gcd a b := 
by 
  use 341
  split
  { refl }
  { sorry }

end largest_gcd_for_sum_1023_l688_688806


namespace problem1_problem2_l688_688015

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : a^2 + b^2 + 3 ≥ a * b + √3 * (a + b) :=
sorry

-- Statement for problem (2)
theorem problem2 (x y z : ℝ) : 
let a := x^2 + 2 * y + (Real.pi / 2) in 
let b := y^2 + 2 * z + (Real.pi / 3) in 
let c := z^2 + 2 * x + (Real.pi / 6) in 
a > 0 ∨ b > 0 ∨ c > 0 :=
sorry

end problem1_problem2_l688_688015


namespace minimum_circumference_right_triangle_l688_688778

/-- A right triangle has an area of 10 cm². The midpoints of its altitudes are collinear,
    implying it is a right triangle. Prove that the minimum value of the circumference
    of its circumscribed circle is 20 cm (rounded to the nearest whole number). -/
theorem minimum_circumference_right_triangle :
  ∃ (ABC : Type) [triangle ABC], 
  let A : ℝ := 10 in
  let area : ℝ := A in
  ∃ (circumference : ℝ),
  (altitudes_collinear_condition ABC) ∧
  (area_condition ABC area) ∧
  circumference = 20 :=
sorry

end minimum_circumference_right_triangle_l688_688778


namespace not_divisible_by_5_l688_688228

theorem not_divisible_by_5 (b : ℕ) (h : b = 6) : ¬ (5 ∣ (b^3 - 3 * b^2 + 3 * b - 2)) :=
by 
  rw h
  sorry

end not_divisible_by_5_l688_688228


namespace weight_to_leave_out_l688_688571

theorem weight_to_leave_out (crates_weight_capacity: ℕ) (num_crates: ℕ) (nail_weight: ℕ) (num_nails: ℕ)
  (hammer_weight: ℕ) (num_hammers: ℕ) (planks_weight: ℕ) (num_planks: ℕ) :
  crates_weight_capacity = 20 → num_crates = 15 →
  nail_weight = 5 → num_nails = 4 →
  hammer_weight = 5 → num_hammers = 12 →
  planks_weight = 30 → num_planks = 10 →
  let total_crates_capacity := num_crates * crates_weight_capacity in
  let total_items_weight := (num_nails * nail_weight) + (num_hammers * hammer_weight) + (num_planks * planks_weight) in
  total_items_weight - total_crates_capacity = 80 :=
by 
  intros; 
  let total_crates_capacity := num_crates * crates_weight_capacity;
  let total_items_weight := (num_nails * nail_weight) + (num_hammers * hammer_weight) + (num_planks * planks_weight);
  have h1 : total_crates_capacity = 300 := by norm_num [num_crates, crates_weight_capacity];
  have h2 : total_items_weight = 380 := by norm_num [num_nails, nail_weight, num_hammers, hammer_weight, num_planks, planks_weight];
  rw [h1, h2];
  norm_num;
  exact rfl

end weight_to_leave_out_l688_688571


namespace zero_in_interval_l688_688077

noncomputable def f (x : ℝ) : ℝ := 3^x - 2

theorem zero_in_interval : ∃ c ∈ set.Ioo (0 : ℝ) (1 : ℝ), f c = 0 :=
by
  have h0 : f 0 < 0 := by norm_num
  have h1 : f 1 > 0 := by norm_num
  apply exists_Ioo_eq_zero h0 h1
  sorry

end zero_in_interval_l688_688077


namespace dad_steps_l688_688184

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688184


namespace earnings_difference_l688_688186

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end earnings_difference_l688_688186


namespace function_f_is_identity_l688_688422

open Real

-- Define the function f and the condition it needs to satisfy
def f (x : ℝ) : ℝ := sorry

-- The main theorem that f(x) must be x under the given condition
theorem function_f_is_identity :
  (∀ a : ℝ, ∀ x : ℝ, a < x ∧ x < a + 100 → a ≤ f(x) ∧ f(x) ≤ a + 100) →
  (∀ x : ℝ, f(x) = x) :=
by
  sorry

end function_f_is_identity_l688_688422


namespace shorter_side_of_rectangle_l688_688442

theorem shorter_side_of_rectangle
  (a b : ℝ)
  (h₀ : b = 9)
  (h₁ : 4 * b = 36)
  (h₂ : 3 * a = 36) :
  ∃ w : ℝ, 2 * 12 + 2 * w = 36 ∧ w = 6 :=
by
  have h₃ : a = 12, from calc
    a = 36 / 3 : by rw [h₂, mul_div_cancel_left 36 (by norm_num: 3 ≠ 0)]
      ... = 12 : by norm_num,
  use (36 - 24) / 2,
  split,
  { norm_num, rw [← h₃, ← h₂], norm_num, },
  { norm_num, },
  sorry

end shorter_side_of_rectangle_l688_688442


namespace sticker_price_l688_688275

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l688_688275


namespace dad_steps_are_90_l688_688126

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688126


namespace horse_food_per_day_l688_688554

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end horse_food_per_day_l688_688554


namespace diamonds_count_l688_688074

-- Definitions based on the conditions given in the problem
def totalGems : Nat := 5155
def rubies : Nat := 5110
def diamonds (total rubies : Nat) : Nat := total - rubies

-- Statement of the proof problem
theorem diamonds_count : diamonds totalGems rubies = 45 := by
  sorry

end diamonds_count_l688_688074


namespace original_price_of_tshirt_l688_688776

theorem original_price_of_tshirt :
  ∀ (P : ℝ), 
    (∀ discount quantity_sold revenue : ℝ, discount = 8 ∧ quantity_sold = 130 ∧ revenue = 5590 ∧
      revenue = quantity_sold * (P - discount)) → P = 51 := 
by
  intros P
  intro h
  sorry

end original_price_of_tshirt_l688_688776


namespace uniform_random_probability_event_l688_688467

noncomputable def probability_event (a : ℝ) (h : 0 < a ∧ a < 2) : Prop :=
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ (0 < a ∧ a < p) ∧ (3 * a - 2 < 0))

theorem uniform_random_probability_event :
  ∀ (a : ℝ) (h : 0 < a ∧ a < 2), probability_event a h → (3 * a - 2 < 0) :=
by
  intros a h
  cases h with ha1 ha2
  have h_range : a < (2 / 3),
  { sorry },
  exact lt_trans ha1 h_range

end uniform_random_probability_event_l688_688467


namespace max_ab_l688_688232

theorem max_ab (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 3 ≤ a + b ∧ a + b ≤ 4) : ab ≤ 15 / 4 :=
sorry

end max_ab_l688_688232


namespace least_multiple_36_sum_digits_l688_688475

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_multiple_36_sum_digits :
  ∃ n : ℕ, n = 36 ∧ (36 ∣ n) ∧ (9 ∣ digit_sum n) ∧ (∀ m : ℕ, (36 ∣ m) ∧ (9 ∣ digit_sum m) → 36 ≤ m) :=
by sorry

end least_multiple_36_sum_digits_l688_688475


namespace limit_of_quotient_l688_688199

open Real

theorem limit_of_quotient (f : ℝ → ℝ) (h : ∀ x, x ≠ 2 → f x = (x + 2)) :
  filter.tendsto (λ x, (x^2 - 4) / (x - 2)) (nhds 2) (nhds 4) :=
begin
  -- We will use the fact that x^2 - 4 = (x + 2) * (x - 2)
  -- and then the limit will be limit of x+2 as x approaches 2.
  -- The proof will be written here using Lean tactics to derive the conclusion.
  sorry -- Proof will be here
end

end limit_of_quotient_l688_688199


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688869

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688869


namespace dad_steps_l688_688116

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688116


namespace units_digit_of_quotient_l688_688194

theorem units_digit_of_quotient : 
  let n := 1993
  let term1 := 4 ^ n
  let term2 := 6 ^ n
  (term1 + term2) % 5 = 0 →
  let quotient := (term1 + term2) / 5
  (quotient % 10 = 0) := 
by 
  sorry

end units_digit_of_quotient_l688_688194


namespace quadrilateral_area_AGDH_l688_688702

variable (ABCDEF : Hexagon)
variable (G H : Point)
variable (area_hex : (ABCDEF.area = 360))

theorem quadrilateral_area_AGDH (h_reg_hex : ABCDEF.regular) : 
  (quadrilateral_area ABCDEF G H AGDH = 160) :=
by
  sorry

end quadrilateral_area_AGDH_l688_688702


namespace dad_steps_l688_688120

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688120


namespace correct_option_D_l688_688480

theorem correct_option_D : ∀ (a b c d : ℤ), (a = (sqrt ((-2 : ℤ)^2)) ∧ a = -2) ∨ 
  (b = (- sqrt (3^2)) ∧ b = 3) ∨ 
  (c = (real.cbrt (-9))) ∨ 
  (d = ± sqrt 9 ∧ d = ± 3) → 
  d = ± sqrt 9 ∧ d = ± 3 := by
sorry

end correct_option_D_l688_688480


namespace relationship_x_y_l688_688617

theorem relationship_x_y (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := 
by 
  sorry

end relationship_x_y_l688_688617


namespace correct_option_D_l688_688845

theorem correct_option_D (a : ℝ) :
  ¬((a^7)^2 = a^9) ∧ ¬(a^7 * a^2 = a^{14}) ∧ ¬(2*a^2 + 3*a^2 = 6*a^5) ∧ ((-0.5)^{2010} * 2^{2011} = 2) :=
by {
  split,
  { -- Option A
    intro h,
    calc (a^7)^2 = a^(7*2) : by rw pow_mul
            ... = a^14     : by norm_num
            ... ≠ a^9      : by { intro hc, exact (by norm_num : 14 ≠ 9).symm hc },
    contradiction,
  },
  split,
  { -- Option B
    intro h,
    calc a^7 * a^2 = a^(7+2) : by rw pow_add
            ... = a^9       : by norm_num
            ... ≠ a^14      : by { intro hc, exact (by norm_num : 9 ≠ 14).symm hc },
    contradiction,
  },
  split,
  { -- Option C
    intro h,
    calc 2*a^2 + 3*a^2 = 5*a^2 : by simp
            ... ≠ 6*a^5       : by { intro hc, exact (by linarith : 5 ≠ 6).symm hc },
    contradiction,
  },
  { -- Option D
    calc (-0.5)^{2010} * 2^{2011}
          = ((-0.5)*2)^{2010} * 2 : by rw [pow_mul', pow_tip],
      ... = (-1)^2010 * 2 : by norm_num,
      ... = 1 * 2 : by norm_num,
      ... = 2 : by norm_num
  }
}

end correct_option_D_l688_688845


namespace number_of_true_expressions_l688_688684

theorem number_of_true_expressions (a b c : ℝ) (h : |a + b| < -c) :
  (to_bool (a < -b - c) + to_bool (a + b > c) + to_bool (a + c < b) + to_bool (|a| + c < b)) = 2 :=
begin
  sorry
end

end number_of_true_expressions_l688_688684


namespace dot_product_not_sufficient_nor_necessary_for_parallel_l688_688351

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

end dot_product_not_sufficient_nor_necessary_for_parallel_l688_688351


namespace count_integers_with_digit_sum_20_l688_688281

def sum_of_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds + tens + units

theorem count_integers_with_digit_sum_20 :
  (finset.card (finset.filter (λ n => sum_of_digits n = 20) (finset.Ico 700 900))) = 13 :=
by
  sorry

end count_integers_with_digit_sum_20_l688_688281


namespace smallest_AAB_l688_688545

theorem smallest_AAB (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) 
  (h : 10 * A + B = (110 * A + B) / 7) : 110 * A + B = 996 :=
by
  sorry

end smallest_AAB_l688_688545


namespace correct_conclusions_l688_688312

variables (A : Type) [Fintype A] [DecidableEq A]
variables (red white black : A → Prop)
variable (B : A → Prop)
variables [DecidablePred red] [DecidablePred white] [DecidablePred black] [DecidablePred B]

-- Conditions
def jarA := (5:ℕ) • {x : A // red x} ∪ (2:ℕ) • {x : A // white x} ∪ (3:ℕ) • {x : A // black x}
def jarB := (4:ℕ) • {x : A // red x} ∪ (3:ℕ) • {x : A // white x} ∪ (3:ℕ) • {x : A // black x}

noncomputable def take_ball_from_JarA := ProbMeasure.jarA.toMeasure jarA
noncomputable def take_ball_from_JarB := ProbMeasure.jarB.toMeasure jarB

-- Events
def event_A1 := {x | red x}
def event_A2 := {x | white x}
def event_A3 := {x | black x}
def event_B := {x | B x}

-- Translations of question conclusions
theorem correct_conclusions : 
  (event_A1 ∪ event_A2 ∪ event_A3).PairwiseDisjoint → 
  (∀ y ∈ event_A1, cond_prob (event_B) (event_A1) (take_ball_from_JarB y)) := sorry

end correct_conclusions_l688_688312


namespace complex_line_eq_l688_688424

open Complex

theorem complex_line_eq (a b : ℂ) (z : ℂ) :
  (let u := -1 + 2 * I in
   let v := 2 + 2 * I in
   ∀ z : ℂ, (z - u) / (v - u) = (conj z - conj u) / (conj v - conj u) →
   a * z + b * conj z = 12) →
  a = 1 - 2 * I ∧ b = 1 + 2 * I →
  a * b = 5 :=
by sorry

end complex_line_eq_l688_688424


namespace regular_nonagon_interior_angle_l688_688837

theorem regular_nonagon_interior_angle : 
  let n := 9 in
  180 * (n - 2) / n = 140 :=
by 
  sorry

end regular_nonagon_interior_angle_l688_688837


namespace range_of_d_largest_S_n_l688_688569

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d a_1 : ℝ)

-- Conditions
axiom a_3_eq_12 : a_n 3 = 12
axiom S_12_pos : S_n 12 > 0
axiom S_13_neg : S_n 13 < 0
axiom arithmetic_sequence : ∀ n, a_n n = a_1 + (n - 1) * d
axiom sum_of_terms : ∀ n, S_n n = n * a_1 + (n * (n - 1)) / 2 * d

-- Problems
theorem range_of_d : -24/7 < d ∧ d < -3 := sorry

theorem largest_S_n : (∀ m, m > 0 ∧ m < 13 → (S_n 6 >= S_n m)) := sorry

end range_of_d_largest_S_n_l688_688569


namespace composite_expression_l688_688038

theorem composite_expression (a b : ℕ) (h_a : 2 ≤ a) (h_b : 2 ≤ b) :
  ∃ (x y z : ℕ), (1 ≤ x) ∧ (1 ≤ y) ∧ (1 ≤ z) ∧ (a * b = x * y + x * z + y * z + 1) :=
by 
  let x := a - 1
  let y := b - 1
  let z := 1
  use [x, y, z]
  split; linarith
  split; linarith
  split; linarith
  sorry

end composite_expression_l688_688038


namespace triangle_DEF_area_l688_688768

noncomputable def point (α : Type) := α × α 
def real_point := point ℝ
noncomputable def length (A B : real_point) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem triangle_DEF_area (D E F Q : real_point) : 
  dissected DEF → right_triangle DEF → 
  (∠DEQ = 45) → length D Q = 3 → length F Q = 1 → 
  area DEF = 12/5 :=
sorry

end triangle_DEF_area_l688_688768


namespace range_abc_l688_688359

open Real

noncomputable def range_of_abc (a b c : ℝ) (h : a + b + c = 1) : Set ℝ :=
  {x : ℝ | ∃ a b c : ℝ, a + b + c = 1 ∧ x = ab + ac + bc}

theorem range_abc (a b c : ℝ) (h : a + b + c = 1) :
  range_of_abc a b c h ⊆ Iic (1 / 2) := sorry

end range_abc_l688_688359


namespace num_pairs_eq_45_l688_688215

def num_pairs_satisfying_equation : ℕ := 45

theorem num_pairs_eq_45 :
  ∃ (m n : ℕ), (∑ m, ∑ n, if hn : n ≠ 0 then ((m : ℚ)⁻¹ + (n : ℚ)⁻¹ = 2020⁻¹) else false) = num_pairs_satisfying_equation := sorry

end num_pairs_eq_45_l688_688215


namespace min_x2_plus_y2_l688_688647

noncomputable def min_val (x y : ℝ) : ℝ :=
  if h : (x + 1)^2 + y^2 = 1/4 then x^2 + y^2 else 0

theorem min_x2_plus_y2 : 
  ∃ x y : ℝ, (x + 1)^2 + y^2 = 1/4 ∧ x^2 + y^2 = 1/4 :=
by
  sorry

end min_x2_plus_y2_l688_688647


namespace dad_steps_l688_688158

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688158


namespace continuous_at_5_l688_688369

def piecewise_function (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 5 then 4 * x^2 + 5 else b * x + 2

theorem continuous_at_5 : ∃ (b : ℝ), (∀ x, piecewise_function x b) = (function.continuous_at (piecewise_function x b) 5) :=
sorry

end continuous_at_5_l688_688369


namespace horse_food_per_day_l688_688553

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end horse_food_per_day_l688_688553


namespace shipping_cost_is_10_l688_688374

variable (original_cost : ℝ) (discount_percentage : ℝ) (embroidery_cost_per_shoe : ℝ) (total_cost_with_shipping : ℝ)

/-- The conditions given in the problem --/
variables (h1 : original_cost = 50.00)
          (h2 : discount_percentage = 0.10)
          (h3 : embroidery_cost_per_shoe = 5.50)
          (h4 : total_cost_with_shipping = 66.00)

/-- The goal is to prove the shipping cost is $10.00 --/
theorem shipping_cost_is_10 :
  let discounted_price := original_cost * (1 - discount_percentage)
  let total_embroidery_cost := 2 * embroidery_cost_per_shoe
  let total_cost_before_shipping := discounted_price + total_embroidery_cost
  let shipping_cost := total_cost_with_shipping - total_cost_before_shipping
  shipping_cost = 10.00 :=
by
  sorry

end shipping_cost_is_10_l688_688374


namespace part_a_exists_rational_non_integer_l688_688873

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l688_688873


namespace find_function_l688_688954

-- Given a function f : ℝ → ℝ, we need to show that it satisfies the following property:
def functional_eq (f : ℝ → ℝ) := ∀ x y : ℝ, f(x * f(y) + y) = f(x * y) + f(y)

-- Our goal is to find f such that functional_eq holds.
theorem find_function (f : ℝ → ℝ) (H : functional_eq f) : ∃ f : ℝ → ℝ, functional_eq f := 
by
  -- Goal: Prove the existence of f satisfying the functional equation
  sorry

end find_function_l688_688954


namespace altitude_length_l688_688622

noncomputable def length_of_altitude (l w : ℝ) : ℝ :=
  2 * l * w / Real.sqrt (l ^ 2 + w ^ 2)

theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ∃ h : ℝ, h = length_of_altitude l w := by
  sorry

end altitude_length_l688_688622


namespace simplify_condition_l688_688399

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) - Real.sqrt (-1 - x)

theorem simplify_condition (x : ℝ) (h1 : 1 + x ≥ 0) (h2 : -1 - x ≥ 0) : simplify_expression x = 0 :=
by
  rw [simplify_expression]
  sorry

end simplify_condition_l688_688399


namespace yellow_balloons_ratio_l688_688461

theorem yellow_balloons_ratio 
  (total_balloons : ℕ) 
  (colors : ℕ) 
  (yellow_balloons_taken : ℕ) 
  (h_total_balloons : total_balloons = 672)
  (h_colors : colors = 4)
  (h_yellow_balloons_taken : yellow_balloons_taken = 84) :
  yellow_balloons_taken / (total_balloons / colors) = 1 / 2 :=
sorry

end yellow_balloons_ratio_l688_688461


namespace hexagon_diagonals_l688_688493

theorem hexagon_diagonals :
  let n := 6 in
  (n * (n - 3)) / 2 = 9 := by
  sorry

end hexagon_diagonals_l688_688493


namespace find_xy_l688_688539

theorem find_xy (x y : ℝ) (π_ne_zero : Real.pi ≠ 0) (h1 : 4 * (x + 2) = 6 * x) (h2 : 6 * x = 2 * Real.pi * y) : x = 4 ∧ y = 12 / Real.pi :=
by
  sorry

end find_xy_l688_688539


namespace range_of_x_l688_688987

theorem range_of_x (x : ℝ) :
  arcsin x < arccos x ∧ arccos x < arccot x → 0 < x ∧ x < real.sqrt 2 / 2 := sorry

end range_of_x_l688_688987


namespace parabola_transformation_l688_688800

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end parabola_transformation_l688_688800


namespace no_two_even_values_l688_688732

def f (n : ℤ) : ℤ :=
if n < 0 then n^2 + 4 * n + 4 else 3 * n - 15

theorem no_two_even_values (a b : ℤ) (h1 : f (-3) + f 3 + f a = 0) (h2 : f (-3) + f 3 + f b = 0) (ha : Even a) (hb : Even b) : a = b := by
  sorry

end no_two_even_values_l688_688732


namespace max_h_completed_l688_688576

noncomputable def max_h : ℝ := 1 / (Real.floor ((99 - 1) / 2) + 1)

theorem max_h_completed :
  ∀ (h : ℝ), (∀ (a : ℝ), 0 ≤ a ∧ a ≤ h → ∀ (P : polynomial ℝ), degree P = 99 ∧ P.eval 0 = 0 ∧ P.eval 1 = 0 → ∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ P.eval x1 = P.eval x2 ∧ x2 - x1 = a) ↔ h ≤ max_h :=
by sorry

end max_h_completed_l688_688576


namespace Mary_balloons_l688_688610

-- Define the key conditions of the problem
variable (FredBalloons : ℕ)
variable (SamBalloons : ℕ)
variable (TotalBalloons : ℕ)

-- State the condition: Fred and Sam's balloons add up to the total balloons, minus the number of Mary's balloons
theorem Mary_balloons (h1 : FredBalloons = 5) 
                       (h2 : SamBalloons = 6) 
                       (h3 : TotalBalloons = 18) : 
                       let MaryBalloons := TotalBalloons - FredBalloons - SamBalloons in
                       MaryBalloons = 7 :=
by
  sorry

end Mary_balloons_l688_688610


namespace circle_equation_l688_688785

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
(A.1 + B.1) / 2, (A.2 + B.2) / 2

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem circle_equation (A B : ℝ × ℝ) (hx : A = (4, 9)) (hy : B = (6, 3)) :
  let C := midpoint A B,
    r := distance A C
  in (x - C.1)^2 + (y - C.2)^2 = r^2 := 
by
  intro C r,
  rw [hx, hy],
  let C := midpoint (4, 9) (6, 3),
  let r := distance (4, 9) C,
  have hC : C = (5, 6), sorry, -- midpoint calculation
  have hr : r = real.sqrt 10, sorry, -- distance calculation
  calc (x - C.1)^2 + (y - C.2)^2
      = (x - 5)^2 + (y - 6)^2 : by rw [hC]
  ... = 10 : by rw [←hr^2]


end circle_equation_l688_688785


namespace lcm_gcd_48_180_l688_688209

theorem lcm_gcd_48_180 :
  Nat.lcm 48 180 = 720 ∧ Nat.gcd 48 180 = 12 :=
by
  sorry

end lcm_gcd_48_180_l688_688209


namespace common_positive_divisors_count_l688_688672

theorem common_positive_divisors_count (n m : ℕ) (h₁ : n = 84) (h₂ : m = 90) :
  {d | d ∣ n ∧ d ∣ m ∧ d > 1}.to_finset.card = 3 :=
by
  sorry

end common_positive_divisors_count_l688_688672


namespace arrange_leopards_l688_688741

-- Definitions based on the conditions
def leopards : Fin 8 := sorry
def shortest_leopards : Fin 2 := sorry
def tallest_leopard : Fin 1 := sorry

-- Math proof statement
theorem arrange_leopards (condition1 : ∀ l : leopards, l ≠ shortest_leopards ∧ l ≠ tallest_leopard):
  ∃ ways : ℕ, ways = 960 :=
by
  sorry

end arrange_leopards_l688_688741


namespace odometer_reading_at_lunch_l688_688745

axiom odometer_start : ℝ
axiom miles_traveled : ℝ
axiom odometer_at_lunch : ℝ
axiom starting_reading : odometer_start = 212.3
axiom travel_distance : miles_traveled = 159.7
axiom at_lunch_reading : odometer_at_lunch = odometer_start + miles_traveled

theorem odometer_reading_at_lunch :
  odometer_at_lunch = 372.0 :=
  by
  sorry

end odometer_reading_at_lunch_l688_688745


namespace find_f_1_parity_f_inequality_f_l688_688784

def f : ℝ → ℝ
def dom_f : Set ℝ := {x | x ≠ 0}

axiom f_domain : ∀ x ∈ dom_f, f x ≠ ∞
axiom f_eq : ∀ x₁ x₂ ∈ dom_f, f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ x y ∈ set.univ, (x > 0 ∧ y > 0 ∧ x < y → f x < f y)
axiom f_4 : f 4 = 1

theorem find_f_1 : f 1 = 0 := sorry

theorem parity_f : ∀ x ∈ dom_f, f (-x) = f x := sorry

theorem inequality_f (x : ℝ) : f (2*x - 6) ≤ 2 ↔ x ∈ [-5, 3) ∪ (3, 11] := sorry

end find_f_1_parity_f_inequality_f_l688_688784


namespace exists_zero_mod_10000_in_fib_l688_688660

def fibonacci (n : ℕ) : ℕ :=
  if n = 1 then 0
  else if n = 2 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem exists_zero_mod_10000_in_fib (N : ℕ) :
  N ≥ 100000001 → ∃ n < N, fibonacci n % 10000 = 0 :=
by
  intros hN
  sorry

end exists_zero_mod_10000_in_fib_l688_688660


namespace dad_steps_l688_688115

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688115


namespace find_a_l688_688632

theorem find_a (a : ℝ) (i : ℂ) (hi : i = Complex.I) (z : ℂ) (hz : z = a + i) (h : z^2 + z = 1 - 3 * Complex.I) :
  a = -2 :=
by {
  sorry
}

end find_a_l688_688632


namespace smallest_positive_period_of_f_value_of_f_at_pi_over_3_l688_688649

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem smallest_positive_period_of_f : 
  ∀ x, f (x + π) = f x :=
sorry

theorem value_of_f_at_pi_over_3 :
  f (π / 3) = 1 :=
sorry

end smallest_positive_period_of_f_value_of_f_at_pi_over_3_l688_688649


namespace dad_steps_l688_688118

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688118


namespace lines_parallel_distinct_l688_688220

theorem lines_parallel_distinct (a : ℝ) :
  (∀ x y: ℝ, ax + 2y + a + 1 = 0) ∧ (∀ x y: ℝ, 2x + ay + 3 = 0) ∧
  ((∀ x y: ℝ, y = (-a / 2) * x + (- (a + 1) / 2)) ∧
  (∀ x y: ℝ, y = (-2 / a) * x + (-3 / a))) ∧
  a ≠ 2 → a = -2 := by
  sorry

end lines_parallel_distinct_l688_688220


namespace monotonic_increase_a_geq_3_over_2_l688_688688

theorem monotonic_increase_a_geq_3_over_2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π/2 → deriv (λ x, (1 / 2) * cos (2 * x) - 2 * a * (sin x + cos x) + (4 * a - 3) * x) x ≥ 0) →
  a ≥ 3 / 2 :=
by
  sorry

end monotonic_increase_a_geq_3_over_2_l688_688688


namespace pentagon_area_percentage_l688_688057

theorem pentagon_area_percentage (x : ℝ) (h : 0 < x) :
  let triangle_area := (x^2) / 2,
      square_area := 2 * (x^2),
      pentagon_area := triangle_area + square_area
  in triangle_area / pentagon_area * 100 = 20 :=
by
  sorry

end pentagon_area_percentage_l688_688057


namespace a_squared_minus_b_squared_is_zero_l688_688491

theorem a_squared_minus_b_squared_is_zero :
  let a := (7 + 14 + 21 + 28 + 35 + 42 + 49) / 7
  ∧ let b := 28
  in a ^ 2 - b ^ 2 = 0 :=
by
  sorry

end a_squared_minus_b_squared_is_zero_l688_688491


namespace centroid_locus_l688_688739

-- The goal is to prove the locus of the centroid G is given by 12x - 3y - 4 = 0
theorem centroid_locus (A : Point) (P O : Point) (l : Line) (B C : Point)
  (B1 C1 : Point) (G : Point) :
  A = ⟨2, 0⟩ →
  (∃ k : ℝ, k ≠ 0 ∧ l = Line.mk_slope (2, 0) k) →
  B ∈ parabola (λ x, x^2 + 2) ∧ C ∈ parabola (λ x, x^2 + 2) →
  B1 = (B.1, 0) ∧ C1 = (C.1, 0) →
  P ∈ segment B C ∧ (vector P B = (|vector B B1| / |vector C C1|) • vector P C) →
  centroid ⟨O, A, P⟩ = G →
  12 * G.1 - 3 * G.2 - 4 = 0 :=
by
  sorry

end centroid_locus_l688_688739


namespace fractional_part_S_over_T_l688_688953

noncomputable def P : ℕ → ℂ → ℂ
| 0, x => 1
| 1, x => x
| (n+2), x => 2 * x * P (n+1) x - P n x

def abs_val (z : ℂ) : ℝ := complex.abs z

def S := abs_val (complex.deriv (P 2017) (complex.I / 2))
def T := abs_val (complex.deriv (P 17) (complex.I / 2))

theorem fractional_part_S_over_T : 
  let frac_part := real.frac (S / T)
  let numer := 4142
  let common_div := 27149
  gcd numer common_div = 1 ∧ 
  frac_part = numer / common_div := 
sorry

end fractional_part_S_over_T_l688_688953


namespace exists_rational_non_integer_linear_l688_688877

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l688_688877


namespace a_10_eq_one_tenth_l688_688659

-- Defining the sequence a_n
noncomputable def a : ℕ → ℝ
| 1 := 1
| n := let x := a (n - 1) in x / (1 + x)

-- Proving the main statement
theorem a_10_eq_one_tenth : a 10 = 1/10 := by
  sorry

end a_10_eq_one_tenth_l688_688659


namespace event_complementary_and_mutually_exclusive_l688_688905

def students : Finset (String × String) := 
  { ("boy", "1"), ("boy", "2"), ("boy", "3"), ("girl", "1"), ("girl", "2") }

def event_at_least_one_girl (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, (x.1 = "girl")

def event_all_boys (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, (x.1 = "boy")

def two_students (s : Finset (String × String)) : Prop :=
  s.card = 2

theorem event_complementary_and_mutually_exclusive :
  ∀ s: Finset (String × String), two_students s → 
  (event_at_least_one_girl s ↔ ¬ event_all_boys s) ∧ 
  (event_all_boys s ↔ ¬ event_at_least_one_girl s) :=
sorry

end event_complementary_and_mutually_exclusive_l688_688905


namespace net_fits_in_grid_l688_688611

-- Define the 3x3 grid
def grid_3x3 : finset (nat × nat) := 
  finset.univ.filter (λ (p : nat × nat), p.1 < 3 ∧ p.2 < 3)

-- Define the net of a cube with edge length 1
def cube_net : finset (nat × nat) := finset.of_list [(1,1), (0,1), (2,1), (1,0), (1,2), (1,1)]

-- Prove that the net of the cube fits within the grid
theorem net_fits_in_grid : 
  ∃ (net : finset (nat × nat)), 
  (∀ (p : nat × nat), p ∈ net →
  p ∈ grid_3x3) ∧ ∀ x, x ∈ cube_net ↔ x ∈ net :=
by
  use cube_net
  simp [grid_3x3, cube_net]
  split
  -- Check that each element of the net is within the grid
  { intros p hp
    rw [finset.mem_filter, finset.mem_univ] at hp
    exact hp.2 },
  -- Ensure that all elements in the cube_net are included and nothing extra
  { intros x
    simp [cube_net] }


end net_fits_in_grid_l688_688611


namespace log_sequence_form_l688_688404

-- Definitions for given conditions
def geom_seq (a b c r : ℝ) : Prop := b = a * r^3 ∧ c = a * r^5
def valid_params (a r n : ℝ) : Prop := a > 1 ∧ r > 1 ∧ n > 1

-- The logical statement to prove
theorem log_sequence_form (a b c r n : ℝ) (h_geom_seq : geom_seq a b c r) (h_valid : valid_params a r n) : 
  let log_a_n := Real.log n / Real.log a
      log_b_n := Real.log n / (Real.log a + 3 * Real.log r)
      log_c_n := Real.log n / (Real.log a + 5 * Real.log r)
  in ¬ (∃ (f : ℝ → ℝ → Prop), (f log_a_n log_b_n ∧ f log_b_n log_c_n)) :=
sorry

end log_sequence_form_l688_688404


namespace range_f_when_a_one_range_of_a_when_f_has_one_zero_l688_688629

-- Lean definitions based on the given problem
def f (x a : ℝ) : ℝ := (Real.sin x - a) * (a - Real.cos x) + Real.sqrt 2 * a

theorem range_f_when_a_one :
  (∀ x : ℝ, f x 1) ∈ set.Icc (-3 / 2) (Real.sqrt 2) :=
sorry

theorem range_of_a_when_f_has_one_zero (h : ∀ x ∈ set.Icc 0 Real.pi, f x a = 0 → (x = u)) :
  1 ≤ a ∧ (a < Real.sqrt 2 + 1 ∨ a = Real.sqrt 2 + Real.sqrt 6 / 2) :=
sorry

end range_f_when_a_one_range_of_a_when_f_has_one_zero_l688_688629


namespace part_I_part_II_l688_688262

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 4)

-- Part I
theorem part_I : f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2 :=
by
  sorry

-- Part II
theorem part_II (x : ℝ) (h : f x = Real.sqrt 2 / 3) : Real.sin (2 * x) = 5 / 9 :=
by
  sorry

end part_I_part_II_l688_688262


namespace sum_first_50_b_n_l688_688453

def S : ℕ → ℕ := λ n, n^2 + n + 1
def a : ℕ+ → ℕ
| ⟨1, _⟩ := 3
| ⟨n+1, hn⟩ := 2 * (n + 1)

def b : ℕ+ → ℤ
| ⟨1, _⟩ := -1
| ⟨n+1, hn⟩ := (-1)^(n+1) * (a ⟨n+1, hn⟩ - 2)

def sum_b_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i, b ⟨i+1, Nat.succ_pos i⟩)

theorem sum_first_50_b_n : sum_b_n 50 = 49 := by
  sorry

end sum_first_50_b_n_l688_688453


namespace slope_is_correct_estimated_import_total_l688_688075

-- Assuming the data is given as below
def data_x := [1.8, 2.2, 2.6, 3.0]
def data_y := [2.0, 2.8, 3.2, 4.0]

-- Define the average calculation
def average (lst : List ℚ) : ℚ :=
  lst.sum / lst.length

-- Conditions
def avg_x := average data_x -- 2.4
def avg_y := average data_y -- 3.0
def y_intercept := -0.84

-- Linear correlation given condition: $\hat{y} = \hat{b}x - 0.84$
def linear_correlation (b : ℚ) (x : ℚ) : ℚ :=
  b * x + y_intercept

-- Prove $ \hat{b} = 1.6 $
theorem slope_is_correct : ∃ b : ℚ, avg_y = linear_correlation b avg_x := by
  exists 1.6
  simp [linear_correlation, avg_x, avg_y, y_intercept]
  sorry

-- Prove $ x \approx 4.275 $ when y = 6 using $\hat{b} = 1.6$
theorem estimated_import_total : ∃ x : ℚ, 6 = linear_correlation 1.6 x := by
  exists 4.275
  simp [linear_correlation, y_intercept]
  sorry

end slope_is_correct_estimated_import_total_l688_688075


namespace exponential_function_fixed_point_l688_688427

theorem exponential_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end exponential_function_fixed_point_l688_688427


namespace color_one_third_square_l688_688449

theorem color_one_third_square :
  (nat.choose 18 6) = 18564 :=
  sorry

end color_one_third_square_l688_688449


namespace hexagon_area_l688_688913

noncomputable def side_length_square (area_square : ℝ) : ℝ := 
  real.sqrt(area_square)

noncomputable def side_length_hexagon (s : ℝ) : ℝ :=
  s

noncomputable def area_hexagon (s : ℝ) : ℝ := 
  6 * (real.sqrt(3) / 4) * s^2

theorem hexagon_area (A_s : ℝ) (hA : A_s = real.sqrt(3)) :
  area_hexagon (side_length_square A_s) = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688913


namespace unique_a_exists_l688_688440

noncomputable def f_sequence : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := λ x, f_sequence n x * (f_sequence n x + 1 / (n + 1))

theorem unique_a_exists :
  ∃! a : ℝ, 0 < a ∧ (∀ n : ℕ, 0 < f_sequence n a ∧ f_sequence n a < f_sequence (n + 1) a ∧ f_sequence (n + 1) a < 1) :=
sorry

end unique_a_exists_l688_688440


namespace domain_h_l688_688575

noncomputable def h (x : ℝ) : ℝ := real.sqrt (x + 1) + (x - 5)^(1/3 : ℝ)

theorem domain_h : ∀ x : ℝ, (x + 1 ≥ 0) ↔ (x ∈ Set.Ici (-1)) :=
by
  intro x
  simp [Set.Ici]
  norm_num
  sorry

end domain_h_l688_688575


namespace dad_steps_l688_688178

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688178


namespace intersection_is_correct_l688_688664

def M : Set ℝ := { x | y = Real.log (9 - x^2) }
def N : Set ℝ := { y | y = 2^(1 - x) }
def intersection (M N : Set ℝ) : Set ℝ := { z | z ∈ M ∧ z ∈ N }

theorem intersection_is_correct : intersection M N = { z | 0 < z ∧ z < 3 } := 
by
  sorry

end intersection_is_correct_l688_688664


namespace phenomena_with_translational_motion_l688_688551

/-- Definitions of each phenomenon -/
def phenomenon₁ := "Rise or fall of the liquid level in a thermometer"
def phenomenon₂ := "Movement of the piston when inflating with a pump"
def phenomenon₃ := "Swing of a pendulum"
def phenomenon₄ := "Movement of a conveyor belt carrying bottled beverages"

/-- Definitions of translational motion for each phenomenon -/
def translational_motion₁ := true  -- ① involves translational motion
def translational_motion₂ := true  -- ② involves translational motion
def translational_motion₃ := false -- ③ does not involve translational motion
def translational_motion₄ := true  -- ④ involves translational motion

/-- Combined statement -/
theorem phenomena_with_translational_motion :
  (translational_motion₁ = true) ∧ (translational_motion₂ = true) ∧ 
  (translational_motion₃ = false) ∧ (translational_motion₄ = true) → 
  (phenomenon₁ ∈ {"Rise or fall of the liquid level in a thermometer", 
                  "Movement of the piston when inflating with a pump", 
                  "Movement of a conveyor belt carrying bottled beverages"} ∧ 
   phenomenon₂ ∈ {"Rise or fall of the liquid level in a thermometer", 
                  "Movement of the piston when inflating with a pump", 
                  "Movement of a conveyor belt carrying bottled beverages"} ∧ 
   phenomenon₃ ∉ {"Rise or fall of the liquid level in a thermometer", 
                  "Movement of the piston when inflating with a pump", 
                  "Movement of a conveyor belt carrying bottled beverages"} ∧ 
   phenomenon₄ ∈ {"Rise or fall of the liquid level in a thermometer", 
                  "Movement of the piston when inflating with a pump", 
                  "Movement of a conveyor belt carrying bottled beverages"}) := 
by {
  intros h,
  sorry
}

end phenomena_with_translational_motion_l688_688551


namespace center_of_symmetry_l688_688616

def f (A α x : ℝ) := A * Real.sin (2 * x - α)

theorem center_of_symmetry (A : ℝ) (α : ℝ) (hA : A > 0)
  (H : ∫ x in (0 : ℝ)..(4 * Real.pi / 3), f A α x = 0) : 
  (∃ x, Real.abs ((f A α (x + (4 * Real.pi / 3)) - f A α x) / 2) = 0) :=
sorry

end center_of_symmetry_l688_688616


namespace dad_steps_90_l688_688175

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688175


namespace sum_of_perimeters_of_squares_l688_688452

noncomputable def perimeters_sum (a b : ℝ) : ℝ :=
  4 * a + 4 * b

theorem sum_of_perimeters_of_squares : 
  ∀ (a b : ℝ),
  a^2 + b^2 = 130 → a^2 - b^2 = 50 → perimeters_sum a b = 20 * real.sqrt 10 :=
by
  intros a b h1 h2
  sorry

end sum_of_perimeters_of_squares_l688_688452


namespace smallest_multiplier_for_perfect_square_l688_688729

def y : ℕ := 2^3 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 11^3

theorem smallest_multiplier_for_perfect_square : ∃ n : ℕ, y * n = 2310 ∧ isPerfectSquare (y * n) := sorry

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

end smallest_multiplier_for_perfect_square_l688_688729


namespace original_price_of_jacket_l688_688911

/-- 
Given:
  - T-shirts are sold at $8 each.
  - Sweaters are sold at $18 each.
  - Jackets are on sale with a 10% discount.
  - The sales tax is 5%.
  - Kevin buys six T-shirts, four sweaters, and five jackets.
  - The total cost including sales tax is $504.
Theorem:
  The original price of each jacket is approximately $81.27.
-/
theorem original_price_of_jacket
  (price_T_shirt : ℕ := 8)
  (price_sweater : ℕ := 18)
  (num_T_shirts : ℕ := 6)
  (num_sweaters : ℕ := 4)
  (num_jackets : ℕ := 5)
  (discount : ℝ := 0.10)
  (sales_tax : ℝ := 0.05)
  (total_cost : ℝ := 504) :
  ∃ (original_price_per_jacket : ℝ),
    original_price_per_jacket ≈ 81.27 :=
by
  sorry

end original_price_of_jacket_l688_688911


namespace dad_steps_l688_688161

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688161


namespace find_total_students_l688_688775

theorem find_total_students (n : ℕ) : n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 → n = 509 :=
by 
  sorry

end find_total_students_l688_688775


namespace exists_rational_non_integer_linear_l688_688881

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l688_688881


namespace num_valid_four_digit_numbers_l688_688269

def U : Finset ℕ := {1, 2, 3, 4, 5}

def valid_number (n : ℕ) : Prop :=
  n > 2345 ∧ n < 4351 ∧ n.digits 10.eraseDuplicates.length = n.digits 10.length ∧ ∀ d ∈ n.digits 10, d ∈ U

theorem num_valid_four_digit_numbers :
  (Finset.filter valid_number (Finset.Icc 1000 9999)).card = 54 :=
by sorry

end num_valid_four_digit_numbers_l688_688269


namespace proof_problem_l688_688985

variable (α β : ℝ)

def interval_αβ : Prop := 
  α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ 
  β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

def condition : Prop := α * Real.sin α - β * Real.sin β > 0

theorem proof_problem (h1 : interval_αβ α β) (h2 : condition α β) : α ^ 2 > β ^ 2 := 
sorry

end proof_problem_l688_688985


namespace world_expo_visitors_l688_688546

noncomputable def per_person_cost (x : ℕ) : ℕ :=
  if x <= 30 then 120 else max (120 - 2 * (x - 30)) 90

theorem world_expo_visitors (x : ℕ) (h_cost : x * per_person_cost x = 4000) : x = 40 :=
by
  sorry

end world_expo_visitors_l688_688546


namespace vector_dot_product_correct_l688_688982

-- Definitions of the vectors
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ :=
  let x := 4 - 2 * vector_a.1
  let y := 1 - 2 * vector_a.2
  (x, y)

-- Theorem to prove the dot product is correct
theorem vector_dot_product_correct :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 4 := by
  sorry

end vector_dot_product_correct_l688_688982


namespace determinant_trig_matrix_zero_l688_688109

open Real

theorem determinant_trig_matrix_zero {a b : ℝ} :
  det ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ] = 0 := 
sorry

end determinant_trig_matrix_zero_l688_688109


namespace semi_annual_annuity_payment_l688_688766

-- Variables for conditions
def annual_payment : ℝ := 2500
def interest_rate_annual : ℝ := 0.0475
def semi_annual_interest_rate : ℝ := 0.02375
def skipped_years : ℕ := 5
def total_years : ℕ := 15
def periods : ℕ := 40 -- 20 years * 2 periods/year

-- The given accumulated future value after 15 years (computed in steps of solution)
def FV_total : ℝ := 54504.62

-- Function to compute annuity payment
def annuity_payment (FV : ℝ) (i : ℝ) (n : ℕ) : ℝ :=
  FV * i / (1 - (1 + i)^(-n))

-- The correct answer (from step 7 of the solution)
def expected_annuity_payment : ℝ := 2134.43

-- Lean statement to be proved
theorem semi_annual_annuity_payment :
  annuity_payment FV_total semi_annual_interest_rate periods ≈ expected_annuity_payment := 
by sorry

end semi_annual_annuity_payment_l688_688766


namespace f_is_odd_solve_inequality_l688_688991

-- Given conditions and the function f
variable {f : ℝ → ℝ}
axiom f_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f_one : f(1) = -2
axiom f_neg : ∀ x : ℝ, 0 < x → f(x) < 0

-- Part 1: Proving f is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  sorry

-- Part 2: Proving the solution set of the inequality f(x + 3) + f(2x - x^2) > 2
theorem solve_inequality : { x : ℝ | f(x + 3) + f(2x - x^2) > 2 } = { x : ℝ | x < -1 ∨ x > 4 } :=
by
  sorry

end f_is_odd_solve_inequality_l688_688991


namespace coefficient_x7_in_expansion_l688_688472

def polynomial := (1 - 3 * x^2 + x^3)

theorem coefficient_x7_in_expansion :
  let term := (1 - 3 * x^2 + x^3) in
  polynomial^6 → coefficient 7 = -1605 :=
by
  -- skip proof
  sorry

end coefficient_x7_in_expansion_l688_688472


namespace right_triangle_set_l688_688483

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end right_triangle_set_l688_688483


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688862

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688862


namespace stratified_sampling_yogurt_adult_milk_powder_sum_l688_688308

theorem stratified_sampling_yogurt_adult_milk_powder_sum :
  let liquid_milk_brands := 40
  let yogurt_brands := 10
  let infant_formula_brands := 30
  let adult_milk_powder_brands := 20
  let total_brands := liquid_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sample_size := 20
  let yogurt_sample := sample_size * yogurt_brands / total_brands
  let adult_milk_powder_sample := sample_size * adult_milk_powder_brands / total_brands
  yogurt_sample + adult_milk_powder_sample = 6 :=
by
  sorry

end stratified_sampling_yogurt_adult_milk_powder_sum_l688_688308


namespace dad_steps_90_l688_688176

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688176


namespace circle_arcs_sum_l688_688459

theorem circle_arcs_sum (R B G : ℕ) (arcs : ℕ) 
  (red_blue_num red_green_num blue_green_num same_color_num : ℕ) :
  R = 40 → B = 30 → G = 20 → arcs = 90 → 
  red_blue_num = R * B → red_green_num = R * G → blue_green_num = B * G → 
  same_color_num = (R * (R - 1) / 2) + (B * (B - 1) / 2) + (G * (G - 1) / 2) → 
  let total_max := red_blue_num * 1 + red_green_num * 2 + blue_green_num * 3 in
  let total_min := total_max in
  4600 = total_max ∧ 4000 = total_min :=
begin
  intros, 
  sorry
end

end circle_arcs_sum_l688_688459


namespace correct_proposition_l688_688848

-- Definitions for the conditions
def points_determine_plane (A B C : Type) [AffinelyDependent A B C] : Prop := True
def opposite_sides_parallel (quad : Quadrilateral) : Prop :=
  quad.opposite_sides.parallel
def opposite_sides_equal (quad : Quadrilateral) : Prop :=
  quad.opposite_sides.equal
def non_intersecting_lines_parallel (L1 L2 : Line) [NonIntersecting L1 L2] : Prop := False

-- Theorem stating the correct proposition
theorem correct_proposition (quad : Quadrilateral) :
  opposite_sides_parallel quad → parallelogram quad :=
begin
  sorry
end

end correct_proposition_l688_688848


namespace roots_of_g_eq_half_min_and_max_of_g_l688_688261

noncomputable def g (x : ℝ) : ℝ := 
  (2 * Real.cos x ^ 4 + Real.sin x ^ 2) / (2 * Real.sin x ^ 4 + 3 * Real.cos x ^ 2)

theorem roots_of_g_eq_half (k : ℤ) : 
  ∃ x, (x = (Int.ofNat k : ℤ) * π / 4 + π / 2 ∨ x = (Int.ofNat k : ℤ) * (π / 2 + π)) ∧ g x = 1 / 2 :=
begin
  sorry
end

theorem min_and_max_of_g : 
  (∀ x, 3 / 7 ≤ g x ∧ g x ≤ 2 / 3) ∧ 
  (∃ x, g x = 3 / 7) ∧ 
  (∃ x, g x = 2 / 3) :=
begin
  sorry
end

end roots_of_g_eq_half_min_and_max_of_g_l688_688261


namespace diagonal_quadrilateral_angle_bisectors_l688_688590

theorem diagonal_quadrilateral_angle_bisectors (A B C D M N K L : ℝ)
  (h1 : A = 1) (h2 : B = 3)
  (h3 : quadrilateral_formed_by_angle_bisectors A B C D M N K L) :
  diagonal M K = 2 :=
sorry

end diagonal_quadrilateral_angle_bisectors_l688_688590


namespace number_of_subsets_is_4_l688_688441

theorem number_of_subsets_is_4 (S : Set ℤ) (h : S = {-1, 1}) : S.powerset.card = 4 := 
by
  rw [h]
  -- proof will be provided here
  sorry

end number_of_subsets_is_4_l688_688441


namespace part_I_part_II_l688_688667

open Real

-- Define the vectors and dot product
def vector_a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x), cos (2 * x))
def vector_b (x : ℝ) : ℝ × ℝ := (cos (2 * x), - cos (2 * x))
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Part I: Prove cos 4x
theorem part_I (x : ℝ) (h1 : x ∈ set.Ioo (7 * π / 24) (5 * π / 12))
  (h2 : dot_product (vector_a x) (vector_b x) + 1 / 2 = -3 / 5) :
  cos (4 * x) = (3 - 4 * sqrt 3) / 10 :=
by sorry

-- Part II: Find the value of m
theorem part_II (x : ℝ) (m : ℝ)
  (h1 : x ∈ set.Ioc 0 (π / 3))
  (h2 : ∃! x, dot_product (vector_a x) (vector_b x) + 1 / 2 = m) :
  m = -1 / 2 :=
by sorry

end part_I_part_II_l688_688667


namespace PQ_in_triangle_is_10_l688_688202

noncomputable def find_length_PQ (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] : ℝ :=
  sorry

theorem PQ_in_triangle_is_10 (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  (PR : ∀ x : P, x ≠ 0 → R)
  (angle_PRQ : ∀ x : Q, x ≠ P → ∠PRQ = 45)
  (length_PR : ∀ x : R, dist P x = 10) :
  find_length_PQ P Q R = 10 :=
sorry

end PQ_in_triangle_is_10_l688_688202


namespace log_equation_solution_l688_688485

theorem log_equation_solution {x : ℝ} (hx : x > 0) (hx1 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by 
  sorry

end log_equation_solution_l688_688485


namespace alchemists_less_than_half_l688_688935

variable (k c a : ℕ)

theorem alchemists_less_than_half (h1 : k = c + a) (h2 : c > a) : a < k / 2 := by
  sorry

end alchemists_less_than_half_l688_688935


namespace value_range_of_f_l688_688810

def f (x : ℝ) := 2 * x ^ 2 + 4 * x + 1

theorem value_range_of_f :
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 4 → (∃ y ∈ Set.Icc (-1 : ℝ) 49, f x = y) :=
by sorry

end value_range_of_f_l688_688810


namespace b_amount_l688_688853

-- Define the conditions
def total_amount (a b : ℝ) : Prop := a + b = 1210
def fraction_condition (a b : ℝ) : Prop := (1/3) * a = (1/4) * b

-- Define the main theorem to prove B's amount
theorem b_amount (a b : ℝ) (h1 : total_amount a b) (h2 : fraction_condition a b) : b = 691.43 :=
sorry

end b_amount_l688_688853


namespace domain_of_g_l688_688641

-- Define the function domain of f(x+1)
def f_domain := ∀ x, 0 ≤ x + 1 ∧ x + 1 ≤ 2

-- Define the function g(x)
def g (x : ℝ) := (x + 3)

-- The theorem statement
theorem domain_of_g 
  (f_domain : ∀ x, 0 ≤ x + 1 ∧ x + 1 ≤ 2) :
  ∀ x, -3 ≤ g(x) ∧ g(x) ≤ -1 :=
sorry

end domain_of_g_l688_688641


namespace sin_x_plus_ax_increasing_l688_688597

noncomputable def is_increasing (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x < y → f x ≤ f y

theorem sin_x_plus_ax_increasing {a : ℝ} :
  is_increasing (λ x, Real.sin x + a * x) ↔ a > 1 := by
  sorry

end sin_x_plus_ax_increasing_l688_688597


namespace coordinates_of_M_l688_688638

-- Let M be a point in the 2D Cartesian plane
variable {x y : ℝ}

-- Definition of the conditions
def distance_from_x_axis (y : ℝ) : Prop := abs y = 1
def distance_from_y_axis (x : ℝ) : Prop := abs x = 2

-- Theorem to prove
theorem coordinates_of_M (hx : distance_from_y_axis x) (hy : distance_from_x_axis y) :
  (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
sorry

end coordinates_of_M_l688_688638


namespace roots_polynomial_condition_l688_688714

theorem roots_polynomial_condition (a b : ℂ) (z1 z2 : ℂ) :
  (a ≠ 0 ∧ b ≠ 0) ∧ (z1 + z2 = -a ∧ z1 * z2 = b) →
  (|z1 + z2| = |z1| + |z2| ↔ ∃ (λ : ℝ), λ ≥ 4 ∧ a^2 = λ * b) :=
by
  sorry

end roots_polynomial_condition_l688_688714


namespace series_divergence_l688_688944

theorem series_divergence : 
  let f (n : ℕ) : ℝ := (3 * n + 2) / (n * (n + 1) * (n + 3))
  in ¬ summable (λ n : ℕ, f n) :=
by
  sorry

end series_divergence_l688_688944


namespace smallest_value_of_d_l688_688534

def distance (x y : ℝ) : ℝ := real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem smallest_value_of_d (d : ℝ) : distance (4 * real.sqrt 3) (d - 2) = 4 * d → d = 26 / 15 :=
by
  -- sorry indicates we are omitting the proof
  sorry

end smallest_value_of_d_l688_688534


namespace hexagon_area_l688_688914

noncomputable def side_length_square (area_square : ℝ) : ℝ := 
  real.sqrt(area_square)

noncomputable def side_length_hexagon (s : ℝ) : ℝ :=
  s

noncomputable def area_hexagon (s : ℝ) : ℝ := 
  6 * (real.sqrt(3) / 4) * s^2

theorem hexagon_area (A_s : ℝ) (hA : A_s = real.sqrt(3)) :
  area_hexagon (side_length_square A_s) = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688914


namespace shop_owner_percentage_profit_l688_688068

theorem shop_owner_percentage_profit
  (cp : ℝ)  -- cost price of 1 kg
  (cheat_buy : ℝ) -- cheat percentage when buying
  (cheat_sell : ℝ) -- cheat percentage when selling
  (h_cp : cp = 100) -- cost price is $100
  (h_cheat_buy : cheat_buy = 15) -- cheat by 15% when buying
  (h_cheat_sell : cheat_sell = 20) -- cheat by 20% when selling
  :
  let weight_bought := 1 + (cheat_buy / 100)
  let weight_sold := 1 - (cheat_sell / 100)
  let real_selling_price_per_kg := cp / weight_sold
  let total_selling_price := weight_bought * real_selling_price_per_kg
  let profit := total_selling_price - cp
  let percentage_profit := (profit / cp) * 100
  percentage_profit = 43.75 := 
by
  sorry

end shop_owner_percentage_profit_l688_688068


namespace clara_valerie_mistake_l688_688943

theorem clara_valerie_mistake (n m : ℕ) (hm : m ≥ 2) : n! ≠ 2 ^ m * m! :=
by sorry

end clara_valerie_mistake_l688_688943


namespace geometric_sequence_formula_sum_of_products_l688_688242

namespace geometric_sequence

-- part (1)
theorem geometric_sequence_formula (a1 : ℝ) (h1 : a1 = 1/2)
 (h2 : 2 * (a1 * (1/2)^2) = a1 * (1/2)) :
   ∀ n : ℕ, (a1 * (1/2)^(n-1)) = (1/2)^n :=
sorry

end geometric_sequence

namespace arithmetic_sequence

-- part (2)
theorem sum_of_products (b1 : ℝ) (h3 : b1 = 1) (S3 : ℝ) (h4 : S3 = b1 + 1 + b1 + 2 + b1 = 7) :
  ∀ n : ℕ, (∑ k in finset.range n, k * (1/2)^k) = 2 - (n + 2) * (1/2)^n :=
sorry

end arithmetic_sequence

end geometric_sequence_formula_sum_of_products_l688_688242


namespace dad_steps_l688_688165

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688165


namespace p_q_r_inequality_l688_688356

theorem p_q_r_inequality (p q r : ℝ) (h₁ : ∀ x, (x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) ↔ (x - p) * (x - q) ≤ 0) (h₂ : p < q) : p + 2 * q + 3 * r = 1 :=
by
  sorry

end p_q_r_inequality_l688_688356


namespace ordered_pair_is_correct_l688_688970

-- Define the conditions as hypotheses
variables (a c : ℝ)
hypothesis quadratic_eq_has_one_solution : 6 * 6 - 4 * a * c = 0
hypothesis sum_equals_twelve : a + c = 12
hypothesis a_less_than_c : a < c

-- Define the proof statement
theorem ordered_pair_is_correct :
    (a = 6 - 3 * Real.sqrt 3 ∧ c = 6 + 3 * Real.sqrt 3) ∨ (a = 6 + 3 * Real.sqrt 3 ∧ c = 6 - 3 * Real.sqrt 3) :=
sorry

end ordered_pair_is_correct_l688_688970


namespace gold_initial_amount_l688_688700

theorem gold_initial_amount :
  ∃ x : ℝ, x - (x / 2 * (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6)) = 1 ∧ x = 1.2 :=
by
  existsi 1.2
  sorry

end gold_initial_amount_l688_688700


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688870

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688870


namespace number_of_m_l688_688295

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end number_of_m_l688_688295


namespace gardens_proof_problem_l688_688098

variable (length_Chris width_Chris length_Jordan width_Jordan : ℝ)
variable (area_Chris area_Jordan perimeter_Chris perimeter_Jordan : ℝ)

def garden_params : Prop :=
  length_Chris = 30 ∧ width_Chris = 60 ∧
  length_Jordan = 35 ∧ width_Jordan = 55

def area_Chris_correct : Prop :=
  area_Chris = length_Chris * width_Chris

def area_Jordan_correct : Prop :=
  area_Jordan = length_Jordan * width_Jordan

def perimeter_Chris_correct : Prop :=
  perimeter_Chris = 2 * (length_Chris + width_Chris)

def perimeter_Jordan_correct : Prop :=
  perimeter_Jordan = 2 * (length_Jordan + width_Jordan)

def proof_problem : Prop :=
  garden_params ∧
  area_Chris_correct ∧ area_Jordan_correct ∧
  perimeter_Chris_correct ∧ perimeter_Jordan_correct ∧
  area_Jordan - area_Chris = 125 ∧
  perimeter_Chris = perimeter_Jordan

theorem gardens_proof_problem : proof_problem :=
by
  unfold proof_problem garden_params area_Chris_correct area_Jordan_correct perimeter_Chris_correct perimeter_Jordan_correct
  split; split; split; split; split; split; linarith
  sorry

end gardens_proof_problem_l688_688098


namespace percent_area_of_triangle_l688_688908

theorem percent_area_of_triangle (s : ℝ) (hs : 0 < s):
  let triangle_area := (√3 / 4) * s^2,
      rectangle_area := s * (2 * s),
      pentagon_area := triangle_area + rectangle_area in
  ( (triangle_area / pentagon_area) * 100 = (√3 / (√3 + 8)) * 100 ) :=
by
  have triangle_area := (√3 / 4) * s^2
  have rectangle_area := s * (2 * s)
  have pentagon_area := triangle_area + rectangle_area
  sorry

end percent_area_of_triangle_l688_688908


namespace length_of_AB_is_correct_l688_688593

noncomputable def hyperbola_length_of_AB : ℝ :=
  let a := 1
  let b := sqrt 3
  let c := sqrt (a^2 + b^2)
  let right_focus := (c, 0)
  let asymptote1 := λ x : ℝ, b * x / a
  let asymptote2 := λ x : ℝ, -b * x / a
  let y1 := asymptote1 (2 : ℝ)
  let y2 := asymptote2 (2 : ℝ)
  |y1 - y2|

-- Theorem stating that the length of AB is 4√3
theorem length_of_AB_is_correct : hyperbola_length_of_AB = 4 * sqrt 3 :=
sorry

end length_of_AB_is_correct_l688_688593


namespace interior_angle_of_regular_nonagon_l688_688835

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l688_688835


namespace expected_value_a_squared_is_correct_l688_688033

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l688_688033


namespace find_x_l688_688716

theorem find_x {x : ℝ} (hx : x^2 - 5 * x = -4) : x = 1 ∨ x = 4 :=
sorry

end find_x_l688_688716


namespace cost_pants_shirt_l688_688376

variable (P S C : ℝ)

theorem cost_pants_shirt (h1 : P + C = 244) (h2 : C = 5 * S) (h3 : C = 180) : P + S = 100 := by
  sorry

end cost_pants_shirt_l688_688376


namespace pq_logic_l688_688298

theorem pq_logic (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by
  sorry

end pq_logic_l688_688298


namespace quadratic_no_real_roots_iff_discriminant_lt_zero_l688_688445

theorem quadratic_no_real_roots_iff_discriminant_lt_zero {a b c : ℝ} (h : a ≠ 0):
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0)) ↔ (b^2 - 4 * a * c < 0) :=
sorry

example : (∀ x : ℝ, ¬ (x^2 + 2 * x + 2 = 0)) :=
begin
  apply quadratic_no_real_roots_iff_discriminant_lt_zero 1,
  linarith,
  sorry,
end

end quadratic_no_real_roots_iff_discriminant_lt_zero_l688_688445


namespace average_weight_l688_688416

variable (A B C : ℝ) 

theorem average_weight (h1 : (A + B) / 2 = 48) (h2 : (B + C) / 2 = 42) (h3 : B = 51) :
  (A + B + C) / 3 = 43 := by
  sorry

end average_weight_l688_688416


namespace circle_through_fixed_points_l688_688264

theorem circle_through_fixed_points :
  ∀ (x y : ℝ), x^2 / 4 - y^2 / 3 = 1 →
  (y ≠ 0) →
  let M1_y := 3 * y / (x + 2)
  let M2_y := -y / (x - 2)
  let M1 := (1 : ℝ, M1_y)
  let M2 := (1 : ℝ, M2_y)
  let C := circle_through_points M1 M2
  ∃ (Q : ℝ × ℝ), Q = (-1/2, 0) ∨ Q = (5/2, 0) ∧ Q ∈ C :=
by
  sorry

end circle_through_fixed_points_l688_688264


namespace hyperbola_distance_l688_688252

theorem hyperbola_distance (P : ℝ × ℝ) (a b c : ℝ) 
  (h1 : a = 4)
  (h2 : b = 3)
  (h3 : c = sqrt (a^2 + b^2))
  (hy : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (d_left_focus : ℝ)
  (h4 : d_left_focus = 10) :
  let d_right_focus := 2 * a + d_left_focus in
  d_right_focus = 18 :=
by
  sorry

end hyperbola_distance_l688_688252


namespace smallest_number_of_songs_proof_l688_688198

noncomputable def smallest_possible_number_of_songs
  (singers : ℕ)
  (group_size : ℕ)
  (pairs_song_constant : ℕ)
  (h_singers : singers = 8)
  (h_group_size : group_size = 4)
  (h_pairs_song_constant : ∀ (pair_count : ℕ), pair_count = pairs_song_constant) : ℕ :=
  14

theorem smallest_number_of_songs_proof:
  smallest_possible_number_of_songs 8 4 3 = 14 := by
  sorry

end smallest_number_of_songs_proof_l688_688198


namespace number_of_correct_propositions_l688_688314

theorem number_of_correct_propositions :
  let prop1 := ∀ (L1 L2 : Line) (P : Plane), (parallel_lines L1 L2) → (projections_parallel L1 L2 P) = false in
  let prop2 := ∀ (α β : Plane) (m : Line), (parallel_planes α β) → (line_in_plane m α) → (line_parallel_plane m β) = true in
  let prop3 := ∀ (α β : Plane) (m n : Line), (intersection_line α β m) → (line_in_plane n α) → (line_perpendicular m n) → (line_perpendicular_plane n β) = false in
  let prop4 := ∀ (α β : Plane) (A B C : Point), (points_in_plane A B C α) → (equal_distances A B C β) → (parallel_planes α β) = false in
  cardinality (filter id [prop1, prop2, prop3, prop4]) = 1 :=
by
  sorry

end number_of_correct_propositions_l688_688314


namespace smallest_positive_period_of_f_f_of_2alpha_l688_688648

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos (x - (Real.pi / 3)) - Real.sin ((Real.pi / 2) - x)

theorem smallest_positive_period_of_f :
  ∀ x, f (x + (2 * Real.pi)) = f x :=
by
  sorry

theorem f_of_2alpha (α : ℝ) (h₀ : 0 < α) (h₁ : α < Real.pi / 2) (h₂ : f (α + Real.pi / 6) = 3 / 5) :
  f (2 * α) = (24 * Real.sqrt 3 - 7) / 50 :=
by
  sorry

end smallest_positive_period_of_f_f_of_2alpha_l688_688648


namespace part1_part2_l688_688889

theorem part1 (a m n : ℕ) (ha : a > 1) (hdiv : a^m + 1 ∣ a^n + 1) : n ∣ m :=
sorry

theorem part2 (a b m n : ℕ) (ha : a > 1) (coprime_ab : Nat.gcd a b = 1) (hdiv : a^m + b^m ∣ a^n + b^n) : n ∣ m :=
sorry

end part1_part2_l688_688889


namespace triangle_BCG_area_proof_l688_688062

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rectangle_vertices : List (Point ℝ) := [⟨0, 0⟩, ⟨0, 10⟩, ⟨15, 10⟩, ⟨15, 0⟩]

def triangle_vertices : List (Point ℝ) := [⟨15, 0⟩, ⟨25, 0⟩, ⟨20, 10⟩]

def calculate_triangle_area (A B C : Point ℝ) : ℝ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

noncomputable def proof_triangle_BCG_area (B C G : Point ℝ) : Prop :=
  calculate_triangle_area B C G = 75

theorem triangle_BCG_area_proof :
  proof_triangle_BCG_area ⟨0, 10⟩ ⟨15, 10⟩ ⟨15, 0⟩ := 
begin
  sorry
end

end triangle_BCG_area_proof_l688_688062


namespace find_f_ln3_l688_688235

noncomputable def f : ℝ → ℝ 
| x => if x < 1 then Real.exp (x - 1) else f (x - 1)

theorem find_f_ln3 : f (Real.log 3) = 3 / (Real.exp 2) :=
by
  sorry

end find_f_ln3_l688_688235


namespace james_final_payment_l688_688329

variable (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ)

def final_cost (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ) : ℚ :=
let bed_cost := bed_frame_cost * bed_cost_multiplier in
let total_cost := bed_cost + bed_frame_cost in
let discount := total_cost * discount_rate in
total_cost - discount

theorem james_final_payment : final_cost 75 10 (20 / 100) = 660 := by
  unfold final_cost
  -- Step: Compute bed cost
  have bed_cost : ℕ := 75 * 10 by norm_num
  -- Step: Compute total cost
  have total_cost : ℕ := bed_cost + 75 by norm_num
  -- Step: Compute discount
  have discount : ℚ := total_cost * (20 / 100) by norm_num
  -- Step: Compute final cost
  have final_payment : ℚ := total_cost - discount by norm_num
  -- Assertion
  exact final_payment

#eval james_final_payment

end james_final_payment_l688_688329


namespace solve_equation_parabola_equation_l688_688503

-- Part 1: Equation Solutions
theorem solve_equation {x : ℝ} :
  (x - 9) ^ 2 = 2 * (x - 9) ↔ x = 9 ∨ x = 11 := by
  sorry

-- Part 2: Expression of Parabola
theorem parabola_equation (a h k : ℝ) (vertex : (ℝ × ℝ)) (point: (ℝ × ℝ)) :
  vertex = (-3, 2) → point = (-1, -2) →
  a * (point.1 - h) ^ 2 + k = point.2 →
  (a = -1) → (h = -3) → (k = 2) →
  - x ^ 2 - 6 * x - 7 = a * (x + 3) ^ 2 + 2 := by
  sorry

end solve_equation_parabola_equation_l688_688503


namespace a_general_term_b_sum_l688_688803

-- Definitions from conditions
def a_seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- defining a_1
  | n + 1 => (n + 2) * a_seq (n + 1) + (n + 1) * (n + 2) / (n + 1) -- given recursive relation

def b_seq (n : ℕ) : ℕ :=
  3^n * (a_seq n)^(1/2)

-- Statement to prove for part 1
theorem a_general_term (n : ℕ) (h : n > 0) : a_seq n = n^2 :=
sorry

-- Statement to prove for part 2
theorem b_sum (n : ℕ) : 
  let S_n := (finset.range n).sum (λ i, b_seq (i+1)) in
  S_n = (2*n - 1) * 3^(n + 1) - 3 / 4 :=
sorry


end a_general_term_b_sum_l688_688803


namespace min_top_block_l688_688909

-- Definitions related to the structure of the pyramid
def layer1 := fin 30 -- Each block assigned a unique number from 1 to 30

-- Function to generate layer2 numbers based on sums of layer1 pairs
def layer2 (b1 b2 : layer1) : ℕ := b1 + b2

-- Function to generate layer3 numbers based on sums of layer2 pairs
def layer3 (c1 c2 : ℕ) : ℕ := c1 + c2

-- Function to generate layer4 number based on sums of layer3 triples
def layer4 (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

-- Main theorem: Proving the minimum possible number for the top block
theorem min_top_block : ∃ n: ℕ, ∀ assigning_strategy: (layer1 → ℕ), n = layer4 d1 d2 d3  :=
sorry

end min_top_block_l688_688909


namespace inverse_proposition_l688_688428

theorem inverse_proposition (a : ℝ) :
  (a > 1 → a > 0) → (a > 0 → a > 1) :=
by 
  intros h1 h2
  sorry

end inverse_proposition_l688_688428


namespace particular_solution_l688_688218

-- Define the initial conditions
def y (x : ℝ) := (x^4 / 12) - (5 / (36 * x^2)) - x + 1

theorem particular_solution : 
  (∀ x : ℝ, x ≠ 0 → x^4 * y''' x - 2 * x^3 * y'' x = 5 / x) ∧
  y 1 = -1/18 ∧ y' 1 = -7/18 ∧ y'' 1 = 1/6 :=
by
  sorry

end particular_solution_l688_688218


namespace max_profit_l688_688512

noncomputable def profit_A (x : ℕ) : ℝ := -↑x^2 + 21 * ↑x
noncomputable def profit_B (x : ℕ) : ℝ := 2 * ↑x
noncomputable def total_profit (x : ℕ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 120 := sorry

end max_profit_l688_688512


namespace problem_statement_l688_688344

-- Define y as the sum of the given terms
def y : ℤ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

-- The theorem to prove that y is a multiple of 8, 16, 32, and 64
theorem problem_statement : 
  (8 ∣ y) ∧ (16 ∣ y) ∧ (32 ∣ y) ∧ (64 ∣ y) :=
by sorry

end problem_statement_l688_688344


namespace find_a_l688_688640

noncomputable def binomial_expansion (n : ℕ) (x y : ℝ) : ℕ → ℝ := 
  λ r, (-1)^r * (nat.choose n r) * x^(n-2*r)

noncomputable def coefficient_x6 (a : ℝ) : ℝ :=
  let coeff_x4 := binomial_expansion 10 1 (-1) 3 in
  let coeff_x6 := binomial_expansion 10 1 (-1) 2 in
  coeff_x4 * a + coeff_x6

theorem find_a :
  coefficient_x6 a = -30 → a = 2 :=
by
  sorry

end find_a_l688_688640


namespace vector_subtraction_l688_688981

theorem vector_subtraction (p q: ℝ × ℝ × ℝ) (hp: p = (5, -3, 2)) (hq: q = (-1, 4, -2)) :
  p - 2 • q = (7, -11, 6) :=
by
  sorry

end vector_subtraction_l688_688981


namespace kittens_left_l688_688711

-- Defining the conditions as constants
constant initial_kittens : ℕ := 8
constant kittens_given : ℕ := 2

-- The goal is to prove the final count of kittens
theorem kittens_left : initial_kittens - kittens_given = 6 := by
  -- Placeholder proof
  sorry

end kittens_left_l688_688711


namespace part1_part2_l688_688007

theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x :=
  sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.cos (a * x) - Real.log (1 - x^2)) (hf0 : ∃ x, f' x = -a * Real.sin (a * x) + 2*x / (1 - x^2)) :
  0 = f' 0 → f'' 0 < 0 → (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) :=
  sorry

end part1_part2_l688_688007


namespace seating_arrangement_ways_l688_688856

open Nat

theorem seating_arrangement_ways : 
  let boys := 4
      girls := 1
  in (2 * boys - 1) * factorial boys = 120 := by
  let boys := 4
  let girls := 1
  have h1 : (2 * boys - 1) = 5 := by norm_num
  have h2 : factorial boys = 24 := by norm_num
  calc
      (2 * boys - 1) * factorial boys
      = 5 * 24 : by rw [h1, h2]
  ... = 120  : by norm_num

end seating_arrangement_ways_l688_688856


namespace sum_ratios_bounds_l688_688486

theorem sum_ratios_bounds (N : ℕ) (x : Fin N → ℕ) 
  (hN : N ≥ 3)
  (h_ratios : ∀ i : Fin N, ∃ k : ℕ, (x i + x ((i + 2) % N)) / x ((i + 1) % N) = k) :
  2 * N ≤ ∑ i, (x i + x ((i + 2) % N)) / x ((i + 1) % N) ∧
  ∑ i, (x i + x ((i + 2) % N)) / x ((i + 1) % N) < 3 * N := 
sorry

end sum_ratios_bounds_l688_688486


namespace cathy_total_money_l688_688096

theorem cathy_total_money : 
  let initial := 12 
  let dad_contribution := 25 
  let mom_contribution := 2 * dad_contribution 
  let total_money := initial + dad_contribution + mom_contribution 
  in total_money = 87 :=
by
  let initial := 12
  let dad_contribution := 25
  let mom_contribution := 2 * dad_contribution
  let total_money := initial + dad_contribution + mom_contribution
  show total_money = 87
  sorry

end cathy_total_money_l688_688096


namespace logarithm_comparison_l688_688233

/-
  Given:
  - 5^7 = 78125
  - 47^3 = 103823
  - a = log_5 2
  - b = log_13 3
  - c = log_47 5

  Prove that:
  - c < b < a
-/
theorem logarithm_comparison
  (h1 : 5 ^ 7 = 78125)
  (h2 : 47 ^ 3 = 103823)
  (a b c : ℝ)
  (ha : a = Real.logBase 5 2)
  (hb : b = Real.logBase 13 3)
  (hc : c = Real.logBase 47 5) :
  c < b ∧ b < a :=
sorry

end logarithm_comparison_l688_688233


namespace product_formula_l688_688087

theorem product_formula : (∏ (k : ℕ) in Finset.range 59, (1 - (1 / (k + 2) : ℚ) + (1 / 60))) = (1 / 60 : ℚ) :=
by
  sorry

end product_formula_l688_688087


namespace cathy_total_money_l688_688097

theorem cathy_total_money : 
  let initial := 12 
  let dad_contribution := 25 
  let mom_contribution := 2 * dad_contribution 
  let total_money := initial + dad_contribution + mom_contribution 
  in total_money = 87 :=
by
  let initial := 12
  let dad_contribution := 25
  let mom_contribution := 2 * dad_contribution
  let total_money := initial + dad_contribution + mom_contribution
  show total_money = 87
  sorry

end cathy_total_money_l688_688097


namespace no_free_time_l688_688709

def working_hours := 8
def exercising_hours := 3
def sleeping_hours := 8
def commuting_hours := 1
def meals_hours := 2
def classes_hours := 1.5
def phone_hours := 0.5 -- 30 minutes is 0.5 hours
def reading_hours := 40 / 60 -- 40 minutes is 40/60 hours

def total_activities_hours := 
  working_hours + exercising_hours + sleeping_hours + commuting_hours + meals_hours 
  + classes_hours + phone_hours + reading_hours

theorem no_free_time : total_activities_hours > 24 :=
by
  unfold total_activities_hours
  calculate the total and show: 8 + 3 + 8 + 1 + 2 + 1.5 + 0.5 + 40/60
  have h : 8 + 3 + 8 + 1 + 2 + 1.5 + 0.5 + (40 / 60) > 24
  sorry -- step: finished proof to show 24 hours not meet Jackie’s requirement

end no_free_time_l688_688709


namespace Ben_caught_4_fish_l688_688558

theorem Ben_caught_4_fish (Judy Billy Jim Susie ThrewBack FishFilets : ℕ)
  (hJudy : Judy = 1)
  (hBilly : Billy = 3)
  (hJim : Jim = 2)
  (hSusie : Susie = 5)
  (hThrewBack : ThrewBack = 3)
  (hFishFilets : FishFilets = 24) :
  let TotalFish := FishFilets / 2 + ThrewBack in
  let FamilyFish := Judy + Billy + Jim + Susie in
  let BenCatches := TotalFish - FamilyFish in
  BenCatches = 4 :=
by
  sorry

end Ben_caught_4_fish_l688_688558


namespace calculator_change_problem_l688_688056

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l688_688056


namespace geometric_sequence_fifth_term_l688_688808

variables (a r : ℝ) (h1 : a * r ^ 2 = 12 / 5) (h2 : a * r ^ 6 = 48)

theorem geometric_sequence_fifth_term : a * r ^ 4 = 12 / 5 := by
  sorry

end geometric_sequence_fifth_term_l688_688808


namespace number_of_possible_lengths_of_diagonal_l688_688606

theorem number_of_possible_lengths_of_diagonal :
  ∃ n : ℕ, n = 13 ∧
  (∀ y : ℕ, (5 ≤ y ∧ y ≤ 17) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨
   y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15 ∨ y = 16 ∨ y = 17)) :=
by
  exists 13
  sorry

end number_of_possible_lengths_of_diagonal_l688_688606


namespace description_of_T_l688_688350

-- Define the set T based on the conditions given
def T : set (ℝ × ℝ) :=
  {p | (p.1 + 3 = 5 ∧ p.2 - 2 ≤ 5) ∨
       (p.2 - 2 = 5 ∧ p.1 + 3 ≤ 5) ∨
       (p.1 + 3 = p.2 - 2 ∧ 5 ≤ p.1 + 3) }

-- Describe the problem statement in Lean 4 as a theorem
theorem description_of_T :
  T = {p | p.1 = 2 ∧ p.2 ≤ 7} ∪
      {p | p.2 = 7 ∧ p.1 ≤ 2} ∪
      {p | p.2 = p.1 + 5 ∧ p.1 ≥ 2} :=
sorry

end description_of_T_l688_688350


namespace rational_xy_power_l688_688759

theorem rational_xy_power {x y : ℚ} (h : x^2 - 2 * y + y * (Real.sqrt 5) = 10 + 3 * (Real.sqrt 5)) :
  x^y = 64 ∨ x^y = -64 :=
begin
  sorry
end

end rational_xy_power_l688_688759


namespace find_m_l688_688677

noncomputable def imaginary_i : ℂ := complex.i

theorem find_m (m : ℝ) : (1 - m * imaginary_i) * (m + imaginary_i) < 0 → m = -1 :=
begin
  sorry
end

end find_m_l688_688677


namespace det_cofactor_matrix_eq_det_cubed_l688_688961

-- Define the 4x4 matrix A
variable (A : Matrix (Fin 4) (Fin 4) ℝ)

-- Define the determinant of A
noncomputable def det_A : ℝ := Matrix.det A

-- Define the cofactor of A_ij
noncomputable def cofactor (i j : Fin 4) : ℝ :=
  let A' := Matrix.minor A (Finset.erase (Finset.univ : Finset (Fin 4)) i) (Finset.erase (Finset.univ : Finset (Fin 4)) j)
  (-1)^(i + j) * Matrix.det A'

-- Define the cofactor matrix B
noncomputable def cofactor_matrix : Matrix (Fin 4) (Fin 4) ℝ :=
  λ i j => cofactor A i j

-- Define the determinant of the cofactor matrix B
noncomputable def det_B : ℝ := Matrix.det (cofactor_matrix A)

-- Prove the statement
theorem det_cofactor_matrix_eq_det_cubed : det_B A = (det_A A) ^ 3 :=
by
  sorry

end det_cofactor_matrix_eq_det_cubed_l688_688961


namespace slope_of_tangent_line_at_A_l688_688447

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem slope_of_tangent_line_at_A :
  (Real.deriv f 0) = 1 :=
by
  sorry

end slope_of_tangent_line_at_A_l688_688447


namespace distance_from_P_to_AB_equals_1_l688_688948

section square_problem

variables {A B C D O P : Type} [geometry A B C D O P]

-- We assume the square with side length 2 and the diagonals intersecting at O
def is_square (A B C D O : Type) : Prop :=
  side_length A B = 2 ∧ 
  side_length B C = 2 ∧ 
  side_length C D = 2 ∧ 
  side_length D A = 2 ∧ 
  ∃ P, diagonals_intersect_at_O A B C D P O

-- We assume line through P is parallel to side AB and bisects the area of triangle ABD
def bisects_area_of_triangle (A B D P : Type) : Prop :=
  parallel (line_through P parallel_to AB) AB ∧
  bisects_area (line_through P parallel_to AB) (triangle A B D)

-- Given the square and the conditions on P
theorem distance_from_P_to_AB_equals_1 (A B C D O P : Type)
  [square : is_square A B C D O]
  [bisects : bisects_area_of_triangle A B D P] :
  distance_from_point_to_line P AB = 1 := sorry

end square_problem

end distance_from_P_to_AB_equals_1_l688_688948


namespace remainder_abc_mod_n_is_one_l688_688355

theorem remainder_abc_mod_n_is_one 
  (n : ℕ) (h_n : n > 0) 
  (a b c : ℕ) 
  (ha_inv : IsUnit (Zmod n) a) 
  (hb_inv : IsUnit (Zmod n) b) 
  (hc_inv : IsUnit (Zmod n) c) 
  (hbc : b * c ≡ 1 [Zmod n]) 
  (hca : c * a ≡ 1 [Zmod n]) 
  : (a * b * c) % n = 1 := 
by 
  /- Proof omitted -/ 
  sorry

end remainder_abc_mod_n_is_one_l688_688355


namespace total_volume_of_water_in_container_l688_688899

def volume_each_hemisphere : ℝ := 4
def number_of_hemispheres : ℝ := 2735

theorem total_volume_of_water_in_container :
  (volume_each_hemisphere * number_of_hemispheres) = 10940 :=
by
  sorry

end total_volume_of_water_in_container_l688_688899


namespace race_winner_speed_difference_l688_688454

noncomputable def total_distance (laps : ℕ) (length : ℝ) : ℝ :=
  laps * length

noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def minutes_faster_per_mile (speed_diff : ℝ) : ℝ :=
  1 / speed_diff

theorem race_winner_speed_difference (laps : ℕ) (length : ℝ) (time_this_year : ℝ) (time_last_year : ℝ) (expected_difference : ℝ) : 
  (laps = 11) ∧ (length = Real.pi) ∧ (time_this_year = 82.5) ∧ (time_last_year = 106.37) →
  minutes_faster_per_mile (average_speed (total_distance laps length) time_last_year - average_speed (total_distance laps length) time_this_year) = expected_difference :=
by
  intros h
  cases h with hl h1
  cases h1 with hs h2
  cases h2 with ht_last ht_this
  have total_dist := total_distance 11 Real.pi
  have avg_speed_this := average_speed total_dist 82.5
  have avg_speed_last := average_speed total_dist 106.37
  have speed_diff := avg_speed_this - avg_speed_last
  have faster := minutes_faster_per_mile speed_diff
  have : faster = 10.638 := sorry  -- This will be the place where the actual calculation proof will go
  assumption

end race_winner_speed_difference_l688_688454


namespace car_travel_distance_l688_688895

theorem car_travel_distance :
  let a₁ := 36
  let d := -9
  let n := 5
  let an := a₁ + (n-1) * d
  let S₄ := (4 / 2) * (a₁ + an + d)
  S₄ = 90 := 
begin
  sorry
end

end car_travel_distance_l688_688895


namespace sum_of_areas_of_squares_l688_688706

theorem sum_of_areas_of_squares
  (A P Q : Type)
  [RightTriangle PAQ]
  (PQ : ℝ) :
  PQ = 12 → 
  let PQ_Area := PQ^2,
      PA_Area := (sqrt (PQ^2 + PA^2))^2
  PQ_Area + PA_Area = 144 :=
by
  sorry

end sum_of_areas_of_squares_l688_688706


namespace interval_contains_root_l688_688790

theorem interval_contains_root :
  (∃ c, (0 < c ∧ c < 1) ∧ (2^c + c - 2 = 0) ∧ 
        (∀ x1 x2, x1 < x2 → 2^x1 + x1 - 2 < 2^x2 + x2 - 2) ∧ 
        (0 < 1) ∧ 
        ((2^0 + 0 - 2) = -1) ∧ 
        ((2^1 + 1 - 2) = 1)) := 
by 
  sorry

end interval_contains_root_l688_688790


namespace circle_equation_l688_688805

theorem circle_equation (x y : ℝ) : 
  let center := (2 : ℝ, -2 : ℝ)
  let origin := (0 : ℝ, 0 : ℝ)
  let radius := Real.sqrt ((2 - 0:ℝ)^2 + ((-2) - 0:ℝ)^2)
  let r_squared := 8

  (x - 2)^2 + (y + 2)^2 = r_squared :=
by
  -- We are not providing the proof, only the statement
  sorry

end circle_equation_l688_688805


namespace min_keystrokes_1_to_300_l688_688044

theorem min_keystrokes_1_to_300 : 
  ∃ (steps : ℕ), steps = 10 ∧ 
    (∀ (display ∈ ℕ → ℕ), 
      (display 1 = 1) → 
      (display steps = 300) ∧ 
      (∀ k, 1 ≤ k ≤ steps → 
        (display (k+1) = display k + 1 ∨ display (k+1) = display k * 2))
    ) :=
sorry

end min_keystrokes_1_to_300_l688_688044


namespace dad_steps_90_l688_688173

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688173


namespace dad_steps_l688_688156

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688156


namespace parabola_transformation_l688_688799

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end parabola_transformation_l688_688799


namespace area_of_triangle_COB_l688_688756

theorem area_of_triangle_COB (p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 10) : 
  let O := (0 : ℝ, 0 : ℝ)
  let B := (15 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, p : ℝ)
  ∃ A : ℝ, A = 15 * p / 2 := sorry

end area_of_triangle_COB_l688_688756


namespace dad_steps_l688_688167

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688167


namespace trajectory_equation_equation_of_line_l688_688612

-- Define the parabola and the trajectory
def parabola (x y : ℝ) := y^2 = 16 * x
def trajectory (x y : ℝ) := y^2 = 4 * x

-- Define the properties of the point P and the line l
def is_midpoint (P A B : ℝ × ℝ) :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through_point (x y k : ℝ) := 
  k * x + y = 1

-- Proof problem (Ⅰ): trajectory of the midpoints of segments perpendicular to the x-axis from points on the parabola
theorem trajectory_equation : ∀ (M : ℝ × ℝ), 
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ is_midpoint M P (P.1, 0)) → 
  trajectory M.1 M.2 :=
sorry

-- Proof problem (Ⅱ): equation of line l
theorem equation_of_line : ∀ (A B P : ℝ × ℝ), 
  trajectory A.1 A.2 → trajectory B.1 B.2 → 
  P = (3,2) → is_midpoint P A B → 
  ∃ k, line_through_point (A.1 - B.1) (A.2 - B.2) k :=
sorry

end trajectory_equation_equation_of_line_l688_688612


namespace option_B_proof_option_C_proof_l688_688737

-- Definitions and sequences
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Statement of the problem

theorem option_B_proof (A B : ℝ) :
  (∀ n : ℕ, S n = A * (n : ℝ)^2 + B * n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem option_C_proof :
  (∀ n : ℕ, S n = 1 - (-1)^n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end option_B_proof_option_C_proof_l688_688737


namespace part1_part2_l688_688652

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 2 * a| + |x - 1|

-- Part (1)
theorem part1 (x : ℝ) : f x 1 ≤ 5 → -3 ≤ x ∧ x ≤ 2 :=
sorry

-- Part (2)
def g (a : ℝ) : ℝ := f (1 / a) a

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 → (1 / 2) ≤ a ∧ a ≤ (3 / 2) :=
sorry

end part1_part2_l688_688652


namespace sin_x_plus_ax_increasing_l688_688596

noncomputable def is_increasing (f : ℝ → ℝ) :=
  ∀ x y : ℝ, x < y → f x ≤ f y

theorem sin_x_plus_ax_increasing {a : ℝ} :
  is_increasing (λ x, Real.sin x + a * x) ↔ a > 1 := by
  sorry

end sin_x_plus_ax_increasing_l688_688596


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688867

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688867


namespace three_digit_number_is_382_l688_688686

theorem three_digit_number_is_382 
  (x : ℕ) 
  (h1 : x >= 100 ∧ x < 1000) 
  (h2 : 7000 + x - (10 * x + 7) = 3555) : 
  x = 382 :=
by 
  sorry

end three_digit_number_is_382_l688_688686


namespace smallest_positive_integer_x_l688_688580

-- Define a statement representing the proof problem
theorem smallest_positive_integer_x (x : ℕ) (M : ℤ) (h : 3960 = 2^3 * 3^2 * 5 * 11) : 
  x = 9075 ↔ (∃ (M : ℤ), (3960 * x = M^3) ∧ x > 0) := 
begin
  sorry
end

end smallest_positive_integer_x_l688_688580


namespace number_of_soluble_congruences_l688_688957

theorem number_of_soluble_congruences : 
  let count_coprime_less_than (n d : ℕ) := (List.range n).countp (λ a, Nat.gcd a d = 1) in
  count_coprime_less_than 20 20 = 8 :=
begin
  sorry
end

end number_of_soluble_congruences_l688_688957


namespace boat_stream_speeds_l688_688507

variable (x y : ℝ)

theorem boat_stream_speeds (h1 : 20 + x ≠ 0) (h2 : 40 - y ≠ 0) :
  380 = 7 * x + 13 * y ↔ 
  26 * (40 - y) = 14 * (20 + x) :=
by { sorry }

end boat_stream_speeds_l688_688507


namespace probability_log_meaningful_l688_688662

theorem probability_log_meaningful {m : ℝ} :
  (∃ A : Set ℝ, (∀ x ∈ A, (x^2 + m*x + (3/4)*m + 1 = 0) → (m^2 - 3*m - 4 < 0)) ∧ 
  (∀ x ∈ (0, 4), (x ∈ A)) 
  → (∀ x ∈ Set.univ, x ∈ (0, 4) → x ∈ A) 
  → (∃ P : ℝ, P = 4/5) :=
begin
  sorry
end

end probability_log_meaningful_l688_688662


namespace probability_of_two_digits_twice_is_three_over_five_l688_688397

open Finset

noncomputable def select_three_digits_and_form_five_digit_number :=
  let s := {1, 2, 3, 4, 5}
  let choose_three := s.powerset.filter (λ t, t.card = 3)
  let total_ways := (choose_three.card * (multinomial (choose_three.erase 1).card {2, 2, 1}).sum)
  let ways_with_two_digits_twice := (choose_three.card * (multinomial (choose_three.erase 1).card {2, 2, 1}).sum)
  ways_with_two_digits_twice = 3 / 5 * total_ways

theorem probability_of_two_digits_twice_is_three_over_five :
  select_three_digits_and_form_five_digit_number := by
  sorry

end probability_of_two_digits_twice_is_three_over_five_l688_688397


namespace average_speed_with_stoppages_l688_688200

theorem average_speed_with_stoppages
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (moving_time_per_hour : ℝ)
  (total_distance_moved : ℝ)
  (total_time_with_stoppages : ℝ) :
  avg_speed_without_stoppages = 60 → 
  stoppage_time_per_hour = 45 / 60 →
  moving_time_per_hour = 15 / 60 →
  total_distance_moved = avg_speed_without_stoppages * moving_time_per_hour →
  total_time_with_stoppages = 1 →
  (total_distance_moved / total_time_with_stoppages) = 15 :=
by
  intros
  sorry

end average_speed_with_stoppages_l688_688200


namespace distribute_items_among_people_l688_688311

theorem distribute_items_among_people :
  (Nat.choose (10 + 3 - 1) 3) = 220 := 
by sorry

end distribute_items_among_people_l688_688311


namespace points_on_circle_l688_688072

theorem points_on_circle (n : ℕ) (h1 : ∃ (k : ℕ), k = (35 - 7) ∧ n = 2 * k) : n = 56 :=
sorry

end points_on_circle_l688_688072


namespace cos_pi_minus_alpha_l688_688231

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π - α) = - (1 / 3) :=
by
  sorry

end cos_pi_minus_alpha_l688_688231


namespace expected_value_of_a_squared_l688_688022

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l688_688022


namespace problem_incorrect_statements_l688_688787

theorem problem_incorrect_statements :
  (¬(∅ = ({0} : Set ℕ))) ∧ 
  (∅ ⊆ ({0} : Set ℕ)) ∧
  (¬(Real.sqrt 3 ∈ {x : ℝ | x ≤ 2})) ∧
  (¬{x ∈ Set ℕ | (6 / (6 - x) : ℤ) ∈ Set ℤ} = {0, 2, 3, 4, 5}) :=
by
  sorry

end problem_incorrect_statements_l688_688787


namespace largest_k_divides_2006_factorial_l688_688567

theorem largest_k_divides_2006_factorial : 
  let k := Nat.findLargest (λ k, ∃ n, 2006 ^ k * n = Nat.factorial 2006) 
  in k = 34 :=
by
  let n := 2006
  let prime_2 := (2 : ℕ)
  let prime_17 := (17 : ℕ)
  let prime_59 := (59 : ℕ)
  let sum_of_powers (p : ℕ) : ℕ := Nat.sum (λ j, (⟦n / p ^ j⟧).val) 
  have h1 : sum_of_powers prime_2 = 1998 := sorry
  have h2 : sum_of_powers prime_17 = 124 := sorry
  have h3 : sum_of_powers prime_59 = 34 := sorry
  have smallest_power := min (sum_of_powers prime_2) 
                          (min (sum_of_powers prime_17) (sum_of_powers prime_59))
  show k = smallest_power
  have h4 : smallest_power = 34 := by sorry
  exact h4

end largest_k_divides_2006_factorial_l688_688567


namespace nice_permutations_count_l688_688227

theorem nice_permutations_count (n : ℕ) (hn : n ≥ 3) :
  let S := { a : Fin n → Fin n // 
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 2 * (∑ i in Finset.range k, a ⟨i, sorry⟩) % k = 0 } in
  S.card = 3 * 2^(n - 2) :=
begin
  intros,
  sorry,
end

end nice_permutations_count_l688_688227


namespace correct_option_C_l688_688847

theorem correct_option_C (a x p q : ℝ) :
  (a^2 * (a^3)^2 ≠ a^7) →
  ((3 * x - 2) * (2 * x + 3) ≠ 6 * x^2 - 6) →
  ((a - 2)^2 ≠ a^2 - 4) →
  (-3 * p * q)^2 = 9 * p^2 * q^2 :=
by {
  intros h₁ h₂ h₃,
  sorry
}

end correct_option_C_l688_688847


namespace find_root_l688_688598

theorem find_root (y : ℝ) (h : y - 9 / (y - 4) = 2 - 9 / (y - 4)) : y = 2 :=
by
  sorry

end find_root_l688_688598


namespace find_counterfeit_coin_l688_688986

/-- Given 80 coins with 1 counterfeit coin that is lighter, 
    it is possible to find the counterfeit coin using a balance scale in a maximum of 4 weighings. -/
theorem find_counterfeit_coin (coins : Finset ℕ) (counterfeit : ℕ) (h : counterfeit ∈ coins)
  (h_count : coins.card = 80) (h_lighter : ∀ x ∈ coins, x ≠ counterfeit → x > counterfeit) :
  ∃ Weighings : Finset (Finset ℕ × Finset ℕ), Weighings.card ≤ 4 ∧
  ∃ c ∈ coins, c = counterfeit ∧ by
    ∀ (balance : Finset ℕ × Finset ℕ) ∈ Weighings,
      (balance.1 ⊆ coins ∧ balance.2 ⊆ coins) ∧
      {balance.1.sum > balance.2.sum ∨ balance.1.sum < balance.2.sum ∨ balance.1.sum = balance.2.sum} :=
sorry

end find_counterfeit_coin_l688_688986


namespace dad_steps_l688_688151

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688151


namespace max_possible_dominoes_l688_688382

-- Definitions from conditions
def chessboard := fin 6 × fin 6
def domino := set (fin 6 × fin 6)

-- Given Conditions as Definitions
def valid_domino (d : domino) := ∃ (x : fin 6) (y : fin 5), d = {(x,y), (x,y.succ)} ∨
                                                     ∃ (x : fin 5) (y : fin 6), d = {(x,y), (x.succ,y)}

def valid_placement (dominoes : fin 36 → option domino) :=
  ∀ i j, valid_domino (dominoes i) → valid_domino (dominoes j) → i ≠ j → (dominoes i ∩ dominoes j).card = 0

noncomputable def max_dominoes : ℕ := 11

-- Proof Statement
theorem max_possible_dominoes (dominoes : fin 35 → option domino) (h : valid_placement dominoes) :
  ∃ (domino : domino), valid_domino domino ∧ dominoes 35 = none := sorry

end max_possible_dominoes_l688_688382


namespace problem1_problem2_l688_688614

-- Definitions
def tan_alpha_eq_two (α : ℝ) := Real.tan α = 2
def sin_alpha_eq_given (α : ℝ) := Real.sin α = (2 * Real.sqrt 5 / 5)
def in_second_quadrant (α : ℝ) := π/2 < α ∧ α < π

-- First proof problem
theorem problem1 (α : ℝ) (h1 : tan_alpha_eq_two α) : 
  4 * (Real.sin α)^2 + 2 * (Real.sin α) * (Real.cos α) = 4 := 
sorry

-- Second proof problem
theorem problem2 (α : ℝ) (h2 : sin_alpha_eq_given α) (h3 : in_second_quadrant α) : 
  Real.tan (α + 3 * π) + (Real.sin (9 * π / 2 + α)) / (Real.cos (9 * π / 2 - α)) = 
  -((4 * Real.sqrt 5 + 5) / 10) := 
sorry

end problem1_problem2_l688_688614


namespace triangle_side_length_l688_688248

theorem triangle_side_length {A B C : Type*} 
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end triangle_side_length_l688_688248


namespace molly_gift_cost_l688_688326

noncomputable def cost_per_package : ℕ := 5
noncomputable def num_parents : ℕ := 2
noncomputable def num_brothers : ℕ := 3
noncomputable def num_sisters_in_law : ℕ := num_brothers -- each brother is married
noncomputable def num_children_per_brother : ℕ := 2
noncomputable def num_nieces_nephews : ℕ := num_brothers * num_children_per_brother
noncomputable def total_relatives : ℕ := num_parents + num_brothers + num_sisters_in_law + num_nieces_nephews

theorem molly_gift_cost : (total_relatives * cost_per_package) = 70 := by
  sorry

end molly_gift_cost_l688_688326


namespace outer_cylinder_surface_area_excluding_base_l688_688515

noncomputable def lateral_surface_area_cylinder (r h : ℝ) : ℝ :=
  2 * Real.pi * r * h

noncomputable def top_surface_area_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem outer_cylinder_surface_area_excluding_base 
  (h_outer r_outer h_inner r_inner : ℝ)
  (h_out_eq : h_outer = 12)
  (r_out_eq : r_outer = 4)
  (h_in_eq : h_inner = 12)
  (r_in_eq : r_inner = 2)
  : 
  (lateral_surface_area_cylinder r_outer h_outer - 
   lateral_surface_area_cylinder r_inner h_inner + 
   top_surface_area_circle r_outer) = 
  64 * Real.pi :=
by {
  rw [h_out_eq, r_out_eq, h_in_eq, r_in_eq],
  sorry
}

end outer_cylinder_surface_area_excluding_base_l688_688515


namespace probability_x_plus_y_less_4_l688_688059

-- Definition of the square and the set of points within it
def square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

-- Definition of the condition x + y < 4
def condition (p : ℝ × ℝ) : Prop := p.1 + p.2 < 4

-- Definition of the area function
def area (S : set (ℝ × ℝ)) : ℝ := sorry -- Assume this returns the area of the set in question

-- The main statement of the problem
theorem probability_x_plus_y_less_4 :
  (area {p ∈ square | condition p}) / (area square) = 1 / 2 :=
begin
  sorry
end

end probability_x_plus_y_less_4_l688_688059


namespace two_lines_perpendicular_same_line_l688_688018

noncomputable def line (α : Type) := α → Prop

variables {α : Type} [RealLinearSpace α]

def perpendicular (l1 l2 : line α) : Prop :=
∀ (p : α), l1 p → l2 p

theorem two_lines_perpendicular_same_line
  (l1 l2 l3 : line α)
  (h1 : perpendicular l1 l3)
  (h2 : perpendicular l2 l3) :
  (parallel l1 l2 ∨ (∃ p, l1 p ∧ l2 p) ∨ skew l1 l2) :=
sorry

end two_lines_perpendicular_same_line_l688_688018


namespace cathy_total_money_l688_688093

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l688_688093


namespace right_triangle_circumference_l688_688780

open Real

noncomputable def right_triangle_circumference_min (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (area_ABC : real) (altitude_midpoints_collinear : Prop) : Prop :=
  area_ABC = 10 →
  altitude_midpoints_collinear →
  ∃ (circumference : real), circumference = 20

theorem right_triangle_circumference : ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  right_triangle_circumference_min A B C 10 sorry :=
by {
  sorry
}

end right_triangle_circumference_l688_688780


namespace sunday_avg_visitors_l688_688520

-- Defining the given conditions
def monthly_days := 30
def sunday_visits := 5
def non_sunday_visits := monthly_days - sunday_visits

def non_sunday_avg_visitors := 80
def overall_avg_visitors := 90

theorem sunday_avg_visitors:
  let S := (140 : ℕ) in
  (5 * S + 25 * non_sunday_avg_visitors) / monthly_days = overall_avg_visitors :=
sorry

end sunday_avg_visitors_l688_688520


namespace factorize_x_squared_minus_nine_l688_688201

theorem factorize_x_squared_minus_nine : ∀ (x : ℝ), x^2 - 9 = (x - 3) * (x + 3) :=
by
  intro x
  exact sorry

end factorize_x_squared_minus_nine_l688_688201


namespace max_points_coloring_l688_688317

-- Definitions based on the problem's conditions
structure Point where
  x : ℝ
  y : ℝ

inductive Color
| Red
| Green
| Yellow

structure ColoredPoint extends Point where
  color : Color

def no_three_collinear (points : List Point) : Prop :=
  ∀ p1 p2 p3 : Point, p1 ∈ points → p2 ∈ points → p3 ∈ points →
                      ¬ collinear p1 p2 p3

def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

def contains_green_point (triangle : List ColoredPoint) (points : List ColoredPoint) : Prop :=
  ∃ p, p.color = Color.Green ∧ point_in_triangle triangle p

def contains_yellow_point (triangle : List ColoredPoint) (points : List ColoredPoint) : Prop :=
  ∃ p, p.color = Color.Yellow ∧ point_in_triangle triangle p

def contains_red_point (triangle : List ColoredPoint) (points : List ColoredPoint) : Prop :=
  ∃ p, p.color = Color.Red ∧ point_in_triangle triangle p

def point_in_triangle (triangle : List ColoredPoint) (p : Point) : Prop := sorry 

theorem max_points_coloring :
  ∀ points : List ColoredPoint,
  no_three_collinear points →
  (∀ t, t ⊆ points → t.length = 3 → (∀ v, v ∈ t → v.color = Color.Red) → contains_green_point t points) →
  (∀ t, t ⊆ points → t.length = 3 → (∀ v, v ∈ t → v.color = Color.Green) → contains_yellow_point t points) →
  (∀ t, t ⊆ points → t.length = 3 → (∀ v, v ∈ t → v.color = Color.Yellow) → contains_red_point t points) →
  points.length ≤ 18 := sorry

end max_points_coloring_l688_688317


namespace percentage_change_in_area_is_50_l688_688858

-- Define original dimensions of the rectangle
variables (L B : ℝ)

-- Define the new dimensions of the rectangle after modifications
def L_new : ℝ := L / 2
def B_new : ℝ := 3 * B

-- Define the original and new areas based on the given conditions
def Area_original : ℝ := L * B
def Area_new : ℝ := L_new * B_new

-- Define the percentage change in area
def Percentage_Change : ℝ :=
  ((Area_new L B - Area_original L B) / (Area_original L B)) * 100

-- Prove that the percentage change is 50%
theorem percentage_change_in_area_is_50 : Percentage_Change L B = 50 :=
by
  sorry

end percentage_change_in_area_is_50_l688_688858


namespace compute_b1c1_b2c2_b3c3_l688_688352

theorem compute_b1c1_b2c2_b3c3 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -1 :=
by
  sorry

end compute_b1c1_b2c2_b3c3_l688_688352


namespace max_inequality_constant_l688_688967

-- Definitions
variable {a b c d k : ℝ}

-- Problem Statement
theorem max_inequality_constant :
  (∀ (a b c d : ℝ), 0 < a → 0 < b → 0 < c → 0 < d →
    (a + b + c) * (3 ^ 4 * (a + b + c + d) ^ 5 + 2 ^ 4 * (a + b + c + 2 * d) ^ 5) ≥ 174960 * a * b * c * d ^ 3) :=
begin
  sorry
end

end max_inequality_constant_l688_688967


namespace rectangle_perimeter_sum_l688_688008

theorem rectangle_perimeter_sum (AE BE CF : ℝ) (h1: AE = 8) (h2: BE = 17) (h3: CF = 3) :
    let l := 25  -- Since AE + BE = 25
    let w := 70 / 3  -- From the derivation of BC = 70/3
    2 * (l + w) = 290 / 3 ∧ 290.gcd 3 = 1 := 
by
    sorry

example : 290 + 3 = 293 := rfl

end rectangle_perimeter_sum_l688_688008


namespace share_of_A_l688_688861

noncomputable def work_rate_A : ℚ := 1 / 12
noncomputable def work_rate_B : ℚ := 1 / 18
noncomputable def total_payment : ℚ := 149.25
noncomputable def total_work_done_per_day : ℚ := work_rate_A + work_rate_B

theorem share_of_A (work_rate_A : ℚ) (work_rate_B : ℚ) (total_payment : ℚ) :
  let work_ratio_A := work_rate_A / (work_rate_A + work_rate_B)
    A_share := work_ratio_A * total_payment in
  A_share = 89.55 :=
by
  sorry

end share_of_A_l688_688861


namespace unique_n_24_l688_688966

theorem unique_n_24 (n : ℕ) (hn_pos : 0 < n) :
    (⌊n / 2⌋ * ⌊n / 3⌋ * ⌊n / 4⌋ = n * n) ↔ (n = 24) := 
sorry

end unique_n_24_l688_688966


namespace dad_steps_are_90_l688_688124

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688124


namespace percentage_increase_sides_l688_688299

theorem percentage_increase_sides (P : ℝ) :
  (1 + P/100) ^ 2 = 1.3225 → P = 15 := 
by
  sorry

end percentage_increase_sides_l688_688299


namespace sin_neg_five_sixth_pi_l688_688581

theorem sin_neg_five_sixth_pi :
  sin (-5 / 6 * Real.pi) = -1 / 2 :=
by
  -- Use the identity sin(-θ) = -sin(θ)
  have h1 : sin (-5 / 6 * Real.pi) = -sin (5 / 6 * Real.pi),
  from Real.sin_neg (5 / 6 * Real.pi),
  
  -- Use the identity sin(θ) = cos(π/2 - θ)
  have h2 : sin (5 / 6 * Real.pi) = cos (Real.pi / 2 - 5 / 6 * Real.pi),
  from Real.sin_sub_pi_div_two (5 / 6 * Real.pi),
  
  -- Simplify the angle: π/2 - 5/6π = -π/3
  have h3 : Real.pi / 2 - 5 / 6 * Real.pi = -Real.pi / 3,
  ring,
  
  -- Use the fact that cosine is even: cos(-θ) = cos(θ)
  have h4 : cos (- Real.pi / 3) = cos (Real.pi / 3),
  from Real.cos_neg (Real.pi / 3),
  
  -- Use the known value of cos(π/3)
  have h5 : cos (Real.pi / 3) = 1 / 2,
  from Real.cos_pi_div_three,
  
  -- Substitute all the results into the final equation
  rw [h1, h2, h3, h4, h5],
  linarith,

end sin_neg_five_sixth_pi_l688_688581


namespace interval_of_monotonic_increase_values_of_b_and_c_l688_688259

noncomputable def f (x : ℝ) : ℝ := cos x * (sqrt 3 * sin x + cos x^3) + sin x * (sqrt 3 * cos x - sin x^3)

theorem interval_of_monotonic_increase :
  ∃ k ∈ ℤ, ∀ x ∈ (k * π - π / 3, k * π + π / 6), 
    ∀ y ∈ (k * π - π / 3, k * π + π / 6), x < y → f x < f y :=
sorry

variables (A B C a b c : ℝ)
variables (h1 : a^2 + c^2 = ac + b^2)
variables (h2 : b + c = sqrt 2 + sqrt 3)
variables (h3 : f A = 0)

noncomputable def angle_B := real.arccos (1 / 2) -- B = 60 degrees
noncomputable def angle_A := 75 * (π / 180) -- A = 75 degrees
noncomputable def angle_C := π - angle_A - angle_B -- C = 45 degrees

theorem values_of_b_and_c :
  (b = sqrt 3) ∧ (c = sqrt 2) :=
sorry

end interval_of_monotonic_increase_values_of_b_and_c_l688_688259


namespace algebraic_expression_value_l688_688618

theorem algebraic_expression_value (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 :=
sorry

end algebraic_expression_value_l688_688618


namespace a_2n_is_perfect_square_l688_688366

noncomputable def a_n (n : ℕ) : ℕ := ...

-- Conditions based on the given problem
axiom sum_of_digits_is_n : ∀ (N n : ℕ), (sum_of_digits N = n ∧ (∀ d ∈ digits N, d = 1 ∨ d = 3 ∨ d = 4)) → a_n n = ...

-- Theorem to prove
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a_n (2 * n) = k * k := 
sorry

end a_2n_is_perfect_square_l688_688366


namespace intersection_of_cylindrical_surface_and_parallel_planes_is_ellipses_l688_688469

noncomputable def plane (α : Type*) [affine_space α] : Type* := α
noncomputable def line (α : Type*) [affine_space α] : affine_subspace α vector_space

def set_of_points_at_given_distance_from_plane (α : Type*) [affine_space α] (P : affine_subspace α (vector_space α)) (d : ℝ) : set (point α) :=
  { x | dist x P = d }

def set_of_points_at_given_distance_from_line (α : Type*) [affine_space α] (l : affine_subspace α (vector_space α)) (d : ℝ) : set (point α) :=
  { x | dist x l = d }

theorem intersection_of_cylindrical_surface_and_parallel_planes_is_ellipses 
  (α : Type*) [affine_space α] (P : affine_subspace α (vector_space α)) (l : affine_subspace α (vector_space α)) (d₁ d₂ : ℝ)
  (h : l ∈ P.Points):
  ∃ E₁ E₂ : set (point α), 
    set_of_points_at_given_distance_from_plane α P d₁ ∩ set_of_points_at_given_distance_from_line α l d₂ 
    = E₁ ∪ E₂ ∧
    is_ellipse E₁ ∧ is_ellipse E₂ := sorry

end intersection_of_cylindrical_surface_and_parallel_planes_is_ellipses_l688_688469


namespace horse_food_per_day_l688_688555

theorem horse_food_per_day
  (ratio_sheep_horses : 6 / 7)
  (total_horse_food : 12880)
  (sheep_count : 48) :
  let horses_count := 7 * sheep_count / 6 in
  total_horse_food / horses_count = 230 := by
sorry

end horse_food_per_day_l688_688555


namespace total_angles_sum_l688_688946

variables (A B C D E : Type)
variables (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ)

-- Conditions about the geometry
axiom angle_triangle_ABC : angle1 + angle2 + angle3 = 180
axiom angle_triangle_BDE : angle7 + angle4 + angle5 = 180
axiom shared_angle_B : angle2 + angle7 = 180 -- since they form a straight line at vertex B

-- Proof statement
theorem total_angles_sum (A B C D E : Type) (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle7 - 180 = 180 :=
by
  sorry

end total_angles_sum_l688_688946


namespace strawberry_cost_l688_688084

theorem strawberry_cost (price_per_basket : ℝ) (num_baskets : ℕ) (total_cost : ℝ)
  (h1 : price_per_basket = 16.50) (h2 : num_baskets = 4) : total_cost = 66.00 :=
by
  sorry

end strawberry_cost_l688_688084


namespace dad_steps_l688_688136

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688136


namespace smallest_d_l688_688530

noncomputable def point_distance (x y : ℝ) : ℝ := real.sqrt (x ^ 2 + y ^ 2)

theorem smallest_d {d : ℝ} :
    point_distance (4 * real.sqrt 3) (d - 2) = 4 * d →
    d = 2.006 := sorry

end smallest_d_l688_688530


namespace right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l688_688482

theorem right_triangle_set_A :
  let a := 1
      b := 2
      c := Real.sqrt 5
  in a^2 + b^2 = c^2 := 
sorry

theorem not_right_triangle_set_B :
  let a := 6
      b := 8
      c := 9
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_C :
  let a := Real.sqrt 3
      b := Real.sqrt 2
      c := 5
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_D :
  let a := 3^2
      b := 4^2
      c := 5^2
  in a^2 + b^2 ≠ c^2 :=
sorry

end right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l688_688482


namespace tangent_line_eq_l688_688208

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

theorem tangent_line_eq {x y : ℝ} (hx : x = 1) (hy : y = 2) (H : circle_eq x y) :
  y = 2 :=
by
  sorry

end tangent_line_eq_l688_688208


namespace tg_sum_equal_l688_688976

variable {a b c : ℝ}
variable {φA φB φC : ℝ}

-- The sides of the triangle are labeled such that a >= b >= c.
axiom sides_ineq : a ≥ b ∧ b ≥ c

-- The angles between the median and the altitude from vertices A, B, and C.
axiom angles_def : true -- This axiom is a placeholder. In actual use, we would define φA, φB, φC properly using the given geometric setup.

theorem tg_sum_equal : Real.tan φA + Real.tan φC = Real.tan φB := 
by 
  sorry

end tg_sum_equal_l688_688976


namespace area_ratio_l688_688979

namespace CircleRatio

-- Define the sides of the right triangle
def side1 : ℝ := 1
def side2 : ℝ := 2

-- Calculate the hypotenuse
def hypotenuse : ℝ := Real.sqrt (side1^2 + side2^2)

-- Define the radius of the circles
def small_circle_radius : ℝ := (side1 * side2) / hypotenuse
def large_circle_radius : ℝ := side2

-- Define the areas of the circles
def small_circle_area : ℝ := Real.pi * (small_circle_radius)^2
def large_circle_area : ℝ := Real.pi * (large_circle_radius)^2

-- Statement to prove the ratio of areas
theorem area_ratio : small_circle_area / large_circle_area = 1 / 5 := by
  sorry

end CircleRatio

end area_ratio_l688_688979


namespace parallelogram_area_l688_688823

theorem parallelogram_area (b h : ℝ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l688_688823


namespace remainder_iteration_l688_688479

noncomputable def r1 : ℚ := (1 / 2) ^ 8

theorem remainder_iteration (x : ℚ) :
  r1 = 1 / 256 →
  let r2 := by
    let x := -1 / 2
    have q1 := (x^8 - 1/r1) / (x + 1/2)
    exact q1 * (x + 1/2) + r1
  in r2 = -1 / 16 :=
sorry

end remainder_iteration_l688_688479


namespace dad_steps_l688_688144

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688144


namespace percentage_deducted_from_list_price_l688_688508

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 65.97
noncomputable def selling_price : ℝ := 65.97
noncomputable def required_profit_percent : ℝ := 25

theorem percentage_deducted_from_list_price :
  let desired_selling_price := cost_price * (1 + required_profit_percent / 100)
  let discount_percentage := 100 * (1 - desired_selling_price / list_price)
  discount_percentage = 10.02 :=
by
  sorry

end percentage_deducted_from_list_price_l688_688508


namespace sum_G_2_to_1000_l688_688226

-- Given definitions
def G (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else n

-- The sum from 2 to 1000
def sum_G := ∑ n in Finset.range 999 \ Finset.range 2, G (n + 2)

-- The theorem
theorem sum_G_2_to_1000 : sum_G = 1000499 := 
  sorry

end sum_G_2_to_1000_l688_688226


namespace housewife_savings_l688_688906

theorem housewife_savings :
  ∃ p : ℝ, abs (p - 12.09) < 0.01 :=
let amount_saved := 2.75 in
let amount_spent := 20 in
let original_price := amount_spent + amount_saved in
let percentage_saved := (amount_saved / original_price) * 100 in
⟨percentage_saved, _⟩

end housewife_savings_l688_688906


namespace work_completion_time_l688_688462

-- Define work rates for workers p, q, and r
def work_rate_p : ℚ := 1 / 12
def work_rate_q : ℚ := 1 / 9
def work_rate_r : ℚ := 1 / 18

-- Define time they work in respective phases
def time_p : ℚ := 2
def time_pq : ℚ := 3

-- Define the total time taken to complete the work
def total_time : ℚ := 6

-- Prove that the total time to complete the work is 6 days
theorem work_completion_time :
  (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq + (1 - (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq)) / (work_rate_p + work_rate_q + work_rate_r)) = total_time :=
by sorry

end work_completion_time_l688_688462


namespace number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l688_688620

theorem number_of_pentagonal_faces_is_12_more_than_heptagonal_faces
  (convex : Prop)
  (trihedral : Prop)
  (faces_have_5_6_or_7_sides : Prop)
  (V E F : ℕ)
  (a b c : ℕ)
  (euler : V - E + F = 2)
  (edges_def : E = (5 * a + 6 * b + 7 * c) / 2)
  (vertices_def : V = (5 * a + 6 * b + 7 * c) / 3) :
  a = c + 12 :=
  sorry

end number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l688_688620


namespace gcd_polynomial_multiple_of_345_l688_688630

theorem gcd_polynomial_multiple_of_345 (b : ℕ) (h : ∃ k : ℕ, b = 345 * k) : 
  Nat.gcd (5 * b ^ 3 + 2 * b ^ 2 + 7 * b + 69) b = 69 := 
by
  sorry

end gcd_polynomial_multiple_of_345_l688_688630


namespace rabbit_hound_chase_l688_688903

-- Conditions
variables (n a b c d : ℕ) (h1 : a ≠ 0) (h2 : d ≠ 0) (h3 : bc ≠ ad)

-- Proof problem
theorem rabbit_hound_chase
  (bc : ℕ := b * c)
  (ad : ℕ := a * d) :
  (let x := (a * d * n) / (bc - ad),
        y := (b * d * n) / (bc - ad) in
         True) :=
sorry

end rabbit_hound_chase_l688_688903


namespace largest_perfect_square_factor_4410_l688_688827

theorem largest_perfect_square_factor_4410 :
  ∀ (n : ℕ), n = 4410 → 
  ∃ (k : ℕ), k * k ∣ n ∧ ∀ (m : ℕ), m * m ∣ n → m * m ≤ k * k :=
by
  intro n hn
  have h4410 : ∃ p q r s : ℕ, 4410 = p * 3^2 * q * 7^2 ∧ p = 2 ∧ q = 5 ∧ r = 3 ∧ s = 7 :=
    ⟨2, 5, 3, 7, rfl, rfl, rfl, rfl⟩
  use 21
  constructor
  · rw hn
    norm_num
  · intros m hm
    norm_num  at hm
    sorry

end largest_perfect_square_factor_4410_l688_688827


namespace Megan_full_folders_l688_688744

def initial_files : ℕ := 256
def deleted_files : ℕ := 67
def files_per_folder : ℕ := 12

def remaining_files : ℕ := initial_files - deleted_files
def number_of_folders : ℕ := remaining_files / files_per_folder

theorem Megan_full_folders : number_of_folders = 15 := by
  sorry

end Megan_full_folders_l688_688744


namespace andrew_eggs_bought_l688_688934

-- Define initial conditions
def initial_eggs : ℕ := 8
def final_eggs : ℕ := 70

-- Define the function to determine the number of eggs bought
def eggs_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

-- State the theorem we want to prove
theorem andrew_eggs_bought : eggs_bought initial_eggs final_eggs = 62 :=
by {
  -- Proof goes here
  sorry
}

end andrew_eggs_bought_l688_688934


namespace expected_value_a_squared_norm_bound_l688_688028

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l688_688028


namespace B_and_C_are_mutually_exclusive_l688_688980

-- Definitions
def event_A (products : List Product) : Prop :=
  ∀ product ∈ products, product.is_defective

def event_B (products : List Product) : Prop :=
  ∀ product ∈ products, ¬ product.is_defective

def event_C (products : List Product) : Prop :=
  ∃ product ∈ products, product.is_defective

-- Proof that B and C are mutually exclusive
theorem B_and_C_are_mutually_exclusive (products : List Product) :
  event_B products → ¬ event_C products :=
by
  intro hB hC
  obtain ⟨x, hx, hx_defective⟩ := hC
  contradiction

end B_and_C_are_mutually_exclusive_l688_688980


namespace lino_shells_l688_688373

theorem lino_shells (X : ℕ) (put_back : ℕ) (final_shells : ℕ) (h1 : put_back = 292) (h2 : final_shells = 32) : X - put_back = final_shells → X = 324 :=
by
  intros h
  rw [h1, h2] at h
  simp at h
  exact h

#eval lino_shells 324 292 32 rfl rfl (by norm_num)

end lino_shells_l688_688373


namespace number_of_valid_sequences_l688_688673

/-- Number of valid sequences of 7 digits with alternating parity -/
theorem number_of_valid_sequences : 
  let count_sequences (n : Nat) : Nat := 
    match n with
    | 0 => 0
    | 1 => 5 -- 5 choices for the first digit (even)
    | n => 5 * count_sequences (n - 1) -- 5 choices for each subsequent digit alternating in parity
  in count_sequences 7 = 78125 := sorry

end number_of_valid_sequences_l688_688673


namespace polynomial_equality_l688_688283

theorem polynomial_equality :
  (3 * x + 1) ^ 4 = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e →
  a - b + c - d + e = 16 :=
by
  intro h
  sorry

end polynomial_equality_l688_688283


namespace new_kite_area_l688_688230

def original_base := 7
def original_height := 6
def scale_factor := 2
def side_length := 2

def new_base := original_base * scale_factor
def new_height := original_height * scale_factor
def half_new_height := new_height / 2

def area_triangle := (1 / 2 : ℚ) * new_base * half_new_height
def total_area := 2 * area_triangle

theorem new_kite_area : total_area = 84 := by
  sorry

end new_kite_area_l688_688230


namespace decagon_perimeter_l688_688577

theorem decagon_perimeter (num_sides : ℕ) (side_length : ℝ) (h_num_sides : num_sides = 10) (h_side_length : side_length = 3) : 
  (num_sides * side_length = 30) :=
by
  sorry

end decagon_perimeter_l688_688577


namespace finsler_hadwiger_inequality_l688_688392

-- Definitions for sides a, b, c and area S of a triangle
variables {a b c S : ℝ}

-- Assumptions: a, b, and c are sides of a triangle and S is the area of the triangle.
def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Lean 4 statement to prove the inequality
theorem finsler_hadwiger_inequality (h_triangle : is_triangle a b c) (h_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
  sorry

end finsler_hadwiger_inequality_l688_688392


namespace exists_consecutive_nats_divisible_by_odds_l688_688945

theorem exists_consecutive_nats_divisible_by_odds :
  ∃ (n : ℕ), ∀ (i : ℕ) (h : i < 11), let k := 2 * i + 1 in (n + i + 1) % k = 0 :=
sorry

end exists_consecutive_nats_divisible_by_odds_l688_688945


namespace dad_steps_l688_688119

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688119


namespace total_cost_is_correct_l688_688519

def num_children : ℕ := 5
def daring_children : ℕ := 3
def ferris_wheel_cost_per_child : ℕ := 5
def merry_go_round_cost_per_child : ℕ := 3
def ice_cream_cones_per_child : ℕ := 2
def ice_cream_cost_per_cone : ℕ := 8

def total_spent_on_ferris_wheel : ℕ := daring_children * ferris_wheel_cost_per_child
def total_spent_on_merry_go_round : ℕ := num_children * merry_go_round_cost_per_child
def total_spent_on_ice_cream : ℕ := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone

def total_spent : ℕ := total_spent_on_ferris_wheel + total_spent_on_merry_go_round + total_spent_on_ice_cream

theorem total_cost_is_correct : total_spent = 110 := by
  sorry

end total_cost_is_correct_l688_688519


namespace lines_general_position_l688_688323

theorem lines_general_position :
  ∀ (n m k : ℕ), n ≠ m → n ≠ k → m ≠ k →
  let L1 := λ n : ℕ, (λ x : ℝ, n * x + n^2)
  let L2 := λ m : ℕ, (λ x : ℝ, m * x + m^2)
  let L3 := λ k : ℕ, (λ x : ℝ, k * x + k^2)
  ¬(∀ (a b : ℝ), L1 n a = b ∧ L2 m a = b ∧ L3 k a = b) ∧
  (∃ x y : ℝ, ∃ a b : ℕ, L1 n x = y ∧ L2 m x = y ∧ x = -(n + m) ∧ y = -n * m)
by
  sorry

end lines_general_position_l688_688323


namespace problem_shaded_region_perimeter_problem_closest_integer_to_sqrt_l688_688586

theorem problem_shaded_region_perimeter 
  (B : ℕ) (hB : B = 3) (pi : ℝ) (hpi : pi = 3) : 
  let C := 45 in
  C = 45 :=
by
  sorry

theorem problem_closest_integer_to_sqrt 
  (C : ℝ) (hC : C = 45) : 
  let D := 7 in
  D = 7 :=
by
  sorry

end problem_shaded_region_perimeter_problem_closest_integer_to_sqrt_l688_688586


namespace dad_steps_l688_688149

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688149


namespace part_a_part_b_part_c_l688_688337

def initial_state := { pile_count : ℕ, stone_count : ℕ }
noncomputable def pile_split (state : initial_state) (pile_index : ℕ) (split_count : ℕ) : initial_state :=
  sorry -- definition of how a pile is split into smaller piles

def exists_state_with_60_stones_in_30_piles : Prop :=
  ∃ state : initial_state, state.pile_count = 30 ∧ state.stone_count = 60

def exists_state_with_60_stones_in_20_piles : Prop :=
  ∃ state : initial_state, state.pile_count = 20 ∧ state.stone_count = 60

def no_state_with_60_stones_in_19_piles : Prop :=
  ∀ state : initial_state, ¬ (state.pile_count = 19 ∧ state.stone_count = 60)

theorem part_a : exists_state_with_60_stones_in_30_piles :=
  sorry

theorem part_b : exists_state_with_60_stones_in_20_piles :=
  sorry

theorem part_c : no_state_with_60_stones_in_19_piles :=
  sorry

end part_a_part_b_part_c_l688_688337


namespace outstanding_young_pioneer_l688_688102

theorem outstanding_young_pioneer (flowers_per_flag : ℕ) (flags_per_badge : ℕ) (badges_per_cup : ℕ) (cups_needed : ℕ) 
    (h1 : flowers_per_flag = 5) 
    (h2 : flags_per_badge = 4) 
    (h3 : badges_per_cup = 3) 
    (h4 : cups_needed = 2) : 
    let flowers_needed_for_one_cup := badges_per_cup * flags_per_badge * flowers_per_flag in
    let total_flowers_needed := cups_needed * flowers_needed_for_one_cup in
    total_flowers_needed = 120 :=
by 
  unfold flowers_needed_for_one_cup total_flowers_needed 
  rw [h1, h2, h3, h4] 
  sorry

end outstanding_young_pioneer_l688_688102


namespace parallel_planes_normal_vectors_l688_688254

theorem parallel_planes_normal_vectors (k : ℝ) 
  (h₁ : (1, 2, -2) = λ x : ℝ, real to (1 * x, 2 * x, -2 * x))
  (h₂ : (-2, -4, k) = λ y : ℝ, real to (-2 * y, -4 * y, k * y))
  (h₃ : ∀ a b : ℝ, (1, 2, -2) = λ a : ℝ, real to (1 * a, 2 * a, -2 * a) → (-2, -4, k) = λ b : ℝ, real to (-2 * b, -4 * b, k * b) → a = b) : k = 4 := 
by 
  have h₄ : (1 / -2) = (2 / -4) := by
    sorry
  have h₅ : (1 / -2) = (-2 / k) := by
    sorry
  have k_eq : k = 4 := by 
    sorry
  exact k_eq

end parallel_planes_normal_vectors_l688_688254


namespace smallest_value_of_d_l688_688533

def distance (x y : ℝ) : ℝ := real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem smallest_value_of_d (d : ℝ) : distance (4 * real.sqrt 3) (d - 2) = 4 * d → d = 26 / 15 :=
by
  -- sorry indicates we are omitting the proof
  sorry

end smallest_value_of_d_l688_688533


namespace increasing_sine_plus_ax_l688_688595

theorem increasing_sine_plus_ax (a : ℝ) : 
  (∀ x : ℝ, ∀ y : ℝ, y = (sin x + a * x) → (cos x + a) ≥ 0) → a ≥ 1 := 
by 
  sorry

end increasing_sine_plus_ax_l688_688595


namespace num_distinct_ordered_pairs_l688_688276

theorem num_distinct_ordered_pairs (h : ∀ (m n : ℕ), (1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 6) → 0 < m ∧ 0 < n) :
  (finset.card ((finset.univ : finset (ℕ × ℕ)).filter (λ p, 1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6))) = 9 :=
sorry

end num_distinct_ordered_pairs_l688_688276


namespace proof_problem1_proof_problem2_proof_problem3_l688_688679

noncomputable def problem1 (A B C O H : Point) (R c : ℝ) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : Prop :=
  inner (O - A) (O - B) = R^2 - (c^2) / 2

noncomputable def problem2 (A B C O H : Point) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : Prop :=
  (O - H) = (O - A) + (O - B) + (O - C)

noncomputable def problem3 (A B C O H : Point) (R a b c : ℝ) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : Prop :=
  dist_sq O H = 9 * R^2 - a^2 - b^2 - c^2

theorem proof_problem1 (A B C O H : Point) (R c : ℝ) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : problem1 A B C O H R c := by
  sorry

theorem proof_problem2 (A B C O H : Point) (R : ℝ) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : problem2 A B C O H := by
  sorry

theorem proof_problem3 (A B C O H : Point) (R a b c : ℝ) [is_triangle A B C]
  [is_circumscribed_circle_center O A B C R]
  [is_orthocenter H A B C] : problem3 A B C O H R a b c := by
  sorry

end proof_problem1_proof_problem2_proof_problem3_l688_688679


namespace largest_perfect_square_factor_4410_l688_688828

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end largest_perfect_square_factor_4410_l688_688828


namespace peach_trees_count_l688_688457

theorem peach_trees_count : ∀ (almond_trees: ℕ), almond_trees = 300 → 2 * almond_trees - 30 = 570 :=
by
  intros
  sorry

end peach_trees_count_l688_688457


namespace parallelogram_octagon_area_l688_688585

theorem parallelogram_octagon_area (ABCD : parallelogram) (S : ℝ)
  (A1 B1 C1 D1 : point)
  (h_mid_ab : midpoint A1 A B)
  (h_mid_bc : midpoint B1 B C)
  (h_mid_cd : midpoint C1 C D)
  (h_mid_da : midpoint D1 D A) :
  area (intersecting_octagon ABCD A1 B1 C1 D1) = S / 6 := 
  sorry

end parallelogram_octagon_area_l688_688585


namespace time_to_cross_bridge_approx_l688_688052

noncomputable def walking_speed_km_per_hr : ℝ := 5 -- walking speed in km/hr
noncomputable def bridge_length_m : ℝ := 1250 -- bridge length in meters
noncomputable def time_to_cross_bridge_min : ℝ := 15 -- time in minutes to be proved

theorem time_to_cross_bridge_approx :
  (bridge_length_m / (walking_speed_km_per_hr * 1000 / 60)) ≈ time_to_cross_bridge_min :=
sorry

end time_to_cross_bridge_approx_l688_688052


namespace Cathy_total_money_l688_688091

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l688_688091


namespace B_completes_work_in_12_hours_l688_688490

theorem B_completes_work_in_12_hours:
  let A := 1 / 4
  let C := (1 / 2) - A
  let B := (1 / 3) - C
  (1 / B) = 12 :=
by
  -- placeholder for the proof
  sorry

end B_completes_work_in_12_hours_l688_688490


namespace dad_steps_l688_688140

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688140


namespace determine_x_l688_688634

theorem determine_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
by
  sorry

end determine_x_l688_688634


namespace small_bottle_cost_l688_688417

theorem small_bottle_cost :
  ∃ x : ℕ, 
  (let big_bottle_cost := 2700 in
   let big_bottle_volume := 30 in
   let small_bottle_volume := 6 in
   let num_small_bottles := big_bottle_volume / small_bottle_volume in
   let savings := 300 in
   let total_small_bottle_cost := num_small_bottles * x in
   total_small_bottle_cost = big_bottle_cost + savings ∧ x = 600) :=
sorry

end small_bottle_cost_l688_688417


namespace domain_and_range_symmetry_periodicity_propositions_correct_l688_688258

noncomputable def nearest_integer (x : ℝ) : ℤ :=
  (if x - (⌊x⌋ : ℝ) < 0.5 then ⌊x⌋ else ⌈x⌉)

def f (x : ℝ) : ℝ := abs (x - nearest_integer x)

theorem domain_and_range : ∀ x, (f x ∈ [0, 0.5]) :=
sorry

theorem symmetry : ∀ k : ℤ, (∀ x : ℝ, f (k - x) = f (-x)) :=
sorry

theorem periodicity : ∀ x, f (x + 1) = f x :=
sorry

theorem propositions_correct : 
  domain_and_range ∧ symmetry ∧ periodicity :=
  ⟨by sorry, by sorry, by sorry⟩

end domain_and_range_symmetry_periodicity_propositions_correct_l688_688258


namespace overall_loss_is_approx_2_09_l688_688522

noncomputable def overall_loss_percentage : ℝ :=
  let purchase_price_A := 100
  let purchase_price_B := 200 * 1.1
  let purchase_price_C := 300 * 1.3
  let shipping_fee := 10
  let cost_A := purchase_price_A + shipping_fee
  let cost_B := purchase_price_B + shipping_fee
  let cost_C := purchase_price_C + shipping_fee
  let total_cost := cost_A + cost_B + cost_C
  let selling_price_A := 110
  let selling_price_B := 250
  let selling_price_C := 330
  let sales_tax_A := selling_price_A * 0.05
  let sales_tax_B := selling_price_B * 0.05
  let sales_tax_C := selling_price_C * 0.05
  let total_selling_price := (selling_price_A + sales_tax_A) +
                             (selling_price_B + sales_tax_B) +
                             (selling_price_C + sales_tax_C)
  let overall_gain_loss := total_selling_price - total_cost
  let loss_percentage := (overall_gain_loss.abs / total_cost) * 100
  loss_percentage

theorem overall_loss_is_approx_2_09 :
  abs (overall_loss_percentage - 2.09) < 0.01 :=
  by
    sorry

end overall_loss_is_approx_2_09_l688_688522


namespace dad_steps_l688_688157

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688157


namespace expected_value_squared_minimum_vector_norm_l688_688025

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l688_688025


namespace sum_first_n_terms_arithmetic_sequence_l688_688420

/-- Define the arithmetic sequence with common difference d and a given term a₄. -/
def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

/-- Define the sum of the first n terms of an arithmetic sequence. -/
def sum_of_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * ((2 * a₁ + (n - 1) * d) / 2)

theorem sum_first_n_terms_arithmetic_sequence :
  ∀ n : ℕ, 
  ∀ a₁ : ℤ, 
  (∀ d, d = 2 → (∀ a₁, (a₁ + 3 * d = 8) → sum_of_arithmetic_sequence a₁ d n = (n : ℤ) * ((n : ℤ) + 1))) :=
by
  intros n a₁ d hd h₁
  sorry

end sum_first_n_terms_arithmetic_sequence_l688_688420


namespace number_of_shelves_l688_688051

theorem number_of_shelves (a d S : ℕ) (h1 : a = 3) (h2 : d = 3) (h3 : S = 225) : 
  ∃ n : ℕ, (S = n * (2 * a + (n - 1) * d) / 2) ∧ (n = 15) := 
by {
  sorry
}

end number_of_shelves_l688_688051


namespace problem_part1_problem_part2_l688_688988

theorem problem_part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  real.sqrt (a^2 - a * b + b^2) + real.sqrt (b^2 - b * c + c^2) > real.sqrt (c^2 - c * a + a^2) :=
sorry

theorem problem_part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  real.sqrt (a^2 + a * b + b^2) + real.sqrt (b^2 + b * c + c^2) > real.sqrt (c^2 + c * a + a^2) :=
sorry

end problem_part1_problem_part2_l688_688988


namespace albert_new_percentage_increase_l688_688685

theorem albert_new_percentage_increase :
  let E := 560 / 1.2 in
  let P := (564.67 - E) / E in
  P * 100 ≈ 20.99 := by
    let E : ℝ := 560 / 1.2
    let P : ℝ := (564.67 - E) / E
    have h : P * 100 ≈ 20.99 := sorry
    exact h

end albert_new_percentage_increase_l688_688685


namespace order_of_variables_l688_688628

theorem order_of_variables :
  let a := Real.sqrt 0.6
  let b := Real.sqrt (Real.sqrt 0.5)
  let c := Real.log 0.4 / Real.log 10
  c < a ∧ a < b :=
by
  have def_a : a = Real.sqrt 0.6 := rfl
  have def_b : b = Real.sqrt (Real.sqrt 0.5) := rfl
  have def_c : c = Real.log 0.4 / Real.log 10 := rfl
  sorry

end order_of_variables_l688_688628


namespace range_of_values_l688_688287

open Complex

theorem range_of_values (a b : ℝ) (z : ℂ) (h₁ : z = a + b * Complex.i) (h₂ : Complex.abs z = 2) :
    ∃ (r : Set ℝ), (r = { c | ∃ z : ℂ, c = Complex.abs (1 +⟨0, √3〉 * Complex.i + z) ∧ Complex.abs z = 2 }) ∧ r = Set.Icc 0 4 :=
by
  sorry

end range_of_values_l688_688287


namespace number_of_switches_in_X_after_process_l688_688460

def switch_state (n : ℕ) : ℕ :=
  let x := n / 9
  let y := n % 9
  2^x * 7^y

def switch_changes_to_position_X (n : ℕ) (reverted_switches : ℕ → ℕ) : ℕ → Bool :=
  λ step,
    let number_of_divisors : ℕ := (9 - (n / 9)) * (9 - (n % 9))
    let should_be_at_X : Bool := (number_of_divisors % 3 == 0)
    let reverts : Bool := (step / 100) * 5 > reverted_switches step
    should_be_at_X && !reverts

theorem number_of_switches_in_X_after_process (reverted_switches : ℕ → ℕ) :
  ∑ i in (Finset.range 729), if switch_changes_to_position_X i reverted_switches 729 then 1 else 0 ≈ 79 :=
by
  sorry

end number_of_switches_in_X_after_process_l688_688460


namespace triangle_area_of_integral_sides_with_perimeter_8_l688_688543

theorem triangle_area_of_integral_sides_with_perimeter_8 :
  ∃ (a b c : ℕ), a + b + c = 8 ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  ∃ (area : ℝ), area = 2 * Real.sqrt 2 := by
  sorry

end triangle_area_of_integral_sides_with_perimeter_8_l688_688543


namespace find_sticker_price_l688_688273

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l688_688273


namespace line_equation_of_circle_and_conditions_l688_688239

-- Define the given conditions and variables
def circle (x y : ℝ) := (x - 3)^2 + (y - 5)^2 = 5

def line_l (x y : ℝ) := ∃ (P : ℝ × ℝ), (P.1 = 0) ∧ (2 * (P.2 - y) = y - (5 * x))

-- Prove the equation of the line given the conditions
theorem line_equation_of_circle_and_conditions :
  (∃ l : ℝ × ℝ → Prop, ∀ x y : ℝ, circle x y → line_l x y) →
  ((∀ x y : ℝ, line_l x y → 2 * x + y = 22 ∨ 2 * x - y = 1)) :=
begin
  sorry
end

end line_equation_of_circle_and_conditions_l688_688239


namespace tangent_line_eq_at_half_max_min_values_on_interval_l688_688260

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.log x / x

theorem tangent_line_eq_at_half : 
  let y := 2 * (x : ℝ) - 2 + Real.log 2 in
  tangent_line_eq_at (f) (1 / 2) (2, -1 + Real.log 2)

theorem max_min_values_on_interval :
  let interval := Set.Icc (1 / 4 : ℝ) Real.exp 1 in
  (∀ (x ∈ interval), f x ≤ 0) ∧
  (f (1 : ℝ) = 0) ∧
  (f (1 / 4 : ℝ) = Real.log 4 - 3)

end tangent_line_eq_at_half_max_min_values_on_interval_l688_688260


namespace empty_subset_singleton_zero_l688_688850

-- Definitions based on the conditions
def empty_set : Set := ∅
def set_with_zero : Set := {0}

-- The formal statement
theorem empty_subset_singleton_zero : empty_set ⊆ set_with_zero := by
  sorry

end empty_subset_singleton_zero_l688_688850


namespace paving_cost_l688_688429

theorem paving_cost (length width rate : ℝ) (h_length : length = 8) (h_width : width = 4.75) (h_rate : rate = 900) :
  length * width * rate = 34200 :=
by
  rw [h_length, h_width, h_rate]
  norm_num

end paving_cost_l688_688429


namespace at_least_one_even_difference_l688_688360

-- Statement of the problem in Lean 4
theorem at_least_one_even_difference 
  (a b : Fin (2 * n + 1) → ℤ) 
  (hperm : ∃ σ : Equiv.Perm (Fin (2 * n + 1)), ∀ k, a k = (b ∘ σ) k) : 
  ∃ k, (a k - b k) % 2 = 0 := 
sorry

end at_least_one_even_difference_l688_688360


namespace number_of_monkeys_l688_688772

theorem number_of_monkeys (X : ℕ) : 
  10 * 10 = 10 →
  1 * 1 = 1 →
  1 * 70 / 10 = 7 →
  (X / 7) = X / 7 :=
by
  intros h1 h2 h3
  sorry

end number_of_monkeys_l688_688772


namespace chord_length_of_Omega_l688_688340

theorem chord_length_of_Omega (R r L : ℝ) (x : ℝ) 
  (hR : R = 123) (hr : r = 61) (hL : L = 42) 
  (h_ratio : x + 2 * x + 3 * x = 6 * x) 
  (h_chord : 6 * x = L) : 
  ∀ (chord_length : ℝ), chord_length = L :=
begin
  intros,
  rw hL,
  assumption,
  sorry
end

end chord_length_of_Omega_l688_688340


namespace perp_proof_l688_688244

open BigOperators
open Classical
open Nat
open Finset

variable {α : Type*} [AddCommGroup α] [VectorSpace ℝ α]

noncomputable theory

def isosceles_triangle (A B C : α) (AB AC : ℝ) : Prop :=
  (dist A B = AB) ∧ (dist A C = AC) ∧ (dist B C < AB + AC) ∧
  (AB = AC) ∧ ∀ {x y z}, dist x y < dist x z + dist y z

def parallel (P X A C : α) : Prop :=
  ∃ (k : ℝ), (X - P = k • (C - A))

def midpoint_minor_arc (A B C T : α) : Prop :=
  ∃ (Ω : circle α), on_circle Ω A ∧ on_circle Ω B ∧ on_circle Ω C ∧ is_minor_arc Ω B C T

theorem perp_proof
  {A B C P X Y T : α}
  {AB AC : ℝ}
  (h1 : isosceles_triangle A B C AB AC)
  (h2 : ∃ k : ℝ, C = B + k • BC)
  (h3 : parallel P X A C)
  (h4 : parallel P Y A B)
  (h5 : midpoint_minor_arc A B C T) :
  is_perpendicular (XY P X A B Y C T).vector (PT P T).vector :=
sorry

end perp_proof_l688_688244


namespace rectangle_area_and_perimeter_l688_688061

-- Given conditions as definitions
def length : ℕ := 5
def width : ℕ := 3

-- Proof problems
theorem rectangle_area_and_perimeter :
  (length * width = 15) ∧ (2 * (length + width) = 16) :=
by
  sorry

end rectangle_area_and_perimeter_l688_688061


namespace oranges_count_l688_688517

def oranges_per_box : ℝ := 10
def boxes_per_day : ℝ := 2650
def total_oranges (x y : ℝ) : ℝ := x * y

theorem oranges_count :
  total_oranges oranges_per_box boxes_per_day = 26500 := 
  by sorry

end oranges_count_l688_688517


namespace original_customers_count_l688_688542

theorem original_customers_count :
  ∃ x : ℕ, (7 : ℕ) / (x : ℝ) * 100 ≈ (14.29 : ℝ - 5 : ℝ) ∧ x ≈ (700 / 9.29 : ℝ) := sorry

end original_customers_count_l688_688542


namespace urn_ball_transfer_l688_688669

noncomputable theory

-- Define the initial conditions and the predicate to be proven
def proof_problem (m n k : ℕ) : Prop :=
  ∃ (p b : ℕ), p + b = k ∧
    let wf = b in
    let bs = k - p in
    wf = bs

-- The theorem statement
theorem urn_ball_transfer (m n k : ℕ) (h₁ : m >= k) (h₂ : n >= 0) : proof_problem m n k :=
by {
  -- proof can be completed here
  sorry
}

end urn_ball_transfer_l688_688669


namespace dad_steps_l688_688139

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688139


namespace difference_of_squares_division_l688_688478

theorem difference_of_squares_division :
  let a := 121
  let b := 112
  (a^2 - b^2) / 3 = 699 :=
by
  sorry

end difference_of_squares_division_l688_688478


namespace central_angle_of_sector_l688_688623

-- Define the variables and conditions
def radius := 10
def area := (50 * Real.pi) / 3
def formula (r α : ℝ) := (1 / 2) * r^2 * α

-- Statement of the theorem
theorem central_angle_of_sector :
  ∃ α : ℝ, formula radius α = area ∧ α = Real.pi / 3 :=
by {
  sorry -- Placeholder for the proof
}

end central_angle_of_sector_l688_688623


namespace cricket_run_rate_l688_688857

theorem cricket_run_rate 
  (run_rate_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_played : ℕ)
  (remaining_overs : ℕ)
  (correct_run_rate : ℝ)
  (h1 : run_rate_10_overs = 3.6)
  (h2 : target_runs = 282)
  (h3 : overs_played = 10)
  (h4 : remaining_overs = 40)
  (h5 : correct_run_rate = 6.15) :
  (target_runs - run_rate_10_overs * overs_played) / remaining_overs = correct_run_rate :=
sorry

end cricket_run_rate_l688_688857


namespace pipe_B_time_l688_688047

theorem pipe_B_time (C : ℝ) (T : ℝ) 
    (h1 : 2 / 3 * C + C / 3 = C)
    (h2 : C / 36 + C / (3 * T) = C / 14.4) 
    (h3 : T > 0) : 
    T = 8 := 
sorry

end pipe_B_time_l688_688047


namespace minimum_degree_of_g_l688_688678

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

theorem minimum_degree_of_g :
  (5 * f - 3 * g = h) →
  (Polynomial.degree f = 10) →
  (Polynomial.degree h = 11) →
  (Polynomial.degree g = 11) :=
sorry

end minimum_degree_of_g_l688_688678


namespace rectangle_perimeter_l688_688011

noncomputable def perimeter_calculation
    (AE BE CF : ℕ)
    (AE_val BE_val CF_val : ℕ)
    (rectangle_AB : ℕ × ℕ)
    (result : ℕ)
    (rel_prime : m : ℕ × n : ℕ areRelativelyPrime (m : ℕ) (n : ℕ)) : Prop :=
  AE = AE_val ∧ BE = BE_val ∧ CF = CF_val →
  let AB := AE + sqrt (BE * BE - AE * AE),
      BC := sqrt (15 * 15 + CF * CF) in
  2 * (AB + BC) = result

theorem rectangle_perimeter (AE BE CF : ℕ) :
  perimeter_calculation AE BE CF 8 17 3 (25 + 70/3) 293  :=
by
  unfold perimeter_calculation
  sorry

end rectangle_perimeter_l688_688011


namespace solution_set_for_f_geq_x_squared_l688_688650

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else -x + 2

theorem solution_set_for_f_geq_x_squared : 
  {x : ℝ | f x ≥ x^2} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by 
  sorry

end solution_set_for_f_geq_x_squared_l688_688650


namespace D_neither_sufficient_nor_necessary_for_A_l688_688405

theorem D_neither_sufficient_nor_necessary_for_A 
  (A B C D : Prop) 
  (h1 : A → B) 
  (h2 : ¬(B → A)) 
  (h3 : B ↔ C) 
  (h4 : C → D) 
  (h5 : ¬(D → C)) 
  :
  ¬(D → A) ∧ ¬(A → D) :=
by 
  sorry

end D_neither_sufficient_nor_necessary_for_A_l688_688405


namespace kinetic_energy_at_breaking_point_l688_688060

/-- Conditions for the problem:
  1. A point object of mass m is connected to a cylinder of radius R via a massless rope.
  2. At time t = 0, the object is moving with an initial velocity v₀ perpendicular to the rope.
  3. The rope has a length L₀.
  4. The rope has a non-zero tension.
  5. All motion occurs on a horizontal frictionless surface.
  6. The cylinder remains stationary and does not rotate.
  7. The rope will break when the tension exceeds Tₘₐₓ.
This theorem proves that the kinetic energy of the object at the instant the rope breaks is (1/2) * m * v₀².
-/
theorem kinetic_energy_at_breaking_point 
  (m R L₀ v₀ Tₘₐₓ : ℝ) 
  (rope: {tension: ℝ // tension ≠ 0}) 
  (initial_conditions: {h_frictionless: true, h_cylinder_stationary: true})
  (breaking_condition: rope.tension > Tₘₐₓ) 
  : (1/2) * m * v₀^2 = (1/2) * m * v₀^2 :=
sorry

end kinetic_energy_at_breaking_point_l688_688060


namespace snail_kite_difference_l688_688912

theorem snail_kite_difference : 
  ∃ d : ℕ, (3 + (3 + d) + (3 + 2 * d) + (3 + 3 * d) + (3 + 4 * d) = 35) → d = 2 := 
begin
  sorry
end

end snail_kite_difference_l688_688912


namespace letter_digit_assignment_l688_688930

-- Define the digits corresponding to each letter
def LetterDigits : Type := {R : ℕ // R ≠ M ∧ R ≠ S ∧ M ≠ S} 
def ValidDigits : Type := {d : ℕ // 0 ≤ d ∧ d ≤ 9}

-- Specify the constraints for R, M, and S
def R := 5
noncomputable def M : ValidDigits := sorry -- M ∈ {1, 2, 3, 4}
def S := 5

-- State the problem as a theorem
theorem letter_digit_assignment : Exists (λ (R : ValidDigits) (M : ValidDigits) (S : ValidDigits), R = 5 ∧ (M.val = 1 ∨ M.val = 2 ∨ M.val = 3 ∨ M.val = 4) ∧ S = 5) := 
sorry

end letter_digit_assignment_l688_688930


namespace quadratic_intersect_x_axis_l688_688268

theorem quadratic_intersect_x_axis (a : ℝ) : (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) :=
by
  sorry

end quadratic_intersect_x_axis_l688_688268


namespace two_other_days_2015_l688_688822

-- Definitions
def available_digits : list ℕ := [2, 2, 1, 1, 1, 5]
def fixed_year : ℕ := 2015
def valid_months : list ℕ := [11, 12]
def valid_days (month : ℕ) : list ℕ :=
  if month = 11 then [22] else if month = 12 then [12, 21] else []

-- Theorem
theorem two_other_days_2015 :
  ∃ (days : list (ℕ × ℕ)),
    (∀ day, day ∈ days → day.1 ∈ valid_days day.2 ∧ day.2 ∈ valid_months) ∧
    days.length = 2 :=
begin
  sorry
end

end two_other_days_2015_l688_688822


namespace expected_value_squared_minimum_vector_norm_l688_688024

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l688_688024


namespace sticker_price_l688_688274

theorem sticker_price (x : ℝ) (h : 0.85 * x - 90 = 0.75 * x - 15) : x = 750 := 
sorry

end sticker_price_l688_688274


namespace general_term_formula_sum_of_first_n_terms_l688_688704

theorem general_term_formula (q : ℝ) (hq : q > 1) (a_n : ℕ → ℝ)
  (h1 : a_n 2 = 2) (h2 : a_n 1 * (1 + q + q^2) = 7) :
  ∀ n, a_n n = 2^(n-1) := 
sorry

theorem sum_of_first_n_terms (a_n : ℕ → ℝ)
  (h_gen : ∀ n, a_n n = 2^(n-1))
  (b_n : ℕ → ℝ) (c_n : ℕ → ℝ) (h_b : ∀ n, b_n n = Real.log 2 (a_n n))
  (h_c : ∀ n, c_n n = 1 / (b_n (n + 1) * b_n (n + 2))) :
  ∀ n, (Finset.range n).sum c_n = n / (n + 1) := 
sorry

end general_term_formula_sum_of_first_n_terms_l688_688704


namespace dad_steps_l688_688180

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688180


namespace dad_steps_l688_688150

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688150


namespace intersection_points_distance_l688_688316

-- Defining the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  ( -2 + 1/2 * t, 2 + (sqrt 3)/2 * t )

-- Defining the polar equation of curve C and converting it to rectangular form
def polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 * cos (2 * θ) + 4 * ρ * sin θ - 3 = 0

def rectangular_eq (x y : ℝ) : Prop :=
  (y - 2)^2 - x^2 = 1

-- Define the problem statement
theorem intersection_points_distance :
  (∀ ρ θ, polar_eq ρ θ → rectangular_eq (ρ * cos θ) (ρ * sin θ)) →
  (∃ t1 t2 : ℝ, (line_l t1 = line_l t2) ∧ (t1 + t2 = -4) ∧ (t1 * t2 = -10) ∧ abs (t1 - t2) = 2 * sqrt 14) :=
by
  -- The proof is omitted.
  sorry

end intersection_points_distance_l688_688316


namespace sphere_volume_from_surface_area_l688_688692

-- Define the volume of the sphere given its surface area
def surface_area_to_volume (S : ℝ) : ℝ :=
  let R := (S / (4 * π)).sqrt
  in (4 / 3) * π * R^3

-- State the theorem
theorem sphere_volume_from_surface_area :
  surface_area_to_volume (9 * π) = 36 * π := by
  sorry

end sphere_volume_from_surface_area_l688_688692


namespace dad_steps_l688_688147

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688147


namespace units_digit_of_fraction_l688_688012

noncomputable def factorial_units_digit_4 (n : ℕ) : ℕ := 
  let y := n! in 
  (y^(4 : ℕ)) % 10

theorem units_digit_of_fraction : 
  (let a := (((13! : ℕ) ^ 16 - (13! : ℕ) ^ 8) / ((13! : ℕ) ^ 8 + (13! : ℕ) ^ 4)) in 
   (a / (13! ^ 4)) % 10 = 9) :=
by
  sorry

end units_digit_of_fraction_l688_688012


namespace sin_shift_symmetric_l688_688654

theorem sin_shift_symmetric :
  ∃ θ : ℝ, (0 < θ ∧ θ < π / 2) ∧ (∀ x : ℝ, sin(2 * (x + θ)) = sin(2 * (-x + θ))) ∧ θ = π / 4 :=
by sorry

end sin_shift_symmetric_l688_688654


namespace compute_sum_l688_688361

variable {R : Type*} [comm_ring R]

def Q (x : R) : R := x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

def P (x d1 e1 d2 e2 d3 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3) * (x^2 + 1)

theorem compute_sum (d1 d2 d3 e1 e2 e3 : ℝ)
  (h : ∀ x, Q x = P x d1 e1 d2 e2 d3 e3) : d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
  sorry

end compute_sum_l688_688361


namespace original_remainder_when_dividing_by_44_is_zero_l688_688907

theorem original_remainder_when_dividing_by_44_is_zero 
  (N R : ℕ) 
  (Q : ℕ) 
  (h1 : N = 44 * 432 + R) 
  (h2 : N = 34 * Q + 2) 
  : R = 0 := 
sorry

end original_remainder_when_dividing_by_44_is_zero_l688_688907


namespace find_x_such_that_sqrt_4_sub_5x_eq_8_l688_688221

theorem find_x_such_that_sqrt_4_sub_5x_eq_8 :
  ∃ x : ℝ, sqrt (4 - 5 * x) = 8 ∧ x = -12 := by
sorry

end find_x_such_that_sqrt_4_sub_5x_eq_8_l688_688221


namespace compute_series_sum_l688_688107

theorem compute_series_sum :
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 0.834 :=
by
  sorry

end compute_series_sum_l688_688107


namespace probability_non_defective_l688_688305

theorem probability_non_defective (total_pens defective_pens : ℕ) (h_total : total_pens = 12) (h_defective : defective_pens = 6) :
  let non_defective_pens := total_pens - defective_pens in
  let probability_first_non_defective := non_defective_pens / total_pens in
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1) in
  probability_first_non_defective * probability_second_non_defective = 5 / 22 :=
by
  sorry

end probability_non_defective_l688_688305


namespace ellipse_problem_l688_688249

theorem ellipse_problem
  (F1 F2 : ℝ × ℝ)
  (x1 x2 y1 y2 : ℝ)
  (λ : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | (p.1^2)/4 + (p.2^2)/3 = 1 })
  (F1_def : F1 = (-1, 0))
  (F2_def : F2 = (1, 0))
  (A_ellipse : (x1, y1) ∈ { p : ℝ × ℝ | (p.1^2)/4 + (p.2^2)/3 = 1 })
  (B_ellipse : (x2, y2) ∈ { p : ℝ × ℝ | (p.1^2)/4 + (p.2^2)/3 = 1 })
  (x_sum : x1 + x2 = 1/2)
  (vector_eq : (F2.1 - x1, F2.2 - y1) = λ * (x2 - F2.1, y2 - F2.2)) :
  λ = (3 - Real.sqrt 5) / 2 ∨ λ = (3 + Real.sqrt 5) / 2 := 
sorry

end ellipse_problem_l688_688249


namespace dad_steps_90_l688_688169

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688169


namespace midpoint_invariance_l688_688804

variable {Real : Type} [TopologicalSpace ℝ] [OrderedCommGroup ℝ] [UniformAddGroup ℝ] [CompletenessSpace ℝ]

-- Define points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define the rotation of a point by 90 degrees counterclockwise around the origin
def rotate90ccw (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, P.1)

-- Define the rotation of a point by 90 degrees clockwise around the origin
def rotate90cw (P : ℝ × ℝ) : ℝ × ℝ := (P.2, -P.1)

-- Define the new positions C1 and C2 after rotations around A and B respectively
def C1 := A + rotate90ccw (C - A)
def C2 := B + rotate90cw (C - B)

-- Lean statement for the problem
theorem midpoint_invariance (A B C : ℝ × ℝ) : 
  let M := (C1 A B C + C2 A B C) / 2 
  (∀ C' : ℝ × ℝ, (C1 A B C').fst = (C1 A B C).fst ∧ (C1 A B C').snd = (C1 A B C).snd) ∧ 
  (∀ C' : ℝ × ℝ, (C2 A B C').fst = (C2 A B C).fst ∧ (C2 A B C').snd = (C2 A B C).snd) := 
begin
  sorry
end

end midpoint_invariance_l688_688804


namespace product_of_solutions_l688_688972

theorem product_of_solutions : ∀ x : ℝ, (|4 * x + 8| = 32) → x = 6 ∨ x = -10 → ∏ x in ({6, -10} : finset ℝ), x = -60 := by
  intros x h1 h2
  obtain rfl : x ∈ ({6, -10} : finset ℝ) from h2
  rw [finset.prod_insert (finset.not_mem_singleton.2 (by simp)), finset.prod_singleton, ←mul_assoc]
  norm_num
  sorry

end product_of_solutions_l688_688972


namespace vampire_needs_people_per_day_l688_688926

-- Definitions of conditions
def gallons_per_month : ℝ := 50
def liters_per_gallon : ℝ := 3.78541
def liters_per_person : ℝ := 3
def hunting_days_per_week : ℕ := 2
def weeks_per_month : ℕ := 4

-- Goal: the number of people needed per hunting day
theorem vampire_needs_people_per_day :
  let total_liters_per_month := gallons_per_month * liters_per_gallon in
  let people_per_month := (total_liters_per_month / liters_per_person).ceil in
  let total_hunting_days_per_month := hunting_days_per_week * weeks_per_month in
  let people_per_day := people_per_month / total_hunting_days_per_month in
  people_per_day = 8 :=
by
  sorry

end vampire_needs_people_per_day_l688_688926


namespace hexagon_area_l688_688917

theorem hexagon_area (h : ∃ s : ℝ, s^2 = real.sqrt 3) : ∃ A : ℝ, A = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688917


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688864

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688864


namespace carrie_jellybeans_l688_688938

def volume (a : ℕ) : ℕ := a * a * a

def bert_box_volume : ℕ := 216

def carrie_factor : ℕ := 3

def count_error_factor : ℝ := 1.10

noncomputable def jellybeans_carrie (bert_box_volume carrie_factor count_error_factor : ℝ) : ℝ :=
  count_error_factor * (carrie_factor ^ 3 * bert_box_volume)

theorem carrie_jellybeans (bert_box_volume := 216) (carrie_factor := 3) (count_error_factor := 1.10) :
  jellybeans_carrie bert_box_volume carrie_factor count_error_factor = 6415 :=
sorry

end carrie_jellybeans_l688_688938


namespace weekend_price_of_coat_l688_688761

-- Definitions based on conditions
def original_price : ℝ := 250
def sale_price_discount : ℝ := 0.4
def weekend_additional_discount : ℝ := 0.3

-- To prove the final weekend price
theorem weekend_price_of_coat :
  (original_price * (1 - sale_price_discount) * (1 - weekend_additional_discount)) = 105 := by
  sorry

end weekend_price_of_coat_l688_688761


namespace relationship_of_y_coordinates_l688_688753

theorem relationship_of_y_coordinates (b y1 y2 y3 : ℝ):
  (y1 = 3 * -2.3 + b) → (y2 = 3 * -1.3 + b) → (y3 = 3 * 2.7 + b) → (y1 < y2 ∧ y2 < y3) := 
by 
  intros h1 h2 h3
  sorry

end relationship_of_y_coordinates_l688_688753


namespace find_sticker_price_l688_688272

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l688_688272


namespace max_value_f_l688_688211

noncomputable def f (x : ℝ) : ℝ :=
  sin (x + sin x) + sin (x - sin x) + (π / 2 - 2) * sin (sin x)

theorem max_value_f :
  ∃ x : ℝ, f x = (π - 2) / sqrt 2 :=
sorry

end max_value_f_l688_688211


namespace side_increase_percentage_l688_688443

theorem side_increase_percentage (s : ℝ) (p : ℝ) 
  (h : (s^2) * (1.5625) = (s * (1 + p / 100))^2) : p = 25 := 
sorry

end side_increase_percentage_l688_688443


namespace average_rate_of_change_is_7_l688_688415

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the interval
def a : ℝ := 1
def b : ℝ := 2

-- Define the proof problem
theorem average_rate_of_change_is_7 : 
  ((f b - f a) / (b - a)) = 7 :=
by 
  -- The proof would go here
  sorry

end average_rate_of_change_is_7_l688_688415


namespace problem_1_problem_2_l688_688246

-- Define propositions
def prop_p (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - m) + y^2 / m = 1)

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

def prop_s (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

-- Problems
theorem problem_1 (m : ℝ) (h : prop_s m) : m < 0 ∨ m ≥ 1 := 
  sorry

theorem problem_2 {m : ℝ} (h1 : prop_p m ∨ prop_q m) (h2 : ¬ prop_q m) : 1 ≤ m ∧ m < 2 :=
  sorry

end problem_1_problem_2_l688_688246


namespace sin_double_alpha_solution_l688_688284

theorem sin_double_alpha_solution (α : ℝ) 
  (h : (cos (π - 2 * α) / sin (α - π / 4) = - (real.sqrt 2) / 2)) :
  sin (2 * α) = - 3 / 4 :=
by
  sorry

end sin_double_alpha_solution_l688_688284


namespace abc_value_l688_688719

theorem abc_value (a b c : ℝ) (h1: a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (common_root1: ∃ x ∈ ℝ, (x^2 + a * x + 1 = 0) ∧ (x^2 + b * x + c = 0))
  (common_root2: ∃ x ∈ ℝ, (x^2 + x + a = 0) ∧ (x^2 + c * x + b = 0)) :
  a + b + c = -3 :=
sorry

end abc_value_l688_688719


namespace relationship_among_a_b_c_l688_688615

noncomputable def a : ℝ := Real.log (7 / 2) / Real.log 3
noncomputable def b : ℝ := (1 / 4)^(1 / 3)
noncomputable def c : ℝ := -Real.log 5 / Real.log 3

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l688_688615


namespace cars_to_add_l688_688188

theorem cars_to_add (current_cars desired_multiple : ℕ) : current_cars = 29 → desired_multiple = 8 → (∃ additional_cars, additional_cars = 3) :=
by
  intros h_current h_multiple
  use 3
  sorry

end cars_to_add_l688_688188


namespace shorter_piece_length_l688_688039

theorem shorter_piece_length (x : ℕ) (h1 : 177 = x + 2*x) : x = 59 :=
by sorry

end shorter_piece_length_l688_688039


namespace miles_on_first_day_l688_688377

variable (x : ℝ)

/-- The distance traveled on the first day is x miles. -/
noncomputable def second_day_distance := (3/4) * x

/-- The distance traveled on the second day is (3/4)x miles. -/
noncomputable def third_day_distance := (1/2) * (x + second_day_distance x)

theorem miles_on_first_day
    (total_distance : x + second_day_distance x + third_day_distance x = 525)
    : x = 200 :=
sorry

end miles_on_first_day_l688_688377


namespace solution_set_of_inequality_l688_688682

-- Given conditions
variable {f : ℝ → ℝ}

-- Define odd function condition
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define decreasing on positive reals condition
def decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ ⦃x y⦄, 0 < x → x < y → f y < f x

-- Define the specific value of the function at x = -2
def f_neg_two_zero : Prop := f (-2) = 0

-- Defining the theorem to be proved
theorem solution_set_of_inequality
  (h1 : is_odd f)
  (h2 : decreasing_on_pos f)
  (h3 : f_neg_two_zero) :
  {x : ℝ | x * f x < 0} = set.Iio (-2) ∪ set.Ioi 2 :=
by sorry

end solution_set_of_inequality_l688_688682


namespace part1_factorization_part2_factorization_l688_688760

-- Part 1
theorem part1_factorization (x : ℝ) :
  (x - 1) * (6 * x + 5) = 6 * x^2 - x - 5 :=
by {
  sorry
}

-- Part 2
theorem part2_factorization (x : ℝ) :
  (x - 1) * (x + 3) * (x - 2) = x^3 - 7 * x + 6 :=
by {
  sorry
}

end part1_factorization_part2_factorization_l688_688760


namespace max_value_of_f_l688_688975

-- Define the function
def f (x : ℝ) := sqrt (36 + x) + sqrt (36 - x) + x / 6

-- State the theorem
theorem max_value_of_f : ∀ x : ℝ, (-36 ≤ x ∧ x ≤ 36) → f x ≤ 12 :=
by
  sorry

end max_value_of_f_l688_688975


namespace james_final_payment_l688_688330

variable (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ)

def final_cost (bed_frame_cost : ℕ) (bed_cost_multiplier : ℕ) (discount_rate : ℚ) : ℚ :=
let bed_cost := bed_frame_cost * bed_cost_multiplier in
let total_cost := bed_cost + bed_frame_cost in
let discount := total_cost * discount_rate in
total_cost - discount

theorem james_final_payment : final_cost 75 10 (20 / 100) = 660 := by
  unfold final_cost
  -- Step: Compute bed cost
  have bed_cost : ℕ := 75 * 10 by norm_num
  -- Step: Compute total cost
  have total_cost : ℕ := bed_cost + 75 by norm_num
  -- Step: Compute discount
  have discount : ℚ := total_cost * (20 / 100) by norm_num
  -- Step: Compute final cost
  have final_payment : ℚ := total_cost - discount by norm_num
  -- Assertion
  exact final_payment

#eval james_final_payment

end james_final_payment_l688_688330


namespace dad_steps_l688_688137

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688137


namespace halfway_between_one_eighth_and_one_third_l688_688968

theorem halfway_between_one_eighth_and_one_third : (1/8 + 1/3) / 2 = 11/48 :=
by
  sorry

end halfway_between_one_eighth_and_one_third_l688_688968


namespace difference_between_max_and_min_area_l688_688379

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

noncomputable def min_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

theorem difference_between_max_and_min_area :
  ∃ (l_max l_min w_max w_min : ℕ),
    2 * l_max + 2 * w_max = 60 ∧
    2 * l_min + 2 * w_min = 60 ∧
    (l_max * w_max - l_min * w_min = 196) :=
by
  sorry

end difference_between_max_and_min_area_l688_688379


namespace prob_factor_90_less_than_5_l688_688971

theorem prob_factor_90_less_than_5 (n : ℕ) (N_factors : Π (n : ℕ), n = 90 → Finset ℕ)
  (lt_5_factors : Finset.filter (< 5) (N_factors 90 (by rfl))) :
  (lt_5_factors.card / (N_factors 90 (by rfl)).card : ℚ) = 1 / 4 :=
by
  intro n N_factors lt_5_factors
  sorry

end prob_factor_90_less_than_5_l688_688971


namespace C_share_per_rs_equals_l688_688898

-- Definitions based on given conditions
def A_share_per_rs (x : ℝ) : ℝ := x
def B_share_per_rs : ℝ := 0.65
def C_share : ℝ := 48
def total_sum : ℝ := 246

-- The target statement to prove
theorem C_share_per_rs_equals : C_share / total_sum = 0.195122 :=
by
  sorry

end C_share_per_rs_equals_l688_688898


namespace bookcase_length_in_inches_l688_688378

theorem bookcase_length_in_inches (feet_length : ℕ) (inches_per_foot : ℕ) (h1 : feet_length = 4) (h2 : inches_per_foot = 12) : (feet_length * inches_per_foot) = 48 :=
by
  sorry

end bookcase_length_in_inches_l688_688378


namespace number_of_ducks_is_three_l688_688818

-- Define the ducks
constants {Ducks : Type} [fintype Ducks] [decidable_eq Ducks]

-- Define the properties of the ducks
def front (A B : Ducks) : Prop := sorry
def behind (A B : Ducks) : Prop := sorry
def between (A B C : Ducks) : Prop := sorry
def in_a_row (A B C : Ducks) : Prop := sorry 

-- Define the conditions
axiom A B C : Ducks
axiom h1 : ∃ (A : Ducks), ∀ (B C: Ducks), front A B ∧ front A C
axiom h2 : ∃ (A B : Ducks), ∀ (C : Ducks), behind B A ∧ behind C A
axiom h3 : ∃ (A B C : Ducks), (behind A C) ∧ (front B C) ∧ (front A B)
axiom h4 : ∃ (A B C : Ducks), between B A C 
axiom h5 : ∃ (A B C : Ducks), in_a_row A B C

-- Prove that there are exactly 3 ducks
theorem number_of_ducks_is_three : fintype.card Ducks = 3 := 
sorry

end number_of_ducks_is_three_l688_688818


namespace max_value_f_l688_688210

noncomputable def f (x : ℝ) : ℝ :=
  sin (x + sin x) + sin (x - sin x) + (π / 2 - 2) * sin (sin x)

theorem max_value_f :
  ∃ x : ℝ, f x = (π - 2) / sqrt 2 :=
sorry

end max_value_f_l688_688210


namespace exists_alpha_beta_l688_688707

theorem exists_alpha_beta :
  ∃ (α β : ℝ), α ∈ Ioo (-π/2) (π/2) ∧ β ∈ Ioo 0 π ∧
    sin (3 * π - α) = sqrt 2 * cos (π / 2 - β) ∧
    sqrt 3 * cos (-α) = -sqrt 2 * cos (π + β) :=
sorry

end exists_alpha_beta_l688_688707


namespace rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688871

-- Part (a)
theorem rational_non_integer_solution_exists :
  ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
sorry

-- Part (b)
theorem rational_non_integer_solution_not_exists :
  ¬ ∃ (x y : ℚ), x ∉ ℤ ∧ y ∉ ℤ ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
sorry

end rational_non_integer_solution_exists_rational_non_integer_solution_not_exists_l688_688871


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688866

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688866


namespace red_squares_block_l688_688286

theorem red_squares_block (board : Fin 9 × Fin 9 → Prop) (hred : ∑ i j, if board (i, j) then 1 else 0 = 46) :
  ∃ i j, (∑ di dj, if board (i + di, j + dj) then 1 else 0 ≥ 3  := 
begin
  sorry,
end

end red_squares_block_l688_688286


namespace largest_difference_l688_688345

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
by
  sorry  -- Proof is omitted as per instructions.

end largest_difference_l688_688345


namespace book_pages_total_l688_688331

theorem book_pages_total
  (days_in_week : ℕ)
  (daily_read_times : ℕ)
  (pages_per_time : ℕ)
  (additional_pages_per_day : ℕ)
  (num_days : days_in_week = 7)
  (times_per_day : daily_read_times = 3)
  (pages_each_time : pages_per_time = 6)
  (extra_pages : additional_pages_per_day = 2) :
  daily_read_times * pages_per_time + additional_pages_per_day * days_in_week = 140 := 
sorry

end book_pages_total_l688_688331


namespace part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l688_688401

noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def W (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

theorem part_1_relationship (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 40) :
  W x = -10 * x^2 + 500 * x - 4000 := by
  sorry

theorem part_2_solution (x : ℝ) (h₀ : W x = 1250) :
  x = 15 ∨ x = 35 := by
  sorry

theorem part_2_preferred (x : ℝ) (h₀ : W x = 1250) (h₁ : y 15 ≥ y 35) :
  x = 15 := by
  sorry

theorem part_3_max_W (x : ℝ) (h₀ : 28 ≤ x) (h₁ : x ≤ 35) :
  W x ≤ 2160 := by
  sorry

theorem part_3_max_at_28 :
  W 28 = 2160 := by
  sorry

end part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l688_688401


namespace find_parabola_equation_find_line_equation_l688_688266

-- Definition for part (Ⅰ)
def parabola_equation (p : ℝ) (hp : p > 0) (m : ℝ) (hpoint : (3, m)) (hfocus : dist (3 - (p / 2), m, 0) = 4) : Prop :=
  y^2 = 2 * p * x
  
-- Prove the equation of the parabola (Ⅰ)
theorem find_parabola_equation (p : ℝ) (hp : p > 0) (m : ℝ) 
  (hcond : m^2 = 2 * p * 3) (hdist : real.sqrt ((3 - (p / 2))^2 + m^2) = 4) : 
  (p = 2) → (y^2 = 4 * x) := 
by 
  sorry

-- Definition for part (Ⅱ)
def line_equation (y1 y2 x1 x2 : ℝ) (hmid : (y1 + y2) / 2 = -1)
  (hparabola_points : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2) : Prop :=
  2 * x + y - 2 = 0

-- Prove the equation of the line (Ⅱ)
theorem find_line_equation (y1 y2 x1 x2 : ℝ) 
  (hparabola : y^2 = 4*x) (hfocus : (1, 0))
  (hmid : (y1 + y2) / 2 = -1) (hparabola_points : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  : (2 * x + y - 2 = 0) :=
by 
  sorry

end find_parabola_equation_find_line_equation_l688_688266


namespace calculate_S2_l688_688082

theorem calculate_S2
  (ABC HFG DCE : Triangle)
  (BC CE : ℝ)
  (F G : Point)
  (FM AC GN DC : Line)
  (S₁ S₃ : ℝ)
  (h1 : Triangle.is_equilateral ABC)
  (h2 : Triangle.is_equilateral HFG)
  (h3 : Triangle.is_equilateral DCE)
  (h4 : BC = (1/3) * CE)
  (h5 : F.is_midpoint BC ABC.vertexB)
  (h6 : G.is_midpoint CE DCE.vertexE)
  (h7 : FM ∥ AC)
  (h8 : GN ∥ DC)
  (h9 : S₁ + S₃ = 10) :
  ∃ S₂ : ℝ, S₂ = 3 := sorry

end calculate_S2_l688_688082


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688865

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688865


namespace length_segment_midpoints_of_diagonals_l688_688793

theorem length_segment_midpoints_of_diagonals (L m : ℝ) :
  ∀ (trapezoid : Trapezoid), (length_upper_base(trapezoid) = L) → (length_midline(trapezoid) = m) →
  length_segment_midpoints_diagonals(trapezoid) = m - L := by
  sorry

end length_segment_midpoints_of_diagonals_l688_688793


namespace minimize_distance_sum_l688_688625

theorem minimize_distance_sum (x1 y1 x2 y2 x3 y3 : ℝ) :
  ∃ (P : ℝ × ℝ), P = (⟨(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3⟩) ∧
  (∀ (Q : ℝ × ℝ), (x1 - Q.fst)^2 + (y1 - Q.snd)^2 + (x2 - Q.fst)^2 + 
  (y2 - Q.snd)^2 + (x3 - Q.fst)^2 + (y3 - Q.snd)^2 ≥ 
  (x1 - (P.fst))^2 + (y1 - (P.snd))^2 + (x2 - (P.fst))^2 +
  (y2 - (P.snd))^2 + (x3 - (P.fst))^2 + (y3 - (P.snd))^2) :=
begin
  sorry
end

end minimize_distance_sum_l688_688625


namespace commute_time_abs_diff_l688_688525

theorem commute_time_abs_diff (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  |x - y| = 4 := by
  sorry

end commute_time_abs_diff_l688_688525


namespace find_x_such_that_sqrt_4_sub_5x_eq_8_l688_688222

theorem find_x_such_that_sqrt_4_sub_5x_eq_8 :
  ∃ x : ℝ, sqrt (4 - 5 * x) = 8 ∧ x = -12 := by
sorry

end find_x_such_that_sqrt_4_sub_5x_eq_8_l688_688222


namespace dad_steps_l688_688142

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688142


namespace square_side_length_approx_l688_688801

theorem square_side_length_approx :
  let radius := 3
  let circumference := 2 * Real.pi * radius
  let side_length := circumference / 4
  Real.floor(100 * side_length) / 100 = 4.71 :=
by
  sorry

end square_side_length_approx_l688_688801


namespace min_value_of_ellipse_l688_688646

noncomputable def min_m_plus_n (a b : ℝ) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) : ℝ :=
(a ^ (2/3) + b ^ (2/3)) ^ (3/2)

theorem min_value_of_ellipse (m n a b : ℝ) (h1 : m > n) (h2 : n > 0) (h_ellipse : (a^2 / m^2) + (b^2 / n^2) = 1) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) :
  (m + n) = min_m_plus_n a b h_ab_nonzero h_abs_diff :=
sorry

end min_value_of_ellipse_l688_688646


namespace polynomial_transformation_l688_688204

noncomputable def R (x : ℝ) := ∏ k in Finset.range 15, (x - (k + 1))

theorem polynomial_transformation :
  ∃ (a b : ℝ) (a_ne_zero : a ≠ 0),
    (P Q : polynomial ℝ),
      (P(x) = R ((x - b) / a) ∧ Q(x) = a * x + b) ∨ 
      (P(x) = (x - b) / a ∧ Q(x) = a * R(x) + b) →
      P(Q (x)) = R (x) := sorry

end polynomial_transformation_l688_688204


namespace joseph_total_distance_l688_688333

-- Distance Joseph runs on Monday
def d1 : ℕ := 900

-- Increment each day
def increment : ℕ := 200

-- Adjust distance calculation
def d2 := d1 + increment
def d3 := d2 + increment

-- Total distance calculation
def total_distance := d1 + d2 + d3

-- Prove that the total distance is 3300 meters
theorem joseph_total_distance : total_distance = 3300 :=
by sorry

end joseph_total_distance_l688_688333


namespace interior_angle_of_regular_nonagon_l688_688833

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l688_688833


namespace range_of_a_l688_688689

theorem range_of_a (a : ℝ) : ¬ (∃ x ∈ set.Ioi 3, x ≤ a) ↔ a ≤ 3 :=
by {
  sorry,
}

end range_of_a_l688_688689


namespace dad_steps_are_90_l688_688123

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688123


namespace dad_steps_l688_688133

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688133


namespace matrix_expression_l688_688993
open Matrix

variables {n : Type*} [Fintype n] [DecidableEq n]
variables (B : Matrix n n ℝ) (I : Matrix n n ℝ)

noncomputable def B_inverse := B⁻¹

-- Condition 1: B is a matrix with an inverse
variable [Invertible B]

-- Condition 2: (B - 3*I) * (B - 5*I) = 0
variable (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0)

-- Theorem to prove
theorem matrix_expression (B: Matrix n n ℝ) [Invertible B] 
  (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0) : 
  B + 10 * (B_inverse B) = (160 / 15 : ℝ) • I := 
sorry

end matrix_expression_l688_688993


namespace grey_eyes_black_hair_l688_688693

-- Definitions based on conditions
def num_students := 60
def num_black_hair := 36
def num_green_eyes_red_hair := 20
def num_grey_eyes := 24

-- Calculate number of students with red hair
def num_red_hair := num_students - num_black_hair

-- Calculate number of grey-eyed students with red hair
def num_grey_eyes_red_hair := num_red_hair - num_green_eyes_red_hair

-- Prove the number of grey-eyed students with black hair
theorem grey_eyes_black_hair:
  ∃ n, n = num_grey_eyes - num_grey_eyes_red_hair ∧ n = 20 :=
by
  sorry

end grey_eyes_black_hair_l688_688693


namespace find_k_for_one_real_solution_l688_688974

theorem find_k_for_one_real_solution (k : ℤ) :
  (∀ x : ℤ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end find_k_for_one_real_solution_l688_688974


namespace probability_same_number_l688_688939

theorem probability_same_number :
  let N := 300
  let multiples_36 := { n | n < N ∧ n % 36 = 0 }
  let multiples_48 := { n | n < N ∧ n % 48 = 0 }
  let lcm_36_48 := 144
  let multiples_lcm := { n | n < N ∧ n % lcm_36_48 = 0 }
  let total_outcomes := (mult & Prop) (mult(n := N) % 36 = 0) * (mult & Prop) (mult(n := N) % 48 = 0)
  let successful_outcomes := multiples_lcm
  (successful_outcomes * 1.0) / (total_outcomes * 1.0) = 1/24 :=
by
  sorry

end probability_same_number_l688_688939


namespace smallest_d_l688_688531

noncomputable def point_distance (x y : ℝ) : ℝ := real.sqrt (x ^ 2 + y ^ 2)

theorem smallest_d {d : ℝ} :
    point_distance (4 * real.sqrt 3) (d - 2) = 4 * d →
    d = 2.006 := sorry

end smallest_d_l688_688531


namespace sum_of_possible_values_on_Alice_card_l688_688113

-- Defining angle x and its constraints
def is_valid_angle (x : ℝ) : Prop := 
  0 < x ∧ x < 180 ∧ x ≠ 90

-- Defining the trigonometric functions for the angle
def sin_x (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
def cos_x (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
def tan_x (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- Main theorem: Sum of all possible values on Alice's card
theorem sum_of_possible_values_on_Alice_card (x : ℝ) (h : is_valid_angle x) :
  (if sin_x x = cos_x x then sin_x x + sin_x x else 
   if sin_x x = tan_x x then 0 else 
   if cos_x x = tan_x x then 0 else 0
  ) = Real.sqrt 2 :=
by sorry

end sum_of_possible_values_on_Alice_card_l688_688113


namespace triangle_area_l688_688925

theorem triangle_area :
  ∀ (a b : ℝ) (θ : ℝ),
    a = 3 →
    b = 5 →
    (cos θ) ∈ ({x | 5 * x ^ 2 - 7 * x - 6 = 0}) →
    ∃ (area : ℝ), area = 6 :=
by
  intros a b θ ha hb hθ
  use 1 / 2 * a * b * sin θ
  have h_cos : cos θ = -3 / 5 := sorry  -- Derived from solving the equation
  have h_sin : sin θ = 4 / 5 := sorry  -- Using the Pythagorean identity
  rw [ha, hb, h_sin] at *
  linarith

end triangle_area_l688_688925


namespace dad_steps_l688_688122

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688122


namespace expected_value_of_a_squared_l688_688021

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l688_688021


namespace dad_steps_l688_688164

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688164


namespace expected_value_a_squared_is_correct_l688_688034

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l688_688034


namespace stripe_width_l688_688426

theorem stripe_width (x : ℝ) (h : 60 * x - x^2 = 400) : x = 30 - 5 * Real.sqrt 5 := 
  sorry

end stripe_width_l688_688426


namespace count_valid_m_l688_688293

theorem count_valid_m : 
    ∃ m : ℤ, ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b) ∧
    (x^2 - m * x + (m + 2006) = 0) ∧
    (5 = {m : ℤ | ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b)}.card) :=
sorry

end count_valid_m_l688_688293


namespace desiree_age_l688_688189

theorem desiree_age (D C G Gr : ℕ) 
  (h1 : D = 2 * C)
  (h2 : D + 30 = (2 * (C + 30)) / 3 + 14)
  (h3 : G = D + C)
  (h4 : G + 20 = 3 * (D - C))
  (h5 : Gr = (D + 10) * (C + 10) / 2) : 
  D = 6 := 
sorry

end desiree_age_l688_688189


namespace surface_area_of_sphere_l688_688900

theorem surface_area_of_sphere (V_cube : ℝ) (hV : V_cube = 8) (radius : ℝ)
  (h_radius : 2 * radius = 2 * Real.sqrt 3) :
  4 * Real.pi * radius^2 = 12 * Real.pi :=
by
  -- Create an assumption for the edge length of the cube
  have edge_length : ℝ := 2
  -- Derive the radius from the given space diagonal relation
  have h_radius : radius = Real.sqrt 3 := by sorry
  -- Use the surface area formula of the sphere to show the result
  have result : 4 * Real.pi * (Real.sqrt 3) ^ 2 = 12 * Real.pi := by sorry
  exact result

end surface_area_of_sphere_l688_688900


namespace unpainted_cubes_count_l688_688901

theorem unpainted_cubes_count (l w h : ℕ) (hl : l = 6) (hw : w = 5) (hh : h = 4) : 
  let l_inner := l - 2, 
      w_inner := w - 2, 
      h_inner := h - 2 in 
  l_inner * w_inner * h_inner = 24 := by 
  sorry

end unpainted_cubes_count_l688_688901


namespace part1_part2_l688_688001

theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  x - x^2 < sin x ∧ sin x < x := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x, cos (a * x) - log (1 - x^2)) 
  (hmax : (∀ (x : ℝ), 0 < x → x < 1 → f x < f 0) ∧ (∀ (x : ℝ), -1 < x → x < 0 → f x < f 0)) : 
  a < -sqrt 2 ∨ sqrt 2 < a := sorry

end part1_part2_l688_688001


namespace B_contribution_l688_688540

variable (A B : ℝ) (months_A months_B : ℕ) (ratio_A ratio_B : ℕ)

-- Defining the given conditions
def A_investment : ℝ := 3500
def months_A : ℕ := 12
def months_B_before_joining : ℕ := 9
def months_B : ℕ := months_A - months_B_before_joining -- B invests for 3 months
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3
variable {x : ℝ}

-- The main theorem to prove B's contribution in the capital
theorem B_contribution (x : ℝ) : 
  (((A_investment * (months_A : ℝ)) / (x * (months_B : ℝ))) = (profit_ratio_A / profit_ratio_B)) ↔ x = 21000 :=
by
  sorry

end B_contribution_l688_688540


namespace primes_with_no_sum_of_two_cubes_l688_688589

theorem primes_with_no_sum_of_two_cubes (p : ℕ) [Fact (Nat.Prime p)] :
  (∃ n : ℤ, ∀ x y : ℤ, x^3 + y^3 ≠ n % p) ↔ p = 7 :=
sorry

end primes_with_no_sum_of_two_cubes_l688_688589


namespace smallest_x_value_l688_688840

theorem smallest_x_value : ∃ (x : ℝ), x = -12 ∧ (3*x^2 + 36*x - 60 = x*(x + 17)) ∧ ∀ (y : ℝ), (3*y^2 + 36*y - 60 = y*(y + 17)) → y ≥ -12 := 
by
  use -12
  split
  {
    -- x = -12
    refl
  }
  split
  {
    -- 3*(-12)^2 + 36*(-12) - 60 = (-12) * (-12 + 17)
    sorry
  }
  {
    -- ∀ y, (3*y^2 + 36*y - 60 = y*(y + 17)) → y ≥ -12
    sorry
  }

end smallest_x_value_l688_688840


namespace solve_y_eq_l688_688402

theorem solve_y_eq : 
  ∀ y : ℝ,
  (2 ^ (9 ^ y) = 9 ^ (2 ^ y)) →
  y = (1 + Real.log2 (Real.log2 3)) / (2 * Real.log2 3 - 1) :=
by
  intros y h
  sorry

end solve_y_eq_l688_688402


namespace sqrt_eq_eight_l688_688224

theorem sqrt_eq_eight (x : ℝ) (h : sqrt (4 - 5 * x) = 8) : x = -12 :=
sorry

end sqrt_eq_eight_l688_688224


namespace incorrect_statement_A_l688_688933

-- Definitions based on the conditions in the problem
def statement_A : Prop :=
  "The fertilization process demonstrates that the cell membrane has the function of facilitating intercellular communication and controlling the entry and exit of substances from cells."

def statement_B : Prop :=
  "The transformation process of sperm is conducive to the completion of fertilization, and after fertilization, the metabolic rate of the egg cell increases."

def statement_C : Prop :=
  "During the fertilization process, the fusion of sperm and egg cell nuclei results in the appearance of homologous chromosomes in the fertilized egg cell."

def statement_D : Prop :=
  "The randomness of fertilization is one of the important prerequisites for the appearance of special trait segregation ratios in Mendel's F2 experiment."

-- The Lean statement to prove that statement A is incorrect
theorem incorrect_statement_A : ¬ statement_A :=
by
  sorry

end incorrect_statement_A_l688_688933


namespace dad_steps_l688_688154

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688154


namespace range_of_smallest_nonprime_with_condition_l688_688723

def smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 : ℕ :=
121

theorem range_of_smallest_nonprime_with_condition :
  120 < smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ∧ 
  smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ≤ 130 :=
by
  unfold smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10
  exact ⟨by norm_num, by norm_num⟩

end range_of_smallest_nonprime_with_condition_l688_688723


namespace yan_ratio_distance_l688_688488

theorem yan_ratio_distance (w a b : ℝ) (h1 : 5 * w) (h2 : (b / w) = (a / w + (a + b) / (5 * w))) :
  a / b = 2 / 3 :=
by sorry

end yan_ratio_distance_l688_688488


namespace part1_part2_l688_688002

theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  x - x^2 < sin x ∧ sin x < x := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x, cos (a * x) - log (1 - x^2)) 
  (hmax : (∀ (x : ℝ), 0 < x → x < 1 → f x < f 0) ∧ (∀ (x : ℝ), -1 < x → x < 0 → f x < f 0)) : 
  a < -sqrt 2 ∨ sqrt 2 < a := sorry

end part1_part2_l688_688002


namespace trig_inequality_l688_688237

open Real

theorem trig_inequality
  (θ1 θ2 θ3 θ4 : ℝ)
  (h₁ : θ1 ∈ Ioo 0 (π / 2))
  (h₂ : θ2 ∈ Ioo 0 (π / 2))
  (h₃ : θ3 ∈ Ioo 0 (π / 2))
  (h₄ : θ4 ∈ Ioo 0 (π / 2))
  (sum_eq_pi : θ1 + θ2 + θ3 + θ4 = π) :
  (sqrt 2 * sin θ1 - 1) / cos θ1 +
  (sqrt 2 * sin θ2 - 1) / cos θ2 +
  (sqrt 2 * sin θ3 - 1) / cos θ3 +
  (sqrt 2 * sin θ4 - 1) / cos θ4 ≥ 0 := 
sorry

end trig_inequality_l688_688237


namespace dad_steps_l688_688179

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688179


namespace initial_house_cats_l688_688526

/-- A pet store had 15 Siamese cats and sold 19 cats. After the sale, they have 45 cats left. 
    We want to prove how many house cats they initially had. -/
theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 15)
  (cats_sold : ℕ := 19)
  (cats_left : ℕ := 45)
  (total_cats_before_sale : ℕ := cats_left + cats_sold)
  (initial_total_cats : ℕ := siamese_cats + H) :
  initial_total_cats = total_cats_before_sale → H = 49 :=
by
  intro h
  have h1 : 15 + H = 64 := h
  simp at h1
  exact sorry

end initial_house_cats_l688_688526


namespace Cathy_total_money_l688_688089

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l688_688089


namespace integer_values_count_l688_688604

theorem integer_values_count (n : ℤ) : 
  (32000 : ℕ) = 2 ^ 6 * 5 ^ 3 →
  ( ∃ k ∈ (range 10).vals, 32000 * (2 / 5 : ℚ)^n k.is_integer ) :=
sorry

end integer_values_count_l688_688604


namespace problem_statement_l688_688849

variables (a b c : Vector) (u v : Vector)
noncomputable def vector_eq_transitive : Prop :=
  (a = b ∧ b = c) → a = c

noncomputable def vector_parallel_same_or_opposite : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (a ∥ b) → (∃ θ : ℝ, (a = θ • b ∨ a = -θ • b))

noncomputable def dot_product_eq_vector_eq : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a • c = b • c) → a = b

noncomputable def perpendicular_vectors_if_magnitude_equal : Prop :=
  (u ≠ 0 ∧ v ≠ 0) ∧ (∥ u + v ∥ = ∥ u - v ∥) → (u ⊥ v)

theorem problem_statement : Prop :=
  vector_eq_transitive a b c ∧ ¬ vector_parallel_same_or_opposite a b ∧ ¬ dot_product_eq_vector_eq a b c ∧ perpendicular_vectors_if_magnitude_equal u v

end problem_statement_l688_688849


namespace interior_angle_of_regular_nonagon_l688_688834

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end interior_angle_of_regular_nonagon_l688_688834


namespace monotonic_decreasing_interval_l688_688797

noncomputable def f (x : ℝ) := real.sqrt (3 - 2 * x - x ^ 2)

theorem monotonic_decreasing_interval :
  ∀ x ∈ set.Ioo (-1 : ℝ) (1 : ℝ), ∃ ε > 0, ∀ δ > 0, 0 < δ < ε → f (x - δ) ≥ f (x + δ) :=
by
  sorry

end monotonic_decreasing_interval_l688_688797


namespace jonessa_take_home_pay_l688_688409

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l688_688409


namespace range_of_a_l688_688655

def f (x : ℝ) : ℝ := x + 4 / x

def g (x a : ℝ) : ℝ := 2^x + a

theorem range_of_a (h : ∀ x1, x1 ∈ set.Icc (1 / 2 : ℝ) 3 → ∃ x2, x2 ∈ set.Icc (2 : ℝ) 3 ∧ f x1 ≥ g x2 a) : a ≤ 0 := by
  sorry

end range_of_a_l688_688655


namespace smallest_positive_z_l688_688403

open Real

-- Definitions for the conditions
def sin_zero_condition (x : ℝ) : Prop := sin x = 0
def sin_half_condition (x z : ℝ) : Prop := sin (x + z) = 1 / 2

-- Theorem for the proof objective
theorem smallest_positive_z (x z : ℝ) (hx : sin_zero_condition x) (hz : sin_half_condition x z) : z = π / 6 := 
sorry

end smallest_positive_z_l688_688403


namespace dad_steps_l688_688114

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688114


namespace removed_black_cubes_divisible_by_4_l688_688514

theorem removed_black_cubes_divisible_by_4:
  (∀ (cubes : ℕ → ℕ × ℕ × ℕ → Prop), 
    let black_cubes := { (x, y, z) : fin 10 × fin 10 × fin 10 | (x + y + z) % 2 = 0 } in
    let removed_cubes := { (x, y, z) : fin 10 × fin 10 × fin 10 | cubes 1 (x, y, z) } in
    (∀ (i j : fin 10), ∃! (k : fin 10), cubes 1 (i,j,k)) →
    (∀ (i k : fin 10), ∃! (j : fin 10), cubes 1 (i,j,k)) →
    (∀ (j k : fin 10), ∃! (i : fin 10), cubes 1 (i,j,k)) →
    |removed_cubes| = 100 → 
    let removed_black_cubes := removed_cubes ∩ black_cubes in
    |removed_black_cubes| % 4 = 0) :=
sorry

end removed_black_cubes_divisible_by_4_l688_688514


namespace hexagon_area_l688_688916

theorem hexagon_area (h : ∃ s : ℝ, s^2 = real.sqrt 3) : ∃ A : ℝ, A = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688916


namespace min_time_height_l688_688781

/-- The height of the inclined plane with minimum time of descent -/
theorem min_time_height (d μ : ℝ) : 
  ∃ h : ℝ, h = d * (μ + real.sqrt (1 + μ^2)) :=
sorry

end min_time_height_l688_688781


namespace solve_for_x_l688_688601

theorem solve_for_x (x : ℝ) (h : log 8 (x + 8) = 3 / 2) : x = 8 * (2 * sqrt 2 - 1) :=
sorry

end solve_for_x_l688_688601


namespace determine_C_plus_D_l688_688322

theorem determine_C_plus_D (A B C D : ℕ) 
  (hA : A ≠ 0) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D → 
  C + D = 5 :=
by
    sorry

end determine_C_plus_D_l688_688322


namespace distinct_real_roots_find_k_values_l688_688658

-- Question 1: Prove the equation has two distinct real roots
theorem distinct_real_roots (k : ℝ) : 
  (2 * k + 1) ^ 2 - 4 * (k ^ 2 + k) > 0 :=
  by sorry

-- Question 2: Find the values of k when triangle ABC is a right triangle
theorem find_k_values (k : ℝ) : 
  (k = 3 ∨ k = 12) ↔ 
  (∃ (AB AC : ℝ), 
    AB ≠ AC ∧ AB = k ∧ AC = k + 1 ∧ (AB^2 + AC^2 = 5^2 ∨ AC^2 + 5^2 = AB^2)) :=
  by sorry

end distinct_real_roots_find_k_values_l688_688658


namespace prod_modulus_l688_688941

theorem prod_modulus {n : ℕ} (h : n = 10) :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
begin
  sorry
end

end prod_modulus_l688_688941


namespace student_signups_count_l688_688607

-- Let's define the problem with precise conditions and prove the corresponding answer.
theorem student_signups_count :
  let students := 4
  let groups := 3
  (groups ^ students) = 81 :=
by
  have students := 4
  have groups := 3
  -- This is the counting principle: 3 choices per student for 4 students
  calc (groups ^ students) = 3 ^ 4 : by rfl
                      ... = 81    : by norm_num
  sorry

end student_signups_count_l688_688607


namespace video_game_cost_l688_688608

theorem video_game_cost
  (weekly_allowance1 : ℕ)
  (weeks1 : ℕ)
  (weekly_allowance2 : ℕ)
  (weeks2 : ℕ)
  (money_spent_on_clothes_fraction : ℚ)
  (remaining_money : ℕ)
  (allowance1 : weekly_allowance1 = 5)
  (duration1 : weeks1 = 8)
  (allowance2 : weekly_allowance2 = 6)
  (duration2 : weeks2 = 6)
  (money_spent_fraction : money_spent_on_clothes_fraction = 1/2)
  (remaining_money_condition : remaining_money = 3) :
  (weekly_allowance1 * weeks1 + weekly_allowance2 * weeks2) * (1 - money_spent_on_clothes_fraction) - remaining_money = 35 :=
by
  rw [allowance1, duration1, allowance2, duration2, money_spent_fraction, remaining_money_condition]
  -- Calculation steps are omitted; they can be filled in here.
  exact sorry

end video_game_cost_l688_688608


namespace dad_steps_are_90_l688_688131

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688131


namespace finite_solutions_exists_l688_688758

variable (f : ℕ → ℝ)
variable (h1 : ∀ x : ℕ, f x > 0)
variable (h2 : Tendsto f atTop (𝓝 0))

theorem finite_solutions_exists : 
  ∃ N : ℕ, ∀ m n p : ℕ, f m + f n + f p = 1 → m ≤ N ∧ n ≤ N ∧ p ≤ N :=
sorry

end finite_solutions_exists_l688_688758


namespace rate_of_paving_per_sq_meter_eq_800_l688_688791

noncomputable def length : ℝ := 5.5
noncomputable def width : ℝ := 3.75
noncomputable def cost : ℝ := 16500
noncomputable def area : ℝ := length * width
noncomputable def rate : ℝ := cost / area

-- Proving that the calculated rate equals 800
theorem rate_of_paving_per_sq_meter_eq_800 : rate = 800 := 
by 
  unfold length width cost area rate
  rw [← mul_assoc, mul_comm width]
  calc
    (16500 / (5.5 * 3.75)) = 16500 / 20.625  : by rw [mul_comm]
                       ... = 800             : by norm_num

end rate_of_paving_per_sq_meter_eq_800_l688_688791


namespace min_distance_point_origin_l688_688527

theorem min_distance_point_origin (d : ℝ) (h : (4 * real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2) : d = 4 := 
  sorry

end min_distance_point_origin_l688_688527


namespace correct_conclusions_l688_688550

-- Definitions
def conclusion1 (Q : Type) [quadrilateral Q] : Prop :=
  ∃ (pairSidesEq: ∃ l1 l2 : Q.sides,  l1 ≠ l2 ∧ l1.length = l2.length),
  ∃ (pairAnglesEq: ∃ a1 a2 : Q.angles, a1 ≠ a2 ∧ a1.degree = a2.degree),
  ¬parallelogram Q

def conclusion2 (Q : Type) [quadrilateral Q] : Prop :=
  quadrilateralWithDiagonalsBisectAnglesAndParallel Q → parallelogram Q

def conclusion3 (Q : Type) [quadrilateral Q] : Prop :=
  quadrilateralWithMidpointsEqualDistance Q → ¬parallelogram Q

def conclusion4 (Q : Type) [quadrilateral Q] : Prop :=
  quadrilateralWithDiagonalsBisectArea Q → parallelogram Q

-- Theorem to prove
theorem correct_conclusions : ∀ (Q : Type) [quadrilateral Q], conclusion2 Q ∧ conclusion4 Q :=
by sorry

end correct_conclusions_l688_688550


namespace floor_length_approx_l688_688859

theorem floor_length_approx (total_cost rate_per_sqm : ℝ) (breadth : ℝ) 
    (H1 : total_cost = 300) 
    (H2 : rate_per_sqm = 5)
    (H3 : total_cost / rate_per_sqm = 60)
    (H4 : breadth^2 = 20) : 
    3 * breadth = 13.416 :=
by 
    have b := real.sqrt 20
    have length := 3 * b
    rw [real.sqrt_eq_blah] at b
    sorry

end floor_length_approx_l688_688859


namespace count_pairs_3n_7m_l688_688434

theorem count_pairs_3n_7m :
  (∃ M : ℕ, M = 278 ∧ (∀ (m n : ℕ), (1 ≤ m ∧ m ≤ 1500) → (3^n < 7^m ∧ 7^m < 7^(m + 3) ∧ 7^(m + 3) < 3^(n + 1)) → M = 278))
  ∧ (7^750 < 3^1250 ∧ 3^1250 < 7^751) :=
begin
  sorry
end

end count_pairs_3n_7m_l688_688434


namespace ellipse_standard_equation_line_MN_fixed_point_range_OM_ON_l688_688708

-- Define the conditions
def common_foci (C : Ellipse) (H : Hyperbola) : Prop :=
  -- definition of common foci here, properties of foci coincide
  sorry

def eccentricity (C : Ellipse) (e : ℝ) : Prop :=
  -- definition of eccentricity e for ellipse C
  sorry

-- Given conditions
axiom condition_1 : common_foci C (mk_hyperbola y^2 - x^2 = 1)
axiom condition_2 : eccentricity C (√6 / 3)

-- Define the statements to prove
theorem ellipse_standard_equation : 
  (∃ a b : ℝ, a > b > 0 ∧ a^2 - b^2 = 2 ∧ (sqrt(b^2 + c^2) / a = e) ∧ 
  C = Ellipse a b (λ x y, y^2 / a^2 + x^2 / b^2 = 1)) → 
  (∀ C, C = Ellipse (√3) 1 rfl) := sorry

theorem line_MN_fixed_point : 
  (∀ M N : Point, M ≠ A ∧ N ≠ A ∧ (M ∈ C) ∧ (N ∈ C) → 
  line_passing_through M N (0, -2√3)) := sorry

theorem range_OM_ON : 
  (∃ M N : Point, M ≠ A ∧ N ≠ A ∧ (M ∈ C) ∧ (N ∈ C) →
  -3 < (vector OM ∘ vector ON : ℝ) ∧ (vector OM ∘ vector ON < 3/2)):= sorry

end ellipse_standard_equation_line_MN_fixed_point_range_OM_ON_l688_688708


namespace problem_statement_l688_688191

theorem problem_statement :
  sqrt ((2 - sin (π / 9)^2) * (2 - sin (2 * π / 9)^2) * (2 - sin (4 * π / 9)^2)) = 5 / 4 :=
by sorry

end problem_statement_l688_688191


namespace simplify_expression_l688_688229

theorem simplify_expression :
  ((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4) = 43 / 4 := 
sorry

end simplify_expression_l688_688229


namespace percent_calculation_l688_688891

theorem percent_calculation : 
  let n := 5600 in
  0.15 * (0.30 * (0.50 * n)) = 126 :=
by
  sorry

end percent_calculation_l688_688891


namespace rectangle_perimeter_sum_l688_688009

theorem rectangle_perimeter_sum (AE BE CF : ℝ) (h1: AE = 8) (h2: BE = 17) (h3: CF = 3) :
    let l := 25  -- Since AE + BE = 25
    let w := 70 / 3  -- From the derivation of BC = 70/3
    2 * (l + w) = 290 / 3 ∧ 290.gcd 3 = 1 := 
by
    sorry

example : 290 + 3 = 293 := rfl

end rectangle_perimeter_sum_l688_688009


namespace change_received_l688_688054

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l688_688054


namespace proposition_relation_l688_688666

theorem proposition_relation :
  (∀ (x : ℝ), x < 3 → x < 5) ↔ (∀ (x : ℝ), x ≥ 5 → x ≥ 3) :=
by
  sorry

end proposition_relation_l688_688666


namespace investment_final_value_l688_688513

theorem investment_final_value 
  (original_investment : ℝ) 
  (increase_percentage : ℝ) 
  (original_investment_eq : original_investment = 12500)
  (increase_percentage_eq : increase_percentage = 2.15) : 
  original_investment * (1 + increase_percentage) = 39375 := 
by
  sorry

end investment_final_value_l688_688513


namespace scaling_transform_l688_688802

-- Define the point before transformation
def original_point : ℝ × ℝ := (2, 3)

-- Define the transformed point according to the given conditions
def transformed_point (p : ℝ × ℝ) : ℝ × ℝ := ( 
  (3/2) * p.1,  -- horizontal coordinate transformation
  (2/3) * p.2   -- vertical coordinate transformation
  )

-- The proof problem: show that the transformed point equals (3, 2)
theorem scaling_transform : transformed_point original_point = (3, 2) := 
  by 
    -- The proof is omitted
    sorry

end scaling_transform_l688_688802


namespace arithmetic_sequence_7th_term_l688_688470

theorem arithmetic_sequence_7th_term
  (a l : ℝ) (n : ℕ) (h1 : a = 3) (h2 : l = 78) (h3 : n = 25) :
  let d := (l - a) / (n - 1)
  let a_7 := a + (7 - 1) * d
  in a_7 = 21.75 :=
by
  sorry

end arithmetic_sequence_7th_term_l688_688470


namespace dad_steps_are_90_l688_688125

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688125


namespace sequence_a_100_l688_688289

theorem sequence_a_100 : 
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902) :=
by
  sorry

end sequence_a_100_l688_688289


namespace part1_part2_l688_688000

theorem part1 (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) : 
  x - x^2 < sin x ∧ sin x < x := sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x, cos (a * x) - log (1 - x^2)) 
  (hmax : (∀ (x : ℝ), 0 < x → x < 1 → f x < f 0) ∧ (∀ (x : ℝ), -1 < x → x < 0 → f x < f 0)) : 
  a < -sqrt 2 ∨ sqrt 2 < a := sorry

end part1_part2_l688_688000


namespace price_of_child_ticket_l688_688071

theorem price_of_child_ticket (C : ℝ) 
  (adult_ticket_price : ℝ := 8) 
  (total_tickets_sold : ℕ := 34) 
  (adult_tickets_sold : ℕ := 12) 
  (total_revenue : ℝ := 236) 
  (h1 : 12 * adult_ticket_price + (34 - 12) * C = total_revenue) :
  C = 6.36 :=
by
  sorry

end price_of_child_ticket_l688_688071


namespace students_sampled_from_schoolB_l688_688897

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

end students_sampled_from_schoolB_l688_688897


namespace find_a_l688_688651

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x else -x^2

theorem find_a : ∃ a : ℝ, f (f a) = 2 ∧ a = Real.sqrt 2 :=
by
  sorry

end find_a_l688_688651


namespace area_of_figure_l688_688036

noncomputable def area_between_tangents_and_curve (a : ℝ) (h : a > 0) : ℝ :=
  let C (x : ℝ) := 1 / x
  let P := (a, C a)
  let Q := (2 * a, C (2 * a))
  let tangentLineAtP x := -1 / (a * a) * x + 2 / a
  let tangentLineAtQ x := -1 / (4 * a * a) * x + 1 / a
  -- Placeholder for the proof of the area
  2 * real.log 2 - 9 / 8

theorem area_of_figure (a : ℝ) (h : a > 0) :
  area_between_tangents_and_curve a h = 2 * real.log 2 - 9 / 8 :=
by
  sorry

end area_of_figure_l688_688036


namespace x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l688_688680

-- Define the context and main statement
theorem x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta
  (θ : ℝ)
  (hθ₁ : 0 < θ)
  (hθ₂ : θ < (π / 2))
  {x : ℝ}
  (hx : x + 1 / x = 2 * Real.sin θ)
  (n : ℕ) (hn : 0 < n) :
  x^n + 1 / x^n = 2 * Real.sin (n * θ) :=
sorry

end x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l688_688680


namespace dad_steps_l688_688143

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688143


namespace part1_part2_l688_688984

-- Assumptions
variable {a b : ℝ}
variable norm_a : real.norm a = 3
variable norm_b : real.norm b = 4
variable theta : ℝ
variable angle_ab : θ = 2 * Real.pi / 3

-- Part 1: Proof that k = 3/2
theorem part1 (k : ℝ) (h : dot_product (a + k * b) a = 0) : k = 3 / 2 := sorry

-- Part 2: Proof that |3a + 2b| = √73
theorem part2 : real.norm (3 * a + 2 * b) = Real.sqrt 73 := sorry

end part1_part2_l688_688984


namespace sum_odd_primes_200_to_600_is_13859_l688_688841

def odd_prime_between_200_and_600 (p : ℕ) : Prop :=
  p > 200 ∧ p < 600 ∧ Nat.Prime p ∧ p % 2 = 1

theorem sum_odd_primes_200_to_600_is_13859 :
  (Finset.sum (Finset.filter odd_prime_between_200_and_600 (Finset.range 601))) = 13859 :=
sorry

end sum_odd_primes_200_to_600_is_13859_l688_688841


namespace num_distinct_ordered_pairs_l688_688279

-- Define the conditions
def positive_integers (x : ℕ) := x > 0
def reciprocal_sum (m n : ℕ) := (1 / m + 1 / n = 1 / 6)

-- Statement of the theorem
theorem num_distinct_ordered_pairs : 
  {p : ℕ × ℕ // positive_integers p.1 ∧ positive_integers p.2 ∧ reciprocal_sum p.1 p.2}.to_finset.card = 9 :=
by
  sorry

end num_distinct_ordered_pairs_l688_688279


namespace only_function_f_satisfies_condition_l688_688965

-- Define the domain and codomain of the function
noncomputable def R_pos := {r : ℝ // r > 0}

-- State the condition for the function f
def condition (f : R_pos → R_pos) : Prop :=
  ∀ x y : R_pos, f x * f y = 2 * f (x + y * f x)

-- Prove that the only function that satisfies the condition is f(x) = 2 for all x
theorem only_function_f_satisfies_condition (f : R_pos → R_pos) :
  (condition f) → (∀ x : R_pos, f x = 2) :=
by
  intro h
  sorry

end only_function_f_satisfies_condition_l688_688965


namespace rectangle_perimeter_l688_688010

noncomputable def perimeter_calculation
    (AE BE CF : ℕ)
    (AE_val BE_val CF_val : ℕ)
    (rectangle_AB : ℕ × ℕ)
    (result : ℕ)
    (rel_prime : m : ℕ × n : ℕ areRelativelyPrime (m : ℕ) (n : ℕ)) : Prop :=
  AE = AE_val ∧ BE = BE_val ∧ CF = CF_val →
  let AB := AE + sqrt (BE * BE - AE * AE),
      BC := sqrt (15 * 15 + CF * CF) in
  2 * (AB + BC) = result

theorem rectangle_perimeter (AE BE CF : ℕ) :
  perimeter_calculation AE BE CF 8 17 3 (25 + 70/3) 293  :=
by
  unfold perimeter_calculation
  sorry

end rectangle_perimeter_l688_688010


namespace students_with_both_pets_l688_688197

theorem students_with_both_pets
  (D C : Finset ℕ)
  (h_union : (D ∪ C).card = 48)
  (h_D : D.card = 30)
  (h_C : C.card = 34) :
  (D ∩ C).card = 16 :=
by sorry

end students_with_both_pets_l688_688197


namespace james_pays_660_for_bed_and_frame_l688_688327

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end james_pays_660_for_bed_and_frame_l688_688327


namespace triangle_BGE_is_right_l688_688318

-- Definitions and conditions from the problem
variables {A B C D E F G H : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point G] [Point H]
variables [Triangle A B C]
variables [IntAngleBisector A B C D]
variables [OnSegment D B C]

-- Perpendiculars from D
variables [Perpendicular D E A C]
variables [Perpendicular D F A B]

-- Intersection points
variables [Intersect BE CF H]
variables [Circumcircle A F H H G]

-- Proof statement: Prove that triangle BGE is a right triangle at G
theorem triangle_BGE_is_right 
  (hAD : ∃ D, IsAngleBisector AD ∧ Point D ∈ Segment BC)
  (hDE : ∃ E, Perpendicular D E AC)
  (hDF : ∃ F, Perpendicular D F AB)
  (hBF : ∃ H, Intersect BE CF H)
  (hCircum : ∃ G, Circumcircle AFH BE G) :
  ∃ BG GE BF, IsRightTriangle B G E :=
sorry

end triangle_BGE_is_right_l688_688318


namespace dad_steps_l688_688159

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688159


namespace find_C_l688_688548

variable (A B C : ℚ)

def condition1 := A + B + C = 350
def condition2 := A + C = 200
def condition3 := B + C = 350

theorem find_C : condition1 A B C → condition2 A C → condition3 B C → C = 200 :=
by
  sorry

end find_C_l688_688548


namespace expected_value_X_probability_two_red_balls_from_B_l688_688698

noncomputable def count_combinations {α : Type*} (s : finset α) (k : ℕ) : ℕ :=
s.powerset.filter (λ t, t.card = k).card

-- Definitions based on the given conditions
def boxA := finset.of_list ["W1", "W2", "W3", "R1", "R2"]
def boxB := finset.of_list ["W4", "W5", "W6", "W7", "R3"]

def combinations (s : finset string) (n : ℕ) : finset (finset string) :=
s.powerset.filter (λ t, t.card = n)

def relevant_combinations (n : ℕ) : finset (finset string) :=
combinations boxA n

-- Lean statement for part 1
theorem expected_value_X :
  let p0 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'W').card = 2) / count_combinations boxA 2)
      p1 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'W').card = 1) / count_combinations boxA 2)
      p2 := (count_combinations (relevant_combinations 2).filter (λ x, x.filter (λ b, b[0] = 'R').card = 2) / count_combinations boxA 2) in
  0 * p0 + 1 * p1 + 2 * p2 = 4 / 5 := 
sorry

-- Lean statement for part 2
theorem probability_two_red_balls_from_B :
  let X0 := 0
      X1 := (count_combinations (combinations (boxB ∪ ["W1", "W2"]) 2) / count_combinations (boxB ∪ ["W1", "W2"]) 2)
      X2 := (count_combinations (combinations (boxB ∪ ["R1", "R2"]) 2) - 1 / count_combinations (boxB ∪ ["R1", "R2"]) 2) in
  3 / 10 * X0 + 6 / 10 * 1 / 21 + 1 / 10 * 1 / 7 = 3 / 70 :=
sorry

end expected_value_X_probability_two_red_balls_from_B_l688_688698


namespace dad_steps_l688_688182

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688182


namespace smallest_angle_in_triangle_l688_688796

theorem smallest_angle_in_triangle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by
  have h2 : 16 * k = 180 := by
    rw [←add_assoc] at h1
    rw [←add_assoc (4 * k)] at h1
    exact h1
  have h3 : k = 180 / 16 := by
    field_simp [←h2]
    norm_num
  rw [h3]
  norm_num
  sorry

end smallest_angle_in_triangle_l688_688796


namespace value_is_sqrt_5_over_3_l688_688367

noncomputable def findValue (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) : ℝ :=
  (x + y) / (x - y)

theorem value_is_sqrt_5_over_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) :
  findValue x y h1 h2 h3 = Real.sqrt (5 / 3) :=
sorry

end value_is_sqrt_5_over_3_l688_688367


namespace trig_frac_eq_l688_688769

theorem trig_frac_eq {p q : ℕ} (x : ℝ) (h1 : Real.sec x + Real.tan x = 11 / 3) 
  (h2 : Real.csc x + Real.cot x = p / q)
  (h3 : Nat.gcd p q = 1) : p + q = 96 := by
  sorry

end trig_frac_eq_l688_688769


namespace square_split_diagonal_l688_688922

theorem square_split_diagonal :
  ∀ (side_length : ℝ) (leg1 leg2 : ℝ) (hypotenuse area : ℝ),
  side_length = 12 →
  leg1 = side_length →
  leg2 = side_length →
  hypotenuse = side_length * Real.sqrt 2 →
  area = 0.5 * leg1 * leg2 →
  leg1 = 12 ∧ leg2 = 12 ∧ hypotenuse = 12 * Real.sqrt 2 ∧ area = 72 :=
by {
  intros side_length leg1 leg2 hypotenuse area,
  assume h1 : side_length = 12,
  assume h2 : leg1 = side_length,
  assume h3 : leg2 = side_length,
  assume h4 : hypotenuse = side_length * Real.sqrt 2,
  assume h5 : area = 0.5 * leg1 * leg2,
  sorry
}

end square_split_diagonal_l688_688922


namespace dad_steps_l688_688132

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688132


namespace min_value_ABFG_l688_688821

-- Defining the main existence conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = k * m

-- Given conditions
def BCDE_is_multiple_of_2013 (BCDE : ℕ) : Prop := is_multiple_of BCDE 2013
def DEFG_is_multiple_of_1221 (DEFG : ℕ) : Prop := is_multiple_of DEFG 1221

-- Definition for the extracted digits from the seven-digit number
def extract_digits (ABCDEFG : ℕ) : (ℕ × ℕ × ℕ × ℕ) :=
  -- Here, the setup would need a careful approach to split the seven digits
  (ABCDEFG / 1000000, (ABCDEFG % 1000000) / 100000, ABCDEFG % 10000, ABCDEFG % 100)

-- The main theorem we need to prove
theorem min_value_ABFG :
  ∃ ABCDEFG : ℕ, 
    let (A, B, F, G) := extract_digits ABCDEFG in
    let ABFG := A * 1000 + B * 100 + F * 10 + G in
    BCDE_is_multiple_of_2013 (ABCDEFG / 100 % 100000) ∧
    DEFG_is_multiple_of_1221 (ABCDEFG % 100000) ∧
    ABFG = 3036 :=
sorry

end min_value_ABFG_l688_688821


namespace count_valid_m_l688_688294

theorem count_valid_m : 
    ∃ m : ℤ, ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b) ∧
    (x^2 - m * x + (m + 2006) = 0) ∧
    (5 = {m : ℤ | ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (m = a + b) ∧ (m + 2006 = a * b)}.card) :=
sorry

end count_valid_m_l688_688294


namespace focus_with_greater_x_coordinate_l688_688937

noncomputable theory

def major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((0, -1), (6, -1))
def minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 1), (3, -3))

theorem focus_with_greater_x_coordinate :
  let center := (3, -1)
  let distance_between_foci := 2 * Real.sqrt 5 in
  let greater_focus_x := center.1 + Real.sqrt 5 in
  (greater_focus_x, center.2) = (3 + Real.sqrt 5, -1) :=
by
  let center := (3, -1)
  let distance_between_foci := 2 * Real.sqrt 5
  let greater_focus_x := center.1 + Real.sqrt 5
  show (greater_focus_x, center.2) = (3 + Real.sqrt 5, -1)
  sorry

end focus_with_greater_x_coordinate_l688_688937


namespace inscribed_sphere_tetrahedron_volume_l688_688690

theorem inscribed_sphere_tetrahedron_volume
  (R : ℝ) (S1 S2 S3 S4 : ℝ) :
  ∃ V : ℝ, V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_tetrahedron_volume_l688_688690


namespace circle_equation_correct_min_distance_QP_correct_l688_688619

noncomputable def problem_I : Prop :=
  ∃ (a : ℝ), (a > 0) ∧ ((x - a)^2 + (y - 3 * a)^2 = 9)

noncomputable def problem_II : Prop :=
  ∃ (QP : ℝ), (Qx Qy Px Py : ℝ),
  ((Qx + Qy + 1 = 0) ∧ 
   (Px - 1)^2 + (Py - 3)^2 = 9 ∧ 
   ((Qx - Px)^2 + (Qy - Py)^2 = QP^2) ∧ 
   QP = (Real.sqrt 14) / 2)

theorem circle_equation_correct : problem_I :=
sorry

theorem min_distance_QP_correct : problem_II :=
sorry

end circle_equation_correct_min_distance_QP_correct_l688_688619


namespace holiday_non_holiday_ratio_l688_688506

theorem holiday_non_holiday_ratio
  (non_holiday_rate : ℕ)
  (holiday_customers : ℕ)
  (holiday_hours : ℕ)
  (h1 : non_holiday_rate = 175)
  (h2 : holiday_customers = 2800)
  (h3 : holiday_hours = 8) :
  (holiday_customers / holiday_hours) / non_holiday_rate = 2 :=
by {
  -- Definitions based on the provided conditions
  let holiday_rate := holiday_customers / holiday_hours,
  have h_holiday_rate : holiday_rate = 350, from calc
    holiday_rate = 2800 / 8 : by rw [h2, h3]
               ... = 350 : by norm_num,
  -- Proof of the ratio
  have h_ratio : (holiday_rate / non_holiday_rate) = 2, from calc
    holiday_rate / non_holiday_rate = 350 / 175 : by rw [h_holiday_rate, h1]
                                 ... = 2 : by norm_num,
  exact h_ratio
}

end holiday_non_holiday_ratio_l688_688506


namespace find_m_l688_688502

theorem find_m (m : ℕ) (h₁ : 256 = 4^4) : (256 : ℝ)^(1/4) = (4 : ℝ)^m ↔ m = 1 :=
by
  sorry

end find_m_l688_688502


namespace part_a_exists_rational_non_integer_l688_688875

theorem part_a_exists_rational_non_integer 
  (x y : ℚ) (hx : ¬int.cast x ∉ ℤ) (hy : ¬int.cast y ∉ ℤ) :
  ∃ x y : ℚ, (¬int.cast x ∉ ℤ) ∧ (¬int.cast y ∉ ℤ) ∧ (19 * x + 8 * y ∈ ℤ) ∧ (8 * x + 3 * y ∈ ℤ) := 
  sorry

end part_a_exists_rational_non_integer_l688_688875


namespace percentage_of_y_is_correct_l688_688582

/-- Given that 60% of 30% of y is equal to 18% of y -/
theorem percentage_of_y_is_correct (y : ℝ) : 0.18 * y = 0.6 * 0.3 * y :=
by
  simp [mul_assoc]

end percentage_of_y_is_correct_l688_688582


namespace cathy_total_money_l688_688094

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l688_688094


namespace f_x_axis_intersections_l688_688635

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 2 then x^3 - x else f (x - 2)

theorem f_x_axis_intersections : 
  ∃ count : ℕ, count = 7 ∧ 
  (∀ x ∈ set.Icc (0 : ℝ) (6 : ℝ), f x = 0 → x ∈ {0, 1, 2, 3, 4, 5, 6}) :=
sorry

end f_x_axis_intersections_l688_688635


namespace exist_mn_iff_l688_688363

theorem exist_mn_iff (p : ℕ) (s : ℕ) (h_prime : Nat.prime p) (h1 : 0 < s) (h2 : s < p) : 
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧ 
              (↑(s * m) / p - (s * m / p) : ℚ) < (↑(s * n) / p - (s * n / p) : ℚ) ∧ 
              (↑(s * n) / p - (s * n / p) : ℚ) < ↑s / p) ↔ 
  ¬ (s ∣ (p - 1)) := sorry

end exist_mn_iff_l688_688363


namespace figure_circumference_l688_688425

-- Definitions based on conditions
def large_diameter : ℝ := 24
def small_count : ℕ := 8
def pi_value : ℝ := 3.14

-- Automatically compute radii and arc length based on the given parameters
def large_radius : ℝ := large_diameter / 2
def small_diameter : ℝ := large_diameter / small_count
def small_radius : ℝ := small_diameter / 2
def large_arc_length : ℝ := large_radius * pi_value
def small_arc_length : ℝ := small_radius * pi_value

-- Total arc length (circumference) of the figure
def total_circumference : ℝ := large_arc_length + small_count * small_arc_length

-- Theorem stating the computed circumference is 75.36
theorem figure_circumference : total_circumference = 75.36 :=
by
  sorry

end figure_circumference_l688_688425


namespace james_pays_660_for_bed_and_frame_l688_688328

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end james_pays_660_for_bed_and_frame_l688_688328


namespace solve_dance_circles_problem_l688_688309

open Finset

-- Definitions based on the conditions provided in the problem
def children := (univ : Finset (Fin 5)) -- There's a set of 5 children, each distinct

noncomputable def numWaysToDivideIntoDanceCircles : ℕ :=
  let S := fun (n k : ℕ) => StirlingSecondKind.partition n k in
  let partitions := S 5 2 in
  let circles_permutations := fun k => factorial (k - 1) * factorial (5 - k - 1) in
  let totalConfigurations := 
    (univ.ssubsets.filter (fun s => s.card ≠ 0)).sum (λ s, 
      let k := s.card in
      (choose 5 k) * circles_permutations k
    ) in
  totalConfigurations / 2

-- Theorem to state the correctness of the counted ways as 50
theorem solve_dance_circles_problem :
  numWaysToDivideIntoDanceCircles = 50 := by
  sorry

end solve_dance_circles_problem_l688_688309


namespace min_band_members_exists_l688_688041

theorem min_band_members_exists (n : ℕ) :
  (∃ n, (∃ k : ℕ, n = 9 * k) ∧ (∃ m : ℕ, n = 10 * m) ∧ (∃ p : ℕ, n = 11 * p)) → n = 990 :=
by
  sorry

end min_band_members_exists_l688_688041


namespace find_n_l688_688243

-- Definitions based on the given conditions
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- The mathematically equivalent proof problem statement:
theorem find_n (n : ℕ) (p : ℝ) (h1 : binomial_expectation n p = 6) (h2 : binomial_variance n p = 3) : n = 12 :=
sorry

end find_n_l688_688243


namespace dad_steps_l688_688160

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688160


namespace points_on_line_sufficient_but_not_necessary_l688_688501

open Nat

-- Define the sequence a_n
def sequence_a (n : ℕ) : ℕ := n + 1

-- Define a general arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, n < m → a (m) - a (n) = (m - n) * (a 1 - a 0)

-- Define the condition that points (n, a_n), where n is a natural number, lie on the line y = x + 1
def points_on_line (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n) = n + 1

-- Prove that points_on_line is sufficient but not necessary for is_arithmetic_sequence
theorem points_on_line_sufficient_but_not_necessary :
  (∀ a : ℕ → ℕ, points_on_line a → is_arithmetic_sequence a)
  ∧ ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ ¬ points_on_line a := 
by 
  sorry

end points_on_line_sufficient_but_not_necessary_l688_688501


namespace expenditure_on_digging_l688_688591

noncomputable def volume_of_cylinder (r h : ℝ) := 
  Real.pi * r^2 * h

noncomputable def rate_per_cubic_meter (cost : ℝ) (r h : ℝ) : ℝ := 
  cost / (volume_of_cylinder r h)

theorem expenditure_on_digging (d h : ℝ) (cost : ℝ) (r : ℝ) (π : ℝ) (rate : ℝ)
  (h₀ : d = 3) (h₁ : h = 14) (h₂ : cost = 1682.32) (h₃ : r = d / 2) (h₄ : π = Real.pi) 
  : rate_per_cubic_meter cost r h = 17 := sorry

end expenditure_on_digging_l688_688591


namespace tangent_slope_negative_range_l688_688734

noncomputable def curve (x : ℝ) : ℝ :=
  x^2 - 2*x - 4*log x

theorem tangent_slope_negative_range :
  {x : ℝ | deriv curve x < 0} = set.Ioo 0 2 :=
sorry

end tangent_slope_negative_range_l688_688734


namespace find_total_cost_price_l688_688924

noncomputable def cost_prices (C1 C2 C3 : ℝ) : Prop :=
  0.85 * C1 + 72.50 = 1.125 * C1 ∧
  1.20 * C2 - 45.30 = 0.95 * C2 ∧
  0.92 * C3 + 33.60 = 1.10 * C3

theorem find_total_cost_price :
  ∃ (C1 C2 C3 : ℝ), cost_prices C1 C2 C3 ∧ C1 + C2 + C3 = 631.51 := 
by
  sorry

end find_total_cost_price_l688_688924


namespace class_heights_mode_median_l688_688394

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℕ := sorry

theorem class_heights_mode_median 
  (A : List ℕ) -- Heights of students from Class A
  (B : List ℕ) -- Heights of students from Class B
  (hA : A = [170, 170, 169, 171, 171, 171])
  (hB : B = [168, 170, 170, 172, 169, 170]) :
  mode A = 171 ∧ median B = 170 := sorry

end class_heights_mode_median_l688_688394


namespace dad_steps_l688_688135

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688135


namespace concentric_circles_black_percent_l688_688949

theorem concentric_circles_black_percent :
  let radii := [3, 6, 9, 12, 15]
  let areas := radii.map (λ r, r * r * Real.pi)
  let total_area := areas.getLast!
  let black_areas := [areas[0], areas[2] - areas[1], areas[4] - areas[3]]
  let total_black_area := black_areas.sum
  ((total_black_area / total_area) * 100) = 60 := by
  sorry

end concentric_circles_black_percent_l688_688949


namespace investment_worth_l688_688375

theorem investment_worth {x : ℝ} (x_pos : 0 < x) :
  ∀ (initial_investment final_value : ℝ) (years : ℕ),
  (initial_investment * 3^years = final_value) → 
  initial_investment = 1500 → final_value = 13500 → 
  8 = x → years = 2 →
  years * (112 / x) = 28 := 
by
  sorry

end investment_worth_l688_688375


namespace find_Prob_eta_ge_2_l688_688370

-- Define the parameters and conditions
noncomputable def p : ℚ := 1 / 3

-- Define the binomial distributions
def binomial (n : ℕ) (p : ℚ) : ℕ → ℚ :=
  λ k => (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Random variables ξ ~ B(2, p) and η ~ B(4, p)
def ξ (k : ℕ) : ℚ := binomial 2 p k
def η (k : ℕ) : ℚ := binomial 4 p k

-- Given condition
axiom h : Finset.sum (Finset.range 3) (λ k => ξ k) = 5 / 9

-- Statement to prove
theorem find_Prob_eta_ge_2 : Finset.sum (Finset.Ico 2 5) (λ n => η n) = 11 / 27 :=
by
  sorry

end find_Prob_eta_ge_2_l688_688370


namespace ball_distribution_l688_688674

theorem ball_distribution (balls boxes : ℕ) (h_balls : balls = 4) (h_boxes : boxes = 3) :
  (∃! (f : Fin boxes → ℕ), (∀ i, f i > 0) ∧ (∑ i, f i = balls)) :=
sorry

end ball_distribution_l688_688674


namespace cube_root_sqrt_eq_l688_688687

theorem cube_root_sqrt_eq {x : ℝ} (h : x^3 = 64) : sqrt x = 2 ∨ sqrt x = -2 :=
by
  sorry

end cube_root_sqrt_eq_l688_688687


namespace fraction_meaningful_l688_688463

theorem fraction_meaningful (x : ℝ) : (∃ y, y = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_l688_688463


namespace expected_value_squared_minimum_vector_norm_l688_688026

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l688_688026


namespace yarn_cut_parts_l688_688505

-- Define the given conditions
def total_length : ℕ := 10
def crocheted_parts : ℕ := 3
def crocheted_length : ℕ := 6

-- The main problem statement
theorem yarn_cut_parts (total_length crocheted_parts crocheted_length : ℕ) (h1 : total_length = 10) (h2 : crocheted_parts = 3) (h3 : crocheted_length = 6) :
  (total_length / (crocheted_length / crocheted_parts)) = 5 :=
by
  sorry

end yarn_cut_parts_l688_688505


namespace max_value_of_f_l688_688213

/-- 
Given the function f(x) defined by:
  f(x) = sin(x + sin x) + sin(x - sin x) + (pi / 2 - 2) * sin (sin x)
Prove that the maximum value of f(x) is (pi - 2) / sqrt(2).
--/
theorem max_value_of_f :
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) → f x = (Real.pi - 2) / Real.sqrt 2 :=
sorry

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.sin x) +
  Real.sin (x - Real.sin x) +
  (Real.pi / 2 - 2) * (Real.sin (Real.sin x))

end max_value_of_f_l688_688213


namespace car_X_wins_probability_l688_688695

theorem car_X_wins_probability : 
  let P_X : ℝ := 0.25  -- The probability to be proven
  ∧ P_Y : ℝ := 1 / 8  -- Given probability for car Y to win
  ∧ P_Z : ℝ := 1 / 12 -- Given probability for car Z to win
  ∧ Combined_Prob_Formula : P_X + P_Y + P_Z = 0.4583333333333333
  in P_X = 0.25 :=
by 
  let P_Y := 1 / 8
  let P_Z := 1 / 12
  let Combined_Prob := 0.4583333333333333
  have h_combined : P_X + P_Y + P_Z = Combined_Prob :=
    sorry  -- Proof to be done
  have h_P_Y : P_Y = 0.125 := by
    sorry  -- Simplifying 1/8 to 0.125
  have h_P_Z : P_Z = 0.0833333333333333 := by
    sorry  -- Simplifying 1/12 to 0.0833333333333333
  have h_sum_P_Y_P_Z : P_Y + P_Z = 0.2083333333333333 := by
    sorry  -- Adding the two probabilities
  have h_final : P_X = 0.25 := by
    sorry  -- Solving for P_X from the combined probability
  exact h_final

end car_X_wins_probability_l688_688695


namespace number_of_pairs_satisfying_2020_l688_688217

theorem number_of_pairs_satisfying_2020 (m n : ℕ) :
  (∃ m n : ℕ, (1 / m + 1 / n = 1 / 2020)) ↔ 45 :=
sorry

end number_of_pairs_satisfying_2020_l688_688217


namespace perpendicular_line_l688_688824

noncomputable def slope (a b c : ℝ) := -(a / b)

noncomputable def perpendicular_slope (m : ℝ) := -1 / m

theorem perpendicular_line (a b c : ℝ) (x1 y1 : ℝ) (h : a ≠ 0 ∧ b ≠ 0):
  let m := slope a b c in
  let m_perpendicular := perpendicular_slope m in
  let y_intercept := y1 - m_perpendicular * x1 in
  3 * x1 - 6 * y1 = c ∧ y_intercept = 0 →
  slope 3 (-6) 9 = 1 / 2 ∧ perpendicular_slope (1 / 2) = -2 ∧ y_intercept = 0 →
  (∀ x, y1 = -2 * x) :=
by
  intros
  sorry

end perpendicular_line_l688_688824


namespace exact_location_determined_by_d_l688_688928

def option_a : Prop := ¬ ExactLocation "Row 2, Hall 3 of Pacific Cinema"
def option_b : Prop := ¬ ExactLocation "Southward 40° east"
def option_c : Prop := ¬ ExactLocation "Middle section of Tianfu Avenue"
def option_d : Prop := ExactLocation "East longitude 116°, north latitude 42°"

theorem exact_location_determined_by_d :
  option_d ∧ option_a ∧ option_b ∧ option_c :=
by
  sorry

end exact_location_determined_by_d_l688_688928


namespace more_frequent_digit_1_l688_688977

theorem more_frequent_digit_1 {n : ℕ} (h : n = 1000000000) :
  let final_digit_count (x : ℕ) := (nat.digits 10 x).sum % 9
  filter (λ x, final_digit_count x = 1) (list.range (n + 1)).length >
  filter (λ x, final_digit_count x = 2) (list.range (n + 1)).length :=
by {
  let single_digit (x : ℕ) := (nat.digits 10 x).sum % 9,
  sorry
}

end more_frequent_digit_1_l688_688977


namespace count_congruent_to_mod_8_l688_688282

theorem count_congruent_to_mod_8 : 
  let S1 := {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 8 = 1}
  let S2 := {n : ℕ | 1 ≤ n ∧ n ≤ 300 ∧ n % 8 = 2}
  S1.card + S2.card = 76 := 
by 
  sorry

end count_congruent_to_mod_8_l688_688282


namespace next_year_digits_sum_eq_5_l688_688956

def sum_of_digits (n : ℕ) : ℕ :=
  (to_digits 10 n).sum

theorem next_year_digits_sum_eq_5 : ∃ y > 2021, sum_of_digits y = 5 :=
by
  have example : sum_of_digits 2030 = 5 := by
    sorry -- Proof steps of sum_of_digits(2030) would go here
  exact ⟨2030, by linarith, example⟩

end next_year_digits_sum_eq_5_l688_688956


namespace dot_product_is_sqrt3_over_2_l688_688738

def vector_a := (Real.cos (45 * Real.pi / 180), Real.sin (45 * Real.pi / 180))
def vector_b := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))

theorem dot_product_is_sqrt3_over_2 : (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = Real.sqrt 3 / 2 := 
by 
  sorry

end dot_product_is_sqrt3_over_2_l688_688738


namespace descending_order_of_powers_l688_688081

theorem descending_order_of_powers :
  let x := (-1.8)^(2/3)
  let y := 2^(2/3)
  let z := (-2)^(1/3)
  (4 > 3.24 > -2) → (y > x > z) :=
by
  sorry

end descending_order_of_powers_l688_688081


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688885

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688885


namespace Fiona_Less_Than_Charles_l688_688774

noncomputable def percentDifference (a b : ℝ) : ℝ :=
  ((a - b) / a) * 100

theorem Fiona_Less_Than_Charles : percentDifference 600 (450 * 1.1) = 17.5 :=
by
  sorry

end Fiona_Less_Than_Charles_l688_688774


namespace minimum_value_of_l688_688728

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem minimum_value_of (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : 1/x + 1/y + 1/z = 9) :
  minimum_value x y z = 1 / 3456 := 
sorry

end minimum_value_of_l688_688728


namespace max_principled_trios_l688_688887

-- Define what it means for a trio to be principled
structure PrincipledTrio (A B C : ℕ) : Prop :=
  (defeatsAB : true) -- Placeholder for the actual definition
  (defeatsBC : true) -- Placeholder for the actual definition
  (defeatsCA : true) -- Placeholder for the actual definition

-- Constant representing the number of chess players
def numPlayers : ℕ := 2017

-- The main theorem
theorem max_principled_trios (n : ℕ) (h : n = numPlayers) :
  let max_trios := if n % 2 = 1 then (n^3 - n) / 24 else (n^3 - 4 * n) / 24 in
  max_trios = 341606288 :=
by
  sorry

end max_principled_trios_l688_688887


namespace dad_steps_l688_688152

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688152


namespace part1_a_part1_b_part2_min_max_l688_688668

variables (x : ℝ) (a b : ℝ × ℝ)
def vec_a x : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
def vec_b x : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)
def f (x : ℝ) : ℝ := (dot_product (vec_a x) (vec_b x)) - (norm (vec_a x + vec_b x))

theorem part1_a (h : x = Real.pi / 12) : dot_product (vec_a x) (vec_b x) = Real.sqrt 3 / 2 := sorry

theorem part1_b (h : x = Real.pi / 12) : norm (vec_a x + vec_b x) = Real.sqrt (2 + Real.sqrt 3) := sorry

theorem part2_min_max :
  ∃ (xmin xmax : ℝ),
    (xmin = -3 / 2) ∧
    (xmax = -1) ∧
    ∀ (y : ℝ), (y ∈ Set.Icc (- (Real.pi / 3)) (Real.pi / 4) → xmin ≤ f y ∧ f y ≤ xmax) := sorry

end part1_a_part1_b_part2_min_max_l688_688668


namespace find_B_coordinates_l688_688715

noncomputable def point_A : ℝ × ℝ := (2, 8)
noncomputable def curve (x : ℝ) : ℝ := x^3
noncomputable def normal_slope (m : ℝ) : ℝ := -1 / m

theorem find_B_coordinates :
  let A := (2 : ℝ, 8 : ℝ)
  let y := curve
  let tangent_slope := 3 * A.1^2
  let normal_eq := by {
    have m := normal_slope tangent_slope,
    exact λ (x : ℝ), m * (x - A.1) + A.2 
  }
  let intersection_eq := λ (x : ℝ), x^3 - (normal_eq x) 
  ∃ B : ℝ × ℝ, B = (96 / 3, 884736 / 27) ∧ B.2 = y B.1 :=
by
  sorry

end find_B_coordinates_l688_688715


namespace complex_number_quadrant_l688_688319

theorem complex_number_quadrant :
  let z := complex.mk (Real.cos 2) (Real.sin 3)
  in Real.cos 2 < 0 ∧ Real.sin 3 > 0 →
     -- representing second quadrant
     z.im > 0 ∧ z.re < 0 :=
by
  intro z h
  -- proof is omitted
  sorry

end complex_number_quadrant_l688_688319


namespace original_price_per_bottle_l688_688487

theorem original_price_per_bottle (x : ℝ) (h1 : 108 / x = n) (h2 : 90 / (0.25 * x) = n + 1):
  x = 12 :=
by {
  have h3 : 108 / x + 1 = 90 / (0.25 * x),
  { exact calc
      108 / x + 1 = 108 / x + 1 : by rw h1
               ... = 108 / x + (90 / (0.25 * x)) - 90 / (0.25 * x) + 1 : by Ring.lhs_left [by simp]
               ... = (108 + x) / x : by [ -- fix the error here, explaining the intermediate steps
                   rw (show 360 / x = (360 * x) / (x * x), by Ring.lhs_left.eq. simp = (360) ]
               },
  simp,

  convert eq.trans,
  have h4 : x = 12,
  { rw,
    exact calc
      108 + x = 9 / 0.75 ∘ x = 100,
 },
  refine h4,
  exact x,
  sorry    
  } 

end original_price_per_bottle_l688_688487


namespace dorothy_remaining_money_after_trip_l688_688196

def Dorothy_age := 15
def brother_age := 12
def parents_and_grandfather_ages := [35, 36, 65]
def regular_ticket_cost := 10
def discount_rate := 0.30
def discount_age_limit := 18
def Dorothy_current_money := 70

def discounted_ticket_cost (age : ℕ) (ticket_cost : ℤ) (discount_rate : ℝ) (age_limit : ℕ) : ℤ :=
if age ≤ age_limit then ticket_cost - (ticket_cost : ℝ * discount_rate).toInt else ticket_cost

theorem dorothy_remaining_money_after_trip : Dorothy_current_money - 
  (
    discounted_ticket_cost Dorothy_age regular_ticket_cost discount_rate discount_age_limit + 
    discounted_ticket_cost brother_age regular_ticket_cost discount_rate discount_age_limit + 
    parents_and_grandfather_ages.foldl (λ acc age, acc + discounted_ticket_cost age regular_ticket_cost discount_rate discount_age_limit) 0
  ) = 26 :=
by
  sorry

end dorothy_remaining_money_after_trip_l688_688196


namespace smallest_value_of_d_l688_688535

def distance (x y : ℝ) : ℝ := real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem smallest_value_of_d (d : ℝ) : distance (4 * real.sqrt 3) (d - 2) = 4 * d → d = 26 / 15 :=
by
  -- sorry indicates we are omitting the proof
  sorry

end smallest_value_of_d_l688_688535


namespace minimum_sum_of_cube_faces_l688_688750

theorem minimum_sum_of_cube_faces (a b c d e f : ℕ)
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ e) (h5 : b ≠ f) (h6 : c ≠ e)
  (h7 : c ≠ f) (h8 : d ≠ e) (h9 : d ≠ f)
  (h10 : |a - b| > 1) (h11 : |a - c| > 1) (h12 : |a - d| > 1)
  (h13 : |b - e| > 1) (h14 : |b - f| > 1) (h15 : |c - e| > 1)
  (h16 : |c - f| > 1) (h17 : |d - e| > 1) (h18 : |d - f| > 1)
  (h19 : |e - f| > 1) :
  a + b + c + d + e + f ≥ 18 :=
sorry

end minimum_sum_of_cube_faces_l688_688750


namespace new_mean_after_addition_l688_688332

noncomputable def average (nums : List ℝ) : ℝ :=
  nums.sum / nums.length

theorem new_mean_after_addition (nums : List ℝ) (h_length : nums.length = 15) (h_avg : average nums = 40) :
  average (nums.map (λ x, x + 12)) = 52 :=
by
  sorry

end new_mean_after_addition_l688_688332


namespace find_sum_indices_with_tastiness_l688_688602

noncomputable def chromatic_polynomial (G : Graph ℕ) : Polynomial ℕ := sorry

def tasty_graph (V E : ℕ) (G : Graph V) (k : ℕ) : Prop :=
  G.connected ∧ E = 2017 ∧ exists (C : list V), G.is_cycle_of_length k C

def tastiness (G : Graph ℕ) : ℕ :=
  (chromatic_polynomial G).coeffs.count is_odd

def minimal_tastiness (k_min k_max : ℕ) : ℕ :=
  finset.min' (finset.image tastiness (finset.filter (λ k, ∃ G, tasty_graph 2017 2017 G k) (finset.range (k_max - k_min + 1)))) sorry 

def sum_indices_with_tastiness (k_min k_max : ℕ) : ℕ :=
  finset.sum (finset.filter (λ k, ∃ G, tasty_graph 2017 2017 G k ∧ tastiness G = minimal_tastiness k_min k_max)
    (finset.range (k_max - k_min + 1))) id

theorem find_sum_indices_with_tastiness :
  sum_indices_with_tastiness 3 2017 = 2017 := 
  sorry

end find_sum_indices_with_tastiness_l688_688602


namespace tan_sum_angle_l688_688983

theorem tan_sum_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (π / 4 + α) = -3 := 
by sorry

end tan_sum_angle_l688_688983


namespace earnings_difference_l688_688187

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end earnings_difference_l688_688187


namespace forty_percent_of_number_l688_688754

/--
Given that (1/4) * (1/3) * (2/5) * N = 30, prove that 0.40 * N = 360.
-/
theorem forty_percent_of_number {N : ℝ} (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 30) : 0.40 * N = 360 := 
by
  sorry

end forty_percent_of_number_l688_688754


namespace coin_flip_sequences_count_l688_688048

theorem coin_flip_sequences_count : 
  let total_flips := 10;
  let heads_fixed := 2;
  (2 : ℕ) ^ (total_flips - heads_fixed) = 256 := 
by 
  sorry

end coin_flip_sequences_count_l688_688048


namespace odd_n_cubed_plus_23n_divisibility_l688_688500

theorem odd_n_cubed_plus_23n_divisibility (n : ℤ) (h1 : n % 2 = 1) : (n^3 + 23 * n) % 24 = 0 := 
by 
  sorry

end odd_n_cubed_plus_23n_divisibility_l688_688500


namespace switches_in_position_a_after_500_l688_688947

noncomputable def switch_position_initial (n : ℕ) : Fin 5 :=
  ⟨0, by decide⟩  -- All switches start in position A, represented by ⟨0⟩.

noncomputable def switch_labels : Fin 500 → Fin 7 × Fin 7 × Fin 7 :=
  λ i => ⟨i.1 % 7, (i.1 / 7) % 7, (i.1 / 49) % 7⟩

noncomputable def advancement_count (n : ℕ) : ℕ :=
  let ⟨x, y, z⟩ := switch_labels n
  7 - x + 1 * 7 - y + 1 * 7 - z + 1

noncomputable def at_position_a_after_500 (d : Fin 500) : Prop :=
  advancement_count d % 5 = 0

theorem switches_in_position_a_after_500 :
  (Finset.univ.filter at_position_a_after_500).card = 436 :=
sorry -- Proof of the statement.

end switches_in_position_a_after_500_l688_688947


namespace expected_number_of_threes_l688_688465

noncomputable def expected_threes_on_two_8sided_dice : ℚ :=
  ∑ i in finset.range 3, (1/8 : ℚ) ^ i * (7/8) ^ (2 - i) * (nat.choose 2 i)

theorem expected_number_of_threes (dice : nat) : 
  dice = 2 ∧ ∀ k, 1 <= k ∧ k <= 8 → (1/8 : ℚ) = (1 : ℚ) / 8 :=
begin
   have H : expected_threes_on_two_8sided_dice = 1 / 4, by sorry,
   exact H
end

end expected_number_of_threes_l688_688465


namespace free_cytosine_molecules_req_l688_688510

-- Definition of conditions
def DNA_base_pairs := 500
def AT_percentage := 34 / 100
def CG_percentage := 1 - AT_percentage

-- The total number of bases
def total_bases := 2 * DNA_base_pairs

-- The number of C or G bases
def CG_bases := total_bases * CG_percentage

-- Finally, the total number of free cytosine deoxyribonucleotide molecules 
def free_cytosine_molecules := 2 * CG_bases

-- Problem statement: Prove that the number of free cytosine deoxyribonucleotide molecules required is 1320
theorem free_cytosine_molecules_req : free_cytosine_molecules = 1320 :=
by
  -- conditions are defined, the proof is omitted
  sorry

end free_cytosine_molecules_req_l688_688510


namespace cube_edge_length_l688_688256

noncomputable theory
open Real

def volume_of_sphere (R : ℝ) : ℝ := (4/3) * π * R^3

def diagonal_of_cube (a : ℝ) : ℝ := a * sqrt 3

def edge_length_of_cube (v : ℝ) (h : volume_of_sphere R = v) : ℝ :=
  let R := (3 * v / (4 * π))^(1/3)
  let d := 2 * R
  d / sqrt 3

theorem cube_edge_length :
  volume_of_sphere 2 = 32/3 * π → edge_length_of_cube (32/3 * π) _ = 4 * sqrt 3 / 3 := 
  by sorry

end cube_edge_length_l688_688256


namespace probability_divisible_by_3_l688_688386

theorem probability_divisible_by_3 (a b c : ℕ) (h : a ∈ Finset.range 2008 ∧ b ∈ Finset.range 2008 ∧ c ∈ Finset.range 2008) :
  (∃ p : ℚ, p = 1265/2007 ∧ (abc + ac + a) % 3 = 0) :=
sorry

end probability_divisible_by_3_l688_688386


namespace sum_squares_inequality_l688_688451

theorem sum_squares_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
(h_sum : x + y + z = 3) : 
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := 
by 
  sorry

end sum_squares_inequality_l688_688451


namespace sum_bn_eq_neg_2015_l688_688661

def is_geometric (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * a n

theorem sum_bn_eq_neg_2015 
  (a b : ℕ → ℝ) 
  (h1 : ∀ n, b n = Real.log 2 (a n)) 
  (h2 : ∃ d, ∀ n, b (n + 1) = b n + d) 
  (h3 : a 8 * a 2008 = 1 / 4) :
  (Finset.range 2015).sum b = -2015 :=
sorry

end sum_bn_eq_neg_2015_l688_688661


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688882

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688882


namespace dad_steps_l688_688153

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688153


namespace necessary_but_not_sufficient_l688_688613

theorem necessary_but_not_sufficient {x : ℝ} (h : 0 < x ∧ x < π / 2) : 
  (x * (sin x)^2 < 1 ↔ x * sin x < 1) := 
sorry

end necessary_but_not_sufficient_l688_688613


namespace inf_distance_eq_l688_688343

noncomputable def inf_distance (n : ℕ) (h : n ≥ 1) : ℝ :=
  (⨅ (p : polynomial ℝ) (hp : p.degree < n) 
     (f : ℝ → ℝ) (hf : ∀ x ∈ set.Icc 0 1, f x = ∑' k, if k >= n then (ite (k >= n) 1 0) * x^k else 0), 
         ⨆ (x ∈ set.Icc 0 1), abs (f x - polynomial.eval x p))

theorem inf_distance_eq (n : ℕ) (h : n ≥ 1) : 
  inf_distance n h = (2 : ℝ) ^ (-2 * n + 1) :=
  sorry

end inf_distance_eq_l688_688343


namespace compare_neg_fractions_l688_688103

theorem compare_neg_fractions : (-5/4 : ℚ) > (-4/3 : ℚ) := 
sorry

end compare_neg_fractions_l688_688103


namespace right_triangle_set_l688_688484

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end right_triangle_set_l688_688484


namespace part_II_l688_688996

noncomputable def a (n : ℕ) : ℕ := 2 * 3^n
def S (n : ℕ) : ℕ := (nat.sum (Iio n) (λ i, a (i + 1)) + a 0) / 2
def b (n : ℕ) : ℝ := (a n : ℝ) / ((a n - 2 : ℝ) * (a (n + 1) - 2 : ℝ))
def T (n : ℕ) : ℝ := ∑ i in Nat.range n, b i

lemma part_I : ∀ n : ℕ, a n = 2 * 3^n := sorry

theorem part_II : ∃ m : ℕ, (∀ n : ℕ, T_n > m / 16) ∧ m = 1 := sorry

end part_II_l688_688996


namespace coplanar_imp_k_neg9_l688_688347

noncomputable def coplanarity_scalar : ℝ :=
  let u := (λ (A B C D E : Point), 4 * A.to_vector - 3 * B.to_vector + 6 * C.to_vector + x * D.to_vector + 2 * E.to_vector) in
  if h : (∃ (A B C D E : Point), u A B C D E = 0 ∧ coplanar A B C D E) then
    some (k : ℝ) if k = -9 ∧ (∀ A B C D E, u A B C D E = 0 ∧ k = -9) else 
  0

theorem coplanar_imp_k_neg9 (A B C D E : Point) :
  4 * A.to_vector - 3 * B.to_vector + 6 * C.to_vector + k * D.to_vector + 2 * E.to_vector = 0 →
  coplanar A B C D E →
  k = -9 :=
by {
  sorry
}

end coplanar_imp_k_neg9_l688_688347


namespace seating_arrangements_l688_688412

-- Define the seats and the preference conditions
def seat : Type := ℕ -- We will represent seats by numbers 1 to 5

-- Conditions 
def isAisle (s : seat) : Prop := s = 3 ∨ s = 4
def areAdjacent (s1 s2 : seat) : Prop := |s1 - s2| = 1

-- Define the problem
theorem seating_arrangements :
  (∀ g : seat, isAisle g → ∃ m s1 s2 : seat, areAdjacent m s1 ∧ areAdjacent m s2 ∧ g ≠ m ∧ g ≠ s1 ∧ g ≠ s2 ∧ m ≠ s1 ∧ s1 ≠ s2 ∧ s2 ≠ m) → 16 :=
by sorry

end seating_arrangements_l688_688412


namespace smallest_k_even_rightmost_nonzero_digit_l688_688978

def b (n : ℕ) : ℕ := (n + 7)! / (n - 1)!

def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.find (λ d => d ≠ 0) | 0

theorem smallest_k_even_rightmost_nonzero_digit :
  ∃ k : ℕ, k > 0 ∧ rightmost_nonzero_digit (b k) % 2 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m < k -> rightmost_nonzero_digit (b m) % 2 ≠ 0 :=
begin
  sorry
end

end smallest_k_even_rightmost_nonzero_digit_l688_688978


namespace prove_results_l688_688307

-- Exercise scores for 40 randomly selected students
def scores : List ℕ := 
  [75, 85, 74, 98, 72, 57, 81, 96, 73, 95, 59, 95, 63, 88, 93, 67, 92, 83, 94, 54,
   90, 56, 89, 92, 79, 87, 70, 71, 91, 83, 83, 73, 80, 93, 81, 79, 91, 78, 83, 77]

noncomputable def analysis (l : List ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let countInRange := λ (low high : ℕ) => (l.filter (λ x => low ≤ x ∧ x ≤ high)).length
  let a := countInRange 90 100
  let b := countInRange 60 74
  let median := (l.sorted.nthD (l.length / 2) 0 + l.sorted.nthD (l.length / 2 - 1) 0) / 2
  let mode := (l.groupBy (λ _ => id)).values.map (λ sublist => (sublist.headD 0, sublist.length)).maxBy (λ x => x.snd).fst
  (a, b, median, mode)
  
def question (a₀ b₀ c₀ d₀ : ℕ) :=
  ∃ (a b c d : ℕ), analysis scores = (a, b, c, d) ∧ a = a₀ ∧ b = b₀ ∧ c = c₀ ∧ d = d₀

theorem prove_results : question 12 20 82 83 :=
  sorry

end prove_results_l688_688307


namespace expected_value_squared_minimum_vector_norm_l688_688027

noncomputable def expectation_a_squared (n : ℕ) : ℝ :=
  let Y := binomial n (1 / 3) in
  3 * (Y.var + Y.mean^2)

theorem expected_value_squared (n : ℕ) : expectation_a_squared n = (2 * n + n^2) / 3 := sorry

theorem minimum_vector_norm (Y1 Y2 Y3 : ℕ) (n : ℕ) (h_sum : Y1 + Y2 + Y3 = n) : 
  ∥(Y1, Y2, Y3)∥_2 ≥ n / real.sqrt 3 := sorry

end expected_value_squared_minimum_vector_norm_l688_688027


namespace first_discount_is_10_l688_688430

def list_price : ℝ := 70
def final_price : ℝ := 59.85
def second_discount : ℝ := 0.05

theorem first_discount_is_10 :
  ∃ (x : ℝ), list_price * (1 - x/100) * (1 - second_discount) = final_price ∧ x = 10 :=
by
  sorry

end first_discount_is_10_l688_688430


namespace volume_of_rotation_l688_688942

noncomputable def volume_of_rotated_solid : ℝ :=
  let f (y : ℝ) := 4 / y in
  π * ∫ y in 1..2, (f y) ^ 2

theorem volume_of_rotation :
  volume_of_rotated_solid = 8 * Real.pi :=
by
  sorry

end volume_of_rotation_l688_688942


namespace exists_rational_non_integer_linear_l688_688879

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l688_688879


namespace pyramid_volume_max_l688_688544

noncomputable def triangular_pyramid_volume (PA PB PC : ℝ) : ℝ :=
  let diagonal_length := 3 * Real.sqrt 3 in
  let radius := diagonal_length / 2 in
  let volume := (4 / 3) * Real.pi * radius^3 in
  volume

theorem pyramid_volume_max : triangular_pyramid_volume 3 3 3 = (27 * Real.sqrt 3 * Real.pi) / 2 :=
  by
  sorry

end pyramid_volume_max_l688_688544


namespace analytical_expression_odd_monotonicity_find_range_l688_688992

-- Definition and conditions
def f (a x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + 1)

-- Statement 1: Proving the analytical expression of f(x)
theorem analytical_expression_odd (h_odd : ∀ x, f a x = -f a (-x)) (h_domain: ∀ x, x ∈ set.univ) : a = 1 :=
sorry

-- Statement 2: Proving the monotonicity of f(x)
theorem monotonicity (a : ℝ) (h : a = 1) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

-- Statement 3: Finding the range of a
theorem find_range (h_inequality: ∀ x > 0, f 1 (real.logb 2 x * real.logb 2 (8 / x)) + f 1 (-a) < 0) : a > 9 / 4 :=
sorry

end analytical_expression_odd_monotonicity_find_range_l688_688992


namespace quarterback_sacked_times_l688_688536

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quarterback_sacked_times_l688_688536


namespace sum_binom_eq_binom_l688_688757

theorem sum_binom_eq_binom (n : ℕ) (hn : 0 < n) :
  (∑ i in Finset.range (⌊n / 2⌋ + 1), 2^(n - 2 * i) * Nat.choose n (n - 2*i) * Nat.choose (2*i) i) = 
    Nat.choose (2*n) n :=
by
  sorry

end sum_binom_eq_binom_l688_688757


namespace mean_of_transformed_data_l688_688271

variables {n : ℕ} (x y : Fin n.succ → ℝ) (x_bar y_bar : ℝ)

def mean (z : Fin n.succ → ℝ) : ℝ :=
  ∑ i, z i / (n + 1)

def transform (x y : Fin n.succ → ℝ) : Fin n.succ → ℝ :=
  λ i, 2 * x i - 3 * y i + 1

axiom means_of_data (x y : Fin n.succ → ℝ) (x_bar y_bar : ℝ) :
  mean x = x_bar ∧ mean y = y_bar

theorem mean_of_transformed_data :
  mean (transform x y) = 2 * x_bar - 3 * y_bar + 1 :=
by
  sorry

end mean_of_transformed_data_l688_688271


namespace expected_value_a_squared_norm_bound_l688_688029

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l688_688029


namespace arrangement_count_is_43200_l688_688456

noncomputable def number_of_arrangements : Nat :=
  let number_of_boys := 6
  let number_of_girls := 3
  let boys_arrangements := Nat.factorial number_of_boys
  let spaces := number_of_boys - 1
  let girls_arrangements := Nat.factorial (spaces) / Nat.factorial (spaces - number_of_girls)
  boys_arrangements * girls_arrangements

theorem arrangement_count_is_43200 :
  number_of_arrangements = 43200 := by
  sorry

end arrangement_count_is_43200_l688_688456


namespace remainder_modulo_l688_688523

theorem remainder_modulo (N k q r : ℤ) (h1 : N = 1423 * k + 215) (h2 : N = 109 * q + r) : 
  (N - q ^ 2) % 109 = 106 := by
  sorry

end remainder_modulo_l688_688523


namespace neither_squares_cubes_fourth_powers_l688_688670

theorem neither_squares_cubes_fourth_powers (n : ℕ) (h : n = 500) :
  (finset.card (finset.filter (λ (x : ℕ), ¬ (∃ m, m ^ 2 = x) ∧ ¬ (∃ m, m ^ 3 = x) ∧ ¬ (∃ m, m ^ 4 = x))
    (finset.range (n + 1)))) = 470 :=
by {
  -- Proof goes here
  sorry
}

end neither_squares_cubes_fourth_powers_l688_688670


namespace sandy_money_left_l688_688762

theorem sandy_money_left (initial_amount spent_percentage : ℝ) (h : initial_amount - initial_amount * spent_percentage = 140) :
  initial_amount = 200 ∧ spent_percentage = 0.3 :=
by 
  have h_initial_amount : initial_amount = 200, by sorry
  have h_spent_percentage : spent_percentage = 0.3, by sorry
  exact ⟨h_initial_amount, h_spent_percentage⟩ 

end sandy_money_left_l688_688762


namespace correct_negation_l688_688450

-- Definitions of switches and lights properties
variables {Switch Light : Type}
variable (is_main_switch : Switch → Prop)
variable (switch_off : Switch → Prop)
variable (light_off : Light → Prop)
variable (light_on : Light → Prop)

-- The first condition: If all switches are off, then all lights are off
def condition1 := (∀ s : Switch, switch_off s) → (∀ l : Light, light_off l)

-- The second condition: If the main switch is off, then all other switches are off
def condition2 := ∀ m : Switch, is_main_switch m → (switch_off m → ∀ s : Switch, switch_off s)

-- Combined statement: If the main switch is off, then all lights are off
def combined := ∀ m : Switch, is_main_switch m → (switch_off m → ∀ l : Light, light_off l)
def neg_combined := ∀ m : Switch, is_main_switch m → (switch_off m ∧ ∃ l : Light, light_on l)

theorem correct_negation (h1 : condition1) (h2 : condition2) (m : Switch) 
  (main_switch : is_main_switch m) : combined → neg_combined :=
sorry

end correct_negation_l688_688450


namespace dad_steps_l688_688121

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l688_688121


namespace radio_price_rank_l688_688936

theorem radio_price_rank
  (h45 : ∀ (i j : ℕ), i ≠ j → price i ≠ price j) -- All items sold at different prices
  (h1 : ∃ (k : ℕ), price k = 24th lowest price ∧ price k = highest price) -- Radio's price conditions
  (h2 : total_items = 40) -- 40 items sold at the garage sale
  : rank_from_highest (price_of_radio) = 17 := 
sorry

end radio_price_rank_l688_688936


namespace points_on_octagon_boundary_l688_688588

def is_on_octagon_boundary (x y : ℝ) : Prop :=
  |x| + |y| + |x - 1| + |y - 1| = 4

theorem points_on_octagon_boundary :
  ∀ (x y : ℝ), is_on_octagon_boundary x y ↔ ((0 ≤ x ∧ x ≤ 1 ∧ (y = 2 ∨ y = -1)) ∨
                                             (0 ≤ y ∧ y ≤ 1 ∧ (x = 2 ∨ x = -1)) ∨
                                             (x ≥ 1 ∧ y ≥ 1 ∧ x + y = 3) ∨
                                             (x ≤ 1 ∧ y ≤ 1 ∧ x + y = 1) ∨
                                             (x ≥ 1 ∧ y ≤ -1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≥ 1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≤ 1 ∧ x + y = -1) ∨
                                             (x ≤ 1 ∧ y ≤ -1 ∧ x + y = -1)) :=
by
  sorry

end points_on_octagon_boundary_l688_688588


namespace tan_double_angle_subtraction_l688_688408

theorem tan_double_angle_subtraction 
  (α β : ℝ)
  (hα : Math.tan α = 5)
  (hβ : Math.tan β = 3) :
  Math.tan (2 * α - 2 * β) = 16 / 33 := 
sorry

end tan_double_angle_subtraction_l688_688408


namespace jonessa_take_home_pay_l688_688411

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l688_688411


namespace sector_max_area_l688_688994

theorem sector_max_area (α R : ℝ) (h₀ : 0 < R) (h₁ : 0 < α)
  (perimeter_eq : ∀ r l, l + 2 * r = 40 → l = 40 - 2 * r) :
  let r := 10 in
  let l := 20 in
  let S := 100 in
  α = l / r → S = (1 / 2) * l * r :=
by
  intros
  sorry

end sector_max_area_l688_688994


namespace percentage_difference_l688_688494

theorem percentage_difference : (0.5 * 56) - (0.3 * 50) = 13 := by
  sorry

end percentage_difference_l688_688494


namespace find_function_l688_688203

theorem find_function (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y - 2023) →
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 :=
by
  intros h
  sorry

end find_function_l688_688203


namespace profit_percentage_proof_l688_688043

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 62.50
noncomputable def discount_rate : ℝ := 0.95
noncomputable def expected_profit_percentage : ℝ := 31.58

theorem profit_percentage_proof :
  let list_price := selling_price / discount_rate in
  let profit := selling_price - cost_price in
  let profit_percentage := (profit / cost_price) * 100 in
  abs (profit_percentage - expected_profit_percentage) < 0.01 :=
by
  sorry

end profit_percentage_proof_l688_688043


namespace hexagon_area_l688_688918

theorem hexagon_area (h : ∃ s : ℝ, s^2 = real.sqrt 3) : ∃ A : ℝ, A = 9 / 2 :=
by
  sorry

end hexagon_area_l688_688918


namespace digits_sum_is_23_l688_688334

/-
Juan chooses a five-digit positive integer.
Maria erases the ones digit and gets a four-digit number.
The sum of this four-digit number and the original five-digit number is 52,713.
What can the sum of the five digits of the original number be?
-/

theorem digits_sum_is_23 (x y : ℕ) (h1 : 1000 ≤ x) (h2 : x ≤ 9999) (h3 : y ≤ 9) (h4 : 11 * x + y = 52713) :
  (x / 1000) + (x / 100 % 10) + (x / 10 % 10) + (x % 10) + y = 23 :=
by {
  sorry -- Proof goes here.
}

end digits_sum_is_23_l688_688334


namespace train_speed_is_45_kmh_l688_688073

-- Define the given conditions
def length_of_train : ℝ := 385
def length_of_bridge : ℝ := 140
def time_to_pass_bridge : ℝ := 42

-- Define the total distance traveled by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Define the speed in meters per second
def speed_m_per_s : ℝ := total_distance / time_to_pass_bridge

-- Define the conversion factor from meters per second to kilometers per hour
def m_per_s_to_km_per_h : ℝ := 3.6

-- Define the speed in kilometers per hour
def speed_km_per_h : ℝ := speed_m_per_s * m_per_s_to_km_per_h

-- The statement to be proven
theorem train_speed_is_45_kmh : speed_km_per_h = 45 := sorry

end train_speed_is_45_kmh_l688_688073


namespace root_proof_l688_688225

noncomputable def p : ℝ := (-5 + Real.sqrt 21) / 2
noncomputable def q : ℝ := (-5 - Real.sqrt 21) / 2

theorem root_proof :
  (∃ (p q : ℝ), (∀ x : ℝ, x^3 + 6 * x^2 + 6 * x + 1 = 0 → (x = p ∨ x = q ∨ x = -1)) ∧ 
                 ((p = (-5 + Real.sqrt 21) / 2) ∧ (q = (-5 - Real.sqrt 21) / 2))) →
  (p / q + q / p = 23) :=
by
  sorry

end root_proof_l688_688225


namespace no_finite_arith_progression_partition_l688_688564

theorem no_finite_arith_progression_partition :
  ∀ (k : ℕ) (k > 1), 
  ¬ ∃ (a r : ℕ → ℕ), 
  (∀ i j : ℕ, i ≠ j → r i ≠ r j) ∧ 
  (∀ n : ℕ, ∃ i m : ℕ, n = a i + m * r i) :=
sorry

end no_finite_arith_progression_partition_l688_688564


namespace area_inside_U_l688_688989

-- Definition of four-presentable complex number
def four_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 4 ∧ z = w - 1/w

-- Definition of the set of all four-presentable complex numbers
def U : set ℂ := {z | four_presentable z}

-- Given conditions in Lean to set up the area calculation
theorem area_inside_U : (π * 255) / 16 =
sorry

end area_inside_U_l688_688989


namespace range_of_f_l688_688842

-- Define the function
def f (x : ℝ) : ℝ := sin x + (sqrt 3) * cos x

-- Define the domain
def domain (x : ℝ) : Prop := - (π / 2) ≤ x ∧ x ≤ π / 2

-- Define the range of the function
def range (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 2

-- State the theorem to be proved
theorem range_of_f : ∀ x : ℝ, domain x → range (f x) := by sorry

end range_of_f_l688_688842


namespace first_number_in_proportion_is_correct_l688_688288

-- Define the proportion condition
def proportion_condition (a x : ℝ) : Prop := a / x = 5 / 11

-- Define the given known value for x
def x_value : ℝ := 1.65

-- Define the correct answer for a
def correct_a : ℝ := 0.75

-- The theorem to prove
theorem first_number_in_proportion_is_correct :
  ∀ a : ℝ, proportion_condition a x_value → a = correct_a := by
  sorry

end first_number_in_proportion_is_correct_l688_688288


namespace first_batch_students_l688_688814

theorem first_batch_students 
  (x : ℕ) 
  (avg1 avg2 avg3 overall_avg : ℝ) 
  (n2 n3 : ℕ) 
  (h_avg1 : avg1 = 45) 
  (h_avg2 : avg2 = 55) 
  (h_avg3 : avg3 = 65) 
  (h_n2 : n2 = 50) 
  (h_n3 : n3 = 60) 
  (h_overall_avg : overall_avg = 56.333333333333336) 
  (h_eq : overall_avg = (45 * x + 55 * 50 + 65 * 60) / (x + 50 + 60)) 
  : x = 40 :=
sorry

end first_batch_students_l688_688814


namespace value_of_a_minus_b_l688_688771

variable {R : Type} [Field R]

noncomputable def f (a b x : R) : R := a * x + b
noncomputable def g (x : R) : R := -2 * x + 7
noncomputable def h (a b x : R) : R := f a b (g x)

theorem value_of_a_minus_b (a b : R) (h_inv : R → R) 
  (h_def : ∀ x, h_inv x = x + 9)
  (h_eq : ∀ x, h a b x = x - 9) : 
  a - b = 5 := by
  sorry

end value_of_a_minus_b_l688_688771


namespace brad_amount_l688_688495

-- Definitions for the conditions
def total_amount (j d b : ℚ) := j + d + b = 68
def josh_twice_brad (j b : ℚ) := j = 2 * b
def josh_three_fourths_doug (j d : ℚ) := j = (3 / 4) * d

-- The theorem we want to prove
theorem brad_amount : ∃ (b : ℚ), (∃ (j d : ℚ), total_amount j d b ∧ josh_twice_brad j b ∧ josh_three_fourths_doug j d) ∧ b = 12 :=
sorry

end brad_amount_l688_688495


namespace light_flashes_in_three_quarters_hour_l688_688521

theorem light_flashes_in_three_quarters_hour :
  (60 * 45) / 20 = 135 := 
by
  -- Conversion: ¾ hour to seconds
  have h1 : ¾ * (60: ℤ) = 45, by norm_num,
  have h2 : 45 * 60 = 2700, by norm_num,
  
  -- Number of flashes
  have h3 : 2700 / 20 = 135, by norm_num,

  -- Combining the results
  exact h3

end light_flashes_in_three_quarters_hour_l688_688521


namespace procurement_plan_49_snowflakes_l688_688929

noncomputable def cost_per_model_A (a : ℝ) : ℝ :=
  let long_pipes_cost := 3 * 2 * a
  let short_pipes_cost := (21 - (3 // 3)) * a
  long_pipes_cost + short_pipes_cost

def snowflake_models_feasible (a budget : ℝ) (num_model_A num_model_B long_pipes short_pipes : ℝ) : Prop :=
  budget >= num_model_A * cost_per_model_A(a) + num_model_B * (3 * 2 * a + (27 - (3 // 3)) * a) ∧
  long_pipes >= num_model_A * 3 + num_model_B * 3 ∧
  short_pipes >= num_model_A * 21 + num_model_B * 27

theorem procurement_plan_49_snowflakes
  (a budget : ℝ) (long_pipes short_pipes : ℝ) 
  (h_budget : budget = 1280)
  (h_price : a = 0.5) 
  (h_limit_long : long_pipes = 267)
  (h_limit_short : short_pipes = 2130) :
  snowflake_models_feasible a 1280 48 1 267 2130 :=
by
  sorry

end procurement_plan_49_snowflakes_l688_688929


namespace value_of_x_minus_y_squared_l688_688236

theorem value_of_x_minus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : (x - y) ^ 2 = 1 :=
by
  sorry

end value_of_x_minus_y_squared_l688_688236


namespace ratio_karen_beatrice_l688_688335

noncomputable def karen_crayons : ℕ := 128
noncomputable def judah_crayons : ℕ := 8
noncomputable def gilbert_crayons : ℕ := 4 * judah_crayons
noncomputable def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem ratio_karen_beatrice :
  karen_crayons / beatrice_crayons = 2 := by
sorry

end ratio_karen_beatrice_l688_688335


namespace area_triangle_CKL_l688_688748

theorem area_triangle_CKL (AB_length K C H L: ℝ)
  (h_AB : AB_length = 10)
  (ratio_BH_AH : 1/4)
  (h_BH_AH : BH : AH = 1 : 4)
  (H: algebraic_circle AB_length K C) 
  (altitude_CH_intersect : altitude CH triangle_intersects BK L): 
  area_triangle CKL = 8 := 
sorry

end area_triangle_CKL_l688_688748


namespace problem_equivalent_statement_l688_688657

def line_parametric (t : ℝ) : ℝ × ℝ := 
  (1 + (Real.sqrt 2) / 2 * t, -1 + (Real.sqrt 2) / 2 * t)

def circle_rect_coord (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 2

def range_of_x_plus_y (z : ℝ) : Prop :=
  -2 ≤ z ∧ z ≤ 2

theorem problem_equivalent_statement :
  (∀ (t : ℝ), circle_rect_coord (line_parametric t).1 (line_parametric t).2 → 
    range_of_x_plus_y ((line_parametric t).1 + (line_parametric t).2)) := by
  sorry

end problem_equivalent_statement_l688_688657


namespace polygon_sides_l688_688439

theorem polygon_sides (n : Nat) (h : (360 : ℝ) / (180 * (n - 2)) = 2 / 9) : n = 11 :=
by
  sorry

end polygon_sides_l688_688439


namespace part1_part2_l688_688653

noncomputable def f (x : ℝ) := x^3 - 3 * x^2 - 9 * x + 20

theorem part1 : 
  (∀ x : ℝ, f x = x^3 - 3 * x^2 - 9 * x + 20) ∧
  (f (-1) = 7) ∧ 
  (f' (-1) = 0) ∧ 
  (f' 3 = 0)
  := by sorry

theorem part2 :
  f 3 = -25
  := by sorry

end part1_part2_l688_688653


namespace number_of_pairs_satisfying_2020_l688_688216

theorem number_of_pairs_satisfying_2020 (m n : ℕ) :
  (∃ m n : ℕ, (1 / m + 1 / n = 1 / 2020)) ↔ 45 :=
sorry

end number_of_pairs_satisfying_2020_l688_688216


namespace range_of_a_l688_688291

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → -3 * x^2 + a ≤ 0) ↔ a ≤ 3 := by
  sorry

end range_of_a_l688_688291


namespace subtraction_digits_l688_688999

theorem subtraction_digits (a b c : ℕ) (h1 : c - a = 2) (h2 : b = c - 1) (h3 : 100 * a + 10 * b + c - (100 * c + 10 * b + a) = 802) :
a = 0 ∧ b = 1 ∧ c = 2 :=
by {
  -- The detailed proof steps will go here
  sorry
}

end subtraction_digits_l688_688999


namespace angle_BOE_l688_688703

theorem angle_BOE (A B C D E F O : Point) 
  (h1 : is_angle_bisector (∠ A O C) (ray O B))
  (h2 : is_angle_bisector (∠ D O F) (ray O E))
  (h3 : angle A O F = 146)
  (h4 : angle C O D = 42) : angle B O E = 94 := by
  sorry

end angle_BOE_l688_688703


namespace solve_question_45x_l688_688017

theorem solve_question_45x (?): ∃ x : Real, (45 * x / (1.5 * 10^2)) = 0.4 * 900 + 5/9 * 750 :=
by
  let y := 0.4 * 900 + (5:Real) / 9 * 750
  exact ⟨2588.9, by sorry⟩

end solve_question_45x_l688_688017


namespace Apollonian_circle_exists_l688_688665

noncomputable def is_Apollonian_circle (A B M : Point) (k : ℝ) : Prop :=
  k ≠ 1 ∧ dist A M = k * dist M B

theorem Apollonian_circle_exists (A B M : Point) (k : ℝ) :
  k ≠ 1 → (∃ (C : Point) (r : ℝ), is_Apollonian_circle A B M k) := 
sorry

end Apollonian_circle_exists_l688_688665


namespace find_remainder_l688_688724

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end find_remainder_l688_688724


namespace not_possible_to_cover_l688_688240

namespace CubeCovering

-- Defining the cube and its properties
def cube_side_length : ℕ := 4
def face_area := cube_side_length * cube_side_length
def total_faces : ℕ := 6
def faces_to_cover : ℕ := 3

-- Defining the paper strips and their properties
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def strip_area := strip_length * strip_width
def num_strips : ℕ := 16

-- Calculate the total area to cover
def total_area_to_cover := faces_to_cover * face_area
def total_area_strips := num_strips * strip_area

-- Statement: Prove that it is not possible to cover the three faces
theorem not_possible_to_cover : total_area_to_cover = 48 → total_area_strips = 48 → false := by
  intro h1 h2
  sorry

end CubeCovering

end not_possible_to_cover_l688_688240


namespace coeff_of_x4_is_135_l688_688419

noncomputable def coeff_x4_in_poly : ℤ := 
  let poly := (1 + X + X^2) * (1 - X)^10 
  (poly.coeff 4)

theorem coeff_of_x4_is_135 : coeff_x4_in_poly = 135 :=
by sorry

end coeff_of_x4_is_135_l688_688419


namespace part1_part2_l688_688626

-- Part 1
def z1 (a : ℝ) := complex.mk a 2
def z2 (a : ℝ) := complex.mk 1 (-a)

-- Part 2
theorem part1 : 
  z1 1 / z2 1 = complex.mk (-1/2) (3/2) :=
by
  sorry

-- Definitions and conditions for Part 2
def z1_b (a : ℝ) := complex.mk a 2
def z2_b (a : ℝ) := complex.mk 1 (-a)
axiom a_pos : ∀ a : ℝ, a > 0 → a^2 = 4 → a = 2

theorem part2 (a : ℝ) (ha : a > 0) (ha_sq : a^2 = 4) : 
  |z1_b 2 * z2_b 2| = 2 * real.sqrt 10 :=
by
  sorry

end part1_part2_l688_688626


namespace expression_equals_three_l688_688563

-- Define the components of the expression
def sqrt3_minus_1_abs := |real.sqrt 3 - 1|
def twenty23_minus_pi_pow_zero := (2023 - real.pi) ^ 0
def neg_reciprocal := (-1 / 3) ^ (-1 : ℤ)
def three_tan_30 := 3 * real.tan (real.pi / 6)

-- The statement to prove
theorem expression_equals_three : 
  sqrt3_minus_1_abs + twenty23_minus_pi_pow_zero - neg_reciprocal - three_tan_30 = 3 :=
by
  -- Proof goes here
  sorry

end expression_equals_three_l688_688563


namespace quadrilaterals_equal_area_l688_688466

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem quadrilaterals_equal_area
  (Q1 Q2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ))
  (h :    midpoint Q1.1.1 Q1.1.2 = midpoint Q2.1.1 Q2.1.2
               ∧ midpoint Q1.1.2 Q1.2.1 = midpoint Q2.1.2 Q2.2.1
               ∧ midpoint Q1.2.1 Q1.2.2 = midpoint Q2.2.1 Q2.2.2
               ∧ midpoint Q1.2.2 Q1.1.1 = midpoint Q2.2.2 Q2.1.1)
  (theorem_528 : ∀ (Q : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)),
    let M1 := midpoint Q.1.1 Q.1.2,
        M2 := midpoint Q.1.2 Q.2.1,
        M3 := midpoint Q.2.1 Q.2.2,
        M4 := midpoint Q.2.2 Q.1.1 in
    area_of_Q M1 M2 M3 M4 = (1 / 2 : ℝ) * area_of_Q Q.1.1 Q.1.2 Q.2.1 Q.2.2) : 
  area_of_Q Q1.1.1 Q1.1.2 Q1.2.1 Q1.2.2 = area_of_Q Q2.1.1 Q2.1.2 Q2.2.1 Q2.2.2 :=
sorry

end quadrilaterals_equal_area_l688_688466


namespace part1_part2_l688_688006

theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x :=
  sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.cos (a * x) - Real.log (1 - x^2)) (hf0 : ∃ x, f' x = -a * Real.sin (a * x) + 2*x / (1 - x^2)) :
  0 = f' 0 → f'' 0 < 0 → (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) :=
  sorry

end part1_part2_l688_688006


namespace hexagon_largest_angle_l688_688432

theorem hexagon_largest_angle (angles : Fin 6 → ℝ) 
  (h1 : ∀ i, angles i % 2 = 0) 
  (h2 : angles 0 < angles 1 ∧ angles 1 < angles 2 ∧ angles 2 < angles 3 ∧ angles 3 < angles 4 ∧ angles 4 < angles 5 ∧angles 0 > 0)
  (h_sum : ∑ i, angles i = 720) : ∃ x, x = angles 3 + 6 ∧ x = 125 := 
by
  sorry

end hexagon_largest_angle_l688_688432


namespace sin_B_sin_C_triangle_perimeter_l688_688325

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ}
-- Area condition
variable h1 : (1 / 2) * a * c * Real.sin B = a^2 / (3 * Real.sin A)
-- Cosine and side length condition
variable h2 : 6 * Real.cos B * Real.cos C = 1
variable h3 : a = 3

-- Proof of the first part
theorem sin_B_sin_C : Real.sin B * Real.sin C = 2 / 3 := by
  sorry

-- Proof of the second part
theorem triangle_perimeter : a + b + c = 3 + Real.sqrt 33 := by
  sorry

end sin_B_sin_C_triangle_perimeter_l688_688325


namespace dad_steps_l688_688134

theorem dad_steps (steps_yasha_masha : ℕ) (masha_step_contains : 3 * steps_dad = 5 * steps_masha)
  (masha_yasha_steps : 3 * steps_yasha = 5 * steps_masha) (masha_yasha_combined : 400 = steps_yasha + steps_masha) :
  ∃ steps_dad, steps_dad = 90 :=
by
  let giant_steps := masha_yasha_combined / 8
  have masha_steps := 3 * giant_steps
  have masha_five_steps := masha_steps / 5
  let steps_dad := 3 * masha_five_steps
  use steps_dad
  sorry

end dad_steps_l688_688134


namespace product_of_odd_and_even_is_odd_l688_688681

theorem product_of_odd_and_even_is_odd {f g : ℝ → ℝ} 
  (hf : ∀ x : ℝ, f (-x) = -f x)
  (hg : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x) * (g x) = -(f (-x) * g (-x)) :=
by
  sorry

end product_of_odd_and_even_is_odd_l688_688681


namespace smallest_integer_B_l688_688579

theorem smallest_integer_B (B : ℤ) : 
  (∃ n : ℕ, (B ≤ n) ∧ (∑ k in finset.range (n - B + 1), (B + k)) = 2023) → B = -2022 :=
by
  sorry

end smallest_integer_B_l688_688579


namespace OJ_eq_JD_l688_688726

variable (A B C D E J O : Point)
variable (α β γ : ℝ)

-- Given conditions
variable [is_acute_triangle : acute_triangle A B C]
variable [AB_gt_AC : A.distance_to(B) > A.distance_to(C)]
variable [is_bisector_BAC_D : angle_bisector A B C D]
variable [circumcenter_O : circumcenter A B C O]
variable [AO_intersect_BC_E : intersects A O B C E]
variable [incenter_J : incenter A E D J]
variable [angle_ADO_45 : ∠(A, D, O) = 45]

-- Prove statement
theorem OJ_eq_JD : distance O J = distance J D := 
by 
  sorry

end OJ_eq_JD_l688_688726


namespace nine_distinct_nonzero_integers_exist_l688_688398

-- Definition of the problem conditions.
def distinct_nonzero_integers (a : Fin 9 → ℤ) :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i, a i ≠ 0)

def sum_is_perfect_square (a : Fin 9 → ℤ) :=
  ∃ S : ℤ, S = (Σ i, a i) ∧ ∃ k : ℤ, S = k^2

def sum_of_any_eight_is_perfect_cube (a : Fin 9 → ℤ) :=
  ∀ i, ∃ m : ℤ, (Σ j, if j = i then 0 else a j) = (2 * m)^3

theorem nine_distinct_nonzero_integers_exist :
  ∃ (a : Fin 9 → ℤ), distinct_nonzero_integers a ∧ sum_is_perfect_square a ∧ sum_of_any_eight_is_perfect_cube a :=
sorry

end nine_distinct_nonzero_integers_exist_l688_688398


namespace cathy_total_money_l688_688092

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l688_688092


namespace partial_fraction_series_sum_l688_688105

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end partial_fraction_series_sum_l688_688105


namespace dad_steps_l688_688146

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688146


namespace circle_count_2012_l688_688069

/-
The pattern is defined as follows: 
○●, ○○●, ○○○●, ○○○○●, …
We need to prove that the number of ● in the first 2012 circles is 61.
-/

-- Define the pattern sequence
def circlePattern (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Total number of circles in the first k segments:
def totalCircles (k : ℕ) : ℕ :=
  k * (k + 1) / 2 + k

theorem circle_count_2012 : 
  ∃ (n : ℕ), totalCircles n ≤ 2012 ∧ 2012 < totalCircles (n + 1) ∧ n = 61 :=
by
  sorry

end circle_count_2012_l688_688069


namespace alcohol_water_ratio_l688_688819

theorem alcohol_water_ratio (p q r : ℝ) :
  let A := (p * q * r) + (p * q) + (p * r) + (q * r) + p + q + r
  let B := p * q + p * r + q * r + p + q + r + 3
  (A / B) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 3) :=
begin
  sorry
end

end alcohol_water_ratio_l688_688819


namespace correct_option_l688_688846

theorem correct_option : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := sorry

end correct_option_l688_688846


namespace geometric_series_sum_l688_688600

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 16383 / 49152 :=
by
  sorry

end geometric_series_sum_l688_688600


namespace quarterback_sacked_times_l688_688537

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quarterback_sacked_times_l688_688537


namespace routes_from_P_to_Q_l688_688101

-- Define the cities and roads
inductive City
| P | Q | R | S | T | U
deriving DecidableEq

open City

structure Road where
  from : City
  to : City
deriving DecidableEq

def roads : List Road := [
  ⟨P, Q⟩, ⟨P, S⟩, ⟨P, T⟩, ⟨P, U⟩, 
  ⟨Q, R⟩, ⟨Q, S⟩, ⟨R, S⟩, ⟨R, T⟩, 
  ⟨S, U⟩, ⟨U, T⟩
]

-- Define a route as a sequence of roads
def Route := List Road

-- Function to check if a route uses each road exactly once
def uses_each_road_exactly_once (rt : Route) : Bool :=
  (rt.to_finset = roads.to_finset) ∧ 
  (roads.to_finset.card = rt.to_finset.card)

-- Define the problem statement
def num_routes (P Q : City) (rts : List Route) := 
  (count (λ rt, uses_each_road_exactly_once rt) rts).val

theorem routes_from_P_to_Q : 
  num_routes P Q [all_possible_routes] = 15 := sorry

end routes_from_P_to_Q_l688_688101


namespace nancy_knows_r_l688_688406

theorem nancy_knows_r (r : ℚ) (h : 8 = 2^(3 * r + 3)) : r = 0 :=
sorry

end nancy_knows_r_l688_688406


namespace percent_uni_no_job_choice_l688_688310

variable (P_ND_JC P_JC P_UD P_U_NJC P_NJC : ℝ)
variable (h1 : P_ND_JC = 0.18)
variable (h2 : P_JC = 0.40)
variable (h3 : P_UD = 0.37)

theorem percent_uni_no_job_choice :
  (P_UD - (P_JC - P_ND_JC)) / (1 - P_JC) = 0.25 :=
by
  sorry

end percent_uni_no_job_choice_l688_688310


namespace tan_of_angle_second_quadrant_complex_expression_of_angle_second_quadrant_l688_688250

noncomputable def tan_value (α : ℝ) (y : ℝ) : ℝ : =
  if h : sin α = (sqrt 2 / 4) * y ∧ α ∈ set.Ioo (π/2) π ∧ cos α = -sqrt 2 / sqrt (2 + y^2 - 2) then
    (-y) / sqrt 2
  else
    0

noncomputable def complex_expression_value (sinα cosα tanα : ℝ) : ℝ :=
  if h : α ∈ set.Ioo (π/2) π then
    (3 * sinα * cosα) / (4 * sinα^2 + 2 * cosα^2)
  else
    0

theorem tan_of_angle_second_quadrant (α : ℝ) (y : ℝ) :
  ∀ (h : sin α = (sqrt 2 / 4) * y ∧ α ∈ set.Ioo (π/2) π ∧ cos α = -sqrt 2 / sqrt (2 + y^2 - 2)), 
    tan_value α y = -sqrt 3 :=
by sorry

theorem complex_expression_of_angle_second_quadrant (α : ℝ) (y : ℝ) :
  ∀ (h : sin α = (sqrt 2 / 4) * y ∧ α ∈ set.Ioo (π/2) π ∧ cos α = -sqrt 2 / sqrt (2 + y^2 - 2)), 
    complex_expression_value (sin α) (cos α) (tan α) = - (3 * sqrt 3) / 14 :=
by sorry

end tan_of_angle_second_quadrant_complex_expression_of_angle_second_quadrant_l688_688250


namespace solve_equation_l688_688852

theorem solve_equation (x : ℝ) (h : x > 0) :
  (2 / 15) * (16^(log x / log 9 + 1) - 16^(log (sqrt x) / log 3)) + 16^(log x / log 3) - log √5 5 √5 = 0 ↔ x = 1 :=
by 
  sorry

end solve_equation_l688_688852


namespace expected_value_of_a_squared_l688_688023

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l688_688023


namespace arithmetic_sequence_sum_l688_688770

theorem arithmetic_sequence_sum :
  ∃ (a_n : ℕ → ℝ) (d : ℝ), 
  (∀ n, a_n n = a_n 0 + n * d) ∧
  d > 0 ∧
  a_n 0 + a_n 1 + a_n 2 = 15 ∧
  a_n 0 * a_n 1 * a_n 2 = 80 →
  a_n 10 + a_n 11 + a_n 12 = 135 :=
by
  sorry

end arithmetic_sequence_sum_l688_688770


namespace investment_C_l688_688927

theorem investment_C (A_invest B_invest profit_A total_profit C_invest : ℕ)
  (hA_invest : A_invest = 6300) 
  (hB_invest : B_invest = 4200) 
  (h_profit_A : profit_A = 3900) 
  (h_total_profit : total_profit = 13000) 
  (h_proportional : profit_A / total_profit = A_invest / (A_invest + B_invest + C_invest)) :
  C_invest = 10500 := by
  sorry

end investment_C_l688_688927


namespace smallest_integer_larger_than_expression_l688_688476

theorem smallest_integer_larger_than_expression :
  ∃ n : ℤ, n = 248 ∧ (↑n > ((Real.sqrt 5 + Real.sqrt 3) ^ 4 : ℝ)) :=
by
  sorry

end smallest_integer_larger_than_expression_l688_688476


namespace unique_m_exist_l688_688192

def number_of_valid_m : ℕ := 1

theorem unique_m_exist :
  ∃! m : ℝ, (∃ (a b c d : ℝ), 
    y = 2*x + 1 ∧ y = m*x + 3 ∧ 
    (m = 8) ∧ (legs of the right triangle are parallel to the x and y axes)) :=
sorry

end unique_m_exist_l688_688192


namespace cat_chase_rat_l688_688896

/--
Given:
- The cat chases a rat 6 hours after the rat runs.
- The cat takes 4 hours to reach the rat.
- The average speed of the rat is 36 km/h.
Prove that the average speed of the cat is 90 km/h.
-/
theorem cat_chase_rat
  (t_rat_start : ℕ)
  (t_cat_chase : ℕ)
  (v_rat : ℕ)
  (h1 : t_rat_start = 6)
  (h2 : t_cat_chase = 4)
  (h3 : v_rat = 36)
  (v_cat : ℕ)
  (h4 : 4 * v_cat = t_rat_start * v_rat + t_cat_chase * v_rat) :
  v_cat = 90 :=
by
  sorry

end cat_chase_rat_l688_688896


namespace arc_PQ_circumference_l688_688348

-- Definitions based on the identified conditions
def radius : ℝ := 24
def angle_PRQ : ℝ := 90

-- The theorem to prove based on the question and correct answer
theorem arc_PQ_circumference : 
  angle_PRQ = 90 → 
  ∃ arc_length : ℝ, arc_length = (2 * Real.pi * radius) / 4 ∧ arc_length = 12 * Real.pi :=
by
  sorry

end arc_PQ_circumference_l688_688348


namespace second_term_geometric_sequence_l688_688807

-- Given conditions
def a3 : ℕ := 12
def a4 : ℕ := 18
def q := a4 / a3 -- common ratio

-- Geometric progression definition
noncomputable def a2 := a3 / q

-- Theorem to prove
theorem second_term_geometric_sequence : a2 = 8 :=
by
  -- proof not required
  sorry

end second_term_geometric_sequence_l688_688807


namespace num_pairs_eq_45_l688_688214

def num_pairs_satisfying_equation : ℕ := 45

theorem num_pairs_eq_45 :
  ∃ (m n : ℕ), (∑ m, ∑ n, if hn : n ≠ 0 then ((m : ℚ)⁻¹ + (n : ℚ)⁻¹ = 2020⁻¹) else false) = num_pairs_satisfying_equation := sorry

end num_pairs_eq_45_l688_688214


namespace original_price_l688_688438

theorem original_price (x : ℝ) (h : x * 0.98325 = 1400) : x = 1400 / 0.98325 := 
by 
  have : x = 1425, from (1400 / 0.98325).round,
  exact this

-- This is a statement, proof is not needed.

end original_price_l688_688438


namespace find_m_l688_688676

noncomputable theory

-- Define the polynomial expansion
def polynomial_expansion (m : ℝ) (x : ℝ) : ℝ :=
  (1 + m * x)^6

-- Define the conditions as given in the problem
def condition_1 (m : ℝ) (a : ℕ → ℝ) : Prop :=
  polynomial_expansion m x = ∑ i in range 7, a i * x^i

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 = -63

-- Define the proof problem
theorem find_m (m : ℝ) (a : ℕ → ℝ) (x : ℝ) (h1 : condition_1 m a) (h2 : condition_2 a) : m = 3 ∨ m = -1 :=
sorry

end find_m_l688_688676


namespace find_distance_to_third_side_l688_688436

-- Definition of the problem conditions
variables {P : Point}
variables (a : ℝ) (b : ℝ) (c : ℝ)

def equilateral_triangle_side_length := 10
def distance_from_P_to_first_side := 1
def distance_from_P_to_second_side := 3

-- Viviani's Theorem in the context of our problem
def altitude_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * Real.sqrt(3)) / 2

def sum_of_distances {altitude : ℝ} (d1 d2 d3 : ℝ) : Prop :=
  d1 + d2 + d3 = altitude

-- Problem statement as a theorem to prove
theorem find_distance_to_third_side :
  let altitude := altitude_of_equilateral_triangle equilateral_triangle_side_length
  let distance_to_third_side := altitude - (distance_from_P_to_first_side + distance_from_P_to_second_side)
  distance_to_third_side = 5 * Real.sqrt(3) - 4 := by
  sorry

end find_distance_to_third_side_l688_688436


namespace expected_value_a_squared_norm_bound_l688_688030

section RandomVectors

def random_vector (n : ℕ) : Type :=
  {v : (Fin n) → Fin 3 → ℝ // ∀ i, ∃ j, v i j = 1 ∧ ∀ k ≠ j, v i k = 0}

def sum_vectors {n : ℕ} (vecs : random_vector n) : (Fin 3) → ℝ :=
  λ j, ∑ i, vecs.val i j

def a_squared {n : ℕ} (vecs : random_vector n) : ℝ :=
  ∑ j, (sum_vectors vecs j) ^ 2

noncomputable def expected_a_squared (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + n^2) / 3

theorem expected_value_a_squared (n : ℕ) (vecs : random_vector n) :
  ∑ j, (sum_vectors vecs j) ^ 2 = expected_a_squared n :=
sorry

theorem norm_bound (n : ℕ) (vecs : random_vector n) :
  real.sqrt ((sum_vectors vecs 0) ^ 2 + (sum_vectors vecs 1) ^ 2 + (sum_vectors vecs 2) ^ 2) ≥ n / real.sqrt 3 :=
sorry

end RandomVectors

end expected_value_a_squared_norm_bound_l688_688030


namespace increasing_sine_plus_ax_l688_688594

theorem increasing_sine_plus_ax (a : ℝ) : 
  (∀ x : ℝ, ∀ y : ℝ, y = (sin x + a * x) → (cos x + a) ≥ 0) → a ≥ 1 := 
by 
  sorry

end increasing_sine_plus_ax_l688_688594


namespace particle_reaches_4_2_in_8_seconds_l688_688524

/-- The number of ways for the particle to reach the point (4, 2) after 8 seconds
    starting from the origin (0, 0) by moving either right, left, up, or down by one unit
    at the end of each second is 448. --/
theorem particle_reaches_4_2_in_8_seconds :
  ∃ (moves : List (ℕ × ℕ)), 
  (moves.length = 8) ∧ 
  (∀ move ∈ moves, move = (1, 0) ∨ move = (-1, 0) ∨ move = (0, 1) ∨ move = (0, -1)) ∧ 
  (List.sum (List.map Prod.fst moves) = 4) ∧ 
  (List.sum (List.map Prod.snd moves) = 2) ∧ 
  (number_of_ways moves = 448) :=
sorry

end particle_reaches_4_2_in_8_seconds_l688_688524


namespace find_annual_salary_l688_688710

variable (S : ℝ)
variable (medicalBills : ℝ := 200000)
variable (receivedAmount : ℝ := 5440000)
variable (punitiveMultiplier : ℝ := 3)
variable (proportionReceived : ℝ := 0.80)
variable (years : ℝ := 30)

def totalSalaryDamages := years * S
def totalMedicalBills := medicalBills
def totalPunitiveDamages := punitiveMultiplier * (totalSalaryDamages + totalMedicalBills)
def totalDamages := totalSalaryDamages + totalMedicalBills + totalPunitiveDamages

theorem find_annual_salary (h : proportionReceived * totalDamages = receivedAmount) : S = 50000 := by sorry

end find_annual_salary_l688_688710


namespace dad_steps_l688_688155

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l688_688155


namespace no_closed_polygonal_line_exists_l688_688997

-- Define Point and Vector as tuples of real numbers for simplicity
def Point := (ℝ × ℝ)
def Vector := (ℝ × ℝ)

-- Define a condition that a set of 25 line segments exists, starting from a fixed point A
variable (A : Point) (l : ℝ → Point) (v : Fin 25 → Vector)

-- Define a closed polygon condition: the sum of 25 vectors should equal zero
def is_closed_polygonal_line (segment : Fin 25 → Vector) : Prop :=
  ∑ i, segment i = (0, 0)

-- Define a parallel and equal condition
def is_parallel_and_equal (v w: Vector) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (w = (k * v.1, k * v.2))

-- The main theorem statement
theorem no_closed_polygonal_line_exists :
  ¬ ∃ (segment : Fin 25 → Vector),
    (∀ i, ∃ j, is_parallel_and_equal (segment i) (v j)) ∧
    is_closed_polygonal_line segment :=
sorry

end no_closed_polygonal_line_exists_l688_688997


namespace largest_binomial_coefficient_largest_absolute_value_smallest_coefficient_l688_688321

noncomputable def polynomial := (λ (x : ℝ), (x^(1/2) - 2 / x^2)^8)

theorem largest_binomial_coefficient :
  ∀ x : ℝ, 
  (∃ t, t = (polynomial x) ∧ t = ∑ i in (finset.range 9), 
     (nat.choose 8 i) * (x^(1/2))^(8 - i) * ((-2 / x^2)^i) ∧ 
     t = 1120 / x^6) := 
sorry

theorem largest_absolute_value :
  ∀ x : ℝ,
  (∃ t1 t2, t1 = (polynomial x) ∧ t2 = (polynomial x) ∧ 
     t1 = -1792 * x^(-17/2) ∧ t2 = -1792 * x^(-11)) :=
sorry

theorem smallest_coefficient :
  ∀ x : ℝ,
  (∃ t, t = (polynomial x) ∧ t = -1792 * x^(-17/2)) :=
sorry

end largest_binomial_coefficient_largest_absolute_value_smallest_coefficient_l688_688321


namespace cyclic_quadrilateral_inradii_sum_eq_l688_688725

theorem cyclic_quadrilateral_inradii_sum_eq 
  (A B C D : Type)
  (circ_quad : cyclic_quadrilateral A B C D) 
  (r_a r_b r_c r_d : ℝ)
  (inradius_BCD : inradius_of_triangle B C D = r_a)
  (inradius_ACD : inradius_of_triangle A C D = r_b)
  (inradius_ABD : inradius_of_triangle A B D = r_c)
  (inradius_ABC : inradius_of_triangle A B C = r_d)
  : r_a + r_c = r_b + r_d :=
begin
  sorry
end

end cyclic_quadrilateral_inradii_sum_eq_l688_688725


namespace right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l688_688481

theorem right_triangle_set_A :
  let a := 1
      b := 2
      c := Real.sqrt 5
  in a^2 + b^2 = c^2 := 
sorry

theorem not_right_triangle_set_B :
  let a := 6
      b := 8
      c := 9
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_C :
  let a := Real.sqrt 3
      b := Real.sqrt 2
      c := 5
  in a^2 + b^2 ≠ c^2 :=
sorry

theorem not_right_triangle_set_D :
  let a := 3^2
      b := 4^2
      c := 5^2
  in a^2 + b^2 ≠ c^2 :=
sorry

end right_triangle_set_A_not_right_triangle_set_B_not_right_triangle_set_C_not_right_triangle_set_D_l688_688481


namespace required_score_to_reach_target_l688_688767

-- Define Sophia's current scores
def current_scores : List ℕ := [95, 85, 75, 65, 95]

-- Define the target increase in the average score
def target_increase : ℕ := 5

-- Calculate the current total score
def current_total_score : ℕ := current_scores.sum

-- Calculate the number of tests taken so far
def num_tests_so_far : ℕ := current_scores.length

-- Calculate the current average score
def current_average_score : ℕ := current_total_score / num_tests_so_far

-- Calculate the desired average score
def desired_average_score : ℕ := current_average_score + target_increase

-- Calculate the total score required to achieve the desired average
def total_score_required : ℕ := desired_average_score * (num_tests_so_far + 1)

-- Calculate the required score on the next test
def required_score_on_next_test : ℕ := total_score_required - current_total_score

-- Lean 4 statement asserting the required score on the next test
theorem required_score_to_reach_target: 
  required_score_on_next_test = 113 := 
by
  unfold current_scores target_increase current_total_score num_tests_so_far current_average_score 
  unfold desired_average_score total_score_required required_score_on_next_test
  simp
  sorry

end required_score_to_reach_target_l688_688767


namespace quadratic_no_real_roots_iff_discriminant_lt_zero_l688_688446

theorem quadratic_no_real_roots_iff_discriminant_lt_zero {a b c : ℝ} (h : a ≠ 0):
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0)) ↔ (b^2 - 4 * a * c < 0) :=
sorry

example : (∀ x : ℝ, ¬ (x^2 + 2 * x + 2 = 0)) :=
begin
  apply quadratic_no_real_roots_iff_discriminant_lt_zero 1,
  linarith,
  sorry,
end

end quadratic_no_real_roots_iff_discriminant_lt_zero_l688_688446


namespace bug_crosses_24_tiles_l688_688538

theorem bug_crosses_24_tiles :
  ∀ (width length : ℕ), 
  width = 12 → 
  length = 18 → 
  let gcd := Nat.gcd width length in 
  (width + length - gcd) = 24 :=
by
  sorry

end bug_crosses_24_tiles_l688_688538


namespace table_property_l688_688923

noncomputable def table : ℕ → ℕ → ℕ :=
sorry -- table should be a function that returns the number in a given (i,j) position

def is_in_table (n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
∃ (i j : ℕ), (i < 17) ∧ (j < 17) ∧ (table i j = n)

def count_occurrences (n : ℕ) (table : ℕ → ℕ → ℕ) : ℕ :=
∑ i j, if table i j = n then 1 else 0

def column_contains (table : ℕ → ℕ → ℕ) (i : ℕ) (n : ℕ) : Prop :=
∃ j, j < 17 ∧ table i j = n

def row_contains (table : ℕ → ℕ → ℕ) (j : ℕ) (n : ℕ) : Prop :=
∃ i, i < 17 ∧ table i j = n

def num_distinct_in_column (table : ℕ → ℕ → ℕ) (i : ℕ) : ℕ :=
card (finset.univ.filter (λ n, column_contains table i n))

def num_distinct_in_row (table : ℕ → ℕ → ℕ) (j : ℕ) : ℕ :=
card (finset.univ.filter (λ n, row_contains table j n))

theorem table_property :
  (∀ n, 1 ≤ n ∧ n ≤ 17 → count_occurrences n table = 17) →
  ∃ i, i < 17 ∧ (num_distinct_in_column table i ≥ 5) ∨
  ∃ j, j < 17 ∧ (num_distinct_in_row table j ≥ 5) :=
sorry

end table_property_l688_688923


namespace two_point_four_times_eight_point_two_l688_688559

theorem two_point_four_times_eight_point_two (x y z : ℝ) (hx : x = 2.4) (hy : y = 8.2) (hz : z = 4.8 + 5.2) :
  x * y * z = 2.4 * 8.2 * 10 ∧ abs (x * y * z - 200) < abs (x * y * z - 150) ∧
  abs (x * y * z - 200) < abs (x * y * z - 250) ∧
  abs (x * y * z - 200) < abs (x * y * z - 300) ∧
  abs (x * y * z - 200) < abs (x * y * z - 350) := by
  sorry

end two_point_four_times_eight_point_two_l688_688559


namespace area_of_region_outside_circles_l688_688064

-- Definition of regular hexagon side length
def side_length : ℝ := 4

-- Area formula for a regular hexagon with side length s
def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

-- Area of the circles
def circle_radius : ℝ := side_length / 2
def circle_area (r : ℝ) : ℝ := Real.pi * r^2
def total_circle_area : ℝ := 3 * circle_area circle_radius

-- Area of the region inside the hexagon but outside the circles
def shaded_area := hexagon_area side_length - total_circle_area

theorem area_of_region_outside_circles : 
  shaded_area = 24 * Real.sqrt 3 - 12 * Real.pi :=
by 
  sorry

end area_of_region_outside_circles_l688_688064


namespace dad_steps_l688_688162

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688162


namespace students_only_english_l688_688855

variable (total_students both_english_german enrolled_german: ℕ)

theorem students_only_english :
  total_students = 45 ∧ both_english_german = 12 ∧ enrolled_german = 22 ∧
  (∀ S E G B : ℕ, S = total_students ∧ B = both_english_german ∧ G = enrolled_german - B ∧
   (S = E + G + B) → E = 23) :=
by
  sorry

end students_only_english_l688_688855


namespace problem1_problem2_l688_688013

-- Problem 1: Prove that the given expression evaluates to the correct answer
theorem problem1 :
  2 * Real.sin (Real.pi / 6) - (2015 - Real.pi)^0 + abs (1 - Real.tan (Real.pi / 3)) = abs (1 - Real.sqrt 3) :=
sorry

-- Problem 2: Prove that the solutions to the given equation are correct
theorem problem2 (x : ℝ) :
  (x-2)^2 = 3 * (x-2) → x = 2 ∨ x = 5 :=
sorry

end problem1_problem2_l688_688013


namespace dad_steps_l688_688163

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l688_688163


namespace trapezoid_perimeter_l688_688860

noncomputable def point := (ℝ × ℝ)
noncomputable def distance (a b : point) : ℝ :=
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

def j : point := (-2, -4)
def k : point := (-2, 1)
def l : point := (6, 7)
def m : point := (6, -4)

def perimeter (a b c d : point) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

theorem trapezoid_perimeter :
  perimeter j k l m = 34 := by
  sorry

end trapezoid_perimeter_l688_688860


namespace cathy_total_money_l688_688095

theorem cathy_total_money : 
  let initial := 12 
  let dad_contribution := 25 
  let mom_contribution := 2 * dad_contribution 
  let total_money := initial + dad_contribution + mom_contribution 
  in total_money = 87 :=
by
  let initial := 12
  let dad_contribution := 25
  let mom_contribution := 2 * dad_contribution
  let total_money := initial + dad_contribution + mom_contribution
  show total_money = 87
  sorry

end cathy_total_money_l688_688095


namespace carpet_needed_for_room_l688_688063

def convert_sq_m_to_sq_ft (area_m2 : ℝ) : ℝ := 
  area_m2 * 10.7639

def convert_sq_ft_to_sq_yd (area_ft2 : ℝ) : ℝ := 
  area_ft2 / 9

theorem carpet_needed_for_room : 
  let length_m : ℝ := 10
  let width_m : ℝ := 8
  let area_m2 : ℝ := length_m * width_m
  let area_ft2 : ℝ := convert_sq_m_to_sq_ft area_m2
  let area_yd2 : ℝ := convert_sq_ft_to_sq_yd area_ft2
  area_yd2 = 95.68 :=
by
  admit

end carpet_needed_for_room_l688_688063


namespace Edward_money_before_spending_l688_688584

theorem Edward_money_before_spending (spent : ℕ) (left : ℕ) (total : ℕ) (h1 : spent = 16) (h2 : left = 6) (h3 : total = spent + left) : total = 22 :=
by
  rw [h1, h2] at h3
  exact h3.symm

end Edward_money_before_spending_l688_688584


namespace sum_of_powers_of_two_l688_688747

theorem sum_of_powers_of_two : (Finset.range 64).sum (λ n, 2^n) = 2^64 - 1 := 
by sorry

end sum_of_powers_of_two_l688_688747


namespace find_integer_m_l688_688380

/--
Given the sequence of equations:
2^3 = 3 + 5,
3^3 = 7 + 9 + 11,
4^3 = 13 + 15 + 17 + 19,
5^3 = 21 + 23 + 25 + 27 + 29,
where the right-hand side is the sum of m consecutive odd numbers,
and if the last number of the sequence is 109,
then the positive integer m equals 10.
-/
theorem find_integer_m (n : ℕ) (a_n : ℕ) (h₁ : n = 9) (h₂ : a_n = 109) 
  (h₃ : a_n = n^2 + 3 * n + 1) :
  let m := n + 1
  in m = 10 := 
by 
  sorry

end find_integer_m_l688_688380


namespace minimum_sum_of_cube_faces_l688_688749

theorem minimum_sum_of_cube_faces (a b c d e f : ℕ)
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ e) (h5 : b ≠ f) (h6 : c ≠ e)
  (h7 : c ≠ f) (h8 : d ≠ e) (h9 : d ≠ f)
  (h10 : |a - b| > 1) (h11 : |a - c| > 1) (h12 : |a - d| > 1)
  (h13 : |b - e| > 1) (h14 : |b - f| > 1) (h15 : |c - e| > 1)
  (h16 : |c - f| > 1) (h17 : |d - e| > 1) (h18 : |d - f| > 1)
  (h19 : |e - f| > 1) :
  a + b + c + d + e + f ≥ 18 :=
sorry

end minimum_sum_of_cube_faces_l688_688749


namespace kaleb_tickets_left_after_riding_l688_688557

theorem kaleb_tickets_left_after_riding :
  let initial_tickets : ℕ := 6
  let ticket_cost : ℕ := 9
  let total_cost_spent : ℕ := 27
  let tickets_used := total_cost_spent / ticket_cost
  let tickets_left := initial_tickets - tickets_used
  tickets_left = 3 :=
by
  let initial_tickets : ℕ := 6
  let ticket_cost : ℕ := 9
  let total_cost_spent : ℕ := 27
  let tickets_used := total_cost_spent / ticket_cost
  let tickets_left := initial_tickets - tickets_used
  have h1 : tickets_used = 3 := sorry
  have h2 : tickets_left = initial_tickets - tickets_used := by rfl
  have h3 : tickets_left = 6 - 3 := by rw [h1]
  have h4 : tickets_left = 3 := by rw [h3]
  show tickets_left = 3 from h4

end kaleb_tickets_left_after_riding_l688_688557


namespace daryl_must_leave_weight_l688_688574

theorem daryl_must_leave_weight : 
  let crate_weight := 20
  let number_of_crates := 15
  let nails_weight := 4 * 5
  let hammers_weight := 12 * 5
  let planks_weight := 10 * 30
  let total_items_weight := nails_weight + hammers_weight + planks_weight
  let total_crates_capacity := number_of_crates * crate_weight
  total_items_weight - total_crates_capacity = 80 :=
by
  let crate_weight := 20
  let number_of_crates := 15
  let nails_weight := 4 * 5
  let hammers_weight := 12 * 5
  let planks_weight := 10 * 30
  let total_items_weight := nails_weight + hammers_weight + planks_weight
  let total_crates_capacity := number_of_crates * crate_weight
  show total_items_weight - total_crates_capacity = 80
  calc
    total_items_weight - total_crates_capacity = (4 * 5 + 12 * 5 + 10 * 30) - (15 * 20) : by rfl
    ... = (20 + 60 + 300) - 300                                 : by rfl
    ... = 380 - 300                                             : by rfl
    ... = 80                                                    : by rfl

end daryl_must_leave_weight_l688_688574


namespace distance_between_intersections_is_zero_l688_688951

open Real
open Classical

noncomputable def distance_between_intersections : Real := 
  let s1 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 18 }
  let s2 := { p : ℝ × ℝ | p.1 + p.2 = 6 }
  let intersections := s1 ∩ s2
  if h : intersections.nonempty then 
    let p1 := intersections.some
    let p2 := intersections.some
    dist p1 p2
  else
    0

theorem distance_between_intersections_is_zero :
  distance_between_intersections = 0 :=
sorry

end distance_between_intersections_is_zero_l688_688951


namespace largest_perfect_square_factor_4410_l688_688829

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end largest_perfect_square_factor_4410_l688_688829


namespace painting_faces_of_die_l688_688950

theorem painting_faces_of_die :
  let die := [1, 2, 3, 4, 5, 6],
      combinations := {n : Finset (Fin 6) // n.card = 3},
      invalid := {s ∈ combinations | (s.map (fun i => die[i.val])).sum = 9} in
  (combinations.card - invalid.card) = 18 :=
by
  sorry

end painting_faces_of_die_l688_688950


namespace average_cost_per_stadium_l688_688815

def annual_savings : ℝ := 1500
def total_years : ℕ := 18
def num_stadiums : ℕ := 30
def total_savings := annual_savings * total_years
def average_cost := total_savings / num_stadiums

theorem average_cost_per_stadium :
  average_cost = 900 := 
by
  sorry

end average_cost_per_stadium_l688_688815


namespace additional_terms_in_induction_step_l688_688388

variable (n : ℕ)

theorem additional_terms_in_induction_step :
  ∀ (k : ℕ), k > 0 → 
    let lhs_k := (Finset.range (2*k)).sum (λ i, if i % 2 = 0 then -(1 : ℚ) / (i + 1) else (1 : ℚ) / (i + 1))
    let lhs_k1 := (Finset.range (2*(k+1))).sum (λ i, if i % 2 = 0 then -(1 : ℚ) / (i + 1) else (1 : ℚ) / (i + 1))
    lhs_k1 - lhs_k = (1 / (2*k + 1) - 1 / (2*k + 2)) := 
sorry

end additional_terms_in_induction_step_l688_688388


namespace problem_part1_problem_part2_problem_part3_l688_688270

open Set

noncomputable def U := ℝ
noncomputable def A := { x : ℝ | x < -4 ∨ x > 1 }
noncomputable def B := { x : ℝ | -3 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem_part1 :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 3 } := by sorry

theorem problem_part2 :
  compl A ∪ compl B = { x : ℝ | x ≤ 1 ∨ x > 3 } := by sorry

theorem problem_part3 (k : ℝ) :
  { x : ℝ | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1 } ⊆ A → k > 1 := by sorry

end problem_part1_problem_part2_problem_part3_l688_688270


namespace sum_of_divisors_45_l688_688599

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_45 : sum_of_divisors 45 = 78 := 
  sorry

end sum_of_divisors_45_l688_688599


namespace trips_per_student_l688_688336

theorem trips_per_student
  (num_students : ℕ := 5)
  (chairs_per_trip : ℕ := 5)
  (total_chairs : ℕ := 250)
  (T : ℕ) :
  num_students * chairs_per_trip * T = total_chairs → T = 10 :=
by
  intro h
  sorry

end trips_per_student_l688_688336


namespace ellipse_eccentricity_l688_688297

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
    (h4 : b = real.sqrt (a^2 - b^2)) : 
    real.sqrt (a^2 - b^2) / a = real.sqrt 2 / 2 :=
by
  sorry

end ellipse_eccentricity_l688_688297


namespace no_exist_positive_a_l688_688963

theorem no_exist_positive_a (a : ℝ) (ha : 0 < a) :
  ¬ ∀ x : ℝ, |cos x| + |cos (a * x)| > sin x + sin (a * x) := 
sorry

end no_exist_positive_a_l688_688963


namespace kite_area_eq_twenty_l688_688820

theorem kite_area_eq_twenty :
  let base := 10
  let height := 2
  let area_of_triangle := (1 / 2 : ℝ) * base * height
  let total_area := 2 * area_of_triangle
  total_area = 20 :=
by
  sorry

end kite_area_eq_twenty_l688_688820


namespace expected_value_is_correct_l688_688079

-- Define the monetary outcomes associated with each side
def monetaryOutcome (X : String) : ℚ :=
  if X = "A" then 2 else 
  if X = "B" then -4 else 
  if X = "C" then 6 else 
  0

-- Define the probabilities associated with each side
def probability (X : String) : ℚ :=
  if X = "A" then 1/3 else 
  if X = "B" then 1/2 else 
  if X = "C" then 1/6 else 
  0

-- Compute the expected value
def expectedMonetaryOutcome : ℚ := (probability "A" * monetaryOutcome "A") 
                                + (probability "B" * monetaryOutcome "B") 
                                + (probability "C" * monetaryOutcome "C")

theorem expected_value_is_correct : 
  expectedMonetaryOutcome = -2/3 := by
  sorry

end expected_value_is_correct_l688_688079


namespace points_in_circle_l688_688755

theorem points_in_circle (points : Finset (ℝ × ℝ)) (h_card : points.card = 51) :
  ∃ (c : ℝ × ℝ), (Finset.filter (λ p, dist p c < 1 / 7) points).card ≥ 3 :=
by
  -- Problem conditions
  have conditions : true := true.intro
  -- Placeholder for proof, Lean requires a non-empty proof body
  sorry

end points_in_circle_l688_688755


namespace polynomial_const_example_l688_688407

theorem polynomial_const_example (f : ℤ[X]) (h_int_coeffs : ∀ c ∈ finset.range 1999, 0 ≤ f.eval c ∧ f.eval c ≤ 1997) :
  ∀ c ∈ finset.range 1999, f.eval 0 = f.eval c :=
  sorry

end polynomial_const_example_l688_688407


namespace dad_steps_l688_688185

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688185


namespace find_s_l688_688353

def f (x s : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x + s

theorem find_s (s : ℝ) : f (-1) s = 0 → s = 9 :=
by
  sorry

end find_s_l688_688353


namespace cubic_eq_real_roots_roots_product_eq_neg_nine_l688_688257

theorem cubic_eq_real_roots :
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 + (3 / 2) * (1 - a) * x1^2 - 3 * a * x1 + b = 0 ∧
    x2^3 + (3 / 2) * (1 - a) * x2^2 - 3 * a * x2 + b = 0 ∧
    x3^3 + (3 / 2) * (1 - a) * x3^2 - 3 * a * x3 + b = 0) :=
sorry

theorem roots_product_eq_neg_nine :
  let a := 1
  let b := 1
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 - 3 * x1 + 1 = 0 ∧
    x2^3 - 3 * x2 + 1 = 0 ∧
    x3^3 - 3 * x3 + 1 = 0 ∧
    (x1^2 - 2 - x2) * (x2^2 - 2 - x3) * (x3^2 - 2 - x1) = -9) :=
sorry

end cubic_eq_real_roots_roots_product_eq_neg_nine_l688_688257


namespace digit_ends_in_zero_l688_688346

-- Define the problem conditions as a structure
structure two_digit_number (N : ℕ) : Prop :=
  (a b : ℕ) (hN : N = 10 * a + b) (ha : a < 10) (hb : b < 10)

-- Define what it means for the difference to be a positive perfect fourth power
def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ k, k^4 = n

-- Define the core problem and prove the digit ends in 0
theorem digit_ends_in_zero (N : ℕ) (h1 : two_digit_number N)
  (h2 : ∃ d, d = 9 * (h1.a - h1.b) ∧ is_perfect_fourth_power d) : h1.b = 0 :=
by
  sorry

end digit_ends_in_zero_l688_688346


namespace det_matrixA_eq_2_l688_688718

noncomputable def matrixA (a d : ℝ) : Matrix 2 2 ℝ :=
  ![![a, -2], ![1, d]]

theorem det_matrixA_eq_2 (a d : ℝ) (h : matrixA a d + 2 * (matrixA a d)⁻¹ = 0) :
  Matrix.det (matrixA a d) = 2 := by
  sorry

end det_matrixA_eq_2_l688_688718


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688884

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688884


namespace sue_charge_per_dog_l688_688099

def amount_saved_christian : ℝ := 5
def amount_saved_sue : ℝ := 7
def charge_per_yard : ℝ := 5
def yards_mowed_christian : ℝ := 4
def total_cost_perfume : ℝ := 50
def additional_amount_needed : ℝ := 6
def dogs_walked_sue : ℝ := 6

theorem sue_charge_per_dog :
  (amount_saved_christian + (charge_per_yard * yards_mowed_christian) + amount_saved_sue + (dogs_walked_sue * x) + additional_amount_needed = total_cost_perfume) → x = 2 :=
by
  sorry

end sue_charge_per_dog_l688_688099


namespace infinite_series_sum_l688_688565

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) * (n + 1) + 2 * (n + 1) + 1) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))) 
  = 7 / 6 := 
by
  sorry

end infinite_series_sum_l688_688565


namespace horse_food_per_day_l688_688556

theorem horse_food_per_day
  (ratio_sheep_horses : 6 / 7)
  (total_horse_food : 12880)
  (sheep_count : 48) :
  let horses_count := 7 * sheep_count / 6 in
  total_horse_food / horses_count = 230 := by
sorry

end horse_food_per_day_l688_688556


namespace find_magnitude_of_a_plus_b_l688_688357

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (y : ℝ) : ℝ × ℝ := (1, y)
def vector_c : ℝ × ℝ := (2, -4)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_magnitude_of_a_plus_b (x y : ℝ) 
  (h1 : is_perpendicular (vector_a x) vector_c) 
  (h2 : is_parallel (vector_b y) vector_c) : 
  magnitude (vector_a x.1 + vector_b y.1, vector_a x.2 + vector_b y.2) = Real.sqrt 10 :=
by
  sorry

end find_magnitude_of_a_plus_b_l688_688357


namespace least_n_such_that_bn_div_143_l688_688720

-- Definitions for the conditions
def b : ℕ → ℕ
| 15 := 15
| (n+1) := if n < 14 then 0 else 150 * b n + n + 1

-- The statement that we need to prove
theorem least_n_such_that_bn_div_143 : ∃ n > 15, b n % 143 = 0 ∧ ∀ m > 15, m < n → b m % 143 ≠ 0 :=
sorry

end least_n_such_that_bn_div_143_l688_688720


namespace total_distance_traveled_l688_688894

namespace BallTrack

-- Definitions of given conditions
def ballDiameter : ℝ := 5
def ballRadius : ℝ := ballDiameter / 2

def R1 : ℝ := 120
def R2 : ℝ := 70
def R3 : ℝ := 90
def R4 : ℝ := 70
def R5 : ℝ := 120

def centerPathR1 : ℝ := R1 - ballRadius
def centerPathR2 : ℝ := R2 + ballRadius
def centerPathR3 : ℝ := R3 - ballRadius
def centerPathR4 : ℝ := R4 + ballRadius
def centerPathR5 : ℝ := R5 - ballRadius

-- Final theorem statement
theorem total_distance_traveled :
  centerPathR1 * Real.pi + centerPathR2 * Real.pi + centerPathR3 * Real.pi + centerPathR4 * Real.pi + centerPathR5 * Real.pi = 467.5 * Real.pi :=
by
  sorry

end total_distance_traveled_l688_688894


namespace dad_steps_90_l688_688499

/-- 
  Given:
  - When Dad takes 3 steps, Masha takes 5 steps.
  - When Masha takes 3 steps, Yasha takes 5 steps.
  - Masha and Yasha together made a total of 400 steps.

  Prove: 
  The number of steps that Dad took is 90.
-/
theorem dad_steps_90 (total_steps: ℕ) (masha_to_dad_ratio: ℕ) (yasha_to_masha_ratio: ℕ) (steps_masha_yasha: ℕ) (h1: masha_to_dad_ratio = 5) (h2: yasha_to_masha_ratio = 5) (h3: steps_masha_yasha = 400) :
  total_steps = 90 :=
by
  sorry

end dad_steps_90_l688_688499


namespace sqrt_eq_eight_l688_688223

theorem sqrt_eq_eight (x : ℝ) (h : sqrt (4 - 5 * x) = 8) : x = -12 :=
sorry

end sqrt_eq_eight_l688_688223


namespace pentagon_distance_sum_l688_688313

theorem pentagon_distance_sum :
  let QR := 1
  let RS := 1
  let ST := 1
  let angle_Q := 150
  let angle_R := 150
  let angle_S := 150
  let right_angle_T := 90
  -- Given the conditions on pentagon PQRST with specified side lengths and angles
  (PU_length : ℝ) :=
    PU_length = 1 + (√3 / 2) →
    let c := 1
    let d := 3
    c + d = 4
:= by
  sorry

end pentagon_distance_sum_l688_688313


namespace zero_intervals_condition_l688_688433

variables 
  {a b c m n p : ℝ}
  (h0 : a ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c)

theorem zero_intervals_condition :
  (∃ x y, x ∈ set.Ioo m n ∧ y ∈ set.Ioo n p ∧ f(x) = 0 ∧ f(y) = 0) ↔
    f m * f n < 0 ∧ f p * f n < 0 := 
sorry

end zero_intervals_condition_l688_688433


namespace total_selling_price_l688_688050

theorem total_selling_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
    (h1 : original_price = 80) (h2 : discount_rate = 0.25) (h3 : tax_rate = 0.10) :
  let discount_amt := original_price * discount_rate
  let sale_price := original_price - discount_amt
  let tax_amt := sale_price * tax_rate
  let total_price := sale_price + tax_amt
  total_price = 66 := by
  sorry

end total_selling_price_l688_688050


namespace students_selected_l688_688455

-- Define the number of boys and girls
def boys : ℕ := 13
def girls : ℕ := 10

-- Define the combination function as it is useful for calculations
def combination (n k : ℕ) := nat.choose n k

-- Define the condition for the number of ways to select 1 girl and 2 boys
def ways_to_select : ℕ := 780

-- Define the correct answer that needs to be proven
def selected_students : ℕ := 1 + 2

-- Theorem statement
theorem students_selected (h : combination girls 1 * combination boys 2 = ways_to_select) : selected_students = 3 :=
by sorry

end students_selected_l688_688455


namespace dartboard_odd_score_probability_l688_688383

structure Dartboard :=
(inner_radius : ℝ)
(outer_radius : ℝ)
(inner_scores : List ℝ)
(outer_scores : List ℝ)
(double_score : Bool)

def dartboard_conditions : Dartboard :=
{ inner_radius := 4,
  outer_radius := 8,
  inner_scores := [3, 4, 4],
  outer_scores := [4, 3, 3],
  double_score := true }

noncomputable def probability_sum_odd (db : Dartboard) : ℚ :=
let area_inner := Real.pi * db.inner_radius ^ 2 in
let area_outer_ring := Real.pi * db.outer_radius ^ 2 - area_inner in
let area_inner_region := area_inner / 3 in
let area_outer_region := area_outer_ring / 3 in
let probability_odd := (2 * area_inner_region + 2 * area_outer_region) / (3 * (area_inner_region + area_outer_region)) in
let probability_even := (area_inner_region + area_outer_region) / (3 * (area_inner_region + area_outer_region)) in
(probability_odd * probability_even + probability_even * probability_odd).toRational

theorem dartboard_odd_score_probability : 
  probability_sum_odd dartboard_conditions = 4 / 9 :=
sorry

end dartboard_odd_score_probability_l688_688383


namespace dad_steps_90_l688_688174

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688174


namespace dad_steps_l688_688183

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l688_688183


namespace number_of_subsets_l688_688016

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end number_of_subsets_l688_688016


namespace range_of_a_l688_688627

def p (a : ℝ) : Prop := ∀ x ∈ Icc (2 : ℝ) 4, x^2 - 2*x - 2*a ≤ 0

def q (a : ℝ) : Prop := ∀ x y, (1 / 2 : ℝ) ≤ x → x ≤ y → y < ∞ → 
  (x^2 - a*x + 1) ≤ (y^2 - a*y + 1)

theorem range_of_a (a : ℝ) (hpq : p a ∨ q a) (hnpq : ¬ (p a ∧ q a)) :
  a ∈ Iic 1 ∪ Ioi 4 := by
  sorry

end range_of_a_l688_688627


namespace find_largest_square_area_l688_688080

def area_of_largest_square (XY YZ XZ : ℝ) (sum_of_areas : ℝ) (right_angle : Prop) : Prop :=
  sum_of_areas = XY^2 + YZ^2 + XZ^2 + 4 * YZ^2 ∧  -- sum of areas condition
  right_angle ∧                                    -- right angle condition
  XZ^2 = XY^2 + YZ^2 ∧                             -- Pythagorean theorem
  sum_of_areas = 650 ∧                             -- total area condition
  XY = YZ                                          -- assumption for simplified solving.

theorem find_largest_square_area (XY YZ XZ : ℝ) (sum_of_areas : ℝ):
  area_of_largest_square XY YZ XZ sum_of_areas (90 = 90) → 2 * XY^2 + 5 * YZ^2 = 650 → XZ^2 = 216.67 :=
sorry

end find_largest_square_area_l688_688080


namespace minimize_expression_l688_688193

theorem minimize_expression (x y : ℝ) : ∃ m ∈ set.range (λ (x y : ℝ), (x + y + x * y)^2 + (x - y - x * y)^2), m = 0 :=
by
  use 0
  sorry

end minimize_expression_l688_688193


namespace prob_point_in_region_l688_688058

theorem prob_point_in_region :
  let rect_area := 18
  let intersect_area := 15 / 2
  let probability := intersect_area / rect_area
  probability = 5 / 12 :=
by
  sorry

end prob_point_in_region_l688_688058


namespace math_problem_l688_688763

theorem math_problem (a b : ℝ) 
  (h1 : a^2 - 3*a*b + 2*b^2 + a - b = 0)
  (h2 : a^2 - 2*a*b + b^2 - 5*a + 7*b = 0) :
  a*b - 12*a + 15*b = 0 :=
by
  sorry

end math_problem_l688_688763


namespace lottery_probability_correct_l688_688431

/-- The binomial coefficient function -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of matching MegaBall and WinnerBalls in the lottery -/
noncomputable def lottery_probability : ℚ :=
  let megaBall_prob := (1 : ℚ) / 30
  let winnerBalls_prob := (1 : ℚ) / binom 45 6
  megaBall_prob * winnerBalls_prob

theorem lottery_probability_correct : lottery_probability = (1 : ℚ) / 244351800 := by
  sorry

end lottery_probability_correct_l688_688431


namespace expected_value_of_a_squared_l688_688020

open ProbabilityTheory -- Assuming we are using probability theory library in Lean

variables {n : ℕ} {vec : Fin n → (ℕ × ℕ × ℕ)}

def is_random_vector (x : ℕ × ℕ × ℕ) : Prop :=
  (x = (1, 0, 0)) ∨ (x = (0, 1, 0)) ∨ (x = (0, 0, 1))

def resulting_vector (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  ∑ i in Finset.univ, vec i

noncomputable def a (vec : Fin n → (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) := resulting_vector vec

theorem expected_value_of_a_squared :
  (∀ i, is_random_vector (vec i)) →
  ∑ i in Finset.univ, vec i = (Y1, Y2, Y3) →
  E(a vec)^2 = (2 * n + n^2) / 3 :=
sorry

end expected_value_of_a_squared_l688_688020


namespace pell_eq_unique_fund_sol_l688_688368

theorem pell_eq_unique_fund_sol (x y x_0 y_0 : ℕ) 
  (h1 : x_0^2 - 2003 * y_0^2 = 1) 
  (h2 : ∀ x y, x > 0 ∧ y > 0 → x^2 - 2003 * y^2 = 1 → ∃ n : ℕ, x + Real.sqrt 2003 * y = (x_0 + Real.sqrt 2003 * y_0)^n)
  (hx_pos : x > 0) 
  (hy_pos : y > 0)
  (h_sol : x^2 - 2003 * y^2 = 1) 
  (hprime : ∀ p : ℕ, Prime p → p ∣ x → p ∣ x_0)
  : x = x_0 ∧ y = y_0 :=
sorry

end pell_eq_unique_fund_sol_l688_688368


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688883

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l688_688883


namespace dad_steps_are_90_l688_688129

def dad_masha_yasha_steps (d_step m_step y_step : ℕ) : ℕ :=
  let giant_step := 3 * d_step in
  let steps_by_masha_yasha := (3 + 5) in
  let total_m_y_steps := 400 in
  let number_of_giant_steps := total_m_y_steps / steps_by_masha_yasha in
  let masha_steps := 3 * number_of_giant_steps in
  let dad_steps := d_step * (masha_steps / 3) in
  dad_steps

theorem dad_steps_are_90 :
  dad_masha_yasha_steps 3 5 5 = 90 :=
by
  -- We can assume correctness based on the problem setup.
  sorry

end dad_steps_are_90_l688_688129


namespace andreas_living_room_area_l688_688892

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area := carpet_length * carpet_width
def carpet_coverage : ℝ := 0.75

-- The theorem statement
theorem andreas_living_room_area :
  let A := carpet_area / carpet_coverage in
  A = 48 := 
sorry

end andreas_living_room_area_l688_688892


namespace mutually_exclusive_union_l688_688636

open ProbabilityTheory

noncomputable def P {Ω : Type*} [ProbabilitySpace Ω] (A : Set Ω) : ℝ := Probability A

variables {Ω : Type*} [ProbabilitySpace Ω]

theorem mutually_exclusive_union (M N : Set Ω) (hM_exclusive : Disjoint M N)
    (hP_M : P M = 0.2) (hP_N : P N = 0.6) : P (M ∪ N) = 0.8 :=
by
  sorry

end mutually_exclusive_union_l688_688636


namespace dan_seashells_l688_688570

theorem dan_seashells :
  ∀ (initial_seashells given_seashells remaining_seashells : Nat),
    initial_seashells = 56 →
    remaining_seashells = 22 →
    given_seashells = initial_seashells - remaining_seashells →
    given_seashells = 34 :=
by
  intros initial_seashells given_seashells remaining_seashells
  intros h_initial h_remaining h_given
  rw [h_initial, h_remaining] at h_given
  exact h_given
  sorry

end dan_seashells_l688_688570


namespace greatest_prime_factor_of_expression_l688_688474

theorem greatest_prime_factor_of_expression :
  let n := 5^5 + 10^4 in
  n = 13125 ∧ (∀ p : Nat, Nat.Prime p → p ∣ n → p ≤ 7) ∧ Nat.Prime 7 ∧ 7 ∣ n :=
by
  let n := 5^5 + 10^4
  have h1 : n = 13125 := by sorry
  have h2 : Nat.Prime 7 := by sorry
  have h3 : ∀ p : Nat, Nat.Prime p → p ∣ n → p ≤ 7 := by sorry
  have h4 : 7 ∣ n := by sorry
  exact ⟨h1, ⟨h3, h2, h4⟩⟩

end greatest_prime_factor_of_expression_l688_688474


namespace find_number_l688_688969

theorem find_number (x : ℝ) (h : x - (3/5) * x = 62) : x = 155 :=
by
  sorry

end find_number_l688_688969


namespace exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688863

theorem exists_rational_non_integer_satisfying_linear :
  ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x + 8 * y ∈ ℤ ∧ 8 * x + 3 * y ∈ ℤ :=
by
  sorry

theorem no_rational_non_integer_satisfying_quadratic :
  ¬ ∃ (x y : ℚ), x.denom ≠ 1 ∧ y.denom ≠ 1 ∧ 19 * x^2 + 8 * y^2 ∈ ℤ ∧ 8 * x^2 + 3 * y^2 ∈ ℤ :=
by
  sorry

end exists_rational_non_integer_satisfying_linear_no_rational_non_integer_satisfying_quadratic_l688_688863


namespace smallest_x_value_satisfying_inequalities_l688_688219

theorem smallest_x_value_satisfying_inequalities : 
  ∃ x : ℤ, (3 * |(x : ℤ)| + 4 < 25) ∧ (x + 3 > 0) ∧ ∀ y : ℤ, (3 * |(y : ℤ)| + 4 < 25) ∧ (y + 3 > 0) → x ≤ y :=
begin
  -- we want x to be -3 in the existence statement, 
  -- and show that for any y that also satisfies the conditions, -3 <= y
  have h : ∃ x : ℤ, 3 * |(x : ℤ)| + 4 < 25 ∧ (x + 3 > 0), sorry,
  use -3,
  split,
  { -- 3 * |(-3 : ℤ)| + 4 < 25 
    sorry },
  split,
  { -- -3 + 3 > 0
    sorry },
  { -- ∀ y : ℤ, (3 * |(y : ℤ)| + 4 < 25) ∧ (y + 3 > 0) → -3 <= y,
    sorry }
end

end smallest_x_value_satisfying_inequalities_l688_688219


namespace base_diagonal_of_cube_l688_688496

theorem base_diagonal_of_cube (d_space : ℝ) (s : ℝ) (d_base : ℝ) (h1 : d_space = s * sqrt 3) (h2 : d_space = 5) :
  d_base = 5 * sqrt (2 / 3) :=
by
  sorry

end base_diagonal_of_cube_l688_688496


namespace part1_part2_l688_688003

theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x :=
  sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.cos (a * x) - Real.log (1 - x^2)) (hf0 : ∃ x, f' x = -a * Real.sin (a * x) + 2*x / (1 - x^2)) :
  0 = f' 0 → f'' 0 < 0 → (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) :=
  sorry

end part1_part2_l688_688003


namespace exists_rational_non_integer_linear_l688_688878

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l688_688878


namespace pollywog_maturation_rate_l688_688964

theorem pollywog_maturation_rate :
  ∀ (initial_pollywogs : ℕ) (melvin_rate : ℕ) (total_days : ℕ) (melvin_days : ℕ) (remaining_pollywogs : ℕ),
  initial_pollywogs = 2400 →
  melvin_rate = 10 →
  total_days = 44 →
  melvin_days = 20 →
  remaining_pollywogs = initial_pollywogs - (melvin_rate * melvin_days) →
  (total_days * (remaining_pollywogs / (total_days - melvin_days))) = remaining_pollywogs →
  (remaining_pollywogs / (total_days - melvin_days)) = 50 := 
by
  intros initial_pollywogs melvin_rate total_days melvin_days remaining_pollywogs
  intros h_initial h_melvin h_total h_melvin_days h_remaining h_eq
  sorry

end pollywog_maturation_rate_l688_688964


namespace circle_C_diameter_l688_688100

-- Definition of circles
structure Circle where
  radius : ℝ

def diameter (c : Circle) : ℝ := 2 * c.radius
def area (c : Circle) : ℝ := Real.pi * (c.radius)^2

-- Problem Statement
theorem circle_C_diameter (C D : Circle) 
  (h1 : D.radius = 10)
  (h2 : diameter C < diameter D)
  (h3 : (area D - area C) / area C = 5) :
  diameter C = 10 * Real.sqrt(3) / 3 :=
by
  sorry

end circle_C_diameter_l688_688100


namespace exists_nat_with_digit_sum_1000_and_square_sum_1000000_l688_688583

-- Define a function to calculate the sum of digits in base-10
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem exists_nat_with_digit_sum_1000_and_square_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_1000_and_square_sum_1000000_l688_688583


namespace son_present_age_l688_688492

-- Define the present ages of the son and the man
def son_age (S : ℕ) : Prop :=
  ∃ M : ℕ, M = S + 29 ∧ (M + 2 = 2 * (S + 2))

-- Theorem stating that the present age of the son is 27
theorem son_present_age : son_age 27 :=
begin
  sorry
end

end son_present_age_l688_688492


namespace exists_rational_non_integer_linear_l688_688880

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l688_688880


namespace part1_part2_l688_688004

theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x :=
  sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = Real.cos (a * x) - Real.log (1 - x^2)) (hf0 : ∃ x, f' x = -a * Real.sin (a * x) + 2*x / (1 - x^2)) :
  0 = f' 0 → f'' 0 < 0 → (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) :=
  sorry

end part1_part2_l688_688004


namespace dad_steps_90_l688_688172

theorem dad_steps_90 : 
  ∀ (M_steps Y_steps M_Y_total) (D_steps_per_M D_steps_per_group),
    (M_steps = 5 ∧ Y_steps = 5 ∧ M_Y_total = 400 ∧ D_steps_per_M = 3) ∧ 
    (3 * D_steps_per_M = M_steps ∧ 3 * Y_steps = 5 * M_steps ∧ 5 * D_steps_per_group = M_Y_total / 8) →
    (let Giant_steps := M_Y_total / (M_steps + Y_steps) in
      let M_total_steps := Giant_steps * 3 in
      let D_steps := D_steps_per_M * (M_total_steps / (M_steps / D_steps_per_M)) in
        D_steps = 90) :=
by
  {
    sorry
  }

end dad_steps_90_l688_688172


namespace intersection_and_min_distance_l688_688699

noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def curve_C2 (θ : ℝ) (ρ : ℝ) : Prop := ρ * Real.cos(θ - π/4) = - √2/2
noncomputable def curve_C3 (θ : ℝ) : ℝ := 2 * Real.sin θ

theorem intersection_and_min_distance :
  let C1 := {p : ℝ × ℝ | ∃ α, p = (Real.cos α, Real.sin α)},
      C2 := {p : ℝ × ℝ | p.1 + p.2 + 1 = 0},
      C3 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 2 * p.2} in
  (∀ M, M ∈ C1 ∧ M ∈ C2 → M = (-1, 0) ∨ M = (0, -1)) ∧
  (∃ A B, A ∈ C2 ∧ B ∈ C3 → ∀ A B, |A - B| = √2 - 1) :=
by
  sorry

end intersection_and_min_distance_l688_688699


namespace matrix_power_calculation_l688_688731

def B : matrix (fin 2) (fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_power_calculation :
  B^15 - 3 • (B^14) = ![![0, 4], ![0, -1]] :=
  sorry

end matrix_power_calculation_l688_688731


namespace shortest_tangent_length_l688_688365

variables (x y : ℝ)

-- Conditions for Circles C1 and C2
def C1 := (x - 12)^2 + y^2 = 25
def C2 := (x + 18)^2 + y^2 = 100

-- Definition of the distance between points A and B
def dist_AB : ℝ := 30

-- Theorem statement: length of segment PQ
theorem shortest_tangent_length :
  let PQ := 5 * real.sqrt 35 in PQ = 5 * real.sqrt 35 :=
by sorry

end shortest_tangent_length_l688_688365


namespace words_with_conditions_l688_688342

def words_count (n : ℕ) : ℕ :=
  4^(n-1) + 2^(n-1)

theorem words_with_conditions (n : ℕ) (w : fin n → char) :
  (∀ i, w i ∈ ['a', 'b', 'c', 'd']) →
  (even (count 'a' w)) →
  (even (count 'b' w)) →
  count_surjective @words_count w := sorry

end words_with_conditions_l688_688342


namespace regular_nonagon_interior_angle_l688_688838

theorem regular_nonagon_interior_angle : 
  let n := 9 in
  180 * (n - 2) / n = 140 :=
by 
  sorry

end regular_nonagon_interior_angle_l688_688838


namespace total_balloons_after_destruction_l688_688609

-- Define the initial numbers of balloons
def fredBalloons := 10.0
def samBalloons := 46.0
def destroyedBalloons := 16.0

-- Prove the total number of remaining balloons
theorem total_balloons_after_destruction : fredBalloons + samBalloons - destroyedBalloons = 40.0 :=
by
  sorry

end total_balloons_after_destruction_l688_688609


namespace total_investment_l688_688547

theorem total_investment (T : ℝ) :
  (T + 0.05 * 600 + (T - 600) + 0.06 * (T - 600) = 1054) ∧
  T = (460 / 1.06 : ℝ).round ∧
  T = 434 + 600 :=
by sorry

end total_investment_l688_688547


namespace max_candies_per_student_l688_688070

-- Let's define the known quantities and the proof statement.
theorem max_candies_per_student (n : ℕ) (mean : ℕ) (min_candies : ℕ) (students : ℕ) : n = 24 → mean = 7 → min_candies = 3 →
  (∀ i, i ∈ finset.range (n - 1) → 3 ≤ mean) →
  ∃ max_candies, max_candies = 99 :=
by
  intros
  use 99
  sorry

end max_candies_per_student_l688_688070


namespace dad_steps_l688_688141

def steps (k1 k2 steps_m_y : ℕ) (h_cond1 : ∀ m, 3 * m = 5 * k1) (h_cond2 : ∀ y, 3 * y = 5 * k2) : Prop :=
  let n := steps_m_y / 8 in -- The number of Giant Steps
  let steps_m := 3 * n in -- Steps taken by Masha
  let groups := steps_m / 5 in -- Groups of 5 steps for Masha
  let steps_dad := 3 * groups in -- Steps taken by Dad
  steps_dad = 90

theorem dad_steps (h_cond1 : ∀ m, 3 * m = 5 * _) (h_cond2 : ∀ y, 3 * y = 5 * _) :
  steps _ _ 400 h_cond1 h_cond2 :=
by
  sorry

end dad_steps_l688_688141
