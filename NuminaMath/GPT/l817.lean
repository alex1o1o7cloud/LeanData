import Mathlib

namespace NUMINAMATH_GPT_part_I_part_II_l817_81745

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 2 * x + 2 * a

theorem part_I (a : ℝ) :
  let x := Real.log 2
  ∃ I₁ I₂ : Set ℝ,
    (∀ x ∈ I₁, f a x > f a (Real.log 2)) ∧
    (∀ x ∈ I₂, f a x < f a (Real.log 2)) ∧
    I₁ = Set.Iio (Real.log 2) ∧
    I₂ = Set.Ioi (Real.log 2) ∧
    f a (Real.log 2) = 2 * (1 - Real.log 2 + a) :=
by sorry

theorem part_II (a : ℝ) (h : a > Real.log 2 - 1) (x : ℝ) (hx : 0 < x) :
  Real.exp x > x^2 - 2 * a * x + 1 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l817_81745


namespace NUMINAMATH_GPT_white_pieces_remaining_after_process_l817_81729

-- Definition to describe the removal process
def remove_every_second (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

-- Recursive function to model the process of removing pieces
def remaining_white_pieces (initial_white : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0     => initial_white
  | n + 1 => remaining_white_pieces (remove_every_second initial_white) n

-- Main theorem statement
theorem white_pieces_remaining_after_process :
  remaining_white_pieces 1990 4 = 124 :=
by
  sorry

end NUMINAMATH_GPT_white_pieces_remaining_after_process_l817_81729


namespace NUMINAMATH_GPT_cos_alpha_sub_beta_sin_alpha_l817_81718

open Real

variables (α β : ℝ)

-- Conditions:
-- 0 < α < π / 2
def alpha_in_first_quadrant := 0 < α ∧ α < π / 2

-- -π / 2 < β < 0
def beta_in_fourth_quadrant := -π / 2 < β ∧ β < 0

-- sin β = -5/13
def sin_beta := sin β = -5 / 13

-- tan(α - β) = 4/3
def tan_alpha_sub_beta := tan (α - β) = 4 / 3

-- Theorem statements (follows directly from the conditions and the equivalence):
theorem cos_alpha_sub_beta : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → cos (α - β) = 3 / 5 := sorry

theorem sin_alpha : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → sin α = 33 / 65 := sorry

end NUMINAMATH_GPT_cos_alpha_sub_beta_sin_alpha_l817_81718


namespace NUMINAMATH_GPT_inequality_proof_l817_81710

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1 / a < 1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l817_81710


namespace NUMINAMATH_GPT_parabola_problem_l817_81757

-- defining the geometric entities and conditions
variables {x y k x1 y1 x2 y2 : ℝ}

-- the definition for the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- the definition for point M
def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 2)

-- the definition for line passing through focus with slope k intersecting the parabola at A and B
def line_through_focus_and_k (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1))

-- the definition for vectors MA and MB having dot product zero
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 - 2 * (y1 + y2) + 4 = 0)

-- the main statement to be proved
theorem parabola_problem
  (h_parabola_A : parabola x1 y1)
  (h_parabola_B : parabola x2 y2)
  (h_point_M : point_M 0 2)
  (h_line_through_focus_and_k : line_through_focus_and_k x1 y1 x2 y2 k)
  (h_orthogonal_vectors : orthogonal_vectors x1 y1 x2 y2) :
  k = 1 :=
sorry

end NUMINAMATH_GPT_parabola_problem_l817_81757


namespace NUMINAMATH_GPT_numbers_not_necessarily_equal_l817_81743

theorem numbers_not_necessarily_equal (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) : 
  ¬(a = b ∧ b = c) := 
sorry

end NUMINAMATH_GPT_numbers_not_necessarily_equal_l817_81743


namespace NUMINAMATH_GPT_ike_mike_total_items_l817_81716

theorem ike_mike_total_items :
  ∃ (s d : ℕ), s + d = 7 ∧ 5 * s + 3/2 * d = 35 :=
by sorry

end NUMINAMATH_GPT_ike_mike_total_items_l817_81716


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l817_81732

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x * g x a h
noncomputable def G (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x - g x a h + (a - 1) * x 

theorem part_1 (a : ℝ) (h : 0 < a) :
  ∃(x : ℝ), x = -(a / (4 * Real.exp 1)) :=
sorry

theorem part_2 (a : ℝ) (h1 : 0 < a) : 
  (∃ x1 x2, (1/e) < x1 ∧ x1 < e ∧ (1/e) < x2 ∧ x2 < e ∧ G x1 a h1 = 0 ∧ G x2 a h1 = 0) 
    ↔ (a > (2 * Real.exp 1 - 1) / (2 * (Real.exp 1)^2 + 2 * Real.exp 1) ∧ a < 1/2) :=
sorry

theorem part_3 : 
  ∀ {x : ℝ}, 0 < x → Real.log x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l817_81732


namespace NUMINAMATH_GPT_hotel_profit_calculation_l817_81717

theorem hotel_profit_calculation
  (operations_expenses : ℝ)
  (meetings_fraction : ℝ) (events_fraction : ℝ) (rooms_fraction : ℝ)
  (meetings_tax_rate : ℝ) (meetings_commission_rate : ℝ)
  (events_tax_rate : ℝ) (events_commission_rate : ℝ)
  (rooms_tax_rate : ℝ) (rooms_commission_rate : ℝ)
  (total_profit : ℝ) :
  operations_expenses = 5000 →
  meetings_fraction = 5/8 →
  events_fraction = 3/10 →
  rooms_fraction = 11/20 →
  meetings_tax_rate = 0.10 →
  meetings_commission_rate = 0.05 →
  events_tax_rate = 0.08 →
  events_commission_rate = 0.06 →
  rooms_tax_rate = 0.12 →
  rooms_commission_rate = 0.03 →
  total_profit = (operations_expenses * (meetings_fraction + events_fraction + rooms_fraction)
                - (operations_expenses
                  + operations_expenses * (meetings_fraction * (meetings_tax_rate + meetings_commission_rate)
                  + events_fraction * (events_tax_rate + events_commission_rate)
                  + rooms_fraction * (rooms_tax_rate + rooms_commission_rate)))) ->
  total_profit = 1283.75 :=
by sorry

end NUMINAMATH_GPT_hotel_profit_calculation_l817_81717


namespace NUMINAMATH_GPT_technician_round_trip_completion_l817_81709

theorem technician_round_trip_completion (D : ℝ) (h0 : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center := 0.30 * D
  let traveled := to_center + from_center
  traveled / round_trip * 100 = 65 := 
by
  sorry

end NUMINAMATH_GPT_technician_round_trip_completion_l817_81709


namespace NUMINAMATH_GPT_cos_theta_correct_projection_correct_l817_81769

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (norm_a * norm_b)

noncomputable def projection (b : ℝ × ℝ) (cosθ : ℝ) : ℝ :=
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  norm_b * cosθ

theorem cos_theta_correct :
  cos_theta vec_a vec_b = 4 * Real.sqrt 65 / 65 :=
by
  sorry

theorem projection_correct :
  projection vec_b (cos_theta vec_a vec_b) = 8 * Real.sqrt 13 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_correct_projection_correct_l817_81769


namespace NUMINAMATH_GPT_largest_b_for_box_volume_l817_81767

theorem largest_b_for_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) 
                                 (h4 : c = 3) (volume : a * b * c = 360) : 
    b = 8 := 
sorry

end NUMINAMATH_GPT_largest_b_for_box_volume_l817_81767


namespace NUMINAMATH_GPT_point_on_x_axis_coordinates_l817_81711

-- Define the conditions
def lies_on_x_axis (M : ℝ × ℝ) : Prop := M.snd = 0

-- State the problem
theorem point_on_x_axis_coordinates (a : ℝ) :
  lies_on_x_axis (a + 3, a + 1) → (a = -1) ∧ ((a + 3, 0) = (2, 0)) :=
by
  intro h
  rw [lies_on_x_axis] at h
  sorry

end NUMINAMATH_GPT_point_on_x_axis_coordinates_l817_81711


namespace NUMINAMATH_GPT_N_properties_l817_81730

def N : ℕ := 3625

theorem N_properties :
  (N % 32 = 21) ∧ (N % 125 = 0) ∧ (N^2 % 8000 = N % 8000) :=
by
  sorry

end NUMINAMATH_GPT_N_properties_l817_81730


namespace NUMINAMATH_GPT_minimum_value_of_functions_l817_81747

def linear_fn (a b c: ℝ) := a ≠ 0 
def f (a b: ℝ) (x: ℝ) := a * x + b 
def g (a c: ℝ) (x: ℝ) := a * x + c

theorem minimum_value_of_functions (a b c: ℝ) (hx: linear_fn a b c) :
  (∀ x: ℝ, 3 * (f a b x)^2 + 2 * g a c x ≥ -19 / 6) → (∀ x: ℝ, 3 * (g a c x)^2 + 2 * f a b x ≥ 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_functions_l817_81747


namespace NUMINAMATH_GPT_math_problem_l817_81720

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end NUMINAMATH_GPT_math_problem_l817_81720


namespace NUMINAMATH_GPT_intersection_M_N_l817_81781

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | ∃ k : ℕ, x = 2 * k}

theorem intersection_M_N :
  M ∩ N = {2, 4, 8} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l817_81781


namespace NUMINAMATH_GPT_fibonacci_inequality_l817_81786

def Fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | n + 2 => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_inequality (n : ℕ) (h : n > 0) : 
  Real.sqrt (Fibonacci (n+1)) > 1 + 1 / Real.sqrt (Fibonacci n) := 
sorry

end NUMINAMATH_GPT_fibonacci_inequality_l817_81786


namespace NUMINAMATH_GPT_general_term_of_sequence_l817_81726

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 9
  | 3 => 14
  | 4 => 21
  | 5 => 30
  | _ => sorry

theorem general_term_of_sequence :
  ∀ n : ℕ, seq n = 5 + n^2 :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l817_81726


namespace NUMINAMATH_GPT_one_fourths_in_five_eighths_l817_81734

theorem one_fourths_in_five_eighths : (5/8 : ℚ) / (1/4) = (5/2 : ℚ) := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_one_fourths_in_five_eighths_l817_81734


namespace NUMINAMATH_GPT_tan_two_alpha_l817_81788

theorem tan_two_alpha (α β : ℝ) (h₁ : Real.tan (α - β) = -3/2) (h₂ : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := 
sorry

end NUMINAMATH_GPT_tan_two_alpha_l817_81788


namespace NUMINAMATH_GPT_most_suitable_survey_l817_81771

-- Define the options as a type
inductive SurveyOption
| A -- Understanding the crash resistance of a batch of cars
| B -- Surveying the awareness of the "one helmet, one belt" traffic regulations among citizens in our city
| C -- Surveying the service life of light bulbs produced by a factory
| D -- Surveying the quality of components of the latest stealth fighter in our country

-- Define a function determining the most suitable for a comprehensive survey
def mostSuitableForCensus : SurveyOption :=
  SurveyOption.D

-- Theorem statement that Option D is the most suitable for a comprehensive survey
theorem most_suitable_survey :
  mostSuitableForCensus = SurveyOption.D :=
  sorry

end NUMINAMATH_GPT_most_suitable_survey_l817_81771


namespace NUMINAMATH_GPT_non_red_fraction_l817_81761

-- Define the conditions
def cube_edge : ℕ := 4
def num_cubes : ℕ := 64
def num_red_cubes : ℕ := 48
def num_white_cubes : ℕ := 12
def num_blue_cubes : ℕ := 4
def total_surface_area : ℕ := 6 * (cube_edge * cube_edge)

-- Define the non-red surface area exposed
def white_cube_exposed_area : ℕ := 12
def blue_cube_exposed_area : ℕ := 0

-- Calculating non-red area
def non_red_surface_area : ℕ := white_cube_exposed_area + blue_cube_exposed_area

-- The theorem to prove
theorem non_red_fraction (cube_edge : ℕ) (num_cubes : ℕ) (num_red_cubes : ℕ) 
  (num_white_cubes : ℕ) (num_blue_cubes : ℕ) (total_surface_area : ℕ) 
  (non_red_surface_area : ℕ) : 
  (non_red_surface_area : ℚ) / (total_surface_area : ℚ) = 1 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_non_red_fraction_l817_81761


namespace NUMINAMATH_GPT_value_of_expression_l817_81755

theorem value_of_expression (a b : ℝ) (h1 : a ≠ b)
  (h2 : a^2 + 2 * a - 2022 = 0)
  (h3 : b^2 + 2 * b - 2022 = 0) :
  a^2 + 4 * a + 2 * b = 2018 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l817_81755


namespace NUMINAMATH_GPT_polynomial_no_in_interval_l817_81748

theorem polynomial_no_in_interval (P : Polynomial ℤ) (x₁ x₂ x₃ x₄ x₅ : ℤ) :
  (-- Conditions
  P.eval x₁ = 5 ∧ P.eval x₂ = 5 ∧ P.eval x₃ = 5 ∧ P.eval x₄ = 5 ∧ P.eval x₅ = 5 ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅)
  -- No x such that -6 <= P(x) <= 4 or 6 <= P(x) <= 16
  → (∀ x : ℤ, ¬(-6 ≤ P.eval x ∧ P.eval x ≤ 4) ∧ ¬(6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_no_in_interval_l817_81748


namespace NUMINAMATH_GPT_quadratic_solution_range_l817_81768

noncomputable def quadratic_inequality_real_solution (c : ℝ) : Prop :=
  0 < c ∧ c < 16

theorem quadratic_solution_range :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 8 * x + c < 0) ↔ quadratic_inequality_real_solution c :=
by
  intro c
  simp only [quadratic_inequality_real_solution]
  sorry

end NUMINAMATH_GPT_quadratic_solution_range_l817_81768


namespace NUMINAMATH_GPT_yearly_return_500_correct_l817_81721

noncomputable def yearly_return_500_investment : ℝ :=
  let total_investment : ℝ := 500 + 1500
  let combined_yearly_return : ℝ := 0.10 * total_investment
  let yearly_return_1500 : ℝ := 0.11 * 1500
  let yearly_return_500 : ℝ := combined_yearly_return - yearly_return_1500
  (yearly_return_500 / 500) * 100

theorem yearly_return_500_correct : yearly_return_500_investment = 7 :=
by
  sorry

end NUMINAMATH_GPT_yearly_return_500_correct_l817_81721


namespace NUMINAMATH_GPT_maximum_triangles_in_right_angle_triangle_l817_81785

-- Definition of grid size and right-angled triangle on graph paper
def grid_size : Nat := 7

-- Definition of the vertices of the right-angled triangle
def vertices : List (Nat × Nat) := [(0,0), (grid_size,0), (0,grid_size)]

-- Total number of unique triangles that can be identified
theorem maximum_triangles_in_right_angle_triangle (grid_size : Nat) (vertices : List (Nat × Nat)) : 
  Nat :=
  if vertices = [(0,0), (grid_size,0), (0,grid_size)] then 28 else 0

end NUMINAMATH_GPT_maximum_triangles_in_right_angle_triangle_l817_81785


namespace NUMINAMATH_GPT_smallest_three_digit_integer_l817_81793

theorem smallest_three_digit_integer (n : ℕ) (h : 75 * n ≡ 225 [MOD 345]) (hne : n ≥ 100) (hn : n < 1000) : n = 118 :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_integer_l817_81793


namespace NUMINAMATH_GPT_cassidy_number_of_posters_l817_81770

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end NUMINAMATH_GPT_cassidy_number_of_posters_l817_81770


namespace NUMINAMATH_GPT_no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l817_81702

theorem no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0 :
  ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
sorry

end NUMINAMATH_GPT_no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l817_81702


namespace NUMINAMATH_GPT_sum_inequality_l817_81725

theorem sum_inequality 
  {a b c : ℝ}
  (h : a + b + c = 3) : 
  (1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2)) ≤ 3 / 2 := 
sorry

end NUMINAMATH_GPT_sum_inequality_l817_81725


namespace NUMINAMATH_GPT_line_tangent_parabola_unique_d_l817_81796

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end NUMINAMATH_GPT_line_tangent_parabola_unique_d_l817_81796


namespace NUMINAMATH_GPT_volume_ratio_of_sphere_surface_area_l817_81795

theorem volume_ratio_of_sphere_surface_area 
  {V1 V2 V3 : ℝ} 
  (h : V1/V3 = 1/27 ∧ V2/V3 = 8/27) 
  : V1 + V2 = (1/3) * V3 := 
sorry

end NUMINAMATH_GPT_volume_ratio_of_sphere_surface_area_l817_81795


namespace NUMINAMATH_GPT_john_using_three_colors_l817_81750

theorem john_using_three_colors {total_paint liters_per_color : ℕ} 
    (h1 : total_paint = 15) 
    (h2 : liters_per_color = 5) :
    total_ppaint / liters_per_color = 3 := 
by
  sorry

end NUMINAMATH_GPT_john_using_three_colors_l817_81750


namespace NUMINAMATH_GPT_angle_ABC_is_50_l817_81744

theorem angle_ABC_is_50
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a = 90)
  (h2 : b = 60)
  (h3 : a + b + c = 200): c = 50 := by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_angle_ABC_is_50_l817_81744


namespace NUMINAMATH_GPT_fraction_meaningful_l817_81733

theorem fraction_meaningful (x : ℝ) : x - 5 ≠ 0 ↔ x ≠ 5 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l817_81733


namespace NUMINAMATH_GPT_exists_n_not_represented_l817_81756

theorem exists_n_not_represented (a b c d : ℤ) (a_gt_14 : a > 14)
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) :
  ∃ (n : ℕ), ¬ ∃ (x y z : ℤ), n = x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
sorry

end NUMINAMATH_GPT_exists_n_not_represented_l817_81756


namespace NUMINAMATH_GPT_overall_effect_l817_81705
noncomputable def effect (x : ℚ) : ℚ :=
  ((x * (5 / 6)) * (1 / 10)) + (2 / 3)

theorem overall_effect (x : ℚ) : effect x = (x * (5 / 6) * (1 / 10)) + (2 / 3) :=
  by
  sorry

-- Prove for initial number 1
example : effect 1 = 3 / 4 :=
  by
  sorry

end NUMINAMATH_GPT_overall_effect_l817_81705


namespace NUMINAMATH_GPT_root_of_equation_l817_81797

theorem root_of_equation (a : ℝ) (h : a^2 * (-1)^2 + 2011 * a * (-1) - 2012 = 0) : 
  a = 2012 ∨ a = -1 :=
by sorry

end NUMINAMATH_GPT_root_of_equation_l817_81797


namespace NUMINAMATH_GPT_max_n_arithmetic_seq_sum_neg_l817_81738

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + ((n - 1) * d)

-- Define the terms of the sequence
def a₃ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 3
def a₆ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 6
def a₇ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 7

-- Condition: a₆ is the geometric mean of a₃ and a₇
def geometric_mean_condition (a₁ : ℤ) : Prop :=
  (a₃ a₁) * (a₇ a₁) = (a₆ a₁) * (a₆ a₁)

-- Sum of the first n terms of the arithmetic sequence
def S_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

-- The goal: the maximum value of n for which S_n < 0
theorem max_n_arithmetic_seq_sum_neg : 
  ∃ n : ℕ, ∀ k : ℕ, geometric_mean_condition (-13) →  S_n (-13) 2 k < 0 → n ≤ 13 := 
sorry

end NUMINAMATH_GPT_max_n_arithmetic_seq_sum_neg_l817_81738


namespace NUMINAMATH_GPT_complement_of_intersection_l817_81713

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_of_intersection (AuB AcB : Set ℤ) :
  (A ∪ B) = AuB ∧ (A ∩ B) = AcB → 
  A ∪ B = ∅ ∨ A ∪ B = AuB → 
  (AuB \ AcB) = {-1, 1} :=
by
  -- Proof construction method placeholder.
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l817_81713


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l817_81779

-- Defining the problem in Lean 4 terms.
noncomputable def geom_seq_cond (a : ℕ → ℕ) (m n p q : ℕ) : Prop :=
  m + n = p + q → a m * a n = a p * a q

theorem necessary_but_not_sufficient (a : ℕ → ℕ) (m n p q : ℕ) (h : m + n = p + q) :
  geom_seq_cond a m n p q → ∃ b : ℕ → ℕ, (∀ n, b n = 0 → (m + n = p + q → b m * b n = b p * b q))
    ∧ (∀ n, ¬ (b n = 0 → ∀ q, b (q+1) / b q = b (q+1) / b q)) := sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l817_81779


namespace NUMINAMATH_GPT_bounded_area_l817_81774

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

def g (y : ℝ) : ℝ := y + 1

theorem bounded_area : 
  (∫ y in (0:ℝ)..(1:ℝ), (g y - f (g y))) = (5/8 : ℝ) := by
  sorry

end NUMINAMATH_GPT_bounded_area_l817_81774


namespace NUMINAMATH_GPT_find_n_l817_81776

theorem find_n (n a b : ℕ) (h1 : n ≥ 2)
  (h2 : n = a^2 + b^2)
  (h3 : a = Nat.minFac n)
  (h4 : b ∣ n) : n = 8 ∨ n = 20 := 
sorry

end NUMINAMATH_GPT_find_n_l817_81776


namespace NUMINAMATH_GPT_rate_per_sqm_l817_81798

theorem rate_per_sqm (length width : ℝ) (cost : ℝ) (Area : ℝ := length * width) (rate : ℝ := cost / Area) 
  (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 8250) : 
  rate = 400 :=
sorry

end NUMINAMATH_GPT_rate_per_sqm_l817_81798


namespace NUMINAMATH_GPT_profit_ratio_l817_81782

variables (P_s : ℝ)

theorem profit_ratio (h1 : 21 * (7 / 3) + 3 * P_s = 175) : P_s / 21 = 2 :=
by
  sorry

end NUMINAMATH_GPT_profit_ratio_l817_81782


namespace NUMINAMATH_GPT_molecular_weight_calculated_l817_81749

def atomic_weight_Ba : ℚ := 137.33
def atomic_weight_O  : ℚ := 16.00
def atomic_weight_H  : ℚ := 1.01

def molecular_weight_compound : ℚ :=
  (1 * atomic_weight_Ba) + (2 * atomic_weight_O) + (2 * atomic_weight_H)

theorem molecular_weight_calculated :
  molecular_weight_compound = 171.35 :=
by {
  sorry
}

end NUMINAMATH_GPT_molecular_weight_calculated_l817_81749


namespace NUMINAMATH_GPT_new_ratio_of_milk_to_water_l817_81752

theorem new_ratio_of_milk_to_water
  (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
  (h_total_volume : total_volume = 45)
  (h_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1)
  (h_added_water : added_water = 11) :
  let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let new_water := initial_water + added_water
  let gcd := Nat.gcd initial_milk new_water
  (initial_milk / gcd : ℕ) = 9 ∧ (new_water / gcd : ℕ) = 5 :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_of_milk_to_water_l817_81752


namespace NUMINAMATH_GPT_perimeter_of_square_l817_81789

theorem perimeter_of_square (A : ℝ) (hA : A = 400) : exists P : ℝ, P = 80 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l817_81789


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_l817_81753

noncomputable def max_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  (|a 3| = |a 11| ∧ 
   (∃ d : ℤ, d < 0 ∧ 
   (∀ n, a (n + 1) = a n + d) ∧ 
   (∀ m, S m = (m * (2 * a 1 + (m - 1) * d)) / 2)) →
   ((n = 6) ∨ (n = 7)))

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  max_sum_n a S 6 ∨ max_sum_n a S 7 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_l817_81753


namespace NUMINAMATH_GPT_bags_of_soil_needed_l817_81736

theorem bags_of_soil_needed
  (length width height : ℕ)
  (beds : ℕ)
  (volume_per_bag : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_beds : beds = 2)
  (h_volume_per_bag : volume_per_bag = 4) :
  (length * width * height * beds) / volume_per_bag = 16 :=
by
  sorry

end NUMINAMATH_GPT_bags_of_soil_needed_l817_81736


namespace NUMINAMATH_GPT_fraction_simplified_to_p_l817_81762

theorem fraction_simplified_to_p (q : ℕ) (hq_pos : 0 < q) (gcd_cond : Nat.gcd 4047 q = 1) :
    (2024 / 2023) - (2023 / 2024) = 4047 / q := sorry

end NUMINAMATH_GPT_fraction_simplified_to_p_l817_81762


namespace NUMINAMATH_GPT_unit_price_of_each_chair_is_42_l817_81737

-- Definitions from conditions
def total_cost_desks (unit_price_desk : ℕ) (number_desks : ℕ) : ℕ := unit_price_desk * number_desks
def remaining_cost_chairs (total_cost : ℕ) (cost_desks : ℕ) : ℕ := total_cost - cost_desks
def unit_price_chairs (remaining_cost : ℕ) (number_chairs : ℕ) : ℕ := remaining_cost / number_chairs

-- Given conditions
def unit_price_desk := 180
def number_desks := 5
def total_cost := 1236
def number_chairs := 8

-- The question: determining the unit price of each chair
theorem unit_price_of_each_chair_is_42 : 
  unit_price_chairs (remaining_cost_chairs total_cost (total_cost_desks unit_price_desk number_desks)) number_chairs = 42 := sorry

end NUMINAMATH_GPT_unit_price_of_each_chair_is_42_l817_81737


namespace NUMINAMATH_GPT_seconds_in_9_point_4_minutes_l817_81799

def seconds_in_minute : ℕ := 60
def minutes : ℝ := 9.4
def expected_seconds : ℝ := 564

theorem seconds_in_9_point_4_minutes : minutes * seconds_in_minute = expected_seconds :=
by 
  sorry

end NUMINAMATH_GPT_seconds_in_9_point_4_minutes_l817_81799


namespace NUMINAMATH_GPT_max_and_min_of_z_in_G_l817_81703

def z (x y : ℝ) : ℝ := x^2 + y^2 - 2*x*y - x - 2*y

def G (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

theorem max_and_min_of_z_in_G :
  (∃ (x y : ℝ), G x y ∧ z x y = 12) ∧ (∃ (x y : ℝ), G x y ∧ z x y = -1/4) :=
sorry

end NUMINAMATH_GPT_max_and_min_of_z_in_G_l817_81703


namespace NUMINAMATH_GPT_vector_combination_l817_81739

open Complex

def z1 : ℂ := -1 + I
def z2 : ℂ := 1 + I
def z3 : ℂ := 1 + 4 * I

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 4)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

def x : ℝ := sorry
def y : ℝ := sorry

theorem vector_combination (hx : OC = ( - x + y, x + y )) : 
    x + y = 4 :=
by
    sorry

end NUMINAMATH_GPT_vector_combination_l817_81739


namespace NUMINAMATH_GPT_socks_pair_count_l817_81722

theorem socks_pair_count :
  let white := 5
  let brown := 5
  let blue := 3
  let green := 2
  (white * brown) + (white * blue) + (white * green) + (brown * blue) + (brown * green) + (blue * green) = 81 :=
by
  intros
  sorry

end NUMINAMATH_GPT_socks_pair_count_l817_81722


namespace NUMINAMATH_GPT_fixed_point_sum_l817_81712

theorem fixed_point_sum (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (m, n) = (1, a * (1-1) + 2)) : m + n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_fixed_point_sum_l817_81712


namespace NUMINAMATH_GPT_range_of_a_l817_81731

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l817_81731


namespace NUMINAMATH_GPT_find_f2023_l817_81735

-- Define the function and conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def satisfies_condition (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Define the main statement to prove that f(2023) = 2 given conditions
theorem find_f2023 (f : ℝ → ℝ)
  (h1 : is_even f)
  (h2 : satisfies_condition f)
  (h3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x) :
  f 2023 = 2 :=
sorry

end NUMINAMATH_GPT_find_f2023_l817_81735


namespace NUMINAMATH_GPT_number_of_outfits_l817_81784

-- Define the number of shirts, pants, and jacket options.
def shirts : Nat := 8
def pants : Nat := 5
def jackets : Nat := 3

-- The theorem statement for the total number of outfits.
theorem number_of_outfits : shirts * pants * jackets = 120 := 
by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l817_81784


namespace NUMINAMATH_GPT_frustum_volume_correct_l817_81728

noncomputable def volume_frustum (base_edge_original base_edge_smaller altitude_original altitude_smaller : ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let volume_smaller := (1 / 3) * base_area_smaller * altitude_smaller
  volume_original - volume_smaller

theorem frustum_volume_correct :
  volume_frustum 16 8 10 5 = 2240 / 3 :=
by
  have h1 : volume_frustum 16 8 10 5 = 
    (1 / 3) * (16^2) * 10 - (1 / 3) * (8^2) * 5 := rfl
  simp only [pow_two] at h1
  norm_num at h1
  exact h1

end NUMINAMATH_GPT_frustum_volume_correct_l817_81728


namespace NUMINAMATH_GPT_tan_neg_240_eq_neg_sqrt_3_l817_81714

theorem tan_neg_240_eq_neg_sqrt_3 : Real.tan (-4 * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_neg_240_eq_neg_sqrt_3_l817_81714


namespace NUMINAMATH_GPT_calculate_expression_l817_81790

theorem calculate_expression :
  ((-1 -2 -3 -4 -5 -6 -7 -8 -9 -10) * (1 -2 +3 -4 +5 -6 +7 -8 +9 -10) = 275) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l817_81790


namespace NUMINAMATH_GPT_value_of_T_l817_81740

theorem value_of_T (S : ℝ) (T : ℝ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) : T = 96 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_T_l817_81740


namespace NUMINAMATH_GPT_desired_salt_percentage_is_ten_percent_l817_81715

-- Define the initial conditions
def initial_pure_water_volume : ℝ := 100
def saline_solution_percentage : ℝ := 0.25
def added_saline_volume : ℝ := 66.67
def total_volume : ℝ := initial_pure_water_volume + added_saline_volume
def added_salt : ℝ := saline_solution_percentage * added_saline_volume
def desired_salt_percentage (P : ℝ) : Prop := added_salt = P * total_volume

-- State the theorem and its result
theorem desired_salt_percentage_is_ten_percent (P : ℝ) (h : desired_salt_percentage P) : P = 0.1 :=
sorry

end NUMINAMATH_GPT_desired_salt_percentage_is_ten_percent_l817_81715


namespace NUMINAMATH_GPT_combined_selling_price_l817_81766

theorem combined_selling_price :
  let cost_cycle := 2300
  let cost_scooter := 12000
  let cost_motorbike := 25000
  let loss_cycle := 0.30
  let profit_scooter := 0.25
  let profit_motorbike := 0.15
  let selling_price_cycle := cost_cycle - (loss_cycle * cost_cycle)
  let selling_price_scooter := cost_scooter + (profit_scooter * cost_scooter)
  let selling_price_motorbike := cost_motorbike + (profit_motorbike * cost_motorbike)
  selling_price_cycle + selling_price_scooter + selling_price_motorbike = 45360 := 
by
  sorry

end NUMINAMATH_GPT_combined_selling_price_l817_81766


namespace NUMINAMATH_GPT_arithmetic_seq_a4_value_l817_81719

theorem arithmetic_seq_a4_value
  (a : ℕ → ℤ)
  (h : 4 * a 3 + a 11 - 3 * a 5 = 10) :
  a 4 = 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_a4_value_l817_81719


namespace NUMINAMATH_GPT_decimal_equivalent_of_squared_fraction_l817_81763

theorem decimal_equivalent_of_squared_fraction : (1 / 5 : ℝ)^2 = 0.04 :=
by
  sorry

end NUMINAMATH_GPT_decimal_equivalent_of_squared_fraction_l817_81763


namespace NUMINAMATH_GPT_y_intercept_of_line_l817_81746

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end NUMINAMATH_GPT_y_intercept_of_line_l817_81746


namespace NUMINAMATH_GPT_greatest_length_measures_exactly_l817_81783

theorem greatest_length_measures_exactly 
    (a b c : ℕ) 
    (ha : a = 700)
    (hb : b = 385)
    (hc : c = 1295) : 
    Nat.gcd (Nat.gcd a b) c = 35 := 
by
  sorry

end NUMINAMATH_GPT_greatest_length_measures_exactly_l817_81783


namespace NUMINAMATH_GPT_number_of_marbles_l817_81723

theorem number_of_marbles (T : ℕ) (h1 : 12 ≤ T) : 
  (T - 12) * (T - 12) * 16 = 9 * T * T → T = 48 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_marbles_l817_81723


namespace NUMINAMATH_GPT_evaluate_expression_l817_81741

-- Define the ceiling of square roots for the given numbers
def ceil_sqrt_3 := 2
def ceil_sqrt_27 := 6
def ceil_sqrt_243 := 16

-- Main theorem statement
theorem evaluate_expression :
  ceil_sqrt_3 + ceil_sqrt_27 * 2 + ceil_sqrt_243 = 30 :=
by
  -- Sorry to indicate that the proof is skipped
  sorry

end NUMINAMATH_GPT_evaluate_expression_l817_81741


namespace NUMINAMATH_GPT_intersecting_lines_l817_81754

theorem intersecting_lines (a b : ℝ) (h1 : 3 = (1 / 3) * 6 + a) (h2 : 6 = (1 / 3) * 3 + b) : a + b = 6 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l817_81754


namespace NUMINAMATH_GPT_diving_club_capacity_l817_81780

theorem diving_club_capacity :
  (3 * ((2 * 5 + 4 * 2) * 5) = 270) :=
by
  sorry

end NUMINAMATH_GPT_diving_club_capacity_l817_81780


namespace NUMINAMATH_GPT_inequality_1_solution_set_inequality_2_solution_set_l817_81765

theorem inequality_1_solution_set (x : ℝ) : 
  (2 + 3 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 2) := 
by sorry

theorem inequality_2_solution_set (x : ℝ) :
  (x * (3 - x) ≤ x * (x + 2) - 1) ↔ (x ≤ -1/2 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_GPT_inequality_1_solution_set_inequality_2_solution_set_l817_81765


namespace NUMINAMATH_GPT_minimize_expense_l817_81764

def price_after_first_discount (initial_price : ℕ) (discount : ℕ) : ℕ :=
  initial_price * (100 - discount) / 100

def final_price_set1 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 15
  let step2 := price_after_first_discount step1 25
  price_after_first_discount step2 10

def final_price_set2 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 25
  let step2 := price_after_first_discount step1 10
  price_after_first_discount step2 10

theorem minimize_expense (initial_price : ℕ) (h : initial_price = 12000) :
  final_price_set1 initial_price = 6885 ∧ final_price_set2 initial_price = 7290 ∧
  final_price_set1 initial_price < final_price_set2 initial_price := by
  sorry

end NUMINAMATH_GPT_minimize_expense_l817_81764


namespace NUMINAMATH_GPT_determine_constant_l817_81706

/-- If the function f(x) = a * sin x + 3 * cos x has a maximum value of 5,
then the constant a must be ± 4. -/
theorem determine_constant (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x + 3 * Real.cos x ≤ 5) :
  a = 4 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_determine_constant_l817_81706


namespace NUMINAMATH_GPT_person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l817_81708

-- Define the probability of hitting the target for Person A and Person B
def p_hit_A : ℚ := 2 / 3
def p_hit_B : ℚ := 3 / 4

-- Define the complementary probabilities (missing the target)
def p_miss_A := 1 - p_hit_A
def p_miss_B := 1 - p_hit_B

-- Prove the probability that Person A, shooting 4 times, misses the target at least once
theorem person_A_misses_at_least_once_in_4_shots :
  (1 - (p_hit_A ^ 4)) = 65 / 81 :=
by 
  sorry

-- Prove the probability that Person B stops shooting exactly after 5 shots
-- due to missing the target consecutively 2 times
theorem person_B_stops_after_5_shots_due_to_2_consecutive_misses :
  (p_hit_B * p_hit_B * p_miss_B * (p_miss_B * p_miss_B)) = 45 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l817_81708


namespace NUMINAMATH_GPT_pradeep_failed_by_25_marks_l817_81760

theorem pradeep_failed_by_25_marks :
  (35 / 100 * 600 : ℝ) - 185 = 25 :=
by
  sorry

end NUMINAMATH_GPT_pradeep_failed_by_25_marks_l817_81760


namespace NUMINAMATH_GPT_problem_positive_l817_81727

theorem problem_positive : ∀ x : ℝ, x < 0 → -3 * x⁻¹ > 0 :=
by 
  sorry

end NUMINAMATH_GPT_problem_positive_l817_81727


namespace NUMINAMATH_GPT_quadratic_solutions_1_quadratic_k_value_and_solutions_l817_81758

-- Problem (Ⅰ):
theorem quadratic_solutions_1 {x : ℝ} :
  x^2 + 6 * x + 5 = 0 ↔ x = -5 ∨ x = -1 :=
sorry

-- Problem (Ⅱ):
theorem quadratic_k_value_and_solutions {x k : ℝ} (x1 x2 : ℝ) :
  x1 + x2 = 3 ∧ x1 * x2 = k ∧ (x1 - 1) * (x2 - 1) = -6 ↔ (k = -4 ∧ (x = 4 ∨ x = -1)) :=
sorry

end NUMINAMATH_GPT_quadratic_solutions_1_quadratic_k_value_and_solutions_l817_81758


namespace NUMINAMATH_GPT_a_when_a_minus_1_no_reciprocal_l817_81794

theorem a_when_a_minus_1_no_reciprocal (a : ℝ) (h : ¬ ∃ b : ℝ, (a - 1) * b = 1) : a = 1 := 
by
  sorry

end NUMINAMATH_GPT_a_when_a_minus_1_no_reciprocal_l817_81794


namespace NUMINAMATH_GPT_projectile_height_reaches_35_l817_81700

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end NUMINAMATH_GPT_projectile_height_reaches_35_l817_81700


namespace NUMINAMATH_GPT_evaluate_expression_l817_81707

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l817_81707


namespace NUMINAMATH_GPT_power_i_2015_l817_81778

theorem power_i_2015 (i : ℂ) (hi : i^2 = -1) : i^2015 = -i :=
by
  have h1 : i^4 = 1 := by sorry
  have h2 : 2015 = 4 * 503 + 3 := by norm_num
  sorry

end NUMINAMATH_GPT_power_i_2015_l817_81778


namespace NUMINAMATH_GPT_candy_left_l817_81751

variable (x : ℕ)

theorem candy_left (x : ℕ) : x - (18 + 7) = x - 25 :=
by sorry

end NUMINAMATH_GPT_candy_left_l817_81751


namespace NUMINAMATH_GPT_sam_total_cans_l817_81742

theorem sam_total_cans (bags_saturday bags_sunday bags_total cans_per_bag total_cans : ℕ)
    (h1 : bags_saturday = 3)
    (h2 : bags_sunday = 4)
    (h3 : bags_total = bags_saturday + bags_sunday)
    (h4 : cans_per_bag = 9)
    (h5 : total_cans = bags_total * cans_per_bag) : total_cans = 63 :=
sorry

end NUMINAMATH_GPT_sam_total_cans_l817_81742


namespace NUMINAMATH_GPT_legs_paws_in_pool_l817_81787

def total_legs_paws (num_humans : Nat) (human_legs : Nat) (num_dogs : Nat) (dog_paws : Nat) : Nat :=
  (num_humans * human_legs) + (num_dogs * dog_paws)

theorem legs_paws_in_pool :
  total_legs_paws 2 2 5 4 = 24 := by
  sorry

end NUMINAMATH_GPT_legs_paws_in_pool_l817_81787


namespace NUMINAMATH_GPT_symmetric_pentominoes_count_l817_81701

-- Assume we have exactly fifteen pentominoes
def num_pentominoes : ℕ := 15

-- Define the number of pentominoes with particular symmetrical properties
def num_reflectional_symmetry : ℕ := 8
def num_rotational_symmetry : ℕ := 3
def num_both_symmetries : ℕ := 2

-- The theorem we wish to prove
theorem symmetric_pentominoes_count 
  (n_p : ℕ) (n_r : ℕ) (n_b : ℕ) (n_tot : ℕ)
  (h1 : n_p = num_pentominoes)
  (h2 : n_r = num_reflectional_symmetry)
  (h3 : n_b = num_both_symmetries)
  (h4 : n_tot = n_r + num_rotational_symmetry - n_b) :
  n_tot = 9 := 
sorry

end NUMINAMATH_GPT_symmetric_pentominoes_count_l817_81701


namespace NUMINAMATH_GPT_joel_garden_size_l817_81791

-- Definitions based on the conditions
variable (G : ℕ) -- G is the size of Joel's garden.

-- Condition 1: Half of the garden is for fruits.
def half_garden_fruits (G : ℕ) := G / 2

-- Condition 2: Half of the garden is for vegetables.
def half_garden_vegetables (G : ℕ) := G / 2

-- Condition 3: A quarter of the fruit section is used for strawberries.
def quarter_fruit_section (G : ℕ) := (half_garden_fruits G) / 4

-- Condition 4: The quarter for strawberries takes up 8 square feet.
axiom strawberry_section : quarter_fruit_section G = 8

-- Hypothesis: The size of Joel's garden is 64 square feet.
theorem joel_garden_size : G = 64 :=
by
  -- Insert the logical progression of the proof here.
  sorry

end NUMINAMATH_GPT_joel_garden_size_l817_81791


namespace NUMINAMATH_GPT_factorize_expression_l817_81772

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l817_81772


namespace NUMINAMATH_GPT_repeating_decimals_count_l817_81775

theorem repeating_decimals_count : 
  ∀ n : ℕ, 1 ≤ n ∧ n < 1000 → ¬(∃ k : ℕ, n + 1 = 2^k ∨ n + 1 = 5^k) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_count_l817_81775


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l817_81773

def lines_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + 2 * y = 0) ↔ (x + (a + 1) * y + 4 = 0)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = 1 → lines_parallel a) ∧ ¬(lines_parallel a → a = 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l817_81773


namespace NUMINAMATH_GPT_bill_fine_amount_l817_81792

-- Define the conditions
def ounces_sold : ℕ := 8
def earnings_per_ounce : ℕ := 9
def amount_left : ℕ := 22

-- Calculate the earnings
def earnings : ℕ := ounces_sold * earnings_per_ounce

-- Define the fine as the difference between earnings and amount left
def fine : ℕ := earnings - amount_left

-- The proof problem to solve
theorem bill_fine_amount : fine = 50 :=
by
  -- Statements and calculations would go here
  sorry

end NUMINAMATH_GPT_bill_fine_amount_l817_81792


namespace NUMINAMATH_GPT_five_point_questions_l817_81724

-- Defining the conditions as Lean statements
def question_count (x y : ℕ) : Prop := x + y = 30
def total_points (x y : ℕ) : Prop := 5 * x + 10 * y = 200

-- The theorem statement that states x equals the number of 5-point questions
theorem five_point_questions (x y : ℕ) (h1 : question_count x y) (h2 : total_points x y) : x = 20 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_five_point_questions_l817_81724


namespace NUMINAMATH_GPT_gasoline_added_l817_81777

variable (tank_capacity : ℝ := 42)
variable (initial_fill_fraction : ℝ := 3/4)
variable (final_fill_fraction : ℝ := 9/10)

theorem gasoline_added :
  let initial_amount := tank_capacity * initial_fill_fraction
  let final_amount := tank_capacity * final_fill_fraction
  final_amount - initial_amount = 6.3 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_added_l817_81777


namespace NUMINAMATH_GPT_bag_weight_l817_81704

theorem bag_weight (W : ℕ) 
  (h1 : 2 * W + 82 * (2 * W) = 664) : 
  W = 4 := by
  sorry

end NUMINAMATH_GPT_bag_weight_l817_81704


namespace NUMINAMATH_GPT_kaleb_savings_l817_81759

theorem kaleb_savings (x : ℕ) (h : x + 25 = 8 * 8) : x = 39 := 
by
  sorry

end NUMINAMATH_GPT_kaleb_savings_l817_81759
