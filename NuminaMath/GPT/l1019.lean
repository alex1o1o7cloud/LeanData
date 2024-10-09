import Mathlib

namespace bottles_remaining_l1019_101912

-- Define the initial number of bottles.
def initial_bottles : ℝ := 45.0

-- Define the number of bottles Maria drank.
def maria_drinks : ℝ := 14.0

-- Define the number of bottles Maria's sister drank.
def sister_drinks : ℝ := 8.0

-- The value that needs to be proved.
def bottles_left : ℝ := initial_bottles - maria_drinks - sister_drinks

-- The theorem statement.
theorem bottles_remaining :
  bottles_left = 23.0 :=
by
  sorry

end bottles_remaining_l1019_101912


namespace value_of_a_l1019_101901

theorem value_of_a (a : ℝ) :
  ((abs ((1) - (2) + a)) = 1) ↔ (a = 0 ∨ a = 2) :=
by
  sorry

end value_of_a_l1019_101901


namespace fraction_of_second_year_given_not_third_year_l1019_101955

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year_l1019_101955


namespace alpha_sufficient_not_necessary_l1019_101992

def A := {x : ℝ | 2 < x ∧ x < 3}

def B (α : ℝ) := {x : ℝ | (x + 2) * (x - α) < 0}

theorem alpha_sufficient_not_necessary (α : ℝ) : 
  (α = 1 → A ∩ B α = ∅) ∧ (∃ β : ℝ, β ≠ 1 ∧ A ∩ B β = ∅) :=
by
  sorry

end alpha_sufficient_not_necessary_l1019_101992


namespace initial_total_perimeter_l1019_101935

theorem initial_total_perimeter (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 2 * m)
  (h2 : 40 = 2 * a * m)
  (h3 : 4 * n - 6 * m = 4 * n - 40) :
  4 * n = 280 :=
by sorry

end initial_total_perimeter_l1019_101935


namespace a_share_is_approx_560_l1019_101977

noncomputable def investment_share (a_invest b_invest c_invest total_months b_share : ℕ) : ℝ :=
  let total_invest := a_invest + b_invest + c_invest
  let total_profit := (b_share * total_invest) / b_invest
  let a_share_ratio := a_invest / total_invest
  (a_share_ratio * total_profit)

theorem a_share_is_approx_560 
  (a_invest : ℕ := 7000) 
  (b_invest : ℕ := 11000) 
  (c_invest : ℕ := 18000) 
  (total_months : ℕ := 8) 
  (b_share : ℕ := 880) : 
  ∃ (a_share : ℝ), abs (a_share - 560) < 1 :=
by
  let a_share := investment_share a_invest b_invest c_invest total_months b_share
  existsi a_share
  sorry

end a_share_is_approx_560_l1019_101977


namespace seating_arrangements_l1019_101905

def valid_seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  if total_seats = 8 ∧ people = 3 then 12 else 0

theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 8) (h2 : people = 3) :
  valid_seating_arrangements total_seats people = 12 :=
by
  rw [valid_seating_arrangements, h1, h2]
  simp
  done

end seating_arrangements_l1019_101905


namespace phillip_spent_on_oranges_l1019_101902

theorem phillip_spent_on_oranges 
  (M : ℕ) (A : ℕ) (C : ℕ) (L : ℕ) (O : ℕ)
  (hM : M = 95) (hA : A = 25) (hC : C = 6) (hL : L = 50)
  (h_total_spending : O + A + C = M - L) : 
  O = 14 := 
sorry

end phillip_spent_on_oranges_l1019_101902


namespace vincent_correct_answer_l1019_101933

theorem vincent_correct_answer (y : ℕ) (h : (y - 7) / 5 = 23) : (y - 5) / 7 = 17 :=
by
  sorry

end vincent_correct_answer_l1019_101933


namespace angle_measure_l1019_101984

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l1019_101984


namespace joes_speed_l1019_101993

theorem joes_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_minutes : ℝ) (distance : ℝ) (h1 : joe_speed = 2 * pete_speed) (h2 : time_minutes = 40) (h3 : distance = 16) : joe_speed = 16 :=
by
  sorry

end joes_speed_l1019_101993


namespace inverse_value_ratio_l1019_101972

noncomputable def g (x : ℚ) : ℚ := (3 * x + 1) / (x - 4)

theorem inverse_value_ratio :
  (∃ (a b c d : ℚ), ∀ x, g ((a * x + b) / (c * x + d)) = x) → ∃ a c : ℚ, a / c = -4 :=
by
  sorry

end inverse_value_ratio_l1019_101972


namespace measure_angle_PQR_is_55_l1019_101963

noncomputable def measure_angle_PQR (POQ QOR : ℝ) : ℝ :=
  let POQ := 120
  let QOR := 130
  let POR := 360 - (POQ + QOR)
  let OPR := (180 - POR) / 2
  let OPQ := (180 - POQ) / 2
  let OQR := (180 - QOR) / 2
  OPQ + OQR

theorem measure_angle_PQR_is_55 : measure_angle_PQR 120 130 = 55 := by
  sorry

end measure_angle_PQR_is_55_l1019_101963


namespace find_A_l1019_101981

axiom power_eq_A (A : ℝ) (x y : ℝ) : 2^x = A ∧ 7^(2*y) = A
axiom reciprocal_sum_eq_2 (x y : ℝ) : (1/x) + (1/y) = 2

theorem find_A (A x y : ℝ) : 
  (2^x = A) ∧ (7^(2*y) = A) ∧ ((1/x) + (1/y) = 2) -> A = 7*Real.sqrt 2 :=
by 
  sorry

end find_A_l1019_101981


namespace arithmetic_sequence_15th_term_l1019_101959

/-- 
The arithmetic sequence with first term 1 and common difference 3.
The 15th term of this sequence is 43.
-/
theorem arithmetic_sequence_15th_term :
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → n = 15 → (a1 + (n - 1) * d) = 43 :=
by
  sorry

end arithmetic_sequence_15th_term_l1019_101959


namespace locus_of_points_l1019_101922

def point := (ℝ × ℝ)

variables (F_1 F_2 : point) (r k : ℝ)

def distance (P Q : point) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def on_circle (P : point) (center : point) (radius : ℝ) : Prop :=
  distance P center = radius

theorem locus_of_points
  (P : point)
  (r1 r2 PF1 PF2 : ℝ)
  (h_pF1 : r1 = distance P F_1)
  (h_pF2 : PF2 = distance P F_2)
  (h_outside_circle : PF2 = r2 + r)
  (h_inside_circle : PF2 = r - r2)
  (h_k : r1 + PF2 = k) :
  (∀ P, distance P F_1 + distance P F_2 = k →
  ( ∃ e_ellipse : Prop, on_circle P F_2 r → e_ellipse) ∨ 
  ( ∃ h_hyperbola : Prop, on_circle P F_2 r → h_hyperbola)) :=
by
  sorry

end locus_of_points_l1019_101922


namespace solution_to_system_of_inequalities_l1019_101916

variable {x y : ℝ}

theorem solution_to_system_of_inequalities :
  11 * (-1/3 : ℝ)^2 + 8 * (-1/3 : ℝ) * (2/3 : ℝ) + 8 * (2/3 : ℝ)^2 ≤ 3 ∧
  (-1/3 : ℝ) - 4 * (2/3 : ℝ) ≤ -3 :=
by
  sorry

end solution_to_system_of_inequalities_l1019_101916


namespace cosine_120_eq_neg_one_half_l1019_101971

theorem cosine_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1/2 :=
by
-- Proof omitted
sorry

end cosine_120_eq_neg_one_half_l1019_101971


namespace quadratic_eq_proof_l1019_101969

noncomputable def quadratic_eq := ∀ (a b : ℝ), 
  (a ≠ 0 → (∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0) →
    (a = b^2 ∧ a = 1 ∧ b = 1) ∨ (a > 1 ∧ 0 < b ∧ b < 1 → ¬ ∃ (x : ℝ), a * x^2 + b * x + 1/4 = 0))

theorem quadratic_eq_proof : quadratic_eq := 
by
  sorry

end quadratic_eq_proof_l1019_101969


namespace no_real_solutions_to_equation_l1019_101998

theorem no_real_solutions_to_equation :
  ¬ ∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) :=
by
  sorry

end no_real_solutions_to_equation_l1019_101998


namespace no_integer_solutions_other_than_zero_l1019_101990

theorem no_integer_solutions_other_than_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  sorry

end no_integer_solutions_other_than_zero_l1019_101990


namespace transport_cost_6725_l1019_101921

variable (P : ℝ) (T : ℝ)

theorem transport_cost_6725
  (h1 : 0.80 * P = 17500)
  (h2 : 1.10 * P = 24475)
  (h3 : 17500 + T + 250 = 24475) :
  T = 6725 := 
sorry

end transport_cost_6725_l1019_101921


namespace point_A_coords_l1019_101957

theorem point_A_coords (x y : ℝ) (h : ∀ t : ℝ, (t + 1) * x - (2 * t + 5) * y - 6 = 0) : x = -4 ∧ y = -2 := by
  sorry

end point_A_coords_l1019_101957


namespace scarf_cost_is_10_l1019_101952

-- Define the conditions as given in the problem statement
def initial_amount : ℕ := 53
def cost_per_toy_car : ℕ := 11
def num_toy_cars : ℕ := 2
def cost_of_beanie : ℕ := 14
def remaining_after_beanie : ℕ := 7

-- Calculate the cost of the toy cars
def total_cost_toy_cars : ℕ := num_toy_cars * cost_per_toy_car

-- Calculate the amount left after buying the toy cars
def amount_after_toys : ℕ := initial_amount - total_cost_toy_cars

-- Calculate the amount left after buying the beanie
def amount_after_beanie : ℕ := amount_after_toys - cost_of_beanie

-- Define the cost of the scarf
def cost_of_scarf : ℕ := amount_after_beanie - remaining_after_beanie

-- The theorem stating that cost_of_scarf is 10 dollars
theorem scarf_cost_is_10 : cost_of_scarf = 10 := by
  sorry

end scarf_cost_is_10_l1019_101952


namespace variance_transformation_l1019_101954

theorem variance_transformation (a1 a2 a3 : ℝ) 
  (h1 : (a1 + a2 + a3) / 3 = 4) 
  (h2 : ((a1 - 4)^2 + (a2 - 4)^2 + (a3 - 4)^2) / 3 = 3) : 
  ((3 * a1 - 2 - (3 * 4 - 2))^2 + (3 * a2 - 2 - (3 * 4 - 2))^2 + (3 * a3 - 2 - (3 * 4 - 2))^2) / 3 = 27 := 
sorry

end variance_transformation_l1019_101954


namespace bill_needs_paint_cans_l1019_101979

theorem bill_needs_paint_cans :
  let bedrooms := 3
  let other_rooms := 2 * bedrooms
  let gallons_per_room := 2
  let color_paint_cans := 6 -- (bedrooms * gallons_per_room) / 1-gallon per can
  let white_paint_cans := 4 -- (other_rooms * gallons_per_room) / 3-gallons per can
  (color_paint_cans + white_paint_cans) = 10 := sorry

end bill_needs_paint_cans_l1019_101979


namespace ratio_after_girls_leave_l1019_101946

-- Define the initial conditions
def initial_conditions (B G : ℕ) : Prop :=
  B = G ∧ B + G = 32

-- Define the event of girls leaving
def girls_leave (G : ℕ) : ℕ :=
  G - 8

-- Define the final ratio of boys to girls
def final_ratio (B G : ℕ) : ℕ :=
  B / (girls_leave G)

-- Prove the final ratio is 2:1
theorem ratio_after_girls_leave (B G : ℕ) (h : initial_conditions B G) :
  final_ratio B G = 2 :=
by
  sorry

end ratio_after_girls_leave_l1019_101946


namespace first_term_of_geometric_series_l1019_101995

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l1019_101995


namespace width_of_sheet_of_paper_l1019_101919

theorem width_of_sheet_of_paper (W : ℝ) (h1 : ∀ (W : ℝ), W > 0) (length_paper : ℝ) (margin : ℝ)
  (width_picture_area : ∀ (W : ℝ), W - 2 * margin = (W - 3)) 
  (area_picture : ℝ) (length_picture_area : ℝ) :
  length_paper = 10 ∧ margin = 1.5 ∧ area_picture = 38.5 ∧ length_picture_area = 7 →
  W = 8.5 :=
by
  sorry

end width_of_sheet_of_paper_l1019_101919


namespace wendy_albums_l1019_101996

theorem wendy_albums (total_pictures remaining_pictures pictures_per_album : ℕ) 
    (h1 : total_pictures = 79)
    (h2 : remaining_pictures = total_pictures - 44)
    (h3 : pictures_per_album = 7) :
    remaining_pictures / pictures_per_album = 5 := by
  sorry

end wendy_albums_l1019_101996


namespace number_of_people_l1019_101974

theorem number_of_people (total_bowls : ℕ) (bowls_per_person : ℚ) : total_bowls = 55 ∧ bowls_per_person = 1 + 1/2 + 1/3 → total_bowls / bowls_per_person = 30 :=
by
  sorry

end number_of_people_l1019_101974


namespace smallest_x_l1019_101926

theorem smallest_x (x : ℕ) (h : 67 * 89 * x % 35 = 0) : x = 35 := 
by sorry

end smallest_x_l1019_101926


namespace length_of_greater_segment_l1019_101999

theorem length_of_greater_segment (x : ℤ) (h1 : (x + 2)^2 - x^2 = 32) : x + 2 = 9 := by
  sorry

end length_of_greater_segment_l1019_101999


namespace find_inverse_l1019_101986

noncomputable def inverse_matrix_2x2 (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  if ad_bc : (a * d - b * c) = 0 then (0, 0, 0, 0)
  else (d / (a * d - b * c), -b / (a * d - b * c), -c / (a * d - b * c), a / (a * d - b * c))

theorem find_inverse :
  inverse_matrix_2x2 5 7 2 3 = (3, -7, -2, 5) :=
by 
  sorry

end find_inverse_l1019_101986


namespace problem_l1019_101960

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l1019_101960


namespace sasha_age_l1019_101937

theorem sasha_age :
  ∃ a : ℕ, 
    (M = 2 * a - 3) ∧
    (M = a + (a - 3)) ∧
    (a = 3) :=
by
  sorry

end sasha_age_l1019_101937


namespace cos_double_angle_l1019_101928

variable (θ : ℝ)

theorem cos_double_angle (h : Real.tan (θ + Real.pi / 4) = 3) : Real.cos (2 * θ) = 3 / 5 :=
sorry

end cos_double_angle_l1019_101928


namespace most_representative_sample_l1019_101920

/-- Options for the student sampling methods -/
inductive SamplingMethod
| NinthGradeStudents : SamplingMethod
| FemaleStudents : SamplingMethod
| BasketballStudents : SamplingMethod
| StudentsWithIDEnding5 : SamplingMethod

/-- Definition of representativeness for each SamplingMethod -/
def isMostRepresentative (method : SamplingMethod) : Prop :=
  method = SamplingMethod.StudentsWithIDEnding5

/-- Prove that the students with ID ending in 5 is the most representative sampling method -/
theorem most_representative_sample : isMostRepresentative SamplingMethod.StudentsWithIDEnding5 :=
  by
  sorry

end most_representative_sample_l1019_101920


namespace evaluate_expression_l1019_101907

theorem evaluate_expression : 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 :=
by
  sorry

end evaluate_expression_l1019_101907


namespace shaded_area_l1019_101980

theorem shaded_area (whole_squares partial_squares : ℕ) (area_whole area_partial : ℝ)
  (h1 : whole_squares = 5)
  (h2 : partial_squares = 6)
  (h3 : area_whole = 1)
  (h4 : area_partial = 0.5) :
  (whole_squares * area_whole + partial_squares * area_partial) = 8 :=
by
  sorry

end shaded_area_l1019_101980


namespace min_sum_x_y_l1019_101911

theorem min_sum_x_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0 ∧ y > 0) (h3 : (1 : ℚ)/x + (1 : ℚ)/y = 1/12) : x + y = 49 :=
sorry

end min_sum_x_y_l1019_101911


namespace find_x_l1019_101965

theorem find_x (x : ℝ) (h : 121 * x^4 = 75625) : x = 5 :=
sorry

end find_x_l1019_101965


namespace correct_statement_is_C_l1019_101997

theorem correct_statement_is_C :
  (∃ x : ℚ, ∀ y : ℚ, x < y) = false ∧
  (∃ x : ℚ, x < 0 ∧ ∀ y : ℚ, y < 0 → x < y) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, abs x ≤ abs y) ∧
  (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → x ≤ y) = false :=
sorry

end correct_statement_is_C_l1019_101997


namespace sphere_volume_l1019_101975

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) : (4/3) * Real.pi * r^3 = 36 * Real.pi := 
sorry

end sphere_volume_l1019_101975


namespace second_player_wins_l1019_101982

theorem second_player_wins : 
  ∀ (a b c : ℝ), (a ≠ 0) → 
  (∃ (first_choice: ℝ), ∃ (second_choice: ℝ), 
    ∃ (third_choice: ℝ), 
    ((first_choice ≠ 0) → (b^2 + 4 * first_choice^2 > 0)) ∧ 
    ((first_choice = 0) → (b ≠ 0)) ∧ 
    first_choice * (first_choice * b + a) = 0 ↔ ∃ x : ℝ, a * x^2 + (first_choice + second_choice) * x + third_choice = 0) :=
by sorry

end second_player_wins_l1019_101982


namespace unique_g_zero_l1019_101940

theorem unique_g_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) = g (x) + g (y) - 1) : g 0 = 1 :=
by
  sorry

end unique_g_zero_l1019_101940


namespace taller_building_height_l1019_101983

theorem taller_building_height
  (H : ℕ) -- H is the height of the taller building
  (h_ratio : (H - 36) / H = 5 / 7) -- heights ratio condition
  (h_diff : H > 36) -- height difference must respect physics
  : H = 126 := sorry

end taller_building_height_l1019_101983


namespace probability_of_white_ball_l1019_101966

theorem probability_of_white_ball (red_balls white_balls : ℕ) (draws : ℕ)
    (h_red : red_balls = 4) (h_white : white_balls = 2) (h_draws : draws = 2) :
    ((4 * 2 + 1) / 15 : ℚ) = 3 / 5 := by sorry

end probability_of_white_ball_l1019_101966


namespace grazing_area_proof_l1019_101944

noncomputable def grazing_area (s r : ℝ) : ℝ :=
  let A_circle := 3.14 * r^2
  let A_sector := (300 / 360) * A_circle
  let A_triangle := (1.732 / 4) * s^2
  let A_triangle_part := A_triangle / 3
  let A_grazing := A_sector - A_triangle_part
  3 * A_grazing

theorem grazing_area_proof : grazing_area 5 7 = 136.59 :=
  by
  sorry

end grazing_area_proof_l1019_101944


namespace value_of_sum_l1019_101951

theorem value_of_sum (a x y : ℝ) (h1 : 17 * x + 19 * y = 6 - a) (h2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := 
sorry

end value_of_sum_l1019_101951


namespace quadratic_function_property_l1019_101970

theorem quadratic_function_property
    (a b c : ℝ)
    (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = a * x^2 + b * x + c)
    (h_vertex : f (-2) = a^2)
    (h_point : f (-1) = 6)
    (h_vertex_condition : -b / (2 * a) = -2)
    (h_a_neg : a < 0) :
    (a + c) / b = 1 / 2 :=
by
  sorry

end quadratic_function_property_l1019_101970


namespace river_width_l1019_101924

def boat_width : ℕ := 3
def num_boats : ℕ := 8
def space_between_boats : ℕ := 2
def riverbank_space : ℕ := 2

theorem river_width : 
  let boat_space := num_boats * boat_width
  let between_boat_space := (num_boats - 1) * space_between_boats
  let riverbank_space_total := 2 * riverbank_space
  boat_space + between_boat_space + riverbank_space_total = 42 :=
by
  sorry

end river_width_l1019_101924


namespace simplify_expression_l1019_101939

-- Definitions of intermediate calculations
def a : ℤ := 3 + 5 + 6 - 2
def b : ℚ := a * 2 / 4
def c : ℤ := 3 * 4 + 6 - 4
def d : ℚ := c / 3

-- The statement to be proved
theorem simplify_expression : b + d = 32 / 3 := by
  sorry

end simplify_expression_l1019_101939


namespace find_A_minus_C_l1019_101909

/-- There are three different natural numbers A, B, and C. 
    When A + B = 84, B + C = 60, and A = 6B, find the value of A - C. -/
theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84) 
  (h2 : B + C = 60) 
  (h3 : A = 6 * B) 
  (h4 : A ≠ B) 
  (h5 : A ≠ C) 
  (h6 : B ≠ C) :
  A - C = 24 :=
sorry

end find_A_minus_C_l1019_101909


namespace stone_statue_cost_l1019_101994

theorem stone_statue_cost :
  ∃ S : Real, 
    let total_earnings := 10 * S + 20 * 5
    let earnings_after_taxes := 0.9 * total_earnings
    earnings_after_taxes = 270 ∧ S = 20 :=
sorry

end stone_statue_cost_l1019_101994


namespace prob_win_3_1_correct_l1019_101932

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end prob_win_3_1_correct_l1019_101932


namespace largest_non_summable_composite_l1019_101964

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def can_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

theorem largest_non_summable_composite : ∀ m : ℕ, (m < 11 → ¬ can_be_sum_of_two_composites m) ∧ (m ≥ 11 → can_be_sum_of_two_composites m) :=
by sorry

end largest_non_summable_composite_l1019_101964


namespace color_points_l1019_101947

def is_white (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1) ∧ (p.2 % 2 = 1)
def is_black (p : ℤ × ℤ) : Prop := (p.1 % 2 = 0) ∧ (p.2 % 2 = 0)
def is_red (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1 ∧ p.2 % 2 = 0) ∨ (p.1 % 2 = 0 ∧ p.2 % 2 = 1)

theorem color_points :
  (∀ n : ℤ, ∃ (p : ℤ × ℤ), (p.2 = n) ∧ is_white p ∧
                             is_black ⟨p.1, n * 2⟩ ∧
                             is_red ⟨p.1, n * 2 + 1⟩) ∧ 
  (∀ (A B C : ℤ × ℤ), 
    is_white A → is_red B → is_black C → 
    ∃ D : ℤ × ℤ, is_red D ∧ 
    (A.1 + C.1 - B.1 = D.1 ∧
     A.2 + C.2 - B.2 = D.2)) := sorry

end color_points_l1019_101947


namespace min_value_fraction_geq_3_div_2_l1019_101934

theorem min_value_fraction_geq_3_div_2 (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h1 : q > 0) 
  (h2 : ∀ k, a (k + 2) = q * a (k + 1)) (h3 : a 2016 = a 2015 + 2 * a 2014) 
  (h4 : a m * a n = 16 * (a 1) ^ 2) :
  (∃ q, q = 2 ∧ m + n = 6) → 4 / m + 1 / n ≥ 3 / 2 :=
by sorry

end min_value_fraction_geq_3_div_2_l1019_101934


namespace sum_of_first_six_terms_l1019_101929

def geometric_sequence (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = -2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℤ) : ℤ := 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms (a : ℕ → ℤ) 
  (h : geometric_sequence a) :
  sum_first_six_terms a = -21 :=
sorry

end sum_of_first_six_terms_l1019_101929


namespace radius_of_two_equal_circles_eq_16_l1019_101988

noncomputable def radius_of_congruent_circles : ℝ := 16

theorem radius_of_two_equal_circles_eq_16 :
  ∃ x : ℝ, 
    (∀ r1 r2 r3 : ℝ, r1 = 4 ∧ r2 = r3 ∧ r2 = x ∧ 
    ∃ line : ℝ → ℝ → Prop, 
    (line 0 r1) ∧ (line 0 r2)  ∧ 
    (line 0 r3) ∧ 
    (line r2 r3) ∧
    (line r1 r2)  ∧ (line r1 r3) ∧ (line (r1 + r2) r2) ) 
    → x = 16 := sorry

end radius_of_two_equal_circles_eq_16_l1019_101988


namespace log_tan_ratio_l1019_101923

noncomputable def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
noncomputable def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
noncomputable def tan_ratio (α β : ℝ) : ℝ := Real.tan α / Real.tan β

theorem log_tan_ratio (α β : ℝ)
  (h1 : sin_add α β = 1 / 2)
  (h2 : sin_sub α β = 1 / 3) :
  Real.logb 5 (tan_ratio α β) = 1 := by
sorry

end log_tan_ratio_l1019_101923


namespace circumscribed_circle_radius_l1019_101976

theorem circumscribed_circle_radius (b c : ℝ) (cosA : ℝ)
  (hb : b = 2) (hc : c = 3) (hcosA : cosA = 1 / 3) : 
  R = 9 * Real.sqrt 2 / 8 :=
by
  sorry

end circumscribed_circle_radius_l1019_101976


namespace broker_wealth_increase_after_two_years_l1019_101953

theorem broker_wealth_increase_after_two_years :
  let initial_investment : ℝ := 100
  let first_year_increase : ℝ := 0.75
  let second_year_decrease : ℝ := 0.30
  let end_first_year := initial_investment * (1 + first_year_increase)
  let end_second_year := end_first_year * (1 - second_year_decrease)
  end_second_year - initial_investment = 22.50 :=
by
  sorry

end broker_wealth_increase_after_two_years_l1019_101953


namespace bridge_length_l1019_101950

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) 
  (train_speed_m_s : train_speed_kmh * (1000 / 3600) = 15) : 
  train_length = 110 → train_speed_kmh = 54 → time_to_cross_bridge = 16.13204276991174 → 
  ((train_speed_kmh * (1000 / 3600)) * time_to_cross_bridge - train_length = 131.9806415486761) :=
by
  intros h1 h2 h3
  sorry

end bridge_length_l1019_101950


namespace ab_value_l1019_101989

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 25.3125 :=
by
  sorry

end ab_value_l1019_101989


namespace distance_origin_to_line_l1019_101958

theorem distance_origin_to_line : 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  distance = 1 :=
by 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  sorry

end distance_origin_to_line_l1019_101958


namespace eq_30_apples_n_7_babies_min_3_max_6_l1019_101930

theorem eq_30_apples_n_7_babies_min_3_max_6 (x : ℕ) 
    (h1 : 30 = x + 7 * 4)
    (h2 : 21 ≤ 30) 
    (h3 : 30 ≤ 42) 
    (h4 : x = 2) :
  x = 2 :=
by
  sorry

end eq_30_apples_n_7_babies_min_3_max_6_l1019_101930


namespace solve_congruence_l1019_101925

open Nat

theorem solve_congruence (x : ℕ) (h : x^2 + x - 6 ≡ 0 [MOD 143]) : 
  x = 2 ∨ x = 41 ∨ x = 101 ∨ x = 140 :=
by
  sorry

end solve_congruence_l1019_101925


namespace sum_of_roots_eq_14_l1019_101945

theorem sum_of_roots_eq_14 (x : ℝ) :
  (x - 7) ^ 2 = 16 → ∃ r1 r2 : ℝ, (x = r1 ∨ x = r2) ∧ r1 + r2 = 14 :=
by
  intro h
  sorry

end sum_of_roots_eq_14_l1019_101945


namespace faster_speed_l1019_101918

theorem faster_speed (S : ℝ) (actual_speed : ℝ := 10) (extra_distance : ℝ := 20) (actual_distance : ℝ := 20) :
  actual_distance / actual_speed = (actual_distance + extra_distance) / S → S = 20 :=
by
  sorry

end faster_speed_l1019_101918


namespace eleven_pow_four_l1019_101949

theorem eleven_pow_four : 11 ^ 4 = 14641 := 
by sorry

end eleven_pow_four_l1019_101949


namespace Solomon_collected_66_l1019_101942

-- Definitions
variables (J S L : ℕ) -- J for Juwan, S for Solomon, L for Levi

-- Conditions
axiom C1 : S = 3 * J
axiom C2 : L = J / 2
axiom C3 : J + S + L = 99

-- Theorem to prove
theorem Solomon_collected_66 : S = 66 :=
by
  sorry

end Solomon_collected_66_l1019_101942


namespace range_of_a_l1019_101987

def new_operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, new_operation x (x - a) > 1) ↔ (a < -3 ∨ 1 < a) := 
by
  sorry

end range_of_a_l1019_101987


namespace find_numerator_l1019_101978

variable {y : ℝ} (hy : y > 0) (n : ℝ)

theorem find_numerator (h: (2 * y / 10) + n = 1 / 2 * y) : n = 3 :=
sorry

end find_numerator_l1019_101978


namespace vector_BC_l1019_101938

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_BC (BA CA BC : ℝ × ℝ) (BA_def : BA = (1, 2)) (CA_def : CA = (4, 5)) (BC_def : BC = vector_sub BA CA) : BC = (-3, -3) :=
by
  subst BA_def
  subst CA_def
  subst BC_def
  sorry

end vector_BC_l1019_101938


namespace complement_A_U_l1019_101903

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := U \ A

-- Theorem: The complement of A with respect to U is {2, 4}
theorem complement_A_U : C_U_A = {2, 4} := by
  sorry

end complement_A_U_l1019_101903


namespace problem_proof_l1019_101931

-- Define positive integers and the conditions given in the problem
variables {p q r s : ℕ}

-- The product of the four integers is 7!
axiom product_of_integers : p * q * r * s = 5040  -- 7! = 5040

-- The equations defining the relationships
axiom equation1 : p * q + p + q = 715
axiom equation2 : q * r + q + r = 209
axiom equation3 : r * s + r + s = 143

-- The goal is to prove p - s = 10
theorem problem_proof : p - s = 10 :=
sorry

end problem_proof_l1019_101931


namespace moles_of_naoh_combined_number_of_moles_of_naoh_combined_l1019_101915

-- Define the reaction equation and given conditions
def reaction_equation := "2 NaOH + Cl₂ → NaClO + NaCl + H₂O"

-- Given conditions
def moles_chlorine : ℕ := 2
def moles_water_produced : ℕ := 2
def moles_naoh_needed_for_one_mole_water : ℕ := 2

-- Stoichiometric relationship from the reaction equation
def moles_naoh_per_mole_water : ℕ := 2

-- Theorem to prove the number of moles of NaOH combined
theorem moles_of_naoh_combined (moles_water_produced : ℕ)
  (moles_naoh_per_mole_water : ℕ) : ℕ :=
  moles_water_produced * moles_naoh_per_mole_water

-- Statement of the theorem
theorem number_of_moles_of_naoh_combined : moles_of_naoh_combined 2 2 = 4 :=
by sorry

end moles_of_naoh_combined_number_of_moles_of_naoh_combined_l1019_101915


namespace ones_digit_of_73_pow_355_l1019_101948

theorem ones_digit_of_73_pow_355 : (73 ^ 355) % 10 = 7 := 
  sorry

end ones_digit_of_73_pow_355_l1019_101948


namespace average_weight_of_section_B_l1019_101917

theorem average_weight_of_section_B
  (num_students_A : ℕ) (num_students_B : ℕ)
  (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_A : ℝ := num_students_A * avg_weight_A)
  (total_weight_B : ℝ := total_weight_class - total_weight_A)
  (avg_weight_B : ℝ := total_weight_B / num_students_B) :
  num_students_A = 50 →
  num_students_B = 40 →
  avg_weight_A = 50 →
  avg_weight_class = 58.89 →
  avg_weight_B = 70.0025 :=
by intros; sorry

end average_weight_of_section_B_l1019_101917


namespace one_greater_l1019_101906

theorem one_greater (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) 
  (h5 : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
sorry

end one_greater_l1019_101906


namespace combined_work_rate_l1019_101910

theorem combined_work_rate (W : ℝ) 
  (A_rate : ℝ := W / 10) 
  (B_rate : ℝ := W / 5) : 
  A_rate + B_rate = 3 * W / 10 := 
by
  sorry

end combined_work_rate_l1019_101910


namespace quotient_A_div_B_l1019_101967

-- Define A according to the given conditions
def A : ℕ := (8 * 10) + (13 * 1)

-- Define B according to the given conditions
def B : ℕ := 30 - 9 - 9 - 9

-- Prove that the quotient of A divided by B is 31
theorem quotient_A_div_B : (A / B) = 31 := by
  sorry

end quotient_A_div_B_l1019_101967


namespace speed_of_man_in_still_water_l1019_101904

variable (v_m v_s : ℝ)

-- Conditions
def downstream_distance : ℝ := 51
def upstream_distance : ℝ := 18
def time : ℝ := 3

-- Equations based on the conditions
def downstream_speed_eq : Prop := downstream_distance = (v_m + v_s) * time
def upstream_speed_eq : Prop := upstream_distance = (v_m - v_s) * time

-- The theorem to prove
theorem speed_of_man_in_still_water : downstream_speed_eq v_m v_s ∧ upstream_speed_eq v_m v_s → v_m = 11.5 :=
by
  intro h
  sorry

end speed_of_man_in_still_water_l1019_101904


namespace arithmetic_expression_eval_l1019_101962

theorem arithmetic_expression_eval : 
  5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 :=
by
  sorry

end arithmetic_expression_eval_l1019_101962


namespace max_possible_percentage_l1019_101936

theorem max_possible_percentage (p_wi : ℝ) (p_fs : ℝ) (h_wi : p_wi = 0.4) (h_fs : p_fs = 0.7) :
  ∃ p_both : ℝ, p_both = min p_wi p_fs ∧ p_both = 0.4 :=
by
  sorry

end max_possible_percentage_l1019_101936


namespace shanghai_world_expo_l1019_101927

theorem shanghai_world_expo (n : ℕ) (total_cost : ℕ) 
  (H1 : total_cost = 4000)
  (H2 : n ≤ 30 → total_cost = n * 120)
  (H3 : n > 30 → total_cost = n * (120 - 2 * (n - 30)) ∧ (120 - 2 * (n - 30)) ≥ 90) :
  n = 40 := 
sorry

end shanghai_world_expo_l1019_101927


namespace intersection_of_A_and_B_l1019_101941

-- Conditions: definitions of sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B : Set ℝ := {x | x < 1}

-- The proof goal: A ∩ B = {x | -1 ≤ x ∧ x < 1}
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l1019_101941


namespace paper_clips_in_two_cases_l1019_101991

theorem paper_clips_in_two_cases (c b : ℕ) : 
  2 * c * b * 200 = 2 * (c * b * 200) :=
by
  sorry

end paper_clips_in_two_cases_l1019_101991


namespace range_of_m_l1019_101914

theorem range_of_m
  (m : ℝ)
  (h1 : (m - 1) * (3 - m) ≠ 0) 
  (h2 : 3 - m > 0) 
  (h3 : m - 1 > 0) 
  (h4 : 3 - m ≠ m - 1) :
  1 < m ∧ m < 3 ∧ m ≠ 2 :=
sorry

end range_of_m_l1019_101914


namespace correct_average_marks_l1019_101943

theorem correct_average_marks (n : ℕ) (average initial_wrong current_correct : ℕ) 
  (h_n : n = 10) 
  (h_avg : average = 100) 
  (h_wrong : initial_wrong = 60)
  (h_correct : current_correct = 10) : 
  (average * n - initial_wrong + current_correct) / n = 95 := 
by
  -- This is where the proof would go
  sorry

end correct_average_marks_l1019_101943


namespace problem_ab_plus_a_plus_b_l1019_101968

noncomputable def polynomial := fun x : ℝ => x^4 - 6 * x - 2

theorem problem_ab_plus_a_plus_b :
  ∀ (a b : ℝ), polynomial a = 0 → polynomial b = 0 → (a * b + a + b) = 4 :=
by
  intros a b ha hb
  sorry

end problem_ab_plus_a_plus_b_l1019_101968


namespace max_min_of_f_find_a_and_theta_l1019_101900

noncomputable def f (x θ a : ℝ) : ℝ :=
  Real.sin (x + θ) + a * Real.cos (x + 2 * θ)

theorem max_min_of_f (a θ : ℝ) (h1 : a = Real.sqrt 2) (h2 : θ = π / 4) :
  (∀ x ∈ Set.Icc 0 π, -1 ≤ f x θ a ∧ f x θ a ≤ (Real.sqrt 2) / 2) := sorry

theorem find_a_and_theta (a θ : ℝ) (h1 : f (π / 2) θ a = 0) (h2 : f π θ a = 1) :
  a = -1 ∧ θ = -π / 6 := sorry

end max_min_of_f_find_a_and_theta_l1019_101900


namespace fifteenth_number_with_digit_sum_15_is_294_l1019_101985

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def numbers_with_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) (List.range (10 ^ 3)) -- Assume a maximum of 3-digit numbers

def fifteenth_number_with_digit_sum (s : ℕ) : ℕ :=
  (numbers_with_digit_sum s).get! 14 -- Get the 15th element (0-indexed)

theorem fifteenth_number_with_digit_sum_15_is_294 : fifteenth_number_with_digit_sum 15 = 294 :=
by
  sorry -- Proof is omitted

end fifteenth_number_with_digit_sum_15_is_294_l1019_101985


namespace jane_needs_9_more_days_l1019_101913

def jane_rate : ℕ := 16
def mark_rate : ℕ := 20
def mark_days : ℕ := 3
def total_vases : ℕ := 248

def vases_by_mark_in_3_days : ℕ := mark_rate * mark_days
def vases_by_jane_and_mark_in_3_days : ℕ := (jane_rate + mark_rate) * mark_days
def remaining_vases_after_3_days : ℕ := total_vases - vases_by_jane_and_mark_in_3_days
def days_jane_needs_alone : ℕ := (remaining_vases_after_3_days + jane_rate - 1) / jane_rate

theorem jane_needs_9_more_days :
  days_jane_needs_alone = 9 :=
by
  sorry

end jane_needs_9_more_days_l1019_101913


namespace necessary_but_not_sufficient_l1019_101956

-- Define the geometric mean condition between 2 and 8
def is_geometric_mean (m : ℝ) := m = 4 ∨ m = -4

-- Prove that m = 4 is a necessary but not sufficient condition for is_geometric_mean
theorem necessary_but_not_sufficient (m : ℝ) :
  (is_geometric_mean m) ↔ (m = 4) :=
sorry

end necessary_but_not_sufficient_l1019_101956


namespace solve_inequality_l1019_101973

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_at_one_third (f : ℝ → ℝ) : Prop :=
  f (1/3) = 0

theorem solve_inequality (f : ℝ → ℝ) (x : ℝ) :
  even_function f →
  increasing_on_nonnegatives f →
  f_at_one_third f →
  (0 < x ∧ x < 1/2) ∨ (x > 2) ↔ f (Real.logb (1/8) x) > 0 :=
by
  -- the proof will be filled in here
  sorry

end solve_inequality_l1019_101973


namespace circumradius_eq_exradius_opposite_BC_l1019_101908

-- Definitions of points and triangles
variable {A B C : Point}
variable (O I D : Point)
variable {α β γ : Angle}

-- Definitions of circumcenter, incenter, altitude, and collinearity
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_incenter (I : Point) (A B C : Point) : Prop := sorry
def is_altitude (A D B C : Point) : Prop := sorry
def collinear (O D I : Point) : Prop := sorry

-- Definitions of circumradius and exradius
def circumradius (A B C : Point) : ℝ := sorry
def exradius_opposite_BC (A B C : Point) : ℝ := sorry

-- Main theorem statement
theorem circumradius_eq_exradius_opposite_BC
  (h_circ : is_circumcenter O A B C)
  (h_incenter : is_incenter I A B C)
  (h_altitude : is_altitude A D B C)
  (h_collinear : collinear O D I) : 
  circumradius A B C = exradius_opposite_BC A B C :=
sorry

end circumradius_eq_exradius_opposite_BC_l1019_101908


namespace distance_between_trees_l1019_101961

-- Define the conditions
def yard_length : ℝ := 325
def number_of_trees : ℝ := 26
def number_of_intervals : ℝ := number_of_trees - 1

-- Define what we need to prove
theorem distance_between_trees:
  (yard_length / number_of_intervals) = 13 := 
  sorry

end distance_between_trees_l1019_101961
