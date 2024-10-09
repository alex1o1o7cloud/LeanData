import Mathlib

namespace line_through_fixed_point_l2001_200100

theorem line_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (1, 2)) ∧ (∀ x y, a * x + y - a - 2 = 0 → P = (x, y)) ∧
  ((∃ a, x + y = a ∧ x = 1 ∧ y = 2) → (a = 3)) :=
by
  sorry

end line_through_fixed_point_l2001_200100


namespace increase_corrosion_with_more_active_metal_rivets_l2001_200150

-- Definitions representing conditions
def corrosion_inhibitor (P : Type) : Prop := true
def more_active_metal_rivets (P : Type) : Prop := true
def less_active_metal_rivets (P : Type) : Prop := true
def painted_parts (P : Type) : Prop := true

-- Main theorem statement
theorem increase_corrosion_with_more_active_metal_rivets (P : Type) 
  (h1 : corrosion_inhibitor P)
  (h2 : more_active_metal_rivets P)
  (h3 : less_active_metal_rivets P)
  (h4 : painted_parts P) : 
  more_active_metal_rivets P :=
by {
  -- proof goes here
  sorry
}

end increase_corrosion_with_more_active_metal_rivets_l2001_200150


namespace ten_row_triangle_total_l2001_200153

theorem ten_row_triangle_total:
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  rods + connectors = 231 :=
by
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  sorry

end ten_row_triangle_total_l2001_200153


namespace polygon_sides_eq_six_l2001_200136

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_eq_six_l2001_200136


namespace trig_inequality_l2001_200117

open Real

theorem trig_inequality (a b c : ℝ) (h₁ : a = sin (2 * π / 7))
  (h₂ : b = cos (2 * π / 7)) (h₃ : c = tan (2 * π / 7)) :
  c > a ∧ a > b :=
by 
  sorry

end trig_inequality_l2001_200117


namespace difference_of_squares_550_450_l2001_200157

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end difference_of_squares_550_450_l2001_200157


namespace rectangle_length_reduction_l2001_200142

theorem rectangle_length_reduction:
  ∀ (L W : ℝ) (X : ℝ),
  W > 0 →
  L > 0 →
  (L * (1 - X / 100) * (4 / 3)) * W = L * W →
  X = 25 :=
by
  intros L W X hW hL hEq
  sorry

end rectangle_length_reduction_l2001_200142


namespace tim_has_33_books_l2001_200183

-- Define the conditions
def b := 24   -- Benny's initial books
def s := 10   -- Books given to Sandy
def total_books : Nat := 47  -- Total books

-- Define the remaining books after Benny gives to Sandy
def remaining_b : Nat := b - s

-- Define Tim's books
def tim_books : Nat := total_books - remaining_b

-- Prove that Tim has 33 books
theorem tim_has_33_books : tim_books = 33 := by
  -- This is a placeholder for the proof
  sorry

end tim_has_33_books_l2001_200183


namespace normal_intersects_at_l2001_200197

def parabola (x : ℝ) : ℝ := x^2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x

-- C = (2, 4) is a point on the parabola
def C : ℝ × ℝ := (2, parabola 2)

-- Normal to the parabola at C intersects again at point D
-- Prove that D = (-9/4, 81/16)
theorem normal_intersects_at (D : ℝ × ℝ) :
  D = (-9/4, 81/16) :=
sorry

end normal_intersects_at_l2001_200197


namespace inning_count_l2001_200130

-- Definition of the conditions
variables {n T H L : ℕ}
variables (avg_total : ℕ) (avg_excl : ℕ) (diff : ℕ) (high_score : ℕ)

-- Define the conditions
def conditions :=
  avg_total = 62 ∧
  high_score = 225 ∧
  diff = 150 ∧
  avg_excl = 58

-- Proving the main theorem
theorem inning_count (avg_total := 62) (high_score := 225) (diff := 150) (avg_excl := 58) :
   conditions avg_total avg_excl diff high_score →
   n = 104 :=
sorry

end inning_count_l2001_200130


namespace Iggy_Tuesday_Run_l2001_200113

def IggyRunsOnTuesday (total_miles : ℕ) (monday_miles : ℕ) (wednesday_miles : ℕ) (thursday_miles : ℕ) (friday_miles : ℕ) : ℕ :=
  total_miles - (monday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem Iggy_Tuesday_Run :
  let monday_miles := 3
  let wednesday_miles := 6
  let thursday_miles := 8
  let friday_miles := 3
  let total_miles := 240 / 10
  IggyRunsOnTuesday total_miles monday_miles wednesday_miles thursday_miles friday_miles = 4 :=
by
  sorry

end Iggy_Tuesday_Run_l2001_200113


namespace solution_of_inequalities_l2001_200125

theorem solution_of_inequalities (x : ℝ) :
  (2 * x / 5 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) ↔ (-5 ≤ x ∧ x < -3 / 2) := by
  sorry

end solution_of_inequalities_l2001_200125


namespace length_of_courtyard_l2001_200151

-- Given conditions

def width_of_courtyard : ℝ := 14
def brick_length : ℝ := 0.25
def brick_width : ℝ := 0.15
def total_bricks : ℝ := 8960

-- To be proven
theorem length_of_courtyard : brick_length * brick_width * total_bricks / width_of_courtyard = 24 := 
by sorry

end length_of_courtyard_l2001_200151


namespace daryl_age_l2001_200173

theorem daryl_age (d j : ℕ) 
  (h1 : d - 4 = 3 * (j - 4)) 
  (h2 : d + 5 = 2 * (j + 5)) :
  d = 31 :=
by sorry

end daryl_age_l2001_200173


namespace smallest_square_side_lengths_l2001_200184

theorem smallest_square_side_lengths (x : ℕ) 
    (h₁ : ∀ (y : ℕ), y = x + 8) 
    (h₂ : ∀ (z : ℕ), z = 50) 
    (h₃ : ∀ (QS PS RT QT : ℕ), QS = 8 ∧ PS = x ∧ RT = 42 - x ∧ QT = x + 8 ∧ (8 / x) = ((42 - x) / (x + 8))) : 
  x = 2 ∨ x = 32 :=
by 
  sorry

end smallest_square_side_lengths_l2001_200184


namespace jordan_running_time_l2001_200134

-- Define the conditions given in the problem
variables (time_steve : ℕ) (distance_steve distance_jordan_1 distance_jordan_2 distance_jordan_3 : ℕ)

-- Assign the known values
axiom time_steve_def : time_steve = 24
axiom distance_steve_def : distance_steve = 3
axiom distance_jordan_1_def : distance_jordan_1 = 2
axiom distance_jordan_2_def : distance_jordan_2 = 1
axiom distance_jordan_3_def : distance_jordan_3 = 5

axiom half_time_condition : ∀ t_2, t_2 = time_steve / 2

-- The proof problem
theorem jordan_running_time : ∀ t_j1 t_j2 t_j3, 
  (t_j1 = time_steve / 2 ∧ 
   t_j2 = t_j1 / 2 ∧ 
   t_j3 = t_j2 * 5) →
  t_j3 = 30 := 
by
  intros t_j1 t_j2 t_j3 h
  sorry

end jordan_running_time_l2001_200134


namespace line_intersects_iff_sufficient_l2001_200109

noncomputable def sufficient_condition (b : ℝ) : Prop :=
b > 1

noncomputable def condition (b : ℝ) : Prop :=
b > 0

noncomputable def line_intersects_hyperbola (b : ℝ) : Prop :=
b > 2 / 3

theorem line_intersects_iff_sufficient (b : ℝ) (h : condition b) : 
  (sufficient_condition b) → (line_intersects_hyperbola b) ∧ ¬(line_intersects_hyperbola b) → (sufficient_condition b) :=
by {
  sorry
}

end line_intersects_iff_sufficient_l2001_200109


namespace range_of_k_l2001_200174

theorem range_of_k (k : ℝ) (h : k ≠ 0) : (k^2 - 6 * k + 8 ≥ 0) ↔ (k ≥ 4 ∨ k ≤ 2) := 
by sorry

end range_of_k_l2001_200174


namespace card_draw_probability_l2001_200135

theorem card_draw_probability : 
  let P1 := (12 / 52 : ℚ) * (4 / 51 : ℚ) * (13 / 50 : ℚ)
  let P2 := (1 / 52 : ℚ) * (3 / 51 : ℚ) * (13 / 50 : ℚ)
  P1 + P2 = (63 / 107800 : ℚ) :=
by
  sorry

end card_draw_probability_l2001_200135


namespace range_of_a_l2001_200170

theorem range_of_a (a : ℝ) : 
  (∃ n : ℕ, (∀ x : ℕ, 1 ≤ x → x ≤ 5 → x < a) ∧ (∀ y : ℕ, x ≥ 1 → y ≥ 6 → y ≥ a)) ↔ (5 < a ∧ a < 6) :=
by
  sorry

end range_of_a_l2001_200170


namespace total_oranges_l2001_200127

theorem total_oranges (joan_oranges : ℕ) (sara_oranges : ℕ) 
                      (h1 : joan_oranges = 37) 
                      (h2 : sara_oranges = 10) :
  joan_oranges + sara_oranges = 47 := by
  sorry

end total_oranges_l2001_200127


namespace fraction_subtraction_equals_one_l2001_200115

theorem fraction_subtraction_equals_one (x : ℝ) (h : x ≠ 1) : (x / (x - 1)) - (1 / (x - 1)) = 1 := 
by sorry

end fraction_subtraction_equals_one_l2001_200115


namespace dave_initial_boxes_l2001_200188

def pieces_per_box : ℕ := 3
def boxes_given_away : ℕ := 5
def pieces_left : ℕ := 21
def total_pieces_given_away := boxes_given_away * pieces_per_box
def total_pieces_initially := total_pieces_given_away + pieces_left

theorem dave_initial_boxes : total_pieces_initially / pieces_per_box = 12 := by
  sorry

end dave_initial_boxes_l2001_200188


namespace find_c_for_root_ratio_l2001_200122

theorem find_c_for_root_ratio :
  ∃ c : ℝ, (∀ x1 x2 : ℝ, (4 * x1^2 - 5 * x1 + c = 0) ∧ (x1 / x2 = -3 / 4)) → c = -75 := 
by {
  sorry
}

end find_c_for_root_ratio_l2001_200122


namespace min_value_sin_cos_l2001_200185

open Real

theorem min_value_sin_cos : ∃ x : ℝ, sin x * cos x = -1 / 2 := by
  sorry

end min_value_sin_cos_l2001_200185


namespace rhombus_diagonal_l2001_200192

theorem rhombus_diagonal (a b : ℝ) (area_triangle : ℝ) (d1 d2 : ℝ)
  (h1 : 2 * area_triangle = a * b)
  (h2 : area_triangle = 75)
  (h3 : a = 20) :
  b = 15 :=
by
  sorry

end rhombus_diagonal_l2001_200192


namespace sufficient_but_not_necessary_condition_ellipse_l2001_200171

theorem sufficient_but_not_necessary_condition_ellipse (a : ℝ) :
  (a^2 > 1 → ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1)) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → (a^2 > 1 ∨ 0 < a^2 ∧ a^2 < 1)) → ¬ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_ellipse_l2001_200171


namespace triangle_ratio_l2001_200143

-- Given conditions:
-- a: one side of the triangle
-- h_a: height corresponding to side a
-- r: inradius of the triangle
-- p: semiperimeter of the triangle

theorem triangle_ratio (a h_a r p : ℝ) (area_formula_1 : p * r = 1 / 2 * a * h_a) :
  (2 * p) / a = h_a / r :=
by {
  sorry
}

end triangle_ratio_l2001_200143


namespace license_plates_possible_l2001_200131

open Function Nat

theorem license_plates_possible :
  let characters := ['B', 'C', 'D', '1', '2', '2', '5']
  let license_plate_length := 4
  let plate_count_with_two_twos := (choose 4 2) * (choose 5 2 * 2!)
  let plate_count_with_one_two := (choose 4 1) * (choose 5 3 * 3!)
  let plate_count_with_no_twos := (choose 5 4) * 4!
  let plate_count_with_three_twos := (choose 4 3) * (choose 4 1)
  plate_count_with_two_twos + plate_count_with_one_two + plate_count_with_no_twos + plate_count_with_three_twos = 496 := 
  sorry

end license_plates_possible_l2001_200131


namespace temperature_at_midnight_l2001_200114

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l2001_200114


namespace seats_with_middle_empty_l2001_200121

-- Define the parameters
def chairs := 5
def people := 4
def middle_empty := 3

-- Define the function to calculate seating arrangements
def number_of_ways (people : ℕ) (chairs : ℕ) (middle_empty : ℕ) : ℕ := 
  if chairs < people + 1 then 0
  else (chairs - 1) * (chairs - 2) * (chairs - 3) * (chairs - 4)

-- The theorem to prove the number of ways given the conditions
theorem seats_with_middle_empty : number_of_ways 4 5 3 = 24 := by
  sorry

end seats_with_middle_empty_l2001_200121


namespace unique_solution_value_l2001_200146

theorem unique_solution_value (k : ℝ) :
  (∃ x : ℝ, x^2 = 2 * x + k ∧ ∀ y : ℝ, y^2 = 2 * y + k → y = x) ↔ k = -1 := 
by
  sorry

end unique_solution_value_l2001_200146


namespace boat_speed_in_still_water_l2001_200195

variable (x : ℝ) -- speed of the boat in still water in km/hr
variable (current_rate : ℝ := 4) -- rate of the current in km/hr
variable (downstream_distance : ℝ := 4.8) -- distance traveled downstream in km
variable (downstream_time : ℝ := 18 / 60) -- time traveled downstream in hours

-- The main theorem stating that the speed of the boat in still water is 12 km/hr
theorem boat_speed_in_still_water : x = 12 :=
by
  -- Express the downstream speed and time relation
  have downstream_speed := x + current_rate
  have distance_relation := downstream_distance = downstream_speed * downstream_time
  -- Simplify and solve for x
  simp at distance_relation
  sorry

end boat_speed_in_still_water_l2001_200195


namespace simplify_fraction_l2001_200137

noncomputable def simplify_expression (x : ℂ) : Prop :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) =
  (x - 3) / (x^2 - 6*x + 8)

theorem simplify_fraction (x : ℂ) : simplify_expression x :=
by
  sorry

end simplify_fraction_l2001_200137


namespace mike_changed_64_tires_l2001_200116

def total_tires_mike_changed (motorcycles : ℕ) (cars : ℕ) (tires_per_motorcycle : ℕ) (tires_per_car : ℕ) : ℕ :=
  motorcycles * tires_per_motorcycle + cars * tires_per_car

theorem mike_changed_64_tires :
  total_tires_mike_changed 12 10 2 4 = 64 :=
by
  sorry

end mike_changed_64_tires_l2001_200116


namespace sum_of_cosines_dihedral_angles_l2001_200104

-- Define the conditions of the problem
def sum_of_plane_angles_trihederal (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Define the problem statement
theorem sum_of_cosines_dihedral_angles (α β γ : ℝ) (d1 d2 d3 : ℝ)
  (h : sum_of_plane_angles_trihederal α β γ) : 
  d1 + d2 + d3 = 1 :=
  sorry

end sum_of_cosines_dihedral_angles_l2001_200104


namespace balloons_in_each_bag_of_round_balloons_l2001_200160

variable (x : ℕ)

-- Definitions based on the problem's conditions
def totalRoundBalloonsBought := 5 * x
def totalLongBalloonsBought := 4 * 30
def remainingRoundBalloons := totalRoundBalloonsBought x - 5
def totalRemainingBalloons := remainingRoundBalloons x + totalLongBalloonsBought

-- Theorem statement based on the question and derived from the conditions and correct answer
theorem balloons_in_each_bag_of_round_balloons : totalRemainingBalloons x = 215 → x = 20 := by
  -- We acknowledge that the proof steps will follow here (omitted as per instructions)
  sorry

end balloons_in_each_bag_of_round_balloons_l2001_200160


namespace lcm_of_12_15_18_is_180_l2001_200177

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end lcm_of_12_15_18_is_180_l2001_200177


namespace tan_sixty_eq_sqrt_three_l2001_200140

theorem tan_sixty_eq_sqrt_three : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
by
  sorry

end tan_sixty_eq_sqrt_three_l2001_200140


namespace mix_solutions_l2001_200172

theorem mix_solutions {x : ℝ} (h : 0.60 * x + 0.75 * (20 - x) = 0.72 * 20) : x = 4 :=
by
-- skipping the proof with sorry
sorry

end mix_solutions_l2001_200172


namespace donation_fifth_sixth_l2001_200154

-- Conditions definitions
def total_donation := 10000
def first_home := 2750
def second_home := 1945
def third_home := 1275
def fourth_home := 1890

-- Proof statement
theorem donation_fifth_sixth : 
  (total_donation - (first_home + second_home + third_home + fourth_home)) = 2140 := by
  sorry

end donation_fifth_sixth_l2001_200154


namespace number_of_nonsimilar_triangles_l2001_200194
-- Import the necessary library

-- Define the problem conditions
def angles_in_arithmetic_progression (a d : ℕ) : Prop :=
  0 < d ∧ d < 30 ∧ 
  (a - d > 0) ∧ (a + d < 180) ∧ -- Ensures positive and valid angles
  (a - d) + a + (a + d) = 180  -- Triangle sum property

-- Declare the theorem
theorem number_of_nonsimilar_triangles : 
  ∃ n : ℕ, n = 29 ∧ ∀ (a d : ℕ), angles_in_arithmetic_progression a d → d < 30 → a = 60 :=
sorry

end number_of_nonsimilar_triangles_l2001_200194


namespace roger_trips_required_l2001_200129

variable (carry_trays_per_trip total_trays : ℕ)

theorem roger_trips_required (h1 : carry_trays_per_trip = 4) (h2 : total_trays = 12) : total_trays / carry_trays_per_trip = 3 :=
by
  -- proof follows
  sorry

end roger_trips_required_l2001_200129


namespace marble_problem_l2001_200120

-- Define the given conditions
def ratio (red blue green : ℕ) : Prop := red * 3 * 4 = blue * 2 * 4 ∧ blue * 2 * 4 = green * 2 * 3

-- The total number of marbles
def total_marbles (red blue green : ℕ) : ℕ := red + blue + green

-- The number of green marbles is given
def green_marbles : ℕ := 36

-- Proving the number of marbles and number of red marbles
theorem marble_problem
  (red blue green : ℕ)
  (h_ratio : ratio red blue green)
  (h_green : green = green_marbles) :
  total_marbles red blue green = 81 ∧ red = 18 :=
by
  sorry

end marble_problem_l2001_200120


namespace groupC_is_all_polyhedra_l2001_200165

inductive GeometricBody
| TriangularPrism : GeometricBody
| QuadrangularPyramid : GeometricBody
| Sphere : GeometricBody
| Cone : GeometricBody
| Cube : GeometricBody
| TruncatedCone : GeometricBody
| HexagonalPyramid : GeometricBody
| Hemisphere : GeometricBody

def isPolyhedron : GeometricBody → Prop
| GeometricBody.TriangularPrism => true
| GeometricBody.QuadrangularPyramid => true
| GeometricBody.Sphere => false
| GeometricBody.Cone => false
| GeometricBody.Cube => true
| GeometricBody.TruncatedCone => false
| GeometricBody.HexagonalPyramid => true
| GeometricBody.Hemisphere => false

def groupA := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Sphere, GeometricBody.Cone]
def groupB := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.TruncatedCone]
def groupC := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.HexagonalPyramid]
def groupD := [GeometricBody.Cone, GeometricBody.TruncatedCone, GeometricBody.Sphere, GeometricBody.Hemisphere]

def allPolyhedra (group : List GeometricBody) : Prop :=
  ∀ b, b ∈ group → isPolyhedron b

theorem groupC_is_all_polyhedra : 
  allPolyhedra groupC ∧
  ¬ allPolyhedra groupA ∧
  ¬ allPolyhedra groupB ∧
  ¬ allPolyhedra groupD :=
by
  sorry

end groupC_is_all_polyhedra_l2001_200165


namespace repeating_decimal_exceeds_decimal_l2001_200148

noncomputable def repeating_decimal_to_fraction : ℚ := 9 / 11
noncomputable def decimal_to_fraction : ℚ := 3 / 4

theorem repeating_decimal_exceeds_decimal :
  repeating_decimal_to_fraction - decimal_to_fraction = 3 / 44 :=
by
  sorry

end repeating_decimal_exceeds_decimal_l2001_200148


namespace decreasing_interval_l2001_200138

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem decreasing_interval : ∀ x ∈ Set.Ioo (Real.pi / 6) (5 * Real.pi / 6), 
  (1 / 2 - Real.sin x) < 0 := sorry

end decreasing_interval_l2001_200138


namespace twenty_mul_b_sub_a_not_integer_l2001_200198

theorem twenty_mul_b_sub_a_not_integer {a b : ℝ} (hneq : a ≠ b) (hno_roots : ∀ x : ℝ,
  (x^2 + 20 * a * x + 10 * b) * (x^2 + 20 * b * x + 10 * a) ≠ 0) :
  ¬ ∃ n : ℤ, 20 * (b - a) = n :=
sorry

end twenty_mul_b_sub_a_not_integer_l2001_200198


namespace car_a_speed_l2001_200163

theorem car_a_speed (d_gap : ℕ) (v_B : ℕ) (t : ℕ) (d_ahead : ℕ) (v_A : ℕ) 
  (h1 : d_gap = 24) (h2 : v_B = 50) (h3 : t = 4) (h4 : d_ahead = 8)
  (h5 : v_A = (d_gap + v_B * t + d_ahead) / t) : v_A = 58 :=
by {
  exact (sorry : v_A = 58)
}

end car_a_speed_l2001_200163


namespace describe_T_l2001_200196

def T : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ (
      (5 = x + 3 ∧ y - 6 ≤ 5) ∨
      (5 = y - 6 ∧ x + 3 ≤ 5) ∨
      (x + 3 = y - 6 ∧ x + 3 ≤ 5 ∧ y - 6 ≤ 5)
  )}

theorem describe_T : T = { p | ∃ x y : ℝ, p = (2, y) ∧ y ≤ 11 ∨
                                      p = (x, 11) ∧ x ≤ 2 ∨
                                      p = (x, x + 9) ∧ x ≤ 2 ∧ x + 9 ≤ 11 } :=
by
  sorry

end describe_T_l2001_200196


namespace find_linear_function_b_l2001_200181

theorem find_linear_function_b (b : ℝ) :
  (∃ b, (∀ x y, y = 2 * x + b - 2 → (x = -1 ∧ y = 0)) → b = 4) :=
sorry

end find_linear_function_b_l2001_200181


namespace find_number_l2001_200179

-- Definitions based on conditions
def sum : ℕ := 555 + 445
def difference : ℕ := 555 - 445
def quotient : ℕ := 2 * difference
def remainder : ℕ := 70
def divisor : ℕ := sum

-- Statement to be proved
theorem find_number : (divisor * quotient + remainder) = 220070 := by
  sorry

end find_number_l2001_200179


namespace additional_savings_l2001_200145

def window_price : ℕ := 100

def special_offer (windows_purchased : ℕ) : ℕ :=
  windows_purchased + windows_purchased / 6 * 2

def dave_windows : ℕ := 10

def doug_windows : ℕ := 12

def total_windows := dave_windows + doug_windows

def calculate_windows_cost (windows_needed : ℕ) : ℕ :=
  if windows_needed % 8 = 0 then (windows_needed / 8) * 6 * window_price
  else ((windows_needed / 8) * 6 + (windows_needed % 8)) * window_price

def separate_savings : ℕ :=
  window_price * (dave_windows + doug_windows) - (calculate_windows_cost dave_windows + calculate_windows_cost doug_windows)

def combined_savings : ℕ :=
  window_price * total_windows - calculate_windows_cost total_windows

theorem additional_savings :
  separate_savings + 200 = combined_savings :=
sorry

end additional_savings_l2001_200145


namespace remaining_area_after_cut_l2001_200189

theorem remaining_area_after_cut
  (cell_side_length : ℝ)
  (grid_side_length : ℕ)
  (total_area : ℝ)
  (removed_area : ℝ)
  (hyp1 : cell_side_length = 1)
  (hyp2 : grid_side_length = 6)
  (hyp3 : total_area = (grid_side_length * grid_side_length) * cell_side_length * cell_side_length) 
  (hyp4 : removed_area = 9) :
  total_area - removed_area = 27 := by
  sorry

end remaining_area_after_cut_l2001_200189


namespace translation_preserves_coordinates_l2001_200178

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end translation_preserves_coordinates_l2001_200178


namespace wrench_weight_relation_l2001_200101

variables (h w : ℕ)

theorem wrench_weight_relation (h w : ℕ) 
  (cond : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) : w = 2 * h := 
by sorry

end wrench_weight_relation_l2001_200101


namespace judy_shopping_trip_l2001_200152

-- Define the quantities and prices of the items
def num_carrots : ℕ := 5
def price_carrot : ℕ := 1
def num_milk : ℕ := 4
def price_milk : ℕ := 3
def num_pineapples : ℕ := 2
def price_pineapple : ℕ := 4
def num_flour : ℕ := 2
def price_flour : ℕ := 5
def price_ice_cream : ℕ := 7

-- Define the promotion conditions
def pineapple_promotion : ℕ := num_pineapples / 2

-- Define the coupon condition
def coupon_threshold : ℕ := 40
def coupon_value : ℕ := 10

-- Define the total cost without coupon
def total_cost : ℕ := 
  (num_carrots * price_carrot) + 
  (num_milk * price_milk) +
  (pineapple_promotion * price_pineapple) +
  (num_flour * price_flour) +
  price_ice_cream

-- Define the final cost considering the coupon condition
def final_cost : ℕ :=
  if total_cost < coupon_threshold then total_cost else total_cost - coupon_value

-- The theorem to be proven
theorem judy_shopping_trip : final_cost = 38 := by
  sorry

end judy_shopping_trip_l2001_200152


namespace sum_first_3m_terms_l2001_200128

variable (m : ℕ) (a₁ d : ℕ)

def S (n : ℕ) := n * a₁ + (n * (n - 1)) / 2 * d

-- Given conditions
axiom sum_first_m_terms : S m = 0
axiom sum_first_2m_terms : S (2 * m) = 0

-- Theorem to be proved
theorem sum_first_3m_terms : S (3 * m) = 210 :=
by
  sorry

end sum_first_3m_terms_l2001_200128


namespace A_odot_B_correct_l2001_200155

open Set

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x < 0 ∨ x > 2 }
def A_union_B : Set ℝ := A ∪ B
def A_inter_B : Set ℝ := A ∩ B
def A_odot_B : Set ℝ := { x | x ∈ A_union_B ∧ x ∉ A_inter_B }

theorem A_odot_B_correct : A_odot_B = (Iio 0) ∪ Icc 1 2 :=
by
  sorry

end A_odot_B_correct_l2001_200155


namespace maximum_xy_l2001_200132

variable {a b c x y : ℝ}

theorem maximum_xy 
  (h1 : a * x + b * y + 2 * c = 0)
  (h2 : c ≠ 0)
  (h3 : a * b - c^2 ≥ 0) :
  ∃ (m : ℝ), m = x * y ∧ m ≤ 1 :=
sorry

end maximum_xy_l2001_200132


namespace infinite_solutions_eq_l2001_200161

/-
Proving that the equation x - y + z = 1 has infinite solutions under the conditions:
1. x, y, z are distinct positive integers.
2. The product of any two numbers is divisible by the third one.
-/
theorem infinite_solutions_eq (x y z : ℕ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) 
(h4 : ∃ m n k : ℕ, x = m * n ∧ y = n * k ∧ z = m * k)
(h5 : (x*y) % z = 0) (h6 : (y*z) % x = 0) (h7 : (z*x) % y = 0) : 
∃ (m : ℕ), x - y + z = 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by sorry

end infinite_solutions_eq_l2001_200161


namespace area_between_tangent_circles_l2001_200175

theorem area_between_tangent_circles (r : ℝ) (h_r : r > 0) :
  let area_trapezoid := 4 * r^2 * Real.sqrt 3
  let area_sector1 := π * r^2 / 3
  let area_sector2 := 3 * π * r^2 / 2
  area_trapezoid - (area_sector1 + area_sector2) = r^2 * (24 * Real.sqrt 3 - 11 * π) / 6 := by
  sorry

end area_between_tangent_circles_l2001_200175


namespace vector_dot_product_problem_l2001_200147

variables {a b : ℝ}

theorem vector_dot_product_problem (h1 : a + 2 * b = 0) (h2 : (a + b) * a = 2) : a * b = -2 :=
sorry

end vector_dot_product_problem_l2001_200147


namespace ab_bc_ca_negative_l2001_200180

theorem ab_bc_ca_negative (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : abc > 0) : ab + bc + ca < 0 :=
sorry

end ab_bc_ca_negative_l2001_200180


namespace arithmetic_sequence_problem_l2001_200156

theorem arithmetic_sequence_problem 
    (a : ℕ → ℝ)  -- Define the arithmetic sequence as a function from natural numbers to reals
    (a1 : ℝ)  -- Represent a₁ as a1
    (a8 : ℝ)  -- Represent a₈ as a8
    (a9 : ℝ)  -- Represent a₉ as a9
    (a10 : ℝ)  -- Represent a₁₀ as a10
    (a15 : ℝ)  -- Represent a₁₅ as a15
    (h1 : a 1 = a1)  -- Hypothesis that a(1) is represented by a1
    (h8 : a 8 = a8)  -- Hypothesis that a(8) is represented by a8
    (h9 : a 9 = a9)  -- Hypothesis that a(9) is represented by a9
    (h10 : a 10 = a10)  -- Hypothesis that a(10) is represented by a10
    (h15 : a 15 = a15)  -- Hypothesis that a(15) is represented by a15
    (h_condition : a1 + 2 * a8 + a15 = 96)  -- Condition of the problem
    : 2 * a9 - a10 = 24 := 
sorry

end arithmetic_sequence_problem_l2001_200156


namespace cost_per_blue_shirt_l2001_200176

theorem cost_per_blue_shirt :
  let pto_spent := 2317
  let num_kindergarten := 101
  let cost_orange := 5.80
  let total_orange := num_kindergarten * cost_orange

  let num_first_grade := 113
  let cost_yellow := 5
  let total_yellow := num_first_grade * cost_yellow

  let num_third_grade := 108
  let cost_green := 5.25
  let total_green := num_third_grade * cost_green

  let total_other_shirts := total_orange + total_yellow + total_green
  let pto_spent_on_blue := pto_spent - total_other_shirts

  let num_second_grade := 107
  let cost_per_blue_shirt := pto_spent_on_blue / num_second_grade

  cost_per_blue_shirt = 5.60 :=
by
  sorry

end cost_per_blue_shirt_l2001_200176


namespace electricity_cost_per_kWh_is_14_cents_l2001_200187

-- Define the conditions
def powerUsagePerHour : ℕ := 125 -- watts
def dailyUsageHours : ℕ := 4 -- hours
def weeklyCostInCents : ℕ := 49 -- cents
def daysInWeek : ℕ := 7 -- days
def wattsToKilowattsFactor : ℕ := 1000 -- conversion factor

-- Define a function to calculate the cost per kWh
def costPerKwh (powerUsagePerHour : ℕ) (dailyUsageHours : ℕ) (weeklyCostInCents : ℕ) (daysInWeek : ℕ) (wattsToKilowattsFactor : ℕ) : ℕ :=
  let dailyConsumption := powerUsagePerHour * dailyUsageHours
  let weeklyConsumption := dailyConsumption * daysInWeek
  let weeklyConsumptionInKwh := weeklyConsumption / wattsToKilowattsFactor
  weeklyCostInCents / weeklyConsumptionInKwh

-- State the theorem
theorem electricity_cost_per_kWh_is_14_cents :
  costPerKwh powerUsagePerHour dailyUsageHours weeklyCostInCents daysInWeek wattsToKilowattsFactor = 14 :=
by
  sorry

end electricity_cost_per_kWh_is_14_cents_l2001_200187


namespace abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l2001_200102

variable (x b : ℝ)

theorem abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1 :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := sorry

end abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l2001_200102


namespace quadratic_has_exactly_one_root_l2001_200105

noncomputable def discriminant (b c : ℝ) : ℝ :=
b^2 - 4 * c

noncomputable def f (x b c : ℝ) : ℝ :=
x^2 + b * x + c

noncomputable def transformed_f (x b c : ℝ) : ℝ :=
(x - 2020)^2 + b * (x - 2020) + c

theorem quadratic_has_exactly_one_root (b c : ℝ)
  (h_discriminant : discriminant b c = 2020) :
  ∃! x : ℝ, f (x - 2020) b c + f x b c = 0 :=
sorry

end quadratic_has_exactly_one_root_l2001_200105


namespace circle_radius_zero_l2001_200162

theorem circle_radius_zero (x y : ℝ) : 2*x^2 - 8*x + 2*y^2 + 4*y + 10 = 0 → (x - 2)^2 + (y + 1)^2 = 0 :=
by
  intro h
  sorry

end circle_radius_zero_l2001_200162


namespace inequality_proof_l2001_200118

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a + b + c + a * b + b * c + c * a + a * b * c = 7)

theorem inequality_proof : 
  (Real.sqrt (a ^ 2 + b ^ 2 + 2) + Real.sqrt (b ^ 2 + c ^ 2 + 2) + Real.sqrt (c ^ 2 + a ^ 2 + 2)) ≥ 6 := by
  sorry

end inequality_proof_l2001_200118


namespace value_of_x_l2001_200169

variable {x y z : ℤ}

theorem value_of_x
  (h1 : x + y = 31)
  (h2 : y + z = 47)
  (h3 : x + z = 52)
  (h4 : y + z = x + 16) :
  x = 31 := by
  sorry

end value_of_x_l2001_200169


namespace min_value_abs_expr_l2001_200144

noncomputable def minExpr (a b : ℝ) : ℝ :=
  |a + b| + |(1 / (a + 1)) - b|

theorem min_value_abs_expr (a b : ℝ) (h₁ : a ≠ -1) : minExpr a b ≥ 1 ∧ (minExpr a b = 1 ↔ a = 0) :=
by
  sorry

end min_value_abs_expr_l2001_200144


namespace cost_per_book_l2001_200164

theorem cost_per_book
  (books_sold_each_time : ℕ)
  (people_bought : ℕ)
  (income_per_book : ℕ)
  (profit : ℕ)
  (total_income : ℕ := books_sold_each_time * people_bought * income_per_book)
  (total_cost : ℕ := total_income - profit)
  (total_books : ℕ := books_sold_each_time * people_bought)
  (cost_per_book : ℕ := total_cost / total_books) :
  books_sold_each_time = 2 ->
  people_bought = 4 ->
  income_per_book = 20 ->
  profit = 120 ->
  cost_per_book = 5 :=
  by intros; sorry

end cost_per_book_l2001_200164


namespace positive_difference_of_solutions_l2001_200119

theorem positive_difference_of_solutions:
  ∀ (s : ℝ), s ≠ -3 → (s^2 - 5*s - 24) / (s + 3) = 3*s + 10 →
  abs (-1 - (-27)) = 26 :=
by
  sorry

end positive_difference_of_solutions_l2001_200119


namespace time_to_walk_without_walkway_l2001_200111

theorem time_to_walk_without_walkway 
  (vp vw : ℝ) 
  (h1 : (vp + vw) * 40 = 80) 
  (h2 : (vp - vw) * 120 = 80) : 
  80 / vp = 60 :=
by
  sorry

end time_to_walk_without_walkway_l2001_200111


namespace product_f_g_l2001_200199

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem product_f_g (x : ℝ) (hx : 0 < x) : f x * g x = Real.sqrt (x + 1) := 
by 
  sorry

end product_f_g_l2001_200199


namespace borgnine_lizards_l2001_200106

theorem borgnine_lizards (chimps lions tarantulas total_legs : ℕ) (legs_per_chimp legs_per_lion legs_per_tarantula legs_per_lizard lizards : ℕ)
  (H_chimps : chimps = 12)
  (H_lions : lions = 8)
  (H_tarantulas : tarantulas = 125)
  (H_total_legs : total_legs = 1100)
  (H_legs_per_chimp : legs_per_chimp = 4)
  (H_legs_per_lion : legs_per_lion = 4)
  (H_legs_per_tarantula : legs_per_tarantula = 8)
  (H_legs_per_lizard : legs_per_lizard = 4)
  (H_seen_legs : total_legs = (chimps * legs_per_chimp) + (lions * legs_per_lion) + (tarantulas * legs_per_tarantula) + (lizards * legs_per_lizard)) :
  lizards = 5 := 
by
  sorry

end borgnine_lizards_l2001_200106


namespace arc_length_sector_l2001_200149

theorem arc_length_sector (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 150 * Real.pi / 180) :
  θ * r = 5 * Real.pi / 2 :=
by
  rw [h_r, h_θ]
  sorry

end arc_length_sector_l2001_200149


namespace tan_alpha_minus_pi_over_4_l2001_200166

theorem tan_alpha_minus_pi_over_4 (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
  (Real.tan (α - π / 4) = -1/7) ∨ (Real.tan (α - π / 4) = -7) :=
by
  sorry

end tan_alpha_minus_pi_over_4_l2001_200166


namespace simplify_fraction_l2001_200124

theorem simplify_fraction :
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end simplify_fraction_l2001_200124


namespace part_one_part_two_l2001_200103

variable (α : Real) (h : Real.tan α = 2)

theorem part_one (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6 / 11 := 
by
  sorry

theorem part_two (h : Real.tan α = 2) : 
  (1 / 4 * Real.sin α ^ 2 + 1 / 3 * Real.sin α * Real.cos α + 1 / 2 * Real.cos α ^ 2 + 1) = 43 / 30 := 
by
  sorry

end part_one_part_two_l2001_200103


namespace triangle_def_ef_value_l2001_200123

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end triangle_def_ef_value_l2001_200123


namespace mn_value_l2001_200141
open Real

-- Define the conditions
def L_1_scenario_1 (m n : ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), θ₁ = 2 * θ₂ ∧ m = tan θ₁ ∧ n = tan θ₂ ∧ m = 4 * n

-- State the theorem
theorem mn_value (m n : ℝ) (hL1 : L_1_scenario_1 m n) (hm : m ≠ 0) : m * n = 2 :=
  sorry

end mn_value_l2001_200141


namespace denis_fourth_board_score_l2001_200158

theorem denis_fourth_board_score :
  ∀ (darts_per_board points_first_board points_second_board points_third_board points_total_boards : ℕ),
    darts_per_board = 3 →
    points_first_board = 30 →
    points_second_board = 38 →
    points_third_board = 41 →
    points_total_boards = (points_first_board + points_second_board + points_third_board) / 2 →
    points_total_boards = 34 :=
by
  intros darts_per_board points_first_board points_second_board points_third_board points_total_boards h1 h2 h3 h4 h5
  sorry

end denis_fourth_board_score_l2001_200158


namespace geometric_sequence_min_n_l2001_200159

theorem geometric_sequence_min_n (n : ℕ) (h : 2^(n + 1) - 2 - n > 1020) : n ≥ 10 :=
sorry

end geometric_sequence_min_n_l2001_200159


namespace arithmetic_sequence_sum_l2001_200107

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_roots : (a 3) * (a 10) - 3 * (a 3 + a 10) - 5 = 0) : a 5 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l2001_200107


namespace jane_started_babysitting_at_age_18_l2001_200133

-- Define the age Jane started babysitting
def jane_starting_age := 18

-- State Jane's current age
def jane_current_age : ℕ := 34

-- State the years since Jane stopped babysitting
def years_since_jane_stopped := 12

-- Calculate Jane's age when she stopped babysitting
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped

-- State the current age of the oldest person she could have babysat
def current_oldest_child_age : ℕ := 25

-- Calculate the age of the oldest child when Jane stopped babysitting
def age_oldest_child_when_stopped : ℕ := current_oldest_child_age - years_since_jane_stopped

-- State the condition that the child was no more than half her age at the time
def child_age_condition (jane_age : ℕ) (child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- The theorem to prove the age Jane started babysitting
theorem jane_started_babysitting_at_age_18
  (jane_current : jane_current_age = 34)
  (years_stopped : years_since_jane_stopped = 12)
  (current_oldest : current_oldest_child_age = 25)
  (age_when_stopped : jane_age_when_stopped = 22)
  (child_when_stopped : age_oldest_child_when_stopped = 13)
  (child_condition : ∀ {j : ℕ}, child_age_condition j age_oldest_child_when_stopped → False) :
  jane_starting_age = 18 :=
sorry

end jane_started_babysitting_at_age_18_l2001_200133


namespace square_plot_area_l2001_200186

theorem square_plot_area (cost_per_foot : ℕ) (total_cost : ℕ) (P : ℕ) :
  cost_per_foot = 54 →
  total_cost = 3672 →
  P = 4 * (total_cost / (4 * cost_per_foot)) →
  (total_cost / (4 * cost_per_foot)) ^ 2 = 289 :=
by
  intros h_cost_per_foot h_total_cost h_perimeter
  sorry

end square_plot_area_l2001_200186


namespace mod_equiv_1_l2001_200190

theorem mod_equiv_1 : (179 * 933 / 7) % 50 = 1 := by
  sorry

end mod_equiv_1_l2001_200190


namespace subtract_fifteen_result_l2001_200191

theorem subtract_fifteen_result (x : ℕ) (h : x / 10 = 6) : x - 15 = 45 :=
by
  sorry

end subtract_fifteen_result_l2001_200191


namespace worm_length_l2001_200110

theorem worm_length (l1 l2 : ℝ) (h1 : l1 = 0.8) (h2 : l2 = l1 + 0.7) : l1 = 0.8 :=
by
  exact h1

end worm_length_l2001_200110


namespace tangent_sum_formula_application_l2001_200112

-- Define the problem's parameters and statement
noncomputable def thirty_three_degrees_radian := Real.pi * 33 / 180
noncomputable def seventeen_degrees_radian := Real.pi * 17 / 180
noncomputable def twenty_eight_degrees_radian := Real.pi * 28 / 180

theorem tangent_sum_formula_application :
  Real.tan seventeen_degrees_radian + Real.tan twenty_eight_degrees_radian + Real.tan seventeen_degrees_radian * Real.tan twenty_eight_degrees_radian = 1 := 
sorry

end tangent_sum_formula_application_l2001_200112


namespace A_contribution_is_500_l2001_200167

-- Define the contributions
variables (A B C : ℕ)

-- Total amount spent
def total_contribution : ℕ := 820

-- Given ratios
def ratio_A_to_B : ℕ × ℕ := (5, 2)
def ratio_B_to_C : ℕ × ℕ := (5, 3)

-- Condition stating the sum of contributions
axiom sum_contribution : A + B + C = total_contribution

-- Conditions stating the ratios
axiom ratio_A_B : 5 * B = 2 * A
axiom ratio_B_C : 5 * C = 3 * B

-- The statement to prove
theorem A_contribution_is_500 : A = 500 :=
by
  sorry

end A_contribution_is_500_l2001_200167


namespace arith_seq_sum_ratio_l2001_200168

theorem arith_seq_sum_ratio 
  (S : ℕ → ℝ) 
  (a1 d : ℝ) 
  (h1 : S 1 = 1) 
  (h2 : (S 4) / (S 2) = 4) :
  (S 6) / (S 4) = 9 / 4 :=
sorry

end arith_seq_sum_ratio_l2001_200168


namespace percent_increase_correct_l2001_200193

variable (p_initial p_final : ℝ)

theorem percent_increase_correct : p_initial = 25 → p_final = 28 → (p_final - p_initial) / p_initial * 100 = 12 := by
  intros h_initial h_final
  sorry

end percent_increase_correct_l2001_200193


namespace locus_of_Q_max_area_of_triangle_OPQ_l2001_200139

open Real

theorem locus_of_Q (x y : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x = 3 * x_0 ∧ y = 4 * y_0 →
  (x / 6)^2 + (y / 4)^2 = 1 :=
sorry

theorem max_area_of_triangle_OPQ (S : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x_0 > 0 ∧ y_0 > 0 →
  S <= sqrt 3 / 2 :=
sorry

end locus_of_Q_max_area_of_triangle_OPQ_l2001_200139


namespace find_a7_l2001_200108

-- Define the arithmetic sequence
def a (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions
def S5 : ℤ := 25
def a2 : ℤ := 3

-- Main Goal: Find a_7
theorem find_a7 (a1 d : ℤ) (h1 : sum_first_n_terms a1 d 5 = S5)
                     (h2 : a a1 d 2 = a2) :
  a a1 d 7 = 13 := 
sorry

end find_a7_l2001_200108


namespace pricePerRedStamp_l2001_200126

namespace StampCollection

-- Definitions for the conditions
def totalRedStamps : ℕ := 20
def soldRedStamps : ℕ := 20
def totalBlueStamps : ℕ := 80
def soldBlueStamps : ℕ := 80
def pricePerBlueStamp : ℝ := 0.8
def totalYellowStamps : ℕ := 7
def pricePerYellowStamp : ℝ := 2
def totalTargetEarnings : ℝ := 100

-- Derived definitions from conditions
def earningsFromBlueStamps : ℝ := soldBlueStamps * pricePerBlueStamp
def earningsFromYellowStamps : ℝ := totalYellowStamps * pricePerYellowStamp
def earningsRequiredFromRedStamps : ℝ := totalTargetEarnings - (earningsFromBlueStamps + earningsFromYellowStamps)

-- The statement asserting the main proof obligation
theorem pricePerRedStamp :
  (earningsRequiredFromRedStamps / soldRedStamps) = 1.10 :=
sorry

end StampCollection

end pricePerRedStamp_l2001_200126


namespace apples_taken_from_each_basket_l2001_200182

theorem apples_taken_from_each_basket (total_apples : ℕ) (baskets : ℕ) (remaining_apples_per_basket : ℕ) 
(h1 : total_apples = 64) (h2 : baskets = 4) (h3 : remaining_apples_per_basket = 13) : 
(total_apples - (remaining_apples_per_basket * baskets)) / baskets = 3 :=
sorry

end apples_taken_from_each_basket_l2001_200182
