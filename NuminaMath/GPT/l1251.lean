import Mathlib

namespace domain_of_f_l1251_125175

noncomputable def f (x : ℝ) := Real.log x / Real.log 6

noncomputable def g (x : ℝ) := Real.log x / Real.log 5

noncomputable def h (x : ℝ) := Real.log x / Real.log 3

open Set

theorem domain_of_f :
  (∀ x, x > 7776 → ∃ y, y = (h ∘ g ∘ f) x) :=
by
  sorry

end domain_of_f_l1251_125175


namespace time_A_to_complete_work_alone_l1251_125165

theorem time_A_to_complete_work_alone :
  ∃ (x : ℝ), (1 / x) + (1 / 20) = (1 / 8.571428571428571) ∧ x = 15 :=
by
  sorry

end time_A_to_complete_work_alone_l1251_125165


namespace total_truck_loads_l1251_125164

-- Using definitions from conditions in (a)
def sand : ℝ := 0.16666666666666666
def dirt : ℝ := 0.3333333333333333
def cement : ℝ := 0.16666666666666666

-- The proof statement based on the correct answer in (b)
theorem total_truck_loads : sand + dirt + cement = 0.6666666666666666 := 
by
  sorry

end total_truck_loads_l1251_125164


namespace range_of_years_of_service_l1251_125190

theorem range_of_years_of_service : 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  ∃ min max, (min ∈ years ∧ max ∈ years ∧ (max - min = 14)) :=
by 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  use 3, 17 
  sorry

end range_of_years_of_service_l1251_125190


namespace tiling_impossible_l1251_125182

theorem tiling_impossible (T2 T14 : ℕ) :
  let S_before := 2 * T2
  let S_after := 2 * (T2 - 1) + 1 
  S_after ≠ S_before :=
sorry

end tiling_impossible_l1251_125182


namespace quadratic_eq_one_solution_m_eq_49_div_12_l1251_125113

theorem quadratic_eq_one_solution_m_eq_49_div_12 (m : ℝ) : 
  (∃ m, ∀ x, 3 * x ^ 2 - 7 * x + m = 0 → (b^2 - 4 * a * c = 0) → m = 49 / 12) :=
by
  sorry

end quadratic_eq_one_solution_m_eq_49_div_12_l1251_125113


namespace fraction_between_stops_l1251_125162

/-- Prove that the fraction of the remaining distance traveled between Maria's first and second stops is 1/4. -/
theorem fraction_between_stops (total_distance first_stop_distance remaining_distance final_leg_distance : ℝ)
  (h_total : total_distance = 400)
  (h_first_stop : first_stop_distance = total_distance / 2)
  (h_remaining : remaining_distance = total_distance - first_stop_distance)
  (h_final_leg : final_leg_distance = 150)
  (h_second_leg : remaining_distance - final_leg_distance = 50) :
  50 / remaining_distance = 1 / 4 :=
by
  { sorry }

end fraction_between_stops_l1251_125162


namespace point_to_line_distance_l1251_125161

theorem point_to_line_distance :
  let circle_center : ℝ×ℝ := (0, 1)
  let A : ℝ := -1
  let B : ℝ := 1
  let C : ℝ := -2
  let line_eq (x y : ℝ) := A * x + B * y + C == 0
  ∀ (x0 : ℝ) (y0 : ℝ),
    circle_center = (x0, y0) →
    (|A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)) = (Real.sqrt 2 / 2) := 
by 
  intros
  -- Proof goes here
  sorry -- Placeholder for the proof.

end point_to_line_distance_l1251_125161


namespace avg10_students_correct_l1251_125187

-- Definitions for the conditions
def avg15_students : ℝ := 70
def num15_students : ℕ := 15
def num10_students : ℕ := 10
def num25_students : ℕ := num15_students + num10_students
def avg25_students : ℝ := 80

-- Total percentage calculation based on conditions
def total_perc25_students := num25_students * avg25_students
def total_perc15_students := num15_students * avg15_students

-- The average percent of the 10 students, based on the conditions and given average for 25 students.
theorem avg10_students_correct : 
  (total_perc25_students - total_perc15_students) / (num10_students : ℝ) = 95 := by
  sorry

end avg10_students_correct_l1251_125187


namespace symmetric_point_x_axis_l1251_125170

theorem symmetric_point_x_axis (x y z : ℝ) : 
    (x, -y, -z) = (-2, -1, -9) :=
by 
  sorry

end symmetric_point_x_axis_l1251_125170


namespace sqrt_range_l1251_125125

theorem sqrt_range (x : ℝ) : 3 - 2 * x ≥ 0 ↔ x ≤ 3 / 2 := 
    sorry

end sqrt_range_l1251_125125


namespace projectile_first_reach_height_56_l1251_125142

theorem projectile_first_reach_height_56 (t : ℝ) (h1 : ∀ t, y = -16 * t^2 + 60 * t) :
    (∃ t : ℝ, y = 56 ∧ t = 1.75 ∧ (∀ t', t' < 1.75 → y ≠ 56)) :=
by
  sorry

end projectile_first_reach_height_56_l1251_125142


namespace number_of_trips_l1251_125174

theorem number_of_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ)
  (h1 : bags_per_trip = 10)
  (h2 : weight_per_bag = 50)
  (h3 : total_weight = 10000) : 
  total_weight / (bags_per_trip * weight_per_bag) = 20 :=
by
  sorry

end number_of_trips_l1251_125174


namespace A_minus_B_l1251_125116

theorem A_minus_B (x y m n A B : ℤ) (hx : x > y) (hx1 : x + y = 7) (hx2 : x * y = 12)
                  (hm : m > n) (hm1 : m + n = 13) (hm2 : m^2 + n^2 = 97)
                  (hA : A = x - y) (hB : B = m - n) :
                  A - B = -4 := by
  sorry

end A_minus_B_l1251_125116


namespace problem_l1251_125197

theorem problem (θ : ℝ) (htan : Real.tan θ = 1 / 3) : Real.cos θ ^ 2 + 2 * Real.sin θ = 6 / 5 := 
by
  sorry

end problem_l1251_125197


namespace find_number_of_appliances_l1251_125198

-- Declare the constants related to the problem.
def commission_per_appliance : ℝ := 50
def commission_percent : ℝ := 0.1
def total_selling_price : ℝ := 3620
def total_commission : ℝ := 662

-- Define the theorem to solve for the number of appliances sold.
theorem find_number_of_appliances (n : ℝ) 
  (H : n * commission_per_appliance + commission_percent * total_selling_price = total_commission) : 
  n = 6 := 
sorry

end find_number_of_appliances_l1251_125198


namespace total_metal_rods_needed_l1251_125135

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l1251_125135


namespace exists_ratios_eq_l1251_125199

theorem exists_ratios_eq (a b z : ℕ) (ha : 0 < a) (hb : 0 < b) (hz : 0 < z) (h : a * b = z^2 + 1) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (a : ℚ) / b = (x^2 + 1) / (y^2 + 1) :=
by
  sorry

end exists_ratios_eq_l1251_125199


namespace find_a_l1251_125159
-- Import necessary Lean libraries

-- Define the function and its maximum value condition
def f (a x : ℝ) := -x^2 + 2*a*x + 1 - a

def has_max_value (f : ℝ → ℝ) (M : ℝ) (interval : Set ℝ) : Prop :=
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = M

theorem find_a (a : ℝ) :
  has_max_value (f a) 2 (Set.Icc 0 1) → (a = -1 ∨ a = 2) :=
by
  sorry

end find_a_l1251_125159


namespace Liz_team_deficit_l1251_125168

theorem Liz_team_deficit :
  ∀ (initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points : ℕ),
    initial_deficit = 20 →
    liz_free_throws = 5 →
    liz_three_pointers = 3 →
    liz_jump_shshots = 4 →
    opponent_points = 10 →
    (initial_deficit - (liz_free_throws * 1 + liz_three_pointers * 3 + liz_jump_shshots * 2 - opponent_points)) = 8 := by
  intros initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points
  intros h_initial_deficit h_liz_free_throws h_liz_three_pointers h_liz_jump_shots h_opponent_points
  sorry

end Liz_team_deficit_l1251_125168


namespace triangle_angle_C_30_degrees_l1251_125171

theorem triangle_angle_C_30_degrees 
  (A B C : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) 
  (h3 : A + B + C = 180) 
  : C = 30 :=
  sorry

end triangle_angle_C_30_degrees_l1251_125171


namespace inequality_of_four_numbers_l1251_125103

theorem inequality_of_four_numbers 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a ≤ 3 * b) (h2 : b ≤ 3 * a) (h3 : a ≤ 3 * c)
  (h4 : c ≤ 3 * a) (h5 : a ≤ 3 * d) (h6 : d ≤ 3 * a)
  (h7 : b ≤ 3 * c) (h8 : c ≤ 3 * b) (h9 : b ≤ 3 * d)
  (h10 : d ≤ 3 * b) (h11 : c ≤ 3 * d) (h12 : d ≤ 3 * c) : 
  a^2 + b^2 + c^2 + d^2 < 2 * (ab + ac + ad + bc + bd + cd) :=
sorry

end inequality_of_four_numbers_l1251_125103


namespace max_q_minus_r_839_l1251_125111

theorem max_q_minus_r_839 : ∃ (q r : ℕ), (839 = 19 * q + r) ∧ (0 ≤ r ∧ r < 19) ∧ q - r = 41 :=
by
  sorry

end max_q_minus_r_839_l1251_125111


namespace purely_imaginary_sol_l1251_125141

theorem purely_imaginary_sol (x : ℝ) 
  (h1 : (x^2 - 1) = 0)
  (h_imag : (x^2 + 3 * x + 2) ≠ 0) :
  x = 1 :=
sorry

end purely_imaginary_sol_l1251_125141


namespace fraction_not_integer_l1251_125107

theorem fraction_not_integer (a b : ℤ) : ¬ (∃ k : ℤ, (a^2 + b^2) = k * (a^2 - b^2)) :=
sorry

end fraction_not_integer_l1251_125107


namespace div_eq_210_over_79_l1251_125180

def a_at_b (a b : ℕ) : ℤ := a^2 * b - a * (b^2)
def a_hash_b (a b : ℕ) : ℤ := a^2 + b^2 - a * b

theorem div_eq_210_over_79 : (a_at_b 10 3) / (a_hash_b 10 3) = 210 / 79 :=
by
  -- This is a placeholder and needs to be filled with the actual proof.
  sorry

end div_eq_210_over_79_l1251_125180


namespace find_foci_l1251_125194

def hyperbolaFoci : Prop :=
  let eq := ∀ x y, 2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0
  ∃ foci : ℝ × ℝ, foci = (-2 - Real.sqrt (5 / 6), -2) ∨ foci = (-2 + Real.sqrt (5 / 6), -2)

theorem find_foci : hyperbolaFoci :=
by
  sorry

end find_foci_l1251_125194


namespace determinant_example_l1251_125160

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end determinant_example_l1251_125160


namespace box_surface_area_is_276_l1251_125146

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Define the pricing function
def pricing (x y z : ℝ) : ℝ := 0.30 * x + 0.40 * y + 0.50 * z

-- Define the condition for the box fee
def box_fee (x y z : ℝ) (fee : ℝ) := pricing x y z = fee

-- Define the constraint that no faces are squares
def no_square_faces (l w h : ℝ) : Prop := 
  l ≠ w ∧ w ≠ h ∧ h ≠ l

-- Define the surface area calculation
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The main theorem stating the problem
theorem box_surface_area_is_276 (l w h : ℝ) 
  (H1 : box_fee l w h 8.10 ∧ box_fee w h l 8.10)
  (H2 : box_fee l w h 8.70 ∧ box_fee w h l 8.70)
  (H3 : no_square_faces l w h) : 
  surface_area l w h = 276 := 
sorry

end box_surface_area_is_276_l1251_125146


namespace plate_729_driving_days_l1251_125112

def plate (n : ℕ) : Prop := n >= 0 ∧ n <= 999

def monday (n : ℕ) : Prop := n % 2 = 1

def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

def tuesday (n : ℕ) : Prop := sum_digits n >= 11

def wednesday (n : ℕ) : Prop := n % 3 = 0

def thursday (n : ℕ) : Prop := sum_digits n <= 14

def count_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

def friday (n : ℕ) : Prop :=
  let (d1, d2, d3) := count_digits n
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

def saturday (n : ℕ) : Prop := n < 500

def sunday (n : ℕ) : Prop := 
  let (d1, d2, d3) := count_digits n
  d1 <= 5 ∧ d2 <= 5 ∧ d3 <= 5

def can_drive (n : ℕ) (day : String) : Prop :=
  plate n ∧ 
  (day = "Monday" → monday n) ∧ 
  (day = "Tuesday" → tuesday n) ∧ 
  (day = "Wednesday" → wednesday n) ∧ 
  (day = "Thursday" → thursday n) ∧ 
  (day = "Friday" → friday n) ∧ 
  (day = "Saturday" → saturday n) ∧ 
  (day = "Sunday" → sunday n)

theorem plate_729_driving_days :
  can_drive 729 "Monday" ∧
  can_drive 729 "Tuesday" ∧
  can_drive 729 "Wednesday" ∧
  ¬ can_drive 729 "Thursday" ∧
  ¬ can_drive 729 "Friday" ∧
  ¬ can_drive 729 "Saturday" ∧
  ¬ can_drive 729 "Sunday" :=
by
  sorry

end plate_729_driving_days_l1251_125112


namespace speed_of_boat_in_still_water_l1251_125132

variable (x : ℝ)

theorem speed_of_boat_in_still_water (h : 10 = (x + 5) * 0.4) : x = 20 :=
sorry

end speed_of_boat_in_still_water_l1251_125132


namespace volume_ratio_l1251_125179

theorem volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : b^3 / a^3 = 125 / 27 :=
by
  -- Skipping the proof by adding 'sorry'
  sorry

end volume_ratio_l1251_125179


namespace next_term_geometric_sequence_l1251_125131

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l1251_125131


namespace part1_part2_l1251_125181

-- Part 1: Inequality solution
theorem part1 (x : ℝ) :
  (1 / 3 * x - (3 * x + 4) / 6 ≤ 2 / 3) → (x ≥ -8) := 
by
  intro h
  sorry

-- Part 2: System of inequalities solution
theorem part2 (x : ℝ) :
  (4 * (x + 1) ≤ 7 * x + 13) ∧ ((x + 2) / 3 - x / 2 > 1) → (-3 ≤ x ∧ x < -2) := 
by
  intro h
  sorry

end part1_part2_l1251_125181


namespace angle_bisector_slope_l1251_125134

theorem angle_bisector_slope (k : ℚ) : 
  (∀ x : ℚ, (y = 2 * x ∧ y = 4 * x) → (y = k * x)) → k = -12 / 7 :=
sorry

end angle_bisector_slope_l1251_125134


namespace smallest_spherical_triangle_angle_l1251_125109

-- Define the conditions
def is_ratio (a b c : ℕ) : Prop := a = 4 ∧ b = 5 ∧ c = 6
def sum_of_angles (α β γ : ℕ) : Prop := α + β + γ = 270

-- Define the problem statement
theorem smallest_spherical_triangle_angle 
  (a b c α β γ : ℕ)
  (h1 : is_ratio a b c)
  (h2 : sum_of_angles (a * α) (b * β) (c * γ)) :
  a * α = 72 := 
sorry

end smallest_spherical_triangle_angle_l1251_125109


namespace two_connected_iff_constructible_with_H_paths_l1251_125156

-- A graph is represented as a structure with vertices and edges
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop

-- Function to check if a graph is 2-connected
noncomputable def isTwoConnected (G : Graph) : Prop := sorry

-- Function to check if a graph can be constructed by adding H-paths
noncomputable def constructibleWithHPaths (G H : Graph) : Prop := sorry

-- Given a graph G and subgraph H, we need to prove the equivalence
theorem two_connected_iff_constructible_with_H_paths (G H : Graph) :
  (isTwoConnected G) ↔ (constructibleWithHPaths G H) := sorry

end two_connected_iff_constructible_with_H_paths_l1251_125156


namespace Kiera_envelopes_total_l1251_125172

-- Define variables for different colored envelopes
def E_b : ℕ := 120
def E_y : ℕ := E_b - 25
def E_g : ℕ := 5 * E_y
def E_r : ℕ := (E_b + E_y) / 2  -- integer division in lean automatically rounds down
def E_p : ℕ := E_r + 71
def E_total : ℕ := E_b + E_y + E_g + E_r + E_p

-- The statement to be proven
theorem Kiera_envelopes_total : E_total = 975 := by
  -- intentionally put the sorry to mark the proof as unfinished
  sorry

end Kiera_envelopes_total_l1251_125172


namespace london_to_baglmintster_distance_l1251_125128

variable (D : ℕ) -- distance from London to Baglmintster

-- Conditions
def meeting_point_condition_1 := D ≥ 40
def meeting_point_condition_2 := D ≥ 48
def initial_meeting := D - 40
def return_meeting := D - 48

theorem london_to_baglmintster_distance :
  (D - 40) + 48 = D + 8 ∧ 40 + (D - 48) = D - 8 → D = 72 :=
by
  intros h
  sorry

end london_to_baglmintster_distance_l1251_125128


namespace second_polygon_sides_l1251_125155

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l1251_125155


namespace average_children_in_families_with_children_l1251_125147

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l1251_125147


namespace eight_b_equals_neg_eight_l1251_125130

theorem eight_b_equals_neg_eight (a b : ℤ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := 
by
  sorry

end eight_b_equals_neg_eight_l1251_125130


namespace cannot_determine_position_l1251_125153

-- Define the conditions
def east_longitude_122_north_latitude_43_6 : Prop := true
def row_6_seat_3_in_cinema : Prop := true
def group_1_in_classroom : Prop := false
def island_50_nautical_miles_north_northeast_another : Prop := true

-- Define the theorem
theorem cannot_determine_position :
  ¬ ((east_longitude_122_north_latitude_43_6 = false) ∧
     (row_6_seat_3_in_cinema = false) ∧
     (island_50_nautical_miles_north_northeast_another = false) ∧
     (group_1_in_classroom = true)) :=
by
  sorry

end cannot_determine_position_l1251_125153


namespace chocolate_bars_squares_l1251_125139

theorem chocolate_bars_squares
  (gerald_bars : ℕ)
  (teacher_rate : ℕ)
  (students : ℕ)
  (squares_per_student : ℕ)
  (total_squares : ℕ)
  (total_bars : ℕ)
  (squares_per_bar : ℕ)
  (h1 : gerald_bars = 7)
  (h2 : teacher_rate = 2)
  (h3 : students = 24)
  (h4 : squares_per_student = 7)
  (h5 : total_squares = students * squares_per_student)
  (h6 : total_bars = gerald_bars + teacher_rate * gerald_bars)
  (h7 : squares_per_bar = total_squares / total_bars)
  : squares_per_bar = 8 := by 
  sorry

end chocolate_bars_squares_l1251_125139


namespace min_value_ratio_l1251_125177

noncomputable def min_ratio (a : ℝ) (h : a > 0) : ℝ :=
  let x_A := 4^(-a)
  let x_B := 4^(a)
  let x_C := 4^(- (18 / (2*a + 1)))
  let x_D := 4^((18 / (2*a + 1)))
  let m := abs (x_A - x_C)
  let n := abs (x_B - x_D)
  n / m

theorem min_value_ratio (a : ℝ) (h : a > 0) : 
  ∃ c : ℝ, c = 2^11 := sorry

end min_value_ratio_l1251_125177


namespace max_african_team_wins_max_l1251_125184

-- Assume there are n African teams and (n + 9) European teams.
-- Each pair of teams plays exactly once.
-- European teams won nine times as many matches as African teams.
-- Prove that the maximum number of matches that a single African team might have won is 11.

theorem max_african_team_wins_max (n : ℕ) (k : ℕ) (n_african_wins : ℕ) (n_european_wins : ℕ)
  (h1 : n_african_wins = (n * (n - 1)) / 2) 
  (h2 : n_european_wins = ((n + 9) * (n + 8)) / 2 + k)
  (h3 : n_european_wins = 9 * (n_african_wins + (n * (n + 9) - k))) :
  ∃ max_wins, max_wins = 11 := by
  sorry

end max_african_team_wins_max_l1251_125184


namespace evaluate_expression_l1251_125173

theorem evaluate_expression (b : ℕ) (h : b = 4) : (b ^ b - b * (b - 1) ^ b) ^ b = 21381376 := by
  sorry

end evaluate_expression_l1251_125173


namespace chess_tournament_participants_l1251_125152

/-- If each participant of a chess tournament plays exactly one game with each of the remaining participants, and 231 games are played during the tournament, then the number of participants is 22. -/
theorem chess_tournament_participants (n : ℕ) (h : (n - 1) * n / 2 = 231) : n = 22 :=
sorry

end chess_tournament_participants_l1251_125152


namespace least_product_xy_l1251_125154

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l1251_125154


namespace eval_oplus_otimes_l1251_125150

-- Define the operations ⊕ and ⊗
def my_oplus (a b : ℕ) := a + b + 1
def my_otimes (a b : ℕ) := a * b - 1

-- Statement of the proof problem
theorem eval_oplus_otimes : my_oplus (my_oplus 5 7) (my_otimes 2 4) = 21 :=
by
  sorry

end eval_oplus_otimes_l1251_125150


namespace fraction_of_number_l1251_125195

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l1251_125195


namespace total_pens_l1251_125136

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l1251_125136


namespace contrapositive_statement_l1251_125176

theorem contrapositive_statement 
  (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h3 : a + b < 0) : 
  b < 0 :=
sorry

end contrapositive_statement_l1251_125176


namespace range_of_a_for_monotonically_decreasing_function_l1251_125163

theorem range_of_a_for_monotonically_decreasing_function {a : ℝ} :
    (∀ x y : ℝ, (x > 2 → y > 2 → (ax^2 + x - 1) ≤ (a*y^2 + y - 1)) ∧
                (x ≤ 2 → y ≤ 2 → (-x + 1) ≤ (-y + 1)) ∧
                (x > 2 → y ≤ 2 → (ax^2 + x - 1) ≤ (-y + 1)) ∧
                (x ≤ 2 → y > 2 → (-x + 1) ≤ (a*y^2 + y - 1))) →
    (a < 0 ∧ - (1 / (2 * a)) ≤ 2 ∧ 4 * a + 1 ≤ -1) →
    a ≤ -1 / 2 :=
by
  intro hmonotone hconditions
  sorry

end range_of_a_for_monotonically_decreasing_function_l1251_125163


namespace lisa_needs_additional_marbles_l1251_125133

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l1251_125133


namespace smallest_x_for_multiple_l1251_125129

theorem smallest_x_for_multiple (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 640 = 2^7 * 5^1) :
  (450 * x) % 640 = 0 ↔ x = 64 :=
sorry

end smallest_x_for_multiple_l1251_125129


namespace simplify_eval_expression_l1251_125104

variables (a b : ℝ)

theorem simplify_eval_expression :
  a = Real.sqrt 3 →
  b = Real.sqrt 3 - 1 →
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 :=
by
  sorry

end simplify_eval_expression_l1251_125104


namespace volume_OABC_is_l1251_125192

noncomputable def volume_tetrahedron_ABC (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) : ℝ :=
  1 / 6 * a * b * c

theorem volume_OABC_is (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) :
  volume_tetrahedron_ABC a b c hx hy hz = (5 / 6) * Real.sqrt 30.375 :=
by
  sorry

end volume_OABC_is_l1251_125192


namespace factor_correct_l1251_125127

theorem factor_correct (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end factor_correct_l1251_125127


namespace average_visitors_per_day_l1251_125126

/-- A library has different visitor numbers depending on the day of the week.
  - On Sundays, the library has an average of 660 visitors.
  - On Mondays through Thursdays, there are 280 visitors on average.
  - Fridays and Saturdays see an increase to an average of 350 visitors.
  - This month has a special event on the third Saturday, bringing an extra 120 visitors that day.
  - The month has 30 days and begins with a Sunday.
  We want to calculate the average number of visitors per day for the entire month. -/
theorem average_visitors_per_day
  (num_days : ℕ) (starts_on_sunday : Bool)
  (sundays_visitors : ℕ) (weekdays_visitors : ℕ) (weekend_visitors : ℕ)
  (special_event_extra_visitors : ℕ) (sundays : ℕ) (mondays : ℕ)
  (tuesdays : ℕ) (wednesdays : ℕ) (thursdays : ℕ) (fridays : ℕ)
  (saturdays : ℕ) :
  num_days = 30 → starts_on_sunday = true →
  sundays_visitors = 660 → weekdays_visitors = 280 → weekend_visitors = 350 →
  special_event_extra_visitors = 120 →
  sundays = 4 → mondays = 5 →
  tuesdays = 4 → wednesdays = 4 → thursdays = 4 → fridays = 4 → saturdays = 4 →
  ((sundays * sundays_visitors +
    mondays * weekdays_visitors +
    tuesdays * weekdays_visitors +
    wednesdays * weekdays_visitors +
    thursdays * weekdays_visitors +
    fridays * weekend_visitors +
    saturdays * weekend_visitors +
    special_event_extra_visitors) / num_days = 344) :=
by
  intros
  sorry

end average_visitors_per_day_l1251_125126


namespace determine_numbers_l1251_125196

theorem determine_numbers (n : ℕ) (m : ℕ) (x y z u v : ℕ) (h₁ : 10000 <= n ∧ n < 100000)
(h₂ : n = 10000 * x + 1000 * y + 100 * z + 10 * u + v)
(h₃ : m = 1000 * x + 100 * y + 10 * u + v)
(h₄ : x ≠ 0)
(h₅ : n % m = 0) :
∃ a : ℕ, (10 <= a ∧ a <= 99 ∧ n = a * 1000) :=
sorry

end determine_numbers_l1251_125196


namespace trapezoid_perimeter_l1251_125118

noncomputable def isosceles_trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) : ℝ :=
  8 * R / (Real.sin α)

theorem trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) :
  ∃ (P : ℝ), P = isosceles_trapezoid_perimeter R α hα := by
    sorry

end trapezoid_perimeter_l1251_125118


namespace passengers_got_off_l1251_125148

theorem passengers_got_off :
  ∀ (initial_boarded new_boarded final_left got_off : ℕ),
    initial_boarded = 28 →
    new_boarded = 7 →
    final_left = 26 →
    got_off = initial_boarded + new_boarded - final_left →
    got_off = 9 :=
by
  intros initial_boarded new_boarded final_left got_off h_initial h_new h_final h_got_off
  rw [h_initial, h_new, h_final] at h_got_off
  exact h_got_off

end passengers_got_off_l1251_125148


namespace tommy_balloons_l1251_125106

/-- Tommy had some balloons. He received 34 more balloons from his mom,
gave away 15 balloons, and exchanged the remaining balloons for teddy bears
at a rate of 3 balloons per teddy bear. After these transactions, he had 30 teddy bears.
Prove that Tommy started with 71 balloons -/
theorem tommy_balloons : 
  ∃ B : ℕ, (B + 34 - 15) = 3 * 30 ∧ B = 71 := 
by
  have h : (71 + 34 - 15) = 3 * 30 := by norm_num
  exact ⟨71, h, rfl⟩

end tommy_balloons_l1251_125106


namespace cost_of_patent_is_correct_l1251_125114

-- Defining the conditions
def c_parts : ℕ := 3600
def p : ℕ := 180
def n : ℕ := 45

-- Calculation of total revenue
def total_revenue : ℕ := n * p

-- Calculation of cost of patent
def cost_of_patent (total_revenue c_parts : ℕ) : ℕ := total_revenue - c_parts

-- The theorem to be proved
theorem cost_of_patent_is_correct (R : ℕ) (H : R = total_revenue) : cost_of_patent R c_parts = 4500 :=
by
  -- this is where your proof will go
  sorry

end cost_of_patent_is_correct_l1251_125114


namespace parabola_vertex_coordinates_l1251_125178

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), y = -3 * (x + 1)^2 - 2 → (x, y) = (-1, -2) := by
  sorry

end parabola_vertex_coordinates_l1251_125178


namespace percentage_x_minus_y_l1251_125183

variable (x y : ℝ)

theorem percentage_x_minus_y (P : ℝ) :
  P / 100 * (x - y) = 20 / 100 * (x + y) ∧ y = 20 / 100 * x → P = 30 :=
by
  intros h
  sorry

end percentage_x_minus_y_l1251_125183


namespace molecular_weight_H2O_7_moles_l1251_125108

noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00
noncomputable def num_atoms_H_in_H2O : ℝ := 2
noncomputable def num_atoms_O_in_H2O : ℝ := 1
noncomputable def moles_H2O : ℝ := 7

theorem molecular_weight_H2O_7_moles :
  (num_atoms_H_in_H2O * atomic_weight_H + num_atoms_O_in_H2O * atomic_weight_O) * moles_H2O = 126.112 := by
  sorry

end molecular_weight_H2O_7_moles_l1251_125108


namespace joe_speed_l1251_125140

theorem joe_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_run : ℝ) (distance : ℝ) 
  (h1 : joe_speed = 2 * pete_speed)
  (h2 : time_run = 2 / 3)
  (h3 : distance = 16)
  (h4 : distance = 3 * pete_speed * time_run) :
  joe_speed = 16 :=
by sorry

end joe_speed_l1251_125140


namespace cost_of_paving_l1251_125101

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 1400
def expected_cost : ℝ := 28875

theorem cost_of_paving (l w r : ℝ) (h_l : l = length) (h_w : w = width) (h_r : r = rate_per_sqm) :
  (l * w * r) = expected_cost := by
  sorry

end cost_of_paving_l1251_125101


namespace probability_of_point_in_smaller_square_l1251_125105

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

end probability_of_point_in_smaller_square_l1251_125105


namespace log_product_computation_l1251_125100

theorem log_product_computation : 
  (Real.log 32 / Real.log 2) * (Real.log 27 / Real.log 3) = 15 := 
by
  -- The proof content, which will be skipped with 'sorry'.
  sorry

end log_product_computation_l1251_125100


namespace fraction_of_menu_items_i_can_eat_l1251_125115

def total_dishes (vegan_dishes non_vegan_dishes : ℕ) : ℕ := vegan_dishes + non_vegan_dishes

def vegan_dishes_without_soy (vegan_dishes vegan_with_soy : ℕ) : ℕ := vegan_dishes - vegan_with_soy

theorem fraction_of_menu_items_i_can_eat (vegan_dishes non_vegan_dishes vegan_with_soy : ℕ)
  (h_vegan_dishes : vegan_dishes = 6)
  (h_menu_total : vegan_dishes = (total_dishes vegan_dishes non_vegan_dishes) / 3)
  (h_vegan_with_soy : vegan_with_soy = 4)
  : (vegan_dishes_without_soy vegan_dishes vegan_with_soy) / (total_dishes vegan_dishes non_vegan_dishes) = 1 / 9 :=
by
  sorry

end fraction_of_menu_items_i_can_eat_l1251_125115


namespace base_subtraction_proof_l1251_125117

def convert_base8_to_base10 (n : Nat) : Nat :=
  5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1

def convert_base9_to_base10 (n : Nat) : Nat :=
  4 * 9^3 + 3 * 9^2 + 2 * 9^1 + 1

theorem base_subtraction_proof :
  convert_base8_to_base10 54321 - convert_base9_to_base10 4321 = 19559 :=
by
  sorry

end base_subtraction_proof_l1251_125117


namespace inscribed_sphere_radius_l1251_125123

theorem inscribed_sphere_radius (b d : ℝ) : 
  (b * Real.sqrt d - b = 15 * (Real.sqrt 5 - 1) / 4) → 
  b + d = 11.75 :=
by
  intro h
  sorry

end inscribed_sphere_radius_l1251_125123


namespace correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l1251_125119

-- Define the constants k and b
variables (k b : ℝ)

-- Define the function y = k * t + b
def linear_func (t : ℝ) : ℝ := k * t + b

-- Define the data points as conditions
axiom data_point1 : linear_func k b 1 = 7
axiom data_point2 : linear_func k b 2 = 12
axiom data_point3 : linear_func k b 3 = 17
axiom data_point4 : linear_func k b 4 = 22
axiom data_point5 : linear_func k b 5 = 27

-- Define the water consumption rate and total minutes in a day
def daily_water_consumption : ℝ := 1500
def minutes_in_one_day : ℝ := 1440
def days_in_month : ℝ := 30

-- The expression y = 5t + 2
theorem correct_functional_relationship : (k = 5) ∧ (b = 2) :=
by
  sorry

-- Estimated water amount at the 20th minute
theorem water_amount_20th_minute (t : ℝ) (ht : t = 20) : linear_func 5 2 t = 102 :=
by
  sorry

-- The water leaked in a month (30 days) can supply the number of days
theorem water_amount_supply_days : (linear_func 5 2 (minutes_in_one_day * days_in_month)) / daily_water_consumption = 144 :=
by
  sorry

end correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l1251_125119


namespace find_m_l1251_125121

def A (m : ℤ) : Set ℤ := {2, 5, m ^ 2 - m}
def B (m : ℤ) : Set ℤ := {2, m + 3}

theorem find_m (m : ℤ) : A m ∩ B m = B m → m = 3 := by
  sorry

end find_m_l1251_125121


namespace z_share_per_rupee_x_l1251_125158

-- Definitions according to the conditions
def x_gets (r : ℝ) : ℝ := r
def y_gets_for_x (r : ℝ) : ℝ := 0.45 * r
def y_share : ℝ := 18
def total_amount : ℝ := 78

-- Problem statement to prove z gets 0.5 rupees for each rupee x gets.
theorem z_share_per_rupee_x (r : ℝ) (hx : x_gets r = 40) (hy : y_gets_for_x r = 18) (ht : total_amount = 78) :
  (total_amount - (x_gets r + y_share)) / x_gets r = 0.5 := by
  sorry

end z_share_per_rupee_x_l1251_125158


namespace smaller_angle_between_clock_hands_3_40_pm_l1251_125193

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l1251_125193


namespace car_distance_l1251_125189

variable (T_initial : ℕ) (T_new : ℕ) (S : ℕ) (D : ℕ)

noncomputable def calculate_distance (T_initial T_new S : ℕ) : ℕ :=
  S * T_new

theorem car_distance :
  T_initial = 6 →
  T_new = (3 / 2) * T_initial →
  S = 16 →
  D = calculate_distance T_initial T_new S →
  D = 144 :=
by
  sorry

end car_distance_l1251_125189


namespace initial_hamburgers_correct_l1251_125157

-- Define the initial problem conditions
def initial_hamburgers (H : ℝ) : Prop := H + 3.0 = 12

-- State the proof problem
theorem initial_hamburgers_correct (H : ℝ) (h : initial_hamburgers H) : H = 9.0 :=
sorry

end initial_hamburgers_correct_l1251_125157


namespace find_x_in_sequence_l1251_125167

theorem find_x_in_sequence :
  ∃ x y z : Int, (z + 3 = 5) ∧ (y + z = 5) ∧ (x + y = 2) ∧ (x = -1) :=
by
  use -1, 3, 2
  sorry

end find_x_in_sequence_l1251_125167


namespace actual_speed_of_valentin_l1251_125188

theorem actual_speed_of_valentin
  (claimed_speed : ℕ := 50) -- Claimed speed in m/min
  (wrong_meter : ℕ := 60)   -- Valentin thought 1 meter = 60 cm
  (wrong_minute : ℕ := 100) -- Valentin thought 1 minute = 100 seconds
  (correct_speed : ℕ := 18) -- The actual speed in m/min
  : (claimed_speed * wrong_meter / wrong_minute) * 60 / 100 = correct_speed :=
by
  sorry

end actual_speed_of_valentin_l1251_125188


namespace time_to_pay_back_l1251_125169

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l1251_125169


namespace polygon_side_intersections_l1251_125185

theorem polygon_side_intersections :
  let m6 := 6
  let m7 := 7
  let m8 := 8
  let m9 := 9
  let pairs := [(m6, m7), (m6, m8), (m6, m9), (m7, m8), (m7, m9), (m8, m9)]
  let count_intersections (m n : ℕ) : ℕ := 2 * min m n
  let total_intersections := pairs.foldl (fun total pair => total + count_intersections pair.1 pair.2) 0
  total_intersections = 80 :=
by
  sorry

end polygon_side_intersections_l1251_125185


namespace cos_120_eq_neg_half_l1251_125144

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l1251_125144


namespace stack_map_A_front_view_l1251_125137

def column1 : List ℕ := [3, 1]
def column2 : List ℕ := [2, 2, 1]
def column3 : List ℕ := [1, 4, 2]
def column4 : List ℕ := [5]

def tallest (l : List ℕ) : ℕ :=
  l.foldl max 0

theorem stack_map_A_front_view :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 2, 4, 5] := by
  sorry

end stack_map_A_front_view_l1251_125137


namespace unit_digit_of_12_pow_100_l1251_125149

def unit_digit_pow (a: ℕ) (n: ℕ) : ℕ :=
  (a ^ n) % 10

theorem unit_digit_of_12_pow_100 : unit_digit_pow 12 100 = 6 := by
  sorry

end unit_digit_of_12_pow_100_l1251_125149


namespace jeff_total_cabinets_l1251_125145

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end jeff_total_cabinets_l1251_125145


namespace min_value_frac_ineq_l1251_125124

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : a + b = 5) : 
  (1 / (a - 1) + 9 / (b - 2)) = 8 :=
sorry

end min_value_frac_ineq_l1251_125124


namespace volume_after_increasing_edges_l1251_125151

-- Defining the initial conditions and the theorem to prove regarding the volume.
theorem volume_after_increasing_edges {a b c : ℝ} 
  (h1 : a * b * c = 8) 
  (h2 : (a + 1) * (b + 1) * (c + 1) = 27) : 
  (a + 2) * (b + 2) * (c + 2) = 64 :=
sorry

end volume_after_increasing_edges_l1251_125151


namespace A_work_days_l1251_125143

variables (r_A r_B r_C : ℝ) (h1 : r_A + r_B = (1 / 3)) (h2 : r_B + r_C = (1 / 3)) (h3 : r_A + r_C = (5 / 24))

theorem A_work_days :
  1 / r_A = 9.6 := 
sorry

end A_work_days_l1251_125143


namespace book_donation_growth_rate_l1251_125191

theorem book_donation_growth_rate (x : ℝ) : 
  400 + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 :=
sorry

end book_donation_growth_rate_l1251_125191


namespace cuboid_surface_area_correct_l1251_125120

-- Define the dimensions of the cuboid
def l : ℕ := 4
def w : ℕ := 5
def h : ℕ := 6

-- Define the function to calculate the surface area of the cuboid
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

-- The theorem stating that the surface area of the cuboid is 148 cm²
theorem cuboid_surface_area_correct : surface_area l w h = 148 := by
  sorry

end cuboid_surface_area_correct_l1251_125120


namespace triangle_height_l1251_125122

theorem triangle_height (base area height : ℝ)
    (h_base : base = 4)
    (h_area : area = 16)
    (h_area_formula : area = (base * height) / 2) :
    height = 8 :=
by
  sorry

end triangle_height_l1251_125122


namespace smallest_four_digit_divisible_by_3_and_8_l1251_125102

theorem smallest_four_digit_divisible_by_3_and_8 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 3 = 0 ∧ m % 8 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_3_and_8_l1251_125102


namespace hotel_elevator_cubic_value_l1251_125110

noncomputable def hotel_elevator_cubic : ℚ → ℚ := sorry

theorem hotel_elevator_cubic_value :
  hotel_elevator_cubic 11 = 11 ∧
  hotel_elevator_cubic 12 = 12 ∧
  hotel_elevator_cubic 13 = 14 ∧
  hotel_elevator_cubic 14 = 15 →
  hotel_elevator_cubic 15 = 13 :=
sorry

end hotel_elevator_cubic_value_l1251_125110


namespace aiyanna_cookies_l1251_125138

theorem aiyanna_cookies (a b : ℕ) (h₁ : a = 129) (h₂ : b = a + 11) : b = 140 := by
  sorry

end aiyanna_cookies_l1251_125138


namespace kids_difference_l1251_125166

def kidsPlayedOnMonday : Nat := 11
def kidsPlayedOnTuesday : Nat := 12

theorem kids_difference :
  kidsPlayedOnTuesday - kidsPlayedOnMonday = 1 := by
  sorry

end kids_difference_l1251_125166


namespace angle_Z_90_l1251_125186

-- Definitions and conditions from step a)
def Triangle (X Y Z : ℝ) : Prop :=
  X + Y + Z = 180

def in_triangle_XYZ (X Y Z : ℝ) : Prop :=
  Triangle X Y Z ∧ (X + Y = 90)

-- Proof problem from step c)
theorem angle_Z_90 (X Y Z : ℝ) (h : in_triangle_XYZ X Y Z) : Z = 90 :=
  by
  sorry

end angle_Z_90_l1251_125186
