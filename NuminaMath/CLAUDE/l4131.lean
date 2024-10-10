import Mathlib

namespace sector_area_l4131_413104

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = 72 → radius = 20 → 
  (centralAngle / 360) * Real.pi * radius^2 = 80 * Real.pi := by
  sorry

end sector_area_l4131_413104


namespace f_derivative_at_zero_l4131_413117

noncomputable def f (x : ℝ) : ℝ := Real.exp x / (x + 2)

theorem f_derivative_at_zero : 
  deriv f 0 = 1/4 := by sorry

end f_derivative_at_zero_l4131_413117


namespace largest_common_number_l4131_413181

theorem largest_common_number (n m : ℕ) : 
  67 = 1 + 6 * n ∧ 
  67 = 4 + 7 * m ∧ 
  67 ≤ 100 ∧ 
  ∀ k, (∃ p q : ℕ, k = 1 + 6 * p ∧ k = 4 + 7 * q ∧ k ≤ 100) → k ≤ 67 :=
by sorry

end largest_common_number_l4131_413181


namespace inequality_implication_l4131_413137

theorem inequality_implication (a b : ℝ) : 
  a^2 - b^2 + 2*a - 4*b - 3 ≠ 0 → a - b ≠ 1 := by
  sorry

end inequality_implication_l4131_413137


namespace system_solution_proof_l4131_413148

theorem system_solution_proof : ∃ (x y : ℝ), 
  (2 * x + 7 * y = -6) ∧ 
  (2 * x - 5 * y = 18) ∧ 
  (x = 4) ∧ 
  (y = -2) := by
  sorry

end system_solution_proof_l4131_413148


namespace product_repeating_third_twelve_l4131_413132

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of 0.333... and 12 is 4 --/
theorem product_repeating_third_twelve : repeating_third * 12 = 4 := by
  sorry

end product_repeating_third_twelve_l4131_413132


namespace prism_base_side_length_l4131_413176

/-- Given a rectangular prism with a square base, prove that with the given dimensions and properties, the side length of the base is 2 meters. -/
theorem prism_base_side_length (height : ℝ) (density : ℝ) (weight : ℝ) (volume : ℝ) (side : ℝ) :
  height = 8 →
  density = 2700 →
  weight = 86400 →
  volume = weight / density →
  volume = side^2 * height →
  side = 2 := by
  sorry


end prism_base_side_length_l4131_413176


namespace fifteen_initial_points_theorem_l4131_413184

/-- The number of points after n iterations of the marking process -/
def points_after_iteration (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * (points_after_iteration initial_points k) - 1

/-- The theorem stating that 15 initial points result in 225 points after 4 iterations -/
theorem fifteen_initial_points_theorem :
  ∃ (initial_points : ℕ), 
    initial_points > 0 ∧ 
    points_after_iteration initial_points 4 = 225 ∧ 
    initial_points = 15 := by
  sorry


end fifteen_initial_points_theorem_l4131_413184


namespace lucas_100_mod5_l4131_413112

/-- The Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℕ := lucas n % 5

/-- The cycle length of the Lucas sequence modulo 5 -/
def cycle_length : ℕ := 10

theorem lucas_100_mod5 :
  lucas_mod5 100 = 3 := by sorry

end lucas_100_mod5_l4131_413112


namespace john_can_buy_max_notebooks_l4131_413188

/-- The amount of money John has, in cents -/
def johns_money : ℕ := 4575

/-- The cost of each notebook, in cents -/
def notebook_cost : ℕ := 325

/-- The maximum number of notebooks John can buy -/
def max_notebooks : ℕ := 14

theorem john_can_buy_max_notebooks :
  (max_notebooks * notebook_cost ≤ johns_money) ∧
  ∀ n : ℕ, n > max_notebooks → n * notebook_cost > johns_money :=
by sorry

end john_can_buy_max_notebooks_l4131_413188


namespace complex_magnitude_l4131_413180

theorem complex_magnitude (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + i) / 2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l4131_413180


namespace program_arrangements_l4131_413190

/-- The number of ways to arrange n items in k positions --/
def arrangement (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to insert 3 new programs into a list of 10 existing programs --/
theorem program_arrangements : 
  arrangement 11 3 + arrangement 3 2 * arrangement 11 2 + arrangement 3 3 * arrangement 11 1 = 1716 := by
  sorry

end program_arrangements_l4131_413190


namespace first_coaster_speed_is_50_l4131_413198

/-- The speed of the first rollercoaster given the speeds of the other four and the average speed -/
def first_coaster_speed (second_speed third_speed fourth_speed fifth_speed average_speed : ℝ) : ℝ :=
  5 * average_speed - (second_speed + third_speed + fourth_speed + fifth_speed)

/-- Theorem stating that the first coaster's speed is 50 mph given the problem conditions -/
theorem first_coaster_speed_is_50 :
  first_coaster_speed 62 73 70 40 59 = 50 := by
  sorry

end first_coaster_speed_is_50_l4131_413198


namespace complex_multiplication_l4131_413139

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (-1 + i) * (2 - i) = -1 + 3*i := by
  sorry

end complex_multiplication_l4131_413139


namespace r_value_when_n_is_3_l4131_413105

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s := 2^(n+1) + 2
  let r := 3^s - 2*s + 1
  r = 387420454 := by
sorry

end r_value_when_n_is_3_l4131_413105


namespace max_book_combination_l4131_413163

theorem max_book_combination (total : ℕ) (math_books logic_books : ℕ → ℕ) : 
  total = 20 →
  (∀ k, math_books k + logic_books k = total) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → math_books k = 10 - k ∧ logic_books k = 10 + k) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → Nat.choose (math_books k) 5 * Nat.choose (logic_books k) 5 ≤ (Nat.choose 10 5)^2) :=
by sorry

end max_book_combination_l4131_413163


namespace otimes_calculation_l4131_413122

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 - a*b

-- State the theorem
theorem otimes_calculation :
  otimes (-5) (otimes 3 (-2)) = 100 := by
  sorry

end otimes_calculation_l4131_413122


namespace distance_calculation_l4131_413186

/-- The distance between Xiao Ming's home and his grandmother's house -/
def distance_to_grandma : ℝ := 36

/-- Xiao Ming's speed in km/h -/
def xiao_ming_speed : ℝ := 12

/-- Father's speed in km/h -/
def father_speed : ℝ := 36

/-- Time Xiao Ming departs before his father in hours -/
def time_before_father : ℝ := 2.5

/-- Time father arrives after Xiao Ming in hours -/
def time_after_xiao_ming : ℝ := 0.5

theorem distance_calculation :
  ∃ (t : ℝ),
    t > 0 ∧
    distance_to_grandma = father_speed * t ∧
    distance_to_grandma = xiao_ming_speed * (t + time_before_father - time_after_xiao_ming) :=
by
  sorry

#check distance_calculation

end distance_calculation_l4131_413186


namespace chord_length_is_2_sqrt_2_l4131_413166

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 6 = 0

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  ∃ (chord_length : ℝ), 
    (∀ (x y : ℝ), line_eq x y → circle_eq x y → chord_length = 2 * Real.sqrt 2) :=
sorry

end chord_length_is_2_sqrt_2_l4131_413166


namespace peach_baskets_l4131_413155

theorem peach_baskets (peaches_per_basket : ℕ) (eaten_peaches : ℕ) 
  (small_boxes : ℕ) (peaches_per_small_box : ℕ) :
  peaches_per_basket = 25 →
  eaten_peaches = 5 →
  small_boxes = 8 →
  peaches_per_small_box = 15 →
  (small_boxes * peaches_per_small_box + eaten_peaches) / peaches_per_basket = 5 :=
by
  sorry

end peach_baskets_l4131_413155


namespace sin_alpha_equals_half_l4131_413141

/-- If the terminal side of angle α passes through the point (-√3, 1), then sin α = 1/2 -/
theorem sin_alpha_equals_half (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -Real.sqrt 3 ∧ t * Real.sin α = 1) → 
  Real.sin α = 1/2 := by
sorry

end sin_alpha_equals_half_l4131_413141


namespace wire_weight_proportional_l4131_413160

/-- Given that a 25 m roll of wire weighs 5 kg, prove that a 75 m roll of wire weighs 15 kg. -/
theorem wire_weight_proportional (length_short : ℝ) (weight_short : ℝ) (length_long : ℝ) :
  length_short = 25 →
  weight_short = 5 →
  length_long = 75 →
  (length_long / length_short) * weight_short = 15 := by
  sorry

end wire_weight_proportional_l4131_413160


namespace unique_integer_solution_l4131_413123

theorem unique_integer_solution : ∃! (x y : ℤ), 10*x + 18*y = 28 ∧ 18*x + 10*y = 56 :=
by sorry

end unique_integer_solution_l4131_413123


namespace largest_prime_divisor_of_sum_of_squares_l4131_413127

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 49^2) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof would go here
  sorry

end largest_prime_divisor_of_sum_of_squares_l4131_413127


namespace simple_interest_rate_l4131_413167

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h1 : principal = 10000) 
  (h2 : time = 1) (h3 : interest = 500) : 
  (interest / (principal * time)) * 100 = 5 := by
  sorry

end simple_interest_rate_l4131_413167


namespace bus_rows_theorem_l4131_413151

/-- Represents a school bus with rows of seats split by an aisle -/
structure SchoolBus where
  total_students : ℕ
  students_per_section : ℕ
  sections_per_row : ℕ

/-- Calculates the number of rows in a school bus -/
def num_rows (bus : SchoolBus) : ℕ :=
  (bus.total_students / bus.students_per_section) / bus.sections_per_row

/-- Theorem stating that a bus with 52 students, 2 students per section, and 2 sections per row has 13 rows -/
theorem bus_rows_theorem (bus : SchoolBus) 
  (h1 : bus.total_students = 52)
  (h2 : bus.students_per_section = 2)
  (h3 : bus.sections_per_row = 2) :
  num_rows bus = 13 := by
  sorry

#eval num_rows { total_students := 52, students_per_section := 2, sections_per_row := 2 }

end bus_rows_theorem_l4131_413151


namespace no_g_sequence_to_nine_l4131_413100

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^3 + 9 else n / 2

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem no_g_sequence_to_nine :
  ∀ n : ℤ, -100 ≤ n ∧ n ≤ 100 → ¬∃ k : ℕ, iterateG n k = 9 :=
by sorry

end no_g_sequence_to_nine_l4131_413100


namespace tom_remaining_pieces_l4131_413189

/-- 
Given the initial number of boxes, the number of boxes given away, 
and the number of pieces per box, calculate the number of pieces Tom still had.
-/
def remaining_pieces (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - boxes_given_away) * pieces_per_box

theorem tom_remaining_pieces : 
  remaining_pieces 12 7 6 = 30 := by
  sorry

#eval remaining_pieces 12 7 6

end tom_remaining_pieces_l4131_413189


namespace smallest_integer_in_set_l4131_413154

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((7 * n + 21) / 7)) → (0 ≤ n) :=
sorry

end smallest_integer_in_set_l4131_413154


namespace bottle_capacity_proof_l4131_413168

theorem bottle_capacity_proof (total_milk : ℝ) (bottle1_capacity : ℝ) (bottle1_milk : ℝ) (bottle2_capacity : ℝ) :
  total_milk = 8 →
  bottle1_capacity = 8 →
  bottle1_milk = 5.333333333333333 →
  (bottle1_milk / bottle1_capacity) = ((total_milk - bottle1_milk) / bottle2_capacity) →
  bottle2_capacity = 4 := by
sorry

end bottle_capacity_proof_l4131_413168


namespace students_in_either_not_both_l4131_413196

/-- The number of students taking both geometry and statistics -/
def both_subjects : ℕ := 18

/-- The total number of students taking geometry -/
def geometry_total : ℕ := 35

/-- The number of students taking only statistics -/
def only_statistics : ℕ := 16

/-- Theorem: The number of students taking geometry or statistics but not both is 33 -/
theorem students_in_either_not_both : 
  (geometry_total - both_subjects) + only_statistics = 33 := by
  sorry

end students_in_either_not_both_l4131_413196


namespace tangency_points_on_sphere_l4131_413153

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Predicate to check if two spheres are tangent -/
def are_tangent (s1 s2 : Sphere) : Prop := sorry

/-- Function to get the tangency point of two spheres -/
def tangency_point (s1 s2 : Sphere) : Point := sorry

/-- Predicate to check if a point lies on a sphere -/
def point_on_sphere (p : Point) (s : Sphere) : Prop := sorry

theorem tangency_points_on_sphere 
  (s1 s2 s3 s4 : Sphere) 
  (h1 : are_tangent s1 s2) (h2 : are_tangent s1 s3) (h3 : are_tangent s1 s4)
  (h4 : are_tangent s2 s3) (h5 : are_tangent s2 s4) (h6 : are_tangent s3 s4) :
  ∃ (s : Sphere), 
    point_on_sphere (tangency_point s1 s2) s ∧
    point_on_sphere (tangency_point s1 s3) s ∧
    point_on_sphere (tangency_point s1 s4) s ∧
    point_on_sphere (tangency_point s2 s3) s ∧
    point_on_sphere (tangency_point s2 s4) s ∧
    point_on_sphere (tangency_point s3 s4) s :=
  sorry

end tangency_points_on_sphere_l4131_413153


namespace smallest_solution_congruence_l4131_413185

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (52 * x + 14) % 24 = 6 ∧
  ∀ (y : ℕ), y > 0 ∧ (52 * y + 14) % 24 = 6 → x ≤ y :=
by sorry

end smallest_solution_congruence_l4131_413185


namespace icosahedron_edge_probability_l4131_413110

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (vertex_count : vertices.card = 20)
  (edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (I : Icosahedron) : ℚ :=
  (I.edges.card : ℚ) / (I.vertices.card.choose 2 : ℚ)

/-- The main theorem stating the probability is 10/19 -/
theorem icosahedron_edge_probability (I : Icosahedron) :
  edge_probability I = 10 / 19 := by
  sorry

end icosahedron_edge_probability_l4131_413110


namespace expression_equality_l4131_413174

theorem expression_equality (a b c n : ℝ) 
  (h1 : a + b = c * n) 
  (h2 : b + c = a * n) 
  (h3 : a + c = b * n) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end expression_equality_l4131_413174


namespace fast_food_theorem_l4131_413144

/-- A fast food composition -/
structure FastFood where
  total_mass : ℝ
  fat_percentage : ℝ
  protein_mass : ℝ → ℝ
  mineral_mass : ℝ → ℝ
  carb_mass : ℝ → ℝ

/-- Conditions for the fast food -/
def fast_food_conditions (ff : FastFood) : Prop :=
  ff.total_mass = 500 ∧
  ff.fat_percentage = 0.05 ∧
  (∀ x, ff.protein_mass x = 4 * ff.mineral_mass x) ∧
  (∀ x, (ff.protein_mass x + ff.carb_mass x) / ff.total_mass ≤ 0.85)

/-- Theorem about the mass of fat and maximum carbohydrates in the fast food -/
theorem fast_food_theorem (ff : FastFood) (h : fast_food_conditions ff) :
  ff.fat_percentage * ff.total_mass = 25 ∧
  ∃ x, ff.carb_mass x = 225 ∧ 
    ∀ y, ff.carb_mass y ≤ ff.carb_mass x :=
by sorry

end fast_food_theorem_l4131_413144


namespace sequence_properties_l4131_413169

def sequence_a (n : ℕ) : ℝ := 6 * 2^(n-1) - 3

def sum_S (n : ℕ) : ℝ := 6 * 2^n - 3 * n - 6

theorem sequence_properties :
  let a := sequence_a
  let S := sum_S
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n - 3 * n) →
  (a 1 = 3 ∧
   ∀ n : ℕ, a (n + 1) = 2 * a n + 3 ∧
   ∀ n : ℕ, n ≥ 1 → a n = 6 * 2^(n-1) - 3 ∧
   ∀ n : ℕ, n ≥ 1 → S n = 6 * 2^n - 3 * n - 6) :=
by
  sorry

end sequence_properties_l4131_413169


namespace point_not_on_graph_l4131_413147

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

theorem point_not_on_graph :
  ¬(inverse_proportion 1 5) ∧ 
  (inverse_proportion (-2) (-3)) ∧ 
  (inverse_proportion (-3) (-2)) ∧ 
  (inverse_proportion 4 1.5) :=
by sorry

end point_not_on_graph_l4131_413147


namespace area_ratio_incenter_centroids_l4131_413158

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_incenter_centroids 
  (ABC : Triangle) 
  (P : ℝ × ℝ) 
  (G₁ G₂ G₃ : ℝ × ℝ) :
  P = incenter ABC →
  G₁ = centroid (Triangle.mk P ABC.B ABC.C) →
  G₂ = centroid (Triangle.mk ABC.A P ABC.C) →
  G₃ = centroid (Triangle.mk ABC.A ABC.B P) →
  area (Triangle.mk G₁ G₂ G₃) = (1 / 9) * area ABC :=
by sorry

end area_ratio_incenter_centroids_l4131_413158


namespace volume_ratio_cubes_l4131_413140

/-- The ratio of the volume of a cube with edge length 4 inches to the volume of a cube with edge length 2 feet -/
theorem volume_ratio_cubes : 
  let small_edge_inches : ℚ := 4
  let large_edge_feet : ℚ := 2
  let inches_per_foot : ℚ := 12
  let small_edge_feet : ℚ := small_edge_inches / inches_per_foot
  let volume_ratio : ℚ := (small_edge_feet / large_edge_feet) ^ 3
  volume_ratio = 1 / 216 := by sorry

end volume_ratio_cubes_l4131_413140


namespace high_school_twelve_games_l4131_413126

/-- The number of teams in the "High School Twelve" basketball league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team in the league -/
def games_per_pair : ℕ := 3

/-- The number of games each team plays against non-league teams -/
def non_league_games_per_team : ℕ := 6

/-- The total number of games in a season for the "High School Twelve" basketball league -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_league_games_per_team

theorem high_school_twelve_games :
  total_games = 270 := by sorry

end high_school_twelve_games_l4131_413126


namespace arithmetic_geometric_mean_inequality_l4131_413124

theorem arithmetic_geometric_mean_inequality {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l4131_413124


namespace comparison_theorem_l4131_413182

theorem comparison_theorem :
  (∀ x : ℝ, x^2 - x > x - 2) ∧
  (∀ a : ℝ, (a + 3) * (a - 5) < (a + 2) * (a - 4)) := by
sorry

end comparison_theorem_l4131_413182


namespace quiz_statistics_l4131_413194

def scores : List ℕ := [7, 5, 6, 8, 7, 9]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem quiz_statistics :
  mean scores = 7 ∧ mode scores = 7 := by
  sorry

end quiz_statistics_l4131_413194


namespace students_per_fourth_grade_class_l4131_413106

/-- Proves that the number of students in each fourth-grade class is 30 --/
theorem students_per_fourth_grade_class
  (total_cupcakes : ℕ)
  (pe_class_students : ℕ)
  (fourth_grade_classes : ℕ)
  (h1 : total_cupcakes = 140)
  (h2 : pe_class_students = 50)
  (h3 : fourth_grade_classes = 3)
  : (total_cupcakes - pe_class_students) / fourth_grade_classes = 30 := by
  sorry

#check students_per_fourth_grade_class

end students_per_fourth_grade_class_l4131_413106


namespace chris_candy_distribution_chris_total_candy_l4131_413164

theorem chris_candy_distribution (first_group : Nat) (first_amount : Nat) 
  (second_group : Nat) (remaining_amount : Nat) : Nat :=
  let total_first := first_group * first_amount
  let total_second := second_group * (2 * first_amount)
  total_first + total_second + remaining_amount

theorem chris_total_candy : chris_candy_distribution 10 12 7 50 = 338 := by
  sorry

end chris_candy_distribution_chris_total_candy_l4131_413164


namespace prob_different_fruits_l4131_413116

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of Joe eating the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

/-- The probability of Joe eating at least two different kinds of fruit in a day -/
theorem prob_different_fruits : (1 : ℚ) - prob_same_fruit = 63 / 64 := by
  sorry

end prob_different_fruits_l4131_413116


namespace units_produced_today_l4131_413178

/-- Calculates the number of units produced today given previous production data -/
theorem units_produced_today (n : ℕ) (prev_avg : ℝ) (new_avg : ℝ) 
  (h1 : n = 4)
  (h2 : prev_avg = 50)
  (h3 : new_avg = 58) : 
  (n + 1 : ℝ) * new_avg - n * prev_avg = 90 := by
  sorry

end units_produced_today_l4131_413178


namespace sqrt_pi_squared_minus_6pi_plus_9_l4131_413113

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end sqrt_pi_squared_minus_6pi_plus_9_l4131_413113


namespace fifth_minus_fourth_rectangles_l4131_413119

def rectangle_tiles (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem fifth_minus_fourth_rectangles : rectangle_tiles 5 - rectangle_tiles 4 = 32 := by
  sorry

end fifth_minus_fourth_rectangles_l4131_413119


namespace three_number_problem_l4131_413121

theorem three_number_problem :
  ∃ (x y z : ℝ),
    x = 45 ∧ y = 37.5 ∧ z = 22.5 ∧
    x - y = (1/3) * z ∧
    y - z = (1/3) * x ∧
    z - 10 = (1/3) * y :=
by
  sorry

end three_number_problem_l4131_413121


namespace choir_size_proof_l4131_413159

theorem choir_size_proof : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 ∧ 
  n % 10 = 0 ∧ 
  n % 11 = 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) ∧
  n = 1080 :=
by sorry

end choir_size_proof_l4131_413159


namespace intersection_of_M_and_N_l4131_413161

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l4131_413161


namespace smith_bought_six_boxes_l4131_413138

/-- Calculates the number of new boxes of markers bought by Mr. Smith -/
def new_boxes_bought (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) : ℕ :=
  (final_markers - initial_markers) / markers_per_box

/-- Proves that Mr. Smith bought 6 new boxes of markers -/
theorem smith_bought_six_boxes :
  new_boxes_bought 32 9 86 = 6 := by
  sorry

#eval new_boxes_bought 32 9 86

end smith_bought_six_boxes_l4131_413138


namespace max_value_on_circle_l4131_413173

theorem max_value_on_circle (x y : ℝ) : 
  (x - 3)^2 + (y - 4)^2 = 9 → 
  ∃ (z : ℝ), z = 3*x + 4*y ∧ z ≤ 40 ∧ ∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 4)^2 = 9 ∧ 3*x₀ + 4*y₀ = 40 :=
by sorry

end max_value_on_circle_l4131_413173


namespace parallelogram_ABCD_area_l4131_413133

-- Define the parallelogram vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (5, 1)
def C : ℝ × ℝ := (7, 4)
def D : ℝ × ℝ := (3, 4)

-- Define a function to calculate the area of a parallelogram given two vectors
def parallelogramArea (v1 v2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  abs (x1 * y2 - x2 * y1)

-- Theorem statement
theorem parallelogram_ABCD_area :
  parallelogramArea (B.1 - A.1, B.2 - A.2) (D.1 - A.1, D.2 - A.2) = 12 := by
  sorry

end parallelogram_ABCD_area_l4131_413133


namespace product_equals_243_l4131_413171

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_equals_243_l4131_413171


namespace lcm_hcf_relation_l4131_413149

theorem lcm_hcf_relation (x y : ℕ+) (h_lcm : Nat.lcm x y = 1637970) (h_hcf : Nat.gcd x y = 210) (h_x : x = 10780) : y = 31910 := by
  sorry

end lcm_hcf_relation_l4131_413149


namespace david_average_speed_l4131_413199

def distance : ℚ := 49 / 3  -- 16 1/3 miles as a fraction

def time : ℚ := 7 / 3  -- 2 hours and 20 minutes as a fraction of hours

def average_speed (d t : ℚ) : ℚ := d / t

theorem david_average_speed :
  average_speed distance time = 7 := by sorry

end david_average_speed_l4131_413199


namespace food_cost_theorem_l4131_413114

def sandwich_cost : ℝ := 4

def juice_cost (sandwich_cost : ℝ) : ℝ := 2 * sandwich_cost

def milk_cost (sandwich_cost juice_cost : ℝ) : ℝ :=
  0.75 * (sandwich_cost + juice_cost)

def total_cost (sandwich_cost juice_cost milk_cost : ℝ) : ℝ :=
  sandwich_cost + juice_cost + milk_cost

theorem food_cost_theorem :
  total_cost sandwich_cost (juice_cost sandwich_cost) (milk_cost sandwich_cost (juice_cost sandwich_cost)) = 21 := by
  sorry

end food_cost_theorem_l4131_413114


namespace part1_part2_part3_l4131_413142

-- Define the variables and constants
variable (x y : ℝ)  -- x: quantity of vegetable A, y: quantity of vegetable B
def total_weight : ℝ := 40
def total_cost : ℝ := 180
def wholesale_price_A : ℝ := 4.8
def wholesale_price_B : ℝ := 4
def retail_price_A : ℝ := 7.2
def retail_price_B : ℝ := 5.6
def new_total_weight : ℝ := 80
def min_profit : ℝ := 176

-- Part 1
theorem part1 : 
  x + y = total_weight ∧ 
  wholesale_price_A * x + wholesale_price_B * y = total_cost → 
  x = 25 ∧ y = 15 := by sorry

-- Part 2
def m (n : ℝ) : ℝ := wholesale_price_A * n + wholesale_price_B * (new_total_weight - n)

theorem part2 : m n = 0.8 * n + 320 := by sorry

-- Part 3
def profit (n : ℝ) : ℝ := (retail_price_A - wholesale_price_A) * n + 
                           (retail_price_B - wholesale_price_B) * (new_total_weight - n)

theorem part3 : 
  ∀ n : ℝ, profit n ≥ min_profit → n ≥ 60 := by sorry

end part1_part2_part3_l4131_413142


namespace triangle_special_area_implies_angle_l4131_413109

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S of the triangle is (√3/4)(a² + b² - c²), then angle C measures π/3 --/
theorem triangle_special_area_implies_angle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) / 2 = 
            (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
sorry

end triangle_special_area_implies_angle_l4131_413109


namespace largest_value_proof_l4131_413101

theorem largest_value_proof (a b c d e : ℚ) :
  a = 0.9387 →
  b = 0.9381 →
  c = 9385 / 10000 →
  d = 0.9379 →
  e = 0.9389 →
  max a (max b (max c (max d e))) = e :=
by sorry

end largest_value_proof_l4131_413101


namespace fish_for_white_duck_l4131_413111

/-- The number of fish for each white duck -/
def fish_per_white_duck : ℕ := sorry

/-- The number of fish for each black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish for each multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

theorem fish_for_white_duck :
  fish_per_white_duck * white_ducks +
  fish_per_black_duck * black_ducks +
  fish_per_multicolor_duck * multicolor_ducks = total_fish ∧
  fish_per_white_duck = 5 := by sorry

end fish_for_white_duck_l4131_413111


namespace sector_central_angle_l4131_413128

/-- Given a circular sector with perimeter 8 and area 3, 
    its central angle is either 6 or 2/3 radians. -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 8) →  -- perimeter condition
  (1 / 2 * l * r = 3) →  -- area condition
  (l / r = 6 ∨ l / r = 2 / 3) := by
sorry

end sector_central_angle_l4131_413128


namespace number_satisfying_proportion_l4131_413115

theorem number_satisfying_proportion : 
  let x : ℚ := 3
  (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
sorry

end number_satisfying_proportion_l4131_413115


namespace simplify_expression_l4131_413192

theorem simplify_expression (a : ℝ) :
  (((a ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((a ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = a ^ 3 :=
by sorry

end simplify_expression_l4131_413192


namespace number_of_factors_of_60_l4131_413107

theorem number_of_factors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end number_of_factors_of_60_l4131_413107


namespace correct_calculation_l4131_413165

theorem correct_calculation (x y : ℝ) : 2 * x * y^2 - x * y^2 = x * y^2 := by
  sorry

end correct_calculation_l4131_413165


namespace axis_of_symmetry_shifted_l4131_413183

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the axis of symmetry for a function
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry_shifted (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 2)) (-2) := by sorry

end axis_of_symmetry_shifted_l4131_413183


namespace binomial_coefficient_20_8_l4131_413150

theorem binomial_coefficient_20_8 :
  let n : ℕ := 20
  let k : ℕ := 8
  let binomial := Nat.choose
  binomial 18 5 = 8568 →
  binomial 18 7 = 31824 →
  binomial n k = 83656 := by
sorry

end binomial_coefficient_20_8_l4131_413150


namespace jelly_bean_multiple_l4131_413108

/-- The number of vanilla jelly beans -/
def vanilla_beans : ℕ := 120

/-- The total number of jelly beans -/
def total_beans : ℕ := 770

/-- The number of grape jelly beans as a function of the multiple -/
def grape_beans (x : ℕ) : ℕ := 50 + x * vanilla_beans

/-- The theorem stating that the multiple of vanilla jelly beans taken as grape jelly beans is 5 -/
theorem jelly_bean_multiple :
  ∃ x : ℕ, x = 5 ∧ vanilla_beans + grape_beans x = total_beans :=
sorry

end jelly_bean_multiple_l4131_413108


namespace factorial_products_perfect_square_l4131_413177

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem factorial_products_perfect_square : 
  is_perfect_square (factorial 99 * factorial 100) ∧
  ¬is_perfect_square (factorial 97 * factorial 98) ∧
  ¬is_perfect_square (factorial 97 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 100) :=
by sorry

end factorial_products_perfect_square_l4131_413177


namespace right_triangle_proof_l4131_413157

-- Define a structure for a triangle with three angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define the theorem
theorem right_triangle_proof (t : Triangle) (h : t.γ = t.α + t.β) : 
  t.α = 90 ∨ t.β = 90 ∨ t.γ = 90 := by
  sorry


end right_triangle_proof_l4131_413157


namespace traffic_police_distribution_l4131_413187

def officers : ℕ := 5
def specific_officers : ℕ := 2
def intersections : ℕ := 3

theorem traffic_police_distribution :
  (Nat.choose (officers - specific_officers + 1) (intersections - 1)) *
  (Nat.factorial intersections) = 36 := by sorry

end traffic_police_distribution_l4131_413187


namespace no_function_satisfies_conditions_l4131_413118

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end no_function_satisfies_conditions_l4131_413118


namespace expression_evaluation_l4131_413162

theorem expression_evaluation (x y : ℚ) (hx : x = 1) (hy : y = -3) :
  ((x - 2*y)^2 + (3*x - y)*(3*x + y) - 3*y^2) / (-2*x) = -11 := by
  sorry

end expression_evaluation_l4131_413162


namespace mathematics_letter_probability_l4131_413145

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  (unique_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 :=
by sorry

end mathematics_letter_probability_l4131_413145


namespace messages_sent_theorem_l4131_413131

/-- Calculates the total number of messages sent over three days given the conditions -/
def totalMessages (luciaFirstDay : ℕ) (alinaDifference : ℕ) : ℕ :=
  let alinaFirstDay := luciaFirstDay - alinaDifference
  let firstDayTotal := luciaFirstDay + alinaFirstDay
  let luciaSecondDay := luciaFirstDay / 3
  let alinaSecondDay := alinaFirstDay * 2
  let secondDayTotal := luciaSecondDay + alinaSecondDay
  firstDayTotal + secondDayTotal + firstDayTotal

theorem messages_sent_theorem :
  totalMessages 120 20 = 680 := by
  sorry

#eval totalMessages 120 20

end messages_sent_theorem_l4131_413131


namespace circle_equation_tangent_line_l4131_413136

/-- The equation of a circle with center (3, -1) that is tangent to the line 3x + 4y = 0 is (x-3)² + (y+1)² = 1 -/
theorem circle_equation_tangent_line (x y : ℝ) : 
  let center : ℝ × ℝ := (3, -1)
  let line (x y : ℝ) := 3 * x + 4 * y = 0
  let circle_eq (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = 1
  let is_tangent (circle : (ℝ → ℝ → Prop) ) (line : ℝ → ℝ → Prop) := 
    ∃ (x y : ℝ), circle x y ∧ line x y ∧ 
    ∀ (x' y' : ℝ), line x' y' → (x' = x ∧ y' = y) ∨ ¬(circle x' y')
  is_tangent circle_eq line → circle_eq x y := by
sorry


end circle_equation_tangent_line_l4131_413136


namespace candidate_d_votes_l4131_413191

theorem candidate_d_votes (total_votes : ℕ) (invalid_percentage : ℚ)
  (candidate_a_percentage : ℚ) (candidate_b_percentage : ℚ) (candidate_c_percentage : ℚ)
  (h1 : total_votes = 10000)
  (h2 : invalid_percentage = 1/4)
  (h3 : candidate_a_percentage = 2/5)
  (h4 : candidate_b_percentage = 3/10)
  (h5 : candidate_c_percentage = 1/5) :
  ↑total_votes * (1 - invalid_percentage) * (1 - (candidate_a_percentage + candidate_b_percentage + candidate_c_percentage)) = 750 := by
  sorry

#check candidate_d_votes

end candidate_d_votes_l4131_413191


namespace smallest_product_of_given_numbers_l4131_413152

theorem smallest_product_of_given_numbers : 
  let numbers : List ℕ := [10, 11, 12, 13, 14]
  let smallest := numbers.minimum?
  let next_smallest := numbers.filter (· ≠ smallest.getD 0) |>.minimum?
  smallest.isSome ∧ next_smallest.isSome → 
  smallest.getD 0 * next_smallest.getD 0 = 110 := by
sorry

end smallest_product_of_given_numbers_l4131_413152


namespace sqrt_equation_solution_l4131_413134

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →  -- This condition is necessary to ensure the square root is defined
  (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 18 / 7 := by
sorry

end sqrt_equation_solution_l4131_413134


namespace square_difference_factorization_l4131_413130

theorem square_difference_factorization (x y : ℝ) : 
  49 * x^2 - 36 * y^2 = (-6*y + 7*x) * (6*y + 7*x) := by
  sorry

end square_difference_factorization_l4131_413130


namespace largest_prime_sum_under_30_l4131_413193

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem largest_prime_sum_under_30 :
  is_prime 19 ∧
  19 < 30 ∧
  is_sum_of_two_primes 19 ∧
  ∀ n : ℕ, is_prime n → n < 30 → is_sum_of_two_primes n → n ≤ 19 :=
sorry

end largest_prime_sum_under_30_l4131_413193


namespace first_caterer_cheaper_at_17_l4131_413143

/-- The least number of people for which the first caterer is cheaper -/
def least_people_first_caterer_cheaper : ℕ := 17

/-- Cost function for the first caterer -/
def cost_first_caterer (people : ℕ) : ℚ := 200 + 18 * people

/-- Cost function for the second caterer -/
def cost_second_caterer (people : ℕ) : ℚ := 250 + 15 * people

/-- Theorem stating that 17 is the least number of people for which the first caterer is cheaper -/
theorem first_caterer_cheaper_at_17 :
  (∀ n : ℕ, n < least_people_first_caterer_cheaper →
    cost_first_caterer n ≥ cost_second_caterer n) ∧
  cost_first_caterer least_people_first_caterer_cheaper < cost_second_caterer least_people_first_caterer_cheaper :=
by sorry

end first_caterer_cheaper_at_17_l4131_413143


namespace factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l4131_413172

theorem factorial_difference_is_cubic_polynomial (n : ℕ) (h : n ≥ 9) :
  (((n + 3).factorial - (n + 2).factorial) / n.factorial : ℚ) = (n + 2)^2 * (n + 1) :=
by sorry

theorem cubic_polynomial_form (n : ℕ) (h : n ≥ 9) :
  ∃ (a b c d : ℚ), (n + 2)^2 * (n + 1) = a * n^3 + b * n^2 + c * n + d :=
by sorry

end factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l4131_413172


namespace squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l4131_413156

-- Define a type for the different problems
inductive Problem
  | oppositeNumber
  | squarePerimeter
  | maxOfThree
  | binaryToDecimal

-- Function to determine if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.oppositeNumber => False
  | Problem.squarePerimeter => False
  | Problem.maxOfThree => True
  | Problem.binaryToDecimal => True

-- Theorem stating that the square perimeter problem doesn't require conditional statements
theorem squarePerimeter_doesnt_require_conditional_statements :
  ¬(requiresConditionalStatements Problem.squarePerimeter) :=
by
  sorry

-- Theorem stating that the square perimeter problem is the only one among the four that doesn't require conditional statements
theorem only_squarePerimeter_doesnt_require_conditional_statements :
  ∀ (p : Problem), ¬(requiresConditionalStatements p) → p = Problem.squarePerimeter :=
by
  sorry

end squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l4131_413156


namespace scaling_circle_not_hyperbola_l4131_413175

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define a scaling transformation
def ScalingTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a hyperbola
def Hyperbola := Set (ℝ × ℝ)

-- Theorem statement
theorem scaling_circle_not_hyperbola (c : Circle) (s : ScalingTransformation) :
  ∀ h : Hyperbola, (s '' c) ≠ h :=
sorry

end scaling_circle_not_hyperbola_l4131_413175


namespace q_expression_l4131_413179

/-- Given a function q(x) satisfying the equation
    q(x) + (x^6 + 4x^4 + 5x^3 + 12x) = (8x^4 + 26x^3 + 15x^2 + 26x + 3),
    prove that q(x) = -x^6 + 4x^4 + 21x^3 + 15x^2 + 14x + 3 -/
theorem q_expression (q : ℝ → ℝ) 
    (h : ∀ x, q x + (x^6 + 4*x^4 + 5*x^3 + 12*x) = 8*x^4 + 26*x^3 + 15*x^2 + 26*x + 3) :
  ∀ x, q x = -x^6 + 4*x^4 + 21*x^3 + 15*x^2 + 14*x + 3 := by
  sorry

end q_expression_l4131_413179


namespace square_perimeter_problem_l4131_413146

theorem square_perimeter_problem :
  ∀ (a b c : ℝ),
  (4 * a = 16) →  -- Perimeter of square A is 16
  (4 * b = 32) →  -- Perimeter of square B is 32
  (c = 4 * (b - a)) →  -- Side length of C is 4 times the difference of A and B's side lengths
  (4 * c = 64) :=  -- Perimeter of square C is 64
by
  sorry

end square_perimeter_problem_l4131_413146


namespace set_operations_l4131_413195

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by sorry

end set_operations_l4131_413195


namespace winnie_the_pooh_fall_damage_ratio_l4131_413135

/-- The ratio of damages in Winnie-the-Pooh's fall -/
theorem winnie_the_pooh_fall_damage_ratio 
  (g M τ H : ℝ) 
  (n k : ℝ) 
  (h : ℝ := H / n) 
  (V_I : ℝ := Real.sqrt (2 * g * H)) 
  (V_1 : ℝ := Real.sqrt (2 * g * h)) 
  (V_1_prime : ℝ := (1 / k) * Real.sqrt (2 * g * h)) 
  (V_II : ℝ := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))) 
  (I_I : ℝ := M * V_I * τ) 
  (I_II : ℝ := M * τ * ((V_1 - V_1_prime) + V_II)) 
  (hg : g > 0) 
  (hM : M > 0) 
  (hτ : τ > 0) 
  (hH : H > 0) 
  (hn : n > 0) 
  (hk : k > 0) : 
  I_II / I_I = 5 / 4 := by
  sorry


end winnie_the_pooh_fall_damage_ratio_l4131_413135


namespace simplify_expression_l4131_413129

theorem simplify_expression (x : ℝ) : ((-3 * x)^2) * (2 * x) = 18 * x^3 := by
  sorry

end simplify_expression_l4131_413129


namespace find_divisor_l4131_413120

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 166) 
  (h2 : quotient = 8) 
  (h3 : remainder = 6) 
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 20 := by
sorry

end find_divisor_l4131_413120


namespace motorcycle_friction_speed_relation_l4131_413125

/-- Proves that the minimum friction coefficient for a motorcycle riding on vertical walls
    is inversely proportional to the square of its speed. -/
theorem motorcycle_friction_speed_relation 
  (m : ℝ) -- mass of the motorcycle
  (g : ℝ) -- acceleration due to gravity
  (r : ℝ) -- radius of the circular room
  (s : ℝ) -- speed of the motorcycle
  (μ : ℝ → ℝ) -- friction coefficient as a function of speed
  (h_positive : m > 0 ∧ g > 0 ∧ r > 0 ∧ s > 0) -- positivity conditions
  (h_equilibrium : ∀ s, μ s * (m * s^2 / r) = m * g) -- equilibrium condition
  : ∃ (k : ℝ), ∀ s, μ s = k / s^2 :=
sorry

end motorcycle_friction_speed_relation_l4131_413125


namespace negative_integer_equation_solution_l4131_413197

theorem negative_integer_equation_solution :
  ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 :=
by
  -- Proof goes here
  sorry

end negative_integer_equation_solution_l4131_413197


namespace drama_club_neither_math_nor_physics_l4131_413102

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : math = 36) 
  (h3 : physics = 27) 
  (h4 : both = 20) : 
  total - (math + physics - both) = 17 := by
  sorry

end drama_club_neither_math_nor_physics_l4131_413102


namespace sum_of_lengths_l4131_413103

-- Define the conversion factors
def meters_to_cm : ℝ := 100
def meters_to_mm : ℝ := 1000

-- Define the values in their original units
def length_m : ℝ := 2
def length_cm : ℝ := 3
def length_mm : ℝ := 5

-- State the theorem
theorem sum_of_lengths :
  length_m + length_cm / meters_to_cm + length_mm / meters_to_mm = 2.035 := by
  sorry

end sum_of_lengths_l4131_413103


namespace girls_in_class_l4131_413170

/-- Given a class with a 3:4 ratio of girls to boys and 35 total students,
    prove that the number of girls is 15. -/
theorem girls_in_class (g b : ℕ) : 
  g + b = 35 →  -- Total number of students
  4 * g = 3 * b →  -- Ratio of girls to boys is 3:4
  g = 15 := by
sorry

end girls_in_class_l4131_413170
